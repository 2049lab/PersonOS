"""A durable, per-session ordered message queue — reliable, deduplicating, and
shared across replicas. This is the foundation on which ingest is consumed in
order.

Background: in a wearable setting, conversation is short and interruptible, so
messages for one session arrive frequently and concurrently. The old path, a task
submission plus a spin lock, gave mutual exclusion but no ordering: messages in
one session could be consumed out of order, and tasks that failed to win the lock
spun until they filled the thread pool. This pushes both ordering and reliability
down into one durable queue:

- **one queue per session**: a message is LPUSHed onto mq:{user}:{session},
  entering on the left, and RPOPLPUSH takes from the right, which makes it FIFO
  with the oldest first.
- **reliable dequeue**: taking a message moves it onto a processing list
  (mq:{u}:{s}:proc), and it is only LREMed — acked — once consumption succeeds.
  If a worker dies the message is still in proc, and the next recover() replays
  it. Consumption may be delayed, but nothing is lost.
- **a noticeboard**: a session with messages waiting is registered in the SET
  mq:active, which is how the dispatcher discovers who needs work. It is removed
  only once the queue drains, and after removing we re-check, to close the orphan
  race where a new message arrives at the exact moment we deregister.
- **no duplicates**: every message carries a msg_id (a uuid) and a seq (a
  monotonic INCR counter within the session). The consumer stores a cursor, the
  highest seq it has applied, and a message whose seq is at or below the cursor is
  a redelivery and gets acked and skipped. So re-reading the same message during
  crash recovery cannot record it twice.

Constraints: the shared cluster must not accumulate permanent keys, so every key
carries a TTL of 24h, matching seg_store. An active session renews it, and a
session silent for over a day is discarded whole, which is consistent with letting
a trailing segment expire. Everything runs on single-command paths with no
pipelines, because the Redis proxy can misframe pipelined responses — see the
seg_store module docstring. The one exception is the enqueue lock's release, a
small token-checked Lua EVAL shared with session_lock (compare-and-delete
cannot be done in a single plain command).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..config import get_config
from .redis_client import _esc, key as _key

# Releasing the enqueue lock: a single EVAL comparing the token, the same pattern
# session_lock uses, so a holder only ever deletes its own lock.
_ENQ_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def _env() -> str:
    return get_config().env


def _sess_tag(user_id: str, session_id: str) -> str:
    """The session hash tag.

    Redis Cluster assigns a slot from the content inside {} only, so tagging puts
    a session's main, proc, seq and cursor keys in the same slot, which is what
    makes multi-key commands like RPOPLPUSH legal — without it they raise
    ClusterCrossSlotError. The inner parts are escaped so two sessions cannot
    produce the same tag.
    """
    return "{" + _esc(user_id) + ":" + _esc(session_id) + "}"

# Sliding TTL on the queue keys, on the same terms as seg_store: activity in the
# session renews it, and more than a day of silence discards the whole thing.
QUEUE_TTL_S = 24 * 3600


class EnqueueBusy(RuntimeError):
    """The enqueue lock could not be taken within the deadline, meaning severe
    contention enqueueing into one session. The caller should retry shortly.

    Losing a message is far worse than making the caller retry, so we refuse on
    timeout and never force our way in alongside a live lock holder, which would
    corrupt the ordering.
    """


@dataclass
class Envelope:
    """One message envelope in the queue: the dedup and ordering metadata, the
    business payload, and the raw string that ack LREMs by.
    """
    msg_id: str
    seq: int
    kind: str                      # "ingest" | "session_end"; the wrap-up task queues too, and is handled in order
    payload: dict = field(default_factory=dict)
    raw: str = ""                  # the exact JSON string as enqueued; ack removes by it, and no two envelopes share one

    @classmethod
    def from_raw(cls, raw: str) -> "Envelope":
        d = json.loads(raw)
        return cls(msg_id=d["msg_id"], seq=int(d["seq"]), kind=d.get("kind", "ingest"),
                   payload=d.get("payload") or {}, raw=raw)

    @staticmethod
    def make(msg_id: str, seq: int, kind: str, payload: dict) -> str:
        """Build the JSON string to enqueue.

        The field order is fixed, so one message's raw string is stable and can be
        used for LREM.
        """
        return json.dumps({"msg_id": msg_id, "seq": seq, "kind": kind, "payload": payload},
                          ensure_ascii=False, sort_keys=True)


class MsgQueue(Protocol):
    def enqueue(self, user_id: str, session_id: str, payload: dict,
                *, kind: str = "ingest") -> tuple[str, int]: ...
    def active_sessions(self, limit: int = 256) -> list[tuple[str, str]]: ...
    def recover(self, user_id: str, session_id: str) -> list[Envelope]: ...
    def reserve(self, user_id: str, session_id: str) -> Optional[Envelope]: ...
    def ack(self, user_id: str, session_id: str, env: Envelope) -> None: ...
    def cursor_get(self, user_id: str, session_id: str) -> int: ...
    def cursor_set(self, user_id: str, session_id: str, seq: int) -> None: ...
    def mark_failed(self, user_id: str, session_id: str, msg_id: str) -> int: ...
    def clear_failed(self, user_id: str, session_id: str, msg_id: str) -> None: ...
    def depth(self, user_id: str, session_id: str) -> int: ...
    def head_kind(self, user_id: str, session_id: str) -> str: ...
    def is_empty(self, user_id: str, session_id: str) -> bool: ...
    def deactivate_if_empty(self, user_id: str, session_id: str) -> bool: ...
    def pending_total(self) -> int: ...
    def reconcile_pending(self) -> int: ...


def _member(user_id: str, session_id: str) -> str:
    """Encode a noticeboard SET member as a JSON array, which carries (user, session) unambiguously and cannot collide on a separator."""
    return json.dumps([user_id, session_id], ensure_ascii=False)


def _decode_member(m) -> tuple[str, str]:
    if isinstance(m, bytes):
        m = m.decode("utf-8")
    u, s = json.loads(m)
    return u, s


def _s(v) -> Optional[str]:
    """Normalize a Redis return value to str, since the client may hand back bytes."""
    if v is None:
        return None
    return v.decode("utf-8") if isinstance(v, bytes) else v


class RedisMsgQueue:
    """The session queue in Redis. The key family, all of which carry a TTL:

      mq:{u}:{s}          the main queue (a List; LPUSH in on the left, RPOPLPUSH
                          out on the right, giving FIFO)
      mq:{u}:{s}:proc     the in-flight list (a List; the staging area that makes
                          dequeue reliable, removed only on ack)
      mqseq:{u}:{s}       the monotonic sequence number within the session (INCR)
      mqcur:{u}:{s}       the consumption cursor (the highest applied seq, used to
                          skip duplicates quickly)
      mq:active           the noticeboard (a SET whose members are encoded (u,s)
                          pairs, removed once the queue drains)
    """

    def __init__(self, client, ttl_s: int = QUEUE_TTL_S):
        self._c = client
        self._ttl = ttl_s

    # A session's keys share one {tag} so they land in one slot, which is what makes
    # RPOPLPUSH legal. active is a single key and needs no tag.
    def _mk(self, u, s): return f"{_env()}:personos:mq:{_sess_tag(u, s)}"
    def _pk(self, u, s): return f"{_env()}:personos:mq:{_sess_tag(u, s)}:proc"
    def _sk(self, u, s): return f"{_env()}:personos:mqseq:{_sess_tag(u, s)}"
    def _ck(self, u, s): return f"{_env()}:personos:mqcur:{_sess_tag(u, s)}"
    def _fk(self, u, s, msg_id): return f"{_env()}:personos:mqfail:{_sess_tag(u, s)}:{_esc(msg_id)}"
    def _elk(self, u, s): return f"{_env()}:personos:enqlock:{_sess_tag(u, s)}"
    def _ak(self): return _key("mq", "active")
    def _tk(self): return f"{_env()}:personos:mq:pending"   # global backlog counter (all sessions, all pods)

    def enqueue(self, user_id, session_id, payload, *, kind="ingest"):
        # The enqueue lock is separate from the consumption lock and guards a
        # critical section only milliseconds long: it makes "INCR for a number" and
        # "LPUSH into the list" atomic together.
        # Without it, concurrent enqueues into one session can have the order INCR
        # assigned and the order LPUSH physically landed disagree. The consumer then
        # takes the higher seq first in physical order, advances the cursor, and
        # misjudges the physically-late lower-seq message as a redelivery and drops
        # it — breaking both ordering and durability at once.
        # Because it is a separate key it never contends with the long drain lock on
        # the consumption side, so it does not slow the ingest 202 down.
        # The PX of 2s is deliberately shorter than the 3s spin deadline: if a holder
        # crashes, its lock lapses after 2s and a waiter can still take it before
        # giving up, so a crash never triggers the forced, out-of-order path. The
        # critical section is a handful of Redis round trips, so 2s is ample.
        elk = self._elk(user_id, session_id)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + 3.0
        while not self._c.set(elk, token, nx=True, px=2000):
            if time.monotonic() >= deadline:
                # Refuse on timeout rather than forcing our way in. Since the
                # deadline (3s) exceeds the PX (2s), a crashed holder's lock is
                # certainly expired by now, so reaching here means genuine
                # contention with a live holder. We raise, the layer above answers
                # 503, and the caller retries. We never run INCR and LPUSH
                # concurrently with a live holder, because that corrupts the
                # physical order and gets the lower seq dropped as a redelivery.
                raise EnqueueBusy(f"timed out contending for the enqueue lock: {user_id}/{session_id}")
            time.sleep(0.003)
        try:
            seq = int(self._c.incr(self._sk(user_id, session_id)))
            if seq == 1:
                # A new epoch: the seq key did not exist, or expired while idle.
                # Clear any leftover cursor, so that seq restarting at 1 cannot run
                # into a stale high cursor and have its messages dropped.
                self._c.delete(self._ck(user_id, session_id))
            msg_id = uuid.uuid4().hex
            raw = Envelope.make(msg_id, seq, kind, payload)
            mk = self._mk(user_id, session_id)
            self._c.lpush(mk, raw)                                 # in on the left
            self._c.sadd(self._ak(), _member(user_id, session_id))  # onto the noticeboard
            # Renew the main queue, the sequence key and the noticeboard. The cursor
            # is renewed by cursor_set, and proc by reserve and recover.
            self._c.expire(mk, self._ttl)
            self._c.expire(self._sk(user_id, session_id), self._ttl)
            self._c.expire(self._ak(), self._ttl)
            # Inside the enqueue lock, so the counter can never outrun the push.
            # No TTL on the counter key: it must live as long as the deployment.
            self._c.incr(self._tk())
        finally:
            self._c.eval(_ENQ_RELEASE_LUA, 1, elk, token)
        return msg_id, seq

    def active_sessions(self, limit=256):
        members = self._c.smembers(self._ak()) or set()
        # Sort before truncating. SMEMBERS returns a set in no particular order, and
        # sorting is what gives the dispatcher's round-robin a meaningful, stable
        # starting point, which is how it avoids starving a session. It also makes
        # "take the first limit entries" a determinate set rather than random drift
        # once there are more sessions than the limit.
        out = sorted(_decode_member(m) for m in members)
        return out[:limit]

    def recover(self, user_id, session_id):
        """Replay the in-flight list — messages left by an earlier crash, reserved
        but never acked — in ascending seq order, oldest first.
        """
        pk = self._pk(user_id, session_id)
        raws = self._c.lrange(pk, 0, -1) or []
        if raws:
            self._c.expire(pk, self._ttl)     # renew, so a message that only ever gets replayed by recover is not silently lost after 24h
        envs = [Envelope.from_raw(_s(r)) for r in raws]
        envs.sort(key=lambda e: e.seq)
        return envs

    def reserve(self, user_id, session_id):
        """Take one message reliably: RPOPLPUSH from the right end of the main queue,
        which is the oldest, onto the in-flight list. Returns None if there is none.
        """
        raw = _s(self._c.rpoplpush(self._mk(user_id, session_id), self._pk(user_id, session_id)))
        if raw is None:
            return None
        self._c.expire(self._pk(user_id, session_id), self._ttl)  # renew the in-flight list
        return Envelope.from_raw(raw)

    def head_kind(self, user_id, session_id):
        """Peek at the kind of the message at the head — the right end, the oldest —
        **without dequeuing it**.

        This lets the scheduler route by type, keeping video and text in separate
        pools. An empty queue or a parse failure returns an empty string, and the
        caller then uses the default pool.
        """
        try:
            raw = self._c.lrange(self._mk(user_id, session_id), -1, -1)
            if not raw:
                return ""
            return Envelope.from_raw(_s(raw[0])).kind or ""
        except Exception:  # noqa: BLE001  a failed peek must not affect scheduling
            return ""

    def ack(self, user_id, session_id, env):
        """Acknowledge consumption: remove exactly this envelope from the in-flight list, matching on its raw string, one occurrence."""
        if self._c.lrem(self._pk(user_id, session_id), 1, env.raw):
            # Decrement only when something was really removed: if the in-flight
            # list already expired (TTL), the counter keeps its (drifting) value
            # and reconcile_pending() puts it right.
            self._c.decr(self._tk())

    def cursor_get(self, user_id, session_id):
        v = _s(self._c.get(self._ck(user_id, session_id)))
        return int(v) if v else 0

    def cursor_set(self, user_id, session_id, seq):
        self._c.set(self._ck(user_id, session_id), str(int(seq)), ex=self._ttl)

    def mark_failed(self, user_id, session_id, msg_id):
        """Count one message's cumulative failures, keyed by msg_id, and return the
        new value. At the limit the message is judged poisonous.
        """
        fk = self._fk(user_id, session_id, msg_id)
        n = int(self._c.incr(fk))
        self._c.expire(fk, self._ttl)
        return n

    def clear_failed(self, user_id, session_id, msg_id):
        """Clear the failure count.

        Called once a message has been judged poisonous, so no junk key is left
        behind. On the success path there is no key and this does nothing.
        """
        self._c.delete(self._fk(user_id, session_id, msg_id))

    def depth(self, user_id, session_id):
        return int(self._c.llen(self._mk(user_id, session_id)) or 0)

    def is_empty(self, user_id, session_id):
        return (int(self._c.llen(self._mk(user_id, session_id)) or 0) == 0
                and int(self._c.llen(self._pk(user_id, session_id)) or 0) == 0)

    def pending_total(self):
        return int(_s(self._c.get(self._tk())) or 0)

    def reconcile_pending(self):
        """Recompute the global backlog counter from the queues themselves.

        The INCR/DECR counter drifts when a pod dies mid-drain or a session's
        keys expire unconsumed (TTL). The drift is usually upward (an ack lost
        to a crash is never replayed), which would eventually reject every
        write; a crash between LPUSH and INCR drifts downward instead. Either
        way, the dispatcher calls this periodically to sum the real queue
        lengths and overwrite the counter.
        """
        sessions = [_decode_member(m) for m in (self._c.smembers(self._ak()) or set())]
        # Per-key single commands, NOT a pipeline: the module's hard rule (the
        # proxy can misframe pipelined responses — see the docstring above).
        # This runs once a minute, so the extra round trips cost nothing.
        total = sum(int(self._c.llen(self._mk(u, s)) or 0) + int(self._c.llen(self._pk(u, s)) or 0)
                    for u, s in sessions)
        self._c.set(self._tk(), str(total))
        return total

    def deactivate_if_empty(self, user_id, session_id):
        """Once the queue drains, take the session off the noticeboard.

        After removing it we re-check the queue and put it back if a message just
        arrived, which closes the orphan race.
        """
        if not self.is_empty(user_id, session_id):
            return False
        self._c.srem(self._ak(), _member(user_id, session_id))
        # The re-check uses is_empty, which includes proc. If, in the instant we
        # deregistered, a new message entered main or another replica reserved one
        # into proc, we put the session back. Otherwise it would never be dispatched
        # again and the unacked messages in proc would sit there until the TTL
        # discarded them — an orphaned session.
        if not self.is_empty(user_id, session_id):
            self._c.sadd(self._ak(), _member(user_id, session_id))
            self._c.expire(self._ak(), self._ttl)
            return False
        return True


class MemoryMsgQueue:
    """The in-process implementation, for local use, a single replica, and unit
    tests. It is thread-safe, because the dispatcher's threads access one instance
    concurrently.

    Its semantics match RedisMsgQueue point for point, except that it does no TTL
    expiry, which is unnecessary within one process lifetime.
    """

    _CAP = 4096   # ceiling on session metadata, so a long-running single replica cannot leak; over it, drained sessions are collected

    def __init__(self):
        self._guard = threading.Lock()
        self._main: dict[tuple[str, str], list[str]] = {}   # in on the left, out on the right: index 0 is newest, -1 oldest
        self._proc: dict[tuple[str, str], list[str]] = {}
        self._seq: dict[tuple[str, str], int] = {}
        self._cur: dict[tuple[str, str], int] = {}
        self._fail: dict[tuple[str, str, str], int] = {}   # (u,s,msg_id) -> cumulative failure count
        self._active: set[tuple[str, str]] = set()
        self._pending = 0   # global backlog: queued + in-flight across all sessions

    def _gc_locked(self):
        """Call with _guard already held.

        When session metadata exceeds the cap, drop the leftover metadata of
        sessions that have drained, meaning both main and proc are empty.
        """
        if len(self._seq) <= self._CAP:
            return
        for k in [k for k in self._seq if not self._main.get(k) and not self._proc.get(k)]:
            self._seq.pop(k, None)
            self._cur.pop(k, None)
            for fk in [fk for fk in self._fail if fk[:2] == k]:
                self._fail.pop(fk, None)

    def enqueue(self, user_id, session_id, payload, *, kind="ingest"):
        k = (user_id, session_id)
        with self._guard:
            self._gc_locked()
            self._seq[k] = self._seq.get(k, 0) + 1
            seq = self._seq[k]
            if seq == 1:
                self._cur.pop(k, None)      # a new epoch: clear the stale cursor, as the Redis variant does
            msg_id = uuid.uuid4().hex
            raw = Envelope.make(msg_id, seq, kind, payload)
            self._main.setdefault(k, []).insert(0, raw)         # in on the left
            self._active.add(k)
            self._pending += 1
            return msg_id, seq

    def active_sessions(self, limit=256):
        with self._guard:
            return sorted(self._active)[:limit]   # sorted, as in the Redis variant, to keep the round-robin start point stable

    def recover(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            envs = [Envelope.from_raw(r) for r in self._proc.get(k, [])]
        envs.sort(key=lambda e: e.seq)
        return envs

    def head_kind(self, user_id, session_id):
        """The kind at the head — the right end, the oldest — without dequeuing. Same semantics as the Redis implementation."""
        with self._guard:
            main = self._main.get((user_id, session_id)) or []
            if not main:
                return ""
            try:
                return Envelope.from_raw(main[-1]).kind or ""
            except Exception:  # noqa: BLE001
                return ""

    def reserve(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            main = self._main.get(k) or []
            if not main:
                return None
            raw = main.pop()                                    # out on the right, the oldest
            self._proc.setdefault(k, []).insert(0, raw)
            return Envelope.from_raw(raw)

    def ack(self, user_id, session_id, env):
        k = (user_id, session_id)
        with self._guard:
            proc = self._proc.get(k)
            if proc and env.raw in proc:
                proc.remove(env.raw)
                self._pending -= 1

    def cursor_get(self, user_id, session_id):
        with self._guard:
            return self._cur.get((user_id, session_id), 0)

    def cursor_set(self, user_id, session_id, seq):
        with self._guard:
            self._cur[(user_id, session_id)] = int(seq)

    def mark_failed(self, user_id, session_id, msg_id):
        with self._guard:
            k = (user_id, session_id, msg_id)
            self._fail[k] = self._fail.get(k, 0) + 1
            return self._fail[k]

    def clear_failed(self, user_id, session_id, msg_id):
        with self._guard:
            self._fail.pop((user_id, session_id, msg_id), None)

    def depth(self, user_id, session_id):
        with self._guard:
            return len(self._main.get((user_id, session_id)) or [])

    def is_empty(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            return not (self._main.get(k) or self._proc.get(k))

    def pending_total(self):
        with self._guard:
            return self._pending

    def reconcile_pending(self):
        """Same contract as the Redis variant: recompute from the queues. The
        in-memory counter cannot drift (one process, no TTL), so this only
        matters for tests and for keeping the two implementations interchangeable."""
        with self._guard:
            self._pending = sum(len(self._main.get(k) or []) + len(self._proc.get(k) or [])
                                for k in self._active)
            return self._pending

    def deactivate_if_empty(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            if self._main.get(k) or self._proc.get(k):
                return False
            self._active.discard(k)
            if self._main.get(k):                               # re-check: a message arrived in the instant we deregistered
                self._active.add(k)
                return False
            return True
