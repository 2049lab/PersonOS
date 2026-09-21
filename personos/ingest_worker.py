"""The two roles of ordered consumption: SessionConsumer (consumes one session's queue)
and Dispatcher (decides who consumes what).

Architecture (see the design alignment notes under docs/design):
- **One queue per session** (msg_queue); messages queue up FIFO by seq and are durable
  across replicas.
- **Dispatcher**: one background loop per pod that scans the board (the sessions with
  pending messages) and, for each session, submits a "drain this session" job to the
  ingest pool; if the pool is full it skips this round (backpressure) and comes back on
  the next one — it never spins and never holds a thread.
- **SessionConsumer.drain_session**: take the session lock without blocking
  (single-flight: only one consumer per session at a time, mutually exclusive across
  replicas) -> first replay the in-flight messages left over from the previous crash
  (recover) -> then drain FIFO for at most max_drain messages (fairness: a chatty
  session must not hog a thread) -> release the lock -> once drained, take the session
  off the board.
- **No duplicates**: every message's seq is compared with the session cursor; if
  seq <= cursor it is a redelivery and is acked and skipped; after a successful
  consumption the cursor advances and only then is the message acked. A reliable queue
  plus a cursor gives exactly-once in the normal case, and at-least-once with idempotent
  dedup after a crash.

Dependencies are injected (no coupling to the runtime or FastAPI), which keeps unit
testing easy: just pass mq / lock / the writer_for callback.
"""

from __future__ import annotations

import base64
import threading
import uuid
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from . import obs
from .online.write_path import FeedMsg, SessionWriter
from .storage.msg_queue import Envelope, MsgQueue
from .storage.session_lock import SessionLock


@dataclass
class DrainReport:
    """The result of one drain_session call (for the dispatcher, tests and logs)."""
    locked: bool               # Whether we got the lock (False = someone else is consuming this session, this call did nothing)
    applied: int = 0           # Number of messages actually applied
    skipped: int = 0           # Number of messages skipped as redeliveries
    poisoned: int = 0          # Number of messages skipped as poison messages (consecutive failures hit the limit)
    more: bool = False         # The queue still had a backlog when we released the lock (we stopped at max_drain, or new messages arrived meanwhile)


class SessionConsumer:
    """Consumes a single session queue: single-flight + ordered + idempotent dedup. It
    holds no state of its own, so several threads can share one instance."""

    def __init__(self, mq: MsgQueue, lock: SessionLock,
                 writer_for: Callable[[str, str], SessionWriter],
                 *, max_drain: int = 20, max_retries: int = 5, renew_interval_s: float = 200.0,
                 after_drain: Callable[[str, str, str], None] | None = None,
                 video_deps: Callable[[str], Any] | None = None,
                 task_store: Any | None = None):
        self._mq = mq
        self._lock = lock
        self._writer_for = writer_for
        # Getter for the video dependencies (runtime.video_deps); None = this process
        # does not handle video (a text-only pod).
        self._video_deps = video_deps
        self._task_store = task_store   # Records permanent clip failures (may be None: no record, but nothing is blocked)
        self._max_drain = max_drain
        self._max_retries = max_retries   # Once a message fails this many times in a row it is a poison message and is skipped
        self._renew_interval = renew_interval_s   # Lease renewal interval (must be < the lock TTL; we use TTL/3)
        # Optional post-segment hook: called after some messages were applied, e.g. to
        # trigger profile consolidation. It runs after the lock is released, and a
        # failure there does not affect consumption.
        # Its third argument is the scenario of the most recent ingest batch that
        # triggered it (a per-request value passed through to profile consolidation).
        self._after_drain = after_drain

    def _record_failure(self, user_id: str, session_id: str, kind: str, detail: str) -> None:
        """Record a failure in the tasks table, so "this content never made it into
        memory" is queryable and alertable instead of only sitting in logs that will be
        rotated away. kind looks like video_clip_rejected / poisoned_video /
        poisoned_ingest. A failure to record does not block consumption."""
        if self._task_store is None:
            return
        try:
            tid = f"fail_{uuid.uuid4().hex[:16]}"
            self._task_store.create(tid, kind, user_id, session_id)
            self._task_store.mark_error(tid, detail)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not record the failure (consumption not blocked): {e}")

    def _do_write(self, user_id: str, session_id: str, env: Envelope,
                  last_scenario: dict | None = None) -> None:
        """The actual write (may raise): session_end forces the segment closed, ingest
        feeds a batch in.

        last_scenario: a mutable carrier (local to one drain) that records this batch's
        scenario so after_drain can pass it through to profile consolidation (the
        per-request scenario is never persisted, so it has to be carried out by "the
        ingest batch that triggered it").
        """
        writer = self._writer_for(user_id, session_id)
        # The langfuse root trace: consuming one message = one trace (the chat calls of
        # W1 boundary detection, W2 episode/atom extraction and W2.5 chain assignment
        # nest under this span automatically). input is set to the message text so the
        # langfuse main view is readable rather than blank.
        # Note: the first message of a segment is only buffered and triggers no LLM call,
        # so that trace has no sub-stages (expected — the LLM work shows up in the trace
        # of the message that closes the segment).
        if env.kind == "session_end":
            _inp = "(session end: close the trailing text segment + final video review)"
        elif env.kind == "video":
            # Careful: the video_oss_key key exists but its value may be None (when
            # video_url is used instead), so dict.get's default never kicks in — the
            # `or ''` fallback is required, otherwise None[-16:] raises TypeError and
            # turns the whole message into a poison message.
            _inp = " | ".join(
                f"[clip {(m.get('video_oss_key') or m.get('video_url') or '')[-16:]}]"
                for m in env.payload.get("messages", []))[:500]
        else:
            _inp = " | ".join((str(m.get("text")) if m.get("text") else "[image]")
                              for m in env.payload.get("messages", []))[:500]
        with obs.root_span(f"ingest.{env.kind}", user_id=user_id, session_id=session_id,
                           input=_inp, metadata={"msg_id": env.msg_id, "seq": env.seq},
                           trace_id=env.payload.get("trace_id")) as _sp:
            if _sp is not None:
                logger.info(f"ingest langfuse trace_id={obs.current_trace_id()} "
                            f"kind={env.kind} u={user_id} s={session_id}")
            scenario = env.payload.get("scenario") or ""
            if last_scenario is not None:
                last_scenario["scenario"] = scenario   # Remember the latest batch's scenario (for after_drain -> profile)
            if env.kind == "session_end":
                writer.end_session(task_type=env.payload.get("task_type"),
                                   scenario=scenario)   # Wrap up: close the trailing text segment
                # Final video review: if this session has a pending video draft, commit
                # the session and write the per-row attributions into memory
                # (finalize_video checks for pending work itself and is a no-op without
                # video). It runs serially inside the same drain lock as the trailing
                # text segment.
                if self._video_deps is not None:
                    from personos.online import video_ingest
                    video_ingest.finalize_video(self._video_deps(user_id), session_id=session_id)
            elif env.kind == "video":
                if self._video_deps is None:
                    raise RuntimeError("video backends are not configured in this process (text-only worker), cannot consume a video message")
                from personos.online import video_ingest
                deps = self._video_deps(user_id)
                for m in env.payload.get("messages", []):     # One batch may hold several clips; process them in order
                    src = m.get("video_oss_key") or m.get("video_url") or ""
                    try:
                        video_ingest.process_clip(
                            deps, session_id=session_id,
                            clip_key=m.get("video_oss_key") or "",
                            clip_url=m.get("video_url") or "",
                            scene=scenario or video_ingest.DEFAULT_SCENE,
                            clip_meta={"clip_index": m.get("clip_index"),
                                       "duration_sec": m.get("duration_sec")})
                    except video_ingest.ClipRejected as e:
                        # A permanent failure (dead external link, too large, too long,
                        # wrong format): retrying is pointless, so **record it** and skip
                        # this clip — never drop it silently (the caller can look up the
                        # task to see which clip never made it into memory). The rest of
                        # the batch continues.
                        self._record_failure(user_id, session_id, "video_clip_rejected",
                                             f"src={src}\n{e}")
                        logger.warning(f"clip rejected (recorded, skipped) u={user_id} s={session_id} "
                                       f"src={src[-40:]} reason={e}")
            else:
                p = env.payload
                msgs = [FeedMsg(speaker=m.get("speaker", "user"), text=m.get("text", ""),
                                image=base64.b64decode(m["image_b64"]) if m.get("image_b64") else None,
                                image_content_type=m.get("image_content_type", "image/jpeg"))
                        for m in p.get("messages", [])]       # One queue message = one atomic batch
                writer.feed_batch(msgs, source_extra=p.get("source_extra"),
                                  task_type=p.get("task_type"), scenario=scenario)

    def _apply(self, user_id: str, session_id: str, env: Envelope,
               last_scenario: dict | None = None) -> str:
        """Apply one message (the lock is already held). Returns 'applied' | 'skipped'
        (a redelivery) | 'poisoned' (a poison message that was skipped).

        - Idempotent: seq <= cursor means a redelivery, so we ack and skip (re-fetching
          the same message during crash recovery never records it twice).
        - On a write failure: while the failure count is below the limit, re-raise (the
          message stays in flight and is retried next round, so a transient fault heals
          itself); once the limit is reached, treat it as a poison message, log an ERROR
          and advance the cursor past it (which unblocks the head of the queue so later
          messages can proceed).
        """
        cur = self._mq.cursor_get(user_id, session_id)
        if env.seq <= cur:
            logger.debug(f"dedup skip u={user_id} s={session_id} seq={env.seq}<=cursor={cur}")
            self._mq.ack(user_id, session_id, env)
            return "skipped"
        try:
            self._do_write(user_id, session_id, env, last_scenario)
        except Exception:   # noqa: BLE001
            n = self._mq.mark_failed(user_id, session_id, env.msg_id)
            if n < self._max_retries:
                raise                                   # Below the limit: let drain_session log it and leave the message in flight for a retry
            text = ""
            if env.payload.get("messages"):
                text = str(env.payload["messages"][0].get("text") or "")[:80]   # Guard against a non-string text
            extra = "(the trailing segment was never closed, so the tail of this session will not become a cell; it expires via the 24h seg TTL)" \
                if env.kind == "session_end" else ""
            logger.error(f"poison message skipped ({n} consecutive failures) u={user_id} s={session_id} seq={env.seq} "
                         f"msg_id={env.msg_id} kind={env.kind} text={text!r}{extra}")
            # Declared poison: advance the cursor and dequeue, treating it as handled
            # (its content never reaches memory). Besides the ERROR log we **also leave a
            # task row** so "the memory of this message was lost" is queryable and
            # alertable — which matters most for video: if a clip never made it into
            # memory, once the logs rotate there is no way to trace it.
            self._record_failure(user_id, session_id, f"poisoned_{env.kind}",
                                 f"seq={env.seq} msg_id={env.msg_id} kind={env.kind} "
                                 f"text={text!r}{extra}\nskipped as a poison message after {n} consecutive failures")
            self._mq.cursor_set(user_id, session_id, env.seq)
            self._mq.ack(user_id, session_id, env)
            self._mq.clear_failed(user_id, session_id, env.msg_id)   # Clear the counter key once used, leaving no garbage behind
            return "poisoned"
        # Advance the cursor before acking: if we crash between the two steps, the
        # re-fetch will find seq <= cursor and skip it, so nothing is applied twice.
        self._mq.cursor_set(user_id, session_id, env.seq)
        self._mq.ack(user_id, session_id, env)
        return "applied"

    def drain_session(self, user_id: str, session_id: str) -> DrainReport:
        """Consume one session: take the lock -> replay in-flight messages -> drain FIFO
        (at most max_drain) -> release the lock -> if drained, take it off the board."""
        token = self._lock.try_acquire(user_id, session_id)
        if not token:
            return DrainReport(locked=False)            # Someone else is consuming it; leave it to them
        applied = skipped = poisoned = 0
        last_scenario = {"scenario": ""}   # Scenario of the latest ingest batch in this drain (passed through to profile consolidation)

        def _tally(r):
            nonlocal applied, skipped, poisoned
            applied += r == "applied"
            skipped += r == "skipped"
            poisoned += r == "poisoned"

        # Lease renewal: a background thread renews every TTL/3, which covers both "the
        # whole drain" and "a single very long apply" (a batch holds up to 20 utterances
        # and each one may spend tens of seconds in W1/W2/image understanding, so one
        # apply can approach the TTL). As long as this pod lives the lease keeps being
        # renewed, so the lock never expires mid-work; if the pod dies the heartbeat dies
        # with the process, the lock expires, and another replica takes over. This is
        # stricter than renewing only after each message finishes (which does nothing for
        # a single overlong message).
        stop_hb = threading.Event()

        def _heartbeat():
            while not stop_hb.wait(self._renew_interval):
                self._lock.renew(user_id, session_id, token)

        hb = threading.Thread(target=_heartbeat, name="drain-hb", daemon=True)
        hb.start()
        try:
            for env in self._mq.recover(user_id, session_id):   # Left over from a crash: replay in order first
                _tally(self._apply(user_id, session_id, env, last_scenario))
            n = 0
            while n < self._max_drain:
                env = self._mq.reserve(user_id, session_id)
                if env is None:
                    break
                _tally(self._apply(user_id, session_id, env, last_scenario))
                n += 1
        except Exception:   # noqa: BLE001  One failed message must not swallow the rest: it stays in the processing set and is retried by the next recover
            logger.exception(f"session consumption error u={user_id} s={session_id} (message stays in flight, retried next round)")
        finally:
            stop_hb.set()
            hb.join(timeout=1)
            self._lock.release(user_id, session_id, token)
        more = self._mq.depth(user_id, session_id) > 0
        if not more:
            self._mq.deactivate_if_empty(user_id, session_id)   # Drained, so take it off the board (this also re-checks the orphan race)
        if applied or skipped or poisoned:
            logger.info(f"session consumption u={user_id} s={session_id} applied={applied} "
                        f"skipped={skipped} poisoned={poisoned} more={more}")
        if applied and self._after_drain is not None:           # Something was written -> trigger profile consolidation (after releasing the lock, non-blocking)
            try:
                self._after_drain(user_id, session_id, last_scenario["scenario"])
            except Exception:   # noqa: BLE001  A failed profile trigger must never affect the ingest consumption result
                logger.exception(f"after_drain hook error u={user_id} s={session_id}")
        return DrainReport(locked=True, applied=applied, skipped=skipped,
                           poisoned=poisoned, more=more)


class _Admission:
    """Non-blocking in-flight permit gate: reject when full. Same semantics as
    app.admission.AdmissionGate; inlined here to avoid a circular import."""

    def __init__(self, cap: int):
        self._sem = threading.BoundedSemaphore(cap)

    def try_enter(self) -> bool:
        return self._sem.acquire(blocking=False)

    def leave(self) -> None:
        self._sem.release()


class Dispatcher:
    """One per pod: a background loop that scans the board and hands sessions with
    pending work to the ingest pool. When the pool is full it applies backpressure and
    skips."""

    def __init__(self, mq: MsgQueue, consumer: SessionConsumer, pool, cap: int,
                 *, tick_s: float = 0.05, idle_tick_s: float = 0.5, batch: int = 256,
                 video_pool=None, video_cap: int = 0):
        self._mq = mq
        self._consumer = consumer
        self._pool = pool                 # ThreadPoolExecutor (text ingest only)
        self._gate = _Admission(cap)      # Same capacity as the pool: caps in-flight drain jobs (backpressure)
        # A separate video pool: processing a clip is heavy work (writing to disk + local
        # inference + 2-3 min of multimodal inference), and sharing a pool with text
        # would occupy every worker and starve text consumption. Routing is by the
        # **kind of the message at the head of the session queue**: a session whose head
        # is video goes to the video pool, everything else to the text pool — so a text
        # session is never queued behind a video job. If no video pool is configured we
        # fall back to sharing one (backward compatible).
        self._video_pool = video_pool
        # In-flight permits matching the video pool's capacity (backpressure: don't push
        # more in once the pool is full). This is unrelated to video_ingest — that module
        # used to have a semaphore by the same name, now removed; video concurrency is
        # determined solely by the pool size.
        self._video_admission = _Admission(video_cap) if video_pool is not None else None
        self._tick = tick_s
        self._idle_tick = idle_tick_s
        self._batch = batch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Sessions dispatched but not yet finished within this pod: never dispatch the
        # same session twice (otherwise repeated dispatches of an early session fill
        # every slot and starve sessions that arrive later — the lock only guarantees
        # single-flight across replicas, so in-process starvation needs this dedup layer)
        self._inflight: set[tuple[str, str]] = set()
        self._inflight_guard = threading.Lock()
        # Dispatch order: the session dispatched **longest ago** goes first (LRU). We
        # used to keep a round-robin start pointer that advanced every round, but that
        # pointer moved on the tick (milliseconds) while permits were released on drain
        # completion (seconds to minutes), so whoever the pointer happened to point at
        # the instant a permit freed up was pure chance — a session that had just
        # finished would often grab the permit right back, and later sessions had a
        # "chance" but no guarantee. Measured (cap=1, a long session of 30 messages plus
        # a short session arriving after it), in about 10% of runs the short session
        # waited until the long one had drained completely.
        # With LRU, "the session just dispatched goes to the back of the line" is
        # deterministic and independent of timing phase.
        self._dispatch_seq = 0
        self._last_dispatch: dict[tuple[str, str], int] = {}

    def run_once(self) -> int:
        """One scheduling round (unit-testable): scan the board and, while permits are
        available, dispatch drain jobs. Returns how many jobs this round dispatched.

        Even if the same session is dispatched several times, the consumer's
        non-blocking lock guarantees only one of them actually drains while the rest
        return immediately — so the dispatcher needs no "in progress" state of its own,
        the lock dedups naturally.
        """
        sessions = self._mq.active_sessions(self._batch)
        if sessions:
            # A session never dispatched counts as -1, so it sorts ahead of everything
            # already dispatched (new sessions go first and aren't blocked by an old
            # chatty one)
            sessions = sorted(sessions, key=lambda k: self._last_dispatch.get(k, -1))
            live = set(sessions)            # Drop sessions that left the board promptly, so the dict doesn't grow without bound over time
            if len(self._last_dispatch) > len(live):
                self._last_dispatch = {k: v for k, v in self._last_dispatch.items()
                                       if k in live}
        dispatched = 0
        video_full = text_full = False
        for user_id, session_id in sessions:
            if video_full and text_full:
                break                     # Both pools are full: stop this round and come back next time (backpressure, no spinning)
            key = (user_id, session_id)
            with self._inflight_guard:
                if key in self._inflight:
                    continue              # This session is already dispatched: skip it and give the slot to another session (anti-starvation)
            pool, gate, is_video = self._route(user_id, session_id)
            if not gate.try_enter():
                # That pool is full: skip only sessions of that kind, the other kind can
                # still be dispatched (busy video doesn't drag text down). We only stop
                # once both are full.
                if is_video:
                    video_full = True
                else:
                    text_full = True
                continue
            # Do the check-and-add under a single lock hold, which makes the invariant
            # hold without relying on the fragile assumption that only one thread ever
            # calls run_once.
            with self._inflight_guard:
                if key in self._inflight:
                    gate.leave()
                    continue
                self._inflight.add(key)
            try:
                pool.submit(self._job, user_id, session_id, gate)
                self._dispatch_seq += 1          # Just dispatched -> go to the back of the line (next round others go first)
                self._last_dispatch[key] = self._dispatch_seq
            except Exception:   # noqa: BLE001  Submission failed (pool shut down, etc.): roll back inflight and return the permit, leaking neither
                with self._inflight_guard:
                    self._inflight.discard(key)
                gate.leave()
                raise
            dispatched += 1
        return dispatched

    def _route(self, user_id: str, session_id: str):
        """Pick a pool by the **kind of the message at the head of the session queue**:
        video goes to the video pool, everything else to the text pool. Returns
        (pool, gate, is_video). Without a video pool configured, everything goes to the
        text pool (backward compatible)."""
        if self._video_pool is not None and self._mq.head_kind(user_id, session_id) == "video":
            return self._video_pool, self._video_admission, True
        return self._pool, self._gate, False

    def _job(self, user_id: str, session_id: str, gate=None) -> None:
        try:
            self._consumer.drain_session(user_id, session_id)
        finally:
            with self._inflight_guard:
                self._inflight.discard((user_id, session_id))
            (gate or self._gate).leave()   # The permit must go back to **the pool it came from**, on every path

    def _loop(self) -> None:
        logger.info("ingest dispatcher started")
        while not self._stop.is_set():
            try:
                n = self.run_once()
            except Exception:   # noqa: BLE001  The scheduling loop must never exit because one round raised
                logger.exception("dispatcher scheduling round failed")
                n = 0
            self._stop.wait(self._tick if n else self._idle_tick)
        logger.info("ingest dispatcher stopped")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ingest-dispatcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
