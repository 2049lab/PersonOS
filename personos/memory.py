"""The public entry point: :class:`Memory`.

    from personos import Memory

    m = Memory()
    m.add("I moved to Shanghai in June", user_id="alice", session_id="chat-1")
    m.end_session(user_id="alice", session_id="chat-1")
    print(m.search("where do I live?", user_id="alice").ans.answer)

``add`` and ``search`` are thin wrappers. They normalise arguments, check that
the capabilities the call needs are actually configured, and hand off to the
pipeline unchanged — the memory algorithms are reached by exactly the same code
path whether you call this class or the HTTP server.

Multi-tenancy: one process-wide object holds connections, model clients and
thread pools; each user gets its own bound stores (cached, LRU-capped). The
stores carry their user_id internally, so isolation is structural rather than a
filter someone has to remember to add.

Stateless by design, so several workers can run: unclosed segments, session
locks and the ingest queue live in Redis when it is configured, and in memory
otherwise. The write state machine holds no instance state.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from . import obs
from .providers.registry import build as build_provider
from .config import settings
from .admission import AdmissionGate, TaskOverloaded
from .ingest_worker import Dispatcher, SessionConsumer
from .logging_setup import setup_logging
from .models import now
from .online.profile_consolidate import run_user_consolidation, should_consolidate
from .online.rerank import ScoringReranker
from .storage.profile_store import ProfileStore
from .online.write_path import MAX_SEGMENT_TURNS, SessionWriter
from .storage.atom_store import AtomStore
from .storage.cell_store import CellStore
from .storage.chain_store import ChainStore
from .storage.db import Database
from .storage.evidence_store import EvidenceStore
from .storage.media import media_store_from_settings
from .storage.msg_queue import MemoryMsgQueue, MsgQueue, RedisMsgQueue
from .storage.redis_client import get_redis
from .storage.seg_store import MemorySegStore, RedisSegStore, SegStore
from .storage.session_lock import LOCK_TTL_S, MemorySessionLock, RedisSessionLock, SessionLock
from .storage.task_store import TaskStore
from .storage.user_store import UserStore

_UCTX_CAP = 1024          # LRU cap on user contexts; rebuilding stores is free, so evict the oldest past the cap
_SWEEP_INTERVAL_S = 300   # How often the task table sweep runs (reaping zombies, clearing expired rows)
_MAX_PENDING = 200        # Cap on in-flight background tasks: reject (503) when full, see admission.AdmissionGate


class UserContext:
    """One user's bound context: its stores filter to that user automatically (the write
    state machine no longer holds instance state)."""

    def __init__(self, user_id: str, db: Database):
        self.user_id = user_id
        self.evidence = EvidenceStore(db, user_id=user_id)
        self.atoms = AtomStore(db, user_id=user_id)
        self.cells = CellStore(db, user_id=user_id)
        self.chains = ChainStore(db, user_id=user_id)


class Memory:
    """Layered long-term memory. See the module docstring for a quickstart."""

    def __init__(self):
        setup_logging(settings.log_dir)
        self.db = Database()
        self.users = UserStore(self.db)
        # Defaults (the single-tenant view for setups without a user system) — kept for
        # older scripts; the service API always goes through for_user
        self.evidence = EvidenceStore(self.db)
        self.atoms = AtomStore(self.db)
        self.cells = CellStore(self.db)
        # Providers are resolved by name, so a fork can register its own gateway
        # without editing this file. Chat and embedding are separate objects
        # even though one endpoint usually serves both.
        self.llm = build_provider("llm", settings.llm_provider)
        self.embedder = build_provider("embedder", settings.embedder_provider)
        reranker_provider = settings.reranker_provider or (
            "openai" if settings.rerank_api_key and settings.rerank_model else "noop")
        scorer = build_provider("reranker", reranker_provider) if reranker_provider == "openai" else None
        # R2: wrap the scorer so a failed rerank degrades to pass-through order
        # instead of blocking the main path (see ScoringReranker).
        self.reranker = ScoringReranker(scorer) if scorer else build_provider("reranker", "noop")
        self.task_store = TaskStore(self.db)
        # Image input: the multimodal client that looks at images (available=False when
        # no key is configured, in which case the write path degrades to text only).
        # Object storage is built lazily — wired up on the first ingest that carries an
        # image; if credentials are missing media_store stays None, so the original
        # image isn't kept but nothing is blocked.
        self.mllm = build_provider(
            "mllm", settings.mllm_provider if settings.mllm_api_key else "none")
        self._media_store = None
        self._media_guard = threading.Lock()
        # -- Cross-replica session state (segments / locks): wired up on first use
        # (the pool is built lazily, so importing never touches the network) --
        self._warn_redis_env_mismatch()
        self._state_guard = threading.Lock()
        self._seg_store: SegStore | None = None
        self._session_lock: SessionLock | None = None
        # -- Async plumbing: a background thread pool (every state transition is written
        # to the tasks table) plus the in-flight cap gate --
        # The legacy background task pool (submit_task; ingest now runs off the queue, so
        # this pool only handles occasional low-frequency tasks). Not a small value like 4.
        self._executor = obs.ContextThreadPoolExecutor(max_workers=16, thread_name_prefix="rt-bg")
        self._admission = AdmissionGate(_MAX_PENDING)
        self._sweep_at = 0.0
        # -- Ordered-consumption task system (Phase B): a pool per kind + session queues
        # + a dispatcher --
        # ingest_exec runs the drain jobs the dispatcher hands out. recall no longer gets
        # its own pool: the sync endpoint already occupies one anyio thread, and
        # submitting to another pool and blocking on it would occupy two threads for no
        # gain; it now runs on the current thread and only uses recall_gate for rate
        # limiting and isolation.
        # These are context-preserving pools: each drain task runs inside a copy of the
        # submitting thread's context (the dispatcher, which has no active span), which
        # isolates it from OTel context left behind in a reused worker thread —
        # otherwise, under concurrency, the previous task's leftover span pollutes the
        # next one and build_cell's episode_weave/atom_extract/chain_assign detach from
        # the ingest root, becoming orphan traces.
        self.ingest_exec = obs.ContextThreadPoolExecutor(max_workers=settings.ingest_pool_size,
                                                         thread_name_prefix="ingest")
        # A separate pool for video consumption: processing a clip is heavy work (writing
        # to disk + local face/voiceprint inference + 2-3 min of multimodal inference),
        # and sharing one pool with text would occupy every worker and starve text
        # consumption. The dispatcher routes work here based on the kind at the head of
        # the session queue.
        self.video_exec = obs.ContextThreadPoolExecutor(max_workers=settings.video_pool_size,
                                                        thread_name_prefix="video")
        self.recall_gate = AdmissionGate(settings.recall_pool_size)
        self._msgq: MsgQueue | None = None
        self._consumer: SessionConsumer | None = None
        self._dispatcher: Dispatcher | None = None
        # -- Profile consolidation pool (does not go through the queue): triggered after a
        # segment closes -> submitted to this pool -> per-user single-flight lock (on the
        # pseudo-session "profile"). If the previous run hasn't finished, the next trigger
        # fails to take the lock and simply returns, and the trigger after that heals it
        # (the work is idempotent — it reads all new cells).
        self.profile_exec = obs.ContextThreadPoolExecutor(max_workers=settings.profile_pool_size,
                                                          thread_name_prefix="profile")
        # user -> bound context (LRU capped; the oldest is evicted past the cap)
        self._uctx_guard = threading.Lock()
        self._uctx: OrderedDict[str, UserContext] = OrderedDict()
        # -- Video identity backends: a process-wide lazily loaded singleton (the heavy
        # InsightFace/ECAPA models are loaded once and shared by every drain thread;
        # gated by the env var PERSONOS_VIDEO_BACKEND, default real). Session drafts are
        # cached per user (the Redis implementation is stateless; the in-memory one must
        # persist within the same instance).
        self._video_backends: dict | None = None
        self._video_guard = threading.Lock()
        self._draft_guard = threading.Lock()
        self._draft_stores: "OrderedDict[str, Any]" = OrderedDict()

    # -- Multi-tenancy --
    def for_user(self, user_id: str) -> UserContext:
        """Get a user's bound context (LRU cached): its stores act on that user only."""
        with self._uctx_guard:
            ctx = self._uctx.get(user_id)
            if ctx is None:
                ctx = UserContext(user_id, self.db)
                self._uctx[user_id] = ctx
                if len(self._uctx) > _UCTX_CAP:
                    self._uctx.popitem(last=False)
            else:
                self._uctx.move_to_end(user_id)
            return ctx

    def ctx_by_token(self, token: str) -> UserContext | None:
        """Resolve a caller's context from its token; an unknown token -> None (the API
        layer turns that into a 401)."""
        uid = self.users.user_id_by_token(token)
        return self.for_user(uid) if uid else None

    def close_user(self, user_id: str) -> None:
        # The stores share the globally pooled connection, so there is no per-user
        # connection to close; we only drop the entry from the LRU explicitly.
        with self._uctx_guard:
            self._uctx.pop(user_id, None)

    def writer_for(self, user_id: str, session_id: str,
                   max_turns: int = MAX_SEGMENT_TURNS) -> SessionWriter:
        """Build a write state machine (no instance state: segment state lives in
        seg_store and is shared across requests and replicas)."""
        ctx = self.for_user(user_id)
        return SessionWriter(self.llm, self.embedder, ctx.evidence, ctx.cells, ctx.atoms,
                             session_id=session_id, user_id=user_id, max_turns=max_turns,
                             seg_store=self._seg(), chain_store=ctx.chains,
                             media_store=self._media(), mllm=self.mllm)

    def _media(self):
        """Build the object storage client lazily: missing credentials or a missing SDK
        -> return None (an ingest with images can still look at them, the originals just
        aren't kept)."""
        if self._media_store is None:
            with self._media_guard:
                if self._media_store is None:
                    try:
                        self._media_store = media_store_from_settings(settings)
                    except Exception as e:   # noqa: BLE001
                        logger.warning(f"media object storage not wired up (images will not be kept): {e}")
                        self._media_store = False   # Mark that we tried, so we don't retry on every call
        return self._media_store or None

    # -- Cross-replica session state --
    def _seg(self) -> SegStore:
        if self._seg_store is None:
            with self._state_guard:
                if self._seg_store is None:
                    self._seg_store, self._session_lock = self._make_state()
        return self._seg_store

    def session_lock(self, user_id: str, session_id: str):
        """Per-(user, session) write lock: ingest and session-end for the same session
        contend for the same lock, so they never interleave.

        The Redis implementation gives cross-replica mutual exclusion; without
        REDIS_CLUSTER configured it is in-process (single replica). Returns a context
        manager.
        """
        if self._session_lock is None:
            with self._state_guard:
                if self._session_lock is None:
                    self._seg_store, self._session_lock = self._make_state()
        return self._session_lock(user_id, session_id)

    # -- Dependencies for video identity consumption --
    def video_backends(self) -> dict:
        """Process-wide lazily loaded singleton of the video identity backends (the heavy
        models are loaded once and shared by every drain thread).
        The profile comes from the env var PERSONOS_VIDEO_BACKEND (**default real**; mock
        is for tests only — it fabricates scripts and fake face vectors, so running mock
        in production means writing forged memories into the real database)."""
        if self._video_backends is None:
            with self._video_guard:
                if self._video_backends is None:
                    from .identity.backends.factory import make_backends
                    self._video_backends = make_backends()   # the env gate lives inside make_backends
                    logger.info("video identity backends ready (process-wide singleton)")
        return self._video_backends

    def _draft_for(self, user_id: str):
        """Session draft storage (cached per user): the Redis implementation is stateless
        (caching only saves construction), while the in-memory one must persist within
        the same instance."""
        with self._draft_guard:
            d = self._draft_stores.get(user_id)
            if d is None:
                from .identity.draft import MemoryDraftStore, RedisDraftStore
                d = (RedisDraftStore(get_redis(), user_id) if settings.redis_url
                     else MemoryDraftStore(user_id))
                self._draft_stores[user_id] = d
                if len(self._draft_stores) > _UCTX_CAP:
                    self._draft_stores.popitem(last=False)
            return d

    def video_deps(self, user_id: str):
        """Assemble the full dependency set for video consumption (VideoDeps): backends
        is the process-wide singleton, everything else is per user."""
        from .identity.cloud import CloudEngine
        from .identity.store import CharacterStore
        from .online.video_ingest import VideoDeps
        ctx = self.for_user(user_id)
        store = CharacterStore(self.db, user_id)
        return VideoDeps(
            store=store, cloud=CloudEngine(store), draft=self._draft_for(user_id),
            backends=self.video_backends(), media_store=self._media(),
            llm=self.llm, embedder=self.embedder,
            evidence=ctx.evidence, cells=ctx.cells, atoms=ctx.atoms, chains=ctx.chains)

    def visual_deps(self, user_id: str):
        """Assemble the dependencies for visual query rewriting on the recall side
        (VisualDeps).

        It shares the same backends singleton as video consumption (the heavy models
        exist once per process) and the same identity assets; the difference is that it
        **writes no drafts** — looking at an image to rewrite a query is a read path and
        produces no identity changes.
        draft is passed as None: only registry.candidate_card is used, and that path
        never touches the draft.
        """
        from .identity.cloud import CloudEngine
        from .identity.registry import AnchorRegistry
        from .identity.store import CharacterStore
        from .online.visual_query import VisualDeps
        store = CharacterStore(self.db, user_id)
        cloud = CloudEngine(store)
        return VisualDeps(store=store, cloud=cloud, backends=self.video_backends(),
                          registry=AnchorRegistry(store, cloud, None,
                                                  media_store=self._media()))

    def _make_state(self) -> tuple[SegStore, SessionLock]:
        """Build (seg_store, session_lock) according to configuration. A Redis failure is
        not swallowed: it is raised as-is on first use."""
        if settings.redis_url:
            client = get_redis()
            return RedisSegStore(client), RedisSessionLock(client)
        return MemorySegStore(), MemorySessionLock()

    # -- Ordered-consumption task system (Phase B) --
    def msg_queue(self) -> MsgQueue:
        """The session message queue (built lazily): the Redis implementation spans
        replicas; without REDIS_CLUSTER configured it is in-process, single replica."""
        if self._msgq is None:
            with self._state_guard:
                if self._msgq is None:
                    self._msgq = (RedisMsgQueue(get_redis()) if settings.redis_url
                                  else MemoryMsgQueue())
        return self._msgq

    def _get_consumer(self) -> SessionConsumer:
        if self._consumer is None:
            # Build the dependencies outside the lock first: msg_queue() and _seg() each
            # take _state_guard themselves, and calling them while this method holds the
            # lock would deadlock on the non-reentrant Lock (we got burned by this).
            mq = self.msg_queue()
            self._seg()   # Make sure session_lock is wired up (it comes from the same place as seg)
            with self._state_guard:
                if self._consumer is None:
                    self._consumer = SessionConsumer(
                        mq, self._session_lock, self.writer_for,
                        max_drain=settings.max_drain_per_cycle,
                        max_retries=settings.max_ingest_retries,
                        renew_interval_s=LOCK_TTL_S / 3,   # Lease renewal interval = 1/3 of the lock TTL
                        after_drain=self.trigger_profile,  # Trigger profile consolidation after a segment closes (without blocking ingest)
                        video_deps=self.video_deps,        # Dependencies for consuming video messages (backends load lazily)
                        task_store=self.task_store)        # Leave a record when a clip fails permanently (queryable, alertable)
        return self._consumer

    # -- Profile consolidation trigger (not queued; one single-flight lock per user) --
    def trigger_profile(self, user_id: str, session_id: str = "", scenario: str = "") -> None:
        """Callback SessionConsumer runs after closing a segment: evaluate the trigger
        conditions and, if they are met, submit to the profile pool.

        It is called on the ingest thread, so it must be fast and must never raise (a
        failed profile only costs freshness; it must not drag ingest consumption down).
        scenario: the business-scenario description of the ingest batch that triggered
        this (a per-request value passed through to consolidation, never persisted).
        """
        try:
            ctx = self.for_user(user_id)
            ps = ProfileStore(self.db, user_id)
            cur = ps.current()
            cells = ctx.cells.cells_after(cur.up_to_cell_id if cur else "")
            if not cells:
                return                                   # No new cells (no segment closed) -> don't trigger
            ep_chars = sum(len(c.episode or "") for c in cells)
            version_count = cur.version if cur else 0    # Versions are never deleted, so the number is the count
            if not should_consolidate(n_new=len(cells), ep_chars=ep_chars,
                                      version_count=version_count,
                                      ep_chars_trigger=settings.profile_ep_chars_trigger):
                return
            # Submit unconditionally: the pool's worker count is the real concurrency
            # cap and anything beyond it queues up (nothing is ever dropped). The
            # per-user single-flight lock deduplicates (a duplicate submission is just a
            # cheap no-op run). We used to drop the work by returning when a gate was
            # full, but the last segment close of a session has no "next time" to heal
            # it — so whenever concurrency exceeded the pool capacity, some users'
            # profiles were reliably skipped. Hence the drop semantics are gone.
            self.profile_exec.submit(self._run_user_profile, user_id, scenario)
        except Exception:   # noqa: BLE001
            logger.exception(f"profile trigger failed user={user_id} (ingest unaffected)")

    def _run_user_profile(self, user_id: str, scenario: str = "") -> None:
        """Runs on the profile pool: take the per-user single-flight lock -> consolidate
        into a new version -> release. If the lock can't be taken, just return (the next
        trigger heals it)."""
        try:
            token = self._session_lock.try_acquire(user_id, "profile") if self._session_lock else None
            if not token:
                return                                   # Another consolidation is running; leave it to that one (idempotent)
            try:
                ctx = self.for_user(user_id)
                run_user_consolidation(self.llm, cells_store=ctx.cells, atoms_store=ctx.atoms,
                                       profile_store=ProfileStore(self.db, user_id),
                                       today=now().date(), scenario=scenario)
            finally:
                self._session_lock.release(user_id, "profile", token)
        except Exception:   # noqa: BLE001
            logger.exception(f"profile consolidation failed user={user_id}")

    def enqueue_message(self, user_id: str, session_id: str, payload: dict,
                        *, kind: str = "ingest") -> tuple[str, int]:
        """Enqueue one session message (ingest/session_end) and return (msg_id, seq)
        immediately. Consumption is driven asynchronously by the dispatcher."""
        return self.msg_queue().enqueue(user_id, session_id, payload, kind=kind)

    def queue_depth(self, user_id: str, session_id: str) -> int:
        return self.msg_queue().depth(user_id, session_id)

    def drain_once(self, user_id: str, session_id: str):
        """Consume one session synchronously (for a single replica, tests, or local
        scripts; does not rely on the background dispatcher)."""
        return self._get_consumer().drain_session(user_id, session_id)

    def start_dispatcher(self) -> None:
        """Start the background dispatcher (called from the FastAPI startup hook; tests
        and scripts don't call it and drive things manually with drain_once)."""
        if self._dispatcher is None:
            self._dispatcher = Dispatcher(
                self.msg_queue(), self._get_consumer(), self.ingest_exec,
                settings.ingest_pool_size,
                tick_s=settings.dispatcher_tick_s, idle_tick_s=settings.dispatcher_idle_tick_s,
                video_pool=self.video_exec, video_cap=settings.video_pool_size)
        self._dispatcher.start()

    def stop_dispatcher(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.stop()

    @staticmethod
    def _warn_redis_env_mismatch() -> None:
        """Announce where cross-replica session state lives, at startup.

        Without Redis the process keeps segment state, locks and the queue in
        memory, which is correct for a single worker and silently wrong for
        several. Saying so once at startup is cheaper than diagnosing it later.
        """
        env = settings.env
        if not settings.redis_url:
            logger.warning(
                "Redis is not configured: running single-process. Segment state, "
                "session locks and the ingest queue are in memory, so multiple "
                "workers would not see each other. Set PERSONOS_REDIS_URL to share them.")
            return
        logger.info(f"Cross-replica session state on Redis "
                    f"(env={env}, key prefix {env}:personos:*)")

    # -- Async plumbing (state lands in the tasks table, queryable from any replica) --
    def submit_task(self, kind: str, user_id: str, session_id: str, fn) -> str:
        """Run fn on the background thread pool, writing its state to the tasks table so
        any replica can query it. Returns the task_id.

        Raises TaskOverloaded when in-flight capacity is full (the API layer turns that
        into a 503) — we reject before creating the row, so no orphan row is left stuck
        in pending.
        """
        if not self._admission.try_enter():
            raise TaskOverloaded(f"background tasks in flight are at capacity (<={_MAX_PENDING}), retry later")
        tid = uuid.uuid4().hex
        self.task_store.create(tid, kind, user_id, session_id)
        self._maybe_sweep()

        def run():
            self.task_store.mark_running(tid)
            try:
                res = fn()
                self.task_store.mark_done(tid, res)
            except Exception as e:  # noqa: BLE001  A background failure is recorded in the tasks table, never crashing the pool
                logger.exception(f"background task failed kind={kind} user={user_id} session={session_id}")
                self.task_store.mark_error(tid, str(e))
            finally:
                self._admission.leave()

        try:
            self._executor.submit(run)
        except Exception:   # Submission failed (e.g. the process is shutting down): the permit must be returned
            self._admission.leave()
            raise
        return tid

    def _maybe_sweep(self) -> None:
        """Low-frequency trigger for the task table sweep (reaping zombies, clearing
        expired rows); a race here is harmless because the operation is idempotent."""
        if time.time() - self._sweep_at < _SWEEP_INTERVAL_S:
            return
        self._sweep_at = time.time()
        try:
            self.task_store.sweep()
        except Exception:  # noqa: BLE001  A failed sweep doesn't affect submission; the next cycle retries
            logger.exception("task table sweep failed")

    def get_task(self, task_id: str) -> dict | None:
        return self.task_store.get(task_id)

    # ══════════════════════════════════════════════════════════════════
    # Public API
    #
    # Everything below is a wrapper. The bodies stay short and free of
    # branching on purpose: if library and server ever disagree about what a
    # call does, it should be impossible to blame this layer.
    # ══════════════════════════════════════════════════════════════════

    def add(self, messages, *, user_id: str = "default", session_id: str,
            now_dt=None, source_extra: dict | None = None,
            task_type: list[str] | None = None, scenario: str = ""):
        """Record a turn, or a batch of turns, into memory.

        ``messages`` is a string, one dict, or a list of dicts::

            m.add("I am allergic to peanuts", user_id="alice", session_id="s1")
            m.add([{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}], ...)

        A dict may carry ``image`` (bytes or a path) alongside its text.

        The batch is atomic with respect to segmentation: it either joins the
        current segment or starts a new one, and is never split down the
        middle. Returns a ``BatchStepResult`` whose ``warnings`` lists anything
        that partially succeeded.
        """
        from .errors import image_not_understood

        self._require_core()
        msgs, videos = _normalise_messages(messages)

        # Video takes a different path: a clip is not a turn to be appended to a
        # segment, it is a recording to be watched. Mixing the two in one call
        # would make the batch non-atomic, so they are rejected together.
        if videos and msgs:
            raise ValueError(
                "a single add() carries either conversation turns or video clips, "
                "not both — send them as separate calls")
        if videos:
            return self._add_videos(videos, user_id=user_id, session_id=session_id)

        warn = []
        if any(m.image for m in msgs) and not getattr(self.mllm, "available", False):
            warn = [image_not_understood()]

        # The same session lock the server takes around a write. Segment state
        # is shared, so two threads adding to one session would interleave it;
        # different sessions stay fully parallel.
        with self._session_scope(user_id, session_id):
            writer = self.writer_for(user_id, session_id)
            out = writer.feed_batch(msgs, now_dt=now_dt, source_extra=source_extra,
                                    task_type=task_type, scenario=scenario)
        out.warnings.extend(warn)
        # A closed segment is new material for the profile. Checking here rather
        # than only at end_session is what makes the profile fill in as someone
        # talks: a long session closes many segments, and waiting for the end
        # would leave the profile stale for the whole conversation.
        if out.closed_cell is not None:
            self.trigger_profile(user_id, session_id, scenario)
        return out

    def _add_videos(self, videos: list, *, user_id: str, session_id: str):
        """Consume clips into the session's identity draft.

        Nothing is committed here. Faces, body shots and voice samples
        accumulate across clips, and who-is-who is only decided at
        end_session() — a person glimpsed in one clip and seen clearly in the
        next has to be resolvable as one person, which cannot be settled while
        clips are still arriving.
        """
        from .errors import no_identity_backend, no_vision
        from .online.video_ingest import process_clip

        if not getattr(self.mllm, "available", False):
            raise no_vision()
        try:
            deps = self.video_deps(user_id)
        except RuntimeError as e:          # backends disabled or not installed
            raise no_identity_backend() from e

        from .online.write_path import BatchStepResult

        out = BatchStepResult(evidence_ids=[], records=[])
        media = self._media()
        for item in videos:
            if isinstance(item, (str, Path)) or getattr(item, "read", None):
                data = Path(item).read_bytes() if not hasattr(item, "read") else item.read()
                stored = media.save_video(data, owner=user_id)
                index = process_clip(deps, session_id=session_id, clip_key=stored.key)
            else:                           # already a URL the model service can fetch
                index = process_clip(deps, session_id=session_id, clip_url=str(item))
            out.evidence_ids.append(f"clip:{index}")
        return out

    def end_session(self, *, user_id: str = "default", session_id: str,
                    task_type: list[str] | None = None, scenario: str = "",
                    update_profile: bool = True):
        """Close the trailing segment so its memories are built now.

        Without this the last segment stays open until enough turns accumulate,
        which for a conversation that simply ended means its memories are never
        written. Returns the cells built by the close.
        """
        self._require_core()
        with self._session_scope(user_id, session_id):
            cells = self.writer_for(user_id, session_id).end_session(
                task_type=task_type, scenario=scenario)
        # If clips were fed into this session, identity is adjudicated now and
        # the attributed dialogue becomes memory through the same build_cell
        # path as text. Sessions with no video simply skip it.
        video_cell = self._finalize_video(user_id, session_id)
        if video_cell is not None:
            cells = [*cells, video_cell]
        # Closing a session always produces new material, so check again here —
        # the final segment of a conversation has no later write to catch it.
        # Threshold-gated and run off the caller's thread, so this does not add
        # latency to end_session; pass update_profile=False to schedule it
        # yourself instead.
        if update_profile:
            self.trigger_profile(user_id, session_id, scenario)
        return cells

    def consolidate_profile(self, *, user_id: str = "default", scenario: str = "",
                            force: bool = True) -> int | None:
        """Refresh the distilled profile from everything written since the last one.

        Returns the new version number, or None when nothing was due. With
        ``force=False`` the threshold decides — which is how end_session calls it.

        Idempotent by construction: it reads "the cells since the last published
        version", so a skipped or failed run is healed by the next one rather
        than leaving a gap.
        """
        from .online.profile_consolidate import run_user_consolidation, should_consolidate
        from .storage.profile_store import ProfileStore

        self._require_core()
        ctx = self.for_user(user_id)
        profiles = ProfileStore(self.db, user_id)
        current = profiles.current()
        if not force:
            pending = ctx.cells.cells_after(current.up_to_cell_id if current else "")
            if not pending:
                return None
            if not should_consolidate(
                    n_new=len(pending),
                    ep_chars=sum(len(c.episode or "") for c in pending),
                    version_count=current.version if current else 0,
                    ep_chars_trigger=settings.profile_ep_chars_trigger):
                return None
        try:
            return run_user_consolidation(
                self.llm, cells_store=ctx.cells, atoms_store=ctx.atoms,
                profile_store=profiles, today=now().date(), scenario=scenario)
        except Exception:  # noqa: BLE001  a profile is an enhancement; never fail a write for it
            logger.exception(f"profile consolidation failed user={user_id}")
            return None

    def _finalize_video(self, user_id: str, session_id: str):
        try:
            from .online.video_ingest import finalize_video
        except ImportError:
            return None
        try:
            deps = self.video_deps(user_id)
        except (RuntimeError, ImportError):
            return None                     # video was never configured for this process
        if not deps.draft.pending_chains(session_id):
            return None                     # this session had no clips
        return finalize_video(deps, session_id=session_id)

    def search(self, query: str, *, user_id: str = "default", session_id: str = "",
               mode: str = "auto", top_k: int = 30, rewrite: bool = True,
               now_dt=None, image: bytes | None = None,
               image_content_type: str = "image/jpeg", scenario: str = "",
               with_profile: bool = True):
        """Answer a question from memory.

        ``mode`` is ``auto`` (fast path, escalating to the deep agent when the
        answer is judged insufficient), ``fast``, or ``deep``.

        Returns a ``RecallOutcome``: the answer plus how it was reached — the
        rewritten query, the retrieved atoms, the ranked materials, the
        adjudication verdicts. Call ``.to_public()`` for a plain dict.
        """
        from .errors import no_deep_track, visual_recall_unavailable
        from .models import now as _now
        from .online.recall_flow import run_recall

        self._require_core()
        warn: list[str] = []
        if mode == "deep" and not _deep_available():
            raise no_deep_track()
        if image is not None and not getattr(self.mllm, "available", False):
            warn.append(visual_recall_unavailable())
            image = None

        ctx = self.for_user(user_id)
        profile_full = profile_traits = ""
        if with_profile:
            profile_full, profile_traits = self._profile_strings(user_id)

        out = run_recall(
            self.llm, self.embedder, ctx.atoms, ctx.cells, ctx.evidence,
            session_id=session_id or "default", query=query, now_dt=now_dt or _now(),
            mode=mode, top_k=top_k, rewrite=rewrite, reranker=self.reranker,
            media_store=self._media(), mllm=self.mllm, image=image,
            image_content_type=image_content_type,
            visual_deps=self.visual_deps(user_id) if image is not None else None,
            profile_full=profile_full, profile_traits=profile_traits, scenario=scenario)
        out.warnings.extend(warn)
        return out

    def profile(self, *, user_id: str = "default") -> dict:
        """The distilled profile as a plain dict.

        Same renderer the HTTP API uses, so both describe a profile the same
        way. Never built yet yields ``{"exists": False, ...}`` rather than None:
        callers read it unconditionally, and an empty profile is a normal
        state.
        """
        from .online.views import profile_view

        return profile_view(ProfileStore(self.db, user_id).current())

    def trace(self, node_id: str, *, user_id: str = "default") -> dict | None:
        """Follow provenance in whichever direction the id implies.

        An atom id yields the forward chain — the evidence it was drawn from.
        An evidence id yields the backward chain — the original turn plus the
        memories that cite it. The kind is discovered by trying, rather than by
        parsing the id, so the caller does not have to know which it holds.

        Returns None when the id belongs to neither.
        """
        from .online.trust import build_trust_chain, trace_evidence

        ctx = self.for_user(user_id)
        media = self._media()
        chain = build_trust_chain([node_id], ctx.atoms, ctx.evidence, media)
        node = chain[0] if chain else None
        if node is not None and not node.get("missing"):
            return {"node": "memory", **node}
        return trace_evidence(node_id, ctx.evidence, ctx.atoms, media)

    def capabilities(self) -> list:
        """What this configuration can do. Same data as ``personos doctor``."""
        from .diagnostics import inspect

        return inspect()

    def reset(self, *, user_id: str) -> None:
        """Delete everything belonging to one user. Irreversible.

        Session state goes too, not just the stored rows: an unclosed segment
        or a half-built identity draft outlives the database wipe otherwise,
        and the next add() would resume into memories that no longer exist.
        """
        from .storage.db.ddl import TABLE_NAMES

        for table in TABLE_NAMES:
            if table == "users":
                continue
            self.db.execute(f"DELETE FROM {table} WHERE user_id=%s", (user_id,))
        for clear, what in ((getattr(self._seg(), "clear_user", None), "segments"),
                            (getattr(self._draft_for(user_id), "clear_user", None), "drafts")):
            if callable(clear):
                try:
                    clear(user_id)
                except Exception:  # noqa: BLE001
                    logger.warning(f"could not clear {what} for {user_id}")
        self.close_user(user_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Release the database and the background pools.

        A library must not leave non-daemon threads behind: an application that
        finishes its work should exit, not hang waiting for a profile job.
        """
        for pool in (getattr(self, "profile_exec", None), getattr(self, "ingest_exec", None),
                     getattr(self, "video_exec", None)):
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
        self.db.close()

    # —— internals ——

    @contextmanager
    def _session_scope(self, user_id: str, session_id: str):
        """Hold the session lock for a write, when one is configured.

        Without Redis the lock is process-local, which is the right scope for a
        single-process library. The point is the same either way: one writer
        per session at a time.
        """
        lock = self.session_lock(user_id, session_id)
        if lock is None:
            yield
            return
        with lock:
            yield

    def _require_core(self) -> None:
        """Required capabilities are checked here, not at construction.

        Constructing Memory() must stay cheap and side-effect free — it is the
        first line of every quickstart, and failing there would make the
        library look broken before the user has asked it to do anything.
        """
        from .errors import no_embedder, no_llm

        if not getattr(self.llm, "available", True):
            raise no_llm()
        if not getattr(self.embedder, "available", True):
            raise no_embedder()

    def _profile_strings(self, user_id: str) -> tuple[str, str]:
        from .online.profile_render import render as render_profile

        try:
            version = ProfileStore(self.db, user_id).current()
            if not version:
                return "", ""
            # render() takes the profile itself, not the version wrapper.
            return (render_profile(version.profile, mode="full"),
                    render_profile(version.profile, mode="traits"))
        except Exception:  # noqa: BLE001  a profile is an enhancement, never a blocker
            logger.exception(f"profile rendering failed user={user_id}; recalling without it")
            return "", ""


def _as_image_bytes(image):
    """Accept raw bytes, a filesystem path, a data URL, or bare base64.

    All four are things people reasonably pass, and the HTTP API takes base64
    for the same field — so treating a base64 string as a filename (which is
    what a naive path check does) fails in a thoroughly confusing way.
    """
    import base64
    import binascii

    if image is None or isinstance(image, (bytes, bytearray)):
        return image
    if isinstance(image, Path):
        return image.read_bytes()
    if isinstance(image, str):
        if image.startswith("data:"):
            return base64.b64decode(image.split(",", 1)[-1])
        candidate = Path(image)
        # A path is short and exists; base64 is long and does not. Check
        # existence first so a filename that happens to be valid base64 still
        # reads as a file.
        if len(image) < 4096 and candidate.exists():
            return candidate.read_bytes()
        try:
            return base64.b64decode(image, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(
                f"image is neither an existing path nor valid base64: {image[:60]!r}") from e
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def _deep_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("langchain_classic") is not None


def _normalise_messages(messages) -> tuple[list, list]:
    """Split the input into conversation turns and video clips.

    Being liberal here is worth it: ``m.add("...")`` is what people try first,
    and OpenAI-shaped dicts are what they paste from an existing app.
    """
    from .online.write_path import FeedMsg

    if isinstance(messages, (str, dict)):
        messages = [messages]
    out, videos = [], []
    for raw in messages:
        if isinstance(raw, str):
            out.append(FeedMsg(speaker="user", text=raw))
            continue
        if isinstance(raw, FeedMsg):
            out.append(raw)
            continue
        video = raw.get("video")
        if video:
            videos.append(video)
            continue
        speaker = raw.get("speaker") or raw.get("role") or "user"
        text = raw.get("text") or raw.get("content") or ""
        image = _as_image_bytes(raw.get("image"))
        out.append(FeedMsg(speaker=speaker, text=text, image=image,
                           image_content_type=raw.get("image_content_type", "image/jpeg")))
    return out, videos



# Deliberately no module-level instance here. Constructing Memory opens a
# database connection, and doing that merely because someone imported the
# module is the same import-time side effect the configuration layer works to
# avoid. The server owns its singleton in server/runtime.py.

# The class was called Runtime before it grew a public API.
Runtime = Memory
