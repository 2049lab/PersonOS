"""The public memory service API (the stable contract external agents call).

Hard design rules:
- Only expose fields that are meaningful as memory; every memory item returned carries
  atom_id + evidence_refs (traceable = the basis for trusting it).
- Never hand out internal quantities (scoring breakdown/score/tier/salience, prompts and
  raw output, deep-track steps, full state).
- The write side is narrow (ingest); the read side is generous (recall / trace).
- **Multi-tenancy**: POST /users/register first to get a token; every memory operation
  afterwards carries the X-User-Token header and can only read and write that user's own
  memory (shared across sessions within a user, physically isolated between users at the
  store layer).
"""

from __future__ import annotations

import base64
import os
import uuid

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from personos import obs
from personos.config import settings
from personos.errors import QueueBusy
from personos.logging_setup import trace
from personos.models import now
from personos.online.profile_render import render as render_profile
from personos.online.recall_flow import PUBLIC_MODES, run_recall
from personos.online.trust import build_trust_chain, trace_evidence
from personos.online.views import profile_view
from personos.session_scope import scoped_session, valid_user_id
from personos.storage.msg_queue import EnqueueBusy
from personos.storage.profile_store import ProfileStore
from server.runtime import UserContext, rt

from .response import EnvelopeRoute
from .signing import verify_signature

# route_class: centrally wraps whatever each handler returns into {code, data, msg}, so
# no endpoint body has to be changed.
# dependencies: AK/SK request signing at the router level, which runs before each
# endpoint's _ctx; it is a no-op under pytest. Every /api/v1 endpoint (including
# register) is signed.
router = APIRouter(prefix="/api/v1", route_class=EnvelopeRoute,
                   dependencies=[Depends(verify_signature)])

# How much "supporting memory" we expose: the matched atoms of the top N units after
# reranking (internally the answer consumes every material unit, externally we return
# only a handful).
_PUBLIC_FAST_MEMORIES = 10
# Cap on one ingest batch: keeps a single task from running too long (each batch costs
# one W1 boundary decision, and a boundary shift additionally costs the W2 weave).
_MAX_INGEST_MESSAGES = 20


def _ctx(x_user_token: str = Header(default="")) -> UserContext:
    """token -> that user's bound context. Missing or unknown -> 401."""
    ctx = rt.ctx_by_token(x_user_token)
    if ctx is None:
        raise HTTPException(status_code=401, detail="X-User-Token is missing or invalid, POST /api/v1/users/register first")
    return ctx


def _sid(caller: str, session_id: str) -> str:
    """Caller scoping plus id hygiene at the entry point (invalid -> 400; the rules live
    in session_scope)."""
    try:
        return scoped_session(caller, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -- Request bodies --
class RegisterBody(BaseModel):
    user_id: str | None = None    # Optional; generated automatically when absent. Letters/digits/underscore/hyphen only, <=128


class IngestMessage(BaseModel):
    speaker: str                  # Who is speaking: the person talking to the assistant is always "user"; in a multi-party conversation use the name from the transcript
    text: str = ""                # Message text (may be empty for an image or video message)
    image_b64: str | None = None  # Optional: a base64 image (with a data:image/...;base64, prefix or as plain base64)
    image_content_type: str = "image/jpeg"
    # A video clip (clips are too large to inline into the queue, so pick one of the two;
    # backward compatible: older callers set neither):
    # - video_url: any downloadable http(s) address, including a pre-signed URL from the
    #   caller's own bucket — **recommended**, we do not require uploading to our bucket;
    #   the consumer downloads it and re-stores it in our object storage (deduplicated by
    #   content address) so the multimodal model can access it via a signed URL and the
    #   evidence trail keeps a copy.
    # - video_oss_key: the shortcut when the clip already lives in our object storage
    #   (skips the download and re-store).
    # One video message = one clip; the consumer runs the identity pipeline clip by clip
    # and does a single final review at session_end before writing to memory.
    video_url: str | None = None
    video_oss_key: str | None = None
    clip_index: int = 0           # The caller's own numbering (metadata only; the real clip index is assigned by a globally monotonic counter on the consumer side, so interleaving can't collide)
    duration_sec: float | None = None


class CallerContext(BaseModel):
    """A container for the caller's business information (one place for all of it): the
    scenario and vocabularies the calling application passes in.

    - scenario: free text describing the business scenario and what memory should focus
      on (e.g. "a diet and health app, interested in the user's food preferences and
      dietary restrictions"). It is injected into the high-leverage LLM stages of writing
      (segmentation / episode / atom), recall (rewriting / answering / deep track) and
      profile consolidation — it only adjusts attention and level of detail; it never
      changes facts, invents anything, or drops anything. Empty = the default path,
      byte for byte.
    - task_type: the candidate vocabulary for episode classification (moved here from the
      old top-level IngestBody field); it is ignored if a recall request sets it.
    """
    scenario: str = ""
    task_type: list[str] | None = None


class IngestBody(BaseModel):
    caller: str = ""              # The calling party's identifier (different callers may use the same session_id, so we prefix it at the entry point to isolate them)
    session_id: str               # Letters/digits/underscore/hyphen/dot only, <=95 (no colon: it's the Redis key separator)
    messages: list[IngestMessage] # A batch of messages (atomic): the whole batch joins the current segment or the whole batch starts a new one; a batch is never split
    context: CallerContext | None = None  # Optional: the caller's scenario + classification vocabulary (see CallerContext)
    sync: bool = False            # False (default): return 202 as soon as the batch is queued. True: hold the request until this session's queue has drained (the batch's memories exist by the time we answer)


class RecallBody(BaseModel):
    caller: str = ""              # Same as ingest: it must match what was used at write time to reach the same session's context
    session_id: str
    query: str
    # Optional: send one image along with the question (multimodal recall supports text +
    # image; video is not supported yet). With an image we run one extra visual
    # understanding rewrite before R0 — it writes the people and scene from the image
    # into the query, which is the only way the text-only pipeline downstream can use the
    # visual information.
    image_b64: str | None = None          # With a data:image/...;base64, prefix or as plain base64
    image_content_type: str = "image/jpeg"
    mode: str = "auto"           # auto (fast path, escalating to the deep track when adjudication fails) | fast (fast path only) | deep (straight to the deep track)
    top_k: int = 30              # Cap on the R1 atom pool (how many survive the RRF fusion of the two retrieval paths; material units are derived from it)
    context: CallerContext | None = None  # Optional: the caller's scenario (recall uses scenario only; task_type is ignored)


class SessionEndBody(BaseModel):
    caller: str = ""              # Same as ingest
    session_id: str
    context: CallerContext | None = None  # Optional: the caller's scenario + the classification vocabulary for the trailing segment (see CallerContext)
    sync: bool = False            # True: hold the request until the wrap-up has actually run (the trailing segment is closed into memories by the time we answer)


_SYNC_WAIT_S = 600.0    # Upper bound for sync=True holds (same bound the library's flush uses)


def _wait_drained(user_id: str, session_id: str) -> bool:
    """Block until the session's queue is fully consumed (sync=True requests).

    Any pod may be the one consuming, so this just watches the shared queue
    state rather than driving consumption locally. False means the backlog did
    not drain in time — the messages are not lost, poll GET /queue/status.
    """
    try:
        rt.flush(user_id=user_id, session_id=session_id, timeout_s=_SYNC_WAIT_S)
        return True
    except TimeoutError:
        return False


@router.get("/health")
def health():
    atoms = rt.db.fetch_one("SELECT COUNT(*) AS n FROM atoms")["n"]
    ev = rt.db.fetch_one("SELECT COUNT(*) AS n FROM evidence")["n"]
    cells = rt.db.fetch_one("SELECT COUNT(*) AS n FROM memcells")["n"]
    return {"status": "ok", "users": rt.users.count(), "atoms": atoms,
            "cells": cells, "evidence": ev,
            "db": settings.db_url.rsplit("@", 1)[-1] if settings.db_url else "sqlite"}


@router.post("/users/register", status_code=201)
def register(body: RegisterBody):
    """Register a user and issue a unique token. The token is returned in full exactly
    once, in this response, so the caller must store it persistently."""
    try:
        uid = valid_user_id(body.user_id)          # Invalid character set -> 400 (as opposed to 409 for already exists)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    try:
        return rt.users.register(uid)
    except ValueError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})


# Size cap for images going into the queue: an image rides along with its message as
# base64 in the Redis queue (parked there until consumption dequeues it), so anything
# larger is rejected.
_MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _decode_image_b64(value: str | None, where: str) -> tuple[bytes | None, str | None]:
    """Validate and normalize an image parameter, returning (bytes, the plain base64 with
    any prefix stripped); an empty value gives (None, None).

    /ingest and /recall share this one implementation: writing it twice would inevitably
    drift (the size cap, the data: prefix, the validate flag), and "which images you may
    send" is part of the public contract, so the two must agree.
    """
    if not value:
        return None, None
    raw = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{where} is not valid base64")
    if len(decoded) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"{where} image is too large (>{_MAX_IMAGE_BYTES // 1024 // 1024}MB)")
    return decoded, raw


# Size and duration caps for a video clip (the same numbers the consumer side,
# video_ingest, uses)
_MAX_CLIP_BYTES = int(os.environ.get("PERSONOS_VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))
_MAX_CLIP_DURATION_S = float(os.environ.get("PERSONOS_VIDEO_MAX_DURATION_S", "150"))
# The cap on what the upstream can fetch: for the script stage we hand the multimodal
# model provider a **signed URL** and it downloads the whole video itself.
# It has its own download window — measured, 106MB (2 min @ 7.3Mbps) always produced
# `Download multimodal file timed out`, while 55MB was stable over a long period. So the
# real constraint is not whether we can upload it but **whether the upstream can pull
# it**, which is far stricter than _MAX_CLIP_BYTES (our own storage limit) and therefore
# needs its own check. We use 64MB: safely above the 55MB that is proven to work and far
# below the 106MB that is proven to fail.
# Anything larger gets a 400 at the entry point, telling the caller to lower the bitrate
# or cut shorter clips — better than accepting a 202 and failing silently minutes later.
_MAX_CLIP_UPSTREAM_BYTES = int(os.environ.get("PERSONOS_VIDEO_UPSTREAM_MAX_BYTES",
                                              str(64 * 1024 * 1024)))


def _too_big_for_upstream(total: int) -> str:
    """Size above what the upstream can fetch -> the error reason returned to the caller
    (with actionable advice); an empty string means it passed."""
    if not total or total <= _MAX_CLIP_UPSTREAM_BYTES:
        return ""
    return (f"size {total / 1024 / 1024:.1f}MB exceeds the {_MAX_CLIP_UPSTREAM_BYTES / 1024 / 1024:.0f}MB "
            f"limit (the upstream model service would time out fetching it) "
            f"-- please lower the bitrate or cut shorter clips")


# The pre-check on an external URL must time out quickly, so it doesn't slow the entry
# point down
_URL_PRECHECK_TIMEOUT_S = float(os.environ.get("PERSONOS_VIDEO_URL_PRECHECK_TIMEOUT_S", "5"))


def _precheck_video_url(url: str) -> str:
    """A quick pre-check on an external URL (**a GET with a Range header that fetches one
    byte**): an empty string means it passed, otherwise the error reason returned to the
    caller.

    Why do it at the entry point: consumption is asynchronous, so the caller is gone as
    soon as it has its 202; if a mistyped or expired address were only discovered on the
    consumer side, the caller would have no idea (their only recourse would be digging
    through the failure records afterwards). Spending a few hundred milliseconds probing
    at the entry point reports the common mistakes on the spot.

    Why not HEAD (we got burned): **a pre-signed URL's signature is bound to the HTTP
    method**, so when it was signed for GET, HEAD always returns 403 — using HEAD would
    wrongly reject the most common kind of valid external link, a pre-signed address from
    the caller's own bucket. A GET with a Range header matches the method and transfers
    just one byte, so it costs about the same as HEAD.
    The overall stance stays "let through rather than reject": we only block a clear
    4xx/5xx or a declared size over the limit, and any exception or flakiness is let
    through for the real download on the consumer side to judge.
    """
    try:
        import httpx
        r = httpx.get(url, headers={"Range": "bytes=0-0"},
                      timeout=_URL_PRECHECK_TIMEOUT_S, follow_redirects=True)
        if r.status_code >= 400:
            return f"is not reachable (HTTP {r.status_code})"
        # A 206 carries Content-Range: bytes 0-0/<total>; a 200 (the peer ignored Range)
        # means Content-Length is the full length
        total = 0
        cr = r.headers.get("content-range") or ""
        if "/" in cr:
            total = int(cr.rsplit("/", 1)[-1] or 0)
        elif r.status_code == 200:
            total = int(r.headers.get("content-length") or 0)
        if total and total > _MAX_CLIP_BYTES:
            return f"size {total} exceeds the limit of {_MAX_CLIP_BYTES} bytes"
        err = _too_big_for_upstream(total)
        if err:
            return err
    except Exception:  # noqa: BLE001  Flakiness, or a peer that doesn't support Range -> let it through and leave the verdict to the real download on the consumer side
        return ""
    return ""


@router.post("/ingest", status_code=202)
def ingest(body: IngestBody, ctx: UserContext = Depends(_ctx)):
    """Feed in a batch of messages (enqueue, fire-and-forget): the whole batch goes into
    **that session's durable ordered queue**, and the dispatcher consumes it FIFO
    asynchronously.

    **The batch is atomic**: this batch plays the role a single utterance used to — at
    consumption time it either joins the current topic segment as a whole or starts a new
    segment as a whole, and a batch (a Q&A pair, say) is never split across two cells.
    Batch size is 1-20; anything else -> 400.

    Callers do not need to "wait for the previous batch to finish before sending the next
    one" — ordering, no loss and no duplication are guaranteed by the consumption system
    (see ingest_worker): messages of one session queue up by seq, a single-flight
    consumer processes them in order, dequeuing is reliable (a message is only acked
    after it was consumed successfully) and the cursor dedups. Consumption can lag (when
    there is a backlog), but every message is guaranteed to be consumed correctly exactly
    once. When caller is non-empty the session_id is rewritten to
    f"{caller}:{session_id}", so identically named sessions from different callers are
    isolated automatically. Returns msg_id + seq (it no longer returns per-batch segment
    closing results; check progress with GET /api/v1/queue/status).
    """
    if not body.messages or len(body.messages) > _MAX_INGEST_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"messages must contain 1-{_MAX_INGEST_MESSAGES} items, got {len(body.messages)}")
    sid = _sid(body.caller, body.session_id)
    # Validate and normalize images right at the entry point (bad base64 -> 400, too
    # large -> 413); the plain base64 goes into the message payload and the consumer
    # decodes it to look at the image.
    # Video batches and text/image batches are never mixed: a batch is either all video
    # (kind=video, each clip goes through the identity pipeline) or all text/image
    # (kind=ingest, which goes through feed_batch). Video clips are too large to inline —
    # the message carries only a URL or storage key and the consumer fetches it
    # asynchronously.
    def _is_vid(m) -> bool:
        return bool(m.video_url or m.video_oss_key)

    is_video = any(_is_vid(m) for m in body.messages)
    if is_video and any(not _is_vid(m) for m in body.messages):
        raise HTTPException(status_code=400, detail="one batch cannot mix video with text/image_b64, send them in separate batches")
    msgs: list[dict] = []
    for i, m in enumerate(body.messages):
        if not m.speaker.strip():
            raise HTTPException(status_code=400, detail=f"messages[{i}].speaker must not be empty")
        if _is_vid(m):
            # A single message must not carry both video and text/image — otherwise one
            # of the two would be dropped silently, and an explicit rejection is better.
            if m.text.strip() or m.image_b64:
                raise HTTPException(
                    status_code=400,
                    detail=f"messages[{i}] cannot carry video and text/image_b64 at once, split it into two messages sent in separate batches")
            url = (m.video_url or "").strip()
            if url and not url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400,
                                    detail=f"messages[{i}].video_url must be an http(s) address")
            # If the duration the caller declared already exceeds the limit, reject at the
            # entry point (a cheap check; the consumer verifies the real duration again)
            if m.duration_sec and m.duration_sec > _MAX_CLIP_DURATION_S:
                raise HTTPException(
                    status_code=400,
                    detail=f"messages[{i}] video duration {m.duration_sec}s exceeds the "
                           f"{_MAX_CLIP_DURATION_S}s limit, please cut shorter clips")
            # Reachability pre-check on the external link (a GET with Range and a short
            # timeout): a dead address or an oversized clip errors out immediately,
            # instead of the caller getting a 202 and then failing quietly
            if url:
                err = _precheck_video_url(url)
                if err:
                    raise HTTPException(status_code=400,
                                        detail=f"messages[{i}].video_url {err}")
            else:
                # Our own key: ask object storage for the size directly (one head
                # request, no download). This path used to have no validation at all — a
                # caller could push an oversized clip in and we'd only find out minutes
                # later when the screenplay model failed.
                # If the size can't be read (missing key, storage flaking) we always let
                # it through for the consumer to judge (accept rather than reject).
                try:
                    size = rt._media().object_size((m.video_oss_key or "").strip())
                except Exception:  # noqa: BLE001
                    size = 0
                err = _too_big_for_upstream(size)
                if err:
                    raise HTTPException(status_code=400,
                                        detail=f"messages[{i}].video_oss_key {err}")
            msgs.append({"speaker": m.speaker.strip(), "video_url": url or None,
                         "video_oss_key": (m.video_oss_key or "").strip() or None,
                         "clip_index": m.clip_index, "duration_sec": m.duration_sec})
            continue
        if not m.text.strip() and not m.image_b64:
            raise HTTPException(
                status_code=400,
                detail=f"messages[{i}] must have at least one of text / image_b64 / video_url (or video_oss_key)")
        image_b64 = _decode_image_b64(m.image_b64, f"messages[{i}].image_b64")[1]
        msgs.append({"speaker": m.speaker.strip(), "text": m.text,
                     "image_b64": image_b64, "image_content_type": m.image_content_type})
    # Backpressure: reject once one session's queue backlog is over the limit (so a
    # flooding session can't blow the queue up); the caller should slow down and retry
    if rt.queue_depth(ctx.user_id, sid) >= settings.max_queue_depth:
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "too many messages backed up on this session, slow down and retry"})
    tid = uuid.uuid4().hex   # End-to-end trace_id: passed through the queue so consumption uses the same id for the langfuse trace and the log tracing id
    cc = body.context or CallerContext()
    try:
        msg_id, seq = rt.enqueue_message(
            ctx.user_id, sid,
            {"messages": msgs, "task_type": cc.task_type, "scenario": cc.scenario,
             "trace_id": tid}, kind="video" if is_video else "ingest")
    except EnqueueBusy:   # Severe contention enqueuing on this session timed out: reject and let the caller retry (never force it through and corrupt the order or lose a message)
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "enqueuing on this session is busy, retry later"})
    except QueueBusy:     # Global backlog over the limit (50 x pod count): the fleet is saturated
        return JSONResponse(status_code=503, headers={"Retry-After": "5"},
                            content={"error": "too many requests, retry later"})
    out = {"accepted": True, "msg_id": msg_id, "seq": seq,
           "queue_depth": rt.queue_depth(ctx.user_id, sid), "trace_id": tid}
    if body.sync:
        out["consumed"] = _wait_drained(ctx.user_id, sid)
    return out


@router.post("/session/end", status_code=202)
def session_end(body: SessionEndBody, ctx: UserContext = Depends(_ctx)):
    """End of session (enqueues a wrap-up task): it queues behind the session's existing
    messages, and when consumption reaches it, it force-closes the still-open trailing
    segment.

    Because it travels the same ordered queue, the wrap-up is guaranteed to happen after
    every earlier message has been consumed (it can never run before an unconsumed
    ingest). caller means the same thing as in ingest. Returns msg_id + seq, echoing back
    the original session_id (the caller's own naming).
    """
    sid = _sid(body.caller, body.session_id)
    tid = uuid.uuid4().hex
    cc = body.context or CallerContext()
    try:
        msg_id, seq = rt.enqueue_message(
            ctx.user_id, sid,
            {"task_type": cc.task_type, "scenario": cc.scenario, "trace_id": tid},
            kind="session_end")
    except EnqueueBusy:   # Same contention case as ingest: reject and let the caller retry
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "session busy, retry later"})
    except QueueBusy:     # Global backlog over the limit (50 x pod count): the fleet is saturated
        return JSONResponse(status_code=503, headers={"Retry-After": "5"},
                            content={"error": "too many requests, retry later"})
    out = {"accepted": True, "msg_id": msg_id, "seq": seq,
           "session_id": body.session_id, "trace_id": tid}
    if body.sync:
        out["consumed"] = _wait_drained(ctx.user_id, sid)
    return out


@router.get("/queue/status")
def queue_status(session_id: str, caller: str = "", ctx: UserContext = Depends(_ctx)):
    """Consumption progress of a session: depth = the backlog still to consume, cursor =
    the largest seq consumed so far (so a caller can tell whether its own seq has been
    consumed)."""
    sid = _sid(caller, session_id)
    mq = rt.msg_queue()
    return {"session_id": session_id, "depth": mq.depth(ctx.user_id, sid),
            "cursor": mq.cursor_get(ctx.user_id, sid)}


@router.get("/tasks/{task_id}")
def task_status(task_id: str, ctx: UserContext = Depends(_ctx)):
    """Query the status of an async task (only tasks belonging to your own user):
    pending | running | done | error."""
    t = rt.get_task(task_id)
    if t is None or t.get("user_id") != ctx.user_id:
        return JSONResponse(status_code=404, content={"error": "unknown task", "task_id": task_id})
    return t


@router.post("/recall")
def recall(body: RecallBody, ctx: UserContext = Depends(_ctx)):
    """Recall: either the whole fast path (R0 -> R1 -> R2 -> R5 draft -> R3' adjudication,
    escalating to the deep track automatically when adjudication fails) or straight to
    the deep track; returns the answer plus the memories it rests on.

    answer is a plain statement of fact (third person, neutral, with absolute dates) that
    the caller turns into its own dialogue.
    verdict is the adjudication grade (ok = the draft passed / answer_defect = still
    defective after being rewritten with the critique / insufficient_material = not
    enough material), and critique is the adjudicator's criticism — the caller can use it
    to ask a follow-up or rephrase; retried says whether the answer was rewritten.
    Every memory inlines its evidence (the original text plus the Q&A), so it carries its
    own provenance.

    You may send one image along with the question (image_b64): before R0 we run one
    visual understanding rewrite that writes the people in the image (matched against the
    people in memory) and the scene into the query, which is what lets the text-only
    pipeline downstream use the visual information. Without an image the pipeline is
    unchanged, byte for byte.
    """
    if body.mode not in PUBLIC_MODES:
        return JSONResponse(status_code=400,
                            content={"error": f"mode only supports {list(PUBLIC_MODES)}", "got": body.mode})
    img, _ = _decode_image_b64(body.image_b64, "image_b64")   # Bad base64 -> 400, too large -> 413
    if img is not None:
        # The query image gets an **extra** format check: it is query input, so if we
        # can't recognize the format there is no visual understanding to speak of. Rather
        # than returning a 200 plus an answer that quietly ignored the image (which the
        # caller has no way to notice), we tell them right away that the image is bad.
        # /ingest images don't get this check: there the image is **content**, and if we
        # can't look at it the message degrades to text and still has value.
        from personos.storage.media._common import _detect_image_type
        if _detect_image_type(img[:32]) is None:
            raise HTTPException(status_code=400,
                                detail="image_b64 is not a recognizable image (supported: jpeg/png/webp/gif/heic)")
    sid = _sid(body.caller, body.session_id)
    # A snapshot read: it neither queues nor waits for unconsumed ingests (it answers
    # from whatever is committed in the DB right now). Only recall_gate limits it: the
    # sync endpoint is already running on an anyio thread, so we run run_recall on this
    # very thread (no submitting to another pool and blocking on it, which would occupy
    # two threads). The gate's cap is below the anyio thread pool size, so a burst of
    # recalls can't take every thread and drag the probes down, and it is naturally
    # isolated from the ingest consumption pool.
    if not rt.recall_gate.try_enter():
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "recall concurrency is at capacity, retry later"})
    # Profile injection (an empty profile gives an empty string, so behavior is unchanged
    # byte for byte): full goes to R0 and the deep track, traits goes to R5
    _pv = ProfileStore(rt.db, ctx.user_id).current()
    p_full = render_profile(_pv.profile, mode="full") if _pv else ""
    p_traits = render_profile(_pv.profile, mode="traits") if _pv else ""
    tid = uuid.uuid4().hex   # End-to-end trace_id: the langfuse trace, the log tracing id and the response body all use the same id
    try:
        with trace(tid), \
                obs.root_span("recall", user_id=ctx.user_id, session_id=sid,
                              input=body.query, trace_id=tid) as _sp:
            if _sp is not None:
                logger.info(f"recall langfuse trace_id={obs.current_trace_id()}")
            o = run_recall(rt.llm, rt.embedder, ctx.atoms, ctx.cells, ctx.evidence,
                           session_id=sid, query=body.query, now_dt=now(),
                           mode=body.mode, top_k=body.top_k, reranker=rt.reranker,
                           media_store=rt._media(), mllm=rt.mllm,
                           image=img, image_content_type=body.image_content_type,
                           visual_deps=rt.visual_deps(ctx.user_id) if img else None,
                           profile_full=p_full, profile_traits=p_traits,
                           scenario=(body.context.scenario if body.context else ""))
    except httpx.HTTPStatusError as e:
        # An upstream 429: the client only does one or two short backoff retries before
        # raising, and we pass it straight through — the caller should back off and retry
        # per rate-limit semantics, rather than having the request hang in a deep backoff
        # on our side (under concurrency that drags workers down = retry hell)
        if getattr(e.response, "status_code", None) == 429:
            logger.warning(f"recall upstream rate limited user={ctx.user_id} session={sid} q={body.query!r}")
            return JSONResponse(status_code=429, headers={"Retry-After": "5"},
                                content={"error": "the upstream model service is rate limiting, retry later", "detail": str(e)})
        logger.exception(f"recall failed user={ctx.user_id} session={sid} q={body.query!r}")
        return JSONResponse(status_code=502,
                            content={"error": "the memory service is temporarily unavailable: a call to the upstream model/embedding service failed and this recall could not complete",
                                     "detail": str(e)})
    except Exception as e:   # noqa: BLE001  An upstream (model/embedding) failure: each stage degrades internally, so anything reaching here means retrieval itself is unavailable
        logger.exception(f"recall failed user={ctx.user_id} session={sid} q={body.query!r}")
        return JSONResponse(status_code=502,
                            content={"error": "the memory service is temporarily unavailable: a call to the upstream model/embedding service failed and this recall could not complete",
                                     "detail": str(e)})
    finally:
        rt.recall_gate.leave()          # The permit must always be returned (success, exception and early return all pass through here)
    # The supporting memories come from the same place as the final answer: on the fast
    # path they are the matched atoms of the top units after R2 reranking; for a
    # deep-track answer (mode=deep directly, or after an auto escalation overrode it)
    # they are all atoms of the cells the final answer cited (cited = supporting).
    # When adjudication rules the material insufficient we return none of them: a
    # half-relevant item would be misread by the caller as "the basis of the answer", so
    # with no answer we honestly leave it empty and let the objective account inside
    # answer (what was searched, the conclusion, the reason) speak. A negative deep-track
    # answer may also cite the cells it "looked at" (after an auto escalation the review
    # is still the fast path's verdict) — the same final-adjudication gate applies.
    # One renderer, shared with the library: Memory.search(...).to_public() and
    # this endpoint produce the same shape by construction, so the two cannot
    # drift into describing the same recall differently.
    payload = o.to_public(atoms=ctx.atoms, evidence=ctx.evidence,
                          media_store=rt._media(), max_memories=_PUBLIC_FAST_MEMORIES)
    payload["trace_id"] = tid
    return payload


@router.get("/profile")
def get_profile(ctx: UserContext = Depends(_ctx)):
    """Get this user's current profile (structured, not consolidated; ownership comes
    from token -> user)."""
    return profile_view(ProfileStore(rt.db, ctx.user_id).current())


def _episode_vo(cell) -> dict:
    """memcell -> the public episode view object, at segment granularity: id, session,
    start/end, topic, narrative, classification. The payload, atoms and vectors are never
    exposed."""
    return {
        "memcell_id": cell.id,
        "session_id": cell.session_id,
        "start_time": cell.t_start.isoformat() if cell.t_start else None,
        "end_time": cell.t_end.isoformat() if cell.t_end else None,
        "topic": cell.topic,
        "episode": cell.episode,
        "episode_type": cell.episode_type,
    }


@router.get("/episodes")
def list_episodes(ctx: UserContext = Depends(_ctx),
                  episode_type: str | None = Query(default=None, description="filter by classification; omit for all"),
                  start: str | None = Query(default=None, description="start time (ISO, inclusive); filters on the segment's t_start"),
                  end: str | None = Query(default=None, description="end time (ISO, inclusive)"),
                  page: int = Query(default=1, ge=1),
                  page_size: int = Query(default=20, ge=1, le=100)):
    """Page through this user's episodes (segments) by episode_type and time range,
    ordered by t_start descending (most recent first).

    Ownership comes from token -> user; every filter (episode_type/start/end) is
    optional, and page/page_size have defaults (with a cap of 100).
    Returns a list of segment-level view objects plus pagination metadata
    (total/page/page_size).
    """
    offset = (page - 1) * page_size
    cells = ctx.cells.list_by_type(episode_type=episode_type, start=start, end=end,
                                   limit=page_size, offset=offset)
    total = ctx.cells.count_by_type(episode_type=episode_type, start=start, end=end)
    return {"items": [_episode_vo(c) for c in cells],
            "total": total, "page": page, "page_size": page_size}


@router.get("/trace/{node_id}")
def trace_node(node_id: str, ctx: UserContext = Depends(_ctx)):
    """Trace provenance by id (only within this user's memory); both memories and
    evidence are supported and dispatched automatically by the id:

    - a memory atom_id -> the forward chain: the memory + its attribution and knowledge
      state + a drill-down to the original evidence (including the Q&A at the time),
      `node="memory"`.
    - an evidence_id -> the backward chain: the original evidence text + the Q&A of the
      same turn + which memories cite it (cited_by), `node="evidence"`.
    If neither lookup finds it -> 404.
    """
    media = rt._media()
    chain = build_trust_chain([node_id], ctx.atoms, ctx.evidence, media)   # Try it as a memory first
    node = chain[0] if chain else None
    if node is not None and not node.get("missing"):
        return {"node": "memory", **node}
    ev_node = trace_evidence(node_id, ctx.evidence, ctx.atoms, media)      # Then try it as evidence
    if ev_node is not None:
        return ev_node
    return JSONResponse(status_code=404, content={"error": "not found", "id": node_id})
