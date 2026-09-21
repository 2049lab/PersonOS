"""Consumer-side orchestration of video clips (the production path): update the Redis draft clip by
clip, then run the final review at the end of the session and land it in memory.

Extracted into library functions from scripts/video/verify_pipeline._process so the consumer
(ingest_worker) and the script share one copy. Two entry points:
- process_clip: one video message (the clip's OSS key) -> download -> screenplay -> harvest ->
  observe -> (for refreshed chains) batched adjudication + collision repair -> stage the material,
  lines and roster. The clip index uses a session-global monotonic counter (never trust the caller's
  numbering, which prevents collisions when batches interleave).
- finalize_video: end of session -> commit_session (the two-phase final identity review) +
  flush_session_to_memory (line ownership -> memory).

The accumulated draft state lives in RedisDraftStore (across messages and across replicas); it is
independent of text segments, so any interleaving order is still correct.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from personos.identity.chains import ChainBook, resolve_chain_collisions
from personos.identity.cloud import CloudEngine
from personos.identity.commit import commit_session
from personos.identity.draft import DraftStore
from personos.identity.harvest import harvest_clip
from personos.identity.recognize import build_arbitration_prompt, parse_verdicts
from personos.identity import repair
from personos.identity.registry import AnchorRegistry
from personos.identity.screenplay import (WEARER_CAST_ID, build_clip_prompt, parse_clip_output,
                                          rewrite_ids)
from personos.identity.store import CharacterStore
from personos.identity.types import CastEvidence
from personos.online.video_memory import CellBuild, flush_session_to_memory

DEFAULT_SCENE = "first-person robot home-assistant"


@dataclass
class VideoDeps:
    """The full dependency set for the video consumer (assembled by runtime.video_deps). `backends`
    is a process singleton (heavy models); everything else is per-user (store / cloud / draft / the
    four memory stores)."""

    store: CharacterStore          # identity persistence layer (MySQL)
    cloud: CloudEngine             # the probabilistic identity cloud
    draft: DraftStore              # session draft (Redis, accumulated across clips)
    backends: dict[str, Any]       # {mm_runner, face_detector, voiceprint} -- process singletons
    media_store: Any               # OSS
    llm: Any                       # chat model, for building the memory cell
    embedder: Any                  # embedding model; a separate object since the
                                   # provider split — one endpoint often serves both,
                                   # but they are not the same client.
    evidence: Any
    cells: Any
    atoms: Any
    chains: Any                    # atom_chain store


def _duration(path: str) -> Optional[float]:
    import av
    c = av.open(path)
    try:
        return float(c.duration) / 1_000_000 if c.duration else None
    finally:
        c.close()


def _merge_by_cast(ev_by_local: dict, cast_map: dict) -> dict[str, CastEvidence]:
    out: dict[str, CastEvidence] = {}
    for local, ev in ev_by_local.items():
        cast = cast_map.get(local)
        if not cast:
            continue
        tgt = out.setdefault(cast, CastEvidence(cast_id=cast))
        tgt.faces.extend(ev.faces)
        tgt.voices.extend(ev.voices)
    return out


@contextmanager
def _timed(costs: dict[str, float], stage: str):
    """Record one stage's wall-clock cost into costs[stage] (accumulating: repeated calls with the
    same stage name are summed).

    Why measure it: video consumption is minutes-long heavy work, and once it is in production the
    question "which stage is slow?" has to be answerable straight from the logs — download
    (environment-dependent), the screenplay MLLM (scales with frame count), harvest (scales with the
    number of nominations) and adjudication (scales with the number of people) each need completely
    different optimizations.
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        costs[stage] = costs.get(stage, 0.0) + (time.monotonic() - t0)


def _fmt(costs: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.1f}s" for k, v in costs.items())


CLIP_FETCH_TIMEOUT_S = 120.0
# Clip size / duration caps: the size cap reuses media_store's convention; the duration cap is new
# (video is an order of magnitude more expensive than text, and an over-long clip drags one message's
# processing time past the lock TTL as well as blowing out the MLLM context).
MAX_CLIP_BYTES = int(os.environ.get("PERSONOS_VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))
# The cap is 150 rather than 120: a caller cutting "2-minute clips" actually produces slightly-over
# pieces like 120.1s, and stopping exactly at 120 would reject every normal call. A 25% margin both
# accommodates a nominal 2min and still blocks clips that are clearly too long.
MAX_CLIP_DURATION_S = float(os.environ.get("PERSONOS_VIDEO_MAX_DURATION_S", "150"))


class ClipRejected(ValueError):
    """The clip is permanently unusable (dead URL, too large, too long, bad format) — retrying is
    pointless, so it should be recorded and skipped rather than silently dropped."""


def _reject_if_too_long(dur: Optional[float]) -> None:
    """An over-long clip: its processing time would punch through the session lock TTL and blow out
    the MLLM context."""
    if dur and dur > MAX_CLIP_DURATION_S:
        raise ClipRejected(f"video duration {dur:.1f}s exceeds the {MAX_CLIP_DURATION_S}s limit (please cut shorter clips)")


def _materialize_clip(media_store: Any, *, clip_key: str, clip_url: str, owner: str,
                      ) -> tuple[str, str]:
    """Materialize the clip into a **local temp file** and return (tmp_path, clip_key).

    Memory discipline (critical): a clip is routinely tens of MB while the MLLM call that follows
    takes 2-3 minutes — the bytes must never ride along in memory for the whole flow. The rule here
    is "release as soon as it is on disk":
    - our own key: the bytes from read_bytes are deleted immediately after being written to disk (so
      they never reach the MLLM stage);
    - an external URL: **streamed straight to disk**, never assembled in memory (the old
      b"".join(chunks) had a 2x instantaneous peak); it is read back in one go only to upload to our
      OSS (briefly), then released.
    """
    fd = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = fd.name
    try:
        if clip_key:
            try:
                data = media_store.read_bytes(clip_key)
            except Exception as e:  # noqa: BLE001
                raise ClipRejected(f"failed to read the clip from our object store key={clip_key[-24:]}: {e}") from e
            if len(data) > MAX_CLIP_BYTES:
                raise ClipRejected(f"video exceeds the {MAX_CLIP_BYTES} byte limit")
            fd.write(data)
            del data                        # release as soon as it is on disk; do not carry it into the MLLM stage
            fd.close()
            return tmp_path, clip_key
        if not clip_url:
            raise ClipRejected("process_clip needs either clip_key or clip_url")
        _stream_to_file(clip_url, fd)       # stream to disk while enforcing the size cap as it goes
        fd.close()
        data = open(tmp_path, "rb").read()  # read once, only to compute the sha and upload, then release
        try:
            key = media_store.save_video(data, owner=owner).key
        except Exception as e:  # noqa: BLE001
            raise ClipRejected(f"failed to re-store the video: {e}") from e
        finally:
            del data
        return tmp_path, key
    except Exception:
        fd.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)             # leave no junk file behind even on failure
        raise


def _stream_to_file(url: str, fd) -> None:
    """Stream the download into an already-open file handle, enforcing the size cap as it goes (so an
    oversized file is never read into memory)."""
    import httpx
    try:
        with httpx.stream("GET", url, timeout=CLIP_FETCH_TIMEOUT_S, follow_redirects=True) as r:
            if r.status_code >= 400:
                raise ClipRejected(f"video address not reachable (HTTP {r.status_code}): {url[:80]}")
            declared = int(r.headers.get("content-length") or 0)
            if declared and declared > MAX_CLIP_BYTES:
                raise ClipRejected(f"video exceeds the {MAX_CLIP_BYTES} byte limit (declared {declared})")
            total = 0
            for chunk in r.iter_bytes():
                total += len(chunk)
                if total > MAX_CLIP_BYTES:
                    raise ClipRejected(f"video exceeds the {MAX_CLIP_BYTES} byte limit (exceeded mid-download)")
                fd.write(chunk)
    except ClipRejected:
        raise
    except Exception as e:  # noqa: BLE001  DNS / connection / timeout / certificate -> the URL is unusable (a permanent failure)
        raise ClipRejected(f"video address download failed ({type(e).__name__}: {e}): {url[:80]}") from e


# There used to be a second concurrency gate here (BoundedSemaphore(4)), which guarded against
# blowing up memory/GPU before there was a dedicated video thread pool. Once video_pool_size existed,
# the two governed the same thing, and **the stricter one always wins** — give the pool 50 threads
# and 46 of them sit waiting at the gate, each still holding its own session lock (mutual exclusion
# across replicas), occupying a slot while producing nothing.
# So it was removed: video concurrency is **decided by the single PERSONOS_VIDEO_POOL knob** (see
# config.video_pool_size).

# Frame sampling rate for the screenplay MLLM: **the number one variable in per-clip cost** (frames
# sent to the model = duration x fps, and cost is close to linear in that).
# The default 1.0 (one frame per second) halves the screenplay stage compared to 2.0, and is the only
# effective lever for pushing one clip down to near real-time.
# If you need finer localization of nomination moments (fast action, many short shots) you can dial
# it back to 2.0, at the cost of doubling the time.
VIDEO_FPS = float(os.environ.get("PERSONOS_VIDEO_FPS", "1.0"))


def process_clip(deps: VideoDeps, *, session_id: str, clip_key: str = "", clip_url: str = "",
                 scene: str = DEFAULT_SCENE, clip_meta: Optional[dict] = None) -> int:
    """Consume one video clip: update the session draft (chains / roster / staged / lines). Returns
    this clip's global index.

    The clip comes from one of two sources: clip_key (already in our OSS, the fast path) or clip_url
    (any http(s) URL, including a presigned URL to the caller's own bucket — downloaded and then
    re-stored into our OSS). clip_index uses a session-global monotonic sequence number (never trust
    the caller's numbering, which prevents collisions when several batches interleave).
    harvest needs a local file, so the clip is downloaded from our OSS to a temp file and deleted
    afterwards.
    """
    from personos.identity.backends.omni import ContentRejectedError, MediaUnfetchableError

    try:
        return _process_clip_locked(deps, session_id=session_id, clip_key=clip_key,
                                    clip_url=clip_url, scene=scene, clip_meta=clip_meta)
    except (MediaUnfetchableError, ContentRejectedError) as e:
        # These two are deterministic failures of **this clip itself** (the file is too large for
        # upstream to fetch / content moderation rejected it), so retrying the same clip just burns
        # the 2-minute failure window five more times. Convert to ClipRejected -> the consumer
        # records and skips it, and the rest of the batch carries on.
        raise ClipRejected(str(e)) from e
    except ImportError as e:
        # A missing dependency (say the image has no PyAV) is a **deployment problem, not a transient
        # fault**: retrying on this pod changes nothing. And the retry cost is enormous — on the
        # "our own key" path the import happens **after** the screenplay MLLM, so every retry first
        # wastes another 2-minute screenplay call; five retries is ten minutes of upstream quota.
        # Real incident: the SIT image was missing av, video processing was wiped out entirely, and
        # all that was left was a ModuleNotFoundError in the logs.
        # Converting to ClipRejected leaves a record (queryable in the tasks table) and skips it,
        # carrying the original error text so it is obvious at a glance which package is missing.
        raise ClipRejected(f"video dependency missing (a deployment problem, not a problem with this clip): {e}; "
                           f"check that the image installed the video dependency section of requirements.txt") from e


def _process_clip_locked(deps: VideoDeps, *, session_id: str, clip_key: str, clip_url: str,
                         scene: str, clip_meta: Optional[dict]) -> int:
    omni = deps.backends["mm_runner"]
    draft, store, cloud, ms = deps.draft, deps.store, deps.cloud, deps.media_store
    book = ChainBook(draft, media_store=ms)
    registry = AnchorRegistry(store, cloud, draft, media_store=ms)

    # Minimize the temp file's lifetime: only harvest (material extraction) needs the local file; the
    # screenplay MLLM only consumes a signed URL.
    # - our own key: run the screenplay first (nothing on disk at that point) and download only when
    #   it is needed -> the file lives roughly only for the duration of harvest;
    # - an external URL: we must download before we can re-store it and get our own key (the MLLM has
    #   to reach our signed URL), and since the file is already in hand we validate the duration
    #   right away — an over-long clip can be rejected immediately, saving a 2-3min screenplay call.
    costs: dict[str, float] = {}           # per-stage cost (used in production to locate the bottleneck; see _timed above)
    t_all = time.monotonic()
    tmp_path, dur = "", None
    if not clip_key:
        with _timed(costs, "fetch"):       # external URL: download + re-store into our OSS (environment-dependent; should be near 0 on the internal network)
            tmp_path, clip_key = _materialize_clip(ms, clip_key="", clip_url=clip_url,
                                                   owner=store.user_id)
        logger.info(f"clip re-stored from external link into our object store session={session_id} "
                    f"key={clip_key[-24:]} cost={costs['fetch']:.1f}s")

    # Clip-level idempotency: a queue redelivery replays the **whole message** (up to 20 clips), so
    # clips that were already fully processed are skipped; otherwise the earlier clips get
    # double-counted (presence, lines and material all recorded twice). See draft.clip_done_index.
    done = draft.clip_done_index(session_id, clip_key)
    if done is not None:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)                 # on the external-URL path it is already on disk; release it before skipping
        logger.info(f"clip already processed, skipping (message redelivery replay) session={session_id} "
                    f"clip_seq={done} key={clip_key[-24:]}")
        return done

    clip_index = draft.next_clip_seq(session_id, clip_key)   # globally monotonic + records clip_index -> key

    try:
        if tmp_path:                       # external-URL path: the file is in hand, so check the duration before paying for the screenplay
            dur = _duration(tmp_path)
            _reject_if_too_long(dur)
        roster_cards = registry.roster_cards_for_prompt(session_id)
        with _timed(costs, "script"):      # screenplay MLLM: cost is roughly linear in frame count (duration x VIDEO_FPS)
            prompt, imgs = build_clip_prompt(scene_setting=scene, roster_cards=roster_cards)
            raw = omni.chat(prompt, video_url=ms.sign_url(clip_key), images_b64=imgs,
                            max_tokens=20000, temperature=0.0, video_fps=VIDEO_FPS)
        if not tmp_path:                   # our-own-key path: only now is a local file needed (nothing was on disk during the screenplay)
            with _timed(costs, "fetch"):
                tmp_path, clip_key = _materialize_clip(ms, clip_key=clip_key, clip_url="",
                                                       owner=store.user_id)
            dur = _duration(tmp_path)
            _reject_if_too_long(dur)
        script = parse_clip_output(raw, duration_sec=dur)
        # The screenplay is the source of the entire identity path (naming, descriptions, nominations
        # and continuations all come from here), so it is logged verbatim.
        # This stretch previously had no logging at all: when identity went wrong (a name failed to
        # bind, a chain split) there was nothing in the logs to look at and the only recourse was
        # digging through langfuse — which contradicts the "debug from persisted logs, do not guess"
        # convention. This matches what chat_json does on the text side.
        logger.info(
            f"screenplay MLLM session={session_id} clip={clip_index} parsed_ok={script.parsed_ok} "
            f"issues={script.issues}\n"
            f"  ── casts ──\n" + "\n".join(
                f"    {c.local_id} name={c.name!r} name_evidence={c.name_evidence!r} "
                f"prev={script.cont.get(c.local_id)!r} desc={c.desc!r}" for c in script.casts)
            + f"\n  ── lines={len(script.lines)} noms={len(script.nominations)} "
              f"voices={len(script.voice_ranges)} ──\n"
              f"  -- output(raw) --\n{raw}")
        # Physical-consistency guard: detect contradictions in the screenplay that are "impossible in
        # the real world" in one pass -> hand them back to the model once to fix -> conservatively
        # degrade if it still does not pass. It runs **before harvest**: bad nominations are culled
        # before any face is extracted, so a wrong face never gets into the probability cloud.
        with _timed(costs, "guard"):
            script, guard_rep = repair.enforce(
                script, omni=omni, clip_url=ms.sign_url(clip_key), roster_cards=roster_cards,
                duration_sec=dur, session_id=session_id, clip_index=clip_index)
        if guard_rep.get("found"):
            logger.warning(
                f"screenplay physical contradiction session={session_id} clip={clip_index} "
                f"mode={guard_rep['mode']} found={guard_rep['found']} repair_rounds={guard_rep['attempts']} "
                f"remaining={guard_rep['remaining'] or 'none'} degraded={guard_rep['degraded']}")

        registry.map_casts(session_id, script)
        with _timed(costs, "harvest"):     # local inference: frame selection + face detection + voiceprint (scales with the number of nominations, not with duration)
            ev_by_local = harvest_clip(tmp_path, script, deps.backends)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)            # release right after the material is extracted (the adjudication and staging stages no longer hold the clip)

    evidence = _merge_by_cast(ev_by_local, script.cast_map)
    refreshed = book.observe_clip(session_id, clip_index, script, evidence)

    eval_casts = [c for c in script.session_cast_ids()
                  if c != WEARER_CAST_ID
                  and draft.canonical_chain(draft.chain_ref(session_id, c)) in refreshed]
    n_cand = 0
    if eval_casts:
        with _timed(costs, "arbitrate"):   # candidate recall + batched adjudication MLLM (scales with the number of people to judge and of candidates)
            qbc = {c: book.query_card(session_id, c, script, evidence=evidence.get(c))
                   for c in eval_casts}
            extra = {c: book.pending_cards(session_id, for_chain=draft.chain_ref(session_id, c))
                     for c in eval_casts}
            cand = registry.build_candidates(eval_casts, evidence, script, extra_cards=extra)
            pool, seen = [], set()
            for cards in cand.values():
                for card in cards:
                    if card.character_id not in seen:
                        seen.add(card.character_id); pool.append(card)
            n_cand = len(pool)
            if pool:
                ap, ai, aa = build_arbitration_prompt([qbc[c] for c in eval_casts], pool)
                araw = omni.chat(ap, images_b64=ai, audio_b64_list=aa,
                                 max_tokens=2048, temperature=0.0)
                verdicts, _issues = parse_verdicts(araw, cast_ids=eval_casts,
                                                   candidate_ids=[c.character_id for c in pool])
                logger.info(
                    f"identity arbitration session={session_id} clip={clip_index} "
                    f"pending={eval_casts} candidates={[c.character_id[:14] for c in pool]} "
                    f"verdicts={verdicts} issues={_issues}\n  -- output(raw) --\n{araw}")
                for cast, verdict in verdicts.items():
                    ch = draft.canonical_chain(draft.chain_ref(session_id, cast))
                    book.apply_verdict(draft.chain_ref(session_id, cast), verdict,
                                       session_id=session_id, clip_index=clip_index,
                                       reason="+".join(refreshed[ch]), issues=[])
                resolve_chain_collisions(book, script, session_id=session_id,
                                         clip_index=clip_index, queries_by_cast=qbc,
                                         pool=pool, omni=omni)

    with _timed(costs, "stage"):           # crop the material and upload to OSS + write lines/roster into the Redis draft
        for cast, ev in evidence.items():
            canonical = draft.canonical_chain(draft.chain_ref(session_id, cast))
            draft.stage_evidence(canonical, session_id=session_id, clip_index=clip_index,
                                 evidence=ev, media_store=ms)
        draft.stage_lines(session_id, clip_index,
                          [(l.t0, l.t1, script.cast_map.get(l.who, l.who), l.kind,
                            rewrite_ids(l.text, script.cast_map))
                           for l in script.lines])
        registry.update_roster(session_id, clip_index, script, bindings={},
                               evidence_by_cast=evidence)
        # Only mark it done after every stage has landed in the draft — marking too early would make
        # a clip that failed part-way get skipped on redelivery (silently losing that memory)
        draft.mark_clip_done(session_id, clip_key, clip_index)
    # One structured cost line: aggregating on this line in production answers "which stage is slow,
    # and does it grow with duration or with the number of people?".
    # `frames` estimates how many frames went into the screenplay MLLM (duration x fps); the script
    # stage should be roughly proportional to it.
    total = time.monotonic() - t_all
    logger.info(f"video clip consumed session={session_id} clip_seq={clip_index} "
                f"casts={script.session_cast_ids()} | cost total={total:.1f}s {_fmt(costs)} "
                f"| dur={dur or 0:.1f}s fps={VIDEO_FPS} frames≈{int((dur or 0) * VIDEO_FPS)} "
                f"lines={len(script.lines)} casts_n={len(script.session_cast_ids())} "
                f"cand={n_cand}")
    return clip_index


def finalize_video(deps: VideoDeps, *, session_id: str) -> Optional[CellBuild]:
    """End of session: the two-phase final identity review (commit_session) -> land the attributed
    lines into memory (flush_session_to_memory).
    Returns None when there is no pending video draft (this session had no video)."""
    if not deps.draft.pending_chains(session_id):
        return None
    store, cloud, draft, ms = deps.store, deps.cloud, deps.draft, deps.media_store
    book = ChainBook(draft, media_store=ms)
    registry = AnchorRegistry(store, cloud, draft, media_store=ms)
    costs: dict[str, float] = {}
    t_all = time.monotonic()
    with _timed(costs, "commit"):          # two-phase final identity review: one MLLM re-check per pending chain (grows with the number of chains)
        report = commit_session(store, cloud, registry, book, session_id=session_id,
                                omni=deps.backends["mm_runner"], media_store=ms)
    logger.info(f"video final review session={session_id} registered={len(report['registered'])} "
                f"wearer={bool(report.get('wearer'))} name_merges={report.get('name_merges', [])} "
                f"cost={costs['commit']:.1f}s")
    with _timed(costs, "flush"):           # attributed lines -> evidence persisted + build_cell (episode / atoms / chain assignment)
        cb = flush_session_to_memory(
            draft, report["by_chain"], session_id=session_id, char_store=store,
            evidence_store=deps.evidence, cell_store=deps.cells, atom_store=deps.atoms,
            chain_store=deps.chains, llm=deps.llm, embedder=deps.embedder, media_store=ms,
            clip_keys=draft.clip_keys(session_id))
    # The end of a session is a **fixed tail that runs once per session** (weakly correlated with
    # clip count, strongly with the number of characters), so it gets its own line for aggregation
    logger.info(f"video session end session={session_id} cost total={time.monotonic() - t_all:.1f}s "
                f"{_fmt(costs)} | chains={len(report['by_chain'])} "
                f"clips={len(draft.clip_keys(session_id))} "
                f"memcell={cb.cell.id if cb else '-'} atoms={len(cb.atoms) if cb else 0}")
    # Final review succeeded -> clear the session draft (matching seg_store.clear; the people are now
    # persisted as characters in the store, so a later clip in the same session starts a fresh chain
    # and is recognized as an existing acquaintance through adjudication). `clear` is a method both
    # the Redis and the Memory implementation provide.
    clear = getattr(draft, "clear", None)
    if callable(clear):
        clear(session_id)
    return cb
