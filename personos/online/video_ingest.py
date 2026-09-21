"""视频 clip 的消费侧编排(生产链路):逐 clip 更新 Redis 草稿 + 会话末终审落记忆。

从 scripts/video/verify_pipeline._process 抽成库函数,消费侧(ingest_worker)与脚本共用,去重。
两条入口:
- process_clip:一条视频消息(clip 的 OSS key)→ 下载→剧本→harvest→observe→(刷新链)批量仲裁
  +碰撞修复→stage 素材/行/roster。clip 序号用会话全局单调计数(不信调用方,防交错撞车)。
- finalize_video:会话末 → commit_session(身份两阶段终审)+ flush_session_to_memory(行归属→记忆)。

草稿累积态在 RedisDraftStore(跨消息/跨副本);与文本 seg 独立,交错任意顺序都正确。
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
    """视频消费侧的一整套依赖(runtime.video_deps 装配)。backends 是进程单例(重模型),
    其余 per-user(store/cloud/draft/四个记忆 store)。"""

    store: CharacterStore          # 身份持久层(MySQL)
    cloud: CloudEngine             # 概率身份云
    draft: DraftStore              # 会话草稿(Redis,跨 clip 累积)
    backends: dict[str, Any]       # {mm_runner, face_detector, voiceprint}——进程单例
    media_store: Any               # OSS
    maas: Any                      # 文本 LLM + embedder(build_cell 用)
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
    """记一个环节的墙钟耗时到 costs[stage](累加:同名环节多次调用合并)。

    为什么要打:视频消费是分钟级的重活,上线后"慢在哪一环"必须能从日志直接回答——
    下载(环境相关)/剧本 MLLM(随帧数)/harvest(随提名数)/仲裁(随人数)四者的优化手段完全不同。
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        costs[stage] = costs.get(stage, 0.0) + (time.monotonic() - t0)


def _fmt(costs: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.1f}s" for k, v in costs.items())


CLIP_FETCH_TIMEOUT_S = 120.0
# clip 体积/时长上限:体积复用 media_store 的口径;时长新增(视频比文本贵一个量级,
# 过长 clip 会把单条消息的处理时间拖到锁 TTL 之外,也会撑爆 MLLM 上下文)。
MAX_CLIP_BYTES = int(os.environ.get("PERSONOS_VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))
# 上限取 150 而非 120:调用方切「2 分钟 clip」实际会切出 120.1s 这种略超的片子,卡在正好
# 120 会把正常调用全拒掉。留 25% 余量,既容得下标称 2min,又挡得住明显过长的片子。
MAX_CLIP_DURATION_S = float(os.environ.get("PERSONOS_VIDEO_MAX_DURATION_S", "150"))


class ClipRejected(ValueError):
    """clip 永久性不可用(地址失效/超大/超长/格式错)——重试无意义,应留痕跳过而非静默丢。"""


def _reject_if_too_long(dur: Optional[float]) -> None:
    """过长 clip:处理时间会顶穿会话锁 TTL,也撑爆 MLLM 上下文。"""
    if dur and dur > MAX_CLIP_DURATION_S:
        raise ClipRejected(f"视频时长 {dur:.1f}s 超过上限 {MAX_CLIP_DURATION_S}s(请切成更短的 clip)")


def _materialize_clip(media_store: Any, *, clip_key: str, clip_url: str, owner: str,
                      ) -> tuple[str, str]:
    """把 clip 落成**本地临时文件**,返回 (tmp_path, clip_key)。

    内存纪律(要命):clip 动辄几十 MB,而后续 MLLM 调用要 2-3 分钟——字节绝不能在内存里跟着
    整个流程走。这里做到"落盘即释放":
    - 我方 key:read_bytes 拿到的字节写盘后立刻 del(不带进 MLLM 阶段);
    - 外链:**边下边写盘**,不在内存拼整包(原先 b"".join(chunks) 有 2 倍瞬时峰值);
      上传我方 OSS 时才一次性读回(短暂),随即释放。
    """
    fd = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = fd.name
    try:
        if clip_key:
            try:
                data = media_store.read_bytes(clip_key)
            except Exception as e:  # noqa: BLE001
                raise ClipRejected(f"我方 OSS 取 clip 失败 key={clip_key[-24:]}: {e}") from e
            if len(data) > MAX_CLIP_BYTES:
                raise ClipRejected(f"视频超过 {MAX_CLIP_BYTES} 字节上限")
            fd.write(data)
            del data                        # 落盘即释放,不带进 MLLM 阶段
            fd.close()
            return tmp_path, clip_key
        if not clip_url:
            raise ClipRejected("process_clip 需要 clip_key 或 clip_url 之一")
        _stream_to_file(clip_url, fd)       # 边下边写盘 + 边卡体积上限
        fd.close()
        data = open(tmp_path, "rb").read()  # 仅为算 sha/上传,读一次即释放
        try:
            key = media_store.save_video(data, owner=owner).key
        except Exception as e:  # noqa: BLE001
            raise ClipRejected(f"视频转存失败:{e}") from e
        finally:
            del data
        return tmp_path, key
    except Exception:
        fd.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)             # 失败也不留垃圾文件
        raise


def _stream_to_file(url: str, fd) -> None:
    """流式下载到已打开的文件句柄,边下边卡体积上限(不把超大文件读进内存)。"""
    import httpx
    try:
        with httpx.stream("GET", url, timeout=CLIP_FETCH_TIMEOUT_S, follow_redirects=True) as r:
            if r.status_code >= 400:
                raise ClipRejected(f"视频地址不可访问(HTTP {r.status_code}):{url[:80]}")
            declared = int(r.headers.get("content-length") or 0)
            if declared and declared > MAX_CLIP_BYTES:
                raise ClipRejected(f"视频超过 {MAX_CLIP_BYTES} 字节上限(声明 {declared})")
            total = 0
            for chunk in r.iter_bytes():
                total += len(chunk)
                if total > MAX_CLIP_BYTES:
                    raise ClipRejected(f"视频超过 {MAX_CLIP_BYTES} 字节上限(下载中超限)")
                fd.write(chunk)
    except ClipRejected:
        raise
    except Exception as e:  # noqa: BLE001  DNS/连接/超时/证书 → 地址不可用(永久失败)
        raise ClipRejected(f"视频地址下载失败({type(e).__name__}: {e}):{url[:80]}") from e


# 这里曾有第二道并发闸(BoundedSemaphore(4)),在独立视频线程池之前用来防内存/GPU 打爆。
# 有了 video_pool_size 之后两者管的是同一件事,而**生效的永远是更严的那道**——池给 50 个线程,
# 46 个会卡在闸上干等,且各自还握着自己会话的锁(跨副本互斥),坑位占着不产出。
# 故删去,视频并发**只由 PERSONOS_VIDEO_POOL 一个旋钮决定**(见 config.video_pool_size)。

# 剧本 MLLM 的抽帧率:**单 clip 耗时的头号变量**(送进模型的帧数 = 时长 × fps,耗时近线性)。
# 默认 1.0(1s 一帧):相比 2.0 把剧本环节砍半,是把单 clip 压到接近实时的唯一有效杠杆。
# 需要更细的提名时刻定位(快速动作/短镜头多)可回调 2.0,代价是耗时翻倍。
VIDEO_FPS = float(os.environ.get("PERSONOS_VIDEO_FPS", "1.0"))


def process_clip(deps: VideoDeps, *, session_id: str, clip_key: str = "", clip_url: str = "",
                 scene: str = DEFAULT_SCENE, clip_meta: Optional[dict] = None) -> int:
    """消费一条视频 clip:更新会话草稿(chains/roster/staged/lines)。返回本 clip 的全局序号。

    clip 来源二选一:clip_key(已在我方 OSS,快捷)/ clip_url(任意 http(s),含调用方自己桶的预签名
    URL——下载后转存我方 OSS)。clip_index 用会话全局单调序号(不信调用方编号,交错多批时防撞)。
    harvest 需本地文件 → 从我方 OSS 下载到临时文件,用完删。
    """
    from personos.identity.backends.omni import ContentRejectedError, MediaUnfetchableError

    try:
        return _process_clip_locked(deps, session_id=session_id, clip_key=clip_key,
                                    clip_url=clip_url, scene=scene, clip_meta=clip_meta)
    except (MediaUnfetchableError, ContentRejectedError) as e:
        # 这两类是**这条 clip 本身**的确定性失败(文件太大上游拉不动 / 内容审查拒绝),
        # 重试同一个 clip 只会把每次 2 分钟的失败窗口再烧 5 遍。转成 ClipRejected →
        # 消费侧留痕跳过,整批其余 clip 继续。
        raise ClipRejected(str(e)) from e
    except ImportError as e:
        # 缺依赖(如镜像没装 PyAV)是**部署问题,不是瞬时故障**:本 pod 重试多少次都一样。
        # 而重试代价极高——我方 key 路径的 import 发生在剧本 MLLM **之后**,每重试一次
        # 就先白跑一次 2 分钟的剧本调用,5 次就是十分钟上游配额。
        # 实测事故:SIT 镜像缺 av,视频全军覆没且只在日志里留 ModuleNotFoundError。
        # 转 ClipRejected → 留痕(tasks 表可查)+ 跳过,错误原文照带,一眼看出缺哪个包。
        raise ClipRejected(f"视频依赖缺失(部署问题,非本 clip 的问题):{e};"
                           f"请确认镜像已装 requirements.txt 的视频依赖段") from e


def _process_clip_locked(deps: VideoDeps, *, session_id: str, clip_key: str, clip_url: str,
                         scene: str, clip_meta: Optional[dict]) -> int:
    omni = deps.backends["mm_runner"]
    draft, store, cloud, ms = deps.draft, deps.store, deps.cloud, deps.media_store
    book = ChainBook(draft, media_store=ms)
    registry = AnchorRegistry(store, cloud, draft, media_store=ms)

    # 临时文件生命周期最小化:本地文件只有 harvest(抽素材)才需要,剧本 MLLM 只吃签名 URL。
    # - 我方 key:先跑剧本(此刻盘上无文件),用到时才下载 → 文件存活 ≈ 只覆盖 harvest;
    # - 外链:必须先下载才能转存拿到我方 key(MLLM 要访问我方签名 URL),文件已在手就顺带
    #   先做时长校验——过长可立刻拒,省掉一次 2-3min 的剧本调用。
    costs: dict[str, float] = {}           # 环节耗时(上线后据此定位瓶颈,见文件末 _timed)
    t_all = time.monotonic()
    tmp_path, dur = "", None
    if not clip_key:
        with _timed(costs, "fetch"):       # 外链:下载 + 转存我方 OSS(环境相关,内网应接近 0)
            tmp_path, clip_key = _materialize_clip(ms, clip_key="", clip_url=clip_url,
                                                   owner=store.user_id)
        logger.info(f"clip 外链转存我方 OSS session={session_id} key={clip_key[-24:]} "
                    f"耗时={costs['fetch']:.1f}s")

    # clip 级幂等:队列重投重放的是**整条消息**(最多 20 个 clip),已完整处理过的直接跳过,
    # 否则前面的 clip 会被重复记账(presence/lines/素材各记两遍)。见 draft.clip_done_index。
    done = draft.clip_done_index(session_id, clip_key)
    if done is not None:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)                 # 外链路径已下载到盘,跳过前先释放
        logger.info(f"clip 已处理过,跳过(消息重投重放)session={session_id} "
                    f"clip_seq={done} key={clip_key[-24:]}")
        return done

    clip_index = draft.next_clip_seq(session_id, clip_key)   # 全局单调 + 记 clip_index→key

    try:
        if tmp_path:                       # 外链路径:文件已在手,先卡时长再花钱跑剧本
            dur = _duration(tmp_path)
            _reject_if_too_long(dur)
        roster_cards = registry.roster_cards_for_prompt(session_id)
        with _timed(costs, "script"):      # 剧本 MLLM:耗时 ≈ 帧数(时长×VIDEO_FPS)线性
            prompt, imgs = build_clip_prompt(scene_setting=scene, roster_cards=roster_cards)
            raw = omni.chat(prompt, video_url=ms.sign_url(clip_key), images_b64=imgs,
                            max_tokens=20000, temperature=0.0, video_fps=VIDEO_FPS)
        if not tmp_path:                   # 我方 key 路径:到这才需要本地文件(剧本期间盘上无文件)
            with _timed(costs, "fetch"):
                tmp_path, clip_key = _materialize_clip(ms, clip_key=clip_key, clip_url="",
                                                       owner=store.user_id)
            dur = _duration(tmp_path)
            _reject_if_too_long(dur)
        script = parse_clip_output(raw, duration_sec=dur)
        # 剧本是整条身份链路的源头(取名/描述/提名/续接全从这里来),原样落盘。
        # 此前这一段完全没有日志:身份出问题(名字没绑上、链分裂)时日志里什么都查不到,
        # 只能去 langfuse 翻——与"调试靠日志落盘,不猜"的约定相悖。对齐文本侧 chat_json 的做法。
        logger.info(
            f"剧本 MLLM session={session_id} clip={clip_index} parsed_ok={script.parsed_ok} "
            f"issues={script.issues}\n"
            f"  ── casts ──\n" + "\n".join(
                f"    {c.local_id} name={c.name!r} name_evidence={c.name_evidence!r} "
                f"prev={script.cont.get(c.local_id)!r} desc={c.desc!r}" for c in script.casts)
            + f"\n  ── lines={len(script.lines)} noms={len(script.nominations)} "
              f"voices={len(script.voice_ranges)} ──\n"
              f"  ── 输出(raw)──\n{raw}")
        # 物理一致性守护:统一检出剧本里"现实世界不可能"的矛盾 → 一次性交回模型改 →
        # 仍不过则保守降级。**放在 harvest 之前**:坏提名在抽脸前就被剔除,错脸没机会进概率云。
        with _timed(costs, "guard"):
            script, guard_rep = repair.enforce(
                script, omni=omni, clip_url=ms.sign_url(clip_key), roster_cards=roster_cards,
                duration_sec=dur, session_id=session_id, clip_index=clip_index)
        if guard_rep.get("found"):
            logger.warning(
                f"剧本物理矛盾 session={session_id} clip={clip_index} mode={guard_rep['mode']} "
                f"检出={guard_rep['found']} 重修轮次={guard_rep['attempts']} "
                f"残留={guard_rep['remaining'] or '无'} 降级={guard_rep['degraded']}")

        registry.map_casts(session_id, script)
        with _timed(costs, "harvest"):     # 本地推理:挑帧 + 人脸检测 + 声纹(随提名数,不随时长)
            ev_by_local = harvest_clip(tmp_path, script, deps.backends)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)            # 抽完素材立即释放(仲裁/staging 阶段不再持有 clip)

    evidence = _merge_by_cast(ev_by_local, script.cast_map)
    refreshed = book.observe_clip(session_id, clip_index, script, evidence)

    eval_casts = [c for c in script.session_cast_ids()
                  if c != WEARER_CAST_ID
                  and draft.canonical_chain(draft.chain_ref(session_id, c)) in refreshed]
    n_cand = 0
    if eval_casts:
        with _timed(costs, "arbitrate"):   # 候选召回 + 批量仲裁 MLLM(随待判人数/候选数)
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
                    f"身份仲裁 session={session_id} clip={clip_index} "
                    f"待判={eval_casts} 候选={[c.character_id[:14] for c in pool]} "
                    f"判定={verdicts} issues={_issues}\n  ── 输出(raw)──\n{araw}")
                for cast, verdict in verdicts.items():
                    ch = draft.canonical_chain(draft.chain_ref(session_id, cast))
                    book.apply_verdict(draft.chain_ref(session_id, cast), verdict,
                                       session_id=session_id, clip_index=clip_index,
                                       reason="+".join(refreshed[ch]), issues=[])
                resolve_chain_collisions(book, script, session_id=session_id,
                                         clip_index=clip_index, queries_by_cast=qbc,
                                         pool=pool, omni=omni)

    with _timed(costs, "stage"):           # 素材切图上传 OSS + 行/roster 写 Redis 草稿
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
        # 全部环节落草稿后才标完成——标早了会让中途失败的 clip 在重投时被跳过(静默丢记忆)
        draft.mark_clip_done(session_id, clip_key, clip_index)
    # 单行结构化耗时:上线后按此行聚合即可回答"慢在哪一环、是否随时长/人数涨"。
    # frames 是送进剧本 MLLM 的帧数估算(时长×fps),script 环节应与它近似成正比。
    total = time.monotonic() - t_all
    logger.info(f"视频 clip 消费 session={session_id} clip_seq={clip_index} "
                f"casts={script.session_cast_ids()} | 耗时 total={total:.1f}s {_fmt(costs)} "
                f"| dur={dur or 0:.1f}s fps={VIDEO_FPS} frames≈{int((dur or 0) * VIDEO_FPS)} "
                f"lines={len(script.lines)} casts_n={len(script.session_cast_ids())} "
                f"cand={n_cand}")
    return clip_index


def finalize_video(deps: VideoDeps, *, session_id: str) -> Optional[CellBuild]:
    """会话末:身份两阶段终审(commit_session)→ 行归属落记忆(flush_session_to_memory)。
    无 pending 视频草稿则返回 None(该 session 没视频)。"""
    if not deps.draft.pending_chains(session_id):
        return None
    store, cloud, draft, ms = deps.store, deps.cloud, deps.draft, deps.media_store
    book = ChainBook(draft, media_store=ms)
    registry = AnchorRegistry(store, cloud, draft, media_store=ms)
    costs: dict[str, float] = {}
    t_all = time.monotonic()
    with _timed(costs, "commit"):          # 身份两阶段终审:每条待定链一次 MLLM 复核(随链数涨)
        report = commit_session(store, cloud, registry, book, session_id=session_id,
                                omni=deps.backends["mm_runner"], media_store=ms)
    logger.info(f"视频终审 session={session_id} registered={len(report['registered'])} "
                f"wearer={bool(report.get('wearer'))} name_merges={report.get('name_merges', [])} "
                f"耗时={costs['commit']:.1f}s")
    with _timed(costs, "flush"):           # 行归属→evidence 入库 + build_cell(episode/atom/判链)
        cb = flush_session_to_memory(
            draft, report["by_chain"], session_id=session_id, char_store=store,
            evidence_store=deps.evidence, cell_store=deps.cells, atom_store=deps.atoms,
            chain_store=deps.chains, llm=deps.maas, embedder=deps.maas, media_store=ms,
            clip_keys=draft.clip_keys(session_id))
    # 会话末是**每 session 一次的固定尾巴**(与 clip 数弱相关、与角色数强相关),单独成行便于聚合
    logger.info(f"视频会话末 session={session_id} 耗时 total={time.monotonic() - t_all:.1f}s "
                f"{_fmt(costs)} | chains={len(report['by_chain'])} "
                f"clips={len(draft.clip_keys(session_id))} "
                f"memcell={cb.cell.id if cb else '-'} atoms={len(cb.atoms) if cb else 0}")
    # 终审成功 → 清会话草稿(对齐 seg_store.clear;人已落成库里 character,后续同 session 新 clip
    # 从新链起、经仲裁认作老熟人)。clear 是 Redis/Memory 两实现都有的方法。
    clear = getattr(draft, "clear", None)
    if callable(clear):
        clear(session_id)
    return cb
