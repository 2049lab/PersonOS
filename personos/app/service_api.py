"""对外记忆服务 API(外部 agent 调用的稳定契约)。

设计铁律:
- 只给"记忆本身有意义"的字段;凡返回的记忆条目必带 atom_id + evidence_refs(可溯源=可信基础)。
- 不下发任何内部量(打分 breakdown/score/tier/salience、prompt/raw、深轨 steps、full state)。
- 写侧收口(ingest),读侧放权(recall / trace)。
- 【多租户】先 POST /users/register 拿 token;此后所有记忆操作带 X-User-Token 头,
  只能读写该 user 自己的记忆(user 内跨会话共享,跨 user 由 store 层物理隔离)。
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

from .. import obs
from ..config import settings
from ..logging_setup import trace
from ..models import now
from ..online.profile_render import render as render_profile
from ..online.trust import build_trust_chain, trace_evidence
from ..storage.msg_queue import EnqueueBusy
from ..storage.profile_store import BANDS, ProfileStore
from fastapi import Depends as _Depends
from .recall_flow import PUBLIC_MODES, run_recall
from .response import EnvelopeRoute
from .runtime import UserContext, rt
from .session_scope import scoped_session, valid_user_id
from .signing import verify_signature
from .views import memory_view as _memory_view

# route_class:中心化把每个 handler 的返回响应包成 {code,data,msg}(不逐个改端点体)
# dependencies:router 级 AK/SK 验签,先于各端点 _ctx 跑;pytest 下 no-op。所有 /api/v1 接口(含 register)都验签。
router = APIRouter(prefix="/api/v1", route_class=EnvelopeRoute,
                   dependencies=[_Depends(verify_signature)])

# 对外"依据记忆"收敛:精排后前 N 个单元的命中 atoms(内部作答吃全部材料单元,对外只给一小撮)
_PUBLIC_FAST_MEMORIES = 10
# 单批 ingest 上限:防单个 task 拖太久(每批 1 次 W1 判界,转移还要 W2 织写)
_MAX_INGEST_MESSAGES = 20


def _ctx(x_user_token: str = Header(default="")) -> UserContext:
    """token → 该 user 的绑定上下文。缺失/未知 → 401。"""
    ctx = rt.ctx_by_token(x_user_token)
    if ctx is None:
        raise HTTPException(status_code=401, detail="X-User-Token 缺失或无效,请先 POST /api/v1/users/register")
    return ctx


def _sid(caller: str, session_id: str) -> str:
    """入口处的调用方改写 + id 卫生(不合法 → 400;规则见 session_scope)。"""
    try:
        return scoped_session(caller, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# —— 请求体 ——
class RegisterBody(BaseModel):
    user_id: str | None = None    # 可选;缺省自动生成。仅字母/数字/下划线/连字符,≤128


class IngestMessage(BaseModel):
    speaker: str                  # 说话人:与助手对话的本人固定 "user";多参与者对话用原话里的名字
    text: str = ""                # 消息文本(图片/视频消息可空)
    image_b64: str | None = None  # 可选:base64 图片(带 data:image/...;base64, 前缀或纯 base64 均可)
    image_content_type: str = "image/jpeg"
    # 视频 clip(clip 太大不内联走队列,二选一;向前兼容:老调用不填):
    # - video_url:任意 http(s) 可下载地址(含调用方自己桶的预签名 URL)——**推荐**,不要求传我们桶;
    #   消费侧下载后转存我方 OSS(内容寻址去重),供 MLLM 签名访问 + evidence 溯源留底。
    # - video_oss_key:已在我方 OSS 时的快捷路径(跳过下载转存)。
    # 一条视频消息 = 一个 clip;消费侧逐 clip 跑身份管线,session_end 时统一终审落记忆。
    video_url: str | None = None
    video_oss_key: str | None = None
    clip_index: int = 0           # 调用方编号(仅元数据;真实 clip 序号由消费侧全局单调计,防交错撞车)
    duration_sec: float | None = None


class CallerContext(BaseModel):
    """调用方业务信息容器(统一口子):把业务方传入的场景/词表都收进来。

    - scenario:一段自由文本,描述业务场景与记忆侧重(如"饮食健康 App,关注用户饮食偏好与忌口")。
      注入写入(切段/episode/atom)与召回(改写/作答/深轨)及画像整理的高杠杆 LLM 环节——
      只调关注度/详略,不改事实、不编造、不漏。空 = 走默认链路(逐字节不变)。
    - task_type:episode 分类候选词表(原 IngestBody 顶层字段迁入此处);召回请求填了也不用。
    """
    scenario: str = ""
    task_type: list[str] | None = None


class IngestBody(BaseModel):
    caller: str = ""              # 调用方标识(多调用方 session_id 可能重复,入口处拼前缀隔离)
    session_id: str               # 仅字母/数字/下划线/连字符/点,≤95(禁冒号:Redis 键分隔符)
    messages: list[IngestMessage] # 一批消息(原子):整批并入当前段或整批开启新段,批内不切
    context: CallerContext | None = None  # 可选:业务方场景 + 分类词表(见 CallerContext)


class RecallBody(BaseModel):
    caller: str = ""              # 同 ingest:须与写入时一致,才能命中同一会话的上下文
    session_id: str
    query: str
    # 可选:随问题带一张图(多模态召回只支持文字+图片,视频暂不支持)。带图则在 R0 之前多跑
    # 一道视觉理解改写——把图里的人/场景写进 query,后面的纯文本链路才用得上视觉信息。
    image_b64: str | None = None          # 带 data:image/...;base64, 前缀或纯 base64 均可
    image_content_type: str = "image/jpeg"
    mode: str = "auto"           # auto(快链,核判不过自动升深轨)| fast(只快链)| deep(深轨直达)
    top_k: int = 30              # R1 atom 池上限(两路 RRF 融合后保留数;材料单元由此派生)
    context: CallerContext | None = None  # 可选:业务方场景(召回只用 scenario;task_type 忽略)


class SessionEndBody(BaseModel):
    caller: str = ""              # 同 ingest
    session_id: str
    context: CallerContext | None = None  # 可选:业务方场景 + 尾段分类词表(见 CallerContext)


@router.get("/health")
def health():
    atoms = rt.db.fetch_one("SELECT COUNT(*) AS n FROM atoms")["n"]
    ev = rt.db.fetch_one("SELECT COUNT(*) AS n FROM evidence")["n"]
    cells = rt.db.fetch_one("SELECT COUNT(*) AS n FROM memcells")["n"]
    return {"status": "ok", "users": rt.users.count(), "atoms": atoms,
            "cells": cells, "evidence": ev,
            "db": f"{settings.mysql_host}/{settings.mysql_database}"}


@router.post("/users/register", status_code=201)
def register(body: RegisterBody):
    """注册用户,签发唯一 token。token 只在此响应完整返回一次,调用方须持久保存。"""
    try:
        uid = valid_user_id(body.user_id)          # 字符集不合法 → 400(区别于 409 已存在)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    try:
        return rt.users.register(uid)
    except ValueError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})


# 图片入队体积上限:图片以 base64 随消息进 Redis 队列(暂存,消费即出队),超限拒收
_MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _decode_image_b64(value: str | None, where: str) -> tuple[bytes | None, str | None]:
    """校验并归一化图片入参,返回 (字节, 去前缀的纯 base64);value 为空则 (None, None)。

    /ingest 与 /recall 共用一份:两处各写一遍必然漂(上限、data: 前缀、validate 开关),
    而"哪种图能传"是对外契约的一部分,必须一致。
    """
    if not value:
        return None, None
    raw = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{where} 不是合法的 base64")
    if len(decoded) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"{where} 图片过大(>{_MAX_IMAGE_BYTES // 1024 // 1024}MB)")
    return decoded, raw


# 视频 clip 体积/时长上限(与消费侧 video_ingest 同口径)
_MAX_CLIP_BYTES = int(os.environ.get("PERSONOS_VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))
_MAX_CLIP_DURATION_S = float(os.environ.get("PERSONOS_VIDEO_MAX_DURATION_S", "150"))
# 上游可拉取上限:剧本 MLLM 是把**签名 URL** 交给模型服务、由它自己去下载整段视频的。
# 它那边有自己的下载窗口——实测 106MB(2min@7.3Mbps)必现 `Download multimodal file timed out`,
# 55MB 则长期稳定。所以真正的约束不是我们能不能传,而是**上游拉不拉得动**,这比 _MAX_CLIP_BYTES
# (我方存储口径)严得多,必须单独一道。取 64MB:安全高于已验证可用的 55MB,远低于已验证失败的 106MB。
# 超限在入口就 400 让调用方降码率/切短,而不是让它收了 202 再在几分钟后静默失败。
_MAX_CLIP_UPSTREAM_BYTES = int(os.environ.get("PERSONOS_VIDEO_UPSTREAM_MAX_BYTES",
                                              str(64 * 1024 * 1024)))


def _too_big_for_upstream(total: int) -> str:
    """体积超上游可拉取上限 → 返回给调用方的错误原因(含可操作建议);通过则空串。"""
    if not total or total <= _MAX_CLIP_UPSTREAM_BYTES:
        return ""
    return (f"体积 {total / 1024 / 1024:.1f}MB 超过上限 "
            f"{_MAX_CLIP_UPSTREAM_BYTES / 1024 / 1024:.0f}MB(模型服务拉取会超时)"
            f"——请降低码率或切成更短的 clip")


# 外链预检超时要短,别拖慢入口
_URL_PRECHECK_TIMEOUT_S = float(os.environ.get("PERSONOS_VIDEO_URL_PRECHECK_TIMEOUT_S", "5"))


def _precheck_video_url(url: str) -> str:
    """外链快速预检(**带 Range 的 GET,只取 1 字节**):返回空串=通过,否则返回给调用方的错误原因。

    为什么在入口做:消费是异步的,调用方拿到 202 就走了;地址写错/已过期若只在消费侧才发现,
    调用方毫不知情(只能靠留痕事后查)。入口花几百毫秒探一下,常见错误当场报。

    为什么不用 HEAD(踩过):**预签名 URL 的签名绑定 HTTP 方法**,签的是 GET 时 HEAD 一律 403 ——
    用 HEAD 会把最常见的合法外链(调用方自己桶的预签名地址)全部误杀。带 Range 的 GET 方法匹配,
    且只传 1 字节,代价与 HEAD 相当。
    总体取向仍是"宁放勿杀":只拦明确的 4xx/5xx 与超大声明,任何异常/抖动一律放行,交消费侧真下载判定。
    """
    try:
        import httpx
        r = httpx.get(url, headers={"Range": "bytes=0-0"},
                      timeout=_URL_PRECHECK_TIMEOUT_S, follow_redirects=True)
        if r.status_code >= 400:
            return f"不可访问(HTTP {r.status_code})"
        # 206 带 Content-Range: bytes 0-0/<total>;200(对端忽略 Range)则 Content-Length 即全长
        total = 0
        cr = r.headers.get("content-range") or ""
        if "/" in cr:
            total = int(cr.rsplit("/", 1)[-1] or 0)
        elif r.status_code == 200:
            total = int(r.headers.get("content-length") or 0)
        if total and total > _MAX_CLIP_BYTES:
            return f"体积 {total} 超过上限 {_MAX_CLIP_BYTES} 字节"
        err = _too_big_for_upstream(total)
        if err:
            return err
    except Exception:  # noqa: BLE001  抖动/对端不支持 Range → 放行,交给消费侧真下载判定
        return ""
    return ""


@router.post("/ingest", status_code=202)
def ingest(body: IngestBody, ctx: UserContext = Depends(_ctx)):
    """喂一批消息(入队,fire-and-forget):整批进【该会话的持久有序队列】,dispatcher 异步 FIFO 消费。

    **整批原子**:这批消息地位等于以前的一句——消费时要么整批并入当前话题段,要么整批开启
    新段,批内(如一对 QA)永远不切进两个 cell。批大小 1-20,超出 → 400。

    调用方无需"等上一批 done 再送下一批"——顺序/不丢/不重复由消费系统保证(见 ingest_worker):
    同会话消息按 seq 排队、单飞消费者按序处理、可靠出队(消费成功才 ack)、游标去重。
    消费有延后性(积压时),但所有消息保证被正确消费一次。caller 非空时 session_id 改写为
    f"{caller}:{session_id}",不同调用方同名会话天然隔离。返回 msg_id + seq(不再回逐批段闭合
    结果;进度可查 GET /api/v1/queue/status)。
    """
    if not body.messages or len(body.messages) > _MAX_INGEST_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"messages 须为 1-{_MAX_INGEST_MESSAGES} 条,收到 {len(body.messages)}")
    sid = _sid(body.caller, body.session_id)
    # 入口即校验并归一图片(坏 base64→400 / 过大→413);存纯 base64 进消息载荷(消费侧 decode 看图)
    # 视频批与文本/图片批不混:一批要么全视频(kind=video,逐 clip 走身份管线)、要么全文本/图片
    # (kind=ingest,走 feed_batch)。视频 clip 太大不内联——只带 URL/OSS key,消费侧异步取。
    def _is_vid(m) -> bool:
        return bool(m.video_url or m.video_oss_key)

    is_video = any(_is_vid(m) for m in body.messages)
    if is_video and any(not _is_vid(m) for m in body.messages):
        raise HTTPException(status_code=400, detail="一批消息不能混合视频与文本/图片,请分批发送")
    msgs: list[dict] = []
    for i, m in enumerate(body.messages):
        if not m.speaker.strip():
            raise HTTPException(status_code=400, detail=f"messages[{i}].speaker 不能为空")
        if _is_vid(m):
            # 同一条消息不得既带视频又带文本/图片——否则另一半会被静默丢弃(宁可明确拒绝)
            if m.text.strip() or m.image_b64:
                raise HTTPException(
                    status_code=400,
                    detail=f"messages[{i}] 不能同时带视频与 text/image_b64,请拆成两条消息分批发送")
            url = (m.video_url or "").strip()
            if url and not url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400,
                                    detail=f"messages[{i}].video_url 须是 http(s) 地址")
            # 调用方给的时长若已超限,入口就拒(便宜的一道;真实时长消费侧还会再验一次)
            if m.duration_sec and m.duration_sec > _MAX_CLIP_DURATION_S:
                raise HTTPException(
                    status_code=400,
                    detail=f"messages[{i}] 视频时长 {m.duration_sec}s 超过上限 "
                           f"{_MAX_CLIP_DURATION_S}s,请切成更短的 clip")
            # 外链可达性预检(带 Range 的 GET,短超时):地址失效/超大立刻报错,
            # 不让调用方收了 202 才悄悄失败
            if url:
                err = _precheck_video_url(url)
                if err:
                    raise HTTPException(status_code=400,
                                        detail=f"messages[{i}].video_url {err}")
            else:
                # 我方 key:体积直接问 OSS(一次 head,无下载)。这条路径此前完全没校验——
                # 调用方传个超大 clip 进来,要到几分钟后剧本 MLLM 挂了才知道。
                # 取不到大小(key 不存在/OSS 抖动)一律放行,交消费侧判定(宁放勿杀)。
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
                detail=f"messages[{i}] 须至少有 text / image_b64 / video_url(或 video_oss_key)之一")
        image_b64 = _decode_image_b64(m.image_b64, f"messages[{i}].image_b64")[1]
        msgs.append({"speaker": m.speaker.strip(), "text": m.text,
                     "image_b64": image_b64, "image_content_type": m.image_content_type})
    # 背压:单会话队列积压超限则拒收(防刷屏会话撑爆队列),调用方降速重试
    if rt.queue_depth(ctx.user_id, sid) >= settings.max_queue_depth:
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "该会话消息积压过多,请降速后重试"})
    tid = uuid.uuid4().hex   # 端到端 trace_id:透传队列 → 消费用同 id 作 langfuse trace + 日志 xrayTraceId
    cc = body.context or CallerContext()
    try:
        msg_id, seq = rt.enqueue_message(
            ctx.user_id, sid,
            {"messages": msgs, "task_type": cc.task_type, "scenario": cc.scenario,
             "trace_id": tid}, kind="video" if is_video else "ingest")
    except EnqueueBusy:   # 同会话入队严重争用超时:拒收让调用方重试(绝不硬上写坏顺序丢消息)
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "该会话入队繁忙,请稍后重试"})
    return {"accepted": True, "msg_id": msg_id, "seq": seq,
            "queue_depth": rt.queue_depth(ctx.user_id, sid), "xrayTraceId": tid}


@router.post("/session/end", status_code=202)
def session_end(body: SessionEndBody, ctx: UserContext = Depends(_ctx)):
    """会话结束(入队收尾任务):排在该会话既有消息之后,消费到它时强制闭合未闭合尾段。

    走同一条有序队列 → "收尾"保证发生在所有先到消息被消费之后(不会先于未消费的 ingest 执行)。
    caller 语义同 ingest。返回 msg_id + seq;回显原始 session_id(调用方自己的命名)。
    """
    sid = _sid(body.caller, body.session_id)
    tid = uuid.uuid4().hex
    cc = body.context or CallerContext()
    msg_id, seq = rt.enqueue_message(
        ctx.user_id, sid,
        {"task_type": cc.task_type, "scenario": cc.scenario, "trace_id": tid},
        kind="session_end")
    return {"accepted": True, "msg_id": msg_id, "seq": seq, "session_id": body.session_id,
            "xrayTraceId": tid}


@router.get("/queue/status")
def queue_status(session_id: str, caller: str = "", ctx: UserContext = Depends(_ctx)):
    """会话消费进度:depth=待消费积压,cursor=已消费到的最大 seq(调用方可判"我的 seq 消费了没")。"""
    sid = _sid(caller, session_id)
    mq = rt.msg_queue()
    return {"session_id": session_id, "depth": mq.depth(ctx.user_id, sid),
            "cursor": mq.cursor_get(ctx.user_id, sid)}


@router.get("/tasks/{task_id}")
def task_status(task_id: str, ctx: UserContext = Depends(_ctx)):
    """查异步任务状态(只能查自己 user 的任务):pending|running|done|error。"""
    t = rt.get_task(task_id)
    if t is None or t.get("user_id") != ctx.user_id:
        return JSONResponse(status_code=404, content={"error": "unknown task", "task_id": task_id})
    return t


@router.post("/recall")
def recall(body: RecallBody, ctx: UserContext = Depends(_ctx)):
    """召回:快链一条龙(R0→R1→R2→R5草稿→R3'核判,核判不过自动升深轨)或深轨直达,返回作答 + 依据记忆。

    answer 是一段标准事实陈述(第三人称/中立/带绝对日期),由调用方组织成自己的对话。
    verdict 是核判判级(ok=草稿通过对 / answer_defect=带指正重答后仍缺陷 / insufficient_material
    =材料不足),critique 是核判指正——调用方可据此追问或换问法;retried 表示是否重答过。
    每条 memory 内联 evidence(原文 + Q↔A),自带溯源。

    可随问题带一张图(image_b64):R0 之前先做一道视觉理解改写,把图里的人(与记忆里的人物对上)
    和场景写进 query,后面的纯文本链路才用得上视觉信息。不带图则链路逐字节不变。
    """
    if body.mode not in PUBLIC_MODES:
        return JSONResponse(status_code=400,
                            content={"error": f"mode 只支持 {list(PUBLIC_MODES)}", "got": body.mode})
    img, _ = _decode_image_b64(body.image_b64, "image_b64")   # 坏 base64→400 / 超大→413
    if img is not None:
        # 查询图**额外**做格式校验:它是查询输入,认不出格式则整个视觉理解无从谈起,
        # 与其回 200 + 一个没用上图的答案(调用方无从察觉),不如当场告诉他图有问题。
        # /ingest 的图片不加这道:那是**内容**,看不了就降级成纯文本,消息本身仍有价值。
        from ..storage.media_store import _detect_image_type
        if _detect_image_type(img[:32]) is None:
            raise HTTPException(status_code=400,
                                detail="image_b64 不是可识别的图片(支持 jpeg/png/webp/gif/heic)")
    sid = _sid(body.caller, body.session_id)
    # 快照读:不排队、不等未消费的 ingest(以当前 DB 已提交态作答)。仅用 recall_gate 限流:
    # sync 端点已在 anyio 线程内,直接在本线程跑 run_recall(不再 submit 到别池阻塞等 = 免双线程占用);
    # 闸门上限 < anyio 线程池,recall 突发不占满线程拖垮探针,与 ingest 消费池天然隔离。
    if not rt.recall_gate.try_enter():
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"error": "召回并发已满,请稍后重试"})
    # 画像注入(空画像 → 空串,行为与今天逐字节一致):full→R0/深轨,traits→R5
    _pv = ProfileStore(rt.db, ctx.user_id).current()
    p_full = render_profile(_pv.profile, mode="full") if _pv else ""
    p_traits = render_profile(_pv.profile, mode="traits") if _pv else ""
    tid = uuid.uuid4().hex   # 端到端 trace_id:langfuse trace + 日志 xrayTraceId + 响应体同 id
    try:
        with trace(tid), \
                obs.root_span("recall", user_id=ctx.user_id, session_id=sid,
                              input=body.query, trace_id=tid) as _sp:
            if _sp is not None:
                logger.info(f"recall langfuse trace_id={obs.current_trace_id()}")
            o = run_recall(rt.maas, rt.maas, ctx.atoms, ctx.cells, ctx.evidence,
                           session_id=sid, query=body.query, now_dt=now(),
                           mode=body.mode, top_k=body.top_k, reranker=rt.reranker,
                           media_store=rt._media(), mllm=rt.mllm,
                           image=img, image_content_type=body.image_content_type,
                           visual_deps=rt.visual_deps(ctx.user_id) if img else None,
                           profile_full=p_full, profile_traits=p_traits,
                           scenario=(body.context.scenario if body.context else ""))
    except httpx.HTTPStatusError as e:
        # 上游 429:客户端侧只做 1-2 次短退避重试即抛,这里原样透传——调用方按限流语义
        # 退避重试,而不是让请求在服务端深退避里挂等(并发下会拖垮 worker = 重试地狱)
        if getattr(e.response, "status_code", None) == 429:
            logger.warning(f"recall 上游限流 user={ctx.user_id} session={sid} q={body.query!r}")
            return JSONResponse(status_code=429, headers={"Retry-After": "5"},
                                content={"error": "上游模型服务限流,请稍后重试", "detail": str(e)})
        logger.exception(f"recall 失败 user={ctx.user_id} session={sid} q={body.query!r}")
        return JSONResponse(status_code=502,
                            content={"error": "记忆服务暂时不可用:上游模型/向量服务调用失败,本次召回未能完成",
                                     "detail": str(e)})
    except Exception as e:   # noqa: BLE001  上游(模型/向量)故障:各工位内部已各自降级,漏到这里=检索本身不可用
        logger.exception(f"recall 失败 user={ctx.user_id} session={sid} q={body.query!r}")
        return JSONResponse(status_code=502,
                            content={"error": "记忆服务暂时不可用:上游模型/向量服务调用失败,本次召回未能完成",
                                     "detail": str(e)})
    finally:
        rt.recall_gate.leave()          # 名额务必归还(成功/异常/早返回都经此)
    # 依据记忆:与终答同源——快链=R2 精排 top 单元的命中 atoms;
    # 深轨作答(mode=deep 直达,或 auto 升级后覆盖)= 终答引用 cells 的全部 atoms(cited 即依据)。
    # 核判判材料不足(insufficient)时不下发:半相关条目会被调用方当"答案依据"误读,
    # 无答案就诚实地空着,让 answer 里的客观交代(查了什么/结论/原因)说话。深轨负面作答
    # 也可能引用"查过"的 cells(auto 升级后 review 仍是快链判定)——同样按最终核判门禁。
    verdict = o.reviews[-1].verdict if o.reviews else None
    final_insuff = verdict == "insufficient_material"
    mem_atoms: list = []
    if o.deep and o.ans and o.ans.cited_cells and not final_insuff:
        for cid in o.ans.cited_cells:
            mem_atoms.extend(ctx.atoms.list_by_cell(cid))
    elif verdict is not None and not final_insuff:
        for h in o.ranked[:_PUBLIC_FAST_MEMORIES]:
            mem_atoms.extend(a.atom for a in h.atoms)
    memories = [_memory_view(a, ctx.evidence, rt._media()) for a in mem_atoms[:_PUBLIC_FAST_MEMORIES]
                if a is not None]
    return {
        "query": o.query, "mode": o.mode,
        "verdict": verdict,
        "critique": o.reviews[-1].critique if o.reviews else "",
        "retried": o.retried,
        "answer": o.ans.answer if o.ans else "",
        "cited_cells": o.ans.cited_cells if o.ans else [],
        "memories": memories,
        "xrayTraceId": tid,
        # 带图时回显视觉理解结果:没有它,调用方分不清"认出人后答不上来"与"根本没认出人"——
        # 两者该给用户的提示完全不同(换个问法 vs 换张清楚的照片)。
        # 这三项都是对**调用方输入**的解释,不是内部量(不含打分/prompt/深轨 steps)。
        **({"visual": {
            "faces": o.vis.faces,
            "matched": [{"character_id": m.get("character_id", ""), "name": m.get("name", "")}
                        for m in o.vis.matched],
            "resolved_query": o.vis.query,
        }} if o.vis is not None else {}),
    }


def _public_profile(cur) -> dict:
    """当前画像 → 对外结构化视图(纯函数,便于单测):剥内部量(f_id、sources=cell_id),不做整合。

    无画像(cur=None)→ {"exists": false}(调用方会话开始无脑拉一次,空画像不是错误,非 404)。
    """
    if cur is None:
        return {"exists": False, "version": 0, "traits": {}, "facts": {b: [] for b in BANDS}}
    p = cur.profile
    traits = {k: {"text": v.text, "status": v.status, "last_confirmed": v.last_confirmed}
              for k, v in p.traits.items() if v}
    facts = {b: [{"text": f.text, "last_confirmed": f.last_confirmed} for f in p.facts.get(b, [])]
             for b in BANDS}
    updated = cur.created_at
    return {
        "exists": True, "version": cur.version,
        "updated_at": updated.isoformat() if hasattr(updated, "isoformat") else str(updated),
        "traits": traits, "facts": facts,
    }


@router.get("/profile")
def get_profile(ctx: UserContext = Depends(_ctx)):
    """取本 user 的当前画像(结构化,不做整合;token→user 归属)。"""
    return _public_profile(ProfileStore(rt.db, ctx.user_id).current())


def _episode_vo(cell) -> dict:
    """memcell → 对外 episode VO(段粒度):id/会话/起止/主题/叙事/分类。不下发 payload/atoms/向量。"""
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
                  episode_type: str | None = Query(default=None, description="按分类过滤;不传=全部"),
                  start: str | None = Query(default=None, description="起始时间(ISO,含);按段起始 t_start 过滤"),
                  end: str | None = Query(default=None, description="结束时间(ISO,含)"),
                  page: int = Query(default=1, ge=1),
                  page_size: int = Query(default=20, ge=1, le=100)):
    """按 episode_type + 时间范围分页检索本 user 的 episode(段),t_start 倒序(最近在前)。

    token→user 归属;过滤项(episode_type/start/end)皆可选,分页 page/page_size 有默认(上限 100)。
    返回段粒度 VO 列表 + 分页元信息(total/page/page_size)。
    """
    offset = (page - 1) * page_size
    cells = ctx.cells.list_by_type(episode_type=episode_type, start=start, end=end,
                                   limit=page_size, offset=offset)
    total = ctx.cells.count_by_type(episode_type=episode_type, start=start, end=end)
    return {"items": [_episode_vo(c) for c in cells],
            "total": total, "page": page, "page_size": page_size}


@router.get("/trace/{node_id}")
def trace_node(node_id: str, ctx: UserContext = Depends(_ctx)):
    """按 id 溯源(只在该 user 的记忆内),记忆和证据都支持(按 id 自动分派):

    - 记忆 atom_id → 正向链:记忆 + 归属/认识状态 + 下钻到原始证据(含当时 Q↔A),`node="memory"`。
    - 证据 evidence_id → 反向链:证据原文 + 同轮 Q↔A + 被哪些记忆引用(cited_by),`node="evidence"`。
    两者都查不到 → 404。
    """
    media = rt._media()
    chain = build_trust_chain([node_id], ctx.atoms, ctx.evidence, media)   # 先当记忆查
    node = chain[0] if chain else None
    if node is not None and not node.get("missing"):
        return {"node": "memory", **node}
    ev_node = trace_evidence(node_id, ctx.evidence, ctx.atoms, media)      # 再当证据查
    if ev_node is not None:
        return ev_node
    return JSONResponse(status_code=404, content={"error": "not found", "id": node_id})
