"""Langfuse(公司 xray)LLM 可观测埋点:薄封装,禁用时全无操作、零副作用。

设计原则:pk/sk 未配 → 客户端 None,所有 span/observation 上下文管理器退化为空操作
(不上报、不影响主流程)。**埋点只加观测,绝不改变业务返回或异常路径**——init/开 span/
update/flush 任一失败都吞掉并降级,主链路照常。

用法:
    with obs.root_span("recall", user_id=u, session_id=s, input=q) as span:
        with obs.observation("chat", model=m, input=msgs) as gen:
            resp = llm.chat(...)
            obs.update(gen, output=resp, usage={"input": pt, "output": ct, "total": tt})
"""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Optional

from loguru import logger

from .config import settings

_lock = threading.Lock()
_inited = False
_client = None   # Langfuse | None(懒建)

# 当前 LLM 阶段名(R0/R5/R3'/W1/W2…),供 MaasClient 给 generation span 命名。
# contextvar 沿同线程调用栈传播:stage() 在某阶段入口设,栈下游的 maas.chat 读到。
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("lf_stage", default="")


@contextmanager
def stage(name: str):
    """标注当前 LLM 阶段(嵌套调用会临时覆盖,退出还原)。"""
    tok = _stage.set(name or "")
    try:
        yield
    finally:
        _stage.reset(tok)


def current_stage() -> str:
    return _stage.get()


# 当前请求的 trace_id(与 langfuse trace 同 id),供日志 patcher 写进 xrayTraceId 把一次请求
# 的所有日志(跨 ingest/recall 各 worker 线程)串起来。contextvar 随 copy_context 传播到子线程。
_req_trace: contextvars.ContextVar[str] = contextvars.ContextVar("req_trace", default="")


def request_trace_id() -> str:
    return _req_trace.get()


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """保 contextvars 的线程池:submit 时捕获提交线程的 context(含 OTel 活动 span + stage),
    worker 里以 copy_context().run 执行 → 并行子任务的 LLM span 正确 nest 到父 trace、stage
    随之传播。标准库 ThreadPoolExecutor 默认丢弃提交线程的 contextvars,会把并行 LLM 调用
    甩成孤儿根 trace(见 weave 织写)。公司 redinfra 只有 asyncio 版(async_to_sync),不适用同步 fan-out。"""

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)


def client():
    """懒建 langfuse 客户端单例;pk/sk 未配 → None(埋点全关)。init 失败也降级为 None。"""
    global _inited, _client
    if _inited:
        return _client
    with _lock:
        if _inited:
            return _client
        _inited = True
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.info("langfuse 未配置(无 pk/sk),LLM 埋点关闭")
            return _client
        try:
            from langfuse import Langfuse
            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                environment=settings.langfuse_environment or None,
                release=settings.langfuse_release or None,
            )
            logger.info(f"langfuse 已启用 env={settings.langfuse_environment} host={settings.langfuse_host}")
        except Exception as e:   # noqa: BLE001  埋点初始化失败绝不拖垮服务
            logger.warning(f"langfuse 初始化失败,埋点关闭: {e}")
            _client = None
    return _client


def _set_attrs(span, user_id, session_id, *, as_root: bool) -> None:
    """标 as_root(上游不接 langfuse 时链路入口必须)+ 挂 user/session。失败忽略。"""
    try:
        if as_root:
            span._otel_span.set_attribute("langfuse.internal.as_root", "true")   # 必须字符串
        if user_id:
            span._otel_span.set_attribute("langfuse.user.id", str(user_id))
        if session_id:
            span._otel_span.set_attribute("langfuse.session.id", str(session_id))
    except Exception:   # noqa: BLE001
        pass


@contextmanager
def root_span(name: str, *, user_id=None, session_id=None, input: Any = None,
              metadata: Any = None, trace_id: Optional[str] = None):
    """请求入口根 span(recall/ingest/profile/深轨)。标 as_root。禁用/失败时 yield None。

    入口先把 OTel 上下文重置为空再开 span:线程池(ingest_exec / FastAPI anyio)复用线程时,
    上一任务遗留的 span 上下文会串扰后一任务,使其嵌套观测丢父 → 甩成孤儿根 trace。从空上下文
    起,本任务的 attach/detach 自成一栈,既保证自己是真根、又清掉残留,recall/ingest/profile 通吃。

    trace_id(可选,32 hex):端到端贯穿——入口生成、经队列透传;传入则 langfuse trace 用该 id,
    且日志 xrayTraceId 也用它(即便 langfuse 禁用/开 span 失败也灌进 _req_trace,保证日志可查)。
    """
    c = client()

    @contextmanager
    def _bind(tid: str):
        """把请求 trace_id 绑到 _req_trace(供 _xray_patcher 写日志 xrayTraceId);空则不绑。"""
        tok = None
        try:
            if tid:
                tok = _req_trace.set(tid)
            yield
        finally:
            if tok is not None:
                try:
                    _req_trace.reset(tok)
                except Exception:   # noqa: BLE001
                    pass

    if c is None:                                  # langfuse 关:仍灌 trace_id 让日志带上
        with _bind(trace_id or ""):
            yield None
        return
    reset_token = None
    try:
        from opentelemetry import context as _otel_ctx
        reset_token = _otel_ctx.attach(_otel_ctx.Context())   # 干净空上下文,隔离线程残留
    except Exception:   # noqa: BLE001  OTel 不可用时退化:不重置,行为同前
        reset_token = None
    try:
        kw = {"name": name, "input": input, "metadata": metadata}
        if trace_id:
            kw["trace_context"] = {"trace_id": trace_id}   # 令 langfuse trace 用传入 id
        cm = c.start_as_current_span(**kw)
    except Exception as e:   # noqa: BLE001
        logger.debug(f"langfuse root_span 失败(忽略): {e}")
        if reset_token is not None:
            try:
                from opentelemetry import context as _otel_ctx
                _otel_ctx.detach(reset_token)
            except Exception:   # noqa: BLE001
                pass
        with _bind(trace_id or ""):
            yield None
        return
    try:
        with cm as span:
            _set_attrs(span, user_id, session_id, as_root=True)
            with _bind(trace_id or current_trace_id() or ""):
                yield span
    finally:
        if reset_token is not None:
            try:
                from opentelemetry import context as _otel_ctx
                _otel_ctx.detach(reset_token)
            except Exception:   # noqa: BLE001
                pass


@contextmanager
def observation(name: str, *, as_type: str = "generation", model: Optional[str] = None,
                input: Any = None, metadata: Any = None):
    """一次 LLM/工具/检索观测(嵌在当前 span 下)。禁用/失败时 yield None。"""
    c = client()
    if c is None:
        yield None
        return
    try:
        cm = c.start_as_current_observation(as_type=as_type, name=name, model=model,
                                            input=input, metadata=metadata)
    except Exception as e:   # noqa: BLE001
        logger.debug(f"langfuse observation 失败(忽略): {e}")
        yield None
        return
    with cm as obs:
        yield obs


def update(obs, *, output: Any = None, usage: Optional[dict] = None,
           level: Optional[str] = None, status_message: Optional[str] = None) -> None:
    """回填观测 output/token(usage 值须数字)/错误。obs 为 None 时无操作。绝不抛。"""
    if obs is None:
        return
    try:
        kw: dict = {}
        if output is not None:
            kw["output"] = output
        if usage:
            kw["usage_details"] = usage
        if level:
            kw["level"] = level
        if status_message:
            kw["status_message"] = status_message
        if kw:
            obs.update(**kw)
    except Exception:   # noqa: BLE001
        pass


def current_trace_id() -> Optional[str]:
    """当前活跃 trace 的 id(供日志/回给调用方去 xray 检索)。禁用/无活跃 → None。"""
    c = client()
    if c is None:
        return None
    try:
        return c.get_current_trace_id()
    except Exception:   # noqa: BLE001
        return None


def flush() -> None:
    """退出前把缓冲的 trace 冲出去(短命进程/请求收尾必调)。"""
    c = client()
    if c is not None:
        try:
            c.flush()
        except Exception:   # noqa: BLE001
            pass
