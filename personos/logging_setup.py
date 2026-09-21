"""结构化日志落盘 + trace-id 贯穿一次请求(dev 原则 §3)。

用 loguru 的 contextualize 注入 trace_id,而非 patcher——后者在 enqueue 下会丢
上下文(此前 pipecat 项目踩过)。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

_configured = False
_reinstalled = False   # 文件 sink 补挂只做一次(redinfra 掀翻是 import 期一次性动作)

_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <5}</level> "
    "| <cyan>{extra[trace_id]}</cyan> | <level>{message}</level>"
)


def _file_sink(log_dir: Path) -> dict:
    """按天轮转的文件 sink 配置;setup 与补挂共用一份,行为不漂移。"""
    return dict(sink=log_dir / "personos_{time:YYYY-MM-DD}.log",
                format=_FMT, level="DEBUG",
                rotation="00:00", retention="14 days", encoding="utf-8")


def _redinfra_ready() -> bool:
    """redinfra 是否已初始化日志(已装它的 stdout xray sink)。未装/不可用 → False。"""
    try:
        import redinfra.log.logger as _rl
        return bool(getattr(_rl, "_initialized", False))
    except Exception:   # noqa: BLE001
        return False


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """幂等初始化:控制台 + 按天轮转的文件 sink,格式含 trace_id。

    关键:若 redinfra 已先于本函数装好 stdout xray sink(某些 pod 启动序会在 app 之前
    auto-init),**绝不 logger.remove() 掀翻它**——否则容器内业务日志再也进不了 xray
    (redinfra init 幂等不会重装)。此时只补文件 sink,stdout 交给 redinfra。
    """
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(extra={"trace_id": "-"})  # 缺省 trace_id,避免 file sink 格式 KeyError
    if _redinfra_ready():
        logger.add(**_file_sink(log_dir))       # 保留 redinfra 的 stdout sink,只补落盘
    else:
        logger.remove()                         # 本地/正常序:自建 stderr+file;redinfra 若后跑会自行重配 stdout
        logger.add(sys.stderr, format=_FMT, level=level)
        logger.add(**_file_sink(log_dir))
    _configured = True


def _xray_patcher(record: dict) -> None:
    """loguru 全局 patcher:先跑 redinfra 的 inject_context(填 cat/userId 等),再用本请求的
    trace_id 覆盖 xrayTraceId——让一次请求跨 ingest/recall 各 worker 线程的日志共用一个 id,
    日志中心据此把请求串起来,且与 langfuse trace 同 id 可互跳。绝不抛。"""
    try:
        from redinfra.log.context_injector import inject_context
        inject_context(record)
    except Exception:   # noqa: BLE001  redinfra 不可用:至少保证下面几个键存在
        for k in ("xrayTraceId", "catRootId", "catParentId", "catMsgId", "userId"):
            record["extra"].setdefault(k, "")
    try:
        from . import obs
        tid = obs.request_trace_id()
        if tid:
            record["extra"]["xrayTraceId"] = tid
    except Exception:   # noqa: BLE001
        pass


def reinstall_file_sink(log_dir: Path) -> None:
    """Redis 接入后补挂文件 sink + 安装带请求 trace_id 的 patcher(幂等,只补一次)。

    redinfra.redis.pool 在 import 时执行 init_logger() → logger.remove() 掀翻进程里
    全部 loguru sink、并把 patcher 设成它自己的 inject_context(xrayTraceId 恒空,因我们不走
    thrift)。建池后调用本函数:①补回文件 sink(落盘);②把 patcher 换成 _xray_patcher,
    给每条日志灌本请求的 trace_id。容器内平台采集走 redinfra 的 stdout sink,两侧并存。
    """
    global _reinstalled
    if _reinstalled or not _configured:
        return
    _reinstalled = True
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(**_file_sink(log_dir))
    logger.configure(patcher=_xray_patcher)     # 覆盖 redinfra 的 patcher(handlers 不动)


@contextmanager
def trace(trace_id: str):
    """把一次请求的 trace_id 绑到该作用域内所有日志。"""
    with logger.contextualize(trace_id=trace_id):
        yield
