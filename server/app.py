"""PersonOS FastAPI 入口:只挂对外记忆服务(service_api,前缀 /api/v1)。

运行:  uvicorn server.app:app --reload --port 8000
       python -m personos          # 生产单进程(host/port 从 env 读)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from personos.config import settings
from personos.storage.redis_client import get_redis
from server.runtime import rt
from personos.online.event_forward import install as _install_event_forward
from .response import install as _install_envelope
from .api import router as service_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """进程内启后台 ingest 调度器(消费会话队列);退出时优雅停。

    每个 pod 一个 dispatcher,扫看板把有活的会话派给 ingest 池 FIFO 消费——
    多副本下靠会话锁单飞,互不重复消费。
    """
    rt.start_dispatcher()
    logger.info("PersonOS 启动:ingest dispatcher 已就绪")
    try:
        yield
    finally:
        rt.stop_dispatcher()
        # 优雅停机:等在途 drain 作业跑完(它们持会话锁,跑完才松),取消尚未开始的排队作业。
        # 不等的话硬 kill 会把在途会话锁卡到 TTL 才释放(消息不丢,靠 proc 可靠恢复)。
        rt.ingest_exec.shutdown(wait=True, cancel_futures=True)
        # 画像整理池同理收尾:在途 consolidate 持 profile 单飞锁,跑完才松;排队的取消(下次触发自愈)。
        rt.profile_exec.shutdown(wait=True, cancel_futures=True)
        from personos import obs
        obs.flush()                                    # 冲出缓冲的 langfuse trace
        logger.info("PersonOS 停止:ingest dispatcher 已收尾")


app = FastAPI(title="PersonOS", lifespan=lifespan)


# —— 接口文档静态页:构建期由 scripts/build_api_doc.py 生成,运行时零依赖直接 serve ——
_API_DOC_HTML = ""
try:
    _API_DOC_HTML = (Path(__file__).parent / "api_doc.html").read_text(encoding="utf-8")
except OSError:
    logger.warning("api_doc.html 缺失(未构建?运行 scripts/build_api_doc.py);/ 与 /api-doc 将返回占位")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api-doc", response_class=HTMLResponse, include_in_schema=False)
def api_doc():
    """接口文档页:业务方/agent 直接访问本域名根路径即可阅读最新用法。"""
    return _API_DOC_HTML or "<h1>personos-mem</h1><p>接口文档未构建,请见 docs/personos-mem-api.md</p>"


# —— 平台探针:在 /api/v1 之外、无鉴权 ——
@app.get("/healthz")
def healthz():
    """存活探针:进程能响应即 ok,不碰 DB/外部依赖。"""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """就绪探针:MySQL + Redis(如启用)可达才收流量——写路径的硬依赖。"""
    try:
        rt.db.fetch_one("SELECT 1")
        if settings.redis_cluster:
            get_redis().ping()
        return {"status": "ready"}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "not-ready", "error": str(e)})


app.include_router(service_router)                 # 对外记忆服务:唯一 router
_install_envelope(app)                             # 全局异常处理器:框架/依赖抛出的异常也走信封(仅 /api/v1)

# 记忆变更事件转发(可选):未配 MEMORY_EVENT_URL 时不挂任何回调,零开销。
# 记忆服务只负责广播「什么变了」,订阅/签名/投递都在业务层。
_install_event_forward()
