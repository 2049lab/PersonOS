"""把记忆变更事件转发给业务层(可选)。

默认不启用:未配 MEMORY_EVENT_URL 时 subscribe 都不会挂,零开销。
配了才转发——所以这段代码对只跑记忆服务的人完全透明。

为什么放在记忆服务里:事件源在这里。但它只知道「有个 URL 要通知」,
不知道 webhook、订阅关系、签名给谁看——那些都在业务层。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from typing import Any

import httpx
from loguru import logger

from .events import subscribe

_URL = os.environ.get("MEMORY_EVENT_URL", "")
_SECRET = os.environ.get("INTERNAL_EVENT_SECRET", "")
_TIMEOUT = httpx.Timeout(5.0, connect=2.0)


def _post(user_id: str, event: str, payload: dict[str, Any]) -> None:
    body = json.dumps(
        {"user_id": user_id, "event": event, "payload": payload}, ensure_ascii=False
    ).encode()
    sig = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    try:
        httpx.post(
            _URL,
            content=body,
            headers={"Content-Type": "application/json", "X-Internal-Signature": sig},
            timeout=_TIMEOUT,
        )
    except Exception as e:  # noqa: BLE001  业务层挂了不该影响记忆写入
        logger.warning(f"记忆事件转发失败 event={event}: {e}")


def install() -> None:
    """在服务启动时调用。未配置则什么都不做。"""
    if not _URL or not _SECRET:
        return
    subscribe(
        lambda uid, ev, pl: threading.Thread(
            target=_post, args=(uid, ev, pl), daemon=True
        ).start()
    )
    logger.info(f"记忆事件转发已启用 → {_URL}")
