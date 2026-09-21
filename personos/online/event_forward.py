"""Forward memory-change events to the application layer (optional).

Off by default: without MEMORY_EVENT_URL configured, no subscriber is even registered, so the cost
is zero. Forwarding only happens once it is configured — which makes this code completely invisible
to anyone running just the memory service.

Why it lives in the memory service: the events originate here. But all it knows is "there is a URL
to notify" — it knows nothing about webhooks, subscription relationships, or who the signature is
meant for; those all live in the application layer.
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
    except Exception as e:  # noqa: BLE001  the application layer being down must not affect memory writes
        logger.warning(f"memory event forwarding failed event={event}: {e}")


def install() -> None:
    """Call this at service startup. Does nothing when not configured."""
    if not _URL or not _SECRET:
        return
    subscribe(
        lambda uid, ev, pl: threading.Thread(
            target=_post, args=(uid, ev, pl), daemon=True
        ).start()
    )
    logger.info(f"memory event forwarding enabled -> {_URL}")
