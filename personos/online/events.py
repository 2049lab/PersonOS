"""记忆变更事件钩子。

记忆服务只负责**如实广播「什么变了」**——订阅了谁、要不要签名、往哪投递，
一概不管,那些是业务层(personos-web 控制台)的事。所以这里只有一个
进程内回调注册点,没有 HTTP、没有表、没有配置。

默认无订阅者时是零开销的空转;回调抛异常也吞掉——
通知失败绝不能让记忆写入失败。
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

# 事件名与 personos-web 的 webhook 事件集一致(对标 mem0)
EVENT_ADD = "memory.add"
EVENT_UPDATE = "memory.update"

_subscribers: list[Callable[[str, str, dict[str, Any]], None]] = []


def subscribe(fn: Callable[[str, str, dict[str, Any]], None]) -> None:
    """注册回调:fn(user_id, event, payload)。业务层在启动时挂上。"""
    _subscribers.append(fn)


def emit(user_id: str, event: str, payload: dict[str, Any]) -> None:
    """广播一个事件。没有订阅者时直接返回,不产生任何开销。"""
    if not _subscribers:
        return
    for fn in _subscribers:
        try:
            fn(user_id, event, payload)
        except Exception as e:  # noqa: BLE001  通知失败不该拖垮写入
            logger.warning(f"记忆事件回调失败 event={event} user={user_id}: {e}")
