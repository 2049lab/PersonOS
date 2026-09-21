"""Memory-change event hooks.

The memory service is only responsible for **faithfully broadcasting what changed** — who subscribed,
whether to sign, where to deliver are all none of its business; those belong to the application layer
(the personos-web console). So all that lives here is one in-process callback registry: no HTTP, no
tables, no configuration.

With no subscribers it is a zero-cost no-op by default, and exceptions from callbacks are swallowed —
a failed notification must never make a memory write fail.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

# Event names match personos-web's webhook event set (aligned with mem0)
EVENT_ADD = "memory.add"
EVENT_UPDATE = "memory.update"

_subscribers: list[Callable[[str, str, dict[str, Any]], None]] = []


def subscribe(fn: Callable[[str, str, dict[str, Any]], None]) -> None:
    """Register a callback: fn(user_id, event, payload). The application layer hooks in at startup."""
    _subscribers.append(fn)


def emit(user_id: str, event: str, payload: dict[str, Any]) -> None:
    """Broadcast one event. Returns immediately when there are no subscribers, costing nothing."""
    if not _subscribers:
        return
    for fn in _subscribers:
        try:
            fn(user_id, event, payload)
        except Exception as e:  # noqa: BLE001  a failed notification must not drag the write down with it
            logger.warning(f"memory event callback failed event={event} user={user_id}: {e}")
