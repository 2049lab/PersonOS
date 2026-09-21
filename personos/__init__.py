"""PersonOS: layered long-term memory for agents.

    from personos import Memory

    m = Memory()
    m.add("I moved to Shanghai in June", user_id="alice", session_id="chat-1")
    m.end_session(user_id="alice", session_id="chat-1", sync=True)
    print(m.search("where do I live?", user_id="alice").ans.answer)

add()/end_session() queue the work and return immediately; consumption runs
on a background dispatcher. sync=True (or m.flush(...)) blocks until the
session's queue has drained — use it when you need to read your own writes.

Importing this package is free: no configuration is read and no connection is
opened until a Memory is actually constructed.
"""

from __future__ import annotations

from .config import Config, get_config, load_config, set_config
from .errors import MissingCapability, PersonOSError, QueueBusy
from .memory import AddReceipt, Memory

__all__ = [
    "Memory",
    "AddReceipt",
    "Config",
    "get_config",
    "load_config",
    "set_config",
    "MissingCapability",
    "PersonOSError",
    "QueueBusy",
    "__version__",
]

__version__ = "0.1.0.dev0"
