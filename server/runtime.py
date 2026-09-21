"""The server's process-wide Memory instance.

A long-running service wants exactly one of these — one connection pool, one
set of model clients, one dispatcher. The library does not, which is why the
singleton lives here and not in the package.
"""

from __future__ import annotations

from personos.memory import Memory, UserContext

rt = Memory()

__all__ = ["rt", "Memory", "UserContext"]
