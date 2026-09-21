"""Entry-layer id validation and caller-scoped session rewriting (pure functions, no
runtime dependency, unit-testable offline).

Two jobs:
- **id hygiene**: user_id / session_id / caller are all caller-controlled strings that
  get concatenated into Redis keys and MySQL rows. Redis keys use ":" as the segment
  separator (seg/lock keys, redis_client.key); if ":" were allowed inside a segment,
  (u="a:b", s="c") and (u="a", s="b:c") would build the same key = cross-user
  read/write into an unclosed segment (a P1 finding from the pre-launch security
  review). We block that here at the entry point with a character allowlist;
  redis_client.key adds segment escaping as a fallback (defense in depth, see its
  docstring).
- **caller isolation**: different callers (agents integrating with the memory service)
  may each use a duplicate session_id like "chat-001". At the entry point we rewrite
  session_id to f"{caller}:{session_id}"; everything downstream (write/recall/
  session_context/trace) only ever sees the rewritten id and notices no difference.
  An absent caller means no rewriting (backward compatible with older callers).
"""

from __future__ import annotations

import re

# caller: the calling party's identifier, a short machine-readable token (goes into the
# session_id prefix and into a MySQL column)
RE_CALLER = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Raw session_id: chosen by the client; ":" is banned (Redis key separator) and so is
# "/" (ambiguous in object-storage keys and paths)
RE_SESSION = re.compile(r"^[A-Za-z0-9_.-]{1,95}$")
# user_id: optionally chosen by the client at registration; matches MySQL VARCHAR(128)
RE_USER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Max total length of the rewritten session_id: caller (<=32) + ":" + raw (<=95) = 128,
# matching the column width
_SCOPED_MAX = 128


def valid_user_id(user_id: str | None) -> str | None:
    """Validate a client-chosen user_id; raise ValueError if invalid. None/blank means
    "generate one automatically" and is valid."""
    uid = (user_id or "").strip()
    if not uid:
        return None
    if not RE_USER.match(uid):
        raise ValueError(
            f"user_id allows only letters/digits/underscore/hyphen, length 1-128 "
            f"(no colon, slash, etc.): {uid[:40]!r}")
    return uid


def scoped_session(caller: str, session_id: str) -> str:
    """Caller scoping: non-empty caller -> f"{caller}:{session_id}"; empty -> unchanged
    (backward compatible).

    Validate the character sets of caller and the raw session_id first (invalid input
    raises ValueError, which the entry layer turns into a 400), then prepend the
    prefix — so the id everything downstream receives is guaranteed unambiguous and
    fits the Redis key / MySQL column width.
    """
    caller = (caller or "").strip()
    if caller and not RE_CALLER.match(caller):
        raise ValueError(
            f"caller allows only letters/digits/underscore/hyphen, length 1-32: {caller[:40]!r}")
    if not RE_SESSION.match(session_id or ""):
        raise ValueError(
            f"session_id allows only letters/digits/underscore/hyphen/dot, length 1-95 "
            f"(no colon): {(session_id or '')[:40]!r}")
    sid = f"{caller}:{session_id}" if caller else session_id
    if len(sid) > _SCOPED_MAX:
        raise ValueError(f"caller+session_id total length exceeds {_SCOPED_MAX}: {len(sid)}")
    return sid
