"""SQL dialect translation, applied at the database layer.

The store layer writes MySQL. Rather than teach every store about two dialects,
each statement is rewritten on its way to the driver. This is a deliberate
trade-off, and the reasoning matters more than the code:

There are 234 ``%s`` placeholders across eleven files, and the SQL is *not* all
in module-level constants — ``identity/store.py`` alone has 78 of them written
inline in method bodies. Branching per store would mean roughly a hundred edits
scattered through code that sits right next to the memory algorithms. That is
the highest-risk way to make a purely mechanical change: every one of those
edits is a chance to alter behaviour while claiming to be reshaping packaging.

Translating here is ~60 lines, unit-testable in isolation, and leaves every
store byte-for-byte unchanged. The MySQL path is the identity function, so the
existing deployment is provably unaffected.

The usual objection to string rewriting is fragility. It is bounded here: the
SQL corpus is closed and small, and ``test_dialect.py`` asserts it stays that
way (no literal ``%``, no backticks, no MySQL-only date functions).
"""

from __future__ import annotations

import re
from functools import lru_cache

from personos.storage.db.ddl import TABLE_PK

MYSQL = "mysql"
SQLITE = "sqlite"

_RE_UPSERT = re.compile(
    r"INSERT\s+INTO\s+(?P<table>\w+)(?P<body>.*?)"
    r"ON\s+DUPLICATE\s+KEY\s+UPDATE\s+(?P<updates>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_RE_VALUES_FN = re.compile(r"VALUES\s*\(\s*(\w+)\s*\)", re.IGNORECASE)


class DialectError(ValueError):
    """A statement this layer cannot translate faithfully. Never guess — raise."""


def _upsert_to_on_conflict(m: re.Match) -> str:
    table, body, updates = m.group("table"), m.group("body"), m.group("updates")
    pk = TABLE_PK.get(table.lower())
    if not pk:
        raise DialectError(
            f"upsert on table {table!r} has no conflict target: add it to TABLE_PK in ddl.py. "
            "SQLite cannot infer the target the way MySQL does.")
    # MySQL's VALUES(col) means "the value that would have been inserted";
    # SQLite spells the same thing excluded.col
    converted = _RE_VALUES_FN.sub(lambda v: f"excluded.{v.group(1)}", updates)
    return (f"INSERT INTO {table}{body}ON CONFLICT({', '.join(pk)}) "
            f"DO UPDATE SET {converted}")


@lru_cache(maxsize=4096)
def translate(sql: str, dialect: str) -> str:
    """Rewrite a MySQL statement for the target dialect. Identity for MySQL."""
    if dialect == MYSQL:
        return sql
    if dialect != SQLITE:
        raise DialectError(f"unknown dialect {dialect!r}")

    # A literal percent would survive as %% in MySQL and break the naive
    # placeholder swap below. The corpus has none, and a test keeps it that way,
    # but assert rather than silently corrupt a statement if that ever changes.
    if "%%" in sql:
        raise DialectError(
            "statement contains a literal %% which placeholder translation cannot "
            f"handle safely: {sql[:120]}")

    out = _RE_UPSERT.sub(_upsert_to_on_conflict, sql)
    # pymysql uses %s; sqlite3 uses ?. Safe as a blind replace precisely because
    # of the check above. Note UNHEX/HEX are deliberately *not* translated: the
    # SQLite backend registers a same-named function instead, so the SQL text of
    # every vector read and write stays identical across backends.
    return out.replace("%s", "?")
