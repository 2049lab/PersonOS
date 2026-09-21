"""Shared database facade: SQLAlchemy Core, no ORM.

``Database`` dispatches on construction — ``Database()`` returns the backend the
configuration asks for — so the fifteen call sites and every ``db: Database``
annotation keep working unchanged while a second backend appears underneath.

Every statement passes through ``self._sql()`` on its way to the driver, which
is where dialect translation happens. For MySQL that is the identity function.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Iterator

from personos.storage.db.dialect import translate

# Test guard: with PERSONOS_TEST_GUARD=1, two levels of interception (production
# and scripts do not set it, so they are unaffected).
# 1) Any instance: DROP/TRUNCATE/unqualified DELETE raises outright — not even a
#    rollback scope makes those acceptable, because a rollback does not undo the
#    transactional visibility risk of wiping real data.
# 2) An instance not inside rollback_scope: every write statement is refused, so
#    that building a Database() by hand cannot quietly pollute a shared database.
#    Reads are unrestricted. The rule is enforced by the mechanism, not by memory.
_RE_DESTRUCTIVE = re.compile(r"^(DROP|TRUNCATE)\b", re.IGNORECASE)
_RE_BARE_DELETE = re.compile(r"^DELETE\s+FROM\s+[`\w.]+\s*;?\s*$", re.IGNORECASE)
_RE_WRITE = re.compile(
    r"^(INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|RENAME|GRANT|REVOKE|LOAD|CALL|MERGE)\b",
    re.IGNORECASE,
)


def _guard(sql: str, *, pinned: bool) -> None:
    if os.environ.get("PERSONOS_TEST_GUARD") != "1":
        return
    s = sql.strip()
    if _RE_DESTRUCTIVE.match(s) or _RE_BARE_DELETE.match(s):
        raise RuntimeError(
            f"test guard blocked a destructive statement: {s[:60]}... "
            "(DROP/TRUNCATE/unqualified DELETE are refused under pytest)"
        )
    if not pinned and _RE_WRITE.match(s):
        raise RuntimeError(
            f"test guard blocked a write outside a rollback scope: {s[:60]}... "
            "(tests must use the db fixture, which wraps every test in rollback_scope; "
            "a hand-built Database() is read-only under pytest)"
        )


def blob_param(b: bytes | None) -> str | None:
    """BLOB write parameter: hex text, paired with ``UNHEX(%s)`` in the SQL.

    Binary literals are not safe to interpolate through every deployment path we
    support (one database proxy mangles high bytes and NULs), so vectors travel
    as hex text in both directions. Costs 2x transfer, irrelevant at this scale,
    and it means the SQL text is identical on MySQL and SQLite.
    """
    return b.hex() if b is not None else None


def blob_of(hexval) -> bytes | None:
    """BLOB read value: the result of ``HEX(col)`` back to bytes."""
    return bytes.fromhex(hexval) if hexval else None


class _Tx:
    """Transaction executor: same method surface as Database, bound to one
    connection so that multiple statements are atomic."""

    def __init__(self, conn, dialect: str = "mysql"):
        self._conn = conn
        self._dialect = dialect

    def _sql(self, sql: str) -> str:
        return translate(sql, self._dialect)

    def fetch_one(self, sql: str, params: tuple = ()):
        return self._conn.exec_driver_sql(self._sql(sql), tuple(params)).mappings().first()

    def fetch_all(self, sql: str, params: tuple = ()):
        return self._conn.exec_driver_sql(self._sql(sql), tuple(params)).mappings().all()

    def execute(self, sql: str, params: tuple = ()) -> int:
        # Under test, _Tx only ever exists inside a rollback_scope SAVEPOINT.
        _guard(sql, pinned=True)
        return self._conn.exec_driver_sql(self._sql(sql), tuple(params)).rowcount


class Database:
    """Backend-dispatching facade.

    - fetch_one/fetch_all/execute: borrow and return a pooled connection;
      execute commits on success.
    - transaction(): multi-statement atomic block.
    - rollback_scope(): pins one connection, routes everything through it, and
      rolls back unconditionally on exit. Tests write without persisting.
    """

    DIALECT = "mysql"

    def __new__(cls, *args, **kwargs):
        if cls is Database:               # Database() -> the configured backend
            return super().__new__(_resolve_backend())
        return super().__new__(cls)

    # —— subclass hooks ——
    def _make_engine(self):
        raise NotImplementedError

    def _existing_tables(self) -> set[str]:
        raise NotImplementedError

    def _create_schema(self, conn) -> None:
        raise NotImplementedError

    def _sql(self, sql: str) -> str:
        return translate(sql, self.DIALECT)

    def __init__(self):
        self.engine = self._make_engine()
        self._pinned = None               # connection pinned by rollback_scope (tests only)
        self._init_schema()

    # —— reads ——
    def fetch_one(self, sql: str, params: tuple = ()):
        sql = self._sql(sql)
        if self._pinned is not None:
            return self._pinned.exec_driver_sql(sql, tuple(params)).mappings().first()
        with self.engine.connect() as conn:
            return conn.exec_driver_sql(sql, tuple(params)).mappings().first()

    def fetch_all(self, sql: str, params: tuple = ()):
        sql = self._sql(sql)
        if self._pinned is not None:
            return self._pinned.exec_driver_sql(sql, tuple(params)).mappings().all()
        with self.engine.connect() as conn:
            return conn.exec_driver_sql(sql, tuple(params)).mappings().all()

    # —— writes ——
    def execute(self, sql: str, params: tuple = ()) -> int:
        _guard(sql, pinned=self._pinned is not None)
        sql = self._sql(sql)
        if self._pinned is not None:
            # Pinned: do not commit; the enclosing rollback_scope undoes everything.
            return self._pinned.exec_driver_sql(sql, tuple(params)).rowcount
        with self.engine.begin() as conn:
            return conn.exec_driver_sql(sql, tuple(params)).rowcount

    @contextmanager
    def transaction(self) -> Iterator[_Tx]:
        """Atomic block; uses a SAVEPOINT when pinned so the outer rollback still wins."""
        if self._pinned is not None:
            nested = self._pinned.begin_nested()
            try:
                yield _Tx(self._pinned, self.DIALECT)
                nested.commit()
            except Exception:
                nested.rollback()
                raise
        else:
            if os.environ.get("PERSONOS_TEST_GUARD") == "1":
                raise RuntimeError(
                    "test guard: transaction() must run inside the db fixture's "
                    "rollback_scope, otherwise its writes are really committed"
                )
            with self.engine.begin() as conn:
                yield _Tx(conn, self.DIALECT)

    @contextmanager
    def rollback_scope(self) -> Iterator["Database"]:
        """Test atomicity: borrow a connection, open a transaction, pin it, and
        roll back unconditionally on exit.

        While pinned every method reuses that one connection, so a test reads its
        own uncommitted writes and none of them survive. Not reentrant; tests only.
        """
        if self._pinned is not None:
            raise RuntimeError("rollback_scope cannot be nested")
        conn = self.engine.connect()
        tx = conn.begin()
        self._pinned = conn
        try:
            yield self
        finally:
            self._pinned = None
            tx.rollback()                 # pass or fail, always roll back
            conn.close()

    def _init_schema(self) -> None:
        """Create missing tables. Idempotent; never alters an existing table."""
        from personos.storage.db.ddl import TABLE_NAMES

        missing = [t for t in TABLE_NAMES if t not in self._existing_tables()]
        if not missing:
            return
        try:
            with self.engine.begin() as conn:
                self._create_schema(conn)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"tables {missing} are missing and could not be created ({e}). "
                f"If the database account has no DDL rights, apply the schema from "
                f"`personos schema --mysql` out of band and restart."
            ) from e

    def close(self) -> None:
        self.engine.dispose()


def _resolve_backend():
    """Pick the backend from configuration: a database URL if given, else SQLite."""
    from personos.config import get_config
    from personos.storage.db.mysql import MySQLDatabase
    from personos.storage.db.sqlite import SQLiteDatabase

    cfg = get_config()
    if cfg.db_url or cfg.mysql_host:
        return MySQLDatabase
    return SQLiteDatabase
