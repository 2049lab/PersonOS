"""SQLite backend — the zero-configuration default.

Two things make this viable without touching any retrieval code: vector search
is already a full scan with numpy rather than a database feature, and the store
layer's SQL goes through dialect translation instead of growing branches.

The one real limitation is write concurrency. SQLite in WAL mode allows one
writer at a time, so the wide ingest/recall pools that make sense against MySQL
will queue here. That is stated at startup rather than left to be discovered.
"""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from personos.config import get_config
from personos.storage.db.base import Database

_BUSY_TIMEOUT_MS = 30_000


class SQLiteDatabase(Database):
    DIALECT = "sqlite"

    def __init__(self, path: str | None = None):
        self._path = path
        super().__init__()

    def _make_engine(self):
        cfg = get_config()
        path = self._path or str(cfg.sqlite_path)
        if path != ":memory:":
            cfg.data_dir.mkdir(parents=True, exist_ok=True)

        # :memory: MUST use StaticPool. With a normal pool every checkout gets a
        # *different* empty database, which does not error — it silently loses
        # every write. That failure mode is quiet enough to burn an afternoon.
        engine = create_engine(
            f"sqlite+pysqlite:///{path}",
            poolclass=StaticPool if path == ":memory:" else None,
            connect_args={"check_same_thread": False, "timeout": _BUSY_TIMEOUT_MS / 1000},
        )

        # pysqlite ships a legacy transaction mode: it opens transactions
        # implicitly, commits before DDL, and as a result SAVEPOINT does not
        # nest correctly. Our rollback_scope relies on SAVEPOINT (transaction()
        # uses begin_nested), so without this the writes inside a nested block
        # survive the outer rollback and leak into the next test — which is
        # exactly how it presented: 60 tests failing in the suite, every one of
        # them passing alone. This is SQLAlchemy's documented remedy: take
        # transaction control away from the driver and issue BEGIN ourselves.
        @event.listens_for(engine, "connect")
        def _disable_implicit_begin(dbapi_conn, _record):
            dbapi_conn.isolation_level = None

        @event.listens_for(engine, "begin")
        def _explicit_begin(conn):
            conn.exec_driver_sql("BEGIN")

        @event.listens_for(engine, "connect")
        def _configure(dbapi_conn, _record):
            # UNHEX is a MySQL function. Registering it here rather than
            # translating the SQL keeps every vector read and write textually
            # identical across backends — one less thing that can diverge.
            # Do not rely on SQLite's own unhex(): it only exists from 3.41.
            dbapi_conn.create_function(
                "unhex", 1, lambda h: bytes.fromhex(h) if h else None)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")     # readers do not block the writer
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return engine

    def _existing_tables(self) -> set[str]:
        rows = self.fetch_all("SELECT name AS t FROM sqlite_master WHERE type='table'")
        return {r["t"] for r in rows}

    def _create_schema(self, conn) -> None:
        from personos.storage.db.ddl import sqlite_ddl

        for stmt in sqlite_ddl():
            conn.exec_driver_sql(stmt)
