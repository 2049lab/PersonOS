"""MySQL backend. Dialect translation is the identity function here, which is
what makes this path provably unchanged by the SQLite work."""

from __future__ import annotations

from sqlalchemy import create_engine

from personos.config import get_config
from personos.storage.db.base import Database


class MySQLDatabase(Database):
    DIALECT = "mysql"

    def _make_engine(self):
        cfg = get_config()
        if not cfg.db_url:
            from personos.errors import MissingCapability

            raise MissingCapability(
                "MySQL storage",
                "no database URL is configured",
                "set PERSONOS_DB_URL, e.g. "
                "mysql+pymysql://user:pass@host:3306/personos?charset=utf8mb4")
        url = cfg.db_url
        # pool_pre_ping: check liveness on checkout. Proxies and managed MySQL
        # both drop idle connections, and the failure is otherwise a confusing
        # error on the first query after a quiet period.
        # The pool has to be wide enough for the ingest/recall/profile thread
        # pools, or threads block on checkout and the pool sizes become fiction.
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=cfg.db_pool_size,
            max_overflow=cfg.db_max_overflow,
            pool_timeout=10,          # never hang a thread forever waiting for a connection
            connect_args={"connect_timeout": 10},
        )

    def _existing_tables(self) -> set[str]:
        rows = self.fetch_all(
            "SELECT table_name AS t FROM information_schema.tables WHERE table_schema = %s",
            (self.engine.url.database,),
        )
        return {r["t"] for r in rows}

    def _create_schema(self, conn) -> None:
        from personos.storage.db.ddl import _DDL

        for ddl in _DDL:
            conn.exec_driver_sql(ddl)
