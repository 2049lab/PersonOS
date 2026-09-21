"""Storage engine.

``Database()`` returns whichever backend the configuration selects: SQLite by
default, MySQL when a database URL or host is set. Everything the store layer
imported from the old single-module ``db`` is re-exported here, so no store
changed when the second backend arrived.
"""

from __future__ import annotations

from personos.storage.db.base import Database, _guard, _Tx, blob_of, blob_param
from personos.storage.db.ddl import TABLE_NAMES, mysql_schema_sql, sqlite_ddl
from personos.storage.db.dialect import DialectError, translate
from personos.storage.db.mysql import MySQLDatabase
from personos.storage.db.sqlite import SQLiteDatabase

# Kept for the handful of modules that referenced the private name.
_TABLE_NAMES = TABLE_NAMES

__all__ = [
    "Database",
    "MySQLDatabase",
    "SQLiteDatabase",
    "DialectError",
    "TABLE_NAMES",
    "blob_param",
    "blob_of",
    "translate",
    "sqlite_ddl",
    "mysql_schema_sql",
    "_Tx",
    "_guard",
    "_TABLE_NAMES",
]
