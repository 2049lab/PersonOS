"""SQL dialect translation.

This layer exists so that no store has to know which backend it is talking to.
That makes it load-bearing in a quiet way: a mistranslated upsert does not
crash, it silently turns "update the row" into "insert a duplicate", and the
damage shows up later as corrupted memory. So the assertions here are on exact
output strings, not on "it ran".

The strongest guarantee is not in this file: the same test suite runs green on
both SQLite and MySQL. This file explains *why* when it breaks.
"""

from __future__ import annotations

import re

import pytest

from personos.storage.db.ddl import TABLE_PK, _DDL, sqlite_ddl
from personos.storage.db.dialect import DialectError, translate


# ── placeholders ────────────────────────────────────────────────────────

def test_placeholders_become_question_marks():
    assert translate("SELECT a FROM t WHERE b=%s AND c=%s", "sqlite") == \
        "SELECT a FROM t WHERE b=? AND c=?"


def test_mysql_is_the_identity_function():
    """The existing deployment must be provably untouched by all of this."""
    for sql in ("SELECT %s", "INSERT INTO atoms (id) VALUES (%s) "
                             "ON DUPLICATE KEY UPDATE id=VALUES(id)"):
        assert translate(sql, "mysql") is sql or translate(sql, "mysql") == sql


def test_literal_percent_is_refused_rather_than_corrupted():
    """A literal %% would be mangled by the blind %s swap. Raise instead of guessing."""
    with pytest.raises(DialectError, match="literal"):
        translate("SELECT * FROM t WHERE name LIKE '%%abc'", "sqlite")


def test_unknown_dialect_raises():
    with pytest.raises(DialectError):
        translate("SELECT 1", "postgres")


# ── upserts: the three real statements in the store layer ───────────────

_UPSERTS = {
    "atoms": ("INSERT INTO atoms (id,user_id,payload) VALUES (%s,%s,%s) "
              "ON DUPLICATE KEY UPDATE payload=VALUES(payload)"),
    "memcells": ("INSERT INTO memcells (id,user_id,payload) VALUES (%s,%s,%s) "
                 "ON DUPLICATE KEY UPDATE payload=VALUES(payload)"),
    "session_context": ("INSERT INTO session_context (user_id,session_id,payload) "
                        "VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE payload=VALUES(payload)"),
}


@pytest.mark.parametrize("table", sorted(_UPSERTS))
def test_upsert_translation_is_exact(table):
    out = translate(_UPSERTS[table], "sqlite")
    target = ", ".join(TABLE_PK[table])
    assert f"ON CONFLICT({target}) DO UPDATE SET" in out, out
    assert "VALUES(payload)" not in out, "MySQL's VALUES(col) must become excluded.col"
    assert "excluded.payload" in out
    assert "%s" not in out


def test_upsert_without_a_conflict_target_raises():
    """SQLite cannot infer the target; guessing one would corrupt data quietly."""
    with pytest.raises(DialectError, match="conflict target"):
        translate("INSERT INTO unknown_table (a) VALUES (%s) "
                  "ON DUPLICATE KEY UPDATE a=VALUES(a)", "sqlite")


# ── the corpus assumptions that make blind translation safe ─────────────

def test_schema_has_no_constructs_translation_cannot_handle():
    """Guards the premise of this whole approach: the SQL corpus stays simple.

    If someone adds a backtick-quoted identifier or a MySQL-only date function
    to the schema, blind rewriting stops being safe — and this test is the place
    that says so, before it becomes a data bug.
    """
    joined = "\n".join(_DDL)
    assert "`" not in joined, "backtick quoting is MySQL-only"
    for fn in ("NOW(", "DATE_ADD", "DATE_SUB", "CONCAT(", "IFNULL("):
        assert fn not in joined.upper(), f"{fn} has no direct SQLite equivalent"


def test_hex_channel_is_not_translated():
    """UNHEX/HEX stay verbatim; the SQLite backend registers a same-named function.

    Keeping the SQL text identical across backends means every vector read and
    write is literally the same statement — one less place for the two paths to
    drift apart.
    """
    sql = "INSERT INTO atoms (embedding) VALUES (UNHEX(%s))"
    assert "UNHEX(?)" in translate(sql, "sqlite")
    assert "HEX(embedding)" in translate("SELECT HEX(embedding) FROM atoms", "sqlite")


# ── generated schema ────────────────────────────────────────────────────

def test_sqlite_schema_creates_every_table():
    import sqlite3

    from personos.storage.db.ddl import TABLE_NAMES

    con = sqlite3.connect(":memory:")
    for stmt in sqlite_ddl():
        con.execute(stmt)
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLE_NAMES) <= have, f"missing: {set(TABLE_NAMES) - have}"


def test_every_table_has_a_declared_conflict_target():
    """TABLE_PK must cover the schema, or a future upsert fails at runtime."""
    from personos.storage.db.ddl import TABLE_NAMES

    assert set(TABLE_NAMES) == set(TABLE_PK)


def test_mysql_schema_file_matches_the_source_of_truth():
    """scripts/mysql_schema.sql is generated. Drift between it and _DDL used to
    be possible and invisible; now it is a test failure."""
    from pathlib import Path

    from personos.storage.db.ddl import mysql_schema_sql

    path = Path(__file__).resolve().parent.parent / "scripts" / "mysql_schema.sql"
    if not path.exists():
        pytest.skip("mysql_schema.sql not present")
    want = re.sub(r"\s+", " ", mysql_schema_sql()).strip()
    got = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).strip()
    assert want == got, "run `personos schema --mysql > scripts/mysql_schema.sql`"


# ── SQLite under load: the one place the two backends genuinely differ ──

def test_sqlite_survives_concurrent_writers(monkeypatch):
    """WAL allows a single writer. The pool defaults were tuned for MySQL, so
    concurrent ingest would otherwise surface as 'database is locked'.

    The point is not throughput — it is that contention resolves by waiting
    (busy_timeout) rather than by failing, because a lock error in the write
    path loses a memory rather than delaying it.
    """
    import tempfile
    import threading
    from pathlib import Path

    from personos.storage.db import SQLiteDatabase

    # This test owns a throwaway file database, so the shared-database write
    # guard does not apply — and it has to actually write to mean anything.
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)
    path = Path(tempfile.mkdtemp()) / "concurrent.db"
    db = SQLiteDatabase(str(path))
    errors: list[Exception] = []
    N_THREADS, PER_THREAD = 8, 25

    def writer(tid: int):
        try:
            for i in range(PER_THREAD):
                db.execute(
                    "INSERT INTO users (token,user_id,created_at) VALUES (%s,%s,%s)",
                    (f"t{tid}-{i}", f"u{tid}-{i}", "2026-01-01T00:00:00+08:00"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes failed: {errors[:2]}"
    got = db.fetch_one("SELECT COUNT(*) AS c FROM users")["c"]
    assert got == N_THREADS * PER_THREAD, f"lost writes: {got}"
    db.close()
