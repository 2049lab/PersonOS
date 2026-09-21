"""The guard mechanism proving itself: three red lines under PERSONOS_TEST_GUARD, each of
which raises RuntimeError when crossed.

These three are what mechanically guarantees "tests leave no residue" (see the header comment
in conftest):
1. A self-constructed Database() that has not entered a rollback scope is read-only — writes
   are rejected outright;
2. Even inside a rollback scope (the normal fixture path), DROP/TRUNCATE/bare DELETE are still
   refused;
3. On the normal path (injecting the db fixture) reads and writes work as usual, and the scope
   rolls back automatically on exit.

Anyone writing a test that tries to bypass the convention gets a clear error telling them what
to do, rather than silent pollution.
"""

import pytest

from personos.storage.db import Database


def test_unpinned_database_is_read_only():
    """A self-constructed Database() that never entered rollback_scope is read-only inside the
    test process: writes are rejected, reads still work."""
    other = Database()
    try:
        assert other.fetch_one("SELECT 1 AS v")["v"] == 1   # reads are unrestricted
        with pytest.raises(RuntimeError, match="outside a rollback scope"):
            other.execute(
                "INSERT INTO users(token, user_id, created_at) VALUES(%s,%s,%s)",
                ("tok_guard", "guard-probe", "2026-09-02T00:00:00+00:00"),
            )
        with pytest.raises(RuntimeError, match="rollback_scope"):
            with other.transaction() as tx:
                tx.execute("DELETE FROM atoms WHERE user_id=%s", ("nonexistent",))
    finally:
        other.close()


def test_destructive_sql_blocked_even_inside_scope(db: Database):
    """Once inside a rollback scope (the db fixture), ordinary writes are available, but
    DROP/TRUNCATE/bare DELETE are always refused."""
    db.execute(
        "INSERT INTO users(token, user_id, created_at) VALUES(%s,%s,%s)",
        ("tok_in_scope", "guard-in-scope", "2026-09-02T00:00:00+00:00"),
    )
    with pytest.raises(RuntimeError, match="destructive statement"):
        db.execute("DROP TABLE atoms")
    with pytest.raises(RuntimeError, match="destructive statement"):
        db.execute("TRUNCATE TABLE atoms")
    with pytest.raises(RuntimeError, match="destructive statement"):
        db.execute("DELETE FROM atoms")           # whole-table DELETE with no WHERE
    # A DELETE carrying a WHERE clause is not restricted.
    db.execute("DELETE FROM users WHERE token=%s", ("tok_in_scope",))


def test_guard_probe_leaves_no_trace(db: Database):
    """The guard test itself also leaves zero residue: the probes above are invisible in the
    database afterwards."""
    row = db.fetch_one("SELECT COUNT(*) AS n FROM users WHERE user_id LIKE %s", ("guard-%",))
    assert row["n"] == 0
