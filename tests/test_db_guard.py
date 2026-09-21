"""护栏机制自证:PERSONOS_TEST_GUARD 下的三条红线,违反即 RuntimeError。

这三条是"测试零污染"的机制保证(见 conftest 头注释):
1. 未进回退域的自建 Database() 只读——写语句直接拒绝;
2. 就算进了回退域(正常 fixture 路径),DROP/TRUNCATE/裸 DELETE 也不放行;
3. 正常路径(注入 db fixture)读写照常,退出自动回退。

任何人写测试想绕过规范,得到的不是静默污染,而是清晰的报错指引。
"""

import pytest

from personos.storage.db import Database


def test_unpinned_database_is_read_only():
    """自建 Database()(未进 rollback_scope)在测试进程里只读:写拒绝、读正常。"""
    other = Database()
    try:
        assert other.fetch_one("SELECT 1 AS v")["v"] == 1   # 读不受限
        with pytest.raises(RuntimeError, match="未回退域的写"):
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
    """进回退域(db fixture)后常规写可用,但 DROP/TRUNCATE/裸 DELETE 恒拒绝。"""
    db.execute(
        "INSERT INTO users(token, user_id, created_at) VALUES(%s,%s,%s)",
        ("tok_in_scope", "guard-in-scope", "2026-09-02T00:00:00+00:00"),
    )
    with pytest.raises(RuntimeError, match="破坏性 SQL"):
        db.execute("DROP TABLE atoms")
    with pytest.raises(RuntimeError, match="破坏性 SQL"):
        db.execute("TRUNCATE TABLE atoms")
    with pytest.raises(RuntimeError, match="破坏性 SQL"):
        db.execute("DELETE FROM atoms")           # 无 WHERE 全表 DELETE
    db.execute("DELETE FROM users WHERE token=%s", ("tok_in_scope",))  # 带 WHERE 的删不受限


def test_guard_probe_leaves_no_trace(db: Database):
    """护栏测试自身也零污染:上述探针在库中不可见。"""
    row = db.fetch_one("SELECT COUNT(*) AS n FROM users WHERE user_id LIKE %s", ("guard-%",))
    assert row["n"] == 0
