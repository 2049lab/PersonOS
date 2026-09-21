"""MySQL(RedHub) 连接池与建表。

设计取向:可查询字段单列出来建索引,其余整体塞进 payload(JSON),
既能高效过滤又不用在模型演进时频繁改表。向量单独存 embedding 列。

多租户:所有业务表带 user_id 列;user 隔离不在每条业务查询里穿参,
而是【store 实例按 user 绑定】(构造注入 user_id,内部 SQL 自动带过滤)
——隔离逻辑收敛在 store 层一处,业务代码零改动。

MySQL 方言注意:
- 占位符一律 %s(pymysql 客户端插值);语句文本里出现字面 % 必须写成 %%(现有 SQL 无)。
- 时间列存 ISO 字符串(VARCHAR):字典序=时间序,与原 sqlite 行为逐位一致。
- 连接参数 pool_pre_ping/pool_recycle 是 RedHub 代理空闲断连的防线(参照 wallace-workspace)。
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from personos.config import settings

# 7 张表;索引内联进 CREATE TABLE(MySQL 无 CREATE INDEX IF NOT EXISTS)。
# 类型映射:id/token/sha256/holder/memcell_id→VARCHAR(64);user_id/session_id(外部输入)→VARCHAR(128);
# payload/summary→MEDIUMTEXT(16MB);embedding/centroid→MEDIUMBLOB(4096 维 float32=16KB);covered→INT。
_DDL = [
    # 对外令牌表:token 注册签发,调用方持久保存
    """
    CREATE TABLE IF NOT EXISTS users (
        token       VARCHAR(64) NOT NULL,
        user_id     VARCHAR(128) NOT NULL,
        created_at  VARCHAR(64) NOT NULL,
        PRIMARY KEY (token),
        UNIQUE KEY uq_users_user (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence (
        id          VARCHAR(64) NOT NULL,
        user_id     VARCHAR(128) NOT NULL DEFAULT '',
        sha256      VARCHAR(64),
        holder      VARCHAR(64),
        modality    VARCHAR(16),
        captured_at VARCHAR(64),
        payload     MEDIUMTEXT NOT NULL,
        embedding   MEDIUMBLOB,
        PRIMARY KEY (id),
        KEY idx_evidence_user_sha (user_id, sha256)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS atoms (
        id           VARCHAR(64) NOT NULL,
        user_id      VARCHAR(128) NOT NULL DEFAULT '',
        memcell_id   VARCHAR(64),
        chain_id     VARCHAR(64),
        prev_atom_id VARCHAR(64),
        next_atom_id VARCHAR(64),
        object_type  VARCHAR(32),
        holder       VARCHAR(64),
        recorded_at  VARCHAR(64),
        updated_at   VARCHAR(64),
        payload      MEDIUMTEXT NOT NULL,
        embedding    MEDIUMBLOB,
        PRIMARY KEY (id),
        KEY idx_atoms_cell (user_id, memcell_id),
        KEY idx_atoms_chain (user_id, chain_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # atom 链(派生视图,只分组不消解):链级信息;成员关系在 atoms 三列上(一 atom 至多一链)。
    # 存量库加列走 scripts/mysql_schema.sql 的 ALTER(本 DDL 仅全新库生效,_init_schema 只查表存在性)。
    """
    CREATE TABLE IF NOT EXISTS atom_chains (
        id           VARCHAR(64) NOT NULL,
        user_id      VARCHAR(128) NOT NULL DEFAULT '',
        n_atoms      INT NOT NULL DEFAULT 0,
        head_atom_id VARCHAR(64),
        tail_atom_id VARCHAR(64),
        centroid     MEDIUMBLOB,
        payload      MEDIUMTEXT NOT NULL,
        created_at   VARCHAR(64) NOT NULL,
        updated_at   VARCHAR(64) NOT NULL,
        PRIMARY KEY (id),
        KEY idx_chains_user (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # MemCell:一段话题对话的加工产物。topic 向量单独列存,段粒度检索面。
    """
    CREATE TABLE IF NOT EXISTS memcells (
        id              VARCHAR(64) NOT NULL,
        user_id         VARCHAR(128) NOT NULL DEFAULT '',
        session_id      VARCHAR(128),
        t_start         VARCHAR(64),
        t_end           VARCHAR(64),
        episode_type    VARCHAR(64) NOT NULL DEFAULT 'unknown',
        payload         MEDIUMTEXT NOT NULL,
        topic_embedding MEDIUMBLOB,
        PRIMARY KEY (id),
        KEY idx_cells_session (user_id, session_id),
        KEY idx_cells_type (user_id, episode_type, t_start)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 会话对话历史的滚动压缩缓存(派生物,非真相源;evidence 仍原样保留)。
    # 每会话单条、覆盖式:summary=已压缩的更早历史,covered=已折进 summary 的"轮"数(水位)。
    # 主键 (user_id, session_id):同会话名在不同用户下互不相干。
    """
    CREATE TABLE IF NOT EXISTS session_context (
        user_id     VARCHAR(128) NOT NULL DEFAULT '',
        session_id  VARCHAR(128) NOT NULL,
        summary     MEDIUMTEXT NOT NULL,
        covered     INT NOT NULL DEFAULT 0,
        updated_at  VARCHAR(64),
        PRIMARY KEY (user_id, session_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 异步任务登记簿(202+轮询):之前在 rt.tasks 内存 dict,重部署/多副本即丢。
    # 生命周期短;done/error 行由清扫按 updated_at 过期删(MySQL 无原生 TTL)。
    # idx(status,updated_at) 供僵尸任务收割(running 超时判 worker lost)。
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id     VARCHAR(64) NOT NULL,
        kind        VARCHAR(32) NOT NULL,
        user_id     VARCHAR(128) NOT NULL,
        session_id  VARCHAR(128) NOT NULL,
        status      VARCHAR(16) NOT NULL,
        result      MEDIUMTEXT,
        error       TEXT,
        created_at  VARCHAR(64) NOT NULL,
        updated_at  VARCHAR(64) NOT NULL,
        PRIMARY KEY (task_id),
        KEY idx_tasks_user (user_id),
        KEY idx_tasks_status (status, updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 用户画像版本:每次 consolidate 整版留档;当前画像 = 该 user 最新一版(单行读,无一致性问题)。
    # profile_json 是全量画像(traits + facts,含 f_id);up_to_cell_id 是消费到的 cell 游标(下次取其后)。
    # 画像是派生视图(可从 cell 重蒸馏),user_id 强隔离——严禁跨用户串信息。
    """
    CREATE TABLE IF NOT EXISTS profile_versions (
        id            VARCHAR(64) NOT NULL,
        user_id       VARCHAR(128) NOT NULL,
        version       INT NOT NULL,
        profile_json  MEDIUMTEXT NOT NULL,
        up_to_cell_id VARCHAR(64) NOT NULL DEFAULT '',
        created_at    DATETIME NOT NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uq_user_version (user_id, version),
        KEY idx_user_created (user_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

_TABLE_NAMES = ("users", "evidence", "atoms", "atom_chains", "memcells",
                "session_context", "tasks", "profile_versions")

# 测试护栏:PERSONOS_TEST_GUARD=1 时两级拦截(生产/脚本不设此变量,零影响)。
# 1) 任何实例:DROP/TRUNCATE/无 WHERE 全表 DELETE 直接 RuntimeError——就算进了
#    回退域也不放行(回退救不了误清真实数据的事务可见性风险,干脆禁止)。
# 2) 未进 rollback_scope 的实例:一切写语句(INSERT/UPDATE/DELETE/CREATE/...)拒绝
#    ——绕过 conftest 的 db fixture 自建 Database() 时,写不进共享库,
#    静默污染变成响亮报错(读不受限)。规范由机制强制,不靠自觉。
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
            f"测试护栏拦截破坏性 SQL: {s[:60]}..."
            "(pytest 下禁止 DROP/TRUNCATE/无 WHERE 的全表 DELETE,共享库不做任何持久清理)"
        )
    if not pinned and _RE_WRITE.match(s):
        raise RuntimeError(
            f"测试护栏拦截未回退域的写: {s[:60]}... "
            "(测试写必须走 conftest 的 db fixture——autouse 已包进 rollback_scope;"
            "自建 Database() 只读,写会被本护栏拒绝,防止污染共享 SIT 库)"
        )


def blob_param(b: bytes | None) -> str | None:
    """BLOB 写参:hex 文本,配合 SQL 里的 UNHEX(%s)。

    RedHub 代理对 pymysql 客户端插值出的 _binary'...' 字面量非 binary-safe
    (高字节/NUL 会被截断或解析报错 RHC-4500),向量等二进制一律走 hex 文本通道
    (传输量 ×2,P0 规模可忽略),读回用 HEX() 列函数 + blob_of()。
    """
    return b.hex() if b is not None else None


def blob_of(hexval) -> bytes | None:
    """BLOB 读值:HEX() 结果(大写 hex 串)→ bytes。"""
    return bytes.fromhex(hexval) if hexval else None


class _Tx:
    """事务执行器:与 Database 同签名的方法,绑定同一条连接,保证多语句原子。"""

    def __init__(self, conn):
        self._conn = conn

    def fetch_one(self, sql: str, params: tuple = ()):
        return self._conn.exec_driver_sql(sql, tuple(params)).mappings().first()

    def fetch_all(self, sql: str, params: tuple = ()):
        return self._conn.exec_driver_sql(sql, tuple(params)).mappings().all()

    def execute(self, sql: str, params: tuple = ()) -> int:
        _guard(sql, pinned=True)  # 测试态下 _Tx 只存在于 rollback_scope 的 SAVEPOINT 内,写必回退
        return self._conn.exec_driver_sql(sql, tuple(params)).rowcount


class Database:
    """MySQL 访问薄 facade:SQLAlchemy Core 引擎 + PyMySQL,不用 ORM。

    - fetch_one/fetch_all/execute:短借短还(池化,线程安全),execute 成功自动 commit。
    - transaction():多语句原子块。
    - rollback_scope():测试专用——钉住一条连接,所有读写走它,退出无条件 rollback,
      测试内写不落库(零污染)。生产路径永不 pin。
    """

    def __init__(self):
        url = URL.create(
            "mysql+pymysql",
            username=settings.mysql_user,
            password=settings.mysql_password,
            host=settings.mysql_host,
            port=settings.mysql_port,
            database=settings.mysql_database,
            query={"charset": "utf8mb4"},
        )
        # pool_pre_ping:借出前探活,断连自动重连(RedHub 代理空闲杀连接的防线)
        # 连接池须撑得起 50 宽的 ingest/recall/profile 线程池,否则线程卡在连接借出上、
        # 池大小成虚设。env 可调:总连接 ≈ (pool_size+max_overflow) × 副本数,拉高前确认
        # RedHub 代理 per-instance 连接预算,别把共享库打到 too many connections。
        pool_size = int(os.environ.get("PERSONOS_DB_POOL_SIZE", "32"))
        max_overflow = int(os.environ.get("PERSONOS_DB_MAX_OVERFLOW", "32"))
        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=10,        # 借不到连接最多等 10s 即抛,不无限挂住线程
            connect_args={"connect_timeout": 10},
        )
        self._pinned = None  # rollback_scope 钉住的连接(仅测试,单线程使用)
        self._init_schema()

    # —— 读 ——
    def fetch_one(self, sql: str, params: tuple = ()):
        if self._pinned is not None:
            return self._pinned.exec_driver_sql(sql, tuple(params)).mappings().first()
        with self.engine.connect() as conn:
            return conn.exec_driver_sql(sql, tuple(params)).mappings().first()

    def fetch_all(self, sql: str, params: tuple = ()):
        if self._pinned is not None:
            return self._pinned.exec_driver_sql(sql, tuple(params)).mappings().all()
        with self.engine.connect() as conn:
            return conn.exec_driver_sql(sql, tuple(params)).mappings().all()

    # —— 写 ——
    def execute(self, sql: str, params: tuple = ()) -> int:
        _guard(sql, pinned=self._pinned is not None)
        if self._pinned is not None:
            # 钉住时不 commit:由外层 rollback_scope 统一回退
            return self._pinned.exec_driver_sql(sql, tuple(params)).rowcount
        with self.engine.begin() as conn:
            return conn.exec_driver_sql(sql, tuple(params)).rowcount

    @contextmanager
    def transaction(self) -> Iterator[_Tx]:
        """多语句原子块;pinned 状态下用 SAVEPOINT,不破坏外层回退。"""
        if self._pinned is not None:
            nested = self._pinned.begin_nested()
            try:
                yield _Tx(self._pinned)
                nested.commit()
            except Exception:
                nested.rollback()
                raise
        else:
            if os.environ.get("PERSONOS_TEST_GUARD") == "1":
                raise RuntimeError(
                    "测试护栏:transaction() 写块必须走 conftest 的 db fixture(rollback_scope),"
                    "未回退域的写会被真实提交——请注入 db fixture 而非自建 Database()"
                )
            with self.engine.begin() as conn:
                yield _Tx(conn)

    @contextmanager
    def rollback_scope(self) -> Iterator["Database"]:
        """测试原子性:借一条连接开启事务并 pin,测试结束无条件 rollback。

        pin 后本对象所有方法(fetch/execute/transaction)都复用这条连接:
        读得到自己未提交的写(等价原 sqlite :memory: 语义),退出时全部回退,
        共享库零污染(Django TestCase 同款模式)。不可嵌套;仅测试单线程使用。
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
            tx.rollback()  # 成败皆回退——这就是"原子测试"的全部
            conn.close()

    def _init_schema(self) -> None:
        """幂等建表:7 表齐则跳过;缺表则补建,无 DDL 权限时报错指路(不抛晦涩的代理错误)。

        只查表存在性不查列——存量 atoms 缺链三列时须走 mysql_schema.sql 的 ALTER,
        本方法不自动改既有表结构。
        """
        rows = self.fetch_all(
            "SELECT table_name AS t FROM information_schema.tables WHERE table_schema = %s",
            (self.engine.url.database,),
        )
        have = {r["t"] for r in rows}
        missing = [t for t in _TABLE_NAMES if t not in have]
        if not missing:
            return
        try:
            with self.engine.begin() as conn:
                for ddl in _DDL:
                    conn.exec_driver_sql(ddl)
        except Exception as e:  # noqa: BLE001  典型:personos 账号 DML-only,建表须走 DMS 工单
            raise RuntimeError(
                f"缺表 {missing},且当前账号建表失败({e})。"
                f"请拿 scripts/mysql_schema.sql 走 DMS 工单建表后重启。"
            ) from e

    def close(self) -> None:
        self.engine.dispose()
