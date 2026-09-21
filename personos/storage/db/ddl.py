"""Schema: one source of truth, written in MySQL syntax, rendered per backend.

The table definitions below are the authority. ``scripts/mysql_schema.sql`` used
to be a hand-maintained second copy of the same thing, which is exactly the kind
of duplication that drifts silently; it is now generated from here.

SQLite gets the same schema through :func:`to_sqlite`, a deterministic rewrite
rather than a second hand-written definition, for the same reason.

Design: filterable fields are real columns with indexes, everything else lives in
a JSON ``payload`` column. Vectors get their own BLOB column. Timestamps are ISO
strings in VARCHAR, so lexical order equals chronological order on every backend.
"""

from __future__ import annotations

import re

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
    # —— 视频身份层三表 ——
    # 此前只存在于设计文档、靠手工在实例上建,于是任何新部署(以及 SQLite)都缺表。
    # 结构取自线上实际表,不是照抄文档:文档会过期,实例不会。
    """
    CREATE TABLE IF NOT EXISTS characters (
        id            VARCHAR(64) NOT NULL,
        user_id       VARCHAR(128) NOT NULL DEFAULT '',
        is_wearer     TINYINT NOT NULL DEFAULT 0,
        primary_name  VARCHAR(128) DEFAULT NULL,
        status        VARCHAR(16) NOT NULL DEFAULT 'active',
        merged_into   VARCHAR(64) DEFAULT NULL,
        face_tau      DOUBLE NOT NULL DEFAULT 0,
        voice_tau     DOUBLE NOT NULL DEFAULT 0,
        payload       MEDIUMTEXT,
        created_at    VARCHAR(64) NOT NULL,
        updated_at    VARCHAR(64) NOT NULL,
        PRIMARY KEY (id),
        KEY idx_char_user (user_id, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS character_assets (
        id            VARCHAR(64) NOT NULL,
        user_id       VARCHAR(128) NOT NULL DEFAULT '',
        character_id  VARCHAR(64) NOT NULL,
        kind          VARCHAR(16) NOT NULL,
        quality       DOUBLE NOT NULL DEFAULT 0,
        embedding     MEDIUMBLOB,
        dim           INT DEFAULT NULL,
        status        VARCHAR(16) NOT NULL DEFAULT 'active',
        payload       MEDIUMTEXT,
        created_at    VARCHAR(64) NOT NULL,
        PRIMARY KEY (id),
        KEY idx_asset_char (user_id, character_id, kind, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS character_cloud (
        id            VARCHAR(64) NOT NULL,
        user_id       VARCHAR(128) NOT NULL DEFAULT '',
        character_id  VARCHAR(64) NOT NULL,
        modality      VARCHAR(16) NOT NULL,
        slot          VARCHAR(16) NOT NULL,
        embedding     MEDIUMBLOB,
        dim           INT DEFAULT NULL,
        payload       MEDIUMTEXT,
        updated_at    VARCHAR(64) NOT NULL,
        PRIMARY KEY (id),
        KEY idx_cloud_char (user_id, character_id, modality, slot)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

_TABLE_NAMES = ("users", "evidence", "atoms", "atom_chains", "memcells",
                "session_context", "tasks", "profile_versions",
                "characters", "character_assets", "character_cloud")


# Conflict targets for INSERT ... ON DUPLICATE KEY UPDATE. SQLite requires the
# target to be named explicitly, MySQL infers it; this map is what lets the
# dialect layer translate the three upsert statements in the store layer without
# any store having to know which backend it is talking to.
TABLE_PK: dict[str, tuple[str, ...]] = {
    "users": ("token",),
    "evidence": ("id",),
    "atoms": ("id",),
    "atom_chains": ("id",),
    "memcells": ("id",),
    "session_context": ("user_id", "session_id"),
    "tasks": ("id",),
    "profile_versions": ("id",),
    "characters": ("id",),
    "character_assets": ("id",),
    "character_cloud": ("id",),
}

TABLE_NAMES = _TABLE_NAMES

# MySQL column types that SQLite does not know, and what they become. SQLite is
# dynamically typed, so these are mostly about being parseable, not about storage.
_TYPE_MAP = [
    (r"\bMEDIUMTEXT\b", "TEXT"),
    (r"\bLONGTEXT\b", "TEXT"),
    (r"\bMEDIUMBLOB\b", "BLOB"),
    (r"\bLONGBLOB\b", "BLOB"),
    (r"\bVARCHAR\(\d+\)", "TEXT"),
    (r"\bDOUBLE\b", "REAL"),
    (r"\bTINYINT\b", "INTEGER"),
    (r"\bBIGINT\b", "INTEGER"),
    (r"\bINT\b", "INTEGER"),
    (r"\bDATETIME\b", "TEXT"),
]
_RE_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", re.IGNORECASE)
_RE_KEY = re.compile(r"^\s*KEY\s+(\w+)\s*\(([^)]*)\)\s*,?\s*$", re.IGNORECASE | re.MULTILINE)
_RE_UNIQUE_KEY = re.compile(r"^\s*UNIQUE KEY\s+\w+\s*\(([^)]*)\)", re.IGNORECASE | re.MULTILINE)
_RE_ENGINE = re.compile(r"\)\s*ENGINE=\w+[^\n]*", re.IGNORECASE)
_RE_AUTOINC = re.compile(r"\bBIGINT\s+NOT NULL\s+AUTO_INCREMENT\b", re.IGNORECASE)


def to_sqlite(ddl: str) -> list[str]:
    """Rewrite one MySQL CREATE TABLE into SQLite statements.

    Returns the CREATE TABLE followed by its indexes: MySQL declares secondary
    indexes inline, SQLite needs separate CREATE INDEX statements, so a single
    input becomes several outputs.
    """
    table = _RE_TABLE.search(ddl)
    name = table.group(1) if table else ""

    # AUTO_INCREMENT only works on SQLite for INTEGER PRIMARY KEY, and it has to
    # be declared on the column rather than in a separate PRIMARY KEY clause.
    autoinc = bool(_RE_AUTOINC.search(ddl))

    indexes: list[str] = []
    for m in _RE_KEY.finditer(ddl):
        idx_name, cols = m.group(1), m.group(2).strip()
        indexes.append(f"CREATE INDEX IF NOT EXISTS {name}_{idx_name} ON {name} ({cols})")
    out = _RE_KEY.sub("", ddl)                          # inline KEY -> separate statements
    out = _RE_UNIQUE_KEY.sub(lambda m: f"    UNIQUE ({m.group(1).strip()})", out)
    for pat, repl in _TYPE_MAP:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    out = _RE_ENGINE.sub(")", out)
    if autoinc:
        out = re.sub(r"\bid\s+INTEGER\s+NOT NULL\s+AUTO_INCREMENT\b",
                     "id INTEGER PRIMARY KEY AUTOINCREMENT", out, flags=re.IGNORECASE)
        out = re.sub(r",?\s*PRIMARY KEY\s*\(id\)", "", out, flags=re.IGNORECASE)
    out = re.sub(r",(\s*)\)", r"\1)", out)               # trailing comma left by removals
    out = re.sub(r"\n\s*\n+", "\n", out)
    return [out.strip(), *indexes]


def sqlite_ddl() -> list[str]:
    """The whole schema, ready to execute on SQLite."""
    return [stmt for ddl in _DDL for stmt in to_sqlite(ddl)]


def mysql_schema_sql() -> str:
    """Render the checked-in .sql file, so it can never drift from _DDL."""
    head = ("-- Generated by `personos schema --mysql`. Do not edit by hand:\n"
            "-- the source of truth is personos/storage/db/ddl.py.\n\n")
    return head + "\n".join(d.strip().rstrip() + ";\n" for d in _DDL)
