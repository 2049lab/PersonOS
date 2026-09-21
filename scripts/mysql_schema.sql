-- ============================================================================
-- PersonOS 建表 DDL(共 7 张表,按顺序依次执行)
--
-- 用法:DMS 工单 / 平台控制台,目标库选 personos,逐条执行下面的 CREATE TABLE。
--      全部带 IF NOT EXISTS,重复执行安全(已建的表会跳过)。
-- 来源:与 personos/storage/db.py 的 _DDL 逐条一致;表建好后代码启动时
--      会查 information_schema 自动跳过建表,无需再改代码。
-- 注意:【存量库】atoms 已存在,链三列走文末的 ALTER(见「存量库迁移」),不要重跑 CREATE。
-- ============================================================================

-- (若控制台已选库,本行可省略)
USE personos;

-- ----------------------------------------------------------------------------
-- 1/6 用户注册表:token ↔ user 映射,多租户入口
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    token       VARCHAR(64)  NOT NULL COMMENT '对外令牌(注册签发,调用方持久保存)',
    user_id     VARCHAR(128) NOT NULL COMMENT '用户标识(注册时指定或自动生成,全局唯一)',
    created_at  VARCHAR(64)  NOT NULL COMMENT '注册时间(ISO 8601 字符串)',
    PRIMARY KEY (token),
    UNIQUE KEY uq_users_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户注册表(token↔user)';

-- ----------------------------------------------------------------------------
-- 2/6 证据表:append-only 不可变,唯一真相源
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence (
    id          VARCHAR(64)  NOT NULL COMMENT '证据 id(ULID)',
    user_id     VARCHAR(128) NOT NULL DEFAULT '' COMMENT '归属用户(''为兼容旧数据的默认命名空间)',
    sha256      VARCHAR(64)  NULL COMMENT '内容哈希(去重/幂等键)',
    holder      VARCHAR(64)  NULL COMMENT '说话人(user/assistant/人名)',
    modality    VARCHAR(16)  NULL COMMENT '模态(text/image/audio/video)',
    captured_at VARCHAR(64)  NULL COMMENT '发生时间(ISO 8601 字符串,字典序=时间序)',
    payload     MEDIUMTEXT   NOT NULL COMMENT '完整 EvidenceRecord JSON',
    embedding   MEDIUMBLOB   NULL COMMENT 'float32 向量(语义兜底检索)',
    PRIMARY KEY (id),
    KEY idx_evidence_user_sha (user_id, sha256)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='证据表(append-only 真相源)';

-- ----------------------------------------------------------------------------
-- 3/7 原子表:检索/记忆单元,upsert 语义
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS atoms (
    id           VARCHAR(64)  NOT NULL COMMENT '原子 id(ULID)',
    user_id      VARCHAR(128) NOT NULL DEFAULT '' COMMENT '归属用户',
    memcell_id   VARCHAR(64)  NULL COMMENT '归属 MemCell id',
    chain_id     VARCHAR(64)  NULL COMMENT '归属 atom 链 id(游离为 NULL;唯一事实源)',
    prev_atom_id VARCHAR(64)  NULL COMMENT '链上前驱(链首为 NULL)',
    next_atom_id VARCHAR(64)  NULL COMMENT '链上后继(链尾为 NULL)',
    object_type  VARCHAR(32)  NULL COMMENT '对象类型(事实/偏好/事件/说法等)',
    holder       VARCHAR(64)  NULL COMMENT '归属人(记忆讲的是谁的事)',
    recorded_at  VARCHAR(64)  NULL COMMENT '发生时间(ISO 8601 字符串)',
    updated_at   VARCHAR(64)  NULL COMMENT '入库/修订时间(ISO 8601 字符串)',
    payload      MEDIUMTEXT   NOT NULL COMMENT '完整 MemoryAtom JSON(链字段为展示副本)',
    embedding    MEDIUMBLOB   NULL COMMENT 'float32 向量(检索面)',
    PRIMARY KEY (id),
    KEY idx_atoms_cell (user_id, memcell_id),
    KEY idx_atoms_chain (user_id, chain_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='记忆原子表(检索单元)';

-- ----------------------------------------------------------------------------
-- 4/7 atom 链表:同「事情」原子事实的时间线(派生视图,只分组不消解)
-- 成员关系在 atoms 三列上(一 atom 至多一链),本表只存链级信息 + 质心。
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS atom_chains (
    id           VARCHAR(64)  NOT NULL COMMENT '链 id(ULID,chn_ 前缀)',
    user_id      VARCHAR(128) NOT NULL DEFAULT '' COMMENT '归属用户',
    n_atoms      INT          NOT NULL DEFAULT 0 COMMENT '成员原子数(链行冗余计数,recompute 对齐)',
    head_atom_id VARCHAR(64)  NULL COMMENT '链首原子 id',
    tail_atom_id VARCHAR(64)  NULL COMMENT '链尾原子 id(新成员唯一追加点)',
    centroid     MEDIUMBLOB   NULL COMMENT '成员向量均值(判链预筛的候选面)',
    payload      MEDIUMTEXT   NOT NULL COMMENT '完整 ChainInfo JSON(title/溯源)',
    created_at   VARCHAR(64)  NOT NULL COMMENT '建链时间(ISO 8601 字符串)',
    updated_at   VARCHAR(64)  NOT NULL COMMENT '最近追加时间(ISO 8601 字符串)',
    PRIMARY KEY (id),
    KEY idx_chains_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='atom 链表(派生视图,TRUNCATE-rebuild 可重建)';

-- ----------------------------------------------------------------------------
-- 5/7 记忆单元表:一段话题对话的加工产物(段粒度组织单元)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memcells (
    id              VARCHAR(64)  NOT NULL COMMENT 'cell id(ULID)',
    user_id         VARCHAR(128) NOT NULL DEFAULT '' COMMENT '归属用户',
    session_id      VARCHAR(128) NULL COMMENT '来源会话 id',
    t_start         VARCHAR(64)  NULL COMMENT '段起始时间(ISO 8601 字符串)',
    t_end           VARCHAR(64)  NULL COMMENT '段结束时间(ISO 8601 字符串)',
    episode_type    VARCHAR(64)  NOT NULL DEFAULT 'unknown' COMMENT 'episode 分类(调用方 task_type 词表选一;未分类=unknown)',
    payload         MEDIUMTEXT   NOT NULL COMMENT '完整 MemCell JSON(topic/episode/证据引用)',
    topic_embedding MEDIUMBLOB   NULL COMMENT 'topic 一句话的 float32 向量',
    PRIMARY KEY (id),
    KEY idx_cells_session (user_id, session_id),
    KEY idx_cells_type (user_id, episode_type, t_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='记忆单元表(段粒度组织)';

-- 存量库迁移(2026-09-14,episode 分类上线):memcells 加 episode_type 列 + 索引。
-- ALTER TABLE memcells
--     ADD COLUMN episode_type VARCHAR(64) NOT NULL DEFAULT 'unknown' AFTER t_end,
--     ADD KEY idx_cells_type (user_id, episode_type, t_start);

-- ----------------------------------------------------------------------------
-- 6/7 会话上下文表:对话历史滚动压缩缓存(派生物,可覆盖)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_context (
    user_id     VARCHAR(128) NOT NULL DEFAULT '' COMMENT '归属用户',
    session_id  VARCHAR(128) NOT NULL COMMENT '会话 id',
    summary     MEDIUMTEXT   NOT NULL COMMENT '已压缩的更早历史',
    covered     INT          NOT NULL DEFAULT 0 COMMENT '已折进 summary 的轮数(水位)',
    updated_at  VARCHAR(64)  NULL COMMENT '最近更新时间(ISO 8601 字符串)',
    PRIMARY KEY (user_id, session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话上下文缓存(滚动压缩)';

-- ----------------------------------------------------------------------------
-- 7/7 异步任务登记簿:202+轮询模式的任务状态(ingest / session-end)
-- 之前在 rt.tasks 内存 dict 里——重部署/多副本即丢(轮询 404);落库后任意副本可查。
-- 生命周期短(秒级),done/error 行由清扫任务按 updated_at 过期删除( MySQL 无原生 TTL)。
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    task_id     VARCHAR(64)  NOT NULL COMMENT '任务 id(uuid4.hex)',
    kind        VARCHAR(32)  NOT NULL COMMENT '任务类型(ingest/session-end)',
    user_id     VARCHAR(128) NOT NULL COMMENT '归属用户(轮询鉴权:只能查自己的单)',
    session_id  VARCHAR(128) NOT NULL COMMENT '来源会话 id',
    status      VARCHAR(16)  NOT NULL COMMENT '状态(pending/running/done/error)',
    result      MEDIUMTEXT   NULL COMMENT '完成产物 JSON(边界判定/闭合 cell 摘要)',
    error       TEXT         NULL COMMENT '失败原因(error 态)',
    created_at  VARCHAR(64)  NOT NULL COMMENT '创建时间(ISO 8601 字符串)',
    updated_at  VARCHAR(64)  NOT NULL COMMENT '最近状态变更时间(僵尸任务收割/过期清理依据)',
    PRIMARY KEY (task_id),
    KEY idx_tasks_user (user_id),
    KEY idx_tasks_status (status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='异步任务登记簿(202+轮询)';

-- ----------------------------------------------------------------------------
-- 8/8 用户画像版本:每次 consolidate 整版留档;当前画像 = 该 user 最新一版(单行读)。
-- 画像是 atom 层之上的派生视图,可从 cell 重蒸馏;user_id 强隔离,严禁跨用户串信息。
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS profile_versions (
    id            VARCHAR(64)  NOT NULL COMMENT '版本 id(pv_ULID)',
    user_id       VARCHAR(128) NOT NULL COMMENT '归属用户(强隔离)',
    version       INT          NOT NULL COMMENT '该 user 内单调递增版本号',
    profile_json  MEDIUMTEXT   NOT NULL COMMENT '全量画像(traits + facts,含 f_id)',
    up_to_cell_id VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '本版消费到的最后一个 cell 游标(下次取其后新 cell)',
    created_at    DATETIME     NOT NULL COMMENT '出版本时间',
    PRIMARY KEY (id),
    UNIQUE KEY uq_user_version (user_id, version),
    KEY idx_user_created (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户画像版本(consolidate 整版留档)';

-- ============================================================================
-- 存量库迁移(2026-09-03,atom 链上线):atoms 加链三列 + 索引。
-- 仅对【已存在 atoms 表】的库执行;全新库走上面的 CREATE,勿重复加列。
-- 全部为 ADD COLUMN/ADD KEY,可重复提交前先查列存在性;只影响新增列,不动存量数据。
-- ============================================================================
-- ALTER TABLE atoms
--     ADD COLUMN chain_id     VARCHAR(64) NULL AFTER memcell_id COMMENT '归属 atom 链 id(游离为 NULL;唯一事实源)',
--     ADD COLUMN prev_atom_id VARCHAR(64) NULL AFTER chain_id COMMENT '链上前驱(链首为 NULL)',
--     ADD COLUMN next_atom_id VARCHAR(64) NULL AFTER prev_atom_id COMMENT '链上后继(链尾为 NULL)',
--     ADD KEY idx_atoms_chain (user_id, chain_id);
--
-- (SIT 已由代码侧重放执行,勿再提交;此处留档供 prod 等其他环境走 DMS 工单)

-- ============================================================================
-- 建完后自查(应返回 7):
-- ============================================================================
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'personos'
--   AND table_name IN ('users','evidence','atoms','atom_chains','memcells','session_context','tasks');
--
-- 存量库另查链三列(应各返回 1 行):
-- SELECT column_name FROM information_schema.columns
-- WHERE table_schema = 'personos' AND table_name = 'atoms'
--   AND column_name IN ('chain_id','prev_atom_id','next_atom_id');
--
-- 备注若报 "Specified key was too long"(767 字节限制):
-- 把对应复合索引改成前缀索引,如 KEY idx_cells_session (user_id(64), session_id(64)),
-- 告知我即可,代码无需改动。
-- ============================================================================
