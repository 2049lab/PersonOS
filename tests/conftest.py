"""测试夹具:共享 SIT 库 + 原子回退(零污染)。

规范由机制强制(personos/storage/db.py 的 PERSONOS_TEST_GUARD 护栏),不靠自觉:
1. 每个测试被 autouse 的 `db` fixture 包进 Database.rollback_scope()——
   测试内所有写退出时无条件回退,库不留痕迹。固定 id / 固定 user_id 都安全。
   新写测试无需任何 opt-in,天然原子。
2. 自建 Database() 在测试进程里【只读】:未进回退域的 INSERT/UPDATE/DELETE/
   transaction() 一律 RuntimeError(写不进共享库,静默污染变成响亮报错)。
3. DROP/TRUNCATE/无 WHERE 全表 DELETE 在任何路径下都 RuntimeError——
   就算进了回退域也不放行。
"""

import os

# 必须在 import personos 之前设好(护栏作用于 Database.execute/_Tx.execute)
os.environ["PERSONOS_TEST_GUARD"] = "1"
# 测试强制关 langfuse:置空 pk/sk(load_dotenv override=False,不覆盖已存在的 env)——
# 否则 .env 里的密钥会让单测真连 SIT langfuse 上报,污染平台 + 拖慢。
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
# 单测钉死假身份后端:PERSONOS_VIDEO_BACKEND 的**缺省值是 real**(生产正确优先——缺依赖
# 当场炸,好过静默写假数据),但单测既不该下模型权重也不该吃 CPU 推理。
# 测试自己定环境,不继承生产缺省,也别指望跑测试的人记得在命令行传。
os.environ["PERSONOS_VIDEO_BACKEND"] = "mock"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from personos.storage.atom_store import AtomStore  # noqa: E402
from personos.storage.db import Database  # noqa: E402
from personos.storage.evidence_store import EvidenceStore  # noqa: E402


@pytest.fixture(scope="session")
def _db() -> Database:
    """整个测试进程共用一个 engine(建表只发生一次;连接由池复用)。"""
    d = Database()
    yield d
    d.close()


@pytest.fixture(autouse=True)
def db(_db: Database):
    """autouse:每个测试一个原子作用域——结束无条件 rollback,共享库零污染。

    即使测试不需要 db,也被包进回退域;将来新写的测试无需显式 opt-in 即天然原子。
    """
    with _db.rollback_scope():
        yield _db


@pytest.fixture
def atom_store(db):
    return AtomStore(db)


@pytest.fixture
def evidence_store(db):
    return EvidenceStore(db)


@pytest.fixture
def rng_vec():
    """生成固定维度的伪向量(测试用,不调远端 embedding)。"""
    def _make(seed: int, dim: int = 8) -> np.ndarray:
        r = np.random.default_rng(seed)
        return r.standard_normal(dim).astype(np.float32)
    return _make
