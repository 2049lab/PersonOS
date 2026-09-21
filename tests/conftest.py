"""Test fixtures: an in-memory database, one atomic scope per test.

Every test used to require a real MySQL instance. The session fixture connected
on construction and the `db` fixture is autouse, so even a test that never
touches storage dragged a database connection along. For an open-source project
that means nobody outside the original network can run the suite — which means
nobody can contribute.

Now it is SQLite in memory: no network, no credentials, milliseconds to build.

The guard rails are kept, because they cost nothing and still protect anyone who
points the suite at a real database through PERSONOS_DB_URL:
1. Each test is wrapped in Database.rollback_scope(), so writes never persist.
   Fixed ids and fixed user_ids are therefore safe, and new tests are atomic
   without opting in.
2. A hand-built Database() is read-only under pytest: writes outside a rollback
   scope raise, turning silent pollution into a loud error.
3. DROP/TRUNCATE/unqualified DELETE raise on every path, rollback scope or not.
"""

import os

# Must be set before importing personos: the guard hooks Database.execute.
os.environ["PERSONOS_TEST_GUARD"] = "1"
# Force tracing off. load_dotenv does not override existing variables, so a
# developer's .env would otherwise make unit tests report to a real Langfuse
# project — slow, and noise in someone else's dashboard.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
# Pin the fake identity backends. The default is "none" (see backends/factory),
# but tests should neither download model weights nor burn CPU on inference.
# Tests set their own environment rather than inheriting a production default,
# or relying on whoever runs pytest to remember a flag.
os.environ["PERSONOS_VIDEO_BACKEND"] = "mock"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from personos.storage.atom_store import AtomStore  # noqa: E402
from personos.storage.db import Database, SQLiteDatabase  # noqa: E402
from personos.storage.evidence_store import EvidenceStore  # noqa: E402


@pytest.fixture(scope="session")
def _db() -> Database:
    """One engine for the whole run; the schema is created once.

    SQLite unless PERSONOS_TEST_BACKEND=mysql. Running this identical suite
    against both backends is what proves the dialect translation preserves
    behaviour, so that path stays one environment variable away.
    """
    use_mysql = os.environ.get("PERSONOS_TEST_BACKEND", "").lower() == "mysql"
    d = Database() if use_mysql else SQLiteDatabase(":memory:")
    yield d
    d.close()


@pytest.fixture(autouse=True)
def db(_db: Database):
    """autouse: one atomic scope per test — unconditional rollback on exit."""
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
    """Deterministic pseudo-vectors, so tests never call a real embedder."""
    def _make(seed: int, dim: int = 8) -> np.ndarray:
        r = np.random.default_rng(seed)
        return r.standard_normal(dim).astype(np.float32)
    return _make
