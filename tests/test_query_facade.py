"""Verify the public SDK and mounted HTTP routes reach the scoped browse implementation."""

import pytest

from personos.memory import Memory, UserContext
from personos.models import ChainInfo, MemoryAtom
from personos.storage.atom_store import AtomStore
from personos.storage.chain_store import ChainStore


def test_browse_facade_needs_no_providers_and_keeps_user_scope(db):
    memory = Memory.__new__(Memory)
    memory.db = db
    atom = MemoryAtom(id="atom_sdk", text="The user walks to work.")
    AtomStore(db, "alice").upsert(atom)
    ChainStore(db, "alice").create_chain(ChainInfo(id="chn_sdk", title="Commute"), atom)
    assert memory.list_memories(user_id="alice")["items"][0]["atom_id"] == "atom_sdk"
    assert memory.list_chains(user_id="alice")["items"][0]["chain_id"] == "chn_sdk"
    assert memory.get_chain("chn_sdk", user_id="alice")["state"] == "complete"
    assert memory.list_memories(user_id="bob")["items"] == []
    assert memory.list_chains(user_id="bob")["items"] == []
    assert memory.get_chain("chn_sdk", user_id="bob") is None
    with pytest.raises(ValueError):
        memory.list_memories(user_id="invalid:id")


def test_query_routes_are_mounted_on_authenticated_api(db):
    from fastapi.testclient import TestClient

    from server import api
    from server.app import app
    from server.signing import verify_signature

    AtomStore(db, "alice").upsert(MemoryAtom(id="atom_http", text="The user reads fiction."))
    app.dependency_overrides[api._ctx] = lambda: UserContext("alice", db)
    app.dependency_overrides[verify_signature] = lambda: None
    client = TestClient(app)
    try:
        response = client.get("/api/v1/memories?limit=1")
        assert response.status_code == 200
        assert response.json()["data"]["items"][0]["atom_id"] == "atom_http"
        assert client.get("/api/v1/chains").json()["data"]["items"] == []
        assert client.get("/api/v1/chains/absent").status_code == 404
        for path in ("/", "/api-doc"):
            docs = client.get(path, follow_redirects=False)
            assert docs.status_code == 307 and docs.headers["location"] == "/docs"
        schema = client.get("/openapi.json").json()
        assert "/api/v1/memories" in schema["paths"]
        assert "/api/v1/chains/{chain_id}" in schema["paths"]
    finally:
        client.close()
        app.dependency_overrides.clear()
