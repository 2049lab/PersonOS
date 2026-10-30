"""Browsing preserves attribution, bounded pages and chain link integrity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from personos.models import ChainInfo, EvidenceRecord, EvidenceRef, MemCell, MemoryAtom
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.evidence_store import EvidenceStore
from personos.storage.memory_query import MemoryQuery
from server.query_api import install_query_routes
from server.response import EnvelopeRoute

STAMP = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def seed(db, user="alice", suffix="1", *, session="s1", text="Alice likes tea.",
         kind="K06", domain="D13", object_type="fact"):
    evidence = EvidenceRecord(id=f"ev_{user}_{suffix}", holder="user", content_inline=text,
                              captured_at=STAMP, source={"session_id": session})
    EvidenceStore(db, user).append(evidence)
    cell = MemCell(id=f"cell_{user}_{suffix}", session_id=session, topic="Preferences",
                   episode=text, t_start=STAMP, t_end=STAMP,
                   evidence_refs=[{"evidence_id": evidence.id}])
    CellStore(db, user).upsert(cell)
    atom = MemoryAtom(id=f"atom_{user}_{suffix}", memcell_id=cell.id, text=text,
                      kind=kind, domains=[domain], object_type=object_type,
                      recorded_at=STAMP, evidence_refs=[{"evidence_id": evidence.id}])
    AtomStore(db, user).upsert(atom)
    return atom, cell, evidence


def chain_seed(db, *, user="alice", order=("1", "2", "3")):
    atoms = [seed(db, user, n, text=f"Observation {n}.")[0] for n in order]
    store = ChainStore(db, user)
    info = ChainInfo(id=f"chn_{user}", title="Tea preference", created_at=STAMP, updated_at=STAMP)
    store.create_chain(info, atoms[0], None)
    for atom in atoms[1:]:
        store.append_atom(info, atom, None)
    return atoms, info


def test_memory_pages_filter_before_slicing_and_resolve_provenance(db):
    seed(db, suffix="1", session="other", text="A different session.")
    two, cell, ev = seed(db, suffix="2", text="Alice likes tea.")
    three, _, _ = seed(db, suffix="3", text="Alice likes green tea.")
    seed(db, "bob", text="Alice likes tea.")
    browse = MemoryQuery(db, "alice")
    page = browse.memories(session_id="s1", query="tea", kind="K06", domain="D13", limit=1)
    assert [i["atom_id"] for i in page["items"]] == [three.id]
    assert page["has_more"]
    second = browse.memories(session_id="s1", query="tea", kind="K06", domain="D13",
                             limit=1, offset=1)
    item = second["items"][0]
    assert item["atom_id"] == two.id
    assert item["cell"]["cell_id"] == cell.id
    assert item["evidence"][0]["id"] == ev.id
    assert item["evidence"][0]["content"] == "Alice likes tea."
    assert not second["has_more"]
    assert browse.memories(object_type="event")["items"] == []
    assert browse.memories(holder="someone_else")["items"] == []
    assert browse.memories(domain="D05")["items"] == []


def test_text_filter_is_literal_and_targets_text_not_other_json_fields(db):
    atom, _, _ = seed(db, suffix="1", text='Alice chose "100%_!" tea.')
    seed(db, suffix="2", text="Alice chose 100 sweet tea.")
    browse = MemoryQuery(db, "alice")
    assert [i["atom_id"] for i in browse.memories(query='"100%_!"')["items"]] == [atom.id]
    assert browse.memories(query="K06")["items"] == []
    assert browse.memories(query="' OR 1=1 --")["items"] == []


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": 1.5},
    {"offset": -1}, {"offset": 100001}, {"offset": True},
    {"kind": "unknown"}, {"domain": "D99"}, {"object_type": "opinion"},
    {"kind": []}, {"domain": []}, {"object_type": []},
    {"query": ""}, {"session_id": ""}, {"holder": "x" * 65},
])
def test_invalid_query_does_not_reach_storage(db, monkeypatch, kwargs):
    def forbidden(*args, **kw):
        raise AssertionError("validation must happen before storage")
    monkeypatch.setattr(db, "fetch_all", forbidden)
    with pytest.raises(ValueError):
        MemoryQuery(db, "alice").memories(**kwargs)


def test_forged_cross_user_references_never_resolve(db):
    bob, bob_cell, bob_ev = seed(db, "bob", text="Private Bob content.")
    alice, _, _ = seed(db, "alice", text="Alice content.")
    alice.memcell_id = bob_cell.id
    alice.evidence_refs = [EvidenceRef(evidence_id=bob_ev.id)]
    AtomStore(db, "alice").upsert(alice)
    out = MemoryQuery(db, "alice").memories()
    assert len(out["items"]) == 1
    assert out["items"][0]["cell"] is None
    assert out["items"][0]["evidence"] == []
    assert "Private Bob content" not in json.dumps(out)
    assert MemoryQuery(db, "nobody").memories()["items"] == []


def test_membership_columns_override_stale_payload(db):
    atoms, info = chain_seed(db)
    row = db.fetch_one("SELECT payload FROM atoms WHERE user_id=%s AND id=%s", ("alice", atoms[0].id))
    payload = json.loads(row["payload"])
    payload.update(chain_id="stale", next_atom_id="stale", prev_atom_id="stale")
    db.execute("UPDATE atoms SET payload=%s WHERE user_id=%s AND id=%s",
               (json.dumps(payload), "alice", atoms[0].id))
    items = MemoryQuery(db, "alice").memories()["items"]
    item = next(i for i in items if i["atom_id"] == atoms[0].id)
    assert item["chain_id"] == info.id
    assert item["prev_atom_id"] is None
    assert item["next_atom_id"] == atoms[1].id


def test_chain_index_does_not_read_members_or_vectors(db, monkeypatch):
    _, info = chain_seed(db)
    original, queries = db.fetch_all, []
    def capture(sql, params=()):
        queries.append(sql)
        return original(sql, params)
    monkeypatch.setattr(db, "fetch_all", capture)
    out = MemoryQuery(db, "alice").chains(limit=1)
    assert out["items"][0]["chain_id"] == info.id
    assert out["items"][0]["n_atoms"] == 3
    assert not out["has_more"]
    assert len(queries) == 1
    assert "FROM atom_chains" in queries[0]
    assert "centroid" not in queries[0]
    assert "items" not in out["items"][0]
    assert MemoryQuery(db, "bob").chain(info.id) is None


def test_chain_index_pagination_uses_limit_plus_one(db):
    _, first = chain_seed(db)
    atom, _, _ = seed(db, suffix="4", text="Another subject.")
    second = ChainInfo(id="chn_z", title="Another subject")
    ChainStore(db, "alice").create_chain(second, atom)
    query = MemoryQuery(db, "alice")
    page = query.chains(limit=1)
    assert page["has_more"]
    assert page["items"][0]["chain_id"] == second.id
    page = query.chains(limit=1, offset=1)
    assert not page["has_more"]
    assert page["items"][0]["chain_id"] == first.id


def test_chain_cursor_follows_links_even_when_ids_are_out_of_order(db):
    atoms, info = chain_seed(db, order=("3", "1", "2"))
    query = MemoryQuery(db, "alice")
    first = query.chain(info.id, limit=2)
    assert [i["atom_id"] for i in first["items"]] == [a.id for a in atoms[:2]]
    assert first["state"] == "more" and first["has_more"]
    second = query.chain(info.id, limit=2, cursor=first["next_cursor"])
    assert [i["atom_id"] for i in second["items"]] == [atoms[2].id]
    assert second["state"] == "complete"
    assert second["next_cursor"] is None and not second["has_more"]


def test_cursor_detects_append_between_pages(db):
    _, info = chain_seed(db)
    query = MemoryQuery(db, "alice")
    first = query.chain(info.id, limit=1)
    atom, _, _ = seed(db, suffix="4", text="New tea preference.")
    ChainStore(db, "alice").append_atom(info, atom, None)
    second = query.chain(info.id, cursor=first["next_cursor"])
    assert second["state"] == "changed"
    assert second["items"] == [] and second["next_cursor"] is None


def test_chain_read_discards_page_if_metadata_changes_during_read(db, monkeypatch):
    _, info = chain_seed(db)
    query = MemoryQuery(db, "alice")
    original, calls = query._chain, []
    def changed(chain_id):
        value = original(chain_id)
        calls.append(chain_id)
        if len(calls) == 2:
            value["n_atoms"] += 1
        return value
    monkeypatch.setattr(query, "_chain", changed)
    result = query.chain(info.id, limit=1)
    assert result["state"] == "changed"
    assert result["items"] == []


def test_memory_page_batches_provenance_reads(db, monkeypatch):
    for index in range(25):
        seed(db, suffix=f"{index:02}", text=f"Observation {index}.")
    original, queries = db.fetch_all, []
    def capture(sql, params=()):
        queries.append((sql, params))
        return original(sql, params)
    monkeypatch.setattr(db, "fetch_all", capture)
    page = MemoryQuery(db, "alice").memories(limit=20)
    assert len(page["items"]) == 20 and page["has_more"]
    assert len(queries) == 3  # page, its cells, its evidence
    assert queries[0][1][-2:] == (21, 0)
    assert "embedding" not in " ".join(sql for sql, _ in queries)


@pytest.mark.parametrize("corruption", ["missing", "cycle", "wrong_prev", "count"])
def test_broken_chains_return_only_confirmed_prefix(db, corruption):
    atoms, info = chain_seed(db)
    if corruption == "missing":
        db.execute("DELETE FROM atoms WHERE user_id=%s AND id=%s", ("alice", atoms[1].id))
    elif corruption == "cycle":
        db.execute("UPDATE atoms SET next_atom_id=%s WHERE user_id=%s AND id=%s",
                   (atoms[0].id, "alice", atoms[1].id))
    elif corruption == "wrong_prev":
        db.execute("UPDATE atoms SET prev_atom_id=%s WHERE user_id=%s AND id=%s",
                   ("wrong", "alice", atoms[1].id))
    else:
        db.execute("UPDATE atom_chains SET n_atoms=%s WHERE user_id=%s AND id=%s",
                   (8, "alice", info.id))
    page = MemoryQuery(db, "alice").chain(info.id)
    assert page["state"] == "incomplete"
    assert not page["has_more"] and page["next_cursor"] is None
    assert page["items"][0]["atom_id"] == atoms[0].id


@pytest.mark.parametrize("cursor", ["garbage", "W10=", "e30=", "" ])
def test_malformed_cursor_is_rejected(db, cursor):
    with pytest.raises(ValueError, match="cursor"):
        MemoryQuery(db, "alice").chain("chn_alice", cursor=cursor)


def test_routes_inherit_authentication_and_envelope(db):
    seed(db, session="app:s1")
    _, info = chain_seed(db, user="bob")
    authenticated = {"allowed": True}
    def auth():
        if not authenticated["allowed"]:
            raise HTTPException(status_code=401)
    def ctx():
        return SimpleNamespace(atoms=AtomStore(db, "alice"), user_id="alice")
    app = FastAPI()
    router = APIRouter(prefix="/api/v1", route_class=EnvelopeRoute, dependencies=[Depends(auth)])
    install_query_routes(router, ctx)
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/api/v1/memories", params={"session_id": "s1", "caller": "app"})
        assert response.status_code == 200
        assert response.json()["code"] == 0
        assert len(response.json()["data"]["items"]) == 1
        assert client.get("/api/v1/memories?kind=invalid").status_code == 400
        assert client.get("/api/v1/memories?caller=app").status_code == 400
        assert client.get("/api/v1/memories?limit=101").status_code == 422
        assert client.get(f"/api/v1/chains/{info.id}").status_code == 404
        authenticated["allowed"] = False
        assert client.get("/api/v1/chains").status_code == 401
