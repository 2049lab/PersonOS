import json

import numpy as np
import pytest

from personos.models import ChainInfo, MemoryAtom
from personos.storage.atom_store import AtomStore
from personos.storage.chain_store import ChainStore


def _atom(memcell_id: str, text: str, **kw) -> MemoryAtom:
    return MemoryAtom(memcell_id=memcell_id, text=text, **kw)


def _chain(title: str, cell_id: str = "c1") -> ChainInfo:
    return ChainInfo(title=title, origin_cell_id=cell_id)


def _seed(atom_store: AtomStore, chain_store: ChainStore, texts: list[str], vecs):
    """Upsert a series of atoms: the first one creates the chain, the rest are appended.
    Returns (chain, atoms)."""
    atoms = [_atom("c1", t, object_type="fact") for t in texts]
    for a, v in zip(atoms, vecs):
        atom_store.upsert(a, embedding=v)
    chain = chain_store.create_chain(_chain("测试链"), atoms[0], centroid=vecs[0])
    for a, v in zip(atoms[1:], vecs[1:]):
        chain_store.append_atom(chain, a, centroid=vecs[0] + v)  # any interim centroid; recompute will redo it
    return chain, atoms


def test_create_chain_double_write(db, rng_vec):
    """Chain creation writes the chain row, the three chain columns on the atom, and the payload
    copy -- and both sides must agree."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    a1 = _atom("c1", "Caroline 每周三练瑜伽(2026-08)", object_type="fact")
    atom_store.upsert(a1, embedding=rng_vec(1))
    chain = chain_store.create_chain(_chain("Caroline 练瑜伽"), a1, centroid=rng_vec(9))

    got = chain_store.get_chain(chain.id)
    assert got.n_atoms == 1 and got.head_atom_id == got.tail_atom_id == a1.id
    assert got.title == "Caroline 练瑜伽" and got.origin_cell_id == "c1"

    row = db.fetch_one("SELECT chain_id, prev_atom_id, next_atom_id, payload FROM atoms WHERE id=%s", (a1.id,))
    assert row["chain_id"] == chain.id and row["prev_atom_id"] is None and row["next_atom_id"] is None
    copy = json.loads(row["payload"])  # the payload display copy comes from the same source as the columns
    assert copy["chain_id"] == chain.id and copy["prev_atom_id"] == "" and copy["next_atom_id"] == ""


def test_append_walks_tail_and_links(db, rng_vec):
    """Appending at the tail moves the doubly-linked pointers forward, and fetching the whole
    chain returns the atoms in append order."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vecs = [rng_vec(s) for s in (1, 2, 3)]
    chain, atoms = _seed(atom_store, chain_store, ["a1", "a2", "a3"], vecs)

    got = chain_store.get_chain(chain.id)
    assert got.n_atoms == 3 and got.head_atom_id == atoms[0].id and got.tail_atom_id == atoms[2].id

    ordered = chain_store.full_chain(chain.id)
    assert [a.id for a in ordered] == [a.id for a in atoms]  # chain order == append order
    assert (ordered[0].prev_atom_id, ordered[0].next_atom_id) == ("", atoms[1].id)
    assert (ordered[1].prev_atom_id, ordered[1].next_atom_id) == (atoms[0].id, atoms[2].id)
    assert (ordered[2].prev_atom_id, ordered[2].next_atom_id) == (atoms[1].id, "")

    # The payload copy stays in sync (AtomStore.get reads from payload)
    mid = atom_store.get(atoms[1].id)
    assert mid.chain_id == chain.id and mid.prev_atom_id == atoms[0].id and mid.next_atom_id == atoms[2].id


def test_append_rejects_double_link_and_rolls_back(db, rng_vec):
    """Attaching an already-linked atom to another chain raises and rolls the whole write back,
    leaving the original chain structure untouched."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vecs = [rng_vec(s) for s in (1, 2, 3)]
    chain_a, _ = _seed(atom_store, chain_store, ["a1"], vecs[:1])
    chain_b, atoms_b = _seed(atom_store, chain_store, ["b1"], [vecs[2]])

    with pytest.raises(RuntimeError, match="chain link conflict"):
        chain_store.append_atom(chain_a, atoms_b[0])

    # Chain A is uncontaminated: the count is unchanged and its tail still has no successor
    got_a = chain_store.get_chain(chain_a.id)
    assert got_a.n_atoms == 1
    assert chain_store.full_chain(chain_a.id)[0].next_atom_id == ""
    # Chain B is unchanged
    assert chain_store.get_chain(chain_b.id).n_atoms == 1


def test_append_missing_atom_raises(db, rng_vec):
    """The atom being attached does not exist -> raise and roll back; the chain does not advance."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vecs = [rng_vec(s) for s in (1, 2)]
    chain, atoms = _seed(atom_store, chain_store, ["a1"], vecs[:1])
    ghost = _atom("c1", "不存在的原子", object_type="fact")

    with pytest.raises(RuntimeError, match="does not exist"):
        chain_store.append_atom(chain, ghost)
    assert chain_store.get_chain(chain.id).n_atoms == 1


def test_free_atom_and_unknown_chain_noop(db, rng_vec):
    """An unlinked atom stays free, and fetching an unknown chain returns nothing."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    free = _atom("c1", "游离原子", object_type="fact")
    atom_store.upsert(free, embedding=rng_vec(1))
    assert atom_store.get(free.id).chain_id == ""
    assert chain_store.full_chain("chn_nonexistent") == []
    assert chain_store.list_chains() == []
    assert chain_store.get_chain("chn_nonexistent") is None


def test_list_chains_returns_centroid(db, rng_vec):
    """Centroid round-trip through the hex encoding channel."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vec = rng_vec(5)
    a1 = _atom("c1", "a1", object_type="fact")
    atom_store.upsert(a1, embedding=vec)
    chain_store.create_chain(_chain("带质心"), a1, centroid=vec)
    pairs = chain_store.list_chains()
    assert len(pairs) == 1
    info, centroid = pairs[0]
    assert info.title == "带质心"
    assert centroid is not None and np.allclose(centroid, vec, atol=1e-6)


def test_recompute_from_members(db, rng_vec):
    """recompute recalculates the centroid from the member vectors and re-syncs the count; this is
    where drift accumulated by incremental centroid updates gets corrected."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vecs = [rng_vec(s) for s in (1, 2, 3)]
    chain, _ = _seed(atom_store, chain_store, ["a1", "a2", "a3"], vecs)

    got = chain_store.recompute(chain.id)
    assert got.n_atoms == 3
    _, centroid = chain_store.list_chains()[0]
    assert np.allclose(centroid, np.mean(np.stack(vecs), axis=0), atol=1e-6)


def test_clear_user_resets_both_sides(db, rng_vec):
    """clear_user deletes the chain rows and zeroes out both the three chain columns on atoms and
    the payload copy."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vecs = [rng_vec(s) for s in (1, 2, 3)]
    chain, atoms = _seed(atom_store, chain_store, ["a1", "a2"], vecs[:2])
    _seed(atom_store, chain_store, ["b1"], [vecs[2]])

    n = chain_store.clear_user()
    assert n == 2
    assert chain_store.list_chains() == []
    assert chain_store.full_chain(chain.id) == []
    for a in atoms:  # columns and copy are cleared from the same source
        row = db.fetch_one("SELECT chain_id, payload FROM atoms WHERE id=%s", (a.id,))
        assert row["chain_id"] is None
        assert json.loads(row["payload"])["chain_id"] == ""


def test_user_isolation(db, rng_vec):
    """Chains are isolated per user: u1's chain is invisible to u2, and u2 cannot attach to it."""
    s1a, s1c = AtomStore(db, "u1"), ChainStore(db, "u1")
    s2c = ChainStore(db, "u2")
    a1 = _atom("c1", "u1 的原子", object_type="fact")
    s1a.upsert(a1, embedding=rng_vec(1))
    chain = s1c.create_chain(_chain("u1 的链"), a1, centroid=rng_vec(2))

    assert s2c.list_chains() == []
    assert s2c.full_chain(chain.id) == []


def test_upsert_replay_preserves_chain(db, rng_vec):
    """A replayed upsert (whose ON DUPLICATE KEY UPDATE does not touch the three chain columns)
    must not lose chain membership, and reads take the column values as authoritative."""
    atom_store, chain_store = AtomStore(db), ChainStore(db)
    vecs = [rng_vec(s) for s in (1, 2)]
    chain, atoms = _seed(atom_store, chain_store, ["a1", "a2"], vecs)

    replay = MemoryAtom(memcell_id="c1", text="a1", object_type="fact")  # a fresh free object (its generated id differs)
    replay.id = atoms[0].id  # simulate a replay: same id, no chain fields
    atom_store.upsert(replay, embedding=vecs[0])

    ordered = chain_store.full_chain(chain.id)
    assert [a.id for a in ordered] == [a.id for a in atoms]  # columns were not wiped, chain intact
    # all_with_embeddings lets the columns override payload, so the hot path still sees the
    # column values for chain membership
    hot = {a.id: a for a, _ in atom_store.all_with_embeddings()}
    assert hot[atoms[0].id].chain_id == chain.id
    assert hot[atoms[0].id].next_atom_id == atoms[1].id
