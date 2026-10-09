
from personos.models import MemoryAtom
from personos.storage.atom_store import AtomStore


def _atom(memcell_id: str, text: str, **kw) -> MemoryAtom:
    return MemoryAtom(memcell_id=memcell_id, text=text, **kw)


def test_upsert_and_get(db, rng_vec):
    store = AtomStore(db)
    atom = _atom("c1", "Caroline 上周(2026-08-18)打篮球时扭伤了右脚踝", object_type="event")
    store.upsert(atom, embedding=rng_vec(1))
    got = store.get(atom.id)
    assert got.text == "Caroline 上周(2026-08-18)打篮球时扭伤了右脚踝"
    assert got.memcell_id == "c1"
    assert store.get_embedding(atom.id).shape == (8,)


def test_update_preserves_embedding(db, rng_vec):
    store = AtomStore(db)
    atom = _atom("c1", "Caroline 住上海", object_type="fact")
    store.upsert(atom, embedding=rng_vec(2))
    # A second upsert without an embedding (remember only changes fields) must not wipe the vector.
    atom.holder = "Caroline"
    store.upsert(atom)
    assert store.get(atom.id).holder == "Caroline"
    assert store.get_embedding(atom.id) is not None


def test_list_by_cell_orders_and_filters(db, rng_vec):
    store = AtomStore(db)
    store.upsert(_atom("c1", "a1", object_type="fact"), rng_vec(3))
    store.upsert(_atom("c2", "b1", object_type="claim"), rng_vec(4))
    store.upsert(_atom("c2", "b2", object_type="event"), rng_vec(5))
    only_c2 = store.list_by_cell("c2")
    assert [a.text for a in only_c2] == ["b1", "b2"]
    assert all(a.memcell_id == "c2" for a in only_c2)
    assert store.list(limit=10) and len(store.all_with_embeddings()) == 3


def test_upsert_many_is_atomic(db, rng_vec):
    """W2 persistence runs in a single transaction: a failure partway through rolls the whole
    batch back, so a half-written cell never appears."""
    store = AtomStore(db)
    store.upsert_many([(_atom("c1", "ok", object_type="fact"), rng_vec(6))])
    bad = _atom("c1", "boom", object_type="fact")
    items = [(_atom("c1", "x", object_type="fact"), rng_vec(7)), (bad, "not-an-embedding")]
    try:
        store.upsert_many(items)
    except Exception:
        pass
    texts = [a.text for a in store.list_by_cell("c1")]
    assert texts == ["ok"]   # transaction rolled back: x never landed
