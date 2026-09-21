"""C 视频→记忆桥单测:draft 剧本行 + by_chain 归属 → 带人物归属的 evidence + build_cell。

FakeLLM(episode→atoms 两段)+ 真 SIT(db fixture=rollback 回退)。验证:
- 行证据 holder=展示名(Bob/user/env)、source.character_id 正确;
- raw_clip 原始媒体证据落库(modality=video、无 content_inline);
- build_cell 产 memcell + atoms、atom.holder 沿用展示名、evidence_refs 回链行证据;
- "剧本行→文本 evidence"后完全走现有 build_cell,与文本零区别。
"""

from __future__ import annotations

from personos.identity.draft import MemoryDraftStore
from personos.identity.store import CharacterStore
from personos.online.video_memory import flush_session_to_memory
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder, FakeLLM

U = "vtest_vmem"
S = "s1"

_EPISODE = '{"topic":"hiking chat","episode":"user and Bob chat about hiking in the living room","domains":[]}'
_ATOMS = ('{"atoms":[{"text":"Bob loves hiking","holder":"Bob","object_type":"claim",'
          '"kind":"K06","quote":"I love hiking"}]}')


def test_flush_session_to_memory(db):
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    # 库:Bob(具名)+ wearer 档
    bob = store.create_character(session_id=S)
    store.add_name_claim(bob, "Bob", "explicit_dialogue")
    wearer = store.create_character(session_id=S, is_wearer=True)
    store.mark_wearer(wearer)
    store.set_session_wearer(S, wearer)

    draft = MemoryDraftStore(U)
    for cast in ("S1", "SW"):
        draft.ensure_chain(S, cast)
    draft.stage_lines(S, 0, [
        (0.0, 1.0, "S1", "speech", "I love hiking"),
        (1.0, 2.0, "SW", "speech", "Noted"),
        (2.0, 3.0, "ENV", "environment", "a bright living room with a green sofa"),
    ])
    by_chain = {draft.chain_ref(S, "S1"): bob, draft.chain_ref(S, "SW"): wearer}

    cb = flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, _ATOMS]), embedder=FakeEmbedder(),
        clip_keys={0: "oss/clip0.mp4"})

    # 证据落库:1 raw_clip + 3 行
    recs = ev.by_session(S)
    by_kind = {(r.source or {}).get("kind"): r for r in recs}
    raw = by_kind["raw_clip"]
    assert raw.modality == "video" and raw.content_ref == "oss/clip0.mp4" and not raw.content_inline

    speech = [r for r in recs if (r.source or {}).get("kind") == "speech"]
    s1 = next(r for r in speech if r.source["cast_id"] == "S1")
    sw = next(r for r in speech if r.source["cast_id"] == "SW")
    assert s1.holder == "Bob" and s1.source["character_id"] == bob      # 展示名 + 精确 id
    assert sw.holder == "user" and sw.source["character_id"] == wearer  # SW→user
    env = next(r for r in recs if (r.source or {}).get("kind") == "environment")
    assert env.holder == "env"
    # 行证据都回指 raw_clip + modality=video
    assert s1.source["raw_evidence_id"] == raw.id and s1.modality == "video"

    # build_cell 产物:memcell + atom 沿用展示名 + 回链行证据
    assert cb is not None and cb.cell is not None
    assert len(cb.atoms) == 1 and cb.atoms[0].holder == "Bob"
    assert cb.atoms[0].evidence_refs and cb.atoms[0].evidence_refs[0].evidence_id == s1.id


def test_flush_no_lines_returns_none(db):
    store = CharacterStore(db, U)
    draft = MemoryDraftStore(U)
    out = flush_session_to_memory(
        draft, {}, session_id="empty", char_store=store,
        evidence_store=EvidenceStore(db, U), cell_store=CellStore(db, U),
        atom_store=AtomStore(db, U), chain_store=ChainStore(db, U),
        llm=FakeLLM([]), embedder=FakeEmbedder())
    assert out is None


def test_unnamed_gets_stable_handle(db):
    """无名人物 → 稳定短标 人物#N。"""
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    anon = store.create_character(session_id=S)          # 无名
    draft = MemoryDraftStore(U)
    draft.ensure_chain(S, "S1")
    draft.stage_lines(S, 0, [(0.0, 1.0, "S1", "speech", "hello there")])
    by_chain = {draft.chain_ref(S, "S1"): anon}
    flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, '{"atoms":[]}']), embedder=FakeEmbedder(),
        clip_keys={0: "oss/c.mp4"})
    line = next(r for r in ev.by_session(S) if (r.source or {}).get("kind") == "speech")
    assert line.holder == "人物#1" and line.source["character_id"] == anon


def test_rewrite_ids_word_boundary_and_longest_first():
    """id 改写:只碰完整 token、长 key 优先、mapping 外的 id 原样留着。"""
    from personos.identity.screenplay import rewrite_ids

    m = {"P1": "Alice", "P12": "Bob", "SW": "user"}
    assert rewrite_ids("P1 enters; P12 sits; SW films", m) == "Alice enters; Bob sits; user films"
    assert rewrite_ids("P12 is not P1", m) == "Bob is not Alice"        # 长 key 不被 P1 切一半
    assert rewrite_ids("SPAM P1X nothing", m) == "SPAM P1X nothing"     # 词边界:不碰 P1X/SPAM
    assert rewrite_ids("P3 unknown", m) == "P3 unknown"                 # 不在表里的原样留
    assert rewrite_ids("", m) == "" and rewrite_ids("x", {}) == "x"


def test_flush_rewrites_ids_in_text_and_marks_actions(db):
    """行文本里的 cast id → 展示名;action 行加标记;无名人物按**出场顺序**编号且可复现。

    回归真问题:episode 里出现过 "P1 enters holding an orange basketball"、
    "where P2 is seated" —— 归属改写只动了 holder,文本里的裸 id 一路漏进 episode/atom,
    同一个人有三种叫法(holder 叫 人物#1、文本里叫 P1、别人嘴里叫 Alice)。
    另:动作行与台词行渲染成同一形状("holder: text"),会被 episode LLM 读成这个人说的话。
    """
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    bob = store.create_character(session_id=S)
    store.add_name_claim(bob, "Bob", "explicit_dialogue")
    anon = store.create_character(session_id=S)          # 无名:应拿 人物#N
    wearer = store.create_character(session_id=S, is_wearer=True)
    store.mark_wearer(wearer)
    store.set_session_wearer(S, wearer)

    draft = MemoryDraftStore(U)
    for cast in ("S1", "S2", "SW"):
        draft.ensure_chain(S, cast)
    draft.stage_lines(S, 0, [
        # S2(无名)先出场 → 应是 人物#1;文本里的 S1/S2/SW 都要被换掉
        (0.0, 1.0, "S2", "action", "S2 enters holding a basketball while SW films"),
        (1.0, 2.0, "S1", "speech", "S2, you are so stinky"),
        (2.0, 3.0, "ENV", "environment", "S1 is seated at the table"),
    ])
    by_chain = {draft.chain_ref(S, "S1"): bob, draft.chain_ref(S, "S2"): anon,
                draft.chain_ref(S, "SW"): wearer}

    flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, _ATOMS]), embedder=FakeEmbedder(), clip_keys={})

    texts = {(r.source or {}).get("cast_id"): (r.content_inline or "")
             for r in ev.by_session(S) if (r.source or {}).get("kind") != "raw_clip"}
    assert "S1" not in texts["S2"] and "S2" not in texts["S2"] and "SW" not in texts["S2"], \
        f"文本里仍残留裸 id:{texts['S2']!r}"
    assert texts["S2"] == "(action) 人物#1 enters holding a basketball while user films", texts["S2"]
    assert texts["S1"] == "人物#1, you are so stinky", texts["S1"]     # 先出场的 S2 = 人物#1
    assert texts["ENV"] == "Bob is seated at the table", texts["ENV"]  # env 行不加 action 标记


def test_anon_person_gets_one_intro_line_with_description(db):
    """无名人物:在对话之前**单列一行**外观说明,且每人只列一次;有名字的不列。

    为什么要:没有这行,记忆里的「人物#1」就是个空壳编号——episode/atom 读到它完全不知道
    是谁,既判断不了跨会话是否同一人,作答时也只能干巴巴复述编号。
    """
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    bob = store.create_character(session_id=S)
    store.add_name_claim(bob, "Bob", "explicit_dialogue")
    anon = store.create_character(session_id=S)
    wearer = store.create_character(session_id=S, is_wearer=True)
    store.mark_wearer(wearer)
    store.set_session_wearer(S, wearer)

    draft = MemoryDraftStore(U)
    for cast in ("S1", "S2", "SW"):
        draft.ensure_chain(S, cast)
    # 链上的外观描述(剧本 cast.desc → chains.observe_clip 写入 desc_text)
    draft.update_chain(draft.chain_ref(S, "S2"),
                       desc_text="a woman in a pink dress with a ponytail")
    draft.update_chain(draft.chain_ref(S, "S1"), desc_text="a man in a grey hoodie")
    draft.stage_lines(S, 0, [
        (0.0, 1.0, "S2", "speech", "You ruined it"),
        (1.0, 2.0, "S1", "speech", "Sorry"),
        (2.0, 3.0, "S2", "speech", "Again!"),          # 同一人再次出现,不得再列一行
    ])
    by_chain = {draft.chain_ref(S, "S1"): bob, draft.chain_ref(S, "S2"): anon,
                draft.chain_ref(S, "SW"): wearer}

    flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, _ATOMS]), embedder=FakeEmbedder(), clip_keys={})

    intros = [r for r in ev.by_session(S) if (r.source or {}).get("kind") == "cast_intro"]
    assert len(intros) == 1, f"无名人物应恰好一行说明(Bob 有名字不列):{[r.content_inline for r in intros]}"
    assert intros[0].content_inline == "人物#1 is a woman in a pink dress with a ponytail"
    assert intros[0].holder == "env", "说明是旁白,不该挂在人物名下(否则像是他自己说的)"
    assert (intros[0].source or {}).get("character_id") == anon
