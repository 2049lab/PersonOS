"""Write path unit tests: W0 stores evidence / W1 boundaries (segmentation, safety valve,
conservative continuation) / W2 cell generation (degrade, validation, quote back-links) / D1 no dedup.

No real LLM gateway calls: RoutingLLM inspects the system prompt and routes boundary, episode and
atom calls to their own response queues.
"""

from __future__ import annotations

from datetime import datetime

from personos.models import now
from personos.online.write_path import (
    FeedMsg,
    SessionWriter,
    _match_evidence_refs,
    append_utterance,
    build_cell,
    detect_boundary,
)
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder, FakeLLM

_T0 = datetime(2026, 8, 25, 10, 0)


class RoutingLLM:
    """Routes on system prompt features to three separate response queues (boundary/episode/atoms);
    records the call order."""

    def __init__(self, boundary=(), episode=(), atoms=()):
        self.q = {"boundary": list(boundary), "episode": list(episode), "atoms": list(atoms)}
        self.calls: list[str] = []

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        kind = ("boundary" if "boundary detector" in sysp
                else "episode" if "episode weaver" in sysp
                else "atoms" if "atomic-memory extractor" in sysp else "other")
        assert kind != "other", f"unknown system prompt: {sysp[:40]}"
        self.calls.append(kind)
        assert self.q[kind], f"unexpected {kind} call (its queue is empty)"
        resp = self.q[kind].pop(0)
        return resp(messages[-1]["content"]) if callable(resp) else resp


class Env:
    """One isolated database plus the three stores -- the shared base for the write path tests."""

    def __init__(self, db):
        self.ev = EvidenceStore(db)
        self.cells = CellStore(db)
        self.atoms = AtomStore(db)

    def writer(self, llm, session_id="t", max_turns=30) -> SessionWriter:
        return SessionWriter(llm, FakeEmbedder(), self.ev, self.cells, self.atoms,
                             session_id=session_id, max_turns=max_turns)


BOUNDARY_END = '{"should_end": true, "confidence": 0.9, "topic_summary": "画展筹备"}'
BOUNDARY_KEEP = '{"should_end": false, "confidence": 0.8, "topic_summary": "画展筹备"}'
EPISODE_OK = ('{"topic": "Caroline 筹备与 Rob 的联合画展", '
              '"episode": "Caroline 正在筹备与 Rob 的联合画展,展期定在下个月(2026-09)。", '
              '"domains": ["D13"]}')
ATOMS_OK = ('{"atoms": ['
            '{"text": "Caroline 在筹备与 Rob 的联合画展", "object_type": "fact", '
            '"holder": "Caroline", "kind": "K12", "domains": ["D13"], '
            '"when": "2026-08-25", "quote": "筹备她和 Rob 的联合画展"}, '
            '{"text": "展期定在 2026-09 的第二个周末", "object_type": "event", '
            '"holder": "user", "kind": "K12", "domains": ["D13"], "when": null, '
            '"quote": "展期定在下个月"}]}')


# —— W0 ——

def test_w0_appends_evidence_with_speaker(db):
    eid = append_utterance(Env(db).ev, session_id="s", speaker="Sophia",
                           text="我上周扭了脚踝", now_dt=_T0)
    rec = Env(db).ev.get(eid)
    assert rec.holder == "Sophia"
    assert rec.content_inline == "我上周扭了脚踝"
    assert rec.source["session_id"] == "s"


# —— W1 ——

def test_first_utterance_skips_boundary_llm(db):
    """The first utterance of a segment has no boundary to judge: no LLM call, boundary=None."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])   # a stray call would fail the pop or leave an unused response
    r = env.writer(llm).feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    assert r.boundary is None and not r.forced_close and r.closed_cell is None
    assert llm.calls == []


def test_boundary_end_closes_cell_and_starts_new_segment(db):
    """W1 decides to cut: W2 builds a cell from the old segment and the new utterance starts a new one."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备她和 Rob 的联合画展", now_dt=_T0)
    r = w.feed("user", "对了,我最近开始跑跑步机了", now_dt=datetime(2026, 8, 25, 10, 8))
    assert llm.calls == ["boundary", "episode", "atoms"]
    assert r.closed_cell is not None and r.boundary.should_end
    assert len(w.seg) == 1 and w.seg[0].content_inline.startswith("对了")   # the new utterance is in the new segment
    # The output is stored and queryable: cell ordering, atoms belong to the cell, quote hits evidence
    cell = env.cells.get(r.closed_cell.cell.id)
    assert cell.topic.startswith("Caroline") and cell.domains == ["D13"]
    cas = env.atoms.list_by_cell(cell.id)
    assert len(cas) == 2
    assert cas[0].holder == "Caroline" and cas[0].object_type == "fact"
    assert cas[0].evidence_refs and cas[0].evidence_refs[0].evidence_id in {
        e.id for e in env.ev.by_session("t")}
    assert cell.evidence_refs and env.cells.list_session("t") == [cell]


def test_boundary_continue_keeps_segment(db):
    """W1 decides to continue: no cell is built and the segment grows."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "Thanks! 先把场地定了", now_dt=datetime(2026, 8, 25, 10, 6))
    assert r.boundary is not None and not r.boundary.should_end
    assert r.closed_cell is None and len(w.seg) == 2
    assert llm.calls == ["boundary"]


def test_boundary_parse_failure_continues_conservatively(db):
    llm = RoutingLLM(boundary=["### 不是 JSON ###"])
    d = detect_boundary(llm, seg=[_rec("旧句")], new_records=[_rec("新句")])
    assert d.should_end is False   # continue conservatively: a wrong close damages the episode, while continuing is covered by the safety valve


def test_max_turns_forced_close_without_boundary_llm(db):
    """Safety valve: once a segment reaches max_turns, the next utterance forces a close and W1 is not called for it."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm, max_turns=2)
    w.feed("user", "第一句", now_dt=_T0)                       # segment head: no call
    w.feed("user", "第二句", now_dt=_T0)                       # 1<2 inside the segment: W1 runs and continues
    r = w.feed("user", "第三句", now_dt=_T0)                   # 2>=2 inside the segment: forced close (no W1)
    assert llm.calls == ["boundary", "episode", "atoms"]       # the forced close made no boundary call
    assert r.forced_close and r.boundary is None and r.closed_cell is not None


def test_end_session_closes_open_segment(db):
    """Forced close at the end of a session -- this is what guarantees W1 segmentation covers the whole session."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    w.feed("user", "展期下个月", now_dt=_T0)
    assert env.cells.list_session("t") == []
    cells = w.end_session()
    assert len(cells) == 1 and not w.seg
    w.end_session()   # no open segment left: idempotent, no further LLM call
    assert llm.calls == ["boundary", "episode", "atoms"]


# —— W2 ——

def test_w2_call1_failure_preserves_evidence_without_memory(db):
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=["invalid JSON"] * 3)
    w = env.writer(llm)
    first = w.feed("user", "I am organizing an exhibition", now_dt=_T0)
    result = w.feed("user", "A different topic", now_dt=_T0)
    assert result.closed_cell.skipped and result.closed_cell.cell is None
    assert result.closed_cell.skip_reason == "Episode generation failed"
    assert env.ev.get(first.evidence_id).content_inline == "I am organizing an exhibition"
    assert env.cells.list_session("t") == [] and env.atoms.list() == []
    assert w.cells == [] and len(w.seg) == 1
    assert "atoms" not in llm.calls


def test_w2_call2_failure_keeps_episode_without_fabricated_atoms(db):
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK],
                     atoms=["invalid JSON"] * 3)
    w = env.writer(llm)
    w.feed("user", "I am organizing an exhibition", now_dt=_T0)
    result = w.feed("user", "A different topic", now_dt=_T0)
    assert env.cells.get(result.closed_cell.cell.id) is not None
    assert env.atoms.list_by_cell(result.closed_cell.cell.id) == []
    assert llm.calls.count("atoms") == 3  # malformed JSON is still retried


def test_w2_call2_retries_then_succeeds(db):
    """Call 2 first returns non-JSON, the retry returns valid JSON, and the atoms are stored normally so no information is lost."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK],
                     atoms=["先是一段废话", ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "换个话题", now_dt=_T0)
    assert len(env.atoms.list_by_cell(r.closed_cell.cell.id)) == 2
    assert llm.calls.count("atoms") == 2                      # one bad response plus a successful retry


def test_atom_system_prompt_carries_retrieval_anchor_clauses():
    """An atom is a retrieval anchor (MECE coverage of distinct facts, merged repetition, skipped
    trivia): this is a regression guard on the prompt so a rewrite cannot silently drop a clause.
    (It replaces the old exhaustiveness clauses -- atoms moved from an utterance-by-utterance log to
    retrieval-friendly MECE anchors, and exhaustive detail is now the episode's job.)"""
    from personos.online.write_path import _ATOM_SYSTEM
    for clause in ("# Coverage as retrieval anchors", "- One atom per DISTINCT fact:",
                   "- Merge repetition; never one atom per utterance:", "- Skip process trivia:",
                   "- Distinct items & attributes DO each get their own atom"):
        assert clause in _ATOM_SYSTEM


def test_write_prompts_carry_time_and_detail_clauses():
    """Two P1-C calibration guards: the atom `when` anchoring counter-example (which fixes off-by-one
    dates) and the episode date-detail clause, plus the type-2a coverage discipline (walk the
    transcript line by line; small facts must not give way to the narrative)."""
    from personos.online.write_path import _ATOM_SYSTEM, _EPISODE_SYSTEM
    assert "the week BEFORE the dialogue" in _ATOM_SYSTEM
    assert "a bare date is often the entire answer" in _EPISODE_SYSTEM
    assert "walk the transcript utterance by utterance" in _EPISODE_SYSTEM
    assert "never \"departed in mid-July\"" in _EPISODE_SYSTEM


def test_atom_fields_validated(db):
    """An invalid object_type becomes claim, an invalid kind becomes K01, and excess domains are truncated."""
    env = Env(db)
    bad = ('{"atoms": [{"text": "x", "object_type": "opinion", "holder": "user", '
           '"kind": "随便", "domains": ["D13", "D05", "D06", "D01"], "when": null, "quote": ""}]}')
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[bad])
    w = env.writer(llm)
    w.feed("user", "s1", now_dt=_T0)
    r = w.feed("user", "s2", now_dt=_T0)
    a = env.atoms.list_by_cell(r.closed_cell.cell.id)[0]
    assert a.object_type == "claim" and a.kind == "K01" and len(a.domains) == 3
    assert a.evidence_refs == []                                  # an empty quote must not be attached to anything


def test_quote_match_caps_and_skips_foreign_text(db):
    """A quote matches the evidence that contains it as a substring, at most 3 of them, and an empty list when nothing matches."""
    recs = [_rec(f"第{i}句里都有画展两个字") for i in range(4)] + [_rec("无关句")]
    ids = {r.id for r in recs}
    hit = _match_evidence_refs(recs, "画展")        # all 4 contain it, so the list is cut to 3
    assert 0 < len(hit) <= 3 and all(h.evidence_id in ids for h in hit)
    assert _match_evidence_refs(recs, "完全不存在的引文") == []
    assert _match_evidence_refs(recs, "") == []


def test_when_null_falls_back_to_segment_date(db):
    """When `when` is uncertain, the segment start date (the capture time) is used so the field is never empty."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "换话题", now_dt=datetime(2026, 8, 25, 11, 0))
    cas = env.atoms.list_by_cell(r.closed_cell.cell.id)
    assert cas[0].occurrence_time.date() == _T0.date()    # the first atom carries when="2026-08-25"
    assert cas[1].occurrence_time.date() == _T0.date()    # the second has when=null, so seg_date is used


def test_no_dedup_identical_facts_both_stored(db):
    """D1, the write path does not deduplicate: if two cells each extract the same fact, both are kept
    (a redundant index; the reconciliation happens at answer time)."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END, BOUNDARY_END],
                     episode=[EPISODE_OK, EPISODE_OK], atoms=[ATOMS_OK, ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    w.feed("user", "聊聊别的", now_dt=_T0)
    w.feed("user", "又提到 Caroline 在筹备画展", now_dt=_T0)
    w.feed("user", "再聊聊别的", now_dt=_T0)
    texts = [a.text for a in env.atoms.list(limit=99)]
    assert texts.count("Caroline 在筹备与 Rob 的联合画展") == 2   # nothing is resolved away; both copies coexist
    assert len(env.cells.iter_all()) == 2


def test_build_cell_embeds_topic_and_atoms(db):
    """Stored vectors: one per topic plus one per atom -- the only two kinds of vector in the database."""
    env = Env(db)
    llm = RoutingLLM(episode=[EPISODE_OK], atoms=[ATOMS_OK])
    recs = [_rec("我在帮 Caroline 筹备画展", _T0)]
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, recs, session_id="t")
    assert len(env.cells.all_with_embeddings()) == 1
    assert len(env.atoms.all_with_embeddings()) == len(cb.atoms)


def _rec(text: str, when=None):
    from personos.models import EvidenceRecord
    return EvidenceRecord(holder="user", content_inline=text,
                          source={"session_id": "t"}, captured_at=when or now())


# -- episode classification (the task_type vocabulary) --

_EP_TYPE = ('{"topic": "t", "episode": "e", "domains": ["D13"], "episode_type": "work"}')
_EP_BADTYPE = ('{"topic": "t", "episode": "e", "domains": ["D13"], "episode_type": "made_up"}')


def test_episode_type_classified_when_task_type_given(db):
    env = Env(db)
    llm = RoutingLLM(episode=[_EP_TYPE], atoms=[ATOMS_OK])
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, [_rec("在忙工作", _T0)],
                    session_id="t", task_type=["work", "health"])
    assert cb.cell.episode_type == "work"                      # in the vocabulary, so it is accepted


def test_episode_type_falls_back_when_not_in_list(db):
    env = Env(db)
    llm = RoutingLLM(episode=[_EP_BADTYPE], atoms=[ATOMS_OK])
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, [_rec("在忙工作", _T0)],
                    session_id="t", task_type=["work", "health"])
    assert cb.cell.episode_type == "unknown"                   # the LLM output is not in the vocabulary, so it falls back to unknown


def test_episode_type_unknown_without_task_type(db):
    """Without task_type the classification prompt is not assembled and episode_type defaults to unknown."""
    env = Env(db)
    llm = RoutingLLM(episode=[EPISODE_OK], atoms=[ATOMS_OK])
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, [_rec("随便聊", _T0)],
                    session_id="t")
    assert cb.cell.episode_type == "unknown"
    assert "Episode classification" not in llm_last_episode_system(cb)   # no classification instruction was appended


def llm_last_episode_system(cb):
    """Pull the system prompt that W2 call 1 actually used out of CellBuild.gen, to check the conditional assembly."""
    return (cb.gen.get("call1") or {}).get("system", "")


# -- feed_batch (batch atomicity: the whole-batch semantics of the /ingest API) --

def test_feed_batch_first_batch_skips_boundary_llm(db):
    """The first batch of a segment has no boundary to judge: no LLM call, the whole batch enters the segment."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])   # a stray call would fail the pop
    w = env.writer(llm)
    r = w.feed_batch([FeedMsg(speaker="user", text="猜猜我住哪?"),
                      FeedMsg(speaker="assistant", text="裕廊西?")], now_dt=_T0)
    assert r.boundary is None and r.closed_cell is None and not r.forced_close
    assert r.evidence_ids and len(r.evidence_ids) == 2 and len(r.records) == 2
    assert len(w.seg) == 2 and llm.calls == []


def test_feed_batch_atomic_on_topic_shift(db):
    """On a topic shift the closed old segment does not contain this batch; the whole batch becomes the
    new segment, so a question and its answer inside one batch are never split apart."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed_batch([FeedMsg(speaker="user", text="下周三看牙医是几点?"),
                      FeedMsg(speaker="assistant", text="上午 10 点")],
                     now_dt=datetime(2026, 8, 25, 10, 8))
    assert r.boundary is not None and r.boundary.should_end and r.closed_cell is not None
    # The closed cell's evidence refs are exactly the old segment's utterance; both QA lines stay in the new segment
    closed_ids = {ref.evidence_id for ref in r.closed_cell.cell.evidence_refs}
    all_ids = {e.id for e in env.ev.by_session("t")}
    assert closed_ids == all_ids - set(r.evidence_ids)
    assert [rec.content_inline for rec in w.seg] == ["下周三看牙医是几点?", "上午 10 点"]
    assert llm.calls == ["boundary", "episode", "atoms"]


def test_feed_batch_continue_merges_whole(db):
    """On continue, the whole batch is merged into the old segment and no cell is built."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed_batch([FeedMsg(speaker="user", text="场地定了"),
                      FeedMsg(speaker="user", text="展期下个月")],
                     now_dt=datetime(2026, 8, 25, 10, 6))
    assert r.boundary is not None and not r.boundary.should_end and r.closed_cell is None
    assert len(w.seg) == 3


def test_feed_batch_overrun_valve_closes_old_segment(db):
    """The safety valve counts the total length after merging: old segment 2 + batch 2 > max 3, so the
    old segment is force-closed and the batch starts a new one (W1 is not called)."""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm, max_turns=3)
    w.feed("user", "第一句", now_dt=_T0)
    w.feed("user", "第二句", now_dt=_T0)
    r = w.feed_batch([FeedMsg(speaker="user", text="第三句"),
                      FeedMsg(speaker="user", text="第四句")], now_dt=_T0)
    assert r.forced_close and r.boundary is None and r.closed_cell is not None
    # Feeding the second utterance alone is 1+1<=3, so W1 still runs; when the batch arrives 2+2>3, so
    # the segment is force-closed and the batch itself does not call W1
    assert len(w.seg) == 2 and llm.calls == ["boundary", "episode", "atoms"]


def test_zero_atoms_accepted_without_retry_or_fallback(db):
    env = Env(db)
    llm = RoutingLLM(episode=[EPISODE_OK], atoms=['{"atoms":[]}'])
    w = env.writer(llm)
    w.feed("user", "I am organizing an exhibition", now_dt=_T0)
    result = w.end_session()
    assert len(result) == 1 and result[0].atoms == []
    assert env.cells.get(result[0].cell.id) is not None
    assert env.atoms.list() == []
    assert llm.calls == ["episode", "atoms"]


def test_explicit_skip_clears_segment_and_keeps_evidence(db):
    env = Env(db)
    llm = RoutingLLM(episode=['{"skip":true,"reason":"Only task instructions"}'])
    w = env.writer(llm)
    first = w.feed("user", "Set the button padding to 12 pixels", now_dt=_T0)
    assert w.end_session(task_type=["work"]) == []
    assert w.seg == []
    assert env.ev.get(first.evidence_id) is not None
    assert env.cells.iter_all() == [] and env.atoms.list() == []
    assert llm.calls == ["episode"]
    assert w.end_session() == []  # no duplicate model call after skip


def test_skip_during_boundary_keeps_new_batch(db):
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=['{"skip":true}'])
    w = env.writer(llm)
    first = w.feed("assistant", "The build passed", now_dt=_T0)
    result = w.feed("user", "I am moving to a new home", now_dt=_T0)
    assert result.closed_cell.skipped and result.closed_cell.cell is None
    assert w.cells == []
    assert [rec.content_inline for rec in w.seg] == ["I am moving to a new home"]
    assert env.ev.get(first.evidence_id) is not None
    assert len(env.ev.list()) == 2
    assert llm.calls == ["boundary", "episode"]


def test_empty_episode_is_skipped_without_embedding(db):
    class UnexpectedEmbedder:
        def embed(self, texts):
            raise AssertionError("Skipped segments must not be embedded")

    env = Env(db)
    llm = RoutingLLM(episode=['{"topic":"Task instructions","episode":" "}'])
    w = SessionWriter(llm, UnexpectedEmbedder(), env.ev, env.cells, env.atoms, session_id="t")
    w.feed("user", "Change the padding", now_dt=_T0)
    assert w.end_session() == []
    assert env.cells.iter_all() == []
    assert llm.calls == ["episode"]


def test_skip_with_blank_reason_still_skips(db):
    env = Env(db)
    llm = RoutingLLM(episode=['{"skip":true,"reason":"   "}'])
    writer = env.writer(llm)
    writer.feed("user", "Move the button", now_dt=_T0)
    assert writer.end_session() == []
    assert env.cells.iter_all() == [] and env.atoms.list() == []
    assert llm.calls == ["episode"]


def test_episode_takes_precedence_over_conflicting_skip(db):
    env = Env(db)
    llm = RoutingLLM(episode=['{"skip":true,"topic":"Home","episode":"The user lives in Oslo."}'],
                     atoms=['{"atoms":[]}'])
    w = env.writer(llm)
    w.feed("user", "I live in Oslo", now_dt=_T0)
    result = w.end_session()
    assert len(result) == 1 and not result[0].skipped
    assert result[0].cell.episode == "The user lives in Oslo."


def test_subject_scope_shared_and_scenario_cannot_broaden_it():
    from personos.online.write_path import (
        _ATOM_SYSTEM,
        _EPISODE_SYSTEM,
        _SCEN_DIR_ATOM,
        _SCEN_DIR_EPISODE,
        _SUBJECT_LENS,
    )
    assert _SUBJECT_LENS in _ATOM_SYSTEM and _SUBJECT_LENS in _EPISODE_SYSTEM
    assert "Episodes contain no inference" in _SUBJECT_LENS
    assert 'object_type="claim"' in _SUBJECT_LENS
    assert "Never infer unstated sensitive attributes" in _SUBJECT_LENS
    assert "task payload" in _SUBJECT_LENS
    assert "narrow or re-weight" in _SCEN_DIR_EPISODE
    assert "narrow or re-weight" in _SCEN_DIR_ATOM
    assert "Always return at least one" not in _ATOM_SYSTEM


def test_grounded_atoms_keep_normal_source(db):
    """Normal extraction still preserves its evidence and source metadata."""
    from personos.models import EvidenceRecord
    from personos.storage.atom_store import AtomStore
    from personos.storage.cell_store import CellStore
    from personos.storage.evidence_store import EvidenceStore

    U3 = "vtest_atom_nofloor"
    ev, cells, atoms_st = EvidenceStore(db, U3), CellStore(db, U3), AtomStore(db, U3)
    rec = EvidenceRecord(holder="user", content_inline="我住在上海", modality="text")
    ev.append(rec)
    EP = '{"topic":"住处","episode":"user 说他住在上海","domains":[]}'
    OK = ('{"atoms":[{"text":"user 住在上海","holder":"user","object_type":"fact",'
          '"kind":"K01","quote":"我住在上海"}]}')
    cb = build_cell(FakeLLM([EP, OK]), FakeEmbedder(), ev, cells, atoms_st,
                    [rec], session_id="floor2")
    assert len(cb.atoms) == 1 and cb.atoms[0].source == "w2"


def test_supplied_message_ids_replay_original_records(db):
    env = Env(db)
    writer = env.writer(RoutingLLM())
    messages = [FeedMsg(speaker="user", text="Yes"), FeedMsg(speaker="assistant", text="Yes")]
    first = writer.feed_batch(messages, now_dt=_T0, evidence_ids=["ev_first", "ev_second"])
    replay = writer.feed_batch(messages, evidence_ids=["ev_first", "ev_second"])
    assert replay.evidence_ids == first.evidence_ids
    assert [r.model_dump() for r in replay.records] == [env.ev.get(eid).model_dump() for eid in first.evidence_ids]
    assert len(writer.seg) == 2 and len(env.ev.list()) == 2
    assert replay.boundary is None and replay.closed_cell is None


def test_message_identity_validation_happens_before_writes(db):
    import pytest

    env = Env(db)
    writer = env.writer(RoutingLLM())
    messages = [FeedMsg(speaker="user", text="Yes")]
    for ids in [[], [""], ["ev_" * 30], ["ev_one", "ev_two"]]:
        with pytest.raises(ValueError, match="evidence_ids"):
            writer.feed_batch(messages, evidence_ids=ids)
    assert env.ev.list() == [] and writer.seg == []


def test_invalid_episode_type_is_reported_as_generation_failure(db):
    env = Env(db)
    writer = env.writer(RoutingLLM(episode=['{"episode":["not a narrative"]}']))
    first = writer.feed("user", "I live in Oslo", now_dt=_T0)
    result = build_cell(writer.llm, writer.embedder, env.ev, env.cells, env.atoms,
                        writer.seg, session_id="t")
    assert result.skipped and result.cell is None
    assert result.skip_reason == "Episode generation failed"
    assert "episode must be a string" in result.gen["call1"]["raw"]
    assert env.ev.get(first.evidence_id) is not None
    assert env.cells.iter_all() == []


def test_classified_skip_keeps_its_reason_separate_from_failure(db):
    env = Env(db)
    writer = env.writer(RoutingLLM(episode=['{"skip":true,"reason":"Only task instructions"}']))
    writer.feed("user", "Move the button", now_dt=_T0)
    result = build_cell(writer.llm, writer.embedder, env.ev, env.cells, env.atoms,
                        writer.seg, session_id="t")
    assert result.skipped and result.cell is None
    assert result.skip_reason == "Only task instructions"
    assert result.gen["skip"] == "Only task instructions"
    assert env.cells.iter_all() == []
