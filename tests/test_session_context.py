"""Rolling compression of session history: triggering, the coverage watermark, reuse, keeping the
most recent turns verbatim, and the hard-truncation fallback. FakeLLM keeps this deterministic."""

from personos.models import EvidenceRecord
from personos.online.session_context import build_history
from personos.storage.evidence_store import EvidenceStore
from personos.storage.session_store import SessionContextStore
from tests.fakes import FakeLLM


def _add(ev, sid, holder, text):
    ev.append(EvidenceRecord(holder=holder, content_inline=text, source={"session_id": sid}))


def _turns(ev, sid, n):
    for i in range(n):
        _add(ev, sid, "user", f"用户第{i}句话内容比较长用来撑字符数XXXXXXXXXX")
        _add(ev, sid, "assistant", f"助手第{i}句回复")


def test_no_compress_when_under_cap(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 2)
    h = build_history(ev, "s1", llm=FakeLLM([]), cap=100000, keep_recent=3)
    # Not triggered: no summary line, everything verbatim (4 records)
    assert all(holder != "summary" for holder, _ in h)
    assert len(h) == 4


def test_compress_triggers_and_keeps_recent_verbatim(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)                    # 6 turns
    llm = FakeLLM(["【压缩摘要】用户聊了若干事"])
    h = build_history(ev, "s1", llm=llm, cap=80, keep_recent=2)  # a small cap forces the trigger; keep the last 2 turns
    assert h[0] == ("summary", "【压缩摘要】用户聊了若干事")   # the first line is the summary
    # The last 2 turns (4 records) are kept verbatim at the tail
    tail = [t for holder, t in h if holder != "summary"]
    assert "用户第5句" in tail[-2] and "助手第5句" in tail[-1]
    # The watermark is persisted: covered = 6 - 2 = 4 turns
    assert SessionContextStore(db).get("s1") == ("【压缩摘要】用户聊了若干事", 4)


def test_summary_reused_no_new_llm_call(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)
    build_history(ev, "s1", llm=FakeLLM(["S1"]), cap=80, keep_recent=2)   # first compression
    # No new turns -> no recompression; an empty FakeLLM raises if it is called (empty queue)
    h = build_history(ev, "s1", llm=FakeLLM([]), cap=80, keep_recent=2)
    assert h[0] == ("summary", "S1")        # the stored summary is reused


def test_hard_truncate_fallback(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)
    big = "长" * 500
    llm = FakeLLM([big, big])                # both compressions exceed the cap -> hard truncation
    h = build_history(ev, "s1", llm=llm, cap=100, keep_recent=2)
    summary = h[0][1]
    assert h[0][0] == "summary"
    assert summary.endswith("…[truncated]") and len(summary) <= 100 + 13


def test_dated_block_carries_date_anchor():
    """The compression input is rendered as `[date] holder: text`, giving the LLM a time anchor so
    it can convert relative times into absolute ones; output lines stay as (holder, text)."""
    from datetime import datetime, timezone

    from personos.models import EvidenceRecord
    from personos.online.session_context import _dated_block, _out_lines, _pair_turns

    recs = [
        EvidenceRecord(holder="user", content_inline="上周搬家了",
                       captured_at=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)),
        EvidenceRecord(holder="assistant", content_inline="搬到哪了？",
                       captured_at=datetime(2026, 8, 20, 9, 1, tzinfo=timezone.utc)),
    ]
    turns = _pair_turns(recs)
    block = _dated_block(turns)
    assert "[2026-08-20] user: 上周搬家了" in block        # carries an absolute date anchor
    assert "[2026-08-20] assistant: 搬到哪了？" in block
    assert _out_lines(turns) == [("user", "上周搬家了"), ("assistant", "搬到哪了？")]  # output is still 2-tuples


# -- Collapsing a video segment into an episode (only active when cell_store is supplied;
#    without it the old behaviour is preserved) --

class _FakeCells:
    """Returns preset memcells per session; implements only the list_session call that
    build_history uses."""

    def __init__(self, by_session):
        self._m = by_session

    def list_session(self, sid):
        return list(self._m.get(sid, []))


def _cell(cid, episode, ev_ids, t_start=None):
    from personos.models import EvidenceRef, MemCell
    return MemCell(id=cid, session_id="s1", topic="t", episode=episode, t_start=t_start,
                   evidence_refs=[EvidenceRef(evidence_id=e) for e in ev_ids])


def _vid(ev, sid, holder, text):
    """Write one video-line evidence record and return its id, for attaching to a memcell's
    evidence_refs."""
    r = EvidenceRecord(holder=holder, content_inline=text, modality="video",
                       source={"session_id": sid})
    ev.append(r)
    return r.id


def test_video_lines_collapse_into_one_episode_turn(db):
    """20-plus individual lines from one video collapse into a single episode turn in the history.

    Without collapsing, the holder is a person's name (so the lines never pair into turns) and the
    raw_clip record has an empty content_inline, so the history becomes a pile of verbatim dialogue
    plus an empty turn -- noisy and with no narrative.
    """
    ev = EvidenceStore(db)
    ids = [_vid(ev, "s1", "Bob", f"台词{i}") for i in range(20)]
    ids.append(_vid(ev, "s1", "", ""))          # raw_clip: no holder, no body text
    cells = _FakeCells({"s1": [_cell("c1", "Bob 抱着篮球进屋弄脏了作业本", ids)]})

    h = build_history(ev, "s1", llm=FakeLLM([]), cell_store=cells)
    assert h == [("video", "Bob 抱着篮球进屋弄脏了作业本")], h


def test_video_collapse_is_opt_in(db):
    """Without cell_store the old verbatim behaviour applies, so existing callers are unaffected."""
    ev = EvidenceStore(db)
    for i in range(3):
        _vid(ev, "s1", "Bob", f"台词{i}")
    h = build_history(ev, "s1", llm=FakeLLM([]))
    assert len(h) == 3 and all(holder == "Bob" for holder, _ in h)


def test_mixed_session_keeps_text_verbatim_and_appends_episode(db):
    """Mixed session: text stays verbatim and the video collapses into an episode appended after
    it."""
    ev = EvidenceStore(db)
    _add(ev, "s1", "user", "我们来看段视频")
    _add(ev, "s1", "assistant", "好的")
    ids = [_vid(ev, "s1", "Bob", "台词A"), _vid(ev, "s1", "Alice", "台词B")]
    cells = _FakeCells({"s1": [_cell("c1", "两人为作业本吵了一架", ids)]})

    h = build_history(ev, "s1", llm=FakeLLM([]), cell_store=cells)
    assert h == [("user", "我们来看段视频"), ("assistant", "好的"),
                 ("video", "两人为作业本吵了一架")], h


def test_text_cell_not_duplicated_as_episode(db):
    """A memcell from a text segment must not be re-inserted as if it were a video segment -- its
    lines are already in the history verbatim."""
    ev = EvidenceStore(db)
    _add(ev, "s1", "user", "我住在上海")
    txt_ids = [r.id for r in ev.by_session("s1")]
    cells = _FakeCells({"s1": [_cell("c_text", "user 说他住在上海", txt_ids)]})

    h = build_history(ev, "s1", llm=FakeLLM([]), cell_store=cells)
    assert h == [("user", "我住在上海")], f"the episode of a text cell must not enter the history: {h}"


def test_covered_watermark_survives_video_collapse(db):
    """Collapsing must not disturb the compression watermark: covered counts **text turns** only,
    and video turns always come after them.

    This is where the collapsing design is easiest to get wrong -- if a video turn were inserted in
    the middle of the text turns, turns[covered:] would slice at the wrong position, so already
    summarized content would reappear in the tail (duplication) or content that should be kept
    would be swallowed (loss).
    """
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)
    llm = FakeLLM(["【摘要】早前聊过的事"])
    build_history(ev, "s1", llm=llm, cap=80, keep_recent=2)     # compress once first, writing covered
    covered_before = SessionContextStore(db, user_id="").get("s1")[1]
    assert covered_before > 0

    # The video is persisted (after session_end), then the history is fetched again
    ids = [_vid(ev, "s1", "Bob", "台词A")]
    cells = _FakeCells({"s1": [_cell("c1", "视频里两人吵架", ids)]})
    h = build_history(ev, "s1", llm=FakeLLM([]), cap=100000, keep_recent=2, cell_store=cells)

    assert SessionContextStore(db, user_id="").get("s1")[1] == covered_before, "collapsing must not move the watermark"
    assert h[0][0] == "summary"
    assert h[-1] == ("video", "视频里两人吵架"), h[-1]
    # Early turns already covered by the summary must not reappear in the verbatim tail
    assert not any("用户第0句" in t for holder, t in h if holder != "summary"), h
