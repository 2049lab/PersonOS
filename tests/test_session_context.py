"""会话历史滚动压缩:触发、覆盖水位、复用、最近轮逐字、硬截断兜底。用 FakeLLM 保持确定性。"""

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
    # 未触发:无摘要行,全部逐字(4 条记录)
    assert all(holder != "summary" for holder, _ in h)
    assert len(h) == 4


def test_compress_triggers_and_keeps_recent_verbatim(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)                    # 6 轮
    llm = FakeLLM(["【压缩摘要】用户聊了若干事"])
    h = build_history(ev, "s1", llm=llm, cap=80, keep_recent=2)  # 小 cap 逼触发,保留最近 2 轮
    assert h[0] == ("summary", "【压缩摘要】用户聊了若干事")   # 首行是摘要
    # 最近 2 轮(4 条记录)逐字保留在尾部
    tail = [t for holder, t in h if holder != "summary"]
    assert "用户第5句" in tail[-2] and "助手第5句" in tail[-1]
    # 水位落库:covered = 6 - 2 = 4 轮
    assert SessionContextStore(db).get("s1") == ("【压缩摘要】用户聊了若干事", 4)


def test_summary_reused_no_new_llm_call(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)
    build_history(ev, "s1", llm=FakeLLM(["S1"]), cap=80, keep_recent=2)   # 首次压缩
    # 无新增轮次 → 不该再压缩;给一个空 FakeLLM,若被调用会抛(队列空)
    h = build_history(ev, "s1", llm=FakeLLM([]), cap=80, keep_recent=2)
    assert h[0] == ("summary", "S1")        # 复用已存摘要


def test_hard_truncate_fallback(db):
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)
    big = "长" * 500
    llm = FakeLLM([big, big])                # 两次压缩都超 cap → 硬截断
    h = build_history(ev, "s1", llm=llm, cap=100, keep_recent=2)
    summary = h[0][1]
    assert h[0][0] == "summary"
    assert summary.endswith("…[truncated]") and len(summary) <= 100 + 13


def test_dated_block_carries_date_anchor():
    """压缩输入按 [日期] holder: text 渲染,给 LLM 时间锚以把相对时间换算成绝对;输出行仍是 (holder,text)。"""
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
    assert "[2026-08-20] user: 上周搬家了" in block        # 带绝对日期锚
    assert "[2026-08-20] assistant: 搬到哪了？" in block
    assert _out_lines(turns) == [("user", "上周搬家了"), ("assistant", "搬到哪了？")]  # 输出仍 2-tuple


# —— 视频段折叠成 episode(cell_store 给了才生效;不给保持老行为)——

class _FakeCells:
    """按 session 返回预置 memcell;只实现 build_history 用到的 list_session。"""

    def __init__(self, by_session):
        self._m = by_session

    def list_session(self, sid):
        return list(self._m.get(sid, []))


def _cell(cid, episode, ev_ids, t_start=None):
    from personos.models import EvidenceRef, MemCell
    return MemCell(id=cid, session_id="s1", topic="t", episode=episode, t_start=t_start,
                   evidence_refs=[EvidenceRef(evidence_id=e) for e in ev_ids])


def _vid(ev, sid, holder, text):
    """写一条视频行证据,返回它的 id(用于挂到 memcell 的 evidence_refs 上)。"""
    r = EvidenceRecord(holder=holder, content_inline=text, modality="video",
                       source={"session_id": sid})
    ev.append(r)
    return r.id


def test_video_lines_collapse_into_one_episode_turn(db):
    """一段视频的 20+ 逐行 → 历史里只占一轮 episode。

    不折叠的话:holder 是人名(不会被并轮)、raw_clip 那条 content_inline 为空,
    历史就成了一堆逐字对白 + 空轮,又吵又没叙事。
    """
    ev = EvidenceStore(db)
    ids = [_vid(ev, "s1", "Bob", f"台词{i}") for i in range(20)]
    ids.append(_vid(ev, "s1", "", ""))          # raw_clip:无 holder 无正文
    cells = _FakeCells({"s1": [_cell("c1", "Bob 抱着篮球进屋弄脏了作业本", ids)]})

    h = build_history(ev, "s1", llm=FakeLLM([]), cell_store=cells)
    assert h == [("video", "Bob 抱着篮球进屋弄脏了作业本")], h


def test_video_collapse_is_opt_in(db):
    """不传 cell_store → 老行为逐字(向前兼容,老调用方不受影响)。"""
    ev = EvidenceStore(db)
    for i in range(3):
        _vid(ev, "s1", "Bob", f"台词{i}")
    h = build_history(ev, "s1", llm=FakeLLM([]))
    assert len(h) == 3 and all(holder == "Bob" for holder, _ in h)


def test_mixed_session_keeps_text_verbatim_and_appends_episode(db):
    """混合 session:文本仍逐字,视频折叠成 episode 追加在后面。"""
    ev = EvidenceStore(db)
    _add(ev, "s1", "user", "我们来看段视频")
    _add(ev, "s1", "assistant", "好的")
    ids = [_vid(ev, "s1", "Bob", "台词A"), _vid(ev, "s1", "Alice", "台词B")]
    cells = _FakeCells({"s1": [_cell("c1", "两人为作业本吵了一架", ids)]})

    h = build_history(ev, "s1", llm=FakeLLM([]), cell_store=cells)
    assert h == [("user", "我们来看段视频"), ("assistant", "好的"),
                 ("video", "两人为作业本吵了一架")], h


def test_text_cell_not_duplicated_as_episode(db):
    """文本段的 memcell 不得被当成视频段再塞一遍 —— 它的行已经逐字在历史里了。"""
    ev = EvidenceStore(db)
    _add(ev, "s1", "user", "我住在上海")
    txt_ids = [r.id for r in ev.by_session("s1")]
    cells = _FakeCells({"s1": [_cell("c_text", "user 说他住在上海", txt_ids)]})

    h = build_history(ev, "s1", llm=FakeLLM([]), cell_store=cells)
    assert h == [("user", "我住在上海")], f"文本 cell 的 episode 不该进历史:{h}"


def test_covered_watermark_survives_video_collapse(db):
    """压缩水位不被折叠打乱:covered 是按**文本轮**攒的,视频轮恒在其后。

    这是折叠方案最容易出错的地方——若视频轮插进文本轮中间,turns[covered:] 会切错位置,
    已压缩过的内容会重新出现在 tail 里(重复)或该保留的被吞掉(丢失)。
    """
    ev = EvidenceStore(db)
    _turns(ev, "s1", 6)
    llm = FakeLLM(["【摘要】早前聊过的事"])
    build_history(ev, "s1", llm=llm, cap=80, keep_recent=2)     # 先压一轮,写下 covered
    covered_before = SessionContextStore(db, user_id="").get("s1")[1]
    assert covered_before > 0

    # 视频落库(session_end 之后),再取历史
    ids = [_vid(ev, "s1", "Bob", "台词A")]
    cells = _FakeCells({"s1": [_cell("c1", "视频里两人吵架", ids)]})
    h = build_history(ev, "s1", llm=FakeLLM([]), cap=100000, keep_recent=2, cell_store=cells)

    assert SessionContextStore(db, user_id="").get("s1")[1] == covered_before, "水位不该被折叠改动"
    assert h[0][0] == "summary"
    assert h[-1] == ("video", "视频里两人吵架"), h[-1]
    # 已被摘要覆盖的早期轮不得重新出现在逐字 tail 里
    assert not any("用户第0句" in t for holder, t in h if holder != "summary"), h
