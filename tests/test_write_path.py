"""写入链路单测:W0 落证据 / W1 边界(切分·安全阀·保守延续) / W2 cell 生成(降级·校验·quote 回链) / D1 不查重。

不调真实 MAAS:RoutingLLM 按 system prompt 把边界/episode/atom 三类调用路由到各自响应队列。
"""

from __future__ import annotations

from datetime import datetime

from personos.models import now
from personos.online.write_path import (
    FeedMsg, SessionWriter, _match_evidence_refs, append_utterance, build_cell, detect_boundary,
)
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder, FakeLLM

_T0 = datetime(2026, 8, 25, 10, 0)


class RoutingLLM:
    """按 system prompt 特征路由到 boundary/episode/atoms 三个独立响应队列;记录调用序。"""

    def __init__(self, boundary=(), episode=(), atoms=()):
        self.q = {"boundary": list(boundary), "episode": list(episode), "atoms": list(atoms)}
        self.calls: list[str] = []

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        kind = ("boundary" if "boundary detector" in sysp
                else "episode" if "episode weaver" in sysp
                else "atoms" if "atomic-memory extractor" in sysp else "other")
        assert kind != "other", f"未知 system prompt: {sysp[:40]}"
        self.calls.append(kind)
        assert self.q[kind], f"未预期的 {kind} 调用(队列已空)"
        resp = self.q[kind].pop(0)
        return resp(messages[-1]["content"]) if callable(resp) else resp


class Env:
    """一个隔离库 + 三 store,write_path 单测的公共底座。"""

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
    """段首句无界可判:不调 LLM,boundary=None。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])   # 若误调会 pop 失败/留下未用响应
    r = env.writer(llm).feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    assert r.boundary is None and not r.forced_close and r.closed_cell is None
    assert llm.calls == []


def test_boundary_end_closes_cell_and_starts_new_segment(db):
    """W1 判切 → 旧段 W2 建 cell,新句归新段。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备她和 Rob 的联合画展", now_dt=_T0)
    r = w.feed("user", "对了,我最近开始跑跑步机了", now_dt=datetime(2026, 8, 25, 10, 8))
    assert llm.calls == ["boundary", "episode", "atoms"]
    assert r.closed_cell is not None and r.boundary.should_end
    assert len(w.seg) == 1 and w.seg[0].content_inline.startswith("对了")   # 新句已入新段
    # 产物落库可查:cell 时序、atoms 归属 cell、quote 命中证据
    cell = env.cells.get(r.closed_cell.cell.id)
    assert cell.topic.startswith("Caroline") and cell.domains == ["D13"]
    cas = env.atoms.list_by_cell(cell.id)
    assert len(cas) == 2
    assert cas[0].holder == "Caroline" and cas[0].object_type == "fact"
    assert cas[0].evidence_refs and cas[0].evidence_refs[0].evidence_id in {
        e.id for e in env.ev.by_session("t")}
    assert cell.evidence_refs and env.cells.list_session("t") == [cell]


def test_boundary_continue_keeps_segment(db):
    """W1 判延续 → 不建 cell,段增长。"""
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
    assert d.should_end is False   # 保守延续:错闭合伤 episode,延续有安全阀


def test_max_turns_forced_close_without_boundary_llm(db):
    """安全阀:段满 max_turns 后的下一句强制闭合,该句不调 W1。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm, max_turns=2)
    w.feed("user", "第一句", now_dt=_T0)                       # 段首:无调用
    w.feed("user", "第二句", now_dt=_T0)                       # 段内 1<2:调 W1 判延续
    r = w.feed("user", "第三句", now_dt=_T0)                   # 段内 2>=2:强制闭合(无 W1)
    assert llm.calls == ["boundary", "episode", "atoms"]       # 强制闭合那次没有 boundary 调用
    assert r.forced_close and r.boundary is None and r.closed_cell is not None


def test_end_session_closes_open_segment(db):
    """session 末强制闭合:W1 分段覆盖整个会话的保证。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    w.feed("user", "展期下个月", now_dt=_T0)
    assert env.cells.list_session("t") == []
    cells = w.end_session()
    assert len(cells) == 1 and not w.seg
    w.end_session()   # 已无未闭合段:幂等,不再调 LLM
    assert llm.calls == ["boundary", "episode", "atoms"]


# —— W2 ——

def test_w2_call1_failure_falls_back_to_transcript_episode(db):
    """Call① 失败 → 原话当 episode(信息不丢),topic 用边界 hint 兜底。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=["### 坏了 ###"], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "换个话题", now_dt=_T0)
    cell = r.closed_cell.cell
    assert "user: 我在帮 Caroline 筹备画展" in cell.episode      # _transcript 原话渲染
    assert cell.topic == "画展筹备"                                # 边界 topic_summary 兜底
    assert len(env.atoms.list_by_cell(cell.id)) == 2              # Call② 仍正常跑


def test_w2_call2_failure_still_leaves_a_retrieval_anchor(db):
    """Call② 三次重试全失败 → cell 照常落库,**且仍留一条概括 atom 当检索触角**。

    不变量已改(此前是"atoms 留空"):快链只检索 atom 向量,0 atom 的 cell 彻底不可达——
    episode 躺在库里却任何问法都召不到。抽取失败不该让整段记忆消失。
    """
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK],
                     atoms=["### 坏了 ###", "还是坏的", "依旧坏的"])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "换个话题", now_dt=_T0)
    assert env.cells.get(r.closed_cell.cell.id) is not None
    got = env.atoms.list_by_cell(r.closed_cell.cell.id)
    assert len(got) == 1 and got[0].source == "w2_fallback", got
    assert llm.calls.count("atoms") == 3                      # num_tries=3 用满


def test_w2_call2_retries_then_succeeds(db):
    """Call② 首次输出非 JSON → 重试拿到合法 JSON → atoms 正常落库(信息不丢)。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK],
                     atoms=["先是一段废话", ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "换个话题", now_dt=_T0)
    assert len(env.atoms.list_by_cell(r.closed_cell.cell.id)) == 2
    assert llm.calls.count("atoms") == 2                      # 坏一次 + 重试成功


def test_atom_system_prompt_carries_retrieval_anchor_clauses():
    """atom=检索触角(MECE 覆盖独立事实、合并复述、跳琐碎):prompt 回归护栏,防改版悄悄丢条款。
    (取代旧穷尽三条款——atom 从逐句流水账改为检索友好的 MECE 锚点,明细穷尽让渡给 episode。)"""
    from personos.online.write_path import _ATOM_SYSTEM
    for clause in ("# Coverage as retrieval anchors", "- One atom per DISTINCT fact:",
                   "- Merge repetition; never one atom per utterance:", "- Skip process trivia:",
                   "- Distinct items & attributes DO each get their own atom"):
        assert clause in _ATOM_SYSTEM


def test_write_prompts_carry_time_and_detail_clauses():
    """P1-C 两处校准护栏:atom when 锚定反例(治 off-by-one)+ episode 日期细节条款
    + 类二①覆盖纪律(逐行走查,小事实不为叙事让路)。"""
    from personos.online.write_path import _ATOM_SYSTEM, _EPISODE_SYSTEM
    assert "the week BEFORE the dialogue" in _ATOM_SYSTEM
    assert "a bare date is often the entire answer" in _EPISODE_SYSTEM
    assert "walk the transcript utterance by utterance" in _EPISODE_SYSTEM
    assert "never \"departed in mid-July\"" in _EPISODE_SYSTEM


def test_atom_fields_validated(db):
    """非法 object_type→claim、非法 kind→K01、超额 domains 截断。"""
    env = Env(db)
    bad = ('{"atoms": [{"text": "x", "object_type": "opinion", "holder": "user", '
           '"kind": "随便", "domains": ["D13", "D05", "D06", "D01"], "when": null, "quote": ""}]}')
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[bad])
    w = env.writer(llm)
    w.feed("user", "s1", now_dt=_T0)
    r = w.feed("user", "s2", now_dt=_T0)
    a = env.atoms.list_by_cell(r.closed_cell.cell.id)[0]
    assert a.object_type == "claim" and a.kind == "K01" and len(a.domains) == 3
    assert a.evidence_refs == []                                  # 空 quote 不乱挂


def test_quote_match_caps_and_skips_foreign_text(db):
    """quote 命中:子串所在证据;上限 3;命不中留空。"""
    recs = [_rec(f"第{i}句里都有画展两个字") for i in range(4)] + [_rec("无关句")]
    ids = {r.id for r in recs}
    hit = _match_evidence_refs(recs, "画展")        # 4 句都含 → 截到 3
    assert 0 < len(hit) <= 3 and all(h.evidence_id in ids for h in hit)
    assert _match_evidence_refs(recs, "完全不存在的引文") == []
    assert _match_evidence_refs(recs, "") == []


def test_when_null_falls_back_to_segment_date(db):
    """when 拿不准 → 用段起始日期(记录时间),不落空。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed("user", "换话题", now_dt=datetime(2026, 8, 25, 11, 0))
    cas = env.atoms.list_by_cell(r.closed_cell.cell.id)
    assert cas[0].occurrence_time.date() == _T0.date()    # 第一条 when="2026-08-25"
    assert cas[1].occurrence_time.date() == _T0.date()    # 第二条 when=null → seg_date


def test_no_dedup_identical_facts_both_stored(db):
    """D1 写入不查重:两个 cell 各自抽出同一条事实 → 都在(冗余索引,消费在作答时)。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END, BOUNDARY_END],
                     episode=[EPISODE_OK, EPISODE_OK], atoms=[ATOMS_OK, ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    w.feed("user", "聊聊别的", now_dt=_T0)
    w.feed("user", "又提到 Caroline 在筹备画展", now_dt=_T0)
    w.feed("user", "再聊聊别的", now_dt=_T0)
    texts = [a.text for a in env.atoms.list(limit=99)]
    assert texts.count("Caroline 在筹备与 Rob 的联合画展") == 2   # 不消解,两条并存
    assert len(env.cells.iter_all()) == 2


def test_build_cell_embeds_topic_and_atoms(db):
    """落库向量:topic 一个 + atom 逐条(全库仅有的两种向量)。"""
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


# —— episode 分类(task_type 词表)——

_EP_TYPE = ('{"topic": "t", "episode": "e", "domains": ["D13"], "episode_type": "work"}')
_EP_BADTYPE = ('{"topic": "t", "episode": "e", "domains": ["D13"], "episode_type": "made_up"}')


def test_episode_type_classified_when_task_type_given(db):
    env = Env(db)
    llm = RoutingLLM(episode=[_EP_TYPE], atoms=[ATOMS_OK])
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, [_rec("在忙工作", _T0)],
                    session_id="t", task_type=["work", "health"])
    assert cb.cell.episode_type == "work"                      # 命中词表 → 采用


def test_episode_type_falls_back_when_not_in_list(db):
    env = Env(db)
    llm = RoutingLLM(episode=[_EP_BADTYPE], atoms=[ATOMS_OK])
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, [_rec("在忙工作", _T0)],
                    session_id="t", task_type=["work", "health"])
    assert cb.cell.episode_type == "unknown"                   # LLM 输出不在词表 → 回落 unknown


def test_episode_type_unknown_without_task_type(db):
    """不传 task_type:不装配分类提示词,episode_type 默认 unknown。"""
    env = Env(db)
    llm = RoutingLLM(episode=[EPISODE_OK], atoms=[ATOMS_OK])
    cb = build_cell(llm, FakeEmbedder(), env.ev, env.cells, env.atoms, [_rec("随便聊", _T0)],
                    session_id="t")
    assert cb.cell.episode_type == "unknown"
    assert "Episode classification" not in llm_last_episode_system(cb)   # 未追加分类指令


def llm_last_episode_system(cb):
    """从 CellBuild.gen 取 W2① 实际用的 system prompt(验证条件装配)。"""
    return (cb.gen.get("call1") or {}).get("system", "")


# —— feed_batch(批原子:API /ingest 的整批语义)——

def test_feed_batch_first_batch_skips_boundary_llm(db):
    """段首批无界可判:不调 LLM,整批直接入段。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])   # 若误调会 pop 失败
    w = env.writer(llm)
    r = w.feed_batch([FeedMsg(speaker="user", text="猜猜我住哪?"),
                      FeedMsg(speaker="assistant", text="裕廊西?")], now_dt=_T0)
    assert r.boundary is None and r.closed_cell is None and not r.forced_close
    assert r.evidence_ids and len(r.evidence_ids) == 2 and len(r.records) == 2
    assert len(w.seg) == 2 and llm.calls == []


def test_feed_batch_atomic_on_topic_shift(db):
    """判切时:闭合的旧段不含本批,整批成为新段——批内 QA 永不分离。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm)
    w.feed("user", "我在帮 Caroline 筹备画展", now_dt=_T0)
    r = w.feed_batch([FeedMsg(speaker="user", text="下周三看牙医是几点?"),
                      FeedMsg(speaker="assistant", text="上午 10 点")],
                     now_dt=datetime(2026, 8, 25, 10, 8))
    assert r.boundary is not None and r.boundary.should_end and r.closed_cell is not None
    # 闭合 cell 的证据引用 = 旧段那句;QA 两条全留在新段
    closed_ids = {ref.evidence_id for ref in r.closed_cell.cell.evidence_refs}
    all_ids = {e.id for e in env.ev.by_session("t")}
    assert closed_ids == all_ids - set(r.evidence_ids)
    assert [rec.content_inline for rec in w.seg] == ["下周三看牙医是几点?", "上午 10 点"]
    assert llm.calls == ["boundary", "episode", "atoms"]


def test_feed_batch_continue_merges_whole(db):
    """判延续:整批并入旧段,不建 cell。"""
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
    """安全阀按"并入后总长"计:旧段 2 + 批 2 > max 3 → 强制闭合旧段,批归新段(不调 W1)。"""
    env = Env(db)
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm, max_turns=3)
    w.feed("user", "第一句", now_dt=_T0)
    w.feed("user", "第二句", now_dt=_T0)
    r = w.feed_batch([FeedMsg(speaker="user", text="第三句"),
                      FeedMsg(speaker="user", text="第四句")], now_dt=_T0)
    assert r.forced_close and r.boundary is None and r.closed_cell is not None
    # 第二句单喂时 1+1≤3 仍走 W1;批到达时 2+2>3 → 强制闭合,批本身不调 W1
    assert len(w.seg) == 2 and llm.calls == ["boundary", "episode", "atoms"]


def test_zero_atoms_triggers_one_retry_then_accepts(db):
    """抽到 0 atom 就重抽一次;第二次仍 0 才认定确实无可记(不无限重试)。

    为什么:atom 是检索触角,一条都没有 = 这个 cell 任何查询都召不回,episode 还在但找不到。
    而 chat_json 的 num_tries 只管 JSON 解析失败,合法的 {"atoms":[]} 会被直接放行。
    """
    from personos.models import EvidenceRecord

    U2 = "vtest_zero_atom"
    ev, cells, atoms_st = EvidenceStore(db, U2), CellStore(db, U2), AtomStore(db, U2)
    rec = EvidenceRecord(holder="user", content_inline="我住在上海", modality="text")
    ev.append(rec)
    EP = '{"topic":"住处","episode":"user 说他住在上海","domains":[]}'
    A_EMPTY, A_OK = '{"atoms":[]}', ('{"atoms":[{"text":"user 住在上海","holder":"user",'
                                     '"object_type":"fact","kind":"K01","quote":"我住在上海"}]}')

    # ① 首次空 → 重抽拿到 1 条
    llm = FakeLLM([EP, A_EMPTY, A_OK])
    cb = build_cell(llm, FakeEmbedder(), ev, cells, atoms_st, [rec], session_id="zs1")
    assert len(cb.atoms) == 1, "首次抽空应重抽一次并采纳结果"

    # ② 两次都空 → 不再抽第三次(FakeLLM 只喂两条 atom 响应,多抽会 IndexError),
    #    但**不接受 0**:落一条概括 atom 兜底(有 cell 就必须有检索触角)
    llm2 = FakeLLM([EP, A_EMPTY, A_EMPTY])
    cb2 = build_cell(llm2, FakeEmbedder(), ev, cells, atoms_st, [rec], session_id="zs2")
    assert len(cb2.atoms) == 1 and cb2.atoms[0].source == "w2_fallback", cb2.atoms


def test_zero_atoms_falls_back_to_one_summary_atom(db):
    """铁律:**有 memcell 就必须有至少一条 atom**(否则该段永久不可召回)。

    快链 R1 只检索 atom 向量,cell 的 topic_embedding 只在深轨对已选中子集用 ——
    0 atom 的 cell 在快链里彻底不可达:episode 还躺在库里,但任何问法都召不到。
    实测事故:单 clip 视频段稳定抽 0 atom,换 4 种问法全 miss。
    """
    from personos.models import EvidenceRecord
    from personos.storage.atom_store import AtomStore
    from personos.storage.cell_store import CellStore
    from personos.storage.evidence_store import EvidenceStore

    U2 = "vtest_atom_floor"
    ev, cells, atoms_st = EvidenceStore(db, U2), CellStore(db, U2), AtomStore(db, U2)
    rec = EvidenceRecord(holder="Alice", content_inline="Bob 又把我的笔记本弄脏了", modality="video")
    ev.append(rec)
    EP = '{"topic":"Bob 弄脏了 Alice 的笔记本","episode":"两人为笔记本争执","domains":["D07"]}'
    EMPTY = '{"atoms":[]}'

    # 抽取与重抽都返回空 → 仍须落一条概括 atom
    cb = build_cell(FakeLLM([EP, EMPTY, EMPTY]), FakeEmbedder(), ev, cells, atoms_st,
                    [rec], session_id="floor1")
    assert len(cb.atoms) == 1, "0 atom 时必须合成概括 atom 兜底"
    a = cb.atoms[0]
    assert a.text == "Bob 弄脏了 Alice 的笔记本", a.text          # 用 topic 当概括
    assert a.source == "w2_fallback", "合成的要可区分,便于统计触发频率"
    assert a.memcell_id == cb.cell.id
    assert [r.evidence_id for r in a.evidence_refs] == [rec.id], "溯源要挂到全段证据"
    # 真落库了才算数(检索取的是库里的向量,不是返回值)
    assert any(x.id == a.id for x in atoms_st.list(limit=50))


def test_fallback_not_used_when_extraction_works(db):
    """正常抽到 atom 时不得触发兜底 —— 兜底只在"一条都没有"时出手。"""
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
