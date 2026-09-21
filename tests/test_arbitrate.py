"""R3' 核判单测:三档 verdict / ok 契约 / 解析失败保守 answer_defect / 材料与草稿渲染。

核判与作答分离(两个调用、两种性格),这里只测核判器本身:草稿+同款材料逐条核对。
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.models import MemCell, MemoryAtom
from personos.online.arbitrate import review_answer
from personos.online.retrieval import AtomHit, CellHit, MemoryAnswer

from .fakes import FakeLLM


def _hit(topic="画展", episode="Caroline 筹备画展", atom_text="展期定在 2026-09",
         when=None) -> CellHit:
    c = MemCell(topic=topic, episode=episode, t_start=datetime(2026, 8, 20, tzinfo=timezone.utc),
                t_end=datetime(2026, 8, 21, tzinfo=timezone.utc))
    a = MemoryAtom(memcell_id=c.id, text=atom_text, occurrence_time=when)
    return CellHit(cell=c, score=0.02, best_sim=0.8, atoms=[AtomHit(atom=a, similarity=0.8)])


def test_verdict_and_critique_parse():
    llm = FakeLLM(['{"verdict":"answer_defect","critique":"材料共 4 处地点(c2,c5,c7,c9),'
                   '草稿只列 2 处——补全"}'])
    res = review_answer(llm, query="在哪些地方练瑜伽", draft=MemoryAnswer(answer="两处。"),
                        hits=[_hit()])
    assert res.verdict == "answer_defect"
    assert "补全" in res.critique


def test_ok_forces_empty_critique():
    """ok 契约:就算模型嘴硬给了 critique 也清空(无指正)。"""
    llm = FakeLLM(['{"verdict":"ok","critique":"顺手提一句措辞可以更好"}'])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="答"), hits=[_hit()])
    assert res.verdict == "ok" and res.critique == ""


def test_insufficient_empty_critique_gets_synthesized_gap():
    """真机踩坑:草稿已如实拒答时核判常不写指正;critique 是深轨交接的缺口方向,
    空值丢方向——从问题+格数合成客观缺口。"""
    llm = FakeLLM(['{"verdict":"insufficient_material","critique":""}'])
    res = review_answer(llm, query="我的驾照什么时候到期?", draft=MemoryAnswer(answer="无法确定。"),
                        hits=[_hit(), _hit()])
    assert res.verdict == "insufficient_material"
    assert "驾照" in res.critique and "2" in res.critique   # 带问题与格数,深轨有方向
    # 模型自己写了缺口 → 原样保留,不被覆盖
    llm2 = FakeLLM(['{"verdict":"insufficient_material","critique":"缺驾照到期日"}'])
    res2 = review_answer(llm2, query="驾照", draft=MemoryAnswer(answer="无法确定。"), hits=[_hit()])
    assert res2.critique == "缺驾照到期日"


def test_unknown_verdict_degrades_to_conservative_defect():
    llm = FakeLLM(['{"verdict":"maybe","critique":""}'])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="答"), hits=[_hit()])
    assert res.verdict == "answer_defect"
    assert "re-answer" in res.critique                     # 保守指正:重答一次是便宜兜底


def test_parse_failure_keeps_raw_and_defect():
    llm = FakeLLM(["核判器跑偏了,不是 JSON"])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="答"), hits=[_hit()])
    assert res.verdict == "answer_defect"
    assert "核判器跑偏" in res.raw                          # 溯源:保留模型原文


def test_prompt_carries_draft_materials_and_subject():
    """user prompt:DRAFT ANSWER 节 + 材料块(编号分隔行+头+episode,atoms 不进)+ 疑问主体。"""
    llm = FakeLLM(['{"verdict":"ok"}'])
    res = review_answer(llm, query="画展哪天", draft=MemoryAnswer(answer="2026 年 9 月。"),
                        hits=[_hit(atom_text="展期 2026-09")],
                        resolved="Caroline 的画展哪天", subject="Caroline")
    u = res.user
    assert "DRAFT ANSWER\n2026 年 9 月。" in u              # 草稿进 prompt(核对对象)
    assert "━━━ m1 ━━━" in u                                # 编号强分隔行(快链窗口 mN)
    assert "[dialogue 2026-08-20 to 2026-08-21 | topic: 画展]" in u
    assert "Caroline 筹备画展" in u                         # episode 主料
    assert "展期" not in u                                  # atom 文本不进材料
    assert "QUESTION SUBJECT\nCaroline" in u                # 归属锚点进材料
    assert "Resolved question (references resolved)" in u   # resolved ≠ query 时单列


def test_empty_draft_marked_as_refusal():
    """空草稿 → prompt 里显式标成 refusal,核判按拒答路径查材料。"""
    llm = FakeLLM(['{"verdict":"ok"}'])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="  "), hits=[_hit()])
    assert "DRAFT ANSWER\n(the draft is EMPTY — treat as a refusal)" in res.user


def test_review_prompt_carries_itemized_checks():
    """四条硬核对 + 客观评判护栏 + 双保守校准(限事实疑点)的护栏:防后续改版悄悄丢条款。"""
    from personos.online.arbitrate import _REVIEW_SYS
    for clause in ("1. Grounded:", "2. Complete:", "3. Current:", "4. Refusals:",
                   "Judge facts objectively", "not a defect",
                   "BREVITY IS NOT A DEFECT",                      # 松绑:简短但答到点 ≠ 缺陷
                   "never convert phrasing, hedging, tense, or brevity into a defect",
                   "Unsure whether a CORE claim is grounded",      # 事实疑点保守(仅核心主张)
                   'Unsure whether coverage is "full enough" → ok',  # 丰满度疑点反向:放行
                   "Unsure between answer_defect and insufficient_material"):
        assert clause in _REVIEW_SYS
