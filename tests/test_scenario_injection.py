"""业务方场景注入(CallerContext.scenario)的守则测试。

两条铁律逐环节验证:
1. 空 scenario → prompt 与今天**逐字节相同**(默认链路零影响,向前兼容)。
2. 带 scenario → 场景区块 + 该环节 directive 出现在系统提示词里,且插在锚点段之前。

覆盖 7 个高杠杆环节:W1 切段 / W2① episode / W2② atom / R0 改写 / R5 作答 /
深轨 agent 系统提示词 / Profile consolidate。用 with_scenario 直测各 prompt 装配,
不打真 LLM(装配是纯字符串变换,真值由端到端脚本另验)。
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.app.recall_flow import run_recall
from personos.online.llm import _SCEN_HEADER, with_scenario
from personos.online import write_path as W
from personos.online import retrieval as R
from personos.online import profile_consolidate as P
from personos.online import deep_recall as D
from personos.online.write_path import FeedMsg

from .test_retrieval import Env as REnv, TableEmbedder, _v
from .test_write_path import Env as WEnv, ATOMS_OK, BOUNDARY_END, EPISODE_OK

_SCEN = "饮食健康 App:用户记录三餐并咨询饮食习惯,重点记饮食偏好/忌口/热量目标。"
_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)

# (被测 prompt, 锚点, 该环节 directive)——与源码装配点一一对应
_CASES = [
    (W._BOUNDARY_SYSTEM, "# Decision dimensions (by priority)", W._SCEN_DIR_BOUNDARY),
    (W._EPISODE_SYSTEM, "# Task (on the FULL segment transcript provided)", W._SCEN_DIR_EPISODE),
    (W._ATOM_SYSTEM, "# Input", W._SCEN_DIR_ATOM),
    (R._REWRITE_SYS, "# Input", R._SCEN_DIR_REWRITE),
    (R._ANSWER_SYS, "# How to read the materials", R._SCEN_DIR_ANSWER),
    (D._AGENT_SYS, "# What the memory store looks like (four layers)", D._SCEN_DIR_AGENT),
    (P._CONSOLIDATE_SYS, "# Output: a PATCH, not a full rewrite (JSON only)", P._SCEN_DIR_PROFILE),
]


def test_empty_scenario_is_byte_identical():
    """空 scenario:各环节 prompt 逐字节不变(默认链路零影响)。"""
    for prompt, anchor, directive in _CASES:
        assert with_scenario(prompt, anchor, "", directive) is prompt
        assert with_scenario(prompt, anchor, "   ", directive) is prompt


def test_scenario_block_inserted_before_anchor():
    """带 scenario:场景头 + 原文 + directive 出现,且整块插在锚点段之前。"""
    for prompt, anchor, directive in _CASES:
        out = with_scenario(prompt, anchor, _SCEN, directive)
        assert _SCEN_HEADER in out
        assert _SCEN in out
        assert directive in out
        # 锚点仍在,且场景块在它前面
        assert anchor in out
        assert out.index(_SCEN_HEADER) < out.index(anchor)
        # 原 prompt 主体不被破坏(锚点之后内容原样保留)
        assert prompt.split(anchor, 1)[1] in out


def test_anchor_all_present_in_source_prompts():
    """锚点必须真的在各 prompt 里(否则 with_scenario 会走末尾兜底,插错位置)。"""
    for prompt, anchor, _ in _CASES:
        assert anchor in prompt, f"锚点 {anchor!r} 不在目标 prompt 中"


def test_missing_anchor_falls_back_to_tail():
    """锚点缺失(日后改标题):场景块兜底追加到末尾,绝不静默丢失。"""
    out = with_scenario("# Role\nhello", "# No Such Anchor", _SCEN, "dir")
    assert out.startswith("# Role\nhello")          # 原文保留在前
    assert _SCEN_HEADER in out and _SCEN in out     # 场景块兜底追加,未丢失
    assert out.rstrip().endswith("dir")             # 追加在末尾(directive 收尾)


def test_agent_prompt_escapes_braces():
    """深轨 agent 系统提示词:scenario 里的花括号被转义,ChatPromptTemplate 不当模板变量。

    否则 {tools}/{tool_names} 之外多出的 {x} 会让模板解析报 KeyError。
    """
    tmpl = D._agent_prompt("关注 {calories} 与 {macros} 字段")   # 含花括号的场景
    # partial 掉两个真实变量后 format,不应因场景里的 {calories} 抛 KeyError
    msg = tmpl.partial(tools="T", tool_names="t1").format(input="hi",
                                                          agent_scratchpad=[])
    assert "calories" in msg and "macros" in msg


def test_agent_prompt_empty_reuses_constant():
    """空 scenario:深轨复用模块级常量模板(逐字节不变)。"""
    assert D._agent_prompt("") is D._AGENT_PROMPT
    assert D._agent_prompt("  ") is D._AGENT_PROMPT


# —— 链路穿透:驱动真实 SessionWriter / run_recall,证明各调用点确实透传了 scenario ——

class _RecWriteLLM:
    """记录写入侧三 stage 收到的 system prompt,返回 canned JSON。"""

    def __init__(self):
        self.sys: dict[str, str] = {}

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        if "boundary detector" in sysp:
            self.sys["boundary"] = sysp
            return BOUNDARY_END
        if "episode weaver" in sysp:
            self.sys["episode"] = sysp
            return EPISODE_OK
        if "atomic-memory extractor" in sysp:
            self.sys["atoms"] = sysp
            return ATOMS_OK
        raise AssertionError(f"未知写入 prompt: {sysp[:40]!r}")


def test_write_side_threads_scenario(db):
    """feed_batch(scenario=) → 切段/episode/atom 三处 system prompt 都带上场景块。"""
    w = WEnv(db).writer(_RecWriteLLM())
    w.feed_batch([FeedMsg(speaker="user", text="今早吃了燕麦配蓝莓")], now_dt=_T, scenario=_SCEN)
    # 第二批异话题 → 触发 W1 边界(should_end)→ 闭合首段 → W2 episode+atom
    w.feed_batch([FeedMsg(speaker="user", text="换个事:周末想去爬山")], now_dt=_T, scenario=_SCEN)
    llm = w.llm
    for stage in ("boundary", "episode", "atoms"):
        assert _SCEN in llm.sys[stage], f"写入 {stage} 未注入 scenario"


class _RecRecallLLM:
    """记录召回侧各 stage 收到的 system prompt,返回 canned JSON。"""

    def __init__(self):
        self.sys: dict[str, str] = {}

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        if "query preprocessor" in sysp:
            self.sys["rewrite"] = sysp
            return ('{"resolved":"早餐吃了什么","subject":"user","expansions":[],'
                    '"time_start":null,"time_end":null,"domains":[]}')
        if "answer reviewer" in sysp:       # 必须先于 "answerer" 匹配(核判文案里也有 answerer)
            self.sys["review"] = sysp
            return '{"verdict":"ok","critique":""}'
        if "answerer" in sysp:
            self.sys["answer"] = sysp
            return '{"answer":"燕麦配蓝莓。","cells":["m1"]}'
        raise AssertionError(f"未知召回 prompt: {sysp[:50]!r}")


def test_recall_side_threads_scenario_and_review_stays_neutral(db, evidence_store):
    """run_recall(scenario=) → R0/R5 带场景块;R3' 核判刻意不带(保持中立)。"""
    env = REnv(db)
    env.add_cell(topic="早餐", episode="user 今早吃了燕麦配蓝莓。",
                 atoms=[{"text": "早餐吃燕麦配蓝莓", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"早餐吃了什么": _v(1, 0, 0, 0)})
    llm = _RecRecallLLM()
    run_recall(llm, emb, env.atoms, env.cells, evidence_store,
               session_id="s", query="早餐吃了啥", now_dt=_T, mode="fast", scenario=_SCEN)
    assert _SCEN in llm.sys["rewrite"]          # R0 改写注入
    assert _SCEN in llm.sys["answer"]           # R5 作答注入
    assert _SCEN not in llm.sys.get("review", "")   # R3' 核判中立(不注入)
