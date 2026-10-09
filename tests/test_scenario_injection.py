"""Rules for caller-supplied scenario injection (CallerContext.scenario).

Two hard rules, verified at every stage:
1. An empty scenario leaves the prompt **byte-for-byte identical** to today's, so the default path
   is unaffected and existing callers stay compatible.
2. A non-empty scenario makes the scenario block plus that stage's directive appear in the system
   prompt, inserted before the anchor section.

Seven high-leverage stages are covered: W1 segmentation, W2 (1) episode, W2 (2) atom, R0 rewrite,
R5 answering, the deep-track agent system prompt, and profile consolidation. Each prompt assembly
is tested directly through with_scenario without calling a real LLM, since assembly is a pure
string transformation and the real-value behaviour is checked separately by end-to-end scripts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.online import deep_recall as D
from personos.online import profile_consolidate as P
from personos.online import retrieval as R
from personos.online import write_path as W
from personos.online.llm import _SCEN_HEADER, with_scenario
from personos.online.recall_flow import run_recall
from personos.online.write_path import FeedMsg

from .test_retrieval import Env as REnv
from .test_retrieval import TableEmbedder, _v
from .test_write_path import ATOMS_OK, BOUNDARY_END, EPISODE_OK
from .test_write_path import Env as WEnv

_SCEN = "饮食健康 App:用户记录三餐并咨询饮食习惯,重点记饮食偏好/忌口/热量目标。"
_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)

# (prompt under test, anchor, that stage's directive) -- one entry per assembly point in the source
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
    """Empty scenario: every stage's prompt is byte-for-byte unchanged, so the default path is
    unaffected."""
    for prompt, anchor, directive in _CASES:
        assert with_scenario(prompt, anchor, "", directive) is prompt
        assert with_scenario(prompt, anchor, "   ", directive) is prompt


def test_scenario_block_inserted_before_anchor():
    """With a scenario: the scenario header, the scenario text, and the directive all appear, and
    the whole block is inserted before the anchor section."""
    for prompt, anchor, directive in _CASES:
        out = with_scenario(prompt, anchor, _SCEN, directive)
        assert _SCEN_HEADER in out
        assert _SCEN in out
        assert directive in out
        # The anchor is still present, with the scenario block ahead of it
        assert anchor in out
        assert out.index(_SCEN_HEADER) < out.index(anchor)
        # The body of the original prompt is intact (everything after the anchor is preserved)
        assert prompt.split(anchor, 1)[1] in out


def test_anchor_all_present_in_source_prompts():
    """Each anchor must really exist in its prompt, otherwise with_scenario falls back to appending
    at the tail and the block lands in the wrong place."""
    for prompt, anchor, _ in _CASES:
        assert anchor in prompt, f"anchor {anchor!r} is not present in the target prompt"


def test_missing_anchor_falls_back_to_tail():
    """Missing anchor (someone renames a heading later): the scenario block falls back to being
    appended at the end, and is never silently dropped."""
    out = with_scenario("# Role\nhello", "# No Such Anchor", _SCEN, "dir")
    assert out.startswith("# Role\nhello")          # the original text stays in front
    assert _SCEN_HEADER in out and _SCEN in out     # the scenario block is appended, not lost
    assert out.rstrip().endswith("dir")             # appended at the end, with the directive last


def test_agent_prompt_escapes_braces():
    """Deep-track agent system prompt: braces inside the scenario are escaped so that
    ChatPromptTemplate does not read them as template variables.

    Otherwise any {x} beyond the real {tools}/{tool_names} makes template parsing raise KeyError.
    """
    tmpl = D._agent_prompt("关注 {calories} 与 {macros} 字段")   # a scenario containing braces
    # Format after partialing the two real variables; the {calories} in the scenario must not
    # raise KeyError
    msg = tmpl.partial(tools="T", tool_names="t1").format(input="hi",
                                                          agent_scratchpad=[])
    assert "calories" in msg and "macros" in msg


def test_agent_prompt_empty_reuses_constant():
    """Empty scenario: the deep track reuses the module-level constant template, byte for byte."""
    assert D._agent_prompt("") is D._AGENT_PROMPT
    assert D._agent_prompt("  ") is D._AGENT_PROMPT


# -- End-to-end threading: drive the real SessionWriter / run_recall to prove every call site
#    actually passes the scenario through --

class _RecWriteLLM:
    """Records the system prompt seen by each of the three write-path stages and returns canned
    JSON."""

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
        raise AssertionError(f"unknown write-path prompt: {sysp[:40]!r}")


def test_write_side_threads_scenario(db):
    """feed_batch(scenario=) makes all three system prompts -- segmentation, episode, atom --
    carry the scenario block."""
    w = WEnv(db).writer(_RecWriteLLM())
    w.feed_batch([FeedMsg(speaker="user", text="今早吃了燕麦配蓝莓")], now_dt=_T, scenario=_SCEN)
    # The second batch changes topic -> W1 fires a boundary (should_end) -> the first segment is
    # closed -> W2 runs episode + atom
    w.feed_batch([FeedMsg(speaker="user", text="换个事:周末想去爬山")], now_dt=_T, scenario=_SCEN)
    llm = w.llm
    for stage in ("boundary", "episode", "atoms"):
        assert _SCEN in llm.sys[stage], f"write-path {stage} did not get the scenario injected"


class _RecRecallLLM:
    """Records the system prompt seen by each recall-side stage and returns canned JSON."""

    def __init__(self):
        self.sys: dict[str, str] = {}

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        if "query preprocessor" in sysp:
            self.sys["rewrite"] = sysp
            return ('{"resolved":"早餐吃了什么","subject":"user","expansions":[],'
                    '"time_start":null,"time_end":null,"domains":[]}')
        if "answer reviewer" in sysp:       # must be matched before "answerer" (the adjudication text also contains answerer)
            self.sys["review"] = sysp
            return '{"verdict":"ok","critique":""}'
        if "answerer" in sysp:
            self.sys["answer"] = sysp
            return '{"answer":"燕麦配蓝莓。","cells":["m1"]}'
        raise AssertionError(f"unknown recall prompt: {sysp[:50]!r}")


def test_recall_side_threads_scenario_and_review_stays_neutral(db, evidence_store):
    """run_recall(scenario=) gives R0 and R5 the scenario block; R3' adjudication deliberately does
    not get it, so it stays neutral."""
    env = REnv(db)
    env.add_cell(topic="早餐", episode="user 今早吃了燕麦配蓝莓。",
                 atoms=[{"text": "早餐吃燕麦配蓝莓", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"早餐吃了什么": _v(1, 0, 0, 0)})
    llm = _RecRecallLLM()
    run_recall(llm, emb, env.atoms, env.cells, evidence_store,
               session_id="s", query="早餐吃了啥", now_dt=_T, mode="fast", scenario=_SCEN)
    assert _SCEN in llm.sys["rewrite"]          # injected into the R0 rewrite
    assert _SCEN in llm.sys["answer"]           # injected into the R5 answer
    assert _SCEN not in llm.sys.get("review", "")   # R3' adjudication stays neutral (not injected)
