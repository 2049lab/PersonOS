"""统一的会话对话历史构建:滚动压缩 + 近若干轮逐字。

替代此前 ingest / recall / arbitrate 各自为政的"取近 N 轮"与临时压缩,统一成一处:
  - 一个 session 内维护单条【滚动摘要】(覆盖式,持久化);
  - 每次构建 = 摘要 + 自上次压缩以来的全部 tail;
  - 当 len(摘要) + len(tail 除最近 keep_recent 轮外) 超过 cap → 把「旧摘要 + 老化轮次」压成新摘要,
    目标长度 ≈ cap/10;仍超则重压(最多 k 次)再不行按长度硬截断;
  - 最近 keep_recent 轮永远逐字保留、且不计入 cap(防单条巨型对话把压缩逼死)。

摘要面向【对话连续性/指代/当前诉求】,不是抽取长期事实——那是 reconcile→atom 的职责。
摘要是派生缓存,evidence 原样保留。
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from ..models import EvidenceRecord
from ..storage.evidence_store import EvidenceStore
from ..storage.session_store import SessionContextStore

DEFAULT_CAP = 100000        # 历史上下文字符上限(不含逐字保留的最近轮)
DEFAULT_KEEP_RECENT = 3    # 压缩时永远逐字保留的最近轮数
_MAX_RECOMPRESS = 2        # 压完仍超上限时的重压次数,再不行硬截断
_SUMMARY_HOLDER = "summary"
_VIDEO_HOLDER = "video"    # 视频段在历史里的 holder(一段录像 = 一轮 episode)


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], temperature: float = ..., max_tokens: int = ...) -> str: ...


_COMPACT_SYS = """# Role
You are compressing the conversation history between a personal assistant and its user, to preserve context continuity for future conversations.
Note: you are NOT extracting long-term facts (that is the memory store's job) — you keep only the context needed to continue the conversation.

# Task
Compress the given conversation history into one coherent summary, targeting roughly {target} characters (stay under it if possible).
Cover the history's features across these 9 dimensions, as MECE as you can (write what is there, skip what is not — never pad):
1. Who the user is / basic identity and current situation (only what appears in the dialogue)
2. The user's current main needs, goals and intents (including implicit ones)
3. The user's preferences, habits, constraints and taboos
4. Key people / places / things / times — entities (for later reference resolution)
5. Fact changes and corrections (moving, renaming, preference changes, …): write them as "X→Y" with the absolute date of the change
6. The user's emotions, attitudes and feedback (especially corrections, dissatisfaction, approval directed at the assistant)
7. Topics already discussed / resolved, and their conclusions
8. Open, pending topics or tasks to follow up
9. The topic being discussed right now (to carry the conversation forward)

# Requirements
- Keep specific names and numbers.
- Times always as absolute dates — convert "last week / a few days ago / last month" into absolute dates (e.g. "2026-07") using the [date] prefixed to each line; no relative time expressions may survive in the summary (the summary itself carries no time anchor — a relative phrase can never be restored later).
- Write in the third person ("the user ...").
- Language follows the source dialogue.
- Output only the summary body: no title, no numbered points, no explanations."""


def _pair_turns(recs: list[EvidenceRecord]) -> list[list[tuple[str, str, str]]]:
    """把时序证据配成"轮":非助手句(user/third_party)开一轮,紧随的助手句并入本轮。
    每行带 date(captured_at 的日期)——压缩时用它把'上周/前几天'锚成绝对日期。"""
    turns: list[list[tuple[str, str, str]]] = []
    for r in recs:
        d = r.captured_at.date().isoformat() if r.captured_at else "?"
        line = (r.holder, r.content_inline or "", d)
        if r.holder == "assistant" and turns:
            turns[-1].append(line)
        else:
            turns.append([line])
    return turns


def _video_turns(video_recs: list[EvidenceRecord], session_id: str,
                 cell_store: Any) -> list[list[tuple[str, str, str]]]:
    """视频段 → 每个视频 memcell 一轮 `(video, episode)`。

    为什么不用逐行:视频 flush 写进 evidence 的是**剧本逐行**(一段 60s 的 clip 就有 20+ 行),
    holder 是人名、captured_at 没设、raw_clip 那几条 content_inline 还是空的 —— 直接进历史
    就是 20+ 轮逐字对白外加几条空轮,又吵又没有叙事。episode 才是这段录像的"讲了什么"。

    认哪些 cell 是视频的:按 **evidence_refs 与视频证据 id 是否相交**,不能只按 session 取
    ——同一个 session 里文本段也会出 memcell,那些行已经逐字在历史里了,再塞一遍就重了。

    时序:视频 evidence 只在 session_end 的 flush 里一次性写入,恒在文本轮之后,故这些轮
    统一追加在文本轮末尾(与 by_session 的返回序一致),不会插进文本中间打乱 covered 水位。
    """
    if cell_store is None or not video_recs:
        return []
    vids = {r.id for r in video_recs if r.id}
    turns: list[list[tuple[str, str, str]]] = []
    try:
        cells = cell_store.list_session(session_id)
    except Exception as e:  # noqa: BLE001  取不到 cell 不阻塞历史构建(退化=该段无上下文)
        logger.warning(f"取 session memcell 失败,视频段不进历史: {e}")
        return []
    for cell in cells:
        refs = {getattr(r, "evidence_id", "") for r in (cell.evidence_refs or [])}
        if not (refs & vids):
            continue                     # 文本段的 cell:它的行已经逐字在历史里了
        ep = (cell.episode or "").strip()
        if not ep:
            continue
        d = cell.t_start.date().isoformat() if cell.t_start else "?"
        turns.append([(_VIDEO_HOLDER, ep, d)])
    return turns


def _turn_len(turn: list[tuple[str, str, str]]) -> int:
    return sum(len(t) for _, t, _ in turn)


def _out_lines(turns: list[list[tuple[str, str, str]]]) -> list[tuple[str, str]]:
    """下游历史输出:只要 (holder, text)(下游按 now_dt 解相对时间,逐字轮不需要行内日期)。"""
    return [(h, t) for turn in turns for (h, t, _) in turn]


def _dated_block(turns: list[list[tuple[str, str, str]]]) -> str:
    """压缩输入渲染:`[date] holder: text`——给 LLM 时间锚,便于把相对时间换算成绝对。"""
    return "\n".join(f"[{d}] {h}: {t}" for turn in turns for (h, t, d) in turn)


def _aged_text(turns: list[list[tuple[str, str, str]]]) -> str:
    """压缩失败退化拼接用:纯文本。"""
    return " ".join(t for turn in turns for (_, t, _) in turn)


def _compact(llm: ChatLLM, prior_summary: str, aged: list[list[tuple[str, str]]], target: int) -> str:
    """把[旧摘要 + 老化轮次]压成一段新摘要。失败则退化为拼接旧摘要+原文(上层再截断兜底)。"""
    block = ""
    if prior_summary:
        block += f"Existing summary (earlier history):\n{prior_summary}\n\n"
    block += ("Earlier dialogue (the [date] before each line is that utterance's absolute "
              "date):\n" + _dated_block(aged))
    sys = _COMPACT_SYS.replace("{target}", str(target))
    try:
        out = llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": block}],
                       temperature=0.2).strip()
        return out or block
    except Exception as e:  # noqa: BLE001  压缩失败不阻塞主链路
        logger.warning(f"会话历史压缩失败,退化拼接: {e}")
        return (prior_summary + " " + _aged_text(aged)).strip()


def build_history(
    evidence_store: EvidenceStore,
    session_id: str,
    *,
    llm: ChatLLM,
    cap: int = DEFAULT_CAP,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    target: int | None = None,
    cell_store: Any = None,
) -> list[tuple[str, str]]:
    """构建喂给下游的对话历史 = [摘要?] + tail 逐字。必要时滚动压缩并持久化。

    注意:ingest 须在 append 当前消息【之前】调用,才不会把当前 in-flight 消息算进历史。

    cell_store:给了就把**视频段折叠成 episode**(见 _video_turns);不给则视频行按逐字走
    (保持老行为,向前兼容)。
    """
    store = SessionContextStore(evidence_store.db, user_id=getattr(evidence_store, "user_id", ""))
    summary, covered = store.get(session_id)
    recs = evidence_store.by_session(session_id)
    if cell_store is not None:
        video = [r for r in recs if r.modality == "video"]
        turns = (_pair_turns([r for r in recs if r.modality != "video"])
                 + _video_turns(video, session_id, cell_store))
    else:
        turns = _pair_turns(recs)

    if covered > len(turns):        # 库被清过/水位错位 → 重置(容错)
        covered = 0
        summary = ""
    tail = turns[covered:]

    # 触发判据:摘要 + (tail 除最近 keep_recent 轮外) 的字符数;最近 keep_recent 轮不计入
    aged = tail[:-keep_recent] if len(tail) > keep_recent else []
    metric = len(summary) + sum(_turn_len(t) for t in aged)

    if aged and metric > cap:
        tgt = target or max(5000, cap // 10)
        new_summary = _compact(llm, summary, aged, tgt)
        # 压完仍超上限:重压(更狠的目标),最多 _MAX_RECOMPRESS 次,再不行硬截断
        attempts = 1
        while len(new_summary) > cap and attempts < _MAX_RECOMPRESS:
            new_summary = _compact(llm, new_summary, [], max(300, tgt // 2))
            attempts += 1
        if len(new_summary) > cap:
            new_summary = new_summary[:cap] + " …[truncated]"
        summary = new_summary
        covered += len(aged)
        store.save(session_id, summary, covered)
        tail = turns[covered:]
        logger.info(f"会话历史压缩 session={session_id} covered={covered} summary_len={len(summary)}")

    out: list[tuple[str, str]] = []
    if summary:
        out.append((_SUMMARY_HOLDER, summary))
    out.extend(_out_lines(tail))
    return out
