"""单元组装(docs/atom-chain-design.md §5.2/§5.3):R1 atom 池 → 材料单元工位。

规则一条:每个池内 atom 反查链——≥2 节点链交给织写器(S4)织成 memcell′(LLM 只看
query+链title+atom清单+episodes,按 query 定侧重),单节点链/游离 atom → 它所在的
memcell(零 LLM);按链/格去重(不同 atom 指向同链/同格只出一个单元)。

memcell′ 是临时视图:与普通单元同构(CellHit,topic=链 title、episode=织文、
时间=成员格跨度),covers 只做引用长短映射(mN → 成员格),下游(精排/作答/核判)
对两者完全无感知。链面是纯派生物:无链/织写失败一律降级为格粒度普通单元
(该链 atoms 回各自格桶),绝不阻塞读主链。
"""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from .. import obs
from ..models import ChainInfo, MemCell, ensure_aware
from ..online.retrieval import AtomHit, CellHit
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from .llm import ChatLLM

_BOUNDARY_TITLES = 5   # 残缺提示里点名的池外链 title 数上限(提示经济学)
_MAX_EPISODES = 15     # 织写原料成员格上限(超取最近 15 格,输入头部标注截断)
import os

# 单次 recall 内并行织写的链数上限(每链一次 LLM,IO-bound;延迟 ≈ +1 段)。
# 注:这是"每请求"的 fan-out 宽度,不是全局池——worst case 并发 MAAS = recall 并发 × 本值。
_WEAVE_WORKERS = int(os.environ.get("PERSONOS_WEAVE_WORKERS", "8"))
_SPAN_FLOOR = datetime(1, 1, 1, tzinfo=timezone.utc)   # t_start 缺失的排序下界(aware)


@dataclass
class UnitAssembly:
    """一次单元组装的产物:材料单元 + 残缺提示 + 透视计数。"""
    units: list[CellHit] = field(default_factory=list)   # 池序(链/格首现位)
    boundary: str = ""        # 通用残缺提示(零 LLM;直供 R5/R3' 的 BOUNDARY 节,D-C10)
    n_pool: int = 0           # R1 池 atom 数
    n_chains: int = 0         # 池命中的 ≥2 节点链数(织写任务数)
    n_woven: int = 0          # 织写成功数
    n_plain: int = 0          # 普通单元数(含织写失败降级回来的格)


def assemble_units(
    pool: list[AtomHit], chains: ChainStore, cells: CellStore,
    llm: ChatLLM | None, *, query: str,
    beyond: list[AtomHit] | None = None,
) -> UnitAssembly:
    """池 atom → 材料单元(织写 memcell′ + 普通单元)+ 残缺提示。

    chains:链元数据源(title/n_atoms);cells:普通单元与织写原料的格来源;
    llm:织写器(≥2 节点链;None=全降级普通单元,测试/降级用);beyond:R1 池外
    命中 atom(残缺提示点名其链 title)。组装异常退化为全普通单元,绝不抛。
    """
    asm = UnitAssembly(n_pool=len(pool))
    try:
        _assemble(asm, pool, chains, cells, llm, query=query, beyond=beyond or [])
    except Exception as e:   # noqa: BLE001  组装失败:退回格粒度(链 atoms 也回各自的格),绝不抛
        logger.warning(f"单元组装失败,退化为全普通单元: {e}")
        asm.units = _plain_units(pool, cells)
        asm.boundary = ""
        asm.n_chains = asm.n_woven = 0
        asm.n_plain = len(asm.units)
    unit_detail = "\n".join(
        f"    [{i}] topic={u.cell.topic!r} atoms={len(u.atoms)}\n"
        f"      episode/织文={u.cell.episode!r}" for i, u in enumerate(asm.units, 1))
    logger.info(f"单元组装 pool={asm.n_pool} chains={asm.n_chains} woven={asm.n_woven} "
                f"plain={asm.n_plain} boundary={asm.boundary!r} q={query!r}\n"
                f"  材料单元(n={len(asm.units)}):\n{unit_detail}")
    return asm


def _assemble(asm: UnitAssembly, pool: list[AtomHit], chains: ChainStore,
              cells: CellStore, llm, *, query: str, beyond: list[AtomHit]) -> None:
    # 1) 分桶:池 atom → 链桶(≥2 节点链)或格桶(游离/单节点链/悬空 chain_id)
    infos: dict[str, ChainInfo] = {c.id: c for c, _ in chains.list_chains()}
    chain_atoms: dict[str, list[AtomHit]] = {}   # chain_id -> 池内命中 atoms(池序)
    plain_atoms: dict[str, list[AtomHit]] = {}   # cell_id -> 池内命中 atoms(池序)
    for ah in pool:
        info = infos.get(ah.atom.chain_id or "")
        if info is not None and info.n_atoms >= 2:
            chain_atoms.setdefault(info.id, []).append(ah)
        elif ah.atom.memcell_id:
            plain_atoms.setdefault(ah.atom.memcell_id, []).append(ah)
    asm.n_chains = len(chain_atoms)

    # 2) 织写(≥2 节点链各一次 LLM;失败/缺席 → 该链 atoms 回格桶,信息不丢)
    woven = _weave_all(llm, query, chain_atoms, infos, chains, cells)
    for cid, hits in chain_atoms.items():
        if cid not in woven:
            for ah in hits:
                plain_atoms.setdefault(ah.atom.memcell_id, []).append(ah)

    # 3) 单元序:沿池序首现位——链单元在链首成员位出场;同格游离 atom 与链内 atom
    #    并存时,普通单元与织写单元都保留(格 episode 与链织文粒度不同,不互替)
    units: list[CellHit] = []
    emitted: set[tuple[str, str]] = set()
    for ah in pool:
        a = ah.atom
        key = (("chain", a.chain_id) if a.chain_id in woven
               else ("cell", a.memcell_id))
        if key in emitted:
            continue
        if key[0] == "chain":
            emitted.add(key)
            units.append(woven[a.chain_id])
        else:
            cell = cells.get(a.memcell_id)
            if cell is None:
                continue                                   # 孤儿 atom(cell 已删):跳过不崩
            emitted.add(key)
            units.append(_plain_unit(cell, plain_atoms[a.memcell_id]))
    asm.units = units
    asm.n_woven = sum(1 for u in units if u.covers)
    asm.n_plain = len(units) - asm.n_woven

    # 4) 通用残缺提示(D-C10):池外 atom 里只算【事实真不在材料里】的——
    #    已织链的池外成员(织写覆盖全链)与所在格已在材料里的 atom 都不算缺料,
    #    否则会给下游一个假的"缺料"信号(误导 R5/R3' 的 count/枚举判断)
    covered = {cid for u in units for cid in (u.covers or [u.cell.id])}
    missing = [ah for ah in beyond
               if (ah.atom.chain_id or "") not in woven
               and ah.atom.memcell_id not in covered]
    asm.boundary = _boundary_note(missing, infos)


def _plain_units(pool: list[AtomHit], cells: CellStore) -> list[CellHit]:
    """组装失败兜底:全池按格去重的普通单元(链 atoms 也回各自的格,按池序首现位)。"""
    per_cell: dict[str, list[AtomHit]] = {}
    order: list[str] = []
    for ah in pool:
        cid = ah.atom.memcell_id
        if not cid:
            continue
        if cid not in per_cell:
            per_cell[cid] = []
            order.append(cid)
        per_cell[cid].append(ah)
    out = []
    for cid in order:
        cell = cells.get(cid)
        if cell is not None:
            out.append(_plain_unit(cell, per_cell[cid]))
    return out


def _plain_unit(cell: MemCell, hits: list[AtomHit]) -> CellHit:
    """普通单元:该格命中的池内 atoms(相似度降序),分 = 池内最佳 atom。"""
    atoms = sorted(hits, key=lambda h: -h.similarity)
    best = max(hits, key=lambda h: h.rrf)
    return CellHit(cell=cell, score=best.rrf, best_sim=best.similarity, atoms=atoms)


# —— 织写(§5.3,D-C4):线程池并行,每链一次 LLM ——

def _weave_all(llm, query: str, chain_atoms: dict[str, list[AtomHit]],
               infos: dict[str, ChainInfo], chains: ChainStore,
               cells: CellStore) -> dict[str, CellHit]:
    """织全部 ≥2 节点链 → {chain_id: 织写单元}。失败链不入返回值(调用方降级)。

    现行栈同步,不引 asyncio——查询侧延迟 ≈ +1 段,不是 +N。DB 读取全在调用线程
    串行完成:共享连接非线程安全,并发读会把 pymysql 协议状态打乱。单链失败只降级
    该链(WARNING),绝不阻塞读主链。
    """
    if llm is None or not chain_atoms:
        return {}
    prepped = []          # (chain_id, members, member_cells, episodes, truncated) —— 先串行取料
    for chain_id in chain_atoms:
        try:
            members, member_cells, episodes, trunc = _gather(chains, cells, infos[chain_id])
            prepped.append((chain_id, members, member_cells, episodes, trunc))
        except Exception as e:   # noqa: BLE001  原料获取失败:该链降级,不挡其余
            logger.warning(f"织写原料获取失败,链 «{infos[chain_id].title}» "
                           f"退化为成员格普通单元: {e}")
    out: dict[str, CellHit] = {}
    if not prepped:
        return out
    # 保上下文线程池:让并行织写的 maas.chat 正确 nest 到 recall 根 trace(否则甩成孤儿根 trace)
    with obs.ContextThreadPoolExecutor(max_workers=min(_WEAVE_WORKERS, len(prepped))) as ex:
        futs = {ex.submit(weave_chain, llm, query=query, chain=infos[chain_id],
                          members=members, episodes=episodes, truncated=trunc):
                (chain_id, member_cells)
                for chain_id, members, member_cells, episodes, trunc in prepped}
        for fut in as_completed(futs):
            chain_id, member_cells = futs[fut]
            try:
                out[chain_id] = _synth_unit(infos[chain_id], member_cells,
                                            chain_atoms[chain_id], fut.result())
            except Exception as e:   # noqa: BLE001  织写失败:该链降级,不挡其余
                logger.warning(f"织写失败,链 «{infos[chain_id].title}» "
                               f"退化为成员格普通单元: {e}")
    return out


def _gather(chains: ChainStore, cells: CellStore, info: ChainInfo):
    """一条链的织写原料(调用线程串行执行):整链 atom(链序)+ 成员格全集。

    episodes 只含最近 ≤15 格(超则在头部标注截断);atom 清单与 covers 用全集
    ——前者是覆盖自检口径,后者是 cited 展开口径(§5.2)。
    """
    members = chains.full_chain(info.id)       # 整链捞出,链序
    cell_by_id: dict[str, MemCell] = {}
    for a in members:                          # 成员格去重保序(链序)
        if a.memcell_id and a.memcell_id not in cell_by_id:
            c = cells.get(a.memcell_id)
            if c is not None:
                cell_by_id[a.memcell_id] = c
    member_cells = list(cell_by_id.values())
    shown = sorted(member_cells, key=lambda c: ensure_aware(c.t_start) or _SPAN_FLOOR)
    trunc = None
    if len(shown) > _MAX_EPISODES:
        trunc = (_MAX_EPISODES, len(shown))
        shown = shown[-_MAX_EPISODES:]                    # 最近 15 格
    return members, member_cells, [(c, c.episode or "") for c in shown], trunc


def _synth_unit(info: ChainInfo, cells: list[MemCell],
                hits: list[AtomHit], woven: str) -> CellHit:
    """织文 → 织写单元(cell.id=链id,topic=链title,episode=织文,covers=成员格全集)。"""
    starts = [t for t in (ensure_aware(c.t_start) for c in cells) if t]
    ends = [t for t in (ensure_aware(c.t_end) for c in cells) if t]
    atoms = sorted(hits, key=lambda h: -h.similarity)
    best = max(hits, key=lambda h: h.rrf)
    return CellHit(
        cell=MemCell(id=info.id, topic=info.title, episode=woven,
                     t_start=min(starts) if starts else None,
                     t_end=max(ends) if ends else None),
        score=best.rrf, best_sim=best.similarity,    # 相关性代表 = 池内最佳成员
        atoms=atoms, covers=[c.id for c in cells])


def _boundary_note(missing: list[AtomHit], infos: dict[str, ChainInfo]) -> str:
    """D-C10 通用残缺提示(零 LLM):池外还有【材料未覆盖】的命中 atom,点名其 ≥2 节点链 title。

    入参已过滤:已织链成员与所在格已在材料里的 atom 不算缺料(见 _assemble 第 4 步)。
    """
    if not missing:
        return ""
    titles: list[str] = []
    for ah in missing:
        info = infos.get(ah.atom.chain_id or "")
        if (info is not None and info.n_atoms >= 2 and info.title
                and info.title not in titles):
            titles.append(info.title)
            if len(titles) >= _BOUNDARY_TITLES:
                break
    titled = f" (chains: {', '.join(titles)})" if titles else ""
    return (f"{len(missing)} more matching atoms exist beyond the shown materials{titled}. "
            "The shown materials may not cover the full set — enumerate what is shown "
            "and state the boundary when the list cannot be completed.")


# —— 织写器(§5.3,D-C4)——

_WEAVE_SYSTEM = """# Role
You are the chain weaver in a memory read pipeline: ONE fact-chain holds the statements a user made
about the same matter over time (oldest → newest). Weave its member segments' episodes into ONE
coherent narrative that will serve as answer material.

# Input
- Query: the current question — it decides EMPHASIS only (what to expand vs compress), never what
  to keep.
- Chain title, and the chain's atoms in chain order, each with its date.
- Source episodes of the member segments — the ONLY fact source.

# Hard rules
1. Integrate, never adjudicate. Old and new statements are BOTH kept, each with its own date.
   A "correction" may be written only when the dialogue itself explicitly corrected an earlier
   statement (render it as: first said X, later corrected to Y). You never pick a winner —
   downstream answer-time resolution is the designed fallback; do not preempt it.
2. Coverage first. Before writing, walk the atom checklist: EVERY atom's fact must be traceable
   in your narrative. A missing chain fact is a defect.
3. Episodes are the only fact source. Anything an atom claims that NO source episode contains must
   not be written (extraction noise must not become hallucination). Better to omit one sentence
   than to invent.
4. Emphasis = detail level, not selection. Expand what the query touches, compress the rest to a
  clause — but no chain fact is dropped.
5. Language follows the source episodes. Third-person narration. Time double-annotated: every
   dated statement carries its absolute date inline (e.g. 2026-08-10).

# Output
The woven narrative ONLY — no preamble, no headings, no commentary. Episode-style prose."""


def weave_chain(llm: ChatLLM, *, query: str, chain: ChainInfo, members: list,
                episodes: list[tuple[MemCell, str]],
                truncated: tuple[int, int] | None = None) -> str:
    """织一条链:整链 atom 清单 + 成员格 episode 原料 → 一段织文(episode 格式)。

    members:整链 atom(链序,各带 occurrence_time);episodes:(成员格, 其 episode) 列表;
    truncated:(实给格数, 成员格总数) —— 超 15 格截最近时标注。空织文视为失败(抛 ValueError)。
    """
    atom_lines = []
    for a in members:
        when = a.occurrence_time.strftime("%Y-%m-%d") if getattr(a, "occurrence_time", None) else ""
        atom_lines.append(f"[{when}] {a.text}")
    ep_lines = []
    for c, ep in episodes:
        ep_lines.append(f"—— segment {c.id} · topic: {c.topic or '(no topic)'} ——\n{ep or '(empty)'}")
    # 截断消解:atom 清单是全集而 episodes 只给最近 15 格——被截段所属 atom 按所示
    # episode 能支撑的程度写,写不出就略过,不算覆盖缺陷(封死 Hard rule 2 vs 3 的矛盾)
    cut = (f"\n(showing the most recent {truncated[0]} of {truncated[1]} segments; earlier ones "
           f"truncated — atoms dating from those hidden segments count as covered when the shown "
           f"episodes support them; otherwise omit them silently, that omission is not a defect)\n"
           if truncated else "\n")
    user = (f"—— Query ——\n{query}\n\n"
            f"—— Chain: {chain.title or '(untitled)'} ——\n"
            f"atoms (oldest → newest), the coverage checklist:\n" + "\n".join(atom_lines)
            + f"\n\n—— Source episodes ({len(episodes)} segments) ——{cut}" + "\n".join(ep_lines))
    with obs.stage("weave_chain"):
        text = llm.chat([{"role": "system", "content": _WEAVE_SYSTEM},
                         {"role": "user", "content": user}], temperature=0.2, max_tokens=900).strip()
    logger.info(f"LLM[weave_chain] 链«{chain.title}»\n  ── 输入(user)──\n{user}\n  ── 输出(织文)──\n{text}")
    if not text:
        raise ValueError("织写返回空文本")
    return text
