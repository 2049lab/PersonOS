"""The profile patch validator (harness): runs **purely mechanical field checks** on the patch
produced by consolidation — it never parses the text and never guesses at content.

Under this clean structure, what gets checked is what the LLM can get wrong: unknown domains,
enum values, lengths, band validity, and **that the source handles are real**. Band rolling and band
assignment are the LLM's call, and the item cap is the engine's backstop — neither is checked here
(structural invariants do not rely on the harness).

`valid_cell_ids`: the set of **short handles** allowed as sources (c1..cN for this round's new
cells) — the LLM only ever sees these short handles, so a reference outside the set is a hallucinated
source. Mapping back to the real long ids happens in profile_consolidate; the harness only knows
short handles (see the id mapping convention).

Returns a list of errors (empty = passed); a non-empty list is sent back by consolidate, error by
error, for the LLM to rewrite.
"""

from __future__ import annotations

from ..storage.profile_store import BANDS, PMO16

_STATUS = {"confirmed", "inferred"}
_PMO16 = set(PMO16)
_BANDS = set(BANDS)


def _check_sources(src, valid: set[str], where: str, errs: list, *, required: bool) -> None:
    if src is None:
        if required:
            errs.append(f"{where}: sources 缺失(至少 1 个出处短标,如 c1)")
        return
    if not isinstance(src, list) or (required and not src):
        errs.append(f"{where}: sources 须为非空短标列表")
        return
    for cid in src:
        if cid not in valid:
            errs.append(f"{where}: 出处 {cid!r} 不在本轮可见 cell 短标内(疑似幻觉出处)")


def _check_text(text, limit: int, where: str, errs: list, *, required: bool) -> None:
    if text is None:
        if required:
            errs.append(f"{where}: text 缺失")
        return
    if not isinstance(text, str) or (required and not text.strip()):
        errs.append(f"{where}: text 须为非空字符串")
        return
    if len(text) > limit:
        errs.append(f"{where}: text 长度 {len(text)} 超上限 {limit}")


def validate_patch(patch: dict, *, valid_cell_ids: set[str],
                   trait_max: int = 500, fact_max: int = 500) -> list[str]:
    """Validate the patch and return a list of errors (empty = passed). valid_cell_ids is the set of
    short handles for this round's cells (c1..cN)."""
    errs: list[str] = []
    if not isinstance(patch, dict):
        return ["补丁根须为 JSON 对象"]

    # —— traits ——
    traits = patch.get("traits") or {}
    if not isinstance(traits, dict):
        errs.append("traits 须为对象(域名→侧写|null)")
        traits = {}
    for dom, val in traits.items():
        if dom not in _PMO16:
            errs.append(f"traits: 未知域 {dom!r}(只允许 PMO-16 的 16 个 key)")
            continue
        if val is None:
            continue                                    # explicit clear, which is valid
        if not isinstance(val, dict):
            errs.append(f"traits[{dom}]: 须为对象或 null")
            continue
        where = f"traits[{dom}]"
        _check_text(val.get("text"), trait_max, where, errs, required=True)
        if val.get("status") not in _STATUS:
            errs.append(f"{where}: status 须为 confirmed|inferred,得到 {val.get('status')!r}")
        _check_sources(val.get("sources"), valid_cell_ids, where, errs, required=True)

    # —— facts ——
    facts = patch.get("facts") or {}
    if not isinstance(facts, dict):
        errs.append("facts 须为对象(add/rewrite/drop)")
        facts = {}

    add = facts.get("add") or []
    if not isinstance(add, list):
        errs.append("facts.add 须为列表")
        add = []
    for i, a in enumerate(add):
        where = f"facts.add[{i}]"
        if not isinstance(a, dict):
            errs.append(f"{where}: 须为对象")
            continue
        _check_text(a.get("text"), fact_max, where, errs, required=True)
        if a.get("band") not in _BANDS:
            errs.append(f"{where}: band 须为 {sorted(_BANDS)} 之一(新增必须指明放哪个带)")
        _check_sources(a.get("sources"), valid_cell_ids, where, errs, required=True)

    rewrite = facts.get("rewrite") or []
    if not isinstance(rewrite, list):
        errs.append("facts.rewrite 须为列表")
        rewrite = []
    for i, rw in enumerate(rewrite):
        where = f"facts.rewrite[{i}]"
        if not isinstance(rw, dict):
            errs.append(f"{where}: 须为对象")
            continue
        if not rw.get("id"):
            errs.append(f"{where}: 缺 id(rewrite 必须指明改哪条 fact)")
        if "text" in rw:
            _check_text(rw.get("text"), fact_max, where, errs, required=True)
        if "sources" in rw:
            _check_sources(rw.get("sources"), valid_cell_ids, where, errs, required=True)
        if "band" in rw and rw["band"] not in _BANDS:
            errs.append(f"{where}: band 须为 {sorted(_BANDS)} 之一")

    drop = facts.get("drop") or []
    if not isinstance(drop, list) or any(not isinstance(x, str) for x in drop):
        errs.append("facts.drop 须为 f_id 字符串列表")

    return errs
