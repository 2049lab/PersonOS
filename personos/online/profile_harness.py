"""画像补丁校验器(harness):对整理产的补丁做**纯字段机械校验**,不解析文本、不猜内容。

干净结构下校验的是 LLM 能写错的东西:未知域 / 枚举 / 长度 / band 合法 / **出处短标真实**。
带的滚动/分带由 LLM 决定、引擎兜底条数上限——都不在此校验(结构不变量不靠 harness)。

`valid_cell_ids`:允许作为出处的**短标**集合(本轮新 cell 的 c1..cN)——LLM 只看得到这些短标,
引用集外的即幻觉出处。真实长 id 的回填在 profile_consolidate,harness 只认短标(见 id 映射规范)。

返回错误列表(空=通过);非空由 consolidate 带逐条错误打回让 LLM 重写。
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
    """校验补丁,返回错误列表(空=通过)。valid_cell_ids 是本轮 cell 的短标集(c1..cN)。"""
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
            continue                                    # 显式清空,合法
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
