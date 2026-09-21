"""ChainBook(⑤b):身份链的证据台账(会话内事务性绑定)。

移植自 mneme anchor/chains.py,逐处对齐:链承载会话内身份(roster 续接维持),全局归属只是
链的**当前假设**;本模块维护链的证据台账,并决定"新最佳证据"刷新触发——纯相对比较,不引新阈值、
不调模型;评估调用与终审在编排层 / commit.py。

有意偏离(对齐 draft.py):store→DraftStore(Redis 草稿);素材以 oss_key 存草稿,做卡片时经
media_store 取 b64(复用 recognize._read_b64);向量类型走 personos types(FacePick.crop_b64/
body_crop_b64、VoiceSample.wav_bytes,非 mneme face_b64/body_b64/wav_ref)。
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from loguru import logger

from personos.identity.draft import DraftStore, read_b64
from personos.identity.screenplay import ClipScript
from personos.identity.types import CandidateCard, CastEvidence, FacePick, VoiceSample

# 触发原因常量(写入 chain 评估 reason)。
FIRST_SEEN = "first_seen"
BETTER_FACE = "better_face"
BETTER_VOICE = "better_voice"
FIRST_NAME = "first_name"

# 计票箱:只统计证据触发的评估;碰撞重裁(collision_*)与终审(final_*)不计。
# 强多数 = 稳定的证据共识,供终审弃权兜底 + 碰撞时保护强侧。
EVIDENCE_REASONS = frozenset({FIRST_SEEN, BETTER_FACE, BETTER_VOICE, FIRST_NAME})
MAJORITY_MIN_COUNT = 3
MAJORITY_MIN_RATIO = 0.6


class ChainBook:
    """一个会话的身份链台账。draft=会话草稿存储;media_store 用于把草稿素材 oss_key 还原成 b64。"""

    def __init__(self, store: DraftStore, media_store: Any = None) -> None:
        self.store = store
        self.media_store = media_store

    def _b64(self, oss_key: str) -> str:
        return read_b64(self.media_store, oss_key)

    # ── 台账更新 + 刷新决策(s2 后,无模型调用)────────────────────────
    def observe_clip(
        self, session_id: str, clip_index: int, script: ClipScript,
        evidence_by_cast: dict[str, CastEvidence],
    ) -> dict[str, list[str]]:
        refreshed: dict[str, list[str]] = {}
        for cast_id in script.session_cast_ids():
            self.store.ensure_chain(session_id, cast_id)
            canonical = self.store.canonical_chain(self.store.chain_ref(session_id, cast_id))
            chain = self.store.get_chain(canonical)
            evidence = evidence_by_cast.get(cast_id) or CastEvidence(cast_id=cast_id)
            decl = self._decl(script, cast_id)

            # 台账更新:best q / named / desc / presence 无论是否触发刷新都记录。
            updates: dict[str, Any] = {}
            best_face = max((p for p in evidence.faces if p.embedding is not None),
                            key=lambda p: p.q, default=None)
            face_improved = best_face is not None and best_face.q > chain["best_face_q"]
            if face_improved:
                updates["best_face_q"] = best_face.q
            best_voice = max((v for v in evidence.voices if v.embedding is not None),
                             key=lambda v: v.q, default=None)
            voice_improved = best_voice is not None and best_voice.q > chain["best_voice_q"]
            if voice_improved:
                updates["best_voice_q"] = best_voice.q
            # WHY(对齐 mneme):被介绍是别人对话里的强名字证据,首次得名必须触发 FIRST_NAME,
            # 否则被介绍的名字永不带链回到与注册档的仲裁。spoken/self_introduction/visible_text
            # 仍不触发(弱证据)。
            name_first = bool(
                decl and decl.name
                and decl.name_evidence in ("explicit_dialogue", "introduction")
                and not chain["named"])
            if name_first:
                updates["named"] = 1
            if decl and decl.desc:
                updates["desc_text"] = decl.desc
            updates["presence"] = sorted(set(chain["presence"]) | {clip_index})

            # 刷新决策:first_seen = 该链根从未被观测(presence 空→非空),已含"首次最佳证据",
            # 故不与 better_*/first_name 并列。
            reasons: list[str] = []
            if not chain["presence"]:
                reasons.append(FIRST_SEEN)
            if FIRST_SEEN not in reasons:
                if face_improved:
                    reasons.append(BETTER_FACE)
                if voice_improved:
                    reasons.append(BETTER_VOICE)
                if name_first:
                    reasons.append(FIRST_NAME)

            self.store.update_chain(canonical, **updates)
            if reasons:
                merged = refreshed.setdefault(canonical, [])
                merged.extend(r for r in reasons if r not in merged)
                logger.debug(f"链 {canonical} 在 clip {clip_index} 刷新: {reasons}")
        return refreshed

    @staticmethod
    def _decl(script: ClipScript, cast_id: str):
        for local_id, mapped in script.cast_map.items():
            if mapped == cast_id:
                decl = script.cast_decl(local_id)
                if decl is not None:
                    return decl
        return None

    # ── verdict 应用(只改假设,不注册)────────────────────────────────
    def apply_verdict(self, chain_ref: str, verdict: str, *, session_id: str,
                      clip_index: int, reason: str, issues: list[str]) -> str:
        canonical = self.store.canonical_chain(chain_ref)
        chain = self.store.get_chain(canonical)
        # 同框守卫(与终审同规,提前到 clip 级):s4 候选池是跨 cast 的并集,模型可能把两条同框链
        # 判成同一人——合并不可逆,之后碰撞规则会把它们看作一条链复现,永久对错误失明。
        is_merge = self.store.is_chain_ref(verdict)
        target = self.store.canonical_chain(verdict) if is_merge else None
        rejected = (is_merge and target != canonical and self.copresent(canonical, target))
        if rejected:
            issues = [*issues, f"copresent merge rejected: {target}"]
        self.store.add_chain_evaluation(
            canonical, session_id=session_id, clip_index=clip_index, reason=reason,
            verdict=verdict, issues=issues,
            evidence={"best_face_q": chain["best_face_q"],
                      "best_voice_q": chain["best_voice_q"], "named": chain["named"]})
        if rejected:
            return canonical                             # 假设不变
        if is_merge:
            if target != canonical:
                self.store.merge_chain(canonical, target)
            return target
        self.store.update_chain(canonical, hypothesis=verdict, hypo_method=reason)
        return canonical

    # ── 评估材料(query 卡 / 合成召回证据 / pending 候选卡)──────────────
    def query_card(self, session_id: str, cast_id: str, script: ClipScript, *,
                   evidence: Optional[CastEvidence] = None) -> dict[str, Any]:
        canonical = self.store.canonical_chain(self.store.chain_ref(session_id, cast_id))
        chain = self.store.get_chain(canonical) or {}
        pair = self.store.best_pair(canonical) or {}
        # s4 评估在 s5 暂存前:首见时暂存为空,重评时本 clip 证据尚未存——当前证据质量更高时用当前
        # 图,否则用暂存最佳。脸与全身取同一更高 q 源(暂存 vs 当前),但两图不保证同一瞬间。
        staged_q = float(pair["quality"]) if pair.get("quality") is not None else -1.0
        best_face = evidence.best_face() if evidence is not None else None
        if best_face is not None and best_face.q > staged_q:
            best_body = evidence.best_body()
            face_b64 = best_face.crop_b64 or ""
            body_b64 = (best_body.body_crop_b64 or "") if best_body else ""
        else:
            face_b64 = self._b64(str(pair.get("face_oss_key") or ""))
            body_b64 = self._b64(str(pair.get("body_oss_key") or ""))
        names = self.store.names_for(canonical)
        decl = self._decl(script, cast_id)
        key_lines = [line.text for line in script.lines
                     if line.kind == "speech" and script.cast_map.get(line.who) == cast_id][:3]
        # 声纹样本同策:暂存最佳 vs 本 clip 证据,q 高者胜(必须带真 wav)。
        staged_voice = self.store.best_voice(canonical) or {}
        staged_voice_q = (float(staged_voice["quality"])
                          if staged_voice.get("quality") is not None else -1.0)
        cur_voice = evidence.best_voice_wav() if evidence is not None else None
        if cur_voice is not None and cur_voice.q > staged_voice_q:
            voice_b64 = base64.b64encode(cur_voice.wav_bytes).decode() if cur_voice.wav_bytes else ""
        else:
            voice_b64 = self._b64(str(staged_voice.get("oss_key") or ""))
        return {
            "cast_id": cast_id,
            "desc": chain.get("desc_text") or (decl.desc if decl else ""),
            "name": (decl.name if decl and decl.name else None) or (names[0] if names else None),
            "key_lines": key_lines,
            "face_b64": face_b64,
            "body_b64": body_b64,
            "voice_b64": voice_b64,
        }

    def synthetic_evidence(self, chain_ref: str) -> CastEvidence:
        """从暂存最佳素材合成一份证据(供大库粗召回;只需 embedding+q)。"""
        canonical = self.store.canonical_chain(chain_ref)
        evidence = CastEvidence(cast_id=canonical)
        for asset in self.store.active_staged(canonical, "face")[:1]:   # 已按 q 降序
            if asset.get("embedding") is not None:
                evidence.faces.append(FacePick(t=float(asset.get("t") or 0),
                                               embedding=asset["embedding"], q=float(asset["q"])))
        for asset in self.store.active_staged(canonical, "voice")[:1]:
            if asset.get("embedding") is not None:
                evidence.voices.append(VoiceSample(t0=float(asset.get("t0") or 0),
                                                   t1=float(asset.get("t1") or 0),
                                                   embedding=asset["embedding"], q=float(asset["q"])))
        return evidence

    def copresent(self, ref_a: str, ref_b: str) -> bool:
        a = self.store.get_chain(self.store.canonical_chain(ref_a))
        b = self.store.get_chain(self.store.canonical_chain(ref_b))
        if a is None or b is None:
            return False
        return bool(set(a["presence"]) & set(b["presence"]))

    def vote_summary(self, chain_ref: str) -> dict[str, Any]:
        """证据触发评估的计票(canonical + 别名合并;链-链合并 verdict 不计)。"""
        canonical = self.store.canonical_chain(chain_ref)
        counts: dict[str, int] = {}
        total = 0
        for ref in (canonical, *self.store.aliases_of(canonical)):
            for row in self.store.evaluations_for(ref):
                reasons = set(str(row["reason"] or "").split("+"))
                if not reasons & EVIDENCE_REASONS:
                    continue
                verdict = str(row["verdict"] or "")
                if not verdict or self.store.is_chain_ref(verdict):
                    continue
                total += 1
                counts[verdict] = counts.get(verdict, 0) + 1
        return {"counts": counts, "total": total}

    def strong_majority(self, chain_ref: str) -> Optional[str]:
        """强多数档:票数≥MAJORITY_MIN_COUNT 且占比≥MAJORITY_MIN_RATIO 的 character_id
        (NEW 从不作目标,但计入分母)。"""
        summary = self.vote_summary(chain_ref)
        total = summary["total"]
        if not total:
            return None
        target, count = max(((t, c) for t, c in summary["counts"].items() if t != "NEW"),
                            key=lambda item: item[1], default=(None, 0))
        if target is None:
            return None
        if count >= MAJORITY_MIN_COUNT and count / total >= MAJORITY_MIN_RATIO:
            return target
        return None

    def pending_cards(self, session_id: str, *, for_chain: str) -> list[CandidateCard]:
        """其它 pending 链作候选(供断裂链回场时并入);曾同框的链物理上不可能同人,排除。"""
        cards: list[CandidateCard] = []
        for chain in self.store.pending_chains(session_id):
            ref = chain["chain_ref"]
            if ref == self.store.canonical_chain(for_chain) or self.copresent(ref, for_chain):
                continue
            pair = self.store.best_pair(ref) or {}
            voice = self.store.best_voice(ref) or {}
            names = self.store.names_for(ref)
            cards.append(CandidateCard(
                character_id=ref, name=names[0] if names else "",
                desc=chain["desc_text"] or "",
                face_b64=self._b64(str(pair.get("face_oss_key") or "")),
                body_b64=self._b64(str(pair.get("body_oss_key") or "")),
                voice_b64=self._b64(str(voice.get("oss_key") or "")),
                last_seen_session=session_id))
        return cards


# ── 链级碰撞规则:同框链共享假设(检测 / 重裁 / 证据优先降级)──────────────
def _chain_proposed(book: ChainBook, script: ClipScript, session_id: str) -> dict[str, str]:
    from personos.identity.inspect import present_casts
    proposed: dict[str, str] = {}
    for cast_id in present_casts(script):
        canonical = book.store.canonical_chain(book.store.chain_ref(session_id, cast_id))
        chain = book.store.get_chain(canonical)
        if chain is None:
            continue
        proposed[cast_id] = (chain["hypothesis"] if chain["hypothesis"] != "NEW" else canonical)
    return proposed


def resolve_chain_collisions(book: ChainBook, script: ClipScript, *, session_id: str,
                             clip_index: int, queries_by_cast: dict[str, dict[str, Any]],
                             pool: list[CandidateCard], omni: Any,
                             dump: Any = None) -> list[dict[str, Any]]:
    """同框链共享假设的检测→重裁(≤2 轮)→证据优先降级。逐处对齐 mneme。"""
    from personos.identity.inspect import inspect_bind_collisions
    from personos.identity.recognize import build_arbitration_prompt, parse_verdicts
    log: list[dict[str, Any]] = []

    def _violations() -> list:
        vs = inspect_bind_collisions(script, _chain_proposed(book, script, session_id))
        # 两 cast 在同一 canonical 上"自撞"不算(同一链复现):过滤 cast→canonical 同值的。
        return [v for v in vs if len({
            book.store.canonical_chain(book.store.chain_ref(session_id, c))
            for c in v.cast_ids}) > 1]

    violations = _violations()
    if not violations:
        return log
    candidate_ids = [card.character_id for card in pool]

    # 保护强侧:争议档上恰有一条链握强多数时,它是证据最支持的身份——不进重裁、不降级。
    def _contested_target(violation):
        proposed = _chain_proposed(book, script, session_id)
        targets = {proposed.get(c) for c in violation.cast_ids}
        targets.discard(None)
        return next(iter(targets)) if len(targets) == 1 else None

    def _protected(violation) -> set[str]:
        target = _contested_target(violation)
        if not target or book.store.is_chain_ref(target):
            return set()
        strong = [c for c in violation.cast_ids
                  if book.strong_majority(book.store.chain_ref(session_id, c)) == target]
        return set(strong) if len(strong) == 1 else set()

    for attempt in (1, 2):
        conflicted = [c for v in violations for c in v.cast_ids
                      if c in queries_by_cast and c not in _protected(v)]
        conflicted = list(dict.fromkeys(conflicted))
        if not conflicted or not pool:
            break
        prompt, images, audios = build_arbitration_prompt(
            [queries_by_cast[c] for c in conflicted], pool)
        prompt += ("\n\nCONSTRAINT VIOLATION FEEDBACK:\n"
                   + "\n".join(f"- {v.detail}" for v in violations)
                   + "\nPeople appearing together in the same clip are physically"
                     " distinct people — they can never be the same character."
                     " Re-judge ONLY the queries above: pick a different registered"
                     " character, or NEW if unsure.")
        if dump:
            dump(f"s4_chain_rearbitration_prompt_a{attempt}.txt", prompt)
        try:
            raw = omni.chat(prompt, images_b64=images, audio_b64_list=audios, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            log.append({"kind": "chain_bind", "attempt": attempt,
                        "outcome": "call_failed", "error": str(exc)})
            break
        if dump:
            dump(f"s4_chain_rearbitration_raw_a{attempt}.txt", raw)
        verdicts, issues = parse_verdicts(raw, cast_ids=conflicted, candidate_ids=candidate_ids)
        for cast_id, verdict in verdicts.items():
            book.apply_verdict(book.store.chain_ref(session_id, cast_id), verdict,
                               session_id=session_id, clip_index=clip_index,
                               reason="collision_rearbitration", issues=issues)
        log.append({"kind": "chain_bind", "attempt": attempt,
                    "outcome": "rearbitrated", "verdicts": verdicts})
        violations = _violations()
        if not violations:
            return log
    # 证据优先降级:保强多数链,否则保 best_face_q 最高(早出场破平);其余降 NEW
    # (错误的拆分可恢复,错误的合并不可)。
    for violation in violations:
        chains = {c: book.store.get_chain(book.store.canonical_chain(
            book.store.chain_ref(session_id, c))) for c in violation.cast_ids}
        protected = _protected(violation)
        keep = (next(iter(protected)) if protected else
                max(chains, key=lambda c: (chains[c]["best_face_q"],
                                           -min(chains[c]["presence"] or [10**9]))))
        demoted = []
        for cast_id, chain in chains.items():
            if cast_id == keep:
                continue
            book.apply_verdict(chain["chain_ref"], "NEW", session_id=session_id,
                               clip_index=clip_index, reason="collision_degrade",
                               issues=[violation.detail])
            demoted.append(cast_id)
        log.append({"kind": "chain_bind", "outcome": "degraded", "keep": keep,
                    "demoted": demoted, "detail": violation.detail})
    return log
