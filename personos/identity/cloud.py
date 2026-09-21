"""CloudEngine:概率身份云(移植自 mneme-release anchor/cloud.py)。

只做两件事,从不判决(判决在 MLLM 仲裁):
1. 粗召回:把大库压成 top-K 候选(产出排序,不是结论);
2. 注册基底:q 加权精度累加(见得越多认得越准),支撑跨会话认人。

无硬阈值:learn 用连续权重(糊脸自动权重趋零,不是被门拒绝),score 产出连续排序分。
数学:PFE 共轭更新 `tau'=tau+w; mu'=normalize(tau*mu + w*x)`;
打分 `max(cos mean, max cos templates)` 抗多姿态欠拟合。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from personos.identity.types import CastEvidence, normalized

# 冷启动基线 s0_m = 该模态的跨人相似度中位数(代理值);标定 LLR 表后可无缝替换。env 可覆盖。
_S0_DEFAULT = {"face": 0.15, "voice": 0.25}


def _s0(modality: str) -> float:
    env = os.getenv(f"PERSONOS_ANCHOR_S0_{modality.upper()}")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _S0_DEFAULT.get(modality, 0.2)


class CloudEngine:
    def __init__(self, store, *, template_cap: int = 12) -> None:
        self.store = store
        self.template_cap = template_cap

    # ── 学习(精度加权,无硬门)────────────────────────────────────────
    def learn(self, character_id: str, modality: str, embedding: Any, q: float,
              *, payload: dict[str, Any] | None = None) -> bool:
        """tau'=tau+w; mu'=normalize(tau*mu + w*x),w=q(1/σ²(q) 精度的代理)。

        q≈0 的观测数学上自然淡出(不是被门拒绝)。返回该观测是否真的进了云。
        """
        emb = normalized(embedding)
        weight = max(0.0, float(q))
        if emb is None or weight <= 0.0:
            return False
        mean, tau, n_obs = self.store.load_prototype(character_id, modality)
        if mean is None:
            new_mean, new_tau = emb, weight
        else:
            if mean.shape != emb.shape:
                return False
            blended = normalized(mean * tau + emb * weight)
            if blended is None:
                return False
            new_mean, new_tau = blended, tau + weight
        self.store.save_prototype(character_id, modality, new_mean, new_tau, n_obs + 1)
        self._admit_template(character_id, modality, emb, weight, payload=payload)
        return True

    def _admit_template(self, character_id: str, modality: str, emb: np.ndarray, q: float,
                        *, payload: dict[str, Any] | None = None) -> None:
        """模板集准入(无参规则):未满直接进;满员时与最近邻模板拼质量,低者被逐。

        同视角冗余被自动挤出,跨姿态/跨会话多样性自然保留,无需姿态阈值。
        """
        templates = self.store.templates(character_id, modality)
        if len(templates) < self.template_cap:
            self.store.add_template(character_id, modality, emb, q, payload=payload)
            return
        nearest = max((t for t in templates if t["embedding"] is not None),
                      key=lambda t: float(np.dot(t["embedding"], emb)), default=None)
        if nearest is None or q <= float(nearest["q"]):
            return
        self.store.remove_template(nearest["template_id"])
        self.store.add_template(character_id, modality, emb, q, payload=payload)

    # ── 打分 / 粗召回 ─────────────────────────────────────────────────
    def score_observation(self, character_id: str, modality: str,
                          embedding: Any, q: float) -> float | None:
        """一条观测对一个档案的排序分:q*(s-s0),s=max(cos mean, max cos templates)。

        该档案该模态无云 → None(缺席,调用方按 0 贡献处理)。
        """
        emb = normalized(embedding)
        if emb is None:
            return None
        mean, _tau, _n = self.store.load_prototype(character_id, modality)
        best = None
        if mean is not None and mean.shape == emb.shape:
            best = float(np.dot(mean, emb))
        for template in self.store.templates(character_id, modality):
            t_emb = template["embedding"]
            if t_emb is None or t_emb.shape != emb.shape:
                continue
            cos = float(np.dot(t_emb, emb))
            best = cos if best is None else max(best, cos)
        if best is None:
            return None
        return max(0.0, float(q)) * (best - _s0(modality))

    def score_evidence(self, character_id: str, evidence: CastEvidence) -> float:
        """score 级融合:各模态各观测组求和,缺席=0 贡献(天然抗模态缺失)。"""
        total = 0.0
        for pick in evidence.faces:
            score = self.score_observation(character_id, "face", pick.embedding, pick.q)
            if score is not None:
                total += score
        for sample in evidence.voices:
            score = self.score_observation(character_id, "voice", sample.embedding, sample.q)
            if score is not None:
                total += score
        return total

    def coarse_recall(self, evidence: CastEvidence, character_ids: list[str],
                      k: int) -> list[tuple[str, float]]:
        """top-K 排序(唯一验收指标 hit@K;排错了仲裁还能答 NEW 救)。"""
        scored = [(cid, self.score_evidence(cid, evidence)) for cid in character_ids]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, int(k))]
