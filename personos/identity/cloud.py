"""CloudEngine: the probability identity cloud.

It does exactly two things and never decides who someone is — that verdict comes
from multimodal-LLM arbitration:

1. Coarse recall: squeeze a large library down to top-K candidates. The output is
   a ranking, not a conclusion.
2. Enrollment basis: q-weighted precision accumulation, so the more we see of
   someone the better we know them. This is what lets recognition survive across
   sessions.

There are no hard thresholds anywhere. learn() uses a continuous weight, so a
blurry face drifts toward zero weight on its own rather than being rejected by a
gate, and score() returns a continuous ranking number.

The math is the PFE conjugate update ``tau' = tau + w; mu' = normalize(tau*mu + w*x)``.
Scoring takes ``max(cos mean, max cos templates)``, which guards against
underfitting when someone appears in many poses.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from personos.identity.types import CastEvidence, normalized

# The cold-start baseline s0_m is the median cross-person similarity for that
# modality, used here as a proxy. Once an LLR table is calibrated it can drop in
# without touching anything else. Overridable by env.
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

    # ── Learning: precision-weighted, with no hard gate ──────────────────
    def learn(self, character_id: str, modality: str, embedding: Any, q: float,
              *, payload: dict[str, Any] | None = None) -> bool:
        """tau' = tau + w; mu' = normalize(tau*mu + w*x), with w = q as a proxy for the precision 1/sigma^2(q).

        An observation with q close to 0 fades out mathematically rather than
        being turned away by a gate. Returns whether the observation actually
        entered the cloud.
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
        """Admission to the template set, by a rule with no tunable parameters.

        While the set has room, admit. Once it is full, the new template competes
        on quality with its nearest neighbour and the weaker of the two is evicted.

        This squeezes out redundancy from the same viewpoint by itself, while
        diversity across poses and across sessions survives — no pose threshold
        needed.
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

    # ── Scoring and coarse recall ────────────────────────────────────────
    def score_observation(self, character_id: str, modality: str,
                          embedding: Any, q: float) -> float | None:
        """How one observation ranks one profile: q*(s-s0), where s = max(cos mean, max cos templates).

        Returns None when that profile has no cloud for that modality; the caller
        treats an absent modality as contributing 0.
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
        """Fuse at the score level: sum over every observation of every modality.

        An absent modality contributes 0, which makes this robust to a missing
        modality without any special case.
        """
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
        """The top-K ranking. hit@K is the only acceptance metric here, because if
        the ranking is wrong arbitration can still rescue it by answering NEW.
        """
        scored = [(cid, self.score_evidence(cid, evidence)) for cid in character_ids]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, int(k))]
