"""Identity layer: character entities + multimodal assets + a probability identity cloud.

- store.CharacterStore: per-user access to the three tables characters /
  character_assets / character_cloud.
- cloud.CloudEngine: the probability identity cloud. It ranks a coarse recall set
  and provides the enrollment basis; it does not decide who someone is — that
  verdict comes from multimodal-LLM arbitration.
- types: evidence value types (FacePick/VoiceSample/CastEvidence) + CandidateCard
  + normalized.
- draft.DraftStore: session state for identity chains, roster, evaluations and
  staged assets (Redis only; sqlite is not allowed here).
- chains.ChainBook: the chain evidence ledger, refresh triggers, and re-arbitration
  or degrade on same-frame collisions.
- registry.AnchorRegistry: cast mapping, candidate construction, roster updates.
- inspect: present_casts + same-frame collision detection (Rule 6).
- recognize: the batched BIND arbitration protocol (build/parse) + enroll_evidence.
- commit.commit_session: end-of-session two-phase final adjudication, then
  settlement into MySQL (ownership mapping + wearer).
"""
