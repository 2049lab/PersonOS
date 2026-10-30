# Changelog

## Unreleased

This update is available in the source tree. It has not been published to PyPI.

### Added

- `Memory.search(mode="flash")` and HTTP recall support for source material
  without generated answers, query rewriting or answer review. Embeddings and
  the configured reranker still run.
- Model-free, user-scoped `list_memories`, `list_chains` and `get_chain`, with
  corresponding HTTP routes, bounded pages, source evidence and chain cursors.
- Bilingual framework documentation, a shorter GitHub homepage and a signing
  guide. Historical documentation URLs now lead to the current OpenAPI UI.

### Changed

- Text extraction focuses on the user. It may retain evidence without creating
  an episode, and an episode may have no retrieval atoms. Empty extraction no
  longer triggers another extraction or a fabricated summary atom.
- Failed episode generation retains the evidence instead of promoting the raw
  transcript to an episode. Deep-recall write-back follows the personal-memory
  scope and preserves fact, claim and event types.
- New text evidence is distinguished by message identity, so separate messages
  with identical wording retain their own provenance. Retrying the same queued
  message reuses its evidence IDs; derived writes are not an exactly-once
  transaction. Existing content-deduplicated records are unchanged.
- Recall now validates a nonempty query and an integer `top_k` from 1 to 100.
- Historical evaluation results are explicitly separated from this revision;
  no new benchmark or real-device accuracy result is claimed.

See the [framework notes](docs/framework.md#compatibility-notes) for migration
considerations and the capabilities deferred from this update.
