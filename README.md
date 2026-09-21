# PersonOS

Layered long-term memory for agents — and the only one that watches video and
remembers *who* was in it.

> **Status: pre-release (0.1.0.dev0).** This repository is being reshaped from an
> internal service into a standalone library. The public API below is the target,
> not yet the shipped surface. See [the roadmap](#roadmap) for what still moves.

## Why another memory library

Most memory libraries store a flat list of facts and search it. PersonOS keeps a
**layered, append-only record** and never resolves contradictions at write time:

```
evidence  ──►  memcell (episode)  ──►  atom  ──►  atom_chain
raw turns      the narrative unit     the          the same fact
never edited   used for answering     retrieval    over time, grouped
                                      anchor       but not collapsed
```

Two consequences that matter in practice:

- **Contradictions survive.** "I have 15 fish" → "actually 13" → "actually 11" stay
  as three linked atoms with their timestamps. The answer layer decides what is
  current; the memory layer never silently overwrites the past.
- **Answers cite episodes, not fragments.** Atoms are the retrieval index; the
  narrative episode is what the model reads. Retrieval granularity and answering
  granularity are deliberately different.

### Multimodal: video and person identity

Every other open memory framework is text-only, or converts an image to a caption
at ingest. PersonOS takes **video clips** and builds stable *character* entities
from faces, body shots and voiceprints, so a person recognised in clip 12 is the
same person recognised three sessions later — without anyone enrolling them first.

This is optional (`pip install personos[identity]`) and adds roughly 2 GB of model
dependencies. The text memory core does not import any of it.

## Install

```bash
pip install personos
```

## Quickstart

```python
from personos import Memory

m = Memory()                      # zero config: SQLite under ~/.personos, one env var

m.add([{"role": "user", "content": "I moved from Hangzhou to Shanghai in June"}],
      user_id="alice", session_id="chat-1")
m.end_session(user_id="alice", session_id="chat-1")

out = m.search("where do I live now?", user_id="alice")
print(out.ans.answer)
```

The only required configuration is a chat/embedding endpoint:

```bash
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible gateway
```

Everything else — MySQL, Redis, object storage, reranker, vision, video identity,
tracing — is optional. Leave it unset and the corresponding capability degrades or
is reported as unavailable; it never crashes the text path.

## Configuration tiers

| | Unset means |
|---|---|
| **Chat LLM + embedder** | **required** — nothing works without them |
| Database | SQLite at `~/.personos/personos.db`, tables auto-created. Set `PERSONOS_DB_URL` for MySQL. |
| Multimodal LLM | Text memory is unaffected. Images are stored but not understood; video is rejected with a clear error. |
| Object storage | Files land on the local filesystem instead. |
| Reranker | Retrieval keeps its fusion order. |
| Redis | Single process instead of multi-worker. |

## Roadmap

This repo is mid-migration from an internal deployment. Landed / remaining:

- [x] Behavioural baseline harness (record & replay, so the refactor is provably shape-only)
- [ ] Config rewrite — no import-time evaluation, no internal secret manager
- [ ] SQLite backend + `Database` protocol
- [ ] OpenAI-compatible providers replacing the internal model gateway
- [ ] Local media store
- [ ] Capability contract — actionable errors for unconfigured features
- [ ] `Memory` facade; HTTP server moves to `server/`
- [ ] Dependency slimming and extras
- [ ] English docstrings throughout
- [ ] Docs and first release

## License

Apache-2.0. See [LICENSE](LICENSE).
