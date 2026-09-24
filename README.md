<div align="center">

# PersonOS

**Any in, memory out** — the multimodal long-term memory layer for agents,
robots, and smart hardware.

Whatever your agent perceives — conversations, images, live video — goes in;
layered, traceable memory comes out. It is also the only open memory framework
that watches video and remembers *who* was in it.

[![PyPI](https://img.shields.io/pypi/v/personos)](https://pypi.org/project/personos/)
[![Python](https://img.shields.io/pypi/pyversions/personos)](https://pypi.org/project/personos/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

English · [简体中文](README.zh-CN.md)

[Quickstart](#quickstart) · [Benchmarks](#benchmarks) · [Design](#why-personos) · [Installation](#installation) · [Configuration](#configuration) · [API](#api) · [Examples](examples/README.md)

</div>

## Quickstart

```bash
pip install personos
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible gateway
```

```python
from personos import Memory

m = Memory()                      # SQLite under ~/.personos; the two env vars above are all it needs

m.add("I moved from Hangzhou to Shanghai in June", user_id="alice", session_id="s1")
m.end_session(user_id="alice", session_id="s1", sync=True)   # wait for the queue to drain

print(m.search("where do I live?", user_id="alice").ans.answer)
```

Not sure what your configuration can do? `personos doctor` reads it and tells
you what works, what is off, and what to set.

## Benchmarks

**[LoCoMo-10](https://github.com/snap-research/locomo)** — long-term
conversational memory (10 conversations, ~300 turns each, 1,536 answerable
questions; adversarial questions excluded per community convention):

| Multi-hop | Temporal | Open-domain | Single-hop | **Overall** |
|---|---|---|---|---|
| 77.3% | 79.8% | 58.7% | 89.3% | **83.3%** |

**[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent)** — long-video
memory from a robot's perspective (100 videos, 1,276 QA), against the
benchmark's reference agent:

| | Overall | Cross-Modal Reasoning | General Knowledge | Multi-Hop | Multi-evidence | Person Understanding |
|---|---|---|---|---|---|---|
| **PersonOS** | **61.5%** | 59.0% | 48.0% | 63.5% | 62.8% | 73.5% |
| m3-agent (baseline) | 37.4% | 37.0% | 28.4% | 42.4% | 37.1% | 50.9% |

**[Video-MME](https://github.com/BradyFU/Video-MME)** (long split, without
subtitles) — general video understanding:

| Overall | Info. Synopsis | Object Recog. | Spatial Reason. | Object Reason. | Temporal Reason. | Action Recog. | Action Reason. | Temporal Percep. | Attribute Percep. | OCR | Counting | Spatial Percep. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **87.0%** | 93.3% | 92.6% | 90.9% | 87.9% | 87.9% | 84.1% | 85.0% | 83.3% | 85.2% | 78.6% | 70.8% | 33.3% |

The LoCoMo run is fully reproducible: dataset download, evaluation script,
scoring protocol and judge configuration all live in
[scripts/bench](scripts/bench/README.md).

## Why PersonOS

Agents are leaving the chat box. A robot arm, a pair of smart glasses, a
desktop copilot — they perceive continuously, in more modalities than text,
and they need memory that keeps up. Most frameworks store a flat list of facts
and search it. PersonOS keeps a **layered, append-only record** and refuses to
resolve contradictions at write time:

```
evidence  ──►  memcell (episode)  ──►  atom  ──►  atom_chain
raw turns      the narrative unit      the          the same fact over time,
never edited   used for answering      retrieval    grouped, never collapsed
                                       anchor
```

Two consequences that show up in practice:

**Contradictions survive.** "15 fish" → "actually 13" → "actually 11" stay as
three linked atoms with their timestamps. The answer layer decides what is
current; the memory layer never silently overwrites the past. Ask *"how many
fish do I have"* and you get the current count; ask *"did that change"* and
the history is still there.

**Answers cite episodes, not fragments.** Atoms are the retrieval index —
short, self-contained propositions that embed well. The narrative episode is
what the model actually reads. Retrieval granularity and answering granularity
are deliberately different, because what makes a good search key makes a poor
answer.

### Recall shows its work

`search()` returns the answer *and* how it got there: the rewritten query, the
atoms retrieved, the materials ranked, the adjudication verdicts, whether it
escalated to the deep agent. When an answer is wrong, you can see which stage
went wrong instead of guessing.

```python
out = m.search("how many fish?", user_id="alice")
out.ans.answer        # the answer
out.rw.subject        # who the question was resolved to be about
out.hits              # atoms retrieved, in fusion order
out.reviews           # what the adjudicator said about the draft
out.to_public()       # ...or a plain dict, if you just want the answer
```

### Video and person identity

Every other open memory framework is text-only, or reduces an image to a
caption at ingest. PersonOS takes **video clips** — the stream a robot or a
pair of glasses actually lives in — and builds stable *character* entities
from faces, body shots and voiceprints: someone recognised in clip 12 is the
same person three sessions later, without anyone enrolling them first.

Optional (`pip install personos[identity]`, ~2 GB of model dependencies). The
text core imports none of it.

## Installation

The core is deliberately small — 8 dependencies, no torch, no langchain, no
web framework. Everything heavier is an extra you opt into:

| Install | Unlocks |
|---|---|
| `pip install personos` | Text memory: layered write path, fast + deep recall, profiles |
| `personos[deep]` | The multi-step deep-recall agent (recommended) |
| `personos[image]` | Image ingest + recall over photos |
| `personos[identity]` | Video: face / body / voiceprint character identity (~2 GB) |
| `personos[mysql]` `personos[redis]` | Multi-process deployments |
| `personos[oss]` | Object storage instead of local media files |
| `personos[all]` | Everything |

### Model weights

Optional capabilities download their own weights on first use — no manual
setup:

| Capability | Weights | Where they land |
|---|---|---|
| Face recognition | InsightFace `buffalo_l` (~300 MB) | `~/.insightface` (auto-download) |
| Voiceprints | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace cache; override with `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` |
| Multimodal understanding | none — remote API | `PERSONOS_MLLM_*` points at any MLLM endpoint |

Behind the GFW or on an offline machine: set `HF_ENDPOINT=https://hf-mirror.com`
for HuggingFace, or pre-download and point the `*_DIR` variables at local
copies.

## Configuration

Everything except the model endpoint is optional. Unset means a capability is
off or degraded — never that the text path breaks.

| | Default | Unset means |
|---|---|---|
| **Chat + embeddings** | — | **required** |
| Database | SQLite at `~/.personos` | set `PERSONOS_DB_URL` for MySQL — needed only for several workers |
| Multimodal model | off | images are stored but contribute nothing to retrieval; video is refused with instructions |
| Media storage | local files | set `PERSONOS_MEDIA_BACKEND=oss` for object storage |
| Reranker | off | retrieval keeps its fusion order |
| Deep recall | off | `pip install personos[deep]` for the multi-step agent |
| Redis | off | single process; session state is in memory |
| Tracing | off | every tracing call is a no-op |

See [.env.example](.env.example) for the full list with explanations.

### Errors tell you what to do

Ask for something unconfigured and you get a sentence, not silence:

```
MissingCapability: video understanding is unavailable: no multimodal model is configured
  To enable it: set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL
```

The rule: **cannot do it at all → raise; did it partially → return and say
so** in `result.warnings`. A missing optional capability never fails a write.

## API

```python
m.add(messages, user_id=..., session_id=...)   # text, a dict with an image, or video clips
m.end_session(user_id=..., session_id=...)     # close the segment and build memories
m.search(query, user_id=..., mode="auto")      # "auto" | "fast" | "deep"
m.profile(user_id=...)                         # distilled user profile
m.trace(node_id, user_id=...)                  # provenance, both directions
m.capabilities()                               # what this configuration can do
m.reset(user_id=...)                           # delete one user's data
```

**Writes are asynchronous by default.** `add()`/`end_session()` enqueue onto a
per-session ordered queue — the same machinery the server deployment uses:
FIFO per session, fair scheduling across sessions, backpressure when a session
is overloaded — and return an `AddReceipt` immediately, so memory writes never
block your application's own work:

```python
receipt = m.add(..., user_id=..., session_id=...)   # returns at once
# ... your code keeps running; a background dispatcher builds the memories ...

m.flush(user_id=..., session_id=...)                # or: wait until the queue drains
m.end_session(..., sync=True)                       # or: close and wait in one call
```

Pass `sync=True` to `add()`/`end_session()`, or call `flush()`, whenever you
need to read your own writes. `queue_status()` reports a session's depth and
cursor. A full queue raises `QueueBusy` — the same contract the HTTP API
expresses as `503 + Retry-After`.

Reads (`search`, `profile`, `trace`) are synchronous. There is no `AsyncMemory`
yet; rather than pretend, the honest workaround is
`await asyncio.to_thread(m.search, q)`.

## Examples

One story in three chapters:
[a week of conversation](examples/quickstart.py) (watch the profile build
itself between days) · [with a photo](examples/images.py) ·
[with video and person identity](examples/video.py). See
[examples/README.md](examples/README.md) for what each needs and real output.

## Running it as a service

[`server/`](server/README.md) is a FastAPI deployment — ordered ingestion
across processes, backpressure, token-scoped multi-tenancy. It is **not** part
of the pip package; it has its own dependencies and lifecycle.

If you are embedding memory in an application, you do not need it.

## Status

`0.1.0` — the memory pipeline has been running in a production deployment; the
packaging around it is new, and the public API may still move before `1.0`.
Releases follow [RELEASE.md](RELEASE.md).

## Contributing

Issues and PRs are welcome at
[github.com/2049lab/personos](https://github.com/2049lab/personos). The test
suite runs with `pytest` and needs no external services.

## License

Apache-2.0. See [LICENSE](LICENSE).
