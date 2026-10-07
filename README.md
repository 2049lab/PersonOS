<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/brand/lockup-tagline-dark.svg">
  <img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/brand/lockup-tagline.svg" alt="PersonOS — Memory that knows who." width="420">
</picture>

<br>

**Multimodal long-term memory for robots, smart glasses and agents.**<br>
Video, images and conversations in — memory about *people*, with names, out.

<br>

[![PyPI](https://img.shields.io/pypi/v/personos?color=6C4CF1&label=pypi)](https://pypi.org/project/personos/)
[![Python](https://img.shields.io/pypi/pyversions/personos?color=3B82F6)](https://pypi.org/project/personos/)
[![License](https://img.shields.io/badge/license-Apache_2.0-10B981)](LICENSE)
[![M3-Bench-robot](https://img.shields.io/badge/M3--Bench--robot-61.5%25-F43F5E)](#benchmarks)
[![Video-MME long](https://img.shields.io/badge/Video--MME_long-87.0%25-F59E0B)](#benchmarks)

English · [简体中文](https://github.com/2049lab/personos/blob/main/README.zh-CN.md)

[The film](#the-film) · [Why identity](#why-identity) · [Highlights](#highlights) · [Benchmarks](#benchmarks) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Install](#installation) · [Roadmap](#roadmap)

</div>

<br>

## The film

https://github.com/user-attachments/assets/7252ab1e-9fe8-4c83-b2b1-d3da03b83504

**Who ate the Halloween candy?** Pebble, a little home robot, watches a party
through its one big lens. Three kids arrive as strangers, earn their names from
what they call each other, vanish under identical white sheets — and in the
middle of a blackout, one of them puts on someone else's sheet to frame him.

The notebook on the right is Pebble's memory, and it works the way PersonOS
works: the **screenplay** is written the moment things happen, with neutral
codes (`P1`, `P2`…); **who's who** is kept separately, revised as evidence comes
in, and only applied to the screenplay when the night's memory is saved. Next
morning everyone blames the boy whose sheet it was. Pebble doesn't.

<table>
  <tr>
    <td width="33%"><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-1-names.jpg" alt="Names learned from dialogue"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-2-ghosts.jpg" alt="Ghosts keep their names"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-3-conflict.jpg" alt="Leo in two places?"></td>
  </tr>
  <tr>
    <td><b>Names are earned.</b> "Whoa… Mia, the candy's over there!" turns stranger <code>P1</code> into a guess, <i>Mia?</i> — confirmed only when it happens again.</td>
    <td><b>No faces, still known.</b> Under the sheets, shoes, height, a bell on a shoelace and voices keep every ghost attached to the right kid.</td>
    <td><b>Contradictions are caught.</b> The ketchup sheet says <i>Leo</i>; Leo's voice comes from the kitchen. One person can't be in two places.</td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-4-fixed.jpg" alt="Re-identified as Mia"></td>
    <td><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-5-saved.jpg" alt="Screenplay rewritten with names"></td>
    <td><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-6-recall.jpg" alt="Recall with evidence"></td>
  </tr>
  <tr>
    <td><b>…and repaired.</b> A high giggle and pink sneakers move the ghost to Mia's card. What was recorded is never edited — only who it refers to.</td>
    <td><b>Identity lands last.</b> When the night is saved, codes resolve to names across the whole screenplay: one complete record, every person correct.</td>
    <td><b>Answers cite evidence.</b> "Who was it?" — Mia, with the exact lines and a snapshot of the sneakers that gave her away.</td>
  </tr>
</table>

<sub>An original, code-drawn short (p5.js + p5.brush); voices and sound by ElevenLabs. The story is staged, the mechanics are the real ones — see <a href="#how-it-works">How it works</a>.</sub>

## Why identity

A memory system that understands video but not *people* writes this down:

```diff
- [21:15] A ghost in a ketchup-stained sheet went to the candy bowl.
- [21:16] A ghost took a candy.  [21:22] A ghost took a candy.  [21:29] A ghost took a candy.
```

Which ghost? Ask *"who ate the candy?"* and the best it can do is follow the
sheet — straight to the wrong kid. PersonOS writes this instead:

```diff
+ [21:15] Mia — wearing Leo's ketchup-stained sheet — went to the candy bowl.
+ [21:16] Mia took a candy.  [21:22] Mia took a candy.  [21:29] Mia took a candy.
```

Every line points at a stable **character** that persists across clips and
sessions. Names are **earned, not enrolled** — someone becomes "Mia" because
people call her Mia; until then she is a stable handle you can still ask about.
And the device itself is a character too, because *"who did this?"* sometimes
has the answer *"you did"*.

## Highlights

<table>
  <tr>
    <td width="50%" valign="top">
      <h4>People, not pixels</h4>
      Faces, body shots and voiceprints accumulate into a per-character identity
      cloud. Someone seen in clip 12 is the same person three sessions later —
      no enrollment step, no face database to maintain.
    </td>
    <td width="50%" valign="top">
      <h4>It catches its own mistakes</h4>
      Multimodal models hallucinate identities. PersonOS checks every screenplay
      for physically impossible contradictions in code, asks the model to repair
      them, and degrades by subtraction — it would rather forget a name than
      learn a wrong one.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>Facts now, identity when it's sure</h4>
      What happened is recorded the moment it is seen; who it was is settled
      once the evidence is in. Correcting an identity changes who a record
      refers to — never what was observed.
    </td>
    <td valign="top">
      <h4>Memory that never overwrites the past</h4>
      An append-only record — evidence → episode → atom → chain. "15 fish → 13 →
      11" stays as linked history; the answer layer decides what is current.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>Answers show their work</h4>
      Every answer cites the episode, clip and timestamp it came from, with the
      full retrieval and adjudication trace one attribute away.
    </td>
    <td valign="top">
      <h4>Built for devices that live in the world</h4>
      Clips stream in asynchronously through an ordered per-session queue with
      backpressure; nothing blocks the device loop. Text, images and video land
      in one store.
    </td>
  </tr>
</table>

## Benchmarks

**[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent)** — long-video
memory from a robot's point of view: 100 videos, 1,276 questions. This is the
benchmark identity is built for.

| | **Overall** | Person understanding | Multi-hop | Multi-evidence | Cross-modal | General knowledge |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PersonOS** | **61.5** | **73.5** | **63.5** | **62.8** | **59.0** | **48.0** |
| M3-Agent, same models¹ | 37.4 | 50.9 | 42.4 | 37.1 | 37.0 | 28.4 |
| M3-Agent, as published | 30.7 | 43.3 | 29.4 | 32.8 | 31.2 | 19.1 |

**[Video-MME](https://github.com/BradyFU/Video-MME)**, long videos, no subtitles —
answered *from memory alone*: the video is ingested into PersonOS and the
answering model never sees it.

| **Overall** | Synopsis | Object recog. | Spatial reas. | Object reas. | Temporal reas. | Action recog. | Action reas. | Temporal perc. | Attribute perc. | OCR | Counting | Spatial perc. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **87.0** | 93.3 | 92.6 | 90.9 | 87.9 | 87.9 | 84.1 | 85.0 | 83.3 | 85.2 | 78.6 | 70.8 | 33.3 |

**Text memory, no compromise** — the same store on conversational benchmarks:

| Benchmark | Overall | Breakdown |
|---|:---:|---|
| [LoCoMo-10](https://github.com/snap-research/locomo) · 1,536 q, adversarial excluded | **83.3** | single-hop 89.3 · temporal 79.8 · multi-hop 77.3 · open-domain 58.7 |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) · 500 q, official judge prompts | **80.6** | knowledge-update 91.7 · single-session-assistant 98.2 · single-session-user 89.1 · multi-session 76.9 · temporal 76.4 · preference 43.3 · abstention 73.3 |

<sub>Accuracy in %. Video memory built with Qwen3.5-Omni-Plus, questions answered from memory by GPT-5.5. ¹ Same dataset, memory model and answer model as PersonOS; prompt templates and retrieval budgets are not guaranteed identical. Text benchmarks reproduce from <a href="scripts/bench/README.md"><code>scripts/bench</code></a>; video evaluation code ships with the accompanying paper.</sub>

## Quickstart

```bash
pip install personos
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible gateway
```

```python
from personos import Memory

m = Memory()   # SQLite under ~/.personos — nothing to provision

m.add("I moved from Hangzhou to Shanghai in June", user_id="alice", session_id="s1")
m.end_session(user_id="alice", session_id="s1", sync=True)

print(m.search("where do I live?", user_id="alice").ans.answer)
# -> Shanghai
```

**Give it eyes.** Same three calls, now with video and person identity:

```bash
pip install 'personos[identity]'
export PERSONOS_VIDEO_BACKEND=real
export PERSONOS_MLLM_API_KEY=sk-...  PERSONOS_MLLM_MODEL=...   # any model that accepts video
```

```python
for clip in ["clip000.mp4", "clip001.mp4", "clip002.mp4"]:          # consecutive clips
    m.add([{"role": "user", "content": "", "video": clip}],
          user_id="robot", session_id="living-room")

m.end_session(user_id="robot", session_id="living-room", sync=True)  # identities are committed here

out = m.search("What's the name of the girl at the table, and what does she study?", user_id="robot")
print(out.ans.answer)       # Alice ... math
print(out.ans.cited_cells)  # the episodes (and clips) the answer came from
```

Not sure what your setup can do? Run `personos doctor`. More in
[examples/](examples/README.md): a week of conversation · a photo · video with
person identity.

## How it works

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/architecture-light.svg">
  <img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/architecture.svg" alt="Perceive → Resolve identity → Remember → Recall" width="100%">
</picture>

**① Perceive.** A multimodal model watches each clip and writes a
**screenplay**: who is on screen — as clip-local codes, not names — who says
what, what happens, when.

**② Resolve identity.** Codes are matched against known characters through a
face / body / voiceprint cloud. Code-level checks look for physical
contradictions (one person in two places, two people sharing one identity) and
send them back to the model for repair. Within a session, **character chains**
collect evidence — a first sighting, a better face, a clearer voice, a name said
out loud — and are adjudicated into persistent characters when the session
closes.

**③ Remember.** The screenplay is resolved through those identities and
written into a layered, append-only store:

```
evidence  ──►  memcell (episode)  ──►  atom  ──►  atom_chain
raw input      the narrative unit      the          the same fact over time,
never edited   used for answering      retrieval    grouped, never collapsed
                                       anchor
```

Atoms are short propositions that embed well — the retrieval index. Episodes
are what the model actually reads.

**④ Recall.** `search()` rewrites the question, resolves *who* it is about,
retrieves, reranks, adjudicates a draft and escalates to a multi-step agent
when needed — and returns how it got there:

```python
out = m.search("who ate the candy?", user_id="pebble")
out.ans.answer   # the answer
out.rw.subject   # who the question was resolved to be about
out.hits         # atoms retrieved, in fusion order
out.reviews      # what the adjudicator said about the draft
out.to_public()  # ...or a plain dict
```

## Built for

| | |
|---|---|
| **Home & service robots** | Remember household members, guests and what each of them asked for — across days, without enrollment. |
| **Smart glasses & wearables** | *"Who was the person I met at the booth on Tuesday, and what did we talk about?"* |
| **Desktop & companion agents** | One memory across chat, screenshots and calls, every answer traceable to its source. |

## Installation

The core has 7 dependencies — no torch, no LangChain, no web framework.
Everything heavier is opt-in:

| Install | Unlocks |
|---|---|
| `pip install personos` | Text memory: layered write path, fast + deep recall, profiles |
| `personos[deep]` | Multi-step deep-recall agent (recommended) |
| `personos[image]` | Image ingest and recall over photos |
| `personos[identity]` | Video with face / body / voiceprint identity (~2 GB) |
| `personos[mysql]` `personos[redis]` | Multi-process deployments |
| `personos[oss]` | Object storage instead of local media files |
| `personos[all]` | Everything |

<details>
<summary><b>Model weights</b></summary>

<br>

Optional capabilities download their own weights on first use:

| Capability | Weights | Where they land |
|---|---|---|
| Face recognition | InsightFace `buffalo_l` (~300 MB) | `~/.insightface` (auto-download) |
| Voiceprints | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace cache; override with `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` |
| Multimodal understanding | none — remote API | `PERSONOS_MLLM_*` points at any MLLM endpoint |

Offline or behind a firewall: set `HF_ENDPOINT=https://hf-mirror.com`, or
pre-download and point the `*_DIR` variables at local copies.

</details>

<details>
<summary><b>Configuration</b></summary>

<br>

Everything except the model endpoint is optional. Unset means a capability is
off or degraded — never that the text path breaks.

| | Default | Unset means |
|---|---|---|
| **Chat + embeddings** | — | **required** |
| Database | SQLite at `~/.personos` | set `PERSONOS_DB_URL` for MySQL — needed only for several workers |
| Multimodal model | off | images are stored but not understood; video is refused with instructions |
| Media storage | local files | set `PERSONOS_MEDIA_BACKEND=oss` for object storage |
| Reranker | off | retrieval keeps its fusion order |
| Deep recall | off | `pip install personos[deep]` |
| Redis | off | single process; session state in memory |
| Tracing | off | every tracing call is a no-op |

The model service fetches video clips **by URL**, so with local media storage
set `PERSONOS_MEDIA_BASE_URL` to a publicly reachable prefix (or use OSS).
See [.env.example](.env.example) for every option.

Ask for something unconfigured and you get a sentence, not silence:

```
MissingCapability: video understanding is unavailable: no multimodal model is configured
  To enable it: set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL
```

**Cannot do it at all → raise; did it partially → return and say so** in
`result.warnings`. A missing optional capability never fails a write.

</details>

<details>
<summary><b>API</b></summary>

<br>

```python
m.add(messages, user_id=..., session_id=...)   # text, a dict with an image, or video clips
m.end_session(user_id=..., session_id=...)     # close the segment; build memories and commit identities
m.search(query, user_id=..., mode="auto")      # "auto" | "fast" | "deep"
m.profile(user_id=...)                         # distilled user profile
m.trace(node_id, user_id=...)                  # provenance, both directions
m.capabilities()                               # what this configuration can do
m.reset(user_id=...)                           # delete one user's data
```

Writes are **asynchronous by default**: `add()` / `end_session()` enqueue onto
a per-session ordered queue (FIFO per session, fair across sessions,
backpressure when overloaded) and return an `AddReceipt` at once. Pass
`sync=True`, or call `m.flush(...)`, when you need to read your own writes. A
full queue raises `QueueBusy` — the HTTP API's `503 + Retry-After`.

Reads (`search`, `profile`, `trace`) are synchronous; from async code use
`await asyncio.to_thread(m.search, q)`.

</details>

<details>
<summary><b>Running it as a service</b></summary>

<br>

[`server/`](server/README.md) is a FastAPI deployment — ordered ingestion across
processes, backpressure, token-scoped multi-tenancy. It is not part of the pip
package; if you are embedding memory in an application, you do not need it.

</details>

## Roadmap

- [x] Layered append-only memory with fast + deep recall
- [x] Image ingest and recall
- [x] Video memory with face / body / voiceprint identity, self-repair and cross-session recognition
- [x] Async ordered ingestion, HTTP server
- [ ] Memory inspector UI — timeline, character gallery, answer traces
- [ ] MCP server for Claude Code, Cursor and other MCP clients
- [ ] Integrations: LangGraph, OpenAI Agents SDK, ROS 2
- [ ] `AsyncMemory`
- [ ] Streaming ingestion for live camera feeds
- [ ] Lighter identity install (faces without torch)
- [ ] Paper + full video evaluation code

## Status

`0.1.x` — the memory pipeline runs in production; the packaging around it is
new, and the public API may still move before `1.0`. Releases follow
[RELEASE.md](RELEASE.md).

## Contributing

Issues and PRs are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The test
suite runs with `pytest` and needs no external services.

## Acknowledgements

- The film is original work, drawn in code with [p5.js](https://p5js.org) and
  [p5.brush](https://github.com/acamposuribe/p5.brush); voices and sound effects
  by [ElevenLabs](https://elevenlabs.io).
- The M3-Bench-robot benchmark and the video samples used in the examples come
  from [M3-Agent / M3-Bench](https://github.com/bytedance-seed/m3-agent)
  (ByteDance-Seed, CC BY-NC-SA 4.0). Samples are downloaded on demand and are
  not redistributed in this repository.

## License

Apache-2.0. See [LICENSE](LICENSE).
