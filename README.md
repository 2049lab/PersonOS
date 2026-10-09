<p align="center">
  <a href="https://personos-ai.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/wordmark-dark.svg">
      <img src="docs/assets/brand/wordmark.svg" alt="PersonOS" width="320">
    </picture>
  </a>
</p>

<p align="center"><strong>Long-term multimodal memory for personal agents</strong></p>

<p align="center">
  <a href="https://personos-ai.com/">Website</a> ·
  <a href="docs/usage.md">Documentation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#case-study">Case study</a> ·
  <a href="docs/evaluation.md">Evaluation</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/personos/"><img src="https://img.shields.io/pypi/v/personos?style=flat-square&color=526479" alt="PyPI version"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-526479?style=flat-square" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-526479?style=flat-square" alt="Apache 2.0 license"></a>
</p>

PersonOS is an open-source memory framework for personal agents. It helps agents retain experiences from conversations, images, and video, keep a record of changing facts and preferences, and retrieve relevant memories with supporting evidence.

Developed at [2049lab](https://github.com/2049lab), it is part of the [PersonOS initiative](https://personos-ai.com/) toward human-centered, lifelong multimodal memory. This repository provides the Python memory API and HTTP service. The website covers the broader [research roadmap](https://personos-ai.com/research.html) and [ecosystem](https://personos-ai.com/ecosystem.html).

## Capabilities

- **Multimodal ingestion.** Organize conversations, images, and video clips into episodic records linked to source evidence.
- **Persistent, layered memory.** Keep source evidence, episodes and retrieval anchors in separate layers. Store data locally with SQLite or configure shared storage for service deployments.
- **Fact history and user profiles.** Append new observations, link related statements, and consolidate user profiles as evidence accumulates. Earlier records remain available for retrieval and inspection.
- **Evidence-backed retrieval.** Retrieve relevant episodes and return answers with citations and review information. Optional deep recall supports multi-step retrieval.
- **Person association in video.** With the video identity backend enabled, accumulate face, voice and contextual evidence, revise identity hypotheses, and link events to persistent characters at session close.

## Quick start

### 1. Install and configure text memory

Requires **Python 3.10+** and access to both chat completions and embeddings. The following uses the package's OpenAI defaults; the key must have access to both models.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install personos

export PERSONOS_LLM_API_KEY="your-api-key"
export PERSONOS_LLM_BASE_URL="https://api.openai.com/v1"
export PERSONOS_LLM_MODEL="gpt-4o-mini"
export PERSONOS_EMBEDDING_MODEL="text-embedding-3-small"
export PERSONOS_EMBEDDING_DIM=1536

personos doctor
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` and set the same variables in PowerShell or a `.env` file. `personos doctor` checks configuration and installed dependencies; it does not test API credentials, model availability or network access.

Using another provider? Set its chat model and embedding model explicitly. The embedding endpoint and key inherit `PERSONOS_LLM_BASE_URL` and `PERSONOS_LLM_API_KEY`; override `PERSONOS_EMBEDDING_BASE_URL` and `PERSONOS_EMBEDDING_API_KEY` if they differ. `PERSONOS_EMBEDDING_DIM` must match the vectors your model returns. Chat API compatibility does not imply embedding support.

### 2. Write a memory and ask about it

```python
from personos import Memory

with Memory() as memory:
    memory.add(
        "I moved from Hangzhou to Shanghai in June.",
        user_id="quickstart-alice",
        session_id="first-conversation",
    )
    memory.end_session(
        user_id="quickstart-alice",
        session_id="first-conversation",
        sync=True,
    )

    result = memory.search("Where do I live?", user_id="quickstart-alice")
    print(result.ans.answer if result.ans else "No answer returned.")
    print(result.ans.cited_cells if result.ans else [])
```

The expected answer refers to Shanghai; wording and retrieval quality depend on your models. `add()` queues work. `end_session(sync=True)` closes the episode and waits for ingestion before the read. The example keeps its memory for later questions.

<details>
<summary><b>Optional: video memory and person identity</b></summary>

Keep the text configuration above. Install the identity dependencies and configure a multimodal endpoint that supports the backend's `video_url`, `image_url` and `input_audio` content formats. A text-only or image-only chat endpoint is insufficient.

```bash
python -m pip install 'personos[identity]'
export PERSONOS_VIDEO_BACKEND=real
export PERSONOS_MLLM_API_KEY="your-multimodal-api-key"
export PERSONOS_MLLM_BASE_URL="https://your-provider.example/v1"
export PERSONOS_MLLM_MODEL="your-video-model"
export PERSONOS_MEDIA_BASE_URL="https://your-media-host.example/media"
```

Replace the example URLs and model name with your service settings. Serve the contents of `~/.personos/media` at the media URL so the model service can fetch clips, or configure [OSS storage](.env.example). Setting the URL does not start a media server. Identity dependencies and model weights require additional downloads; see the installation details below.

```python
from personos import Memory

clips = ["clip000.mp4", "clip001.mp4", "clip002.mp4"]  # your consecutive clips

with Memory() as memory:
    for clip in clips:
        memory.add(
            [{"role": "user", "content": "", "video": clip}],
            user_id="robot",
            session_id="living-room",
        )

    memory.end_session(
        user_id="robot", session_id="living-room",
        sync=True, timeout_s=600 * len(clips) + 600,
    )
    result = memory.search("Who appeared, and what did they do?", user_id="robot")
    print(result.ans.answer if result.ans else "No answer returned.")
    print(result.ans.cited_cells if result.ans else [])
```

Video is processed in clip order, with identities committed at session close. Allow minutes per clip depending on the model and recording. See [the video example](examples/README.md#3-video-and-person-identity) for sample preparation, hosting and data-source terms.

</details>

<details>
<summary><b>Installation options</b></summary>

Install only the extras you need. Use `python -m pip install 'personos[EXTRA]'`; combine extras with commas, for example `'personos[deep,image]'`.

| Install | What it adds |
|---|---|
| `personos` | Layered text writes, fast recall, user profiles and local SQLite |
| `personos[deep]` | Multi-step deep-recall agent (recommended) |
| `personos[image]` | Image ingest and recall over photos |
| `personos[identity]` | Video with face / build / voiceprint identity (~2 GB) |
| `personos[mysql]` `personos[redis]` | Multi-process deployments |
| `personos[oss]` | Object storage instead of local media files |
| `personos[anthropic]` | Anthropic-protocol chat; embeddings need their own endpoint |
| `personos[observability]` | Optional Langfuse tracing |
| `personos[all]` | All optional capabilities |

</details>

<details>
<summary><b>Model weights</b></summary>

<br>

(Optional) models required for person identification:

| Capability | Weights | Location |
|---|---|---|
| Face recognition | InsightFace `buffalo_l` (~300 MB) | `~/.insightface` (auto-download) |
| Voiceprints | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace cache; override with `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` |
| Multimodal understanding | none, remote API | `PERSONOS_MLLM_*` points at a compatible endpoint |

Model weights and external datasets have their own terms, separate from the library's license.

</details>

<details>
<summary><b>Configuration</b></summary>

<br>

Chat and embeddings are required. Optional capabilities need the corresponding dependencies and configuration.

| | Default | If unset |
|---|---|---|
| **Chat + embeddings** | — | **required** |
| Database | SQLite at `~/.personos` | set `PERSONOS_DB_URL` for MySQL (only needed with several workers) |
| Multimodal model | off | images are stored but not understood; video is refused with instructions |
| Media storage | local files | set `PERSONOS_MEDIA_BACKEND=oss` for object storage |
| Reranker | off | retrieval keeps its fusion order |
| Deep recall | off | install `personos[deep]`; `auto` can then escalate to the multi-step agent |
| Redis | off | single process; session state in memory |
| Tracing | off | every tracing call is a no-op |

The model service fetches video clips by URL, so with local media storage set `PERSONOS_MEDIA_BASE_URL`
to a publicly reachable prefix (or use OSS). Primary options are listed in [.env.example](.env.example). For multiple workers, configure both shared MySQL and Redis.

The library reads `.env` from the working directory or its parents; already exported variables take precedence. Use `PERSONOS_ENV_FILE` for a specific configuration file and `PERSONOS_DATA_DIR` for a separate data directory.

Unavailable required capabilities raise `MissingCapability`; some optional paths, such as image understanding, emit a warning and retain the source. Diagnostics include setup guidance:

```
MissingCapability: video understanding is unavailable: no multimodal model is configured
  To enable it: set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL
```

</details>

<details>
<summary><b>Core API</b></summary>

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

Writes are asynchronous by default. `add()` and `end_session()` enqueue onto a per-session ordered queue (FIFO per session,
fair across sessions, backpressure when overloaded) and return an `AddReceipt` immediately. For a synchronous write,
pass `sync=True` or call `m.flush(...)`. When the queue is full, `QueueBusy` is raised; the HTTP API reports it as
`503 + Retry-After`.

Close the session to finalize its trailing episode. Profile consolidation can continue separately in the background. Reads (`search`, `profile`, `trace`) are synchronous. Requesting `mode="deep"` without the extra raises a capability error.

The recall result also includes the answer, resolved person, evidence and review:

```python
out = m.search("who ate the candy?", user_id="pebble")
out.ans.answer   # the answer, if out.ans is present
out.rw.subject   # who the question was resolved to be about
out.hits         # atoms retrieved, in fusion order
out.reviews      # the adjudicator's verdict on the draft
out.to_public()  # the same, as a plain dict
```

Model-generated answers and identity matches can be wrong. Inspect citations and original source records when checking an answer.

</details>

<details>
<summary><b>Running as a service</b></summary>

<br>

[`server/`](server/README.md) exposes PersonOS long-term memory as a network service.

</details>

## Architecture

PersonOS separates source evidence, episodic records, retrieval anchors and user profiles. New inputs extend the memory store; later queries retrieve relevant material and return answers linked to that evidence.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture.svg">
  <img src="docs/assets/architecture-light.svg" alt="Perceive → Remember → Recall; person association is a video-specific path" width="100%">
</picture>

1. **Perceive.** Process text, images and video into records of conversations, events and observations. Optional media backends provide image understanding and video processing.
2. **Remember.** Preserve source evidence, form episodes, extract atomic retrieval anchors, link related statements and consolidate user profiles.
3. **Recall.** Retrieve and rank relevant material, assemble an answer, and return citations and review information. Optional deep recall can take multiple steps through the store.

```text
evidence ──────► memcell ─────────► atom ─────────────► atom_chain
source input     episode            retrieval anchor    related statements
```

Atoms serve as retrieval anchors for episodic records. Atom chains group related statements in append order; they do not by themselves decide which statement is currently true. User profiles are consolidated separately from the accumulated memory.

For video, **event records and identity hypotheses are maintained separately**. A clip can use temporary person codes while face, voice and contextual evidence accumulate. Matches can be revised before the session is committed; names are learned from observed evidence, and a person can remain unnamed. This extends the memory pipeline for video and is not required for text or image memory.

Storage defaults to SQLite and local media files under `~/.personos`. Model calls use the endpoints you configure; local storage alone does not make inference offline.

## Applications

| Application | Memory it needs | PersonOS provides |
|---|---|---|
| Personal or desktop agents | Preferences, past conversations and changing facts | Layered text memory, fact histories and user profiles |
| Wearable and camera assistants | Events seen across consecutive recordings | Video episodes linked to people and source clips |
| Home robot prototypes | Who said or did what in recorded interactions | Identity evidence accumulated across clips, revisable matches and persistent characters |

## Examples

The repository includes executable text, image, and video examples. Outputs depend on the configured models; the scenarios below describe what each script demonstrates.

| Example | Scenario | What it demonstrates |
|---|---|---|
| [Text: a week of Alice's life](examples/quickstart.py) | The fish count changes from 15 to 13, then to 11 across conversations | Current facts, their history, and a user profile built over time |
| [Images: a whiteboard image](examples/images.py) | Store a locally generated whiteboard image alongside text and retain its original file | Image descriptions, text retrieval, and provenance; retaining the image with a warning when no vision model is configured |
| [Video: people and events across clips](examples/video.py) | In the M3-Bench sample, Bob enters with a basketball while other people may remain unnamed | Accumulating face, voice, and body evidence and resolving identities at session close |

See the [examples guide](examples/README.md) for commands, sample preparation, media hosting, and dataset terms.

## Case study

An animated walkthrough of identity resolution across video clips. Three children wear matching ghost costumes; new voice, location, and visual evidence helps resolve conflicting identity matches.

https://github.com/user-attachments/assets/2faea78b-7db6-4a7a-8f6d-22c7cd4a7a38

Events are recorded with temporary person codes such as `P1` and `P2`, while identity hypotheses are maintained separately. New evidence can revise those hypotheses before session consolidation links observations to persistent characters. Retrieval then returns the relevant events and supporting material.

This is an animated design case, not a live robot recording or an accuracy test. The [animation source and storyboard](docs/demo/README.md) are included; see the [video example](examples/video.py) for an executable ingestion workflow.

<details>
<summary><b>Full scenario, six-panel storyboard, and event records</b></summary>

### Who ate the Halloween candy?

Pebble, a small home robot, watches three children arrive for trick-or-treat. It initially knows none of them, learns candidate names from their conversations, and continues associating observations after they put on matching ghost costumes. During a blackout, Mia swaps into Leo's ketchup-stained sheet to make him appear responsible for taking the candy. Voices, locations, and a pair of pink sneakers provide evidence for revisiting that identity match.

The notebook on the right separates two records. An **episodic screenplay** records events using temporary person codes such as `P1` and `P2`; a **character table** holds hypotheses about who those codes refer to. New evidence revises the character table. When the session is saved, observations are linked to persistent characters. The next morning, Pebble uses the records to answer who ate the candy and show the supporting material.

<table>
  <tr>
    <td width="33%"><img src="docs/assets/demo/beat-1-names.jpg" alt="Learning Mia's name from natural conversation"></td>
    <td width="33%"><img src="docs/assets/demo/beat-2-ghosts.jpg" alt="Combining identity cues after the children put on matching ghost costumes"></td>
    <td width="33%"><img src="docs/assets/demo/beat-3-conflict.jpg" alt="Conflicting identity evidence from a stained sheet and a voice in the kitchen"></td>
  </tr>
  <tr>
    <td><b>Learn a name.</b> A spoken reference gives stranger <code>P1</code> the candidate name Mia, without prior enrollment.</td>
    <td><b>Combine cues.</b> With faces covered, voice, build, and spatio-temporal continuity can still help connect observations.</td>
    <td><b>Review a conflict.</b> The stained sheet suggests Leo, but Leo's voice comes from the kitchen. The earlier match needs review.</td>
  </tr>
  <tr>
    <td><img src="docs/assets/demo/beat-4-fixed.jpg" alt="A giggle and pink sneakers support revising the match to Mia"></td>
    <td><img src="docs/assets/demo/beat-5-saved.jpg" alt="Linking episodic records to persistent characters at session close"></td>
    <td><img src="docs/assets/demo/beat-6-recall.jpg" alt="Returning an answer with episodic records and supporting imagery"></td>
  </tr>
  <tr>
    <td><b>Revise the match.</b> A giggle and pink sneakers point to Mia. The temporary code is reassigned while the recorded events are preserved.</td>
    <td><b>Commit the session.</b> Consolidate evidence and link temporary identities to persistent characters. A person can remain unnamed when the evidence does not establish a name.</td>
    <td><b>Recall with evidence.</b> Answer “who ate the candy?” with the relevant screenplay events and the sneakers image as supporting material.</td>
  </tr>
</table>

### Event records in this scenario

These illustrative records describe the animation. An appearance-based description records the action; linking it to a person also makes the event usable for questions about who did what.

Appearance-based description:

```text
[21:15] A ghost in a ketchup-stained sheet went to the candy bowl.
[21:16] A ghost took a candy.
[21:22] A ghost took a candy.
[21:29] A ghost took a candy.
```

After identity resolution:

```text
[21:15] Mia, wearing Leo's ketchup-stained sheet, went to the candy bowl.
[21:16] Mia took a candy.
[21:22] Mia took a candy.
[21:29] Mia took a candy.
```

</details>

## Evaluation

| Benchmark | Reported score (%) | Scope |
|---|---:|---|
| [M3-Bench-robot](https://github.com/bytedance-seed/m3-agent) | **61.5** | Robot-view multimodal memory |
| [Video-MME](https://github.com/BradyFU/Video-MME) | **87.0** | Long videos, no subtitles |
| [LoCoMo-10](https://github.com/snap-research/locomo) | **83.3** | Conversational memory |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | **80.6** | Long-term conversational memory |

Text evaluation scripts are available. Complete model manifests, video evaluation code, and per-question artifacts for these reported scores are not yet published. See [evaluation notes](docs/evaluation.md) for category results, scoring conventions, and reproduction status.

<details>
<summary><b>Category breakdowns and historical baselines</b></summary>

**[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent)** — a multimodal memory benchmark shot from a robot's first-person view across everyday environments.

| | **Overall** | Person understanding | Multi-hop | Multi-evidence | Cross-modal | General knowledge |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PersonOS** | **61.5** | **73.5** | **63.5** | **62.8** | **59.0** | **48.0** |
| M3-Agent, reported as “same models” | 37.4 | 50.9 | 42.4 | 37.1 | 37.0 | 28.4 |
| M3-Agent, reported as “as published” | 30.7 | 43.3 | 29.4 | 32.8 | 31.2 | 19.1 |

The historical baseline labels are retained from the earlier report; a model manifest for “same models” is not included in this repository.

**[Video-MME](https://github.com/BradyFU/Video-MME)** — long videos, no subtitles.

| **Overall** | Synopsis | Object recog. | Spatial reas. | Object reas. | Temporal reas. | Action recog. | Action reas. | Temporal perc. | Attribute perc. | OCR | Counting | Spatial perc. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **87.0** | 93.3 | 92.6 | 90.9 | 87.9 | 87.9 | 84.1 | 85.0 | 83.3 | 85.2 | 78.6 | 70.8 | 33.3 |

Reported text-memory results:

| Benchmark | Overall | Breakdown |
|---|:---:|---|
| [LoCoMo-10](https://github.com/snap-research/locomo) | **83.3** | single-hop 89.3 · temporal 79.8 · multi-hop 77.3 · open-domain 58.7 |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | **80.6** | knowledge-update 91.7 · single-session-assistant 98.2 · single-session-user 89.1 · multi-session 76.9 · temporal 76.4 · preference 43.3 · abstention 73.3 |

</details>

## Documentation

| Resource | Contents |
|---|---|
| [Product website](https://personos-ai.com/) | Research initiative, ecosystem, and research roadmap |
| [Usage guide](docs/usage.md) | Installation, model configuration, video ingestion, and API reference |
| [Examples](examples/README.md) | Runnable text, image, and video examples |
| [HTTP service](server/README.md) | Service API, authentication, queues, and deployment |
| [Evaluation notes](docs/evaluation.md) | Reported results, scoring conventions, and reproducibility |
| [Configuration reference](.env.example) | Provider, storage, worker, and tracing settings |
| [Animated case study](docs/demo/README.md) | Storyboard, animation source, and rendering instructions |
| [Benchmark guide](scripts/bench/README.md) | Text evaluation commands, judge configuration, and scoring protocols |
| [Contributing](CONTRIBUTING.md) | Development setup and checks |
| [Releases](https://github.com/2049lab/personos/releases) · [PyPI](https://pypi.org/project/personos/) | Source releases and installable packages |

## Development

The current package is an alpha release. It includes persistent text memory, fact histories, user profiles, image understanding, video person association, asynchronous ingestion, and an HTTP service. Behavior depends on the configured models and input quality.

Planned work for this repository:

- [ ] Memory inspector: timeline, character gallery, and episodic records
- [ ] MCP server for agent clients
- [ ] Streaming ingestion for live camera feeds
- [ ] Paper, full video evaluation code, and reproducible result artifacts

The broader initiative studies lifelong memory, including parametric memory and learning from feedback. These are [research directions](https://personos-ai.com/research.html), not capabilities shipped by this package. [PersonOS Context and Memory Workbench](https://personos-ai.com/ecosystem.html) are separate ecosystem tools; the website identifies their prototype and access status.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [SECURITY.md](SECURITY.md) for reporting security issues. Project questions and reproducible bug reports can be submitted through [GitHub Issues](https://github.com/2049lab/personos/issues).

## License

PersonOS is licensed under [Apache 2.0](LICENSE). External model weights and datasets retain their respective licenses. The separately downloaded M3-Bench sample is provided by [ByteDance-Seed](https://github.com/bytedance-seed/m3-agent) under CC BY-NC-SA 4.0.
