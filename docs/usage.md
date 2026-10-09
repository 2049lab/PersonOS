# Usage guide

[← PersonOS](../README.md) · English · [简体中文](usage.zh-CN.md) · [Evaluation](evaluation.md)

PersonOS supports text, image and video memory through the Python `Memory` API or a separately deployed HTTP service. This guide covers installation, model configuration, storage and the write/read lifecycle.

[Text memory](#text-memory) · [Video memory](#video-memory) · [Installation options](#installation-options) · [Model weights](#model-weights) · [Configuration](#configuration) · [Core API](#core-api) · [HTTP service](#http-service)

## Text memory

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

For another provider, set its chat model, embedding model and vector dimension explicitly. See [provider configuration](#provider-configuration) for endpoint and credential inheritance.

### Write and retrieve

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

## Video memory

Keep the text configuration above. Install the identity dependencies and configure a multimodal endpoint that supports the backend's `video_url`, `image_url` and `input_audio` content formats. A text-only or image-only chat endpoint is insufficient.

```bash
python -m pip install 'personos[identity]'
export PERSONOS_VIDEO_BACKEND=real
export PERSONOS_MLLM_API_KEY="your-multimodal-api-key"
export PERSONOS_MLLM_BASE_URL="https://your-provider.example/v1"
export PERSONOS_MLLM_MODEL="your-video-model"
export PERSONOS_MEDIA_BASE_URL="https://your-media-host.example/media"
```

Replace the example URLs and model name with your service settings. With local storage, serve the contents of `<PERSONOS_DATA_DIR>/media` (default `~/.personos/media`) at the media URL so the model service can fetch clips, or configure [OSS storage](../.env.example). Setting the URL does not start a media server. Identity dependencies and model weights require additional downloads; see [model weights](#model-weights).

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

Video is processed in clip order, with identities committed at session close. Allow minutes per clip depending on the model and recording. See [the video example](../examples/README.md#3-video-and-person-identity) for sample preparation, hosting and data-source terms.

## Installation options

Use `python -m pip install 'personos[EXTRA]'`; combine extras with commas, for example `'personos[deep,image]'`.

| Install | What it adds |
|---|---|
| `personos` | Layered text writes, fast recall, user profiles and local SQLite |
| `personos[deep]` | Multi-step deep-recall agent |
| `personos[image]` | Image handling helpers; image understanding also requires an MLLM endpoint |
| `personos[identity]` | Video with face / build / voiceprint identity (~2 GB) |
| `personos[mysql]` `personos[redis]` | Multi-process deployments |
| `personos[oss]` | Object storage instead of local media files |
| `personos[anthropic]` | Anthropic-protocol chat; embeddings need their own endpoint |
| `personos[observability]` | Optional Langfuse tracing |
| `personos[all]` | All optional capabilities |

## Model weights

The real identity backend uses the following models:

| Capability | Weights | Location |
|---|---|---|
| Face recognition | InsightFace `buffalo_l` (~300 MB) | `~/.insightface` (auto-download) |
| Voiceprints | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace cache; override with `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` |
| Multimodal understanding | none, remote API | `PERSONOS_MLLM_*` points at a compatible endpoint |

Model weights and external datasets have their own terms, separate from the library's license.

## Configuration

### Provider configuration

The default providers use the OpenAI-compatible wire format. Chat completions and embeddings are separate capabilities, even when one endpoint serves both.

| Setting | Default or inheritance |
|---|---|
| `PERSONOS_LLM_BASE_URL` | `https://api.openai.com/v1` |
| `PERSONOS_LLM_MODEL` | `gpt-4o-mini` |
| `PERSONOS_LLM_API_KEY` | Required for the default chat provider |
| `PERSONOS_EMBEDDING_BASE_URL` | Inherits `PERSONOS_LLM_BASE_URL` when empty |
| `PERSONOS_EMBEDDING_API_KEY` | Inherits `PERSONOS_LLM_API_KEY` when empty |
| `PERSONOS_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `PERSONOS_EMBEDDING_DIM` | `1536`; must match the vectors returned by the embedding model |
| `PERSONOS_MLLM_BASE_URL` | Inherits `PERSONOS_LLM_BASE_URL` when empty |
| `PERSONOS_MLLM_API_KEY` / `PERSONOS_MLLM_MODEL` | Set explicitly to enable multimodal understanding; no key or model inheritance |
| `PERSONOS_MLLM_ENDPOINT` | Optional full URL override for the multimodal endpoint |

A chat-compatible service may not support embeddings. Set `PERSONOS_EMBEDDING_BASE_URL` and `PERSONOS_EMBEDDING_API_KEY` when using a separate embedding provider. Changing the configured vector dimension does not request a different output dimension from the provider.

For Anthropic-protocol chat, install `personos[anthropic]`, set `PERSONOS_LLM_PROVIDER=anthropic`, and configure `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`; `ANTHROPIC_BASE_URL` selects a compatible gateway. Configure embeddings separately through the `PERSONOS_EMBEDDING_*` variables. Other provider implementations can be selected through the [provider registry](../personos/providers/registry.py).

### Storage and optional capabilities

| Capability | Default | Configuration |
|---|---|---|
| Database | SQLite at `~/.personos/personos.db` | `PERSONOS_DB_URL` selects MySQL |
| Media storage | Files under `~/.personos/media` | `PERSONOS_MEDIA_BACKEND=oss` selects object storage |
| Image understanding | Off | Configure an MLLM; otherwise images are stored without a searchable description |
| Video identity | Off | Identity dependencies, `PERSONOS_VIDEO_BACKEND=real`, a compatible MLLM and fetchable media URLs |
| Reranking | Off; retains retrieval fusion order | Set `PERSONOS_RERANK_API_KEY`, `PERSONOS_RERANK_MODEL` and the endpoint as needed |
| Deep recall | Off | Install `personos[deep]` |
| Shared session state | In-process | Configure `PERSONOS_REDIS_URL` for multiple workers |
| Tracing | Off | Install `personos[observability]` and configure Langfuse |

`PERSONOS_DATA_DIR` relocates the local database, media and default logs. Model calls use the configured endpoints; local storage does not determine where inference runs. Multiple workers require a shared MySQL database and Redis.

The library reads `.env` from the working directory or its parents; already exported variables take precedence. Use `PERSONOS_ENV_FILE` to select a specific file. [`.env.example`](../.env.example) lists model, storage, timeout, worker and tracing settings.

Unavailable required capabilities raise `MissingCapability`. Optional image understanding emits a warning and retains the source when unavailable. `personos doctor` reports configured capabilities and setup guidance.

## Core API

The following calls use an existing `Memory` instance:

```python
memory.add(messages, user_id=..., session_id=...)
memory.end_session(user_id=..., session_id=..., sync=True)
memory.flush(user_id=..., session_id=...)
memory.search(query, user_id=..., mode="auto")  # auto | fast | deep
memory.profile(user_id=...)                    # distilled user profile
memory.trace(node_id, user_id=...)             # provenance in both directions
memory.capabilities()                         # configuration and dependency checks
memory.reset(user_id=...)                      # delete this user's data
```

`add()` accepts a string, a message dictionary or a list of dictionaries. A message can contain an `image` (bytes or a path), or a `video` (path, file object or URL). Send video clips and conversation turns in separate calls. See the [text, image and video examples](../examples/README.md) for complete inputs.

### Write ordering

`add()` and `end_session()` are asynchronous by default: they enqueue work and return an `AddReceipt`. Processing is FIFO within each session, with fair scheduling across sessions and backpressure when overloaded.

Pass `sync=True` or call `flush(user_id=..., session_id=...)` after enqueueing to wait for that session's work to finish. Close the session to finalize its trailing episode and commit video identities. `flush()` drains the queue; it does not itself close the session. Profile consolidation can continue separately in the background.

`timeout_s` controls the wait for synchronous operations. A wait that exceeds this limit raises `TimeoutError`. A full queue raises `QueueBusy`; the HTTP API returns `503` with `Retry-After`.

### Recall and provenance

Reads (`search`, `profile`, `trace`) are synchronous. `mode="fast"` uses the fast recall path; `mode="auto"` can escalate when deep recall is installed; `mode="deep"` requires the `deep` extra.

```python
result = memory.search("Who ate the candy?", user_id="pebble")
print(result.ans.answer if result.ans else "No answer returned.")
print(result.ans.cited_cells if result.ans else [])
```

| Result field | Contents |
|---|---|
| `result.ans` | Answer and cited episodes, when an answer is present |
| `result.rw.subject` | Resolved subject of the question |
| `result.hits` | Retrieved atoms in fusion order |
| `result.reviews` | Review verdicts for the draft answer |
| `result.to_public()` | Public representation as a plain dictionary |

Use the cited episodes and `trace()` to inspect the source evidence behind an answer or identity match.

## HTTP service

[`server/`](../server/README.md) provides the HTTP wrapper and has its own dependencies. It is distributed with the source repository, separately from the library wheel. From the repository root, with the model configuration above:

```bash
python -m pip install -r server/requirements.txt
python -m pip install -e .
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

The service uses the library configuration, plus:

| Setting | Purpose |
|---|---|
| `PERSONOS_AKSK_MAP` | JSON mapping of access keys to signing secrets |
| `PERSONOS_DB_URL` | Shared MySQL database for multiple workers |
| `PERSONOS_REDIS_URL` | Shared ingest queues, session locks and segment state |
| `PERSONOS_ENV` | Namespace for shared Redis state |

Without Redis, queues and session state remain process-local. Configure shared MySQL and Redis before adding workers. Pool sizes and queue limits are documented in [`.env.example`](../.env.example); API routes and signing details are in the [HTTP API reference](../server/api_doc.html).

[← PersonOS](../README.md) · [简体中文](usage.zh-CN.md) · [Evaluation](evaluation.md)
