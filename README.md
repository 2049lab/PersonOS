# PersonOS

Layered long-term memory for agents — and the only open one that watches video
and remembers *who* was in it.

```bash
pip install personos
```

```python
from personos import Memory

m = Memory()                      # SQLite under ~/.personos; one env var to set

m.add("I moved from Hangzhou to Shanghai in June", user_id="alice", session_id="s1")
m.end_session(user_id="alice", session_id="s1", sync=True)   # wait for the queue to drain

print(m.search("where do I live?", user_id="alice").ans.answer)
```

The only required configuration is a chat/embedding endpoint:

```bash
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible gateway
```

Not sure what your configuration can do? `personos doctor` reads it and tells
you what works, what is off, and what to set.

## Why another memory library

Most store a flat list of facts and search it. PersonOS keeps a **layered,
append-only record** and refuses to resolve contradictions at write time:

```
evidence  ──►  memcell (episode)  ──►  atom  ──►  atom_chain
raw turns      the narrative unit     the          the same fact over time,
never edited   used for answering     retrieval    grouped, never collapsed
                                      anchor
```

Two consequences that show up in practice:

**Contradictions survive.** "15 fish" → "actually 13" → "actually 11" stay as
three linked atoms with their timestamps. The answer layer decides what is
current; the memory layer never silently overwrites the past. Ask *"how many
fish do I have"* and you get the current count; ask *"did that change"* and the
history is still there.

**Answers cite episodes, not fragments.** Atoms are the retrieval index — short,
self-contained propositions that embed well. The narrative episode is what the
model actually reads. Retrieval granularity and answering granularity are
deliberately different, because what makes a good search key makes a poor
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

Every other open memory framework is text-only, or turns an image into a caption
at ingest. PersonOS takes **video clips** and builds stable *character* entities
from faces, body shots and voiceprints — so someone recognised in clip 12 is the
same person three sessions later, without anyone enrolling them first.

Optional (`pip install personos[identity]`, ~2 GB of model dependencies). The
text core imports none of it.

## Configuration

Everything except the model endpoint is optional. Unset means a capability is
off or degraded, never that the text path breaks.

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

The rule is: **cannot do it at all → raise; did it partially → return and say
so** in `result.warnings`. A missing optional capability never fails a write.

## Examples

One story in three chapters:
[a week of conversation](examples/quickstart.py) (watch the profile build
itself between days) · [with a photo](examples/images.py) ·
[with video and person identity](examples/video.py). See
[examples/README.md](examples/README.md) for what each needs and real output.

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
per-session ordered queue (the same machinery the server deployment uses —
FIFO per session, fair scheduling across sessions, backpressure when a session
is overloaded) and return an `AddReceipt` immediately, so memory writes never
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
yet, and rather than pretend, the honest workaround is
`await asyncio.to_thread(m.search, q)`.

## Running it as a service

[`server/`](server/README.md) is a FastAPI deployment — ordered ingestion across
processes, backpressure, token-scoped multi-tenancy. It is **not** part of the
pip package; it has its own dependencies and lifecycle.

If you are embedding memory in an application, you do not need it.

## Status

Pre-release (`0.1.0.dev0`). The memory pipeline has been running in a production
deployment; the packaging around it is new, and the public API may still move
before 0.1.0.

## License

Apache-2.0. See [LICENSE](LICENSE).
