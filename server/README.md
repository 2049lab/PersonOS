# PersonOS HTTP service

[← PersonOS](../README.md) · [Python API](../docs/usage.md) · [Framework design](../docs/framework.md)

The service exposes the memory library over HTTP, with user tokens, request signing and ordered ingestion. It ships with the source repository and has its own dependencies; it is not included in the library wheel.

## Run locally

From the **repository root**, in a Python 3.10+ environment:

```bash
python -m pip install -e .
python -m pip install -r server/requirements.txt
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Configure chat and embedding providers as described in the [usage guide](../docs/usage.md#text-memory). Browse the running service's `/docs` for its current OpenAPI schema. The historical `/` and `/api-doc` addresses redirect there. [HTTP authentication](authentication.md) documents signing, user tokens and a runnable client example.

| Setting | Purpose |
|---|---|
| `PERSONOS_AKSK_MAP` | JSON mapping of access keys to signing secrets |
| `PERSONOS_DB_URL` | Shared MySQL database for multiple workers |
| `PERSONOS_REDIS_URL` | Shared queues, session locks and pending segment state |
| `PERSONOS_ENV` | Namespace for shared Redis state |

Without Redis, queue and session state are process-local. Configure **both shared MySQL and Redis** before running multiple workers. Provider, storage and pool options are listed in [`.env.example`](../.env.example).

## API surface

Business routes use the `/api/v1` prefix and the `{code, data, msg}` response envelope. Memory operations resolve `X-User-Token` to the owning user's stores; request signing is separate from this user scope. Register a user through `POST /api/v1/users/register` and retain the returned token.

| Route | Use |
|---|---|
| `POST /ingest` | Queue text, image or video input |
| `POST /session/end` | Close the trailing segment and finish video identity work |
| `GET /queue/status` | Inspect a session's ingest backlog |
| `GET /tasks/{task_id}` | Inspect an asynchronous task |
| `POST /recall` | Retrieve with `flash`, `fast`, `auto` or `deep` |
| `GET /memories` | Filter and page through saved atoms with episode context and source evidence |
| `GET /chains` | Page through chain metadata without loading full histories |
| `GET /chains/{chain_id}` | Read a bounded page of chain history |
| `GET /episodes` | Browse episodic records |
| `GET /profile` | Read the consolidated user profile |
| `GET /trace/{node_id}` | Inspect a source or memory's provenance |

All paths in the table are relative to `/api/v1`. `caller` scopes session identifiers; use the same caller on writes and session-specific reads.

### Browse memory

`GET /memories` accepts `session_id`, `caller`, `object_type`, `holder`, `kind`, `domain` and `query`. The query is a literal text filter; it does not call a model. Memory and chain lists accept `limit` (1–100, default 20) and `offset` (0–100000), returning `items` and `has_more` with the pagination values.

A chain detail uses `limit` and the returned `next_cursor`. Its `state` is `complete`, `more`, `incomplete` or `changed`. Restart from the first page after a change. An unknown chain or another user's chain returns `404`. Invalid filter values or cursors return `400`; query type errors and pagination bounds return `422`.

### Return material to an agent

Set `mode="flash"` in `POST /recall` for a text-only query. It uses embeddings and any configured reranker, with no chat-based rewriting, answer generation or review. The response includes `materials` (episodes and references) and `memories` (atoms with evidence), with an empty `answer`, empty `cited_cells` and `verdict: null`. The calling agent decides how to use the retrieved candidates.

For every recall mode, `top_k` must be an integer from 1 to 100. HTTP rejects booleans and non-integer values with `422`; an out-of-range integer or unknown mode returns `400`. Flash does not use `scenario` or profile injection and does not accept query images. Other modes retain their existing behavior; `deep` requires the corresponding optional dependencies.

## Ingestion and retries

`POST /ingest` and `POST /session/end` return `202` after queueing by default. `sync=true` waits for consumption up to the service timeout. A full queue returns `503` with `Retry-After`.

Each session is consumed in order. Replaying the same queued message reuses its evidence identity; identical text in distinct messages remains distinct. This is not an end-to-end exactly-once guarantee: a crash between generated-memory persistence and queue/segment progress can repeat derived work.

Queue completion means the submitted work has been processed. The personal-memory write path may retain evidence without creating an episode or atoms. Extraction behavior and compatibility changes are described in [framework design](../docs/framework.md).
