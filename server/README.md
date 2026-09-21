# PersonOS HTTP server

A deployable wrapper around the library. **Not part of the pip package** — it
has its own dependencies and lifecycle, the same arrangement mem0, Graphiti and
Memobase use.

If you are embedding memory in your own application, you do not need this: use
`from personos import Memory` directly. Run this when you want memory as a
service — several clients, ordered ingestion across processes, backpressure.

## What it adds over the library

- **Ordered ingestion.** `POST /ingest` returns 202 immediately and queues the
  message. A per-session queue with a cursor guarantees that concurrent writes
  to one session are consumed in order, exactly once.
- **Backpressure.** A session whose queue is backed up gets 503 with Retry-After
  rather than an unbounded buffer.
- **Multi-tenancy over HTTP**: token registration, per-request user scoping.

## Run

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Configuration is the library's (see the root `.env.example`), plus:

| Variable | Meaning |
| --- | --- |
| `PERSONOS_AKSK_MAP` | JSON of `{access_key: secret}` for request signing |
| `PERSONOS_REDIS_URL` | Required for more than one worker: the queue, session locks and segment state live there |

With several workers and no Redis the queue is per-process, so ordering holds
only within a worker. The server logs this at startup.
