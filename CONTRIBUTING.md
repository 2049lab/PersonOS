# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest -q          # ~600 tests, under 20 seconds, no network, no database
```

The suite runs against in-memory SQLite. Nothing external is required, and
nothing you run locally can touch a shared system.

To run the same suite against MySQL — which is how the SQL dialect translation
is shown to be behaviour-preserving — set `PERSONOS_TEST_BACKEND=mysql` plus
`PERSONOS_DB_URL`.

## The checks, and why each exists

```bash
python scripts/check_cold_start.py         # importing works with no configuration
python scripts/check_clean_install.py      # the core installs without the extras
python scripts/check_no_internal_refs.py   # no internal identifiers ship
python scripts/check_translation_safe.py   # a comment-only change really is one
```

They are cheap and they each caught something real:

- **Cold start** — a module that reads a secret while being imported breaks
  `from personos import Memory` for anyone without that secret, and the failure
  lands at import time, the least debuggable place. Note it imports several
  modules explicitly: checking only `import personos` was permanently green and
  caught nothing, because the package `__init__` is nearly empty.
- **Clean install** — declaring a small dependency list is easy; keeping it true
  is not. One module-level import of an optional package silently turns an extra
  into a requirement, and only the minimal installer ever sees it.
- **Translation safety** — prompt constants are English strings that models
  read. Editing one changes pipeline output with no test failure to show for it.
  This parses before and after, strips docstrings, and compares the AST.

## What the tests are for

Most test names describe a *failure mode*, not a function. The docstrings say
why the assertion is what it is — often reconstructing an incident. Keep that
habit: a test that says `test_rerank` teaches nothing when it breaks at 2am.

When touching retrieval or writing, the useful question is not "do the tests
pass" but "did the structure of the answer change". The suite cannot see that
on its own, because the pipeline is driven by a language model. There is a
record-and-replay harness in `scripts/baseline/` for exactly this: it freezes
one real run and replays it offline, comparing structural fingerprints rather
than answer text.

## Style

- Comments explain **why**, not what. The code already says what.
- Prefer stating the trade-off you rejected. "8 rather than 50 because the
  bottleneck is upstream latency, not local CPU" is worth five lines; "the
  concurrency limit" is worth none.
- Line length 100, `ruff` enforces the rest.

## Adding a provider

One entry in `personos/providers/registry.py` and one class. The registry
imports by dotted path on demand, which is what lets heavy dependencies stay
optional — so do not import your SDK at module level.
