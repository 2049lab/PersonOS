# Benchmarks

Reproducible evaluations of personos against public memory benchmarks. One
script per dataset, each self-contained: download the data, run the benchmark,
write the full pipeline trace and the reports to disk. Everything calls the
core directly — no HTTP server is involved.

Artifacts land in `data/bench/runs/<run_id>/` (git-ignored). Only datasets we
have actually run live here; a script is added together with its first result,
never ahead of it.

## LoCoMo-10

Ten long multi-session conversations (~300 turns each) with ~200 QA pairs per
conversation across five categories (multi-hop / temporal / open-domain /
single-hop / adversarial — the last excluded by community convention).

```bash
# 1. Configure the environment (see below), then fetch the dataset
python -m scripts.bench.locomo --download

# 2. Smoke run: one conversation, 3 sessions, 8 questions
python -m scripts.bench.locomo --conv conv-26

# 3. Full benchmark: all ten conversations, every answerable question
python -m scripts.bench.locomo --all --n-sessions 99 --n-questions 999 --concurrency 4
```

Useful flags: `--skip-ingest` (reuse a loaded conversation, re-answer only),
`--ingest-only`, `--questions-file` (re-run specific questions), `--mode
auto|fast|deep`.

### Configuration

The memory system reads the normal service settings; the judge reads its own,
so the model being measured and the model scoring it can differ:

```bash
# the memory system under test
PERSONOS_LLM_API_KEY=...            PERSONOS_LLM_BASE_URL=...  PERSONOS_LLM_MODEL=...
PERSONOS_EMBEDDING_API_KEY=...      PERSONOS_EMBEDDING_MODEL=...  PERSONOS_EMBEDDING_DIM=...
# optional but recommended: a reranker (any dialect — see .env.example)
PERSONOS_RERANKER_PROVIDER=...      PERSONOS_RERANK_BASE_URL=...
PERSONOS_RERANK_API_KEY=...         PERSONOS_RERANK_MODEL=...
# the judge (empty = reuse the service LLM)
PERSONOS_JUDGE_BASE_URL=...         PERSONOS_JUDGE_API_KEY=...   PERSONOS_JUDGE_MODEL=...
```

One environment caveat: `.env` loses to variables already exported in your
shell (`load_dotenv` does not override). If you launch the benchmark from an
agent shell that sets its own `ANTHROPIC_*` (Claude Code does), clear them
first — `env -u ANTHROPIC_BASE_URL -u ANTHROPIC_API_KEY -u ANTHROPIC_MODEL ...`
— or the benchmark will call that endpoint instead of your configured one.

### Scoring protocol

Two conventions are reported side by side — declare which one you are quoting:

- **Mem0 convention** (`score`): the recall outcome is handed to a mode-A
  answerer (brief + atoms from the top-20 reranked cells, grouped by cell), and
  the LLM judge scores the answerer's output. Comparable with published Mem0
  numbers.
- **Product convention** (`score_r5`): the judge scores the R5 answer exactly
  as the product returns it.

The judge is a Mem0-style binary judge (CORRECT/WRONG), v2: relative times in
gold and prediction are converted to absolute dates deterministically by the
harness before judging. Accuracy = CORRECT / questions; category 5
(adversarial, no gold answer) is excluded from the denominator. The judge
model and prompts are written into every report's `meta` — if the judge is not
declared, the numbers are marketing.

### Artifacts

Per conversation: `<conv>.trace.json` (every ingest boundary decision, every
closed cell, and per question the rewrite / retrieval pool / ranked materials /
review / deep trace / memories / gold-evidence chain with its break point) and
`<conv>.report.md` (totals, per-category breakdown, error analysis). Per run:
`summary.json` / `summary.md`.
