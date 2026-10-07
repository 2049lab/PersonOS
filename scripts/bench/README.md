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

## LongMemEval-S

LongMemEval-S is the short variant from
[xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval): 500
questions, each its own synthetic user with ~40 haystack sessions stitched
from ShareGPT / UltraChat, six question_types plus 30 abstention (`_abs`).
The dataset and judge are official — judge prompts are copied character-for-
character from `src/evaluation/evaluate_qa.py:get_anscheck_prompt` and the
official `yes/no` parser is used (`label = 'yes' in eval_response.lower()`).

```bash
# 1. Fetch the dataset (~277 MB; xiaowu0162/longmemeval-cleaned, MIT)
python -m scripts.bench.longmemeval --download

# 2. Smoke run: 3 questions of any type (one ingest takes ~1 min with --n-sessions 2)
python -m scripts.bench.longmemeval --n-sessions 2 \
    --questions-file <(printf "gpt4_59c863d7\ngpt4_45189cb4\nd7c942c3\n") \
    --run-dir data/bench/runs/lme-smoke --concurrency 1

# 3. Full benchmark: every question, every haystack session (~493 turns each)
python -m scripts.bench.longmemeval --all --concurrency 5 \
    --run-dir data/bench/runs/lme-$(date +%Y%m%d)
```

### Multi-day resumable runs

Each question is independent (its own `user_id = "lme-<qid>"`, its own
trace JSON in the run dir), so a 500-question run can be split across
processes / days and accumulated into one summary:

- `--questions-file <ids.jsonl>` — one question_id per line. Use this to
  shard a run (e.g. `head -250 longmemeval_s_cleaned.json` question_ids
  on shard A, the tail on shard B). Pass the **same `--run-dir`** to both
  shards so traces accumulate.
- A question whose `<qid>.trace.json` already exists is **skipped** on
  re-run (use `--force-reanswer` to re-judge even when the trace exists).
- A question whose user namespace already has atoms is **not re-ingested**
  (use `--force-reingest` to clear the store and start over).
- After each question completes, a row is appended to
  `<run-dir>/progress.jsonl` (`ts`, `question_id`, `task`, `abstention`,
  `score`, `secs`). Tail it to see live progress; aggregate into
  `summary.md` after every launch.

### Scoring protocol

The official LME protocol:

- The model produces one free-text answer per question (here, the R5 answer
  from `run_recall` — no second-stage answerer, matching what an LLM-only
  baseline would output).
- A separate judge LLM (defaults to `PERSONOS_JUDGE_*`, falls back to the
  service LLM) scores each answer via the per-task prompt from
  `get_anscheck_prompt`. Six question_types share three templates:
  - `single-session-user`, `single-session-assistant`, `multi-session` —
    "If the response is equivalent to the correct answer or contains all
    the intermediate steps ... answer yes."
  - `temporal-reasoning` — same, plus "do not penalize off-by-one errors
    for the number of days."
  - `knowledge-update` — same, plus "if the response contains some previous
    information along with an updated answer, ... as long as the updated
    answer is the required answer."
  - `single-session-preference` — the prompt swaps "Correct Answer" for
    "Rubric": "the response is correct as long as it recalls and utilizes
    the user's personal information correctly."
- Abstention (`_abs` in question_id, 30 in LME-S) flips the prompt: the
  "Correct Answer" becomes an "Explanation" and the judge asks whether the
  model identified the question as unanswerable. Both answerable and
  abstention scores are reported side by side.
- Accuracy is reported **per question_type** and as an overall answerable
  score (abstention excluded from the headline number). The judge model
  and prompts are written into every report's `meta` block.

### Artifacts

Per question: `<qid>.trace.json` (full pipeline trace: ingest decisions,
closed cells, per-question rewrite / retrieval pool / ranked materials /
review / deep trace / memories) — exactly one question per trace, so a
crash mid-question never corrupts a half-written sibling. Per run:
`summary.json` / `summary.md` (aggregated across every trace on disk,
including yesterday's shards) and `progress.jsonl` (live append-only log,
safe to `tail -f` from a cron watcher).
