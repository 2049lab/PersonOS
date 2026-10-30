# Framework design

[← PersonOS](../README.md) · English · [简体中文](framework.zh-CN.md) · [API guide](usage.md)

PersonOS builds long-term memory about a person from the material an application submits. The application chooses the user and the source; the framework retains evidence, extracts relevant memory and retrieves it later.

## The memory boundary

The text write path asks whether a segment tells us something about the user: their circumstances, relationships, preferences, goals, plans, commitments or experiences. A current project or a grounded work preference can qualify. An instruction to perform a task, the assistant's implementation steps, a proposed answer or a third party's unrelated story does not establish a personal fact.

Speaker attribution matters. A suggestion becomes evidence about the user only when the conversation supports that attribution. An inference must be identified as an inference and stored as a `claim`; the episode itself records what the source supports. The deep-recall write-back tool has a narrower rule: it can retain explicitly stated personal facts, plans and opinions, but must not introduce new inferences into memory or episode text.

An application can pass `scenario` to focus extraction, for example on dietary preferences. This narrows attention within personal memory. It cannot make unrelated task content into a fact about the user.

These are extraction rules implemented through prompts and parsing. They are not a guarantee that every model will classify every conversation correctly. Keep the original evidence available for inspection and evaluation.

## Memory layers

| Layer | Responsibility | What it preserves |
|---|---|---|
| Evidence | Retain the source | Original utterances, timestamps and media references |
| Episode (`MemCell`) | Keep the personal event in context | Narrative, topic, time window and evidence references |
| Atom (`MemoryAtom`) | Make facts retrievable | A bounded proposition, attribution and links to its episode and evidence |
| Chain (`ChainInfo`) | Connect related statements | Membership and append order across observations |
| Profile | Consolidate a view of the user | A derived profile built from accumulated episodes |

Retrieval searches atoms and assembles the corresponding episodes into material for answering. Chains group related observations without overwriting their earlier states. The answer must consider the statements' dates and context when resolving a change.

Text extraction may return no episode. It may also produce an episode with no atoms. Original evidence is retained in either case. An episode without atoms remains available through episode browsing but does not become an invented fallback atom merely to make it retrievable.

If episode generation fails, the write path preserves evidence and skips the derived episode. It does not silently turn the raw transcript into personal memory. Failed or skipped extraction is distinct from a successful memory write; queue completion alone does not establish memory quality.

## Read paths

| Operation | Model work | Intended use |
|---|---|---|
| `list_memories`, `list_chains`, `get_chain` | None | Inspect saved records, build a memory browser, check provenance |
| `search(mode="flash")` | Embeddings and optional reranking | Give an agent relevant material for its own response |
| `search(mode="fast")` | Query processing, retrieval, answer generation and review | Ask a question and receive an evidence-linked answer |
| `search(mode="auto")` | Fast recall, with optional escalation | Use deep recall when available and needed |
| `search(mode="deep")` | Multi-step model-driven retrieval | Explore memory beyond the fast path; requires the `deep` extra |

`flash` returns retrieved candidates without a generated answer or an answer-review verdict. Its results should not be presented as verified conclusions. See the [API guide](usage.md#recall-and-provenance) for result fields and restrictions.

Browsing is bounded and user-scoped. Chain indexes return metadata; callers fetch a chain's history separately. Chain detail reports incomplete or changed histories explicitly instead of inventing a continuous timeline.

## Framework refresh

This revision adds a focused set of general capabilities to the existing open-source implementation. It keeps SQLite and configurable public model providers as the default development path.

| Capability | Decision | Reason |
|---|---|---|
| Personal-memory extraction and valid empty results | Included | Prevents nonpersonal conversation segments from being forced into memory |
| Evidence identity | Included | Distinct messages can contain identical text; record IDs determine retry identity |
| Paginated memory and chain browsing | Included | Makes records inspectable without extra model calls or unbounded history responses |
| Material-only recall | Included | Lets the host agent control its own answer generation |
| Declarative tag schemas | Deferred | Needs a stable public schema, filtering rules and failure handling across write paths |
| Direct episode submission | Deferred | Needs an explicit contract for caller-authored text and records without source evidence |
| Session deletion | Deferred | Requires consistent cleanup across profiles, queues, derived chains and media references |
| Additional vector stores | Deferred | Needs an optional backend contract and migration tests without changing local defaults |
| Application-specific device workflows | Kept outside the core | Belong in adapters or applications with their own dependencies and deployment requirements |

Existing optional image and video backends remain available. They have separate dependencies, data flows and model requirements; this revision does not establish new video accuracy claims.

### Compatibility notes

- Existing SDK methods and recall modes remain available. Browsing methods and `flash` are additive.
- All recall modes now validate a nonempty query, a known mode and integer `top_k` from 1 to 100. Booleans are not accepted as integers. Invalid values raise `ValueError` in the SDK; HTTP schema errors return `422`, and invalid modes or out-of-range integers return `400`.
- Text extraction now accepts an empty result. Applications must not assume every submitted segment yields an episode or at least one atom.
- New evidence records use message identity rather than text identity. Repeated wording from distinct messages is retained. Retrying a storage append with the same record ID is idempotent; this is not an application-wide exactly-once delivery guarantee.
- Existing content-deduplicated records are not automatically split, re-extracted or migrated. The change applies to new writes.
- The original evidence is still stored for skipped personal-memory extraction. Skipping memory is not source deletion.
- Historical benchmark numbers predate this revision. Unit tests check behavior and isolation; they do not re-establish extraction or retrieval quality on those benchmarks.

New APIs are documented against the source tree. They become part of a published package only when a release containing these changes is built and uploaded.

## Storage and deployment

The Python API scopes memory operations to `user_id`; your application is responsible for binding that identifier to an authenticated user. The HTTP service resolves its user token before memory operations. Neither interface makes a client-supplied user ID into an authorization policy by itself.

A single process can use SQLite and in-process session state. Multiple processes require shared MySQL and Redis. Local storage does not imply local inference: content is sent to the model endpoints you configure. Keep deployment, access control and data retention decisions explicit in the application using the framework.

`reset(user_id=...)` clears stored records and selected in-process session state. It is not a complete erasure workflow for queued work, externally hosted media or provider-side data. Do not describe it as one.

[Usage guide](usage.md) · [HTTP service](../server/README.md) · [Evaluation](evaluation.md) · [Contributing](../CONTRIBUTING.md)
