# Contributing

[← PersonOS](README.md) · [Framework design](docs/framework.md)

Contributions should keep the core usable with local SQLite and configurable model providers. Application-specific infrastructure and heavy optional dependencies belong behind adapters.

## Set up

Requires Python 3.10+.

```bash
git clone https://github.com/2049lab/PersonOS.git
cd PersonOS
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The default test suite uses in-memory SQLite and fake model responses. It needs no model credentials or shared database. Leave `PERSONOS_TEST_BACKEND` unset for local tests.

## Check a change

```bash
python -m ruff check .
python -m pytest -q
python scripts/check_cold_start.py
python scripts/check_no_internal_refs.py
python scripts/check_clean_install.py
```

- **Tests** exercise storage, ingestion, recall, API behavior and user isolation. The normal database fixture wraps each test in a rollback scope and rejects writes outside that scope. Dedicated tests of database behavior can use their own temporary databases.
- **Cold start** checks that representative modules import without configuration or network connections.
- **Internal-reference check** scans tracked files for private dependencies and deployment identifiers. Stage intended new files before the final scan so they are included.
- **Clean install** checks that the text core runs without optional dependencies.

For changes limited to comments and docstrings, `python scripts/check_translation_safe.py` can compare the executable AST. It is not a gate for an intentional prompt or code change.

The suite can also target MySQL by setting `PERSONOS_TEST_BACKEND=mysql` and `PERSONOS_DB_URL`. Use a disposable database. SQLite tests alone do not validate a deployment's MySQL version, permissions or connection settings.

## Memory behavior

Keep source evidence, user isolation and empty-result semantics intact. A segment may contain no personal memory. Do not turn assistant output, task instructions or failed extraction into fallback user facts.

Test observable outcomes and failure modes: repeated input, cross-user references, malformed model responses, unavailable optional providers and pagination during concurrent changes. Use neutral synthetic inputs. Do not publish private conversations, credentials or captured model responses containing personal data.

Prompt changes need more than green unit tests to establish model quality. Use the [evaluation tools](scripts/bench/README.md) and record the models, configuration and dataset scope for any performance claim. The optional [record/replay harness](scripts/baseline/) helps compare structural behavior when suitable fixtures are available.

## Code and documentation

Explain the reason for non-obvious code, keep public behavior backward compatible where practical, and document intentional changes. The lint configuration in `pyproject.toml` defines the code style.

New providers register in [`personos/providers/registry.py`](personos/providers/registry.py). Import optional SDKs lazily so adding an integration does not enlarge the required core.

Keep the English and Chinese READMEs aligned. Put setup details and API examples in the usage guides; keep the repository homepage focused on what the framework does and how to start. New public methods need examples and a clear account of model calls, user scope and failure behavior.

Use [GitHub Issues](https://github.com/2049lab/PersonOS/issues) for reproducible bugs and focused proposals. Report vulnerabilities through [SECURITY.md](SECURITY.md).
