# Releasing personos

How to publish an update to PyPI. Anyone with the PyPI API token can do this.

## Versioning

SemVer while in 0.x:

- `0.1.0 → 0.1.1` — bug fixes, no public API change
- `0.1.x → 0.2.0` — new features, backwards compatible
- `x.0.0` — breaking changes (must be called out loudly in the release notes)

## Checklist

```bash
# 1. Everything green — never release with a red suite
pytest

# 2. Bump the version (single source of truth)
$EDITOR pyproject.toml          # version = "0.1.1"

# 3. Clean rebuild and validate
rm -rf dist
python -m build
twine check dist/*

# 4. Commit and push the bump
git add pyproject.toml
git commit -m "chore(release): v0.1.1"
git push

# 5. Upload to PyPI (credentials in ~/.pypirc, user __token__)
twine upload dist/*

# 6. Tag the exact commit that was published
git tag v0.1.1
git push origin v0.1.1

# 7. Verify what users will actually get, from a clean environment
python -m venv /tmp/verify-personos
/tmp/verify-personos/bin/pip install -U personos
/tmp/verify-personos/bin/personos doctor
```

## Rules that bite if forgotten

- **PyPI versions are immutable.** You cannot re-upload or edit `0.1.1` once it
  is live — not even to fix the README. A bad release is handled by
  `yank` (`pypi.org` → manage → options) followed by a new version number.
- **Tag what you shipped.** The tag must point at the commit whose
  `pyproject.toml` carries that version.
- **Users upgrade with `pip install -U personos`.** No action is needed to
  notify them; PyPI is the channel.
- Commit messages and release notes are in English.
