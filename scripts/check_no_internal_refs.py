"""Release gate: no internal company identifiers may survive in the repository.

Why automate it: internal identifiers do not only hide in obvious config. They
are scattered through docstrings ("ported from <internal project>"), comments
("behind the <internal> proxy"), dependency names, environment-variable
prefixes, and the internal hostnames baked into exception messages. A human
review will miss some, and publishing is **irreversible**: once it is pushed
and crawled, it cannot be taken back.

Usage:
    python scripts/check_no_internal_refs.py            # report hits; exit 1 if any
    python scripts/check_no_internal_refs.py --list     # group by pattern (use as a worklist)

Meant to live as a pre-commit hook. It stays red for the duration of an
open-sourcing effort — that is expected, and the red entries are the worklist.
It must be green before release.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pattern -> why it must not appear. Spell out the reason so that nobody later
# mistakes this for fussiness and adds a casual exemption.
PATTERNS: dict[str, str] = {
    r"xiaohongshu": "company domain",
    r"\bxhs\b|XHS_": "company abbreviation / env prefix",
    r"redkms|redenv|redmetrics|redinfra|python-infra-framework|\bhyx\b":
        "internal private packages (not installable from the public index)",
    r"RedHub|redhub": "internal database proxy",
    r"\bcorvus\b": "internal Redis cluster implementation",
    r"\bmaas\b|MAAS_": "internal model gateway",
    r"\bapollo\b|pyapollo": "internal configuration service",
    r"xray-langfuse|xray": "internal observability platform",
    r"mneme|meme-backend|wallace": "internal project codenames",
    r"DMS 工单|内部系统开发安全规范": "internal process / policy document names",
}

# Skipped: binaries, build artefacts, and this file itself (which of course
# contains every one of these words).
# Note "baseline" is NOT skipped by name: that would also skip scripts/baseline,
# the behaviour-baseline tooling, which does ship and does need scanning. Only
# the recorded artefacts at the repository root are excluded, and those are
# gitignored anyway so they never reach the file list.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "logs",
             ".venv", "venv", "node_modules", ".idea"}
SKIP_SUFFIX = {".pyc", ".gz", ".png", ".jpg", ".jpeg", ".ico", ".lock", ".json"}
SKIP_FILES = {"check_no_internal_refs.py"}


def _files() -> list[Path]:
    """Only files git actually tracks.

    What ships is what is committed. Scanning the working tree instead would
    flag a developer's local .env — noise that trains people to ignore this
    check — while *missing* nothing, since anything untracked cannot leak.
    If git is unavailable, fall back to walking the tree.
    """
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                                 capture_output=True, text=True, check=True)
        candidates = [ROOT / n for n in listing.stdout.split("\0") if n]
    except (subprocess.CalledProcessError, FileNotFoundError):
        candidates = list(ROOT.rglob("*"))
    out = []
    for p in candidates:
        if not p.is_file() or p.name in SKIP_FILES or p.suffix in SKIP_SUFFIX:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def scan() -> dict[str, list[tuple[Path, int, str]]]:
    hits: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    compiled = {pat: re.compile(pat, re.IGNORECASE) for pat in PATTERNS}
    for f in _files():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat, rx in compiled.items():
                if rx.search(line):
                    hits[pat].append((f.relative_to(ROOT), i, line.strip()[:110]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="list every hit (use it as a worklist)")
    a = ap.parse_args()

    hits = scan()
    if not hits:
        print("PASS: no internal identifiers remain")
        return 0

    total = sum(len(v) for v in hits.values())
    files = {f for v in hits.values() for f, _, _ in v}
    print(f"FAIL: {total} internal identifiers across {len(files)} files\n")
    for pat, items in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        by_file: dict[Path, int] = defaultdict(int)
        for f, _, _ in items:
            by_file[f] += 1
        print(f"-- {PATTERNS[pat]}  ({len(items)} hits / {len(by_file)} files)  /{pat}/")
        for f, n in sorted(by_file.items(), key=lambda kv: -kv[1])[:8 if not a.list else 999]:
            print(f"     {n:>4}  {f}")
        if not a.list and len(by_file) > 8:
            print(f"          ... and {len(by_file) - 8} more files (--list to see all)")
        if a.list:
            for f, i, line in items[:40]:
                print(f"          {f}:{i}  {line}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
