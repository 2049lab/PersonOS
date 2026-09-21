"""Cold-start gate: importing must succeed in an empty environment, with zero
configuration supplied.

This is the only hard guarantee behind the "works with no configuration"
promise. What it catches is **evaluation at import time** — a module that reads
a secret, connects to a database, or builds a client while it is being
imported. If any module does that, the very first line after
`pip install personos`, `from personos import Memory`, blows up, and no amount
of README polish will save it.

Why testing `import personos` alone is not enough: the package `__init__.py`
may be nearly empty, in which case the check is permanently green and catches
nothing (observed in practice — `import personos` passed in an empty
environment while `import personos.online.write_path` raised "missing API
key"). So representative modules are listed **explicitly**, covering the
configuration, storage, write and recall paths.

Usage: python scripts/check_cold_start.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Representative modules: each covers one class of import-time risk. Add an
# entry here whenever a new subsystem appears.
MODULES = [
    ("personos", "package entry point (the Memory facade ends up here)"),
    ("personos.config", "configuration layer - where import-time evaluation usually hides"),
    ("personos.models", "pure data models; these must import under any conditions"),
    ("personos.storage.db", "storage layer: must not connect to a database at import time"),
    ("personos.online.write_path", "write path"),
    ("personos.online.retrieval", "recall path"),
]


def main() -> int:
    py = sys.executable
    # The equivalent of `env -i`: keep only what the interpreter needs to start,
    # and clear everything else.
    clean = {"PATH": os.environ.get("PATH", ""), "HOME": "/tmp/personos-coldstart"}
    if sys.platform == "win32":
        clean["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

    failed = []
    for mod, why in MODULES:
        r = subprocess.run([py, "-c", f"import {mod}"], env=clean, cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"ok   {mod}")
        else:
            tail = (r.stderr or "").strip().splitlines()
            print(f"FAIL {mod}  ({why})\n     {tail[-1] if tail else 'unknown error'}")
            failed.append(mod)

    if failed:
        print(f"\nCOLD_START FAIL - {len(failed)}/{len(MODULES)} modules cannot be "
              f"imported in an empty environment."
              f"\nThe zero-configuration promise does not hold: these modules demand "
              f"configuration while being imported."
              f"\nUsual causes: a module-level `settings = load_settings()`, settings "
              f"used as a default argument, or a module-level client singleton.")
        return 1
    print(f"\nCOLD_START PASS - all {len(MODULES)} modules import in an empty environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
