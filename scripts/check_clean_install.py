"""Install the core into an empty virtualenv and prove it is actually usable.

Declaring a small dependency list is easy; keeping it true is not. A single
module-level import of an optional package turns an extra into a hard
requirement, and the failure only appears for the user who installed the
minimal set — never for a developer whose environment has everything.

So this builds a throwaway venv, installs the core with no extras, and checks:
  1. `from personos import Memory` works
  2. `Memory()` constructs and reports its capabilities
  3. none of the heavy optional packages got pulled in

Usage: python scripts/check_clean_install.py [--keep]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Packages that must NOT arrive with a bare install. Each is here because it is
# big, or because it commits the user to a stack they may not want.
FORBIDDEN = ["torch", "langchain", "langchain-classic", "fastapi", "uvicorn",
             "pymysql", "redis", "oss2", "onnxruntime", "insightface", "speechbrain"]

PROBE = """
import sys
from personos import Memory

with Memory() as m:
    caps = m.capabilities()
    assert caps, "capabilities() returned nothing"
    required = [c for c in caps if c.required]
    assert required, "no required capabilities are declared"
    storage = next(c for c in caps if c.name == "storage")
    assert storage.available, "storage should work out of the box"
    assert "SQLite" in storage.detail, storage.detail
print("PROBE_OK")
"""


def _run(*cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="do not delete the venv")
    args = ap.parse_args()

    venv = Path(tempfile.mkdtemp(prefix="personos-clean-"))
    data = venv / "data"
    py = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    ok = True
    try:
        print(f"building a clean venv at {venv}")
        _run(sys.executable, "-m", "venv", str(venv), check=True)

        r = _run(str(py), "-m", "pip", "install", "-q", str(ROOT))
        if r.returncode != 0:
            print(f"FAIL: install failed\n{r.stderr[-1500:]}")
            return 1

        installed = _run(str(py), "-m", "pip", "list", "--format=freeze").stdout.lower()
        leaked = [p for p in FORBIDDEN if f"\n{p}==" in "\n" + installed]
        if leaked:
            ok = False
            print(f"FAIL: a bare install pulled in optional packages: {leaked}\n"
                  f"      something imports them at module level, which makes an "
                  f"extra mandatory.")
        else:
            print(f"ok: none of {len(FORBIDDEN)} optional packages were installed")

        # An empty environment except for the data directory, so the probe
        # cannot accidentally pick up the developer's configuration.
        env = {"PATH": "/usr/bin:/bin", "PERSONOS_DATA_DIR": str(data), "HOME": str(venv)}
        r = _run(str(py), "-c", PROBE, env=env, cwd=str(venv))
        if "PROBE_OK" in r.stdout:
            print("ok: `from personos import Memory` works with no configuration")
        else:
            ok = False
            print(f"FAIL: the probe did not complete\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")
    finally:
        if args.keep:
            print(f"venv kept at {venv}")
        else:
            shutil.rmtree(venv, ignore_errors=True)

    print("\nCLEAN_INSTALL " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
