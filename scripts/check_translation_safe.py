"""Prove a translation commit changed only comments and docstrings.

Translating ~5000 lines of commentary is mechanical, repetitive, and therefore
exactly the kind of work where a stray edit slips through unnoticed. Two things
make that dangerous here:

1. **Prompt text is not commentary.** Constants like ``ATOM_TEXT_SPEC`` and the
   system prompts are English strings that the models read. Editing one changes
   what the memory pipeline produces, with no test failure to show for it.
2. A diff of several thousand lines is not reviewable by eye.

So the check is structural: parse the file before and after, strip docstrings,
and compare the remaining AST. If anything but comments and docstrings moved,
the trees differ and this fails — regardless of how large the diff is.

Usage:
    python scripts/check_translation_safe.py <git-ref>    # default: HEAD~1
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Constants whose *value* is read by a model. Renaming or rewording these is a
# behaviour change wearing a translation's clothes. Listed explicitly so the
# check can say so by name rather than just "the AST differs".
PROMPT_CONSTANTS = {
    "ATOM_TEXT_SPEC", "RERANK_INSTRUCTION", "CONFLICT_RULE",
    "_REWRITE_SYS", "_ANSWER_SYS", "_MOST_RECENT_RULE",
    "_SCEN_HEADER", "_SCEN_DIR_REWRITE", "_SCEN_DIR_ANSWER",
    "_SYSTEM", "_HEADER", "_LOOK_SYSTEM", "PROMPT_HEADER",
}


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstrings so that rewriting them is invisible to the comparison.

    Comments never reach the AST at all, so they need no handling.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _normalise(source: str) -> str:
    return ast.dump(_strip_docstrings(ast.parse(source)), annotate_fields=True)


def _prompt_values(source: str) -> dict[str, str]:
    """Module-level assignments of listed prompt constants, by name."""
    out: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id in PROMPT_CONSTANTS and node.value:
                out[t.id] = ast.dump(node.value)
    return out


def _at_ref(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?", default="HEAD~1")
    args = ap.parse_args()

    changed = subprocess.run(
        ["git", "diff", "--name-only", args.ref, "--", "*.py"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not changed:
        print("no Python files changed")
        return 0

    problems: list[str] = []
    checked = 0
    for rel in changed:
        current = ROOT / rel
        before = _at_ref(args.ref, rel)
        if before is None or not current.exists():
            continue                      # added or deleted: nothing to compare
        after = current.read_text(encoding="utf-8")
        try:
            if _normalise(before) != _normalise(after):
                problems.append(f"{rel}: code changed, not just comments and docstrings")
        except SyntaxError as e:
            problems.append(f"{rel}: could not parse ({e})")
            continue

        before_prompts, after_prompts = _prompt_values(before), _prompt_values(after)
        for name, value in before_prompts.items():
            if after_prompts.get(name) != value:
                problems.append(
                    f"{rel}: prompt constant {name} changed — this is model input, "
                    f"not commentary; changing it changes what the pipeline produces")
        checked += 1

    for p in problems:
        print(f"FAIL  {p}")
    print(f"\nTRANSLATION_SAFE {'PASS' if not problems else 'FAIL'} "
          f"({checked} files compared against {args.ref})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
