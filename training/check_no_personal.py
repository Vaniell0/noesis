#!/usr/bin/env python3
"""training/check_no_personal.py — refuse a commit that would put the owner's own session data in git.

Called by .git/hooks/pre-commit (hooks are local and untracked; the hook only runs this file, so the
rules live here and are versioned. A fresh clone needs the two-line hook re-created).

Blocks any staged path that is
  * under training/corpus/ or training/sanitised/ (raw / sanitised Claude CLI traces), except the
    converter script training/corpus/convert_anthropic_to_dsl.py and RECLASSIFIED.md;
  * the artifact of a catalog record marked personal (or derived from one) — action_chains*,
    step6_mixed_*, step9_combined_*, step9b_combined_*;
  * a data file (.jsonl / .pt) under training/corpus_open/ or training/tokenised/ at all (they are
    gitignored; a forced add is the only way they get here).

    training/.venv/bin/python training/check_no_personal.py            # checks the staged files
    training/.venv/bin/python training/check_no_personal.py --paths a b   # checks given paths
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ALLOWED = {"training/corpus/convert_anthropic_to_dsl.py", "training/corpus/RECLASSIFIED.md"}
BLOCK_DIRS = ("training/corpus/", "training/sanitised/")
DATA_DIRS = ("training/corpus_open/", "training/tokenised/")
DATA_SUFFIXES = (".jsonl", ".pt", ".json")


def personal_artifacts() -> set[str]:
    try:
        from training._common import catalog
        return {r.out_path for r in catalog.all_records() if r.out_path and catalog.is_personal(r.name)}
    except Exception:  # catalog unreadable: fall back to the directory rules only
        return set()


def check(paths: list[str]) -> list[str]:
    personal = personal_artifacts()
    bad = []
    for p in paths:
        if p in ALLOWED:
            continue
        if p.startswith(BLOCK_DIRS):
            bad.append(f"{p}: raw/sanitised session traces never go to git")
        elif p in personal:
            bad.append(f"{p}: catalogued as personal (derived from the owner's own sessions)")
        elif p.startswith(DATA_DIRS) and p.endswith(DATA_SUFFIXES) and not p.endswith("PROVENANCE.md"):
            bad.append(f"{p}: corpus data is gitignored; forced add refused (catalog holds its record)")
    return bad


def main() -> int:
    if "--paths" in sys.argv:
        paths = sys.argv[sys.argv.index("--paths") + 1:]
    else:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout
        paths = [l for l in out.splitlines() if l]
    bad = check(paths)
    if bad:
        print("commit refused — personal or corpus data staged:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
