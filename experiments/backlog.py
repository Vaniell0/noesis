#!/usr/bin/env python3
"""List `_meta`-stamped results that the hypotheses/ directory hasn't cited yet.

    python experiments/backlog.py

A result counts as "processed" once its own repo-relative path (e.g.
`experiments/_common/results/g1i_battery_gpu/mlp_ipc.json`) appears as a
literal substring somewhere under `hypotheses/*.md` — which is already
this project's existing citation convention (every reconciled number is
followed by a path to the file it came from). No new bookkeeping file,
no schema change to results or hypotheses: this is a pure checker over
two conventions that already exist (`_meta.hypothesis` stamped by
`experiments._common.results.save_result`, and file-path citations in
hypothesis prose/frontmatter).

Grouped by hypothesis ID so a run of this script is directly the queue
for a "fold results into the record" pass.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.results import _iter_meta_stamped

_HYPOTHESES_DIR = _REPO_ROOT / "hypotheses"


def _all_hypothesis_text() -> str:
    return "\n".join(p.read_text() for p in sorted(_HYPOTHESES_DIR.glob("*.md")))


def find_backlog(hyp_text: str, root: Path | None = None) -> dict[str, list[tuple[Path, dict]]]:
    root = root or (_REPO_ROOT / "experiments")
    by_hypothesis: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path, meta in sorted(_iter_meta_stamped(root)):
        try:
            rel = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            rel = str(path)
        if rel in hyp_text:
            continue
        for h in meta.get("hypothesis") or ["(untagged)"]:
            by_hypothesis[h].append((path, meta))
    return by_hypothesis


if __name__ == "__main__":
    hyp_text = _all_hypothesis_text()
    backlog = find_backlog(hyp_text)

    if not backlog:
        print("[backlog] nothing pending — every stamped result is cited under hypotheses/")
        sys.exit(0)

    total = sum(len(v) for v in backlog.values())
    print(f"[backlog] {total} result file(s) not yet cited under hypotheses/, across {len(backlog)} hypothesis tag(s)\n")

    for h in sorted(backlog, key=lambda k: (k == "(untagged)", k)):
        entries = backlog[h]
        print(f"## {h} ({len(entries)})")
        for path, meta in entries:
            rel = path.relative_to(_REPO_ROOT)
            summary = meta.get("summary") or {}
            summary_str = "; ".join(f"{k}={v}" for k, v in summary.items()) or f"status={meta.get('status', '?')}"
            print(f"  - {rel}  [{meta.get('date', '?')}]  {summary_str}")
        print()
