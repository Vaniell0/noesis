#!/usr/bin/env python3
"""List `_meta`-stamped results that the hypotheses/ directory hasn't cited yet.

    python experiments/backlog.py            # by hypothesis and by the folder that holds the run
    python experiments/backlog.py --files    # every file

A result counts as "cited" once a link under hypotheses/ names it: its
own path from the repository root, the folder that holds its run, or a
glob that matches it (what a link is, is in `experiments/bayes_lite.py`).
Links are read from the prose and from the records on every run. No new
bookkeeping file, no schema change to results or hypotheses: this is a
pure checker over two conventions that already exist (`_meta.hypothesis`
stamped by `experiments._common.results.save_result`, and links in the
hypotheses).

Grouped by hypothesis ID so a run of this script is directly the queue
for a "fold results into the record" pass.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.results import _iter_meta_stamped
from experiments.bayes_lite import cited_paths, is_cited, run_folders

_RESULTS_ROOT = _REPO_ROOT / "experiments"


def find_backlog(cited: set[str], root: Path | None = None) -> dict[str, list[tuple[str, dict]]]:
    """Stamped results under root that no cited path names, by hypothesis tag: (path from the root, meta)."""
    root = root or _RESULTS_ROOT
    by_hypothesis: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for path, meta in sorted(_iter_meta_stamped(root)):
        try:
            rel = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            rel = str(path)
        if is_cited(rel, cited):
            continue
        for h in meta.get("hypothesis") or ["(untagged)"]:
            by_hypothesis[h].append((rel, meta))
    return by_hypothesis


if __name__ == "__main__":
    stamped = _iter_meta_stamped(_RESULTS_ROOT)
    backlog = find_backlog(cited_paths())
    uncited = {rel for entries in backlog.values() for rel, _ in entries}

    print(f"[backlog] {len(stamped)} stamped result file(s) under experiments/: {len(stamped) - len(uncited)} "
          f"cited by a link under hypotheses/, {len(uncited)} not\n")
    if not backlog:
        sys.exit(0)

    for h in sorted(backlog, key=lambda k: (k == "(untagged)", k)):
        entries = backlog[h]
        print(f"## {h} ({len(entries)})")
        held = Counter()
        for rel, _ in entries:
            folders = run_folders(rel)
            held[(folders[1], True) if len(folders) > 1 else (rel, False)] += 1
        for (place, is_folder), n in sorted(held.items()):
            print(f"  {n:4}  {place}" + ("/" if is_folder else ""))
        if "--files" in sys.argv:
            for rel, meta in entries:
                summary = meta.get("summary") or {}
                summary_str = "; ".join(f"{k}={v}" for k, v in summary.items()) or f"status={meta.get('status', '?')}"
                print(f"        {rel}  [{meta.get('date', '?')}]  {summary_str}")
        print()
