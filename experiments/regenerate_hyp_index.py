#!/usr/bin/env python3
"""Rebuild the auto-generated status table in hypotheses/README.md.

    python experiments/regenerate_hyp_index.py

Walks every hypotheses/H*.md (skips README.md itself), pulls each
file's structured record via bayes_lite.parse_records, and renders one
row per file that has one. Files with no record yet show up with an
explicit "no structured record" status rather than being silently
omitted — the point is to make the backlog of un-reviewed hypotheses
visible, not to hide it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.bayes_lite import parse_records, posterior

_HYP_DIR = _REPO_ROOT / "hypotheses"
_README = _HYP_DIR / "README.md"
_AUTO_MARKER = "<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python experiments/regenerate_hyp_index.py` -->"

_ID_ORDER_RE = re.compile(r"^H(\d+)([a-z]?)$")


def _sort_key(hid: str):
    m = _ID_ORDER_RE.match(hid)
    if not m:
        return (999, hid)
    return (int(m.group(1)), m.group(2))


def build_rows() -> list[tuple]:
    rows = []
    for path in sorted(_HYP_DIR.glob("*.md")):
        if path.name in ("README.md",):
            continue
        text = path.read_text()
        records = parse_records(text)
        if not records:
            # no structured record yet -- still list it, explicitly
            stem = path.stem
            if _ID_ORDER_RE.match(stem):
                rows.append((stem, "no structured record", "-", "-", path.name))
            continue
        for record in records:
            hid = record.get("id", path.stem)
            status = record.get("status", "-")
            prior = record.get("prior")
            post = posterior(prior, record.get("evidence"))
            post_str = "-" if post is None else f"{post:.3f}"
            rows.append((hid, status, prior if prior is not None else "-", post_str, path.name))
    rows.sort(key=lambda r: _sort_key(r[0]))
    return rows


if __name__ == "__main__":
    rows = build_rows()

    table = ["| H | Status | Prior | Posterior | File |", "|---|--------|-------|-----------|------|"]
    for hid, status, prior, post, fname in rows:
        table.append(f"| {hid} | {status} | {prior} | {post} | `{fname}` |")

    text = _README.read_text()
    if _AUTO_MARKER not in text:
        print(f"[regenerate_hyp_index] marker not found in {_README}, appending", file=sys.stderr)
        text = text.rstrip() + "\n\n" + _AUTO_MARKER + "\n"
    head = text.split(_AUTO_MARKER)[0].rstrip() + "\n\n"
    _README.write_text(head + _AUTO_MARKER + "\n\n" + "\n".join(table) + "\n")
    print(f"[regenerate_hyp_index] wrote {len(rows)} row(s) to {_README.relative_to(_REPO_ROOT)}")
