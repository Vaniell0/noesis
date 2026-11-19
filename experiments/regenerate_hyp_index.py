#!/usr/bin/env python3
"""Rebuild the auto-generated status table in hypotheses/README.md.

    python experiments/regenerate_hyp_index.py           # README table and the ZLK claims table in HZLK.md
    python experiments/regenerate_hyp_index.py --check   # only report broken ZLK links and missing cited files

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
        return (-1, hid)  # unnumbered programme hypotheses (ZLK) sit above H0
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


_HZLK = _HYP_DIR / "HZLK.md"
_CLAIMS_BEGIN = "<!-- ZLK TABLE: generated from the claims records above by experiments/regenerate_hyp_index.py -->"
_CLAIMS_END = "<!-- END ZLK TABLE -->"
_PATH_RE = re.compile(r"`([\w./~-]+/[\w.-]+\.(?:json|md|py|jsonl|pth|pt))`")
_REF_RE = re.compile(r"\b(ZLK\.\d+|H\d{1,2}[a-z]?)\b")


def claim_sections(text: str) -> dict[str, str]:
    """The prose of every '**ZLK.n Title.**' paragraph in §4, up to the next claim or heading."""
    out = {}
    for m in re.finditer(r"^\*\*(ZLK\.\d+)\b.*?(?=^\*\*ZLK\.\d+\b|^#{2,4} )", text, re.S | re.M):
        out[m.group(1)] = m.group(0)
    return out


def claims_report() -> tuple[list[str], list[str]]:
    """Table rows for the ZLK claims and the problems found: cited files that do not exist, references to
    ids that exist nowhere. Files and links are read from each claim's own §4 prose every run, never stored."""
    text = _HZLK.read_text()
    recs = next((r for r in parse_records(text) if r.get("id") == "HZLK"), {}).get("claims") or []
    if not recs:                                                # a broken record must not read as "all clear"
        return [], ["no ZLK claims records parsed from HZLK.md (YAML error above, or the claims list is gone)"]
    secs = claim_sections(text)
    known = {r["id"] for r in recs} | {p.stem for p in _HYP_DIR.glob("H*.md")} | set(re.findall(
        r"^id:\s*(H\d+[a-z]?)", "\n".join(p.read_text() for p in _HYP_DIR.glob("H*.md")), re.M))
    elsewhere = {}
    for p in sorted(_HYP_DIR.glob("H*.md")):
        if p != _HZLK:
            for zid in set(re.findall(r"\bZLK\.\d+\b", p.read_text())):
                elsewhere.setdefault(zid, []).append(p.stem)
    rows = ["| id | statement (short) | status | regime | evidence files | linked |", "|---|---|---|---|---|---|"]
    problems = []
    for r in recs:
        sec = secs.get(r["id"], "")
        if not sec:
            problems.append(f"{r['id']}: no '**{r['id']} ...**' paragraph in §4")
        files = sorted(set(_PATH_RE.findall(sec)))
        missing = [f for f in files if not ((_REPO_ROOT / f).exists() or (_REPO_ROOT / "experiments" / f).exists()
                                            or Path(f).expanduser().exists())]
        problems += [f"{r['id']}: cited file not found: {f}" for f in missing]
        refs = sorted({x for x in _REF_RE.findall(sec) if x != r["id"]}, key=lambda x: (x[0], x))
        problems += [f"{r['id']}: reference to unknown id {x}" for x in refs if x not in known]
        if r.get("status") and not r.get("regime") and not r["status"].startswith(("proposed", "not run", "not started")):
            problems.append(f"{r['id']}: a measured status without a regime (data and model state)")
        links = ", ".join(refs + [f"<-{h}" for h in elsewhere.get(r["id"], [])]) or "-"
        ev = f"{len(files)}" + (f" ({len(missing)} missing)" if missing else "")
        rows.append(f"| {r['id']} | {r['claim']} | {r['status']} | {r.get('regime') or '-'} | {ev} | {links} |")
    return rows, problems


def write_claims_table(rows: list[str]) -> None:
    text = _HZLK.read_text()
    if _CLAIMS_BEGIN not in text:
        print(f"[regenerate_hyp_index] ZLK table markers not found in {_HZLK.name}", file=sys.stderr)
        return
    head, rest = text.split(_CLAIMS_BEGIN, 1)
    tail = rest.split(_CLAIMS_END, 1)[1]
    _HZLK.write_text(head + _CLAIMS_BEGIN + "\n\n" + "\n".join(rows) + "\n\n" + _CLAIMS_END + tail)


if __name__ == "__main__":
    claim_rows, claim_problems = claims_report()
    for p in claim_problems:
        print(f"[regenerate_hyp_index] CLAIMS: {p}")
    if "--check" in sys.argv:                                   # read only: exit 1 on any problem
        sys.exit(1 if claim_problems else 0)
    if not claim_rows:                                          # unreadable records: leave every table as it is
        sys.exit(1)
    write_claims_table(claim_rows)
    print(f"[regenerate_hyp_index] wrote {len(claim_rows) - 2} ZLK claim row(s) to hypotheses/HZLK.md, "
          f"{len(claim_problems)} problem(s)")
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
