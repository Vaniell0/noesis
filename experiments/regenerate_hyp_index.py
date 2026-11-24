#!/usr/bin/env python3
"""Rebuild the auto-generated status table in hypotheses/README.md.

    python experiments/regenerate_hyp_index.py           # README table and the ZLK claims table in HZLK.md
    python experiments/regenerate_hyp_index.py --check   # read only: every link under hypotheses/, alive or dead

Walks every hypotheses/H*.md (skips README.md itself), pulls each
file's structured record via bayes_lite.load_records (`records/<ID>.yml`
when the file has one, else its comment block), and renders one row per
file that has a record. Files with no record yet show up with an
explicit "no structured record" status rather than being silently
omitted — the point is to make the backlog of un-reviewed hypotheses
visible, not to hide it.

`--check` resolves every link under hypotheses/ (what a link is, is in
`experiments/bayes_lite.py`): a path from the repository root, a path on
this machine outside the repository, or a glob over a family of runs. It
prints how many are alive and names every dead one, with the place the
data lies when the tree holds it under one other path. It also names the
lines over 1000 characters. Exit 1 on any dead link, long line or claims
problem. The same check runs after a regeneration; a dead link is then
one WARNING line.
"""
from __future__ import annotations

import collections
import functools
import glob
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.bayes_lite import (
    _NO_REFUTER_NEEDED, expand_link, hypothesis_links, links_in, load_records, missing_paths, posterior)

_HYP_DIR = _REPO_ROOT / "hypotheses"
_README = _HYP_DIR / "README.md"
_AUTO_MARKER = "<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python experiments/regenerate_hyp_index.py` -->"
_LINE_CAP = 1000

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
        records, _ = load_records(path)
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
_CLAIMS_BEGIN = "<!-- ZLK TABLE: generated from the claims in hypotheses/records/HZLK.yml by experiments/regenerate_hyp_index.py -->"
_CLAIMS_END = "<!-- END ZLK TABLE -->"
_REF_RE = re.compile(r"\b(ZLK\.\d+|H\d{1,2}[a-z]?)\b")


def claim_sections(text: str) -> dict[str, str]:
    """The prose of every '**ZLK.n Title.**' paragraph in §4, up to the next claim or heading."""
    out = {}
    for m in re.finditer(r"^\*\*(ZLK\.\d+)\b.*?(?=^\*\*ZLK\.\d+\b|^#{2,4} )", text, re.S | re.M):
        out[m.group(1)] = m.group(0)
    return out


_CLAIM_STATUSES = {"PROPOSED", "BLOCKED", "REGULAR", "PARTIAL", "CONTESTED", "SUPPORTED", "REFUTED"}


def claim_groups(text: str) -> dict[str, str]:
    """The '#### A. Title' heading each claim of §4 sits under: the groups of the graph."""
    out, title = {}, ""
    for line in text.split("\n"):
        m = re.match(r"^#{2,4} (.*)", line)
        if m:
            title = m.group(1).strip()
        m = re.match(r"^\*\*(ZLK\.\d+)\b", line)
        if m:
            out[m.group(1)] = title
    return out


def claims_report() -> tuple[list[str], list[str]]:
    """Table rows for the ZLK claims and the problems found: links of a claim's §4 prose that resolve from
    no root, references to ids that exist nowhere. Files and links are read from each claim's own §4 prose
    every run, never stored."""
    records, _ = load_records(_HZLK)
    recs = next((r for r in records if r.get("id") == "HZLK"), {}).get("claims") or []
    if not recs:                                                # a broken record must not read as "all clear"
        return [], ["no ZLK claims records parsed from HZLK.md (YAML error above, or the claims list is gone)"]
    secs = claim_sections(_HZLK.read_text())
    known = {r["id"] for r in recs} | {p.stem for p in _HYP_DIR.glob("H*.md")} | {
        r.get("id") for p in _HYP_DIR.glob("H*.md") for r in load_records(p)[0]}
    elsewhere = {}
    for p in sorted(_HYP_DIR.glob("H*.md")):
        if p != _HZLK:
            for zid in set(re.findall(r"\bZLK\.\d+\b", p.read_text())):
                elsewhere.setdefault(zid, []).append(p.stem)
    rows = ["| id | statement (short) | status | regime | data links | builds on | linked |", "|---|---|---|---|---|---|---|"]
    problems = []
    groups = claim_groups(_HZLK.read_text())
    counts = collections.Counter((r.get("status") or "").upper() for r in recs)
    rows.append("| " + f"{len(recs)} claims: " + ", ".join(f"{n} {s.lower()}" for s, n in sorted(counts.items(), key=lambda x: -x[1]))
                + " | | | | | | |")
    nums = sorted(int(x["id"].split(".")[1]) for x in recs)
    gaps = [n for n in range(1, max(nums) + 1) if n not in nums]
    if gaps:
        why = next((x.get("unused") or {} for x in load_records(_HZLK)[0] if x.get("id") == "HZLK"), {})
        parts = [f"{n}: {why[str(n)]}" if why.get(str(n)) else str(n) for n in gaps]
        rows.append(f"| numbering, up to {max(nums)}: {len(gaps)} not used ({'; '.join(parts)}); a number is "
                    "never reused | | | | | | |")
    last = None
    for r in sorted(recs, key=lambda x: list(secs).index(x["id"]) if x["id"] in secs else len(secs)):
        sec = secs.get(r["id"], "")
        status = (r.get("status") or "").upper()
        if status not in _CLAIM_STATUSES:
            problems.append(f"{r['id']}: status '{r.get('status')}' is not one of {', '.join(sorted(_CLAIM_STATUSES))}")
        if status not in ("PROPOSED", "BLOCKED", "") and not r.get("regime"):
            problems.append(f"{r['id']}: a {status} claim without a regime (data and model state)")
        for a in r.get("after") or []:
            if a not in {x["id"] for x in recs}:
                problems.append(f"{r['id']}: 'after' names unknown id {a}")
        g = groups.get(r["id"], "")
        if g != last:
            rows.append(f"| **{g}** | | | | | | |" if g else "| **(no group)** | | | | | | |")
            last = g
        if not sec:
            problems.append(f"{r['id']}: no '**{r['id']} ...**' paragraph in §4")
        found = sorted(set(links_in(sec)))
        dead = [(tok, miss) for tok, kind in found for miss in [missing_paths(tok, kind)] if miss]
        problems += [f"{r['id']}: link resolves from no root: {tok}"
                     + (f" ({len(miss)} of {len(expand_link(tok))} paths missing)" if len(miss) > 1 else "")
                     for tok, miss in dead]
        data = [tok for tok, _ in found if not tok.startswith(("hypotheses/", "docs/"))]   # another file is a reference
        refs = sorted({x for x in _REF_RE.findall(sec) if x != r["id"]}, key=lambda x: (x[0], x))
        problems += [f"{r['id']}: reference to unknown id {x}" for x in refs if x not in known]
        links = ", ".join(refs + [f"<-{h}" for h in elsewhere.get(r["id"], [])]) or "-"
        after = ", ".join(f"**{a}**" for a in r.get("after") or []) or "-"
        ev = f"{len(data)}" + (f" ({len(dead)} dead)" if dead else "")
        rows.append(f"| {r['id']} | {r['claim']} | {status} | {r.get('regime') or '-'} | {ev} | {after} | {links} |")
    return rows, problems


@functools.lru_cache(maxsize=None)
def _tree_suffixes() -> dict[str, list[str]]:
    """Every path of the tree under each of its suffixes of two to four segments."""
    out: dict[str, list[str]] = {}
    for d, ds, fs in os.walk(_REPO_ROOT):
        ds[:] = [x for x in ds if x not in (".git", "__pycache__", "node_modules") and not x.startswith(".venv")]
        for name in fs + ds:
            parts = os.path.relpath(os.path.join(d, name), _REPO_ROOT).split("/")
            for k in range(max(1, len(parts) - 4), len(parts) - 1):
                out.setdefault("/".join(parts[k:]), []).append("/".join(parts))
    return out


def _where_it_lies(path: str) -> str:
    """The path at which data cited from another root lies in the tree now, when the tree holds it under
    exactly one. A glob is followed through the part of its path that has no wildcard."""
    if "*" not in path:
        hits = _tree_suffixes().get(path.rstrip("/"), [])
        return hits[0] if len(hits) == 1 else ""
    head = path.split("*")[0]
    cut = head.rfind("/")
    if cut < 0:
        return ""
    folder, pattern = head[:cut], path[cut + 1:]
    found = [(f"{c}/{pattern}", len(glob.glob(str(_REPO_ROOT / c / pattern), recursive=True)))
             for c in _tree_suffixes().get(folder, [])]
    found = [f for f in found if f[1]]
    return f"{found[0][0]} ({found[0][1]} files)" if len(found) == 1 else ""


def check_links() -> tuple[list[int], list[str]]:
    """([alive, dead, external, marked missing], one line per dead link under hypotheses/)."""
    counts, lines = [0, 0, 0, 0], []
    for name, found in sorted(hypothesis_links().items()):
        for tok, kind in sorted(found):
            missing = missing_paths(tok, kind)
            if kind in ("web", "missing"):
                counts[2 if kind == "web" else 3] += 1
            elif not missing:
                counts[0] += 1
            else:
                counts[1] += 1
                line = f"DEAD {name}: `{tok}`"
                if len(missing) > 1:
                    line += f" ({len(missing)} of {len(expand_link(tok))} paths missing)"
                lies = sorted({w for w in (_where_it_lies(p) for p in missing) if w})
                lines.append(line + (f" -> lies at {', '.join(lies)}" if lies else ""))
    return counts, lines


def check_lines() -> list[str]:
    """The `.md` files under hypotheses/ with a line over the cap."""
    out = []
    for p in sorted(_HYP_DIR.glob("*.md")):
        long = [i for i, l in enumerate(p.read_text().splitlines(), 1) if len(l) > _LINE_CAP]
        if long:
            out.append(f"LONG {p.name}: {len(long)} line(s) over {_LINE_CAP} characters, the first at {long[0]}")
    return out


def write_claims_table(rows: list[str]) -> None:
    text = _HZLK.read_text()
    if _CLAIMS_BEGIN not in text:
        print(f"[regenerate_hyp_index] ZLK table markers not found in {_HZLK.name}", file=sys.stderr)
        return
    head, rest = text.split(_CLAIMS_BEGIN, 1)
    tail = rest.split(_CLAIMS_END, 1)[1]
    _HZLK.write_text(head + _CLAIMS_BEGIN + "\n\n" + "\n".join(rows) + "\n\n" + _CLAIMS_END + tail)


def status_problems() -> list[str]:
    """A top-level status outside the vocabulary (README, rule 7). A word the scripts do not know must not pass
    as agreed."""
    allowed = _CLAIM_STATUSES | set(_NO_REFUTER_NEEDED) | {"WAGER"}
    out = []
    for p in sorted(_HYP_DIR.glob("H*.md")):
        for r in load_records(p)[0]:
            s = str(r.get("status") or "").upper()
            if s and s != "WAGER" and s not in allowed:
                out.append(f"{r.get('id', p.stem)}: status '{r.get('status')}' is not one of {', '.join(sorted(allowed))}")
    return out


if __name__ == "__main__":
    claim_rows, claim_problems = claims_report()
    stat_problems = status_problems()
    (alive, dead, external, marked), dead_lines = check_links()
    long_lines = check_lines()
    for p in claim_problems:
        print(f"[regenerate_hyp_index] CLAIMS: {p}")
    for p in stat_problems:
        print(f"[regenerate_hyp_index] STATUS: {p}")
    summary = (f"links under hypotheses/: {alive + dead + external + marked} distinct, {alive} alive, {dead} dead, "
               f"{external} external and not checked, {marked} marked missing; {len(long_lines)} file(s) with a line "
               f"over {_LINE_CAP} characters; {len(claim_problems)} claims problem(s), {len(stat_problems)} status problem(s)")
    if "--check" in sys.argv:                                   # read only: exit 1 on any problem
        for l in dead_lines + long_lines:
            print(f"[regenerate_hyp_index] {l}")
        print(f"[regenerate_hyp_index] {summary}")
        sys.exit(1 if claim_problems or stat_problems or dead or long_lines else 0)
    if dead or long_lines:
        print(f"[regenerate_hyp_index] WARNING: {dead} dead link(s), {len(long_lines)} file(s) with a long "
              "line under hypotheses/ (--check names them)")
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
    print(f"[regenerate_hyp_index] wrote {len(rows)} row(s) to {_README.relative_to(_REPO_ROOT)}; {summary}")
