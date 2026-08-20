#!/usr/bin/env python3
"""Read every hypothesis's structured record (YAML embedded in an HTML
comment above its `## H<N>.` prose section) and:

  1. compute a rough log-odds posterior from its `evidence` list, purely
     as a forcing function — the point is that filling in `prior`,
     `depends_on`, and especially `contradicts_if` makes you commit to
     something checkable, not that the arithmetic is trustworthy. Never
     written back to the file; just printed next to the hand-set
     `status` for comparison.
  2. lint the records: dangling H-id references in `depends_on` /
     `supports` / `contradicts_if`; a hypothesis with no `contradicts_if`
     at all (a hypothesis with no stated refutation condition is badly
     formed regardless of what the posterior says); `status` whose
     polarity disagrees with the computed posterior.

    python experiments/bayes_lite.py [FILE ...]     # defaults to hypotheses/*.md

A record block looks like:

    <!--
    (free-text preamble, ignored)
    id: H8
    status: SUPPORTED
    prior: 0.6
    ...
    -->

Parsing finds `id:` inside each `<!-- -->` block and YAML-parses from
there to the end of the comment — the preamble before it can be
arbitrary prose.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HYPOTHESES_DIR = _REPO_ROOT / "hypotheses"

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_ID_START_RE = re.compile(r"^id:\s*\S+", re.MULTILINE)
_HEADER_ID_RE = re.compile(r"^##\s+(H\d+[a-z]?)\.", re.MULTILINE)
_ANY_ID_RE = re.compile(r"\bH\d{1,2}[a-z]?\b")

# call -> likelihood ratio, by strength. Deliberately coarse.
_LR = {
    ("supports", "strong"): 3.0,
    ("supports", "weak"): 1.3,
    ("neutral", "strong"): 1.0,
    ("neutral", "weak"): 1.0,
    ("methodology", "strong"): 1.0,
    ("methodology", "weak"): 1.0,
    ("contradicts", "weak"): 1 / 1.3,
    ("contradicts", "strong"): 1 / 3.0,
}


def parse_records(text: str) -> list[dict]:
    records = []
    for m in _COMMENT_RE.finditer(text):
        block = m.group(1)
        id_m = _ID_START_RE.search(block)
        if not id_m:
            continue
        yaml_text = block[id_m.start():]
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            print(f"[bayes_lite] WARNING: failed to parse block near {id_m.start()}: {e}", file=sys.stderr)
            continue
        if isinstance(data, dict) and "id" in data:
            records.append(data)
    return records


def posterior(prior: float, evidence: list[dict]) -> float | None:
    if prior is None:
        return None
    logit = math.log(prior / (1 - prior))
    for e in evidence or []:
        lr = _LR.get((e.get("call"), e.get("strength")))
        if lr is None:
            continue
        logit += math.log(lr)
    return 1 / (1 + math.exp(-logit))


def lint(record: dict, known_ids: set[str]) -> list[str]:
    problems = []
    hid = record.get("id", "?")

    for field in ("depends_on", "supports"):
        for ref in record.get(field) or []:
            if ref not in known_ids:
                problems.append(f"{field} references unknown id {ref!r}")

    contradicts_if = record.get("contradicts_if") or []
    if not contradicts_if:
        problems.append("no contradicts_if — hypothesis has no stated refutation condition")

    ev = record.get("evidence") or []
    # File existence / citation coverage is backlog.py's job, not duplicated here.

    status = (record.get("status") or "").upper()
    post = posterior(record.get("prior"), ev)
    if post is not None and status in ("SUPPORTED", "REFUTED"):
        implied_support = post >= 0.5
        status_support = status == "SUPPORTED"
        if implied_support != status_support and abs(post - 0.5) > 0.1:
            problems.append(
                f"status={status} but evidence-only posterior={post:.2f} "
                f"disagrees in direction (status likely rests on evidence not in this log, or evidence list is stale)"
            )

    return problems


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or sorted(_HYPOTHESES_DIR.glob("*.md"))

    all_text = ""
    for f in files:
        all_text += f.read_text() + "\n"
    known_ids = set(_HEADER_ID_RE.findall(all_text)) | set(_ANY_ID_RE.findall(all_text))

    any_records = False
    for f in files:
        text = f.read_text()
        for record in parse_records(text):
            any_records = True
            hid = record["id"]
            prior = record.get("prior")
            post = posterior(prior, record.get("evidence"))
            status = record.get("status", "?")
            n_ev = len(record.get("evidence") or [])
            n_contra = sum(1 for e in (record.get("evidence") or []) if e.get("call") == "contradicts")

            print(f"=== {hid} ({f.name}) ===")
            print(f"  status={status}  prior={prior}  posterior={'n/a' if post is None else round(post, 3)}  "
                  f"evidence={n_ev} ({n_contra} contradicts)")
            problems = lint(record, known_ids)
            for p in problems:
                print(f"  LINT: {p}")
            print()

    if not any_records:
        print(f"[bayes_lite] no structured records found in {', '.join(str(f) for f in files)}")
