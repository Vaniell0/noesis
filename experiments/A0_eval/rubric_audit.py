#!/usr/bin/env python3
"""A0.2 rubric audit — post-hoc tolerant re-scoring.

Reads existing ``results/*_np2048.json`` (from ``eval.py``), applies a
set of *tolerant* rubric variants to failed responses only, and emits
``results/*_np2048_audited.json`` alongside. The original per-task
provenance stays intact — this tool never overwrites the eval output
and never modifies ``eval.py``.

## What "tolerant" means

Three families of matcher, tried in order for each task that failed
under the original rubric. First hit wins.

1. ``exact_tolerant`` / ``contains_tolerant``
   Both sides are normalized: markdown emphasis and ``<answer>…</answer>``
   wrappers stripped, unicode math symbols (``·``, ``²``, ``μ`` …) mapped
   to ASCII, whitespace collapsed, case folded. Then exact-equality or
   substring is checked.

2. ``unit_normalized``
   For tasks whose original rubric is ``regex`` (e.g. ``sym_dim_02``
   expected ``kg/(m*s^2)`` matched with ``kg\\s*/\\s*\\(?...``), the
   response's unicode math symbols are normalized to ASCII before the
   original regex runs. Rescues answers that were correct except for
   ``·`` vs ``*`` or ``²`` vs ``^2``.

3. ``unit_optional``
   For expected values of the shape ``<number> <unit>`` where the
   response contains just ``<number>`` (unit dropped). Applies only
   when the numeric literal is unambiguously present. Rescues
   ``sym_unit_01`` (expected ``2.5 kg``, response ``2.5``).

## Guardrails

- **Does not touch originals.** Output files are ``*_audited.json``.
- **Original-correct results are preserved verbatim** — no re-check,
  so nothing that was OK can become FAIL through audit.
- **≥2pp guard.** If the audit shifts any model's overall accuracy by
  more than 2 percentage points, the tool prints a warning + diff and
  requires ``--commit`` to actually write files. Default is dry-run
  when the shift exceeds the guard.

Usage
-----

    # Dry-run summary + upgrade diff, no files written
    python rubric_audit.py results/rwkv7_29b_g1h_np2048.json

    # Multiple inputs
    python rubric_audit.py results/*_np2048.json

    # Force write regardless of >2pp shift
    python rubric_audit.py --commit results/*_np2048.json

    # Alternative output suffix / directory
    python rubric_audit.py --suffix _audit2 results/foo_np2048.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any


# --------------------------------------------------------------------------- #
# Normalisation primitives
# --------------------------------------------------------------------------- #

_UNICODE_MATH: dict[str, str] = {
    "·": "*",
    "×": "*",
    "⋅": "*",
    "•": "*",
    "∗": "*",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁻": "-",
    "⁰": "^0",
    "¹": "^1",
    "μ": "u",
    "Ω": "ohm",
    "α": "alpha",
    "β": "beta",
    "π": "pi",
    "√": "sqrt",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
}


def normalize_symbols(s: str) -> str:
    for k, v in _UNICODE_MATH.items():
        s = s.replace(k, v)
    return s


_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_LATEX_INLINE = re.compile(r"\\\(([^)]*)\\\)|\\\[([^\]]*)\\\]")
_MD_EMPH = re.compile(r"\*+")


def strip_prose_wrappers(s: str) -> str:
    """Peel common wrapper prose that hides an otherwise-correct answer."""
    # <answer>X</answer> → X
    m = _ANSWER_TAG.search(s)
    if m:
        s = m.group(1)
    # \boxed{X} → X
    s = _BOXED.sub(r"\1", s)
    # markdown ** emphasis
    s = _MD_EMPH.sub("", s)
    # surrounding whitespace + typographic wrappers
    s = s.strip().strip("\"'` ")
    return s


def norm_response(s: str) -> str:
    """Full normalization pipeline for tolerant matching."""
    s = strip_prose_wrappers(s)
    s = normalize_symbols(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Tolerant matchers
# --------------------------------------------------------------------------- #

def match_exact_tolerant(expected: str, response: str) -> bool:
    return norm_response(expected).casefold() == norm_response(response).casefold()


def match_contains_tolerant(expected: str, response: str) -> bool:
    e = norm_response(expected).casefold()
    r = norm_response(response).casefold()
    return bool(e) and e in r


def match_unit_normalized(rubric_regex: str, response: str) -> bool:
    r = normalize_symbols(response)
    try:
        return bool(re.search(rubric_regex, r, re.IGNORECASE))
    except re.error:
        return False


_NUM_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+([A-Za-z%/·^*]+)\s*$")


def match_unit_optional(expected: str, response: str) -> bool:
    """Accept a response that carries only the numeric part of a
    ``<number> <unit>`` expected answer."""
    m = _NUM_UNIT.match(expected)
    if not m:
        return False
    num = m.group(1)
    r_norm = norm_response(response)
    return bool(re.search(r"(?<!\d)" + re.escape(num) + r"(?!\d)", r_norm))


# --------------------------------------------------------------------------- #
# Per-task audit
# --------------------------------------------------------------------------- #

def audit_result(res: dict[str, Any]) -> dict[str, Any]:
    """Return an audited copy of the per-task result dict.

    Adds ``original_correct`` (== ``res['correct']``), ``audited_correct``,
    ``audit_detail``, and ``audit_upgraded`` (True iff FAIL→OK).
    """
    out = dict(res)
    out["original_correct"] = res["correct"]

    if res["correct"]:
        out["audited_correct"] = True
        out["audit_detail"] = "original PASS (unchanged)"
        out["audit_upgraded"] = False
        return out

    response = res.get("response", "") or ""
    expected = res.get("expected", "")
    rubric = res.get("rubric", {})
    rt = rubric.get("type", "")

    # 1. exact/contains tolerant — ONLY when the original rubric was already
    #    loose (`exact` or `contains`). If the eval author chose `regex`,
    #    that regex was the authoritative precision check (often with
    #    word-boundary anchors); a naive contains_tolerant would strictly
    #    loosen it and produce false positives on short expected values
    #    (e.g. sched_05 expected="1", response contains "1" inside "$12$").
    if rt in ("exact", "contains") and isinstance(expected, str) and expected.strip():
        if match_exact_tolerant(expected, response):
            out["audited_correct"] = True
            out["audit_detail"] = "exact_tolerant (prose-strip + unicode + whitespace + case)"
            out["audit_upgraded"] = True
            return out
        if match_contains_tolerant(expected, response):
            out["audited_correct"] = True
            out["audit_detail"] = "contains_tolerant (prose-strip + unicode + whitespace + case)"
            out["audit_upgraded"] = True
            return out

    # 2. Regex rubric: try after unicode symbol normalization.
    if rt == "regex":
        rv = rubric.get("value", "")
        if match_unit_normalized(rv, response):
            out["audited_correct"] = True
            out["audit_detail"] = "unit_normalized (unicode-math ASCII-ised before regex)"
            out["audit_upgraded"] = True
            return out

    # 3. Unit-optional — applies when expected is `<number> <unit>` and
    #    the response has the numeric literal. Only enabled when the
    #    expected string parses as number+unit; safe across rubric types.
    if isinstance(expected, str) and expected.strip():
        if match_unit_optional(expected, response):
            out["audited_correct"] = True
            out["audit_detail"] = "unit_optional (numeric literal present, unit dropped)"
            out["audit_upgraded"] = True
            return out

    out["audited_correct"] = False
    out["audit_detail"] = "no tolerant match"
    out["audit_upgraded"] = False
    return out


# --------------------------------------------------------------------------- #
# Aggregate summary
# --------------------------------------------------------------------------- #

def summarise(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r[key])
    per_cat = {
        cat: {"n": len(vs), "correct": sum(vs),
              "accuracy": sum(vs) / len(vs) if vs else 0.0}
        for cat, vs in by_cat.items()
    }
    n_total = sum(x["n"] for x in per_cat.values())
    n_correct = sum(x["correct"] for x in per_cat.values())
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "overall_accuracy": n_correct / n_total if n_total else 0.0,
        "per_category": per_cat,
    }


# --------------------------------------------------------------------------- #
# File-level audit
# --------------------------------------------------------------------------- #

def audit_file(path: str) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)

    audited = [audit_result(r) for r in payload["results"]]
    agg_orig = summarise(audited, "original_correct")
    agg_audited = summarise(audited, "audited_correct")

    upgrades = [
        {"id": r["id"], "category": r["category"], "detail": r["audit_detail"]}
        for r in audited if r["audit_upgraded"]
    ]

    return {
        "input_path": path,
        "model": payload.get("model", "?"),
        "aggregate_original": agg_orig,
        "aggregate_audited": agg_audited,
        "delta_pp": (agg_audited["overall_accuracy"] - agg_orig["overall_accuracy"]) * 100,
        "n_upgraded": len(upgrades),
        "upgrades": upgrades,
        "audited_payload": {
            **payload,
            "audit_note": (
                "Post-hoc tolerant re-scoring via rubric_audit.py. "
                "Original eval.py scores preserved under 'original_correct'."
            ),
            "aggregate_original": agg_orig,
            "aggregate_audited": agg_audited,
            "results": audited,
        },
    }


def out_path_for(in_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(in_path)
    return f"{base}{suffix}{ext}"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(audit: dict[str, Any]) -> None:
    o = audit["aggregate_original"]
    a = audit["aggregate_audited"]
    delta = audit["delta_pp"]
    model = audit["model"]

    print(f"\n=== {model}  ({audit['input_path']}) ===")
    print(f"  original overall: {o['n_correct']}/{o['n_total']} "
          f"= {o['overall_accuracy']:.1%}")
    print(f"  audited  overall: {a['n_correct']}/{a['n_total']} "
          f"= {a['overall_accuracy']:.1%}  (Δ {delta:+.2f} pp)")
    if audit["upgrades"]:
        print(f"  upgraded ({audit['n_upgraded']}):")
        for u in audit["upgrades"]:
            print(f"    - {u['id']:20s} [{u['category']}]  {u['detail']}")
    else:
        print("  no upgrades")


def print_category_diff(audit: dict[str, Any]) -> None:
    o = audit["aggregate_original"]["per_category"]
    a = audit["aggregate_audited"]["per_category"]
    cats = sorted(set(o) | set(a))
    any_diff = False
    print("  per-category (orig -> audited):")
    for c in cats:
        ov = o.get(c, {"correct": 0, "n": 0, "accuracy": 0.0})
        av = a.get(c, {"correct": 0, "n": 0, "accuracy": 0.0})
        if ov["correct"] != av["correct"]:
            any_diff = True
            print(f"    {c:20s}  {ov['correct']}/{ov['n']} "
                  f"({ov['accuracy']:.1%})  ->  "
                  f"{av['correct']}/{av['n']} ({av['accuracy']:.1%})")
    if not any_diff:
        print("    (no per-category change)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc rubric audit for A0.2 eval results.")
    ap.add_argument("inputs", nargs="+",
                    help="Result JSON files or globs (results/*_np2048.json).")
    ap.add_argument("--suffix", default="_audited",
                    help="Output filename suffix (default: _audited).")
    ap.add_argument("--commit", action="store_true",
                    help="Write files even when overall shift > 2 pp for any model.")
    ap.add_argument("--threshold-pp", type=float, default=2.0,
                    help="Guard threshold: abort write when |Δ| > this many pp.")
    args = ap.parse_args()

    # Expand globs
    files: list[str] = []
    for pat in args.inputs:
        matches = sorted(glob.glob(pat))
        if not matches and os.path.exists(pat):
            matches = [pat]
        files.extend(matches)
    if not files:
        print("[audit] no input files matched", file=sys.stderr)
        return 2

    audits = [audit_file(f) for f in files]
    for a in audits:
        print_report(a)
        print_category_diff(a)

    over = [a for a in audits if abs(a["delta_pp"]) > args.threshold_pp]
    if over and not args.commit:
        print()
        print(f"[audit] STOP: {len(over)} file(s) shift > {args.threshold_pp:.1f} pp.")
        print("[audit] dry-run only; use --commit after confirming the diff.")
        for a in over:
            print(f"        {a['model']} Δ {a['delta_pp']:+.2f} pp  ({a['input_path']})")
        return 1

    written: list[str] = []
    for a in audits:
        out = out_path_for(a["input_path"], args.suffix)
        with open(out, "w") as f:
            json.dump(a["audited_payload"], f, indent=2)
        written.append(out)

    print()
    print(f"[audit] wrote {len(written)} file(s):")
    for p in written:
        print(f"        {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
