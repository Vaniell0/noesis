#!/usr/bin/env python3
"""Manual sanitization audit gate.

Picks a random sample of sanitised rollouts, prints a compact summary of
each one, and asks the human operator to approve / reject / flag-for-
pattern-refinement. Decisions are appended to
`training/sanitised/audit_decisions.jsonl`.

This must be run at least once before Step 5 (tokenisation) — the point is
to catch regex false negatives before we bake a bad corpus into weights.

Usage:
    python audit_sample.py            # default: 200 samples
    python audit_sample.py -n 20      # smaller batch
    python audit_sample.py --review   # only re-show rollouts already
                                      # flagged for refinement
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

DEFAULT_SANITISED = Path("training/sanitised")


def load_audit(sanitised_dir: Path) -> dict[str, dict[str, Any]]:
    """Map source file name -> audit record from sanitize.py."""
    audit_path = sanitised_dir / "audit.jsonl"
    out: dict[str, dict[str, Any]] = {}
    if not audit_path.exists():
        return out
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("src_file")
            if src:
                out[src] = rec
    return out


def load_decisions(decisions_path: Path) -> dict[str, dict[str, Any]]:
    """Return already-recorded decisions keyed by file name."""
    out: dict[str, dict[str, Any]] = {}
    if not decisions_path.exists():
        return out
    with decisions_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("src_file")
            if src:
                out[src] = rec
    return out


def _short(text: str, limit: int = 240) -> str:
    text = text.replace("\n", " \\n ")
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def summarise(rollout: dict[str, Any], audit_rec: dict[str, Any] | None) -> str:
    chain = rollout.get("chain") or []
    lines: list[str] = []
    lines.append(f"session_id: {rollout.get('session_id')}")
    meta = rollout.get("meta") or {}
    lines.append(
        f"events={len(chain)}  tool_use={meta.get('n_tool_use')}  "
        f"tool_result={meta.get('n_tool_result')}  first_ts={meta.get('first_ts')}"
    )
    if audit_rec:
        red = audit_rec.get("redactions") or []
        pats = sorted({p for r in red for p in r.get("patterns", [])})
        ds = sorted({p for r in red for p in r.get("detect_secrets", [])})
        lines.append(f"redactions: n={len(red)} patterns={pats} ds={ds}")
    lines.append("--- chain head ---")
    for ev in chain[:4]:
        role = ev.get("role")
        if role == "tool_use":
            body = f"{ev.get('name')} {json.dumps(ev.get('input'), ensure_ascii=False)}"
        else:
            body = str(ev.get("content", ""))
        lines.append(f"[{role}] {_short(body)}")
    if len(chain) > 8:
        lines.append(f"... ({len(chain) - 8} events elided) ...")
        for ev in chain[-4:]:
            role = ev.get("role")
            if role == "tool_use":
                body = f"{ev.get('name')} {json.dumps(ev.get('input'), ensure_ascii=False)}"
            else:
                body = str(ev.get("content", ""))
            lines.append(f"[{role}] {_short(body)}")
    return "\n".join(lines)


def prompt_decision() -> tuple[str, str]:
    print("\n[a]ccept  [r]eject  [f]lag pattern refinement  [s]kip  [q]uit")
    while True:
        raw = input("decision> ").strip().lower()
        if raw in ("a", "accept"):
            return "accept", ""
        if raw in ("r", "reject"):
            note = input("reject reason> ").strip()
            return "reject", note
        if raw in ("f", "flag"):
            note = input("pattern to refine> ").strip()
            return "flag_refine", note
        if raw in ("s", "skip"):
            return "skip", ""
        if raw in ("q", "quit"):
            return "quit", ""
        print("unknown input; enter one of a/r/f/s/q")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_SANITISED,
                   help="sanitised rollouts directory")
    p.add_argument("-n", "--sample-size", type=int, default=200,
                   help="target audit sample size (capped by available rollouts)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for reproducible sample selection")
    p.add_argument("--review", action="store_true",
                   help="only show rollouts previously flagged for refinement")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sanitised_dir: Path = args.dir
    if not sanitised_dir.exists():
        print(f"no such dir: {sanitised_dir}", file=sys.stderr)
        return 1

    audit = load_audit(sanitised_dir)
    decisions_path = sanitised_dir / "audit_decisions.jsonl"
    prior = load_decisions(decisions_path)

    all_files = sorted(p for p in sanitised_dir.glob("*.jsonl")
                       if p.name not in ("audit.jsonl", "audit_decisions.jsonl"))
    if args.review:
        candidates = [p for p in all_files
                      if prior.get(p.name, {}).get("decision") == "flag_refine"]
    else:
        candidates = [p for p in all_files if p.name not in prior]

    if not candidates:
        print("no candidates to audit "
              "(all rollouts already have decisions; try --review).",
              file=sys.stderr)
        return 0

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.sample_size]

    print(f"auditing {len(sample)} of {len(candidates)} eligible rollouts. "
          f"prior decisions: {len(prior)}", file=sys.stderr)

    with decisions_path.open("a", encoding="utf-8") as out_f:
        for i, path in enumerate(sample, 1):
            try:
                with path.open("r", encoding="utf-8") as f:
                    rollout = json.loads(f.readline())
            except Exception as e:
                print(f"skip {path.name}: {e}", file=sys.stderr)
                continue

            print(f"\n=== [{i}/{len(sample)}] {path.name} ===")
            print(summarise(rollout, audit.get(path.name)))

            decision, note = prompt_decision()
            if decision == "quit":
                print("stopping — remaining rollouts unaudited.", file=sys.stderr)
                break
            if decision == "skip":
                continue

            record = {
                "src_file": path.name,
                "session_id": rollout.get("session_id"),
                "decision": decision,
                "note": note,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"decisions written to {decisions_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
