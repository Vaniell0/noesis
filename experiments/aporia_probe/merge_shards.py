#!/usr/bin/env python3
"""Merge H20 shard outputs into a single results.jsonl + report.md."""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run import _write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--meta", default="{}")
    args = ap.parse_args()

    rows: List[Dict] = []
    for d in args.shard_dirs:
        p = os.path.join(d, "results.jsonl")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    # Preserve item order by id.
    id_order = [r["id"] for r in rows]
    print(f"[merge] {len(rows)} rows from {len(args.shard_dirs)} shards", file=sys.stderr)

    out_results = os.path.join(args.out, "results.jsonl")
    with open(out_results, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    meta = json.loads(args.meta)
    meta.setdefault("model", "merged")
    meta.setdefault("n_samples", rows[0]["n_samples"] if rows else 0)
    meta.setdefault("max_new_tokens", 0)
    meta.setdefault("temperature", 1.0)
    meta.setdefault("top_p", 0.85)
    meta.setdefault("wall_total_s", sum(r["wall_s"] for r in rows))
    _write_report(rows, args.out, meta=meta)
    print(f"[merge] wrote {out_results} + report.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
