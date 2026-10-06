#!/usr/bin/env python3
"""training/datasets.py — the one command for "what data do we have and can we trust it".

    training/.venv/bin/python training/datasets.py ls [--role train] [--state used-unmeasured]
    training/.venv/bin/python training/datasets.py show NAME
    training/.venv/bin/python training/datasets.py usage            # confirmed / used / idle, all datasets
    training/.venv/bin/python training/datasets.py verify [NAME] [--fast]
    training/.venv/bin/python training/datasets.py lineage NAME
    training/.venv/bin/python training/datasets.py overlap EVAL TRAIN    # exact-prompt contamination check
    training/.venv/bin/python training/datasets.py rows NAME -n 3
    training/.venv/bin/python training/datasets.py backfill [--no-hash] [--only REGEX] [--overwrite]
    training/.venv/bin/python training/datasets.py index            # rewrite the auto-generated doc tables

Records live in training/catalog/ (tracked); artifacts stay local and are found by path
and checked by SHA-256. See training/_common/catalog.py for the design.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from training._common import catalog  # noqa: E402


def _hints() -> dict:
    import yaml
    p = catalog.CATALOG_DIR / "hints.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def cmd_ls(a):
    recs = catalog.all_records()
    print(f"{'name':44s} {'role':11s} {'format':9s} {'rows':>9s} {'MB':>8s}  {'status':10s} state")
    for r in recs:
        if a.role and r.role != a.role:
            continue
        u = catalog.usage(r.name)
        if a.state and u["state"] != a.state:
            continue
        mb = (r.out_size_bytes or 0) / 1e6
        flag = " [personal]" if catalog.is_personal(r.name) else ""
        print(f"{r.name:44s} {r.role or '-':11s} {r.format or '-':9s} {str(r.n_rows if r.n_rows is not None else '-'):>9s} "
              f"{mb:8.2f}  {r.status:10s} {u['state']}{flag}")


def cmd_show(a):
    r = catalog.load(a.name)
    d = asdict(r)
    d["usage"] = catalog.usage(a.name)
    print(json.dumps(d, indent=2, ensure_ascii=False))


def cmd_usage(a):
    from collections import Counter, defaultdict
    groups = defaultdict(list)
    for r in catalog.all_records():
        groups[catalog.usage(r.name)["state"]].append(r.name)
    order = ["measured", "has-results", "used-unmeasured", "unreferenced", "planned", "superseded", "reclassified", "empty", "scratch"]
    for k in order + [k for k in groups if k not in order]:
        if k in groups:
            print(f"\n{k} ({len(groups[k])})")
            for n in groups[k]:
                r = catalog.load(n)
                ev = "; ".join(f"{e.get('verdict', '?')}: {e.get('run', '')}" for e in r.evidence)
                if not ev and k == "has-results":
                    res = catalog.results_using(n)
                    ev = f"{len(res)} result files, e.g. {res[0]}"
                print(f"  {n:44s} {ev}")
    print("\nverdicts recorded:", dict(Counter(e.get("verdict", "?") for r in catalog.all_records() for e in r.evidence)))


def cmd_verify(a):
    names = [a.name] if a.name else catalog.names()
    bad = 0
    for n in names:
        probs = catalog.verify(n, check_hash=not a.fast)
        if probs:
            bad += 1
            print(f"FAIL {n}: " + "; ".join(probs))
        elif a.name:
            print(f"ok   {n}")
    print(f"{len(names) - bad}/{len(names)} ok" + (" (hash not checked)" if a.fast else ""))
    return 1 if bad else 0


def cmd_lineage(a):
    print(" -> ".join(catalog.lineage(a.name)))


def cmd_overlap(a):
    print(json.dumps(catalog.overlap(a.a, a.b), indent=2))


def cmd_rows(a):
    for i, row in enumerate(catalog.rows(a.name, limit=a.n, allow_personal=a.allow_personal)):
        print(json.dumps({k: (v if not hasattr(v, "shape") else f"tensor{tuple(v.shape)}") for k, v in row.items()},
                         ensure_ascii=False)[:600])


def cmd_backfill(a):
    hints = _hints()
    planned = catalog.register_planned(hints.get("_planned", []), overwrite=a.overwrite)
    if planned:
        print("planned sources recorded:", ", ".join(planned))
    written = catalog.backfill(hints, hash_it=not a.no_hash, only=a.only, overwrite=a.overwrite)
    print(f"wrote {len(written)} records to {catalog.CATALOG_DIR}")
    for n in written:
        print("  +", n)


def cmd_index(a):
    sys.path.insert(0, str(HERE))
    import regenerate_corpus_index as rci
    rci.main()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ls"); p.add_argument("--role"); p.add_argument("--state"); p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("show"); p.add_argument("name"); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("usage"); p.set_defaults(fn=cmd_usage)
    p = sub.add_parser("verify"); p.add_argument("name", nargs="?"); p.add_argument("--fast", action="store_true")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("lineage"); p.add_argument("name"); p.set_defaults(fn=cmd_lineage)
    p = sub.add_parser("overlap"); p.add_argument("a"); p.add_argument("b"); p.set_defaults(fn=cmd_overlap)
    p = sub.add_parser("rows"); p.add_argument("name"); p.add_argument("-n", type=int, default=3)
    p.add_argument("--allow-personal", action="store_true", help="read data derived from the owner's own sessions")
    p.set_defaults(fn=cmd_rows)
    p = sub.add_parser("backfill"); p.add_argument("--no-hash", action="store_true")
    p.add_argument("--only"); p.add_argument("--overwrite", action="store_true"); p.set_defaults(fn=cmd_backfill)
    p = sub.add_parser("index"); p.set_defaults(fn=cmd_index)
    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
