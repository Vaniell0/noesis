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
    training/.venv/bin/python training/datasets.py map [NAME] [--by f1,f2] [--model S] [--M k]   # what works and when
    training/.venv/bin/python training/datasets.py improve NAME --model LABEL [--M k] [--rows N] [--dry]  # next round

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


def cmd_map(a):
    from training._common import datamap
    if a.name and not (catalog.REPO / a.name).exists():   # a catalog name: the set and every round made from it
        from training._common import improve
        runs = datamap.solve_runs()
        for path in improve.family_paths(a.name):
            _map_file(a, path, datamap.build(path, runs).get(path))
        return
    m = datamap.build(a.name)
    if a.name:
        return _map_file(a, a.name, m.get(a.name))
    print(f"{'dataset file':44s} {'model':44s} {'M':>2s} {'hint':>4s} {'solved':>11s}  dead/zone/solved cells")
    for s in datamap.summary(m):
        mod = s["model"] + (f"+{s['weights']}" if s["weights"] else "")
        if a.model and a.model not in mod:
            continue
        print(f"{Path(s['file']).stem:44s} {mod:44s} {s['M']:>2d} {s['hint']:>4d} "
              f"{s['solved']:>5d}/{s['n']:<5d}  {s['dead']}/{s['zone']}/{s['solved_cells']}")


def _map_file(a, path, e):
    from collections import defaultdict
    from training._common import datamap
    if not e:
        print(f"{path}: no result names it yet")
        return
    by = a.by.split(",") if a.by else e["fields"]
    agg = defaultdict(lambda: [0, 0])
    for c in e["cells"]:
        mod = c["model"] + (f"+{c['weights']}" if c["weights"] else "")
        if (a.model and a.model not in mod) or (a.M is not None and c["M"] != a.M):
            continue
        k = (mod, c["M"], c["hint"], tuple(c["cell"].get(f) for f in by))
        agg[k][0] += c["n"]; agg[k][1] += c["solved"]
    print(f"{path}: cells by {', '.join(by)}")
    for (mod, M, hint, vals), (n, s) in sorted(agg.items(), key=lambda kv: (kv[0][:3], str(kv[0][3]))):
        print(f"  {mod:40s} M={M} h={hint}  {' '.join(f'{f}={v}' for f, v in zip(by, vals)):60s} "
              f"{s:>4d}/{n:<4d} {datamap.zone(s, n)}")
    for s in e["skipped"]:
        print("  skipped:", s)


def cmd_improve(a):
    from training._common import improve
    r = improve.make_round(a.name, a.model, a.M, a.rows, a.seed, hint=a.hint, explore=a.explore, dry=a.dry)
    if "error" in r:
        print(r["error"])
        return 1
    print(f"{r['name']}: {len(r['cells'])} cells weighted, {r['rows']} rows" + ("" if a.dry else f" -> {r['out']}"))
    for c in r["cells"]:
        cell = " ".join(f"{k}={v}" for k, v in c["cell"].items() if v is not None)
        print(f"  w={c['weight']:.2f} rows={c.get('rows', 0):>4d} {c['solved']:>3d}/{c['n']:<3d} {cell:58s} {c['why']}")
    for f in r.get("failed", []):
        print("  not made:", f)


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
    p = sub.add_parser("map", help="what works and when: solve rate per dataset cell, model and ticks, from results")
    p.add_argument("name", nargs="?", help="catalog name or repo-relative path; none: summary of every dataset")
    p.add_argument("--by", help="comma list of cell fields to group by (default: all)")
    p.add_argument("--model", help="substring of the model (+weights) label")
    p.add_argument("--M", type=int, help="ticks")
    p.set_defaults(fn=cmd_map)
    p = sub.add_parser("improve", help="next training round where the model solves sometimes (needs map results)")
    p.add_argument("name"); p.add_argument("--model", required=True, help="run label as `map` prints it")
    p.add_argument("--M", type=int, default=0); p.add_argument("--hint", type=int, default=0)
    p.add_argument("--rows", type=int, default=400); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--explore", type=float, default=0.1, help="weight of a frontier cell")
    p.add_argument("--dry", action="store_true", help="print the plan, write nothing")
    p.set_defaults(fn=cmd_improve)
    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
