"""training/_common/improve.py — the data improver: the next training round from the map of what works.

Plan: ~/.claude/plans/noesis-data-improver-2026-11-13.md (approved 11-13). For one generated dataset, one model
(+ weights) and one number of ticks M it reads the map (datamap.py, over the dataset and every round made from it)
and writes a round of fresh rows where the model solves sometimes:

    weight of a measured cell = 0 if nothing solved, 1 - p otherwise (Absolute Zero's learnability reward; a solved
                                cell gets 0)
    frontier cells, weight `explore`: on a numeric knob, the value between a live cell and a dead neighbour (where
                                the edge of "works" lies), the dead neighbour itself when no value fits between,
                                and one step past a solved cell at the end of the grid (the zone moves outward)
    rows per cell             = drawn in proportion to the weights (temperature 1: MATES lost diversity at 0)

A generator joins by exposing make_cell_rows(cell, n, rng, style) in the script its catalog record names; the
round is saved as a catalog record whose parents are the base set, so its own results enter the map.
"""
from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from training._common import catalog, datamap

OUT_DIR = catalog.REPO / "training" / "corpus_open" / "improve"


def family_paths(name: str) -> list[str]:
    """The dataset's own file and every catalogued round made from it (parents contain it)."""
    rec = catalog.load(name)
    out = [str(Path(rec.out_path))]
    out += [str(Path(r.out_path)) for r in catalog.all_records() if name in (r.parents or []) and r.out_path]
    return out


def measured(name: str, model: str, M: int, hint: int = 0) -> tuple[list[str], dict]:
    """(cell fields, {cell tuple: [n, solved]}) for one run label over the dataset family."""
    fields = datamap.declared_cells(str(Path(catalog.load(name).out_path)))
    runs = datamap.solve_runs()
    agg: dict = defaultdict(lambda: [0, 0])
    for p in family_paths(name):
        e = datamap.build(p, runs).get(p)
        if not e:
            continue
        fields = fields or e["fields"]
        for c in e["cells"]:
            label = c["model"] + (f"+{c['weights']}" if c["weights"] else "")
            if label == model and c["M"] == M and c["hint"] == hint:
                k = tuple(c["cell"].get(f) for f in fields)
                agg[k][0] += c["n"]; agg[k][1] += c["solved"]
    return fields, dict(agg)


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def plan(fields: list[str], cells: dict, explore: float = 0.1) -> list[dict]:
    """Weighted cells: the measured ones by learnability, the frontier ones at `explore`."""
    out = {}
    for k, (n, s) in cells.items():
        p = s / n
        out[k] = {"cell": dict(zip(fields, k)), "n": n, "solved": s, "zone": datamap.zone(s, n),
                  "weight": 0.0 if s in (0, n) else 1.0 - p, "why": "measured"}
    live = {k for k, (n, s) in cells.items() if s > 0}
    for i, f in enumerate(fields):
        for k in live:
            v = k[i]
            if not _numeric(v):
                continue
            # the measured values of f among cells equal to k on every other field
            line = sorted({c[i] for c in cells if c[:i] + c[i + 1:] == k[:i] + k[i + 1:] and _numeric(c[i])})
            j = line.index(v)
            for nb in [line[j - 1] if j > 0 else None, line[j + 1] if j + 1 < len(line) else None]:
                if nb is None:
                    continue
                kn = k[:i] + (nb,) + k[i + 1:]
                if cells[kn][1] > 0:
                    continue                                   # the neighbour works too: no edge here
                ints = isinstance(v, int) and isinstance(nb, int)
                if ints and abs(nb - v) < 2:                   # nothing fits between: try the dead neighbour again
                    km, why = kn, f"edge on {f}: dead neighbour {nb}"
                else:
                    mid = (v + nb) // 2 if ints else (v + nb) / 2
                    km, why = k[:i] + (mid,) + k[i + 1:], f"edge on {f}: between {v} and {nb}"
                if km not in out:
                    out[km] = {"cell": dict(zip(fields, km)), "n": 0, "solved": 0, "zone": "new",
                               "weight": explore, "why": why}
                elif out[km]["weight"] == 0.0 and out[km]["zone"] == "dead":
                    out[km]["weight"], out[km]["why"] = explore, why
            n, s = cells[k]
            if s == n and len(line) > 1 and v in (line[0], line[-1]):
                step = line[1] - line[0] if v == line[0] else line[-1] - line[-2]
                beyond = v - step if v == line[0] else v + step
                if beyond >= 0:
                    kb = k[:i] + (beyond,) + k[i + 1:]
                    out.setdefault(kb, {"cell": dict(zip(fields, kb)), "n": 0, "solved": 0, "zone": "new",
                                        "weight": explore, "why": f"past solved {f}={v}"})
    return sorted(out.values(), key=lambda c: -c["weight"])


def _generator(script: str):
    path = catalog.REPO / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_cell_rows"):
        raise NotImplementedError(f"{script} has no make_cell_rows(cell, n, rng, style): it is on the map but cannot "
                                  f"make rows for a cell yet")
    return mod.make_cell_rows


def make_round(name: str, model: str, M: int, rows: int, seed: int, hint: int = 0, explore: float = 0.1,
               dry: bool = False) -> dict:
    rec = catalog.load(name)
    fields, cells = measured(name, model, M, hint)
    if not cells:
        return {"error": f"no result for {model} M={M} hint={hint} on {name}: measure it first (m0_pass)"}
    weighted = [c for c in plan(fields, cells, explore) if c["weight"] > 0]
    if not weighted:
        return {"error": "no live cell for this model: the map has no zone here; it needs a hint knob in the "
                         "generator (an easier rung), not more rows", "cells": len(cells)}
    rng = random.Random(seed)
    total = sum(c["weight"] for c in weighted)
    counts = defaultdict(int)
    for _ in range(rows):
        x, acc = rng.random() * total, 0.0
        for i, c in enumerate(weighted):
            acc += c["weight"]
            if x <= acc:
                counts[i] += 1
                break
    for i, c in enumerate(weighted):
        c["rows"] = counts[i]
    label = f"{model}_M{M}" + (f"_h{hint}" if hint else "")
    k = 1 + sum(1 for r in catalog.all_records() if name in (r.parents or []) and label in r.name)
    out_name = f"{name}__{label}_r{k}"
    report = {"name": out_name, "fields": fields, "cells": weighted, "rows": rows}
    if dry:
        return report
    first = next(catalog.rows(name, limit=1))
    make = _generator(rec.script)
    data, failed = [], []
    for c in weighted:
        if not c["rows"]:
            continue
        try:
            data += make(c["cell"], c["rows"], rng, style=first.get("style", "plain"))
        except (AssertionError, KeyError, ValueError) as e:  # a frontier cell the generator cannot make
            failed.append({"cell": c["cell"], "error": f"{type(e).__name__}: {e}"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{out_name.replace('/', '__')}.jsonl"
    with open(out, "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meas = catalog.describe(out)
    cmd = ["python", "training/datasets.py", "improve", name, "--model", model, "--M", str(M), "--hint", str(hint),
           "--rows", str(rows), "--seed", str(seed), "--explore", str(explore)]
    new = catalog.make_record(
        name=out_name, stage="generate", provenance="generated",
        origin=f"data improver round {k} for {model} at M={M}: cells weighted by the map", script=rec.script,
        out_path=str(out.relative_to(catalog.REPO)), out_sha256=meas.get("out_sha256"),
        out_size_bytes=meas["out_size_bytes"], n_rows=meas.get("n_rows"), profile=meas.get("profile", {}),
        format=meas["format"], role="train", status="live", parents=[name],
        recipe={"cmd": cmd, "seed": seed, "deterministic": True}, verifier=rec.verifier,
        notes="weights: 1 - p on measured cells with 0 < p < 1, explore on frontier cells (training/_common/improve.py)",
        extra={"cells": fields, "round": {"model": model, "M": M, "hint": hint, "explore": explore,
                                          "plan": [{k2: c[k2] for k2 in ("cell", "n", "solved", "weight", "why",
                                                                          "rows")} for c in weighted],
                                          "failed": failed}})
    catalog.save(new)
    report.update({"out": str(out.relative_to(catalog.REPO)), "written": len(data), "failed": failed})
    return report
