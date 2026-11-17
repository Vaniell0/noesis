"""training/_common/datamap.py — the map of what works and when, per dataset cell.

Why (2026-11-13): solve rates were read one result file at a time; whether a cell of a generated set is dead, in
the zone or solved for a given model and number of ticks lived in chat and in HZLK tables. The map puts it beside
the catalog and is computed the same way the catalog's usage state is: never declared, always derived from the
result files that name a dataset's path and record which rows were solved (m0_pass: files, model, weights, ticks,
hint, right_ids, responses), joined with the rows' own low-cardinality fields (the cells).

    cell   = the values of a row's descriptive fields (cipher, payload kind, length, rung, ...): every scalar field
             with 2..64 distinct values over the file, except text and id fields
    run    = model (+ trained weights) x ticks M x hint level
    zone   = dead (no row solved), zone (some), solved (all); the improver trains where a cell is in the zone
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from training._common.catalog import REPO

RESULT_ROOTS = ["experiments", "training/runs"]
_MAX_BYTES = 80_000_000
# text, ids and answers are never cells; neither is any field unique per row
_NOT_CELL = {"id", "prompt", "answer", "content_id", "notes", "trace", "prefill", "gold", "text", "user", "think",
             "system", "rubric", "plaintext", "ciphertext", "message", "question"}
_MAX_VALUES = 64


def _rel(p: str) -> str:
    q = Path(p)
    if q.is_absolute():
        try:
            return str(q.resolve().relative_to(REPO))
        except ValueError:
            return str(q)
    return str(q)


def _model_label(meta_model: str, model: str) -> tuple[str, str]:
    """(model, weights) as short names; run.py records 'MODEL + WEIGHTS' in _meta.model when --weights is given."""
    base, _, w = (meta_model or model or "?").partition(" + ")
    name = Path(base).name.removesuffix(".pth")
    name = re.sub(r"-\d{8}-ctx\d+$", "", name)              # rwkv7-g1d-0.4b-20260210-ctx8192 -> rwkv7-g1d-0.4b
    return name, (Path(w).name.removesuffix(".pth") if w else "")


def solve_runs() -> list[dict]:
    """Every result file that records per-row outcomes on dataset files."""
    out = []
    for root in RESULT_ROOTS:
        for dp, dn, fn in os.walk(REPO / root):
            dn[:] = [d for d in dn if d not in {".venv", "__pycache__", "checkpoints"}]
            for f in fn:
                if not f.endswith(".json") or "partial" in f or "status" in f:
                    continue
                p = Path(dp) / f
                try:
                    if p.stat().st_size > _MAX_BYTES:
                        continue
                    text = p.read_text(errors="ignore")
                except OSError:
                    continue
                if '"right_ids"' not in text or '"files"' not in text:
                    continue
                try:
                    d = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(d, dict) or "right_ids" not in d:
                    continue
                meta = d.get("_meta", {})
                model, weights = _model_label(meta.get("model", ""), d.get("model", ""))
                out.append({"result": str(p.relative_to(REPO)), "files": [_rel(x) for x in d.get("files", [])],
                            "model": model, "weights": weights, "M": int(d.get("ticks") or 0),
                            "hint": int(d.get("hint") or 0), "date": meta.get("date", ""),
                            "n": d.get("n"), "right": set(d["right_ids"]),
                            "answered": set(d["responses"]) if isinstance(d.get("responses"), dict) else None})
    return out


_rows_cache: dict = {}


def _rows(path: str) -> Optional[dict]:
    if path not in _rows_cache:
        p = REPO / path
        if not p.exists():
            _rows_cache[path] = None
        else:
            rows = {}
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if isinstance(r, dict) and "id" in r:
                            rows[r["id"]] = r
            _rows_cache[path] = rows
    return _rows_cache[path]


def cell_fields(rows: dict) -> list[str]:
    vals = defaultdict(set)
    for r in rows.values():
        for k, v in r.items():
            if k not in _NOT_CELL and isinstance(v, (str, int, float, bool)) and not isinstance(v, dict):
                vals[k].add(v)
    n = len(rows)
    return sorted(k for k, s in vals.items() if 2 <= len(s) <= _MAX_VALUES and len(s) < n)


def declared_cells(path: str) -> list[str]:
    """Cell fields a catalog record declares (hint `cells`: the generator's own knobs), else []. Declared because
    a scan cannot tell a knob from a derived field (the ladder's `letters` is the length of each payload)."""
    from training._common.catalog import all_records
    if "recs" not in _rows_cache:
        _rows_cache["recs"] = {r.out_path: (r.extra or {}).get("cells", []) for r in all_records() if r.out_path}
    return list(_rows_cache["recs"].get(path, []))


def zone(solved: int, n: int) -> str:
    return "dead" if solved == 0 else ("solved" if solved == n else "zone")


def build(path: Optional[str] = None, runs: Optional[list] = None) -> dict:
    """{dataset path: {"fields": [...], "cells": [{run keys, cell, n, solved, results}], "skipped": [...]}}.
    path: one dataset file (repo-relative); None: every file any result names."""
    runs = runs if runs is not None else solve_runs()
    out: dict = {}
    for run in runs:
        for f in run["files"]:
            if path and f != path:
                continue
            rows = _rows(f)
            entry = out.setdefault(f, {"fields": [], "agg": defaultdict(lambda: [0, 0, set()]), "skipped": []})
            if rows is None:
                entry["skipped"].append(f"{run['result']}: dataset file missing")
                continue
            entry["fields"] = entry["fields"] or declared_cells(f) or cell_fields(rows)
            ids = run["answered"] & set(rows) if run["answered"] is not None else None
            if ids is None:
                if run["n"] != len(rows):     # no per-row record and not every row: which rows were run is unknown
                    entry["skipped"].append(f"{run['result']}: {run['n']} of {len(rows)} rows, no responses")
                    continue
                ids = set(rows)
            for i in ids:
                r = rows[i]
                cell = tuple((k, r.get(k)) for k in entry["fields"])
                key = (run["model"], run["weights"], run["M"], run["hint"], cell)
                a = entry["agg"][key]
                a[0] += 1; a[1] += i in run["right"]; a[2].add(run["result"])
    for f, e in out.items():
        e["cells"] = [{"model": k[0], "weights": k[1], "M": k[2], "hint": k[3], "cell": dict(k[4]),
                       "n": a[0], "solved": a[1], "zone": zone(a[1], a[0]), "results": sorted(a[2])}
                      for k, a in sorted(e.pop("agg").items(), key=lambda kv: (kv[0][:4], str(kv[0][4])))]
    return out


def summary(m: dict) -> list[dict]:
    """Per dataset file and run: rows measured, rows solved, cells dead / zone / solved."""
    out = []
    for f, e in sorted(m.items()):
        per = defaultdict(lambda: {"n": 0, "solved": 0, "dead": 0, "zone": 0, "solved_cells": 0})
        for c in e["cells"]:
            s = per[(c["model"], c["weights"], c["M"], c["hint"])]
            s["n"] += c["n"]; s["solved"] += c["solved"]
            s["solved_cells" if c["zone"] == "solved" else c["zone"]] += 1
        for k, s in sorted(per.items()):
            out.append({"file": f, "model": k[0], "weights": k[1], "M": k[2], "hint": k[3], **s})
    return out
