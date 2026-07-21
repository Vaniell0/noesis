#!/usr/bin/env python3
"""A0.5 analysis — aggregate per-seed JSON into H8-causal verdict.

Reads ``<cell>/seed_*.json`` + ``<cell>/summary.json`` per cell directory
and computes:

- **Per-corruption cell stats** — mean±SD of kl_next, entropy_change,
  argmax_flip, rank_shift (and trajectory metrics if present) across all
  (seed × checkpoint) pairs.
- **H8-causal-A** — σ-response: KL_next as f(σ) on gauss corruption.
  Fits log-log slope; monotonicity check; effect at σ=0.1 vs noise_floor.
  Predict: strictly increasing, slope ≳ 1 (superlinear).
- **H8-causal-B** — layer localisation: per-layer KL_next for zero_layer.
  Coefficient of variation (CV = SD/mean); high CV → localised work,
  low CV → uniform.
- **H8-causal-C** — cross-prompt vs matched baseline: KL(cross_prompt) /
  KL(gauss@0.1) — if state carries prompt-conditional computation,
  swapping in another prompt's state should dominate a norm-matched noise
  injection at comparable Frobenius shift.

Emits a Markdown table per cell + a 2×2 grid comparison table.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_cell(cell_dir: str) -> Dict[str, Any]:
    with open(os.path.join(cell_dir, "summary.json")) as f:
        summary = json.load(f)
    seed_files = sorted(glob.glob(os.path.join(cell_dir, "seed_*.json")))
    seeds = []
    for sf in seed_files:
        with open(sf) as f:
            seeds.append(json.load(f))
    return {"summary": summary, "seeds": seeds, "dir": cell_dir}


# --------------------------------------------------------------------------- #
# Flatten corruption records into per-type buckets
# --------------------------------------------------------------------------- #

def _bucket_key(record: Dict[str, Any]) -> str:
    """Group key: type + parametric slot (sigma/alpha/layer)."""
    t = record["type"]
    if t == "gauss":
        return f"gauss@σ={record['sigma']:g}"
    if t == "scale":
        return f"scale@α={record['alpha']:g}"
    if t in ("zero_layer", "shuffle_heads"):
        return f"{t}@L{record['layer']}"
    return t


def bucket_seeds(cell: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Return {bucket_key: [record, ...]} across all seeds × checkpoints."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seed in cell["seeds"]:
        for cp in seed["checkpoints"]:
            for rec in cp["corruptions"]:
                buckets[_bucket_key(rec)].append(rec)
    return buckets


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #

def _msd(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": n}


def stat_bucket(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in ("kl_next", "argmax_flip", "entropy_change", "rank_shift",
                "token_overlap_N", "cum_KL_N"):
        vals = [float(r[key]) for r in records if key in r]
        if vals:
            out[key] = _msd(vals)
    return out


# --------------------------------------------------------------------------- #
# H8-causal-A: σ-response fit
# --------------------------------------------------------------------------- #

def h8a_sigma_response(buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Extract KL_next(σ) for gauss corruption; fit log-log slope; check
    monotonicity."""
    gauss_keys = sorted(
        [k for k in buckets if k.startswith("gauss@σ=")],
        key=lambda k: float(k.split("=")[1]),
    )
    if not gauss_keys:
        return {"status": "no_gauss_data"}

    points = []
    for key in gauss_keys:
        sigma = float(key.split("=")[1])
        vals = [float(r["kl_next"]) for r in buckets[key]]
        points.append({"sigma": sigma, "kl_mean": statistics.fmean(vals),
                       "kl_sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                       "n": len(vals)})

    monotonic = all(points[i]["kl_mean"] <= points[i + 1]["kl_mean"]
                    for i in range(len(points) - 1))

    logs = [(math.log(p["sigma"]), math.log(max(p["kl_mean"], 1e-12)))
            for p in points]
    n = len(logs)
    mx = statistics.fmean([x for x, _ in logs])
    my = statistics.fmean([y for _, y in logs])
    num = sum((x - mx) * (y - my) for x, y in logs)
    den = sum((x - mx) ** 2 for x, _ in logs)
    slope = num / den if den > 0 else float("nan")

    noise_floor = buckets.get("noise_floor", [])
    nf_kl = statistics.fmean([float(r["kl_next"]) for r in noise_floor]) if noise_floor else 0.0

    return {
        "status": "ok",
        "points": points,
        "loglog_slope": slope,
        "monotonic": monotonic,
        "noise_floor_kl": nf_kl,
    }


# --------------------------------------------------------------------------- #
# H8-causal-B: layer localisation
# --------------------------------------------------------------------------- #

def h8b_layer_profile(buckets: Dict[str, List[Dict[str, Any]]],
                     corr_type: str = "zero_layer") -> Dict[str, Any]:
    """Per-layer KL_next mean; coefficient of variation across layers."""
    keys = sorted(
        [k for k in buckets if k.startswith(f"{corr_type}@L")],
        key=lambda k: int(k.split("L")[1]),
    )
    if not keys:
        return {"status": f"no_{corr_type}_data"}

    profile = []
    for key in keys:
        layer = int(key.split("L")[1])
        vals = [float(r["kl_next"]) for r in buckets[key]]
        profile.append({"layer": layer, "kl_mean": statistics.fmean(vals),
                        "kl_sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals)})

    kls = [p["kl_mean"] for p in profile]
    mean_kl = statistics.fmean(kls)
    sd_kl = statistics.pstdev(kls) if len(kls) > 1 else 0.0
    cv = sd_kl / mean_kl if mean_kl > 0 else float("nan")

    return {"status": "ok", "profile": profile, "mean_kl": mean_kl,
            "sd_kl": sd_kl, "cv": cv}


# --------------------------------------------------------------------------- #
# H8-causal-C: cross-prompt vs matched baseline
# --------------------------------------------------------------------------- #

def h8c_cross_ratio(buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """KL(cross_prompt) / KL(gauss@σ=0.1). Prompt-conditional if ratio >> 1."""
    cross = buckets.get("cross_prompt", [])
    if not cross:
        return {"status": "no_cross_data"}
    baseline_key = "gauss@σ=0.1"
    baseline = buckets.get(baseline_key, [])
    if not baseline:
        return {"status": f"no_baseline_({baseline_key})"}

    cross_kl = statistics.fmean([float(r["kl_next"]) for r in cross])
    base_kl = statistics.fmean([float(r["kl_next"]) for r in baseline])
    ratio = cross_kl / base_kl if base_kl > 0 else float("inf")

    return {"status": "ok", "cross_kl": cross_kl, "baseline_kl": base_kl,
            "baseline_key": baseline_key, "ratio": ratio}


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #

def _fmt(v: float, digits: int = 3) -> str:
    if v != v:
        return "—"
    if abs(v) < 1e-4 or abs(v) > 1e4:
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def _fmt_msd(msd: Dict[str, float], digits: int = 3) -> str:
    return f"{_fmt(msd['mean'], digits)}±{_fmt(msd['sd'], digits)}"


def report_cell(cell: Dict[str, Any]) -> str:
    tag = os.path.basename(cell["dir"].rstrip("/"))
    buckets = bucket_seeds(cell)

    lines = [f"### Cell: {tag}", ""]
    lines.append(f"- model: `{cell['summary']['model']}`")
    lines.append(f"- prompt: {cell['summary']['prompt']}, cross: {cell['summary']['cross_prompt']}")
    lines.append(
        f"- seeds={cell['summary']['seeds']}, tokens={cell['summary']['max_new_tokens']}, "
        f"k_cp={cell['summary']['k_checkpoints']}, cont_N={cell['summary']['continuation_steps']}"
    )
    lines.append("")

    lines.append("#### Per-corruption cell stats (mean±SD over seed×checkpoint)")
    lines.append("")
    lines.append("| corruption | KL_next | entropy Δ | argmax_flip | rank_shift | n |")
    lines.append("|---|---|---|---|---|---|")

    def _rowkey(k: str) -> tuple:
        if k == "noise_floor":
            return (0, 0.0, "")
        if k.startswith("gauss@"):
            return (1, float(k.split("=")[1]), "")
        if k.startswith("scale@"):
            return (2, float(k.split("=")[1]), "")
        if k.startswith("zero_layer@"):
            return (3, float(k.split("L")[1]), "")
        if k.startswith("shuffle_heads@"):
            return (4, float(k.split("L")[1]), "")
        if k == "freeze_prev":
            return (5, 0.0, "")
        if k == "cross_prompt":
            return (6, 0.0, "")
        return (99, 0.0, k)

    for key in sorted(buckets.keys(), key=_rowkey):
        s = stat_bucket(buckets[key])
        kl = _fmt_msd(s.get("kl_next", {"mean": float("nan"), "sd": float("nan")}))
        en = _fmt_msd(s.get("entropy_change", {"mean": float("nan"), "sd": float("nan")}))
        af = _fmt_msd(s.get("argmax_flip", {"mean": float("nan"), "sd": float("nan")}), 2)
        rk = _fmt_msd(s.get("rank_shift", {"mean": float("nan"), "sd": float("nan")}), 1)
        n = s.get("kl_next", {"n": 0})["n"]
        lines.append(f"| {key} | {kl} | {en} | {af} | {rk} | {n} |")
    lines.append("")

    h8a = h8a_sigma_response(buckets)
    lines.append("#### H8-causal-A — σ-response")
    lines.append("")
    if h8a["status"] == "ok":
        lines.append(f"- log-log slope: **{_fmt(h8a['loglog_slope'], 2)}** "
                     f"(>1 = superlinear KL growth in σ)")
        lines.append(f"- monotonic in σ: **{h8a['monotonic']}**")
        lines.append(f"- noise-floor KL: {_fmt(h8a['noise_floor_kl'])}")
        for p in h8a["points"]:
            lines.append(f"  - σ={p['sigma']:g}: KL={_fmt(p['kl_mean'])}±{_fmt(p['kl_sd'])} (n={p['n']})")
    else:
        lines.append(f"- {h8a['status']}")
    lines.append("")

    h8b = h8b_layer_profile(buckets, "zero_layer")
    lines.append("#### H8-causal-B — layer localisation (zero_layer)")
    lines.append("")
    if h8b["status"] == "ok":
        lines.append(f"- across-layer CV: **{_fmt(h8b['cv'], 2)}** "
                     f"(low = uniform; high = localised)")
        lines.append(f"- mean KL across layers: {_fmt(h8b['mean_kl'])}, "
                     f"SD across layers: {_fmt(h8b['sd_kl'])}")
        for p in h8b["profile"]:
            lines.append(f"  - L{p['layer']}: KL={_fmt(p['kl_mean'])}±{_fmt(p['kl_sd'])} (n={p['n']})")
    else:
        lines.append(f"- {h8b['status']}")
    lines.append("")

    h8b_sh = h8b_layer_profile(buckets, "shuffle_heads")
    lines.append("#### H8-causal-B — layer localisation (shuffle_heads)")
    lines.append("")
    if h8b_sh["status"] == "ok":
        lines.append(f"- across-layer CV: **{_fmt(h8b_sh['cv'], 2)}**")
        lines.append(f"- mean KL: {_fmt(h8b_sh['mean_kl'])}, SD: {_fmt(h8b_sh['sd_kl'])}")
        for p in h8b_sh["profile"]:
            lines.append(f"  - L{p['layer']}: KL={_fmt(p['kl_mean'])}±{_fmt(p['kl_sd'])} (n={p['n']})")
    else:
        lines.append(f"- {h8b_sh['status']}")
    lines.append("")

    h8c = h8c_cross_ratio(buckets)
    lines.append("#### H8-causal-C — cross-prompt vs norm-matched noise")
    lines.append("")
    if h8c["status"] == "ok":
        lines.append(f"- cross_prompt KL: {_fmt(h8c['cross_kl'])}")
        lines.append(f"- baseline ({h8c['baseline_key']}) KL: {_fmt(h8c['baseline_kl'])}")
        lines.append(f"- ratio: **{_fmt(h8c['ratio'], 2)}** "
                     f"(>1 = state carries prompt-conditional signal)")
    else:
        lines.append(f"- {h8c['status']}")
    lines.append("")

    return "\n".join(lines)


def report_grid(cells: List[Dict[str, Any]]) -> str:
    """Compact 2×2 grid comparison across cells."""
    lines = ["### H8-causal grid summary", ""]
    lines.append("| cell | σ-slope | monot | zero_L CV | shuf_L CV | cross/base | freeze_prev KL |")
    lines.append("|---|---|---|---|---|---|---|")
    for cell in cells:
        tag = os.path.basename(cell["dir"].rstrip("/"))
        buckets = bucket_seeds(cell)
        h8a = h8a_sigma_response(buckets)
        h8b_z = h8b_layer_profile(buckets, "zero_layer")
        h8b_s = h8b_layer_profile(buckets, "shuffle_heads")
        h8c = h8c_cross_ratio(buckets)
        fp = buckets.get("freeze_prev", [])
        fp_kl = statistics.fmean([float(r["kl_next"]) for r in fp]) if fp else float("nan")

        row = [
            tag,
            _fmt(h8a.get("loglog_slope", float("nan")), 2) if h8a["status"] == "ok" else "—",
            str(h8a.get("monotonic", "—")) if h8a["status"] == "ok" else "—",
            _fmt(h8b_z.get("cv", float("nan")), 2) if h8b_z["status"] == "ok" else "—",
            _fmt(h8b_s.get("cv", float("nan")), 2) if h8b_s["status"] == "ok" else "—",
            _fmt(h8c.get("ratio", float("nan")), 2) if h8c["status"] == "ok" else "—",
            _fmt(fp_kl),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="A0.5 aggregator — H8-causal verdicts.")
    ap.add_argument("cells", nargs="+", help="Cell dirs (each must contain summary.json + seed_*.json).")
    ap.add_argument("--out", default=None, help="If set, write report to this file. Else stdout.")
    args = ap.parse_args()

    cells = [load_cell(d) for d in args.cells]

    report = []
    report.append("# A0.5 — H8-causal analysis")
    report.append("")
    report.append(report_grid(cells))
    for cell in cells:
        report.append("")
        report.append(report_cell(cell))

    text = "\n".join(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[a05_analyze] wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
