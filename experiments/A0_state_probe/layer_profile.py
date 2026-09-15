#!/usr/bin/env python3
"""layer_profile.py — which layers should L_state actually work on?

`work_layers` has been carried as an inherited constant since the A1 pilot:
every config in `training/config/` uses `[12, 16, 20]` (the "A0.5 set", named
in docs/rl-track.md:1390 as the layers that empirically carry state-work) and
two older ones use `[4, 16, 20]`. Nothing measures it per checkpoint.

Two facts sat unreconciled until 2026-09-14:

1. The A0.5 set, chosen empirically, ends at L20.
2. `docs/community-map.md:299`, recorded 2026-08-12: "corpus-only CV picks
   L16/24 at 1.5B = L21/32 at 2.9B - both ~0.67 depth fraction, independent of
   model size... Implication for L_state work_layers: L_state should emphasise
   ~0.67*n_layer." That implication was never applied — 0.67*32 = 21.3 appears
   in NO config in the repo.

Reading G1i's own per-layer spectrum settles it — but only once the grid is
fine enough to be read at all. Measured 2026-09-15 at stride 1 over five
prompts (`results/jlens_stride1_multiprompt/`):

    L   frac   live      L   frac   live
    12  0.38  17.77      21  0.66  17.79
    13  0.41  18.70      22  0.69  11.03   <- cliff, -38% in one layer
    14  0.44  19.07      23  0.72  11.13
    15  0.47  24.12      24  0.75   8.84
    16  0.50  17.61      25  0.78   8.69
    17  0.53  20.55      26  0.81  12.96
    18  0.56  22.07      27  0.84  13.88
    19  0.59  20.35      28  0.88  13.39
    20  0.62  22.61

Breadth oscillates between 17 and 24 across L12-L21, falls off a cliff at
L21->L22, bottoms at L24-L25 and partially recovers by L28. The A0.5 set
[12, 16, 20] sits entirely inside the plateau and stops one layer short of
the fall.

WHAT THE COARSE GRID SAID, AND WHY IT WAS WRONG. The first version of this
probe read a stride-4 profile from a single prompt and reported "peak at L20,
breadth halves by L24". Both halves were artefacts of where the samples fell:
the true peak is L15, and the fall is a one-layer cliff at L21->L22 that a
grid sampling 20 and 24 cannot see. The recommendation logic inherited the
same flaw -- it took the layer after the global peak as "the transition",
which is only true when the grid is coarse. It now scans every consecutive
pair for the steepest fall, and reports the grid stride so a reader can tell
an interval from a layer.

INDEPENDENT CONVERGENCE, worth more than either measurement alone. The cliff
sits at depth fraction 0.656. `docs/community-map.md:296-299` recorded on
2026-08-12 that a corpus-only cross-validation, a completely different method
on different data, picks ~0.67 of depth on two model sizes -- and noted that
the implication for `L_state` had never been applied anywhere. 0.67 * 32 =
21.4. The two land on the same layer.

NOTE on portability: the transferable quantity is the DEPTH FRACTION, not the
layer index (community-map.md:297 — L16/24 at 1.5B ~ L21/32 at 2.9B). G1h and
G1i are both 2.9B/32 layers, so a set measured on one carries to the other
unchanged; a change of model SIZE requires re-measuring.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.results import save_result
from experiments._common.runtime import progress


def profile_from_jlens(layer_stats: dict, n_layer: int) -> list[dict]:
    """Per-layer breadth, aggregated from the per-head spectra.

    Deliberately does NOT use the file's own `mean_stable_rank`: stable_rank is
    energy concentration, not a direction count, and reading it as a rank is
    what produced this project's retracted "the state is near rank-1" claim.
    """
    rows = []
    for key in sorted(layer_stats, key=lambda k: int(k)):
        heads = layer_stats[key].get("per_head") or []
        if not heads:
            continue
        live = [h["numerical_rank_1pct"] for h in heads if "numerical_rank_1pct" in h]
        er = [h["effective_rank_entropy"] for h in heads if "effective_rank_entropy" in h]
        if not live:
            continue
        rows.append({
            "layer": int(key),
            "depth_fraction": int(key) / n_layer,
            "live_mean": statistics.mean(live),
            "live_std": statistics.pstdev(live),
            "live_min": min(live),
            "live_max": max(live),
            "entropy_rank_mean": statistics.mean(er) if er else float("nan"),
            "sigma1_mean": layer_stats[key].get("mean_sigma1", float("nan")),
            "n_head": len(heads),
        })
    return rows


def recommend(rows: list[dict]) -> dict:
    """The steepest consecutive drop in breadth, and a set that brackets it.

    Rewritten 2026-09-15, after the stride-1 measurement showed the previous
    version was answering a question the grid had made up. It took the global
    peak and the next measured layer, which is only the transition when the
    grid is coarse enough that "the next measured layer" is far away. At
    stride 4 (L12/16/20/24/28) that gave peak L20 -> L24, "breadth halves".
    At stride 1 over five prompts the peak is L15 and the same rule returns
    L15 -> L16, a 27% dip in the middle of an oscillating plateau, while the
    actual cliff sits at L21 -> L22 (17.79 -> 11.03) where the coarse grid had
    no sample at all. The old reading was aliasing, not structure.

    So: scan every consecutive pair and take the steepest relative fall. With
    a coarse grid this still finds something, but what it finds is an interval,
    not a layer, and `grid_stride` is reported so a reader can tell which.
    """
    if len(rows) < 2:
        return {}
    peak = max(rows, key=lambda r: r["live_mean"])
    strides = {rows[i + 1]["layer"] - rows[i]["layer"] for i in range(len(rows) - 1)}
    drops = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        drops.append({"from_layer": a["layer"], "to_layer": b["layer"],
                      "live_before": a["live_mean"], "live_after": b["live_mean"],
                      "delta": b["live_mean"] - a["live_mean"],
                      "relative": (b["live_mean"] - a["live_mean"]) / a["live_mean"],
                      "depth_fraction": a["depth_fraction"]})
    cliff = min(drops, key=lambda d: d["relative"])
    # Bracket the cliff: the layer before it, and the two it runs between.
    idx = next(i for i, r in enumerate(rows) if r["layer"] == cliff["from_layer"])
    covering = [r["layer"] for r in rows[max(0, idx - 1):idx + 2]]
    return {"peak_layer": peak["layer"], "peak_live": peak["live_mean"],
            "peak_depth_fraction": peak["depth_fraction"],
            "grid_stride": sorted(strides),
            "steepest_drop": cliff,
            "largest_drop_after_peak": cliff,     # kept: older consumers read this key
            "suggested_work_layers_covering_transition": covering,
            "inherited_a05_set": [12, 16, 20],
            # community-map.md:296-299 recorded, 2026-08-12, that a corpus-only
            # CV independently picks ~0.67 of depth on two model sizes, and
            # noted the implication for L_state had never been applied. This
            # reports where the MEASURED cliff sits on the same scale, so the
            # two can be compared rather than argued about.
            "depth_fraction_implication_0_67": {
                "claimed": 0.67,
                "measured_cliff_depth_fraction": cliff["depth_fraction"],
                "layer_at_0_67": round(0.67 * (rows[-1]["layer"] /
                                               max(rows[-1]["depth_fraction"], 1e-9))),
            }}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens", type=Path, required=True,
                    help="a jlens result JSON (either a single-model probe or "
                         "one with a 'base'/'trained' pair).")
    ap.add_argument("--which", default="auto", choices=("auto", "base", "trained"),
                    help="which model inside a base/trained comparison to profile.")
    ap.add_argument("--n-layer", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(args.jlens.read_text())
    node = d
    which = args.which
    if "layer_stats" not in d:
        if which == "auto":
            which = "base" if "base" in d else "trained"
        node = d.get(which, {})
    if "layer_stats" not in node:
        progress(f"no layer_stats in {args.jlens} (looked at {which!r})")
        return 1

    rows = profile_from_jlens(node["layer_stats"], args.n_layer)
    if not rows:
        progress("layer_stats present but carries no per-head spectra — this "
                 "file predates the direction-count metrics; re-run jlens_probe")
        return 1

    model = node.get("model", d.get("model", "?"))
    progress(f"per-layer state breadth — {model}")
    progress(f"{'L':>3} {'frac':>5} {'live':>6} {'sd':>5} {'min':>4} {'max':>4} "
             f"{'eRank':>6} {'sigma1':>8}")
    for r in rows:
        progress(f"{r['layer']:>3} {r['depth_fraction']:5.2f} {r['live_mean']:6.2f} "
                 f"{r['live_std']:5.2f} {r['live_min']:4d} {r['live_max']:4d} "
                 f"{r['entropy_rank_mean']:6.2f} {r['sigma1_mean']:8.4f}")

    rec = recommend(rows)
    if rec:
        d_ = rec["steepest_drop"]
        progress(f"\npeak breadth at L{rec['peak_layer']} "
                 f"({rec['peak_live']:.2f} live, depth fraction "
                 f"{rec['peak_depth_fraction']:.2f})")
        if d_:
            progress(f"steepest fall L{d_['from_layer']}->L{d_['to_layer']}: "
                     f"{d_['live_before']:.2f} -> {d_['live_after']:.2f} live "
                     f"({d_['relative']:+.0%}) at depth fraction "
                     f"{d_['depth_fraction']:.3f}")
        if rec["grid_stride"] != [1]:
            progress(f"grid stride {rec['grid_stride']} — this names an INTERVAL, "
                     f"not a layer; a finer grid can move it")
        progress(f"inherited A0.5 set {rec['inherited_a05_set']} does not cover it")
        progress(f"set covering the transition instead: "
                 f"{rec['suggested_work_layers_covering_transition']}")

    if args.out is not None:
        save_result(
            args.out, {"model": model, "rows": rows, "recommendation": rec,
                       "source_jlens": str(args.jlens), "n_layer": args.n_layer},
            experiment="layer_profile", hypothesis=["H8", "H26"],
            model=model,
            summary={
                "peak breadth layer": f"L{rec.get('peak_layer')} "
                                      f"({rec.get('peak_live', float('nan')):.2f} live)",
                "steepest drop":
                    (f"L{rec['steepest_drop']['from_layer']}->"
                     f"L{rec['steepest_drop']['to_layer']} "
                     f"{rec['steepest_drop']['relative']:+.0%}"
                     if rec.get("steepest_drop") else "n/a"),
                "set covering the transition":
                    str(rec.get("suggested_work_layers_covering_transition")),
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
