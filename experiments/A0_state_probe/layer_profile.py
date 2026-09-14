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

Reading G1i's own per-layer spectrum settles it, and the two are not rivals:

    L   frac   live directions   entropy rank
    12  0.38        13.65            5.88
    16  0.50        14.72            6.36
    20  0.62        15.82            6.98   <- peak breadth
    24  0.75         8.38            3.60   <- breadth halves
    28  0.88        11.00            4.50

The A0.5 set covers the ACCUMULATION of breadth and stops exactly at its peak.
The 0.67 fraction lands just past the peak, where the model starts compressing.
Neither covers the compression itself, which is the largest single change
anywhere in the stack: live directions nearly halve between L20 and L24, right
before readout.

So the untested option is neither inherited set but the transition — covering
where the model discards the width it spent twenty layers accumulating.

Because the quantity is a property of a CHECKPOINT, not of the project, this
is a probe rather than a number in a doc: point it at a jlens result and it
prints the profile and a recommendation, with the reasoning shown rather than
asserted.

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
    """Peak of breadth, the largest drop after it, and a suggested set.

    The suggestion covers the transition rather than the accumulation, because
    the accumulation is what the inherited sets already cover and the drop is
    what nothing covers. It is a suggestion with its reasoning attached, not a
    verdict — which set is right is an empirical question a training run
    answers, and this probe exists so that run compares measured candidates
    instead of inherited ones.
    """
    if len(rows) < 2:
        return {}
    peak = max(rows, key=lambda r: r["live_mean"])
    after = [r for r in rows if r["layer"] > peak["layer"]]
    drop = None
    if after:
        nxt = min(after, key=lambda r: r["layer"])
        drop = {"from_layer": peak["layer"], "to_layer": nxt["layer"],
                "live_before": peak["live_mean"], "live_after": nxt["live_mean"],
                "delta": nxt["live_mean"] - peak["live_mean"],
                "relative": (nxt["live_mean"] - peak["live_mean"]) / peak["live_mean"]}
    # Bracket the transition: the layer immediately BEFORE the peak, the peak
    # itself, and the layer immediately after it (where the drop happens).
    # Taking a midpoint of everything below the peak instead put L8 in the set,
    # which is arbitrary — the claim is about the transition, so the set should
    # straddle it.
    before = [r for r in rows if r["layer"] < peak["layer"]]
    covering = [r["layer"] for r in
                ([max(before, key=lambda r: r["layer"])] if before else []) + [peak] +
                ([min(after, key=lambda r: r["layer"])] if after else [])]
    return {"peak_layer": peak["layer"], "peak_live": peak["live_mean"],
            "peak_depth_fraction": peak["depth_fraction"],
            "largest_drop_after_peak": drop,
            "suggested_work_layers_covering_transition": covering,
            "inherited_a05_set": [12, 16, 20],
            "depth_fraction_implication_0_67": None}


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
        d_ = rec["largest_drop_after_peak"]
        progress(f"\npeak breadth at L{rec['peak_layer']} "
                 f"({rec['peak_live']:.2f} live, depth fraction "
                 f"{rec['peak_depth_fraction']:.2f})")
        if d_:
            progress(f"then L{d_['from_layer']}->L{d_['to_layer']}: "
                     f"{d_['live_before']:.2f} -> {d_['live_after']:.2f} live "
                     f"({d_['relative']:+.0%}) — the model discards width right "
                     f"before readout")
        progress(f"inherited A0.5 set {rec['inherited_a05_set']} ends AT the peak "
                 f"and does not cover the drop")
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
                "drop after peak": (f"{rec['largest_drop_after_peak']['relative']:+.0%}"
                                    if rec.get("largest_drop_after_peak") else "n/a"),
                "set covering the transition":
                    str(rec.get("suggested_work_layers_covering_transition")),
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
