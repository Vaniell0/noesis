#!/usr/bin/env python3
"""Cross-reference which hypotheses and which probes/experiments intersect.

    python experiments/hyp_xref.py

Joins two sources that already exist and were never combined:
- `registry.py`'s `@registry.probe(name, hypothesis=[...])` — declared
  once, per probe, at registration time ("what this measurement is for").
- `results.py`'s per-run `_meta.hypothesis` — declared per actual run,
  and can broaden a probe's reach beyond its registration (e.g. `rlens`
  is registered for H8 but individual runs have also tagged H9, when
  the run's author judged the number relevant there too).

No new declarations needed anywhere; this only reads what both sides
already record and unions the two views into one table each way.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import registry
from experiments._common.results import _iter_meta_stamped


def build_xref(root: Path | None = None):
    root = root or (_REPO_ROOT / "experiments")
    registry.import_known_probes()

    h_to_probes: dict[str, set[str]] = defaultdict(set)
    probe_to_h: dict[str, set[str]] = defaultdict(set)

    # Registration-time declarations (probe -> H)
    for spec in registry.all_specs():
        for h in spec.hypothesis:
            h_to_probes[h].add(spec.name)
            probe_to_h[spec.name].add(h)

    # Run-time declarations (experiment name -> H), from every stamped result
    h_to_experiments: dict[str, set[str]] = defaultdict(set)
    experiment_to_h: dict[str, set[str]] = defaultdict(set)
    for _path, meta in _iter_meta_stamped(root):
        exp = meta.get("experiment")
        if not exp:
            continue
        for h in meta.get("hypothesis") or []:
            h_to_experiments[h].add(exp)
            experiment_to_h[exp].add(h)
            # union into the probe view too, so a run-time-only H shows
            # up next to the registration-time ones for the same name
            h_to_probes[h].add(exp)
            probe_to_h[exp].add(h)

    return h_to_probes, probe_to_h, h_to_experiments, experiment_to_h


if __name__ == "__main__":
    h_to_probes, probe_to_h, h_to_experiments, experiment_to_h = build_xref()

    print("=== H -> probes/experiments ===\n")
    for h in sorted(h_to_probes, key=lambda x: (len(x), x)):
        names = ", ".join(sorted(h_to_probes[h]))
        run_only = h_to_experiments.get(h, set()) - {
            s.name for s in registry.all_specs() if h in s.hypothesis
        }
        flag = f"  (run-time only: {', '.join(sorted(run_only))})" if run_only else ""
        print(f"  {h:6s} <- {names}{flag}")

    print("\n=== probe/experiment -> H ===\n")
    for name in sorted(probe_to_h):
        hs = ", ".join(sorted(probe_to_h[name]))
        reg_hs = set()
        try:
            reg_hs = set(registry.get(name).hypothesis)
        except KeyError:
            pass
        extra = set(probe_to_h[name]) - reg_hs
        flag = f"  (beyond registration: {', '.join(sorted(extra))})" if reg_hs and extra else ""
        print(f"  {name:20s} -> {hs}{flag}")
