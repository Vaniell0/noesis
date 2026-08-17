#!/usr/bin/env python3
"""Rebuild the auto-generated section of experiments/RESULTS.md, and every
registered probe's own <name>.info.md.

    python experiments/regenerate_results.py

Scans experiments/**/*.json for `_meta` blocks (written by
`experiments._common.results.save_result`) and rewrites everything below
the `AUTO-GENERATED` marker in RESULTS.md. The hand-written historical
section above the marker is left untouched. Also re-renders each probe
listed in `experiments._common.registry.KNOWN_PROBE_MODULES` into an
`<name>.info.md` next to its module file, from the registry entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import registry
from experiments._common.results import regenerate_index, write_probe_info

if __name__ == "__main__":
    registry.import_known_probes()

    for spec in registry.all_specs():
        info_path = write_probe_info(spec)
        print(f"[regenerate_results] probe info -> {info_path.relative_to(_REPO_ROOT)}")

    n = regenerate_index()
    print(f"[regenerate_results] wrote {n} row(s) to experiments/RESULTS.md")
