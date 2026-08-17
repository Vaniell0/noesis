#!/usr/bin/env python3
"""Load one model, run a selected battery of probes against it.

    python experiments/run.py --list
    python experiments/run.py --model ~/.libs/models/rwkv7/rwkv7-g1i-2.9b-*.pth \\
        --device cpu --tests ipc,think_geometry --out-dir experiments/_common/results/adhoc

Each `--tests` name must be a probe registered via `experiments._common.registry.probe`
in a module listed in `experiments._common.registry.KNOWN_PROBE_MODULES` — add yours
there once it registers. The model is loaded exactly once
(`experiments._common.model.load_model`) and handed to every selected probe in turn;
results are written via `_common.results.save_result` (which stamps them for
`experiments/regenerate_results.py` to pick up).
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import registry
from experiments._common.model import load_model
from experiments._common.results import save_result


def main() -> None:
    # conflict_handler="resolve": probes that share underlying logic (e.g.
    # ipc_analysis.py and mlp_probe.py both call collect_trajectory) share
    # flag names with the *same* meaning — --n-tokens means the same thing
    # in both, so one shared flag is correct, not a conflict. This assumes
    # a shared flag name always means shared semantics across probes; if a
    # future probe needs a same-named flag with different meaning, that
    # probe needs a prefixed flag name instead, not a parser-level fix.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                  conflict_handler="resolve")
    ap.add_argument("--model", help="checkpoint path or HF owner/repo:file.pth")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--tests", help="comma-separated probe names, or 'all'")
    ap.add_argument("--list", action="store_true", help="list registered probes and exit")
    ap.add_argument("--out-dir", default="experiments/_common/results/adhoc",
                     help="where each probe's <name>.json is written")

    registry.import_known_probes()

    # Let every registered probe attach its own flags to the shared parser,
    # so e.g. `--tests ipc` also exposes `--n-tokens`/`--max-lag`/etc.
    for spec in registry.all_specs():
        if spec.add_args is not None:
            group = ap.add_argument_group(f"{spec.name} ({', '.join(spec.hypothesis) or 'no H'})")
            spec.add_args(group)

    args = ap.parse_args()

    if args.list:
        if not registry.names():
            print("(no probes registered — check registry.KNOWN_PROBE_MODULES)")
            return
        for spec in registry.all_specs():
            h = ",".join(spec.hypothesis) or "—"
            print(f"{spec.name:20s} [{h:8s}] {spec.description}")
        return

    if not args.model:
        ap.error("--model is required unless --list")
    if not args.tests:
        ap.error("--tests is required (comma-separated names, or 'all') unless --list")

    selected = registry.names() if args.tests == "all" else [t.strip() for t in args.tests.split(",")]
    unknown = [t for t in selected if t not in registry.names()]
    if unknown:
        ap.error(f"unknown probe(s): {', '.join(unknown)} — known: {', '.join(registry.names())}")

    print(f"[run] loading model once: {args.model} (device={args.device})")
    model, tokenizer = load_model(args.model, device=args.device)
    print(f"[run] model loaded — running {len(selected)} probe(s): {', '.join(selected)}")

    out_dir = Path(args.out_dir)
    for name in selected:
        spec = registry.get(name)
        print(f"[run] --- {name} ---")
        result = spec.fn(model, tokenizer, args)
        out_path = save_result(
            out_dir / f"{name}.json", result,
            experiment=name, hypothesis=spec.hypothesis, model=args.model,
            script=inspect.getfile(spec.fn),
        )
        print(f"[run]     -> {out_path}")


if __name__ == "__main__":
    main()
