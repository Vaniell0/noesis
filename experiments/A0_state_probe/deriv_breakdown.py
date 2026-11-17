"""Where a derivation run's accuracy comes from: by depth x answer (true / false), at M = 0, M = ticks, ticks shuffled.

A model that derives must be right on true items (t reached from t0) as well as on false ones (near misses). A
plausibility shortcut (a mutated term looks locally wrong) shows as false items right at every depth, true items
near the base rate, and little difference between M = 0 and M = ticks.

    python experiments/A0_state_probe/deriv_breakdown.py results/closed_loop_toy/s1e_ce_seed0.pt --split val_laws
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.A0_state_probe import closed_loop_toy as T  # noqa: E402


@torch.no_grad()
def run(model, items, M, shuffle=False, bs=256):
    hits = []
    g = torch.Generator().manual_seed(0)
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        L = max(len(x[0]) for x in chunk)
        toks = torch.full((len(chunk), L), T.PAD, dtype=torch.long)
        for b, x in enumerate(chunk):
            toks[b, L - len(x[0]):] = torch.tensor(x[0])
        perm = torch.randperm(len(chunk), generator=g) if shuffle else None
        pred = model(toks, M, perm=perm)[0].argmax(-1)
        hits += (pred == torch.tensor([x[1] for x in chunk])).tolist()
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", default="val_laws")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for mp in args.models:
        run_args = json.loads(Path(mp).with_suffix(".json").read_text())["args"]
        T.DERIV.clear()
        T.load_deriv(run_args["deriv_file"], run_args["deriv_answer"])
        model = T.ToyRWKV7(run_args["width"], run_args["heads"], run_args["layers"], T.VOCAB + len(T.DERIV_SYMS))
        model.load_state_dict(torch.load(mp, map_location="cpu"), strict=False)   # extra heads (pred_self) unused
        items = T.DERIV[args.split][0]
        true_id = T.DERIV_ID["true"]
        print(f"== {Path(mp).name}  split {args.split}  n {len(items)}")
        for label, M, shuf in (("M0", 0, False), (f"M{run_args['ticks']}", run_args["ticks"], False),
                               ("shuffled", run_args["ticks"], True)):
            h = run(model, items, M, shuf)
            row = []
            for d in range(1, T.KMAX + 1):
                for kind in ("true", "false") if run_args["deriv_answer"] == "truth" else ("all",):
                    sel = [i for i, x in enumerate(items) if x[2] == d and
                           (kind == "all" or (x[1] == true_id) == (kind == "true"))]
                    row.append(f"d{d}{kind[0]} {sum(h[i] for i in sel) / max(1, len(sel)):.2f}")
            print(f"  {label:9s} " + "  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
