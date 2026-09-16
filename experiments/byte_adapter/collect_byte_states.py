#!/usr/bin/env python3
"""collect_byte_states.py — does the frozen backbone's WKV state carry the NEXT BYTE?

This is the gate on `byte_adapter.py`, and it deliberately does not use it.

`ByteAdapter` maps a byte to a `model_dim` vector through an `nn.Embedding(256, D)`
that nothing in this repository ever trains (no `backward()`, no optimizer, no
loss — the CLI offers only `probe` and `info`), and whose `head` is initialised to
**zeros**, so the decoder emits identical logits for all 256 bytes. Feeding the
backbone through that untrained encoder means feeding it vectors it has never seen,
and whatever comes out the far side measures the encoder, not the state. So the
input side here is ordinary World tokens: the model runs exactly as it always
does, and the only question asked is whether the state it builds linearly predicts
the next byte.

That is the question that actually gates the byte path:

- if the state already carries the next byte, a byte head is worth training and
  `embed` is worth GPU time;
- if it does not, the byte path needs input-side training as well, which is a much
  larger job than fitting a 256-row head, and worth knowing before starting.

**The target is per-BIT, not the byte value.** A byte is a nominal label; fitting a
scalar regression to "byte id" would ask the probe to reconstruct base-2 ordering
that carries no meaning. Eight independent binary targets is the natural
decomposition, and it is the same choice `wkv_linear_probe.py` already made for
XOR.

This script only COLLECTS. The fit is `wkv_linear_probe.py::held_out_linear_probe`,
which already does train-only centering, a permutation floor and a
design-effective-rank read — a null is only interpretable next to both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experiments.rl.loader import load_rwkv7
from experiments._common.results import save_result
from experiments.rl.wkv_linear_probe import held_out_linear_probe


def load_texts(path: Path | None, n: int) -> list[str]:
    if path is not None:
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return lines[:n]
    from experiments.A0_state_probe import prompts as P
    pool = [P.SHORT, P.MEDIUM, P.NARRATIVE, P.LONG]
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]


def collect(loaded, texts: list[str], layers: tuple, max_tokens: int):
    """One row per token position: the flattened WKV state, and the first byte of
    the text the NEXT token decodes to.

    The state is read after the current token, so the target is genuinely a
    prediction rather than a readout of something already consumed.
    """
    tok = loaded.tokenizer
    rows, targets = [], []
    with torch.no_grad():
        for ti, text in enumerate(texts):
            ids = tok.encode(text) if hasattr(tok, "encode") else tok(text)["input_ids"]
            ids = ids[:max_tokens + 1]
            if len(ids) < 2:
                continue
            state = loaded.new_state(batch=1)
            for pos in range(len(ids) - 1):
                x = torch.tensor([[ids[pos]]], dtype=torch.long, device=loaded.device)
                _, state = loaded.forward_stateful(x, state)
                nxt = tok.decode([ids[pos + 1]])
                b = nxt.encode("utf-8", errors="ignore")
                if not b:
                    continue
                stack = loaded.wkv_stack(state)          # backend-agnostic
                rows.append(torch.cat([stack[L].float().flatten() for L in layers]).cpu())
                targets.append(b[0])
            print(f"  [{ti + 1}/{len(texts)}] {len(rows)} rows so far", flush=True)
    return torch.stack(rows), torch.tensor(targets, dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work-layers", default="16,20,24")
    ap.add_argument("--texts", type=Path, default=None,
                    help="one text per line; defaults to the shared prompt pool")
    ap.add_argument("--n-texts", type=int, default=8,
                    help="rows = n_texts * max_tokens, and the fit needs rows "
                         "against ~n_layers*n_head*head_size^2 features — 3 "
                         "layers of a 2.9B is 491520 of them, so a few hundred "
                         "rows is not a null, it is no measurement")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="positions per text — rows = n_texts * max_tokens")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default=None, choices=("peft", "blink"),
                    help="default: peft on cuda, blink on cpu. peft imports "
                         "triton and cannot load without a GPU.")
    ap.add_argument("--dump", type=Path, default=None,
                    help="save the raw design matrix here; without it the fit "
                         "runs in-process and only the result is kept")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.work_layers.split(","))
    backend = args.backend or ("peft" if args.device.startswith("cuda") else "blink")
    loaded = load_rwkv7(args.model, device=args.device, backend=backend)
    texts = load_texts(args.texts, args.n_texts)
    print(f"[collect_byte_states] {len(texts)} texts, layers {layers}, "
          f"<= {args.max_tokens} positions each, backend {backend}")

    X, y = collect(loaded, texts, layers, args.max_tokens)
    print(f"[collect_byte_states] design matrix {tuple(X.shape)}, "
          f"{len(set(y.tolist()))} distinct target bytes")
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X, "y": y, "layers": layers, "model": args.model}, args.dump)
        print(f"[collect_byte_states] raw design -> {args.dump}")

    n_train = int(len(X) * args.train_frac)
    results = {}
    print(f"\n{'target':>10} {'held-out R2':>12} {'shuffled':>10} {'margin':>8} {'>floor?':>8}")
    for bit in range(8):
        t = ((y >> bit) & 1).float()
        if t.std() < 1e-6:
            print(f"{'bit%d' % bit:>10}   constant across the whole sample — skipped")
            continue
        if t[n_train:].std() < 1e-6:
            # R2 divides by the held-out target's variance; a bit that happens to
            # be constant in the test split gives nan, not a null. Bit 7 is
            # always 0 on ASCII text and bits 4-6 go constant easily on a small
            # sample, so this fires for real rather than defensively.
            print(f"{'bit%d' % bit:>10}   constant in the HELD-OUT split — "
                  f"not evaluable, need more rows")
            continue
        r = held_out_linear_probe(X, t, n_train)
        results[f"bit{bit}"] = r
        print(f"{'bit%d' % bit:>10} {r['held_out_r2']:>+12.4f} "
              f"{r.get('shuffled_held_out_mean', float('nan')):>+10.4f} "
              f"{r.get('margin_sd', float('nan')):>7.1f}sd "
              f"{str(r.get('above_shuffled_max')):>8}")

    any_r = next(iter(results.values()), {})
    print(f"\ndesign effective rank {any_r.get('design_effective_rank', float('nan')):.1f} "
          f"over {any_r.get('n_train', 0)} training rows and "
          f"{any_r.get('n_features', 0)} features — a null is only readable if this "
          f"design could have carried a signal at all")

    if args.out is not None:
        best = max((v["held_out_r2"] for v in results.values()), default=float("nan"))
        save_result(
            args.out, {"results": results, "layers": list(layers),
                       "n_rows": int(len(X)), "config": {k: str(v) for k, v in vars(args).items()}},
            experiment="byte_state_readout", hypothesis=["H8"], model=args.model,
            summary={
                "best held-out R2 over 8 bit probes": f"{best:+.4f}",
                "bits above their permutation floor":
                    f"{sum(1 for v in results.values() if v.get('above_shuffled_max'))}/{len(results)}",
                "design effective rank":
                    f"{any_r.get('design_effective_rank', float('nan')):.1f} "
                    f"({any_r.get('n_train', 0)} rows, {any_r.get('n_features', 0)} features)",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
