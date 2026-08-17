#!/usr/bin/env python3
"""mlp_probe.py — nonlinear IPC probe via 2-layer MLP.

Replaces ridge regression in ipc_analysis.py with a 2-layer MLP to measure
information that is present in WKV state but not linearly decodable.

Linear IPC (ridge) ≈ 0 on held-out data (G1i, 2026-08-16) → H8: state encodes
nonlinearly. This probe asks: is the information there nonlinearly?

Architecture:
    MLP: Linear(n_proj, 256) → ReLU → Linear(256, 1)
    Loss: MSE  |  Optimizer: Adam 1e-3  |  Epochs: 200  |  Train/val: 80/20

Usage:
    python3 experiments/A0_state_probe/mlp_probe.py \\
        --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \\
        --n-tokens 256 --layers 0,4,8,16,24,31 \\
        --out experiments/A0_state_probe/results/mlp_ipc_g1i_base.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Bootstrap repo root onto sys.path before any `experiments.*` absolute
# import — same reason as ipc_analysis.py: this file is run both as a bare
# script (only its own directory on sys.path) and imported as
# `experiments.A0_state_probe.mlp_probe` (by experiments/run.py's probe
# registry), which needs the repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.A0_state_probe.ipc_analysis import collect_trajectory, legendre, PROMPT
from experiments.A0_state_probe.probe import load_model
from experiments._common import registry


# ── MLP head ─────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_r2(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    train_frac: float = 0.8,
) -> float:
    """Train a 2-layer MLP and return held-out R²."""
    T = len(y)
    n_train = max(4, int(T * train_frac))
    if n_train >= T - 1:
        return 0.0

    X_t = torch.from_numpy(X[:n_train]).float()
    y_t = torch.from_numpy(y[:n_train]).float()
    X_v = torch.from_numpy(X[n_train:]).float()
    y_v = torch.from_numpy(y[n_train:]).float()

    model = _MLP(X.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # experiments._common.model.load_model (the loader every caller of this
    # function goes through) sets torch.set_grad_enabled(False) globally —
    # correct for the read-only probes it's meant for, wrong here, where an
    # MLP actually needs to train. Without this, .backward() below raises
    # "element 0 of tensors does not require grad and does not have a
    # grad_fn" — found 2026-08-17 running this function for the first time
    # ever (mlp_probe.py was written 2026-08-16, never successfully run).
    with torch.enable_grad():
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss_fn(model(X_t), y_t).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        y_hat = model(X_v).numpy()
    y_v_np = y_v.numpy()
    ss_res = np.sum((y_v_np - y_hat) ** 2)
    ss_tot = np.sum((y_v_np - y_v_np.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


# ── IPC via MLP ───────────────────────────────────────────────────────────────

def compute_mlp_ipc(
    token_ids: list[int],
    layer_states: dict[int, np.ndarray],
    max_lag: int,
    max_degree: int,
    vocab_size: int = 65536,
    hidden: int = 256,
    epochs: int = 200,
) -> dict:
    T = len(token_ids)
    u = np.array(token_ids, dtype=np.float32)
    u = (u / (vocab_size - 1)) * 2.0 - 1.0

    basis_bound = max_lag * max_degree
    results = {}
    for layer, X in layer_states.items():
        layer_result = {"by_lag_degree": {}, "MC": 0.0, "IPC_total": 0.0}
        for k in range(1, max_lag + 1):
            if T - k < 10:
                break
            u_lag = u[:T - k]
            X_cur = X[k:]
            for d in range(1, max_degree + 1):
                target = legendre(u_lag, d)
                r2 = mlp_r2(X_cur, target, hidden=hidden, epochs=epochs)
                layer_result["by_lag_degree"].setdefault(k, {})[d] = round(r2, 5)
                layer_result["IPC_total"] += r2
                if d == 1:
                    layer_result["MC"] += r2
                print(f"    L{layer} lag={k} deg={d}  r2={r2:.4f}", flush=True)
        layer_result["IPC_total"] = round(layer_result["IPC_total"], 4)
        layer_result["MC"] = round(layer_result["MC"], 4)
        results[layer] = layer_result
    return results, basis_bound


# ── registered probe ─────────────────────────────────────────────────────────
# Shared args, then the model-independent core logic. `run()` assumes
# `model`/`tokenizer` are already loaded — see ipc_analysis.py's equivalent
# split for the convention this follows.

def _add_mlp_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--max-degree", type=int, default=2)
    ap.add_argument("--n-proj", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--layers", default="0,4,8,16,24,31")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trajectory-in", type=pathlib.Path)


@registry.probe(
    "mlp_ipc",
    hypothesis=["H8"],
    description="Nonlinear IPC via 2-layer MLP replacing ridge regression — "
                "measures whether WKV state carries information the linear "
                "ipc probe can't see.",
    add_args=_add_mlp_args,
)
def run(model, tokenizer, args: argparse.Namespace) -> dict:
    target_layers = [int(x) for x in args.layers.split(",")]

    teacher_forced_tokens: Optional[list[int]] = None
    if args.trajectory_in is not None:
        payload = json.loads(args.trajectory_in.read_text())
        teacher_forced_tokens = payload["token_ids"] if isinstance(payload, dict) else payload

    print(f"[mlp_probe] layers={target_layers}  n_proj={args.n_proj}")
    print(f"[mlp_probe] n_tokens={args.n_tokens}  max_lag={args.max_lag}  deg={args.max_degree}")
    print(f"[mlp_probe] MLP hidden={args.hidden}  epochs={args.epochs}")

    token_ids, layer_states = collect_trajectory(
        model, tokenizer, args.prompt,
        n_tokens=args.n_tokens,
        target_layers=target_layers,
        n_proj=args.n_proj,
        device=args.device,
        seed=args.seed,
        teacher_forced_tokens=teacher_forced_tokens,
    )
    print(f"[mlp_probe] collected {len(token_ids)} tokens")

    print("[mlp_probe] training MLP probes…")
    ipc, basis_bound = compute_mlp_ipc(
        token_ids, layer_states,
        max_lag=args.max_lag,
        max_degree=args.max_degree,
        hidden=args.hidden,
        epochs=args.epochs,
    )

    print(f"\n[mlp_probe] === RESULTS ===")
    print(f"  basis_bound={basis_bound}  n_proj={args.n_proj}")
    print(f"{'Layer':>6} {'MC':>8} {'IPC':>8}")
    for l in sorted(ipc):
        print(f"{l:>6} {ipc[l]['MC']:>8.3f} {ipc[l]['IPC_total']:>8.3f}")

    mean_ipc = sum(ipc[l]["IPC_total"] for l in ipc) / len(ipc) if ipc else 0.0
    peak_layer = max(ipc, key=lambda l: ipc[l]["IPC_total"]) if ipc else None

    return {
        "model": getattr(args, "model", None),
        "probe": "mlp_2layer",
        "hidden": args.hidden,
        "epochs": args.epochs,
        "n_tokens": args.n_tokens,
        "max_lag": args.max_lag,
        "max_degree": args.max_degree,
        "n_proj": args.n_proj,
        "basis_ipc_bound": basis_bound,
        "note": "Nonlinear IPC via 2-layer MLP. R2 on held-out 20%.",
        "layers": target_layers,
        "results": {str(l): ipc[l] for l in sorted(ipc)},
        "_summary": {
            f"Mean nonlinear IPC_total (L{','.join(str(l) for l in sorted(ipc))})": f"{mean_ipc:.3f} / {basis_bound}",
            "Peak layer": f"L{peak_layer} ({ipc[peak_layer]['IPC_total']:.3f})" if peak_layer is not None else "—",
        },
    }


# ── standalone entry point ───────────────────────────────────────────────────
# `python mlp_probe.py --model ... --out ...`. Unchanged CLI surface from
# before the registry refactor (2026-08-17).

def main():
    from experiments._common.results import save_result

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out", default="results/mlp_ipc.json")
    _add_mlp_args(ap)
    args = ap.parse_args()

    print(f"[mlp_probe] model={args.model}")
    model, tokenizer = load_model(args.model, device=args.device)

    out = run(model, tokenizer, args)

    out_path = save_result(args.out, out, experiment="mlp_ipc", hypothesis=["H8"], script=__file__)
    print(f"\n[mlp_probe] saved -> {out_path}")


if __name__ == "__main__":
    main()
