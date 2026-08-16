#!/usr/bin/env python3
"""IPC (Information Processing Capacity) analysis of WKV state trajectory.

Measures how much information about the input token history is linearly
recoverable from WKV states. Based on Dambre et al. 2012 (Sci. Reports).

IPC = Σ_{k,d} R²(x_L(t), P_d(u(t-k)))

where:
  x_L(t)  = WKV state at layer L, timestep t  (random-projected to N_proj dims)
  u(t-k)  = token ID at lag k (normalized to [-1, 1])
  P_d(·)  = Legendre polynomial of degree d

Two quantities:
  MC (linear memory)   = Σ_k R²(x, P_1(u(t-k)))   d=1 only
  IPC_total            = Σ_{k,d} R²(x, P_d(u(t-k)))

Upper bound: IPC_total ≤ N_proj (state dimensionality after projection).
Capacity utilization ratio = IPC_total / N_proj.

H8 signal: high IPC_total = state encodes world-model info about past tokens.
H10 signal: compare IPC at early layers vs late; gap = what readout could decode
            but output head ignores.

Usage:
    python ipc_analysis.py \
        --model ~/.libs/models/rwkv7/rwkv-step9b-e1.pth \
        --n-tokens 256 \
        --max-lag 8 \
        --max-degree 2 \
        --n-proj 128 \
        --layers 0,4,8,12,16,20,24,28,31 \
        --out results/ipc_step9b_e1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from probe import load_model

# ---------------------------------------------------------------------------
# Legendre polynomials (normalized, degree 1..max_degree)
# ---------------------------------------------------------------------------

def legendre(x: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate Legendre polynomial P_d(x), x in [-1, 1]."""
    if degree == 1:
        return x
    if degree == 2:
        return (3 * x**2 - 1) / 2
    if degree == 3:
        return (5 * x**3 - 3 * x) / 2
    if degree == 4:
        return (35 * x**4 - 30 * x**2 + 3) / 8
    raise ValueError(f"degree {degree} not implemented")


# ---------------------------------------------------------------------------
# Ridge regression R²
# ---------------------------------------------------------------------------

def ridge_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1e-3, train_frac: float = 0.8) -> float:
    """R² of ridge regression on held-out test split (fit on train_frac, eval on rest)."""
    T, D = X.shape
    n_train = max(D + 2, int(T * train_frac))
    if n_train >= T - 1:
        # fallback: in-sample when trajectory too short to split
        if T < D + 2:
            return 0.0
        A = X.T @ X + alpha * np.eye(D)
        w = np.linalg.solve(A, X.T @ y)
        y_hat = X @ w
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    A = X_tr.T @ X_tr + alpha * np.eye(D)
    w = np.linalg.solve(A, X_tr.T @ y_tr)
    y_hat = X_te @ w
    ss_res = np.sum((y_te - y_hat) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


# ---------------------------------------------------------------------------
# State trajectory collection
# ---------------------------------------------------------------------------

PROMPT = (
    "<think>\nLet me work through this carefully.\n"
    "The sequence continues: 3, 6, 9, 12, 15. What comes next?\n"
    "Each number increases by 3. So the next is 18.\n"
    "Another: 2, 4, 8, 16, 32. Each doubles. Next is 64.\n"
    "Pattern: A B C A B C. Third element is C. Fifth is B.\n"
    "The matrix has rows [1 2 3], [4 5 6], [7 8 9]. Sum of diagonal: 1+5+9=15.\n"
    "Word hidden in grid: PYTHON. Reading left-to-right, row 2.\n"
    "Arithmetic: 17 + 28 = 45. Carry: 7+8=15, write 5 carry 1. 1+2+1=4. So 45.\n"
    "Binary XOR: 1010 XOR 0110 = 1100. Bit by bit: 1^0=1, 0^1=1, 1^1=0, 0^0=0.\n"
    "Sudoku row [1,_,3,4,5,6,7,8,9] — missing is 2.\n"
    "State encodes all of this as it reads.\n</think>\n"
    "The answer is"
)


def collect_trajectory(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int,
    target_layers: list[int],
    n_proj: int,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Run model for n_tokens steps, collect WKV state projections.

    Returns:
        token_ids: list of generated token IDs (length = n_tokens)
        layer_states: {layer_idx: np.ndarray of shape (n_tokens, n_proj)}
    """
    import os
    os.environ.setdefault("RWKV_V7_ON", "1")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "0")

    rng = np.random.default_rng(seed)

    # Build random projection matrices per layer (fixed across time)
    n_state = None
    proj_matrices: dict[int, np.ndarray] = {}

    token_ids: list[int] = []
    layer_states: dict[int, list[np.ndarray]] = {l: [] for l in target_layers}

    # Prefill
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    state = None
    for tok in input_ids:
        out, state = model.forward([tok], state)

    # Determine state dim from first layer after prefill
    def _wkv(s, l):
        return s[3 * l + 1]  # [n_head, head_size, head_size]

    sample_wkv = _wkv(state, target_layers[0])
    n_state = int(np.prod(sample_wkv.shape))

    for l in target_layers:
        # Gaussian random projection: (n_state, n_proj), columns normalized
        P = rng.standard_normal((n_state, n_proj)).astype(np.float32)
        P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
        proj_matrices[l] = P

    # Autoregressive generation
    last_token = input_ids[-1]
    for step in range(n_tokens):
        out, state = model.forward([last_token], state)
        logits = out.float().numpy() if hasattr(out, 'numpy') else np.array(out)
        next_token = int(np.argmax(logits))
        token_ids.append(next_token)
        last_token = next_token

        for l in target_layers:
            wkv = _wkv(state, l).float().numpy().ravel()  # (n_state,)
            proj = wkv @ proj_matrices[l]  # (n_proj,)
            layer_states[l].append(proj)

        if step % 32 == 0:
            print(f"  step {step+1}/{n_tokens} tok={next_token}", flush=True)

    return token_ids, {l: np.stack(layer_states[l]) for l in target_layers}


# ---------------------------------------------------------------------------
# IPC computation
# ---------------------------------------------------------------------------

def compute_ipc(
    token_ids: list[int],
    layer_states: dict[int, np.ndarray],
    max_lag: int,
    max_degree: int,
    vocab_size: int = 65536,
) -> dict:
    """Compute IPC per layer.

    Returns dict with per-layer results:
      {layer: {lag: {degree: r2}}}
    and summary MC/IPC_total per layer.
    """
    T = len(token_ids)
    # Normalize token IDs to [-1, 1]
    u = np.array(token_ids, dtype=np.float32)
    u = (u / (vocab_size - 1)) * 2.0 - 1.0  # in [-1, 1]

    basis_bound = max_lag * max_degree  # max observable IPC with this lag/degree set

    results = {}
    for layer, X in layer_states.items():
        # X: (T, n_proj)
        layer_result = {"by_lag_degree": {}, "MC": 0.0, "IPC_total": 0.0}
        for k in range(1, max_lag + 1):
            if T - k < 10:
                break
            u_lag = u[:T - k]       # target at lag k: u(t-k) for t=k..T-1
            X_cur = X[k:]           # state at t=k..T-1
            for d in range(1, max_degree + 1):
                target = legendre(u_lag, d)
                r2 = max(0.0, ridge_r2(X_cur, target))
                layer_result["by_lag_degree"].setdefault(k, {})[d] = round(r2, 5)
                layer_result["IPC_total"] += r2
                if d == 1:
                    layer_result["MC"] += r2
        layer_result["IPC_total"] = round(layer_result["IPC_total"], 4)
        layer_result["MC"] = round(layer_result["MC"], 4)
        results[layer] = layer_result

    return results, basis_bound


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--max-degree", type=int, default=2)
    ap.add_argument("--n-proj", type=int, default=128,
                    help="Random projection dimensionality (IPC upper bound)")
    ap.add_argument("--layers", default="0,4,8,16,24,31",
                    help="Comma-separated layer indices")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="results/ipc_analysis.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]
    n_proj = args.n_proj

    print(f"[ipc] model={args.model}")
    print(f"[ipc] layers={target_layers}  n_proj={n_proj}")
    print(f"[ipc] n_tokens={args.n_tokens}  max_lag={args.max_lag}  max_degree={args.max_degree}")
    print(f"[ipc] IPC upper bound per layer = {n_proj}")

    print("[ipc] loading model…")
    model, tokenizer = load_model(args.model)

    print("[ipc] collecting trajectory…")
    token_ids, layer_states = collect_trajectory(
        model, tokenizer, args.prompt,
        n_tokens=args.n_tokens,
        target_layers=target_layers,
        n_proj=n_proj,
        seed=args.seed,
    )
    print(f"[ipc] collected {len(token_ids)} tokens across {len(target_layers)} layers")

    print("[ipc] computing IPC…")
    ipc, basis_bound = compute_ipc(token_ids, layer_states, args.max_lag, args.max_degree)

    # Summary table
    print("\n[ipc] === RESULTS ===")
    print(f"  basis_bound = max_lag({args.max_lag}) × max_degree({args.max_degree}) = {basis_bound}")
    print(f"  n_proj (Dambre theoretical bound) = {n_proj}")
    print(f"  R² evaluated on held-out 20% of trajectory")
    print(f"{'Layer':>6} {'MC':>8} {'IPC':>8} {'Util%':>8}  (of basis_bound={basis_bound})")
    for l in sorted(ipc):
        mc = ipc[l]["MC"]
        total = ipc[l]["IPC_total"]
        util = 100.0 * total / basis_bound
        print(f"{l:>6} {mc:>8.3f} {total:>8.3f} {util:>7.1f}%")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "n_tokens": args.n_tokens,
        "max_lag": args.max_lag,
        "max_degree": args.max_degree,
        "n_proj": n_proj,
        "basis_ipc_bound": basis_bound,
        "dambre_theoretical_bound": n_proj,
        "note": "R2 evaluated on held-out 20% of trajectory. basis_ipc_bound = max_lag * max_degree.",
        "layers": target_layers,
        "results": {str(l): ipc[l] for l in sorted(ipc)},
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[ipc] saved → {out_path}")


if __name__ == "__main__":
    main()
