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

Measured basis bound: max_lag × max_degree (the basis actually evaluated).
The separate Dambre theoretical bound remains N_proj (state dimensionality
after projection). Capacity utilization is reported against the basis bound.

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
from typing import Optional, Sequence

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


def load_token_trajectory(path: pathlib.Path) -> tuple[list[int], Optional[str]]:
    """Load a saved post-prompt token trajectory.

    The native format is ``{"token_ids": [...], "prompt": "..."}``, but
    accepting a bare JSON list keeps the loader convenient for hand-built
    fixtures. The optional prompt is returned so callers can reject a
    trajectory recorded with a different prompt.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_prompt: Optional[str] = None
    if isinstance(payload, dict):
        raw_token_ids = payload.get("token_ids")
        if isinstance(payload.get("prompt"), str):
            saved_prompt = payload["prompt"]
    else:
        raw_token_ids = payload

    if not isinstance(raw_token_ids, list):
        raise ValueError(f"trajectory file {path} must contain a token_ids list")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in raw_token_ids
    ):
        raise ValueError(f"trajectory file {path} contains an invalid token id")
    return [int(token_id) for token_id in raw_token_ids], saved_prompt


def save_token_trajectory(path: pathlib.Path, token_ids: Sequence[int], prompt: str) -> None:
    """Save the exact post-prompt token sequence used by a probe run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_ids": [int(token_id) for token_id in token_ids],
        "n_tokens": len(token_ids),
        "prompt": prompt,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _argmax_token(logits: object) -> int:
    """Take argmax without copying a CUDA vocabulary-sized tensor to CPU."""
    if isinstance(logits, torch.Tensor):
        return int(torch.argmax(logits.reshape(-1)).item())
    return int(np.argmax(np.asarray(logits).reshape(-1)))


def _project_wkv(wkv: torch.Tensor, projection: object, device: str) -> np.ndarray:
    """Project one WKV tensor, moving only the compact result off CUDA."""
    if device == "cuda":
        # Keep the large WKV tensor and projection on the GPU. The returned
        # vector is only n_proj elements, so this is the sole device-to-host
        # transfer in the projection path.
        projected = torch.matmul(
            wkv.detach().to(dtype=torch.float32).reshape(-1),
            projection,
        )
        return projected.detach().cpu().numpy()

    # Preserve the existing CPU path and its NumPy projection semantics.
    return wkv.detach().to(dtype=torch.float32).numpy().reshape(-1) @ projection


def collect_trajectory(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int,
    target_layers: list[int],
    n_proj: int,
    device: str = "cpu",
    seed: int = 42,
    teacher_forced_tokens: Optional[Sequence[int]] = None,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Run model for n_tokens steps, collect WKV state projections.

    Returns:
        token_ids: list of generated token IDs (length = n_tokens)
        layer_states: {layer_idx: np.ndarray of shape (n_tokens, n_proj)}

    If ``teacher_forced_tokens`` is provided, those exact post-prompt tokens
    are fed to the model instead of sampling each checkpoint's own sequence.
    This is the comparability path for cross-checkpoint runs.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if teacher_forced_tokens is not None and len(teacher_forced_tokens) != n_tokens:
        raise ValueError(
            "teacher_forced_tokens length must match n_tokens "
            f"({len(teacher_forced_tokens)} != {n_tokens})"
        )

    rng = np.random.default_rng(seed)

    # Build random projection matrices per layer (fixed across time)
    n_state = None
    proj_matrices: dict[int, object] = {}

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
        if device == "cuda":
            proj_matrices[l] = torch.from_numpy(P).to(device="cuda")
        else:
            proj_matrices[l] = P

    # The final prompt token has already been consumed by the prefill above.
    # Start from its logits so it is not accidentally fed a second time.
    next_logits = out
    for step in range(n_tokens):
        if teacher_forced_tokens is None:
            next_token = _argmax_token(next_logits)
        else:
            next_token = int(teacher_forced_tokens[step])

        next_logits, state = model.forward([next_token], state)
        token_ids.append(next_token)

        for l in target_layers:
            wkv = _wkv(state, l)
            proj = _project_wkv(wkv, proj_matrices[l], device)
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
                    help="Random projection dimensionality (Dambre theoretical bound)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="Inference device (default: cpu)")
    ap.add_argument("--layers", default="0,4,8,16,24,31",
                    help="Comma-separated layer indices")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="results/ipc_analysis.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trajectory-in", type=pathlib.Path,
                    help="JSON token trajectory to teacher-force across checkpoints")
    ap.add_argument("--trajectory-out", type=pathlib.Path,
                    help="Save the generated/used token trajectory as JSON")
    args = ap.parse_args()

    teacher_forced_tokens: Optional[list[int]] = None
    if args.trajectory_in is not None:
        teacher_forced_tokens, saved_prompt = load_token_trajectory(args.trajectory_in)
        if saved_prompt is not None and saved_prompt != args.prompt:
            ap.error("--trajectory-in was recorded with a different prompt")
        if len(teacher_forced_tokens) != args.n_tokens:
            ap.error(
                "--trajectory-in length must match --n-tokens "
                f"({len(teacher_forced_tokens)} != {args.n_tokens})"
            )

    target_layers = [int(x) for x in args.layers.split(",")]
    n_proj = args.n_proj

    print(f"[ipc] model={args.model}")
    print(f"[ipc] layers={target_layers}  n_proj={n_proj}")
    print(f"[ipc] n_tokens={args.n_tokens}  max_lag={args.max_lag}  max_degree={args.max_degree}")
    print(f"[ipc] device={args.device}")
    print(f"[ipc] n_proj (Dambre theoretical bound) = {n_proj}")
    print(f"[ipc] trajectory mode = {'teacher-forced' if teacher_forced_tokens is not None else 'autoregressive'}")

    print("[ipc] loading model…")
    model, tokenizer = load_model(args.model, device=args.device)

    print("[ipc] collecting trajectory…")
    token_ids, layer_states = collect_trajectory(
        model, tokenizer, args.prompt,
        n_tokens=args.n_tokens,
        target_layers=target_layers,
        n_proj=n_proj,
        device=args.device,
        seed=args.seed,
        teacher_forced_tokens=teacher_forced_tokens,
    )
    print(f"[ipc] collected {len(token_ids)} tokens across {len(target_layers)} layers")

    if args.trajectory_out is not None:
        save_token_trajectory(args.trajectory_out, token_ids, args.prompt)
        print(f"[ipc] saved trajectory -> {args.trajectory_out}")

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
        "device": args.device,
        "basis_ipc_bound": basis_bound,
        "dambre_theoretical_bound": n_proj,
        "note": "R2 evaluated on held-out 20% of trajectory. basis_ipc_bound = max_lag * max_degree.",
        "layers": target_layers,
        "token_ids": token_ids,
        "results": {str(l): ipc[l] for l in sorted(ipc)},
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[ipc] saved -> {out_path}")


if __name__ == "__main__":
    main()
