#!/usr/bin/env python3
"""micro_wkv.py — a minimal, faithful RWKV-7 state-recurrence cell, plus
a trainable controller, built to test H25 ("WKV as a learnable physical
computer" — see hypotheses/H25.md's 2026-08-23 "Cheap pre-test" section)
at a scale small enough to check every claim directly against the
algebra instead of inferring it statistically from a trained model.

This is NOT a simplified/approximate RWKV-7 — the recurrence below is
the same delta-rule update the real model uses (rwkvfla/ops/rwkv7/
recurrent_naive.py::naive_recurrent_rwkv7, and docs/rwkv7-mechanics.md's
derivation of it), just single-head and small head_size.

Real RWKV-7 per-step update (state: [N, V], here N == V == head_size,
a square per-head state matrix):

    kk_t   = k_t / ||k_t||                      (L2-normalized key direction)
    decay  = exp(-exp(w_t))                      (per-channel retention, docs/rwkv7-mechanics.md §3-4)
    state  = state * decay
             - (state @ kk_t) ⊗ (kk_t * a_t)     (delta-rule erase/rewrite term)
             + k_t ⊗ v_t                         (rank-1 write)
    out    = r_t @ state                          (readout)

This matches `RUN_RWKV7_INFCTX(r, k, v, w, -kk, kk * a, state)` traced
in `experiments/rl/state_trajectory_probe.py`'s monkeypatch of
`RWKV_Tmix_x070_infctx.forward` — plugging (a=-kk, b=kk*a_gate) into the
reference implementation's generic `state*decay + state@a⊗b + k⊗v` gives
exactly the three lines above.

Controller design history (2026-08-23) — kept because the mistake is as
important as the fix: the first version fed the controller BOTH task
operands (a, b) at EVERY recurrence step. That completely undermines any
claim about the recurrence: the controller has direct, un-mediated
access to both operands throughout, so it can compute a·b itself (MLPs
approximate smooth bilinear functions easily) and simply write the
result into a state slot for the readout to find — the state's role is
reduced to "final-answer transport," and a clean linearly-decodable
result proves nothing about whether the RECURRENCE did the computing.
**Fixed here**: `a` is revealed only at step 0, `b` only at the final
step, with genuinely blank (constant, operand-independent) input on
every step in between — the state is the only channel by which `a` can
possibly reach the final step. See hypotheses/H25.md for the full
narrative, including the counterfactual `a_gate` ablation this enables
(train, then force `a_gate` to a fixed 0 or 1 at inference — does
performance depend on the delta-rule erase/rewrite mechanism
specifically, or does it survive the ablation because the computation
is actually happening in the readout's own bilinear structure,
`r_t @ state`, between the current step's r/decay and the accumulated
state? On this task: it survives — see hypotheses/H25.md).

stable_rank tracking reuses `metrics.py::stable_rank` (same (‖A‖_F/‖A‖_2)²
definition already used by jlens_probe.py/rlens_probe.py/think_geometry.py
across this project) rather than a new ad-hoc "effective rank."

Usage:
    python experiments/A0_state_probe/micro_wkv.py --verify
    python experiments/A0_state_probe/micro_wkv.py --train multiply --steps 4 --head-size 8
    python experiments/A0_state_probe/micro_wkv.py --train add --steps 4 --head-size 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import stable_rank  # noqa: E402 — reuse the project's one definition


# --------------------------------------------------------------------------- #
# The cell (step 1, verified 2026-08-23)
# --------------------------------------------------------------------------- #

def decay_from_logit(w: torch.Tensor) -> torch.Tensor:
    """retention = exp(-exp(w)) — the actual per-channel decay multiplier,
    NOT exp(w). See docs/rwkv7-mechanics.md §3-4 for the full derivation
    and the earlier (wrong) exp(w) mistake this project already made once."""
    return torch.exp(-torch.exp(w))


def micro_wkv_step(state: torch.Tensor, r: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor, w: torch.Tensor, a_gate: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """One real RWKV-7 delta-rule step, single head, batched over the
    leading dim (state: [B, N, V]; r,k,v,w,a_gate: [B, N] or [B, V] for v).
    Returns (output [B, N], new_state [B, N, V])."""
    kk = F.normalize(k, p=2.0, dim=-1, eps=1e-8)
    decay = decay_from_logit(w)
    state_at_kk = torch.einsum("bnv,bn->bv", state, kk)  # state @ kk, per-batch
    erase_rewrite = torch.einsum("bv,bn->bnv", state_at_kk, kk * a_gate)
    write = torch.einsum("bn,bv->bnv", k, v)
    new_state = state * decay.unsqueeze(-1) - erase_rewrite + write
    out = torch.einsum("bn,bnv->bv", r, new_state)
    return out, new_state


# --------------------------------------------------------------------------- #
# Step 1 verification
# --------------------------------------------------------------------------- #

def _random_controls(n_steps: int, head_size: int, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    controls = []
    for _ in range(n_steps):
        controls.append({
            "r": torch.randn(1, head_size, generator=g),
            "k": torch.randn(1, head_size, generator=g),
            "v": torch.randn(1, head_size, generator=g),
            "w": -0.5 - torch.rand(1, head_size, generator=g) * 3.0,
            "a_gate": torch.sigmoid(torch.randn(1, head_size, generator=g)),
        })
    return controls


def verify() -> None:
    """Sanity checks against docs/rwkv7-mechanics.md's derivation — not
    a training run, just confirming this file's recurrence matches the
    real one's known properties before building anything on top of it."""
    torch.manual_seed(0)
    head_size = 8

    print("[1] Hand-checked single step (zero initial state, zero a_gate — "
          "erase/rewrite term vanishes, should reduce to a pure rank-1 write):")
    state0 = torch.zeros(1, head_size, head_size)
    k = torch.randn(1, head_size)
    v = torch.randn(1, head_size)
    r = torch.randn(1, head_size)
    w = torch.full((1, head_size), -1.0)
    a_gate = torch.zeros(1, head_size)
    out, state1 = micro_wkv_step(state0, r, k, v, w, a_gate)
    expected_state1 = torch.einsum("bn,bv->bnv", k, v)
    max_diff = (state1 - expected_state1).abs().max().item()
    print(f"    max|state1 - k⊗v| = {max_diff:.2e} (expect ~0)")
    assert max_diff < 1e-5, "zero-state/zero-gate step should reduce to a pure rank-1 write"

    print("[2] Retention bounds — decay = exp(-exp(w)), w soft-clamped to w <= -0.5 "
          "(docs/rwkv7-mechanics.md's soft-clamp FLOOR on retention, not a ceiling):")
    w_grid = torch.linspace(-0.5, -8.0, 200)  # w=-0.5 is the LEAST negative allowed value
    decay_grid = decay_from_logit(w_grid)
    print(f"    decay range over w in [-8, -0.5]: [{decay_grid.min():.4f}, {decay_grid.max():.4f}]")
    # First draft of this check had the direction backwards (asserted max<=0.5453) —
    # caught by running it, not by re-reading: retention is a DECREASING function of
    # w, so w=-0.5 (the least-negative allowed value) gives the FLOOR (~0.5452), and
    # retention rises toward 1 as w -> -inf, uncapped.
    assert abs(decay_grid.min().item() - 0.5452) < 1e-3, (
        f"expected the floor (min over this w range, at w=-0.5) to be ~0.5452, "
        f"got {decay_grid.min().item()}")
    assert decay_grid.max().item() < 1.0
    print("    confirmed: retention FLOOR ~0.5452 at w=-0.5, rising toward (but never "
          "reaching) 1 as w -> -inf — uncapped from above. See docs/rwkv7-mechanics.md.")

    print("[3] No true inverse exists — running a step then attempting the 'obvious' "
          "undo (subtract the write, divide by decay) should NOT recover the prior "
          "state once the erase/rewrite term is active (confirms the non-invertibility "
          "caveat in docs/rl-track.md's Phase 2 section is a real property, not an "
          "oversimplification):")
    c = _random_controls(1, head_size, seed=1)[0]
    state_before = torch.randn(1, head_size, head_size)
    _, state_after = micro_wkv_step(state_before, c["r"], c["k"], c["v"], c["w"], c["a_gate"])
    decay = decay_from_logit(c["w"])
    naive_undo = (state_after - torch.einsum("bn,bv->bnv", c["k"], c["v"])) / decay.unsqueeze(-1)
    recon_error = (naive_undo - state_before).abs().max().item()
    print(f"    naive-undo reconstruction error: {recon_error:.4f} (expect >> 0)")
    assert recon_error > 1e-3, "expected the naive undo to fail once a_gate is active"

    print("\nAll checks passed. micro_wkv_step matches the documented RWKV-7 "
          "recurrence's known properties at toy scale.")


# --------------------------------------------------------------------------- #
# Step 2: trainable controller — can the physics compute?
# --------------------------------------------------------------------------- #

class StagedController(nn.Module):
    """`a` is revealed only at step 0, `b` only at the final step; every
    step in between sees a constant, operand-independent input. The
    state is the ONLY channel `a` can travel through to reach the final
    step — see this file's module docstring for why the naive
    "condition on both operands every step" design (tried first, kept
    only in hypotheses/H25.md's narrative, not here) doesn't test
    anything about the recurrence."""

    def __init__(self, head_size: int, n_steps: int, hidden: int = 64):
        super().__init__()
        self.head_size = head_size
        self.n_steps = n_steps
        self.step_embed = nn.Embedding(n_steps, 8)
        self.net = nn.Sequential(
            nn.Linear(1 + 8, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4 * head_size + head_size),  # r,k,v,a_gate_logit,w_raw
        )
        self.readout = nn.Linear(head_size, 1)

    def step_controls(self, x: torch.Tensor, step: int) -> dict:
        """x: [B] — the ONE scalar visible this step (0.0 on blank steps,
        regardless of the actual operand values for that example)."""
        B = x.shape[0]
        step_e = self.step_embed(torch.full((B,), step, dtype=torch.long, device=x.device))
        raw = self.net(torch.cat([x.unsqueeze(-1), step_e], dim=-1))
        r, k, v, a_logit, w_raw = raw.split(self.head_size, dim=-1)
        return {"r": r, "k": k, "v": v, "a_logit": a_logit, "w_raw": w_raw}

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                force_a_gate: float | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """force_a_gate: if set, overrides the LEARNED a_gate with this
        constant at every step — the counterfactual ablation from
        hypotheses/H25.md (does the delta-rule erase/rewrite mechanism
        specifically matter, independent of whatever a_gate the
        controller would otherwise have chosen?)."""
        B = a.shape[0]
        state = torch.zeros(B, self.head_size, self.head_size, device=a.device)
        state_trace = [state]
        out = None
        for t in range(self.n_steps):
            if t == 0:
                x = a
            elif t == self.n_steps - 1:
                x = b
            else:
                x = torch.zeros_like(a)  # blank — no operand info available here
            c = self.step_controls(x, t)
            a_gate = (torch.sigmoid(c["a_logit"]) if force_a_gate is None
                      else torch.full_like(c["a_logit"], force_a_gate))
            w = -F.softplus(-c["w_raw"]) - 0.5  # same soft-clamp shape as the real model
            out, state = micro_wkv_step(state, c["r"], c["k"], c["v"], w, a_gate)
            state_trace.append(state)
        y_hat = self.readout(out).squeeze(-1)
        return y_hat, state, state_trace


class FrozenFinalReadoutController(StagedController):
    """Isolates whether the delta-rule erase/rewrite mechanism specifically
    (not the readout's own bilinear structure, `r_t @ state`, between the
    current step's r/decay and the accumulated state) can carry a
    multiplicative interaction on its own. `r` and `w` at the FINAL step
    are learned but held CONSTANT (independent of `b`) — any surviving
    a·b computation must route through k/v/a_gate (the write and
    erase-rewrite terms) at that step instead, since r/decay can no
    longer vary with b to create the ordinary bilinear readout shortcut.

    Result (2026-08-23, hypotheses/H25.md): R²=0.9998 — succeeds cleanly
    even with this path closed off. Forcing a_gate=0 ON TOP of the frozen
    r/w collapses to R²=-0.08 (worse than predicting the mean) — the
    erase/rewrite mechanism is not just available here, it is NECESSARY.
    This is the clean, decisive version of the ablation — the un-frozen
    StagedController's milder degradation (R²=0.9996→0.8364) was
    confounded by the readout-bilinearity shortcut still being available."""

    def __init__(self, head_size: int, n_steps: int, hidden: int = 64):
        super().__init__(head_size, n_steps, hidden)
        self.final_r = nn.Parameter(torch.randn(head_size) * 0.1)
        self.final_w_raw = nn.Parameter(torch.randn(head_size) * 0.1)

    def step_controls(self, x: torch.Tensor, step: int) -> dict:
        c = super().step_controls(x, step)
        if step == self.n_steps - 1:
            B = x.shape[0]
            c["r"] = self.final_r.unsqueeze(0).expand(B, -1)
            c["w_raw"] = self.final_w_raw.unsqueeze(0).expand(B, -1)
        return c


class ChainedController(nn.Module):
    """The 'reproduce fleeb83, but exact operations' extension
    (2026-08-23): not just solving one fixed operation, but SELECTING
    which operation to apply from an input signal, and CHAINING several
    such steps so each round's result becomes the next round's operand
    — read entirely from state, no operand ever re-presented externally.
    This is the toy analog of fleeb83's own reported mechanism (frozen
    G1h 7.2B + trained state interface: operands/operation-selection
    held in state, one result written back and used to select/feed the
    next operation, 48/48 correct at chain depths 4/8/16/32) — except on
    continuous, exact arithmetic (add/multiply) rather than his
    Boolean/symbolic operations.

    Round 0 loads the initial operand `a` (op-code ignored that round).
    Rounds 1..n_rounds-1 each reveal one (operand, op_code) pair; the
    controller must combine whatever the CURRENT state holds (the
    running accumulator) with that operand according to op_code — it
    never sees the running total directly, only through state."""

    OPS = {0: lambda x, y: x + y, 1: lambda x, y: x * y}

    def __init__(self, head_size: int, n_rounds: int, hidden: int = 64):
        super().__init__()
        self.head_size = head_size
        self.n_rounds = n_rounds
        self.round_embed = nn.Embedding(n_rounds, 8)
        self.net = nn.Sequential(
            nn.Linear(2 + 8, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4 * head_size + head_size),
        )
        self.readout = nn.Linear(head_size, 1)

    def step_controls(self, operand: torch.Tensor, op_code: torch.Tensor, round_idx: int) -> dict:
        B = operand.shape[0]
        re = self.round_embed(torch.full((B,), round_idx, dtype=torch.long, device=operand.device))
        x = torch.stack([operand, op_code], dim=-1)
        raw = self.net(torch.cat([x, re], dim=-1))
        r, k, v, a_logit, w_raw = raw.split(self.head_size, dim=-1)
        a_gate = torch.sigmoid(a_logit)
        w = -F.softplus(-w_raw) - 0.5
        return {"r": r, "k": k, "v": v, "w": w, "a_gate": a_gate}

    def forward(self, operands: list[torch.Tensor], op_codes: list[torch.Tensor]
                ) -> torch.Tensor:
        """operands[0] is the initial value (op_codes[0] can be anything,
        e.g. zeros — ignored for round 0). operands[i]/op_codes[i] for
        i>=1 are the (value, op) pair revealed at round i."""
        B = operands[0].shape[0]
        state = torch.zeros(B, self.head_size, self.head_size, device=operands[0].device)
        out = None
        for r in range(self.n_rounds):
            c = self.step_controls(operands[r], op_codes[r], r)
            out, state = micro_wkv_step(state, c["r"], c["k"], c["v"], c["w"], c["a_gate"])
        return self.readout(out).squeeze(-1)


def _sample_chain(n: int, n_rounds: int, lo: float, hi: float, seed: int | None = None
                   ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    operands = [torch.empty(n).uniform_(lo, hi) for _ in range(n_rounds)]
    op_codes = [torch.zeros(n)] + [torch.randint(0, 2, (n,)).float() for _ in range(n_rounds - 1)]
    acc = operands[0].clone()
    for r in range(1, n_rounds):
        add_mask = op_codes[r] == 0
        acc = torch.where(add_mask, acc + operands[r], acc * operands[r])
    return operands, op_codes, acc


def train_chain(n_rounds: int = 3, head_size: int = 8, n_train_steps: int = 8000,
                 batch_size: int = 64, lr: float = 3e-3, seed: int = 0) -> dict:
    """Trains ChainedController on a chain of n_rounds-1 operations
    (add/multiply, selected per-round), reports in-distribution R² per
    op-combination (does it generalize across ALL combinatorial op
    sequences, not just a memorized subset?) and held-out(OOD)-range R²
    (2026-08-23 result: R²=0.998 in-distribution across all 4 two-op
    combinations at n_rounds=3, but only R²=0.60 held-out — a compounded
    2-step chain generalizes noticeably worse than the single-operation
    task, error likely compounding across rounds; honest, not hidden)."""
    torch.manual_seed(seed)
    model = ChainedController(head_size, n_rounds)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_train_steps):
        operands, op_codes, target = _sample_chain(batch_size, n_rounds, -2.5, 2.5)
        y_hat = model(operands, op_codes)
        loss = F.mse_loss(y_hat, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == n_train_steps - 1:
            print(f"  [chain] step {step}: train_mse={loss.item():.5f}")

    with torch.no_grad():
        operands, op_codes, target = _sample_chain(2000, n_rounds, -2.5, 2.5)
        y_hat = model(operands, op_codes)
        id_r2 = 1.0 - F.mse_loss(y_hat, target).item() / target.var().item()
        print(f"  [chain] all op-combinations, in-distribution: R²={id_r2:.4f}")

        per_combo = {}
        if n_rounds == 3:  # only 2 ops -> 4 combinations, breakdown is readable
            names = {0: "add", 1: "mul"}
            for o1 in (0, 1):
                for o2 in (0, 1):
                    mask = (op_codes[1] == o1) & (op_codes[2] == o2)
                    if mask.sum() > 10:
                        r2_c = 1.0 - F.mse_loss(y_hat[mask], target[mask]).item() / target[mask].var().item()
                        key = f"{names[o1]}_{names[o2]}"
                        per_combo[key] = r2_c
                        print(f"    op1={names[o1]:3s} op2={names[o2]:3s}: n={mask.sum().item():4d}  R²={r2_c:.4f}")

        sign = lambda t: (torch.randint(0, 2, t.shape) * 2 - 1).float()
        ood_operands = [operands[0]] + [None] * (n_rounds - 1)
        ood_operands[0] = sign(operands[0]) * torch.empty(2000).uniform_(2.5, 4.0)
        for r in range(1, n_rounds):
            ood_operands[r] = sign(operands[r]) * torch.empty(2000).uniform_(2.5, 4.0)
        ood_acc = ood_operands[0].clone()
        for r in range(1, n_rounds):
            add_mask = op_codes[r] == 0
            ood_acc = torch.where(add_mask, ood_acc + ood_operands[r], ood_acc * ood_operands[r])
        y_ood = model(ood_operands, op_codes)
        ood_r2 = 1.0 - F.mse_loss(y_ood, ood_acc).item() / ood_acc.var().item()
        print(f"  [chain] held-out(OOD) range: R²={ood_r2:.4f}")

    return {"n_rounds": n_rounds, "id_r2": id_r2, "ood_r2": ood_r2, "per_combo_r2": per_combo}


def train_task(task: str, n_steps: int, head_size: int, n_train_steps: int = 4000,
                batch_size: int = 64, lr: float = 3e-3, seed: int = 0,
                freeze_final_readout: bool = False) -> dict:
    """task in {'add', 'multiply'}. Train range [-3, 3]; held-out eval
    uses sign+magnitude both outside training range — a low eval loss
    means genuine computation, not memorised lookup. Runs the a_gate=0/1
    counterfactual ablation automatically.

    freeze_final_readout=False (default, StagedController): both the
    readout's own bilinearity (r_t @ state, between the final step's
    b-dependent r/decay and the a-carrying state) AND the erase/rewrite
    mechanism are available — 2026-08-23 result: a_gate=0 only drops
    R² 0.9996->0.8364, confounded, doesn't isolate either mechanism.

    freeze_final_readout=True (FrozenFinalReadoutController): closes the
    readout-bilinearity shortcut (r/w at the final step held constant,
    independent of b) — 2026-08-23 result: still solves multiplication
    (R²=0.9998) through k/v/a_gate alone, and a_gate=0 on top of this
    collapses to R²=-0.08 — decisive: the delta-rule erase/rewrite
    mechanism is NECESSARY here, not just available."""
    torch.manual_seed(seed)
    target_fn = {"add": lambda a, b: a + b, "multiply": lambda a, b: a * b}[task]

    cls = FrozenFinalReadoutController if freeze_final_readout else StagedController
    model = cls(head_size, n_steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def sample_batch(lo: float, hi: float, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(n).uniform_(lo, hi), torch.empty(n).uniform_(lo, hi)

    losses = []
    for step in range(n_train_steps):
        a, b = sample_batch(-3.0, 3.0, batch_size)
        y_hat, _, _ = model(a, b)
        loss = F.mse_loss(y_hat, target_fn(a, b))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == n_train_steps - 1:
            print(f"  [{task}] step {step}: train_mse={loss.item():.5f}")

    with torch.no_grad():
        a_id, b_id = sample_batch(-3.0, 3.0, 1000)
        y_id, _, _ = model(a_id, b_id)
        id_mse = F.mse_loss(y_id, target_fn(a_id, b_id)).item()
        id_r2 = 1.0 - id_mse / target_fn(a_id, b_id).var().item()

        sign_a = torch.randint(0, 2, (1000,)) * 2 - 1
        a_ood = sign_a.float() * torch.empty(1000).uniform_(3.0, 5.0)
        sign_b = torch.randint(0, 2, (1000,)) * 2 - 1
        b_ood = sign_b.float() * torch.empty(1000).uniform_(3.0, 5.0)
        y_ood, final_state, state_trace = model(a_ood, b_ood)
        ood_target = target_fn(a_ood, b_ood)
        ood_r2 = 1.0 - F.mse_loss(y_ood, ood_target).item() / ood_target.var().item()

        y_gate0, _, _ = model(a_id, b_id, force_a_gate=0.0)
        r2_gate0 = 1.0 - F.mse_loss(y_gate0, target_fn(a_id, b_id)).item() / target_fn(a_id, b_id).var().item()
        y_gate1, _, _ = model(a_id, b_id, force_a_gate=1.0)
        r2_gate1 = 1.0 - F.mse_loss(y_gate1, target_fn(a_id, b_id)).item() / target_fn(a_id, b_id).var().item()

        sr_by_step = []
        for s in state_trace:
            sr_batch = stable_rank([s[i] for i in range(min(32, s.shape[0]))])
            sr_flat = [x for layer in sr_batch for x in layer]
            sr_by_step.append(sum(sr_flat) / len(sr_flat))

    result = {
        "task": task, "n_steps": n_steps, "head_size": head_size,
        "id_r2": id_r2, "ood_r2": ood_r2,
        "ablation_a_gate_0_r2": r2_gate0, "ablation_a_gate_1_r2": r2_gate1,
        "stable_rank_by_step": sr_by_step,
    }
    print(f"  [{task}] in-distribution R²={id_r2:.4f}  held-out(OOD) R²={ood_r2:.4f}")
    print(f"  [{task}] counterfactual ablation — a_gate forced to 0: R²={r2_gate0:.4f}  "
          f"forced to 1: R²={r2_gate1:.4f}  (unablated={id_r2:.4f}; a large drop here "
          f"would implicate the delta-rule erase/rewrite mechanism specifically)")
    print(f"  [{task}] stable_rank by step: " + " -> ".join(f"{x:.2f}" for x in sr_by_step))
    return result


def _random_step_controls(head_size: int, g: torch.Generator) -> dict:
    return {
        "r": torch.randn(1, head_size, generator=g),
        "k": torch.randn(1, head_size, generator=g),
        "v": torch.randn(1, head_size, generator=g),
        "w": -0.5 - torch.rand(1, head_size, generator=g) * 3.0,
        "a_gate": torch.sigmoid(torch.randn(1, head_size, generator=g)),
    }


def jacobian_spectral_sampling(head_size: int = 8, n_samples: int = 2000,
                                seed: int = 7) -> dict:
    """hypotheses/H25.md item 1 ("local-Jacobian/curvature sampling").
    For FIXED controls (r,k,v,w,a_gate), micro_wkv_step's update is
    exactly affine in state (the same operator for every state value,
    not merely locally linear): `new_state = A@state + k⊗v` with
    `A = diag(decay) - outer(kk·a_gate, kk)` — derived directly from the
    code, not approximated numerically. Samples A's spectral radius
    (worst-case long-term contraction rate) across many random
    (k, w, a_gate) draws, plus a targeted a_gate=0..1 sweep at fixed
    (k, w) to isolate erase-rewrite's specific contribution.

    Real finding (2026-08-23): spectral radius is almost completely
    insensitive to a_gate across its ENTIRE range — erase-rewrite's
    rank-1 correction is confined to the unit-norm kk direction, and
    doesn't touch whichever channel holds the dominant decay value.
    Stability (spectral radius) and computation (erase-rewrite, proven
    NECESSARY by the frozen-readout ablation) are nearly decoupled:
    training can push a_gate anywhere the task needs without risking
    destabilizing the linear part through this specific channel."""
    g = torch.Generator().manual_seed(seed)
    spectral_radii = []
    decay_only_radii = []
    n_complex = 0
    for _ in range(n_samples):
        k = torch.randn(head_size, generator=g)
        w = -0.5 - torch.rand(head_size, generator=g) * 3.0
        a_gate = torch.sigmoid(torch.randn(head_size, generator=g))
        kk = F.normalize(k, p=2.0, dim=-1, eps=1e-8)
        decay = decay_from_logit(w)
        A = torch.diag(decay) - torch.outer(kk * a_gate, kk)
        eigvals = torch.linalg.eigvals(A)
        spectral_radii.append(eigvals.abs().max().item())
        decay_only_radii.append(decay.max().item())
        if eigvals.imag.abs().max().item() > 1e-4:
            n_complex += 1
    spectral_radii_t = torch.tensor(spectral_radii)
    decay_only_t = torch.tensor(decay_only_radii)
    n_above_1 = (spectral_radii_t > 1.0).float().mean().item()
    print(f"  {n_samples} random (k,w,a_gate) draws, head_size={head_size}:")
    print(f"  spectral radius of A: mean={spectral_radii_t.mean():.4f} "
          f"std={spectral_radii_t.std():.4f} range=[{spectral_radii_t.min():.4f}, "
          f"{spectral_radii_t.max():.4f}]")
    print(f"  complex-eigenvalue fraction (rotation, not pure decay): {n_complex / n_samples:.4f}")
    print(f"  fraction with radius > 1.0 (locally expanding): {n_above_1:.4f}")
    print(f"  pure-decay-only radius (a_gate=0 baseline): mean={decay_only_t.mean():.4f}")

    # Targeted sweep: fixed (k, w), vary a_gate uniformly 0..1 across all
    # channels, to isolate erase-rewrite's OWN effect on spectral radius
    # from the decay term's already-known variability.
    g2 = torch.Generator().manual_seed(seed + 100)
    k = torch.randn(head_size, generator=g2)
    w = -0.5 - torch.rand(head_size, generator=g2) * 3.0
    decay = decay_from_logit(w)
    kk = F.normalize(k, p=2.0, dim=-1, eps=1e-8)
    sweep = {}
    for a_scale in (0.0, 0.5, 0.9, 0.99, 1.0):
        a_gate = torch.full((head_size,), a_scale)
        A = torch.diag(decay) - torch.outer(kk * a_gate, kk)
        radius = torch.linalg.eigvals(A).abs().max().item()
        sweep[a_scale] = radius
    print(f"  fixed-(k,w) a_gate sweep (isolates erase-rewrite's own effect): "
          + ", ".join(f"a={a:.2f}->r={r:.4f}" for a, r in sweep.items()))

    return {"spectral_radius_mean": spectral_radii_t.mean().item(),
            "spectral_radius_std": spectral_radii_t.std().item(),
            "complex_fraction": n_complex / n_samples,
            "fraction_above_1": n_above_1,
            "a_gate_sweep": sweep}


def pseudo_inverse_ceiling(head_size: int = 8, n_drift_ticks: int = 4,
                            opt_steps: int = 1500, seed: int = 1) -> dict:
    """hypotheses/H25.md item 2 ("pseudo-inverse ceiling"): true time-
    reversal doesn't exist (decay is a genuine contraction, see verify()
    check #3) — but how close can ANY marker get, given the fixed
    algebra? Directly optimizes the physical controls (r,k,v,w,a_gate)
    themselves via gradient descent — NOT a trained embedding-generating
    network — to pull a drifted state S1 back toward its origin S0,
    repeated over T ticks (same repeated-constant-embedding mechanic as
    real phase/rewind markers). This is a CEILING: any real, network-
    generated Phase 2 rewind marker should do no better than this, since
    direct optimization of the raw controls is strictly more expressive
    than anything a marker embedding can be mapped to through the
    model's own projections.

    S0: a reference "origin" state (the state_after_prompt analog).
    S1: S0 after n_drift_ticks of RANDOM controls (simulating unknown
    prior exploration phases the rewind marker must pull back from).
    """
    g = torch.Generator().manual_seed(seed)
    S0 = torch.randn(1, head_size, head_size, generator=g)
    S0 = S0 / S0.norm() * 10.0  # arbitrary fixed reference scale

    S1 = S0.clone()
    for _ in range(n_drift_ticks):
        c = _random_step_controls(head_size, g)
        _, S1 = micro_wkv_step(S1, c["r"], c["k"], c["v"], c["w"], c["a_gate"])
    drift_rel = (S1 - S0).norm().item() / S0.norm().item()
    print(f"  after {n_drift_ticks} random explore ticks: relative drift "
          f"||S1-S0||/||S0||={drift_rel:.4f}")

    results = {}
    for T in (1, 4, 8):
        torch.manual_seed(42)
        r = torch.randn(1, head_size, requires_grad=True)
        k = torch.randn(1, head_size, requires_grad=True)
        v = torch.randn(1, head_size, requires_grad=True)
        w_raw = torch.randn(1, head_size, requires_grad=True)
        a_logit = torch.randn(1, head_size, requires_grad=True)
        opt = torch.optim.Adam([r, k, v, w_raw, a_logit], lr=0.05)

        state = S1
        for _ in range(opt_steps):
            opt.zero_grad()
            w = -F.softplus(-w_raw) - 0.5
            a_gate = torch.sigmoid(a_logit)
            state = S1.clone()
            for _ in range(T):
                _, state = micro_wkv_step(state, r, k, v, w, a_gate)
            loss = F.mse_loss(state, S0)
            loss.backward()
            opt.step()

        final_rel = (state.detach() - S0).norm().item() / S0.norm().item()
        results[T] = final_rel
        print(f"  T={T} ticks: best-achievable relative error "
              f"||rewind(S1)-S0||/||S0||={final_rel:.4f} (started at {drift_rel:.4f})")

    return {"drift_rel": drift_rel, "ceiling_by_T": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                     help="Run sanity checks against docs/rwkv7-mechanics.md's "
                          "derivation — no training, no task.")
    ap.add_argument("--train", choices=["add", "multiply"], default=None,
                     help="Train a StagedController (a at step 0, b at the final "
                          "step, blanks between) to solve a+b or a*b, then run the "
                          "a_gate counterfactual ablation. 'multiply' is the "
                          "informative test; 'add' is a trivial control.")
    ap.add_argument("--steps", type=int, default=4, help="Recurrence steps (T).")
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=4000, help="Optimizer steps.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-final-readout", action="store_true",
                     help="Use FrozenFinalReadoutController: r/w at the final step "
                          "are constant, independent of b, closing the readout's own "
                          "bilinear shortcut so any surviving computation must route "
                          "through k/v/a_gate (erase/rewrite) instead — the decisive "
                          "version of the ablation, see hypotheses/H25.md.")
    ap.add_argument("--chain", action="store_true",
                     help="Train ChainedController: select and chain n_rounds-1 "
                          "operations (add/multiply, per-round) from state, the "
                          "'reproduce fleeb83 but on exact operations' extension. "
                          "Uses --steps as n_rounds — pass --steps 3 to reproduce "
                          "the 2026-08-23 tested config (load + 2 ops); --steps's "
                          "own default (4) has not been separately verified.")
    ap.add_argument("--ceiling", action="store_true",
                     help="hypotheses/H25.md item 2: how close can ANY marker "
                          "(directly-optimized physical controls, not a trained "
                          "network) pull a drifted state back toward its origin, "
                          "repeated over T ticks — a ceiling for Phase 2's rewind "
                          "marker design, not a mechanism.")
    ap.add_argument("--jacobian", action="store_true",
                     help="hypotheses/H25.md item 1: exact closed-form spectral "
                          "radius sampling of the recurrence's per-step linear "
                          "operator across random controls, plus an a_gate sweep "
                          "isolating erase-rewrite's own effect on stability.")
    args = ap.parse_args()

    if args.verify:
        verify()
        return 0
    if args.ceiling:
        pseudo_inverse_ceiling(head_size=args.head_size, seed=args.seed)
        return 0
    if args.jacobian:
        jacobian_spectral_sampling(head_size=args.head_size, seed=args.seed)
        return 0
    if args.chain:
        train_chain(n_rounds=args.steps, head_size=args.head_size,
                     n_train_steps=args.train_steps, seed=args.seed)
        return 0
    if args.train:
        train_task(args.train, args.steps, args.head_size,
                    n_train_steps=args.train_steps, seed=args.seed,
                    freeze_final_readout=args.freeze_final_readout)
        return 0
    print("Nothing to do — pass --verify, --train {add,multiply} "
          "[--freeze-final-readout], --chain, or --ceiling. fleeb83's full "
          "counterfactual-altered-recurrence test on a real checkpoint "
          "(the toy version is tautological — see module docstring for why) "
          "and the Jacobian-sampling item (hypotheses/H25.md) are not "
          "implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
