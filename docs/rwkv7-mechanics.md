# RWKV-7 per-token mechanics — code-grounded reference

Written 2026-08-22. Companion to `docs/state-and-reasoning.md §1` (which
states the architecture at paper level, citing arXiv directly) — this
file is the CODE level: every quantity below is traced to an exact
file:line in this repo's vendored `training/rwkv-peft` and the
`rwkvfla` package it depends on, not reconstructed from memory of the
paper. Written after a live mistake this session (see §5) where
reasoning from the paper's abstract formula alone, without checking the
actual kernel source, produced a backwards conclusion about which
direction of decay the architecture actually constrains. That is the
reason this file exists: probe-tool trend-watching (`state_trajectory_probe.py`)
is only as trustworthy as the mechanics it's implicitly assuming, and
those need to be pinned down once, explicitly, rather than re-derived
under time pressure each time a number looks surprising.

**Verification status:** §1–§4 quote or directly compute from source
read this session (`rwkvt/rwkv7/att.py`, `rwkvt/operator/rwkvop.py`,
`rwkvfla/ops/rwkv7/recurrent_naive.py` on the training VM's `.venv`).
§5 numbers are from a real probe run on the base G1i checkpoint
(`experiments/rl/results/state_trajectory_base_1787392002.json`,
untrained ThinkChain markers). §6 is explicit inference, marked as such.

---

## 1. Symbol dictionary — code name ↔ paper symbol ↔ meaning

| Code (`att.py`) | Paper (`state-and-reasoning.md §1`) | Meaning |
|---|---|---|
| `r` (`self.receptance(xr)`) | — (query-like) | Reads the state at the answer step; not an input to the state *update* itself |
| `k` (after `self.key(xk)`, then modified — see §2) | `k_t` in `v_t^T k_t` write term | Key: which "row" of state gets written |
| `v` (`self.value(xv)`, + value-residual) | `v_t` | Value: what gets written along that row |
| `w` (raw, `self.w0 + tanh(xw@w1)@w2`, ±soft-clamp) | **NOT** the paper's `w_t` directly — see §3 | Pre-activation decay logit |
| `kk` (`k * self.k_k`, L2-normalized) | `κ̂_t` | Normalized key direction used for the erase/write pair |
| `a` (`sigmoid(...)`, called "in-context learning rate") | `a_t ∈ (0,1)^d` | How much of the erase-then-write happens this step |
| `g` (`sigmoid(...)`) | — (output gate) | Gates the final output, not part of the state update |
| kernel's positional `a` (`= -kk`) | `z_t = -κ̂_t` | First arg of the paper's rank-1 term `z_t^T b_t` |
| kernel's positional `b` (`= kk * a`) | `b_t = κ̂_t ⊙ a_t` | Second arg of the rank-1 term |

The paper's own formula (already in `state-and-reasoning.md §1`):

```
S_t = S_{t-1} · (diag(w_t) + z_t^T b_t) + v_t^T k_t
```

`w_t` in *that* equation is the paper's decay **multiplier** (already
in (0,1) space) — not the code's raw `w` logit. This is the exact
ambiguity that caused this session's error; §3 makes the code-to-paper
mapping for `w` explicit and unambiguous.

---

## 2. Per-token pipeline (`RWKV_Tmix_x070_infctx.forward`, `rwkvt/rwkv7/att.py:235-279`)

For one input token/embedding `x_t` (shape `[B,1,C]` when fed one at a
time, as every script in this repo does — `train_think_distill.py`,
`state_trajectory_probe.py`, `wkv_loop.py`):

1. **Time-shift delta**: `xx_t = x_{t-1} - x_t` (`x_{t-1}` = the
   previous call's last input, carried as `shift_state`; att.py:245).
   For a REPEATED constant input (e.g. `train_think_distill.py`'s
   ThinkChain phase-marker repeat, `experiments/rl/train_think_distill.py:376-385`),
   `xx_t = 0` exactly from the second repeat onward — no numerical
   approximation, `x_{t-1}` and `x_t` are the literal same tensor.
2. **Per-target mixing**: six learned per-channel coefficients
   (`self.x_r/x_w/x_k/x_v/x_a/x_g`, each `[1,1,C]`) blend `x_t` with
   `xx_t`: `x_r = x_t + xx_t · μ_r`, and identically for w/k/v/a/g
   (`self.addcmul_kernel` — assigned to `torch_addcmul` or
   `fused_addcmul` at att.py:44/46, called at att.py:248).
3. **Projections**:
   - `r = W_r · x_r`
   - `w_raw = self.w0 + tanh(x_w · W1) · W2` — see §3 for what happens next
   - `k = W_k · x_k`
   - `v = W_v · x_v`, then for `layer_id > 0`: `v = v + (v_first - v) · σ(v0 + x_v·V1·V2)`
     (value-residual: blends in the FIRST layer's `v`, gated)
   - `a = σ(a0 + x_a·A1·A2)` — "in-context learning rate"
   - `g = σ(x_g·G1)·G2` — output gate
4. **Key shaping**: `kk = normalize(k · κ, dim=head, p=2)` (κ =
   `self.k_k`), then `k = k · (1 + (a-1)·α)` (α = `self.k_a`) — the
   key actually used for the write/state-recurrence is this MODIFIED
   `k`, not the raw projection.
5. **State update** — see §3/§4.
6. **Output**: `x_out = LN(state_output) + Σ_head[(r·k·r_k) v]` (a
   learned per-head bonus term, `self.r_k`), then `x_out = W_out ·
   (x_out · g)`.

---

## 3. The state recurrence — ground truth from `rwkvfla/ops/rwkv7/recurrent_naive.py`

`RUN_RWKV7_INFCTX(r, k, v, w, a, b, s)` (`rwkvt/operator/rwkvop.py:32`,
active when `WKV=fla`, confirmed the live setting on the training VM —
`os.environ["WKV"] == "fla"` checked directly this session) calls
`chunk_rwkv7(...)`, whose reference (non-fused) semantics are
`naive_recurrent_rwkv7` in `rwkvfla/ops/rwkv7/recurrent_naive.py:10-77`.
Per head, per step (state `S ∈ R^{N×V}`):

```python
sab   = einsum('ik,k,j->ij', S, a_t, b_t)      # rank-1 erase/write correction
S     = S * exp(-exp(w_t)) + sab + outer(k_t, v_t)   # decay + correction + write
o_t   = einsum('j,ij->i', q_t, S)              # read (q_t = r_t here)
```

Two things this pins down exactly:

**(a) The actual decay multiplier is `exp(-exp(w_t))`, a DOUBLE
exponential of the code's raw `w`** — not `exp(w_t)`. This is the
paper's `w_t` from §1's table. `att.py`'s `w` (called `w_raw` in this
doc's §1 table) is what gets passed into the kernel; the kernel itself
applies the second exponential (`recurrent_naive.py:63`, comment: "when
use w=, -exp(w) is not needed... otherwise use log_w=-torch.exp(w)" —
i.e. THIS call path passes raw `w` straight through and the kernel
exponentiates it twice internally).

**(b) The `a`/`b` kernel arguments are `att.py`'s `(-kk, kk·a)`**,
matching §1's `z_t = -κ̂_t`, `b_t = κ̂_t⊙a_t`: `sab = S·(-κ̂_t) ⊗
(κ̂_t·a_t) = -a_t · (S·κ̂_t) ⊗ κ̂_t` — literally "read the state's
current content along the κ̂_t direction, then subtract a fraction
`a_t` of it back out along that same direction" (erase), immediately
followed by `+ v_t⊗k_t` (write). `a_t ∈ (0,1)` (sigmoid) controls how
much erasure happens before the new value is written — `a_t→0` means
"barely erase, just add on top" (values along that key direction
accumulate); `a_t→1` means "fully overwrite."

---

## 4. Decay bounds — what's architecturally constrained, and in which direction

`att.py:255-258` (the `os.environ["WKV"]` branch, confirmed `!= 'cuda'`
on this VM, so this is the LIVE formula):

```python
w_raw = -softplus(-(w0 + tanh(x_w·W1)·W2)) - 0.5
```

`softplus(x) ≥ 0` always, so `w_raw ≤ -0.5` — **upper-bounded**,
**unbounded below**. Composing with §3's `retention = exp(-exp(w_raw))`,
which is monotonically DECREASING in `w_raw`:

| `w_raw` | `retention = exp(-exp(w_raw))` |
|---|---|
| `-0.5` (the ceiling — closest to 0) | `0.5452` — **minimum possible retention** |
| `-2` | `0.8734` |
| `-5` | `0.9933` |
| `-10` | `0.999955` |
| `→ -∞` (unbounded) | `→ 1` — **no architectural ceiling** |

**The soft-clamp on `w_raw` is a FLOOR on retention (~0.545), not a
ceiling.** A channel can never forget faster than ~45.5%/step at the
theoretical extreme — but nothing in this formula prevents `w_raw`
from training toward large negative values, pushing retention
arbitrarily close to 1 (near-total memory, near-zero forgetting).

This is the opposite of what I told the user minutes before writing
this file (I had computed `exp(w_raw)` instead of `exp(-exp(w_raw))`,
and read the clamp direction backwards as a result). The corrected
conclusion **reopens**, rather than closes, `community-map.md`'s
question about whether training pushes decay toward the instability
boundary — this architecture does not prevent it on the `WKV=fla`
path, and on the `WKV=cuda` path `w_raw` isn't clamped at all (full
`(0,1)` retention range reachable in principle, per `att.py:255-256`'s
`if` branch, unverified against a `cuda`-mode run this session — the
live VM uses `fla`).

---

## 5. What the base model actually does (real probe run, not hypothetical)

`state_trajectory_probe.py` run against `models/rwkv7-g1i-2.9b-20260805-ctx16384.pth`
(base, untrained ThinkChain markers), 4 fixed prompts, all layers,
read/loop/chain branches (`experiments/rl/results/state_trajectory_base_1787392002.json`):

- `w_raw.max()` is **exactly `-0.5`, at every layer, every position,
  every branch** — some channel is always sitting at the clamp
  ceiling. `retention = 0.5452` there, universally.
- `w_raw.min()` **varies by layer** (roughly `-7.5` to `-15+` observed),
  giving `retention(w_raw.min())` between `0.9995` and `1.0` —
  **already, on the untrained base model, some channel in every layer
  is retaining essentially all of its state per step.**

This is not necessarily a defect. **H25's own stated mechanism**
("learned time-decay `w` attenuates earlier writes before the full
matrix is assembled," `hypotheses/H25.md`) *requires* some channels to
sit in this near-1-retention regime — that's how a rank-1 write made
many tokens ago could still be readable at the answer step at all. The
open, NOT YET ANSWERED question is whether this is a small number of
dedicated, task-appropriate slow channels (healthy — the H25
mechanism) or whether LoRA/full-FT training pushes an increasing
fraction of channels into this regime uncontrolled (the failure mode
`community-map.md` and the earlier-documented 3-4x state-norm growth
after training both point at). This file makes the question
answerable, not the answer itself — the next step is the SAME probe,
run against a trained checkpoint once one exists, comparing the
DISTRIBUTION of `w_raw` (not just min/max) between base and trained.

---

## 6. Open, explicitly-marked inference (not verified this session)

- Whether `WKV=cuda`'s unclamped `w_raw` is ever actually used anywhere
  in this project's real training runs, or whether `fla` is the only
  path exercised in practice — not checked.
- The exact numerical behavior of `chunk_rwkv7`'s CHUNKED (non-naive)
  kernel vs. the naive reference above — assumed equivalent (that's the
  whole point of a reference implementation existing), not independently
  re-derived from the chunked kernel's own source this session.
- Whether the near-1-retention channels found in §5 are the SAME
  channels across different prompts/layers (a stable, dedicated subset)
  or drift per-input — not measured; would need per-channel (not just
  min/max-over-channels) tracking, which `state_trajectory_probe.py`
  doesn't currently do.
