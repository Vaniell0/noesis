# rwkv-peft-patches

`training/rwkv-peft/` is entirely `.gitignore`d (vendored, treated as a
pinned external tree per `training/lora_train.py`'s own docstring) — so
any local modification made directly inside it is invisible to git and
was at risk of being lost (only existed on this machine + whatever got
rsynced to a training VM). This directory is the durable, tracked copy of
every file actually changed inside `training/rwkv-peft/` during the
2026-09-10 Muon-integration work, so the work survives even if the
vendored tree is ever wiped/re-cloned.

**To apply**: copy each file here to the same relative path under
`training/rwkv-peft/rwkvt/...` (or `training/rwkv-peft/train.py`),
overwriting the vendored version.

| File here | Real location | What changed |
|---|---|---|
| `muon_opt.py` | `rwkvt/muon_opt.py` | **New file.** `MuonWithAuxAdam` — the class `light_rwkv.py`'s `configure_optimizers` already referenced (`args.optimizer=='muon'`) but that was never actually vendored anywhere; this is the real implementation, same Newton-Schulz math as `experiments/rl/loader.py::MuonHybrid`. |
| `light_rwkv.py` | `rwkvt/lightning_train/light_rwkv.py` | Added `from rwkvt.muon_opt import MuonWithAuxAdam` import; both `args.optimizer=='muon'` branches in `configure_optimizers` now call it with real params (`named_parameters()`, `muon_lr`/`lr_init`/`weight_decay`) instead of the undefined-name crash they had before. |
| `peft_loading.py` | `rwkvt/peft_loading.py` | Fixed a real bug: `load_peft_model`'s `--peft none` (plain full-FT) path never wrapped the bare `RWKVModel` in the Lightning `RWKV` class, so `trainer.fit()` crashed with `model must be a LightningModule, got RWKV7`. Added the missing `else: model = RWKV(args, model=model)` branch. |
| `train.py` | `train.py` | Added `--muon_lr` CLI arg (LR for Muon's hidden-matrix group; 0 falls back to `--lr_init`). |

Companion driver (already tracked, not vendored):
`training/scripts/run_state_reg_muon.py` — sets the env vars
`light_rwkv_state_reg_patch.apply()` needs *before* `train.py`'s own
argparse runs (model-class selection, WKV backend, deepspeed stub), then
execs `train.py`. This was the missing piece that made
`training/light_rwkv_state_reg_patch.py` (the real state_reg monkey-patch,
written earlier but never actually driven by anything) runnable for the
first time.

## Known issue in `muon_opt.py`, found 2026-09-14 — read before the next Muon LoRA run

`_is_muon_param` (`muon_opt.py:64`) selects any 2D `.weight` under
`.att.`/`.ffn.`, which under PEFT includes the injected `lora_A` and
`lora_B`. Each is then orthogonalized independently and rescaled by the
upstream aspect correction (`muon_opt.py:58`):

```python
update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
```

Measured on this project's real shapes (r=32, d=2560):

| tensor | shape | scale | update RMS |
|---|---|---|---|
| `lora_A` | 32×2560 | 1.000 | 0.0199 |
| `lora_B` | 2560×32 | **8.944** | **0.1776** |

One adapter, one shared `lr`, and one factor steps ~9× faster than the
other purely because it is tall rather than wide. The coefficient is not
wrong upstream — it normalises a standalone layer weight's per-element
RMS to Adam-like scale. It is blind here because the model never sees a
LoRA factor, only the product: `ΔW = s(B δA + δB A)` weights the two
factors by `‖B‖` and `‖A‖`, not by their shapes. PEFT initialises
`lora_B` to zeros and `lora_A` kaiming, so at step 0 the entire update
flows through `δB·A` — i.e. entirely through the factor that is also
being stepped ~9× too fast.

~~Second half of the same finding: **factor-wise Muon is itself a breadth
term.**~~ **RETRACTED 2026-09-16 — this paragraph was wrong twice and was
propagated into an external draft before it was caught.**

It read: "Newton-Schulz drives every singular value of each factor to 1 and the
product inherits it — measured at init, the induced `ΔW` came out rank 32 with
σ₃₂/σ₁ = 0.733 and entropy-rank 31.88 of 32, i.e. an almost perfectly flat
full-rank-r update injected every step by construction."

**(1) The measurement is a tautology at the point it was taken.** It was made at
initialisation on i.i.d. Gaussian gradients — orthogonalise Gaussian noise and the
spectrum is flat by construction. It says nothing about real gradients, where the
induced `ΔW` is a sum of two rank-r terms (`B·δA + δB·A`) and reaches rank up to
**2r**, not r.

**(2) Flatness is not what causes the damage, and this probe's own column says
so.** `muon_balanced` fixes retention while having an adapter spectrum
indistinguishable from the broken arm's — entropy-rank 4.667 vs 4.674, σ_r/σ_1
0.0555 vs 0.0538 — with retention differing by 0.255. Both facts were already
recorded in `hypotheses/H26.md`'s APPLICATION-vs-ALGORITHM entry; this file was
never updated to match.

The surviving statement is §2.2's: the damage is step SIZE. See
`docs/muon-rwkv7-finetuning.md` §2.3.

**PATCHED 2026-09-15.** The reason for holding off — that three candidate
fixes existed and choosing between them was an open question — closed when
the dose probe's `lr_mul` arm ran. Putting both factors on the large step,
symmetrically, is catastrophic (retain_r2 -747 at coefficient 4.0, -9892 at
5.66), far worse than the asymmetry it removes. So the damage is step SIZE,
the pairing asymmetry matters only because it carries `lora_B` past a
threshold near coefficient ~3, and any fix must equalise by LOWERING.

`MuonWithAuxAdam` now pairs `lora_A` with `lora_B` and steps them together,
scaling each factor's contribution to the update down to the smaller of the
two. Standalone weights are untouched and still take the aspect coefficient,
which is correct for them. `_balanced_pair_scale` is duplicated from
`experiments/rl/loader.py` rather than imported: this file is vendored into
`rwkvt/` on the training host, where `experiments/` does not exist. The two
copies must be changed together.

Both live Muon paths now carry the fix — this one, which `light_rwkv.py`
reaches via `args.optimizer=='muon'`, and `MuonHybrid`, which
`train_think_distill.py` uses. Until today only the second had it.

The three candidates are still implemented and compared at toy scale in
`experiments/A0_state_probe/lora_muon_probe.py` (invariants pinned in
`test_lora_muon.py`): equalise the two factors' contribution to `ΔW`;
orthogonalize the induced `ΔW` itself and push it back through the
factorisation; or drop the shape-derived scale for a measured
state-displacement one. Do not change `muon_opt.py` until that comparison
has a verdict — and when it does, change it here and in
`experiments/rl/loader.py::MuonHybrid`, which carries the same line.
