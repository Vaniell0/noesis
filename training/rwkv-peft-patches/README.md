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
