# noesis / training

A1 fine-tune pipeline — **open tool-call + honesty API** target
(post-2026-07-22 pivot; see runtime-decisions snapshot §7 "Variant A
locked"). State-regularization hook wired in, disabled by default
(α=0). Full design: `plans/cosmic-purring-cocke.md`.

**Corpus policy (locked 2026-07-22):**

- **In weights (fine-tune signal):** open sources only. Anthropic
  public tool-use docs, open MCP schema examples, constitutional /
  honesty datasets, hand-crafted "I don't know → invoke tool"
  patterns. Fixture: `training/fixtures/tool_call_open.jsonl`.
- **NOT in weights (runtime-retrieval only):** the personal Claude
  CLI corpus under `training/corpus/` and `training/sanitised/`.
  Those directories carry `RECLASSIFIED.md` describing the pivot;
  they remain useful for **runtime retrieval prep**, not training.
- **Not supervision:** open reasoning traces (o1, DeepSeek-R1,
  Anthropic reasoning corpora) go to **eval sets**, not fine-tune —
  noesis has its own reasoning surface (state-based), so imitating
  another model's CoT tokens would be counterproductive.

## Status

Built in the 2026-07-21 parallel scaffolding session (brief:
`~/.claude/plans/parallel_a1_scaffolding.md`), pivoted 2026-07-22
per runtime-decisions §7:

- **Step 1 — env**: `.venv/` (system-site-packages), `peft 0.19.1`,
  `detect-secrets 1.5.0`, and `rwkv 0.8.32` installed from local
  wheels in `~/.libs/python/`. `bitsandbytes 0.49.2` wheel downloaded
  but not installed (QLoRA path, only needed at train time on GPU).
- **Steps 3-4 — personal-corpus prep (RECLASSIFIED to retrieval-only
  as of 2026-07-22)**: `extract_traces.py` and
  `sanitize.py` / `sanitize_patterns.py` still operate as documented,
  but their output under `training/corpus/` and `training/sanitised/`
  is now a **runtime-retrieval-corpus prep artifact**, not a
  fine-tune input. See `training/corpus/RECLASSIFIED.md` and
  `training/sanitised/RECLASSIFIED.md`. The `audit_sample.py` gate
  is still valid as a privacy gate for retrieval content, but is no
  longer a prerequisite for Step 5-6 (those now consume
  `training/fixtures/`).
- **Open fixture (new, 2026-07-22)** (`training/fixtures/tool_call_open.jsonl`):
  34 turns from Anthropic public tool-use docs + open MCP schema
  examples + hand-crafted "I don't know → invoke tool" patterns.
  Open-attributed per rollout `source` field. 2598 tokens, 30.4 %
  supervised (`<tool_use>` positions only).
- **Step 5 — fixture tokeniser** (`tokenize_fixture.py`):
  `rwkv_vocab_v20230424` → `.pt` with `ids` / `loss_mask` / `starts`.
  Mini-version; the fuller `tokenize_rollouts.py` for a bigger open
  corpus is future work.
- **Step 6 skeleton**: `JL-er/RWKV-PEFT` vendored into
  `training/rwkv-peft/` at commit
  `5704c39f8ab1d2ac63936ab392aadb6ba526e1a5` (`.git` removed).
  `state_reg.py` + `lora_train.py` (with `StateCapture` +
  `train_step`) + `config/pilot.yaml` live; the concrete patch on
  `light_rwkv.py::training_step` (infctx branch, chunk_size=1) is
  documented in `lora_train.py`'s module docstring but not applied
  — the vendored tree stays untouched, patched on top when the real
  train run happens on CUDA.

## Pending (main-session work)

- **Step 2 — A0.2 held-out eval** (upstream). Blocks any verdict on
  the pilot: without A0.2 numbers we cannot claim "beats pretrained by
  ≥ 5 pp". A0.2 is a fixed public task set — reasoning traces from
  o1/R1/Anthropic open corpora enter here as **task/probe material**,
  not fine-tune supervision.
- **Step 6 real train — hookup DONE, execution BLOCKED on compute.**
  As of 2026-07-22 the monkey-patch on `light_rwkv.RWKV.training_step`
  is implemented in `training/light_rwkv_state_reg_patch.py` and
  wired through `training/train_pilot.py` (which sets env, applies
  the patch, then `runpy`'s the vendored `train.py`). CPU-runnable
  smoke `tests/test_light_rwkv_patch.py` verifies the INACTIVE
  (mode=off) path without loading deepspeed/CUDA. Actual execution
  needs a GPU: BlinkDL `rwkv 0.8.32` is `torch.no_grad`-only (no
  autograd on CPU), and the RWKV-PEFT trainer requires deepspeed +
  CUDA kernels. **Compute path options**: (a) local GTX 1050 4GB
  via WSL2 using `training/bootstrap_pilot_gpu.sh`; (b) cloud burst
  (explicit decision per CLAUDE.md "cheap by construction").
- **Step 7 — pilot smoke run** on the open fixture, 0.4B, one α
  (α=0 baseline). Goal: verify convergence direction, not final
  metrics. Post-smoke eval on A0.2 subset (bit_book + arithmetic)
  to gauge whether tuned 0.4B closes any gap toward g1h (2.9B).
  Baseline substrate already measured: 0.4B pre-tune scores 14.6 %
  overall on A0.2 (see `experiments/A0_eval/results.md` §6).
- **Step 8 — A0.5 verdict integration** (already done). `state_reg.py`
  runs the `trajectory_reg` branch with A0.5-derived layer weights.
  Re-pilot with α > 0 and compare pending Step 7 baseline.

## Deferred (not-in-pilot)

- **RFC / technical corpus** (RFCs, CLI/tool docs, spec text). Planned
  for full A1, not the pilot smoke. Interim: runtime retrieval fills
  this from open sources.
- **State-transfer MVP** (guest Claude asks background noesis for
  context instead of grepping docs). Runtime-side feature — belongs to
  A0.6/A0.7 (intra-model swap / inter-checkpoint transfer) and to the
  Rust supervisor design, not to this training pipeline.
- **Variant B / C corpus** (sanitised pattern extraction from personal
  corpus / reopening the constraint). Off-table for Phase 1; may be
  reconsidered only if open-fixture A1 fails to teach the tool-call
  surface.

## State-reg hook — live (A0.5 PASS)

A0.5's three sub-tests all passed
(`experiments/A0_state_probe/results/a05_ext/verdict.md`) so the stub
has been replaced with the `trajectory_reg` branch:

- **`state_reg.py`** — `compute_state_reg` implements the delta + curvature
  penalty over A0.5-derived per-layer weights (`_A05_ZERO_LAYER_KL`).
  Returns 0 for T<3 (empty summation range), no exception.
- **`lora_train.py`** — `StateCapture` context manager + `train_step`
  primitive. Wires per-timestep WKV state into `compute_state_reg` and
  adds `cfg.alpha * L_state` to CE. When `alpha==0` or `mode=="off"`
  state capture is skipped entirely (zero hook overhead) and CE is
  returned untouched — same numerical curve as the pre-hookup trainer.
- **`config/pilot.yaml`** — `state_reg` block exposes `mode`, `alpha`,
  `lambda_delta`, `lambda_curvature`, `work_layers`. Default is
  `mode=off, alpha=0` (baseline sanity), sweep from there.

### Running the pilot (bring-up plan)

```bash
cd /home/vaniello/Desktop/projects/noesis

# 0. Smoke test — must pass before any real training.
./training/.venv/bin/python training/tests/test_state_reg_hookup.py

# 1. Baseline CE (mode=off, alpha=0.0). Edit pilot.yaml or override
#    at the CLI once the Lightning entry point supports it.
#    NOTE: the vendored trainer at training/rwkv-peft/train.py must
#    be patched per the docstring in training/lora_train.py — the
#    injection point is light_rwkv.py::training_step, infctx branch.
#    Until that patch lands, this driver exposes reusable primitives
#    (StateCapture, train_step) and the smoke test validates them
#    against a mock model without CUDA.

# 2. Sanity: mode=trajectory_reg, alpha=0.0. CE curve must equal step 1.
# 3. Sweep alpha in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}. Watch bit_book_05.
```

### What the smoke test verifies

`training/tests/test_state_reg_hookup.py` — three assertions, all must
pass before merging any state_reg change:

- (a) Two forward+backward steps on a mock micro-model produce finite
  gradients on every parameter.
- (b) `total_loss` at `alpha=0.1` differs from `total_loss` at
  `alpha=0.0` (with identical CE inputs) — proves state_reg is actually
  plumbed in and not silently zero'd.
- (c) `compute_state_reg` returns exactly `0.0` for sequence lengths
  `T ∈ {0, 1, 2}` — matches the "Zero for t < 2" docstring contract.

The mock model in the test (`MockTmix`) mimics the RWKV-7 attention
module contract: `layer_id` attribute + class-name containing `Tmix`
+ `_captured_wkv` set during forward. Real production integration
uses protocol B instead (`TimeMixState.wkv_state` in the module's
forward return tuple, `RWKV_TRAIN_TYPE=infctx` + `chunk_size=1`).

`VALID_MODES` in `state_reg.py` still rejects unknown modes at config
load so a typo cannot silently activate a non-existent loss.

## Directory layout

```
training/
  .venv/                       (gitignored) system-site-packages venv
  README.md                    this file
  extract_traces.py            retrieval-corpus prep (was Step 3)
  sanitize.py                  retrieval-corpus prep (was Step 4 driver)
  sanitize_patterns.py         retrieval-corpus prep (regex + class map)
  audit_sample.py              privacy gate (was Step 4 audit gate)
  state_reg.py                 Step 6 loss (A0.5 trajectory_reg branch)
  lora_train.py                Step 6 primitives (StateCapture + train_step)
  light_rwkv_state_reg_patch.py  Step 6 monkey-patch for vendored trainer
  train_pilot.py               Step 6 driver (patch + runpy vendored train.py)
  tokenize_fixture.py          Step 5 mini (open fixture only)
  tests/
    test_state_reg_hookup.py   Step 6 smoke test (mock model, 3 assertions)
    test_light_rwkv_patch.py   Step 6 patch smoke on CPU (4 assertions)
  fixtures/
    tool_call_open.jsonl       open sources — A1 fine-tune signal
    tool_call_open.pt          tokenised (Step 5 output)
  config/
    pilot.yaml                 pilot config (α=0 baseline)
  corpus/
    RECLASSIFIED.md            2026-07-22 pivot note (retrieval-only)
    raw/                       (gitignored) personal Claude CLI logs —
                               NOT training data anymore
  sanitised/
    RECLASSIFIED.md            2026-07-22 pivot note (retrieval-only)
    <rid>.jsonl                (gitignored) sanitised rollouts +
                               audit.jsonl + audit_decisions.jsonl
  tokenised/                   (gitignored) reserved for future
                               big-open-corpus tokeniser
  runs/                        (gitignored) reserved for Step 7
  rwkv-peft/                   (gitignored) vendored JL-er/RWKV-PEFT
                               commit 5704c39f
```

## Reproducing

### A1 pilot smoke (open fixture only)

```bash
cd /home/vaniello/Desktop/projects/noesis

# Build the tokenised fixture (fast, CPU).
training/.venv/bin/python training/tokenize_fixture.py

# Smoke-test the state_reg hookup on a mock model (no CUDA, no .pth).
training/.venv/bin/python training/tests/test_state_reg_hookup.py
# Expect: 3/3 tests PASSED.

# Real pilot smoke run: pending compute (see §Pending Step 6-7).
```

### Retrieval-corpus prep (personal Claude CLI logs)

Not training. Feeds runtime retrieval, subject to privacy gate.

```bash
cd /home/vaniello/Desktop/projects/noesis
source training/.venv/bin/activate

python training/extract_traces.py --dry-run --limit 10
python training/extract_traces.py

python training/sanitize.py

# Privacy gate — required for any content that will be surfaced back
# to a running model context via retrieval:
python training/audit_sample.py -n 50
```
