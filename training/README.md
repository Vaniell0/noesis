# noesis / training

A1 is the fine-tune that turns a pretrained RWKV-7 checkpoint into a
coordinator capable of running noesis's cognitive runtime. The runtime
itself (collectors, store, composer, dispatcher) exists independently;
the model is what makes it coherent enough to use. A1 targets three
properties the pretrained checkpoint does not have out of the box:

- **State coherence under continuous input** — via the
  state-regularization hook (`state_reg.py`), the training-time arm of
  H4b (reasoning as state evolution). Wired in, disabled by default
  (α=0); sweep once the baseline holds.
- **Tool-call surface + honesty ("I don't know → invoke tool")** — via
  the open fixture (`fixtures/tool_call_open.jsonl`), so the model
  learns to reach for the runtime's tool dispatcher instead of
  hallucinating.
- **Zero personal-corpus contamination in weights** — CLAUDE.md hard
  constraint. Personal Claude CLI logs are runtime-retrieval material
  only; they never enter A1 supervision.

A1 is a training experiment, on the same axis as A0.* probes — not a
release checkpoint. Its success criterion is stated in `hypotheses/README.md`
(§H7 and adjacent) and its runbook lives in this file. Configs in
`training/config/`; verdicts in `docs/verdicts/`.

Post-2026-07-30 pivot (Variant C hybrid locked): action-cloning
primary + adaptable reasoning restructured secondary; personal
corpus stays reclassified to retrieval-only. See §Corpus policy
and `docs/policies.md § A1 fine-tune corpus scope` (both files
tell the same story after the 2026-07-30 reconciliation).

**Corpus policy (locked 2026-07-30, Variant C hybrid):**

- **Primary in weights (fine-tune signal, majority of the mix):**
  open action-cloning corpora — public agent / function-calling
  traces with clean licences (Salesforce/xlam-function-calling-60k,
  glaive-ai/glaive-function-calling-v2, thunlp/ToolBench,
  THUDM/AgentInstruct). Loss masked to `tool_use` tokens only.
  Aggregate target: ~30-50k rollouts for the micro-pilot,
  ~150-200k for full A1.
- **Secondary in weights (limited, only if restructured):**
  adaptable open reasoning traces (DeepSeek-R1 open distills,
  competition-math CoT, open code-reasoning) converted into
  linked step-and-tool structure. Free-form thinking text stays
  out.
- **Existing fixture** (`training/fixtures/tool_call_open.jsonl`,
  34 turns from Anthropic public tool-use docs + open MCP schema
  examples + hand-crafted "I don't know → invoke tool" patterns)
  remains as **smoke-test scale**. Full corpus adds on top of it.
- **NOT in weights (runtime-retrieval only):** the personal
  Claude CLI corpus under `training/corpus/raw/` and
  `training/sanitised/`. Those directories carry `RECLASSIFIED.md`;
  they remain useful for **runtime retrieval prep**, never A1
  supervision. This is a permanent boundary, not a Phase-1
  workaround.
- **NOT in weights (character-contamination or licence rot):**
  Anthropic-derived reasoning distills (Fable-5,
  Complete-FABLE.5-traces, other Claude-output-launderings).
  AGPL-viral Claude Code dumps (Glint-Research/Fable-5-traces).
  See `docs/training-data-shortlist.md § 3` for the full
  rejected list.

## Status

Scaffolded 2026-07-21, pivoted 2026-07-22 per runtime-decisions §7:

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
  Mini-version for the 34-turn smoke fixture.
- **Step 5 (full) — rollouts tokeniser** (`tokenize_rollouts.py`,
  new 2026-07-30): same rendering + loss-mask contract as
  `tokenize_fixture.py` (`_render_turns` imported directly, so the two
  paths cannot drift), plus multi-file glob, deterministic hash-based
  train/val split, and `.pt` output under `training/tokenised/`.
  End-to-end smoke on 5 rows: `normalize_xlam.py` → JSONL →
  `tokenize_rollouts.py --val-pct 20` → `xlam_mini_train.pt` /
  `xlam_mini_val.pt` (verified 2026-07-30).
- **Variant C primary normaliser** (`scripts/normalize_xlam.py`,
  new 2026-07-30): converts `Salesforce/xlam-function-calling-60k`
  rows (Apache-2.0) to the common rollouts JSONL shape (same as
  `training/fixtures/tool_call_open.jsonl`). Drops empty queries,
  zero-tool-call rows, rows matching a small secret-pattern regex
  belt; truncates over-long JSON argument fields. `--sample N`
  prints the first N accepted rollouts for inspection.
  HF dataset download is triggered lazily via `huggingface_hub`
  when `--input` is omitted and no local cache is present.
- **Step 6 skeleton**: `JL-er/RWKV-PEFT` vendored into
  `training/rwkv-peft/` at commit
  `5704c39f8ab1d2ac63936ab392aadb6ba526e1a5` (`.git` removed).
  `state_reg.py` + `lora_train.py` (with `StateCapture` +
  `train_step`) + `config/pilot.yaml` live; the concrete patch on
  `light_rwkv.py::training_step` (infctx branch, chunk_size=1) is
  documented in `lora_train.py`'s module docstring but not applied
  — the vendored tree stays untouched, patched on top when the real
  train run happens on CUDA.

## Pilot runs (2026-07-31 – 2026-08-08, Selectel RTX 4090, now deleted)

All steps complete. See `docs/verdicts/` for per-step writeups.

| Step | Corpus | α / notes | Eval (48-task) | Status |
|------|--------|-----------|----------------|--------|
| 1 | glaive-v2 | 1e-4 | — | DONE. CE 1.70→0.02 sanity. |
| 3 | glaive-v2 | 1e-4 | — | DONE. state_reg invisible (1:50 vs CE). |
| 4 | glaive-v2 | 1e-3 | — | DONE. Active regime, interrupted at step 3500. |
| 5 | glaive-v2 | 1e-3 | 6.2% (3/48) | DONE. **FAILED** — corpus mismatch: glaive-v2 is reactive (1-2 tool calls); RWKV needs reflexive multi-step. See `FAILED.md`. |
| 6 | action chains + ToolBench | 1e-3, trajectory_reg | no eval | DONE (partial epoch, VM died). No verdict recorded. |
| 7 | action chains (Opus traces) | 1e-3 | 8.3% (4/48) | DONE. Autodiscovered `<think>` format; first honest reflexive test. |
| 8 | DSL + L_state full span-mask | 1e-3, all-layers | 10.4% (5/48) | DONE. H10 sweep: state_readout degenerate; N=3 collapse −27pp. |
| **9** | RFC binary-protocol QA + ε-mask | 1e-3, rank=32 | **43.75% (21/48)** | **DONE. BEST.** First to beat all baselines. epoch 1 lost (disk full). |
| **9b** | 6-source corpus, ctx_len=512 | 1e-3 + H12b.i | 39.6% (19/48) | DONE. Regression: ctx_len=512 skips L_state (T<3). extraction 62.5%→12.5%. |

**Best checkpoint:** step9 epoch0. Eval: 43.75% vs baselines (Gemma3-4B 41.7%, G1h-base 39.6%, Qwen 37.5%).
Checkpoint on HuggingFace `Vaniello/noesis-rwkv7-g1h-2.9b`. Not present locally.

**Bugs fixed during runs (eval.py must use these):**
- `eval.py`: added `_strip_tool_use()` — step3500 eval was invalid without it (all answers wrapped in `<tool_use>` tags at 45% epoch, parser scored 0).
- `peft_loading.py`: added `NOESIS_RESUME_LORA` env var for LoRA-only checkpoint resume.
- FLA/Triton conflict on VM: replaced `fla/ops/__init__.py` with rwkv-only stub.
- deepspeed fused_adam: patched `light_rwkv.py` to use `torch.optim.AdamW`.
- `run_effort_sweep.sh`: fixed wrong JSON keys for H10 sweep.

## Step 10 — next training run

Goal: restore L_state effectiveness lost in step9b (ctx_len=512 bug).

- `ctx_len=2048–4096` (fixes T<3 short-circuit, restores ε-mask)
- Keep RFC fraction dominant (~25-30%)
- Add harder RFCs targeting `bit_decoding` gap (CRC, bitfield exercises)
- Reduce `hhrlhf` fraction or normalize format (suspected noise source)
- Config: `training/config/pilot_step9b.yaml` as template (update ctx_len)

To run eval on any merged checkpoint:
```bash
RWKV_CUDA_ON=1 python experiments/A0_eval/eval.py \
    --backend rwkv \
    --model <merged.pth> \
    --tasks experiments/A0_eval/tasks.jsonl \
    --out experiments/A0_eval/results/step10_e0.json \
    --num-predict 512
```

To merge LoRA adapter:
```bash
python training/merge_lora.py \
    --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
    --lora training/runs/<run_dir>/rwkv-0.pth \
    --out training/runs/<run_dir>/merged_e0.pth \
    --rank 32 --lora-alpha 64
```

## Deferred (not-in-pilot)

- **RFC / technical corpus** (RFCs, CLI/tool docs, spec text). Planned
  for full A1, not the pilot smoke. Interim: runtime retrieval fills
  this from open sources.
- **State-transfer MVP** (guest Claude asks background noesis for
  context instead of grepping docs). Runtime-side feature — belongs to
  A0.6/A0.7 (intra-model swap / inter-checkpoint transfer) and to the
  Rust supervisor design, not to this training pipeline.
- **Personal-corpus reopening (was "Variant B / C" in the 2026-07-22
  framing).** Superseded 2026-07-30 by Variant C hybrid (this file's
  §Corpus policy). The personal Claude CLI corpus stays out of
  weights permanently — not "off-table for Phase 1", but
  off-table for A1 by CLAUDE.md hard constraint. If open action
  corpora underperform, the answer is more open corpora
  (§Corpus policy §Secondary — adaptable open reasoning traces
  restructured into tool-shaped steps), not a reopening of the
  personal corpus. See `FAILED.md` 2026-07-30 "Variant A A1-corpus
  scope superseded" for the audit trail.

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
cd "$(git rev-parse --show-toplevel)"

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
  state_reg.py                 L_state loss (trajectory_reg + ε-mask)
  lora_train.py                LoRA primitives (StateCapture + train_step)
  light_rwkv_state_reg_patch.py  monkey-patch for vendored trainer
  noesis_dataset_patch.py      dataset loader with 4-tuple state_mask support
  train_pilot.py               driver (patch + runpy vendored train.py)
  merge_lora.py                merge LoRA adapter into base .pth
  tokenize_fixture.py          smoke-test tokenizer (34-turn open fixture)
  tokenize_rollouts.py         multi-file rollouts tokenizer (hash split)
  tokenize_dsl_rollouts.py     DSL-format tokenizer (step 8)
  tokenize_plain_cot.py        plain-CoT tokenizer + state_mask generator (step 9)
  extract_traces.py            retrieval-corpus prep (personal logs, NOT training)
  normalize_traces.py          Claude cache → rollouts JSONL
  normalize_toolbench.py       ToolBench G123-DFS → ReAct JSONL
  sanitize.py / sanitize_patterns.py  privacy gate for retrieval content
  audit_sample.py              privacy gate sampler
  dry_run_100.py               100-step smoke run
  scripts/
    normalize_xlam.py          xlam-60k → rollouts JSONL (Variant C primary)
    normalize_glaive.py        glaive-v2 → rollouts JSONL (SUPERSEDED, step5)
    normalize_hh_rlhf.py       hh-rlhf → plain-CoT JSONL
    normalize_react.py         glaive+toolbench → ReAct <think> JSONL
    restructure_rfc.py         RFC parser → QA tasks JSONL (step 9)
    combine_step9_corpus.py    combine rfc+selfcot+action .pt blobs
    combine_step9b_corpus.py   combine 6-source .pt blobs (step 9b)
  tests/
    test_state_reg_hookup.py   smoke test (mock model, 3 assertions)
    test_light_rwkv_patch.py   patch smoke on CPU (4 assertions)
  fixtures/
    tool_call_open.jsonl       open-source tool-use rollouts (smoke scale)
    tool_call_open.pt          tokenised fixture
  config/
    pilot.yaml                 baseline config (α=0)
    pilot_step5_from3500.yaml  step5 resume from step4 checkpoint
    pilot_step6_action_chains.yaml  step6: action chains + ToolBench
    pilot_step7_action_chains.yaml  step7: Opus action traces
    pilot_step8_dsl.yaml       step8: DSL format + L_state all-layers
    pilot_step9.yaml           step9: RFC + ε-mask, rank=32
    pilot_step9b.yaml          step9b: 6-source, H12b.i, train_parts=time+ln
  corpus/
    RECLASSIFIED.md            2026-07-22 note — retrieval-only, NOT training
    raw/                       (gitignored) personal Claude CLI logs
  sanitised/
    RECLASSIFIED.md            2026-07-22 note — retrieval-only
  corpus_open/                 (gitignored) open corpora (downloaded or generated)
  tokenised/                   (gitignored) .pt blobs (step9_rfc_train.pt etc.)
  runs/                        (gitignored) training checkpoints and logs
  rwkv-peft/                   (gitignored) vendored JL-er/RWKV-PEFT
```

## Reproducing

### A1 pilot smoke (open fixture only)

```bash
cd "$(git rev-parse --show-toplevel)"

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
cd "$(git rev-parse --show-toplevel)"
source training/.venv/bin/activate

python training/extract_traces.py --dry-run --limit 10
python training/extract_traces.py

python training/sanitize.py

# Privacy gate — required for any content that will be surfaced back
# to a running model context via retrieval:
python training/audit_sample.py -n 50
```
