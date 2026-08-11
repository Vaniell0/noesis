# A1 training pipeline

How a corpus becomes a checkpoint — and why each step exists.

---

## The loop

```
corpus selection
      ↓
normalization (raw data → rollouts JSONL)
      ↓
tokenization (.jsonl → .pt blob)
      ↓
training (LoRA + L_state)
      ↓
merge (LoRA adapter + base → full .pth)
      ↓
eval (48-task A0.2 rubric)
      ↓
verdict → next corpus decision
```

Each iteration is one "step" in the ledger. The corpus changes between steps;
the training code (LoRA rank, α, layer set) mostly carries over.

---

## Scripts by stage

### 1. Normalization — raw corpus → rollouts JSONL

All normalizers produce the same JSONL shape used by tokenizers:
`{"turns": [{"role": "user"|"assistant", "content": "..."}], "source": "..."}`.

| Script | Input | Purpose |
|--------|-------|---------|
| `scripts/normalize_xlam.py` | Salesforce/xlam-60k (HF) | Variant C primary — tool-use rollouts |
| `scripts/normalize_glaive.py` | glaive-function-calling-v2 | SUPERSEDED (steps 1–5, reactive corpus) |
| `normalize_traces.py` | Claude CLI `.jsonl` archives | Opus action-chain traces → rollouts |
| `normalize_toolbench.py` | ToolBench G123-DFS | ReAct `<think>` format rollouts |
| `scripts/normalize_hh_rlhf.py` | Anthropic hh-rlhf (HF) | Constitutional AI → plain-CoT |
| `scripts/normalize_react.py` | glaive + ToolBench | ReAct `<think>` JSONL |

### 2. Tokenization — JSONL → `.pt` blob

Blobs contain `ids` (token ids), `loss_mask` (1 = supervised position),
and optionally `state_mask` (1 = inside `<think>` span).

| Script | Format | State mask |
|--------|--------|------------|
| `tokenize_fixture.py` | Open fixture (34 turns) | No |
| `tokenize_rollouts.py` | Generic rollouts JSONL | No |
| `tokenize_dsl_rollouts.py` | DSL format (step 8) | No |
| `tokenize_plain_cot.py` | Plain-CoT `<think>…</think>` | **Yes** (ε-mask input) |

### 3. Corpus combining

When training on a mix of sources, combine pre-tokenized blobs:

| Script | Used in |
|--------|---------|
| `scripts/combine_step9_corpus.py` | Step 9 (rfc + selfcot + action) |
| `scripts/combine_step9b_corpus.py` | Step 9b (6 sources, weighted fractions) |

### 4. Training

Entry point: `train_pilot.py` (patches vendored `rwkv-peft/train.py`, then runs it).
Config: `training/config/pilot_step*.yaml`.

Key config fields:

| Field | Meaning |
|-------|---------|
| `ctx_len` | Sequence length. Must be ≥ 3 for L_state to fire (T<3 short-circuit). Use 2048+ for step 10. |
| `alpha` | L_state weight. 0 = CE only; 1e-3 = active regime (from step 3 calibration). |
| `work_layers` | Which WKV layers accumulate L_state gradient. [0,4,8,…,28] = all. |
| `outside_epsilon` | ε-mask: L_state weight outside `<think>` spans. 0.05 = 5% (step 9+). |
| `lora_rank` | LoRA rank. Step 9 doubled to 32 (from 16) — more capacity for RFC patterns. |
| `train_parts` | Which parameters train. `time+ln` = time-decay + LayerNorm (step 9b). |
| `h12b_i_enabled` | H12b.i rank entropy regularizer (penalizes LoRA rank collapse). |

### 5. Merge

LoRA adapter + base weights → single usable `.pth`:

```bash
python training/merge_lora.py \
    --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
    --lora training/runs/<run>/rwkv-0.pth \
    --out  training/runs/<run>/merged_e0.pth \
    --rank 32 --lora-alpha 64
```

### 6. Eval

```bash
RWKV_CUDA_ON=1 python experiments/A0_eval/eval.py \
    --backend rwkv \
    --model  training/runs/<run>/merged_e0.pth \
    --tasks  experiments/A0_eval/tasks.jsonl \
    --out    experiments/A0_eval/results/<step>_e0.json \
    --num-predict 512
```

`num_predict=512` is calibrated for the G1h 2.9B fine-tuned models.
Default (2048) over-generates and collapses scores (confirmed step 9).

---

## Corpus evolution — why each step happened

### Steps 1–5: glaive-v2 (FAILED)

glaive-function-calling-v2 is a tool-dispatch corpus: every rollout starts
with `assistant: tool_use`. The eval format is direct-answer (no tool
infrastructure). The model learned to dispatch correctly — it just couldn't
answer without a tool executor. This is not a training failure; it is a
corpus mismatch. **Root cause (H2):** reactive corpora cannot exercise WKV
state accumulation across multi-step reasoning.

### Step 6: action chains + ToolBench (partial epoch, no eval)

First honest reflexive corpus: Opus CLI session traces (10–70 tool_use per
session, multi-step) mixed with ToolBench G123-DFS (ReAct format). VM died
before eval. Mixing problem found in retrospect: ToolBench uses a different
format from Anthropic traces; 13% supervised ratio dilutes the signal.

### Step 7: Opus action traces only

Dropped ToolBench. Ran Opus traces alone. Eval: **8.3% (4/48)**. More
importantly: model spontaneously produced `<think>` tags without explicit
supervision. This confirmed that multi-step reflexive sequences train WKV
state in the right direction.

### Step 8: DSL format + L_state all-layers

Switched to DSL output format (noesis runtime's native schema). Applied
L_state to all 8 layers ([0,4,8,12,16,20,24,28]). H10 sweep on this
checkpoint revealed two failures: `state_readout` axis degenerate (it
equals `prompt_cot` at every cell); N=3 re-feed causes −27pp collapse.
Eval: **10.4% (5/48)**. DSL format also killed extraction: 100% DSL
corpus causes the model to generate DSL for plain-text tasks too.

### Step 9: RFC QA + ε-mask (BEST)

Key changes:
1. **Corpus:** 297 structured reasoning tasks from 20+ RFCs (SCTP, NTP,
   HTTP/2, OSPF, BGP). Binary protocol parsing = multi-step state-carry.
   Mixed format (plain text answers, not DSL).
2. **ε-mask:** L_state weight = α inside `<think>`, 0.05α outside. Forces
   the model to produce think tokens that update WKV usefully.
3. **LoRA rank 32** (doubled from 16) — needed capacity for RFC patterns.

Result: **43.75% (21/48)** — first checkpoint to beat all baselines.
`bit_decoding` still 0/16 (needs harder CRC/bitfield exercises).

### Step 9b: 6-source mix + ctx_len=512 (REGRESSION)

Added hh-rlhf, react, selfcot, base fraction to the mix. But `ctx_len=512`
triggered the T<3 short-circuit in `state_reg.py` — L_state fires only
when sequence length > 3 tokens, and most RFC QA chunks exceed 512 tokens,
so they get split into chunks where L_state = 0. Extraction collapsed from
62.5% to 12.5%. Overall: **39.6% (19/48)**.

**Fix for step 10:** `ctx_len=2048–4096`.

---

## Step 10 — what to change and why

| Change | Why |
|--------|-----|
| `ctx_len=2048` or `ctx_len=4096` | Restores L_state (T<3 short-circuit at 512) |
| Keep RFC fraction dominant (≥25%) | RFC = best signal for multi-step state-carry |
| Harder RFCs (CRC, bitfield exercises) | bit_decoding 0/16 is the remaining wall |
| Reduce hhrlhf or normalize format | Suspected noise source in step 9b mix |
| Wire H12b.i hookup in `light_rwkv.py` | `compute_h12bi_aux()` implemented but not connected |

Config base: `training/config/pilot_step9b.yaml` — change `ctx_len`, adjust fractions.
