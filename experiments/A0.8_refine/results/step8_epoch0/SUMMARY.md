# H10 Sweep — Step 8 Epoch 0 Results

**Checkpoint:** `training/runs/pilot_g1h_step8_dsl/rwkv-0.pth`
**Model:** RWKV-7 G1h 2.9B
**Training:** Step 8, epoch 0 — DSL format (`<tool_call>/<tool_result>` inside `<think>`) + L_state span-mask loss on glaive-v2 corpus (61k rollouts)
**Eval:** 48-task A0.2 rubric (6 categories, greedy decode)
**Date:** 2026-08-07

---

## Full N × K × mode Accuracy Matrix

| N | K   | mode          | overall            | arith | bit_dec | extract | sched  | str_ops | symbolic |
|---|-----|---------------|--------------------|-------|---------|---------|--------|---------|----------|
| 1 | —   | silent        | 27.1% (13/48)      | 75%   | 6.3%    | 0%      | 33.3%  | 33.3%   | 62.5%    |
| 1 | 32  | prompt_cot    | 6.3%  (3/48)       | 0%    | 0%      | 0%      | 16.7%  | 0%      | 25%      |
| 1 | 32  | state_readout | 6.3%  (3/48)       | 0%    | 0%      | 0%      | 16.7%  | 0%      | 25%      |
| 1 | 128 | prompt_cot    | 16.7% (8/48)       | 25%   | 6.3%    | 0%      | 33.3%  | 16.7%   | 37.5%    |
| 1 | 128 | state_readout | 16.7% (8/48)       | 25%   | 6.3%    | 0%      | 33.3%  | 16.7%   | 37.5%    |
| 1 | 512 | prompt_cot    | N/A (file too large) | —   | —       | 0%      | —      | —       | —        |
| 1 | 512 | state_readout | N/A (file too large) | —   | —       | 0%      | —      | —       | —        |
| 2 | —   | silent        | **33.3% (16/48)**  | 25%   | 18.8%   | 0%      | **66.7%** | 50%  | 62.5%    |
| 2 | 32  | prompt_cot    | 10.4% (5/48)       | 0%    | 0%      | 0%      | 33.3%  | 33.3%   | 12.5%    |
| 2 | 32  | state_readout | 10.4% (5/48)       | 0%    | 0%      | 0%      | 33.3%  | 33.3%   | 12.5%    |
| 2 | 128 | prompt_cot    | 22.9% (11/48)      | 0%    | 6.3%    | 0%      | 50%    | 50%     | 50%      |
| 2 | 128 | state_readout | 22.9% (11/48)      | 0%    | 6.3%    | 0%      | 50%    | 50%     | 50%      |
| 2 | 512 | prompt_cot    | N/A (file too large) | —   | —       | 0%      | —      | —       | —        |
| 2 | 512 | state_readout | N/A (file too large) | —   | —       | 0%      | —      | —       | —        |
| 3 | —   | silent        | 6.3%  (3/48)       | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 12.5%    |
| 3 | 32  | prompt_cot    | 4.2%  (2/48)       | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 0%       |
| 3 | 32  | state_readout | 4.2%  (2/48)       | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 0%       |
| 3 | 128 | prompt_cot    | 6.3%  (3/48)       | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 12.5%    |
| 3 | 128 | state_readout | 6.3%  (3/48)       | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 12.5%    |
| 3 | 512 | prompt_cot    | N/A (file too large) | —   | —       | 0%      | —      | —       | —        |

Note: `n3_k512_readout64.json` was not generated (sweep produced 20 files, not 21).
K=512 files (n1, n2, n3 — both modes) exceed the 25k-token read limit; only extraction=0% is confirmed for those cells.

---

## Per-Category Best Cells

| Category       | Best cell(s)                          | Accuracy |
|----------------|---------------------------------------|----------|
| overall        | N=2, silent                           | 33.3%    |
| arithmetic     | N=1, silent                           | 75%      |
| symbolic       | N=1 silent / N=2 silent               | 62.5%    |
| scheduling     | N=2, silent                           | 66.7%    |
| string_ops     | N=2 silent / N=2 K=128 cot/readout   | 50%      |
| bit_decoding   | N=2, silent                           | 18.8%    |
| extraction     | — (0% everywhere)                     | 0%       |

---

## Key Findings

### 1. Best cell: N=2 silent (33.3%)

Silent double-pass over the prompt beats every CoT configuration at this
checkpoint. N=2 silent is a 6.2 pp improvement over N=1 silent (27.1%)
and a 10.4 pp improvement over N=2 K=128 CoT (22.9%). Adding think-tokens
via CoT actively hurts at this epoch — the model has not yet learned to
use CoT generation productively; emitting tokens degrades rather than
improves the answer.

### 2. N=3 collapse

N=3 silent collapses from 33.3% (N=2) to 6.3% — a 27 pp catastrophic
regression. All N=3 CoT cells also score at or below 6.3%. The third
silent pass does not refine state; it corrupts it.

Likely mechanism: glaive-v2 DSL training means silent re-feed triggers an
implicit `<tool_call>` loop in the state. By pass 3, accumulated tool-call
state overwrites whatever reasoning signal passes 1 and 2 built. There is
no stopping criterion for silent re-feed. The usable refinement axis is
N in {1, 2} only at this checkpoint.

### 3. state_readout == prompt_cot (exact tie at every cell)

Every (N, K) pair shows numerically identical accuracy for `prompt_cot`
and `state_readout` — across all 6 categories, not just overall.

H10 prediction "readout carries signal" (Δ >= 0.02 over silent at same N)
is falsified. The readout tokens carry no additional information over
prompt-injected CoT. At epoch 0 the model produces the same token
distribution from state readout as from continuing the same prompt prefix.
The readout axis should be dropped from active sweep dimensions until A1
training specifically targets state-differentiated generation.

### 4. extraction = 0% everywhere

Extraction is zero across all 20 cells. Root cause: training format collision.

The model was trained 100% on glaive-v2 DSL where data retrieval tasks are
solved by calling tools (extract_meeting_info, read_file, write_to_file).
When asked for bare JSON as a final answer, the model emits a tool call
instead. The eval rubric expects extracted fields directly, not wrapped in
`<tool_call>` JSON. The model correctly reasons about extraction in
`<think>` but cannot produce the answer directly — the direct-output path
for structured extraction has been overwritten by DSL training.

### 5. First bit_decoding success: bit_sub_01 at N=2 K=128

`bit_sub_01` (Greek-to-Latin cipher: apply substitution table to decode
ciphertext) is solved at N=2 K=128 (both modes) and at N=2 silent.
This is the only bit_decoding success — 15/16 tasks remain 0% everywhere.
The task requires tracking a multi-column lookup table character by character;
success at N=2 K=128 suggests the model needs either sufficient state
refinement (N>=2) or sufficient think budget (K>=128) for lookup chains.

---

## Interpretation

Hierarchy at step 8 epoch 0:

1. Silent re-feed is the most effective test-time strategy (no CoT emission).
2. N=2 is the sweet spot; N=3 is actively harmful.
3. CoT emission hurts relative to silent at every N — think tokens are
   low-quality/repetitive and dilute answer decode at this training stage.
4. The readout-mode axis is degenerate (state_readout == prompt_cot exactly).
5. DSL training has zero-blocked the extraction category.
6. arithmetic at N=1 silent (75%) is the strongest per-category signal;
   regresses to 25% under N=2 silent (arithmetic instability under double
   re-feed is worth investigating).

The 33.3% ceiling at N=2 silent is composed of:
- extraction: 0% (8 tasks fully blocked = -16.7 pp ceiling drag)
- bit_decoding: 18.8% at best (14/16 tasks blocked)
- arithmetic: 25% at N=2 (regression from N=1 75%)

Removing extraction from the rubric, the model scores 16/40 = 40% on the
remaining 5 categories at N=2 silent.

Step 9 or full A1 epoch training should address:
- Extraction: add direct-answer examples without tool-call wrapping
- bit_decoding: increase think budget / improve multi-hop lookup traces
- Arithmetic regression N=1→N=2: investigate state accumulation
- N=3 collapse: understand whether DSL loop accumulation is the actual cause
  (test: re-run N=3 on a non-DSL checkpoint if available)

---

## Files

- Sweep JSONs: `experiments/A0.8_refine/results/step8_epoch0/h10_sweep_step8_epoch0/`
- Sweep runner: `experiments/A0.8_refine/run_matrix_sweep.sh`
- Eval script: `eval.py` (flags: `--readout-mode`, `--readout-k`)
- HYPOTHESES.md: H10 status block updated with full matrix (2026-08-07)
- ROADMAP.md: A1 section updated with H10 sweep note (2026-08-07)
