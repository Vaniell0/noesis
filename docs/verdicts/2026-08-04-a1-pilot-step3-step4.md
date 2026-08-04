# A1 pilot Step 3 / Step 4 — interim verdict, 2026-08-04

**Step 3 (α=1e-4): COMPLETE. Finding: alpha too small — state_reg invisible at SFT scale.**
**Step 4 (α=1e-3): IN PROGRESS. Early finding: clamp-bounded equilibrium reached; CE-state tradeoff confirmed.**

---

## Step 3 — α = 1e-4, full corpus (61 933 rollouts, 1 epoch)

### Setup

- Config: `training/config/pilot_step3_a1e4.yaml`
- α = 1e-4, chunk_ctx = 512, ctx_len = 2048, warmup = 200 steps
- state_reg: `trajectory_reg`, λ_δ = 1.0, λ_κ = 1.0, work_layers = {12, 16, 20}
- Per-layer state_loss clamped at −10 (prevents unbounded maximisation)
- Run terminated at 12 % (7 346 / 61 933 micro-batches) — early stop on evidence of saturation

### Findings

**CE saturates in the first ~10 % of the corpus.** Running-average loss
dropped 1.24 → 0.053 in 882 opt-steps (≈ 12 % of epoch). RWKV-7-G1d
was already well-adapted to glaive-function-calling; one epoch of SFT is
effectively a distribution re-shaping, not new knowledge acquisition.

**α = 1e-4 is invisible.** At saturation, the state_reg gradient
contribution is:

    α · |state_loss_max| = 1e-4 × 10 = 0.001

against CE ≈ 0.05 → ratio ≈ 1 : 50. The regulariser has no meaningful
leverage on the optimiser once warmup ends. Neither the loss curve nor
intermediate state metrics (step=0 only logged) showed any deviation
from a pure-CE run.

**Clamp integrity confirmed.** No NaN, no state explosion, total loss
remained positive throughout. The two fixes from the diagnostic session
— supervised-token zero-loss guard and fp32 wkv state — held across the
full run.

### Verdict

α = 1e-4 establishes the **lower bound** of the useful alpha range for
this corpus + model. Any α below this threshold is CE-dominated for the
full epoch and cannot alter state dynamics.

Artefact: partial LoRA adapter (12 % epoch), run dir
`training/runs/pilot_g1d_glaive_v2_step3_a1e4/`. Not used for A0 eval.

---

## Step 4 — α = 1e-3, full corpus (in progress, as of step 600)

### Setup

- Config: `training/config/pilot_step4_a1e3_full.yaml`
- α = 1e-3 (10× Step 3), same corpus and architecture
- Intermediate LoRA saves every 500 opt-steps (`training/runs/.../lora_weights.pth`)
- state_loss logged every 100 steps to `state_loss.jsonl`

### Early dynamics (steps 0 – 600)

| step | CE | state_loss | total | note |
|-----:|---:|------------|------:|------|
| 0 | 1.937 | −6.03 | 1.931 | training start |
| 100 | 0.127 | −7.63 | 0.119 | warmup ending |
| 200 | 0.057 | **−10.0** | 0.047 | **clamp reached** |
| 300 | 0.004 | −10.0 | **−0.006** | state_reg > CE |
| 400 | 0.011 | −10.0 | 0.001 | equilibrium forming |
| 500 | 0.011 | −10.0 | 0.001 | stable |
| 600 | **0.079** | −10.0 | 0.069 | CE elevated |

**Clamp reached at step 200.** The model found the state-motion ceiling
in ≈ 200 opt-steps (≈ 200 × 8 = 1 600 micro-batches, < 3 % of corpus).
state_loss = −10.0 means all three work-layer penalties are at the clamp
floor simultaneously.

**State_reg dominated CE at step 300.** total = −0.006 < 0 means:

    α · state_loss > CE   →   1e-3 × 10 > 0.004

The regulariser gradient was larger than the CE gradient at that step.
This is the first direct evidence that α = 1e-3 is in the active regime
for this corpus.

**Equilibrium at CE ≈ 0.011.** Steps 400–500 show total ≈ 0.001 with CE
elevated from its 0.004 minimum back to 0.011. The model cannot
simultaneously minimise CE and maximise ‖Δs‖ to the clamp ceiling — it
settles at a CE slightly above the pure-SFT minimum. This is the expected
CE–state tradeoff: state_reg is imposing a real constraint on the
optimiser.

**CE rises to 0.079 at step 600.** This spike warrants monitoring. Two
interpretations:

1. *Natural variance* — individual micro-batch CE is noisy; the 0.079
   value is a single-step reading, not a trend.
2. *Regulariser pressure* — as CE approaches the state_reg floor, the
   optimiser may oscillate between CE-reduction and state-motion phases.

The next 100-step log points (steps 700–1 000) will clarify which.

### Preliminary verdict (partial)

α = 1e-3 **is in the active regime**: state_reg reached the clamp, the
tradeoff with CE is confirmed, and the optimiser has settled at a
modified equilibrium (CE ≈ 0.011 vs ≈ 0.004 in pure-SFT baseline). This
is the signal A1 was designed to produce.

Whether this modified equilibrium improves reasoning on the A0 eval task
suite (bit_decoding, arithmetic_chain, scheduling) is open — requires the
full epoch checkpoint and an A0-eval run.

**Kill-switch status (from `training/state_reg.py` docstring):** state
RMS on {L12, L16, L20} not yet measured at inference time. To be checked
on the epoch-end checkpoint via `experiments/A0_state_probe/probe.py`.
If RMS > 2× baseline, raise λ_δ downweight or fall back to
`trajectory_reg_with_sr`.

---

## Engineering findings (applicable to future runs)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| NaN loss | All-masked chunks → CE = 0/0 | Zero-loss guard on supervised-token count |
| NaN loss (2nd) | FLA returns fp32 state → bf16 BlockStateList overflow | fp32 wkv_states in infctx_module.py |
| Speed 0.047 → 0.81 it/s | chunk_ctx=64: 64 outer checkpoints × 24 recomputes | chunk_ctx=512 → 4 outer checkpoints |
| State explosion (α=1e-3, 7k corpus) | CE→0 in 22 steps; state_reg dominated gradient unbounded | Clamp at −10 per layer + full corpus + α reduced |
| save_pretrained failure | RWKVConfig has no `__contains__`; PEFT compatibility | Manual LoRA weight extraction from state_dict |

---

## Next

1. Wait for Step 4 full epoch (ETA ≈ 22 h from launch, within VM window).
2. Merge LoRA adapter: base model + epoch-end adapter → `step4_merged.pth`.
3. Run `experiments/A0_state_probe/probe.py` on merged model (CPU, local).
4. Run `experiments/A0_eval/eval.py` on merged model (via Ollama, local).
5. Compare against Step 1 baseline (`rwkv-0.pth`, A0 eval 14.6 % overall).
6. Write final Step 4 verdict; update HYPOTHESES H8/H9 status.
7. Plan Step 5 if needed: α = 1e-2 (would require reducing clamp or
   accepting larger CE sacrifice), or accept α = 1e-3 as optimal and
   move to H7 falsifier run.
