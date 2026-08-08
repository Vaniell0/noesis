# Training data shortlist (A1)

The noesis A1 corpus plan. Reconciled 2026-07-30 with
`docs/policies.md § A1 fine-tune corpus scope` (Variant C hybrid).
Both files are now the same story: this file is the operational
detail, `policies.md` is the constraint envelope.

## Framing decision (locked 2026-07-30, Variant C hybrid)

**Primary shape: action sequences. Secondary shape (limited):
adaptable open reasoning traces restructured into tool-shaped
steps. Personal data: excluded from weights entirely.**

- "Action sequence" = the objective, structured record of `tool_use →
  tool_result → tool_use`. Observable, reproducible, and cheap to
  verify (either you ran `git status` and got `X`, or you didn't).
- "Reasoning trace" = the model's internal thinking string.
  Subjective, stylistic, prone to character contamination
  ("as an AI assistant…", "let me think…"). Enters A1 **only** if
  transformed into linked step-and-tool structure; free-form
  thinking text does not.

Loss target: standard next-token loss on `tool_use` tokens only.
`tool_result` tokens are context (inputs); assistant thinking is
excluded from the loss mask. Behavior-cloning on *what to do
next*, not *how to sound while thinking*.

Rationale:

1. Actions are legally cleaner (they're structured JSON, not creative
   text; open agent corpora ship under Apache-2.0 / MIT with clear
   ToS on downstream training).
2. Character contamination avoided by construction — thinking
   tokens are never targets.
3. Verifiable at eval time: run the trained model on an agent
   benchmark and count how many tasks it completes end-to-end.

**Change from the earlier (superseded) framing.** The prior version
of this document (through 2026-07-29) named user's local Claude
Code traces as primary. That framing conflicted with
`policies.md § A1 fine-tune corpus scope` and with the CLAUDE.md
hard constraint "no personal corpus in weights". Reconciliation
2026-07-30 removes personal traces from A1 entirely and promotes
public agent corpora from supplementary to primary.

## Corpus (in priority order)

### 1. Public agent / function-calling corpora — **primary**

Open-licensed action-cloning material. Aggregate to ~30-50k
sanitised rollouts for the A1 micro-pilot; grow to ~150-200k for a
full-scale A1 campaign.

- **`Salesforce/xlam-function-calling-60k`** — Apache-2.0, 60k
  function-call chains, clean and balanced. Recommended anchor of
  the mix (highest quality/quantity signal-to-noise ratio).
- **`glaive-ai/glaive-function-calling-v2`** — Apache-2.0, ~113k
  entries, popular baseline. Filter for low-quality entries before
  inclusion (many single-turn low-effort rollouts).
- **`thunlp/ToolBench`** — MIT, 16k real APIs with long ReAct-style
  chains. Best for multi-step / error-recovery coverage.
- **`THUDM/AgentInstruct`** — Apache-2.0, 6-way agent tasks
  (Alfworld, WebShop, HotpotQA, KG, OS, DB). Good for cross-domain
  coverage. Some tasks are more instruction-following than
  tool-invocation — subset by tool-use density.

**Preprocessing pipeline (required before A1):**

1. **Licence check.** Verify each dataset's licence on the exact
   version we download (dataset cards get amended). Snapshot the
   SHA + card to `training/corpus/PROVENANCE.md`.
2. **Sanitisation.** Public corpora still contain example API keys,
   fake-but-realistic-looking secrets in demo prompts, and stray
   private paths. Regex filter (`detect-secrets` or equivalent) +
   sample-audit on ~200 random rollouts. This is a safety belt,
   not a substitute for source-selection.
3. **Filter for tool-use density.** Drop rollouts with fewer than
   N tool invocations or where thinking-token count dominates
   (character-contamination surface).
4. **Format normalisation.** All rollouts converted to the same
   `<user>...<tool_use>...<tool_result>...` shape, regardless of
   the source dataset's native format. Truncate `tool_result`
   bodies at N chars.
5. **Tokenise** with `rwkv_vocab_v20230424` (the vocab that the
   World and G1 checkpoints share).

### 2. Adaptable open reasoning traces — **secondary (limited)**

Enter A1 **only if restructured** into linked step-and-tool
sequences. Free-form thinking text does not.

- **DeepSeek-R1 distill subsets** (open-licensed variants only) —
  reasoning steps parsable into linked structure. Filter for
  presence of clear step delimiters (`Step 1:`, numbered lists,
  etc.); reject free-form paragraphs.
- **Competition-math CoT (open)** — restructure "compute X, then
  Y" into `<step>` blocks linkable to a `calculator` /
  `python_exec` tool schema.
- **Open code-reasoning traces** — link to `read_file` / `edit` /
  `run` tool schemas.

**Not primary:** these help only if a targeted conversion pipeline
lands cheaply. If restructuring cost exceeds the value they add
beyond §1, skip and stay pure action-cloning.

### 3. Explicitly rejected

- **`HelioAI/Fable-5-Distill-Reasoning-462x`** — `license: unknown`
  and filename literally says `Claude-Opus-4.7-4.8-DeepReason`.
  Anthropic-derived, no clean licence — reject outright.
- **`Crownelius/Complete-FABLE.5-traces-2M`** — MIT-license
  laundering of Anthropic outputs. Even if enforcement risk is
  practically low for a research-only project, the deeper issue is
  character-contamination: these traces are stylistically Claude,
  which would make the RWKV output "sound like a stuttering Claude"
  — the opposite of what noesis wants.
- **`Glint-Research/Fable-5-traces`** — AGPL-3.0 raw Claude Code
  session dumps. AGPL propagates to weights and is by itself a
  reject. Even at zero licence risk, Claude Code session dumps are
  character-contaminated Anthropic-style text.
- **`open-thoughts/OpenThoughts-114k`**, **`Bespoke-Stratos-17k`**,
  **`NuminaMath-CoT`** — reasoning corpora with structured
  step-by-step content. **Reclassified 2026-07-30:** moved from
  outright-reject to §2 secondary candidates (adaptable reasoning).
  Decision per-dataset at the corpus-prep stage: if step delimiters
  are clean and each can be linked to a tool schema (calculator,
  code-exec, retrieval), they enter A1 in restructured form. If
  the restructuring cost exceeds their marginal value beyond the
  public agent corpora in §1, they stay out. Check licence
  hygiene per-dataset — some contain distilled Anthropic outputs.

### 4. Evaluation only (not for weights)

- **`Anthropic/hh-rlhf`** — preference pairs for helpful/harmless
  behaviour. Use for eval sets, not for training the character.
  noesis character is defined by CLAUDE.md + mini-constitution, not
  by copying Anthropic's alignment target.
- **AgentBench** / **τ-bench** — end-task success rates on agent
  benchmarks. This is the honest downstream measurement of whether
  action-cloning actually works.

## Fine-tune plan (sketch)

- **Base:** RWKV-7 G1d 0.4B (reasoning-line — starts with better
  inductive bias for state utilisation, per H9). 2.9B upgrade
  deferred until GPU inventory reappears (see
  `project_noesis_hardware_2026_07_30`).
- **Method:** QLoRA. Rank 16-32 for the micro-pilot; target
  modules per the A0.5 verdict (layers that empirically carry
  state-work).
- **Curriculum:**
  1. Main tune on (1) — public agent / function-calling corpora —
     3-5 epochs (or 1-2 for the micro-pilot), LR 3e-5, loss
     masked to `tool_use` targets → learns the target agent-action
     distribution.
  2. Optional warm-up on (2) — adaptable reasoning restructured
     into tool-shaped steps — 1 epoch, LR 1e-4, only if the
     conversion pipeline landed cheaply per §2's "if
     restructuring cost exceeds their marginal value" test.
  3. Optional character-adapter on a tiny hand-written
     mini-constitution corpus (~100 examples), extremely low LR
     → shapes the assistant voice without contaminating action
     policy.
- **Eval:** τ-bench / AgentBench success rate on the tuned
  checkpoint versus the un-tuned G1d-0.4B baseline (already in
  `experiments/A0_eval/results/rwkv7_g1d_04b_np2048.json`,
  2026-07-22). Plus A0.5 probe re-run — does state utilisation
  shift after action-tuning? (follow-up test of H8 conditioned
  on training).

## Step 9 — Serious training plan (2026-08-08)

### Design principles

Step 8 result (epoch 0): silent 27.1%, N=2 silent 33.3%, CoT modes all worse than
silent — model fills K tokens with DSL tool_call syntax instead of WKV-optimal state
tokens. extraction = 0% (DSL format overfitting). bit_decoding = 6% (unchanged).

**Three objectives for step 9:**
1. Preserve G1h baseline (don't regress symbolic/scheduling/arithmetic)
2. Fix bit_decoding via RFC binary-protocol corpus
3. Teach model to generate WKV-optimal CoT (latent CoT principle)

### Step 9 corpus (three layers)

**Layer 1 — RFC binary-protocol tasks (new, primary signal for bit_decoding)**

Slice QA tasks from IETF RFCs that define binary wire formats, bit fields, flag tables:
- IP header (RFC 791): ToS, flags, fragmentation offset → bit manipulation tasks
- TCP header (RFC 793): flag byte, sequence arithmetic → multi-step reasoning
- DNS wire format (RFC 1035): label encoding, pointer compression → extraction tasks
- TLS record (RFC 5246/8446): content type, version bytes → binary reasoning

Format: `(RFC excerpt + scenario) → <think>apply rule</think> → plain-text answer`.
No DSL tool_call inside think — pure reasoning format.
Script: `training/scripts/restructure_rfc.py` (to be written).

**Layer 2 — Self-generated CoT corpus (new, primary signal for N/K/mode learning)**

1. Run step 8 checkpoint on diverse tasks (RFC QA + A0-like variants) with N=2, K=512
2. Filter: keep rollouts where final answer is correct
3. Relabel: correct `<think>` trace → training example with strong L_state inside think
4. This teaches the model to produce think-tokens that actually help (not DSL noise)

Script: `experiments/A0.8_refine/generate_cot_corpus.py` (to be written).

**Layer 3 — Existing action-chain rollouts (small fraction, prevent forgetting)**

~10% mix from step 7 action-chain corpus to prevent catastrophic forgetting of
tool-dispatch format. No new DSL saturation — already have enough.

### Step 9 loss changes

**ε-mask for L_state outside `<think>`:**
```python
# In training_step, after StateCapture:
effective_mask = span_mask * (1.0 - outside_epsilon) + outside_epsilon  # outside_epsilon = 0.05
state_loss = compute_state_reg(capture.per_layer(), cfg, weight_mask=effective_mask)
```
Prevents WKV from going completely passive outside think-spans. Fixes extraction collapse.

**All-layer L_state inside `<think>`:**
```yaml
# step9 config:
state_reg:
  work_layers: [0, 4, 8, 12, 16, 20, 24, 28]  # all 8 sampled layers inside think
  outside_epsilon: 0.05  # soft mask outside think
  inside_layers: "all"   # full-model L_state inside think spans
```

### What LoRA can and cannot do

- **Can learn**: format patterns, when/how to use `<think>`, N/K/mode selection cues
- **Cannot learn from scratch**: bit manipulation not in G1h base weights → RFC corpus
  activates latent knowledge, not creates it; if G1h doesn't know RFC bit formats,
  LoRA won't help. Validate: test G1h base on 5 RFC tasks before committing corpus.
- **Rank consideration**: rank 16 may be insufficient for multi-layer state shaping.
  Step 9 trial: rank 32, target_modules += output projection.

### N/K/mode curriculum (deferred to step 10)

After self-CoT corpus is built: generate training examples labeled by which (N,K,mode)
solved each task. Model learns to self-select compute budget. Eventually model emits
optimal K tokens autonomously (H16 gate) without explicit budget token.

## Open questions (not to resolve here)

- Do we need to preserve World3-base linguistic breadth by including
  a small percentage of general-language data in the mix? Or is
  reasoning + action enough for our runtime?
- Does character-adapter go last, or interleaved during main tune?
- What is the right per-tool loss weight? Uniformly, `tool_use`
  targets are shorter than `tool_result` context — but some tools
  (Write, Edit) matter more than others (Grep).
- Sanitisation false-positive rate — what's acceptable? Aggressive
  regex may drop 20% of the corpus; too lax leaves secrets in.
