# Training data shortlist (Phase 2)

Draft. This file is the noesis Phase-2 corpus plan. It is **not** a
locked decision — it is the current best answer to "what would we
fine-tune the RWKV-7 backbone on if we started tomorrow". Revisit
before any actual training run.

## Framing decision (locked)

**We fine-tune on action sequences, not on reasoning traces.**

- "Reasoning trace" = the model's internal thinking string, subjective,
  stylistic, prone to character contamination ("as an AI assistant…",
  "let me think about…").
- "Action sequence" = the objective, structured record of `tool_use →
  tool_result → tool_use`. Observable, reproducible, and cheap to
  verify (either you ran `git status` and got `X`, or you didn't).

Loss target: standard next-token loss on `tool_use` tokens only.
`tool_result` tokens are context (inputs), assistant thinking is
excluded from the loss mask. This is behavior-cloning on *what to do
next*, not *how to sound while thinking*.

Rationale:

1. Actions are legally cleaner (they're structured JSON, not creative
   text; Anthropic ToS on model outputs is specifically about
   generative content, not tool-invocation JSON).
2. Character contamination avoided by construction — thinking
   tokens are never targets.
3. Verifiable at eval time: run the trained model on an agent
   benchmark and count how many tasks it completes end-to-end.

## Corpus (in priority order)

### 1. Local Claude Code traces — **primary**

- **Size:** ~100M+ tokens (user's own history, growing).
- **License:** the user's own actions on his own machine. Legally the
  cleanest possible source.
- **Location:** user's local Claude Code history (path TBD; the user
  is packing into backups at the repo root).
- **Why primary:** matches the exact target domain (how the user
  personally uses an agent), not a generic assistant.

**Preprocessing pipeline (required before any tune):**

1. **Session split.** Split by `session_id` or by timestamp gap ≥ N
   minutes. Do not let the model see one session's context while
   trained on another — prevents context bleed.
2. **Extract action tokens.** From each session's `history.jsonl`:
   keep `user`, `tool_use`, `tool_result`, plus short assistant
   preambles that connect them. Drop long conversational assistant
   text (that is the character-contamination surface).
3. **Sanitise secrets — non-negotiable.** LLMs memorise literal
   strings that repeat ≥ 3-5 times. In agent traces expect:
   API keys (`sk-…`, `xoxb-…`, `ghp_…`), `.env` contents, SSH keys,
   private paths, hostnames, IPs, tokens in URLs. Regex filter
   (`detect-secrets` or equivalent) followed by manual
   sample-audit on ~200 random rollouts. **Even models that never
   leave the user's machine should not memorise secrets** — one
   accidental completion of `OPENAI_API_KEY=` in the terminal is
   enough to leak the key into logs or screenshots.
4. **Format.** Group as rollouts:
   ```
   <user>{prompt}</user>
   <tool_use>{json}</tool_use>
   <tool_result>{output_summary}</tool_result>
   <tool_use>{next_json}</tool_use>
   ...
   ```
   Truncate `tool_result` bodies at N chars — full file contents
   from a `Read` are useless targets and inflate the corpus 10x.
5. **Tokenise** with `rwkv_vocab_v20230424` (the vocab the World and
   G1 checkpoints share).

### 2. Public function-calling corpora — **supplementary**

Only if (1) turns out too narrow (single-user domain). Provides
breadth: different function-call styles, different APIs, different
error-recovery patterns.

- **`Salesforce/xlam-function-calling-60k`** — Apache-2.0, 60k
  function-call chains, clean and balanced.
- **`glaive-ai/glaive-function-calling-v2`** — Apache-2.0, ~113k
  entries, the popular baseline. Some low-quality entries; needs a
  filter pass.
- **`thunlp/ToolBench`** — MIT, 16k real APIs with long ReAct-style
  chains.
- **`THUDM/AgentInstruct`** — Apache-2.0, 6-way agent tasks
  (Alfworld, WebShop, HotpotQA, KG, OS, DB). Good for cross-domain
  coverage.

Rule: use only for pre-training / mid-training warm-up. Final
fine-tune must end on (1) so the model's final behavioural
distribution matches the target domain.

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
  session dumps. Only viable if strictly filtered to `tool_use` /
  `tool_result` (drop all thinking), and then it's redundant with
  (1) since the user already has his own traces at the same scale.
  Not worth the licence-viral risk (AGPL propagates to weights).
- **`open-thoughts/OpenThoughts-114k`**, **`Bespoke-Stratos-17k`**,
  **`NuminaMath-CoT`** — these are reasoning traces (thinking-CoT),
  not action sequences. Not what we want per the framing decision
  above. Kept as fallback if the action-cloning approach ever fails
  the phase-2 gate.

### 4. Evaluation only (not for weights)

- **`Anthropic/hh-rlhf`** — preference pairs for helpful/harmless
  behaviour. Use for eval sets, not for training the character.
  noesis character is defined by CLAUDE.md + mini-constitution, not
  by copying Anthropic's alignment target.
- **AgentBench** / **τ-bench** — end-task success rates on agent
  benchmarks. This is the honest downstream measurement of whether
  action-cloning actually works.

## Fine-tune plan (sketch)

- **Base:** RWKV-7 G1 (reasoning-line — starts with better inductive
  bias for state utilisation, per H9).
- **Method:** LoRA. Rank + target modules TBD after the A0.4 verdict
  narrows down which layers actually carry state-work.
- **Curriculum:**
  1. Warm-up on (2), 1 epoch, LR 1e-4 → adapts to function-call
     format.
  2. Main tune on (1), 3-5 epochs, LR 3e-5, loss masked to
     `tool_use` targets → learns *your* action distribution.
  3. Optional: character-adapter on a tiny hand-written
     mini-constitution corpus (~100 examples), extremely low LR →
     shapes the assistant voice without contaminating action
     policy.
- **Eval:** τ-bench / AgentBench success rate + A0.4 probe re-run
  (does state utilisation shift after action-tuning? — a follow-up
  test of H8 conditioned on training).

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
