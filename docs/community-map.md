# Community map — RWKV / SSM landscape vs noesis divergences

Written 2026-07-25. Last updated 2026-08-12. Consolidates prior-art
scattered across `docs/state-and-reasoning.md §2`, "Frontier adjacency"
boxes inside `HYPOTHESES.md` H8/H12, and per-topic snippets in
`docs/effort-frontier.md`. Purpose: a single flat map of what the
RWKV / state-space community *has*, what noesis is *actually adding*,
and what merits deeper external research.

Not a marketing comparison — a design document for future sessions
so we do not re-discover the same landscape every chat.

---

## 1. What the community actually ships

Verified 2026-07-25 by direct read of the referenced repos (BlinkDL/RWKV-LM
main branch, JL-er/RWKV-PEFT, RWKV/rwkv.cpp, josStorer/RWKV-Runner,
BlinkDL/ChatRWKV, Ai00 server) and arXiv 2503.14456v2.

### Architecture

- **RWKV-7 delta-rule state update** with vector-valued gating and
  input-dependent in-context learning rates (paper §3). This is an
  *architectural* feature of the forward pass — not a loss, not a
  UI knob. Note this because external write-ups routinely misattribute
  it as either a training objective ("Vector-Valued ICL Loss") or an
  inference-time slider ("3D-sliders for R/W/K/V"). Neither exists.
- **State layout:** `state[3·i+0]` = attn `x_prev`,
  `state[3·i+1]` = WKV `[n_head, head_size, head_size]`,
  `state[3·i+2]` = channel-mixing `x_prev`. Same across v5/v6/v7.
- **Formal expressivity:** TC⁰-surpassing (Theorem 2, App. D.1),
  recognises all regular languages at constant depth. Direct
  contrast to diagonal-transition RNNs and Transformers (limited to
  TC⁰). This is the architectural gap the delta-rule closes.

### RWKV-8 "Heron" + ROSA (experimental, verified 2026-07-30)

- **Codename:** RWKV-v8 = **"Heron" 🪶**. The rose emoji 🌹 that
  circulates in BlinkDL's posts is a visual marker for the ROSA
  mechanism, **not** an alternative codename. "Rose" is not
  used as a codename in the RWKV-LM repo or in BlinkDL's own posts.
- **ROSA** = *Rapid Online Suffix Automaton* (per DeepWiki-generated
  reference on `BlinkDL/RWKV-LM/3.2-rwkv-v8-with-rosa`; BlinkDL's
  own twitter thread confirms "SA ≠ Self-Attention"). Positioned
  as a neurosymbolic infinite-range exact-match propagator: excels
  at digit reversal, multi-digit arithmetic, copy / count tasks
  that pure attention or pure state-recurrence handle poorly at
  long spans.
- **Not a substrate swap.** RWKV-v8 uses the RWKV-v7 "Goose"
  time-mixing block (`RWKV_Tmix_x070`) as its foundation; ROSA
  is added as a separate block (`ROSA_QKV_B_1bit`) combining
  time-shifting with 1-bit linear projections. Concretely: an
  existing RWKV-7 (G1 etc.) checkpoint **cannot** be upgraded to
  RWKV-8 by dropping in ROSA — the ROSA block is part of the
  network graph and must be trained with the rest.
- **Public checkpoints are toy-scale (2026-07-30).**
  - `260123_reverse_L2.pth` — 39.6K parameters, reverse-sequence
    task, 99.6% accuracy.
  - `251024_rosaQKV_L4_digit40.pth` — ~1M parameters, 40-digit
    arithmetic, 99% per-digit accuracy.
  - `BlinkDL/rwkv-8-pile` — early-testing Pile run
    (332B tokens; ~135 downloads/month; author states "early
    testing versions"). No 0.4B+ production checkpoint.
- **What ROSA does NOT do (per the same source).** No claim of
  verbatim recall of a training corpus in weights, no claim of
  drop-in replacement for a trained RWKV-7 model, no claim of
  solving word-problem / semantic math. Advertised strengths are
  algorithmic-precision tasks (exact-copy, exact-arithmetic,
  long-range exact matches), which is orthogonal to noesis's
  H7 "knowledge in context" wager and to H21/H22 truth-system.
- **Implication for noesis.** If RWKV-8 matures at 0.4B+ scale it
  is a *layer addition* over the current substrate, not a
  substrate change. The `noesis-rwkv-sys` state-API contract
  (WKV get/eval/clone) still applies to the Goose portion; ROSA
  adds its own state that would need its own persistence surface
  (unscoped, deferred until a runnable checkpoint at reasoning
  scale exists). Do not plan A1 / A2 around ROSA.

### so(3)/SO(3)/Cayley state PoC (SimpleRose, 2026-08-02)

Proposed alternative recurrence replacing RWKV compressed state with a
constant-size state based on Lie algebra so(3) and Cayley composition:

- State = 3-vector → antisymmetric matrix in so(3) (3 DOF, 3×3)
- Token update → second so(3) element from `tanh(W·x_t)`
- Composition via Cayley map: `C(X) = (I-X)(I+X)^{-1}` → both terms
  are rotations → product `R_prev · R_token ∈ SO(3)`, non-commutative
  (order-sensitive without KV cache)
- Gram correction at temporal boundaries (PolARM): removes axis
  anisotropy from mixed-precision accumulation
- Decay: `λ(t) = sigmoid(W_λ · x_t + b_λ)`, RWKV-like forgetting
- Compress back via inverse Cayley (exact for SO(3), singularity at 180°)

Result: O(L) compute, constant 3-param state per head, orthogonality
preserved by construction.

**Significance for noesis.** Cayley composition is an architectural solution
to WKV state matrix degeneracy — the same problem H12b.i (rank entropy
regularizer) addresses via training-time penalty. so(3) approach: 3 DOF
per head (very low capacity, architecturally bounded). WKV state:
`head_size × head_size` = 4096 parameters per head (high capacity,
degeneracy not architecturally prevented). These are orthogonal solutions
to the same problem at different capacity/constraint tradeoffs. Not
applicable to G1h without full retraining.

### Community ROSA training stack (reported 2026-08-04)

Third-party toolchain that has grown around ROSA between the toy
checkpoints and any production-scale run. Not directly re-verified
against source; treat as pointers, not endorsement.

- **`johanwind/wind_rosa`** — CUDA kernels for ROSA with **exact
  gradients via finite differences**. Solves the "discrete
  operations are hard to differentiate" problem head-on. Reference
  point for anyone writing their own kernel.
- **`wjie98/rosa_soft`** — production-shape package (pip-installable)
  with **soft-surrogate gradient**, dropout, `mismatch_scale`,
  variable-length sequences. If we ever wire ROSA into a noesis
  training run we do not need to write kernels from scratch — this
  is the drop-in building block.
- **`zyaaa-ux/ROSA-Tuning`** — third training-track project;
  positioning vs the two above is unclear without a direct read.
- **`bcml-ai/rosa-plus`** — pure-statistics ROSA (no NN backbone).
  Ships coherent "Shakespeare-like" text at the surface level but
  characters drift and rhythm breaks. **Negative-result reference:**
  ROSA-only is a surface-pattern engine, not an understanding
  engine.
- **`x-0D/RASP`** — adds syntactic structure on top of ROSA; still
  bottlenecks on lack of semantic depth. Complements the ROSA+
  negative result.

**Framing for noesis Variant C (hybrid).** The rosa-plus and RASP
projects are the *positive* evidence for the Variant C wager: pure
suffix-automaton statistics is provably strong on surface patterns
and provably weak on understanding. We are not betting on ROSA as
the reasoning engine — we bet on **WKV for understanding + ROSA
(if/when it matures at scale) for exact-precision operations**
(arithmetic, verbatim copy, factual lookup). This mirrors the
RWKV-8 "Heron" architectural split, not competes with it.

**A1 / A2 impact: none.** Same as the parent section — do not plan
A1 or A2 around ROSA. These are runway pointers for the A2 /
H12b / eventual reasoning-scale RWKV-8 window, so we do not
re-discover the training stack from zero when the moment comes.

### Training

- **Objective:** next-token cross-entropy + `L2Wrap` (~10⁻⁴ scale on
  max logit, spike-suppression). That is *all* — no auxiliary state
  losses, no contrastive terms, no DPO/RLHF baked into pretrain.
  Verified in `RWKV-v5/src/model.py`; RWKV-v7 uses the same shape.
- **G1/G0 curriculum:** data-side only. G0x = <1 epoch, G1x = >1
  epoch; letter suffix (a…h…) increments the dataset. **`h` is a
  data-version, not an architectural variant.** No custom loss, no
  DPO. World3.5, 5.16T tokens, multilingual, open.
- **Precision:** bf16 activations + fp32 kernel for WKV
  (paper §8 p. 15). Anything below fp32 in the kernel measurably
  distorts state trajectories — relevant for probe design.

### Fine-tuning (community)

- **JL-er/RWKV-PEFT** ships LoRA, PISSA, Bone/MiSS/DiSHA, plus
  **State Tuning**. State Tuning trains the **initial state vector**
  (per-layer, fixed-length, effectively prompt-tuning with a
  numerical prefix), *not* a running-WKV objective. Origin:
  Jellyfish042/RWKV-StateTuning. External blogs that describe state
  tuning as "training against state trajectories" or "modifying the
  running WKV memory" overshoot the actual mechanism.
- **Blealtan/RWKV-LM-LoRA** — infinite-context training branch.

### State-tuning production pipeline (Scarletwolf champion recipe, 2026-08-12)

Verified production pipeline for RWKV state tuning (not LoRA). Output is
**the initial hidden state only** (11 MB at 2.9B) — not a weight delta,
no merge step needed.

**Chain:**
```
jsonl {"text": ...}
  → json2binidx (RWKVTokenizer, --append-eod)
  → RWKV-PEFT --peft state --op fla --my_testing x070
     lr 1e-2→1e-3 cos, warmup 50, one epoch, bf16, grad_cp 1
  → rwkv-0.pth  ← this IS the state (overwrite st[3i+1] at inference)
```

BlinkDL note (2026-08-12): 1e-2→1e-3 LR "too high" — try 1e-3→1e-5 or
1e-4→1e-6; warmup 10 sufficient; larger models need smaller LR.

**Silent failure modes (in order hit in production):**
1. `ctx_len` shorter than longest formatted example → binidx truncates without warning
   (20-tool JSON schemas ~6k tok/example; went 4096→6144)
2. Eval priming must match training byte-for-byte — same state behind different
   priming dropped 71→58
3. `time_state` injection orientation: overwrite `st[3i+1]` directly; transposed
   loads and generates garbage; probe one known query before burning an eval
4. Token id 0 = end of document; `clean_txt` (collapse repeated newlines) on
   system/user — both from official G1x templates file
5. Tiny corpus: repeat with seeded passes to reach recipe's step count, else
   cosine schedule never leaves warmup and state-tuning appears to not work

**Inference serving:**
- State = initial hidden state; overwrite `st[3i+1]` before prefill; nothing
  to merge. Cache state after fixed prefix, reuse across requests → prefill
  cost collapses.
- **GGUF/llama.cpp cannot carry state** — their CPU production cannot use tuned
  states. Ollama GGUF is inference-only; state injection requires `.pth` +
  rwkv-cpp.

**Hardware / cost:** single 3090 or A40 comfortable at 2.9B (~5.5 GB bf16).
17.7M tokens, 2885 steps, ~110 min on A40 at $0.44/h. First complete run
under $2.

**Corpus lessons:**
- A few hundred examples installs fence format; two things matter more than
  exact mix: (a) train inside the scaffold you'll serve, (b) include nulls
  (~15% of mix). Pair negatives with positives from same domain — unpaired
  negatives install a silence reflex, not a boundary.
- Name confusion: 135 targeted examples → 0 improvement. Capacity question,
  not data question.
- Abstention installs inside its curriculum then drops by half on fresh cases
  in a different style. Test out-of-curriculum before shipping.

**Multi-turn empirical law (production, 2026-08-12):** 0/2 for every model
measured (tuned or not, RWKV or Transformer). "States install dispositions,
not procedures." Multi-step procedures must live in the agent harness.

**Checkpoint hygiene:** g1g and g1h bases deleted from BlinkDL/RWKV-LM main
on 2026-08-05/08-08. Resolve snapshot `6d5762253b34` preserves them. SHA256
everything before relying on a checkpoint.

### State-tuning empirical results (Scarletwolf, 2026-08)

Independent benchmark: `scarletwolf_ai/rwkv-toolcaller-bench` (Codeberg),
82-case frozen judge, greedy decode, G1x format, tool selection + abstention.

**G1i raw (zero tuning):**

| Size  | raw/82 |
|-------|--------|
| 1.5B  | 29     |
| 2.9B  | 36     |
| 7.2B  | 43     |
| 13.3B | 52     |

G1i raw *scales with size* — G1g raw was flat 28–29 across all sizes.
Attributed to value residual in G1i (BlinkDL: "rwkv7 improved value
residual"; prior art ResiDual, arXiv 2304.14802).

**G1i state-tuned (same recipe and data, 1 epoch):**

| Base  | state-tuned/82 | Δ abstention | notes |
|-------|---------------|--------------|-------|
| G1i 1.5B | **55/82** | 0/17 → 16/17 | 4 unparsable outputs; selection 9/18, args 15/22 |
| G1i 2.9B | **63/82** (bit-replicated) | 0/17 → 16/17 | |
| G1g 7.2B | **69/82** | — | |
| G1i 13.3B | **70/82** | 0/17 → 17/17 | |

McNemar G1i 13.3B vs 2.9B: p = 0.119 (not significant). Bench saturates
at ~69–70. 17 of 18 gained points from abstention alone. **Disposition
installs at 1.5B** — what degrades going down is output format and argument
mapping, not the core decision.

**Readout corrector (exploratory, 2026-08-12):** post-hoc reranker on WKV
state readout, 2.9B policy, no retraining. Free 57 → corrected 64, peaks 65.
Gain by size: **+7 (1.5B), +6 (2.9B), +2 (7.2B)** — smaller models benefit
most. 1.5B tuned + corrector ≈ bare 2.9B tuned (63). CPU-deployable at
inference, no GPU required.

**Decision layer depth fraction (2026-08-12):** corpus-only CV picks L16/24
at 1.5B ≈ L21/32 at 2.9B — both ≈ **0.67 depth fraction**, independent of
model size. The readout zone tracks depth fraction, not absolute layer index.
Implication for L_state work_layers: L_state should emphasise ~0.67×n_layer.

**Key finding:** *"The base carries the knowledge, the state installs the
disposition."* Abstention = 0/17 at every raw size and model (including
Qwen3.5-4B: 4/17). One epoch sufficient — epochs 2–3 bought nothing.

**Corpus design lesson (pre-registered, 2026-08-11):** Filling distribution
gap (145 tool calls / 6 abstentions → +160 negatives + paired positives)
improved missed abstentions 18→7 (p=0.013 at 2.9B, p=0.002 at 7.2B).
Falsifiable control (name confusions) flat. "Whether a disposition responds
to data is a capacity question" — hardest category inert at 2.9B, active
at 7.2B. Source: `scarletwolf.ai/en/blog/rwkv-enseigner-ou-lire`.

**Format sensitivity:** state does not survive format change — same state
behind different priming dropped 71→58 (real reasoning priming). Empty
`<think></think>` is neutral (71 unchanged, 20/82 generations differ but
same pass/fail). Training eval format must match byte-for-byte.

**ctx_len pitfall:** binidx silently truncates examples longer than ctx_len.
At 20-tool JSON schemas ~6k tok/example; needed ctx_len 4096→6144.
Same class of bug as our step9b T<3 short-circuit (ctx_len=512).

### Inference & state persistence

- **rwkv.cpp:** exposes `state_in` / `state_out` FP32 buffers +
  `rwkv_get_state_len()`. Named save/load API absent, but state is a
  plain vector — `fwrite` / `fread` are trivial. **No canonical
  example** of a server that persists state across HTTP requests.
- **ChatRWKV `API_DEMO_CHAT.py`** and `rwkv_v7_demo_rnn.py` demo
  in-process persistent state: a global `model_state`, each turn
  submits only the new user message, WKV holds accumulated context.
  This is the canonical noesis pattern — but it exists only as an
  in-process script, not a productionised server.
- **RWKV-Runner:** has a trie-based **prompt-prefix cache** (longest
  matched prefix reuses state), but the OpenAI-compat API still
  requires the client to send `messages[]` in full. Not
  session-native. Endpoints `/add-state` / `/longest-prefix-state`
  are commented out in `backend-python/routes/state_cache.py`.
- **Ai00 server:** mounts a `.state` file as the initial condition
  for a session — again, prompt-tuning bias, not per-turn WKV
  persistence.
- **web-rwkv (Rust crate)** explicitly punts: "state caching is not
  included in the library, must be handled at higher level." Nobody
  has implemented that higher level as a shared server pattern.

**Lucas's 2-prefill + Linear merge (2026-08-11, fishy's basement).**
Proposed architecture for non-forgettable system prompt injection:

```python
S_sys = prefill(system_prompt)       # prefill with frozen params
S_ctx = prefill(context)             # prefill with same params
delta = Linear(concat(S_sys, S_ctx)) # 2D → D; only this layer is trained
S    = S_ctx + alpha * delta
decode(S)
```

Post-train: freeze all weights except the Linear layer; use contrastive
system prompts as dataset. ~30 min on a single GPU on G1i. "System
prompt will not have decay (=1) — non-causal, never forgotten." Can be
extended: multiple prefill sources (context, system, persona/style) with
separate merge layers. "The best is to do a gated merge during decode"
(= H16 emit gate at inference time). Implication for noesis: our trained
initial state = S_sys; Lucas's Linear layer is the learned merge we do
not yet have. Could compose: trained L_state disposition + inference-time
Linear merge of context state into disposition.

**Verdict:** stateless-by-turn inference with persistent WKV between
turns is **architecturally supported everywhere and productionised
nowhere.** OpenAI-compat convention (send `messages[]` each turn) is
what wins at the transport layer; server-side WKV persistence
requires sticky sessions, storage, and rollback semantics no one has
committed to as a library.

### G1I model (verified 2026-08-11)

- `rwkv7-g1i-2.9b-20260805-ctx16384.pth` (5.9 GB) on `BlinkDL/rwkv7-g1`
  HuggingFace. ctx=16384 > g1h's 10240. BlinkDL: "better inner
  representation." Already in production in `rwkv-agent` at 13.3B scale.
- `i` suffix = data-version iteration (same RWKV-7 architecture, new
  curriculum / dataset run). Not a new architecture.
- Baseline A0.2 eval vs. g1h-base (39.6%) — not yet measured; download
  in progress as of 2026-08-11.

### WKV state semantic embedding (verified 2026-08-11)

Source: `github.com/cgisky1980/rwkv7-state-embedding` (RWKV-7 0.4B,
albatross inference engine, August 2026).

**Key finding:** Hidden state (last TMix layer, L12 output) contains
semantic information; WKV state alone does not.

| Method | Metric | Value |
|--------|--------|-------|
| WKV state (any aggregation: Q-Readout, row\_sum, diag) | clustering v\_measure | **0.11** |
| Hidden state, unsupervised cosine | STS Spearman | 0.46 |
| Hidden state, unsupervised KMeans | clustering v\_measure | 0.29–0.47 |
| Hidden state + supervised MLP projector (3.15M params) | STS Spearman | **0.82** |
| Hidden state + supervised contrastive projector | clustering v\_measure | **0.95** (MTEB short-text) |
| Hidden state + MLP classifier | task classification | **0.93** |

Root cause of unsupervised failure: **severe anisotropy** in raw hidden
state. Supervised projectors unlock the latent geometry; unsupervised
methods (PCA, UMAP, whitening, DeepCluster) all plateau below 0.35.

**Task-specific projectors cannot be mixed:** STS projector transferred
to clustering drops *below* unsupervised baseline (0.14 < 0.34). STS
learns ranking distance; clustering needs absolute class separation —
objectives are incompatible.

**Implication for noesis.** WKV state being informationally sparse for
external projectors does not undermine lens functionality — lenses feed
WKV state *back to the model*, which reads it natively via WKV
attention. However: any programmatic inspection of lens contents
(H12a working-memory characterisation, H19 contamination probe) requires
a supervised projector. The hidden state (not WKV state) is the natural
signal for `ib_probe`.

### WKV Jacobian analysis — gemlog (verified 2026-08-11)

Source: `clehaxze.tw/gemlog/2026/08-07-notes-on-replicating-j-lens-on-rwkvv7`

Analytical mean-field Jacobian of the WKV update function achieves
**cosine similarity 0.76** with the numerically computed Jacobian on
G1h. This shows the WKV update rule has tractable gradient structure
(no chaotic regime) — the mean-field approximation is tight.

Connection to `L_state`: maximising state motion (Δ between chunks)
trains for large singular values of the WKV update Jacobian. J-lens
(Anthropic's mechanistic interpretability via Jacobians) measures these
eigenvalues post-hoc. `L_state` is a training-time proxy for the same
quantity. **Testable prediction:** step9 checkpoint should show higher
WKV Jacobian eigenvalues in `work_layers` than g1h-base when J-lens
analysis is run post-hoc.

### BLT / T-FREE (community architecture research, 2026-08)

Reported in RWKV Discord; not yet in production RWKV builds.

- **BLT (Byte Latent Transformer):** compresses raw bytes into latent
  patch tokens before passing to WKV / attention. Fewer WKV updates per
  semantic unit → state saturates slower → attractor activates later.
  Architectural solution to the N=3 attractor collapse class of problems.
- **T-FREE (Tokenizer-Free via sparse trigrams):** arXiv 2406.19223
  (Aleph Alpha). Replaces BPE with sparse trigram overlap vectors.
  Composes with BLT as a pure-RWKV-7 byte-level stack.
- **Community recommendation:** BLT + T-FREE for tokenizer-free RWKV-7.
  Not yet trained at reasoning scale (no public checkpoint).

**Implication for noesis.** Architecturally non-applicable to current
A1 (RWKV-7 G1h fixed substrate). The principle maps to an existing
noesis mechanism: `<think>` tokens already function as latent patches —
if the model places intermediate computation inside `<think>` spans, WKV
updates within the span are local-scope and less prone to global attractor
activation. ε-mask (α\_eff = 0.05 outside `<think>`) already enforces
this boundary. BLT/T-FREE is runway context for A2/H24 if RWKV-8+BLT
matures.

### Test-time compute

- **Effort dials in industry** — Anthropic/OpenAI/Gemini expose a
  single scalar (`fast` / `normal` / `thinking`) that translates to a
  CoT-token budget. All treat prompt-conditioned CoT tokens as the
  sole test-time compute mechanism.
- **RWKV community** has no analogous concept of separate refinement
  passes / silent readout modes. The `N` axis (state-refinement over
  prompt without emitting tokens) and `state_readout` mode
  (decoding directly from refined state) are **not community
  patterns** — they follow from the delta-rule framing but are
  unexplored empirically.

### MoE / multi-expert losses (adjacent field, not native RWKV)

- **Switch Transformer, GShard, Mixtral, DynMoLE, SimSMoE** —
  standard toolkit: load-balancing loss (per-expert token count ×
  routing prob), slot-usage entropy, cross-slot cosine dissimilarity.
- All operate on FFN experts in Transformers. **No prior art for
  applying these to state-space recurrent slots** (Mamba multi-head
  state, RWKV multi-slot WKV, etc.).

### State-trajectory losses (SSM / RNN / representation learning)

Prior art exists but goes in the **opposite direction** from what
noesis wants:

- **State-Regularized RNNs** (arXiv 1901.08817) — architectural
  constraint (finite state set), not loss.
- **Slow Feature Analysis / Slow-and-Steady** (arXiv 1506.04714) —
  **minimizes** `‖z_t − z_{t-1}‖` (slowness) and second-derivative
  steadiness. Requires unit-variance constraint against trivial
  constant solution.
- **VICReg / Seq-VCR / LeJEPA** — variance regularization against
  collapse, but variance across dimensions, not temporal delta reward.
- **Decision Mamba** (arXiv 2406.05427) — self-evolution reg for
  offline RL policy trajectory, not hidden state motion.

None of these reward state motion. noesis's `L_state` is a
sign-flipped SFA with per-layer weights derived from empirical
measurement (A0.5) — a mirror-image of known technique, not a
breakthrough.

### Multimodal RWKV

- **VisualRWKV** line (BlinkDL + academic follow-ups) demonstrates
  RWKV variants absorbing visual token streams via bolted-on vision
  encoders. Precedent for H13a's claim that state absorbs geometry.
- **Unified multimodal RWKV** (single state format for text ⊕ image
  ⊕ audio) — not shipped as a community model. H13a wager.

---

## 2. noesis divergences

Things noesis is designing/building that have no direct precedent in
the community landscape above. Not "novel per component" in every
case — sometimes it is a known technique applied to a domain where
nobody applied it, sometimes it is a mirror-image of known
technique. Each divergence is explicit about which.

| Divergence | Nature | Where documented |
|------------|--------|------------------|
| **`L_state` — state-motion reward + curvature reward on load-bearing layers** | Mirror-image of Slow-and-Steady SFA (they minimize, we maximize). Per-layer A0.5-derived weights are empirical noesis contribution. **Risk: no unit-variance analogue anchor; CE alone may not prevent state-norm blow-up.** | `training/state_reg.py`, HYPOTHESES §H8/H9 |
| **Runtime-persistent WKV across sessions with lens-scoped snapshots** | Community has in-process chat demos + trie prefix cache. Server-side session-persistent WKV as first-class abstraction — nobody built it. | HYPOTHESES §H17, plan §5 (lens cache), plan §10 (context transform) |
| **K=4 tail + retrieval instead of full-history injection** | Direct consequence of persistent WKV. Community accepts OpenAI-compat convention (send everything). We wager the substrate holds it. | HYPOTHESES §H17, plan §10 |
| **N/K/readout_mode 3D effort frontier** | Industry uses single-scalar effort dial. RWKV community has no effort framework at all. `state_readout` mode (decode from refined state, no CoT scaffold) — noesis original. | HYPOTHESES §H10, `docs/effort-frontier.md` |
| **Gated externalisation (H16 drip + emit gate)** | RWKV is strictly autoregressive; needs external token to fire. Poll-mode is universal community pattern. Silent drip stream + trained emit gate — nobody built. | HYPOTHESES §H16 |
| **Multi-slot LoRA with utilisation regularizer (H12b + H12b.i)** | Standard MoE toolkit (entropy + dissimilarity + coverage) applied to RWKV WKV state. **First application of MoE anti-collapse tricks to recurrent state slots.** | HYPOTHESES §H12b/§H12b.i |
| **Startup thermal calibration → drip rate derived** | Community drip experiments (rare) all use hardcoded rates. Per-machine calibration + fan-off invariant — noesis specific. | HYPOTHESES §H1, plan §11 |
| **Unified multimodal substrate wager (H13a)** | VisualRWKV is bolted encoder. Unified state (text ⊕ image ⊕ audio through one delta-rule) — no community model ships this. | HYPOTHESES §H13a |
| **Cognitive runtime as an OS-level substrate** | Only `noesis` uid; peer-user sandbox; extensions-as-embodiments (Minecraft, browser, IDE) all bind to the same state. Neither G-Assist (plugin-store bolted on driver) nor Copilot+ PC (cloud-tethered OS feature) has this shape. | README, `docs/policies.md`, `docs/extensions.md` |

**Framing note.** These are hypotheses of the form "if the substrate
holds, this will work"; several sit on H4b (state-evolution wager).
None of them is "novel because nobody thought of it" — each is
novel because nobody staked a system on it.

---

## 3. Pressure points — where deeper research pays

Order = value-per-hour, not urgency.

### 3.1 A1 pilot must instrument state-norm per-layer per-step

`L_state` rewards motion + curvature. If CE anchor is weaker than
the reward signal at some layer, the pathological solution is
`‖s_L‖_F → ∞` without semantic gain. Analogy: SFA needs unit-variance
constraint; we do not have one. Instrument `state_norm_layer_L_step_t`
in A1 tensorboard. If any layer's norm grows > 10× over baseline
during pilot, add explicit norm-anchor loss or clip.

### 3.2 Stateless-mode server pattern as a first citizen

ChatRWKV demonstrates in-process persistent WKV; nobody productised
it. noesis needs: (a) sticky session, (b) WKV serialisation to
`/var/lib/noesis/lenses/<id>/wkv.snapshot`, (c) rollback semantics
for the "user says undo the last turn" case (snapshot on each turn
or accept lossy rollback via re-play), (d) API surface that a
client can drive without knowing about state. This is a
contribution the community would use — if we build it cleanly,
it is worth publishing as a spec.

### 3.3 State Tuning as cold-start for lens hydration

RWKV-PEFT State Tuning trains an initial state vector as
prompt-equivalent bias. noesis lens hydration currently loads a
full WKV snapshot. Alternative: per-lens State-Tuned initial state
(small, cheap to train, portable across model swaps). Could combine:
snapshot for the recent session, State-Tuned bias as the "cold
personality" of a lens. Investigate whether these compose.

### 3.4 arXiv 2504.05097 as external comparison for H12b alternatives

The third-party State Tuning paper (April 2025) also proposes
kernel-based virtual state upscaling (Gaussian kernel to expand
`R^{M×M}`). This is *not* noesis's multi-slot LoRA direction, but
sits in the same problem space (H12b: working-memory width). Read
carefully before H12b LoRA lands — the kernel approach may be
cheaper for the same width gain, or may show it does not work,
saving us the LoRA hours.

### 3.5 DPO / preference pairs for H15 butler-persona

Current H15 spec is SFT-only. Butler character is largely defined
by contrast (what NOT to do — no sycophancy, no padding, no
throat-clearing). DPO natively encodes contrast; SFT does not.
Anthropic HH-RLHF honesty subset as principle source, hand-crafted
positive/negative pairs from user's own chat traces as style
source. Update H15 with DPO as a candidate variant, not just SFT.

### 3.6 Loss for H16 gate itself

H16 spec has the *architecture* (small MLP head over WKV state
producing `p(emit)`) but no loss formulation. Candidates:
supervised (labelled "was this a good moment to speak") vs
self-play (drift from optimal spacing) vs online RL from user
feedback. Each has different data requirements. Pick one before
H16 becomes buildable.

### 3.7 L_length as formal hypothesis

Discussed 2026-07-21 as `β·L_length` alongside `α·L_state` — never
landed in code or HYPOTHESES. Directly relevant to H15 (short
responses) and H16 (dense state-think per emit token). Either
promote to H18 with prediction + falsifier, or write it off with
reasoning.

---

## 4. External claims explicitly rejected

For future sessions: these have been verified as false or
misattributed. Do not re-fold them without new evidence.

- **"State-Consistency Loss"** (penalising WKV motion) — not in
  RWKV-LM, RWKV-PEFT, or paper. Logically **opposite** to `L_state`
  (which rewards motion). Direct contradiction of an external claim
  from an AI-generated write-up.
- **"Vector-Valued In-Context Learning Loss"** — misattribution.
  The paper describes *architectural* delta-rule feature, not a
  training loss.
- **"Triplet-Block Diffusion / DBP in G1h May-June 2026"** — real
  papers (arXiv 2605.25969, 2504.05097) but **third-party academic
  work, not in G1h, not in BlinkDL codebase.** Attribution to G1h
  is false.
- **"3D-sliders" for R/W/K/V axes** — R, W, K, V are architectural
  symbols. No inference-time knob with these names in RWKV-Runner
  or anywhere else.
- **"Kernel-based state upscaling shipped in G1h"** — same as
  above; third-party academic (2504.05097), not in G1h.
- **"Self-Rollouts" (silent padding tokens for hidden thinking) in
  RWKV** — Transformer-line technique (Goyal et al.); not in RWKV
  codebase or docs. Our own H16 drip is the RWKV-native cousin.
- **"G1h = new architecture"** — false. `h` is a data-version
  suffix; architecture is RWKV-7 as in the March 2025 paper.
  `mollysama` is a GGUF re-distributor, not an architectural author.
- **"State Tuning trains running-WKV trajectory"** — false. Trains
  the *initial state vector* (per-layer, fixed-length), effectively
  a numerical prompt bias. Not a trajectory objective.
- **"WKV state is a semantic embedding"** — false. WKV state aggregations
  (Q-Readout, row\_sum, diag, trace) yield clustering v\_measure=0.11,
  far below hidden state (0.93 with supervised projector). WKV state is
  a *computation register*, not a compressed semantic representation.
  Source: cgisky1980/rwkv7-state-embedding, verified 2026-08-11.
- **"RWKV-8 codename is Rose"** — false. Codename is "Heron" 🪶.
  The rose 🌹 is a visual emoji tied to the ROSA mechanism inside
  Heron. Verified against BlinkDL twitter and RWKV-LM repo
  2026-07-30.
- **"RWKV-8 Heron ships production-scale checkpoints"** — false as
  of 2026-07-30. Public released weights are toy-scale (39.6K for
  reverse-sequence, ~1M for 40-digit arithmetic) plus an early-
  testing Pile run. No 0.4B+ RWKV-8 checkpoint at reasoning scale.
- **"RWKV-8 remembers everything verbatim"** — not supported by
  primary sources. The documented ROSA claims are around
  *exact-sequence manipulation* (digit reversal, arithmetic per
  digit, long-range exact matches) and "genuine infinite context"
  as a mechanism property. Neither wording says the network stores
  its training corpus verbatim in weights, and the toy-scale
  checkpoints do not demonstrate that either. Community rephrasing
  as "verbatim recall of the corpus" overshoots the source.
- **"ROSA can be dropped into an existing RWKV-7 checkpoint"** —
  false. `ROSA_QKV_B_1bit` is a network block sitting alongside
  `RWKV_Tmix_x070`; upgrading requires training the ROSA block,
  not just loading a new inference-time module.

---

## Maintenance

- Update this file when a new claim about RWKV community is verified
  or rejected.
- When a noesis divergence lands with empirical support, move it
  from §2 to a "Confirmed" table at the top.
- When a pressure point in §3 gets addressed (probe, decision,
  ablation), record the outcome and prune the entry.
