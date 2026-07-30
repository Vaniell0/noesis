# Community map — RWKV / SSM landscape vs noesis divergences

Written 2026-07-25. Consolidates prior-art scattered across
`docs/state-and-reasoning.md §2`, "Frontier adjacency" boxes inside
`HYPOTHESES.md` H8/H12, and per-topic snippets in
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

**Verdict:** stateless-by-turn inference with persistent WKV between
turns is **architecturally supported everywhere and productionised
nowhere.** OpenAI-compat convention (send `messages[]` each turn) is
what wins at the transport layer; server-side WKV persistence
requires sticky sessions, storage, and rollback semantics no one has
committed to as a library.

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
