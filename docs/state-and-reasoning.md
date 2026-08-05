# State and reasoning in RWKV-7

Reference notes assembled before designing the `A0.4` state-utilisation
probe (see `experiments/A0_state_probe/`). Flat facts + citations, no
interpretation. Interpretation lives in the probe writeup, once
measurements exist.

Sources: arXiv 2503.14456v2 (Peng et al., "RWKV-7 'Goose' with
Expressive Dynamic State Evolution", March 2025); BlinkDL/RWKV-LM
GitHub repo at commit `846b08c1` (2025-03-17); Ollama registry entry
for `mollysama/rwkv-7-g1h`.

## 1. Goose architecture — what changed in RWKV-7

**State update rule (paper §3, Architecture, p. 5).** RWKV-7 replaces
the diagonal-only state transition of RWKV-6 with a rank-one
input-dependent perturbation:

```
    S_t = S_{t-1} · (diag(w_t) + z_t^T b_t) + v_t^T k_t
    with z_t = −κ̂_t,  b_t = κ̂_t ⊙ a_t,  a_t ∈ (0, 1)^d
```

Contrast (paper §3): RWKV-6 was `S_t = S_{t-1} · diag(w_t) + v_t^T k_t`
— transitions were diagonal, so each channel of the state evolved
independently. RWKV-7's `z_t^T b_t` term couples channels through an
input-dependent low-rank update; this is what the paper calls
"generalized delta rule with vector-valued gating and in-context
learning rates" (abstract).

An alternative parametrisation RWKV-7a (paper Appendix I) admits full
negative eigenvalues; used by authors for board-game modelling. Not
the default; not shipped in the World3 checkpoints we use.

**Explicit test-time-learning claim (paper §2, Background, p. 4).**
The state update is described verbatim as "equivalent to a single step
of stochastic gradient descent, training the state S_t at test time to
output the desired values v_t for the keys k_t as inputs". This is the
formal statement of the philosophy we're interested in probing: state
evolution = compute, not just memory.

Note on scope of that claim: it is stated for the *delta rule step*,
not for the sequence as a whole. Whether cumulative multi-token
evolution actually behaves like an ongoing SGD trajectory is a
question the paper does not answer directly and A0.4 partly targets.

**Expressivity results (paper §3 p. 5; Appendix D.1–D.2, pp. 33–36).**

- "RWKV-7 possesses expressive power surpassing that of TC^0 under
  standard complexity conjectures and can recognize all regular
  languages" (§3).
- Theorem 2 (Appendix D.1): RWKV-7 solves an NC^1-complete problem
  under AC^0 reductions.
- Corollary via Lemma 2: RWKV-7 can track swaps on 5 elements.
- Main result (Appendix D.2): recognizes all regular languages with a
  constant number of layers.
- Explicit contrast (paper §3, citing Merrill et al. 2024):
  Transformers and RNNs with diagonal transition matrices are limited
  to functions in TC^0. This is presented as the concrete
  architectural gap RWKV-7 closes.

**State dimensions for 2.9B World3** (paper Appendix E, Table of model
architectures, p. 38):

| quantity        | value |
|-----------------|-------|
| n_layer         | 32    |
| model dim (D)   | 2560  |
| head_size (d_h) | 64    |
| n_head          | 40    |
| WKV state per layer | `n_head × d_h × d_h` = `40 × 64 × 64` = 163 840 elements |
| WKV state per layer (bf16) | ~320 kB |
| Full parameter count | 2 947 735 040 |

Per-token snapshot of the *whole* WKV state (all layers, bf16) is
therefore `32 × 163 840 × 2` B ≈ **10.5 MB** — not the 13 MB × 32 that
the plan's back-of-envelope assumed. Correcting: 256 tokens × 10.5 MB
= 2.7 GB per sequence. Still too large for full retention across all
seeds × models × prompts, so online metric computation remains the
right call, but the state per step is genuinely tractable to hold in
RAM briefly.

**Context extrapolation (paper §7.5, Long Context, p. 13).** RWKV-7
2.9B reliably retrieves passkeys up to 30k tokens; degrades near 50k.
Trained on ≤128k. Not directly relevant to the probe but bounds the
lengths at which state dynamics can be honestly measured.

**Authors' own state instrumentation (paper Appendix J, State
Inspections, p. 50).** Authors themselves examine WKV state matrices:

- Metrics used: root-mean-square of matrix elements (RMS) and stable
  rank `SR(A) = (‖A‖_F / ‖A‖_2)^2`.
- Corpus: 10 PG19 validation samples, each ≥ 8192 tokens.
- Comparison across RWKV-5 / RWKV-6 / RWKV-7 at 1.5B.
- Reported example: "Layer 0 Head 4, SR: 2.03, RMS: 96.74".

**Implication for A0.4**: authors provide a *specific* pair of metrics
(RMS, SR) they consider meaningful. Adopting SR as one of A0.4's three
metrics has the added benefit of direct comparability to the paper.
The plan's original three metrics (delta norm, layer entropy,
curvature) do not overlap with SR — recommend adding SR as a fourth
metric or replacing layer-entropy with SR. Decision pending A0.4
design.

## 2. G1 training

**No G1 documentation in RWKV-LM repo** (commit `846b08c1`). Grep for
`G1`, `g1_`, `reasoning`, `<think>` in the repo returns nothing
substantive. G1 training code and configuration are *not* in the main
BlinkDL/RWKV-LM repository as of this commit. The G1 corpus, curriculum
and reasoning-markup scheme are documented only in external artifacts
(BlinkDL blog, HF/Ollama model cards, mollysama's registry). This is a
material gap — we cannot reproduce G1 training from RWKV-LM alone.

**Standard World3 training config visible in the repo** (`RWKV-v7/README.md`
+ `RWKV-v5/train.py`, lines 14–122):

- Training entry: use `RWKV-v5/train.py` with `--my_testing "x070"`
  and `head_size_a=64`. Same script as v5, versioned by flag.
- Tokenizer: RWKV World (vocab 65 536).
- Data: RWKV World v3 corpus, 3.1 T tokens, multilingual, open.
- Framework: PyTorch Lightning + DeepSpeed.
- Precision: bf16 activations + fp32 kernel for WKV (paper §8, p. 15).

Nothing in that config is G1-specific.

**LoRA / adapter tooling is external.** RWKV-LM README points to two
outside repos:
- `JL-er/RWKV-PEFT` — LoRA, QLoRA, PiSSA, Qpissa, **State Tuning**.
  State Tuning is *initial-state prompt-tuning*: a single learnable
  `[n_layer, n_head, head_size, head_size]` vector is prepended so the
  base model starts each sequence from a trained initial WKV state.
  The trajectory across tokens is not part of the loss — CE alone
  drives training. See `docs/community-map.md` §1 for the full
  landscape; noesis's `L_state` (state-motion + curvature reward) has
  no direct prior art in the RWKV community.
- `Blealtan/RWKV-LM-LoRA` — infinite-ctxlen training branch.

For any A1 fine-tune step involving state, `RWKV-PEFT` is the
reference implementation to check first, not RWKV-LM proper.

**No built-in state-inspection utilities in RWKV-LM.** State variables
are accessed only in RNN-mode inference demo:
- File: `RWKV-v7/rwkv_v7_demo_rnn.py`, lines 92–102 (forward pass
  access) and 284–288 (initialisation).
- Layout of state list: `state[3·i + 0]` = attention `x_prev` (shape
  `[D]`), `state[3·i + 1]` = **WKV state** (shape
  `[n_head, head_size, head_size]`), `state[3·i + 2]` = channel-mixing
  `x_prev` (shape `[D]`).
- No visualisation, no probing helpers — the probe writes these itself.

## 3. Model availability & weight format

**Native bf16/fp16 weights.** Attempts to fetch model cards from
`huggingface.co/mollysama/RWKV-7-G1h` and `huggingface.co/BlinkDL/rwkv-7-g1`
returned 401 during the literature scan. HuggingFace public model
cards should not require auth for read; suspect this is a scan-side
rate-limit / redirect issue rather than genuine gating. **Manual
verification required before the probe session**; if native bf16
weights are actually gated, we fall back to (a) BlinkDL's non-G1
World3 weights (public, native) as the only World-side data point,
paired with (b) a G1 checkpoint from BlinkDL's release page — likely
`RWKV-x070-Goose-World3-2.9B-*.pth` on HF `RWKV/` org, which paper §6
(Pre-Trained Models, p. 9) lists as the released set.

**Available public sizes** (paper §6, p. 9): RWKV-7 World v3 at 0.19B,
0.4B, 1.5B, 2.9B. Training tokens 1.6–5.6 T depending on size.
Official release channel: `https://huggingface.co/RWKV`.

**Ollama registry (`mollysama/rwkv-7-g1h`).** Weights are GGUF (Q4_K_M
for our pulled 2.9 GB tag). Quantisation method disclosed only as
GGUF, no specific Q level for the base card. **Not suitable for the
probe** — Q4 quantisation of weights is expected to distort state
trajectories at a level that would confound H8/H9 measurement. Paper
§8 (p. 15) explicitly uses fp32 kernel for WKV during training; we
should mirror that at inference for the probe (bf16 weights, fp32 WKV
accumulator).

## 4. WKV state as an Information Bottleneck channel

**Framing.** RWKV-7's WKV state is a fixed-capacity, lossy compression
channel. All input the model has ever seen `X_{1..t}` must be
represented in the same `10.5 MB` per-layer-summed matrix; whatever
does not fit is lost. This maps directly onto Tishby et al.'s
Information Bottleneck (Tishby, Pereira & Bialek, *The Information
Bottleneck Method*, 1999; arXiv 0004057), which formalises finding a
compressed representation `Z` of an input `X` that maximally preserves
information about a target `Y` under a fixed channel capacity:

```
    min  I(X ; Z) - β · I(Z ; Y)
     Z
```

Here `X` = input token stream (context window content), `Z` = WKV
state after ingesting `X`, `Y` = the downstream prediction target
(next token, or a task-relevant readout). The Lagrangian says:
minimise redundant information the state carries about the input,
while maximising the information the state carries about what
actually matters downstream. `β` is the tradeoff — under noesis's
constraints, `β` is *not* a free hyperparameter; it is fixed by the
architecture (state size) and the training objective (CE + any
auxiliary `L_state`).

**Why this framing is load-bearing for noesis (not just formal window
dressing).**

- **P4/P5 (constant cost, cheap by construction)** foreclose growing
  the state to fit more input. The channel width is what the shipped
  architecture gives us; we do not scale it away.
- **P1 (state external, cognition internal)** does the equivalent of
  "the state does not have to hold everything, retrieval fills gaps"
  — an IB architecture with an *external* memory tier at hand does
  not have to allocate its bits to storing facts, and can spend them
  on reasoning-relevant abstractions instead. This is not a hedge
  around the bottleneck; it is the *architectural response* to it.
- **P14 (agility over omniscience)** is the direct pragmatic
  consequence: the bottleneck *cannot* carry everything, so a system
  that pretends to must be confabulating. Honest not-knowing is
  what a well-calibrated IB channel *should* produce when the
  requested `I(Z;Y)` cannot be supported by the channel width.
- **P13 (reasoning in state, not in tokens)** raises the stakes:
  if state *is* the computation, then the IB channel is not just
  a memory allocation problem — every bit spent on redundant input
  representation is a bit *not* spent on computation.

**Practical consequences that shape design and hypotheses.**

1. **Forgetting is necessary and structured, not incidental.** A
   fixed-width channel *must* discard information about `X` to leave
   room for information about `Y`. What gets forgotten is
   downstream-task-dependent, not just recency-dependent. This
   reframes "why does the model forget X" from a bug to a
   consequence of what it is optimising `I(Z;Y)` for. The right
   research question is not "how do we make the state remember more"
   but "how do we make the state forget the right things".
2. **Long-horizon planning has a hard ceiling.** Any task that
   requires threading `> capacity` bits of state through a decision
   sequence *cannot* be solved by state alone — it must decompose
   into a chain of state + external-memory reads (P1) or a chain
   of state resets keyed by retrieval. Trying to lengthen the
   effective context by pushing more tokens through the same WKV
   is subject to diminishing returns bounded by channel capacity,
   not by context window length.
3. **Uneven importance is real signal, not artifact.** If the model
   allocates disproportionate `I(Z;Y)` bits to some inputs over
   others, that allocation is the model's own *learned* importance
   function — noesis can read it (a state-readout head on the
   distribution of "what did the state retain") and use it directly
   as an emit-gate, a retrieval-relevance signal, or a compression
   heuristic. This is H16 and H10 seen through IB glasses.
4. **`L_state` has a formal target.** The state-motion + curvature
   reward sketched in the community-map (§2) is a proxy for
   pushing `I(Z;Y)` up per unit `I(X;Z)`; making the training
   objective explicitly IB-shaped (measure `I(Z;Y)` on a proxy
   downstream task and reward states that carry it) is an option we
   have not exercised. Noted for A2 planning.

**Reconstruction / mutual-information probes as measurement.** IB
gives a first-principles handle on H8 and H12b: rather than only
measuring state *dynamics* (delta-norm, curvature, SR — §1 above),
also measure state *content* by training a small decoder from `Z`
to reconstruct `X` (upper-bounds `I(X;Z)`) or to predict `Y`
(lower-bounds `I(Z;Y)`). These are standard IB estimation moves
(cf. Alemi et al., *Deep Variational Information Bottleneck*, 2017;
Saxe et al., *On the Information Bottleneck Theory of Deep
Learning*, ICLR 2018 — with the caveat that Saxe et al. showed the
IB "phase transition" narrative was tanh-nonlinearity-specific, so
we import IB as *framing*, not as a prediction about training
dynamics). Concretely for noesis:

- **Reconstruction probe (upper bound `I(X;Z)`).** Train an MLP
  head on frozen WKV state at position `t` to reconstruct tokens
  at positions `t-k .. t`. Reconstruction quality vs. `k` gives a
  practical decay curve — how many past tokens the state materially
  preserves.
- **Downstream probe (lower bound `I(Z;Y)`).** Same head, different
  target: task-relevant readout (H21 premise-validity is one
  candidate; H20 aporia-hold is another). Ratio of downstream-probe
  quality to reconstruction quality is the *bit-efficiency* of the
  state for that task.
- **H12 as capacity dimensions.** Width (`n_head × head_size²`) and
  decay `w_t` are the two independent capacity knobs — width sets
  raw channel size, decay sets how quickly channel is freed up.
  IB framing predicts: at fixed width, tighter decay (faster
  forgetting) should *increase* `I(Z;Y)` for tasks whose relevant
  horizon is short, and *decrease* it for long-horizon tasks. Direct
  falsifier for H12b utilisation claims.

**Runtime metric candidate: channel-budget accounting.** noesis
currently accounts for compute in CPU-seconds (calibration protocol,
policies.md § CPU and thermal governance). An IB-shaped runtime
would *additionally* account for `I(X;Z)` bits consumed per input —
how much of the state's channel capacity a given input burned. A
long noisy transcript that contributed nothing measurable to
downstream tasks would show up as a *channel-budget cost* even if
its CPU cost was small. Not a proposal to build this now; a note
that the accounting hook exists in principle and would give the
truth-system another signal to work with (H16 emit-gate could
consult it: "did the last minute of input actually change what the
state carries?" as an emit criterion).

**Retrieval scoring by info-gain (parking lot).** Once
reconstruction/downstream probes are in place, retrieval candidates
can be scored by *estimated `I(Z_after; Y) − I(Z_before; Y)`* — pick
the candidate that most raises task-relevant channel content. This
is a stronger objective than cosine similarity in embedding space
(similarity is at best a proxy for `I(X_retrieved; X_query)`, which
is only weakly related to what actually helps the state). Not
scheduled — noted here so future A3/retrieval design has the
framing to reach for.

**What this framing does not claim.**

- Not a claim that noesis will *train* the state via an explicit IB
  loss (though `L_state` is a step in that direction). The state is
  currently trained implicitly via CE on next-token prediction; IB
  is a *description* of what that training is doing under a
  fixed-capacity channel constraint.
- Not a claim that RWKV-7 achieves the IB *optimum* — likely far
  from it. IB is the frame that says the optimum *exists* and is
  measurable; how far current weights sit from it is an empirical
  question the reconstruction/downstream probe pair could answer.
- Not a claim that IB dictates architecture. RWKV-7 was chosen for
  reasons in P4/H4b/H8; IB gives a language for reasoning about the
  consequences of that choice, not a competing justification for it.

## What this means for A0.4 design

Not interpretation of the probe results (those come later) — only
things this literature review changes about the *design* itself:

1. **Adopt SR as one of the state metrics.** Authors use it; direct
   comparability to Appendix J of the paper is worth more than the
   layer-entropy metric I originally sketched. Keep delta-norm and
   curvature; add SR; drop layer-entropy (or keep as fourth if it
   fits).
2. **Per-token state snapshot is 10.5 MB, not ~420 MB.** The plan's
   volume estimate was wrong (I over-multiplied). Full sequence
   (256 tokens) = 2.7 GB. Still online-metric territory for the full
   experiment budget, but individual sequences can be held in RAM.
3. **State layout is well-defined** — `state[3i+1]` is WKV per layer,
   shape `[n_head, head_size, head_size]`. Direct hook on RWKV block
   forward, no monkey-patching required.
4. **G1h weights resolved (2026-08-05).** `rwkv7-g1h-2.9b-20260710-ctx10240.pth`
   is available at `~/.libs/models/rwkv7/` locally and at `/root/.libs/models/rwkv7/`
   on the VM. H8 and H9 have been run and SUPPORTED at both 0.4B and 2.9B scale.
   See `experiments/A0_state_probe/results/a05_2.9b_h9_verdict.md` for numerical results.
5. **`JL-er/RWKV-PEFT` "State Tuning" is initial-state prompt-tuning,
   not a trajectory objective.** Verified 2026-07-25 — a single
   learnable initial WKV vector, CE-only loss. The community is *not*
   training against state trajectories. This means noesis's `L_state`
   (state-motion reward + curvature) is an unclaimed slot in the
   design space, not a re-invention. Landscape and adjacent work
   catalogued in `docs/community-map.md` (§1 for what ships, §2 for
   noesis's divergences, §4 for external claims we rejected during
   verification).
6. **Test-time-learning claim is per-step, not sequence-scope.** The
   paper's SGD-step framing (§2) is limited to a single delta-rule
   application. Cumulative state evolution being SGD-like is a
   *stronger* claim the paper does not make; H8's falsification bar
   must account for this — a single non-linear step is trivially
   SGD-like; we're really testing whether the sequence-length dynamics
   accumulate meaningfully.
