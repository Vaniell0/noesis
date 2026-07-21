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
- `JL-er/RWKV-PEFT` — LoRA, QLoRA, PiSSA, Qpissa, **state tuning**
  (last item is notable — state-tuning suggests the community already
  operates on state trajectories as a training target).
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
4. **G1h weight availability is unresolved** — must verify HF
   accessibility manually before the probe session. Fallback: any
   BlinkDL G1 checkpoint (even a different sibling like G1e/G1f) is
   acceptable *if* it's native bf16.
5. **`JL-er/RWKV-PEFT` has "state tuning"** — worth reading before we
   design A1, because the community may already be training against
   state trajectories, which would change what "original contribution"
   means for the state-regularised loss branch in the plan's decision
   gate.
6. **Test-time-learning claim is per-step, not sequence-scope.** The
   paper's SGD-step framing (§2) is limited to a single delta-rule
   application. Cumulative state evolution being SGD-like is a
   *stronger* claim the paper does not make; H8's falsification bar
   must account for this — a single non-linear step is trivially
   SGD-like; we're really testing whether the sequence-length dynamics
   accumulate meaningfully.
