# State and reasoning in RWKV-7: proofs and reference

The frame (`hypotheses/README.md` §3) states what follows from the setup alone. This document proves
those statements and keeps the sourced facts about the architecture they rest on. Notation is the
frame's (§2). Rewritten 2026-11-11 from the literature notes of 2026-07/08, which prepared the first
state probe; their facts are kept in §6-§7.

Sources: arXiv 2503.14456v2 (Peng et al., "RWKV-7 'Goose' with Expressive Dynamic State Evolution",
2025); the reference kernel `fla/ops/rwkv7/recurrent_naive.py`; `docs/rwkv7-mechanics.md` (the code
path, checked line by line).

## 1. The step

Per head, in the kernel's layout (value × key), with `S ∈ R^{n×n}`, `n = 64`:

    S_t = S_{t−1} A_t + v_t k̃_tᵀ,        A_t = D_t − κ̂_t (α_t ⊙ κ̂_t)ᵀ,        o_t = S_t r_t

- `D_t = diag(d_t)`, `d_t = exp(−exp(w_t))`. The model soft-clamps `w ≤ −0.5`, so `d ∈ [0.5452, 1)`.
- `κ̂_t` is the erase key, unit norm; `α_t ∈ (0, 1)^n` the erase gate (a sigmoid).
- `k̃_t` the write key, `v_t` the value, `r_t` the receptance.

The paper writes the same step with row vectors, `S_t = S_{t−1}(diag(w_t) + z_tᵀ b_t) + v_tᵀ k_t` with
`z = −κ̂`, `b = κ̂ ⊙ a` (§3, p. 5); the kernel computes `S·diag(d) + (S a) bᵀ + v kᵀ` with `a = −κ̂`,
`b = κ̂ ⊙ α`. The toy cell `experiments/A0_state_probe/micro_wkv.py` stores the transpose (key × value).
Every statement below holds in either layout.

The controls `(r, w, k̃, v, α, κ̂)_t` are functions of the layer's input at step `t`. Between layers and
across ticks that input depends on states read earlier; this is what §4 calls the closed loop.

## 2. The state is a linear map trained in context

**Identity.** For the loss `L(S) = ½ ‖S κ̂ − u‖²` of a linear map that should send the key `κ̂` to the
value `u`, one gradient step with rate `η` is

    S − η (S κ̂ − u) κ̂ᵀ = S (I − η κ̂ κ̂ᵀ) + η u κ̂ᵀ.

This equals the step of §1 when `d_t = 1`, `α_t = η·1` and `v_t k̃_tᵀ = η u κ̂ᵀ`. RWKV-7 generalises the
three: a per-channel rate `α` (the paper's in-context learning rate), retention `d` (weight decay), and a
write key decoupled from the erase key. The paper states the step as "equivalent to a single step of
stochastic gradient descent, training the state S_t at test time to output the desired values v_t for
the keys k_t as inputs" (§2, p. 4). The claim is per step; whether a sequence of steps behaves like an
optimisation trajectory is not claimed there.

**Consequences for capacity** (linear associative memory):

- *Crosstalk.* With pure writes `S = Σ_i u_i κ̂_iᵀ`, reading at `κ̂_j` gives `u_j + Σ_{i≠j} (κ̂_i·κ̂_j) u_i`.
  Exact recall needs orthogonal keys, so at most `n` items.
- *Overwrite.* The erase term removes what `S` currently returns at `κ̂` before writing. Repeated over a
  set of bindings the delta rule converges to the least-squares map, exact for up to `n` linearly
  independent keys.
- *Horizon.* Retention below 1 removes a binding at rate `d` per step on the channels it occupies.

The size `n × n` is a ceiling; key overlap and decay set what is recalled. This is the basis of `H12.md`.

## 3. Lemmas with the controls frozen

### 3.1 Tick lemma

**Statement.** Let the controls be constant, `A_t = A = D − κ̂ (α ⊙ κ̂)ᵀ`, `C = v k̃ᵀ`. Then:

1. `A` is similar to the symmetric matrix `B = D − w wᵀ`, `w = √α ⊙ κ̂`; its spectrum is real.
2. With eigenvalues `λ₁ ≥ … ≥ λₙ` and retentions sorted `d₍₁₎ ≥ … ≥ d₍ₙ₎`: `d₍ᵢ₊₁₎ ≤ λᵢ ≤ d₍ᵢ₎`, and
   `λₙ ≥ d_min − ‖w‖² ≥ d_min − max α > −1`.
3. `ρ(A) = λ₁ ∈ [d₍₂₎, d₍₁₎]`, so `ρ(A) < 1`, and no erase gate moves it past the two largest retentions.
4. `S_t = S* + (S₀ − S*) Aᵗ` with `S* = C (I − A)⁻¹` of rank at most 1, and
   `‖S_t − S*‖ ≤ √(max α / min α) · ρ(A)ᵗ · ‖S₀ − S*‖`.

**Proof.** Write `Λ = diag(α)`, so `(α ⊙ κ̂)ᵀ = κ̂ᵀ Λ` and `A = D − κ̂ κ̂ᵀ Λ`. Diagonal matrices commute,
so `Λ^{½} A Λ^{−½} = D − (Λ^{½} κ̂)(Λ^{½} κ̂)ᵀ = B`, which is symmetric (1). `B` is `D` minus a rank-1
positive semidefinite matrix; by Weyl's inequality `λᵢ ≤ d₍ᵢ₎`, and by Cauchy interlacing for a rank-1
update `λᵢ ≥ d₍ᵢ₊₁₎`. The smallest eigenvalue is at least `d_min − ‖w‖²`, and
`‖w‖² = Σ αⱼ κ̂ⱼ² ≤ max α · ‖κ̂‖² = max α < 1` (2). Since `d_min ≥ 0.5452`, `λₙ > −0.4548`, while
`λ₁ ≥ d₍₂₎ ≥ 0.5452`; so `|λₙ| < λ₁` and `ρ(A) = λ₁ ≤ d₍₁₎ < 1` (3, for `n ≥ 2`). For (4), `S* = S* A + C`
has the unique solution `C (I − A)⁻¹` because 1 is not an eigenvalue, and its rank is at most that of
`C = v k̃ᵀ`. Subtracting, `S_t − S* = (S_{t−1} − S*) A`. Finally `Aᵗ = Λ^{−½} Bᵗ Λ^{½}` and
`‖Bᵗ‖ = ρᵗ`, so `‖Aᵗ‖ ≤ ‖Λ^{−½}‖ ‖Λ^{½}‖ ρᵗ = √(max α / min α) ρᵗ`. ∎

**Reading.** Settling and forgetting are one number: the part of the state left by the prompt and by
earlier turns decays as `Aᵗ`, at the rate at which the loop settles.

**Premises.** Exact for layer 0 when the same vector is fed every tick, since layer 0's controls depend
on its input only. For deeper layers it holds asymptotically once the layers below have settled. It
says nothing about the closed loop, where the fed latent, and with it every `A_t`, changes from tick to
tick.

**Checks.** 5000 random draws at `n = 64` with `d ∈ [0.5452, 1)`, `α ∈ (0, 1)` and unit `κ̂`: no complex
eigenvalue, no `λ₁` outside `[d₍₂₎, d₍₁₎]`, no `λₙ ≤ d_min − max α`, no fixed point of rank above 1, the
decay bound never exceeded (session check of 2026-11-11). In the repository:
`micro_wkv.py::jacobian_spectral_sampling` (2000 draws at `n = 8`: `ρ` mean 0.9527, none above 1, none
complex).

### 3.2 Open-loop lemma

**Statement.** Call a cell *open-loop* when every control is a function of the step's input `x_t` and of
`t`, never of the state. With `S₀ = 0`,

    S_T = Σ_{t ≤ T} v_t k̃_tᵀ A_{t+1} ⋯ A_T,        o_T = S_T r_T,

and every factor depends on a single input.

**Proof.** Unroll `S_t = S_{t−1} A_t + v_t k̃_tᵀ` from `S₀ = 0`. ∎

So information from two different times meets only through the products of `A` (decay and erase) and
the final read `r_T`; writes alone superpose.

*Corollary 1 (additivity).* Let `a` arrive at the first step and `b` at the last, blanks between. If
`r_T` and `d_T` are constants and the erase gate is 0 at every step, then `A_t = D_t` and
`o_T = f(a) + g(b)`. For independent, zero-mean, symmetric `a` and `b` and any linear readout `y`,
`E[y·ab] = 0`, hence `R²(y, ab) = 1 − (E[y²] + Var(ab)) / Var(ab) = −E[y²] / Var(ab) ≤ 0`.

*Corollary 2 (silent steps).* A step with blank input has constant controls: `S ↦ S A + C`. A run of
silent steps composes to `S ↦ S A_tot + C_tot`; followed by a read `r` it gives `o = S (A_tot r) + C_tot r`.
When the final read is a free parameter and `A_tot` is invertible (the generic case), `A_tot r` ranges
over all vectors, so the set of computable functions is the same for every number of silent steps.

**Checks.** `micro_wkv.py --verify` [4] (mixed second difference of the frozen cell at gate 0: 2.7e-15;
0.22 with the learned gate) and [5] (the parity cell's logit equals `(m·k₀(x))(u·v₀(x)) + c` over all 256
patterns at 0, 2 and 8 silent steps, to 7e-15).

**Reading.** The real model is not open-loop: within a token, layer `l` computes its controls from the
output of layer `l − 1`, which read its state; across ticks, the fed latent is the last layer's output.
Any gain from silent time has to come from that closure.

## 4. Information bounds

**Data processing.** With the input fixed, the state after any number of ticks is a function of the
state after reading it, `S₀`; so `I(S_t; Y) ≤ I(S₀; Y)`. Ticks raise the usable information `I_V`, the
part a bounded reader (the model's remaining layers and head) extracts, never the information itself.

**Input bound.** A fact absent from the input and from the weights is out of reach for any number of
ticks. A reward with `|R|` outcomes carries at most `log₂|R|` bits per episode, so reinforcement
reweights what the model holds; new facts come only from a teacher or a context that has them. What is
derivable from the weights by computation counts as held, though it is not stored as a separate item
(`H12.md` §4).

## 5. The state as an information bottleneck

The state is a fixed-capacity, lossy channel: everything the model has read must be represented in
`n_layer × n_head × 64 × 64` numbers, and what does not fit is lost. This is the setting of the
information bottleneck (Tishby, Pereira, Bialek 1999, arXiv physics/0004057): a representation `Z` of
input `X` that keeps information about a target `Y` under a capacity limit,

    min_Z  I(X; Z) − β · I(Z; Y),

with `Z` the state after reading `X`, and `Y` the next token or a task readout. Here `β` is not a free
knob: the architecture fixes the channel and the training objective fixes what is kept. IB is used as a
description of what training does under a fixed channel, not as a prediction about training dynamics
(Saxe et al., ICLR 2018, showed the "phase transition" narrative to be specific to tanh networks).

What follows for the programme:

1. *Forgetting is required, and structured by the task.* A fixed channel must drop information about
   `X` to keep information about `Y`. The question is not how to make the state remember more but how to
   make it drop the right things. The tick lemma gives the mechanism inside one head: forgetting runs
   at `ρ(A)`.
2. *Long horizons need a second store.* A task that threads more information through a decision
   sequence than the channel holds cannot be done in the state alone; it needs exact recall beside it
   (`HZLK.md` §6, QKV-ROSA).
3. *Uneven allocation is a signal.* What the state keeps is the model's learned importance; readouts of
   it are instruments (H19-H22).
4. *Capacity has three knobs, not two.* Width (`n_head × 64²`), decay, and the overlap of the keys the
   weights write with (§2); measured, the third binds first (`H12.md`).

Estimators that follow from the framing: a reconstruction probe (a head on the state at `t` that
recovers tokens `t − k .. t`; an upper-bound view of `I(X; Z)` and a decay curve), and a downstream
probe (the same head predicting a task variable; a lower bound on `I(Z; Y)`). The ratio is the state's
bit-efficiency for that task. Measured instances: held-out linear IPC (`H10.md` §5) and the readout
lens and capacity probe (`H25.md` §5).

## 6. Architecture facts

**Expressivity** (paper §3 p. 5; Appendix D.1-D.2, pp. 33-36): RWKV-7 recognises all regular languages
with a constant number of layers and solves an NC¹-complete problem under AC⁰ reductions (tracking swaps
on 5 elements), beyond TC⁰, the class bounding Transformers and RNNs with diagonal transitions (citing
Merrill et al. 2024). These results are about transitions that change with the input token. With a
constant fed vector the transition is fixed and §3.1 applies; ticks inherit this expressivity only
through the closed loop. A variant with full negative eigenvalues (RWKV-7a, Appendix I) exists and is
not in the shipped checkpoints.

**State size** (paper Appendix E, p. 38), with `head_size = 64` across the line:

| model | n_layer | D | n_head | state, all layers, bf16 |
|---|---|---|---|---|
| 0.4B | 24 | 1024 | 16 | 3.1 MB |
| 2.9B | 32 | 2560 | 40 | 10.5 MB |

Width buys more 64 × 64 memories per layer, never larger ones; `head_size` is the rank ceiling of each
head's memory (`docs/muon-rwkv7-finetuning.md` §5).

**Context** (paper §7.5, p. 13): 2.9B retrieves passkeys reliably to about 30k tokens and degrades near
50k; trained on up to 128k.

**The authors' own state instruments** (paper Appendix J, p. 50): RMS of the elements and stable rank
`(‖A‖_F / ‖A‖_2)²` on PG19 samples of at least 8192 tokens (example: layer 0 head 4, stable rank 2.03).
Stable rank is an energy-concentration ratio, not a count of live directions: a matrix with one dominant
and thirty-one real singular values scores near 1 (`docs/effort-frontier.md`, 2026-09-13). The project's
1.1-1.3 sits in that regime while 13-16 of 64 directions are live per head on G1i.

## 7. Reference facts about the line

- G1 training code and corpus are not in `BlinkDL/RWKV-LM` (checked at commit `846b08c1`); the World v3
  recipe is (`RWKV-v5/train.py --my_testing "x070"`, `head_size_a=64`, World tokenizer of 65 536 tokens,
  bf16 activations with an fp32 WKV kernel).
- State tuning in `JL-er/RWKV-PEFT` learns one initial state, `[n_layer, n_head, 64, 64]`, with CE alone;
  the trajectory is not in the loss.
- In the RNN demo (`RWKV-v7/rwkv_v7_demo_rnn.py`) the state list holds, per layer `i`, the attention
  shift vector at `3i`, the WKV state `[n_head, 64, 64]` at `3i + 1`, and the channel-mix shift vector at
  `3i + 2`.
- GGUF Q4 weights (Ollama) distort state trajectories and are not used for state measurements; probes
  run bf16 weights with an fp32 WKV accumulator, as in training.
