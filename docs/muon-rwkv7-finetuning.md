# Muon on a trained RWKV-7: where the step size actually goes

*Assembled 2026-09-15 for external review. Audience: RWKV maintainers and anyone
running Muon on a pretrained RWKV-7.*

BlinkDL's standing question is "Muon works for RWKV-7 pretraining; for fine-tuning
a trained RWKV-7 model, no idea — please let us know." This is our answer so far,
written so the parts that depend on our own code are separable from the parts that
do not.

**Summary.** Muon's per-tensor aspect rescale is correct for a standalone weight
and silently wrong for a *factor pair*, because a pair is two tensors of
transposed shape whose product is the update. Attaching a LoRA adapter to a
Muon-trained RWKV-7 therefore steps one factor 9-18x faster than the other under
one shared learning rate. We show this is a step-size effect and not a geometry
effect, give the threshold where it starts to hurt, and give a cheap fix. We also
show that RWKV-7's **own** low-rank pairs escape the same treatment only by an
accident of naming — which is the part of this that is about the architecture
rather than about us.

---

## 1. Measurements anyone can reproduce in minutes

Everything in this section is read off a public checkpoint
(`rwkv7-g1i-2.9b`, n_embd 2560, dim_ffn 10240, n_layer 32) with no training and no
private data. Any RWKV-7 of the same shape gives the same numbers.

### 1.1 The coefficient lands on one member of a pair

The upstream rescale is

```python
update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
```

computed per tensor. A LoRA adapter on a linear layer of shape (d_out, d_in) adds
`lora_A` of shape (r, d_in) and `lora_B` of shape (d_out, r). The model never sees
either one: it sees `ΔW = s·(B·δA + δB·A)`. But the coefficient does not know that,
so at r=32 on this checkpoint:

| tensor | shape | coefficient | effective lr at `--muon-lr 0.002` |
|---|---|---|---|
| `att.{r,k,v,o}.lora_A` | (32, 2560) | 1.000 | 0.0020 |
| `att.{r,k,v,o}.lora_B` | (2560, 32) | **8.944** | **0.0179** |
| `ffn.key.lora_A` | (32, 2560) | 1.000 | 0.0020 |
| `ffn.key.lora_B` | (10240, 32) | **17.889** | **0.0358** |
| `ffn.value.lora_A` | (32, 10240) | 1.000 | 0.0020 |
| `ffn.value.lora_B` | (2560, 32) | 8.944 | 0.0179 |

PEFT initialises `lora_B` to zeros and `lora_A` by kaiming, so the first steps flow
entirely through the factor carrying the larger one.

In our own runs, 0.02 is the learning rate on record as collapsing this model and
0.002 the one on record as stable. At the *stable* setting the B factors are
already stepping at or above the collapsing one. The effective step was never the
configured step.

### 1.2 Lower rank is monotonically worse — which is backwards

The coefficient on `lora_B` is `sqrt(d_out / r)`, so it grows as rank falls:

| r | `att.*` `lora_B` | `ffn.key` `lora_B` |
|---|---|---|
| 128 | 4.472 | 8.944 |
| 64 | 6.325 | 12.649 |
| 32 | 8.944 | 17.889 |
| 16 | 12.649 | 25.298 |
| 8 | 17.889 | 35.777 |
| 4 | 25.298 | 50.596 |

This inverts the usual intuition that a smaller adapter is a gentler intervention.
Under this implementation a smaller adapter is a *larger* step, without touching
the learning rate. At r=4 the `ffn.key` B factor moves at fifty times the
configured rate.

### 1.3 RWKV-7's own factor pairs escape only by a naming accident

RWKV-7 is full of low-rank pairs already. On this checkpoint:

| parameter | shape | coefficient it would receive |
|---|---|---|
| `att.w1` | (2560, 96) | **5.164** |
| `att.a1` | (2560, 96) | **5.164** |
| `att.v1` | (2560, 64) | **6.325** |
| `att.g1` | (2560, 320) | **2.828** |
| `att.w2`, `a2`, `v2`, `g2` | (rank, 2560) | 1.000 |

These are the decay, in-context-learning-rate, value-residual and gate
projections — `w = w0 + tanh(x·W1)·W2` and friends. They are structurally LoRA:
two rectangular factors whose product is the operator.

They are excluded from Muon in every implementation we have seen **only** because
the selection predicate requires the name to end in `.weight`, and these are bare
`nn.Parameter`s. The exclusion is therefore load-bearing and invisible. Anyone who
tidies that predicate — say, to `p.ndim == 2 and ".att." in name` — drops the same
pathology onto the architecture's own factorisation, with coefficients up to 6.3
on the decay path.

**This is the part we would most like checked by someone who knows the reference
implementation's intent.** If the exclusion is deliberate, it deserves a comment
saying so. If it is incidental, it is one refactor away from breaking.

### 1.4 Full fine-tuning is a milder case of the same thing

With no adapter, the only tensor receiving a coefficient above 1.0 is `ffn.key`
(10240, 2560) at **2.000**, while `ffn.value` (2560, 10240) receives 1.000 — two
halves of one FFN block, one shared learning rate, a 2x asymmetry between them.
Every `att.*` projection is square and receives 1.000.

---

## 2. What the damage is, measured on a toy

The toy is a small recurrent controller with a WKV-shaped state, pretrained with
Muon, then LoRA-finetuned on a narrow slice; the number that matters is retention
of the pretrained ability. 6 seeds, four arms, six coefficient levels, with the
coefficient moved by **rank at fixed width** so that capacity is held constant
(moving it by width changes capacity too, and then the effect disappears into it).

### 2.1 It is a threshold, not a dose

Retention relative to Adam on the same base and budget:

| aspect coefficient | 1.41 | 2.00 | 2.83 | 4.00 | 5.66 | 8.00 |
|---|---|---|---|---|---|---|
| damage vs Adam | −0.004 | −0.004 | −0.001 | **−0.524** | **−1.693** | **−1.144** |

Nothing happens up to ~2.83. Above it, the model stops retaining. Note where the
real coefficients from §1.1 sit relative to that line.

### 2.2 It is the step size, not the pairing

This is the control that matters, and it cost us a day to realise we had not run
it. Two obvious interventions — dropping the coefficient, or dividing the shared
lr by it — both remove the damage (100% and 98% of the gap to Adam recovered). But
**both lower the B factor's step**, so neither separates "the two factors were
mismatched" from "one step was simply too big".

The arm that separates them keeps the large step and removes the asymmetry: both
factors at `lr · aspect`, symmetric. If pairing were the mechanism this is
harmless. It is catastrophic — retention −747 at coefficient 4.0 and −9892 at 5.66,
orders of magnitude worse than the production asymmetry it removes, and already
broken at 2.83 where production is still intact.

So the damage is the **size** of the step. The pairing asymmetry matters only
because it carries one factor across the threshold that the configured learning
rate alone would not reach.

### 2.3 What `r` means under this optimizer — it stops being a budget

Worth stating separately, because it is not a step-size effect and it does not go
away when the step is fixed.

Newton–Schulz drives every singular value of each factor to 1, and the product
inherits it. Measured on the induced `ΔW` at init: **rank 32 with σ₃₂/σ₁ = 0.733
and entropy-rank 31.88 of 32** — an almost perfectly flat, full-rank-`r` update,
injected every step by construction, regardless of what the loss wanted. Adam in
the same slot produces a spiky one.

So under factor-wise Muon the adapter's rank is not a ceiling the update may use
if it needs to. It is a **mandate to use all of it, flat, every step**. That is a
different object from what `r` usually denotes.

Set against what the state can hold, on this checkpoint: the rank ceiling of a
WKV write is `min(n_recurrence_steps, head_size)` — one rank-1 write per step,
head_size 64 — and the measured live-direction count per head, at stride 1 over
five prompts, runs **8.8 at L24 to 22.6 at L20, out of 64**. An `r=32` adapter
under this optimizer therefore writes a flat 32-direction update, every step, into
a state that is carrying 9–23.

We have a separate measurement suggesting the extra breadth is not the good part:
in a controlled toy, training arms that deliberately raised the live-direction
count did reach a higher count and scored **worse** held-out than the plain arm
(+0.6085 vs +0.7501). Breadth was buildable and bought nothing. Factor-wise Muon
does that by construction, for free, on every step.

### 2.4 The fix, and the direction it must go in

Equalise the two factors' contributions to the update — `‖B·δA‖` and `‖δB·A‖` —
**downward, to the smaller of the two**, never to their geometric mean. §2.2 is the
evidence that raising any factor's step is what does the harm.

Both norms are computable in r×r space, so the out×in product is never formed:

```
‖B·δA‖²_F = ⟨BᵀB, δA·δAᵀ⟩        ‖δB·A‖²_F = ⟨δBᵀδB, A·Aᵀ⟩
```

Two small matmuls per adapter per step. One edge case: PEFT zeroes `lora_B`, so
`‖B·δA‖` is exactly 0 on step 0 and equalising to a geometric mean of zero would
scale *both* factors to zero — the adapter would never leave the origin. A
vanishing contribution must fall back to no rescale.

---

### 2.5 A state-derived metric: measured, not adopted, offered anyway

One arm replaced Muon's fixed per-tensor scale with one measured from the
**state** — how far a unit weight step actually moves the recurrent state,
recalibrated during training, displacement ratio bounded at 16x. The question it
was built for is whether the object worth normalising is the weight matrix or the
state the weights write into.

Six converged seeds, LoRA factors, same protocol as §2:

| arm | retain | adapt | adapter ΔW entropy-rank (of r=8) |
|---|---|---|---|
| adam | +0.9993 | +1.0000 | 5.92 |
| muon, factor-wise (production) | +0.7441 | +0.9833 | 4.67 |
| muon, contributions equalised | +0.9995 | +0.9976 | 4.67 |
| muon, induced ΔW orthogonalised | +0.9987 | +0.9997 | 5.94 |
| **muon, state-derived metric** | **+0.8558** | **+0.9485** | **6.05** |

We are not adopting it: it recovers +0.112 of factor-wise Muon's −0.256 retention
loss, where two far simpler shape fixes recover +0.255 each. Paying for a running
state measurement to get half of what a reparameterisation gives for free is not a
trade we can justify.

It is reported because one thing in that row is its own and is reproduced by no
other arm: it produces the **broadest** update of the five (6.05 against Adam's
5.92 and the two winners' 4.67) and the **narrowest fit** to the finetune data
(0.9485 against 0.9976–1.0000). Those are the same fact twice — a state-derived
metric spends the update across more directions and therefore commits less of it
to the data in front of it. If the objective is breadth of the learned update
rather than fit — continual learning, multi-task adapters, anything where
over-committing to the current slice is the failure you fear — that arm is aimed
at a different problem than ours, and the numbers are here rather than in a
drawer.

Caveats a reader should carry: toy scale; this arm has the widest seed spread of
the five (±0.169 against ±0.0005 for the best), so its middle number is the least
trustworthy in the table; and it was measured on LoRA factors, not full weights.

---

## 3. What this does not say

**It is not an upstream bug report.** The optimizer class our runs used is our own
file; `light_rwkv.py` references `args.optimizer=='muon'` but no such class was
ever vendored, so that branch references an undefined name and crashes. That, and
only that, is the upstream report. The aspect rescale itself is correct for what
it was written for — a standalone layer weight — and we are not proposing it be
changed for that case.

**It does not explain the full fine-tuning collapse.** Our full-FT Muon runs on
this checkpoint pass every training-stability check and then fail real generation
eval. §1.4's only coefficient above 1.0 is 2.000, which §2.1 measures as
harmless. So that collapse is a second, independent problem and nothing here
should be read as having accounted for it.

**The toy is a toy.** A small recurrent controller with a WKV-shaped state is not
a 2.9B model. §1 is checkpoint arithmetic and holds regardless; §2 is a mechanism
claim at toy scale and should be treated as one until someone reproduces the
threshold at real scale.

---

## 4. What we would like to know

1. **Is the exclusion of `w1/w2`, `a1/a2`, `v1/v2`, `g1/g2` from Muon deliberate?**
   It currently rests on those parameters not being named `.weight`. If it is
   intentional, a comment would protect it; if it is incidental, §1.3 says what a
   refactor would cost.
2. **Has anyone run Muon with LoRA on a pretrained RWKV-7, at what rank and what
   learning rate?** §1.2 predicts that lower rank is worse under the current
   implementation, which is the opposite of the usual expectation, and that is a
   cheap thing to falsify with two runs. A second, separable question is
   §2.3's: under factor-wise Muon, `r` stops being a capacity ceiling and becomes
   a forced flat-rank-`r` update every step. Is that the intended reading of rank
   when Muon is used with a low-rank adapter, or an unexamined side effect?
3. **For pretraining, does the coefficient's behaviour on `ffn.key` vs
   `ffn.value` (§1.4) match what you would expect?** A 2x asymmetry between the
   two halves of one FFN block under one learning rate is below our measured
   damage threshold, but we would rather hear it is intended than assume it.

---

## Reproducing

`§1` needs only a state dict:

```python
for name, t in state_dict.items():
    if t.ndim == 2:
        coeff = max(1, t.shape[0] / t.shape[1]) ** 0.5
```

`§2` is `experiments/A0_state_probe/aspect_dose_probe.py` in this repository, CPU,
about an hour across three shards; the probe carries its own pre-registered
verdict logic and writes `results/aspect_dose_toy.json`. The related arms
(equalising contributions, orthogonalising the induced ΔW, a state-derived metric)
are in `experiments/A0_state_probe/lora_muon_probe.py`.
