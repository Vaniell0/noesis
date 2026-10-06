# Muon on a trained RWKV-7: where the step size actually goes

*Assembled 2026-09-15 for external review; §2.6 and the list of what is not yet done
(Limitations) added 2026-10-06. Audience: RWKV maintainers and anyone running Muon on a
pretrained RWKV-7.*

**Attach a LoRA adapter to a Muon-trained RWKV-7 and one factor steps 9-18x
faster than the other under one shared learning rate.** At r=32 and n_embd 2560,
`lora_A` receives a shape coefficient of 1.000 and `lora_B` receives 8.944;
`ffn.key`'s `lora_B` receives 17.889. At `--muon-lr 0.002` — the value we had on
record as stable — those factors are stepping at 0.0179 and 0.0358, at and above
the 0.02 we had on record as collapsing the model. The configured step was never
the step.

The cause is that Muon's per-tensor aspect rescale is correct for a standalone
weight and blind to a *factor pair*, which is two tensors of transposed shape
whose product is the update. Below: the arithmetic off a public checkpoint (§1),
what the damage actually is — a step-size threshold near coefficient 2.83, not a
geometry effect (§2), a fix costing two small matmuls (§2.4), and where this sits
relative to current Muon work (§3).

The part we would most want a maintainer to see is §1.3. The reference
implementation already keeps every low-rank factor away from Muon, deliberately
and completely — but it does so through a naming convention that catches nothing
once the names change to RWKV-LM's, and nothing at all when the factors come from
a PEFT adapter.

A second reading of the same measurements is in §2.6: held at a matched step rather
than a matched learning rate, Muon is usable where AdamW is not.

Written in answer to BlinkDL's standing question: *"muon works for rwkv7
pretraining, but for finetuning a trained rwkv7 model, no idea. please let us
know."* Why we were running Muon, what has since failed, and what we can no
longer defend are in **Limitations** at the end, where they belong.

---

## 0. Why Muon

**A mechanism result, not memory.** On 2026-08-23, testing something else, we
found that a delta-rule erase-rewrite channel which Adam's solutions relied on
heavily stops mattering under Muon at matched accuracy: forcing `a_gate = 0`
destroys Adam's solution on every seed (−1.376 ± 1.326, worst −3.30) and barely
touches Muon's (−0.029 ± 0.032). A 10-seed rerun held the shape (−2.619 ± 2.223
against −0.295 ± 0.563, distributions not overlapping) at matched R². Same task,
same architecture, same accuracy, reliably different internal solution.

**And it fits.** Muon carries one momentum buffer where Adam carries two moments.
On this hardware Muon fits and full-FT Adam does not: a 2.9B full fine-tune under
Muon peaks at 19.8GB of a 24GB A30, where AdamW's two fp32 moments are 23.6GB
before anything else is allocated. That is the whole of the practical case, and
it is a statement about our hardware rather than about the optimizer — FORGE's
`optimizer_only_adamw_int8state` would bring those moments to ~5.8GB and let Adam
fit here too. We have not used it; anyone choosing between the two should know it
exists, because "Muon is what fits" stops being a reason if it does. See
Limitations for the narrower scope this argument had in an earlier draft.

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
Under this implementation a smaller adapter is a *larger* step on the factor,
without touching the learning rate. At r=4 the `ffn.key` B factor moves at fifty
times the configured rate.

**Measured, and the effect on the weight is much smaller than the coefficient —
say the second, not the first.** One Muon step at `lr = 0.002` on
`rwkv7-g1i-2.9b`, 192 pairs, reporting the update the pair actually induces on
the weight it adapts, `‖Δ(BA)‖_F / ‖W_base‖_F`:

| r | coefficient on `lora_B` | median induced `‖ΔW‖/‖W‖` |
|---|---|---|
| 128 | 4.472 | 0.002236 |
| 32 | 8.944 | 0.002464 |
| 8 | 17.889 | 0.002472 |

The direction holds — smaller `r` is a larger step — but the coefficient grows
4x across that range while the induced update grows about 10%. Newton-Schulz
normalises each factor's singular values to ~1, so most of the `r` dependence
cancels in the product. The honest form of this section's claim is therefore
about the **factor**, and anyone reasoning about damage should look at the
product instead. That is the same distinction §2.4's fix is built on, arriving
here from the measurement side.

A note on how this has to be measured at all: the natural quantity,
`‖Δw‖/‖w‖` per tensor, is **undefined for a LoRA pair at the first step** —
PEFT initialises `lora_B` to zeros, so the denominator is zero and the ratio
comes back around 1e11. A factor's step size has no meaning in the factor's own
units; only the induced `ΔW` does.

### 1.3 The reference already avoids this — by a naming convention that does not travel

**How we got here, because it changes what this section is.** Our own optimizer
selected Muon parameters by `name.endswith(".weight")` plus an `.att.`/`.ffn.`
test. On RWKV-7's own factors that agrees with the reference's exclusion, so when
we checked the two against each other on 2026-09-02 they matched and the check was
recorded as passing. Then PEFT attached a LoRA adapter, whose factors are named
`...lora_A.default.weight` — which **passes** a `.weight` filter. Factor pairs went
under Muon and no one decided that.

So this is not a section about someone else's oversight. It is a section about our
own, which then turned out to have a more general shape: a verification is only
valid over the set of parameters it was run against, and `get_peft_model` changed
that set two weeks after the check was filed as done.

We wrote an earlier draft of this section as a question: are RWKV-7's own low-rank
pairs excluded from Muon on purpose? Reading `modded-nanogpt-rwkv/train_rwkv7.py`
answers it. **On purpose, and completely:**

```python
optimizer3 = Muon([p for n, p in params if p.ndim == 2
                   and '_w1' not in n and '_w2' not in n], lr=args.muon_lr, momentum=0.95)
optimizer4 = torch.optim.Adam([p for n, p in params if
                               (p.ndim != 2 or '_w1' in n or '_w2' in n) and ...], ...)
```

and every low-rank factor in that model is named to match — `time_decay_w1/w2`,
`time_aaa_w1/w2`, `mv_w1/w2`, `gate_w1/w2`. All eight go to Adam. Muon never sees a
factor pair. The aspect rescale is in that file too —

```python
g = zeropower_backend(g, steps=group['backend_steps'])
g *= max(1, g.size(0) / g.size(1)) ** 0.5
```

— and it is correct there precisely *because* of the line above it. The
coefficient never meets a pair whose partner it would have to know about.

**The problem is that the exclusion is carried entirely by a string.** Two
renamings break it silently, and both are things people actually do:

| parameter name | caught by `'_w1' not in n and '_w2' not in n`? |
|---|---|
| `time_decay_w1` (this reference) | excluded — goes to Adam |
| `blocks.0.att.w1` (RWKV-LM naming) | **not caught — goes to Muon** |
| `blocks.0.att.a1` / `.v1` / `.g1` (RWKV-LM) | **not caught — goes to Muon** |
| `...lora_A.default.weight` (PEFT) | **not caught — goes to Muon** |

In the RWKV-LM naming the underscore is simply not there, so porting the
optimizer setup verbatim excludes **nothing** — all four factor pairs land under
Muon with coefficients up to 6.325 on the decay path (§1.3 table below). And PEFT's
adapters were never in scope for that filter at all, which is how an ordinary
`get_peft_model` call puts factor pairs under Muon without anyone deciding to.

On the G1i checkpoint, what those factors would receive:

| parameter | shape | coefficient |
|---|---|---|
| `att.w1` (decay) | (2560, 96) | **5.164** |
| `att.a1` (in-context LR) | (2560, 96) | **5.164** |
| `att.v1` (value residual) | (2560, 64) | **6.325** |
| `att.g1` (gate) | (2560, 320) | **2.828** |
| `att.w2`, `a2`, `v2`, `g2` | (rank, 2560) | 1.000 |

So this is not a bug report against the reference. The reference is right and knew
it. It is a report that **its correctness lives in a naming convention rather than
in the code's structure**, and that the convention does not survive the two most
common ways someone would reuse it.

### 1.4 Full fine-tuning is a milder case of the same thing

With no adapter, the only tensor receiving a coefficient above 1.0 is `ffn.key`
(10240, 2560) at **2.000**, while `ffn.value` (2560, 10240) receives 1.000 — two
halves of one FFN block, one shared learning rate, a 2x asymmetry between them.
Every `att.*` projection is square and receives 1.000.

So at full fine-tuning the pairing story has almost nothing to work with, and for
a long time we had no account of why full-FT collapsed anyway. §1.5 is that
account, and it is not about pairing at all.

### 1.5 How big the step actually is, in units of the weight

The coefficient arithmetic above is relative — it says one tensor moves 9x
another. It does not say whether *either* is a sane distance to move a converged
model. That absolute question has a one-line answer nobody needs a sweep for:
take one optimizer step and print, per tensor,

```python
rel = (w_after - w_before).norm() / w_before.norm()
```

On `rwkv7-g1d-0.4b`, full fine-tune, first step, no adapter:

| optimizer | lr | max `‖Δw‖/‖w‖` | median |
|---|---|---|---|
| **Muon** | **0.02** | **3.494** | **0.0406** |
| Muon | 5e-4 | 0.0875 | 0.00102 |
| Muon | 1e-4 | 0.0175 | 0.000204 |
| AdamW | 1e-5 | 0.00175 | 0.0000514 |

At `lr = 0.02` the worst tensor is replaced **three and a half times over in a
single step**, and a typical one moves 4%. There is no threshold argument to make
here and no pairing involved: the step is simply not a fine-tuning step.

The reason is structural rather than a tuning accident. Newton-Schulz
orthogonalises the momentum, so the update's singular values are ~1 *whatever the
gradient was*; the step norm is about `lr · sqrt(min(m, n))` and carries no
information about how converged the model is. That is a feature when pretraining
— it is most of why Muon works there — and it is precisely the wrong property
when the gradients have become small because the model is already good.

**Two consequences we would put in front of a maintainer.**

First, the learning rate can be *computed* rather than searched. Pick the relative
step you want — AdamW at a typical fine-tuning rate sits near 1e-3 on the median —
and divide. For us that gave 5e-4, and the measurement confirmed it: median
1.02e-03 at that rate.

Second, and more interesting: **one global learning rate cannot do it.** At the
rate that puts the median at 1e-3, the worst tensor is still moving 8.75%. The
spread across tensors is roughly two orders of magnitude, which is the same
observation §1.1 makes about factor pairs, generalised — the aspect coefficient is
a *shape*-derived guess at a quantity that can simply be measured. A per-tensor
budget on `‖Δw‖/‖w‖` is the version of §2.4's fix that needs no knowledge of
pairing at all.

And the practical outcome, same model and budget, 30 steps: Muon at 1e-4 reaches
a **lower** training loss than AdamW at 1e-5 (0.380 against 0.538) while both
improve held-out cross-entropy. Muon fine-tunes this architecture perfectly well.
Our own earlier reports that it does not were reports about `lr = 0.02`.

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

### 2.3 What `r` means here — and a claim of ours that does not survive

A correction first, because we made it ourselves and it is the kind that
propagates. We have written elsewhere that factor-wise Newton–Schulz "injects a
near-flat rank-`r` update by construction", citing an entropy-rank of 31.88 out
of 32. **That measurement was taken at initialisation on i.i.d. Gaussian
gradients, where it is a tautology** — orthogonalise Gaussian noise and of course
the spectrum is flat. On real gradients it does not hold.

Two things are true instead, and they are less dramatic.

**The update spans up to 2r, not r.** `ΔW = s·(B·δA + δB·A)` is a sum of two
rank-`r` terms, so an `r=32` adapter can write into as many as 64 directions, not
32. That is a property of the factorisation, not of the optimizer.

**Flatness is not what does the damage.** Our own probe's spectrum column settles
this: the arm that fixes retention (`muon_balanced`) has an adapter spectrum
indistinguishable from the broken one — entropy-rank 4.667 vs 4.674, σ_r/σ_1
0.0555 vs 0.0538 — while retention differs by **0.255**. Whatever separates them,
it is not the shape of the update's spectrum.

What remains worth putting next to `r` is the receiving end. The rank ceiling of a
WKV write is `min(n_recurrence_steps, head_size)` — one rank-1 write per step,
head_size 64 on this checkpoint — and the measured live-direction count per head,
at stride 1 over five prompts, runs **8.8 at L24 to 22.6 at L20, out of 64**. So
an `r=32` adapter can address up to 64 weight-space directions into a state that
is currently carrying 9–23.

Whether raising that count is desirable is a separate question we have partly
answered against ourselves: in a controlled toy, arms that deliberately raised the
live-direction count did reach a higher count — up to the structural ceiling,
15.7-16.0 against 6.6-7.4 for plain training — and scored **worse** held-out than
the plain arm (+0.6085 vs +0.7501). Breadth was buildable and bought nothing.

**But note exactly which claim that kills, because it is the weaker one.** Those
arms raised breadth with an explicit breadth term — they optimised the metric
directly, which is the Goodhart case our own pre-registered criteria name. So what
is refuted is *"force the direction count up and quality follows"*. What is
untouched is *"a model that genuinely needs several directions at once will use
them"*. Coercion and need are different claims and we only tested the first.

The experiment that separates them, which we have not run: a task with real
deferred ambiguity — several answer candidates that must coexist until a later
token disambiguates — and then ask whether the live-direction count rises **on its
own**, with no breadth term anywhere, and whether it correlates with getting the
answer right. If it does, breadth is a genuine capacity and rank is worth spending
on. If the count stays flat even where holding candidates is objectively required,
the model is solving such tasks some other way and the whole line closes.

We state this because the negative result above is easy to over-read, including by
us. "We raised it artificially and it did not help" is not "the state does not
benefit from carrying more at once".

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

**What it does on the real model**, measured the same way as §1.2 — one step at
`lr = 0.002`, r=32, 192 pairs on `rwkv7-g1i-2.9b`, median induced
`‖Δ(BA)‖_F / ‖W_base‖_F`:

| | median induced step |
|---|---|
| upstream shape coefficient | 0.002464 |
| equalised downward | **0.000264** |

A factor of **9.3**. The fix is not a rounding correction; it changes the size of
the update the adapter actually applies by nearly an order of magnitude. Whether
that lands on the *right* size is a separate question this does not answer — it
equalises, and §1.5 is the argument that the target itself should be set against
the weight rather than against the shape.

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

### 2.6 The same numbers read the other way: how wide the usable step is

Everything above asks what goes wrong when Muon's step is too big. Compared at a
matched step instead of a matched learning rate, the measurements say something we
did not set out to find: at a step where AdamW does damage, Muon does not.

**A real model, one run.** `rwkv7-g1d-0.4b`, full fine-tune, 30 steps, no adapter
(`optimizer_geometry_probe.py`). Steps are the first step's per-tensor `‖Δw‖/‖w‖`, both
the median over tensors and the largest; ΔCE is held-out cross-entropy after minus
before, so negative is better:

| optimizer | lr | median step | largest step | final train loss | ΔCE held-out |
|---|---|---|---|---|---|
| Muon | 1e-4 | 2.0e-4 | 1.75e-2 | 0.380 | −0.131 |
| AdamW | 1e-4 | 5.1e-4 | 1.75e-2 | 1.038 | **+0.991** |
| AdamW | 3e-5 | 1.5e-4 | 5.25e-3 | 0.851 | +0.256 |
| AdamW | 1e-5 | 5.1e-5 | 1.75e-3 | 0.539 | −0.219 |

*n = 1: the three "seeds" of each arm returned bit-identical numbers (the run had no
stochastic element), and Muon appears at one learning rate only.*

Read on the **median** step, which is the bulk of the tensors: AdamW already makes
held-out loss worse at a median step of 1.5e-4, which is *smaller* than the 2.0e-4
at which Muon improves it. The largest step is the wrong thing to match on: it
falls on `blocks.22.ffn.x_k`, a 1024-vector that both optimizers update the same
way (it sits in the AdamW group of the Muon hybrid), so it is equal at equal learning
rate by construction. Muon is not free to go further — at 0.02 it moves the worst
tensor by 349% (§1.5) — so what it has is a **wider usable window**, not an
unlimited one. It is not shown to be a better optimum: on this run AdamW at its safe
rate has the larger held-out gain (−0.219 against −0.131) while Muon reaches the
lower training loss (0.380 against 0.539); a single 30-step run gives no estimate of
the spread of held-out CE, so we leave that comparison open. What this table does not
contain is Muon at 1e-5 and 3e-5: whether Muon is at least as good as AdamW at every
step inside the shared window, or only better at the upper end, is not known.

**The toy, 30 seeds** (`attractor_depth_probe.py`, LoRA r=8, matched induced step
≈ 8e-4, old skill and a new skill, both measured by R²): factor-wise Muon reached a higher
new-skill R² than Adam in 30 of 30 seeds (0.9974 against 0.9930), kept more of the old
skill in 24 of 30 (+0.051 on average, sign test p = 0.0014), left the state less
concentrated on one direction in 27 of 30 (top-1 energy 0.81 against 0.90), and depended
less on the delta-rule erase channel — zeroing `a_gate` costs R² −0.87 under Muon and
−4.20 under Adam (medians). That is §0's mechanism result again, now with 30 seeds and
on LoRA. With full fine-tuning at a matched step (~1e-3) the two arms are
indistinguishable on every one of these measures, which is what the 0.4B table predicts
when both are inside their safe window.

**What we withdrew along the way.** An earlier six-seed reading said Muon also had
cleaner *tails* — fewer bases collapsing under repeated blank ticks. At 30 seeds it does
not: 10 of 30 against 9 of 30 with full fine-tuning, 13 of 30 against 10 of 30 with LoRA
(Fisher p = 1.0 and 0.60). Fine-tuning at all leaves 30-43% of bases collapsing under 32
blank ticks, against 3% for the untouched base, whichever optimizer did it.

**How this connects across scale.** The toy, the 0.4B run and the 2.9B arithmetic of §1
agree on one thing: what decides the outcome is the step measured in units of the
weight, and Muon's flat spectrum makes it usable at steps where a coordinate-wise rule
is not. They do not agree on magnitude, and we would not translate a number between
them: the toy's advantage shows only with LoRA, and the 0.4B comparison is full
fine-tuning. Nothing here is measured on LoRA at real scale. The one run that could
carry it is §4 question 6.

---

## 3. Where this sits relative to current Muon work

Checked 2026-09-16, because "improved Muon with various tricks" was mentioned to
us and we had not followed it up.

**The shape rescale is current, not legacy.** `KellerJordan/Muon`'s reference
implementation still applies

```python
update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
```

after Newton–Schulz, with Nesterov on by default and AdamW-style weight decay. So
§1 is about the rule as it stands, not about an old version someone has since
fixed.

**Upstream guidance has a hole exactly where factorised weights would go.** The
reference's own instruction is: *"Muon should only be used for hidden weight
layers. The input embedding, final output layer, and any internal gains or biases
should be optimized using a standard method such as AdamW."* Embeddings, output,
gains, biases — and nothing about a weight that is **one factor of a product**.
That omission is what BlinkDL patched locally by name (§1.3), and it is why the
patch is a convention rather than a rule.

**The speedrun has independently found that the tensor is not always the right
unit to orthogonalise over.** Record 80 of `modded-nanogpt`: *"In Muon
orthogonalize Q and K matrices in pairs of heads, instead of across the full 6
head matrix."* That is the same class of move as §2.4 — the parameter tensor as
stored is not necessarily the object whose spectrum you want to control. We think
the factor-pair case is the same discovery on a different axis, and we would
rather say that than present it as unprecedented.

**A smaller observation, offered as a reading rather than a claim.** Record 27 is
*"Transpose one of the MLP matrices + add Triton kernel for symmetric matmul"*,
and the stated reason is the kernel. But transposing one MLP matrix also changes
which shape coefficient it receives: stored the usual way, `W_in` (d_ff, d_model)
takes `sqrt(d_ff/d_model)` and `W_out` (d_model, d_ff) takes 1.0 — the same 2x
asymmetry inside one block that §1.4 reports for `ffn.key` vs `ffn.value`.
Transposed, both take the same coefficient. Whether any of the speedup came from
that rather than from the kernel is checkable and, as far as we can see, unchecked.

**NorMuon is adjacent but does not cover this.** arXiv 2510.05491 adds row-wise
(per-neuron) normalisation after orthogonalisation plus per-neuron second-moment
statistics, because Muon "produces highly non-uniform neuron norms, causing
certain neurons to dominate". That is a within-tensor non-uniformity. The
factor-pair problem is a **between-tensor** one: no amount of normalising rows of
`lora_B` tells the optimizer that `lora_A` is the other half of the same update.
The two are orthogonal, and a system using both would still want §2.4.

---

## 4. What we would like to know

0. **Which "improved Muon" did you mean?** We followed the pointer as far as
   `KellerJordan/Muon` (shape rescale still current), the `modded-nanogpt` record
   list, and NorMuon — §3. If the tricks you had in mind are elsewhere, several
   of the questions below may already be answered there and we would rather read
   than ask.
1. **Would you consider making the factor-pair exclusion structural rather than
   by-name?** §1.3 shows it is deliberate and complete in `train_rwkv7.py`, and
   that it is carried entirely by the `_w1`/`_w2` convention — which catches
   nothing under RWKV-LM's `att.w1`/`att.a1` naming and nothing under PEFT's
   adapters. A predicate over shape and pairing, or simply a comment marking the
   filter as load-bearing, would make the intent survive a rename.
2. **Has anyone run Muon with LoRA on a pretrained RWKV-7, at what rank and what
   learning rate?** §1.2 predicts that lower rank is worse under the current
   implementation, which is the opposite of the usual expectation, and that is a
   cheap thing to falsify with two runs. A second, separable question is
   §2.3's: under factor-wise Muon, `r` stops being a capacity ceiling and becomes
   a forced flat-rank-`r` update every step. Is that the intended reading of rank
   when Muon is used with a low-rank adapter, or an unexamined side effect?
3. **For pretraining: is there a reason to prefer Muon beyond the ones we can no
   longer defend?** Limitations explains why we cannot currently argue "because it builds
   the state's breadth" — our own probe found breadth buildable and worthless, and
   the attribution to Muon-pretraining was never tested against another lineage.
   Your experience says Muon works for pretraining RWKV-7; we would like to know
   what it is actually buying there, because we would otherwise be repeating a
   reason we have retired.
4. **For pretraining, does the coefficient's behaviour on `ffn.key` vs
   `ffn.value` (§1.4) match what you would expect?** A 2x asymmetry between the
   two halves of one FFN block under one learning rate is below our measured
   damage threshold, but we would rather hear it is intended than assume it.
5. **What learning rate do you use when Muon touches an already-trained
   checkpoint, and does anything normalise the step against the weight?** This is
   the question we would most like answered, because §1.5 says a shape-derived
   coefficient cannot by itself keep `‖Δw‖/‖w‖` in range: at the rate that puts
   the median at 1e-3 the worst tensor is still at 8.75%, a spread of two orders
   of magnitude. Some Muon implementations match the update's RMS to the
   parameter's; if that or an equivalent trust region is standard practice and we
   simply missed it, most of §1.5 collapses into "we were using it wrong", which
   would be a perfectly good outcome and worth saying plainly.
6. **Does Muon's wider usable step window show up for you with LoRA at real scale?**
   §2.6 finds it on a toy with LoRA and on a 0.4B full fine-tune, but nothing that
   combines the two. The run that would answer it is cheap on a real model: LoRA r=32,
   Muon and AdamW each at three learning rates chosen to give the same set of first
   steps, 150 steps, held-out CE and old-skill retention per arm. If AdamW holds at
   the steps where Muon holds, §2.6 is a toy artefact and we would rather know.

---

## 5. A separate question, about `head_size` — curiosity, not a request

Nothing here depends on anything above and none of it asks for work. These are
questions about pretraining design, and we would rather ask than assume.

The paper describes the state update as "equivalent to a single step of
stochastic gradient descent, training the state `S_t` at test time to output the
desired values `v_t` for the keys `k_t` as inputs". Taken literally, each head's
state is a small regression refitted as the sequence runs, and `head_size` is not
a projection width — it is **the rank ceiling of that head's memory**. In
attention the same number is only a width. Same constant, a different job.

And it is held at 64 across the line: 0.4B has 16 heads, 1.5B 32, 2.9B 40. Width
always buys **more 64-slot memories**, never deeper ones. `train.py` marks the
knob as live — `--head_size_a`, "can try larger values for larger models" — so it
is a choice, not an inevitability.

1. **Was 64 chosen for the memory role, or carried over from attention?** If the
   number sets how many key directions a head can hold at once, the reasoning
   behind it is different from the reasoning behind an attention head dim, and we
   cannot tell from outside which one applied.
2. **Is "more heads, same depth" the intended scaling?** Two 64-slot memories and
   one 128-slot memory have the same parameter cost and very different behaviour
   — one holds more things, the other holds more about each thing. Was that
   trade-off examined, or is 64 simply what has not needed changing?
3. **Does the state actually fill?** On a trained checkpoint the occupied
   fraction is easy to read and varies a lot with depth. What we cannot know is
   whether anything in pretraining — the data, the objective, the decay
   initialisation — pushes toward using distinct keys rather than overwriting the
   same ones, or whether occupancy is simply whatever falls out.
4. **Is there an intended way to use more of it at inference?** One readout is a
   single linear map applied to a state that may be carrying many directions at
   once. We do not know whether that asymmetry is by design, a known limitation,
   or a thing nobody has had a reason to look at.

---

## Limitations

**What is not done yet (state of 2026-10-06).** Listed first because it is what a reader
should weigh the rest against.

1. **Nothing is measured on LoRA at real scale.** §2 is a toy; §2.6's real-model table is full
   fine-tuning on a 0.4B. The 2.9B run that would test the pair fix (on against off, 150
   steps, where the collapse signature appears at step 34) has not been run.
2. **The 0.4B table is one run.** Muon appears at one learning rate (1e-4) against three AdamW
   rates, held-out ΔCE has no spread estimate, and the three "seeds" are bit-identical. Muon at
   1e-5 and 3e-5, with shuffled data order, is written and waiting for a GPU session (~30 min).
3. **AdamW was tried at three learning rates and nothing else** — no warmup, decay or tuned
   weight decay. "AdamW harms at that step" means at those rates and that schedule.
4. **Loss is not generation.** The earlier full fine-tuning collapse passed every stability
   check and failed real generation evaluation; at 1e-4 we report training loss and held-out
   cross-entropy only, not a generation eval of the fine-tuned model.
5. **Whether the toy's concentration gap is LoRA capacity or Adam's use of its step** is being
   settled by a rank sweep with the step controlled (running); until it finishes §2.6 says only
   that the gap exists, not why.
6. **Nothing here concerns pretraining**, which is where Muon's case for RWKV-7 is made.

**The toy is a toy.** §1 is checkpoint arithmetic and holds regardless. §2 is a
mechanism claim measured on a small recurrent controller with a WKV-shaped state,
not on a 2.9B model, and should be treated as one until someone reproduces the
threshold at real scale. §2.6's real-model table is a single 30-step run on one
0.4B model (its three seeds are bit-identical), and nothing in the document is
measured on LoRA at real scale.

**The full fine-tuning collapse turned out to be a learning rate, and it was
ours.** This paragraph previously said the collapse was a second, unexplained
problem. It is explained, and the explanation is embarrassing rather than deep:
at `lr = 0.02` a Muon step moves the worst tensor by **349% of its own norm**,
and a typical tensor by 4% (§1.5). Every "Muon breaks fine-tuning" run this
project produced was Muon at a pretraining learning rate on a converged
checkpoint. At `lr = 1e-4` the same setup trains normally and reaches a *lower*
training loss than AdamW at its own tuned rate over the same budget, while
improving held-out cross-entropy. So we cannot offer "Muon is wrong for
fine-tuning RWKV-7" as a finding, and this document no longer claims it. What
survives is §1 — which is checkpoint arithmetic and never depended on any of our
training runs — and §2's threshold, which §1.5 now reaches from the other side.

**This is not an upstream bug report.** The optimizer class our runs used is our
own file. The genuine upstream issue is narrower and unrelated: `light_rwkv.py`
references `args.optimizer=='muon'` but no such class was ever vendored, so that
branch references an undefined name and crashes.

**The VRAM argument had a narrower scope than we first wrote.** Against naive fp32
Adam, 23.2GB of optimizer state against one 5.8GB momentum buffer is decisive.
Against the `Int8AdamW`-with-offload baseline we were actually replacing, two int8
states come to the same 5.8GB in host RAM; Muon's first real GPU run OOMed on
backward because its buffer was resident and Adam's was not, and offload was
written for it the same day.

**We cannot argue for Muon in pretraining on the grounds we used to.** The claim
we carried was a build/bake split — a geometry-shaping update rule *builds* the
state's spectral structure during pretraining while a coordinate-wise adaptive one
*bakes* it. Two problems. It has never been tested against a non-Muon-pretrained
lineage of the same architecture, so the attribution is assumed. And our own
controlled probe found breadth buildable and worthless: arms with an explicit
breadth term reached the structural ceiling on live directions and scored worse
held-out than plain training, while the one thing breadth was credited with
rescuing turned out to be reproduced by a rank-blind term on state energy. If Muon
is right for pretraining RWKV-7, and BlinkDL's experience says it is, we cannot
currently say it is right *because* it builds breadth.

**Three different things are called "rank" here, and we have used the word
loosely.** An adapter's `r`; the rank of the induced `ΔW`, which is up to 2r; and
the number of live directions the state carries, 8.8 to 22.6 per head out of 64
depending on depth. "Holding several answer directions at once" is a property of
the third — the state and the internal steps that traverse it — not of the
optimizer and not of an adapter's rank.

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
