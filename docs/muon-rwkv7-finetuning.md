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
also show that the reference implementation deliberately keeps every low-rank
factor away from Muon — and that it does so through a naming convention which
catches nothing once the names change to RWKV-LM's or the factors come from a PEFT
adapter. That last part is the finding we would most want a maintainer to see.

---

## 0. Why we are running Muon at all

Worth stating, because it changes how the rest should be read, and because our own
history here is less tidy than the sections below might suggest.

**What made Muon interesting was a mechanism result, not memory.** On 2026-08-23,
testing something else, we found that a delta-rule erase-rewrite channel which
Adam's solutions relied on heavily stops mattering under Muon at matched accuracy:
forcing `a_gate = 0` destroys Adam's solution on every seed (−1.376 ± 1.326, worst
−3.30) and barely touches Muon's (−0.029 ± 0.032). A 10-seed rerun held the shape
(−2.619 ± 2.223 vs −0.295 ± 0.563, distributions not overlapping) at matched R².
Same task, same architecture, same accuracy, reliably different internal solution.
That is what "the optimizer selects the mechanism" means here, and it is the
observation that put Muon on our roadmap.

**The adoption decision, ten days later, added two reasons that hold regardless of
that result.** Muon carries a momentum buffer and no second-moment state, which
directly answers a fixed-cost VRAM wall — weights 5.90GB + gradients 5.90GB +
Adam's own ~5.8GB of int8 second-moment buffers on a 16GB card. And our Phase
1.5/2 is supervised distillation rather than RL, so an unfamiliar optimizer's
quirks are far cheaper to diagnose there than inside policy-gradient noise. Those
were chosen as backstops precisely because they do not depend on a toy's R².

**The memory argument holds against plain Adam and is a wash against what we were
actually replacing.** On a 2.9B model the arithmetic is: naive fp32 Adam carries
23.2GB of optimizer state (two moments); Muon carries one momentum buffer, 5.8GB
at bf16. That difference is real and it is the difference between fitting and not
fitting on a small card.

But our baseline was not naive Adam — it was `Int8AdamW` with CPU offload, whose
two int8 states come to the same 5.8GB and sit in host RAM rather than VRAM. Muon's
first real GPU run OOMed on backward every time, because its momentum buffer was
resident while Adam's was not; offload was written for it the same day, after
which the two are roughly even on host RAM and Muon is ahead only by carrying one
state instead of two. So: a strong argument in general, a narrow one against the
specific thing we swapped out.

**And a third reason was formalised afterwards** — the build/bake split in §0's
last part — with pre-registered falsification criteria, one of which has since
fired against it. That one we present as a claim under test, not as a motivation.

**What we are actually trying to build**, since it explains which properties we
care about. The training track is meant to give the model more internal steps
before it answers — time and room to write instructions to itself in its own
recurrent state and read them back — so that answering draws on what the model
already knows rather than on a longer visible chain of text. In that picture the
quantity that matters is how many independent directions the state can hold and
traverse. Our measurements put that at **8.8 to 22.6 live directions per head out
of 64**, depending on depth, on a 2.9B checkpoint.

One correction to our own vocabulary, because we have used it loosely and it
matters for §2.3: the "hold several answer directions at once" property belongs to
the **state** and to the number of internal steps that traverse it, not to the
optimizer and not to an adapter's rank. Those are different axes that happen to
share the word "rank".

**Which brings us to pretraining, where we genuinely do not know.** The claim we
have been carrying is a build/bake split: a geometry-shaping update rule
(orthogonalised, spectral-norm step) **builds** the state's spectral structure
during pretraining, while a coordinate-wise adaptive rule **bakes** it —
specialising inside a structure it leaves intact. G1i's pretraining used Muon, so
on that reading the breadth we measure is Muon-built and Adam-preserved.

We flag two problems with using that as an argument for Muon in pretraining.
First, it has never been tested against a non-Muon-pretrained lineage of the same
architecture, so the attribution is assumed rather than measured. Second, and
worse for the story, our own controlled probe found that **breadth is buildable
and buys nothing**: arms with an explicit breadth term reached the structural
ceiling on live directions and scored worse held-out than plain training, and the
one thing breadth was credited with rescuing turned out to be reproduced by a
rank-blind term on state energy. So if Muon is right for pretraining RWKV-7 — and
BlinkDL's experience says it is — we cannot currently claim it is right *because*
it builds breadth. That reason is the one we would have given a month ago, and it
is the one our own data declines to support.

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

### 1.3 The reference already avoids this — by a naming convention that does not travel

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

## 3.5 Where this sits relative to current Muon work

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
   list, and NorMuon — §3.5. If the tricks you had in mind are elsewhere, several
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
   longer defend?** §0 explains why we cannot currently argue "because it builds
   the state's breadth" — our own probe found breadth buildable and worthless, and
   the attribution to Muon-pretraining was never tested against another lineage.
   Your experience says Muon works for pretraining RWKV-7; we would like to know
   what it is actually buying there, because we would otherwise be repeating a
   reason we have retired.
4. **For pretraining, does the coefficient's behaviour on `ffn.key` vs
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
