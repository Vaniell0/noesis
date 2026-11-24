# Hypotheses

What noesis claims, in a form a measurement can refute, and what the
measurements said. One file per claim (`H<N>.md`, `HZLK.md`). The
record of a claim (`status`, `prior` / `posterior`, `contradicts_if`,
`evidence`) is data and lives beside the files, in
`hypotheses/records/<ID>.yml`; a file not yet moved carries it in a
comment at its top. The `.md` holds prose. The status table at the
bottom of this page is generated from the records. The pre-split prose
version lives at git tag `hypotheses-v1-prose-era`.

The repository holds two things under one name, and they are judged
differently:

- a **research programme** on how a recurrent-state model computes, and on
  how it can be trained to compute more with what it already holds. Its
  claims are hypotheses: each has an estimator and a refuter (sections 1-6);
- a **runtime** (collectors, zones, composer, supervisor) built around one
  such model. The numbered items that belong to it are design wagers and
  operating decisions (section 7).

A claim belongs to the first group only when it can be written as: the
objects, a statement about them, the number that estimates it, and the
outcome that refutes it.

## How claims are judged

The programme is empirical. Most of what it knows came from trying an
approach, measuring whether it works, and trying the next; that path is how
the formal statements in this directory were found, and it stays open. Two
kinds of work are kept apart by what they may change in the record, not by
whether they are allowed:

- *Exploration.* Any approach, any reframing, any number of variants. Its
  results are recorded as measured, with file and command, and say what was
  tried. Exploration may propose a claim, an estimator or a refuter. It does
  not move a status.
- *Test.* A status moves only on a run whose estimator and refuter were
  written in the claim's file before the run. If the refuter fires, the file
  says so; the claim may then be restated, and the restated claim waits for
  its own test.

Corrections are made when found: a misread column, a wrong formula, a number
from one seed. The old reading stays in the file's evidence log, dated.
Reading an old result anew under a proved statement is a correction, not a
rescue.

**What counts.** A proof from the setup (section 3). A number with its file
and the command that produced it, on held-out items, paired by item when two
conditions are compared, over more than one seed when training is involved.
A refutation.

**What the record taught us to check before a number moves a status.**

- The registered estimator, not a substitute that reads better (H8, H9, H20).
- A control that a trivial reader would pass (bag of words beat the state
  head in H22).
- More than one seed (the parity toy in H25).
- What the set-up forces by construction (the open-loop lemma in H25).
- Which of weights, state and interface the result moved.

Benchmarks are references, not verdicts. Runtime properties (memory, CPU,
quality over days of operation) belong to the runtime wagers of section 7.

## Form of a file and its record

The rules of a file in this directory. The scripts named at the end
check them; a green run is what "kept" means here.

1. **The record is data.** It lives in `hypotheses/records/<ID>.yml`:
   `status`, `prior`, `depends_on`, `supports`, `contradicts_if`,
   `evidence` (dated entries with `call`, `strength`, `file`, `note`).
   A file not yet moved carries the same block in a comment at its top;
   the scripts read either. The `.md` holds prose and opens with one
   line naming its record.
2. **The prose of a claim has one order:** the statement; the objects;
   the estimator; the refuter; what the status rests on (dated, one
   link per line); then the sections it needs. No line over 1000
   characters. Tables only for enumerable facts.
3. **A link to data is one path in backticks, resolved from the
   repository root:** a file, the folder that holds the run, or a glob
   over a family of runs (`*`, `{a,b}`). A path outside the repository
   starts with a tilde or a slash and is checked on this machine. The
   data is not copied into this directory: a stamped result carries its
   own provenance (`_meta`: date, hypothesis, model, script, summary).
4. **A dead link is not deleted.** It is replaced by the place where
   the data now lies, or marked missing with the date: in a record a
   `missing:` date beside the link, in prose the path without backticks
   and the date in words. The old reading stays, dated.
5. **A claim of the research tree** names an estimator and a refuter
   in its prose and carries a `contradicts_if` in its record. A retired
   claim (`RETRACTED`, `WITHDRAWN`, `SUPERSEDED`, `RETIRED`) or a wager
   of the runtime (`WAGER`, section 7) carries a one-line record and
   needs no refuter.
6. **A correction is an entry of the record** (`call: correction`),
   dated, that restates the reading of an entry above it. It moves no
   odds. A claim restated in form only gets one; its status does not
   move on it.

7. **A status is one word of one list:** `PROPOSED` (written, not run),
`BLOCKED` (estimator and refuter written, waiting on a condition outside
the claim), `REGULAR` (every run so far agrees), `PARTIAL` (measured in part),
`CONTESTED` (the evidence runs against it), `SUPPORTED`, `REFUTED`;
the closed ones `RETRACTED`, `WITHDRAWN`, `SUPERSEDED`, `RETIRED`,
`WAGER`. A free phrase is a finding, not a status: it goes in
`finding`, so the set can be counted. A claim measured in any part
names its `regime`. Every status has a way out that a check can see:
a run moves `PROPOSED`, `PARTIAL`, `SUPPORTED`, `CONTESTED`, a fact
outside the claim moves `BLOCKED`, the closed ones are final. A decision
about work is not a status: it is the field `priority` (level, date,
reason), so a claim left aside stays counted by what was measured.
The scripts name the words they know, and a word they do not name is a
problem, never silence: what is not checked must not read as agreed.
8. **The claims of a programme are a graph, not a list of numbers.** The
groups are the sections of its file; `after` in the record names the
claims whose result a claim builds on (its edges). A number is a key:
it is never reused and never renumbered, whatever the order of the
file. The table "at a glance" is generated with the count, the groups
and the edges.

The calls that carry odds are `supports` and `contradicts`, weak and
strong; `methodology`, `neutral` and `correction` carry none. An unrun
claim carries none: a `PROPOSED` claim has no odds until its estimator
and refuter are run.


**What the scripts check.**

- `python experiments/regenerate_hyp_index.py --check` resolves every
  link under `hypotheses/`, prints how many are alive, and names every
  dead one with the place the data lies when the tree holds it under
  one other path. It also names the lines over 1000 characters. Exit 1
  on any.
- `python experiments/bayes_lite.py` prints each record with its
  posterior and its lint: a reference to an unknown id, a claim without
  a refuter or an estimator, an evidence entry the scorer does not
  know, a status that disagrees with its posterior, a record kept in
  two places.
- `python experiments/backlog.py` prints how many stamped results a
  link names and lists the rest by hypothesis and by the folder that
  holds the run.
- `python experiments/regenerate_hyp_index.py` rewrites the status
  table below and the claims table of `HZLK.md` from the records. Run
  it after an edit to a file under `hypotheses/`.

## 1. The question

A trained model holds more than one forward pass brings to its output. How
much of what it holds can be brought out, at what cost in compute, and by
which organisation of the computation: where it runs (the recurrent state or
emitted tokens), for how long, and under what training?

In symbols. For a question `q` with answer `y*`, let `S_0` be the state after
reading `q`. The information `I(S_0; Y)` is fixed by the weights and the
input. One pass delivers only the part a bounded reader can extract: the
usable information `I_V(S_0 -> Y)`, where the reader `V` is the model's own
remaining layers and head. The programme studies the gap `I - I_V` and what
closes it per unit of cost (ticks, tokens, parameters).

The working bet is that organisation is cheaper than size: a small model
that reads what it holds, takes the time a task needs and no more, and takes
its facts from context, does the work of a larger model that stores more.

## 2. Objects and notation

Shared by every file in this directory. Code names for the same symbols:
`docs/rwkv7-mechanics.md`.

**Model.** RWKV-7 with weights `θ`. Each head of each layer keeps a state
`S ∈ R^{n×n}`, `n = 64`. One step, in the kernel's layout (value × key):

    S_t = S_{t-1} A_t + v_t k̃_tᵀ        A_t = diag(d_t) − κ̂_t (α_t ⊙ κ̂_t)ᵀ
    o_t = S_t r_t

- `d_t = exp(−exp(w_t)) ∈ [0.5452, 1)`: retention per channel;
- `κ̂_t`: the erase key, unit norm; `α_t ∈ (0, 1)`: the erase gate;
- `k̃_t`: the write key; `v_t`: the value; `r_t`: the receptance (read key).

A step is a decay, a rank-1 erase and a rank-1 write. The controls
`(r, w, k, v, α)_t` are functions of the layer's input at step `t`. The toy
cell (`experiments/A0_state_probe/micro_wkv.py`) stores the transpose
(key × value); every statement here holds in both layouts.

**Three things kept apart.** A result says which of them it moved.

- *Weights* `θ`: knowledge and skills. Fixed at inference.
- *State* `S`: what of the weights is selected for this context, the policy
  of reading them. Set by the input; a trained initial state changes form
  and task behaviour and adds no facts.
- *Interface*: the input embedding and the head. Which vector counts as
  which word; the format and the language of the output.

**Time.**

- *token step*: the input is the embedding of a token;
- *tick*: the input is the model's own last-layer vector, `u_t = h_{t−1}`;
  no token is emitted;
- `M`: the number of ticks before the answer. `K`: the number of emitted
  reasoning tokens before the answer. Feeding the question again is not an
  axis.

**Quantities.**

- `p_M(y | q)`: the answer distribution after `q` and `M` ticks; `acc(M)`:
  greedy accuracy on an item set.
- *Gain* `G_M(q) = log p_M(y* | q) − log p_0(y* | q)`, on one checkpoint.
  It estimates the change in usable information at one item.
- *Fact classes* at `M = 0`: **read** (the greedy answer is right);
  **stored, not read** (the greedy answer is wrong, the gold ranks high
  under teacher forcing or appears in samples); **lost** (neither).
  Estimators: `experiments/rl/m0_pass_probe.py --m0-rank`, `--m0-samples`.
- *Channel spread* at tick `t` over a prompt set: effective rank and PC1
  share of the centred fed latents, centred step size `step_c`, settle tick
  (`experiments/rl/latent_geometry_probe.py`).
- *Capacity of a head*: live directions (singular directions that carry the
  state) and separable facts (how many stored bindings a linear probe tells
  apart).
- *Efficiency* `= ΔI_V / (M · Σ‖ΔS‖)`: usable information gained per tick
  per unit of state motion (`HZLK.md` §2).

**Working terms.** The project's own words for the objects above, and where each is defined.

| term | object | where |
|---|---|---|
| second pretraining | the stage chain S0-S5: organise what the weights hold so it is readable and less interfered, with the loop learned from early on | `HZLK.md` §6 |
| tick, latent channel | one pass fed with the model's own last-layer vector; the vector itself | §2 above |
| КПД (efficiency) | usable information gained per tick per unit of state motion, `R_density` | `HZLK.md` §2 |
| bad (erased) sectors | facts in the class "stored, not read": held, missed by the one-pass read | §2 above; `H7.md`, `H19.md` |
| interference | crosstalk between stored items through overlapping keys | `H12.md` §1 |
| the neck of the heads | a layer's capacity: about 7 linearly readable bound values on G1d 0.4B (L11-13), falling from L16 to the head | `H25.md` §5 |
| attractor, collapse | the fixed point a settling loop reaches; one point for all prompts is the collapse | `H25.md` §4 |
| refusal to converge, open channel | ticks that keep changing the controls, so the loop does not settle at tick 1 | §3 (regular); `H10.md` §2 |
| pass-through | a trained loop whose checkpoint at `M = 0` solves the same items | §3 (regular) |
| self-regulation | the model chooses its own number of ticks | `HZLK.md` §6, S3 |
| MoE in time | each tick applies its own operator, computed from the fed latent, from the same weights | `docs/state-and-reasoning.md` §3 |
| repair, not refilling | a stage that unlocks held facts and adds none | `H7.md` |
| constitution | a trained initial state: form and dispositions, no facts | `H7.md` §3 |
| ROSA | the QKV suffix-automaton branch that keeps the exact past beside the settling state | `HZLK.md` §6, S5; `H25.md` §7 |

## 3. Proved, regular, hypothesised

**Proved.** These follow from the setup. A run cannot refute them; it can
only fall outside their premises.

- *Data processing.* With the input fixed, the state after any number of
  ticks is a function of `S_0`, so `I(S_t; Y) ≤ I(S_0; Y)`. A loop adds
  usable information, never information. `G_M` is what it can raise.
- *Input bound.* A fact absent from the input and the weights is out of
  reach for any `M`. A reward with `|R|` outcomes carries at most
  `log2 |R|` bits per episode, so reinforcement reweights what the model
  holds; new knowledge comes only from a teacher that has it.
- *Tick lemma.* With the controls frozen, a head's tick is affine,
  `S' = S A + v k̃ᵀ`. `A` is similar to the symmetric matrix
  `diag(d) − (√α ⊙ κ̂)(√α ⊙ κ̂)ᵀ`, a rank-1 negative update of `diag(d)`.
  Hence its spectrum is real and interlaces the retentions; the spectral
  radius lies between the two largest retentions, so it is below 1 and the
  erase gate cannot move it past them; `λ_min > d_min − max α > −1`. The
  iteration converges at rate `ρ(A)^t` to the fixed point
  `S* = v k̃ᵀ (I − A)⁻¹`, which has rank 1. Settling and forgetting are one
  number: what the prompt left in the state decays at the same rate.
  Premises: exact for layer 0 under a constant feed, asymptotic for deeper
  layers, silent about the closed loop, where the fed latent and therefore
  the controls change from tick to tick. Proof: `docs/state-and-reasoning.md`
  §3.1. Numerical sample: `micro_wkv.py::jacobian_spectral_sampling` (2000
  draws, no complex eigenvalue, none above 1).
- *Open-loop lemma.* If the controls are functions of the step's input
  and never of the state, then `S_T = Σ_t v_t k̃_tᵀ A_{t+1} ⋯ A_T`: inputs
  from two times meet only through decay and erase (`A`) and the final
  read. With the last read and decay fixed and the erase gate at 0 the
  output is additive, `f(a) + g(b)`; a step with blank input only
  reparametrises, so silent steps cannot widen what an open-loop cell
  computes. The real model is closed: a layer's controls come from the
  layer below, which read its state, and a tick feeds the last layer back.
  Proof: `docs/state-and-reasoning.md` §3.2; checks: `micro_wkv.py --verify`
  [4], [5].

**Regular.** Every run so far agrees; one clean counterexample refutes.

- *Pass-through.* An output-level loss is lowered more cheaply by repairing
  the one-pass read than by computing in the ticks, whenever the needed
  computation fits one pass plus the answer's own tokens. Ticks are needed
  only for short answers whose computation is long.
- *Open channel.* If ticks carry intermediate results they differ across
  prompts and across ticks. Every training breakage on record narrowed that
  spread; every success kept it.
- *Non-word vector.* The loop needs an interface that lets a fed vector be
  something other than a word. A head tied to the input embeddings reads
  any fed vector as the next token.

`HZLK.md` §1b states these five as T1-T5 and turns them into a gate of five
questions that a run must pass before it is started.

**Hypothesised.** Section 4.

## 4. The claims

Each entry: the statement, its estimator, its refuter, and what its status
rests on today. Statuses are in the table at the bottom; this section says
what stands behind them.

### The programme

**HZLK. A trained, self-regulated latent loop.**
After training, `M ≥ 2` ticks beat the same model at `M = 0` and at `M = 1`
on held-out items of at least two task families, for fewer generated tokens
than the model's own chain of thought, and the model chooses `M` itself.
*Estimator:* `acc(M)` paired by item; tokens at equal accuracy.
*Refuter:* seven conditions in the frontmatter; the first is a paired gain
below 5 points or McNemar `p ≥ 0.05`.
*Rests on:* nothing yet. Every trained checkpoint so far is a pass-through:
`M = 4` against `M = 0` gives 44 / 46 of 304 and 75 / 81 of 289 on G1d 0.4B,
197 / 204 of 289 on Qwen3.5 0.8B. All of those tasks fit one pass. The test
on a task that does not fit one pass has not been run.

### The state computes

**H8. The state carries computation, not only context.**
*Registered estimator:* effect size `d ≥ 1.0` on at least two of three state
statistics (delta norm, curvature, stable rank) between reasoning and
control text.
*Rests on:* a different measurement, the slope of output KL against noise
injected into the state (1.56-2.10 on G1d 0.4B, 1.58-1.67 on G1h 2.9B, 1.76
on G1i 2.9B) and argmax flips. The causal dependence is measured; the
registered test is not what supports the status. Capacity side: linear
held-out IPC 13.2 of 16 on G1d and 1.48 of 16 on G1i; 13-16 live directions
of 64 per head on G1i.

**H9. Reasoning-oriented pretraining raises the share of computation routed
through the state.**
*Registered refuter:* G1h and World indistinguishable on the three H8
statistics.
*Rests on:* the same substitute. Noise slope G1h 1.67 / 1.58 against World
1.19 / 1.13; output KL at noise 0.1 is 0.23 against 0.0055. World holds the
identity of the prompt in its state more firmly (cross-prompt KL 9.64
against 7.92) and computes less through it. World 0.4B and G1d 0.4B share
trained weights, so the difference is readable matrix by matrix.

**H25. The state is a substrate a model can be trained to compute on.**
Four statements: register, delegation, attractor, capacity. See `H25.md`.
*Rests on:* register shown on a toy cell and partly on one real checkpoint;
delegation fixed by algebra on the toy (open-loop lemma) and not
discriminated on the real model (one readable layer of three, graded);
attractor is the tick lemma plus its measurement on four real models;
capacity on G1d 0.4B: a linear probe of one layer's state reads up to about 7
bound values (sum of held-out R2; L12 holds most), not the 1-2 per head an
erratic solver reported before 2026-11-12.

**H12. Working memory is bounded by interference between keys, not by
the size of the state.**
`RETIRED` on the owner's word of 2026-11-24, a decision: the bound was not one
of room, the model stopped telling apart what was written. The question of how
much a head holds is carried by H25 and ZLK.12. See `H12.md`.

### Time buys computation

**H10. The frontier `acc(M)`.**
After training for it, accuracy rises with `M` on task families whose
computation is longer than one pass, and nowhere else. See `H10.md`.
*Refuter:* `M = 0` of the same checkpoint is not worse (paired by item).
*Rests on:* nothing yet. The one toy offered for it, parity with silent
steps, runs on an open-loop cell, where silent steps cannot widen the
function class; its gain (0.882 -> 1.000) is one seed of an optimisation
spread (`H25.md` §2). On every trained real checkpoint the refuter
currently holds.

### Training decides the mechanism

**H26. The optimizer selects which mechanism a solution uses.**
Weak: at matched accuracy, the optimizer changes the form of the solution.
Strong: the right step metric for a recurrent-state model is induced by the
state. See `H26.md`.
*Rests on:* weak, one toy task over 10 seeds, where both optimizers'
solutions need the erase path entirely and differ in size: without it,
Adam's leave an additive output of 0.21-7.46 times the target variance,
Muon's at most 0.08 (`H25.md` §3), and raising Adam's step moves it as
far as switching optimizer does (the registered confound fired). Its own
three-task refuter not run. Strong: the premise survives (no fixed weight norm predicts state
displacement, `|corr|` 0.648 against a threshold of 0.9); the mechanisms
proposed for it are refuted by their own criteria, and at real scale the
fine-tuning collapse is explained by step size.

### The state as context

**H17. A persistent state absorbs the history once and replaces
re-injecting it.**
*Refuter:* answers that need the history are worse from the state than from
the full history, also after loop training.
*Rests on:* `RETIRED` on the owner's word of 2026-11-24, a decision: not a
claim about the model but a property of a community runtime, whose state takes
the current prompt only and keeps no history. Never tested; its estimator was
written before ticks and M. The question of history in the state is carried by
H10, H18 and H20.

**H18. States can be forked and merged.**
*Rests on:* a fork is exact; a merge kept both contexts on 1 of 3 pairs.

### Where the answer comes from

How the model handles information: whether it can tell what an answer is
taken from (context or weights), keep two readings open, refuse a broken
question, and mark a claim without a source. Each is a readout of the state,
used as an instrument and as the measurable side of honest not-knowing.
None of the four was measured in a way that separates the state from the
surface: every set so far had a lexical route, and a head on pooled state
statistics may read too little. Proposed 2026-11-24, not run (it waits for
the model to stand in RSC): the same tokens on another state, the state
prefilled from one item's context or left empty and the question fed byte
for byte alike, read on the checkpoint trained with ticks at `M = 0` and
`M = 4`, head and a fresh reader side by side. Each file carries its own
design under that date.

**H19. The source of an answer, context or weights, is recoverable.**
*Estimator:* the same item with its source passage, without it, and with an
unrelated passage, paired by item; at `M = 0` the fact class of the answer
(read, stored but not read, lost). An answer that survives removal of the
passage came from the weights; one that needs the passage and is not faked
by an unrelated one came from the context.
The claim: the source is readable from the state before the answer, better
than a bag-of-words control. See `H19.md`.
*Rests on:* nothing yet; never run. Written for a fine-tune that was never
made; the estimator above needs no fine-tune.

**H7. Training a skill moves facts between classes; new facts come only
from context.**
A stage whose data adds no facts should turn "stored, not read" into "read"
and lose fewer read items than it gains. See `H7.md`.
*Refuter:* on items answerable from the weights, read -> lost outnumbers
stored-not-read -> read (paired, McNemar).
*Rests on:* per-path counts only (Qwen3.5 0.8B self-distillation: question
items 17 -> 35, two-hop 0 -> 76, context 87 -> 84, 18 of 104 right items
lost); G1h -> G1L costs news and encyclopedic compression, less at larger
size. The transition table is not measured.

**H20. The state holds two contradictory beliefs without collapsing to
one.**
*Registered estimator:* `collapse_first`, the share of items with `p > 0.9`
on the first decoded token; holds below 0.30, refuted above 0.70.
*Rests on:* G1d 0.4B gives 0.672 on the registered number. The prose cites a
later, softer one (0.541). Under alternation the model entrenches (the
log-odds gap grows 10.6 -> 11.4 nats). Once read as consistent with 1-2
separable facts per head; that count was a solver artefact (2026-11-12), and
the state keeps about 5.5 of 8 values readable, so storage does not explain it.

**H21. Premise validity is readable from the state before the answer.**
*Estimator:* leave-one-out F1 of a small head on per-head state statistics.
*Rests on:* 0.789 (G1d 0.4B, 40 items), 0.850 (G1h 2.9B), 0.614 on 280
surface-matched items. A bag-of-words control is beaten by a small margin
with no significance test (0.618 against 0.571). Structural invalidity is
read; plain falsehood is a knowledge question. Feeding the prompt again does
not help (F1 0.811 -> 0.769 over five passes). On a third set (2026-10-20)
whose classes share their words the state head reaches 0.992 and a lexical
MLP head on the same unigrams 0.992: there the state adds nothing over a
word conjunction. `PARTIAL` since the run of 2026-07-30 on 280 items
(F1 0.614, false positives on valid items 37%), applied to the record on
2026-11-24: the state carries structural invalidity; a head on it is not
shown to serve as a gate. The F1 on the structural set alone was never
computed.

**H22. Unattributed collective claims are a separate readable failure.**
*Rests on:* refuted on its second item set: a bag-of-words classifier beats
the state head (0.975 against 0.947). Distinct from H21 on the overlap
(`ρ = −0.054`). A third set (2026-10-20) whose classes share their words is
solved by the state head (1.000) and nearly by a lexical MLP head (0.946):
the task reduces to a string match. A third design is needed, one that a
lexical head cannot solve.

## 5. One finding, seen from several files

Two facts recur across files that were written apart, and the programme
rests on them more than on any single status.

- **Untrained time is not used.** Re-reading lowers H21's readout; `M = 0`
  matches or beats `M > 0` on every trained checkpoint (H10, HZLK §6); the
  base 0.4B's ticks contract toward a point; H10's early gain from a second
  pass is most likely recall. This is the pass-through regularity.
- **The state is a narrow neck.** Of 64 directions per head 12-29 are
  occupied and the answer depends on about half of them. A linear probe
  reads more than the model recalls (about 5.5 of 8 stored values against 3
  recalled, G1d 0.4B L11-13), and per layer it falls from L15 to L16 and
  stays low to the head (H25, HZLK ZLK.20); H20's collapse and H12's width
  are the same neck seen from the output. This is why time has to be spent in the state
  and why the channel must stay open.

## 6. What the programme is for

A small local model that reaches answers by a process over context rather
than by stored knowledge. Each part of that is a claim above, not a slogan:

| aim | where it is a claim |
|---|---|
| thought independent of the interface | non-word vector regularity; `HZLK.md` §3 (content first, format rendered after) |
| self-regulation: the model chooses its own time | HZLK (learned stop); H25 attractor (a tick sets its own rate of forgetting) |
| the model's knowledge of itself and of its sources | H19, H20, H21, H22 |
| facts in context, skills in weights | H7; H17, H18; the exact-recall branch in `HZLK.md` §6 |
| repair instead of refilling | fact classes (section 2); H9 |
| losses that carry more than the next token | `HZLK.md` §6; H26 |

## 7. Runtime wagers and retired items

Outside the research tree. A wager here is a design decision of the runtime
or a premise stated before the programme had estimators. Each carries a
record of a few lines beside its file (since 2026-11-24): status, the
estimator and the refuter of its own words, and what the status rests on.
A closed one names whether it was closed by a run or by a decision.

- *Operating policy:* H1 (constant-cost background operation, moved to
  `docs/policies.md`).
- *Premises, carried today by the claims above:* H0 (framing, a stub that
  points to HZLK), H4b (state-evolution architectures are viable for
  reasoning), H2 (withdrawn), H4a (parity with a same-size Transformer).
- *Memory and handoff:* H3, H5, H11.
- *Runtime as a peer:* H6, H14, H15, H16 (retracted).
- *Multimodal:* H13a, H13b, `note-multimodal-architecture.md`.
- *Retired mechanisms:* H23 (superseded by the exact-recall branch), H24
  (its baseline table is kept).

<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python experiments/regenerate_hyp_index.py` -->

| H | Status | Prior | Posterior | File |
|---|--------|-------|-----------|------|
| HZLK | PROPOSED | - | - | `HZLK.md` |
| H0 | CONTESTED | 0.6 | 0.333 | `H0.md` |
| H1 | RETRACTED | - | - | `H1.md` |
| H2 | WITHDRAWN | - | - | `H2.md` |
| H3 | PROPOSED | - | - | `H3.md` |
| H4a | PROPOSED | - | - | `H4a.md` |
| H4b | WAGER | - | - | `H4b.md` |
| H5 | PROPOSED | - | - | `H5.md` |
| H6 | PROPOSED | - | - | `H6.md` |
| H7 | PROPOSED | - | - | `H7.md` |
| H8 | SUPPORTED | 0.6 | 0.707 | `H8.md` |
| H9 | PARTIAL | 0.65 | 0.650 | `H9.md` |
| H10 | PARTIAL | 0.5 | 0.500 | `H10.md` |
| H11 | PROPOSED | - | - | `H11.md` |
| H12 | RETIRED | 0.25 | 0.250 | `H12.md` |
| H13a | WAGER | - | - | `H13a.md` |
| H13b | PROPOSED | - | - | `H13b.md` |
| H14 | PROPOSED | - | - | `H14.md` |
| H15 | PROPOSED | - | - | `H15.md` |
| H16 | RETRACTED | 0.2 | 0.200 | `H16.md` |
| H17 | RETIRED | 0.3 | 0.248 | `H17.md` |
| H18 | PROPOSED | 0.4 | 0.400 | `H18.md` |
| H19 | PROPOSED | - | - | `H19.md` |
| H20 | CONTESTED | 0.4 | 0.146 | `H20.md` |
| H21 | PARTIAL | 0.75 | 0.628 | `H21.md` |
| H22 | CONTESTED | 0.4 | 0.024 | `H22.md` |
| H23 | SUPERSEDED | - | - | `H23.md` |
| H24 | RETIRED | - | - | `H24.md` |
| H25 | PARTIAL | - | - | `H25.md` |
| H26 | PARTIAL | 0.4 | 0.040 | `H26.md` |
