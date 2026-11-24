## HZLK. Second pretraining: make what the weights hold readable, and read it through a self-regulated latent loop

Record: `hypotheses/records/HZLK.yml`. Runs: `hypotheses/records/HZLK.runs.yml`.

*Codename ZLK. The programme that the numbered hypotheses organise, written as the owner's interpretation of
everything the project has measured, then as claims in the form of the frame (`hypotheses/README.md`): objects,
statement, estimator, refuter, and what each rests on today. Rebuilt 2026-11-12; the version before it is in git
(`af4da08`). Rebuilt in form on 2026-11-24: the record and the readings of the runs moved beside the file, every
link written from the repository root, no status moved. Status PROPOSED: no sub-claim of the programme has
passed a registered test; thresholds below are proposals for the owner to confirm before the run that tests
them.*

How to read this file. The table below is generated from the record, `hypotheses/records/HZLK.yml`. §1 is the
interpretation, in short numbered theses (14), each pointing at the claims that make it testable. §1b is what is
proved and what is regular. §1c is reach per growth (`S`) as the direction of training where there is no
geometry. §2 defines the objects. §3 says how the model should work. §4 is the claims, each with the place of
its data. §5 keeps the positions that are still argued. §6 is the training chain derived from the claims; §6b
names every run with the place of its data, and `hypotheses/records/HZLK.runs.yml` holds the setup and the
reading of each. §7-§11 are tests, code, paid constraints, literature, and what is not claimed.

### Sub-claims at a glance

<!-- ZLK TABLE: generated from the claims in hypotheses/records/HZLK.yml by experiments/regenerate_hyp_index.py -->

| id | statement (short) | status | regime | data links | builds on | linked |
|---|---|---|---|---|---|---|
| 22 claims: 13 partial, 6 proposed, 2 contested, 1 regular | | | | | | |
| numbering, up to 24: 2 not used (23: proposed and withdrawn 2026-11-24 (a loop learned beside the store); the mechanism it misread moved to ZLK.21 on 2026-11-25; 24: proposed 2026-11-24 as a mirror of the paper's open claim, different steps draw different heads; withdrawn 2026-11-25 on the owner's word, before any run: not worth one); a number is never reused | | | | | | |
| **A. What one pass leaves behind** | | | | | | |
| ZLK.1 | CoT spends many tokens after the answer is already there | PARTIAL | G1i 2.9B pretrained, 32 pattern items | 2 | - | - |
| ZLK.2 | untrained time (re-read, ticks, pauses) is not used | REGULAR | G1i 2.9B pretrained, untrained ticks and re-read | 2 | - | - |
| ZLK.3 | tuning narrows the output; the state survives it | PARTIAL | pretrained, LoRA readout tuning | 1 | - | - |
| ZLK.11 | a pretrained model holds facts it does not read in one pass | PARTIAL | Qwen3.5 0.8B pretrained | 2 | - | - |
| ZLK.20 | the neck of one-pass reading is the late fall of rank, where key interference is highest | PARTIAL | G1d 0.4B and 2.9B pretrained | 2 | **ZLK.18** | H12, ZLK.18 |
| **B. What a reader can bring out** | | | | | | |
| ZLK.12 | a trained reader raises usable capacity toward what is held | PARTIAL | G1d 0.4B pretrained (state readout; head vs reader, same target) | 1 | **ZLK.11** | ZLK.11, <-H19, <-H22 |
| ZLK.13 | storage and reading trained together pack more at equal readability | PROPOSED | - | 0 | **ZLK.12** | - |
| ZLK.14 | relations derived from held facts are usable knowledge the loop can add | PROPOSED | - | 0 | - | H12 |
| **C. The channel** | | | | | | |
| ZLK.4 | a channel of the model's own latent can be trained without erasing the answer | PARTIAL | G1d 0.4B pretrained | 1 | - | - |
| ZLK.18 | trained ticks widen the attractor instead of collapsing it, and the output grows more expressive | PARTIAL | G1d 0.4B pretrained; long-distance regime | 2 | **ZLK.4** | - |
| ZLK.10 | an exact-recall branch queried at every tick keeps the past the settling state forgets | PROPOSED | - | 0 | - | - |
| ZLK.21 | mixture of experts in time without a partition of the weights: each tick reads a view of the same weights, taken from its own latent | PROPOSED | - | 0 | **ZLK.9**, **ZLK.20** | ZLK.12, ZLK.14, ZLK.20, ZLK.9 |
| **D. Training** | | | | | | |
| ZLK.15 | knowledge enters through a fixed teacher on the student's own tokens without breaking the rest | PARTIAL | Qwen3.5 0.8B pretrained | 1 | - | H7 |
| ZLK.19 | separability is installed by tasks that need it, not by an answer loss alone | PARTIAL | toy from scratch; P1 on G1d 0.4B | 2 | - | - |
| ZLK.16 | a loop learned from a plain pretrain beats the same chain grafted on a finished model | PARTIAL | toy from scratch (chase); pretrained not run | 1 | **ZLK.12** | H9, ZLK.12 |
| ZLK.17 | World trained by the chain against G1d compares two training methods from one base | PROPOSED | - | 0 | - | - |
| **E. Self-regulation and the interface** | | | | | | |
| ZLK.5 | the model can learn to choose its own number of ticks | PARTIAL | toy from scratch (chase), start from a trained C model; pretrained not run | 2 | - | - |
| ZLK.6 | with search the loop exceeds its teacher's one-pass reading | PROPOSED | - | 0 | **ZLK.1** | ZLK.1 |
| ZLK.7 | two views of one content land close without collapse | PARTIAL | toy from scratch (two scripts); 0.4B pretrained, inference only | 1 | - | - |
| ZLK.8 | content first, format from the question; honest self-decisions | PARTIAL | G1i 2.9B pretrained, silent answers | 0 | - | - |
| ZLK.9 | a persistent state absorbs the configuration once | CONTESTED | none measured (the owner's sessions) | 0 | - | H17 |
| ZLK.22 | loop training on a mix of domains folds them into one system that transfers to a held-out domain, below a specialist but smoother | CONTESTED | toy from scratch, 55K (1n) and 271K (1j), one seed each; pretrained not run | 3 | - | - |

<!-- END ZLK TABLE -->

### 1. The interpretation (the owner's reading of the record, in my words)

Each thesis is dated by when he stated it and points at the claims that test it. Where the record contradicts a
thesis, §5 says so.

1. **A model is weights plus harness.** The weights are the constant and hold what the model knows and can do;
   the state and the activations are the present moment. Output is an interface, not the thought. Training that
   attaches only to output funnels the model to one target and never lets it use its environment for itself.
   (2026-10-01/02; ZLK.3, ZLK.8)
2. **The explosion is in extraction, not in storage.** The heads' directions collapse, the model cannot bring
   most of what it holds to the output in one pass, and it converges instantly on the most probable answer: the
   attractor. Time in the loop is how what is held gets searched. The neck is the fall of rank toward the end
   of the model, where interference is highest (2026-11-12); the first image models built themselves a memory
   there, and the loop needs work instead (2026-09-22). (2026-10-02; ZLK.2, ZLK.11, ZLK.12, ZLK.18,
   ZLK.20)
3. **Efficiency of extraction is the measure.** A CoT model spends hundreds of tokens to move a narrow state far
   enough; efficiency is information gained about the answer per tick per unit of state motion (R_density,
   2026-08-19). Success is a better answer than the CoT teacher for less compute. (ZLK.1, ZLK.6)
4. **The model regulates itself.** It decides how far and where to move its state, how many ticks, and when to
   stop, by what it generates: an unreadable latent tick continues, a vocabulary token speaks. No threshold is
   written by us; confidence is optimizer pressure, not a stop rule. Reflection is the model taking apart and
   reorganising what it already holds. (2026-10-06, 10-22; ZLK.5, ZLK.18)
5. **The attractor is the enemy.** However large, it is bounded; the chain should widen both its capacity and
   the way it forms, moving the distribution of the latent space, not the distribution of the output.
   (2026-10-19, 10-22; ZLK.18; refined in 14 on 2026-11-21: the enemy is the global or premature attractor, while
   a per-answer attractor reached after the steps is where the program halts)
6. **Pretraining asks the state to be sufficient; the loop needs it separable.** Design the task, not the
   objective: a task where keeping items apart is the only way to answer makes the answer reward the
   separability reward. (2026-09-22; ZLK.19)
7. **Pretraining stored facts with losses to interference, like bad sectors.** Our training is a second
   pretraining that repairs the reading of what is stored and finds what was badly written so that the repair
   can be targeted. (2026-11-04/05; ZLK.11, ZLK.12)
8. **A reading policy refills what interference blocked, and the model grows in usable size without more
   parameters.** A reader that untangles allows denser packing, and relations the model derives between
   scattered facts are new usable understanding, which is what the processor needs. (2026-11-12; ZLK.12,
   ZLK.13, ZLK.14)
9. **New knowledge enters only on the model's own tokens.** Anything else breaks the rest; after S1 the model
   should be filled with knowledge as in pretraining, so the channel saturates and the model stays plastic,
   learning facts and the linking of pieces from different sources: weights times context, plus the latent
   channel that operates on both. (2026-10-22, 11-04; ZLK.15)
10. **The loop is learned early, not grafted on.** Late checkpoints have a settled read path; a plain pretrain
    (World) trained by our chain, against G1d (World after Peng's G1 training), compares two training methods
    from one base. (2026-11-11/12; ZLK.16, ZLK.17)
11. **The exact past lives beside the state.** A QKV-ROSA branch queried at every tick keeps the exact history
    while the state settles and forgets; that last stage makes the model autonomous on weak hardware, where long
    context is the main pain. (2026-10-28; ZLK.10)
12. **Tools adapt to the information, not the reverse.** Vocabulary tokens appear only where the model itself
    starts to speak; the loop runs on its own latent. (2026-10-21; ZLK.4)
13. **Neuron count does not set intelligence; depth does, and depth can come from time.** Model growth paid off
    because it brought depth. Image models went through this once: a new channel (recurrence, memory) brought
    memory and then thinking without a wider interface. A tick is the same layers run again on the task, so
    each pass works it further instead of letting it slip through: mixture of experts in time, not in weights.
    Depth grows through the policy and through data the model generates for itself. What weights can store
    still grows with size; skills, reading and work with context need not. (2026-09-22, 2026-11-12; ZLK.12,
    ZLK.16, ZLK.20). The vision record this rests on: a shallow RNN with weights shared across steps is
    exactly a very deep ResNet with weights shared across layers and performs close to it with orders of
    magnitude fewer parameters (Liao & Poggio 2016, arXiv 1604.03640); deep classifiers end in a compressing
    "tunnel" (ZLK.20).
14. **Freedom is in the choice of path; the end of the path is the answer's point, not a global attractor.**
    Freedom is freedom from the interface: the model explores its own latent space and what it can do there,
    and uses what it finds to make the work simpler. In the loop the end of a trajectory is set by the model's
    own choices along it, not by its start alone: the states reachable from one post-question state within `M`
    ticks have volume, and the model picks where in it to go. Refined 2026-11-21 by the toy's attractors
    (`attractor_probe.py`): a trained loop steps k times and then settles into a region of the answer, shared
    across contexts, and its own stop is that settling. So convergence is the program halting, not a loss of
    freedom; what kills freedom is one point for all prompts (no computation) or a per-prompt point at tick 1
    (`kl_T2_k4`: nothing left to choose). The volume belongs to the transient before the halt; at the halt the
    map should contract, or the answer would not be stable.
    The ticks are the scratchpad: re-reading rewrites the model's own record more densely (proved for the
    state, T6), which frees room to turn. RL that binds heterogeneous information more tightly raises the
    bits each part of the model carries, the loop decodes them, and later stages shorten the loop and order
    the knowledge unevenly for a cheaper answer. In H25's terms the weights are the processor (policies, and
    the knowledge that classifies objects and picks the policy) and the WKV state is the memory the program
    runs on. Freedom is not noise: rollouts that differ by the model's choices must still answer right.
    (2026-11-15; ZLK.18, ZLK.5, ZLK.13; `hypotheses/H25.md` A)

### 1b. Theory: what is proved, what is regular, and which runs are false before they start

Proved from the setup (proofs in `docs/state-and-reasoning.md`):

- **T1. The loop adds usable information, never information.** With the input fixed, the state after any number
  of ticks is a function of the post-question state `S_0`, so `I(S_t; Y) <= I(S_0; Y)`. What can grow is the
  usable information `I_V` that a bounded reader extracts (Xu et al. 2020), the reader being the model's own
  remaining layers and head. The gain `G_M = log p_M(y*) - log p_0(y*)` of one checkpoint estimates it.
  "Decompression" is raising `I_V` toward `I`; a bad sector is `I > I_V` for one pass.
- **T2. New information enters only through the input.** A fact absent from the input and the weights is out of
  reach for any `M`. A reward with `|R|` outcomes carries at most `log2 |R|` bits per episode, so reinforcement
  reweights what the model holds; new knowledge comes from a teacher that has it, in its context or its weights,
  passed on the student's own tokens with the teacher fixed. What is derivable from the weights by computation
  counts as held (`hypotheses/H12.md` §4): T2 bounds facts, not derived relations.
- **T6. Repeated delta-rule passes converge where one pass interferes.** A head's step is a generalised SGD step
  on a linear key-to-value map (`docs/state-and-reasoning.md` §2). One pass of pure writes reads back each
  binding plus crosstalk from every overlapping key; the delta step repeated over the same bindings converges to
  the least-squares map, exact for up to `n` linearly independent keys. Within a context, re-reading writes the
  residual the first pass left. This is proved for the state only; for the weights it is the analogy behind
  ZLK.13, not a theorem.

Regular (every run agrees; one counterexample refutes):

- **T3. Pass-through is the optimum unless the task is deeper than one pass.** An output-level loss is lowered
  more cheaply by repairing the one-pass read than by computing in the ticks whenever the computation fits one
  pass plus the answer's own tokens. Ticks are needed by construction only for short answers whose computation is
  long: k-hop chains with k >= 3, "which two share a city", tracking, iteration, counting.
- **T4. The loop needs a channel that is not collapsed.** If ticks carry intermediate results they differ across
  prompts and ticks (effective rank > 1, step_c > 0). Every breakage on record narrowed the channel (Muon step
  size; CE sharpening, P1; one attractor for all prompts, G1k / G1l; one format; a self-teacher drifting with the
  student); every success kept it wide (KL at T=2 to the own distribution; a frozen teacher on the own tokens).
- **T5. The channel needs an architecture that lets a vector be a non-word.** With a head tied to the input
  embeddings and a soft feed, a fed vector is read as the next word (Qwen3.5: ticks are word embeddings, one
  trajectory for every prompt). RWKV-7 has a separate head and ln0 on its input, so only the direction arrives.

**Pre-run gate.** A run is false, and is not started, unless all five answers are yes:
1. Is the needed information in the input plus the weights of the model or of its fixed teacher? (T1, T2)
2. If the run claims ticks: is the answer short while the computation is longer than one pass? (T3)
3. Is every target a teacher's distribution, with the teacher fixed, and no foreign token string as a target?
   (T2, T4; the one exception is §6 W0, form for a base that cannot answer at all, named there)
4. Is it gated by `G_M` at `M = k` vs `M = 0` of the same checkpoint and vs an arm trained at `M = 0`, on facts
   split from training, plus the channel's spread and the output read in chat? (T1, T3, T4)
5. Can the architecture carry a non-word vector? (T5)

Applied to the record, the runs that would have been stopped: every Qwen ticks arm (5); every S1 on one-pass
tasks (2, 4); gold-string decode CE (3); the self-teacher on-policy run (3); fresh-text compression as a repair
gate (1: its high bits are mostly absent information); the next-word chat game (the instrument failed).

### 1c. Reach per growth: the training direction where there is no geometry

The owner's rule for code (2026-11-04) is `S = T / (N⁺ + 1)`: a change is better the more of the existing
description it re-expresses (`T`, reach) and the less it grows it (`N⁺`, net growth). He reads it (2026-11-12)
not as a number but as a policy, an optimizer and a gradient for the places where training has no geometry, and
names the Kolmogorov criterion of compression as the base of the whole project (2026-11-05). In the frame's
terms:

*Objects.* A description `D` (code, weights, the context or state the model writes, the data it makes for
itself, a rollout of the loop), its behaviour `F(D)`, and a change `Δ`. Reach `T(Δ)`: how much of `D` the change
re-expresses, measured where the behaviour is read (lines touched in code; for the loop, how far the ticks moved
the answer: `G_M`, or `KL(p_M || p_0)`). Growth `N⁺(Δ)`: what the change adds to the description (lines added;
for the loop, ticks, emitted tokens, state motion `Σ‖ΔS‖`).

- **C1 (identity: one criterion at every level).** КПД = `ΔI_V / (M · Σ‖ΔS‖)` is `S` for the loop: usable
  information gained per motion spent. Accuracy per emitted word (DE) is `S` for the output; facts read per
  parameter (ZLK.13) is `S` for storage; `S` itself is the criterion for code. The programme measures one thing
  at four levels: reach per growth.
- **C2 (proved: a gradient needs no geometry of the outcome space).** For a policy `π_θ` that proposes outcomes
  `a` in any space (an edit, what to write to ROSA, continue or stop, which data to make), `∇_θ E[u(a)] =
  E[u(a) ∇_θ log π_θ(a)]` (Williams 1992). The geometry used is that of `θ`; the outcome space needs only a
  score `u`.
- **C3 (proved: order is enough).** Replacing `u` by its rank within a group of rollouts leaves the direction
  invariant to every monotone transform of the score; GRPO's group normalisation is invariant to affine ones
  only. So `S` never has to be calibrated as a number: a judge that orders two rollouts gives the same
  direction as one that scores them.
- **C4 (identity: the geometry the model induces).** Where the parameters carry no metric suited to the task,
  the model's own outputs supply one: `KL(p_θ || p_{θ+δ}) = ½ δᵀ F δ + O(δ³)`, `F` the Fisher information. The
  losses of the chain (KL at T=2 to the own distribution, reverse KL to a frozen teacher on the student's own
  tokens) step in the geometry of the model's outputs; the natural policy gradient (Kakade 2001) is the same on
  the policy side. It is the output-side analogue of H26's strong form (the step metric induced by what the
  weights write rather than by their coordinates).
- **C5 (proved: compression is not a matter of taste).** For a hypothesis described in `L` bits that fits `n`
  independent samples, with probability `1 − δ` its error is at most `(L ln 2 + ln(1/δ)) / n` (Occam bound,
  Blumer et al. 1987, realisable case). Among changes that keep the behaviour on the data, the one that shortens
  the description has the tighter bound; `S` ranks changes by that.
- **C6 (conjecture: the curriculum is the slope of S).** A task is worth training on while the model's `S` on it
  is still rising; where `S` has saturated it adds nothing, where `S` is zero it is out of reach (Schmidhuber's
  compression progress as the drive of curiosity, 2009, cited from memory). This is the owner's "saturate the
  loop with work", with no threshold of ours: items are drawn by the slope of their `S` over training.
  *Prior work (read 2026-11-12).* Absolute Zero (arXiv 2505.03335) is the nearest recipe: one model proposes
  and solves, an executor verifies, and the proposer is paid 1 - solve rate, 0 at solve rate 0; it gains most
  on larger bases, an untrained proposer costs only 1.4 points, and stripping hints (comments, leaked
  variables) from the tasks hurt, because hints make unsolvable tasks solvable and the solver climbs out of
  them. MATES (arXiv 2406.06046) scores data by the drop of a reference loss after one training step on it and
  refits a small scorer as the model's preferences shift. Both rank by the effect on loss or accuracy; C6 ranks
  by reach per growth of the loop, which neither measures. The cipher generators can play the executor (exact
  gold), so a learned critic only orders and cannot be gamed into inventing truth.

*Where the chain has no geometry* and takes its direction from C2-C3: the continue/stop decision (S3), what a
tick asks of ROSA (S5), the data the model generates for itself, which items enter the curriculum, and edits
to code and context outside the model. Each needs a policy whose log-probability carries the gradient and an
order on outcomes; nothing else.

*What it makes easier.* The S3 reward needs no calibrated value: ordering rollouts by the right answer, then by
fewer ticks, is the whole reward. A critic for open tasks only has to compare two rollouts, which is easier to
keep fixed and honest than a scorer; anchor items with a known answer keep it honest, and it adds only what it
knows itself (T2). The curriculum comes from the slope of `S`, not from thresholds we write (thesis 4).

*Estimator and refuter of the conjecture part.* C1 as a regularity: across checkpoints, reach per growth at the
loop level (R_density, tokens at equal accuracy) predicts held-out gain better than the training loss does;
refuted if it does not correlate. C6: a curriculum drawn by the slope of `S` against a uniform one at equal
steps, gain on held-out deep items; refuted if equal. *My caveat:* reach must be read on the answer (`G_M`,
`KL`), not on the motion, or ticks that move the state far and change nothing would count as reach; the zero-
motion guard of R_density applies to `S` in the same way.

### 2. Definitions

- **own latent**: the model's last-layer vector (before the head) fed back as the next input, Coconut-style
  (`loader.py::_peft_forward_embeds`). Not a vocabulary row, not the word hull, not a constant. RL stochasticity
  comes from a noisy continuous feed (`z = h + sigma * eps`, a Gaussian log pi), not from discreteness.
- **tick**: one forward step on a fed vector, no token emitted. **M**: the number of ticks before the answer, one
  number (owner 2026-10-27). A **phase** exists only where a boundary does something; the natural boundary is the
  model emitting a vocabulary token between runs of ticks (S3). Feeding the input again (**N**) is not an axis of
  the loop; inside it, re-reading the exact history is what ROSA gives a tick (ZLK.10). **K**: emitted CoT
  tokens. **route**: one way to an answer from the same post-question state.
- **I, I_V**: the information the state after the question holds about the answer, and the part the model's own
  reader extracts. **G_M**: the per-item gain of §1b T1.
- **fact classes** at `M = 0`: **read** (greedy right); **stored, not read** (greedy wrong, the gold ranks high
  under teacher forcing or appears among samples at T = 1); **lost** (neither). Estimators: `m0_pass_probe.py
  --m0-rank`, `--m0-samples`. The **transition table** is the classes of the same items before and after a stage.
- **derived relation**: an answer computed from two or more held facts and never stated as one (who shares a
  city, the capital of the country where X lives).
- **separable facts per head**: stored bindings a held-out linear probe tells apart on one head's state
  (`state_capacity_probe.py`), read beside the design's effective rank; with `--ticks M` after `M` own-latent
  ticks.
- **effective size**: the size of the plain model of the same family that matches a checkpoint (with its ticks)
  on the same item set. The measure of "bigger without parameters".
- **КПД (efficiency)** = R_density = `ΔI_V / (M · Σ‖ΔS‖)`: usable information gained per tick per unit of state
  motion (owner 2026-08-19). Proxy for `ΔI_V`: `G_M`, or the drop of answer entropy on the gold. Per tick:
  `(H_{t-1} - H_t) / rel_motion_t`, undefined when the motion is below 1e-3 (`noesis-tick-yield-probe` spec).
- **channel spread**: effective rank, PC1 share and centred step size `step_c` of the fed latents across a prompt
  set per tick; settle tick; cross-prompt cosine on the state `S`, not only on `h`.
- **waste fraction** w = CoT tokens after the first correct checkpoint / all CoT tokens; **tokens at equal
  accuracy** (loop tokens / teacher tokens at matched accuracy on the same items).
- **basin width**: entropy of the answer at small T and the T* at which it leaves (output side); the noise norm at
  which it leaves (state side).
- **re-read ratio** (system level): cache-read input tokens / output tokens over a session.
- **view**: the same content under another format, paraphrase, cipher or seed (`content_id` + `view`).

### 3. How the model should work (inference)

```
input   the question only (+ ROSA tags); the harness is the question each time
read    S_0 = state after the question
loop    own latent (h -> next input) -> state moves -> the model's decision, in one softmax:
        an unreadable latent tick continues | a vocabulary token speaks | (fork) | (query ROSA)
        ticks differ in content; there is no per-tick answer reward
commit  content is fixed first; the format is rendered after it, from the question (butler standard, H15)
```

What the model controls: how far to move the state, how many ticks, when to stop, what to pull from ROSA,
whether to fork and merge (H18). What we control: the environment, the verifier, the price of time, the data.
State motion (|dS|/|S|, direction) and answer entropy are kept apart; confidence is never a stop gate (a
readout-confidence exit fired after one tick on ~80% of steps because templated answers made the first token
predictable).

### 4. The claims

Each: statement; estimator; refuter; what it rests on; and the reading (the owner's and mine where they differ).

#### A. What one pass leaves behind

**ZLK.1 Waste.** A CoT teacher spends a family-dependent share of its tokens after its answer is already right.
*Estimator:* w and tokens to first correct per family (`cot_vs_routes_probe.py`). *Refuter:* w < 10% on every
family. *Rests on:* G1i 2.9B, w = 47%, 17 of 32 at the 768 cap on patterns; nothing to save on arithmetic. The
Qwen3.5 0.8B own CoT finds the capital at 192-256 tokens and then keeps going (71 of 73 kept CoTs never close in
768). *Data:* `experiments/rl/results/cot_vs_routes_g1i_pattern.json`, `experiments/rl/results/cot_vs_routes_g1i_arith.json`.

**ZLK.2 Untrained time is not used.** Re-reading, latent ticks and pauses do not beat answering at once on
reasoning items of an untrained model; re-reading helps recall items. *Estimator:* the same items at once, read
twice, with `M` ticks; recall vs reasoning items; neutral filler of equal length. *Refuter:* re-reading helps
reasoning items beyond the filler control. *Rests on:* G1i 0.469 vs 0.531 (patterns), ticks 0.344; one untrained
latent tick erases the answer (G1i 0.85 -> 0.50, G1k 0.95 -> 0.50). *Data:* `experiments/rl/results/cot_vs_routes_g1i_pattern.json`, `experiments/rl/results/cot_vs_routes_g1i_arith.json`.

**ZLK.3 Narrowness is at the output.** Readout-level tuning shifts answer entropy and T* by more than the base
model's seed spread while the state-noise basin and routing do not move. *Estimator:* T-basin on output and
state, before and after; LoRA vs full FT at matched budget. *Refuter:* the state basin moves with the readout.
*Rests on:* LoRA left routing at 0.97-1.02x (2026-08-14); a 100% tool-call corpus made every answer a tool call;
held-out CE rises while task scores rise (+0.109 to +0.402 on 0.4B). Basin not measured. *Data:* `experiments/rl/results/l_state_ablation_g1d04b.json`.

**ZLK.11 Stored, not read (bad sectors).** For a pretrained model, a measurable share of the facts answerable from
its weights is in the class stored-not-read; the greedy misses land on neighbours of the gold or on a common
attractor, not on random answers. *Estimator:* fact classes on an entity battery with type-matched alternatives
(the capital probe generalised). *Refuter:* stored-not-read under 5% of misses on a battery of at least 200 facts;
or misses uncorrelated with the gold's rank. *Rests on:* Qwen3.5 0.8B, 49 capital questions: greedy right 17, gold
ranked first among the 32 capitals while greedy misses 25; misses are neighbours (ZURICH for Bern) or a generic
LONDON; own CoT reads the question rung 1 -> 10 of 10. Two hops: the capital asked directly ranks first (BERN rank
1), inside a two-hop item rank 22. Not measured on RWKV: the battery is not built; a per-token rank map of 959
Wikipedia intros exists (`badsector_map_vital3`, Qwen) and cannot separate a badly read fact from a free
continuation without a question form. *Data:* `experiments/rl/results/qwen35_08b/rank_capitals_question/`, `experiments/rl/results/qwen35_08b/rank_capitals_mixed/`.

**ZLK.20 The neck is the late fall of rank.** The bottleneck of one-pass reading is the narrowing of the state
before the readout: from a depth fraction near 0.656 to the head, heads hold fewer live directions and fewer
separable facts than mid-stack, and the erase keys of distinct items overlap more. *Estimator:* live directions
per head at stride 1 over layers (`jlens_probe.py`); separable facts per layer (`state_capacity_probe.py --per-layer`,
ridge with lambda by CV); the mean overlap `|κ̂_i · κ̂_j|` of the erase keys written for distinct items,
per layer (`hypotheses/H12.md` estimator 4). *Refuter:* late layers separate as many facts as mid-stack, or their key
overlap is not higher (then the narrowing is compression toward the answer, not interference). *Rests on:* G1i
and G1k 2.9B fall from 16.0 to 10.9 live directions at L21 -> L22 (depth 0.656, both corpora) and hold 8-11 at
L24-L28; on G1d 0.4B the directions the answer depends on are 8.7 / 8.2 / 6.3 / 6.4 at L8 / L12 / L16 / L20,
and late layers depend on directions below the occupancy threshold; P1's sharpening hit L20 hardest (6.4 ->
2.6). The 0.4B cliff, predicted at L15 -> L16, is measured (2026-11-12, `experiments/rl/results/state_capacity_profile_g1d04b_M0.json`,
one layer = its 16 heads, CV ridge): facts read, as the sum of held-out R2 over N = 8 bound values, are 4.2 / 3.4 /
1.6 / 0.7 at L14-L17 and 0.0-0.4 at L18-L23, against 0.4-2.2 at L2-L11; at N = 16 L15 holds 0.5 and L16 none.
L12 alone keeps rising with stored pairs (3.6 / 5.5 / 6.8 at N = 4 / 8 / 16; its design rank is also the
highest). The capacity half of the refuter did not fire on 0.4B; key overlap is not measured anywhere. *Reading
for the loop:* a tick feeds back the vector that has passed the neck and sends it through every layer again, so
the gain of ticks should be looked for at the late layers. *History it rests on* (image classifiers; the owner's
remark of 2026-09-22 that the first image models built themselves a memory, and that we need work instead): deep
classifiers split into an extractor that builds linearly separable representations and a "tunnel" that
compresses them, with the numerical rank of the representations falling toward the number of classes; the
tunnel forms early in training, its depth grows with capacity relative to the task, and it hurts
out-of-distribution transfer (Masarczyk et al., NeurIPS 2023, arXiv 2305.19753; not universal, arXiv
2405.15018). At the end of training the last-layer features collapse onto their class means and the classifier
becomes nearest-mean (neural collapse, Papyan, Han, Donoho, PNAS 2020): the attractor of ZLK.18 in classifier
form. Memorisation of randomly labelled examples sits mostly in the deeper layers, where object manifolds shrink
in radius and dimension, and rewinding the last layers restores generalisation (Stephenson et al., ICLR 2021,
arXiv 2105.14602; later work finds it spread wider). The transfer to RWKV is not direct: these are residual
representations of classifiers with few classes, our numbers are per-head state ranks of a language model. The
same estimator (numerical rank of the residual stream per layer) applied to G1d 0.4B and World 0.4B makes the
comparison exact, and the per-layer weight deltas of P1 and `kl_T2_k4` say whether our training moved the
tunnel. *Data:* `experiments/rl/results/state_capacity_profile_g1d04b_M0.json`, `experiments/rl/results/state_readout_lens_g1i.json`.

#### B. What a reader can bring out

**ZLK.12 A trained reader raises usable capacity.** On one checkpoint, reading through `M` ticks makes more of
what is held usable than one pass: more separable facts per head, and stored-not-read facts read, without any new
fact in the data. *Estimator:* separable facts per head at `M = 0` and `M = 4` on the same checkpoint
(`state_capacity_probe.py --ticks`); the transition stored-not-read -> read at `M = k` vs `M = 0` on held-out
facts; effective size. *Refuter:* ticks never read more separable facts than one pass on a trained checkpoint, and
the stored-not-read class does not shrink at `M = k` beyond the `M = 0` arm. *Bound:* `I_V <= I` (T1); the claim is
a rising readable share, never knowledge from nothing. *Rests on:* externally, a post-hoc state-readout corrector
gains +7 / +6 / +2 points at 1.5B / 2.9B / 7.2B, and 1.5B with it equals 2.9B without it (Scarletwolf, 2026-08):
a better reader pays most at the smallest size. Internally, own CoT reads facts one pass misses (ZLK.11); the
one-pass baseline is at least 4 separable facts per head on G1d 0.4B (L11-13: 1/1, 2/2, 4/4 at R2 1.00 / 0.98 /
0.89 with a ridge fit; the earlier "2" came from an erratic lstsq driver, 2026-11-12). *Measured (T15,
2026-11-12, `experiments/rl/results/state_capacity_g1d04b_{base,klT2k4}_M{0,4}.cv_ridge.json`, L11-13, N = 1..32, same items
at `M = 0` and `M = 4`):* four ticks raise held-out R2 per bound value by about +0.005 at N = 16 and 32 (paired:
26/32 slots up on the base, 23/32 on `kl_T2_k4`), about 2% more read; the gain is the same on the untrained
base as on the S1 checkpoint. So the loop reads marginally more, and S1 training did not make it a better
reader of stored bindings. The first half of the refuter does not fire as written; the claim is not supported
either. The stored-not-read class was not measured here. *Data:* `experiments/rl/results/state_capacity_g1d04b_{base,klT2k4}_M{0,4}.cv_ridge.json`.

**ZLK.13 Denser packing under a trained reader.** When storage and reading are trained together, the model holds
more items in superposition at equal readability than a model trained to be read in one pass. *Estimator:* after
the same training budget, with ticks (arm B) and at `M = 0` (arm C, §6): the number of facts of a fixed battery in
class read at the arm's own `M`; separable facts per head with ticks minus without, compared across the two arms.
*Refuter:* the arm trained with ticks reads no more facts at its `M` than the `M = 0` arm reads at 0, or the
ticks-minus-one-pass gap is the same in both arms. *Rests on:* the proved part T6 for the state (iterated delta
steps converge to the least-squares map where one pass leaves crosstalk) and compressed sensing as an analogy (an
iterative decoder recovers sparse signals from fewer measurements than one linear read separates); nothing
measured. *My reading:* plausible as mathematics; the weights side is untested, and an arm C is required, since
any distillation also raises what one pass reads.

**ZLK.14 Derived relations are usable knowledge the loop can add.** A trained loop answers relation items whose
component facts the model reads at `M = 0` but whose relation it does not, and this gain transfers to relations
never seen in training. *Estimator:* held-out relation items (two-hop, shared city, comparisons) built only from
facts in class read; accuracy at `M` vs `M = 0` of the same checkpoint vs arm C. *Refuter:* the `M = 0` arm
reaches the same accuracy (then it is a one-pass skill, T3), or the gain is only on trained relations. *Rests on:*
`hypotheses/H12.md` §4 (a state that holds a relation answers queries about bindings it never stored). Qwen two hops distil
into one pass when the facts are held (control 76 of 120 mixed); the 0.4B, which does not hold the capitals,
stays at 0-2 of 120. *Reading:* the owner calls this new understanding; I agree it is new usable knowledge (I_V of
the relation), and it is not new information about the world (T1, T2). The paper must say which "new" it means.

#### C. The channel

**ZLK.4 The own-latent channel can be trained.** After S1, accuracy through the ticks is not below the untrained
model at `M = 0` on held-out cells, and ticks decode to prompt-specific content. *Estimator:* free answers at `M`
and `M = 0`, content and form apart; `chat`; `latent_geometry`. *Controls:* a constant feed must stay at its fixed
point; a feed swapped between examples must break the answer; teacher content shuffled. *Refuter:* frontmatter
item 7. *Rests on:* 0.4B: one tick no longer erases (held 0.11 -> 0.56-0.89); every trained checkpoint a
pass-through (44 / 46 of 304; 75 / 81 of 289; Qwen 197 / 204 of 289); KL T=2 keeps the tick path fluent where CE
narrows it. *Data:* `experiments/rl/results/s1_pilot_g1d04b/`.

**ZLK.18 Trained ticks widen the attractor.** After training, ticks keep a point per prompt (never one point for
all), settle later than tick 1 on deep items, and the output grows more expressive at equal content. *Estimator:*
channel spread over ticks and cross-prompt cosine on `S`; settle tick; dMI of the output (distinct words per
answer, temperature split of sharpening vs new support, a shuffled-key control); disclaimers and loops in chat;
the volume of thesis 14: effective rank of the Jacobian of the state with respect to the model's own latent
choices (noise in the k~16 subspace), taken on the transient before the halt (at the halt it should contract),
counted only where those rollouts keep accuracy >= `M = 0` - 0.1; the program form of the end (toy, 2026-11-21,
`attractor_probe.py`): past the last step the latent holds the answer (0.89-0.99) and not the next node (chance),
its per-tick move drops 2-6x, the end falls into a per-answer region shared across tables (nearest answer centroid
0.88-0.94), and the own stop coincides with that settling within one tick on 0.69 / 0.75 of examples.
*Refuter:* collapse to one point for all prompts (cross-prompt cosine on `S` >= 0.95 for two consecutive gated
checkpoints), or expressiveness rising only with loops or disclaimers. *Rests on:* `kl_T2_k4` (0.4B): effective
rank over ticks 3.2 -> 7.9, a point per prompt; chat distinct words 48.5 vs base 42.3 (`M = 0`) / 44.8 (base
`M = 4`), n = 6 questions, no disclaimer; but its latent sits at a per-prompt fixed point from tick 1. G1k and G1l
collapse to one point for all prompts by tick 6-8 (read on ticks they were never trained for, so no weight here;
dropped from the paper 2026-11-21); G1i spreads (0.88); at 2.9B a shared component holds PC1 at
0.999 already at tick 0 in all three, so the per-file breakdown has to confirm "one point". The behavioural side,
measured by the community on G1l 7B preview (No-22, 568-question agent exam, 22 720 passes, shared by BlinkDL
2026-10-24): the dominant failure is inertia, 53% of questions fall into repeat, rejection, forced close; 56% of
repeats copy the immediately preceding step, usually a call that had succeeded; prompt, nudge and sampling
changes cut it by at most about 20%, and the author concludes only data fixes it. The owner's note that reflection needs a
refusal to converge is this claim's settle-tick half: by the tick lemma a tick at fixed controls settles at
`ρ(A)`, so ticks that keep reading must keep changing the controls. *Data:* `experiments/rl/results/s1_pilot_g1d04b/`, `experiments/rl/results/chat_1t_M8_grpo/`.

**ZLK.10 The exact past beside the settling state.** A QKV-ROSA branch queried at every tick lets the loop recall
a fact from an earlier turn after `M` ticks as well as at `M = 0`, without a common attractor across turns.
*Estimator:* multi-turn recall at `M` vs `M = 0`; cross-turn cosine. *Refuter:* recall after ticks below `M = 0`
with ROSA trained; or ROSA read copies the previous turn instead of the needed one. *Rests on:* nothing; no trained
ROSA exists here (`rosa_soft` not installed). States install dispositions, not procedures: multi-turn recall 0 of 2
on every model measured externally. With ROSA a tick can re-read exact history: the write-side counterpart of
reading (T6), which the retired axis N approximated by re-feeding the prompt.

**ZLK.21 Mixture of experts in time, without a partition of the weights.** On each tick the projections do
not read `W_0` but a view of it taken from the latent of that tick: `W_eff(z_t) = W_0 + Σ_i g_i(z_t) B_i A_i`,
the state common to all ticks. No weights are split into experts; what differs from tick to tick is the part of
the same weights a tick looks through, and `g` is trained the way the control over the latent is trained.
*Estimator:* depth on derivations at equal parameter count against a width-matched plain loop; the per-tick move
of the latent; reading at `M = 16` after training at `M = 4`; the same figures on chase, where one rule repeats;
on the toy, whether the width-depth law moves at `w24`. Every characteristic is noted twice, in the arm and in
its control, and given as a delta with its error; a characteristic noted in one arm only does not count.
*Refuters:* depth not above the width-matched plain loop
at equal `P`; the latent's motion falling after `k` ticks; reading at `M = 16` after training at `M = 4` failing;
a gain of the same size on chase. *Rests on:* nothing run. *Links:* ZLK.9, the latent has to keep moving for
the views to differ; ZLK.20, where in depth the views would sit; ZLK.12 to ZLK.14, the storage and the reading
that they would let coexist.

#### D. Training

**ZLK.15 Knowledge through a fixed teacher on the student's own tokens.** On-policy distillation (reverse KL on
tokens the student sampled through its ticks) from a frozen teacher that holds the fact, in its context or its
weights, lifts the trained facts and the held-out ones, with read -> lost under 5% of read items and chat intact.
A target that is a foreign string, or a teacher that moves with the student, breaks the rest. *Estimator:* the
transition table on trained and held-out facts, split by fact; chat; fresh-text bpb at `M = 0` not worse than the
base by more than 0.01. *Refuter:* read -> lost outnumbers stored-not-read -> read on held-out facts (H7's
refuter), or the on-policy arm loses more than an off-policy arm on the same data. *Rests on:* Qwen 0.8B,
frozen-teacher on-policy: trained countries 6 -> 16 of 28, held-out 11 -> 13 of 21, form and chat kept, bpb +0.003;
the self-teacher run collapsed into a placeholder ("Answer: <answer in capital letters>"); gold-string CE on a
decode task learned the English payload prior (0 of 60); a hint teacher's answers taught "<answer>" tags. *Reading:*
the owner's "only through RL" is this mechanism: an on-policy dense per-token signal from a teacher that has the
fact, not a 1-bit reward (T2). *Data:* `experiments/rl/results/qwen35_08b/s1_onpolF_*`.

**ZLK.19 Separability comes from tasks that need it.** A stage whose items can only be answered by keeping several
bindings apart raises separable facts per head and spreads the answer over fewer shared directions; an answer loss
on items that do not need it sharpens the readout instead. *Estimator:* the pair separable facts per head +
ablation-read directions per answer, before and after (fewer directions with more separable facts = taking apart;
fewer with fewer = sharpening). *Refuter:* the task-designed stage also sharpens. *Rests on:* P1 staged recall:
recall 0.72 -> 0.96 while ablation-read directions fell 40-60% and separable facts 7/32 -> 3/32 (sharpening);
`kl_T2_k4` held or raised ablation-read at L8-L20 (no P1 dissociation). The level-3 context rows ("which two share
a city") are the first items built so that separability is load-bearing; the 0.4B solves 0 of 160 of them. *Data:* `experiments/rl/results/staged_recall_p1_step100_final.json`, `experiments/rl/results/s1_pilot_g1d04b/`.

**ZLK.16 Early loop.** The same chain run from a plain pretrain (World 0.4B) yields a wider channel and a larger
tick gain than when grafted on a finished model (G1d 0.4B), at equal data and steps. *Estimator:* arms B and D of
§6: `G_M` and tick-only items at `M = 4` vs `M = 0` on deep held-out items; channel spread; ZLK.12 separable facts.
*Refuter:* D >= B on tick gain and on effective rank. *Rests on:* nothing measured with ticks. World routes far less
through its state than G1 at 2.9B (KL under state noise 0.0055 vs 0.23) and 2-3x less at 0.4B, while holding the
prompt's identity more tightly (`hypotheses/H9.md`); G1k / G1l collapse to one attractor, a settled read path. *Data:* `experiments/A0_state_probe/results/closed_loop_toy/sO_graft_*`.

**ZLK.17 Two training methods from one base.** World 0.4B trained by this chain is not below G1d 0.4B with our best
tuning (`kl_T2_k4`) on deep held-out items and on the fact classes of a common battery, with fresh-text bpb at
`M = 0` within 0.01 of G1d's. *Estimator:* arms A and B of §6 on the same battery. *Refuter:* A beats B with
p < 0.05 on both the deep items and the fact classes. *Rests on:* nothing. G1d shares World's trained weights
(lineage cosine 0.743), so what each method moved is readable matrix by matrix. *Reading:* G1's data budget is far
larger than ours, so "not below" is the claim, not "above"; thresholds are the owner's to set.

#### E. Self-regulation and the interface

**ZLK.5 Self-chosen time.** A learned stop reaches the best fixed `M` per family with <= 85% of its ticks on a
mixed-family set, and moves with family. *Controls:* random stop at equal mean `M`; best fixed `M`; price sweep.
*Refuter:* frontmatter item 4. *Rests on:* toy only (labels first, then reward; lr 3e-3 collapses into stop-early).
Huginn learns backtracking from variable recurrence depth alone (arXiv 2602.08100): a reason to train at random `M`.
*Measured 11-20 (toy, `sL_ramp1_seed{4,5}`, plan 1l):* no stop head; the answer head is read after each tick and
the loop stops at the first confident tick; trained by imitating a frozen snapshot at every tick, weight (t/8)^1.
Own stop 0.905 / 0.912 equals fixed M8 on the same model, within 0.03 of the start (0.923 / 0.936), at 64% / 62%
of the ticks; a random stop at the same ticks 0.60 / 0.58; ticks used rise with the hop count (1.2-1.6 to 8.0). *Data:* `experiments/A0_state_probe/results/closed_loop_toy/sL_*`, `experiments/A0_state_probe/results/halting_chain.json`.

**ZLK.6 Beyond the teacher.** At <= 60% of the teacher's tokens, accuracy >= teacher + 3 points on at least one
family, none worse than -3. Bounded by T1/T2: beyond the teacher's one-pass reading, never beyond what it holds.
*Rests on:* the room estimate of ZLK.1 only.

**ZLK.7 Views without collapse.** A representation term over two views of one content (stop-gradient target,
contrast across examples) raises the similarity of post-question states across views without collapsing the spread
across examples, and raises accuracy on held-out formats. *Refuter:* collapse, or no gain over the answer loss.
*Rests on:* format changes the content of the answer in 32 of 48 grids (G1i; paraphrase floor not measured). *Data:* `experiments/A0_state_probe/results/closed_loop_toy/sN_joint_*`.

**ZLK.8 Content first, format from the question.** Format is read from the question; self-decisions are trained
with a proper scoring rule. *Rests on:* JSON and sentence formats obeyed 100%, a bare number 0% (G1i, silent);
nanodiff: same accuracy, ECE 0.333 -> 0.036.

**ZLK.9 Absorption.** After the configuration is fed once, answers that depend on it are as accurate as with the
history re-injected, at a fraction of the tokens read. *Refuter:* H17's. *Rests on:* nothing; the owner's sessions
re-read every input token about 79 times (98.8% of input tokens were cache reads), which motivates it.

**ZLK.22 One system across domains.** A loop trained on a mix of domains without domain D solves D better than
the same model trained by CE on the same mix, and below a model specialised on D; its accuracy falls more slowly with
distance from the training mix, and the tick latents of different domains share directions. *Estimator:* held-out
family accuracy (e.g. 5 of 25 cipher families, after the cipher control shows the cipher is read) for loop, CE-mix
and specialist; the slope of accuracy over distance; cross-domain similarity of tick latents before and after.
*Refuter:* no gain on the held-out family over CE-mix beyond seed spread, or gains only on families in training.
*Rests on:* the owner's question 11-19; outside: RL transfers to unseen variants better than imitation (Chu et al.
2025), representations converge across models (Huh et al. 2024), both without a latent loop. Toy so far: one step
generalises within a task (1b) and did not carry from context to stored facts (cs). *My reading:* plausible because
shared tick weights make one general step shorter than one path per domain; the real test is on World 0.4B. *Data:* `experiments/A0_state_probe/results/closed_loop_toy/sN_joint_*`, `experiments/A0_state_probe/results/closed_loop_toy/sJ_*`, `experiments/A0_state_probe/results/closed_loop_toy/s1c_cs_*`.

### 5. Positions still argued

- **What A0.5 says about World.** The owner reads it as World's WKV doing no computation. It measures how much the
  next token depends on the state, not computation. At 0.4B World depends 2-3x less than G1d; at 2.9B 40x less,
  while keeping the prompt's identity more tightly. On the 2.9B pair his reading holds in the form "a plain pretrain
  routes little through its state"; whether its state computes is ZLK.16's question.
- **Self-distillation of World.** Agreed: World's own answers carry nothing to re-read (T1), so its teacher is
  foreign and frozen.
- **A trained initial state teaches form, adds no facts.** Mine; consistent with the external measurement that a
  LoRA recalls injected identity facts about 100% and state tuning about 30% (RWKV-LM #338).
- **Weight edits remove an obstacle, they do not add an ability.** Mine (Sparks of Weight-Space Programming,
  2026-11-02, not re-read since).
- **"New" in ZLK.14.** See the claim; both readings are kept and the paper separates them.
- **Mixture of experts in time, and the weights.** Thesis 13 reads a tick as mixture of experts in time, the same
  weights at every tick. The owner's reading of 2026-11-23: for the interference of the weights it did not
  confirm; the indicators came from the balance of the WKV state, and the loop touched more areas without
  closing the gaps. For computation it did: time buys depth (W4). ZLK.23 gives the loop weights of its own;
  whether they must change over the ticks is left to its estimator.
- **S2 teaches knowledge.** The earlier text said S2 and later optimise a policy only. The owner's word of
  2026-11-12 puts knowledge widening into S2; the reading that keeps both true is ZLK.15: knowledge enters by
  on-policy distillation from a teacher that has it, and from S3 on the stages are policy.

### 6. Training chain

Common: full fine-tuning, no LoRA; Muon lr 3e-5 to 1e-4, aux AdamW on its own lr; the teacher is fixed (a cache or
a frozen copy, never the weights being trained); targets are teacher distributions on the student's own tokens;
splits by fact; every run names its result path, dataset names + SHA-256 and a written prediction first; at least
three data orders per claim; a checkpoint per eval, none deleted; every set has a no-prior control (random payload,
shuffled key); every run is read on its output (chat, samples), not only on accuracy.

Arms for the comparison of ZLK.12-ZLK.17:

| arm | start | trained by | role |
|---|---|---|---|
| A | G1d 0.4B | `kl_T2_k4` (KL T=2 to its own `M = 0`, `M = 4`, step 100) | our best on the G1 side |
| B | World 0.4B | W0, then S1-S2 with ticks (`M = 4` until S3) | the claim |
| C | World 0.4B | the same data and steps at `M = 0` | separates the teacher from the ticks |
| D | `kl_T2_k4` | S1-S2 of the same chain | the same chain on a finished model |
| refs | World, G1d, the teacher at `M = 0` | none | floor and ceiling |

| stage | what is trained | signal | gate | tests | state |
|---|---|---|---|---|---|
| S0 data | the items, sorted by fact class on the student and on the teacher | - | every family has items the teacher solves and the student does not; the level-3 rows and k-hop chains have a teacher | ZLK.11 | plain and cipher rungs measured; level-3 rows: 0.4B 0 of 160; teacher pass and k-hop generator not run |
| W0 form (a plain base only) | answer form for a base that cannot answer: off-policy KD on the teacher's sampled answers (full-vocabulary KL, T=2), then a growing on-policy share | KL to the frozen teacher | chat fluent EN/RU, no loops, format obedience per format, fresh-text bpb at `M = 0` within 0.01 of the base | - | not run; the one place where the teacher's token string enters the context (rule 3 of the gate), until the on-policy share takes over |
| S1 channel | the answer through `M` own-latent ticks, on-policy reverse KL to the frozen teacher (own `M = 0` for a base that answers, a bigger teacher otherwise), anchors on items the student reads | KL | the tick does not erase (free answers at `M` >= `M = 0` base); same checkpoint at `M = 0` vs `M` on deep held-out items; ticks prompt-specific; chat fluent; channel gate on `S` | ZLK.4, ZLK.18 | 0.4B and Qwen pilots: pass-through (§6b) |
| S2 knowledge widening | facts and relations the teacher holds: knowledge text (Wikipedia vital intros, extendable) with the teacher's per-token distribution as target and ticks at random boundaries with random `M` in 0..4; the fact battery and relation items on-policy, the teacher reading the fact in context or from its weights; saturated, contradictory sequences as a share of the rows | KL, on-policy | transition table on held-out facts (stored-not-read -> read up, read -> lost < 5%); `G_M` on derived relations vs arm C; bpb on seen text at `M = k` below `M = 0`; fresh-text bpb at `M = 0` within 0.01 | ZLK.12-ZLK.16 | not run |
| S3 time and policy | continue = an unreadable latent tick in the answer softmax, speak = a vocabulary token; noise in a k~16 subspace of the latent inside log pi; verified answer or continuation-gain reward minus a price per tick | policy gradient | learned stop reaches the best fixed `M` with <= 85% of its ticks; random stop worse; the noise window opens (rollouts disagree at acc >= `M = 0` - 0.1); settle tick later than 1 with distinct ticks | ZLK.5, ZLK.6, ZLK.18 | toy only |
| S4 format and honesty | required format read from the question; honest self-decisions; the constitution later | CE / proper scoring rule | obedience per format; ECE | ZLK.8 | obedience measured |
| S5 ROSA | a QKV-ROSA branch beside WKV (output zero-initialised, Q/K/V trained), queried at every tick | S4 loss + recall of earlier turns | ZLK.10 | ZLK.10 | not started |

Order is strict; a stage opens when the previous gate is measured. Every gate is one `experiments/run.py --backend
peft [--weights CKPT] --tests m0_pass,chat,latent_geometry,latent_noise` run plus the stage's own number;
`state_capacity_probe --ticks` and `state_readout_lens` read the neck as a pair. M>1 is opened by the decision
rule of the tick-yield spec (`~/.claude/plans/noesis-tick-yield-probe-2026-10-26.md`), not by accuracy alone.

#### 6b. Record of the runs

#### 6b. Record of the runs

Each run with its verdict and the place of its data, a path from the repository root. The setup and
the full reading of each are in `hypotheses/records/HZLK.runs.yml`, under the run's name.

| run (date) | model | verdict | data |
|---|---|---|---|
| P1 staged recall (10-02) | G1d 0.4B | binding skill trainable; the gain is sharpening, not capacity | `experiments/rl/results/staged_recall_p1_step100_final.json` |
| S1 pilot (10-27) | G1d 0.4B | tick no longer erases; pass-through | `experiments/rl/results/s1_pilot_g1d04b/` |
| S1 `M = 4` (10-28) | G1d 0.4B | pass-through (44 vs 46 of 304); latent fixed from tick 1 |  (same) |
| hint distillation (10-28) | `kl_T2_k4` | pass-through (75 vs 81 of 289); the teacher's tags and loops copied | `experiments/rl/results/s1_pilot_g1d04b/hint3_*` |
| latent noise (10-28) | 0.4B | ticks damp noise after S1; the window is unreadable here | `experiments/rl/results/s1_pilot_g1d04b/` |
| Qwen hint-3 (10-29) | Qwen3.5 0.8B | ticks below the control (119 / 105 vs 195 of 289); the tied head spells the answer in ticks | `experiments/rl/results/qwen35_08b/` |
| Qwen ciphers (10-29/30) | Qwen3.5 0.8B | 0 on every rung; decode SFT learned the payload prior (0 of 60) | `experiments/rl/results/qwen35_08b/` |
| Qwen self-KL (11-04) | Qwen3.5 0.8B | chat kept; stored-not-read not repaired | `experiments/rl/results/qwen35_08b/` |
| Qwen on-policy (11-05) | Qwen3.5 0.8B | frozen teacher: trained 6 -> 16 of 28, held-out 11 -> 13 of 21; the self-teacher collapsed | `experiments/rl/results/qwen35_08b/s1_onpolF_*` |
| compression (11-05) | Qwen3.5 0.8B | ticks cost about 0.08 bpb in raw text; frozen arm +0.003 at M = 0 | `experiments/rl/results/qwen35_08b/compress_*` |
| RWKV chain (11-07) | G1d 0.4B | 0 of 160 eval, 2 of 400 train; stopped before the G1L teacher | `experiments/rl/results/rwkv_2026_11_07/` |
| cipher ladder (11-13) | G1d 0.4B | 0 of 576; the answer copies the ciphertext or echoes the shown prefix | `experiments/rl/results/ladder_g1d04b_M0/` |
| cipher lookup (11-13) | G1d 0.4B | cipher->plain 0.57-0.81; plain->cipher 0.002-0.004 | `experiments/rl/results/cipher_lookup_g1d04b/` |
| ticks vs inverse lookup (11-13) | G1d 0.4B, `kl_T2_k4` | refuted: a tick does not transpose an association (inverse <= 0.009 at M = 4) | `experiments/rl/results/cipher_lookup_g1d04b_{base,klT2k4}_M{0,4}/`, `experiments/rl/results/cipher_lookup_g1d04b_base_twice_M{0,4}/` |
| cipher ladder v2 (11-13) | G1d 0.4B | 1 of 576; the answer walks the key table in order | `experiments/rl/results/ladder_trace_g1d04b_M0/` |
| cipher ladder v3 (11-13) | G1d 0.4B | 28 of 576; the first zone with a solve rate between 0 and 1 | `experiments/rl/results/ladder_indexed_g1d04b_M0/` |
| training with ticks vs inverse lookup (11-13; order 0 trained, read pending) | G1d 0.4B | inverse lookup learned without ticks from ~400-600 rows; order 0 trained, read pending | `~/tmp/inv/{A_M0,B_M4}_s0*` |
| closed-loop toy from scratch, stage 1 (11-15) | RWKV-7, 2 layers, width 48, 49.8K params, trained from scratch | ticks buy depth: CONFIRMED (+0.74 / +0.74 / +0.78 at k = 3); SIGReg fails | `experiments/A0_state_probe/results/closed_loop_toy/s1_*` |
| closed-loop toy, stage 1b: affine cipher seen through runs (11-16) | same toy, C recipe (M = 8, per-tick target), A (M = 0), and a chase + affine mix | a rule from runs not learned; the shift is won only past its closed form (k >= 6) | `experiments/A0_state_probe/results/closed_loop_toy/s1b_*` |
| closed-loop toy, stage 1f: time against parameters (11-17) | one-pass A (M = 0) with 4 / 8 layers vs C (2 layers, 8 ticks, 49.8K) | time for depth: 2 layers with 8 ticks beat a 4-layer pass at half its parameters | `experiments/A0_state_probe/results/closed_loop_toy/s1f_*` |
| closed-loop toy, phase A: stabilising the recipe (11-17) | chase C recipe (8 ticks, per-tick node target, curriculum), one knob at a time, 5 seeds | no knob gives 5 of 5; the stall rule REFUTED; the base recipe passes 14 of 15 | `experiments/A0_state_probe/results/closed_loop_toy/sA_*` |
| closed-loop toy, stage 1e: spatium rewrite derivations (11-17) | same toy, vocab +30 symbols; CE vs CE + owner's ZLK term (tick t -> EMA latent of hop t's term) | truth 0.90-0.98 at every depth; the ZLK term refuted on both answers over 3 seeds | `experiments/A0_state_probe/results/closed_loop_toy/s1e_*`, `~/tmp/toy/breakdown_s0.txt` |
| closed-loop toy, stage 1c: facts in the weights (11-16) | same toy, vocab +64 id tokens, 55,968 params | one pass stores up to 1.46 bit/param; facts in the weights are not composed on unseen starts | `experiments/A0_state_probe/results/closed_loop_toy/s1c_*` |
| closed-loop toy, 1e variants (11-17/18) | last_rule on derivations, depth 4-8 on val_laws at M = 8 | no gain on clean derivations: anonymised CE, goal, self and both refuted | `experiments/A0_state_probe/results/closed_loop_toy/s1e_lr_*` |
| closed-loop toy, 1d ciphers from scratch (11-17/20) | width 96, 271K params; character-level decryption, key or table shown, 25 ciphers, HH-RLHF texts | 0.605-0.608 in every arm, exact 0: the English prior | `experiments/A0_state_probe/results/closed_loop_toy/s1d_*` |
| closed-loop toy, 1g distillation (11-19) | teacher: base-recipe C at M = 8 (seeds 14 / 10 / 15) | born-again student carries the iteration: CONFIRMED; a one-pass mimic cannot (<= 0.24) | `experiments/A0_state_probe/results/closed_loop_toy/sG_*` |
| closed-loop toy, 1h width and 1i doubt (11-19/20) | chase C recipe at widths 12 / 24 / 30 / 36 / 42 / 96; doubt: K stored tables, 8 pairs in context, the queried pair sometimes a lie | eight steps fit from 30K (w36 2 of 2) and not at 22K (w30 0 of 2); w48 6 of 7; a lie is overruled (1.00) | `experiments/A0_state_probe/results/closed_loop_toy/sH_*`, `experiments/A0_state_probe/results/closed_loop_toy/sI_*` |
| closed-loop toy, 1k GRPO on the latent (11-19/20) | from arm B (3-hop wall) | REFUTED: an outcome reward with latent exploration does not break the wall | `~/tmp/toyk/results/sK_*` |
| closed-loop toy, 1l own stop without a head (11-19/20) | from C (fixed M8 0.92-0.94) | ZLK.5 on the toy CONFIRMED: own stop equals fixed M8 at 62-64% of the ticks | `experiments/A0_state_probe/results/closed_loop_toy/sL_*` |
| closed-loop toy, 1o ticks grafted on a one-pass toy (11-20) | the toy, started from a trained one-pass run | grafted ticks pass k6 in 2 of 3 seeds, as reliable as grown: the toy form of ZLK.16 REFUTED | `experiments/A0_state_probe/results/closed_loop_toy/sO_graft_*` |
| closed-loop toy, 1n two domains, plain and Cyrillic (11-20) | the toy, 55K parameters | no transfer in bpc (joint +0.46 against Cyrillic-only: REFUTED); deep-layer CKA between scripts 0.44 -> 0.71 | `experiments/A0_state_probe/results/closed_loop_toy/sN_joint_*` |
| closed-loop toy, 1j plain and cipher domains (11-20) | the toy, 271K parameters | joint -0.014 bpc | `experiments/A0_state_probe/results/closed_loop_toy/sJ_*` |
| 1t ticks without the pretraining term (11-21) | G1d 0.4B | 100 full-weight steps on 160 Caesar rows, KL + GRPO, no pretraining CE beside it: the chat broke through its ticks (loops, refusals) | `experiments/rl/results/chat_1t_M8_grpo/` |

### 6c. The closed-loop toy as the bench (11-16)

One small RWKV-7 trained from scratch (`experiments/A0_state_probe/closed_loop_toy.py`, plan
`noesis-closed-loop-toy-2026-11-15`) tests the claims where the 0.4B cannot separate them. Claim -> stage -> estimator:

| claim | toy stage | estimator |
|---|---|---|
| thesis 13, depth from time | 1 (done) | accuracy by hop count at M = 8 vs M = 0, same parameters |
| ZLK.16, ZLK.19 | 1 (done) | M = 0 read of a tick-trained model; tick-latent shuffle cost |
| ZLK.13, ZLK.20 (write vs read; rank) | 1 probe | linear decode of every pair from the WKV state; effective rank after pairs, prompt, ticks |
| rule from runs (owner 11-16) | 1b | affine cipher seen only through runs, query unseen; M = 8 vs M = 0 |
| ZLK.11, ZLK.12, H25 bits per parameter | 1c | facts in the weights: capacity in bits per parameter at M = 0; k-hop use of the same facts with ticks |
| ZLK.5, thesis 14 | 2 | own stop; noise in log pi; ticks used vs best fixed M |
| deep derivation, the owner's ZLK term (RSC student) | 1e | spatium rewrite items, depth 1..8, held-out laws: CE vs CE + per-tick term latent (EMA target); cost vs exact BFS |
| §1c S, R_density, owner's rule | 3 | order reward vs CE only on unseen compositions |

Dropped from the toy line: rewarding motion (L_state, refuted), SIGReg at 0.05 (costs accuracy), weight rounding
as a capacity measure (it measures precision, not information; kept only as the cost of a long chain).

### 6d. Methodology from the toy: the recipe, why it works, what not to retry (11-17)

Settled on the closed-loop toy (2-layer RWKV-7, ~50K parameters, from scratch; numbers in §6b). Each item says what
was measured; the scale is the toy's, the transfer to trained models is §6d.5.

**6d.1 Recipe.** (R1) A tick feeds the last layer's normalised output back through `ln0` as the next input; weights
are shared across ticks; M fixed during training (8). (R2) Ticks from the first step: grown, not grafted. (R3)
Tasks one pass cannot solve (depth beyond what the layers give) and where nothing repeats (fresh context per
example). (R4) A curriculum on depth, k_max 1 -> 8 over 3/4 of the steps; uniform depth from the start drowns the
gradient. (R5) Where the intermediate step is exact and atomic (one node), a per-tick target: tick t predicts a
frozen embedding of step t. (R6) CE on the answer always.

**6d.2 Why it works (mechanism, each measured).** (W1) The tick latent becomes a register holding the current value:
feeding each tick another example's latent costs 0.78-0.87 at the depths solved, with or without R5. (W2) Without R5
the model learns an unrolled program of fixed depth (3 hops in every seed, then chance); with R5 the same weights
apply one step per tick (to 8 hops in 6 of 7 seeds): the per-step target turns a circuit into an iterable
instruction. (W3) The loop changes the write side: pairs written to the WKV state decode better (0.81 -> 0.92-0.96)
in fewer dominant directions; no collapse toward rank 1 over ticks (3.5-6.8). (W4) Time substitutes for depth: one
pass gains about one hop per layer (4 layers, 1.9x parameters: chance at 6-8 hops); 2 layers with 8 ticks hold
0.72-0.92 there. (W5) The iterated step costs the same weight precision per step as the unrolled one; long chains
need about one bit more (rounding error accumulates over ticks). (W6) On derivations over held-out laws the loop
names the rule applied at the last step at ~3x chance (0.39 vs 0.125), M = 0 at chance: partial derivation.

**6d.3 Limits found.** (L1) Facts stored in the weights are not composed on unseen starts, with or without ticks,
even when every edge is stored: trained composites are memorised (the two-hop curse holds under the loop). (L2) In
50K parameters a context task and storage in the weights compete (stored edge accuracy 0.16-0.41 vs 1.00 alone, 0.01-0.10 bit/param; the M = 0 control
starves too). (L3) Seed
instability: 1 of 7 C seeds never passes one hop; derivation CE ranges 0.27-0.48 across seeds.

**6d.4 Do not retry (each with the reason it failed).** (N1) Grafting ticks on a finished model with tasks one pass
solves: pass-through (T3, T20) - the task never asked for the ticks. (N2) Rewarding state motion: motion is not
use (held-out CE +0.40). (N3) Ungated shaping in policy gradient near zero accuracy: shaping takes over (run6).
(N4) SIGReg against collapse: there is no collapse to prevent (W3) and it costs accuracy. (N5) Latent auxiliary
targets on composite terms over CE (the hop-term target, goal, self): no gain over 3 seeds; R5 pays only when the
intermediate is exact and atomic. (N6) Rule tasks with a closed form in k (shift; affine mod 16, whose powers are
the identity) as depth tests: one pass computes them. (N7) Judging an add-on by final accuracy at the trained depths
at a ceiling: inside the seed spread; measure what the idea predicts (extrapolation in depth or M, data
efficiency, per-tick decode). (N8) Storing facts first, then the curriculum, to make the loop compose them: no
change (L1). Caution on N5: the label-free targets were tested on clean derivations only, where CE
already carries the signal; their claim is about data where the structure must be found (stage 1d, text
enciphered), not tested yet.

**6d.5 Transfer.** General RWKV-7 models: the recipe asks for R3 data (anonymised derivations, chase) and trained
ticks with CE; ZLK.16 predicts a model closer to raw pretraining (World) learns them more readily than one tuned
further (G1d); the laptop precursor (plan noesis-world-a30 §0) tests it. RWKV-8: RWKV-8 keeps the RWKV-7 time mix
and adds ROSA, an exact-match suffix-automaton block over the history, trained with the rest (community map; v8 code
not read yet); its public checkpoints are toy-scale (39.6K and ~1M parameters), the scale of this toy. L1 and ZLK.10
mark where ROSA belongs: the exact past is fetched from history at each tick instead of being stored and composed
in the state or the weights. Extension, not built: the same recipe with a ROSA block queried per tick, measured on
chase with more pairs than the WKV state holds (n beyond state capacity) and on composition over facts in context
far back; read BlinkDL's v8 code whole before writing it.

### 7. Tests with controls

| test | instrument | control | decision rule | claims | state |
|---|---|---|---|---|---|
| T-gap | `cot_vs_routes_probe.py` | same items every route; paraphrase floor | w, tokens to first correct per family | ZLK.1 | done once |
| T-recall | re-read on recall vs reasoning items | neutral filler | re-read helps recall only | ZLK.2 | not run |
| T-classes | `m0_pass --m0-rank --m0-samples` on an entity battery | type-matched alternatives; held-out facts | transition table per stage | ZLK.11, 12, 15 | capitals only (Qwen) |
| T-capacity | `state_capacity_probe --ticks`, `state_readout_lens` | design eff-rank; multiplicity; P1 as negative | separable facts with ticks vs one pass, per arm | ZLK.12, 13, 19 | T15 done on G1d 0.4B (CV ridge) |
| T-relations | held-out relation items from read facts | arm C | `G_M` and tick-only items vs arm C | ZLK.14 | not built |
| T-yield | tick-yield probe | constant feed; shuffled answer; zero-motion guard | R_density(t) CI > 0 and paired acc rise | ZLK.4, 5 | spec only |
| T-channel | `latent_geometry` + S-side cosine | constant arm; swapped feed | collapse criterion of ZLK.18 | ZLK.4, 18 | measured per checkpoint |
| T-basin | temperature sweep; state noise | neutral-filler / random-word prefix | output vs state basin | ZLK.3 | not run |
| T-text | `compress_probe` | fresh text (nothing broken) vs seen text (repair) | bpb at `M = k` vs `M = 0` | ZLK.12, 15 | Qwen only |
| T-source | three conditions (`hypotheses/H19.md`) | unrelated passage | context gap grows on in-context items | ZLK.15 | not run |
| T-stop | S3 gate | fixed `M`; random stop at equal mean | ZLK.5 refuter | ZLK.5 | after S3 |
| T-views | similarity across views | shuffled views | ZLK.7 refuter | ZLK.7 | after S1 |
| T-data | method fixed, composition varied | same token budget | does composition matter once the method is rich | - | not run |

### 8. Code and data

| status | where |
|---|---|
| exists | `experiments/rl/`: `train_latent_s1.py` (self / cache / gold targets, KL T, `--on-policy`, `--frozen-teacher`, `--teacher-model`, `--anchor-items`, `--teacher-hint`, `--teacher-cot`, checkpoint per eval, `--device/--dtype`), `loader.py` (`forward_stateful_embeds(..., return_hidden=True)`, `MuonHybrid`), `hf_loader.py`, probes registered in `experiments/run.py` (`m0_pass` with `--m0-ticks --m0-rank --m0-samples --m0-hint`, `chat`, `latent_geometry`, `latent_noise`, `known_heads`), `state_capacity_probe.py` (`--ticks`, uncommitted), `state_readout_lens.py`, `compress_probe.py`, `cot_vs_routes_probe.py`; data: `training/scripts/gen_ciphers.py` (missing since 2026-11-25) (decode / context / question / mixed, levels, hint ladder, formats), the catalog, Wikipedia vital intros (959, outside the repo), UncheatableEval 2026-07 |
| must change | `train_latent_s1.py`: an off-policy teacher-sample mode with an on-policy share schedule (W0); a text mode (teacher KL per token on passages, ticks at random boundaries, random `M`) (S2); a run manifest; `gen_ciphers`: name and city pools widened (22 / 16 / 32 now), fact-split |
| to write | the entity fact battery from Wikidata triples of the vital entities, with type-matched alternatives; a k-hop generator (k = 3..5); relation items built from read facts; the tick-yield probe; the continue class in the answer softmax and its trainer (S3); k-subspace latent noise in log pi; a QKV-ROSA branch (S5) |

### 9. Constraints already paid for

LoRA is out. Markers and constant feeds collapse to a fixed point by tick 3-4; distinct inputs reset the direction
(0.097). Discrete word tokens are a test rig; the channel is the own latent. RL does not start on all-wrong groups.
GRPO's log pi over answer tokens only makes shaping on a deterministic loop inert (advantage change 6e-8). No reward
for per-tick correctness. A teacher state is not a target. The teacher is fixed. Gold strings are not targets.
Muon at 0.02 moves the worst tensor 349% of its norm in one step. A training corpus without an end-of-answer token
teaches loops. Personal session data never enters a corpus or git.

### 10. Literature, and what is ours

Continuous latent reasoning by feeding back the last hidden state (Coconut, arXiv 2412.06769); recurrent-depth
reasoning with backtracking from variable depth (Huginn, arXiv 2602.08100) and per-depth process rewards (LSRL,
Findings EMNLP 2025); pause tokens help only after training (arXiv 2310.02226); just-read-twice for recurrent LMs
(arXiv 2407.05483); overthinking after the first correct answer (arXiv 2412.21187); LoRA learns less and forgets
less (arXiv 2405.09673); JEPA objectives for LLMs (arXiv 2509.14252); usable information (Xu et al. 2020);
on-policy distillation from a student's own samples (GKD, Agarwal et al. 2024, cited from memory); a state-readout
corrector that pays most at small size (Scarletwolf, 2026-08); the tunnel effect in deep classifiers (arXiv
2305.19753), neural collapse (PNAS 2020), memorisation in deep layers (arXiv 2105.14602), read for ZLK.20;
weight-shared recurrence as depth in vision (Liao & Poggio 2016, arXiv 1604.03640), for §1 thesis 13;
REINFORCE (Williams 1992), the natural policy gradient (Kakade 2001), the Occam bound (Blumer et al. 1987) and
compression progress (Schmidhuber 2009, from memory), for §1c. The knowledge capacity of a language model,
2 bits per parameter whatever the routing, a mixture of 32 experts included (Allen-Zhu and Li, ICLR 2025,
arXiv 2404.05405), and 2 bits per synapse of a perceptron (Cover 1965; Gardner 1987), for ZLK.23. Not found
elsewhere: a latent loop on a fixed-size
recurrent state whose continue/stop decision is the model's own action, trained against a verifier and a price on
time, measured by the readable share of what the model holds. (A search, not a survey.)

### 11. Not claimed

No claim that the loop beats the best CoT model, only this teacher on tasks where time helps. No claim about
pretraining from scratch: "second pretraining" is a post-training chain on a pretrained base; a latent loop inside
pretraining (an offer to the RWKV-8 work) is a proposal outside these tests. No priority claims. The Muon write-up
(`docs/muon-rwkv7-finetuning.md`) is a separate piece.
