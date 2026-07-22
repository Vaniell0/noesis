# A0.6 + A0.7 — state portability probes

State-transplant experiments. Ask two questions the A0.5 causal grid
left open:

- **A0.6.** Within a single model, does the WKV state carry *portable
  content* that continues to influence decoding after the prompt is
  swapped out? A0.5's cross-prompt corruption produced 21–99× the KL
  of matched noise (H8-causal-C PASS), but only measured single-token
  perturbation. A0.6 measures the full continuation trajectory and
  asks whether the injected state's semantic content leaks into
  generation.

- **A0.7 tier-1.** Does the WKV state survive a full continued-
  pretrain (not just a LoRA delta)? Direct swap between two same-
  architecture, same-size checkpoints: World-0.4B ↔ G1d-0.4B and
  World-1.5B ↔ G1H-1.5B. The verdict here decides whether noesis's
  runtime can migrate reasoning state across model bumps, or whether
  state is tied to a single frozen checkpoint.

## Files

| file                    | purpose |
|-------------------------|---------|
| `metrics.py`            | 5 portability metrics (task-lexicon hit rate, top-k Jaccard, alignment vs donor, first-divergence step, surface-garble proxy). Stateless, operate on materialised outputs. |
| `tasks.jsonl`           | 3 prompt pairs (math+narrative, code+prose, tasklist+reflection). Each pair has `prompt_A` and `prompt_B` designed so their content-word lexicons are disjoint. |
| `a06_run.py`            | Intra-model runner. 3 pairs × 2 prompt directions × 3 injection depths × 2 swap modes (full vs A0.5-hotspot); ``before_B × hotspot`` skipped as undefined. Emits one JSON per (pair, direction, depth, mode) cell. |
| `a07_tier1_run.py`      | Cross-checkpoint runner. Same directional matrix as A0.6 plus a checkpoint-direction axis; state transfer is WKV-only via ``load_wkv_into_state`` because donor and recipient are different models. Supports the ``0.4b`` (World↔G1d) and ``1.5b`` (World↔G1H) same-size pairs. |
| `verdict_a06.py`        | Aggregator over ``results/a06/*.json`` → per-cell + per-pair PASS/FAIL/CAVEAT summary, ready to paste into ``results_a06.md``. |
| `results_a06.md`        | *(pending)* Human-readable verdict for A0.6. |
| `results_a07.md`        | *(pending)* Human-readable verdict for A0.7. |
| `results/`              | JSON dumps of every run (populated by the runners). |

## Reused from `../A0_state_probe/`

- `probe.py::load_model` — RWKV-7 checkpoint loader (bf16, CPU).
- `probe.py::_extract_wkv_per_layer` — pull WKV list from flat state.
- `a05_intervene.py::snapshot` — deep-copy invariant against in-place
  mutation by the BlinkDL rwkv package.
- `a05_intervene.py::corrupt_cross` — full-state donor swap (A0.5's
  cross-prompt corruption); the base case for A0.6 "full swap" mode.
- `a05_intervene.py::corrupt_cross_layers` — **new for A0.6**.
  Selective donor swap at specified layer indices. Powers the
  "A0.5-hotspot" mode.
- `a05_intervene.py::load_wkv_into_state` — **new for A0.7**. Write
  extracted WKV list back into a target model's fresh state before
  decoding. Required because A0.7's donor is a *different model*, so
  we cannot reuse the donor's full state list directly (shift buffers
  are model-specific).
- `a05_intervene.py::greedy_continue` — deterministic continuation
  from a fixed state, returns tokens + full logit sequence.
- `a05_intervene.py::trajectory_metrics` — cumulative KL + token
  overlap for continuations. Powers `alignment_vs_donor` inputs.

## Design decisions

The following are locked; anything else is up for negotiation before
`a06_run.py` lands.

### D1. Continuation length

**64 tokens per continuation** (down from A0.5's 4-checkpoint
128-token sweeps). Rationale: the alignment signal decays across
continuation length, and 64 tokens is enough to see it emerge without
paying full-length inference cost. On i5-1235U bf16 the 0.4B model
runs ~1.5 tok/s greedy, so one continuation ≈ 45 s. A single A0.6
cell (3 pairs × 3 depths × 2 modes × 3 continuations {clean_A,
clean_B, cross}) = 54 continuations × ~45 s = ~40 minutes.

### D2. Injection depths

Three points, defined as "how much of prompt B has been processed
before we swap in state_A":

- `depth=0`     — swap state before any of prompt B is fed. Model
  sees `state_A` then decodes prompt B tokens on top.
- `depth=mid`   — swap state after processing ~half of prompt B (the
  first ⌊len(prompt_B) / 2⌋ tokens).
- `depth=full`  — swap state after processing all of prompt B (i.e.
  right before the first decoded token).

The design tests where state's portability is most influential. If
`depth=0` is loudest, state is being overwritten by prompt B. If
`depth=full` is loudest, state persists to affect the very next
token.

### D3. Layer selection for hotspot mode

Full-swap mode uses `corrupt_cross` (all layers). Hotspot mode uses
`corrupt_cross_layers` with `layer_indices` derived from A0.5's
`_A05_ZERO_LAYER_KL` (see `../../training/state_reg.py:100`). For the
0.4B model with 24 layers, the A0.5 sampled points are
`{0, 4, 8, 12, 16, 20}`; the load-bearing hotspot subset is
`{12, 16, 20}`. This is the same default used by A1's state-reg loss.

### D4. Decoding

Greedy (temperature 0, no sampling) for all continuations. Rationale:
alignment/lexicon signals must come from state, not from sampling
variance. A0.5's rationale for nucleus sampling was that per-seed
variance was the noise floor being measured; here, per-seed variance
would be noise on top of the actual signal.

### D5. Model scope

- **A0.6**: two models (World-0.4B and G1d-0.4B) — matches the A0.5
  grid. G1H at 2.9B remains a Gate-2 GPU follow-up.
- **A0.7 tier-1**: two checkpoint pairs — `World-0.4B ↔ G1d-0.4B`
  (definitely runnable, both are in cache) and `World-1.5B ↔ G1H-1.5B`
  (needs G1H-1.5B availability check — deferred to `a07_tier1_run.py`
  design if not readily downloadable).

### D6. Directional symmetry

Every (pair, depth, mode) combination is probed in *both* directions
per the ROADMAP A0.6 spec:

- ``AB`` — donor is state_A, recipient is prompt B. Tests whether an
  A-shaped state pulls a B-prompted continuation toward A's content.
- ``BA`` — mirror: donor is state_B, recipient is prompt A. Guards
  against asymmetric artefacts (e.g. prompt A having an intrinsic
  structural pull independent of state).

The two clean baselines (``clean_A``, ``clean_B``) are shared across
directions — only the cross continuation is re-computed for each.
All metric names in the emitted JSON are donor/recipient-relative
(``hit_donor_in_cross`` etc.), so the schema stays direction-agnostic
and downstream aggregation treats the two directions symmetrically.
A0.7 tier-1 adds an outer ``checkpoint direction`` axis on top of
this (donor ckpt vs recipient ckpt); both axes are mandated for the
full sweep.

## Verdict rules

### A0.6

Per pair × depth × mode, compute:

- `alignment` mean over the 64-step continuation (from
  `alignment_vs_donor` on cumulative KLs)
- `lexicon_hit_A` and `lexicon_hit_B` for the cross continuation
- `topk_jaccard` mean vs clean-B logits (k=10)
- `first_divergence_step` from clean-B tokens
- `surface_garble` on decoded cross text

**PASS on portability**: alignment ≤ −0.3 (cross pulls toward donor)
**and** `lexicon_hit_A > lexicon_hit_A_null + 0.05` where the null is
`hit_rate(donor=A, generated=clean_B)`, **and** `surface_garble.
coherence_flag = 1` (not degenerate). Result must hold in ≥ 2 of 3
pairs.

**FAIL on portability**: alignment ≥ 0 (cross behaves like clean-B) or
lexicon hit fails to exceed null.

### A0.7 tier-1 (per plan)

- **PASS**: alignment ≤ −0.3 with `coherence_flag = 1` at rate
  > 50 % of the same-checkpoint (A0.6) baseline for that pair.
- **FAIL** (text-bottleneck only): alignment ≥ 0 or coherence_flag = 0
  at rate < 20 %.
- **Caveat zone** (per-layer analysis needed): 20–50 %.

## Deliverables

- `results_a06.md` — verdict table, per-cell breakdown, interpretation
- `results_a07.md` — same for A0.7 tier-1
- `results/*.json` — one file per (pair, depth, mode, seed) run with
  raw metrics, continuations, and configuration

## Non-goals

- No 2.9B runs. GPU budget, defer to Gate-2.
- No tier-2 (learned projector) or tier-3 (text bottleneck). Both are
  Phase-2 track.
- No parallel inference / WKV-state fanout. Deferred until A0.6/A0.7
  verdicts land.
- No RL / active state manipulation. Static swap only.

## Wall-time estimate

Estimates below use the actual observed timing from the world-0.4b
smoke test: ~55 s prefill and ~30 s per 64-token greedy on i5-1235U
bf16. Cross-cell wall is dominated by re-processing prompt B up to
the swap point plus the 64-token greedy decode (~85–90 s per cell).

| stage | cells per pair | pairs | per-cell wall | total |
|-------|----------------|-------|---------------|-------|
| A0.6 world-0.4b (2 dirs × 3 depths × 2 modes − skip) | 10 | 3 | ~90 s (+~4 min per-pair baselines) | ~55 min |
| A0.6 g1d-0.4b | 10 | 3 | ~90 s (+baselines) | ~55 min |
| A0.7 tier-1 0.4b pair (2 ckpt dirs × 2 prompt dirs × …) | 20 | 3 | ~90 s (+~6 min per-pair baselines) | ~2 h |
| A0.7 tier-1 1.5b pair (must-run if G1H-1.5B lands) | 20 | 3 | ~4 min (1.5B on i5 CPU) | ~5 h |
| **Total wall (0.4B only)** | | | | **~4 h** |
| **Total wall (with 1.5B)** | | | | **~9 h** |

The 0.4B-only wall still fits the plan file's "~3–4 h" budget for
Phase A. The 1.5B addition is optional and only justified if the
0.4B verdict lands in the CAVEAT zone — then the 1.5B evidence would
either sharpen or refute it.
