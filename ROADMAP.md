# Roadmap

noesis develops across three parallel tracks that converge at integration
milestones. Cognitive and memory tracks progress independently — do not
serialize them.

## Track A — Cognitive (model training)

### A0. Baseline (weeks 1–2)
- Install RWKV-7-G1 2.9B via **in-process rwkv-cpp** (Ollama removed as
  backend 2026-07-25 — see `docs/policies.md` § Model backend
  sandboxing; rwkv-cpp is required because it exposes
  `rwkv_get_state_len` / `rwkv_eval(..., state_out)` /
  `rwkv_clone_state`, which the state-work workstream — lens
  persistence, H17, H18 — depends on). Verify inference throughput on
  the user's hardware. **Measured 2026-07-23**: 0.4B World Q8_0 via
  rwkv.cpp on i5-1235U (Alder Lake, 4 threads) delivers ~30 tok/s
  steady-state; load 506 ms; RSS ~1.16 GB. 2.9B numbers measured on cloud VM (RTX 4090).
- Assemble a held-out eval set of 30–50 real reasoning tasks drawn from
  the user's actual workflow — not GSM8K, not MMLU.
- Baseline eval: RWKV-7-G1 2.9B against Qwen-2.5-3B-Instruct and Phi-4-mini
  as reference points. Numbers only, no philosophy.

### A0.3. 24 h runtime polygon (reframed 2026-07-25 — was "sustained CPU% idle")
- Refocus: H1 retracted to `docs/policies.md` § CPU / thermal, so this is
  no longer "does idle CPU stay under X%". It is a **polygon test of the
  runtime as a whole** on the target host, measuring the three axes that
  actually matter for whether the seedling is operating as intended:
  - (a) **Store health.** Per-zone `db_bytes` trajectory over 24 h vs.
    the `RetentionConfig` caps (200 MB input_events / 500 MB system_obs /
    100 MB session_scratch / unbounded personal_vault). Retention sweep
    fires on schedule, `retention_stats` events show `bytes_after` under
    cap, no zone drifts.
  - (b) **Model coping.** In-process rwkv-cpp handles the drip cadence
    the calibration protocol derived. Latency/throughput of live bursts
    matches the pilot `tokens_per_cpu_second` fallback or the measured
    per-machine value; no thermal throttling events; fan-off invariant
    holds outside the interactive regime.
  - (c) **Token regulation.** The drip rate the runtime picks — from
    `docs/policies.md` calibration protocol — actually gets served, and
    utility-path bursts finish inside their budget without starving the
    background stream.
- Trigger: Phase B skeleton is running
  (`verdicts/2026-07-22-phase-b-skeleton.md` — 6 collectors + retention +
  HTTP heartbeat validated on the earlier Ollama backend); revalidate the
  loop on the 2026-07-25 in-process rwkv-cpp backend before the 24 h run.
  Runs on the i5-1235U laptop under a realistic wake pattern — idle
  mostly, periodic ingest ticks, one daily-digest-style burst.
- Deliverables: `experiments/A0_idle/results.md` with per-zone
  `db_bytes` timeseries (from `retention_stats`), rwkv-cpp burst
  latency/tok-s histogram, coretemp trace vs. `fan_safe_cpu_percent`,
  and a pass/fail per axis. No H1 verdict — see policies.md.
- Cost: 24 h wall, ~zero attention. Blocks nothing on the hypothesis
  ledger; blocks confidence in the seedling as a substrate for the
  state-work workstream (lens persistence, H17, H18).

### A0.4. State-utilisation probe (weeks 2–3)
- Instrument RWKV-7 hidden WKV state during autoregressive generation
  (HF `transformers` + torch hooks, native bf16 weights).
- Metrics: delta-norm, trajectory curvature, stable rank (matching
  paper Appendix J). See `docs/state-and-reasoning.md`.
- Tests H8 (state-as-computation) and H9 (G1 amplifies state
  utilisation). Result decides A1 loss formulation — SFT-only vs
  state-regularised.
- Blocks A1. Foundation + skeleton landed with the same commit as
  A0.1; execution deferred to a dedicated session.

### A0.4b. Rich WKV probe + TransformerLens path (2026-08-13)
- **`experiments/A0_state_probe/rich_probe.py`** — per-head spectral analysis
  + layer trajectory + gradient saliency (R-lens proxy). CPU-compatible
  (BlinkDL rwkv package, no triton). New metrics vs old mean+std pooling:
  - `sigma1[l,h]`: per-head top singular value (H8 metric, per-head resolution)
  - `stable_rank[l,h]`: multi-slot capacity per head
  - `sv_entropy[l,h]`: head specialisation signal
  - `layer_trajectory[l]`: relative state change across layers
  - `gradient_saliency[t]`: per-token relevance (R-lens proxy via autograd)
- **TransformerLens + RWKV7**: installed (`fla` v0.5.2, TL RWKV7ArchitectureAdapter).
  Blocked on CPU: `fla` requires `triton` (CUDA). Full TL path deferred to
  GPU session (Selectel). CPU path: `rich_probe.py` above.
- **COMPLETE (2026-08-14):**
  - J-lens G1h base vs step9b-e1 → `results/jlens_g1h_vs_step9b_e1.json` (L4 σ₁ +1.66, SR −0.09)
  - R-lens G1h base vs step9b-e1 → `results/rlens_g1h_vs_step9b_e1.json` (small SR decrease, L31 reasoning σ₁ −4.9)
  - Word-search baseline G1h + G1i → 0/56 both (see A1.5 below)

### A0.5. Causal intervention grid (COMPLETE — 2026-08-04)
- Extended runner in `experiments/A0_state_probe/a05_run.py` +
  aggregator in `a05_analyze.py`. Grid: 2×2 (World-0.4B, G1d-0.4B) ×
  (medium, narrative). Corruption family: gauss(σ), scale, zero_layer,
  zero_head, shuffle_heads, freeze_prev, cross_prompt.
- Verdict: H8 and H9 SUPPORTED. σ-slopes 1.56–2.10 across all 4 cells;
  cross-prompt ratio 21–99×; monotone response confirmed.
- Decision: A1 activates state-reg loss with α > 0.

### A0.6. Intra-model state swap (COMPLETE — 2026-07-22)
- Take state after reasoning-prompt processing, transplant as initial
  state for narrative-prompt decode (and vice versa). Same model, same
  weights.
- Design: 3 reasoning × 3 narrative × 2 directions = 18 pairs; two
  modes — full-state swap vs layer-selective (only middle layers,
  informed by A0.5 zero_layer results).
- Metrics: continuation-content drift (task-lexicon hit rate), style
  drift (perplexity under reasoning-LM vs narrative-LM), top-k
  next-token overlap. Baseline sanity: state_A→A and state_B→B.
- Cost: ~3–4h wall on i5. Answers "is state a portable computational
  mode within a single model, or only fresh working memory?"
- **Infrastructure note (2026-07-25).** State save/load/clone
  primitives from rwkv-cpp are now the mandatory backend
  (`docs/policies.md` § Model backend). State-work is first-class
  (see project memory `project_noesis_state_work_first_class`), so
  A0.6/A0.7 are no longer blocked on "if state manipulation becomes
  supported someday" — the APIs exist and lens-persistence work is
  wiring them. Verdict-blocking on A0.5 remains (need to know state
  is doing work before swapping it), but tooling readiness is not
  the bottleneck any more.

### A0.7. Inter-checkpoint state portability (COMPLETE tier-1 — 2026-07-22)
- **Tier 1 (in A0.7)**: same-arch, same-size, different training —
  RWKV-7-World-1.5B → RWKV-7-G1h-1.5B. State shape identical, direct
  swap. Answers "does state survive a fine-tune of the weights?" —
  design-critical for noesis's continual-LoRA operating model.
- **Tier 2 (deferred to Phase 2)**: same family, different size —
  requires a learned projector (MLP or SVD-shared subspace) between
  state manifolds. Non-trivial ML work.
- **Tier 3 (deferred to Phase 2, feeds C3 and memory-track design)**:
  cross-architecture transfer via *text bottleneck* — state → compressed
  natural-language summary → re-prompt target. This is potentially the
  actual inter-model transfer protocol referenced in C3; also relevant
  to memory-track (state ↔ persistable representation).

### A1. Logic-only fine-tune (weeks 3–8) — *current focus*
- **Corpus**: reasoning traces only. **No RFCs, no CLI docs, no personal
  data, no *domain* knowledge in weights.** Domain knowledge is deferred
  — general knowledge that noesis needs at runtime enters through the
  context window (retrieval, tool observations), not through fine-tune.
  See H2 and H7 in HYPOTHESES.md.
- **Sources** (open only): DeepSeek-R1 distill traces, publicly available
  Anthropic process-supervision / Constitutional-AI methodology material,
  synthetic step-by-step derivations, competition-math CoT, code reasoning
  traces from open datasets.
- **Reasoning supervision**: apply supervision on reasoning steps in the
  training data. The concrete markup (thinking-block delimiters, tool-
  call syntax, step separators) is *not* locked here — decide during
  A1 based on what the base G1 already expects and what the eval set
  best discriminates.
- **Method**: QLoRA on cloud VM (RTX 4090). Local machine is CPU-only.
- **Success criterion**: measurable improvement on the A0 held-out eval
  without regression on general-capability probes.
- **Corpus — evolution.** Variant C primary was glaive-v2 (Steps 1–5);
  superseded 2026-08-06 after structural mismatch analysis (see `FAILED.md
  §2026-08-06`). Step 6+: action chains (Claude cache, reflexive multi-step),
  RFC QA (binary-protocol), ToolBench ReAct, hh-rlhf, selfcot. DeepSeek-R1
  traces and Variant A scope superseded. Corpus policy: `docs/training-data-shortlist.md`.
- **Step results** — see `docs/verdicts/` for full writeups.
  - Step 1 (glaive-v2, G1d-0.4B): PASS. CE 1.70→0.02, +33.3 pp scheduling.
  - Step 3 (α=1e-4, G1d-0.4B): state_reg invisible (ratio 1:50 vs CE). Lower bound.
  - Step 4 (α=1e-3, G1d-0.4B): active regime confirmed. Interrupted at step 3500
    (45% epoch). Eval invalid (tool_use wrapper bug — fixed 2026-08-05).
  - Step 5 (G1d-0.4B, full epoch): **FAILED**. 3/48 = 6.25% — regression vs base.
    Root cause: glaive-v2 is reactive (1–2 tool calls/session); RWKV WKV state
    not exercised. See `FAILED.md §2026-08-06`.
  - Step 6 (G1h-2.9B, action chains + ToolBench, ctx=32768): corpus switch.
    Config: `training/config/pilot_step6_action_chains.yaml`.
  - Step 7 (G1h-2.9B, action chains only): 4/48 = 8.3%.
  - Step 8 (G1h-2.9B, DSL format + L_state span-mask, glaive-v2 DSL corpus):
    CE 4.41→0.55; state_loss saturated at cap (−10.0) by step 100. A0 not run
    directly; H10 sweep as proxy. Config: `training/config/pilot_step8_dsl.yaml`.
  - Step 9 (G1h-2.9B, RFC QA + ε-mask + H12b.i, r=32): **43.75% (21/48) —
    first checkpoint to exceed all baselines** (Gemma3-4B 41.7%, G1h-base 39.6%, Qwen 37.5%).
    ε-mask restored extraction (0%→62.5%). Config: `training/config/pilot_step9.yaml`.
  - Step 9b (G1h-2.9B, 6-source mixed, ctx=512): 39.58% — regression.
    ctx_len=512 short-circuited L_state (T<3); extraction collapsed 62.5%→12.5%.
    Config: `training/config/pilot_step9b.yaml`.
    step9b-e1 uploaded to HF: `Vaniello/noesis-rwkv7-g1h-2.9b` (836 downloads 2026-08-14).
- **Checkpoint status (2026-08-14).** All local checkpoints LOST (disk corruption). HF is canonical source.
  Best available: step9b-e1 (39.6%). Best ever seen locally: step9 e0 (43.75%) — irrecoverable.
- **A0.2 eval results (2026-08-14):** G1i chatwrap 41.7% (20/48, extraction 87.5%, symbolic 75%);
  step9b-e1 39.6% (arithmetic 100%, symbolic 87.5%, scheduling 50%); Gemma4 e4b 37.5% (symbolic 37.5%, scheduling 0%).
  RWKV step9b-e1 wins symbolic by 2.3×. КПД baseline: step9b-e1=0.15, G1i=0.27, Gemma4=7.26 (see H24).
- H7 falsifier (retrieval-parity contrast) remains open.
- **H10 sweep (step8 epoch0, 2026-08-07).** Peak: N=2 silent 33.3% (16/48).
  N=3 collapses to 6.3% — third pass corrupts state. Prior state_readout data
  invalid (bug: shared greedy path with prompt_cot — fixed 2026-08-13;
  now injects `</think>` trigger token before decode). Re-run pending on G1i.
  Full table: `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md`.
- **Base model migration.** Step 9 base was G1h-2.9B; G1i-2.9B
  (ctx16384, 2026-08-05) is the new baseline. All step 10+ runs use G1i.
  G1i eval DONE (2026-08-14): 41.7% chatwrap. Local checkpoints lost; HF is canonical.

### A1.5. RL track — matrix task curriculum

**Paradigm shift from A1.** Steps 1–9b were SFT: CE + L_state supervision,
corpus-driven. A1.5 is reward-only GRPO — no CE loss, no corpus labels,
binary verifiable reward. The step numbering stops at step 9b; A1.5 is a
separate track.

See `docs/rl-track.md` for the full design.

- **Algorithm.** GRPO (`n_samples=8`, T=0.7 rollout, greedy eval).
- **Task family.** Unified matrix task curriculum — 5 types in one generator:
  wordsearch (30%), crossword (10%), arithmetic (20%), pattern (20%), bits (20%).
  Generator: `experiments/A0_eval/gen_tasks.py` → `training/corpus_open/matrix_tasks.jsonl`
  (65 899 tasks, ~20M tokens). Replaces word-search-only framing and bit-decoding.
  Wordsearch curriculum: 7 levels × mixed orientations (see `docs/rl-track.md`).
- **Colab notebook.** `experiments/rl/noesis_rl_colab.ipynb` — Drive mount,
  auto-detect VRAM (T4/A100), gen_tasks on-the-fly, checkpoint to Drive.
- **Hypothesis connections.** H8 (state-as-world-model): matrix tasks require
  integrating spatial context across many positions — WKV state must hold
  directional and structural information. H10 (readout frontier): multi-pass
  sweeps are a direct handle on state convergence. H16 (gated externalisation):
  RL on think-phase trains latent use before gate head (A4). H19 (weight-knowledge
  contamination): L6–L7 wordsearch accuracy is a pure context-dependency proxy.
- **State metrics and new aux losses** — see `docs/rl-track.md §State metrics`.
  IPC DONE (2026-08-16): linear IPC ≈ 0 held-out → state encodes nonlinearly (H8 confirmed).
  L_mem skipped permanently. MLP probe after RL checkpoint 1 replaces linear IPC.
- **Implementation details** (in `docs/rl-track.md`):
  binary ε-mask outside answer span (ε=0.05); entropy routing reward
  shaping (α·Δentropy_reduction in think span); Switch-GRPO curriculum
  (arXiv 2606.13106) for multi-task stability.
- **Blockers.** GPU only. G1i download DONE; G1i eval DONE (41.7%);
  word-search baseline CONFIRMED 0/56. IPC DONE (linear≈0, H8 confirmed).
  Next: Selectel 4090 (~₽1500/24h) or Colab A100.

### A2. Memory-policy tuning (after A1 and Track B2)
- Reproduce Memory-R1 (Yan et al., ACL 2026) approach: RL-trained Memory
  Manager with ADD/UPDATE/DELETE/NOOP operations + Answer Agent with
  memory distillation.
- Uses the external memory system from Track B, so cannot start until B2
  is usable.

### A3. Domain knowledge integration (deferred, no earlier than Gate 2)
- Candidate domains: RFC corpus (~9500 RFCs from rfc-editor.org), CLI
  tooling docs (man pages, tldr, `--help` dumps, top CLI tool docs),
  user-relevant technical documentation.
- **Open decision, deliberately unlocked**: does domain knowledge enter
  through fine-tune, through retrieval into the context window, or
  through a hybrid? H7 in HYPOTHESES.md is the specific claim to test
  before committing to a strategy here.
- Only starts once A1 shows the reasoning-first strategy actually works.

### A4. Truth-system detector heads (sequenced after A1)
- Purpose: convert H19 / H20 / H21 / H22 from *probes* into shipped
  *detectors*. Current pilots read state through hand-labelled item
  sets with no dedicated training; A4 trains small heads on top of
  the frozen A1 backbone, one per signal, on targeted corpora.
- **Corpora (to build during A4 prep):**
  - **H19 provenance:** ~5k triples of `(question,
    no-context-answer, with-retrieval-answer)`, supervision on the
    source-of-truth signal (weights vs context).
  - **H20 aporia:** ~5k `(ambiguous, unambiguous-control)` pairs.
    Rebuilt after n=100 refutation of the `cf > ba > ui` monotone
    ordering (see `experiments/aporia_probe/report.md`); target is
    a state-variance signature, not collapse fraction.
  - **H21 structural:** grow the 280-item TruthfulQA seed to
    ~2-5k with balanced category / impossible / counterfactual and
    an adversarial slice.
  - **H22 attribution:** grow the 240-item C4 seed to ~2-5k of
    hedge-vs-concrete-source pairs.
- **Compute.** Cheap per head (frozen backbone; ~1-2 GPU-h at 0.4B
  each). Total A4 compute budget ~8-15 GPU-h across all four heads.
  Same spot-4090 economics as the A1 micro-pilot.
- **Product surface link.** Directly upgrades SaaS §2 "truth-system"
  claim from "four probes suggest these signals exist" to "four
  trained detectors ship in the runtime".
- **Blockers:** A1 backbone must be locked (heads read A1 state,
  not un-tuned state); redesigned H20 probe design needed before
  H20 head training starts.

## Track B — Memory (external system)

### B0. Schema draft (weeks 1–3, parallel with A0)
- Layered memory: working / episodic / semantic / skill-embedding.
- Storage: SQLite (mirrors key-daemon's choice) + vector store (repurpose
  local-search's HNSW).
- Event log: keystroke / window / file / git events, from collectors
  originally built for key-daemon.

### B1. Retrieval baseline (weeks 3–6)
- BM25 + vector merge (RRF), following local-search's proven pattern.
- Query interface exposed to the agent loop.

### B2. Memory operations (weeks 6–10)
- Structured ADD / UPDATE / DELETE with rationale.
- Enables A2 to start.

## Track C — Integration (runtime)

### C0. Runtime + CLI wiring *(landed 2026-08-13)*

- **noesis-runtime** runs as a systemd user service (port 11435). Embeds
  rwkv-cpp in-process; heartbeat disabled by default so the 2.9B substrate
  idles at ~3% CPU and activates only on HTTP request.
- **HTTP shim** (`noesis-http`) exposes:
  - Ollama wire format: `HEAD /`, `/api/version`, `/api/tags`, `/api/ps`,
    `/api/show`, `/api/pull` (stub), `/api/generate`, `/api/chat`.
  - Anthropic Messages API: `POST /v1/messages` (SSE + buffered).
  - OpenAI chat completions: `POST /v1/chat/completions`.
- **Client launchers** (`noesis launch <client>`):
  - `claude` — sets `ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY=noesis`, execs `claude`.
  - `codex` — sets `OPENAI_BASE_URL + OPENAI_API_KEY=noesis`, execs `codex`.
  - `ollama` — sets `OLLAMA_HOST`, execs `ollama run <discovered-model>`.
  - API key overrideable via `NOESIS_API_KEY` env var.
- **Open WebUI** connects via "Ollama External" at `http://127.0.0.1:11435`.
- **Context transform** (plan §10): `tail_turns=20`, retrieval slot, preamble.
  Stop sequences: `<|im_end|>`, `<|endoftext|>`, `<|im_start|>` (prevents
  role-loop at generation start).
- **Lens persistence** (`/lens/*`): per-session WKV snapshots — enabled when
  `lens_root` is set in `runtime.toml`. Used by Open WebUI sessions to avoid
  re-priming the full history on each turn.
- **Remaining gap.** True session-level state carry-over in `/api/chat` for
  multi-turn Open WebUI conversations — currently re-builds the full ChatML
  prompt each turn. Lens-based optimisation deferred.

### C1. Event-stream ingestion (weeks 4–8)
- Absorb key-daemon collectors (libevdev, Hyprland IPC) as noesis modules
  rather than standalone daemons.
- Absorb local-search extractors (pdftotext, pandoc, taglib, yt-dlp).
- Single-process footprint; measure vs the current standalone daemons.

### C2. Daily-summary MVP (weeks 6–10)
- Agent loop consumes the event stream every N minutes and writes a daily
  summary into Obsidian.
- First real feedback signal for whether A1 fine-tune is actually helping.

### C3. Escalation UX (weeks 10+ — infra unblocked 2026-07-25)
- Human-triggered remote Claude call. When invoked, Claude reads noesis's
  background-agent state, aligns with it, then continues the task.
- This is where the inter-model transfer protocol becomes practically
  required, not just theoretical.
- **Infrastructure note (2026-07-25).** With state-work first-class
  and rwkv-cpp state APIs in the runtime, the lens/scratch DSL that
  H11 relies on can be generated *from actual state* (via H10's
  `state_readout` mode or an equivalent readout head), not synthesised
  from a message-history summary. C3 wiring is a runtime protocol
  problem now, not a "waiting for the backend to expose state" problem.

## Milestone gates

- **Gate 1** (end of week ~4). A0 done, B0 drafted, C0 wired. If A0
  baseline numbers make the RWKV bet untenable, re-evaluate *before*
  investing in A1 training compute. Also decides A1 loss objective
  via A0.4 (SFT-only vs state-regularised). H8 refutation triggers a
  backbone re-open only after the staged flow in HYPOTHESES.md §H8
  (first failure → verify metric implementation and hooks → repeat →
  sustained failure across independent replications). A single null
  A0.4 run is not sufficient.
- **Gate 2** (end of week ~10). A1 fine-tune completed and evaluated;
  C2 daily-summary MVP running in background for 7 consecutive days.
  Assess whether constant-background operation is real or aspirational.
- **Gate 3** (~month 3). B2 + A2 combined; noesis has learned memory
  policy on the user's real data. Assess whether the memory hypothesis
  holds.

## Open questions

- **Cloud training budget.** Local-only or cloud burst for
  continued pretraining? Decide before A3. Two distinct scales:
  - **Micro-pilot (H7 falsifier).** QLoRA on G1d-0.4B, rank 16-32,
    ~30-50k rollouts drawn from the Variant C corpus mix in
    `docs/training-data-shortlist.md` (public agent / function-
    calling traces + adaptable open reasoning traces restructured
    as tool-shaped steps), 1-2 epochs — enough signal to validate
    or refute H7 (understanding-in-weights, knowledge-in-context).
    Fits a single-consumer-GPU spot rental: 4090 spot (Vast/RunPod)
    ~$0.35-0.5/h × 10-14 h ≈ **$5-10** wall-cost. A40 spot ~$6-9
    as an alternative. This is a personally-payable experiment,
    not a compute campaign, and is the minimum-cost path to
    converting the SaaS §2 "correct answers without prior
    knowledge" narrative from wager to measurement.
  - **Full-scale A1 / H12b campaign.** Full Variant C corpus,
    multiple epochs, ablations, plus H12b LoRA + H12b.i utilisation
    regularizer (K=4 WKV slots with slot-usage entropy + cross-slot
    dissimilarity losses, <24 GPU-hours at 0.4B) if the H12a v2
    verdict lands as width-bottleneck. Cloud VM (RTX 4090) is the training platform; local machine is CPU-only.
    Only worth spending once the pilot confirms the direction holds.
  Prerequisite for either: an eval-suite that gives the H7-relevant
  metrics (mixed reasoning + current-facts tasks; retrieval-parity
  contrast) prepared and CPU-baselined on the un-tuned G1d-0.4B
  *before* GPU time is booked, so the fine-tune → eval loop closes
  in one rental window rather than needing a second.
- **Model size.** Start with 2.9B for fast iteration; revisit 13.3B after
  Gate 1.
- **Escalation semantics.** When user invokes Claude, does it replace
  noesis for the current task or run alongside? To be defined at C3.
- **Universal state representation** (Phase 2 research question). If
  A0.7-tier-1 shows state does *not* survive fine-tune of the weights,
  the persistent-runtime story requires either (a) a learned projector
  between checkpoint state spaces, or (b) a text-bottleneck protocol
  (state ↔ natural-language summary ↔ re-prompt). Which one — or both —
  becomes the actual inter-model transfer mechanism informs both the
  memory-track schema and the C3 escalation UX.
- **Multimodal substrate (Phase 3+).** H13a (state absorbs visual
  patches through the same delta-rule update), H13b (image-in-context
  beats text-digest for screen tasks) and H16 (gated externalisation
  from a continuous silent think-stream so the model self-initiates
  rather than being polled) are logged in HYPOTHESES.md. All three are
  Phase 3+; H13b is the cheap near-term probe if noesis needs
  screen-content assistance before Phase 3 lands.
