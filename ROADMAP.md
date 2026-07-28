# Roadmap

noesis develops across three parallel tracks that converge at integration
milestones. Cognitive and memory tracks progress independently — do not
serialize them.

## Track A — Cognitive (model training)

### A0. Baseline (weeks 1–2)
- Install RWKV-7-G1 2.9B via Ollama, verify inference throughput on the
  user's hardware. **Measured 2026-07-23**: 0.4B World Q8_0 via rwkv.cpp
  on i5-1235U (Alder Lake, 4 threads) delivers ~30 tok/s steady-state;
  load 506 ms; RSS ~1.16 GB. 2.9B target and GTX 1050 numbers still
  untested.
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
  state-work workstream (`memory: project_noesis_state_work_first_class`).

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

### A0.5. Causal intervention grid (in flight)
- Extended runner in `experiments/A0_state_probe/a05_run.py` +
  aggregator in `a05_analyze.py`. Grid: 2×2 (World-0.4B, G1d-0.4B) ×
  (medium, narrative). Corruption family: gauss(σ), scale, zero_layer,
  zero_head, shuffle_heads, freeze_prev, cross_prompt.
- Verdict feeds H8-causal-A (σ-response superlinearity),
  H8-causal-B (layer localisation), H8-causal-C (cross-prompt ratio).
- Result decides whether A1 activates state-reg loss (α > 0) or ships
  as pure SFT (α = 0). Currently on seed 2 of medium cells;
  narrative cells to follow (~2h remaining wall).

### A0.6. Intra-model state swap (pending A0.5 verdict)
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

### A0.7. Inter-checkpoint state portability (pending A0.5 verdict, tier-1 only in A0)
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
- **Method**: QLoRA on the 1050 for the 2.9B base if VRAM allows; cloud
  burst if not.
- **Success criterion**: measurable improvement on the A0 held-out eval
  without regression on general-capability probes.

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

### C0. Ollama + CLI wiring (weeks 1–2)
- Verify that noesis-as-Ollama-model works cleanly inside the Claude-Code
  CLI workflow (Ollama's OpenAI-compatible endpoint). If the integration
  path is more involved than a config change, spec the shim explicitly.
- No custom harness beyond what integration verification requires.

### C1. Event-stream ingestion (weeks 4–8)
- Absorb key-daemon collectors (libevdev, Hyprland IPC) as noesis modules
  rather than standalone daemons.
- Absorb local-search extractors (pdftotext, pandoc, taglib, yt-dlp).
- Single-process footprint; measure vs the current standalone daemons.

### C2. Daily-summary MVP (weeks 6–10)
- Agent loop consumes the event stream every N minutes and writes a daily
  summary into Obsidian.
- First real feedback signal for whether A1 fine-tune is actually helping.

### C3. Escalation UX (weeks 10+)
- Human-triggered remote Claude call. When invoked, Claude reads noesis's
  background-agent state, aligns with it, then continues the task.
- This is where the inter-model transfer protocol becomes practically
  required, not just theoretical.

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

- **Cloud training budget.** Local-only or cloud burst on A100 for
  continued pretraining? Decide before A3. Also required for H12b LoRA
  + H12b.i utilisation regularizer (K=4 WKV slots with slot-usage
  entropy + cross-slot dissimilarity losses, <24 GPU-hours at 0.4B)
  if the H12a v2 verdict lands as width-bottleneck. Local GTX 1050
  sits at the edge of feasibility; A100 burst (~24 wall-hours × ~$2/h)
  is the realistic path.
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
