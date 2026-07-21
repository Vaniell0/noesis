# Roadmap

noesis develops across three parallel tracks that converge at integration
milestones. Cognitive and memory tracks progress independently — do not
serialize them.

## Track A — Cognitive (model training)

### A0. Baseline (weeks 1–2)
- Install RWKV-7-G1 2.9B via Ollama, verify inference throughput on the
  user's hardware (target: reproduce the observed ~15–20 tok/s CPU and
  ~50+ tok/s on 1050).
- Assemble a held-out eval set of 30–50 real reasoning tasks drawn from
  the user's actual workflow — not GSM8K, not MMLU.
- Baseline eval: RWKV-7-G1 2.9B against Qwen-2.5-3B-Instruct and Phi-4-mini
  as reference points. Numbers only, no philosophy.

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
  continued pretraining? Decide before A3.
- **Model size.** Start with 2.9B for fast iteration; revisit 13.3B after
  Gate 1.
- **Escalation semantics.** When user invokes Claude, does it replace
  noesis for the current task or run alongside? To be defined at C3.
