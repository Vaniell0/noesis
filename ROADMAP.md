# Roadmap

noesis develops along three tracks: the model (A), the external memory (B) and the runtime (C). They
meet at the gates at the end. Track A is the research programme of `hypotheses/README.md`; its claims,
estimators and refuters live there, and this file says only what is being built, in what order, and
what decides each step. Rewritten 2026-11-11; the record of how each step went is in `docs/rl-track.md`,
`docs/verdicts/` and the hypothesis files.

## Track A — the model

### Where it stands

The path is the stage chain of `hypotheses/HZLK.md` §6: teach a model to answer through ticks of its own
latent, then to choose how many ticks it takes, then to read the exact history beside its state. What has
been measured so far:

- **The substrate.** The state carries computation within one pass (H8, by a substitute measurement), is
  wider than one read uses (13-16 live directions of 64 per head on G1i), and keeps 1-2 separable facts
  per head (`H12.md`, `H25.md`). With its controls frozen a tick must settle to a rank-1 point at the rate
  at which it forgets (`docs/state-and-reasoning.md` §3).
- **Trained ticks.** S1 on G1d 0.4B and on Qwen3.5 0.8B: the tick stops erasing the answer, and every
  trained checkpoint is a pass-through, solving at `M = 0` what it solves with ticks (`H10.md` §3). The
  task families meant to need ticks have no teacher yet.
- **Training settings.** The 2.9B fine-tuning collapse was a learning rate; step size is the lever
  (`H26.md` §3).

### A0. Baseline and the substrate — done

- **Inference on the target laptop** (2026-07-23): World 0.4B Q8_0 through rwkv.cpp on the i5-1235U, four
  threads, about 30 tok/s steady, load 506 ms, RSS about 1.16 GB. 2.9B numbers come from cloud VMs.
- **Eval set** of 48 tasks from the owner's workflow (`experiments/A0_eval/`); best single-pass baseline
  G1i 2.9B with chat wrapping, 41.7% (20/48).
- **State probe and causal grid** (A0.4, A0.5; `experiments/A0_state_probe/`): noise injected into the
  state moves the output with slopes 1.56-2.10 on World 0.4B and G1d 0.4B, cross-prompt ratio 21-99×.
  Recorded as H8/H9 support; the registered three-statistic test is not what was run (frame §4).
- **State swap within a model** (A0.6, 2026-07-22, World 0.4B and G1d 0.4B,
  `experiments/A0_portability/results_a06_summary.md`): 0 of 3 prompt pairs on either model; a transplanted
  state does not steer the continuation. **Across checkpoints** (A0.7, G1d 0.4B ↔ World 0.4B,
  `results_a07_summary.md`): 0 of 6. The state is not a portable mode; the same-family projector (tier 2)
  and the text bottleneck (tier 3) were not built.
- **Per-head instruments** (A0.4b and later): spectra, readout lens validated against ablation, capacity
  probe (`H25.md` §5).
- **24-hour runtime run** (A0.3): store health, model coping, token regulation on the laptop under a
  realistic wake pattern. Not run; it belongs to Track C's gate.

### A1, A1.5. Fine-tuning by corpus, then by reward — closed

- **A1** (July-August 2026): LoRA with the trajectory loss `L_state`, corpus changed per step. Best: step 9
  (RFC QA with an ε-mask, r = 32) 43.75% (21/48), lost with the local disk; step 9b-e1 39.6% is on HF
  (`Vaniello/noesis-rwkv7-g1h-2.9b`). Record: `docs/training-pipeline.md`, `docs/verdicts/`.
- **A1.5** (from 2026-08-16): GRPO on a matrix-task curriculum with a self-feed loop and a step-count price.
  Paused 2026-08-19: the loop's steps were never task content. ThinkChain (learned phase markers, LoRA)
  followed; replaced 2026-10-22 by the stage chain below. Record: `docs/rl-track.md`.

### A5. The latent loop: stages S0-S5 — current

| stage | what is trained | gate (what must hold before the next) | state |
|---|---|---|---|
| S0 | data: the model's own answers on ciphers and tables, content and form scored apart | every family has items the model sometimes solves | M=0 passes on G1i/G1k/G1l 2.9B and the 0.4B; ciphers 0 at M=0 on every model, so no teacher |
| S1 | the answer through `M` ticks of the model's own latent, KL to its own `M = 0` answers | the tick stops erasing the answer; the same checkpoint at `M = 0` does not solve the new items (else pass-through); ticks decode to prompt-specific content; everyday questions stay fluent | 0.4B and Qwen3.5 0.8B: pass-through |
| S2 | the same loop as a policy: noise in a small latent subspace plus a verified answer or a continuation-gain reward | the reward window is open: rollouts disagree at accuracy near `M = 0` | not run; S1 made the ticks damp noise |
| S3 | the model chooses `M`: an unreadable latent tick and a vocabulary token in one softmax | the attractor moves later with distinct ticks; a learned stop reaches the best fixed `M` with fewer ticks | toy only (stopping) |
| S4 | format and honesty | format obedience per format; calibration | obedience measured, near 0 |
| S5 | a QKV-ROSA branch queried at every tick, exact history beside the settling state | a fact from turn 1 recalled after `M` ticks as well as at `M = 0`; no common attractor across turns | not started |

What a stage must show, beside its gate: on knowledge items, the fact classes move from "stored, not read"
to "read" and not from "read" to "lost" (`H7.md`); the ticks beat an arm trained at `M = 0` on the same
data (`H10.md`); where an answer comes from stays readable (`H19.md`).

Models on disk: G1d 0.4B and World 0.4B (shared trained weights, readable matrix by matrix), G1i, G1k, G1l
2.9B, G1L 1.5B, Qwen3.5 0.8B. The chain starts from World 0.4B, a plain pretrain: the loop is learned from
early on rather than grafted onto a finished model, and World trained by this chain against G1d 0.4B (the
same kind of base after the G1 training) compares the two training methods from one base (`HZLK.md` §6). The aim for the small
model: bring it toward what G1L 2.9B reads, by repair and reorganisation rather than by adding facts.
Qwen is a teacher and a read-path testbed only: its head is tied to its embeddings, so its ticks are
words, not a latent.

### A2. Memory-policy tuning (after B2)

An RL-trained memory manager (ADD / UPDATE / DELETE / NOOP) with an answer agent and memory distillation,
after Memory-R1 (Yan et al., ACL 2026). Needs Track B's operations.

### A3. Domain knowledge

Whether domain knowledge (RFCs, CLI documentation, the owner's technical material) enters through weights,
through retrieval, or both is decided by `H7.md`: a stage that teaches reading from context should raise
context items without losing read ones. Not started.

### A4. Detector heads for the provenance group

Turn H19-H22 from probes into trained heads on a frozen backbone, one per signal. Item sets: H19 by its
three conditions (with the passage, without, with an unrelated one); H20 pairs of ambiguous items and
controls, read as a state signature, not as first-token collapse; H21 grown from the 280-item seed with
structural categories and an adversarial slice; H22 rebuilt so the classes differ in where the fact lives,
not in wording. Every head is compared with a bag-of-words control. Needs a backbone from the stage chain.

## Track B — Memory (external system)

### B0. Schema draft

- Layered memory: working / episodic / semantic / skill-embedding.
- Storage: SQLite (mirrors key-daemon's choice) + vector store (repurpose local-search's HNSW).
- Event log: keystroke / window / file / git events, from collectors originally built for key-daemon.

### B1. Retrieval baseline

- BM25 + vector merge (RRF), following local-search's proven pattern.
- Query interface exposed to the agent loop.

### B2. Memory operations

- Structured ADD / UPDATE / DELETE with rationale.
- Enables A2.

## Track C — Integration (runtime)

### C0. Runtime + CLI wiring *(landed 2026-08-13)*

- **noesis-runtime** runs as a systemd user service (port 11435). Embeds rwkv-cpp in-process (Ollama
  removed as a backend 2026-07-25: it does not expose `rwkv_get_state_len`, `rwkv_eval(..., state_out)`,
  `rwkv_clone_state`); heartbeat disabled by default so the 2.9B substrate idles at about 3% CPU and
  activates only on request.
- **HTTP shim** (`noesis-http`) exposes:
  - Ollama wire format: `HEAD /`, `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/pull` (stub),
    `/api/generate`, `/api/chat`.
  - Anthropic Messages API: `POST /v1/messages` (SSE + buffered).
  - OpenAI chat completions: `POST /v1/chat/completions`.
- **Client launchers** (`noesis launch <client>`): `claude`, `codex`, `ollama`; API key overridable via
  `NOESIS_API_KEY`.
- **Open WebUI** connects via "Ollama External" at `http://127.0.0.1:11435`.
- **Context transform**: `tail_turns=20`, retrieval slot, preamble. Stop sequences `<|im_end|>`,
  `<|endoftext|>`, `<|im_start|>`.
- **Lens persistence** (`/lens/*`): per-session WKV snapshots when `lens_root` is set in `runtime.toml`.
- **Remaining gap.** Session-level state carry-over in `/api/chat` for multi-turn conversations; each turn
  rebuilds the full prompt. Whether a persistent state can replace re-injected history is `H17.md`.

### C1. Event-stream ingestion

- Absorb key-daemon collectors (libevdev, Hyprland IPC) as noesis modules rather than standalone daemons.
- Absorb local-search extractors (pdftotext, pandoc, taglib, yt-dlp).
- Single-process footprint; measure against the current standalone daemons.

### C2. Daily-summary MVP

- An agent loop consumes the event stream every N minutes and writes a daily summary into Obsidian.
- The first real-use signal for whether a trained checkpoint helps.

### C3. Escalation UX

- A human-triggered remote Claude call reads noesis's background state, aligns with it, then continues the
  task. With the rwkv-cpp state APIs in the runtime, the handoff can be generated from the actual state
  (a readout of it), not from a message-history summary. This is a runtime protocol problem now.

## Gates

- **Gate 1 — passed.** A0 done, B0 drafted, C0 wired; the RWKV wager was not made untenable by the
  baseline numbers.
- **Gate 2 — the loop.** A trained checkpoint passes the HZLK refuter: `M ≥ 2` beats the same model at
  `M = 0` and `M = 1` on held-out items of at least two task families (paired, gain of at least 5 points,
  McNemar `p < 0.05`), against an arm trained at `M = 0`. Beside it, C2 running in the background for seven
  consecutive days.
- **Gate 3 — memory.** B2 and A2 combined: noesis has learned a memory policy on the owner's real data.

## Open questions

- **Compute.** The laptop runs inference and 0.4B training; 2.9B training needs a rented GPU (A30 or
  similar), decided per run. Which rung the next distillation runs on, and on which model, is open.
- **Model size.** The working sizes are 0.4B (repair) and 2.9B (teacher and target); 13.3B only if a
  result at 2.9B asks for it.
- **Escalation semantics.** When the owner invokes Claude, does it replace noesis for the task or run
  alongside? To be defined at C3.
- **Carrying state between checkpoints.** A0.7 found no direct transfer; a learned projector or a
  text-bottleneck protocol would be needed for the persistent runtime to survive a weight update.
- **Multimodal substrate.** H13a (the state absorbs visual patches through the same update) and H13b
  (image in context beats a text digest for screen tasks) are runtime wagers outside the current chain.
