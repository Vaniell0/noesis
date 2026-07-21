# A0.1 — baseline throughput

## Purpose

First numeric signal for the project: how many tokens per second the
local RWKV-7 2.9B produces under realistic workload shapes, and how the
number degrades when the runner is cold. Blocking input for Gate 1: if
sustained-background operation is not economically viable, spending on
A1 training compute is unjustified.

Hypotheses tested: **H1** (constant-cost background operation, <3 GB
RSS, <10 % avg CPU under 24 h workload) and **H6** (cognitive layer on
modest hardware, <2 s warm-cache latency, no thermal limits).

**H4 is not tested here** — H4 is about quality, not throughput.

## Design

### Three cache-temperature phases

- **cold** — runner was just unloaded (`keep_alive: 0`). First request
  pays weight-load, page-in and KV-cache-init cost. Represents worst-
  case first-response latency for a background agent that hasn't been
  used in a while.
- **warm** — request #2. Runner is up, weights resident, but any
  intra-request caches are empty. Represents the agent waking to
  handle a new event.
- **hot** — request #3+. Steady state. What matters most for a
  background agent — this is the number the daily-summary loop and the
  event-stream digest will actually feel.

Each prompt is run once cold, once warm, then `--hot-repeats` times hot
(default 3). Reported figures are per-phase medians.

### Realistic prompts (see `prompts.py`)

| label | words | shape | why |
|-------|-------|-------|-----|
| `short` | ~40  | interactive follow-up on a debug session | represents CLI Q&A |
| `medium` | ~350 | 15-minute event-stream digest ask | represents C2 daily-summary chunk |
| `long` | ~1400 | retrieval-augmented question about GoodNet | represents C3 handoff prep |

Toy prompts (`What is 2+2?`) are not used — they don't stress prefill and
don't reflect the workload noesis is designed for.

### Metrics per sample

- **decode tok/s** = `eval_count / eval_duration` from Ollama `done` payload.
- **prefill tok/s** = `prompt_eval_count / prompt_eval_duration`.
- **TTFT ms** — time to first streamed token (`response != ""`).
- **wall s** — total request time.
- **RSS MB** — VmRSS of the actual `llama-server` subprocess (not
  `ollama serve`; the parent holds no weights and reports <40 MB).

## Setup

- Laptop: i5-1235U, 32 GB RAM, Intel Iris Xe (**CPU-only inference**).
- Server: GTX 1050 Ti, 4 GB VRAM (address to be discovered on port
  11434 when it comes online).
- Ollama HTTP API: `http://127.0.0.1:11434` locally.
- Models under test:
  - `rwkv7-2.9b:latest` — Q4_K_M base = BlinkDL **RWKV-7-World / Goose3**.
  - `mollysama/rwkv-7-g1h:2.9b` — Q4_K_M of **RWKV-7-G1h**, the reasoning-
    line variant that CLAUDE.md's locked decision refers to. Registry
    source: <https://ollama.com/mollysama>.

## Runs

```bash
# Laptop CPU
python3 bench.py --model rwkv7-2.9b               --out results-laptop-cpu-world.json
python3 bench.py --model mollysama/rwkv-7-g1h:2.9b --out results-laptop-cpu-g1h.json

# Server (once discovered)
python3 bench.py --host http://<server>:11434 \
    --model rwkv7-2.9b               --out results-server-gpu-world.json
python3 bench.py --host http://<server>:11434 \
    --model mollysama/rwkv-7-g1h:2.9b --out results-server-gpu-g1h.json
```

Results and interpretation live in `results.md`; raw JSON in
`results-*.json`.

## Prior work referenced

`~/Desktop/projects/models/bench.py` (2026-07-18) produced numbers for
RWKV-7-World and Qwen/Gemma references on this laptop and on the
1050 Ti server. Those numbers are **not comparable** to this bench:
they collapsed prefill and decode into a single throughput figure, did
not measure TTFT, did not separate cache phases and used toy prompts.
They stay in that directory as historical deployment notes.
