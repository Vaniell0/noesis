# A0.1 — baseline throughput results

Numbers below are per-phase medians for the hot phase (n=3), single value
for cold and warm (n=1). Raw JSON in `results-{host}-{model}.json`;
full stdout in `bench-{host}-{model}.log`.

Bench harness: `bench.py` (this dir), stdlib-only HTTP client against
Ollama `/api/generate` with `stream=true`. Metrics from Ollama's own
`prompt_eval_*` and `eval_*` fields plus wall-clock TTFT.

> ⚠ **Server GPU numbers are provisional.** The 1050 Ti host was not
> quiesced before the bench — other user processes were live on the
> box during measurement. Individual samples show larger spread than
> the laptop CPU run (see the `long-warm` RSS spike and G1h TTFT jitter
> below). Read the GPU columns as a *smoke test*, not a calibrated
> baseline. A rerun on a clean environment is a follow-up.

## Setup

| host             | hardware                                       | ollama backend |
|------------------|------------------------------------------------|----------------|
| `127.0.0.1`      | i5-1235U, 32 GB RAM, Intel Iris Xe (no CUDA)   | CPU            |
| `10.156.70.198`  | GTX 1050 Ti, 4 GB VRAM (ZeroTier peer)         | CUDA           |

Models (both Q4_K_M, ~1.9 GB on disk):

- `rwkv7-2.9b:latest` — BlinkDL **RWKV-7-World / Goose3** base.
- `mollysama/rwkv-7-g1h:2.9b` — **RWKV-7-G1h**, reasoning-line variant
  (see CLAUDE.md locked decision). Registry:
  <https://ollama.com/mollysama/rwkv-7-g1h>.

Prompts (`prompts.py`): `short` (~40 w / 70 tok), `medium` (~280 w /
~890 tok event-stream digest), `long` (~560 w / ~920 tok
retrieval-augmented question). Whitespace-tokenised word counts differ
from RWKV tokenizer output; the ~1.5× tok/word ratio holds only for
prose — `medium` and `long` inflate on structured / dense text.

## Results

Format per cell: `decode tok/s (TTFT ms, wall s)`.

### RWKV-7-World

| prompt | tok  | phase | laptop CPU              | server GPU              |
|--------|------|-------|-------------------------|-------------------------|
| short  |   69 | cold  | **11.4** (4619, 26.99)  | **19.1** (4203, 17.53)  |
| short  |   69 | warm  | **11.2** ( 224, 23.07)  | **16.3** ( 347, 15.99)  |
| short  |   69 | hot   | **10.9** ( 252, 23.85)  | **16.9** ( 380, 15.53)  |
| medium |  886 | cold  | **11.2** (26316, 39.71) | **16.8** (10905, 26.10) |
| medium |  886 | warm  | **11.6** ( 207,  7.36)  | **16.8** ( 414, 10.65)  |
| medium |  886 | hot   | **11.6** ( 211,  9.43)  | **16.9** ( 395,  8.15)  |
| long   |  921 | cold  | **11.8** (28224, 49.99) | **16.8** (11915, 27.07) |
| long   |  921 | warm  | **11.8** ( 192, 21.97)  | **16.8** ( 391, 12.21)  |
| long   |  921 | hot   | **10.8** ( 227, 24.02)  | **16.7** ( 465, 15.66)  |

### RWKV-7-G1h (reasoning-line)

| prompt | tok  | phase | laptop CPU              | server GPU              |
|--------|------|-------|-------------------------|-------------------------|
| short  |   70 | cold  | **11.6** (5140, 26.45)  | **27.9** (30432, 30.43) |
| short  |   70 | warm  | **11.7** ( 791, 22.09)  | **27.9** (  663,  5.36) |
| short  |   70 | hot   | **11.4** ( 855, 22.74)  | **16.7** (15710, 15.71) |
| medium |  887 | cold  | **11.7** (28603, 41.33) | **16.6** (43286, 43.29) |
| medium |  887 | warm  | **11.0** (  802,  8.90) | **16.6** (  743, 11.63) |
| medium |  887 | hot   | **11.4** (22517, 22.52) | **16.8** (  843, 10.28) |
| long   |  922 | cold  | **11.5** (50242, 50.24) | **26.7** (35297, 35.30) |
| long   |  922 | warm  | **11.4** (22769, 22.77) | **17.3** (15035, 15.04) |
| long   |  922 | hot   | **11.9** (21835, 21.83) | **15.8** (16567, 16.57) |

RSS of the `llama-server` runner stays at **1957–2110 MB** across every
cell and phase — nearly identical for World and G1h (weights dominate,
RWKV state is tiny).

## Quality per second (framework — awaits A0.2)

Tok/s alone is misleading: a fast model that produces wrong or
uninteresting output has zero effective throughput. The number that
actually matters for a background agent is **quality-adjusted
throughput** — how many *useful* outputs per unit time.

Definition used here: `score / latency_s`, where score ∈ [0, 1] is the
per-task grade on the A0.2 held-out eval (not yet built) and
`latency_s` is the median wall time for a single response of the
prompt shape the task uses.

Fill target — one row per (model, host); scores TBD after A0.2:

| model              | host          | score (∈ [0,1]) | latency s (hot medium) | quality / s |
|--------------------|---------------|-----------------|------------------------|-------------|
| RWKV-7-World 2.9B  | laptop CPU    | TBD (A0.2)      |  9.43                  | TBD         |
| RWKV-7-World 2.9B  | server GPU *  | TBD (A0.2)      |  8.15                  | TBD         |
| RWKV-7-G1h 2.9B    | laptop CPU    | TBD (A0.2)      | 22.52                  | TBD         |
| RWKV-7-G1h 2.9B    | server GPU *  | TBD (A0.2)      | 10.28                  | TBD         |

`*` — see provisional-GPU caveat above.

The `latency s` column uses the `medium` prompt hot phase because that
is the closest per-request shape to the target background-agent
workload (event-stream digest). If A0.2 introduces per-task-shape
latency, that column should be split per task.

**Interpretation deferred to A0.2.** This table is scaffolding; do not
read it as a decision input yet. What it makes explicit *now* is that a
G1h win on tok/s (there isn't one — see below) or a raw-World win on
latency (there is one, 2–3×) is not the same as winning on quality/s.
That is the number Gate 1 should be judged on.

## What this says about Gate 1

### H1 — constant-cost background operation (<3 GB RSS)

**Pass on the RSS axis.** Runner RSS steady at ~2.0–2.1 GB after
weights load; no growth across the ~4 min bench. Ollama's `--no-mmap`
means we see the honest resident set, not just page-mapped weights.
The unloaded state costs ~40 MB (parent `ollama serve` only), so the
delta of loading the model is ~2.0 GB — well under the 3 GB budget.

Not tested here: 24 h idle CPU% under a realistic wake pattern. That's
the observability piece and belongs in A0.3, not this bench.

### H6 — cognitive layer on modest hardware

**Warm/hot latency, World, laptop CPU:** TTFT 190–250 ms, decode
~11 tok/s. A 256-token response arrives in 23–24 s cold, 7–22 s hot
depending on prompt. **Interactive follow-up (short prompt) is
sub-second first-token, ~23 s to full answer** — usable for a CLI Q&A
loop, marginal for a chat feel.

**Warm/hot latency, G1h, laptop CPU:** decode is the same
(~11 tok/s), but **TTFT balloons to 0.8–22 s** on hot repeats. This is
the reasoning line — G1h thinks silently before user-visible tokens,
and hot-phase runs sometimes trigger long CoT-style prefaces (see the
21–22 s TTFT rows). For a background agent that produces a daily
digest this is fine; for an interactive prompt this fails the "warm
<2 s" bar unless we cap the thinking budget.

**GPU (1050 Ti) speedup:** decode +50 % (~11 → ~17 tok/s), cold
prefill 3–4× (~35 → ~110 tok/s), cold TTFT ~2× faster on medium/long.
The GPU is old (2016, 4 GB VRAM) and this is close to what you'd
expect — RWKV is memory-bandwidth-bound at decode. A more modern
server-class GPU would move decode into 40+ tok/s, but the 1050 Ti
alone is not the throughput multiplier the roadmap needs.

### What "hot prefill" reports do NOT measure

The 5000–6000 tok/s "hot prefill" numbers are not compute throughput.
Ollama caches the tokenised prompt across identical requests; the
runner sees a zero-cost prefill and reports the full token count over
a near-zero duration. Real prefill cost lives in the **cold** row:
laptop CPU ~35 tok/s, server GPU ~110 tok/s.

For noesis's actual workload (each event-stream digest is a fresh
prompt) we should mostly plan against the **cold prefill** number, not
hot. This is a change from how bench numbers are usually read.

## Notes on measurement fidelity

- **RSS** is `VmRSS` of the largest `llama-server` subprocess, not the
  Ollama parent. When multiple runners are alive (embedding model +
  reasoning model), we pick the one with the largest RSS as a proxy
  for "the model under test". One `long-warm` sample on the server
  briefly showed 331 MB — the runner had just respawned and was still
  loading; treat that row's RSS as noise.
- **TTFT** is the wall time to the first streamed chunk with
  non-empty `response`. For reasoning models this may include hidden
  thinking tokens; the model's honest first-*visible*-token latency
  may be shorter than reported, but as a UX proxy this is what the
  user perceives.
- **Cold** = first request after `POST /api/generate keep_alive=0`
  and confirmed runner death by pgrep. **Warm** = request #2 to the
  same runner. **Hot** = request #3+; median of `--hot-repeats`
  (default 3).

## Follow-ups

- **G1h thinking budget.** Cap `<think>` output via Ollama's
  `options.stop` or a prompt-level instruction; re-measure TTFT to
  see if H6 warm-latency bar is reachable with a tuned system prompt.
  Blocker for using G1h in the interactive path.
- **Server-class GPU.** 1050 Ti is a floor number. Repeat on the
  GPU that will actually host production (TBD) before drawing final
  Gate 1 conclusions on H6.
- **A0.2 — held-out eval.** Throughput says nothing about H4. Separate
  session; needs a task list + rubric first.
- **A0.3 — 24 h idle.** For H1 we still need the sustained-CPU
  measurement, not just RSS. Belongs in the observability slice, not
  here.
