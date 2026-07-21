# A0.2 — pretrained baseline results

## Setup

- 48 hand-written held-out tasks in 6 categories (see `tasks.jsonl`) —
  base 42 + 6 `bit_book_*` (in-context codebook decode, added mid-run;
  reported separately below).
- Rubric-based scoring: `exact`, `contains`, `regex`, `json_subset`.
- Backend: Ollama HTTP, `temperature=0.0`, `num_predict=256`
  (128 for bit_book subset), timeout=180s per request.
- Hardware: laptop i5-1235U, CPU-only.

The per-model / per-category tables below refer to the **original 42
tasks**. The `bit_book_*` subset (6 tasks) was run separately on 3 of
the 4 models (qwen skipped for battery budget — see §Constraints).

## Per-model overall

| model | size | overall | wall | notes |
|---|---:|---:|---:|---|
| qwen2.5:1.5b (Transformer) | 1.5 B | **38.1%** | 427 s | reference at target size |
| rwkv7-1.5b:latest (World)  | 1.5 B | **26.2%** | 384 s | RWKV-7-World base, pre-A1 baseline |
| mollysama/rwkv-7-g1h:2.9b  | 2.9 B | **7.1%**  | 1715 s | reasoning-tuned; collapsed on sym/sched/str — see note |
| gemma3:4b (Transformer)    | 4 B | **38.1%** | 1046 s | larger reference; ties qwen 1.5B overall |

## Per-category × per-model

| category | qwen 1.5B | rwkv7-1.5B | g1h-2.9B | gemma3-4B |
|---|---:|---:|---:|---:|
| bit_decoding      | 0/10 (0.0%)   | 0/10 (0.0%)   | 0/10 (0.0%)  | 1/10 (10.0%)  |
| symbolic          | 6/8 (75.0%)   | 5/8 (62.5%)   | 0/8 (0.0%)   | 3/8 (37.5%)   |
| extraction        | 4/8 (50.0%)   | 4/8 (50.0%)   | 3/8 (37.5%)  | 6/8 (75.0%)   |
| scheduling        | 3/6 (50.0%)   | 1/6 (16.7%)   | 0/6 (0.0%)   | 2/6 (33.3%)   |
| string_ops        | 3/6 (50.0%)   | 1/6 (16.7%)   | 0/6 (0.0%)   | 2/6 (33.3%)   |
| arithmetic_chain  | 0/4 (0.0%)    | 0/4 (0.0%)    | 0/4 (0.0%)   | 2/4 (50.0%)   |

## Key observations

### 1. bit_decoding stays zero up to 2.9 B; only gemma3-4B cracks the *easiest* item

Every model ≤ 2.9 B fails all 10 bit_decoding tasks. gemma3-4B passes
exactly one: `bit_dec_01`, which is a **single-step parallel lookup**
(decimal codepoints → ASCII letters), *not* a carry-state task. All 9
multi-step bit_decoding tasks (binary→ASCII multi-letter, hex-decode,
base64, ROT-N, XOR chain, Caesar substitution) still fail at 4 B.

**Implication for A1.** The "accumulating-decoded-prefix working
memory" wall holds at this size class. If A1 with state-reg (α > 0)
moves any of `bit_bin_*`, `bit_hex_*`, `bit_rot_*`, `bit_xor_*`,
`bit_sub_*` from 0 → non-zero *without* gaining size (still 2.9 B G1h
substrate), that isolates state-carry improvement from
raw-capacity/knowledge improvement. This is the cleanest experimental
knob for the state-reg hypothesis.

### 2. arithmetic_chain: same split — parallel/single-step passes at 4 B, chain still zero

gemma3-4B solves `arith_02` (single base conversion 173 → binary) and
`arith_03` (single hex → decimal). It still fails `arith_01`
(5-step parenthesised expression) and `arith_04` (fraction word
problem requiring 2 substitutions). Same pattern as (1): scale unlocks
single-step operations, but state-carry through a chain of
intermediates remains a wall.

Substrate is the same as bit_decoding — state-carried intermediate
results — so the "state-reg lifts chains" hypothesis is testable on
this category too, but the sample is small (n=4).

### 3. mollysama/rwkv-7-g1h:2.9b collapse

**Concerning.** The reasoning-tuned RWKV-7-G1h — the A1 backbone target
— scored 7.1 % vs 26.2 % for the base RWKV-7-World at half the size.
Failed all symbolic (including `3x+7=22`), all scheduling, all string_ops.

Hypotheses:
- Chat template mismatch — g1h expects a specific system-prompt / role
  wrapping that raw `/api/generate` doesn't provide.
- Deterministic decode (temperature=0.0) may get stuck in a
  reasoning-preamble loop for this model.
- Output is verbose "let me think..." style that fails rubric even when
  the answer is present buried in prose.

**To investigate before A1**: sample responses from mollysama on
sym_alg_01 (`3x+7=22`) — if answer is present but wrapped in prose,
rubric is under-scoring; if answer is absent, model has a real regression.

### 4. bit_book_* subset — 0/6 across all 3 tested models (cleanest H8 signal)

`bit_book_*` gives the model an **explicit codebook in the prompt**
(e.g. `a=00, b=01, c=10, d=11`) plus a short bitstring, and asks it to
chunk + look up + concatenate. This removes the memorized-encoding
confound from `bit_bin_*` / `bit_hex_*` / `bit_b64_*`: the only skill
under test is *in-context lookup with state accumulation*.

| model | bit_book overall | qualitative failure mode |
|---|---:|---|
| gemma3:4b        | 0/6 (0.0%) | plausible-shaped output, wrong chunk-to-code assignment; often extra/missing letters (e.g. `bit_book_01` → `cbcd` vs expected `badc`; `bit_book_03` → `r e a d y s` — first four right but hallucinates 2 extra codes) |
| rwkv7-1.5b       | 0/6 (0.0%) | rambles "let me split the string…" preamble, does not commit to a final answer; on `bit_book_06` outputs `c=d=a=b=` (echoes codebook back) |
| mollysama-g1h    | 0/6 (0.0%) | 5/6 responses **empty**; consistent with the g1h collapse observed in (3) |

**Two things this tells us**:

1. The wall really is *state-carry* competence, not missing knowledge —
   the codebook is right there in the prompt and still no model gets it.
2. gemma's `read y s` on `bit_book_03` is the most interesting single
   response: the first 4 codes are correctly resolved, then the model
   fails to *terminate* — a classic RNN-like halt-condition failure
   even from a Transformer. This is what state-reg is supposed to
   sharpen. Watch this task specifically post-A1.

### 5. category ranking is stable — extraction > symbolic > scheduling ≈ string_ops > bit ≈ arith

Every model that scores anything on procedural categories scores
extraction and symbolic first. This is the expected "content
completion" strength of pretraining. **A1's success signature** should
show either:
- lift on **bit_decoding + arithmetic_chain** (state-carried working
  memory improved), or
- lift on **scheduling** (combinatorial search improved),
- without regression on extraction/symbolic (basic competence preserved).

## Files

- Raw per-task JSON: `results/qwen25_15b.json`, `results/rwkv7_15b_world.json`,
  `results/rwkv7_29b_g1h.json`, `results/gemma3_4b.json`.
- `bit_book_*` subset: `results/bit_book_gemma3_4b.json`,
  `results/bit_book_rwkv7_15b_world.json`,
  `results/bit_book_rwkv7_29b_g1h.json` (+ matching `.log`).
- Per-model log: `results/*.log`.

## Constraints imposed on this run

- **Battery-limited session** (laptop unplugged, ≤20% remaining when
  bit_book eval was launched). Consequences:
  - `bit_book_*` was run on only **3 of 4 models** — qwen2.5:1.5b was
    skipped since its 0-6 outcome is highly likely given the other 3
    results and the wall-time budget was better spent progressing
    A0.5 grid.
  - No follow-up re-run at higher `num_predict` for RWKV rambling
    responses (would trade compute for possibly-nothing).
- **A0.5 grid still running in background** during bit_book eval —
  ollama CPU contention accepted since the grid is `.venv/python` on
  its own cores, not through ollama.

## Next

1. ~~Complete gemma3:4b baseline.~~ **Done.** 38.1 % overall, matches
   qwen 1.5B; bit_decoding wall partially breached (1/10, single-step
   only).
2. Investigate mollysama collapse (sample responses on `sym_alg_01`) —
   distinguish rubric under-scoring from real regression before A1
   picks this checkpoint as substrate.
3. Wait for A0.5 verdict → decide on state-reg loss activation for A1
   pilot.
4. Run A1 pilot on `mollysama/rwkv-7-g1h:2.9b` → re-run this eval →
   `compare.py results/rwkv7_29b_g1h.json results/a1_pilot.json`.
   Expected A1-success signal: multi-step bit_decoding and arith_chain
   move from 0 → non-zero at same 2.9 B footprint.
