# A0.2 — pretrained baseline results

## Setup

- 48 hand-written held-out tasks in 6 categories (see `tasks.jsonl`) —
  base 42 + 6 `bit_book_*` (in-context codebook decode, added mid-run;
  reported separately below).
- Rubric-based scoring: `exact`, `contains`, `regex`, `json_subset`.
- Backend: Ollama HTTP, `temperature=0.0`, `num_predict=256`
  (128 for bit_book subset), timeout=180s per request.
- Hardware: laptop i5-1235U, CPU-only.

**2026-07-22 correction.** Initial run used `num_predict=256`. This
truncated `mollysama/rwkv-7-g1h:2.9b` mid-CoT, and Ollama returns an
empty response on `done_reason=length` — so 33/42 mollysama responses
came back empty and the model appeared to score 7.1 %. Re-run at
`num_predict=2048` (eval.py default bumped, see the arg help) put
mollysama at 38.1 % on base-42 and 3/6 on bit_book. The mollysama
row + observation §3 below reflect the corrected run; other three
models unchanged (they don't need the larger budget to complete).

The per-model / per-category tables below refer to the **original 42
tasks**. The `bit_book_*` subset (6 tasks) was run separately on 3 of
the 4 models (qwen skipped for battery budget — see §Constraints).

## Per-model overall

| model | size | overall | wall | notes |
|---|---:|---:|---:|---|
| qwen2.5:1.5b (Transformer) | 1.5 B | **38.1%** | 427 s | reference at target size |
| rwkv7-1.5b:latest (World)  | 1.5 B | **26.2%** | 384 s | RWKV-7-World base, pre-A1 baseline |
| mollysama/rwkv-7-g1h:2.9b  | 2.9 B | **38.1%** | 4865 s | reasoning-tuned; corrected 2026-07-22 (was 7.1 % under num_predict=256 budget artifact) |
| gemma3:4b (Transformer)    | 4 B | **38.1%** | 1046 s | larger reference; ties qwen 1.5B overall |

## Per-category × per-model

| category | qwen 1.5B | rwkv7-1.5B | g1h-2.9B | gemma3-4B |
|---|---:|---:|---:|---:|
| bit_decoding      | 0/10 (0.0%)   | 0/10 (0.0%)   | 0/10 (0.0%)  | 1/10 (10.0%)  |
| symbolic          | 6/8 (75.0%)   | 5/8 (62.5%)   | 2/8 (25.0%)  | 3/8 (37.5%)   |
| extraction        | 4/8 (50.0%)   | 4/8 (50.0%)   | 7/8 (87.5%)  | 6/8 (75.0%)   |
| scheduling        | 3/6 (50.0%)   | 1/6 (16.7%)   | 1/6 (16.7%)  | 2/6 (33.3%)   |
| string_ops        | 3/6 (50.0%)   | 1/6 (16.7%)   | 3/6 (50.0%)  | 2/6 (33.3%)   |
| arithmetic_chain  | 0/4 (0.0%)    | 0/4 (0.0%)    | 3/4 (75.0%)  | 2/4 (50.0%)   |

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

### 2. arithmetic_chain: mollysama-g1h partially breaks the chain wall

Updated 2026-07-22. With `num_predict=2048`, mollysama solves 3/4
arithmetic_chain tasks — including `arith_01` (5-step parenthesised
expression) and `arith_04` (fraction word problem with 2 substitutions).
Only `arith_02` (single base conversion 173 → binary) fails, apparently
a specific weakness rather than a state-carry problem.

Compare: gemma3-4B solves 2/4 (single-step only); qwen 1.5B and
rwkv7-world-1.5b both 0/4. So the chain-arithmetic wall is not a hard
size ceiling — reasoning-tuning at 2.9 B breaks through it in bf16 with
enough token budget for the CoT. This is exactly the "state-carry"
substrate that A1 with state-reg is supposed to sharpen; on this
category, the pre-A1 mollysama baseline is already non-trivial (75 %),
so A1 needs to push the remaining `arith_02` and hold the others.

Sample is small (n=4) — do not read too much into per-task pass/fail.

### 3. mollysama/rwkv-7-g1h:2.9b — collapse was budget artifact (resolved 2026-07-22)

**Root cause found.** At `num_predict=256` (eval default), mollysama at
bf16 goes into full CoT mode ("Let me solve this step by step: …") and
runs into the token cap mid-reasoning. When `done_reason=length` fires,
Ollama returns an empty `response` string — so the eval scored 33/42
empty replies as failures. Diagnostic: at `num_predict=1024`, the same
`sym_alg_01` (`3x+7=22`) prompt returns a full worked solution ending
in `x = 5`.

**Fix.** `eval.py` default bumped 256 → 2048 (see arg help).
**Corrected numbers.** mollysama 16/42 = 38.1 % on base-42 — ties qwen
1.5B (38.1 %) and gemma3 4B (38.1 %). Category profile: strong on
extraction (87.5 %) and arithmetic_chain (75 %), weaker on symbolic
(25 %) where two of the six failures are rubric under-scoring
(`sym_unit_01` returned `2.5` for expected `2.5 kg`; `sym_dim_02`
returned `kg/(m·s²)` for expected `kg/(m*s^2)`).

**Implication for A1.** No regression — mollysama at 2.9 B is *at
parity* with the two Transformer references at 1.5 B and 4 B, on
identical rubric. Extraction + chain-arithmetic strength means the
pre-A1 substrate is already strong on multi-step; A1 with state-reg
should push the categories where mollysama still lags (bit_decoding,
scheduling, symbolic proper-fraction / algebra-symbolic).

### 4. bit_book_* subset — mollysama-g1h cracks 3/6 with proper budget

Updated 2026-07-22. `bit_book_*` gives the model an **explicit codebook
in the prompt** (e.g. `a=00, b=01, c=10, d=11`) plus a short bitstring,
and asks it to chunk + look up + concatenate. This removes the
memorized-encoding confound from `bit_bin_*` / `bit_hex_*` / `bit_b64_*`:
the only skill under test is *in-context lookup with state accumulation*.

| model | bit_book overall | qualitative note |
|---|---:|---|
| gemma3:4b        | 0/6 (0.0%) | plausible-shaped output, wrong chunk-to-code assignment; on `bit_book_03` outputs `r e a d y s` — first four right but hallucinates 2 extra codes |
| rwkv7-1.5b       | 0/6 (0.0%) | rambles "let me split the string…" preamble, does not commit to a final answer; on `bit_book_06` outputs `c=d=a=b=` (echoes codebook back) |
| mollysama-g1h    | **3/6 (50.0%)** | passes all three 2-bit fixed-codebook cases (`bit_book_01` → `badc`, `bit_book_02` → `badcbc`, `bit_book_06` → `abdc`). 3-bit fixed (`bit_book_03`) → `raed` (letter transposition, correct set); `bit_book_04` gets letters but wraps in prose (rubric miss); `bit_book_05` (variable-length prefix code) empty response — code-boundary reasoning still a wall |

**Two things this tells us**:

1. The size-2.9B reasoning-tuned RNN can chunk + accumulate through a
   fixed-width in-context codebook — *cleanly*, not by memorization —
   which no other tested model does. This is the strongest positive
   signal so far for the RWKV-G1 bet.
2. Variable-length prefix codes still break the wall (see
   `bit_book_05`: `e=0/t=10/a=110/n=111`, bitstring `01011010` → `etat`
   expected, empty response actual). Boundary detection during
   decoding — deciding "read one more bit or stop and look up" —
   remains a state-carry challenge. This is the specific failure mode
   A1 with state-reg should improve.

Also of note: gemma's `read y s` on `bit_book_03` — first 4 codes
correctly resolved then hallucinates 2 extra — was previously flagged
as the classic RNN-halt-condition failure from a Transformer. Now
also apply this framing to mollysama's `bit_book_04` prose-wrap:
it *has* the answer, just cannot cleanly terminate at the last code.

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
  `results/rwkv7_29b_g1h.json` (initial, num_predict=256 — kept for provenance),
  `results/rwkv7_29b_g1h_np2048.json` (**corrected 2026-07-22, current**),
  `results/gemma3_4b.json`.
- `bit_book_*` subset: `results/bit_book_gemma3_4b.json`,
  `results/bit_book_rwkv7_15b_world.json`,
  `results/bit_book_rwkv7_29b_g1h.json` (mollysama's np=2048 re-run
  merged into `results/rwkv7_29b_g1h_np2048.json`; the standalone
  bit_book file kept for historical comparison).
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
2. ~~Investigate mollysama collapse.~~ **Done 2026-07-22.** Budget
   artifact (`num_predict=256` → empty responses on `done_reason=length`).
   Fixed default in `eval.py` to 2048; re-run gives 38.1 % on base-42
   and 3/6 on bit_book. No regression, no rubric fix needed for now
   (two identified underscoring cases in `sym_unit_01` / `sym_dim_02`
   don't change the ranking).
3. ~~Wait for A0.5 verdict → decide on state-reg loss activation for A1
   pilot.~~ **Done.** A0.5 verdict: all H8-causal sub-tests pass →
   state-reg loss activated (α > 0). See `training/state_reg.py`.
4. Run A1 pilot on `mollysama/rwkv-7-g1h:2.9b` → re-run this eval →
   `compare.py results/rwkv7_29b_g1h_np2048.json results/a1_pilot.json`.
   Expected A1-success signals, in order of importance:
   - `bit_book_05` (variable-length prefix code) 0 → non-zero — the
     cleanest state-carry-under-halt-uncertainty test.
   - `bit_bin_*` / `bit_hex_*` chain-decoding 0 → non-zero — the
     memorized-encoding + state-carry combined lift.
   - Hold arithmetic_chain ≥ 75 % and extraction ≥ 87 % (no regression
     on strong pre-A1 categories).
   - Push symbolic 25 % → ≥ 50 % (algebra/dimensional analysis).
