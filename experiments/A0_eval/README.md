# A0.2 — held-out reasoning eval

30–50 memorization-free reasoning tasks, drawn from workflow shapes
noesis is designed to serve. Yardstick for A1 fine-tune: A1 must
measurably improve at least one category without regressing others.

## Design

- **Memorization-free.** Every parametric value (bit string, key length,
  constants, names) is fresh per task — no MMLU-style facts, no
  benchmark-leakage risk. A model can only solve these by *running the
  procedure* on the given input.
- **Diagnostic.** Categories are chosen so a per-category score maps to
  a specific cognitive substrate: bit-decoding = state-carried working
  memory (accumulating prefix), scheduling = combinatorial search,
  extraction = instruction-following + parsing, symbolic = symbol
  manipulation.
- **Format-tolerant scoring.** Rubric-based, not exact-match by default,
  so a correct answer wrapped in prose or formatted differently still
  scores; but the substantive content is validated.

## Task schema (tasks.jsonl)

Each line is one JSON object:

```json
{
  "id": "bit_ascii_01",
  "category": "bit_decoding",
  "prompt": "…user-facing prompt…",
  "answer": "expected substantive answer",
  "rubric": {"type": "exact"|"contains"|"regex"|"json_subset"|"manual",
             "value": "…"},
  "notes": "why this task discriminates"
}
```

**Rubric types.**
- `exact` — output stripped and lowercased must equal `value`.
- `contains` — case-insensitive substring match.
- `regex` — Python `re.search(value, output, re.IGNORECASE)`.
- `json_subset` — model output parsed as JSON; `value` is a dict whose
  keys+values must appear as a subset (deep) inside the parsed output.
- `manual` — flagged for human review; auto-scored as 0 until a
  reviewer marks it.

## Categories

| category | tasks | why |
|---|---:|---|
| `bit_decoding` | ~10 | binary/hex/base64/ROT/XOR/custom substitution → accumulating-prefix working memory |
| `symbolic` | ~8 | small algebra, unit conversions → symbol manipulation |
| `extraction` | ~8 | noisy text + schema → JSON — instruction-following + parsing |
| `scheduling` | ~6 | small CSP puzzles — combinatorial search |
| `string_ops` | ~6 | reverse, transform, apply rules — sequence manipulation |
| `arithmetic_chain` | ~4 | multi-step arithmetic, base conversion — carrying intermediate results |

## Corpus blocklist

Tasks curated from the user's `.claude` history must have their
`session_id` recorded in `corpus_blocklist.txt` **before** corpus
extraction runs, to prevent train/eval leak.

## bit_book extension (`tasks_bit_book_ext.jsonl`)

14 additional in-context codebook-decode tasks. Purpose: bring
`bit_book_*` sample size from n=6 to n=20 so cross-model results can
be reported with a confidence interval instead of a raw proportion
(arxiv-brief §4.2 nice-to-have).

**Held separately from `tasks.jsonl`** — do NOT merge. The base 48-task
set is the comparison anchor for §3 of the paper; merging the
extension mid-flight would move the headline denominator and break
provenance with `results/*_np2048.json`.

Split (all memorization-free-by-construction — codebook + bitstring
live entirely in the prompt, no shared codebooks with the base set):

| slice                       | n | ids                            |
|-----------------------------|--:|--------------------------------|
| 2-bit fixed codebook        | 4 | `bit_book_ext_01`..`_04`       |
| 3-bit fixed codebook        | 4 | `bit_book_ext_05`..`_08`       |
| variable-length prefix code | 3 | `bit_book_ext_09`..`_11`       |
| mixed-case / whitespace     | 3 | `bit_book_ext_12`..`_14`       |

Differences from base `bit_book_01..06`:

- **Codebook diversity.** New symbol sets in each task (w/x/y/z,
  p/q/r/s, m/n/o/k …), so a model that memorised the base six
  codebooks gets no shortcut.
- **Length range.** 4–8 chars decoded; longer sequences (ext_08 = 8
  chars) probe carry-state further than base (max 6 chars).
- **Fixed-vs-prefix balance.** 3 prefix-code tasks (ext_09..11) vs 1
  in the base set — the boundary-detection variant is the specific
  failure mode A1 aims to move.
- **Case + whitespace slice.** ext_12..14 test whether decoded output
  preserves the character *shape* the codebook mandates
  (uppercase/lowercase, spaces, hyphens). The `contains` rubric is
  case-insensitive, so strict case-fidelity is a follow-up check for
  `rubric_audit.py` — see task notes.

Run against a model:

```bash
python3 eval.py --model mollysama/rwkv-7-g1h:2.9b \
                --tasks tasks_bit_book_ext.jsonl \
                --out results/bit_book_ext_g1h_np2048.json
```

## Post-hoc rubric audit (`rubric_audit.py`)

Re-scores existing `results/*_np2048.json` under tolerant matcher
variants without touching `eval.py` or the original files. Rescues
answers that failed on trivial formatting differences (unicode math
symbols, dropped units, prose wrappers) — flagged in arxiv-brief §4.2.

```bash
# Dry-run: show upgrade diff, refuse to write when |Δ| > 2 pp
python3 rubric_audit.py results/*_np2048.json

# Commit: force write regardless of shift (after diff review)
python3 rubric_audit.py --commit results/*_np2048.json
```

Output goes to `results/<orig>_audited.json` next to the original.
Full `results/<orig>.json` provenance is preserved; audited numbers
live under new keys (`aggregate_audited`, `audit_upgraded` per task).

## Scoring

```bash
# Baseline: run the eval against a pretrained model
python3 eval.py --model rwkv7-2.9b --out baseline_g1h.json

# After A1 fine-tune, compare
python3 eval.py --model noesis-a1 --out a1.json
python3 compare.py baseline_g1h.json a1.json
```

Per-category and overall accuracy; category deltas surface which
cognitive substrate A1 shifted.
