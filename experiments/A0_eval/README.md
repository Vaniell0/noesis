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
