# PROVENANCE — training/corpus_open/

Snapshot manifest for corpora that enter A1 weights under Variant C
hybrid (see `docs/policies.md § A1 fine-tune corpus scope`). One
entry per source dataset; verify SHA-256 before any downstream
step. Do NOT commit the raw JSON files here — they are large and
pointed at via `.gitignore`. This manifest is the traceable record.

## glaiveai/glaive-function-calling-v2

- **File:** `glaive_function_calling_v2.json`
- **Size:** 271,190,065 bytes (258.6 MB)
- **SHA-256:** `e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad`
- **Downloaded:** 2026-07-30 via
  `curl --socks5-hostname 127.0.0.1:2080` from
  `https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json`
- **HF repo:** `glaiveai/glaive-function-calling-v2`
  (gated=False, Apache-2.0)
- **Raw rows:** 112,960 dicts of `{system, chat}`.
- **Normalizer:** `training/scripts/normalize_glaive.py`.
- **Normalized file:** `glaive_v2.jsonl` (59,932,037 bytes, 63,218
  rollouts). Drop reasons: `no_tools_in_system=34,598`,
  `no_tool_uses=15,144`.
- **Tokenized (`rwkv_vocab_v20230424`):**
  - `training/tokenised/glaive_v2_train.pt` — 61,934 rollouts,
    11,808,829 tokens, 2,322,476 supervised (19.7%).
  - `training/tokenised/glaive_v2_val.pt` — 1,284 rollouts,
    252,394 tokens, 48,818 supervised (19.3%).
  - Split: `blake2b(id, 4) % 100 < 2` → val, deterministic.

## Salesforce/xlam-function-calling-60k

- **Status:** NOT DOWNLOADED. HF repo is gated
  (`GatedRepoError 401 at hf_hub_download`) — requires an HF account
  with the Apache-2.0 license accepted on the dataset page.
- **Next step:** when a token is available, set `HF_TOKEN` in the
  shell and re-run
  `training/.venv/bin/python training/scripts/normalize_xlam.py`.
  Retokenize as `training/tokenised/xlam_60k_train.pt` +
  `xlam_60k_val.pt`.

## thunlp/ToolBench

- **Status:** not attempted this session. MIT licence, ~16k real
  APIs with long ReAct-style chains. Best for multi-step / error-
  recovery coverage per shortlist §1.

## matrix_tasks — locally generated RL curriculum (A1.5)

- **File:** `matrix_tasks.jsonl`
- **Size:** 38,405,794 bytes (36.6 MB)
- **SHA-256:** `0ca16df762fee70cdbd25f09bbacf138ac8ff41a2897b4c7832decaa312a20e4`
- **Generated:** 2026-08-16 via `experiments/A0_eval/gen_tasks.py`
  Current file is the base run (no sudoku/ARC), ~20M estimated tokens. Full run with external data:
  ```bash
  python3 experiments/A0_eval/gen_tasks.py \
      --n-tokens 20_000_000 --seed 42 \
      --out training/corpus_open/matrix_tasks.jsonl \
      --sudoku-csv ~/data/sudoku.csv \
      --arc-dir ~/data/ARC-AGI/data/training
  ```
- **Total tasks:** 65,797
- **Task breakdown:**

  | Category | Count | RL role |
  |---|---|---|
  | `matrix_wordsearch` | 13,098 | primary task (position, L1–L7) |
  | `matrix_wordsearch_name` | 6,679 | bootstrap warmup — name the word; L1/L2 rare (~5%), L3-7 dominant |
  | `arithmetic_matrix` | 12,988 | auxiliary — column arithmetic, carry, error detection |
  | `bits_matrix` | 13,126 | auxiliary — XOR/AND/OR/NOT, reverse lookup |
  | `pattern_matrix` | 13,312 | auxiliary — sequence extrapolation, rule induction |
  | `crossword_enum` | 3,297 | auxiliary — constrained word retrieval |
  | `crossword_fill` | 3,297 | auxiliary — constrained word retrieval |

- **Rubric types:** `regex` (wordsearch position, crossword), `exact` (wordsearch_name, bits, arithmetic, pattern).
- **Format:** each line is `{"id":..., "category":..., "level":..., "prompt":..., "answer":..., "rubric":{...}}`.
- **Usage:** `--tasks training/corpus_open/matrix_tasks.jsonl` in `train_wordsearch.py`.
  GRPO samples G=8 rollouts per prompt; `r_correct` checks rubric; curriculum advances
  wordsearch level when batch acc > 80%.
- **Not committed** (gitignored — large; regenerate with gen_tasks.py if lost).

## THUDM/AgentInstruct

- **Status:** listed on HF, distributed as multiple parquet files
  under `data/`. Not attempted this session — `pyarrow`/`pandas`
  wheels not present in `training/.venv`.
