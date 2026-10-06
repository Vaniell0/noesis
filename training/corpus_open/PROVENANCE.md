# PROVENANCE — training/corpus_open/

> The table at the end of this file is generated from the dataset catalog
> (`training/catalog/`, see `docs/datasets.md`); the per-source notes above it are the
> original hand-written snapshots (download method, licences, SHA-256 at download time).

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

<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python training/regenerate_corpus_index.py` -->

| Name | Role | Format | Rows | Provenance | Origin | State | Evidence | SHA-256 |
|------|------|--------|------|------------|--------|-------|----------|---------|
| g1i_warmup_v3_eos_train | — | pt-blob | 10529 | unknown | — | used-unmeasured | — | `3c7bb85f3756…` |
| A0_eval/tasks | eval | items | 48 | generated | Claude-authored 48-task A0.2 rubric set | measured | unsound | `9e7f9a93ba29…` |
| A0_eval/tasks_action | eval | items | 8 | generated | eval slice of the matrix generators | unreferenced | — | `7bf819fae9c3…` |
| A0_eval/tasks_bit_book | eval | items | 6 | generated | eval slice of the matrix generators | unreferenced | — | `033492da6fe8…` |
| A0_eval/tasks_bit_book_ext | eval | items | 14 | generated | eval slice of the matrix generators | unreferenced | — | `780d5d6ffa60…` |
| A0_eval/tasks_crossword | eval | tasks | 24 | generated | eval slice of the matrix generators | unreferenced | — | `9b0a2924331a…` |
| A0_eval/tasks_matrix_wordsearch | eval | tasks | 56 | generated | eval slice of the matrix generators | used-unmeasured | — | `8cca0ed5c9e3…` |
| A0_eval/tasks_matrix_wordsearch_name | eval | tasks | 56 | generated | eval slice of the matrix generators | unreferenced | — | `761c0501ba5c…` |
| A0_eval/tasks_matrix_wordsearch_nsp | eval | tasks | 56 | generated | eval slice of the matrix generators | has-results | — | `826247423f95…` |
| A0_eval/tasks_v2 | eval | items | 80 | generated | algorithmic eval generator, built because tasks.jsonl is not clean | used-unmeasured | — | `c485418a09ee…` |
| action_chains_dsl_step8_val | eval | pt-blob | 19 | generated | action_chains | used-unmeasured (personal) | — | `9498f309b4b3…` |
| action_chains_val | eval | pt-blob | 19 | generated | action_chains | unreferenced (personal) | — | `b9001f8fb1e0…` |
| aporia_val | eval | pt-blob | 9 | generated | — | used-unmeasured | — | `01d3054db817…` |
| archimob | eval | — | — | external-other | ArchiMob: spoken Swiss German oral-history interviews, dialectal and normalised transcription | planned | — | — |
| bitsub_val | eval | pt-blob | 17 | generated | — | used-unmeasured | — | `e19a95ad0b3c…` |
| g1i_warmup_v3_eos_val | eval | pt-blob | 1171 | unknown | — | used-unmeasured | — | `fdc977b67ce7…` |
| glaive_v2_val | eval | pt-blob | 1283 | external-hf | glaiveai/glaive-function-calling-v2 | superseded | — | `6a1f3ecbe01c…` |
| glaive_v2_val_dry100 | eval | pt-blob | 49 | external-hf | glaiveai/glaive-function-calling-v2 | superseded | — | `fd2e3539c592…` |
| hh_rlhf_val | eval | pt-blob | 3086 | external-hf | Anthropic/hh-rlhf | unreferenced | — | `b59c12b05c5e…` |
| lingoly | eval | — | — | external-other | LINGOLY: 1,133 Linguistics-Olympiad puzzles, 90+ mostly low-resource languages, 6 formats, 5 difficulty levels, with a no-context baseline that penalises memorisation (best model 38.7% on hard) | planned | — | — |
| mtob_kalamang | eval | — | — | external-other | MTOB (Tanzer et al., ICLR 2024): learn Kalamang <-> English from one grammar book; <250K tokens English, <25K tokens Kalamang; word list; 375 paired sentences. Kalamang (<200 speakers) is held out of the web. | planned | — | — |
| p1s4_eval | eval | recall | 540 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | unreferenced | — | `bada97213079…` |
| p1s8_eval | eval | recall | 540 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | unreferenced | — | `eee99282c3a1…` |
| p2_hard_eval | eval | recall | 48 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | used-unmeasured | — | `f12286f9c0cd…` |
| premise_refusal_val | eval | pt-blob | 11 | generated | — | used-unmeasured | — | `a2527c79306f…` |
| react_val | eval | pt-blob | 2932 | external-hf | react | unreferenced | — | `325f22c2ec32…` |
| sds200 | eval | — | — | external-other | SDS-200: 189 h Swiss German speech with Standard German text translations | planned | — | — |
| selfcot_val | eval | pt-blob | 5 | generated | selfcot | unreferenced | — | `f66a650e89db…` |
| sr_shared4_eval | eval | recall | 360 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | has-results | — | `323e6031e47e…` |
| sr_shared8_eval | eval | recall | 240 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | has-results | — | `b26d85db3963…` |
| staged_recall_eval | eval | recall | 1080 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | used-unmeasured | — | `8aa2643e6216…` |
| step6_mixed_val | eval | pt-blob | 3803 | unknown | — | unreferenced (personal) | — | `227fc26db65e…` |
| step9_rfc_val | eval | pt-blob | 29 | generated | rfc_qa | used-unmeasured | — | `4868c946f905…` |
| swissdial | eval | — | — | external-other | SwissDial: parallel spoken Swiss German in 8 dialects (AG, BE, BS, GR, LU, SG, VS, ZH) with High German transcripts, ~3 h audio per dialect, 26 h total; Grisons ~3x over-represented | planned | — | — |
| toolbench_sample_val | eval | pt-blob | 212 | external-hf | Yhyu13/ToolBench_toolllama_G123_dfs | unreferenced | — | `5c6edf008876…` |
| scots_wikipedia | negative-control | — | — | external-other | Scots Wikipedia: ~49% of the entries (20,000+) were written by one US editor who did not speak Scots, by grafting words from an online dictionary onto English sentences | planned | — | — |
| flores200 | payload | — | — | external-other | Meta NLLB: 3001 sentences from 842 Wikimedia articles, professionally translated into ~200 languages (fully parallel) | planned | — | — |
| tatoeba | payload | — | — | external-other | Tatoeba: volunteer-translated example sentences; the multilingual benchmark covers 112 languages with up to 1000 English-aligned pairs each | planned | — | — |
| udhr_unicode | payload | — | — | external-other | Universal Declaration of Human Rights in Unicode: 419 translations (project count from search); one identical text in every language | planned | — | — |
| A0_H12a_working_memory/tasks-N16 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `aed38a4473cc…` |
| A0_H12a_working_memory/tasks-N32 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `85b1dfde39bf…` |
| A0_H12a_working_memory/tasks-N4 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `c70864c8b4d9…` |
| A0_H12a_working_memory/tasks-N64 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `9ec4038fb07d…` |
| A0_H12a_working_memory/tasks-N8 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `82caf8e2ac3d…` |
| A0_H12a_working_memory/tasks-dist-1000 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `287cbe6faefe…` |
| A0_H12a_working_memory/tasks-dist-200 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `952d3198969f…` |
| A0_H12a_working_memory/tasks-dist-50 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `0876edec2abb…` |
| A0_H12a_working_memory/tasks-dist-500 | probe-items | items | 3 | generated | H12a working-memory triples | has-results | — | `0059796a752a…` |
| A0_portability/tasks | probe-items | unknown | 3 | generated | A0.6 portability prompt pairs | used-unmeasured | — | `0022ebb06b9c…` |
| aporia_probe/items | probe-items | items | 30 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `c2ead9850f52…` |
| aporia_probe/items_100 | probe-items | items | 100 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | unreferenced | — | `2cb7f8278884…` |
| attribution_probe/items | probe-items | items | 19 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `8fe5cbb83b3d…` |
| attribution_probe/items_overlap | probe-items | items | 32 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `a480acc2e19b…` |
| attribution_probe/items_v2 | probe-items | items | 243 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `30e09311ad35…` |
| premise_validator/items | probe-items | items | 40 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `c139ec98ea70…` |
| premise_validator/items_ctx | probe-items | items | 80 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `64e23734484b…` |
| premise_validator/items_v2 | probe-items | items | 280 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `1f1a2abf3096…` |
| premise_validator/items_v4_clean | probe-items | items | 40 | generated | probe item sets (TruthfulQA / C4 mining where noted in the probe's mine_*.py) | used-unmeasured | — | `62bfa8fe027d…` |
| glaive_function_calling_v2 | raw | json | — | external-hf | glaiveai/glaive-function-calling-v2 (Apache-2.0) | used-unmeasured | — | `e9b5d671812b…` |
| matrix_tasks | tasks | tasks | 65797 | generated | procedural matrix tasks (wordsearch, arithmetic, bits, pattern, crossword) | used-unmeasured | — | `0ca16df762fe…` |
| action_chains | train | rollouts | 1154 | generated | Claude CLI session traces, sanitised (training/corpus/sanitised) | used-unmeasured (personal) | — | `22e0d9f3e56e…` |
| action_chains_dsl | train | rollouts | 1154 | generated | action_chains converted to the runtime DSL | measured (personal) | failed | `953d972cf6ab…` |
| action_chains_dsl_step8_train | train | pt-blob | 1133 | generated | action_chains | used-unmeasured (personal) | — | `322905809241…` |
| action_chains_train | train | pt-blob | 1133 | generated | action_chains | measured (personal) | no-effect | `d4a3c155d179…` |
| aporia_train.jsonl | train | think | 80 | generated | hand-written ambiguity cases | used-unmeasured | — | `2438d04d4c86…` |
| aporia_train.pt | train | pt-blob | 71 | generated | — | used-unmeasured | — | `b7d127d04619…` |
| bitsub_train.jsonl | train | think | 120 | generated | procedural bit-substitution tasks | used-unmeasured | — | `19039e91afe4…` |
| bitsub_train.pt | train | pt-blob | 103 | generated | — | used-unmeasured | — | `c7133690cc2f…` |
| g1i_warmup_v3 | train | think | 11700 | generated | matrix_tasks with procedural <think> that restates the answer | measured | confirmed | `f7cf4e3ee411…` |
| g1i_warmup_v3_flat | train | text | 11700 | generated | matrix_tasks with procedural <think> that restates the answer | measured | confirmed | `acad0123cda4…` |
| glaive_v2 | train | rollouts | 63218 | external-hf | glaiveai/glaive-function-calling-v2 | superseded | failed | `5d413fb97f2b…` |
| glaive_v2_train | train | pt-blob | 61933 | external-hf | glaiveai/glaive-function-calling-v2 | superseded | — | `59e243b3ed9f…` |
| glaive_v2_train_dry100 | train | pt-blob | 99 | external-hf | glaiveai/glaive-function-calling-v2 | superseded | — | `2ce99e898969…` |
| hh_rlhf | train | think | 30000 | external-hf | Anthropic/hh-rlhf (chosen side, CAI-style filter) | used-unmeasured | — | `304e6c26edef…` |
| hh_rlhf_train | train | pt-blob | 26914 | external-hf | Anthropic/hh-rlhf | used-unmeasured | — | `df06a53c703f…` |
| p1s4_train | train | recall | 2400 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | unreferenced | — | `23940b6909c3…` |
| p1s8_train | train | recall | 2400 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | unreferenced | — | `2823232a4240…` |
| p2_hard_train | train | recall | 4 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | unreferenced | — | `f6c5975ff7c0…` |
| premise_refusal_train.jsonl | train | think | 80 | generated | false-premise questions with refusals | used-unmeasured | — | `f29220332eca…` |
| premise_refusal_train.pt | train | pt-blob | 69 | generated | — | used-unmeasured | — | `b9d240f98473…` |
| react | train | think | 30000 | external-hf | glaive + ToolBench re-rendered as ReAct <think> turns | used-unmeasured | — | `040e95fc2a4e…` |
| react_train | train | pt-blob | 27068 | external-hf | react | used-unmeasured | — | `5dc2bc20723f…` |
| relax_v1 | train | items | 1776 | generated | relaxed-format task variants | used-unmeasured | — | `3dec2c140260…` |
| rfc_qa | train | think | 297 | generated | IETF RFC text restructured into reasoning tasks | used-unmeasured | — | `294e7036cb00…` |
| selfcot | train | think | 29 | generated | model's own CoT on A0 tasks, filtered (same-model generator rule) | used-unmeasured | — | `7aa560427a05…` |
| selfcot_train | train | pt-blob | 24 | generated | selfcot | used-unmeasured | — | `ae72f2f5d63d…` |
| sr_shared4_train | train | unknown | 0 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | empty | — | — |
| sr_shared8_train | train | unknown | 0 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | empty | — | — |
| staged_recall_train | train | recall | 3600 | generated | key->value pairs, filler gap, one query (answer only recoverable from state) | used-unmeasured | — | `d30fe9e7fe97…` |
| step9_rfc_train | train | pt-blob | 268 | generated | rfc_qa | measured | helped, unsound | `fe2ee8bce664…` |
| step9b_combined_flat | train | text | 837 | generated | step9b_combined rendered flat for probes | used-unmeasured (personal) | — | `3bcc73ae0590…` |
| toolbench_sample_train | train | pt-blob | 9786 | external-hf | Yhyu13/ToolBench_toolllama_G123_dfs | unreferenced | — | `9dd9d1ef179f…` |
| toolbench_train | train | rollouts | 187536 | external-hf | Yhyu13/ToolBench_toolllama_G123_dfs (MIT) | used-unmeasured | — | `62dad814693d…` |
