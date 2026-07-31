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

## THUDM/AgentInstruct

- **Status:** listed on HF, distributed as multiple parquet files
  under `data/`. Not attempted this session — `pyarrow`/`pandas`
  wheels not present in `training/.venv`.
