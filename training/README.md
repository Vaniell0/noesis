# noesis / training

A1 fine-tune pipeline — action-cloning target on the user's local Claude
Code corpus, with a state-regularization hook wired in but disabled by
default (α=0). Full design: `plans/cosmic-purring-cocke.md`.

## Status

Built in the 2026-07-21 parallel scaffolding session (brief:
`~/.claude/plans/parallel_a1_scaffolding.md`):

- **Step 1 — env**: `.venv/` (system-site-packages), `peft 0.19.1` and
  `detect-secrets 1.5.0` installed from local wheels in
  `~/.libs/python/`. `bitsandbytes 0.49.2` wheel is downloaded but not
  installed into the venv (QLoRA path, only needed at Step 6 train time).
- **Step 3 — corpus extraction** (`extract_traces.py`): scans
  `~/.claude/projects/**/*.jsonl` + `~/.claude/history.jsonl`, filters
  sessions with ≥ 2 tool_use events, strips assistant thinking/text,
  emits `training/corpus/raw/<rid>.jsonl` (61 rollouts from 83 jsonls
  as of the initial run).
- **Step 4 — sanitization** (`sanitize_patterns.py` + `sanitize.py`):
  regex families for vendor API keys (Anthropic / OpenAI / xAI / Slack /
  GitHub / Stripe / HuggingFace), AWS access-key + secret co-location,
  GCP service-account JSON, high-entropy `.env` KV assignments, SSH /
  PGP private-key blocks, and private IPv4 addresses. `detect-secrets`
  runs in parallel with its high-noise plugins disabled (Base64 /
  HexHighEntropyString / KeywordDetector — they flag every hash and
  UUID as a "secret").

  Policy: `tool_result` and `user` matches are **redacted in place**
  (`<REDACTED:{pattern}>`). `tool_use` matches drop the whole rollout
  when the pattern is credential-class (see `PATTERN_CLASS` in
  `sanitize_patterns.py`), otherwise they are also redacted — the model
  should not learn to emit `<REDACTED>` as an action.

  Output: `training/sanitised/<rid>.jsonl` per rollout, plus
  `training/sanitised/audit.jsonl` (one JSON record per input rollout
  describing what happened).

- **Step 4 audit gate** (`audit_sample.py`): interactive TUI. Presents
  a random sample of sanitised rollouts, records
  `accept / reject / flag_refine / skip` in
  `training/sanitised/audit_decisions.jsonl`. **Must be run at least
  once before Step 5.** Recommended: `python audit_sample.py -n 50` as a
  first pass; increase to 200 once a couple of regex refinements have
  been made.

- **Step 6 skeleton (partial)**: `JL-er/RWKV-PEFT` vendored into
  `training/rwkv-peft/` at commit
  `5704c39f8ab1d2ac63936ab392aadb6ba526e1a5` (`.git` removed).
  `state_reg.py` and `config/pilot.yaml` are stubs — see below.

## Pending (main-session work)

- **Step 2 — A0.2 held-out eval** (upstream). Blocks any verdict on the
  pilot: without A0.2 numbers we cannot claim "beats pretrained by ≥ 5 pp".
- **Step 5 — tokenization** (`tokenize_rollouts.py`): apply
  `rwkv_vocab_v20230424` to rollouts formatted with `<user>` /
  `<tool_use>` / `<tool_result>` tags; build `loss_mask` selecting
  tool_use positions only.
- **Step 6 — full LoRA harness** (`lora_train.py`): uses vendored
  RWKV-PEFT + `state_reg.py`. Config already exists as
  `config/pilot.yaml` with α=0.
- **Step 7 — pilot run** on 5-10 % corpus subset, 0.4B.
- **Step 8 — A0.5 verdict integration**: replace `state_reg.py` stub
  with concrete branch (encourage_motion / penalize_still /
  per_layer_weighted) once A0.5 sharpens the signal. Then re-pilot with
  α > 0 and compare.

## State-reg hook — deliberately a stub

`state_reg.compute_state_reg(...)` returns `0.0` unconditionally, in
every mode. This is by design: A0.5 (causal state-intervention on
0.4B) has not been run yet, so we do not know which branch of Design
§7 to implement. The stub keeps the pipeline surface stable — enabling
state-reg once A0.5 lands is a config edit + a function body.

`VALID_MODES` in `state_reg.py` deliberately rejects unknown modes at
config load so a typo does not silently activate a non-existent loss.

## Directory layout

```
training/
  .venv/                       (gitignored) system-site-packages venv
  README.md                    this file
  extract_traces.py            Step 3
  sanitize.py                  Step 4 driver
  sanitize_patterns.py         Step 4 regex families + class map
  audit_sample.py              Step 4 manual audit gate
  state_reg.py                 Step 6 hook (STUB)
  config/
    pilot.yaml                 Step 7 pilot config (α=0)
  corpus/
    raw/                       (gitignored) extract_traces.py output
  sanitised/                   (gitignored) sanitize.py output +
                               audit.jsonl + audit_decisions.jsonl
  tokenised/                   (gitignored) reserved for Step 5
  runs/                        (gitignored) reserved for Step 7
  rwkv-peft/                   (gitignored) vendored JL-er/RWKV-PEFT
                               commit 5704c39f
```

## Reproducing

```bash
cd /home/vaniello/Desktop/projects/noesis

# activate venv
source training/.venv/bin/activate

# 1. extract (dry-run first)
python training/extract_traces.py --dry-run --limit 10
python training/extract_traces.py

# 2. sanitize
python training/sanitize.py

# 3. audit — required before tokenisation
python training/audit_sample.py -n 50
```
