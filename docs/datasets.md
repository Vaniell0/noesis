# Datasets — one catalog for everything the project trains or measures on

Catalogued: 90 artifacts on disk and 9 planned external sources (2026-10-06).

Data used to be produced by about twenty scripts in four families (SFT normalizers,
procedural generators, RL task sets, probe items) and lived in `training/corpus_open/`,
`training/tokenised/` and `experiments/*/`. Nothing recorded how a file was made, which
code read it, or whether a result had been measured on it. The catalog answers those
three questions for every artifact, and answers a fourth that was missing: **is the claim
about this data confirmed, or only used.**

```
training/datasets.py   ls | show | usage | verify | lineage | overlap | rows | backfill | index
training/_common/catalog.py     reader, verifier, lineage, usage, overlap, backfill
training/catalog/*.json         one record per artifact (tracked in git)
training/catalog/hints.yaml     facts a scan cannot see: origin, generator command, parents, evidence
training/tests/test_catalog.py  throw-away-repo tests
```

## Records, not sidecars

`training/tokenised/` and `corpus_open/*.json*` are gitignored (the artifacts are large), so a
`.provenance.json` beside each file would never reach git and would vanish with the data.
Records therefore live apart, in `training/catalog/`, in the same `ProvenanceRecord` schema
the pipeline stages already write (`training/_common/provenance.py`). The artifact is found
by `out_path` and checked by `out_sha256`: a fresh machine that regenerated or copied the
data runs `datasets.py verify` and learns whether it holds the files the results were
measured on.

A record carries what was measured about the file (size, SHA-256, rows, tokens, a profile of
arms / categories / levels) and what is known about it:

| field | meaning |
|---|---|
| `role` | `train`, `eval`, `tasks`, `probe-items`, `raw` |
| `format` | `pt-blob`, `rollouts`, `think`, `tasks`, `recall`, `items`, `text` — detected from the rows |
| `parents` | catalog names it was derived from (raw → jsonl → pt → combined) |
| `recipe` | the command that regenerates it, with the seed and whether it is deterministic |
| `verifier` | what decides "correct" for the rows (the same checker serves reward, teacher filtering, eval) |
| `status` | `live`, `superseded`, `reclassified`, `empty`, `unreadable`, `scratch` |
| `evidence` | what was measured on it: run, result, verdict, and the document that holds the number |
| `license`, `url`, `languages` | for external sources; a share-alike licence carries to every set derived from it |
| `sensitivity` | `personal` for data derived from the owner's own sessions (inherited, see below) |

Names are the file stem, prefixed with the experiment directory under `experiments/`. When two
files share a stem (`aporia_train.jsonl` and its tokenised `aporia_train.pt`) both get their
extension, so neither record overwrites the other.

## Is it confirmed? (`datasets.py usage`)

The state is computed, never declared:

| state | meaning |
|---|---|
| `measured` | curated evidence exists: a verdict with a number and its source |
| `has-results` | result files exist whose recorded arguments name this file; no curated verdict yet |
| `used-unmeasured` | code or config mentions the file; no result names it |
| `unreferenced` | nothing mentions the file name. Paths built in code or passed on the command line are invisible to the scan, so this is a lead to check, not a verdict |
| `planned` | a source we intend to use but have not downloaded; its licence, size and caveats are recorded so the "may we, under what terms" decision is made once. `verify` skips it until a download gives it a file |
| `superseded`, `empty`, `unreadable` | taken from the record |

Verdicts: `helped`, `hurt`, `no-effect`, `failed`, `unsound`, `confirmed`, `untested`. `unsound`
marks a measurement that cannot be trusted as stated (an eval set authored by the model family
being evaluated; a run whose result has two confounded causes).

The `has-results` link is read from the result files themselves: every result JSON records its
`args`, so `staged_recall_p1_step100_final.json` is what shows that P1 trained and evaluated on
`p1_shared_*` and on no other file.

## Personal data

Rows derived from the owner's own Claude sessions (`action_chains*` and everything combined from
them: `step6_mixed_*`, `step9_combined_train`, `step9b_combined_*`) are marked `sensitivity: personal`.
The mark is inherited down `parents`, so a new combination of a personal set is personal without
anyone remembering to say so. `rows()` refuses such sets unless called with `allow_personal=True`
(`datasets.py rows NAME --allow-personal`); `ls` and the generated index show them as `personal`.
The catalog scan never walks `training/corpus/` or `training/sanitised/` (the raw traces). Records
themselves hold metadata only — hashes, counts, paths — never row text.

## Checks that come with it

- `verify [NAME]` — artifact present, size and SHA-256 equal to the record, status not
  `unreadable`, parents catalogued, no second record for the same path.
- `overlap EVAL TRAIN` — exact-prompt overlap after lowercasing and whitespace normalisation;
  the cheap contamination check. It does not catch paraphrases.
- `lineage NAME` — first-parent chain back to the source.
- `rows NAME` — one reader for every format, including `.pt` blobs (memory-mapped).

## What the first pass found (2026-10-06, 90 artifacts)

- Two generated files are empty by design: `sr_shared4_train` and `sr_shared8_train` came from calls
  with `--train-per-cell 0` (eval-only sets). They are recorded as `empty`.
- `p1_shared_eval` is byte-identical to `p1s4_eval` + `p1s8_eval`; `p1_shared_train` has the same rows
  as `p1s4_train` + `p1s8_train` in another order. P1 was trained on that union.
- The step-10 corpus parts were built with `--n 80/120/80`, not the `60/100/60` examples in the
  generators' docstrings (row counts match the `pilot_step10.yaml` header).
- The generator commands of `p1s4`, `p1s8`, `sr_shared4/8`, `p2_hard` existed only in a session
  transcript; they are now in the records.
- `A0_eval/tasks` (the 48-task rubric set) was authored by Claude and the step-9 score is an upper
  bound until a clean generator's set (`tasks_v2`) is used (`docs/verdicts/2026-08-08-step8-step9.md`).
- `action_chains*` derive from personal Claude CLI traces; `training/corpus/RECLASSIFIED.md` and
  `docs/policies.md` keep personal data out of weights, and steps 6-8 trained on them after that
  decision. The catalog records the derivation; the policy question is the owner's.

## Adding a dataset

Produce it through a generator registered in `training/_common/registry.py` (kind `normalize`,
`tokenize` or `generate`) or write its entry in `hints.yaml`, then run `datasets.py backfill`
(existing records are kept, hand edits survive; `--overwrite` regenerates them). A record with no
`recipe` is an honest gap, not an error.
