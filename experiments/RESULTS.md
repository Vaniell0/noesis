# Results index

Single flat index of numeric results. Not a replacement for the hypothesis write-ups in
`HYPOTHESES.md` — those carry the full methodology, falsification criteria, and caveats.
This is a lookup table: which H, which model, what number, where the raw file is.

Two sections:
- **Historical** (below) — hand-backfilled 2026-08-17 from HYPOTHESES.md text, one-time,
  static. `lost` = the raw result file was on the Selectel VM (deleted) or in `/tmp` and
  was never persisted; the number survives only as prose. `no file cited` = HYPOTHESES.md
  reports the number without pointing at a file.
- **Live index** (bottom, below the `AUTO-GENERATED` marker) — rebuilt by
  `python experiments/regenerate_results.py` from `_meta` blocks that
  `experiments._common.results.save_result` stamps into result JSON files. Grows
  automatically as scripts adopt the shared framework (`experiments/_common/`,
  `experiments/run.py`) — nothing here needs manual transcription. Do not hand-edit
  below the marker; it will be overwritten on next regeneration.

## Historical

| H | Model | Date | Metric | Value | File |
|---|-------|------|--------|-------|------|
| H8 | G1i 2.9B base | 2026-08-16 | IPC_total, held-out R², L0 | 0.53 | `experiments/A0_state_probe/results/ipc_g1i_fixed.json` |
| H8 | G1i 2.9B base | 2026-08-16 | IPC_total, held-out R², L4–L31 | ≈0.0 | `experiments/A0_state_probe/results/ipc_g1i_fixed.json` |
| H8 | G1d/G1h/G1i/step9b-e1 | 2026-08-16 | Mean IPC, native 512 tok (fleeb83) — **unreconciled vs. row above** | 4.17 / 2.73 / 2.76 / 3.67 | no file (email) |
| H8 | G1d/G1h/G1i/step9b-e1 | 2026-08-16 | Mean IPC, G1h-base teacher-forced traj, 1024 tok (fleeb83) | 7.43 / 7.44 / 7.46 / 7.66 | no file (email) |
| H8 | G1G (Xyra) | 2026-08-16 | L31 IPC, BASE → RDP-8 → RDP-17 | 0.97 → 0.64 → 3.16 | no file (email) |
| H9 | G1d 0.4B | 2026-08-04 | state-motion cross-prompt ratio vs World3 | 21–99× | no file cited |
| H9 | G1h 2.9B | 2026-08-05 | σ-slope, G1h vs World3 | 1.67/1.58 vs 1.19/1.13 | `experiments/A0_state_probe/results/a05_2.9b_h9_verdict.md` |
| H9 | G1h 2.9B | 2026-08-05 | KL @ σ=0.1, G1h vs World3 (40× gap) | 0.230 vs 0.0055 | `experiments/A0_state_probe/results/a05_2.9b_h9_verdict.md` |
| H10 | G1h 2.9B step8 e0 | 2026-08-07 | overall acc, N=1 silent (baseline) | 27.1% | `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md` |
| H10 | G1h 2.9B step8 e0 | 2026-08-07 | overall acc, N=2 silent (best cell) | 33.3% | `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md` |
| H10 | G1h 2.9B step8 e0 | 2026-08-07 | overall acc, N=3 silent (collapse) | 6.3% | `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md` |
| H10 | G1i 2.9B base | pending (PID 133922) | state_readout N=1 K=32, overall acc | TBD | `experiments/A0_eval/results/h10_state_readout_g1i_base.json` |
| H12a | G1d 0.4B | 2026-07-23 | recall vs gap, decay axis (gap 14→229) | 0.40 → 0.02 | `experiments/A0_H12a_working_memory/` (v1) |
| H12a | G1d 0.4B | 2026-08-11 | recall vs N, width axis (tail-gap=50w fixed) | 0.40 → 0.40 → 0.05 (N=4/8/16) | `experiments/A0_H12a_working_memory/results-v2/` |
| H12b | G1d 0.4B / G1h base / G1h step7 | 2026-08-07 | aggregated accuracy, K=2 (all P) | 18% / 60% / 36% | `experiments/A0_H12b_multislot/results/report_2026-08-07.md` |
| H12b | G1d 0.4B / G1h base / G1h step7 | 2026-08-07 | aggregated accuracy, K=8 (all P) | 7% / 26% / 23% | `experiments/A0_H12b_multislot/results/report_2026-08-07.md` |
| H18 | G1d 0.4B | 2026-08-13 | fork determinism (sub-claim 1) | CONFIRMED, 100% identical greedy | `experiments/H18_merge/test_arithmetic_merge.py` |
| H18 | G1d 0.4B | 2026-08-13 | arithmetic merge, science_recipe pair, α∈{0.3,0.5,0.7} | both contexts preserved at all α | `experiments/H18_merge/results/h18_merge_g1d_04b.json` |
| H20 | G1d 0.4B | 2026-07-30 | collapse_cont aggregate (100 items) | 0.541 | `experiments/aporia_probe/report.md` |
| H20 | G1d 0.4B | 2026-07-30 | p(neither) aggregate (100 items) | 0.746 | `experiments/aporia_probe/report.md` |
| H20 | G1h 2.9B base / step7 | 2026-08-07 | collapse_cont aggregate (100 items) | 0.665 / 0.639 | lost (`/tmp/ts_results/`) |
| H21 | G1d 0.4B | 2026-07-30 | LOO F1, pilot (40 items) | 0.789 | `experiments/premise_validator/report.md` |
| H21 | G1h 2.9B | 2026-07-30 | LOO F1, genuine LOO (same pilot item set as G1d) | 0.850 | lost (`/tmp/ts_results/`) |
| H21 | G1h 2.9B | 2026-08-07 | single 8-item held-out split (items_v4_clean.jsonl) — **not LOO, previously mislabeled as such** | 0.889 | lost (`/tmp/ts_results/`) |
| H21 | G1h 2.9B | 2026-08-08 | train/test F1, 32/8 split (v3_29b) | 1.000 | `experiments/premise_validator/v3_29b/loo_results.jsonl` |
| H21 | 2.9B / 7.2B | 2026-08-11 | external validation (Scarletwolf): abstention missed, before→after | 18→7 (p=0.013) / — (p=0.002) | external, `scarletwolf.ai/en/blog/rwkv-enseigner-ou-lire` |
| H22 | G1d 0.4B | 2026-07-30 | LOO F1, v2 (240 items, C4-mined) | 0.947 | `experiments/attribution_probe/v2/report.md` |
| H22 | G1h 2.9B base / step7 | 2026-08-07 | LOO F1 (243 items) | 0.929 (both identical) | lost (`/tmp/ts_results/`) |
| H22 | G1h 2.9B | 2026-08-07 | distinctness ρ(H21, H22) on overlap set | −0.054 | lost (`/tmp/ts_results/`) |
| H24 | Gemma4 e4b | 2026-08-14 | pre-RL DE (acc/tokens × 100) | 7.26 | no file cited (A0.2 eval) |
| H24 | G1i chatwrap | 2026-08-14 | pre-RL DE | 0.27 | no file cited (A0.2 eval) |
| H24 | step9b-e1 | 2026-08-14 | pre-RL DE | 0.15 | no file cited (A0.2 eval) |

## Coverage note

This table intentionally excludes `Untested` hypotheses (H1, H3–H7, H11, H13a/b, H14–H15,
H17, H19, H23, H25) — no numbers exist to index yet. See the "Model coverage table" near
the top of `HYPOTHESES.md` for the compressed per-checkpoint verdict grid this table
expands on.

<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python experiments/regenerate_results.py` -->

| H | Model | Date | Metric | Value | Result | Code |
|---|-------|------|--------|-------|--------|------|
| H8 | /home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth | 2026-08-17 | Mean IPC_total (L0,4,8,16,23) | 12.955 / 16 | `experiments/_common/results/adhoc/ipc.json` | `experiments/A0_state_probe/ipc_analysis.py` |
| H8 | /home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth | 2026-08-17 | Peak layer | L8 (15.307) | `experiments/_common/results/adhoc/ipc.json` | `experiments/A0_state_probe/ipc_analysis.py` |
