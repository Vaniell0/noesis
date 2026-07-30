# H20 aporia probe — pilot report

- Model: `rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 100
- Samples per item: 10, max_new_tokens=20, T=1.0, top_p=0.85
- Wall total: 18273.9 s

## Aggregate

### all (n=100)

- collapse_first (0=balanced, 1=collapsed): {'n': 100, 'mean': 0.6721204745087932, 'std': 0.3509371487193369, 'median': 0.822000080607206}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 100, 'mean': 0.5413015873015873, 'std': 0.46049049730733344, 'median': 0.6666666666666667}
- logit_gap:  {'n': 100, 'mean': 2.9925048828125, 'std': 2.711082330951977, 'median': 2.326171875}
- p(neither branch): {'n': 100, 'mean': 0.746, 'std': 0.27328373533746936, 'median': 0.8500000000000001}

### bounded_ambiguity (n=35)

- collapse_first (0=balanced, 1=collapsed): {'n': 35, 'mean': 0.6884583486621829, 'std': 0.30868893118045604, 'median': 0.8113131805844773}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 35, 'mean': 0.4775510204081633, 'std': 0.473938308725491, 'median': 0.33333333333333326}
- logit_gap:  {'n': 35, 'mean': 2.8414481026785716, 'std': 2.4621282626708423, 'median': 2.26171875}
- p(neither branch): {'n': 35, 'mean': 0.8542857142857142, 'std': 0.17943221106408666, 'median': 0.9}

### contested_facts (n=35)

- collapse_first (0=balanced, 1=collapsed): {'n': 35, 'mean': 0.6970445275501911, 'std': 0.34451398392044835, 'median': 0.8639191461901885}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 35, 'mean': 0.588843537414966, 'std': 0.44920250083914964, 'median': 1.0}
- logit_gap:  {'n': 35, 'mean': 3.399428013392857, 'std': 3.0065478041955283, 'median': 2.6171875}
- p(neither branch): {'n': 35, 'mean': 0.76, 'std': 0.25545477654008125, 'median': 0.9}

### underdetermined_inference (n=30)

- collapse_first (0=balanced, 1=collapsed): {'n': 30, 'mean': 0.6239815594482078, 'std': 0.3968974226951599, 'median': 0.8576316205731548}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 30, 'mean': 0.5602116402116403, 'std': 0.449038040025184, 'median': 0.7222222222222222}
- logit_gap:  {'n': 30, 'mean': 2.693994140625, 'std': 2.56372213341993, 'median': 2.569091796875}
- p(neither branch): {'n': 30, 'mean': 0.6033333333333334, 'std': 0.31778749013906904, 'median': 0.5}

## Per-item table

| id | cat | logit_gap | share_x_first | branch_x/y/none | collapse_cont |
|---|---|---|---|---|---|
| cf_01 | contested_facts | 9.930 | 1.00 | 10/0/0 | 1.00 |
| cf_02 | contested_facts | 4.371 | 0.99 | 3/2/5 | 0.20 |
| cf_03 | contested_facts | 0.477 | 0.38 | 1/0/9 | 1.00 |
| cf_04 | contested_facts | 3.297 | 0.04 | 1/0/9 | 1.00 |
| cf_05 | contested_facts | 7.984 | 1.00 | 0/0/10 | 0.00 |
| cf_06 | contested_facts | 0.000 | 0.50 | 0/9/1 | 1.00 |
| cf_07 | contested_facts | 2.141 | 0.89 | 2/0/8 | 1.00 |
| cf_08 | contested_facts | 1.195 | 0.77 | 2/1/7 | 0.33 |
| cf_09 | contested_facts | 1.099 | 0.75 | 0/0/10 | 0.00 |
| cf_10 | contested_facts | 10.875 | 0.00 | 0/1/9 | 1.00 |
| cf_11 | contested_facts | 8.195 | 1.00 | 1/1/8 | 0.00 |
| cf_12 | contested_facts | 1.695 | 0.16 | 0/1/9 | 1.00 |
| cf_13 | contested_facts | 7.312 | 1.00 | 4/0/6 | 1.00 |
| cf_14 | contested_facts | 3.324 | 0.03 | 1/0/9 | 1.00 |
| cf_15 | contested_facts | 3.195 | 0.96 | 1/5/4 | 0.67 |
| cf_16 | contested_facts | 0.000 | 0.50 | 0/0/10 | 0.00 |
| cf_17 | contested_facts | 3.477 | 0.97 | 4/0/6 | 1.00 |
| cf_18 | contested_facts | 1.469 | 0.81 | 4/3/3 | 0.14 |
| cf_19 | contested_facts | 1.484 | 0.82 | 0/1/9 | 1.00 |
| cf_20 | contested_facts | 2.352 | 0.91 | 0/0/10 | 0.00 |
| cf_21 | contested_facts | 0.492 | 0.62 | 2/1/7 | 0.33 |
| cf_22 | contested_facts | 2.617 | 0.93 | 0/0/10 | 0.00 |
| cf_23 | contested_facts | 4.484 | 0.01 | 1/1/8 | 0.00 |
| cf_24 | contested_facts | 5.859 | 1.00 | 1/4/5 | 0.60 |
| cf_25 | contested_facts | 0.000 | 0.50 | 1/2/7 | 0.33 |
| cf_26 | contested_facts | 8.131 | 1.00 | 2/0/8 | 1.00 |
| cf_27 | contested_facts | 1.756 | 0.15 | 1/0/9 | 1.00 |
| cf_28 | contested_facts | 0.906 | 0.71 | 0/0/10 | 0.00 |
| cf_29 | contested_facts | 3.891 | 0.02 | 0/1/9 | 1.00 |
| cf_30 | contested_facts | 0.312 | 0.42 | 1/0/9 | 1.00 |
| cf_31 | contested_facts | 4.391 | 0.99 | 5/0/5 | 1.00 |
| cf_32 | contested_facts | 1.625 | 0.84 | 0/0/10 | 0.00 |
| cf_33 | contested_facts | 3.006 | 0.95 | 0/1/9 | 1.00 |
| cf_34 | contested_facts | 0.000 | 0.50 | 0/0/10 | 0.00 |
| cf_35 | contested_facts | 7.637 | 1.00 | 2/0/8 | 1.00 |
| ba_01 | bounded_ambiguity | 1.154 | 0.24 | 0/0/10 | 0.00 |
| ba_02 | bounded_ambiguity | 1.758 | 0.85 | 0/0/10 | 0.00 |
| ba_03 | bounded_ambiguity | 2.828 | 0.94 | 0/0/10 | 0.00 |
| ba_04 | bounded_ambiguity | 4.430 | 0.01 | 1/0/9 | 1.00 |
| ba_05 | bounded_ambiguity | 2.359 | 0.09 | 3/0/7 | 1.00 |
| ba_06 | bounded_ambiguity | 3.477 | 0.03 | 0/0/10 | 0.00 |
| ba_07 | bounded_ambiguity | 7.531 | 0.00 | 0/1/9 | 1.00 |
| ba_08 | bounded_ambiguity | 2.301 | 0.91 | 1/0/9 | 1.00 |
| ba_09 | bounded_ambiguity | 0.000 | 0.50 | 0/0/10 | 0.00 |
| ba_10 | bounded_ambiguity | 5.883 | 1.00 | 1/0/9 | 1.00 |
| ba_11 | bounded_ambiguity | 4.881 | 0.99 | 0/0/10 | 0.00 |
| ba_12 | bounded_ambiguity | 3.195 | 0.04 | 0/3/7 | 1.00 |
| ba_13 | bounded_ambiguity | 1.871 | 0.87 | 2/1/7 | 0.33 |
| ba_14 | bounded_ambiguity | 0.531 | 0.63 | 0/1/9 | 1.00 |
| ba_15 | bounded_ambiguity | 4.062 | 0.98 | 6/0/4 | 1.00 |
| ba_16 | bounded_ambiguity | 9.219 | 1.00 | 1/0/9 | 1.00 |
| ba_17 | bounded_ambiguity | 0.703 | 0.33 | 2/1/7 | 0.33 |
| ba_18 | bounded_ambiguity | 3.418 | 0.97 | 5/0/5 | 1.00 |
| ba_19 | bounded_ambiguity | 0.031 | 0.51 | 0/0/10 | 0.00 |
| ba_20 | bounded_ambiguity | 0.234 | 0.56 | 2/0/8 | 1.00 |
| ba_21 | bounded_ambiguity | 0.604 | 0.35 | 1/0/9 | 1.00 |
| ba_22 | bounded_ambiguity | 8.234 | 1.00 | 0/0/10 | 0.00 |
| ba_23 | bounded_ambiguity | 1.062 | 0.26 | 2/0/8 | 1.00 |
| ba_24 | bounded_ambiguity | 2.262 | 0.91 | 0/0/10 | 0.00 |
| ba_25 | bounded_ambiguity | 0.404 | 0.60 | 0/0/10 | 0.00 |
| ba_26 | bounded_ambiguity | 1.081 | 0.25 | 1/1/8 | 0.00 |
| ba_27 | bounded_ambiguity | 1.898 | 0.13 | 0/0/10 | 0.00 |
| ba_28 | bounded_ambiguity | 1.303 | 0.79 | 0/3/7 | 1.00 |
| ba_29 | bounded_ambiguity | 8.859 | 1.00 | 0/0/10 | 0.00 |
| ba_30 | bounded_ambiguity | 2.275 | 0.91 | 0/0/10 | 0.00 |
| ba_31 | bounded_ambiguity | 1.719 | 0.85 | 1/2/7 | 0.33 |
| ba_32 | bounded_ambiguity | 3.301 | 0.96 | 0/0/10 | 0.00 |
| ba_33 | bounded_ambiguity | 3.570 | 0.97 | 0/0/10 | 0.00 |
| ba_34 | bounded_ambiguity | 2.234 | 0.10 | 0/2/8 | 1.00 |
| ba_35 | bounded_ambiguity | 0.777 | 0.31 | 1/6/3 | 0.71 |
| ui_01 | underdetermined_inference | 2.602 | 0.07 | 0/0/10 | 0.00 |
| ui_02 | underdetermined_inference | 1.656 | 0.84 | 2/2/6 | 0.00 |
| ui_03 | underdetermined_inference | 0.000 | 0.50 | 0/0/10 | 0.00 |
| ui_04 | underdetermined_inference | 2.281 | 0.09 | 2/3/5 | 0.20 |
| ui_05 | underdetermined_inference | 1.312 | 0.21 | 2/3/5 | 0.20 |
| ui_06 | underdetermined_inference | 2.938 | 0.95 | 0/0/10 | 0.00 |
| ui_07 | underdetermined_inference | 0.000 | 0.50 | 0/0/10 | 0.00 |
| ui_08 | underdetermined_inference | 5.391 | 1.00 | 8/0/2 | 1.00 |
| ui_09 | underdetermined_inference | 0.359 | 0.41 | 5/1/4 | 0.67 |
| ui_10 | underdetermined_inference | 0.000 | 0.50 | 0/0/10 | 0.00 |
| ui_11 | underdetermined_inference | 3.271 | 0.96 | 0/2/8 | 1.00 |
| ui_12 | underdetermined_inference | 6.398 | 1.00 | 9/0/1 | 1.00 |
| ui_13 | underdetermined_inference | 11.281 | 1.00 | 0/0/10 | 0.00 |
| ui_14 | underdetermined_inference | 0.734 | 0.68 | 0/6/4 | 1.00 |
| ui_15 | underdetermined_inference | 4.508 | 0.99 | 7/0/3 | 1.00 |
| ui_16 | underdetermined_inference | 0.160 | 0.46 | 1/0/9 | 1.00 |
| ui_17 | underdetermined_inference | 2.537 | 0.07 | 2/5/3 | 0.43 |
| ui_18 | underdetermined_inference | 3.047 | 0.95 | 0/7/3 | 1.00 |
| ui_19 | underdetermined_inference | 0.000 | 0.50 | 0/0/10 | 0.00 |
| ui_20 | underdetermined_inference | 6.891 | 1.00 | 0/6/4 | 1.00 |
| ui_21 | underdetermined_inference | 0.000 | 0.50 | 0/2/8 | 1.00 |
| ui_22 | underdetermined_inference | 4.383 | 0.99 | 2/4/4 | 0.33 |
| ui_23 | underdetermined_inference | 2.836 | 0.06 | 0/2/8 | 1.00 |
| ui_24 | underdetermined_inference | 3.906 | 0.02 | 8/1/1 | 0.78 |
| ui_25 | underdetermined_inference | 0.602 | 0.65 | 1/0/9 | 1.00 |
| ui_26 | underdetermined_inference | 3.698 | 0.98 | 5/0/5 | 1.00 |
| ui_27 | underdetermined_inference | 5.438 | 1.00 | 0/0/10 | 0.00 |
| ui_28 | underdetermined_inference | 3.005 | 0.05 | 0/8/2 | 1.00 |
| ui_29 | underdetermined_inference | 1.586 | 0.17 | 2/3/5 | 0.20 |
| ui_30 | underdetermined_inference | 0.000 | 0.50 | 0/8/2 | 1.00 |

## Notes on interpretation

- Well-formed aporia items should show:
  - `collapse_first` near 0 (first-token mass roughly balanced)
  - `collapse_cont` well below 1 (continuations don't uniformly commit to one branch)
  - non-zero `p(neither branch)` allowed (hedging / no-commit answers)
- High `logit_gap` with low `collapse_cont` = model expressed indecision downstream even though
  it had a favourite first token — the state carries the disagreement.
- Category-level pattern: contested_facts is expected to collapse more than
  bounded_ambiguity (semantic ambiguity has no pretraining preference), which in
  turn collapses more than underdetermined_inference (task ambiguity is deeper).
