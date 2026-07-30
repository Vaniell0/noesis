# H20 aporia probe — pilot report

- Model: `g1d-0.4b`
- Items: 30
- Samples per item: 10, max_new_tokens=20, T=1.0, top_p=0.85
- Wall total: 2832.0 s

## Aggregate

### all (n=30)

- collapse_first (0=balanced, 1=collapsed): {'n': 30, 'mean': 0.6543899805184203, 'std': 0.3622858869079591, 'median': 0.8162539490896092}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 30, 'mean': 0.4533333333333333, 'std': 0.4658405386045824, 'median': 0.19999999999999996}
- logit_gap:  {'n': 30, 'mean': 2.9876139322916666, 'std': 2.8915574681168987, 'median': 2.291015625}
- p(neither branch): {'n': 30, 'mean': 0.7766666666666667, 'std': 0.2894631045381931, 'median': 0.9}

### bounded_ambiguity (n=10)

- collapse_first (0=balanced, 1=collapsed): {'n': 10, 'mean': 0.7669892292902601, 'std': 0.292066406341833, 'median': 0.8578533753782669}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 10, 'mean': 0.5, 'std': 0.5, 'median': 0.5}
- logit_gap:  {'n': 10, 'mean': 3.1720703125, 'std': 2.137670900528724, 'median': 2.59375}
- p(neither branch): {'n': 10, 'mean': 0.93, 'std': 0.09000000000000001, 'median': 0.95}

### contested_facts (n=10)

- collapse_first (0=balanced, 1=collapsed): {'n': 10, 'mean': 0.696188808777392, 'std': 0.3434550593360374, 'median': 0.8591109205145202}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 10, 'mean': 0.6533333333333333, 'std': 0.4338970692072795, 'median': 1.0}
- logit_gap:  {'n': 10, 'mean': 4.136865234375, 'std': 3.8327548187929237, 'median': 2.71875}
- p(neither branch): {'n': 10, 'mean': 0.6799999999999999, 'std': 0.34583232931581165, 'median': 0.8500000000000001}

### underdetermined_inference (n=10)

- collapse_first (0=balanced, 1=collapsed): {'n': 10, 'mean': 0.4999919034876088, 'std': 0.39015961990253145, 'median': 0.6276651723488779}
- collapse_cont  (0=balanced, 1=collapsed): {'n': 10, 'mean': 0.20666666666666664, 'std': 0.3312602199681292, 'median': 0.0}
- logit_gap:  {'n': 10, 'mean': 1.65390625, 'std': 1.6401489264626437, 'median': 1.484375}
- p(neither branch): {'n': 10, 'mean': 0.72, 'std': 0.29597297173897485, 'median': 0.8}

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
