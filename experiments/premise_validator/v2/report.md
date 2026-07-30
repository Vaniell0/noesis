# H21 premise-validity probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 280 (valid=140, invalid=140)
- Feature dim: 768  (per-layer, per-head mean+std of WKV)
- Split: stratified 224 train / 56 test  (seed=13)
- Head: 128→64→1 MLP, BCE, Adam lr=0.001, wd=0.001, epochs=500

## Aggregate

- Test F1 (best over training):  **0.600**  (pilot target 0.75)
- Test accuracy:                 0.571
- Confusion (test):  TP=18  FP=14  TN=14  FN=10

## Per-category on test set

| category | n | correct |
|---|---|---|
| category | 2 | 2/2 |
| counterfactual | 1 | 0/1 |
| factual | 24 | 12/24 |
| impossible | 1 | 0/1 |
| valid | 28 | 18/28 |

## Per-item test predictions

| id | category | invalid_type | true | pred | p_valid |
|---|---|---|---|---|---|
| inv_fact_04 | invalid | factual | 0 | 0 | 0.065 |
| inv_cat_01 | invalid | category | 0 | 0 | 0.000 |
| inv_cat_02 | invalid | category | 0 | 0 | 0.000 |
| inv_cf_02 | invalid | counterfactual | 0 | 1 | 0.505 |
| inv_imp_05 | invalid | impossible | 0 | 1 | 0.755 |
| val_procedure_03 | valid | - | 1 | 1 | 0.750 |
| val_procedure_04 | valid | - | 1 | 1 | 0.955 |
| val_open_02 | valid | - | 1 | 1 | 0.998 |
| inv_fact_006 | invalid | factual | 0 | 0 | 0.019 |
| inv_fact_007 | invalid | factual | 0 | 1 | 1.000 |
| inv_fact_020 | invalid | factual | 0 | 1 | 0.920 |
| inv_fact_022 | invalid | factual | 0 | 0 | 0.097 |
| inv_fact_024 | invalid | factual | 0 | 1 | 1.000 |
| inv_fact_028 | invalid | factual | 0 | 0 | 0.000 |
| inv_fact_034 | invalid | factual | 0 | 1 | 0.628 |
| inv_fact_035 | invalid | factual | 0 | 1 | 0.994 |
| inv_fact_037 | invalid | factual | 0 | 1 | 1.000 |
| inv_fact_038 | invalid | factual | 0 | 1 | 0.987 |
| inv_fact_039 | invalid | factual | 0 | 1 | 0.603 |
| inv_fact_055 | invalid | factual | 0 | 1 | 0.710 |
| inv_fact_059 | invalid | factual | 0 | 0 | 0.492 |
| inv_fact_070 | invalid | factual | 0 | 0 | 0.000 |
| inv_fact_072 | invalid | factual | 0 | 0 | 0.037 |
| inv_fact_078 | invalid | factual | 0 | 1 | 0.971 |
| inv_fact_084 | invalid | factual | 0 | 1 | 0.976 |
| inv_fact_095 | invalid | factual | 0 | 0 | 0.002 |
| inv_fact_108 | invalid | factual | 0 | 0 | 0.000 |
| inv_fact_109 | invalid | factual | 0 | 0 | 0.000 |
| inv_fact_114 | invalid | factual | 0 | 1 | 0.987 |
| inv_fact_122 | invalid | factual | 0 | 0 | 0.224 |
| inv_fact_124 | invalid | factual | 0 | 0 | 0.369 |
| val_fact_007 | valid | - | 1 | 0 | 0.031 |
| val_fact_015 | valid | - | 1 | 1 | 1.000 |
| val_fact_018 | valid | - | 1 | 1 | 0.885 |
| val_fact_021 | valid | - | 1 | 0 | 0.002 |
| val_fact_027 | valid | - | 1 | 1 | 0.996 |
| val_fact_032 | valid | - | 1 | 0 | 0.011 |
| val_fact_037 | valid | - | 1 | 1 | 0.865 |
| val_fact_038 | valid | - | 1 | 1 | 0.506 |
| val_fact_045 | valid | - | 1 | 1 | 0.636 |
| val_fact_051 | valid | - | 1 | 1 | 0.897 |
| val_fact_055 | valid | - | 1 | 1 | 1.000 |
| val_fact_059 | valid | - | 1 | 0 | 0.000 |
| val_fact_060 | valid | - | 1 | 0 | 0.007 |
| val_fact_066 | valid | - | 1 | 0 | 0.030 |
| val_fact_074 | valid | - | 1 | 1 | 0.998 |
| val_fact_076 | valid | - | 1 | 1 | 0.999 |
| val_fact_081 | valid | - | 1 | 0 | 0.429 |
| val_fact_082 | valid | - | 1 | 1 | 0.910 |
| val_fact_089 | valid | - | 1 | 0 | 0.295 |
| val_fact_097 | valid | - | 1 | 1 | 0.967 |
| val_fact_099 | valid | - | 1 | 0 | 0.008 |
| val_fact_108 | valid | - | 1 | 1 | 1.000 |
| val_fact_116 | valid | - | 1 | 1 | 1.000 |
| val_fact_118 | valid | - | 1 | 1 | 1.000 |
| val_fact_123 | valid | - | 1 | 0 | 0.120 |

## Notes

- 40-item pilot; head is overparameterised relative to sample size.
  Interpret F1 as a **necessary** signal for production: if the pooled state
  doesn't separate at 40 items, it won't at 200 either without richer features.
- Feature pooling drops most of the WKV rank structure. If the pilot fails,
  next attempt should try per-head Frobenius + top-k singular values, or
  concatenated head-diagonals.
- False positives on the valid set (`fp`) are the operational cost:
  flagging a well-formed query as suspicious causes needless aporia.
