# H21 premise-validity probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 80 (valid=40, invalid=40)
- Feature dim: 768  (per-layer, per-head mean+std of WKV)
- Split: stratified 64 train / 16 test  (seed=13)
- Head: 128→64→1 MLP, BCE, Adam lr=0.001, wd=0.001, epochs=500

## Aggregate

- Test F1 (best over training):  **0.353**  (pilot target 0.75)
- Test accuracy:                 0.312
- Confusion (test):  TP=3  FP=6  TN=2  FN=5

## Per-category on test set

| category | n | correct |
|---|---|---|
| factual | 8 | 2/8 |
| valid | 8 | 3/8 |

## Per-item test predictions

| id | category | invalid_type | true | pred | p_valid |
|---|---|---|---|---|---|
| inv_fact_006_ctx | invalid | factual | 0 | 1 | 0.650 |
| inv_fact_011 | invalid | factual | 0 | 1 | 0.803 |
| inv_fact_012 | invalid | factual | 0 | 1 | 0.658 |
| inv_fact_013_ctx | invalid | factual | 0 | 1 | 1.000 |
| inv_fact_016 | invalid | factual | 0 | 0 | 0.232 |
| inv_fact_018 | invalid | factual | 0 | 1 | 0.970 |
| inv_fact_019 | invalid | factual | 0 | 0 | 0.073 |
| inv_fact_025_ctx | invalid | factual | 0 | 1 | 0.689 |
| val_fact_006 | valid | - | 1 | 0 | 0.001 |
| val_fact_011 | valid | - | 1 | 0 | 0.002 |
| val_fact_012_ctx | valid | - | 1 | 1 | 0.989 |
| val_fact_014_ctx | valid | - | 1 | 0 | 0.162 |
| val_fact_015 | valid | - | 1 | 1 | 1.000 |
| val_fact_016 | valid | - | 1 | 0 | 0.126 |
| val_fact_016_ctx | valid | - | 1 | 0 | 0.123 |
| val_fact_021 | valid | - | 1 | 1 | 0.568 |

## Notes

- 40-item pilot; head is overparameterised relative to sample size.
  Interpret F1 as a **necessary** signal for production: if the pooled state
  doesn't separate at 40 items, it won't at 200 either without richer features.
- Feature pooling drops most of the WKV rank structure. If the pilot fails,
  next attempt should try per-head Frobenius + top-k singular values, or
  concatenated head-diagonals.
- False positives on the valid set (`fp`) are the operational cost:
  flagging a well-formed query as suspicious causes needless aporia.
