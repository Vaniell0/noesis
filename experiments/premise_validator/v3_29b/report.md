# H21 premise-validity probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth`
- Items: 40 (valid=20, invalid=20)
- Feature dim: 2560  (per-layer, per-head mean+std of WKV)
- Split: stratified 32 train / 8 test  (seed=13)
- Head: 128→64→1 MLP, BCE, Adam lr=0.001, wd=0.001, epochs=500

## Aggregate

- Test F1 (best over training):  **1.000**  (pilot target 0.75)
- Test accuracy:                 1.000
- Confusion (test):  TP=4  FP=0  TN=4  FN=0

## Per-category on test set

| category | n | correct |
|---|---|---|
| category | 1 | 1/1 |
| impossible | 3 | 3/3 |
| valid | 4 | 4/4 |

## Per-item test predictions

| id | category | invalid_type | true | pred | p_valid |
|---|---|---|---|---|---|
| inv_cat_02 | invalid | category | 0 | 0 | 0.000 |
| inv_imp_01 | invalid | impossible | 0 | 0 | 0.000 |
| inv_imp_02 | invalid | impossible | 0 | 0 | 0.002 |
| inv_imp_05 | invalid | impossible | 0 | 0 | 0.281 |
| val_reason_01 | valid | - | 1 | 1 | 0.987 |
| val_reason_02 | valid | - | 1 | 1 | 0.999 |
| val_open_01 | valid | - | 1 | 1 | 0.999 |
| val_open_02 | valid | - | 1 | 1 | 1.000 |

## Notes

- 40-item pilot; head is overparameterised relative to sample size.
  Interpret F1 as a **necessary** signal for production: if the pooled state
  doesn't separate at 40 items, it won't at 200 either without richer features.
- Feature pooling drops most of the WKV rank structure. If the pilot fails,
  next attempt should try per-head Frobenius + top-k singular values, or
  concatenated head-diagonals.
- False positives on the valid set (`fp`) are the operational cost:
  flagging a well-formed query as suspicious causes needless aporia.
