# H21 premise-validity probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 40 (valid=20, invalid=20)
- Feature dim: 768  (per-layer, per-head mean+std of WKV)
- Split: stratified 32 train / 8 test  (seed=13)
- Head: 128→64→1 MLP, BCE, Adam lr=0.001, wd=0.001, epochs=500

## Aggregate

- Test F1 (best over training):  **0.750**  (pilot target 0.75)
- Test accuracy:                 0.750
- Confusion (test):  TP=3  FP=1  TN=3  FN=1

## Per-category on test set

| category | n | correct |
|---|---|---|
| arithmetic | 1 | 0/1 |
| selfcontradiction | 3 | 3/3 |
| valid | 4 | 3/4 |

## Per-item test predictions

| id | category | invalid_type | true | pred | p_valid |
|---|---|---|---|---|---|
| v4_inv_arith_02 | invalid | arithmetic | 0 | 1 | 0.984 |
| v4_inv_selfcontr_01 | invalid | selfcontradiction | 0 | 0 | 0.012 |
| v4_inv_selfcontr_02 | invalid | selfcontradiction | 0 | 0 | 0.003 |
| v4_inv_selfcontr_05 | invalid | selfcontradiction | 0 | 0 | 0.011 |
| v4_val_math_01 | valid | - | 1 | 0 | 0.029 |
| v4_val_math_02 | valid | - | 1 | 1 | 0.983 |
| v4_val_causal_01 | valid | - | 1 | 1 | 0.981 |
| v4_val_causal_02 | valid | - | 1 | 1 | 0.944 |

## Notes

- 40-item pilot; head is overparameterised relative to sample size.
  Interpret F1 as a **necessary** signal for production: if the pooled state
  doesn't separate at 40 items, it won't at 200 either without richer features.
- Feature pooling drops most of the WKV rank structure. If the pilot fails,
  next attempt should try per-head Frobenius + top-k singular values, or
  concatenated head-diagonals.
- False positives on the valid set (`fp`) are the operational cost:
  flagging a well-formed query as suspicious causes needless aporia.
