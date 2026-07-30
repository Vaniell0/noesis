# H21 premise-validity probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 40 (valid=20, invalid=20)
- Feature dim: 768  (per-layer, per-head mean+std of WKV)
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
| inv_imp_02 | invalid | impossible | 0 | 0 | 0.074 |
| inv_imp_05 | invalid | impossible | 0 | 0 | 0.042 |
| val_reason_01 | valid | - | 1 | 1 | 0.973 |
| val_reason_02 | valid | - | 1 | 1 | 0.998 |
| val_open_01 | valid | - | 1 | 1 | 0.998 |
| val_open_02 | valid | - | 1 | 1 | 0.999 |

## Leave-one-out (honest re-estimate)

The 8-item stratified split above landed on the easy invalid types
(`category` + `impossible`) and missed `factual` + `counterfactual`
entirely. LOO over all 40 items gives a more honest number.

- **LOO F1: 0.789**  acc: 0.800  (still above pilot target 0.75)
- Confusion (LOO):  TP=15  FP=3  TN=17  FN=5

Per invalid-type recall (on the 20 invalid items):

| invalid_type    | recall |
|-----------------|--------|
| category        | 5/5    |
| impossible      | 5/5    |
| counterfactual  | 4/5    |
| factual         | 3/5    |

`category` and `impossible` are separated cleanly in the WKV state.
`factual` invalids share structure with valid factual claims — only
truth-value differs, and the base 0.4B model may not know the fact
either. This is the operational risk: false-fact detection needs the
model to actually know what's true, not just the state to look funny.

## Notes

- 40-item pilot; head is overparameterised relative to sample size.
  Interpret single-split F1=1.000 as a lucky test-fold draw; LOO F1=0.789
  is the number to carry forward. Pilot passes the H21 threshold on
  LOO but the classifier is running near the ceiling of what pooled
  mean+std features can achieve at this dataset size.
- Feature pooling drops most of the WKV rank structure. Next attempt:
  per-head Frobenius + top-k singular values, or concatenated
  head-diagonals — see if `factual` recall lifts above 3/5.
- False positives on the valid set (`fp`) are the operational cost:
  flagging a well-formed query as suspicious causes needless aporia.
  LOO shows 3/20 valid-flagged-as-invalid = 15 % FP rate, well above
  the 3 % production target. Not a pilot blocker but a real gap.

## Followup dataset expansion

Toward 200-item production:
- Mine TruthfulQA misconceptions (~800 available) for invalid `factual`
  items — that's the axis LOO shows we underperform on.
- Mine (QA)² / CREPE for counterfactual and impossible premises.
- Author 25 valid items per shape (reasoning / factual / procedure /
  open) from A0.2 tasks + query history.
