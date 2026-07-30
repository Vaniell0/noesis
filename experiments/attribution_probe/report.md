# H22 attribution probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 19  (attributable=8, unattributed=8, ambiguous=3)
- Feature dim: 768  (per-layer, per-head mean+std of WKV)
- Protocol: leave-one-out over 16 labelled items; ambiguous held out

## Aggregate (labelled subset, LOO)

- LOO F1:     **1.000**  (pilot target 0.75)
- LOO acc:    1.000
- Confusion:  TP=8  FP=0  TN=8  FN=0

## Per-item (labelled)

| id | category | y_true | y_pred | p_attributable |
|---|---|---|---|---|
| attr_01 | attributable | 1 | 1 | 0.873 |
| attr_02 | attributable | 1 | 1 | 0.998 |
| attr_03 | attributable | 1 | 1 | 0.921 |
| attr_04 | attributable | 1 | 1 | 0.994 |
| attr_05 | attributable | 1 | 1 | 0.999 |
| attr_06 | attributable | 1 | 1 | 1.000 |
| attr_07 | attributable | 1 | 1 | 0.991 |
| attr_08 | attributable | 1 | 1 | 0.999 |
| unattr_01 | unattributed | 0 | 0 | 0.000 |
| unattr_02 | unattributed | 0 | 0 | 0.003 |
| unattr_03 | unattributed | 0 | 0 | 0.000 |
| unattr_04 | unattributed | 0 | 0 | 0.015 |
| unattr_05 | unattributed | 0 | 0 | 0.008 |
| unattr_06 | unattributed | 0 | 0 | 0.019 |
| unattr_07 | unattributed | 0 | 0 | 0.025 |
| unattr_08 | unattributed | 0 | 0 | 0.001 |

## Ambiguous items (held out; diagnostic only)

Trained on all 16 labelled items, scored on ambiguous set:

| id | p_attributable |
|---|---|
| amb_01 | 0.994 |
| amb_02 | 0.967 |
| amb_03 | 0.114 |

A well-calibrated head would place ambiguous scores near 0.5; hard
commitments in either direction suggest the head is latching onto
surface features (e.g. presence of `I` / `my`) rather than genuine
attribution structure.

## Distinctness vs H21 (deferred)

H22 must remain distinct from H21 (premise-validity). The design metric
is ρ < 0.4 between head decisions on **overlap items** (invalid-premise +
unattributed / valid-premise + attributable). The current 19-item pilot
has no cross-labelled overlap; deferring to A1 dataset expansion where
cross-labels get authored explicitly. As a weaker proxy, compare feature
distributions across the two probes (both share the same pooling scheme)
in ``features.npz`` here and in ``../premise_validator/features.npz``.

## Notes

- 19-item pilot: F1 is noisy, use as *directional* signal, not verdict.
- Ambiguous scoring is the interesting output — where the model 'thinks'
  the borderline cases sit tells us where the classifier decision surface
  actually cuts.
