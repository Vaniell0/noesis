# H22 attribution probe — pilot report

- Model: `/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth`
- Items: 243  (attributable=120, unattributed=120, ambiguous=3)
- Feature dim: 768  (per-layer, per-head mean+std of WKV)
- Protocol: leave-one-out over 16 labelled items; ambiguous held out

## Aggregate (labelled subset, LOO)

- LOO F1:     **0.947**  (pilot target 0.75)
- LOO acc:    0.946
- Confusion:  TP=115  FP=8  TN=112  FN=5

## Per-item (labelled)

| id | category | y_true | y_pred | p_attributable |
|---|---|---|---|---|
| attr_01 | attributable | 1 | 1 | 0.889 |
| attr_02 | attributable | 1 | 1 | 0.998 |
| attr_03 | attributable | 1 | 0 | 0.051 |
| attr_04 | attributable | 1 | 1 | 0.999 |
| attr_05 | attributable | 1 | 1 | 1.000 |
| attr_06 | attributable | 1 | 1 | 1.000 |
| attr_07 | attributable | 1 | 1 | 0.938 |
| attr_08 | attributable | 1 | 1 | 1.000 |
| unattr_01 | unattributed | 0 | 0 | 0.000 |
| unattr_02 | unattributed | 0 | 0 | 0.001 |
| unattr_03 | unattributed | 0 | 0 | 0.000 |
| unattr_04 | unattributed | 0 | 0 | 0.000 |
| unattr_05 | unattributed | 0 | 0 | 0.001 |
| unattr_06 | unattributed | 0 | 0 | 0.005 |
| unattr_07 | unattributed | 0 | 0 | 0.038 |
| unattr_08 | unattributed | 0 | 0 | 0.000 |
| attr_009 | attributable | 1 | 1 | 0.862 |
| unattr_009 | unattributed | 0 | 0 | 0.053 |
| unattr_010 | unattributed | 0 | 1 | 0.641 |
| attr_010 | attributable | 1 | 1 | 1.000 |
| attr_011 | attributable | 1 | 1 | 1.000 |
| attr_012 | attributable | 1 | 1 | 1.000 |
| attr_013 | attributable | 1 | 1 | 1.000 |
| unattr_011 | unattributed | 0 | 0 | 0.000 |
| unattr_012 | unattributed | 0 | 0 | 0.000 |
| unattr_013 | unattributed | 0 | 0 | 0.000 |
| attr_014 | attributable | 1 | 1 | 0.983 |
| attr_015 | attributable | 1 | 1 | 1.000 |
| unattr_014 | unattributed | 0 | 0 | 0.000 |
| unattr_015 | unattributed | 0 | 0 | 0.002 |
| unattr_016 | unattributed | 0 | 0 | 0.025 |
| attr_016 | attributable | 1 | 1 | 0.991 |
| unattr_017 | unattributed | 0 | 0 | 0.000 |
| attr_017 | attributable | 1 | 1 | 1.000 |
| unattr_018 | unattributed | 0 | 0 | 0.001 |
| attr_018 | attributable | 1 | 1 | 0.999 |
| attr_019 | attributable | 1 | 1 | 0.997 |
| attr_020 | attributable | 1 | 0 | 0.288 |
| attr_021 | attributable | 1 | 1 | 0.999 |
| attr_022 | attributable | 1 | 1 | 1.000 |
| attr_023 | attributable | 1 | 1 | 0.712 |
| attr_024 | attributable | 1 | 1 | 1.000 |
| attr_025 | attributable | 1 | 1 | 0.999 |
| unattr_019 | unattributed | 0 | 0 | 0.000 |
| attr_026 | attributable | 1 | 0 | 0.034 |
| unattr_020 | unattributed | 0 | 0 | 0.000 |
| attr_027 | attributable | 1 | 1 | 1.000 |
| attr_028 | attributable | 1 | 1 | 1.000 |
| attr_029 | attributable | 1 | 1 | 0.987 |
| attr_030 | attributable | 1 | 1 | 1.000 |
| unattr_021 | unattributed | 0 | 0 | 0.006 |
| attr_031 | attributable | 1 | 1 | 0.994 |
| attr_032 | attributable | 1 | 1 | 1.000 |
| attr_033 | attributable | 1 | 1 | 1.000 |
| attr_034 | attributable | 1 | 1 | 1.000 |
| attr_035 | attributable | 1 | 1 | 1.000 |
| attr_036 | attributable | 1 | 1 | 0.992 |
| unattr_022 | unattributed | 0 | 0 | 0.001 |
| attr_037 | attributable | 1 | 1 | 0.906 |
| attr_038 | attributable | 1 | 1 | 0.666 |
| unattr_023 | unattributed | 0 | 1 | 0.507 |
| unattr_024 | unattributed | 0 | 0 | 0.000 |
| attr_039 | attributable | 1 | 1 | 0.998 |
| attr_040 | attributable | 1 | 1 | 0.998 |
| attr_041 | attributable | 1 | 1 | 0.999 |
| attr_042 | attributable | 1 | 1 | 0.638 |
| attr_043 | attributable | 1 | 1 | 1.000 |
| attr_044 | attributable | 1 | 1 | 1.000 |
| unattr_025 | unattributed | 0 | 0 | 0.000 |
| attr_045 | attributable | 1 | 1 | 0.995 |
| attr_046 | attributable | 1 | 1 | 1.000 |
| attr_047 | attributable | 1 | 1 | 0.984 |
| attr_048 | attributable | 1 | 1 | 0.808 |
| attr_049 | attributable | 1 | 1 | 0.999 |
| attr_050 | attributable | 1 | 1 | 1.000 |
| attr_051 | attributable | 1 | 1 | 1.000 |
| attr_052 | attributable | 1 | 1 | 1.000 |
| attr_053 | attributable | 1 | 1 | 0.985 |
| attr_054 | attributable | 1 | 1 | 0.962 |
| attr_055 | attributable | 1 | 1 | 1.000 |
| attr_056 | attributable | 1 | 1 | 1.000 |
| unattr_026 | unattributed | 0 | 0 | 0.009 |
| unattr_027 | unattributed | 0 | 0 | 0.000 |
| unattr_028 | unattributed | 0 | 0 | 0.000 |
| unattr_029 | unattributed | 0 | 0 | 0.000 |
| attr_057 | attributable | 1 | 1 | 0.896 |
| attr_058 | attributable | 1 | 1 | 1.000 |
| attr_059 | attributable | 1 | 1 | 0.930 |
| unattr_030 | unattributed | 0 | 0 | 0.000 |
| unattr_031 | unattributed | 0 | 0 | 0.000 |
| unattr_032 | unattributed | 0 | 0 | 0.005 |
| attr_060 | attributable | 1 | 1 | 1.000 |
| attr_061 | attributable | 1 | 1 | 0.785 |
| unattr_033 | unattributed | 0 | 0 | 0.000 |
| attr_062 | attributable | 1 | 1 | 1.000 |
| attr_063 | attributable | 1 | 1 | 0.995 |
| attr_064 | attributable | 1 | 1 | 0.983 |
| attr_065 | attributable | 1 | 1 | 0.997 |
| attr_066 | attributable | 1 | 1 | 1.000 |
| unattr_034 | unattributed | 0 | 0 | 0.000 |
| unattr_035 | unattributed | 0 | 0 | 0.000 |
| attr_067 | attributable | 1 | 1 | 1.000 |
| attr_068 | attributable | 1 | 1 | 1.000 |
| attr_069 | attributable | 1 | 1 | 1.000 |
| unattr_036 | unattributed | 0 | 0 | 0.000 |
| attr_070 | attributable | 1 | 1 | 0.999 |
| attr_071 | attributable | 1 | 1 | 0.999 |
| attr_072 | attributable | 1 | 1 | 1.000 |
| attr_073 | attributable | 1 | 1 | 1.000 |
| attr_074 | attributable | 1 | 1 | 1.000 |
| unattr_037 | unattributed | 0 | 0 | 0.000 |
| attr_075 | attributable | 1 | 1 | 1.000 |
| attr_076 | attributable | 1 | 1 | 1.000 |
| attr_077 | attributable | 1 | 1 | 0.987 |
| attr_078 | attributable | 1 | 1 | 1.000 |
| attr_079 | attributable | 1 | 1 | 1.000 |
| attr_080 | attributable | 1 | 1 | 0.998 |
| attr_081 | attributable | 1 | 1 | 1.000 |
| attr_082 | attributable | 1 | 1 | 0.650 |
| attr_083 | attributable | 1 | 0 | 0.046 |
| attr_084 | attributable | 1 | 1 | 0.836 |
| attr_085 | attributable | 1 | 1 | 1.000 |
| attr_086 | attributable | 1 | 1 | 1.000 |
| attr_087 | attributable | 1 | 1 | 0.995 |
| attr_088 | attributable | 1 | 1 | 0.999 |
| attr_089 | attributable | 1 | 1 | 1.000 |
| attr_090 | attributable | 1 | 1 | 0.959 |
| attr_091 | attributable | 1 | 1 | 0.991 |
| attr_092 | attributable | 1 | 1 | 0.982 |
| attr_093 | attributable | 1 | 1 | 1.000 |
| attr_094 | attributable | 1 | 1 | 0.853 |
| attr_095 | attributable | 1 | 1 | 1.000 |
| attr_096 | attributable | 1 | 1 | 1.000 |
| unattr_038 | unattributed | 0 | 0 | 0.002 |
| attr_097 | attributable | 1 | 1 | 0.978 |
| attr_098 | attributable | 1 | 1 | 1.000 |
| attr_099 | attributable | 1 | 1 | 1.000 |
| attr_100 | attributable | 1 | 1 | 1.000 |
| attr_101 | attributable | 1 | 1 | 1.000 |
| attr_102 | attributable | 1 | 1 | 1.000 |
| attr_103 | attributable | 1 | 1 | 0.999 |
| attr_104 | attributable | 1 | 1 | 1.000 |
| attr_105 | attributable | 1 | 0 | 0.003 |
| unattr_039 | unattributed | 0 | 0 | 0.000 |
| attr_106 | attributable | 1 | 1 | 1.000 |
| attr_107 | attributable | 1 | 1 | 1.000 |
| unattr_040 | unattributed | 0 | 0 | 0.000 |
| attr_108 | attributable | 1 | 1 | 1.000 |
| attr_109 | attributable | 1 | 1 | 1.000 |
| attr_110 | attributable | 1 | 1 | 0.998 |
| attr_111 | attributable | 1 | 1 | 1.000 |
| attr_112 | attributable | 1 | 1 | 0.999 |
| attr_113 | attributable | 1 | 1 | 0.998 |
| attr_114 | attributable | 1 | 1 | 0.961 |
| attr_115 | attributable | 1 | 1 | 1.000 |
| attr_116 | attributable | 1 | 1 | 0.999 |
| attr_117 | attributable | 1 | 1 | 0.986 |
| attr_118 | attributable | 1 | 1 | 1.000 |
| attr_119 | attributable | 1 | 1 | 0.930 |
| attr_120 | attributable | 1 | 1 | 0.997 |
| unattr_041 | unattributed | 0 | 0 | 0.000 |
| unattr_042 | unattributed | 0 | 1 | 0.646 |
| unattr_043 | unattributed | 0 | 0 | 0.000 |
| unattr_044 | unattributed | 0 | 0 | 0.000 |
| unattr_045 | unattributed | 0 | 0 | 0.000 |
| unattr_046 | unattributed | 0 | 0 | 0.000 |
| unattr_047 | unattributed | 0 | 0 | 0.000 |
| unattr_048 | unattributed | 0 | 0 | 0.001 |
| unattr_049 | unattributed | 0 | 0 | 0.000 |
| unattr_050 | unattributed | 0 | 0 | 0.105 |
| unattr_051 | unattributed | 0 | 0 | 0.000 |
| unattr_052 | unattributed | 0 | 0 | 0.000 |
| unattr_053 | unattributed | 0 | 0 | 0.075 |
| unattr_054 | unattributed | 0 | 0 | 0.003 |
| unattr_055 | unattributed | 0 | 0 | 0.000 |
| unattr_056 | unattributed | 0 | 0 | 0.000 |
| unattr_057 | unattributed | 0 | 0 | 0.001 |
| unattr_058 | unattributed | 0 | 0 | 0.006 |
| unattr_059 | unattributed | 0 | 0 | 0.002 |
| unattr_060 | unattributed | 0 | 1 | 0.682 |
| unattr_061 | unattributed | 0 | 0 | 0.000 |
| unattr_062 | unattributed | 0 | 0 | 0.006 |
| unattr_063 | unattributed | 0 | 0 | 0.012 |
| unattr_064 | unattributed | 0 | 0 | 0.000 |
| unattr_065 | unattributed | 0 | 0 | 0.027 |
| unattr_066 | unattributed | 0 | 0 | 0.000 |
| unattr_067 | unattributed | 0 | 0 | 0.003 |
| unattr_068 | unattributed | 0 | 1 | 0.997 |
| unattr_069 | unattributed | 0 | 0 | 0.000 |
| unattr_070 | unattributed | 0 | 0 | 0.004 |
| unattr_071 | unattributed | 0 | 0 | 0.000 |
| unattr_072 | unattributed | 0 | 0 | 0.000 |
| unattr_073 | unattributed | 0 | 0 | 0.002 |
| unattr_074 | unattributed | 0 | 1 | 0.682 |
| unattr_075 | unattributed | 0 | 0 | 0.000 |
| unattr_076 | unattributed | 0 | 0 | 0.011 |
| unattr_077 | unattributed | 0 | 0 | 0.000 |
| unattr_078 | unattributed | 0 | 0 | 0.000 |
| unattr_079 | unattributed | 0 | 0 | 0.000 |
| unattr_080 | unattributed | 0 | 0 | 0.000 |
| unattr_081 | unattributed | 0 | 0 | 0.000 |
| unattr_082 | unattributed | 0 | 0 | 0.000 |
| unattr_083 | unattributed | 0 | 0 | 0.001 |
| unattr_084 | unattributed | 0 | 0 | 0.000 |
| unattr_085 | unattributed | 0 | 0 | 0.000 |
| unattr_086 | unattributed | 0 | 0 | 0.344 |
| unattr_087 | unattributed | 0 | 0 | 0.000 |
| unattr_088 | unattributed | 0 | 0 | 0.000 |
| unattr_089 | unattributed | 0 | 0 | 0.000 |
| unattr_090 | unattributed | 0 | 0 | 0.000 |
| unattr_091 | unattributed | 0 | 0 | 0.000 |
| unattr_092 | unattributed | 0 | 0 | 0.152 |
| unattr_093 | unattributed | 0 | 0 | 0.000 |
| unattr_094 | unattributed | 0 | 0 | 0.000 |
| unattr_095 | unattributed | 0 | 0 | 0.004 |
| unattr_096 | unattributed | 0 | 0 | 0.000 |
| unattr_097 | unattributed | 0 | 0 | 0.000 |
| unattr_098 | unattributed | 0 | 0 | 0.002 |
| unattr_099 | unattributed | 0 | 0 | 0.003 |
| unattr_100 | unattributed | 0 | 1 | 0.997 |
| unattr_101 | unattributed | 0 | 0 | 0.001 |
| unattr_102 | unattributed | 0 | 0 | 0.001 |
| unattr_103 | unattributed | 0 | 0 | 0.000 |
| unattr_104 | unattributed | 0 | 0 | 0.000 |
| unattr_105 | unattributed | 0 | 0 | 0.000 |
| unattr_106 | unattributed | 0 | 0 | 0.002 |
| unattr_107 | unattributed | 0 | 0 | 0.000 |
| unattr_108 | unattributed | 0 | 0 | 0.000 |
| unattr_109 | unattributed | 0 | 0 | 0.014 |
| unattr_110 | unattributed | 0 | 0 | 0.001 |
| unattr_111 | unattributed | 0 | 0 | 0.000 |
| unattr_112 | unattributed | 0 | 1 | 0.996 |
| unattr_113 | unattributed | 0 | 0 | 0.000 |
| unattr_114 | unattributed | 0 | 0 | 0.000 |
| unattr_115 | unattributed | 0 | 0 | 0.000 |
| unattr_116 | unattributed | 0 | 0 | 0.188 |
| unattr_117 | unattributed | 0 | 0 | 0.000 |
| unattr_118 | unattributed | 0 | 0 | 0.003 |
| unattr_119 | unattributed | 0 | 0 | 0.010 |
| unattr_120 | unattributed | 0 | 0 | 0.001 |

## Ambiguous items (held out; diagnostic only)

Trained on all 16 labelled items, scored on ambiguous set:

| id | p_attributable |
|---|---|
| amb_01 | 0.126 |
| amb_02 | 0.892 |
| amb_03 | 0.002 |

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
