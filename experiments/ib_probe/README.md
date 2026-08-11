# ib_probe — WKV state as Information Bottleneck channel

Tests whether the RWKV-7 WKV state acts as an efficient IB channel, as
formalized in `docs/state-and-reasoning.md §4`.

**Status:** script written, not yet run. No blocker other than time.

## What it measures

Two complementary probes (from `run.py`):

1. **Reconstruction probe** — upper-bounds I(X;Z). At each token t, trains a
   linear decoder from pooled WKV state Z_t to predict past token X_{t-k}
   at lags k ∈ {1,2,4,8,16,32,64}. Reports CE loss vs. unigram baseline.
   `fraction_explained = 1 - CE_probe / CE_unigram`. Decay curve shows how
   far back the state materially remembers.

2. **Downstream probe** — lower-bounds I(Z;Y). Uses the WKV state at end of
   a labelled prompt to predict a task label (same protocol as H21/H22 probes).
   Requires `--labels` JSONL.

**Ratio** downstream_quality / reconstruction_quality = state bit-efficiency:
how much of what the state stores is actually task-relevant.

## Hypothesis addressed

Not a named H directly. Validates the IB framing in `docs/state-and-reasoning.md §4`:
if the WKV state is a good IB channel, it should show high fraction_explained at
short lags (stores recent context) with graceful decay, and high downstream quality
relative to reconstruction (stores task-relevant, not just positional, information).

Comparing base vs. A1-fine-tuned model: if state-reg (`L_state`) improves the
IB channel quality (better downstream/reconstruction ratio), that is the
strongest mechanistic support for the L_state design.

## Run command

```bash
python experiments/ib_probe/run.py \
    --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
    --corpus experiments/ib_probe/corpus.txt \
    --out experiments/ib_probe/results_g1d_base \
    --lags 1,2,4,8,16,32,64
```

CPU-only, runs on i5-1235U. Corpus.txt contains 15 reference papers on
Information Bottleneck (Tishby et al., Alemi et al., Saxe et al.).

## Files

- `run.py` — probe implementation
- `corpus.txt` — reference corpus for reconstruction probe (IB papers)
