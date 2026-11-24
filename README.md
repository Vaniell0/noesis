# noesis

Research notes on RWKV-7. One question runs through them: can a model compute
in its recurrent state, between two tokens, and what does training have to do
to make that possible?

There is no product here, no runtime, nothing to install. A reader looking for
a tool will not find one.

## Where things are

- `hypotheses/README.md`: the frame, how a claim is written and checked, and
  the status table of every claim.
- `hypotheses/HZLK.md`: the programme. A second pretraining that makes what
  the weights already hold readable, and a loop that reads it through a
  self-fed latent, one tick at a time.
- `hypotheses/records/`: the record of each claim, its evidence with dates and
  the files it rests on.
- `experiments/`: the probes and the results they wrote.
- `docs/`: the notes the hypotheses cite.
- `training/`: three modules that the probes import.

## What is established

As of 2026-11-25, of 29 claims:

| status | count | meaning |
|---|---|---|
| SUPPORTED | 1 | measured, the refuter did not fire |
| PARTIAL | 5 | holds in part of the regime measured |
| CONTESTED | 3 | runs point both ways |
| PROPOSED | 11 | written with an estimator and a refuter, not run |
| closed | 9 | retracted, withdrawn, superseded, retired, or a wager |

Nothing here is a result to build on. Each file names what would refute it.

## Reading a claim

Every open claim carries an estimator (the number that would show it) and a
refuter (the outcome that would kill it), written before the run. A status
moves on a run, or on a dated decision, and the record says which.

Some links name files that are not published. They are marked missing with
the date; the claim then rests on what was read from them at that time.

## Checks

Two scripts keep the tree honest:

```
python experiments/regenerate_hyp_index.py --check   # every link alive, every status a known word
python experiments/bayes_lite.py                     # every record beside its file, linted
```

## Licence

Apache 2.0, see `LICENSE`.
