# noesis

*νόησις* is Greek for the act of thinking itself, as distinct from the thing thought
(*noema*). noesis is a research notebook on RWKV-7: what a recurrent model can think
in its state, between two tokens, and what training it takes to get there.

There is no product here, no runtime, nothing to install. A reader looking for a
tool will not find one.

## TL;DR

- **The question.** Can a model compute in its recurrent state, between tokens,
  instead of in a chain of generated tokens?
- **What holds so far.** The state computes; it is not only a rolling summary of the
  prompt (H8, the one claim supported). Time spent in the state buys accuracy only
  where one pass is not enough (H10, partial). The state's geometry changes with
  training steps, and that change is readable (H25, H26, partial).
- **What does not hold yet.** The programme that would train for it, HZLK, is
  proposed: 22 sub-claims, 13 of them partial, and none supported.
- **The state of it.** Of 29 claims, one is supported, five partial, three
  contested. Eleven are written and not run. Nine are closed.

Nothing here is a result to build on yet. Each file names what would refute it.

## The wager

A model that reasons in generated text pays for every step out of its context:
it writes the step as tokens, then reads them back. A recurrent state has no
such bill: a step between two tokens is a state update, and the context does
not grow.

That there is something to save is measured (ZLK.1): on G1i 2.9B a chain of
thought spends 47% of its tokens after its answer is already right, on one
family of items. That the state cannot yet take the work is measured too
(ZLK.2): in an untrained model, re-reading, extra ticks and pauses do not beat
answering at once.

So the bet is narrow. Not that time in the state is free, but that training can
make it work: a second pretraining that makes what the weights hold readable,
then a loop that reads it, one tick at a time (HZLK). Whether that is cheaper
than thinking in tokens, and by how much, is not measured here. That is the
question the notebook is for.

## The regime of every number

The measurements are small. RWKV-7 G1 at 0.4B and 2.9B, pretrained and
fine-tuned; a toy model trained from scratch; a laptop CPU and one cloud machine.
A number on these does not carry to a larger model without a run.

## Where things are

- `hypotheses/README.md`: the frame, how a claim is written and checked, and
  the status table of every claim.
- `hypotheses/HZLK.md`: the programme. A second pretraining that makes what
  the weights already hold readable, and a loop that reads it through a
  self-fed latent, one tick at a time.
- `hypotheses/records/`: the record of each claim, its evidence with dates and
  the files it rests on; `HZLK.runs.yml` there holds the setup and the reading
  of every run of the programme.
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
| WAGER | 2 | a bet that cannot be tested at this scale |
| closed | 7 | retracted, withdrawn, superseded or retired |

Eight of the files are not published: the closed ones and the one about
fine-tuning on a domain. They keep a row in the status table and a record.

## Reading a claim

Every open claim carries an estimator (the number that would show it) and a
refuter (the outcome that would kill it), written before the run. A status
moves on a run, or on a dated decision, and the record says which.

Some links name files that are not published. They are marked missing with
the date; the claim then rests on what was read from them at that time.

A reader with ten minutes: `hypotheses/H8.md`, `hypotheses/H10.md`, then the
table and §1 of `hypotheses/HZLK.md`.

## Checks

Two scripts keep the tree honest:

```
python experiments/regenerate_hyp_index.py --check   # every link alive, every status a known word
python experiments/bayes_lite.py                     # every record beside its file, linted
```

## Licence

Apache 2.0, see `LICENSE`.
