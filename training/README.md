# training

What is left of the training pipeline: the three modules the probes under
`experiments/` import. Nothing else of the pipeline is published.

- `state_reg.py`: the state regulariser and the state read-out helpers.
- `datasets.py`: the corpus readers.
- `regenerate_corpus_index.py`: builds the index that `datasets.py` reads.

They read corpora and checkpoints that are not in this repository.
