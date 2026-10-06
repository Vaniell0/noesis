# Cipher tasks — decoding over useful text

`training/_common/ciphers.py` (25 invertible codecs), `training/scripts/gen_ciphers.py` (generator,
registered as stage `ciphers`), `training/tests/test_ciphers.py`. Catalog entries: `ciphers_v1_train`
(7 500 rows, 2 500 contents) and `ciphers_v1_eval` (750 rows, 250 contents), generated over the answer
sentences of `hh_rlhf` read through the dataset catalog (`docs/datasets.md`), so a personal-derived set
cannot be used as payload. Nothing has been trained or evaluated on them yet.

## Why ciphers

Decoding has an exact answer, a crisp rule and a key that can be given or hidden. Hidden-key tasks are
hypothesis search with an exact verifier. Published work found the same asymmetry we care about: models
translate ciphered text well and reason in it poorly (arXiv 2510.09714: 28 ciphers, up to 10 models,
prompted and fine-tuned; the gap tracks how common a cipher is in pretraining, and fine-tuning data helps
slowly), and CipherBank (arXiv 2504.19093) gives a taxonomy (substitution, transposition, custom) and
baselines. Variety, not one cipher, is the lever.

## What is in it

| family | ciphers | key |
|---|---|---|
| shift / reversal | caesar, atbash | shift / none |
| polyalphabetic | vigenere | keyword |
| letter → symbol tables | 15 alphabets: katakana, cyrillic, hebrew (1 token per symbol), runic, ogham, glagolitic, cherokee, braille, tifinagh, yijing (~3), shavian, deseret, ugaritic, linear_b, phaistos (4) | random permutation, shown as a table |
| fractionating | polybius6, tap6, bacon (7 symbol pairs), bifid6, adfgvx | random 6x6 square / keyword |
| transposition | railfence, columnar | rails / keyword |

Token cost per symbol is measured with the World tokenizer (`measure_costs`, re-checked by a test), so a
grid or a context budget can pick alphabets by cost. Unicode blocks were checked assigned in Unicode 15.1.

## Rows

`{id, category: cipher_<name>, level 1-6, prompt, answer, rubric {exact}, content_id, view, mode, alphabet_cost}`
(the matrix-task shape plus three fields). Rows with one `content_id` are different ciphers of one text:
matched views for a representation-level objective and for the format-versus-content control. `mode` is
`known` (key in the prompt) or `unknown` (only the cipher name, with enough text for the key to be
recoverable: caesar/atbash ≥ 10 characters, railfence 30, columnar 60, vigenere 100, symbol tables 120).
The generator re-decodes every row's own answer before writing it. Train and eval contents are split by
`md5(content_id) % 10`, so no text is in both.

## Not covered

Straddling checkerboard, Nihilist and the other classical systems; payloads in other languages (the
generators are English-only, so FLORES-200, Tatoeba and UDHR are catalogued as `planned` sources);
keyboard-layout errors, which need non-English text; and an invented language with a written grammar,
the "language nobody knows" task.
