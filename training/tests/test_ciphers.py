"""Every cipher must round-trip exactly on random text and random keys; alphabets must be real."""
from __future__ import annotations

import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training._common import ciphers as C  # noqa: E402

TEXTS = ["The quick brown fox jumps over the lazy dog",
         "I like to host guests at my home from time to time",
         "A", "AB CD", "Zebras quiz jovial wax mixing vexes pygmy"]


def test_roundtrip_all_ciphers():
    rng = random.Random(0)
    for name, c in C.CIPHERS.items():
        for text in TEXTS:
            for _ in range(20):
                key = c.make_key(rng)
                pt = C.prepare(c, text)
                if not pt:
                    continue
                ct = c.encode(pt, key)
                assert c.decode(ct, key) == pt, (name, text, key, ct)
                assert isinstance(c.key_text(key), str)


def test_ciphertext_is_not_plaintext():
    rng = random.Random(1)
    for name, c in C.CIPHERS.items():
        pt = C.prepare(c, "Meet me at the old bridge at midnight and bring the map")
        changed = sum(c.encode(pt, c.make_key(rng)) != pt for _ in range(10))
        assert changed >= 9, name  # a random key may rarely be the identity (e.g. shift is never 0 by construction)


def test_alphabets_are_assigned_and_distinct():
    for name, a in C.ALPHABETS.items():
        assert len(a.symbols) == len(set(a.symbols)) == 26, name
        assert all(unicodedata.name(s, None) for s in a.symbols), name
        assert a.cost in ("1", "3", "4"), name


def test_declared_token_costs_match_the_world_tokenizer():
    vocab = Path(__file__).resolve().parents[1] / "rwkv-peft/json2binidx_tool/rwkv_vocab_v20230424.txt"
    tools = vocab.parent / "tools"
    if not vocab.exists():
        return  # vendored tree absent (fresh clone): nothing to compare against
    sys.path.insert(0, str(tools))
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        from rwkv_tokenizer import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(str(vocab))
    for name, mean in C.measure_costs(tok.encode).items():
        assert abs(mean - int(C.ALPHABETS[name].cost)) <= 0.35, (name, mean, C.ALPHABETS[name].cost)


def test_split_is_disjoint_and_stable():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gen_ciphers as G
    ids = [f"row{i}#0" for i in range(2000)]
    ev = {i for i in ids if G.in_split(i, "eval")}
    tr = {i for i in ids if G.in_split(i, "train")}
    assert ev and tr and not (ev & tr) and ev | tr == set(ids)
    assert 0.05 < len(ev) / len(ids) < 0.15
    assert all(G.in_split(i, "all") for i in ids)


def test_normalize():
    assert C.normalize("Hello,  World! It's 3pm.") == "HELLO WORLD IT S PM"


def test_deterministic_given_seed():
    c = C.CIPHERS["script_runic"]
    k1, k2 = c.make_key(random.Random(7)), c.make_key(random.Random(7))
    assert k1 == k2 and c.encode("HELLO", k1) == c.encode("HELLO", k2)


if __name__ == "__main__":
    for fn in (test_roundtrip_all_ciphers, test_ciphertext_is_not_plaintext, test_alphabets_are_assigned_and_distinct,
               test_declared_token_costs_match_the_world_tokenizer, test_split_is_disjoint_and_stable, test_normalize, test_deterministic_given_seed):
        fn()
        print("ok", fn.__name__)
    print(len(C.CIPHERS), "ciphers:", ", ".join(C.CIPHERS))
