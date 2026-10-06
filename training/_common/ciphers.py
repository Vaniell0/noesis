"""training/_common/ciphers.py — character-level ciphers with exact, invertible codecs.

Every cipher here is a pair (encode, decode) over a normalised payload (capital letters A-Z and,
where the cipher keeps them, single spaces), with a random key drawn from a seed. "Correct" is the
exact plaintext, so the same checker serves eval, reward and teacher filtering. The point of the
families (user, 2026-10-06): many rules with a crisp key, alphabets that look nothing like Latin,
and useful text as the payload — see docs/datasets.md and the research note on ciphered reasoning
(arXiv 2510.09714: models translate ciphertext but reason poorly in it, and the gap tracks how
common the cipher is in pretraining, so variety matters more than one cipher).

Normalised payload: `normalize(text)` -> upper-case A-Z and single spaces. Ciphers with
`keep_spaces=False` drop the spaces from the plaintext and the prompt says so.

Alphabets are Unicode blocks verified assigned in Unicode 15.1 (python unicodedata); each yields at
least 26 distinct symbols. World-tokenizer cost differs a lot between them (katakana/Cyrillic/Hebrew 1 token
per char, runes/Ogham/Braille/Cherokee/Glagolitic ~3, Shavian/Deseret/Linear B/Phaistos ~4,
docs/rl-track.md) — `ALPHABETS[name].cost` carries the measured class so a generator can pick.
"""
from __future__ import annotations

import random
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

LETTERS = string.ascii_uppercase
SQUARE_SYMBOLS = LETTERS + string.digits  # the 6x6 square


def normalize(text: str) -> str:
    t = re.sub(r"[^A-Za-z ]+", " ", text).upper()
    return re.sub(r" +", " ", t).strip()


# ---------------------------------------------------------------- alphabets

def _block(lo: int, hi: int, n: int = 26) -> list[str]:
    out = [chr(c) for c in range(lo, hi + 1)
           if unicodedata.name(chr(c), None) and unicodedata.category(chr(c)).startswith(("L", "S", "N"))]
    if len(out) < n:
        raise ValueError(f"block U+{lo:04X}..U+{hi:04X} has only {len(out)} symbols")
    return out[:n]


@dataclass(frozen=True)
class Alphabet:
    name: str
    symbols: tuple
    cost: str  # measured World-tokenizer class: "1" | "3" | "4" tokens per symbol (docs/rl-track.md)


ALPHABETS: dict[str, Alphabet] = {a.name: a for a in [
    Alphabet("katakana", tuple("アイウエオカキクケコサシスセソタチツテトナニヌネノハ"), "1"),
    Alphabet("cyrillic", tuple(_block(0x0430, 0x044F)), "1"),
    Alphabet("hebrew", tuple(_block(0x05D0, 0x05EA)), "1"),
    Alphabet("runic", tuple(_block(0x16A0, 0x16EA)), "3"),
    Alphabet("ogham", tuple(_block(0x1681, 0x169A)), "3"),
    Alphabet("glagolitic", tuple(_block(0x2C30, 0x2C5F)), "3"),
    Alphabet("cherokee", tuple(_block(0x13A0, 0x13F5)), "3"),
    Alphabet("braille", tuple(_block(0x2801, 0x283F)), "3"),
    Alphabet("tifinagh", tuple(_block(0x2D30, 0x2D67)), "3"),
    Alphabet("shavian", tuple(_block(0x10450, 0x1047F)), "4"),
    Alphabet("deseret", tuple(_block(0x10428, 0x1044F)), "4"),
    Alphabet("ugaritic", tuple(_block(0x10380, 0x1039D)), "4"),
    Alphabet("linear_b", tuple(_block(0x10000, 0x1005D)), "4"),
    Alphabet("phaistos", tuple(_block(0x101D0, 0x101FC)), "4"),
    Alphabet("yijing", tuple(_block(0x4DC0, 0x4DFF)), "3"),
]}
BACON_PAIRS = [("a", "b"), ("0", "1"), ("x", "y"), ("☀", "☾"), ("●", "○"),
               ("♥", "♠"), ("あ", "い")]
ADFGVX = "ADFGVX"


def measure_costs(encode) -> dict:
    """Mean tokens per symbol of each alphabet's 26 symbols under `encode(str) -> list[int]`
    (the World tokenizer). Costs in ALPHABETS are declared from this measurement
    (2026-10-06, training/tests/test_ciphers.py re-checks them when the vocab file is present)."""
    return {n: sum(len(encode(sym)) for sym in a.symbols) / len(a.symbols) for n, a in ALPHABETS.items()}


# ---------------------------------------------------------------- helpers

def _shift(ch: str, k: int) -> str:
    return LETTERS[(LETTERS.index(ch) + k) % 26]


def _keyword_order(word: str) -> list[int]:
    """Column read order for a transposition keyword (ties broken left to right)."""
    return sorted(range(len(word)), key=lambda i: (word[i], i))


def _columnar_encode(text: str, word: str) -> str:
    n = len(word)
    cols = [text[i::n] for i in range(n)]
    return "".join(cols[i] for i in _keyword_order(word))


def _columnar_decode(ct: str, word: str) -> str:
    n = len(word)
    base, extra = divmod(len(ct), n)
    lens = [base + (1 if i < extra else 0) for i in range(n)]
    cols: dict[int, str] = {}
    pos = 0
    for i in _keyword_order(word):
        cols[i] = ct[pos:pos + lens[i]]
        pos += lens[i]
    out = []
    for r in range(base + (1 if extra else 0)):
        for i in range(n):
            if r < len(cols[i]):
                out.append(cols[i][r])
    return "".join(out)


def _rail_pattern(n: int, rails: int) -> list[int]:
    if rails == 1:
        return [0] * n
    cyc = list(range(rails)) + list(range(rails - 2, 0, -1))
    return [cyc[i % len(cyc)] for i in range(n)]


def _random_square(rng: random.Random) -> str:
    s = list(SQUARE_SYMBOLS)
    rng.shuffle(s)
    return "".join(s)


def _square_str(sq: str) -> str:
    return "\n".join(" ".join(sq[r * 6:(r + 1) * 6]) for r in range(6))


# ---------------------------------------------------------------- the ciphers

@dataclass
class Cipher:
    name: str
    title: str
    rule: str
    keep_spaces: bool
    make_key: Callable[[random.Random], Any]
    encode: Callable[[str, Any], str]
    decode: Callable[[str, Any], str]
    key_text: Callable[[Any], str]
    min_len: int = 20
    unknown_key_ok: bool = False  # an unknown-key version is well posed given enough text


def _caesar() -> Cipher:
    enc = lambda t, k: "".join(_shift(c, k) if c != " " else c for c in t)
    dec = lambda t, k: "".join(_shift(c, -k) if c != " " else c for c in t)
    return Cipher("caesar", "Caesar shift", "each letter moves forward in the alphabet by the shift; spaces stay",
                  True, lambda r: r.randint(1, 25), enc, dec, lambda k: f"Shift: {k}", 10, True)


def _atbash() -> Cipher:
    m = {a: b for a, b in zip(LETTERS, LETTERS[::-1])}
    f = lambda t, k: "".join(m.get(c, c) for c in t)
    return Cipher("atbash", "Atbash", "A<->Z, B<->Y, C<->X, and so on; spaces stay",
                  True, lambda r: None, f, f, lambda k: "Key: none (the alphabet reversed)", 10, True)


def _vigenere() -> Cipher:
    def enc(t, k):
        out, i = [], 0
        for c in t:
            if c == " ":
                out.append(c)
            else:
                out.append(_shift(c, LETTERS.index(k[i % len(k)])))
                i += 1
        return "".join(out)

    def dec(t, k):
        out, i = [], 0
        for c in t:
            if c == " ":
                out.append(c)
            else:
                out.append(_shift(c, -LETTERS.index(k[i % len(k)])))
                i += 1
        return "".join(out)

    return Cipher("vigenere", "Vigenere", "letter i moves forward by the value of the i-th keyword letter "
                  "(A=0), the keyword repeats over letters only; spaces stay", True,
                  lambda r: "".join(r.choice(LETTERS) for _ in range(r.randint(3, 6))), enc, dec,
                  lambda k: f"Keyword: {k}", 40, True)


def _script(alpha: Alphabet) -> Cipher:
    def make(r):
        s = list(alpha.symbols)
        r.shuffle(s)
        return dict(zip(LETTERS, s))

    enc = lambda t, k: "".join(k.get(c, c) for c in t)

    def dec(t, k):
        inv = {v: a for a, v in k.items()}
        return "".join(inv.get(c, c) for c in t)

    return Cipher(f"script_{alpha.name}", f"{alpha.name} letter substitution",
                  f"every Latin letter is replaced by one {alpha.name} symbol according to the table; spaces stay",
                  True, make, enc, dec,
                  lambda k: "Table: " + " ".join(f"{a}={k[a]}" for a in LETTERS), 20, True)


def _polybius6() -> Cipher:
    def enc(t, sq):
        return " ".join(f"{sq.index(c) // 6 + 1}{sq.index(c) % 6 + 1}" for c in t)

    def dec(t, sq):
        return "".join(sq[(int(g[0]) - 1) * 6 + int(g[1]) - 1] for g in t.split())

    return Cipher("polybius6", "Polybius square (6x6)",
                  "each letter becomes row digit then column digit in the square (rows and columns numbered 1-6); "
                  "groups are separated by spaces", False, _random_square, enc, dec,
                  lambda sq: "Square (rows top to bottom):\n" + _square_str(sq), 15)


def _tap6() -> Cipher:
    def enc(t, sq):
        return " ".join("." * (sq.index(c) // 6 + 1) + "," + "." * (sq.index(c) % 6 + 1) for c in t)

    def dec(t, sq):
        out = []
        for g in t.split():
            r, c = g.split(",")
            out.append(sq[(len(r) - 1) * 6 + len(c) - 1])
        return "".join(out)

    return Cipher("tap6", "Tap code (6x6)",
                  "each letter is two groups of dots, row then column in the square, separated by a comma; "
                  "letters are separated by spaces", False, _random_square, enc, dec,
                  lambda sq: "Square (rows top to bottom):\n" + _square_str(sq), 15)


def _bacon() -> Cipher:
    def enc(t, k):
        a, b = k
        return " ".join("".join(b if (LETTERS.index(c) >> (4 - i)) & 1 else a for i in range(5)) for c in t)

    def dec(t, k):
        a, b = k
        out = []
        for g in t.split():
            v = 0
            for ch in g:
                v = v * 2 + (1 if ch == b else 0)
            out.append(LETTERS[v])
        return "".join(out)

    return Cipher("bacon", "Baconian binary", "each letter is its position A=0..Z=25 written in 5 bits, 0 as the first "
                  "symbol and 1 as the second; groups are separated by spaces", False,
                  lambda r: r.choice(BACON_PAIRS), enc, dec,
                  lambda k: f"Symbols: 0 = {k[0]}, 1 = {k[1]}", 10)


def _railfence() -> Cipher:
    def enc(t, rails):
        pat = _rail_pattern(len(t), rails)
        return "".join(t[i] for r in range(rails) for i in range(len(t)) if pat[i] == r)

    def dec(t, rails):
        pat = _rail_pattern(len(t), rails)
        order = [i for r in range(rails) for i in range(len(t)) if pat[i] == r]
        out = [""] * len(t)
        for ch, i in zip(t, order):
            out[i] = ch
        return "".join(out)

    return Cipher("railfence", "Rail fence", "the text is written in a zigzag over the rails and read off rail by "
                  "rail; spaces are ordinary characters", True, lambda r: r.randint(2, 5), enc, dec,
                  lambda k: f"Rails: {k}", 30, True)


def _columnar() -> Cipher:
    return Cipher("columnar", "Columnar transposition", "the text is written in rows under the keyword and the "
                  "columns are read out in alphabetical order of the keyword letters (ties left to right); the last "
                  "row may be short; spaces are ordinary characters", True,
                  lambda r: "".join(r.sample(LETTERS, r.randint(4, 7))),
                  _columnar_encode, _columnar_decode, lambda k: f"Keyword: {k}", 30, True)


def _bifid6() -> Cipher:
    def coords(c, sq):
        i = sq.index(c)
        return i // 6, i % 6

    def enc(t, key):
        sq, p = key
        out = []
        for s in range(0, len(t), p):
            blk = t[s:s + p]
            rs, cs = zip(*[coords(c, sq) for c in blk])
            seq = list(rs) + list(cs)
            out.append("".join(sq[seq[2 * i] * 6 + seq[2 * i + 1]] for i in range(len(blk))))
        return "".join(out)

    def dec(t, key):
        sq, p = key
        out = []
        for s in range(0, len(t), p):
            blk = t[s:s + p]
            seq = []
            for c in blk:
                seq.extend(coords(c, sq))
            n = len(blk)
            rs, cs = seq[:n], seq[n:]
            out.append("".join(sq[rs[i] * 6 + cs[i]] for i in range(n)))
        return "".join(out)

    return Cipher("bifid6", "Bifid (6x6)", "for each block of letters, write all row numbers then all column numbers, "
                  "then read the sequence back in pairs and look the pairs up in the square", False,
                  lambda r: (_random_square(r), r.choice([5, 6, 7])), enc, dec,
                  lambda k: f"Block length: {k[1]}\nSquare (rows top to bottom):\n" + _square_str(k[0]), 20)


def _adfgvx() -> Cipher:
    def enc(t, key):
        sq, word = key
        pairs = "".join(ADFGVX[sq.index(c) // 6] + ADFGVX[sq.index(c) % 6] for c in t)
        return _columnar_encode(pairs, word)

    def dec(t, key):
        sq, word = key
        pairs = _columnar_decode(t, word)
        return "".join(sq[ADFGVX.index(pairs[i]) * 6 + ADFGVX.index(pairs[i + 1])] for i in range(0, len(pairs), 2))

    return Cipher("adfgvx", "ADFGVX", "each letter becomes two of the letters A D F G V X (row then column in the "
                  "square), then the whole string goes through a columnar transposition with the keyword", False,
                  lambda r: (_random_square(r), "".join(r.sample(LETTERS, r.randint(4, 6)))), enc, dec,
                  lambda k: f"Keyword: {k[1]}\nSquare (rows top to bottom; row and column labels A D F G V X):\n"
                  + _square_str(k[0]), 15)


CIPHERS: dict[str, Cipher] = {c.name: c for c in (
    [_caesar(), _atbash(), _vigenere()] + [_script(a) for a in ALPHABETS.values()] +
    [_polybius6(), _tap6(), _bacon(), _railfence(), _columnar(), _bifid6(), _adfgvx()])}


def prepare(cipher: Cipher, text: str) -> str:
    """The plaintext the model must return: normalised, spaces removed when the cipher drops them."""
    t = normalize(text)
    return t if cipher.keep_spaces else t.replace(" ", "")


def roundtrip_ok(cipher: Cipher, text: str, key: Any) -> bool:
    pt = prepare(cipher, text)
    return cipher.decode(cipher.encode(pt, key), key) == pt
