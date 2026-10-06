"""Cipher tasks over useful text: decode a message, answer checked exactly.

Payload is real sentences read through the dataset catalog (`catalog.rows`), so a personal-derived
set is refused by construction. Each payload text is emitted under `--views` different ciphers: the
rows share `content_id` and differ in `view`, which gives the matched pairs a JEPA-style objective
and a format-vs-content control need (same content, different surface). Known-key tasks put the key
in the prompt; unknown-key tasks (`--unknown-share`) give only the cipher name and enough text for
the key to be recoverable, which is the hypothesis-search-with-a-verifier case.

Row (same shape as matrix_tasks.jsonl, plus content_id / view / mode):
  {id, category: "cipher_<name>", level, prompt, answer, rubric: {type: "exact", value}, content_id,
   view, mode: "known"|"unknown", alphabet_cost, notes}

Usage:
    training/.venv/bin/python training/scripts/gen_ciphers.py \\
        --payload hh_rlhf --field answer --n 3000 --views 3 --seed 7 \\
        --out training/corpus_open/ciphers_v1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training._common import ciphers as C  # noqa: E402

BASE_LEVEL = {"caesar": 1, "atbash": 1, "polybius6": 2, "tap6": 2, "bacon": 2, "railfence": 2,
              "vigenere": 3, "columnar": 3, "bifid6": 4, "adfgvx": 4}
# smallest plaintext (characters) for which the key is recoverable from the text alone
UNKNOWN_MIN = {"caesar": 10, "atbash": 10, "railfence": 30, "columnar": 60, "vigenere": 100}
UNKNOWN_SUBST_MIN = 120


def _level(c: C.Cipher, n_chars: int, mode: str) -> int:
    lv = BASE_LEVEL.get(c.name, 1)
    lv += 1 if n_chars > 100 else 0
    lv += 1 if mode == "unknown" else 0
    return min(lv, 6)


def sentences(text: str, lo: int, hi: int) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        letters = sum(ch.isalpha() or ch == " " for ch in s)
        if not s or letters / len(s) < 0.85:
            continue  # code, markup, numbers: not prose
        n = C.normalize(s)
        if lo <= len(n) <= hi and len(n.split()) >= 6:
            out.append(n)
    return out


def payload_texts(name: str, field: str, lo: int, hi: int, long_hi: int, rng: random.Random, limit: int):
    """Yield (content_id, text). A text is 1-3 consecutive sentences of one row, up to long_hi characters;
    short and long texts are both produced so unknown-key tasks have enough material."""
    from training._common import catalog
    seen = set()
    for i, row in enumerate(catalog.rows(name, limit=limit)):
        text = row.get(field) if isinstance(row, dict) else None
        if not isinstance(text, str):
            continue
        sents = sentences(text, lo, hi)
        if not sents:
            continue
        k = rng.choice([1, 1, 2, 3])
        start = rng.randrange(len(sents))
        chunk = " ".join(sents[start:start + k])
        if len(chunk) > long_hi:
            chunk = sents[start]
        if chunk in seen:
            continue
        seen.add(chunk)
        yield f"{row.get('id', i)}#{start}", chunk


def in_split(content_id: str, split: str) -> bool:
    """Held-out contents are chosen by hash, so train and eval never share a text whatever the seed."""
    if split == "all":
        return True
    is_eval = int(hashlib.md5(content_id.encode()).hexdigest(), 16) % 10 == 0
    return is_eval if split == "eval" else not is_eval


def make_row(c: C.Cipher, content_id: str, text: str, mode: str, rng: random.Random) -> dict:
    key = c.make_key(rng)
    pt = C.prepare(c, text)
    ct = c.encode(pt, key)
    assert c.decode(ct, key) == pt  # never ship a task whose own answer does not decode
    spaces = "" if c.keep_spaces else " Spaces are not encoded: give the plaintext as one run of capital letters."
    if mode == "known":
        body = f"Cipher: {c.title}. Rule: {c.rule}.\n{c.key_text(key)}\n"
    else:
        body = (f"Cipher: {c.title}. Rule: {c.rule}.\nThe key is not given. The plaintext is ordinary English; "
                f"work the key out from the message.\n")
    prompt = f"Decode the message.\n{body}Message:\n{ct}\nAnswer with the plaintext only, in capital letters.{spaces}"
    kh = hashlib.md5(json.dumps(key, default=str, sort_keys=True).encode()).hexdigest()[:6]
    ch = hashlib.md5(content_id.encode()).hexdigest()[:8]
    alpha = c.name.removeprefix("script_") if c.name.startswith("script_") else ""
    return {
        "id": f"cipher_{c.name}_{mode}_{ch}_{kh}", "category": f"cipher_{c.name}",
        "level": _level(c, len(pt), mode), "prompt": prompt, "answer": pt,
        "rubric": {"type": "exact", "value": pt.lower()}, "content_id": content_id, "view": c.name,
        "mode": mode, "alphabet_cost": C.ALPHABETS[alpha].cost if alpha else "",
        "notes": f"{c.name} {mode}-key, {len(pt)} plaintext chars",
    }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="hh_rlhf", help="catalog name of the text source (never a personal set)")
    ap.add_argument("--field", default="answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="rows to write")
    ap.add_argument("--views", type=int, default=3, help="ciphers per content (shared content_id)")
    ap.add_argument("--families", default="all", help="comma list of cipher names, or all")
    ap.add_argument("--unknown-share", type=float, default=0.2)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=160)
    ap.add_argument("--long-chars", type=int, default=300)
    ap.add_argument("--source-rows", type=int, default=60000, help="payload rows to read at most")
    ap.add_argument("--split", default="all", choices=("all", "train", "eval"),
                    help="eval = the ~10%% of contents whose md5 falls in a fixed bucket; train = the rest")
    ap.add_argument("--seed", type=int, default=7)
    return ap


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    names = list(C.CIPHERS) if args.families == "all" else args.families.split(",")
    ciphers = [C.CIPHERS[n] for n in names]
    out_rows, per_cipher = [], {n: 0 for n in names}
    for content_id, text in payload_texts(args.payload, args.field, args.min_chars, args.max_chars,
                                          args.long_chars, rng, args.source_rows):
        if len(out_rows) >= args.n:
            break
        if not in_split(content_id, args.split):
            continue
        picks = sorted(ciphers, key=lambda c: (per_cipher[c.name], rng.random()))[:args.views]  # balance families
        for c in picks:
            mode = "known"
            if c.unknown_key_ok and rng.random() < args.unknown_share:
                need = UNKNOWN_MIN.get(c.name, UNKNOWN_SUBST_MIN)
                if len(C.prepare(c, text)) >= need:
                    mode = "unknown"
            if len(C.prepare(c, text)) < c.min_len:
                continue
            out_rows.append(make_row(c, content_id, text, mode, rng))
            per_cipher[c.name] += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_ciphers] {len(out_rows)} rows, {len({r['content_id'] for r in out_rows})} contents, "
          f"{len(names)} ciphers -> {args.out}")
    return {"out_path": args.out, "n_rows": len(out_rows), "n_contents": len({r['content_id'] for r in out_rows})}


try:
    from training._common import registry as _registry
    _registry.stage(
        "ciphers", kind="generate", provenance="generated",
        origin="character-level ciphers over catalogued payload text (training/_common/ciphers.py)",
        out_default="training/corpus_open/ciphers_v1.jsonl",
        description="Cipher decoding tasks, exact check; rows sharing content_id are views of one text.",
    )(run)
except ImportError:
    pass


if __name__ == "__main__":
    run(_build_argparser().parse_args())
