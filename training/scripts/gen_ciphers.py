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

Tasks (`--task`, default `decode` = the behaviour above). The other three choose WHAT is encrypted so that
decoding is only the way in and the work is on knowledge (owner 2026-10-22: weights x context + the latent channel):
  context   a table of k facts "<NAME> LIVES IN <CITY>" is encrypted; the question (plain text) needs the
            context only. level 1-2: find one person's city (lookup); level 3+: name the two people who
            share a city (every fact has to be read and compared - the k-items-over-a-gap case).
  question  a world-knowledge question is encrypted ("WHAT IS THE CAPITAL OF <COUNTRY>"); the answer is in
            the weights. Whether the model knows it is decided later by its own M=0 pass, not here.
  mixed     the encrypted table refers to weights knowledge ("<NAME> LIVES IN THE CAPITAL OF <COUNTRY>");
            the question needs both the decoded context and the fact.
Answer formats are diluted per row (`--answer-formats`): plain, quoted, json, line; the rubric keeps the
format, so a verifier can check content and form apart. These rows carry `task`, `k`, `answer_format`.

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


# ---------------------------------------------------------------- knowledge tasks (context / question / mixed)

NAMES = ["ALICE", "BORIS", "CLARA", "DMITRI", "ELENA", "FELIX", "GRETA", "HUGO", "IRINA", "JONAS", "KIRA",
         "LEON", "MARTA", "NIKO", "OLGA", "PAVEL", "ROSA", "SAMUEL", "TANYA", "VIKTOR", "YANA", "ZORAN"]
CITIES = ["OSLO", "LIMA", "ROME", "CAIRO", "DELHI", "QUITO", "RIGA", "SEOUL", "TUNIS", "PERTH", "MINSK",
          "DAKAR", "HANOI", "BERN", "ACCRA", "SOFIA"]
# world facts with a one-word answer; known or not is measured by the model's own M=0 pass
CAPITALS = {"FRANCE": "PARIS", "JAPAN": "TOKYO", "ITALY": "ROME", "SPAIN": "MADRID", "EGYPT": "CAIRO",
            "PERU": "LIMA", "NORWAY": "OSLO", "KENYA": "NAIROBI", "CANADA": "OTTAWA", "CHILE": "SANTIAGO",
            "GREECE": "ATHENS", "POLAND": "WARSAW", "AUSTRIA": "VIENNA", "IRELAND": "DUBLIN", "CUBA": "HAVANA",
            "TURKEY": "ANKARA", "SWEDEN": "STOCKHOLM", "FINLAND": "HELSINKI", "PORTUGAL": "LISBON",
            "HUNGARY": "BUDAPEST", "THAILAND": "BANGKOK", "VIETNAM": "HANOI", "ARGENTINA": "BUENOS AIRES",
            "COLOMBIA": "BOGOTA", "MOROCCO": "RABAT", "GHANA": "ACCRA", "SENEGAL": "DAKAR", "LATVIA": "RIGA",
            "BULGARIA": "SOFIA", "SWITZERLAND": "BERN", "AUSTRALIA": "CANBERRA", "BRAZIL": "BRASILIA"}

ANSWER_FORMATS = {
    "plain": ("Answer with the answer only, in capital letters.", lambda a: a),
    "quoted": ('Answer with the answer only, in capital letters, inside double quotes.', lambda a: f'"{a}"'),
    "json": ('Answer as JSON: {"answer": "<ANSWER IN CAPITAL LETTERS>"}.', lambda a: json.dumps({"answer": a})),
    "line": ("End with one line of the form ANSWER: <answer in capital letters>.", lambda a: f"ANSWER: {a}"),
}


def knowledge_item(task: str, k: int, level_hint: int, rng: random.Random) -> tuple[str, str, str, int]:
    """(plaintext to encrypt, plain-text question, answer, level). Every answer is computed from the facts."""
    if task == "question":
        country = rng.choice(sorted(CAPITALS))
        return f"WHAT IS THE CAPITAL OF {country}", "Answer the decoded question.", CAPITALS[country], 2
    names = rng.sample(NAMES, k)
    if task == "mixed":
        countries = rng.sample(sorted(CAPITALS), k)
        facts = [f"{n} LIVES IN THE CAPITAL OF {c}" for n, c in zip(names, countries)]
        who = rng.randrange(k)
        q = f"In which city does {names[who].title()} live, according to the decoded message?"
        return ". ".join(facts), q, CAPITALS[countries[who]], 2 + (k > 3)
    # context
    if level_hint <= 2 or k < 3:
        cities = rng.sample(CITIES, k)
        who = rng.randrange(k)
        facts = [f"{n} LIVES IN {c}" for n, c in zip(names, cities)]
        q = f"In which city does {names[who].title()} live, according to the decoded message?"
        return ". ".join(facts), q, cities[who], 1 + (k > 3)
    cities = rng.sample(CITIES, k - 1)
    a, b = rng.sample(range(k), 2)
    shared = cities[0]
    assign, rest = {}, iter(cities[1:])
    for i in range(k):
        assign[i] = shared if i in (a, b) else next(rest)
    facts = [f"{names[i]} LIVES IN {assign[i]}" for i in range(k)]
    q = "Which two people live in the same city, according to the decoded message? Give both names joined by AND, in alphabetical order."
    return ". ".join(facts), q, " AND ".join(sorted([names[a], names[b]])), 3 + (k > 4)


def make_knowledge_row(c, task: str, k: int, level_hint: int, fmt: str, rng: random.Random) -> dict:
    """c = None is the rung below every cipher: the same message in plain text. It separates "cannot do the
    task" from "cannot read the cipher" (G1k at M=0, 2026-10-22: perfect answer format, content = the most
    probable filler - NEW YORK, ALICE AND BOB - on every cipher, even caesar)."""
    text, question, answer, level = knowledge_item(task, k, level_hint, rng)
    instr, render = ANSWER_FORMATS[fmt]
    if c is None:
        key, name = None, "plain"
        prompt = f"Read the message, then answer the question.\nMessage:\n{text}\nQuestion: {question}\n{instr}"
        level = max(1, level - 1)
    else:
        key, name = c.make_key(rng), c.name
        pt = C.prepare(c, text)
        ct = c.encode(pt, key)
        assert c.decode(ct, key) == pt
        spaces = "" if c.keep_spaces else " The message was encrypted without spaces."
        prompt = (f"Decode the message, then answer the question.\nCipher: {c.title}. Rule: {c.rule}.\n"
                  f"{c.key_text(key)}\nMessage:\n{ct}\n{spaces.strip()}\nQuestion: {question}\n{instr}").replace("\n\n", "\n")
    content_id = hashlib.md5(text.encode()).hexdigest()[:12]
    kh = hashlib.md5(json.dumps(key, default=str, sort_keys=True).encode()).hexdigest()[:6]
    return {
        "id": f"cipher_{task}_{name}_{content_id}_{kh}_{fmt}", "category": f"cipher_{task}",
        "level": min(level + (BASE_LEVEL.get(name, 1) > 2), 6), "prompt": prompt, "answer": render(answer),
        "rubric": {"type": "exact", "value": answer.lower(), "format": fmt}, "content_id": content_id,
        "view": name, "mode": "known" if c else "plain", "task": task, "k": k, "answer_format": fmt,
        "alphabet_cost": "", "notes": f"{task} k={k} via {name}, plaintext {len(text)} chars",
    }


def hint_ladder(row: dict) -> list[str]:
    """Hints for a PLAIN-rung knowledge row, weakest first, each one closer to the answer without stating it in the
    requested format. S1 privileged-information distillation (2026-10-27): the teacher is the same model reading the
    hint at M=0, the student gets no hint and must rebuild that state in its latent ticks - an item the model already
    solves without a hint lets a pass-through tick lower the loss. Parsed from the prompt, so it needs no new data."""
    if row.get("view") != "plain" or "task" not in row:
        return []
    msg = row["prompt"].split("Message:\n", 1)[1].split("\nQuestion:", 1)[0]
    facts = [f.strip() for f in msg.split(".") if f.strip()]
    q = row["prompt"].split("Question: ", 1)[1].split("\n", 1)[0]
    if row["task"] == "question":
        return []
    m = re.search(r"does (\w+) live", q)
    if m:                                                    # one person: context or mixed
        who = m.group(1).upper()
        i, fact = next((i, f) for i, f in enumerate(facts) if f.startswith(who + " "))
        ladder = [f"Hint: {who.title()} is in sentence {i + 1} of the message.",
                  f"Hint: the sentence that matters is \"{fact}\"."]
        c = re.search(r"THE CAPITAL OF (\w+)", fact)
        if c:
            ladder.append(ladder[-1] + f" The capital of {c.group(1).title()} is {CAPITALS[c.group(1)].title()}.")
        return ladder
    cities = [f.split(" LIVES IN ")[1] for f in facts]       # the pair task: two people share a city
    shared = next(c for c in cities if cities.count(c) == 2)
    pair = [f.split(" LIVES IN ")[0] for f in facts if f.endswith(" " + shared)]
    return [f"Hint: one of the two is {pair[0].title()}.",
            f"Hint: the shared city is {shared.title()}."]


def with_hint(row: dict, level: int) -> str:
    """The row's prompt with hint `level` (1-based) inserted before the question; level 0 or no hint -> the prompt."""
    ladder = hint_ladder(row)
    if not level or not ladder:
        return row["prompt"]
    h = ladder[min(level, len(ladder)) - 1]
    head, tail = row["prompt"].split("\nQuestion:", 1)
    return f"{head}\n{h}\nQuestion:{tail}"


def check_answer(row: dict, response: str) -> tuple[bool, bool]:
    """(content right, form right) for a knowledge row - one checker for reward, teacher filter and eval.
    Form is read through the requested format; content does not depend on it."""
    gold = row["rubric"]["value"].strip().upper()
    fmt = row["rubric"].get("format", "plain")
    text = response.strip().split("\n\n")[0]
    norm = lambda t: re.sub(r"[^A-Z ]+", " ", t.upper()).split()
    got, form = None, False
    if fmt == "json":
        m = re.search(r"\{[^{}]*\}", text)
        try:
            got, form = (str(json.loads(m.group(0)).get("answer", "")), True) if m else (None, False)
        except (json.JSONDecodeError, AttributeError):
            got = None
    elif fmt == "quoted":
        m = re.fullmatch(r'\s*"([^"]+)"\s*', text)
        got, form = (m.group(1), True) if m else (None, False)
    elif fmt == "line":
        m = re.search(r"^ANSWER:\s*(.+)$", text, re.M)
        got, form = (m.group(1), True) if m else (None, False)
    else:
        got, form = text, bool(re.fullmatch(r"[A-Z ]+", text.strip()))
    # Content is one rule for every requested format (2026-10-28: it was exact for plain and a substring check
    # for the others, which moved every score): the gold answer is said, and no other answer of its kind is.
    said = " " + " ".join(norm(text)) + " "
    pool = set(NAMES) if " AND " in f" {gold} " else set(CITIES) | set(CAPITALS.values())
    others = [c for c in pool if c not in gold.split(" AND ") and f" {c} " in said]
    parts = gold.split(" AND ")                                # a pair counts in either order
    content = all(f" {' '.join(norm(x))} " in said for x in parts) and not others
    return bool(content), bool(form)


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
    ap.add_argument("--task", default="decode", choices=("decode", "context", "question", "mixed"))
    ap.add_argument("--k", default="2,3,4,6", help="facts per table for context/mixed (comma list, sampled per row)")
    ap.add_argument("--answer-formats", default="plain,quoted,json,line")
    return ap


def run_knowledge(args: argparse.Namespace, ciphers: list) -> dict:
    rng = random.Random(args.seed)
    ks = [int(x) for x in str(args.k).split(",")]
    fmts = args.answer_formats.split(",")
    rows, seen = [], set()
    while len(rows) < args.n:
        c = ciphers[len(rows) % len(ciphers)]
        r = make_knowledge_row(c, args.task, rng.choice(ks), rng.choice([1, 2, 3, 4]), fmts[len(rows) % len(fmts)], rng)
        if r["content_id"] in seen and args.task != "question":
            continue
        if not in_split(r["content_id"], args.split):
            continue
        seen.add(r["content_id"])
        rows.append(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_ciphers] task={args.task}: {len(rows)} rows -> {args.out}")
    return {"out_path": args.out, "n_rows": len(rows), "n_contents": len(seen)}


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    names = list(C.CIPHERS) if args.families == "all" else args.families.split(",")
    ciphers = [None if n == "plain" else C.CIPHERS[n] for n in names]
    if getattr(args, "task", "decode") != "decode":
        return run_knowledge(args, ciphers)
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
