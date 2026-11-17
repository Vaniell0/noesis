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
    if row.get("style") in ("trace", "indexed"):                            # the letters after '=', in order, are the plaintext
        got = "".join(re.findall(r"=\s*([A-Za-z])", text)).upper()
        return got == gold.replace(" ", ""), bool(re.fullmatch(r"\S+=[A-Z]( (/ )?\S+=[A-Z])*", text.strip()))
    if row.get("task") == "ladder":                            # the whole plaintext, said as one run
        said = " " + " ".join(norm(text)) + " "
        return f" {' '.join(norm(gold))} " in said, bool(re.fullmatch(r"[A-Z ]+", text.strip()))
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
    ap.add_argument("--task", default="decode", choices=("decode", "context", "question", "mixed", "ladder"))
    ap.add_argument("--ladder-lengths", default="6,12,24", help="ladder: plaintext letters per cell (comma list)")
    ap.add_argument("--ladder-style", default="plain", choices=("plain", "trace", "indexed"),
                    help="ladder: plain = plaintext answer, prefix shown in the prompt; trace = cipher->plain table, "
                         "symbol=letter pairs, the first pairs written as the answer's prefill; indexed = trace with "
                         "positions on the message and the answer (1:ク=A)")
    ap.add_argument("--ladder-table", default="cipher_plain", choices=("cipher_plain", "plain_cipher"),
                    help="ladder trace/indexed: the table's direction; plain_cipher (A=ク) makes decoding an inverse lookup")
    ap.add_argument("--ladder-rungs", default="", help="ladder trace/indexed: pairs shown per rung (comma list; "
                                                       "default 4,2,1,0; 0 alone = whole answers, as training rows)")
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


# ---------------------------------------------------------------- ladder (a decode task the model can enter)

# Short common words, none of them a city or capital (check_answer's pool) - the payload with a language prior.
LADDER_WORDS = ["THE", "AND", "SUN", "RED", "DOG", "CAT", "TREE", "BOOK", "HAND", "BLUE", "MILK", "SHIP", "STONE",
                "WATER", "HOUSE", "GREEN", "LIGHT", "RIVER", "APPLE", "CHAIR", "BREAD", "NIGHT", "SMALL", "COLD"]
LADDER_REVEALED = (0.75, 0.5, 0.25, 0.0)


def _rung(grid: tuple, v) -> int:
    """1-based rung on the standard grid; 0 for a value between rungs (a cell the data improver adds)."""
    return grid.index(v) + 1 if v in grid else 0


def ladder_payload(kind: str, length: int, rng: random.Random) -> str:
    """About `length` letters (spaces not counted) of whole common words, or exactly `length` random letters in
    word-sized groups: the second has no language prior, so only the table can produce it."""
    out, n = [], 0
    while n < length:
        w = rng.choice(LADDER_WORDS) if kind == "words" else "".join(
            rng.choice(C.LETTERS) for _ in range(rng.randint(3, 5)))[:length - n]
        out.append(w)
        n += len(w)
    return " ".join(out)


def make_ladder_row(c: C.Cipher, kind: str, length: int, revealed: float, rng: random.Random) -> dict:
    """A known-key decode whose first `revealed` share of letters is shown already decoded. The rungs run from
    copying the rest of a prefix the model can see (0.75) to decoding all of it (0), so a model that solves
    none of the full task still solves some rungs (Absolute Zero D.5: hints are how a learner climbs)."""
    key = c.make_key(rng)
    pt = ladder_payload(kind, length, rng)
    ct = c.encode(pt, key)
    assert c.decode(ct, key) == pt
    k = round(revealed * len(pt))
    k = k if k == 0 or pt[k - 1] != " " else k - 1           # a prefix never ends on a space
    shown = (f"The first {len(pt[:k].replace(' ', ''))} letters are already decoded: {pt[:k]}\n"
             if k else "")
    prompt = (f"Decode the message.\nCipher: {c.title}. Rule: {c.rule}.\n{c.key_text(key)}\nMessage:\n{ct}\n"
              f"{shown}Answer with the whole plaintext only, in capital letters.")
    h = hashlib.md5(f"{c.name}|{kind}|{length}|{revealed}|{pt}|{ct}".encode()).hexdigest()[:10]
    return {"id": f"ladder_{c.name}_{kind}_{length}_r{int(revealed * 100)}_{h}", "category": f"cipher_{c.name}",
            "task": "ladder", "level": _rung(LADDER_REVEALED, revealed),
            "answer_format": f"{c.name}/{kind}/{length}", "prompt": prompt, "answer": pt,
            "rubric": {"type": "exact", "value": pt.lower(), "format": "plain"}, "content_id": f"ladder_{h}",
            "view": c.name, "mode": "known", "revealed": revealed, "revealed_chars": k, "payload_kind": kind,
            "length": length, "letters": len(pt.replace(" ", "")),
            "notes": f"ladder rung: {c.name}, {kind}, {length} letters, {revealed:.2f} shown"}


LADDER_PAIRS_SHOWN = (4, 2, 1, 0)


def trace_of(pt: str, enc: dict, indexed: bool = False) -> str:
    """The pair answer: 'ク=A ス=B / ...' (cipher symbol, '=', its plain letter; '/' between words); indexed:
    '1:ク=A 2:ス=B / ...' (the position in the message first)."""
    words, out, i = pt.split(), [], 0
    for w in words:
        pairs = []
        for a in w:
            i += 1
            pairs.append(f"{i}:{enc[a]}={a}" if indexed else f"{enc[a]}={a}")
        out.append(" ".join(pairs))
    return " / ".join(out)


def make_ladder_trace_row(c: C.Cipher, kind: str, length: int, shown: int, rng: random.Random,
                          indexed: bool = False, inverse: bool = False) -> dict:
    """The ladder as the 0.4B can enter it (cipher_lookup_probe, 2026-11-13): the key written cipher->plain
    (decoding is then a forward lookup; the plain->cipher table of key_text gave 0.002-0.004 per new letter,
    cipher->plain 0.57-0.81), the answer as symbol=letter pairs, and the first `shown` pairs of the answer
    already written (row["prefill"], continued by the model): lookup starts only after 1-3 worked pairs.
    indexed (v3): every message symbol carries its position ('1:ク 2:ス') and so does the answer ('1:ク=A'): in v2
    greedy answers wrote the right letter for their own symbol (0.67-0.96) but the next message symbol only 0.02-0.04
    (they walked the table in order); a unique position key makes the next symbol a forward lookup too.
    inverse: the table written plain->cipher (A=ク), as key_text writes it - decoding is then an inverse lookup;
    the rows a model is trained on to test whether training (with or without ticks) opens that lookup."""
    key = c.make_key(rng)
    pt = ladder_payload(kind, length, rng)
    ct = c.encode(pt, key)
    assert c.decode(ct, key) == pt
    enc = {a: c.encode(a, key) for a in C.LETTERS}
    assert len(set(enc.values())) == 26
    table = "Table: " + " ".join(f"{a}={enc[a]}" if inverse else f"{enc[a]}={a}" for a in C.LETTERS)
    answer = trace_of(pt, enc, indexed)
    pairs = re.findall(r"\S+=[A-Z]", answer)
    if indexed:
        ct = " / ".join(" ".join(p.split("=")[0] for p in w.split()) for w in answer.split(" / "))
    assert shown < len(pairs)                                  # at least one pair is left to the model
    prefill = answer[:answer.index(pairs[shown - 1]) + len(pairs[shown - 1])] if shown else ""
    rule = ("each plain letter before '=' in the table is written as the cipher symbol after it" if inverse else
            "each cipher symbol stands for the plain letter after '=' in the table")
    prompt = (f"Decode the message.\nCipher: {c.title}. Rule: {rule}; spaces stay.\n{table}\nMessage:\n{ct}\n"
              + ("Answer position by position: its number, ':', the cipher symbol, '=', its plain letter; '/' "
                 "between words." if indexed else
                 "Answer symbol by symbol: each cipher symbol, '=', its plain letter; '/' between words."))
    style = "indexed" if indexed else "trace"
    tdir = "plain_cipher" if inverse else "cipher_plain"
    h = hashlib.md5(f"{c.name}|{kind}|{length}|{style}{shown}|{'inv|' if inverse else ''}{pt}|{ct}".encode()).hexdigest()[:10]
    return {"id": f"ladder_{style}{'_inv' if inverse else ''}_{c.name}_{kind}_{length}_p{shown}_{h}", "table": tdir, "category": f"cipher_{c.name}",
            "task": "ladder", "style": style, "level": _rung(LADDER_PAIRS_SHOWN, shown), "answer_format": f"{c.name}/{kind}/{length}", "prompt": prompt,
            "prefill": prefill, "answer": pt, "trace": answer,
            "rubric": {"type": "exact", "value": pt.lower(), "format": "trace"}, "content_id": f"ladder_{h}",
            "view": c.name, "mode": "known", "pairs_shown": shown, "payload_kind": kind,
            "length": length, "letters": len(pt.replace(" ", "")),
            "notes": f"ladder rung (trace): {c.name}, {kind}, {length} letters, {shown} pairs written"}


def make_cell_rows(cell: dict, n: int, rng: random.Random, style: str = "indexed") -> list:
    """The data improver's entry (training/_common/improve.py): n fresh rows (fresh keys) of one ladder cell, the
    cell being the declared knobs (catalog hint `cells`): view, payload_kind, length, pairs_shown (trace/indexed)
    or revealed (plain), table. A cell the grid never had (a length between two measured ones) is made the same way."""
    c = C.CIPHERS[cell["view"]]
    kind, length = cell["payload_kind"], int(cell["length"])
    if style == "plain":
        return [make_ladder_row(c, kind, length, float(cell["revealed"]), rng) for _ in range(n)]
    shown = int(cell["pairs_shown"])
    return [make_ladder_trace_row(c, kind, length, shown, rng, indexed=style == "indexed",
                                  inverse=cell.get("table") == "plain_cipher") for _ in range(n)]


def run_ladder(args: argparse.Namespace) -> dict:
    """Every cell of cipher x payload kind x length x rung, `--n` rows per cell. Rung = revealed share of the
    plaintext (style plain) or pairs of the answer already written (style trace)."""
    rng = random.Random(args.seed)
    names = args.families.split(",") if args.families != "all" else ["caesar", "script_katakana", "script_cyrillic"]
    lengths = [int(x) for x in args.ladder_lengths.split(",")]
    if args.ladder_style == "plain":
        make, rungs = make_ladder_row, LADDER_REVEALED
    else:
        make = lambda *a: make_ladder_trace_row(*a, indexed=args.ladder_style == "indexed",
                                                inverse=args.ladder_table == "plain_cipher")
        rungs = [int(x) for x in args.ladder_rungs.split(",")] if args.ladder_rungs else LADDER_PAIRS_SHOWN
    rows = [make(C.CIPHERS[n], kind, L, rv, rng)
            for n in names for kind in ("words", "random") for L in lengths for rv in rungs
            for _ in range(args.n)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_ciphers] task=ladder: {len(rows)} rows ({len(rows) // args.n} cells x {args.n}) -> {args.out}")
    return {"out_path": args.out, "n_rows": len(rows)}


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    if getattr(args, "task", "decode") == "ladder":
        return run_ladder(args)
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
