"""Generate tasks_v2.jsonl — programmatically generated, seed-reproducible A0 eval tasks.

Unlike tasks.jsonl (Claude-authored, upper-bound), tasks_v2 answers are computed
exactly from the generation parameters. Same 6 categories, same rubric schema,
same eval harness.

Usage:
    python experiments/A0_eval/gen_tasks_v2.py \
        --out experiments/A0_eval/tasks_v2.jsonl \
        --n 80 \
        --seed 42

Output: JSONL, each line:
  {"id", "category", "prompt", "answer", "rubric": {"type", "value"}, "notes"}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from fractions import Fraction
from pathlib import Path


# ── arithmetic_chain ─────────────────────────────────────────────────────────

ARITH_OPS = [
    ("add {b}", lambda a, b: a + b, "add"),
    ("subtract {b}", lambda a, b: a - b, "subtract"),
    ("multiply by {b}", lambda a, b: a * b, "multiply"),
]


def _gen_arithmetic(rng: random.Random, idx: int) -> dict:
    start = rng.randint(1, 50)
    n_steps = rng.randint(3, 5)
    steps = []
    val = start
    for _ in range(n_steps):
        op_name, op_fn, verb = rng.choice(ARITH_OPS)
        b = rng.randint(1, 20)
        steps.append(op_name.format(b=b))
        val = op_fn(val, b)
    steps_str = ", then ".join(steps)
    prompt = (
        f"Start with {start}. {steps_str.capitalize()}. "
        f"What is the final result? Output only the number."
    )
    return {
        "id": f"v2_arith_{idx:03d}",
        "category": "arithmetic_chain",
        "prompt": prompt,
        "answer": str(val),
        "rubric": {"type": "regex", "value": r"(?<!\d)" + re.escape(str(val)) + r"(?!\d)"},
        "notes": f"chain len={n_steps}, start={start}, result={val}",
    }


# ── bit_decoding ──────────────────────────────────────────────────────────────

GREEK = list("αβγδεζηθικλμνξοπρστυφχψω")
ASCII_SYM = list("@#$%&*!?+=~^<>{}[]|")
ALPHA = list(string.ascii_lowercase)


def _gen_bitsub(rng: random.Random, idx: int) -> dict:
    sym_type, symbols = rng.choice([("greek", GREEK), ("ascii", ASCII_SYM)])
    table_size = rng.choice([4, 6, 8])
    seq_len = rng.choice([2, 3, 4])
    if table_size > len(symbols):
        table_size = len(symbols)
    syms = rng.sample(symbols, table_size)
    letters = rng.sample(ALPHA, len(syms))
    table = dict(zip(syms, letters))
    rev = {v: k for k, v in table.items()}
    word = [rng.choice(letters) for _ in range(seq_len)]
    encoded = "".join(rev[c] for c in word)
    target = "".join(word)
    table_str = ", ".join(f"{s}={c}" for s, c in table.items())
    prompt = (
        f"Given the substitution table {table_str}, "
        f"decode the following: {encoded}. Output only the decoded word."
    )
    return {
        "id": f"v2_bitsub_{idx:03d}",
        "category": "bit_decoding",
        "prompt": prompt,
        "answer": target,
        "rubric": {"type": "contains", "value": target},
        "notes": f"sym={sym_type}, table={table_size}, seq={seq_len}",
    }


def _gen_hex(rng: random.Random, idx: int) -> dict:
    word = "".join(rng.choices(ALPHA, k=rng.randint(3, 6)))
    hex_str = " ".join(f"{ord(c):02x}" for c in word)
    prompt = (
        f"These hex bytes represent ASCII text. "
        f"Decode: {hex_str}. Output only the decoded text."
    )
    return {
        "id": f"v2_hex_{idx:03d}",
        "category": "bit_decoding",
        "prompt": prompt,
        "answer": word,
        "rubric": {"type": "contains", "value": word},
        "notes": f"hex→ascii, word={word}",
    }


def _gen_caesar(rng: random.Random, idx: int) -> dict:
    shift = rng.randint(1, 12)
    words = ["ship", "code", "zero", "jump", "fire", "dark", "move", "east",
             "wind", "mist", "cave", "bolt", "lake", "dusk", "gate", "path"]
    word = rng.choice(words)
    encoded = "".join(chr((ord(c) - ord('a') + shift) % 26 + ord('a')) for c in word)
    prompt = (
        f"The word '{encoded}' is a Caesar cipher shift of an English word. "
        f"Shift N (1 ≤ N ≤ 25). Find and output only the decoded word."
    )
    return {
        "id": f"v2_caesar_{idx:03d}",
        "category": "bit_decoding",
        "prompt": prompt,
        "answer": word,
        "rubric": {"type": "contains", "value": word},
        "notes": f"caesar shift={shift}, word={word}",
    }


# ── extraction ────────────────────────────────────────────────────────────────

FIRST_NAMES = ["Alice", "Bob", "Carol", "Dan", "Eva", "Frank", "Grace", "Hans",
               "Iris", "Jan", "Kira", "Leo", "Mia", "Nico", "Ora", "Pete"]
CITIES = ["Berlin", "Tokyo", "Oslo", "Cairo", "Lima", "Seoul", "Lagos", "Rome"]
JOBS = ["engineer", "analyst", "designer", "researcher", "manager", "developer"]
PRODUCTS = ["Model-A sensor", "T-700 relay", "Mark-IV valve",
            "Type-B adapter", "Series-5 filter", "Unit-X module"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF"]


def _gen_person_extract(rng: random.Random, idx: int) -> dict:
    name = rng.choice(FIRST_NAMES)
    age = rng.randint(22, 65)
    job = rng.choice(JOBS)
    city = rng.choice(CITIES)
    text = f"{name}, aged {age}, works as a {job} in {city}."
    prompt = (
        f"From the text: '{text}' Extract fields into JSON "
        f"with keys name, age, occupation, city. Output only valid JSON."
    )
    answer_obj = {"name": name, "age": age, "occupation": job, "city": city}
    return {
        "id": f"v2_ext_person_{idx:03d}",
        "category": "extraction",
        "prompt": prompt,
        "answer": json.dumps(answer_obj),
        "rubric": {"type": "json_subset", "value": answer_obj},
        "notes": f"person: {name}, {age}, {job}, {city}",
    }


def _gen_order_extract(rng: random.Random, idx: int) -> dict:
    product = rng.choice(PRODUCTS)
    qty = rng.randint(2, 99)
    price = round(rng.uniform(1.5, 99.0), 2)
    currency = rng.choice(CURRENCIES)
    text = f"Order: {qty} units of {product} at {currency} {price:.2f} each."
    prompt = (
        f"Extract from this text into JSON with keys product, quantity, "
        f"unit_price, currency. Text: '{text}'. Output only JSON."
    )
    answer_obj = {"product": product, "quantity": qty,
                  "unit_price": price, "currency": currency}
    return {
        "id": f"v2_ext_order_{idx:03d}",
        "category": "extraction",
        "prompt": prompt,
        "answer": json.dumps(answer_obj),
        "rubric": {"type": "json_subset",
                   "value": {"quantity": qty, "unit_price": price, "currency": currency}},
        "notes": f"order: {product}, qty={qty}, price={price} {currency}",
    }


# ── scheduling ────────────────────────────────────────────────────────────────

def _topological_sort(constraints: list[tuple[str, str]], nodes: list[str],
                      rng: random.Random) -> list[str] | None:
    """Kahn's algorithm; returns None if cycle."""
    from collections import defaultdict, deque
    indegree = {n: 0 for n in nodes}
    adj = defaultdict(list)
    for a, b in constraints:
        adj[a].append(b)
        indegree[b] += 1
    q = deque([n for n in nodes if indegree[n] == 0])
    result = []
    while q:
        # pick randomly among zero-indegree nodes for variety
        items = list(q)
        rng.shuffle(items)
        n = items[0]
        q.remove(n)
        result.append(n)
        for m in adj[n]:
            indegree[m] -= 1
            if indegree[m] == 0:
                q.append(m)
    return result if len(result) == len(nodes) else None


def _gen_scheduling(rng: random.Random, idx: int) -> dict:
    """Generate a total-order scheduling task (unique valid answer)."""
    labels = ["A", "B", "C", "D"]
    n = rng.randint(3, 4)
    order = rng.sample(labels[:n], n)  # the unique valid ordering
    # generate total-order constraints (consecutive pairs = unique solution)
    constraints = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    # optionally add a transitive constraint for flavor
    if n == 4 and rng.random() < 0.5:
        constraints.append((order[0], order[2]))
    rng.shuffle(constraints)
    constraints_str = "; ".join(f"{a} before {b}" for a, b in constraints)
    order_str = " ".join(order)
    nodes_str = ", ".join(sorted(order))
    prompt = (
        f"Schedule tasks {nodes_str} given these constraints: "
        f"{constraints_str}. "
        f"Output the unique valid ordering as a single line, tasks separated by spaces."
    )
    return {
        "id": f"v2_sched_{idx:03d}",
        "category": "scheduling",
        "prompt": prompt,
        "answer": order_str,
        "rubric": {"type": "contains", "value": order_str},
        "notes": f"total order: {order_str}, constraints={constraints}",
    }


# ── string_ops ────────────────────────────────────────────────────────────────

WORDS = ["python", "rwkv", "recurrent", "state", "noesis", "delta",
         "training", "memory", "gradient", "latent", "corpus", "layer"]


def _gen_string_reverse(rng: random.Random, idx: int) -> dict:
    word = rng.choice(WORDS)
    rev = word[::-1]
    prompt = f"Reverse the string '{word}'. Output only the reversed string."
    return {
        "id": f"v2_str_rev_{idx:03d}",
        "category": "string_ops",
        "prompt": prompt,
        "answer": rev,
        "rubric": {"type": "exact", "value": rev},
        "notes": f"reverse: {word} → {rev}",
    }


def _gen_string_count(rng: random.Random, idx: int) -> dict:
    text = rng.choice(WORDS) + " " + rng.choice(WORDS) + " " + rng.choice(WORDS)
    char = rng.choice("aeioustrn")
    count = text.count(char)
    prompt = (
        f"Count the occurrences of the character '{char}' in the string "
        f"'{text}'. Output only the number."
    )
    return {
        "id": f"v2_str_count_{idx:03d}",
        "category": "string_ops",
        "prompt": prompt,
        "answer": str(count),
        "rubric": {"type": "regex",
                   "value": r"(?<!\d)" + re.escape(str(count)) + r"(?!\d)"},
        "notes": f"count '{char}' in '{text}' = {count}",
    }


def _gen_string_replace(rng: random.Random, idx: int) -> dict:
    base = rng.choice(WORDS)
    old_char = rng.choice([c for c in set(base)])
    new_char = rng.choice([c for c in "xyz"])
    result = base.replace(old_char, new_char)
    prompt = (
        f"Replace every occurrence of '{old_char}' with '{new_char}' "
        f"in the string '{base}'. Output only the result."
    )
    return {
        "id": f"v2_str_repl_{idx:03d}",
        "category": "string_ops",
        "prompt": prompt,
        "answer": result,
        "rubric": {"type": "exact", "value": result},
        "notes": f"replace '{old_char}'→'{new_char}' in '{base}' → '{result}'",
    }


def _gen_string_nth(rng: random.Random, idx: int) -> dict:
    word = rng.choice(WORDS)
    pos = rng.randint(1, len(word))
    char = word[pos - 1]
    ordinals = {1: "1st", 2: "2nd", 3: "3rd"}
    ord_str = ordinals.get(pos, f"{pos}th")
    prompt = (
        f"What is the {ord_str} character (1-indexed) of the string '{word}'? "
        f"Output only the character."
    )
    return {
        "id": f"v2_str_nth_{idx:03d}",
        "category": "string_ops",
        "prompt": prompt,
        "answer": char,
        "rubric": {"type": "exact", "value": char},
        "notes": f"nth: '{word}'[{pos}] = '{char}'",
    }


# ── symbolic ──────────────────────────────────────────────────────────────────

def _gen_linear_eq(rng: random.Random, idx: int) -> dict:
    """ax + b = c → x = (c-b)/a, integer solutions only."""
    a = rng.choice([2, 3, 4, 5, 6])
    x_true = rng.randint(-10, 10)
    b = rng.randint(-20, 20)
    c = a * x_true + b
    if b >= 0:
        eq = f"{a}x + {b} = {c}"
    else:
        eq = f"{a}x - {abs(b)} = {c}"
    prompt = f"Solve for x: {eq}. Give only the numeric value."
    return {
        "id": f"v2_sym_lin_{idx:03d}",
        "category": "symbolic",
        "prompt": prompt,
        "answer": str(x_true),
        "rubric": {"type": "regex",
                   "value": r"(?<![0-9\-])" + re.escape(str(x_true)) + r"(?![0-9])"},
        "notes": f"linear: {eq} → x={x_true}",
    }


def _gen_fn_compose(rng: random.Random, idx: int) -> dict:
    """f(x) = ax+b, g(x) = x^2 or cx; compute f(g(n))."""
    a = rng.randint(1, 5)
    b = rng.randint(0, 10)
    n = rng.randint(1, 8)
    use_square = rng.choice([True, False])
    if use_square:
        gn = n * n
        g_expr = f"g(x) = x²"
    else:
        c = rng.randint(2, 5)
        gn = c * n
        g_expr = f"g(x) = {c}x"
    result = a * gn + b
    f_expr = f"f(x) = {a}x + {b}" if b else f"f(x) = {a}x"
    prompt = (
        f"Given {f_expr} and {g_expr}, compute f(g({n})). "
        f"Answer numerically only."
    )
    return {
        "id": f"v2_sym_fn_{idx:03d}",
        "category": "symbolic",
        "prompt": prompt,
        "answer": str(result),
        "rubric": {"type": "regex",
                   "value": r"(?<!\d)" + re.escape(str(result)) + r"(?!\d)"},
        "notes": f"compose: f(g({n})) = {result}",
    }


def _gen_modular(rng: random.Random, idx: int) -> dict:
    a = rng.randint(10, 200)
    m = rng.choice([7, 9, 11, 13, 17])
    result = a % m
    prompt = (
        f"What is {a} mod {m}? "
        f"Output only the numeric remainder."
    )
    return {
        "id": f"v2_sym_mod_{idx:03d}",
        "category": "symbolic",
        "prompt": prompt,
        "answer": str(result),
        "rubric": {"type": "regex",
                   "value": r"(?<!\d)" + re.escape(str(result)) + r"(?!\d)"},
        "notes": f"mod: {a} % {m} = {result}",
    }


# ── dispatcher ────────────────────────────────────────────────────────────────

GENERATORS = {
    "arithmetic_chain": [_gen_arithmetic],
    "bit_decoding":     [_gen_bitsub, _gen_hex, _gen_caesar],
    "extraction":       [_gen_person_extract, _gen_order_extract],
    "scheduling":       [_gen_scheduling],
    "string_ops":       [_gen_string_reverse, _gen_string_count,
                         _gen_string_replace, _gen_string_nth],
    "symbolic":         [_gen_linear_eq, _gen_fn_compose, _gen_modular],
}

# target fraction per category (must sum to 1.0)
FRACTIONS = {
    "arithmetic_chain": 0.15,
    "bit_decoding":     0.25,
    "extraction":       0.15,
    "scheduling":       0.15,
    "string_ops":       0.15,
    "symbolic":         0.15,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/A0_eval/tasks_v2.jsonl")
    ap.add_argument("--n", type=int, default=80, help="Total tasks to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    counters = {cat: 0 for cat in GENERATORS}

    # build pool: (category, generator_fn) pairs proportional to FRACTIONS
    pool: list[tuple[str, callable]] = []
    for cat, frac in FRACTIONS.items():
        target = max(1, int(args.n * frac))
        gens = GENERATORS[cat]
        for i in range(target):
            pool.append((cat, gens[i % len(gens)]))

    rng.shuffle(pool)

    for cat, gen_fn in pool:
        if len(tasks) >= args.n:
            break
        item = gen_fn(rng, counters[cat])
        counters[cat] += 1
        tasks.append(item)

    # pad to --n if needed
    all_gens = [(cat, gf) for cat, gfs in GENERATORS.items() for gf in gfs]
    while len(tasks) < args.n:
        cat, gen_fn = rng.choice(all_gens)
        item = gen_fn(rng, counters[cat])
        counters[cat] += 1
        tasks.append(item)

    rng.shuffle(tasks)

    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Written {len(tasks)} tasks → {out}")
    by_cat = {}
    for t in tasks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    print("Category distribution:", dict(sorted(by_cat.items())))


if __name__ == "__main__":
    main()
