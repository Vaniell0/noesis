#!/usr/bin/env python3
"""gen_tasks.py — unified matrix task generator for noesis RL curriculum.

Task types (all rendered as ASCII matrices):
  wordsearch   L1–L7   letter grids, find word position
  crossword    L8–L11  multi-word intersecting (enumerate / fill-blank)
  arithmetic   L1–L7   column arithmetic (find sum / missing digit / error)
  pattern      L1–L7   number sequence grid (find missing value)
  bits         L1–L7   binary row operations (XOR/AND/OR/NOT)
  sudoku       —       9×9 grid, fill one blank cell  (--sudoku-csv)
  arc          —       ARC-AGI input→output cell query (--arc-dir)

Usage:
  python3 experiments/A0_eval/gen_tasks.py \\
      --out training/corpus_open/matrix_tasks.jsonl \\
      --n-tokens 20_000_000

  # With external datasets:
  python3 ... --sudoku-csv ~/data/sudoku.csv --arc-dir ~/data/ARC-AGI/data/training
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import random
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from gen_matrix_wordsearch import LEVEL_SPECS as _WS_SPECS, make_task as _ws_make_task
from gen_crossword import XWORD_LEVEL_SPECS, make_type_a_task, make_type_b_task

CHARS_PER_TOK = 3  # rough chars→tokens for WorldTokenizer on ASCII


def _tok_est(task: dict) -> int:
    return (len(task.get("prompt", "")) + len(task.get("answer", ""))) // CHARS_PER_TOK + 200


# ── wordsearch ────────────────────────────────────────────────────────────────

def _wordsearch_gen(rng: random.Random, idx: int) -> Optional[dict]:
    level = rng.randint(1, 7)
    n, mn, mx, orients = _WS_SPECS[level]
    return _ws_make_task(rng, level, level, idx, n, mn, mx, orients, mode="position")


def _wordsearch_name_gen(rng: random.Random, idx: int) -> Optional[dict]:
    level = rng.randint(3, 7)  # L3+ ensures H_RL/V_TD — guessing impossible
    n, mn, mx, orients = _WS_SPECS[level]
    return _ws_make_task(rng, level, level, idx, n, mn, mx, orients, mode="name")


# ── crossword ─────────────────────────────────────────────────────────────────

def _crossword_gen(rng: random.Random, idx: int) -> Optional[dict]:
    level = rng.choice(list(XWORD_LEVEL_SPECS))
    n, n_words, mn, mx, orients, typ = XWORD_LEVEL_SPECS[level]
    if typ == "A":
        return make_type_a_task(rng, level, idx, n, n_words, mn, mx, orients)
    return make_type_b_task(rng, level, idx, n, n_words, mn, mx, orients)


# ── arithmetic_matrix ─────────────────────────────────────────────────────────

_ARITH_SPEC = {
    1: (2, 2, "add", "find_sum"),
    2: (3, 2, "add", "find_sum"),
    3: (4, 2, "add", "find_sum"),
    4: (3, 2, "sub", "find_sum"),
    5: (3, 2, "add", "find_missing"),
    6: (4, 2, "add", "find_error"),
    7: (4, 3, "add", "find_sum"),
}


def _digits(n: int, width: int) -> List[str]:
    s = str(abs(n))
    return [" "] * (width - len(s)) + list(s)


def _render_col_arith(
    addends: List[int], op: str, result: int, width: int,
    blank_row: int = -1, blank_col: int = -1,
    wrong_result: int = -1,
) -> str:
    op_sym = "+" if op == "add" else "-"
    lines = []
    for i, a in enumerate(addends):
        d = _digits(a, width)
        if i == blank_row and blank_col >= 0:
            d[blank_col] = "?"
        prefix = (op_sym if i > 0 else " ") + " "
        lines.append(prefix + " ".join(d))
    lines.append("-" * (2 + width * 2 - 1))
    shown = wrong_result if wrong_result >= 0 else result
    lines.append("  " + " ".join(_digits(shown, width)))
    return "\n".join(lines)


def _arith_gen(rng: random.Random, idx: int) -> Optional[dict]:
    level = rng.randint(1, 7)
    n_digits, n_addends, op, task_type = _ARITH_SPEC[level]
    lo = 10 ** (n_digits - 1)
    hi = 10 ** n_digits - 1

    if op == "sub":
        a = rng.randint(lo, hi)
        b = rng.randint(lo // 10, a - 1)
        addends = [a, b]
        result = a - b
    else:
        addends = [rng.randint(lo // n_addends, hi // n_addends + 1)
                   for _ in range(n_addends)]
        result = sum(addends)

    width = max(len(str(result)), n_digits)
    op_label = "addition" if op == "add" else "subtraction"

    if task_type == "find_sum":
        grid = _render_col_arith(addends, op, result, width)
        lines = grid.split("\n")
        lines[-1] = "  " + " ".join(["?"] * width)
        prompt = (
            f"The matrix shows column {op_label}. Each column is a digit position "
            f"(leading spaces = leading zeros). The result row is '?'.\n\n"
            + "\n".join(lines)
            + "\n\nCompute the result. Output only the number."
        )
        answer = str(result)

    elif task_type == "find_missing":
        m_row = rng.randint(0, n_addends - 1)
        d_list = _digits(addends[m_row], width)
        valid = [i for i, d in enumerate(d_list) if d != " "]
        m_col = rng.choice(valid)
        answer = d_list[m_col]
        grid = _render_col_arith(addends, op, result, width,
                                  blank_row=m_row, blank_col=m_col)
        prompt = (
            f"The matrix shows column {op_label}. One digit is missing ('?'). "
            f"The result row is correct.\n\n{grid}\n\n"
            f"What is the missing digit? Output only the digit."
        )

    else:  # find_error
        res_d = _digits(result, width)
        valid = [i for i, d in enumerate(res_d) if d != " "]
        e_col = rng.choice(valid)
        wrong_d = rng.choice([d for d in "0123456789" if d != res_d[e_col]])
        wr = list(res_d)
        wr[e_col] = wrong_d
        wrong_result = int("".join(c if c != " " else "0" for c in wr))
        grid = _render_col_arith(addends, op, result, width, wrong_result=wrong_result)
        prompt = (
            f"The matrix shows column {op_label}. The result row has one wrong digit. "
            f"What is the correct result?\n\n{grid}\n\nOutput only the number."
        )
        answer = str(result)

    rubric_val = r"(?<!\d)" + re.escape(answer) + r"(?!\d)"
    return {
        "id": f"arith_L{level}_{op}_{task_type}_{idx:06d}",
        "category": "arithmetic_matrix",
        "level": level,
        "prompt": prompt,
        "answer": answer,
        "rubric": {"type": "regex", "value": rubric_val},
        "notes": f"L{level} {op_label} {task_type}, addends={addends}, result={result}",
    }


# ── pattern_matrix ────────────────────────────────────────────────────────────

def _pattern_gen(rng: random.Random, idx: int) -> Optional[dict]:
    level = rng.randint(1, 7)

    if level == 1:  # 1D arithmetic sequence
        start, step, n = rng.randint(1, 20), rng.randint(1, 10), rng.randint(5, 8)
        seq = [start + step * i for i in range(n)]
        bp = rng.randint(1, n - 2)
        answer = seq[bp]
        row = ["?" if i == bp else str(v) for i, v in enumerate(seq)]
        grid = "  ".join(row)
        prompt = (
            f"The sequence follows an arithmetic rule (constant step). "
            f"One value is missing ('?').\n\n{grid}\n\nOutput only the missing number."
        )

    elif level == 2:  # 2D arithmetic
        R, C = rng.randint(3, 5), rng.randint(4, 6)
        sr, sc = rng.randint(1, 5), rng.randint(1, 5)
        s0 = rng.randint(1, 10)
        g = [[s0 + r * sr + c * sc for c in range(C)] for r in range(R)]
        br, bc = rng.randint(0, R - 1), rng.randint(0, C - 1)
        answer = g[br][bc]
        lines = [
            "  ".join("?" if (r == br and c == bc) else str(v)
                      for c, v in enumerate(row))
            for r, row in enumerate(g)
        ]
        grid = "\n".join(lines)
        prompt = (
            f"Each row and column in the matrix follows an arithmetic sequence. "
            f"One value is missing ('?').\n\n{grid}\n\nOutput only the missing number."
        )

    elif level == 3:  # 1D geometric
        ratio, start = rng.randint(2, 4), rng.randint(1, 5)
        n = rng.randint(5, 7)
        seq = [start * (ratio ** i) for i in range(n)]
        bp = rng.randint(1, n - 2)
        answer = seq[bp]
        row = ["?" if i == bp else str(v) for i, v in enumerate(seq)]
        grid = "  ".join(row)
        prompt = (
            f"The sequence follows a geometric rule (multiply by a constant each step). "
            f"One value is missing ('?').\n\n{grid}\n\nOutput only the missing number."
        )

    elif level == 4:  # alternating 0/1
        R, C = rng.randint(3, 5), rng.randint(5, 8)
        offset = rng.randint(0, 1)
        g = [[(r + c + offset) % 2 for c in range(C)] for r in range(R)]
        br, bc = rng.randint(0, R - 1), rng.randint(0, C - 1)
        answer = g[br][bc]
        lines = [
            " ".join("?" if (r == br and c == bc) else str(v)
                     for c, v in enumerate(row))
            for r, row in enumerate(g)
        ]
        grid = "\n".join(lines)
        prompt = (
            f"The matrix contains 0s and 1s in a regular alternating pattern. "
            f"One value is missing ('?').\n\n{grid}\n\nOutput only 0 or 1."
        )

    elif level == 5:  # multiplication table
        R, C = rng.randint(4, 6), rng.randint(4, 6)
        g = [[(r + 1) * (c + 1) for c in range(C)] for r in range(R)]
        br, bc = rng.randint(0, R - 1), rng.randint(0, C - 1)
        answer = g[br][bc]
        lines = [
            "  ".join("?" if (r == br and c == bc) else str(v)
                      for c, v in enumerate(row))
            for r, row in enumerate(g)
        ]
        grid = "\n".join(lines)
        prompt = (
            f"The matrix is a multiplication table (cell = row_index × col_index, 1-indexed). "
            f"One value is missing ('?').\n\n{grid}\n\nOutput only the missing number."
        )

    elif level == 6:  # modular 2D
        R, C = rng.randint(4, 6), rng.randint(5, 8)
        mod = rng.randint(4, 9)
        sr, sc = rng.randint(1, 3), rng.randint(1, 3)
        s0 = rng.randint(0, mod - 1)
        g = [[(s0 + r * sr + c * sc) % mod for c in range(C)] for r in range(R)]
        br, bc = rng.randint(0, R - 1), rng.randint(0, C - 1)
        answer = g[br][bc]
        lines = [
            " ".join("?" if (r == br and c == bc) else str(v)
                     for c, v in enumerate(row))
            for r, row in enumerate(g)
        ]
        grid = "\n".join(lines)
        prompt = (
            f"The matrix follows a modular arithmetic rule (values in 0–{mod - 1}). "
            f"One value is missing ('?').\n\n{grid}\n\n"
            f"Output only the missing number (0–{mod - 1})."
        )

    else:  # level 7: Fibonacci-like
        a0, a1 = rng.randint(1, 5), rng.randint(1, 5)
        n = rng.randint(7, 10)
        seq = [a0, a1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        bp = rng.randint(3, n - 2)
        answer = seq[bp]
        row = ["?" if i == bp else str(v) for i, v in enumerate(seq)]
        grid = "  ".join(row)
        prompt = (
            f"Each value in the sequence equals the sum of the two before it. "
            f"One value is missing ('?').\n\n{grid}\n\nOutput only the missing number."
        )

    answer_str = str(answer)
    rubric_val = r"(?<!\d)" + re.escape(answer_str) + r"(?!\d)"
    return {
        "id": f"pattern_L{level}_{idx:06d}",
        "category": "pattern_matrix",
        "level": level,
        "prompt": prompt,
        "answer": answer_str,
        "rubric": {"type": "regex", "value": rubric_val},
        "notes": f"L{level} pattern, answer={answer}",
    }


# ── bits_matrix ───────────────────────────────────────────────────────────────

_BITS_SPEC = {
    1: ("xor", 4, "result_bit"),
    2: ("and", 4, "result_bit"),
    3: ("or",  4, "result_bit"),
    4: ("not", 4, "result_bit"),
    5: ("xor", 6, "missing_input"),
    6: ("xor", 8, "error_bit"),
    7: ("xor", 8, "key_bit"),
}
_OP_NAME = {"xor": "XOR", "and": "AND", "or": "OR", "not": "NOT"}


def _rb(rng: random.Random, n: int) -> List[int]:
    return [rng.randint(0, 1) for _ in range(n)]


def _bs(bits: List[int]) -> str:
    return " ".join(str(b) for b in bits)


def _bits_gen(rng: random.Random, idx: int) -> Optional[dict]:
    level = rng.randint(1, 7)
    op, n, task_type = _BITS_SPEC[level]
    name = _OP_NAME[op]

    if task_type == "key_bit":
        pt, key = _rb(rng, n), _rb(rng, n)
        ct = [p ^ k for p, k in zip(pt, key)]
        q = rng.randint(0, n - 1)
        key_row = " ".join("?" if i == q else str(v) for i, v in enumerate(key))
        grid = (f"Plaintext  = {_bs(pt)}\n"
                f"Ciphertext = {_bs(ct)}\n"
                f"Key        = {key_row}")
        answer = str(key[q])
        prompt = (
            f"Plaintext was XOR-encrypted with a secret key to produce the ciphertext. "
            f"One key bit is missing.\n\n{grid}\n\n"
            f"What is the key bit at position {q}? Output only 0 or 1."
        )

    elif op == "not":
        a = _rb(rng, n)
        result = [1 - b for b in a]
        q = rng.randint(0, n - 1)
        res_row = " ".join("?" if i == q else str(v) for i, v in enumerate(result))
        grid = f"A     = {_bs(a)}\nNOT A = {res_row}"
        answer = str(result[q])
        prompt = (
            f"The matrix shows bitwise NOT (each bit is flipped).\n\n{grid}\n\n"
            f"What is the missing result bit at position {q}? Output only 0 or 1."
        )

    else:
        a, b = _rb(rng, n), _rb(rng, n)
        if op == "xor":   result = [x ^ y for x, y in zip(a, b)]
        elif op == "and":  result = [x & y for x, y in zip(a, b)]
        else:              result = [x | y for x, y in zip(a, b)]
        sep = "─" * (n * 2 + 14)

        if task_type == "result_bit":
            q = rng.randint(0, n - 1)
            answer = str(result[q])
            res_row = " ".join("?" if i == q else str(v) for i, v in enumerate(result))
            grid = (f"A           = {_bs(a)}\n"
                    f"B           = {_bs(b)}\n"
                    f"{sep}\n"
                    f"A {name} B = {res_row}")
            prompt = (
                f"The matrix shows bitwise {name}.\n\n{grid}\n\n"
                f"What is the missing result bit at position {q}? Output only 0 or 1."
            )

        elif task_type == "missing_input":
            q = rng.randint(0, n - 1)
            answer = str(a[q])
            a_row = " ".join("?" if i == q else str(v) for i, v in enumerate(a))
            grid = (f"A           = {a_row}\n"
                    f"B           = {_bs(b)}\n"
                    f"{sep}\n"
                    f"A {name} B = {_bs(result)}")
            prompt = (
                f"The matrix shows bitwise {name}. One input bit in A is missing.\n\n{grid}\n\n"
                f"What is the missing bit at position {q}? Output only 0 or 1."
            )

        else:  # error_bit
            q = rng.randint(0, n - 1)
            answer = str(result[q])
            wrong = list(result)
            wrong[q] = 1 - wrong[q]
            grid = (f"A           = {_bs(a)}\n"
                    f"B           = {_bs(b)}\n"
                    f"{sep}\n"
                    f"A {name} B = {_bs(wrong)}")
            prompt = (
                f"The matrix shows bitwise {name}. The result row has one wrong bit "
                f"(at position {q}).\n\n{grid}\n\n"
                f"What should bit {q} be? Output only 0 or 1."
            )

    rubric_val = r"(?<!\d)" + re.escape(answer) + r"(?!\d)"
    return {
        "id": f"bits_L{level}_{op}_{task_type}_{idx:06d}",
        "category": "bits_matrix",
        "level": level,
        "prompt": prompt,
        "answer": answer,
        "rubric": {"type": "regex", "value": rubric_val},
        "notes": f"L{level} {name} {task_type}, n={n}, ans={answer}",
    }


# ── sudoku_matrix ─────────────────────────────────────────────────────────────

def _sudoku_gen_factory(csv_path: str) -> Callable:
    pairs: List[Tuple[str, str]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("puzzle") or row.get("quizzes", "")
            s = row.get("solution") or row.get("solutions", "")
            if len(p) == 81 and len(s) == 81:
                pairs.append((p, s))
    print(f"[gen] sudoku: loaded {len(pairs):,} puzzles")

    def _gen(rng: random.Random, idx: int) -> Optional[dict]:
        if not pairs:
            return None
        puzzle, solution = pairs[idx % len(pairs)]
        blanks = [(r, c) for r in range(9) for c in range(9)
                  if puzzle[r * 9 + c] == "0"]
        if not blanks:
            return None
        r, c = rng.choice(blanks)
        answer = solution[r * 9 + c]
        rows_out = []
        for ri in range(9):
            cells = []
            for ci in range(9):
                ch = puzzle[ri * 9 + ci]
                if ri == r and ci == c:
                    cells.append("?")
                elif ch == "0":
                    cells.append(".")
                else:
                    cells.append(ch)
            rows_out.append(" ".join(cells))
        grid = "\n".join(rows_out)
        rubric_val = r"(?<!\d)" + re.escape(answer) + r"(?!\d)"
        return {
            "id": f"sudoku_{idx:06d}_r{r}c{c}",
            "category": "sudoku_matrix",
            "level": 5,
            "prompt": (
                f"The 9×9 matrix is a Sudoku puzzle. '.' = empty cell. "
                f"'?' = the cell you must determine. "
                f"Each row, column, and 3×3 box contains 1–9 exactly once.\n\n"
                f"{grid}\n\n"
                f"What digit goes at row={r} col={c}? Output only the digit."
            ),
            "answer": answer,
            "rubric": {"type": "regex", "value": rubric_val},
            "notes": f"Sudoku cell ({r},{c}), ans={answer}",
        }

    return _gen


# ── arc_matrix ────────────────────────────────────────────────────────────────

def _render_arc_grid(grid: List) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def _arc_gen_factory(arc_dir: str) -> Callable:
    arc_tasks = []
    for p in sorted(Path(arc_dir).glob("*.json")):
        with open(p) as f:
            arc_tasks.append(json.load(f))
    print(f"[gen] ARC: loaded {len(arc_tasks)} tasks")

    def _gen(rng: random.Random, idx: int) -> Optional[dict]:
        if not arc_tasks:
            return None
        data = arc_tasks[idx % len(arc_tasks)]
        train = data.get("train", [])
        test = data.get("test", [])
        if not test:
            return None
        ex = test[0]
        inp = ex.get("input", [])
        out = ex.get("output", [])
        if not inp or not out or not out[0]:
            return None
        ctx_parts = []
        for i, t in enumerate(train[:2]):
            ctx_parts.append(
                f"Example {i + 1} input:\n{_render_arc_grid(t['input'])}\n"
                f"Example {i + 1} output:\n{_render_arc_grid(t['output'])}"
            )
        ctx = "\n\n".join(ctx_parts)
        R, C = len(out), len(out[0])
        r, c = rng.randint(0, R - 1), rng.randint(0, C - 1)
        answer = str(out[r][c])
        rubric_val = r"(?<!\d)" + re.escape(answer) + r"(?!\d)"
        return {
            "id": f"arc_{idx:04d}_r{r}c{c}",
            "category": "arc_matrix",
            "level": min(len(train) + 1, 5),
            "prompt": (
                f"Study the transformation pattern from the examples, "
                f"then apply it to the test input.\n\n{ctx}\n\n"
                f"Test input:\n{_render_arc_grid(inp)}\n\n"
                f"What value is at row={r} col={c} in the test output? "
                f"Output only the digit."
            ),
            "answer": answer,
            "rubric": {"type": "regex", "value": rubric_val},
            "notes": f"ARC task {idx}, cell ({r},{c}), ans={answer}",
        }

    return _gen


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Unified matrix task generator.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tokens", type=int, default=20_000_000,
                    help="Target token estimate (default 20M).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sudoku-csv", default="", metavar="PATH",
                    help="Kaggle 1M sudoku CSV (optional).")
    ap.add_argument("--arc-dir", default="", metavar="PATH",
                    help="ARC-AGI training JSON directory (optional).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Registry: (name, fn, weight)
    generators: List[Tuple[str, Callable, float]] = [
        ("wordsearch",      _wordsearch_gen,      2.0),
        ("wordsearch_name", _wordsearch_name_gen, 1.0),
        ("crossword",       _crossword_gen,       1.0),
        ("arithmetic",      _arith_gen,           2.0),
        ("pattern",         _pattern_gen,         2.0),
        ("bits",            _bits_gen,            2.0),
    ]
    if args.sudoku_csv and Path(args.sudoku_csv).exists():
        generators.append(("sudoku", _sudoku_gen_factory(args.sudoku_csv), 4.0))
    elif args.sudoku_csv:
        print(f"[gen] WARNING: sudoku CSV not found: {args.sudoku_csv}")
    if args.arc_dir and Path(args.arc_dir).exists():
        generators.append(("arc", _arc_gen_factory(args.arc_dir), 1.0))
    elif args.arc_dir:
        print(f"[gen] WARNING: ARC dir not found: {args.arc_dir}")

    total_w = sum(w for _, _, w in generators)
    type_counts: dict = {}
    tasks: List[dict] = []
    total_tok = 0
    idx = 0
    skipped = 0

    print(f"[gen] types: {[n for n,_,_ in generators]}")
    print(f"[gen] target ~{args.n_tokens:,} tokens")

    while total_tok < args.n_tokens:
        x = rng.random() * total_w
        s = 0.0
        chosen_name, chosen_fn = generators[0][0], generators[0][1]
        for name, fn, w in generators:
            s += w
            if x <= s:
                chosen_name, chosen_fn = name, fn
                break

        try:
            task = chosen_fn(rng, idx)
        except Exception:
            skipped += 1
            idx += 1
            continue

        if task is None:
            skipped += 1
            idx += 1
            continue

        tok = _tok_est(task)
        tasks.append(task)
        total_tok += tok
        type_counts[chosen_name] = type_counts.get(chosen_name, 0) + 1
        idx += 1

        if len(tasks) % 5000 == 0:
            print(f"  {len(tasks):,} tasks | ~{total_tok:,} tok")

    rng.shuffle(tasks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\n[gen] {len(tasks):,} tasks → {out_path}")
    print(f"[gen] ~{total_tok:,} estimated tokens  (chars/{CHARS_PER_TOK} + 200/task)")
    print(f"[gen] skipped: {skipped}")
    for name, _, _ in generators:
        print(f"  {name}: {type_counts.get(name, 0):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
