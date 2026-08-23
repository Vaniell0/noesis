"""Build a plain, non-task "M=0 relaxation" corpus for Phase 1.5's
curriculum-mixing fix (docs/rl-track.md's Phase 1.5 section, 2026-08-23
design discussion) — real text with no matrix/arithmetic/xor structure,
mixed into every batch alongside the harder M=1..3 examples so the bare
M=0 path doesn't keep degrading the way it did on v3 (LoRA reshaping
weights toward "expect the phase marker" as normal operation).

Two purposes, kept explicitly separate — do not conflate them:
  1. Anti-forgetting (the actual Phase 1.5 need): plain, simple,
     non-task text is enough on its own. Doesn't need injected noise.
  2. Robustness to noisy input (a different, independently valuable
     capability, bundled in here because it's cheap to add, not because
     it's required for (1)): --error-rate injects realistic typos
     (adjacent-key substitution, deletion, transposition) at a
     controllable rate.

Text source: this repo's own git log messages and docs/*.md prose —
real, already-available, zero-cost. NOT synthetic/hallucinated dialect
text — procedurally faking a real dialect would teach the model our
guess at a dialect, not the dialect itself; skipped here, needs a real
corpus if pursued (see docs/rl-track.md's Phase 1.5 note).

Usage:
    training/.venv/bin/python training/scripts/gen_relax_corpus.py \\
        --out training/corpus_open/relax_v1.jsonl \\
        --n 2000 --error-rate 0.0 --seed 7

    # with typo injection at ~5% per-character rate:
    training/.venv/bin/python training/scripts/gen_relax_corpus.py \\
        --out training/corpus_open/relax_v1_noisy.jsonl \\
        --n 2000 --error-rate 0.05 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# QWERTY adjacency, lowercase only — real-world typo source #1
_ADJACENT = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "erfcxs", "e": "wsdr",
    "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg", "i": "ujko", "j": "uikmnh",
    "k": "iolmj", "l": "opk", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def inject_typos(text: str, error_rate: float, rng: random.Random) -> str:
    """Per-character corruption at `error_rate` probability, one of three
    real typo mechanisms chosen uniformly: adjacent-key substitution,
    deletion, or transposition with the next character. Case-preserving
    for substitution (falls back to no-op if the char has no listed
    neighbor, e.g. digits/punctuation)."""
    if error_rate <= 0:
        return text
    chars = list(text)
    i = 0
    out = []
    while i < len(chars):
        c = chars[i]
        if rng.random() < error_rate and c.isalpha():
            kind = rng.choice(("sub", "del", "transpose"))
            lower = c.lower()
            if kind == "sub" and lower in _ADJACENT:
                repl = rng.choice(_ADJACENT[lower])
                out.append(repl.upper() if c.isupper() else repl)
                i += 1
                continue
            if kind == "del":
                i += 1
                continue
            if kind == "transpose" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _git_log_snippets(n: int, rng: random.Random) -> list[str]:
    """Real commit subject+body pairs from this repo's own history —
    genuinely varied register (terse fix messages, longer design
    rationale), zero synthetic content."""
    raw = subprocess.run(
        ["git", "log", "--no-merges", "-n", "2000", "--format=%B%x00"],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    messages = [m.strip() for m in raw.split("\x00") if m.strip()]
    messages = [m for m in messages if 40 <= len(m) <= 600]
    rng.shuffle(messages)
    return messages[:n]


def _docs_prose_snippets(n: int, rng: random.Random) -> list[str]:
    """Paragraph-length prose chunks from docs/*.md, stripped of markdown
    code fences and headers — plain explanatory text, still real, not
    synthetic; excludes matrix-task-shaped content since docs/rl-track.md
    etc. describe those tasks rather than being formatted as one."""
    paras: list[str] = []
    for path in (_REPO_ROOT / "docs").glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para.startswith("#") or para.startswith("|") or para.startswith("-"):
                continue
            if 80 <= len(para) <= 500:
                paras.append(re.sub(r"\s+", " ", para))
    rng.shuffle(paras)
    return paras[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--error-rate", type=float, default=0.0,
                     help="Per-character typo probability (0 = clean text, "
                          "the anti-forgetting-only default). ~0.03-0.08 is "
                          "a realistic human-typing range; higher becomes "
                          "unreadable, not more realistic.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_git = args.n // 2
    n_docs = args.n - n_git
    sources = ([("git_log", t) for t in _git_log_snippets(n_git, rng)]
               + [("docs_prose", t) for t in _docs_prose_snippets(n_docs, rng)])
    rng.shuffle(sources)

    items = []
    for source, text in sources:
        noisy = inject_typos(text, args.error_rate, rng)
        uid = hashlib.md5(f"{source}:{text[:60]}".encode()).hexdigest()[:10]
        items.append({
            "id": f"relax_{source}_{uid}",
            "system": "You are a helpful assistant.",
            "user": f"Continue or summarize the following in one sentence:\n\n{noisy}",
            "think": "",  # explicit: M=0, no phase marker, no <think> content at all
            "answer": text if args.error_rate > 0 else "",  # denoised target only when noisy
            "source": f"relax_{source}",
            "error_rate": args.error_rate,
        })

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[gen_relax_corpus] {len(items)} examples "
          f"({n_git} git_log + {n_docs} docs_prose, error_rate={args.error_rate}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
