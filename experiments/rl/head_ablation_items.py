#!/usr/bin/env python3
"""Items for head_ablation_probe.py — H20, H21, H22 reworked as "which heads hold what"
(owner 2026-10-21). Each item: {"id", "prompt", "conts": {name: text}, "y"?}.

  h20  aporia items (experiments/aporia_probe/items_100.jsonl): the prompt as written, conts
       X / Y = the two incompatible alternatives, OFF = alternative X of the next item (a
       continuation of the same grammatical kind that is not an interpretation of this prompt).
       Question: do X and Y depend on different heads (held side by side) or on the same ones
       (one slot, competition)? OFF gives the overlap any two continuations have anyway.
  h22  the minimal-pair memos (attribution_probe/items_mp_h22.jsonl), read out with a short
       suffix after the memo: binding question "Does the memo say that CLAIM?" and, as a
       control on the same memo, presence question "Does the memo mention ENTITY?" (answer
       always Yes). conts YES / NO. The binding answer needs which-entity-goes-with-which; the
       presence answer needs only the word.
  h21  the minimal-pair questions (items_mp_h21.jsonl), readout "Is this question sensible?",
       conts YES / NO, y = 1 for valid.

The readout is a short suffix fed after the prompt's own state - never the prompt again
(re-feeding the context breaks the WKV update logic; owner 2026-10-20).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "experiments" / "attribution_probe"))
from gen_minimal_pairs import H22_ENTITIES  # noqa: E402

YN = {"YES": " Yes", "NO": " No"}


def h20(src: Path) -> list:
    rows = [json.loads(l) for l in open(src) if l.strip()]
    out = []
    for i, r in enumerate(rows):
        nxt = rows[(i + 1) % len(rows)]
        out.append({"id": r["id"], "category": r["category"], "prompt": r["prompt"],
                    "conts": {"X": " " + r["alternatives"][0], "Y": " " + r["alternatives"][1],
                              "OFF": " " + nxt["alternatives"][0]}})
    return out


def h22(src: Path, chat: bool = False) -> list:
    """chat=True: second readout, the G1 chat template (added after the plain readout failed its
    gate on G1d-0.4B: AUC 0.524, and 'Yes' to only 13% of presence questions whose answer is always
    Yes - the format, not the binding, was not understood)."""
    def q(memo, question):
        if chat:
            return f"User: {memo}\n{question} Answer Yes or No.\n\nAssistant:"
        return f"{memo}\nQuestion: {question}\nAnswer:"
    out = []
    for r in (json.loads(l) for l in open(src) if l.strip()):
        memo, claim_line = r["prompt"].split("\n")
        claim = re.match(r"According to the memo above, (.*)\.$", claim_line).group(1)
        ent = next(e for e in H22_ENTITIES if claim.startswith(e))
        out.append({"id": r["id"], "y": r["y"], "kind": "binding",
                    "prompt": q(memo, f"Does the memo say that {claim}?"), "conts": YN})
        if r["y"] == 1:  # one presence control per memo (both members share the memo)
            out.append({"id": r["id"][:-1] + "p", "y": 1, "kind": "presence",
                        "prompt": q(memo, f"Does the memo mention {ent}?"), "conts": YN})
    return out


def h21(src: Path) -> list:
    return [{"id": r["id"], "y": r["y"], "kind": r["category"],
             "prompt": f"Question: \"{r['prompt']}\"\nIs this a sensible question? Answer:", "conts": YN}
            for r in (json.loads(l) for l in open(src) if l.strip())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["h20", "h21", "h22", "h22chat"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src = {"h20": _ROOT / "experiments/aporia_probe/items_100.jsonl",
           "h21": _ROOT / "experiments/attribution_probe/items_mp_h21.jsonl",
           "h22": _ROOT / "experiments/attribution_probe/items_mp_h22.jsonl"}[args.which.replace("chat", "")]
    rows = {"h20": h20, "h21": h21, "h22": h22, "h22chat": lambda p: h22(p, chat=True)}[args.which](src)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} items -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
