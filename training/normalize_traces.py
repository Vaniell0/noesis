"""Convert sanitised extract_traces rollouts to the turns format for tokenize_rollouts.py.

Reads every *.jsonl in corpus/sanitised/ (one rollout per file, chain format)
and writes a single JSONL file (one rollout per line, turns format) to
corpus_open/<name>.jsonl.

Chain format (extract_traces output):
    {"role": "user",        "content": "..."}
    {"role": "tool_use",    "name": "...", "input": {...}, "id": "..."}
    {"role": "tool_result", "content": "...", "tool_use_id": "..."}

Turns format (tokenize_rollouts input, _render_turns compatible):
    {"role": "user",        "content": "..."}
    {"role": "assistant",   "tool_use": {"name": "...", "input": {...}}}
    {"role": "tool_result", "content": "..."}

Loss mask in tokenize_rollouts:
    supervised=True  → role=="assistant" with tool_use present
    supervised=False → user, tool_result

Usage:
    python training/normalize_traces.py \\
        --in  training/corpus/sanitised \\
        --out training/corpus_open/action_chains.jsonl \\
        --name action_chains
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def chain_to_turns(chain: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for item in chain:
        role = item.get("role")
        if role == "user":
            turns.append({"role": "user", "content": item.get("content", "")})
        elif role == "tool_use":
            turns.append({
                "role": "assistant",
                "tool_use": {"name": item.get("name"), "input": item.get("input")},
            })
        elif role == "tool_result":
            turns.append({"role": "tool_result", "content": item.get("content", "")})
    return turns


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="src", type=Path,
                   default=Path("training/corpus/sanitised"),
                   help="directory with sanitised rollout jsonl files")
    p.add_argument("--out", type=Path,
                   default=Path("training/corpus_open/action_chains.jsonl"),
                   help="output jsonl path (one rollout per line)")
    args = p.parse_args()

    src: Path = args.src
    out: Path = args.out
    if not src.exists():
        print(f"source dir not found: {src}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0

    with out.open("w", encoding="utf-8") as fout:
        for f in sorted(src.iterdir()):
            if f.suffix != ".jsonl":
                continue
            try:
                rollout = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"skip {f.name}: {e}", file=sys.stderr)
                dropped += 1
                continue

            chain = rollout.get("chain", [])
            turns = chain_to_turns(chain)
            # drop rollouts with no supervised (tool_use) turns
            if not any(t["role"] == "assistant" and "tool_use" in t for t in turns):
                dropped += 1
                continue

            out_rollout = {
                "id": rollout.get("session_id", f.stem),
                "source": "claude_action_chains",
                "turns": turns,
                "meta": rollout.get("meta", {}),
            }
            fout.write(json.dumps(out_rollout, ensure_ascii=False) + "\n")
            kept += 1

    print(f"normalize_traces: kept={kept} dropped={dropped} → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
