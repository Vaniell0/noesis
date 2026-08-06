"""Convert ToolBench G123-DFS JSON to turns format for tokenize_rollouts.py.

Source: Yhyu13/ToolBench_toolllama_G123_dfs
Format per conversation turn:
    {"from": "system",    "value": "..."}
    {"from": "user",      "value": "..."}
    {"from": "assistant", "value": "Thought: ...\nAction: name\nAction Input: {json}"}
    {"from": "function",  "value": "{json result}"}

Mapping to training turns:
    system   → dropped (AutoGPT preamble not useful)
    user     → {"role": "user", "content": value}
    assistant thought → {"role": "user", "content": "<thought>..."} (context, no loss)
    assistant action  → {"role": "assistant", "tool_use": {"name": ..., "input": {...}}}
    function → {"role": "tool_result", "content": value[:512]}

Loss mask: only assistant turns with tool_use get loss=1, matching action_chains policy.

Usage:
    python training/normalize_toolbench.py \\
        --in  ~/.cache/.../toolllama_G123_dfs_train.json \\
        --out training/corpus_open/toolbench_train.jsonl \\
        --limit 10000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ACTION_RE = re.compile(
    r"Thought:(.*?)(?:\nAction:\s*(\S+)\nAction Input:\s*(\{.*\})\s*$|$)",
    re.DOTALL,
)
_TRUNCATE = 512


def _parse_assistant(value: str) -> tuple[str | None, str | None, dict | None]:
    """Return (thought, action_name, action_input_dict) or Nones if no action."""
    m = _ACTION_RE.search(value.strip())
    if not m:
        return value.strip(), None, None
    thought = (m.group(1) or "").strip()
    name = (m.group(2) or "").strip() or None
    input_raw = (m.group(3) or "").strip()
    if name and input_raw:
        try:
            inp = json.loads(input_raw)
        except json.JSONDecodeError:
            inp = {"raw": input_raw}
        return thought, name, inp
    return thought, None, None


def conv_to_turns(conversations: list[dict]) -> list[dict] | None:
    """Convert ToolBench conversation to turns list. Returns None if unusable."""
    turns: list[dict] = []
    has_tool_use = False

    for item in conversations:
        frm = item.get("from", "")
        val = str(item.get("value", ""))

        if frm == "system":
            continue  # AutoGPT boilerplate

        elif frm == "user":
            turns.append({"role": "user", "content": val.strip()})

        elif frm == "assistant":
            thought, name, inp = _parse_assistant(val)
            if thought:
                # Thought → unsupervised context (wrapped so model can see it)
                turns.append({"role": "user", "content": f"<thought>{thought}"})
            if name and inp is not None:
                turns.append({"role": "assistant",
                               "tool_use": {"name": name, "input": inp}})
                has_tool_use = True

        elif frm == "function":
            content = val[:_TRUNCATE] + ("..." if len(val) > _TRUNCATE else "")
            turns.append({"role": "tool_result", "content": content})

    if not has_tool_use:
        return None
    return turns


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="src", type=Path, required=True,
                   help="path to toolllama_G123_dfs_train.json")
    p.add_argument("--out", type=Path,
                   default=Path("training/corpus_open/toolbench_train.jsonl"))
    p.add_argument("--limit", type=int, default=None,
                   help="cap rollout count (debug / subset selection)")
    args = p.parse_args()

    if not args.src.exists():
        print(f"not found: {args.src}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.load(open(args.src, encoding="utf-8"))
    kept = dropped = 0

    with args.out.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(data):
            if args.limit and kept >= args.limit:
                break
            turns = conv_to_turns(item.get("conversations", []))
            if turns is None:
                dropped += 1
                continue
            rollout = {
                "id": item.get("id", f"tb_{i}"),
                "source": "toolbench_g123_dfs",
                "turns": turns,
            }
            fout.write(json.dumps(rollout, ensure_ascii=False) + "\n")
            kept += 1

    print(f"normalize_toolbench: kept={kept} dropped={dropped} → {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
