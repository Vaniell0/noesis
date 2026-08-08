"""Convert glaive_v2 / toolbench turns to ReAct plain-CoT JSONL.

Tool-use sequences become <think> spans in ReAct format:
  Thought: ...
  Action: tool_name(arg1=val1, ...)
  Observation: <tool result>
  [repeat for each tool call]
  Thought: I now have enough to answer.

The final assistant content (after all tool calls) is the answer.
Dialogues with no tool calls are emitted as plain SFT (think="").

Output: id, system, user, think, answer — compatible with tokenize_plain_cot.py.

Usage:
    training/.venv/bin/python training/scripts/normalize_react.py \\
        --in  training/corpus_open/glaive_v2.jsonl \\
               training/corpus_open/toolbench_train.jsonl \\
        --out training/corpus_open/react.jsonl \\
        --max-items 25000 \\
        --max-think-words 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

_SYSTEM = (
    "You are a helpful assistant. When you need information, use available tools. "
    "Reason step by step inside <think> before giving your final answer."
)

_MAX_OBS_CHARS = 400  # truncate long tool results in Observation


def _fmt_args(inp: dict | list | str | None) -> str:
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp[:200]
    if isinstance(inp, list):
        return ", ".join(str(x)[:60] for x in inp[:5])
    # dict
    parts = []
    for k, v in list(inp.items())[:6]:
        sv = json.dumps(v) if not isinstance(v, str) else v
        parts.append(f"{k}={sv[:60]!r}")
    return ", ".join(parts)


def _fmt_obs(content: str) -> str:
    if len(content) > _MAX_OBS_CHARS:
        content = content[:_MAX_OBS_CHARS] + "…"
    return content


def _build_react_think(tool_steps: list[dict]) -> str:
    """Convert list of {thought, action_name, action_args, observation} to ReAct text."""
    parts = []
    for i, step in enumerate(tool_steps):
        thought = step.get("thought", f"I need to call {step['action_name']} to proceed.")
        parts.append(f"Thought: {thought}")
        args_str = _fmt_args(step["action_args"])
        call = f"{step['action_name']}({args_str})" if args_str else f"{step['action_name']}()"
        parts.append(f"Action: {call}")
        parts.append(f"Observation: {_fmt_obs(step['observation'])}")
    parts.append("Thought: I now have enough information to answer.")
    return "\n".join(parts)


def _extract_thought(text: str | None, tool_name: str = "", args: dict | None = None) -> str:
    """Generate thought before a tool call."""
    if text:
        text = text.strip()
        m = re.match(r"([^.!?\n]{10,120}[.!?])", text)
        if m:
            return m.group(1)
    # Synthesize from tool name and first arg
    name_readable = tool_name.replace("_", " ")
    if args and isinstance(args, dict):
        first_val = next(iter(args.values()), None)
        if first_val and isinstance(first_val, str):
            return f"I need to {name_readable} for '{first_val[:40]}'."
    return f"I need to use {name_readable} to get this information."


def _segment_to_items(turns: list[dict], base_id: str) -> list[dict]:
    """
    Split a turn list into (user, tool_chain*, answer) segments.
    Each segment anchors at a user turn and collects all subsequent
    tool calls + results until the next user turn.
    """
    items = []
    i = 0
    seg_idx = 0
    while i < len(turns):
        turn = turns[i]
        if turn["role"] != "user":
            i += 1
            continue

        user_text = turn.get("content", "").strip()
        if not user_text:
            i += 1
            continue

        # Collect tool steps and final answer from following turns
        tool_steps: list[dict] = []
        final_answer = ""
        j = i + 1
        pending_action: dict | None = None

        while j < len(turns):
            t = turns[j]
            if t["role"] == "user":
                break  # next user message → end of segment
            if t["role"] == "assistant":
                if "tool_use" in t and t["tool_use"]:
                    tu = t["tool_use"]
                    preceding_thought = _extract_thought(t.get("content"))
                    tname = tu.get("name", "tool")
                    targs = tu.get("input")
                    preceding_thought = _extract_thought(
                        t.get("content"), tname, targs if isinstance(targs, dict) else None
                    )
                    pending_action = {
                        "thought":     preceding_thought,
                        "action_name": tname,
                        "action_args": targs,
                        "observation": "",
                    }
                elif t.get("content"):
                    if pending_action and not pending_action["observation"]:
                        # assistant spoke before observation came in — unlikely but handle
                        pending_action["observation"] = "(no result)"
                        tool_steps.append(pending_action)
                        pending_action = None
                    final_answer = t["content"].strip()
            elif t["role"] == "tool_result":
                obs = t.get("content", "")
                if pending_action is not None:
                    pending_action["observation"] = obs
                    tool_steps.append(pending_action)
                    pending_action = None
                # else: orphan tool_result, ignore
            j += 1

        # Flush dangling action
        if pending_action is not None:
            pending_action["observation"] = pending_action["observation"] or "(no result)"
            tool_steps.append(pending_action)

        if not final_answer and not tool_steps:
            i = j
            continue

        # Build think span
        think = _build_react_think(tool_steps) if tool_steps else ""

        uid = hashlib.md5(f"{base_id}:{seg_idx}:{user_text[:20]}".encode()).hexdigest()[:8]
        items.append({
            "id":     f"react_{base_id}_{uid}",
            "system": _SYSTEM,
            "user":   user_text,
            "think":  think,
            "answer": final_answer,
            "source": "react_tool",
        })
        seg_idx += 1
        i = j

    return items


def _passes_filter(item: dict, max_think_words: int) -> bool:
    if not item["answer"] or len(item["answer"].split()) < 5:
        return False
    think_words = len(item["think"].split())
    if think_words > max_think_words:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="inputs", nargs="+", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--max-items",       type=int, default=25000)
    ap.add_argument("--max-think-words", type=int, default=300)
    ap.add_argument("--seed",            type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load all items from all input files
    all_items: list[dict] = []
    for path in args.inputs:
        print(f"[react] reading {path} …")
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                segs = _segment_to_items(ex.get("turns", []), ex.get("id", "?"))
                all_items.extend(segs)

    print(f"[react] {len(all_items)} raw segments from {len(args.inputs)} files")
    rng.shuffle(all_items)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w") as fout:
        for item in all_items:
            if written >= args.max_items:
                break
            if not _passes_filter(item, args.max_think_words):
                skipped += 1
                continue
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    with_think = sum(1 for _ in open(out_path) if '"action_name"' not in _ and
                     '"think": ""' not in json.loads(_).get("think", "X"))
    print(f"[react] written={written} skipped={skipped}")
    print(f"[react] → {args.out}")


if __name__ == "__main__":
    main()
