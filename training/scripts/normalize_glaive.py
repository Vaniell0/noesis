"""Normalize glaiveai/glaive-function-calling-v2 into noesis rollouts JSONL.

Variant C hybrid § Primary — second anchor after xlam-60k (per
docs/training-data-shortlist.md § 1). ~113k rows, Apache-2.0.

Source format (single JSON array of dicts):
    {"system": "SYSTEM: You are a helpful assistant with access to the
                 following functions...\n{JSON schema}\n",
     "chat":   "USER: ...\n\nASSISTANT: <functioncall> {\"name\": \"...\",
                  \"arguments\": '{...}'} <|endoftext|>\n\nFUNCTION RESPONSE:
                  {...}\n\nASSISTANT: ... <|endoftext|>\n\n"}

Filter policy (Variant C):
    - drop rows whose system prompt says "no access to external functions"
      (row 99346-shape — pure chat with no tool_use).
    - drop rows with zero <functioncall> blocks after parsing.
    - drop rows shorter than MIN_ROLLOUT_CHARS or longer than MAX_ROLLOUT_CHARS.
    - drop rows matching a small secret-pattern regex belt.
    - truncate individual tool_result strings at MAX_TOOL_RESULT_CHARS.

Target format matches training/fixtures/tool_call_open.jsonl exactly:
    {"id": "glaive_<idx>", "source": "glaive_ai_function_calling_v2",
     "turns": [{"role": "user"|"assistant"|"tool_result", ...}]}

Usage (from repo root):
    training/.venv/bin/python training/scripts/normalize_glaive.py \\
        --input training/corpus_open/glaive_function_calling_v2.json \\
        --output training/corpus_open/glaive_v2.jsonl \\
        --sample 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MIN_ROLLOUT_CHARS = 20
MAX_ROLLOUT_CHARS = 12000
MAX_TOOL_RESULT_CHARS = 2048

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

TURN_SPLIT = re.compile(r"\n*(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):)")
# glaive stores tool calls as:
#   <functioncall> {"name": "foo", "arguments": '<inner json>'} <|endoftext|>
# The 'arguments' value is a single-quoted string whose *content* is valid
# JSON. That means the outer envelope is not itself valid JSON (single
# quotes), and the inner braces trip a naive non-greedy regex. So we
# extract the whole <functioncall>…<|endoftext|> block, then pull out
# `name` and `arguments` with two small regexes.
FUNCTIONCALL = re.compile(
    r"<functioncall>\s*(.+?)\s*(?:<\|endoftext\|>|$)",
    re.DOTALL,
)
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_ARGS_STR_RE = re.compile(r"\"arguments\"\s*:\s*'(.*?)'\s*\}\s*$", re.DOTALL)
_ARGS_OBJ_RE = re.compile(r'"arguments"\s*:\s*(\{.*\})\s*\}\s*$', re.DOTALL)
ENDOFTEXT = re.compile(r"\s*<\|endoftext\|>\s*$")
NO_TOOLS = re.compile(r"no access to external functions", re.IGNORECASE)


def looks_like_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def _clean(text: str) -> str:
    text = ENDOFTEXT.sub("", text)
    return text.strip()


def _parse_functioncall(payload: str) -> dict | None:
    """Parse a glaive <functioncall>…</functioncall> body.

    payload is the text between <functioncall> and <|endoftext|>. Two
    accepted shapes:

    A. arguments as single-quoted JSON-encoded string (the common case):
       {"name": "foo", "arguments": '{"a":1,"b":2}'}

    B. arguments as a proper JSON object (rarer):
       {"name": "foo", "arguments": {"a":1,"b":2}}

    Returns {"name": str, "input": dict} or None on parse failure.
    """
    body = payload.strip()
    m_name = _NAME_RE.search(body)
    if not m_name:
        return None
    name = m_name.group(1)

    args: dict | None = None
    m_args = _ARGS_STR_RE.search(body)
    if m_args:
        args_raw = m_args.group(1)
        try:
            parsed = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {"_raw": args_raw[:MAX_TOOL_RESULT_CHARS]}
        else:
            args = parsed if isinstance(parsed, dict) else {"_value": parsed}
    else:
        m_obj = _ARGS_OBJ_RE.search(body)
        if m_obj:
            try:
                parsed = json.loads(m_obj.group(1))
            except json.JSONDecodeError:
                return None
            args = parsed if isinstance(parsed, dict) else {"_value": parsed}
        else:
            # No arguments field at all — treat as empty-input call.
            args = {}

    return {"name": name, "input": args}


def parse_chat(chat: str) -> list[dict] | None:
    if len(chat) < MIN_ROLLOUT_CHARS or len(chat) > MAX_ROLLOUT_CHARS:
        return None

    turns: list[dict] = []
    for block in TURN_SPLIT.split(chat):
        block = block.strip()
        if not block:
            continue
        if block.startswith("USER:"):
            content = _clean(block[len("USER:"):])
            if content:
                turns.append({"role": "user", "content": content})
        elif block.startswith("ASSISTANT:"):
            body = _clean(block[len("ASSISTANT:"):])
            m = FUNCTIONCALL.search(body)
            if m:
                fc = _parse_functioncall(m.group(1))
                prose = _clean(body[: m.start()])
                assistant_turn: dict = {"role": "assistant"}
                if prose:
                    assistant_turn["content"] = prose
                if fc is None:
                    if prose:
                        turns.append(assistant_turn)
                    continue
                assistant_turn["tool_use"] = fc
                turns.append(assistant_turn)
            elif body:
                turns.append({"role": "assistant", "content": body})
        elif block.startswith("FUNCTION RESPONSE:"):
            content = _clean(block[len("FUNCTION RESPONSE:"):])
            if content:
                turns.append(
                    {
                        "role": "tool_result",
                        "content": content[:MAX_TOOL_RESULT_CHARS],
                    }
                )
    return turns if turns else None


def normalize_row(idx: int, row: dict, drop_stats: dict) -> dict | None:
    system = row.get("system", "")
    if isinstance(system, str) and NO_TOOLS.search(system):
        drop_stats["no_tools_in_system"] += 1
        return None

    chat = row.get("chat", "")
    if not isinstance(chat, str) or not chat.strip():
        drop_stats["empty_chat"] += 1
        return None

    turns = parse_chat(chat)
    if turns is None:
        drop_stats["parse_failed_or_oob"] += 1
        return None

    tool_uses = sum(1 for t in turns if t.get("role") == "assistant" and "tool_use" in t)
    if tool_uses == 0:
        drop_stats["no_tool_uses"] += 1
        return None

    joined = json.dumps(turns, ensure_ascii=False)
    if looks_like_secret(joined):
        drop_stats["secret_pattern"] += 1
        return None

    return {
        "id": f"glaive_{idx}",
        "source": "glaive_ai_function_calling_v2",
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[normalize_glaive] loading {src.name}...", file=sys.stderr)
    with src.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    print(f"[normalize_glaive] loaded {len(rows)} rows", file=sys.stderr)

    drop_stats = {
        "no_tools_in_system": 0,
        "empty_chat": 0,
        "parse_failed_or_oob": 0,
        "no_tool_uses": 0,
        "secret_pattern": 0,
    }
    accepted = 0

    with out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            rollout = normalize_row(idx, row, drop_stats)
            if rollout is None:
                continue
            out.write(json.dumps(rollout, ensure_ascii=False))
            out.write("\n")
            accepted += 1
            if args.sample and accepted <= args.sample:
                print(
                    f"[sample {accepted}] "
                    + json.dumps(rollout, ensure_ascii=False)[:400],
                    file=sys.stderr,
                )
            if args.limit and accepted >= args.limit:
                break

    print(
        f"[normalize_glaive] source={src.name} seen={len(rows)} accepted={accepted}",
        file=sys.stderr,
    )
    for k, v in drop_stats.items():
        if v > 0:
            print(f"[normalize_glaive] dropped[{k}]={v}", file=sys.stderr)
    print(f"[normalize_glaive] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
