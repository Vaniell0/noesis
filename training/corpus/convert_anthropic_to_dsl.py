#!/usr/bin/env python3
"""Convert sanitised Anthropic-chain rollouts to DSL-inside-think format.

Reads every *.jsonl in corpus/sanitised/ (chain format produced by
extract_traces.py → sanitize.py) and emits one JSONL file where each
line is a rollout whose tool interactions are rendered as noesis DSL
inside <think> blocks.

Input chain format (per sanitised rollout file, single JSON line):
    {
        "session_id": str,
        "chain": [
            {"role": "user",        "content": "..."},
            {"role": "tool_use",    "id": "...", "name": "...", "input": {...}},
            {"role": "tool_result", "tool_use_id": "...", "content": "..."},
            ...
        ]
    }

Output per-line rollout format (one JSON per line in --out):
    {
        "id": str,
        "source": "anthropic_dsl",
        "turns": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "think_dsl": "<dsl block(s)>"},
            ...
        ]
    }

The "think_dsl" field contains one or more consecutive tool_call / tool_result
DSL blocks, meant to be wrapped in <think>…</think> during tokenization.
Step 8 tokenizer masks L_state loss to <think> spans only.

Usage:
    python training/corpus/convert_anthropic_to_dsl.py \\
        --in  training/corpus/sanitised \\
        --out training/corpus_open/action_chains_dsl.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# --- DSL rendering -----------------------------------------------------------

def _dsl_str(s: str) -> str:
    escaped = (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _dsl_value(v, depth: int = 0) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        return s if "." in s or "e" in s else s + ".0"
    if isinstance(v, str):
        return _dsl_str(v)
    if isinstance(v, list):
        if not v:
            return "[]"
        items = ", ".join(_dsl_value(i, depth + 1) for i in v)
        return f"[{items}]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        sep = "  " if depth == 0 else " "
        fields = sep.join(f"{k}={_dsl_value(vv, depth + 1)}" for k, vv in v.items())
        return "{ " + fields + " }"
    return _dsl_str(str(v))


_MAX_CONTENT = 512


def _truncate(s: str, limit: int = _MAX_CONTENT) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"...<+{len(s) - limit}>"


def render_tool_call(name: str, args: dict | None, call_id: str) -> str:
    args = args or {}
    fields = "  ".join(f"{k}={_dsl_value(v)}" for k, v in args.items())
    body = f" {fields} " if fields else " "
    return f"tool_call {name} {{{body}}}"


def render_tool_result(name: str, call_id: str, content: str, ok: bool = True) -> str:
    status = "true" if ok else "false"
    content_val = _dsl_str(_truncate(content))
    return f"tool_result {name} {{ call_id={_dsl_str(call_id)} ok={status} content={content_val} }}"


# --- Conversion logic --------------------------------------------------------

def chain_to_dsl_turns(chain: list[dict]) -> list[dict]:
    """Convert a chain to turns with DSL think blocks.

    Groups consecutive tool_use+tool_result pairs following a user message
    into a single assistant turn with think_dsl field.
    """
    # Build an id→name index from tool_use events for matching tool_results.
    id_to_name: dict[str, str] = {}
    for item in chain:
        if item.get("role") == "tool_use":
            uid = item.get("id") or ""
            name = item.get("name") or "unknown"
            id_to_name[uid] = name

    turns: list[dict] = []
    pending_dsl: list[str] = []

    def flush_pending() -> None:
        if pending_dsl:
            turns.append({"role": "assistant", "think_dsl": "\n".join(pending_dsl)})
            pending_dsl.clear()

    for item in chain:
        role = item.get("role")

        if role == "user":
            flush_pending()
            content = item.get("content", "").strip()
            if content:
                turns.append({"role": "user", "content": content})

        elif role == "tool_use":
            name = item.get("name") or "unknown"
            args = item.get("input") or {}
            call_id = item.get("id") or ""
            pending_dsl.append(render_tool_call(name, args, call_id))

        elif role == "tool_result":
            tool_use_id = item.get("tool_use_id") or ""
            name = id_to_name.get(tool_use_id, "unknown")
            content = item.get("content") or ""
            pending_dsl.append(render_tool_result(name, tool_use_id, content))

    flush_pending()
    return turns


def _has_tool_interaction(turns: list[dict]) -> bool:
    return any(t.get("role") == "assistant" and "think_dsl" in t for t in turns)


# --- Main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", type=Path,
                    default=Path(__file__).parent / "sanitised",
                    help="directory with sanitised chain rollout jsonl files")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "corpus_open" / "action_chains_dsl.jsonl",
                    help="output jsonl path (one rollout per line)")
    ap.add_argument("--min-tool-calls", type=int, default=1,
                    help="drop rollouts with fewer tool interactions than this")
    args = ap.parse_args()

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

            chain = rollout.get("chain") or rollout.get("turns") or []
            if not chain:
                dropped += 1
                continue

            turns = chain_to_dsl_turns(chain)
            n_interactions = sum(1 for t in turns if "think_dsl" in t)
            if n_interactions < args.min_tool_calls:
                dropped += 1
                continue

            out_rollout = {
                "id": rollout.get("session_id") or f.stem,
                "source": "anthropic_dsl",
                "turns": turns,
            }
            fout.write(json.dumps(out_rollout, ensure_ascii=False) + "\n")
            kept += 1

    print(f"converted: kept={kept} dropped={dropped} → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
