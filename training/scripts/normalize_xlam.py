"""Normalize Salesforce/xlam-function-calling-60k into noesis rollouts JSONL.

Variant C hybrid § Primary — the anchor of the A1 fine-tune signal.

Source format (xlam-function-calling-60k, Apache-2.0):
    {"id": ..., "query": "...", "tools": "[{...tool schemas...}]",
     "answers": "[{\"name\": ..., \"arguments\": {...}}, ...]"}
    (`tools` and `answers` are JSON-encoded strings in the released file.)

Target format (matches training/fixtures/tool_call_open.jsonl):
    {"id": "xlam_<id>", "source": "salesforce_xlam_v1", "turns": [
        {"role": "user", "content": "<query>"},
        {"role": "assistant",
         "tool_use": {"name": "<name>", "input": {<arguments>}}},
        ...
    ]}

xlam is single-turn (no tool_result, no assistant follow-up). We emit
the user turn + one tool_use assistant turn per answer. Loss-mask
downstream (per Variant C policy) will target the tool_use tokens only.

Usage (from repo root):
    training/.venv/bin/python training/scripts/normalize_xlam.py \\
        --input <path/to/xlam_function_calling_60k.json> \\
        --output training/corpus_open/xlam_60k.jsonl

If --input is omitted the script tries to resolve
    ~/.cache/huggingface/datasets/xlam_function_calling_60k.json
first, then falls back to hf_hub_download (requires network).

Filters (Variant C corpus-prep pipeline, §2 of
docs/training-data-shortlist.md):
    - drop rows with 0 tool calls in `answers`
    - drop rows where `query` is empty or > MAX_QUERY_CHARS
    - drop rows with suspicious secret patterns (regex belt — not a
      substitute for the full training/sanitize.py pass, but catches
      the obvious ones early so `--audit-only` runs on cleaner data)
    - truncate individual JSON `arguments` fields at MAX_ARG_CHARS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

MAX_QUERY_CHARS = 4096
MAX_ARG_CHARS = 2048

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

DEFAULT_INPUT_CANDIDATES = [
    Path.home() / ".cache/huggingface/xlam-function-calling-60k/xlam_function_calling_60k.json",
    Path.home() / ".cache/huggingface/datasets/xlam_function_calling_60k.json",
    Path("training/corpus_open/xlam_function_calling_60k.json"),
]

HF_REPO_ID = "Salesforce/xlam-function-calling-60k"
HF_FILENAME = "xlam_function_calling_60k.json"


def looks_like_secret(text: str) -> bool:
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            return True
    return False


def parse_field(value: Any) -> Any:
    """xlam encodes `tools` and `answers` as JSON strings; passthrough otherwise."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def normalize_row(row: dict, drop_stats: dict) -> dict | None:
    query = row.get("query", "")
    if not isinstance(query, str) or not query.strip():
        drop_stats["empty_query"] += 1
        return None
    if len(query) > MAX_QUERY_CHARS:
        drop_stats["query_too_long"] += 1
        return None

    answers = parse_field(row.get("answers"))
    if not isinstance(answers, list) or not answers:
        drop_stats["no_answers"] += 1
        return None

    turns: list[dict] = [{"role": "user", "content": query}]
    tool_uses_emitted = 0

    for ans in answers:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name")
        args = ans.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args, dict):
            args = {"_raw": str(args)[:MAX_ARG_CHARS]}
        else:
            args = {k: _truncate_arg(v) for k, v in args.items()}

        turns.append(
            {"role": "assistant", "tool_use": {"name": name, "input": args}}
        )
        tool_uses_emitted += 1

    if tool_uses_emitted == 0:
        drop_stats["no_tool_uses"] += 1
        return None

    joined = json.dumps(turns, ensure_ascii=False)
    if looks_like_secret(joined):
        drop_stats["secret_pattern"] += 1
        return None

    row_id = row.get("id", None)
    if row_id is None:
        row_id = drop_stats["_seq"]
        drop_stats["_seq"] += 1

    return {
        "id": f"xlam_{row_id}",
        "source": "salesforce_xlam_v1",
        "turns": turns,
    }


def _truncate_arg(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_ARG_CHARS:
        return value[:MAX_ARG_CHARS] + "…[truncated]"
    return value


def resolve_input(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            sys.exit(f"input file not found: {p}")
        return p
    for cand in DEFAULT_INPUT_CANDIDATES:
        if cand.exists():
            return cand
    return _hf_download()


def _hf_download() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub not installed. Provide --input <path> or install "
            "huggingface_hub in training/.venv."
        )
    print(f"[normalize_xlam] downloading {HF_REPO_ID}/{HF_FILENAME} via HF hub...")
    path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset"
    )
    return Path(path)


def iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            yield from data
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=None, help="path to xlam JSON file")
    parser.add_argument(
        "--output",
        default="training/corpus_open/xlam_60k.jsonl",
        help="output rollouts JSONL",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N accepted rollouts"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="print first N rollouts to stderr"
    )
    args = parser.parse_args()

    src = resolve_input(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    drop_stats = {
        "empty_query": 0,
        "query_too_long": 0,
        "no_answers": 0,
        "no_tool_uses": 0,
        "secret_pattern": 0,
        "_seq": 0,
    }
    accepted = 0
    seen = 0

    with out_path.open("w", encoding="utf-8") as out:
        for row in iter_rows(src):
            seen += 1
            rollout = normalize_row(row, drop_stats)
            if rollout is None:
                continue
            out.write(json.dumps(rollout, ensure_ascii=False))
            out.write("\n")
            accepted += 1
            if args.sample and accepted <= args.sample:
                print(
                    f"[sample {accepted}] "
                    + json.dumps(rollout, ensure_ascii=False)[:280],
                    file=sys.stderr,
                )
            if args.limit and accepted >= args.limit:
                break

    print(
        f"[normalize_xlam] source={src.name} seen={seen} accepted={accepted}",
        file=sys.stderr,
    )
    for k, v in drop_stats.items():
        if k.startswith("_"):
            continue
        if v > 0:
            print(f"[normalize_xlam] dropped[{k}]={v}", file=sys.stderr)
    print(f"[normalize_xlam] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
