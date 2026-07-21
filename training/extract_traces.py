#!/usr/bin/env python3
"""Extract per-session rollouts from Claude Code jsonl history.

Walks ~/.claude/projects/**/*.jsonl and ~/.claude/history.jsonl. For each
session (one jsonl = one session by convention) collects the ordered chain
of {user, tool_use, tool_result} events, strips assistant thinking/text
that is not a tool_use, keeps only sessions with >= 2 tool_use events,
and emits one file per rollout under training/corpus/raw/<hash>.jsonl.

Rollout schema per emitted file (single JSON line):
    {
        "session_id": str,
        "thinking_stripped": true,
        "chain": [{"role": "user"|"tool_use"|"tool_result", "content": ...}, ...],
        "meta": {"src_path": str, "n_tool_use": int, "n_tool_result": int,
                 "first_ts": str|None, "last_ts": str|None}
    }

Usage:
    python extract_traces.py --dry-run --limit 10
    python extract_traces.py --out training/corpus/raw/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ROOTS = [
    Path.home() / ".claude" / "projects",
    Path.home() / ".claude" / "history.jsonl",
]
SKIP_DIR_NAMES = {"session-env", "cache", "paste-cache"}
SKIP_FILE_NAMES = {".credentials.json"}
TOOL_RESULT_TRUNCATE = 512


def iter_jsonl_paths(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".jsonl":
            if root.name in SKIP_FILE_NAMES:
                continue
            yield root
            continue
        for path in root.rglob("*.jsonl"):
            parts = set(path.parts)
            if parts & SKIP_DIR_NAMES:
                continue
            if path.name in SKIP_FILE_NAMES:
                continue
            yield path


def _content_items(msg_content: Any) -> list[dict[str, Any]]:
    """Normalise assistant/user message content into list-of-dicts."""
    if isinstance(msg_content, str):
        return [{"type": "text", "text": msg_content}]
    if isinstance(msg_content, list):
        return [x for x in msg_content if isinstance(x, dict)]
    return []


def _truncate(s: str, limit: int = TOOL_RESULT_TRUNCATE) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"...<truncated {len(s) - limit} chars>"


def _tool_result_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for sub in content:
            if isinstance(sub, dict):
                if sub.get("type") == "text":
                    parts.append(str(sub.get("text", "")))
                elif "text" in sub:
                    parts.append(str(sub["text"]))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _user_text(item: dict[str, Any]) -> str | None:
    """Extract free-form user text (not tool_result) from a user message item."""
    if item.get("type") == "text":
        t = item.get("text")
        return t if isinstance(t, str) else None
    return None


def build_chain(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (chain, stats) for one session's raw records."""
    chain: list[dict[str, Any]] = []
    n_tool_use = 0
    n_tool_result = 0
    first_ts: str | None = None
    last_ts: str | None = None

    for rec in records:
        ts = rec.get("timestamp") or rec.get("time")
        if isinstance(ts, str):
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        rtype = rec.get("type")
        if rtype not in ("user", "assistant"):
            continue
        msg = rec.get("message") or {}
        role = msg.get("role") or rtype
        items = _content_items(msg.get("content"))

        if role == "user":
            user_texts: list[str] = []
            for item in items:
                itype = item.get("type")
                if itype == "tool_result":
                    body = _truncate(_tool_result_text(item))
                    chain.append({
                        "role": "tool_result",
                        "tool_use_id": item.get("tool_use_id"),
                        "content": body,
                    })
                    n_tool_result += 1
                elif itype == "text":
                    txt = _user_text(item)
                    if txt:
                        user_texts.append(txt)
            joined = "\n".join(t for t in user_texts if t.strip())
            if joined.strip():
                chain.append({"role": "user", "content": joined})
            continue

        if role == "assistant":
            for item in items:
                if item.get("type") == "tool_use":
                    chain.append({
                        "role": "tool_use",
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "input": item.get("input"),
                    })
                    n_tool_use += 1

    return chain, {
        "n_tool_use": n_tool_use,
        "n_tool_result": n_tool_result,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def session_id_from(records: list[dict[str, Any]], path: Path) -> str:
    for rec in records:
        sid = rec.get("sessionId")
        if isinstance(sid, str):
            return sid
    return path.stem


def rollout_hash(session_id: str, src_path: Path) -> str:
    h = hashlib.sha1(f"{session_id}:{src_path}".encode("utf-8")).hexdigest()
    return h[:16]


def process_file(path: Path) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return None
    chain, stats = build_chain(records)
    if stats["n_tool_use"] < 2:
        return None
    sid = session_id_from(records, path)
    return {
        "session_id": sid,
        "thinking_stripped": True,
        "chain": chain,
        "meta": {
            "src_path": str(path),
            **stats,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("training/corpus/raw"),
                   help="output directory for rollouts (default: training/corpus/raw)")
    p.add_argument("--roots", nargs="*", type=Path, default=None,
                   help="jsonl roots (default: ~/.claude/projects + ~/.claude/history.jsonl)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after processing N jsonl files")
    p.add_argument("--dry-run", action="store_true",
                   help="do not write files; print rollout summaries to stdout")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.roots if args.roots else DEFAULT_ROOTS
    out_dir = args.out
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    kept = 0
    dropped = 0
    for path in iter_jsonl_paths(roots):
        if args.limit is not None and total_files >= args.limit:
            break
        total_files += 1
        try:
            rollout = process_file(path)
        except Exception as e:
            print(f"error processing {path}: {e}", file=sys.stderr)
            continue
        if rollout is None:
            dropped += 1
            continue
        kept += 1
        rid = rollout_hash(rollout["session_id"], path)
        if args.dry_run:
            summary = {
                "rid": rid,
                "session_id": rollout["session_id"],
                "n_events": len(rollout["chain"]),
                "n_tool_use": rollout["meta"]["n_tool_use"],
                "src": os.path.basename(rollout["meta"]["src_path"]),
                "first_event_role": rollout["chain"][0]["role"] if rollout["chain"] else None,
            }
            print(json.dumps(summary, ensure_ascii=False))
        else:
            out_path = out_dir / f"{rid}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rollout, f, ensure_ascii=False)
                f.write("\n")

    print(f"scanned={total_files} kept={kept} dropped={dropped} out={out_dir if not args.dry_run else '(dry-run)'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
