#!/usr/bin/env python3
"""Sanitize extracted rollouts before training.

Two-mode policy:
  * `tool_result` and `user` bodies are context (input, no loss) -> REDACT
    matches in place with `<REDACTED:{pattern}>`. The model consumes
    redacted context, which is fine and consistent.
  * `tool_use` bodies are the training target -> if any match hits here,
    DROP the entire rollout. We do not want the model learning to emit
    `<REDACTED:...>` as an action.

Uses both `detect_secrets` (baseline scanner, high-recall for generic API
keys) and the custom regex families in `sanitize_patterns.py` (targeted for
the specific vendors and shapes we care about).

Usage:
    python sanitize.py --in training/corpus/raw --out training/sanitised
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sanitize_patterns import find_matches, is_critical, match_names, redact

# Plugins we do NOT want firing on user prose:
#   * Base64HighEntropyString / HexHighEntropyString flag every UUID, JWT
#     header, SHA hash, git commit SHA in tool outputs — noise, not signal.
#     The regex layer in sanitize_patterns.py already catches the vendor
#     keys we care about, so entropy plugins add mostly false positives.
#   * KeywordDetector matches on any string containing "password", "secret"
#     etc. — same false-positive story on doc/user text.
# The remaining plugins target concrete formats (AWS, Slack, PGP, JWT, etc.).
_DETECT_SECRETS_DISABLED = {
    "Base64HighEntropyString",
    "HexHighEntropyString",
    "KeywordDetector",
}
try:
    from detect_secrets.core.scan import scan_line  # type: ignore[import-not-found]
    from detect_secrets.settings import transient_settings  # type: ignore[import-not-found]
    from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class  # type: ignore[import-not-found]

    def _ds_config() -> dict:
        plugins = []
        for cls in get_mapping_from_secret_type_to_class().values():
            if cls.__name__ in _DETECT_SECRETS_DISABLED:
                continue
            plugins.append({"name": cls.__name__})
        return {"plugins_used": plugins}

    _HAS_DETECT_SECRETS = True
except ImportError:  # detect-secrets not installed
    _HAS_DETECT_SECRETS = False
    def _ds_config() -> dict:  # type: ignore[misc]
        return {}


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def detect_secrets_hits(text: str) -> list[str]:
    """Return list of detect-secrets plugin names that flagged the text.

    Empty list if detect-secrets is unavailable — pattern layer still runs.
    """
    if not _HAS_DETECT_SECRETS:
        return []
    names: set[str] = set()
    with transient_settings(_ds_config()):
        for line in text.splitlines():
            if not line.strip():
                continue
            for finding in scan_line(line):
                names.add(finding.type)
    return sorted(names)


def process_rollout(rollout: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return (sanitised_rollout_or_None, audit_entry).

    If the rollout is dropped, sanitised_rollout is None. The audit entry
    always records what happened.
    """
    session_id = rollout.get("session_id")
    audit: dict[str, Any] = {
        "session_id": session_id,
        "action": "kept",
        "drops": [],
        "redactions": [],
        "detect_secrets": [],
    }
    chain = rollout.get("chain") or []
    new_chain: list[dict[str, Any]] = []

    for idx, event in enumerate(chain):
        role = event.get("role")
        if role == "tool_use":
            payload = _stringify(event.get("input"))
            names = match_names(payload)
            ds_names = detect_secrets_hits(payload)
            critical = [n for n in names if is_critical(n)]
            if critical or ds_names:
                audit["action"] = "dropped"
                audit["drops"].append({
                    "event_idx": idx,
                    "role": role,
                    "patterns": critical,
                    "info_patterns": [n for n in names if not is_critical(n)],
                    "detect_secrets": ds_names,
                })
                if ds_names:
                    audit["detect_secrets"].extend(ds_names)
                return None, audit
            if names:
                redacted, hits = redact(payload)
                audit["redactions"].append({
                    "event_idx": idx,
                    "role": role,
                    "patterns": names,
                    "detect_secrets": [],
                    "n_hits": len(hits),
                })
                new_event = dict(event)
                try:
                    new_event["input"] = json.loads(redacted)
                except (json.JSONDecodeError, TypeError):
                    new_event["input"] = redacted
                new_chain.append(new_event)
            else:
                new_chain.append(event)
            continue

        if role in ("user", "tool_result"):
            content = event.get("content") or ""
            names = match_names(content)
            ds_names = detect_secrets_hits(content)
            if names or ds_names:
                redacted, hits = redact(content)
                audit["redactions"].append({
                    "event_idx": idx,
                    "role": role,
                    "patterns": names,
                    "detect_secrets": ds_names,
                    "n_hits": len(hits),
                })
                if ds_names:
                    audit["detect_secrets"].extend(ds_names)
                new_event = dict(event)
                new_event["content"] = redacted
                new_chain.append(new_event)
            else:
                new_chain.append(event)
            continue

        new_chain.append(event)

    sanitised = dict(rollout)
    sanitised["chain"] = new_chain
    sanitised.setdefault("meta", {})["sanitised"] = True
    return sanitised, audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", type=Path, default=Path("training/corpus/raw"),
                   help="input directory containing extracted rollouts")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("training/sanitised"),
                   help="output directory for sanitised rollouts + audit log")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N rollouts (for testing)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "audit.jsonl"

    kept = 0
    dropped = 0
    redacted_count = 0
    with audit_path.open("w", encoding="utf-8") as audit_f:
        for i, path in enumerate(sorted(args.in_dir.glob("*.jsonl"))):
            if args.limit is not None and i >= args.limit:
                break
            try:
                with path.open("r", encoding="utf-8") as f:
                    rollout = json.loads(f.readline())
            except Exception as e:
                print(f"skip {path.name}: {e}", file=sys.stderr)
                continue

            sanitised, audit = process_rollout(rollout)
            audit["src_file"] = path.name
            audit_f.write(json.dumps(audit, ensure_ascii=False) + "\n")

            if sanitised is None:
                dropped += 1
                continue
            kept += 1
            if audit["redactions"]:
                redacted_count += 1

            out_path = args.out_dir / path.name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(sanitised, f, ensure_ascii=False)
                f.write("\n")

    if not _HAS_DETECT_SECRETS:
        print("WARNING: detect_secrets not importable; only regex layer ran.",
              file=sys.stderr)
    print(f"kept={kept} dropped={dropped} redacted={redacted_count} audit={audit_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
