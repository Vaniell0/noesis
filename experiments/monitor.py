#!/usr/bin/env python3
"""Dumb status monitor — read a heartbeat.py status.json and print it.

Standalone on purpose: no repo imports, no dependencies beyond the
standard library, so it keeps working even if something else in the
tree is broken. Point it at any status.json a running job is writing to
(see experiments/_common/heartbeat.py).

Usage:
    python experiments/monitor.py path/to/status.json          # one-shot
    python experiments/monitor.py path/to/status.json --watch   # refresh every 5s until Ctrl-C
    python experiments/monitor.py path/to/status.json --watch --interval 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _print_status(path: Path) -> bool:
    """Print the current status. Returns False if the file is missing/unreadable."""
    if not path.exists():
        print(f"[monitor] {path} does not exist yet — job hasn't written a status yet, or wrong path")
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[monitor] couldn't read {path}: {e}")
        return False

    age_s = time.time() - data.get("ts", 0)
    stale = " (STALE — no update in >5min, job may have died)" if age_s > 300 else ""
    print(f"--- {data.get('ts_human', '?')}  ({age_s:.0f}s ago{stale}) ---")
    for k, v in data.items():
        if k in ("ts", "ts_human"):
            continue
        print(f"  {k}: {v}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Read and print a heartbeat status.json.")
    ap.add_argument("status_file", help="Path to status.json written by heartbeat.write_heartbeat")
    ap.add_argument("--watch", action="store_true", help="Refresh continuously until Ctrl-C")
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between refreshes with --watch")
    args = ap.parse_args()

    path = Path(args.status_file)

    if not args.watch:
        return 0 if _print_status(path) else 1

    try:
        while True:
            _print_status(path)
            print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
