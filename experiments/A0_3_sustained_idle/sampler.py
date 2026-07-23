#!/usr/bin/env python3
"""A0.3 — sustained-idle sampler.

Every ``--interval`` seconds, record one JSONL row containing:

- per-zone SQLite event counts under ``~/.local/share/noesis/<zone>/db.sqlite``
- total bytes under ``~/.local/share/noesis/logs``
- ``%CPU %MEM RSS(kB)`` for the noesis-runtime PID (via
  ``systemctl --user show noesis-runtime.service --property=MainPID``)
- the latest ``retention_stats`` payload from ``system_obs`` zone

Rows are appended to ``results/sample-<start_ts>.jsonl``. Use ``report.py``
to post-process.

Design constraints:

- Read-only SQLite access via ``mode=ro`` URI to avoid contending with
  the running runtime (WAL is fine, but PRAGMA-free read is cheapest).
- PID lookup on every sample so the sampler survives a runtime restart.
- No third-party deps; runs against Nixpkgs Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, Optional

DATA_DIR = pathlib.Path.home() / ".local" / "share" / "noesis"
ZONES = ("input_events", "system_obs", "personal_vault", "session_scratch")
LOGS_DIR = DATA_DIR / "logs"
SERVICE = "noesis-runtime.service"
# Ollama child processes worth sampling separately — user reported fans
# spinning under background load, which points to the resident-model
# process, not the noesis-runtime supervisor itself.
EXTRA_PROC_PATTERNS = ("llama-server", "ollama")


def get_main_pid() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "show", SERVICE, "--property=MainPID"],
            text=True, timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if line.startswith("MainPID="):
            val = line.split("=", 1)[1].strip()
            try:
                pid = int(val)
                return pid if pid > 0 else None
            except ValueError:
                return None
    return None


def find_extra_pids() -> Dict[str, list]:
    """Return {pattern: [pid, ...]} for each pattern in EXTRA_PROC_PATTERNS.

    Uses ``pgrep -f`` so we catch e.g. Ollama's ``llama-server`` child even
    when it lives under a different unit than noesis-runtime.
    """
    out: Dict[str, list] = {}
    for pat in EXTRA_PROC_PATTERNS:
        try:
            res = subprocess.check_output(
                ["pgrep", "-f", pat], text=True, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            out[pat] = []
            continue
        pids = []
        for line in res.strip().splitlines():
            try:
                pids.append(int(line.strip()))
            except ValueError:
                pass
        out[pat] = pids
    return out


def ps_stats(pid: int) -> Dict[str, Optional[float]]:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "%cpu=,%mem=,rss=", "-p", str(pid)],
            text=True, timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"cpu_pct": None, "mem_pct": None, "rss_kb": None}
    if not out:
        return {"cpu_pct": None, "mem_pct": None, "rss_kb": None}
    parts = out.split()
    if len(parts) != 3:
        return {"cpu_pct": None, "mem_pct": None, "rss_kb": None}
    try:
        return {
            "cpu_pct": float(parts[0]),
            "mem_pct": float(parts[1]),
            "rss_kb": int(parts[2]),
        }
    except ValueError:
        return {"cpu_pct": None, "mem_pct": None, "rss_kb": None}


def zone_event_count(zone: str) -> Optional[int]:
    db = DATA_DIR / zone / "db.sqlite"
    if not db.exists():
        return None
    uri = f"file:{db}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            cur = conn.execute("SELECT count(*) FROM events")
            return int(cur.fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def zone_db_bytes(zone: str) -> int:
    db_dir = DATA_DIR / zone
    if not db_dir.exists():
        return 0
    total = 0
    for p in db_dir.iterdir():
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def logs_bytes() -> int:
    if not LOGS_DIR.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(LOGS_DIR):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def latest_retention_stats() -> Optional[Dict[str, Any]]:
    db = DATA_DIR / "system_obs" / "db.sqlite"
    if not db.exists():
        return None
    uri = f"file:{db}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            cur = conn.execute(
                "SELECT ts_us, payload FROM events "
                "WHERE kind = 'retention_stats' "
                "ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                return None
            ts_us, payload = row
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = payload
            return {"ts_us": int(ts_us), "payload": parsed}
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def one_sample() -> Dict[str, Any]:
    now_wall = time.time()
    pid = get_main_pid()
    stats = ps_stats(pid) if pid else {"cpu_pct": None, "mem_pct": None, "rss_kb": None}
    counts = {z: zone_event_count(z) for z in ZONES}
    db_bytes = {z: zone_db_bytes(z) for z in ZONES}
    # Extra processes (Ollama child etc.) — aggregate per pattern so
    # multi-instance cases (e.g. two llama-server workers) are captured.
    extra_raw = find_extra_pids()
    extra_stats: Dict[str, Dict[str, Any]] = {}
    for pat, pids in extra_raw.items():
        agg = {"pids": pids, "cpu_pct": 0.0, "mem_pct": 0.0, "rss_kb": 0, "n": len(pids)}
        for p in pids:
            s = ps_stats(p)
            for k in ("cpu_pct", "mem_pct", "rss_kb"):
                v = s.get(k)
                if isinstance(v, (int, float)):
                    agg[k] += v
        if not pids:
            agg["cpu_pct"] = None
            agg["mem_pct"] = None
            agg["rss_kb"] = None
        extra_stats[pat] = agg
    return {
        "wall_ts": now_wall,
        "pid": pid,
        **stats,
        "extra": extra_stats,
        "events": counts,
        "db_bytes": db_bytes,
        "logs_bytes": logs_bytes(),
        "retention_stats_latest": latest_retention_stats(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.3 sustained-idle sampler.")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="Sampling interval in seconds (default: 60).")
    ap.add_argument("--duration", type=float, default=24 * 3600.0,
                    help="Total run duration in seconds (default: 24h).")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path. Default: results/sample-<start_ts>.jsonl.")
    ap.add_argument("--flush-every", type=int, default=1,
                    help="Flush the file after every N samples (default: 1 — every sample).")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    results_dir = here / "results"
    results_dir.mkdir(exist_ok=True)

    start_ts = int(time.time())
    out_path = pathlib.Path(args.out) if args.out else results_dir / f"sample-{start_ts}.jsonl"

    print(f"[sampler] out={out_path}", file=sys.stderr, flush=True)
    print(f"[sampler] interval={args.interval}s duration={args.duration}s",
          file=sys.stderr, flush=True)

    deadline = time.time() + args.duration
    n = 0
    with out_path.open("a") as f:
        # Header — one meta row per run so report.py can pin start conditions.
        meta = {
            "meta": True,
            "start_wall_ts": time.time(),
            "interval_s": args.interval,
            "duration_s": args.duration,
            "sampler_pid": os.getpid(),
            "python": sys.version.split()[0],
            "data_dir": str(DATA_DIR),
        }
        f.write(json.dumps(meta) + "\n")
        f.flush()

        while time.time() < deadline:
            t0 = time.time()
            try:
                sample = one_sample()
            except Exception as e:  # noqa: BLE001
                sample = {"wall_ts": time.time(), "error": str(e)}
            f.write(json.dumps(sample) + "\n")
            n += 1
            if n % args.flush_every == 0:
                f.flush()
            elapsed = time.time() - t0
            sleep_for = args.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print(f"[sampler] wrote {n} samples to {out_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
