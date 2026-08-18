#!/usr/bin/env python3
"""Aggregate dashboard — one page: system load (CPU/GPU) + every heartbeat.py
status.json under a root directory, most-recently-updated first.

Motivation (2026-08-18): `python -m http.server` over a results directory
gives a raw file listing — no system/GPU load, and finding the one
status.json that's actually live means guessing which subdirectory to open.
This walks the whole tree and renders one page with everything at a glance,
which is what "мониторинг по статусу задач" actually meant.

Usage:
    python experiments/dashboard.py --root . --port 8080
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _system_load() -> dict:
    load1, load5, load15 = os.getloadavg()
    load = {"cpu_load1": round(load1, 2), "cpu_load5": round(load5, 2), "cpu_load15": round(load15, 2)}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            util, mem_used, mem_total, temp = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
            load["gpu_util_pct"] = float(util)
            load["gpu_mem_used_mib"] = float(mem_used)
            load["gpu_mem_total_mib"] = float(mem_total)
            load["gpu_temp_c"] = float(temp)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass  # no GPU / nvidia-smi missing — CPU-only host, load dict just stays smaller
    return load


def _find_statuses(root: Path) -> list[dict]:
    jobs = []
    for p in root.rglob("status.json"):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_path"] = str(p.relative_to(root))
        jobs.append(data)
    jobs.sort(key=lambda d: d.get("ts", 0), reverse=True)
    return jobs


def _render(root: Path) -> str:
    now = time.time()
    load = _system_load()
    jobs = _find_statuses(root)

    load_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in load.items())

    job_html = []
    for j in jobs:
        age = now - j.get("ts", 0)
        stale = age > 300
        progress = j.get("progress")
        if progress and len(progress) == 2 and progress[1]:
            cur, total = progress
            pct = 100.0 * cur / total
            bar = (f'<progress value="{cur}" max="{total}" style="width:100%;height:1.2em"></progress>'
                   f' {cur}/{total} ({pct:.1f}%)')
        else:
            bar = "(no progress reported)"
        msg = j.get("message", "")
        style = "opacity:.5" if stale else ""
        job_html.append(
            f'<div style="margin:1em 0;padding:.5em;border:1px solid #ccc;{style}">'
            f'<b>{j["_path"]}</b> — {age:.0f}s ago{" (STALE)" if stale else ""}<br>'
            f'{bar}<br><span style="color:#666">{msg}</span></div>'
        )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>noesis dashboard</title><meta http-equiv="refresh" content="10">
<style>body{{font-family:monospace;max-width:800px;margin:2em auto}}
table{{border-collapse:collapse}}td{{padding:.3em .8em .3em 0}}</style></head>
<body><h2>noesis — system load</h2><table>{load_rows}</table>
<p style="color:#999">{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))}</p>
<h2>jobs ({len(jobs)})</h2>{"".join(job_html) or "<p>no status.json found yet</p>"}
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def do_GET(self):
        body = _render(self.root).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet — runs unattended


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="Directory to search recursively for status.json files")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    _Handler.root = Path(args.root).resolve()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"[dashboard] serving on :{args.port}, root={_Handler.root}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
