#!/usr/bin/env python3
"""A0.3 — sustained-idle sampler post-processing.

Reads a JSONL produced by ``sampler.py`` and prints a plain-text report
covering:

- RSS trend (first sample, midpoint, last, min/max, %-drift relative
  to first).
- CPU% distribution (mean, p50, p90, p99, max) — informs H1 two-regime
  check.
- Per-zone event rate stability (events per second, computed from
  consecutive samples).
- Retention firings: how many samples showed non-zero ``pruned`` counts.
- Disk footprint per zone (initial, final, delta) + total logs bytes
  (initial, final, delta).

No plotting — writes tables to stdout so it works on a headless idle
machine and pipes cleanly into a markdown verdict file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any, Dict, List, Optional


ZONES = ("input_events", "system_obs", "personal_vault", "session_scratch")


def load(path: pathlib.Path) -> "tuple[Dict[str, Any], List[Dict[str, Any]]]":
    meta: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("meta"):
                meta = obj
            else:
                rows.append(obj)
    return meta, rows


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    kb = n / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KiB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.1f} MiB"
    return f"{mb/1024:.2f} GiB"


def _series(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def rss_report(rows: List[Dict[str, Any]]) -> str:
    rss = _series(rows, "rss_kb")
    if not rss:
        return "RSS: no samples with rss_kb."
    n = len(rss)
    first, last, mid = rss[0], rss[-1], rss[n // 2]
    lo, hi = min(rss), max(rss)
    drift_pct = (last - first) / first * 100.0 if first else 0.0
    return (
        f"RSS (kB): n={n}\n"
        f"  first  = {first:>10.0f}\n"
        f"  mid    = {mid:>10.0f}\n"
        f"  last   = {last:>10.0f}   ({drift_pct:+.1f}% drift vs first)\n"
        f"  min    = {lo:>10.0f}\n"
        f"  max    = {hi:>10.0f}"
    )


def cpu_report(rows: List[Dict[str, Any]]) -> str:
    cpu = _series(rows, "cpu_pct")
    if not cpu:
        return "CPU%: no samples with cpu_pct."
    def q(p: float) -> float:
        srt = sorted(cpu)
        idx = min(len(srt) - 1, max(0, int(round(p * (len(srt) - 1)))))
        return srt[idx]
    return (
        f"CPU% distribution: n={len(cpu)}\n"
        f"  mean = {statistics.fmean(cpu):.2f}\n"
        f"  p50  = {q(0.50):.2f}\n"
        f"  p90  = {q(0.90):.2f}\n"
        f"  p99  = {q(0.99):.2f}\n"
        f"  max  = {max(cpu):.2f}"
    )


def extra_procs_report(rows: List[Dict[str, Any]]) -> str:
    """Summarise the Ollama-child and similar auxiliary process series.

    Sampler stores per-pattern aggregate (sum across matching PIDs) under
    ``extra[<pattern>]`` on every row. We report:

      - live-count distribution (0 vs >=1) — was the process resident?
      - CPU% mean/p50/p90/p99/max across samples that had it live
      - RSS(kB) first / mid / last / min / max
    """
    if not rows or "extra" not in rows[0]:
        return "Extra processes: no extra samples recorded."
    patterns = list(rows[0].get("extra", {}).keys())
    if not patterns:
        return "Extra processes: none tracked."

    def q(series: List[float], p: float) -> float:
        srt = sorted(series)
        idx = min(len(srt) - 1, max(0, int(round(p * (len(srt) - 1)))))
        return srt[idx]

    lines = ["Extra processes (aggregated across matching PIDs):"]
    for pat in patterns:
        cpu_series: List[float] = []
        rss_series: List[float] = []
        n_series: List[int] = []
        for r in rows:
            slot = (r.get("extra") or {}).get(pat)
            if not isinstance(slot, dict):
                continue
            n = slot.get("n")
            if isinstance(n, int):
                n_series.append(n)
            c = slot.get("cpu_pct")
            if isinstance(c, (int, float)):
                cpu_series.append(float(c))
            m = slot.get("rss_kb")
            if isinstance(m, (int, float)):
                rss_series.append(float(m))
        live_frac = (sum(1 for n in n_series if n and n > 0) / len(n_series)) if n_series else 0.0
        max_pids = max(n_series) if n_series else 0
        lines.append(f"  [{pat}] live in {live_frac:.0%} of samples, max concurrent PIDs = {max_pids}")
        if cpu_series:
            lines.append(
                f"      CPU% mean={statistics.fmean(cpu_series):.2f} "
                f"p50={q(cpu_series, 0.50):.2f} "
                f"p90={q(cpu_series, 0.90):.2f} "
                f"p99={q(cpu_series, 0.99):.2f} "
                f"max={max(cpu_series):.2f}"
            )
        if rss_series:
            lines.append(
                f"      RSS(kB) first={rss_series[0]:.0f} "
                f"last={rss_series[-1]:.0f} "
                f"min={min(rss_series):.0f} "
                f"max={max(rss_series):.0f}"
            )
        else:
            lines.append(f"      (no live samples)")
    return "\n".join(lines)


def event_rate_report(rows: List[Dict[str, Any]]) -> str:
    if len(rows) < 2:
        return "Events: need >= 2 samples for a rate."
    lines = ["Per-zone event rate (events / s), first→last:"]
    t0 = rows[0].get("wall_ts")
    t1 = rows[-1].get("wall_ts")
    if not (isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 > t0):
        return "Events: bad wall_ts on first/last samples."
    dt = t1 - t0
    for z in ZONES:
        c0 = (rows[0].get("events") or {}).get(z)
        c1 = (rows[-1].get("events") or {}).get(z)
        if isinstance(c0, int) and isinstance(c1, int):
            rate = (c1 - c0) / dt if dt > 0 else 0.0
            lines.append(f"  {z:<18} {c0:>10} → {c1:>10}  Δ={c1 - c0:>+10}  ({rate:>8.2f}/s)")
        else:
            lines.append(f"  {z:<18} n/a")
    return "\n".join(lines)


def retention_report(rows: List[Dict[str, Any]]) -> str:
    non_zero = 0
    per_zone_nonzero: Dict[str, int] = {z: 0 for z in ZONES}
    last_payload: Optional[Any] = None
    for r in rows:
        rs = r.get("retention_stats_latest")
        if not rs:
            continue
        pl = rs.get("payload") if isinstance(rs, dict) else None
        if not isinstance(pl, dict):
            continue
        last_payload = pl
        pruned = pl.get("pruned") or {}
        row_hit = False
        for z, n in pruned.items():
            if isinstance(n, int) and n > 0:
                per_zone_nonzero[z] = per_zone_nonzero.get(z, 0) + 1
                row_hit = True
        if row_hit:
            non_zero += 1
    lines = [
        f"Retention firings: {non_zero} samples had non-zero prune counts.",
        "  per-zone samples-with-prune:",
    ]
    for z in ZONES:
        lines.append(f"    {z:<18} {per_zone_nonzero.get(z, 0)}")
    if last_payload is not None:
        lines.append(f"  last retention_stats payload: {json.dumps(last_payload, sort_keys=True)}")
    return "\n".join(lines)


def disk_report(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Disk: no samples."
    first = rows[0]
    last = rows[-1]
    lines = ["Disk footprint (per-zone db.sqlite + WAL + SHM):"]
    for z in ZONES:
        b0 = (first.get("db_bytes") or {}).get(z)
        b1 = (last.get("db_bytes") or {}).get(z)
        if isinstance(b0, int) and isinstance(b1, int):
            delta = b1 - b0
            sign = "+" if delta >= 0 else "-"
            lines.append(
                f"  {z:<18} {fmt_bytes(b0):>12} → {fmt_bytes(b1):>12}  "
                f"({sign}{fmt_bytes(abs(delta))})"
            )
        else:
            lines.append(f"  {z:<18} n/a")
    lb0 = first.get("logs_bytes")
    lb1 = last.get("logs_bytes")
    if isinstance(lb0, int) and isinstance(lb1, int):
        delta = lb1 - lb0
        sign = "+" if delta >= 0 else "-"
        lines.append(
            f"  logs/              {fmt_bytes(lb0):>12} → {fmt_bytes(lb1):>12}  "
            f"({sign}{fmt_bytes(abs(delta))})"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.3 sampler post-processing.")
    ap.add_argument("path", type=pathlib.Path, help="JSONL from sampler.py")
    args = ap.parse_args()

    meta, rows = load(args.path)
    if not rows:
        print("No sample rows found.", file=sys.stderr)
        return 1

    hdr = [
        f"=== A0.3 report — {args.path.name} ===",
        f"samples: {len(rows)}",
    ]
    if meta:
        hdr.append(f"start_wall_ts: {meta.get('start_wall_ts')}")
        hdr.append(f"interval_s:    {meta.get('interval_s')}")
        hdr.append(f"duration_s:    {meta.get('duration_s')}")
    t0 = rows[0].get("wall_ts")
    t1 = rows[-1].get("wall_ts")
    if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
        hdr.append(f"wall span:     {t1 - t0:.0f}s ({(t1 - t0) / 3600.0:.2f}h)")
    print("\n".join(hdr))
    print()
    for block in (
        rss_report(rows),
        cpu_report(rows),
        extra_procs_report(rows),
        event_rate_report(rows),
        retention_report(rows),
        disk_report(rows),
    ):
        print(block)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
