#!/usr/bin/env python3
"""Durable serial queue for unfinished probe cells.

    setsid nohup python3 experiments/queue/queue_run.py \
        >> experiments/_logs/queue/runner.log 2>&1 &

Why a separate runner: the gaps (capacity/lens after S1, geometry to 16 ticks,
noise on 2.9B, motion) are independent measurements; they must survive the
agent's shell and resume after a crash. Every job is one argv, its output path
is the idempotency key (an existing non-empty output = done), and a failed job
does not stop the queue.

Behaviour:
  * waits until no file under experiments/rl/results changed for --idle-seconds
    (the previous shell's queue writes there; this one must not sit on top of it);
  * takes a lock file, so only one queue runs;
  * runs one job at a time on E-cores 4-7 (taskset), --threads is the probe's own;
  * logs each job to experiments/_logs/queue/<id>.log, state to queue/state/;
  * `touch experiments/queue/STOP` stops it after the current job.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "experiments/queue"
RESULTS = ROOT / "experiments/rl/results"
LOGS = ROOT / "experiments/_logs/queue"
STATE = QUEUE / "state"
PIDFILE = STATE / "queue.pid"


def log(msg: str) -> None:
    print(f"[queue {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_env() -> str | None:
    """The uv cache holds the torch/rwkv/peft environment the probes were run with.
    Pick any archive that has all three; PYTHONPATH is enough (system python3 is 3.13)."""
    for sp in sorted(Path.home().glob(".cache/uv/archive-v0/*/lib/python3.13/site-packages")):
        if all((sp / p).is_dir() for p in ("torch", "peft", "rwkv")):
            return str(sp)
    return None


def newest_mtime(root: Path) -> float:
    t = 0.0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                t = max(t, p.stat().st_mtime)
        except OSError:
            pass
    return t


def write_status(payload: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(STATE / "status.json")


def job_output(job: dict) -> Path | None:
    out = job.get("out")
    return ROOT / out if out else None


def done(job: dict) -> bool:
    out = job_output(job)
    return bool(out and out.exists() and out.stat().st_size > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, default=QUEUE / "jobs.json")
    ap.add_argument("--idle-seconds", type=int, default=1800,
                    help="wait until experiments/rl/results has been quiet this long")
    ap.add_argument("--only", default="", help="comma-separated job ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-idle-wait", action="store_true")
    args = ap.parse_args()

    jobs = json.loads(args.jobs.read_text())["jobs"]
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        jobs = [j for j in jobs if j["id"] in keep]
    if args.dry_run:
        for j in jobs:
            print(f"{j['id']:14s} {'DONE' if done(j) else 'PENDING':8s} {j.get('out','')}")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    lock = open(QUEUE / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another queue holds the lock; exiting")
        return 0
    PIDFILE.write_text(str(os.getpid()))

    env_path = find_env()
    if env_path is None:
        log("FATAL: no uv archive with torch+peft+rwkv; cannot run probes")
        return 2
    log(f"env: {env_path}")

    if not args.no_idle_wait:
        while True:
            age = time.time() - newest_mtime(RESULTS)
            if age >= args.idle_seconds:
                break
            log(f"waiting for the machine: results activity {age / 60:.1f} min ago")
            time.sleep(60)

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": env_path,
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
    })
    failures: list[tuple[str, int]] = []
    for i, job in enumerate(jobs, 1):
        if (QUEUE / "STOP").exists():
            log("STOP file present; ending")
            break
        if done(job):
            log(f"{i}/{len(jobs)} {job['id']}: already done, skip")
            continue
        logfile = LOGS / f"{job['id']}.log"
        cmd = ["taskset", "-c", "4-7", *job["argv"]]
        log(f"{i}/{len(jobs)} {job['id']}: start -> {logfile}")
        write_status({"job": job["id"], "index": i, "total": len(jobs),
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "log": str(logfile)})
        t0 = time.time()
        try:
            with open(logfile, "w") as f:
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                                    stderr=subprocess.STDOUT,
                                    timeout=job.get("timeout", 21600)).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        log(f"{i}/{len(jobs)} {job['id']}: exit={rc} in {time.time() - t0:.0f}s")
        if rc != 0:
            failures.append((job["id"], rc))
        write_status({"job": job["id"], "index": i, "total": len(jobs), "exit": rc,
                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "seconds": round(time.time() - t0), "log": str(logfile)})
        if done(job):
            job_result = "done"
        elif rc == 0:
            job_result = "exit0-but-no-output"
        else:
            job_result = "failed"
        log(f"    {job['id']}: {job_result}")

    log(f"queue finished; {len(failures)} failure(s): {failures}")
    write_status({"finished": True, "failures": failures})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
