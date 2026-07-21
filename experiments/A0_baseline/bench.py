#!/usr/bin/env python3
"""A0.1 baseline throughput bench for noesis.

Measures prefill tok/s, decode tok/s, TTFT, wall-clock, and llama-server
RSS across a fixed prompt set at three temperature-of-cache phases:

  cold  — first request after `POST /api/generate keep_alive: 0` unload.
  warm  — second request; runner is up, weights loaded, kv-caches empty.
  hot   — steady-state median over the tail of a repeated series.

Prompts are shaped like real noesis workloads (see prompts.py), not toy
Q&A. Zero third-party deps: stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Iterable

import prompts as PROMPT_BANK

DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class Sample:
    prompt_label: str
    phase: str
    ordinal: int
    ttft_s: float
    wall_s: float
    prefill_tokens: int
    prefill_s: float
    decode_tokens: int
    decode_s: float
    rss_mb: float | None

    @property
    def prefill_tps(self) -> float:
        return self.prefill_tokens / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s > 0 else 0.0


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
        return [int(x) for x in out.strip().splitlines() if x.strip().isdigit()]
    except subprocess.CalledProcessError:
        return []


def runner_pid() -> int | None:
    """Return pid of the *largest* llama-server subprocess.

    Ollama spawns a `llama-server` per loaded model; that's where the
    real RSS lives (`--no-mmap` on this Ollama build). More than one
    can be alive (e.g. an embedding model like nomic-embed-text alongside
    the model under test), so we pick the one with the largest VmRSS —
    that's the ~GB-class reasoning model we care about here. Falls back
    to `ollama serve` only if no runner is up (that number is
    misleading — the parent holds no weights).
    """
    candidates = _pgrep("llama-server")
    best_pid: int | None = None
    best_rss = -1.0
    for pid in candidates:
        r = _read_rss_mb(pid)
        if r is not None and r > best_rss:
            best_rss = r
            best_pid = pid
    if best_pid is not None:
        return best_pid
    for pid in _pgrep("ollama serve"):
        return pid
    return None


def _read_rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    return _read_rss_mb(pid)


def _post(host: str, path: str, payload: dict, timeout: float) -> None:
    """Fire-and-forget request; used for keep_alive=0 unload."""
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for _ in resp:
            pass


def unload_model(host: str, model: str, timeout: float = 30.0) -> None:
    """Ask Ollama to unload the model runner. Idempotent."""
    _post(
        host,
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout,
    )
    # give the OS a moment to reap the runner
    for _ in range(20):
        if runner_pid() is None:
            return
        time.sleep(0.25)


def stream_generate(
    host: str,
    model: str,
    prompt: str,
    num_predict: int,
    timeout: float,
) -> tuple[float, float, int, float, int, float]:
    """Return (ttft_s, wall_s, prefill_tokens, prefill_s, decode_tokens, decode_s)."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": num_predict},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    prefill_tokens = 0
    prefill_ns = 0
    decode_tokens = 0
    decode_ns = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            obj = json.loads(line.decode())
            if ttft is None and obj.get("response"):
                ttft = time.perf_counter() - t0
            if obj.get("done"):
                prefill_tokens = obj.get("prompt_eval_count", 0) or 0
                prefill_ns = obj.get("prompt_eval_duration", 0) or 0
                decode_tokens = obj.get("eval_count", 0) or 0
                decode_ns = obj.get("eval_duration", 0) or 0

    wall = time.perf_counter() - t0
    return (
        ttft if ttft is not None else wall,
        wall,
        prefill_tokens,
        prefill_ns / 1e9,
        decode_tokens,
        decode_ns / 1e9,
    )


def run_prompt_series(
    host: str,
    model: str,
    label: str,
    prompt: str,
    hot_repeats: int,
    num_predict: int,
    timeout: float,
) -> list[Sample]:
    """Unload → cold → warm → hot × N. Returns all samples with phase tags."""
    unload_model(host, model)
    samples: list[Sample] = []
    for ordinal, phase in enumerate(
        ["cold", "warm"] + ["hot"] * hot_repeats, start=1
    ):
        ttft, wall, pt, ps, dt, ds = stream_generate(
            host, model, prompt, num_predict, timeout
        )
        rss = rss_mb(runner_pid())
        s = Sample(
            prompt_label=label,
            phase=phase,
            ordinal=ordinal,
            ttft_s=ttft,
            wall_s=wall,
            prefill_tokens=pt,
            prefill_s=ps,
            decode_tokens=dt,
            decode_s=ds,
            rss_mb=rss,
        )
        samples.append(s)
        print(
            f"  [{label} #{ordinal} {phase:<4}] "
            f"ttft={s.ttft_s*1000:.0f} ms "
            f"prefill={s.prefill_tps:.1f} tok/s "
            f"decode={s.decode_tps:.1f} tok/s "
            f"wall={s.wall_s:.2f} s "
            f"(out={s.decode_tokens} tok, rss={s.rss_mb or 0:.0f} MB)",
            flush=True,
        )
    return samples


def summarise(samples: list[Sample]) -> dict:
    """Per (label, phase): median of throughput, ttft, wall, rss."""
    by_key: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        by_key.setdefault((s.prompt_label, s.phase), []).append(s)

    def med(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    rows = []
    for (label, phase), group in by_key.items():
        rows.append(
            {
                "prompt": label,
                "phase": phase,
                "n": len(group),
                "ttft_ms": round(med([s.ttft_s * 1000 for s in group]), 1),
                "prefill_tps": round(med([s.prefill_tps for s in group]), 1),
                "decode_tps": round(med([s.decode_tps for s in group]), 1),
                "wall_s": round(med([s.wall_s for s in group]), 2),
                "prompt_tokens": group[-1].prefill_tokens,
                "decode_tokens": int(med([s.decode_tokens for s in group])),
                "rss_mb": round(med([s.rss_mb for s in group if s.rss_mb]), 1)
                if any(s.rss_mb for s in group)
                else None,
            }
        )
    return rows


def print_summary(rows: list[dict], meta: dict) -> None:
    print()
    print(f"model:    {meta['model']}")
    print(f"host:     {meta['host']}")
    print(f"schedule: cold=1, warm=1, hot={meta['hot_repeats']}")
    print()
    print(
        f"{'prompt':<7} {'phase':<5} {'n':>2} "
        f"{'p_tok':>6} {'ttft ms':>9} "
        f"{'prefill tps':>12} {'decode tps':>11} "
        f"{'rss MB':>7} {'wall s':>7}"
    )
    print("-" * 79)
    order = {"cold": 0, "warm": 1, "hot": 2}
    for row in sorted(rows, key=lambda r: (r["prompt"], order.get(r["phase"], 9))):
        rss = f"{row['rss_mb']:.0f}" if row["rss_mb"] else "  —"
        print(
            f"{row['prompt']:<7} {row['phase']:<5} {row['n']:>2} "
            f"{row['prompt_tokens']:>6} {row['ttft_ms']:>9.0f} "
            f"{row['prefill_tps']:>12.1f} {row['decode_tps']:>11.1f} "
            f"{rss:>7} {row['wall_s']:>7.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=list(PROMPT_BANK.ALL.keys()),
        choices=list(PROMPT_BANK.ALL.keys()),
    )
    ap.add_argument("--hot-repeats", type=int, default=3, help="hot samples per prompt (default: 3)")
    ap.add_argument("--num-predict", type=int, default=256, help="cap on decoded tokens (default: 256)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", help="write full JSON report to this path")
    args = ap.parse_args()

    host = args.host.rstrip("/")

    print(f">>> model={args.model} host={host}")
    for label in args.prompts:
        wc = PROMPT_BANK.word_count(PROMPT_BANK.ALL[label])
        print(f"    prompt {label}: {wc} words")
    print()

    all_samples: list[Sample] = []
    for label in args.prompts:
        prompt = PROMPT_BANK.ALL[label]
        print(f">>> {label} ({PROMPT_BANK.word_count(prompt)} words)")
        try:
            all_samples.extend(
                run_prompt_series(
                    host=host,
                    model=args.model,
                    label=label,
                    prompt=prompt,
                    hot_repeats=args.hot_repeats,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
            )
        except urllib.error.URLError as e:
            print(f"cannot reach Ollama at {host}: {e}", file=sys.stderr)
            return 1

    rows = summarise(all_samples)
    meta = {
        "host": host,
        "model": args.model,
        "hot_repeats": args.hot_repeats,
        "num_predict": args.num_predict,
    }
    print_summary(rows, meta)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "meta": meta,
                    "prompt_word_counts": {
                        label: PROMPT_BANK.word_count(PROMPT_BANK.ALL[label])
                        for label in args.prompts
                    },
                    "phases": rows,
                    "samples": [asdict(s) for s in all_samples],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nsaved: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
