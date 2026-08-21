#!/usr/bin/env python3
"""Local-level + trend (constant-velocity) Kalman filter over a training
log's per-step series — gives an online smoothed level + slope
("convergence speed") estimate that's more honest than a raw window
average, since it explicitly carries forward uncertainty instead of
pretending a handful of noisy points is a clean trend.

State x = [level, slope]^T
Transition: level_t = level_{t-1} + slope_{t-1} + w1, slope_t = slope_{t-1} + w2
Observation: y_t = level_t + v_t

Two observation-noise modes:
- --binomial: R_t = p(1-p)/n (accuracy/proportion series, n=N_NOMINAL
  rollouts/step) — the original run8 use case.
- default: R estimated from the series itself (half the variance of
  first differences, the standard local-level heuristic) — for
  continuous losses like answer_ce/state_loss where binomial variance
  doesn't apply.

Moved here from a job-tmp scratch file (2026-08-21) — used every real
training run this session for on-the-spot diagnosis, belongs in the
repo next to the scripts it diagnoses, not off in a temp dir.

Usage:
    python kalman_convergence.py                        # run8's hardcoded 9-point accuracy series (original use)
    python kalman_convergence.py --log FILE              # no --field: auto-detects every numeric field in the log, summary only
    python kalman_convergence.py --log FILE --field answer_ce
    python kalman_convergence.py --log FILE --field answer_ce,state_loss,norm_penalty   # summary only, one call
    python kalman_convergence.py --log FILE --field state_loss --full   # full per-step trace, single field only
"""
import argparse
import json
import re
import sys

RUN8_STEPS = [
    {"step": 51, "accuracy": 0.5},
    {"step": 52, "accuracy": 0.3125},
    {"step": 53, "accuracy": 0.1875},
    {"step": 54, "accuracy": 0.0},
    {"step": 55, "accuracy": 0.0},
    {"step": 56, "accuracy": 0.0625},
    {"step": 57, "accuracy": 0.0},
    {"step": 58, "accuracy": 0.0},
    {"step": 59, "accuracy": 0.0},
]

N_NOMINAL = 8.0        # G=8 rollouts/step, nominal (real n varies w/ OOM attrition)
Q_LEVEL = 1e-4          # process noise: level can drift slowly step to step
Q_SLOPE = 1e-5          # process noise: slope changes even more slowly


def detect_numeric_fields(path: str) -> list[str]:
    """Scan the log for every field that's a plain number on at least one
    line (JSONL only — the free-text .log fallback needs an explicit
    --field since there's no schema to scan). Skips 'step' itself and
    any field that's ever null (e.g. grad_norm before the first clip)."""
    fields: dict[str, bool] = {}  # name -> "seen a non-numeric/null value"
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        for k, v in d.items():
            if k == "step":
                continue
            is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
            fields.setdefault(k, True)
            if not is_num:
                fields[k] = False
    return [k for k, ok in fields.items() if ok]


def load_log_series(path, field):
    """Handles both formats the two log-producing paths actually emit:
    distill_log.jsonl (one JSON object per line, e.g. train_think_distill.py)
    and the stdout-redirect .log files (free text with 'step N: ... field=value').
    Tries JSON first per line, falls back to the regex — so a mixed/partial
    file (e.g. truncated by a VM reboot mid-line) still parses what it can."""
    pat = re.compile(rf"step (\d+): .*{field}=([\d.]+)")
    steps, vals = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if "step" in d and field in d and d[field] is not None:
                steps.append(int(d["step"]))
                vals.append(float(d[field]))
                continue
        except json.JSONDecodeError:
            pass
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            vals.append(float(m.group(2)))
    return [{"step": s, field: v} for s, v in zip(steps, vals)]


def run_kalman(series, key, binomial):
    x = [series[0][key], 0.0]
    P = [[1.0, 0.0], [0.0, 1.0]]

    if not binomial:
        diffs = [series[i][key] - series[i - 1][key] for i in range(1, len(series))]
        R_fixed = (sum(d * d for d in diffs) / len(diffs)) / 2 if diffs else 1.0
        P[0][0] = max(R_fixed, 1e-6)

    results = []
    for i, s in enumerate(series):
        y = s[key]

        if i > 0:
            level_pred = x[0] + x[1]
            slope_pred = x[1]
            p00, p01, p10, p11 = P[0][0], P[0][1], P[1][0], P[1][1]
            P = [
                [p00 + p01 + p10 + p11 + Q_LEVEL, p01 + p11],
                [p10 + p11, p11 + Q_SLOPE],
            ]
            x = [level_pred, slope_pred]

        if binomial:
            p_est = min(max(x[0], 0.02), 0.98)
            R = p_est * (1 - p_est) / N_NOMINAL
        else:
            R = R_fixed

        y_err = y - x[0]
        S = P[0][0] + R
        K0 = P[0][0] / S
        K1 = P[1][0] / S
        x = [x[0] + K0 * y_err, x[1] + K1 * y_err]
        P = [
            [P[0][0] - K0 * P[0][0], P[0][1] - K0 * P[0][1]],
            [P[1][0] - K1 * P[0][0], P[1][1] - K1 * P[0][1]],
        ]

        results.append({
            "step": s["step"], "raw": y,
            "level": round(x[0], 4), "slope": round(x[1], 5),
            "level_std": round(P[0][0] ** 0.5, 4),
            "slope_std": round(P[1][1] ** 0.5, 5),
        })
    return results


class OnlineKalman:
    """Same local-level+trend filter as run_kalman, but incremental — one
    .update(y) call per new observation, for live monitoring inside a
    training loop instead of post-hoc analysis of a full log file.
    Reuses the identical transition math and Q_LEVEL/Q_SLOPE constants.

    One real difference from run_kalman, by necessity: run_kalman knows
    the whole series up front and computes a single fixed observation-
    noise R from all of it before filtering; an online filter can't see
    the future, so R here is re-estimated from the diffs seen *so far*
    on every call (same half-variance-of-diffs formula, growing window).
    Early in a run this makes R noisier — the filter is honestly less
    confident with 3 points than with 300 — which is the correct
    behavior for live monitoring, not a bug to reconcile against the
    batch tool's numbers.

    Added 2026-08-21 so train_think_distill.py can self-monitor
    state_loss and auto-stop on a sustained bad trend instead of
    burning the rest of a run before a human runs kalman_convergence.py
    after the fact and finds the reversal happened hundreds of steps
    ago — exactly what happened twice this session.
    """
    def __init__(self, first_value: float):
        self.x = [first_value, 0.0]
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self._diffs: list = []
        self._prev_value = first_value

    def update(self, y: float) -> dict:
        self._diffs.append(y - self._prev_value)
        self._prev_value = y
        R = max((sum(d * d for d in self._diffs) / len(self._diffs)) / 2, 1e-6)

        level_pred = self.x[0] + self.x[1]
        slope_pred = self.x[1]
        p00, p01, p10, p11 = self.P[0][0], self.P[0][1], self.P[1][0], self.P[1][1]
        self.P = [
            [p00 + p01 + p10 + p11 + Q_LEVEL, p01 + p11],
            [p10 + p11, p11 + Q_SLOPE],
        ]
        self.x = [level_pred, slope_pred]

        y_err = y - self.x[0]
        S = self.P[0][0] + R
        K0 = self.P[0][0] / S
        K1 = self.P[1][0] / S
        self.x = [self.x[0] + K0 * y_err, self.x[1] + K1 * y_err]
        self.P = [
            [self.P[0][0] - K0 * self.P[0][0], self.P[0][1] - K0 * self.P[0][1]],
            [self.P[1][0] - K1 * self.P[0][0], self.P[1][1] - K1 * self.P[0][1]],
        ]
        return {"level": self.x[0], "slope": self.x[1],
                "level_std": self.P[0][0] ** 0.5, "slope_std": self.P[1][1] ** 0.5}


class KalmanTrack:
    """One monitored metric for the online multi-track auto-stop config
    (train_think_distill.py --kalman-config). Wraps an OnlineKalman with
    a *direction* that counts as bad for this specific metric — rising
    is bad for a loss (state_loss, answer_ce), falling is bad for
    something you want to stay high (cos_sim) — and a consecutive-bad-
    check streak, so one noisy check doesn't trigger a stop.

    Added 2026-08-21: the single-field, CLI-flag-only version (still
    supported as a fallback — see train_think_distill.py's
    --kalman-check-every/--kalman-max-rising) only watched state_loss,
    and its thresholds lived in whatever the operator happened to type
    at launch. For a real release run this is a safety mechanism, not a
    convenience knob — it belongs in a checked-in, reviewable config
    (training/config/kalman_watch.yaml), not something a launch command
    can silently omit or mistype.
    """
    def __init__(self, field: str, bad_direction: str, max_bad_streak: int):
        if bad_direction not in ("rising", "falling"):
            raise ValueError(f"bad_direction must be 'rising' or 'falling', got {bad_direction!r}")
        self.field = field
        self.bad_direction = bad_direction
        self.max_bad_streak = max_bad_streak
        self.filter: "OnlineKalman | None" = None
        self.streak = 0

    def update(self, value: float) -> dict:
        if self.filter is None:
            self.filter = OnlineKalman(value)
            return {"field": self.field, "level": value, "slope": 0.0,
                    "slope_std": 0.0, "bad": False, "streak": 0}
        r = self.filter.update(value)
        ci_excludes_zero = abs(r["slope"]) >= r["slope_std"]
        is_bad = ci_excludes_zero and (
            (self.bad_direction == "rising" and r["slope"] > 0)
            or (self.bad_direction == "falling" and r["slope"] < 0)
        )
        self.streak = self.streak + 1 if is_bad else 0
        return {"field": self.field, "level": r["level"], "slope": r["slope"],
                "slope_std": r["slope_std"], "bad": is_bad, "streak": self.streak}

    @property
    def critical(self) -> bool:
        return self.streak >= self.max_bad_streak


def load_kalman_watch_config(path: str) -> dict:
    """Load a training/config/kalman_watch.yaml-style file:
        check_every: 100
        tracks:
          - {field: state_loss, bad_direction: rising, max_bad_streak: 3}
          - {field: answer_ce,  bad_direction: rising, max_bad_streak: 3}
          - {field: cos_sim,    bad_direction: falling, max_bad_streak: 3}
    Returns {"check_every": int, "tracks": [KalmanTrack, ...]}.
    """
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tracks = [
        KalmanTrack(t["field"], t["bad_direction"], t.get("max_bad_streak", 3))
        for t in cfg.get("tracks", [])
    ]
    return {"check_every": cfg.get("check_every", 100), "tracks": tracks}


def report_full(results, label):
    for r in results:
        print(f"step={r['step']:>4}  raw={r['raw']:.4f}  "
              f"level={r['level']:.4f}±{r['level_std']:.4f}  "
              f"slope={r['slope']:+.5f}±{r['slope_std']:.5f} /step")
    report_summary(results, label)


def report_summary(results, label):
    final = results[-1]
    ci_zero = abs(final["slope"]) < final["slope_std"]
    print(f"[{label:>14}] n={len(results):<5} level={final['level']:>10.3f} (±{final['level_std']:.3f})  "
          f"slope={final['slope']:>+9.5f}/step (±{final['slope_std']:.5f})  "
          f"{'flat/noise' if ci_zero else ('rising' if final['slope'] > 0 else 'falling')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="training log to parse (defaults to hardcoded run8 series)")
    ap.add_argument("--field", default=None,
                     help="field name(s), comma-separated (e.g. answer_ce,state_loss). "
                          "Omit with --log to auto-detect every numeric field in the log.")
    ap.add_argument("--binomial", action="store_true", help="use binomial p(1-p)/n obs noise (accuracy series)")
    ap.add_argument("--full", action="store_true",
                     help="print the full per-step trace, not just the final summary. "
                          "Only valid with a single --field.")
    args = ap.parse_args()

    if not args.log:
        results = run_kalman(RUN8_STEPS, "accuracy", binomial=True)
        report_full(results, "accuracy")
        sys.exit(0)

    fields = args.field.split(",") if args.field else detect_numeric_fields(args.log)
    if not fields:
        print(f"no numeric fields found in {args.log} — pass --field explicitly", file=sys.stderr)
        sys.exit(1)

    if args.full and len(fields) != 1:
        print("--full requires exactly one --field", file=sys.stderr)
        sys.exit(1)

    for field in fields:
        series = load_log_series(args.log, field)
        if not series:
            print(f"[{field:>14}] no matches found in {args.log}", file=sys.stderr)
            continue
        results = run_kalman(series, field, args.binomial)
        if args.full:
            report_full(results, field)
        else:
            report_summary(results, field)
