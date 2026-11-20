#!/usr/bin/env python3
"""One-call reading of the closed-loop toy results against the refuters written in the toy plan.

    python experiments/A0_state_probe/toy_digest.py [--dir DIR ...] [--stage 1k]

Each stage below holds its file pattern, the numbers worth one line per run, and its refuter as code (the wording
is in ~/.claude/plans/noesis-closed-loop-toy-2026-11-15.md, the section named by the stage). A stage with fewer
finished runs than its refuter needs says so instead of a verdict.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent / "results" / "closed_loop_toy"
K = range(1, 9)


def load(dirs, pattern):
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob(pattern)):
            if f.name.endswith(".json"):
                try:
                    out[f.stem] = json.loads(f.read_text())
                except json.JSONDecodeError:
                    pass
    return out


def acc(fin, key):
    v = fin.get(key) or {}
    return [v.get(str(k), v.get(k, float("nan"))) for k in K]


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def seed(name):
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else -1


def row(name, xs):
    return f"  {name:28s} " + " ".join(f"{x:.2f}" for x in xs)


def start_fixed(dirs, s):
    """Mean fixed-M8 accuracy of the C checkpoint a stage-1l run started from."""
    r = load(dirs, f"s1_C_M8_seed{s}.json").get(f"s1_C_M8_seed{s}")
    return mean(acc(r["final"], "M8")) if r else float("nan")


def verdict(passed, n, need=2, of=3):
    if n < of:
        return f"{n} of {of} runs finished: no verdict yet"
    return f"{passed} of {n} pass (needs {need})"


def stage_1g(dirs):
    runs = load(dirs, "sG_*.json")
    born = {k: v for k, v in runs.items() if "born" in k}
    l8 = {k: v for k, v in runs.items() if "mimic_L8" in k}
    for k, v in runs.items():
        key = "M8" if "born" in k else "M0"
        print(row(k, acc(v["final"], key)))
    bp = sum(acc(v["final"], "M8")[5] >= 0.5 for v in born.values())
    mp = sum(mean(acc(v["final"], "M0")[5:]) >= 0.5 for v in l8.values())
    print(f"  born-again k6 >= 0.5: {verdict(bp, len(born))} -> {'CONFIRMED' if len(born) >= 3 and bp >= 2 else '-'}")
    print(f"  mimic L8 k6-8 >= 0.5 (refuter): {verdict(mp, len(l8))} -> "
          f"{'REFUTER MET' if len(l8) >= 3 and mp >= 2 else 'not met' if len(l8) >= 3 else '-'}")


def stage_1k(dirs):
    runs = load(dirs, "sK_*.json")
    by = {}
    for k, v in runs.items():
        arm = k.split("_")[1]
        by.setdefault(arm, {})[seed(k)] = acc(v["final"], "M8")
        print(row(k, by[arm][seed(k)]))
    ce = by.get("ce", {})
    for arm in ("g1", "g3"):
        rs = by.get(arm, {})
        ok = sum(rs[s][5] >= 0.5 and s in ce and mean(rs[s][3:]) - mean(ce[s][3:]) >= 0.10 for s in rs)
        print(f"  {arm}: k6 >= 0.5 and k4-8 - ce >= +0.10: {verdict(ok, len(rs))}")
    if "g0" in by and ce:
        d = [mean(by["g0"][s][3:]) - mean(ce[s][3:]) for s in by["g0"] if s in ce]
        print(f"  g0 - ce at k4-8 (prediction |d| <= 0.05): " + " ".join(f"{x:+.3f}" for x in d))


def stage_1l(dirs):
    runs = load(dirs, "sL_*.json")
    for arm in sorted({k.rsplit("_seed", 1)[0] for k in runs}):
        rs = {seed(k): v["final"] for k, v in runs.items() if k.rsplit("_seed", 1)[0] == arm}
        ok = 0
        for s, f in sorted(rs.items()):
            st = start_fixed(dirs, s)
            print(f"  {arm}_seed{s:<3d} own {f['own_mean']:.3f} fixed {f['fixed_mean']:.3f} start {st:.3f} "
                  f"random {f['random_stop_mean']:.3f} share {f['tick_share']:.3f} ticks "
                  + " ".join(f"{f['ticks'][str(k)]:.1f}" for k in K))
            ok += (f["own_mean"] >= st - 0.03 and f["tick_share"] <= 0.85
                   and f["own_mean"] - f["random_stop_mean"] >= 0.10)
        print(f"  {arm}: own within 0.03 of start, share <= 0.85, own - random >= 0.10: {verdict(ok, len(rs))}")


def stage_cipher(dirs):
    real = load(dirs, "s1d_M0_seed0.json").get("s1d_M0_seed0")
    ctrl = load(dirs, "s1d_control_M0_seed0.json").get("s1d_control_M0_seed0")
    if not real or not ctrl:
        print("  control not finished")
        return
    for h in ctrl["history"]:
        r = next((x for x in real["history"] if x["step"] == h["step"]), None)
        if r:
            a, b = r["eval"]["M0"]["char_acc"], h["eval"]["M0"]["char_acc"]
            print(f"  step {h['step']:5d} real {a:.3f} control {b:.3f} read from ciphertext {a - b:+.3f}")


def stage_1h(dirs):
    runs = load(dirs, "sH_*.json")
    runs.update({k.replace("s1_C_M8", "sH_C_w48"): v for k, v in load(dirs, "s1_C_M8_seed*.json").items()})
    widths = {}                                                                 # the base C recipe is width 48
    for k, v in runs.items():
        w = int(re.search(r"_w(\d+)", k).group(1))
        a = acc(v["final"], "M8")
        widths.setdefault(w, []).append(a[5])
        print(row(f"{k} ({v.get('n_params', '?')} p)", a))
    print("  k6 >= 0.5 by width: " + ", ".join(f"w{w} {sum(x >= 0.5 for x in xs)}/{len(xs)}"
                                            for w, xs in sorted(widths.items())))


def stage_1i(dirs):
    for k, v in load(dirs, "sI_*.json").items():
        f = v["final"]
        cap = f.get("capacity_M0") or {}
        lie, new = f.get("M0_known_lie", float("nan")), f.get("M0_new", float("nan"))
        print(f"  {k:28s} clean {f.get('M0_known_clean', float('nan')):.2f} lie {lie:.2f} new {new:.2f} "
              f"repeats lie {f.get('M0_lie_repeated', float('nan')):.2f} capacity {json.dumps(cap)[:60]}"
              f"{'  REFUTER MET (trusts context)' if lie <= 0.5 and new >= 0.9 else ''}")


def stage_1j(dirs):
    for k, v in load(dirs, "sJ_*.json").items():
        f = v["final"]
        print(f"  {k:28s} " + " ".join(f"{key} {val:.3f}" for key, val in f.items() if isinstance(val, float)))


def stage_1n(dirs):
    runs = {k: v for k, v in load(dirs, "sN_*.json").items() if "final" in v}
    if not runs:
        print("  no runs finished")
        return
    fin = {}
    for k, v in sorted(runs.items()):
        f = v["final"]
        M = max(int(x[1:-4]) for x in f if x.startswith("M") and x.endswith("_bpc"))
        b = f[f"M{M}_bpc"]
        fin[k.rsplit("_seed", 1)[0]] = b
        print(f"  {k:24s} M{M} bpc C plain {b['C_plain']:.3f} cyr {b['C_cyr']:.3f} | A cyr {b['A_cyr']:.3f} "
              f"B plain {b['B_plain']:.3f} | cka same {f['cka_same_text']} other {f['cka_other_text']} "
              f"emb_align {f['emb_align']}")
    j, cy, pl, j0 = (fin.get(x) for x in ("sN_joint_M4", "sN_cyr_M4", "sN_plain_M4", "sN_joint_M0"))
    if j and cy:
        d = j["C_cyr"] - cy["C_cyr"]
        print(f"  one world: joint C_cyr - cyr-only = {d:+.3f} bpc -> "
              f"{'CONFIRMED (<= 0.05)' if d <= 0.05 else 'REFUTED (> 0.10)' if d > 0.10 else 'between'}")
    if j and pl:
        d = j["C_plain"] - pl["C_plain"]
        print(f"  capacity: joint C_plain - plain-only = {d:+.3f} bpc -> "
              f"{'only the interface / a second copy (> 0.05)' if d > 0.05 else 'no capacity cost'}")
    if j and j0:
        print(f"  loop share: M4 - M0 C_cyr {j['C_cyr'] - j0['C_cyr']:+.3f}, C_plain {j['C_plain'] - j0['C_plain']:+.3f} bpc")
    ft, rl = fin.get("sN_joint_M4_ceft"), fin.get("sN_joint_M4_rlft")
    if ft and rl:
        d = max(abs(rl[x] - ft[x]) for x in ("C_plain", "C_cyr"))
        print(f"  RL on the ticks vs CE continuation: max |d| {d:.3f} bpc -> {'adds nothing (<= 0.01)' if d <= 0.01 else 'differs'}"
              f" (C_cyr {rl['C_cyr'] - ft['C_cyr']:+.3f}, C_plain {rl['C_plain'] - ft['C_plain']:+.3f})")


def stage_1o(dirs):
    runs = load(dirs, "sO_graft_M8_seed*.json")
    for k, v in runs.items():
        print(row(k + " M8", acc(v["final"], "M8")))
        print(row(k + " M0", acc(v["final"], "M0")))
    ok = sum(acc(v["final"], "M8")[5] >= 0.5 for v in runs.values())
    print(f"  grafted k6 >= 0.5: {verdict(ok, len(runs))}; grown C: 6 of 7 -> "
          f"{'ZLK.16 toy form REFUTED (grafting as reliable)' if len(runs) >= 3 and ok >= 2 else 'supported' if len(runs) >= 3 else '-'}")


def stage_1n3(dirs):
    """Time against scale: deepest-layer CKA (plain vs Cyrillic, same text) and bpc by width and ticks."""
    rows = {}
    for k, v in load(dirs, "sN_joint_M*_seed*.json").items():
        m = re.match(r"sN_joint_M(\d)(?:_w(\d+))?_seed(\d+)$", k)
        if not m or "final" not in v:
            continue
        M, w, sd = int(m.group(1)), int(m.group(2) or 48), int(m.group(3))
        f = v["final"]
        rows.setdefault((w, M), []).append((sd, f["cka_same_text"][-1], f[f"M{M}_bpc"]["C_cyr"], f[f"M{M}_bpc"]["C_plain"]))
    if not rows:
        print("  no runs")
        return
    cka = {}
    for (w, M), xs in sorted(rows.items()):
        c = sum(x[1] for x in xs) / len(xs)
        cka[(w, M)] = c
        print(f"  w{w:<4d} M{M}  seeds {[x[0] for x in xs]}  deep CKA {c:.3f} ({' '.join(f'{x[1]:.2f}' for x in xs)})  "
              f"bpc cyr {sum(x[2] for x in xs) / len(xs):.3f} plain {sum(x[3] for x in xs) / len(xs):.3f}")
    ws = sorted({w for w, _ in cka})
    for i, w in enumerate(ws):
        loop = cka.get((w, 4), float("nan")) - cka.get((w, 0), float("nan"))
        dbl = (cka.get((ws[i + 1], 0), float("nan")) - cka.get((w, 0), float("nan"))) if i + 1 < len(ws) else float("nan")
        print(f"  w{w}: ticks M4 - M0 {loop:+.3f}; next width at M0 {dbl:+.3f}")


STAGES = {"1g": stage_1g, "1k": stage_1k, "1l": stage_1l, "cipher": stage_cipher, "1h": stage_1h, "1i": stage_1i,
          "1j": stage_1j, "1n": stage_1n, "1o": stage_1o, "1n3": stage_1n3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", action="append", default=[], help="results dir (repeatable; default the toy results)")
    ap.add_argument("--stage", choices=list(STAGES), help="one stage only")
    a = ap.parse_args()
    dirs = a.dir or [HERE]
    for name, fn in STAGES.items():
        if a.stage and name != a.stage:
            continue
        print(f"== {name}")
        fn(dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
