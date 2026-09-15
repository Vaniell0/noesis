#!/usr/bin/env python3
"""breadth_growth_probe.py — can training be made to BUILD state breadth,
or does it only ever bake what pretraining handed over?

H26's central criterion after the 2026-09-13 reframing:

    If no training procedure that can be built raises the state's
    live-direction count above the pretrained baseline, then "builds the
    geometry" has no operational content and the distinction is decoration.

The baseline this is aimed at is real, not hypothetical: G1i base carries
13-16 live directions of 64 at the WKV state, and the Phase-1 LoRA checkpoint
that *works* carries the same 13-16 with near-identical per-head distributions
(`results/rank_recheck*/jlens.json`). Ordinary fine-tuning does not move that
number. The question here is whether anything does.

Arms, same task/seed/protocol as the rest of the H25/H26 toy family:

  adam                  the baseline that works
  muon                  the other pure optimizer
  {opt}_breadth_L<w>    plus an explicit breadth term at weight w
  {opt}_l2state_L<w>    plus a rank-BLIND shrink term      -- control
  {opt}_frob_L<w>       plus a rank-BLIND growth term      -- control

The breadth term maximises the entropy of the state's normalised singular
values — i.e. it directly optimises `effective_rank_entropy`, the same
quantity the probe reports. **This is deliberate Goodhart and is named as
such**: of course the metric moves when you optimise it. The informative part
is the pair of things it does NOT trivially determine — whether task quality
survives, and whether the mechanism signature (the a_gate ablation) changes.
H26's second refutation criterion is exactly this: breadth gained with quality
flat or falling is metric-chasing, not a result.

**The controls are load-bearing, not decoration** (added 2026-09-14, after the
first OOD-corrected full run). That run showed plain Adam failing to converge
on 2 of 3 seeds while every Adam+breadth arm converged and posted the best
held-out score of the whole table. That is a rescue, and H26 pre-registered
exactly this confound: "any auxiliary term of this magnitude stabilises
training". So the controls apply a term through the SAME differentiable path
(the final state) with the same character but no rank content — one that
shrinks total state energy, one that grows it, neither expressing any
preference about how that energy is distributed across directions. They are
swept over several weights rather than matched at one "equivalent" value, for
the same reason optimizer_switch_probe.py sweeps LR multipliers: picking a
single equivalence would be assuming the answer. If ANY rank-blind weight
rescues Adam, the rescue is not about breadth.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController
from experiments.A0_state_probe.muon_vs_adam_toy import (
    SingleDeviceMuon,
    _split_muon_adam_params,
)
from experiments._common.convergence import CONVERGED_ID_R2 as _CONVERGED_ID_R2
from experiments._common.results import save_result
from experiments._common.runtime import limit_threads, progress

# Shared with every other toy probe — see experiments/_common/convergence.py,
# which records why (this probe's own plain-Adam mean of 6.59 live directions
# was two dead seeds at exactly head_size/2 plus one real one at 3.76).
CONVERGED_ID_R2 = _CONVERGED_ID_R2


def _entropy_rank(state: torch.Tensor) -> torch.Tensor:
    """exp(H(p)) over normalised singular values, averaged over the batch —
    differentiable, so it can be used as a loss term. Same definition as the
    `effective_rank_entropy` the jlens probe reports, so the training signal
    and the measurement are literally the same quantity."""
    sv = torch.linalg.svdvals(state.float())          # [B, h]
    p = sv / (sv.sum(dim=-1, keepdim=True) + 1e-9)
    ent = -(p * (p + 1e-9).log()).sum(dim=-1)
    return ent.exp().mean()


def _aux_term(kind: str, state: torch.Tensor) -> torch.Tensor:
    """The auxiliary loss term, returned with the sign already applied so the
    caller just adds `w * term`.

    `breadth` is the quantity under test. The other two are rank-blind by
    construction: both are functions of ‖state‖_F alone, which is invariant to
    how the spectrum is distributed — rotate the singular values among each
    other and neither term changes, while `breadth` changes maximally."""
    if kind == "breadth":
        return -_entropy_rank(state)
    if kind == "l2state":                    # shrink total energy
        return state.float().pow(2).mean()
    if kind == "frob":                       # grow total energy
        return -state.float().pow(2).mean().sqrt()
    raise ValueError(f"unknown aux term {kind!r}")


@torch.no_grad()
def _measure(model, target_fn, n_recurrence_steps: int, n: int = 1000) -> dict:
    a = torch.empty(n).uniform_(-3.0, 3.0)
    b = torch.empty(n).uniform_(-3.0, 3.0)
    y = target_fn(a, b)
    y_hat, final_state, trace = model(a, b)
    id_r2 = 1.0 - F.mse_loss(y_hat, y).item() / y.var().item()

    # Held-out range. Added after the first full run reported "quality did not
    # fall" off id_r2 alone — which sits at 0.998-0.9999 in every arm, i.e. it
    # is saturated and cannot show a cost. OOD is where this task family has
    # actual headroom (0.75-0.78 in the 10-seed baseline), so it is the metric
    # that can answer whether breadth was bought at a price.
    sa = (torch.randint(0, 2, (n,)) * 2 - 1).float()
    sb = (torch.randint(0, 2, (n,)) * 2 - 1).float()
    ao = sa * torch.empty(n).uniform_(3.0, 5.0)
    bo = sb * torch.empty(n).uniform_(3.0, 5.0)
    yo = target_fn(ao, bo)
    ood_r2 = 1.0 - F.mse_loss(model(ao, bo)[0], yo).item() / yo.var().item()

    yg, _, _ = model(a, b, force_a_gate=0.0)
    abl = 1.0 - F.mse_loss(yg, y).item() / y.var().item()

    # Breadth of the state, measured the way the real probe measures it:
    # count of singular values above 1% of the largest, plus entropy rank.
    sv = torch.linalg.svdvals(final_state.float())     # [B, h]
    top = sv[:, :1]
    live = (sv > 0.01 * top).sum(dim=-1).float().mean().item()
    return {
        "id_r2": id_r2,
        "ood_r2": ood_r2,
        "ablation_a_gate_0_r2": abl,
        "live_directions_final": live,
        "entropy_rank_final": float(_entropy_rank(final_state)),
        "head_size": final_state.shape[-1],
        "converged": bool(id_r2 > CONVERGED_ID_R2),
        # Reported because the first smoke run of this file saturated it and
        # the verdict misread saturation as "no growth": the state gains at
        # most one rank-1 write per recurrence step, so the reachable rank is
        # min(n_steps, head_size), NOT head_size. Any arm sitting exactly on
        # this number is at the ceiling and the comparison is uninformative.
        "rank_ceiling": min(n_recurrence_steps, final_state.shape[-1]),
    }


def run_arm(arm: str, seed: int, *, n_steps: int, head_size: int,
            n_train_steps: int, batch_size: int, muon_lr: float,
            adam_lr: float, aux_kind: str, aux_w: float) -> dict:
    torch.manual_seed(seed)
    model = FrozenFinalReadoutController(head_size, n_steps)
    target_fn = lambda x, y: x * y  # noqa: E731
    muon_params, adam_params = _split_muon_adam_params(model)

    use_muon = arm.startswith("muon")
    opt_main = (SingleDeviceMuon(muon_params, lr=muon_lr) if use_muon
                else torch.optim.AdamW(muon_params, lr=adam_lr))
    opt_aux = torch.optim.AdamW(adam_params, lr=adam_lr)

    for _ in range(n_train_steps):
        a = torch.empty(batch_size).uniform_(-3.0, 3.0)
        b = torch.empty(batch_size).uniform_(-3.0, 3.0)
        y_hat, final_state, _ = model(a, b)
        loss = F.mse_loss(y_hat, target_fn(a, b))
        if aux_kind != "none" and aux_w > 0:
            loss = loss + aux_w * _aux_term(aux_kind, final_state)
        opt_main.zero_grad()
        opt_aux.zero_grad()
        loss.backward()
        opt_main.step()
        opt_aux.step()

    out = {"arm": arm, "seed": seed, "aux_kind": aux_kind, "breadth_weight": aux_w}
    out.update(_measure(model, target_fn, n_recurrence_steps=n_steps))
    return out


def _aggregate(runs: list[dict], arms: list[str]) -> dict:
    """Per-arm means. `converged_frac` is reported alongside, and the breadth
    metrics are ALSO given restricted to converged seeds, because breadth read
    off a model that never learned the task is not a measurement of anything."""
    summary = {}
    keys = ("id_r2", "ood_r2", "ablation_a_gate_0_r2",
            "live_directions_final", "entropy_rank_final")
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        ok = [r for r in rs if r["converged"]]
        line = {k: {"mean": sum(r[k] for r in rs) / len(rs),
                    "std": statistics.pstdev([r[k] for r in rs]) if len(rs) > 1 else 0.0}
                for k in keys}
        line["converged_frac"] = len(ok) / len(rs)
        line["n_seeds"] = len(rs)
        line["converged_only"] = (
            {k: sum(r[k] for r in ok) / len(ok) for k in keys} if ok else None
        )
        summary[arm] = line
    return summary


def _verdict(summary: dict, *, n_steps: int, head_size: int) -> tuple[str, dict]:
    """Read the table per optimizer, not globally.

    The first version of this took the single breadth arm with the most live
    directions and compared it against the best plain arm of EITHER optimizer.
    With every breadth arm pinned at the rank ceiling that selection is
    arbitrary — it happened to pick the worst-quality arm and reported
    "metric-chasing" for a table whose actual content was that the term
    rescues one optimizer and does nothing for the other. The comparison that
    carries meaning is each optimizer against itself."""
    ceiling = min(n_steps, head_size)
    per_opt: dict[str, dict] = {}
    for opt in ("adam", "muon"):
        if opt not in summary:
            continue
        plain = summary[opt]
        # Converged-only where it exists. `_aggregate` computes `converged_only`
        # and the first version of this verdict then read the all-runs mean
        # anyway — so plain Adam entered the comparison at ood +0.2470 (1 of 3
        # seeds converged; the other two never learned the task) instead of its
        # real +0.7400, and every delta against it was inflated by ~0.5. That is
        # the same defect experiments/_common/convergence.py was written for,
        # left live in the one code path whose output reaches RESULTS.md.
        co = plain.get("converged_only")
        base_live = (co or plain["live_directions_final"])["live_directions_final"] \
            if co else plain["live_directions_final"]["mean"]
        base_q = co["ood_r2"] if co else plain["ood_r2"]["mean"]
        base_conv = plain["converged_frac"]
        entry = {"plain_live": base_live, "plain_ood": base_q,
                 "plain_converged_frac": base_conv,
                 "plain_at_ceiling": base_live >= ceiling - 0.05, "terms": {}}
        for arm, line in summary.items():
            if not arm.startswith(f"{opt}_"):
                continue
            kind = arm.split("_")[1].split("L")[0].rstrip("_") or arm.split("_")[1]
            lco = line.get("converged_only")
            arm_live = lco["live_directions_final"] if lco else line["live_directions_final"]["mean"]
            arm_ood = lco["ood_r2"] if lco else line["ood_r2"]["mean"]
            entry["terms"][arm] = {
                "kind": kind,
                "live": arm_live,
                "ood": arm_ood,
                "d_ood": arm_ood - base_q,
                "converged_frac": line["converged_frac"],
                "d_converged": line["converged_frac"] - base_conv,
                "at_ceiling": arm_live >= ceiling - 0.05,
            }
        per_opt[opt] = entry

    def best(opt: str, kind_pred) -> tuple[str | None, dict | None]:
        cands = [(a, t) for a, t in per_opt.get(opt, {}).get("terms", {}).items()
                 if kind_pred(t["kind"])]
        if not cands:
            return None, None
        return max(cands, key=lambda kv: kv[1]["ood"])

    lines = [f"rank ceiling for this config = min(n_steps={n_steps}, "
             f"head={head_size}) = {ceiling}"]
    for opt, e in per_opt.items():
        if e["plain_at_ceiling"]:
            lines.append(f"  [!] plain {opt} is already AT the ceiling — this config "
                         f"cannot show growth for it, rerun with more recurrence steps")
        ba, bt = best(opt, lambda k: k == "breadth")
        ca, ct = best(opt, lambda k: k != "breadth")
        seg = (f"  {opt}: plain live={e['plain_live']:.2f} ood={e['plain_ood']:+.4f} "
               f"conv={e['plain_converged_frac']:.0%}")
        if bt:
            seg += (f" | best breadth {ba} live={bt['live']:.2f}"
                    + ("(at ceiling)" if bt["at_ceiling"] else "")
                    + f" ood={bt['ood']:+.4f} ({bt['d_ood']:+.4f}) "
                      f"conv={bt['converged_frac']:.0%}")
        if ct:
            seg += (f" | best rank-blind control {ca} ood={ct['ood']:+.4f} "
                    f"({ct['d_ood']:+.4f}) conv={ct['converged_frac']:.0%}")
        lines.append(seg)

    # The claim H26 actually cares about: the term does something for one
    # optimizer that it does not do for the other, AND a rank-blind term of
    # comparable magnitude does not reproduce it.
    calls = []
    for opt, e in per_opt.items():
        ba, bt = best(opt, lambda k: k == "breadth")
        ca, ct = best(opt, lambda k: k != "breadth")
        if bt is None:
            continue
        grew = bt["live"] > e["plain_live"] + 0.5
        helped = bt["d_ood"] > 0.02 or bt["d_converged"] > 0.0
        hurt = bt["d_ood"] < -0.02
        ctrl_reproduces = ct is not None and (
            ct["ood"] >= bt["ood"] - 0.02 or ct["d_converged"] >= bt["d_converged"] > 0.0
        )
        if helped and ct is None:
            calls.append(f"{opt}: breadth HELPS ({bt['d_ood']:+.4f} ood) "
                         f"but no rank-blind control was run — confound open")
        elif helped and ctrl_reproduces:
            calls.append(f"{opt}: breadth helps but a RANK-BLIND term does as well "
                         f"({ca}) — the gain is not about breadth")
        elif helped:
            calls.append(f"{opt}: breadth helps and no rank-blind term reproduces it "
                         f"— the gain is breadth-specific")
        elif hurt:
            calls.append(f"{opt}: breadth rose but held-out quality fell "
                         f"({bt['d_ood']:+.4f}) — metric-chasing, per H26's Goodhart "
                         f"criterion")
        elif grew:
            calls.append(f"{opt}: breadth rose {e['plain_live']:.2f}->{bt['live']:.2f} "
                         f"with quality flat ({bt['d_ood']:+.4f}) — buildable, "
                         f"but this optimizer gains nothing from it")
        else:
            calls.append(f"{opt}: breadth did not rise above the plain arm")
    lines.append("-> " + "; ".join(calls))
    return "\n".join(lines), per_opt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--breadth-weights", default="0.01,0.1")
    ap.add_argument("--control-weights", default="0.01,0.1",
                    help="weights for the rank-blind control terms; empty string "
                         "to skip them (leaves H26's confound criterion open).")
    ap.add_argument("--optimizers", default="adam,muon")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap torch intra-op threads (default 4, or "
                         "$NOESIS_PROBE_THREADS). Set this when running "
                         "several probes at once: torch otherwise sizes "
                         "its pool from the core count and concurrent "
                         "probes oversubscribe the machine.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n_threads = limit_threads(args.threads)
    progress(f"[{Path(__file__).stem}] torch threads = {n_threads}")

    opts = [o for o in args.optimizers.split(",") if o]
    wl = lambda s: [float(x) for x in s.split(",") if x.strip()]  # noqa: E731

    arms: list[tuple[str, str, float]] = [(o, "none", 0.0) for o in opts]
    for o in opts:
        for w in wl(args.breadth_weights):
            arms.append((f"{o}_breadth_L{w:g}", "breadth", w))
    for o in opts:
        for kind in ("l2state", "frob"):
            for w in wl(args.control_weights):
                arms.append((f"{o}_{kind}_L{w:g}", kind, w))

    runs = []
    for arm, kind, w in arms:
        for seed in range(args.seeds):
            r = run_arm(arm, seed, n_steps=args.steps, head_size=args.head_size,
                        n_train_steps=args.train_steps, batch_size=args.batch_size,
                        muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                        aux_kind=kind, aux_w=w)
            runs.append(r)
            progress(f"[{arm:22s} seed={seed}] id_r2={r['id_r2']:+.4f} "
                  f"ood={r['ood_r2']:+.4f} abl={r['ablation_a_gate_0_r2']:+.4f} "
                  f"live={r['live_directions_final']:.2f}/{r['rank_ceiling']} "
                  f"eRank={r['entropy_rank_final']:.2f}"
                  + ("" if r["converged"] else "   [DID NOT CONVERGE]"))

    print("\n=== summary over seeds ===")
    summary = _aggregate(runs, [a for a, _, _ in arms])
    for arm, line in summary.items():
        co = line["converged_only"]
        print(f"  {arm:22s} live={line['live_directions_final']['mean']:5.2f}"
              f"±{line['live_directions_final']['std']:.2f}  "
              f"eRank={line['entropy_rank_final']['mean']:5.2f}  "
              f"id_r2={line['id_r2']['mean']:+.4f}  "
              f"ood={line['ood_r2']['mean']:+.4f}±{line['ood_r2']['std']:.4f}  "
              f"abl={line['ablation_a_gate_0_r2']['mean']:+.4f}  "
              f"conv={line['converged_frac']:.0%}"
              + (f"  [conv-only ood={co['ood_r2']:+.4f} live={co['live_directions_final']:.2f}]"
                 if co and line["converged_frac"] < 1.0 else ""))

    verdict, per_opt = _verdict(summary, n_steps=args.steps, head_size=args.head_size)
    print(f"\n{verdict}")

    if args.out is not None:
        def _fmt(opt: str) -> str:
            e = per_opt.get(opt)
            if not e:
                return "n/a"
            br = [t for t in e["terms"].values() if t["kind"] == "breadth"]
            cn = [t for t in e["terms"].values() if t["kind"] != "breadth"]
            b = max(br, key=lambda t: t["ood"]) if br else None
            c = max(cn, key=lambda t: t["ood"]) if cn else None
            s = f"plain ood={e['plain_ood']:+.4f} conv={e['plain_converged_frac']:.0%}"
            if b:
                s += f"; +breadth ood={b['ood']:+.4f} conv={b['converged_frac']:.0%}"
            if c:
                s += f"; +rank-blind ood={c['ood']:+.4f} conv={c['converged_frac']:.0%}"
            return s

        save_result(
            args.out,
            {"runs": runs, "summary": summary, "verdict": verdict,
             "per_optimizer": per_opt,
             "config": vars(args) | {"out": str(args.out)}},
            experiment="breadth_growth_toy", hypothesis=["H26"],
            summary={
                "adam": _fmt("adam"),
                "muon": _fmt("muon"),
                "rank ceiling": str(min(args.steps, args.head_size)),
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
