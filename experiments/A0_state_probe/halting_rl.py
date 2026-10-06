#!/usr/bin/env python3
"""halting_rl.py — can "when to stop" be learned from outcome alone? (H25 toy)

`halting_chain.py` showed stopping is READ from state, but it was taught with
the oracle's stop round as a label. In the real harness no such label exists
(memory: project_noesis_freedom_thesis — the harness is driven by the model, and
`mean_M` stayed frozen at 2.0 because the stop decision was never a sampled token
inside log-pi). This is the toy of the real situation:

  * a hidden bit b; each round one noisy observation x_t = (2b-1)*MU + N(0,1)
    arrives and is never shown again (sequential hypothesis testing: the
    sufficient statistic is the running sum, which exists only in the state);
  * question: what is b?
  * at every round the policy either CONTINUES or STOPS-and-ANSWERS; both are
    sampled, so both are inside log-pi, while the gradient also flows through the
    latent state computation (the same split as the real GRPO loop);
  * reward = 1[answer correct] - LAMBDA * rounds_used. No stop label anywhere.

The cost of time is what makes stopping early worth anything; without it the
optimal policy is "always use every round" (flagged in the plan).

Reference policies, computed from the data, no model:
  fixed    the best single stop round for everybody, answer = sign of the sum
           (a counter-only policy; the floor the adaptive policy must beat)
  sprt     stop the first round |running sum| >= c, c grid-searched on separate
           data (the near-optimal adaptive policy; the ceiling)
Controls: a memoryless controller (state reset every round) trained the same way;
and a state swap at round 3 on the trained model (does the decision follow the
donor's state?).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.A0_state_probe.micro_wkv import micro_wkv_step  # noqa: E402

LAMBDA = 0.01
MU = 0.35
STOP_BIAS0 = -2.0   # init: ~12% stop per round; without it the policy
                    # collapses to 'stop at round 0' in <300 steps (measured)


class Policy(nn.Module):
    def __init__(self, head_size: int, n_rounds: int, hidden: int = 64,
                 memoryless: bool = False):
        super().__init__()
        self.head_size, self.T, self.memoryless = head_size, n_rounds, memoryless
        self.round_embed = nn.Embedding(n_rounds, 8)
        self.net = nn.Sequential(nn.Linear(1 + 8, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 5 * head_size))
        self.stop_head = nn.Linear(head_size, 1)
        self.answer_head = nn.Linear(head_size, 1)
        nn.init.constant_(self.stop_head.bias, STOP_BIAS0)

    def run(self, operands: torch.Tensor, intervene=None):
        """operands [B,T] -> (stop_logit [B,T], answer_logit [B,T])."""
        B = operands.shape[0]
        state = torch.zeros(B, self.head_size, self.head_size)
        stops, answers = [], []
        for t in range(self.T):
            if self.memoryless:
                state = torch.zeros_like(state)
            if intervene is not None:
                state = intervene(t, state)
            re = self.round_embed(torch.full((B,), t, dtype=torch.long))
            raw = self.net(torch.cat([operands[:, t:t + 1], re], -1))
            r, k, v, a_logit, w_raw = raw.split(self.head_size, -1)
            out, state = micro_wkv_step(state, r, k, v, -F.softplus(-w_raw) - 0.5,
                                        torch.sigmoid(a_logit))
            stops.append(self.stop_head(out).squeeze(-1))
            answers.append(self.answer_head(out).squeeze(-1))
        return torch.stack(stops, 1), torch.stack(answers, 1)


def make_batch(n: int, T: int):
    label = torch.randint(0, 2, (n,)).float()
    ops = (2 * label[:, None] - 1) * MU + torch.randn(n, T)
    return ops, label


def sprt_stop(ops: torch.Tensor, c: float) -> torch.Tensor:
    n, T = ops.shape
    hit = ops.cumsum(1).abs() >= c
    return torch.where(hit.any(1), hit.float().argmax(1), torch.full((n,), T - 1))


def rollout(policy: Policy, ops: torch.Tensor, label: torch.Tensor, sample: bool,
            intervene=None):
    n, T = ops.shape
    stop_logit, ans_logit = policy.run(ops, intervene)
    p_stop = torch.sigmoid(stop_logit)
    if sample:
        u = torch.bernoulli(p_stop.detach())
    else:
        u = (p_stop > 0.5).float()
    u[:, T - 1] = 1.0                                   # forced at the last round
    t_stop = u.argmax(1)                                # first stop
    idx = torch.arange(T)[None, :]
    # log-prob of the stop/continue decisions actually taken
    before = (idx < t_stop[:, None]).float()
    at = (idx == t_stop[:, None]).float()
    last = (t_stop == T - 1).float()
    logp = (before * F.logsigmoid(-stop_logit)).sum(1) \
        + (at * F.logsigmoid(stop_logit)).sum(1) * (1 - last)
    a_logit = ans_logit.gather(1, t_stop[:, None]).squeeze(1)
    a = torch.bernoulli(torch.sigmoid(a_logit).detach()) if sample else (a_logit > 0).float()
    logp = logp + a * F.logsigmoid(a_logit) + (1 - a) * F.logsigmoid(-a_logit)
    reward = (a == label).float() - LAMBDA * (t_stop + 1).float()
    return logp, reward, t_stop, (a == label).float()


def pretrain_supervised(policy: Policy, steps: int, batch: int, lr: float, c: float):
    """Stage 1 of the curriculum: teacher-forced stop labels from the SPRT oracle
    (stop head BCE up to the oracle's round, answer head BCE at that round). This
    is the SAME noisy task the RL run uses, so supervised-vs-reward is a fair
    comparison — the earlier 0.953 (halting_chain.py) was a different task."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    T = policy.T
    for step in range(steps):
        ops, label = make_batch(batch, T)
        t_or = sprt_stop(ops, c)
        stop_logit, ans_logit = policy.run(ops)
        idx = torch.arange(T)[None, :]
        upto = (idx <= t_or[:, None]).float()
        lab = (idx == t_or[:, None]).float()
        # the last round is forced, so it carries no stop label
        upto[:, T - 1] = 0
        bce = (F.binary_cross_entropy_with_logits(stop_logit, lab, reduction="none")
               * upto).sum() / upto.sum().clamp(min=1)
        a_logit = ans_logit.gather(1, t_or[:, None]).squeeze(1)
        loss = bce + F.binary_cross_entropy_with_logits(a_logit, label)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  [supervised] step {step} loss {loss.item():.4f}", flush=True)


def train(policy: Policy, steps: int, batch: int, lr: float, seed: int):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    base = 0.0
    for step in range(steps):
        ops, label = make_batch(batch, policy.T)
        logp, reward, t_stop, correct = rollout(policy, ops, label, sample=True)
        adv = reward - reward.mean()
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [{'memoryless' if policy.memoryless else 'state'}] step {step} "
                  f"reward {reward.mean():.3f} acc {correct.mean():.3f} "
                  f"mean_stop_round {t_stop.float().mean():.2f}", flush=True)


@torch.no_grad()
def evaluate(policy: Policy, n: int = 4000, seed: int = 777) -> dict:
    torch.manual_seed(seed)
    ops, label = make_batch(n, policy.T)
    _, reward, t_stop, correct = rollout(policy, ops, label, sample=False)
    return {"reward": reward.mean().item(), "accuracy": correct.mean().item(),
            "mean_stop_round": t_stop.float().mean().item()}


def _score(ops, label, t_stop):
    cs = ops.cumsum(1).gather(1, t_stop[:, None]).squeeze(1)
    correct = ((cs > 0).float() == label).float()
    return {"reward": (correct - LAMBDA * (t_stop + 1).float()).mean().item(),
            "accuracy": correct.mean().item(),
            "mean_stop_round": t_stop.float().mean().item()}


@torch.no_grad()
def references(T: int, n: int = 20000) -> dict:
    torch.manual_seed(5)
    ops, label = make_batch(n, T)                 # tuning data
    best_fixed = max(range(T), key=lambda t: _score(ops, label, torch.full((n,), t))["reward"])
    best_c = max([x * 0.25 for x in range(1, 40)],
                 key=lambda c: _score(ops, label, sprt_stop(ops, c))["reward"])
    torch.manual_seed(777)
    ops, label = make_batch(4000, T)              # held-out evaluation data
    return {"fixed": {"stop_round": best_fixed,
                      **_score(ops, label, torch.full((4000,), best_fixed))},
            "sprt": {"threshold": best_c, **_score(ops, label, sprt_stop(ops, best_c))}}


@torch.no_grad()
def swap_control(policy: Policy, c: float, at: int = 3, n: int = 4000, seed: int = 31) -> dict:
    """Swap states across examples before round `at`; the stop round of the
    swapped run should follow the donor's prefix, not the recipient's."""
    torch.manual_seed(seed)
    ops, label = make_batch(n, policy.T)
    perm = torch.randperm(n)
    swap = lambda t, s: s[perm] if t == at else s
    stop_logit, _ = policy.run(ops)
    p_base = (torch.sigmoid(stop_logit) > 0.5).float()
    p_base[:, -1] = 1; t_base = p_base.argmax(1)
    sl, _ = policy.run(ops, intervene=swap)
    p_sw = (torch.sigmoid(sl) > 0.5).float()
    p_sw[:, -1] = 1; t_sw = p_sw.argmax(1)
    mixed = torch.cat([ops[perm][:, :at], ops[:, at:]], 1)
    t_donor = sprt_stop(mixed, c)
    t_own = sprt_stop(ops, c)
    differs = t_donor != t_own
    return {"n_where_oracles_differ": int(differs.sum()),
            "swapped_stop_equals_donor_oracle": (t_sw == t_donor)[differs].float().mean().item(),
            "swapped_stop_equals_own_oracle": (t_sw == t_own)[differs].float().mean().item(),
            "mean_abs_stop_shift_vs_unswapped": (t_sw - t_base).abs().float().mean().item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain-steps", type=int, default=0,
                    help="Supervised stop-label stage before RL (curriculum). "
                         "Reports the policy after stage 1 and again after RL.")
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/A0_state_probe/results/halting_rl.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    res = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "lambda": LAMBDA, "references": references(args.rounds)}
    print("references:", json.dumps(res["references"]))
    for name, memless in (("state", False), ("memoryless", True)):
        torch.manual_seed(args.seed)
        p = Policy(args.head_size, args.rounds, memoryless=memless)
        res[name] = {}
        if args.pretrain_steps:
            pretrain_supervised(p, args.pretrain_steps, 256, 3e-3,
                                res["references"]["sprt"]["threshold"])
            res[name]["after_supervised"] = evaluate(p)
            print(name, "after supervised:", res[name]["after_supervised"], flush=True)
        train(p, args.steps, args.batch, args.lr, args.seed)
        res[name]["eval"] = evaluate(p)
        if not memless:
            res[name]["swap"] = swap_control(p, res["references"]["sprt"]["threshold"])
        print(name, json.dumps(res[name], indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
