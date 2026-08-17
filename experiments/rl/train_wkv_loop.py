#!/usr/bin/env python3
"""train_wkv_loop.py — WKV-loop RL training (replaces train_wordsearch.py).

Stack:
    loader.py        — LoadedModel (peft for GPU, blink for CPU smoke)
    wkv_loop.py      — generate_rollout (no <think> tokens)
    rewards.py       — compute_wkv_loop_rewards
    grpo.py          — compute_advantages + PPO-clip surrogate
    corpus.py        — CorpusScheduler (curriculum advance/drop)
    monitor.py       — TrainingMonitor (emergency stops)
    vm_watchdog.py   — WatchdogHook (24h Selectel VM deadline)
    probes.py        — run_inline_probes (stable_rank + effort frontier)

Usage (GPU, peft backend):
    python3 experiments/rl/train_wkv_loop.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \\
        --out experiments/rl/runs/wkv_loop_01 \\
        --feed-mode discrete --G 8 --M-max 16 --lr 1e-5 --steps 2000

CPU smoke (discrete only, no grad update):
    python3 experiments/rl/train_wkv_loop.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \\
        --out experiments/rl/runs/smoke_cpu \\
        --feed-mode discrete --G 2 --M-max 4 --steps 2 --no-update
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rl.corpus import load_corpus
from experiments.rl.loader import load_rwkv7, LoadedModel
from experiments.rl.wkv_loop import generate_rollout, WKVLoopRollout, _last_vec
from experiments.rl.rewards import compute_wkv_loop_rewards
from experiments.rl.grpo import compute_advantages
from experiments.rl.monitor import TrainingMonitor
from experiments.rl.vm_watchdog import VMWatchdog, WatchdogHook
from experiments.rl.probes import run_inline_probes
from experiments._common.results import save_result


CORPUS_PATH = ROOT / "training/corpus_open/matrix_tasks.jsonl"


# ------------------------------------------------------------------
# Log-prob recompute (WKV-aware)

def _recompute_wkv_log_probs(
    loaded: LoadedModel,
    rollout: WKVLoopRollout,
    feed_mode: str,
    mlp_delta: Optional[torch.nn.Module],
    alpha: float,
) -> torch.Tensor:
    """Recompute log π_θ(answer | prompt + WKV-loop) with gradients.

    Strategy: replay prefill + WKV loop (deterministic), then compute
    log-probs for answer tokens with gradients flowing through
    the peft model's parameters.

    Gradient flows through answer-token forward passes. Loop steps are
    replayed deterministically (same argmax choices); their WKV state
    influence propagates through the answer-phase forward passes.
    """
    state = loaded.new_state(batch=1)

    # Prefill
    if loaded.backend == "peft":
        inp = torch.tensor([rollout.prompt_ids], dtype=torch.long,
                           device=loaded.device)
    else:
        inp = rollout.prompt_ids
    logits, state = loaded.forward_stateful(inp, state)

    # WKV loop replay (deterministic, same choices as rollout)
    emb_w = loaded.embedding_weight if feed_mode != "discrete" else None
    for _ in range(rollout.M):
        v = _last_vec(logits)
        if feed_mode == "discrete":
            next_id = int(v.argmax().item())
            if loaded.backend == "peft":
                step_inp = torch.tensor([[next_id]], dtype=torch.long,
                                        device=loaded.device)
            else:
                step_inp = [next_id]
            logits, state = loaded.forward_stateful(step_inp, state)
        else:
            probs = F.softmax(v.float(), dim=-1)
            expected = (probs.unsqueeze(0) @ emb_w.float()).to(loaded.dtype)
            if feed_mode == "residual" and mlp_delta is not None:
                expected = expected + alpha * mlp_delta(expected)
            logits, state = loaded.forward_stateful_embeds(
                expected.unsqueeze(1), state)

    # Answer tokens: compute log-probs with grad
    log_probs: List[torch.Tensor] = []
    for tok_id in rollout.answer_ids:
        v = _last_vec(logits)
        lp = F.log_softmax(v.float(), dim=-1)
        log_probs.append(lp[tok_id])
        if loaded.backend == "peft":
            step_inp = torch.tensor([[tok_id]], dtype=torch.long,
                                    device=loaded.device)
        else:
            step_inp = [tok_id]
        logits, state = loaded.forward_stateful(step_inp, state)

    if not log_probs:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(log_probs)  # [T_answer], has grad


# ------------------------------------------------------------------
# GRPO loss over a batch of (rollout, reward) pairs

def wkv_grpo_loss(
    loaded: LoadedModel,
    batch_rollouts: List[List[WKVLoopRollout]],   # [n_prompts][G]
    batch_rewards: List[torch.Tensor],             # [n_prompts] each [G]
    feed_mode: str,
    mlp_delta: Optional[torch.nn.Module],
    alpha: float,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> torch.Tensor:
    """PPO-clip GRPO loss over batch."""
    total_loss = torch.tensor(0.0, device=loaded.device if loaded.backend == "peft" else "cpu",
                              requires_grad=True)
    n_tokens = 0

    for rollouts, rewards in zip(batch_rollouts, batch_rewards):
        advantages = compute_advantages(rewards)  # [G]

        for rollout, adv in zip(rollouts, advantages.tolist()):
            if not rollout.answer_ids:
                continue

            log_pi_theta = _recompute_wkv_log_probs(
                loaded, rollout, feed_mode, mlp_delta, alpha
            )  # [T_ans], grad

            # Old log-probs (stored at rollout time)
            if rollout.answer_log_probs:
                log_pi_old = torch.tensor(
                    rollout.answer_log_probs[:len(rollout.answer_ids)],
                    dtype=torch.float32,
                )
            else:
                # answer_log_probs should always be populated by generate_rollout
                # (wkv_loop.py) for a non-empty answer — an empty/None value here
                # means rollout generation likely dropped log-probs due to a bug,
                # not a deliberate on-policy mode. ratio collapses to 1 (REINFORCE-
                # equivalent: gradient flows only through log_pi_theta), which is a
                # safe fallback but should not happen silently.
                print(f"[train] WARNING: rollout missing answer_log_probs "
                      f"({len(rollout.answer_ids)} answer tokens) — falling back "
                      f"to on-policy ratio=1", file=sys.stderr)
                log_pi_old = log_pi_theta.detach()  # on-policy fallback

            ratio = torch.exp(log_pi_theta - log_pi_old)
            adv_t = torch.full_like(ratio, adv)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            surrogate = -torch.min(unclipped, clipped).mean()

            # KL penalty
            kl = (ratio - 1 - (log_pi_theta - log_pi_old)).mean()
            surrogate = surrogate + kl_coef * kl

            total_loss = total_loss + surrogate
            n_tokens += len(rollout.answer_ids)

    return total_loss / max(n_tokens, 1)


# ------------------------------------------------------------------
# Checkpoint

def _save_checkpoint(out_dir: Path, loaded: LoadedModel, step: int,
                     mlp_delta: Optional[torch.nn.Module] = None) -> None:
    ckpt = {"step": step}
    if loaded.backend == "peft":
        # Save only trainable params (LoRA A/B or full diff)
        trainable = {n: p for n, p in loaded.model.named_parameters()
                     if p.requires_grad}
        ckpt["model"] = {n: p.detach().cpu() for n, p in trainable.items()}
    if mlp_delta is not None:
        ckpt["mlp_delta"] = {n: p.detach().cpu()
                             for n, p in mlp_delta.named_parameters()}
    path = out_dir / f"ckpt_step{step:06d}.pt"
    torch.save(ckpt, path)
    print(f"[train] checkpoint → {path}")


# ------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feed-mode", default="discrete",
                    choices=["discrete", "expected", "residual"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--G", type=int, default=8, help="Rollouts per prompt")
    ap.add_argument("--batch", type=int, default=4, help="Prompts per update")
    ap.add_argument("--M-max", type=int, default=16)
    ap.add_argument("--max-answer", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-update", action="store_true",
                    help="Rollout + reward only, no gradient update (CPU smoke)")
    ap.add_argument("--vm-lifetime", type=float, default=24.0,
                    help="Selectel VM lifetime in hours (default 24)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    backend = "peft" if args.device != "cpu" else "blink"
    loaded = load_rwkv7(args.model, device=args.device, backend=backend)

    # MLP_delta for residual mode
    mlp_delta: Optional[torch.nn.Module] = None
    if args.feed_mode == "residual":
        D = loaded.embedding_weight.shape[1]
        from experiments.rl.sweep_alpha import _MLPDelta
        mlp_delta = _MLPDelta(D).to(args.device)

    # Optimiser (only for peft backend with grad)
    optimizer = None
    if not args.no_update and loaded.backend == "peft":
        params = [p for p in loaded.model.parameters() if p.requires_grad]
        if mlp_delta:
            params += list(mlp_delta.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # Corpus + curriculum
    sched = load_corpus(str(CORPUS_PATH), start_level=1, rng_seed=42)

    # Monitor + watchdog
    monitor = TrainingMonitor()
    wd = VMWatchdog(lifetime_hours=args.vm_lifetime)
    wd.print_status()

    def _checkpoint():
        _save_checkpoint(out_dir, loaded, global_step, mlp_delta)

    hook = WatchdogHook(wd, _checkpoint,
                        force_ckpt_hours=2.0, stop_hours=0.25,
                        check_interval_steps=50)

    global_step = 0
    print(f"[train] feed_mode={args.feed_mode} alpha={args.alpha} "
          f"G={args.G} M_max={args.M_max} device={args.device}")

    for step in range(args.steps):
        global_step = step + 1

        # Sample batch
        tasks = sched.sample_batch(args.batch)

        # Rollout: G rollouts per prompt
        batch_rollouts: List[List[WKVLoopRollout]] = []
        for task in tasks:
            group = [
                generate_rollout(
                    loaded, task["prompt"],
                    feed_mode=args.feed_mode,
                    M_max=args.M_max,
                    tau_commit=0.90,
                    eps_plateau=0.02,
                    max_answer_tokens=args.max_answer,
                    answer_temperature=0.7,
                    mlp_delta=mlp_delta,
                    alpha=args.alpha,
                    eos_id=0,
                )
                for _ in range(args.G)
            ]
            batch_rollouts.append(group)

        # Rewards
        batch_rewards: List[torch.Tensor] = []
        all_r_correct: List[torch.Tensor] = []
        for rollouts, task in zip(batch_rollouts, tasks):
            rewards, diag = compute_wkv_loop_rewards(rollouts, task["rubric"])
            batch_rewards.append(rewards)
            all_r_correct.append(diag["r_correct"])

        # Monitor
        all_rollouts_flat = [r for g in batch_rollouts for r in g]
        all_rewards_flat = torch.cat(batch_rewards)
        combined_diag = {"r_correct": torch.cat(all_r_correct)}
        stop, flags = monitor.step(all_rollouts_flat, all_rewards_flat, combined_diag)

        # Gradient update
        loss_val = float("nan")
        if not args.no_update and optimizer is not None:
            optimizer.zero_grad()
            loss = wkv_grpo_loss(
                loaded, batch_rollouts, batch_rewards,
                feed_mode=args.feed_mode,
                mlp_delta=mlp_delta,
                alpha=args.alpha,
            )
            loss.backward()
            if loaded.backend == "peft":
                torch.nn.utils.clip_grad_norm_(
                    [p for p in loaded.model.parameters() if p.requires_grad], 1.0
                )
            optimizer.step()
            loss_val = float(loss.item())

        # Curriculum update
        accuracy = float((all_rewards_flat > 0).float().mean().item())
        action, cur_level = sched.update_accuracy(sched.current_level, accuracy)

        # Log
        log_entry = {
            "step": global_step,
            "loss": loss_val,
            "accuracy": accuracy,
            "current_level": cur_level,
            "curriculum_action": action,
            "mean_M": sum(r.M for r in all_rollouts_flat) / max(len(all_rollouts_flat), 1),
            "flags": flags,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if global_step % 10 == 0 or flags:
            print(f"  step {global_step:5d}: loss={loss_val:.4f} acc={accuracy:.2%} "
                  f"level={cur_level} {('!!! ' + str(flags)) if flags else ''}")

        # Inline probes
        if global_step % args.probe_every == 0:
            probe_result = run_inline_probes(
                loaded, all_rollouts_flat, label=f"step{global_step}"
            )
            probe_path = out_dir / f"probes_step{global_step:06d}.json"
            save_result(
                probe_path, probe_result,
                experiment="rl_inline_probes", hypothesis=["H8", "H10"],
                model=args.model, script=__file__,
                summary={
                    "shortcut_score": f"{probe_result['shortcut_score']:.2f}",
                    "M_mean": f"{probe_result['M_mean']:.1f}",
                },
            )
            print(f"  [probe] sr_reasoning_L4={probe_result.get('sr_reasoning_L4', float('nan')):.3f}  "
                  f"shortcut={probe_result['shortcut_score']:.2f}  "
                  f"M_mean={probe_result['M_mean']:.1f}")

        # Checkpoint
        if global_step % args.ckpt_every == 0:
            _checkpoint()

        # VM watchdog
        if hook.tick():
            print("[train] VM deadline — stopping.")
            break

        # Emergency stop
        if stop:
            print(f"[train] Emergency stop: {flags}")
            _checkpoint()
            break

    _checkpoint()
    print(f"\n[train] done. {global_step} steps → {out_dir}")


if __name__ == "__main__":
    main()
