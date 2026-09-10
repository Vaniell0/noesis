"""Real generation + exact-match accuracy eval for ThinkChain checkpoints.

Fills the gap `docs/phase15-gap-matrix.md` flags: `distill_step`'s
`answer_ce` is teacher-forced loss (how well the model predicts the REAL
answer tokens), not free-generation accuracy -- a model can have a
low training loss while still generating something different when
actually decoding on its own, no teacher available. This script does
the latter: real free-running generation after the (optional) ThinkChain
phases, exact-string-match against the held-out example's own answer.

No teacher chunk_lens available at eval time (that's teacher-derived,
training-only) -- each phase always runs its full `--phase-repeat-ticks`
budget unless `--dynamic-phase-stop` is set, same flag/semantics as
`train_think_distill.py`.

Usage (base model, no ThinkChain -- M=0 direct-answer baseline):
    python experiments/rl/eval_thinkchain.py --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \
        --val training/tokenised/g1i_warmup_v3_eos_val.pt --limit 40 --out results_base.json

Usage (step500, Phase 1, M=1):
    python experiments/rl/eval_thinkchain.py --model models/rwkv7-g1i-2.9b-zlk_phase1_v3-step500-merged.pth \
        --think-marker --chain-phases 1 --lora-r 0 \
        --warm-start-marker experiments/rl/checkpoints/g1i_zlk_phase1_v3_step500 \
        --val training/tokenised/g1i_warmup_v3_eos_val.pt --limit 40 --out results_step500.json

Usage (phase15_muon, Phase 1.5, M=2, trained weights + marker both from --resume):
    python experiments/rl/eval_thinkchain.py --model models/rwkv7-g1i-2.9b-zlk_phase1_v3-step500-merged.pth \
        --think-marker --chain-phases 2 --lora-r 0 --dynamic-phase-stop \
        --resume experiments/rl/runs/phase15_muon/ckpt_step000050 \
        --val training/tokenised/g1i_warmup_v3_eos_val.pt --limit 40 --out results_muon_phase15.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import load_checkpoint
from experiments.rl.train_think_distill import load_examples, _infer_category, warm_start_marker
from experiments.rl.wkv_loop import _last_vec


def _run_phases(loaded, prompt_ids, think_marker, chain_phases, phase_repeat_ticks,
                 dynamic_phase_stop, eps_plateau, layers):
    device = loaded.device
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    state = loaded.new_state(batch=1)
    logits, state = loaded.forward_stateful(prompt, state)
    if think_marker is None:
        return logits, state
    marker = think_marker.step(0).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
    logits, state = loaded.forward_stateful_embeds(marker, state)
    for i in range(chain_phases):
        phase_marker = think_marker.step(i + 1).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
        prev_wkv = None
        for _ in range(phase_repeat_ticks):
            logits, state = loaded.forward_stateful_embeds(phase_marker, state)
            if dynamic_phase_stop:
                cur_wkv = {L: state.wkv[L] for L in layers}
                if prev_wkv is not None:
                    delta = sum(torch.linalg.vector_norm(
                        (cur_wkv[L].float() - prev_wkv[L].float()).flatten()).item() for L in layers)
                    ref = sum(torch.linalg.vector_norm(cur_wkv[L].float().flatten()).item() for L in layers)
                    if delta < eps_plateau * max(ref, 1e-6):
                        break
                prev_wkv = cur_wkv
    return logits, state


@torch.no_grad()
def generate_answer(loaded, logits, state, max_answer_tokens: int, eos_id: int = 0) -> list[int]:
    """Greedy decode (deterministic -- eval reproducibility over sampling
    diversity) from wherever the phase loop left off.

    Uses `_last_vec` (wkv_loop.py) to handle both backends' logits shape:
    peft returns [B,T,V], blink returns [V] (last token only, native)."""
    device = loaded.device
    out: list[int] = []
    for _ in range(max_answer_tokens):
        next_id = int(_last_vec(logits).argmax().item())
        if next_id == eos_id:
            break
        out.append(next_id)
        if loaded.backend == "peft":
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        else:
            x = torch.tensor([next_id], dtype=torch.long, device=device)
        logits, state = loaded.forward_stateful(x, state)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=40,
                     help="Eval a subset, not the full 1171-example val set -- "
                          "this is a comparison signal, not a leaderboard run.")
    ap.add_argument("--think-marker", action="store_true")
    ap.add_argument("--chain-phases", type=int, default=1, help="M for this checkpoint.")
    ap.add_argument("--phase-repeat-ticks", type=int, default=25,
                     help="Ceiling per phase when no teacher chunk_lens exists "
                          "(eval/inference) -- matches train_think_distill.py's "
                          "documented 15-25 real range; --dynamic-phase-stop "
                          "should make this a ceiling, not the typical case.")
    ap.add_argument("--dynamic-phase-stop", action="store_true")
    ap.add_argument("--eps-plateau", type=float, default=0.05)
    ap.add_argument("--max-answer-tokens", type=int, default=32)
    ap.add_argument("--work-layers", default="12,16,20")
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--warm-start-marker", type=Path, default=None,
                     help="Load only the trained ThinkChain marker (Phase 1 case).")
    ap.add_argument("--resume", type=Path, default=None,
                     help="Load full trained weights + marker (Phase 1.5 case, "
                          "a checkpoint written under the SAME --lora-r as here).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="peft", choices=["peft", "blink"],
                     help="'blink' is a much lighter inference-only path (no "
                          "peft/LoRA wrapper) -- use it for the no-marker "
                          "(base model, chain_phases=0) case on RAM-tight "
                          "boxes. --think-marker requires 'peft' (blink has "
                          "no forward_stateful_embeds).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.think_marker and args.backend != "peft":
        raise ValueError("--think-marker requires --backend peft "
                          "(forward_stateful_embeds is peft-only)")

    layers = tuple(int(x) for x in args.work_layers.split(","))
    loaded = load_rwkv7(args.model, device=args.device, backend=args.backend,
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    think_marker = None
    if args.think_marker:
        from experiments.rl.train_think_distill import ThinkChain
        think_marker = ThinkChain(loaded.n_embd, args.chain_phases).to(args.device)
        if args.warm_start_marker is not None:
            warm_start_marker(args.warm_start_marker, think_marker)

    if args.resume is not None:
        step = load_checkpoint(args.resume, loaded, mlp_delta=think_marker)
        print(f"[eval] resumed weights + marker from {args.resume} at step {step}")

    examples = load_examples(args.val)
    if args.limit:
        examples = examples[:args.limit]
    tok = loaded.tokenizer

    per_cat = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
    records = []
    for ex in examples:
        logits, state = _run_phases(
            loaded, ex["prompt_ids"], think_marker, args.chain_phases,
            args.phase_repeat_ticks, args.dynamic_phase_stop, args.eps_plateau, layers)
        pred_ids = generate_answer(loaded, logits, state, args.max_answer_tokens)
        pred_text = tok.decode(pred_ids).strip()
        # answer_ids ends with EOS (id 0); generate_answer() never includes
        # EOS in its returned list (stops AT it, doesn't append it) -- strip
        # it here too so both sides decode the same way. Found 2026-09-10:
        # decoding a sequence ending in id 0 produced garbage ('�'), not
        # just an extra character, so this isn't cosmetic.
        gold_ids = ex["answer_ids"][:-1] if ex["answer_ids"][-1:] == [0] else ex["answer_ids"]
        gold_text = tok.decode(gold_ids).strip()
        cat = _infer_category(tok.decode(ex["prompt_ids"]))
        is_correct = pred_text == gold_text
        per_cat[cat][0] += int(is_correct)
        per_cat[cat][1] += 1
        records.append({"category": cat, "pred": pred_text, "gold": gold_text, "correct": is_correct})

    total_correct = sum(c for c, _ in per_cat.values())
    total = sum(t for _, t in per_cat.values())
    summary = {
        "model": args.model, "resume": str(args.resume) if args.resume else None,
        "warm_start_marker": str(args.warm_start_marker) if args.warm_start_marker else None,
        "chain_phases": args.chain_phases if args.think_marker else 0,
        "n": total, "accuracy": total_correct / total if total else 0.0,
        "per_category": {c: {"correct": v[0], "total": v[1], "acc": v[0] / v[1]} for c, v in per_cat.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
