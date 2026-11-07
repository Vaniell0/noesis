#!/usr/bin/env python3
"""train_latent_s1.py — stage S1 (HZLK §6): the answer through ONE tick of the model's own latent.

The untrained loop erases the answer: feed the last block's output back once (Coconut-style,
`forward_stateful_embeds(..., return_hidden=True)`) and G1i drops from 0.85 to 0.50, G1k from 0.95
to 0.50 (latent_noise, 2026-10-22). S1 teaches the tick not to lose what the answer needs.

    S1a  self-distillation: items the student answers right at M=0 (m0_pass right_ids); target =
         the student's own M=0 answer distribution, computed once before any update.
    S1b  teacher: items the teacher (G1k) answers right at M=0 and the student does not; target =
         the teacher's answer distribution from a cache built beforehand (--build-cache), so the
         teacher is never loaded together with the student. Same tokenizer, so KL is on logits.

Sequence per item (silent route, as m0_pass):
    "User: {q}\\n\\nAssistant: <think>\\n</think>\\n"  -> [one latent tick] -> answer tokens (teacher-forced)
The target at M=0 is read at the same answer positions without the tick.

Loss (--loss): ce = CE on the gold answer tokens; kl = KL(target_T || student_T) * T^2 on every
answer position (Hinton; T>1 keeps a confident target's alternatives visible - owner 2026-10-22:
"a negative answer shown, the result in a larger space"). The pilot compares ce, kl T=1, kl T=2 by
the gates, not by loss value: accuracy after the tick vs M=0, and the readout-lens pair (ablation-
read directions + separable facts; P1 got fewer directions AND fewer separable facts = sharpening).

No term on the geometry of the latent (HZLK claim 5). A decorrelation term is added only if the
pre-written collapse criterion fires (plan B.3a); this file does not have it yet on purpose.

    python experiments/rl/train_latent_s1.py --build-cache --model TEACHER --items F --right-from T.json --cache-out C.pt
    python experiments/rl/train_latent_s1.py --model STUDENT --items F --right-from S.json --target self --out R.json
    python experiments/rl/train_latent_s1.py --model STUDENT --items F --right-from T.json --exclude-right-from S.json \\
        --target cache --cache C.pt --out R.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, MuonHybrid, load_weights_into  # noqa: E402
from experiments._common.runtime import limit_threads, progress  # noqa: E402
from experiments._common.heartbeat import write_heartbeat  # noqa: E402


def load_items(files, right_from, exclude_from) -> list:
    rows = {}
    for f in files:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                rows[r["id"]] = r
    keep = set(json.load(open(right_from))["right_ids"]) if right_from else set(rows)  # no file: every row
    if exclude_from:
        keep -= set(json.load(open(exclude_from))["right_ids"])
    return [rows[i] for i in sorted(keep) if i in rows]


def answer_ids(tok, r) -> list:
    """The model's OWN answer that m0_pass judged right (set in r["_ans"]), not the task's gold string:
    the rubric accepts a phrasing ("The answer is 60.") that the gold "60" would not match token by token.
    Ends with the blank line m0_pass stops at, so the end of the answer is learned too."""
    return r["_ans"]


@torch.no_grad()
def own_answer_ids(loaded, tok, prompt: str, max_new: int = 48, ctx_ids: list | None = None) -> list:
    """The greedy M=0 answer as the model's own token ids, up to and including the token that ends it
    (blank line or EOS). Same route and stop rule as m0_pass, but never decoded to text and re-encoded:
    the round trip changes the token split and drops the leading space/newline the model really emitted.
    ctx_ids: the full context up to the answer as ids (the teacher's prompt + its own CoT), instead of the prompt."""
    ids = ctx_ids or tok.encode(loaded.silent_prompt(prompt))
    logits, st = loaded.forward_stateful(torch.tensor([ids]), loaded.new_state(batch=1))
    out = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].float().argmax())
        out.append(nxt)
        if nxt in loaded.eos_ids or "\n\n" in tok.decode(out[-4:]) or "User:" in tok.decode(out[-4:]):
            break
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
    return out


def attach_own_answers(loaded, tok, items) -> None:
    for i, r in enumerate(items):
        r["_ans"] = own_answer_ids(loaded, tok, r.get("_teacher_prompt") or r["prompt"], ctx_ids=r.get("_teacher_ctx"))
        if (i + 1) % 10 == 0:
            progress(f"[s1] own answers {i + 1}/{len(items)}")


def seq_logits(loaded, tok, r, ticks: int, prompt: str | None = None, ctx_ids: list | None = None, ans: list | None = None):
    """Logits [L, V] at the answer positions, with `ticks` latent ticks between prompt and answer.
    `prompt` overrides r["prompt"] (the teacher's hinted prompt; the student always reads the plain one);
    `ctx_ids` replaces the whole context (the teacher's prompt + its own CoT). With ticks, the loader's tick_frame
    (Qwen: NOESIS_HF_FRAME) sets the text before the ticks and after them - the same route as m0_pass."""
    E = loaded.embedding_weight
    q = prompt or r["prompt"]
    head, close = loaded.tick_frame(q) if ticks and not ctx_ids and hasattr(loaded, "tick_frame") else (None, "")
    p = ctx_ids or tok.encode(head or loaded.silent_prompt(q))
    a = ans or answer_ids(tok, r)
    lg, st, h = loaded.forward_stateful_embeds(E[torch.tensor([p])], loaded.new_state(batch=1), return_hidden=True)
    lg, h = lg[:, -1:], h[:, -1:]
    for _ in range(ticks):
        lg, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
    if close:
        lg, st = loaded.forward_stateful_embeds(E[torch.tensor([tok.encode(close)])], st)
        lg = lg[:, -1:]
    rest, _ = loaded.forward_stateful_embeds(E[torch.tensor([a[:-1]])], st) if len(a) > 1 else (lg[:, :0], st)
    return torch.cat([lg, rest], dim=1)[0].float(), torch.tensor(a, device=lg.device)


@torch.no_grad()
def evaluate(loaded, tok, items, ticks: int = 1) -> dict:
    """Teacher-forced exact match of the whole answer, at M=0 and through the M ticks trained (key acc_M<ticks>)."""
    out = {}
    for M in (0, ticks):
        ok = 0
        for r in items:
            lg, a = seq_logits(loaded, tok, r, M)
            ok += bool((lg.argmax(-1) == a).all())
        out[f"acc_M{M}"] = round(ok / max(1, len(items)), 3)
    return out


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load(args):
    from experiments.rl.hf_loader import is_hf_dir, load_hf
    if is_hf_dir(args.model):                                   # a HF model directory (Qwen3.5, plan B.3c)
        return load_hf(args.model, args.device, DTYPES[args.dtype])
    return load_rwkv7(args.model, device=args.device, dtype=DTYPES[args.dtype], backend="peft", lora_r=0, ctx_len=2048)


@torch.no_grad()
def own_cot_ids(loaded, tok, prompt: str, max_cot: int) -> tuple:
    """(prompt ids, CoT ids, closed): the thinking turn and the model's own greedy CoT up to and including </think>."""
    ids = tok.encode(loaded.think_prompt(prompt))
    logits, st = loaded.forward_stateful(torch.tensor([ids]), loaded.new_state(batch=1))
    cot = []
    for _ in range(max_cot):
        nxt = int(logits[0, -1].float().argmax())
        if nxt in loaded.eos_ids:
            break
        cot.append(nxt)
        if "</think>" in tok.decode(cot[-6:]):
            return ids, cot, True
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
    return ids, cot, False


def cot_ctx(tok, ids: list, cot: list, k: int | None = None) -> list:
    """Context up to the answer after the first k CoT tokens: the CoT closed by force (as cot_vs_routes_probe's
    checkpoints do), then the blank line the silent route also ends with - teacher and student both answer right
    after "</think>\n\n". k None: the whole CoT, which already ends with </think>."""
    if k is None:
        return ids + cot + tok.encode("\n\n")
    return ids + cot[:k] + tok.encode("\n</think>\n\n")


def first_right_k(loaded, tok, r, ids: list, cot: list, every: int):
    """The shortest checked CoT prefix after which the answer is already right (the rest is water), or None."""
    from training.scripts.gen_ciphers import check_answer
    for k in range(0, len(cot), every):
        if check_answer(r, tok.decode(own_answer_ids(loaded, tok, r["prompt"], ctx_ids=cot_ctx(tok, ids, cot, k))))[0]:
            return k
    return None


def build_cache(args) -> int:
    loaded = load(args)
    tok = loaded.tokenizer
    items = load_items(args.items, args.right_from, args.exclude_right_from)
    if args.limit:
        items = items[:args.limit]
    if args.teacher_hint:
        from training.scripts.gen_ciphers import with_hint
        for r in items:
            r["_teacher_prompt"] = with_hint(r, args.teacher_hint)
    cots, dropped = {}, {"unclosed": 0, "wrong": 0}
    with torch.no_grad():
        if args.teacher_cot:                                    # B.3b: the teacher is the model after its own CoT
            from training.scripts.gen_ciphers import check_answer
            kept = []
            for i, r in enumerate(items):
                ids, cot, closed = own_cot_ids(loaded, tok, r["prompt"], args.teacher_cot)
                if not closed and not args.cot_check:
                    dropped["unclosed"] += 1
                    continue
                # a CoT that found the answer and then kept going ("Wait, ..." to the budget) is cut where the answer
                # first reads right - its rest is water, so not closing is no reason to drop it
                k = first_right_k(loaded, tok, r, ids, cot, args.cot_check) if args.cot_check else None
                if args.cot_check and k is None:
                    dropped["wrong" if closed else "unclosed"] += 1
                    continue
                r["_teacher_ctx"] = cot_ctx(tok, ids, cot, k)       # cut at the first right checkpoint: no water
                r["_ans"] = own_answer_ids(loaded, tok, r["prompt"], ctx_ids=r["_teacher_ctx"])
                if not check_answer(r, tok.decode(r["_ans"]))[0]:
                    dropped["wrong"] += 1
                    continue
                cots[r["id"]] = {"cot": tok.decode(cot), "n_tokens": len(cot), "closed": closed, "first_right": k}
                kept.append(r)
                progress(f"[s1 cot] {i + 1}/{len(items)} kept {len(kept)} dropped {dropped}")
            items = kept
        else:
            attach_own_answers(loaded, tok, items)
        if not args.keep_raw_teacher:                           # read the teacher's answers before training on them
            from training.scripts.gen_ciphers import check_answer
            clean = []
            for r in items:
                t = tok.decode(r["_ans"])
                if re.search(r"</?(answer|think)>", t, re.I):
                    dropped["tag"] = dropped.get("tag", 0) + 1
                elif "task" in r and not check_answer(r, t)[0]:
                    dropped["wrong"] += 1
                else:
                    clean.append(r)
            items = clean
        cache = {}
        for i, r in enumerate(items):
            lg, _ = seq_logits(loaded, tok, r, 0, r.get("_teacher_prompt"), ctx_ids=r.get("_teacher_ctx"))
            cache[r["id"]] = lg.half().cpu()
            progress(f"[s1 cache] {i + 1}/{len(items)}")
    torch.save({"model": args.model, "teacher_hint": args.teacher_hint, "teacher_cot": args.teacher_cot,
                "logits": cache, "answers": {r["id"]: r["_ans"] for r in items}, "cots": cots, "dropped": dropped,
                "ctx": {r["id"]: r["_teacher_ctx"] for r in items if r.get("_teacher_ctx")}},
               args.cache_out)
    print(f"[s1 cache] {len(cache)} items -> {args.cache_out} (dropped {dropped})")
    return 0


@torch.no_grad()
def student_sample(loaded, tok, r, ticks: int, temp: float, max_new: int) -> list:
    """The student's own answer, sampled at `temp` through the same route seq_logits teacher-forces (plain prompt,
    tick_frame, M ticks), up to and including the token that ends it (EOS or a blank line) - on-policy data."""
    E = loaded.embedding_weight
    head, close = loaded.tick_frame(r["prompt"]) if ticks and hasattr(loaded, "tick_frame") else (None, "")
    p = tok.encode(head or loaded.silent_prompt(r["prompt"]))
    lg, st, h = loaded.forward_stateful_embeds(E[torch.tensor([p])], loaded.new_state(batch=1), return_hidden=True)
    h = h[:, -1:]
    for _ in range(ticks):
        lg, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
    if close:
        lg, st = loaded.forward_stateful_embeds(E[torch.tensor([tok.encode(close)])], st)
    out = []
    for _ in range(max_new):
        q = torch.softmax(lg[0, -1].float() / temp, dim=-1)
        nxt = int(torch.multinomial(q, 1))
        out.append(nxt)
        if nxt in loaded.eos_ids or "\n\n" in tok.decode(out[-4:]):
            break
        lg, st = loaded.forward_stateful_embeds(E[torch.tensor([[nxt]])], st)
    return out


@torch.no_grad()
def teacher_logits(loaded, ctx: list, a: list) -> torch.Tensor:
    """Teacher logits [len(a), V] at the student's answer tokens, the teacher reading `ctx` (its own CoT cut at the
    first right checkpoint, or a hinted prompt, or the plain prompt) - the same model, no gradient."""
    lg, _ = loaded.forward_stateful(torch.tensor([ctx + a[:-1]]), loaded.new_state(batch=1))
    return lg[0, len(ctx) - 1:].float()


def reverse_kl(student, teacher) -> torch.Tensor:
    """KL(student || teacher) per answer token over the whole vocabulary, mean over tokens (on-policy distillation:
    the student is pulled toward what the teacher would say at the tokens the student itself wrote)."""
    ls, lt = F.log_softmax(student.float(), -1), F.log_softmax(teacher, -1)
    return (ls.exp() * (ls - lt)).sum(-1).mean()


def kl_loss(student, target, T: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(student / T, -1), F.log_softmax(target.float() / T, -1),
                    log_target=True, reduction="batchmean") * T * T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=Path, action="append", required=True)
    ap.add_argument("--right-from", type=Path, help="m0_pass result whose right_ids are trained (none: every row; "
                                                     "the --teacher-cot cache then keeps what its CoT gets right)")
    ap.add_argument("--exclude-right-from", type=Path, help="m0_pass result of the student (S1b: teacher-only items)")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--cache-out", type=Path)
    ap.add_argument("--target", choices=("self", "cache", "gold"), default="self",
                    help="gold: the row's answer string + end of turn, CE only - for a skill the model lacks at M=0 "
                         "(Qwen 0.8B decodes 0 of 60 short ciphers, 2026-10-29), so it has no own answer to distil")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--loss", choices=("ce", "kl"), default="kl")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--teacher-hint", type=int, default=0,
                    help="privileged teacher: the same model at M=0 reads hint level k (gen_ciphers.hint_ladder); the "
                         "student reads the plain prompt through the ticks. Pair with --right-from = the hinted m0_pass "
                         "and --exclude-right-from = the plain M=0 pass, so every item needs the ticks to work")
    ap.add_argument("--teacher-cot", type=int, default=0,
                    help="B.3b, with --build-cache: the teacher is the same model after its OWN CoT (thinking turn, at "
                         "most this many CoT tokens); items whose CoT does not close or whose answer is wrong are dropped. "
                         "Pair with --exclude-right-from = the plain M=0 pass, so every item needs the CoT")
    ap.add_argument("--keep-raw-teacher", action="store_true",
                    help="keep teacher answers that carry <answer>/<think> tags or a wrong content (2026-10-28: 77 of 211 "
                         "hint-teacher answers began with <answer>, and the student learned to loop on the tag)")
    ap.add_argument("--cot-check", type=int, default=0,
                    help="with --teacher-cot: every this many CoT tokens force </think> and read the answer; the teacher "
                         "context is cut at the first right checkpoint (cot_vs_routes_probe's checkpoints; the rest of "
                         "the CoT is water). 0: the whole CoT")
    ap.add_argument("--on-policy", action="store_true",
                    help="each step the student samples its own answer through the ticks (--student-temp) and the "
                         "loss is KL(student || teacher) at those tokens; the teacher is the same model reading its "
                         "own cut CoT (--target cache from --teacher-cot), the hinted prompt (--teacher-hint) or the "
                         "plain prompt. Learns only on its own tokens (owner 2026-11-04: knowledge without breaking the rest)")
    ap.add_argument("--frozen-teacher", action="store_true",
                    help="with --on-policy: the teacher is a frozen bf16 copy of the starting model, not the model being "
                         "trained (2026-11-05: a self-teacher drifted with the student - 'Answer: <answer in capital "
                         "letters>', quote loops)")
    ap.add_argument("--teacher-model", help="with --frozen-teacher: a bigger model of the same tokenizer family as the "
                         "frozen teacher (Qwen3.5 0.8B/4B share tokenizer.json; RWKV G1 sizes share one) - new information "
                         "the student lacks enters only through the teacher (2026-11-07)")
    ap.add_argument("--anchor-items", type=Path, action="append", default=[],
                    help="with --on-policy: rows the model already reads, added to train with the frozen teacher reading "
                         "the plain prompt - KL(student || base) on its own tokens, so the rest does not move")
    ap.add_argument("--student-temp", type=float, default=1.0)
    ap.add_argument("--student-max-new", type=int, default=32)
    ap.add_argument("--ticks", "--phase-len", dest="ticks", type=int, default=1,
                    help="M = own-latent ticks fed back before the answer (HZLK: M is the number of ticks; "
                         "--phase-len is the old name, kept for old scripts)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--muon-lr", type=float, default=3e-5)
    ap.add_argument("--aux-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--heldout-every", type=int, default=5, help="every n-th item is held out")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="first n items only (smoke tests)")
    ap.add_argument("--eval-max", type=int, default=0, help="teacher-forced eval on at most this many train items (0: all)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="fp32", help="bf16 on the GPU, fp32 on the CPU")
    args = ap.parse_args()
    limit_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.build_cache:
        return build_cache(args)

    loaded = load(args)
    if args.weights:
        load_weights_into(loaded, args.weights)
    tok, model = loaded.tokenizer, loaded.model
    items = load_items(args.items, args.right_from, args.exclude_right_from)
    if args.limit:
        items = items[:args.limit]
    if args.teacher_hint:
        from training.scripts.gen_ciphers import with_hint
        for r in items:
            r["_teacher_prompt"] = with_hint(r, args.teacher_hint)
    if args.target == "cache":                                  # a CoT cache keeps only the items its teacher got right
        cached = set(torch.load(args.cache)["answers"])
        items = [r for r in items if r["id"] in cached]
    held = items[::args.heldout_every]
    train = [r for r in items if r not in held]
    anchors = load_items(args.anchor_items, None, None) if args.anchor_items else []
    if anchors:
        assert args.on_policy, "--anchor-items is an on-policy anchor"
        with torch.no_grad():
            attach_own_answers(loaded, tok, anchors)
        for r in anchors:
            r["_teacher_ids"] = tok.encode(loaded.silent_prompt(r["prompt"]))
        train += anchors
    progress(f"[s1] {len(train)} train / {len(held)} held-out items, loss={args.loss} T={args.temp} target={args.target}")

    if args.target == "gold":
        assert args.loss == "ce", "--target gold has no distribution to KL against"
        end = [loaded._hf_tok.convert_tokens_to_ids("<|im_end|>")] if getattr(loaded, "backend", "") == "hf" else tok.encode("\n\n")
        for r in items:
            r["_ans"] = tok.encode(r["answer"]) + end
        target = {}
    elif args.target == "self":
        with torch.no_grad():
            attach_own_answers(loaded, tok, items)
            target = {r["id"]: seq_logits(loaded, tok, r, 0, r.get("_teacher_prompt"))[0].half() for r in items}
    else:
        c = torch.load(args.cache)
        target = {k: v.to(args.device) for k, v in c["logits"].items()}
        for r in items:
            r["_ans"] = c["answers"][r["id"]]                  # the teacher's own right answer
    teacher = loaded
    from experiments.rl.hf_loader import is_hf_dir
    if args.on_policy and args.frozen_teacher:
        import copy
        ta = copy.copy(args)
        ta.dtype = "bf16" if args.device == "cuda" or is_hf_dir(args.model) else args.dtype
        if args.teacher_model:
            ta.model = args.teacher_model
        teacher = load(ta)
        if args.weights and not args.teacher_model:
            load_weights_into(teacher, args.weights)
        for q in teacher.model.parameters():
            q.requires_grad_(False)
    if args.on_policy:                                          # what the teacher reads before the student's tokens
        assert args.target != "gold", "--on-policy needs a teacher distribution, not a gold string"
        ctx = c.get("ctx", {}) if args.target == "cache" else {}
        for r in items:
            if r["id"] in ctx:
                r["_teacher_ids"] = ctx[r["id"]]
            elif args.target == "cache" and r["id"] in c.get("cots", {}):   # caches before 2026-11-04 kept the CoT text
                cc = c["cots"][r["id"]]
                r["_teacher_ids"] = cot_ctx(tok, tok.encode(loaded.think_prompt(r["prompt"])), tok.encode(cc["cot"]),
                                            cc["first_right"])
            else:
                r["_teacher_ids"] = tok.encode(loaded.silent_prompt(r.get("_teacher_prompt") or r["prompt"]))
    status = args.out.with_suffix(".status.json")
    ev_train = train[:args.eval_max] if args.eval_max else train  # the train half of the eval is a sanity check
    history = [{"step": 0, "train": evaluate(loaded, tok, ev_train, args.ticks),
                "held": evaluate(loaded, tok, held, args.ticks)}]
    progress(f"[s1] step 0 {history[-1]}")

    opt = MuonHybrid(model.named_parameters(), lr=args.muon_lr, momentum_warmup_steps=0)
    aux = torch.optim.AdamW(opt.other_params, lr=args.aux_lr)
    rng, order, pos, run, t0 = random.Random(args.seed), list(range(len(train))), 0, [], time.time()
    for step in range(1, args.steps + 1):
        scale = min(1.0, step / max(1, args.warmup))
        opt.lr = args.muon_lr * scale
        for g in aux.param_groups:
            g["lr"] = args.aux_lr * scale
        opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if pos == 0:
                rng.shuffle(order)
            r = train[order[pos]]; pos = (pos + 1) % len(order)
            if args.on_policy:
                a = student_sample(loaded, tok, r, args.ticks, args.student_temp, args.student_max_new)
                lg, _ = seq_logits(loaded, tok, r, args.ticks, ans=a)
                loss = reverse_kl(lg, teacher_logits(teacher, r["_teacher_ids"], a))
            else:
                lg, a = seq_logits(loaded, tok, r, args.ticks)
                loss = F.cross_entropy(lg, a) if args.loss == "ce" else kl_loss(lg, target[r["id"]], args.temp)
            (loss / args.accum).backward()
            run.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); aux.step()
        if step <= 5 or step % 10 == 0:
            progress(f"[s1] step {step} loss {sum(run[-40:]) / len(run[-40:]):.4f} {time.time() - t0:.0f}s")
        write_heartbeat(status, progress=(step, args.steps), job="train_latent_s1", loss=run[-1])
        if step % args.eval_every == 0 or step == args.steps:
            history.append({"step": step, "loss": sum(run[-40:]) / len(run[-40:]),
                            "train": evaluate(loaded, tok, ev_train, args.ticks),
                            "held": evaluate(loaded, tok, held, args.ticks)})
            progress(f"[s1] step {step} {history[-1]}")
            ck = args.out.with_suffix(".pth")
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ck.with_suffix(".tmp"))
            ck.with_suffix(".tmp").replace(ck)
            if step != args.steps:                                 # every checkpoint is gated, not only the last
                shutil.copyfile(ck, ck.with_name(f"{ck.stem}_s{step}.pth"))

    from experiments._common.results import save_result
    args_json = {k: (str(v) if isinstance(v, Path) else ([str(x) for x in v] if isinstance(v, list) else v))
                 for k, v in vars(args).items()}
    last = history[-1]
    save_result(args.out, {"model": args.model, "args": args_json, "n_train": len(train), "n_held": len(held),
                           "history": history, "checkpoint": str(args.out.with_suffix(".pth"))},
                experiment="train_latent_s1", hypothesis=["HZLK"], model=args.model,
                script="experiments/rl/train_latent_s1.py",
                summary={f"held acc M0 -> M{args.ticks} (step 0)": f"{history[0]['held']['acc_M0']} -> {history[0]['held'][f'acc_M{args.ticks}']}",
                         f"held acc M{args.ticks} (last)": str(last["held"][f"acc_M{args.ticks}"]), "loss": f"{args.loss} T={args.temp}",
                         "ticks": f"M={args.ticks}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
