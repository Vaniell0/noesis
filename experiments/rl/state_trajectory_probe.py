#!/usr/bin/env python3
"""state_trajectory_probe.py — per-token mechanism trace: WKV state motion
AND the raw R/K/V/decay/in-context-LR/gate values the CUDA kernel actually
consumes, for three input regimes on the same fixed prompts:

  read  — real prompt tokens, prefill, token-by-token (unchanged from the
          2026-08-21 version).
  loop  — the OLD self-feed mechanism (argmax next-token, feed back,
          repeat) — kept as the empirical baseline to compare against.
  chain — the NEW ThinkChain mechanism (train_think_distill.py's
          ThinkChain): M explicitly-distinct learned phase markers plus
          a shared entry cue, fed straight into WKV via
          forward_stateful_embeds, no self-feed loop.

2026-08-21 rewrite (redefinition, not an extension): the previous version
only recorded WKV state norm/delta per token — a downstream *symptom* of
whatever R/K/V/decay/a/g the model computed, not the mechanism itself.
This still doesn't answer "why does loop collapse toward a fixed point
and chain not" at the level RWKV-7's actual per-token recurrence works
at: r/k/v/decay/a/g are plain local variables inside
`rwkvt/rwkv7/att.py::RWKV_Tmix_x070_infctx.forward`, computed before the
CUDA kernel call, never returned by `loader.py::forward_stateful[_embeds]`
(only the post-kernel `wkv` state is). w/a/g are inline matmuls on raw
`nn.Parameter`s (`self.w1/w2`, `self.a1/a2`, `self.g1/g2`), not separate
`nn.Linear` submodules — so there is no hookable module to attach a
`register_forward_hook` to for them. Only a temporary monkeypatch of
`RWKV_Tmix_x070_infctx.forward` itself (mirroring the exact reference
body, same convention as `loader.py::_peft_forward_embeds`) can capture
them; see `_capture_rkvwag` below. Re-sync that function against
`rwkvt/rwkv7/att.py:235-279` if the vendored file changes.

2026-08-23 addition: Huginn-style backtrack detection (arXiv 2602.08100,
"Emergent Search and Backtracking in Latent Reasoning Models" — found
that backtracking in a looped latent-reasoning transformer emerges
without being explicitly trained for, just from variable recurrence
depth, and is detectable for free by decoding the readout distribution
at every internal step and watching the majority-vote answer flip:
one token dominant >=3 steps, then a DIFFERENT token dominant >=3
steps). ThinkChain's `forward_stateful_embeds` already returns fresh
readout logits after every call, same as Huginn's decoder-ready coda —
no new probe needed, just decode more often. The `chain` branch below
now repeats each phase marker `--phase-repeat-ticks` times (was: one
call per phase) and records the per-tick readout argmax; `_detect_backtrack`
runs the same rule against both this new chain stream and the existing
`loop` branch's real self-fed token stream (already free — those tokens
ARE argmax-decoded each step), so the two mechanisms are directly
comparable on this measure. This is the task-#12 diagnostic ahead of
building a dedicated Phase-2 rewind marker (`docs/rl-track.md` §Track
status) — first check whether answer-flips already occur without one.

`retention` (per layer, per token): the ACTUAL per-channel decay
multiplier applied to the WKV state each step is `exp(-exp(w))` — a
DOUBLE exponential of `w` (RWKV-7's per-channel log-decay logit,
soft-clamped to `(-inf, -0.5)` off the CUDA path, unclamped on it —
`os.environ["WKV"]`), not `exp(w)` (this field's name until 2026-08-22,
when computing plain `exp(w)` here was caught as the wrong quantity —
see `docs/rwkv7-mechanics.md` §3-4 for the full derivation, done after
this mistake, not before). The soft-clamp on `w` bounds `retention`
from BELOW (~0.545 minimum — a channel can never forget faster than
that), not from above: nothing prevents `w` training toward large
negative values, pushing retention arbitrarily close to 1 (near-total
memory). This is the direct, measurable quantity behind the still-open
`docs/community-map.md` §"Latent imagination" question — decay's
effective contractivity is NOT architecturally guaranteed on this
backend, contrary to an earlier (wrong, same-day) reading of this
exact field.

Usage:
    python experiments/rl/state_trajectory_probe.py \\
        --model models/rwkv7-g1i-2.9b-....pth \\
        --out experiments/rl/results/state_trajectory_<name>.json \\
        [--lora-r 32 --lora-alpha 64 --resume path/to/ckpt_stepN] \\
        [--chain-phases 3] [--capture-layers all|12,16,20] [--save-raw]
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import load_checkpoint
from experiments.rl.wkv_loop import _last_vec
from experiments.rl.train_think_distill import ThinkChain
from experiments._common.results import save_result
from training.state_reg import DEFAULT_WORK_LAYERS

# Same 4 fixed prompts as the earlier one-off diagnostic decoder
# (_diag_think_content.py, deleted after use) — matrix addition,
# wordsearch, arithmetic sequence, XOR — kept identical so results are
# directly comparable to that session's qualitative findings.
PROMPTS = {
    "matrix_addition": "You are a precise reasoning assistant. Work step by step.\n\n"
        "Add these two 2x2 matrices:\nA = [[1, 2], [3, 4]]\nB = [[5, 6], [7, 8]]\n\n<think>\n",
    "wordsearch": "You are a precise reasoning assistant. Work step by step.\n\n"
        "Below is a 4x4 letter matrix. Rows are separated by newlines; letters within a row are "
        "separated by single spaces.\n\nC A T S\nD O G X\nB I R D\nF I S H\n\n"
        "Find the word CAT reading in any direction (horizontal, vertical, or diagonal).\n\n<think>\n",
    "arithmetic_sequence": "You are a precise reasoning assistant. Work step by step.\n\n"
        "What is the next number in this sequence: 2, 4, 6, 8, ?\n\n<think>\n",
    "xor": "You are a precise reasoning assistant. Work step by step.\n\n"
        "Compute the bitwise XOR of 1010 and 0110.\n\n<think>\n",
}

MAX_LOOP_TOKENS = 40  # generous over the corpus's real ~8-token phase budget, to see saturation


def _state_norms(wkv, layers) -> dict[int, float]:
    return {L: float(torch.linalg.vector_norm(wkv[L].float().flatten()).item()) for L in layers}


def _delta_norms(wkv, prev_wkv, layers) -> dict[int, float] | None:
    """‖S_t - S_{t-1}‖ per layer — true displacement, not difference of
    norms (a state can rotate a lot at near-constant magnitude, or barely
    move at large magnitude; norm(a)-norm(b) conflates both with actual
    motion). None on the first token of a trace (no previous state)."""
    if prev_wkv is None:
        return None
    return {L: float(torch.linalg.vector_norm((wkv[L].float() - prev_wkv[L].float()).flatten()).item())
            for L in layers}


def _detect_backtrack(answer_stream: list[int], min_streak: int = 3) -> dict:
    """Huginn-style backtrack detection (arXiv 2602.08100): a backtrack is
    one answer-token dominating >= min_streak consecutive internal steps,
    then a DIFFERENT answer-token dominating the next >= min_streak steps
    (their operational definition, not a looser "did it ever change").
    Runs on any per-step token-id stream — the chain branch's per-tick
    readout argmax, or the loop branch's real self-fed tokens (same rule,
    two different streams, see module docstring)."""
    streaks: list[tuple[int, int, int]] = []  # (token_id, start_idx, length)
    cur_tok, cur_start = None, 0
    for i, tok in enumerate(answer_stream):
        if tok != cur_tok:
            if cur_tok is not None:
                streaks.append((cur_tok, cur_start, i - cur_start))
            cur_tok, cur_start = tok, i
    if answer_stream:
        streaks.append((cur_tok, cur_start, len(answer_stream) - cur_start))
    qualifying = [s for s in streaks if s[2] >= min_streak]
    streak_dicts = [{"token": t, "start": s, "len": l} for t, s, l in streaks]
    for a, b in zip(qualifying, qualifying[1:]):
        if a[0] != b[0]:
            return {"backtracked": True, "from_token": a[0], "to_token": b[0],
                    "from_streak_start": a[1], "to_streak_start": b[1],
                    "streaks": streak_dicts}
    return {"backtracked": False, "streaks": streak_dicts}


def _tensor_stats(t: torch.Tensor) -> dict:
    tf = t.detach().float()
    return {"norm": float(torch.linalg.vector_norm(tf.flatten()).item()),
            "mean": float(tf.mean().item()),
            "min": float(tf.min().item()),
            "max": float(tf.max().item())}


@contextmanager
def _capture_rkvwag(capture_layers, save_raw: bool):
    """Monkeypatches RWKV_Tmix_x070_infctx.forward (mirror of
    rwkvt/rwkv7/att.py:235-279 — see module docstring) to record r/w/k/v/
    a/g/kk per layer per call, restoring the original forward on exit.
    Yields a list; each Tmix forward() call (one per layer, per outer
    forward_stateful[_embeds] call) appends one dict when
    self.layer_id in capture_layers.
    """
    import os
    from rwkvt.rwkv7.att import RWKV_Tmix_x070_infctx
    from rwkvt.infctx_module import TimeMixState
    from rwkvt.operator.rwkvop import RUN_RWKV7_INFCTX

    buf: list = []
    orig_forward = RWKV_Tmix_x070_infctx.forward

    def patched(self, x, v_first, last_state, attention_mask=None):
        B, T, C = x.size()
        H = self.n_head
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        shift_state = last_state.shift_state
        wkv_state = last_state.wkv_state.clone().contiguous()
        xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)
        shift_state = x[:, -1, :]

        r = self.receptance(xr)
        if os.environ["WKV"] == 'cuda':
            w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        else:
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        if self.layer_id in capture_layers:
            # retention = exp(-exp(w)) — the ACTUAL per-channel decay
            # multiplier RUN_RWKV7_INFCTX applies to the state each step
            # (see rwkvfla/ops/rwkv7/recurrent_naive.py:63, and
            # docs/rwkv7-mechanics.md §3-4). NOT exp(w) — that was this
            # probe's original (wrong) field, `exp_w_max`, computed
            # 2026-08-22 and caught the same day: exp(w) silently used
            # the wrong direction (w is soft-clamped ABOVE at -0.5, which
            # bounds retention=exp(-exp(w)) from BELOW, not above — the
            # opposite of what "exp_w_max ≈ constant ⇒ contractivity
            # capped" was read to mean). Computed on the real tensor
            # directly, not inferred from w's summary stats, to avoid
            # exactly this kind of indirection error a second time.
            retention = torch.exp(-torch.exp(w.float()))
            rec = {"layer": self.layer_id,
                   "r": _tensor_stats(r), "w": _tensor_stats(w),
                   "k": _tensor_stats(k), "v": _tensor_stats(v),
                   "a": _tensor_stats(a), "g": _tensor_stats(g),
                   "kk": _tensor_stats(kk),
                   "retention": _tensor_stats(retention)}
            if save_raw:
                rec["raw"] = {name: t.detach().cpu()
                               for name, t in (("r", r), ("w", w), ("k", k),
                                                ("v", v), ("a", a), ("g", g), ("kk", kk))}
            buf.append(rec)

        x_out, wkv_state = RUN_RWKV7_INFCTX(r, k, v, w, -kk, kk * a, wkv_state)
        x_out = self.ln_x(x_out.view(B * T, C)).view(B, T, C)
        x_out = x_out + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
                          .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x_out = self.output(x_out * g)
        return x_out, v_first, TimeMixState(shift_state, wkv_state)

    RWKV_Tmix_x070_infctx.forward = patched
    try:
        yield buf
    finally:
        RWKV_Tmix_x070_infctx.forward = orig_forward


def _step(loaded, x_or_ids, state, layers, capture_layers, save_raw, use_embeds: bool):
    with _capture_rkvwag(capture_layers, save_raw) as cap:
        if use_embeds:
            logits, state = loaded.forward_stateful_embeds(x_or_ids, state)
        else:
            logits, state = loaded.forward_stateful(x_or_ids, state)
    return logits, state, cap


def trace_prompt(loaded, prompt_text: str, layers, capture_layers, save_raw: bool,
                  think_marker, n_chain_phases: int, tok, phase_repeat_ticks: int) -> dict:
    ids = tok.encode(prompt_text)
    device = loaded.device

    def entry(pos_or_step, token_id, state, prev_wkv, prev_delta, cap):
        """Returns (entry_dict, new_delta) — new_delta is this step's
        wkv[L]-prev_wkv[L] per layer, threaded back in as the caller's
        next prev_delta. delta_cos_prev = cosine similarity between THIS
        step's delta and the PREVIOUS step's delta, per layer — the
        direct test for loop-collapse (not just magnitude, which
        delta_norms already covers): if consecutive deltas converge
        toward the same direction (cos -> 1), the mechanism is applying
        an increasingly self-similar transformation each step rather than
        doing genuinely different work — this is the specific,
        measurable signature the whole ThinkChain rewrite was motivated
        by (see docs/rwkv7-mechanics.md and the 2026-08-21 commit
        message), previously only argued from RWKV mechanics, not
        measured. None on the first two steps of a trace (no prior delta
        to compare against)."""
        new_delta = None if prev_wkv is None else {
            L: (state.wkv[L].float() - prev_wkv[L].float()).flatten() for L in layers}
        cos_prev = None if (new_delta is None or prev_delta is None) else {
            L: float(F.cosine_similarity(new_delta[L].unsqueeze(0),
                                          prev_delta[L].unsqueeze(0)).item())
            for L in layers}
        d = {"pos": pos_or_step, "token_id": token_id,
             "norms": _state_norms(state.wkv, layers),
             "delta_norms": _delta_norms(state.wkv, prev_wkv, layers),
             "delta_cos_prev": cos_prev,
             "rkvwag": cap}
        return d, new_delta

    # --- read: real prompt tokens, prefill, token-by-token ---
    state = loaded.new_state(batch=1)
    read_trace = []
    prev_wkv = None
    prev_delta = None
    for pos, tid in enumerate(ids):
        x = torch.tensor([[tid]], device=device)
        logits, state, cap = _step(loaded, x, state, layers, capture_layers, save_raw, use_embeds=False)
        e, prev_delta = entry(pos, tid, state, prev_wkv, prev_delta, cap)
        read_trace.append(e)
        prev_wkv = state.wkv
    read_end_state, read_end_logits = state, logits

    # --- loop: OLD mechanism — argmax next token, feed back, repeat ---
    # prev_wkv/prev_delta carry over from the read phase above, so the
    # first loop step's delta_norms/delta_cos_prev measure distance and
    # direction-change from end-of-read, not from nothing.
    state = read_end_state
    logits = read_end_logits
    loop_trace = []
    for step in range(MAX_LOOP_TOKENS):
        v = _last_vec(logits)
        next_id = int(v.argmax().item())
        x = torch.tensor([[next_id]], device=device)
        logits, state, cap = _step(loaded, x, state, layers, capture_layers, save_raw, use_embeds=False)
        e, prev_delta = entry(step, next_id, state, prev_wkv, prev_delta, cap)
        loop_trace.append(e)
        prev_wkv = state.wkv
        if next_id == 0:  # EOS
            break

    # --- chain: NEW mechanism — ThinkChain markers, no self-feed loop ---
    # Independent branch point off read_end_state (forward_stateful[_embeds]
    # returns fresh state objects, never mutates in place — same reasoning
    # as the loop branch above, so this is unaffected by the loop trace
    # already having run). prev_wkv/prev_delta reset to end-of-read for
    # the same reason as the loop branch.
    state = read_end_state
    chain_trace = []
    prev_wkv = read_end_state.wkv
    prev_delta = None
    marker0 = think_marker.step(0).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
    logits, state, cap = _step(loaded, marker0, state, layers, capture_layers, save_raw, use_embeds=True)
    e, prev_delta = entry(0, "<entry>", state, prev_wkv, prev_delta, cap)
    chain_trace.append(e)
    prev_wkv = state.wkv
    # readout_stream: argmax of the readout logits after EVERY tick (not
    # just once per phase) — the quantity Huginn's decoder-ready coda
    # exposes for free, used to test for answer-flips across ticks/phases
    # (see module docstring, task #12). First entry is the readout right
    # after the shared entry cue, before any phase-specific work.
    readout_stream: list[int] = [int(_last_vec(logits).argmax().item())]
    for i in range(n_chain_phases):
        marker_i = think_marker.step(i + 1).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
        for tick in range(phase_repeat_ticks):
            logits, state, cap = _step(loaded, marker_i, state, layers, capture_layers, save_raw, use_embeds=True)
            e, prev_delta = entry(f"{i}.{tick}", f"<phase{i}:{tick}>", state, prev_wkv, prev_delta, cap)
            chain_trace.append(e)
            prev_wkv = state.wkv
            readout_stream.append(int(_last_vec(logits).argmax().item()))

    chain_backtrack = _detect_backtrack(readout_stream)
    # loop branch's own tokens already ARE argmax-decoded, real self-fed
    # generation — no new capture needed, same detector, direct comparison.
    loop_backtrack = _detect_backtrack([e["token_id"] for e in loop_trace])

    return {"read": read_trace, "loop": loop_trace, "chain": chain_trace,
            "chain_readout_stream": readout_stream, "chain_backtrack": chain_backtrack,
            "loop_backtrack": loop_backtrack, "prompt_n_tokens": len(ids)}


def _load_think_marker(n_embd: int, n_phases: int, device: str) -> ThinkChain:
    """Fresh, randomly-initialized ThinkChain — loading a trained one (if
    --resume is set) happens via checkpoint.py::load_checkpoint's own
    mlp_delta= parameter in main(), in the SAME call that loads the base
    weights, not a separate manual file read. (2026-08-22 fix: this
    function used to reach for a `model/think_marker.chain.pt` file that
    never existed — save_checkpoint stores the marker under meta.pt's
    `mlp_delta` key, same convention train_think_distill.py's own
    save_checkpoint(..., mlp_delta=think_marker) call uses. The old code
    silently fell back to random init on every --resume, never actually
    loading a trained checkpoint's markers.)"""
    return ThinkChain(n_embd, n_phases).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None,
                     help="Optional checkpoint dir to load on top of --model (LoRA or "
                          "full-FT); also tried for model/think_marker.chain.pt.")
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--work-layers", default=",".join(str(x) for x in DEFAULT_WORK_LAYERS),
                     help="Layers for the existing wkv state norm/delta metrics.")
    ap.add_argument("--capture-layers", default="all",
                     help="Layers for the new per-token r/w/k/v/a/g/kk capture. "
                          "'all' or a comma list, e.g. '12,16,20'.")
    ap.add_argument("--save-raw", action="store_true",
                     help="Also dump full raw r/w/k/v/a/g/kk tensors (not just "
                          "norm/mean/min/max) to <out>.raw.pt. Off by default — "
                          "full tensors at every captured layer/token add up fast.")
    ap.add_argument("--chain-phases", type=int, default=3,
                     help="M for the ThinkChain trace branch (3 phases + shared "
                          "entry cue, per 2026-08-21 decision).")
    ap.add_argument("--phase-repeat-ticks", type=int, default=8,
                     help="Repeat ticks per phase marker in the chain branch "
                          "(matches train_think_distill.py's --max-phase-tokens "
                          "default) — needed for the Huginn-style backtrack "
                          "check to have enough per-phase readout samples.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.work_layers.split(","))
    loaded = load_rwkv7(args.model, device=args.device, backend="peft",
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    think_marker = _load_think_marker(loaded.n_embd, args.chain_phases, args.device)
    if args.resume is not None:
        # mlp_delta=think_marker: loads the checkpoint's trained ThinkChain
        # markers in the SAME call as the base weights, matching how
        # train_think_distill.py's save_checkpoint(..., mlp_delta=
        # think_marker) wrote them. A pre-ThinkChain checkpoint (e.g. the
        # archived step200 loop-based run) DOES have an "mlp_delta" key —
        # it's the OLD ThinkMarker's single `embed` vector — so
        # load_state_dict(strict=True) raises on the key mismatch rather
        # than silently doing nothing; caught here specifically because a
        # diagnostic tool comparing old vs. new checkpoints is exactly the
        # situation where trying the wrong marker class on purpose (to see
        # what happens) is a reasonable thing to do, unlike training's own
        # resume path where a mismatch should stay a hard failure.
        try:
            step = load_checkpoint(args.resume, loaded, mlp_delta=think_marker)
            print(f"[state_trajectory_probe] resumed base weights + trained think_marker from {args.resume} at step {step}")
        except RuntimeError as e:
            step = load_checkpoint(args.resume, loaded, mlp_delta=None)
            print(f"[state_trajectory_probe] resumed base weights from {args.resume} at step {step}; "
                  f"WARNING: think_marker NOT loaded (checkpoint's mlp_delta doesn't match "
                  f"ThinkChain's shape — likely a pre-ThinkChain checkpoint): {e}")
    else:
        print("[state_trajectory_probe] no --resume — randomly-initialized ThinkChain "
              "(mechanism-only comparison, no trained weights)")

    capture_layers = (set(range(loaded.n_layer)) if args.capture_layers == "all"
                       else set(int(x) for x in args.capture_layers.split(",")))

    tok = loaded.tokenizer
    results = {}
    raw_out = {} if args.save_raw else None
    with torch.no_grad():
        for name, prompt_text in PROMPTS.items():
            print(f"[state_trajectory_probe] tracing {name} ...")
            r = trace_prompt(loaded, prompt_text, layers, capture_layers, args.save_raw,
                              think_marker, args.chain_phases, tok, args.phase_repeat_ticks)
            if args.save_raw:
                raw_out[name] = {}
                for branch in ("read", "loop", "chain"):
                    branch_raw = []
                    for e in r[branch]:
                        for layer_rec in e["rkvwag"]:
                            raw = layer_rec.pop("raw", None)  # removes "raw" from the
                            if raw is not None:                # JSON-bound dict too —
                                branch_raw.append({"pos": e["pos"], "layer": layer_rec["layer"], **raw})
                    raw_out[name][branch] = branch_raw
            else:
                for branch in ("read", "loop", "chain"):
                    for e in r[branch]:
                        for layer_rec in e["rkvwag"]:
                            layer_rec.pop("raw", None)
            results[name] = r
            n_read, n_loop, n_chain = len(r["read"]), len(r["loop"]), len(r["chain"])
            loop_hit_eos = r["loop"][-1]["token_id"] == 0 if n_loop else False
            print(f"  read={n_read} tok, loop={n_loop} tok (eos={loop_hit_eos}), chain={n_chain} steps")
            L0 = layers[0]
            for branch in ("read", "loop", "chain"):
                deltas = [e["delta_norms"][L0] for e in r[branch] if e["delta_norms"] is not None]
                if deltas:
                    mean_d = sum(deltas) / len(deltas)
                    print(f"  [{branch}] L{L0} delta_norm: mean={mean_d:.2f} max={max(deltas):.2f}")
                # delta_cos_prev TREND, not just average — the direct
                # loop-collapse signature is cos drifting toward 1 over
                # the course of the branch (each step's delta more
                # parallel to the last), not a static mean. First-half vs
                # second-half mean makes a drift visible at a glance.
                coss = [e["delta_cos_prev"][L0] for e in r[branch] if e["delta_cos_prev"] is not None]
                if len(coss) >= 4:
                    half = len(coss) // 2
                    first_half = sum(coss[:half]) / half
                    second_half = sum(coss[half:]) / (len(coss) - half)
                    print(f"  [{branch}] L{L0} delta_cos_prev: first_half={first_half:.3f} "
                          f"second_half={second_half:.3f} (drift toward 1 = loop-collapse signature)")
            if capture_layers:
                Lc = sorted(capture_layers)[0]
                for branch in ("read", "loop", "chain"):
                    ret = [next((lr["retention"]["max"] for lr in e["rkvwag"] if lr["layer"] == Lc), None)
                           for e in r[branch]]
                    ret = [x for x in ret if x is not None]
                    if ret:
                        print(f"  [{branch}] L{Lc} retention(max-over-channels): "
                              f"mean={sum(ret)/len(ret):.4f} max={max(ret):.4f}")
            for branch, bt in (("chain", r["chain_backtrack"]), ("loop", r["loop_backtrack"])):
                if bt["backtracked"]:
                    print(f"  [{branch}] BACKTRACK: token {bt['from_token']} "
                          f"(from tick {bt['from_streak_start']}) -> token {bt['to_token']} "
                          f"(from tick {bt['to_streak_start']})")
                else:
                    n_streaks = len(bt["streaks"])
                    print(f"  [{branch}] no backtrack ({n_streaks} distinct streak(s), "
                          f"min-streak=3 rule)")

    # Real min/max retention actually observed this run (not a hardcoded
    # guess) — every captured layer/token/branch/prompt, so the summary
    # can't silently go stale if a future run behaves differently.
    all_ret = [lr["retention"][stat] for name in results for branch in ("read", "loop", "chain")
               for e in results[name][branch] for lr in e["rkvwag"] for stat in ("min", "max")]
    ret_summary = f"{min(all_ret):.4f}-{max(all_ret):.4f} across all layers/tokens/branches/prompts"

    n_prompts = len(results)
    n_chain_backtrack = sum(1 for r in results.values() if r["chain_backtrack"]["backtracked"])
    n_loop_backtrack = sum(1 for r in results.values() if r["loop_backtrack"]["backtracked"])
    backtrack_summary = (f"chain={n_chain_backtrack}/{n_prompts} prompts backtracked, "
                          f"loop={n_loop_backtrack}/{n_prompts} (Huginn min-streak=3 rule; "
                          f"n=4 fixed prompts, not a statistical claim — see arXiv 2602.08100 "
                          f"for their 32%/34%-accuracy numbers on 260 real instances)")

    save_result(
        args.out,
        {"model": str(args.model), "resume": str(args.resume) if args.resume else None,
         "work_layers": list(layers), "capture_layers": sorted(capture_layers),
         "chain_phases": args.chain_phases, "results": results},
        experiment="state_trajectory",
        hypothesis=["H25"],
        status="done" if args.resume else "partial",  # partial: mechanism-only
        # comparison on random ThinkChain markers until a checkpoint with
        # REAL trained ThinkChain markers exists to --resume from (see
        # docs/rwkv7-mechanics.md §5) — true regardless of whether --model
        # itself is trained (e.g. the pre-ThinkChain step200 checkpoint).
        summary={"retention_range": ret_summary, "backtrack": backtrack_summary},
        model=str(args.model),
        script=str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
    )
    print(f"[state_trajectory_probe] -> {args.out}")
    if args.save_raw:
        raw_path = args.out.with_suffix(args.out.suffix + ".raw.pt")
        torch.save(raw_out, raw_path)
        print(f"[state_trajectory_probe] raw tensors -> {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
