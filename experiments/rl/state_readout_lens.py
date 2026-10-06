#!/usr/bin/env python3
"""state_readout_lens.py — which WKV-state directions does the model actually read?

Three numbers on one checkpoint, for the same states:

  occupied   singular directions of a head's state above 1% of sigma_1
             (what `jlens_probe.py` has always reported — energy, not use)
  read_grad  directions whose removal would move the future output, to first
             order: |u_i^T G v_i| * sigma_i, where G = d(future log-prob)/dS
  read_abl   the same, measured by actually removing sigma_i u_i v_i^T from
             the state and re-running the continuation (causal, no linearisation)

`read_*` uses the same 1%-of-max rule as `occupied`, so the three are counted in
the same units. An entropy-based effective count (exp of the entropy of the
normalised relevance) is reported beside each, since a 1% threshold is only a
convention.

## Where this comes from

Anthropic's Jacobian lens (arXiv 2607.15495, "Verbalizable Representations
Form a Global Workspace in Language Models") averages
d h_final,t' / d h_l,t over all LATER positions to find what a model is
"positioned to verbalize". This is the same idea pointed at a different
object: not the residual stream of a transformer but the recurrent WKV state,
which persists between tokens — the property the paper lists as absent from
transformers. It is also what this project's own `jlens_probe.py` was once
meant to be: that file's docstring records that its Jacobian was never wired
up and was removed, and it measures a state SVD. Same name, different
measurement; see `reference_anthropic_jspace` in memory.

## The question it answers

On g1d-0.4b the state carries 12-23 occupied directions per head, but a linear
probe retrieves only 2 key->value bindings (`state_capacity_probe.py`). Between
"energy sits here" and "a probe can decode it" there is a third quantity that
neither measures: what the model's OWN downstream computation is sensitive to.
That is this.

## Controls, per the discriminability rule (memory: metric_discriminability)

- `read_grad` is first-order; `read_abl` is causal. Their per-head Spearman
  correlation is reported. If it is low, the gradient lens is not measuring
  what ablation measures and should not be quoted alone.
- Two prompt families: natural text (continuation = the next tokens) and
  key->value records queried after the fact (continuation = the answer). The
  second is where we already know only ~2 bindings are retrievable.
- Model parameters have requires_grad disabled: only the state is a leaf, so
  the gradient is with respect to the state and nothing else.

Inference-grade on CPU via the peft backend (`_enable_peft_on_cpu`).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, _PeftState
from experiments.rl.state_capacity_probe import make_sample, KEYS


def _rank_count(x: torch.Tensor, frac: float = 0.01) -> int:
    x = x.abs()
    return int((x > frac * x.max().clamp_min(1e-30)).sum()) if x.numel() else 0


def _eff_count(x: torch.Tensor) -> float:
    p = x.abs() / x.abs().sum().clamp_min(1e-30)
    return float(torch.exp(-(p * (p + 1e-30).log()).sum()))


def _spearman(a: list, b: list) -> float:
    if len(a) < 3:
        return float("nan")
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / d)


def build_prompts(tok, n_text: int, n_kv: int, data: Path, seed: int,
                  prompt_max: int, cont_len: int) -> list:
    rng = random.Random(seed)
    out = []
    texts = []
    with open(data) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    rng.shuffle(texts)
    for t in texts:
        if len(out) >= n_text:
            break
        ids = tok.encode(t)
        if len(ids) < 64 + cont_len:
            continue
        cut = rng.randint(48, min(len(ids) - cont_len, prompt_max))
        out.append({"family": "text", "prompt": ids[:cut],
                    "cont": ids[cut:cut + cont_len],
                    "target_from": 0})
    for j in range(n_kv):
        n = 2 if j % 2 == 0 else 8
        rec, vals = make_sample(rng, n, 99)
        q = rng.randrange(n)
        p_ids = tok.encode(rec)
        query = tok.encode(f"\n{KEYS[q]}=")
        ans = tok.encode(str(vals[q]))
        out.append({"family": f"kv{n}", "prompt": p_ids, "cont": query + ans,
                    "target_from": len(query)})
    return out


def _future_logp(loaded, state, cont: list, target_from: int,
                 grad: bool) -> tuple:
    """Feed cont[:-1] from `state`, score cont[1:]. Returns (sum logp, per-pos
    log-softmax). Positions before target_from are fed but not scored, so for a
    key->value query only the ANSWER tokens count."""
    inp = torch.tensor([cont[:-1]])
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits, _ = loaded.forward_stateful(inp, state)
        lsm = F.log_softmax(logits[0].float(), dim=-1)          # [T-1, V]
        tgt = torch.tensor(cont[1:])
        lp = lsm.gather(1, tgt[:, None])[:, 0]
        mask = torch.arange(len(cont) - 1) >= max(target_from - 1, 0)
        return lp[mask].sum(), lsm


def run_prompt(loaded, item: dict, layers: list, abl_layers: list,
               abl_heads: int, abl_dirs: int) -> dict:
    with torch.no_grad():
        st = loaded.new_state(batch=1)
        _, st = loaded.forward_stateful(torch.tensor([item["prompt"]]), st)
    base_shift = st.shift.detach().clone()
    base_wkv = st.wkv.detach().clone()

    # --- gradient lens: one backward covers every layer and head ---------
    wkv_leaf = base_wkv.clone().requires_grad_(True)
    lp, _ = _future_logp(loaded, _PeftState(base_shift.clone(), wkv_leaf),
                         item["cont"], item["target_from"], grad=True)
    lp.backward()
    G = wkv_leaf.grad.detach()
    base_lp = float(lp.detach())
    with torch.no_grad():
        _, base_lsm = _future_logp(loaded, _PeftState(base_shift, base_wkv),
                                   item["cont"], item["target_from"], grad=False)

    rows = []
    for L in layers:
        S_all = base_wkv[L, 0].float()                   # [H, h, h]
        G_all = G[L, 0].float()
        for h in range(S_all.shape[0]):
            U, sig, Vh = torch.linalg.svd(S_all[h], full_matrices=False)
            # first-order change in log-prob from deleting component i:
            #   -<G, sigma_i u_i v_i^T> = -sigma_i * u_i^T G v_i
            r_grad = (sig * torch.einsum("ji,jk,ik->i", U, G_all[h], Vh)).abs()
            row = {"layer": L, "head": h,
                   "occupied": _rank_count(sig),
                   "read_grad": _rank_count(r_grad),
                   "eff_grad": _eff_count(r_grad),
                   "sigma1": float(sig[0])}
            if L in abl_layers and h < abl_heads:
                deltas = []
                m = min(abl_dirs, int((sig > 0.01 * sig[0]).sum()))
                for i in range(m):
                    w = base_wkv.clone()
                    comp = sig[i] * torch.outer(U[:, i], Vh[i, :])
                    w[L, 0, h] = (w[L, 0, h].float() - comp).to(w.dtype)
                    lp_i, lsm_i = _future_logp(
                        loaded, _PeftState(base_shift.clone(), w),
                        item["cont"], item["target_from"], grad=False)
                    deltas.append(abs(float(lp_i) - base_lp))
                d = torch.tensor(deltas) if deltas else torch.zeros(0)
                row.update({"read_abl": _rank_count(d),
                            "eff_abl": _eff_count(d) if deltas else float("nan"),
                            "spearman_grad_abl": _spearman(
                                r_grad[:m].tolist(), deltas),
                            "abl_dirs_tested": m,
                            "max_abl_delta_logp": float(d.max()) if deltas else 0.0})
            rows.append(row)
    return {"family": item["family"], "base_logp": base_lp, "heads": rows}


def summarise(results: list, layers: list) -> dict:
    out = {}
    for fam in sorted({r["family"] for r in results}):
        rs = [r for r in results if r["family"] == fam]
        per_layer = {}
        for L in layers:
            hs = [h for r in rs for h in r["heads"] if h["layer"] == L]
            if not hs:
                continue
            mean = lambda k: (sum(h[k] for h in hs if k in h and not
                                  (isinstance(h[k], float) and math.isnan(h[k])))
                              / max(1, sum(1 for h in hs if k in h and not
                                           (isinstance(h[k], float) and math.isnan(h[k])))))
            per_layer[L] = {k: round(mean(k), 3) for k in
                            ("occupied", "read_grad", "eff_grad", "read_abl",
                             "eff_abl", "spearman_grad_abl")
                            if any(k in h for h in hs)}
        out[fam] = per_layer
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", required=True,
                    help="Layers to read. From THIS checkpoint's profile.")
    ap.add_argument("--abl-layers", default="",
                    help="Subset of --layers where the causal ablation runs.")
    ap.add_argument("--abl-heads", type=int, default=16)
    ap.add_argument("--abl-dirs", type=int, default=16)
    ap.add_argument("--n-text", type=int, default=24)
    ap.add_argument("--n-kv", type=int, default=24)
    ap.add_argument("--prompt-max", type=int, default=192)
    ap.add_argument("--cont-len", type=int, default=16)
    ap.add_argument("--data", type=Path,
                    default=Path("training/corpus_open/g1i_warmup_v3_flat.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    abl_layers = [int(x) for x in args.abl_layers.split(",") if x.strip()]

    print(f"[lens] loading {args.model}", flush=True)
    loaded = load_rwkv7(args.model, device=args.device, dtype=torch.float32,
                        backend="peft", ctx_len=2048)
    for p in loaded.model.parameters():
        p.requires_grad_(False)          # the state is the only leaf

    items = build_prompts(loaded.tokenizer, args.n_text, args.n_kv, args.data,
                          args.seed, args.prompt_max, args.cont_len)
    print(f"[lens] {len(items)} prompts, layers {layers}, ablation on "
          f"{abl_layers or 'none'}", flush=True)

    results, t0 = [], time.time()
    for k, it in enumerate(items):
        results.append(run_prompt(loaded, it, layers, abl_layers,
                                  args.abl_heads, args.abl_dirs))
        el = time.time() - t0
        print(f"[lens] {k + 1}/{len(items)} {it['family']:<5} "
              f"{el:.0f}s elapsed, ~{el / (k + 1) * (len(items) - k - 1):.0f}s left",
              flush=True)
        if (k + 1) % 4 == 0 or k + 1 == len(items):
            # checkpoint the partial result so a killed night still leaves data
            partial = {"model": args.model, "layers": layers,
                       "abl_layers": abl_layers, "done": k + 1,
                       "summary": summarise(results, layers),
                       "per_prompt": results}
            args.out.with_suffix(".partial.json").write_text(json.dumps(partial))

    summ = summarise(results, layers)
    print("\n=== occupied vs read (mean per head) ===")
    for fam, pl in summ.items():
        print(f"--- {fam}")
        for L, v in pl.items():
            print(f"  L{L:<3} " + "  ".join(f"{k}={v[k]}" for k in v))

    from experiments._common.results import save_result
    save_result(args.out, {"model": args.model, "layers": layers,
                           "abl_layers": abl_layers, "cont_len": args.cont_len,
                           "summary": summ, "per_prompt": results,
                           "_summary": {
                               f"{fam} L{L}": (f"occ {v.get('occupied')} / "
                                               f"grad {v.get('read_grad')} / "
                                               f"abl {v.get('read_abl', '-')}")
                               for fam, pl in summ.items()
                               for L, v in pl.items()}},
                experiment="state_readout_lens", hypothesis=["H25", "H26"],
                model=args.model, script="experiments/rl/state_readout_lens.py")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
