"""
Proper ROSA-4bit eval on A0 tasks with correct judge.

Loads the ROSA-4bitLM architecture (BlinkDL RWKV-v8/260222_rosa4bitLM_L12.py),
runs on A0 tasks, judges using rubric (exact/contains/regex), reports per-category.

Usage:
  python experiments/rosa_probe/run_rosa_eval.py \
      --model ~/.libs/models/rwkv7/rosa/rwkv-rosa4bit-minipile.pth \
      --tasks experiments/A0_eval/tasks.jsonl \
      --categories bit_decoding arithmetic \
      --n-tasks 16 --max-tokens 120 \
      --out /tmp/rosa_eval.json
"""

import argparse, json, re, time, os, sys
from pathlib import Path
import torch
import torch.nn as nn

# ── Tokenizer (same as run.py) ────────────────────────────────────────────────

class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        for l in open(file_name, encoding="utf-8"):
            idx = int(l[:l.index(" ")])
            x = eval(l[l.index(" "):l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            sorted_tokens.append(x)
            self.idx2token[idx] = x
        self.token2idx = {v: k for k, v in self.idx2token.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0] * 256
        for s in reversed(sorted_tokens):
            if len(s) >= 2:
                s0, s1 = s[0], s[1]
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encode(self, src: str) -> list:
        src = src.encode("utf-8")
        tokens, i = [], 0
        while i < len(src):
            s = src[i:i+1]
            if i < len(src) - 1:
                s0, s1 = src[i], src[i+1]
                if s1 in self.good[s0]:
                    sss = src[i:i+self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decode(self, tokens: list) -> str:
        return b"".join(self.idx2token[t] for t in tokens).decode("utf-8", errors="replace")


# ── ROSA-4bit model (from BlinkDL RWKV-v8/260222_rosa4bitLM_L12.py) ──────────

def _rosa_ref(q, k, v):
    n = len(q)
    idx = [0] * n
    ln  = [0] * n
    for i in range(n):
        found = False
        for w in range(i+1, 0, -1):
            t = q[i+1-w:i+1]
            for j in range(i-w, -1, -1):
                if k[j:j+w] == t:
                    idx[i] = v[j+w]
                    ln[i]  = w
                    found  = True
                    break
            if found:
                break
    return idx, ln


class _ROSA4bit(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.emb = nn.Parameter(torch.full((1, 1, C), 1.0))

    def forward(self, q, k, v):
        B, T, C = q.shape
        bits = 4
        G    = C // bits
        qb   = (q > 0).to(torch.uint8).cpu()
        kb   = (k > 0).to(torch.uint8).cpu()
        vb   = (v > 0).to(torch.uint8).cpu()
        ee   = self.emb.detach().cpu()
        out  = torch.zeros((B, T, C), dtype=q.dtype)
        for b in range(B):
            for g in range(G):
                qs, ks, vs = [0]*T, [0]*T, [0]*T
                for bb in range(bits):
                    ch = g*bits + bb
                    for t in range(T):
                        qs[t] |= int(qb[b,t,ch]) << bb
                        ks[t] |= int(kb[b,t,ch]) << bb
                        vs[t] |= int(vb[b,t,ch]) << bb
                idx, ln = _rosa_ref(qs, ks, vs)
                for t in range(T):
                    if ln[t] > 0:
                        sym = idx[t]
                        for bb in range(bits):
                            ch = g*bits + bb
                            sign = 1.0 if ((sym >> bb) & 1) else -1.0
                            out[b,t,ch] = sign * ee[0,0,ch].item()
        return out.to(q.device)


class _ROSAAttn(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.x_q = nn.Parameter(torch.zeros(1,1,C))
        self.x_k = nn.Parameter(torch.zeros(1,1,C))
        self.x_v = nn.Parameter(torch.zeros(1,1,C))
        self.q   = nn.Linear(C, C)
        self.k   = nn.Linear(C, C)
        self.v   = nn.Linear(C, C)
        self.rosa = _ROSA4bit(C)
        self.o   = nn.Linear(C, C)

    def forward(self, x):
        xx = self.time_shift(x) - x
        return self.o(self.rosa(self.q(x + xx*self.x_q),
                                self.k(x + xx*self.x_k),
                                self.v(x + xx*self.x_v)))


class _CMix(nn.Module):
    def __init__(self, C, ffn):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.x_k   = nn.Parameter(torch.empty(1,1,C))
        self.key   = nn.Linear(C, ffn, bias=False)
        self.value = nn.Linear(ffn, C, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = torch.relu(self.key(x + xx*self.x_k)) ** 2
        return self.value(k)


class _Block(nn.Module):
    def __init__(self, C, ffn, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.ln3 = nn.LayerNorm(C)
        self.rosa = _ROSAAttn(C)
        self.ffn  = _CMix(C, ffn)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.rosa(self.ln3(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ROSA4bitLM(nn.Module):
    def __init__(self, n_layer, n_embd, vocab_size):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([_Block(n_embd, n_embd*4, i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head   = nn.Linear(n_embd, vocab_size, bias=False)

    def logits(self, ids):
        x = self.emb(torch.tensor(ids, dtype=torch.long).reshape(1,-1))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_out(x))[0,-1]


def load_rosa(path, n_layer=12, n_embd=768, vocab_size=65536):
    params = torch.load(path, map_location="cpu", weights_only=False)
    # Auto-detect architecture from checkpoint keys
    layer_keys = [k for k in params if k.startswith("blocks.")]
    if layer_keys:
        n_layer = max(int(k.split(".")[1]) for k in layer_keys) + 1
    emb_shape = params.get("emb.weight", params.get("blocks.0.rosa.q.weight"))
    if "emb.weight" in params:
        n_embd = params["emb.weight"].shape[1]
        vocab_size = params["emb.weight"].shape[0]
    print(f"[rosa] auto-detected: n_layer={n_layer} n_embd={n_embd} vocab={vocab_size}")
    model = ROSA4bitLM(n_layer, n_embd, vocab_size).to(dtype=torch.float16)
    model.load_state_dict(params, strict=False)
    model.eval()
    return model


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge(response: str, task: dict) -> bool:
    rubric = task.get("rubric", {})
    rtype  = rubric.get("type", "")
    resp   = response.strip().lower()
    if rtype == "exact":
        return rubric.get("value", "").lower() in resp
    if rtype == "contains":
        val = rubric.get("value", "")
        return all(w.lower() in resp for w in val.split())
    if rtype == "regex":
        pattern = rubric.get("value", "")
        return bool(re.search(pattern, response, re.IGNORECASE))
    expected = str(task.get("expected", "")).strip().lower()
    return expected != "" and expected in resp


# ── Eval ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True)
    ap.add_argument("--tasks",      required=True)
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Filter categories (default: all)")
    ap.add_argument("--n-tasks",    type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--ctx-limit",  type=int, default=512)
    ap.add_argument("--out",        required=True)
    args = ap.parse_args()

    vocab_path = os.path.join(
        os.path.dirname(__import__("rwkv").__file__),
        "rwkv_vocab_v20230424.txt",
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

    # Auto-detect vocab size to pick the right tokenizer
    _probe = torch.load(args.model, map_location="cpu", weights_only=False)
    _vocab_size = _probe.get("emb.weight", torch.empty(65536, 0)).shape[0]
    del _probe

    print(f"[rosa_eval] loading tokenizer (vocab={_vocab_size}) …")
    if _vocab_size <= 50304:
        from transformers import AutoTokenizer as _AT
        _hf_tok = _AT.from_pretrained("EleutherAI/gpt-neox-20b", local_files_only=True)

        class _HFTokWrapper:
            def encode(self, text):
                return _hf_tok.encode(text)
            def decode(self, ids):
                return _hf_tok.decode(ids, skip_special_tokens=True)

        tok = _HFTokWrapper()
    else:
        tok = RWKV_TOKENIZER(vocab_path)

    print(f"[rosa_eval] loading model {args.model} …")
    t0 = time.time()
    model = load_rosa(args.model)
    print(f"[rosa_eval] loaded in {time.time()-t0:.1f}s")

    # Load tasks
    all_tasks = [json.loads(l) for l in open(args.tasks)]
    if args.categories:
        all_tasks = [t for t in all_tasks if t.get("category") in args.categories]
    tasks = all_tasks[:args.n_tasks]
    print(f"[rosa_eval] {len(tasks)} tasks (categories: {args.categories or 'all'})\n")

    results, by_cat = [], {}
    for task in tasks:
        t0 = time.time()
        ids = tok.encode(task["prompt"])[-args.ctx_limit:]
        orig_len = len(ids)
        with torch.no_grad():
            for _ in range(args.max_tokens):
                lg  = model.logits(ids[-args.ctx_limit:])
                nxt = int(torch.argmax(lg).item())
                if nxt == 0:
                    break
                ids.append(nxt)
        response = tok.decode(ids[orig_len:])
        elapsed  = time.time() - t0
        correct  = judge(response, task)
        cat      = task.get("category", "?")
        by_cat.setdefault(cat, []).append(correct)
        results.append({
            "id": task["id"], "category": cat, "correct": correct,
            "response": response[:300], "elapsed": round(elapsed, 1),
            "rubric": task.get("rubric"),
        })
        mark = "✓" if correct else "✗"
        print(f"  {mark} [{cat}] {task['id']} ({elapsed:.0f}s)")
        print(f"    rubric:   {task.get('rubric')}")
        print(f"    response: {response[:100]!r}\n")

    n_correct = sum(r["correct"] for r in results)
    print(f"\n=== Results ===")
    print(f"Overall: {n_correct}/{len(results)} = {100*n_correct//max(len(results),1)}%")
    for cat, vals in sorted(by_cat.items()):
        print(f"  {cat}: {sum(vals)}/{len(vals)} = {100*sum(vals)//len(vals)}%")

    summary = {
        "model": args.model, "n": len(results), "correct": n_correct,
        "accuracy": round(n_correct/max(len(results),1), 3),
        "by_category": {c: {"correct": sum(v), "total": len(v)} for c, v in by_cat.items()},
        "results": results,
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
