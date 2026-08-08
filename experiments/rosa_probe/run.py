"""
RWKV-8 ROSA-4bit vs RWKV-7 G1h: spot-check on A0 eval tasks.

Runs a small subset of A0 tasks (symbolic + arithmetic_chain, ≤ 5 each)
to qualitatively compare ROSA pattern-matching vs RWKV-7 WKV reasoning.

WARNING: ROSA forward is pure-Python O(T²) — expect ~60s per task.
Run with a small --n-tasks budget.

Usage:
  python experiments/rosa_probe/run.py \
      --model ~/.libs/models/rwkv7/rosa/rwkv-rosa4bit-minipile.pth \
      --vocab /path/to/rwkv_vocab_v20230424.txt \
      --tasks experiments/A0_eval/tasks.jsonl \
      --n-tasks 6 \
      --out /tmp/rosa_probe.json
"""

import argparse, json, time, types, math
import torch, torch.nn as nn
from torch.nn import functional as F


# ── Tokenizer ────────────────────────────────────────────────────────────────

class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        for l in open(file_name, "r", encoding="utf-8").readlines():
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
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
            s = src[i : i + 1]
            if i < len(src) - 1:
                s0, s1 = src[i], src[i + 1]
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decode(self, tokens: list) -> str:
        return b"".join(self.idx2token[t] for t in tokens).decode("utf-8", errors="replace")


# ── ROSA model (from BlinkDL/RWKV-LM/RWKV-v8/260222_rosa4bitLM_L12.py) ─────

def _rosa_ref(q, k, v):
    n = len(q)
    idx = [0] * n
    ln = [0] * n
    for i in range(n):
        found = False
        for w in range(i + 1, 0, -1):
            t = q[i + 1 - w : i + 1]
            for j in range(i - w, -1, -1):
                if k[j : j + w] == t:
                    idx[i] = v[j + w]
                    ln[i] = w
                    found = True
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
        G = C // bits
        qb = (q > 0).to(torch.uint8).cpu()
        kb = (k > 0).to(torch.uint8).cpu()
        vb = (v > 0).to(torch.uint8).cpu()
        ee = self.emb.detach().cpu()
        out = torch.zeros((B, T, C), dtype=q.dtype)
        for b in range(B):
            for g in range(G):
                qs, ks, vs = [0] * T, [0] * T, [0] * T
                for bb in range(bits):
                    ch = g * bits + bb
                    for t in range(T):
                        qs[t] |= int(qb[b, t, ch]) << bb
                        ks[t] |= int(kb[b, t, ch]) << bb
                        vs[t] |= int(vb[b, t, ch]) << bb
                idx, ln = _rosa_ref(qs, ks, vs)
                for t in range(T):
                    if ln[t] > 0:
                        sym = idx[t]
                        for bb in range(bits):
                            ch = g * bits + bb
                            sign = 1.0 if ((sym >> bb) & 1) else -1.0
                            out[b, t, ch] = sign * ee[0, 0, ch].item()
        return out.to(q.device)


class _ROSAAttn(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.q = nn.Linear(C, C)
        self.k = nn.Linear(C, C)
        self.v = nn.Linear(C, C)
        self.rosa = _ROSA4bit(C)
        self.o = nn.Linear(C, C)

    def forward(self, x):
        xx = self.time_shift(x) - x
        return self.o(self.rosa(self.q(x + xx * self.x_q),
                                self.k(x + xx * self.x_k),
                                self.v(x + xx * self.x_v)))


class _CMix(nn.Module):
    def __init__(self, C, ffn):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = nn.Linear(C, ffn, bias=False)
        self.value = nn.Linear(ffn, C, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = torch.relu(self.key(x + xx * self.x_k)) ** 2
        return self.value(k)


class _Block(nn.Module):
    def __init__(self, C, ffn, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.ln3 = nn.LayerNorm(C)
        self.rosa = _ROSAAttn(C)
        self.ffn = _CMix(C, ffn)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.rosa(self.ln3(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ROSA4bitLM(nn.Module):
    def __init__(self, n_layer, n_embd, vocab_size):
        super().__init__()
        ffn = n_embd * 4
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([_Block(n_embd, ffn, i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, ids):  # ids: list[int] or 1-D tensor
        x = self.emb(torch.tensor(ids, dtype=torch.long).reshape(1, -1))
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)

    def logits(self, ids):
        return self.head(self.forward(ids))[0, -1]


def load_rosa(model_path: str, n_layer=12, n_embd=768, vocab_size=65536, dtype=torch.float16):
    params = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ROSA4bitLM(n_layer, n_embd, vocab_size).to(dtype=dtype)
    model.load_state_dict(params, strict=False)
    model.eval()
    return model


def generate(model, tokenizer, prompt: str, max_tokens=128, ctx_limit=512) -> str:
    ids = tokenizer.encode(prompt)[-ctx_limit:]
    with torch.no_grad():
        for _ in range(max_tokens):
            lg = model.logits(ids[-ctx_limit:])
            nxt = int(torch.argmax(lg).item())
            if nxt == 0:
                break
            ids.append(nxt)
    return tokenizer.decode(ids[len(tokenizer.encode(prompt)[-ctx_limit:]):])


# ── Eval ──────────────────────────────────────────────────────────────────────

def load_tasks(path, n, categories=("symbolic", "arithmetic_chain")):
    tasks = []
    for line in open(path):
        t = json.loads(line)
        if t.get("category") in categories:
            tasks.append(t)
        if len(tasks) >= n:
            break
    return tasks


def judge(response: str, task: dict) -> bool:
    rubric = task.get("rubric", {})
    rtype = rubric.get("type", "")
    expected = str(task.get("expected", "")).strip()
    resp = response.strip()
    if rtype == "exact":
        return expected.lower() in resp.lower()
    if rtype == "contains":
        return all(k.lower() in resp.lower() for k in rubric.get("keywords", [expected]))
    return expected.lower() in resp.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[rosa_probe] loading tokenizer …")
    tok = RWKV_TOKENIZER(args.vocab)

    print(f"[rosa_probe] loading model {args.model} …")
    t0 = time.time()
    model = load_rosa(args.model)
    print(f"[rosa_probe] model loaded in {time.time()-t0:.1f}s")

    tasks = load_tasks(args.tasks, args.n_tasks)
    print(f"[rosa_probe] running {len(tasks)} tasks (pure-Python ROSA — slow) …\n")

    results = []
    for task in tasks:
        t0 = time.time()
        resp = generate(model, tok, task["prompt"], max_tokens=args.max_tokens)
        elapsed = time.time() - t0
        correct = judge(resp, task)
        results.append({
            "id": task["id"],
            "category": task["category"],
            "correct": correct,
            "response": resp[:400],
            "expected": task.get("expected", ""),
            "elapsed": round(elapsed, 1),
        })
        mark = "✓" if correct else "✗"
        print(f"  {mark} [{task['category']}] {task['id']} ({elapsed:.0f}s)")
        print(f"    expected: {task.get('expected','')!r}")
        print(f"    got:      {resp[:120]!r}\n")

    n_correct = sum(r["correct"] for r in results)
    summary = {
        "model": args.model,
        "n": len(results),
        "correct": n_correct,
        "accuracy": round(n_correct / len(results), 3) if results else 0,
        "results": results,
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"[rosa_probe] {n_correct}/{len(results)} correct → {args.out}")


if __name__ == "__main__":
    main()
