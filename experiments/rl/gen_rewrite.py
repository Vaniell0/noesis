#!/usr/bin/env python3
"""Derivation items for noesis: premises as rewrite rules, applied to random start terms.

    gen_rewrite.py <laws7.json> <rsc_laws> <out.jsonl> [--train N] [--val N] [--seed S]

Each item: eight premises (equalities that hold on the 23 spaces of derivations.jsonl, under local ids 1..8,
shuffled per item), a start term t0 and a question t0 = t. True items: t is reached from t0 by a chain of
rewrites with the premises (either direction, at any subterm, the premise's letters bound by matching); the
depth is the shortest chain over the same eight premises, found by breadth-first search. False items: t is a
near miss -- one atom or one argument order changed -- that the oracle refutes. Every question, true or false,
goes through `rsc_laws check` on the 23 spaces: a true one must hold wherever defined, a false one must fail
on some tuple where defined. Splits: a tenth of the laws are held out; an item using one as a premise is in
`val_laws`, never in `train`.
"""
import collections, json, random, subprocess, sys

LAWS, BIN, OUT = sys.argv[1:4]
def opt(name, dflt):
    return type(dflt)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else dflt
N_TRAIN, N_VAL, SEED = opt("--train", 40000), opt("--val", 2000), opt("--seed", 1)
MAXD, NPREM, MAXSIZE, BFS_CAP = 8, 8, 17, 40000

ARITY = {"log": 2, "exp": 2, "d": 2, "g": 3, "scale": 2, "neg": 1, "vadd": 2, "add": 2, "mul": 2}
POINTS, VECTORS, CONSTS = ("p", "q", "r"), ("u", "w"), ("0", "1/2", "2", "-1")
LETTERS = POINTS + VECTORS

def parse(s):
    toks = s.replace("(", " ( ").replace(")", " ) ").split()
    def go(i):
        if toks[i] != "(": return toks[i], i + 1
        op, i, args = toks[i + 1], i + 2, []
        while toks[i] != ")":
            a, i = go(i); args.append(a)
        return (op, *args), i + 1
    return go(0)[0]

def sexp(t):
    return t if isinstance(t, str) else "(" + " ".join([t[0]] + [sexp(a) for a in t[1:]]) + ")"

def polish(t):
    return [t] if isinstance(t, str) else [t[0]] + [x for a in t[1:] for x in polish(a)]

def size(t):
    return 1 if isinstance(t, str) else 1 + sum(size(a) for a in t[1:])

def sort(t):
    if isinstance(t, str): return "P" if t in POINTS else "V" if t in VECTORS else "R"
    return {"log": "V", "exp": "P", "d": "R", "g": "R", "scale": "V", "neg": "V", "vadd": "V", "add": "R", "mul": "R"}[t[0]]

def base(t):
    """The point a vector term is tangent at."""
    if isinstance(t, str): return "p"
    if t[0] == "log": return t[1]
    if t[0] == "scale": return base(t[2])
    return base(t[1])

def letters(t):
    return {t} if isinstance(t, str) and t in LETTERS else set() if isinstance(t, str) else set().union(*map(letters, t[1:]))

def match(pat, t, env):
    if isinstance(pat, str):
        if pat in LETTERS:
            if sort(t) != sort(pat): return None
            if pat in env: return env if env[pat] == t else None
            return {**env, pat: t}
        return env if pat == t else None
    if isinstance(t, str) or t[0] != pat[0] or len(t) != len(pat): return None
    for a, b in zip(pat[1:], t[1:]):
        env = match(a, b, env)
        if env is None: return None
    return env

def tangent_ok(env):
    """u and w are tangent at p: bound vectors must sit at the bound p (or at one common point)."""
    at = {base(env[v]) for v in VECTORS if v in env}
    if "p" in env: at.add(env["p"])
    return len(at) <= 1

def subst(t, env):
    return env.get(t, t) if isinstance(t, str) else (t[0], *[subst(a, env) for a in t[1:]])

def positions(t, path=()):
    """Subterms in preorder, each with its path; the preorder index is the hop's position."""
    yield path, t
    if not isinstance(t, str):
        for i, a in enumerate(t[1:], 1): yield from positions(a, path + (i,))

def put(t, path, new):
    if not path: return new
    i = path[0]
    return t[:i] + (put(t[i], path[1:], new),) + t[i + 1:]

_directed = {}
def directed(rules):
    """The usable directions of the rules, by the head of the side matched: (head, rule index, direction, a, b)."""
    key = tuple(rules)
    if key not in _directed:
        by = collections.defaultdict(list)
        for ri, (l, r) in enumerate(rules):
            for dr, (a, b) in enumerate(((l, r), (r, l))):
                if letters(b) <= letters(a):
                    by[sort(a) if isinstance(a, str) and a in LETTERS else a if isinstance(a, str) else a[0]].append((ri, dr, a, b))
        if len(_directed) > 5000: _directed.clear()
        _directed[key] = by
    return _directed[key]

def steps(t, rules):
    """Every one-step rewrite of t: (rule index, direction, preorder position, result)."""
    by = directed(rules)
    for pos, (path, sub) in enumerate(positions(t)):
        head = sub if isinstance(sub, str) else sub[0]
        for ri, dr, a, b in by.get(head, []) + by.get(sort(sub), []):
            env = match(a, sub, {})
            if env is None or not tangent_ok(env): continue
            out = put(t, path, subst(b, env))
            if out != t and size(out) <= MAXSIZE: yield ri, dr, pos, out

def shortest(t0, rules, goal):
    """Breadth-first over the item's own premises: the shortest chain from t0 to goal, as hops."""
    prev, frontier = {t0: None}, [t0]
    for _ in range(MAXD):
        nxt = []
        for t in frontier:
            for ri, dr, pos, out in steps(t, rules):
                if out in prev: continue
                prev[out] = (t, ri, dr, pos)
                if out == goal:
                    hops = []
                    while prev[out]: s, ri_, dr_, pos_ = prev[out]; hops.append((ri_, dr_, pos_, out)); out = s
                    return hops[::-1]
                nxt.append(out)
                if len(prev) > BFS_CAP: return None
        frontier = nxt
    return None

def mutate(t, rng):
    """A near miss: one letter swapped for another of its sort, or the arguments of d or log swapped."""
    cands = []
    for path, sub in positions(t):
        if isinstance(sub, str) and sub in LETTERS:
            for o in (POINTS if sub in POINTS else VECTORS):
                if o != sub: cands.append(put(t, path, o))
        elif not isinstance(sub, str) and sub[0] in ("log", "d", "g") and sub[1] != sub[2] and sub[0] != "g":
            cands.append(put(t, path, (sub[0], sub[2], sub[1])))
    rng.shuffle(cands)
    return cands

def family(rules):
    ops = collections.Counter(x for l, r in rules for x in polish(l) + polish(r) if x in ARITY)
    if ops["exp"] + ops["log"] >= ops["d"] + ops["g"] and ops["exp"] + ops["log"] > 0: return "exp-log"
    if ops["d"] + ops["g"] > 0: return "metric"
    return "tangent-algebra"

def main():
    rng = random.Random(SEED)
    J = json.load(open(LAWS))
    spaces = J["spaces"]
    use = [s for s in spaces if s not in ("Torus-chart", "Sphere-implicit")]
    def holds(l):
        return l["rel"] == "=" and (l["hold"] == "all" or all(l["hold"][i] == "1" for i, s in enumerate(spaces) if s in use))
    laws = [(parse(l["lhs_form"]), parse(l["rhs_form"])) for l in J["laws"] if holds(l)]
    order = list(range(len(laws))); random.Random(0).shuffle(order)   # the same held-out laws in every shard
    held = set(order[: len(laws) // 10])
    starts = [s for l in laws for s in l if not isinstance(s, str)]

    def walk(depth, pool_idx):
        """A random chain of `depth` rewrites from a random start, with rules from the pool."""
        t0 = rng.choice(starts)
        if rng.random() < 0.5:   # room to work: the start inside a larger context of its sort
            other = rng.choice([s for s in starts if sort(s) == sort(t0)] or [t0])
            t0 = {"R": ("add", t0, other) if sort(other) == "R" else t0, "P": ("d", t0, rng.choice(POINTS)),
                  "V": ("g", base(t0), t0, t0)}[sort(t0)]
            if size(t0) > MAXSIZE: return None
        t, used = t0, []
        for _ in range(depth):
            opts = [(pool_idx[ri], out) for ri, _, _, out in steps(t, [laws[i] for i in pool_idx]) if out != t0]
            if not opts: return None
            ri, t = rng.choice(opts); used.append(ri)
            if len(set(used)) > NPREM: return None
        return t0, t, used

    def distractors(used, k, pool_idx):
        """Premises sharing the most symbols with the chain's, so symbol overlap does not point at the rule."""
        sym = collections.Counter(x for ri in used for s in laws[ri] for x in polish(s))
        cand = [i for i in pool_idx if i not in used]
        cand.sort(key=lambda i: -sum(sym[x] for s in laws[i] for x in polish(s)) + rng.random() * 6)
        return cand[:k]

    want = {"train": N_TRAIN, "val": N_VAL, "val_laws": N_VAL}
    per_depth = {sp: collections.Counter() for sp in want}
    items, pending = [], []
    open(OUT, "w").close()
    train_pool = [i for i in range(len(laws)) if i not in held]
    all_pool = list(range(len(laws)))
    tries = 0
    while any(sum(per_depth[sp].values()) < n for sp, n in want.items()):
        tries += 1
        split = rng.choice([sp for sp, n in want.items() if sum(per_depth[sp].values()) < n])
        pool = train_pool if split != "val_laws" else all_pool
        d = rng.choice([x for x in range(1, MAXD + 1) if per_depth[split][x] < want[split] / MAXD] or [1])
        w = walk(d, pool)
        if not w: continue
        t0, t, used = w
        if split == "val_laws" and not (set(used) & held): continue
        uniq = list(dict.fromkeys(used))
        prem = uniq + distractors(uniq, NPREM - len(uniq), pool)
        if split == "val_laws" and not any(i in held for i in prem): continue
        rng.shuffle(prem)
        rules = [laws[i] for i in prem]
        hops = shortest(t0, rules, t)
        if not hops or len(hops) != d: continue   # the walk took a long way round: not this depth
        truth = rng.random() < 0.5
        q = t if truth else next(iter(mutate(t, rng)), None)
        if q is None: continue
        pending.append(dict(split=split, depth=d, t0=t0, q=q, truth=truth, prem=prem, hops=hops, rules=rules))
        per_depth[split][d] += 1
        if len(pending) >= 500 or not any(sum(per_depth[sp].values()) < n for sp, n in want.items()):
            kept = settle(pending, use, per_depth)
            write(kept, len(items))   # each confirmed batch on disk at once: a hang loses one batch
            items += kept
            pending = []
            print(f"tries {tries} items {len(items)} " + " ".join(f"{sp}:{sum(c.values())}" for sp, c in per_depth.items()),
                  file=sys.stderr, flush=True)

def settle(batch, use, per_depth):
    """The oracle's verdict on every question; an item it does not confirm is dropped and its slot reopened."""
    lines = [f"{i}\t{sexp(it['t0'])}\t=\t{sexp(it['q'])}" for i, it in enumerate(batch)]
    out = subprocess.run([BIN, "check", *use], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    kept = []
    for line in out.splitlines():
        i, hold, dfn = line.split("\t")
        it = batch[int(i)]
        defined = [k for k, c in enumerate(dfn) if c == "1"]
        good = defined and (all(hold[k] == "1" for k in defined) if it["truth"] else any(hold[k] == "0" for k in defined))
        if good: kept.append(it)
        else: per_depth[it["split"]][it["depth"]] -= 1
    return kept

def write(items, first):
    with open(OUT, "a") as f:
        for n, it in enumerate(items, first):
            toks = []
            for k, (l, r) in enumerate(it["rules"], 1): toks += [f"#{k}", *polish(l), "=", *polish(r)]
            toks += ["?", *polish(it["t0"]), "=", *polish(it["q"])]
            hops = [{"rule": ri + 1, "dir": "lr" if dr == 0 else "rl", "pos": pos, "term": sexp(t)} for ri, dr, pos, t in it["hops"]]
            f.write(json.dumps({"id": n, "split": it["split"], "family": family(it["rules"]), "depth": it["depth"],
                                "tokens": toks, "answer": "true" if it["truth"] else "false",
                                "last_rule": hops[-1]["rule"], "premises": [f"{sexp(l)} = {sexp(r)}" for l, r in it["rules"]],
                                "question": f"{sexp(it['t0'])} = {sexp(it['q'])}", "hops": hops}) + "\n")

if __name__ == "__main__":
    main()
