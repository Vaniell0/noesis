# Training data shortlist

Canonical record of what went into each A1 pilot step and what is
queued for future steps. Update this file when the corpus changes.

---

## Step 6 — current (2026-08-06)

**File:** `training/tokenised/step6_mixed_train.pt` (3.8 GB)

| Source | Rollouts | Total tokens | Supervised tokens | Sup % | Format |
|--------|----------|-------------|-------------------|-------|--------|
| Claude action chains | 1 134 | ~80M | ~63M | 78% | Anthropic tool_use JSON |
| ToolBench G123-DFS | 183 751 | ~173M | ~23M | 13% | ReAct text → tool_use JSON |
| **Combined** | **184 885** | **~253M** | **~86M** | **34%** | — |

Actual counts from dataset patch at load time: 170 602 007 tokens,
22 090 740 supervised (12.9%) — delta vs above because 11 069 rollouts
truncated to ctx_len=2048.

**Loss mask:** tool_use turns only (role=assistant with tool_use).
Unsupervised: user, tool_result, thought context.

**Base model:** `rwkv7-g1h-2.9b-20260710-ctx10240.pth` (n_layer=32, n_embd=2560)

**Why:** First reflexive corpus (10-70 tool_use/session). Enables first
valid test of H2 and H10. glaive-v2 was reactive (1-2 tool_use/session)
and confounded both hypotheses.

**Known issues:**
- ToolBench (99.4% of rollouts) dilutes training time ~162× relative to
  action chains. In 24h VM window (~27% epoch) model sees ~294 action
  chain sessions.
- If next run needed: cap ToolBench ≤ 10 000 → 11 134 total rollouts,
  full epoch in ~5h.

---

## Steps 1–5 — glaive-v2 (RETIRED)

**File:** `training/tokenised/glaive_v2_train.pt` (181 MB)

| Source | Examples | Notes |
|--------|----------|-------|
| glaive-function-calling-v2 | ~61 000 | reactive: 1-2 tool_use/session |

**Verdict:** FAILED. All SFT variants scored below G1d base. glaive-v2
trains format-dispatch, not reasoning. 40% "direct" entries are
pre-tool-call phrases, not answers. See `FAILED.md` §2026-08-06 and
`docs/verdicts/2026-08-06-a1-pilot-step5.md`.

---

## Queued / candidates

| Dataset | Status | Notes |
|---------|--------|-------|
| ToolBench G123-DFS (capped 10K) | Ready to retokenize | `normalize_toolbench.py --limit 10000` |
| NuminaMath / OpenMath | Not downloaded | Direct-answer reasoning; would add non-tool signal for A0 eval |
| Open CoT→tool (o1-style) | Not identified | Future: if H2 needs more reasoning-before-action signal |
| noesis DSL corpus (synthetic) | Not created | Needed for DSL surface fine-tune before runtime deployment |

---

## Format note

Action chains use Anthropic tool_use JSON format (from Claude Code
sessions). noesis runtime uses its own DSL (`tool_call name { key=val }`).
These are different surfaces. A1 pilot trains the underlying reflexive
capability; DSL surface requires a separate fine-tune or in-context
adaptation via composer preamble.
