# Engaging with noesis

**noesis is one person's daily bot, not an open-source product taking
feature requests.** That is worth stating up front so nobody spends time
in the wrong direction. That said, several kinds of engagement are
genuinely useful.

## The most useful thing you can do

**Try to falsify a hypothesis.** `hypotheses/README.md` lists every specific
wager (H1..H17) with a stated prediction and a falsifier. A clean
falsification on your own setup — "H7 predicts X, I ran the experiment
with these parameters and observed Y" — is more valuable than any
feature. `FAILED.md` is the graveyard where refuted claims land, and it
is meant to grow.

Push-back on the *framing* of a hypothesis is also welcome. Pair it with
an empirical test proposal — no RNN-vs-Transformer mysticism.

## Non-negotiable rules

Read `README.md` §Hard constraints and `docs/principles.md` first. The
load-bearing locks:

- **Single reasoning model.** noesis has one reasoning backbone
  (RWKV-7-G1). Small utility NNs (embedders, classifiers, routers) are
  welcome where they earn their keep. Additional local reasoning models
  are not. Heuristic: *if it emits tokens that participate in a chain of
  thought, it is a reasoning model.*
- **No personal data in weights.** Open sources only for fine-tune.
  Personal corpus is a runtime retrieval channel. Narrow carve-out:
  persona/style SFT (§H15).
- **Not a Transformer.** RWKV chosen for constant-cost streaming
  inference. A switch requires empirical re-open, not architectural
  drift.
- **Cheap by construction.** Laptop i5-1235U for inference (CPU-only).
  Cloud burst for training allowed but explicit.

A PR pushing against any of these will bounce unless it carries
empirical evidence, not a preference argument.

## What kinds of PRs actually help

- **A failed replication of a hypothesis.** Exact configuration and
  observed metric. Bonus for a reproducer script.
- **An extension spec or prototype for your own noesis.** If you build a
  browser extension, IDE extension, or Minecraft-bot against the surface
  in `docs/extensions.md`, sharing the manifest schema and event layout
  helps stabilise the interface for the next builder.
- **Bug reports against the runtime skeleton** (`runtime/`) — collectors,
  retention, HTTP shim. Reproducer preferred.
- **Corrections to design docs where they contradict the code.** Spot a
  stale claim, file it.

## What kinds of PRs don't

- **Features to make noesis useful for you.** noesis is one person's
  bot; your fork is where those live.
- **Refactors chasing prettier code** without a stated regression risk.
- **New reasoning-model integrations** (see single-reasoning-model lock).

## PR conventions

- One logical change per PR.
- Include: which hypothesis it relates to (if any), what changed, what
  would falsify.
- English for code, commits, docs. Russian OK in discussion.
- No `--no-verify` on commits.
