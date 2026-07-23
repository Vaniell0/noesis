# Contributing to noesis

Before contributing, read:
- `CLAUDE.md` — locked decisions and hard constraints.
- `HYPOTHESES.md` — the falsifiable claims noesis is testing.
- `docs/principles.md` — the twelve architecture principles.

## Non-negotiable rules

**Single reasoning model.** noesis has one reasoning backbone
(RWKV-7-G1). Additional local reasoning models will be rejected.
Small utility NNs (embedders, classifiers, routers, tool-call
formatters) are welcome where they earn their keep — the rule
(CLAUDE.md verbatim): *"if it emits tokens that participate in a
chain of thought, it is a reasoning model."*

**No personal data in weights.** Open sources only for fine-tune.
Personal corpus is a runtime retrieval channel, never a training
signal. Narrow carve-out: persona / style SFT (§H15) only.

**Not a Transformer.** RWKV chosen deliberately for constant-cost
streaming inference. Any switch requires empirical re-open, not
architectural drift.

**Cheap by construction.** Assume GTX 1050 for inference and small
LoRA. Cloud burst allowed but must be explicit.

## Interaction style

- **Punch-list style for audits.** Propose changes as a list, wait
  for review, no stealth-fix.
- **Any philosophical claim about RNN vs Transformer must be paired
  with an empirical test proposal.** No architectural mysticism.
- **Russian OK for discussion, English for code / commits / docs.**

## Where to start

- **Hypotheses to test.** Anything in `HYPOTHESES.md` marked
  *Status: Untested* is fair game. Each has a stated prediction and
  falsifier — the falsifier is the contract.
- **Bugs / open questions.** GitHub Issues, or the open questions
  section in `docs/policies.md`.
- **Extension development.** `docs/extensions.md` — extension
  surface is Phase-2 docs-only right now; contributions to the spec
  (manifest schema, first-example embodiment) welcome.
- **Runtime.** `runtime/` — collectors, retention, HTTP shim. Phase-B
  skeleton validated 2026-07-22 (`docs/verdicts/`). Next blockers:
  composer, tool-dispatcher.

## Pull request conventions

- One logical change per PR.
- Include: which hypothesis it relates to (if any), what changed,
  what test would falsify.
- No `--no-verify` on commits.
- Signed-off commits preferred but not required.

## What noesis is NOT

- Not a Claude replacement. Heavy reasoning goes to remote Claude
  by user's explicit call.
- Not coupled to Compilerium. Runtime retrieval OK; fine-tune
  signal never.
- Not a SaaS. noesis is a personal daily bot, not a product.

If your PR pushes against any of these, expect push-back — bring an
empirical argument, not a preference.
