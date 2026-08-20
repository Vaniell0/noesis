## Architectural note — unified multimodal RWKV, not split backends
### *(locked 2026-07-23)*

**Decision.** noesis targets a *unified* multimodal backbone (one
model, one state format, text ⊕ image ⊕ (possibly) audio through
the same delta-rule update) rather than a split perception-backend
+ reasoning-backend architecture. If, at any point, adding a vision
head means introducing a second model with a serialised handoff
protocol between the two, the architectural drift needs to be
challenged before committing.

**Why.** Split backends carry real costs:
- **Coordination overhead** — two schedulers, two lifecycles, two
  memory footprints resident.
- **Format translation** — perception-side output has to be
  serialised into text (or a synthetic embedding format) that the
  reasoning-side model can consume; the serialisation itself is
  lossy and slow.
- **Latency stack-up** — inference on both models in sequence, plus
  the translation step, dominates any per-step wins from
  specialising each backend.
- **Frontier signal.** MyTHOS-line and OpenMythos work
  (Recurrent-Depth Transformer + MoE + memory-compression) is
  investing in *state-side* computation and multimodal-in-context,
  not in inter-model orchestration. If the frontier is unifying, a
  small research project should not be gluing.

**How this shapes near-term work.** H13a and H13b are the two probes
that inform the unified-substrate wager. H13b is the cheap
near-term test (does image-in-context yield with any vision-capable
substrate?); H13a is the deep wager (does the *RWKV* state itself
carry that yield without a bolted-on encoder?). Both are worth
running; neither justifies introducing a second local reasoning
model to service perception.

**Escape hatch.** If H13a fails clearly (state cannot absorb visual
tokens) *and* H13b holds (image-in-context yield is real, but only
via an external vision head), the escape hatch is a *fused*
architecture where the vision head produces tokens or embeddings
consumed inside the same forward pass of the reasoning backbone —
not a split-backends handoff protocol. This preserves the
single-cognitive-engine constraint from `CLAUDE.md`.

**Recorded from.** User push-back 2026-07-23 in response to a design
sketch that proposed a split perception/reasoning stack. Recorded to
prevent architectural drift over the next 3–6 months while H13
probes are pending.
