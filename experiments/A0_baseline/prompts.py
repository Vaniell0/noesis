"""Prompt bank shared across A0 experiments.

Prompts are shaped like real noesis workloads, not toy Q&A:

- short     — interactive query the user might type into the CLI.
- medium    — background event-stream digest (last ~15 min of activity).
- long      — retrieval-augmented question with concrete document context.
- narrative — descriptive prose of similar length to `medium`, with no
              task, no question, no reasoning demand. Used by A0.4 as a
              non-reasoning control against `medium` when measuring
              state dynamics; not used by A0.1 throughput bench.

Word counts are given for reference; RWKV-7-World tokenizer is roughly
1.4-1.6 tok/word for English technical prose, so tokens are ~1.5× words.
"""

from __future__ import annotations

# ~40 words → ~60 tokens
SHORT = (
    "I ran `git bisect` on the `rewrite/proto-v2` branch and it landed on "
    "commit 9a3f21c but the failing test is `test_envelope_roundtrip`. "
    "Given the change, what is the most likely root cause and which file "
    "should I look at first?"
)


# ~350 words → ~520 tokens
MEDIUM = (
    "Summarise the last fifteen minutes of the user's activity from the "
    "event stream below and produce a one-paragraph digest oriented toward "
    "an on-call engineer. Highlight any potentially destructive operations, "
    "unusual latency, or events that suggest an incident is developing. "
    "Do not repeat the stream verbatim.\n\n"
    "EVENT STREAM (utc ISO-8601, actor, kind, payload):\n"
    "13:04:12  vaniello  window.focus  {app: 'firefox', title: 'Grafana — API latency'}\n"
    "13:04:38  vaniello  window.focus  {app: 'kitty', title: 'zsh ~/Desktop/projects/goodnet'}\n"
    "13:04:47  vaniello  shell.exec    {cmd: 'git status', rc: 0, dur_ms: 42}\n"
    "13:05:01  vaniello  shell.exec    {cmd: 'git checkout dev', rc: 0, dur_ms: 118}\n"
    "13:05:03  vaniello  shell.exec    {cmd: 'git pull --ff-only', rc: 0, dur_ms: 610}\n"
    "13:05:14  vaniello  shell.exec    {cmd: 'nix build .#test', rc: 0, dur_ms: 92117}\n"
    "13:06:47  cron      job.tick      {name: 'noesis-summary', last_ok: '12:52:12'}\n"
    "13:07:02  vaniello  shell.exec    {cmd: 'ctest -R kernel.', rc: 1, dur_ms: 47201}\n"
    "13:07:50  vaniello  file.edit     {path: 'src/kernel/protocol_registry.cpp', diff_lines: 34}\n"
    "13:08:14  vaniello  shell.exec    {cmd: 'nix build .#test', rc: 0, dur_ms: 88940}\n"
    "13:09:44  vaniello  shell.exec    {cmd: 'ctest -R kernel.', rc: 0, dur_ms: 46320}\n"
    "13:10:02  monitor   alert.fire    {board: 'api-latency', p99_ms: 812, threshold_ms: 400}\n"
    "13:10:18  vaniello  window.focus  {app: 'firefox', title: 'Grafana — API latency'}\n"
    "13:10:44  vaniello  shell.exec    {cmd: 'kubectl top pods -n api', rc: 0, dur_ms: 411}\n"
    "13:11:05  vaniello  shell.exec    {cmd: 'kubectl logs -n api api-6f47 --tail 200', rc: 0, dur_ms: 260}\n"
    "13:11:57  vaniello  file.edit     {path: 'ops/incident/2026-07-21.md', diff_lines: 12}\n"
    "13:13:03  monitor   alert.clear   {board: 'api-latency', p99_ms: 291}\n"
    "13:13:39  vaniello  shell.exec    {cmd: 'git commit -m fix: cap ...', rc: 0, dur_ms: 88}\n"
    "13:14:10  vaniello  shell.exec    {cmd: 'git push origin dev', rc: 0, dur_ms: 1140}\n"
    "13:14:44  vaniello  shell.exec    {cmd: 'rm -rf ~/tmp/api-diag.old', rc: 0, dur_ms: 210}\n"
    "13:15:02  vaniello  window.focus  {app: 'obsidian', title: 'Dev/GoodNet'}\n"
    "13:16:00  vaniello  file.edit     {path: 'Dev/GoodNet.md', diff_lines: 18}\n\n"
    "Constraint: one paragraph, no bullets, ≤120 words."
)


# ~1400 words → ~2050 tokens
LONG = (
    "You are the noesis background agent. Using ONLY the retrieved context "
    "block below, answer the user's question. If the context is insufficient, "
    "say so plainly and describe the specific gap. Do not fabricate.\n\n"
    "USER QUESTION:\n"
    "How does the GoodNet kernel decide which protocol layer processes an "
    "inbound envelope, and what changes did the 2026-05-08 protocol-relax "
    "merge introduce? Explain in terms an engineer new to the project can "
    "act on.\n\n"
    "RETRIEVED CONTEXT (source: Dev/GoodNet.md, docs/contracts/protocol-layer.md):\n"
    "GoodNet is a networking subsystem with pluggable transports, security "
    "providers, protocol layers, and handlers. The analogy is Linux kernel "
    "with namespaces and eBPF and kernel modules. The kernel process is "
    "agnostic; it owns a ConnectionRegistry (sharded), a HandlerRegistry "
    "keyed on (namespace, protocol_id, msg_id) which resolves to a priority "
    "chain of handlers, a LinkRegistry mapping URI scheme to link plugin, a "
    "ProtocolLayerRegistry mapping protocol_id to an IProtocolLayer, a "
    "SecurityRegistry pointing at the active provider, an ExtensionRegistry "
    "of named vtables with version, plus TimerRegistry, SignalChannel, "
    "MetricsRegistry, ServiceResolver and a PluginManager that loads plugins "
    "via dlopen and verifies a manifest SHA-256.\n\n"
    "There are nine loadable plugins, each in its own git repo under the "
    "goodnet-io GitHub organisation: handler-heartbeat, link-tcp, link-udp, "
    "link-ws, link-ipc, link-tls, link-ice, security-noise and security-null. "
    "Two protocol plugins are statically linked into the kernel and cannot "
    "be replaced at runtime: gnet-v1 which supplies mandatory mesh framing "
    "and raw-v1 which is a loopback and intranode passthrough with no "
    "framing overhead. Every plugin declares its meta through the C ABI "
    "surface host_api_t defined in sdk/host_api.h.\n\n"
    "Before 2026-05-08 the kernel held a single active IProtocolLayer slot. "
    "That design forced a global choice: either the mesh envelope was used "
    "everywhere or a single alternative had to replace it globally. This did "
    "not compose with the intended plugin model, where different transports "
    "might want different framings. The 2026-05-08 relax merge (commit "
    "4824ffc) replaced the singleton with a ProtocolLayerRegistry indexed by "
    "protocol_id. Per-link declaration is done via a new field "
    "gn_register_meta_t::protocol_id, which each link plugin fills in during "
    "its plugin_register callback. The kernel routes an inbound envelope by "
    "reading the frame's protocol_id header field and looking up the matching "
    "IProtocolLayer implementation in the registry. Cross-protocol envelopes "
    "are isolated: a handler chain registered against protocol_id X never "
    "sees traffic from protocol_id Y even if it also matches the namespace "
    "and msg_id.\n\n"
    "The relax also opened a path to SSH-as-protocol-layer, which is a "
    "modern-only implementation using ed25519 for identity, curve25519 for "
    "key exchange, and chacha20-poly1305 for authenticated encryption. It "
    "was not shipped in the same merge but is now unblocked as a separate "
    "plugin because the kernel no longer imposes a single protocol layer.\n\n"
    "Handler chains resolve as follows. When a plugin registers a handler, "
    "it supplies a gn_register_meta_t containing namespace_id, protocol_id, "
    "msg_id, and priority. The HandlerRegistry uses (namespace, protocol_id, "
    "msg_id) as the key and stores a priority-ordered vector of handler "
    "function pointers plus their owning plugin identity. On envelope "
    "arrival the kernel: (1) reads protocol_id from the envelope, (2) looks "
    "up the IProtocolLayer via ProtocolLayerRegistry, (3) invokes the "
    "layer's decode() to obtain namespace_id and msg_id, (4) resolves the "
    "handler chain from HandlerRegistry, (5) invokes handlers in priority "
    "order until one signals STOP or the chain is exhausted. Namespaces were "
    "introduced in the 2026-05-09 slice-1 merge (commit d6c2efd) and enable "
    "operator-driven tenant teardown via Kernel::drain_namespace(ns, "
    "deadline), which prevents new envelopes from entering the namespace "
    "and awaits in-flight ones with a bounded deadline before returning."
)


# ~280 words → ~420 tokens; matched length to MEDIUM for the A0.4 control.
# Descriptive prose only. No question, no instruction, no explicit demand
# on the model. Continuation of running text is a completion task; any
# reasoning the model does is emergent, not requested — which is the
# point of the control.
NARRATIVE = (
    "The village of Lower Ashcombe sits in the fold of the valley where "
    "the river Ash bends west before dropping into the marshes. From the "
    "ridge above, on a clear morning, the roofs form a broken line of "
    "slate along the water, interrupted here and there by the pale square "
    "of a chapel or the yellow brick of a mill. The oldest house in the "
    "village is the one belonging to the Norrey family, though nobody by "
    "that name has lived there for four generations; the deed passed to "
    "an aunt in the 1890s, and from her to a series of tenants who never "
    "stayed longer than a season. Its windows face east, so in winter "
    "the low sun catches the frost on the panes and turns them the colour "
    "of weak tea. Farther down the lane, past the smithy and the row of "
    "almshouses, the road widens into a green where the market used to "
    "be held on the second Thursday of every month. The green is now "
    "little more than a triangle of grass with a war memorial at its "
    "apex, but the low stone kerb around it still bears the marks where "
    "the traders' carts rested. Beyond the green the lane climbs again, "
    "past the churchyard where the yew hedge has grown tall enough to "
    "screen the older graves from the road, and past the paddock where "
    "the schoolmaster's daughter used to keep a Welsh cob called Sixpence. "
    "The paddock is empty now, and the schoolhouse itself has been sold "
    "and converted into a pair of cottages, but the bell above the porch "
    "was left in place at the buyer's request, and every so often, when "
    "the wind is from the west, a passing stranger will still hear it "
    "ring faintly at odd hours of the afternoon."
)


ALL = {
    "short": SHORT,
    "medium": MEDIUM,
    "long": LONG,
    "narrative": NARRATIVE,
}


def word_count(text: str) -> int:
    return len(text.split())
