# Phase B skeleton — verdict, 2026-07-22

**Verdict: PASS.** Rust supervisor + 6 collectors + Ollama HTTP heartbeat +
retention scheduler + home-manager auto-rebuild all validated on-host.
Task `#2 Memory Phase B skeleton + NixOS auto-compile сервис` closes here.

## Setup

- Commit at gate: `7d7e0d3` (pushed to `origin/main`).
- Backend: Ollama, model `mollysama/rwkv-7-g1d:0.4b`, `heartbeat_secs=60`.
- Deployment: home-manager generation 209, `services.noesis-runtime`
  enabled + `autoStart` + `autoRebuild` with `sourcePath` pointing at
  the on-host checkout.
- Store root: `~/.local/share/noesis/{input_events,system_obs,
  personal_vault,session_scratch}/db.sqlite`.
- Environment: laptop on battery (41 % → 18 %), screen off, no other
  desktop activity.

## Cadence check (t = 22:29 → 23:39, uptime 1 h 10 min)

Every ticker hit its configured interval to within one sample:

| kind              | count | expected @ interval | interval |
|-------------------|-------|---------------------|----------|
| cpu_ticks         | 423   | 420                 | 10 s     |
| net_counters      | 282   | 280                 | 15 s     |
| host_snapshot     | 141   | 140                 | 30 s     |
| inference_health  |  71   |  70                 | 60 s     |
| runtime_footprint |  71   |  70                 | 60 s     |
| retention_stats   |   4   |   4                 | 15 min   |
| ollama_generation |  56 (probe window) | 56       | 60 s     |

Journal collector produced 23 k `journal_line` events; evdev produced
671 k `input_event` rows across 15 devices. No crashes, no restarts.

## Ollama round-trip

70 completions collected total (14 pre-probe + 56 in-probe). Response
text is coherent RWKV-7 output. Wall-clock distribution:

- median 11.9 s
- min 4.2 s
- max 26.7 s

The tail is dominated by contention: llama-server (~700 MB RSS, ~30 %
CPU during eval) shares the CPU with the evdev polling loop and the
journal stream. Acceptable for a background runtime; not acceptable
for user-facing latency, which is not the goal.

## Retention

Four `retention_stats` firings, all counts zero — nothing in the store
is old enough after 1 h to pass any of the per-zone TTL floors
(input_events 24 h, system_obs 7 d, session_scratch 24 h, vault
never). Correct behaviour under the current knobs.

## Resource footprint

- `noesis-runtime` process: 19 MB RSS steady, 4.1 % CPU average.
- `llama-server` process: 707 MB RSS resident, ~30 % CPU during
  heartbeat, near-idle between.
- Battery drain: 41 % → 18 % across 57 min with screen off. Sustainable
  for short probes on a laptop; the deferred 24 h A0.3 successor
  requires AC.

## Substitutions vs original plan

- **A0.3 (24 h sustained-idle)** — permanently deferred on this host.
  Laptop cannot stay awake 24 h without external intervention. The
  1 h shortened probe above is the substitute for Phase B validation;
  the 24 h probe waits until noesis lives on hardware that can hold
  a plug.

## What follows

- Phase B verified. New goals + roadmap adjustment come next.
- Task #3 (A0.8 test-time compute frontier) still gated by A1
  checkpoint availability.
