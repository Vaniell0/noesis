# Operating Policies

noesis runs as a persistent process alongside the user, effectively as a
peer Linux user on the same machine. This file records both the *open
questions* (still to be answered) and the *locked policies* (decided,
dated, applied by the runtime).

**Reading order.** Locked policies at the top are load-bearing for
runtime and build-out; open questions at the bottom track what is still
undecided. The default for unlocked items is *ask the user*.

---

## Locked policies

### User separation and process model (locked 2026-07-22, updated 2026-07-25)

- noesis runs under a **dedicated Linux user** `noesis`, own uid, home
  directory `/var/lib/noesis`. Never as the primary user's own uid.
- **The model runs in-process.** `noesis-runtime` links `noesis-rwkv`
  (rwkv-cpp bindings) directly into its address space; there is **no
  separate model child process**, no Ollama daemon, no unix-socket
  forwarding to a model backend. Rationale: (a) WKV state save / clone
  / load must be reachable from lens, H14/H15/H16, and H18 code paths,
  which requires same-address-space access to the C context; (b)
  `/proc/self/stat` accounting for the CPU budget covers all model cost
  without cgroup plumbing; (c) `RwkvContext::clone_for_parallel` yields
  concurrent contexts sharing the weight mmap, which the drip loop and
  HTTP shim use. See runtime plan §6 for the full lock rationale.
- The Ollama-shape HTTP dialect on TCP (§`inference::rwkv_http`) is a
  **client wire format**, not a delegation path. External clients that
  speak Ollama's `/api/generate` reach the runtime; behind the socket,
  generation runs in-process on rwkv-cpp.
- Delivered as a **systemd system service** with the following
  hardening, minimum set:
  ```
  User=noesis
  Group=noesis
  DynamicUser=no                # own home + persistent state
  ProtectHome=yes               # cannot see /home/vaniello
  ProtectSystem=strict          # /usr, /boot, /etc read-only
  ReadWritePaths=/var/lib/noesis
  PrivateTmp=yes
  NoNewPrivileges=yes
  CapabilityBoundingSet=        # empty
  SystemCallFilter=@system-service
  SystemCallErrorNumber=EPERM
  RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
  RestrictNamespaces=yes
  RestrictSUIDSGID=yes
  LockPersonality=yes
  MemoryDenyWriteExecute=yes
  ```
- **Signal boundary.** Only the systemd unit and the primary user
  (through `systemctl` and `journalctl`) may signal noesis. noesis has
  no signal reach into other user processes.

### Disk encryption for the memory store (locked 2026-07-22)

- Everything under `/var/lib/noesis/store` (SQLite + vector store +
  event log) lives on an **encrypted volume**.
- Preferred: **LUKS-encrypted BTRFS subvolume** mounted at
  `/var/lib/noesis/store`, keyfile in `/etc/noesis/keyfile` (mode 0400,
  root:root). Unlocked at boot via `crypttab` before the noesis unit
  starts.
- Fallback (simpler, less strong): **fscrypt** per-directory encryption
  on the existing `/var` filesystem.
- SSD hardware encryption alone is **not** sufficient (known firmware
  attacks; often not actually enabled). Software layer is required.

### CPU budget and thermal (locked 2026-07-22, refactored 2026-07-25 — authoritative source, was HYPOTHESES §H1 until retracted)

**This section is now the load-bearing definition** of the CPU / thermal
policy. HYPOTHESES §H1 was retracted 2026-07-25 — it was recording an
operating decision as a falsifiable hypothesis, which was the wrong
shape. The policy is a decision, not a wager.

**Single hard rule (ambient mode): fans do not spin up.** CPU% is a
proxy, not the constraint — the same 5% package CPU can be silent on a
passively-cooled machine and audible on a poorly-tuned laptop. The
authoritative signal is inaudibility.

**Two disjoint regimes.**

- **Ambient** — no user activity within `interactive_window_minutes`
  (default 5). Fan-off invariant absolutely. Drip rate is *derived*
  per-machine at supervisor startup — see calibration protocol below.
  Only collectors, scheduler, and rate-limited drip run.
- **Interactive** — user activity within `interactive_window_minutes`.
  Bounded only by hardware thermal limits. Fan-off does NOT apply;
  the user requested the compute, tolerates noise for responsive
  answers. Long-running interactive jobs may be fragmented for
  fairness with other user processes, never for CPU compliance.

Ambient-mode jobs that overrun their CPU window are deferred to the
next window, never extended. Interactive-mode jobs run to completion.
Enforced by `noesis-scheduler` (Rust runtime).

### Startup calibration protocol (locked 2026-07-25)

Runs on supervisor start; results cached in
`/var/lib/noesis/calibration.toml` and reused across restarts.

1. **Warm-up.** One 32-token burst to force kernel compile and CPU
   frequency ramp; discard measurements.
2. **Throughput measurement.** Three burst runs of ~20 tokens each,
   in-process rwkv-cpp. Median `CPU-s / token` across runs; recorded
   as `tokens_per_cpu_second`. Until `measure_throughput` is wired to
   a live rwkv-cpp context (runtime task pending), the conservative
   fallback `9.4 tok/CPU-s` (i5-1235U pilot) is used.
3. **Thermal-safe threshold determination.** Primary signal: package
   die temperature (`coretemp` `Package id N`, falling back to
   max-of-Cores). Rationale: fans respond to temperature; temperature
   predicts fan response before spool-up. `coretemp` reads without
   special privileges on any Intel/AMD Linux box. Fan-RPM sweep was
   dropped — hwmon `fan*_input` returns EIO on ACPI laptops (confirmed
   pilot i5-1235U), and it does not exist at all on fanless hardware.
4. **Auto-sweep algorithm** (shipped 2026-07-25, `calibration::sweep`):
   baseline coretemp for 10 s, then for each step in `[10%, 20%, 30%]`
   package CPU spawn `n_cores` busy-loop threads at the given duty
   cycle, wait 5 s settle, sample peak coretemp over 25 s. Unsafe if
   peak > baseline + 8 °C or > 75 °C absolute (guards idle-hot
   machines). Return highest safe step, or `1%` fallback if even the
   smallest step failed.
5. **Persistence.**
   ```toml
   [calibration]
   tokens_per_cpu_second = 9.4
   fan_safe_cpu_percent = 6.0
   cpu_model = "12th Gen Intel(R) Core(TM) i5-1235U"
   n_cores = 12
   kernel = "6.11.0-9-generic"
   backend = "rwkv-cpp-in-process"
   measured_at = "2026-07-25T14:32:00Z"
   ```
6. **Invalidation triggers.** Recalibrate on: kernel version change,
   backend swap, `uptime > 30 d`, manual `noesis recalibrate`, CPU
   governor change.
7. **Interactive fallback.** For hosts without `coretemp` (AMD Zen
   `k10temp` — different hwmon name; future ARM boards) or where auto
   returns `1%` because the machine idles hot: `noesis-runtime
   calibrate --interactive` CLI — shipped (corrected 2026-08-23; this
   line previously said "planned"), `runtime/noesis-runtime/src/
   calibrate_interactive.rs`, real throughput measurement + interactive
   fan-safe% prompt + `calibration.toml` persistence, plus a
   `--thread-sweep` variant.

**Derived drip formula:**
```
drip.rate_tokens_per_sec =
    fan_safe_cpu_percent / 100 × n_cores × tokens_per_cpu_second × safety_margin
```
`fan_safe_cpu_percent` is package CPU% (matches `top`);
`tokens_per_cpu_second` is per full-core-second of CPU time; the
`× n_cores` factor bridges the two. `safety_margin` default `0.6` —
leaves a 40 % buffer below the thermal threshold to absorb transient
load from other workloads.

**Reference numbers (i5-1235U pilot, historical, small-LLM heartbeat
before in-process rwkv-cpp):**
- `tokens_per_cpu_second` = 9.4 (0.106 CPU-s / token, single burst).
  Kept as fallback until `measure_throughput` lands.
- With `fan_safe_cpu_percent ≈ 6 %` from the shipped thermal sweep
  and 12 cores: ambient drip ≈ 3.4 tok/s.
- Post-in-process rwkv-cpp (~30 tok/s Q8_0, 0.033 CPU-s / token):
  same 6 % ceiling → ~10 tok/s ambient. This is the number the
  runtime plan §11 Task 18 will land as `tokens_per_cpu_second`.

**Budget accounting invariant.** Any new background job (second
distiller path, extension broker tick, retention sweep beyond its
current cost) must declare its per-tick CPU cost in the same units;
the supervisor sums to verify against `fan_safe_cpu_percent` at
startup. No unbudgeted background work. The point of derivation-from-
calibration is that the ceiling is knowable and enforcement is
arithmetic, not guesswork.

**Falsification of a policy has a different shape than falsification
of a hypothesis.** A policy is not falsified — it is *found not to
hold* by the running system, and updated. If any fan-audible episode
occurs in ambient mode over sustained operation, the response order
is: revisit the calibration protocol, revisit the `safety_margin`
default, revisit the scheduler duty cycle. Failure is a
reconfiguration signal, not a research verdict.

### Model backend sandboxing (locked 2026-07-25 — supersedes Ollama child sandboxing)

- Model files under `/var/lib/noesis/models` (owned by `noesis` uid,
  mode 0700). Read-only after install.
- **No child process to sandbox.** The model runs in-process on
  rwkv-cpp linked into `noesis-runtime`. The systemd unit hardening
  above (`ProtectHome=yes`, `ProtectSystem=strict`, empty
  `CapabilityBoundingSet`, `IPAddressDeny=any` + narrow whitelist)
  applies to the whole runtime process and is the sole enforcement
  boundary — no bubblewrap wrapper is needed because there is no
  separate process to isolate. The earlier "Ollama child sandboxing"
  section was removed 2026-07-25 because it referred to an
  architecture we no longer deploy.
- The Ollama-shape HTTP surface on TCP is served by
  `inference::rwkv_http` — a *client wire format* on the runtime's
  own bind, not a delegation to any external Ollama daemon. Bind
  address is configured via `NOESIS_CONFIG`
  (`rwkv_cpp.http_bind`). External clients that speak Ollama's
  `/api/generate` reach the runtime directly.

### Zone-level filesystem permissions (locked 2026-07-22)

Zones (see `memory/README.md`) map to filesystem policies:

| Zone                 | Location                                | Perm to noesis uid |
| -------------------- | --------------------------------------- | ------------------ |
| `input-events`       | `/var/lib/noesis/store/episodic.db`     | read+write         |
| `system-observations`| `/var/lib/noesis/store/episodic.db`     | read+write         |
| `personal-vault`     | `/home/vaniello/Documents/` (Obsidian)  | **read-only bind** |
| `session-scratch`    | RAM (supervisor process memory)         | in-process         |
| `models`             | `/var/lib/noesis/models`                | read-only after install |

Personal-vault is mounted with `ro,bind,nosuid,nodev,noexec` into
noesis's namespace. Never writable.

### Data egress (locked 2026-07-22)

- **Default:** no network egress. Systemd `IPAddressDeny=any` +
  `IPAddressAllow=` targeted whitelist.
- **Whitelist:**
  - Anthropic API host, for user-initiated H5 handoff only. Not
    reachable except during an active handoff turn.
  - NTP (`time.cloudflare.com`) if the host clock is untrusted.
- **No telemetry.** noesis never phones home for its own operation.
- **Handoff redaction.** When the user escalates to remote Claude, the
  handoff payload is prepared by the composer using **only** what
  the retrieval policy pulled for that turn. Anything not explicitly
  in the composer's output does not leave the machine. Redaction is
  by construction (only-what-composer-emitted), not by post-filter.

### A1 fine-tune corpus scope (locked 2026-07-30, Variant C hybrid)

Supersedes the earlier Variant A ("open reasoning traces only",
locked 2026-07-22) after a conflict was surfaced between the
policies-side spec and the corpus-side plan in
`docs/training-data-shortlist.md`. Variant C aligns the two.

- **Primary (majority of the mix): public agent / function-calling
  traces.** Open-licensed, structured `tool_use → tool_result →
  tool_use` rollouts. Candidate datasets and their licences:
  - `Salesforce/xlam-function-calling-60k` — Apache-2.0.
  - `glaive-ai/glaive-function-calling-v2` — Apache-2.0.
  - `thunlp/ToolBench` — MIT.
  - `THUDM/AgentInstruct` — Apache-2.0.
  Loss target: standard next-token loss on `tool_use` tokens only;
  `tool_result` tokens are context (inputs); assistant thinking is
  excluded from the loss mask. Behavior-cloning on *what to do
  next*, not *how to sound while thinking*.
- **Secondary (adaptable open reasoning traces).** Open reasoning
  traces (DeepSeek-R1 distill subsets, competition-math CoT, open
  code-reasoning) may enter A1 **only if restructured** into
  tool-shaped step sequences — reasoning steps linked to tool
  invocations, not free-form thinking text. Untransformed
  reasoning traces do not enter A1 weights.
- **NOT in A1 weights:**
  - User's personal Claude CLI logs, any personal transcripts,
    correspondence, or private corpus.
  - `personal-vault` (Obsidian) content.
  - Any Anthropic-derived reasoning distills (e.g. Fable-5,
    Complete-FABLE.5-traces, other Claude-output-launderings).
    Character-contamination + licence-hygiene both fail.
  - Domain knowledge as such (facts; that goes through retrieval
    per H7).
- **Rationale.** Matches CLAUDE.md hard constraint "open sources
  only, no personal corpus in weights" and the P11 lineage
  requirement. Chooses action-cloning as the primary shape
  (verifiable — τ-bench / AgentBench success rate), keeps the
  narrow reasoning-trace door open only for structured
  transformation, and closes the personal-data door completely.
- **Sanitisation.** Applies to any external corpus before it
  reaches A1 training: secret-regex filter (`detect-secrets` or
  equivalent) followed by sampled manual audit
  (~200 random rollouts). Documented in the eventual
  `training/corpus/README.md`.
- **Variant D (future).** If A1 on Variant C fails the τ-bench
  bar, the reopen path is not "add personal data", it is
  "expand supplementary open corpora and re-run". Personal data
  stays out of weights across all A-track workstreams.

### Credentials-and-secrets handling (locked 2026-07-22, partial)

- **Discovery.** File collectors skip by extension and by canonical
  path prefix: `.env`, `.pem`, `.key`, `id_rsa*`, `id_ecdsa*`,
  `.password-store/**`, `.gnupg/**`, `.ssh/**`, `.aws/**`,
  `.config/*/credentials`, `.mozilla/firefox/*/logins.json`, and the
  user's password-store equivalents. Same skip list applies to inotify
  events.
- **Content matching.** No content scan for secret detection at ingest.
  This is a *conservative* stance — better to miss an event than to
  read a secret while trying to filter it. Secret scanning is out of
  scope for the runtime.
- **Contamination purge.** If a secret does land in `session-scratch`
  (RAM), it dies with the session. If a secret lands in the encrypted
  store on disk, purge requires: (a) locate by content hash across
  `episodic`/`working`, (b) delete rows, (c) rewrite the WAL (SQLite
  `VACUUM`), (d) re-embed if it landed in the vector store, (e) log
  a supersession entry. See §Open questions for the audit tool.

---

## Open questions

### Autonomy vs ask

- **Default posture.** For ambiguous requests, does noesis act on
  best-inference or always ask? Current lean: ask, because the
  H5-handoff model puts the human in the loop by design (P6).
- **Confidence thresholds.** At what internal confidence does noesis
  proceed without confirmation? Untested; needs A0.2-style calibration.
- **Interruption etiquette.** May noesis surface unsolicited
  observations, or is interaction strictly user-initiated? Current
  lean: strictly user-initiated in Phase 1; opportunistic surface only
  after Gate 2 shows the model can time its interruptions well.

### Command execution

- **Command allowlist.** For a peer Linux user, what's the equivalent
  of Claude Code's tool allowlist? Read-only commands (`ls`, `git
  status`, `cat`) are candidates for auto-approve; anything mutating
  requires confirmation.
- **Package management.** May noesis install packages? Under what
  scope? Current lean: **no** in Phase 1. Package installs are a user
  action.
- **Long-running processes.** May noesis start services or daemons?
  Current lean: **no**. All background work happens inside the noesis
  supervisor's own process tree.
- **Destructive ops.** rm, git push --force, DROP, dd, systemctl stop
  — always ask. Never on an allow pattern in Phase 1.

### Secret contamination — audit tool

The purge procedure above is a specification, not a tool. A CLI
`noesis secrets purge --content-hash <sha256>` needs to exist before
we can honour it. Not built yet.

### Off-machine retrieval

- Web fetch, arxiv, github reads — same policy questions as local read
  scope. Current stance: off by default; if enabled, only through the
  supervisor's own outbound HTTP client with the network whitelist
  above.

### Recovery from encrypted-volume failure

- If the LUKS volume fails to unlock (bad key, corrupt superblock),
  what does the supervisor do? Refuse to start (safe) or fall back to
  a plaintext store with a loud warning (recoverable)? Not decided.
  Current default: refuse to start.

---

*Every open question above is a decision the user still owns. When a
question moves to locked, it gets a date and a rationale, and it moves
above the divider.*
