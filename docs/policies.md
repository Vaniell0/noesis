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

### User separation and process model (locked 2026-07-22)

- noesis runs under a **dedicated Linux user** `noesis`, own uid, home
  directory `/var/lib/noesis`. Never as the primary user's own uid.
- The Ollama inference backend runs as a **child process** of the Rust
  supervisor under the same `noesis` uid (`fork` + `setsid`, own process
  group). It is not a separate systemd service and is not reachable
  from outside the supervisor.
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

### CPU budget (locked 2026-07-22, mirrors HYPOTHESES §H1)

Two disjoint regimes; the scheduler enforces both:

- **Steady:** < 1 % CPU. Model resident but idle
  (Ollama `keep_alive: -1`); only collectors and scheduler run.
- **Burst:** up to ~20 % CPU for tens of seconds at
  minute-scale periodicity. Every LLM job — composer, incremental
  digest, reflection, retrieval-rerank — is a burst.

Long-running jobs are **fragmented into burst chunks**, never allowed
to sustain. A job that overruns its budget window is deferred to the
next window, not extended. Enforced by `noesis-scheduler` (Rust
runtime).

### Ollama child sandboxing (locked 2026-07-22)

- Model files under `/var/lib/noesis/models` (owned by `noesis` uid,
  mode 0700).
- Ollama child is launched with **bubblewrap**:
  ```
  bwrap --unshare-all --die-with-parent \
        --ro-bind /nix /nix \
        --ro-bind /var/lib/noesis/models ~/.ollama \
        --bind   /var/lib/noesis/ollama-tmp /tmp \
        --dev /dev --proc /proc \
        --setenv OLLAMA_HOST unix:///var/lib/noesis/ollama.sock \
        ollama serve
  ```
- Ollama listens **only on a unix socket** owned by `noesis`. No TCP
  bind, no network egress from the child.
- Supervisor holds the socket end and forwards from an OpenAI-compatible
  endpoint if/when one is exposed outward.

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

### A1 fine-tune corpus scope (locked 2026-07-22, Variant A)

- **In A1 weights:** open reasoning traces (DeepSeek-R1 distill,
  competition-math CoT, open code-reasoning), public Anthropic
  tool-use documentation, open MCP tool-schema examples.
- **NOT in A1 weights:**
  - User's personal Claude CLI logs.
  - Any personal transcripts, correspondence, or private corpus.
  - `personal-vault` (Obsidian) content.
- **Rationale.** Matches CLAUDE.md hard constraint "open sources only,
  no personal corpus in weights" without any reopen. The safe-first
  baseline; may be revisited (Variant B — sanitised pattern extraction)
  only if A1 eval shows the model cannot learn the noesis tool surface
  from open corpora alone.

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
