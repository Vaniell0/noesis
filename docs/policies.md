# Operating Policies

noesis runs as a persistent process alongside the user, effectively as a
peer Linux user on the same machine. That framing raises operational
policy questions which must be answered before noesis is entrusted with
autonomy on the system.

**Status.** This file is a stub. Every entry below is an **open
question**, not a decision. Do not treat unlocked items as permissive
by default — the default is *ask the user*.

## Filesystem access

- **Read scope.** Which directories may noesis read without asking?
- **Write scope.** Which directories may noesis write to at all, and
  where does writing require an explicit user confirmation?
- **Forbidden paths.** What is never touched (SSH keys, GPG keyring,
  browser profiles, password stores, `.env` files)?

## Execution

- **Command allowlist.** Which commands may noesis execute
  autonomously vs which require confirmation? What's the equivalent of
  Claude Code's tool allowlist for a peer Linux user?
- **Package management.** May noesis install packages? Under what
  scope (user, system)?
- **Long-running processes.** May noesis start services or daemons?
- **Destructive operations.** rm, git push --force, DROP, `dd`,
  systemctl stop — always ask, or defined allow patterns?

## Autonomy vs ask

- **Default posture.** For ambiguous requests, does noesis act on best
  inference or always ask?
- **Confidence thresholds.** At what internal confidence does noesis
  proceed without confirmation?
- **Interruption etiquette.** May noesis surface unsolicited
  observations (e.g., "your build has been failing the same way for
  20 minutes"), or is interaction strictly user-initiated?

## Data egress

- **Remote Claude handoff.** When the user escalates, what info about
  the local system is included in the handoff? What is redacted by
  default?
- **Telemetry.** Any noesis activity logged in a way that could leak
  off-machine? (Baseline expectation: no.)
- **Off-machine retrieval.** Web fetch, arxiv, github reads — same
  policy questions as local read scope.

## Credentials and secrets

- **Discovery.** How does noesis avoid inadvertently ingesting secrets
  (dotenv files, key material embedded in code, cookies)?
- **Handling.** If a secret enters noesis's context accidentally, how
  is it purged from working state and from the memory system?
- **Verification.** How is "did not persist a secret" auditable after
  the fact?

## User separation

- **Account model.** Does noesis run as a distinct Linux user (own
  uid, own home dir), as a systemd user service under the primary
  user, as a nixos module, or as a plain shell process?
- **File ownership.** How does the account choice interact with the
  read/write scope above? A distinct uid gives free OS-level
  enforcement; sharing the uid puts the whole burden on application
  logic.
- **Signal boundaries.** Who can send signals to noesis, and vice
  versa?

---

All items above require explicit user decision before production
deployment. Until decided, noesis's operational default is **ask
before doing**. Where a policy is locked, it is captured here with a
date and rationale, moved out of "open questions" into "locked".

## Locked policies

*(none yet)*
