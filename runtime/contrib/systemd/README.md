# noesis as a systemd system service

This directory ships a plain-systemd unit that runs `noesis-runtime` under a
dedicated `noesis` uid, with the hardening block from `docs/policies.md
§ User separation and process model`. It is intentionally distribution-neutral
— no NixOS or nixpkgs assumption. NixOS users have a `services.noesis.enable`
option in the flake (`nix/nixos-module.nix`) that generates the same unit
declaratively.

For hot-reload dev on a nix machine, prefer the home-manager module
(`nix/hm-module.nix`) instead — it runs as a user service under your own uid
and reloads on source edits.

## Prerequisites

- systemd ≥ 246 (for `StateDirectory=` with multiple entries and
  `ProtectProc=invisible`)
- A pre-built `noesis-runtime` binary. Either:
  - `nix build .#noesis-runtime` and copy `result/bin/noesis-runtime` to
    `/usr/local/bin/`, or
  - `cd runtime && cargo build --release -p noesis-runtime` and copy
    `target/release/noesis-runtime` to `/usr/local/bin/`.
- rwkv-cpp shared library reachable at runtime (either via `LD_LIBRARY_PATH`
  in a drop-in or statically linked at build time).
- One or two RWKV-7 model files in rwkv.cpp `.bin` format (substrate +
  optional utility).

## Install

```sh
# 1. Create the dedicated system user (no shell, no home in /home).
sudo useradd \
    --system \
    --home-dir /var/lib/noesis \
    --shell /usr/sbin/nologin \
    --comment "noesis persistent cognitive runtime" \
    noesis

# 2. Install the unit and the config skeleton.
sudo install -m 0644 noesis.service /etc/systemd/system/noesis.service
sudo install -d -m 0750 -o root -g noesis /etc/noesis
sudo install -m 0640 -o root -g noesis \
    config.toml.example /etc/noesis/config.toml

# 3. Edit /etc/noesis/config.toml — set model_path to a real .bin file.
sudo $EDITOR /etc/noesis/config.toml

# 4. Drop model files under /var/lib/noesis/models/ (created on first start).
#    Or create the directory now:
sudo install -d -m 0700 -o noesis -g noesis /var/lib/noesis/models
sudo cp /path/to/rwkv7-g1h-2.9b-Q5_1.bin /var/lib/noesis/models/
sudo chown noesis:noesis /var/lib/noesis/models/*
sudo chmod 0400 /var/lib/noesis/models/*.bin

# 5. Enable + start.
sudo systemctl daemon-reload
sudo systemctl enable --now noesis.service

# 6. Watch it come up.
journalctl -u noesis -f
```

Expected first-boot log lines:

```
noesis-runtime starting
config loaded state_path=/var/lib/noesis/store backend=rwkv-cpp
all zone stores open
ambient drip ceiling source=fallback tokens_per_cpu_second=9.4 …
collectors spawned count=6
heartbeat
```

`source=fallback` means the calibration protocol hasn't finished yet — a
sweep runs in the background and writes `/var/lib/noesis/store/calibration.toml`
after ~2 minutes. Next restart uses measured numbers.

## Encrypted store (optional but per `docs/policies.md`)

`docs/policies.md § Disk encryption for the memory store` requires
`/var/lib/noesis/store` to live on an encrypted volume. The unit's
`RequiresMountsFor=/var/lib/noesis/store` waits for the mount before
starting; no changes to the unit needed.

Preferred (LUKS over BTRFS subvolume):

```sh
# Create + open the LUKS container (details out of scope for this README).
# Assume /dev/mapper/noesis-store is the unlocked mapping.
sudo mkfs.btrfs /dev/mapper/noesis-store
echo '/dev/mapper/noesis-store /var/lib/noesis/store btrfs defaults 0 2' \
    | sudo tee -a /etc/fstab
sudo systemctl daemon-reload
sudo mount /var/lib/noesis/store
sudo chown noesis:noesis /var/lib/noesis/store
sudo chmod 0700 /var/lib/noesis/store
```

Fallback (fscrypt) — simpler, still meets the policy:

```sh
sudo fscrypt encrypt /var/lib/noesis/store --user=noesis
```

Plain disk works too (no encryption) — everything falls into a normal
`/var/lib/noesis/store` directory owned by `noesis`. Not policy-compliant,
but useful for dev.

## Personal-vault (Obsidian) bind

`docs/policies.md § Zone-level FS permissions` binds the user's Obsidian
vault into noesis's namespace read-only. Path is machine-specific — set it
via drop-in override:

```sh
sudo systemctl edit noesis
```

Paste:

```ini
[Service]
BindReadOnlyPaths=/home/YOUR_USER/Documents:/var/lib/noesis/vault
```

`ProtectHome=yes` in the base unit hides `/home/*` from the runtime by
default; the bind grafts one specific subtree back in as `/var/lib/noesis/vault`.
The runtime does not yet read this path automatically — the option is here so
you can wire the vault-ingest collector once it exists without touching the
base unit.

## Calibration

The first `systemctl start noesis` runs on the fallback throughput number
(9.4 tok/CPU-s, from the pilot i5-1235U). A background thermal + throughput
sweep completes within ~2 minutes and writes measured values to
`/var/lib/noesis/store/calibration.toml`. Restart the unit to pick them up:

```sh
sudo systemctl restart noesis
journalctl -u noesis -n 20 | grep 'ambient drip ceiling'
```

To force a fresh calibration interactively (recommended when moving the
service to a new machine):

```sh
sudo -u noesis NOESIS_CONFIG=/etc/noesis/config.toml \
    /usr/local/bin/noesis-runtime calibrate --interactive
```

Thread sweep for optimal HTTP throughput:

```sh
sudo -u noesis NOESIS_CONFIG=/etc/noesis/config.toml \
    /usr/local/bin/noesis-runtime calibrate --thread-sweep
```

## Observability

- `journalctl -u noesis` — full log.
- `journalctl -u noesis -f` — follow live.
- `journalctl -u noesis -p warning` — warnings and up.
- `systemctl status noesis` — process state, memory, tasks.
- `systemd-cgtop /system.slice/noesis.service` — live CPU / mem.
- `sqlite3 /var/lib/noesis/store/system_obs/db.sqlite "select count(*) from events;"` — event count.

## Network egress

Default is `IPAddressDeny=any` — no outbound. If you enable the H5 handoff
to Anthropic, uncomment the corresponding `IPAddressAllow=` line in the
unit (or add to a drop-in). NTP whitelist likewise.

## Uninstall

```sh
sudo systemctl disable --now noesis.service
sudo rm /etc/systemd/system/noesis.service
sudo systemctl daemon-reload
# State (contains encrypted personal data) is left intact — remove manually:
sudo rm -rf /var/lib/noesis /etc/noesis
sudo userdel noesis
```

## Multi-tenant (future — not yet shipped)

Current shape is single-tenant: one `noesis` uid, one `/var/lib/noesis/store`.
The natural extension for a shared host is `noesis@.service` template — one
instance per human, uid `noesis-alice` / `noesis-bob`, own state trees, shared
read-only `/var/lib/noesis/models/` via mmap. Deferred until the design
question in `docs/policies.md` is closed. Do not attempt to run this
single-tenant unit as multiple users by hand — bind-mounts and
`SupplementaryGroups` collide.
