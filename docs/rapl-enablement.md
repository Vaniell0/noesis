# RAPL enablement for noesis calibration

Startup calibration (`HYPOTHESES.md` §H1) derives the ambient drip
ceiling from thermal headroom and rwkv-cpp throughput. Intel/AMD RAPL
adds a third, orthogonal signal — joules per token — that lets the
sweep cap on package watts rather than temperature delta alone. RAPL
is a **bonus**: the runtime works without it, and the calibration
picks a silent drip either way.

This document covers the enablement path. Reader wiring is a separate
follow-up (see runtime plan §11, Task 17).

## Why it's gated behind root

`/sys/class/powercap/*/energy_uj` was readable by any user until the
mitigation for **CVE-2020-8694 (Platypus)** landed. A process polling
`energy_uj` at high rate can distinguish ~65 mV energy differences
correlated with AES key material and other cryptographic operations.
After the mitigation, the sysfs attribute is mode `0400`, root-only.

noesis-runtime runs as the unprivileged `noesis` service user (see
`docs/policies.md` §"User separation and process model"), so a fresh
install cannot read RAPL. The calibration job logs a `rapl_advice`
event pointing to this document (planned) instead of failing.

## Two enablement options

### Option A — udev rule (preferred)

Ships as `runtime/contrib/udev/60-noesis-rapl.rules`. Installs a
`SUBSYSTEM=="powercap"` rule that `chgrp noesis` + `chmod g+r`s every
`energy_uj` attribute on device add. Effect: only the `noesis` user
gains read access; every other unprivileged user on the box is still
blocked at the original `0400` root-only mode via group.

Install:
```
sudo cp runtime/contrib/udev/60-noesis-rapl.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=powercap
```

Verify:
```
sudo -u noesis cat /sys/class/powercap/intel-rapl:0/energy_uj
```
Should print a monotonic microjoule counter.

**Trade-off.** Enables the Platypus side-channel *for the noesis user
only*. Acceptable when noesis is trusted (it's already reading input
events and journal on the same box). Not acceptable on hosts where
other tenants share the `noesis` group — the systemd defaults ship
`Group=noesis` with `DynamicUser=no`, so this only matters if the
operator has explicitly added other accounts to the group.

### Option B — systemd `AmbientCapabilities`

Add `AmbientCapabilities=CAP_SYS_RAWIO` (or `CAP_DAC_READ_SEARCH`) to
the `noesis-runtime.service` unit. The noesis process then bypasses
the sysfs mode check for `energy_uj`.

Heavier-handed than option A because `CAP_SYS_RAWIO` grants raw I/O
access far beyond RAPL (raw block devices, `/dev/mem` on some kernels,
`iopl()`), which conflicts with the `CapabilityBoundingSet=` empty
policy in `docs/policies.md`. Prefer option A unless udev is
unavailable (containers, embedded distros).

## Doing nothing

No enablement → calibration keeps `throughput_source: "measured"` and
`fan_safe_cpu_percent` from the thermal sweep. The bonus RAPL signal
is skipped and the drip ceiling is derived from the two-signal path,
which is fine — the pilot number (`FALLBACK_TOKENS_PER_CPU_SECOND =
9.4`) was measured without RAPL and produced silent operation on the
i5-1235U reference host.
