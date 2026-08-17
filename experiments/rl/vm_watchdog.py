"""vm_watchdog.py — Selectel VM lifetime tracking for RL training.

Selectel VMs have a 24h lifetime from boot (confirmed in reference_selectel_vm_lifetime.md).
There is no API-accessible countdown — only the Selectel panel shows the deadline.
Approximation: boot_time + 24h, derived from /proc/uptime or `who -b`.

Usage:
    from experiments.rl.vm_watchdog import VMWatchdog
    wd = VMWatchdog(lifetime_hours=24.0, warn_at=[4.0, 2.0, 1.0, 0.5, 0.25])
    wd.print_status()

    # In training loop:
    if wd.should_checkpoint(force_if_remaining_hours=2.0):
        save_checkpoint(...)
    if wd.should_stop(stop_if_remaining_hours=0.25):
        break

For the actual remaining time, log into the Selectel panel — this estimate
can drift by ±5 min due to boot sequence overhead.
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


_LIFETIME_HOURS = 24.0
_DEFAULT_WARN_AT = [4.0, 2.0, 1.0, 0.5, 0.25]   # hours remaining


def _uptime_seconds() -> float:
    """Read /proc/uptime (Linux only). Returns float seconds."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except (FileNotFoundError, ValueError):
        return 0.0


def _boot_time_from_who() -> Optional[datetime]:
    """Parse boot time from `who -b` output."""
    try:
        out = subprocess.check_output(["who", "-b"], text=True, timeout=5)
        for line in out.splitlines():
            if "system boot" in line.lower() or "boot" in line.lower():
                parts = line.strip().split()
                if len(parts) >= 3:
                    dt_str = " ".join(parts[-2:])
                    for fmt in ("%Y-%m-%d %H:%M", "%b %d %H:%M"):
                        try:
                            dt = datetime.strptime(dt_str, fmt)
                            if dt.year == 1900:
                                dt = dt.replace(year=datetime.now().year)
                            return dt
                        except ValueError:
                            continue
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _estimate_boot_datetime() -> datetime:
    """Best-effort boot datetime. Prefers /proc/uptime, falls back to who -b."""
    uptime = _uptime_seconds()
    if uptime > 0:
        return datetime.now() - timedelta(seconds=uptime)
    who_dt = _boot_time_from_who()
    if who_dt is not None:
        return who_dt
    raise RuntimeError(
        "Cannot determine VM boot time. "
        "Check /proc/uptime or `who -b` manually."
    )


class VMWatchdog:
    """Tracks remaining lifetime of the Selectel VM."""

    def __init__(
        self,
        lifetime_hours: float = _LIFETIME_HOURS,
        warn_at: Optional[List[float]] = None,
    ):
        self.lifetime_hours = lifetime_hours
        self.warn_at = sorted(warn_at or _DEFAULT_WARN_AT, reverse=True)
        self._boot_dt = _estimate_boot_datetime()
        self._deadline = self._boot_dt + timedelta(hours=lifetime_hours)
        self._warned: set = set()

    @property
    def boot_datetime(self) -> datetime:
        return self._boot_dt

    @property
    def deadline(self) -> datetime:
        return self._deadline

    def remaining_hours(self) -> float:
        delta = self._deadline - datetime.now()
        return max(0.0, delta.total_seconds() / 3600.0)

    def remaining_str(self) -> str:
        h = self.remaining_hours()
        hrs = int(h)
        mins = int((h - hrs) * 60)
        return f"{hrs}h {mins:02d}m"

    def print_status(self) -> None:
        print(
            f"[watchdog] boot={self._boot_dt:%Y-%m-%d %H:%M}  "
            f"deadline={self._deadline:%Y-%m-%d %H:%M}  "
            f"remaining={self.remaining_str()}"
        )

    def check_warnings(self) -> List[float]:
        """Print warnings for passed thresholds. Returns newly triggered thresholds."""
        rem = self.remaining_hours()
        triggered = []
        for thresh in self.warn_at:
            if rem <= thresh and thresh not in self._warned:
                self._warned.add(thresh)
                triggered.append(thresh)
                h = int(thresh)
                m = int((thresh - h) * 60)
                label = f"{h}h {m:02d}m" if m else f"{h}h"
                print(
                    f"[watchdog] !!! {label} REMAINING — "
                    f"checkpoint NOW if you haven't. "
                    f"Deadline: {self._deadline:%H:%M}"
                )
        return triggered

    def should_checkpoint(self, force_if_remaining_hours: float = 2.0) -> bool:
        """Return True if remaining time is at or below the forced-checkpoint threshold."""
        return self.remaining_hours() <= force_if_remaining_hours

    def should_stop(self, stop_if_remaining_hours: float = 0.25) -> bool:
        """Return True if training should abort to avoid mid-run expiry."""
        return self.remaining_hours() <= stop_if_remaining_hours


# ------------------------------------------------------------------
# Convenience: hook into a training loop

class WatchdogHook:
    """Mixin for training loops. Call .tick() after each batch."""

    def __init__(
        self,
        watchdog: VMWatchdog,
        checkpoint_fn,           # callable() → None, saves a checkpoint
        stop_hours: float = 0.25,
        force_ckpt_hours: float = 2.0,
        check_interval_steps: int = 50,
    ):
        self._wd = watchdog
        self._checkpoint_fn = checkpoint_fn
        self._stop_hours = stop_hours
        self._force_ckpt_hours = force_ckpt_hours
        self._interval = check_interval_steps
        self._step = 0
        self._forced_ckpt_done = False

    def tick(self) -> bool:
        """Call every batch. Returns True if training should stop."""
        self._step += 1
        if self._step % self._interval != 0:
            return False

        self._wd.check_warnings()

        if not self._forced_ckpt_done and self._wd.should_checkpoint(self._force_ckpt_hours):
            print(f"[watchdog] forcing checkpoint (≤{self._force_ckpt_hours}h remaining)")
            self._checkpoint_fn()
            self._forced_ckpt_done = True

        if self._wd.should_stop(self._stop_hours):
            print(f"[watchdog] stopping training — ≤{self._stop_hours}h remaining")
            return True

        return False


# ------------------------------------------------------------------
# CLI smoke

if __name__ == "__main__":
    wd = VMWatchdog()
    wd.print_status()
    triggered = wd.check_warnings()
    if not triggered:
        print(f"[watchdog] no warnings triggered (remaining: {wd.remaining_str()})")
