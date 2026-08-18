"""Regression guard: catch silent CLI flag conflicts across registered probes.

`experiments/run.py` shares ONE argparse parser across every registered
probe's `add_args`, with `conflict_handler="resolve"` — a flag registered
by two probes doesn't raise, it silently lets whichever probe was added
last win. That's the right behavior when two probes genuinely share a
flag's meaning (e.g. `--layers`), and a real, already-hit bug when they
don't (`--prompt` meant different things for `ipc_analysis` and
`think_geometry`, one silently got the other's default — found
2026-08-18, fixed by prefixing the mismatched one).

This test can't know which same-name collisions are *intentional* vs a
bug — that's a judgment call. It just makes every collision visible
instead of silent, so the judgment actually gets made instead of one
probe quietly losing its flag.

Run: `python experiments/_common/test_registry.py`
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import registry


def _flag_defs(probe_name: str, add_args) -> dict:
    """Call a probe's add_args on a scratch parser, return {flag: (default, help)}."""
    ap = argparse.ArgumentParser()
    add_args(ap)
    out = {}
    for action in ap._actions:
        for opt in action.option_strings:
            out[opt] = (action.default, action.help)
    return out


def check_flag_collisions() -> list[str]:
    """Returns a list of human-readable warnings, empty if nothing suspicious."""
    registry.import_known_probes()
    seen: dict[str, tuple[str, object, str]] = {}  # flag -> (probe, default, help)
    warnings: list[str] = []

    for spec in registry.all_specs():
        if spec.add_args is None:
            continue
        for flag, (default, help_text) in _flag_defs(spec.name, spec.add_args).items():
            if flag in seen:
                prev_probe, prev_default, prev_help = seen[flag]
                # Only the *default value* differing is a real bug candidate —
                # conflict_handler='resolve' silently changes runtime behavior
                # for that. Differing/missing help text alone is a docs
                # inconsistency, not a silent-wrong-answer risk; not flagged.
                if prev_default != default:
                    warnings.append(
                        f"{flag!r}: {prev_probe!r} default={prev_default!r}, "
                        f"but {spec.name!r} default={default!r} — "
                        f"conflict_handler='resolve' means {spec.name!r} silently wins "
                        f"when both are selected together via run.py"
                    )
            else:
                seen[flag] = (spec.name, default, help_text)

    return warnings


def main() -> int:
    warnings = check_flag_collisions()
    if not warnings:
        print(f"[test_registry] OK — no flag conflicts across "
              f"{len(registry.names())} registered probes ({', '.join(registry.names())})")
        return 0
    print(f"[test_registry] FOUND {len(warnings)} potential flag conflict(s):")
    for w in warnings:
        print(f"  - {w}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
