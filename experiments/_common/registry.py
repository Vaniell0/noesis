"""Probe registry — decouple "which model is loaded" from "which tests run".

Every probe script does the same three things: load a model, run some
measurement against it, save the result. Loading is the expensive part
(seconds to load 0.4B, much longer for 2.9B on CPU), and until now every
probe paid that cost independently — running IPC then think_geometry then
mlp_probe on the same checkpoint meant loading that checkpoint three times,
serially, each a separate process competing for the same CPU.

A probe registers itself once with `@probe(...)`; `experiments/run.py`
loads the model exactly once and runs whichever registered probes were
selected against that single loaded instance.

Usage in a probe module::

    from experiments._common import registry

    def _add_args(ap):
        ap.add_argument("--n-tokens", type=int, default=256)

    @registry.probe("ipc", hypothesis=["H8"], description="...", add_args=_add_args)
    def run(model, tokenizer, args) -> dict:
        ...
        return {"model": args.model, "results": {...}}

Registration only happens on import, so callers must import the probe
module before it shows up in `--list` / `--tests` / `write_probe_info`.
See `KNOWN_PROBE_MODULES` below — add your new module's dotted path there
when you register a probe; `experiments/run.py` and
`experiments/regenerate_results.py` both import from this one list, so
there's a single place to register a new probe module, not two.
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# Deliberate opt-in list, not an auto-discovering glob — a stray or
# half-written probe script can't silently join the battery just by
# existing somewhere in the tree.
KNOWN_PROBE_MODULES = [
    "experiments.A0_state_probe.ipc_analysis",
    "experiments.A0_state_probe.mlp_probe",
    "experiments.A0_state_probe.rlens_probe",
]


def import_known_probes() -> None:
    for mod in KNOWN_PROBE_MODULES:
        importlib.import_module(mod)

ProbeFn = Callable[[object, object, argparse.Namespace], dict]
AddArgsFn = Callable[[argparse.ArgumentParser], None]


@dataclass
class ProbeSpec:
    name: str
    fn: ProbeFn
    hypothesis: List[str] = field(default_factory=list)
    description: str = ""
    add_args: Optional[AddArgsFn] = None


_REGISTRY: Dict[str, ProbeSpec] = {}


def probe(
    name: str,
    *,
    hypothesis: Sequence[str] = (),
    description: str = "",
    add_args: Optional[AddArgsFn] = None,
):
    """Decorator: register `fn(model, tokenizer, args) -> dict` under `name`."""

    def deco(fn: ProbeFn) -> ProbeFn:
        if name in _REGISTRY:
            raise ValueError(
                f"probe {name!r} already registered "
                f"(by {_REGISTRY[name].fn.__module__}.{_REGISTRY[name].fn.__qualname__})"
            )
        _REGISTRY[name] = ProbeSpec(
            name=name, fn=fn, hypothesis=list(hypothesis),
            description=description, add_args=add_args,
        )
        return fn

    return deco


def get(name: str) -> ProbeSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"no probe named {name!r} registered — known: {', '.join(names()) or '(none imported)'}"
        )
    return _REGISTRY[name]


def names() -> List[str]:
    return sorted(_REGISTRY)


def all_specs() -> List[ProbeSpec]:
    return [_REGISTRY[n] for n in names()]
