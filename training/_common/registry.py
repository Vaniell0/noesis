"""training/_common/registry.py — normalizer/tokenizer registry.

Mirrors experiments/_common/registry.py's `@registry.probe(...)` pattern
(decouple "which script does the work" from "which stages get run"),
applied to the corpus pipeline instead of probes: `normalize_*.py`
scripts turn raw external/generated data into rollouts JSONL,
`tokenize_*.py` scripts turn JSONL into a `.pt` blob. Both stages
register themselves here so `training/build_corpus.py` can look a
source up by name instead of every recipe hardcoding which script to
invoke with which flags.

Usage in a normalizer or tokenizer module::

    from training._common import registry

    @registry.stage("xlam", kind="normalize", provenance="external-hf",
                     origin="Salesforce/xlam-function-calling-60k",
                     out_default="training/corpus_open/xlam.jsonl")
    def run(args) -> dict:
        ...
        return {"out_path": ..., "n_rows": ...}

Registration only happens on import — see KNOWN_MODULES below, same
deliberate opt-in list (not a glob) as the experiments side, so a
half-written script can't silently join a recipe just by existing
somewhere in the tree.
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence

# Deliberate opt-in list. Add a module's dotted path here when it
# registers a @registry.stage — training/build_corpus.py and
# training/regenerate_corpus_index.py both import from this one list.
KNOWN_MODULES: List[str] = [
    "training.scripts.normalize_hh_rlhf",
    "training.scripts.gen_g1i_warmup",
]

Kind = Literal["normalize", "tokenize"]
Provenance = Literal["generated", "external-hf", "external-other"]

StageFn = Callable[[argparse.Namespace], dict]


@dataclass
class StageSpec:
    name: str
    fn: StageFn
    kind: Kind
    provenance: Provenance
    origin: str  # HF dataset id / URL (external) or a short description (generated)
    out_default: Optional[str] = None
    description: str = ""
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None


_REGISTRY: Dict[str, StageSpec] = {}


def stage(
    name: str,
    *,
    kind: Kind,
    provenance: Provenance,
    origin: str,
    out_default: Optional[str] = None,
    description: str = "",
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """Decorator: register `fn(args) -> dict` under `name`. `fn`'s return
    dict should include at least the fields build_corpus.py and
    provenance.py need to chain stages (e.g. `out_path`, row/token
    counts) — see an individual migrated script for the exact shape."""

    def deco(fn: StageFn) -> StageFn:
        if name in _REGISTRY:
            raise ValueError(
                f"stage {name!r} already registered "
                f"(by {_REGISTRY[name].fn.__module__}.{_REGISTRY[name].fn.__qualname__})"
            )
        _REGISTRY[name] = StageSpec(
            name=name, fn=fn, kind=kind, provenance=provenance, origin=origin,
            out_default=out_default, description=description, add_args=add_args,
        )
        return fn

    return deco


def import_known_modules() -> None:
    for mod in KNOWN_MODULES:
        importlib.import_module(mod)


def get(name: str) -> StageSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"no stage named {name!r} registered — known: {', '.join(names()) or '(none imported)'}"
        )
    return _REGISTRY[name]


def names(kind: Optional[Kind] = None) -> List[str]:
    if kind is None:
        return sorted(_REGISTRY)
    return sorted(n for n, s in _REGISTRY.items() if s.kind == kind)


def all_specs(kind: Optional[Kind] = None) -> List[StageSpec]:
    return [_REGISTRY[n] for n in names(kind)]
