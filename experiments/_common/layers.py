"""Model-depth-relative layer selection — one place instead of N hardcoded lists.

Every probe that picks "which layers to look at" used to hardcode an
absolute list (`[4, 16, 28]`, `"0,4,8,16,24,31"`, ...) written for a
32-layer model. Every one of those breaks or silently degrades (crash,
NaN, or a wrong Util% denominator) on a smaller model — G1d 0.4B has 24
layers, not 32. Found and patched three separate instances of this same
bug in one night (`ipc_analysis.py`, `probes.py`'s silent-NaN case,
`jlens_probe.py`'s original version) before deciding it needed a shared
fix instead of a fourth ad-hoc guard.

`default_layers(n_layer)` picks layers by *fractional depth* — early,
mid-early, mid, mid-late, late — so the same relative positions get
probed regardless of how many layers the loaded model actually has.
"""
from __future__ import annotations

from typing import List, Sequence

DEFAULT_FRACTIONS = (0.0, 0.125, 0.5, 0.875, 1.0)


def default_layers(n_layer: int, fractions: Sequence[float] = DEFAULT_FRACTIONS) -> List[int]:
    """Layer indices at roughly the given fractional depths (deduplicated, sorted).

    `fractions=0.0` -> layer 0, `1.0` -> the last layer (`n_layer - 1`).
    For small `n_layer`, distinct fractions can round to the same index —
    deduplicated rather than padded, so the caller may get fewer layers
    back than fractions given (this is correct: a 4-layer model genuinely
    doesn't have 5 distinct "depths" worth separating).
    """
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")
    idx = {max(0, min(n_layer - 1, round(f * (n_layer - 1)))) for f in fractions}
    return sorted(idx)


def validate_layers(layers: Sequence[int], n_layer: int, *, context: str = "") -> None:
    """Raise a clear error if any requested layer is out of range for this model.

    Use at the top of a probe when layers were explicitly passed in (e.g.
    via a CLI flag) rather than computed by `default_layers` — an explicit
    request for an out-of-range layer is a caller mistake worth failing
    loudly on, not silently NaN-ing.
    """
    out_of_range = [l for l in layers if l < 0 or l >= n_layer]
    if out_of_range:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"layer(s) {out_of_range} out of range{where} "
            f"(n_layer={n_layer}, valid range 0..{n_layer - 1})"
        )
