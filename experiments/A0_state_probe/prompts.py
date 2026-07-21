"""Re-export A0_baseline prompts so A0.4 doesn't duplicate them.

Loaded via importlib by absolute path rather than sys.path insertion,
because a plain `from prompts import ...` would resolve to *this* file
(same name, same directory as `run.py`), causing self-import.
"""

from __future__ import annotations

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE_PROMPTS = os.path.normpath(
    os.path.join(_HERE, "..", "A0_baseline", "prompts.py")
)

_spec = importlib.util.spec_from_file_location(
    "_a0_baseline_prompts", _BASELINE_PROMPTS
)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load A0_baseline prompts from {_BASELINE_PROMPTS}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ALL = _mod.ALL
SHORT = _mod.SHORT
MEDIUM = _mod.MEDIUM
LONG = _mod.LONG
NARRATIVE = _mod.NARRATIVE
word_count = _mod.word_count

__all__ = ["ALL", "SHORT", "MEDIUM", "LONG", "NARRATIVE", "word_count"]
