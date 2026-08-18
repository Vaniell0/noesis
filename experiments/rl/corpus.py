"""corpus.py — curriculum-aware task sampler for WKV-loop RL.

CorpusScheduler indexes matrix_tasks.jsonl by (category, level) and samples
batches according to the current curriculum level. It tracks per-level accuracy
and advances or drops the level automatically.

Curriculum rules (from docs/rl-track.md):
    L0 anchor: matrix_wordsearch_name always in mix (~10% share)
    Advance    if batch accuracy >= advance_thresh (default 0.80)
    Drop back  if batch accuracy  < drop_thresh    (default 0.50)
    Max level  capped per category by available data

Category base weights (static):
    matrix_wordsearch       0.20
    matrix_wordsearch_name  0.10
    crossword_enum          0.05
    crossword_fill          0.05
    arithmetic_matrix       0.20
    pattern_matrix          0.20
    bits_matrix             0.20
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_DEFAULT_WEIGHTS: Dict[str, float] = {
    "matrix_wordsearch":      0.20,
    "matrix_wordsearch_name": 0.10,
    "crossword_enum":         0.05,
    "crossword_fill":         0.05,
    "arithmetic_matrix":      0.20,
    "pattern_matrix":         0.20,
    "bits_matrix":            0.20,
}

# Categories pinned to L0-only (always included regardless of current level)
_L0_ANCHOR = {"matrix_wordsearch_name"}


class CorpusScheduler:
    """Curriculum-aware sampler. Thread-unsafe — single-process use only."""

    def __init__(
        self,
        tasks: List[dict],
        *,
        cat_weights: Optional[Dict[str, float]] = None,
        level_window: int = 2,       # sample from [cur_level - window, cur_level]
        advance_thresh: float = 0.80,
        drop_thresh: float = 0.50,
        start_level: int = 1,
        rng_seed: Optional[int] = None,
    ):
        self._weights = cat_weights or _DEFAULT_WEIGHTS
        self._window = level_window
        self._advance_thresh = advance_thresh
        self._drop_thresh = drop_thresh
        self._rng = random.Random(rng_seed)

        # Index: {category: {level: [task, ...]}}
        self._index: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
        for t in tasks:
            cat = t.get("category", "")
            lvl = t.get("level", 1)
            self._index[cat][lvl].append(t)

        # Per-category max available level
        self._max_level: Dict[str, int] = {
            cat: max(lvl_dict.keys())
            for cat, lvl_dict in self._index.items()
        }

        # Per-category current frontier level (advances independently)
        self._cur_level: Dict[str, int] = {
            cat: start_level for cat in self._index
        }
        # L0-anchor cats stay at level 1 regardless
        for cat in _L0_ANCHOR:
            self._cur_level[cat] = 1

        # Accuracy history for per-level tracking (level → recent window)
        self._acc_history: Dict[int, List[float]] = defaultdict(list)
        self._acc_window = 5    # batches to average before deciding to advance/drop

    # ------------------------------------------------------------------
    # Checkpoint state — added 2026-08-18. `_save_checkpoint` in
    # train_wkv_loop.py saved model weights but not curriculum progress;
    # resuming would silently reset every category back to start_level=1
    # and drop the accuracy history that drives advance/drop decisions —
    # correct model weights, but re-earning curriculum progress from
    # scratch on every resume. Matters given the interruptible-instance
    # plan (frequent, unplanned resumes are the expected case, not rare).

    def state_dict(self) -> dict:
        return {
            "cur_level": dict(self._cur_level),
            "acc_history": {k: list(v) for k, v in self._acc_history.items()},
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._cur_level.update(state["cur_level"])
        self._acc_history = defaultdict(list, {
            int(k): list(v) for k, v in state["acc_history"].items()
        })
        self._rng.setstate(state["rng_state"])

    # ------------------------------------------------------------------
    # Public API

    @property
    def current_level(self) -> int:
        """Global current level: max across non-anchor categories."""
        non_anchor = {c: l for c, l in self._cur_level.items()
                      if c not in _L0_ANCHOR}
        return max(non_anchor.values()) if non_anchor else 1

    def sample_batch(self, batch_size: int) -> List[dict]:
        """Sample `batch_size` tasks from the current curriculum frontier."""
        cats = list(self._weights.keys())
        cat_w = [self._weights.get(c, 0.0) for c in cats]
        total_w = sum(cat_w)
        if total_w == 0:
            raise ValueError("All category weights are zero")

        result: List[dict] = []
        for _ in range(batch_size):
            # Pick category by weight
            cat = self._rng.choices(cats, weights=cat_w, k=1)[0]
            task = self._sample_from_cat(cat)
            if task is None:
                # fallback: any non-empty category
                for fallback in self._rng.sample(cats, len(cats)):
                    task = self._sample_from_cat(fallback)
                    if task is not None:
                        break
            if task is not None:
                result.append(task)

        return result

    def update_accuracy(self, level: int, accuracy: float) -> Tuple[str, int]:
        """Record batch accuracy at `level` and advance/drop if threshold met.

        Returns ("advance"|"drop"|"hold", new_global_level).
        """
        hist = self._acc_history[level]
        hist.append(accuracy)
        if len(hist) > self._acc_window:
            hist.pop(0)

        if len(hist) < self._acc_window:
            return "hold", self.current_level

        rolling = sum(hist) / len(hist)
        action = "hold"

        if rolling >= self._advance_thresh:
            for cat in self._cur_level:
                if cat in _L0_ANCHOR:
                    continue
                max_l = self._max_level.get(cat, 1)
                new_l = min(self._cur_level[cat] + 1, max_l)
                if new_l != self._cur_level[cat]:
                    self._cur_level[cat] = new_l
                    action = "advance"
            hist.clear()

        elif rolling < self._drop_thresh:
            for cat in self._cur_level:
                if cat in _L0_ANCHOR:
                    continue
                new_l = max(self._cur_level[cat] - 1, 1)
                if new_l != self._cur_level[cat]:
                    self._cur_level[cat] = new_l
                    action = "drop"
            hist.clear()

        return action, self.current_level

    def status(self) -> dict:
        return {
            "current_level": self.current_level,
            "per_cat_level": dict(self._cur_level),
            "advance_thresh": self._advance_thresh,
            "drop_thresh": self._drop_thresh,
        }

    # ------------------------------------------------------------------
    # Internal

    def _sample_from_cat(self, cat: str) -> Optional[dict]:
        lvl_dict = self._index.get(cat)
        if not lvl_dict:
            return None
        cur = self._cur_level.get(cat, 1)
        lo = max(1, cur - self._window)
        hi = cur
        # collect all available levels in window
        pool: List[dict] = []
        for lvl in range(lo, hi + 1):
            pool.extend(lvl_dict.get(lvl, []))
        if not pool:
            return None
        return self._rng.choice(pool)


# ------------------------------------------------------------------
# Convenience loader

def load_corpus(path: str, **kwargs) -> CorpusScheduler:
    tasks: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return CorpusScheduler(tasks, **kwargs)
