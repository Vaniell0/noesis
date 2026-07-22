"""A0.6 verdict aggregator.

Consumes every per-cell JSON dump produced by ``a06_run.py`` under
``results/a06/`` and emits a compact table plus the PASS/FAIL/CAVEAT
verdict specified in ``README.md``:

    PASS on portability:
        alignment ≤ −0.3
        AND lex_hit_donor_in_cross > lex_hit_donor_in_clean_recipient_null + 0.05
        AND surface_garble.cross.coherence_flag = 1
        (must hold in ≥ 2 of 3 pairs)

    FAIL on portability:
        alignment ≥ 0 OR lex hit fails to exceed null.

Anything in between is a CAVEAT and calls for per-layer analysis.

The aggregator is deliberately stateless — it reads every JSON in the
input directory and returns a summary structure that either a
Markdown writer or a downstream verdict tool can consume. It does not
mutate the JSON dumps themselves.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ALIGNMENT_PASS_THRESHOLD = -0.30
LEX_HIT_DELTA_PASS_THRESHOLD = 0.05
PAIR_PASS_QUORUM = 2  # ≥ 2 of 3 pairs must pass


@dataclass
class Cell:
    path: Path
    model: str
    pair_id: str
    direction: str
    depth: str
    mode: str
    alignment: float
    delta_hit_donor_vs_null: float
    coherence_flag: float
    first_divergence_step: int
    topk_jaccard_k10_mean: float

    def cell_passes(self) -> bool:
        return (self.alignment <= ALIGNMENT_PASS_THRESHOLD
                and self.delta_hit_donor_vs_null > LEX_HIT_DELTA_PASS_THRESHOLD
                and self.coherence_flag >= 1.0)

    def cell_fails(self) -> bool:
        return (self.alignment >= 0
                or self.delta_hit_donor_vs_null <= LEX_HIT_DELTA_PASS_THRESHOLD)

    def verdict(self) -> str:
        if self.cell_passes():
            return "PASS"
        if self.cell_fails():
            return "FAIL"
        return "CAVEAT"


@dataclass
class ModelSummary:
    model: str
    cells: list = field(default_factory=list)

    def by_pair(self) -> "dict[str, list[Cell]]":
        out: "dict[str, list[Cell]]" = defaultdict(list)
        for c in self.cells:
            out[c.pair_id].append(c)
        return out

    def by_axis(self, axis: str) -> "dict[str, list[Cell]]":
        out: "dict[str, list[Cell]]" = defaultdict(list)
        for c in self.cells:
            out[getattr(c, axis)].append(c)
        return out


def _extract_model_from_filename(name: str) -> str:
    # Filename shape: {model}__{pair}__{direction}__{depth}__{mode}.json
    return name.split("__", 1)[0]


def load_cells(results_dir: Path) -> "list[Cell]":
    cells: "list[Cell]" = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue  # log or scratch file, not a cell dump
        with p.open() as f:
            d = json.load(f)
        cfg = d["config"]
        cells.append(Cell(
            path=p,
            model=_extract_model_from_filename(p.name),
            pair_id=cfg["pair_id"],
            direction=cfg["direction"],
            depth=cfg["depth"],
            mode=cfg["mode"],
            alignment=float(d["alignment"]["alignment"]),
            delta_hit_donor_vs_null=float(
                d["lexicon"]["delta_hit_donor_vs_null"]),
            coherence_flag=float(d["surface_garble"]["cross"]["coherence_flag"]),
            first_divergence_step=int(d["first_divergence_step"]),
            topk_jaccard_k10_mean=float(d["topk_jaccard"]["k10_mean"]),
        ))
    return cells


def group_by_model(cells: "list[Cell]") -> "dict[str, ModelSummary]":
    out: "dict[str, ModelSummary]" = {}
    for c in cells:
        out.setdefault(c.model, ModelSummary(model=c.model)).cells.append(c)
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_num(x: float, prec: int = 3) -> str:
    return f"{x:+.{prec}f}" if x < 0 or x > 0 else f"{0.0:+.{prec}f}"


def render_cell_table(cells: "list[Cell]") -> str:
    header = ("| pair | dir | depth | mode | alignment | Δhit_donor "
              "| coh | k10J | fds | verdict |")
    sep = "|------|-----|-------|------|-----------|------------|-----|------|-----|---------|"
    lines = [header, sep]
    for c in sorted(cells, key=lambda c: (c.pair_id, c.direction, c.depth,
                                          c.mode)):
        lines.append(
            f"| {c.pair_id} | {c.direction} | {c.depth} | {c.mode} "
            f"| {_fmt_num(c.alignment)} "
            f"| {_fmt_num(c.delta_hit_donor_vs_null)} "
            f"| {c.coherence_flag:.0f} "
            f"| {c.topk_jaccard_k10_mean:.2f} "
            f"| {c.first_divergence_step} "
            f"| {c.verdict()} |"
        )
    return "\n".join(lines)


def render_pair_verdict(cells: "list[Cell]") -> "tuple[str, str]":
    """Per-pair aggregation: mode-full × depth-{mid_B, after_B} cells
    are the "core" cells (they are the ones both directions define and
    both modes cover). Pair PASSes if the majority of core cells across
    both directions PASS."""
    core = [c for c in cells if c.mode == "full"
            and c.depth in ("mid_B", "after_B")]
    per_pair: "dict[str, list[Cell]]" = defaultdict(list)
    for c in core:
        per_pair[c.pair_id].append(c)

    lines = ["| pair | core cells | pass | fail | caveat | pair verdict |",
             "|------|------------|------|------|--------|--------------|"]
    pair_pass_count = 0
    for pair_id in sorted(per_pair):
        pcells = per_pair[pair_id]
        p = sum(c.cell_passes() for c in pcells)
        f = sum(c.cell_fails() and not c.cell_passes() for c in pcells)
        cav = len(pcells) - p - f
        # A pair passes if majority (> half) of its core cells pass.
        if p > len(pcells) / 2.0:
            pair_verdict = "PASS"
            pair_pass_count += 1
        elif f > len(pcells) / 2.0:
            pair_verdict = "FAIL"
        else:
            pair_verdict = "CAVEAT"
        lines.append(f"| {pair_id} | {len(pcells)} | {p} | {f} | {cav} "
                     f"| {pair_verdict} |")

    n_pairs = len(per_pair)
    if pair_pass_count >= PAIR_PASS_QUORUM:
        model_verdict = "PASS"
    elif pair_pass_count == 0:
        model_verdict = "FAIL"
    else:
        model_verdict = "CAVEAT"

    verdict_line = (f"**Model verdict:** {model_verdict} — "
                    f"{pair_pass_count} of {n_pairs} pairs passed "
                    f"(quorum = {PAIR_PASS_QUORUM}).")
    return "\n".join(lines), verdict_line


def render_summary(cells: "list[Cell]") -> str:
    if not cells:
        return "(no cells found)\n"
    by_model = group_by_model(cells)
    chunks: "list[str]" = []
    for model in sorted(by_model):
        ms = by_model[model]
        chunks.append(f"## {model}\n")
        chunks.append(render_cell_table(ms.cells))
        chunks.append("")
        pair_table, verdict = render_pair_verdict(ms.cells)
        chunks.append("### Per-pair verdict")
        chunks.append(pair_table)
        chunks.append("")
        chunks.append(verdict)
        chunks.append("")
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A0.6 verdict aggregator — reads per-cell JSON dumps",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "a06",
        help="Directory holding per-cell JSON dumps",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If given, write the summary here instead of stdout",
    )
    args = parser.parse_args()

    cells = load_cells(args.results_dir)
    summary = render_summary(cells)

    if args.out:
        args.out.write_text(summary)
        print(f"[verdict_a06] wrote {args.out}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
