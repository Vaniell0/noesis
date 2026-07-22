"""A0.7 tier-1 verdict aggregator.

Reads A0.7 tier-1 per-cell JSON dumps from ``results/a07_tier1/`` and
compares them against the A0.6 same-checkpoint baseline in
``results/a06/`` to apply the tier-1 verdict rule (README):

    PASS: alignment ≤ -0.30 with coherence_flag = 1 at rate
          > 50 % of the same-checkpoint (A0.6) baseline for that pair.
    FAIL: alignment ≥ 0 or coherence_flag = 0 at rate < 20 %.
    CAVEAT: 20–50 % — needs per-layer analysis.

"Same-checkpoint baseline for that pair" is A0.6's per-pair pass
count on the recipient checkpoint. If the recipient is world-0.4b,
the baseline is A0.6's pass count on world-0.4b for that pair. If the
recipient is g1d-0.4b, ditto on g1d-0.4b.

This ratio framing is the point of A0.7: cross-checkpoint state
transfer is expected to be *degraded* vs same-checkpoint; the
question is by how much. > 50% means the transfer preserves majority
of the signal; < 20% means the transfer effectively destroys it;
20–50% is the CAVEAT zone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from verdict_a06 import (
    ALIGNMENT_PASS_THRESHOLD,
    LEX_HIT_DELTA_PASS_THRESHOLD,
    Cell as A06Cell,
    load_cells as load_a06_cells,
)


PASS_RATIO_THRESHOLD = 0.50
CAVEAT_RATIO_THRESHOLD = 0.20


@dataclass
class A07Cell:
    path: Path
    donor_ckpt: str
    recipient_ckpt: str
    pair_id: str
    prompt_direction: str
    depth: str
    mode: str
    alignment: float
    delta_hit_donor_vs_null: float
    coherence_flag: float
    first_divergence_step: int
    topk_jaccard_k10_mean: float

    def core(self) -> bool:
        return self.mode == "full" and self.depth in ("mid_B", "after_B")

    def cell_passes(self) -> bool:
        return (self.alignment <= ALIGNMENT_PASS_THRESHOLD
                and self.delta_hit_donor_vs_null > LEX_HIT_DELTA_PASS_THRESHOLD
                and self.coherence_flag >= 1.0)

    def cell_fails(self) -> bool:
        return (self.alignment >= 0
                or self.delta_hit_donor_vs_null <= LEX_HIT_DELTA_PASS_THRESHOLD)


def load_a07_cells(results_dir: Path) -> "list[A07Cell]":
    cells: "list[A07Cell]" = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        with p.open() as f:
            d = json.load(f)
        cfg = d["config"]
        cells.append(A07Cell(
            path=p,
            donor_ckpt=cfg["donor_ckpt"],
            recipient_ckpt=cfg["recipient_ckpt"],
            pair_id=cfg["pair_id"],
            prompt_direction=cfg["prompt_direction"],
            depth=cfg["depth"],
            mode=cfg["mode"],
            alignment=float(d["alignment"]["alignment"]),
            delta_hit_donor_vs_null=float(
                d["lexicon"]["delta_hit_donor_vs_null"]),
            coherence_flag=float(d["surface_garble"]["cross"]["coherence_flag"]),
            first_divergence_step=(
                int(d["first_divergence_step"])
                if d["first_divergence_step"] is not None
                else -1
            ),
            topk_jaccard_k10_mean=float(d["topk_jaccard"]["k10_mean"]),
        ))
    return cells


def a06_core_pass_count(cells: "list[A06Cell]",
                        recipient_ckpt: str,
                        pair_id: str) -> "tuple[int, int]":
    """A06 baseline: (pass, total) among core cells (mode=full,
    depth ∈ {mid_B, after_B}) for the given pair on the recipient
    checkpoint."""
    subset = [c for c in cells
              if c.model == recipient_ckpt
              and c.pair_id == pair_id
              and c.mode == "full"
              and c.depth in ("mid_B", "after_B")]
    return sum(c.cell_passes() for c in subset), len(subset)


def _fmt_num(x: float, prec: int = 3) -> str:
    return f"{x:+.{prec}f}" if x < 0 or x > 0 else f"{0.0:+.{prec}f}"


def render_summary(a07_cells: "list[A07Cell]",
                   a06_cells: "list[A06Cell]") -> str:
    if not a07_cells:
        return "(no A0.7 cells found)\n"

    # Group A07 by (donor_ckpt, recipient_ckpt, pair_id).
    groups: "dict[tuple, list[A07Cell]]" = defaultdict(list)
    for c in a07_cells:
        groups[(c.donor_ckpt, c.recipient_ckpt, c.pair_id)].append(c)

    # Per-cell table (all cells, all groups).
    header = ("| donor→recipient | pair | pdir | depth | mode | alignment "
              "| Δhit_donor | coh | k10J | fds |")
    sep = ("|-----------------|------|------|-------|------|-----------"
           "|------------|-----|------|-----|")
    lines = ["## Per-cell", header, sep]
    for c in sorted(a07_cells, key=lambda c: (c.donor_ckpt, c.recipient_ckpt,
                                              c.pair_id, c.prompt_direction,
                                              c.depth, c.mode)):
        arrow = f"{c.donor_ckpt}→{c.recipient_ckpt}"
        lines.append(
            f"| {arrow} | {c.pair_id} | {c.prompt_direction} | {c.depth} | "
            f"{c.mode} | {_fmt_num(c.alignment)} | "
            f"{_fmt_num(c.delta_hit_donor_vs_null)} | "
            f"{c.coherence_flag:.0f} | {c.topk_jaccard_k10_mean:.2f} | "
            f"{c.first_divergence_step} |"
        )

    lines.append("")
    lines.append("## Per-group verdict (ratio vs A0.6 baseline)")
    lines.append("| donor→recipient | pair | a07 core pass "
                 "| a07 core total | a06 baseline pass | ratio | verdict |")
    lines.append("|-----------------|------|---------------|----------------"
                 "|-------------------|-------|---------|")
    overall_pass = 0
    overall_total = 0
    for (donor, recipient, pair_id), cells in sorted(groups.items()):
        core = [c for c in cells if c.core()]
        a07_pass = sum(c.cell_passes() for c in core)
        a07_total = len(core)
        base_pass, base_total = a06_core_pass_count(a06_cells, recipient,
                                                    pair_id)
        if base_pass == 0:
            ratio_str = "n/a"
            verdict = ("N/A" if base_total == 0
                       else ("PASS" if a07_pass > 0 else "FAIL"))
        else:
            ratio = a07_pass / base_pass
            ratio_str = f"{ratio:.2f}"
            if ratio > PASS_RATIO_THRESHOLD:
                verdict = "PASS"
            elif ratio < CAVEAT_RATIO_THRESHOLD:
                verdict = "FAIL"
            else:
                verdict = "CAVEAT"
        if verdict == "PASS":
            overall_pass += 1
        overall_total += 1
        lines.append(
            f"| {donor}→{recipient} | {pair_id} | {a07_pass} | {a07_total} "
            f"| {base_pass} | {ratio_str} | {verdict} |"
        )
    lines.append("")
    lines.append(f"**Overall:** {overall_pass} of {overall_total} groups "
                 f"passed the ratio bar.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A0.7 tier-1 verdict aggregator",
    )
    parser.add_argument(
        "--a07-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "a07_tier1",
    )
    parser.add_argument(
        "--a06-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "a06",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    a07_cells = load_a07_cells(args.a07_dir)
    a06_cells = load_a06_cells(args.a06_dir)
    summary = render_summary(a07_cells, a06_cells)

    if args.out:
        args.out.write_text(summary)
        print(f"[verdict_a07] wrote {args.out}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
