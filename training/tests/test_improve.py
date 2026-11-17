"""The improver's weighting rules on a small grid (no repo, no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training._common.improve import plan  # noqa: E402


def _by_cell(rows):
    return {tuple(r["cell"].values()): r for r in rows}


def test_weights_follow_learnability_and_frontier():
    fields = ["view", "length", "pairs_shown"]
    cells = {("caesar", 6, 1): [8, 2],     # zone: 1 - 2/8
             ("caesar", 12, 1): [8, 0],    # dead neighbour on length, gap 6: the midpoint 9 is the edge
             ("caesar", 6, 0): [8, 0],     # dead neighbour on pairs_shown, gap 1: itself gets explore
             ("caesar", 6, 2): [8, 8],     # solved: weight 0; at the end of the grid: one step past (3)
             ("caesar", 12, 2): [8, 0]}
    p = _by_cell(plan(fields, cells, explore=0.1))
    assert p[("caesar", 6, 1)]["weight"] == 0.75
    assert p[("caesar", 6, 2)]["weight"] == 0.0
    assert p[("caesar", 9, 1)]["weight"] == 0.1 and p[("caesar", 9, 1)]["zone"] == "new"
    assert p[("caesar", 6, 0)]["weight"] == 0.1 and p[("caesar", 6, 0)]["zone"] == "dead"
    assert p[("caesar", 6, 3)]["weight"] == 0.1                  # past solved pairs_shown=2
    assert p[("caesar", 12, 1)]["weight"] == 0.0                 # dead, not adjacent to a live cell on one knob
    assert ("caesar", 9, 2) in p                                 # edge between solved 6 and dead 12 at pairs 2


def test_nothing_live_gives_no_weight():
    p = plan(["length"], {(6,): [8, 0], (12,): [8, 0]})
    assert all(r["weight"] == 0 for r in p)
