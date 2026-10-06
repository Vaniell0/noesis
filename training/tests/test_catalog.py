"""Tests for training/_common/catalog.py on a throw-away repo layout.

Covers the failure modes found while building the catalog (2026-10-06):
  * two files with one stem silently overwrote each other's record
  * an empty artifact must be flagged, not counted as data
  * a changed artifact must fail verify (sha256), a missing parent must be reported
  * "confirmed" has to come from evidence or result files, never from the name alone

Run:  training/.venv/bin/python training/tests/test_catalog.py     (or pytest)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training._common import catalog  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fake_repo(tmp: Path) -> None:
    catalog.REPO = tmp
    catalog.CATALOG_DIR = tmp / "training" / "catalog"
    catalog.SCAN_ROOTS = ["training/corpus_open", "training/tokenised"]
    catalog._code_cache.clear()
    catalog._result_cache.clear()
    _write_jsonl(tmp / "training/corpus_open/tasks_a.jsonl",
                 [{"id": i, "category": "c", "level": 1, "prompt": f"Q {i}", "answer": "x"} for i in range(5)])
    _write_jsonl(tmp / "training/corpus_open/dup.jsonl", [{"turns": [{"role": "user", "content": "hi"}]}])
    (tmp / "training/tokenised").mkdir(parents=True)
    (tmp / "training/tokenised/dup.pt").write_bytes(b"not a real blob")      # same stem as dup.jsonl
    (tmp / "training/corpus_open/empty_train.jsonl").write_text("")
    _write_jsonl(tmp / "training/corpus_open/evalset.jsonl",
                 [{"id": 1, "category": "c", "level": 1, "prompt": " q   1 ", "answer": "x"},
                  {"id": 2, "category": "c", "level": 1, "prompt": "Other", "answer": "y"}])
    (tmp / "scripts").mkdir()
    (tmp / "scripts/use.py").write_text("DATA = 'training/corpus_open/dup.jsonl'\n")
    (tmp / "experiments/x/results").mkdir(parents=True)
    (tmp / "experiments/x/results/run.json").write_text(
        json.dumps({"args": {"tasks": "training/corpus_open/tasks_a.jsonl"}}))


def test_catalog_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _fake_repo(tmp)
        # the .pt is not a valid blob: it must be recorded as unreadable, not abort the pass
        written = catalog.backfill({"tasks_a": {"role": "train", "provenance": "generated",
                                                "evidence": [{"run": "r", "verdict": "helped", "result": "ok"}]}})
        names = set(written)
        assert {"tasks_a", "empty_train", "evalset", "dup.jsonl", "dup.pt"} <= names, names
        assert catalog.load("dup.pt").status == "unreadable"
        assert any("could not be read" in x for x in catalog.verify("dup.pt"))

        assert catalog.load("empty_train").status == "empty"
        assert catalog.verify("tasks_a") == []

        # a changed artifact fails verify
        p = tmp / "training/corpus_open/tasks_a.jsonl"
        p.write_text(p.read_text() + json.dumps({"id": 9, "category": "c", "level": 1, "prompt": "n", "answer": "z"}) + "\n")
        assert any("size" in x or "sha256" in x for x in catalog.verify("tasks_a"))

        # confirmation states are computed
        assert catalog.usage("tasks_a")["state"] == "measured"
        assert catalog.usage("dup.jsonl")["state"] == "used-unmeasured"      # code mentions it, nothing measured
        assert catalog.usage("evalset")["state"] == "unreferenced"
        assert catalog.usage("empty_train")["state"] == "empty"
        rec = catalog.load("tasks_a"); rec.evidence = []; catalog.save(rec)
        assert catalog.usage("tasks_a")["state"] == "has-results"       # a result file names it in its args

        # contamination check normalises case and whitespace
        _write_jsonl(tmp / "training/corpus_open/tasks_b.jsonl",
                     [{"id": 1, "category": "c", "level": 1, "prompt": "q 1", "answer": "x"}])
        catalog.backfill({}, only="tasks_b")
        ov = catalog.overlap("evalset", "tasks_b")
        assert ov["shared"] == 1 and ov["n_a"] == 2, ov


def test_personal_data_is_refused_and_inherited():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _fake_repo(tmp)
        (tmp / "training/tokenised/dup.pt").unlink()
        catalog.backfill({"tasks_a": {"sensitivity": "personal"},
                          "evalset": {"parents": ["tasks_a"]}})
        assert catalog.is_personal("tasks_a") and catalog.is_personal("evalset")   # child inherits
        assert not catalog.is_personal("dup")
        try:
            list(catalog.rows("evalset"))
            raise AssertionError("personal-derived rows must be refused")
        except PermissionError:
            pass
        assert len(list(catalog.rows("evalset", allow_personal=True))) == 2


def test_names_do_not_collide():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _fake_repo(tmp)
        (tmp / "training/tokenised/dup.pt").unlink()
        (tmp / "training/tokenised/dup.pt").write_bytes(b"x")
        got = catalog.assign_names([tmp / "training/corpus_open/dup.jsonl", tmp / "training/tokenised/dup.pt",
                                    tmp / "training/corpus_open/tasks_a.jsonl"])
        vals = sorted(got.values())
        assert vals == ["dup.jsonl", "dup.pt", "tasks_a"], vals


def test_format_detection():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fname, row, want in [
            ("a.jsonl", {"turns": []}, "rollouts"),
            ("b.jsonl", {"id": 1, "category": "c", "level": 1, "answer": "x", "prompt": "p"}, "tasks"),
            ("c.jsonl", {"arm": "recall", "n_pairs": 4, "gap_words": 50, "prompt": "p", "answer": "1"}, "recall"),
            ("d.jsonl", {"user": "u", "think": "t", "answer": "a"}, "think"),
            ("e.jsonl", {"text": "t"}, "text"),
        ]:
            _write_jsonl(tmp / fname, [row])
            assert catalog.detect_format(tmp / fname) == want, (fname, catalog.detect_format(tmp / fname))


if __name__ == "__main__":
    for fn in (test_catalog_end_to_end, test_personal_data_is_refused_and_inherited,
               test_names_do_not_collide, test_format_detection):
        fn()
        print("ok", fn.__name__)
