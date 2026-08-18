"""Unit tests for experiments/_common's core modules themselves.

`test_registry.py` guards against probes conflicting with each other;
this guards the plumbing they all sit on (`layers.py`, `results.py`).
Same style deliberately — plain assertions, one `main()`, no pytest —
this repo has no existing test convention to fit into or fork from.

Run: `python experiments/_common/test_core.py`
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.layers import default_layers, validate_layers
from experiments._common.results import save_result, regenerate_index, _iter_meta_stamped, _AUTO_MARKER


def test_default_layers_basic():
    assert default_layers(32) == sorted(set(default_layers(32)))  # sorted, dedup'd
    assert min(default_layers(32)) == 0
    assert max(default_layers(32)) == 31
    assert default_layers(24)[-1] == 23  # never picks an out-of-range index


def test_default_layers_small_model():
    # A model too shallow for 5 distinct fractional depths should dedup,
    # not crash or pad with out-of-range indices.
    layers = default_layers(1)
    assert layers == [0]
    layers = default_layers(4)
    assert all(0 <= l < 4 for l in layers)


def test_default_layers_rejects_nonpositive():
    for bad in (0, -1):
        try:
            default_layers(bad)
            raise AssertionError(f"default_layers({bad}) should have raised")
        except ValueError:
            pass


def test_validate_layers_catches_out_of_range():
    validate_layers([0, 5, 9], n_layer=10)  # should not raise
    try:
        validate_layers([0, 5, 12], n_layer=10)
        raise AssertionError("validate_layers should have raised on layer 12 >= n_layer=10")
    except ValueError as e:
        assert "12" in str(e)


def test_save_result_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "sub" / "result.json"
        save_result(
            out, {"a": 1, "_summary": {"x": "1.0"}},
            experiment="test_probe", hypothesis=["H8"], status="done",
            model="fake-model", script="fake_script.py",
        )
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["_meta"]["experiment"] == "test_probe"
        assert payload["_meta"]["hypothesis"] == ["H8"]
        assert payload["_meta"]["summary"] == {"x": "1.0"}
        assert "_summary" not in payload  # stripped from the stored data, it's metadata
        assert payload["a"] == 1
        # companion markdown written alongside
        assert out.with_suffix(".md").exists()


def test_regenerate_index_finds_stamped_and_skips_plain():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        save_result(root / "real" / "r.json", {"_summary": {"m": "1"}},
                    experiment="e1", hypothesis=["H8"], model="m", script="s.py")
        (root / "plain.json").write_text(json.dumps({"just": "data"}))  # no _meta
        (root / "not_json.txt").write_text("hello")

        found = _iter_meta_stamped(root)
        assert len(found) == 1, f"expected exactly 1 stamped file, found {len(found)}"

        out_md = root / "RESULTS.md"
        n = regenerate_index(root=root, out_path=out_md)
        assert n == 1, f"expected 1 row, got {n}"
        text = out_md.read_text()
        # The table has no "experiment name" column (only H/Model/Date/
        # Status/Metric/Value/Result/Code) — check for what's actually
        # rendered, not the experiment= label passed to save_result.
        assert "not_json" not in text and "plain" not in text, \
            "non-_meta-stamped files must not appear in the index"
        auto_section = text.split(_AUTO_MARKER)[1] if _AUTO_MARKER in text else text
        assert "1" in auto_section, f"expected the summary value to appear, got:\n{text}"


def test_regenerate_index_flags_non_done_status():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        save_result(root / "ext.json", {"_summary": {"m": "1"}},
                    experiment="e2", hypothesis=["H8"], model="m", script="s.py",
                    status="external-reported")
        out_md = root / "RESULTS.md"
        regenerate_index(root=root, out_path=out_md)
        text = out_md.read_text()
        assert "external-reported" in text, (
            "non-'done' status must show up in the table even when a summary "
            "exists — regression check for the 2026-08-18 fix"
        )


def main() -> int:
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
