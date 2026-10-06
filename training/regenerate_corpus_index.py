#!/usr/bin/env python3
"""Rebuild the auto-generated sections of `training/corpus_open/PROVENANCE.md`
and `docs/training-pipeline.md`, from the dataset catalog (`training/catalog/*.json`)
plus any `.provenance.json` sidecars written by migrated stages.

    training/.venv/bin/python training/regenerate_corpus_index.py

Same marker-splice discipline as `experiments/regenerate_results.py` /
`regenerate_hyp_index.py`: everything above the AUTO-GENERATED marker is
hand-written and left untouched; everything below it is rebuilt from
`training._common.provenance.iter_provenance` sidecars each run.

Scan is deliberately scoped to `training/corpus_open/` and
`training/tokenised/` only — never `training/corpus/` or
`training/sanitised/`. Those hold personal Claude CLI trace data
explicitly reclassified OUT of the training path (see the
`RECLASSIFIED.md` in each) to satisfy the "no personal corpus in
weights" constraint; indexing them into a training-facing doc would
undermine the whole point of the reclassification, so this scan never
looks there.

Two tables:
  - PROVENANCE.md gets normalize/tokenize-stage artifacts under
    corpus_open/ (per-dataset entries — mirrors the existing hand-written
    format's fields).
  - training-pipeline.md gets combine-stage artifacts under tokenised/
    (which combined corpora exist, from what sources, at what fractions).

As of this writing nothing has actually been produced through the new
`training._common` pipeline yet (only `normalize_hh_rlhf.py` is
migrated onto the registry, and it hasn't been re-run) — both tables
render as explicit "(none yet)" rather than silently omitting the
section, same as regenerate_hyp_index.py's un-reviewed-hypothesis rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training._common.provenance import ProvenanceRecord, iter_provenance  # noqa: E402

_CORPUS_OPEN = _REPO_ROOT / "training" / "corpus_open"
_TOKENISED = _REPO_ROOT / "training" / "tokenised"
_PROVENANCE_MD = _CORPUS_OPEN / "PROVENANCE.md"
_PIPELINE_MD = _REPO_ROOT / "docs" / "training-pipeline.md"

_MARKER = "<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python training/regenerate_corpus_index.py` -->"


def _splice(out_path: Path, body: str) -> None:
    text = out_path.read_text() if out_path.exists() else f"# {out_path.stem}\n"
    if _MARKER in text:
        head = text.split(_MARKER)[0].rstrip() + "\n\n"
    else:
        head = text.rstrip() + "\n\n"
    out_path.write_text(head + _MARKER + "\n\n" + body + "\n")


def _rel(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return path_str


def build_dataset_table(records: list[ProvenanceRecord]) -> str:
    """One row per catalogued dataset; `state` is computed (training/_common/catalog.py usage)."""
    from training._common import catalog
    rows = [r for r in records if r.stage != "combine"]
    lines = [
        "| Name | Role | Format | Rows | Provenance | Origin | State | Evidence | SHA-256 |",
        "|------|------|--------|------|------------|--------|-------|----------|---------|",
    ]
    if not rows:
        lines.append("| — | — | — | — | — | *(catalog empty — run `training/datasets.py backfill`)* | — | — | — |")
        return "\n".join(lines)
    for r in sorted(rows, key=lambda r: (r.role, r.name)):
        sha = f"`{r.out_sha256[:12]}…`" if r.out_sha256 else "—"
        n = r.n_rows if r.n_rows is not None else "—"
        ev = ", ".join(sorted({e.get("verdict", "?") for e in r.evidence})) or "—"
        lines.append(f"| {r.name} | {r.role or '—'} | {r.format or '—'} | {n} | {r.provenance} | "
                     f"{r.origin or '—'} | {catalog.usage(r.name)['state']}{' (personal)' if catalog.is_personal(r.name) else ''} | {ev} | {sha} |")
    return "\n".join(lines)


def build_combine_table(records: list[ProvenanceRecord]) -> str:
    rows = [r for r in records if r.stage == "combine"]
    lines = [
        "| Name | Sources (fraction) | Date | Tokens | Consumed by | Out |",
        "|------|---------------------|------|--------|-------------|-----|",
    ]
    if not rows:
        lines.append("| — | *(none yet — nothing combined through `training.build_corpus` so far)* | — | — | — | — |")
        return "\n".join(lines)
    for r in sorted(rows, key=lambda r: (r.name, r.date)):
        src_str = ", ".join(
            f"{name}({meta.get('fraction_target', '—')})" for name, meta in (r.sources or {}).items()
        ) or (", ".join(r.parents) if r.parents else "—")
        consumed = r.extra.get("consumed_by", "—") if r.extra else "—"
        lines.append(f"| {r.name} | {src_str} | {r.date} | {r.n_tokens or '—'} | {consumed} | `{_rel(r.out_path)}` |")
    return "\n".join(lines)


def main() -> None:
    from training._common import catalog
    records = catalog.all_records()
    # sidecars written by the migrated stages (build_corpus) still count
    seen = {r.name for r in records}
    for root in (_CORPUS_OPEN, _TOKENISED):
        records += [rec for _, rec in iter_provenance(root) if rec.name not in seen]
    dataset_records = records
    tokenised_records = records

    _splice(_PROVENANCE_MD, build_dataset_table(dataset_records))
    print(f"[regenerate_corpus_index] wrote auto section -> {_PROVENANCE_MD.relative_to(_REPO_ROOT)}")

    _splice(_PIPELINE_MD, build_combine_table(tokenised_records))
    print(f"[regenerate_corpus_index] wrote auto section -> {_PIPELINE_MD.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
