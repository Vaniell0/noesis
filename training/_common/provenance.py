"""training/_common/provenance.py — structured provenance for every
corpus pipeline stage (normalize / tokenize / combine).

Generalizes `training/corpus_open/PROVENANCE.md`'s hand-written schema
(source, SHA-256, download date/method, HF repo + license, raw row
count, normalizer script, normalized file + stats, tokenized file +
stats + split method) into an automatic per-artifact sidecar — same
"stamp metadata next to the artifact, regenerate the human-readable
index from it" convention as `experiments._common.results.save_result`
+ `regenerate_results.py`.

`corpus.py`'s combine stage uses this directly (see `save_combined_corpus`)
instead of keeping its own separate `.meta.json` shape.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Optional


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ProvenanceRecord:
    name: str
    stage: str  # "normalize" | "tokenize" | "combine"
    provenance: str  # "generated" | "external-hf" | "external-other"
    origin: str  # HF dataset id / URL (external) or generator script (generated)
    date: str
    script: str
    out_path: str
    out_sha256: Optional[str] = None
    out_size_bytes: Optional[int] = None
    n_rows: Optional[int] = None
    n_tokens: Optional[int] = None
    n_supervised_tokens: Optional[int] = None
    sources: dict = field(default_factory=dict)  # combine stage: per-source breakdown
    extra: dict = field(default_factory=dict)


def make_record(name: str, stage: str, provenance: str, origin: str, script: str,
                 out_path: Path | str, **kwargs) -> ProvenanceRecord:
    return ProvenanceRecord(
        name=name, stage=stage, provenance=provenance, origin=origin,
        date=_date.today().isoformat(), script=script, out_path=str(out_path), **kwargs,
    )


def save_provenance(record: ProvenanceRecord, sidecar_path: Optional[Path | str] = None) -> Path:
    """Write a `<out_path>.provenance.json` sidecar. Computes
    out_sha256/out_size_bytes from the real file on disk if not already
    set — same 'verify SHA-256 before any downstream step' discipline
    PROVENANCE.md already asks for by hand."""
    out_path = Path(record.out_path)
    if record.out_sha256 is None and out_path.exists():
        record.out_sha256 = sha256_of(out_path)
    if record.out_size_bytes is None and out_path.exists():
        record.out_size_bytes = out_path.stat().st_size

    sidecar = Path(sidecar_path) if sidecar_path else Path(str(out_path) + ".provenance.json")
    sidecar.write_text(json.dumps(asdict(record), indent=2))
    return sidecar


def load_provenance(sidecar_path: Path | str) -> ProvenanceRecord:
    data = json.loads(Path(sidecar_path).read_text())
    return ProvenanceRecord(**data)


def iter_provenance(root: Path | str) -> list[tuple[Path, ProvenanceRecord]]:
    """Walk `root` for `*.provenance.json` sidecars — the shared scan
    primitive `regenerate_corpus_index.py` uses. Same shape as
    `experiments._common.results._iter_meta_stamped`, applied to
    sidecar files instead of `_meta`-in-the-blob."""
    root = Path(root)
    found = []
    for p in root.rglob("*.provenance.json"):
        try:
            rec = load_provenance(p)
        except (json.JSONDecodeError, TypeError):
            continue
        found.append((p, rec))
    return sorted(found, key=lambda t: t[0])
