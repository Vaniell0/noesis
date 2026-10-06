"""training/_common/catalog.py — one place that knows every dataset.

Why this exists (2026-10-06): data was produced by ~20 scripts in four
families (SFT normalizers, procedural generators, RL task sets, probe items)
and lived under training/corpus_open, training/tokenised, experiments/*/ with
nothing recording how a file was made, which script reads it, or whether it is
still the file a result was measured on. Two files were empty (sr_shared*_train)
and the generator command of the P1 corpora existed only in a chat transcript.

Design rules:
  * A catalog record is a `ProvenanceRecord` (same schema every stage already
    writes) stored as `training/catalog/<name>.json` — small, tracked in git.
    The artifact itself stays untracked (it is large; training/tokenised/ and
    corpus_open/*.json* are gitignored) and is located by `out_path` and
    checked by `out_sha256`. A fresh VM can therefore `verify` what it
    regenerated or copied against what the results were measured on.
  * `parents` give lineage (raw -> jsonl -> pt -> combined -> run), `recipe`
    gives the command that regenerates it, `verifier` names the checker that
    decides correct — the same checker serves reward, teacher filtering and eval.
  * One reader, `rows(name)`, for every format, so a probe or trainer does not
    need to know whether it is looking at a rollouts jsonl, a task set or a
    packed .pt blob.
  * Personal Claude-CLI trace data (training/corpus, training/sanitised) is
    never scanned — it was reclassified out of the training path.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from training._common.provenance import (
    ProvenanceRecord, load_provenance, make_record, sha256_of,
)

REPO = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPO / "training" / "catalog"

# Roots the scanner looks in. Deliberately NOT training/corpus or
# training/sanitised (personal traces, reclassified out of training).
SCAN_ROOTS = [
    "training/corpus_open",
    "training/tokenised",
    "experiments/A0_eval",
    "experiments/A0_portability",
    "experiments/A0_H12a_working_memory/tasks",
    "experiments/aporia_probe",
    "experiments/premise_validator",
    "experiments/attribution_probe",
]
_SCAN_SUFFIXES = (".jsonl", ".pt")
_SKIP_PARTS = {"results", "__pycache__", "runs", "v3_29b", "distinctness"}
_SKIP_NAME = re.compile(r"(partial|results|loss_data|provenance|head\.pt)")

FORMATS = {
    "pt-blob":  "packed token blob: ids / loss_mask / state_mask / starts",
    "rollouts": "turns: [{role, content}] per row",
    "think":    "id / system / user / think / answer per row",
    "tasks":    "id / category / level / prompt / answer / rubric per row",
    "recall":   "staged-recall rows: arm / n_pairs / gap_words / prompt / answer",
    "items":    "probe items: id / category / prompt (+ alternatives or gold)",
    "text":     "one flat training string per row: {text}",
    "json":     "single JSON document",
    "unknown":  "not recognised",
}


def detect_format(path: Path) -> str:
    if path.suffix == ".pt":
        return "pt-blob"
    if path.suffix == ".json":
        return "json"
    try:
        with open(path) as f:
            d = json.loads(f.readline())
    except Exception:
        return "unknown"
    if not isinstance(d, dict):
        return "unknown"
    k = set(d)
    if "turns" in k:
        return "rollouts"
    if {"think", "answer"} <= k and ("user" in k or "prompt" in k):
        return "think"
    if {"arm", "n_pairs", "gap_words"} <= k:
        return "recall"
    if {"category", "level", "answer"} <= k:
        return "tasks"
    if "prompt" in k or "category" in k:
        return "items"
    if k == {"text"}:
        return "text"
    return "unknown"


def name_for(path: Path) -> str:
    """Catalog name: file stem, prefixed by the experiment directory when the
    file lives under experiments/ (tasks.jsonl and items.jsonl exist several times)."""
    parts = path.resolve().relative_to(REPO).parts
    if parts[0] == "experiments":
        return f"{parts[1]}/{path.stem}"
    return path.stem


def assign_names(paths: list[Path]) -> dict[Path, str]:
    """Unique catalog names. Two files can share a stem (aporia_train.jsonl and
    aporia_train.pt are the normalized text and its tokenised blob): every member of
    a clashing group gets its extension appended, so neither record overwrites the other."""
    base = {p: name_for(p) for p in paths}
    groups: dict[str, list[Path]] = {}
    for p, n in base.items():
        groups.setdefault(n, []).append(p)
    return {p: (f"{n}{p.suffix}" if len(groups[n]) > 1 else n) for p, n in base.items()}


def record_path(name: str) -> Path:
    return CATALOG_DIR / (name.replace("/", "__") + ".json")


def save(record: ProvenanceRecord) -> Path:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    p = record_path(record.name)
    p.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False) + "\n")
    return p


def load(name: str) -> ProvenanceRecord:
    p = record_path(name)
    if not p.exists():
        raise KeyError(f"no catalog record {name!r} — {len(names())} known, see `datasets.py ls`")
    return load_provenance(p)


def names() -> list[str]:
    if not CATALOG_DIR.exists():
        return []
    return sorted(load_provenance(p).name for p in CATALOG_DIR.glob("*.json"))


def all_records() -> list[ProvenanceRecord]:
    if not CATALOG_DIR.exists():
        return []
    return [load_provenance(p) for p in sorted(CATALOG_DIR.glob("*.json"))]


def artifact_path(rec: ProvenanceRecord) -> Path:
    if not rec.out_path:
        raise FileNotFoundError(f"{rec.name}: not on disk (status {rec.status}); source: {rec.url or rec.origin}")
    p = Path(rec.out_path)
    return p if p.is_absolute() else REPO / p


# ---------------------------------------------------------------- describing

def _count_jsonl(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _pt_stats(path: Path) -> dict:
    import torch
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(blob, dict) or "ids" not in blob:
        return {}
    out = {"n_tokens": int(blob["ids"].numel())}
    if "starts" in blob:
        out["n_rows"] = max(0, int(blob["starts"].numel()) - 1)
    if "loss_mask" in blob:
        out["n_supervised_tokens"] = int(blob["loss_mask"].sum())
    if "state_mask" in blob:
        out["n_state_tokens"] = int(blob["state_mask"].sum())
    if "vocab" in blob:
        out["vocab"] = str(blob["vocab"])
    return out


_PROFILE_KEYS = ("arm", "category", "level", "n_pairs", "gap_words", "vocab", "key_style", "mode", "view",
                 "task_type", "source", "protocol")
_PROFILE_MAX_BYTES = 400_000_000


def _profile_jsonl(path: Path) -> dict:
    """Counts of every low-cardinality field among _PROFILE_KEYS (measured, not declared)."""
    from collections import Counter
    cnt = {k: Counter() for k in _PROFILE_KEYS}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            for k in _PROFILE_KEYS:
                if k in d and not isinstance(d[k], (dict, list)):
                    cnt[k][str(d[k])] += 1
    return {k: dict(sorted(c.items())) for k, c in cnt.items() if 0 < len(c) <= 64}


def describe(path: Path, hash_it: bool = True) -> dict:
    """Measured facts about an artifact on disk (never guessed)."""
    size = path.stat().st_size
    d = {"out_size_bytes": size, "format": detect_format(path)}
    if hash_it and size:
        d["out_sha256"] = sha256_of(path)
    if size == 0:
        d["n_rows"] = 0
    elif path.suffix == ".jsonl":
        d["n_rows"] = _count_jsonl(path)
        if d["format"] in ("tasks", "recall", "think", "items") and size < _PROFILE_MAX_BYTES:
            d["profile"] = _profile_jsonl(path)
    elif path.suffix == ".pt":
        d.update(_pt_stats(path))
    return d


def scan(roots: Optional[list[str]] = None) -> list[Path]:
    found = []
    for r in roots or SCAN_ROOTS:
        base = REPO / r
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if (p.is_file()
                    and (p.suffix in _SCAN_SUFFIXES or (p.suffix == ".json" and "corpus_open" in p.parts))
                    and not (set(p.relative_to(base).parts[:-1]) & _SKIP_PARTS)
                    and not _SKIP_NAME.search(p.name)):
                found.append(p)
    return found


# ---------------------------------------------------------------- reading

def is_personal(name: str, _seen: Optional[set] = None) -> bool:
    """True if this record, or any ancestor in `parents`, is marked sensitivity=personal."""
    _seen = _seen if _seen is not None else set()
    if name in _seen or not record_path(name).exists():
        return False
    _seen.add(name)
    rec = load(name)
    return rec.sensitivity == "personal" or any(is_personal(p, _seen) for p in rec.parents)


def rows(name: str, limit: Optional[int] = None, allow_personal: bool = False) -> Iterator[dict]:
    """Uniform row reader. jsonl formats yield the stored dict; pt blobs yield
    {"ids","loss_mask","state_mask"} tensors per rollout (mmap, no full load).
    Data derived from the owner's own sessions is refused unless allow_personal=True."""
    if not allow_personal and is_personal(name):
        raise PermissionError(f"{name}: derived from personal session data; pass allow_personal=True "
                              f"to read it deliberately")
    rec = load(name)
    path = artifact_path(rec)  # raises for planned sources
    if not path.exists():
        raise FileNotFoundError(f"{name}: artifact missing at {path} — recipe: {rec.recipe or 'not recorded'}")
    if path.suffix == ".pt":
        import torch
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        starts = blob["starts"].tolist()
        sm = blob.get("state_mask")
        for i in range(len(starts) - 1):
            if limit is not None and i >= limit:
                return
            s, e = starts[i], starts[i + 1]
            yield {"ids": blob["ids"][s:e], "loss_mask": blob["loss_mask"][s:e],
                   "state_mask": sm[s:e] if sm is not None else None}
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- verifying

def verify(name: str, check_hash: bool = True) -> list[str]:
    """Problems found (empty list = the artifact is what the record says)."""
    rec = load(name)
    if rec.status == "planned":
        return []  # nothing on disk yet by design; becomes a real check once downloaded
    path = artifact_path(rec)
    if not path.exists():
        return [f"missing: {path}"]
    problems = []
    size = path.stat().st_size
    if rec.out_size_bytes is not None and size != rec.out_size_bytes:
        problems.append(f"size {size} != recorded {rec.out_size_bytes}")
    if size == 0 and rec.status != "empty":
        problems.append("file is empty but record status is not 'empty'")
    if rec.status == "unreadable":
        problems.append("artifact could not be read when it was catalogued: " + rec.notes[-160:])
    if check_hash and rec.out_sha256 and size and sha256_of(path) != rec.out_sha256:
        problems.append("sha256 differs from record (regenerated or modified)")
    for parent in rec.parents:
        if not record_path(parent).exists():
            problems.append(f"parent {parent!r} has no catalog record")
    twins = [r.name for r in all_records() if rec.out_path and r.out_path == rec.out_path and r.name != rec.name]
    if twins:
        problems.append(f"same artifact path recorded under another name: {twins}")
    return problems


def lineage(name: str) -> list[str]:
    """Names from the root source down to `name` (first-parent path)."""
    chain, seen = [name], {name}
    while True:
        rec = load(chain[0])
        if not rec.parents or rec.parents[0] in seen or not record_path(rec.parents[0]).exists():
            if rec.parents and rec.parents[0] not in seen:
                chain.insert(0, rec.parents[0])  # unrecorded root, still show it
            return chain
        chain.insert(0, rec.parents[0])
        seen.add(rec.parents[0])


_CODE_SUFFIXES = (".py", ".sh", ".yaml", ".yml")
_WALK_SKIP = {".venv", ".git", "__pycache__", "models", "checkpoints_backup", "_vm_backup",
              "sanitised", "raw", "catalog"}


_code_cache: dict = {}


def _code_texts() -> dict:
    if "t" not in _code_cache:
        texts = {}
        for dp, dn, fn in os.walk(REPO):
            dn[:] = [d for d in dn if d not in _WALK_SKIP and not d.startswith("result")]
            for f in fn:
                if f.endswith(_CODE_SUFFIXES):
                    p = Path(dp) / f
                    try:
                        texts[str(p.relative_to(REPO))] = p.read_text(errors="ignore")
                    except OSError:
                        pass
        _code_cache["t"] = texts
    return _code_cache["t"]


def consumers(name: str) -> list[str]:
    """Code/config files in the repo that mention the artifact's file name."""
    if not load(name).out_path:
        return []
    base = Path(load(name).out_path).name
    return sorted(p for p, t in _code_texts().items() if base in t)


# ---------------------------------------------------------------- backfill

def backfill(hints: dict, hash_it: bool = True, only: Optional[str] = None,
             overwrite: bool = False) -> list[str]:
    """Create catalog records for every artifact already on disk.

    `hints` maps a catalog name (or "re:<regex>" over the name) to the facts a
    scan cannot see: role, provenance, origin, script, parents, recipe,
    verifier, status, notes. Anything not hinted is recorded with provenance
    "unknown" and the missing facts left blank — an honest gap, not a guess.
    Existing records are kept unless overwrite=True, so hand edits survive.
    """
    written = []
    all_paths = scan()
    assigned = assign_names(all_paths)
    for path in all_paths:
        name = assigned[path]
        if only and not re.search(only, name):
            continue
        if record_path(name).exists() and not overwrite:
            continue
        facts: dict = {}
        for key, val in hints.items():
            if key.startswith("re:"):
                if re.fullmatch(key[3:], name):
                    facts.update(val)
            elif key == name:
                facts.update(val)
        try:
            meas = describe(path, hash_it=hash_it)
            unreadable = ""
        except Exception as e:  # one broken artifact must not abort the whole pass
            meas = {"out_size_bytes": path.stat().st_size, "format": detect_format(path)}
            unreadable = f"{type(e).__name__}: {str(e)[:160]}"
        rec = make_record(
            name=name, stage=facts.pop("stage", "raw"),
            provenance=facts.pop("provenance", "unknown"),
            origin=facts.pop("origin", ""), script=facts.pop("script", ""),
            out_path=str(path.relative_to(REPO)),
            out_sha256=meas.get("out_sha256"), out_size_bytes=meas["out_size_bytes"],
            n_rows=meas.get("n_rows"), n_tokens=meas.get("n_tokens"),
            n_supervised_tokens=meas.get("n_supervised_tokens"),
            profile=meas.get("profile", {}), evidence=facts.pop("evidence", []),
            format=facts.pop("format", meas["format"]),
            role=facts.pop("role", ""), status=facts.pop("status", "live"),
            sensitivity=facts.pop("sensitivity", ""),
            parents=facts.pop("parents", []), recipe=facts.pop("recipe", {}),
            verifier=facts.pop("verifier", ""), notes=facts.pop("notes", ""),
            extra={**facts, **{x: meas[x] for x in ("vocab", "n_state_tokens") if x in meas}},
        )
        if path.suffix == ".pt" and not rec.parents:
            twin = name[:-3] + ".jsonl" if name.endswith(".pt") else None
            if twin and twin in assigned.values():
                rec.parents = [twin]
        if meas.get("n_rows") == 0:
            rec.status = "empty"
        if unreadable:
            rec.status = "unreadable"
            rec.notes = (rec.notes + " " if rec.notes else "") + f"unreadable at backfill: {unreadable}"
        # the artifact's own date (file mtime), not the day the record was written
        from datetime import datetime
        rec.date = datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
        save(rec)
        written.append(name)
    return written


# ---------------------------------------------------------------- usage / confirmation

_RESULT_ROOTS = ["experiments", "training/runs"]
_RESULT_MAX_BYTES = 8_000_000
_result_cache: dict = {}


def _result_files() -> list[tuple[str, str]]:
    """(relative path, text head) for every result JSON/MD that records its inputs."""
    if _result_cache:
        return _result_cache["files"]
    out = []
    for root in _RESULT_ROOTS:
        for dp, dn, fn in os.walk(REPO / root):
            dn[:] = [d for d in dn if d not in {".venv", "__pycache__", "checkpoints"}]
            if "results" not in Path(dp).parts and not Path(dp).name.startswith("results"):
                continue
            for f in fn:
                if f.endswith((".json", ".md")) and "partial" not in f:
                    p = Path(dp) / f
                    try:
                        if p.stat().st_size <= _RESULT_MAX_BYTES:
                            out.append((str(p.relative_to(REPO)), p.read_text(errors="ignore")[:20000]))
                    except OSError:
                        pass
    _result_cache["files"] = out
    return out


def results_using(name: str) -> list[str]:
    """Result files whose recorded arguments / header name this artifact's path.
    This is the machine-readable form of "what was actually run on this data"."""
    rec = load(name)
    if not rec.out_path:
        return []
    rel = str(Path(rec.out_path))
    return sorted(p for p, head in _result_files() if rel in head)


def usage(name: str) -> dict:
    """How well is this dataset confirmed? Computed, never declared:
    measured         curated evidence (a verdict with a number and a source)
    has-results      result files exist whose recorded args name this file, no curated verdict yet
    used-unmeasured  code or config mentions the file, no result names it
    unreferenced     nothing mentions the file name (paths built in code or passed on the
                     command line are invisible here — it is a lead, not a verdict)"""
    rec = load(name)
    cons = consumers(name)
    res = results_using(name)
    verdicts = [e.get("verdict", "") for e in rec.evidence]
    if rec.status in ("superseded", "reclassified", "empty", "scratch", "unreadable", "planned"):
        state = rec.status
    elif rec.evidence:
        state = "measured"
    elif res:
        state = "has-results"
    elif cons:
        state = "used-unmeasured"
    else:
        state = "unreferenced"
    return {"name": name, "state": state, "consumers": cons, "results": res,
            "verdicts": verdicts, "n_evidence": len(rec.evidence)}


def _norm_prompt(d: dict) -> Optional[str]:
    for k in ("prompt", "user", "text"):
        if isinstance(d.get(k), str):
            return re.sub(r"\s+", " ", d[k]).strip().lower()
    return None


def overlap(a: str, b: str, limit: int = 200_000) -> dict:
    """Exact-prompt overlap between two jsonl datasets (normalised whitespace/case).
    The cheap contamination check: eval prompts that also sit in a train set."""
    def prompts(n):
        out = set()
        for i, d in enumerate(rows(n)):
            if i >= limit:
                break
            p = _norm_prompt(d) if isinstance(d, dict) else None
            if p:
                out.add(p)
        return out
    pa, pb = prompts(a), prompts(b)
    both = pa & pb
    return {"a": a, "b": b, "n_a": len(pa), "n_b": len(pb), "shared": len(both),
            "share_of_a": round(len(both) / max(1, len(pa)), 4),
            "share_of_b": round(len(both) / max(1, len(pb)), 4)}


def register_planned(entries: list[dict], overwrite: bool = False) -> list[str]:
    """Catalog sources we intend to use but have not downloaded: license, url, size facts and
    caveats are recorded now, so the decision "can we use this, under what terms" is made once.
    No `out_path`: verify skips them until a download turns the record into a real artifact."""
    written = []
    for e in entries:
        e = dict(e)
        name = e.pop("name")
        if record_path(name).exists() and not overwrite:
            continue
        rec = make_record(name=name, stage="raw", provenance=e.pop("provenance", "external-other"),
                          origin=e.pop("origin", ""), script="", out_path="",
                          status="planned", **e)
        save(rec)
        written.append(name)
    return written
