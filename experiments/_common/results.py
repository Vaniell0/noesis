"""Self-registering results index.

Every probe that saves its output through :func:`save_result` stamps a
small ``_meta`` block into the JSON it writes. :func:`regenerate_index`
walks the tree for those stamps and rewrites the auto-generated section
of ``experiments/RESULTS.md`` from them — the index grows as scripts run,
instead of someone remembering to hand-transcribe a new row.

The historical rows already in RESULTS.md (backfilled 2026-08-17 from
HYPOTHESES.md, before this mechanism existed) are hand-written and live
above the ``AUTO-GENERATED`` marker; regeneration only ever rewrites
below that marker.
"""
from __future__ import annotations

import inspect
import json
from datetime import date as _date
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_MD = _REPO_ROOT / "experiments" / "RESULTS.md"
_AUTO_MARKER = "<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python experiments/regenerate_results.py` -->"


def save_result(
    path: Path | str,
    data: Mapping[str, Any],
    *,
    experiment: str,
    hypothesis: Sequence[str] = (),
    status: str = "done",
    summary: Optional[Mapping[str, str]] = None,
    model: Optional[str] = None,
    script: Optional[str] = None,
) -> Path:
    """Write ``data`` to ``path`` as JSON with a ``_meta`` block stamped in.

    Args:
        path: output file. Parent directories are created if missing.
        data: the probe's own result payload (unchanged, written as-is
            alongside ``_meta``).
        experiment: short registry name, e.g. ``"ipc"`` — matches the
            name a probe was registered under in ``registry.py`` where
            applicable, but this function has no hard dependency on the
            registry (works for one-off scripts too).
        hypothesis: HYPOTHESES.md IDs this result speaks to, e.g. ``["H8"]``.
        status: ``"done"``, ``"partial"``, ``"failed"`` — free text, kept
            short. Does not gate anything; purely descriptive.
        summary: optional ``{metric_label: value_str}`` pairs — becomes
            one row each in the auto-generated RESULTS.md table. Omit for
            results too complex to summarise in one line; the table will
            then just link the file. If omitted, falls back to
            ``data["_summary"]`` if the probe put one there itself (the
            probe's own `run()` function is usually in a better position
            to say "these 2 numbers matter most" than a generic caller
            is) — either way `_summary` is stripped from the stored
            payload, it's metadata, not part of the result.
        model: optional convenience copy of which checkpoint this ran on,
            shown in the table. If omitted, tries ``data["model"]``.
        script: path to the code that produced this result, so RESULTS.md
            can link result *and* method side by side — that pairing is
            the point, a number without its generating code is much
            harder to sanity-check. If omitted, best-effort auto-detected
            from the immediate caller's file — but the shared runner
            (`experiments/run.py`) calls this on every probe's behalf, so
            in that path the caller would resolve to `run.py` itself, not
            the probe; pass `script` explicitly whenever the immediate
            caller isn't the actual measurement code.

    Returns the path written to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if script is None:
        caller = inspect.stack()[1]
        script = caller.filename
    try:
        script_rel = str(Path(script).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        script_rel = str(script)

    if summary is None:
        summary = data.get("_summary")

    meta = {
        "experiment": experiment,
        "hypothesis": list(hypothesis),
        "date": _date.today().isoformat(),
        "status": status,
        "model": model if model is not None else data.get("model"),
        "script": script_rel,
    }
    if summary:
        meta["summary"] = dict(summary)

    payload = {"_meta": meta, **{k: v for k, v in data.items() if k not in ("_meta", "_summary")}}
    out_path.write_text(json.dumps(payload, indent=2))

    # Human-readable companion, next to the JSON — auto-rendered from the
    # same `_meta` every time, so it can't drift the way a hand-written
    # report.md next to raw results used to (see the many empty/stale
    # `report.md` stubs found across experiments/premise_validator/ etc.
    # during the 2026-08-17 audit — this replaces that pattern).
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_result_markdown(meta, out_path.name))

    return out_path


def _render_result_markdown(meta: Mapping[str, Any], json_filename: str) -> str:
    lines = [f"# {meta['experiment']}", ""]
    lines.append(f"- **Hypothesis:** {', '.join(meta.get('hypothesis') or []) or '—'}")
    lines.append(f"- **Date:** {meta.get('date', '—')}")
    lines.append(f"- **Status:** {meta.get('status', '—')}")
    if meta.get("model"):
        lines.append(f"- **Model:** `{meta['model']}`")
    if meta.get("script"):
        lines.append(f"- **Code:** [`{meta['script']}`](/{meta['script']})")
    lines.append("")
    summary = meta.get("summary")
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in summary.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    lines.append(f"Full data: `{json_filename}` (same directory).")
    lines.append("")
    lines.append(
        "*Auto-generated by `experiments._common.results.save_result` from the "
        "run's `_meta` block — do not hand-edit, it is overwritten on the next run "
        "of this probe. See `experiments/RESULTS.md` for the cross-experiment index.*"
    )
    return "\n".join(lines) + "\n"


def write_probe_info(spec, out_dir: Optional[Path | str] = None) -> Path:
    """Render a registered probe's own info page from its registry metadata.

    One file per *probe* (not per run) — "what does this measurement do
    and which hypothesis does it serve", derived entirely from the
    `@registry.probe(...)` decorator arguments and the function's
    docstring, which are the single source of truth in the code. Written
    next to the probe's module file by default, so it sits at the level
    a reader would look for it (one directory up from a specific run's
    JSON/md, at the probe/script itself).
    """
    module_file = Path(inspect.getfile(spec.fn)).resolve()
    target_dir = Path(out_dir) if out_dir else module_file.parent
    out_path = target_dir / f"{spec.name}.info.md"

    doc = inspect.getdoc(spec.fn) or ""
    lines = [f"# {spec.name}", ""]
    lines.append(f"- **Hypothesis:** {', '.join(spec.hypothesis) or '—'}")
    lines.append(f"- **Description:** {spec.description or '—'}")
    lines.append(f"- **Registered in:** `{module_file.relative_to(_REPO_ROOT)}`")
    lines.append("")
    if doc:
        lines.append("## Docstring")
        lines.append("")
        lines.append(doc)
        lines.append("")
    lines.append(
        f"Run standalone: `python {module_file.relative_to(_REPO_ROOT)} --help`. "
        f"Run as part of a battery: `python experiments/run.py --tests {spec.name} --model ... --device ...`."
    )
    lines.append("")
    lines.append(
        "*Auto-generated by `experiments._common.results.write_probe_info` from the "
        "registry entry — do not hand-edit; regenerate via "
        "`python experiments/regenerate_results.py`.*"
    )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _iter_meta_stamped(root: Path) -> list[tuple[Path, dict]]:
    found = []
    for p in root.rglob("*.json"):
        if "/.venv/" in str(p) or "/node_modules/" in str(p):
            continue
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("_meta"), dict):
            found.append((p, obj["_meta"]))
    return found


def regenerate_index(
    root: Path | str = None,
    out_path: Path | str = None,
) -> int:
    """Rebuild the auto-generated section of RESULTS.md from `_meta` stamps.

    Returns the number of rows written. Idempotent — re-running with no
    new stamped results reproduces the same file.
    """
    root = Path(root) if root else (_REPO_ROOT / "experiments")
    out_path = Path(out_path) if out_path else _RESULTS_MD

    rows = []
    for path, meta in sorted(_iter_meta_stamped(root)):
        try:
            rel = path.relative_to(_REPO_ROOT)
        except ValueError:
            # `root` wasn't under the repo (e.g. a tempdir in tests, or any
            # future caller pointing this at an external directory) — same
            # fallback as save_result's script_rel, found via test_core.py.
            rel = path
        h = ", ".join(meta.get("hypothesis") or []) or "—"
        model = meta.get("model") or "—"
        date = meta.get("date") or "—"
        status = meta.get("status") or "—"
        script = meta.get("script")
        code_cell = f"`{script}`" if script else "—"
        # `status != "done"` gets its own column instead of being silent
        # once a summary exists — added 2026-08-18 after noticing
        # externally-reported results (status="external-reported", e.g.
        # fleeb83's emailed numbers) render identically to a real local
        # run the moment they have a `summary`, which is exactly the case
        # that matters (a bare row with no summary already shows
        # `status=X` as its "value" — only the summary branch was silent).
        status_flag = status if status not in ("done", "—") else ""
        summary = meta.get("summary") or {}
        if summary:
            for metric, value in summary.items():
                rows.append((h, model, date, status_flag, metric, value, f"`{rel}`", code_cell))
        else:
            rows.append((h, model, date, status_flag, "—", f"status={status}", f"`{rel}`", code_cell))

    table_lines = [
        "| H | Model | Date | Status | Metric | Value | Result | Code |",
        "|---|-------|------|--------|--------|-------|--------|------|",
    ]
    for h, model, date, status_flag, metric, value, rel, code_cell in rows:
        table_lines.append(f"| {h} | {model} | {date} | {status_flag} | {metric} | {value} | {rel} | {code_cell} |")
    if not rows:
        table_lines.append("| — | — | — | — | — | *(no self-registered results yet)* | — | — |")

    text = out_path.read_text() if out_path.exists() else "# Results index\n"
    if _AUTO_MARKER in text:
        head = text.split(_AUTO_MARKER)[0].rstrip() + "\n\n"
    else:
        head = text.rstrip() + "\n\n"

    out_path.write_text(head + _AUTO_MARKER + "\n\n" + "\n".join(table_lines) + "\n")
    return len(rows)
