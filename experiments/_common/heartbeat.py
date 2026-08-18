"""Dumb, dependency-free status heartbeat for long-running jobs.

Motivation (2026-08-18): SSH/bash sessions used to check on a remote job
have proven unstable (a live example: the classifier backing this
session's Bash tool timed out mid-task tonight). A running job writing
its own status to a plain JSON file — readable directly, or over a
mounted disk, or by any trivial script, independent of any particular
tool's session state — is a more robust way to check "is it still going,
how far along" than polling through a shell.

Usage, inside a training/probe loop:

    from experiments._common.heartbeat import write_heartbeat
    write_heartbeat(out_dir / "status.json", step=step, total=args.steps,
                     message=f"loss={loss:.4f}")

Deliberately not a class, not a background thread, not a server — call it
inline wherever you already print progress. One JSON file, overwritten
each call, nothing to keep running.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/Infinity with None.

    `json.dumps` emits bare `NaN`/`Infinity` tokens by default (a Python
    extension, not valid JSON) — harmless for another Python reader, but
    breaks the actual point of this file: being viewable by a plain
    browser/curl/jq, none of which accept those tokens. `--no-update`
    training runs hit this on every call (loss stays NaN by design).
    """
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _render_html(payload: dict) -> str:
    """A glanceable status page — an actual progress bar, not a JSON dump.

    Added 2026-08-18: opening status.json directly in a browser just shows
    raw text; the point of serving it over HTTP was to check progress at
    a glance without going through a shell. `<progress>` is a native HTML
    element — no CSS/JS framework needed for "dumb code enough."
    """
    progress = payload.get("progress")
    if progress and len(progress) == 2 and progress[1]:
        cur, total = progress
        pct = 100.0 * cur / total
        bar = f'<progress value="{cur}" max="{total}" style="width:100%;height:2em"></progress>' \
              f'<p>{cur} / {total} ({pct:.1f}%)</p>'
    else:
        bar = "<p>(no progress total reported)</p>"

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in payload.items() if k not in ("ts", "ts_human", "progress")
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>noesis status</title><meta http-equiv="refresh" content="10">
<style>body{{font-family:monospace;max-width:640px;margin:2em auto}}
table{{width:100%;border-collapse:collapse}}td{{padding:.3em;border-bottom:1px solid #ccc}}
td:first-child{{font-weight:bold;width:12em}}</style></head>
<body><h2>noesis — {payload.get('ts_human', '?')}</h2>{bar}
<table>{rows}</table></body></html>"""


def write_heartbeat(path: Path | str, *, progress: tuple[int, int] | None = None,
                     **fields: Any) -> None:
    """Overwrite `path` (and a same-named `.html` companion) with status.

    `progress=(current, total)` renders an actual `<progress>` bar in the
    HTML companion — pass it explicitly rather than expecting this
    function to guess which of the caller's `**fields` means "how far
    along" (different callers use different names: step/total_planned,
    probe_index/probe_total, ...).
    """
    now = time.time()
    payload = {
        "ts": now,
        "ts_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "progress": list(progress) if progress else None,
        **fields,
    }
    payload = _json_safe(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename — avoids a reader ever seeing a
    # half-written file if it polls at the wrong instant.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)

    html_path = p.with_suffix(".html")
    html_tmp = html_path.with_suffix(".html.tmp")
    html_tmp.write_text(_render_html(payload))
    html_tmp.replace(html_path)
