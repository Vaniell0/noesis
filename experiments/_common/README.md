# experiments/_common — shared probe framework

Started 2026-08-17 after finding five independent copies of the same
RWKV-7 CPU model-loading code scattered across `experiments/*/`, and
after noticing that running N probes on one checkpoint meant loading
that checkpoint N separate times in N separate processes. This package
is the fix, and the convention new probe scripts should follow.

## What's here

- **`model.py`** — the one canonical CPU/inference-only RWKV-7 loader
  (`load_model(path, device) -> (model, tokenizer)`). If you're writing a
  new read-only probe, import from here. If you're writing something
  that needs gradients (training), see `experiments/rl/loader.py`
  instead — different job, not a duplicate.
- **`registry.py`** — `@registry.probe(name, hypothesis=[...],
  description=..., add_args=...)` decorator. Registers a
  `fn(model, tokenizer, args) -> dict` function so it can run as part of
  a shared battery instead of only standalone.
- **`results.py`** — `save_result(path, data, experiment=, hypothesis=,
  summary=)` stamps a `_meta` block into the JSON you write, so
  `experiments/regenerate_results.py` can find it later. `summary` is an
  optional `{metric_label: value_str}` dict that becomes rows in
  `experiments/RESULTS.md`'s auto-generated section — worth setting when
  the result reduces to a few headline numbers.
- **`experiments/run.py`** (repo-relative, not under `_common/`) — the
  shared CLI: `python experiments/run.py --model X --device cpu --tests
  ipc,think_geometry` loads the model once and runs both. `--list` shows
  every registered probe.
- **`experiments/regenerate_results.py`** — rebuilds the auto-generated
  half of `RESULTS.md` from every `_meta`-stamped JSON under
  `experiments/`. Run it after a batch of probes to refresh the index;
  nothing here needs hand-transcription.
- **`layers.py`** (added 2026-08-18) — `default_layers(n_layer)` picks
  probe layers by fractional depth instead of a hardcoded absolute list.
  Every probe that used to hardcode `"0,4,8,16,24,31"` (a 32-layer
  assumption, silently wrong on G1d's 24 layers) now defaults `--layers`
  to `None` and calls this once the real model depth is known. Use it for
  any new probe's layer default instead of hardcoding one.
- **`heartbeat.py`** (added 2026-08-18) — `write_heartbeat(path, progress=(cur,
  total), **fields)` writes a `status.json` + a `status.html` (real
  `<progress>` bar, auto-refreshing) for any long-running job. Point
  `experiments/monitor.py <path>` at the JSON for a plain-console view, or
  serve the directory over HTTP (`python -m http.server`, see
  `experiments/run_model_battery.sh --status-port`) and open the `.html`
  in a browser. Written after SSH/shell sessions proved unreliable for
  checking on a remote job mid-run — a plain file survives that.
- **`test_registry.py`** — regression guard for CLI flag collisions.
  `experiments/run.py` shares one argparse parser across every probe's
  `add_args`, `conflict_handler="resolve"` — two probes registering the
  same flag with *different* defaults silently means one wins with no
  error. Run `python experiments/_common/test_registry.py` after adding a
  probe; it diffs every registered flag's default across probes and
  fails loudly on a real mismatch (same-name-same-meaning shares fine,
  e.g. `--layers`, and isn't flagged).

## Converting an existing standalone probe script

`experiments/A0_state_probe/ipc_analysis.py` is the worked example — read
it if the steps below are unclear.

1. Add the sys.path bootstrap (needed because scripts here are run both
   as bare `python path/to/script.py`, which puts only the script's own
   directory on `sys.path`, *and* as `experiments.X.Y` module imports by
   `experiments/run.py`, which need the repo root importable):

   ```python
   import sys
   from pathlib import Path
   _REPO_ROOT = Path(__file__).resolve().parents[N]  # N = dirs between this file and repo root
   if str(_REPO_ROOT) not in sys.path:
       sys.path.insert(0, str(_REPO_ROOT))

   from experiments._common import registry
   from experiments._common.model import load_model
   ```

2. Split whatever `main()` currently does into three pieces:
   - `_add_<name>_args(ap)` — the probe's *own* CLI flags only. Do not
     add `--model`/`--device`/`--out*` here; the shared runner already
     owns those globally, and standalone `main()` adds them itself.
   - `run(model, tokenizer, args) -> dict`, decorated with
     `@registry.probe("<name>", hypothesis=[...], description="...",
     add_args=_add_<name>_args)` — the actual measurement, assuming the
     model is already loaded. This is what both standalone `main()` and
     the shared runner call.
   - A thin `main()` that keeps the exact old standalone CLI working:
     builds its own parser (`--model`, `--device`, `--out`, plus
     `_add_<name>_args`), loads the model itself, calls `run(...)`, saves
     via `results.save_result`.

3. Add the module's dotted path to `KNOWN_PROBE_MODULES` in
   `experiments/_common/registry.py` so it actually gets imported (and
   therefore registers itself) when the shared runner starts.

4. Test both invocation styles — `python experiments/A0_state_probe/your_probe.py --help`
   and `python experiments/run.py --list` should both work.

## Non-goals

This is deliberately *not* trying to unify every probe's result schema,
argument names, or output layout — only the loading and indexing.
Measurements stay as different as the questions they answer; only "how
do I get a model into memory" and "how does this result become findable
later" are standardized.
