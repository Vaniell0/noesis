"""training/build_corpus.py — declarative recipe -> combined training corpus.

Operator entrypoint for `training/_common/{registry,provenance,corpus}.py`.
Takes a YAML recipe (an anchor source + fractional sources, each either a
pre-tokenized `.pt` path or a registry stage name to generate one) and
resolves every source, generating whatever's missing (jsonl via a
registered `normalize` stage, `.pt` via a registered `tokenize` stage),
then combines by fraction and saves with full provenance. Replaces "run
normalize_X.py, then tokenize_Y.py, then combine_stepN_corpus.py by hand
in the right order with the right flags" with one recipe file.

Only `training.scripts.normalize_hh_rlhf` is migrated onto the registry
as of this writing (see `training/_common/registry.py`'s KNOWN_MODULES) —
no `tokenize` stage is registered yet. A recipe source that needs
generation from a stage not yet migrated fails loudly (KeyError naming
what *is* registered) rather than silently shelling out to an
unregistered script — the registry is the single source of truth for
what this tool can generate, same discipline as the experiments side.

Recipe schema (YAML):

    name: step10
    out: training/tokenised/step10_combined_train.pt
    seed: 42
    sources:
      # first entry is the anchor — its own fraction is nominal (all of
      # itself is used); every other source is sampled to
      # anchor_tokens * (its fraction / anchor fraction) tokens.
      - name: rfc
        fraction: 0.30
        pt: training/tokenised/step9_rfc_train.pt
        provenance: generated
        origin: "RFC QA (restructure_rfc.py)"
      - name: hhrlhf
        fraction: 0.15
        pt: training/tokenised/hh_rlhf_train.pt   # used directly if it exists
        # ...else generated on demand:
        jsonl: training/corpus_open/hh_rlhf.jsonl
        normalize_stage: hh_rlhf
        normalize_args: {max_items: 30000}
        tokenize_stage: plain_cot                 # not migrated yet -> KeyError until it is
        provenance: external-hf
        origin: Anthropic/hh-rlhf

Usage:
    training/.venv/bin/python training/build_corpus.py \\
        --recipe training/config/recipes/step10.yaml
    training/.venv/bin/python training/build_corpus.py \\
        --recipe training/config/recipes/step10.yaml --dry-run
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from training._common import corpus, registry  # noqa: E402


def _resolve_jsonl(source: dict, seed: int) -> Path:
    jsonl = source.get("jsonl")
    if jsonl and Path(jsonl).exists():
        return Path(jsonl)

    stage_name = source.get("normalize_stage")
    if not stage_name:
        raise FileNotFoundError(
            f"source {source['name']!r}: no jsonl at {jsonl!r} and no "
            f"normalize_stage given to generate one"
        )
    spec = registry.get(stage_name)
    if spec.kind != "normalize":
        raise ValueError(f"stage {stage_name!r} is kind={spec.kind!r}, expected 'normalize'")

    out_path = jsonl or spec.out_default
    if out_path is None:
        raise ValueError(
            f"source {source['name']!r}: stage {stage_name!r} has no out_default "
            f"and recipe gives no jsonl path"
        )
    print(f"[build_corpus] {source['name']}: generating jsonl via {stage_name!r} -> {out_path}")
    ns_kwargs = dict(source.get("normalize_args", {}))
    ns_kwargs.setdefault("seed", seed)
    ns = argparse.Namespace(out=out_path, **ns_kwargs)
    result = spec.fn(ns)
    return Path(result["out_path"])


def _resolve_pt(source: dict, seed: int) -> Path:
    pt = source.get("pt")
    if pt and Path(pt).exists():
        return Path(pt)

    stage_name = source.get("tokenize_stage")
    if not stage_name:
        raise FileNotFoundError(
            f"source {source['name']!r}: no pt file at {pt!r} and no "
            f"tokenize_stage given to generate one"
        )
    spec = registry.get(stage_name)  # raises KeyError naming what IS registered, if not migrated
    if spec.kind != "tokenize":
        raise ValueError(f"stage {stage_name!r} is kind={spec.kind!r}, expected 'tokenize'")

    jsonl_path = _resolve_jsonl(source, seed)
    out_path = pt or spec.out_default
    print(f"[build_corpus] {source['name']}: tokenizing via {stage_name!r} -> {out_path}")
    ns_kwargs = dict(source.get("tokenize_args", {}))
    ns = argparse.Namespace(input=str(jsonl_path), out=out_path, **ns_kwargs)
    result = spec.fn(ns)
    return Path(result.get("out_train") or result["out_path"])


def build(recipe: dict, dry_run: bool = False) -> Path | None:
    registry.import_known_modules()
    seed = recipe.get("seed", 42)
    rng = random.Random(seed)

    sources_cfg = recipe["sources"]
    if not sources_cfg:
        raise ValueError("recipe has no sources")

    specs = []
    for s in sources_cfg:
        pt_path = _resolve_pt(s, seed)
        specs.append(corpus.SourceSpec(
            name=s["name"], path=pt_path, fraction=s["fraction"],
            provenance=s.get("provenance", "generated"), origin=s.get("origin", ""),
        ))

    combined, per_source_meta = corpus.combine_by_fraction(specs, rng)
    rng.shuffle(combined)

    total_tok = sum(m["n_tokens"] for m in per_source_meta.values())
    print(f"[build_corpus] {recipe.get('name', '(unnamed)')}: "
          f"{len(combined)} rollouts, {total_tok} tokens")
    for name, meta in per_source_meta.items():
        pct = 100 * meta["n_tokens"] / max(total_tok, 1)
        print(f"[build_corpus]   {name}: {meta['n_rollouts']} rollouts, {meta['n_tokens']} tok "
              f"({pct:.1f}%, target {meta['fraction_target']*100:.1f}%)")

    if dry_run:
        print("[build_corpus] --dry-run: not writing output")
        return None

    out_path = corpus.save_combined_corpus(
        combined, recipe["out"], per_source_meta, script=str(Path(__file__)),
    )
    print(f"[build_corpus] -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, help="Path to a recipe YAML file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve sources and print the token budget without writing output")
    args = ap.parse_args()

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    build(recipe, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
