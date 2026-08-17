"""Canonical CPU/inference-only RWKV-7 loader (BlinkDL ``rwkv`` package).

Extracted 2026-08-17 from ``A0_state_probe/probe.py``, where this same
``load_model`` had grown five independent near-duplicates across
``A0_state_probe/probe.py``, ``rl/train_wordsearch.py``,
``H18_merge/test_arithmetic_merge.py``, ``rosa_probe/run_hybrid.py`` and
``A0.8_refine/generate_cot_corpus.py``. This is the one canonical
implementation; new probe/eval scripts should import from here.

CPU-only, native bf16/fp32, no triton dependency (deliberately avoids the
HuggingFace + ``fla`` path, which requires triton and is CUDA-only in
practice). For the GPU-capable, differentiable, dual-backend loader used
by RL training, see ``experiments/rl/loader.py`` — that module already
imports ``load_model`` from here for its CPU ("blink") backend; it is a
different tool for a different job (training vs. read-only probing), not
a duplicate to be merged into this one.

Sets ``torch.set_grad_enabled(False)`` globally on load — do not use this
loader anywhere gradients are needed (training, backprop-based probes).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def resolve_weight_path(name: str) -> str:
    """Resolve ``name`` to a local .pth path (without the .pth suffix).

    Accepted forms:

    - ``owner/repo:filename.pth`` — HuggingFace repo + specific weight
      file. Downloaded via ``hf_hub_download`` on first use, cached
      under ``~/.cache/huggingface/hub``.
    - ``/absolute/path/to/model.pth`` or ``model`` (an existing file) —
      used directly.

    The rwkv package appends ``.pth`` to the path passed to its
    constructor, so we return the path *without* the trailing ``.pth``.
    """
    name = os.path.expanduser(name)
    if ":" in name and not os.path.isabs(name):
        repo, filename = name.split(":", 1)
        local = hf_hub_download(repo_id=repo, filename=filename)
        # local ends with .pth; strip it because rwkv.RWKV appends '.pth'.
        if local.endswith(".pth"):
            return local[:-4]
        return local

    # Direct path.
    if name.endswith(".pth"):
        cand = name[:-4]
    else:
        cand = name
    if not os.path.exists(cand + ".pth"):
        raise FileNotFoundError(
            f"Model weight file not found: {cand}.pth "
            f"(hint: pass HF repo as 'owner/repo:filename.pth')"
        )
    return cand


class _TokenizerAdapter:
    """Thin adapter around ``rwkv.utils.PIPELINE`` giving an HF-like API.

    Callers do `tokenizer(prompt, return_tensors="pt")`. The rwkv
    PIPELINE exposes ``.encode()``/``.decode()`` on the World vocab. We
    wrap those so calling code doesn't have to know which tokenizer
    flavour is under the hood.
    """

    def __init__(self, pipeline):
        self._p = pipeline

    def __call__(self, prompt: str, return_tensors: Optional[str] = None):
        ids = self._p.encode(prompt)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._p.decode(ids)


def load_model(name: str, device: str = "cpu") -> Tuple[object, _TokenizerAdapter]:
    """Load a RWKV-7 checkpoint via BlinkDL's ``rwkv`` package.

    Args:
        name: One of

            - ``owner/repo:filename.pth`` — e.g.
              ``BlinkDL/rwkv-7-world:RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth``
              or ``BlinkDL/rwkv7-g1:rwkv7-g1h-2.9b-20260710-ctx10240.pth``
            - Local path to a ``.pth`` file.
        device: ``"cpu"`` (default) or ``"cuda"``.

    Returns ``(model, tokenizer)`` where the model is an ``rwkv.RWKV``
    instance ready to run ``forward(idx, state)`` and the tokenizer is
    the World vocab adapter with an HF-shaped call signature.
    """
    # The ``rwkv`` package gates its RWKV-7 code path behind an env flag.
    # Without ``RWKV_V7_ON=1`` ``rwkv.model.RWKV`` binds to the v4/v5/v6
    # legacy class, whose state layout does not match RWKV-7 x070.
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    os.environ.setdefault("RWKV_V7_ON", "1")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    # The rwkv package reads this before importing its model implementation.
    # Make the explicit device choice authoritative, including over a stale
    # value left in the environment by an earlier probe invocation.
    os.environ["RWKV_CUDA_ON"] = "1" if device == "cuda" else "0"

    from rwkv.model import RWKV  # local import — rwkv monkeypatches jit globals
    from rwkv.utils import PIPELINE

    torch.set_grad_enabled(False)
    weight_path = resolve_weight_path(name)

    # bf16 for weights + state, matches the paper's inference precision.
    strategy = f"{device} bf16"
    model = RWKV(model=weight_path, strategy=strategy)

    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    tokenizer = _TokenizerAdapter(pipeline)
    return model, tokenizer
