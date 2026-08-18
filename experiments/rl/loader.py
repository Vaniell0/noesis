"""Shared RWKV-7 loader for the RL stack (dual-mode).

Two backends, one interface:

- **peft** (GPU, differentiable): RWKV-PEFT `RWKV7` (nn.Module, pure
  PyTorch + rwkvfla kernels). Requires CUDA + rwkvfla. Used for training
  and any WKV-loop that needs backprop.
- **blink** (CPU, inference-only): BlinkDL `rwkv` package via
  `probe.load_model`. Weights in a `model.z` dict; JIT-compiled forward.
  Used for CPU smoke tests and inference-only probes.

The unified interface is `forward_stateful(input_ids, state) → (logits,
state)`, called the same way in both backends. WKV state extraction is
adapted per backend so downstream code (WKV loop, reward computation)
stays backend-agnostic.

Dtype parity — the reason this loader exists — is only meaningful when
rollout and training share weights (GPU + peft backend). On CPU (blink)
we're only validating the loop mechanics and API, not doing RL updates.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_NOESIS_ROOT = Path(__file__).parents[2]
_PEFT_ROOT = _NOESIS_ROOT / "training/rwkv-peft"


# --------------------------------------------------------------------- #
# Env priming
# --------------------------------------------------------------------- #

def _prime_env() -> None:
    """Env vars read at import time by RWKV-PEFT modules."""
    # rwkvfla decorates some ops (e.g. rwkv7/chunk.py::cal_log_w) with
    # @torch.compile(fullgraph=True). Turing (sm_75, e.g. T4) has no native
    # bf16 compile support, so fullgraph compilation of those ops hard-fails
    # ("BF16 is not supported") even though the op itself (here, -exp(w)) is
    # trivial and gains nothing from compilation on this GPU. fullgraph=True
    # raises a hard RuntimeError on failure by design — suppress_errors only
    # catches ordinary graph breaks, not this — so dynamo needs to be fully
    # disabled (torch.compile becomes a plain passthrough) rather than just
    # told to tolerate errors.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch._dynamo
    torch._dynamo.config.disable = True

    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    # forward_stateful/forward_stateful_embeds always call model.forward_infctx —
    # every downstream module (block/att/ffn/rwkvop) branches off this exact env
    # var to reach the infctx codepath; "" silently fell through to forward_normal
    # (wrong arg count → crash) instead of forward_infctx.
    os.environ.setdefault("RWKV_TRAIN_TYPE", "infctx")
    os.environ.setdefault("RWKV_JIT_ON", "0")
    os.environ.setdefault("FUSED_KERNEL", "0")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
    os.environ.setdefault("WKV", "fla")


def wrap_rwkv7_excluding_head(model, **wrap_kwargs):
    """wrap_rwkv7() from FORGE, then revert `head` back to plain nn.Linear.

    FORGE's fused-into-backward update assumes one forward call per training
    step per layer — the optimizer step for a FusedLinear's weight happens
    as a side effect of ITS OWN local backward, the first (and only) time
    autograd expects to touch that weight. Our WKV-loop calls the model
    (and therefore `head`) many times per training step — once per
    generated token, across every rollout in the batch — all accumulated
    into one shared loss, backed by one `loss.backward()` call. Each of
    those `head` invocations independently triggers another premature
    in-place weight update inside that single backward pass, so by the
    time autograd reaches an EARLIER invocation's saved-for-backward
    weight, it has already been mutated by a LATER one:
    `RuntimeError: one of the variables needed for gradient computation
    has been modified by an inplace operation ... version 290; expected
    version 1` (found running --forge on real GPU for the first time,
    2026-08-18). `att`/`ffn` sublayers don't have this problem — each is
    called at most once per token per layer per step, same as the
    single-forward-per-step case FORGE is designed for; only `head` (and
    only because of the WKV-loop's per-token decode structure) is reused
    within one backward. wrap_rwkv7() itself takes no exclude-list
    (asked FORGE upstream isn't an option we control), so this reverts
    just that one layer after the fact and rebuilds the manager so
    `get_non_fused_params()` correctly picks up head's weight/bias for
    the regular optimizer instead.
    """
    from fused_grad_optimizer.model_wrappers import wrap_rwkv7
    from fused_grad_optimizer.module import FusedLinear, FusedOptimizerManager
    import torch.nn as nn

    model, _ = wrap_rwkv7(model, **wrap_kwargs)

    head = model.head
    if isinstance(head, FusedLinear):
        plain = nn.Linear(head.in_features, head.out_features,
                           bias=head.bias is not None)
        plain.weight.data = head.weight.data
        if head.bias is not None:
            plain.bias.data = head.bias.data
        plain = plain.to(device=head.weight.device, dtype=head.weight.dtype)
        model.head = plain

    manager = FusedOptimizerManager(model)
    return model, manager


class Int8AdamW:
    """AdamW with FORGE's int8-quantized moment state, applied to gradients
    from ordinary (non-fused) `backward()` — not FORGE's fused-into-backward
    path.

    This is the actual fix for `--forge` (wrap_rwkv7_excluding_head and the
    per-rollout-backward rewrite in train_wkv_loop.py get partway there but
    don't resolve it — see that docstring). FORGE's fused-into-backward
    mechanism assumes each layer is touched by backward() at most once per
    step; BPTT through the WKV-loop's recurrent state touches every
    FusedLinear layer once per timestep, which the fused path can't
    tolerate no matter how the outer batch loop is structured. This class
    sidesteps that entirely: don't fuse anything into backward, just run
    ordinary autograd (so BPTT gradients are exactly as correct as the
    non-FORGE path), then apply FORGE's standalone
    `optimizer_only_adamw_int8state()` kernel to the resulting `.grad`
    tensors. Keeps the larger of FORGE's two memory wins (int8 optimizer
    state: ~4x smaller than fp32 AdamW moments) without touching backward()
    or truncating gradient flow into the M-loop.

    Only 2D parameters whose last dim is divisible by `qblock` are
    int8-optimized (the kernel's own constraint — matches FusedLinear
    weights: att/ffn projections). Everything else (biases, LayerNorm,
    embeddings, time_decay/time_first) needs a regular optimizer — see
    `other_params`.
    """

    def __init__(self, params, lr: float = 1e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01, qblock: int = 64):
        from fused_grad_optimizer.state import FusedOptimizerState, OptimizerConfig

        params = list(params)
        self.fused_params = [p for p in params
                             if p.dim() == 2 and p.shape[1] % qblock == 0]
        fused_ids = {id(p) for p in self.fused_params}
        self.other_params = [p for p in params if id(p) not in fused_ids]

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0
        self._states = {
            id(p): FusedOptimizerState(p, optimizer_type="adamw",
                                       state_mode="int8", qblock_size=qblock)
            for p in self.fused_params
        }
        self._OptimizerConfig = OptimizerConfig

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.fused_params:
            p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)

    def step(self) -> None:
        from fused_grad_optimizer.autograd import _apply_precomputed
        self._step += 1
        config = self._OptimizerConfig(
            optimizer_type="adamw", lr=self.lr, beta1=self.beta1,
            beta2=self.beta2, eps=self.eps, weight_decay=self.weight_decay,
        )
        for p in self.fused_params:
            if p.grad is None:
                continue
            _apply_precomputed(p.grad.float(), p.data, self._states[id(p)], config)


def _stub_deepspeed_if_missing() -> None:
    """rwkvt/rwkv7/model.py does `import deepspeed` at module scope.

    Only used inside forward when grad_cp==1. With grad_cp=0 (our default)
    runtime never touches deepspeed.
    """
    try:
        import deepspeed  # noqa: F401
        return
    except ImportError:
        pass
    import types
    stub = types.ModuleType("deepspeed")
    ckpt = types.ModuleType("deepspeed.checkpointing")
    def _no_ckpt(*args, **kwargs):
        raise RuntimeError(
            "deepspeed.checkpointing.checkpoint called but deepspeed is not "
            "installed. Set args.grad_cp=0 (default in loader)."
        )
    ckpt.checkpoint = _no_ckpt
    stub.checkpointing = ckpt
    sys.modules["deepspeed"] = stub
    sys.modules["deepspeed.checkpointing"] = ckpt


# --------------------------------------------------------------------- #
# Unified LoadedModel
# --------------------------------------------------------------------- #

@dataclass
class LoadedModel:
    """Backend-agnostic model handle.

    Attributes:
        backend: "peft" or "blink"
        model: underlying model object (RWKV7 nn.Module or rwkv.RWKV)
        tokenizer: has .encode(str) → list[int] and .decode(list[int]) → str
        n_layer, n_embd, n_head, head_size: architectural constants
        device, dtype: physical placement
        vocab_size
    """
    backend: str
    model: object
    tokenizer: object
    n_layer: int
    n_embd: int
    n_head: int
    head_size: int
    vocab_size: int
    device: str
    dtype: torch.dtype
    # backend-specific internals — not read by callers directly
    _peft_bsl: Optional[type] = None
    _emb_weight: Optional[torch.Tensor] = None

    # ---- unified interface -------------------------------------------

    def new_state(self, batch: int = 1):
        """Return an initial (zero) state suitable for forward_stateful."""
        if self.backend == "peft":
            bsl = self._peft_bsl.create(
                N=self.n_layer, B=batch, C=self.n_embd, H=self.n_head,
                device=self.device, dtype=self.dtype,
            )
            return _PeftState(bsl.shift_states, bsl.wkv_states)
        # blink: state is a flat Python list [3*n_layer]; None means "fresh"
        return _BlinkState(state=None)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,       # [B, T]  (peft) or [T]  (blink)
        state,                          # from new_state() or previous call
    ) -> Tuple[torch.Tensor, object]:
        """Run one forward pass, return (logits, new_state).

        peft: input_ids [B, T] → logits [B, T, V]
        blink: input_ids [T] → logits [V] (last token only, native)
        Callers using WKV loop should use the peft backend for gradient flow.
        """
        if self.backend == "peft":
            logits, shift, wkv = self.model.forward_infctx(
                input_ids, state.shift, state.wkv,
            )
            return logits, _PeftState(shift, wkv)
        # blink path
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.tolist()
            if ids and isinstance(ids[0], list):
                if len(ids) != 1:
                    raise ValueError("blink backend supports batch=1 only")
                ids = ids[0]
        else:
            ids = list(input_ids)
        logits, new_state = self.model.forward(ids, state.state)
        return logits, _BlinkState(state=new_state)

    def forward_stateful_embeds(
        self,
        inputs_embeds: torch.Tensor,   # [B, T, D] — bypasses emb layer
        state,
    ) -> Tuple[torch.Tensor, object]:
        """Continuous forward: input is pre-computed embeddings, not token ids.

        Used by the WKV loop for expected-embedding / Coconut-style
        feeds where the "next token" is a distribution, not a sample.
        Fully differentiable through the loop — needed for backprop.

        Only implemented for peft backend. blink (BlinkDL rwkv package)
        exposes forward as a JIT-compiled function over token ids only;
        bypassing its embedding lookup would require reimplementing the
        forward from scratch. Continuous loop → peft backend → GPU.
        """
        if self.backend != "peft":
            raise NotImplementedError(
                "forward_stateful_embeds requires peft backend "
                "(differentiable, GPU). CPU/blink is inference-only."
            )
        logits, shift, wkv = _peft_forward_embeds(
            self.model, self._peft_bsl, inputs_embeds,
            state.shift, state.wkv,
        )
        return logits, _PeftState(shift, wkv)

    def wkv_stack(self, state) -> torch.Tensor:
        """Extract WKV state as tensor [n_layer, ...] for CLIPO/probe use.

        peft: [n_layer, B, n_head, head_size, head_size]
        blink: [n_layer, n_head, head_size, head_size] (batch=1 collapsed)
        """
        if self.backend == "peft":
            return state.wkv.detach()
        raw = state.state
        if raw is None:
            return torch.empty(0)
        layers = []
        for i in range(self.n_layer):
            layers.append(raw[3 * i + 1].detach().to(torch.float32).cpu())
        return torch.stack(layers, dim=0)

    @property
    def embedding_weight(self) -> torch.Tensor:
        """emb.weight tensor for expected-embedding computation in WKV loop."""
        if self._emb_weight is not None:
            return self._emb_weight
        if self.backend == "peft":
            w = self.model.emb.weight
        else:
            w = self.model.z["emb.weight"]
        self._emb_weight = w
        return w


@dataclass
class _PeftState:
    shift: torch.Tensor
    wkv: torch.Tensor


@dataclass
class _BlinkState:
    state: Optional[list]


# --------------------------------------------------------------------- #
# PEFT loader (GPU, differentiable)
# --------------------------------------------------------------------- #

def _load_peft(model_path: str, device: str, dtype: torch.dtype,
               ctx_len: int, grad_cp: int) -> LoadedModel:
    if str(_PEFT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PEFT_ROOT))
    _prime_env()
    _stub_deepspeed_if_missing()
    from rwkvt.args_type import TrainingArgs
    from rwkvt.rwkv7.model import RWKV7
    from rwkvt.infctx_module import BlockStateList

    weight_path = os.path.expanduser(model_path)
    state_dict = torch.load(weight_path, map_location="cpu",
                            weights_only=True, mmap=True)
    n_layer, n_embd, vocab_size = _infer_dims(state_dict)

    args = TrainingArgs()
    args.load_model = weight_path
    args.n_layer = n_layer
    args.n_embd = n_embd
    args.vocab_size = vocab_size
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.dim_att = n_embd
    args.dim_ffn = int(((n_embd * 3.5) // 32) * 32)
    args.ctx_len = ctx_len
    args.chunk_ctx = ctx_len
    args.grad_cp = grad_cp
    args.peft = "none"
    args.train_type = "none"
    args.rwkv_version = "x070"
    args.my_testing = "x070"  # read but unused by RWKV_Tmix_x070.__init__ — TrainingArgs
                               # doesn't declare the field at all (train.py's argparse
                               # namespace has it via a CLI default, this dataclass never did)
    args.dropout = 0.0

    model = RWKV7(args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if unexpected:
        print(f"[loader/peft] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[loader/peft] missing keys (zero-init): {len(missing)}")

    model = model.to(dtype=dtype, device=device)

    n_head = n_embd // args.head_size_a
    tokenizer = _load_world_tokenizer()

    return LoadedModel(
        backend="peft",
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        head_size=args.head_size_a,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
        _peft_bsl=BlockStateList,
    )


def _peft_forward_embeds(model, BlockStateList, inputs_embeds: torch.Tensor,
                         shift_states: torch.Tensor, wkv_states: torch.Tensor):
    """Mirror of RWKV7.forward_infctx but skips the emb() lookup.

    See rwkvt/rwkv7/model.py:79-112 for the reference implementation.
    We reproduce the block loop + final ln_out/head over `inputs_embeds`
    directly, so the caller can feed pseudo-embeddings from the WKV loop.
    """
    args = model.args
    B, T, D = inputs_embeds.shape
    assert D == args.n_embd, f"embed dim {D} != n_embd {args.n_embd}"
    assert T <= args.chunk_ctx, "T exceeds chunk_ctx"
    H = args.dim_att // args.head_size_a

    x = inputs_embeds
    new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                      x.device, x.dtype)
    v_first = torch.empty_like(x)

    for i, (block, block_state) in enumerate(
        zip(model.blocks, BlockStateList(shift_states, wkv_states))
    ):
        x, v_first, new_block_state = block(
            x, v_first, block_state, attention_mask=None,
        )
        new_states[i] = new_block_state

    x = model.ln_out(x)
    x = model.head(x)
    return x, new_states.shift_states, new_states.wkv_states


def _infer_dims(state_dict: dict) -> Tuple[int, int, int]:
    vocab_size, n_embd = state_dict["emb.weight"].shape
    n_layer = 0
    while f"blocks.{n_layer}.att.receptance.weight" in state_dict:
        n_layer += 1
    if n_layer == 0:
        raise ValueError("state_dict has no RWKV-7 blocks (blocks.N.att.receptance)")
    return n_layer, int(n_embd), int(vocab_size)


# --------------------------------------------------------------------- #
# BlinkDL loader (CPU, inference-only)
# --------------------------------------------------------------------- #

def _load_blink(model_path: str, device: str) -> LoadedModel:
    # `experiments.rl.loader` is only importable when the repo root is
    # already on sys.path (this module itself is reached via
    # `experiments.rl.X` absolute imports elsewhere), so the canonical
    # loader is reachable the same way — no sys.path surgery needed here.
    from experiments._common.model import load_model as blink_load
    model, probe_tok = blink_load(model_path, device=device)
    # probe_tok exposes __call__ + decode but no .encode; wrap it.
    class _EncAdapter:
        def __init__(self, inner): self._inner = inner
        def encode(self, text): return self._inner(text)["input_ids"]
        def decode(self, ids): return self._inner.decode(ids)
    tok = _EncAdapter(probe_tok)

    # infer dims from loaded weights (model.z is a dict of tensors)
    emb = model.z["emb.weight"]
    vocab_size, n_embd = int(emb.shape[0]), int(emb.shape[1])
    n_layer = 0
    while f"blocks.{n_layer}.att.receptance.weight" in model.z:
        n_layer += 1
    # infer head_size from a WKV-related weight
    head_size = 64  # RWKV-7 default; matches G1i/G1h/G1d checkpoints
    n_head = n_embd // head_size

    return LoadedModel(
        backend="blink",
        model=model,
        tokenizer=tok,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        head_size=head_size,
        vocab_size=vocab_size,
        device=device,
        dtype=torch.bfloat16,  # blink default
    )


# --------------------------------------------------------------------- #
# Tokenizer (World vocab, shared)
# --------------------------------------------------------------------- #

class _TokenizerAdapter:
    def __init__(self, pipeline):
        self._p = pipeline

    def encode(self, text: str) -> List[int]:
        return self._p.encode(text)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._p.decode(ids)


def _load_world_tokenizer() -> _TokenizerAdapter:
    from rwkv.utils import PIPELINE
    class _Stub:
        args = type("A", (), {"vocab_size": 65536})()
    return _TokenizerAdapter(PIPELINE(_Stub(), "rwkv_vocab_v20230424"))


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #

def load_rwkv7(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    ctx_len: int = 16384,
    grad_cp: int = 0,
    backend: Optional[str] = None,
) -> LoadedModel:
    """Build a LoadedModel from a .pth checkpoint.

    Args:
        model_path: local path (or HF ref for blink backend, see probe.py)
        device: "cuda" or "cpu"
        dtype: torch.bfloat16 (recommended) or torch.float32
        ctx_len: context length (peft only, ignored by blink)
        grad_cp: gradient checkpointing (peft only). Keep 0 for CPU.
        backend: "peft" | "blink" | None. None = auto: peft on cuda, blink on cpu.
    """
    if backend is None:
        backend = "peft" if device == "cuda" else "blink"
    if backend == "peft":
        return _load_peft(model_path, device, dtype, ctx_len, grad_cp)
    if backend == "blink":
        return _load_blink(model_path, device)
    raise ValueError(f"backend must be 'peft' | 'blink' | None, got {backend!r}")


# --------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------- #

def _smoke(model_path: str, device: str = "cpu",
           backend: Optional[str] = None) -> None:
    """Validate loader + stateful forward end-to-end.

    On CPU (blink backend), verifies WKV state extraction and stateful
    stepping — the mechanics needed by the WKV loop. Does not exercise
    backprop (blink is inference-only).
    """
    loaded = load_rwkv7(model_path, device=device, backend=backend)
    print(f"[smoke] backend={loaded.backend} n_layer={loaded.n_layer} "
          f"n_embd={loaded.n_embd} n_head={loaded.n_head} "
          f"vocab={loaded.vocab_size}")

    ids = loaded.tokenizer.encode("Hello, world.")
    print(f"[smoke] encoded '{ids[:8]}...' len={len(ids)}")

    # Initial state, prefill
    state = loaded.new_state(batch=1)
    if loaded.backend == "peft":
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        input_ids = ids
    logits, state = loaded.forward_stateful(input_ids, state)
    print(f"[smoke] prefill: logits.shape={_shape(logits)} "
          f"dtype={_dtype(logits)}")

    # Single stateful step
    last = _last_logits(logits)
    next_id = int(last.argmax())
    if loaded.backend == "peft":
        step_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
    else:
        step_ids = [next_id]
    logits2, state = loaded.forward_stateful(step_ids, state)
    print(f"[smoke] step ok: next_logits.shape={_shape(logits2)}")

    # WKV extraction
    wkv = loaded.wkv_stack(state)
    print(f"[smoke] wkv_stack.shape={tuple(wkv.shape)}")

    # Embedding weight (needed for expected-embedding in WKV loop)
    emb = loaded.embedding_weight
    print(f"[smoke] emb.weight.shape={tuple(emb.shape)}")


def _shape(x):
    return tuple(x.shape) if isinstance(x, torch.Tensor) else "?"


def _dtype(x):
    return x.dtype if isinstance(x, torch.Tensor) else "?"


def _last_logits(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        return x[-1]
    return x[0, -1]  # [B, T, V]
