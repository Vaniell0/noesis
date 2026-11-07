"""hf_loader.py - a Hugging Face causal LM behind the same stateful interface as rl.loader's peft LoadedModel.

The bridge to transformers (plan B.3c, owner 2026-10-28): the S1 loop, its trainer and its gate probes run unchanged
on Qwen3.5 (18 Gated DeltaNet + 6 full-attention layers in the 0.8B). What the probes use, and how it maps:

    new_state()                   an empty HF DynamicCache, advanced by each forward and returned as the state
    forward_stateful(ids, st)     model(input_ids, past_key_values=st) -> (logits, cache)
    forward_stateful_embeds(...)  model(inputs_embeds, ...); return_hidden -> the fed latent, see below
    embedding_weight              the input embedding matrix
    silent_prompt(q)              the chat template with thinking switched off (RWKV: "User: ...\\n\\nAssistant: <think>\\n</think>\\n")
    eos_ids                       end-of-turn ids (RWKV: token 0)
    feed_view(x)                  what the first layer sees of a fed vector (RWKV: ln0(x); here x itself)

The fed latent. RWKV applies ln0 to every input, so only the direction of a fed vector reaches the model. Qwen adds the
input straight to the residual stream, and its last hidden is ~200x a token row (135 vs 0.63 on the 0.8B). So the
latent fed back is the direction of the last hidden (after the final norm, what the head reads), scaled to the mean
embedding-row norm - the same "direction only" contract as RWKV, at the scale the first layer expects.
NOESIS_HF_FEED=soft feeds the expected token embedding instead, softmax(logits) @ E (cot_vs_routes_probe's lat route): a point inside the embedding
cloud (the embeddings are tied, so this is the head's own read of the state). The hidden feed is a shock on Qwen: an
M=0-trained checkpoint drops from 195 to 28 of 289 when fed 4 ticks, and the ticks arm starts S1 at loss 6.1 vs 2.6.
That is not noise: the head is tied, so either feed is read as the next token, and the base spells its answer inside
the ticks (soft: "H", "ANO", "I", end) and has nothing left to say after them (base at M=4: 26 hidden / 23 soft vs 104).
NOESIS_HF_FRAME=think puts the ticks where Qwen's own thinking goes: "<think>\n" + ticks + "\n</think>\n\n", then the
answer (tick_frame); at M=0 this is exactly the no-thinking template.

Gradient through the ticks. transformers updates the DeltaNet conv and recurrent states in place (`copy_`, and a fused
single-token conv update), which breaks backward through a tick (the S1 trainer's BPTT). `_functional_linear_cache`
makes those updates replace the tensor instead, and `new_state` sets `record_past` on the linear layers so the
single-token path goes through the same functional update. A state is still one object advanced by each forward:
good for one sequential run (every caller here starts from new_state()). latent_noise's batched rollouts copy RWKV
state fields and do not work here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _functional_linear_cache() -> None:
    from transformers import cache_utils as cu
    layer = cu.LinearAttentionLayer
    if getattr(layer, "_noesis_functional", False):
        return

    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
        k = self.conv_kernel_size[state_idx]
        if not self.has_previous_state[state_idx]:
            full = conv_states
            self.has_previous_state[state_idx] = True
            if full.shape[-1] < k:
                full = F.pad(full, (k - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        self.conv_states[state_idx] = full[..., -k:]          # keep the last k, a new tensor (no copy_)
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    layer.update_conv_state, layer.update_recurrent_state = update_conv_state, update_recurrent_state
    layer._noesis_functional = True


class _Tok:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, s: str) -> list:
        return self.tok.encode(s, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)


class HFLoaded:
    backend = "hf"

    def __init__(self, path: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        if device == "cpu":
            # rwkv-fla ships an `fla` package; transformers picks its Triton DeltaNet kernels when `fla` imports, and
            # they need bf16 on a GPU. Hiding it makes the Qwen3.5 layers use their PyTorch reference path on the CPU.
            sys.modules["fla"] = None
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _functional_linear_cache()
        self._hf_tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.tokenizer = _Tok(self._hf_tok)
        self.device, self.dtype = device, dtype
        self.feed = os.environ.get("NOESIS_HF_FEED", "hidden")
        assert self.feed in ("hidden", "soft"), self.feed
        self.frame = os.environ.get("NOESIS_HF_FRAME", "closed")
        assert self.frame in ("closed", "think"), self.frame
        self.vocab_size = self.model.get_input_embeddings().weight.shape[0]
        ends = {self._hf_tok.eos_token_id, self._hf_tok.convert_tokens_to_ids("<|im_end|>")}
        self.eos_ids = frozenset(i for i in ends if isinstance(i, int) and i >= 0)
        with torch.no_grad():
            self.feed_scale = float(self.embedding_weight.float().norm(dim=-1).mean())

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    def silent_prompt(self, q: str) -> str:
        return self._hf_tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)

    def think_prompt(self, q: str) -> str:
        """The same turn with thinking on: the model writes its own CoT, then </think>, then the answer."""
        return self._hf_tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)

    def tick_frame(self, q: str) -> tuple:
        """(text before the ticks, text after them). closed: ticks after the empty think block (as RWKV's silent
        prompt); think: ticks inside the think block, closed after the last tick."""
        if self.frame == "think":
            return self.think_prompt(q), "\n</think>\n\n"
        return self.silent_prompt(q), ""

    def feed_view(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def new_state(self, batch: int = 1):
        from transformers import DynamicCache
        from transformers.cache_utils import LinearAttentionLayer
        cache = DynamicCache(config=self.model.config)
        for layer in cache.layers:
            if isinstance(layer, LinearAttentionLayer):
                layer.record_past = True                       # route single-token steps through the functional update
        return cache

    def forward_stateful(self, input_ids: torch.Tensor, state):
        out = self.model(input_ids=input_ids.to(self.device), past_key_values=state, use_cache=True)
        return out.logits, out.past_key_values

    def _layers(self):
        return next(m for n, m in self.model.named_modules()
                    if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) > 1)

    def forward_patched(self, input_ids: torch.Tensor, state, src: int, dst: int, src_ids: torch.Tensor | None = None):
        """Back-patching (Biran et al. 2024, "Hopping Too Late"): the residual after `src` layers at the last position
        (taken from `src_ids`, default the same prompt) replaces the residual entering layer `dst` at the last position,
        then the prompt runs on into `state`. dst=0 is where a tick is fed today (the input embedding); a mid-layer dst
        is the candidate feed point for the loop. Returns (logits, state) like forward_stateful."""
        layers, cap = self._layers(), {}

        def grab(mod, args, out):
            cap["h"] = (out[0] if isinstance(out, tuple) else out)[:, -1].detach().clone()

        hk = layers[src - 1].register_forward_hook(grab)
        try:
            ids = input_ids if src_ids is None else src_ids
            self.model(input_ids=ids.to(self.device), past_key_values=self.new_state(), use_cache=True)
        finally:
            hk.remove()

        def put(mod, args, kwargs):
            h = args[0].clone()
            h[:, -1] = cap["h"]
            return (h,) + tuple(args[1:]), kwargs

        hk = layers[dst].register_forward_pre_hook(put, with_kwargs=True)
        try:
            return self.forward_stateful(input_ids, state)
        finally:
            hk.remove()

    def forward_stateful_embeds(self, inputs_embeds: torch.Tensor, state, return_hidden: bool = False):
        out = self.model(inputs_embeds=inputs_embeds.to(self.dtype), past_key_values=state, use_cache=True,
                         output_hidden_states=return_hidden)
        if not return_hidden:
            return out.logits, out.past_key_values
        if self.feed == "soft":
            h = torch.softmax(out.logits.float(), dim=-1).to(self.dtype) @ self.embedding_weight
        else:
            h = F.normalize(out.hidden_states[-1], dim=-1) * self.feed_scale
        return out.logits, out.past_key_values, h


def is_hf_dir(path) -> bool:
    p = Path(str(path)).expanduser()
    return p.is_dir() and (p / "config.json").exists()


def load_hf(path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> HFLoaded:
    return HFLoaded(str(Path(str(path)).expanduser()), device, dtype)
