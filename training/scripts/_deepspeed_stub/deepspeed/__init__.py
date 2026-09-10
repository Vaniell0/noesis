"""Stub package — real deepspeed is not installed on this box and is not
needed for our config (--grad_cp 0, --strategy auto, no DeepSpeed
optimizers). Only exists so `import deepspeed` and
`from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam` (unconditional
imports in the vendored RWKV-PEFT tree) don't crash at import time. Any
actual call into these would be a real bug, not a case this stub handles."""


class _Checkpointing:
    @staticmethod
    def checkpoint(*args, **kwargs):
        raise NotImplementedError(
            "deepspeed stub: grad_cp=1 path requires real deepspeed, not installed"
        )


checkpointing = _Checkpointing()
