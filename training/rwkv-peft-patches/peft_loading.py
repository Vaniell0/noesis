import os
import torch
import importlib
import json
from rwkvt.lightning_train.light_rwkv import RWKV
from rwkvt.args_type import TrainingArgs
from rwkvt.lightning_train.trainer import generate_init_weight
from accelerate import init_empty_weights

from peft import get_peft_model, LoraConfig, TaskType
from peft import *
try:
    from peft import BoneConfig  # noqa: F401 — vendored code path, unused for lora
except ImportError:
    BoneConfig = None
try:
    from peft import MissConfig  # noqa: F401 — vendored code path, unused for lora
except ImportError:
    MissConfig = None
if "7" in os.environ["RWKV_MY_TESTING"]:
    from rwkvt.rwkv7.model import RWKV7 as RWKVModel
elif "6" in os.environ["RWKV_MY_TESTING"]:
    from rwkvt.rwkv6.model import RWKV6 as RWKVModel
elif "5" in os.environ["RWKV_MY_TESTING"]:
    from rwkvt.rwkv5.model import RWKV5 as RWKVModel
else:
    raise ValueError(f"Unsupported model version: . Valid options: 5,6,7")


class RWKVConfig:
    def __init__(self, n_embd=2048, n_layer=24):
        self.model_type = "rwkv"
        self.tie_word_embeddings = False
        self.n_embd = n_embd
        self.n_layer = n_layer

    def get(self, key, default=None):
        return getattr(self, key, default)

def load_peft_model(args: TrainingArgs):
    with init_empty_weights():
        model = RWKVModel(args)
    model = RWKVModel(args)
    state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
    print(f"########## Loading {args.load_model}... ##########")
    model.load_state_dict(state_dict, strict=(not True), assign=True)
    if os.environ["RWKV_TRAIN_TYPE"] == 'state':
        
        model = RWKV(args, model=model)
        model.requires_grad_(False)
        for name, module in model.named_modules():
            for pname, param in module.named_parameters():
                if 'state' in pname:
                    param.requires_grad = True
            break
    elif args.peft!='none':
       
        model.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)

        # if args.peft == 'lora':
        #     peft_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         r=args.lora_config['lora_r'],
        #         lora_alpha=args.lora_config['lora_alpha'],
        #         lora_dropout=args.lora_config['lora_dropout'],
        #         target_modules=["receptance", "key", "value", "output"],
        #     )
        # elif args.peft == 'miss':
        #     peft_config = MissConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     r=args.miss_config['r'],
        #     target_modules=["receptance", "key", "value", "output"],
        # )
        
        # === 动态加载 PEFT Config 类 ===
        peft_dict={
            "lora": LoraConfig,
            "miss": MissConfig,
            "adalora": AdaLoraConfig,
            "prefix": PrefixTuningConfig,
        }
        ConfigClass = peft_dict[args.peft]

        peft_args = json.loads(args.peft_config)
        peft_config = ConfigClass(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["receptance","key","value","output"],
            **peft_args)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        resume_path = os.environ.get("NOESIS_RESUME_LORA", "")
        if resume_path:
            print(f"[peft_loading] resuming LoRA from {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
            own = dict(model.named_parameters())
            loaded = skipped = 0
            for name, tensor in ckpt.items():
                if name in own and own[name].shape == tensor.shape:
                    own[name].data.copy_(tensor)
                    loaded += 1
                else:
                    skipped += 1
            print(f"[peft_loading] LoRA resume: {loaded} loaded, {skipped} skipped")

        model = RWKV(args, model=model)

    else:
        # Neither RWKV_TRAIN_TYPE=='state' nor args.peft!='none' — plain
        # full-FT (--peft none). Found 2026-09-10: this bare RWKVModel was
        # never wrapped in the Lightning `RWKV` module on this path, so
        # `trainer.fit()` crashed with "model must be a LightningModule,
        # got RWKV7" the first time full-FT (no peft) was actually tried
        # through this trainer. All params stay trainable (default
        # requires_grad=True, untouched here), matching the other two
        # branches' behavior for their own trainable subsets.
        model = RWKV(args, model=model)

    return args, model
