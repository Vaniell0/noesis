"""Free chat check: does the 0.4B answer ordinary questions at all, base vs S1 checkpoints, M=0 and M=1."""
import sys, torch
sys.path.insert(0, ".")
from experiments.rl.loader import load_rwkv7, load_weights_into
from experiments.rl.m0_pass_probe import answer
torch.set_num_threads(2)
M = "/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth"
R = "experiments/rl/results/s1_pilot_g1d04b"
Q = ["What is your favourite colour, and why?", "What car would you like to have, and why?",
     "Name three animals that live in the ocean."]
for tag, w in [("base", None), ("ce_T1", f"{R}/ce_T1.pth"), ("kl_T1", f"{R}/kl_T1.pth"), ("kl_T2", f"{R}/kl_T2.pth")]:
    loaded = load_rwkv7(M, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    if w:
        load_weights_into(loaded, w)
    for q in Q:
        for ticks in (0, 1):
            resp, _ = answer(loaded, loaded.tokenizer, q, 80, cut_blank=True, ticks=ticks)
            print(f"[{tag} M={ticks}] {q}\n    {resp[:300]!r}", flush=True)
