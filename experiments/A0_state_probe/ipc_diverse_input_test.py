#!/usr/bin/env python3
"""Point-4 test from this session's original capacity/decode framework:
achieved IPC (greedy self-generated continuation, the project's existing
convention) vs IPC on a genuinely diverse, teacher-forced continuation —
does the low reported IPC reflect a real channel ceiling, or just that
the model narrows its own input by self-generating a narrow, self-similar
continuation greedily?

CPU-only, real G1i-2.9b checkpoint, via the BlinkDL loader
(experiments/_common/model.py) - confirmed CPU-capable, unlike the
rl/loader.py GPU-only stack used elsewhere this session.
"""
import sys
import time
sys.path.insert(0, "/home/vaniello/Desktop/projects/noesis")

import numpy as np
from experiments._common.model import load_model
from experiments.A0_state_probe.ipc_analysis import collect_trajectory, compute_ipc, PROMPT

MODEL = "/home/vaniello/Desktop/projects/noesis/models/rwkv7-g1i-2.9b-20260805-ctx16384.pth"
N_TOKENS = 96
N_PROJ = 64
MAX_LAG = 8
MAX_DEGREE = 2

DIVERSE_TEXT = (
    "The recipe calls for two cups of flour, a pinch of salt, and fresh rosemary. "
    "Meanwhile in the harbor, fishing boats returned early because of the storm warning. "
    "The 1848 revolutions swept across Europe, toppling several monarchies within months. "
    "She scored the winning goal in the final minute, sending the crowd into celebration. "
    "Mount Kilimanjaro's glaciers have retreated significantly over the past century. "
    "The violin section entered on the third measure, a half-step above the cellos. "
    "Traffic on the coastal highway slowed near the bridge repairs outside the old town. "
    "A new species of beetle was catalogued in the rainforest canopy last spring. "
    "The committee postponed the vote until further budget figures were available. "
    "Lightning struck the old oak twice in one summer, splitting it down the middle."
)

print("Loading model (CPU, bf16)...", flush=True)
t0 = time.time()
model, tokenizer = load_model(MODEL, device="cpu")
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

results = {}

print("\n=== Baseline: greedy self-generated continuation ===", flush=True)
t0 = time.time()
tok_ids_base, states_base = collect_trajectory(
    model, tokenizer, PROMPT, n_tokens=N_TOKENS, target_layers=None,
    n_proj=N_PROJ, device="cpu", seed=42, teacher_forced_tokens=None,
)
ipc_base, basis_bound = compute_ipc(tok_ids_base, states_base, MAX_LAG, MAX_DEGREE)
print(f"  done in {time.time()-t0:.1f}s", flush=True)
results["baseline_greedy"] = {
    "token_ids": tok_ids_base,
    "ipc_total_by_layer": {int(l): v["IPC_total"] for l, v in ipc_base.items()},
    "basis_bound": basis_bound,
}

print("\n=== Diversified: teacher-forced varied real text ===", flush=True)
diverse_ids_full = tokenizer(DIVERSE_TEXT, return_tensors="pt")["input_ids"][0].tolist()
if len(diverse_ids_full) < N_TOKENS:
    raise SystemExit(f"DIVERSE_TEXT only tokenizes to {len(diverse_ids_full)} tokens, need {N_TOKENS} — extend it")
diverse_ids = diverse_ids_full[:N_TOKENS]
t0 = time.time()
tok_ids_div, states_div = collect_trajectory(
    model, tokenizer, PROMPT, n_tokens=N_TOKENS, target_layers=None,
    n_proj=N_PROJ, device="cpu", seed=42, teacher_forced_tokens=diverse_ids,
)
ipc_div, _ = compute_ipc(tok_ids_div, states_div, MAX_LAG, MAX_DEGREE)
print(f"  done in {time.time()-t0:.1f}s", flush=True)
results["diversified_teacher_forced"] = {
    "token_ids": tok_ids_div,
    "ipc_total_by_layer": {int(l): v["IPC_total"] for l, v in ipc_div.items()},
    "basis_bound": basis_bound,
}

print("\n=== Comparison (IPC_total by layer, basis_bound={}) ===".format(basis_bound), flush=True)
for l in sorted(ipc_base.keys()):
    b = results["baseline_greedy"]["ipc_total_by_layer"][l]
    d = results["diversified_teacher_forced"]["ipc_total_by_layer"][l]
    print(f"  layer {l:3d}: greedy={b:.3f}  diverse={d:.3f}  ratio={d/max(b,1e-6):.2f}x")

import json
out_path = "/home/vaniello/Desktop/projects/noesis/experiments/A0_state_probe/results/ipc_diverse_input_g1i.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}", flush=True)
