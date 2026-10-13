"""
benchmark_latency_v6.py: single-sequence inference latency for the v6 teacher and student.

Run in the same Kaggle notebook as the v6 training scripts, after both
checkpoints exist. Measures batch size 1, one sequence of shape (1, SEQ_LEN, FEATURE_DIM),
with 20 warm-up iterations and 200 timed trials, on the GPU (if present) and on the CPU.
Results are printed and saved to latency_v6.json next to the checkpoints.
"""

import sys
_WORKING_DIR = "/kaggle/working"
if _WORKING_DIR not in sys.path:
    sys.path.insert(0, _WORKING_DIR)

import json
import time
from pathlib import Path

import numpy as np
import torch

from config import FEATURE_DIM, SEQ_LEN
from model_tcn_attention import TCNAttentionHAR

CKPT_DIR = Path("/kaggle/working/handwash_tcn_dataset_v6/checkpoints")
WARMUP, TRIALS = 20, 200
CTOR_KEYS = ["input_dim", "num_classes", "branch_kernels", "branch_channels",
             "dilations", "fusion_dim", "gru_hidden", "dropout"]


def load_model(name):
    ckpt = torch.load(CKPT_DIR / f"{name}_best.pt", map_location="cpu", weights_only=False)
    cfg = {k: ckpt["model_config"][k] for k in CTOR_KEYS}
    model = TCNAttentionHAR(**cfg, use_feature_attn=True, use_layer_norm=False)
    model.load_state_dict(ckpt["model"])
    return model.eval(), sum(p.numel() for p in model.parameters())


def time_model(model, device):
    model = model.to(device)
    x = torch.randn(1, SEQ_LEN, FEATURE_DIM, device=device)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        sync()
        for _ in range(TRIALS):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "std_ms": float(t.std()),
            "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99))}


def main():
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    torch.set_num_threads(4)  # same core count as a Raspberry Pi 4B
    results = {}
    for name in ("teacher", "student"):
        model, n_params = load_model(name)
        results[name] = {"parameters": n_params}
        for dev in devices:
            results[name][dev] = time_model(model, dev)
    for dev in devices:
        t, s = results["teacher"][dev]["mean_ms"], results["student"][dev]["mean_ms"]
        results[f"speedup_{dev}"] = t / s
    print(json.dumps(results, indent=2))
    with open(CKPT_DIR / "latency_v6.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {CKPT_DIR / 'latency_v6.json'}")


if __name__ == "__main__":
    main()
