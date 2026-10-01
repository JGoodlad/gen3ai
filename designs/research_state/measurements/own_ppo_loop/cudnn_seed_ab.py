"""Stage 2 unit 3 (`gen3_owned_seeding_v1`): what did sb3's `cudnn.deterministic = True` cost or change?

The ONLY difference owned seeding makes on CUDA is that `_setup_model` no longer flips the process-wide
`torch.backends.cudnn.deterministic` (True -> left at torch's default False; `benchmark` stays False
either way). So the before/after A/B is that one flag, INTERLEAVED in ONE process on identical work:
arm C's weights (`ai_v14_02_lbat_ctrl/final_model.zip`), the learner benchmark's pinned real rollout
buffer (98,304 rows, the K8 acceptance's), C's shape (micro 2,048 x accumulation 32), EAGER (no
compile: backend selection — the only place a cuDNN kernel could enter — is the dispatcher's in both),
fp32 ('highest'), `--epochs` per update. Every repeat restores the same params, optimizer state,
buffer and RNG seeds. Reported per arm: the update's wall time (cuda-synchronised), `train/loss` /
`train/approx_kl`, the post-update parameter sha, and from one PROFILED update per arm the count and
time of kernels whose name mentions cudnn.

    scripts/ops/gpu_lock.sh timeout 1100 scripts/ops/mem_cap.sh 40 env PYTHONPATH=src \\
        python designs/research_state/measurements/own_ppo_loop/cudnn_seed_ab.py --out <json>
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch as th

BUF = Path.home() / "gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl"
N_STEPS, N_ENVS, MICRO, ACCUM = 2048, 48, 2048, 32


def _c_checkpoint() -> Path:
    from utils.paths import main_models_dir
    root = main_models_dir()
    assert root is not None, "no models/ archive"
    return root / "ai_v14_02_lbat_ctrl" / "final_model.zip"


def build(epochs: int) -> Any:
    from stable_baselines3.common.logger import configure

    from agents.training import learner_golden as LG
    from agents.training.learner_benchmark import load_buffer, restore_buffer_state

    model = LG.build_learner()
    with zipfile.ZipFile(_c_checkpoint()) as z:
        sd = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    missing, unexpected = model.policy.load_state_dict(sd, strict=False)
    assert not unexpected, f"C's checkpoint carries params this surface lacks: {unexpected[:5]}"
    dev = th.device("cuda")
    model.device = dev
    model.policy.to(dev)
    b = model.rollout_buffer
    model.n_steps, model.n_envs = N_STEPS, N_ENVS
    model.rollout_buffer = type(b)(N_STEPS, b.observation_space, b.action_space, device=dev, gamma=b.gamma,
                                   gae_lambda=b.gae_lambda, n_envs=N_ENVS)
    saved = load_buffer(BUF)
    model.batch_size, model.grad_accum_steps, model.n_epochs = MICRO, ACCUM, epochs
    model.behaviour_check = "off"
    model._logger = configure(None, [])
    model._current_progress_remaining = 1.0
    return model, saved, restore_buffer_state, missing


def sha(model: Any) -> str:
    import hashlib
    h = hashlib.sha256()
    for n, p in model.policy.named_parameters():
        h.update(n.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def one(model: Any, saved: Any, restore: Any, st0: Any, deterministic: bool, profile: bool) -> Dict[str, Any]:
    from agents.training.learner_benchmark import restore_model_state
    # weights, optimizer AND every plain counter (`_n_updates` drives the --diagnostics-every cadence:
    # without it, repeats ran different optional probes — the first read's confound)
    restore_model_state(model, st0)
    restore(model.rollout_buffer, saved["buffer"])
    for k, v in saved.get("stash", {}).items():
        setattr(model, k, v)
    th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = deterministic, False
    np.random.seed(123)
    th.manual_seed(123)
    th.cuda.synchronize()
    out: Dict[str, Any] = {"deterministic": deterministic}
    if profile:
        from torch.profiler import ProfilerActivity, profile as prof
        with prof(activities=[ProfilerActivity.CUDA]) as p:
            model.train()
            th.cuda.synchronize()
        evs = [e for e in p.events() if e.device_type.name == "CUDA"]
        cud = [e for e in evs if "cudnn" in e.name.lower()]
        out.update({"cuda_kernels": len(evs), "cudnn_kernels": len(cud),
                    "cudnn_kernel_us": float(sum(e.device_time_total for e in cud)),
                    "cudnn_names": sorted({e.name for e in cud})[:10]})
    else:
        t0 = time.perf_counter()
        model.train()
        th.cuda.synchronize()
        out["update_s"] = time.perf_counter() - t0
    lv = model.logger.name_to_value
    out.update({"loss": float(lv.get("train/loss", float("nan"))),
                "approx_kl": float(lv.get("train/approx_kl", float("nan"))), "params": sha(model)})
    model.logger.name_to_value.clear()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--order", default="TFFT", help="timed repeats: T = deterministic (sb3's), F = not")
    ap.add_argument("--profile", action="store_true", help="also one profiled update per setting")
    a = ap.parse_args()
    prev = (th.backends.cudnn.deterministic, th.backends.cudnn.benchmark)
    t0 = time.perf_counter()
    model, saved, restore, missing = build(a.epochs)
    from agents.training.learner_benchmark import capture_model_state
    st0 = capture_model_state(model)
    order = [c == "T" for c in a.order]
    rows: List[Dict[str, Any]] = []
    try:
        one(model, saved, restore, st0, True, False)              # warm-up (allocator, kernels), discarded
        for det in order:                                          # counterbalanced (default T F F T)
            rows.append(one(model, saved, restore, st0, det, False))
        if a.profile:
            for det in (True, False):
                rows.append(one(model, saved, restore, st0, det, True))
    finally:
        th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = prev
    res = {"torch": th.__version__, "gpu": th.cuda.get_device_name(0), "epochs": a.epochs, "order": a.order,
           "missing_keys_from_C": list(missing)[:20], "setup_s": None, "rows": rows,
           "wall_s": time.perf_counter() - t0}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
