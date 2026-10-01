"""THE RIDE-ALONG STEP BENCHMARK — what the detached heads add to one PPO update, measured directly.

    scripts/ops/gpu_lock.sh python3 src/agents/training/ridealong_step_benchmark.py --device cuda

WHY IT EXISTS. The end-to-end learner benchmark (`learner_benchmark.py run --ridealong`) times a whole
`train()` call with and without the heads, but its trainer startup (env workers, the forkserver
compile preload, the compile-trainer parity gate) takes ~15 min on a quiet box and overran a 35-min
GPU-lock unit twice on 2026-09-30 at load 30–80. The heads' cost is SEPARABLE by construction: they
run eager, on `.detach()`ed stashes, with their own optimizer, before PPO's loss exists — so what
they add to an update is exactly (their step's time) × (the minibatches they train on). This times
that step directly, with CUDA synchronisation, on the REAL production policy and real-obs fixture
rows tiled to the production minibatch, through the learner's own `_ridealong_update`.

What it reports: ms per ride-along step (median of K, after warm-up) at batch `--batch`, and that
times the minibatches of one epoch of a 98,304-row rollout at that batch (the one pass the heads
train on, `RIDEALONG_EPOCHS`), against the 67.05 s update the end-to-end benchmark measured for arm C
(MEASURED 2026-09-30, `PREREGISTRATION.md` "Overhead"). BENCHMARKS WARN, THEY NEVER STRETCH: the box
contention is stamped into the output.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: arm C's update, the end-to-end benchmark's baseline (MEASURED 2026-09-30, 98,304 rows / 2048 x 2).
BASELINE_UPDATE_S = 67.05
ROLLOUT_ROWS = 98_304


def _learner(policy: Any) -> Any:
    """The slice of the PPO learner `_ridealong_update` reads: the real mixin, the real policy."""
    from agents.training.instrumented_ppo.ridealong_terms import RideAlongTerms

    class _Learner(RideAlongTerms):
        def __init__(self) -> None:
            self.policy = policy
            self.opp_intent_coef = 0.05
            self.logger = types.SimpleNamespace(record=lambda *a, **k: None)

    return _Learner()


def _production_policy() -> Any:
    import gymnasium as gym
    import torch as th

    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from main.fresh_checkpoint import _production_policy_kwargs

    _args, layout, pk = _production_policy_kwargs()
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (layout["total_dim"],), np.float32),
        "action_mask": gym.spaces.MultiBinary(11)})
    th.manual_seed(0)
    return Gen3DualHeadMaskablePolicy(obs_space, gym.spaces.Discrete(11), lambda _: 3e-4,
                                      **pk), layout


def run(device: str, batch: int, k: int, warmup: int) -> Dict[str, Any]:
    import torch as th

    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.ridealong_heads import RideAlongSpec, build_ridealong
    from agents.training.instrumented_ppo.ridealong_terms import (RIDEALONG_EPOCHS,
                                                                  RideAlongAccumulator)
    from utils.contention import describe_contention

    pol, layout = _production_policy()
    pol.ridealong = build_ridealong(pol.features_extractor, obs_dim=int(layout["total_dim"]),
                                    spec=RideAlongSpec(ensemble=5, rnd=True, adv=5, opp=5))
    pol = pol.to(device)
    rows, masks = load_parity_rows(layout["total_dim"])
    reps = int(np.ceil(batch / len(rows)))
    rows = np.tile(rows, (reps, 1))[:batch]
    masks = np.tile(masks, (reps, 1))[:batch]
    obs = {"observation": th.tensor(rows, device=device),
           "action_mask": th.tensor(masks.astype(np.float32), device=device)}
    actions = th.tensor([int(np.flatnonzero(m)[0]) for m in masks], device=device)
    n = batch
    rd = types.SimpleNamespace(
        observations={**obs, "win_target": (th.arange(n, device=device) % 2).float()[:, None],
                      "win_mask": th.ones(n, 1, device=device),
                      "opp_action_kind": (th.arange(n, device=device) % 2)[:, None].float(),
                      "opp_action_num": th.zeros(n, 1, device=device),
                      "opp_class": th.zeros(n, 1, device=device)},
        advantages=th.linspace(-0.2, 0.2, n, device=device))
    learner = _learner(pol)
    cuda = device.startswith("cuda")
    sync = th.cuda.synchronize if cuda else (lambda: None)
    contention_start = describe_contention()
    times = []      # epoch 0: the heads train AND every meter is read (the only epoch they run)
    for i in range(warmup + k):
        # The real learner calls the step right after `evaluate_actions`, on the stashes that forward
        # left — so run it (untimed, no grad: the heads read only detached stashes) before each call.
        with th.no_grad():
            values, _, _ = pol.evaluate_actions(obs, actions, action_masks=masks)
        sync()
        t0 = time.perf_counter()
        learner._ridealong_update(rd, values, actions, 0, RideAlongAccumulator())
        sync()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    step_ms = 1000.0 * statistics.median(times)
    minibatches = int(np.ceil(ROLLOUT_ROWS / batch))
    added_s = step_ms / 1000.0 * minibatches * RIDEALONG_EPOCHS
    return {"device": device, "batch": batch, "k": k, "warmup": warmup,
            "ridealong_epochs": RIDEALONG_EPOCHS,
            "step_ms_epoch0": {"median": step_ms, "min": 1000.0 * min(times),
                               "max": 1000.0 * max(times)},
            "minibatches_per_epoch": minibatches,
            "added_s_per_update": added_s,
            "baseline_update_s": BASELINE_UPDATE_S,
            "added_pct_of_update": 100.0 * added_s / BASELINE_UPDATE_S,
            "contention_at_start": contention_start,
            "contention_at_end": describe_contention(),
            "loadavg_end": Path("/proc/loadavg").read_text().strip()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = run(a.device, a.batch, a.k, a.warmup)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
