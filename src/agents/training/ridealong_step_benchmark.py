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

What it reports, per configuration (`CONFIGS`: the four heads at the baseline spec, each RND variant
added alone, all of them): ms per ride-along step (median of K, after warm-up) at batch `--batch`,
and one whole epoch-0 PASS — the minibatches of one epoch of a 98,304-row rollout at that batch
(the one pass the heads train on, `RIDEALONG_EPOCHS`) into one accumulator, plus the once-per-update
`decay` pull and the per-update `metrics()` CPU meters — against the 67.05 s update the end-to-end
benchmark measured for arm C (MEASURED 2026-09-30, `PREREGISTRATION.md` "Overhead"). A variant's
overhead is its row minus `core` (`per_variant_minus_core`). Every optimizer is acquired before the
timing (`_ridealong_acquire`). BENCHMARKS WARN, THEY NEVER STRETCH: the box contention is stamped
into the output.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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


#: The configurations timed: the four heads at the baseline spec, each RND variant added ALONE, all of
#: them (the X26 argv), and all but one (LEAVE-ONE-OUT). A variant's cost alone is `+v` − `core`;
#: its MARGINAL cost inside the X26 configuration is `+all` − `-v`. The two differ because the
#: variants share fixed costs (the identification probe's chimera target forward, one backward).
CONFIGS = ("core", "+fast", "+decay", "+small", "+feat", "+all",
           "-fast", "-decay", "-small", "-feat")


def _spec(config: str) -> Any:
    from agents.model.ridealong_heads import RND_VARIANTS, RideAlongSpec

    if config == "core":
        names: Tuple[str, ...] = ()
    elif config == "+all":
        names = RND_VARIANTS
    elif config.startswith("-"):
        names = tuple(v for v in RND_VARIANTS if v != config[1:])
    else:
        names = (config[1:],)
    return RideAlongSpec(ensemble=5, rnd=True, adv=5, opp=5, rnd_variants=names)


def run(device: str, batch: int, k: int, warmup: int,
        configs: Sequence[str] = CONFIGS, passes: int = 3) -> Dict[str, Any]:
    """Per config: (1) the median ride-along STEP (k calls after ``warmup``, CUDA-synchronised);
    (2) one whole epoch-0 PASS — ``ceil(ROLLOUT_ROWS / batch)`` steps into ONE accumulator, the
    once-per-update `begin_update_` pull and the per-update `metrics()` (rank meters, quantiles
    over every row on the CPU) — median of ``passes``. The pass is what the heads add to an update."""
    import torch as th

    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.ridealong_heads import build_ridealong
    from agents.training.instrumented_ppo.ridealong_terms import (RIDEALONG_EPOCHS,
                                                                  RideAlongAccumulator)
    from utils.contention import describe_contention

    pol, layout = _production_policy()
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
    cuda = device.startswith("cuda")
    sync = th.cuda.synchronize if cuda else (lambda: None)
    minibatches = int(np.ceil(ROLLOUT_ROWS / batch))
    contention_start = describe_contention()
    # The real learner calls the step right after `evaluate_actions`, on the stashes that forward
    # left; the rows are fixed here, so one untimed no-grad forward serves every call.
    with th.no_grad():
        values, _, _ = pol.evaluate_actions(obs, actions, action_masks=masks)
    sync()
    out: Dict[str, Any] = {}
    for cfg in configs:
        heads = build_ridealong(pol.features_extractor, obs_dim=int(layout["total_dim"]),
                                spec=_spec(cfg))
        assert heads is not None
        pol.ridealong = heads.to(device)
        learner = _learner(pol)
        learner._ridealong_acquire()          # the startup acquisition: nothing is built while timed
        acc = RideAlongAccumulator()
        times = []
        for i in range(warmup + k):
            sync()
            t0 = time.perf_counter()
            learner._ridealong_update(rd, values, actions, 0, acc)
            sync()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
        pass_s = []
        for _ in range(passes):
            acc = RideAlongAccumulator()
            sync()
            t0 = time.perf_counter()
            for _ in range(minibatches):
                learner._ridealong_update(rd, values, actions, 0, acc)
            sync()
            acc.metrics()
            pass_s.append(time.perf_counter() - t0)
        step_ms = 1000.0 * statistics.median(times)
        added_s = statistics.median(pass_s) * RIDEALONG_EPOCHS
        out[cfg] = {"step_ms": {"median": step_ms, "min": 1000.0 * min(times),
                                "max": 1000.0 * max(times)},
                    "pass_s": {"median": statistics.median(pass_s), "all": pass_s},
                    "added_s_per_update": added_s,
                    "added_pct_of_update": 100.0 * added_s / BASELINE_UPDATE_S}
        pol.ridealong = None
        del learner
        if cuda:
            th.cuda.empty_cache()
    per_variant = {}
    marginal = {}
    if "+all" in out:
        for cfg in configs:
            if cfg.startswith("-"):
                marginal[cfg[1:]] = {
                    "step_ms": out["+all"]["step_ms"]["median"] - out[cfg]["step_ms"]["median"],
                    "added_s_per_update": out["+all"]["added_s_per_update"]
                    - out[cfg]["added_s_per_update"],
                    "added_pct_of_update": out["+all"]["added_pct_of_update"]
                    - out[cfg]["added_pct_of_update"]}
    if "core" in out:
        for cfg in configs:
            if cfg.startswith("+"):
                per_variant[cfg] = {
                    "step_ms": out[cfg]["step_ms"]["median"] - out["core"]["step_ms"]["median"],
                    "added_s_per_update": out[cfg]["added_s_per_update"]
                    - out["core"]["added_s_per_update"],
                    "added_pct_of_update": out[cfg]["added_pct_of_update"]
                    - out["core"]["added_pct_of_update"]}
    return {"device": device, "batch": batch, "k": k, "warmup": warmup, "passes": passes,
            "ridealong_epochs": RIDEALONG_EPOCHS, "minibatches_per_epoch": minibatches,
            "baseline_update_s": BASELINE_UPDATE_S, "configs": out,
            "per_variant_minus_core": per_variant,
            "marginal_in_all": marginal,
            "contention_at_start": contention_start,
            "contention_at_end": describe_contention(),
            "loadavg_end": Path("/proc/loadavg").read_text().strip()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help=f"comma list of {','.join(CONFIGS)}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = run(a.device, a.batch, a.k, a.warmup, configs=tuple(a.configs.split(",")),
              passes=a.passes)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
