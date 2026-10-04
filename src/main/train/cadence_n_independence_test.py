"""F-SZ-3: every STEP-COUNTED training cadence is a TOTAL-env-step quantity, independent of `--n-envs`.

The M5 SIZING study sweeps N (48 → 2048). A cadence counted in VEC-ENV CALLS silently scales with N:
the checkpointer's hardcoded 50 000 vec calls was 2.4M env steps at N = 48 and ~102M at N = 2048 —
a run that never checkpoints. So this resolves each cadence through the REAL resolution code
(`resolve_config` + `build_callbacks`, NOT `--debug`, which forces N = 1) at N = 48 and N = 2048 and
asserts the same total env-step interval.

The audit behind the list (which cadence is total-step, which is per-UPDATE and why those are not
here) is `designs/ops/training_runbook.md` → "Cadences and N". Per-update cadences (diagnostics,
compile canary, adaptive batch, KL controller, team PFSP / win-rate pulls)
are N-independent in steps only while the rollout (`n_steps x n_envs`) is fixed — a sizing decision,
not something a resolver can make equal.

The second half drives the REAL checkpointer through the REAL Rust-collector loop (a scripted,
RAGGED host step: only the envs whose p1 must act return a decision) and asserts it saves at the
total-step boundaries. The async-wave and sync streams are `main/train_rl_agent_test.py`.

Fast and unmarked: argv resolution and callback construction only (no env, no model, no server).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.training.loop_callbacks import BaseCallback, CallbackList
from main.train_rl_agent import build_parser

N_PRODUCTION = 48
N_SWEEP = 2048
DEFAULT_CHECKPOINT_ENV_STEPS = 2_400_000


def _resolved(tmp_path, n_envs: int) -> Dict[str, Any]:
    """Every step-counted cadence, read off the CONSTRUCTED callbacks / the resolved args, as TOTAL
    env steps."""
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config

    p = build_parser()
    args = p.parse_args(["--steps", "15000000", "--n-envs", str(n_envs)])
    resolve_config(args, p)
    assert not args.debug, "a --debug build forces N = 1 and would make this test vacuous"
    bundle = build_callbacks(
        args=args, model_dir=str(tmp_path / f"n{n_envs}"),
        annealing_mode=False, _pool=None, _fixed_opponents=None, _bot_weight_vec=None,
        OPPONENT_CLASSES=(), _specialist_team_str=None, _promote_threshold=0.6,
        _heuristic_floor=0.0, _sp_start_wr=0.5, _sp_full_wr=0.9)
    ckpt = bundle.callbacks[0]
    assert type(ckpt).__name__ == "_TrackingCheckpointCallback"
    assert bundle.eval_callback is not None, "a non-debug run always builds the eval callback"
    eval_freq, _ = bundle.eval_callback._schedule()
    return {
        # A `num_timesteps` boundary (`constants.checkpoint_due`), not a call count.
        "checkpoint": int(ckpt.interval_env_steps),
        # Eval (and with it the snapshot-pool add and the opponent-pool refresh, both driven by the
        # eval cycle) compares `num_timesteps` — already total.
        "eval": int(eval_freq),
        # Compared against `num_timesteps` inside `capacity_telemetry`.
        "canary_reset": int(args.canary_reset_steps),
    }


def test_every_step_counted_cadence_is_the_same_total_env_steps_at_N48_and_N2048(tmp_path, run_archive):
    # `run_archive`: the eval callback's ledger writer (eval U2) is built at assembly, on the run archive
    a = _resolved(tmp_path, N_PRODUCTION)
    b = _resolved(tmp_path, N_SWEEP)
    assert a["checkpoint"] == DEFAULT_CHECKPOINT_ENV_STEPS, "the N = 48 value is the default"
    for key in ("checkpoint", "eval", "canary_reset"):
        assert a[key] == b[key], f"{key}: {a[key]:,} env steps at N=48 but {b[key]:,} at N=2048"


def test_an_explicit_interval_is_total_env_steps_too():
    from main.train.constants import checkpoint_interval_env_steps
    assert checkpoint_interval_env_steps(150_000) == 150_000


# ---------------------------------------------------------------------------------------------
# The RUST env core: the real checkpointer through the real collector loop
# ---------------------------------------------------------------------------------------------

def _counting_collector(n_envs: int, decisions_per_host_step: List[int]):
    """The REAL `RustCollector.collect` loop over a scripted host step — no core, no T2. Only the
    host step and the trigger are scripted; the callback cadence is the production code's."""
    from agents.training.rust_rollout.collector import CollectorStats, RustCollector

    class _Scripted(RustCollector):
        def __init__(self) -> None:                          # bypass the declared startup
            self.n = n_envs
            self._started = True
            self.stats = CollectorStats()
            self._pending_decisions = 0
            self._infos = []
            self._script = list(decisions_per_host_step)

        def ready(self) -> bool:
            return False                                     # the callback ends collection

        def host_step(self) -> int:
            return self._script.pop(0)

    return _Scripted()


class _StopAfter(BaseCallback):
    """Ends collection (`on_step` False) once `num_timesteps` reaches `stop_at`."""

    def __init__(self, stop_at: int) -> None:
        super().__init__()
        self.stop_at = stop_at

    def _on_step(self) -> bool:
        return int(self.model.num_timesteps) < self.stop_at


@pytest.mark.parametrize("n_envs", [N_PRODUCTION, N_SWEEP])
def test_the_rust_collector_checkpoints_at_total_step_boundaries(tmp_path, n_envs):
    """A ragged host step (0..N trainee decisions) and the callbacks fired once per N decisions:
    the checkpointer must save once per interval of TOTAL env steps, at the first call at or past
    each boundary (the collector may batch up to ~2N decisions into one call's advance)."""
    from main.train.run_io import _TrackingCheckpointCallback

    interval = 50 * n_envs + 7                     # deliberately NOT a multiple of N
    rng = np.random.default_rng(n_envs)
    script = [int(k) for k in rng.integers(0, n_envs + 1, size=200_000)]
    model = MagicMock()
    model.num_timesteps = 0
    model.env = None
    ckpt = _TrackingCheckpointCallback(interval_env_steps=interval, save_path=str(tmp_path),
                                       name_prefix="checkpoint")
    cbs = CallbackList([ckpt, _StopAfter(stop_at=10 * interval)])
    cbs.init_callback(model)
    cbs.on_training_start()
    col = _counting_collector(n_envs, script)
    assert col.collect(model, cbs, rollout_buffer=None) is False
    saved = [int(os.path.basename(c.args[0]).split("_")[1]) for c in model.save.call_args_list]
    assert len(saved) == 10, saved
    for k, step in enumerate(saved, start=1):
        assert k * interval <= step < k * interval + 2 * n_envs, (k, step)
