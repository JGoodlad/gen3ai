"""F-SZ-3: every STEP-COUNTED training cadence is a TOTAL-env-step quantity, independent of `--n-envs`.

The M5 SIZING study sweeps N (48 → 2048). A cadence counted in VEC-ENV CALLS silently scales with N:
the checkpointer's hardcoded 50 000 vec calls was 2.4M env steps at N = 48 and ~102M at N = 2048 —
a run that never checkpoints. So this resolves each cadence through the REAL resolution code
(`resolve_config` + `build_callbacks`, NOT `--debug`, which forces N = 1) at N = 48 and N = 2048 and
asserts the same total env-step interval (to within one vec step, the ceil the vec-call conversion
cannot avoid).

The audit behind the list (which cadence is total-step, which is per-UPDATE and why those are not
here) is `designs/ops/training_runbook.md` → "Cadences and N". Per-update cadences (diagnostics,
compile canary, adaptive batch, KL controller, team PFSP / win-rate pulls, distill-anchor refresh)
are N-independent in steps only while the rollout (`n_steps x n_envs`) is fixed — a sizing decision,
not something a resolver can make equal.

The second half pins the conversion's premise on the RUST env core: its collector fires the SB3
callbacks once per N trainee decisions, so one callback call is N env steps there too.

Fast and unmarked: argv resolution and callback construction only (no env, no model, no server).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

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
    args = p.parse_args(["--steps", "15000000", "--use-bridge", "node", "--n-envs", str(n_envs)])
    resolve_config(args, p)
    assert not args.debug, "a --debug build forces N = 1 and would make this test vacuous"
    bundle = build_callbacks(
        args=args, model_dir=str(tmp_path / f"n{n_envs}"), server_config=None,
        annealing_mode=False, _pool=None, _fixed_opponents=None, _bot_weight_vec=None,
        OPPONENT_CLASSES=(), _specialist_team_str=None, _promote_threshold=0.6,
        _heuristic_floor=0.0, _sp_start_wr=0.5, _sp_full_wr=0.9)
    ckpt = bundle.callbacks[0]
    assert type(ckpt).__name__ == "_TrackingCheckpointCallback"
    assert bundle.eval_callback is not None, "a non-debug run always builds the eval callback"
    eval_freq, _ = bundle.eval_callback._schedule()
    return {
        # SB3's `CheckpointCallback` fires on `n_calls % save_freq`, and one call is N env steps.
        "checkpoint": int(ckpt.save_freq) * n_envs,
        # Eval (and with it the snapshot-pool add and the opponent-pool refresh, both driven by the
        # eval cycle) compares `num_timesteps` — already total.
        "eval": int(eval_freq),
        # Compared against `num_timesteps` inside `capacity_telemetry` / the search teacher.
        "canary_reset": int(args.canary_reset_steps),
        "teacher_refresh": int(args.teacher_refresh_steps),
    }


def test_every_step_counted_cadence_is_the_same_total_env_steps_at_N48_and_N2048(tmp_path):
    a = _resolved(tmp_path, N_PRODUCTION)
    b = _resolved(tmp_path, N_SWEEP)
    assert a["checkpoint"] == DEFAULT_CHECKPOINT_ENV_STEPS, (
        "the production-N default must stay byte-identical: 50 000 vec calls x 48")
    assert 0 <= b["checkpoint"] - DEFAULT_CHECKPOINT_ENV_STEPS < N_SWEEP, (
        f"at N={N_SWEEP} the default checkpoint interval is {b['checkpoint']:,} env steps — a "
        f"vec-call cadence that scaled with N (F-SZ-3)")
    for key in ("eval", "canary_reset", "teacher_refresh"):
        assert a[key] == b[key], f"{key}: {a[key]:,} env steps at N=48 but {b[key]:,} at N=2048"


def test_an_explicit_interval_is_total_env_steps_too(tmp_path):
    from main.train.constants import checkpoint_interval_env_steps
    for n in (N_PRODUCTION, N_SWEEP):
        got = checkpoint_interval_env_steps(150_000, n)
        assert 0 <= got - 150_000 < n


# ---------------------------------------------------------------------------------------------
# The RUST env core: one callback call == N trainee decisions (the premise of the conversion)
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


class _CountingCallback:
    def __init__(self, model: Any, stop_at: int) -> None:
        self.model, self.stop_at, self.n_calls = model, stop_at, 0
        self.timesteps_at_call: List[int] = []

    def on_rollout_start(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        pass

    def on_step(self) -> bool:
        self.n_calls += 1
        self.timesteps_at_call.append(int(self.model.num_timesteps))
        return self.n_calls < self.stop_at


class _Model:
    num_timesteps = 0
    env = None

    def _update_info_buffer(self, infos, dones) -> None:
        pass


@pytest.mark.parametrize("n_envs", [N_PRODUCTION, N_SWEEP])
def test_the_rust_collector_fires_one_callback_per_N_trainee_decisions(n_envs):
    """A host step returns a VARIABLE number of trainee decisions (only envs whose p1 must act);
    the callbacks must still fire once per N of them, so `save_freq` vec calls = `save_freq x N`
    env steps on the Rust core exactly as on the Python one."""
    rng = np.random.default_rng(n_envs)
    script = [int(k) for k in rng.integers(0, n_envs + 1, size=4000)]
    model = _Model()
    col = _counting_collector(n_envs, script)
    cb = _CountingCallback(model, stop_at=50)
    assert col.collect(model, cb, rollout_buffer=None) is False
    assert cb.n_calls == 50
    for i, t in enumerate(cb.timesteps_at_call, start=1):
        # at the i-th call at least i*N decisions have been counted, and fewer than one host step
        # (≤ N decisions) beyond the call's boundary
        assert i * n_envs <= t < i * n_envs + 2 * n_envs
