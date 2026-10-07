"""The checkpoint CADENCE.

The periodic checkpoint fires when `num_timesteps` crosses each multiple of a TOTAL-ENV-STEP
interval (`constants.checkpoint_due`; the callback is `run_io._TrackingCheckpointCallback`, whose
trigger tests live in `main/train_rl_agent_test.py`). It used to be SB3's `n_calls % save_freq`, a
CALL count: a hardcoded 50000 read as "50k steps" while at `--n-envs 48` it was 2,400,000 env steps.

Pinned here:

* **THE INTERVAL.** Unset `--checkpoint-every-steps` = 2.4M total env steps (the N = 48 value) at
  every N; a value is taken as total env steps, as asked — no rounding to a call count.
* A non-positive `--checkpoint-every-steps` is refused by the parser check.

Fast and unmarked: the cadence half is pure arithmetic, and the build half calls `resolve_config`
in-process under `--use-bridge node` (which skips the rust binary's cargo build).
"""
from __future__ import annotations

import pytest

from main.train.constants import (
    DEFAULT_CHECKPOINT_EVERY_ENV_STEPS, checkpoint_due, checkpoint_interval_env_steps,
)
from main.train_rl_agent import build_parser


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

class TestCadenceInterval:
    def test_unset_is_the_2_4M_total_env_step_default(self):
        assert DEFAULT_CHECKPOINT_EVERY_ENV_STEPS == 2_400_000
        assert checkpoint_interval_env_steps(None) == 2_400_000

    def test_a_value_is_total_env_steps_as_asked_with_no_call_rounding(self):
        assert checkpoint_interval_env_steps(150_000) == 150_000
        assert checkpoint_interval_env_steps(100_000) == 100_000     # not 2084 x 48

    def test_a_tiny_request_floors_at_one_env_step_not_zero(self):
        """`checkpoint_due` divides by it."""
        assert checkpoint_interval_env_steps(0) == 1
        assert checkpoint_due(0, 5, 0) is True


class TestBoundary:
    def test_due_iff_a_multiple_was_crossed(self):
        assert checkpoint_due(2_399_952, 2_400_000, 2_400_000)        # reached exactly
        assert checkpoint_due(2_399_990, 2_400_010, 2_400_000)        # stepped over
        assert not checkpoint_due(2_400_000, 2_400_048, 2_400_000)   # the boundary already seen
        assert not checkpoint_due(0, 2_399_999, 2_400_000)

    def test_one_save_per_boundary_whatever_the_call_size(self):
        """N steps, a wave, a ragged Rust host step: the saves are the boundaries crossed (for any
        call smaller than the interval — one call that crosses two boundaries saves once)."""
        for step in (1, 17, 48, 2048):
            t, saves = 0, 0
            while t < 24_000:
                saves += checkpoint_due(t, t + step, 2_400)
                t += step
            assert saves == t // 2_400, step


# ---------------------------------------------------------------------------
# The constructed callback — default preservation, end to end
# ---------------------------------------------------------------------------

def _checkpoint_callback_interval(model_dir, *flags) -> int:
    """`build_callbacks`'s ACTUAL checkpointer, for an argv — not a re-derivation of it.

    (`resolve_config` resolves — and, on a fresh worktree, builds — the rust `sim_bridge` binary; it has
    nothing to do with the cadence.)
    """
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config

    p = build_parser()
    args = p.parse_args(["--steps", "1", "--debug-eval", *flags])
    resolve_config(args, p)
    args.debug_eval = False          # skip the eval callback: this is about the checkpointer
    args.debug = True                # ... which needs _run_eval False; n_envs is read separately
    bundle = build_callbacks(
        args=args, model_dir=str(model_dir), annealing_mode=False,
        _pool=None, _fixed_opponents=None, _bot_weight_vec=None, OPPONENT_NAMES=(),
        _specialist_team_str=None, _promote_threshold=0.6, _heuristic_floor=0.0,
        _sp_start_wr=0.5, _sp_full_wr=0.9)
    return bundle.callbacks[0].interval_env_steps


class TestDefaultPreservation:
    """⚠️ These call `build_callbacks` with `args.debug` forced True purely to skip the eval
    callback; the non-debug N=48 / N=2048 build is `main/train/cadence_n_independence_test.py`. What
    these pin is that the constructed callback reads the flag at all."""

    def test_a_flagless_run_builds_the_byte_identical_checkpointer(self, tmp_path):
        """With no `--checkpoint-every-steps` the constructed callback's boundary spacing is the
        2.4M-total-env-step default."""
        assert _checkpoint_callback_interval(tmp_path, "--n-envs", "48") == 2_400_000

    def test_the_flag_changes_it_and_nothing_else_does(self, tmp_path):
        assert _checkpoint_callback_interval(
            tmp_path, "--n-envs", "48", "--checkpoint-every-steps", "500") == 500


# ---------------------------------------------------------------------------
# The parser check
# ---------------------------------------------------------------------------

class TestIntervalRefusal:
    def test_a_nonpositive_interval_is_refused_by_the_parser_check(self):
        p = build_parser()
        args = p.parse_args(["--steps", "1", "--checkpoint-every-steps", "0"])
        from main.train.config import resolve_config
        with pytest.raises(SystemExit):
            resolve_config(args, p)
