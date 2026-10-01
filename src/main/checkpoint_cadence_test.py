"""The checkpoint CADENCE conversion and the counterfactual DUTY-CYCLE refusal.

SB3's `CheckpointCallback.save_freq` counts VEC-ENV CALLS, not env steps, and that multiplier
(`* n_envs`) was invisible at a hardcoded call site for the whole of the R1 counterfactual work: at
`--n-envs 48` the "50k" cadence is 2,400,000 env steps, against a 150,000-step label staleness
bound. The label producer can only stamp labels with the newest checkpoint's step, so 93.75% of
them expired on arrival — observed on `ai_v9_29_rev1_0823` as 6 ingested against 255 expired in two
hours, with every counter on both sides reading healthy.

Two things are pinned here, and the first matters more than the second:

* **DEFAULT PRESERVATION AT THE PRODUCTION N.** A flagless `--n-envs 48` run must construct a
  checkpointer whose `save_freq` is byte-identical to the pre-flag one (50000 vec calls); since
  F-SZ-3 the default is 2.4M TOTAL env steps at every N, which is that value at N = 48.
* **The guard**: with both halves of the cf label path on, the duty cycle is COMPUTED, PRINTED, and
  refused below the floor — because a number nobody computes is exactly how this shipped.

Fast and unmarked: the cadence half is pure arithmetic, and the guard half calls `resolve_config`
in-process under `--use-bridge node` (which skips the rust binary's cargo build).
"""
from __future__ import annotations

import pytest

from main.exit_codes import TrainExitCode
from main.train.constants import (
    CF_DUTY_CYCLE_FLOOR, DEFAULT_CHECKPOINT_EVERY_ENV_STEPS, cf_label_duty_cycle,
    checkpoint_interval_env_steps, checkpoint_save_freq_vec_calls,
)
from main.train_rl_agent import build_parser


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

class TestCadenceConversion:
    def test_none_preserves_the_historical_value_at_the_production_n(self):
        """THE compatibility assertion: no flag at the production `--n-envs 48` ⇒ the literal 50000
        vec calls that were hardcoded. A resume of a 48-env run must not change its cadence."""
        assert checkpoint_save_freq_vec_calls(None, 48) == 50_000
        assert DEFAULT_CHECKPOINT_EVERY_ENV_STEPS == 2_400_000

    def test_none_is_a_TOTAL_env_step_interval_at_every_n(self):
        """F-SZ-3: the default is 2.4M TOTAL env steps, not 50000 x N (~102M at N=2048)."""
        for n_envs in (1, 16, 48, 64, 2048):
            assert checkpoint_save_freq_vec_calls(None, n_envs) == \
                -(-DEFAULT_CHECKPOINT_EVERY_ENV_STEPS // n_envs)
            assert 0 <= checkpoint_interval_env_steps(None, n_envs) - 2_400_000 < n_envs

    def test_env_steps_are_divided_by_n_envs_because_save_freq_counts_vec_calls(self):
        assert checkpoint_save_freq_vec_calls(150_000, 48) == 3125      # 150000 / 48 = 3125 exactly
        assert checkpoint_save_freq_vec_calls(4_800, 48) == 100
        assert checkpoint_save_freq_vec_calls(1_000, 1) == 1_000

    def test_a_non_divisible_request_rounds_UP_never_down(self):
        """Rounding down would checkpoint MORE often than asked — a silent cost increase — so the
        conversion ceils and the effective interval is reported post-rounding."""
        assert checkpoint_save_freq_vec_calls(100_000, 48) == 2084      # 2083.33 -> 2084
        assert checkpoint_interval_env_steps(100_000, 48) == 2084 * 48  # >= the request
        assert checkpoint_interval_env_steps(100_000, 48) >= 100_000

    def test_a_tiny_request_floors_at_one_vec_call_not_zero(self):
        """`n_calls % save_freq` with save_freq 0 is a ZeroDivisionError inside SB3's callback."""
        assert checkpoint_save_freq_vec_calls(1, 64) == 1
        assert checkpoint_save_freq_vec_calls(0, 64) == 1

    def test_the_default_interval_is_the_starving_one_this_flag_exists_for(self):
        assert checkpoint_interval_env_steps(None, 48) == 2_400_000
        assert checkpoint_interval_env_steps(None, 1) == 2_400_000


class TestDutyCycle:
    def test_the_measured_starvation_reproduces(self):
        duty = cf_label_duty_cycle(150_000, checkpoint_interval_env_steps(None, 48))
        assert duty == pytest.approx(0.0625)
        assert duty < CF_DUTY_CYCLE_FLOOR

    def test_the_fix_clears_the_floor(self):
        duty = cf_label_duty_cycle(150_000, checkpoint_interval_env_steps(150_000, 48))
        assert duty == pytest.approx(1.0)

    def test_never_expire_is_unbounded_not_zero_and_not_a_divide_by_zero(self):
        assert cf_label_duty_cycle(0, 2_400_000) == float("inf")
        assert cf_label_duty_cycle(None, 2_400_000) == float("inf")


# ---------------------------------------------------------------------------
# The constructed callback — default preservation, end to end
# ---------------------------------------------------------------------------

def _checkpoint_callback_save_freq(model_dir, *flags) -> int:
    """`build_callbacks`'s ACTUAL checkpointer, for an argv — not a re-derivation of it.

    `--use-bridge node` only to keep `resolve_config` from resolving (and possibly building) the
    rust `sim_bridge` binary; it has nothing to do with the cadence.
    """
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config

    p = build_parser()
    args = p.parse_args(["--steps", "1", "--use-bridge", "node", "--debug-eval", *flags])
    resolve_config(args, p)
    args.debug_eval = False          # skip the eval callback: this is about the checkpointer
    args.debug = True                # ... which needs _run_eval False; n_envs is read separately
    bundle = build_callbacks(
        args=args, model_dir=str(model_dir), server_config=None, annealing_mode=False,
        _pool=None, _fixed_opponents=None, _bot_weight_vec=None, OPPONENT_CLASSES=(),
        _specialist_team_str=None, _promote_threshold=0.6, _heuristic_floor=0.0,
        _sp_start_wr=0.5, _sp_full_wr=0.9)
    return bundle.callbacks[0].save_freq


class TestDefaultPreservation:
    """⚠️ These call `build_callbacks` with `args.debug` forced True purely to skip the eval
    callback, so `n_envs` is 1 here and the ARITHMETIC is covered by `TestCadenceConversion`; the
    non-debug N=48 / N=2048 build is `main/train/cadence_n_independence_test.py`. What these pin is
    that the constructed callback reads the flag at all."""

    def test_a_flagless_run_builds_the_byte_identical_checkpointer(self, tmp_path):
        """With no `--checkpoint-every-steps` the constructed callback's `save_freq` is the 2.4M-env-
        step default at the callback's N — 1 here (`args.debug` is forced, see the class note)."""
        assert _checkpoint_callback_save_freq(tmp_path, "--n-envs", "48") == 2_400_000

    def test_the_flag_changes_it_and_nothing_else_does(self, tmp_path):
        assert _checkpoint_callback_save_freq(
            tmp_path, "--n-envs", "48", "--checkpoint-every-steps", "500") == 500


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

_CF_ON = ["--use-bridge", "node", "--cf-records", "--win-prob-mode", "read_only",
          "--cf-twin-heads", "--cf-twin-coef", "0.1"]


def _resolve(*flags):
    from main.train.config import resolve_config

    p = build_parser()
    args = p.parse_args(["--steps", "1", *flags])
    resolve_config(args, p)
    return args


class TestDutyCycleGuard:
    def test_a_starved_config_exits_FATAL_CONFIG(self):
        with pytest.raises(SystemExit) as exc:
            _resolve(*_CF_ON, "--n-envs", "48")
        assert exc.value.code == int(TrainExitCode.FATAL_CONFIG), (
            "restarting hits the identical config every time — the launcher must give up, not loop")

    def test_the_refusal_message_carries_the_numbers_and_both_remedies(self, capsys):
        with pytest.raises(SystemExit):
            _resolve(*_CF_ON, "--n-envs", "48")
        cap = capsys.readouterr()
        msg = cap.out + cap.err
        assert "150,000" in msg, "the lag bound"
        assert "2,400,000" in msg, "the checkpoint interval in env steps"
        assert "48" in msg, "the n_envs multiplier that hid it"
        assert "6.2%" in msg or "6.3%" in msg, "the computed duty cycle"
        assert "--checkpoint-every-steps" in msg and "--cf-label-lag-steps" in msg, "both remedies"

    def test_a_healthy_config_PRINTS_the_duty_cycle_rather_than_staying_silent(self, capsys):
        args = _resolve(*_CF_ON, "--n-envs", "48", "--checkpoint-every-steps", "150000")
        msg = capsys.readouterr().out
        assert "DUTY CYCLE" in msg and "100.0%" in msg
        assert args.checkpoint_every_steps == 150_000

    def test_debug_is_exempt_but_still_prints(self, capsys):
        """A smoke's duty cycle is an artifact of the smoke; refusing one would make `--debug`
        unusable for exercising this path at all."""
        _resolve(*_CF_ON, "--debug", "--n-envs", "48")
        msg = capsys.readouterr().out
        assert "DUTY CYCLE" in msg and "the floor is not enforced" in msg

    def test_a_run_with_no_cf_consumer_is_untouched(self, capsys):
        """Off is off: no computation, no line, no refusal — the flagless production case."""
        _resolve("--use-bridge", "node", "--n-envs", "48")
        assert "DUTY CYCLE" not in capsys.readouterr().out

    def test_records_off_is_untouched_even_with_a_live_coefficient(self, capsys):
        """Without `--cf-records` this run rings nothing to label, so the duty cycle has no opinion.
        Since gen3_supply_guard_v1 that combination is only LEGAL with an EXTERNAL supply declared
        (the trainer-spawned producer would have nothing to label — refused, see below)."""
        _resolve("--use-bridge", "node", "--win-prob-mode", "read_only",
                 "--cf-winprob-coef", "0.5", "--n-envs", "48", "--cf-label-supply", "external")
        assert "DUTY CYCLE" not in capsys.readouterr().out

    def test_records_off_with_the_default_supply_is_refused_as_no_supply(self, capsys):
        """`ai_v12_12_ladder_cflabels`' class: a live coefficient whose producer cannot exist."""
        with pytest.raises(SystemExit) as exc:
            _resolve("--use-bridge", "node", "--win-prob-mode", "read_only",
                     "--cf-winprob-coef", "0.5", "--n-envs", "48")
        assert exc.value.code == int(TrainExitCode.FATAL_CONFIG)
        out = capsys.readouterr()
        assert "[SUPPLY] FATAL" in out.out + out.err and "DUTY CYCLE" not in out.out + out.err

    def test_the_winprob_consumer_is_guarded_too(self):
        with pytest.raises(SystemExit) as exc:
            _resolve("--use-bridge", "node", "--cf-records", "--win-prob-mode", "read_only",
                     "--cf-winprob-coef", "0.5", "--n-envs", "48")
        assert exc.value.code == int(TrainExitCode.FATAL_CONFIG)

    def test_a_wider_staleness_bound_is_the_other_remedy(self, capsys):
        """The message offers two fixes; both must actually work, or it is advice, not a remedy."""
        _resolve(*_CF_ON, "--n-envs", "48", "--cf-label-lag-steps", "600000")
        assert "DUTY CYCLE" in capsys.readouterr().out

    def test_a_nonpositive_interval_is_refused_by_the_parser_check(self):
        p = build_parser()
        args = p.parse_args(["--steps", "1", "--checkpoint-every-steps", "0"])
        from main.train.config import resolve_config
        with pytest.raises(SystemExit):
            resolve_config(args, p)
