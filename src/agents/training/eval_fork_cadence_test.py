"""A FORK must not inherit an eval-cadence anchor from the FUTURE of its source run.

Regression guard for the fork-starves-eval defect (found 2026-08-21 by an exploiter-gate smoke):
``resume_eval_metadata`` points at the SOURCE run's run-level ``metadata.json``, whose
``latest_eval.step`` is where THAT run last evaluated — not the step of the older checkpoint being
forked from. Forking gen-17's 9,084,672-step checkpoint out of a run that went on to 25M restored
``_last_eval_step = 24,000,000`` against ``num_timesteps = 9,084,672``, and the cadence test
``(now // freq) > (anchor // freq)`` then stays false until the fork itself reaches 26M — so a
2-3M-step fork launches ZERO eval cycles and produces no ``win_rate_vs_*``, no
``eval_results.jsonl`` row and no ELO. Every exploiter-gate readout keys on an eval metric, so the
gate silently measures nothing.

Both eval callbacks inherit the clamp from the shared ``_ForcedEvalMixin`` — pinned here for BOTH,
because the two classes share only that mixin and the defect was identical in each.
"""
import json

import pytest

from agents.training.eval_callback import PerOpponentEvalCallback, EVAL_FREQ_STEPS
from agents.training.selfplay_callback import SelfPlayCallback


def _meta(tmp_path, step, name="source_metadata.json"):
    """A source run's run-level metadata.json recording its last eval at ``step``."""
    p = tmp_path / name
    p.write_text(json.dumps({"latest_eval": {"step": step, "win_rate_mean": 0.9}}))
    return str(p)


class _FakeModel:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps


class _Probe:
    """Only the cadence-anchor restore is under test — bypass __init__'s heavy wiring.

    ``model`` carries the step exactly as SB3 does; the callback's own ``num_timesteps`` mirror is
    left at BaseCallback's 0, which is its real value at ``_init_callback`` time.
    """

    def __init__(self, resume_meta, now, model_dir=None):
        self._model_dir = model_dir
        self._resume_eval_metadata = resume_meta
        self.model = _FakeModel(now)
        self.num_timesteps = 0
        self._last_eval_step = 0


class _BotProbe(_Probe, PerOpponentEvalCallback):
    pass


class _SelfPlayProbe(_Probe, SelfPlayCallback):
    pass


PROBES = pytest.mark.parametrize("probe_cls", [_BotProbe, _SelfPlayProbe],
                                 ids=["per_opponent", "self_play"])


@PROBES
def test_fork_anchor_is_clamped_to_the_current_step(tmp_path, capsys, probe_cls):
    # the literal defect: fork the 9,084,672 checkpoint of a run that evaluated last at 24M
    p = probe_cls(_meta(tmp_path, 24_000_000), now=9_084_672)
    p._restore_last_eval_step()
    assert p._last_eval_step == 9_084_672, (
        "an anchor from the source run's FUTURE starves this fork's eval entirely")
    out = capsys.readouterr().out
    assert "anchor is AHEAD" in out
    assert "24,000,000" in out and "9,084,672" in out, "the warning must name both steps"
    assert "NO eval cycle" in out, "the warning must name the consequence, not just the numbers"


@PROBES
def test_clamped_fork_evaluates_at_the_next_boundary(tmp_path, probe_cls):
    """The point of the clamp: the cadence test fires again within one EVAL_FREQ_STEPS."""
    p = probe_cls(_meta(tmp_path, 24_000_000), now=9_084_672)
    p._restore_last_eval_step()
    fired = lambda now: (now // EVAL_FREQ_STEPS) > (p._last_eval_step // EVAL_FREQ_STEPS)  # noqa: E731
    assert not fired(9_084_672), "must not re-eval the forked checkpoint on the first step"
    assert fired(9_084_672 + EVAL_FREQ_STEPS), "must eval within one cadence window of the fork"


@PROBES
def test_restart_anchor_is_untouched(tmp_path, capsys, probe_cls):
    """A launcher RESTART records at-or-behind the loaded step — the clamp must never bite."""
    p = probe_cls(_meta(tmp_path, 24_000_000), now=24_050_000)
    p._restore_last_eval_step()
    assert p._last_eval_step == 24_000_000
    assert capsys.readouterr().out == "", "a restart must not print a fork warning"


@PROBES
def test_no_recorded_eval_anchors_at_zero(tmp_path, probe_cls):
    p = probe_cls(None, now=9_084_672)
    p._restore_last_eval_step()
    assert p._last_eval_step == 0, "a fresh run must stay eligible for its first-boundary eval"


@PROBES
def test_the_step_is_read_from_the_model_not_the_callback_mirror(tmp_path, probe_cls):
    """``BaseCallback.num_timesteps`` is 0 until the first ``_on_step``.

    Reading it instead of ``model.num_timesteps`` clamps EVERY resume to 0 — which fires an eval
    on the first step of every launcher restart, the exact behaviour the anchor exists to prevent.
    Observed live before the fix: a fork at 9,084,672 logged "this model is at 0".
    """
    p = probe_cls(_meta(tmp_path, 24_000_000), now=24_050_000)
    assert p.num_timesteps == 0 and p.model.num_timesteps == 24_050_000
    p._restore_last_eval_step()
    assert p._last_eval_step == 24_000_000, "the mirror was read instead of the model"


@PROBES
def test_this_runs_own_metadata_also_clamps(tmp_path, probe_cls):
    """The anchor merges THIS run's metadata too (``model_dir``) — clamp covers that source."""
    (tmp_path / "metadata.json").write_text(
        json.dumps({"latest_eval": {"step": 30_000_000, "win_rate_mean": 0.5}}))
    p = probe_cls(None, now=9_084_672, model_dir=str(tmp_path))
    p._restore_last_eval_step()
    assert p._last_eval_step == 9_084_672
