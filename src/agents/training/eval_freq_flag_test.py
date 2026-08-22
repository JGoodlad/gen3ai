"""gen3_eval_freq_flag_v1 — the eval cadence is a KNOB, and its default is byte-identical.

Why this exists: EVAL_FREQ_STEPS was a module constant, so a SHORT arm could not be given a
tighter cadence. A 3M exploiter-gate fork at the 2M default yields 1-2 cycles inside its budget,
which cannot satisfy a >=4-cycle reading discipline — the arm silently under-produces the very
metric its verdict keys on. Both callbacks must honour it, or their cadences drift apart and
cross-arm eval comparisons stop meaning anything.
"""
import pytest

from agents.training.eval_callback import EVAL_FREQ_STEPS, EVAL_GAMES


def _mk(cls, **kw):
    obj = object.__new__(cls)
    obj._eval_games = int(kw["eval_games"]) if kw.get("eval_games") else EVAL_GAMES
    obj._eval_freq = int(kw["eval_freq"]) if kw.get("eval_freq") else EVAL_FREQ_STEPS
    obj._debug = False
    return obj


def _classes():
    from agents.training.eval_callback import PerOpponentEvalCallback
    from agents.training.selfplay_callback import SelfPlayCallback
    return [PerOpponentEvalCallback, SelfPlayCallback]


@pytest.mark.parametrize("cls", _classes())
def test_default_is_byte_identical(cls):
    """None => the historical constant, so every pre-existing command is unchanged."""
    freq, games = _mk(cls)._schedule()
    assert freq == EVAL_FREQ_STEPS == 2_000_000
    assert games == EVAL_GAMES


@pytest.mark.parametrize("cls", _classes())
def test_explicit_value_is_honoured(cls):
    freq, _ = _mk(cls, eval_freq=1_500_000)._schedule()
    assert freq == 1_500_000


@pytest.mark.parametrize("cls", _classes())
def test_zero_and_none_both_fall_back(cls):
    """0 is falsey and must NOT mean 'eval every step' — it means 'unset'."""
    assert _mk(cls, eval_freq=0)._schedule()[0] == EVAL_FREQ_STEPS
    assert _mk(cls, eval_freq=None)._schedule()[0] == EVAL_FREQ_STEPS


def test_both_callbacks_agree_so_cadences_cannot_drift():
    a, b = _classes()
    assert _mk(a, eval_freq=750_000)._schedule()[0] == _mk(b, eval_freq=750_000)._schedule()[0]


def test_a_short_arm_gets_the_registered_cycle_count():
    """The motivating case, pinned as arithmetic: a 3M fork needs <=750k to clear 4 cycles."""
    base, end = 25_067_520, 28_067_520
    cycles = lambda f: len([b for b in range(0, end + f, f) if base < b <= end])
    assert cycles(EVAL_FREQ_STEPS) == 2      # the default under-produces
    assert cycles(1_500_000) == 2
    assert cycles(750_000) >= 4              # what the >=4 discipline needs


def test_the_flag_exists_and_defaults_to_none():
    from main.train_rl_agent import build_parser
    a = build_parser().parse_args(["--steps", "1"])
    assert a.eval_freq is None
    assert build_parser().parse_args(["--steps", "1", "--eval-freq", "1500000"]).eval_freq == 1_500_000
