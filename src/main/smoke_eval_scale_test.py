"""A SMOKE's final eval must be a formality, and must SAY so (`gen3_smoke_eval_scale_v1`).

The problem this fixes, which cost six timeouts and an agent investigation before being understood:
every non-debug run ends with a final evaluation of 9 opponents x `--eval-battles` games — ~900
battles at the old default of 100 — and it runs AFTER the line `Training complete. Model saved to
...`. On a loaded box that is many minutes of silence following a message that reads as terminal.
A working process and a wedged one were indistinguishable from the log.

Two halves, and the second matters as much as the first: scale the work down for short runs, and
state the remaining work out loud so nobody mistakes it for a hang again.
"""
import pytest

from main.train_rl_agent import (
    DEFAULT_EVAL_BATTLES,
    SMOKE_EVAL_BATTLES,
    SMOKE_STEPS,
)


def _resolve(steps, explicit=None):
    """Mirror of the resolution in `main()` — kept in lockstep by the tests below."""
    if explicit is not None:
        return explicit
    return SMOKE_EVAL_BATTLES if steps < SMOKE_STEPS else DEFAULT_EVAL_BATTLES


def test_a_short_run_gets_the_smoke_count():
    assert _resolve(2_000) == SMOKE_EVAL_BATTLES
    assert _resolve(50_000) == SMOKE_EVAL_BATTLES


def test_a_real_run_is_untouched():
    """THE safety property: this must never quietly weaken a production evaluation."""
    assert _resolve(15_000_000) == DEFAULT_EVAL_BATTLES
    assert _resolve(SMOKE_STEPS) == DEFAULT_EVAL_BATTLES, "the threshold is exclusive"


def test_an_explicit_value_always_wins():
    """Including an explicit value that happens to equal a default — the caller's word is final."""
    assert _resolve(2_000, explicit=100) == 100
    assert _resolve(15_000_000, explicit=5) == 5
    assert _resolve(2_000, explicit=SMOKE_EVAL_BATTLES) == SMOKE_EVAL_BATTLES


def test_the_smoke_count_is_small_enough_to_be_fast_and_honest():
    """5 battles/opponent is ~45 total. Small enough to be quick; far too small to be DATA, which
    is why the banner says so rather than letting a reader treat it as a win rate."""
    assert SMOKE_EVAL_BATTLES <= 10
    assert SMOKE_EVAL_BATTLES * 9 < DEFAULT_EVAL_BATTLES, "must be a large saving, not a trim"


def test_the_flag_default_is_None_so_explicit_and_absent_are_distinguishable():
    """If the default were 100 there would be no way to tell "you asked for 100" from "you said
    nothing", and the auto-scale could not exist without overriding a real choice."""
    from main.train import entry_source
    src = entry_source()
    assert '"--eval-battles", type=int, default=None' in src, (
        "--eval-battles must default to None; a concrete default makes 'explicit' unknowable")


@pytest.mark.parametrize("phrase", [
    "Training IS finished and the",                 # the anti-hang statement (the literal is
                                                    # split across two f-strings, so match the first half)
    "opponents x",                                   # the QUANTITY of remaining work
    "NOT a measurement",                             # 5-battle rates are not data
])
def test_the_banner_says_the_things_that_prevent_the_misread(phrase):
    """A reader seeing 'Training complete' then silence concluded 'hang' six times. These three
    statements are what make that misread impossible; losing any one of them re-opens it."""
    from main.train import entry_source
    assert phrase in entry_source(), f"the banner no longer says: {phrase!r}"
