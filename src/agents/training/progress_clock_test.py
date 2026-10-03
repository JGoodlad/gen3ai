"""``ProgressClock`` — the shipped behaviour, pinned. (Its two OPT-IN intent-restoring variants,
`--progress-decision-tense` and `--progress-switch-freeze`, were DELETED in the flag census, P11d: both OFF
everywhere with 0 recorded runs. What was here for them — the F1 / F2b flip-with-the-flag tests and probe M's
alignment discriminator — went with them; the measurement that motivated them is
`designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md`.)

Two things live here:

1. **The default path is a CHECKSUM.** `SCENARIO` is a scripted 15-window episode and
   `test_default_path_matches_the_recorded_trace` pins the exact `n` sequence it produces. The literal was
   captured by running the SAME scenario against the pre-change clock (`git show <pre>:src/agents/training/
   progress_clock.py`, loaded as a separate module), so it is a genuine A/B against the shipped behaviour
   rather than a re-statement of the current code. Any default-path drift fails here — and deleting the two
   variants moved nothing: the trace below is the one captured before they were deleted.

2. **The two variants are GONE and the shipped readings are pinned by name**: the forced-switch sit-out is
   read off the request that CLOSES the window (`phase_is_forced_switch`, decision ``t+1``), and a voluntary
   switch that fails the progress predicate advances ``n`` like any other no-op.

To re-capture the reference trace after a deliberate default-path change::

    git show HEAD:src/agents/training/progress_clock.py > /tmp/old_clock.py
    # load /tmp/old_clock.py as a module, run `run_scenario(OldClock())`, paste the result

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agents.training.progress_clock import PROGRESS_CLOCK_CAP, ProgressClock
from agents.training.reward_manager import RewardConfig


# --------------------------------------------------------------------------- synthetic fixtures

def live(*, our_spikes=0, opp_spikes=0, opp_status=None, opp_volatiles=(),
         our_boosts=None, our_volatiles=(), our_moves=()):
    """A LiveView-shaped stand-in. The clock reads only these attributes."""
    ours = SimpleNamespace(
        active=SimpleNamespace(boosts=dict(our_boosts or {}), volatiles=set(our_volatiles),
                               move_ids=list(our_moves), species="ourmon", fainted=False),
        side_conditions={"spikes": our_spikes})
    opp = SimpleNamespace(
        active=SimpleNamespace(status=opp_status, volatiles=set(opp_volatiles),
                               species="oppmon", fainted=False),
        side_conditions={"spikes": opp_spikes})
    return SimpleNamespace(ours=ours, opp=opp)


def delta(**kw):
    """A TurnDelta-shaped stand-in, defaulting to "we used a move and nothing happened" — the
    canonical charged NO_OP, so each test names only the one fact it is about."""
    base = dict(
        our_move_id="tackle", our_switch_to=None, our_prev_active="ourmon",
        our_damaging_event=None, opp_target_hp_delta=None, our_move_hit_delta=0.0,
        opp_status_applied=None, opp_switch_to=None,
        our_failed_to_move=False, our_move_outcome="hit",
        our_status_applied=None, our_status_cured=None,
        opp_resolved_move_id=None, opp_fainted=False,
        phase_is_forced_switch=False,
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def legal(switches=(1, 2)):
    return SimpleNamespace(switches=list(switches))


def fold(clock, d, *, lv=None, lg=None):
    """One window, returning `n` — the obs scalar's input (the reward no longer reads the clock)."""
    clock.update(d, lv if lv is not None else live(), lg if lg is not None else legal())
    return clock.n


# --------------------------------------------------------------------------- 1. the default checksum

# A scripted episode exercising every classification branch the clock has: a plain no-op, a
# damaging move (PROGRESS), a miss (exogenous freeze), two in-grace heals then a third
# (heal-war), a capped-Spikes short-circuit, a filler Rapid Spin, a trapped window, a voluntary
# switch, a forced-switch window in each tense, and a boost (setup PROGRESS).
_HEAL = np.zeros(6, dtype=np.float32)
_HEAL[0] = 0.4
_HIT = SimpleNamespace(move_id="tackle", target_species="oppmon")

SCENARIO = [
    ("plain no-op", dict(), {}, {}),
    ("damaging move", dict(our_damaging_event=_HIT, opp_target_hp_delta=-0.3,
                           our_move_hit_delta=-0.3), {}, {}),
    ("accuracy miss", dict(our_move_outcome="miss"), {}, {}),
    ("heal 1 (in grace)", dict(our_move_id="recover", our_hp_delta=_HEAL), {}, {}),
    ("heal 2 (in grace)", dict(our_move_id="recover", our_hp_delta=_HEAL), {}, {}),
    ("heal 3 (heal-war)", dict(our_move_id="recover", our_hp_delta=_HEAL), {}, {}),
    ("spikes lays a layer", dict(our_move_id="spikes"), dict(opp_spikes=3), {}),
    ("capped spikes", dict(our_move_id="spikes"), dict(opp_spikes=3), {}),
    ("filler rapid spin", dict(our_move_id="rapidspin"), {}, {}),
    ("trapped no-op", dict(), {}, dict(switches=())),
    ("voluntary switch", dict(our_move_id=None, our_switch_to="benchmon"), {}, {}),
    ("closing-forced window", dict(phase_is_forced_switch=True), {}, {}),
    ("no-op after a replacement (opening-tense read is gone)", dict(), {}, {}),
    ("boost (setup)", dict(our_move_id="calmmind"), dict(our_boosts={"spa": 1, "spd": 1}), {}),
    ("plain no-op again", dict(), {}, {}),
]

# CAPTURED from the pre-fix clock (`git show 4787d1e:src/agents/training/progress_clock.py`,
# loaded as a standalone module and driven by `run_scenario`). This is the byte-identity claim for
# the default path: the clock produces the sequence it always produced.
REFERENCE_TRACE = [
    1,  # plain no-op — charged
    0,  # damaging move — PROGRESS, reset
    0,  # miss — exogenous denial, frozen
    0,  # heal 1 — in grace
    0,  # heal 2 — in grace
    1,  # heal 3 — heal-war, charged
    0,  # Spikes adds the 3rd layer — hazard PROGRESS, reset
    1,  # capped Spikes — short-circuit charge
    2,  # filler Rapid Spin — no progress reset, charged
    3,  # trapped — increments, charge suppressed
    4,  # voluntary switch — charged NO_OP
    4,  # window CLOSING on a forced switch — sit-out
    5,  # the window after a replacement — charged
    0,  # boost — setup PROGRESS, reset
    1,  # plain no-op — charged
]


def run_scenario(clock):
    """Drive `SCENARIO` through `clock` and return the observable trace. Kept importable so the
    reference can be re-captured against an older implementation with the same driver."""
    out = []
    for _name, dkw, lkw, gkw in SCENARIO:
        out.append(fold(clock, delta(**dkw), lv=live(**lkw), lg=legal(**gkw)))
    return out


def test_default_path_matches_the_recorded_trace():
    assert run_scenario(ProgressClock()) == REFERENCE_TRACE


def test_the_scenario_actually_exercises_every_outcome():
    """A checksum over a scenario that only ever charges would pass while proving nothing."""
    ns = list(REFERENCE_TRACE)
    assert 0 in ns and max(ns) > 1                      # resets AND accumulation present
    assert len(SCENARIO) == len(REFERENCE_TRACE) == 15


def test_the_two_deleted_variants_are_gone():
    """No constructor kwarg, no attribute, no `apply_reward_config` seam: the shipped reading is the only one."""
    c = ProgressClock()
    assert not hasattr(c, "decision_tense") and not hasattr(c, "switch_freeze")
    assert not hasattr(c, "apply_reward_config")
    with pytest.raises(TypeError):
        ProgressClock(decision_tense=True)      # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ProgressClock(switch_freeze=True)       # type: ignore[call-arg]


# --------------------------------------------------------------------------- 2. the shipped readings

def test_the_forced_window_is_read_off_the_CLOSING_request():
    """A window that CLOSES on a forced switch sits out; one that merely OPENED on a forced switch (a
    post-faint replacement was the decision) is charged like any other no-op — the closing tense is the
    only one the clock has."""
    assert fold(ProgressClock(), delta(phase_is_forced_switch=True)) == 0
    assert fold(ProgressClock(), delta(our_move_id=None, our_switch_to="benchmon")) == 1


def test_a_voluntary_no_progress_switch_advances_the_clock_like_any_other_no_op():
    """No freeze: an accumulated clock keeps counting through a pivot."""
    c = ProgressClock()
    assert fold(c, delta()) == 1
    assert fold(c, delta()) == 2
    assert fold(c, delta(our_move_id=None, our_switch_to="benchmon")) == 3
    assert fold(c, delta()) == 4


def test_a_switch_that_IS_progress_still_resets_the_clock():
    """Clauses ii/iv/v (the opponent also committed, a residual is ticking) reset on a voluntary switch."""
    c = ProgressClock()
    assert fold(c, delta()) == 1
    assert fold(c, delta(our_move_id=None, our_switch_to="benchmon", opp_switch_to="theirmon")) == 0


def test_the_cap_still_bounds_the_counter():
    c = ProgressClock()
    for _ in range(PROGRESS_CLOCK_CAP + 5):
        fold(c, delta())
    assert c.n == PROGRESS_CLOCK_CAP


@pytest.mark.parametrize("field", ["progress_decision_tense", "progress_switch_freeze"])
def test_the_reward_config_fields_default_off(field):
    """The two RECORDED run fields stay (resume-immutable, value-checked); a recorded True is refused on a
    resume (`reward_defaults_test`)."""
    assert getattr(RewardConfig(), field) is False
