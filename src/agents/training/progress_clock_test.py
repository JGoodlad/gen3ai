"""``ProgressClock`` — the DEFAULT behaviour, pinned; and the two OPT-IN intent-restoring fixes.

Three things live here, and they answer different questions:

1. **The default path is a CHECKSUM.** `SCENARIO` is a scripted 14-window episode and
   `test_default_path_matches_the_recorded_trace` pins the exact `(n, last_penalty)` sequence it
   produces. The literal was captured by running the SAME scenario against the pre-change clock
   (`git show <pre>:src/agents/training/progress_clock.py`, loaded as a separate module — see
   `_capture_reference_trace` in the docstring below), so it is a genuine A/B against the shipped
   behaviour rather than a re-statement of the current code. Any default-path drift fails here.

2. **Each fix flips exactly with its flag.** Every fix assertion runs the SAME synthetic window
   through a flag-OFF clock and a flag-ON clock and asserts they differ in the stated way — which
   is revert-verification stated as a test rather than as a procedure. A revert of the fix makes
   the ON arm equal the OFF arm and the assertion fails.

3. **The F1 window alignment, as probe M measured it.** `bias_tax_head_alignment_2026-08-29.md`
   §1 discriminated the two candidate fold→window alignments with one statistic — "windows where
   the clock nonetheless moved, among those the candidate says it should sit out": **0 / 10,442**
   for the `t+1` (closing) reading against **8,710 / 10,424** for the `t` (opening) one, which is
   how it concluded the shipped alignment is `t+1`. `test_the_alignment_discriminator_*` runs that
   exact statistic over a synthetic decision sequence and asserts the same shape in BOTH flag
   states: the clock's sit-out tracks the CLOSING request by default and the OPENING one under
   `--progress-decision-tense`. That is the fix restated as the measurement that found it.

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

_PENALTY = 0.15


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
        our_damaging_event=None, opp_target_hp_delta=None,
        opp_status_applied=None, opp_switch_to=None,
        our_failed_to_move=False, our_move_outcome="hit",
        our_status_applied=None, our_status_cured=None,
        opp_resolved_move_id=None, opp_fainted=False,
        phase_is_forced_switch=False, decision_was_forced_switch=False,
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def legal(switches=(1, 2)):
    return SimpleNamespace(switches=list(switches))


def fold(clock, d, *, lv=None, lg=None, lg_prev=None):
    """One window, returning the observable pair the obs scalar and the reward read."""
    clock.update(d, lv if lv is not None else live(), lg if lg is not None else legal(),
                 legal_prev=lg_prev)
    return clock.n, round(clock.last_penalty, 6)


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
    ("damaging move", dict(our_damaging_event=_HIT, opp_target_hp_delta=-0.3), {}, {}),
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
    ("opening-forced window", dict(decision_was_forced_switch=True), {}, {}),
    ("boost (setup)", dict(our_move_id="calmmind"), dict(our_boosts={"spa": 1, "spd": 1}), {}),
    ("plain no-op again", dict(), {}, {}),
]

# CAPTURED from the pre-fix clock (`git show 4787d1e:src/agents/training/progress_clock.py`,
# loaded as a standalone module and driven by `run_scenario`). This is the byte-identity claim for
# the default path: with both flags OFF the clock produces the sequence it always produced.
REFERENCE_TRACE = [
    (1, -0.15),   # plain no-op — charged
    (0, 0.0),     # damaging move — PROGRESS, reset
    (0, 0.0),     # miss — exogenous denial, frozen
    (0, 0.0),     # heal 1 — in grace
    (0, 0.0),     # heal 2 — in grace
    (1, -0.15),   # heal 3 — heal-war, charged
    (0, 0.0),     # Spikes adds the 3rd layer — hazard PROGRESS, reset
    (1, -0.15),   # capped Spikes — short-circuit charge
    (2, -0.15),   # filler Rapid Spin — no progress reset, charged
    (3, 0.0),     # trapped — increments, charge suppressed
    (4, -0.15),   # voluntary switch — charged NO_OP
    (4, 0.0),     # window CLOSING on a forced switch — sit-out
    (5, -0.15),   # window OPENING on a forced switch — charged (the off-by-one)
    (0, 0.0),     # boost — setup PROGRESS, reset
    (1, -0.15),   # plain no-op — charged
]


def run_scenario(clock):
    """Drive `SCENARIO` through `clock` and return the observable trace. Kept importable so the
    reference can be re-captured against an older implementation with the same driver."""
    out = []
    for _name, dkw, lkw, gkw in SCENARIO:
        out.append(fold(clock, delta(**dkw), lv=live(**lkw), lg=legal(**gkw)))
    return out


def test_default_path_matches_the_recorded_trace():
    assert run_scenario(ProgressClock(_PENALTY)) == REFERENCE_TRACE


def test_the_scenario_actually_exercises_every_outcome():
    """A checksum over a scenario that only ever charges would pass while proving nothing."""
    ns = [n for n, _ in REFERENCE_TRACE]
    penalties = {p for _, p in REFERENCE_TRACE}
    assert penalties == {0.0, -0.15}                    # charged AND uncharged windows present
    assert 0 in ns and max(ns) > 1                      # resets AND accumulation present
    assert len(SCENARIO) == len(REFERENCE_TRACE) == 15


def test_both_fixes_default_off():
    c = ProgressClock()
    assert c.decision_tense is False and c.switch_freeze is False


def test_a_default_reward_config_leaves_the_clock_alone():
    c = ProgressClock(_PENALTY)
    c.apply_reward_config(RewardConfig())
    assert (c.decision_tense, c.switch_freeze) == (False, False)
    assert run_scenario(c) == REFERENCE_TRACE


# --------------------------------------------------------------------------- 2. F1, the tense fix

def _both(dkw, **fold_kw):
    """The same window folded by an OFF clock and an ON clock. The ONLY difference is the flag, so
    a revert of the fix collapses the two and every caller's assertion fails."""
    off = ProgressClock(_PENALTY)
    on = ProgressClock(_PENALTY, decision_tense=True)
    return fold(off, delta(**dkw), **fold_kw), fold(on, delta(**dkw), **fold_kw)


def test_a_window_opened_by_a_forced_switch_is_charged_off_and_sits_out_on():
    """The zero-agency post-faint replacement: probe M measures it charged 63.9% of the time,
    36.3% of all charges. No action available to it can satisfy the progress predicate."""
    off, on = _both(dict(our_move_id=None, our_switch_to="benchmon",
                         decision_was_forced_switch=True, phase_is_forced_switch=False))
    assert off == (1, -0.15)
    assert on == (0, 0.0)


def test_a_window_closed_by_a_forced_switch_sits_out_off_and_is_charged_on():
    """The mirror half — probe M's SITOUT class: 19,503 FULL-agency decisions exempted because our
    mon happened to be KO'd on them. The costliest class in its corpus at −5.1pp."""
    off, on = _both(dict(phase_is_forced_switch=True, decision_was_forced_switch=False))
    assert off == (0, 0.0)
    assert on == (1, -0.15)


def test_the_two_flags_of_the_same_window_are_independent_facts():
    """A window can be both (KO'd on a replacement turn) or neither; the fix is a choice of WHICH
    one the clock reads, not a redefinition of either."""
    both_true, _ = _both(dict(phase_is_forced_switch=True, decision_was_forced_switch=True))
    assert both_true == (0, 0.0)
    on = ProgressClock(_PENALTY, decision_tense=True)
    assert fold(on, delta(phase_is_forced_switch=True, decision_was_forced_switch=True)) == (0, 0.0)


def test_the_trapped_gate_moves_with_the_tense_third_defect():
    """Probe N §3.3 — the SAME call passes the SAME upcoming-request legality to the helplessness
    gate, so a mon genuinely trapped at t is charged whenever its successor could switch. It is one
    off-by-one in one call; a fix that moved only the phase would be half-done."""
    trapped_then, free_now = legal(switches=()), legal(switches=(1,))
    off = ProgressClock(_PENALTY)
    on = ProgressClock(_PENALTY, decision_tense=True)
    assert fold(off, delta(), lg=free_now, lg_prev=trapped_then) == (1, -0.15)
    assert fold(on, delta(), lg=free_now, lg_prev=trapped_then) == (1, 0.0)


def test_the_trapped_gate_mirror_a_free_choice_is_not_exempted():
    free_then, trapped_now = legal(switches=(1,)), legal(switches=())
    off = ProgressClock(_PENALTY)
    on = ProgressClock(_PENALTY, decision_tense=True)
    assert fold(off, delta(), lg=trapped_now, lg_prev=free_then) == (1, 0.0)
    assert fold(on, delta(), lg=trapped_now, lg_prev=free_then) == (1, -0.15)


def test_an_absent_legal_prev_degrades_to_the_pre_fix_reading_not_to_trapped():
    """`RewardTrackingMixin` builds contexts without a legality snapshot. Reading `None` as
    "trapped" would silently zero every charge on that path — a fix that turns a term off is worse
    than the off-by-one it replaces."""
    on = ProgressClock(_PENALTY, decision_tense=True)
    assert fold(on, delta(), lg=legal(switches=(1,)), lg_prev=None) == (1, -0.15)
    on2 = ProgressClock(_PENALTY, decision_tense=True)
    assert fold(on2, delta(), lg=legal(switches=()), lg_prev=None) == (1, 0.0)


def test_legal_prev_is_ignored_entirely_when_the_flag_is_off():
    """The threading is unconditional; only its CONSUMPTION is gated. Everything the env now passes
    must be inert at the default, or landing this would move a live run."""
    off = ProgressClock(_PENALTY)
    assert fold(off, delta(), lg=legal(switches=(1,)), lg_prev=legal(switches=())) == (1, -0.15)


# ------------------------------------------------- probe M's alignment discriminator, as a test

# A synthetic decision sequence: `True` = that decision is a forced switch (post-faint
# replacement). Window k spans decision k → k+1, so the fold for window k carries
# `decision_was_forced_switch = FORCED[k]` and `phase_is_forced_switch = FORCED[k+1]`.
FORCED = [False, False, True, False, False, True, False, True, False, False]


def _windows():
    return [(FORCED[k], FORCED[k + 1]) for k in range(len(FORCED) - 1)]


def _sitout_violations(clock_kwargs, *, candidate: str):
    """Probe M's statistic: among windows the CANDIDATE alignment says the clock should sit out,
    how many did the clock nonetheless move on? 0 means the clock's real behaviour IS that
    alignment; a large fraction means it is the other one."""
    n_candidate = moved = 0
    for opening, closing in _windows():
        clock = ProgressClock(_PENALTY, **clock_kwargs)
        before = clock.n
        clock.update(delta(decision_was_forced_switch=opening, phase_is_forced_switch=closing),
                     live(), legal(), legal_prev=legal())
        says_sitout = opening if candidate == "opening" else closing
        if says_sitout:
            n_candidate += 1
            if clock.n != before or clock.last_penalty != 0.0:
                moved += 1
    return n_candidate, moved


def test_the_alignment_discriminator_default_clock_tracks_the_CLOSING_request():
    """Reproduces the shape of probe M's table: 0 violations under the alignment the clock really
    uses, a majority under the other. Default ⇒ the closing request (decision t+1)."""
    n_closing, moved_closing = _sitout_violations({}, candidate="closing")
    n_opening, moved_opening = _sitout_violations({}, candidate="opening")
    assert n_closing == 3 and moved_closing == 0
    assert n_opening == 3 and moved_opening == 3


def test_the_alignment_discriminator_the_fix_tracks_the_OPENING_decision():
    """…and the flag moves the 0 to the other column, which is the whole content of F1."""
    on = {"decision_tense": True}
    n_opening, moved_opening = _sitout_violations(on, candidate="opening")
    n_closing, moved_closing = _sitout_violations(on, candidate="closing")
    assert n_opening == 3 and moved_opening == 0
    assert n_closing == 3 and moved_closing == 3


# --------------------------------------------------------------------------- 3. F2b, switch freeze

def _both_freeze(dkw, **fold_kw):
    off = ProgressClock(_PENALTY)
    on = ProgressClock(_PENALTY, switch_freeze=True)
    return fold(off, delta(**dkw), **fold_kw), fold(on, delta(**dkw), **fold_kw)


def test_a_voluntary_no_progress_switch_is_charged_off_and_frozen_on():
    """42.7% of all charges. `_is_progress` is offense-only — none of its eight clauses can be
    satisfied BY a switch — so the tax prices the action KIND, not the choice within it."""
    off, on = _both_freeze(dict(our_move_id=None, our_switch_to="benchmon"))
    assert off == (1, -0.15)
    assert on == (0, 0.0)


def test_a_frozen_switch_freezes_rather_than_resets():
    """FREEZE, not PROGRESS: an accumulated clock must survive a pivot, or a switch would launder
    a stall into a clean slate — the `switch_bouncing_tax` failure mode in a new spelling."""
    on = ProgressClock(_PENALTY, switch_freeze=True)
    assert fold(on, delta()) == (1, -0.15)
    assert fold(on, delta()) == (2, -0.15)
    assert fold(on, delta(our_move_id=None, our_switch_to="benchmon")) == (2, 0.0)
    assert fold(on, delta()) == (3, -0.15)


def test_a_switch_that_IS_progress_still_resets_the_clock_under_the_flag():
    """Probe M measures 27% of voluntary switches escaping via clauses ii/iv/v (the opponent also
    committed, a residual is ticking). Those are RESETS today and must stay resets — the freeze is
    placed after the classification precisely so it only replaces the NO_OP outcome."""
    on = ProgressClock(_PENALTY, switch_freeze=True)
    assert fold(on, delta()) == (1, -0.15)
    assert fold(on, delta(our_move_id=None, our_switch_to="benchmon",
                          opp_switch_to="theirmon")) == (0, 0.0)


def test_a_move_no_op_is_still_charged_under_the_flag():
    """The anti-stall job is not removed, only re-aimed: a pivot-loop still pays on every move turn
    between the pivots. If this ever passes for moves too, the term has been deleted by accident."""
    on = ProgressClock(_PENALTY, switch_freeze=True)
    assert fold(on, delta()) == (1, -0.15)


def test_the_freeze_does_not_rescue_a_forced_replacement_on_its_own():
    """F1 and F2b are independent fixes to independent halves. A post-faint replacement is a switch
    too, so F2b alone would freeze it — but only because it is a switch, not because the window had
    no agency; that is F1's job, and the default clock still charges it here."""
    off, on = _both_freeze(dict(our_move_id=None, our_switch_to="benchmon",
                                decision_was_forced_switch=True))
    assert off == (1, -0.15)
    assert on == (0, 0.0)   # frozen for the WRONG reason — F1 is what makes it a sit-out


def test_both_fixes_together_compose():
    both = ProgressClock(_PENALTY, decision_tense=True, switch_freeze=True)
    assert fold(both, delta()) == (1, -0.15)                                    # move no-op: charged
    assert fold(both, delta(our_move_id=None, our_switch_to="b")) == (1, 0.0)   # pivot: frozen
    assert fold(both, delta(decision_was_forced_switch=True)) == (1, 0.0)       # replacement: sit-out
    assert fold(both, delta(phase_is_forced_switch=True)) == (2, -0.15)         # KO turn: now charged


# --------------------------------------------------------------------------- 4. the config seam

def test_apply_reward_config_threads_all_three_knobs():
    """One call, because hand-threading a reward field is exactly how the eval path once measured a
    different reward than training (RewardConfig's own note)."""
    c = ProgressClock()
    c.apply_reward_config(RewardConfig(no_progress_penalty=0.25,
                                       progress_decision_tense=True,
                                       progress_switch_freeze=True))
    assert (c.no_progress_penalty, c.decision_tense, c.switch_freeze) == (0.25, True, True)


def test_apply_reward_config_none_is_a_no_op():
    c = ProgressClock(0.4, decision_tense=True)
    c.apply_reward_config(None)
    assert (c.no_progress_penalty, c.decision_tense) == (0.4, True)


def test_the_cap_still_bounds_the_counter_under_both_fixes():
    c = ProgressClock(_PENALTY, decision_tense=True, switch_freeze=True)
    for _ in range(PROGRESS_CLOCK_CAP + 5):
        fold(c, delta())
    assert c.n == PROGRESS_CLOCK_CAP


@pytest.mark.parametrize("field", ["progress_decision_tense", "progress_switch_freeze"])
def test_the_reward_config_fields_default_off(field):
    assert getattr(RewardConfig(), field) is False
