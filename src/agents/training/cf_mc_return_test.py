"""The SHADOW critic's label arithmetic (`cf_mc_return.py`, gen3_cf_twin_heads_v1).

Two things are pinned, and both are silent-wrong-answer classes rather than crashes:

1. **The DISCOUNT is applied oldest-first.** A newest-first fold produces perfectly plausible
   magnitudes and a systematically wrong target, which no downstream meter can detect.
2. **The ARMING is off by exactly one decision, on purpose.** `RewardTracker` is deferred — the
   reward for decision k is only computable once decision k+1's board exists — so arming on entry
   to `note` would count the reward of the decision BEFORE the divergence. That is a whole extra
   turn of shaping folded into every label, VARYING WITH THE DIVERGENCE TURN, i.e. a bias shaped
   exactly like a real signal.

The tracker itself is exercised end to end by the producer's bridge-backed integration test; here
the `RewardTracker` is a stub, because what is under test is the bookkeeping around it.
"""
from __future__ import annotations

import pytest

from agents.training.cf_mc_return import CfReturnRecorder, discounted_return


def test_discounted_return_is_oldest_first():
    """`Σ γᵏ·r[k]` with r[0] the EARLIEST reward. Hand-computed, so a reversed fold fails."""
    assert discounted_return([1.0, 2.0, 3.0], 0.5) == pytest.approx(1.0 + 0.5 * 2 + 0.25 * 3)
    # The asymmetry is the point: reversing the sequence must give a DIFFERENT number.
    assert discounted_return([3.0, 2.0, 1.0], 0.5) != pytest.approx(
        discounted_return([1.0, 2.0, 3.0], 0.5))


def test_discounted_return_at_gamma_one_is_the_plain_sum():
    assert discounted_return([1.0, -2.0, 0.5], 1.0) == pytest.approx(-0.5)


def test_an_empty_sequence_is_zero_but_the_recorder_reports_NONE():
    """`discounted_return([])` is 0 by definition; the RECORDER must not report that as a
    measurement. A zero return is the middle of this reward's range, so "no rollout data" and
    "a neutral game" would be indistinguishable — and the shadow head would be trained toward a
    fabricated target on exactly the rollouts that failed."""
    assert discounted_return([], 0.99) == 0.0


class _StubTracker:
    """Stands in for `RewardTracker`: emits a scripted reward per settled decision.

    ⚠️ The two arities below are DELIBERATE and must match the real tracker's: `complete_pending`
    returns `(delta, reward)` and `finalize` returns `(terminal_ctx, delta, reward)`. A stub that
    made them uniform is exactly how the first version of this module shipped a
    `ValueError: not enough values to unpack` that only the bridge-backed composition test caught.
    """

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self._i = 0
        self.has_pending = False
        self._our_slots = self._opp_slots = None

    def complete_pending(self, _ctx, _battle):
        r = self._rewards[self._i]
        self._i += 1
        self.has_pending = False
        return None, r                     # (delta, reward) — TWO values, like the real one

    def begin_turn(self, _ctx, _action, _cursor=0):
        self.has_pending = True

    def finalize(self, _battle):
        r = self._rewards[self._i]
        self._i += 1
        self.has_pending = False
        return None, None, r               # (terminal_ctx, delta, reward) — THREE


def _stub_recorder(monkeypatch, rewards=()):
    """Swap the REAL `RewardTracker`/`BattleContext` out so a recorder can be built without a
    battle. Used by the seam tests, which are about WHICH methods get patched, not about rewards."""
    monkeypatch.setattr("agents.training.cf_mc_return.RewardTracker",
                        lambda *_a, **_kw: _StubTracker(rewards))
    monkeypatch.setattr("agents.training.cf_mc_return.BattleContext",
                        type("_Ctx", (), {"from_battle": staticmethod(lambda *a, **k: object())}))


def _recorder(rewards, monkeypatch):
    monkeypatch.setattr("agents.training.cf_mc_return.RewardTracker",
                        lambda *_a, **_kw: _StubTracker(rewards))
    monkeypatch.setattr("agents.training.cf_mc_return.BattleContext",
                        type("_Ctx", (), {"from_battle": staticmethod(lambda *a, **k: object())}))
    return CfReturnRecorder(object)


class _Battle:
    event_cursor = 0


def test_rewards_before_ARMING_are_discarded(monkeypatch):
    """The prefix's rewards belong to states the label is not about. Including them would inflate
    every label by a constant that varies with the divergence turn."""
    rec = _recorder([10.0, 20.0, 30.0, 40.0], monkeypatch)
    b = _Battle()
    rec.note(b, 0, [1])          # prefix decision 0  (settles nothing)
    rec.note(b, 0, [1])          # settles decision 0 -> 10.0, DISCARDED
    rec.arm_at_next()
    rec.note(b, 0, [1])          # settles decision 1 -> 20.0, STILL discarded (see below)
    rec.note(b, 0, [1])          # settles the ARMED decision -> 30.0, counted
    rec.finalize(b)              # settles the last -> 40.0, counted
    assert rec.rewards() == [30.0, 40.0]


def test_arming_is_delayed_by_exactly_one_decision(monkeypatch):
    """THE OFF-BY-ONE this class exists for.

    `arm_at_next()` is called from the scripted-prefix hook AT the divergence decision, and that
    decision's own reward is settled by the NEXT `note`. Arming eagerly would make the first
    counted reward belong to the decision BEFORE the divergence.
    """
    rec = _recorder([1.0, 2.0, 3.0], monkeypatch)
    b = _Battle()
    rec.note(b, 0, [1])          # decision T-1
    rec.arm_at_next()            # ...the hook fires here, at decision T
    assert rec.armed is False, "arming must not take effect until the next note()"
    rec.note(b, 0, [1])          # decision T: settles T-1's reward (1.0) -> discarded; now armed
    assert rec.armed is True
    rec.note(b, 0, [1])          # settles T's reward (2.0) -> COUNTED
    rec.finalize(b)              # settles T+1's reward (3.0) -> counted
    assert rec.rewards() == [2.0, 3.0]


def test_finalize_is_idempotent(monkeypatch):
    """poke-env's `_battle_finished_callback` can fire more than once for a battle tag; a second
    finalize must not append the terminal reward twice (which would silently double the most
    heavily-weighted term in the whole return)."""
    rec = _recorder([5.0, 7.0], monkeypatch)
    b = _Battle()
    rec.arm_at_next()
    rec.note(b, 0, [1])
    rec.note(b, 0, [1])
    rec.finalize(b)
    before = rec.rewards()
    rec.finalize(b)
    assert rec.rewards() == before


def test_value_folds_the_counted_rewards_with_the_given_gamma(monkeypatch):
    """Armed BEFORE any decision (the degenerate case: the divergence is our first move), so every
    settled reward counts and the fold is the whole sequence."""
    rec = _recorder([9.0, 1.0, 2.0], monkeypatch)
    b = _Battle()
    rec.arm_at_next()
    rec.note(b, 0, [1])          # the ARMED decision (settles nothing yet)
    rec.note(b, 0, [1])          # settles it -> 9.0, counted
    rec.note(b, 0, [1])          # settles the next -> 1.0, counted
    rec.finalize(b)              # settles the last -> 2.0, counted
    assert rec.rewards() == [9.0, 1.0, 2.0]
    assert rec.value(0.5) == pytest.approx(9.0 + 0.5 * 1.0 + 0.25 * 2.0)


def test_a_recorder_that_never_armed_reports_NONE_not_zero(monkeypatch):
    """A rollout whose divergence decision was never reached (the battle ended in the prefix) has
    produced NO measurement. Zero is the middle of this reward's range, so reporting it would be
    indistinguishable from a genuinely neutral game — the most plausible-looking wrong answer
    available, on exactly the rollouts that failed."""
    rec = _recorder([1.0, 2.0], monkeypatch)
    b = _Battle()
    rec.note(b, 0, [1])
    rec.note(b, 0, [1])
    rec.finalize(b)
    assert rec.rewards() == [] and rec.value(0.99) is None


# --------------------------------------------------------------------------------------
# THE TWO SEAM BUGS the adversarial review caught (2026-08-22). Both were shipped in the first
# version of this module, both produced plausible-looking `mc_return` values, and NEITHER was
# visible to the bridge-backed composition test — which only asserted a value was PRESENT.
# --------------------------------------------------------------------------------------

def test_the_recorder_NEVER_patches_action_to_order(monkeypatch):
    """THE AMPLIFICATION BUG, pinned on the mechanism.

    `action_to_order` looks like the perfect seam (it is the commit point and it raises
    `StaleDecisionError` on a superseded attempt) — but `counterfactual._invert_choice` calls it in
    a **LOOP over every legal index** to recover a recorded choice's action number, on every
    scripted decision of the prefix. Recording there fired 6-9 times per scripted turn with actions
    that were never played, each advancing the STATEFUL reward function (PBRS potentials, the
    progress clock) and, once armed, appending bogus rewards. The digest still matched, so nothing
    downstream could tell.

    This test fails if the seam is moved back.
    """
    from agents.training.cf_mc_return import attach_return_recording

    class _Player:
        def _predict_best_action(self, *a, **kw):
            return 3, None, [1, 1]
        def choose_move(self, battle):
            return "order"
        def action_to_order(self, idx, battle):
            return "order"
        def _battle_finished_callback(self, battle):
            return None

    p = _Player()
    _stub_recorder(monkeypatch)
    attach_return_recording(p)
    # Checked on the INSTANCE dict, not by comparing bound methods: `p.action_to_order` builds a
    # fresh bound-method object on every access, so an identity compare there always fails.
    assert "action_to_order" not in vars(p), (
        "the recorder patched action_to_order — `_invert_choice` calls it once per LEGAL ACTION on "
        "every scripted decision, so every mc_return would be built from moves never played")


def test_a_player_without_the_seams_is_REFUSED_rather_than_crashing():
    """A poke-env baseline has neither seam. The producer must ship no `mc_return` — the field is
    optional per row and the buffer masks it — rather than take the whole record down."""
    from agents.training.cf_mc_return import attach_return_recording
    assert attach_return_recording(object()) is None


def test_the_live_seam_records_ONCE_per_decision_even_across_a_re_decide(monkeypatch):
    """`choose_move` is the boundary, not `_predict_best_action`.

    `RLPlayer` re-decides on a `StaleDecisionError` (the live battle advanced under the forward), so
    a decision can PREDICT several times and COMMIT once. Recording per prediction would append a
    reward for a superseded attempt — and, worse, `begin_turn` it with the superseded action.
    """
    from agents.training.cf_mc_return import attach_return_recording
    noted = []

    class _Player:
        def __init__(self):
            self.attempts = 0
        def _predict_best_action(self, *a, **kw):
            self.attempts += 1
            return self.attempts, None, [1, 1]      # a DIFFERENT index per attempt
        def choose_move(self, battle):
            self._predict_best_action()             # attempt 1 — stale
            self._predict_best_action()             # attempt 2 — commits
            return "order"
        def _battle_finished_callback(self, battle):
            return None

    p = _Player()
    _stub_recorder(monkeypatch)
    rec = attach_return_recording(p)
    rec.note = lambda b, i, m: noted.append(i)
    p.choose_move(_Battle())
    assert noted == [2], f"expected one record of the COMMITTED attempt, got {noted}"


def test_arming_and_noting_TOGETHER_counts_the_divergence_turns_own_reward(monkeypatch):
    """THE OFF-BY-ONE, pinned end to end at the recorder's API.

    The producer's `on_scripted_decision` hook must call `arm_at_next()` **and then** `note()` for
    the divergence decision — because turn T's move is SCRIPTED and never reaches the live
    `choose_move`. Arming alone (leaving the note to the first live decision at T+1) makes T+1 the
    armed decision and drops r_T: the label becomes G(s_{T+1}) against an obs row for s_T, biased by
    whatever happened on the divergence turn. This test is the difference between the two.
    """
    rec = _recorder([10.0, 20.0, 30.0], monkeypatch)
    b = _Battle()
    rec.note(b, 0, [1])                 # turn T-1, live
    # --- what the hook does, in the order it must do it ---
    rec.arm_at_next()
    rec.note(b, 7, [1])                 # turn T, SCRIPTED: settles r_{T-1} (10.0) -> discarded
    # --- back to the live path ---
    rec.note(b, 0, [1])                 # turn T+1: settles r_T (20.0) -> COUNTED
    rec.finalize(b)                     # settles r_{T+1} (30.0) -> counted
    assert rec.rewards() == [20.0, 30.0], (
        "r_T is missing — the hook must NOTE the divergence decision, not merely arm for it")


def test_arming_WITHOUT_noting_silently_drops_r_T(monkeypatch):
    """The negative control for the test above: the shape the bug had.

    Kept because the failure is invisible in every scalar — the label is a plausible return of a
    real game, one turn late — so the only defence is this contrast being written down.
    """
    rec = _recorder([10.0, 20.0, 30.0], monkeypatch)
    b = _Battle()
    rec.note(b, 0, [1])
    rec.arm_at_next()                   # the hook arms but does NOT note
    rec.note(b, 0, [1])                 # the first LIVE decision at T+1 settles r_{T-1}, discarded
    rec.finalize(b)
    assert rec.rewards() == [20.0], "preconditions: this is the buggy shape, one reward late"
