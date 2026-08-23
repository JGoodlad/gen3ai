"""MC RETURN labels — the shadow critic's supervision, measured off a real rollout.

`gen3_cf_twin_heads_v1`. The counterfactual label factory already manufactures tight-MC
**win-probability** labels: roll a recorded decision to termination R times and count wins. The
SHADOW CRITIC (``agents.model.aux_value_heads.ShadowValueHead``) needs the same treatment applied to
a different quantity — the **shaped RETURN**, in the units the live critic V actually predicts:

    G(s_T) = Σ_{k≥0} γᵏ · r_{T+k}

averaged over the R rollouts. That is a tight, ground-truth estimate of exactly what V is supposed
to be, and the on-policy stream structurally cannot produce it: it sees ONE realized trajectory per
state, drenched in the variance the whole factory exists to remove.

WHY THIS MODULE EXISTS RATHER THAN A CALL TO ``RewardTracker``
--------------------------------------------------------------
Three facts make the naive wiring wrong, and all three have a bug attached:

1. **``RewardTracker`` accumulates an UNDISCOUNTED total.** ``total_reward`` is Σr over the whole
   battle; a return is Σγᵏr from a *particular* state. So the per-turn rewards have to be captured
   in order, not summed as they arrive.
2. **The rollout's prefix is SCRIPTED, and so is the divergence turn itself.** The counterfactual
   runner replays the recorded prefix and substitutes our choice at turn T, handing control to the
   live policy only from T+1. A reward tracker hooked into the live ``choose_move`` therefore begins
   at T+1 — its return is missing ``r_T`` and carries an extra factor of γ, against the very state
   the label is FOR. That is why `install_scripted_prefix` grew an ``on_scripted_decision`` hook:
   the divergence decision must be tracked too.
3. **The reward is a per-RUN object.** A shaped return means nothing without the `RewardConfig` that
   produced it, so the producer stamps `reward_config_digest(...)` on every row and the consuming
   buffer refuses a mismatch.

THE ARMING CONTRACT. :meth:`CfReturnRecorder.arm_at_next` says "the return starts at the next
decision you tell me about". Rewards settled while un-armed are DISCARDED rather than accumulated:
the prefix's rewards belong to states the label is not about, and silently including them would
inflate every label by a constant that VARIES WITH THE DIVERGENCE TURN — a bias shaped exactly like
a real signal. The rewards are still computed, because the reward function is stateful (PBRS
potentials, the progress clock) and starting it mid-battle would begin the potentials from a board
that never existed.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from agents.training.battle_snapshot import BattleContext
from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_tracker import RewardTracker
from agents.training.slot_registry import SlotRegistry


def discounted_return(rewards, gamma: float) -> float:
    """``Σ γᵏ · rewards[k]`` — the return of a reward SEQUENCE, oldest first.

    Pure and tiny on purpose: it is the one piece of arithmetic in this module that a test can pin
    against a hand-computed number, and getting the ORDER wrong (newest-first) is a silent error
    that produces plausible magnitudes. Horner's form, so the accumulation is exact for the short
    sequences a rollout produces and has no repeated ``gamma**k`` rounding.
    """
    total = 0.0
    for r in reversed(list(rewards)):
        total = float(r) + gamma * total
    return float(total)


class CfReturnRecorder:
    """Per-battle: the ordered shaped rewards from the ARMED decision onward.

    Wraps a `RewardTracker` (the server-free reward path — the same deferred begin/complete/finalize
    dance `Gen3Env` runs, so the numbers are the training reward and not a re-derivation of it) and
    keeps what the tracker throws away: the per-turn reward, in order.
    """

    def __init__(self, reward_fn_factory=Gen3RewardManager) -> None:
        self._tracker = RewardTracker(reward_fn_factory, SlotRegistry(), SlotRegistry())
        self._rewards: List[float] = []
        self._armed = False
        self._pending_arm = False
        self.finished = False

    @property
    def armed(self) -> bool:
        return self._armed

    def note(self, battle, action_idx: int, mask) -> None:
        """One of OUR decisions, at commit time. Mirrors `RewardTrackingMixin._track_reward`.

        The tracker is DEFERRED by construction: the reward for decision k is only computable once
        decision k+1's board exists, so ``complete_pending`` settles the previous turn and
        ``begin_turn`` latches this one. Rewards settled while un-armed are dropped (see the module
        docstring's ARMING CONTRACT) — the tracker still has to run through them, because the reward
        function is STATEFUL (PBRS potentials, the progress clock) and skipping the prefix entirely
        would start the potentials from a board that never existed.
        """
        ctx = BattleContext.from_battle(
            battle, np.asarray(mask), self._tracker._our_slots, self._tracker._opp_slots)
        if self._tracker.has_pending:
            # ASYMMETRIC ARITIES, and they are the tracker's, not a typo: `complete_pending`
            # returns `(delta, reward)` while `finalize` returns `(terminal_ctx, delta, reward)` —
            # the terminal context has to be BUILT (all-zero mask) and callers want it. Taking the
            # reward off the END of each is what keeps that difference from becoming an off-by-one.
            reward = self._tracker.complete_pending(ctx, battle)[-1]
            if self._armed:
                self._rewards.append(float(reward))
        self._tracker.begin_turn(ctx, int(action_idx), getattr(battle, "event_cursor", 0))
        # Arm AFTER settling: `arm_at_next()` means "the return starts at the decision I am about
        # to note", and the reward settled just above belongs to the PREVIOUS one.
        if self._pending_arm:
            self._armed = True
            self._pending_arm = False

    def arm_at_next(self) -> None:
        """Arm so the NEXT :meth:`note` is the first decision whose reward counts.

        The one-decision delay is the off-by-one this class exists to avoid. `note` settles the
        PREVIOUS decision's reward before latching this one, so flipping the flag on entry would
        count the reward of the decision BEFORE the divergence — a whole extra turn of shaping
        folded into every label, varying with the divergence turn, i.e. a bias shaped exactly like
        a real signal.
        """
        self._pending_arm = True

    def finalize(self, battle) -> None:
        """Settle the last pending turn once the battle ends (the terminal reward lives here)."""
        if self.finished:
            return
        self.finished = True
        if self._tracker.has_pending:
            reward = self._tracker.finalize(battle)[-1]      # (terminal_ctx, delta, reward)
            if self._armed:
                self._rewards.append(float(reward))

    def rewards(self) -> List[float]:
        return list(self._rewards)

    def value(self, gamma: float) -> Optional[float]:
        """The discounted return, or None when nothing was ever counted.

        None rather than 0.0 is load-bearing: a rollout that never armed (the divergence turn was
        never reached, the battle ended in the prefix) has produced NO measurement, and a zero would
        be indistinguishable from a genuinely neutral outcome — which is the middle of this reward's
        range, i.e. the most plausible-looking wrong answer available.
        """
        if not self._rewards:
            return None
        return discounted_return(self._rewards, gamma)


def attach_return_recording(player, *, reward_fn_factory=Gen3RewardManager):
    """Give an ``RLPlayer`` per-battle return recording, without reimplementing its decision path.

    Returns the :class:`CfReturnRecorder`, or **None** when the player does not expose the seams
    below — a poke-env baseline has neither, and a producer handed one must ship no ``mc_return``
    rather than crash the record. Refusing is the honest degradation: the shadow critic's stream is
    optional per row and the buffer masks it, so an unmeasurable rollout costs that one field.

    ⚠️ **`action_to_order` IS NOT A SEAM HERE, and that is the whole subtlety.** It looks like the
    ideal one (it is the commit point, and it raises `StaleDecisionError` on a superseded attempt) —
    but `counterfactual._invert_choice` calls ``player.action_to_order(idx, battle)`` in a **LOOP
    over every legal index** to recover a recorded choice's action number, on *every scripted
    decision of the prefix*. Recording there therefore fires 6-9 times per scripted turn with
    actions that were never played, each one advancing the STATEFUL reward function (PBRS
    potentials, the progress clock) and, once armed, appending bogus rewards. Every `mc_return`
    would be garbage, the reward digest would still match, and nothing downstream could tell.

    So the seams are:

    * ``_predict_best_action`` — caches ``(idx, mask)``. Called once per live decision ATTEMPT, so a
      re-decide overwrites the cache with the attempt that actually committed. `_invert_choice`
      never calls it.
    * ``choose_move`` — the record happens when the player's OWN ``choose_move`` returns, i.e. once
      per live decision, whatever happened inside. It must be wrapped BEFORE
      `install_scripted_prefix` runs, because that function captures ``player.choose_move`` as its
      live-path delegate — which is exactly the order the producer uses.
    * ``_battle_finished_callback`` — settles the last pending turn, where the TERMINAL reward
      lives. Without it every label would be missing the largest term in its own return.

    The SCRIPTED decisions (the prefix, and the divergence turn itself) are not covered by any of
    these by design — the divergence move never reaches the live `choose_move`. Its reward is picked
    up by the caller through `install_scripted_prefix`'s ``on_scripted_decision`` hook, which must
    call :meth:`CfReturnRecorder.arm_at_next` and then :meth:`CfReturnRecorder.note` **in that
    order**; see `cf_producer._rollout`.

    One recorder per player per rollout — the producer builds a fresh player for every rollout, so
    there is no cross-battle state to reset.
    """
    if not (hasattr(player, "_predict_best_action") and hasattr(player, "choose_move")):
        return None
    rec = CfReturnRecorder(reward_fn_factory)
    orig_predict = player._predict_best_action
    orig_choose = player.choose_move
    orig_finished = player._battle_finished_callback
    pending: dict = {}

    def _predict(*a, **kw):
        out = orig_predict(*a, **kw)
        pending["idx"], pending["mask"] = out[0], out[2]
        return out

    def _choose(battle):
        pending.clear()                      # a decision that never predicts records nothing
        order = orig_choose(battle)
        if pending.get("idx") is not None and pending.get("mask") is not None:
            rec.note(battle, int(pending["idx"]), pending["mask"])
        return order

    def _finished(battle):
        rec.finalize(battle)
        return orig_finished(battle)

    player._predict_best_action = _predict
    player.choose_move = _choose
    player._battle_finished_callback = _finished
    return rec
