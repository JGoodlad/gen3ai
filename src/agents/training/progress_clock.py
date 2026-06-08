"""``ProgressClock`` — the episode-scoped no-progress counter (design §4 / §5.1).

A cross-turn counter is NOT current-board state, so it cannot live in ``LiveView`` (primitives only,
no past-turn state). The precedent is ``HiddenPowerTracker``: owned by ``EpisodeTracker``, updated at
``record()`` (embed) time, threaded into ``encode()``. ``turns_since_progress`` follows the identical
pattern — and crucially it is read by BOTH the obs encoder (the ``value()`` scalar) and the reward
(``last_penalty``), so obs and reward key on ONE value (the whole point of the Markovian design).

**Timing (design §5.1).** poke-env's ``env.step`` runs ``embed_battle`` (the *next* obs) BEFORE
``calc_reward`` (the *current* reward). ``EpisodeTracker.record`` (inside ``embed_battle``) calls
:meth:`update` for the just-completed decision window — so the obs is always FRESH — and stashes the
penalty for that window in :attr:`last_penalty`; ``Gen3RewardManager.process_turn_reward`` then reads
it. Result: the obs the model saw and the value the penalty keys on are the same number.

**Three outcomes per window** (design §4.1 / §4.1.1 / §4.1.2):
  * PROGRESS — our-attributed offense advanced the game → reset ``n`` to 0.
  * DENIED   — a progress attempt denied by exogenous RNG / opponent action, OR a productive
               defensive action whose value a Φ potential already prices → FREEZE ``n`` (no charge).
  * NO_OP    — a deliberate, obs-knowable wheel-spin → increment ``n`` + charge ``p(n)``.
Forced-switch windows are no-ops of the clock entirely (no increment, no charge); the charge is
suppressed when no switch is legal (trapped-vs-wall helplessness must not be punished).
"""
from __future__ import annotations

import math
from typing import Optional

from agents.gen3_data import moves as _movedex
from agents.gen3_mechanics import INVULNERABLE_MOVES as _INVULNERABLE_MOVES

PROGRESS_DMG_EPS = 0.03    # our-attributed damage floor (3% of a bar) — above Sandstorm/Leech chip
PROGRESS_CLOCK_CAP = 10    # turns_since_progress clamp (obs scalar + penalty)
_LOG_DENOM = math.log(1.0 + PROGRESS_CLOCK_CAP)


class ProgressClock:
    """Episode-scoped ``turns_since_progress`` counter. Owned by ``EpisodeTracker``; read by the obs
    encoder (:meth:`value`) and the reward manager (:attr:`last_penalty`)."""

    def __init__(self, no_progress_penalty: float = 0.15) -> None:
        self.n: int = 0
        self.last_penalty: float = 0.0   # penalty for the most-recently-folded window (read by reward)
        # The FLAT per-no-op magnitude (>0). A per-run constant set once from
        # RewardConfig.no_progress_penalty (the env wires it); inference/standalone use the default,
        # which is inert there (only the reward reads last_penalty). Keeping it on the clock means
        # update() stays an obs-side call that needs no reward param.
        self.no_progress_penalty: float = no_progress_penalty
        self._prev_spikes: int = 0

    def reset(self) -> None:
        self.n = 0
        self.last_penalty = 0.0
        self._prev_spikes = 0

    def value(self) -> float:
        """The obs scalar: log-saturated ``turns_since_progress`` ∈ [0,1] (same form as the global
        turn clock). Saturating because the marginal of one more no-progress turn matters most early."""
        return math.log(1.0 + min(self.n, PROGRESS_CLOCK_CAP)) / _LOG_DENOM

    def update(self, delta, live, legal) -> None:
        """Fold one resolved decision window: classify PROGRESS / DENIED / NO_OP, update ``n``, and
        stash :attr:`last_penalty` (= the FLAT :attr:`no_progress_penalty` on a charged no-op)."""
        opp_spikes_now = self._opp_spikes(live)
        prev_spikes = self._prev_spikes
        self._prev_spikes = opp_spikes_now

        # Forced-switch / post-faint replacement: only switches were legal → the clock sits out.
        if getattr(delta, "phase_is_forced_switch", False):
            self.last_penalty = 0.0
            return

        if self._is_progress(delta, live, prev_spikes, opp_spikes_now):
            self.n = 0
            self.last_penalty = 0.0
            return

        if self._is_denied(delta):
            # Attempted progress denied by RNG/opponent, or a productive defensive action a Φ prices.
            self.last_penalty = 0.0   # FREEZE — neither increment nor reset.
            return

        # NO_OP: a deliberate wheel-spin → increment + charge, unless trapped with no switch.
        self.n = min(self.n + 1, PROGRESS_CLOCK_CAP)
        switch_legal = legal is not None and len(getattr(legal, "switches", ()) or ()) > 0
        self.last_penalty = (-abs(self.no_progress_penalty)) if switch_legal else 0.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _opp_spikes(live) -> int:
        if live is None:
            return 0
        return int((live.opp.side_conditions or {}).get("spikes", 0))

    @staticmethod
    def _is_progress(delta, live, prev_spikes: int, opp_spikes_now: int) -> bool:
        # (i) OUR move dealt net damage above the floor to a non-fainted opp (our-attributed — NOT
        #     net opp HP, which would let passive Sandstorm/Leech chip reset the clock for free).
        ev = getattr(delta, "our_damaging_event", None)
        tgt = getattr(delta, "opp_target_hp_delta", None)
        if ev is not None and tgt is not None and float(tgt) <= -PROGRESS_DMG_EPS:
            return True
        # (ii) a status LANDED on the opp this window (the transition event, not a re-tick).
        if getattr(delta, "opp_status_applied", None) is not None:
            return True
        # (iii) a hazard LAYER was strictly added (not "used Spikes at 3").
        if opp_spikes_now - prev_spikes > 0:
            return True
        # (iv) we forced an opp commit (phaze / forced opp switch).
        if getattr(delta, "opp_switch_to", None) is not None:
            return True
        return False

    @staticmethod
    def _is_denied(delta) -> bool:
        # Our chosen move was PREVENTED (cant: para/sleep/freeze/flinch/focuspunch) — exogenous.
        if getattr(delta, "our_failed_to_move", False):
            return True
        outcome = getattr(delta, "our_move_outcome", None)
        # Accuracy MISS — the agent made a good attempt; RNG denied it.
        if outcome == "miss":
            return True
        # BLOCKED by the opponent's Protect/Detect/Endure (their choice denied our attempt). An
        # immune attack (|-immune|, our_effectiveness==0) is NOT denied — it's a deterministic NO_OP.
        if outcome == "fail":
            opp_mv = getattr(delta, "opp_resolved_move_id", None)
            if opp_mv in _INVULNERABLE_MOVES:
                return True
        # Productive DEFENSIVE action: a heal move that restored real HP (Φ_mat already prices it),
        # so the clock stays out — neither stall nor offense (design §4.1.2).
        mid = getattr(delta, "our_move_id", None)
        if mid is not None:
            try:
                healed = float(delta.our_hp_delta.sum()) > PROGRESS_DMG_EPS
            except (AttributeError, TypeError):   # mock/standalone delta without a real hp array
                healed = False
            if healed:
                md = _movedex.get(mid)
                if md is not None and getattr(md, "is_heal", False):
                    return True
        return False
