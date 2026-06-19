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

**Capped-Spikes short-circuit.** Spikes used at the 3-layer cap is forced to a charged NO_OP BEFORE
the progress/denial classification — it can never add a layer, and must not be rescued by clause (iv)
(an opp switch, incl. a VOLUNTARY pivot into our hazards, counting as our progress). Without this, a
wasted Spikes that banks switch-in material escapes the tax on exactly the turns it is rewarded.

**Filler-RapidSpin gate.** RapidSpin used with NO hazards on our side to clear is a 20-BP filler
pseudo-attack — its trivial chip (and any incidental opp switch) is barred from counting as progress,
so it falls through to the NO_OP charge. A spin that genuinely clears our hazards (our side had spikes)
or lands a KO IS real progress; a spin denied by RNG is still frozen by `_denial_kind`.
"""
from __future__ import annotations

import math
from typing import Optional

from agents.gen3_data import moves as _movedex
from agents.gen3_mechanics import INVULNERABLE_MOVES as _INVULNERABLE_MOVES

PROGRESS_DMG_EPS = 0.03    # our-attributed damage floor (3% of a bar) — above Sandstorm/Leech chip
PROGRESS_CLOCK_CAP = 10    # turns_since_progress clamp (obs scalar + penalty)
_LOG_DENOM = math.log(1.0 + PROGRESS_CLOCK_CAP)

# A productive defensive heal is DENIED (frozen, no charge) for this many CONSECUTIVE heal windows —
# a one-off heal answering incoming pressure is free. Beyond it, sustained healing with no offensive
# progress is a heal-war (the 250-turn mirror-stall failure mode): the freeze lifts and each further
# heal becomes a charged NO_OP, so both the obs clock and the reward finally register the stall. A
# WINNING residual-stall (our Toxic/Leech chipping the opp down) is caught by `_is_progress` first,
# so it resets the streak and is never charged — only a true no-net-progress heal-war is.
HEAL_FREEZE_GRACE = 2

# A Rest-LOOP — our active Rests, wakes, then Rests AGAIN — is a deliberate no-progress wheel-spin, not a
# free defensive heal. So it is denied the HEAL_FREEZE_GRACE and counted as a NO_OP (stalled) turn: it
# advances the no-progress clock + charges no_progress_tax, exactly like any other wheel-spin. EXEMPT: a
# mon carrying one of these moves can act while asleep, so looping Rest is a legitimate strategy (the
# canonical Rest + Sleep Talk CroCune) and is never penalised. A WINNING residual stall (Toxic/Leech
# chipping the opp down while we Rest) is caught by `_is_progress` first, so it is exempt too — only a
# genuinely no-net-progress rest-loop is charged. (frozenset for easy extension, mirroring
# sleep_belief._SLEEP_USABLE_MOVES.)
_REST_LOOP_EXEMPT_MOVES = frozenset({"sleeptalk"})


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
        self._prev_spikes: int = 0        # opp-side spike layers after last window (hazard-add check)
        self._prev_our_spikes: int = 0    # our-side spike layers after last window (filler-spin check)
        self._heal_streak: int = 0   # consecutive productive-heal windows (for HEAL_FREEZE_GRACE)
        # Rest-loop bookkeeping: the per-species set of mons that have already SUCCESSFULLY Rested this
        # episode (so a 2nd Rest is a wake-then-re-Rest loop), and the flag for the just-folded window
        # (a non-Sleep-Talk re-Rest) — read internally to deny the heal-grace and charge it as a NO_OP.
        self._rested_species: set = set()
        self._is_rest_loop: bool = False

    def reset(self) -> None:
        self.n = 0
        self.last_penalty = 0.0
        self._prev_spikes = 0
        self._prev_our_spikes = 0
        self._heal_streak = 0
        self._rested_species = set()
        self._is_rest_loop = False

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

        our_spikes_now = self._our_spikes(live)
        prev_our_spikes = self._prev_our_spikes
        self._prev_our_spikes = our_spikes_now

        # Rest-loop detection drives the heal-grace bypass in the denial section below; fold it up front
        # so the per-species rest history is recorded even on windows that early-return (e.g. a winning
        # residual rest-stall hits the progress branch — exempt, but the Rest still counts as "rested").
        self._update_rest_loop(delta, live)

        # Forced-switch / post-faint replacement: only switches were legal → the clock sits out.
        if getattr(delta, "phase_is_forced_switch", False):
            self.last_penalty = 0.0
            return

        # Spikes AT THE 3-LAYER CAP is a deliberate, obs-knowable wheel-spin: the move can NEVER add a
        # layer (the obs carries opp_spikes==3), so it is a NO_OP by definition. It must NOT be rescued
        # by an incidental opp switch (clause (iv) of _is_progress credits ANY opp commit as progress —
        # including a VOLUNTARY pivot into our hazards — which would reset the clock on exactly the turns
        # the wasted Spikes also banks switch-in material), nor frozen as an "exogenous fail". So charge
        # it as a NO_OP directly, BEFORE the progress / denial classification. (A layer-adding Spikes —
        # opp_spikes strictly rose — still resets via _is_progress clause (iii) below.)
        if (getattr(delta, "our_move_id", None) == "spikes"
                and opp_spikes_now >= 3 and opp_spikes_now - prev_spikes <= 0):
            self.n = min(self.n + 1, PROGRESS_CLOCK_CAP)
            switch_legal = legal is not None and len(getattr(legal, "switches", ()) or ()) > 0
            self.last_penalty = (-abs(self.no_progress_penalty)) if switch_legal else 0.0
            self._heal_streak = 0
            return

        # A self-status-cure move (Refresh) used with NO status to cure is a deliberate wasted no-op: the
        # mon is unstatused, so the move does nothing (it cured nothing this window). Charge it as a NO_OP
        # directly, BEFORE the progress/denial classification — otherwise an incidental WINNING residual
        # (our Leech Seed / Toxic chipping the opp net-down, clause (v)) would launder the wasted Refresh
        # into "progress" and rescue it from the tax (the observed Refresh-spam-while-seeded stall). A
        # Refresh that ACTUALLY cures a status sets `our_status_cured`, so it is NOT wasted and falls
        # through to the normal path.
        if self._is_wasted_self_cure(delta):
            self.n = min(self.n + 1, PROGRESS_CLOCK_CAP)
            switch_legal = legal is not None and len(getattr(legal, "switches", ()) or ()) > 0
            self.last_penalty = (-abs(self.no_progress_penalty)) if switch_legal else 0.0
            self._heal_streak = 0
            return

        # RapidSpin with NO hazards on our side to clear is a filler 20-BP pseudo-attack — not real
        # offense, not utility. Its trivial chip (and any incidental opp switch) must NOT launder into
        # "progress" that resets the stall clock, so we disable the progress reset for it and let it
        # fall through to the NO_OP charge below. A spin that genuinely clears our hazards
        # (prev_our_spikes > 0) or lands a KO (opp_fainted) IS real progress and reaches _is_progress;
        # a spin denied by RNG (flinch / cant) is still frozen by _denial_kind.
        filler_spin = (getattr(delta, "our_move_id", None) == "rapidspin"
                       and prev_our_spikes == 0
                       and not getattr(delta, "opp_fainted", False))

        if not filler_spin and self._is_progress(delta, live, prev_spikes, opp_spikes_now):
            self.n = 0
            self.last_penalty = 0.0
            self._heal_streak = 0
            return

        kind = self._denial_kind(delta)
        if kind == "exogenous":
            # A genuine attempt denied by RNG / the opponent (cant / miss / Protect-block) — not a
            # stall. FREEZE — neither increment nor charge. (Leaves _heal_streak intact so a heal-war
            # interrupted by one miss still accumulates.)
            self.last_penalty = 0.0
            return
        if kind == "heal":
            # A productive defensive heal. Free for the first HEAL_FREEZE_GRACE consecutive windows
            # (answering pressure, priced by Φ_mat); a SUSTAINED heal with no offensive progress is a
            # heal-war → fall through to the NO_OP charge below (do NOT reset the streak, so it keeps
            # charging every subsequent heal turn). A REST-LOOP (the mon already Rested this episode,
            # woke, and re-Rested without Sleep Talk) gets NO grace — a wake-then-re-Rest is a stalled
            # turn the moment it repeats, so it falls straight through to the NO_OP charge.
            self._heal_streak += 1
            if not self._is_rest_loop and self._heal_streak <= HEAL_FREEZE_GRACE:
                self.last_penalty = 0.0
                return
        else:
            self._heal_streak = 0   # a non-heal no-op breaks any heal-war run

        # NO_OP (deliberate wheel-spin) or a sustained heal-war → increment + charge, unless trapped
        # with no switch (helplessness must not be punished).
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
    def _our_spikes(live) -> int:
        if live is None:
            return 0
        return int((live.ours.side_conditions or {}).get("spikes", 0))

    def _update_rest_loop(self, delta, live) -> None:
        """Detect a Rest-LOOP — our active Rested earlier this episode, woke, and Rests AGAIN — and stash
        :attr:`_is_rest_loop` (read in the heal branch to deny the grace). A 2nd successful Rest necessarily
        brackets a wake (a mon can't pick Rest while asleep, and Rest fails when already asleep), so a
        per-species rest count ≥2 IS the wake-then-re-Rest loop. EXEMPT: a Sleep-Talk mon can act while
        asleep, so looping Rest is a legitimate strategy → never flagged. Tracks the per-species rest
        history (episode-scoped)."""
        self._is_rest_loop = False
        if getattr(delta, "our_move_id", None) != "rest" or getattr(delta, "our_failed_to_move", False):
            return
        # A Rest that actually EXECUTED self-applies sleep; a Rest at full HP fails (no sleep) and is not a
        # rest — so it neither flags a loop nor counts toward the history. our_status_applied is the gained
        # Status enum this window; on a Rest turn the only self-gained status is the Rest sleep.
        st = getattr(delta, "our_status_applied", None)
        if st is None or getattr(st, "name", "").upper() != "SLP":
            return
        species = getattr(delta, "our_prev_active", None)   # the mon that used Rest (it did not switch)
        if not species:
            return
        if species in self._rested_species and not self._active_has_move(live, _REST_LOOP_EXEMPT_MOVES):
            self._is_rest_loop = True
        self._rested_species.add(species)

    @staticmethod
    def _is_wasted_self_cure(delta) -> bool:
        """A self-status-cure move (Refresh) used with NO status to cure — it cured nothing this window
        (``our_status_cured`` is None) and was not prevented (a cant). Detected by the move's
        ``cures_self_status`` data flag (Refresh), so it is data-driven, not a hardcoded id. A Refresh
        that DOES cure a status sets ``our_status_cured`` → not wasted. (We move first, so a Refresh that
        is immediately re-statused by the opponent still cured nothing AT cast time → ``our_status_cured``
        stays None → correctly wasted.)"""
        mid = getattr(delta, "our_move_id", None)
        if mid is None or getattr(delta, "our_failed_to_move", False):
            return False
        if getattr(delta, "our_status_cured", None) is not None:
            return False
        md = _movedex.get(mid)
        return md is not None and getattr(md, "cures_self_status", False)

    @staticmethod
    def _active_has_move(live, move_ids) -> bool:
        """True iff OUR active mon's moveset contains any of ``move_ids``. Our own side reveals the full
        4-move set (LiveView), so the Sleep-Talk exemption is exact (no belief needed)."""
        if live is None or getattr(live, "ours", None) is None:
            return False
        mon = live.ours.active
        if mon is None:
            return False
        try:
            return any(mid in move_ids for mid in mon.move_ids)
        except (AttributeError, TypeError):
            return False

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
        # (v) an our-OWNED residual (Toxic/poison/burn status, or Leech Seed / Curse / Nightmare) is
        #     chipping the opp DOWN this window — a slow-damage win condition IS progress even when
        #     our move dealt no direct damage, so a WINNING residual-stall (e.g. Recover-while-Toxic-
        #     ticks) is never charged. Gated on the opp NET-losing HP, so a heal-war where recovery
        #     out-paces the chip (opp HP flat/up) is NOT credited and still charges (HEAL_FREEZE_GRACE).
        #     Status/volatiles on the OPP are our-applied in 1v1 (we never poison/seed ourselves onto
        #     them), so the residual is unambiguously ours; weather is excluded (not per-mon, ambiguous).
        if live is not None and getattr(live, "opp", None) is not None:
            opp_mon = live.opp.active
            if opp_mon is not None:
                status = getattr(opp_mon, "status", None)
                vols = getattr(opp_mon, "volatiles", None) or ()
                owned_residual = (status in ("tox", "psn", "brn")
                                  or "leechseed" in vols or "curse" in vols or "nightmare" in vols)
                if owned_residual:
                    try:
                        opp_net = float(delta.opp_hp_delta.sum())
                    except (AttributeError, TypeError):   # mock delta without a real hp array
                        opp_net = 0.0
                    if opp_net <= -PROGRESS_DMG_EPS:
                        return True
        return False

    @staticmethod
    def _denial_kind(delta) -> "Optional[str]":
        """Classify a non-progress window:
          * ``"exogenous"`` — a genuine attempt denied by RNG / the opponent (cant: para/sleep/freeze/
            flinch/focuspunch; accuracy miss; our ATTACK blocked by the opp's Protect/Detect — NOT our
            OWN failed Protect, which is a charged no-op). NEVER a stall → always frozen.
          * ``"heal"`` — a productive defensive heal move that restored real HP (Φ_mat prices the first
            few; a SUSTAINED run is a heal-war → charged past HEAL_FREEZE_GRACE).
          * ``None`` — a deliberate, obs-knowable wheel-spin → NO_OP (increment + charge)."""
        # Our chosen move was PREVENTED (cant) — exogenous.
        if getattr(delta, "our_failed_to_move", False):
            return "exogenous"
        outcome = getattr(delta, "our_move_outcome", None)
        # Accuracy MISS — the agent made a good attempt; RNG denied it.
        if outcome == "miss":
            return "exogenous"
        # Our ATTACK BLOCKED by the opponent's Protect/Detect/Endure (their choice denied our attempt)
        # is exogenous denial. An immune attack (|-immune|, our_effectiveness==0) is NOT denied — it's a
        # deterministic NO_OP. But a FAILED OWN STALL MOVE — our Protect/Detect/Endure lost its
        # escalating success roll — is NOT denial: it is a deliberate, obs-knowable no-progress
        # wheel-spin (a failed Protect IS a no-progress turn; the model should almost never go back-to-
        # back unless an our-owned residual makes it worth it, which the _is_progress chip clause above
        # already credits). So only freeze when OUR move was a non-stall attack that got blocked — never
        # on our own failed Protect, even on the rare turn the opponent ALSO stalled.
        if outcome == "fail":
            opp_mv = getattr(delta, "opp_resolved_move_id", None)
            our_mv = getattr(delta, "our_move_id", None)
            if opp_mv in _INVULNERABLE_MOVES and our_mv not in _INVULNERABLE_MOVES:
                return "exogenous"
        # Productive DEFENSIVE heal: a heal move that restored real HP (design §4.1.2).
        mid = getattr(delta, "our_move_id", None)
        if mid is not None:
            try:
                healed = float(delta.our_hp_delta.sum()) > PROGRESS_DMG_EPS
            except (AttributeError, TypeError):   # mock/standalone delta without a real hp array
                healed = False
            if healed:
                md = _movedex.get(mid)
                if md is not None and getattr(md, "is_heal", False):
                    return "heal"
        return None
