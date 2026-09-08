"""`RewardBiasTerms` — the per-term computations for the BIAS class.

Split out of `reward_manager.py` (2026-09-07) alongside `reward_config.py` and
`reward_potentials.py`. Every method here computes ONE additive BIAS field of the
`RewardBreakdown` from `(delta, live)` and returns it; not one of them assigns to the breakdown,
reads a flag gate, or knows where in the turn it is called. That is the seam: the ORDER these are
folded in, and the gates that suppress them, stay in `reward_manager.process_turn_reward` as one
straight line — the same rule `instrumented_ppo/ppo.py` keeps for its minibatch fold sequence
(`ccd08003`). A term's arithmetic moving here changes nothing about when it runs.

**A mixin, not a helper module, and deliberately so.** These read cross-turn manager state
(`self._prev_opp_spikes`, `self._consecutive_dead_matchup_stays`, `self.config`) and the
current-board accessors (`self._opp_spikes`, `self._live_eff_mult`) that `reward_manager` owns.
Passing all of that explicitly would turn a move into a rewrite, and a rewrite is exactly what a
byte-identity change cannot afford. `Gen3RewardManager` mixes this in, so every `self.` reference
resolves as it always did and every attribute path is unchanged.

⚠️ **A patch target that names a symbol here must name THIS module.** The stub-vacuity gate
(`src/test_stub_vacuity_gate_test.py`) exists because `ccd08003`'s decomposition left four stubs
pointing at a module that no longer read the symbol; it will fail a stale target rather than let
it pass vacuously.
"""
from typing import Optional

from agents.enums import PokemonType as _PokemonType, Status
from agents.gen3_data import moves as _movedex
from agents.gen3_mechanics import (
    INVULNERABLE_MOVES as _INVULNERABLE_MOVES,
    effective_multiplier_by_types as _effective_multiplier_by_types_fn,
    STATUS_MOVE_IMMUNITY as _STATUS_MOVE_IMMUNITY,
)
from agents.training.reward_config import RewardBreakdown  # noqa: F401 - annotation target
from agents.training.turn_delta import SELF_KO_MOVES, TurnDelta
from agents.training.reward_weights import (
    BOOST_MOVES, BOOST_UTILIZED_SCALE, DEAD_MATCHUP_TAX_FLOOR, DEAD_MATCHUP_TAX_STEP,
    FAILED_ROAR_PENALTY, FINISHING_BLOW_BONUS, FUTILE_ATTACK_PENALTY,
    FUTILE_IMMUNE_PENALTY, FUTILE_SETUP_PENALTY, MATCHUP_PENALTY, PROTECT_SWITCH_BONUS,
    ROAR_BONUS, SE_SWITCH_BONUS, SETUP_LOW_HP_MAX_PENALTY,
    SETUP_LOW_HP_THRESHOLD, SLEEP_SWAP_BONUS, SPIKES_LAYER_BONUS, SPIKES_WASTE_PENALTY,
    STATUS_BONUS, STATUS_IMMUNE_SWITCH_BONUS, STATUS_INFLICTING_MOVES, STATUS_WASTED_PENALTY,
    STAY_RISK_TAX_FLOOR, SWITCH_RISK_THRESHOLD,
)


def _ptype(name) -> "Optional[_PokemonType]":
    """LiveView type-id string (e.g. ``'fire'``) -> ``PokemonType`` enum — the primitive
    the mechanics helpers key on. ``None`` passes through."""
    return _PokemonType[name.upper()] if name else None


def _status_enum(name) -> "Optional[Status]":
    """LiveView status-id string (e.g. ``'slp'``) -> ``Status`` enum. ``None`` passes
    through. Only ``FRZ`` actually changes an effectiveness result (Flash Fire), but we
    convert faithfully so the LiveView path is byte-identical to the raw-battle path."""
    return Status[name.upper()] if name else None


class RewardBiasTerms:
    """Mixin: every `_compute_*` that produces an additive BIAS field. Mixed into
    `Gen3RewardManager`; holds no state of its own."""

    def _compute_roar_bonus(self, delta: TurnDelta, live) -> float:
        """Reward Roar when it forces a switch AND spikes are up or opp had positive boosts.
        Penalise Roar when it fails to force any switch at all (wasted turn).

        Markovian note (design §3 #9 — resolved, no reframe needed): `_prev_opp_boosts` is the opp's
        boosts as of THIS decision (= the opp active-context boosts the model saw in its obs when it
        pressed Roar) — the very boosts the Roar then phazed away. So the term is already
        Markovian-recoverable. There is no cleaner obs source at reward time: by the time the reward
        runs the opp has been phazed and `live` shows the reset board, so the relevant pre-Roar boosts
        live only in the decision-time snapshot. Same value either way → kept unchanged."""
        if delta.our_move_id != "roar":
            return 0.0
        if delta.opp_switch_to is None:
            return FAILED_ROAR_PENALTY
        has_spikes = self._opp_spikes(live) > 0
        had_boosts = any(v > 0 for v in self._prev_opp_boosts.values())
        return ROAR_BONUS if (has_spikes or had_boosts) else 0.0

    def _compute_se_switch_bonus(self, delta: TurnDelta, live) -> float:
        """Reward switching in a mon that threatens the opponent with a SE move.

        First checks revealed moves for confirmed SE; if none are revealed yet,
        falls back to checking whether any of our mon's own types are SE vs the
        opponent (a reliable proxy for STAB moves in Gen 3 OU). Reads current-board
        state through the LiveView ``LivePokemon`` (move power/type via ``gen3_movedex``,
        effectiveness via the mechanics primitive).
        """
        if delta.our_switch_to is None:
            return 0.0
        our_mon = live.ours.active
        opp_mon = live.opp.active
        if not our_mon or not opp_mon:
            return 0.0

        # Only award on voluntary switches; forced post-faint replacements don't count
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        # Opponent must be alive at switch-in
        if opp_mon.fainted:
            return 0.0

        # Gate: only fire if the opponent has switched since this mon was last in (same matchup
        # without an opp switch = bonus already spent). The redesign DROPS this gate (design §3 #25):
        # it keys on the hidden `_last_opp_seen_by` map; the SE-threat fact is in the matchup obs, so
        # the bonus is a clean per-(s,a) bias. The default run keeps the gate (avoids re-paying it).
        our_species = our_mon.species
        opp_species = opp_mon.species
        if not self.config.bias_redesign and self._last_opp_seen_by.get(our_species) == opp_species:
            return 0.0

        # Confirmed SE via revealed move
        for mid in our_mon.move_ids:
            md = _movedex.get(mid)
            if md is None or md.base_power <= 0:
                continue
            if self._live_eff_mult(md.type, opp_mon) >= 2.0:
                return SE_SWITCH_BONUS
        # Fallback: STAB type advantage (no moves revealed yet)
        for t in our_mon.types:
            if self._live_eff_mult(_ptype(t), opp_mon) >= 2.0:
                return SE_SWITCH_BONUS
        return 0.0

    def _compute_status_reward(self, delta: TurnDelta, live) -> tuple[float, int]:
        """One-time reward when a status changes on either side. Returns (reward, d_opp) where
        ``d_opp > 0`` marks "opp GAINED a status this window" (for the status_wasted check).

        Two forms, selected by ``bias_redesign``:
          * default  — the count diff vs the hidden ``_prev_*_statused`` snapshot (today's behavior).
          * redesign — the Markovian reframe (design §3 #29): key on the per-window TRANSITION EVENTS
            (``status_applied`` / ``status_cured``, folded by TurnDelta and present in the obs history)
            rather than a stored prev-count. ``+`` on a status landing on opp / a self-cure; ``−`` on
            us being statused / the opp curing (e.g. Rest), with no dependence on internal state."""
        our_statused, opp_statused = self._statused_counts(live)
        if self.config.bias_redesign:
            d_our = (1 if delta.our_status_applied else 0) - (1 if delta.our_status_cured else 0)
            d_opp = (1 if delta.opp_status_applied else 0) - (1 if delta.opp_status_cured else 0)
        else:
            d_our = our_statused - self._prev_our_statused
            d_opp = opp_statused - self._prev_opp_statused
        self._prev_our_statused = our_statused      # kept current for the (resume-immutable) default path
        self._prev_opp_statused = opp_statused
        return (d_opp - d_our) * STATUS_BONUS, d_opp

    # =========================================================
    # SWITCH REWARDS
    #
    # All switch-specific signals funnel through
    # _compute_all_switch_bonuses, which dispatches to one
    # focused sub-function per outcome type.  The subsidy set
    # by record_action() is applied separately at the end of
    # process_turn_reward (it runs before the turn, not after).
    # =========================================================

    def _compute_pivot_bonus(self, delta: TurnDelta, live) -> tuple[float, float, float]:
        """Return (protect_bonus, status_bonus, damage_bonus) for this switch turn.

        Uses `delta.opp_resolved_move_id` — protocol-truth attribution when a
        damaging event is set, falling back to the inferred `delta.opp_move_id`
        for non-damaging moves (status, Roar, etc.). Avoids the stale-last_move
        class of bug that bit HP attribution. The opp-move presence + power read uses
        the LiveView active mon's revealed moves + ``gen3_movedex``.
        """
        opp_move_id = delta.opp_resolved_move_id
        if delta.opp_switch_to is not None or opp_move_id is None:
            return (0.0, 0.0, 0.0)

        if opp_move_id in _INVULNERABLE_MOVES:
            return (self._pivot_protect_bonus(), 0.0, 0.0)

        opp_mon = live.opp.active
        if not opp_mon or opp_move_id not in opp_mon.move_ids:
            return (0.0, 0.0, 0.0)
        md = _movedex.get(opp_move_id)
        base_power = md.base_power if md is not None else 0

        if base_power == 0:
            return (0.0, self._pivot_status_bonus(opp_move_id, live), 0.0)
        return (0.0, 0.0, self._pivot_damage_bonus(opp_move_id, delta, live))

    def _pivot_protect_bonus(self) -> float:
        """Opponent used Protect/Detect/Endure — we repositioned for free."""
        return PROTECT_SWITCH_BONUS

    def _pivot_status_bonus(self, opp_move_id: str, live) -> float:
        """Opponent used a status move our switch-in was immune to (type or already statused)."""
        new_mon = live.ours.active
        if not new_mon:
            return 0.0
        if self._live_status_move_immune(opp_move_id, new_mon):
            return STATUS_IMMUNE_SWITCH_BONUS
        return 0.0

    def _pivot_damage_bonus(self, opp_move_id, delta: TurnDelta, live) -> float:
        """Opponent used a damaging move — bonus if it hit our new mon less than the old one.

        Signal A: comparison of actual type effectiveness vs old mon vs new mon.
        The raw HP delta is already penalised by hp_ours, so this signal focuses purely
        on whether the switch improved the matchup.

        `mult_vs_new` prefers `delta.opp_damaging_event.effectiveness` when the
        event's target matches our switch-in — it's the protocol-confirmed bucket
        (no drift if our local mechanics disagree with Showdown's). Falls back
        to a local recompute when the event is missing or hit a different
        target (e.g. opp damaged prev_active before switch-in was on field).
        `mult_vs_old` always recomputes — the protocol can't tell us what the
        multiplier *would have been* if we hadn't switched. The move type comes from
        ``gen3_movedex`` and the prev/new mons from the LiveView.
        """
        new_mon = live.ours.active
        if not new_mon:
            return 0.0
        prev_mon = live.ours.get(delta.our_prev_active)
        if prev_mon is None:
            return 0.0
        md = _movedex.get(opp_move_id)
        move_type = md.type if md is not None else None
        opp_event = delta.opp_damaging_event
        if opp_event is not None and opp_event.target_species == new_mon.species:
            mult_vs_new = opp_event.effectiveness
        else:
            mult_vs_new = self._live_eff_mult(move_type, new_mon)
        mult_vs_old = self._live_eff_mult(move_type, prev_mon)
        if mult_vs_new < mult_vs_old:
            return 0.15 if mult_vs_new == 0 else 0.10
        return 0.0

    def _compute_sleep_out_bonus(self, delta: TurnDelta, live) -> float:
        """Reward rotating a sleeping mon to the bench on a voluntary switch.
        Preserving a sleeping mon's PP/position has strategic value; post-faint
        replacements don't qualify since there's no choice involved."""
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        prev = live.ours.get(delta.our_prev_active)
        if prev is None:
            return 0.0
        return SLEEP_SWAP_BONUS if prev.status == "slp" else 0.0

    def _compute_sleep_in_penalty(self, delta: TurnDelta, live) -> float:
        """Penalise sending in a sleeping mon — it can't act and wastes a slot.
        Applies to voluntary switches and post-faint replacements; skipped for roar
        since the phazer chose our slot, not us."""
        if self._last_switch_was_roared:
            return 0.0
        our_mon = live.ours.active
        if our_mon and our_mon.status == "slp":
            return -SLEEP_SWAP_BONUS
        return 0.0

    def _compute_matchup_penalty(self, delta: TurnDelta) -> float:
        """Per-turn penalty for staying in while the opp threatened us last turn — a revealed SE
        move OR the incoming-KO belief (P(KO)·(1-outspeed) ≥ threshold). Uses last-turn's snapshot
        so we only penalise for threats known at decision time. The belief OR-gate lights this up
        for the damage-magnitude / unrevealed / prior-based threats the revealed-SE gate misses."""
        if delta.our_switch_to is not None:
            return 0.0  # we switched out — no staying-in penalty
        threatened = self._prev_opp_se_threat or self._prev_active_ko_risk >= SWITCH_RISK_THRESHOLD
        return MATCHUP_PENALTY if threatened else 0.0

    def _compute_stay_risk_tax(self, delta: TurnDelta) -> float:
        """Belief-risk-scaled penalty for STAYING into a high imminent-KO spot when a safe pivot was
        available (the under-switch lever; design_reward_switching.md §7). OFF unless
        ``switch_bias_weight > 0``. Keyed entirely on DECISION-TIME snapshots
        (``_prev_active_ko_risk`` / ``_prev_safe_pivot``, set at the end of last turn from the board
        the policy then acted on), so it prices the choice the policy actually made.

        Fires only when ALL hold: a switch was LEGAL this decision (``_cur_can_switch`` — never tax a
        TRAPPED stay, the key false-positive guard); we did NOT switch (``our_switch_to is None``); the
        move did NOT fizzle to RNG (``not our_failed_to_move`` — flinch / full-para / sleep / freeze is
        not a deliberate stay, mirroring the progress-clock's FREEZE); the chosen move did NOT KO the
        opponent (``not opp_fainted`` — then staying won the exchange, not a misplay); the decision-time
        KO-risk was high (``>= SWITCH_RISK_THRESHOLD``); and a safe bench pivot existed
        (``_prev_safe_pivot``). Staying-and-fainting IS included (the exact pathology). Tax scales with
        the risk, clamped at ``STAY_RISK_TAX_FLOOR``. Unlike ``pbrs_belief`` (policy-invariant) this is
        an additive BIAS → it changes the objective the actor optimises, which is the point."""
        w = self.config.switch_bias_weight
        if w <= 0.0:
            return 0.0
        if (delta.our_switch_to is not None or delta.opp_fainted
                or delta.our_failed_to_move or not self._cur_can_switch):
            return 0.0
        if self._prev_active_ko_risk < SWITCH_RISK_THRESHOLD or not self._prev_safe_pivot:
            return 0.0
        return max(-w * self._prev_active_ko_risk, STAY_RISK_TAX_FLOOR)

    def _compute_dead_matchup_tax(self, delta: TurnDelta, live) -> float:
        """Escalating penalty for refusing to pivot out of a 0×-only matchup.

        Fires when EVERY damaging move our active Pokémon has does 0× to the
        opponent's active mon (e.g. an Electric attacker staring at a Ground type,
        a Normal attacker into a Ghost) and we chose to stay in rather than switch.
        The per-turn cost grows with how many consecutive turns we've stayed
        trapped, so a switch — which resets the counter to zero — strictly
        dominates clicking another useless move.

        Resets (and charges nothing) whenever we switch, faint, lack a live
        opponent, or have at least one >0× damaging move. Skips forced-switch
        slots entirely (we had no move choice there). Requires at least one
        revealed damaging move to judge — our own active mon's full moveset is
        populated from the request, so this is reliable for the trainee's mon.
        """
        if delta.our_switch_to is not None or delta.we_fainted:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        if delta.phase_is_forced_switch:
            return 0.0

        our_mon = live.ours.active
        opp_mon = live.opp.active
        if not our_mon or not opp_mon or opp_mon.fainted:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        damaging = [
            md for md in (_movedex.get(mid) for mid in our_mon.move_ids)
            if md is not None and md.base_power > 0
        ]
        if not damaging:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        best_mult = max(self._live_eff_mult(md.type, opp_mon) for md in damaging)
        if best_mult > 0.0:
            self._consecutive_dead_matchup_stays = 0
            return 0.0

        # Every damaging option is type-immune and we stayed in — escalate.
        self._consecutive_dead_matchup_stays += 1
        return max(DEAD_MATCHUP_TAX_STEP * self._consecutive_dead_matchup_stays,
                   DEAD_MATCHUP_TAX_FLOOR)


    def _compute_spikes_bonus(self, delta: TurnDelta, live) -> tuple[float, float]:
        """Return (layer_bonus, futile_waste). The layer-added credit is the BIAS term `spikes`
        (its telescoping form is Φ_hazard, reached at bias_additivity→0 — design §2.6); the
        wasted-Spikes-at-cap penalty is split out into the Markovian `futile_spikes` term."""
        curr = self._opp_spikes(live)
        new_layers = curr - self._prev_opp_spikes
        self._prev_opp_spikes = curr
        if new_layers > 0:
            return new_layers * SPIKES_LAYER_BONUS, 0.0
        if delta.our_move_id == "spikes" and curr == 3:
            return 0.0, SPIKES_WASTE_PENALTY
        return 0.0, 0.0

    def _compute_futile_attack_penalty(self, delta: TurnDelta, live) -> float:
        """Penalise attacking moves where the opponent's total HP went up or stayed even
        (Leftovers healed as much or more than we dealt). Skips status moves, switches,
        and cases where we failed to act or the opponent used Rest."""
        if delta.our_move_id is None:
            return 0.0  # we switched
        if delta.our_failed_to_move:
            return 0.0  # paralysis / sleep — not our fault
        if delta.opp_switch_to is not None:
            return 0.0  # they switched; HP delta is noisy (fresh mon entering)
        # Rest detection uses opp_resolved_move_id (protocol-truth when an
        # event fired). Raw delta.opp_move_id could be a stale "rest" from
        # an earlier turn after opp switched between snapshots, mis-skipping
        # the penalty on a normal turn where they didn't actually rest.
        if delta.opp_resolved_move_id == "rest":
            return 0.0  # opponent used Rest; large self-heal is expected
        # Damaging-move gate: our move must be a revealed damaging move (power via gen3_movedex).
        our_mon = live.ours.active
        md = (_movedex.get(delta.our_move_id)
              if our_mon and delta.our_move_id in our_mon.move_ids else None)
        if md is None or md.base_power == 0:
            return 0.0  # status or utility move — handled by other signals
        # Type immunity: 0 damage by definition — use the harder penalty.
        if delta.our_effectiveness == 0.0:
            return FUTILE_IMMUNE_PENALTY
        # Net HP sum across all opp slots: bench mons don't change between turns in Gen 3,
        # so the sum is dominated by the active slot. >= 0 means we made no net progress.
        if delta.opp_hp_delta.sum() >= 0:
            return FUTILE_ATTACK_PENALTY
        return 0.0

    def _compute_futile_setup_penalty(self, delta: TurnDelta) -> float:
        """Penalise using a stat-boosting move when already at the ±6 cap."""
        if delta.our_move_id not in BOOST_MOVES:
            return 0.0
        if delta.our_failed_to_move or delta.we_fainted:
            return 0.0
        # If no boost stage changed, the move had zero mechanical effect
        if delta.our_boost_delta.sum() == 0:
            return FUTILE_SETUP_PENALTY
        return 0.0

    def _compute_setup_low_hp_penalty(self, delta: TurnDelta) -> float:
        """Penalise choosing a setup move below 40% HP."""
        if delta.our_move_id not in BOOST_MOVES:
            return 0.0
        if delta.our_failed_to_move or delta.we_fainted:
            return 0.0
        hp = self._our_active_hp_before
        if hp >= SETUP_LOW_HP_THRESHOLD:
            return 0.0
        return SETUP_LOW_HP_MAX_PENALTY * (1.0 - hp / SETUP_LOW_HP_THRESHOLD)

    def _compute_status_wasted_penalty(self, delta: TurnDelta, d_opp_statused: int) -> float:
        """Penalise status-inflicting moves that produced no status event."""
        if delta.our_move_id not in STATUS_INFLICTING_MOVES:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        if delta.opp_switch_to is not None:
            return 0.0  # opp switched; ambiguous
        if d_opp_statused > 0:
            return 0.0  # status landed — no penalty
        return STATUS_WASTED_PENALTY

    def _compute_boost_utilized(self, delta: TurnDelta, live) -> float:
        """Reward attacking moves that leverage active stat boosts."""
        if delta.our_move_id is None or delta.our_switch_to is not None:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        mon = live.ours.active
        md = (_movedex.get(delta.our_move_id)
              if mon and delta.our_move_id in mon.move_ids else None)
        if md is None or md.base_power == 0:
            return 0.0
        # Use the higher of atk (idx 0) or spa (idx 2) boost
        effective_boost = max(int(self._our_boosts_before[0]), int(self._our_boosts_before[2]))
        if effective_boost <= 0:
            return 0.0
        damage_dealt = max(0.0, -float(delta.opp_hp_delta.sum()))
        return effective_boost * BOOST_UTILIZED_SCALE * damage_dealt

    def _compute_finishing_blow_bonus(self, delta: TurnDelta, live) -> float:
        """Extra bonus when a damaging move secures the KO.

        Suppressed when our own mon faints the same turn (Explosion / Self-Destruct,
        or any mutual KO). A kill that costs us our own mon is not a clean finishing
        blow: without this guard a 1-for-1 *healthy* Explosion trade nets POSITIVE,
        because the symmetric hp_ours/hp_opp and faint_ours/faint_opp terms cancel
        and the +0.5 tips it over — teaching the policy to use Explosion as a free
        KO button. The trade-cost side is handled by the faint_ours material penalty.
        """
        if not delta.opp_fainted:
            return 0.0
        if delta.we_fainted:
            return 0.0
        if delta.our_move_id is None or delta.our_switch_to is not None:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        mon = live.ours.active
        md = (_movedex.get(delta.our_move_id)
              if mon and delta.our_move_id in mon.move_ids else None)
        if md is None or md.base_power == 0:
            return 0.0
        return FINISHING_BLOW_BONUS

    def _compute_self_ko_penalty(self, delta: TurnDelta) -> float:
        """HP-scaled penalty for throwing away a mon via a self-KO move (Explosion / Self-Destruct).

        The symmetric material PBRS prices a 1-for-1 *healthy* self-KO trade at ~0, so the critic
        learns to value a full-HP self-KO POSITIVELY (measured dV ≈ +2.9 → PPO advantage ≈ +1.5 on
        ≥80%-HP self-KOs) and the policy explodes healthy mons (turn-1 full-HP Metagross). The penalty
        ``−w · (our active HP fraction at decision time)`` restores a negative signal the critic can't
        smear, while scaling by HP spares the legitimate low-HP "explode a dying mon to secure a KO"
        play (≈0 penalty). OFF (w=0.0) by default → byte-unchanged.

        Gated on ``we_fainted`` (the self-KO actually cost us the mon — a blocked/failed Explosion that
        survives loses nothing) and ``not our_failed_to_move`` (we executed the self-KO, weren't merely
        KO'd first while holding the move). ``_our_active_hp_before`` is the decision-time HP snapshot
        set in ``record_action`` (the value squandered)."""
        w = self.config.self_ko_hp_penalty
        if (w <= 0.0 or not delta.we_fainted or delta.our_failed_to_move
                or delta.our_move_id not in SELF_KO_MOVES):
            return 0.0
        hp_frac = min(1.0, max(0.0, self._our_active_hp_before))
        return -w * hp_frac

    # =========================================================
    # CURRENT-BOARD ACCESSORS
    #
    # These read current-board facts through the vetted LiveView
    # read-model (built once per turn in process_turn_reward) so
    # the reward manager never reaches into the raw poke-env battle
    # for "what is true now".
    # =========================================================

    def _opp_spikes(self, live) -> int:
        """Spikes layers on the opponent's side (0-3)."""
        return live.opp.side_conditions.get("spikes", 0)

    def _statused_counts(self, live) -> tuple[int, int]:
        """(our, opp) counts of non-fainted mons carrying a status condition."""
        our = sum(1 for m in live.ours.mons if m.status is not None and not m.fainted)
        opp = sum(1 for m in live.opp.mons if m.status is not None and not m.fainted)
        return our, opp

    def _opp_active_boosts(self, live) -> dict:
        """Opponent active mon's current stat-stage boosts ({} if none on field)."""
        return dict(live.opp.active.boosts) if live.opp.active else {}

    def _terminal(self, live) -> tuple:
        """(won, lost, finished) — terminal battle flags."""
        return live.won, live.lost, live.finished

    def _live_eff_mult(self, move_type, live_mon) -> float:
        """``effective_multiplier`` for a LiveView ``LivePokemon``, via the primitive
        ``effective_multiplier_by_types`` (no poke-env ``Pokemon``). Byte-identical to
        the raw-battle path: the LivePokemon carries the same types/ability/status the
        raw ``mon`` would expose, just in id-string form."""
        types = live_mon.types
        t1 = _ptype(types[0]) if types else None
        t2 = _ptype(types[1]) if len(types) > 1 else None
        return _effective_multiplier_by_types_fn(
            move_type, t1, t2, live_mon.ability, _status_enum(live_mon.status)
        )

    def _live_status_move_immune(self, move_id, live_mon) -> bool:
        """``is_status_move_immune`` for a LiveView ``LivePokemon`` — type immunity to
        the status the move inflicts, or the mon already carrying a status."""
        immune_types = _STATUS_MOVE_IMMUNITY.get(move_id, frozenset())
        mon_types = {_ptype(t) for t in live_mon.types}
        return bool(immune_types & mon_types) or live_mon.status is not None

    def _update_opp_se_threat(self, live) -> None:
        """Snapshot whether opp active has a revealed SE move vs our active, for next turn."""
        our_mon = live.ours.active
        opp_mon = live.opp.active
        if not our_mon or not opp_mon:
            self._prev_opp_se_threat = False
            return
        for mid in opp_mon.move_ids:
            md = _movedex.get(mid)
            if md is not None and md.base_power > 0:
                if self._live_eff_mult(md.type, our_mon) >= 2.0:
                    self._prev_opp_se_threat = True
                    return
        self._prev_opp_se_threat = False
