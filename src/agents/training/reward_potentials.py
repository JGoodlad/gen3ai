"""`RewardPotentials` — the PBRS potentials Φ, and the folds that telescope them.

Split out of `reward_manager.py` (2026-09-07) alongside `reward_config.py` and
`reward_bias_terms.py`. Two halves of one subject:

* **The potentials.** `_belief_potential_and_risk` and the seven `_compute_phi_*` — each a PURE
  function of the board (the once-per-turn `LiveView`), which is what makes PBRS policy-invariant.
* **The folds.** `_pbrs_step` (the `F = γ·Φ(s′) − Φ(s)` step with the absorbing `Φ(terminal)=0`
  convention, in ONE place so a new potential cannot forget it), `_hand_pbrs_on` (the gate, which
  delegates to the census's own predicate so the two cannot drift), and the eight `_fold_*_pbrs`
  that write one field each.

**What did NOT move: the ORDER.** `reward_manager.process_turn_reward` still calls these folds as
one straight line, exactly as `instrumented_ppo/ppo.py` keeps its minibatch fold sequence
(`ccd08003`). The per-term math lives here; the sequence lives there.

**A mixin, not a helper module** — the folds mutate `self._prev_phi_*`, `self._prev_active_ko_risk`
and `self._prev_safe_pivot`, and read `self.config` / `self.progress_clock` / `self._belief_memo`.
Mixed into `Gen3RewardManager`, so every attribute path is unchanged by the move.

⚠️ **`_encode_incoming_block` lives HERE now** (Φ_belief's one expensive input). A `patch(...)` of
it must name `agents.training.reward_potentials`; `src/test_stub_vacuity_gate_test.py` fails a
stale target rather than letting it stub nothing.
"""
from typing import Optional

from agents.observation.constants import TEAM_SIZE as _TEAM_SIZE
from agents.observation.incoming_damage_encoder import encode_block as _encode_incoming_block
# gen3_cpu_damage_deleted_v1: the incoming-damage block is no longer part of the OBSERVATION, so
# constants.py no longer re-exports its stride. The PBRS builds the block itself from the LiveView
# (`encode_block`) and reads it here, so it imports the stride from the module that OWNS the layout.
from agents.observation.incoming_damage import PER_MON as _INCOMING_PER_MON
# Named per-mon field offsets — the single source of truth for the incoming-damage block layout
# (incoming_damage.py). Read fields by NAME so a future field insert can't silently desync this
# PBRS read the way a hardcoded `block[base + 4]` once read phys_crit_delta as p_outspeed.
from agents.observation.incoming_damage import (
    IDX_PHYS_PKO as _IDX_PHYS_PKO,
    IDX_SPEC_PKO as _IDX_SPEC_PKO,
    IDX_OUTSPEED as _IDX_OUTSPEED,
)
from agents.training.reward_composition import _pbrs_term_active
from agents.training.reward_config import RewardBreakdown  # noqa: F401 - annotation target
from agents.training.reward_weights import (
    BOOST_WEIGHT, HAZARD_WEIGHT, MAT_HP_WEIGHT, OPP_BOOST_WEIGHT, PBRS_GAMMA,
    PBRS_RISK_WEIGHT, ROAR_BOOST_WEIGHT, SAFE_PIVOT_PKO_MAX, STATUS_TEMPO_WEIGHT,
    _TEMPO_STATUSES,
)


class RewardPotentials:
    """Mixin: the Φ computations and their PBRS folds. Mixed into `Gen3RewardManager`; holds no
    state of its own (the `_prev_phi_*` carry-overs are declared on the manager)."""

    def _belief_potential_and_risk(self, live) -> tuple[float, float, float]:
        """The incoming-KO belief potential Φ(s), the active mon's imminent KO-risk, and the safest
        bench mon's incoming P(KO) — all from the LiveView read-model (one belief encode, no raw
        battle).

        Φ = PBRS_RISK_WEIGHT · ( Σ_alive hp_frac  −  hp_active · risk_active ), where
        risk_active = max(phys_pko, spec_pko)·(1 − P(outspeed)) for the on-field mon. This is
        **expected surviving material**: a benched mon counts its full HP (it isn't hit this turn);
        the active mon is discounted by its imminent KO chance. So switching a doomed active to the
        bench RAISES Φ (its HP stops being discounted) → positive shaping AT the switch decision,
        while a faint never raises Φ (it removes a positive contribution). Φ is a pure function of
        the board state → policy-invariant under PBRS.

        Returns ``(phi, active_risk, min_bench_pko)``. ``active_risk`` is the outspeed-discounted
        KO-risk snapshot for the escape/stay re-gate. ``min_bench_pko`` is the LOWEST RAW incoming
        P(KO) across our non-fainted bench mons (1.0 if none) — raw, not outspeed-discounted, because
        a switch-in always eats the turn's hit (speed can't save it that turn). It feeds the
        safe-pivot gate of the belief-risk stay-tax (a stay is only taxed when a pivot would survive)."""
        block = _encode_incoming_block(live, memo=self._belief_memo)
        mons = live.ours.mons
        total_hp = 0.0
        active_imminent = 0.0
        active_risk = 0.0
        min_bench_pko = 1.0
        for i, lm in enumerate(mons[:_TEAM_SIZE]):
            if lm is None or lm.fainted:
                continue
            hp = float(lm.hp_fraction)
            total_hp += hp
            base = i * _INCOMING_PER_MON
            pko = max(float(block[base + _IDX_PHYS_PKO]), float(block[base + _IDX_SPEC_PKO]))
            if lm.active:
                outspeed = float(block[base + _IDX_OUTSPEED])
                active_risk = pko * (1.0 - outspeed)
                active_imminent = hp * active_risk
            elif hp > 0.0:   # a real, switchable bench mon (guard a 0-HP-not-yet-fainted false "safe")
                min_bench_pko = min(min_bench_pko, pko)
        phi = PBRS_RISK_WEIGHT * (total_hp - active_imminent)
        phi = max(0.0, min(phi, PBRS_RISK_WEIGHT * _TEAM_SIZE))
        return phi, active_risk, min_bench_pko

    def _compute_phi_mat(self, live) -> float:
        """The material potential Φ_mat(s) (design §2.2), from the LiveView read-model.

        Φ_mat = MAT_HP_WEIGHT·(Σ our_hp_frac − Σ opp_hp_frac) + MAT_ALIVE_WEIGHT·(n_alive_ours −
        n_alive_opp), summed over the **declared team size** — unrevealed opponent mons count as
        full-HP-alive. So Φ_mat(s_0)≈0 (6−6 HP, 6−6 alive), there are no opp-reveal discontinuities
        (the opp sum only ever decreases from a 6.0 baseline), and the per-episode telescoping
        constant −Φ_mat(s_0)≈0 with near-zero cross-episode variance. A pure function of the board
        (HP + alive set) → policy-invariant under PBRS.
        """
        our = live.ours
        opp = live.opp
        # No known mons → neutral (production always has all 6 from turn 1; this guards mock/standalone
        # paths where the team list is empty, so Φ_mat doesn't read a degenerate 0-vs-6 state).
        if not our.mons:
            self._last_material_margin = 0.0
            return 0.0
        # Our team is fully known: sum HP over the (≤6) known mons, alive = non-fainted.
        our_hp = sum(float(m.hp_fraction) for m in our.mons[:_TEAM_SIZE] if not m.fainted)
        our_alive = sum(1 for m in our.mons[:_TEAM_SIZE] if not m.fainted)
        # Opp: revealed mons carry their real HP/alive; unrevealed declared slots count full-HP-alive.
        opp_team_size = getattr(opp, "team_size", None) or _TEAM_SIZE
        opp_team_size = min(int(opp_team_size), _TEAM_SIZE)
        opp_revealed = list(opp.mons[:_TEAM_SIZE])
        opp_hp = sum(float(m.hp_fraction) for m in opp_revealed if not m.fainted)
        opp_alive = sum(1 for m in opp_revealed if not m.fainted)
        n_unrevealed = max(0, opp_team_size - len(opp_revealed))
        opp_hp += n_unrevealed * 1.0     # unrevealed → full HP
        opp_alive += n_unrevealed        # unrevealed → alive
        alive_w = self.config.mat_alive_weight
        phi = MAT_HP_WEIGHT * (our_hp - opp_hp) + alive_w * (our_alive - opp_alive)
        bound = MAT_HP_WEIGHT * _TEAM_SIZE + alive_w * _TEAM_SIZE
        clamped = max(-bound, min(phi, bound))
        # Pure by-product (does NOT affect the returned Φ_mat / the reward): the normalized material
        # margin ∈ [−1,1] the win-prob head's training metrics read via gen3_env's win_margin obs key.
        self._last_material_margin = clamped / bound if bound > 0 else 0.0
        return clamped

    def _compute_phi_status(self, live) -> float:
        """The non-damaging-tempo status potential Φ_status(s) (design §2.7), from the LiveView.

        Φ_status = STATUS_TEMPO_WEIGHT · (opp_tempo_statused − our_tempo_statused), counting
        non-fainted mons carrying par / slp / frz (the tempo statuses Φ_mat can't price — Toxic /
        burn / poison's value is the HP chip, already in Φ_mat). Unknown opp bench mons carry no
        status → contribute 0, so Φ_status(s_0)=0 (nobody statused at start) → telescopes to zero net
        with Φ_status(terminal)=0. Sign mirrors Φ_mat: an opponent we put to sleep RAISES Φ (good for
        us); us getting paralysed LOWERS it. A pure function of the board status set → policy-invariant.
        """
        our = live.ours
        if not our.mons:                       # mock/standalone guard (mirrors _compute_phi_mat)
            return 0.0
        our_tempo = sum(1 for m in our.mons[:_TEAM_SIZE]
                        if not m.fainted and m.status in _TEMPO_STATUSES)
        opp_tempo = sum(1 for m in live.opp.mons[:_TEAM_SIZE]
                        if not m.fainted and m.status in _TEMPO_STATUSES)
        return STATUS_TEMPO_WEIGHT * (opp_tempo - our_tempo)

    def _compute_phi_progress(self, live) -> float:
        """Anti-stall potential Φ_progress(s) = −no_progress_penalty·progress_clock.value().
        value()∈[0,1] is the log-saturated turns_since_progress already in obs vec[14]; Φ(s_0)≈0
        (n=0 at start), Φ(terminal)=0 via _pbrs_step. 0.0 when the clock is absent (inference)."""
        if self.progress_clock is None:
            return 0.0
        return -abs(self.config.no_progress_penalty) * float(self.progress_clock.value())

    def _compute_phi_hazard(self, live) -> float:
        """Hazard (spikes) potential Φ_hazard = HAZARD_WEIGHT·(opp_spike_layers − our_spike_layers)
        (design §2.6). Setting an opp layer raises Φ (+shaping); clearing one (Rapid Spin) lowers it;
        a Spikes at the 3-layer cap is ΔΦ=0 (dissolves futile_spikes). Pure side-condition function."""
        opp_layers = self._opp_spikes(live)
        our_sc = getattr(live.ours, "side_conditions", None) or {}
        our_layers = our_sc.get("spikes", 0)
        phi = HAZARD_WEIGHT * (opp_layers - our_layers)
        bound = HAZARD_WEIGHT * 6
        return max(-bound, min(phi, bound))

    def _compute_phi_boost(self, live) -> float:
        """Stored-boost potential Φ_boost = BOOST_WEIGHT·Σmax(0,boost_i)·active_hp_fraction over our
        active mon (LivePokemon.boosts holds only nonzero stages; negatives are excluded by max(0,·)).
        High HP → full value; low HP suppressed; setting at the ±6 cap → ΔΦ=0 (dissolves futile_setup);
        a faint drops Φ. The realized damage is already in Φ_mat; this prices the stored advantage."""
        our = live.ours
        if not our.mons:
            return 0.0
        active = our.active
        if active is None or active.fainted:
            return 0.0
        hp_frac = float(active.hp_fraction)
        if hp_frac <= 0.0:
            return 0.0
        boosts = active.boosts or {}
        positive = sum(max(0, int(b)) for b in boosts.values())
        phi = BOOST_WEIGHT * positive * hp_frac
        return max(0.0, min(phi, BOOST_WEIGHT * 6 * 7))

    @staticmethod
    def _opp_positive_boost_stages(live) -> int:
        """Σ max(0, stage) over the opp active's boosts — the ONE quantity Φ_opp_boosts and Φ_roar
        both reduce to; 0 with no opp mon on the field."""
        if not live.opp.active:
            return 0
        return sum(max(0, int(v)) for v in (live.opp.active.boosts or {}).values())

    def _compute_phi_opp_boosts(self, live, positive: Optional[int] = None) -> float:
        """Opponent-boost-disruption potential Φ_opp_boosts = −OPP_BOOST_WEIGHT·Σmax(0,b) over the opp
        active's positive boost stages. A successful Roar resets them → next window Φ rises (positive
        shaping scaled by the boosts cleared); a failed Roar leaves them → ΔΦ=0 (failed_roar dissolves).
        The forced-in mon's hazard chip is priced by Φ_mat, not here. Pure board function.

        `positive` is the Σ shared with Φ_roar; `None` recomputes it (direct calls stay 1-arg)."""
        if not live.opp.active:
            return 0.0
        if positive is None:
            positive = self._opp_positive_boost_stages(live)
        phi = -OPP_BOOST_WEIGHT * positive
        bound = OPP_BOOST_WEIGHT * 7
        return max(-bound, min(phi, bound))

    def _compute_phi_roar(self, live, positive: Optional[int] = None) -> float:
        """DEDICATED phaze-out-boosts potential Φ_roar = −ROAR_BOOST_WEIGHT·Σmax(0,b) over the opp active's
        positive boost stages (same state-potential shape as Φ_opp_boosts, its OWN weight). A successful
        Roar resets the opp active's stages → next window Φ rises by ROAR_BOOST_WEIGHT·(stages cleared) =
        the proportional roar-out-boosts payout; a failed roar leaves them → ΔΦ=0. Pure board function.

        `positive` is the shared Σ (see `_compute_phi_opp_boosts`); `None` recomputes it."""
        if not live.opp.active:
            return 0.0
        if positive is None:
            positive = self._opp_positive_boost_stages(live)
        bound = ROAR_BOOST_WEIGHT * 7
        return max(-bound, min(-ROAR_BOOST_WEIGHT * positive, bound))


    # =========================================================
    # PBRS / clock / bias-refund FOLDS — the per-class treatment process_turn_reward applies.
    # Kept as named one-purpose methods so the orchestrator reads as a phase sequence and the
    # telescoping math (the §2.3 dominant-win footgun) lives in ONE place.
    # =========================================================
    @staticmethod
    def _pbrs_step(prev: Optional[float], phi_next: float, is_terminal: bool) -> tuple[float, float]:
        """One PBRS shaping step: F = γ·Φ(s′) − Φ(s), with the absorbing ``Φ(terminal)=0`` convention
        and the standard ``prev is None`` first-window skip. Returns ``(shaped, new_prev)``. The
        terminal-zeroing lives HERE once, so a new potential can't forget it (design §2.3).

        ⚠️ DO NOT "optimize" this by carrying/telescoping Φ itself. RECOMPUTING Φ FROM THE
        ONCE-PER-TURN LIVEVIEW IS THE EXACTNESS GUARANTEE: policy-invariance needs Φ to be a pure
        function of the STATE, and a carried Φ drifts with float accumulation over 250 turns —
        making it a function of HISTORY, a silent objective change with nothing to fail. Nor is
        anything quadratic left: `_prev_phi_*` already carries Φ(s), so this is O(1) Φ evaluations
        per turn and the six cheap potentials are 8% of `process_turn_reward` COMBINED (measured
        2026-08-23). The one expensive Φ input, Φ_belief's `encode_block` (60%), got the OTHER
        safe answer instead — CONTENT-keyed memoization of a pure function, which is a function of
        state, not of history (`IncomingBeliefMemo`). That is the shape to reach for here."""
        phi = 0.0 if is_terminal else phi_next
        shaped = (PBRS_GAMMA * phi - prev) if prev is not None else 0.0
        return shaped, phi

    def _hand_pbrs_on(self, name: str) -> bool:
        """Is hand PBRS term `name` folded under this run's config? Delegates to
        ``_pbrs_term_active`` — the SAME predicate `reward_class_composition` censuses, so a run's
        advertised PBRS list and the terms it can actually emit are one declaration, not two."""
        return _pbrs_term_active(self.config, name)

    def _fold_material_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Material PBRS Φ_mat (design §2) → ``bd.pbrs_material``. Replaces the old unconditional
        hp/faint base spine; telescopes to −Φ_mat(s_0) → every win +30, every loss −30.
        OFF (--no-pbrs-material / --no-hand-shaping) → byte-identical: `_prev_phi_mat` stays None
        and the field stays 0.0, the same shape every other fold's early return takes.

        ⚠️ The gate wraps the EMITTED FIELD ONLY, never `_compute_phi_mat` — the same rule
        `_fold_belief_pbrs` states one method down, for the same reason and a stronger one. Φ_mat's
        by-product `_last_material_margin` is an OBSERVATION FEATURE (gen3_env's `win_margin` key,
        read by the win-prob head's closeness-stratified metrics and its `skill_vs_material` Brier
        skill score), and an obs feature must not depend on the REWARD COMPOSITION. It did, for the
        whole of the win-prob arm's life: this method early-returned under `--no-hand-shaping`, so
        `_compute_phi_mat` never ran and the margin was pinned at 0.0 — measured on a 6-alive-vs-2
        board, shaped +0.667 against the winprob composition's 0.0. Downstream, `|margin| < tau` was
        then always true (`contested_frac ≡ 1`, so the contested split selected nothing) and
        `P_mat ≡ 0.5`, so `skill_vs_material` scored the head against a coin flip. The compute is
        ~4 sums over ≤12 mons (measured: +0.002 ms/decision, ~9% of the terminal-only fast path's
        0.023 ms, ~1.3% of the shaped composition's), so there is no margin-only fast path — the
        full potential IS the cheap path, and computing it unconditionally is what the obs is
        entitled to (gen3_obs_margin_unconditional_v1)."""
        phi_mat_next = self._compute_phi_mat(live)      # publishes `_last_material_margin` (obs)
        if not self._hand_pbrs_on("pbrs_material"):
            return
        bd.pbrs_material, self._prev_phi_mat = self._pbrs_step(
            self._prev_phi_mat, phi_mat_next, is_terminal)

    def _fold_belief_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Incoming-KO belief PBRS (design_reward_switching.md) → ``bd.pbrs_belief``; also snapshots
        the active KO-risk AND whether a safe bench pivot exists for next turn's escape/stay re-gate
        (both keyed at decision time — read by `_compute_stay_risk_tax` / `_apply_switch_outcome`).

        ⚠️ The gate wraps the EMITTED FIELD ONLY, never the two snapshots. They are cross-turn
        MUTATIONS feeding BIAS terms that `--no-pbrs-belief` alone leaves live, and this manager's
        standing rule is that a gate skips the COMPUTE, never a mutation — a `return` above the
        snapshots would silently re-price `stay_risk_tax` / `escape_threat_switch`."""
        phi_belief_next, active_risk_next, min_bench_pko_next = self._belief_potential_and_risk(live)
        if self._hand_pbrs_on("pbrs_belief"):
            bd.pbrs_belief, self._prev_phi_belief = self._pbrs_step(
                self._prev_phi_belief, phi_belief_next, is_terminal)
        self._prev_active_ko_risk = 0.0 if is_terminal else active_risk_next
        self._prev_safe_pivot = (not is_terminal) and (min_bench_pko_next <= SAFE_PIVOT_PKO_MAX)

    def _fold_status_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Non-damaging-tempo status PBRS Φ_status (design §2.7 / §7.4 hedge) → ``bd.pbrs_status``.
        Gated on ``bias_redesign``: in the default run the count-diff status BIAS already pays the
        standing value, so folding Φ_status there would double-count (and ``pbrs_status`` stays 0,
        ``_prev_phi_status`` stays None → byte-identical default). Under the event-form redesign the
        standing value was dropped — this restores it as a telescoping, policy-invariant potential.
        Also active under --all-shaping-pbrs (which suppresses the count-diff status BIAS, so Φ_status is
        the only remaining carrier of the tempo-status standing value — no double-count)."""
        if not self._hand_pbrs_on("pbrs_status"):
            return
        bd.pbrs_status, self._prev_phi_status = self._pbrs_step(
            self._prev_phi_status, self._compute_phi_status(live), is_terminal)

    def _fold_progress_pbrs(self, bd: "RewardBreakdown", is_terminal: bool) -> None:
        """Anti-stall PBRS Φ_progress → bd.pbrs_progress. Gated on --stall-pbrs (the "stall" switch; the
        no_progress_tax BIAS otherwise carries the charge). Telescopes via _pbrs_step, Φ(terminal)=0."""
        if not self._hand_pbrs_on("pbrs_progress"):
            return
        bd.pbrs_progress, self._prev_phi_progress = self._pbrs_step(
            self._prev_phi_progress, self._compute_phi_progress(None), is_terminal)

    def _fold_hazard_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Hazard PBRS Φ_hazard (design §2.6) → bd.pbrs_hazard. Gated on all_shaping_pbrs (the
        spikes/futile_spikes BIAS terms are suppressed there). Telescopes, Φ(terminal)=0."""
        if not self._hand_pbrs_on("pbrs_hazard"):
            return
        bd.pbrs_hazard, self._prev_phi_hazard = self._pbrs_step(
            self._prev_phi_hazard, self._compute_phi_hazard(live), is_terminal)

    def _fold_boost_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Stored-boost PBRS Φ_boost → bd.pbrs_boost. Gated on all_shaping_pbrs (boost_utilized/
        futile_setup/setup_low_hp suppressed there). Telescopes, Φ(terminal)=0."""
        if not self._hand_pbrs_on("pbrs_boost"):
            return
        bd.pbrs_boost, self._prev_phi_boost = self._pbrs_step(
            self._prev_phi_boost, self._compute_phi_boost(live), is_terminal)

    def _fold_opp_boosts_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool,
                              positive: Optional[int] = None) -> None:
        """Opp-boost-disruption PBRS Φ_opp_boosts → bd.pbrs_opp_boosts. Gated on all_shaping_pbrs (the
        roar/failed_roar BIAS is suppressed there). Telescopes, Φ(terminal)=0.
        `positive` is the opp-positive-boost Σ shared with `_fold_roar_pbrs` (None recomputes)."""
        if not self._hand_pbrs_on("pbrs_opp_boosts"):
            return
        bd.pbrs_opp_boosts, self._prev_phi_opp_boosts = self._pbrs_step(
            self._prev_phi_opp_boosts, self._compute_phi_opp_boosts(live, positive), is_terminal)

    def _fold_roar_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool,
                        positive: Optional[int] = None) -> None:
        """DEDICATED phaze-out-boosts PBRS Φ_roar → bd.pbrs_roar. Folded INTO --all-shaping-pbrs (its own
        flag was removed per owner request), so it rides alongside Φ_opp_boosts there — the two STACK (safe,
        both policy-invariant) for a stronger proportional roar-out-boosts shaping. Telescopes via
        _pbrs_step, Φ(terminal)=0. OFF (no all_shaping_pbrs) → byte-identical (prev stays None, field 0.0)."""
        if not self._hand_pbrs_on("pbrs_roar"):
            return
        bd.pbrs_roar, self._prev_phi_roar = self._pbrs_step(
            self._prev_phi_roar, self._compute_phi_roar(live, positive), is_terminal)

