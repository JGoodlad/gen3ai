"""The ProgressClock predicate (design_markovian_reward_and_features.md §4): the ternary
PROGRESS / DENIED / NO_OP classification and the gen3_setup_progress_v1 setup-progress rule.
Pure (no battle sim) — the shared fakes live in `reward_test_fakes.py`.
"""
import unittest

import numpy as np

from agents.training.progress_clock import ProgressClock, PROGRESS_CLOCK_CAP, HEAL_FREEZE_GRACE
from agents.training.reward_test_fakes import _Live, _delta, _Legal, _full_team_live


class TestProgressClock(unittest.TestCase):
    """The ternary PROGRESS / DENIED / NO_OP predicate (design §4)."""

    def _clock(self):
        return ProgressClock()

    def test_value_scalar_log_saturated(self):
        c = self._clock()
        self.assertEqual(c.value(), 0.0)            # n=0 → 0
        c.n = PROGRESS_CLOCK_CAP
        self.assertAlmostEqual(c.value(), 1.0, places=6)  # n=cap → 1
        c.n = 1
        self.assertGreater(c.value(), 0.0)
        self.assertLess(c.value(), 1.0)

    def test_our_damage_resets(self):
        c = self._clock()
        c.n = 5
        ev = object()   # our_damaging_event present
        c.update(_delta(our_damaging_event=ev, opp_target_hp_delta=-0.25),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 0)
        self.assertEqual(c.last_penalty, 0.0)

    def test_passive_chip_does_not_reset(self):
        """Sandstorm/Leech-only damage (no our_damaging_event) must NOT reset (our-attributed only)."""
        c = self._clock()
        c.n = 3
        # opp lost HP via passive chip, but OUR move didn't connect (no damaging event).
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625
        c.update(_delta(our_move_id="protect", opp_hp_delta=opp_hp),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 4)            # incremented (a no-op)
        self.assertEqual(c.last_penalty, -0.15)

    def test_failed_protect_charges_as_noop(self):
        """A FAILED Protect (lost its escalating success roll, opp then attacked) is a no-progress
        turn → increment + charge. The user's rule: 'it should be a no-progress turn when protect
        fails' (one should almost never go back-to-back unless it's worth it)."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="protect", our_move_outcome="fail", opp_resolved_move_id="rockslide"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_failed_protect_into_opp_protect_also_charges(self):
        """The edge: even on the rare turn the OPPONENT also stalled, our OWN failed Protect is a
        no-op (NOT exogenous denial — that's only for our ATTACK being blocked)."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="protect", our_move_outcome="fail", opp_resolved_move_id="protect"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_our_attack_blocked_by_opp_protect_is_frozen(self):
        """Regression: our ATTACK denied by the opp's Protect IS exogenous (their choice denied us) —
        the fix must not charge this; only our own failed stall move."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="earthquake", our_move_outcome="fail", opp_resolved_move_id="protect"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 1)            # frozen — neither incremented
        self.assertEqual(c.last_penalty, 0.0)

    def test_protect_with_toxic_residual_not_charged(self):
        """'If it's worth it' — a Protect (even a failed one) while an our-owned Toxic chips the opp
        NET-down is PROGRESS (the points come back), so it is NOT charged."""
        c = self._clock(); c.n = 3
        live = _full_team_live()
        live.opp.active.status = "tox"
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625   # toxic tick
        c.update(_delta(our_move_id="protect", our_move_outcome="fail",
                        opp_resolved_move_id="rockslide", opp_hp_delta=opp_hp),
                 live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)            # progress → reset
        self.assertEqual(c.last_penalty, 0.0)

    def test_failed_protect_no_switch_not_charged(self):
        """Helplessness exemption still applies: a failed Protect with no legal switch ticks the obs
        counter but is not charged (trapped-vs-wall must not be punished)."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="protect", our_move_outcome="fail", opp_resolved_move_id="rockslide"),
                 _full_team_live(), _Legal(switches=[]))
        self.assertEqual(c.n, 2)            # obs counter still ticks
        self.assertEqual(c.last_penalty, 0.0)

    def test_status_landed_resets(self):
        c = self._clock(); c.n = 2
        from agents.enums import Status
        c.update(_delta(our_move_id="toxic", opp_status_applied=Status.TOX),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    # --- Rest-LOOP (gen3_rest_loop_stall_v1): a wake-then-re-Rest is a NO_OP stall, Sleep-Talk-exempt ---
    def _rest(self, species="suicune"):
        """A SUCCESSFUL Rest by ``species``: it self-applies SLP and heals real HP → _denial_kind 'heal'."""
        from agents.enums import Status
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.6   # Rest healed a real chunk
        return _delta(our_move_id="rest", our_prev_active=species,
                      our_status_applied=Status.SLP, our_hp_delta=our_hp)

    def _live_with_moves(self, moves):
        live = _full_team_live()
        live.ours.active.move_ids = tuple(moves)
        return live

    def test_first_rest_free_then_rest_loop_charges(self):
        """The 1st Rest is a free defensive heal (HEAL_FREEZE_GRACE); a wake-then-re-Rest (no Sleep Talk)
        is a NO_OP stalled turn — it advances the obs clock AND charges the no-progress penalty."""
        c = self._clock()
        moves = ["rest", "surf", "calmmind", "icebeam"]
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertEqual(c.n, 0); self.assertEqual(c.last_penalty, 0.0)            # 1st rest: free heal
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertEqual(c.n, 1)                                                   # re-rest: NO_OP
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_sleep_talk_rest_loop_not_charged_by_the_loop(self):
        """A Sleep-Talk mon acts while asleep → looping Rest is legitimate, so the rest-loop bypass NEVER
        fires: the 2nd Rest stays within HEAL_FREEZE_GRACE (identical to the prior heal path)."""
        c = self._clock()
        moves = ["rest", "sleeptalk", "calmmind", "surf"]
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertEqual(c.n, 0); self.assertEqual(c.last_penalty, 0.0)           # frozen, not charged

    def test_winning_residual_rest_stall_is_exempt(self):
        """A Rest while our Toxic chips the opp NET-down is PROGRESS (caught by _is_progress first) → never
        charged, even on the re-Rest (a winning rest-stall is good play, not a no-progress wheel-spin)."""
        c = self._clock()
        moves = ["rest", "surf", "calmmind", "icebeam"]
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625               # toxic tick
        for _ in range(3):
            live = self._live_with_moves(moves); live.opp.active.status = "tox"
            d = self._rest(); d.opp_hp_delta = opp_hp
            c.update(d, live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0); self.assertEqual(c.last_penalty, 0.0)

    def test_failed_full_hp_rest_not_counted_as_a_rest(self):
        """A Rest at full HP FAILS (no sleep, no heal) → it neither flags a loop nor enters the rest
        history, so the next REAL Rest is still the FIRST (free)."""
        c = self._clock()
        moves = ["rest", "surf", "calmmind", "icebeam"]
        c.update(_delta(our_move_id="rest", our_prev_active="suicune"),   # no SLP applied, no heal
                 self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertNotIn("suicune", c._rested_species)                    # not recorded
        c.n = 0
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertEqual(c.last_penalty, 0.0)                             # first REAL rest = free heal

    def test_rest_loop_resets_on_episode_boundary(self):
        """`reset()` clears the per-species rest history so a new episode starts the loop count fresh."""
        c = self._clock()
        moves = ["rest", "surf", "calmmind", "icebeam"]
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)           # was a loop
        c.reset()
        self.assertEqual(c._rested_species, set())
        c.update(self._rest(), self._live_with_moves(moves), _Legal(switches=[1]))
        self.assertEqual(c.last_penalty, 0.0)                             # first rest of the new episode

    # --- Wasted Refresh (folded into gen3_rest_loop_stall_v1): a self-cure with nothing to cure is a NO_OP ---
    def test_wasted_refresh_charges_as_noop(self):
        """A Refresh used with no status to cure (our_status_cured is None) is a wasted wheel-spin → charge."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="refresh"), _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_wasted_refresh_exempt_during_winning_residual(self):
        """The winning-residual INVARIANT: a wasted Refresh while our Toxic/Leech chips the opp NET-down is
        a WINNING play, not a wheel-spin — the wasted-cure short-circuit defers to `_winning_residual`, so
        it is PROGRESS (reset), NEVER taxed by ANY path. (Corrects the earlier 'charge it anyway' behavior;
        the tax still fires on a wasted Refresh with NO winning residual — `test_wasted_refresh_charges_as_noop`.)"""
        c = self._clock(); c.n = 2
        live = _full_team_live(); live.opp.active.status = "tox"
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625      # toxic chipping the opp DOWN
        c.update(_delta(our_move_id="refresh", opp_hp_delta=opp_hp), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)                                         # winning residual → progress (reset)
        self.assertEqual(c.last_penalty, 0.0)

    def test_capped_spikes_exempt_during_winning_residual(self):
        """Same invariant for the capped-Spikes short-circuit: a wasted Spikes-at-the-3-layer-cap while our
        Toxic chips the opp net-down is a winning play → PROGRESS, not taxed."""
        c = self._clock(); c.n = 2; c._prev_spikes = 3
        live = _full_team_live(); live.opp.active.status = "tox"
        live.opp.side_conditions = {"spikes": 3}                        # opp already at the 3-layer cap
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625
        c.update(_delta(our_move_id="spikes", opp_hp_delta=opp_hp), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)
        self.assertEqual(c.last_penalty, 0.0)

    def test_refresh_then_opp_statuses_us_still_wasted(self):
        """We move first: a Refresh cast with no status cures nothing even if the opponent statuses us
        AFTER (our_status_APPLIED set, our_status_CURED still None) — still a wasted NO_OP. (The turn-21
        Milotic case: refresh first, then opp Toxic lands.)"""
        from agents.enums import Status
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="refresh", our_status_applied=Status.TOX),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_productive_refresh_not_a_wasted_cure(self):
        """A Refresh that ACTUALLY cures a status (our_status_cured set) is NOT a wasted no-op — the
        short-circuit doesn't fire, so a winning residual still exempts the legit defensive stall."""
        from agents.enums import Status
        c = self._clock(); c.n = 2
        live = _full_team_live(); live.opp.active.status = "tox"
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625
        c.update(_delta(our_move_id="refresh", our_status_cured=Status.PAR, opp_hp_delta=opp_hp),
                 live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)                                         # winning-residual progress → reset
        self.assertEqual(c.last_penalty, 0.0)

    def test_wasted_refresh_no_switch_not_charged(self):
        """Helplessness exemption: a wasted Refresh with no legal switch ticks the obs counter but is not
        charged (trapped must not be punished — consistent with the other NO_OPs)."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="refresh"), _full_team_live(), _Legal(switches=[]))
        self.assertEqual(c.n, 2)
        self.assertEqual(c.last_penalty, 0.0)

    def test_hazard_layer_resets(self):
        c = self._clock(); c.n = 2; c._prev_spikes = 1
        live = _Live([1.0] * 6, [1.0] * 6); live.opp.side_conditions = {"spikes": 2}
        c.update(_delta(our_move_id="spikes"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    def test_capped_spike_charges_as_noop(self):
        """Spikes at the 3-layer cap (no layer added) is a NO_OP: increment + charge."""
        c = self._clock(); c.n = 2; c._prev_spikes = 3
        live = _Live([1.0] * 6, [1.0] * 6); live.opp.side_conditions = {"spikes": 3}
        c.update(_delta(our_move_id="spikes"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 3)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_capped_spike_not_rescued_by_opp_switch(self):
        """The leak: an opp switch (clause (iv)) must NOT reset the clock when our move was a capped
        Spikes — otherwise the wasted Spikes that banks switch-in material escapes the no-progress tax."""
        c = self._clock(); c.n = 2; c._prev_spikes = 3
        live = _Live([1.0] * 6, [1.0] * 6); live.opp.side_conditions = {"spikes": 3}
        c.update(_delta(our_move_id="spikes", opp_switch_to="swampert"),
                 live, _Legal(switches=[1]))
        self.assertEqual(c.n, 3)                          # charged, NOT reset
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_capped_spike_no_switch_not_charged(self):
        """Trapped-vs-wall helplessness exemption still applies to a capped Spikes."""
        c = self._clock(); c.n = 2; c._prev_spikes = 3
        live = _Live([1.0] * 6, [1.0] * 6); live.opp.side_conditions = {"spikes": 3}
        c.update(_delta(our_move_id="spikes"), live, _Legal(switches=[]))
        self.assertEqual(c.n, 3)                          # obs counter still ticks
        self.assertEqual(c.last_penalty, 0.0)             # but not charged

    def test_layer_adding_spike_still_resets(self):
        """A Spikes that strictly adds a layer (2→3) is real progress and still resets (regression)."""
        c = self._clock(); c.n = 4; c._prev_spikes = 2
        live = _Live([1.0] * 6, [1.0] * 6); live.opp.side_conditions = {"spikes": 3}
        c.update(_delta(our_move_id="spikes"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    def test_filler_rapidspin_no_hazards_charges(self):
        """RapidSpin with NO spikes on our side is a filler attack: its trivial chip must NOT count as
        progress (clause (i)) — it falls through to a charged NO_OP."""
        c = self._clock(); c.n = 2; c._prev_our_spikes = 0
        c.update(_delta(our_move_id="rapidspin", our_damaging_event=object(),
                        opp_target_hp_delta=-0.05, our_move_outcome="hit"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 3)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_filler_rapidspin_not_rescued_by_opp_switch(self):
        """An incidental opp switch (clause (iv)) must not launder a filler RapidSpin into progress."""
        c = self._clock(); c.n = 2; c._prev_our_spikes = 0
        c.update(_delta(our_move_id="rapidspin", opp_switch_to="gengar", our_move_outcome="hit"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 3)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_rapidspin_clearing_our_hazards_is_progress(self):
        """A RapidSpin that actually clears our spikes (our side HAD layers) is real utility — the
        filler gate must NOT fire, so its chip still counts as progress and resets."""
        c = self._clock(); c.n = 4; c._prev_our_spikes = 2
        live = _Live([1.0] * 6, [1.0] * 6)            # our spikes cleared this window → now 0
        c.update(_delta(our_move_id="rapidspin", our_damaging_event=object(),
                        opp_target_hp_delta=-0.05, our_move_outcome="hit"),
                 live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    def test_filler_rapidspin_ko_is_progress(self):
        """A RapidSpin that lands a KO is real progress even with no hazards to clear — not penalised."""
        c = self._clock(); c.n = 3; c._prev_our_spikes = 0
        c.update(_delta(our_move_id="rapidspin", opp_fainted=True, our_damaging_event=object(),
                        opp_target_hp_delta=-0.30, our_move_outcome="hit"),
                 _full_team_live(opp_alive=5), _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    def test_filler_rapidspin_denied_freezes(self):
        """A filler RapidSpin prevented by RNG (flinch/cant) is exogenous → frozen, not charged."""
        c = self._clock(); c.n = 2; c._prev_our_spikes = 0
        c.update(_delta(our_move_id="rapidspin", our_failed_to_move=True, our_cant_reason="flinch"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertEqual(c.last_penalty, 0.0)

    def test_forced_commit_resets(self):
        c = self._clock(); c.n = 4
        c.update(_delta(our_move_id="roar", opp_switch_to="snorlax"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    def test_immune_attack_increments_and_charges(self):
        c = self._clock(); c.n = 0
        # an immune/no-op attack: no damage event, not a miss/cant → NO_OP.
        c.update(_delta(our_move_id="thunderbolt", our_move_outcome="hit"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 1)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_miss_freezes(self):
        """A missed attack FREEZES (design §4.1.1): no increment, no charge."""
        c = self._clock(); c.n = 3
        c.update(_delta(our_move_id="fireblast", our_move_outcome="miss"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 3)              # frozen
        self.assertEqual(c.last_penalty, 0.0)

    def test_prevented_move_freezes(self):
        """A full-para/sleep-prevented move (cant) FREEZES."""
        c = self._clock(); c.n = 2
        c.update(_delta(our_move_id="earthquake", our_failed_to_move=True, our_cant_reason="par"),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertEqual(c.last_penalty, 0.0)

    def test_productive_heal_freezes(self):
        """A Recover that restored real HP FREEZES (Φ_mat prices it; the clock stays out)."""
        c = self._clock(); c.n = 2
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.5   # healed 50%
        c.update(_delta(our_move_id="recover", our_hp_delta=our_hp),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertEqual(c.last_penalty, 0.0)

    def test_forced_switch_window_is_noop(self):
        c = self._clock(); c.n = 4
        c.update(_delta(phase_is_forced_switch=True), _full_team_live(), _Legal())
        self.assertEqual(c.n, 4)              # untouched
        self.assertEqual(c.last_penalty, 0.0)

    def test_trapped_no_switch_no_charge(self):
        """A no-op when no switch is legal (trapped vs a wall) increments the obs counter but is NOT
        charged — no policy could avoid it (design §4.1)."""
        c = self._clock(); c.n = 1
        c.update(_delta(our_move_id="thunderbolt", our_move_outcome="hit"),
                 _full_team_live(), _Legal(switches=[]))   # no legal switch
        self.assertEqual(c.n, 2)
        self.assertEqual(c.last_penalty, 0.0)   # not charged

    # --- heal-war fix (HEAL_FREEZE_GRACE): a sustained no-progress heal-war must eventually charge ---
    def test_sustained_heal_war_charges_after_grace(self):
        """A one-off defensive heal is free, but a SUSTAINED heal with no progress (the mirror
        stall-war) charges once past HEAL_FREEZE_GRACE — both the obs clock and the penalty engage."""
        c = self._clock()
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.5   # Recover restored 50% each turn
        live = _full_team_live()                                  # opp full HP, no residual
        for _ in range(HEAL_FREEZE_GRACE):                        # first GRACE heals: frozen
            c.update(_delta(our_move_id="recover", our_hp_delta=our_hp), live, _Legal(switches=[1]))
            self.assertEqual(c.last_penalty, 0.0)
        self.assertEqual(c.n, 0)                                  # clock idle while frozen
        c.update(_delta(our_move_id="recover", our_hp_delta=our_hp), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 1)                                  # GRACE+1-th heal: clock + charge engage
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_sustained_heal_war_not_charged_when_trapped(self):
        """Even past the grace, a heal-war with NO legal switch is not charged (helplessness)."""
        c = self._clock()
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.5
        for _ in range(HEAL_FREEZE_GRACE + 2):
            c.update(_delta(our_move_id="recover", our_hp_delta=our_hp),
                     _full_team_live(), _Legal(switches=[]))      # trapped
        self.assertEqual(c.last_penalty, 0.0)

    def test_winning_residual_stall_stays_progress(self):
        """A Recover while OUR Toxic ticks the opp DOWN is PROGRESS (a slow-damage win) — never
        charged even when sustained, unlike a no-net-progress heal-war (part-1 guard)."""
        c = self._clock(); c.n = 4
        live = _full_team_live()
        live.opp.active.status = "tox"                            # our-owned residual on the opp
        opp_hp = np.zeros(6, dtype=np.float32); opp_hp[0] = -0.0625   # opp ticked DOWN this window
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.5       # we Recovered
        for _ in range(5):                                        # sustained — a plain heal-war would charge
            c.update(_delta(our_move_id="recover", our_hp_delta=our_hp, opp_hp_delta=opp_hp),
                     live, _Legal(switches=[1]))
            self.assertEqual(c.last_penalty, 0.0)                 # PROGRESS, never charged
            self.assertEqual(c.n, 0)                              # reset every window

    def test_residual_present_but_opp_outheals_still_charges(self):
        """Toxic on the opp but the opp out-heals the tick (opp HP NOT net-down) is a true stall, not
        a win → still charges past the grace. The discriminator is the opp NET-losing HP."""
        c = self._clock()
        live = _full_team_live(); live.opp.active.status = "tox"
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.5
        for _ in range(HEAL_FREEZE_GRACE + 1):                    # opp_hp_delta defaults to 0 (out-healed)
            c.update(_delta(our_move_id="recover", our_hp_delta=our_hp), live, _Legal(switches=[1]))
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_heal_streak_resets_on_progress(self):
        """A real-progress window between heals resets the streak, so the next heal is free again."""
        c = self._clock()
        our_hp = np.zeros(6, dtype=np.float32); our_hp[0] = 0.5
        live = _full_team_live()
        for _ in range(HEAL_FREEZE_GRACE + 1):                    # push past grace → charging
            c.update(_delta(our_move_id="recover", our_hp_delta=our_hp), live, _Legal(switches=[1]))
        self.assertLess(c.last_penalty, 0.0)
        c.update(_delta(our_damaging_event=object(), opp_target_hp_delta=-0.3),
                 live, _Legal(switches=[1]))                      # progress → resets streak
        self.assertEqual(c.n, 0)
        c.update(_delta(our_move_id="recover", our_hp_delta=our_hp), live, _Legal(switches=[1]))
        self.assertEqual(c.last_penalty, 0.0)                     # next heal free again

    def test_exogenous_denial_never_charges_even_sustained(self):
        """Misses/cant are exogenous (RNG/opponent) — always frozen, no streak cap (only heals charge)."""
        c = self._clock()
        for _ in range(HEAL_FREEZE_GRACE + 3):
            c.update(_delta(our_move_id="fireblast", our_move_outcome="miss"),
                     _full_team_live(), _Legal(switches=[1]))
            self.assertEqual(c.last_penalty, 0.0)


class TestSetupProgress(unittest.TestCase):
    """gen3_setup_progress_v1 (unconditional fix): a NON-redundant own setup turn is PROGRESS — our
    active's Σ positive boost stages strictly rose (useful Calm Mind / DD / SD / Curse / Belly Drum), or
    a Substitute was newly created. A REDUNDANT setup (+6 cap / failed re-Sub) is still a charged no-op,
    and a pivot can't false-credit (gen3 resets boosts on switch)."""

    def test_useful_boost_resets(self):
        """Our positive boost-stage sum rose (0→2 via Calm Mind) → PROGRESS (reset, no charge)."""
        c = ProgressClock(); c.n = 4
        live = _full_team_live(); live.ours.active.boosts = {"spa": 1, "spd": 1}
        c.update(_delta(our_move_id="calmmind"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)
        self.assertEqual(c.last_penalty, 0.0)

    def test_redundant_boost_at_cap_still_taxed(self):
        """Calm Mind at the +6 cap (sum unchanged at 12) is a true no-op → still charged."""
        c = ProgressClock(); c.n = 2
        c._prev_our_boost_sum = 12
        live = _full_team_live(); live.ours.active.boosts = {"spa": 6, "spd": 6}
        c.update(_delta(our_move_id="calmmind"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 3)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_new_substitute_resets(self):
        """A newly-created Substitute → PROGRESS (reset)."""
        c = ProgressClock(); c.n = 3
        live = _full_team_live(); live.ours.active.volatiles = {"substitute": 1}
        c.update(_delta(our_move_id="substitute"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)
        self.assertEqual(c.last_penalty, 0.0)

    def test_redundant_substitute_still_taxed(self):
        """A re-Sub that FAILS because one is already up (has_sub unchanged) → still charged."""
        c = ProgressClock(); c.n = 1
        c._prev_our_has_sub = True
        live = _full_team_live(); live.ours.active.volatiles = {"substitute": 1}
        c.update(_delta(our_move_id="substitute", our_move_outcome="fail"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_switch_in_boostless_not_credited(self):
        """A pivot to a boostless mon DROPS the sum (4→0) — a switch is not setup-progress, so the
        strict-rise clause must NOT falsely credit it."""
        c = ProgressClock(); c.n = 2
        c._prev_our_boost_sum = 4
        live = _full_team_live(); live.ours.active.boosts = {}
        c.update(_delta(our_move_id=None, our_switch_to="metagross"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 3)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)

    def test_successful_wish_resets(self):
        """A successful Wish CAST (pending end-of-next-turn heal) is PROGRESS, not a no-op stall."""
        c = ProgressClock(); c.n = 4
        c.update(_delta(our_move_id="wish"), _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 0)
        self.assertEqual(c.last_penalty, 0.0)

    def test_failed_double_wish_still_taxed(self):
        """A 2nd Wish while one is already pending FAILS (outcome 'fail') → a true no-op → still charged."""
        c = ProgressClock(); c.n = 1
        c.update(_delta(our_move_id="wish", our_move_outcome="fail"), _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 2)
        self.assertAlmostEqual(c.last_penalty, -0.15, places=6)


if __name__ == "__main__":
    unittest.main()
