"""Unit tests for the Markovian / PBRS reward redesign
(design_markovian_reward_and_features.md): the reward registry, the material PBRS Φ_mat, the
bias-additivity accumulate-refund, and the ProgressClock predicate. Pure (no battle sim).
"""
import math
import unittest
from dataclasses import fields

import numpy as np

from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, RewardClass, RewardBreakdown,
    MAT_HP_WEIGHT, MAT_ALIVE_WEIGHT, PBRS_GAMMA, VICTORY_VALUE,
    SWITCH_RISK_THRESHOLD, SAFE_PIVOT_PKO_MAX, STAY_RISK_TAX_FLOOR, ESCAPE_RISK_FRACTION,
)
from agents.training.progress_clock import ProgressClock, PROGRESS_CLOCK_CAP, PROGRESS_DMG_EPS


# --------------------------------------------------------------------------- #
# Minimal LiveView / battle / delta stubs for Φ_mat + the fold.                 #
# --------------------------------------------------------------------------- #
class _Mon:
    """Rich enough for the full process_turn_reward fold (status/se/dead-matchup/belief helpers)."""
    def __init__(self, hp_fraction=1.0, species="mon", active=False):
        self.hp_fraction = hp_fraction
        self.fainted = hp_fraction <= 0.0
        self.species = species
        self.active = active
        self.status = None
        self.types = ()
        self.move_ids = ()
        self.boosts = {}
        self.ability = None
        self.item = None
        self.consumed_item = None
        self.stats = {}
        self.current_hp = None
        self.max_hp = None


class _Side:
    def __init__(self, hps, team_size=6, spikes=0):
        self.mons = tuple(_Mon(h, active=(i == 0)) for i, h in enumerate(hps))
        self.active = self.mons[0] if (self.mons and not self.mons[0].fainted) else None
        self.team_size = team_size
        self.side_conditions = {"spikes": spikes} if spikes else {}


class _Live:
    def __init__(self, our_hps, opp_hps, opp_team_size=6, won=False, lost=False, finished=False):
        self.ours = _Side(our_hps)
        self.opp = _Side(opp_hps, team_size=opp_team_size)
        self.weather = None
        self.won, self.lost, self.finished = won, lost, finished


class _Battle:
    def __init__(self, live, turn=1):
        self._live = live
        self.turn = turn
        self.won = live.won
        self.lost = live.lost
        self.finished = live.finished

    def live_view(self):
        return self._live


def _delta(**kw):
    """A minimal TurnDelta-like object exposing only the fields the fold/clock read."""
    from types import SimpleNamespace
    base = dict(
        our_hp_delta=np.zeros(6, dtype=np.float32), opp_hp_delta=np.zeros(6, dtype=np.float32),
        our_boost_delta=np.zeros(7, dtype=np.int8),
        we_fainted=False, opp_fainted=False, our_move_id=None, our_switch_to=None,
        opp_switch_to=None, opp_damaging_event=None, our_damaging_event=None,
        opp_target_hp_delta=None, our_move_outcome=None, our_failed_to_move=False,
        our_cant_reason=None, opp_status_applied=None, opp_resolved_move_id=None,
        phase_is_forced_switch=False, our_status_applied=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Legal:
    def __init__(self, switches=()):
        self.switches = tuple(switches)


def _full_team_live(our_alive=6, opp_alive=6, our_hp=1.0, opp_hp=1.0, **kw):
    """A 6v6 LiveView with `our_alive`/`opp_alive` mons at the given HP, rest fainted."""
    our = [our_hp] * our_alive + [0.0] * (6 - our_alive)
    opp = [opp_hp] * opp_alive + [0.0] * (6 - opp_alive)
    return _Live(our, opp, **kw)


# --------------------------------------------------------------------------- #
class TestRewardRegistry(unittest.TestCase):
    """The registry (design §1.1) is exhaustive, non-overlapping, and drives the fold."""

    def test_every_float_field_has_exactly_one_class(self):
        bd = RewardBreakdown()
        float_fields = {f.name for f in fields(bd) if isinstance(getattr(bd, f.name), float)}
        reg = set(bd._REGISTRY)
        # Only bias_refund (the fold mechanism, not a term) is outside the registry.
        self.assertEqual(float_fields - reg, {"bias_refund"})
        self.assertEqual(reg - float_fields, set())

    def test_groups_cover_every_registry_field_exactly_once(self):
        """Pin the THIRD parallel list (_GROUPS, used by to_dict's log line) against the registry —
        a field added to the registry but missed in _GROUPS would silently vanish from the breakdown
        log with no other test failure (design-review DRY finding #5)."""
        bd = RewardBreakdown()
        grouped = [f for _name, fields_ in bd._GROUPS for f in fields_]
        self.assertEqual(len(grouped), len(set(grouped)), "a field appears in two _GROUPS buckets")
        # Every registry term + the bias_refund mechanism must appear in exactly one bucket.
        expected = set(bd._REGISTRY) | {"bias_refund"}
        self.assertEqual(set(grouped), expected)

    def test_class_membership_matches_design(self):
        bd = RewardBreakdown()
        self.assertEqual(set(bd.registry_fields(RewardClass.TERMINAL)), {"win_loss"})
        self.assertEqual(set(bd.registry_fields(RewardClass.PBRS)), {"pbrs_material", "pbrs_belief"})
        # Everything else is BIAS.
        self.assertIn("no_progress_tax", bd.registry_fields(RewardClass.BIAS))
        self.assertIn("roar", bd.registry_fields(RewardClass.BIAS))

    def test_total_sums_all_terms(self):
        bd = RewardBreakdown(win_loss=30.0, pbrs_material=-1.5, roar=0.2,
                             no_progress_tax=-0.15, bias_refund=0.05)
        self.assertAlmostEqual(bd.total, 30.0 - 1.5 + 0.2 - 0.15 + 0.05, places=6)


# --------------------------------------------------------------------------- #
class TestMaterialPBRS(unittest.TestCase):
    """Φ_mat (design §2): declared-team material potential, telescoping, terminal zeroing."""

    def _mgr(self):
        return Gen3RewardManager()

    def test_phi_mat_start_is_zero(self):
        """6v6 full HP over DECLARED teams → Φ_mat(s_0) ≈ 0 (no reveal jumps / start variance)."""
        m = self._mgr()
        phi = m._compute_phi_mat(_full_team_live())
        self.assertAlmostEqual(phi, 0.0, places=6)

    def test_phi_mat_unrevealed_opp_counts_full(self):
        """Only the opp lead revealed (team_size 6) → opp still counts 6 full-HP-alive → Φ_mat≈0."""
        m = self._mgr()
        live = _Live([1.0] * 6, [1.0], opp_team_size=6)   # opp has 1 revealed mon, 5 unrevealed
        self.assertAlmostEqual(m._compute_phi_mat(live), 0.0, places=6)

    def test_phi_mat_faint_lowers_never_raises(self):
        """An our-faint drops Φ_mat (HP term + alive term); it must never raise it."""
        m = self._mgr()
        before = m._compute_phi_mat(_full_team_live(our_alive=6))
        after = m._compute_phi_mat(_full_team_live(our_alive=5))   # one of ours fainted
        self.assertLess(after, before)
        # Drop ≈ 2·1.0 (HP) + MAT_ALIVE_WEIGHT (alive bit).
        self.assertAlmostEqual(before - after, MAT_HP_WEIGHT * 1.0 + MAT_ALIVE_WEIGHT, places=5)

    def test_telescopes_to_minus_phi0(self):
        """Drive the manager over a controlled-HP trajectory ending terminal; the UNDISCOUNTED sum
        of pbrs_material ≈ −Φ_mat(s_0) + the bounded (γ−1) residual (coarse telescoping check)."""
        m = self._mgr()
        # Trajectory: we chip the opp down, lose one mon, then win. Φ_mat(s_0)=0 (6v6 full).
        livesteps = [
            _full_team_live(opp_hp=1.0),                 # s_1 (first window → no shaping)
            _full_team_live(opp_hp=0.5),                 # dealt damage
            _full_team_live(our_alive=5, opp_hp=0.5),    # we lost a mon
            _full_team_live(our_alive=5, opp_alive=2),   # KO'd some opp
            _full_team_live(our_alive=5, opp_alive=0, won=True, finished=True),   # win (terminal)
        ]
        phi0 = m._compute_phi_mat(livesteps[0])
        s = 0.0
        for i, live in enumerate(livesteps):
            bd = m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            s += m._last_breakdown.pbrs_material
        # Σ(γΦ′−Φ) telescopes to γ^T·Φ_T − Φ_0 + (γ−1)Σ_interior; Φ_T zeroed at terminal.
        resid = abs((PBRS_GAMMA - 1.0)) * 6 * (MAT_HP_WEIGHT * 6 + MAT_ALIVE_WEIGHT * 6)
        self.assertAlmostEqual(s, -phi0, delta=resid + 1e-6)

    def test_terminal_zeroes_phi_no_dominant_bonus(self):
        """The highest-risk bug (design §2.3): a 6-0 win has Φ_mat(post-win)≈+19.5, which MUST be
        zeroed → pbrs_material = −prev (small), NOT a +19.5 dominant-win bonus."""
        m = self._mgr()
        # window 1: even board → set _prev_phi_mat.
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        prev = m._prev_phi_mat
        # window 2: we win 6-0 at full HP (post-win Φ_mat would be +19.5 un-zeroed).
        win_live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win_live, turn=2), _delta(opp_fainted=True))
        bd = m._last_breakdown
        self.assertAlmostEqual(bd.pbrs_material, -prev, places=6)   # = γ·0 − prev
        self.assertLess(abs(bd.pbrs_material), 1.0)                  # small, not +19.5
        self.assertAlmostEqual(bd.win_loss, VICTORY_VALUE, places=6)


# --------------------------------------------------------------------------- #
class TestBiasAdditivity(unittest.TestCase):
    """The bias-additivity accumulate-refund (design §1.2)."""

    def _run_episode(self, lam, bias_per_turn=(0.2, -0.1, 0.3), terminal_win=True):
        """Drive the manager over windows that each inject a known BIAS value via `roar`, then a
        terminal. Returns (per-window totals, episode bias contribution)."""
        m = Gen3RewardManager(config=RewardConfig(bias_additivity=lam))
        totals = []
        acc = 0.0
        n = len(bias_per_turn)
        for i, b in enumerate(bias_per_turn):
            live = _full_team_live()
            is_term = terminal_win and (i == n - 1)
            if is_term:
                live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            # Inject the bias post-hoc by reusing the fold: simplest is to set roar in the breakdown
            # and recompute — instead, assert via a dedicated manager hook. We emulate by adding b to
            # the recorded bias accumulator through the public refund formula.
            acc += b
            totals.append(b)
        return totals, acc

    def test_lambda_one_refund_is_zero(self):
        """λ=1 → bias_refund ≡ 0 on every window (byte no-op for the bias class)."""
        m = Gen3RewardManager(config=RewardConfig(bias_additivity=1.0))
        # A window with a nonzero bias (roar) and a material change.
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.process_turn_reward(_Battle(_full_team_live(opp_hp=0.5), turn=2),
                              _delta(our_move_id="roar"))
        self.assertEqual(m._last_breakdown.bias_refund, 0.0)

    def test_parameterized_blend_episode_sum(self):
        """Episode-summed BIAS contribution == λ·acc for λ ∈ {0, 0.5, 1} (accumulate-refund). We
        drive a known per-window bias accumulator and check the refund column telescopes to
        −(1−λ)·acc (modulo the bounded γ residual)."""
        # Use the manager's own accumulator by feeding windows whose ONLY bias is stall_tax (a
        # deterministic function of turn). Compare the summed (bias + refund) across λ.
        def episode_bias_sum(lam):
            m = Gen3RewardManager(config=RewardConfig(bias_additivity=lam))
            bias_total = 0.0
            refund_total = 0.0
            turns = [120, 140, 160]   # past STALL_TAX_START_TURN → nonzero stall_tax each window
            for i, t in enumerate(turns):
                live = _full_team_live()
                if i == len(turns) - 1:
                    live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
                m.process_turn_reward(_Battle(live, turn=t), _delta())
                bd = m._last_breakdown
                bias_total += sum(getattr(bd, f) for f in bd.registry_fields(RewardClass.BIAS))
                refund_total += bd.bias_refund
            return bias_total, refund_total

        b1, r1 = episode_bias_sum(1.0)
        b0, r0 = episode_bias_sum(0.0)
        bh, rh = episode_bias_sum(0.5)
        # Per-window bias values are IDENTICAL across λ (only the refund differs).
        self.assertAlmostEqual(b1, b0, places=9)
        self.assertAlmostEqual(b1, bh, places=9)
        # λ=1 → refund 0; λ=0 → refund ≈ −acc; λ=0.5 → ≈ −0.5·acc (bounded γ residual).
        self.assertAlmostEqual(r1, 0.0, places=9)
        resid = abs(PBRS_GAMMA - 1.0) * abs(b1) * 3 + 1e-9
        self.assertAlmostEqual(b0 + r0, 0.0, delta=resid)          # contribution λ·acc = 0
        self.assertAlmostEqual(bh + rh, 0.5 * b1, delta=resid)     # contribution 0.5·acc


# --------------------------------------------------------------------------- #
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

    def test_status_landed_resets(self):
        c = self._clock(); c.n = 2
        from agents.enums import Status
        c.update(_delta(our_move_id="toxic", opp_status_applied=Status.TOX),
                 _full_team_live(), _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

    def test_hazard_layer_resets(self):
        c = self._clock(); c.n = 2; c._prev_spikes = 1
        live = _Live([1.0] * 6, [1.0] * 6); live.opp.side_conditions = {"spikes": 2}
        c.update(_delta(our_move_id="spikes"), live, _Legal(switches=[1]))
        self.assertEqual(c.n, 0)

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


# --------------------------------------------------------------------------- #
class TestSwitchBias(unittest.TestCase):
    """The belief-risk-scaled switch BIAS lever (design_reward_switching.md §6) — the under-switch
    fix that `pbrs_belief` (policy-invariant) can't be. Tests the gates + scaling directly off the
    decision-time snapshots (`_prev_active_ko_risk` / `_prev_safe_pivot`), which is what the fold sets."""

    def _mgr(self, weight):
        return Gen3RewardManager(config=RewardConfig(switch_bias_weight=weight))

    def _armed(self, weight, risk=0.7, safe=True):
        """A manager with the decision-time threat snapshot pre-set (as the fold would)."""
        m = self._mgr(weight)
        m._prev_active_ko_risk = risk
        m._prev_safe_pivot = safe
        return m

    # --- stay-into-KO tax -------------------------------------------------- #
    def test_off_by_default(self):
        """switch_bias_weight=0.0 (default) → no tax even in a maximal-danger state (single-variable)."""
        m = self._armed(0.0, risk=1.0, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_stay_tax_fires_scaled_by_risk(self):
        m = self._armed(1.5, risk=0.7, safe=True)
        self.assertAlmostEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), -1.5 * 0.7, places=6)

    def test_stay_tax_clamped_at_floor(self):
        m = self._armed(3.0, risk=1.0, safe=True)   # -3.0 would exceed the floor
        self.assertAlmostEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)),
                               STAY_RISK_TAX_FLOOR, places=6)

    def test_no_tax_without_safe_pivot(self):
        """A forced stay (no bench-in would survive) is NEVER taxed — the key false-positive guard."""
        m = self._armed(1.5, risk=0.9, safe=False)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_no_tax_below_risk_threshold(self):
        m = self._armed(1.5, risk=SWITCH_RISK_THRESHOLD - 0.01, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_no_tax_when_we_switched(self):
        m = self._armed(1.5, risk=0.9, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=2)), 0.0)

    def test_no_tax_when_we_kod_them(self):
        """If our move KO'd the opp, staying WON the exchange — not a misplay, no tax."""
        m = self._armed(1.5, risk=0.9, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None, opp_fainted=True)), 0.0)

    def test_no_tax_when_trapped(self):
        """The key false-positive guard: a TRAPPED stay (no legal switch this decision) is NOT taxed,
        even with a 'safe' bench it can't reach."""
        m = self._armed(1.5, risk=0.95, safe=True)
        m._cur_can_switch = False                       # mask had no legal switch (Arena Trap etc.)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_no_tax_on_rng_fizzle(self):
        """A flinch / full-para / sleep / freeze (our_failed_to_move) is not a deliberate stay."""
        m = self._armed(1.5, risk=0.95, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None, our_failed_to_move=True)), 0.0)

    def test_tax_fires_on_stay_and_die(self):
        """Staying-and-fainting is the TARGET pathology — it must be taxed (not exempted)."""
        m = self._armed(1.5, risk=0.95, safe=True)
        self.assertAlmostEqual(
            m._compute_stay_risk_tax(_delta(our_switch_to=None, we_fainted=True)), -1.5 * 0.95, places=6)

    # --- escape reward ----------------------------------------------------- #
    def test_escape_bonus_fires_scaled_and_asymmetric(self):
        m = self._armed(1.5, risk=0.8, safe=True)
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertAlmostEqual(bd.escape_risk_bonus, 1.5 * ESCAPE_RISK_FRACTION * 0.8, places=6)
        # asymmetric: the escape reward is strictly smaller than the equivalent stay-tax magnitude
        self.assertLess(bd.escape_risk_bonus, 1.5 * 0.8)

    def test_escape_bonus_needs_safe_pivot(self):
        """Reward escaping TO safety, not sacrificing a fresh mon into the same threat (kills the
        no-safe-pivot rotation farm)."""
        m = self._armed(1.5, risk=0.8, safe=False)      # high risk but NO safe pivot
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertEqual(bd.escape_risk_bonus, 0.0)

    def test_escape_bonus_off_by_default(self):
        m = self._armed(0.0, risk=0.8, safe=True)
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertEqual(bd.escape_risk_bonus, 0.0)

    def test_escape_bonus_needs_high_risk(self):
        m = self._armed(1.5, risk=SWITCH_RISK_THRESHOLD - 0.01, safe=True)
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertEqual(bd.escape_risk_bonus, 0.0)

    # --- class membership + plumbing -------------------------------------- #
    def test_new_fields_are_bias_class(self):
        bd = RewardBreakdown()
        self.assertIn("stay_risk_tax", bd.registry_fields(RewardClass.BIAS))
        self.assertIn("escape_risk_bonus", bd.registry_fields(RewardClass.BIAS))

    def test_stay_tax_flows_through_process_turn_reward(self):
        """End-to-end through the full fold (not just the helper): a high-risk stay with a safe pivot
        lands a nonzero stay_risk_tax in the breakdown and in the summed reward."""
        m = self._mgr(1.5)
        m._prev_active_ko_risk = 0.9          # decision-time snapshot, as the prior turn's fold sets
        m._prev_safe_pivot = True
        m._last_reward_metadata = {"type": "ATTACK"}   # a stay (move), not a switch
        m.process_turn_reward(_Battle(_full_team_live(), turn=3),
                              _delta(our_switch_to=None, our_move_id="thunderbolt"))
        bd = m._last_breakdown
        self.assertAlmostEqual(bd.stay_risk_tax, -1.5 * 0.9, places=6)
        self.assertLess(bd.total, 0.0)        # the tax pulls the (otherwise ~0) turn negative

    def test_belief_potential_returns_safe_pivot_signal(self):
        """The fold's source: _belief_potential_and_risk now returns the 3rd value (min bench P(KO))."""
        m = self._mgr(1.5)
        out = m._belief_potential_and_risk(_full_team_live())
        self.assertEqual(len(out), 3)   # (phi, active_risk, min_bench_pko)

    def test_fold_belief_pbrs_sets_safe_pivot_snapshot(self):
        """_fold_belief_pbrs must populate _prev_safe_pivot (so the next turn's stay-tax can gate)."""
        m = self._mgr(1.5)
        bd = RewardBreakdown()
        m._fold_belief_pbrs(bd, _full_team_live(), is_terminal=False)
        self.assertIsInstance(m._prev_safe_pivot, bool)
        # terminal zeroes the snapshot (no shaping carried past the game end)
        m._fold_belief_pbrs(bd, _full_team_live(), is_terminal=True)
        self.assertFalse(m._prev_safe_pivot)


class TestPbrsGammaInvariant(unittest.TestCase):
    def test_pbrs_gamma_default_matches_ppo(self):
        # The train_rl_agent assert pins PBRS_GAMMA == model.gamma (0.9999) at build time.
        self.assertAlmostEqual(PBRS_GAMMA, 0.9999, places=9)
        self.assertAlmostEqual(RewardConfig().gamma, PBRS_GAMMA, places=9)


if __name__ == "__main__":
    unittest.main()
