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
    MAT_HP_WEIGHT, MAT_ALIVE_WEIGHT, STATUS_TEMPO_WEIGHT, PBRS_GAMMA, VICTORY_VALUE,
    SWITCH_RISK_THRESHOLD, SAFE_PIVOT_PKO_MAX, STAY_RISK_TAX_FLOOR, ESCAPE_RISK_FRACTION,
    HAZARD_WEIGHT, BOOST_WEIGHT, OPP_BOOST_WEIGHT, FINISHING_BLOW_BONUS, EXPLOSION_BLOCK_BONUS,
    SPIKES_WASTE_PENALTY,
    _TIMEOUT_TURN_CAP,
)
from agents.training.progress_clock import (
    ProgressClock, PROGRESS_CLOCK_CAP, PROGRESS_DMG_EPS, HEAL_FREEZE_GRACE,
)


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
        self.volatiles = {}
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
        live.turn = turn   # mirror the real LiveView (its .turn comes from battle.turn) — the
                           # terminal block reads live.turn to detect the stall TIMEOUT (turn>=cap).
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
        our_effectiveness=1.0,
        our_cant_reason=None, opp_status_applied=None, opp_resolved_move_id=None,
        phase_is_forced_switch=False, our_status_applied=None,
        our_status_cured=None, opp_status_cured=None,
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
        self.assertEqual(set(bd.registry_fields(RewardClass.PBRS)),
                         {"pbrs_material", "pbrs_belief", "pbrs_status",
                          "pbrs_progress", "pbrs_hazard", "pbrs_boost", "pbrs_opp_boosts"})
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
class TestStatusPBRS(unittest.TestCase):
    """Φ_status (design §2.7 / §7.4): the non-damaging-tempo standing potential. Restores the
    standing value the event-form status BIAS drops — gated on bias_redesign, telescoping, net-zero."""

    @staticmethod
    def _set_status(live, side, idx, status):
        """Set `status` on the idx-th mon of `live.<side>` ('ours'/'opp'). The _Side tuple is
        immutable but each _Mon is mutable."""
        getattr(live, side).mons[idx].status = status

    def test_phi_status_zero_when_nobody_statused(self):
        m = Gen3RewardManager()
        self.assertAlmostEqual(m._compute_phi_status(_full_team_live()), 0.0, places=6)

    def test_phi_status_counts_non_damaging_only(self):
        """par/slp/frz move Φ_status; tox/brn/psn do NOT (their value is the HP chip → Φ_mat)."""
        m = Gen3RewardManager()
        for dmg in ("tox", "brn", "psn"):
            live = _full_team_live()
            self._set_status(live, "opp", 0, dmg)
            self.assertAlmostEqual(m._compute_phi_status(live), 0.0, places=6,
                                   msg=f"{dmg} must not move Φ_status (it is in Φ_mat)")
        for tempo in ("par", "slp", "frz"):
            live = _full_team_live()
            self._set_status(live, "opp", 0, tempo)
            self.assertAlmostEqual(m._compute_phi_status(live), STATUS_TEMPO_WEIGHT, places=6,
                                   msg=f"{tempo} must raise Φ_status by one weight")

    def test_phi_status_side_symmetry(self):
        """opp statused → +weight (good for us); our statused → −weight."""
        m = Gen3RewardManager()
        opp_live = _full_team_live(); self._set_status(opp_live, "opp", 0, "slp")
        our_live = _full_team_live(); self._set_status(our_live, "ours", 0, "par")
        self.assertAlmostEqual(m._compute_phi_status(opp_live), +STATUS_TEMPO_WEIGHT, places=6)
        self.assertAlmostEqual(m._compute_phi_status(our_live), -STATUS_TEMPO_WEIGHT, places=6)

    def test_fold_gated_off_by_default(self):
        """Default run (redesign OFF): pbrs_status ≡ 0 and _prev_phi_status stays None even with a
        sleeping opp mon → byte-identical to before Φ_status existed."""
        m = Gen3RewardManager()   # default: bias_redesign False
        live1 = _full_team_live()
        m.process_turn_reward(_Battle(live1, turn=1), _delta())
        live2 = _full_team_live(); self._set_status(live2, "opp", 0, "slp")
        m.process_turn_reward(_Battle(live2, turn=2), _delta())
        self.assertEqual(m._last_breakdown.pbrs_status, 0.0)
        self.assertIsNone(m._prev_phi_status)

    def test_fold_active_under_redesign_application_and_cure(self):
        """Under redesign: an opp falling asleep RAISES Φ → +pbrs_status; the subsequent cure (wake)
        DROPS Φ → −pbrs_status. A held status pays ≈0 (γ≈1)."""
        m = Gen3RewardManager(config=RewardConfig(bias_redesign=True))
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())  # prev=0
        sleep_live = _full_team_live(); self._set_status(sleep_live, "opp", 0, "slp")
        m.process_turn_reward(_Battle(sleep_live, turn=2), _delta())          # apply
        applied = m._last_breakdown.pbrs_status
        self.assertGreater(applied, 0.0)
        self.assertAlmostEqual(applied, PBRS_GAMMA * STATUS_TEMPO_WEIGHT, places=6)
        held_live = _full_team_live(); self._set_status(held_live, "opp", 0, "slp")
        m.process_turn_reward(_Battle(held_live, turn=3), _delta())           # held
        self.assertLess(abs(m._last_breakdown.pbrs_status), 1e-3)             # ≈0 (γ−1)·w
        m.process_turn_reward(_Battle(_full_team_live(), turn=4), _delta())   # cured
        self.assertLess(m._last_breakdown.pbrs_status, 0.0)

    def test_telescopes_to_zero_net(self):
        """Status applied then cured before a terminal that starts & ends with nobody statused →
        the UNDISCOUNTED sum of pbrs_status ≈ −Φ_status(s_0) = 0 (modulo the bounded γ residual)."""
        m = Gen3RewardManager(config=RewardConfig(bias_redesign=True))
        s1 = _full_team_live()
        s2 = _full_team_live(); self._set_status(s2, "opp", 0, "frz")
        s3 = _full_team_live(); self._set_status(s3, "opp", 0, "frz")
        s4 = _full_team_live()                                                # thawed (cured)
        s5 = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)            # win (terminal)
        total = 0.0
        for i, live in enumerate((s1, s2, s3, s4, s5)):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_status
        resid = abs(PBRS_GAMMA - 1.0) * 5 * STATUS_TEMPO_WEIGHT + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)


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
class TestBiasDrops(unittest.TestCase):
    """De-bias cleanup (audit TIER-1 distorter removal): --drop-redundant-bias / --drop-switch-bias
    ZERO their BIAS terms before the refund fold (so they also leave the accumulator). Both default
    OFF = byte-identical no-op."""

    _REDUNDANT = ("stall_tax", "matchup_penalty")
    _SWITCH = ("switch_base", "switch_bouncing_tax", "escape_threat_switch", "se_switch",
               "pivot_protect", "pivot_status", "pivot_damage", "sleep_out", "sleep_in")

    def _bd_all_bias_nonzero(self):
        bd = RewardBreakdown()
        for f in bd.registry_fields(RewardClass.BIAS):
            setattr(bd, f, 1.0)
        return bd

    def test_dropped_terms_are_registered_bias(self):
        """Every dropped field must be a BIAS-class registry member, else the drop is a typo no-op."""
        bias = set(RewardBreakdown().registry_fields(RewardClass.BIAS))
        for f in self._REDUNDANT + self._SWITCH:
            self.assertIn(f, bias, f"{f} is not a BIAS field")

    def test_default_is_noop(self):
        m = Gen3RewardManager()   # both flags default False
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            self.assertEqual(getattr(bd, f), 1.0, f"{f} must be untouched by default")

    def test_drop_redundant_bias_zeros_only_those(self):
        m = Gen3RewardManager(config=RewardConfig(drop_redundant_bias=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in self._REDUNDANT:
            self.assertEqual(getattr(bd, f), 0.0, f"{f} should be dropped")
        for f in self._SWITCH:
            self.assertEqual(getattr(bd, f), 1.0, f"{f} should be untouched")

    def test_drop_switch_bias_zeros_only_those(self):
        m = Gen3RewardManager(config=RewardConfig(drop_switch_bias=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in self._SWITCH:
            self.assertEqual(getattr(bd, f), 0.0, f"{f} should be dropped")
        for f in self._REDUNDANT:
            self.assertEqual(getattr(bd, f), 1.0, f"{f} should be untouched")

    def test_both_flags_drop_union(self):
        m = Gen3RewardManager(config=RewardConfig(drop_redundant_bias=True, drop_switch_bias=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in self._REDUNDANT + self._SWITCH:
            self.assertEqual(getattr(bd, f), 0.0)

    def test_drop_redundant_zeros_stall_tax_end_to_end(self):
        """A real turn past STALL_TAX_START_TURN charges stall_tax; the flag zeros it in the breakdown."""
        m_off = Gen3RewardManager()
        m_off.process_turn_reward(_Battle(_full_team_live(), turn=120), _delta())
        self.assertLess(m_off._last_breakdown.stall_tax, 0.0)   # charged when off
        m_on = Gen3RewardManager(config=RewardConfig(drop_redundant_bias=True))
        m_on.process_turn_reward(_Battle(_full_team_live(), turn=120), _delta())
        self.assertEqual(m_on._last_breakdown.stall_tax, 0.0)   # dropped when on

    def test_config_flows_through_from_args_and_from_dict(self):
        from types import SimpleNamespace
        from dataclasses import asdict
        rc = RewardConfig.from_args(SimpleNamespace(drop_redundant_bias=True, drop_switch_bias=True))
        self.assertTrue(rc.drop_redundant_bias and rc.drop_switch_bias)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)   # round-trips for eval/resume


# --------------------------------------------------------------------------- #
def _mgr_pbrs(**cfg):
    """A FULLY-PBRS manager (both end-state switches ON) + a real ProgressClock, so every new
    potential is live — incl. Φ_progress, which gates on stall_pbrs (the 'stall' switch)."""
    return Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True, **cfg),
                             progress_clock=ProgressClock())


# --------------------------------------------------------------------------- #
class TestProgressPBRS(unittest.TestCase):
    """Φ_progress (design §5): the anti-stall PBRS potential. Φ = −W·clock.value(); telescoping,
    terminal-zeroing; gated on --stall-pbrs (OFF = byte-identical default; --all-shaping-pbrs alone
    keeps the no_progress_tax tilt instead). The _mgr_pbrs helper turns BOTH switches on."""

    def test_phi_progress_zero_at_start(self):
        """n=0 at episode start → Φ_progress ≈ 0."""
        m = _mgr_pbrs()
        self.assertAlmostEqual(m._compute_phi_progress(None), 0.0, places=6)

    def test_phi_progress_negative_when_stalling(self):
        """n>0 → Φ_progress < 0 (FIRST guard against a sign flip rewarding stalling)."""
        m = _mgr_pbrs()
        m.progress_clock.n = 5
        self.assertLess(m._compute_phi_progress(None), 0.0)

    def test_phi_progress_none_clock_returns_zero(self):
        """Inference / standalone (no clock) → Φ_progress = 0 (no NoneType deref)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))  # no clock
        self.assertEqual(m._compute_phi_progress(None), 0.0)

    def test_phi_progress_scales_with_no_progress_penalty(self):
        """Doubling no_progress_penalty doubles |Φ_progress| (it is the weight)."""
        m1 = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, no_progress_penalty=0.15),
                               progress_clock=ProgressClock())
        m2 = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, no_progress_penalty=0.30),
                               progress_clock=ProgressClock())
        m1.progress_clock.n = m2.progress_clock.n = 4
        self.assertAlmostEqual(m2._compute_phi_progress(None), 2.0 * m1._compute_phi_progress(None),
                               places=6)

    def test_fold_gated_off_by_default(self):
        """Default run (all_shaping_pbrs OFF): pbrs_progress ≡ 0, _prev_phi_progress stays None even
        with the clock charged."""
        m = Gen3RewardManager(progress_clock=ProgressClock())   # default: all_shaping_pbrs False
        m.progress_clock.n = 5
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.progress_clock.n = 6
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        self.assertEqual(m._last_breakdown.pbrs_progress, 0.0)
        self.assertIsNone(m._prev_phi_progress)

    def test_fold_active_on_stall(self):
        """Under the flag, an increasing clock yields a negative pbrs_progress (Φ falls further)."""
        m = _mgr_pbrs()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # prev = Φ(n=0) = 0
        m.progress_clock.n = 4                                                # stalled
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        self.assertLess(m._last_breakdown.pbrs_progress, 0.0)

    def test_terminal_zeroes_phi(self):
        """A terminal window zeroes Φ → shaped = −prev (small), NOT a stall bonus."""
        m = _mgr_pbrs()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.progress_clock.n = 5
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())   # set prev<0
        prev = m._prev_phi_progress
        self.assertLess(prev, 0.0)
        win_live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win_live, turn=3), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.pbrs_progress, -prev, places=6)  # γ·0 − prev

    def test_telescopes_to_zero_net(self):
        """Clock up then reset to 0 then terminal → Σ pbrs_progress ≈ 0 (Φ(s_0)=0, policy-invariant)."""
        m = _mgr_pbrs()
        ns = [0, 3, 6, 0, 0]   # the clock trajectory; last window terminal
        total = 0.0
        for i, n in enumerate(ns):
            m.progress_clock.n = n
            term = (i == len(ns) - 1)
            live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True) if term else _full_team_live()
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_progress
        resid = abs(PBRS_GAMMA - 1.0) * len(ns) * abs(RewardConfig().no_progress_penalty) + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)


# --------------------------------------------------------------------------- #
class TestHazardPBRS(unittest.TestCase):
    """Φ_hazard (design §2.6): the spikes/hazard potential. Telescoping, terminal zeroing,
    futile-waste dissolution."""

    def test_phi_hazard_zero_at_start(self):
        m = Gen3RewardManager()
        self.assertAlmostEqual(m._compute_phi_hazard(_full_team_live()), 0.0, places=6)

    def test_phi_opp_layer_raises(self):
        """Opp gains a spikes layer (0→1) → Φ_hazard increases by HAZARD_WEIGHT."""
        m = Gen3RewardManager()
        live1 = _full_team_live(); live1.opp.side_conditions = {"spikes": 1}
        self.assertAlmostEqual(m._compute_phi_hazard(live1) - m._compute_phi_hazard(_full_team_live()),
                               HAZARD_WEIGHT, places=6)

    def test_phi_our_layer_lowers(self):
        """We gain a spikes layer (our side) → Φ_hazard decreases by HAZARD_WEIGHT (symmetric)."""
        m = Gen3RewardManager()
        live1 = _full_team_live(); live1.ours.side_conditions = {"spikes": 1}
        self.assertAlmostEqual(m._compute_phi_hazard(_full_team_live()) - m._compute_phi_hazard(live1),
                               HAZARD_WEIGHT, places=6)

    def test_phi_symmetry(self):
        """opp 3, us 1 → Φ_hazard = HAZARD_WEIGHT·(3−1) = +1.0."""
        m = Gen3RewardManager()
        live = _full_team_live()
        live.opp.side_conditions = {"spikes": 3}
        live.ours.side_conditions = {"spikes": 1}
        self.assertAlmostEqual(m._compute_phi_hazard(live), HAZARD_WEIGHT * (3 - 1), places=6)

    def test_fold_gated_off_by_default(self):
        """Default run (all_shaping_pbrs OFF): pbrs_hazard ≡ 0, _prev_phi_hazard stays None."""
        m = Gen3RewardManager()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        live2 = _full_team_live(); live2.opp.side_conditions = {"spikes": 1}
        m.process_turn_reward(_Battle(live2, turn=2), _delta(our_move_id="spikes"))
        self.assertEqual(m._last_breakdown.pbrs_hazard, 0.0)
        self.assertIsNone(m._prev_phi_hazard)

    def test_fold_layer_added_then_cleared(self):
        """Under the flag: opp gaining a layer RAISES Φ (+pbrs_hazard); a held layer ≈0; clearing
        (Rapid Spin) DROPS Φ (−pbrs_hazard)."""
        m = _mgr_pbrs()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # prev = 0
        layer = _full_team_live(); layer.opp.side_conditions = {"spikes": 1}
        m.process_turn_reward(_Battle(layer, turn=2), _delta(our_move_id="spikes"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_hazard, PBRS_GAMMA * HAZARD_WEIGHT, places=6)
        held = _full_team_live(); held.opp.side_conditions = {"spikes": 1}
        m.process_turn_reward(_Battle(held, turn=3), _delta())
        self.assertLess(abs(m._last_breakdown.pbrs_hazard), 1e-3)              # ≈0 (γ−1)·w
        m.process_turn_reward(_Battle(_full_team_live(), turn=4), _delta())    # cleared
        self.assertLess(m._last_breakdown.pbrs_hazard, 0.0)

    def test_futile_spikes_dissolves(self):
        """Spikes at the 3-layer cap → ΔΦ_hazard ≈ 0 AND bd.futile_spikes == 0 under suppression."""
        m = _mgr_pbrs()
        capped = _full_team_live(); capped.opp.side_conditions = {"spikes": 3}
        m._prev_opp_spikes = 3   # already at cap before this window
        m.process_turn_reward(_Battle(capped, turn=1), _delta())   # prev φ set
        m.process_turn_reward(_Battle(capped, turn=2), _delta(our_move_id="spikes"))
        self.assertLess(abs(m._last_breakdown.pbrs_hazard), 1e-3)
        self.assertEqual(m._last_breakdown.futile_spikes, 0.0)     # dissolved under suppression

    def test_telescopes_to_zero_net(self):
        """Hazard up then cleared then terminal → Σ pbrs_hazard ≈ 0 (Φ(s_0)=0, policy-invariant)."""
        m = _mgr_pbrs()
        s1 = _full_team_live()
        s2 = _full_team_live(); s2.opp.side_conditions = {"spikes": 1}
        s3 = _full_team_live(); s3.opp.side_conditions = {"spikes": 2}
        s4 = _full_team_live(); s4.opp.side_conditions = {"spikes": 3}
        s5 = _full_team_live()   # cleared
        s6 = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        total = 0.0
        for i, live in enumerate((s1, s2, s3, s4, s5, s6)):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_hazard
        resid = abs(PBRS_GAMMA - 1.0) * 6 * HAZARD_WEIGHT + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)


# --------------------------------------------------------------------------- #
class TestBoostPBRS(unittest.TestCase):
    """Φ_boost (design): stored-boost offensive potential; positive-only, HP-scaled, telescoping."""

    def test_phi_zero_no_boosts(self):
        m = _mgr_pbrs()
        self.assertAlmostEqual(m._compute_phi_boost(_full_team_live()), 0.0, places=6)

    def test_phi_counts_positive_only(self):
        """atk +2, spa +1, spe −1 → Σ max(0,·) = 3; full HP → Φ = BOOST_WEIGHT·3."""
        m = _mgr_pbrs()
        live = _full_team_live()
        live.ours.active.boosts = {"atk": 2, "spa": 1, "spe": -1}
        self.assertAlmostEqual(m._compute_phi_boost(live), BOOST_WEIGHT * 3 * 1.0, places=6)

    def test_phi_scales_with_hp(self):
        """Half-HP active → half the Φ_boost of full HP."""
        m = _mgr_pbrs()
        full = _full_team_live(); full.ours.active.boosts = {"atk": 2}
        half = _full_team_live(our_hp=0.5); half.ours.active.boosts = {"atk": 2}
        self.assertAlmostEqual(m._compute_phi_boost(half), m._compute_phi_boost(full) * 0.5, places=6)

    def test_phi_zero_at_cap_no_bonus(self):
        """Setting a boost already at +6 (capped) → Φ unchanged → ΔΦ ≈ 0."""
        m = _mgr_pbrs()
        capped = _full_team_live(); capped.ours.active.boosts = {"atk": 6}
        m.process_turn_reward(_Battle(capped, turn=1), _delta())   # prev φ at cap
        capped2 = _full_team_live(); capped2.ours.active.boosts = {"atk": 6}
        m.process_turn_reward(_Battle(capped2, turn=2), _delta(our_move_id="swordsdance"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_boost, 0.0, places=3)

    def test_phi_faint_drops_potential(self):
        """A boosted active fainting (new active has no boosts) → Φ drops → −pbrs_boost."""
        m = _mgr_pbrs()
        boosted = _full_team_live(); boosted.ours.active.boosts = {"atk": 3, "spa": 2}
        m.process_turn_reward(_Battle(boosted, turn=1), _delta())
        self.assertGreater(m._prev_phi_boost, 0.0)
        fainted = _full_team_live(our_alive=5)   # mon fainted, new active boostless
        m.process_turn_reward(_Battle(fainted, turn=2), _delta(we_fainted=True))
        self.assertLess(m._last_breakdown.pbrs_boost, 0.0)

    def test_fold_gated_off_by_default(self):
        m = Gen3RewardManager()
        live = _full_team_live(); live.ours.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(live, turn=1), _delta())
        self.assertEqual(m._last_breakdown.pbrs_boost, 0.0)
        self.assertIsNone(m._prev_phi_boost)

    def test_telescopes_to_zero_net(self):
        """Setup → held → faint → terminal: Σ pbrs_boost ≈ 0 (policy-invariant)."""
        m = _mgr_pbrs()
        s1 = _full_team_live()
        s2 = _full_team_live(); s2.ours.active.boosts = {"atk": 2, "spa": 1}
        s3 = _full_team_live(); s3.ours.active.boosts = {"atk": 2, "spa": 1}
        s4 = _full_team_live(our_alive=5)   # boosted mon fainted, new active boostless
        s5 = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        total = 0.0
        for i, live in enumerate((s1, s2, s3, s4, s5)):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_boost
        resid = abs(PBRS_GAMMA - 1.0) * 5 * BOOST_WEIGHT * 6 * 1.0 + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)

    def test_boost_bias_terms_suppressed_under_flag(self):
        """Under the flag, boost_utilized / futile_setup / setup_low_hp all leave the breakdown == 0."""
        m = _mgr_pbrs()
        # A capped setup at low HP would normally charge futile_setup + setup_low_hp.
        m._our_active_hp_before = 0.2
        live = _full_team_live(our_hp=0.2); live.ours.active.boosts = {"atk": 6}
        m.process_turn_reward(_Battle(live, turn=1),
                              _delta(our_move_id="swordsdance",
                                     our_boost_delta=np.zeros(7, dtype=np.int8)))
        bd = m._last_breakdown
        self.assertEqual(bd.boost_utilized, 0.0)
        self.assertEqual(bd.futile_setup, 0.0)
        self.assertEqual(bd.setup_low_hp, 0.0)


# --------------------------------------------------------------------------- #
class TestOppBoostsPBRS(unittest.TestCase):
    """Φ_opp_boosts (design): phaze-boost-disruption potential. Negative with opp boosts; a forced
    switch clearing boosts raises Φ; a failed Roar leaves them → ΔΦ=0. Telescoping, terminal zero."""

    def test_phi_zero_at_start(self):
        m = _mgr_pbrs()
        self.assertAlmostEqual(m._compute_phi_opp_boosts(_full_team_live()), 0.0, places=6)

    def test_phi_negative_with_boosts(self):
        """opp atk +3 → Φ = −OPP_BOOST_WEIGHT·3 = −0.45."""
        m = _mgr_pbrs()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 3}
        self.assertAlmostEqual(m._compute_phi_opp_boosts(live), -OPP_BOOST_WEIGHT * 3, places=6)

    def test_phi_sums_positive(self):
        """All positive stages sum: atk+2, spe+1, spa+3 → −OPP_BOOST_WEIGHT·6."""
        m = _mgr_pbrs()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 2, "spe": 1, "spa": 3}
        self.assertAlmostEqual(m._compute_phi_opp_boosts(live), -OPP_BOOST_WEIGHT * 6, places=6)

    def test_phi_ignores_negative(self):
        """A negative opp boost (def −2) does not contribute."""
        m = _mgr_pbrs()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 2, "def": -2}
        self.assertAlmostEqual(m._compute_phi_opp_boosts(live), -OPP_BOOST_WEIGHT * 2, places=6)

    def test_fold_disabled_by_default(self):
        m = Gen3RewardManager()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(live, turn=1), _delta())
        self.assertEqual(m._last_breakdown.pbrs_opp_boosts, 0.0)
        self.assertIsNone(m._prev_phi_opp_boosts)

    def test_roar_forces_switch_clears_boosts(self):
        """opp +3, then Roar forces a switch (new active boostless) → Φ rises → +pbrs_opp_boosts."""
        m = _mgr_pbrs()
        boosted = _full_team_live(); boosted.opp.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(boosted, turn=1), _delta())
        prev = m._prev_phi_opp_boosts
        self.assertAlmostEqual(prev, -OPP_BOOST_WEIGHT * 3, places=6)
        cleared = _full_team_live()   # forced-in mon, no boosts
        m.process_turn_reward(_Battle(cleared, turn=2),
                              _delta(our_move_id="roar", opp_switch_to="new"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_opp_boosts, PBRS_GAMMA * 0.0 - prev, places=6)
        self.assertGreater(m._last_breakdown.pbrs_opp_boosts, 0.0)

    def test_failed_roar_no_shaping(self):
        """Roar fails (opp stays, boosts unchanged) → ΔΦ ≈ 0 (failed_roar dissolves); only the
        bounded (γ−1)·Φ telescoping residual remains, exactly like a held status."""
        m = _mgr_pbrs()
        b1 = _full_team_live(); b1.opp.active.boosts = {"spa": 2}
        m.process_turn_reward(_Battle(b1, turn=1), _delta())
        b2 = _full_team_live(); b2.opp.active.boosts = {"spa": 2}
        m.process_turn_reward(_Battle(b2, turn=2), _delta(our_move_id="roar", opp_switch_to=None))
        self.assertLess(abs(m._last_breakdown.pbrs_opp_boosts), 1e-3)   # ≈0 (γ−1)·w

    def test_terminal_zeroes_phi(self):
        """A terminal zeroes Φ → shaped = −prev (small)."""
        m = _mgr_pbrs()
        b1 = _full_team_live(); b1.opp.active.boosts = {"atk": 2}
        m.process_turn_reward(_Battle(b1, turn=1), _delta())
        prev = m._prev_phi_opp_boosts
        win = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win, turn=2), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.pbrs_opp_boosts, -prev, places=6)


# --------------------------------------------------------------------------- #
class TestEndStateDrops(unittest.TestCase):
    """The two end-state switches (mirrors TestBiasDrops):
      --all-shaping-pbrs ("everything but stall") zeroes EVERY BIAS term EXCEPT the anti-stall tilt
        `no_progress_tax`.
      --stall-pbrs ("stall") zeroes `no_progress_tax` + `stall_tax`.
    Both OFF = byte-identical no-op; both ON ⇒ the WHOLE BIAS class is zero (fully-PBRS reward)."""

    _KEPT_BY_ALL = "no_progress_tax"            # the one BIAS term --all-shaping-pbrs keeps (the tilt)
    _STALL = ("no_progress_tax", "stall_tax")   # what --stall-pbrs zeroes

    def _bd_all_bias_nonzero(self):
        bd = RewardBreakdown()
        for f in bd.registry_fields(RewardClass.BIAS):
            setattr(bd, f, 1.0)
        return bd

    def test_default_is_noop(self):
        m = Gen3RewardManager()   # both flags default False
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            self.assertEqual(getattr(bd, f), 1.0, f"{f} must be untouched by default")

    def test_all_shaping_zeros_everything_but_stall_tilt(self):
        """--all-shaping-pbrs zeros every BIAS term EXCEPT no_progress_tax (the kept anti-stall tilt) —
        so status, stall_tax, matchup_penalty, the switch family, etc. all go."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            if f == self._KEPT_BY_ALL:
                self.assertEqual(getattr(bd, f), 1.0, "no_progress_tax (stall tilt) must be KEPT")
            else:
                self.assertEqual(getattr(bd, f), 0.0, f"{f} should be zeroed by --all-shaping-pbrs")

    def test_stall_pbrs_zeros_only_the_stall_terms(self):
        """--stall-pbrs alone zeros no_progress_tax + stall_tax; non-stall BIAS is untouched."""
        m = Gen3RewardManager(config=RewardConfig(stall_pbrs=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in self._STALL:
            self.assertEqual(getattr(bd, f), 0.0, f"{f} should be zeroed by --stall-pbrs")
        self.assertEqual(bd.spikes, 1.0)   # a non-stall term is untouched by --stall-pbrs alone

    def test_both_flags_zero_entire_bias_class(self):
        """--all-shaping-pbrs + --stall-pbrs ⇒ EVERY BIAS term is 0 → TERMINAL + PBRS only."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            self.assertEqual(getattr(bd, f), 0.0, f"{f} must be zeroed when both flags on")

    def test_finishing_blow_default_emits_then_dropped(self):
        """A clean opp-KO with a damaging move: OFF → +0.5; ON → 0."""
        live = _full_team_live(our_alive=6, opp_alive=1, opp_hp=0.5)
        live.ours.active.move_ids = ("tackle",)
        delta = _delta(opp_fainted=True, our_move_id="tackle",
                       opp_hp_delta=np.array([-0.5, 0, 0, 0, 0, 0], dtype=np.float32))
        m_off = Gen3RewardManager()
        m_off.process_turn_reward(_Battle(live, turn=1), delta)
        self.assertAlmostEqual(m_off._last_breakdown.finishing_blow, FINISHING_BLOW_BONUS, places=6)
        m_on = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        m_on.process_turn_reward(_Battle(live, turn=1), delta)
        self.assertEqual(m_on._last_breakdown.finishing_blow, 0.0)

    def test_explosion_block_default_then_dropped(self):
        """Surviving an opp Explosion (0 damage): OFF → +1.0; ON → 0."""
        from types import SimpleNamespace
        opp_event = SimpleNamespace(move_id="explosion", effectiveness=0.0, target_species=None)
        delta = _delta(opp_damaging_event=opp_event, we_fainted=False,
                       our_hp_delta=np.zeros(6, dtype=np.float32))
        m_off = Gen3RewardManager()
        m_off.process_turn_reward(_Battle(_full_team_live(), turn=1), delta)
        self.assertAlmostEqual(m_off._last_breakdown.explosion_block, EXPLOSION_BLOCK_BONUS, places=6)
        m_on = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        m_on.process_turn_reward(_Battle(_full_team_live(), turn=1), delta)
        self.assertEqual(m_on._last_breakdown.explosion_block, 0.0)

    def test_all_shaping_keeps_no_progress_tilt(self):
        """--all-shaping-pbrs WITHOUT --stall-pbrs + a charged no-op: no_progress_tax SURVIVES as the
        anti-stall tilt (and is activated even without --bias-redesign), and pbrs_progress stays 0
        (Φ_progress gates on --stall-pbrs)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True),
                              progress_clock=ProgressClock())
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.progress_clock.last_penalty = -m.config.no_progress_penalty
        m.progress_clock.n = 3
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        bd = m._last_breakdown
        self.assertLess(bd.no_progress_tax, 0.0)         # KEPT as the tilt
        self.assertEqual(bd.pbrs_progress, 0.0)          # Φ_progress off (needs --stall-pbrs)

    def test_stall_pbrs_converts_tilt_to_phi_progress(self):
        """Both switches on + a charged no-op: no_progress_tax == 0 (suppressed, after
        _apply_progress_clock) and pbrs_progress < 0 (Φ_progress carries the anti-stall telescoping)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True),
                              progress_clock=ProgressClock())
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # prev φ = 0
        m.progress_clock.last_penalty = -m.config.no_progress_penalty
        m.progress_clock.n = 3
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        bd = m._last_breakdown
        self.assertEqual(bd.no_progress_tax, 0.0)        # suppressed by --stall-pbrs
        self.assertLess(bd.pbrs_progress, 0.0)           # Φ_progress carries it telescoping

    def test_config_round_trips(self):
        from types import SimpleNamespace
        from dataclasses import asdict
        rc = RewardConfig.from_args(SimpleNamespace(all_shaping_pbrs=True, stall_pbrs=True,
                                                    no_progress_penalty=0.25))
        self.assertTrue(rc.all_shaping_pbrs and rc.stall_pbrs)
        self.assertEqual(rc.no_progress_penalty, 0.25)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)   # round-trips for eval/resume


# --------------------------------------------------------------------------- #
class TestAllShapingPbrsNoOpDefault(unittest.TestCase):
    """Global no-op-equivalence: with all_shaping_pbrs=False the four pbrs_* fields stay 0 and the
    four _prev_phi_* slots stay None across a multi-window episode (byte-identical default)."""

    def test_default_leaves_new_pbrs_inert(self):
        m = Gen3RewardManager(progress_clock=ProgressClock())   # default: flag OFF
        livesteps = [
            _full_team_live(),
            (lambda l: (setattr(l.opp, "side_conditions", {"spikes": 1}), l)[1])(_full_team_live()),
            (lambda l: (setattr(l.ours.active, "boosts", {"atk": 2}), l)[1])(_full_team_live()),
            (lambda l: (setattr(l.opp.active, "boosts", {"spa": 3}), l)[1])(_full_team_live()),
            _Live([1.0] * 6, [0.0] * 6, won=True, finished=True),
        ]
        for i, live in enumerate(livesteps):
            m.progress_clock.n = i   # would charge Φ_progress if the flag were on
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            bd = m._last_breakdown
            self.assertEqual((bd.pbrs_progress, bd.pbrs_hazard, bd.pbrs_boost, bd.pbrs_opp_boosts),
                             (0.0, 0.0, 0.0, 0.0))
        self.assertIsNone(m._prev_phi_progress)
        self.assertIsNone(m._prev_phi_hazard)
        self.assertIsNone(m._prev_phi_boost)
        self.assertIsNone(m._prev_phi_opp_boosts)


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


# --------------------------------------------------------------------------- #
class TestDrawPenalty(unittest.TestCase):
    """The DRAW / 250-turn-timeout terminal (draw_penalty). The trainee FORFEITS at the turn cap, so
    the timeout is detected by turn>=cap (not won/lost) and can be scored worse than a clean loss."""

    def _mgr(self, draw_penalty=-30.0):
        return Gen3RewardManager(config=RewardConfig(draw_penalty=draw_penalty))

    def _seed(self, m):
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # set _prev_phi_mat

    def test_timeout_uses_draw_penalty(self):
        m = self._mgr(draw_penalty=-35.0); self._seed(m)
        live = _full_team_live(lost=True, finished=True)          # forfeit at the cap
        m.process_turn_reward(_Battle(live, turn=_TIMEOUT_TURN_CAP), _delta())
        self.assertAlmostEqual(m._last_breakdown.win_loss, -35.0, places=6)

    def test_decisive_loss_before_cap_unaffected(self):
        m = self._mgr(draw_penalty=-35.0); self._seed(m)
        live = _full_team_live(our_alive=0, opp_alive=3, lost=True, finished=True)
        m.process_turn_reward(_Battle(live, turn=40), _delta(we_fainted=True))   # lost well before cap
        self.assertAlmostEqual(m._last_breakdown.win_loss, -VICTORY_VALUE, places=6)

    def test_win_unaffected(self):
        m = self._mgr(draw_penalty=-35.0); self._seed(m)
        live = _full_team_live(our_alive=3, opp_alive=0, won=True, finished=True)
        m.process_turn_reward(_Battle(live, turn=_TIMEOUT_TURN_CAP), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.win_loss, VICTORY_VALUE, places=6)

    def test_default_draw_penalty_equals_loss_value(self):
        """Default -30 == prior behavior: a timeout scores identically to a decisive loss (byte-unchanged)."""
        m = self._mgr(); self._seed(m)                            # default -30
        live = _full_team_live(lost=True, finished=True)
        m.process_turn_reward(_Battle(live, turn=_TIMEOUT_TURN_CAP), _delta())
        self.assertAlmostEqual(m._last_breakdown.win_loss, -VICTORY_VALUE, places=6)


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


# --------------------------------------------------------------------------- #
class TestBiasRedesignReframes(unittest.TestCase):
    """The obs-keyed reframes that activate ONLY under bias_redesign (design §3 #18/#25/#29). The
    default run (redesign OFF) keeps today's hidden-state behavior (byte-identical) — pinned by the
    legacy TestSwitchSubsidy / TestSeSwitchBonus / TestStatus suites running at the default config."""

    def _mgr(self, redesign):
        return Gen3RewardManager(config=RewardConfig(bias_redesign=redesign))

    # --- status (#29): transition events under redesign, count diff by default ---
    def test_status_redesign_keys_on_transition_events(self):
        from agents.enums import Status
        live = _full_team_live()   # mock snapshot shows nobody statused
        # opp GAINED a status this window (the event) → + even though the count snapshot is 0.
        reward, d_opp = self._mgr(True)._compute_status_reward(_delta(opp_status_applied=Status.TOX), live)
        self.assertGreater(reward, 0.0)
        self.assertGreater(d_opp, 0)
        # opp CURED a status (e.g. Rest) → − (we lost the pressure).
        reward_c, _ = self._mgr(True)._compute_status_reward(_delta(opp_status_cured=Status.TOX), live)
        self.assertLess(reward_c, 0.0)

    def test_status_default_keys_on_count_diff(self):
        statused = _full_team_live()
        statused.opp.mons[0].status = "tox"   # a statused opp mon, no transition event in the delta
        reward, d_opp = self._mgr(False)._compute_status_reward(_delta(), statused)
        self.assertGreater(reward, 0.0)   # count diff (0→1) drives it; events would have given 0
        self.assertEqual(d_opp, 1)

    # --- switch anti-bounce (#18 + the ai_v5_6 bounce-farm regression fix) ---
    def _settle_switch(self, m, decision_turn, last_switch_turn, family=None):
        m._last_reward_metadata = {"type": "VOLUNTARY", "decision_turn": decision_turn,
                                   "switch_from": "zapdos", "target_species": "tyranitar"}
        m._last_switched_from = "snorlax"   # != target → not a 2-cycle bounce
        m.last_switch_turn = last_switch_turn
        bd = RewardBreakdown()
        if family:   # pre-set the rewards process_turn_reward folds BEFORE _apply_switch_outcome
            bd.se_switch, bd.pivot_protect, bd.pivot_status, bd.pivot_damage = family
        m._apply_switch_outcome(_delta(our_switch_to="tyranitar"), bd)
        return bd

    def test_switch_base_spam_gated_under_redesign(self):
        """ai_v5_6 fix: switch_base is now spam-gated under the redesign too (was flat = the bug). A
        back-to-back switch (decision 5, last 4) → 0; a switch after committing (>1-turn gap) → full."""
        from agents.training.reward_manager import SWITCH_BASE_BONUS
        self.assertAlmostEqual(self._settle_switch(self._mgr(True), 5, 4).switch_base, 0.0, places=6)
        self.assertAlmostEqual(self._settle_switch(self._mgr(True), 5, 2).switch_base,
                               SWITCH_BASE_BONUS, places=6)

    def test_switch_base_spam_gated_by_default(self):
        bd = self._settle_switch(self._mgr(False), decision_turn=5, last_switch_turn=4)
        self.assertAlmostEqual(bd.switch_base, 0.0, places=6)   # back-to-back → spam-gated to 0

    def test_bounce_farm_zeros_whole_switch_family_under_redesign(self):
        """THE ai_v5_6 regression guard: a BACK-TO-BACK switch must collect NOTHING from any switch
        reward (switch_base / se_switch / escape / pivot_*) under the redesign, so an every-turn
        bounce-farm (any cycle length) is unprofitable. (Real run lost to random at 94% switches.)"""
        m = self._mgr(True)
        m._prev_opp_se_threat = True   # would otherwise fire escape_threat_switch
        bd = self._settle_switch(m, decision_turn=5, last_switch_turn=4,
                                 family=(0.2, 0.1, 0.1, 0.1))   # se_switch + pivots pre-folded
        self.assertEqual(
            (bd.switch_base, bd.se_switch, bd.escape_threat_switch,
             bd.pivot_protect, bd.pivot_status, bd.pivot_damage),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_legit_switch_keeps_family_under_redesign(self):
        """The gate only nukes back-to-back farming — a switch after committing (>1-turn gap) keeps
        the full switch-reward family, so legitimate pivoting is unaffected."""
        m = self._mgr(True)
        m._prev_opp_se_threat = True
        bd = self._settle_switch(m, decision_turn=5, last_switch_turn=2,   # 3-turn gap
                                 family=(0.2, 0.1, 0.1, 0.1))
        self.assertGreater(bd.switch_base, 0.0)
        self.assertEqual(bd.se_switch, 0.2)               # untouched
        self.assertEqual(bd.pivot_damage, 0.1)            # untouched
        self.assertGreater(bd.escape_threat_switch, 0.0)  # paid (threat present, not back-to-back)

    def test_switch_bouncing_tax_survives_clock_under_redesign(self):
        """The escalating 2-cycle bounce tax is NOT suppressed by the no-progress clock (it stays a
        real negative brake); repetition / struggle / dead-matchup ARE still subsumed."""
        from agents.training.progress_clock import ProgressClock
        m = self._mgr(True); m.progress_clock = ProgressClock()
        bd = RewardBreakdown()
        bd.switch_bouncing_tax = -1.5; bd.repetition_tax = -0.2; bd.dead_matchup_tax = -0.3
        m._apply_progress_clock(bd)
        self.assertEqual(bd.switch_bouncing_tax, -1.5)   # KEPT (the fix)
        self.assertEqual(bd.repetition_tax, 0.0)         # subsumed
        self.assertEqual(bd.dead_matchup_tax, 0.0)       # subsumed


class TestPbrsGammaInvariant(unittest.TestCase):
    def test_pbrs_gamma_default_matches_ppo(self):
        # The train_rl_agent assert pins PBRS_GAMMA == model.gamma (0.9999) at build time.
        self.assertAlmostEqual(PBRS_GAMMA, 0.9999, places=9)
        self.assertAlmostEqual(RewardConfig().gamma, PBRS_GAMMA, places=9)


class TestRewardConfigSingleSource(unittest.TestCase):
    """RewardConfig is the single source of truth: `from_args` builds it ONCE from the CLI, `from_dict`
    reconstructs it from model_config.json for eval/resume. So adding a reward flag (field + matching
    CLI arg) flows everywhere with no hand-threading — the gap that once let eval silently measure with
    a default (bias_redesign=False) reward (the ai_v5_6 mismeasurement)."""

    def test_from_args_pulls_every_field_by_name(self):
        from types import SimpleNamespace
        args = SimpleNamespace(bias_additivity=0.5, mat_alive_weight=1.4, no_progress_penalty=0.3,
                               switch_bias_weight=2.0, bias_redesign=True, draw_penalty=-35.0,
                               unrelated="ignored")
        rc = RewardConfig.from_args(args)
        self.assertEqual((rc.bias_additivity, rc.mat_alive_weight, rc.no_progress_penalty,
                          rc.switch_bias_weight, rc.bias_redesign, rc.draw_penalty),
                         (0.5, 1.4, 0.3, 2.0, True, -35.0))
        self.assertAlmostEqual(rc.gamma, 0.9999)   # fixed PPO discount, never from args

    def test_from_args_defaults_match_explicit(self):
        from types import SimpleNamespace
        args = SimpleNamespace(bias_additivity=1.0, mat_alive_weight=1.25, no_progress_penalty=0.15,
                               switch_bias_weight=0.0, bias_redesign=False, draw_penalty=-30.0)
        self.assertEqual(RewardConfig.from_args(args), RewardConfig())

    def test_from_dict_round_trips(self):
        from dataclasses import asdict
        rc = RewardConfig(bias_redesign=True, draw_penalty=-35.0, no_progress_penalty=0.25,
                          switch_bias_weight=1.5, bias_additivity=0.5)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)

    def test_from_dict_ignores_unknown_and_defaults_missing(self):
        # a real model_config.json carries arch fields + use_popart (ignored), and may PREDATE a
        # reward field (defaulted) — reconstruction must never KeyError on a stray/missing key.
        rc = RewardConfig.from_dict({"bias_redesign": True, "arch_signature": "x", "use_popart": True})
        self.assertTrue(rc.bias_redesign)
        self.assertEqual(rc.draw_penalty, RewardConfig().draw_penalty)   # absent → default
        self.assertEqual(RewardConfig.from_dict(None), RewardConfig())   # None → all defaults

    def test_eval_player_and_builder_require_reward_factory(self):
        """No silent default: a missing reward factory must be a loud TypeError, not a default reward
        config — that default is exactly what mismeasured eval. (Signature guard; no heavy construction.)"""
        import inspect
        from agents.training.eval_callback import EvalRLPlayer, build_eval_players
        for fn, pname in ((EvalRLPlayer.__init__, "reward_fn_factory"),
                          (build_eval_players, "reward_fn_factory")):
            p = inspect.signature(fn).parameters[pname]
            self.assertIs(p.default, inspect.Parameter.empty,
                          f"{fn.__qualname__}.{pname} must be REQUIRED (no silent default)")


class TestSelfKoPenalty(unittest.TestCase):
    """The HP-scaled self-KO penalty (--self-ko-hp-penalty): −w·hp when our mon self-KOs, OFF by
    default, gated on actually-fainted + actually-executed, and untouched for non-self-KO moves."""

    def _mgr(self, weight, hp_before=1.0):
        m = Gen3RewardManager(config=RewardConfig(self_ko_hp_penalty=weight))
        m._our_active_hp_before = hp_before
        return m

    def test_off_by_default_is_zero(self):
        """w=0.0 (default) → no penalty even on a healthy self-KO (byte-unchanged)."""
        m = self._mgr(0.0, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="explosion", we_fainted=True)), 0.0)

    def test_healthy_self_ko_charges_full_weight(self):
        """A full-HP Explosion that faints us → −w·1.0 (the blunder this lever targets)."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertAlmostEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="explosion", we_fainted=True)),
            -2.5, places=6)

    def test_low_hp_self_ko_is_scaled_down(self):
        """A dying-mon Self-Destruct (10% HP) is the legitimate sac-for-KO → only −w·0.1."""
        m = self._mgr(2.5, hp_before=0.1)
        self.assertAlmostEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="selfdestruct", we_fainted=True)),
            -0.25, places=6)

    def test_survived_explosion_no_penalty(self):
        """Blocked / immune Explosion (we did NOT faint) threw nothing away → no penalty."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="explosion", we_fainted=False)), 0.0)

    def test_failed_to_move_no_penalty(self):
        """KO'd before our Explosion fired (our_failed_to_move) → the faint wasn't our self-KO."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(
                _delta(our_move_id="explosion", we_fainted=True, our_failed_to_move=True)), 0.0)

    def test_non_self_ko_move_no_penalty(self):
        """Fainting to a normal attack while holding a non-self-KO move → no penalty."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="earthquake", we_fainted=True)), 0.0)

    def test_post_self_ko_forced_switch_window_no_double_charge(self):
        """The forced-switch follow-up window after a self-KO has our_move_id=None — the penalty must
        NOT fire there, so one self-KO is charged exactly once (no double-charge across the window)."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id=None, we_fainted=True)), 0.0)

    def test_hp_clamped_to_unit_interval(self):
        """A stray >1 or <0 HP snapshot can't blow the penalty past [−w, 0]."""
        self.assertAlmostEqual(
            self._mgr(2.0, hp_before=1.5)._compute_self_ko_penalty(
                _delta(our_move_id="explosion", we_fainted=True)), -2.0, places=6)
        self.assertEqual(
            self._mgr(2.0, hp_before=-0.3)._compute_self_ko_penalty(
                _delta(our_move_id="explosion", we_fainted=True)), 0.0)


if __name__ == "__main__":
    unittest.main()
