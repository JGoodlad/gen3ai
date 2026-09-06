"""The TERMINAL-only short circuit changes NOTHING but the clock (`gen3_terminal_only_short_circuit_v1`).

**What is being guarded.** When a run's composition is the ±``victory_value`` terminal ALONE
(``--no-hand-shaping --terminal-indicator`` — the win-prob arm), ``process_turn_reward`` skips a
handful of calls the older ``_active_bias`` fast path could NOT skip: they were deliberately
ungated *because* their cross-turn mutations feed BIAS terms. Under a composition with no BIAS
terms those mutations have no reader, so the whole call is dead — but "has no reader" is a claim
about the WHOLE FILE, not about one turn, and a single turn's arithmetic looks identical whether
or not a cross-turn snapshot was carried. The test that can fail is therefore a MULTI-TURN reward
STREAM, which is what this module runs.

**The oracle is the shadow twin, not a golden file.** ``Gen3RewardManager(_shadow=True)`` disables
every fast path — ``_skip_inactive_bias`` and ``_terminal_only`` alike — so a twin on the same
config IS the un-short-circuited implementation: the "before" arm, in-process, with no recorded
constants to rot and nothing to re-bless when a weight changes. It is the same oracle
``GEN3AI_REWARD_VERIFY=1`` wires up in production.

**Two tiers, on purpose.** Here: the DERIVATION (the flag comes from the census, not from a second
`hand_shaping` predicate), the multi-turn stream over the shared board fakes, and the ``reward/``
export shape. On real bridge battles: ``reward_skip_parity_fuzz_test.py``, which now carries this
composition as a fourth arm and compares every field of every turn across hundreds of decisions.
Neither replaces the other — the fuzz has the boards, this has the corners and runs in the routine
gate.

Run:
    python -m pytest src/agents/training/reward_terminal_only_skip_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import unittest

from agents.training.progress_clock import ProgressClock
from agents.training.reward_manager import (
    Gen3RewardManager,
    RewardBreakdown,
    RewardClass,
    RewardConfig,
    reward_class_composition,
)
from agents.training.reward_term_stats import (
    merge_drained as _merge_drained, reward_term_metrics as _term_metrics,
    term_class_map as _term_class_map)
from agents.training.reward_test_fakes import _Battle, _delta, _full_team_live
from agents.training.reward_weights import _TIMEOUT_TURN_CAP

#: The win-prob arm's composition — the reward this short circuit exists for.
TERMINAL_ONLY = RewardConfig(hand_shaping=False, terminal_indicator=True,
                             victory_value=1.0, draw_penalty=0.0)
#: The shaped production default, which must be untouched.
SHAPED = RewardConfig()


def _switchable(live):
    """Give both sides a ``.get(species)`` lookup.

    The shared `reward_test_fakes._Side` has no `get` — nothing that used those fakes had ever
    folded a VOLUNTARY switch, so `_compute_pivot_bonus` / `_compute_sleep_out_bonus` (which read
    `live.ours.get(delta.our_prev_active)`) had never run against them. Added HERE, locally, rather
    than to the shared fake: this test needs a switch turn because a switch is where the short
    circuit skips the most cross-turn state, and widening a fixture that a dozen other reward tests
    build on is a bigger change than this one is entitled to make.
    """
    for side in (live.ours, live.opp):
        by_species = {m.species: m for m in side.mons}
        side.get = by_species.get      # type: ignore[attr-defined]
    return live


def _episode():
    """A 7-turn episode ending in a WIN, as ``(battle, delta, meta)`` triples.

    Chosen to FIRE every cross-turn snapshot the short circuit drops — a voluntary switch
    (``_apply_switch_outcome`` / ``_last_opp_seen_by`` / ``_prev_safe_pivot``), repeated attacks
    (``_last_attack_had_effect`` → the escalating ``repetition_tax``), opponent HP loss and a
    faint (``_prev_opp_boosts`` / ``_prev_opp_se_threat`` / ``_prev_*_statused``) — because a
    stream that never reaches them would agree for the wrong reason.
    """
    atk = {"type": "ATTACK"}
    return [
        # 1-2: two identical attacks. The second is the repeat whose tax depends on turn 1's
        #      `_last_attack_had_effect` — the purest cross-turn carry in the manager.
        (_Battle(_full_team_live(), turn=1), _delta(our_move_id="spikes"), atk),
        (_Battle(_full_team_live(opp_hp=0.9), turn=2), _delta(our_move_id="spikes"), atk),
        # 3: a VOLUNTARY switch — settles at outcome time against `delta.our_switch_to`.
        (_Battle(_switchable(_full_team_live(opp_hp=0.9)), turn=3),
         _delta(our_switch_to="mon", our_prev_active="mon"),
         {"type": "VOLUNTARY", "target_species": "mon", "decision_turn": 3,
          "switch_from": "mon"}),
        # 4: an attack that damages — flips `_last_attack_had_effect` back to True.
        (_Battle(_full_team_live(opp_hp=0.6), turn=4), _delta(our_move_id="seismictoss"), atk),
        # 5: we take damage and lose a mon.
        (_Battle(_full_team_live(our_alive=5, opp_hp=0.6), turn=5), _delta(we_fainted=True), atk),
        # 6: the opponent loses one.
        (_Battle(_full_team_live(our_alive=5, opp_alive=5), turn=6),
         _delta(opp_fainted=True), atk),
        # 7: TERMINAL — a decisive WIN, well before the turn cap (a cap-turn board is the
        #    forfeit/timeout case, which is a different terminal branch entirely).
        (_Battle(_full_team_live(our_alive=5, opp_alive=0, won=True, finished=True), turn=40),
         _delta(opp_fainted=True), atk),
    ]


def _stream(config, *, shadow: bool):
    """Play the episode through one manager; return ``(rewards, breakdown dicts)``.

    ``shadow=True`` is the un-short-circuited implementation. A real ``ProgressClock`` is shared
    in BOTH arms — without one, ``_apply_progress_clock`` early-returns and ``no_progress_tax``,
    the one BIAS term the shaped composition still charges, would be structurally 0 all run, so
    the shaped control below would be comparing zeros to zeros.
    """
    mgr = Gen3RewardManager(config=config, progress_clock=ProgressClock(), _shadow=shadow)
    rewards, dicts = [], []
    for battle, delta, meta in _episode():
        mgr._last_reward_metadata = dict(meta)
        rewards.append(mgr.process_turn_reward(battle, delta))
        # EVERY field, not `to_dict()` — that render is SPARSE (non-zero fields only, grouped
        # into strings), so two breakdowns that differ only in a field one of them zeroed
        # would compare EQUAL. A parity test cannot be written against a lossy view.
        bd = mgr._last_breakdown
        dicts.append({n: getattr(bd, n) for n in RewardBreakdown.field_names()})
    return rewards, dicts


class TheGateIsTheCensus(unittest.TestCase):
    """The short circuit must key on `reward_class_composition` — the ONE announcer the startup
    line, the `reward/` export and the folds' own gates already share. A second hand-written
    `hand_shaping` predicate is the exact drift `_bias_term_active` was created to end."""

    def test_terminal_only_composition_sets_the_flag(self):
        comp = reward_class_composition(TERMINAL_ONLY)
        self.assertEqual((comp["pbrs"], comp["bias"]), (0, 0))
        self.assertTrue(Gen3RewardManager(config=TERMINAL_ONLY)._terminal_only)

    def test_the_shaped_production_composition_does_not(self):
        comp = reward_class_composition(SHAPED)
        self.assertGreater(comp["pbrs"] + comp["bias"], 0)
        self.assertFalse(Gen3RewardManager(config=SHAPED)._terminal_only)

    def test_the_flag_agrees_with_the_census_on_every_documented_composition(self):
        """The load-bearing invariant, swept: `_terminal_only` is TRUE exactly when the census
        reports no PBRS and no BIAS term — for any config, not just the two above."""
        cases = {
            "production (default)": SHAPED,
            "--no-all-shaping-pbrs": RewardConfig(all_shaping_pbrs=False),
            "--stall-pbrs": RewardConfig(all_shaping_pbrs=True, stall_pbrs=True),
            "--no-hand-shaping": RewardConfig(hand_shaping=False),
            "win-prob arm": TERMINAL_ONLY,
            "--no-hand-shaping + re-armed tilt": RewardConfig(
                hand_shaping=False, terminal_indicator=True,
                no_progress_tax_armed=True, bias_redesign=True),
        }
        for label, cfg in cases.items():
            with self.subTest(composition=label):
                comp = reward_class_composition(cfg)
                expected = comp["pbrs"] == 0 and comp["bias"] == 0
                self.assertEqual(Gen3RewardManager(config=cfg)._terminal_only, expected)

    def test_a_re_armed_anti_stall_tilt_defeats_the_short_circuit(self):
        """`--arm-no-progress-tax` revives ONE BIAS term under `--no-hand-shaping`. The reward is
        then no longer terminal-only and every skipped cross-turn carry must come back — which is
        precisely what a flag reading `hand_shaping` instead of the census would get wrong."""
        cfg = RewardConfig(hand_shaping=False, terminal_indicator=True,
                           no_progress_tax_armed=True, bias_redesign=True)
        if reward_class_composition(cfg)["bias"] == 0:
            self.skipTest("this build does not re-arm the tilt under --no-hand-shaping")
        self.assertFalse(Gen3RewardManager(config=cfg)._terminal_only)

    def test_the_shadow_twin_never_short_circuits(self):
        """The oracle has to compute everything, or it is not an oracle."""
        twin = Gen3RewardManager(config=TERMINAL_ONLY, _shadow=True)
        self.assertFalse(twin._terminal_only)
        self.assertFalse(twin._skip_inactive_bias)


class TheRewardStreamIsUnchanged(unittest.TestCase):
    """The claim the owner is actually buying: the reward is byte-identical, only cheaper."""

    def test_terminal_only_stream_matches_the_full_computation_step_for_step(self):
        fast, fast_d = _stream(TERMINAL_ONLY, shadow=False)
        full, full_d = _stream(TERMINAL_ONLY, shadow=True)
        self.assertEqual(fast, full, "the TERMINAL-only short circuit moved the reward")
        for i, (a, b) in enumerate(zip(fast_d, full_d)):
            self.assertEqual(a, b, f"breakdown diverged at episode turn {i + 1}")

    def test_the_shaped_stream_matches_the_full_computation_step_for_step(self):
        """The other composition must be untouched — the short circuit is opt-in by census, and
        this is the arm that would catch a guard accidentally left unconditional."""
        fast, fast_d = _stream(SHAPED, shadow=False)
        full, full_d = _stream(SHAPED, shadow=True)
        self.assertEqual(fast, full, "the shaped composition changed")
        for i, (a, b) in enumerate(zip(fast_d, full_d)):
            self.assertEqual(a, b, f"breakdown diverged at episode turn {i + 1}")

    def test_the_shaped_arm_actually_charges_something(self):
        """A guard on the guard. Both comparisons above pass trivially if the episode fires
        nothing, so the shaped arm on the SAME episode must produce a non-zero stream."""
        rewards, _ = _stream(SHAPED, shadow=False)
        self.assertTrue(any(r != 0.0 for r in rewards[:-1]),
                        "the scripted episode charges nothing pre-terminal — the parity "
                        "assertions above would be vacuous")

    def test_terminal_only_pays_the_win_and_nothing_else(self):
        """The composition's whole definition: 0.0 every step, `victory_value` on the win."""
        rewards, dicts = _stream(TERMINAL_ONLY, shadow=False)
        self.assertEqual(rewards[:-1], [0.0] * (len(rewards) - 1))
        self.assertEqual(rewards[-1], TERMINAL_ONLY.victory_value)
        for turn, d in enumerate(dicts, start=1):
            for cls in (RewardClass.PBRS, RewardClass.BIAS):
                for name in RewardBreakdown.registry_fields(cls):
                    self.assertEqual(d[name], 0.0,
                                     f"{cls.name} term {name} emitted at turn {turn}")

    def test_a_terminal_loss_still_folds_under_the_indicator(self):
        """The win INDICATOR pays 0.0 on every non-win terminal — the branch that makes
        `V(s) == P(win|s)`. Skipping the shaping must not disturb it."""
        mgr = Gen3RewardManager(config=TERMINAL_ONLY, progress_clock=ProgressClock())
        mgr.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        r = mgr.process_turn_reward(
            _Battle(_full_team_live(our_alive=0, opp_alive=3, lost=True, finished=True), turn=40),
            _delta(we_fainted=True))
        self.assertEqual(r, 0.0)
        self.assertEqual(mgr._last_breakdown.win_loss, 0.0)

    def test_a_timeout_still_folds_under_the_indicator(self):
        """The 250-turn forfeit is detected by TURN COUNT, not by won/lost — a branch that reads
        `live.turn`, which the short circuit must not have stopped populating."""
        mgr = Gen3RewardManager(config=TERMINAL_ONLY, progress_clock=ProgressClock())
        mgr.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        r = mgr.process_turn_reward(
            _Battle(_full_team_live(our_alive=2, opp_alive=2, lost=True, finished=True),
                    turn=_TIMEOUT_TURN_CAP),
            _delta())
        self.assertEqual(r, 0.0)

    def test_the_breakdown_survives_for_the_trace_recorder(self):
        """`battle_recorder` writes `breakdown.to_dict()` into EVERY eval-trace decision, so a
        short circuit that stopped populating `_last_breakdown` would silently blank the traces
        rather than fail anything."""
        mgr = Gen3RewardManager(config=TERMINAL_ONLY, progress_clock=ProgressClock())
        for battle, delta, meta in _episode():
            mgr._last_reward_metadata = dict(meta)
            mgr.process_turn_reward(battle, delta)
            self.assertIsNotNone(mgr._last_breakdown)
            # `to_dict()` is what battle_recorder writes; it always carries "total".
            self.assertIn("total", mgr._last_breakdown.to_dict())


class TheRewardTermExportStaysHonest(unittest.TestCase):
    """`reward/*` must report a real window of ZEROS. An ABSENT series and an all-zero one look
    identical on a chart and mean opposite things — the same reason `--distill-anchor-monitor`
    was made default-on."""

    def _drain(self, config):
        mgr = Gen3RewardManager(config=config, progress_clock=ProgressClock())
        for battle, delta, meta in _episode():
            mgr._last_reward_metadata = dict(meta)
            mgr.process_turn_reward(battle, delta)
        # The exact two-step the `reward/` callback runs: merge the workers' drains, then
        # render the scalars against the composition's own class map.
        return _term_metrics(_merge_drained([mgr.drain_reward_terms()]),
                             _term_class_map(reward_class_composition(config)))

    def test_every_decision_is_counted_under_terminal_only(self):
        sc = self._drain(TERMINAL_ONLY)
        self.assertEqual(sc["n_decisions"], float(len(_episode())),
                         "the export lost decisions to the short circuit")

    def test_the_residual_is_exactly_zero_under_terminal_only(self):
        """`untracked_abs_mean` is `mean |bd.total - sum(tracked)|` — non-zero means the census
        and the folds disagree about what this config emits, which is the defect class the
        short circuit is most able to introduce (skip a compute the census still counts)."""
        self.assertEqual(self._drain(TERMINAL_ONLY)["untracked_abs_mean"], 0.0)

    def test_no_shaping_term_reports_anything_under_terminal_only(self):
        sc = self._drain(TERMINAL_ONLY)
        moved = {k: v for k, v in sc.items()
                 if k.startswith(("pbrs_", "bias_")) and v not in (0.0, 0)}
        self.assertEqual(moved, {}, f"shaping terms non-zero under TERMINAL-only: {moved}")

    def test_the_shaped_composition_still_exports_a_moving_stream(self):
        """The control: the same episode under the shaped reward must move the export, or the
        three assertions above are measuring an episode that does nothing."""
        sc = self._drain(SHAPED)
        self.assertEqual(sc["n_decisions"], float(len(_episode())))
        self.assertGreater(sc["total_abs_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
