import unittest
from unittest.mock import MagicMock
import numpy as np
from agents.training.reward_manager import (
    Gen3RewardManager,
    SWITCH_BASE_BONUS, SE_SWITCH_BONUS, MATCHUP_PENALTY,
    SPIKES_LAYER_BONUS, SPIKES_WASTE_PENALTY, FAILED_ROAR_PENALTY,
    FUTILE_ATTACK_PENALTY, HP_VALUE, FAINTED_VALUE, VICTORY_VALUE,
)
from agents.training.battle_context import BattleContext, TurnDelta
from utils.logging.levels import LogLevel


# ---------------------------------------------------------------------------
# Context / delta / battle factories
# ---------------------------------------------------------------------------

def _ctx(
    turn=1,
    our_active="pikachu",
    opp_active="charizard",
    slot_map=None,
    phase="move_selection",
    action_mask=None,
):
    """Build a minimal BattleContext.

    slot_map: {species: slot_index} for OUR team. Defaults to {our_active: 0}.
    """
    mask = action_mask if action_mask is not None else np.ones(11, dtype=np.int8)
    hp = np.ones(6, dtype=np.float32)
    our_slot_map = slot_map if slot_map is not None else {our_active: 0}
    return BattleContext(
        turn=turn,
        phase=phase,
        mask=mask,
        obs=np.zeros(1, dtype=np.float32),
        our_slot_map=our_slot_map,
        opp_slot_map={opp_active: 0},
        our_hp=hp.copy(),
        opp_hp=hp.copy(),
        our_active=our_active,
        opp_active=opp_active,
        our_fainted_count=0,
        opp_fainted_count=0,
        active_move_ids=[None, None, None, None],
        opp_last_move_id=None,
        opp_all_last_move_ids={},
        opp_active_revealed_moves=frozenset(),
        our_cant_reason=None,
        opp_cant_reason=None,
        our_boosts=np.zeros(7, dtype=np.int8),
        opp_boosts=np.zeros(7, dtype=np.int8),
        our_last_effectiveness=None,
        opp_last_effectiveness=None,
        we_moved_first=None,
    )


def _delta(
    our_hp_delta=0.0,
    opp_hp_delta=0.0,
    we_fainted=False,
    opp_fainted=False,
    our_switch_to=None,
    our_move_id=None,
    opp_switch_to=None,
    opp_move_id=None,
    our_prev_active="pikachu",
    opp_prev_active="charizard",
):
    our = np.zeros(6, dtype=np.float32)
    opp = np.zeros(6, dtype=np.float32)
    our[0] = our_hp_delta
    opp[0] = opp_hp_delta
    return TurnDelta(
        our_move_id=our_move_id,
        our_switch_to=our_switch_to,
        our_prev_active=our_prev_active,
        opp_move_id=opp_move_id,
        opp_switch_to=opp_switch_to,
        opp_prev_active=opp_prev_active,
        opp_move_known=opp_move_id is not None,
        our_hp_delta=our,
        opp_hp_delta=opp,
        we_fainted=we_fainted,
        opp_fainted=opp_fainted,
        our_failed_to_move=False,
        our_cant_reason=None,
        opp_failed_to_move=False,
        opp_cant_reason=None,
        our_boost_delta=np.zeros(7, dtype=np.int8),
        opp_boost_delta=np.zeros(7, dtype=np.int8),
        our_effectiveness=None,
        opp_effectiveness=None,
        we_moved_first=None,
    )


def _battle(won=False, lost=False, finished=False, opp_spikes=0,
            our_mon=None, opp_mon=None):
    battle = MagicMock()
    battle.won = won
    battle.lost = lost
    battle.finished = finished
    battle.turn = 1
    battle.opponent_team = {}
    battle.active_pokemon = our_mon
    battle.opponent_active_pokemon = opp_mon
    from poke_env.battle.side_condition import SideCondition
    battle.opponent_side_conditions = (
        {SideCondition.SPIKES: opp_spikes} if opp_spikes > 0 else {}
    )
    return battle


def _make_mon(type1_name, type2_name=None, moves=None, status=None, ability=None):
    """Build a mock Pokemon with the given types and optional revealed moves."""
    from poke_env.data import GenData
    type_chart = GenData.from_gen(3).type_chart

    # Use real PokemonType objects so damage_multiplier works correctly
    from poke_env.battle.pokemon_type import PokemonType
    type1 = PokemonType[type1_name.upper()]
    type2 = PokemonType[type2_name.upper()] if type2_name else None

    mon = MagicMock()
    mon.type_1 = type1
    mon.type_2 = type2
    mon.status = status
    mon.ability = ability  # e.g. "levitate", "voltabsorb"
    mon.boosts = {}

    if moves:
        mon.moves = {}
        for move_id, move_type_name, base_power in moves:
            move = MagicMock()
            move.base_power = base_power
            move.type = PokemonType[move_type_name.upper()]
            mon.moves[move_id] = move
    else:
        mon.moves = {}

    return mon


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRewardManagerBasics(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_attack_tracking(self):
        self.manager.record_action(_ctx(turn=1), 6)
        self.assertEqual(self.manager.attack_count, 1)
        reward = self.manager.process_turn_reward(_battle(), _delta())
        self.assertIsInstance(reward, float)
        self.assertEqual(self.manager.total_reward, reward)

    def test_repetition_tax(self):
        self.manager.record_action(_ctx(turn=1), 6)
        r1 = self.manager.process_turn_reward(_battle(), _delta())

        self.manager.record_action(_ctx(turn=2), 6)  # same action
        r2 = self.manager.process_turn_reward(_battle(), _delta())

        self.assertAlmostEqual(r2, r1 - 0.02, places=5)

    def test_faint_reward(self):
        self.manager.record_action(_ctx(turn=1), 6)
        reward = self.manager.process_turn_reward(_battle(), _delta(opp_fainted=True))
        self.assertAlmostEqual(reward, FAINTED_VALUE, places=5)

    def test_win_reward(self):
        self.manager.record_action(_ctx(turn=1), 6)
        reward = self.manager.process_turn_reward(_battle(won=True, finished=True), _delta())
        self.assertAlmostEqual(reward, VICTORY_VALUE, places=5)

    def test_hp_delta_reward(self):
        self.manager.record_action(_ctx(turn=1), 6)
        reward = self.manager.process_turn_reward(_battle(), _delta(our_hp_delta=-0.5))
        self.assertAlmostEqual(reward, -0.5 * HP_VALUE, places=5)


class TestSwitchSubsidy(unittest.TestCase):
    """Switch subsidy must be credited in the SAME turn as the switch action."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_voluntary_switch_subsidy_same_turn(self):
        # Slot map: pikachu=0, raichu=1.  Action 1 = switch to raichu.
        ctx = _ctx(
            turn=1, our_active="pikachu",
            slot_map={"pikachu": 0, "raichu": 1},
        )
        self.manager.record_action(ctx, 1)  # switch to slot 1 (raichu)
        reward = self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))

        # No HP change → reward should equal the switch subsidy alone
        self.assertAlmostEqual(reward, SWITCH_BASE_BONUS, places=5)
        self.assertEqual(self.manager.switch_count, 1)

    def test_voluntary_switch_spam_mult_zero(self):
        # Two voluntary switches on consecutive turns → second gets spam_mult=0
        ctx1 = _ctx(turn=1, our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx1, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))

        ctx2 = _ctx(turn=2, our_active="raichu", slot_map={"raichu": 0, "starmie": 1})
        self.manager.record_action(ctx2, 1)
        r2 = self.manager.process_turn_reward(_battle(), _delta(our_switch_to="starmie"))

        # spam_mult=0 → no subsidy, no HP → reward should be 0
        self.assertAlmostEqual(r2, 0.0, places=5)

    def test_bouncing_tax(self):
        # Switch pikachu→raichu then raichu→pikachu triggers bouncing tax
        ctx1 = _ctx(turn=1, our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx1, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))

        # Turn 3 (skip a turn so spam_mult=1): switch back to pikachu
        ctx3 = _ctx(turn=3, our_active="raichu", slot_map={"raichu": 1, "pikachu": 0})
        self.manager.record_action(ctx3, 0)  # action 0 = slot 0 = pikachu
        r3 = self.manager.process_turn_reward(_battle(), _delta(our_switch_to="pikachu"))

        # subsidy (+0.5) - bouncing_tax (-0.15) = +0.35
        self.assertAlmostEqual(r3, SWITCH_BASE_BONUS - 0.15, places=5)

    def test_forced_faint_switch_no_subsidy(self):
        ctx = _ctx(turn=1, our_active="NONE", phase="forced_switch",
                   slot_map={"raichu": 1})
        self.manager.record_action(ctx, 1)
        reward = self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        self.assertAlmostEqual(reward, 0.0, places=5)
        self.assertEqual(self.manager.forced_switch_count, 1)
        self.assertEqual(self.manager.switch_count, 0)

    def test_roar_forced_switch_no_subsidy_and_sets_flag(self):
        # Forced switch while we still have a live active mon → roar
        ctx = _ctx(turn=1, our_active="pikachu", phase="forced_switch",
                   slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx, 1)
        self.assertTrue(self.manager._last_switch_was_roared)
        reward = self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        # No subsidy → reward = 0 (no HP change, no other signals)
        self.assertAlmostEqual(reward, 0.0, places=5)
        self.assertEqual(self.manager.forced_switch_count, 1)
        self.assertEqual(self.manager.switch_count, 0)


class TestSeSwitchBonus(unittest.TestCase):
    """SE switch bonus fires for both revealed moves and type-advantage fallback."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _do_switch(self, our_mon, opp_mon):
        """Trigger one switch turn and return the reward."""
        ctx = _ctx(turn=1, our_active="incoming", slot_map={"incoming": 1, "prev": 0})
        self.manager.record_action(ctx, 1)
        battle = _battle(our_mon=our_mon, opp_mon=opp_mon)
        delta = _delta(our_switch_to="incoming")
        return self.manager.process_turn_reward(battle, delta)

    def test_se_bonus_with_revealed_move(self):
        # Rock move (Rock type) vs Zapdos (Electric/Flying) → SE (2× on Flying)
        our = _make_mon("ROCK", moves=[("rockslide", "ROCK", 75)])
        opp = _make_mon("ELECTRIC", "FLYING")
        reward = self._do_switch(our, opp)
        self.assertAlmostEqual(reward, SWITCH_BASE_BONUS + SE_SWITCH_BONUS, places=5)

    def test_se_bonus_type_fallback_no_moves_revealed(self):
        # Tyranitar (Rock/Dark) vs Zapdos (Electric/Flying): Rock STAB is 2× vs Flying
        # No moves revealed yet — should still fire via type fallback
        our = _make_mon("ROCK", "DARK")  # no moves
        opp = _make_mon("ELECTRIC", "FLYING")
        reward = self._do_switch(our, opp)
        self.assertAlmostEqual(reward, SWITCH_BASE_BONUS + SE_SWITCH_BONUS, places=5)

    def test_no_se_bonus_when_not_super_effective(self):
        # Normal type vs Normal type: 1× — no bonus
        our = _make_mon("NORMAL")
        opp = _make_mon("NORMAL")
        reward = self._do_switch(our, opp)
        self.assertAlmostEqual(reward, SWITCH_BASE_BONUS, places=5)

    def test_no_se_bonus_on_attack(self):
        # SE bonus only fires on switches, not attacks
        our = _make_mon("ROCK", "DARK")
        opp = _make_mon("ELECTRIC", "FLYING")
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        battle = _battle(our_mon=our, opp_mon=opp)
        reward = self.manager.process_turn_reward(battle, _delta())
        self.assertAlmostEqual(reward, 0.0, places=5)

    def test_roar_forced_switch_skips_se_bonus(self):
        # When roared out, SE switch bonus must be skipped even if type advantage exists
        ctx = _ctx(turn=1, our_active="pikachu", phase="forced_switch",
                   slot_map={"pikachu": 0, "tyranitar": 1})
        self.manager.record_action(ctx, 1)
        our = _make_mon("ROCK", "DARK")
        opp = _make_mon("ELECTRIC", "FLYING")
        battle = _battle(our_mon=our, opp_mon=opp)
        reward = self.manager.process_turn_reward(battle, _delta(our_switch_to="tyranitar"))
        # No subsidy, no SE bonus → 0
        self.assertAlmostEqual(reward, 0.0, places=5)


class TestSpikes(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_spikes_layer_bonus(self):
        self.manager.record_action(_ctx(turn=1), 6)
        reward = self.manager.process_turn_reward(_battle(opp_spikes=1), _delta())
        self.assertAlmostEqual(reward, SPIKES_LAYER_BONUS, places=5)

    def test_spikes_two_layers_sequential(self):
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager.process_turn_reward(_battle(opp_spikes=1), _delta())
        self.manager.record_action(_ctx(turn=2), 7)
        reward = self.manager.process_turn_reward(_battle(opp_spikes=2), _delta())
        self.assertAlmostEqual(reward, SPIKES_LAYER_BONUS, places=5)

    def test_spikes_waste_penalty(self):
        self.manager._prev_opp_spikes = 3
        self.manager.record_action(_ctx(turn=1), 6)
        delta = TurnDelta(
            our_move_id="spikes", our_switch_to=None, our_prev_active="pikachu",
            opp_move_id=None, opp_switch_to=None, opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
        )
        reward = self.manager.process_turn_reward(_battle(opp_spikes=3), delta)
        self.assertAlmostEqual(reward, SPIKES_WASTE_PENALTY, places=5)

    def test_no_spikes_bonus_without_change(self):
        self.manager.record_action(_ctx(turn=1), 6)
        reward = self.manager.process_turn_reward(_battle(opp_spikes=0), _delta())
        self.assertAlmostEqual(reward, 0.0, places=5)


class TestMatchupPenalty(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _se_battle(self):
        """Battle where opponent has a revealed SE move vs our mon."""
        our = _make_mon("ELECTRIC", "FLYING")   # Zapdos
        opp = _make_mon("ROCK")
        # Opp has Rock move — 2× vs Flying
        opp_move = MagicMock()
        opp_move.base_power = 80
        from poke_env.battle.pokemon_type import PokemonType
        opp_move.type = PokemonType.ROCK
        opp.moves = {"rockblast": opp_move}
        return _battle(our_mon=our, opp_mon=opp)

    def test_matchup_penalty_fires_on_second_turn(self):
        # Turn 1: opp reveals SE move — penalty should NOT fire yet (unknown at decision time)
        self.manager.record_action(_ctx(turn=1), 6)
        r1 = self.manager.process_turn_reward(self._se_battle(), _delta())
        self.assertAlmostEqual(r1, 0.0, places=5)

        # Turn 2: threat is now known — penalty fires
        self.manager.record_action(_ctx(turn=2), 7)
        r2 = self.manager.process_turn_reward(self._se_battle(), _delta())
        self.assertAlmostEqual(r2, MATCHUP_PENALTY, places=5)

    def test_matchup_penalty_skipped_on_switch(self):
        self.manager._prev_opp_se_threat = True
        self.manager.record_action(
            _ctx(turn=1, our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1}), 1
        )
        r = self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        # subsidy (+0.5) with no penalty → must be > 0
        self.assertGreater(r, 0.0)

    def test_matchup_penalty_absent_when_no_threat(self):
        self.manager._prev_opp_se_threat = False
        self.manager.record_action(_ctx(turn=1), 6)
        r = self.manager.process_turn_reward(_battle(), _delta())
        self.assertAlmostEqual(r, 0.0, places=5)

    def test_matchup_penalty_skipped_when_levitate_blocks_ground(self):
        # Gengar is Ghost/Poison — Poison is 2× weak to Ground — but Levitate grants immunity.
        # The SE threat snapshot should NOT fire, so no matchup penalty next turn.
        gengar = _make_mon("GHOST", "POISON", ability="levitate")
        ttar = _make_mon("ROCK", "DARK")
        eq_move = MagicMock()
        from poke_env.battle.pokemon_type import PokemonType
        eq_move.base_power = 100
        eq_move.type = PokemonType.GROUND
        ttar.moves = {"earthquake": eq_move}

        battle = _battle(our_mon=gengar, opp_mon=ttar)
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager.process_turn_reward(battle, _delta())   # snapshots threat
        # Next turn: threat should NOT have been flagged
        self.assertFalse(self.manager._prev_opp_se_threat)

    def test_matchup_penalty_still_fires_for_non_immune_type(self):
        # Gengar has Levitate but not immunity to all moves; Rock slide still 2× vs Flying.
        # Confirm non-Ground SE moves still flag the threat.
        zapdos = _make_mon("ELECTRIC", "FLYING", ability="levitate")
        opp = _make_mon("ROCK")
        rock_move = MagicMock()
        from poke_env.battle.pokemon_type import PokemonType
        rock_move.base_power = 75
        rock_move.type = PokemonType.ROCK
        opp.moves = {"rockslide": rock_move}

        battle = _battle(our_mon=zapdos, opp_mon=opp)
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager.process_turn_reward(battle, _delta())
        self.assertTrue(self.manager._prev_opp_se_threat)


def _pivot_battle(new_mon, opp_mon, prev_species="prevmon", prev_mon=None):
    """Build a minimal battle mock for pivot bonus tests."""
    battle = MagicMock()
    battle.won = False
    battle.lost = False
    battle.finished = False
    battle.turn = 1
    battle.opponent_team = {}
    battle.active_pokemon = new_mon
    battle.opponent_active_pokemon = opp_mon
    from poke_env.battle.side_condition import SideCondition
    battle.opponent_side_conditions = {}
    if prev_mon is not None:
        prev_mon.species = prev_species
        battle.team = {prev_species: prev_mon}
    else:
        battle.team = {}
    return battle


def _pivot_delta(opp_move_id, opp_switch_to=None, prev_species="prevmon"):
    return TurnDelta(
        our_move_id=None, our_switch_to="newmon", our_prev_active=prev_species,
        opp_move_id=opp_move_id, opp_switch_to=opp_switch_to,
        opp_prev_active="opponentmon", opp_move_known=opp_move_id is not None,
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
        we_fainted=False, opp_fainted=False,
        our_failed_to_move=False, our_cant_reason=None,
        opp_failed_to_move=False, opp_cant_reason=None,
        our_boost_delta=np.zeros(7, dtype=np.int8),
        opp_boost_delta=np.zeros(7, dtype=np.int8),
        our_effectiveness=None,
        opp_effectiveness=None,
        we_moved_first=None,
    )


def _type_mon(type1_name, type2_name=None, status=None, ability=None):
    from poke_env.battle.pokemon_type import PokemonType
    mon = MagicMock()
    mon.type_1 = PokemonType[type1_name.upper()]
    mon.type_2 = PokemonType[type2_name.upper()] if type2_name else None
    mon.status = status
    mon.ability = ability
    mon.moves = {}
    return mon


def _opp_with_status_move(move_id):
    move = MagicMock()
    move.base_power = 0
    opp = MagicMock()
    opp.moves = {move_id: move}
    return opp


class TestPivotProtect(unittest.TestCase):
    """_pivot_protect_bonus: fires on Protect/Detect/Endure."""

    def _run(self, move_id):
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        delta = _pivot_delta(move_id)
        battle = _pivot_battle(_type_mon("NORMAL"), _opp_with_status_move(move_id))
        return sum(manager._compute_pivot_bonus(delta, battle))

    def test_protect(self):
        from agents.training.reward_manager import PROTECT_SWITCH_BONUS
        self.assertAlmostEqual(self._run("protect"), PROTECT_SWITCH_BONUS, places=5)

    def test_detect(self):
        from agents.training.reward_manager import PROTECT_SWITCH_BONUS
        self.assertAlmostEqual(self._run("detect"), PROTECT_SWITCH_BONUS, places=5)

    def test_endure(self):
        from agents.training.reward_manager import PROTECT_SWITCH_BONUS
        self.assertAlmostEqual(self._run("endure"), PROTECT_SWITCH_BONUS, places=5)


class TestPivotStatus(unittest.TestCase):
    """_pivot_status_bonus: type and already-statused immunity."""

    def _run(self, move_id, new_mon):
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        delta = _pivot_delta(move_id)
        battle = _pivot_battle(new_mon, _opp_with_status_move(move_id))
        return sum(manager._compute_pivot_bonus(delta, battle))

    def test_ground_immune_to_thunderwave(self):
        from agents.training.reward_manager import STATUS_IMMUNE_SWITCH_BONUS
        self.assertAlmostEqual(
            self._run("thunderwave", _type_mon("GROUND")),
            STATUS_IMMUNE_SWITCH_BONUS, places=5,
        )

    def test_steel_immune_to_toxic(self):
        from agents.training.reward_manager import STATUS_IMMUNE_SWITCH_BONUS
        self.assertAlmostEqual(
            self._run("toxic", _type_mon("STEEL")),
            STATUS_IMMUNE_SWITCH_BONUS, places=5,
        )

    def test_poison_immune_to_toxic(self):
        from agents.training.reward_manager import STATUS_IMMUNE_SWITCH_BONUS
        self.assertAlmostEqual(
            self._run("toxic", _type_mon("POISON")),
            STATUS_IMMUNE_SWITCH_BONUS, places=5,
        )

    def test_fire_immune_to_willowisp(self):
        from agents.training.reward_manager import STATUS_IMMUNE_SWITCH_BONUS
        self.assertAlmostEqual(
            self._run("willowisp", _type_mon("FIRE")),
            STATUS_IMMUNE_SWITCH_BONUS, places=5,
        )

    def test_already_statused_mon(self):
        from agents.training.reward_manager import STATUS_IMMUNE_SWITCH_BONUS
        from poke_env.battle.status import Status
        self.assertAlmostEqual(
            self._run("thunderwave", _type_mon("NORMAL", status=Status.PAR)),
            STATUS_IMMUNE_SWITCH_BONUS, places=5,
        )

    def test_normal_type_not_immune_to_toxic(self):
        self.assertAlmostEqual(self._run("toxic", _type_mon("NORMAL")), 0.0, places=5)


class TestPivotDamage(unittest.TestCase):
    """_pivot_damage_bonus: Signal A — actual move effectiveness comparison."""

    def _run(self, new_mon, prev_mon, move_type_name, base_power=80):
        from poke_env.battle.pokemon_type import PokemonType
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)

        move = MagicMock()
        move.base_power = base_power
        move.type = PokemonType[move_type_name.upper()]

        opp = MagicMock()
        opp.moves = {"testmove": move}

        delta = _pivot_delta("testmove")
        battle = _pivot_battle(new_mon, opp, prev_mon=prev_mon)
        return sum(manager._compute_pivot_bonus(delta, battle))

    def test_resist_improvement_earns_bonus(self):
        # Fire move: prev=Normal (1×), new=Water (0.5×) — improvement
        result = self._run(_type_mon("WATER"), _type_mon("NORMAL"), "FIRE")
        self.assertAlmostEqual(result, 0.10, places=5)

    def test_immune_switch_earns_larger_bonus(self):
        # Ground move: prev=Normal (1×), new=Flying (0×) — Flying is immune to Ground
        result = self._run(_type_mon("FLYING"), _type_mon("NORMAL"), "GROUND")
        self.assertAlmostEqual(result, 0.15, places=5)

    def test_no_improvement_no_bonus(self):
        # Fire move: prev=Water (0.5×), new=Grass (2×) — worse matchup
        result = self._run(_type_mon("GRASS"), _type_mon("WATER"), "FIRE")
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_equal_effectiveness_no_bonus(self):
        # Both mons are Water vs Fire move (both 0.5×)
        result = self._run(_type_mon("WATER"), _type_mon("WATER"), "FIRE")
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_off_type_coverage_respected(self):
        # Opp is Normal-type but has Ice Beam: prev=Water (0.5× vs Ice), new=Grass (2×)
        # Grass should get NO bonus — the switch was bad against the actual move used
        result = self._run(_type_mon("GRASS"), _type_mon("WATER"), "ICE")
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_opponent_switched_no_bonus(self):
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        delta = _pivot_delta(None, opp_switch_to="newmon")
        battle = _pivot_battle(_type_mon("WATER"), MagicMock())
        result = sum(manager._compute_pivot_bonus(delta, battle))
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_no_opp_move_id_no_bonus(self):
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        delta = _pivot_delta(None)
        battle = _pivot_battle(_type_mon("WATER"), MagicMock())
        result = sum(manager._compute_pivot_bonus(delta, battle))
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_not_a_switch_no_bonus(self):
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        result = sum(manager._compute_pivot_bonus(_delta(), MagicMock()))
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_levitate_switch_in_vs_ground_earns_immunity_bonus(self):
        # Gengar (Ghost/Poison) has Levitate — Ground move is 0× (immune via ability),
        # even though type chart would say 2× (Poison weakness).
        # prev=Normal (1×), new=Gengar+Levitate (0× via ability) → +0.15
        gengar = _type_mon("GHOST", "POISON", ability="levitate")
        prev = _type_mon("NORMAL")
        result = self._run(gengar, prev, "GROUND")
        self.assertAlmostEqual(result, 0.15, places=5)

    def test_levitate_correct_vs_non_ground_move(self):
        # Levitate only blocks Ground. Ice Beam vs Gengar (Ghost/Poison): normal 0.5×
        # (Ghost resists... actually Normal → Ghost is 0, Poison vs Ice is 1×).
        # Point: Levitate should NOT affect Ice Beam effectiveness.
        gengar = _type_mon("GHOST", "POISON", ability="levitate")
        prev = _type_mon("NORMAL")
        # Ice vs Normal (1×) and Ice vs Ghost/Poison (1×) — no improvement, no bonus
        result = self._run(gengar, prev, "ICE")
        self.assertAlmostEqual(result, 0.0, places=5)


class TestRoarBonus(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_failed_roar_penalty(self):
        self.manager.record_action(_ctx(turn=1), 6)
        delta = TurnDelta(
            our_move_id="roar", our_switch_to=None, our_prev_active="pikachu",
            opp_move_id=None, opp_switch_to=None, opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
        )
        reward = self.manager.process_turn_reward(_battle(), delta)
        self.assertAlmostEqual(reward, FAILED_ROAR_PENALTY, places=5)


class TestFutileAttack(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_futile_attack_penalty_fires_when_opp_net_healed(self):
        self.manager.record_action(_ctx(turn=1), 6)

        move_mock = MagicMock()
        move_mock.base_power = 20
        our_mon = MagicMock()
        our_mon.moves = {"rapidspin": move_mock}
        battle = _battle(our_mon=our_mon)

        opp_hp = np.zeros(6, dtype=np.float32)
        opp_hp[0] = 0.05   # opponent net gained 5% (Leftovers > damage)
        delta = TurnDelta(
            our_move_id="rapidspin", our_switch_to=None, our_prev_active="pikachu",
            opp_move_id="struggle", opp_switch_to=None, opp_prev_active="charizard",
            opp_move_known=True,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=opp_hp,
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
        )
        reward = self.manager.process_turn_reward(battle, delta)
        expected = -0.05 * HP_VALUE + FUTILE_ATTACK_PENALTY
        self.assertAlmostEqual(reward, expected, places=4)

    def test_futile_attack_penalty_skips_status_moves(self):
        self.manager.record_action(_ctx(turn=1), 6)
        move_mock = MagicMock()
        move_mock.base_power = 0
        our_mon = MagicMock()
        our_mon.moves = {"thunderwave": move_mock}

        delta = TurnDelta(
            our_move_id="thunderwave", our_switch_to=None, our_prev_active="pikachu",
            opp_move_id=None, opp_switch_to=None, opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
        )
        reward = self.manager.process_turn_reward(_battle(our_mon=our_mon), delta)
        self.assertAlmostEqual(reward, 0.0, places=5)


class TestOriginalScenario(unittest.TestCase):
    """Reproduce the Zapdos→Tyranitar vs Zapdos matchup from the bug report."""

    def test_ttar_switch_reward_components(self):
        """
        Turn 1: we lead Zapdos, opponent leads Zapdos.
        We switch to Tyranitar (Rock/Dark). Opp uses Thunderbolt → Ttar takes -28%.
        Rock STAB is 2× vs opponent's Flying type.

        Expected reward = HP_delta + switch_subsidy + SE_bonus
                        = (-0.28 * 2) + 0.5 + 0.2 = -0.56 + 0.5 + 0.2 = +0.14
        (The old code produced ≈ -0.66 because subsidy was delayed one turn
         and the SE bonus never fired on an unrevealed mon.)
        """
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)

        # Slot 0 = Zapdos (current active), slot 2 = Tyranitar
        ctx = _ctx(
            turn=1, our_active="zapdos",
            slot_map={"zapdos": 0, "skarmory": 1, "tyranitar": 2,
                      "flygon": 3, "jirachi": 4, "charizard": 5},
        )
        manager.record_action(ctx, 2)  # action 2 = switch to slot 2 = tyranitar

        tyranitar = _make_mon("ROCK", "DARK")          # no revealed moves yet
        opp_zapdos = _make_mon("ELECTRIC", "FLYING")

        our_hp = np.zeros(6, dtype=np.float32)
        our_hp[2] = -0.28   # Tyranitar (slot 2) took 28% damage

        delta = TurnDelta(
            our_move_id=None, our_switch_to="tyranitar", our_prev_active="zapdos",
            opp_move_id="thunderbolt", opp_switch_to=None, opp_prev_active="zapdos",
            opp_move_known=True,
            our_hp_delta=our_hp,
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
        )
        battle = _battle(our_mon=tyranitar, opp_mon=opp_zapdos)

        reward = manager.process_turn_reward(battle, delta)

        expected = (-0.28 * HP_VALUE) + SWITCH_BASE_BONUS + SE_SWITCH_BONUS
        self.assertAlmostEqual(reward, expected, places=4)
        self.assertGreater(reward, 0.0, "should be positive: ttar switch is correct")


class TestRewardBreakdownToDict(unittest.TestCase):
    """RewardBreakdown.to_dict() — grouped compact format."""

    def test_total_always_present(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown()
        d = bd.to_dict()
        self.assertIn("total", d)
        self.assertAlmostEqual(d["total"], 0.0, places=5)

    def test_all_zeros_has_only_total(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown()
        d = bd.to_dict()
        self.assertEqual(list(d.keys()), ["total"])

    def test_base_group_appears_for_hp_delta(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown(hp_ours=-0.64, hp_opp=0.20)
        d = bd.to_dict()
        self.assertIn("base", d)
        self.assertNotIn("attack", d)
        self.assertNotIn("switch", d)
        self.assertNotIn("field", d)
        self.assertIn("hp_ours=-0.64", d["base"])
        self.assertIn("hp_opp=+0.2", d["base"])

    def test_attack_group_appears_for_roar(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown(hp_ours=-0.1, roar=0.2)
        d = bd.to_dict()
        self.assertIn("attack", d)
        self.assertIn("roar=+0.2", d["attack"])
        # base also fires since hp_ours != 0
        self.assertIn("base", d)

    def test_switch_group_contains_multiple_signals(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown(switch_base=0.5, se_switch=0.2, pivot_damage=0.1)
        d = bd.to_dict()
        self.assertIn("switch", d)
        self.assertIn("switch_base=+0.5", d["switch"])
        self.assertIn("se_switch=+0.2", d["switch"])
        self.assertIn("pivot_damage=+0.1", d["switch"])

    def test_field_group_contains_stall_tax(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown(hp_ours=-0.05, stall_tax=-1.5)
        d = bd.to_dict()
        self.assertIn("field", d)
        self.assertIn("stall_tax=-1.5", d["field"])

    def test_group_ordering_is_base_attack_switch_field(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown(
            hp_ours=-0.1, roar=0.2, switch_base=0.5, stall_tax=-0.5,
        )
        d = bd.to_dict()
        keys = [k for k in d.keys() if k != "total"]
        self.assertEqual(keys, ["base", "attack", "switch", "field"])

    def test_zero_fields_within_group_are_omitted(self):
        from agents.training.reward_manager import RewardBreakdown
        # Only faint_opp set in base group; hp_ours/hp_opp/win_loss etc should not appear
        bd = RewardBreakdown(faint_opp=2.0)
        d = bd.to_dict()
        self.assertIn("faint_opp=+2", d["base"])
        self.assertNotIn("hp_ours", d["base"])
        self.assertNotIn("hp_opp", d["base"])

    def test_win_scenario(self):
        from agents.training.reward_manager import RewardBreakdown
        bd = RewardBreakdown(hp_opp=-0.3, faint_opp=2.0, win_loss=30.0)
        d = bd.to_dict()
        self.assertAlmostEqual(d["total"], 31.7, places=4)
        self.assertIn("win_loss=+30", d["base"])
        self.assertIn("faint_opp=+2", d["base"])


class TestStallTax(unittest.TestCase):
    def _reward_at_turn(self, turn):
        from agents.training.reward_manager import Gen3RewardManager
        rm = Gen3RewardManager()
        battle = _battle()
        battle.turn = turn
        delta = _delta()
        return rm.process_turn_reward(battle, delta)

    def test_no_stall_tax_at_or_before_turn_125(self):
        reward = self._reward_at_turn(125)
        self.assertAlmostEqual(reward, 0.0, places=4)

    def test_flat_stall_tax_just_after_threshold(self):
        reward = self._reward_at_turn(126)
        self.assertAlmostEqual(reward, -0.1, places=4)

    def test_flat_stall_tax_is_constant_regardless_of_turn(self):
        r126 = self._reward_at_turn(126)
        r237 = self._reward_at_turn(237)
        self.assertAlmostEqual(r126, r237, places=4)


if __name__ == "__main__":
    unittest.main()
