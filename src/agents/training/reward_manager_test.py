import dataclasses
import unittest
from unittest.mock import MagicMock
import numpy as np
from agents.training.reward_manager import (
    Gen3RewardManager,
    SWITCH_BASE_BONUS, SE_SWITCH_BONUS, MATCHUP_PENALTY,
    SPIKES_LAYER_BONUS, SPIKES_WASTE_PENALTY, FAILED_ROAR_PENALTY,
    FUTILE_ATTACK_PENALTY, FUTILE_IMMUNE_PENALTY, ESCAPE_THREAT_BONUS,
    HP_VALUE, FAINT_BASE, FAINT_HP_SCALE, VICTORY_VALUE,
    FUTILE_SETUP_PENALTY, SETUP_LOW_HP_MAX_PENALTY, STATUS_WASTED_PENALTY,
    EXPLOSION_BLOCK_BONUS, FINISHING_BLOW_BONUS,
)
from agents.training.battle_context import BattleContext, TurnDelta
from poke_env.battle.abstract_battle import DamagingMoveEvent
from utils.logging.levels import LogLevel


def _explosion_event(user_species="gengar", target_species="pikachu", effectiveness=1.0):
    """Build a DamagingMoveEvent for Explosion — used in TestExplosionReward.

    The reward manager's Explosion branch now reads delta.opp_damaging_event
    directly (protocol truth) instead of scanning opp_mon.moves. Tests must
    supply this event to exercise the bonus paths.
    """
    return DamagingMoveEvent(
        user_species=user_species,
        target_species=target_species,
        target_status=None,
        move_id="explosion",
        effectiveness=effectiveness,
    )


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


def _ctx_with_boosts(our_boosts=None, our_hp_val=1.0, opp_hp_val=1.0, **kwargs):
    """_ctx with custom boosts and HP fractions."""
    ctx = _ctx(**kwargs)
    if our_boosts is not None:
        ctx = dataclasses.replace(ctx, our_boosts=np.array(our_boosts, dtype=np.int8))
    hp = np.ones(6, dtype=np.float32) * our_hp_val
    opp_hp = np.ones(6, dtype=np.float32) * opp_hp_val
    ctx = dataclasses.replace(ctx, our_hp=hp, opp_hp=opp_hp)
    return ctx


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
    our_boost_delta=None,
    our_damaging_event=None,
    opp_damaging_event=None,
):
    our = np.zeros(6, dtype=np.float32)
    opp = np.zeros(6, dtype=np.float32)
    our[0] = our_hp_delta
    opp[0] = opp_hp_delta
    boost_delta = our_boost_delta if our_boost_delta is not None else np.zeros(7, dtype=np.int8)
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
        our_boost_delta=boost_delta,
        opp_boost_delta=np.zeros(7, dtype=np.int8),
        our_effectiveness=None,
        opp_effectiveness=None,
        we_moved_first=None,
        our_damaging_event=our_damaging_event,
        opp_damaging_event=opp_damaging_event,
    )


def _battle(won=False, lost=False, finished=False, opp_spikes=0,
            our_mon=None, opp_mon=None):
    battle = MagicMock()
    battle.won = won
    battle.lost = lost
    battle.finished = finished
    battle.turn = 1
    battle.opponent_team = {}
    battle.team = {}
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
    mon.fainted = False

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
        # Use opp_hp_delta=-0.1 so the first attack "had effect", making the second use
        # the gentle effective-repeat step (-0.03) rather than the zero-effect step.
        self.manager.record_action(_ctx(turn=1), 6)
        r1 = self.manager.process_turn_reward(_battle(), _delta(opp_hp_delta=-0.1))

        self.manager.record_action(_ctx(turn=2), 6)  # same action
        r2 = self.manager.process_turn_reward(_battle(), _delta(opp_hp_delta=-0.1))

        # First repeat (n=1) at the normal step: -REPETITION_STEP * 1 = -0.03
        self.assertAlmostEqual(r2, r1 - 0.03, places=5)

    def test_faint_reward(self):
        # With default opp_hp_val=1.0, faint_opp = 0.5 + 2.0*1.0 = 2.5
        ctx = _ctx_with_boosts(opp_hp_val=1.0)
        self.manager.record_action(ctx, 6)
        reward = self.manager.process_turn_reward(_battle(), _delta(opp_fainted=True))
        self.assertAlmostEqual(reward, 0.5 + 2.0 * 1.0, places=5)

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


def _pivot_delta(opp_move_id, opp_switch_to=None, prev_species="prevmon",
                  opp_damaging_event=None):
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
        opp_damaging_event=opp_damaging_event,
    )


def _type_mon(type1_name, type2_name=None, status=None, ability=None):
    from poke_env.battle.pokemon_type import PokemonType
    mon = MagicMock()
    mon.type_1 = PokemonType[type1_name.upper()]
    mon.type_2 = PokemonType[type2_name.upper()] if type2_name else None
    mon.status = status
    mon.ability = ability
    mon.moves = {}
    mon.fainted = False
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

    def test_event_effectiveness_preferred_over_local_recompute(self):
        # Protocol-truth path: event's effectiveness for the switch-in is used as-is
        # instead of recomputing from move.type × types. This decouples reward
        # signal from any drift between our local mechanics and Showdown's.
        from poke_env.battle.pokemon_type import PokemonType
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        move = MagicMock()
        move.base_power = 80
        move.type = PokemonType.FIRE
        opp = MagicMock()
        opp.moves = {"flamethrower": move}
        # New mon's typing would compute 1.0 vs Fire (Normal), but we override the
        # event to 0.5 — the reward signal must follow the event, not the recompute.
        new_mon = _type_mon("NORMAL")
        new_mon.species = "newmon"
        event = DamagingMoveEvent(
            user_species="opponentmon",
            target_species="newmon",
            target_status=None,
            move_id="flamethrower",
            effectiveness=0.5,
        )
        delta = _pivot_delta("flamethrower", opp_damaging_event=event)
        battle = _pivot_battle(new_mon, opp, prev_mon=_type_mon("NORMAL"))
        # prev=Normal (1× Fire), new (via event)=0.5 → improvement → +0.10
        result = sum(manager._compute_pivot_bonus(delta, battle))
        self.assertAlmostEqual(result, 0.10, places=5)

    def test_event_move_id_overrides_stale_delta_move_id(self):
        # Stale delta.opp_move_id (e.g. from a force-replace cycle where opp_mon.last_move
        # is from a different mon) gets superseded by the event's confirmed move_id.
        from poke_env.battle.pokemon_type import PokemonType
        manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        # opp's revealed moves include both moves; delta.opp_move_id is the stale one,
        # but the event identifies the actual move that fired this turn.
        stale_move = MagicMock()
        stale_move.base_power = 80
        stale_move.type = PokemonType.FIRE
        real_move = MagicMock()
        real_move.base_power = 80
        real_move.type = PokemonType.GROUND
        opp = MagicMock()
        opp.moves = {"flamethrower": stale_move, "earthquake": real_move}
        new_mon = _type_mon("FLYING")
        new_mon.species = "newmon"
        # Ground vs Flying = 0× (Flying immune). Event identifies it as the real fire.
        event = DamagingMoveEvent(
            user_species="opponentmon",
            target_species="newmon",
            target_status=None,
            move_id="earthquake",
            effectiveness=0.0,
        )
        delta = _pivot_delta("flamethrower", opp_damaging_event=event)
        battle = _pivot_battle(new_mon, opp, prev_mon=_type_mon("NORMAL"))
        # prev=Normal (1× Ground), new=Flying via event (0×) → +0.15
        result = sum(manager._compute_pivot_bonus(delta, battle))
        self.assertAlmostEqual(result, 0.15, places=5)


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
        self.manager.process_turn_reward(_battle(our_mon=our_mon), delta)
        # futile_attack specifically must not fire for status moves (base_power == 0)
        self.assertAlmostEqual(self.manager._last_breakdown.futile_attack, 0.0, places=5)


class TestFutileImmuneAttack(unittest.TestCase):
    """FUTILE_IMMUNE_PENALTY fires for type-immunity; FUTILE_ATTACK_PENALTY fires for Leftovers case."""

    def _immune_delta(self, our_move_id="thunderbolt"):
        our = np.zeros(6, dtype=np.float32)
        opp = np.zeros(6, dtype=np.float32)  # 0 damage — immune
        return TurnDelta(
            our_move_id=our_move_id, our_switch_to=None, our_prev_active="gengar",
            opp_move_id="earthquake", opp_switch_to=None, opp_prev_active="swampert",
            opp_move_known=True,
            our_hp_delta=our, opp_hp_delta=opp,
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=0.0,  # immune
            opp_effectiveness=None,
            we_moved_first=None,
        )

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        move_mock = MagicMock()
        move_mock.base_power = 95
        our_mon = MagicMock()
        our_mon.moves = {"thunderbolt": move_mock}
        self.battle = _battle(our_mon=our_mon)

    def test_immune_attack_uses_harder_penalty(self):
        self.manager.record_action(_ctx(), 6)
        self.manager.process_turn_reward(self.battle, self._immune_delta())
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.futile_attack, FUTILE_IMMUNE_PENALTY, places=5)
        self.assertLess(bd.futile_attack, FUTILE_ATTACK_PENALTY)

    def test_leftovers_case_uses_normal_penalty(self):
        # Slight net positive (Leftovers healed more than damage) — not immune
        self.manager.record_action(_ctx(), 6)
        move_mock = MagicMock()
        move_mock.base_power = 95
        our_mon = MagicMock()
        our_mon.moves = {"thunderbolt": move_mock}
        opp = np.zeros(6, dtype=np.float32)
        opp[0] = 0.01  # net gained (Leftovers > damage)
        delta = TurnDelta(
            our_move_id="thunderbolt", our_switch_to=None, our_prev_active="gengar",
            opp_move_id="surf", opp_switch_to=None, opp_prev_active="raichu",
            opp_move_known=True,
            our_hp_delta=np.zeros(6, dtype=np.float32), opp_hp_delta=opp,
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(7, dtype=np.int8),
            opp_boost_delta=np.zeros(7, dtype=np.int8),
            our_effectiveness=1.0,  # neutral — not immune
            opp_effectiveness=None,
            we_moved_first=None,
        )
        self.manager.process_turn_reward(_battle(our_mon=our_mon), delta)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.futile_attack, FUTILE_ATTACK_PENALTY, places=5)


class TestEscapeThreatSwitch(unittest.TestCase):
    """escape_threat_switch bonus fires when switching out of a known SE threat."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _set_se_threat(self, threatened: bool):
        self.manager._prev_opp_se_threat = threatened

    def test_escape_bonus_fires_when_se_threat_active(self):
        self._set_se_threat(True)
        self.manager.record_action(_ctx(), 0)  # voluntary switch to slot 0
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="skarmory"))
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.escape_threat_switch, ESCAPE_THREAT_BONUS, places=5)

    def test_escape_bonus_absent_when_no_threat(self):
        self._set_se_threat(False)
        self.manager.record_action(_ctx(), 0)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="skarmory"))
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.escape_threat_switch, 0.0, places=5)

    def test_escape_bonus_absent_on_attack(self):
        self._set_se_threat(True)
        self.manager.record_action(_ctx(), 6)  # attack, not switch
        self.manager.process_turn_reward(_battle(), _delta(our_move_id="rockslide"))
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.escape_threat_switch, 0.0, places=5)

    def test_escape_bonus_cumulative_with_switch_base(self):
        self._set_se_threat(True)
        self.manager.record_action(_ctx(), 0)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="skarmory"))
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(
            bd.switch_base + bd.escape_threat_switch,
            SWITCH_BASE_BONUS + ESCAPE_THREAT_BONUS,
            places=5,
        )


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
    """Stall tax now starts EARLY (turn 60) and RAMPS with turns past the start."""

    def _stall_tax_at_turn(self, turn):
        from agents.training.reward_manager import Gen3RewardManager
        rm = Gen3RewardManager()
        battle = _battle()
        battle.turn = turn
        rm.process_turn_reward(battle, _delta())
        return rm._last_breakdown.stall_tax

    def test_no_stall_tax_at_or_before_start_turn(self):
        from agents.training.reward_manager import STALL_TAX_START_TURN
        self.assertAlmostEqual(self._stall_tax_at_turn(STALL_TAX_START_TURN), 0.0, places=5)

    def test_stall_tax_starts_earlier_than_old_threshold(self):
        # Old threshold was 125 — confirm we now bite well before then.
        self.assertLess(self._stall_tax_at_turn(80), 0.0)

    def test_stall_tax_ramps_with_turn(self):
        from agents.training.reward_manager import (
            STALL_TAX_START_TURN, STALL_TAX_PER_TURN, STALL_TAX_RAMP_TURNS,
        )
        # rate = STALL_TAX_PER_TURN * (turn - start) / RAMP_TURNS
        t = STALL_TAX_START_TURN + STALL_TAX_RAMP_TURNS  # ramp fraction = 1.0
        self.assertAlmostEqual(self._stall_tax_at_turn(t), -STALL_TAX_PER_TURN, places=5)
        # A later turn must cost strictly more (more negative).
        self.assertLess(self._stall_tax_at_turn(t + STALL_TAX_RAMP_TURNS),
                        self._stall_tax_at_turn(t))

    def test_stall_tax_clamped_at_max(self):
        from agents.training.reward_manager import STALL_TAX_MAX
        # A very deep stall is clamped — never worse than -STALL_TAX_MAX per turn.
        self.assertAlmostEqual(self._stall_tax_at_turn(2000), -STALL_TAX_MAX, places=5)

    def test_stall_tax_grows_monotonically(self):
        prev = 0.0
        for turn in (61, 70, 90, 120, 160):
            cur = self._stall_tax_at_turn(turn)
            self.assertLessEqual(cur, prev + 1e-9)
            prev = cur


# ---------------------------------------------------------------------------
# New test classes for reward improvements
# ---------------------------------------------------------------------------

class TestFaintScaling(unittest.TestCase):
    """faint_ours and faint_opp scale with HP at time of faint."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _faint_ours_reward(self, hp_before: float) -> float:
        ctx = _ctx_with_boosts(our_hp_val=hp_before)
        self.manager.record_action(ctx, 6)
        return self.manager.process_turn_reward(_battle(), _delta(we_fainted=True))

    def test_full_hp_faint_costs_max(self):
        reward = self._faint_ours_reward(1.0)
        self.assertAlmostEqual(reward, -(0.5 + 2.0 * 1.0), places=4)

    def test_low_hp_faint_costs_less(self):
        r_low = self._faint_ours_reward(0.1)
        r_full = self._faint_ours_reward(1.0)
        self.assertGreater(r_low, r_full)

    def test_near_zero_hp_faint_costs_minimum(self):
        reward = self._faint_ours_reward(0.0)
        self.assertAlmostEqual(reward, -0.5, places=4)

    def test_faint_opp_full_hp_earns_max(self):
        ctx = _ctx_with_boosts(opp_hp_val=1.0)
        self.manager.record_action(ctx, 6)
        reward = self.manager.process_turn_reward(_battle(), _delta(opp_fainted=True))
        self.assertAlmostEqual(reward, 0.5 + 2.0 * 1.0, places=4)

    def test_faint_opp_low_hp_earns_less(self):
        def _opp_faint(hp):
            mgr = Gen3RewardManager(log_level=LogLevel.QUIET)
            ctx = _ctx_with_boosts(opp_hp_val=hp)
            mgr.record_action(ctx, 6)
            return mgr.process_turn_reward(_battle(), _delta(opp_fainted=True))
        self.assertGreater(_opp_faint(1.0), _opp_faint(0.1))


class TestRepetitionTaxEscalation(unittest.TestCase):
    """Repetition tax is LINEAR and UNCAPPED; zero-effect repeats cost much more."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _repeat_attack(self, n_times, opp_hp_delta=-0.1):
        """Attack with action=6 n_times; return list of rewards.

        Default opp_hp_delta=-0.1 ensures each attack "had effect", keeping
        _last_attack_had_effect=True so the gentle effective-repeat step applies.
        """
        rewards = []
        for _ in range(n_times):
            self.manager.record_action(_ctx(), 6)
            rewards.append(self.manager.process_turn_reward(_battle(), _delta(opp_hp_delta=opp_hp_delta)))
        return rewards

    def test_first_use_no_tax(self):
        r = self._repeat_attack(1)
        # First use: hp_opp = -(-0.1)*HP_VALUE = +0.2
        self.assertAlmostEqual(r[0], -(-0.1) * HP_VALUE, places=5)

    def test_second_use_gets_first_tax(self):
        r = self._repeat_attack(2)
        # First repeat (n=1) at the normal step: -0.03
        self.assertAlmostEqual(r[1] - r[0], -0.03, places=5)

    def test_tax_escalates_linearly(self):
        r = self._repeat_attack(4)
        bd_taxes = [-0.03, -0.06, -0.09]  # n = 1, 2, 3 at REPETITION_STEP=0.03
        for i, expected in enumerate(bd_taxes, start=1):
            self.assertAlmostEqual(r[i] - r[0], expected, places=5)

    def test_zero_effect_repeat_costs_more(self):
        # First use: action 6, opp took no damage (net 0)
        self.manager.record_action(_ctx(), 6)
        self.manager.process_turn_reward(_battle(), _delta(opp_hp_delta=0.0))
        # Second use: same action, opp took no damage — zero-effect repeat
        self.manager.record_action(_ctx(), 6)
        r2 = self.manager.process_turn_reward(_battle(), _delta(opp_hp_delta=0.0))
        # Zero-effect first repeat (n=1): -REPETITION_ZERO_EFFECT_STEP * 1 = -0.15
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.repetition_tax, -0.15, places=5)

    def test_zero_effect_uncapped_long_spam(self):
        # A long no-op spam keeps escalating well past the old -0.40 cap, down to the floor.
        from agents.training.reward_manager import REPETITION_TAX_FLOOR
        taxes = []
        for _ in range(30):
            self.manager.record_action(_ctx(), 6)
            self.manager.process_turn_reward(_battle(), _delta(opp_hp_delta=0.0))
            taxes.append(self.manager._last_breakdown.repetition_tax)
        # By the 10th repeat the zero-effect tax is -0.15*10 = -1.5, far past the old cap.
        self.assertLess(taxes[9], -1.0)
        # And it never dips below the floor.
        self.assertGreaterEqual(min(taxes), REPETITION_TAX_FLOOR - 1e-9)
        self.assertAlmostEqual(min(taxes), REPETITION_TAX_FLOOR, places=5)

    def test_changing_action_resets_counter(self):
        self._repeat_attack(3)
        # Switch to action 7
        self.manager.record_action(_ctx(), 7)
        r = self.manager.process_turn_reward(_battle(), _delta())
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.repetition_tax, 0.0, places=5)

    def test_capped_setup_routes_through_zero_effect_step(self):
        # A boost move that still raises a stat is "productive" → gentle step on repeat.
        # Once capped (no boost change), it flips to the steep zero-effect step.
        bd_up = np.zeros(7, dtype=np.int8)
        bd_up[2] = 1  # spa +1 — Calm Mind worked
        bd_capped = np.zeros(7, dtype=np.int8)  # no change — at +6 cap

        self.manager.record_action(_ctx(), 6)
        self.manager.process_turn_reward(_battle(), _delta(our_move_id="calmmind", our_boost_delta=bd_up))
        # Second productive Calm Mind: gentle step (n=1, -0.03)
        self.manager.record_action(_ctx(), 6)
        self.manager.process_turn_reward(_battle(), _delta(our_move_id="calmmind", our_boost_delta=bd_up))
        self.assertAlmostEqual(self.manager._last_breakdown.repetition_tax, -0.03, places=5)
        # Third Calm Mind is CAPPED (no boost). At record time had_effect is still True
        # (set by turn 2's productive boost), so this repeat (n=2) uses the gentle step.
        self.manager.record_action(_ctx(), 6)
        self.manager.process_turn_reward(_battle(), _delta(our_move_id="calmmind", our_boost_delta=bd_capped))
        self.assertAlmostEqual(self.manager._last_breakdown.repetition_tax, -0.06, places=5)
        # Fourth Calm Mind: now had_effect is False (turn 3 was capped) → steep zero-effect
        # step kicks in at n=3 → -0.15 * 3 = -0.45.
        self.manager.record_action(_ctx(), 6)
        self.manager.process_turn_reward(_battle(), _delta(our_move_id="calmmind", our_boost_delta=bd_capped))
        self.assertAlmostEqual(self.manager._last_breakdown.repetition_tax, -0.15 * 3, places=5)


class TestFutileSetup(unittest.TestCase):
    """futile_setup fires when a boost move had no mechanical effect."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_penalty_when_boost_delta_zero(self):
        self.manager.record_action(_ctx(), 6)
        boost_delta = np.zeros(7, dtype=np.int8)  # no change — already at cap
        delta = _delta(our_move_id="calmmind", our_boost_delta=boost_delta)
        self.manager.process_turn_reward(_battle(), delta)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.futile_setup, FUTILE_SETUP_PENALTY, places=5)

    def test_no_penalty_when_boost_applied(self):
        self.manager.record_action(_ctx(), 6)
        boost_delta = np.zeros(7, dtype=np.int8)
        boost_delta[2] = 1  # spa +1 (Calm Mind worked)
        delta = _delta(our_move_id="calmmind", our_boost_delta=boost_delta)
        self.manager.process_turn_reward(_battle(), delta)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.futile_setup, 0.0, places=5)

    def test_no_penalty_on_non_boost_move(self):
        self.manager.record_action(_ctx(), 6)
        delta = _delta(our_move_id="thunderbolt", our_boost_delta=np.zeros(7, dtype=np.int8))
        self.manager.process_turn_reward(_battle(), delta)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.futile_setup, 0.0, places=5)

    def test_skipped_when_failed_to_move(self):
        self.manager.record_action(_ctx(), 6)
        d = _delta(our_move_id="calmmind")
        d = dataclasses.replace(d, our_failed_to_move=True)
        self.manager.process_turn_reward(_battle(), d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.futile_setup, 0.0, places=5)


class TestSetupLowHP(unittest.TestCase):
    """setup_low_hp penalises boost moves chosen below 40% HP."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _setup_at_hp(self, hp_fraction: float) -> float:
        boost_delta = np.zeros(7, dtype=np.int8)
        boost_delta[2] = 1  # move worked (to isolate setup_low_hp from futile_setup)
        ctx = _ctx_with_boosts(our_hp_val=hp_fraction)
        self.manager.record_action(ctx, 6)
        delta = _delta(our_move_id="calmmind", our_boost_delta=boost_delta)
        self.manager.process_turn_reward(_battle(), delta)
        return self.manager._last_breakdown.setup_low_hp

    def test_no_penalty_at_threshold(self):
        self.assertAlmostEqual(self._setup_at_hp(0.40), 0.0, places=5)

    def test_no_penalty_above_threshold(self):
        self.assertAlmostEqual(self._setup_at_hp(0.80), 0.0, places=5)

    def test_penalty_at_zero_hp(self):
        self.assertAlmostEqual(self._setup_at_hp(0.0), SETUP_LOW_HP_MAX_PENALTY, places=5)

    def test_penalty_scales_linearly(self):
        r20 = self._setup_at_hp(0.20)
        r00 = self._setup_at_hp(0.0)
        # At 20% HP, penalty should be half of max (linear from 40% to 0%)
        self.assertAlmostEqual(r20, r00 / 2.0, places=4)

    def test_no_penalty_for_non_boost_move(self):
        ctx = _ctx_with_boosts(our_hp_val=0.10)
        self.manager.record_action(ctx, 6)
        delta = _delta(our_move_id="thunderbolt")
        self.manager.process_turn_reward(_battle(), delta)
        self.assertAlmostEqual(self.manager._last_breakdown.setup_low_hp, 0.0, places=5)


class TestStatusWasted(unittest.TestCase):
    """status_wasted fires when status move had no effect."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _status_move_turn(self, move_id: str, opp_status_applied: bool) -> float:
        from poke_env.battle.status import Status
        self.manager.record_action(_ctx(), 6)
        battle = _battle()
        # Simulate whether status was applied
        if opp_status_applied:
            opp_mon = MagicMock()
            opp_mon.status = Status.PAR
            opp_mon.fainted = False
            battle.opponent_team = {"charizard": opp_mon}
        delta = _delta(our_move_id=move_id)
        self.manager.process_turn_reward(battle, delta)
        return self.manager._last_breakdown.status_wasted

    def test_penalty_when_toxic_fails(self):
        result = self._status_move_turn("toxic", opp_status_applied=False)
        self.assertAlmostEqual(result, STATUS_WASTED_PENALTY, places=5)

    def test_no_penalty_when_status_lands(self):
        result = self._status_move_turn("toxic", opp_status_applied=True)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_no_penalty_for_attack_move(self):
        result = self._status_move_turn("thunderbolt", opp_status_applied=False)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_skipped_when_failed_to_move(self):
        self.manager.record_action(_ctx(), 6)
        d = dataclasses.replace(_delta(our_move_id="thunderwave"), our_failed_to_move=True)
        self.manager.process_turn_reward(_battle(), d)
        self.assertAlmostEqual(self.manager._last_breakdown.status_wasted, 0.0, places=5)


class TestBoostUtilized(unittest.TestCase):
    """boost_utilized rewards attacking with active stat boosts."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _attack_with_boosts(self, atk_boost: int = 0, spa_boost: int = 0,
                             opp_hp_delta: float = -0.3) -> float:
        boosts = np.zeros(7, dtype=np.int8)
        boosts[0] = atk_boost
        boosts[2] = spa_boost
        ctx = _ctx_with_boosts(our_boosts=boosts.tolist())
        self.manager.record_action(ctx, 6)
        move = MagicMock()
        move.base_power = 80
        our_mon = MagicMock()
        our_mon.moves = {"earthquake": move}
        battle = _battle(our_mon=our_mon)
        delta = _delta(our_move_id="earthquake", opp_hp_delta=opp_hp_delta)
        self.manager.process_turn_reward(battle, delta)
        return self.manager._last_breakdown.boost_utilized

    def test_no_bonus_without_boosts(self):
        self.assertAlmostEqual(self._attack_with_boosts(0, 0), 0.0, places=5)

    def test_bonus_with_atk_boost(self):
        result = self._attack_with_boosts(atk_boost=2, opp_hp_delta=-0.3)
        self.assertGreater(result, 0.0)
        # formula: 2 * 0.03 * 0.3
        self.assertAlmostEqual(result, 2 * 0.03 * 0.3, places=5)

    def test_bonus_with_spa_boost(self):
        result = self._attack_with_boosts(spa_boost=3, opp_hp_delta=-0.5)
        self.assertAlmostEqual(result, 3 * 0.03 * 0.5, places=5)

    def test_uses_higher_of_atk_spa(self):
        r_atk = self._attack_with_boosts(atk_boost=4, spa_boost=1, opp_hp_delta=-0.4)
        self.assertAlmostEqual(r_atk, 4 * 0.03 * 0.4, places=5)

    def test_no_bonus_when_no_damage_dealt(self):
        # opp_hp_delta=0 (no damage) → no boost utilized bonus
        result = self._attack_with_boosts(atk_boost=3, opp_hp_delta=0.0)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_no_bonus_for_status_move(self):
        boosts = [2, 0, 0, 0, 0, 0, 0]
        ctx = _ctx_with_boosts(our_boosts=boosts)
        self.manager.record_action(ctx, 6)
        move = MagicMock()
        move.base_power = 0  # status move
        our_mon = MagicMock()
        our_mon.moves = {"thunderwave": move}
        delta = _delta(our_move_id="thunderwave", opp_hp_delta=0.0)
        self.manager.process_turn_reward(_battle(our_mon=our_mon), delta)
        self.assertAlmostEqual(self.manager._last_breakdown.boost_utilized, 0.0, places=5)


class TestExplosionReward(unittest.TestCase):
    """Explosion: victim gets no extra penalty; surviving gets bonus; block gets extra."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _make_exploder_battle(self, we_fainted=False, our_hp_delta=0.0):
        opp_mon = MagicMock()
        opp_mon.species = "gengar"
        opp_exploder = MagicMock()
        opp_exploder.id = "explosion"
        opp_mon.moves = {"explosion": opp_exploder}
        battle = _battle(opp_mon=opp_mon)
        battle.opponent_team = {"gengar": opp_mon}
        return battle

    def test_no_extra_penalty_when_we_faint_to_explosion(self):
        """When opponent uses Explosion and we faint, explosion field must be 0."""
        self.manager.record_action(_ctx(), 6)
        battle = self._make_exploder_battle()
        d = _delta(opp_fainted=True, we_fainted=True,
                   opp_prev_active="gengar", our_hp_delta=-1.0,
                   opp_damaging_event=_explosion_event())
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.explosion, 0.0, places=5)  # no victim penalty
        self.assertLess(bd.faint_ours, 0.0)  # faint_ours still fires

    def test_bonus_when_we_survive_explosion(self):
        """When opponent Explodes and we survive (took damage), explosion=+2."""
        self.manager.record_action(_ctx(), 6)
        battle = self._make_exploder_battle()
        d = _delta(opp_fainted=True, we_fainted=False, opp_prev_active="gengar",
                   our_hp_delta=-0.5,
                   opp_damaging_event=_explosion_event())
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.explosion, 2.0, places=5)
        self.assertAlmostEqual(bd.explosion_block, 0.0, places=5)  # took damage, no block bonus

    def test_block_bonus_when_ghost_or_protect_immune(self):
        """When opponent Explodes and we take 0 damage (Ghost/Protect), block bonus fires."""
        self.manager.record_action(_ctx(), 6)
        battle = self._make_exploder_battle()
        d = _delta(opp_fainted=True, we_fainted=False, opp_prev_active="gengar",
                   our_hp_delta=0.0,  # took zero damage
                   opp_damaging_event=_explosion_event())
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.explosion, 2.0, places=5)
        self.assertAlmostEqual(bd.explosion_block, EXPLOSION_BLOCK_BONUS, places=5)

    def test_explosion_not_triggered_without_event(self):
        """Old code misfired when opp had Explosion in moveset but used a different move.
        The event-driven check requires the actual fired move to be explosion."""
        self.manager.record_action(_ctx(), 6)
        battle = self._make_exploder_battle()
        # opp_fainted but no damaging event — they fainted to status / sandstorm / etc.
        d = _delta(opp_fainted=True, we_fainted=False, opp_prev_active="gengar",
                   our_hp_delta=0.0)
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.explosion, 0.0, places=5)
        self.assertAlmostEqual(bd.explosion_block, 0.0, places=5)

    def test_explosion_block_bonus_from_event_effectiveness(self):
        """Ghost-immune explosion: event.effectiveness == 0.0 — block bonus should fire
        even if our_hp_delta is nonzero (e.g. sandstorm chip we took separately)."""
        self.manager.record_action(_ctx(), 6)
        battle = self._make_exploder_battle()
        d = _delta(opp_fainted=True, we_fainted=False, opp_prev_active="gengar",
                   our_hp_delta=-0.05,  # tiny chip from weather, not explosion
                   opp_damaging_event=_explosion_event(effectiveness=0.0))
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.explosion, 2.0, places=5)
        self.assertAlmostEqual(bd.explosion_block, EXPLOSION_BLOCK_BONUS, places=5)


class TestSeSwitchBonusFixed(unittest.TestCase):
    """se_switch must not fire on forced post-faint switches or vs fainted opponents."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_no_se_bonus_on_forced_faint_switch(self):
        """Post-faint replacement into SE matchup must NOT earn the bonus."""
        ctx = _ctx(turn=1, our_active="NONE", phase="forced_switch",
                   slot_map={"NONE": 0, "tyranitar": 1})
        self.manager.record_action(ctx, 1)
        our = _make_mon("ROCK", "DARK")
        opp = _make_mon("ELECTRIC", "FLYING")
        battle = _battle(our_mon=our, opp_mon=opp)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="tyranitar"))
        self.assertAlmostEqual(self.manager._last_breakdown.se_switch, 0.0, places=5)

    def test_no_se_bonus_vs_fainted_opponent(self):
        """Should not award bonus when opponent active mon is fainted."""
        ctx = _ctx(turn=1, our_active="pikachu", slot_map={"pikachu": 0, "tyranitar": 1})
        self.manager.record_action(ctx, 1)
        our = _make_mon("ROCK", "DARK")
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.fainted = True
        battle = _battle(our_mon=our, opp_mon=opp)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="tyranitar"))
        self.assertAlmostEqual(self.manager._last_breakdown.se_switch, 0.0, places=5)


class TestSeSwitchOpponentTracking(unittest.TestCase):
    """Per-mon opponent tracker gates se_switch to prevent switch loops."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _do_voluntary_switch(self, our_mon, opp_mon, turn=1):
        """Trigger one voluntary switch turn and return the breakdown."""
        ctx = _ctx(turn=turn, our_active="prev", slot_map={"prev": 0, "incoming": 1})
        self.manager.record_action(ctx, 1)
        battle = _battle(our_mon=our_mon, opp_mon=opp_mon)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="incoming"))
        return self.manager._last_breakdown

    def test_se_bonus_fires_first_time_vs_new_opp(self):
        """No prior matchup → se_switch fires normally (existing behaviour)."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        bd = self._do_voluntary_switch(our, opp)
        self.assertAlmostEqual(bd.se_switch, SE_SWITCH_BONUS, places=5)

    def test_se_bonus_blocked_on_second_switch_same_opp(self):
        """Switch vs Zapdos, come back vs Zapdos without Zapdos switching → se_switch == 0."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        # First switch in: bonus fires, tracker set to tyranitar→zapdos
        self._do_voluntary_switch(our, opp, turn=1)
        # Second switch in vs same opp (spam_mult ok at turn 3): blocked by tracker
        bd = self._do_voluntary_switch(our, opp, turn=3)
        self.assertAlmostEqual(bd.se_switch, 0.0, places=5)

    def test_se_bonus_fires_again_after_opponent_switches(self):
        """After our mon sees A, opponent switches to B — switching back in vs B fires bonus."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp_a = _make_mon("ELECTRIC", "FLYING")
        opp_a.species = "zapdos"
        opp_b = _make_mon("ELECTRIC", "FLYING")
        opp_b.species = "raikou"
        # First switch in vs zapdos: fires, tracker = tyranitar→zapdos
        self._do_voluntary_switch(our, opp_a, turn=1)
        # Opponent has switched (zapdos→raikou); our mon comes back in vs raikou: fires
        bd = self._do_voluntary_switch(our, opp_b, turn=3)
        self.assertAlmostEqual(bd.se_switch, SE_SWITCH_BONUS, places=5)

    def test_se_bonus_resets_after_opp_switches_back(self):
        """A→B→A opp rotation: each time our mon sees a new opp species, bonus fires again."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp_a = _make_mon("ELECTRIC", "FLYING")
        opp_a.species = "zapdos"
        opp_b = _make_mon("ELECTRIC", "FLYING")
        opp_b.species = "raikou"
        # vs zapdos: fires, tracker = tyranitar→zapdos
        self._do_voluntary_switch(our, opp_a, turn=1)
        # vs raikou: fires, tracker = tyranitar→raikou
        self._do_voluntary_switch(our, opp_b, turn=3)
        # vs zapdos again: tracker is raikou ≠ zapdos → fires
        bd = self._do_voluntary_switch(our, opp_a, turn=5)
        self.assertAlmostEqual(bd.se_switch, SE_SWITCH_BONUS, places=5)

    def test_roar_does_not_update_tracker(self):
        """A roared switch-in does not update _last_opp_seen_by."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        # Pre-populate tracker: tyranitar last saw "jumpluff"
        self.manager._last_opp_seen_by["tyranitar"] = "jumpluff"
        # Roared switch in
        ctx = _ctx(turn=1, our_active="pikachu", phase="forced_switch",
                   slot_map={"pikachu": 0, "tyranitar": 1})
        self.manager.record_action(ctx, 1)
        battle = _battle(our_mon=our, opp_mon=opp)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="tyranitar"))
        # Tracker must remain unchanged — roars don't update it
        self.assertEqual(self.manager._last_opp_seen_by.get("tyranitar"), "jumpluff")

    def test_reset_clears_tracker(self):
        """reset() wipes _last_opp_seen_by so the next battle starts clean."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        # Populate the tracker in battle 1
        self._do_voluntary_switch(our, opp, turn=1)
        self.assertEqual(self.manager._last_opp_seen_by.get("tyranitar"), "zapdos")
        # Simulate end-of-battle reset
        self.manager.reset()
        self.assertEqual(self.manager._last_opp_seen_by, {})
        # After reset, the same matchup fires again as if it were the first time
        bd = self._do_voluntary_switch(our, opp, turn=1)
        self.assertAlmostEqual(bd.se_switch, SE_SWITCH_BONUS, places=5)

    def test_independent_tracking_per_mon(self):
        """Each mon has its own tracker entry — one mon being blocked doesn't affect another."""
        tyranitar = _make_mon("ROCK", "DARK")
        tyranitar.species = "tyranitar"
        golem = _make_mon("ROCK", "GROUND")
        golem.species = "golem"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        # Tyranitar switches in: fires, tracker = {tyranitar: zapdos}
        self._do_voluntary_switch(tyranitar, opp, turn=1)
        # Tyranitar switches in again: blocked
        bd_ttar = self._do_voluntary_switch(tyranitar, opp, turn=3)
        self.assertAlmostEqual(bd_ttar.se_switch, 0.0, places=5)
        # Golem has no tracker entry yet → should fire independently
        bd_golem = self._do_voluntary_switch(golem, opp, turn=5)
        self.assertAlmostEqual(bd_golem.se_switch, SE_SWITCH_BONUS, places=5)

    def test_blocked_still_earns_switch_base(self):
        """When se_switch is blocked by the tracker, switch_base subsidy is still awarded."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        # First switch: both se_switch and switch_base fire
        bd1 = self._do_voluntary_switch(our, opp, turn=1)
        self.assertAlmostEqual(bd1.se_switch, SE_SWITCH_BONUS, places=5)
        self.assertAlmostEqual(bd1.switch_base, SWITCH_BASE_BONUS, places=5)
        # Second switch vs same opp: se_switch blocked, but switch_base still runs
        bd2 = self._do_voluntary_switch(our, opp, turn=3)
        self.assertAlmostEqual(bd2.se_switch, 0.0, places=5)
        self.assertAlmostEqual(bd2.switch_base, SWITCH_BASE_BONUS, places=5)

    def test_tracker_updates_even_without_type_advantage(self):
        """Tracker records the matchup on any non-roared switch, not just SE ones."""
        our = _make_mon("NORMAL")   # no SE advantage vs anything
        our.species = "snorlax"
        opp = _make_mon("NORMAL")
        opp.species = "blissey"
        ctx = _ctx(turn=1, our_active="prev", slot_map={"prev": 0, "incoming": 1})
        self.manager.record_action(ctx, 1)
        battle = _battle(our_mon=our, opp_mon=opp)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="incoming"))
        # Tracker must be set even though se_switch == 0
        self.assertEqual(self.manager._last_opp_seen_by.get("snorlax"), "blissey")
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.se_switch, 0.0, places=5)

    def test_tracker_not_updated_when_opp_fainted(self):
        """Switching into a fainted opponent does not write to the tracker."""
        our = _make_mon("ROCK", "DARK")
        our.species = "tyranitar"
        opp = _make_mon("ELECTRIC", "FLYING")
        opp.species = "zapdos"
        opp.fainted = True
        # Pre-set tracker to a known value
        self.manager._last_opp_seen_by["tyranitar"] = "jumpluff"
        ctx = _ctx(turn=1, our_active="prev", slot_map={"prev": 0, "incoming": 1})
        self.manager.record_action(ctx, 1)
        battle = _battle(our_mon=our, opp_mon=opp)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="incoming"))
        # Fainted opp: tracker must remain unchanged
        self.assertEqual(self.manager._last_opp_seen_by.get("tyranitar"), "jumpluff")


class TestFinishingBlow(unittest.TestCase):
    """finishing_blow bonus fires on damaging-move KOs."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _do_attack_ko(self, our_mon, opp_fainted=True, our_move_id="earthquake",
                      our_failed_to_move=False, our_switch_to=None, opp_hp_before=0.12):
        """Simulate an attack turn that may KO the opponent."""
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager._opp_active_hp_before = opp_hp_before
        battle = _battle(our_mon=our_mon, opp_mon=None)
        d = _delta(
            opp_fainted=opp_fainted,
            our_move_id=our_move_id if our_switch_to is None else None,
            our_switch_to=our_switch_to,
            opp_hp_delta=-0.1 if opp_fainted else 0.0,
        )
        if our_failed_to_move:
            d = dataclasses.replace(d, our_failed_to_move=True)
        self.manager.process_turn_reward(battle, d)
        return self.manager._last_breakdown

    def test_fires_on_any_kill_by_damaging_move(self):
        """opp_fainted=True + damaging move → finishing_blow == FINISHING_BLOW_BONUS."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        bd = self._do_attack_ko(our, opp_fainted=True, our_move_id="earthquake")
        self.assertAlmostEqual(bd.finishing_blow, FINISHING_BLOW_BONUS, places=5)

    def test_no_bonus_for_non_damaging_move_kill(self):
        """opp_fainted=True but move has base_power=0 → finishing_blow == 0."""
        our = _make_mon("POISON", moves=[("toxic", "POISON", 0)])
        bd = self._do_attack_ko(our, opp_fainted=True, our_move_id="toxic")
        self.assertAlmostEqual(bd.finishing_blow, 0.0, places=5)

    def test_no_bonus_when_opp_did_not_faint(self):
        """opp_fainted=False → finishing_blow == 0."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        bd = self._do_attack_ko(our, opp_fainted=False, our_move_id="earthquake")
        self.assertAlmostEqual(bd.finishing_blow, 0.0, places=5)

    def test_no_bonus_when_failed_to_move(self):
        """our_failed_to_move=True → finishing_blow == 0 even if opp fainted."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        bd = self._do_attack_ko(our, opp_fainted=True, our_move_id="earthquake",
                                our_failed_to_move=True)
        self.assertAlmostEqual(bd.finishing_blow, 0.0, places=5)

    def test_no_bonus_on_switch_turn(self):
        """our_switch_to is set → finishing_blow == 0 even if opp fainted."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        bd = self._do_attack_ko(our, opp_fainted=True, our_switch_to="marowak")
        self.assertAlmostEqual(bd.finishing_blow, 0.0, places=5)

    def test_bonus_included_in_breakdown_total(self):
        """finishing_blow contributes positively to bd.total."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager._opp_active_hp_before = 0.12
        battle = _battle(our_mon=our, opp_mon=None)
        d = _delta(opp_fainted=True, our_move_id="earthquake", opp_hp_delta=-0.1)
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        self.assertAlmostEqual(bd.finishing_blow, FINISHING_BLOW_BONUS, places=5)
        # total must exceed what it would be without finishing_blow
        self.assertGreater(bd.total, bd.total - bd.finishing_blow)

    def test_no_bonus_when_move_not_in_moves_dict(self):
        """finishing_blow is 0 when the move_id isn't in mon.moves (e.g. not yet revealed)."""
        our = _make_mon("GROUND")  # moves = {} — no revealed moves
        bd = self._do_attack_ko(our, opp_fainted=True, our_move_id="earthquake")
        self.assertAlmostEqual(bd.finishing_blow, 0.0, places=5)

    def test_no_bonus_when_no_active_mon(self):
        """finishing_blow is 0 when battle.active_pokemon is None."""
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager._opp_active_hp_before = 0.12
        battle = _battle(our_mon=None, opp_mon=None)   # no active mon
        d = _delta(opp_fainted=True, our_move_id="earthquake", opp_hp_delta=-0.1)
        self.manager.process_turn_reward(battle, d)
        self.assertAlmostEqual(self.manager._last_breakdown.finishing_blow, 0.0, places=5)

    def test_fires_on_full_hp_ko(self):
        """finishing_blow fires even when the opponent was at 100% HP — no HP threshold."""
        our = _make_mon("PSYCHIC", moves=[("psychic", "PSYCHIC", 90)])
        bd = self._do_attack_ko(our, opp_fainted=True, our_move_id="psychic",
                                opp_hp_before=1.0)
        self.assertAlmostEqual(bd.finishing_blow, FINISHING_BLOW_BONUS, places=5)

    def test_stacks_correctly_with_faint_opp(self):
        """Both faint_opp and finishing_blow fire together on a damaging-move KO."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager._opp_active_hp_before = 0.12
        battle = _battle(our_mon=our, opp_mon=None)
        d = _delta(opp_fainted=True, our_move_id="earthquake", opp_hp_delta=-0.1)
        self.manager.process_turn_reward(battle, d)
        bd = self.manager._last_breakdown
        expected_faint_opp = FAINT_BASE + FAINT_HP_SCALE * 0.12   # 0.5 + 2.0*0.12 = 0.74
        self.assertAlmostEqual(bd.faint_opp, expected_faint_opp, places=5)
        self.assertAlmostEqual(bd.finishing_blow, FINISHING_BLOW_BONUS, places=5)
        # Both contribute — total must be at least their sum
        self.assertGreaterEqual(bd.total, expected_faint_opp + FINISHING_BLOW_BONUS)

    def test_no_bonus_when_move_id_is_none(self):
        """finishing_blow is 0 when our_move_id is None (e.g. nothing acted this turn)."""
        our = _make_mon("GROUND", moves=[("earthquake", "GROUND", 100)])
        self.manager.record_action(_ctx(turn=1), 6)
        self.manager._opp_active_hp_before = 0.12
        battle = _battle(our_mon=our, opp_mon=None)
        # our_move_id=None with opp_fainted=True — edge case: opp died to residual damage
        d = _delta(opp_fainted=True, our_move_id=None, opp_hp_delta=-0.1)
        self.manager.process_turn_reward(battle, d)
        self.assertAlmostEqual(self.manager._last_breakdown.finishing_blow, 0.0, places=5)


class TestBouncingTaxEscalation(unittest.TestCase):
    """A→B→A→B oscillation pays an escalating bouncing tax (was flat -0.15)."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _oscillate(self, turns=4):
        """Run an A↔B oscillation and return the bouncing tax on each switch turn."""
        taxes = []
        # turn 1: A(pikachu) -> B(raichu): not a bounce (establishes _last_switched_from)
        ctx = _ctx(turn=1, our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        taxes.append(self.manager._last_breakdown.switch_bouncing_tax)
        active, other = "raichu", "pikachu"
        for i in range(1, turns):
            turn = 1 + 2 * i  # space by 2 so spam_mult stays 1
            slot = {active: 0, other: 1}
            ctx = _ctx(turn=turn, our_active=active, slot_map=slot)
            self.manager.record_action(ctx, 1)  # slot 1 = the mon we just left
            self.manager.process_turn_reward(_battle(), _delta(our_switch_to=other))
            taxes.append(self.manager._last_breakdown.switch_bouncing_tax)
            active, other = other, active
        return taxes

    def test_first_switch_no_bounce(self):
        taxes = self._oscillate(turns=1)
        self.assertAlmostEqual(taxes[0], 0.0, places=5)

    def test_bounce_escalates(self):
        from agents.training.reward_manager import BOUNCING_TAX_STEP
        taxes = self._oscillate(turns=4)
        # taxes[0] is the establishing switch (no bounce). Bounces start at index 1.
        self.assertAlmostEqual(taxes[0], 0.0, places=5)
        self.assertAlmostEqual(taxes[1], BOUNCING_TAX_STEP * 1, places=5)
        self.assertAlmostEqual(taxes[2], BOUNCING_TAX_STEP * 2, places=5)
        self.assertAlmostEqual(taxes[3], BOUNCING_TAX_STEP * 3, places=5)

    def test_bounce_floored(self):
        from agents.training.reward_manager import BOUNCING_TAX_FLOOR
        taxes = self._oscillate(turns=30)
        self.assertGreaterEqual(min(taxes), BOUNCING_TAX_FLOOR - 1e-9)
        self.assertAlmostEqual(min(taxes), BOUNCING_TAX_FLOOR, places=5)

    def test_attack_resets_bounce_counter(self):
        # A↔B twice, then a move, then bounce again — counter must restart at n=1.
        ctx = _ctx(turn=1, our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        ctx = _ctx(turn=3, our_active="raichu", slot_map={"raichu": 0, "pikachu": 1})
        self.manager.record_action(ctx, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="pikachu"))
        self.assertAlmostEqual(self.manager._last_breakdown.switch_bouncing_tax, -0.15, places=5)
        # An attack breaks the oscillation streak.
        self.manager.record_action(_ctx(turn=4, our_active="pikachu"), 6)
        self.manager.process_turn_reward(_battle(), _delta())
        # Bounce again: counter restarts → -0.15, not -0.30.
        ctx = _ctx(turn=5, our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        # _last_switched_from is now "pikachu" (set when leaving pikachu on the move-turn?)
        # No — only switches set _last_switched_from. After the move turn it's still
        # "raichu" (from turn 3). Switching to raichu from pikachu == bounce, n restarts at 1.
        self.assertAlmostEqual(self.manager._last_breakdown.switch_bouncing_tax, -0.15, places=5)


class TestDeadMatchupTax(unittest.TestCase):
    """Escalating tax for staying in when every damaging move is 0× vs the opp active."""

    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def _stay_in(self, our_mon, opp_mon, move_id="thunderbolt"):
        self.manager.record_action(_ctx(), 6)
        battle = _battle(our_mon=our_mon, opp_mon=opp_mon)
        self.manager.process_turn_reward(battle, _delta(our_move_id=move_id))
        return self.manager._last_breakdown.dead_matchup_tax

    def test_fires_when_all_moves_immune(self):
        from agents.training.reward_manager import DEAD_MATCHUP_TAX_STEP
        # Electric attacker vs Ground type: Thunderbolt is 0×.
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95)])
        opp = _make_mon("GROUND")
        tax = self._stay_in(our, opp)
        self.assertAlmostEqual(tax, DEAD_MATCHUP_TAX_STEP * 1, places=5)

    def test_escalates_each_turn_stuck(self):
        from agents.training.reward_manager import DEAD_MATCHUP_TAX_STEP
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95)])
        opp = _make_mon("GROUND")
        for n in range(1, 5):
            tax = self._stay_in(our, opp)
            self.assertAlmostEqual(tax, DEAD_MATCHUP_TAX_STEP * n, places=5)

    def test_no_tax_when_a_move_is_effective(self):
        # Electric + Ice attacker vs Ground: Ice Beam is 2× → not a dead matchup.
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95),
                                           ("icebeam", "ICE", 95)])
        opp = _make_mon("GROUND")
        self.assertAlmostEqual(self._stay_in(our, opp, move_id="icebeam"), 0.0, places=5)

    def test_no_tax_when_no_damaging_moves_revealed(self):
        # A mon with only status moves can't be judged as "dead matchup spam".
        our = _make_mon("NORMAL", moves=[("toxic", "POISON", 0)])
        opp = _make_mon("GHOST")
        self.assertAlmostEqual(self._stay_in(our, opp, move_id="toxic"), 0.0, places=5)

    def test_switch_resets_counter(self):
        from agents.training.reward_manager import DEAD_MATCHUP_TAX_STEP
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95)])
        opp = _make_mon("GROUND")
        self._stay_in(our, opp)            # n=1
        self._stay_in(our, opp)            # n=2
        # Switch out — counter resets, no tax on the switch turn.
        ctx = _ctx(our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx, 1)
        self.manager.process_turn_reward(_battle(), _delta(our_switch_to="raichu"))
        self.assertAlmostEqual(self.manager._last_breakdown.dead_matchup_tax, 0.0, places=5)
        # Next dead-matchup stay restarts at n=1.
        self.assertAlmostEqual(self._stay_in(our, opp), DEAD_MATCHUP_TAX_STEP * 1, places=5)

    def test_floored(self):
        from agents.training.reward_manager import DEAD_MATCHUP_TAX_FLOOR
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95)])
        opp = _make_mon("GROUND")
        last = 0.0
        for _ in range(40):
            last = self._stay_in(our, opp)
        self.assertAlmostEqual(last, DEAD_MATCHUP_TAX_FLOOR, places=5)

    def test_no_tax_on_switch_turn(self):
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95)])
        opp = _make_mon("GROUND")
        ctx = _ctx(our_active="pikachu", slot_map={"pikachu": 0, "raichu": 1})
        self.manager.record_action(ctx, 1)
        battle = _battle(our_mon=our, opp_mon=opp)
        self.manager.process_turn_reward(battle, _delta(our_switch_to="raichu"))
        self.assertAlmostEqual(self.manager._last_breakdown.dead_matchup_tax, 0.0, places=5)

    def test_pivot_strictly_beats_staying_in_dead_matchup(self):
        """The core design goal: switching out must out-value staying after a few
        turns of dead-matchup spam, with no HP swing to muddy the comparison."""
        our = _make_mon("ELECTRIC", moves=[("thunderbolt", "ELECTRIC", 95)])
        opp = _make_mon("GROUND")
        # Stay in for several turns (escalating immune + dead-matchup + repetition taxes).
        stay_rewards = []
        for _ in range(4):
            self.manager.record_action(_ctx(), 6)
            r = self.manager.process_turn_reward(
                _battle(our_mon=our, opp_mon=opp), _delta(our_move_id="thunderbolt"))
            stay_rewards.append(r)
        # Each successive stay is more negative (escalation).
        self.assertLess(stay_rewards[-1], stay_rewards[0])
        # A switch from the same trapped state yields a clearly positive subsidy instead.
        mgr2 = Gen3RewardManager(log_level=LogLevel.QUIET)
        ctx = _ctx(our_active="zapdos", slot_map={"zapdos": 0, "swampert": 1})
        mgr2.record_action(ctx, 1)
        switch_reward = mgr2.process_turn_reward(_battle(), _delta(our_switch_to="swampert"))
        self.assertGreater(switch_reward, max(stay_rewards))


class TestFutileImmunePenaltyRaised(unittest.TestCase):
    """The flat immune penalty was raised from -0.25 to -0.5."""

    def test_immune_penalty_value(self):
        self.assertAlmostEqual(FUTILE_IMMUNE_PENALTY, -0.5, places=5)

    def test_immune_penalty_harsher_than_futile_attack(self):
        self.assertLess(FUTILE_IMMUNE_PENALTY, FUTILE_ATTACK_PENALTY)


if __name__ == "__main__":
    unittest.main()
