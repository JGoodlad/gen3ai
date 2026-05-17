import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from agents.training.reward_manager import Gen3RewardManager
from agents.training.battle_context import BattleContext, TurnDelta
from utils.logging.levels import LogLevel


def _ctx(turn=1, our_active="pikachu", opp_active="charizard", action_mask=None):
    mask = action_mask if action_mask is not None else np.ones(11, dtype=np.int8)
    hp = np.ones(6, dtype=np.float32)
    return BattleContext(
        turn=turn,
        phase="move_selection",
        mask=mask,
        obs=np.zeros(1, dtype=np.float32),
        our_slot_map={our_active: 0},
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
    )


def _delta(our_hp_delta=0.0, opp_hp_delta=0.0, we_fainted=False, opp_fainted=False,
           opp_prev_active="charizard"):
    our = np.zeros(6, dtype=np.float32)
    opp = np.zeros(6, dtype=np.float32)
    our[0] = our_hp_delta
    opp[0] = opp_hp_delta
    return TurnDelta(
        our_move_id=None,
        our_switch_to=None,
        our_prev_active="pikachu",
        opp_move_id=None,
        opp_switch_to=None,
        opp_prev_active=opp_prev_active,
        opp_move_known=False,
        our_hp_delta=our,
        opp_hp_delta=opp,
        we_fainted=we_fainted,
        opp_fainted=opp_fainted,
        our_failed_to_move=False,
        our_cant_reason=None,
        opp_failed_to_move=False,
        opp_cant_reason=None,
    )


def _battle(won=False, lost=False, finished=False, opp_spikes=0):
    battle = MagicMock()
    battle.won = won
    battle.lost = lost
    battle.finished = finished
    battle.turn = 1
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    from poke_env.battle.side_condition import SideCondition
    battle.opponent_side_conditions = (
        {SideCondition.SPIKES: opp_spikes} if opp_spikes > 0 else {}
    )
    return battle


class TestRewardManager(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)

    def test_attack_tracking(self):
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)  # action 6 = first move
        self.assertEqual(self.manager.attack_count, 1)

        reward = self.manager.process_turn_reward(_battle(), _delta())
        self.assertIsInstance(reward, float)
        self.assertEqual(self.manager.total_reward, reward)

    def test_repetition_tax(self):
        ctx1 = _ctx(turn=1)
        self.manager.record_action(ctx1, 6)
        r1 = self.manager.process_turn_reward(_battle(), _delta())

        ctx2 = _ctx(turn=2)
        self.manager.record_action(ctx2, 6)  # same action → repetition tax
        r2 = self.manager.process_turn_reward(_battle(), _delta())

        # Repetition tax is -0.02
        self.assertAlmostEqual(r2, r1 - 0.02, places=5)

    def test_voluntary_switch_subsidy(self):
        ctx1 = _ctx(turn=1, our_active="pikachu")
        self.manager.record_action(ctx1, 6)
        self.manager.process_turn_reward(_battle(), _delta())

        # Turn 2: switch to raichu (action 1 = switch to slot 1)
        ctx2 = _ctx(turn=2, our_active="raichu")
        with patch("agents.training.reward_manager.SwitchDetection") as mock_sd:
            mock_sd.get_switch_type.return_value = (True, True, True)
            self.manager.record_action(ctx2, 1)

        reward = self.manager.process_turn_reward(_battle(), _delta())
        self.assertGreaterEqual(reward, 0.0)
        self.assertEqual(self.manager.switch_count, 1)

    def test_faint_reward(self):
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        # Opponent faints — should add FAINTED_VALUE (2.0) to base reward
        reward = self.manager.process_turn_reward(_battle(), _delta(opp_fainted=True))
        self.assertAlmostEqual(reward, 2.0, places=5)

    def test_win_reward(self):
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        battle = _battle(won=True, finished=True)
        reward = self.manager.process_turn_reward(battle, _delta())
        # Victory value is 30.0
        self.assertAlmostEqual(reward, 30.0, places=5)


    def test_spikes_layer_bonus(self):
        from agents.training.reward_manager import SPIKES_LAYER_BONUS
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        # First spikes layer goes down
        battle = _battle(opp_spikes=1)
        reward = self.manager.process_turn_reward(battle, _delta())
        self.assertAlmostEqual(reward, SPIKES_LAYER_BONUS, places=5)

    def test_spikes_two_layers_bonus(self):
        from agents.training.reward_manager import SPIKES_LAYER_BONUS
        # Simulate spikes going from 1 → 3 in two separate turns
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        self.manager.process_turn_reward(_battle(opp_spikes=1), _delta())

        ctx2 = _ctx(turn=2)
        self.manager.record_action(ctx2, 7)  # different move to avoid repetition tax
        reward = self.manager.process_turn_reward(_battle(opp_spikes=2), _delta())
        self.assertAlmostEqual(reward, SPIKES_LAYER_BONUS, places=5)

    def test_spikes_waste_penalty(self):
        from agents.training.reward_manager import SPIKES_WASTE_PENALTY
        # Pre-set internal state to 3 layers already up
        self.manager._prev_opp_spikes = 3
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        delta = TurnDelta(
            our_move_id="spikes",
            our_switch_to=None,
            our_prev_active="pikachu",
            opp_move_id=None,
            opp_switch_to=None,
            opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False,
            opp_fainted=False,
            our_failed_to_move=False,
            our_cant_reason=None,
            opp_failed_to_move=False,
            opp_cant_reason=None,
        )
        reward = self.manager.process_turn_reward(_battle(opp_spikes=3), delta)
        self.assertAlmostEqual(reward, SPIKES_WASTE_PENALTY, places=5)

    def test_spikes_no_bonus_without_change(self):
        # No spikes → no spikes: zero bonus
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        reward = self.manager.process_turn_reward(_battle(opp_spikes=0), _delta())
        self.assertAlmostEqual(reward, 0.0, places=5)

    def test_failed_roar_penalty(self):
        from agents.training.reward_manager import FAILED_ROAR_PENALTY
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        delta = TurnDelta(
            our_move_id="roar",
            our_switch_to=None,
            our_prev_active="pikachu",
            opp_move_id=None,
            opp_switch_to=None,       # Roar failed — no switch forced
            opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False,
            opp_fainted=False,
            our_failed_to_move=False,
            our_cant_reason=None,
            opp_failed_to_move=False,
            opp_cant_reason=None,
        )
        reward = self.manager.process_turn_reward(_battle(), delta)
        self.assertAlmostEqual(reward, FAILED_ROAR_PENALTY, places=5)

    def test_futile_attack_penalty_fires_when_opp_net_healed(self):
        from agents.training.reward_manager import FUTILE_ATTACK_PENALTY, HP_VALUE
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)

        battle = _battle()
        move_mock = MagicMock()
        move_mock.base_power = 20
        battle.active_pokemon = MagicMock()
        battle.active_pokemon.moves = {"rapidspin": move_mock}

        opp_hp = np.zeros(6, dtype=np.float32)
        opp_hp[0] = 0.05   # opponent net gained 5% HP (Leftovers > damage dealt)
        delta = TurnDelta(
            our_move_id="rapidspin",
            our_switch_to=None,
            our_prev_active="pikachu",
            opp_move_id="struggle",
            opp_switch_to=None,
            opp_prev_active="charizard",
            opp_move_known=True,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=opp_hp,
            we_fainted=False,
            opp_fainted=False,
            our_failed_to_move=False,
            our_cant_reason=None,
            opp_failed_to_move=False,
            opp_cant_reason=None,
        )
        reward = self.manager.process_turn_reward(battle, delta)
        # base: -opp_hp_delta.sum() * HP_VALUE = -0.05 * 2.0 = -0.10
        # futile penalty: -0.05
        # total: -0.15
        expected = -0.05 * HP_VALUE + FUTILE_ATTACK_PENALTY
        self.assertAlmostEqual(reward, expected, places=4)

    def test_futile_attack_penalty_skips_status_moves(self):
        from agents.training.reward_manager import FUTILE_ATTACK_PENALTY
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)

        battle = _battle()
        move_mock = MagicMock()
        move_mock.base_power = 0   # status move
        battle.active_pokemon = MagicMock()
        battle.active_pokemon.moves = {"thunderwave": move_mock}

        delta = TurnDelta(
            our_move_id="thunderwave",
            our_switch_to=None,
            our_prev_active="pikachu",
            opp_move_id=None,
            opp_switch_to=None,
            opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False,
            opp_fainted=False,
            our_failed_to_move=False,
            our_cant_reason=None,
            opp_failed_to_move=False,
            opp_cant_reason=None,
        )
        reward = self.manager.process_turn_reward(battle, delta)
        # Should NOT include FUTILE_ATTACK_PENALTY since base_power == 0
        self.assertNotAlmostEqual(reward, FUTILE_ATTACK_PENALTY, places=5)
        # Reward should just be 0 (no HP change, no other signals)
        self.assertAlmostEqual(reward, 0.0, places=5)

    def _make_battle_with_opp_move(self, move_type_name, base_power, our_type1, our_type2=None):
        """Helper: battle where opp active has one revealed move of given type."""
        battle = MagicMock()
        battle.won = False
        battle.lost = False
        battle.finished = False
        battle.turn = 1
        battle.opponent_team = {}
        battle.opponent_side_conditions = {}

        our_mon = MagicMock()
        our_mon.type_1 = MagicMock()
        our_mon.type_2 = our_type2
        battle.active_pokemon = our_mon

        opp_move = MagicMock()
        opp_move.base_power = base_power
        opp_move.type = MagicMock()

        def se_mult(t1, t2, type_chart):
            return 2.0 if move_type_name == "SE" else 0.5
        opp_move.type.damage_multiplier = se_mult

        opp_mon = MagicMock()
        opp_mon.moves = {"testmove": opp_move}
        opp_mon.boosts = {}
        battle.opponent_active_pokemon = opp_mon

        return battle

    def test_matchup_penalty_fires_on_second_turn(self):
        from agents.training.reward_manager import MATCHUP_PENALTY
        # Turn 1: opp SE move revealed — penalty should NOT fire (unknown at decision time)
        ctx1 = _ctx(turn=1)
        self.manager.record_action(ctx1, 6)
        battle1 = self._make_battle_with_opp_move("SE", 80, None)
        r1 = self.manager.process_turn_reward(battle1, _delta())
        self.assertAlmostEqual(r1, 0.0, places=5)

        # Turn 2: threat is now known at decision time — penalty fires
        ctx2 = _ctx(turn=2)
        self.manager.record_action(ctx2, 7)
        battle2 = self._make_battle_with_opp_move("SE", 80, None)
        r2 = self.manager.process_turn_reward(battle2, _delta())
        self.assertAlmostEqual(r2, MATCHUP_PENALTY, places=5)

    def test_matchup_penalty_skipped_on_switch(self):
        from agents.training.reward_manager import MATCHUP_PENALTY
        self.manager._prev_opp_se_threat = True
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        delta = TurnDelta(
            our_move_id=None,
            our_switch_to="raichu",
            our_prev_active="pikachu",
            opp_move_id=None,
            opp_switch_to=None,
            opp_prev_active="charizard",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False,
            opp_fainted=False,
            our_failed_to_move=False,
            our_cant_reason=None,
            opp_failed_to_move=False,
            opp_cant_reason=None,
        )
        r = self.manager.process_turn_reward(_battle(), delta)
        self.assertGreater(r, MATCHUP_PENALTY)

    def test_matchup_penalty_absent_when_no_threat(self):
        self.manager._prev_opp_se_threat = False
        ctx = _ctx(turn=1)
        self.manager.record_action(ctx, 6)
        r = self.manager.process_turn_reward(_battle(), _delta())
        self.assertAlmostEqual(r, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
