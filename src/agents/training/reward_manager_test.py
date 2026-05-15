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
        our_hp_delta=our,
        opp_hp_delta=opp,
        we_fainted=we_fainted,
        opp_fainted=opp_fainted,
    )


def _battle(won=False, lost=False, finished=False):
    battle = MagicMock()
    battle.won = won
    battle.lost = lost
    battle.finished = finished
    battle.turn = 1
    battle.opponent_team = {}
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


if __name__ == "__main__":
    unittest.main()
