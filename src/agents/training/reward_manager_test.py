import unittest
from unittest.mock import MagicMock
import numpy as np
from agents.training.reward_manager import Gen3RewardManager
from utils.logging.levels import LogLevel

class TestRewardManager(unittest.TestCase):
    def setUp(self):
        self.manager = Gen3RewardManager(log_level=LogLevel.QUIET)
        self.battle = MagicMock()
        self.battle.turn = 1
        self.battle.won = False
        self.battle.lost = False
        self.battle.finished = False
        
        # Mock active pokemon
        self.active_mon = MagicMock()
        self.active_mon.species = "Pikachu"
        self.active_mon.fainted = False
        self.battle.active_pokemon = self.active_mon
        self.battle.opponent_active_pokemon = MagicMock()
        self.battle.opponent_active_pokemon.species = "Charizard"
        self.battle.team = {"Pikachu": self.active_mon}

    def test_attack_tracking(self):
        # Action 6 is the first move
        self.manager.record_action(self.battle, 6)
        self.assertEqual(self.manager.attack_count, 1)
        
        reward = self.manager.process_turn_reward(self.battle, 1.0)
        self.assertEqual(reward, 1.0) # No subsidy for attacks usually
        self.assertEqual(self.manager.total_reward, 1.0)

    def test_repetition_tax(self):
        # First attack
        self.manager.record_action(self.battle, 6)
        self.manager.process_turn_reward(self.battle, 1.0)
        
        # Repeat the same attack
        self.manager.record_action(self.battle, 6)
        reward = self.manager.process_turn_reward(self.battle, 1.0)
        
        # Repetition tax is -0.02
        self.assertAlmostEqual(reward, 0.98)
        self.assertAlmostEqual(self.manager.total_reward, 1.98)

    def test_voluntary_switch_subsidy(self):
        # Initial state: Pikachu active
        self.manager.record_action(self.battle, 6) # Attack turn 1
        self.manager.process_turn_reward(self.battle, 0.0)
        
        # Turn 2: Switch to Raichu
        self.battle.turn = 2
        raichu = MagicMock()
        raichu.species = "Raichu"
        raichu.fainted = False
        self.battle.active_pokemon = raichu
        self.battle.team = {"Pikachu": self.active_mon, "Raichu": raichu}
        self.battle.available_switches = [raichu]
        
        # Mock the SwitchDetection to return voluntary
        with MagicMock() as mock_detect:
            from utils.gen3_utils import SwitchDetection
            import agents.training.reward_manager
            agents.training.reward_manager.SwitchDetection.get_switch_type = MagicMock(return_value=(True, True, True))
            
            self.manager.record_action(self.battle, 1) # Action 1 is switch to slot 1
            reward = self.manager.process_turn_reward(self.battle, 0.0)
            
            self.assertGreater(reward, 0.0)
            self.assertEqual(self.manager.switch_count, 1)

    def test_get_episode_metrics(self):
        self.manager.attack_count = 5
        self.manager.switch_count = 2
        self.manager.total_reward = 15.5
        self.battle.turn = 20
        self.battle.won = True
        
        metrics = self.manager.get_episode_metrics(self.battle)
        self.assertEqual(metrics["reward"], 15.5)
        self.assertEqual(metrics["status"], "WIN")
        self.assertEqual(metrics["attacks"], 5)
        self.assertEqual(metrics["switches_voluntary"], 2)

if __name__ == "__main__":
    unittest.main()
