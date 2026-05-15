import os
import json
import asyncio
import threading
import traceback
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration

from agents.inference.player import RLPlayer
from agents.training.battle_recorder import BattleRecorder
from agents.training.stall import StallConfig

BATTLE_FORMAT = "gen3ou"


class StatTrackingRLPlayer(RLPlayer):
    """
    RLPlayer that records every model invocation to a BattleRecorder.
    All tracking logic lives in BattleRecorder — this class is purely a hook.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorders: dict[str, BattleRecorder] = {}

    def choose_move(self, battle):
        try:
            if battle.battle_tag != self._last_battle_tag:
                self._last_battle_tag = battle.battle_tag
                self._stall_logger.reset()
            if battle.turn >= self._stall_logger.threshold:
                self._stall_logger.log_once(battle, suffix="REPLAY_STALL")
                return ForfeitBattleOrder()

            if battle.battle_tag not in self._recorders:
                self._recorders[battle.battle_tag] = BattleRecorder(battle.battle_tag)

            idx, probs, mask = self._predict_best_action(battle, stochastic=True)
            self._recorders[battle.battle_tag].record(battle, idx, probs, mask)
            return self.action_to_order(idx, battle)
        except Exception as e:
            print(f"\n⚠️ [REPLAY] choose_move() failed at turn {battle.turn}: {e}")
            return ForfeitBattleOrder()


class ReplayCallback(BaseCallback):
    """
    Captures full battle replays and detailed per-invocation JSON summaries
    at training milestones. Saves them under <model_dir>/replays/step_N/.
    """

    def __init__(self, model_dir, mappings, trainee_teambuilder, opponent_teambuilder,
                 save_freq=100000, n_replays=3, stall_config: StallConfig = None, verbose=0):
        super().__init__(verbose)
        self.model_dir = model_dir
        self.mappings = mappings
        self.trainee_teambuilder = trainee_teambuilder
        self.opponent_teambuilder = opponent_teambuilder
        self.save_freq = save_freq
        self.n_replays = n_replays
        self.stall_config = stall_config or StallConfig()
        self.replay_dir = os.path.join(model_dir, "replays")
        os.makedirs(self.replay_dir, exist_ok=True)
        self.last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < 1_000_000:
            interval = 200_000
        elif self.num_timesteps < 10_000_000:
            interval = 1_000_000
        else:
            interval = 2_000_000

        trigger = (self.last_save == 0 and self.num_timesteps > 0) or \
                  (self.num_timesteps // interval) > (self.last_save // interval)

        if trigger:
            self.last_save = self.num_timesteps
            step_dir = os.path.join(self.replay_dir, f"step_{self.num_timesteps}")
            os.makedirs(step_dir, exist_ok=True)
            print(f"\n🎥 [REPLAY] Step {self.num_timesteps}: Recording {self.n_replays} games to {step_dir}...")
            thread = threading.Thread(target=self._run_async_battles, args=(step_dir,), daemon=True)
            thread.start()
            thread.join()

        return True

    def _run_async_battles(self, step_dir):
        async def run_it():
            ts = datetime.now().strftime('%H%M%S')

            replay_player = StatTrackingRLPlayer(
                model=self.model,
                team=self.trainee_teambuilder,
                battle_format=BATTLE_FORMAT,
                server_configuration=LocalhostServerConfiguration,
                mappings=self.mappings,
                stall_config=self.stall_config,
                account_configuration=AccountConfiguration(f"Replay{ts}", "password"),
                save_replays=step_dir,
            )

            opponent = SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT,
                team=self.opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"RepOpp{ts}", "password"),
            )

            await replay_player.battle_against(opponent, n_battles=self.n_replays)

            html_files = sorted(f for f in os.listdir(step_dir) if f.endswith(".html"))
            for i, (tag, recorder) in enumerate(replay_player._recorders.items()):
                battle = replay_player._battles.get(tag)
                if not battle:
                    continue

                recorder.finalize(battle)
                summary = recorder.to_summary(battle, self.num_timesteps)

                with open(os.path.join(step_dir, f"battle_{i + 1}_summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)

                if i < len(html_files):
                    os.rename(
                        os.path.join(step_dir, html_files[i]),
                        os.path.join(step_dir, f"battle_{i + 1}_replay.html"),
                    )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def exception_handler(loop, context):
            msg = context.get("exception", context["message"])
            print(f"\n🛑 [REPLAY FATAL] Background task failed: {msg}")
            os._exit(1)

        loop.set_exception_handler(exception_handler)

        try:
            loop.run_until_complete(run_it())
        except Exception as e:
            print(f"\n🛑 [REPLAY CRASH] Step {self.num_timesteps} failed: {e}")
            traceback.print_exc()
            os._exit(1)
        finally:
            loop.close()
