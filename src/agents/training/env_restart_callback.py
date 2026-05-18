import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from agents.training.watchdog import start_subprocess_watchdog

_SECONDS_PER_HOUR = 3600


class VecEnvRestartCallback(BaseCallback):
    """Recycle all SubprocVecEnv workers at a fixed wall-clock interval.

    Fires at rollout-end boundaries so the in-flight rollout buffer is always
    complete before the swap. Model weights, optimizer state, and the rollout
    buffer are untouched.

    Args:
        make_factories: Callable that returns a fresh list of env _init
            functions. Called each time workers are recycled so that new
            workers get fresh Showdown usernames and connections.
        restart_interval_hours: Hours between worker restarts. Set to 0 to
            disable.
    """

    def __init__(self, make_factories, restart_interval_hours: float, verbose: int = 0):
        super().__init__(verbose)
        self._make_factories = make_factories
        self._restart_interval = restart_interval_hours * _SECONDS_PER_HOUR
        self._last_restart = time.monotonic()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self._restart_interval <= 0:
            return
        if not hasattr(self.model.env, 'processes'):
            return
        elapsed = time.monotonic() - self._last_restart
        if elapsed < self._restart_interval:
            return
        n = self.model.env.num_envs
        print(f"\n♻️  [EnvRestart] {elapsed / _SECONDS_PER_HOUR:.2f}h elapsed — recycling {n} subprocess workers...")
        old_env = self.model.env
        new_env = SubprocVecEnv(self._make_factories())
        self.model.set_env(new_env)
        self.model._last_obs = new_env.reset()
        self.model._last_episode_starts = np.ones(n, dtype=bool)
        start_subprocess_watchdog(new_env, label="train_env")
        old_env.close()
        self._last_restart = time.monotonic()
        print("♻️  [EnvRestart] Workers recycled. Resuming training.")
