"""SB3 callback that sends rollout metrics to the launcher via a pipe fd.

When run without the launcher (LAUNCHER_METRICS_FD absent), this is a no-op.
Safe to include unconditionally in the callbacks list.
"""

import time

from stable_baselines3.common.callbacks import BaseCallback
from main.launcher.ipc import init as init_pipe, send_metrics, close as close_pipe


class MetricsExporterCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        init_pipe()  # grab FD now so close() can flush at training end
        self._start_time: float | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        payload = {
            k: float(v)
            for k, v in self.logger.name_to_value.items()
            if isinstance(v, (int, float))
        }
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf:
            from stable_baselines3.common.utils import safe_mean
            payload["rollout/ep_rew_mean"] = float(safe_mean([ep["r"] for ep in buf]))
            payload["rollout/ep_len_mean"] = float(safe_mean([ep["l"] for ep in buf]))
        # SB3 logs time/fps and time/total_timesteps after on_rollout_end fires,
        # so they're never in name_to_value — compute them here instead.
        payload["time/total_timesteps"] = float(self.num_timesteps)
        if self._start_time is not None and self.num_timesteps > 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed > 0:
                payload["time/fps"] = float(self.num_timesteps / elapsed)
        payload["_step"] = int(self.num_timesteps)
        send_metrics(payload)

    def _on_training_end(self) -> None:
        close_pipe()
