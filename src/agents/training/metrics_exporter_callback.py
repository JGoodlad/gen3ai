"""SB3 callback that sends rollout metrics to the launcher via a pipe fd.

When run without the launcher (LAUNCHER_METRICS_FD absent), this is a no-op.
Safe to include unconditionally in the callbacks list.
"""

from stable_baselines3.common.callbacks import BaseCallback
from main.launcher.ipc import init as init_pipe, send_metrics, close as close_pipe


class MetricsExporterCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        init_pipe()  # grab FD now so close() can flush at training end

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
        payload["_step"] = int(self.num_timesteps)
        send_metrics(payload)

    def _on_training_end(self) -> None:
        close_pipe()
