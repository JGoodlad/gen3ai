"""SB3 callback that sends rollout metrics to the launcher via a pipe fd.

When run without the launcher (LAUNCHER_METRICS_FD absent), this is a no-op.
Safe to include unconditionally in the callbacks list.
"""

import json
import os
from stable_baselines3.common.callbacks import BaseCallback


class MetricsExporterCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._pipe = None
        raw = os.environ.get("LAUNCHER_METRICS_FD", "")
        if raw.lstrip("-").isdigit():
            fd = int(raw)
            if fd >= 0:
                try:
                    self._pipe = os.fdopen(fd, "w", buffering=1)
                except OSError:
                    pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._pipe:
            return
        try:
            payload = {
                k: float(v)
                for k, v in self.logger.name_to_value.items()
                if isinstance(v, (int, float))
            }
            # rollout/ep_rew_mean isn't logged into name_to_value until after this
            # callback fires, so read directly from ep_info_buffer.
            buf = getattr(self.model, "ep_info_buffer", None)
            if buf:
                from stable_baselines3.common.utils import safe_mean
                payload["rollout/ep_rew_mean"] = float(safe_mean([ep["r"] for ep in buf]))
                payload["rollout/ep_len_mean"] = float(safe_mean([ep["l"] for ep in buf]))
            payload["_step"] = int(self.num_timesteps)
            self._pipe.write(json.dumps(payload) + "\n")
        except (OSError, BrokenPipeError):
            self._pipe = None

    def _on_training_end(self) -> None:
        if self._pipe:
            try:
                self._pipe.close()
            except OSError:
                pass
            self._pipe = None
