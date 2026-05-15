import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StallConfig:
    """Configuration for stall detection and logging."""
    threshold: int = 250
    output_dir: Optional[str] = None


class StallLogger:
    """
    Saves an HTML replay the first time a battle stalls and logs to stderr.

    Call log_once() from action_to_order when the stall threshold is reached.
    Call reset() at the start of each episode.
    """

    def __init__(self, config: Optional[StallConfig] = None):
        self._config = config or StallConfig()
        self._logged = False

    @property
    def threshold(self) -> int:
        return self._config.threshold

    def reset(self) -> None:
        self._logged = False

    def log_once(self, battle, suffix: str = "") -> None:
        if self._logged or not self._config.output_dir:
            return
        self._logged = True
        try:
            os.makedirs(self._config.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix_str = f"_{suffix}" if suffix else ""
            filename = f"stall_{battle.battle_tag}_{ts}{suffix_str}.html"
            path = os.path.join(self._config.output_dir, filename)
            battle.save_replay(path)
            sys.stderr.write(
                f"\n[STALL LOGGED] Battle {battle.battle_tag} lasted "
                f"{battle.turn} turns. Replay saved to {path}\n"
            )
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Failed to save stall replay: {e}\n")
            sys.stderr.flush()
