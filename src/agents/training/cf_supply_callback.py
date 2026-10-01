"""The learn()-time half of `gen3_supply_guard_v1` — see `agents.training.cf_supply`.

Once per rollout (i.e. once per completed train(), whose ONE disk poll is what moves the buffer's
ingest counters) it hands the buffer to :class:`~agents.training.cf_supply.CfSupplyGuard`, which
raises `CfLabelSupplyError` → exit `FATAL_SUPPLY` (5) when a live coefficient's stream starved past
its declared floor, or at once when a spawned producer exited. At the end of training it prints the
per-stream totals — LOUD at zero — and stops the spawned producer.
"""
from __future__ import annotations

from typing import Callable, Optional

from stable_baselines3.common.callbacks import BaseCallback

from agents.training.cf_supply import CfProducerSupply, CfSupplyGuard, any_checkpoint


class CfSupplyCallback(BaseCallback):
    def __init__(self, guard: CfSupplyGuard, *, run_dir: str,
                 supply: Optional[CfProducerSupply] = None,
                 checkpoint_probe: Callable[[str], bool] = any_checkpoint,
                 emit: Callable[[str], None] = print, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.guard = guard
        self.run_dir = run_dir
        self.supply = supply
        self._probe = checkpoint_probe
        self._emit = emit
        self._rollouts = 0
        self._ckpt_seen = False

    def _buffer(self):
        return getattr(self.model, "_cf_buffer", None)

    def _on_training_start(self) -> None:
        g = self.guard
        if not g.enabled:
            self._emit("⚠️  [SUPPLY] the in-flight cf supply guard is DISABLED "
                       "(--cf-supply-starve-cycles 0): a dead producer will NOT stop this run.")
        else:
            self._emit(f"🛡️  [SUPPLY] in-flight guard: {', '.join(g.consumers)} must accept a "
                       f"label within {g.starve_cycles} train() cycles AND "
                       f"{g.starve_s / 60.0:g} min once a checkpoint exists, else FATAL_SUPPLY (5)")

    def _on_rollout_start(self) -> None:
        # The FIRST rollout precedes any train(), so there is no poll to judge yet.
        self._rollouts += 1
        if self._rollouts == 1:
            if self.supply is not None:
                self.supply.check_alive()
            return
        buf = self._buffer()
        if buf is None:
            return
        if not self._ckpt_seen:
            self._ckpt_seen = bool(self._probe(self.run_dir))
        self.guard.observe(buf, checkpoint_present=self._ckpt_seen)
        self.logger.record("cf/supply_starved_cycles", float(self.guard.starved_cycles()))
        self.logger.record("cf/supply_guard_armed", float(self.guard.armed))

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        for line in self.guard.summary(self._buffer()):
            self._emit(line)
        if self.supply is not None and self.supply.proc is not None:
            rc = self.supply.stop()
            self._emit(f"🏭 [SUPPLY] stopped the spawned cf_producer (pid {self.supply.pid}, "
                       f"rc={rc})")
