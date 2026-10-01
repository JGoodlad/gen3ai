"""The self-play POOL and the PFSP weights as DECLARED supplies — `gen3_supply_guard_v2`.

A mixin for :class:`~agents.training.selfplay_callback.SelfPlayCallback`, kept beside it so the
1,000-line callback does not grow. The class it closes: `--self-play` with a pool that never seeds
trains against the BOT fallback for the whole run — `ai_v12_27_ladder_ctrl10M_shaped_dense` ran 10M
steps with 0 snapshots (its `win_rate_vs_bots` ended at 0.53, under the 0.55 seeding gate) beside
self-play siblings — and `--pfsp-scale` with no measured sentinel weights the pool uniformly (or by
stale rates). Both are judged once per EVAL CYCLE, FAILED cycles included (a failed cycle cannot
seed the pool or measure a sentinel), by a `lever_supply.DryStreakGuard`; the counters are persisted
in the pool's `summary.json` under ``supply_guard`` (keyed to THIS run dir, because a fork's pool is
seeded with its parent's summary), so the streak is a RUN-level count — an eval cycle is 2M steps
and a per-process count would reset at every launcher restart before a floor of 3 could trip.

The FATAL is raised LAST in `_collect_pending`, after the cycle is fully recorded, and never from
the graceful-shutdown drain (`_draining`): a completed run is not turned into a FATAL from inside
its own shutdown path. The end-of-segment summary is LOUD at zero either way.
"""
from __future__ import annotations

import os

from agents.training.lever_supply import DryStreakGuard, LeverStarvedError, loud, record_scalar


class SelfPlaySupplyMixin:
    """Needs ``self._pool``, ``self._model_dir``, ``self._pfsp_scale``, ``self.win_rate_vs_bots``,
    ``self._self_play_start_wr``, ``self._pfsp_winrate_ema`` and an SB3 ``self.logger``."""

    def _init_supply_guards(self, pool_starve_cycles: int, pfsp_starve_cycles: int) -> None:
        gs = self._load_supply_state()
        self._draining = False
        self._failed_eval_cycles = int(gs.get("failed_eval_cycles", 0))
        self._pool_guard = DryStreakGuard("self_play_pool", pool_starve_cycles,
                                          state=gs.get("self_play_pool"), emit=loud)
        self._pfsp_guard = (DryStreakGuard("pfsp", pfsp_starve_cycles, state=gs.get("pfsp"),
                                           emit=loud)
                            if self._pfsp_scale > 0.0 else None)

    def _announce_supply_guards(self) -> None:
        for g in (self._pool_guard, self._pfsp_guard):
            if g is not None:
                loud(g.announce())

    def _supply_run_key(self) -> str:
        return os.path.abspath(self._model_dir) if self._model_dir else ""

    def _load_supply_state(self) -> dict:
        """The persisted guard counters — only THIS run's: a fork's pool is seeded with its
        parent's summary.json, and inheriting the parent's streak would judge the fork by cycles
        it never ran."""
        try:
            st = self._pool.load_summary().get("supply_guard") or {}
        except Exception:  # noqa: BLE001 — a missing/corrupt summary is a fresh count
            return {}
        return st if st.get("run") == self._supply_run_key() else {}

    def _observe_supply(self, step: int, *, failed: bool, n_sentinels: int,
                        n_measured: int, persist: bool = True) -> None:
        """COUNT one eval cycle for the pool and (if on) PFSP — never raises; :meth:`_judge_supply`
        does. ``persist=False`` when the caller writes `summary.json` itself right after (it merges
        :meth:`_supply_summary_fields` into its own write, so the counters land BEFORE the judge)."""
        if failed:
            self._failed_eval_cycles += 1
        empty = self._pool.is_empty()
        start = self._self_play_start_wr
        if failed:
            pool_why = ("this eval cycle FAILED (no results from any worker) — a failed cycle "
                        "cannot seed the pool; read the .eval_runs worker logs")
        else:
            pool_why = (f"win_rate_vs_bots {self.win_rate_vs_bots:.3f} is below the seeding gate "
                        f"--self-play-start-wr {start:g}, so the pool is never seeded. Lower the "
                        f"gate, or drop --self-play if a bots-only run is the experiment")
        pfsp_live = n_sentinels > 0
        pfsp_why = ("the eval cycle FAILED" if failed else
                    f"{n_sentinels} sentinel(s) were launched and none returned a result")
        record_scalar(self, "eval/failed_cycles_total", self._failed_eval_cycles)
        try:
            self._pool_guard.observe(0 if empty else 1, why=pool_why,
                                     may_raise=False)
            if self._pfsp_guard is not None:
                self._pfsp_guard.observe(n_measured, live=pfsp_live, why=pfsp_why,
                                         may_raise=False)
                if pfsp_live and n_measured == 0:
                    loud(f"⚠️  [PFSP] step {step:,}: no sentinel win-rate measured this cycle — "
                         f"--pfsp-scale {self._pfsp_scale:g} is weighting by STALE rates "
                         f"({len(self._pfsp_winrate_ema)} tracked) or UNIFORM (none).")
        finally:
            if persist:
                self._persist_supply_state()
        record_scalar(self, "supply/selfplay_pool_dry_streak", self._pool_guard.streak)
        if self._pfsp_guard is not None:
            record_scalar(self, "supply/pfsp_dry_streak", self._pfsp_guard.streak)

    def _judge_supply(self) -> None:
        """Raise the typed FATAL when a guard's streak reached its floor — never while draining."""
        if self._draining:
            return
        for g in (self._pool_guard, self._pfsp_guard):
            if g is not None and g.enabled and g.streak >= g.starve_cycles:
                msg = g.fatal_message() + (f"\n  Failed eval cycles so far: "
                                           f"{self._failed_eval_cycles}.")
                raise LeverStarvedError(msg)

    def _supply_summary_fields(self) -> dict:
        """``{"supply_guard": …}`` — merged into `summary.json` (THIS run's counters only)."""
        st = {"run": self._supply_run_key(), "failed_eval_cycles": self._failed_eval_cycles,
              "self_play_pool": self._pool_guard.state()}
        if self._pfsp_guard is not None:
            st["pfsp"] = self._pfsp_guard.state()
        return {"supply_guard": st}

    def _persist_supply_state(self) -> None:
        if not self._model_dir:
            return
        try:
            self._pool.persist_summary(**self._supply_summary_fields())
        except Exception as e:  # noqa: BLE001 — losing the count must not kill training
            print(f"⚠️  [SUPPLY] could not persist the self-play supply counters: {e}", flush=True)

    def supply_summary_lines(self) -> list:
        lines = list(self._pool_guard.summary_lines())
        if self._pfsp_guard is not None:
            lines += self._pfsp_guard.summary_lines()
        if self._failed_eval_cycles:
            lines.append(f"⚠️  [SUPPLY] {self._failed_eval_cycles} eval cycle(s) FAILED outright "
                         f"(no results) over this run")
        return lines
