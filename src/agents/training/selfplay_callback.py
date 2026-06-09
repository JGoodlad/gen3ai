"""Self-play callback: non-blocking bot + pool eval with snapshot promotion.

Mirrors ``PerOpponentEvalCallback``'s frozen-snapshot **subprocess** pattern so eval never
pauses training, extended with pool sentinels + promotion. On the training thread, per
cycle, it only freezes the live weights (`model.save`), picks the sentinels, spawns the
workers, and — at collect — does one cheap `opponent_default_stats` IPC. Everything else
(battles, sentinel model loads, inference) runs in the worker processes.

  1. **Launch** (trigger step): `model.save` the live weights to disk and spawn
     `--eval-workers` `main.eval_worker` subprocesses that **work-steal** BOTH the bot
     roster AND up to 5 pool sentinels from one shared pool. Training continues immediately.
  2. **Collect** (a later poll, when all workers finish): merge per-opponent +
     per-sentinel results → ``win_rate_vs_bots`` / ``win_rate_vs_pool`` /
     ``sentinel_monotonicity``; record to TensorBoard + the TUI + metadata.json; then
       - persist ``win_rate_vs_bots`` (feeds ``heuristic_fraction`` next run),
       - save best model (copy the frozen snapshot),
       - **promote** the FROZEN snapshot into the pool if ``win_rate_vs_pool`` clears the
         threshold (the live model has advanced since launch — promoting it would capture
         the wrong weights).
  3. **Drain**: graceful shutdown waits for the in-flight cycle and records it (`drain()`),
     wired exactly like the bot path.

Emits launcher events on promotion and bot-regression warnings (⚠️).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback
from poke_env.ps_client import LocalhostServerConfiguration

from agents.model.snapshot import record_eval_results
from agents.training.eval_callback import (
    _EVAL_CYCLE_TIMEOUT_SEC,
    _EVAL_SUBPROCESS_CONCURRENCY,
    EVAL_FREQ_STEPS,
    EVAL_GAMES,
    RANDOM_OPPONENT_NAME,
    _b36,
    bot_mean,
    build_bot_eval_block,
    build_externals_block,
    copy_run_config_to_best_model,
    write_best_model_sidecar,
    eval_opponent_names,
    eval_run_nonce,
    external_aggregate,
    kill_eval_workers,
    record_elo,
    record_external_elos,
    record_per_opponent,
    latest_recorded_eval_step,
    merge_eval_results,
    persist_eval_snapshot,
    prune_eval_traces,
    replay_last_eval_to_tui,
    spawn_eval_workers,
    write_eval_manifest,
)
from agents.training.artifact_retention import (
    prune_run_artifacts, KEEP_STALLS_DEFAULT, KEEP_CRASHES_DEFAULT,
)
from agents.training.snapshot_pool import (
    SnapshotPool, heuristic_fraction, HEURISTIC_FLOOR, SELF_PLAY_START, SELF_PLAY_FULL,
)
from main.launcher.ipc import emit, send_event, send_metrics

# Consecutive eval cycles a stable opponent's win_rate must stay ≥ the mastery threshold before the
# (one-way) challenge→floor flip. At EVAL_GAMES=100 the win-rate 1σ band is ±0.04, so a single noisy
# cycle near the threshold could otherwise permanently demote an opponent the trainee hasn't really
# mastered — a 2-cycle confirm guards against that.
_MASTERY_CONFIRM_CYCLES = 2

# Regression guard: warn if a bot the agent was beating well drops below this.
_REGRESSION_WARN_THRESHOLD = 0.60
_REGRESSION_TRIGGER_THRESHOLD = 0.70  # must have reached this first


def _monotonicity_score(win_rates: list[float]) -> float:
    """Kendall's τ for sentinel win rates (index 0 = most recent = hardest).

    Returns +1.0 when perfectly monotone (old snapshots easiest), −1.0 when
    inverted (old snapshots hardest). Below ~0.6 indicates potential cycling.
    """
    n = len(win_rates)
    if n < 2:
        return 1.0
    concordant = sum(
        win_rates[i] <= win_rates[j]
        for i in range(n)
        for j in range(i + 1, n)
    )
    total = n * (n - 1) // 2
    return 2.0 * concordant / total - 1.0


def _distill_step_tag(step: int) -> str:
    """A snapshot's step as a compact label for the Events panel ('seed' / '47.0M')."""
    return "seed" if step == 0 else f"{step / 1e6:.1f}M"


def _distill_job_event_text(h: dict) -> "str | None":
    """Pure formatter: one harvested distill job -> an Events-panel line (or None).

    ``h`` is a ``ReconcileResult.harvested`` entry (manager.py): keys ``step, action,
    rung, n_rungs, next_rung?, speedup, h2h``. Kept pure (no I/O) so it is unit-testable.
    """
    tag = _distill_step_tag(h["step"])
    n = h.get("n_rungs", 1)
    h2h = h.get("h2h")
    speed = h.get("speedup") or 0.0
    action = h.get("action")
    if action == "deployed":
        extra = f"h2h {h2h:.2f}, " if isinstance(h2h, (int, float)) else ""
        return f"⚗ Distilled {tag} ✓ — {extra}{speed:.1f}× faster (rung {h.get('rung', 0) + 1}/{n})"
    if action == "escalated":
        reason = f"h2h {h2h:.2f}" if isinstance(h2h, (int, float)) else "low fidelity"
        return f"⚗ Distill {tag} missed gate ({reason}) — escalating to rung {h.get('next_rung', 0) + 1}/{n}"
    if action == "exhausted":
        return f"⚗ Distill {tag} exhausted the ladder — kept as a full opponent"
    return None


class SelfPlayCallback(BaseCallback):
    """Non-blocking bot + pool eval callback with snapshot promotion.

    Args:
        pool: The SnapshotPool to promote into and eval against.
        model_dir: Run directory (snapshot scratch under ``.eval_runs``, forensic traces,
            metadata.json). None disables eval (nowhere to snapshot/collect).
        server_config / showdown_port: threaded to the eval workers.
        best_model_save_path: directory to copy the best frozen snapshot into.
        promote_threshold: ``win_rate_vs_pool`` threshold to trigger promotion.
        self_play_temp: sampling temperature for the (stochastic) sentinel opponents —
            kept equal to the training opponents' ``--self-play-temp`` so eval sentinels
            behave EXACTLY as in training.
        n_workers / eval_device / eval_concurrency: subprocess eval-pool knobs.
        keep_eval_snapshots / keep_eval_trace_steps: forensic retention caps.
        debug: tiny/fast eval cadence so a short CPU smoke exercises seed → pool eval →
            promotion (the real schedule's 1M-step floor never fires in a 20k smoke).
    """

    # Wait for in-flight workers on a graceful shutdown (10 min) — long enough for a full
    # CPU eval; the restart grace window (--restart-grace-minutes, 20 min) covers it.
    _DRAIN_TIMEOUT_SEC = 600

    def __init__(
        self,
        pool: SnapshotPool,
        *,
        model_dir: str | None = None,
        server_config=LocalhostServerConfiguration,
        showdown_port: int | None = None,
        use_showdown_bridge: bool = False,
        best_model_save_path: str | None = None,
        promote_threshold: float = 0.65,
        self_play_temp: float = 1.0,
        eval_sentinel_greedy: bool = False,
        heuristic_floor: float = HEURISTIC_FLOOR,
        self_play_start_wr: float = SELF_PLAY_START,
        self_play_full_wr: float = SELF_PLAY_FULL,
        n_workers: int = 3,
        eval_device: str = "cpu",
        distill_opponents: bool = False,
        distill_device: str = "cpu",
        eval_concurrency: int = _EVAL_SUBPROCESS_CONCURRENCY,
        keep_eval_snapshots: int = 10,
        keep_eval_trace_steps: int = 20,
        keep_stalls: int = KEEP_STALLS_DEFAULT,
        keep_crashes: int = KEEP_CRASHES_DEFAULT,
        resume_eval_metadata: str | None = None,
        fixed_opponents: "list | None" = None,
        stable_opponent_mastered_wr: float = 0.80,
        debug: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._pool = pool
        # Stable cross-run opponents (FixedOpponentEntry list) — an extra ext_ eval matchup each
        # cycle, kept out of win_rate_vs_bots / win_rate_vs_pool / the ELO fit / promotion. In the
        # TRAINING mix they are challenge opponents until mastered, then floor (see _push_stable_mastered).
        self._fixed_opponents = list(fixed_opponents or [])
        self._stable_opponent_mastered_wr = float(stable_opponent_mastered_wr)
        self._stable_mastered: set[str] = set()  # labels mastered this run (monotonic)
        self._stable_mastery_streak: dict[str, int] = {}  # consecutive ≥-threshold cycles per label
        self._model_dir = model_dir
        self._server_config = server_config
        self._showdown_port = showdown_port
        # Bridge eval: workers play in-process via run_local_battles (no server connection).
        self._use_showdown_bridge = use_showdown_bridge
        self.best_model_save_path = best_model_save_path
        self._promote_threshold = promote_threshold
        self._self_play_temp = self_play_temp
        # Eval the pool sentinels greedy (best-vs-best) instead of stochastic — removes the
        # greedy-trainee-vs-stochastic-sentinel handicap so win_rate_vs_pool / snapshot ELO are
        # honest. Eval-only; the TRAINING opponents (built in the env factory) stay stochastic.
        self._eval_sentinel_greedy = eval_sentinel_greedy
        # Curriculum (transition + floor) — the live per-episode self_play_fraction the eval
        # pushes is 1 - heuristic_fraction(win_rate, floor/start/full). Defaults = the module
        # constants (original curve); raised floor / later `full` keeps the coverage-punishing
        # bots in the TRAINING mix longer (#2). Threaded from --heuristic-floor /
        # --self-play-start-wr / --self-play-full-wr.
        self._heuristic_floor = heuristic_floor
        self._self_play_start_wr = self_play_start_wr
        self._self_play_full_wr = self_play_full_wr
        self._n_workers = max(1, n_workers)
        self._eval_device = eval_device
        self._eval_concurrency = eval_concurrency
        self._keep_eval_snapshots = max(0, keep_eval_snapshots)
        self._keep_eval_trace_steps = max(0, keep_eval_trace_steps)
        # Bound the per-run stalls/ + crashes/ dirs each cycle (0 = keep all).
        self._keep_stalls = max(0, keep_stalls)
        self._keep_crashes = max(0, keep_crashes)
        self._resume_eval_metadata = resume_eval_metadata
        self._debug = debug

        self._last_eval_step = 0
        self._best_aggregate_win_rate = -1.0
        # The in-flight eval cycle, or None.
        self._pending: dict | None = None
        self._eval_root: str | None = None
        # Per-PROCESS account-name nonce (+ per-cycle counter) so two launcher restarts
        # never reuse Showdown account names. The old step-derived tag collided across
        # restarts (the resume re-eval always fires at the same step) and hung a worker on a
        # lingering challenge — wedging self-play eval permanently (the hang pins _pending).
        self._eval_run_nonce = eval_run_nonce()
        self._eval_cycle = 0
        # Parent-side fatal errors only; worker crashes log-and-continue (bot-path parity).
        self.abort_fn = None

        # Shared state: written by collect, read by env factory on next restart.
        self.win_rate_vs_bots: float = pool.load_persisted_win_rate()
        self._bot_peak: dict[str, float] = {}      # resets each run; TensorBoard has history
        self._regression_active: set[str] = set()  # bots currently in warned state
        # Pool generation — bumped whenever the pool changes (seed/promote); pushed to the env
        # workers via env_method so they re-scan the pool dir and pick up new snapshots live.
        self._pool_generation = 0

        # ── Opponent distillation (all-or-nothing; distill_integration.md §8) ──
        # One idempotent reconcile loop keeps the on-disk distilled set == the pool's snapshots:
        # backfill on enable + steady-state are the same call; no-op when nothing's missing.
        self._distill_device = distill_device
        self._distill_mgr = None
        self._last_distill_push = None
        # Tracks the live atomic all-or-nothing state so a transition (full↔100% distilled)
        # fires exactly one Events-panel line. None = not yet known (no transition logged).
        self._distill_deployed = None
        self._last_distill_reconcile = 0
        self._distill_reconcile_interval = 4000 if debug else 100_000
        if distill_opponents:
            from agents.training.distill.manager import DistilledOpponentManager
            self._distill_mgr = DistilledOpponentManager(
                ready_steps_fn=self._pool.gate_passed_steps,
                list_artifacts_fn=self._pool.distilled_artifact_steps,
                run_distill_fn=self._spawn_distill,
                poll_fn=self._poll_distill,
                remove_fn=self._pool.remove_distilled,
                recover_fn=self._pool.failed_distill_manifests,   # restart-safe escalation
                max_concurrent=max(1, n_workers),
            )
            print("[DISTILL] opponent distillation ENABLED — all-or-nothing; "
                  "the first reconcile backfills the pool, then steady-state.")

    # ── SB3 lifecycle ──────────────────────────────────────────────────────

    def _init_callback(self) -> None:
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self._model_dir is not None:
            self._eval_root = os.path.join(self._model_dir, ".eval_runs")
            os.makedirs(self._eval_root, exist_ok=True)
        # NOTE: seeding is gated on competence (win rate ≥ SELF_PLAY_START) — it is NOT done
        # here. train_rl_agent's _maybe_seed_pool seeds at startup if already competent, and
        # _collect_pending seeds the moment the model first crosses the threshold mid-run. So a
        # weak model never seeds a near-random opponent into the pool.
        # Resumed run: re-publish the last eval so the TUI panel isn't blank until the
        # next cycle (which can be millions of steps away).
        replay_last_eval_to_tui(self._model_dir, self._resume_eval_metadata)
        # Restore the last eval step so a resume doesn't re-eval the same checkpoint
        # immediately (it waits for the next cadence boundary instead).
        self._last_eval_step = latest_recorded_eval_step(self._model_dir, self._resume_eval_metadata)

    def _schedule(self) -> tuple[int, int]:
        if self._debug:
            return 4000, 3  # fast cadence for --debug --self-play smoke tests
        return EVAL_FREQ_STEPS, EVAL_GAMES

    def _on_step(self) -> bool:
        if self._pending is not None:
            now = time.monotonic()
            if self._all_done(self._pending):
                self._collect_pending()
            elif now - self._pending.get("launched_at", now) > _EVAL_CYCLE_TIMEOUT_SEC:
                self._abort_pending_cycle()   # hung worker → don't wedge eval forever
        # Reconcile distilled opponents on a throttle (harvests finished jobs + flips the atomic
        # full↔distilled switch promptly during backfill, without waiting for the next eval cycle).
        if (self._distill_mgr is not None
                and self.num_timesteps - self._last_distill_reconcile >= self._distill_reconcile_interval):
            self._last_distill_reconcile = self.num_timesteps
            self._reconcile_distill()
        if self.num_timesteps == 0:
            return True
        freq, _ = self._schedule()
        if (self.num_timesteps // freq) > (self._last_eval_step // freq):
            self._last_eval_step = self.num_timesteps
            if self._pending is not None:
                print(f"[SELFPLAY EVAL] step {self.num_timesteps:,}: previous eval "
                      f"(step {self._pending['step']:,}) still running — skipping this cycle")
            else:
                self._launch_eval()
        return True

    # ── Launch ───────────────────────────────────────────────────────────────

    @staticmethod
    def _all_done(pending: dict) -> bool:
        return all(w["proc"].poll() is not None for w in pending["procs"])

    def _launch_eval(self) -> None:
        if self._eval_root is None:
            return  # no model_dir → nowhere to snapshot/collect; eval disabled
        _, n_games = self._schedule()
        step = self.num_timesteps
        run_dir = os.path.join(self._eval_root, f"step_{step}")
        claim_dir = os.path.join(run_dir, "claims")
        os.makedirs(claim_dir, exist_ok=True)

        snapshot_base = os.path.join(run_dir, "snapshot")
        self.model.save(snapshot_base)  # freeze live weights; SB3 appends .zip
        snapshot_zip = snapshot_base + ".zip"

        bot_names = eval_opponent_names()
        sentinel_entries = self._pool.sentinel_entries(n=5)
        sentinels = [
            {"label": f"sentinel_{i}", "path": str(e.path), "step": e.step}
            for i, e in enumerate(sentinel_entries)
        ]
        sentinel_labels = [s["label"] for s in sentinels]
        # Stable cross-run opponents (ext_<label>) — an extra fixed yardstick alongside the pool.
        fixed_cfgs = [e.to_cfg() for e in self._fixed_opponents]
        fixed_labels = [f["label"] for f in fixed_cfgs]

        # Record exactly which model produced this cycle's traces (the prober reads this).
        write_eval_manifest(self._model_dir, step,
                            opponents=bot_names + sentinel_labels + fixed_labels, n_games=n_games)
        # Process-unique account tag (per-process nonce + per-cycle counter), NOT the step:
        # the resume re-eval fires at the same step every restart, so a step tag collided
        # across restarts and hung a worker on a lingering challenge (wedging eval forever).
        self._eval_cycle += 1
        cycle_tag = f"{self._eval_run_nonce}{_b36(self._eval_cycle, 1)}"
        n_items = len(bot_names) + len(sentinels) + len(fixed_cfgs)
        n_workers = max(1, min(self._n_workers, n_items))
        base_cfg = {
            "snapshot": snapshot_zip,
            "port": self._showdown_port,
            "use_showdown_bridge": self._use_showdown_bridge,
            "model_dir": self._model_dir,
            "step": step,
            "n_games": n_games,
            "opponent_pool": bot_names,        # bots; workers steal from the combined pool
            "sentinels": sentinels,            # pool snapshots to play (stochastic, or greedy below)
            "fixed_opponents": fixed_cfgs,     # stable cross-run opponents (ext_<label>)
            "self_play_temp": self._self_play_temp,
            "eval_sentinel_greedy": self._eval_sentinel_greedy,
            "claim_dir": claim_dir,
            "result_dir": run_dir,             # writes result__<item>.json here
            "concurrency": self._eval_concurrency,
            "device": self._eval_device,
            "cycle_tag": cycle_tag,
            # Run's discount → the recorder's δ uses the real γ; live td-residual tail matches
            # the prober's offline _td at the same γ.
            "gamma": float(self.model.gamma),
        }
        procs = spawn_eval_workers(run_dir, base_cfg, n_workers)

        self._pending = {
            "step": step, "bot_names": bot_names, "sentinels": sentinels,
            "fixed_labels": fixed_labels,
            "sentinel_entries": sentinel_entries, "procs": procs,
            "snapshot": snapshot_zip, "run_dir": run_dir, "n_games": n_games,
            "launched_at": time.monotonic(),
        }
        print(f"[SELFPLAY EVAL] step {step:,}: spawned {n_workers} work-stealing worker(s) on "
              f"{self._eval_device} ({len(bot_names)} bots + {len(sentinels)} sentinels, "
              f"conc {self._eval_concurrency}) — non-blocking")
        send_event(f"🧪 Self-play eval @ {step:,}: started "
                   f"({len(bot_names)} bots + {len(sentinels)} sentinels, {n_workers} worker(s))")

    def _abort_pending_cycle(self) -> None:
        """A cycle overran `_EVAL_CYCLE_TIMEOUT_SEC` → presumed hung (e.g. a Showdown battle
        that never completes). Kill its workers and collect whatever results landed, clearing
        `_pending` so eval resumes — a hung self-play eval otherwise pins `_pending` forever
        and silently skips every later boundary."""
        pending = self._pending
        elapsed = time.monotonic() - pending["launched_at"]
        print(f"⚠️ [SELFPLAY EVAL] step {pending['step']:,}: eval cycle hung "
              f"({elapsed:.0f}s > {_EVAL_CYCLE_TIMEOUT_SEC:.0f}s) — killing workers, "
              f"collecting partial results")
        send_event(f"⚠️ Self-play eval @ {pending['step']:,}: hung — killed after "
                   f"{elapsed:.0f}s, partial")
        kill_eval_workers(pending["procs"])
        self._collect_pending()

    # ── Collect ────────────────────────────────────────────────────────────────

    def _collect_pending(self) -> None:
        pending = self._pending
        self._pending = None
        step = pending["step"]
        run_dir = pending["run_dir"]
        bot_names = pending["bot_names"]
        sentinels = pending["sentinels"]
        sentinel_entries = pending["sentinel_entries"]
        fixed_labels = pending.get("fixed_labels", [])

        for w in pending["procs"]:
            w["log"].close()
        bad_exits = [w for w in pending["procs"] if w["proc"].returncode not in (0, None)]

        all_names = bot_names + [s["label"] for s in sentinels] + fixed_labels
        merged, missing = merge_eval_results(run_dir, all_names)

        if missing:
            print(f"⚠️ [SELFPLAY EVAL] step {step:,}: missing results for {missing} "
                  f"(worker crash mid-item?) — see {run_dir}/worker_*.log")
        for w in bad_exits:
            print(f"⚠️ [SELFPLAY EVAL] worker exited {w['proc'].returncode}; see {w['log_path']}")

        # Bot-only sub-dicts (sentinels are aggregated separately into win_rate_vs_pool).
        wr = merged["win_rates"]
        bot_wr = {n: wr[n] for n in bot_names if n in wr}
        bot_rew = {n: merged["reward_means"][n] for n in bot_names if n in merged["reward_means"]}
        bot_ep = {n: merged["ep_lens"][n] for n in bot_names if n in merged["ep_lens"]}
        _td_tails = merged.get("td_resid_tails", {})
        bot_td = {n: _td_tails[n] for n in bot_names if n in _td_tails}

        if not wr:
            print(f"⚠️ [SELFPLAY EVAL] step {step:,}: no results (all workers failed); skipping record")
            send_event(f"⚠️ Self-play eval @ {step:,}: failed (no results)")
            self._cleanup(pending, keep_logs=True)
            return

        # Sentinel results in pool order (skip any whose worker died): (entry, label, win, reward, ep_len).
        kept_sentinels: list[tuple] = []
        kept_sentinel_tds: list = []  # parallel to kept_sentinels; None where no captured battles
        for s, entry in zip(sentinels, sentinel_entries):
            label = s["label"]
            v = wr.get(label)
            if v is not None:
                kept_sentinels.append((
                    entry, label, v,
                    merged["reward_means"].get(label, 0.0),
                    merged["ep_lens"].get(label, 0.0),
                ))
                kept_sentinel_tds.append(_td_tails.get(label))
        sentinel_win_rates = [v for (_e, _l, v, _rw, _ep) in kept_sentinels]

        tui: dict[str, float] = {}

        # ── Bot metrics (win rate + reward + ep_len, pushed LIVE like the bot-eval path —
        # not just win rate, else the TUI reward column shows the prior eval seeded at resume) ──
        record_per_opponent(self.logger, tui, bot_names, bot_wr, bot_rew, bot_ep)
        for name in bot_names:
            if name in bot_td:
                self.logger.record(f"eval/td_resid_tail_vs_{name}", bot_td[name])
                tui[f"eval/td_resid_tail_vs_{name}"] = bot_td[name]
        # TD-residual tail headline over the bots (#4) — lower = critic more often blindsided.
        td_tail_mean = sum(bot_td.values()) / len(bot_td) if bot_td else None
        if td_tail_mean is not None:
            self.logger.record("eval/td_resid_tail_mean", td_tail_mean)
            tui["eval/td_resid_tail_mean"] = td_tail_mean

        self.win_rate_vs_bots = bot_mean(bot_wr)
        mean_reward_vs_bots = bot_mean(bot_rew)
        mean_ep_len_vs_bots = bot_mean(bot_ep)
        self._pool.persist_win_rate(self.win_rate_vs_bots)
        self._check_bot_regression(bot_wr)

        # ── Pool / sentinel metrics (win rate + reward + ep_len; step → TUI row label) ──
        for i, (entry, _label, v, rw, ep) in enumerate(kept_sentinels):
            self.logger.record(f"eval/win_rate_vs_sentinel_{i}", v)
            self.logger.record(f"eval/mean_reward_vs_sentinel_{i}", rw)
            self.logger.record(f"eval/mean_ep_len_vs_sentinel_{i}", ep)
            tui[f"eval/win_rate_vs_sentinel_{i}"] = v
            tui[f"eval/mean_reward_vs_sentinel_{i}"] = rw
            tui[f"eval/mean_ep_len_vs_sentinel_{i}"] = ep
            # sentinel_<i> is positional (newest→oldest) so its KEY stays stable for a
            # continuous TB curve, but it maps to a DIFFERENT checkpoint each cycle — surface
            # its step so the TUI can label the row "vs sentinel_0 (47.0M)".
            tui[f"eval/sentinel_step_{i}"] = float(entry.step)
            td_i = kept_sentinel_tds[i]
            if td_i is not None:
                self.logger.record(f"eval/td_resid_tail_vs_sentinel_{i}", td_i)
                tui[f"eval/td_resid_tail_vs_sentinel_{i}"] = td_i

        sentinel_rewards = [rw for (_e, _l, _v, rw, _ep) in kept_sentinels]
        sentinel_ep_lens = [ep for (_e, _l, _v, _rw, ep) in kept_sentinels]
        sentinel_tds = [t for t in kept_sentinel_tds if t is not None]
        win_rate_vs_pool = (
            sum(sentinel_win_rates) / len(sentinel_win_rates) if sentinel_win_rates else 0.0
        )
        mean_reward_vs_pool = (
            sum(sentinel_rewards) / len(sentinel_rewards) if sentinel_rewards else 0.0
        )
        mean_ep_len_vs_pool = (
            sum(sentinel_ep_lens) / len(sentinel_ep_lens) if sentinel_ep_lens else 0.0
        )
        monotonicity = _monotonicity_score(sentinel_win_rates) if len(sentinel_win_rates) >= 2 else 1.0
        td_resid_tail_vs_pool = sum(sentinel_tds) / len(sentinel_tds) if sentinel_tds else None
        self.logger.record("eval/win_rate_vs_pool", win_rate_vs_pool)
        self.logger.record("eval/mean_reward_vs_pool", mean_reward_vs_pool)
        self.logger.record("eval/mean_ep_len_vs_pool", mean_ep_len_vs_pool)
        self.logger.record("eval/sentinel_monotonicity", monotonicity)
        tui["eval/win_rate_vs_pool"] = win_rate_vs_pool
        tui["eval/mean_reward_vs_pool"] = mean_reward_vs_pool
        tui["eval/mean_ep_len_vs_pool"] = mean_ep_len_vs_pool
        tui["eval/sentinel_monotonicity"] = monotonicity
        if td_resid_tail_vs_pool is not None:
            self.logger.record("eval/td_resid_tail_vs_pool", td_resid_tail_vs_pool)
            tui["eval/td_resid_tail_vs_pool"] = td_resid_tail_vs_pool

        if monotonicity < 0.6 and len(sentinel_win_rates) >= 3:
            emit(f"⚠️  [SELFPLAY] Cycling signal: sentinel_monotonicity={monotonicity:.2f} "
                 f"at step {step:,} — consider increasing max_snapshots")

        # ── Live curriculum: seed-if-crossing-threshold, then push the fraction to the envs ──
        # frac>0 ⇔ win rate ≥ SELF_PLAY_START. If we just crossed it with an empty pool, seed
        # NOW from the frozen (competent) eval snapshot — so the first self-play opponent is a
        # competent model, never random. Then push the live fraction + pool generation to every
        # training env so the heuristic-vs-pool ratio tracks performance mid-run (no restart).
        sf = 1.0 - heuristic_fraction(
            self.win_rate_vs_bots, floor=self._heuristic_floor,
            start=self._self_play_start_wr, full=self._self_play_full_wr)
        if sf > 0 and self._pool.is_empty():
            self._pool.add_from_path(pending["snapshot"], step)
            self._pool_generation += 1
            emit(f"🌱 [SELFPLAY] Win rate cleared the threshold — seeded the pool from the "
                 f"step {step:,} snapshot (self_play_fraction={sf:.0%})")
        # The live fraction is pushed to the envs at the END of collect (after any promotion),
        # so a same-cycle promotion's pool-generation bump reaches the workers immediately.

        # ── Derived metrics (no extra battles) ──
        pool_size = len(self._pool)
        self.logger.record("train/selfplay_fraction", sf)
        self.logger.record("eval/pool_snapshot_count", float(pool_size))
        tui["train/selfplay_fraction"] = sf
        tui["eval/pool_snapshot_count"] = float(pool_size)

        self.logger.record("eval/win_rate_vs_bots", self.win_rate_vs_bots)
        self.logger.record("eval/mean_reward_vs_bots", mean_reward_vs_bots)
        self.logger.record("eval/mean_ep_len_vs_bots", mean_ep_len_vs_bots)
        tui["eval/win_rate_vs_bots"] = self.win_rate_vs_bots
        tui["eval/mean_reward_vs_bots"] = mean_reward_vs_bots
        tui["eval/mean_ep_len_vs_bots"] = mean_ep_len_vs_bots

        # "all" aggregate — mean over ALL bots INCLUDING Random (matches
        # PerOpponentEvalCallback._record so the TUI "all" row populates identically).
        win_rate_mean = sum(bot_wr.values()) / len(bot_wr) if bot_wr else 0.0
        mean_reward_mean = sum(bot_rew.values()) / len(bot_rew) if bot_rew else 0.0
        self.logger.record("eval/win_rate_mean", win_rate_mean)
        self.logger.record("eval/mean_reward_mean", mean_reward_mean)
        tui["eval/win_rate_mean"] = win_rate_mean
        tui["eval/mean_reward_mean"] = mean_reward_mean

        # ── Stable cross-run opponents (ext_<label>) — a separate fixed yardstick, kept OUT of
        # win_rate_vs_bots / win_rate_vs_pool / the ELO fit / promotion (recorded for display) ──
        ext_wr = {lab: wr[lab] for lab in fixed_labels if lab in wr}
        record_per_opponent(self.logger, tui, fixed_labels, wr,
                            merged["reward_means"], merged["ep_lens"])
        win_rate_vs_external = external_aggregate(ext_wr)
        if win_rate_vs_external is not None:  # only the mini-league (2+) aggregate
            self.logger.record("eval/win_rate_vs_external", win_rate_vs_external)
            tui["eval/win_rate_vs_external"] = win_rate_vs_external

        total_dur = sum(merged["durations_sec"].values())
        self.logger.record("eval/duration_sec", total_dur)
        tui["eval/duration_sec"] = total_dur
        # Worker count so the TUI can show per-worker wall-clock (duration_sec is the SUM of
        # per-opponent durations; the pool runs them across n_workers subprocesses).
        tui["eval/n_workers"] = float(max(1, len(pending["procs"])))

        # Opponent default-rate telemetry — queried on THIS (training) thread; safe because
        # env_method is an IPC round-trip to the SubprocVecEnv we own. Self-play workers
        # report real numbers; heuristic workers report zeros.
        self._record_opponent_default_stats(tui)

        # Anchored-BT ELO over the accumulated results (appends this cycle's row first).
        # bot_wr carries every bot incl. random (the anchor); sentinels are the pool
        # snapshots the trainee just played, keyed by their training step.
        elo_result = record_elo(
            self._model_dir, step, bot_wr,
            [{"step": e.step, "win_rate": v} for e, _l, v, _rw, _ep in kept_sentinels],
            pending["n_games"], self.logger, tui, bot_td_tails=bot_td,
        )
        # ELO for each stable opponent (display-only, out of the fit) → fills the eval table's elo
        # column for the ext_ rows: its OWN recorded ELO when available, else a trainee-derived ballpark.
        if ext_wr:
            record_external_elos(self.logger, tui, elo_result[0] if elo_result else None, ext_wr,
                                 {e.label: e.source_elo for e in self._fixed_opponents})

        self.logger.dump(step)
        tui["_step"] = step
        send_metrics(tui)

        # Stable cross-run opponent(s) win-rate suffix for the eval headlines (avg over ext_, "" if none).
        _stable_suffix = (f"  Stable {sum(ext_wr.values()) / len(ext_wr) * 100:.1f}%"
                          if ext_wr else "")
        print(f"[SELFPLAY EVAL] step {step:,}: Bots {self.win_rate_vs_bots * 100:.1f}%  "
              f"Pool {win_rate_vs_pool * 100:.1f}%{_stable_suffix}  Monotonicity {monotonicity:.2f}  "
              f"SelfPlay {sf * 100:.0f}%  [{total_dur:.0f}s]")
        send_event(f"🧪 Self-play eval @ {step:,}: bots {self.win_rate_vs_bots * 100:.1f}%, "
                   f"pool {win_rate_vs_pool * 100:.1f}%"
                   + (f", stable {sum(ext_wr.values()) / len(ext_wr) * 100:.1f}%" if ext_wr else ""))

        # ── Persist eval metrics to metadata.json (bot block + pool block) ──
        if self._model_dir:
            block = build_bot_eval_block(bot_wr, bot_rew, bot_ep, bot_td)
            pool_block = {
                "win_rate": win_rate_vs_pool,
                "mean_reward": mean_reward_vs_pool,
                "mean_ep_len": mean_ep_len_vs_pool,
                "monotonicity": round(monotonicity, 3),
                "snapshot_count": pool_size,
                "sentinels": [
                    {
                        "step": entry.step,
                        "win_rate": round(v, 4),
                        "mean_reward": round(rw, 4),
                        "mean_ep_len": round(ep, 4),
                        "weight": round(self._pool.entry_weight(entry), 3),
                        "snapshot": entry.path.name,
                        **({"td_resid_tail": round(td, 4)} if td is not None else {}),
                    }
                    for (entry, _label, v, rw, ep), td in zip(kept_sentinels, kept_sentinel_tds)
                ],
            }
            if td_resid_tail_vs_pool is not None:
                pool_block["td_resid_tail"] = td_resid_tail_vs_pool
            block["pool"] = pool_block
            if ext_wr:
                block["externals"] = build_externals_block(
                    ext_wr, wr, merged["reward_means"], merged["ep_lens"])
                if win_rate_vs_external is not None:  # only the mini-league (2+) aggregate
                    block["win_rate_vs_external"] = win_rate_vs_external
            if elo_result:
                block["elo"], block["elo_ci"] = elo_result
            record_eval_results(self._model_dir, step, block)

        # ── Best model (copy the frozen snapshot) — based on bot win rate (excl. Random) ──
        self._maybe_save_best(step, pending, self.win_rate_vs_bots)

        # ── Promotion — promote the FROZEN snapshot (the live model has since advanced) ──
        if win_rate_vs_pool > self._promote_threshold:
            self._pool.add_from_path(pending["snapshot"], step)
            self._pool_generation += 1
            self.logger.record("train/selfplay_promoted_steps", float(step))

        # ── Push the live curriculum target to every training env (after any seed/promote so
        #    the pool-generation bump reaches the workers this cycle) + persist resume state ──
        self._push_self_play_target(sf)
        # A stable opponent the trainee has now mastered "becomes another bot" in every training env.
        self._push_stable_mastered(ext_wr)
        # Reconcile distilled opponents now that any seed/promote bumped the generation: distill
        # the new snapshot (steady-state) or, on first enable, the whole pool (backfill).
        self._reconcile_distill()
        if self._model_dir or self._pool:
            self._pool.persist_summary(
                win_rate_vs_bots=self.win_rate_vs_bots,
                self_play_fraction=sf,
                last_eval_step=step,
                seeded=not self._pool.is_empty(),
                pool_generation=self._pool_generation,
            )

        # Retain the bit-exact snapshot for the prober, groom traces, then drop the scratch.
        persist_eval_snapshot(self._model_dir, step, pending["snapshot"], self._keep_eval_snapshots)
        prune_eval_traces(self._model_dir, self._keep_eval_trace_steps)
        prune_run_artifacts(self._model_dir, self._keep_stalls, self._keep_crashes)  # bound stalls/ + crashes/
        self._cleanup(pending, keep_logs=bool(missing or bad_exits))

    def _push_self_play_target(self, fraction: float) -> None:
        """Push the live (fraction, pool_generation) to every training env via env_method, so
        the heuristic-vs-pool ratio tracks performance mid-run and workers re-scan the pool when
        it changes. Safe on the training thread (IPC to the SubprocVecEnv we own); a heuristic
        env (no set_self_play_target) or a transient failure is non-fatal."""
        try:
            self.training_env.env_method("set_self_play_target", float(fraction), self._pool_generation)
        except Exception as e:  # noqa: BLE001 — telemetry/curriculum push must never break eval
            print(f"[SELFPLAY] set_self_play_target push failed (non-fatal): {e}")

    def _push_stable_mastered(self, ext_wr: dict) -> None:
        """Mark any stable opponent whose win_rate has cleared ``--stable-opponent-mastered-wr`` for
        ``_MASTERY_CONFIRM_CYCLES`` consecutive cycles as MASTERED, and push the (monotonic,
        only-grows) set to every training env so a mastered opponent moves from the challenge bucket
        to the floor ("becomes another bot"). The N-cycle confirm guards against eval-noise flapping
        an irreversible flip; once confirmed it's one-way. Recomputed from eval each cycle →
        resume-safe (the streak just re-warms after a restart). No-op without stable opponents;
        non-fatal like the curriculum push."""
        if not self._fixed_opponents:
            return
        newly = []
        for lab, wr in ext_wr.items():
            if lab in self._stable_mastered:
                continue
            if wr >= self._stable_opponent_mastered_wr:
                self._stable_mastery_streak[lab] = self._stable_mastery_streak.get(lab, 0) + 1
                if self._stable_mastery_streak[lab] >= _MASTERY_CONFIRM_CYCLES:
                    self._stable_mastered.add(lab)
                    newly.append(lab)
            else:
                self._stable_mastery_streak[lab] = 0  # a below-threshold cycle resets the streak
        if newly:
            emit(f"🏇 [SELFPLAY] Mastered stable opponent(s) {sorted(newly)} (win_rate ≥ "
                 f"{self._stable_opponent_mastered_wr:.0%} for {_MASTERY_CONFIRM_CYCLES} cycles) — "
                 "now a coverage-floor opponent")
        try:
            self.training_env.env_method("set_stable_mastered", sorted(self._stable_mastered))
        except Exception as e:  # noqa: BLE001 — must never break eval
            print(f"[SELFPLAY] set_stable_mastered push failed (non-fatal): {e}")

    # ── Opponent distillation reconcile (idempotent; distill_integration.md §8) ──────────────
    def _reconcile_distill(self) -> None:
        """One reconcile tick: make the on-disk distilled set match the pool, then push the atomic
        all-or-nothing switch to the envs (only when it changed) and log the health metrics. No-op
        when nothing is missing. Never fatal — distillation is strictly additive."""
        if self._distill_mgr is None:
            return
        try:
            self._pool._scan()
            active = [e.step for e in self._pool._entries]
            r = self._distill_mgr.reconcile(active)

            # ── Events panel: per-job gate results (deployed / escalated / exhausted) ──
            for h in r.harvested:
                msg = _distill_job_event_text(h)
                if msg:
                    send_event(msg)

            key = (r.all_distilled, tuple(sorted(r.sampleable)))
            if key != self._last_distill_push:
                self._last_distill_push = key
                try:
                    self.training_env.env_method("set_distill_active", r.all_distilled, sorted(r.sampleable))
                except Exception as e:  # noqa: BLE001
                    print(f"[DISTILL] set_distill_active push failed (non-fatal): {e}")
                # Persist a small re-publish block (the per-snapshot record lives in the manifests;
                # this is the at-a-glance deployment state, merged into summary.json on change).
                if self._model_dir:
                    self._pool.persist_summary(distill={
                        "all_distilled": r.all_distilled, "frac": round(r.frac_distilled, 3),
                        "n_active": r.n_active, "n_ready": r.n_ready, "n_exhausted": r.n_exhausted,
                        "deployed_steps": sorted(r.sampleable) if r.all_distilled else []})

            # ── Events panel: the atomic all-or-nothing switch (the speedup is on iff all_distilled) ──
            if r.all_distilled != self._distill_deployed:
                if r.all_distilled:
                    send_event(f"🚀 Opponents now 100% distilled ({r.n_ready} snapshots) — "
                               f"rollout speedup ACTIVE")
                elif self._distill_deployed:  # True/None→False: only narrate a real revert, not the cold start
                    send_event(f"↩️ Opponents reverted to full models "
                               f"({r.frac_distilled * 100:.0f}% distilled) — backfilling")
                self._distill_deployed = r.all_distilled

            self.logger.record("distill/frac_active_opponents_distilled", r.frac_distilled)
            self.logger.record("distill/all_distilled", float(r.all_distilled))
            self.logger.record("distill/n_running", r.n_running)
            self.logger.record("distill/n_exhausted", r.n_exhausted)
            self.logger.record("distill/n_ready", r.n_ready)
            if r.spawned:
                steps = ", ".join(_distill_step_tag(s) for s in sorted(r.spawned))
                send_event(f"⚗ Distilling {len(r.spawned)} snapshot(s) [{steps}] "
                           f"({r.n_running} running) — opponents stay full until all pass")
                print(f"[DISTILL] reconcile: spawned {sorted(r.spawned)} "
                      f"(active={r.n_active} ready={r.n_ready} running={r.n_running} "
                      f"all_distilled={r.all_distilled})")
        except Exception as e:  # noqa: BLE001 — reconcile must never break training
            print(f"[DISTILL] reconcile failed (non-fatal): {e}")

    def _spawn_distill(self, step: int, config: dict):
        """Spawn the (async, GPU-ok) distill+gate subprocess for a snapshot. Returns (proc, step);
        None if the snapshot isn't in the pool. Output streams to the run dir for forensics."""
        import json as _json
        import subprocess
        entry = next((e for e in self._pool._entries if e.step == step), None)
        if entry is None:
            return None
        cmd = [sys.executable, "-m", "agents.training.distill.worker",
               "--snapshot", str(entry.path), "--step", str(step),
               "--distilled-dir", str(self._pool.distilled_dir),
               "--config", _json.dumps(config), "--device", self._distill_device]
        env = dict(os.environ)
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env["PYTHONPATH"] = src + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self._pool.distilled_dir.mkdir(parents=True, exist_ok=True)
        # log lives in distilled/ so it's cleaned with the snapshot's other artifacts on eviction
        log = open(self._pool.distilled_log(step), "w") if self._model_dir else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return (proc, step)

    def _poll_distill(self, handle):
        """None = still running; dict = finished. A crash with no manifest reads as failed, so the
        reconcile loop just re-triggers it next tick (idempotent + self-healing)."""
        import json as _json
        if handle is None:
            return {"passed": False, "speedup": 0.0}
        proc, step = handle
        if proc.poll() is None:
            return None
        mf = self._pool.distilled_manifest(step)
        if mf.exists():
            try:
                m = _json.loads(mf.read_text())
                # h2h/top1 pass through to the Events panel (manager ignores them for gating).
                return {"passed": bool(m.get("passed")), "speedup": float(m.get("speedup") or 0.0),
                        "h2h": m.get("h2h"), "top1": m.get("top1")}
            except (ValueError, OSError):
                pass
        return {"passed": False, "speedup": 0.0}

    def _record_opponent_default_stats(self, tui: dict) -> None:
        """Query the live training envs' self-play opponent default/redecide counters."""
        try:
            ostats = self.training_env.env_method("opponent_default_stats")
            dec = sum(s[0] for s in ostats)
            deft = sum(s[1] for s in ostats)
            redec = sum(s[2] for s in ostats)
            if dec > 0:
                self.logger.record("train/selfplay_opp_default_rate", deft / dec)
                self.logger.record("train/selfplay_opp_redecide_rate", redec / dec)
                self.logger.record("train/selfplay_opp_decisions", float(dec))
                tui["train/selfplay_opp_redecide_rate"] = redec / dec
        except Exception as e:  # noqa: BLE001 — telemetry must never break eval
            print(f"[SELFPLAY] opponent default-stat query failed (non-fatal): {e}")

    def _maybe_save_best(self, step: int, pending: dict, aggregate: float) -> None:
        if aggregate <= self._best_aggregate_win_rate:
            return
        self._best_aggregate_win_rate = aggregate
        if self.best_model_save_path is not None:
            # The frozen snapshot IS the evaluated model — copy it rather than re-saving
            # the (now-advanced) live model.
            dst = os.path.join(self.best_model_save_path, "best_model.zip")
            shutil.copy2(pending["snapshot"], dst)
            copy_run_config_to_best_model(self._model_dir, self.best_model_save_path)  # model_config.json
            write_best_model_sidecar(self._model_dir, dst, self.model)                 # best_model.json (+ELO)
            print(f"[SELFPLAY EVAL] new best ({aggregate * 100:.1f}%) saved to {dst}")

    def _cleanup(self, pending: dict, keep_logs: bool) -> None:
        # Drop the (large) transient run-dir snapshot; persist_eval_snapshot has already
        # copied it into eval_traces/ when retention is on. Keep the run dir only if a
        # worker failed, so its log survives for debugging.
        try:
            if os.path.exists(pending["snapshot"]):
                os.remove(pending["snapshot"])
        except OSError:
            pass
        if not keep_logs:
            shutil.rmtree(pending["run_dir"], ignore_errors=True)

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    def _on_training_end(self) -> None:
        self.drain()

    def drain(self, timeout: float | None = None) -> None:
        """Block (up to `timeout` TOTAL seconds) for the in-flight eval cycle, then record it.

        Called on graceful shutdown so an eval in flight is never orphaned. Idempotent
        (no-op if nothing pending). `timeout` is a total budget across all workers.
        """
        if self._pending is None:
            return
        budget = self._DRAIN_TIMEOUT_SEC if timeout is None else timeout
        deadline = time.monotonic() + budget
        print(f"[SELFPLAY EVAL] graceful shutdown — waiting up to {budget:.0f}s for eval worker(s)...")
        for w in self._pending["procs"]:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                w["proc"].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                w["proc"].kill()
        self._collect_pending()

    # ── Bot regression guard ─────────────────────────────────────────────────

    def _check_bot_regression(self, win_rates: dict[str, float]) -> None:
        for name in (k for k in win_rates if k != RANDOM_OPPONENT_NAME):
            wr = win_rates.get(name)
            if wr is None:
                continue
            peak = self._bot_peak.get(name, 0.0)
            if wr > peak:
                self._bot_peak[name] = wr
                peak = wr
            if peak >= _REGRESSION_WARN_THRESHOLD and wr < _REGRESSION_WARN_THRESHOLD:
                # Edge-trigger: only emit on the first eval where regression is detected.
                if name not in self._regression_active:
                    self._regression_active.add(name)
                    emit(
                        f"⚠️  [SELFPLAY] BOT_REGRESSION vs {name}: "
                        f"peak={peak * 100:.1f}% → now={wr * 100:.1f}% "
                        f"at step {self.num_timesteps:,}"
                    )
            elif wr >= _REGRESSION_WARN_THRESHOLD:
                self._regression_active.discard(name)  # recovered — re-arm
