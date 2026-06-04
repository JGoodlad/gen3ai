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
    eval_opponent_names,
    eval_run_nonce,
    kill_eval_workers,
    latest_recorded_eval_step,
    merge_eval_results,
    persist_eval_snapshot,
    prune_eval_traces,
    replay_last_eval_to_tui,
    spawn_eval_workers,
    write_eval_manifest,
)
from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
from main.launcher.ipc import emit, send_event, send_metrics

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
        n_workers: int = 3,
        eval_device: str = "cpu",
        eval_concurrency: int = _EVAL_SUBPROCESS_CONCURRENCY,
        keep_eval_snapshots: int = 10,
        keep_eval_trace_steps: int = 20,
        resume_eval_metadata: str | None = None,
        debug: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._pool = pool
        self._model_dir = model_dir
        self._server_config = server_config
        self._showdown_port = showdown_port
        # Bridge eval: workers play in-process via run_local_battles (no server connection).
        self._use_showdown_bridge = use_showdown_bridge
        self.best_model_save_path = best_model_save_path
        self._promote_threshold = promote_threshold
        self._self_play_temp = self_play_temp
        self._n_workers = max(1, n_workers)
        self._eval_device = eval_device
        self._eval_concurrency = eval_concurrency
        self._keep_eval_snapshots = max(0, keep_eval_snapshots)
        self._keep_eval_trace_steps = max(0, keep_eval_trace_steps)
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

        # Record exactly which model produced this cycle's traces (the prober reads this).
        write_eval_manifest(self._model_dir, step,
                            opponents=bot_names + sentinel_labels, n_games=n_games)
        # Process-unique account tag (per-process nonce + per-cycle counter), NOT the step:
        # the resume re-eval fires at the same step every restart, so a step tag collided
        # across restarts and hung a worker on a lingering challenge (wedging eval forever).
        self._eval_cycle += 1
        cycle_tag = f"{self._eval_run_nonce}{_b36(self._eval_cycle, 1)}"
        n_items = len(bot_names) + len(sentinels)
        n_workers = max(1, min(self._n_workers, n_items))
        base_cfg = {
            "snapshot": snapshot_zip,
            "port": self._showdown_port,
            "use_showdown_bridge": self._use_showdown_bridge,
            "model_dir": self._model_dir,
            "step": step,
            "n_games": n_games,
            "opponent_pool": bot_names,        # bots; workers steal from the combined pool
            "sentinels": sentinels,            # pool snapshots to play (stochastic)
            "self_play_temp": self._self_play_temp,
            "claim_dir": claim_dir,
            "result_dir": run_dir,             # writes result__<item>.json here
            "concurrency": self._eval_concurrency,
            "device": self._eval_device,
            "cycle_tag": cycle_tag,
        }
        procs = spawn_eval_workers(run_dir, base_cfg, n_workers)

        self._pending = {
            "step": step, "bot_names": bot_names, "sentinels": sentinels,
            "sentinel_entries": sentinel_entries, "procs": procs,
            "snapshot": snapshot_zip, "run_dir": run_dir,
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

        for w in pending["procs"]:
            w["log"].close()
        bad_exits = [w for w in pending["procs"] if w["proc"].returncode not in (0, None)]

        all_names = bot_names + [s["label"] for s in sentinels]
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

        if not wr:
            print(f"⚠️ [SELFPLAY EVAL] step {step:,}: no results (all workers failed); skipping record")
            send_event(f"⚠️ Self-play eval @ {step:,}: failed (no results)")
            self._cleanup(pending, keep_logs=True)
            return

        # Sentinel results in pool order (skip any whose worker died): (entry, label, win, reward, ep_len).
        kept_sentinels: list[tuple] = []
        for s, entry in zip(sentinels, sentinel_entries):
            label = s["label"]
            v = wr.get(label)
            if v is not None:
                kept_sentinels.append((
                    entry, label, v,
                    merged["reward_means"].get(label, 0.0),
                    merged["ep_lens"].get(label, 0.0),
                ))
        sentinel_win_rates = [v for (_e, _l, v, _rw, _ep) in kept_sentinels]

        tui: dict[str, float] = {}

        # ── Bot metrics (win rate + reward + ep_len, pushed LIVE like the bot-eval path —
        # not just win rate, else the TUI reward column shows the prior eval seeded at resume) ──
        for name in bot_names:
            if name in bot_wr:
                self.logger.record(f"eval/win_rate_vs_{name}", bot_wr[name])
                self.logger.record(f"eval/mean_reward_vs_{name}", bot_rew.get(name, 0.0))
                self.logger.record(f"eval/mean_ep_len_vs_{name}", bot_ep.get(name, 0.0))
                tui[f"eval/win_rate_vs_{name}"] = bot_wr[name]
                tui[f"eval/mean_reward_vs_{name}"] = bot_rew.get(name, 0.0)
                tui[f"eval/mean_ep_len_vs_{name}"] = bot_ep.get(name, 0.0)

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

        sentinel_rewards = [rw for (_e, _l, _v, rw, _ep) in kept_sentinels]
        sentinel_ep_lens = [ep for (_e, _l, _v, _rw, ep) in kept_sentinels]
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
        self.logger.record("eval/win_rate_vs_pool", win_rate_vs_pool)
        self.logger.record("eval/mean_reward_vs_pool", mean_reward_vs_pool)
        self.logger.record("eval/mean_ep_len_vs_pool", mean_ep_len_vs_pool)
        self.logger.record("eval/sentinel_monotonicity", monotonicity)
        tui["eval/win_rate_vs_pool"] = win_rate_vs_pool
        tui["eval/mean_reward_vs_pool"] = mean_reward_vs_pool
        tui["eval/mean_ep_len_vs_pool"] = mean_ep_len_vs_pool
        tui["eval/sentinel_monotonicity"] = monotonicity

        if monotonicity < 0.6 and len(sentinel_win_rates) >= 3:
            emit(f"⚠️  [SELFPLAY] Cycling signal: sentinel_monotonicity={monotonicity:.2f} "
                 f"at step {step:,} — consider increasing max_snapshots")

        # ── Live curriculum: seed-if-crossing-threshold, then push the fraction to the envs ──
        # frac>0 ⇔ win rate ≥ SELF_PLAY_START. If we just crossed it with an empty pool, seed
        # NOW from the frozen (competent) eval snapshot — so the first self-play opponent is a
        # competent model, never random. Then push the live fraction + pool generation to every
        # training env so the heuristic-vs-pool ratio tracks performance mid-run (no restart).
        sf = 1.0 - heuristic_fraction(self.win_rate_vs_bots)
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

        total_dur = sum(merged["durations_sec"].values())
        self.logger.record("eval/duration_sec", total_dur)
        tui["eval/duration_sec"] = total_dur

        # Opponent default-rate telemetry — queried on THIS (training) thread; safe because
        # env_method is an IPC round-trip to the SubprocVecEnv we own. Self-play workers
        # report real numbers; heuristic workers report zeros.
        self._record_opponent_default_stats(tui)

        self.logger.dump(step)
        tui["_step"] = step
        send_metrics(tui)

        print(f"[SELFPLAY EVAL] step {step:,}: Bots {self.win_rate_vs_bots * 100:.1f}%  "
              f"Pool {win_rate_vs_pool * 100:.1f}%  Monotonicity {monotonicity:.2f}  "
              f"SelfPlay {sf * 100:.0f}%  [{total_dur:.0f}s]")
        send_event(f"🧪 Self-play eval @ {step:,}: bots {self.win_rate_vs_bots * 100:.1f}%, "
                   f"pool {win_rate_vs_pool * 100:.1f}%")

        # ── Persist eval metrics to metadata.json (bot block + pool block) ──
        if self._model_dir:
            block = build_bot_eval_block(bot_wr, bot_rew, bot_ep)
            block["pool"] = {
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
                    }
                    for entry, _label, v, rw, ep in kept_sentinels
                ],
            }
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
