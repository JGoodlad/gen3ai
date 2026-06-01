"""Self-play callback: bot eval, pool eval, snapshot promotion.

Runs on the same adaptive schedule as PerOpponentEvalCallback. On each eval:
  1. Bot eval   — 100 games each vs. Heuristic/Staller/Aggressive/SetupSweep.
                  Feeds heuristic_fraction(); win rate persisted for next run.
  2. Pool eval  — 100 games each vs. up to 5 sentinel snapshots (evenly spaced,
                  newest to oldest). Computes win_rate_vs_pool and the
                  sentinel_monotonicity score for cycling detection.
  3. Promotion  — if win_rate_vs_pool > promote_threshold, the current checkpoint
                  is added to the pool as a permanent snapshot.
  4. Metrics    — all results logged to TensorBoard and the launcher TUI.

Emits launcher events on promotion and bot-regression warnings (⚠️).
"""

from __future__ import annotations

import asyncio
import os
import threading
import traceback
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration

from agents.inference.player import RLPlayer
from agents.model.snapshot import record_eval_results
from agents.training.eval_callback import EvalRLPlayer, _EVAL_CONCURRENCY, eval_schedule, eval_games_for, RANDOM_OPPONENT_NAME, bot_mean, build_bot_eval_block
from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_tracker import RewardTrackingMixin
from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
from main.launcher.ipc import emit, send_metrics

BATTLE_FORMAT = "gen3ou"

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
    """Combined bot + pool eval callback with snapshot promotion.

    Args:
        pool: The SnapshotPool to promote into and eval against.
        bot_opponents: Pre-constructed list of (name, Player) pairs for bot eval.
            Should include Heuristic, Staller, Aggressive, SetupSweep (not Random).
        trainee_teambuilder: Teambuilder for the RL player.
        opp_teambuilder: Teambuilder for pool opponent players.
        mappings: Gen3 data mappings (lazy-loaded if None).
        promote_threshold: win_rate_vs_pool threshold to trigger promotion (default 0.65).
        best_model_save_path: Optional directory to save best model on aggregate improvement.
    """

    def __init__(
        self,
        pool: SnapshotPool,
        bot_opponents: list[tuple[str, object]],
        trainee_teambuilder,
        opp_teambuilder,
        mappings,
        promote_threshold: float = 0.65,
        best_model_save_path: str | None = None,
        model_dir: str | None = None,
        server_config=LocalhostServerConfiguration,
        self_play_temp: float = 1.0,
        debug: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        # Sampling temperature for sentinel opponents — kept equal to the training
        # opponents' --self-play-temp so eval opponents behave EXACTLY as in training.
        self._self_play_temp = self_play_temp
        # In --debug we run a tiny, fast eval cadence so a short CPU smoke
        # actually exercises seed → pool eval → promotion (the real schedule's
        # 1M-step floor never fires in a 20k-step smoke). Production is unaffected.
        self._debug = debug
        self._pool = pool
        self._bot_opponents = bot_opponents
        self._trainee_teambuilder = trainee_teambuilder
        self._opp_teambuilder = opp_teambuilder
        self._mappings = mappings
        self._server_config = server_config
        self._promote_threshold = promote_threshold
        self.best_model_save_path = best_model_save_path
        self._model_dir = model_dir

        self._rl_player: EvalRLPlayer | None = None
        self._pool_opp_players: list[RLPlayer] = []
        self._last_eval_step = 0
        self._best_aggregate_win_rate = -1.0
        # Set by train_rl_agent after signal handlers are wired.
        self.abort_fn = None

        # Shared state: written by eval thread, read by env factory on next restart.
        self.win_rate_vs_bots: float = pool.load_persisted_win_rate()
        self._bot_peak: dict[str, float] = {}  # resets each run; TensorBoard has history
        self._regression_active: set[str] = set()  # bots currently in warned state

    # ── SB3 lifecycle ──────────────────────────────────────────────────────

    def _init_callback(self) -> None:
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        # Seed the pool if it's empty (fresh start or first --self-play run).
        # Idempotent on subsequent launcher restarts — seed() skips if step-0 exists.
        if self._pool.is_empty():
            self._pool.seed(self.model)

        ts = datetime.now().strftime("%H%M%S")

        self._rl_player = EvalRLPlayer(
            model=self.model,
            team=self._trainee_teambuilder,
            battle_format=BATTLE_FORMAT,
            server_configuration=self._server_config,
            mappings=self._mappings,
            account_configuration=AccountConfiguration(f"SPCbEval{ts}", "password"),
            max_concurrent_battles=_EVAL_CONCURRENCY,
            stochastic=False,  # eval measures the GREEDY policy → stable win-rate signal
        )

        # Pre-create 5 pool opponent player slots. Models are swapped at eval time.
        self._pool_opp_players = [
            RLPlayer(
                model=self.model,  # placeholder; swapped before each sentinel eval
                team=self._opp_teambuilder,
                battle_format=BATTLE_FORMAT,
                server_configuration=self._server_config,
                mappings=self._mappings,
                account_configuration=AccountConfiguration(f"PoolOpp{i}{ts}", "password"),
                max_concurrent_battles=_EVAL_CONCURRENCY,
                # Sentinels act EXACTLY as they do as TRAINING opponents (stochastic, same
                # temperature, stale-tolerant), so win_rate_vs_pool measures the policy
                # against the opponents it actually trains against — mirroring the heuristic
                # bots, which use one player in both training and eval.
                stochastic=True,
                temperature=self._self_play_temp,
                strict_decision=False,
            )
            for i in range(5)
        ]

    def _schedule(self) -> tuple[int, int]:
        if self._debug:
            return 4000, 3  # fast cadence for --debug --self-play smoke tests
        return eval_schedule(self.num_timesteps)

    def _on_step(self) -> bool:
        if self.num_timesteps == 0:
            return True
        freq, n_games = self._schedule()
        if (self.num_timesteps // freq) > (self._last_eval_step // freq):
            self._last_eval_step = self.num_timesteps
            thread = threading.Thread(
                target=self._run_async_eval, args=(n_games,), daemon=True
            )
            thread.start()
            thread.join()
        return True

    # ── Eval runner ────────────────────────────────────────────────────────

    def _run_async_eval(self, n_games: int) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def exception_handler(loop, context):
            msg = context.get("exception", context["message"])
            print(f"\n[SELFPLAY EVAL FATAL] {msg}")
            self._abort("selfplay eval fatal")

        loop.set_exception_handler(exception_handler)
        try:
            loop.run_until_complete(self._eval_all(n_games))
        except Exception as e:
            print(f"\n[SELFPLAY EVAL CRASH] Step {self.num_timesteps}: {e}")
            traceback.print_exc()
            self._abort(f"selfplay eval crash at step {self.num_timesteps}")
        finally:
            loop.close()

    async def _eval_all(self, n_games: int) -> None:
        step = self.num_timesteps
        pool_size = len(self._pool)
        print(
            f"\n[SELFPLAY EVAL] Step {step:,}: "
            f"{len(self._bot_opponents)} bots (≤{n_games} games, capped) + "
            f"{min(pool_size, 5)} pool sentinels × {n_games} games each..."
        )

        win_rates: dict[str, float] = {}
        reward_means: dict[str, float] = {}
        ep_lens: dict[str, float] = {}
        tui_metrics: dict[str, float] = {}
        eval_start = datetime.now()

        # ── 1. Bot eval ────────────────────────────────────────────────────
        for name, opponent in self._bot_opponents:
            games = eval_games_for(name, n_games)
            wr, mean_reward, ep_len = await self._battle(self._rl_player, opponent, games, name)
            win_rates[name] = wr
            reward_means[name] = mean_reward
            ep_lens[name] = ep_len
            self.logger.record(f"eval/win_rate_vs_{name}", wr)
            self.logger.record(f"eval/mean_reward_vs_{name}", mean_reward)
            self.logger.record(f"eval/mean_ep_len_vs_{name}", ep_len)
            tui_metrics[f"eval/win_rate_vs_{name}"] = wr

        # Compute bot aggregates (exclude Random)
        self.win_rate_vs_bots = bot_mean(win_rates)
        mean_reward_vs_bots = bot_mean(reward_means)
        mean_ep_len_vs_bots = bot_mean(ep_lens)
        self._pool.persist_win_rate(self.win_rate_vs_bots)
        self._check_bot_regression(win_rates)

        # ── 2. Pool / sentinel eval ────────────────────────────────────────
        sentinel_win_rates: list[float] = []
        sentinels = self._pool.sentinel_entries(n=5)

        for i, entry in enumerate(sentinels):
            pool_model = self._pool.load_model(entry)
            self._pool_opp_players[i].model = pool_model
            opp = self._pool_opp_players[i]
            label = f"sentinel_{i}"
            wr, _, _ = await self._battle(self._rl_player, opp, n_games, label)
            sentinel_win_rates.append(wr)
            self.logger.record(f"eval/win_rate_vs_{label}", wr)
            tui_metrics[f"eval/win_rate_vs_{label}"] = wr

        win_rate_vs_pool = (
            sum(sentinel_win_rates) / len(sentinel_win_rates)
            if sentinel_win_rates else 0.0
        )
        monotonicity = _monotonicity_score(sentinel_win_rates) if len(sentinel_win_rates) >= 2 else 1.0
        self.logger.record("eval/win_rate_vs_pool", win_rate_vs_pool)
        self.logger.record("eval/sentinel_monotonicity", monotonicity)
        tui_metrics["eval/win_rate_vs_pool"] = win_rate_vs_pool
        tui_metrics["eval/sentinel_monotonicity"] = monotonicity

        if monotonicity < 0.6 and len(sentinels) >= 3:
            emit(f"⚠️  [SELFPLAY] Cycling signal: sentinel_monotonicity={monotonicity:.2f} "
                 f"at step {step:,} — consider increasing max_snapshots")

        # ── 3. Derived metrics (no extra battles) ──────────────────────────
        sf = 1.0 - heuristic_fraction(self.win_rate_vs_bots)
        self.logger.record("train/selfplay_fraction", sf)
        self.logger.record("eval/pool_snapshot_count", pool_size)
        tui_metrics["train/selfplay_fraction"] = sf
        tui_metrics["eval/pool_snapshot_count"] = float(pool_size)

        # Overall aggregate (bots only — pool changes over time, bots are stable)
        bot_aggregate = self.win_rate_vs_bots
        self.logger.record("eval/win_rate_vs_bots", bot_aggregate)
        self.logger.record("eval/mean_reward_vs_bots", mean_reward_vs_bots)
        self.logger.record("eval/mean_ep_len_vs_bots", mean_ep_len_vs_bots)
        tui_metrics["eval/win_rate_vs_bots"] = bot_aggregate
        tui_metrics["eval/mean_reward_vs_bots"] = mean_reward_vs_bots
        tui_metrics["eval/mean_ep_len_vs_bots"] = mean_ep_len_vs_bots
        eval_duration_sec = (datetime.now() - eval_start).total_seconds()
        self.logger.record("eval/duration_sec", eval_duration_sec)

        # Opponent default-rate telemetry: how often self-play TRAINING opponents defer to
        # a default move — no-decision polls (benign) plus the async-race StaleDecision
        # subset (the rate that decides whether a yield-and-retry is worth building). Only
        # self-play workers have RLPlayer opponents; heuristic workers report zeros. Safe to
        # query here: the training loop is paused at thread.join() for the duration of eval.
        try:
            _ostats = self.training_env.env_method("opponent_default_stats")
            _dec = sum(s[0] for s in _ostats)
            _deft = sum(s[1] for s in _ostats)
            _stale = sum(s[2] for s in _ostats)
            if _dec > 0:
                self.logger.record("train/selfplay_opp_default_rate", _deft / _dec)
                self.logger.record("train/selfplay_opp_stale_default_rate", _stale / _dec)
                self.logger.record("train/selfplay_opp_decisions", float(_dec))
                tui_metrics["train/selfplay_opp_stale_default_rate"] = _stale / _dec
        except Exception as _e:
            print(f"[SELFPLAY] opponent default-stat query failed (non-fatal): {_e}")

        self.logger.dump(step)

        tui_metrics["eval/duration_sec"] = eval_duration_sec
        tui_metrics["_step"] = step
        send_metrics(tui_metrics)

        print(
            f"[SELFPLAY EVAL] Bots: {bot_aggregate * 100:.1f}%  "
            f"Pool: {win_rate_vs_pool * 100:.1f}%  "
            f"Monotonicity: {monotonicity:.2f}  "
            f"SelfPlay fraction: {sf * 100:.0f}%  "
            f"[took {eval_duration_sec:.0f}s]"
        )

        # ── 4. Persist eval metrics to metadata.json ───────────────────────
        if self._model_dir:
            block = build_bot_eval_block(win_rates, reward_means, ep_lens)
            block["pool"] = {
                "win_rate": win_rate_vs_pool,
                "monotonicity": round(monotonicity, 3),
                "sentinels": [
                    {
                        "step": entry.step,
                        "win_rate": round(sentinel_win_rates[i], 4),
                        "weight": round(self._pool.entry_weight(entry), 3),
                        "snapshot": entry.path.name,
                    }
                    for i, entry in enumerate(sentinels)
                ],
            }
            block["opponents"] = block.pop("opponents")
            record_eval_results(self._model_dir, step, block)

        # ── 5. Best model save ─────────────────────────────────────────────
        if bot_aggregate > self._best_aggregate_win_rate:
            self._best_aggregate_win_rate = bot_aggregate
            if self.best_model_save_path:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                print(f"[SELFPLAY EVAL] New best ({bot_aggregate * 100:.1f}%) saved.")

        # ── 5. Promotion decision ──────────────────────────────────────────
        if win_rate_vs_pool > self._promote_threshold:
            self._pool.add(self.model, step=step)
            self.logger.record("train/selfplay_promoted_steps", float(step))

    async def _battle(
        self,
        trainee: EvalRLPlayer,
        opponent,
        n_games: int,
        label: str,
    ) -> tuple[float, float, float]:
        """Run n_games, reset both players, return (win_rate, mean_reward, mean_ep_len)."""
        if trainee.n_finished_battles > 0:
            trainee.reset_battles()
        if opponent.n_finished_battles > 0:
            opponent.reset_battles()

        start = datetime.now()
        await trainee.battle_against(opponent, n_battles=n_games)
        duration = datetime.now() - start

        won = trainee.n_won_battles
        finished = trainee.n_finished_battles
        wr = won / finished if finished > 0 else 0.0
        mean_reward = trainee.mean_episode_reward if hasattr(trainee, "mean_episode_reward") else 0.0

        ep_len = self._mean_episode_length(trainee)
        print(
            f"  vs {label}: {wr * 100:.1f}%  ({won}/{finished})  "
            f"ep_len={ep_len:.1f}  reward={mean_reward:.3f}  [{duration}]"
        )
        if hasattr(trainee, "reset_reward_tracking"):
            trainee.reset_reward_tracking()
        return wr, mean_reward, ep_len

    def _abort(self, reason: str) -> None:
        """Delegate to the canonical abort path if wired; fall back to hard exit."""
        if self.abort_fn is not None:
            self.abort_fn(reason)
        else:
            os._exit(1)

    def _mean_episode_length(self, player) -> float:
        battles = [b for b in player._battles.values() if b.finished]
        return sum(b.turn for b in battles) / len(battles) if battles else 0.0

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
