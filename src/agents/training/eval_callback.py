import os
import asyncio
import threading
import traceback
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration

from agents.inference.player import RLPlayer
from agents.model.snapshot import record_eval_results
from agents.opponents import (
    Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer,
    Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player,
    Gen3HeuristicV2Player,
)
from agents.training.reward_tracker import RewardTrackingMixin
from agents.training.reward_manager import Gen3RewardManager
from agents.training.battle_recorder import BattleRecorder, write_battle_record
from main.launcher.ipc import send_metrics

BATTLE_FORMAT = "gen3ou"
# Single-player concurrency ceiling (kept for SelfPlayCallback's one-player path).
_EVAL_CONCURRENCY = 100
# PerOpponentEvalCallback evaluates every opponent concurrently — one RLPlayer per
# opponent, all gathered. Cap the AGGREGATE in-flight battles so N opponents don't
# flood the server with N×100; the per-opponent ceiling is this budget split N ways
# (floored at 16, which already saturates the single-threaded inference pipeline).
_EVAL_TOTAL_CONCURRENCY = 200


def _per_opponent_concurrency(n_opponents: int) -> int:
    """Per-player concurrency so the aggregate stays near _EVAL_TOTAL_CONCURRENCY."""
    if n_opponents <= 0:
        return _EVAL_CONCURRENCY
    return max(16, _EVAL_TOTAL_CONCURRENCY // n_opponents)


# Forensic-trace sample caps per opponent per eval cycle. Once both are filled the
# remaining battles run the cheap fast path and only feed the win-rate count.
_FORENSIC_LOSS_QUOTA = 10
_FORENSIC_WIN_QUOTA = 5

_OPPONENT_NAMES: dict[type, str] = {
    RandomPlayer: "Random",
    SimpleHeuristicsPlayer: "Heuristic",
    Gen3StallerPlayer: "Staller",
    Gen3AggressivePlayer: "Aggressive",
    Gen3SetupSweepPlayer: "SetupSweep",
    # V2 bots — names registered for TUI/TensorBoard; not yet in the eval rotation.
    Gen3HeuristicV2Player: "Heuristic2",
    Gen3StallerV2Player: "StallerV2",
    Gen3AggressiveV2Player: "AggressiveV2",
    Gen3SetupSweepV2Player: "SetupSweepV2",
}


def opponent_name(player_cls: type) -> str:
    """Return the display name for a player class (TensorBoard keys, TUI labels)."""
    return _OPPONENT_NAMES.get(player_cls, player_cls.__name__)


RANDOM_OPPONENT_NAME = opponent_name(RandomPlayer)

# Per-opponent game caps. Eval blocks training (the eval thread is joined in
# _on_step), so games-per-opponent is pure overhead. The narrow playstyle bots
# need only a coarse win-rate, so cap them at 100; the heuristic generalists
# (closest to real play, lower-variance signal worth more games) cap at 200.
# Both are clamped to the schedule's count so early tiers (100 games) are unaffected.
_EVAL_GAMES_CAP_DEFAULT = 100
_EVAL_GAMES_CAP_HEURISTIC = 200
_HEURISTIC_EVAL_NAMES = frozenset({"Heuristic", "Heuristic2"})


def eval_games_for(name: str, scheduled_games: int) -> int:
    """Capped per-opponent game count for the given opponent display name."""
    cap = _EVAL_GAMES_CAP_HEURISTIC if name in _HEURISTIC_EVAL_NAMES else _EVAL_GAMES_CAP_DEFAULT
    return min(scheduled_games, cap)


def bot_mean(d: dict[str, float]) -> float:
    """Average of values across non-Random opponents."""
    vals = [v for k, v in d.items() if k != RANDOM_OPPONENT_NAME]
    return sum(vals) / len(vals) if vals else 0.0


def build_bot_eval_block(
    win_rates: dict[str, float],
    reward_means: dict[str, float],
    ep_lens: dict[str, float],
) -> dict:
    """Build the standard bot-eval metrics dict for metadata.json (opponents last)."""
    return {
        "win_rate_mean": sum(win_rates.values()) / len(win_rates) if win_rates else 0.0,
        "win_rate_vs_bots": bot_mean(win_rates),
        "mean_reward_vs_bots": bot_mean(reward_means),
        "mean_ep_len_vs_bots": bot_mean(ep_lens),
        "opponents": {
            name: {
                "win_rate": win_rates[name],
                "mean_reward": reward_means[name],
                "mean_ep_len": ep_lens[name],
            }
            for name in win_rates
        },
    }


def eval_schedule(num_timesteps: int) -> tuple[int, int]:
    """Shared adaptive eval schedule: returns (freq_steps, n_games).

    0–20M:    every 1M steps, 100 games
    20–50M:   every 2M steps, 200 games
    50–100M:  every 3M steps, 300 games
    100M+:    every 4M steps, 300 games

    `n_games` is the per-tier ceiling; the actual per-opponent count is then
    clamped by `eval_games_for` (100 default / 200 heuristics). At 100M+ the
    win-rate curves move slowly, so the looser 4M cadence trades negligible
    resolution for ~25% less eval overhead.
    """
    if num_timesteps < 20_000_000:
        return 1_000_000, 100
    elif num_timesteps < 50_000_000:
        return 2_000_000, 200
    elif num_timesteps < 100_000_000:
        return 3_000_000, 300
    else:
        return 4_000_000, 300


class EvalRLPlayer(RewardTrackingMixin, RLPlayer):
    """RLPlayer with per-battle reward tracking for eval metrics.

    Optionally captures forensic traces (full per-decision model I/O → JSON +
    .npz, the same format the replay recorder writes) for a bounded sample of
    each cycle's battles: up to `win_quota` wins and `loss_quota` losses. The
    quota is the whole point of the cheap-vs-heavy split — only battles being
    captured pay the aux cost (`predict_values`, probs/`_last_prediction`); once
    both quotas are filled every remaining battle runs the fast path and just
    counts toward the win rate. Call `begin_forensic_cycle(dir, step)` before a
    cycle to (re)arm capture; leave the dir None to disable forensics entirely.
    """

    def __init__(self, *args, reward_fn_factory=Gen3RewardManager,
                 loss_quota=_FORENSIC_LOSS_QUOTA, win_quota=_FORENSIC_WIN_QUOTA, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reward_tracking(reward_fn_factory)
        self._loss_quota = loss_quota
        self._win_quota = win_quota
        self._forensic_dir: str | None = None
        self._forensic_step = 0
        self._wins_kept = 0
        self._losses_kept = 0
        self._trace_idx = 0
        self._recorders: dict[str, BattleRecorder] = {}

    def begin_forensic_cycle(self, forensic_dir: str | None, step: int) -> None:
        """Arm (or disable, if dir is None) forensic capture for one eval cycle."""
        self._forensic_dir = forensic_dir
        self._forensic_step = step
        self._wins_kept = 0
        self._losses_kept = 0
        self._trace_idx = 0
        self._recorders.clear()

    @property
    def _quota_open(self) -> bool:
        """Whether either outcome still needs forensic samples this cycle."""
        return self._forensic_dir is not None and (
            self._wins_kept < self._win_quota or self._losses_kept < self._loss_quota
        )

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "EVAL_STALL")
        if forfeit:
            return forfeit
        # Capture a battle in full only if we already started capturing it (so its
        # trace stays whole even if the quota fills mid-battle) or the quota is still
        # open when it begins. Everything else takes the fast path (need_aux=False).
        capturing = battle.battle_tag in self._recorders or self._quota_open
        idx, probs, mask = self._predict_best_action(
            battle, stochastic=False, need_aux=capturing
        )
        self._track_reward(battle, idx, mask)
        if capturing:
            rec = self._recorders.get(battle.battle_tag)
            if rec is None:
                rec = BattleRecorder(battle.battle_tag, self._reward_fn_factory)
                self._recorders[battle.battle_tag] = rec
            rec.record(battle, idx, probs, mask, state=getattr(self, "_last_prediction", None))
        return self.action_to_order(idx, battle)

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)  # reward finalize (mixin)
        rec = self._recorders.pop(battle.battle_tag, None)
        if rec is None:
            return
        # Persist this trace only if its outcome is one we still want a sample of;
        # otherwise drop the buffered capture (we already have enough of that result).
        if battle.won and self._wins_kept < self._win_quota:
            outcome = "win"
        elif battle.lost and self._losses_kept < self._loss_quota:
            outcome = "loss"
        else:
            return
        self._trace_idx += 1
        prefix = os.path.join(self._forensic_dir, f"{outcome}_{self._trace_idx:03d}")
        write_battle_record(prefix, rec, battle, self._forensic_step)
        if outcome == "win":
            self._wins_kept += 1
        else:
            self._losses_kept += 1


class PerOpponentEvalCallback(BaseCallback):
    """
    Evaluates the RL agent against a fixed list of named opponents at regular
    intervals during training, logging per-opponent win rates to TensorBoard
    and saving the best model when the aggregate improves.

    Adaptive schedule (hardcoded):
      0 – 20M steps:  every 1M steps,  100 games
      20M – 50M steps: every 2M steps, 200 games
      50M+ steps:     every 3M steps,  300 games

    Opponents are constructed outside and passed in so their WebSocket
    connections to the Showdown server persist across eval runs.
    The RLPlayer is created lazily in _init_callback() once self.model is set.
    """

    def __init__(
        self,
        opponents: list[tuple[str, object]],
        trainee_teambuilder,
        mappings,
        best_model_save_path: str | None = None,
        model_dir: str | None = None,
        server_config=LocalhostServerConfiguration,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._opponents = opponents
        self._trainee_teambuilder = trainee_teambuilder
        self._mappings = mappings
        self._server_config = server_config
        self.best_model_save_path = best_model_save_path
        self._model_dir = model_dir
        # One EvalRLPlayer per opponent, keyed by opponent display name, so all
        # opponents can be evaluated concurrently with isolated win/reward counters.
        self._rl_players: dict[str, RLPlayer] = {}
        self._last_eval_step = 0
        self._best_aggregate_win_rate = -1.0
        # Set by train_rl_agent after signal handlers are wired. Used as the
        # single canonical abort path so eval crashes save a proper checkpoint.
        self.abort_fn = None

    def _schedule(self) -> tuple[int, int]:
        return eval_schedule(self.num_timesteps)

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        ts = datetime.now().strftime('%H%M%S')
        per_conc = _per_opponent_concurrency(len(self._opponents))
        # One dedicated player per opponent, persistent across eval runs. They share
        # self.model (predicts serialize in the single eval-thread event loop) and the
        # stateless trainee teambuilder (random.choice), so concurrent use is safe.
        self._rl_players = {
            name: EvalRLPlayer(
                model=self.model,
                team=self._trainee_teambuilder,
                battle_format=BATTLE_FORMAT,
                server_configuration=self._server_config,
                mappings=self._mappings,
                account_configuration=AccountConfiguration(f"RLEv{i}{ts}", "password"),
                max_concurrent_battles=per_conc,
            )
            for i, (name, _opp) in enumerate(self._opponents)
        }

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

    def _run_async_eval(self, n_games: int) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def exception_handler(loop, context):
            msg = context.get("exception", context["message"])
            print(f"\n[EVAL FATAL] Background eval failed: {msg}")
            self._abort("eval fatal")

        loop.set_exception_handler(exception_handler)
        try:
            loop.run_until_complete(self._eval_all_opponents(n_games))
        except Exception as e:
            print(f"\n[EVAL CRASH] Step {self.num_timesteps}: {e}")
            traceback.print_exc()
            self._abort(f"eval crash at step {self.num_timesteps}")
        finally:
            loop.close()

    async def _eval_all_opponents(self, n_games: int) -> None:
        print(
            f"\n[EVAL] Step {self.num_timesteps:,}: "
            f"{len(self._opponents)} opponents × ≤{n_games} games "
            f"(cap {_EVAL_GAMES_CAP_DEFAULT}, heuristics {_EVAL_GAMES_CAP_HEURISTIC})..."
        )
        tui_metrics: dict[str, float] = {}
        eval_start = datetime.now()

        async def eval_one(name, opponent):
            """Run one opponent's full battle set on its dedicated player."""
            rl = self._rl_players[name]
            games = eval_games_for(name, n_games)
            # Players persist across eval runs — clear last run's battles/reward state.
            if rl.n_finished_battles > 0:
                rl.reset_battles()
            if opponent.n_finished_battles > 0:
                opponent.reset_battles()
            rl.reset_reward_tracking()
            # Arm forensic capture for this opponent this cycle (disabled if no model_dir).
            forensic_dir = (
                os.path.join(self._model_dir, "eval_traces", f"step_{self.num_timesteps}", name)
                if self._model_dir else None
            )
            rl.begin_forensic_cycle(forensic_dir, self.num_timesteps)

            start = datetime.now()
            await rl.battle_against(opponent, n_battles=games)
            duration = datetime.now() - start
            # _battle_finished_callback has fired for every battle by this point.

            won = rl.n_won_battles
            finished = rl.n_finished_battles
            win_rate = won / finished if finished > 0 else 0.0
            mean_reward = rl.mean_episode_reward
            ep_len = self._mean_episode_length(rl)
            print(
                f"  vs {name}: {win_rate * 100:.1f}%  "
                f"({won}/{finished})  ep_len={ep_len:.1f}  reward={mean_reward:.3f}  [{duration}]"
            )
            return name, win_rate, mean_reward, ep_len

        # All opponents play concurrently — each on its own player, so the per-opponent
        # counters stay isolated. Inter-opponent overlap removes the serial-loop bubbles.
        results = await asyncio.gather(
            *(eval_one(name, opponent) for name, opponent in self._opponents)
        )

        win_rates: dict[str, float] = {name: wr for name, wr, _, _ in results}
        reward_means: dict[str, float] = {name: mr for name, _, mr, _ in results}
        ep_lens: dict[str, float] = {name: el for name, _, _, el in results}
        for name in win_rates:
            self.logger.record(f"eval/win_rate_vs_{name}", win_rates[name])
            self.logger.record(f"eval/mean_ep_len_vs_{name}", ep_lens[name])
            self.logger.record(f"eval/mean_reward_vs_{name}", reward_means[name])
            tui_metrics[f"eval/win_rate_vs_{name}"] = win_rates[name]
            tui_metrics[f"eval/mean_ep_len_vs_{name}"] = ep_lens[name]
            tui_metrics[f"eval/mean_reward_vs_{name}"] = reward_means[name]

        eval_duration_sec = (datetime.now() - eval_start).total_seconds()
        aggregate = sum(win_rates.values()) / len(win_rates) if win_rates else 0.0
        aggregate_reward = sum(reward_means.values()) / len(reward_means) if reward_means else 0.0
        win_rate_vs_bots = bot_mean(win_rates)
        mean_reward_vs_bots = bot_mean(reward_means)
        mean_ep_len_vs_bots = bot_mean(ep_lens)
        self.logger.record("eval/win_rate_mean", aggregate)
        self.logger.record("eval/win_rate_vs_bots", win_rate_vs_bots)
        self.logger.record("eval/mean_reward_mean", aggregate_reward)
        self.logger.record("eval/mean_reward_vs_bots", mean_reward_vs_bots)
        self.logger.record("eval/mean_ep_len_vs_bots", mean_ep_len_vs_bots)
        self.logger.record("eval/duration_sec", eval_duration_sec)
        self.logger.dump(self.num_timesteps)

        # logger.dump() clears name_to_value before the next rollout, so eval metrics
        # never reach MetricsExporterCallback. Send them directly to the TUI pipe.
        tui_metrics["eval/win_rate_mean"] = aggregate
        tui_metrics["eval/win_rate_vs_bots"] = win_rate_vs_bots
        tui_metrics["eval/mean_reward_mean"] = aggregate_reward
        tui_metrics["eval/mean_reward_vs_bots"] = mean_reward_vs_bots
        tui_metrics["eval/mean_ep_len_vs_bots"] = mean_ep_len_vs_bots
        tui_metrics["eval/duration_sec"] = eval_duration_sec
        tui_metrics["_step"] = self.num_timesteps
        send_metrics(tui_metrics)

        print(
            f"[EVAL] Aggregate: {aggregate * 100:.1f}%  "
            f"(best so far: {self._best_aggregate_win_rate * 100:.1f}%)  "
            f"[took {eval_duration_sec:.0f}s]"
        )

        if self._model_dir:
            record_eval_results(
                self._model_dir,
                self.num_timesteps,
                build_bot_eval_block(win_rates, reward_means, ep_lens),
            )

        if aggregate > self._best_aggregate_win_rate:
            self._best_aggregate_win_rate = aggregate
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                print(f"[EVAL] New best ({aggregate * 100:.1f}%) saved.")

    def _abort(self, reason: str) -> None:
        """Delegate to the canonical abort path if wired; fall back to hard exit."""
        if self.abort_fn is not None:
            self.abort_fn(reason)
        else:
            os._exit(1)

    def _mean_episode_length(self, player) -> float:
        battles = [b for b in player._battles.values() if b.finished]
        if not battles:
            return 0.0
        return sum(b.turn for b in battles) / len(battles)
