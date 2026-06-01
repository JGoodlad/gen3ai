import os
import sys
import time
import json
import shutil
import asyncio
import subprocess
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
from main.launcher.ipc import send_metrics, send_event

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

# Per-opponent in-flight battles in the subprocess eval worker. Eval now runs in a
# separate CPU process (one Python thread does the forwards), so a handful of
# concurrent battles already saturates it — keep it low so the shared Showdown
# server isn't flooded while training is also using it.
_EVAL_SUBPROCESS_CONCURRENCY = 5

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

# The canonical eval roster: (display name, player class, account prefix, is_v2).
# Ordered; V2 bots only included when use_v2_bots is set. Single source of truth so
# the in-process selfplay path, the subprocess worker, and the orchestrator agree.
_EVAL_OPPONENT_SPECS: list[tuple[str, type, str, bool]] = [
    ("Random", RandomPlayer, "CbRand", False),
    ("Heuristic", SimpleHeuristicsPlayer, "CbHeur", False),
    ("Staller", Gen3StallerPlayer, "CbStall", False),
    ("Aggressive", Gen3AggressivePlayer, "CbAggr", False),
    ("SetupSweep", Gen3SetupSweepPlayer, "CbSetup", False),
    ("Heuristic2", Gen3HeuristicV2Player, "CbHeur2", True),
    ("StallerV2", Gen3StallerV2Player, "CbStallV2", True),
    ("AggressiveV2", Gen3AggressiveV2Player, "CbAggrV2", True),
    ("SetupSweepV2", Gen3SetupSweepV2Player, "CbSetupV2", True),
]


def eval_opponent_names(use_v2_bots: bool) -> list[str]:
    """Ordered display names of the eval roster for the given v2 setting."""
    return [n for (n, _c, _p, is_v2) in _EVAL_OPPONENT_SPECS if use_v2_bots or not is_v2]


def build_eval_opponents(server_config, teambuilder, names, tag=""):
    """Construct the opponent players for `names`.

    Each Player opens its own Showdown connection on construction, so build only
    the names this caller actually needs. `tag` is appended to every account name
    and MUST be unique per concurrently-live set — under work stealing a worker
    builds a fresh set per claimed opponent, so the tag carries (cycle, worker,
    claim) to avoid username collisions on the shared server.
    """
    by_name = {n: (cls, prefix) for (n, cls, prefix, _v2) in _EVAL_OPPONENT_SPECS}
    out = []
    for name in names:
        cls, prefix = by_name[name]
        out.append((name, cls(
            battle_format=BATTLE_FORMAT, team=teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"{prefix}{tag}", "password"),
            max_concurrent_battles=_EVAL_CONCURRENCY,  # opponent side high; RL side governs
        )))
    return out


def build_eval_players(model, names, teambuilder, mappings, server_config, concurrency, tag=""):
    """One EvalRLPlayer per opponent name, sharing the (frozen) model + teambuilder."""
    return {
        name: EvalRLPlayer(
            model=model, team=teambuilder, battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"RLEv{tag}{i}", "password"),
            max_concurrent_battles=concurrency,
        )
        for i, name in enumerate(names)
    }


def claim_next_opponent(claim_dir: str, names: list[str]) -> str | None:
    """Atomically claim the next unclaimed opponent from a shared pool (work stealing).

    Each opponent is claimed by exactly one worker via an O_EXCL lock file — the
    first worker to create `<claim_dir>/<name>.lock` owns that opponent; everyone
    else gets FileExistsError and moves on. Returns the claimed name, or None when
    every opponent in `names` is already claimed (this worker is done). Robust
    across independent processes with no shared memory.
    """
    for name in names:
        try:
            fd = os.open(os.path.join(claim_dir, f"{name}.lock"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return name
        except FileExistsError:
            continue
    return None


def read_latest_eval_block(path: str | None) -> dict | None:
    """Return the most recent eval block from a metadata.json, or None.

    `record_eval_results` stores it at the top level as `latest_eval`
    (build_bot_eval_block + step). Used to re-publish the last eval to the TUI
    after a restart. Falls back to the legacy per-checkpoint
    `snapshot_history[<ckpt>]["evals"]` layout for older metadata files.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    latest = meta.get("latest_eval")
    if isinstance(latest, dict):
        return latest
    # Legacy fallback: evals nested under each checkpoint.
    best = None
    for entry in (meta.get("snapshot_history") or {}).values():
        ev = entry.get("evals") if isinstance(entry, dict) else None
        if isinstance(ev, dict) and (best is None or ev.get("step", -1) > best.get("step", -1)):
            best = ev
    return best


def _mean_episode_length(player) -> float:
    battles = [b for b in player._battles.values() if b.finished]
    if not battles:
        return 0.0
    return sum(b.turn for b in battles) / len(battles)


async def run_eval(players, opponents, n_games, model_dir, step) -> dict:
    """Run the per-opponent eval gather and return raw metrics (no logging sinks).

    `players` maps name -> EvalRLPlayer; `opponents` is a list of (name, player).
    Returns dicts keyed by opponent name: win_rates / reward_means / ep_lens /
    durations_sec. Forensic traces are written by the players as battles finish.
    Pure compute + disk (traces) — safe to call from the eval worker process.
    """
    async def eval_one(name, opponent):
        rl = players[name]
        games = eval_games_for(name, n_games)
        if rl.n_finished_battles > 0:
            rl.reset_battles()
        if opponent.n_finished_battles > 0:
            opponent.reset_battles()
        rl.reset_reward_tracking()
        forensic_dir = (
            os.path.join(model_dir, "eval_traces", f"step_{step}", name)
            if model_dir else None
        )
        rl.begin_forensic_cycle(forensic_dir, step)

        start = datetime.now()
        await rl.battle_against(opponent, n_battles=games)
        dur = (datetime.now() - start).total_seconds()

        won = rl.n_won_battles
        finished = rl.n_finished_battles
        win_rate = won / finished if finished > 0 else 0.0
        print(f"  vs {name}: {win_rate * 100:.1f}%  ({won}/{finished})  "
              f"ep_len={_mean_episode_length(rl):.1f}  reward={rl.mean_episode_reward:.3f}  [{dur:.0f}s]")
        return name, win_rate, rl.mean_episode_reward, _mean_episode_length(rl), dur

    results = await asyncio.gather(*(eval_one(n, o) for n, o in opponents))
    return {
        "win_rates": {n: wr for n, wr, _, _, _ in results},
        "reward_means": {n: mr for n, _, mr, _, _ in results},
        "ep_lens": {n: el for n, _, _, el, _ in results},
        "durations_sec": {n: d for n, _, _, _, d in results},
    }

# Per-opponent game caps. Eval now runs in a non-blocking subprocess, but on CPU
# each game still costs wall-clock, so keep games-per-opponent bounded. The narrow
# playstyle bots need only a coarse win-rate, so cap them at 100; the heuristic
# generalists (closest to real play, lower-variance signal worth more games) cap at
# 200. Both are clamped to the schedule's count so early tiers (100 games) are unaffected.
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
    Evaluates the trained agent against the fixed bot roster on the adaptive
    schedule (see `eval_schedule`), but does so in a **separate subprocess** with
    a frozen weight snapshot, NON-BLOCKING:

    - On a trigger step, snapshot the live model to disk (`model.save`) and spawn
      `n_workers` eval-worker processes that **work-steal** opponents from a shared
      pool (atomic O_EXCL claim files) — a worker that finishes an opponent grabs
      the next unclaimed one, so uneven per-opponent cost self-balances. Training
      continues immediately; the workers run on their own CPU(s).
    - On later steps, poll; when all workers finish, merge their per-opponent result
      JSONs and record win-rate / reward / ep-len to TensorBoard + the TUI, append
      to metadata.json, and promote the snapshot to best_model if it won.
    - If a trigger fires while the previous cycle is still running, skip it
      (logged) — on CPU an eval can outlast its interval; cadence just goes sparser.
    - On graceful shutdown (training end OR a SIGTERM-driven restart) the callback
      DRAINS the in-flight cycle so its results land before exit, never orphaned.
    - On startup it re-publishes the most recent eval from metadata.json to the TUI,
      so a resumed run shows the last known eval instead of a blank panel.

    The frozen snapshot is what makes parallel eval correct: a worker can't read
    the trainer's mutating in-memory weights, so each cycle evaluates the model
    exactly as it was at the snapshot step. Running in a fresh process also returns
    all eval memory to the OS on exit, avoiding fragmentation in the trainer.
    """

    # Wait for in-flight workers to finish on a graceful shutdown (10 min) — long
    # enough to let a full CPU eval complete; the restart path's grace window
    # (--restart-grace-minutes, 20 min default) comfortably covers it.
    _DRAIN_TIMEOUT_SEC = 600

    def __init__(
        self,
        model_dir: str | None,
        server_config=LocalhostServerConfiguration,
        *,
        use_v2_bots: bool = False,
        best_model_save_path: str | None = None,
        n_workers: int = 3,
        eval_device: str = "cpu",
        eval_concurrency: int = _EVAL_SUBPROCESS_CONCURRENCY,
        showdown_port: int | None = None,
        resume_eval_metadata: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._model_dir = model_dir
        self._server_config = server_config
        self._use_v2_bots = use_v2_bots
        self.best_model_save_path = best_model_save_path
        self._n_workers = max(1, n_workers)
        self._eval_device = eval_device
        self._eval_concurrency = eval_concurrency
        self._showdown_port = showdown_port
        # metadata.json of the checkpoint being resumed (if any) — read at startup so
        # the TUI shows the last eval immediately after a restart.
        self._resume_eval_metadata = resume_eval_metadata
        self._last_eval_step = 0
        self._best_aggregate_win_rate = -1.0
        # The in-flight eval cycle, or None. dict: step, names, procs[], snapshot, run_dir.
        self._pending: dict | None = None
        self._eval_root: str | None = None
        # Set by train_rl_agent after signal handlers are wired (kept for parity with
        # the old fail-fast path; the subprocess design logs-and-continues instead).
        self.abort_fn = None

    def _schedule(self) -> tuple[int, int]:
        return eval_schedule(self.num_timesteps)

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self._model_dir is not None:
            self._eval_root = os.path.join(self._model_dir, ".eval_runs")
            os.makedirs(self._eval_root, exist_ok=True)
        # Resumed run: re-publish the last eval so the TUI panel isn't blank until
        # the next cycle (which can be millions of steps away).
        self._replay_last_eval_to_tui()

    def _on_step(self) -> bool:
        if self._pending is not None and self._all_done(self._pending):
            self._collect_pending()
        if self.num_timesteps == 0:
            return True
        freq, _ = self._schedule()
        if (self.num_timesteps // freq) > (self._last_eval_step // freq):
            self._last_eval_step = self.num_timesteps
            if self._pending is not None:
                print(f"[EVAL] step {self.num_timesteps:,}: previous eval "
                      f"(step {self._pending['step']:,}) still running — skipping this cycle")
            else:
                self._launch_eval()
        return True

    # ------------------------------------------------------------------ launch

    def _launch_eval(self) -> None:
        if self._eval_root is None:
            return  # no model_dir → nowhere to snapshot/collect; eval disabled
        _, n_games = self._schedule()
        step = self.num_timesteps
        run_dir = os.path.join(self._eval_root, f"step_{step}")
        claim_dir = os.path.join(run_dir, "claims")
        os.makedirs(claim_dir, exist_ok=True)

        snapshot_base = os.path.join(run_dir, "snapshot")
        self.model.save(snapshot_base)  # SB3 appends .zip
        snapshot_zip = snapshot_base + ".zip"

        names = eval_opponent_names(self._use_v2_bots)
        # Short per-cycle tag for unique account names (cycles are ≥1M steps apart).
        cycle_tag = f"{step // 100 % 10000:04d}"
        # Never spawn more workers than opponents to steal.
        n_workers = max(1, min(self._n_workers, len(names)))
        # Don't hand the worker the launcher's metrics pipe FD — only the parent
        # (this process) publishes to the TUI; the FD number is invalid in the child.
        worker_env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}

        procs = []
        for wid in range(n_workers):
            cfg = {
                "snapshot": snapshot_zip,
                "port": self._showdown_port,
                "model_dir": self._model_dir,
                "step": step,
                "n_games": n_games,
                "opponent_pool": names,        # full pool; workers steal from it
                "claim_dir": claim_dir,
                "result_dir": run_dir,         # writes result__<opponent>.json here
                "concurrency": self._eval_concurrency,
                "device": self._eval_device,
                "worker_id": wid,
                "cycle_tag": cycle_tag,
            }
            cfg_path = os.path.join(run_dir, f"config_{wid}.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            log_path = os.path.join(run_dir, f"worker_{wid}.log")
            logf = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, "-m", "main.eval_worker", cfg_path],
                stdout=logf, stderr=subprocess.STDOUT, env=worker_env,
            )
            procs.append({"proc": proc, "log": logf, "log_path": log_path})

        self._pending = {"step": step, "names": names, "procs": procs,
                         "snapshot": snapshot_zip, "run_dir": run_dir}
        print(f"[EVAL] step {step:,}: spawned {n_workers} work-stealing worker(s) on "
              f"{self._eval_device} ({len(names)} opponents, conc {self._eval_concurrency}) "
              f"— non-blocking")
        send_event(f"🧪 Eval @ {step:,}: started "
                   f"({len(names)} opponents, {n_workers} worker(s))")

    # ------------------------------------------------------------------ collect

    @staticmethod
    def _all_done(pending: dict) -> bool:
        return all(w["proc"].poll() is not None for w in pending["procs"])

    def _collect_pending(self) -> None:
        pending = self._pending
        self._pending = None
        step = pending["step"]
        run_dir = pending["run_dir"]

        for w in pending["procs"]:
            w["log"].close()
        bad_exits = [w for w in pending["procs"] if w["proc"].returncode not in (0, None)]

        # Work stealing writes one result__<opponent>.json per opponent, regardless
        # of which worker ran it. Read every expected opponent; missing = a worker
        # died mid-opponent (its claim lock blocks a retry) — log and carry on.
        merged = {"win_rates": {}, "reward_means": {}, "ep_lens": {}, "durations_sec": {}}
        missing = []
        for name in pending["names"]:
            rp = os.path.join(run_dir, f"result__{name}.json")
            if not os.path.exists(rp):
                missing.append(name)
                continue
            with open(rp) as f:
                r = json.load(f)
            merged["win_rates"][name] = r["win_rate"]
            merged["reward_means"][name] = r["reward_mean"]
            merged["ep_lens"][name] = r["ep_len"]
            merged["durations_sec"][name] = r["duration_sec"]

        if missing:
            print(f"⚠️ [EVAL] step {step:,}: missing results for {missing} "
                  f"(worker crash mid-opponent?) — see {run_dir}/worker_*.log")
        for w in bad_exits:
            print(f"⚠️ [EVAL] worker exited {w['proc'].returncode}; see {w['log_path']}")

        if not merged["win_rates"]:
            print(f"⚠️ [EVAL] step {step:,}: no results (all workers failed); skipping record")
            send_event(f"⚠️ Eval @ {step:,}: failed (no results)")
            self._cleanup(pending, keep_logs=True)
            return

        self._record(step, merged)
        self._maybe_save_best(step, pending, merged["win_rates"])
        self._cleanup(pending, keep_logs=bool(missing or bad_exits))

    def _record(self, step: int, merged: dict) -> None:
        win_rates = merged["win_rates"]
        reward_means = merged["reward_means"]
        ep_lens = merged["ep_lens"]
        aggregate = sum(win_rates.values()) / len(win_rates)
        aggregate_reward = sum(reward_means.values()) / len(reward_means) if reward_means else 0.0
        wr_bots = bot_mean(win_rates)
        rew_bots = bot_mean(reward_means)
        eplen_bots = bot_mean(ep_lens)
        total_dur = sum(merged["durations_sec"].values())

        tui: dict[str, float] = {}
        for name in win_rates:
            self.logger.record(f"eval/win_rate_vs_{name}", win_rates[name])
            self.logger.record(f"eval/mean_ep_len_vs_{name}", ep_lens.get(name, 0.0))
            self.logger.record(f"eval/mean_reward_vs_{name}", reward_means.get(name, 0.0))
            tui[f"eval/win_rate_vs_{name}"] = win_rates[name]
            tui[f"eval/mean_ep_len_vs_{name}"] = ep_lens.get(name, 0.0)
            tui[f"eval/mean_reward_vs_{name}"] = reward_means.get(name, 0.0)
        self.logger.record("eval/win_rate_mean", aggregate)
        self.logger.record("eval/win_rate_vs_bots", wr_bots)
        self.logger.record("eval/mean_reward_mean", aggregate_reward)
        self.logger.record("eval/mean_reward_vs_bots", rew_bots)
        self.logger.record("eval/mean_ep_len_vs_bots", eplen_bots)
        self.logger.record("eval/duration_sec", total_dur)
        # Record at the SNAPSHOT step so the eval curve aligns to when the model was
        # frozen, not the (later) step at which the worker happened to finish.
        self.logger.dump(step)

        tui.update({
            "eval/win_rate_mean": aggregate, "eval/win_rate_vs_bots": wr_bots,
            "eval/mean_reward_mean": aggregate_reward, "eval/mean_reward_vs_bots": rew_bots,
            "eval/mean_ep_len_vs_bots": eplen_bots, "eval/duration_sec": total_dur,
            "_step": step,
        })
        send_metrics(tui)

        print(f"[EVAL] step {step:,}: aggregate {aggregate * 100:.1f}% "
              f"(best {self._best_aggregate_win_rate * 100:.1f}%)")
        send_event(f"🧪 Eval @ {step:,}: {aggregate * 100:.1f}% "
                   f"(best {self._best_aggregate_win_rate * 100:.1f}%)")

        if self._model_dir:
            record_eval_results(self._model_dir, step,
                                build_bot_eval_block(win_rates, reward_means, ep_lens))

    def _maybe_save_best(self, step: int, pending: dict, win_rates: dict) -> None:
        aggregate = sum(win_rates.values()) / len(win_rates)
        if aggregate <= self._best_aggregate_win_rate:
            return
        self._best_aggregate_win_rate = aggregate
        if self.best_model_save_path is not None:
            # The frozen snapshot IS the best model — copy it rather than re-saving.
            dst = os.path.join(self.best_model_save_path, "best_model.zip")
            shutil.copy2(pending["snapshot"], dst)
            print(f"[EVAL] new best ({aggregate * 100:.1f}%) saved to {dst}")

    def _cleanup(self, pending: dict, keep_logs: bool) -> None:
        # Always drop the (large) weight snapshot; keep the run dir only if a worker
        # failed so its log survives for debugging.
        try:
            if os.path.exists(pending["snapshot"]):
                os.remove(pending["snapshot"])
        except OSError:
            pass
        if not keep_logs:
            shutil.rmtree(pending["run_dir"], ignore_errors=True)

    def _on_training_end(self) -> None:
        self.drain()

    def drain(self, timeout: float | None = None) -> None:
        """Block (up to `timeout` TOTAL seconds) for the in-flight eval cycle, then record it.

        Called on graceful shutdown so an eval in flight is never orphaned and its
        results land in metadata.json + the TUI. Both the normal training end and the
        (self-initiated) scheduled restart wait the full `_DRAIN_TIMEOUT_SEC` (10 min)
        so a CPU eval can finish — the restart's grace window (--restart-grace-minutes,
        20 min) covers it, and the checkpoint is already saved first regardless.

        `timeout` is a total budget across all workers (not per-worker), so several
        hung workers can't stack past the deadline. Idempotent (no-op if nothing pending).
        """
        if self._pending is None:
            return
        budget = self._DRAIN_TIMEOUT_SEC if timeout is None else timeout
        deadline = time.monotonic() + budget
        print(f"[EVAL] graceful shutdown — waiting up to {budget:.0f}s for eval worker(s)...")
        for w in self._pending["procs"]:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                w["proc"].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                w["proc"].kill()
        self._collect_pending()

    # ------------------------------------------------------------- TUI resume

    def _replay_last_eval_to_tui(self) -> None:
        """Re-publish the most recent persisted eval to the TUI on startup/resume."""
        block = None
        for path in (
            os.path.join(self._model_dir, "metadata.json") if self._model_dir else None,
            self._resume_eval_metadata,
        ):
            block = read_latest_eval_block(path) or block
        if block is None:
            return
        opponents = block.get("opponents", {})
        if not opponents:
            return
        tui: dict[str, float] = {}
        for name, m in opponents.items():
            tui[f"eval/win_rate_vs_{name}"] = m.get("win_rate", 0.0)
            tui[f"eval/mean_ep_len_vs_{name}"] = m.get("mean_ep_len", 0.0)
            tui[f"eval/mean_reward_vs_{name}"] = m.get("mean_reward", 0.0)
        tui.update({
            "eval/win_rate_mean": block.get("win_rate_mean", 0.0),
            "eval/win_rate_vs_bots": block.get("win_rate_vs_bots", 0.0),
            "eval/mean_reward_mean": block.get("mean_reward_mean", block.get("mean_reward_vs_bots", 0.0)),
            "eval/mean_reward_vs_bots": block.get("mean_reward_vs_bots", 0.0),
            "eval/mean_ep_len_vs_bots": block.get("mean_ep_len_vs_bots", 0.0),
            "_step": block.get("step", 0),
        })
        send_metrics(tui)
        print(f"[EVAL] resumed — re-published last eval (step {block.get('step', '?')}, "
              f"{block.get('win_rate_mean', 0.0) * 100:.1f}% mean) to the TUI")
