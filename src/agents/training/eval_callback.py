import os
import re
import sys
import time
import glob
import json
import shutil
import asyncio
import subprocess
from datetime import datetime, timezone

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
from utils.git import get_git_hash

BATTLE_FORMAT = "gen3ou"

EVAL_MANIFEST_NAME = "eval_manifest.json"
EVAL_SNAPSHOT_NAME = "snapshot.zip"


def _read_run_identity(model_dir: str) -> tuple:
    """(git_hash, arch_signature, config_version) for this run, from its on-disk
    model_config.json / metadata.json (written at run start), with a git fallback."""
    arch_signature = config_version = git_hash = None
    try:
        with open(os.path.join(model_dir, "model_config.json")) as f:
            cfg = json.load(f)
        arch_signature = cfg.get("arch_signature")
        config_version = cfg.get("config_version")
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(model_dir, "metadata.json")) as f:
            git_hash = json.load(f).get("git_hash")
    except (OSError, ValueError):
        pass
    if not git_hash:
        try:
            git_hash = get_git_hash()
        except Exception:  # noqa: BLE001 — identity is best-effort
            git_hash = None
    return git_hash, arch_signature, config_version


def write_eval_manifest(model_dir: str, step: int, *, opponents, n_games: int,
                        snapshot: "str | None" = None) -> dict:
    """Write ``<model_dir>/eval_traces/step_<N>/eval_manifest.json`` — the per-cycle
    record of *exactly which model* produced this cycle's forensic traces.

    `snapshot` is the relative filename of the persisted weight snapshot
    (``snapshot.zip``) when `--keep-eval-snapshots` retained it this cycle, else None;
    the prober uses it to reload the bit-exact model, falling back to the nearest
    persisted checkpoint when absent.
    """
    git_hash, arch_signature, config_version = _read_run_identity(model_dir)
    d = os.path.join(model_dir, "eval_traces", f"step_{step}")
    os.makedirs(d, exist_ok=True)
    manifest = {
        "step": step,
        "num_timesteps": step,
        "git_hash": git_hash,
        "arch_signature": arch_signature,
        "config_version": config_version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "opponents": list(opponents),
        "n_games": n_games,
    }
    with open(os.path.join(d, EVAL_MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
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

# In-flight watchdog: if a cycle's workers don't all finish within this wall-clock
# budget, the cycle is presumed HUNG (e.g. a Showdown battle that never completes —
# a worker blocked on a websocket await), so the parent kills the workers, collects
# whatever results landed, and clears `_pending`. Without this a single hung worker
# pins `_pending` forever and every later eval boundary is silently skipped — i.e. eval
# never recovers for the rest of the run. Set generously above a healthy CPU cycle
# (~6-10 min for the full roster incl. 300-game sentinels) so it never trips a slow-but-
# -live eval; only a true hang reaches it.
_EVAL_CYCLE_TIMEOUT_SEC = 1800.0

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
            stochastic=False,  # bot-eval measures the GREEDY policy
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


async def eval_one_matchup(trainee, opponent, n_games, model_dir, step, name) -> dict:
    """Run ONE trainee-vs-opponent matchup; return its metrics + write forensic traces.

    The shared per-matchup body behind both the bot-roster gather (`run_eval`) and the
    self-play eval worker's per-sentinel matchups. `n_games` is the already-resolved game
    count for THIS matchup — the caller applies `eval_games_for` for capped bot opponents,
    and passes the scheduled count straight through for sentinels.

    `trainee` is an `EvalRLPlayer` (reward tracking + forensic capture); `opponent` is any
    poke-env Player. Pure compute + disk (traces) — safe inside the eval worker process.
    Returns {name, win_rate, reward_mean, ep_len, duration_sec}.
    """
    if trainee.n_finished_battles > 0:
        trainee.reset_battles()
    if opponent.n_finished_battles > 0:
        opponent.reset_battles()
    trainee.reset_reward_tracking()
    forensic_dir = (
        os.path.join(model_dir, "eval_traces", f"step_{step}", name)
        if model_dir else None
    )
    trainee.begin_forensic_cycle(forensic_dir, step)

    start = datetime.now()
    await trainee.battle_against(opponent, n_battles=n_games)
    dur = (datetime.now() - start).total_seconds()

    won = trainee.n_won_battles
    finished = trainee.n_finished_battles
    win_rate = won / finished if finished > 0 else 0.0
    print(f"  vs {name}: {win_rate * 100:.1f}%  ({won}/{finished})  "
          f"ep_len={_mean_episode_length(trainee):.1f}  reward={trainee.mean_episode_reward:.3f}  [{dur:.0f}s]")
    return {
        "name": name,
        "win_rate": win_rate,
        "reward_mean": trainee.mean_episode_reward,
        "ep_len": _mean_episode_length(trainee),
        "duration_sec": dur,
    }


async def run_eval(players, opponents, n_games, model_dir, step) -> dict:
    """Run the per-opponent eval gather and return raw metrics (no logging sinks).

    `players` maps name -> EvalRLPlayer; `opponents` is a list of (name, player).
    Returns dicts keyed by opponent name: win_rates / reward_means / ep_lens /
    durations_sec. Forensic traces are written by the players as battles finish.
    Pure compute + disk (traces) — safe to call from the eval worker process.
    """
    async def eval_one(name, opponent):
        games = eval_games_for(name, n_games)
        return await eval_one_matchup(players[name], opponent, games, model_dir, step, name)

    results = await asyncio.gather(*(eval_one(n, o) for n, o in opponents))
    return {
        "win_rates": {m["name"]: m["win_rate"] for m in results},
        "reward_means": {m["name"]: m["reward_mean"] for m in results},
        "ep_lens": {m["name"]: m["ep_len"] for m in results},
        "durations_sec": {m["name"]: m["duration_sec"] for m in results},
    }


# ── Shared subprocess-eval mechanics (used by BOTH eval callbacks) ─────────────
# These keep the bot-eval and self-play-eval cycles spawning / merging / grooming
# identically, so the two non-blocking paths can't drift.

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_NONCE_COUNTER = 0


def _b36(n: int, width: int) -> str:
    """Fixed-width base-36 encoding of `abs(n)` (low digits; wraps at 36**width)."""
    n = abs(int(n))
    out = []
    for _ in range(width):
        out.append(_B36[n % 36])
        n //= 36
    return "".join(reversed(out))


def eval_run_nonce() -> str:
    """A short (3 base-36 char) per-PROCESS nonce for eval account names.

    Eval account names are ``<prefix><cycle_tag><wid><claim_seq>``; ``cycle_tag`` used to
    be ``step // 100 % 10000``, which is NOT unique across launcher restarts — the resume
    re-eval always fires at ~the same step, so every restart reused the same account names
    and collided with the previous (killed) process's lingering Showdown challenges
    (``There's already a challenge between you and ...``) → the battle never starts and the
    worker hangs forever. Mixing the pid + wall-clock (+ a process-global counter so
    successive calls differ) makes the tag unique per process while staying ≤4 chars, so
    the full account name comfortably fits Showdown's 18-char username cap.
    """
    global _NONCE_COUNTER
    _NONCE_COUNTER += 1
    return _b36(os.getpid() * 1_000_003 + int(time.time()) + _NONCE_COUNTER * 7919, 3)


def kill_eval_workers(procs: list[dict], wait_timeout: float = 5.0) -> None:
    """Kill any still-running eval workers and reap them (so none linger as zombies)."""
    for w in procs:
        if w["proc"].poll() is None:
            w["proc"].kill()
        try:
            w["proc"].wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            pass


def spawn_eval_workers(run_dir: str, base_cfg: dict, n_workers: int) -> list[dict]:
    """Write one config_<wid>.json per worker and Popen ``python -m main.eval_worker`` on it.

    ``base_cfg`` carries everything common to the workers (snapshot, port, opponent pool,
    claim/result dirs, concurrency, device, cycle_tag, and — for self-play — the sentinel
    specs); this only adds ``worker_id``. The launcher's metrics pipe FD is stripped from
    the child env (only the parent publishes to the TUI; that FD number is invalid in the
    child). Returns a list of ``{proc, log, log_path}``.
    """
    worker_env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
    procs = []
    for wid in range(n_workers):
        cfg = {**base_cfg, "worker_id": wid}
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
    return procs


def merge_eval_results(run_dir: str, names: list[str]) -> tuple[dict, list]:
    """Read ``result__<name>.json`` for each expected opponent; return ``(merged, missing)``.

    Work stealing writes one result file per opponent regardless of which worker ran it.
    A missing file means a worker died mid-opponent (its claim lock blocks a retry) — the
    caller logs it and carries on. Shared by both eval callbacks.
    """
    merged = {"win_rates": {}, "reward_means": {}, "ep_lens": {}, "durations_sec": {}}
    missing = []
    for name in names:
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
    return merged, missing


def prune_eval_traces(model_dir: str | None, keep_n: int) -> None:
    """Keep only the N most-recent eval step dirs under ``<model_dir>/eval_traces``.

    0 = keep all. Older dirs are removed whole; the current cycle's dir is the newest so
    it is never touched. ``python -m main.prober.groom`` is the manual fallback.
    """
    if keep_n <= 0 or not model_dir:
        return
    base = os.path.join(model_dir, "eval_traces")
    if not os.path.isdir(base):
        return
    dirs = []
    for name in os.listdir(base):
        m = re.match(r"step_(\d+)$", name)
        full = os.path.join(base, name)
        if m and os.path.isdir(full):
            dirs.append((int(m.group(1)), full))
    for _step, path in sorted(dirs, reverse=True)[keep_n:]:
        shutil.rmtree(path, ignore_errors=True)


def prune_eval_snapshots(model_dir: str | None, keep_n: int) -> None:
    """Keep only the N most-recent persisted eval snapshots (``eval_traces/step_*/snapshot.zip``)."""
    if keep_n <= 0 or not model_dir:
        return
    snaps = glob.glob(os.path.join(model_dir, "eval_traces", "step_*", EVAL_SNAPSHOT_NAME))

    def _stepof(p: str) -> int:
        m = re.search(r"step_(\d+)", p)
        return int(m.group(1)) if m else 0

    for p in sorted(snaps, key=_stepof, reverse=True)[keep_n:]:
        try:
            os.remove(p)
        except OSError:
            pass


def persist_eval_snapshot(model_dir: str | None, step: int, snapshot_path: str, keep_n: int) -> None:
    """Copy a cycle's weight snapshot into ``eval_traces/step_<N>/snapshot.zip`` (next to
    its traces), patch that step's manifest to point at it, then prune to the N most-recent.

    No-op when ``keep_n<=0`` (traces still carry the identity manifest). Lets the prober
    reload the bit-exact model that produced a cycle's traces. Shared by both eval callbacks.
    """
    if keep_n <= 0 or not model_dir:
        return
    dst_dir = os.path.join(model_dir, "eval_traces", f"step_{step}")
    os.makedirs(dst_dir, exist_ok=True)
    try:
        shutil.copy2(snapshot_path, os.path.join(dst_dir, EVAL_SNAPSHOT_NAME))
    except OSError as e:
        print(f"⚠️ [EVAL] could not persist snapshot for step {step:,}: {e}")
        return
    mpath = os.path.join(dst_dir, EVAL_MANIFEST_NAME)
    try:
        with open(mpath) as f:
            m = json.load(f)
        m["snapshot"] = EVAL_SNAPSHOT_NAME
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)
    except (OSError, ValueError):
        pass
    prune_eval_snapshots(model_dir, keep_n)


def replay_last_eval_to_tui(model_dir: str | None, resume_eval_metadata: str | None = None) -> None:
    """Re-publish the most recent persisted eval to the TUI on startup/resume.

    Reads ``latest_eval`` from this run's metadata.json (and the resumed checkpoint's, if
    given), and pushes the per-opponent + aggregate win-rate/reward/ep-len keys so the eval
    panel isn't blank until the next (possibly millions-of-steps-away) cycle. Shared by both
    eval callbacks; sentinel rows aren't re-published (the pool differs run to run).
    """
    block = None
    for path in (
        os.path.join(model_dir, "metadata.json") if model_dir else None,
        resume_eval_metadata,
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
        "mean_reward_mean": sum(reward_means.values()) / len(reward_means) if reward_means else 0.0,
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

    0–20M:    every 2M steps, 100 games
    20–100M:  every 3.5M steps, 300 games
    100M+:    every 5M steps, 300 games

    `n_games` is the per-tier ceiling; the actual per-opponent count is then
    clamped by `eval_games_for` (100 default / 200 heuristics), so for bots the
    300 ceiling reduces to ≤200 — it only sets the (unclamped) self-play sentinel
    game count. The cadence widens as training matures and the win-rate curves move
    more slowly, trading negligible resolution for less eval overhead.
    """
    if num_timesteps < 20_000_000:
        return 2_000_000, 100
    elif num_timesteps < 100_000_000:
        return 3_500_000, 300
    else:
        return 5_000_000, 300


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
        if idx is None:
            return self.choose_default_move()
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
        keep_eval_snapshots: int = 10,
        keep_eval_trace_steps: int = 20,
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
        # >0: persist the eval weight snapshot into eval_traces/step_<N>/snapshot.zip
        # (keeping only the N most-recent) so the prober can reload the bit-exact model
        # that produced a cycle's traces. 0 disables (traces still carry the manifest).
        self._keep_eval_snapshots = max(0, keep_eval_snapshots)
        # The trainer writes the forensic traces, so it grooms them: after each cycle
        # keep only the N most-recent eval step dirs (0 = keep all). Older dirs are
        # removed whole. `python -m main.prober.groom` is the manual fallback.
        self._keep_eval_trace_steps = max(0, keep_eval_trace_steps)
        # metadata.json of the checkpoint being resumed (if any) — read at startup so
        # the TUI shows the last eval immediately after a restart.
        self._resume_eval_metadata = resume_eval_metadata
        self._last_eval_step = 0
        self._best_aggregate_win_rate = -1.0
        # The in-flight eval cycle, or None. dict: step, names, procs[], snapshot, run_dir,
        # launched_at.
        self._pending: dict | None = None
        self._eval_root: str | None = None
        # Per-PROCESS account-name nonce (+ a per-cycle counter) so two launcher restarts
        # never reuse Showdown account names — the step-derived tag collided across restarts
        # and hung a worker on a lingering challenge.
        self._eval_run_nonce = eval_run_nonce()
        self._eval_cycle = 0
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
        # Record exactly which model produced this cycle's traces (the prober reads
        # this to reload the right model). snapshot=None now; _persist_snapshot patches
        # it to the retained filename on success when --keep-eval-snapshots is set.
        write_eval_manifest(self._model_dir, step, opponents=names, n_games=n_games)
        # Process-unique account tag (per-process nonce + per-cycle counter), NOT the step:
        # the resume re-eval fires at the same step every restart, so a step tag collided
        # across restarts and hung a worker on a lingering challenge.
        self._eval_cycle += 1
        cycle_tag = f"{self._eval_run_nonce}{_b36(self._eval_cycle, 1)}"
        # Never spawn more workers than opponents to steal.
        n_workers = max(1, min(self._n_workers, len(names)))
        base_cfg = {
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
            "cycle_tag": cycle_tag,
        }
        procs = spawn_eval_workers(run_dir, base_cfg, n_workers)

        self._pending = {"step": step, "names": names, "procs": procs,
                         "snapshot": snapshot_zip, "run_dir": run_dir,
                         "launched_at": time.monotonic()}
        print(f"[EVAL] step {step:,}: spawned {n_workers} work-stealing worker(s) on "
              f"{self._eval_device} ({len(names)} opponents, conc {self._eval_concurrency}) "
              f"— non-blocking")
        send_event(f"🧪 Eval @ {step:,}: started "
                   f"({len(names)} opponents, {n_workers} worker(s))")

    def _abort_pending_cycle(self) -> None:
        """A cycle overran `_EVAL_CYCLE_TIMEOUT_SEC` → presumed hung. Kill its workers,
        then collect whatever results landed (clears `_pending` so eval can resume)."""
        pending = self._pending
        elapsed = time.monotonic() - pending["launched_at"]
        print(f"⚠️ [EVAL] step {pending['step']:,}: eval cycle hung "
              f"({elapsed:.0f}s > {_EVAL_CYCLE_TIMEOUT_SEC:.0f}s) — killing workers, "
              f"collecting partial results")
        send_event(f"⚠️ Eval @ {pending['step']:,}: hung — killed after {elapsed:.0f}s, partial")
        kill_eval_workers(pending["procs"])
        self._collect_pending()

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
        merged, missing = merge_eval_results(run_dir, pending["names"])

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
        self._persist_snapshot(pending)
        self._prune_eval_traces()   # trainer grooms the traces it writes
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

        # _maybe_save_best() updates _best_aggregate_win_rate AFTER this, so the value
        # here is the prior best — surface a new best explicitly rather than printing a
        # stale "best" that the current rate already beats. (-1.0 = no eval yet.)
        prev_best = self._best_aggregate_win_rate
        pct = aggregate * 100
        if prev_best < 0.0:
            summary = f"{pct:.1f}% (first eval)"
        elif aggregate > prev_best:
            summary = f"{pct:.1f}% 🏆 new best (+{(aggregate - prev_best) * 100:.1f}pts)"
        else:
            summary = (f"{pct:.1f}% (best {prev_best * 100:.1f}%, "
                       f"-{(prev_best - aggregate) * 100:.1f}pts)")
        print(f"[EVAL] step {step:,}: aggregate {summary}")
        send_event(f"🧪 Eval @ {step:,}: {summary}")

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

    def _persist_snapshot(self, pending: dict) -> None:
        persist_eval_snapshot(self._model_dir, pending["step"], pending["snapshot"],
                              self._keep_eval_snapshots)

    def _prune_eval_snapshots(self) -> None:
        prune_eval_snapshots(self._model_dir, self._keep_eval_snapshots)

    def _prune_eval_traces(self) -> None:
        prune_eval_traces(self._model_dir, self._keep_eval_trace_steps)

    def _cleanup(self, pending: dict, keep_logs: bool) -> None:
        # Always drop the (large) transient run-dir snapshot; _persist_snapshot has
        # already copied it into eval_traces/ when retention is on. Keep the run dir
        # only if a worker failed, so its log survives for debugging.
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
        replay_last_eval_to_tui(self._model_dir, self._resume_eval_metadata)
