import hashlib
import os
import re
import sys
import math
import time
import glob
import json
import shutil
import threading
import subprocess
from datetime import datetime, timezone

from stable_baselines3.common.callbacks import BaseCallback
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration

from agents.inference.player import RLPlayer
from agents.model.snapshot import record_eval_results, arch_toggles_from_model
from agents.training.fixed_opponent_pool import is_external
from agents.training.artifact_retention import (
    prune_run_artifacts, KEEP_STALLS_DEFAULT, KEEP_CRASHES_DEFAULT,
)
from agents.opponents import (
    Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer,
    Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player,
    Gen3HeuristicV2Player,
)
from agents.training.reward_tracker import RewardTrackingMixin
from agents.training.battle_recorder import BattleRecorder, write_battle_record
from utils.bridge.reconstruction import register_trace_prefix
from main.launcher.ipc import send_metrics, send_event
from utils.git import get_git_hash

BATTLE_FORMAT = "gen3ou"

EVAL_MANIFEST_NAME = "eval_manifest.json"
EVAL_SNAPSHOT_NAME = "snapshot.zip"


def trace_filename_stem(outcome: str, trace_tag: str, idx: int) -> str:
    """The forensic-trace filename stem ``<outcome>_<trace_tag><idx>`` (no suffix).

    THE single source of the naming contract the prober's `discovery._FNAME_RE`
    must invert. ``trace_tag`` is the work-stealing eval's per-shard namespace
    (``""`` un-sharded, or ``f"s{shard}_"`` from `eval_worker`). When sharding was
    added this stem grew an ``s<shard>_`` infix the prober's regex didn't match, so
    every sharded-eval trace parsed as outcome ``"?"`` and the WHOLE prober went
    blind — `eval_callback_test.test_trace_naming_contract` now pins that
    `discovery` parses exactly what this returns, so the two can't drift again."""
    return f"{outcome}_{trace_tag}{idx:03d}"


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
                        snapshot: "str | None" = None,
                        trainee_team_str: "str | None" = None,
                        opponent_pins: "dict | None" = None) -> dict:
    """Write ``<model_dir>/eval_traces/step_<N>/eval_manifest.json`` — the per-cycle
    record of *exactly which model* produced this cycle's forensic traces.

    `snapshot` is the relative filename of the persisted weight snapshot
    (``snapshot.zip``) when `--keep-eval-snapshots` retained it this cycle, else None;
    the prober uses it to reload the bit-exact model, falling back to the nearest
    persisted checkpoint when absent.

    The manifest also records the EVAL REGIME, so a trace dir is self-describing about how its
    numbers were measured (the OOD-eval era was invisible precisely because this was missing):
    ``matchup_hash`` (the run's declared-matchup tag, read from the run metadata),
    ``trainee_team_sha`` (the pin the trainee piloted — None = the default pool builder), and
    ``opponent_pins`` ({ext label: sha} for stable/exploiter opponents measured on their OWN
    pinned team — the fold-back contract).
    """
    git_hash, arch_signature, config_version = _read_run_identity(model_dir)
    from agents.model.snapshot import _read_matchup_hash
    _sha = lambda t: hashlib.sha1(t.encode()).hexdigest()[:10] if t else None
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
        "matchup_hash": _read_matchup_hash(model_dir),
        "trainee_team_sha": _sha(trainee_team_str),
        "opponent_pins": {k: _sha(v) for k, v in (opponent_pins or {}).items() if v},
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

# TD-residual tail metric (#4): the left tail of per-decision critic surprise
# δ = r + γ·V(s') − V(s) (BattleRecorder, the prober's formula) pooled over an eval cycle's
# CAPTURED battles. The headline scalar is the mean of the worst `TD_TAIL_FRAC` fraction (CVaR),
# which isolates the value-cliffs the loss analysis flags (a mean over all turns washes them out;
# a raw min is one-freak-turn noisy). Below `TD_TAIL_MIN_SAMPLES` residuals, report the single
# most-negative one. Sign-meaningful in reward units: more negative = critic more often blindsided,
# so a successful critic-coverage obs change should pull eval/td_resid_tail_mean UP toward 0.
#
# `td_tail` + its constants now live in `eval_sharding.results` (the aggregation layer that owns
# the pooled computation under battle-level work-stealing); re-exported here so the recorder, the
# worker, the prober-parity test, and this module's callers keep their existing import site.
from agents.training.eval_sharding import (  # noqa: E402,F401
    td_tail, TD_TAIL_FRAC, TD_TAIL_MIN_SAMPLES, ShardedEvalPool, PLAN_NAME,
    EvalItem, BOT, FIXED,
)

# Per-opponent in-flight battles in the subprocess eval worker. ONE game at a time.
# Eval inference is single-threaded (one Python thread does every forward in this
# process), so overlapping battles never parallelizes the bottleneck — it only piles
# on contention: extra Node sim procs on the bridge / extra load on the shared
# Showdown server, both fighting training's CPU-saturated env workers. Overlap
# measured slower, not faster. Cross-opponent parallelism still comes from the
# `--eval-workers` (3) subprocesses work-stealing the pool; each plays serially.
_EVAL_SUBPROCESS_CONCURRENCY = 1

# In-flight watchdog: if a cycle's workers don't all finish within this wall-clock
# budget, the cycle is presumed HUNG (e.g. a Showdown battle that never completes —
# a worker blocked on a websocket await), so the parent kills the workers, collects
# whatever results landed, and clears `_pending`. Without this a single hung worker
# pins `_pending` forever and every later eval boundary is silently skipped — i.e. eval
# never recovers for the rest of the run. Set generously above a healthy CPU cycle
# (the full roster — all bots + sentinels at EVAL_GAMES each — runs well under this) so it
# never trips a slow-but-live eval; only a true hang reaches it.
_EVAL_CYCLE_TIMEOUT_SEC = 1800.0

# Flat eval schedule — one cadence, one game count, applied uniformly to every bot
# AND every self-play sentinel. No maturity tiers, no per-opponent caps: eval runs
# non-blocking in a subprocess and skips a cycle whenever the previous one is still
# running, so a heavier roster self-throttles (cadence just goes sparser) instead of
# needing hand-tuned ceilings.
EVAL_FREQ_STEPS = 2_000_000
EVAL_GAMES = 100

# Battle-level work-stealing: each opponent's EVAL_GAMES games are split into shard units of
# (at most) this many games, so any idle worker can drain a straggler's remaining games instead of
# one worker grinding a whole opponent alone. Smaller → finer tail collapse but more player builds
# (and, on the websocket transport, more connection churn — the bridge is preferred for fine
# shards). `>= EVAL_GAMES` ⇒ one shard per opponent == the original opponent-level behaviour.
# Default 25 → 4 shards/opponent: a ~4x shorter tail at modest cost. Tunable via
# `--eval-shard-games`.
EVAL_SHARD_GAMES = 25

_OPPONENT_NAMES: dict[type, str] = {
    RandomPlayer: "random",
    SimpleHeuristicsPlayer: "heuristic",
    Gen3StallerPlayer: "staller",
    Gen3AggressivePlayer: "aggressive",
    Gen3SetupSweepPlayer: "setup_sweep",
    # V2 bots — in the full eval rotation (see _EVAL_OPPONENT_SPECS) and the training roster.
    Gen3HeuristicV2Player: "heuristic2",
    Gen3StallerV2Player: "staller_v2",
    Gen3AggressiveV2Player: "aggressive_v2",
    Gen3SetupSweepV2Player: "setup_sweep_v2",
}


def opponent_name(player_cls: type) -> str:
    """Return the display name for a player class (TensorBoard keys, TUI labels)."""
    return _OPPONENT_NAMES.get(player_cls, player_cls.__name__)


RANDOM_OPPONENT_NAME = opponent_name(RandomPlayer)

# The canonical eval roster: (display name, player class, account prefix). Every bot
# plays — both the v1 and v2 of each archetype, since they play differently and the
# extra diversity is the point. v1/v2 are paired for readable TUI/TensorBoard ordering.
# Single source of truth so the in-process selfplay path, the subprocess worker, and the
# orchestrator agree. Random leads (a cheap "is the model broken" floor; eval-only,
# excluded from `win_rate_vs_bots`).
_EVAL_OPPONENT_SPECS: list[tuple[str, type, str]] = [
    ("random", RandomPlayer, "CbRand"),
    ("heuristic", SimpleHeuristicsPlayer, "CbHeur"),
    ("heuristic2", Gen3HeuristicV2Player, "CbHeur2"),
    ("staller", Gen3StallerPlayer, "CbStall"),
    ("staller_v2", Gen3StallerV2Player, "CbStallV2"),
    ("aggressive", Gen3AggressivePlayer, "CbAggr"),
    ("aggressive_v2", Gen3AggressiveV2Player, "CbAggrV2"),
    ("setup_sweep", Gen3SetupSweepPlayer, "CbSetup"),
    ("setup_sweep_v2", Gen3SetupSweepV2Player, "CbSetupV2"),
]

# Full roster, in spec order. Random first (the broken-model floor).
_EVAL_ROSTER = [name for (name, _cls, _prefix) in _EVAL_OPPONENT_SPECS]


def eval_opponent_names() -> list[str]:
    """Ordered display names of the full eval roster (all bots + Random)."""
    return list(_EVAL_ROSTER)


def build_eval_opponents(server_config, teambuilder, names, tag="", *, start_listening=True):
    """Construct the opponent players for `names`.

    Each Player opens its own Showdown connection on construction, so build only
    the names this caller actually needs. `tag` is appended to every account name
    and MUST be unique per concurrently-live set — under work stealing a worker
    builds a fresh set per claimed opponent, so the tag carries (cycle, worker,
    claim) to avoid username collisions on the shared server.

    `start_listening=False` (bridge eval) opens no websocket — the in-process
    `run_local_battles` driver supplies the transport instead.
    """
    by_name = {n: (cls, prefix) for (n, cls, prefix) in _EVAL_OPPONENT_SPECS}
    out = []
    for name in names:
        cls, prefix = by_name[name]
        out.append((name, cls(
            battle_format=BATTLE_FORMAT, team=teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"{prefix}{tag}", "password"),
            max_concurrent_battles=_EVAL_CONCURRENCY,  # opponent side high; RL side governs
            start_listening=start_listening,
        )))
    return out


def build_eval_players(model, names, teambuilder, mappings, server_config, concurrency,
                       tag="", *, start_listening=True, gamma: float = 0.99, reward_fn_factory):
    """One EvalRLPlayer per opponent name, sharing the (frozen) model + teambuilder. ``reward_fn_factory``
    is REQUIRED — pass ``functools.partial(Gen3RewardManager, config=RewardConfig.from_dict(...))`` so
    eval measures the run's actual reward, not a default one."""
    return {
        name: EvalRLPlayer(
            model=model, team=teambuilder, battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"RLEv{tag}{i}", "password"),
            max_concurrent_battles=concurrency,
            stochastic=False,  # bot-eval measures the GREEDY policy
            start_listening=start_listening,
            gamma=gamma, reward_fn_factory=reward_fn_factory,
        )
        for i, name in enumerate(names)
    }


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


def episode_length_sum(player) -> float:
    """Σ of turn counts over this player's finished battles (the additive numerator a sharded eval
    pools — the parent recovers the mean as Σturns / Σn_finished). Denominator is the FINISHED-battle
    count (poke-env ``n_finished_battles``), distinct from the reward count (``n_reward_episodes``) —
    see ``ShardResult`` / ``aggregate``. Reads ``_battles`` directly (the convention this file + its
    test fakes use)."""
    return float(sum(b.turn for b in player._battles.values() if b.finished))


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
    """Pool the cycle's shard results into the per-opponent ``merged`` dict + the fully-missing names.

    Battle-level work-stealing writes one ``shard__<unit_id>.json`` per played shard (an opponent
    is split into chunks any idle worker can drain). This reads the cycle's ``plan.json`` — the
    launcher writes it before spawning workers, so it is the single source of truth for which
    shards each opponent expects — groups the shards by opponent and aggregates **exactly**
    (Σwon/Σfinished for win_rate; count-weighted reward/ep_len; raw δ pooled then ONE CVaR, since a
    CVaR can't be averaged). The returned ``merged`` is shape-compatible with every downstream
    consumer (record_per_opponent / build_bot_eval_block / record_elo / the pool & externals
    blocks) and additionally carries ``counts`` (exact W/L per opponent — the rating-fidelity
    record) and ``coverage``. ``missing`` = names with ZERO shards present (a whole opponent lost
    to a worker crash), logged by the caller exactly as a missing opponent always was. Shared by
    both eval callbacks. (Per-cycle ``run_dir`` is wiped at cleanup, so no shard/lock ever leaks
    across cycles.)
    """
    empty = {"win_rates": {}, "reward_means": {}, "ep_lens": {},
             "td_resid_tails": {}, "durations_sec": {}, "counts": {}, "coverage": {}}
    if not os.path.exists(os.path.join(run_dir, PLAN_NAME)):
        return empty, list(names)  # no plan written (shouldn't happen live) → nothing to merge
    merged_all, missing_all = ShardedEvalPool.from_plan(run_dir).collect(run_dir)
    wanted = set(names)
    merged = {block: {k: v for k, v in vals.items() if k in wanted}
              for block, vals in merged_all.items()}
    missing = [n for n in names if n in set(missing_all)]
    # Partial coverage (some-but-not-all shards of an opponent reported) → it was measured over
    # fewer games; surface it so a crashed shard doesn't silently degrade win-rate / ELO unnoticed.
    partial = {k: c for k, c in merged.get("coverage", {}).items() if c < 1.0}
    if partial:
        worst = ", ".join(f"{k} {c * 100:.0f}%" for k, c in sorted(partial.items(), key=lambda x: x[1]))
        print(f"⚠️ [EVAL] partial shard coverage (worker crash mid-opponent): {worst}")
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
    eval callbacks.

    The self-play ``pool`` sub-block (aggregate + per-sentinel rows) is re-published too,
    so the Pool/sentinel rows survive a restart exactly like the bot rows. This is safe even
    though sentinels are positional and the pool slides: the pool only changes at an
    eval-collect (seed/promote), which is also when the block is persisted — so the saved
    rows match the pool that's reconstructed from ``snapshots/`` at restart. (Pre-seed evals
    persist an empty ``sentinels`` list; those aren't re-published — nothing to show yet.)
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
    # Self-play pool block: re-publish the aggregate + per-sentinel rows (newest→oldest,
    # positional sentinel_<i>) using the saved step tags, mirroring the live collect path.
    pool = block.get("pool")
    if isinstance(pool, dict) and pool.get("sentinels"):
        tui["eval/win_rate_vs_pool"] = pool.get("win_rate", 0.0)
        tui["eval/mean_reward_vs_pool"] = pool.get("mean_reward", 0.0)
        tui["eval/mean_ep_len_vs_pool"] = pool.get("mean_ep_len", 0.0)
        tui["eval/sentinel_monotonicity"] = pool.get("monotonicity", 1.0)
        if "snapshot_count" in pool:
            tui["eval/pool_snapshot_count"] = float(pool["snapshot_count"])
        for i, s in enumerate(pool["sentinels"]):
            tui[f"eval/win_rate_vs_sentinel_{i}"] = s.get("win_rate", 0.0)
            tui[f"eval/mean_reward_vs_sentinel_{i}"] = s.get("mean_reward", 0.0)
            tui[f"eval/mean_ep_len_vs_sentinel_{i}"] = s.get("mean_ep_len", 0.0)
            tui[f"eval/sentinel_step_{i}"] = float(s.get("step", 0))

    # Skill rating (ELO) — re-publish so the 🏅 badge + per-opponent ELO show immediately on
    # resume instead of blanking until the next (possibly millions-of-steps-away) eval cycle.
    # The saved HEADLINE elo is authoritative — set it FIRST so a fit failure below can never
    # drop it. Then fit the block (best-effort) to (a) compute the headline if the block predates
    # the `elo` field, and (b) recover each opponent's ELO for the panel.
    if "elo" in block:
        tui["eval/elo"] = block["elo"]
        tui["eval/elo_ci"] = block.get("elo_ci", 0.0)
    try:
        from agents.training import elo as elo_mod
        efit = elo_mod.fit_from_block(block)
        if efit is not None:
            if "elo" not in block:
                tr = efit.rating_for_step(int(block.get("step", 0)))
                if tr is not None:
                    tui["eval/elo"] = tr[0]
                    tui["eval/elo_ci"] = elo_mod.ci95(tr[1])
            sentinels = pool.get("sentinels", []) if isinstance(pool, dict) else []
            _record_opponent_elos(efit, opponents, sentinels, tui)
    except Exception as e:  # noqa: BLE001 — telemetry; never break resume
        print(f"⚠️ [ELO] resume-republish compute failed: {e}")
    send_metrics(tui)
    pool_note = (f", pool {pool['win_rate'] * 100:.1f}%"
                 if isinstance(pool, dict) and pool.get("sentinels") else "")
    print(f"[EVAL] resumed — re-published last eval (step {block.get('step', '?')}, "
          f"{block.get('win_rate_mean', 0.0) * 100:.1f}% mean{pool_note}) to the TUI")


def latest_recorded_eval_step(model_dir: str | None, resume_eval_metadata: str | None = None) -> int:
    """The most recent eval step recorded in metadata (this run's + the resumed checkpoint's),
    or 0 if none.

    Used to restore ``_last_eval_step`` on startup so a RESUMED run doesn't immediately re-eval
    the same checkpoint: ``_last_eval_step`` is in-memory and resets to 0 each process, and the
    resumed step is far past a cadence boundary, so a fresh 0 would fire an eval on the first
    step. Restoring the genuine last-eval step makes the cadence purely step-based — restarts
    neither duplicate the eval nor starve it (frequent restarts still eval once the step crosses
    the next boundary). The panel isn't left blank: ``replay_last_eval_to_tui`` re-publishes the
    numbers regardless.
    """
    step = 0
    for path in (
        os.path.join(model_dir, "metadata.json") if model_dir else None,
        resume_eval_metadata,
    ):
        blk = read_latest_eval_block(path)
        if blk:
            step = max(step, int(blk.get("step", 0)))
    return step


def copy_run_config_to_best_model(model_dir: "str | None", best_model_dir: "str | None") -> None:
    """Copy the run-level ``model_config.json`` into ``best_model/`` whenever the best model is saved,
    so ``best_model/`` is a SELF-CONTAINED snapshot (weights + arch sidecar in one dir) — the unified
    place a stable-opponent consumer looks first for the arch gate. (The eval/ELO sidecar is the
    separate ``best_model.json`` written by ``write_best_model_sidecar``.) Best-effort."""
    if not model_dir or not best_model_dir:
        return
    src = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(src):
        return
    try:
        os.makedirs(best_model_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(best_model_dir, "model_config.json"))
    except OSError as e:
        print(f"⚠️ [EVAL] could not copy model_config.json into best_model/: {e}")


def write_best_model_sidecar(model_dir: "str | None", best_zip_path: str, model) -> None:
    """Write ``best_model/best_model.json`` — the per-checkpoint-style sidecar for the best snapshot,
    REUSING ``snapshot.write_checkpoint_metadata`` (which writes ``<zip − .zip>.json``). It carries
    the current ``latest_eval`` block — **including the run's ELO** — so ``best_model/`` is a
    self-contained, ELO-bearing snapshot a stable-opponent consumer can read directly (no parent
    search). Best-effort — never raise into the best-save path."""
    if not model_dir:
        return
    try:
        from agents.model.snapshot import write_checkpoint_metadata, _read_latest_eval
        lr = float(model.policy.optimizer.param_groups[0]["lr"])
        write_checkpoint_metadata(best_zip_path, lr=lr, n_epochs=int(model.n_epochs),
                                  git_hash=get_git_hash(), eval_block=_read_latest_eval(model_dir))
    except Exception as e:  # noqa: BLE001 — best-effort sidecar; must NEVER break the best-save path
        print(f"⚠️ [EVAL] could not write best_model.json sidecar: {e}")


def bot_mean(d: dict[str, float]) -> float:
    """Average of values across the scripted BOTS only — excludes Random (the broken-model
    floor) AND any stable cross-run opponents (``ext_...``, a separate yardstick kept out of
    ``win_rate_vs_bots`` so it never moves the self-play curriculum or the bot aggregate)."""
    vals = [v for k, v in d.items() if k != RANDOM_OPPONENT_NAME and not is_external(k)]
    return sum(vals) / len(vals) if vals else 0.0


def record_per_opponent(logger, tui: dict, names, win_rates: dict,
                        reward_means: dict, ep_lens: dict) -> None:
    """Record ``eval/{win_rate,mean_reward,mean_ep_len}_vs_<name>`` (TB logger + the TUI dict) for
    each ``name`` present in ``win_rates``. The shared per-opponent recorder for BOTH eval callbacks
    (bots + stable opponents); pool sentinels are positional (``sentinel_<i>``) and recorded apart."""
    for name in names:
        if name not in win_rates:
            continue
        for metric, value in (("win_rate", win_rates[name]),
                              ("mean_reward", reward_means.get(name, 0.0)),
                              ("mean_ep_len", ep_lens.get(name, 0.0))):
            key = f"eval/{metric}_vs_{name}"
            logger.record(key, value)
            tui[key] = value


def external_aggregate(ext_wr: dict) -> "float | None":
    """Mean win rate over stable cross-run opponents — only meaningful (and only emitted) for a
    mini-league (2+); with a single one it would just duplicate that opponent's own row."""
    return sum(ext_wr.values()) / len(ext_wr) if len(ext_wr) > 1 else None


def external_elo(trainee_elo: float, win_rate: float) -> float:
    """A BALLPARK ELO for a stable cross-run opponent: invert the Bradley-Terry win probability from
    the trainee's (bot-anchored) rating and the trainee's win rate vs it —
    ``R_opp = R_trainee − (400/ln10)·logit(win_rate)``. A single-edge estimate (rough), but on the
    SAME bot-anchored scale as the rest of the eval ladder, so it's a meaningful ballpark. The stable
    opponent is deliberately NOT a player in the BT fit itself (no ladder distortion) — this is a
    display-only derivation. ``win_rate`` is clamped to keep the logit finite (a 100-game eval can't
    resolve a rate past ~±0.05 of the bounds anyway), capping the gap at ≈±676 ELO."""
    p = min(max(win_rate, 0.02), 0.98)
    return trainee_elo - (400.0 / math.log(10.0)) * math.log(p / (1.0 - p))


def record_external_elos(logger, tui: dict, trainee_elo: "float | None", ext_wr: dict,
                         source_elos: "dict | None" = None) -> None:
    """Record ``eval/elo_vs_<label>`` (display-only) for each stable opponent, so the eval table's
    elo column is populated for the ``ext_`` rows (the TUI reads ``eval/elo_vs_<opp>``).

    Prefers the opponent's **own recorded ELO** (``source_elos[label]`` — read from its run's
    ``metadata.json:latest_eval.elo``; a well-fit, bot-anchored rating). Falls back to a single-edge
    **ballpark** inverted from the live trainee rating + win rate (``external_elo``) only when the
    opponent carries no recorded ELO. ``trainee_elo`` may be ``None`` (no fit yet) — a carried ELO is
    still shown; a fallback-only opponent is skipped that cycle."""
    source_elos = source_elos or {}
    for lab, wr in ext_wr.items():
        carried = source_elos.get(lab)
        if carried is not None:
            e = round(carried)
        elif trainee_elo is not None:
            e = round(external_elo(trainee_elo, wr))
        else:
            continue  # no recorded ELO and no trainee rating yet → nothing to show this cycle
        logger.record(f"eval/elo_vs_{lab}", e)
        tui[f"eval/elo_vs_{lab}"] = e


def build_externals_block(ext_labels, win_rates: dict, reward_means: dict, ep_lens: dict) -> dict:
    """The ``metadata.json:latest_eval`` ``externals`` sub-block for stable opponents (display-only)."""
    return {
        lab: {"win_rate": win_rates[lab],
              "mean_reward": reward_means.get(lab, 0.0),
              "mean_ep_len": ep_lens.get(lab, 0.0)}
        for lab in ext_labels
    }


def build_bot_eval_block(
    win_rates: dict[str, float],
    reward_means: dict[str, float],
    ep_lens: dict[str, float],
    td_resid_tails: "dict[str, float] | None" = None,
) -> dict:
    """Build the standard bot-eval metrics dict for metadata.json (opponents last).

    ``td_resid_tails`` (#4, optional) maps opponent → TD-residual tail (CVaR); when present a
    ``td_resid_tail_mean`` headline (mean over the per-opponent tails) is added and each
    opponent's own tail is folded into its ``opponents[name]`` entry. Omitted entirely when no
    captured battles produced residuals, so the block is byte-identical to before when unused."""
    td_resid_tails = td_resid_tails or {}
    block = {
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
                **({"td_resid_tail": td_resid_tails[name]} if name in td_resid_tails else {}),
            }
            for name in win_rates
        },
    }
    if td_resid_tails:
        block["td_resid_tail_mean"] = sum(td_resid_tails.values()) / len(td_resid_tails)
    return block


def _record_opponent_elos(fit, bot_names, sentinels, tui):
    """Write per-opponent ELO into the TUI dict: each bot's anchored rating (``eval/elo_vs_<name>``)
    + each sentinel's rating positionally (``eval/elo_vs_sentinel_<i>``, matching the win-rate
    rows). Shared by the live ``record_elo`` and the resume republish so keys/format match. The
    single-cycle sentinel rating is anchored only via the (greedy) trainee, so it's a rough
    estimate — the offline `python -m main.elo` fit (full per-snapshot history) is canonical."""
    bot_r = fit.bot_ratings()
    for name in bot_names:
        br = bot_r.get(name)
        if br is not None:
            tui[f"eval/elo_vs_{name}"] = round(br[0])
    for i, s in enumerate(sentinels):
        step = s.get("step") if isinstance(s, dict) else None
        if step is None:
            continue
        sr = fit.rating_for_step(int(step))
        if sr is not None:
            tui[f"eval/elo_vs_sentinel_{i}"] = round(sr[0])


def record_elo(model_dir, step, bot_win_rates, sentinels, n_games, logger, tui,
               bot_td_tails=None, bot_counts=None, externals=None):
    """Append this cycle's results to ``eval_results.jsonl``, refit anchored Bradley-Terry
    ELO, and record ``eval/elo`` + ``eval/elo_ci`` to the SB3 logger + the TUI dict.

    Shared by BOTH eval callbacks so the bot-only and self-play paths surface ELO
    identically. ``sentinels`` is ``[{"step", "win_rate"}, …]`` (``[]`` on the bot path).
    Returns ``(elo, ci_halfwidth)`` for the current snapshot, or ``None``. The live number
    is the best estimate from data SO FAR (batch-BT is global, so early points retro-adjust
    as more cycles land); ``python -m main.elo`` re-fits canonically offline. Best-effort —
    never raises into the eval path. The import is lazy to avoid any import cycle."""
    if not model_dir:
        return None
    try:
        from agents.model.snapshot import append_eval_result_row
        from agents.training import elo as elo_mod

        append_eval_result_row(model_dir, step, n_games, bot_win_rates, sentinels,
                               bot_td_tails=bot_td_tails, bot_counts=bot_counts,
                               externals=externals)
        # Refits the WHOLE accumulated ladder to read this snapshot's rating. Cheap at the
        # expected scale (tens of snapshots → ms); wrapped best-effort so it can never break eval.
        fit = elo_mod.fit_from_run(model_dir, source="log")
        rating = fit.rating_for_step(step)
        if rating is None:
            return None
        elo_val, se = rating
        ci = elo_mod.ci95(se)
        logger.record("eval/elo", elo_val)
        logger.record("eval/elo_ci", ci)
        tui["eval/elo"] = elo_val
        tui["eval/elo_ci"] = ci
        # Per-opponent ELO for the eval panel: each bot's anchored rating + each sentinel's
        # rating (positional, matching the win-rate rows). TUI-only (no per-opponent TB clutter).
        _record_opponent_elos(fit, bot_win_rates, sentinels, tui)
        return elo_val, ci
    except Exception as e:  # noqa: BLE001 — ELO is telemetry; never break eval
        print(f"⚠️ [ELO] live rating failed at step {step}: {e}")
        return None


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

    def __init__(self, *args, reward_fn_factory, gamma: float = 0.99,
                 loss_quota=_FORENSIC_LOSS_QUOTA, win_quota=_FORENSIC_WIN_QUOTA, **kwargs):
        # reward_fn_factory is REQUIRED (no silent default): eval MUST measure with the run's actual
        # RewardConfig, not a bare Gen3RewardManager() — a default here once silently scored eval with
        # bias_redesign=False (the old anti-spam reward), making the eval reward meaningless. Callers
        # build it from RewardConfig.from_dict(<model_dir>/model_config.json). See build_eval_players.
        super().__init__(*args, **kwargs)
        self._init_reward_tracking(reward_fn_factory)
        self._gamma = float(gamma)
        self._loss_quota = loss_quota
        self._win_quota = win_quota
        self._forensic_dir: str | None = None
        self._forensic_step = 0
        self._trace_tag = ""
        self._wins_kept = 0
        self._losses_kept = 0
        self._trace_idx = 0
        self._recorders: dict[str, BattleRecorder] = {}
        # δ residuals pooled across THIS matchup's captured battles (one EvalRLPlayer per
        # opponent), folded into a tail statistic at collect via td_tail(). Reset each cycle.
        self._td_pool: list[float] = []

    def begin_forensic_cycle(self, forensic_dir: str | None, step: int, *,
                             trace_tag: str = "", win_quota: int | None = None,
                             loss_quota: int | None = None) -> None:
        """Arm (or disable, if dir is None) forensic capture for one eval cycle / shard.

        ``trace_tag`` namespaces this player's persisted trace filenames. Under battle-level
        work-stealing several shard units of the SAME opponent write into the same
        ``eval_traces/step_<N>/<opponent>/`` dir, each from a fresh player whose ``_trace_idx``
        restarts at 0 — so without a per-shard tag the files (``win_001`` …) would collide and
        overwrite. ``win_quota`` / ``loss_quota`` override the per-cycle defaults so a sharded
        opponent's per-unit quota can be scaled down (≈ global ÷ shard_count) to keep the total
        traces per opponent bounded."""
        self._forensic_dir = forensic_dir
        self._forensic_step = step
        self._trace_tag = trace_tag
        if win_quota is not None:
            self._win_quota = win_quota
        if loss_quota is not None:
            self._loss_quota = loss_quota
        self._wins_kept = 0
        self._losses_kept = 0
        self._trace_idx = 0
        self._recorders.clear()
        self._td_pool = []

    def td_tail(self):
        """Lower-tail (CVaR) of this matchup's per-decision critic surprise, or None if no
        captured battles produced residuals this cycle. The eval cycle records it as
        eval/td_resid_tail_vs_<opponent>."""
        return td_tail(self._td_pool)

    def td_residuals(self) -> list[float]:
        """The raw per-decision δ samples pooled this matchup (a COPY). A sharded eval ships these
        from each shard and pools them in the parent before the single CVaR — a CVaR cannot be
        averaged across shards, so the raw samples (not a per-shard tail) are the unit of exchange."""
        return list(self._td_pool)

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
                rec = BattleRecorder(battle.battle_tag, self._reward_fn_factory, gamma=self._gamma)
                self._recorders[battle.battle_tag] = rec
            rec.record(battle, idx, probs, mask, state=getattr(self, "_last_prediction", None))
        return self.action_to_order(idx, battle)

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)  # reward finalize (mixin)
        rec = self._recorders.pop(battle.battle_tag, None)
        if rec is None:
            return
        # Harvest δ from EVERY captured battle (even one whose trace we drop below for quota) —
        # the tail metric wants signal from all the V(s) we paid for, not just the persisted sample.
        self._td_pool.extend(rec.td_residuals())
        # Persist this trace only if its outcome is one we still want a sample of;
        # otherwise drop the buffered capture (we already have enough of that result).
        if battle.won and self._wins_kept < self._win_quota:
            outcome = "win"
        elif battle.lost and self._losses_kept < self._loss_quota:
            outcome = "loss"
        else:
            return
        self._trace_idx += 1
        # `_trace_tag` namespaces the file so concurrent shard units of the same opponent (each a
        # fresh player with _trace_idx restarting at 0, all writing this one dir) never collide.
        # `trace_filename_stem` is the single source of the naming contract the prober inverts.
        prefix = os.path.join(self._forensic_dir,
                              trace_filename_stem(outcome, self._trace_tag, self._trace_idx))
        write_battle_record(prefix, rec, battle, self._forensic_step)
        # Bridge battles also have a full-information reconstruction record (seed + both
        # teams + command log, captured at the bridge layer). Registering this trace's
        # prefix lets the registry join the two and persist it as the SEPARATE
        # `<prefix>_reconstruction.json` artifact — deliberately not part of the
        # summary/states the obs pipeline reads (the one-sided/omniscient wall).
        # Websocket eval never produces a record, so the registration just evicts.
        register_trace_prefix(battle.battle_tag, prefix,
                              extra={"trainee_username": self.username})
        if outcome == "win":
            self._wins_kept += 1
        else:
            self._losses_kept += 1


# ── Force-an-eval-now request channel (the launcher "force eval" button) ──────
# The launcher TUI's `f` key (confirm → SIGUSR2) lets an operator trigger an
# off-cadence eval cycle. The SIGUSR2 handler (train_rl_agent._setup_signal_handlers)
# runs in signal context, so it does the minimum — flip this process-global Event;
# whichever eval callback is active CONSUMES it once on its next `_on_step`. A request
# that arrives while a cycle is already in flight is REJECTED (reported to the launcher
# Events panel), mirroring the normal cadence's skip-while-running rule.
_force_eval_event = threading.Event()


def request_forced_eval() -> None:
    """Ask the active eval callback to launch an eval cycle at its next training step.

    Async-signal-safe (``Event.set`` only briefly takes an internal lock and never
    allocates), so it is callable straight from the SIGUSR2 handler. Idempotent:
    stacking N requests before the next ``_on_step`` collapses to a single launch —
    we never want a queue of forced cycles."""
    _force_eval_event.set()


def consume_forced_eval_request() -> bool:
    """Check-and-clear the forced-eval request; ``True`` iff one was pending.

    The only setter is the signal handler and the only consumer is the single training
    thread, so the check-then-clear is race-free in practice (a signal landing in the
    gap is simply caught on the next step)."""
    if _force_eval_event.is_set():
        _force_eval_event.clear()
        return True
    return False


class _ForcedEvalMixin:
    """Shared 'force an eval cycle now' handling for BOTH eval callbacks.

    Mixed into ``PerOpponentEvalCallback`` and ``SelfPlayCallback`` (which otherwise
    share only ``BaseCallback``) so the force path can't drift between them. Each calls
    ``_maybe_force_eval`` from its ``_on_step``; the mixin reads the duck-typed eval
    state both classes expose (``_eval_root`` / ``_pending`` / ``_last_eval_step`` /
    ``num_timesteps``) and drives the subclass's own ``_launch_eval``."""

    def _maybe_force_eval(self) -> None:
        """If a forced-eval request is pending, launch an off-cadence cycle now — or
        REJECT it (reported to the launcher Events panel) when one is already running."""
        if not consume_forced_eval_request():
            return
        if getattr(self, "_eval_root", None) is None:
            send_event("⚡ Force-eval ignored — eval is disabled (no run dir)")
            return
        if self._pending is not None:
            send_event(
                "⏳ Force-eval rejected — an eval cycle is already running "
                f"(step {self._pending['step']:,})"
            )
            return
        send_event(f"⚡ Force-eval — launching an eval cycle now (step {self.num_timesteps:,})")
        # Consume the current cadence bucket too, so the regular schedule check below
        # can't double-launch this same step (the next boundary still fires normally).
        self._last_eval_step = self.num_timesteps
        self._launch_eval()


class PerOpponentEvalCallback(_ForcedEvalMixin, BaseCallback):
    """
    Evaluates the trained agent against the full bot roster on a flat schedule
    (`EVAL_FREQ_STEPS` / `EVAL_GAMES`), but does so in a **separate subprocess** with
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
        best_model_save_path: str | None = None,
        n_workers: int = 3,
        eval_device: str = "cpu",
        eval_concurrency: int = _EVAL_SUBPROCESS_CONCURRENCY,
        eval_shard_games: int = EVAL_SHARD_GAMES,
        showdown_port: int | None = None,
        use_showdown_bridge: bool = False,
        bridge_impl: str = "node",
        resume_eval_metadata: str | None = None,
        keep_eval_snapshots: int = 10,
        keep_eval_trace_steps: int = 20,
        keep_stalls: int = KEEP_STALLS_DEFAULT,
        keep_crashes: int = KEEP_CRASHES_DEFAULT,
        fixed_opponents: "list | None" = None,
        trainee_team_str: "str | None" = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._model_dir = model_dir
        self._server_config = server_config
        # SPECIALIST eval alignment (--trainee-team): the raw Showdown-export team string the
        # TRAINEE is pinned to, threaded into every eval-worker cfg so eval measures the model
        # piloting the team it actually trains — None = the default pool builder. Without this the
        # worker's hardcoded default measured specialists on random teams (pure OOD).
        self._trainee_team_str = trainee_team_str
        # Stable cross-run opponents (FixedOpponentEntry list) — played as an extra ext_ eval
        # matchup each cycle, kept out of win_rate_vs_bots / the ELO fit.
        self._fixed_opponents = list(fixed_opponents or [])
        self.best_model_save_path = best_model_save_path
        self._n_workers = max(1, n_workers)
        self._eval_device = eval_device
        self._eval_concurrency = eval_concurrency
        # Games per work-steal shard unit (battle-level work-stealing); see EVAL_SHARD_GAMES.
        self._eval_shard_games = max(1, eval_shard_games)
        self._showdown_port = showdown_port
        # Bridge eval: workers play in-process via run_local_battles (no server connection).
        self._use_showdown_bridge = use_showdown_bridge
        # Which bridge child the workers spawn when use_showdown_bridge: "node" | "rust".
        self._bridge_impl = bridge_impl
        # >0: persist the eval weight snapshot into eval_traces/step_<N>/snapshot.zip
        # (keeping only the N most-recent) so the prober can reload the bit-exact model
        # that produced a cycle's traces. 0 disables (traces still carry the manifest).
        self._keep_eval_snapshots = max(0, keep_eval_snapshots)
        # The trainer writes the forensic traces, so it grooms them: after each cycle
        # keep only the N most-recent eval step dirs (0 = keep all). Older dirs are
        # removed whole. `python -m main.prober.groom` is the manual fallback.
        self._keep_eval_trace_steps = max(0, keep_eval_trace_steps)
        # Bound the per-run debug-artifact dirs each cycle too: keep the N most-recent
        # stalls/*.html + crashes/*.txt (0 = keep all). See artifact_retention.py.
        self._keep_stalls = max(0, keep_stalls)
        self._keep_crashes = max(0, keep_crashes)
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
        return EVAL_FREQ_STEPS, EVAL_GAMES

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self._model_dir is not None:
            self._eval_root = os.path.join(self._model_dir, ".eval_runs")
            os.makedirs(self._eval_root, exist_ok=True)
        # Resumed run: re-publish the last eval so the TUI panel isn't blank until
        # the next cycle (which can be millions of steps away).
        self._replay_last_eval_to_tui()
        # Restore the last eval step so a resume doesn't re-eval the same checkpoint
        # immediately (it waits for the next cadence boundary instead).
        self._last_eval_step = latest_recorded_eval_step(self._model_dir, self._resume_eval_metadata)

    def _on_step(self) -> bool:
        if self._pending is not None:
            now = time.monotonic()
            if self._all_done(self._pending):
                self._collect_pending()
            elif now - self._pending.get("launched_at", now) > _EVAL_CYCLE_TIMEOUT_SEC:
                self._abort_pending_cycle()   # hung worker → don't wedge eval forever
        if self.num_timesteps == 0:
            return True
        self._maybe_force_eval()   # launcher "force eval" button (SIGUSR2); rejects if running
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
        # Clear any crash-leftover from a prior run at this step (the same step re-evals on resume),
        # so no stale plan/shard/lock files from an aborted cycle are mistaken for this one's.
        shutil.rmtree(run_dir, ignore_errors=True)
        claim_dir = os.path.join(run_dir, "claims")
        os.makedirs(claim_dir, exist_ok=True)

        snapshot_base = os.path.join(run_dir, "snapshot")
        self.model.save(snapshot_base)  # SB3 appends .zip
        snapshot_zip = snapshot_base + ".zip"

        # Full work-steal universe = the bot roster + any stable cross-run opponents (ext_<label>),
        # each an EvalItem the pool splits into shard units. The plan.json (written below) is the
        # single source of truth for the items + shard split — workers and collect read it.
        fixed_cfgs = [e.to_cfg() for e in self._fixed_opponents]
        items = [EvalItem(name, BOT, n_games) for name in eval_opponent_names()]
        items += [EvalItem(f["label"], FIXED, n_games, path=f["path"],
                           config_path=f.get("config_path"), team_str=f.get("team_str"))
                  for f in fixed_cfgs]
        names = [it.key for it in items]
        pool = ShardedEvalPool(items, self._eval_shard_games, step=step)
        pool.write_plan(run_dir)
        # Record exactly which model produced this cycle's traces (the prober reads
        # this to reload the right model). snapshot=None now; _persist_snapshot patches
        # it to the retained filename on success when --keep-eval-snapshots is set.
        write_eval_manifest(self._model_dir, step, opponents=names, n_games=n_games,
                            trainee_team_str=self._trainee_team_str,
                            opponent_pins={e.label: e.team_str for e in self._fixed_opponents})
        # Process-unique account tag (per-process nonce + per-cycle counter), NOT the step:
        # the resume re-eval fires at the same step every restart, so a step tag collided
        # across restarts and hung a worker on a lingering challenge.
        self._eval_cycle += 1
        cycle_tag = f"{self._eval_run_nonce}{_b36(self._eval_cycle, 1)}"
        # Cap by UNITS, not opponents — sharding gives many more units than opponents, so the full
        # worker pool can help drain the tail (the whole point).
        n_workers = max(1, min(self._n_workers, pool.n_units))
        base_cfg = {
            "snapshot": snapshot_zip,
            "port": self._showdown_port,
            "use_showdown_bridge": self._use_showdown_bridge,
            "bridge_impl": self._bridge_impl,
            "model_dir": self._model_dir,
            "step": step,
            "claim_dir": claim_dir,
            "result_dir": run_dir,         # plan.json lives here; workers write shard__<unit>.json
            "concurrency": self._eval_concurrency,
            "device": self._eval_device,
            "cycle_tag": cycle_tag,
            # Run's discount → the recorder's δ uses the real γ (not the 0.99 fallback), so the
            # live td-residual tail matches the prober's offline _td at the same γ.
            "gamma": float(self.model.gamma),
            # This run's arch toggles → the worker's current_model_version gates sentinel/foreign
            # snapshots against the RUN's real arch (belief-ON / popart / …), not a toggle-OFF default
            # that would FATAL on the run's own belief-ON sentinels.
            "arch_toggles": arch_toggles_from_model(self.model),
            # --trainee-team pin (None = default pool): eval measures the trainee ON ITS OWN TEAM.
            "trainee_team_str": self._trainee_team_str,
        }
        procs = spawn_eval_workers(run_dir, base_cfg, n_workers)

        self._pending = {"step": step, "names": names, "procs": procs,
                         "snapshot": snapshot_zip, "run_dir": run_dir, "n_games": n_games,
                         "launched_at": time.monotonic()}
        print(f"[EVAL] step {step:,}: spawned {n_workers} work-stealing worker(s) on "
              f"{self._eval_device} ({len(names)} opponents, {pool.n_units} shard units, "
              f"conc {self._eval_concurrency}) — non-blocking")
        send_event(f"🧪 Eval @ {step:,}: started "
                   f"({len(names)} opponents, {pool.n_units} units, {n_workers} worker(s))")

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

        # Battle-level work stealing writes one shard__<unit_id>.json per played shard; the parent
        # pools an opponent's shards back exactly. Missing = a name with ZERO shards (a worker died
        # holding every one of its claims) — log and carry on; partial coverage is warned inside.
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

        self._record(step, merged, pending["n_games"], n_workers=len(pending["procs"]))
        self._maybe_save_best(step, pending, merged["win_rates"])
        self._persist_snapshot(pending)
        self._prune_eval_traces()   # trainer grooms the traces it writes
        self._prune_run_artifacts()  # …and bounds its stalls/ + crashes/ dirs
        self._cleanup(pending, keep_logs=bool(missing or bad_exits))

    def _record(self, step: int, merged: dict, n_games: int = EVAL_GAMES,
                n_workers: int = 1) -> None:
        win_rates = merged["win_rates"]
        reward_means = merged["reward_means"]
        ep_lens = merged["ep_lens"]
        td_tails = merged.get("td_resid_tails", {})
        # Stable cross-run opponents (ext_<label>) are a SEPARATE yardstick: per-opponent metrics
        # are recorded (the loop below), but they are kept OUT of the bot aggregate / best-model
        # signal / ELO fit so they never move the curriculum (bot_mean already excludes them).
        bot_wr = {k: v for k, v in win_rates.items() if not is_external(k)}
        ext_wr = {k: v for k, v in win_rates.items() if is_external(k)}
        bot_td_tails = {k: v for k, v in td_tails.items() if not is_external(k)}
        aggregate = sum(bot_wr.values()) / len(bot_wr) if bot_wr else 0.0
        bot_rewards = {k: v for k, v in reward_means.items() if not is_external(k)}
        aggregate_reward = sum(bot_rewards.values()) / len(bot_rewards) if bot_rewards else 0.0
        wr_bots = bot_mean(win_rates)
        rew_bots = bot_mean(reward_means)
        eplen_bots = bot_mean(ep_lens)
        wr_external = external_aggregate(ext_wr)
        total_dur = sum(merged["durations_sec"].values())

        tui: dict[str, float] = {}
        # Per-opponent win/reward/ep_len for every opponent incl. stable ones (shared recorder).
        record_per_opponent(self.logger, tui, win_rates, win_rates, reward_means, ep_lens)
        # TD-residual tail is a BOT/sentinel critic-coverage diagnostic — NOT emitted for stable
        # opponents (a display-only yardstick; uniform with the self-play path, which omits it too).
        for name in bot_td_tails:
            self.logger.record(f"eval/td_resid_tail_vs_{name}", bot_td_tails[name])
            tui[f"eval/td_resid_tail_vs_{name}"] = bot_td_tails[name]
        self.logger.record("eval/win_rate_mean", aggregate)
        self.logger.record("eval/win_rate_vs_bots", wr_bots)
        self.logger.record("eval/mean_reward_mean", aggregate_reward)
        self.logger.record("eval/mean_reward_vs_bots", rew_bots)
        self.logger.record("eval/mean_ep_len_vs_bots", eplen_bots)
        if wr_external is not None:
            self.logger.record("eval/win_rate_vs_external", wr_external)
            tui["eval/win_rate_vs_external"] = wr_external
        self.logger.record("eval/duration_sec", total_dur)
        # TD-residual tail headline (#4): mean of the per-opponent tails (a mean-of-CVaRs). Only
        # recorded when captured battles produced residuals — lower/more-negative = critic more
        # often blindsided; the leading indicator for the critic-coverage obs work.
        td_tail_mean = sum(bot_td_tails.values()) / len(bot_td_tails) if bot_td_tails else None
        if td_tail_mean is not None:
            self.logger.record("eval/td_resid_tail_mean", td_tail_mean)
            tui["eval/td_resid_tail_mean"] = td_tail_mean
        # Anchored-BT ELO from the accumulated results (appends this cycle's row first).
        # bot_wr carries every BOT incl. random (a valid anchor); ext_ opponents are kept OUT of the
        # fit (no ladder distortion); no sentinels on the bot-only path.
        bot_counts = {k: merged["counts"][k] for k in bot_wr if k in merged.get("counts", {})}
        # The vs-target record (e.g. the exploiter VERDICT) rides the append-only jsonl too — it
        # used to live only in the overwritten latest_eval block + TensorBoard.
        ext_block = {k: {"win_rate": v, "counts": merged.get("counts", {}).get(k)}
                     for k, v in ext_wr.items()}
        elo_result = record_elo(self._model_dir, step, bot_wr, [], n_games,
                                self.logger, tui, bot_td_tails=bot_td_tails, bot_counts=bot_counts,
                                externals=ext_block or None)
        # ELO for each stable opponent (display-only) → fills the eval table's elo column for the
        # ext_ rows: its OWN recorded ELO when available, else a ballpark from the trainee's rating.
        if ext_wr:
            record_external_elos(self.logger, tui, elo_result[0] if elo_result else None, ext_wr,
                                 {e.label: e.source_elo for e in self._fixed_opponents})
        # Record at the SNAPSHOT step so the eval curve aligns to when the model was
        # frozen, not the (later) step at which the worker happened to finish.
        self.logger.dump(step)

        tui.update({
            "eval/win_rate_mean": aggregate, "eval/win_rate_vs_bots": wr_bots,
            "eval/mean_reward_mean": aggregate_reward, "eval/mean_reward_vs_bots": rew_bots,
            "eval/mean_ep_len_vs_bots": eplen_bots, "eval/duration_sec": total_dur,
            # Worker count so the TUI can show per-worker wall-clock (duration_sec is the
            # SUM of per-opponent durations; the pool runs them across n_workers subprocesses).
            "eval/n_workers": float(max(1, n_workers)),
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
            # Bot-only dicts so the block's win_rate_mean / mean_reward_mean agree with each other
            # and with the live TB values (ext_ is recorded separately in `externals` below).
            block = build_bot_eval_block(bot_wr, bot_rewards, ep_lens, bot_td_tails)
            if elo_result:
                block["elo"], block["elo_ci"] = elo_result
            if ext_wr:
                # Stable cross-run opponents — recorded as a separate yardstick (display-only).
                block["externals"] = build_externals_block(ext_wr, win_rates, reward_means, ep_lens)
                if wr_external is not None:  # only the multi-opponent aggregate
                    block["win_rate_vs_external"] = wr_external
            record_eval_results(self._model_dir, step, block)

    def _maybe_save_best(self, step: int, pending: dict, win_rates: dict) -> None:
        # Best-model is chosen on the BOTS (+ random), not the stable cross-run opponents — an
        # ext_ yardstick must never drive checkpoint selection.
        bot_wr = {k: v for k, v in win_rates.items() if not is_external(k)}
        if not bot_wr:
            return
        aggregate = sum(bot_wr.values()) / len(bot_wr)
        if aggregate <= self._best_aggregate_win_rate:
            return
        self._best_aggregate_win_rate = aggregate
        if self.best_model_save_path is not None:
            # The frozen snapshot IS the best model — copy it rather than re-saving.
            dst = os.path.join(self.best_model_save_path, "best_model.zip")
            shutil.copy2(pending["snapshot"], dst)
            copy_run_config_to_best_model(self._model_dir, self.best_model_save_path)  # model_config.json
            write_best_model_sidecar(self._model_dir, dst, self.model)                 # best_model.json (+ELO)
            print(f"[EVAL] new best ({aggregate * 100:.1f}%) saved to {dst}")

    def _persist_snapshot(self, pending: dict) -> None:
        persist_eval_snapshot(self._model_dir, pending["step"], pending["snapshot"],
                              self._keep_eval_snapshots)

    def _prune_eval_snapshots(self) -> None:
        prune_eval_snapshots(self._model_dir, self._keep_eval_snapshots)

    def _prune_eval_traces(self) -> None:
        prune_eval_traces(self._model_dir, self._keep_eval_trace_steps)

    def _prune_run_artifacts(self) -> None:
        prune_run_artifacts(self._model_dir, self._keep_stalls, self._keep_crashes)

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
