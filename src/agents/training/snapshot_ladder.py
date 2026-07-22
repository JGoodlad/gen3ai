"""Frozen-snapshot ELO ladder — the dense, pay-once internal rating.

The live-trainee ELO (``elo.py`` / ``record_elo``) is noisy at the frontier: the fixed bots
have SATURATED (we're ~400 Elo above them, out on the flat tail of the logistic), so their
edges pin the absolute LEVEL but give almost no RESOLUTION — the fine ordering is driven by
the sparse, near-50% sentinel edges sampled during live cycles (±15 Elo CIs).

This module fixes the resolution the other way. A promoted snapshot is FROZEN, so
snapshot-A-vs-snapshot-B is a STATIONARY Bernoulli parameter: measure it ONCE with a dense
round-robin and it is permanent — no drift, never recomputed. So on each promotion we pay a
bounded one-time "tax" (the new frozen snapshot vs every other frozen snapshot in the pool),
building a densely-connected graph among the frozen nodes. The BT fit over that dense matrix,
still anchored to the pinned bots via each snapshot's historical bot edges, gives a
high-resolution RELATIVE ladder of our own promoted history — the frontier-strength yardstick
the saturated bots can no longer be.

Durability: raw pair results append to ``<run>/snapshot_ladder/games.jsonl`` (forever,
race-safe line appends); the fitted ratings + win-matrix + non-transitivity read are rewritten
to ``<run>/snapshot_ladder/ladder.json`` (the sidecar metric). Frozen-vs-frozen means a pair
already in ``games.jsonl`` is NEVER replayed.

Non-transitivity caveat: if the frozen pool is non-transitive (rock-paper-scissors), NO scalar
Elo represents it faithfully, however densely measured — but the dense matrix at least lets
``fit_quality`` (mean/max |predicted − observed|) QUANTIFY the intransitivity, which the sparse
live fit cannot even see.

CLI:
  python -m agents.training.snapshot_ladder <run_dir> --backfill        # one-time back tax
  python -m agents.training.snapshot_ladder <run_dir> --promote <step>  # per-promotion update
"""
from __future__ import annotations

import os

# Cap torch/BLAS intra-op threads BEFORE any (transitive) torch import — mirrors eval_worker.
# Each shard runs on a shared box; without this every process defaults torch to all cores, so N
# parallel shards spawn N×cores threads → oversubscription thrash → battles blow the 180s bridge
# timeout (the 2026-07-22 4-shard-on-16-cores failure). One thread/process + event-loop concurrency
# is the right shape for B=1 CPU inference.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import asyncio
import glob
import itertools
import json
import sys
from datetime import datetime, timezone

from agents.training import elo as elo_mod


# ── paths / durability ────────────────────────────────────────────────────────────────────
def _ladder_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "snapshot_ladder")


def games_log_path(run_dir: str) -> str:
    return os.path.join(_ladder_dir(run_dir), "games.jsonl")


def ladder_json_path(run_dir: str) -> str:
    return os.path.join(run_dir, "snapshot_ladder", "ladder.json")


def _snapshot_zip(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, "snapshots", f"snapshot_{step:012d}.zip")


def pool_snapshot_steps(run_dir: str) -> list[int]:
    """The frozen snapshot steps currently on disk (the pool), ascending."""
    steps = []
    for p in glob.glob(os.path.join(run_dir, "snapshots", "snapshot_*.zip")):
        try:
            steps.append(int(os.path.basename(p).split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def load_games(run_dir: str) -> dict[tuple[int, int], list[int]]:
    """Read games.jsonl → {(lo, hi): [wins_lo, games]}, summing duplicate lines (independent
    samples of the SAME frozen matchup pool across appends; adding them just tightens the edge)."""
    out: dict[tuple[int, int], list[int]] = {}
    path = games_log_path(run_dir)
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                lo, hi = _pair_key(int(r["a"]), int(r["b"]))
                wins_lo = int(r["wins_a"]) if r["a"] == lo else int(r["games"]) - int(r["wins_a"])
                e = out.setdefault((lo, hi), [0, 0])
                e[0] += wins_lo
                e[1] += int(r["games"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def _append_game(run_dir: str, step_a: int, step_b: int, wins_a: int, games: int) -> None:
    """Append one measured pair (race-safe: a single sub-PIPE_BUF line append is atomic)."""
    os.makedirs(_ladder_dir(run_dir), exist_ok=True)
    row = {"a": int(step_a), "b": int(step_b), "wins_a": int(wins_a), "games": int(games),
           "at": datetime.now(timezone.utc).isoformat()}
    with open(games_log_path(run_dir), "a") as f:
        f.write(json.dumps(row) + "\n")


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ── the fit (dense matrix + bot anchors) ────────────────────────────────────────────────────
def fit_ladder(run_dir: str, base: float | None = None) -> dict:
    """Fit the anchored BT ladder from the DENSE frozen matrix + each snapshot's historical
    bot edges (from eval_results.jsonl, which connect the ladder to the pinned bots for the
    absolute scale). Returns the ladder dict (also written to ladder.json)."""
    anchors = elo_mod.load_bot_anchors()
    pins = (anchors or {}).get("ratings")
    base = base if base is not None else (anchors or {}).get("base", elo_mod.DEFAULT_BASE)

    results: list[tuple[str, str, int, int]] = []
    # (1) DENSE frozen-vs-frozen edges — the resolution.
    games = load_games(run_dir)
    for (lo, hi), (wins_lo, g) in games.items():
        if g > 0:
            results.append((elo_mod.snap_key(lo), elo_mod.snap_key(hi), wins_lo, g))
    # (2) each snapshot's historical bot + sentinel edges — the anchor connection + extra data.
    try:
        for na, nb, wa, g in elo_mod._rows_to_results(elo_mod.load_rows(run_dir, source="log")):
            if g > 0:
                results.append((na, nb, wa, g))
    except Exception:  # noqa: BLE001 — the ladder still works from the frozen matrix alone
        pass

    pinned = {elo_mod.bot_key(n): float(e) for n, e in pins.items()} if pins else None
    ratings, se, converged = elo_mod.fit_pairwise(results, pinned=pinned, base=base)

    # non-transitivity read over the dense frozen pairs only (the part a scalar can misrepresent)
    errs = []
    for (lo, hi), (wins_lo, g) in games.items():
        if g > 0 and elo_mod.snap_key(lo) in ratings and elo_mod.snap_key(hi) in ratings:
            p = elo_mod.win_prob(ratings[elo_mod.snap_key(lo)], ratings[elo_mod.snap_key(hi)])
            errs.append(abs(p - wins_lo / g))
    fit_quality = {"mean_abs_err": round(sum(errs) / len(errs), 4) if errs else 0.0,
                   "max_abs_err": round(max(errs), 4) if errs else 0.0,
                   "n_frozen_pairs": len(errs)}

    steps = pool_snapshot_steps(run_dir)
    snap_ratings = {str(s): round(ratings[elo_mod.snap_key(s)], 1)
                    for s in steps if elo_mod.snap_key(s) in ratings}
    snap_se = {str(s): round(se.get(elo_mod.snap_key(s), 0.0), 1)
               for s in steps if elo_mod.snap_key(s) in ratings}
    ladder = {
        "version": 1,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "anchored_to_bots": bool(pins),
        "converged": converged,
        "n_frozen_pairs_measured": sum(1 for v in games.values() if v[1] > 0),
        "n_pairs_possible": len(list(itertools.combinations(steps, 2))),
        "ratings": snap_ratings,
        "se": snap_se,
        "fit_quality": fit_quality,
    }
    _atomic_write_json(ladder_json_path(run_dir), ladder)
    return ladder


# ── playing a frozen pair (bridge, no server) ───────────────────────────────────────────────
def _play_pair(run_dir, step_a, step_b, n_games, mappings, cv, all_teams, sample_teams,
               concurrency, impl):
    """Round-robin one frozen pair on the bridge; return (wins_a, games_finished)."""
    import torch
    torch.set_num_threads(1)  # defensive: B=1 CPU inference; the parallelism is across shards
    from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
    from agents.inference.player import RLPlayer
    from agents.model.snapshot import load_foreign_opponent
    from utils.teambuilder import Gen3Teambuilder
    from utils.bridge.local_battle_runner import run_local_battles

    # Our own snapshots, but this-run's config (PopArt + every arch toggle) differs from a bare
    # current_model_version → load them as FOREIGN opponents: reads each zip's own saved config
    # and skips check_compatible (the eval FIXED-opponent path). config lives beside the snapshots.
    cfg = os.path.join(run_dir, "snapshots", "model_config.json")
    if not os.path.exists(cfg):
        cfg = os.path.join(run_dir, "model_config.json")

    def _player(step, tag):
        model, _ = load_foreign_opponent(_snapshot_zip(run_dir, step), current_version=cv,
                                         device="cpu", config_path=cfg)
        return RLPlayer(
            model=model, team=Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1),
            battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
            mappings=mappings, account_configuration=AccountConfiguration(f"L{tag}", "pw"),
            stochastic=False, start_listening=False)  # greedy = a stable frozen yardstick

    pa = _player(step_a, f"a{step_a % 100000:05d}")
    pb = _player(step_b, f"b{step_b % 100000:05d}")
    pa.reset_battles(); pb.reset_battles()
    asyncio.run(run_local_battles(pa, pb, n_games, concurrency=concurrency, impl=impl))
    return pa.n_won_battles, pa.n_finished_battles


def _measure_missing(run_dir, target_pairs, n_games, concurrency, impl):
    """Play every (a, b) in target_pairs NOT already in games.jsonl; append each. Returns count."""
    from agents.observation.state_encoder import load_mappings
    from agents.model.snapshot import current_model_version
    from utils.team_loader import TeamLoader

    have = load_games(run_dir)
    todo = [(a, b) for (a, b) in target_pairs if _pair_key(a, b) not in have or have[_pair_key(a, b)][1] == 0]
    if not todo:
        return 0
    mappings = load_mappings()
    cv = current_model_version(mappings)
    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    played = 0
    for a, b in todo:
        try:
            wins_a, finished = _play_pair(run_dir, a, b, n_games, mappings, cv, all_teams,
                                          sample_teams, concurrency, impl)
            _append_game(run_dir, a, b, wins_a, finished)
            played += 1
            print(f"[ladder] {a//1_000_000}M vs {b//1_000_000}M: {wins_a}/{finished}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad pair must not abort the sweep
            import traceback
            print(f"[ladder] pair {a} vs {b} FAILED: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()}", flush=True)
    return played


def update_for_promotion(run_dir, new_step, n_games=100, concurrency=4, impl="node") -> dict:
    """The per-promotion tax: play the newly-promoted frozen snapshot vs every OTHER frozen
    snapshot on disk (skipping already-measured pairs), append, refit. Returns the ladder dict."""
    others = [s for s in pool_snapshot_steps(run_dir) if s != new_step]
    _measure_missing(run_dir, [(new_step, o) for o in others], n_games, concurrency, impl)
    return fit_ladder(run_dir)


def backfill(run_dir, n_games=100, concurrency=4, impl="node", shard=None) -> dict:
    """The one-time back tax: round-robin ALL frozen snapshots currently on disk (only the
    pairs not yet in games.jsonl), then refit. Idempotent — reruns skip measured pairs.

    ``shard`` = (i, n): play only pairs whose index % n == i — the disjoint-slice split for
    running N backfill PROCESSES in parallel (true multi-core: each spawns its own bridge; the
    slices are disjoint so no pair is double-played, and games.jsonl appends stay race-safe).
    A sharded run does NOT refit (the last shard to finish, or a `--fit-only`, does)."""
    steps = pool_snapshot_steps(run_dir)
    pairs = list(itertools.combinations(steps, 2))
    if shard is not None:
        i, n = shard
        pairs = [p for k, p in enumerate(pairs) if k % n == i]
        print(f"[ladder] shard {i}/{n}: {len(pairs)} pairs @ {n_games} games", flush=True)
        _measure_missing(run_dir, pairs, n_games, concurrency, impl)
        return {}  # sharded workers don't refit; caller fits once all shards finish
    print(f"[ladder] backfill over {len(steps)} snapshots = {len(pairs)} pairs "
          f"(measuring the missing ones @ {n_games} games)", flush=True)
    _measure_missing(run_dir, pairs, n_games, concurrency, impl)
    return fit_ladder(run_dir)


def latest_promoted_elo(run_dir: str) -> "tuple[int, float, float] | None":
    """(step, elo, se) of the highest-step snapshot in the ladder sidecar, or None. Read by the
    live eval callback to surface eval/ladder_elo without recomputing."""
    try:
        d = json.load(open(ladder_json_path(run_dir)))
        ratings = d.get("ratings") or {}
        if not ratings:
            return None
        step = max(int(k) for k in ratings)
        return step, float(ratings[str(step)]), float((d.get("se") or {}).get(str(step), 0.0))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen-snapshot ELO ladder (dense, pay-once).")
    ap.add_argument("run_dir")
    ap.add_argument("--backfill", action="store_true", help="round-robin the whole current pool")
    ap.add_argument("--promote", type=int, default=None, help="update for one promoted step")
    ap.add_argument("--n-games", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--impl", default="node")
    ap.add_argument("--shard", default=None, help="I:N — play only pair-slice I of N (parallel workers)")
    ap.add_argument("--fit-only", action="store_true", help="refit from games.jsonl, play nothing")
    a = ap.parse_args()
    if a.fit_only:
        ladder = fit_ladder(a.run_dir)
    elif a.promote is not None:
        ladder = update_for_promotion(a.run_dir, a.promote, a.n_games, a.concurrency, a.impl)
    elif a.backfill:
        shard = tuple(int(x) for x in a.shard.split(":")) if a.shard else None
        ladder = backfill(a.run_dir, a.n_games, a.concurrency, a.impl, shard=shard)
        if not ladder:  # sharded worker — no fit, no ladder table to print
            print(f"[ladder] shard {a.shard} done (fit deferred to --fit-only)", flush=True)
            return 0
    else:
        ap.error("pass --backfill, --promote <step>, or --fit-only")
    ranked = sorted(ladder["ratings"].items(), key=lambda kv: -kv[1])
    print(f"\n[ladder] {ladder['n_frozen_pairs_measured']}/{ladder['n_pairs_possible']} pairs | "
          f"non-transitivity mean|err| {ladder['fit_quality']['mean_abs_err']:.3f}")
    for step, elo in ranked:
        print(f"  {int(step)//1_000_000:4d}M  {elo:7.1f} ± {elo_mod.ci95(ladder['se'].get(step, 0.0)):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
