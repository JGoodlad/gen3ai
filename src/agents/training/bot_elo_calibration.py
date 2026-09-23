"""Bot-vs-bot ELO anchor calibration — incremental, resumable, offline (no server).

WHY
---
The snapshot ELO ladder (``agents.training.elo``) needs a FIXED anchor so a frozen
snapshot lands on one absolute scale. The 9 eval bots (``random`` + 8 archetypes) never
change, so they are the natural yardstick — but in eval data they only appear AS opponents
of snapshots (no bot-vs-bot games), so without a dedicated calibration their relative
strengths are only weakly determined (indirectly, via the snapshots).

This script plays a dense **round-robin** of all 9 bots, fits Bradley-Terry on the
high-N matrix, and writes ``data/gen3_bot_elo_anchors.json``. The ELO fit then *pins* all 9
bots to those high-confidence ratings and fits only snapshots against that rigid frame — and
because the anchor is identical across runs, snapshot ELOs become directly comparable
run-to-run.

INCREMENTAL / RESUMABLE
-----------------------
Game counts accumulate to a **store** (``designs/ai_v5/elo_calibration/gen3_bot_elo_games.json``)
and the anchor is **re-fit and rewritten after every pair-chunk**. The store + the win-matrix
heatmap are calibration *artifacts* (provenance/viz), so they live with the ELO design work under
``designs/ai_v5/``; only the immutable bot anchor (``data/gen3_bot_elo_anchors.json``) — the file the
runtime actually reads — lives in ``data/``. So you never need a big pause: run it in the
background (even alongside training at low ``--concurrency``) and it builds the anchor up over
time toward ``--target-games``; stop with Ctrl-C and the partial anchor is already valid;
resume later and it continues from the accumulated counts. ``--max-minutes`` self-stops
cleanly at a deadline. The store records a ``git_hash`` — if bot logic changed since, it warns
(re-run with ``--reset`` to start the counts over).

Battles run **in-process via the local BattleStream bridge** — no server, no port, no :8001
risk. Bots are cheap decision functions (no NN forward); the cost is one node subprocess per
battle. Bots are built once and reused across pairs (``reset_battles`` between).

Usage::

    # accumulate toward 5000 games/pair, persisting continuously (resumable):
    python -m agents.training.bot_elo_calibration --target-games 5000 --concurrency 14
    # add a quick first pass now, more later — both resume the same store:
    python -m agents.training.bot_elo_calibration --target-games 500
    python -m agents.training.bot_elo_calibration --target-games 2000   # adds 1500 more/pair
    # bound a session to a time window (self-stops cleanly with a valid anchor):
    python -m agents.training.bot_elo_calibration --target-games 5000 --max-minutes 100
"""
from __future__ import annotations

import os
import sys
import time
import asyncio
import argparse
import itertools
from datetime import datetime, timezone

from poke_env.ps_client import LocalhostServerConfiguration

from agents.training import elo as elo_mod
from agents.training import bot_elo_store
from agents.training.eval_callback import build_eval_opponents, eval_opponent_names
from utils.bridge.local_battle_runner import run_local_battles
from utils.git import get_git_hash, get_repo_root
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

# The ANCHOR is runtime reference data the model reads → data/ (immutable until bots change).
# The raw game-count STORE (resume state) and the HEATMAP (a viz) are calibration *artifacts* /
# provenance, not runtime inputs → they live with the ELO design work under designs/ai_v5/.
DEFAULT_OUT = os.path.join("data", "gen3_bot_elo_anchors.json")
_ARTIFACT_DIR = os.path.join("designs", "ai_v5", "elo_calibration")
DEFAULT_STORE = os.path.join(_ARTIFACT_DIR, "gen3_bot_elo_games.json")
DEFAULT_HEATMAP = os.path.join(_ARTIFACT_DIR, "gen3_bot_elo_anchors_heatmap.png")


def _build_bot(name: str, all_teams, sample_teams, tag: str):
    """Construct one roster bot for bridge play (start_listening=False → no websocket).

    Each bot gets its OWN teambuilder instance so the two sides sample teams independently
    (a shared teambuilder would race on ``_current_packed_team``)."""
    tb = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
    (_n, player), = build_eval_opponents(
        LocalhostServerConfiguration, tb, [name], tag, start_listening=False)
    return player


async def _play(p_a, p_b, n_games: int, concurrency: int) -> None:
    await run_local_battles(p_a, p_b, n_games, concurrency=concurrency)


def _play_chunk(pa, pb, n_games: int, concurrency: int) -> tuple[int, int, int]:
    """Play ``n_games`` of one pair; return (wins_a, wins_b, finished) for THIS chunk."""
    pa.reset_battles()
    pb.reset_battles()
    asyncio.run(_play(pa, pb, n_games, concurrency))
    return pa.n_won_battles, pb.n_won_battles, pa.n_finished_battles


# ── Fit + anchor write (the resumable game-count store lives in bot_elo_store.py) ──


def _fit_quality(ratings: dict, results: list[tuple[str, str, int, int]]) -> dict:
    """Mean / max |predicted − observed| win rate across pairs — high error ⇒ the bot
    ensemble is non-transitive (a single ELO per bot is a lossy summary). The anchor still
    works: snapshots play ALL bots, so their rating reflects the ensemble."""
    errs = []
    for a, b, wa, g in results:
        if g <= 0:
            continue
        errs.append(abs(elo_mod.win_prob(ratings[a], ratings[b]) - wa / g))
    if not errs:
        return {"mean_abs_err": 0.0, "max_abs_err": 0.0, "n_pairs": 0}
    return {"mean_abs_err": round(sum(errs) / len(errs), 4),
            "max_abs_err": round(max(errs), 4), "n_pairs": len(errs)}


def _fit_and_write(store: dict, names: list[str], out_path: str, base: float,
                   target: int) -> dict:
    """Re-fit BT from the accumulated store and (atomically) rewrite the anchor json.
    Returns the anchor dict (also when no pair has games yet → empty ratings)."""
    results, win_matrix = bot_elo_store.results_and_matrix(store, names)
    if not results:
        return {}
    ratings, se, converged = elo_mod.fit_pairwise(results, pinned={"random": base}, base=base)
    quality = _fit_quality(ratings, results)
    per_pair_games = {e["a"] + " vs " + e["b"]: e["games"] for e in store["pairs"].values()}
    # min over ALL pairs (an unplayed pair counts as 0), not just the accumulated ones.
    all_pairs = list(itertools.combinations(names, 2))
    min_games = min((bot_elo_store.games(store, a, b) for (a, b) in all_pairs), default=0)
    anchors = {
        "version": 1,
        "target_games": target,
        "min_pair_games": min_games,
        "complete": min_games >= target,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": store.get("git_hash"),
        "elo_per_decade": elo_mod.ELO_PER_DECADE,
        "base": base,
        "converged": converged,
        "ratings": {n: round(ratings.get(n, base), 1) for n in names},
        "se": {n: round(se.get(n, 0.0), 1) for n in names},
        "win_matrix": {a: {b: round(v, 4) for b, v in win_matrix[a].items()} for a in names},
        "pair_games": per_pair_games,
        "fit_quality": quality,
    }
    bot_elo_store.atomic_write_json(out_path, anchors)
    return anchors


def _write_heatmap(win_matrix: dict, names: list, path: str) -> str | None:
    """Best-effort win-matrix heatmap PNG (skipped if matplotlib is unavailable)."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    m = np.array([[win_matrix[a].get(b, 0.5 if a == b else 0.0) for b in names]
                  for a in names])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Bot round-robin win rate (row vs col)")
    fig.colorbar(im, ax=ax, label="row win rate")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ── Driver ────────────────────────────────────────────────────────────────────


def calibrate(target: int, chunk: int, concurrency: int, out_path: str, store_path: str,
              base: float, reset: bool, max_minutes: float | None,
              heatmap_path: str | None = None) -> dict:
    names = eval_opponent_names()
    git_hash = _safe_git_hash()
    store = bot_elo_store.load(store_path, names, git_hash, reset)

    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    print(f"Building {len(names)} bots…", flush=True)
    bots = {name: _build_bot(name, all_teams, sample_teams, f"Cal{i}")
            for i, name in enumerate(names)}

    pairs = list(itertools.combinations(names, 2))
    t0 = time.time()
    deadline = (t0 + max_minutes * 60) if max_minutes else None
    anchors: dict = {}
    rnd = 0
    stop = False
    while not stop:
        under = [(a, b) for (a, b) in pairs if bot_elo_store.games(store, a, b) < target]
        if not under:
            break
        rnd += 1
        for k, (a, b) in enumerate(under):
            if deadline and time.time() >= deadline:
                print(f"⏱  reached --max-minutes ({max_minutes:.0f}m) — stopping cleanly.", flush=True)
                stop = True
                break
            have = bot_elo_store.games(store, a, b)
            n = min(chunk, target - have)
            wa, wb, g = _play_chunk(bots[a], bots[b], n, concurrency)
            bot_elo_store.accumulate(store, a, b, wa, wb, g)
            bot_elo_store.save(store_path, store)
            anchors = _fit_and_write(store, names, out_path, base, target)
            elapsed = time.time() - t0
            tot = bot_elo_store.games(store, a, b)
            min_g = min(bot_elo_store.games(store, x, y) for (x, y) in pairs)
            print(f"[round {rnd}] {a} vs {b}: +{g} → {tot}/{target}  "
                  f"(min pair {min_g}/{target})  [{elapsed:.0f}s]", flush=True)

    # Final heatmap from whatever accumulated.
    _, win_matrix = bot_elo_store.results_and_matrix(store, names)
    hm_path = heatmap_path or (os.path.splitext(out_path)[0] + "_heatmap.png")
    heatmap = _write_heatmap(win_matrix, names, hm_path)

    _print_anchor_summary(anchors, names)
    print(f"wrote {out_path}  +  store {store_path}" +
          (f"  +  {heatmap}" if heatmap else "  (heatmap skipped — matplotlib unavailable)"))
    return anchors


def _print_anchor_summary(anchors: dict, names: list[str]) -> None:
    """Ranked bot ladder + fit quality. Shared by the live calibrate driver and the ``--merge``
    entry so the two outputs stay identical (all fields read from the anchor dict)."""
    if not anchors.get("ratings"):
        return
    r, se, q = anchors["ratings"], anchors["se"], anchors["fit_quality"]
    print(f"\n── Bot ELO anchors (random pinned at {anchors['base']:.0f}; "
          f"min {anchors['min_pair_games']}/{anchors['target_games']} games/pair) ──")
    for n in sorted(names, key=lambda x: -r[x]):
        print(f"  {n:<16} {r[n]:7.1f}  ± {se[n]:5.1f}")
    print(f"fit: converged={anchors['converged']}  complete={anchors['complete']}  "
          f"mean_abs_err={q['mean_abs_err']}  max_abs_err={q['max_abs_err']}  "
          f"(higher ⇒ more non-transitive)")


def _safe_git_hash() -> str | None:
    try:
        return get_git_hash()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Incremental bot-vs-bot ELO anchor calibration (bridge, no server)")
    ap.add_argument("--target-games", type=int, default=5000,
                    help="accumulate until every pair has this many games (default 5000)")
    ap.add_argument("--chunk", type=int, default=250,
                    help="games per pair per round between persists (default 250)")
    ap.add_argument("--concurrency", type=int, default=12,
                    help="concurrent bridge battles per pair (raise on a free box)")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="self-stop cleanly after this many minutes (anchor stays valid)")
    ap.add_argument("--reset", action="store_true",
                    help="discard the accumulated game store and start counts over")
    ap.add_argument("--out", default=None, help=f"anchor json (default <repo>/{DEFAULT_OUT})")
    ap.add_argument("--games-store", default=None,
                    help=f"accumulator json (default <repo>/{DEFAULT_STORE})")
    ap.add_argument("--heatmap", default=None,
                    help=f"win-matrix heatmap PNG (default <repo>/{DEFAULT_HEATMAP}); "
                         "the artifact, not runtime data")
    ap.add_argument("--base", type=float, default=elo_mod.DEFAULT_BASE,
                    help="Elo value to pin `random` at (default 1000)")
    ap.add_argument("--merge", default=None,
                    help="MERGE MODE: glob of store parts to sum into one anchor (no play); "
                         "e.g. '/tmp/elo_parts/part*.json'")
    args = ap.parse_args()

    repo = get_repo_root()
    out_path = args.out or os.path.join(repo, DEFAULT_OUT)
    store_path = args.games_store or os.path.join(repo, DEFAULT_STORE)
    heatmap_path = args.heatmap or os.path.join(repo, DEFAULT_HEATMAP)

    if args.merge:
        import glob as _glob
        names = eval_opponent_names()
        paths = sorted(_glob.glob(args.merge))
        if not paths:
            print(f"no store parts matched {args.merge!r}", file=sys.stderr)
            return 2
        print(f"merging {len(paths)} store parts → {store_path}")
        merged = bot_elo_store.merge(paths, names, _safe_git_hash())
        bot_elo_store.save(store_path, merged)
        anchors = _fit_and_write(merged, names, out_path, args.base, args.target_games)
        _, win_matrix = bot_elo_store.results_and_matrix(merged, names)
        _write_heatmap(win_matrix, names, heatmap_path)
        _print_anchor_summary(anchors, names)
        print(f"wrote {out_path}")
        return 0

    n_pairs = len(list(itertools.combinations(eval_opponent_names(), 2)))
    print(f"Calibrating {len(eval_opponent_names())} bots ({n_pairs} pairs) → "
          f"target {args.target_games} games/pair, chunk {args.chunk}, concurrency "
          f"{args.concurrency}, via the in-process bridge (no server). Resumable; "
          f"anchor rewritten after every pair-chunk.", flush=True)
    calibrate(args.target_games, args.chunk, args.concurrency, out_path, store_path,
              args.base, args.reset, args.max_minutes, heatmap_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
