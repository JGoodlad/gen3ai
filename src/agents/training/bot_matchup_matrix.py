"""Bot-vs-bot RAW MATCHUP accumulator — the high-resolution fixed-bot edge set.

WHY
---
The 9 eval bots are the only players in this project whose strength never changes, so the
bot-vs-bot round-robin is the one *stationary* graph we can measure to arbitrary precision.
``bot_elo_calibration`` already plays that round-robin, but it exists to produce a **rating**
(``data/gen3_bot_elo_anchors.json``) and its store keeps only what a Bradley-Terry fit needs
(``wins_a``/``games``, draws folded invisibly into ``games``). Anything that wants the graph
itself — a loop-closing edge set for a Hodge/curl decomposition, a non-transitivity study, a
different rating model — needs the **raw counts at high N, with draws kept separate**, because a
rating is a lossy projection of the matrix and you cannot un-project it.

This module accumulates exactly that: per unordered bot pair, ``wins_a`` / ``wins_b`` / ``draws``
/ ``n``, toward ``--target-per-pair`` (default 10000), in resumable ~1-hour chunks, into a
committed artifact (``data/gen3_bot_matchups.json`` by default, relative to the CWD so a run
from a worktree never dirties the main checkout).

🚨 IT NEVER TOUCHES ``data/gen3_bot_elo_anchors.json``. That anchor is load-bearing for
cross-run ELO comparability (every snapshot ELO in ``models/`` is on its scale), so
regenerating it from this higher-resolution matrix is a deliberate OWNER decision, not a
side effect of accumulating data.

THE PROTOCOL — inherited verbatim from ``bot_elo_calibration``
--------------------------------------------------------------
The whole point of this module is that its counts are *comparable* to the calibration's, so it
does not re-implement the battle protocol: it CALLS ``bot_elo_calibration._build_bot`` and
``bot_elo_calibration._play_chunk``. What that means concretely (recorded into the artifact's
``protocol`` block, so a reader of the json never has to come here):

* **Roster** — ``eval_callback.eval_opponent_names()``: the 9 fixed eval bots (``random`` + the
  4 v1 archetypes + their 4 v2s). 36 unordered pairs.
* **Bot construction** — ``build_eval_opponents(..., start_listening=False)``: no websocket, no
  server, no :8001 risk. Bots are cheap decision functions (no NN forward, no GPU).
* **Team sampling** — each bot gets its OWN ``Gen3Teambuilder(all_teams, bias_teams=sample_teams,
  bias_prob=0.1)`` (a shared instance would race on ``_current_packed_team``), i.e. uniform over
  the full loaded ``data/teams`` pool with a 10% draw from the curated *sample* subset. A fresh
  team is drawn for BOTH sides on every battle, so an edge's count is a marginal over the team
  distribution, not a fixed-team duel.
* **Transport** — ``utils.bridge.local_battle_runner.run_local_battles`` at ``impl="node"`` (its
  default; the calibration passes no ``impl``), one bridge subprocess per battle, format
  ``gen3ou``.
* **Seeding** — unseeded: the bridge mints a fresh PRNG seed per battle.
* **Turn / forfeit settings** — NONE of the training-side caps apply. The 250-turn stall forfeit
  lives in the ``Gen3Env`` wrapper, which is not in this path; the only bound is the bridge's own
  1000-turn runaway cap plus the contention-scaled per-battle timeout. A battle that does not
  finish is simply not counted (see COUNTING), never scored as a draw.
* **Counting** — ``wins_a`` / ``wins_b`` from poke-env's ``n_won_battles`` on each side,
  ``draws = n_finished − wins_a − wins_b`` (gen-3 ties: mutual last-mon KO, and the sim's own
  turn cap), ``n = n_finished``. **Draws are stored separately and never folded into a win** —
  downstream decides half-win vs exclude; the ELO half of the tree treats them as neither.

CRASH SAFETY / CONCURRENCY
--------------------------
Every commit is *load-from-disk → merge our delta → tmp-write → rename*, so (a) a killed chunk
loses at most the in-flight pair-visit, not the chunk, (b) two processes pointed at the same
artifact sum rather than clobber, and (c) a reader never sees a half-written file. The chunk's
``history`` record is upserted by ``chunk_id`` on every commit, so an interrupted chunk still
leaves an honest record of what it added.

BALANCED FILL
-------------
Each visit schedules the pair with the LOWEST current ``n`` (ties broken by canonical key), so
resolution rises uniformly and **any prefix of the process is a valid balanced sample** — stop
whenever, the matrix is usable.

Usage::

    # a ~1 hour chunk, 4 workers, beside a live training run:
    python -m agents.training.bot_matchup_matrix --max-minutes 60 --concurrency 4
    # bound by battle count instead (whichever limit hits first, if both are given):
    python -m agents.training.bot_matchup_matrix --chunk-battles 2000
    # just read the current resolution, play nothing:
    python -m agents.training.bot_matchup_matrix --summary-only
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import uuid
import argparse
import itertools
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# Relative on purpose: a run from a git worktree writes into THAT worktree's data/, so it can
# never dirty the main checkout (which normally carries the live training run).
DEFAULT_OUT = os.path.join("data", "gen3_bot_matchups.json")

DEFAULT_TARGET = 10_000
# Battles per visit to one pair. Small enough that the balanced scheduler actually rotates
# (36 pairs), large enough that the per-commit json rewrite is noise against battle cost.
DEFAULT_GAMES_PER_VISIT = 25
# The box normally carries a live training run — 4 concurrent bridge children is the ceiling
# the owner sanctioned for this job.
DEFAULT_CONCURRENCY = 4
DEFAULT_NICE = 10

ELO_PER_DECADE = 400.0
# One natural-log unit of logit == this many ELO points (400 / ln 10 ≈ 173.72). The conversion
# that turns a binomial standard error into "± X ELO on this edge".
ELO_PER_LOGIT = ELO_PER_DECADE / math.log(10.0)

# Recorded into every artifact. Prose, deliberately: the json must be readable without this file.
# `bridge_impl` is `run_local_battles`'s own default (the calibration passes no `impl`) — pinned
# against drift by bot_matchup_matrix_test.test_recorded_bridge_impl_matches_the_runner_default.
PROTOCOL: dict = {
    "inherited_from": "agents.training.bot_elo_calibration (_build_bot + _play_chunk, called "
                      "directly — this module does not re-implement the battle protocol)",
    "roster": "agents.training.eval_callback.eval_opponent_names() — the 9 fixed eval bots",
    "bot_construction": "build_eval_opponents(..., start_listening=False) — no websocket, "
                        "no server; bots are heuristic decision functions (no NN forward)",
    "team_sampling": "per-bot Gen3Teambuilder(all_teams, bias_teams=sample_teams, "
                     "bias_prob=0.1); a fresh team is drawn for both sides every battle, so an "
                     "edge is a MARGINAL over the team distribution, not a fixed-team duel",
    "battle_format": "gen3ou",
    "transport": "in-process BattleStream bridge "
                 "(utils.bridge.local_battle_runner.run_local_battles), one child per battle",
    "bridge_impl": "node",
    "seeding": "unseeded — the bridge mints a fresh PRNG seed per battle",
    "turn_caps": "no trainer-side cap in this path (the 250-turn stall forfeit lives in the "
                 "Gen3Env wrapper); only the bridge's 1000-turn runaway cap and the "
                 "contention-scaled per-battle timeout apply",
    "forfeit": "none — bots play to a sim-declared win / loss / tie",
    "counting": "wins from poke-env n_won_battles per side; draws = n_finished - wins_a - "
                "wins_b; n = n_finished. A battle that did not finish (timeout) is EXCLUDED "
                "from n, never scored as a draw.",
    "draws": "stored separately and never folded into a win — downstream decides half-win vs "
             "exclude",
}


# ── Artifact: keys, IO, merge ────────────────────────────────────────────────


def pair_key(a: str, b: str) -> str:
    """Canonical unordered key: the two bot ids sorted, joined by ``|``."""
    return "|".join(sorted((a, b)))


def new_store(target: int = DEFAULT_TARGET, bots: list[str] | None = None,
              git_hash: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "target_per_pair": int(target),
        "git_hash": git_hash,
        "bots": list(bots or []),
        "protocol": dict(PROTOCOL),
        "pairs": {},
        "history": [],
    }


def atomic_write_json(path: str, obj: dict) -> None:
    """tmp + rename, so a reader (or a crash) never sees a half-written artifact."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path: str, target: int | None = None, bots: list[str] | None = None,
         git_hash: str | None = None) -> dict:
    """Load the accumulated artifact, or a fresh empty one.

    A missing, unreadable or structurally-wrong file yields an EMPTY store rather than raising —
    that is the crash-recovery path (a killed chunk can leave a stray ``.tmp``, which
    ``os.replace`` semantics mean is never the live file). Metadata from the caller is applied
    on top so a resume records the current target / roster / git hash.
    """
    store = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
        except (OSError, ValueError):
            print(f"⚠️  unreadable artifact {path} — starting fresh counts", file=sys.stderr)
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("pairs"), dict):
            store = loaded
    if store is None:
        store = new_store(target if target is not None else DEFAULT_TARGET, bots, git_hash)
    store.setdefault("schema_version", SCHEMA_VERSION)
    store.setdefault("pairs", {})
    store.setdefault("history", [])
    store["protocol"] = dict(PROTOCOL)
    if target is not None:
        store["target_per_pair"] = int(target)
    if bots:
        store["bots"] = list(bots)
    if git_hash:
        store["git_hash"] = git_hash
    return store


def save(path: str, store: dict) -> None:
    store["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, store)


def entry(store: dict, a: str, b: str) -> dict:
    """The pair's counts (zeros if unplayed). ``a``/``b`` in the entry are the SORTED names, so
    ``wins_a`` always belongs to the lexicographically-lower bot however the caller asked."""
    lo, hi = sorted((a, b))
    return store["pairs"].get(pair_key(a, b),
                              {"a": lo, "b": hi, "wins_a": 0, "wins_b": 0, "draws": 0, "n": 0})


def n_games(store: dict, a: str, b: str) -> int:
    return entry(store, a, b)["n"]


def accumulate(store: dict, a: str, b: str, wins_a: int, wins_b: int, draws: int) -> dict:
    """Add one visit's counts for the pair (a, b), normalizing to the sorted-name convention.

    ``wins_a``/``wins_b`` are as the CALLER ordered them (wins of ``a`` / wins of ``b``); the
    stored entry always attributes ``wins_a`` to the lexicographically-lower name. ``n`` is
    derived (``wins_a + wins_b + draws``) rather than passed, so the counts can never desync.
    """
    if min(wins_a, wins_b, draws) < 0:
        raise ValueError(f"negative counts for {a} vs {b}: {wins_a}/{wins_b}/{draws}")
    lo, hi = sorted((a, b))
    e = store["pairs"].setdefault(
        pair_key(a, b), {"a": lo, "b": hi, "wins_a": 0, "wins_b": 0, "draws": 0, "n": 0})
    if a == lo:
        e["wins_a"] += wins_a
        e["wins_b"] += wins_b
    else:
        e["wins_a"] += wins_b
        e["wins_b"] += wins_a
    e["draws"] += draws
    e["n"] += wins_a + wins_b + draws
    return e


def merge_stores(base: dict, other: dict) -> dict:
    """Sum ``other``'s pair counts into ``base`` and concatenate history (dedup by ``chunk_id``,
    keeping the LATER record — a chunk that was committed repeatedly has one honest entry).

    Order-consistent because both sides use the same sorted-key / lower-name convention, so
    disjoint and overlapping pair sets both merge correctly (a fleet of processes, or a resume).
    """
    for k, e in (other.get("pairs") or {}).items():
        m = base["pairs"].setdefault(
            k, {"a": e["a"], "b": e["b"], "wins_a": 0, "wins_b": 0, "draws": 0, "n": 0})
        m["wins_a"] += e.get("wins_a", 0)
        m["wins_b"] += e.get("wins_b", 0)
        m["draws"] += e.get("draws", 0)
        m["n"] += e.get("n", 0)
    for rec in other.get("history") or []:
        _upsert_history(base, rec)
    return base


def _upsert_history(store: dict, record: dict) -> None:
    """Replace the history record with this ``chunk_id`` (or append). Lets an in-progress chunk
    rewrite its own record on every commit without piling up duplicates."""
    cid = record.get("chunk_id")
    hist = store.setdefault("history", [])
    for i, rec in enumerate(hist):
        if cid is not None and rec.get("chunk_id") == cid:
            hist[i] = record
            return
    hist.append(record)


def commit(path: str, deltas, *, target: int, bots: list[str] | None = None,
           git_hash: str | None = None, history: dict | None = None) -> dict:
    """Load-merge-write one visit's results. Returns the fresh (on-disk) store.

    ``deltas`` is an iterable of ``(a, b, wins_a, wins_b, draws)``. Re-reading the artifact on
    every commit (rather than holding it in memory) is what makes two concurrent accumulators
    SUM instead of clobber, and what bounds a crash's loss to the in-flight visit.
    """
    store = load(path, target=target, bots=bots, git_hash=git_hash)
    for (a, b, wa, wb, dr) in deltas:
        accumulate(store, a, b, wa, wb, dr)
    if history is not None:
        _upsert_history(store, history)
    save(path, store)
    return store


# ── Scheduling: balanced fill ────────────────────────────────────────────────


def all_pairs(bots) -> list[tuple[str, str]]:
    """Every unordered pair, in canonical (sorted-name) order."""
    names = sorted(bots)
    return list(itertools.combinations(names, 2))


def next_pair(store: dict, pairs, target: int):
    """The pair to play next: **lowest current n first**, ties broken by the sorted-name tuple
    (NOT ``pair_key`` — the ``|`` separator sorts after ``_``, so keys would order
    ``aggressive_v2|heuristic`` before ``aggressive|aggressive_v2``). ``None`` when every pair
    has reached ``target``."""
    under = [p for p in pairs if n_games(store, *p) < target]
    if not under:
        return None
    return min(under, key=lambda p: (n_games(store, *p), tuple(sorted(p))))


# ── Resolution math ──────────────────────────────────────────────────────────


def edge_stats(e: dict) -> dict:
    """Per-edge resolution: empirical p, its logit SE, and that SE in ELO points.

    p is over DECISIVE games (``wins_a + wins_b``) — a draw carries no information about which
    bot is stronger, so folding it in as half a win would shrink the reported error bar without
    having reduced the uncertainty. The SE uses the Haldane-Anscombe corrected
    ``p̂ = (wins_a + ½)/(decisive + 1)`` so a 0-of-N or N-of-N edge still reports a finite,
    honest bound instead of ``inf``; ``p`` itself is reported UNcorrected.

    ``se_logit = 1/sqrt(decisive·p·q)`` is the delta-method SE of the log-odds, and
    ``se_elo = (400/ln10)·se_logit`` converts it to the scale a rating is read on.
    """
    wa, wb, dr, n = e["wins_a"], e["wins_b"], e.get("draws", 0), e["n"]
    decisive = wa + wb
    p = (wa / decisive) if decisive else float("nan")
    if decisive <= 0:
        return {"n": n, "decisive": 0, "draws": dr, "p": p,
                "se_logit": float("inf"), "se_elo": float("inf")}
    p_adj = (wa + 0.5) / (decisive + 1.0)
    se_logit = 1.0 / math.sqrt(decisive * p_adj * (1.0 - p_adj))
    return {"n": n, "decisive": decisive, "draws": dr, "p": p,
            # A 0-of-N / N-of-N edge: its ELO gap is genuinely UNBOUNDED, so its wide SE is a
            # property of the matchup, not a shortage of games. Flagged so it can be kept out
            # of the "worst pair" reading, which is meant to track sample size.
            "saturated": wa == 0 or wb == 0,
            "se_logit": se_logit, "se_elo": ELO_PER_LOGIT * se_logit}


def summary_rows(store: dict, pairs) -> list[dict]:
    rows = []
    for (a, b) in pairs:
        e = entry(store, a, b)
        st = edge_stats(e)
        rows.append({"a": e["a"], "b": e["b"], **st})
    return rows


def format_summary(store: dict, pairs, target: int) -> str:
    """The table the operator reads: per-pair n / p / ±ELO, plus the headline resolution line."""
    rows = summary_rows(store, pairs)
    if not rows:
        return "(no pairs)"
    w = max(len(f"{r['a']} vs {r['b']}") for r in rows)
    out = [f"── Bot matchup matrix ── {len(rows)} pairs, target {target} games/pair ──",
           f"  {'pair'.ljust(w)}  {'n':>7} {'dec':>7} {'draws':>6} {'p':>7} {'±ELO':>8}"]
    for r in sorted(rows, key=lambda r: r["n"]):
        p = "   —   " if math.isnan(r["p"]) else f"{r['p']:7.4f}"
        se = "     ∞  " if math.isinf(r["se_elo"]) else f"{r['se_elo']:8.1f}"
        label = "{} vs {}".format(r["a"], r["b"])
        out.append(f"  {label.ljust(w)}  "
                   f"{r['n']:7d} {r['decisive']:7d} {r['draws']:6d} {p} {se}"
                   f"{'  *' if r['saturated'] else ''}")
    ns = [r["n"] for r in rows]
    total = sum(ns)
    pct = 100.0 * total / max(1, target * len(rows))
    worst = max(rows, key=lambda r: r["se_elo"])
    out.append(f"  min n {min(ns)}  max n {max(ns)}  total {total} games  "
               f"({pct:.2f}% of target)")
    se = "∞ (an unplayed pair)" if math.isinf(worst["se_elo"]) else f"{worst['se_elo']:.1f} ELO"
    out.append(f"resolution now ±{se} per edge (worst pair: "
               f"{worst['a']} vs {worst['b']}, n={worst['n']})")
    # The headline above is dominated by SATURATED edges (`*`), whose gap is unbounded no matter
    # how many games are played — so report the reading that actually tracks sample size too.
    live = [r for r in rows if not r["saturated"] and math.isfinite(r["se_elo"])]
    if live and len(live) < len(rows):
        w2 = max(live, key=lambda r: r["se_elo"])
        med = sorted(r["se_elo"] for r in live)[len(live) // 2]
        out.append(f"  ({len(rows) - len(live)} edge(s) marked * are SATURATED — one bot has "
                   f"never lost, so the gap is unbounded, not merely unmeasured)")
        out.append(f"  excluding those: worst ±{w2['se_elo']:.1f} ELO "
                   f"({w2['a']} vs {w2['b']}, n={w2['n']}), median ±{med:.1f} ELO")
    return "\n".join(out)


# ── The battle seam (the only part that touches the bridge) ──────────────────


def build_bots(names: list[str]) -> dict:
    """Construct the roster ONCE (bots are reused across pairs — building warms the data
    singletons at ~4.5 s each, which dominated cost when the calibration rebuilt per pair).

    Imports are deferred so importing this module stays free for the pure unit tests."""
    from agents.training.bot_elo_calibration import _build_bot
    from utils.team_loader import TeamLoader

    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    print(f"Building {len(names)} bots…", flush=True)
    # Tag namespace distinct from the calibration's "Cal{i}" so both can be live at once.
    return {name: _build_bot(name, all_teams, sample_teams, f"Mx{i}")
            for i, name in enumerate(names)}


def play_pair(bots: dict, a: str, b: str, n_battles: int, concurrency: int):
    """Play ``n_battles`` of one pair via the calibration's own chunk player.

    Returns ``(wins_a, wins_b, draws, finished)``. ``draws`` is derived from poke-env's own
    finished/won accounting, so an unfinished battle lands in NEITHER bucket (see PROTOCOL)."""
    from agents.training.bot_elo_calibration import _play_chunk

    wins_a, wins_b, finished = _play_chunk(bots[a], bots[b], n_battles, concurrency)
    draws = finished - wins_a - wins_b
    if draws < 0:  # would mean poke-env's own counters disagree; refuse to record nonsense
        raise ValueError(f"{a} vs {b}: wins {wins_a}+{wins_b} exceed finished {finished}")
    return wins_a, wins_b, draws, finished


def _safe_git_hash():
    try:
        from utils.git import get_git_hash
        return get_git_hash()
    except Exception:  # noqa: BLE001
        return None


# ── The chunk runner ─────────────────────────────────────────────────────────


def run_chunk(out_path: str, *, target: int = DEFAULT_TARGET,
              chunk_battles: int | None = None, max_minutes: float | None = None,
              concurrency: int = DEFAULT_CONCURRENCY,
              games_per_visit: int = DEFAULT_GAMES_PER_VISIT,
              bot_names: list[str] | None = None,
              build_bots_fn=build_bots, play_fn=play_pair,
              now=time.time, git_hash: str | None = None, verbose: bool = True) -> dict:
    """Accumulate one chunk into ``out_path`` and return the resulting store.

    Stops at whichever comes first: ``chunk_battles`` scheduled, ``max_minutes`` elapsed, or
    every pair at ``target``. Fully resumable — re-running just continues the fill.

    ``build_bots_fn`` / ``play_fn`` / ``now`` are the seams the unit tests replace; nothing else
    in this function touches the bridge.
    """
    if bot_names is None:
        from agents.training.eval_callback import eval_opponent_names
        bot_names = eval_opponent_names()
    names = sorted(bot_names)
    pairs = all_pairs(names)
    git_hash = git_hash or _safe_git_hash()

    store = load(out_path, target=target, bots=names, git_hash=git_hash)
    if next_pair(store, pairs, target) is None:
        if verbose:
            print(f"✅ every pair already has {target} games — nothing to do.", flush=True)
            print(format_summary(store, pairs, target), flush=True)
        return store

    bots = build_bots_fn(names)
    chunk_id = uuid.uuid4().hex[:12]
    t0 = now()
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = (t0 + max_minutes * 60.0) if max_minutes else None
    budget = chunk_battles if chunk_battles else None
    scheduled = added = 0
    touched: dict[str, int] = {}

    while True:
        if budget is not None and scheduled >= budget:
            if verbose:
                print(f"🎯 reached --chunk-battles ({budget}) — stopping cleanly.", flush=True)
            break
        if deadline is not None and now() >= deadline:
            if verbose:
                print(f"⏱  reached --max-minutes ({max_minutes:.1f}m) — stopping cleanly.",
                      flush=True)
            break
        pair = next_pair(store, pairs, target)
        if pair is None:
            if verbose:
                print(f"✅ every pair reached {target} games.", flush=True)
            break
        a, b = pair
        n = min(games_per_visit, target - n_games(store, a, b))
        if budget is not None:
            n = min(n, budget - scheduled)
        if n <= 0:
            break

        wa, wb, dr, finished = play_fn(bots, a, b, n, concurrency)
        # Budget on what was SCHEDULED, not what finished: a pair that keeps timing out must
        # not spin forever (it would never reach `target` either).
        scheduled += n
        added += finished
        touched[pair_key(a, b)] = touched.get(pair_key(a, b), 0) + finished
        elapsed = now() - t0
        record = {
            "chunk_id": chunk_id,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(elapsed, 1),
            "games_added": added,
            "games_scheduled": scheduled,
            "concurrency": concurrency,
            "games_per_visit": games_per_visit,
            "pairs_touched": len(touched),
            "git_hash": git_hash,
        }
        store = commit(out_path, [(a, b, wa, wb, dr)], target=target, bots=names,
                       git_hash=git_hash, history=record)
        if verbose:
            tot = n_games(store, a, b)
            min_n = min(n_games(store, x, y) for (x, y) in pairs)
            rate = added / elapsed if elapsed > 0 else 0.0
            print(f"  {a} vs {b}: +{finished} ({wa}-{wb}, {dr}d) → {tot}/{target}  "
                  f"[min pair {min_n}/{target}]  {added}g in {elapsed:.0f}s "
                  f"({rate:.2f} battles/s)", flush=True)

    elapsed = max(now() - t0, 1e-9)
    if verbose:
        print(f"\nchunk {chunk_id}: {added} battles in {elapsed:.0f}s "
              f"({added / elapsed:.2f} battles/s, concurrency {concurrency})", flush=True)
        print(format_summary(store, pairs, target), flush=True)
        remaining = sum(max(0, target - n_games(store, x, y)) for (x, y) in pairs)
        if remaining and added:
            print(f"projection: {remaining} games remain → "
                  f"{remaining / (added / elapsed) / 3600.0:.1f} wall-hours at this rate",
                  flush=True)
        print(f"wrote {out_path}", flush=True)
    return store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Incremental bot-vs-bot RAW matchup accumulator (bridge, no server). "
                    "Never writes data/gen3_bot_elo_anchors.json.")
    ap.add_argument("--target-per-pair", type=int, default=DEFAULT_TARGET,
                    help=f"accumulate until every pair has this many games "
                         f"(default {DEFAULT_TARGET})")
    ap.add_argument("--chunk-battles", type=int, default=None,
                    help="stop after scheduling this many battles this run")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop after this many minutes of PLAY, whichever limit hits first "
                         "(the one-time ~1 min bot build is on top, and the deadline is "
                         "checked between visits, so a chunk can overrun by one visit)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"concurrent bridge battles (default {DEFAULT_CONCURRENCY}; keep low "
                         f"beside a live training run — each battle is a subprocess)")
    ap.add_argument("--games-per-visit", type=int, default=DEFAULT_GAMES_PER_VISIT,
                    help=f"battles per visit to one pair between persists "
                         f"(default {DEFAULT_GAMES_PER_VISIT})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"artifact path, relative to the CWD (default {DEFAULT_OUT})")
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE,
                    help=f"niceness increment applied at entry (default {DEFAULT_NICE}; "
                         f"0 disables)")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the current table and exit — plays nothing")
    args = ap.parse_args(argv)

    if args.nice:
        try:
            os.nice(args.nice)
        except OSError as exc:  # noqa: BLE001 — a refused nice must not abort the job
            print(f"⚠️  could not nice({args.nice}): {exc}", file=sys.stderr)

    if args.summary_only:
        from agents.training.eval_callback import eval_opponent_names
        names = sorted(eval_opponent_names())
        store = load(args.out)
        print(format_summary(store, all_pairs(names), store.get("target_per_pair",
                                                               args.target_per_pair)))
        return 0

    if args.chunk_battles is None and args.max_minutes is None:
        print("note: neither --chunk-battles nor --max-minutes given — this runs until every "
              "pair reaches the target (Ctrl-C is safe; the artifact is committed per visit).",
              flush=True)
    run_chunk(args.out, target=args.target_per_pair, chunk_battles=args.chunk_battles,
              max_minutes=args.max_minutes, concurrency=args.concurrency,
              games_per_visit=args.games_per_visit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
