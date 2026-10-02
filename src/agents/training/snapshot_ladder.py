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

🚨 THE GAMES THAT SELECTED A SNAPSHOT DO NOT RATE IT (recipe v3, owner decision 2026-09-27).
Promotion is decided by the eval cycle's games vs the pool's sentinels (win rate ≥ the promotion
threshold). Reusing those same games as ladder edges — what recipe v2 did, tagged
``source: "eval_cycle"`` — rates a node on the very sample that selected it: a WINNER'S CURSE of
about +15..+40 Elo at n = 100 (analytic), one reason a new node reads high and drifts down. So on
each promotion the ladder plays a FRESH ``PROMOTION_BASELINE_GAMES`` (200) games against every
sentinel that cycle used (``source: "promotion_baseline"``), plus the usual ``--n-games`` (100)
against every other frozen snapshot. The eval games stay in ``eval_results.jsonl`` only, and
``fit_ladder`` ignores any ``eval_cycle`` row a v2 tree left in ``games.jsonl``
(``--backfill-fresh`` replaces them with fresh pairs).

Two columns. ``ratings`` is the bot-anchored headline, as before. ``ratings_relative`` is Elo
above a pinned frozen REFERENCE node (default: the ``untaught_meter_opponent_v14`` baseline when
the run's ladder holds it, else the first snapshot), fitted from the frozen-vs-frozen matrix ALONE
— no bot edges — so it does not drift with the bot-anchored level.

Non-transitivity caveat: if the frozen pool is non-transitive (rock-paper-scissors), NO scalar
Elo represents it faithfully, however densely measured — but the dense matrix at least lets
``fit_quality`` (mean/max |predicted − observed|) QUANTIFY the intransitivity, which the sparse
live fit cannot even see.

CLI:
  python -m agents.training.snapshot_ladder <run_dir> --backfill        # one-time back tax
  python -m agents.training.snapshot_ladder <run_dir> --promote <step>  # per-promotion update
  python -m agents.training.snapshot_ladder <run_dir> --backfill-fresh [--dry-run]
                                         # replace a v2 ladder's reused eval-cycle pairs
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


def load_games(run_dir: str, *, include_eval_cycle: bool = False
               ) -> dict[tuple[int, int], list[int]]:
    """Read games.jsonl → {(lo, hi): [wins_lo, games]}, summing duplicate lines (independent
    samples of the SAME frozen matrix pool across appends; adding them just tightens the edge).

    🚨 Rows tagged ``source: "eval_cycle"`` are SKIPPED unless ``include_eval_cycle`` — those are
    the promotion-deciding eval games a v2 tree reused as edges, and under recipe v3 the games
    that SELECTED a snapshot never RATE it (the winner's curse; see the module docstring). A pair
    whose only rows are eval-cycle rows therefore reads as UNMEASURED, which is what makes
    ``_measure_missing`` / ``--backfill-fresh`` play it fresh."""
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
                if not include_eval_cycle and r.get("source") == EVAL_CYCLE_SOURCE:
                    continue
                lo, hi = _pair_key(int(r["a"]), int(r["b"]))
                wins_lo = int(r["wins_a"]) if r["a"] == lo else int(r["games"]) - int(r["wins_a"])
                e = out.setdefault((lo, hi), [0, 0])
                e[0] += wins_lo
                e[1] += int(r["games"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def _append_game(run_dir: str, step_a: int, step_b: int, wins_a: int, games: int,
                 source: "str | None" = None) -> None:
    """Append one measured pair (race-safe: a single sub-PIPE_BUF line append is atomic).

    Every row carries ``recipe_version`` — the ``LADDER_FITTER_VERSION`` of the tree that PLAYED
    it (3 from 2026-09-27; rows without the key predate it). ``source`` is PROVENANCE, written
    only when it is not the default: an ordinary round-robin pair carries no ``source`` key, a
    fresh promotion baseline vs an eval sentinel carries ``"source": "promotion_baseline"``, and
    the ``"eval_cycle"`` rows a v2 tree wrote are never written again."""
    os.makedirs(_ladder_dir(run_dir), exist_ok=True)
    row = {"a": int(step_a), "b": int(step_b), "wins_a": int(wins_a), "games": int(games),
           "at": datetime.now(timezone.utc).isoformat(),
           "recipe_version": LADDER_FITTER_VERSION}
    if source:
        row["source"] = str(source)
    with open(games_log_path(run_dir), "a") as f:
        f.write(json.dumps(row) + "\n")


LADDER_SOURCE = "ladder"        #: a pair this module PLAYED (rows carry no `source` key)
#: a pair a v2 tree REUSED from an eval cycle. 🚨 NEVER a rating edge under v3 (winner's curse);
#: `load_games` skips these rows and `--backfill-fresh` replaces them.
EVAL_CYCLE_SOURCE = "eval_cycle"
#: a FRESH pair vs a sentinel the promoting eval cycle used, played at PROMOTION_BASELINE_GAMES.
PROMOTION_BASELINE_SOURCE = "promotion_baseline"
#: Games per fresh promotion-baseline pair (owner decision 2026-09-27). Double the round-robin's
#: 100, because these are the pairs the promotion decision read and the ones a reader looks at.
PROMOTION_BASELINE_GAMES = 200


def pair_sources(run_dir: str) -> dict[tuple[int, int], set[str]]:
    """{(lo, hi): {source, …}} over games.jsonl — a row with no ``source`` key counts as
    ``LADDER_SOURCE``. Read for the ladder dict's ``pairs_by_source`` provenance block; kept
    separate from :func:`load_games` so that function's arithmetic stays exactly what it was."""
    out: dict[tuple[int, int], set[str]] = {}
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
                key = _pair_key(int(r["a"]), int(r["b"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            out.setdefault(key, set()).add(str(r.get("source") or LADDER_SOURCE))
    return out


# ── what the eval cycles measured (READ ONLY — never a rating edge under v3) ──────────────────
def eval_measured_pairs(run_dir: str) -> dict[tuple[int, int], list[int]]:
    """{(lo, hi): [wins_lo, games]} for every frozen pair an eval cycle already played **under the
    ladder's own protocol** — rows whose ``sentinel_regime`` says BOTH ``greedy`` and
    ``symmetric_teams``. Rows without that stamp, or with either half false, are skipped.

    🚨 NOT A LADDER EDGE (recipe v3, 2026-09-27). Recipe v2 folded these into ``games.jsonl`` as
    ``source: "eval_cycle"`` to save ~500 battles a promotion; v3 does not, because these are the
    games that DECIDED the promotion, and rating a node on the sample that selected it is a
    winner's curse (+15..+40 Elo at n = 100). Kept as the read of what the eval MEASURED — the
    same-protocol comparator a winner's-curse check sets beside the fresh
    ``promotion_baseline`` edge — and as the far end of the writer→row contract
    (``elo_row_contract_test.py``).

    Why both regime halves: the asymmetric regime (greedy trainee vs a temperature-1.0 sentinel on
    the flat pool builder) is worth **+8.9 pp [+7.0, +10.7]** to the newer snapshot (2026-09-07,
    ``ai_v12_02_winprob_critic``, 60 pairs) — a different experiment, not a noisier one.

    Rows measuring the SAME pair are SUMMED (a frozen pair is stationary).
    """
    path = os.path.join(run_dir, "eval_results.jsonl")
    out: dict[tuple[int, int], list[int]] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            regime = r.get("sentinel_regime") or {}
            if not (regime.get("greedy") and regime.get("symmetric_teams")):
                continue
            # A MIRRORED-PAIR row (`gen3_mirrored_pairs_v1`) half-plays the trainee on the sentinel's
            # team draw — not the ladder's protocol either, so it is not a same-protocol comparator.
            if r.get("mirrored_pairs"):
                continue
            try:
                trainee = int(r["step"])
                n_default = int(r.get("n_games", 0))
            except (KeyError, TypeError, ValueError):
                continue
            for sent in (r.get("sentinels") or []):
                try:
                    other = int(sent["step"])
                    counts = sent.get("counts")
                    if counts:
                        wins, games = int(counts[0]), int(counts[1])
                    else:
                        # A row written without exact counts: fall back to the cycle's declared game
                        # count. Exact only at full shard coverage, which is why `counts` exists.
                        games = n_default
                        wins = int(round(float(sent["win_rate"]) * games))
                except (KeyError, TypeError, ValueError):
                    continue
                if games <= 0 or other == trainee:
                    continue
                lo, hi = _pair_key(trainee, other)
                wins_lo = wins if trainee == lo else games - wins
                e = out.setdefault((lo, hi), [0, 0])
                e[0] += wins_lo
                e[1] += games
    return out


def promotion_sentinel_steps(run_dir: str, step: int) -> list[int]:
    """The pool sentinels the eval cycle at ``step`` played — i.e. the pairs whose games DECIDED
    that step's promotion — read from ``eval_results.jsonl`` (every row at ``step``, any regime:
    the winner's curse is a property of selection, not of the protocol). Ascending, deduplicated.
    Empty when there is no row (a hand ``--promote``), in which case every pair is an ordinary
    round-robin pair."""
    path = os.path.join(run_dir, "eval_results.jsonl")
    out: set[int] = set()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if int(r.get("step", -1)) != int(step):
                    continue
                for sent in (r.get("sentinels") or []):
                    other = int(sent["step"])
                    if other != int(step):
                        out.add(other)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
    return sorted(out)


def eval_cycle_only_pairs(run_dir: str, steps: "list[int] | None" = None) -> list[tuple[int, int]]:
    """The frozen pairs among ``steps`` (default: the pool on disk) whose ONLY rows in
    ``games.jsonl`` are ``eval_cycle`` rows — what a v2 ladder reused and a v3 fit cannot see.
    This is exactly ``--backfill-fresh``'s work list; each costs ``PROMOTION_BASELINE_GAMES``."""
    keep = set(steps if steps is not None else pool_snapshot_steps(run_dir))
    return sorted(k for k, srcs in pair_sources(run_dir).items()
                  if srcs == {EVAL_CYCLE_SOURCE} and k[0] in keep and k[1] in keep)


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


#: Public alias — `main.elo refit --apply` writes the refitted ladder (and preserves the old one)
#: through the SAME atomic writer the fitter uses, so a half-written ladder.json is not a state
#: either path can produce.
atomic_write_json = _atomic_write_json

#: The name `main.elo refit --apply` preserves a PRE-RECIPE `ladder.json` under, beside the file it
#: replaces. Named here because both the writer and every refusal message quote it.
PRE_RECIPE_BACKUP_NAME = "ladder.pre_recipe.json"


def pre_recipe_backup_path(run_dir: str, committed: "dict | None" = None) -> str:
    """Where `main.elo refit --apply` keeps the committed file it replaces.

    An UNSTAMPED file (and the no-argument call) keeps ``ladder.pre_recipe.json``; a file stamped
    with an older fitter version N keeps ``ladder.pre_recipe_vN.json`` — so a run already
    converted once (v1 → v2, whose backup exists) can still be converted v2 → v3 without the
    refuse-rather-than-clobber rule blocking it, and each banked scale keeps its own copy."""
    r = (committed or {}).get("recipe")
    ver = r.get("fitter_version") if isinstance(r, dict) else None
    if isinstance(ver, int) and ver != LADDER_FITTER_VERSION:
        return os.path.join(_ladder_dir(run_dir), f"ladder.pre_recipe_v{ver}.json")
    return os.path.join(_ladder_dir(run_dir), PRE_RECIPE_BACKUP_NAME)


def _pairs_by_source_counts(run_dir: str, keep_keys: set) -> dict[str, int]:
    """{source: n_pairs} over the frozen pairs inside this fit's prefix. A pair carrying rows from
    both sources counts once per source — that is a real, if unusual, state (a hand `--backfill`
    after a reuse) and hiding it would be the wrong kind of tidy."""
    counts: dict[str, int] = {}
    for (lo, hi), srcs in pair_sources(run_dir).items():
        if elo_mod.snap_key(lo) in keep_keys and elo_mod.snap_key(hi) in keep_keys:
            for src in srcs:
                counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


# ── THE RECIPE STAMP ─────────────────────────────────────────────────────────────────────────
#
# 🚨 A LADDER RATING IS ONLY COMPARABLE TO ANOTHER FITTED BY THE SAME RECIPE. The fit is not a
# property of the games alone: which EDGES go into it is a decision, and changing that decision
# moves every rating without changing a single measured game. MEASURED 2026-09-14
# (`flywheel_armS_reads_2026-09-14/` §2.1): `ai_v12_02_winprob_critic`'s committed `ladder.json`
# (written 2026-09-08, 494 pairs, eval-sentinel edges FOLDED IN) reads 2057.3 at its newest node;
# the current recipe (`3e6875a5`, those edges dropped) refits the SAME 20 nodes to 1984.2 — a
# **+73.1 Elo** gap that FLIPPED THE SIGN of a cross-run delta. Nothing in the old file said so.
#
# So every fit now stamps the recipe it used, and a reader that cannot refit REFUSES rather than
# quoting a number on an unknown scale. The stamp is a BLOCK, not the bare presence of a count
# key: presence catches exactly one historical change, while a named recipe + a fitter version
# catches the NEXT one too.
#: The recipe's NAME. Change it only when the fit's identity changes in kind.
LADDER_RECIPE_NAME = "gen3_ladder_recipe_v1"

#: 🚨 BUMP THIS WHENEVER THE FIT CHANGES WHAT A RATING MEANS — a new edge family, a dropped one, a
#: different anchor set, a different BT solver contract. A bump makes every previously committed
#: `ladder.json` read as `differs`, which is the correct and loud outcome: those numbers are on
#: the old scale. Do NOT bump it for a change that cannot move a rating.
#:
#: 1 — the implicit pre-stamp recipe: dense frozen matrix + bot anchors + eval-cycle SENTINEL
#:     edges. Never written by any code; it is what an UNSTAMPED committed file was fitted with.
#: 2 — `3e6875a5` (2026-09-07): the eval-cycle sentinel edges are DROPPED. Worth +73.1 Elo on one
#:     20-node run and +21..+29 on the newest nodes generally.
#: 3 — 2026-09-27 (owner decision): the eval-cycle PAIR EDGES that v2 REUSED in `games.jsonl`
#:     (`source: "eval_cycle"`) are DROPPED too — the games that selected a snapshot must not rate
#:     it (a winner's curse of ~+15..+40 Elo at n = 100). Each promotion instead plays FRESH
#:     200-game pairs vs the eval's sentinels. Also adds the frozen-only `ratings_relative` column.
#:     A v2 run whose games.jsonl holds no eval_cycle rows refits to IDENTICAL ratings.
LADDER_FITTER_VERSION = 3


class LadderRecipeError(RuntimeError):
    """A committed ``ladder.json`` was fitted by a recipe this reader cannot quote.

    Raised INSTEAD of returning the number, because the failure mode is a number that looks
    perfectly ordinary and is on a different scale — there is no in-band way for a caller to
    notice. The message always names the refit command.
    """


def ladder_recipe(sentinel_edges_dropped: int, eval_cycle_pairs_dropped: int = 0) -> dict:
    """The stamp `fit_ladder` writes: what this fit IS, and which tree produced it."""
    from utils.git import get_git_hash
    try:
        commit = get_git_hash()
    except Exception:                            # noqa: BLE001 — a stamp must never fail a fit
        commit = ""
    return {
        "name": LADDER_RECIPE_NAME,
        "fitter_version": LADDER_FITTER_VERSION,
        # The POLICY (a boolean), beside the COUNT this fit actually dropped. The policy is what
        # makes two files comparable; the count is a property of the run's eval history and is 0
        # for a run that never measured a sentinel pair — which is why the count alone can never
        # be the stamp.
        "eval_sentinel_edges_dropped": True,
        "eval_sentinel_edges_dropped_count": int(sentinel_edges_dropped),
        # v3: the eval-cycle pair rows a v2 tree REUSED in games.jsonl are never a rating edge —
        # the POLICY (always False here) beside the COUNT of such pairs this fit ignored.
        "eval_cycle_pair_edges_used": False,
        "eval_cycle_pair_edges_dropped_count": int(eval_cycle_pairs_dropped),
        "promotion_baseline_games": PROMOTION_BASELINE_GAMES,
        "commit": commit,
    }


def recipe_status(ladder: dict) -> "tuple[str, str]":
    """``(status, detail)`` for a loaded ladder dict — ``current`` / ``absent`` / ``differs``.

    Pure: a dict in, a verdict out, so every reader asks the same question and a fixture can be
    tested with no run directory at all. ``absent`` covers a missing block AND a present-but-null
    one (the 2026-09-08 file records ``eval_sentinel_edges_dropped: null``, which a
    presence-check would have waved straight through).
    """
    r = ladder.get("recipe")
    if not isinstance(r, dict) or not r:
        # Pre-stamp. The one thing we can still say about such a file is whether the older,
        # weaker signal is there — a non-null top-level count means it was at least fitted by a
        # tree that already dropped the sentinel edges.
        count = ladder.get("eval_sentinel_edges_dropped")
        older = ("its top-level `eval_sentinel_edges_dropped` is null/absent, so it was fitted "
                 "WITH the eval-cycle sentinel edges (+21..+29 Elo on the newest nodes, +73.1 on "
                 "one measured 20-node run)" if count is None else
                 f"its top-level `eval_sentinel_edges_dropped` is {count}, so the sentinel edges "
                 "were already dropped — but with no recipe block nothing else about the fit is "
                 "pinned")
        return "absent", f"no `recipe` block ({older})"
    name, ver = r.get("name"), r.get("fitter_version")
    if name != LADDER_RECIPE_NAME:
        return "differs", (f"recipe name {name!r}, this tree fits {LADDER_RECIPE_NAME!r}")
    if ver != LADDER_FITTER_VERSION:
        return "differs", (f"fitter_version {ver!r}, this tree fits v{LADDER_FITTER_VERSION} "
                           f"(commit {r.get('commit') or 'unrecorded'})")
    return "current", (f"{LADDER_RECIPE_NAME} v{LADDER_FITTER_VERSION}"
                       + (f" @ {str(r.get('commit'))[:8]}" if r.get("commit") else ""))


def recipe_refusal(path: str, status: str, detail: str, run_dir: "str | None" = None,
                   *, what: str = "") -> str:
    """The refusal TEXT every reader emits, so the fix is worded once."""
    where = run_dir or os.path.dirname(os.path.dirname(os.path.abspath(path)))
    label = f"{what} " if what else ""
    return (
        f"{label}{path}: STALE LADDER RECIPE — {detail}.\n"
        f"A rating is only comparable to one fitted by the same recipe: measured 2026-09-14, a "
        f"pre-recipe file read +73.1 Elo above the current fit of the SAME 20 nodes and flipped "
        f"the sign of a cross-run delta. Refusing to quote it.\n"
        f"FIX: refit from the raw pair log, which is append-only and never stale —\n"
        f"    python -m agents.training.snapshot_ladder {where} --fit-only\n"
        f"        (refits IN PLACE — the committed numbers are overwritten)\n"
        f"    python -m main.elo refit --apply {where}\n"
        f"        (same fit, but the committed file is KEPT as "
        f"snapshot_ladder/{PRE_RECIPE_BACKUP_NAME}, and the node set stays the committed one so "
        f"every banked number has a successor to compare against)")


def check_recipe(ladder: dict, path: str, *, run_dir: "str | None" = None,
                 what: str = "") -> None:
    """Raise :class:`LadderRecipeError` unless ``ladder`` carries THIS tree's recipe."""
    status, detail = recipe_status(ladder)
    if status != "current":
        raise LadderRecipeError(recipe_refusal(path, status, detail, run_dir, what=what))


# ── the RELATIVE column: Elo above a pinned frozen reference node ─────────────────────────────
#: The registry baseline that is the default reference node when the run's ladder holds it —
#: N0 (`ai_v14_01_base`)'s 24M snapshot, the new lineage's untaught-meter fixed opponent.
DEFAULT_REFERENCE_BASELINE = "untaught_meter_opponent_v14"

#: Prior SD (Elo) of the relative fit. Deliberately WEAK: the prior exists only to keep a 100-0
#: pair finite, and a strong prior centred on the reference would pull every node toward it by an
#: amount that changes whenever a node is added — the very drift this column exists to avoid.
RELATIVE_PRIOR_SD = 2000.0


def _baseline_reference_step(run_dir: str, steps: list[int]) -> "tuple[int | None, str]":
    """The step of :data:`DEFAULT_REFERENCE_BASELINE` in THIS run's ladder, or ``(None, why)``.

    "In the ladder" means the SAME FROZEN FILE, not the same step number: the node is at the
    baseline's step AND either this is the baseline's own run or the snapshot zip's sha256
    matches the registry's (a fork that seeded its pool with the parent's snapshot)."""
    try:
        from agents.training import baselines
        b = baselines.get(DEFAULT_REFERENCE_BASELINE)
    except Exception as e:                        # noqa: BLE001 — a reference must never fail a fit
        return None, f"baseline {DEFAULT_REFERENCE_BASELINE!r} unreadable ({type(e).__name__})"
    step = b.num_timesteps
    if step is None:
        try:
            step = int(os.path.basename(b.checkpoint).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return None, f"baseline {DEFAULT_REFERENCE_BASELINE!r} names no step"
    if step not in steps:
        return None, f"{DEFAULT_REFERENCE_BASELINE} ({b.run} @ {step:,}) is not a node here"
    if os.path.basename(os.path.normpath(run_dir)) == b.run:
        return step, f"baseline {DEFAULT_REFERENCE_BASELINE} ({b.run} @ {step:,})"
    zp = _snapshot_zip(run_dir, step)
    try:
        if os.path.isfile(zp) and baselines.sha256_file(zp) == b.sha256:
            return step, (f"baseline {DEFAULT_REFERENCE_BASELINE} ({b.run} @ {step:,}; "
                          f"same file by sha256)")
    except OSError:
        pass
    return None, (f"step {step:,} here is not {DEFAULT_REFERENCE_BASELINE}'s file "
                  f"(different run, sha256 differs or zip gone)")


def resolve_reference(run_dir: str, steps: list[int],
                      reference: "int | None" = None) -> "tuple[int | None, str]":
    """``(reference_step, reason)`` for the relative column. An explicit ``reference`` wins (and
    must be a node); else the registry baseline when this ladder holds it; else the FIRST
    snapshot. ``(None, …)`` only for an empty ladder."""
    steps = sorted(steps)
    if reference is not None:
        if int(reference) in steps:
            return int(reference), f"explicit step {int(reference):,}"
        return None, f"explicit reference {int(reference):,} is not a node of this fit"
    if not steps:
        return None, "no snapshot in the fit"
    ref, why = _baseline_reference_step(run_dir, steps)
    if ref is not None:
        return ref, why
    return steps[0], f"first snapshot (default baseline not in this ladder: {why})"


def _component(edges, start: str) -> set[str]:
    """The players reachable from ``start`` over ``edges`` ([(a, b, w, g)]) — a node the reference
    cannot reach has no relative rating at all, only a prior."""
    adj: dict[str, set[str]] = {}
    for a, b, _w, g in edges:
        if g > 0:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    seen, todo = {start}, [start]
    while todo:
        for nb in adj.get(todo.pop(), ()):
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return seen


def fit_relative(frozen_edges, reference_step: int) -> "tuple[dict[str, float], dict[str, float]]":
    """Elo ABOVE the reference node, from frozen-vs-frozen edges ONLY (no bot edges, no eval
    edges): ``({step: elo_minus_ref}, {step: se})``, the reference itself at exactly 0.

    Why this does not drift the way the bot-anchored level does: a frozen pair is a stationary
    Bernoulli measured once, and nothing else enters this fit — so adding a node (however strong)
    only adds edges touching that node, and the existing nodes' differences move only by what the
    new edges genuinely say about them (nothing, when the matrix is transitive). The bot-anchored
    column, by contrast, re-solves against bot edges whose newest nodes' eval rows keep arriving.
    Nodes not connected to the reference by frozen edges are OMITTED."""
    ref_key = elo_mod.snap_key(reference_step)
    comp = _component(frozen_edges, ref_key)
    edges = [e for e in frozen_edges if e[0] in comp and e[1] in comp and e[3] > 0]
    if not edges:
        return {str(reference_step): 0.0}, {str(reference_step): 0.0}
    ratings, se, _conv = elo_mod.fit_pairwise(edges, pinned={ref_key: 0.0}, base=0.0,
                                              prior_sd=RELATIVE_PRIOR_SD)
    rel = {str(elo_mod.snapshot_step(k)): round(v, 1) for k, v in ratings.items()}
    rse = {str(elo_mod.snapshot_step(k)): round(se.get(k, 0.0), 1) for k in ratings}
    return rel, rse


# ── the fit (dense matrix + bot anchors) ────────────────────────────────────────────────────
def fit_ladder(run_dir: str, base: float | None = None, *,
               first_n: int | None = None, write: bool = True,
               steps: list[int] | None = None, reference: int | None = None) -> dict:
    """Fit the anchored BT ladder from the DENSE frozen matrix + each snapshot's historical
    bot edges (from eval_results.jsonl, which connect the ladder to the pinned bots for the
    absolute scale). Returns the ladder dict (also written to ladder.json when ``write``).

    The eval cycles' SENTINEL edges (`snap:` vs `snap:`) are EXCLUDED — they measure the same
    frozen pair as the dense matrix but under a different protocol (greedy trainee vs stochastic
    sentinel, asymmetric teambuilder), worth +8.9 pp to the newer snapshot and +21..+29 Elo on the
    newest nodes. The count is returned as ``eval_sentinel_edges_dropped``; see source (2) below.
    Recipe v3 also ignores every ``games.jsonl`` row tagged ``source: "eval_cycle"`` (the
    promotion-deciding games a v2 tree reused) — see :func:`load_games`.

    Two rating columns: ``ratings`` (bot-anchored — THE headline) and ``ratings_relative`` (Elo
    above ``reference.step``, frozen edges only — :func:`fit_relative`). ``reference`` pins the
    reference step; the default is :func:`resolve_reference`'s.

    ``first_n`` restricts the fit to the run's FIRST ``first_n`` snapshots — every frozen pair
    and every bot edge whose snapshot endpoints all lie in that prefix — and never
    writes ``ladder.json``. That is what "matched SNAPSHOT COUNT" means for a cross-run
    comparison: BT re-solves every node on every add and the NEWEST node of a fit is
    systematically inflated (gen-10's 12M fell 2089 → 2021 over 12 refits), so the n-th node of
    a run's FINAL 12-node fit is not the same object as the n-th node of a 4-node fit. Measured
    2026-09-07 on `ai_v9_29_rev1_0823`: its 8M node reads **2052** in a first-4 fit and **1958**
    in the final 12-node fit — 94 Elo of newest-node deflation, which `main.critic_gate` was
    silently handing to the arm it compared against."""
    anchors = elo_mod.load_bot_anchors()
    pins = (anchors or {}).get("ratings")
    base = base if base is not None else (anchors or {}).get("base", elo_mod.DEFAULT_BASE)

    # ``steps`` lets a caller re-slice a COMMITTED ladder whose pool has since been groomed
    # (`main.critic_gate` passes the ladder's own rated steps); the default is the pool on disk.
    steps = sorted(steps) if steps is not None else pool_snapshot_steps(run_dir)
    if first_n is not None:
        if first_n < 1:
            raise ValueError(f"first_n must be >= 1, got {first_n}")
        steps = steps[:first_n]
        write = False
    keep_keys = {elo_mod.snap_key(s) for s in steps}

    def _kept(*names: str) -> bool:
        # every SNAPSHOT endpoint must lie in the prefix; a bot endpoint is always kept
        return all(n in keep_keys for n in names if not n.startswith("bot:"))

    results: list[tuple[str, str, int, int]] = []
    # (1) DENSE frozen-vs-frozen edges — the resolution. `load_games` skips v2's reused
    # `eval_cycle` rows (recipe v3); the count of kept pairs that carried ONLY such rows is stamped.
    games = load_games(run_dir)
    eval_cycle_pairs_dropped = len(eval_cycle_only_pairs(run_dir, steps))
    for (lo, hi), (wins_lo, g) in games.items():
        if g > 0 and _kept(elo_mod.snap_key(lo), elo_mod.snap_key(hi)):
            results.append((elo_mod.snap_key(lo), elo_mod.snap_key(hi), wins_lo, g))
    frozen_edges = list(results)
    # (2) each snapshot's historical BOT edges — the anchor connection, and ONLY that.
    #
    # 🚨 THE EVAL CYCLES' SENTINEL EDGES (`snap:` vs `snap:`) ARE DROPPED HERE, deliberately.
    # `elo._rows_to_results` yields both families off an eval row: trainee-vs-bot (`bot:`) and
    # trainee-vs-sentinel (`snap:`). The sentinel edges are a DIFFERENT MEASUREMENT of the same
    # frozen pair the dense matrix above already measures: an eval cycle plays the GREEDY trainee
    # against a STOCHASTIC sentinel (`eval_worker`: `stochastic=not sentinel_greedy`,
    # `temperature=self_play_temp`) with an ASYMMETRIC teambuilder (the trainee gets the
    # sample-team bias, the sentinel does not), while `_play_pair` above plays greedy-vs-greedy
    # with the SAME biased builder on both sides. Measured 2026-09-07 on
    # `ai_v12_02_winprob_critic` over the 60 pairs both sources cover: the eval edge favours the
    # NEWER snapshot by **+8.9 pp [+7.0, +10.7]** against the ladder's own edge for the same pair
    # — systematic, not noise. Folding both in inflated `ladder.json`'s newest nodes by +21 to +29
    # Elo, compounding exactly the newest-node inflation the matched-count rule exists to control.
    # So: `games.jsonl` is the ONLY snapshot-vs-snapshot source, and the eval rows contribute the
    # bot anchor alone. (The per-cycle `eval/elo` star fit — `elo.fit_from_run` — is UNCHANGED and
    # still uses the sentinel edges; it has no dense matrix to prefer, and its own caveat is
    # documented at `elo.py`'s greedy-vs-stochastic note.)
    sentinel_edges_dropped = 0
    try:
        for na, nb, wa, g in elo_mod._rows_to_results(elo_mod.load_rows(run_dir, source="log")):
            if g <= 0 or not _kept(na, nb):
                continue
            if elo_mod.is_snapshot(na) and elo_mod.is_snapshot(nb):
                sentinel_edges_dropped += 1
                continue
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

    snap_ratings = {str(s): round(ratings[elo_mod.snap_key(s)], 1)
                    for s in steps if elo_mod.snap_key(s) in ratings}
    snap_se = {str(s): round(se.get(elo_mod.snap_key(s), 0.0), 1)
               for s in steps if elo_mod.snap_key(s) in ratings}
    # The RELATIVE column — Elo above a pinned frozen reference node, frozen edges only.
    rated_steps = [s for s in steps if elo_mod.snap_key(s) in ratings]
    ref_step, ref_why = resolve_reference(run_dir, rated_steps or steps, reference)
    rel, rel_se = (fit_relative(frozen_edges, ref_step) if ref_step is not None else ({}, {}))
    ladder = {
        "version": 1,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "anchored_to_bots": bool(pins),
        "converged": converged,
        "n_frozen_pairs_measured": sum(1 for (lo, hi), v in games.items()
                                       if v[1] > 0 and _kept(elo_mod.snap_key(lo),
                                                              elo_mod.snap_key(hi))),
        "first_n": first_n,
        # How many eval-cycle sentinel (`snap:` vs `snap:`) edges were EXCLUDED from this fit —
        # see the comment at source (2). A ladder written before 2026-09-07 has no such key and
        # was fit WITH them (worth +21..+29 Elo on its newest nodes).
        "eval_sentinel_edges_dropped": sentinel_edges_dropped,
        # 🚨 THE RECIPE STAMP — what this fit IS, so a later reader can tell whether its number is
        # on the same scale as one fitted today. See the RECIPE STAMP block above; a file without
        # it is pre-2026-09-22 and `recipe_status` reads `absent`.
        "recipe": ladder_recipe(sentinel_edges_dropped, eval_cycle_pairs_dropped),
        # PROVENANCE of games.jsonl's rows among the kept frozen pairs, per source: `ladder`
        # (round-robin), `promotion_baseline` (fresh 200-game pairs vs the eval's sentinels, v3)
        # and `eval_cycle` (a v2 tree's reuse — LISTED here, but NOT an edge of this fit).
        "pairs_by_source": _pairs_by_source_counts(run_dir, keep_keys),
        "n_pairs_possible": len(list(itertools.combinations(steps, 2))),
        "ratings": snap_ratings,
        "se": snap_se,
        # The SECOND column: Elo above `reference.step`, fitted from the frozen-vs-frozen matrix
        # ALONE (no bot edges), reference pinned at 0. Frozen pairs never change, so this column
        # does not drift with the bot-anchored level. `ratings` stays the headline.
        "reference": {"step": ref_step, "reason": ref_why,
                      "method": "frozen-vs-frozen edges only, BT, reference pinned at 0, "
                                f"prior_sd {RELATIVE_PRIOR_SD:.0f}"},
        "ratings_relative": rel,
        "se_relative": rel_se,
        "fit_quality": fit_quality,
    }
    if write:
        _atomic_write_json(ladder_json_path(run_dir), ladder)
    return ladder


# ── playing a frozen pair (bridge, no server) ───────────────────────────────────────────────
def _play_pair(run_dir, step_a, step_b, n_games, mappings, cv, all_teams, sample_teams,
               concurrency, impl, compile_extractor=True):
    """Round-robin one frozen pair on the bridge; return (wins_a, games_finished).

    ``compile_extractor`` defaults ON here, unlike training where it is an explicit flag: this is an
    OFFLINE tool, nothing is racing it for CPU, both players are frozen no-grad models, and the
    one-time compile is repaid within the first rung of a 100-game ladder. The helper self-validates
    and reverts if the compile does not actually pay, so the default cannot make this slower."""
    import torch
    torch.set_num_threads(1)  # defensive: B=1 CPU inference; the parallelism is across shards
    from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
    from agents.inference.player import RLPlayer
    from agents.model.compile_opponents import maybe_compile_extractor
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
        # Pure frozen CPU inference over many games — exactly the shape --compile-opponents targets.
        # On by default here (unlike training) because this is an offline analysis tool: nothing is
        # racing it, and the ~10-20s compile is repaid within the first ladder rung.
        maybe_compile_extractor(model, compile_extractor, label=f"ladder:{step}", hide_cuda=True)
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


def missing_pairs(run_dir, target_pairs) -> list[tuple[int, int]]:
    """The pairs in ``target_pairs`` with no rating-edge row yet — what ``_measure_missing`` would
    play. An ``eval_cycle``-only pair counts as MISSING (recipe v3; :func:`load_games`)."""
    have = load_games(run_dir)
    return [(a, b) for (a, b) in target_pairs
            if _pair_key(a, b) not in have or have[_pair_key(a, b)][1] == 0]


def _measure_missing(run_dir, target_pairs, n_games, concurrency, impl, *,
                     source: "str | None" = None):
    """Play every (a, b) in target_pairs NOT already in games.jsonl; append each (tagged
    ``source`` when given). Returns the count of pairs played."""
    from agents.observation.state_encoder import load_mappings
    from agents.model.snapshot import current_model_version
    from utils.team_loader import TeamLoader

    todo = missing_pairs(run_dir, target_pairs)
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
            _append_game(run_dir, a, b, wins_a, finished, source=source)
            played += 1
            tag = f" [{source}]" if source else ""
            print(f"[ladder] {a//1_000_000}M vs {b//1_000_000}M: {wins_a}/{finished}{tag}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad pair must not abort the sweep
            import traceback
            print(f"[ladder] pair {a} vs {b} FAILED: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()}", flush=True)
    return played


def promotion_plan(run_dir, new_step, n_games=100,
                   baseline_games=PROMOTION_BASELINE_GAMES) -> dict:
    """What a promotion at ``new_step`` will PLAY, without playing it (recipe v3).

    ``baseline``: the new snapshot vs each sentinel the promoting eval cycle used that is still in
    the pool — FRESH pairs at ``baseline_games`` (200), tagged ``promotion_baseline``; the eval's
    own games are NOT reused. ``round_robin``: the new snapshot vs every other frozen snapshot, at
    ``n_games`` (100). Each list holds only pairs not yet measured (never replayed).
    ``fresh_games`` is the cost line's number."""
    others = [s for s in pool_snapshot_steps(run_dir) if s != new_step]
    sentinels = set(promotion_sentinel_steps(run_dir, new_step))
    base_targets = [(new_step, o) for o in others if o in sentinels]
    rr_targets = [(new_step, o) for o in others if o not in sentinels]
    baseline = missing_pairs(run_dir, base_targets)
    rr = missing_pairs(run_dir, rr_targets)
    return {"step": new_step, "n_pairs": len(others),
            "sentinels": sorted(sentinels),
            "sentinels_not_in_pool": sorted(sentinels - set(others)),
            "baseline": baseline, "baseline_games": int(baseline_games),
            "round_robin": rr, "round_robin_games": int(n_games),
            "already_measured": len(others) - len(baseline) - len(rr),
            "fresh_games": len(baseline) * int(baseline_games) + len(rr) * int(n_games)}


def update_for_promotion(run_dir, new_step, n_games=100, concurrency=4, impl="node",
                         baseline_games=PROMOTION_BASELINE_GAMES) -> dict:
    """The per-promotion tax (recipe v3): play the newly-promoted frozen snapshot vs every OTHER
    frozen snapshot on disk (skipping already-measured pairs), append, refit.

    🚨 The pairs vs the promoting eval cycle's SENTINELS are played FRESH at ``baseline_games``
    (200) and tagged ``promotion_baseline``; the eval's own games — the ones that DECIDED the
    promotion — are never reused as edges (the winner's curse; module docstring). Every other pair
    is the ordinary ``n_games`` round-robin. Prints the fresh-game cost line."""
    plan = promotion_plan(run_dir, new_step, n_games, baseline_games)
    print(f"[ladder] promotion @{new_step}: {plan['n_pairs']} pairs — "
          f"{len(plan['baseline'])} FRESH baseline vs the eval's sentinels @{baseline_games} "
          f"(eval games NOT reused), {len(plan['round_robin'])} round-robin @{n_games}, "
          f"{plan['already_measured']} already measured | FRESH GAMES THIS PROMOTION: "
          f"{plan['fresh_games']:,}", flush=True)
    if plan["sentinels_not_in_pool"]:
        print(f"[ladder] note: eval sentinel(s) no longer in the pool, not played: "
              f"{plan['sentinels_not_in_pool']}", flush=True)
    if not plan["sentinels"]:
        print(f"[ladder] note: no eval_results.jsonl row at step {new_step} names sentinels — "
              f"every pair is an ordinary round-robin pair", flush=True)
    played_b = _measure_missing(run_dir, plan["baseline"], baseline_games, concurrency, impl,
                                source=PROMOTION_BASELINE_SOURCE)
    played_r = _measure_missing(run_dir, plan["round_robin"], n_games, concurrency, impl)
    print(f"[ladder] promotion @{new_step}: played {played_b} baseline + {played_r} round-robin "
          f"pair(s)", flush=True)
    return fit_ladder(run_dir)


def backfill_fresh(run_dir, n_games=PROMOTION_BASELINE_GAMES, concurrency=4, impl="node",
                   dry_run=False) -> dict:
    """Replace a v2 ladder's reused ``eval_cycle`` pairs with FRESH ``promotion_baseline`` pairs
    (``n_games`` each, default 200), then refit. Only pairs among the pool on disk whose ONLY rows
    are eval-cycle rows are played (:func:`eval_cycle_only_pairs`), so it is idempotent. The old
    rows are never deleted — ``games.jsonl`` is append-only and ``load_games`` ignores them.
    ``dry_run`` prints the work list and cost and plays nothing (returns ``{}``)."""
    pairs = eval_cycle_only_pairs(run_dir)
    print(f"[ladder] backfill-fresh: {len(pairs)} eval-cycle-only pair(s) in the pool → "
          f"{len(pairs) * n_games:,} fresh games @{n_games}", flush=True)
    if dry_run:
        for a, b in pairs:
            print(f"  {a:>12,} vs {b:>12,}")
        return {}
    _measure_missing(run_dir, pairs, n_games, concurrency, impl, source=PROMOTION_BASELINE_SOURCE)
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
    # TRIPWIRE: this mode is idempotent BY DESIGN — _measure_missing filters to pairs not yet in
    # games.jsonl. On a COMPLETE ladder it therefore plays nothing and returns a healthy-looking
    # refit, which reads exactly like "the games were added". It is not: a variance-reduction
    # tie-break needs MORE games on pairs that already have some, and `load_games` sums duplicate
    # lines by design. Say so out loud rather than let a no-op be mistaken for a measurement.
    measured = _measure_missing(run_dir, pairs, n_games, concurrency, impl)
    if measured == 0 and pairs:
        print(f"[ladder] ⚠️  0 of {len(pairs)} pairs were missing — this mode measured NOTHING and "
              f"CANNOT add variance-reduction games to an already-complete ladder. To tighten a "
              f"contrast, append duplicate rows for existing pairs (load_games SUMS them); "
              f"--backfill will not do it.", flush=True)
    return fit_ladder(run_dir)


def latest_promoted_elo(run_dir: str) -> "tuple[int, float, float] | None":
    """(step, elo, se) of the highest-step snapshot in the ladder sidecar, or None. Read by the
    live eval callback to surface eval/ladder_elo without recomputing.

    Deliberately does NOT check the recipe stamp. This reads the run's OWN file, written moments
    ago by the same process tree, to emit a WITHIN-RUN trend scalar — the recipe is whatever that
    run is pinned to, by construction, and refusing here would stop a live run from logging its
    own curve. Everything that compares ACROSS runs (`main.critic_gate`, `--exploiter-ladder
    auto`, `main.ops.plateau_signal`) checks it, because that is where a scale mismatch turns
    into a wrong verdict.
    """
    try:
        d = json.load(open(ladder_json_path(run_dir)))
        ratings = d.get("ratings") or {}
        if not ratings:
            return None
        step = max(int(k) for k in ratings)
        return step, float(ratings[str(step)]), float((d.get("se") or {}).get(str(step), 0.0))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def print_ladder_table(ladder: dict) -> None:
    """The ranked table: the bot-anchored headline column, then Elo above the reference node."""
    ref = (ladder.get("reference") or {})
    rel = ladder.get("ratings_relative") or {}
    rel_se = ladder.get("se_relative") or {}
    ref_step = ref.get("step")
    ref_lbl = f"vs {int(ref_step) / 1e6:g}M" if ref_step is not None else "vs ref"
    print(f"[ladder] relative column: Elo above {ref_step if ref_step is None else f'{ref_step:,}'}"
          f" — {ref.get('reason', 'no reference')} (frozen edges only; does not drift with the "
          f"bot anchor)")
    print(f"  {'step':>5}  {'anchored (headline)':>21}  {ref_lbl:>18}")
    ranked = sorted(ladder["ratings"].items(), key=lambda kv: -kv[1])
    for step, elo in ranked:
        r = rel.get(step)
        rtxt = (f"{r:+7.1f} ± {elo_mod.ci95(rel_se.get(step, 0.0)):5.1f}" if r is not None
                else f"{'—':>15}")
        print(f"  {int(step)//1_000_000:4d}M  {elo:7.1f} ± {elo_mod.ci95(ladder['se'].get(step, 0.0)):5.1f}"
              f"        {rtxt}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen-snapshot ELO ladder (dense, pay-once).")
    ap.add_argument("run_dir")
    ap.add_argument("--backfill", action="store_true", help="round-robin the whole current pool")
    ap.add_argument("--backfill-fresh", action="store_true",
                    help="replace a v2 ladder's reused eval-cycle pairs with FRESH "
                         "promotion-baseline pairs (@--baseline-games), then refit")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --backfill-fresh or --promote: print the work list + fresh-game "
                         "cost and play nothing")
    ap.add_argument("--promote", type=int, default=None, help="update for one promoted step")
    ap.add_argument("--n-games", type=int, default=100)
    ap.add_argument("--baseline-games", type=int, default=PROMOTION_BASELINE_GAMES,
                    help="games per FRESH pair vs the promoting eval's sentinels (default 200)")
    ap.add_argument("--reference", type=int, default=None,
                    help="step of the relative column's reference node (default: the "
                         f"{DEFAULT_REFERENCE_BASELINE} baseline when present, else the first)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--impl", default="node")
    ap.add_argument("--shard", default=None, help="I:N — play only pair-slice I of N (parallel workers)")
    ap.add_argument("--fit-only", action="store_true", help="refit from games.jsonl, play nothing")
    a = ap.parse_args()
    if a.fit_only:
        ladder = fit_ladder(a.run_dir, reference=a.reference)
    elif a.backfill_fresh:
        ladder = backfill_fresh(a.run_dir, a.baseline_games, a.concurrency, a.impl,
                                dry_run=a.dry_run)
        if not ladder:
            return 0
    elif a.promote is not None:
        if a.dry_run:
            plan = promotion_plan(a.run_dir, a.promote, a.n_games, a.baseline_games)
            print(json.dumps(plan, indent=2))
            return 0
        ladder = update_for_promotion(a.run_dir, a.promote, a.n_games, a.concurrency, a.impl,
                                      baseline_games=a.baseline_games)
    elif a.backfill:
        shard = tuple(int(x) for x in a.shard.split(":")) if a.shard else None
        ladder = backfill(a.run_dir, a.n_games, a.concurrency, a.impl, shard=shard)
        if not ladder:  # sharded worker — no fit, no ladder table to print
            print(f"[ladder] shard {a.shard} done (fit deferred to --fit-only)", flush=True)
            return 0
    else:
        ap.error("pass --backfill, --backfill-fresh, --promote <step>, or --fit-only")
    print(f"\n[ladder] {ladder['n_frozen_pairs_measured']}/{ladder['n_pairs_possible']} pairs | "
          f"non-transitivity mean|err| {ladder['fit_quality']['mean_abs_err']:.3f}")
    # The recipe, printed beside the numbers it produced — a rating quoted without it is a rating
    # on an unstated scale, which is how a +73.1 Elo recipe gap went unnoticed for six days.
    print(f"[ladder] recipe: {recipe_status(ladder)[1]}")
    print_ladder_table(ladder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
