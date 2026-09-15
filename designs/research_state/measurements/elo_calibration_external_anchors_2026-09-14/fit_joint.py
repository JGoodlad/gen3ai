#!/usr/bin/env python3
"""THE JOINT FIT — one Bradley-Terry solve over bots, our snapshots, and the external anchors.

    python3 fit_joint.py --cells <campaign out dir> --out <dir>

Three arms over the SAME node set and the SAME internal edges, so the anchor effect is isolated
from the pooling effect (a cross-run joint fit moves ratings all by itself, and attributing that
movement to the anchors would be the easiest mistake available here):

    JOINT-NOEXT        bots PINNED · internal edges only
    JOINT-EXT          bots PINNED · internal edges + the campaign's external edges
    JOINT-EXT-FREEBOT  only `random` pinned · the other eight bots FREE · all edges

Nodes are keyed the ladder's own way so they join an existing fit with no translation table, with
one change forced by fitting four runs at once: a snapshot key is RUN-QUALIFIED
(``snap:<run>@<step>``), because two runs both have an 8,000,016 and a bare step would silently
weld them into one player.

Edges, and where each comes from:

  (a) BOT edges      each run's ``eval_results.jsonl``, via ``elo._rows_to_results`` — the same
                     source ``snapshot_ladder.fit_ladder`` uses, and the ONLY thing it takes from
                     the eval rows. The eval-cycle SENTINEL edges (`snap:` vs `snap:`) are DROPPED
                     here exactly as the current ladder recipe drops them: they measure the same
                     frozen pair as (b) under a different protocol, worth +8.9 pp to the newer
                     snapshot (UNDERSTANDING rule 24's recipe).
  (b) DENSE edges    each run's ``snapshot_ladder/games.jsonl``, WITHIN a run only. No cross-run
                     frozen pair has ever been played, so there is no cross-run internal edge and
                     the four runs are connected ONLY through the pinned bots — which is precisely
                     the weakness the external anchors are here to test.
  (c) EXTERNAL       this campaign's cells: 9 bots x 2 anchors and 10 snapshots x 2 anchors, each
                     100 games, greedy-vs-greedy on the home team set.

🚨 Nothing here writes to ``data/``, to ``designs/ai_v5/elo_calibration/``, or to any run's
``ladder.json``. Adopting the anchors into the production fit is a PROPOSAL, not a side effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cells as cells_mod  # noqa: E402

from agents.training import elo as elo_mod           # noqa: E402
from agents.training import snapshot_ladder as sl    # noqa: E402

RUNS = ["ai_v9_29_rev1_0823", "ai_v12_11_ladder_ctrl10M",
        "ai_v12_02_winprob_critic", "ai_v13_01_flywheel_shaped"]


def run_dir(run: str) -> str:
    return f"{cells_mod.MODELS_ROOT}/{run}"


# ────────────────────────────────────────────────────────────────── the edges
def internal_edges(runs=RUNS) -> "tuple[list, dict]":
    """``(edges, nodes_by_run)`` — (a) bot edges + (b) dense frozen edges, run-qualified."""
    edges: list = []
    nodes_by_run: dict = {}
    for run in runs:
        d = run_dir(run)
        steps = sl.pool_snapshot_steps(d)
        nodes_by_run[run] = steps

        # Exactly `fit_ladder`'s node set: the snapshots ON DISK. A run's `eval_results.jsonl`
        # carries bot rows for every cycle, including steps whose snapshot was never promoted or
        # has since been groomed out of the pool; keeping them would add nodes the production fit
        # does not have. (With the bots PINNED such a node is provably inert — it touches only
        # fixed players — but in the FREEBOT arm it is not, and an arm-dependent node set is the
        # kind of difference that shows up later as an unexplained delta.)
        pool = set(steps)

        def key(step: int, _run=run) -> str:
            return f"snap:{_run}@{int(step)}"

        for (lo, hi), (wins_lo, g) in sl.load_games(d).items():
            if g > 0 and lo in pool and hi in pool:
                edges.append((key(lo), key(hi), wins_lo, g, "dense", run))
        for na, nb, wa, g in elo_mod._rows_to_results(elo_mod.load_rows(d, source="log")):
            if g <= 0:
                continue
            if elo_mod.is_snapshot(na) and elo_mod.is_snapshot(nb):
                continue                       # the eval SENTINEL edge — dropped, see the docstring
            # exactly one endpoint is a bot; the other is this run's snapshot
            snap = na if elo_mod.is_snapshot(na) else nb
            if elo_mod.snapshot_step(snap) not in pool:
                continue
            a = na if na.startswith("bot:") else key(elo_mod.snapshot_step(na))
            b = nb if nb.startswith("bot:") else key(elo_mod.snapshot_step(nb))
            edges.append((a, b, wa, g, "bot", run))
    return edges, nodes_by_run


def external_edges(cells_out: Path) -> "tuple[list, list]":
    """(c) — one edge per campaign cell, plus the per-cell table the README prints.

    A cell whose ``status`` is not OK, or whose peer reported no ``argmax_match_rate``, is
    EXCLUDED and listed as excluded. A read that failed its own regime check is not a weaker
    measurement, it is a different one.
    """
    edges, table = [], []
    for cell in cells_mod.cells():
        path = cells_out / "cells" / cell["id"] / "summary.json"
        row = {"id": cell["id"], "kind": cell["kind"], "our_node": cell["our_node"],
               "their_node": cell["their_node"], "status": "MISSING",
               "n": 0, "wins": 0, "losses": 0, "ties": 0, "win_rate": None,
               "wilson95": None, "mean_turns": None, "caps": 0, "argmax_rates": None,
               "included": False, "exclude_reason": "no summary.json"}
        if path.exists():
            blob = json.loads(path.read_text())
            rates = blob.get("their_argmax_match_rates") or []
            row.update({
                "status": blob.get("status"), "n": blob.get("n", 0),
                "wins": blob.get("wins", 0), "losses": blob.get("losses", 0),
                "ties": blob.get("ties", 0), "win_rate": blob.get("win_rate"),
                "wilson95": blob.get("wilson95"), "mean_turns": blob.get("mean_turns"),
                "caps": blob.get("hit_forfeit_limit", 0), "argmax_rates": rates,
                "team_asym": blob.get("team_source_asymmetry"),
                "distinct_our_teams": blob.get("distinct_our_teams"),
            })
            if blob.get("status") != "OK":
                row["exclude_reason"] = f"status={blob.get('status')}"
            elif rates != [1.0]:
                # The peer's greedy is VERIFIED per decision; anything but a clean 1.0 means the
                # cell was not the regime it claims, and an unverified regime is the one number
                # this whole apparatus exists not to emit.
                row["exclude_reason"] = f"peer argmax_match_rate {rates} != [1.0]"
            else:
                row["included"] = True
                row["exclude_reason"] = ""
                edges.append((cell["our_node"], cell["their_node"],
                              blob["wins"], blob["n"], "external", "campaign"))
        table.append(row)
    return edges, table


# ────────────────────────────────────────────────────────────────── the fits
def round_robin_edges() -> list:
    """The 2026-06-06 BOT-vs-BOT round robin itself, as edges.

    🚨 **Without these the FREEBOT arm is not asking the question it looks like it is asking.**
    The nine pins ARE this matrix — 36 pairs, 2,000-2,700 games each. A fit that frees the bots
    but omits their own round robin leaves them determined only by their edges to our snapshots
    and by `random`'s pin, and `random` loses ~100% of everything so its edges are near-perfect
    scores that the Gaussian prior, not the data, resolves. The whole cluster then slides toward
    the prior's centre and every bot reads ~180 Elo low — an artifact of the missing edges, not a
    finding about the calibration. Measured here: -133 to -225 Elo, uniformly negative.

    Read from the calibration STORE rather than from the anchor file, because the store holds the
    raw counts and the anchor file holds only the fit of them.
    """
    import json

    from utils.paths import repo_path

    path = repo_path("designs", "ai_v5", "elo_calibration", "gen3_bot_elo_games.json")
    if not path.exists():
        raise SystemExit(f"no bot round-robin store at {path}")
    store = json.loads(path.read_text())
    out = []
    for e in (store.get("pairs") or {}).values():
        if e.get("games", 0) > 0:
            out.append((elo_mod.bot_key(e["a"]), elo_mod.bot_key(e["b"]),
                        e["wins_a"], e["games"], "roundrobin", "bots"))
    return out


def bot_pins() -> "tuple[dict, float]":
    """The nine PINNED bot ratings, read through an ABSOLUTE path.

    🚨 ``elo.BOT_ANCHORS_PATH`` is the RELATIVE ``data/gen3_bot_elo_anchors.json`` and
    ``load_bot_anchors`` returns None for a path that does not exist — so running this script from
    anywhere but a repo root silently produced an UNPINNED fit whose ratings came out ~700 Elo
    low and whose inflation read the wrong SIGN. A missing anchor must be a refusal here: the
    entire campaign is about what the pinned frame does, and a fit that quietly lost it is not a
    weaker answer, it is an answer to a different question.
    """
    from utils.paths import repo_path

    path = str(repo_path("data", "gen3_bot_elo_anchors.json"))
    anchors = elo_mod.load_bot_anchors(path)
    if not anchors:
        raise SystemExit(f"no bot anchors at {path}: this fit is meaningless without the pins")
    return dict(anchors["ratings"]), float(anchors.get("base", elo_mod.DEFAULT_BASE))


def fit(edges: list, *, pin_bots: bool) -> "tuple[dict, dict, bool]":
    pins, base = bot_pins()
    if pin_bots:
        pinned = {elo_mod.bot_key(n): float(v) for n, v in pins.items()}
    else:
        # `random` alone holds the gauge; the other eight float. Without SOME pin the whole fit
        # is only determined up to a constant.
        pinned = {elo_mod.bot_key("random"): float(pins.get("random", base))}
    results = [(a, b, w, g) for (a, b, w, g, _src, _run) in edges]
    return elo_mod.fit_pairwise(results, pinned=pinned, base=base)


def residuals(ratings: dict, edges: list, source: str | None = None) -> dict:
    errs = []
    for a, b, w, g, src, _run in edges:
        if g <= 0 or (source is not None and src != source):
            continue
        if a not in ratings or b not in ratings:
            continue
        errs.append(abs(elo_mod.win_prob(ratings[a], ratings[b]) - w / g))
    if not errs:
        return {"mean_abs_err": None, "max_abs_err": None, "n_edges": 0}
    return {"mean_abs_err": round(sum(errs) / len(errs), 4),
            "max_abs_err": round(max(errs), 4), "n_edges": len(errs)}


def truncate(edges: list, run: str, keep_steps: "set[int]") -> list:
    """Drop every edge with an endpoint that is a snapshot of ``run`` outside ``keep_steps``.

    This is what ``fit_ladder(first_n=...)`` does, generalised to a fit that holds four runs at
    once: the OTHER runs stay whole because they are the frame, and only the run under test is
    truncated. BT re-solves every node on every add, and the newest node of a fit is
    systematically inflated — measured on `ai_v9_29_rev1_0823`, 2052 in a first-4 fit against
    1958 in the final 12-node fit.
    """
    prefix = f"snap:{run}@"
    keep = {f"{prefix}{s}" for s in keep_steps}

    def ok(n: str) -> bool:
        return (not n.startswith(prefix)) or n in keep

    return [e for e in edges if ok(e[0]) and ok(e[1])]


def inflation(edges: list, run: str, steps: "list[int]", *, pin_bots: bool = True,
              k_from: int = 5) -> dict:
    """``infl(k) = rating_k(run truncated to its first k nodes) - rating_k(full fit)``."""
    full, _se, _c = fit(edges, pin_bots=pin_bots)
    per_k = {}
    for k in range(k_from, len(steps)):
        sub = truncate(edges, run, set(steps[:k]))
        r, _se2, _c2 = fit(sub, pin_bots=pin_bots)
        node = f"snap:{run}@{steps[k - 1]}"
        if node in r and node in full:
            per_k[steps[k - 1]] = round(r[node] - full[node], 1)
    vals = list(per_k.values())
    return {"per_k": per_k, "mean": round(sum(vals) / len(vals), 1) if vals else None,
            "max": max(vals) if vals else None, "n_k": len(vals)}


def verify_against_fit_ladder(run: str = "ai_v9_29_rev1_0823", tol: float = 0.2) -> None:
    """PRECONDITION, asserted rather than branched on: this module's single-run fit must reproduce
    ``snapshot_ladder.fit_ladder`` exactly.

    The joint fit is a generalisation of the production one — same BT core, same pins, same recipe
    (sentinel edges dropped) — and the whole campaign is read as a delta against it. If the two
    disagree, every number below is about a fit nobody else computes. It has already caught one:
    ``elo.BOT_ANCHORS_PATH`` is RELATIVE, so run from the wrong directory this fit lost the pins
    silently, came out ~700 Elo low, and reported the newest-node inflation with the WRONG SIGN.
    """
    edges, nodes = internal_edges([run])
    mine, _se, _c = fit(edges, pin_bots=True)
    theirs = sl.fit_ladder(run_dir(run), write=False)["ratings"]
    bad = []
    for step, val in theirs.items():
        got = mine.get(f"snap:{run}@{step}")
        if got is None or abs(got - float(val)) > tol:
            bad.append((step, val, got))
    if bad:
        raise SystemExit(
            f"fit_joint does NOT reproduce fit_ladder on {run}: {bad[:5]} "
            "(the joint fit is only readable as a delta against the production one)")
    print(f"precondition OK: fit_joint reproduces fit_ladder on {run} "
          f"({len(theirs)} nodes, tol {tol})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True, help="the campaign --out directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inflation-runs", nargs="*",
                    default=["ai_v13_01_flywheel_shaped", "ai_v12_02_winprob_critic"])
    args = ap.parse_args()
    cells_out = Path(args.cells).resolve()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # 🚨 `elo.BOT_ANCHORS_PATH` is RELATIVE (`data/gen3_bot_elo_anchors.json`) and
    # `load_bot_anchors` returns None rather than raising for a path that is not there — so
    # `snapshot_ladder.fit_ladder`, which this module is verified against, silently produces an
    # UNPINNED ladder from any other working directory. Stand in the repo root once, here, rather
    # than leaving every consumer of a relative data path to discover it separately.
    import os

    from utils.paths import repo_root

    os.chdir(repo_root())
    verify_against_fit_ladder()

    internal, nodes_by_run = internal_edges()
    ext, table = external_edges(cells_out)
    print(f"internal edges: {len(internal)}   external edges: {len(ext)} "
          f"({sum(1 for r in table if r['included'])}/{len(table)} cells included)")

    rr = round_robin_edges()
    print(f"bot round-robin edges: {len(rr)} pairs, "
          f"{sum(e[3] for e in rr):,} games")

    arms = {}
    for name, edges, pin in (("JOINT-NOEXT", internal, True),
                             ("JOINT-EXT", internal + ext, True),
                             # The registered FREEBOT arm, kept because it was registered — and
                             # because what it exposes (the round robin is load-bearing) is worth
                             # showing rather than quietly replacing.
                             ("JOINT-EXT-FREEBOT", internal + ext, False),
                             # The arm that actually asks "do the pins move when re-fit jointly":
                             # the bots free, but their OWN round robin in the edge set.
                             ("JOINT-EXT-FREEBOT-RR", internal + ext + rr, False)):
        r, se, conv = fit(edges, pin_bots=pin)
        arms[name] = {
            "ratings": {k: round(v, 1) for k, v in sorted(r.items())},
            "se": {k: round(v, 1) for k, v in sorted(se.items())},
            "converged": conv,
            "residuals_all": residuals(r, edges),
            "residuals_bot_edges": residuals(r, edges, source="bot"),
            "residuals_dense_edges": residuals(r, edges, source="dense"),
            "residuals_external_edges": residuals(r, edges, source="external"),
            "residuals_roundrobin_edges": residuals(r, edges, source="roundrobin"),
            "n_edges": len(edges),
        }
        print(f"{name:20s} converged={conv} "
              f"all={arms[name]['residuals_all']} bot={arms[name]['residuals_bot_edges']}")

    infl = {}
    for run in args.inflation_runs:
        steps = nodes_by_run[run]
        infl[run] = {
            "steps": steps,
            "JOINT-NOEXT": inflation(internal, run, steps),
            "JOINT-EXT": inflation(internal + ext, run, steps),
        }
        print(f"inflation {run}: noext={infl[run]['JOINT-NOEXT']['mean']} "
              f"ext={infl[run]['JOINT-EXT']['mean']}")

    blob = {
        "arms": arms,
        "cells": table,
        "nodes_by_run": nodes_by_run,
        "played_snapshots": [cells_mod.snap_node(r, s) for r, s in cells_mod.SNAPSHOTS],
        "bot_pins": bot_pins()[0],
        "inflation": infl,
        "edge_counts": {"internal": len(internal), "external": len(ext),
                        "roundrobin": len(rr)},
    }
    (out / "joint_fit.json").write_text(json.dumps(blob, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out / 'joint_fit.json'}")

    # THE RAW EVIDENCE, in one file: every game of every cell, each row still carrying its own
    # regime, team set, opponent version, commit and loader — the per-row stamping is the point,
    # so the concatenation adds only `cell_id` and never rewrites a field.
    n = 0
    with open(out / "games.jsonl", "w") as fh:
        for cell in cells_mod.cells():
            src = cells_out / "cells" / cell["id"] / "games.jsonl"
            if not src.exists():
                continue
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["cell_id"] = cell["id"]
                row["our_node"] = cell["our_node"]
                row["their_node"] = cell["their_node"]
                fh.write(json.dumps(row) + "\n")
                n += 1
    print(f"wrote {out / 'games.jsonl'} ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
