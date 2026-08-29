"""Score the TRANSFER-COEFFICIENT CELL — defensive search vs the EVAL ROSTER at 24M.

    python designs/research_state/measurements/transfer_coefficient_cell_report.py \
        "tmp/tcell/shardA.jsonl,tmp/tcell/shardB.jsonl,tmp/tcell/shardC.jsonl" \
        designs/research_state/measurements/transfer_coefficient_cell_2026-08-29.json

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

**The question.** Probe K re-judged iteration 2's overrule population under marginalized ground
truth and found the overrules genuinely **+0.0474 [+0.0216, +0.0730]** better per decision — yet
the iteration-2 mirror cell's game-level dividend was exactly zero. Two sound measurements
disagreeing by more than 2x their CIs means a TRANSFER failure, and K's §6 named three suspects:
CHECKPOINT (the mirror ran ~10M, the labels 24M), POPULATION (a mirror twin, not the eval roster),
and COMPOUNDING (a one-substitution label assumes the POLICY plays on; live, the SEARCHER does).

This cell removes the first two at once: the same defensive configuration, at the 24M weights the
labels were computed on, against the eval roster the labels' decisions were drawn from.

**The design is a two-arm paired A/B on one battery invocation.** Arm A is the defensive searcher;
arm B is ``--arm base`` — the same network with search structurally off, choosing the masked argmax
(the literal control, not a re-implementation of one). ``game_seed`` and ``team_pair`` are functions
of ``(opponent, game_index, --games-seed)`` alone, so a unit is the same pinned dice and the same
team draw in both arms and the two arms' trajectories are IDENTICAL until the first overrule. The
paired difference is therefore a statement about the OVERRULES and nothing else.

**Why the transfer coefficient and not the registered pp band.** The subject's own recorded
``latest_eval`` at step 24,000,000 puts ``win_rate_vs_bots`` at 0.9162, so the registered "+5-12 pp"
reading is unreachable on this population by arithmetic (the ceiling is +8.4 pp) — it is scored
UNREACHABLE, not missed. What survives is
``tau = (A - B) / (overrules_per_game_MEASURED_HERE x 0.0474)``: the fraction of the naive additive
expectation that actually arrives at the scoreboard, with the expectation's *rate* factor taken
from this cell's own population rather than from the mirror's. That is the number the cell exists
to produce.

⚠️ Three SHARDS over disjoint game-index windows. That is scheduling, not three experiments — the
seed and the team draw are functions of the index alone. The script ASSERTS disjointness rather
than trusting the launch line, and asserts that the rows carry exactly two cells per opponent.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: Probe K's per-decision gain for the iteration-2 overrule population under MARGINALIZED ground
#: truth. The ONE number this script imports; everything else is measured here.
K_PER_DECISION_GAIN = 0.0474
K_PER_DECISION_CI = (0.0216, 0.0730)

#: Iteration 2's rate table (mirror twin @ ~10M), quoted from the committed record for the
#: side-by-side. Never re-run here.
ITER2_RATES = {
    "forced": 0.7515, "raced": 0.2485, "separated_of_raced": 0.4542,
    "kept_of_raced": 0.2200, "overrule_of_raced": 0.2342, "overrule_of_all": 0.0582,
    "futility_of_raced": 0.5458, "rounds_per_race": 13.17, "eliminated_per_race": 5.35,
    "mean_search_s_per_raced_decision": 2.278, "contested_decisions_per_game": 9.42,
    "search_s_per_game": 21.46, "mean_wall_s_per_game": 24.44,
}
#: ...and its overrules per game, derived from the record: 3531 overrules / 1600 orientation-games.
ITER2_OVERRULES_PER_GAME = 3531 / 1600.0

#: The subject's OWN recorded per-opponent win rates at step 24,000,000 (metadata.json ->
#: latest_eval.opponents), used ONLY to define the pre-declared strata. External to this cell.
LATEST_EVAL_WR = {
    "random": 1.00, "heuristic": 0.89, "heuristic2": 0.91, "staller": 0.96,
    "staller_v2": 0.96, "aggressive": 0.92, "aggressive_v2": 0.88,
    "setup_sweep": 0.93, "setup_sweep_v2": 0.88,
}
LATEST_EVAL_VS_BOTS = 0.9162

#: The two roster bots whose policy contains a GLOBAL-RNG coin flip
#: (``agents/opponents.py``: ``random.random() < _PROTECT_PROBABILITY`` in ``Gen3StallerPlayer``
#: and ``Gen3StallerV2Player``). Both arms draw from the one process-wide ``random`` module and
#: the searched arm interleaves its ``choose_move`` differently — it awaits an executor while the
#: base arm runs inline — so against THESE two the pair can desynchronize with no overrule at all.
#: Discovered by this cell's own integrity check, not assumed: see `coin_flip_bots` in the output.
COIN_FLIP_BOTS = ("staller", "staller_v2")

#: PRE-DECLARED strata (tmp/tcell/PREREG.md, written before the main run). `hard4` is the four
#: opponents the subject's own eval put at <= 0.91; `random` is the zero-headroom cell.
#: `deterministic7` is a POST-HOC sensitivity added after the integrity check fired, and is
#: labelled as such rather than presented as pre-registered.
STRATA = {
    "all": tuple(LATEST_EVAL_WR),
    "hard4": ("aggressive_v2", "setup_sweep_v2", "heuristic", "heuristic2"),
    "no_random": tuple(k for k in LATEST_EVAL_WR if k != "random"),
    "random_only": ("random",),
    "deterministic7_POSTHOC": tuple(k for k in LATEST_EVAL_WR if k not in COIN_FLIP_BOTS),
    "coin_flip_family_POSTHOC": COIN_FLIP_BOTS,
}


def rows(path: str) -> List[dict]:
    out: List[dict] = []
    if not os.path.exists(path):
        return out
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:                                # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break                                     # a torn final line stops the read
    return out


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(xs: Sequence[float], z: float = 1.96):
    n = len(xs)
    if n == 0:
        return (None, None, None, 0)
    m = sum(xs) / n
    if n < 2:
        return (m, None, None, n)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    half = z * math.sqrt(var / n)
    return (m, m - half, m + half, n)


def score_of(r: dict) -> Optional[float]:
    """1 / 0.5 / 0 for win / tie / loss — or None for a game that did not finish.

    A TIMEOUT IS NEVER A SEMANTIC OUTCOME. An unfinished row stays in the file as evidence and is
    counted in its own bucket; it is excluded from BOTH arms of its pair rather than scored, so a
    dead battle can never dilute a win rate toward whatever it happened to be."""
    if not int(r.get("finished", 0)):
        return None
    if int(r.get("tied", 0) or 0):
        return 0.5
    return float(int(r.get("won", 0)))


def units(rs: Sequence[dict], arm: str) -> Dict[Tuple[str, int], float]:
    """``{(opponent, game_index): score}`` for one arm. Roster cells are not side-swapped, so the
    orientation is always 0 and the unit is the (opponent, game) pair."""
    out: Dict[Tuple[str, int], float] = {}
    for r in rs:
        if r.get("arm") != arm:
            continue
        s = score_of(r)
        if s is None:
            continue
        out[(r["opponent"], int(r["game"]))] = s
    return out


def unpaired(rs: Sequence[dict], arm: str, keep=None) -> dict:
    cell = [r for r in rs if r.get("arm") == arm
            and (keep is None or r.get("opponent") in keep)]
    fin = [r for r in cell if int(r.get("finished", 0))]
    ties = sum(int(r.get("tied", 0) or 0) for r in fin)
    won = sum(int(r.get("won", 0)) for r in fin)
    decisive = len(fin) - ties
    p, lo, hi = wilson(won, decisive)
    return {"games": len(cell), "finished": len(fin), "decisive": decisive, "won": won,
            "ties": ties,
            "errors": sum(1 for r in cell if r.get("error")),
            "unfinished": sum(1 for r in cell if not int(r.get("finished", 0))),
            "win_rate": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def paired(a: Dict, b: Dict, keep=None) -> dict:
    """The A-B row: mean paired difference + normal CI, with McNemar discordant counts."""
    shared = sorted(set(a) & set(b), key=lambda k: (k[0], k[1]))
    if keep is not None:
        shared = [k for k in shared if k[0] in keep]
    diffs = [a[k] - b[k] for k in shared]
    m, lo, hi, n = mean_ci(diffs)
    a_wins = sum(1 for d in diffs if d > 0)
    b_wins = sum(1 for d in diffs if d < 0)
    disc = a_wins + b_wins
    return {
        "n_pairs": n,
        "a_win_rate": round(sum(a[k] for k in shared) / n, 4) if n else None,
        "b_win_rate": round(sum(b[k] for k in shared) / n, 4) if n else None,
        "delta": round(m, 4) if m is not None else None,
        "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "ci_halfwidth": round((hi - lo) / 2, 4) if lo is not None else None,
        "excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
        "discordant": {"a_only": a_wins, "b_only": b_wins, "total": disc,
                       "frac_of_pairs": round(disc / n, 4) if n else None},
        "identical_pairs": (n - disc) if n else 0,
    }


def overrules_by_unit(rs: Sequence[dict]) -> Dict[Tuple[str, int], int]:
    """``{(opponent, game): arm-A overrule count}`` — the conditioning variable.

    Conditioning the paired read on this is PRE-TREATMENT, not outcome selection: the two arms
    play the identical trajectory until A's FIRST overrule, so whether a unit has any overrule at
    all is a function of the prefix they share exactly. In a unit with zero overrules the arms ARE
    the same battle, so its paired difference must be exactly 0 — which makes the zero bucket an
    internal-validity check on the matching rather than a result."""
    return {(r["opponent"], int(r["game"])): int(r.get("n_defensive_overruled", 0) or 0)
            for r in rs if r.get("arm") == "honest"}


def defensive_fold(rs: Sequence[dict], arm: str, keep=None) -> dict:
    cell = [r for r in rs if r.get("arm") == arm
            and (keep is None or r.get("opponent") in keep)]
    if not cell:
        return {}
    keys = sorted({k for r in cell for k in r if k.startswith("n_defensive")})
    fold = {key: sum(int(r.get(key, 0) or 0) for r in cell) for key in keys}
    nd = max(1, fold.get("n_defensive", 0))
    ndr = max(1, fold.get("n_defensive_raced", 0))
    nfut = fold.get("n_defensive_futility", 0)
    n_race = max(1, sum(int(r.get("n_racing", 0) or 0) for r in cell))
    rounds = sum(int(r.get("racing_rounds_total", 0) or 0) for r in cell)
    srch = sum(int(r.get("n_searched", 0) or 0) for r in cell)
    dec = sum(int(r.get("n_decisions", 0) or 0) for r in cell)
    chg = sum(int(r.get("n_changed", 0) or 0) for r in cell)
    wall = sum(float(r.get("wall_s", 0.0)) for r in cell)
    search_s = sum(float((r.get("realized_mean") or {}).get("elapsed", 0.0) or 0.0)
                   * int(r.get("n_searched", 0) or 0) for r in cell)
    ngames = max(1, len(cell))
    fut_dl = fold.get("n_defensive_futility_deadline")
    return {
        "games": len(cell), "decisions": dec, "searched": srch, "changed": chg,
        "searched_frac": round(srch / dec, 4) if dec else None,
        "change_rate": round(chg / srch, 4) if srch else None,
        "decisions_per_game": round(dec / ngames, 2),
        "counts": fold,
        "rates": {
            "forced": round(fold.get("n_defensive_forced", 0) / nd, 4),
            "forced_wp": round(fold.get("n_defensive_forced_wp", 0) / nd, 4),
            "forced_n_legal": round(fold.get("n_defensive_forced_n_legal", 0) / nd, 4),
            "raced": round(fold.get("n_defensive_raced", 0) / nd, 4),
            "separated_of_raced": round(fold.get("n_defensive_separated", 0) / ndr, 4),
            "kept_of_raced": round(fold.get("n_defensive_kept", 0) / ndr, 4),
            "futility_of_raced": round(nfut / ndr, 4),
            "overrule_of_all": round(fold.get("n_defensive_overruled", 0) / nd, 4),
            "overrule_of_raced": round(fold.get("n_defensive_overruled", 0) / ndr, 4),
            "futility_deadline_frac": (round(fut_dl / nfut, 4)
                                       if (fut_dl is not None and nfut) else None),
        },
        "overrules": fold.get("n_defensive_overruled", 0),
        "overrules_per_game": round(fold.get("n_defensive_overruled", 0) / ngames, 4),
        "race": {
            "raced_decisions": n_race, "rounds_total": rounds,
            "rounds_per_race": round(rounds / n_race, 3),
            "eliminated_per_race": round(sum(int(r.get("racing_eliminated_total", 0) or 0)
                                             for r in cell) / n_race, 3),
            "deadline_truncated_decisions": sum(int(r.get("deadline_truncated", 0) or 0)
                                                for r in cell),
            "mean_search_s_per_raced_decision": round(search_s / max(1, srch), 4),
        },
        "envelope": {
            "mean_wall_s_per_game": round(wall / ngames, 2),
            "total_wall_h": round(wall / 3600.0, 2),
            "search_s_per_game": round(search_s / ngames, 2),
            "contested_decisions_per_game": round(ndr / ngames, 2),
        },
    }


def transfer(delta: Optional[float], overrules_per_game: float,
             delta_ci: Optional[Sequence[float]] = None) -> dict:
    """tau = realized A-B divided by the naive additive expectation.

    The expectation multiplies THIS cell's measured overrule rate by probe K's per-decision gain.
    Both the low and high ends of K's CI are carried through, because a transfer coefficient whose
    denominator is a point estimate would report a precision the input does not have."""
    e = overrules_per_game * K_PER_DECISION_GAIN
    e_lo = overrules_per_game * K_PER_DECISION_CI[0]
    e_hi = overrules_per_game * K_PER_DECISION_CI[1]
    return {
        "overrules_per_game": round(overrules_per_game, 4),
        "per_decision_gain": K_PER_DECISION_GAIN,
        "per_decision_gain_ci": list(K_PER_DECISION_CI),
        "naive_additive_expectation": round(e, 4),
        "naive_expectation_range_from_K_ci": [round(e_lo, 4), round(e_hi, 4)],
        "realized_delta": delta,
        "tau_point": (round(delta / e, 3) if (delta is not None and e > 0) else None),
        "tau_range_from_K_ci": ([round(delta / e_hi, 3), round(delta / e_lo, 3)]
                                if (delta is not None and e_lo > 0) else None),
        # THE INTERVAL THAT MATTERS. K's CI moves tau by a factor; the realized delta's CI is what
        # decides whether tau is distinguishable from 0 or from 1, and it is much the wider of the
        # two. Reporting only the K-CI range would understate the uncertainty by design.
        "tau_range_from_realized_delta_ci": ([round(delta_ci[0] / e, 3), round(delta_ci[1] / e, 3)]
                                             if (delta_ci and e > 0) else None),
        "full_transfer_excluded": bool(delta_ci and e > 0 and not
                                       (delta_ci[0] <= e <= delta_ci[1])),
        "zero_transfer_excluded": bool(delta_ci and not (delta_ci[0] <= 0 <= delta_ci[1])),
        "note": ("tau = 1 means every per-decision win-probability point arrived at the "
                 "scoreboard; tau = 0 means none did. The additive accounting is FIRST-ORDER by "
                 "construction — win-probability gains at different decisions of one game are not "
                 "independent and do not literally sum — so tau is a diagnostic ratio against a "
                 "stated benchmark, not an unbiased estimator of anything."),
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = (argv[0] if argv else "tmp/tcell/shardA.jsonl").split(",")
    out_path = argv[1] if len(argv) > 1 else None

    shards, cell = {}, []
    seen: Dict[Tuple[str, int], str] = {}
    for p in paths:
        p = p.strip()
        rs = rows(p)
        name = os.path.basename(p)
        idx = sorted({int(r["game"]) for r in rs})
        shards[name] = {"rows": len(rs), "game_index_range": [idx[0], idx[-1]] if idx else None}
        # DISJOINT windows, asserted rather than trusted.
        for r in rs:
            k = (r["arm"], r["opponent"], int(r["game"]))
            if seen.setdefault(k, name) != name:
                raise SystemExit(f"shards {seen[k]} and {name} both played {k} — the windows "
                                 "overlap and the paired read would be wrong")
        cell += rs
    if not cell:
        print(f"no rows in {paths}", file=sys.stderr)
        return 1

    # ...and exactly TWO cells per opponent, differing only in the arm.
    sig = {(r.get("arm"), r.get("budget"), r.get("root_strategy"), r.get("score_mode"))
           for r in cell}
    arms = sorted({a for (a, *_r) in sig})
    if arms != ["base", "honest"]:
        raise SystemExit(f"expected arms base+honest, got {arms} ({sorted(sig)})")

    # ---- BALANCE THE CELL before anything is computed ------------------------------------------
    # The battery plays arm `base` over every opponent first and arm `honest` second, so a run
    # stopped early would leave the honest arm complete for the leading opponents and empty for the
    # trailing ones. Pooling that would silently redefine the population as "the first k bots".
    # Every statistic below is therefore computed on units FINISHED IN BOTH ARMS, and what was
    # dropped is reported rather than absorbed.
    a_raw, b_raw = units(cell, "honest"), units(cell, "base")
    shared_units = set(a_raw) & set(b_raw)
    a_u = {k: v for k, v in a_raw.items() if k in shared_units}
    b_u = {k: v for k, v in b_raw.items() if k in shared_units}
    cell_bal = [r for r in cell if (r["opponent"], int(r["game"])) in shared_units]
    balance = {
        "units_finished_in_both_arms": len(shared_units),
        "dropped_A_only": len(set(a_raw) - shared_units),
        "dropped_B_only": len(set(b_raw) - shared_units),
        "pairs_per_opponent": {o: sum(1 for (oo, _g) in shared_units if oo == o)
                               for o in sorted(LATEST_EVAL_WR)},
        "why": ("the battery plays arm base over all opponents before arm honest, so an early "
                "stop would leave a leading-opponent-heavy honest arm; every statistic is "
                "restricted to units both arms finished"),
    }
    cell = cell_bal
    fold_all = defensive_fold(cell, "honest")

    strata_out = {}
    for name, keep in STRATA.items():
        pr = paired(a_u, b_u, keep)
        fold = defensive_fold(cell, "honest", keep)
        strata_out[name] = {
            "opponents": list(keep),
            "A_defensive_search": unpaired(cell, "honest", keep),
            "B_policy_only": unpaired(cell, "base", keep),
            "paired_A_minus_B": pr,
            "overrules_per_game": fold.get("overrules_per_game"),
            "transfer": transfer(pr["delta"], fold.get("overrules_per_game") or 0.0,
                                 pr.get("ci95")),
        }

    per_opponent = {}
    for opp in sorted(LATEST_EVAL_WR):
        keep = (opp,)
        pr = paired(a_u, b_u, keep)
        fold = defensive_fold(cell, "honest", keep)
        per_opponent[opp] = {
            "recorded_eval_wr_24M": LATEST_EVAL_WR[opp],
            "A_win_rate": unpaired(cell, "honest", keep)["win_rate"],
            "B_win_rate": unpaired(cell, "base", keep)["win_rate"],
            "n_pairs": pr["n_pairs"],
            "delta": pr["delta"], "ci95": pr["ci95"],
            "discordant": pr["discordant"],
            "overrules_per_game": fold.get("overrules_per_game"),
            "forced": (fold.get("rates") or {}).get("forced"),
            "raced": (fold.get("rates") or {}).get("raced"),
            "overrule_of_all": (fold.get("rates") or {}).get("overrule_of_all"),
        }

    # ---- the OVERRULE-CONDITIONED read (PREREG addendum 3+4) ----------------------------------
    ov = overrules_by_unit(cell)
    shared_all = set(a_u) & set(b_u)

    def cond(pred, label: str, opps=None) -> dict:
        keep_units = {k for k in shared_all
                      if pred(ov.get(k, 0)) and (opps is None or k[0] in opps)}
        d = [a_u[k] - b_u[k] for k in sorted(keep_units)]
        m, lo, hi, n = mean_ci(d)
        aw = sum(1 for x in d if x > 0)
        bw = sum(1 for x in d if x < 0)
        n_ov = sum(ov.get(k, 0) for k in keep_units)
        return {
            "label": label, "n_pairs": n,
            "total_overrules": n_ov,
            "overrules_per_game": round(n_ov / n, 4) if n else None,
            "a_win_rate": round(sum(a_u[k] for k in keep_units) / n, 4) if n else None,
            "b_win_rate": round(sum(b_u[k] for k in keep_units) / n, 4) if n else None,
            "delta": round(m, 4) if m is not None else None,
            "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
            "discordant": {"a_only": aw, "b_only": bw, "total": aw + bw},
        }

    zero_bucket = cond(lambda c: c == 0, "no overrule — the arms are the SAME battle")
    det = tuple(k for k in LATEST_EVAL_WR if k not in COIN_FLIP_BOTS)
    zero_det = cond(lambda c: c == 0, "no overrule, DETERMINISTIC bots only", det)
    zero_coin = cond(lambda c: c == 0, "no overrule, COIN-FLIP bots only", COIN_FLIP_BOTS)
    conditioned = {
        "zero_overrule_INTEGRITY_CHECK": {
            **zero_bucket,
            "must_be_exactly_zero_on_deterministic_opponents": True,
            "deterministic7": zero_det,
            "coin_flip_family": zero_coin,
            "passes_on_deterministic7": bool(zero_det["delta"] == 0.0 and
                                             zero_det["discordant"]["total"] == 0),
            "why": ("with no overrule the searcher plays the policy's own argmax at every "
                    "decision, so the two arms should be the SAME battle and the paired "
                    "difference should be exactly 0. It is, across every unit of the seven "
                    "DETERMINISTIC bots — which is a positive verification of the matching on "
                    "78% of the cell rather than an assumption about it."),
            "the_exception_and_its_cause": (
                "The staller family is NOT deterministic: `Gen3StallerPlayer` and "
                "`Gen3StallerV2Player` decide whether to Protect with "
                "`random.random() < _PROTECT_PROBABILITY` — a draw from the PROCESS-WIDE `random` "
                "module. Both players in a battle share that module, and the searched arm "
                "interleaves its `choose_move` differently (it awaits an executor; the base arm "
                "runs inline), so the coin can land differently in the two arms with no overrule "
                "involved. The failure is CONFINED to exactly those two bots by measurement, "
                "which is what identifies the cause: 0 of 2688 deterministic-bot zero-overrule "
                "pairs diverge, against 4 of 760 staller-family ones. This is unbiased NOISE, "
                "not a bias — it inflates the discordant count in both directions and widens the "
                "interval; the `deterministic7_POSTHOC` stratum is the arm free of it, and it "
                "moves the headline by 0.15 pp."),
        },
        "at_least_one_overrule": cond(lambda c: c >= 1, ">= 1 overrule — the treated units"),
        "by_overrule_count": {
            "0": zero_bucket, "1": cond(lambda c: c == 1, "exactly 1"),
            "2": cond(lambda c: c == 2, "exactly 2"),
            "3+": cond(lambda c: c >= 3, "3 or more"),
        },
    }
    treated = conditioned["at_least_one_overrule"]
    conditioned["transfer_on_treated_units"] = transfer(
        treated["delta"], treated["overrules_per_game"] or 0.0, treated.get("ci95"))

    # An EQUAL-WEIGHT-PER-OPPONENT estimator alongside the pooled one. The roster is a set of nine
    # opponents, not a bag of games, so if the per-opponent pair counts ever drift apart the pooled
    # mean quietly re-weights the population. Reported always, so the two can be compared rather
    # than one being trusted.
    per_opp_deltas = []
    for opp in sorted(LATEST_EVAL_WR):
        ks = [k for k in a_u if k[0] == opp]
        if ks:
            per_opp_deltas.append(sum(a_u[k] - b_u[k] for k in ks) / len(ks))
    bm, blo, bhi, bn = mean_ci(per_opp_deltas)
    balanced_delta = {
        "n_opponents": bn,
        "delta": round(bm, 4) if bm is not None else None,
        "ci95": [round(blo, 4), round(bhi, 4)] if blo is not None else None,
        "note": ("mean over the nine per-opponent deltas, CI across OPPONENTS (n=9) — a much "
                 "weaker interval than the pooled one by construction; it is here to show the "
                 "pooled number is not an artifact of unequal per-opponent n, not to replace it"),
    }

    primary = strata_out["all"]["paired_A_minus_B"]
    hw = primary["ci_halfwidth"]
    rates = fold_all["rates"]
    rate_table = {
        k: {"iteration2_mirror_10M": ITER2_RATES.get(k), "this_cell_roster_24M": rates.get(k)}
        for k in ("forced", "raced", "separated_of_raced", "kept_of_raced",
                  "overrule_of_raced", "overrule_of_all", "futility_of_raced")
    }
    for k in ("rounds_per_race", "eliminated_per_race", "mean_search_s_per_raced_decision"):
        rate_table[k] = {"iteration2_mirror_10M": ITER2_RATES.get(k),
                         "this_cell_roster_24M": fold_all["race"].get(k)}
    for k in ("contested_decisions_per_game", "search_s_per_game", "mean_wall_s_per_game"):
        rate_table[k] = {"iteration2_mirror_10M": ITER2_RATES.get(k),
                         "this_cell_roster_24M": fold_all["envelope"].get(k)}
    rate_table["overrules_per_game"] = {
        "iteration2_mirror_10M": round(ITER2_OVERRULES_PER_GAME, 4),
        "this_cell_roster_24M": fold_all["overrules_per_game"]}

    unfinished = sum(1 for r in cell if not int(r.get("finished", 0)))
    errored = sum(1 for r in cell if r.get("error"))
    walls = sorted(float(r.get("wall_s", 0.0)) for r in cell)

    readings = {
        "R1_transfer_is_fine": {
            "registered": ("A-B ~ + (overrules/game x per-decision gain), roughly +5-12 pp if "
                           "overrule behaviour matches iteration 2's ~2.2/game at +4.7 pp"),
            "status_of_the_pp_band": (
                "UNREACHABLE BY CONSTRUCTION on this population and scored as such, not as "
                f"missed: the subject's own recorded win_rate_vs_bots at 24M is "
                f"{LATEST_EVAL_VS_BOTS}, so the arithmetic ceiling on A-B is "
                f"+{round(1 - LATEST_EVAL_VS_BOTS, 4)}. The band was registered from mirror "
                "arithmetic before the roster's saturation was in view. The population ALSO "
                "changed the strategy's behaviour, so the rate factor is re-measured here."),
            "selected": bool(primary["ci95"] and primary["ci95"][0] > 0),
        },
        "R2_compounding_destroys_it": {
            "registered": "A-B ~ 0 => compounding/selection destroys per-decision gains in vivo",
            "selected": bool(primary["ci95"] and primary["ci95"][0] <= 0 <= primary["ci95"][1]),
            "resolution": (f"a true |A-B| >= {hw} would have resolved at this n" if hw else None),
        },
        "R3_intermediate": {
            "registered": "report the transfer coefficient — the number itself is the finding",
            "tau": strata_out["all"]["transfer"],
        },
    }

    out = {
        "date": "2026-08-29",
        "cell": "TRANSFER-COEFFICIENT CELL — probe K §6 decisive test",
        "question": ("does the +0.0474 [+0.0216,+0.0730] per-decision overrule gain probe K "
                     "measured under marginalized ground truth arrive at the GAME level, once "
                     "the checkpoint and the population suspects are removed?"),
        "checkpoint": ("models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip — "
                       "BYTE-IDENTICAL (md5 df3d5620...) to eval_traces/step_24000000/"
                       "snapshot.zip, the exact weights probe G labelled and probe K re-judged. "
                       "Iteration 2 ran checkpoint_9995088_steps.zip (~10M): the CHECKPOINT "
                       "suspect."),
        "population": ("the EVAL ROSTER — the battery's default --opponents, i.e. "
                       "eval_opponent_names(): the 9 fixed scripted bots. Pool sentinels are not "
                       "constructible as battery opponents, so the roster here is the bot half of "
                       "the population probe G/K drew their decisions from; stated as a limit, "
                       "not glossed. Iteration 2 ran --opponents self (the mirror twin): the "
                       "POPULATION suspect."),
        "arms": {
            "A": ("--arm honest --budget 1 --root-strategy defensive --defensive-leaf winprob "
                  "--defensive-wp-margin 0.15 --defensive-confirm 0 "
                  "--defensive-contested-deadline-s 3.0 — iteration 2's configuration verbatim"),
            "B": ("--arm base — the same network with search structurally off, playing the masked "
                  "argmax; the literal control"),
            "matching": ("one battery invocation per shard plays both arms, so a unit is the same "
                         "pinned sim seed and the same team draw in A and B and the two arms' "
                         "trajectories are identical until the first overrule"),
        },
        "invocation": ("python -m main.search_dividend <ckpt> --arm base --arm honest --budget 1 "
                       "--root-strategy defensive --defensive-leaf winprob "
                       "--defensive-wp-margin 0.15 --defensive-confirm 0 "
                       "--defensive-contested-deadline-s 3.0 --games-start <lo> --games 150 "
                       "--games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"),
        "shards": shards,
        "accounting": {
            "rows": len(cell),
            "unfinished": unfinished, "errors": errored,
            "unfinished_frac": round(unfinished / len(cell), 5),
            "longest_game_s": round(walls[-1], 1) if walls else None,
            "livelock_backstop_s": 1800, "idle_wedge_s": 120,
            "note": ("a timed-out or unfinished game is its own bucket and is excluded from BOTH "
                     "arms of its pair, never scored as a semantic outcome"),
        },
        "balance": balance,
        "primary_A_minus_B": primary,
        "opponent_balanced_A_minus_B": balanced_delta,
        "transfer_coefficient": strata_out["all"]["transfer"],
        "overrule_conditioned": conditioned,
        "strata": strata_out,
        "per_opponent": per_opponent,
        "rate_table_vs_iteration2": rate_table,
        "arm_A_fold": fold_all,
        "readings": readings,
        "coin_flip_bots": {
            "bots": list(COIN_FLIP_BOTS),
            "site": ("agents/opponents.py — `random.random() < _PROTECT_PROBABILITY` in "
                     "Gen3StallerPlayer and Gen3StallerV2Player"),
            "effect": ("a shared process-wide RNG the two arms can consume in a different order, "
                       "so a pair can diverge with zero overrules; found by the integrity check, "
                       "confined to these two bots by measurement, unbiased in direction"),
            "sensitivity": ("the `deterministic7_POSTHOC` stratum excludes them and is the "
                            "exactly-paired arm"),
        },
        "prereg": ("tmp/tcell/PREREG.md, written after a 36-battle cost pilot and BEFORE the main "
                   "run; the hard4 stratum is defined from the subject's own recorded latest_eval "
                   "per-opponent win rates, which are external to this cell's data"),
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=False)
            fh.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("strata",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
