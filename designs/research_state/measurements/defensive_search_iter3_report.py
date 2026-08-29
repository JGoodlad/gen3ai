"""Score DEFENSIVE PAIRED SEARCH iteration 3 — "confirm before you overrule".

    python designs/research_state/measurements/defensive_search_iter3_report.py \
        "tmp/iter3/shardA.jsonl,tmp/iter3/shardB.jsonl,tmp/iter3/shardC.jsonl" \
        designs/research_state/measurements/defensive_search_iter3_2026-08-29.json

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

The ONE change under test is ``--defensive-confirm 6 --defensive-confirm-deadline-s 30``: before
acting on an overrule the race has certified, the top-2 (the race's winner vs the policy's own
action) are settled by up to 6 PAIRED rollouts to a terminal through the playoff mechanism, and
the policy's action stands unless the paired difference clears 2·SE over >= 4 pairs. Everything
else is iteration 2's cell verbatim — contested deadline 3 s, gate 0.15, ``seq`` floor 5, win-prob
leaf, depth 1, the checkpoint, ``--games-seed 7``, side-swap on.

**Why the rollouts, and what they add that the leaf cannot.** Iteration 2 removed the evidence
starvation (separation at 95% of probe I's offline ceiling, 13.2 rounds per race) and the dividend
did not appear: 3,531 overrules — 13x iteration 1's — moved the win rate onto the null exactly.
The mechanism analysis banked with that result is the WINNER'S CURSE of a biased instrument: CRN
pairing cancels the dice and probe G's 72.8% shared per-decision offset, so what a race
CERTIFIES is the leaf's residual DIFFERENTIAL error (RMS 0.122 — larger than most true gaps) as
much as signal. A rollout to a terminal is scored by the SIM rather than by the critic, and it
contains the one thing probe G's +0.0219 was measured without: the opponent's response.

**THE ROWS ARE COMPUTED BY ONE FUNCTION.** ``arm_block`` is imported from the iteration-2 script
rather than re-written here, so iterations 1, 2 and 3 cannot drift apart in how their rates,
envelopes and paired intervals are derived — the same reason iteration 2 shared one function with
iteration 1.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import defensive_search_iter2_report as R                     # noqa: E402

#: ITERATION 2's rows, from the COMMITTED artifact beside this script (never a scratch directory —
#: a comparison that depends on someone's `tmp/` is a comparison that stops reproducing).
ITER2 = os.path.join(HERE, "defensive_search_iter2_2026-08-29_rows.jsonl.gz")

#: The confirm's registered shape. R is a CAP; the realized pair count is what gets reported.
CONFIRM_N = 6
CONFIRM_DEADLINE_S = 30.0


def confirm_block(cell: Sequence[dict]) -> dict:
    """Everything the CONFIRM stage reports about itself, over one cell.

    The headline is ``reject_rate``: the fraction of race-certified overrules that paired terminal
    rollouts would not stand behind. Its four components are kept apart because only two of them
    are findings about the LEAF — ``reversed`` (the rollouts conclusively preferred the policy) and
    ``inconclusive`` (they could not tell) — while ``no_budget`` and ``error`` are findings about
    the clock and the driver, and summing them would let a broken rollout family read as evidence.
    """
    def s(key: str) -> int:
        return sum(int(r.get(key, 0) or 0) for r in cell)

    att = s("n_defensive_confirm_attempted")
    conf = s("n_defensive_confirmed")
    secs = sum(float(r.get("defensive_confirm_s", 0.0) or 0.0) for r in cell)
    pairs = sum(int(r.get("playoff_r_total", 0) or 0) for r in cell)
    ran = s("n_playoff_ran")
    events = [e for r in cell for e in (r.get("defensive_confirm_events") or [])]
    nd = max(1, s("n_defensive"))
    raced = max(1, s("n_defensive_raced"))
    return {
        "attempted": att,
        # ⚠️ THE ROW THAT IS APPLES-TO-APPLES WITH ITERATION 2's 5.82% OVERRULE RATE. An attempt
        # happens exactly when the race separated on a non-policy action — iteration 2's overrule,
        # by the same definition — so this says whether the RACE is unchanged, which it must be.
        # The acted `overrule_of_all` in `rates` is a different quantity now: it is the race's
        # verdict AFTER the filter, and it is what the registered 1.5-3.5% band is about.
        "proposed_overrule_of_all": round(att / nd, 4),
        "proposed_overrule_of_raced": round(att / raced, 4),
        # ...and the other half of the same warning: with the confirm on, `n_defensive_kept`
        # absorbs BOTH "the race separated on the policy's own action" (iteration 2's kept) and
        # "the race separated elsewhere and the rollouts refused it". Only the first is comparable.
        "kept_on_own_action": s("n_defensive_kept") - s("n_defensive_confirm_declined"),
        "kept_on_own_action_of_raced": round(
            (s("n_defensive_kept") - s("n_defensive_confirm_declined")) / raced, 4),
        "confirmed": conf,
        "declined": s("n_defensive_confirm_declined"),
        "reversed": s("n_defensive_confirm_reversed"),
        "inconclusive": s("n_defensive_confirm_inconclusive"),
        "no_budget": s("n_defensive_confirm_no_budget"),
        "error": s("n_defensive_confirm_error"),
        # THE LEAF-BIAS METER IN VIVO — the registered reading.
        "reject_rate": round(1.0 - conf / att, 4) if att else None,
        "uphold_rate": round(conf / att, 4) if att else None,
        "rollout_pairs_total": pairs,
        "realized_pairs_per_attempt": round(pairs / att, 3) if att else None,
        "pair_cap": CONFIRM_N,
        "pairs_truncated_below_cap": sum(1 for e in events if int(e.get("r", 0)) < CONFIRM_N),
        "s_total": round(secs, 1),
        "s_per_attempt": round(secs / att, 2) if att else None,
        "s_per_pair": round(secs / pairs, 3) if pairs else None,
        "s_per_game": round(secs / max(1, len(cell)), 2),
        "attempts_per_game": round(att / max(1, len(cell)), 3),
        "playoff_ran": ran,
        "rollouts_capped": s("n_playoff_capped"),
        "rollout_pairs_failed": s("n_playoff_failed"),
        "min_detectable_paired_difference": min_detectable(CONFIRM_N),
        "_events": events,
    }


def min_detectable(n_pairs: int) -> dict:
    """The smallest paired difference the confirm's own rule can certify at ``n_pairs``.

    Computed by driving the REAL rule (``playoff.paired_stats`` / ``is_conclusive``) over the
    worst-case sample shape — k pairs at d = +1 and the rest at 0 — rather than by quoting an
    arithmetic that could drift from the code. This number is the honest bound on what the confirm
    can be asked: at these budgets the bar is a large FRACTION OF A GAME OUTCOME, not the ~5 pp
    scale of the win-prob edges the leaf trades in, and no affordable ``n`` changes that (a 5 pp
    bar at a per-pair spread of ~0.5 needs several hundred pairs). The confirm is therefore a
    COARSE filter by construction — which is exactly what "reject unless the rollouts insist" is.
    """
    from main.search_dividend.playoff import is_conclusive, paired_stats

    for k in range(1, n_pairs + 1):
        diffs = [1.0] * k + [0.0] * (n_pairs - k)
        mean, se, n = paired_stats(diffs)
        if is_conclusive(mean, se, n):
            return {"n_pairs": n_pairs, "min_pairs_that_must_flip": k,
                    "as_mean_difference": round(k / n_pairs, 4),
                    "note": ("k of n paired rollouts must flip the GAME OUTCOME in the same "
                             "direction; below that the confirm declines")}
    return {"n_pairs": n_pairs, "min_pairs_that_must_flip": None,
            "as_mean_difference": 1.0,
            "note": "no partial pattern clears 2·SE — only a unanimous sweep can"}


def _split(events: Sequence[dict], key: str) -> dict:
    """``key``'s distribution among UPHELD vs REJECTED confirms — the free diagnostic.

    Reported as median + mean + n rather than a test, because the upheld arm is expected to be
    small and a p-value off a handful of points would be a decoration. Where both arms have >= 8
    points a Welch interval on the difference is added and can be read; below that it is ``None``
    and the medians are the whole statement.
    """
    a = [float(e[key]) for e in events if e.get("upheld")]
    b = [float(e[key]) for e in events if not e.get("upheld")]
    out = {"upheld_n": len(a), "rejected_n": len(b),
           "upheld_median": round(st.median(a), 4) if a else None,
           "rejected_median": round(st.median(b), 4) if b else None,
           "upheld_mean": round(st.fmean(a), 4) if a else None,
           "rejected_mean": round(st.fmean(b), 4) if b else None,
           "welch_ci95_on_difference": None}
    if len(a) >= 8 and len(b) >= 8:
        va, vb = st.variance(a), st.variance(b)
        se = (va / len(a) + vb / len(b)) ** 0.5
        d = st.fmean(a) - st.fmean(b)
        out["welch_ci95_on_difference"] = [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)]
        out["excludes_zero"] = bool(abs(d) > 1.96 * se)
    return out


def diagnostic(events: Sequence[dict]) -> dict:
    """WHAT A TRUSTWORTHY OVERRULE LOOKS LIKE — the registered free read.

    Every field here is recorded at the decision the confirm ran on, so this compares the confirms
    that SURVIVED terminal rollouts against those that did not, on features available BEFORE the
    rollouts were spent. A separation would be a cheap surrogate for the expensive stage; a null
    says the leaf's own certification carries no usable signal about its own reliability, which is
    the stronger reading of iteration 2's result rather than a weaker one.
    """
    return {
        "n_events": len(events),
        # The race's own margin between the action it certified and the policy's. If a bigger
        # leaf margin predicted survival, the confirm could be triaged by it.
        "leaf_margin": _split(events, "leaf_margin"),
        "root_win_prob_distance_from_half": _split(
            [{**e, "d": abs(float(e["wp"]) - 0.5)} for e in events], "d"),
        "n_legal": _split(events, "n_legal"),
        "turn": _split(events, "turn"),
        "abs_paired_mean": _split([{**e, "am": abs(float(e["mean"]))} for e in events], "am"),
        "stage_counts": {k: sum(1 for e in events if e.get("stage") == k)
                         for k in sorted({e.get("stage") for e in events})},
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = (argv[0] if argv else "tmp/iter3/shardA.jsonl").split(",")
    out_path = argv[1] if len(argv) > 1 else None

    shards = {}
    cell: List[dict] = []
    seen: Dict[int, str] = {}
    for p in paths:
        p = p.strip()
        rs = R.rows(p)
        name = os.path.basename(p)
        idx = sorted({int(r["game"]) for r in rs})
        shards[name] = {"orientation_games": len(rs),
                        "game_index_range": [idx[0], idx[-1]] if idx else None}
        for r in rs:
            g = int(r["game"])
            if seen.setdefault(g, name) != name:
                raise SystemExit(f"shards {seen[g]} and {name} both played game index {g} — the "
                                 "windows overlap and the paired read would be wrong")
        cell += rs
    if not cell:
        print(f"no rows in {paths}", file=sys.stderr)
        return 1
    sig = {(r.get("arm"), r.get("budget"), r.get("opponent"), r.get("root_strategy"),
            r.get("score_mode")) for r in cell}
    if len(sig) != 1:
        raise SystemExit(f"the shards did not play one cell: {sorted(sig)}")
    # The LEAF, verified in the artifact rather than assumed — the same check iterations 1 and 2
    # publish. A silent fall-back to the scalar value head would be invisible in every counter.
    if {r.get("score_mode") for r in cell} != {"win_prob"} or \
            sum(int(r.get("n_defensive_no_win_prob", 0) or 0) for r in cell):
        raise SystemExit("the cell did not rank on the win-prob head on every decision")

    it3 = R.arm_block(cell, "defensive_1s_contested3s_confirm6", 3.0)
    it3["confirm"] = confirm_block(cell)
    it2_rows = R.rows(ITER2)
    it2 = R.arm_block(it2_rows, "defensive_1s_contested3s (iteration 2)", 3.0) if it2_rows else None
    it1_rows = R.rows(R.ITER1)
    it1 = R.arm_block(it1_rows, "defensive_1s (iteration 1)", None) if it1_rows else None

    hist = {}
    for label, rel, what in R.HISTORICAL:
        hr = R.rows(os.path.join(R.REPO, rel))
        if not hr:
            continue
        u = R.unpaired(hr)
        ps = R.pair_scores(hr)
        m, lo, hi, n = R.mean_ci(list(ps.values()))
        hist[label] = {"what": what, "path": rel, "budget_s": hr[0].get("budget"), **u,
                       "paired_win_rate": round(m, 4) if m is not None else None,
                       "paired_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                       "n_pairs": n, "_pairs": ps}

    def paired_vs(other: dict, note: str) -> dict:
        shared = sorted(set(it3["_pairs"]) & set(other["_pairs"]))
        diffs = [it3["_pairs"][g] - other["_pairs"][g] for g in shared]
        pm, plo, phi, pn = R.mean_ci(diffs)
        return {"n_shared_games": pn,
                "delta": round(pm, 4) if pm is not None else None,
                "ci95": [round(plo, 4), round(phi, 4)] if plo is not None else None,
                "excludes_zero": bool(plo is not None and (plo > 0 or phi < 0)),
                "note": note}

    comparisons = {}
    if it2:
        d, dlo, dhi = R.newcombe(it3["won"], it3["decisive"], it2["won"], it2["decisive"])
        comparisons["vs_iteration2_unpaired"] = {
            "delta": round(d, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
            "excludes_zero": bool(dlo > 0 or dhi < 0)}
        comparisons["vs_iteration2_paired_on_game"] = paired_vs(
            it2, "THE REGISTERED NO-REGRESSION ROW: the two cells differ in exactly one flag "
                 "(--defensive-confirm), and on the shared indices they are the same pinned dice "
                 "and the same team draw")
    if it1:
        comparisons["vs_iteration1_paired_on_game"] = paired_vs(
            it1, "the three-way shared-seed row — iteration 1's whole cell is game indices 0-199")
    bar = hist.get("mirror_honest_1s")
    if bar:
        d, dlo, dhi = R.newcombe(it3["won"], it3["decisive"], bar["won"], bar["decisive"])
        comparisons["vs_honest_1s_unpaired"] = {
            "delta": round(d, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
            "excludes_zero": bool(dlo > 0 or dhi < 0)}
        comparisons["vs_honest_1s_paired_on_game"] = paired_vs(
            bar, "the historical bar — plain depth-1 search at the same budget")

    c = it3["confirm"]
    over3 = it3["rates"]["overrule_of_all"]
    bars = {
        "prediction_overrule_rate_falls": {
            "registered": [0.015, 0.035],
            "iteration2": it2["rates"]["overrule_of_all"] if it2 else None,
            "iteration3": over3,
            "in_range": bool(0.015 <= over3 <= 0.035),
            "fell": bool(it2 and over3 < it2["rates"]["overrule_of_all"]),
            # The race must be UNCHANGED — the confirm is a filter after it, not a different
            # search. If this drifted from iteration 2's 0.0582 the two cells are not comparable.
            "iteration3_PROPOSED_overrule_rate": c["proposed_overrule_of_all"],
            "race_unchanged": bool(
                it2 and abs(c["proposed_overrule_of_all"]
                            - it2["rates"]["overrule_of_all"]) < 0.015),
            "note": ("the registered band was built as iteration 2's 5.82% times a 40-75% "
                     "rejection rate. Probe H had already measured the same filter from the other "
                     "side — the playoff arm's action-change rate collapsed to 0.074 of searched "
                     "decisions under paired rollouts — which implies a much LOWER post-confirm "
                     "rate than the band. Scored as registered either way."),
        },
        "primary_no_regression": {
            "registered": "win rate >= 0.50, and no regression vs iteration 2 on shared seeds",
            "iteration3_paired": it3["paired_win_rate"],
            "iteration3_paired_ci95": it3["paired_ci95"],
            "reaches_null": bool(it3["paired_ci95"] and it3["paired_ci95"][1] >= 0.5),
            "point_at_or_above_null": bool((it3["paired_win_rate"] or 0) >= 0.5),
            "iteration2_paired": it2["paired_win_rate"] if it2 else None,
            "paired_delta_on_shared_games": comparisons.get(
                "vs_iteration2_paired_on_game", {}).get("delta"),
            "regression_excluded": bool(
                (comparisons.get("vs_iteration2_paired_on_game", {}).get("ci95") or [None, None])[1]
                is not None
                and comparisons["vs_iteration2_paired_on_game"]["ci95"][1] > 0),
        },
        "stretch_ci_above_null": {
            "registered": ("paired CI above 0.50 — resolves only if the true rate is >= ~0.525; a "
                           "0.51-ish point estimate is 'real but unresolved', PRE-STATED"),
            "paired_win_rate": it3["paired_win_rate"],
            "paired_ci95": it3["paired_ci95"],
            "paired_ci_above_null": bool(it3["paired_ci95"] and it3["paired_ci95"][0] > 0.5),
            "resolvable_at_this_n": ("a true rate >= "
                                     f"{round(0.5 + (it3['paired_ci_halfwidth'] or 0), 4)} would "
                                     "have cleared 0.50 at this n"),
        },
        "confirmation_rejection_rate": {
            "registered": "report it directly — it is the leaf-bias meter in vivo",
            "attempted": c["attempted"], "confirmed": c["confirmed"],
            "reject_rate": c["reject_rate"],
            "split": {k: c[k] for k in ("reversed", "inconclusive", "no_budget", "error")},
            "leaf_findings_only_reject_rate": (
                round((c["reversed"] + c["inconclusive"]) / c["attempted"], 4)
                if c["attempted"] else None),
            "min_detectable_paired_difference": c["min_detectable_paired_difference"],
        },
        "cost_envelope": {
            "registered": "report the projected and realized s/game",
            "projected_s_per_game": ("25.1 base + 2.2 attempts x 12.2 s = 51.9 (pilot: 12 "
                                     "orientation-games at N=8, 2.03 s per PAIR)"),
            "realized_wall_s_per_game": it3["envelope"]["mean_wall_s_per_game"],
            "realized_confirm_s_per_game": c["s_per_game"],
            "realized_search_s_per_game": it3["envelope"]["search_s_per_game"],
            "iteration2_wall_s_per_game": (it2["envelope"]["mean_wall_s_per_game"]
                                           if it2 else None),
            "uniform_notional_s_per_game": it3["envelope"]["uniform_notional_s_per_game"],
            "banked_frac_of_uniform_SEARCH_only": it3["envelope"]["banked_frac_of_uniform"],
            "confirm_inside_the_bank": bool(
                c["s_per_game"] <= (it3["envelope"]["uniform_notional_s_per_game"]
                                    - it3["envelope"]["search_s_per_game"])),
            "note": ("the SEARCH still banks most of the uniform notional; the CONFIRM does not "
                     "fit inside that bank and is not claimed to — one pair of terminal rollouts "
                     "costs more than the whole 1 s notional of the decision it settles. This "
                     "configuration is an INSTRUMENT for reading the leaf, not a deployable "
                     "ladder setting (iteration 1's uniform 1 s remains that)."),
        },
    }

    out = {
        "date": "2026-08-29",
        "iteration": 3,
        "one_change": (f"--defensive-confirm {CONFIRM_N} --defensive-confirm-deadline-s "
                       f"{CONFIRM_DEADLINE_S:g} — before acting on an overrule, the race's winner "
                       "and the policy's own action are settled by up to 6 PAIRED rollouts to a "
                       "terminal, and the policy's action stands unless the paired difference "
                       "clears 2*SE over >= 4 pairs. Contested deadline 3 s, gate 0.15, seq floor "
                       "5, win-prob leaf, depth 1: ALL UNCHANGED from iteration 2."),
        "checkpoint": ("models/ai_v9_29_rev1_0823/checkpoints/checkpoint_9995088_steps.zip "
                       "(the SAME checkpoint iterations 1-2 and the historical mirror arms "
                       "played)"),
        "invocation": ("python -m main.search_dividend <ckpt> --arm honest --budget 1 "
                       "--root-strategy defensive --defensive-leaf winprob "
                       "--defensive-wp-margin 0.15 --defensive-confirm 6 "
                       "--defensive-confirm-deadline-s 30 --defensive-contested-deadline-s 3.0 "
                       "--games-start <lo> --games <n> --games-seed 7 --opponents self "
                       "--battle-timeout-s 1800 --battle-idle-s 120"),
        "shards": shards,
        "null": 0.5,
        "arm": {k: v for k, v in it3.items() if k not in ("_pairs",)},
        "iteration2": {k: v for k, v in it2.items() if k != "_pairs"} if it2 else None,
        "iteration1": {k: v for k, v in it1.items() if k != "_pairs"} if it1 else None,
        "historical": {k: {kk: vv for kk, vv in v.items() if kk != "_pairs"}
                       for k, v in hist.items()},
        "comparisons": comparisons,
        "bars": bars,
        "confirmed_vs_rejected": diagnostic(c["_events"]),
    }
    out["arm"]["confirm"] = {k: v for k, v in c.items() if k != "_events"}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=False)
            fh.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k != "historical"}, indent=1))
    print("\nHISTORICAL (quoted, not re-run):")
    for k, v in out["historical"].items():
        print(f"  {k:20} n={v['decisive']:<4} wr={v['win_rate']:.4f} {v['ci95']} "
              f"paired={v['paired_win_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
