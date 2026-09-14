#!/usr/bin/env python3
"""Merge every half-cell into ``games.jsonl`` and compute the 2x2 tables.

Two independent views of the same games exist and JOINING them is the point — a win rate that only
one side agrees with is not a measurement:

* OURS   — ``games_raw.jsonl`` from ``run_gen3ai_side.py``: our ``battle_tag``, the battle's TURN
  number (``LiveView.turn``), whether our forfeit limit fired, our decision telemetry, the regime
  our player actually used, and the team we drew.
* THEIRS — Metamon's ``battle_log_<user>_<format>.csv``: its ``Result``, the team file it drew, and
  a ``Turn Count`` that is NOT the battle's turn number but Metamon's own ``turn_counter`` (env
  steps). The merged row keeps both under distinct names.

🚨 THE JOIN IS POSITIONAL, and it has to be (de-risk H3). The CSV's ``Battle ID`` is a fresh random
10-digit string per battle, not the Showdown room, so there is no shared key. The harness plays
strictly sequentially on both sides, so the n-th row on each side is the same game; the CHECK that
this held is ``sides_disagree``, which goes systematically non-zero under a misalignment. A
non-zero count is reported, never averaged away.

Intervals: **Wilson** for a single proportion, **Newcombe's hybrid-score interval** for the
difference of two independent proportions — the difference-of-Wilsons that pairs with the cell
intervals rather than a normal approximation that misbehaves at these n.

    python3 analyze.py --root <dir> --out games.jsonl --summary-out summary.json
"""

import argparse
import csv
import glob
import json
import math
import os
import re

Z = 1.959963985


def wilson(k: int, n: int, z: float = Z):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1, n1, k2, n2):
    """95% CI for p1 - p2, Newcombe's method 10 (hybrid score / 'square-and-add' Wilson).

    Chosen over the Wald interval because the cell proportions are themselves reported as Wilson
    intervals: this is the difference interval that is consistent with them, and it keeps sane
    coverage at n = 100 and near the boundaries.
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, max(-1.0, lo), min(1.0, hi))


def battle_id(tag: str) -> str:
    m = re.search(r"(\d+)$", tag or "")
    return m.group(1) if m else (tag or "")


def _maybe(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def parse_failures(d: str):
    """Protocol-parse deaths on either side. ``battle_event.classify`` raises on an unknown
    keyword BY DESIGN; on a live battle that kills the parse task, sends no choice, and loses on
    the timer — so a traceback naming it is a distinct, countable failure, not noise."""
    out = {"our_tracebacks": 0, "classify_raises": 0, "metamon_tracebacks": 0,
           "our_connection_errors": 0, "team_rejected_popups": 0, "signatures": []}
    for fn, key in (("gen3ai.log", "our"), ("metamon.log", "metamon")):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        text = open(p, errors="replace").read()
        out[f"{key}_tracebacks"] = text.count("Traceback (most recent call last)")
        out["team_rejected_popups"] += text.count("Your team was rejected")
        if key == "our":
            out["classify_raises"] = len(re.findall(r"classify|UnknownKeyword|KeyError", text))
            out["our_connection_errors"] = text.count("ShowdownConnectionError")
        for m in set(re.findall(r"^\w*(?:Error|Exception)[^\n]{0,120}", text, re.M)):
            out["signatures"].append(f"{key}: {m[:140]}")
    out["signatures"] = sorted(set(out["signatures"]))[:20]
    return out


def load_halfcell(d: str):
    """One half-cell directory -> (games, meta). Everything a cell knows about itself is in
    ``cell.json``; nothing is inferred from the directory name."""
    meta = _maybe(os.path.join(d, "cell.json")) or {}
    ours = []
    p = os.path.join(d, "games_raw.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            ours = [json.loads(line) for line in f if line.strip()]
    theirs = []
    for csv_path in sorted(glob.glob(os.path.join(d, "battle_log_*.csv"))):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                theirs.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                               for k, v in row.items()})

    games = []
    for i, r in enumerate(ours):
        t = theirs[i] if i < len(theirs) else {}
        their_result = t.get("Result")
        our_won = r.get("won")
        our_call = "win" if our_won is True else ("loss" if our_won is False else "tie")
        their_call = {"WIN": "loss", "LOSS": "win", "TIE": "tie"}.get(their_result)
        games.append({
            "model": meta.get("agent"),
            "regime": meta.get("regime"),
            "team_set": meta.get("team_set"),
            "challenger": meta.get("challenger"),
            "cell": f"{meta.get('agent')}|{meta.get('regime')}|{meta.get('team_set')}",
            "game_in_halfcell": i + 1,
            "showdown_room": battle_id(r.get("battle_tag", "")),
            "battle_tag": r.get("battle_tag"),
            "metamon_random_battle_id": t.get("Battle ID") or None,
            "our_result": our_call,
            "metamon_says_we": their_call,
            "sides_agree": (their_call is None) or (their_call == our_call),
            "metamon_row_found": bool(t),
            "turns": r.get("turns"),
            "metamon_env_steps": int(t["Turn Count"]) if t.get("Turn Count") else None,
            "our_team": r.get("our_team"),
            "metamon_team": os.path.basename(t.get("Team File", "")) or None,
            "our_team_draws_so_far": r.get("our_team_draws_so_far"),
            "our_battles_so_far": r.get("our_battles_so_far"),
            "hit_forfeit_limit": r.get("hit_forfeit_limit"),
            "forfeit_limit": r.get("forfeit_limit"),
            "our_stochastic": r.get("our_stochastic"),
            "our_temperature": r.get("our_temperature"),
            "our_decisions_cumulative": r.get("n_decisions"),
            "our_defaults_cumulative": r.get("n_defaults"),
            "our_redecides_cumulative": r.get("n_redecides"),
            "t": r.get("t"),
        })
    meta = dict(meta)
    meta["dir"] = d
    meta["our_timing"] = _maybe(os.path.join(d, "our_timing.json"))
    meta["metamon_timing"] = _maybe(os.path.join(d, "metamon_timing.json"))
    meta["failures"] = parse_failures(d)
    meta["n_games"] = len(games)
    return games, meta


def _excl_timeouts(games):
    """Win rate with the 250-turn forfeits and the phantom (<=2-turn) battles REMOVED.

    Reported beside the headline, never instead of it: the headline counts every game the way the
    de-risk did, and this says whether the two events that are not semantic outcomes moved it.
    """
    keep = [g for g in games
            if not g["hit_forfeit_limit"] and (g["turns"] or 99) > 2]
    k = sum(g["our_result"] == "win" for g in keep)
    p, lo, hi = wilson(k, len(keep))
    return {"n": len(keep), "wins": k, "win_rate": p, "lo": lo, "hi": hi,
            "n_removed": len(games) - len(keep)}


def cell_stats(games):
    n = len(games)
    wins = sum(g["our_result"] == "win" for g in games)
    losses = sum(g["our_result"] == "loss" for g in games)
    ties = sum(g["our_result"] == "tie" for g in games)
    p, lo, hi = wilson(wins, n)
    turns = [g["turns"] for g in games if g["turns"] is not None]
    return {
        "n": n, "wins": wins, "losses": losses, "ties": ties,
        "win_rate": p, "wilson95_lo": lo, "wilson95_hi": hi,
        "mean_turns": (sum(turns) / len(turns)) if turns else None,
        "max_turns": max(turns) if turns else None,
        "forfeit_limit_fired": sum(bool(g["hit_forfeit_limit"]) for g in games),
        # A battle that "ended" on turn 1 or 2 is not a game. It is the downstream artifact of
        # Metamon's env force-resetting after our 250-turn forfeit (see the README's H-B), and it
        # is bucketed, never averaged in as a result — UNDERSTANDING rule 12: a timeout is never a
        # semantic outcome, and neither is its wreckage.
        "phantom_games": sum(1 for g in games if (g["turns"] or 99) <= 2),
        "sensitivity_excl_timeouts": _excl_timeouts(games),
        "sides_disagree": sum(not g["sides_agree"] for g in games),
        "metamon_rows_missing": sum(not g["metamon_row_found"] for g in games),
        "distinct_our_teams": len({g["our_team"] for g in games if g["our_team"]}),
        "distinct_metamon_teams": len({g["metamon_team"] for g in games if g["metamon_team"]}),
        "distinct_team_pairs": len({(g["our_team"], g["metamon_team"]) for g in games}),
        # poke-env draws the NEXT battle's team before the current one's finished-callback runs,
        # so the draw counter legitimately leads the battle counter by ONE. Anything else means a
        # draw was consumed without producing a battle — a rejected team, i.e. the de-risk H1 /
        # trailing-blank-line stall class — and the per-game team labels would be shifted.
        "draw_alignment_ok": all(g["our_team_draws_so_far"] - g["our_battles_so_far"] in (0, 1)
                                 for g in games if g["our_team_draws_so_far"] is not None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding the half-cell directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                  if os.path.isfile(os.path.join(d, "cell.json")))
    all_games, metas = [], []
    for d in dirs:
        g, m = load_halfcell(d)
        all_games.extend(g)
        metas.append(m)

    with open(args.out, "w") as f:
        for g in all_games:
            f.write(json.dumps(g) + "\n")

    models = sorted({g["model"] for g in all_games if g["model"]})
    summary = {"halfcells": metas, "models": {}}
    for model in models:
        mg = [g for g in all_games if g["model"] == model]
        cells = {}
        for regime in ("greedy", "t1"):
            for teams in ("home", "away"):
                sel = [g for g in mg if g["regime"] == regime and g["team_set"] == teams]
                if sel:
                    cells[f"{regime}|{teams}"] = cell_stats(sel) | {
                        "by_role": {role: cell_stats([g for g in sel
                                                      if g["challenger"] == role])
                                    for role in sorted({g["challenger"] for g in sel})}}

        def k_n(regime=None, teams=None):
            sel = [g for g in mg
                   if (regime is None or g["regime"] == regime)
                   and (teams is None or g["team_set"] == teams)]
            return sum(g["our_result"] == "win" for g in sel), len(sel)

        effects = {}
        # Temperature effect: greedy - t1, both sides matched, at each team set and pooled.
        for teams in ("home", "away", None):
            a, b = k_n("greedy", teams), k_n("t1", teams)
            d, lo, hi = newcombe(*a, *b)
            effects[f"temperature_greedy_minus_t1@{teams or 'pooled'}"] = {
                "delta": d, "lo": lo, "hi": hi, "n_greedy": a[1], "n_t1": b[1]}
        # Team-set effect: home - away, at each regime and pooled.
        for regime in ("greedy", "t1", None):
            a, b = k_n(regime, "home"), k_n(regime, "away")
            d, lo, hi = newcombe(*a, *b)
            effects[f"teamset_home_minus_away@{regime or 'pooled'}"] = {
                "delta": d, "lo": lo, "hi": hi, "n_home": a[1], "n_away": b[1]}
        # Role effect, the balancing factor: ours-challenging minus metamon-challenging.
        ours = [g for g in mg if g["challenger"] == "ours"]
        them = [g for g in mg if g["challenger"] == "metamon"]
        d, lo, hi = newcombe(sum(g["our_result"] == "win" for g in ours), len(ours),
                             sum(g["our_result"] == "win" for g in them), len(them))
        effects["role_ourschallenge_minus_metamonchallenge"] = {
            "delta": d, "lo": lo, "hi": hi, "n_ours": len(ours), "n_metamon": len(them)}

        primary = cells.get("greedy|away", {})
        summary["models"][model] = {
            "cells": cells,
            "effects": effects,
            "primary_row": "greedy|away",
            "primary": primary,
            # THE registered bar, evaluated mechanically: Wilson 95% lower bound > 0.50.
            "primary_verdict": (
                "BETTER (Wilson 95% LB > 0.50)"
                if primary.get("wilson95_lo", 0) > 0.50 else "NOT DETECTED"),
        }

    text = json.dumps(summary, indent=1)
    print(text)
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
