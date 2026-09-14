#!/usr/bin/env python3
"""Merge both sides' per-battle records into ``games.jsonl`` and report the tables.

Two independent views of the same games exist and JOINING them is the point — a win rate that
only one side agrees with is not a measurement:

* OURS   — ``games_raw.jsonl`` from ``run_gen3ai_side.py``: our ``battle_tag``, the battle's TURN
  number (``LiveView.turn``), whether our forfeit limit fired, and our decision telemetry.
* THEIRS — Metamon's ``battle_log_<user>_<format>.csv``: its ``Result``, the team file it drew,
  and a ``Turn Count`` that is NOT the battle's turn number but Metamon's own ``turn_counter``
  (env steps taken). The two differ by the decision points a turn can contain (switches after a
  faint, and so on), so the merged row keeps both under distinct names.

🚨 THE JOIN IS POSITIONAL, and it has to be. The CSV's ``Battle ID`` column is NOT the Showdown
battle tag — ``metamon/env/wrappers.py`` generates it as ``"".join(str(random.randint(0, 9)) for _
in range(10))``, a fresh random 10-digit string per battle. There is therefore NO shared key
between the two logs, and the only sound pairing is ORDER: this harness plays strictly
sequentially (our side at ``--concurrency 1``, Metamon's acceptor awaiting one battle at a time in
``ChallengeByUsername._accept_challenge_loop``), so the *n*-th row on each side is the same game.
The check that this held is ``sides_disagree``: under a positional join a mis-alignment shows up
as systematic disagreement about who won, so a zero there is evidence the pairing is right — not
an assumption. A non-zero count is reported and never averaged away.

Disagreement between the two on the WINNER is reported as a defect, never averaged away.

    python3 analyze.py --series SmallRL=<dir> --series SyntheticRLV2=<dir> --out games.jsonl
"""

import argparse
import csv
import glob
import json
import math
import os
import re


def wilson(k: int, n: int, z: float = 1.959963985):
    """Wilson score interval — the right one for a proportion at these n, unlike the normal
    approximation, which misbehaves near 0 and 1 and at small n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def battle_id(tag: str) -> str:
    m = re.search(r"(\d+)$", tag or "")
    return m.group(1) if m else (tag or "")


def load_series(name: str, d: str):
    ours = []
    path = os.path.join(d, "games_raw.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            ours = [json.loads(line) for line in f if line.strip()]
    theirs = []
    for csv_path in sorted(glob.glob(os.path.join(d, "battle_log_*.csv"))):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                theirs.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                               for k, v in row.items()})

    timing_ours = _maybe(os.path.join(d, "our_timing.json"))
    timing_theirs = _maybe(os.path.join(d, "metamon_timing.json"))

    games = []
    for i, r in enumerate(ours):
        bid = battle_id(r.get("battle_tag", ""))
        t = theirs[i] if i < len(theirs) else {}
        their_result = t.get("Result")
        our_won = r.get("won")
        # A tie leaves poke-env's `won` at None with `finished` true.
        our_call = "win" if our_won is True else ("loss" if our_won is False else "tie")
        their_call = {"WIN": "loss", "LOSS": "win", "TIE": "tie"}.get(their_result)
        games.append({
            "series": name,
            "metamon_agent": name,
            "showdown_room": bid,
            "metamon_random_battle_id": t.get("Battle ID") or None,
            "battle_tag": r.get("battle_tag"),
            "our_result": our_call,
            "metamon_says_we": their_call,
            "sides_agree": (their_call is None) or (their_call == our_call),
            "metamon_row_found": bool(t),
            "turns": r.get("turns"),
            "metamon_env_steps": int(t["Turn Count"]) if t.get("Turn Count") else None,
            "metamon_team_file": os.path.basename(t.get("Team File", "")) or None,
            "hit_forfeit_limit": r.get("hit_forfeit_limit"),
            "forfeit_limit": r.get("forfeit_limit"),
            "our_decisions_cumulative": r.get("n_decisions"),
            "our_defaults_cumulative": r.get("n_defaults"),
            "our_redecides_cumulative": r.get("n_redecides"),
            "t": r.get("t"),
        })
    return games, timing_ours, timing_theirs


def _maybe(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def parse_failures(d: str):
    """Protocol-parse deaths on OUR side. `battle_event.classify` raises on an unknown keyword BY
    DESIGN; on a live battle that kills the parse task, sends no choice, and loses on the timer —
    so a traceback naming it is a distinct, countable failure, not noise."""
    out = {"our_tracebacks": 0, "classify_raises": 0, "metamon_tracebacks": 0,
           "our_connection_errors": 0, "signatures": []}
    for fn, key in (("gen3ai.log", "our"), ("metamon.log", "metamon")):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        text = open(p, errors="replace").read()
        n = text.count("Traceback (most recent call last)")
        out[f"{key}_tracebacks"] = n
        if key == "our":
            out["classify_raises"] = len(re.findall(r"classify|UnknownKeyword|KeyError", text))
            out["our_connection_errors"] = text.count("ShowdownConnectionError")
        for m in set(re.findall(r"^\w*(?:Error|Exception)[^\n]{0,120}", text, re.M)):
            out["signatures"].append(f"{key}: {m[:140]}")
    out["signatures"] = sorted(set(out["signatures"]))[:20]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    all_games, summary = [], []
    for spec in args.series:
        name, d = spec.split("=", 1)
        games, t_ours, t_theirs = load_series(name, d)
        all_games.extend(games)
        n = len(games)
        wins = sum(g["our_result"] == "win" for g in games)
        losses = sum(g["our_result"] == "loss" for g in games)
        ties = sum(g["our_result"] == "tie" for g in games)
        p, lo, hi = wilson(wins, n)
        turns = [g["turns"] for g in games if g["turns"] is not None]
        wall = (t_ours or {}).get("wall_s")
        summary.append({
            "series": name,
            "n_games": n,
            "our_wins": wins, "our_losses": losses, "ties": ties,
            "our_win_rate": p, "wilson95_lo": lo, "wilson95_hi": hi,
            "mean_turns": sum(turns) / len(turns) if turns else None,
            "max_turns": max(turns) if turns else None,
            "forfeit_limit_fired": sum(bool(g["hit_forfeit_limit"]) for g in games),
            "sides_disagree": sum(not g["sides_agree"] for g in games),
            "metamon_rows_missing": sum(not g["metamon_row_found"] for g in games),
            "our_defaults_total": (games[-1]["our_defaults_cumulative"] if games else 0),
            "our_redecides_total": (games[-1]["our_redecides_cumulative"] if games else 0),
            "distinct_metamon_teams": len({g["metamon_team_file"] for g in games
                                           if g["metamon_team_file"]}),
            "our_inference": t_ours,
            "metamon_inference": t_theirs,
            "our_wall_s": wall,
            "sec_per_game": (wall / n) if (wall and n) else None,
            "failures": parse_failures(d),
            "series_meta": _maybe(os.path.join(d, "series.json")),
        })

    with open(args.out, "w") as f:
        for g in all_games:
            f.write(json.dumps(g) + "\n")
    text = json.dumps(summary, indent=1)
    print(text)
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
