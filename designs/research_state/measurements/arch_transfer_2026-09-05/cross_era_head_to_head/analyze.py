"""Merge the two sides' per-game records, and read the pre-registered verdict off them.

The two sides are joined on `battle_tag`, which BOTH processes observe for the same battle —
so the join is a fact of the protocol, not an assumption about ordering. Every joined game is
cross-checked: the two sides must report the same turn count, exactly one winner, and the team
each side actually fielded must be the team the plan intended.

Two intervals are reported, because they answer different questions:

* **Game-level Wilson 95%** — the conservative headline, and the one the pre-registration is
  written against.
* **Pair-level cluster bootstrap** — the two games of a pair share their two teams (they are
  the same draw, swapped), so the pair is the independent unit. Resampling PAIRS rather than
  games is the project's standing rule about clustering; a game-level interval that ignores it
  can be too narrow.

Run:
    python analyze.py --side-a <a.jsonl> --side-b <b.jsonl> --plan <plan.json> \
        --out-games games.jsonl --out-summary summary.json
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import argparse
import json
import math
import random


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def elo_from_p(p: float):
    """The rating difference a win rate implies under the ladder's own logistic link."""
    if p <= 0.0 or p >= 1.0:
        return float("inf") * (1 if p >= 1 else -1)
    return -400.0 * math.log10(1.0 / p - 1.0)


def p_from_elo(d: float):
    return 1.0 / (1.0 + 10 ** (-d / 400.0))


def cluster_bootstrap(pairs, n_boot=20000, seed=12345):
    """Resample PAIRS with replacement; each pair contributes its own games."""
    rng = random.Random(seed)
    keys = list(pairs)
    if not keys:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(n_boot):
        wins = games = 0
        for _ in range(len(keys)):
            w, g = pairs[keys[rng.randrange(len(keys))]]
            wins += w
            games += g
        draws.append(wins / games if games else 0.0)
    draws.sort()
    return (draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side-a", required=True, help="JSONL from the plan's side_a")
    ap.add_argument("--side-b", required=True, help="JSONL from the plan's side_b")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-games", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--predicted-p", type=float, default=0.6290,
                    help="the PRE-REGISTERED prediction being tested")
    ap.add_argument("--predicted-elo-delta", type=float, default=91.7)
    ap.add_argument("--label-a", default="v9_59")
    ap.add_argument("--label-b", default="v8_14")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    rows_a = [json.loads(l) for l in open(args.side_a)]
    rows_b = [json.loads(l) for l in open(args.side_b)]
    by_tag_b = {r["battle_tag"]: r for r in rows_b}

    games, anomalies = [], []
    for ra in rows_a:
        rb = by_tag_b.get(ra["battle_tag"])
        if rb is None:
            anomalies.append(f"{ra['battle_tag']}: no side_b record")
            continue
        finished = bool(ra["finished"] and rb["finished"])
        timeout = not finished
        # A DRAW is a semantic outcome; a TIMEOUT is not. Both sides reporting won=None on a
        # FINISHED battle is a genuine tie (a double-KO — the pool's Gengar carries Explosion),
        # and folding it into the timeout bucket would both overstate the timeout rate and hide
        # a real game. It is excluded from the win-rate denominator and counted on its own.
        draw = bool(finished and ra["won"] is None and rb["won"] is None)
        if finished and not draw and (bool(ra["won"]) == bool(rb["won"])):
            anomalies.append(
                f"{ra['battle_tag']}: both sides report won={ra['won']} — dropped")
            continue
        if ra["turns"] != rb["turns"]:
            anomalies.append(
                f"{ra['battle_tag']}: turn mismatch {ra['turns']} vs {rb['turns']}")
        for r, side in ((ra, "a"), (rb, "b")):
            if r["team_match"] is False:
                anomalies.append(
                    f"{r['battle_tag']}: side_{side} fielded a team the plan did not intend")
        games.append({
            "battle_tag": ra["battle_tag"],
            "game_index": ra["game_index"],
            "pair_index": ra["pair_index"],
            "orientation": ra["orientation"],
            f"{args.label_a}_team": ra["intended_team"],
            f"{args.label_a}_sha": ra["intended_sha"],
            f"{args.label_b}_team": rb["intended_team"],
            f"{args.label_b}_sha": rb["intended_sha"],
            "winner": (None if (timeout or draw)
                       else (args.label_b if rb["won"] else args.label_a)),
            "b_won": (None if (timeout or draw) else bool(rb["won"])),
            "turns": ra["turns"],
            "timeout": timeout,
            "draw": draw,
            "team_match_a": ra["team_match"],
            "team_match_b": rb["team_match"],
        })

    with open(args.out_games, "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")

    planned = plan["n_games"]
    attempted = len(games)
    decisive = [g for g in games if not g["timeout"] and not g["draw"]]
    draws = sum(1 for g in games if g["draw"])
    n = len(decisive)
    timeouts = planned - len(games) + sum(1 for g in games if g["timeout"])
    timeout_frac = timeouts / planned if planned else 1.0

    b_wins = sum(1 for g in decisive if g["b_won"])
    p_hat = b_wins / n if n else float("nan")
    lo, hi = wilson(b_wins, n)

    pairs = {}
    for g in decisive:
        w, tot = pairs.get(g["pair_index"], (0, 0))
        pairs[g["pair_index"]] = (w + (1 if g["b_won"] else 0), tot + 1)
    plo, phi = cluster_bootstrap(pairs)
    complete_pairs = {k: v for k, v in pairs.items() if v[1] == 2}
    sweep_b = sum(1 for w, t in complete_pairs.values() if w == 2)
    sweep_a = sum(1 for w, t in complete_pairs.values() if w == 0)
    split = sum(1 for w, t in complete_pairs.values() if w == 1)

    # ── the pre-registered decision rule ──────────────────────────────────────────
    if timeout_frac > 0.25:
        verdict = "INCONCLUSIVE"
        reason = (f"timeouts are {timeout_frac:.1%} of the {planned} planned games, over the "
                  f"25% bar — a timeout is never a semantic outcome, so no verdict is read")
    else:
        excludes_half = (lo > 0.5) or (hi < 0.5)
        excludes_pred = (lo > args.predicted_p) or (hi < args.predicted_p)
        if not excludes_half:
            verdict = "NOT DETECTED"
            reason = (f"the Wilson 95% CI [{lo:.4f},{hi:.4f}] straddles 0.500, so no direction "
                      f"may be claimed")
            if excludes_pred:
                reason += (f"; it also EXCLUDES the pre-registered {args.predicted_p:.4f}, so "
                           f"the anchored ELO is contradicted even though the head-to-head "
                           f"itself cannot name a stronger generation")
        else:
            verdict = "SIGNIFICANT"
            direction = args.label_b if p_hat > 0.5 else args.label_a
            reason = (f"the Wilson 95% CI [{lo:.4f},{hi:.4f}] excludes 0.500 — {direction} is "
                      f"the stronger of the two head to head")
            if excludes_pred:
                reason += (f"; and it EXCLUDES the pre-registered {args.predicted_p:.4f}, so the "
                           f"anchored ELO MIS-PREDICTS")
            else:
                reason += (f"; and it CONTAINS the pre-registered {args.predicted_p:.4f}, so the "
                           f"anchored ELO is corroborated")

    summary = {
        "schema": 1,
        "question": f"P({args.label_b} beats {args.label_a}) head to head",
        "prediction": {
            "source": "anchored Bradley-Terry ladder.json, each run's top node",
            "elo_b": 2049.1, "se_b": 13.7, "elo_a": 1957.4, "se_a": 15.0,
            "elo_delta": args.predicted_elo_delta,
            "predicted_p": args.predicted_p,
            "predicted_p_ci_from_rating_se": [0.5741, 0.6807],
        },
        "counts": {
            "planned": planned, "attempted": attempted, "decisive": n,
            "draws": draws,
            "timeouts": timeouts, "timeout_fraction": timeout_frac,
            f"{args.label_b}_wins": b_wins, f"{args.label_a}_wins": n - b_wins,
        },
        "result": {
            "p_hat": p_hat,
            "wilson95": [lo, hi],
            "pair_cluster_bootstrap95": [plo, phi],
            "implied_elo_delta": elo_from_p(p_hat) if n else None,
            "implied_elo_delta_ci": [elo_from_p(lo), elo_from_p(hi)] if n else None,
        },
        "pairs": {
            "n_complete": len(complete_pairs),
            f"{args.label_b}_sweeps": sweep_b,
            f"{args.label_a}_sweeps": sweep_a,
            "splits": split,
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "anomalies": anomalies,
        "plan": {"seed": plan["team_seq_seed"], "n_teams": plan["n_teams"],
                 "n_pairs": plan["n_pairs"], "team_dir": plan["team_dir"]},
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"planned={planned} decisive={n} draws={draws} "
          f"timeouts={timeouts} ({timeout_frac:.1%})")
    print(f"{args.label_b} wins {b_wins}/{n} = {p_hat:.4f}")
    print(f"  Wilson 95%            [{lo:.4f}, {hi:.4f}]")
    print(f"  pair cluster boot 95% [{plo:.4f}, {phi:.4f}]")
    print(f"  implied ELO delta     {elo_from_p(p_hat):+.1f} "
          f"[{elo_from_p(lo):+.1f}, {elo_from_p(hi):+.1f}]  (predicted {args.predicted_elo_delta:+.1f})")
    print(f"  pairs: {sweep_b} {args.label_b}-sweeps / {split} splits / "
          f"{sweep_a} {args.label_a}-sweeps  (n={len(complete_pairs)})")
    print(f"VERDICT: {verdict} — {reason}")
    if anomalies:
        print(f"ANOMALIES ({len(anomalies)}):")
        for a in anomalies[:20]:
            print("   " + a)


if __name__ == "__main__":
    main()
