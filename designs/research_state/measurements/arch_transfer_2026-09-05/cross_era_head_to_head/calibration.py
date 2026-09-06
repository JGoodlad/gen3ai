"""The BOT CALIBRATION table.

Each side's win rate against each shared anchor bot, and the model rating that win rate implies
given the bot's own anchored rating:

    implied_elo(model) = elo(bot) + 400 * log10(p / (1 - p))

This is the check on the head-to-head's prediction. The ladder ratings under test (2049.1 /
1957.4) were fitted from exactly this kind of evidence, so if the implied ratings here disagree
with the ladder, the disagreement is in the *anchor extrapolation*; if they agree, the ladder's
reading of these two checkpoints is reproduced in the same session, on the same server, as the
head-to-head itself.

⚠️ The implied rating is a POINT-TO-POINT reading off one bot, not a Bradley-Terry fit over a
whole ladder, and it inherits the very extrapolation weakness that motivated the head-to-head:
at a ~90% win rate a handful of games moves it by tens of points. Read the head-to-head as the
measurement and this table as the cross-check.

Run:
    python calibration.py --out bot_calibration.json
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import argparse
import json
import math

ANCHORS_PATH = "/home/goodlad/dev/gen3ai/data/gen3_bot_elo_anchors.json"
LADDER_ELO = {"v8_14": 2049.1, "v9_59": 1957.4}
BOTS = ("heuristic2", "staller_v2", "aggressive_v2")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def implied_elo(bot_elo, p):
    if p <= 0.0 or p >= 1.0:
        return float("nan")
    return bot_elo + 400.0 * math.log10(p / (1.0 - p))


# 400/ln(10): the rating scale's derivative w.r.t. the logit.
_ELO_PER_LOGIT = 400.0 / math.log(10.0)


def implied_elo_se(p, n):
    """Delta-method standard error of an implied rating.

    ``implied = E + (400/ln10) * logit(p)`` and ``var(logit p) = 1 / (n p (1-p))``, so the
    rating's se blows up as p approaches 1 — which is the whole reason a 90%-win-rate anchor is
    a weak instrument for a rating. Reporting the point estimate without this is what makes an
    anchor extrapolation look more precise than it is.
    """
    if n <= 0 or p <= 0.0 or p >= 1.0:
        return float("nan")
    return _ELO_PER_LOGIT * math.sqrt(1.0 / (n * p * (1.0 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-jsonl", required=True)
    ap.add_argument("--v9-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h2h-gap", type=float, default=94.9,
                    help="the head-to-head's measured rating gap, for the comparison line")
    args = ap.parse_args()

    anchors = json.load(open(ANCHORS_PATH))["ratings"]
    paths = {"v8_14": args.v8_jsonl, "v9_59": args.v9_jsonl}

    tallies = {}
    for label, path in paths.items():
        per = {}
        for line in open(path):
            r = json.loads(line)
            w, n, t = per.get(r["bot"], (0, 0, 0))
            per[r["bot"]] = (
                w + (1 if r["won"] else 0),
                n + (1 if r["finished"] else 0),
                t + (0 if r["finished"] else 1),
            )
        tallies[label] = per

    out = {"schema": 1,
           "anchors": {b: anchors[b] for b in BOTS},
           "ladder_elo": LADDER_ELO,
           "sides": {}}

    hdr = f"{'bot':<14} {'anchor':>7} |"
    for label in ("v8_14", "v9_59"):
        hdr += f" {label + ' win rate':>30} {'implied':>8} |"
    print(hdr)
    print("-" * len(hdr))

    for bot in BOTS:
        line = f"{bot:<14} {anchors[bot]:>7.1f} |"
        for label in ("v8_14", "v9_59"):
            w, n, t = tallies[label][bot]
            p = w / n
            lo, hi = wilson(w, n)
            line += f" {w:>3}/{n:<3} {p:.3f} [{lo:.3f},{hi:.3f}] {implied_elo(anchors[bot], p):>8.0f} |"
            out["sides"].setdefault(label, {})[bot] = {
                "won": w, "games": n, "timeouts": t, "win_rate": p,
                "wilson95": [lo, hi],
                "anchor_elo": anchors[bot],
                "implied_elo": implied_elo(anchors[bot], p),
                "implied_elo_ci": [implied_elo(anchors[bot], lo),
                                   implied_elo(anchors[bot], hi)],
            }
        print(line)

    print("-" * len(hdr))
    for label in ("v8_14", "v9_59"):
        W = sum(tallies[label][b][0] for b in BOTS)
        N = sum(tallies[label][b][1] for b in BOTS)
        lo, hi = wilson(W, N)
        # Each bot is one independent anchor, so the per-bot implied ratings are averaged
        # with equal weight rather than pooling the games (which would weight by how often
        # each bot happened to be played).
        imps, ses = [], []
        for b in BOTS:
            w_b, n_b, _ = tallies[label][b]
            p_b = w_b / n_b
            imps.append(implied_elo(anchors[b], p_b))
            ses.append(implied_elo_se(p_b, n_b))
        mean_imp = sum(imps) / len(imps)
        # The three bots are independent measurements, so the mean's se is the RSS over 3.
        mean_se = math.sqrt(sum(s * s for s in ses)) / len(ses)
        out["sides"][label]["_pooled"] = {
            "won": W, "games": N, "win_rate": W / N, "wilson95": [lo, hi],
            "mean_implied_elo": mean_imp,
            "mean_implied_elo_se": mean_se,
            "mean_implied_elo_ci95": [mean_imp - 1.96 * mean_se, mean_imp + 1.96 * mean_se],
            "ladder_elo": LADDER_ELO[label],
            "implied_minus_ladder": mean_imp - LADDER_ELO[label],
        }
        print(f"{label}: pooled {W}/{N} = {W/N:.4f} [{lo:.4f},{hi:.4f}]   "
              f"mean implied ELO {mean_imp:.0f} +/- {1.96*mean_se:.0f}   "
              f"ladder {LADDER_ELO[label]:.0f}   "
              f"(implied - ladder = {mean_imp - LADDER_ELO[label]:+.0f})")

    p8 = out["sides"]["v8_14"]["_pooled"]
    p9 = out["sides"]["v9_59"]["_pooled"]
    gap = p8["mean_implied_elo"] - p9["mean_implied_elo"]
    gap_se = math.sqrt(p8["mean_implied_elo_se"] ** 2 + p9["mean_implied_elo_se"] ** 2)
    gap_ci = [gap - 1.96 * gap_se, gap + 1.96 * gap_se]
    ladder_gap = LADDER_ELO["v8_14"] - LADDER_ELO["v9_59"]
    out["implied_gap_v8_minus_v9"] = gap
    out["implied_gap_se"] = gap_se
    out["implied_gap_ci95"] = gap_ci
    out["ladder_gap_v8_minus_v9"] = ladder_gap
    out["head_to_head_measured_gap"] = args.h2h_gap
    out["gap_ci_contains_ladder"] = gap_ci[0] <= ladder_gap <= gap_ci[1]
    out["gap_ci_contains_h2h"] = gap_ci[0] <= args.h2h_gap <= gap_ci[1]
    out["gap_ci_contains_zero"] = gap_ci[0] <= 0.0 <= gap_ci[1]
    print(f"\nGAP (v8_14 - v9_59):")
    print(f"   bot-implied   {gap:+.0f}  95% [{gap_ci[0]:+.0f}, {gap_ci[1]:+.0f}]")
    print(f"   ladder        {ladder_gap:+.1f}   -> inside the bot-implied CI: "
          f"{out['gap_ci_contains_ladder']}")
    print(f"   head-to-head  {args.h2h_gap:+.1f}   -> inside the bot-implied CI: "
          f"{out['gap_ci_contains_h2h']}")
    print(f"   the bot-implied CI also contains ZERO: {out['gap_ci_contains_zero']} "
          f"-- i.e. 720 bot games cannot even establish which model is better")

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
