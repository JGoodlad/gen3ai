"""H18 — the 2x2, the Newcombe contrasts, and the per-EXPOSURE bias.

Reads the four cells' summary.json + games.jsonl and the four exposure scans, and prints the
table the README carries. `newcombe` is imported from `main.anchors.results` rather than
re-derived, so the difference interval is the same one every other anchor contrast uses.
"""
import json
import math
import os
import sys
from math import comb

sys.path.insert(0, "/home/goodlad/dev/gen3ai-wt/h18/src")
from main.anchors.results import newcombe, wilson  # noqa: E402

T = "/home/goodlad/.claude/jobs/9ab51de6/tmp/h18"
CELLS = {("U", "away"): "cell_U_away", ("P", "away"): "cell_P_away",
         ("U", "home"): "cell_U_home", ("P", "home"): "cell_P_home"}
LABEL = {"away": "away (competitive, 8/20 carry BP)",
         "home": "enriched (20/20 carry BP)"}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. The arms play the SAME battles at the same seeds, so the
    unpaired Newcombe interval throws away the pairing — this is the powered read."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_delta(b, c, n):
    """(c - b) / n with its normal 95% interval. b = U won and P lost; c = U lost and P won."""
    d = (c - b) / n
    se = math.sqrt((b + c - (c - b) ** 2 / n) / n ** 2)
    return d, d - 1.96 * se, d + 1.96 * se


def load(cell):
    s = json.load(open(os.path.join(T, cell, "summary.json")))
    rows = [json.loads(x) for x in open(os.path.join(T, cell, "games.jsonl"))]
    return s, rows


def exposure(cell):
    p = os.path.join(T, f"exposure_{cell}.json")
    return json.load(open(p)) if os.path.exists(p) else None


out = {"cells": {}, "contrasts": {}, "integrity": {}}
print(f"{'arm':<4} {'team set':<34} {'n':>4} {'W/L/T':>12} {'win rate':>9}  Wilson 95%")
for (arm, ts), cell in CELLS.items():
    s, rows = load(cell)
    n = s["n"]
    w = sum(1 for r in rows if r.get("won") is True)
    ll = sum(1 for r in rows if r.get("won") is False)
    t = n - w - ll
    p, lo, hi = wilson(w, n)
    print(f"{arm:<4} {LABEL[ts]:<34} {n:>4} {f'{w}/{ll}/{t}':>12} {p:>9.4f}  [{lo:.3f}, {hi:.3f}]")
    out["cells"][f"{arm}_{ts}"] = {
        "arm": arm, "teamset": ts, "n": n, "wins": w, "losses": ll, "ties": t,
        "win_rate": p, "wilson_lo": lo, "wilson_hi": hi,
        "mean_turns": s.get("mean_turns"), "status": s.get("status"),
        "regime_verified_decisions": s.get("regime_verified_decisions"),
        "peer_clean": s.get("peer_clean"),
        "team_source_asymmetry": s.get("team_source_asymmetry"),
        "hit_forfeit_limit": s.get("hit_forfeit_limit"),
        "n_defaults": s.get("n_defaults"),
        "their_argmax_match_rates": s.get("their_argmax_match_rates"),
        "distinct_our_teams": s.get("distinct_our_teams"),
        "exposure": exposure(cell),
    }

print()
print("CONTRAST  P - U  (our win rate). The BIAS on an unpatched anchor number is -(P - U).")
for ts in ("away", "home"):
    cu, cp = out["cells"][f"U_{ts}"], out["cells"][f"P_{ts}"]
    d, lo, hi = newcombe(cp["wins"], cp["n"], cu["wins"], cu["n"])
    det = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
    print(f"  {LABEL[ts]:<34} P-U = {d:+.4f}  Newcombe 95% [{lo:+.4f}, {hi:+.4f}]  {det}")
    print(f"  {'':<34} bias = {-d:+.4f}  [{-hi:+.4f}, {-lo:+.4f}]")
    out["contrasts"][ts] = {"delta_P_minus_U": d, "lo": lo, "hi": hi, "detected": det == "DETECTED",
                            "bias_on_our_number": -d, "bias_lo": -hi, "bias_hi": -lo}

# the exposure-restricted read: only games in which a boost-carrying pass actually happened
print()
print("PER-EXPOSURE — restricted to games whose captured protocol contains a Baton Pass that")
print("CARRIED a nonzero stat stage (the only games the defect can change).")
for ts in ("away", "home"):
    sub = {}
    for arm in ("U", "P"):
        cell = CELLS[(arm, ts)]
        s, rows = load(cell)
        exp = exposure(cell)
        if exp is None:
            print(f"  {LABEL[ts]:<34} {arm}: no exposure scan")
            sub = None
            break
        hit = {r["tag"] for r in exp["per_battle"] if r["n_baton_pass_with_either"] > 0}
        sel = [r for r in rows if r.get("battle_tag") in hit]
        w = sum(1 for r in sel if r.get("won") is True)
        sub[arm] = (w, len(sel))
        p, lo, hi = wilson(w, len(sel)) if sel else (float("nan"),) * 3
        print(f"  {LABEL[ts]:<34} {arm}: {w}/{len(sel)} = {p:.4f}  [{lo:.3f}, {hi:.3f}]"
              f"   ({exp['games_with_a_state_carrying_pass']}/{exp['n_battles_scanned']} games,"
              f" {exp['baton_pass_events_carrying_any_state']}/{exp['baton_pass_events']} passes)")
    if sub and sub.get("U") and sub.get("P") and sub["U"][1] and sub["P"][1]:
        d, lo, hi = newcombe(sub["P"][0], sub["P"][1], sub["U"][0], sub["U"][1])
        det = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
        print(f"  {'':<34} P-U on exposed games = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {det}")
        out["contrasts"][ts + "_exposed"] = {
            "delta_P_minus_U": d, "lo": lo, "hi": hi, "detected": det == "DETECTED",
            "n_P": sub["P"][1], "n_U": sub["U"][1], "w_P": sub["P"][0], "w_U": sub["U"][0]}

print()
print("PAIRED — the arms play the SAME 400 battles at the SAME seeds, so the contrast is paired.")
print("b = U won and P lost (the patch cost us); c = U lost and P won (the patch gained us).")
for ts in ("away", "home"):
    ru = {r["battle_tag"]: r.get("won") for r in load(CELLS[("U", ts)])[1]}
    rp = {r["battle_tag"]: r.get("won") for r in load(CELLS[("P", ts)])[1]}
    tags = sorted(set(ru) & set(rp))
    n = len(tags)
    b = sum(1 for x in tags if ru[x] and not rp[x])
    c = sum(1 for x in tags if (not ru[x]) and rp[x])
    d, lo, hi = paired_delta(b, c, n)
    pv = mcnemar_exact(b, c)
    det = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
    expU = exposure(CELLS[("U", ts)])
    hit = {r["tag"] for r in expU["per_battle"] if r["n_baton_pass_with_either"] > 0}
    disc = [x for x in tags if ru[x] != rp[x]]
    print(f"  {LABEL[ts]:<34} n={n} concordant={n - b - c} b={b} c={c}")
    print(f"  {'':<34} paired P-U = {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
          f"McNemar exact p={pv:.3f}  {det}")
    print(f"  {'':<34} BIAS on the unpatched number = {-d:+.4f} [{-hi:+.4f}, {-lo:+.4f}]")
    print(f"  {'':<34} of {len(disc)} divergent games, "
          f"{sum(1 for x in disc if x in hit)} carried state over a pass")
    out["contrasts"][ts + "_paired"] = {
        "n": n, "b_U_win_P_loss": b, "c_U_loss_P_win": c, "concordant": n - b - c,
        "delta_P_minus_U": d, "lo": lo, "hi": hi, "mcnemar_exact_p": pv,
        "detected": det == "DETECTED", "bias_on_our_number": -d,
        "bias_lo": -hi, "bias_hi": -lo,
        "n_divergent": len(disc),
        "n_divergent_state_carrying": sum(1 for x in disc if x in hit)}

json.dump(out, open(os.path.join(T, "h18_results.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(T, 'h18_results.json')}")
