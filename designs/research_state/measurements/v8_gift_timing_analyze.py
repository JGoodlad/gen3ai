"""v8 GIFT TIMING — analysis: per-cell win rates, per-team gains, the two registered bars.

Reads the `rows_<arm>_s0of1_cells.jsonl` files written by `v8_gift_timing_probe.py` and emits the
committed `.json` + the markdown tables. Pure arithmetic over recorded outcomes — no model, no
battle — so every table below is re-derivable from the committed cells without replaying anything.

STATISTICS (fixed in the pre-registration, before data):
  * per-team WR = wins / finished over that team's 8 opponents x N games
  * gain(arm, team) = WR(arm, team) - WR(parent, team), on the SAME battles (CRN)
  * headline = EQUAL-WEIGHT MEAN over the 16 teams  (probe P's convention)
  * primary interval = CLUSTER BOOTSTRAP OVER TEAMS, 8000 resamples, 95% percentile
  * a pooled battle-level (binomial) interval is reported BESIDE it and is the WEAKER one --
    it ignores team clustering, which is the wrong unit for a claim about teams
  * P2's difference interval is PAIRED at the team level: resample teams, and within a
    resample take gain_final(team) - gain_mid(team)

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):
  python v8_gift_timing_analyze.py <cells_dir> <out_json> [<out_md_tables>]
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys

ARM_ORDER = ["parent", "c278672", "c280656", "c283636", "c287136", "c290116",
             "c291106", "c292101", "foldfinal"]
ARM_STEP = {"parent": 277_583_267, "c278672": 278_671_945, "c280656": 280_656_375,
            "c283636": 283_635_665, "c287136": 287_136_098, "c290116": 290_115_536,
            "c291106": 291_106_373, "c292101": 292_100_648, "foldfinal": 292_623_779}
# arms added AFTER the registered grid returned, to localise the late decline. Exploratory.
FOLLOWUP = {"c291106", "c292101"}
FORK = 277_583_267
MID_ARM = "c283636"          # registered before data
SENSITIVITY_ARM = "c287136"  # reported as a sensitivity only, never as the verdict
LAST_ARM = "foldfinal"
FIRST_ARM = "c278672"
N_BOOT = 8000
SEED = 20260901


def load(cells_dir: str) -> dict:
    arms: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(cells_dir, "rows_*_cells.jsonl"))):
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            a = arms.setdefault(d["arm"], {})
            # a cell may appear twice only if a resume re-ran it; last write wins
            a[(d["team"], d["opp"])] = d
    return arms


def team_stats(cells: dict) -> dict:
    """team -> {wins, finished, requested, opps, arch, per_game (concatenated, opp-ordered)}"""
    out: dict[str, dict] = {}
    for (team, opp), d in sorted(cells.items()):
        t = out.setdefault(team, {"wins": 0, "finished": 0, "requested": 0, "opps": [],
                                  "arch": d.get("arch"), "per_game": {}})
        # counted from `per_game`, not the cell's `wins` counter: a battle that raised (recorded
        # as -1) must leave the DENOMINATOR too, and only the per-game vector can do that.
        t["wins"] += sum(1 for x in d["per_game"] if x == 1)
        t["finished"] += sum(1 for x in d["per_game"] if x in (0, 1))
        t["requested"] += d["requested"]
        t["opps"].append(opp)
        t["per_game"][opp] = d["per_game"]
    return out


def boot_mean(vals: list[float], n=N_BOOT, seed=SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(vals)
    if k == 0:
        return (float("nan"), float("nan"))
    xs = []
    for _ in range(n):
        s = sum(vals[rng.randrange(k)] for _ in range(k))
        xs.append(s / k)
    xs.sort()
    return (xs[int(0.025 * n)], xs[int(0.975 * n) - 1])


def boot_paired(pairs: list[tuple[float, float]], n=N_BOOT, seed=SEED + 1):
    """CI on mean(a - b) resampling the PAIRS (teams), so the two arms move together."""
    rng = random.Random(seed)
    k = len(pairs)
    xs = []
    for _ in range(n):
        s = 0.0
        for _ in range(k):
            a, b = pairs[rng.randrange(k)]
            s += a - b
        xs.append(s / k)
    xs.sort()
    return (xs[int(0.025 * n)], xs[int(0.975 * n) - 1])


def binom_diff_ci(k1, n1, k0, n0, z=1.96):
    if not n1 or not n0:
        return (float("nan"), float("nan"), float("nan"))
    p1, p0 = k1 / n1, k0 / n0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    d = p1 - p0
    return (d - z * se, d + z * se, (d / se) if se else float("nan"))


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    cells_dir = argv[0]
    out_json = argv[1]
    out_md = argv[2] if len(argv) > 2 else None

    arms = load(cells_dir)
    stats = {a: team_stats(c) for a, c in arms.items()}
    if "parent" not in stats:
        raise SystemExit("[vt-an] no parent arm — the baseline is required")

    teams = sorted(stats["parent"])
    par = stats["parent"]
    res = {"_meta": {"cells_dir": cells_dir, "n_boot": N_BOOT, "boot_seed": SEED,
                     "mid_arm_registered": MID_ARM, "sensitivity_arm": SENSITIVITY_ARM,
                     "arm_steps": ARM_STEP, "fork_step": FORK,
                     "unit_of_clustering": "team",
                     "headline_estimator": "equal-weight mean over teams (probe P's convention)"},
           "arms": {}, "per_team": {}, "bars": {}}

    # only teams present in EVERY completed arm enter the comparison
    common = set(teams)
    for a, s in stats.items():
        common &= set(s)
    common = sorted(common)
    res["_meta"]["teams_scored"] = common
    res["_meta"]["n_teams"] = len(common)

    gains: dict[str, list[float]] = {}
    for a in ARM_ORDER:
        if a not in stats:
            continue
        s = stats[a]
        rows = []
        gl = []
        for t in common:
            w, f, r = s[t]["wins"], s[t]["finished"], s[t]["requested"]
            pw, pf = par[t]["wins"], par[t]["finished"]
            wr = w / f if f else float("nan")
            pwr = pw / pf if pf else float("nan")
            g = wr - pwr
            gl.append(g)
            rows.append({"team": t, "arch": s[t]["arch"], "wins": w, "finished": f,
                         "requested": r, "wr": round(wr, 5),
                         "parent_wr": round(pwr, 5), "gain": round(g, 5),
                         "short": r - f})
        gains[a] = gl
        tot_w = sum(x["wins"] for x in rows)
        tot_f = sum(x["finished"] for x in rows)
        tot_r = sum(x["requested"] for x in rows)
        pw = sum(par[t]["wins"] for t in common)
        pf = sum(par[t]["finished"] for t in common)
        lo, hi = boot_mean(gl)
        blo, bhi, z = binom_diff_ci(tot_w, tot_f, pw, pf)
        res["arms"][a] = {
            "step": ARM_STEP[a], "fold_delta": ARM_STEP[a] - FORK,
            "battles": tot_f, "requested": tot_r, "short": tot_r - tot_f,
            "pooled_wr": round(tot_w / tot_f, 5) if tot_f else None,
            "parent_pooled_wr": round(pw / pf, 5) if pf else None,
            "gain_mean_over_teams": round(sum(gl) / len(gl), 5) if gl else None,
            "gain_cluster_ci95": [round(lo, 5), round(hi, 5)],
            "gain_pooled_binomial_ci95": [round(blo, 5), round(bhi, 5)],
            "pooled_z": round(z, 3) if z == z else None,
            "teams_positive": sum(1 for g in gl if g > 0),
            "n_teams": len(gl),
        }
        res["per_team"][a] = rows

    # ------------------------------------------------------------------ the two registered bars
    if FIRST_ARM in gains and LAST_ARM in gains:
        gf = sum(gains[LAST_ARM]) / len(gains[LAST_ARM])
        half = 0.5 * gf
        lo, hi = res["arms"][FIRST_ARM]["gain_cluster_ci95"]
        verdict = ("PASS" if hi < half else "FAIL" if lo > half else "UNDECIDED")
        res["bars"]["P1"] = {
            "statement": ("at least half of the final untaught gain is ABSENT at the first "
                          "retained checkpoint (+1.089M fold steps)"),
            "final_gain": round(gf, 5), "half_gain_line": round(half, 5),
            "first_gain": res["arms"][FIRST_ARM]["gain_mean_over_teams"],
            "first_gain_ci95": [lo, hi], "verdict": verdict,
            "rule": ("PASS iff the whole CI lies BELOW half the final gain; FAIL iff wholly "
                     "above; UNDECIDED iff it straddles"),
        }
        # SUPPLEMENTARY, not the registered bar. The registered rule bootstraps gain(first)
        # alone and treats 0.5*gain(final) as a fixed constant, which throws away the fact that
        # the two arms share the same battles. Pairing the two per team is strictly sharper and
        # is reported BESIDE the registered verdict, never in place of it.
        pairs = [(g1, 0.5 * g2) for g1, g2 in zip(gains[FIRST_ARM], gains[LAST_ARM])]
        d = sum(a - b for a, b in pairs) / len(pairs)
        plo, phi = boot_paired(pairs, seed=SEED + 2)
        res["bars"]["P1_paired_supplementary"] = {
            "statement": "gain(first) - 0.5*gain(final), paired over teams",
            "diff": round(d, 5), "diff_paired_ci95": [round(plo, 5), round(phi, 5)],
            "reads": ("BELOW zero => at least half the final gain is absent early (P1's claim); "
                      "ABOVE zero => more than half is already present"),
            "verdict": ("supports P1" if phi < 0 else
                        "refutes P1" if plo > 0 else "UNDECIDED"),
            "note": "SUPPLEMENTARY — sharper than the registered rule, but not the registered rule",
        }
    for tag, mid in (("P2", MID_ARM), ("P2_sensitivity", SENSITIVITY_ARM)):
        if mid in gains and LAST_ARM in gains:
            pairs = list(zip(gains[LAST_ARM], gains[mid]))
            d = sum(a - b for a, b in pairs) / len(pairs)
            lo, hi = boot_paired(pairs)
            verdict = "PASS" if lo > 0 else "FAIL" if hi < 0 else "UNDECIDED"
            res["bars"][tag] = {
                "statement": (f"gain(final) - gain({mid}) > 0 -- the gift is still rising in the "
                              f"second half"),
                "mid_arm": mid, "mid_fold_delta": ARM_STEP[mid] - FORK,
                "diff": round(d, 5), "diff_paired_ci95": [round(lo, 5), round(hi, 5)],
                "verdict": verdict,
                "rule": "PASS iff the paired CI lies wholly ABOVE zero; FAIL iff wholly below",
                "note": ("REGISTERED bar" if tag == "P2"
                         else "SENSITIVITY only -- not the verdict"),
            }

    # SHAPE — where the curve peaks, and whether the endpoint is below it. Descriptive; the peak
    # arm is SELECTED by being the maximum, so its own interval is upward-biased and the
    # peak-vs-endpoint contrast below is the honest statistic (it is paired, but still selected).
    curve = [(a, res["arms"][a]["gain_mean_over_teams"]) for a in ARM_ORDER
             if a in res["arms"] and a != "parent"]
    if curve and LAST_ARM in gains:
        peak_arm = max(curve, key=lambda kv: kv[1])[0]
        pairs = list(zip(gains[LAST_ARM], gains[peak_arm]))
        d = sum(a - b for a, b in pairs) / len(pairs)
        lo, hi = boot_paired(pairs, seed=SEED + 3)
        res["shape"] = {
            "curve": curve,
            "peak_arm": peak_arm, "peak_fold_delta": ARM_STEP[peak_arm] - FORK,
            "endpoint_minus_peak": round(d, 5),
            "endpoint_minus_peak_paired_ci95": [round(lo, 5), round(hi, 5)],
            "caveat": ("the peak arm is SELECTED as the maximum over the scored grid, so its own "
                       "gain is upward-biased by selection; this contrast is paired but not "
                       "selection-corrected and is DESCRIPTIVE, not a registered test"),
            "followup_arms": sorted(FOLLOWUP & set(res["arms"])),
        }
        # P1 RE-ASKED AGAINST THE PEAK instead of the endpoint. POST-HOC — the pre-registration
        # fixed the endpoint as the denominator, and this is reported only because the curve turned
        # out to be humped, which makes "the final gain" and "the gain the fold reached" two
        # different quantities. It is NOT the registered bar and does not replace its verdict.
        if FIRST_ARM in gains:
            pk = [(g1, 0.5 * g2) for g1, g2 in zip(gains[FIRST_ARM], gains[peak_arm])]
            d2 = sum(a - b for a, b in pk) / len(pk)
            lo2, hi2 = boot_paired(pk, seed=SEED + 4)
            res["bars"]["P1_vs_peak_posthoc"] = {
                "statement": f"gain(first) - 0.5*gain({peak_arm}, the PEAK), paired over teams",
                "diff": round(d2, 5), "diff_paired_ci95": [round(lo2, 5), round(hi2, 5)],
                "verdict": ("supports 'half absent early'" if hi2 < 0 else
                            "refutes it" if lo2 > 0 else "UNDECIDED"),
                "note": ("POST-HOC — the registered denominator is the ENDPOINT, not the peak; the "
                         "peak arm is additionally selected, biasing this contrast toward "
                         "'supports'"),
            }

    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)

    lines = []
    lines.append("| arm | fold Δ | battles | pooled WR | gain vs parent (mean over teams) "
                 "| cluster-boot 95% | pooled binomial 95% | z | teams + |")
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|")
    for a in ARM_ORDER:
        if a not in res["arms"]:
            continue
        r = res["arms"][a]
        g = "—" if a == "parent" else f"**{100*r['gain_mean_over_teams']:+.2f}pp**"
        ci = ("—" if a == "parent" else
              f"[{100*r['gain_cluster_ci95'][0]:+.2f}, {100*r['gain_cluster_ci95'][1]:+.2f}]")
        bci = ("—" if a == "parent" else
               f"[{100*r['gain_pooled_binomial_ci95'][0]:+.2f}, "
               f"{100*r['gain_pooled_binomial_ci95'][1]:+.2f}]")
        zz = "—" if a == "parent" else f"{r['pooled_z']:+.2f}"
        tp = "—" if a == "parent" else f"{r['teams_positive']}/{r['n_teams']}"
        a_lbl = a + (" *(f/u)*" if a in FOLLOWUP else "")
        lines.append(f"| `{a_lbl}` | {(r['fold_delta']/1e6):+.3f}M | {r['battles']} | "
                     f"{r['pooled_wr']:.4f} | {g} | {ci} | {bci} | {zz} | {tp} |")
    lines.append("")
    lines.append("### Per-team gains (pp)")
    hdr = "| team | arch | parent WR | " + " | ".join(
        f"`{a}`" for a in ARM_ORDER if a in res["arms"] and a != "parent") + " |"
    lines.append(hdr)
    lines.append("|---|---|---:|" + "---:|" * (len(hdr.split("|")) - 4))
    for i, t in enumerate(common):
        pr = res["per_team"]["parent"][i]
        cells = []
        for a in ARM_ORDER:
            if a in res["per_team"] and a != "parent":
                cells.append(f"{100*res['per_team'][a][i]['gain']:+.1f}")
        lines.append(f"| `{t}` | {pr['arch']} | {pr['wr']:.3f} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    if out_md:
        open(out_md, "w").write(md + "\n")
    print(md)
    print()
    print(json.dumps(res["bars"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
