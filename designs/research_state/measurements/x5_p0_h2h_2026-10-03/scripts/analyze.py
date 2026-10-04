"""P0 analysis: the head-to-head ROUND-ROBIN of the banked replicates -> the edge table, the seat effect, σ_h with its
df and interval, the comparison with the untaught meter's σ_run, and the A/B's power at σ̂_h.

Reads ../rows (the §0b COUNT rows `main.h2h` wrote) and ../provenance.json; writes ../result.json and ../result.md.
Run: PYTHONPATH=<checkout>/src python3 analyze.py [--no-power]   (the power simulation takes ~1 min)

THE SCALE. An edge is the player's win rate against the opponent, pp above 50. For runs with head-to-head
strengths s_i (pp of win rate; small differences are additive), d_ij = s_i - s_j + u + e_ij with u the SEAT effect
(the player keeps seat p1 in both games of a pair, so it rides every edge) and e_ij the meter's pair-clustered
noise. σ_h is the SD of s_i across runs: `main.h2h.runfloor` (its docstring has the estimator and its interval).

THE MAPPING TO THE UNTAUGHT SCALE (stated, then checked). The untaught meter U reads a run's win rate against ONE
fixed opponent on 8 pinned teams; the head-to-head reads it against another run on the trainee's team
distribution. If a run's strength is one number theta (Bradley-Terry: P(i beats j) = logistic(theta_i - theta_j)),
a run's pp change on either meter is its slope times d theta: d P / d theta = P (1 - P), which is 0.25 at P = 1/2 and
0.225-0.248 over U's observed levels (34-46 %). The two scales then differ by under ~10 %, and sigma_h and
sigma_run(U) are comparable number for number. That holds only if strength is one-dimensional and the 8 pinned
teams / the one opponent are representative of the team distribution; the regression of the head-to-head edges on the
same runs' U differences (below, 6 edges that share runs, so NOT independent) is the empirical check.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "x5_revision_2026-10-03" / "scripts"))

TAGS = ("A", "A2", "Ap", "B")
SETS = {
    "S1 pure seed pair {A2, Ap} (same pin 277f318f, seeds 1001 / 1002, N = 48; A2 has a crash-restart)": ["A2", "Ap"],
    "S2 post-boundary runs {A2, Ap, B} (pins 277f318f / 7ef99979 both CONTAIN f6b32f5a; B is N = 256, n_steps 384)": ["A2", "Ap", "B"],
    "S3 seed + pin {A, A2, Ap} (all N = 48; A is on pin f9349f95, BEFORE f6b32f5a's controller regime boundary; same seed id as A2)": ["A", "A2", "Ap"],
    "S4 all four {A, A2, Ap, B} (the pin boundary AND the N lever are inside this one)": ["A", "A2", "Ap", "B"],
}
PLANNING_SET = next(k for k in SETS if k.startswith("S3"))        # the conservative one the note's table is read against
POST_SET = next(k for k in SETS if k.startswith("S2"))            # the one that matches the A/B's conditions (one commit)
METERS_READ = HERE.parent / "m5_sizing" / "results" / "meters_read.json"           # the banked untaught-meter read
GS_SIM = HERE.parent / "x5_revision_2026-10-03" / "scripts" / "gs_sim.py"                  # the registered design's simulator


def u_scale():
    """The four runs' untaught-meter levels (pp), read from the banked ``meters_read.json`` (``levels[*].mean_pp``)."""
    lv = json.loads(METERS_READ.read_text())["levels"]
    return {t: float(lv[t]["mean_pp"]) for t in TAGS}


def check():
    """--check: resolve every input this readout reads and report any that is MISSING, computing nothing."""
    need = {"provenance.json": HERE / "provenance.json", "the banked untaught-meter read": METERS_READ,
            "the registered design's simulator": GS_SIM}
    missing = [f"{what}: {p}" for what, p in need.items() if not p.is_file()]
    if not list((HERE / "rows").glob("ledger.*.jsonl*")):
        missing.append(f"the head-to-head rows: {HERE / 'rows'}/ledger.*.jsonl")
    for m in missing:
        print("MISSING", m)
    print(f"{'OK' if not missing else 'FAILED'}: P0 readout inputs ({len(need) + 1} resolved or reported)")
    return 1 if missing else 0


def load():
    from agents.training import eval_ledger as L
    from main.h2h import stats as ST

    prov = json.loads((HERE / "provenance.json").read_text())
    sha2tag = {prov[t]["final_sha256"]: t for t in TAGS}
    # ledger v2 (eval U1): a declared read; the v1 rows are upgraded on read (regime ids recomputed, v1 id kept)
    decl = L.ReaderDecl(name="x5_p0.analyze", purposes=frozenset({"audit"}),
                        regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy",
                                              opponent_play="greedy", mirrored=True),
                        requests="any", selection="include", flags_ok=frozenset({"digest_unrecorded"}),
                        inference="conditional")
    rows = list(L.read(decl, root=HERE / "rows").rows)
    unknown = {r["player"]["sha256"] for r in rows} | {r["opponent"]["sha256"] for r in rows}
    unknown -= set(sha2tag)
    if unknown:
        raise SystemExit(f"rows name checkpoints that are not the four finals: {sorted(u[:12] for u in unknown)}")
    edges = ST.group_edges(rows)
    summ = {(sha2tag[p], sha2tag[o]): ST.edge_summary(rs) for (p, o), rs in edges.items()}
    per_edge_rows = {(sha2tag[p], sha2tag[o]): rs for (p, o), rs in edges.items()}
    return prov, rows, summ, per_edge_rows


def edge_table(summ, per_edge_rows):
    out = []
    for (p, o), s in sorted(summ.items(), key=lambda kv: (TAGS.index(kv[0][0]), TAGS.index(kv[0][1]))):
        rs = per_edge_rows[(p, o)]
        pc = s["pair_counts"]
        n = s["pairs"]
        clean = sum(int(r["compute"].get("clean_pairs") or 0) for r in rs)
        clean_off = sum(int(r["compute"].get("clean_pairs_off_center") or 0) for r in rs)
        out.append({
            "player": p, "opponent": o, "pairs": n, "games": s["games"], "w": s["w"], "l": s["l"], "d": s["d"],
            "pair_counts": pc, "score_pct": 100 * s["score"], "d_pp": 100 * (s["score"] - 0.5),
            "ci95_pp": [100 * (s["score_ci95"][0] - 0.5), 100 * (s["score_ci95"][1] - 0.5)],
            "se_pp": s["se_pp"], "pair_sd": s["pair_sd"],
            "pairs_off_center": n - pc[2], "pairs_off_center_frac": (n - pc[2]) / n,
            "near_tie_games": s["near_tie_games"], "near_tie_games_frac": s["near_tie_games"] / s["games"],
            "clean_pairs": clean, "clean_pairs_off_center": clean_off,
            "games_per_s": s["games_per_s"], "wall_s": s["wall_s"], "devices": s["devices"],
            "batches": s["batches"], "regime_id": s["regime_id"],
        })
    return out


def reads_of(summ):
    from main.h2h import runfloor as RF

    return [RF.edge_read(p, o, s) for (p, o), s in summ.items()]


def sigma_blocks(summ, u_used):
    from main.h2h import runfloor as RF

    reads = reads_of(summ)
    res = {}
    for name, runs in SETS.items():
        try:
            both = RF.sigma_h(runs, reads, u=u_used)
        except ValueError as e:
            res[name] = {"error": str(e)}
            continue
        # the note's per-edge recipe for comparison: the FORWARD edge only (one direction), seat effect subtracted
        fwd_only = [e for e in reads if e.player in runs and e.opponent in runs
                    and runs.index(e.player) < runs.index(e.opponent)]
        try:
            fwd = RF.sigma_h(runs, fwd_only, u=u_used)
        except ValueError:
            fwd = None
        res[name] = {"both_directions": both, "forward_edge_only": fwd}
    return res


def u_scale_check(summ, u):
    """Regress the u-corrected head-to-head pair differences on the same pairs' untaught-meter differences."""
    from main.h2h import runfloor as RF

    reads = reads_of(summ)
    pairs = RF.combine_pairs(list(TAGS), reads, u)
    U = u_scale()
    pts = [(U[i] - U[j], d, v) for (i, j), (d, v) in pairs.items()]
    sxx = sum(x * x for x, _d, _v in pts)
    slope = sum(x * d for x, d, _v in pts) / sxx
    resid = sum((d - slope * x) ** 2 for x, d, _v in pts)
    dof = len(pts) - 1
    se = math.sqrt(resid / dof / sxx) if dof > 0 else None
    return {"pairs": [{"pair": f"{i} - {j}", "delta_U_pp": U[i] - U[j], "h2h_pp": d, "h2h_se_pp": math.sqrt(v)}
                      for (i, j), (d, v) in pairs.items()],
            "slope_h2h_per_U_through_origin": slope, "slope_se_naive": se, "dof": dof,
            "note": "6 pair-differences share 4 runs: the 3 independent contrasts, not 6, carry the information"}


def power_block(sig_by_label, cell_se):
    import numpy as np
    from gs_sim import ALPHA, LOOKS, obf_z, runs_cost, sim, t_bounds

    C, zb = obf_z(ALPHA)
    dfs = [2 * (m - 1) for m in LOOKS]
    tb = t_bounds(zb * 1.02, dfs)
    out = []
    for label, sig in sig_by_label.items():
        for delta in (3.5, 4.5):
            r1, l1 = sim("cross", sig, delta, 0.0, tb, fut=1.0, sig_e=cell_se, n=200_000)
            r0, _ = sim("cross", sig, delta, -delta, tb, fut=1.0, sig_e=cell_se, n=200_000)
            out.append({"label": label, "sigma": sig, "delta": delta, "cell_se_pp": cell_se,
                        "power_at_delta_0": float(np.mean(r1 == 1)), "type_I": float(np.mean(r0 == 1)),
                        "E_GPUh_H1": float(runs_cost(l1))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-power", action="store_true")
    a = ap.parse_args()
    from main.h2h import runfloor as RF

    prov, rows, summ, per_edge_rows = load()
    table = edge_table(summ, per_edge_rows)
    reads = reads_of(summ)

    selfs = [e for e in reads if e.player == e.opponent]
    u_self = RF.seat_effect(selfs)
    u_rev = RF.seat_effect_from_reversals(reads)
    # primary u: the two independent estimates pooled by inverse variance (each is a measurement of the same u)
    est = [x for x in (u_self, u_rev) if x["measured"]]
    if est:
        w = [1 / x["se"] ** 2 for x in est]
        u = sum(wi * x["u"] for wi, x in zip(w, est)) / sum(w)
        u_se = math.sqrt(1 / sum(w))
    else:
        u, u_se = 0.0, None
    sig = sigma_blocks(summ, u)
    uchk = u_scale_check(summ, u) if len(summ) >= 6 else None

    clean_total = {"pairs": sum(t["clean_pairs"] for t in table if t["player"] == t["opponent"]),
                   "off_center": sum(t["clean_pairs_off_center"] for t in table if t["player"] == t["opponent"])}
    res = {"edges": table, "seat_effect": {"u_pp": u, "u_se_pp": u_se, "from_self_play": u_self, "from_reversals": u_rev},
           "self_play_clean_pairs": clean_total, "sigma_h": sig, "u_scale_check": uchk,
           "untaught_sigma_run_reference": {"A'-A2 (1 df)": 3.43, "A-A2 (1 df)": 7.98, "pooled A'-A2, W_b-W (2 df)": 2.97,
                                            "pooled with A-A2 (3 df)": 5.21, "SD of four E10 levels (3 df)": 4.84}}

    # the cell noise the A/B would see at 1,000 pairs per cell, from the measured pair SD of the replicate edges
    reps = [t for t in table if t["player"] != t["opponent"]]
    if reps:
        sd = sum(t["pair_sd"] for t in reps) / len(reps)
        res["cell_se_pp_at_1000_pairs"] = 100 * sd / math.sqrt(1000)
    if not a.no_power and reps:
        def pick(set_key, field):
            b = sig.get(set_key, {}).get("both_directions")
            return b[field] if b else None

        cand = {
            "S1 point (pure seed pair, 1 df)": pick(next(k for k in SETS if k.startswith("S1")), "sigma_hat"),
            "S2 point (post-boundary, 2 df)": pick(POST_SET, "sigma_hat"),
            "S2 one-sided 95% upper": pick(POST_SET, "sigma_upper_one_sided"),
            "S3 two-sided CI lower": (pick(PLANNING_SET, "sigma_ci") or [None])[0],
            "S3 point (seed + pin, 2 df)": pick(PLANNING_SET, "sigma_hat"),
            "table row 2.5": 2.5,
            "table row 3.43 (the borrowed untaught value)": 3.43,
            "S4 one-sided 95% upper (3 df)": pick(next(k for k in SETS if k.startswith("S4")), "sigma_upper_one_sided"),
            "S3 one-sided 95% upper (2 df)": pick(PLANNING_SET, "sigma_upper_one_sided"),
        }
        res["power"] = power_block({k: v for k, v in cand.items() if v is not None and math.isfinite(v)},
                                   res["cell_se_pp_at_1000_pairs"])
    (HERE / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    md = render(res)
    (HERE / "result.md").write_text(md)
    print(md)


def render(res):
    L = ["# P0 result (generated by scripts/analyze.py)\n", "## Edges (player vs opponent; every edge a mirrored read at the final snapshots)\n",
         "| player | opponent | pairs | W / L / D | pentanomial | score % | d (pp) [95% pair CI] | SE pp | pairs off centre | near-tie games | games/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in res["edges"]:
        L.append(f"| {t['player']} | {t['opponent']} | {t['pairs']} | {t['w']} / {t['l']} / {t['d']} | {t['pair_counts']} | "
                 f"{t['score_pct']:.2f} | {t['d_pp']:+.2f} [{t['ci95_pp'][0]:+.2f}, {t['ci95_pp'][1]:+.2f}] | {t['se_pp']:.3f} | "
                 f"{t['pairs_off_center']} ({100 * t['pairs_off_center_frac']:.1f}%) | {t['near_tie_games']} "
                 f"({100 * t['near_tie_games_frac']:.1f}%) | {t['games_per_s']:.1f} |")
    s = res["seat_effect"]
    L += ["\n## Seat effect u (the player keeps seat p1)\n",
          f"- from self-play edges: {fmt(s['from_self_play'])}", f"- from reversed edges (d_ij + d_ji = 2u): {fmt(s['from_reversals'])}",
          f"- pooled u = {s['u_pp']:+.3f} pp" + (f" (SE {s['u_se_pp']:.3f})" if s["u_se_pp"] is not None else ""),
          f"- self-play pairs clear of a near-tie decision that were NOT exactly 1/2: {res['self_play_clean_pairs']['off_center']} of {res['self_play_clean_pairs']['pairs']}"]
    L.append("\n## σ_h (run SD on the head-to-head scale; meter variance subtracted, seat effect subtracted)\n")
    for name, blocks in res["sigma_h"].items():
        L.append(f"**{name}**\n")
        if "error" in blocks:
            L.append(f"- {blocks['error']}\n")
            continue
        for label, key in (("both directions of every pair", "both_directions"), ("forward edge only (the note's per-edge recipe)", "forward_edge_only")):
            b = blocks.get(key)
            if not b:
                continue
            L.append(f"- {label}: K = {b['k']}, df = {b['df']}: strengths (pp, centred) "
                     f"{ {k: round(v, 2) for k, v in b['strengths_pp'].items()} }; σ̂_h = **{b['sigma_hat']:.2f}** pp "
                     f"(σ̂² = {b['sigma_hat_sq']:.2f}; raw SD of strengths {b['raw_sd_of_strengths']:.2f}; meter variance subtracted "
                     f"{b['meter_variance_subtracted']:.3f}); 95% CI [{b['sigma_ci'][0]:.2f}, {b['sigma_ci'][1]:.2f}]; one-sided 95% upper {b['sigma_upper_one_sided']:.2f}; "
                     f"max |pairwise| {b['max_abs_pairwise_pp']:.2f}")
        L.append("")
    if res.get("u_scale_check"):
        c = res["u_scale_check"]
        L += ["## Head-to-head vs untaught scale (the same runs)\n", "| pair | ΔU (pp) | h2h d (pp) | h2h SE |", "|---|---|---|---|"]
        L += [f"| {p['pair']} | {p['delta_U_pp']:+.2f} | {p['h2h_pp']:+.2f} | {p['h2h_se_pp']:.2f} |" for p in c["pairs"]]
        L.append(f"\nslope of h2h on ΔU through the origin: {c['slope_h2h_per_U_through_origin']:.2f} (naive SE {c['slope_se_naive']}, {c['dof']} dof; {c['note']})")
    if res.get("power"):
        L += ["\n## A/B power at σ̂_h (registered design, futility stop on, 200,000 replications; cell SE "
              f"{res['cell_se_pp_at_1000_pairs']:.2f} pp at 1,000 pairs)\n", "| σ source | σ | δ | power at Δ = 0 | type-I | E[GPU-h] at Δ = 0 |", "|---|---|---|---|---|---|"]
        L += [f"| {p['label']} | {p['sigma']:.2f} | {p['delta']} | {p['power_at_delta_0']:.3f} | {p['type_I']:.4f} | {p['E_GPUh_H1']:.1f} |" for p in res["power"]]
    return "\n".join(L) + "\n"


def fmt(x):
    if not x.get("measured"):
        return "not measured"
    return f"{x['u']:+.3f} pp (SE {x['se']:.3f})"


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else (main() or 0))
