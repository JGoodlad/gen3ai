"""Derived tables + the two FLOOR tests, off displacement.json and projection.json.

Adds nothing new from the models -- it reads the two artifacts and emits:
  * the three README tables, printed to results_table.txt
  * the P1 replicate-floor test (a funded-vs-unfunded displacement gap must clear the
    funded-vs-funded and unfunded-vs-unfunded replicate gaps)
  * the P3 exact pairing permutation: with FOUR arms there are exactly THREE ways to split them
    into two pairs, so the smallest attainable p on "replicate cosine > cross cosine" is 1/3.
    Stated, not hidden.
  * the arm-level ordering ladder |Delta| -> KL1 -> actual, each against the published column.

Run: python tables.py > results_table.txt
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DISP = json.load(open(os.path.join(HERE, "displacement.json")))
PROJ = json.load(open(os.path.join(HERE, "projection.json")))
G = ["encoders", "team_transformer", "projection_mlp", "belief_op", "action_head", "critic"]
KEYS = DISP["_meta"]["arm_keys"]
IDX = {k: i for i, k in enumerate(KEYS)}
PUB = PROJ["_meta"]["published_offslice_clustered"]


def spearman(a, b):
    def r(x):
        o = np.argsort(np.asarray(x, float), kind="mergesort")
        q = np.empty(len(x))
        q[o] = np.arange(len(x))
        return q
    return float(np.corrcoef(r(a), r(b))[0, 1])


def line(s=""):
    print(s)


line("=" * 118)
line("TABLE 1 -- PER-GROUP DISPLACEMENT  (Delta = theta_arm - theta_parent, POLICY parameters)")
line("=" * 118)
lay = DISP["_meta"]["group_layout"]
P = DISP["_meta"]["n_params"]
line(f"parameter counts: " + "  ".join(
    f"{g}={lay[g][1]-lay[g][0]:,} ({(lay[g][1]-lay[g][0])/P*100:.1f}%)" for g in G))
line(f"parent |theta| = {DISP['_meta']['parent_l2']:.3f}   P = {P:,}")
line("")
line("1a. RELATIVE displacement  |Delta_g| / |theta_g|  (x1000)")
line(f"{'arm':16s}{'step':>12s}{'|Delta|':>9s}{'rel':>8s}" + "".join(f"{g[:9]:>11s}" for g in G))
for k in KEYS:
    v = DISP["arms"][k]
    line(f"{k:16s}{v['steps']:>12,}{v['l2']:>9.4f}{v['rel_l2']:>8.5f}"
         + "".join(f"{v['groups'][g]['rel_l2']*1000:>11.3f}" for g in G))
line("")
line("1b. SHARE of |Delta|^2  (%)   -- read against the parameter counts above")
line(f"{'arm':16s}" + "".join(f"{g[:9]:>11s}" for g in G))
for k in KEYS:
    v = DISP["arms"][k]
    line(f"{k:16s}" + "".join(f"{v['groups'][g]['sq_share']*100:>11.1f}" for g in G))
line("")
line("1c. BUFFERS (never inside a group). Every constant data table is byte-identical across all")
line("    17 arms; the ONLY buffers that moved are PopArt's running normalizer.")
line(f"{'arm':16s}{'popart.mu':>12s}{'popart.sigma':>14s}{'popart.nu':>12s}")
for k in KEYS:
    b = DISP["arms"][k]["buffers_changed"]
    line(f"{k:16s}" + "".join(f"{b.get(n, {}).get('max_abs_change', 0.0):>{w}.5f}"
                              for n, w in (("popart.mu", 12), ("popart.sigma", 14),
                                           ("popart.nu", 12))))

line("")
line("1d. P1 FLOOR TEST at END depth -- funded-vs-unfunded displacement gap against the")
line("    two REPLICATE gaps (arms differing only in seed). WITHIN FLOOR unless the")
line("    cross gap exceeds BOTH replicate gaps.")
fa, fb, ua, ub = "TCFUNDA@end", "TCFUNDB@end", "TCUNFA@end", "TCUNFB@end"
line(f"{'group':20s}{'funded mean':>13s}{'unfund mean':>13s}{'gap':>10s}"
     f"{'repl F':>10s}{'repl U':>10s}  verdict")
floor_rows = {}
for g in ["ALL"] + G:
    def rel(k):
        return (DISP["arms"][k]["rel_l2"] if g == "ALL"
                else DISP["arms"][k]["groups"][g]["rel_l2"])
    F = (rel(fa) + rel(fb)) / 2
    U = (rel(ua) + rel(ub)) / 2
    gap = F - U
    rF, rU = abs(rel(fa) - rel(fb)), abs(rel(ua) - rel(ub))
    v = "CLEARS FLOOR" if abs(gap) > max(rF, rU) else "WITHIN FLOOR"
    floor_rows[g] = dict(funded=F, unfunded=U, gap=gap, repl_F=rF, repl_U=rU, verdict=v)
    line(f"{g:20s}{F*1000:>13.3f}{U*1000:>13.3f}{gap*1000:>10.3f}"
         f"{rF*1000:>10.3f}{rU*1000:>10.3f}  {v}")

line("")
line("=" * 118)
line("TABLE 2 -- FIRST-ORDER PROJECTION vs ACTUAL OFF-SLICE KL")
line("=" * 118)
for sl in ("untaught", "taught"):
    S = PROJ["slices"][sl]
    line(f"--- {sl.upper()}  ({S['n_states']} states / {S['n_teams']} teams)")
    line(f"{'arm':16s}{'actual KL':>10s}{'cluster CI95':>22s}{'KL1':>9s}"
         f"{'KL1/act':>9s}{'pearson':>9s}{'spearman':>10s}")
    for k in KEYS:
        v = S["arms"][k]
        ci = v["actual_kl_cluster"]["ci95"]
        line(f"{k:16s}{v['actual_kl_state_mean']:>10.5f}"
             f"{'[%+.4f,%+.4f]' % (ci[0], ci[1]):>22s}{v['kl1_state_mean']:>9.5f}"
             f"{v['ratio_kl1_over_actual']:>9.3f}{v['pearson_kl1_vs_actual']:>+9.3f}"
             f"{v['spearman_kl1_vs_actual']:>+10.3f}")
    line("")
    line("    per-group ADDITIVE share of KL1 (%; sums to 100 exactly) "
         "[group-ALONE quadratic, does NOT sum]")
    line(f"    {'arm':16s}" + "".join(f"{g[:9]:>17s}" for g in G))
    for k in KEYS:
        v = S["arms"][k]["groups"]
        line(f"    {k:16s}" + "".join(
            f"{v[g]['additive_share']*100:>10.1f}[{v[g]['alone_share_of_KL1']*100:5.1f}]"
            for g in G))
    line("")
    o = S["ordering"]
    dl = [DISP["arms"][k]["l2"] for k in o["arms"]]
    line("    ARM-ORDERING LADDER against the published clustered off-slice KL")
    line(f"      arms        {o['arms']}")
    line(f"      published   {[round(x, 4) for x in o['published']]}")
    line(f"      |Delta|     {[round(x, 4) for x in dl]}   rho={spearman(dl, o['published']):+.3f}")
    line(f"      KL1         {[round(x, 4) for x in o['kl1']]}   "
         f"rho={o['spearman_kl1_vs_published']:+.3f}")
    line(f"      actual      {[round(x, 4) for x in o['actual']]}   "
         f"rho={o['spearman_actual_vs_published']:+.3f}")
    line("")
    line("    PAIRED contrasts (cluster bootstrap over teams, one shared index set)")
    for c, v in S["contrasts"].items():
        a_, k_ = v["actual"], v["kl1"]
        line(f"      {c:28s} actual {a_['delta']:+.4f} "
             f"[{a_['ci95'][0]:+.4f},{a_['ci95'][1]:+.4f}] {a_['verdict']:13s}"
             f" | KL1 {k_['delta']:+.4f} [{k_['ci95'][0]:+.4f},{k_['ci95'][1]:+.4f}] "
             f"{k_['verdict']}")
    line("")

line("=" * 118)
line("TABLE 3 -- COSINE BETWEEN DISPLACEMENTS  (P3)")
line("=" * 118)
GA = ["ALL"] + G
pairs = [(fa, fb, "REPLICATE floor: funded A . funded B"),
         (ua, ub, "REPLICATE floor: unfunded A . unfunded B"),
         (fa, ua, "cross: funded A . unfunded A"),
         (fb, ub, "cross: funded B . unfunded B"),
         (fa, ub, "cross: funded A . unfunded B"),
         (fb, ua, "cross: funded B . unfunded A"),
         ("B2@end", "C1@end", "reuse: B2 . C1 (the loss-off control)"),
         ("R4DOSE3@end", "R4DOSE12@end", "reuse: dose 2.12x . dose 0.53x"),
         (fa, "B2@end", "CROSS-BATCH (not one experiment): TCFUNDA . B2")]
line(f"{'pair':42s}" + "".join(f"{g[:9]:>11s}" for g in GA))
for x, y, lab in pairs:
    line(f"{lab:42s}" + "".join(f"{DISP['cosine'][g][IDX[x]][IDX[y]]:>11.4f}" for g in GA))
line("")
line("depth rotation: cos(arm@p1M, arm@end) -- how far the fold's own direction turns")
line(f"{'arm':42s}" + "".join(f"{g[:9]:>11s}" for g in GA))
for t in ["TCFUNDA", "TCFUNDB", "TCUNFA", "TCUNFB"]:
    line(f"{t:42s}" + "".join(
        f"{DISP['cosine'][g][IDX[t+'@p1M']][IDX[t+'@end']]:>11.4f}" for g in GA))
line("")
line("P3 EXACT PAIRING PERMUTATION. Four arms {FA,FB,UA,UB} admit exactly THREE splits into two")
line("pairs, so the smallest attainable p-value on 'the replicate pairing has the highest mean")
line("within-pair cosine' is 1/3 = 0.333. Reported for every group:")
line(f"{'group':20s}{'within(obs)':>13s}{'cross(obs)':>12s}{'gap':>9s}"
     f"{'rank of obs':>13s}{'exact p':>9s}")
perm_rows = {}
arms4 = [fa, fb, ua, ub]
splits = [((fa, fb), (ua, ub)), ((fa, ua), (fb, ub)), ((fa, ub), (fb, ua))]
for g in GA:
    C = DISP["cosine"][g]

    def c(x, y):
        return C[IDX[x]][IDX[y]]
    stats = []
    for sp in splits:
        wi = np.mean([c(*sp[0]), c(*sp[1])])
        cr = np.mean([c(x, y) for x, y in itertools.combinations(arms4, 2)
                      if (x, y) not in sp and (y, x) not in sp])
        stats.append(wi - cr)
    obs = stats[0]
    rank = int(np.sum(np.array(stats) >= obs))
    perm_rows[g] = dict(within=float(np.mean([c(fa, fb), c(ua, ub)])),
                        cross=float(np.mean([c(fa, ua), c(fb, ub), c(fa, ub), c(fb, ua)])),
                        gap=float(obs), rank=rank, p=rank / 3.0)
    r = perm_rows[g]
    line(f"{g:20s}{r['within']:>13.4f}{r['cross']:>12.4f}{r['gap']:>+9.4f}"
         f"{f'{rank}/3':>13s}{r['p']:>9.3f}")

with open(os.path.join(HERE, "verdict_stats.json"), "w") as f:
    json.dump(dict(p1_floor=floor_rows, p3_pairing_permutation=perm_rows,
                   ordering_ladder={sl: dict(
                       arms=PROJ["slices"][sl]["ordering"]["arms"],
                       rho_delta_l2=spearman([DISP["arms"][k]["l2"]
                                              for k in PROJ["slices"][sl]["ordering"]["arms"]],
                                             PROJ["slices"][sl]["ordering"]["published"]),
                       rho_kl1=PROJ["slices"][sl]["ordering"]["spearman_kl1_vs_published"],
                       rho_actual=PROJ["slices"][sl]["ordering"]["spearman_actual_vs_published"])
                       for sl in ("untaught", "taught")}), f, indent=1)
line("")
line("wrote verdict_stats.json")
