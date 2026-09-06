"""Render analysis.json as the markdown tables the README quotes. Reads, never recomputes.

Run: nice -n 10 python tables.py > results_table.txt
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "analysis.json")))
GROUPS = ["ALL", "action_head", "encoders", "team_transformer", "projection_mlp", "belief_op",
          "critic"]
ARMS = ["A", "B", "C"]


def f(x, n=4):
    return f"{x:.{n}f}"


print("=" * 100)
print("DEPTH GRID (t = steps since that cell's parent)")
print("=" * 100)
for era in ("v8", "gen"):
    print(f"\n--- {era}  parent @ {A['_meta']['parent_steps'][era]:,}  "
          f"P = {A['_meta']['n_params'][era]:,} policy parameters")
    print(f"  {'key':10s} {'t':>10s} {'abs step':>12s} {'|dtheta|':>10s} {'rel':>9s} "
          f"{'KL(p||arm)':>11s}  fit?")
    for k in sorted(A["depth_grid"][era], key=lambda k: (A["depth_grid"][era][k]["arm"],
                                                         A["depth_grid"][era][k]["t"])):
        r = A["depth_grid"][era][k]
        print(f"  {k:10s} {r['t']:>10,} {r['steps']:>12,} {r['l2']:>10.4f} {r['rel_l2']:>9.5f} "
              f"{r['kl_all']:>11.4f}  {'yes' if r['fit_point'] else 'NO (same-depth check)'}")

print("\nv8 same-depth agreement (final_model_interrupted.zip vs checkpoints/*.zip):")
for arm, r in A["v8_same_depth_agreement"].items():
    print(f"  {arm}: step gap {r['step_gap']:>6,}  |d| {r['l2_final']:.4f} vs {r['l2_ckpt']:.4f}  "
          f"rel diff {r['rel_diff']:.2e}  bitwise-equal={r['identical_to_1e9']}")

print()
print("=" * 100)
print("P1 — DISPLACEMENT EXPONENT b in |dtheta| ~ t^b   (0.5 = random walk, 1.0 = directed drift)")
print("=" * 100)
for name in ("v8_2pt", "gen_2pt_outer", "gen_3pt"):
    b = A["P1_displacement_exponent"][name]
    print(f"\n--- {name}  (era {b['era']}, depths {b['depths']})")
    print(f"  {'group':18s} {'A':>8s} {'B':>8s} {'C':>8s} {'mean':>8s} "
          f"{'boot CI over arms':>24s} {'pooled':>8s}")
    for g in GROUPS:
        r = b[g]
        pa = r["per_arm"]
        print(f"  {g:18s} {pa['A']:>8.4f} {pa['B']:>8.4f} {pa['C']:>8.4f} {r['mean']:>8.4f} "
              f"{'[' + f(r['ci'][0]) + ', ' + f(r['ci'][1]) + ']':>24s} {r['pooled']:>8.4f}")

print()
print("HEADLINE CONTRAST — matched two-point windows (v8 d1..d3 vs gen d1..d3)")
print(f"  {'group':18s} {'v8':>8s} {'gen':>8s} {'v8-gen':>9s} {'sep?':>5s} {'CIdisj?':>8s} "
      f"{'permp':>7s} {'spread':>8s}  verdict")
for g in GROUPS:
    r = A["P1_contrast_matched_2pt"][g]
    print(f"  {g:18s} {r['v8_mean']:>8.4f} {r['gen_mean']:>8.4f} {r['diff_v8_minus_gen']:>+9.4f} "
          f"{str(r['complete_separation']):>5s} {str(r['ci_disjoint']):>8s} {r['perm_p']:>7.3f} "
          f"{r['max_within_cell_spread']:>8.4f}  {r['verdict']}")

print()
print("=" * 100)
print("P2 — REPLICATE COSINE cos(dtheta_X, dtheta_Y) at matched depth")
print(f"     random-walk floor 1/sqrt(P): v8 {A['P2_random_walk_floor']['v8']:.2e}  "
      f"gen {A['P2_random_walk_floor']['gen']:.2e}   |   fold-depth reference "
      f"{A['P2_fold_reference_cosine']}")
print("=" * 100)
for era in ("v8", "gen"):
    for dep, blk in A["P2_replicate_cosine"][era].items():
        print(f"\n--- {era} @ {dep}  (t = {blk['t']:,})")
        print(f"  {'group':18s} " + " ".join(f"{p:>8s}" for p in blk["pairs"])
              + f" {'mean':>8s} {'boot CI over pairs':>24s}")
        for g in GROUPS:
            r = blk[g]
            print(f"  {g:18s} " + " ".join(f"{v:>8.4f}" for v in r["values"])
                  + f" {r['mean']:>8.4f} "
                  + f"{'[' + f(r['ci'][0]) + ', ' + f(r['ci'][1]) + ']':>24s}")

print("\nCONTRAST at the end depth (d3):")
print(f"  {'group':18s} {'v8':>8s} {'gen':>8s} {'v8-gen':>9s} {'sep?':>5s} {'permp':>7s} "
      f"{'spread':>8s}  verdict")
for g in GROUPS:
    r = A["P2_contrast_at_end"][g]
    print(f"  {g:18s} {r['v8_mean']:>8.4f} {r['gen_mean']:>8.4f} {r['diff_v8_minus_gen']:>+9.4f} "
          f"{str(r['complete_separation']):>5s} {r['perm_p']:>7.3f} "
          f"{r['max_within_cell_spread']:>8.4f}  {r['verdict']}")

print()
print("=" * 100)
print("P3 — OUTPUT TWIN: c in KL(parent||arm) ~ t^c  (locally-quadratic KL => c ~ 2b)")
print("=" * 100)
for era in ("v8", "gen"):
    b = A["P3_output_twin"][era]
    print(f"\n--- {era}  depths {b['depths']}")
    print(f"  {'slice':14s} {'A':>8s} {'B':>8s} {'C':>8s} {'mean':>8s} {'boot CI over arms':>24s}"
          f" {'pooled':>8s}")
    for s in ("kl_all", "kl_taught", "kl_untaught"):
        r = b[s]
        pa = r["per_arm"]
        print(f"  {s:14s} {pa['A']:>8.4f} {pa['B']:>8.4f} {pa['C']:>8.4f} {r['mean']:>8.4f} "
              f"{'[' + f(r['ci'][0]) + ', ' + f(r['ci'][1]) + ']':>24s} {r['pooled']:>8.4f}")
    print("  KL LEVELS (cluster bootstrap over the 24 teams):")
    for d_, per in b["levels"].items():
        row = "  ".join(f"{arm}={per[arm]['mean']:.4f} [{per[arm]['lo']:.4f},{per[arm]['hi']:.4f}]"
                        for arm in ARMS)
        print(f"    {d_}: {row}")

print("\nCONTRAST:")
print(f"  {'slice':14s} {'v8':>8s} {'gen':>8s} {'v8-gen':>9s} {'sep?':>5s} {'permp':>7s} "
      f"{'spread':>8s}  verdict")
for s in ("kl_all", "kl_taught", "kl_untaught"):
    r = A["P3_contrast"][s]
    print(f"  {s:14s} {r['v8_mean']:>8.4f} {r['gen_mean']:>8.4f} {r['diff_v8_minus_gen']:>+9.4f} "
          f"{str(r['complete_separation']):>5s} {r['perm_p']:>7.3f} "
          f"{r['max_within_cell_spread']:>8.4f}  {r['verdict']}")
print(f"\n  c/b consistency (expect ~2):  v8 {A['P3_c_over_b']['v8']:.3f}   "
      f"gen {A['P3_c_over_b']['gen']:.3f}")

print()
print("=" * 100)
print("E1 — EXPLORATORY (not pre-registered): cos(dtheta_t1, dtheta_t2) WITHIN one arm")
print("     random walk => sqrt(t1/t2);  directed drift => 1")
print("=" * 100)
for era in ("v8", "gen"):
    print(f"\n--- {era}")
    print(f"  {'arm':4s} {'t1':>10s} {'t2':>10s} {'sqrt(t1/t2)':>12s} "
          + " ".join(f"{g[:12]:>13s}" for g in GROUPS))
    for arm in ARMS:
        r = A["E1_within_arm_direction_persistence"][era][arm]
        print(f"  {arm:4s} {r['t_early']:>10,} {r['t_late']:>10,} "
              f"{r['diffusive_prediction']:>12.4f} "
              + " ".join(f"{r[g]:>13.4f}" for g in GROUPS))

print()
print("=" * 100)
print("BUFFERS CHANGED vs parent (excluded from every group)")
print("=" * 100)
for era in ("v8", "gen"):
    seen = {}
    for k, bl in A["buffers_changed"][era].items():
        seen.setdefault(tuple(bl), []).append(k)
    for bl, ks in seen.items():
        print(f"  {era}: {len(ks)} of {len(A['buffers_changed'][era])} checkpoints changed "
              f"{len(bl)} buffer(s): {list(bl)[:6]}{' ...' if len(bl) > 6 else ''}")

print()
print("=" * 100)
print("FOLLOW-UP (not pre-registered) - PopArt BOOKKEEPING inside the critic group")
print("     PopArtNormalizer.update rescales exactly `policy.value_net` (513 params) with no")
print("     gradient behind it. This is the one row a non-learning transform can fake.")
print("=" * 100)
import statistics as _s
for era in ("v8", "gen"):
    p = json.load(open(os.path.join(HERE, f"popart_split_{era}.json")))
    print(f"\n--- {era}  (value_net = {p['n_popart']:,} params; critic remainder {p['n_rest']:,})")
    for k in sorted(p["arms"], key=lambda k: (p["arms"][k]["arm"], p["arms"][k]["t"])):
        r = p["arms"][k]
        tag = "" if r["fit_point"] else "   (same-depth check)"
        print(f"  {k:8s} t={r['t']:>9,}  |d|_popart={r['l2_popart']:.4f}  "
              f"|d|_rest={r['l2_rest']:.4f}  PopArt share of critic sq-disp="
              f"{r['popart_share_of_critic_sq']*100:>6.2f}%{tag}")
    ex = p["b_critic_excluding_popart"]
    on = p["b_critic_popart_only"]
    print(f"  b(critic, PopArt layer EXCLUDED): A={ex['A']:.4f} B={ex['B']:.4f} C={ex['C']:.4f}  "
          f"mean={_s.mean(ex.values()):.4f}")
    print(f"  b(PopArt layer alone)           : A={on['A']:.4f} B={on['B']:.4f} C={on['C']:.4f}  "
          f"mean={_s.mean(on.values()):.4f}")
