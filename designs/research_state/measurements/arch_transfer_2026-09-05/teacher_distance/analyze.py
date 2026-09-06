"""H5 ANALYSIS — does the fold's untaught delta fall with its teachers' off-slice distance?

Joins fold_table.json (deltas, per-team rows, provenance) to dist_gen.json / dist_rev2.json
(D_off, D_on per teacher set) and to content_locality/v8_era_n9.json (v8, REUSED not regenerated).

TWO units, both reported, as pre-registered:
  FOLD   -- the brief's unit. Folds sharing a teacher set share D_off EXACTLY, so ties on x.
  POINT  -- distinct (parent, teacher-set), the honest unit for a dose-response slope.

THREE bootstraps, so the reader can see which noise source dominates:
  BOOT-FOLD  resample folds with replacement
  BOOT-TEAM  resample the 8 untaught teams with replacement and RECOMPUTE every fold's delta
             from the same resampled team set (paired -- every fold is scored on those 8 teams)
  BOOT-BOTH  nested

Run: python analyze.py   (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.join(os.path.dirname(HERE), "content_locality")
NB = 20000
RNG = np.random.default_rng(50905)

# fold tag -> teacher-set key. C1 is EXCLUDED from every fit: distill_coef = 0, so no teacher
# content is transported at all and D_off cannot act. It is carried as a labelled CONTROL.
FOLD_SET = {
    "rev-4 / R4ACTION": "R4set", "rev-3 / R3ACTION": "R3set", "COMPFOLD": "R4set",
    "rev-2 / R2ACTION": "R2set", "B2": "R4set", "C1 (coef 0)": "R4set",
    "N1": "R4set", "N2": "R4set",
    "R4DOSE12": "R4set", "R4DOSE6": "R4set", "R4DOSE3": "R4set",
    "TC_FUND_A": "FUND", "TC_FUND_B": "FUND",
    "TC_UNF_A": "UNF", "TC_UNF_B": "UNF",
    "TC_UNF_K6_A": "UNF", "TC_UNF_K6_B": "UNF",
}
EXCLUDE_FROM_FIT = {"C1 (coef 0)"}


def spearman(x, y):
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); r[o] = np.arange(len(v))
        # average ties
        v = np.asarray(v)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def theil_sen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sl = [(y[j] - y[i]) / (x[j] - x[i])
          for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    return float(np.median(sl)) if sl else np.nan


def ci(v):
    v = np.asarray([z for z in v if np.isfinite(z)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan,)*2


def verdict(d, lo, hi, floor):
    if lo <= 0 <= hi:
        return "NOT DETECTED"
    return "WITHIN FLOOR" if abs(d) < floor else "SIGNIFICANT"


def main():
    ft = json.load(open(f"{HERE}/fold_table.json"))
    dist = {}
    for f in ("dist_gen.json", "dist_rev2.json"):
        dist.update(json.load(open(f"{HERE}/{f}"))["sets"])
    v8 = json.load(open(f"{CL}/v8_era_n9.json"))

    rows = []
    for f in ft["folds"]:
        k = FOLD_SET[f["fold"]]
        s = dist[k]
        rows.append({
            "fold": f["fold"], "set": k, "parent": os.path.basename(
                os.path.dirname(os.path.dirname(s["parent"]))) or s["parent"],
            "parent_run": s["parent"].split("/")[-2],
            "n_teachers": s["n_teachers"], "n_taught_teams": s["n_taught_teams"],
            "D_off": s["D_off"], "D_on": s["D_on"],
            "delta": f["delta_pp"], "ci": f["ci95"],
            "per_team": np.array(f["per_team_delta_pp"]),
            "coef": f["provenance"]["coef"], "gas": f["provenance"]["gas"],
            "fork_lr": f["provenance"]["fork_lr"],
            "in_fit": f["fold"] not in EXCLUDE_FROM_FIT,
        })

    # v8, cross-era, REUSED from content_locality (its own parent, its own untaught 8, greedy).
    # The FLOOR_* keys are the matched-noise checkpoint pairs, NOT teachers -- excluded.
    ptk = v8["per_team_kl_fwd"]
    v8_teachers = [t for t in ptk if not t.startswith("FLOOR")]
    v8_floor = {t: float(np.mean(ptk[t][:8])) for t in ptk if t.startswith("FLOOR")}
    v8_off_final = float(np.mean([np.mean(ptk[t][:8]) for t in v8_teachers]))
    v8_on_all = float(np.mean([np.mean(ptk[t][8:]) for t in v8_teachers]))
    # CORRECTED: content_locality scored final_model_interrupted.zip, which is not a rung of the
    # training path's resolver. v8_checkpoint_fix.py re-scored best_model/ on the SAME states.
    v8fx = json.load(open(f"{HERE}/v8_checkpoint_fix.json"))
    assert v8fx["summary"]["reproduces_content_locality_batch"]
    v8_off = v8fx["summary"]["D_off_best_model"]
    # the gen-era matched-noise floor on the SAME statistic (content_locality, untaught side)
    gen_floor = {"ckpt_28067760": 0.0374, "ckpt_27917760": 0.0654}

    print("=" * 108)
    print("FOLD TABLE  (delta = untaught-8 win rate vs the fold's OWN parent, pp; "
          "D_off/D_on = mean KL(teacher||parent))")
    print("=" * 108)
    hdr = (f"{'fold':20s} {'set':6s} {'par':10s} {'nT':>3s} {'ntm':>4s} "
           f"{'D_off':>7s} {'D_on':>7s} {'delta':>7s} {'95% CI':>18s} {'coef':>7s} {'gas':>4s}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda z: z["D_off"]):
        star = "" if r["in_fit"] else "  <- CONTROL, coef 0, excluded from every fit"
        print(f"{r['fold']:20s} {r['set']:6s} {r['parent_run'][6:16]:10s} {r['n_teachers']:3d} "
              f"{r['n_taught_teams']:4d} {r['D_off']:7.4f} {r['D_on']:7.4f} {r['delta']:+7.2f} "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] {str(r['coef']):>7s} {r['gas']:4d}{star}")
    print(f"{'v8_14 (CROSS-ERA)':20s} {'v8set':6s} {'ai_v8_04':10s} "
          f"{len(v8_teachers):3d} {22:4d} "
          f"{v8_off:7.4f} {v8_on_all:7.4f} {'+4.64':>7s} {'(@+1.09M, era meter)':>18s}")
    print(f"\n  MATCHED-NOISE FLOOR on this statistic (untaught side, from content_locality):")
    print(f"    gen era (two arbitrary R2ACTION checkpoints): "
          + ", ".join(f"{k} {v:.4f}" for k, v in gen_floor.items()))
    print(f"    v8  era (two arbitrary ai_v8_04 checkpoints): "
          + ", ".join(f"{k} {v:.4f}" for k, v in v8_floor.items()))
    fl = float(np.mean(list(gen_floor.values())))
    print(f"  D_off in FLOOR UNITS (gen floor mean {fl:.4f}): "
          + ", ".join(f"{k[1]} {v[0]['D_off']/fl:.1f}x" for k, v in
                      sorted({(r['parent_run'], r['set']): [r] for r in rows}.items(),
                             key=lambda z: z[1][0]['D_off']))
          + f", v8set {v8_off/v8fx['summary']['floor_mean']:.1f}x (own era floor, "
            f"CORRECTED checkpoint; on content_locality's it was "
            f"{v8_off_final/v8fx['summary']['floor_mean']:.1f}x)")

    fit = [r for r in rows if r["in_fit"]]
    x = np.array([r["D_off"] for r in fit]); y = np.array([r["delta"] for r in fit])
    xon = np.array([r["D_on"] for r in fit])
    P = np.array([r["per_team"] for r in fit])            # folds x 8 teams

    # distinct (parent, set) POINTS
    pts = {}
    for r in fit:
        pts.setdefault((r["parent_run"], r["set"]), []).append(r)
    px = np.array([v[0]["D_off"] for v in pts.values()])
    py = np.array([np.mean([q["delta"] for q in v]) for v in pts.values()])
    pP = np.array([np.mean([q["per_team"] for q in v], axis=0) for v in pts.values()])
    pnames = [f"{k[1]}@{k[0][6:16]}" for k in pts]

    print()
    print("=" * 108)
    print(f"PRIMARY (i).  n_folds_in_fit = {len(fit)}   n_distinct_points = {len(px)}")
    print("=" * 108)

    for label, X, Y, M in (("FOLD  unit", x, y, P), ("POINT unit", px, py, pP)):
        rho = spearman(X, Y); sl = theil_sen(X, Y)
        bf = [(lambda i: (spearman(X[i], Y[i]), theil_sen(X[i], Y[i])))(
            RNG.integers(0, len(X), len(X))) for _ in range(NB // 4)]
        bt = []
        for _ in range(NB // 4):
            j = RNG.integers(0, M.shape[1], M.shape[1])
            yy = M[:, j].mean(axis=1)
            bt.append((spearman(X, yy), theil_sen(X, yy)))
        bb = []
        for _ in range(NB // 4):
            i = RNG.integers(0, len(X), len(X)); j = RNG.integers(0, M.shape[1], M.shape[1])
            yy = M[:, j].mean(axis=1)
            bb.append((spearman(X[i], yy[i]), theil_sen(X[i], yy[i])))
        print(f"\n  {label}   Spearman rho = {rho:+.4f}   Theil-Sen slope = {sl:+.2f} pp per unit KL")
        for nm, b in (("BOOT-FOLD", bf), ("BOOT-TEAM", bt), ("BOOT-BOTH", bb)):
            rl, rh = ci([z[0] for z in b]); sl_l, sl_h = ci([z[1] for z in b])
            tag = "excludes 0" if not (rl <= 0 <= rh) else "SPANS 0"
            print(f"    {nm}  rho CI [{rl:+.4f},{rh:+.4f}] {tag:10s} "
                  f"| slope CI [{sl_l:+.1f},{sl_h:+.1f}]  width_rho {rh-rl:.3f}")

    # (iii) D_on
    print()
    print("=" * 108)
    print("(iii)  D_on -- does ON-SLICE distance predict the untaught delta?")
    print("=" * 108)
    rho_on = spearman(xon, y)
    b = [spearman(xon[i], y[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(NB//4))]
    lo, hi = ci(b)
    print(f"  FOLD unit   Spearman(D_on, delta) = {rho_on:+.4f}  BOOT-FOLD CI [{lo:+.4f},{hi:+.4f}]"
          f"  {'excludes 0' if not (lo<=0<=hi) else 'SPANS 0'}")
    print(f"  corr(D_off, D_on) over folds = {np.corrcoef(x, xon)[0,1]:+.4f}  "
          f"over points = {np.corrcoef(px, [pts[k][0]['D_on'] for k in pts])[0,1]:+.4f}")
    print("  -> D_off and D_on are near-collinear across these sets, so (iii) cannot be scored as")
    print("     a partial: there is no set that is far off-slice and near on-slice, or vice versa.")

    # (ii) separation
    print()
    print("=" * 108)
    print("(ii)  Does D_off SEPARATE robbers from neutral/gifting folds?")
    print("=" * 108)
    rob = [r for r in fit if r["ci"][1] < 0]
    neu = [r for r in fit if r["ci"][0] <= 0 <= r["ci"][1]]
    gif = [r for r in fit if r["ci"][0] > 0]
    for nm, g in (("ROBBED (CI<0)", rob), ("NEUTRAL (CI spans 0)", neu), ("GIFTED (CI>0)", gif)):
        if g:
            print(f"  {nm:22s} n={len(g):2d}  D_off {min(r['D_off'] for r in g):.4f}"
                  f"..{max(r['D_off'] for r in g):.4f}  folds: "
                  + ", ".join(sorted(r['fold'] for r in g)))
        else:
            print(f"  {nm:22s} n= 0")
    if rob and neu:
        ov = (min(r["D_off"] for r in rob) <= max(r["D_off"] for r in neu)
              and min(r["D_off"] for r in neu) <= max(r["D_off"] for r in rob))
        print(f"  -> ranges {'OVERLAP' if ov else 'are DISJOINT'}")

    # ASCII scatter
    print()
    print("=" * 108)
    print("ASCII SCATTER  (x = D_off, y = untaught delta pp).  letters = folds, "
          "* = C1 control (excluded)")
    print("=" * 108)
    allr = rows
    xs = [r["D_off"] for r in allr]; ys = [r["delta"] for r in allr]
    x0, x1 = min(xs) - .02, max(xs) + .02
    y0, y1 = min(ys) - 1, max(ys) + 1
    W, H = 84, 22
    grid = [[" "] * W for _ in range(H)]
    keys = {}
    for i, r in enumerate(sorted(allr, key=lambda z: z["D_off"])):
        ch = "*" if not r["in_fit"] else chr(ord("a") + i)
        keys[ch] = r
        cx = int((r["D_off"] - x0) / (x1 - x0) * (W - 1))
        cy = int((y1 - r["delta"]) / (y1 - y0) * (H - 1))
        grid[cy][cx] = ch if grid[cy][cx] == " " else "#"
    zc = int((y1 - 0) / (y1 - y0) * (H - 1))
    for c in range(W):
        if grid[zc][c] == " ":
            grid[zc][c] = "-"
    for rix, row in enumerate(grid):
        lab = f"{y1 - rix*(y1-y0)/(H-1):+6.1f} |"
        print(lab + "".join(row))
    print(" " * 7 + "+" + "-" * W)
    print(" " * 8 + f"{x0:.3f}" + " " * (W - 12) + f"{x1:.3f}")
    print("  key: " + "  ".join(f"{c}={keys[c]['fold']}({keys[c]['set']})" for c in
                                sorted(keys, key=lambda c: keys[c]["D_off"])))

    # leave-one-out
    print()
    print("=" * 108)
    print("LEAVE-ONE-OUT (POINT unit -- removing a whole teacher-set point)")
    print("=" * 108)
    for i, nm in enumerate(pnames):
        m = np.arange(len(px)) != i
        rho = spearman(px[m], py[m])
        b = [spearman(px[m], pP[m][:, RNG.integers(0, 8, 8)].mean(axis=1)) for _ in range(4000)]
        lo, hi = ci(b)
        print(f"  drop {nm:18s} -> n={m.sum()}  rho {rho:+.4f}  BOOT-TEAM CI [{lo:+.4f},{hi:+.4f}]"
              f"  {'excludes 0' if not (lo<=0<=hi) else 'SPANS 0'}")

    # within-parent only (the clean contrast)
    wp = [r for r in fit if r["parent_run"].startswith("ai_v9_59")]
    wx = np.array([r["D_off"] for r in wp]); wy = np.array([r["delta"] for r in wp])
    wP = np.array([r["per_team"] for r in wp])
    print()
    print("=" * 108)
    print(f"WITHIN-PARENT ONLY (parent = R2ACTION; n_folds={len(wp)}, "
          f"n_points={len(set(r['set'] for r in wp))}) -- the clean contrast")
    print("=" * 108)
    rho = spearman(wx, wy)
    b = [spearman(wx, wP[:, RNG.integers(0, 8, 8)].mean(axis=1)) for _ in range(NB // 4)]
    lo, hi = ci(b)
    print(f"  Spearman {rho:+.4f}  BOOT-TEAM CI [{lo:+.4f},{hi:+.4f}] "
          f"{'excludes 0' if not (lo<=0<=hi) else 'SPANS 0'}")
    bf = [spearman(wx[i], wy[i]) for i in (RNG.integers(0, len(wx), len(wx)) for _ in range(NB//4))]
    lo2, hi2 = ci(bf)
    print(f"                     BOOT-FOLD CI [{lo2:+.4f},{hi2:+.4f}] "
          f"{'excludes 0' if not (lo2<=0<=hi2) else 'SPANS 0'}")
    # the pair that carries it
    print("\n  Set means (within-parent):")
    for k in sorted(set(r["set"] for r in wp)):
        g = [r for r in wp if r["set"] == k]
        print(f"    {k:6s} D_off {g[0]['D_off']:.4f}  mean delta {np.mean([r['delta'] for r in g]):+6.2f}"
              f"  (n_folds {len(g)}: " + ", ".join(r["fold"] for r in g) + ")")
    # is it carried by one pair? drop FUND-vs-UNF and see
    for drop in ("FUND", "UNF", "R4set", "R3set"):
        g = [r for r in wp if r["set"] != drop]
        gx = np.array([r["D_off"] for r in g]); gy = np.array([r["delta"] for r in g])
        gP = np.array([r["per_team"] for r in g])
        rr = spearman(gx, gy)
        bb = [spearman(gx, gP[:, RNG.integers(0, 8, 8)].mean(axis=1)) for _ in range(4000)]
        l2, h2 = ci(bb)
        print(f"    drop {drop:6s} -> {len(set(r['set'] for r in g))} points, rho {rr:+.4f} "
              f"BOOT-TEAM CI [{l2:+.4f},{h2:+.4f}] "
              f"{'excludes 0' if not (l2<=0<=h2) else 'SPANS 0'}")

    # replicate-pair floor on the y axis, at IDENTICAL D_off
    print()
    print("=" * 108)
    print("THE y-AXIS FLOOR AT IDENTICAL x  (same teacher set, same parent, same D_off)")
    print("=" * 108)
    for a, b_ in (("N1", "N2"), ("TC_FUND_A", "TC_FUND_B"), ("TC_UNF_A", "TC_UNF_B"),
                  ("TC_UNF_K6_A", "TC_UNF_K6_B")):
        ra = next(r for r in rows if r["fold"] == a); rb = next(r for r in rows if r["fold"] == b_)
        d = ra["per_team"] - rb["per_team"]
        bs = d[RNG.integers(0, 8, (NB, 8))].mean(axis=1)
        lo, hi = ci(bs)
        print(f"  {a:12s} - {b_:12s}  {d.mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}]  "
              f"(D_off identical at {ra['D_off']:.4f})")

    # ---- THE INHERITED GAP, and the budget confound -----------------------------------------
    print()
    print("=" * 108)
    print("THE INHERITED GAP  (every gen-era teacher forked from rev-1 final; the fold parent for "
          "four of the\nfive sets is R2ACTION, which is ITSELF a fold off rev-1)")
    print("=" * 108)
    pg = json.load(open(f"{HERE}/parent_gap.json"))["probes"]
    for k, v in pg.items():
        print(f"  KL({k[:46]:46s} || R2ACTION) = {v['D_off']:.4f}")
    gap = pg["REV1FIN (the gen-era teachers' own fork parent)"]["D_off"]
    print(f"\n  -> a gen-era teacher starts {gap:.4f} from R2ACTION BEFORE it trains one step.")
    print("     For R2set the teachers' fork parent IS the fold parent, so its inherited gap is 0")
    print("     and its D_off is pure EARNED displacement. The other four sets sit on top of "
          f"{gap:.4f}.")
    print("     D_off MINUS the inherited gap (a crude excess -- KL does not subtract, so this is")
    print("     an ordering aid, never a level):")
    for k, v in sorted({(r["parent_run"], r["set"]): r for r in rows}.items(),
                       key=lambda z: z[1]["D_off"]):
        inh = 0.0 if k[1] == "R2set" else gap
        print(f"       {k[1]:6s} D_off {v['D_off']:.4f}  inherited {inh:.4f}  "
              f"excess ~{v['D_off']-inh:+.4f}")

    # teacher BUDGET, the registered confound
    BUDGET_M = {"UNF": 3.07, "R2set": 3.07, "R3set": 5.07, "FUND": 5.07, "R4set": 10.07}
    TEAMS_PER = {"UNF": 2, "R2set": 2, "R3set": 2, "FUND": 2, "R4set": 8}
    print("\n  TEACHER BUDGET (exploiter span in M steps off rev-1 final @25.0M; FUND = UNF + 2.0M):")
    print(f"    {'set':6s} {'budget_M':>9s} {'teams/tchr':>11s} {'D_off':>7s} {'mean delta':>11s}")
    ptsx = {}
    for r in fit:
        ptsx.setdefault(r["set"], []).append(r)
    for k in sorted(ptsx, key=lambda z: BUDGET_M[z]):
        md = float(np.mean([q["delta"] for q in ptsx[k]]))
        print(f"    {k:6s} {BUDGET_M[k]:9.2f} {TEAMS_PER[k]:11d} {ptsx[k][0]['D_off']:7.4f} "
              f"{md:+11.2f}")
    bx = np.array([BUDGET_M[k] for k in ptsx]); dx = np.array([ptsx[k][0]["D_off"] for k in ptsx])
    dy = np.array([np.mean([q["delta"] for q in ptsx[k]]) for k in ptsx])
    print(f"    Spearman(budget, D_off)  = {spearman(bx, dx):+.4f}")
    print(f"    Spearman(budget, delta)  = {spearman(bx, dy):+.4f}")
    print(f"    Spearman(D_off,  delta)  = {spearman(dx, dy):+.4f}")
    print("    -> if these agree, budget and distance are RANK-INDISTINGUISHABLE in this table "
          "and no\n       fold here breaks the confound.")
    ex = np.array([ptsx[k][0]["D_off"] - (0.0 if k == "R2set" else gap) for k in ptsx])
    print(f"    Spearman(EXCESS D_off, delta) = {spearman(ex, dy):+.4f}  "
          f"(excess = D_off - inherited gap)")
    print("    The decisive row: R2set and UNF share a budget (3.07M) and a delta (+0.88/+0.87)")
    print("    while their RAW D_off differs 2.5x (0.2175 vs 0.5536) -- the whole difference is")
    print("    the inherited gap, and it moves the delta by nothing. Raw distance is the wrong")
    print("    axis at the one cross-parent point; EARNED displacement is the right one.")

    # ---- reconciliation against content_locality, which resolved teachers differently ----
    print()
    print("=" * 108)
    print("RECONCILIATION with content_locality (SAME states -- per-team counts identical -- "
          "DIFFERENT checkpoint)")
    print("=" * 108)
    cg = json.load(open(f"{CL}/gen_era_n9.json"))
    print(f"  content_locality untaught state counts 0..7: {cg['_meta']['states_per_team'][:8]}")
    dg = json.load(open(f"{HERE}/dist_gen.json"))
    print(f"  this probe's                            0..7: "
          f"{dg['_meta']['states_per_team'][:8]}")
    same = cg["_meta"]["states_per_team"][:8] == dg["_meta"]["states_per_team"][:8]
    print(f"  -> states {'REPRODUCE exactly' if same else 'DIFFER'}")
    pids = ["00", "02", "04", "06", "08", "10", "12", "14"]
    runs = {"UNF": [f"ai_v9_{n}_R5F{p}_0831" for n, p in
                    zip([92, 94, 96, 98, 100, 102, 104, 106], pids)],
            "FUND": [f"ai_v9_{n}_R5FUND{p}_0901" for n, p in
                     zip([120, 122, 124, 126, 128, 130, 132, 134], pids)]}
    fin, best = {}, {}
    for half, pref in (("UNF", "UNF"), ("FUND", "FUND")):
        fin[half] = np.array([np.mean(cg["per_team_kl_fwd"][f"{pref}{p}"][:8]) for p in pids])
        best[half] = np.array([dg["per_teacher"][r]["d_off"] for r in runs[half]])
        print(f"  {half:5s}  final_model.zip (content_locality) {fin[half].mean():.4f}   "
              f"best_model/best_model.zip (THIS probe, = what the fold loads) "
              f"{best[half].mean():.4f}   delta {best[half].mean()-fin[half].mean():+.4f}")
    bs8t = RNG.integers(0, 8, (NB, 8))
    for nm, d in (("FUNDED-UNFUNDED on final_model  (content_locality's headline)",
                   fin["FUND"] - fin["UNF"]),
                  ("FUNDED-UNFUNDED on best_model   (the checkpoints the fold used)",
                   best["FUND"] - best["UNF"])):
        b = d[bs8t].mean(axis=1)
        lo, hi = ci(b)
        print(f"  {nm}: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{'SIGNIFICANT' if not (lo <= 0 <= hi) else 'NOT DETECTED'}")

    json.dump({"rows": [{k: (list(v) if isinstance(v, np.ndarray) else v)
                         for k, v in r.items()} for r in rows],
               "v8": {"D_off": v8_off, "D_off_content_locality": v8_off_final,
                      "D_on_all_taught": v8_on_all,
                      "floor_mean": v8fx["summary"]["floor_mean"],
                      "note": "D_off CORRECTED by v8_checkpoint_fix.py (best_model/, the rung the "
                              "training path resolves); D_on and the taught column are still "
                              "content_locality's final_model_interrupted numbers. v8's own "
                              "parent, own untaught 8, greedy era meter -- NOT on this y-scale"}},
              open(f"{HERE}/analysis.json", "w"), indent=1)
    print(f"\nwrote {HERE}/analysis.json")


if __name__ == "__main__":
    main()
