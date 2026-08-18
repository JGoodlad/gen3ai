"""Paired BT refit: honest SE(Δ) on the tail-4 ladder contrast.

Reproduces snapshot_ladder.fit_ladder's EXACT inputs (dense frozen matrix + each
snapshot's historical bot/sentinel edges, bots PINNED at gen3_bot_elo_anchors), then
(a) reads the full inverse-Hessian covariance and applies the contrast c'Σc, and
(b) parametric-bootstraps the games and refits, as an independent cross-check.
"""
import json, math, numpy as np
from agents.training import elo as elo_mod
from agents.training import snapshot_ladder as sl

G14 = "models/ai_v9_16_gen14_framedel_v91_0817"
G13 = "models/ai_v9_15_gen13_hb_events_stack_0817"
TAIL = 4

def build_results(run_dir):
    results = []
    games = sl.load_games(run_dir)
    for (lo, hi), (w, g) in games.items():
        if g > 0: results.append((elo_mod.snap_key(lo), elo_mod.snap_key(hi), w, g))
    try:
        for na, nb, wa, g in elo_mod._rows_to_results(elo_mod.load_rows(run_dir, source="log")):
            if g > 0: results.append((na, nb, wa, g))
    except Exception: pass
    return results

anchors = elo_mod.load_bot_anchors() or {}
pins = anchors.get("ratings"); base = anchors.get("base", elo_mod.DEFAULT_BASE)
pinned = {elo_mod.bot_key(n): float(e) for n, e in pins.items()} if pins else None

def fit_with_cov(results):
    """Refit and recover the FULL covariance of the free parameters."""
    names, idx, edges = elo_mod._aggregate(results)
    n = len(names)
    R = np.full(n, base, dtype=np.float64)
    fixed = np.zeros(n, dtype=bool)
    if pinned:
        for k, v in pinned.items():
            if k in idx: R[idx[k]] = v; fixed[idx[k]] = True
    edge_list = [(a, b, w, g) for (a, b), (w, g) in edges.items()]
    inv_var = 1.0 / (elo_mod.DEFAULT_PRIOR_SD ** 2)
    R, se_arr, it, conv = elo_mod._newton(R, fixed, edge_list, base, elo_mod.DEFAULT_PRIOR_SD, 100, 1e-7)
    free = [i for i in range(n) if not fixed[i]]
    free_pos = {p: j for j, p in enumerate(free)}
    _g, Hf = elo_mod._grad_and_hessian(R, fixed, edge_list, free, free_pos, base, inv_var)
    cov = np.linalg.inv(Hf)
    return names, idx, R, cov, free_pos, conv

def tail_contrast(run_dir):
    res = build_results(run_dir)
    names, idx, R, cov, free_pos, conv = fit_with_cov(res)
    steps = sorted(int(k) for k in json.load(open(f"{run_dir}/snapshot_ladder/ladder.json"))["ratings"])
    tail = steps[-TAIL:]
    keys = [elo_mod.snap_key(s) for s in tail]
    pos = [free_pos[idx[k]] for k in keys]
    c = np.zeros(cov.shape[0]); 
    for p in pos: c[p] = 1.0 / TAIL
    mean = float(np.mean([R[idx[k]] for k in keys]))
    var = float(c @ cov @ c)
    naive = float(sum(cov[p, p] for p in pos) / TAIL**2)   # the WRONG independent version
    return mean, var, naive, tail, conv, (names, idx, R, cov, free_pos)

m14, v14, nv14, t14, c14, _ = tail_contrast(G14)
m13, v13, nv13, t13, c13, _ = tail_contrast(G13)
d = m14 - m13
se_joint = math.sqrt(v14 + v13)
se_naive = math.sqrt(nv14 + nv13)
lo, hi = d - 1.96*se_joint, d + 1.96*se_joint
print(f"tail steps: gen-14 {t14}  gen-13 {t13}   converged: {c14}/{c13}")
print(f"tail-4 mean: gen-14 {m14:.1f}   gen-13 {m13:.1f}   Δ = {d:+.2f}")
print(f"  SE(Δ) NAIVE (diag only, what I used first) : {se_naive:.2f}")
print(f"  SE(Δ) JOINT (full c'Σc, correlation-aware) : {se_joint:.2f}")
print(f"  CI95 JOINT = [{lo:+.2f}, {hi:+.2f}]")
verdict = ("NON_INFERIOR" if d >= -15 and lo > -40 else ("INFERIOR" if hi < -15 else "INCONCLUSIVE"))
print(f"  §1 VERDICT = {verdict}")
print(json.dumps({"delta": round(d,2), "se_joint": round(se_joint,2), "se_naive": round(se_naive,2),
                  "ci95": [round(lo,2), round(hi,2)], "verdict": verdict,
                  "tail_steps": {"gen14": t14, "gen13": t13}}, indent=1))

# ── independent cross-check: parametric bootstrap over the GAMES ────────────────
# Resample every edge's wins ~ Binomial(games, p_fitted), refit both runs, recompute Δ.
# This makes no appeal to the Hessian, so agreement with c'Σc validates both.
print("\n=== parametric bootstrap cross-check (refit per draw) ===")
def boot(run_dir, rng, B):
    res = build_results(run_dir)
    names, idx, R, cov, free_pos, conv = fit_with_cov(res)
    steps = sorted(int(k) for k in json.load(open(f"{run_dir}/snapshot_ladder/ladder.json"))["ratings"])
    keys = [elo_mod.snap_key(s) for s in steps[-TAIL:]]
    # fitted win prob per result row, then resample
    out = []
    for _ in range(B):
        sim = []
        for na, nb, w, g in res:
            if na in idx and nb in idx:
                p = elo_mod.win_prob(R[idx[na]], R[idx[nb]])
            else:
                p = w / g if g else 0.5
            sim.append((na, nb, int(rng.binomial(g, min(max(p, 1e-6), 1-1e-6))), g))
        nm2, idx2, R2, _c2, _fp2, _cv2 = fit_with_cov(sim)
        out.append(float(np.mean([R2[idx2[k]] for k in keys if k in idx2])))
    return out

rng = np.random.default_rng(0)
B = 300
b14 = boot(G14, rng, B); b13 = boot(G13, rng, B)
dd = np.array(b14) - np.array(b13)
lo_b, hi_b = np.percentile(dd, 2.5), np.percentile(dd, 97.5)
print(f"  bootstrap B={B}: Δ mean {dd.mean():+.2f}   SE {dd.std(ddof=1):.2f}   CI95 [{lo_b:+.2f}, {hi_b:+.2f}]")
vb = ("NON_INFERIOR" if dd.mean() >= -15 and lo_b > -40 else ("INFERIOR" if hi_b < -15 else "INCONCLUSIVE"))
print(f"  §1 VERDICT (bootstrap) = {vb}")
