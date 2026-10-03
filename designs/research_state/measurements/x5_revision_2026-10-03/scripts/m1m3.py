"""M1 (logistic fixed-size marginals: bisection numerics) + M3 (OTHER mass / recall by budget) on the 719 pool teams.
CPU only, read-only. Beliefs: (P) the Smogon T0 prior (cold start); (M) a pool-MEMORISING proxy = geometric blend of the
prior with an IN-SAMPLE pool naive-Bayes teammate model, blend fitted so the matched per-hidden-slot NLL on pool equals the
banked head's 2.55 (belief_calibration_2026-09-24, pool arm, all k)."""
import json, random, torch
from agents.model.belief_tables import build_species_cooccur_prior
from agents.model.t0_species import species_team_prior_logits
from agents.observation.constants import MAX_SPECIES_ID
from agents import gen3_data
from utils.paths import repo_path
torch.set_num_threads(4)
S = MAX_SPECIES_ID + 1
lm, ll = build_species_cooccur_prior(S)
teams_raw = json.load(open(repo_path('data', 'teams', 'gen3_team_archetypes.json')))['teams']
def num(n):
    sp = gen3_data.species.get(n); return sp.num if sp else None
teams = []
for k, v in teams_raw.items():
    nums = [num(s) for s in v['species']]
    if len(nums) == 6 and all(nums) and len(set(nums)) == 6: teams.append(nums)
print("pool teams parsed", len(teams), "of", len(teams_raw))
# in-sample pool naive Bayes (MEASUREMENT proxy only; never a prior)
C = torch.zeros(S, S, dtype=torch.float64); M = torch.zeros(S, dtype=torch.float64)
for t in teams:
    for a in t:
        M[a] += 1
        for b in t:
            if a != b: C[a, b] += 1
alpha = 0.5
Pm = (M + alpha) / (M.sum() + alpha * S)              # P(a random pool mon is s)
cond = (C + alpha * Pm[None, :] * 5) / (M[:, None] * 5 + alpha * 5)   # P(teammate=s | t) smoothed
pool_lift = torch.log(cond / Pm[None, :]).clamp(-6, 6)                 # [t, s]
random.seed(0)
rows = []  # (r, revealed list, hidden list)
for t in teams:
    for r in range(0, 6):
        for _ in range(3 if r else 1):
            perm = random.sample(t, 6); rows.append((r, perm[:r], perm[r:]))
B = len(rows); print("decisions", B)
ids = torch.zeros(B, 6, dtype=torch.long); hid = torch.ones(B, 6, dtype=torch.bool)
truth = torch.zeros(B, S, dtype=torch.float64); kk = torch.zeros(B, dtype=torch.float64); rr = torch.zeros(B, dtype=torch.long)
valid = torch.ones(B, S, dtype=torch.bool); valid[:, 0] = False
for i, (r, rev, hd) in enumerate(rows):
    for j, s in enumerate(rev): ids[i, j] = s; hid[i, j] = False; valid[i, s] = False
    for s in hd: truth[i, s] = 1
    kk[i] = 6 - r; rr[i] = r
prior_log = species_team_prior_logits(lm, ll, ids, hid).double()
onehot = torch.zeros(B, S, dtype=torch.float64)
for i, (r, rev, hd) in enumerate(rows):
    for s in rev: onehot[i, s] = 1
pool_log = torch.log(Pm)[None, :] + onehot @ pool_lift
NEG = -1e30
def norm(a): a = torch.where(valid, a, torch.full_like(a, NEG)); return torch.log_softmax(a, -1)
def slot_nll(lq, mask=None):
    v = -(lq * truth).sum(-1) / truth.sum(-1).clamp(min=1)
    m = (rr < 6) if mask is None else mask
    return v[m].mean().item()
P_log = norm(prior_log)
print("prior per-slot NLL (all r<6):", round(slot_nll(P_log), 3), " banked prior pool 2.88 (k>=1:", round(slot_nll(P_log, (rr>=1)&(rr<6)), 3), ")")
def blend(lam): return norm((1 - lam) * P_log + lam * norm(pool_log))
# fit lam to head NLL 2.55 on r=1..5 (the banked table's k=1..5)
lo, hi = 0.0, 1.0
m15 = (rr >= 1) & (rr < 6)
for _ in range(50):
    mid = (lo + hi) / 2
    if slot_nll(blend(mid), m15) > 2.55: lo = mid
    else: hi = mid
LAM = (lo + hi) / 2
HEAD = {1: 2.24, 2: 2.51, 3: 2.70, 4: 2.61, 5: 2.54}
lams = {}
for r_ in range(1, 6):
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if slot_nll(blend(mid), rr == r_) > HEAD[r_]: lo = mid
        else: hi = mid
    lams[r_] = (lo + hi) / 2
lams[0] = lams[1]
print("per-r lam", {k_: round(v_, 3) for k_, v_ in lams.items()})
lamvec = torch.tensor([lams[int(x)] if x < 6 else 0.0 for x in rr], dtype=torch.float64)[:, None]
Mlog = norm((1 - lamvec) * P_log + lamvec * norm(pool_log))
print(f"memorising proxy lam={LAM:.3f}: NLL r1..5 {slot_nll(Mlog, m15):.3f}; per-r",
      [round(slot_nll(Mlog, rr == r), 2) for r in range(1, 6)], "head per-k 2.24/2.51/2.70/2.61/2.54;",
      "prior per-r", [round(slot_nll(P_log, rr == r), 2) for r in range(1, 6)], "banked 3.25/3.19/2.91/2.65/2.38")
top1 = Mlog.exp().max(-1).values; print("proxy mean top-1 conf (r1..5)", round(top1[m15].mean().item(), 3), "(head ~0.60)")
sharp_log = norm(pool_log)
# EXACT in-sample memorisation ceiling: P(s | R) = share of pool teams containing R that also contain s (+ eps * prior)
team_sets = [set(t) for t in teams]
ex = torch.zeros(B, S, dtype=torch.float64)
for i, (r, rev, hd) in enumerate(rows):
    R_ = set(rev); sup = [t for t in team_sets if R_ <= t]
    for t in sup:
        for s_ in t - R_: ex[i, s_] += 1.0 / len(sup)
def exact_blend(eps): return norm(torch.log(ex + eps * P_log.exp()))
fits = {}
for r_ in range(1, 6):
    lo, hi = -12.0, 3.0   # log10 eps
    for _ in range(50):
        mid = (lo + hi) / 2
        if slot_nll(exact_blend(10 ** mid), rr == r_) > HEAD[r_]: hi = mid
        else: lo = mid
    fits[r_] = 10 ** ((lo + hi) / 2)
fits[0] = fits[1]
print("exact-memo eps per r", {k_: f"{v_:.3g}" for k_, v_ in fits.items()})
epsvec = torch.tensor([fits[int(x)] if x < 6 else 1.0 for x in rr], dtype=torch.float64)[:, None]
Elog = norm(torch.log(ex + epsvec * P_log.exp()))
print("exact-memo per-r NLL", [round(slot_nll(Elog, rr == r_), 2) for r_ in range(1, 6)])

# ---------- the logistic fixed-size construction --------------------------------------------------------
def fixed_size(a, k, valid, iters=64, dtype=torch.float64, newton=True):
    a = a.to(dtype); kd = k.to(dtype)
    big = torch.tensor(1e30, dtype=dtype)
    amax = torch.where(valid, a, -big).max(-1).values; amin = torch.where(valid, a, big).min(-1).values
    nv = valid.sum(-1).to(dtype)
    kc = kd.clamp(min=0.5).minimum(nv - 0.5)   # bracket only; k=0 and k=nv are structural below
    base = torch.log(kc) - torch.log(nv - kc)
    lo = base - amax; hi = base - amin
    def f(tau): return torch.where(valid, torch.sigmoid(a + tau[:, None]), torch.zeros_like(a)).sum(-1)
    with torch.no_grad():
        for _ in range(iters):
            mid = (lo + hi) / 2
            up = f(mid) < kd
            lo = torch.where(up, mid, lo); hi = torch.where(up, hi, mid)
        tau = (lo + hi) / 2
    width0 = (amax - amin)
    if newton:
        p = torch.where(valid, torch.sigmoid(a + tau[:, None]), torch.zeros_like(a))
        tau = tau - (p.sum(-1) - kd) / (p * (1 - p)).sum(-1).clamp(min=torch.finfo(dtype).tiny)
    logit = a + tau[:, None]
    pi = torch.where(valid, torch.sigmoid(logit), torch.zeros_like(a))
    pi = torch.where((kd == 0)[:, None], torch.zeros_like(pi), pi)
    pi = torch.where((kd >= nv)[:, None] & valid, torch.ones_like(pi), pi)
    return pi, logit, width0
def capped_pps(q, k, valid):
    q = torch.where(valid, q, torch.zeros_like(q)); fixed = torch.zeros_like(q, dtype=torch.bool)
    for _ in range(7):
        rest = k[:, None] - fixed.sum(-1, keepdim=True)
        qq = torch.where(fixed, torch.zeros_like(q), q)
        pi = torch.where(fixed, torch.ones_like(q), rest * qq / qq.sum(-1, keepdim=True).clamp(min=1e-300))
        fixed = pi >= 1
    return pi.clamp(max=1)

out = {}
for name, lq in (("prior", P_log), ("memo_NB_proxy", Mlog), ("memo_EXACT_fit_to_head", Elog)):
    m = rr < 6
    a, k, v, T, R = lq[m], kk[m], valid[m], truth[m], rr[m]
    pi64, lg64, width = fixed_size(a, k, v, dtype=torch.float64)
    res = {}
    for dt, nm in ((torch.float64, "fp64"), (torch.float32, "fp32")):
        for nt in (False, True):
            pi_, _, _ = fixed_size(a, k, v, dtype=dt, newton=nt)
            res[f"{nm}{'+newton' if nt else ''}"] = {"max|sum-k|": float((pi_.double().sum(-1) - k).abs().max()),
                                                    "max|pi-pi64|": float((pi_.double() - pi64).abs().max())}
    # BCE finite? max pi
    bce = -(T * torch.nn.functional.logsigmoid(lg64) + (1 - T) * torch.nn.functional.logsigmoid(-lg64)) * v
    cap = capped_pps(a.exp(), k, v)
    capped_any = (cap >= 1 - 1e-12).any(-1); capped_wrong = ((cap >= 1 - 1e-12) & (T == 0) & v).any(-1)
    stats = {"bracket_width_max": float(width.max()), "numerics": res, "max_pi": float(pi64.max()),
             "bce_finite": bool(torch.isfinite(bce).all()), "bce_per_decision_mean": float(bce.sum(-1).mean()),
             "pps_capped_rate": float(capped_any.double().mean()), "pps_capped_wrong_rate": float(capped_wrong.double().mean())}
    # OTHER mass / recall by budget
    order = torch.sort(pi64, dim=-1, descending=True, stable=True)  # ties: stable over num-ascending input order
    srt, idx = order.values, order.indices
    hits = torch.gather(T, 1, idx)
    csum = srt.cumsum(-1); chit = hits.cumsum(-1)
    budgets = {"k": lambda K: K, "k+2": lambda K: K + 2, "k+6": lambda K: K + 6, "12": lambda K: 12, "18": lambda K: 18}
    tab = {}
    for bn, bf in budgets.items():
        rowsR = []
        for r in range(0, 6):
            sel = R == r; K = 6 - r; H = bf(K)
            other = (K - csum[sel, H - 1]); rec = chit[sel, H - 1] / K
            # near-tie at the H-th boundary (rule 8 read)
            tie = (srt[sel, H - 1] - srt[sel, H]).abs() < 1e-6
            rowsR.append((r, H, round(other.mean().item(), 3), round(other.mean().item() / K, 3), round(rec.mean().item(), 3), int(tie.sum())))
        allK = [x for x in rowsR]
        tab[bn] = rowsR
    # pooled over r=0..5 weighted equally per decision
    stats["by_budget"] = tab
    # PPS (Tillé) comparison at budget k
    pc = torch.sort(cap, dim=-1, descending=True, stable=True).values.cumsum(-1)
    stats["pps_other_share_by_r"] = [round(((6 - r) - pc[R == r, 5 - r]).mean().item() / (6 - r), 3) for r in range(6)]
    # calibration of OTHER under the logistic construction: mean OTHER mass vs realised outside count (budget k)
    stats["other_calib_k"] = [(r, round((6 - r - csum[R == r, 5 - r]).mean().item(), 3), round(((6 - r) - chit[R == r, 5 - r]).mean().item(), 3)) for r in range(6)]
    out[name] = stats
    print("\n==", name, json.dumps({k_: v_ for k_, v_ in stats.items() if k_ not in ("by_budget",)}, indent=None))
    for bn, rowsR in tab.items():
        print(f"  budget {bn:>4}: " + " | ".join(f"r{r} H{H} OTHER {o:.2f} ({s:.0%}) rec {rc:.2f} ties {t}" for r, H, o, s, rc, t in rowsR))
json.dump({"lam": LAM, **out}, open(repo_path("designs/research_state/measurements/x5_revision_2026-10-03/out/m1m3_out.json"), "w"), indent=1)
