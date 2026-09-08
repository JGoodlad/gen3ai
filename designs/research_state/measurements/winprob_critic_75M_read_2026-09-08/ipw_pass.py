"""TEST 3 addendum -- the QUOTA-REWEIGHTED (IPW) read.

The trace tree is LOSS-ENRICHED BY DESIGN (`eval_manifest.selection_rule`: first 10 losses
and first 5 wins per opponent per cycle). Selecting battles on their OUTCOME -- a FUTURE
event -- breaks the martingale property E[delta_t | V_t] = 0 even for a perfectly consistent
critic, so the raw conditional-on-V residual cannot by itself separate the two hypotheses.

The manifest records `capture_rate_win` / `capture_rate_loss` per opponent, so each traced
battle is reweighted by w = 1 / capture_rate(opponent, outcome). That recovers the
POPULATION of battles actually played (100 per opponent) and makes E[delta | V] a real
martingale test again. Every interval is a WEIGHTED CLUSTER BOOTSTRAP OVER BATTLES.
"""
import json, os, sys
import numpy as np

RUN = "/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic"
OUT = "/home/goodlad/.claude/jobs/9ab51de6/tmp/bootstrap_75M"
STEPS = [int(s) for s in sys.argv[1:]] or [50000016, 60000000, 70000032, 74000016]
MAX_TURNS, NBOOT = 250, 2000
BOTS = {"random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"}
BUCKETS = ("early(<=10)", "mid(11-24)", "late(>=25)")
VLEVELS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
from main.prober.session import ProbeSession   # noqa: E402


def _r(x, n=5):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def bucket(t):
    return "early(<=10)" if t <= 10 else ("mid(11-24)" if t < 25 else "late(>=25)")


def wboot(units, stat, seed=0, n_boot=NBOOT):
    """`units` = per-BATTLE (weight, cols...). `stat(w_perdecision, *cols) -> scalar`."""
    units = [u for u in units if u[1].size]
    if not units:
        return {"est": None, "ci_lo": None, "ci_hi": None, "n_battles": 0, "n": 0}
    ncol = len(units[0]) - 1

    def ev(sel):
        w = np.concatenate([np.full(sel[i][1].size, sel[i][0]) for i in range(len(sel))])
        cols = [np.concatenate([u[c + 1] for u in sel]) for c in range(ncol)]
        return stat(w, *cols)
    est = ev(units)
    rng = np.random.default_rng(seed)
    k = len(units)
    b = [ev([units[i] for i in rng.integers(0, k, k)]) for _ in range(n_boot)]
    b = np.sort(np.asarray([x for x in b if x is not None and np.isfinite(x)]))
    return {"est": _r(est), "ci_lo": _r(b[int(.025 * b.size)]) if b.size else None,
            "ci_hi": _r(b[min(b.size - 1, int(.975 * b.size))]) if b.size else None,
            "n_battles": k, "n": int(sum(u[1].size for u in units))}


def wmean(w, x):
    return float((w * x).sum() / w.sum()) if w.sum() > 0 else None


def wbrier(w, v, y):
    return float((w * (v - y) ** 2).sum() / w.sum()) if w.sum() > 0 else None


def wbias(w, v, y):
    return float((w * (v - y)).sum() / w.sum()) if w.sum() > 0 else None


def wover(w, v, y):
    den = float((w * v * (1 - v)).sum())
    return (float((w * (v - y) ** 2).sum()) / den) if den > 1e-12 else None


def wmurphy(w, v, y, n_bins=10):
    W = w.sum()
    pbar = float((w * y).sum() / W)
    unc = pbar * (1 - pbar)
    o = np.argsort(v, kind="stable")
    vs, ys, ws = v[o], y[o], w[o]
    rel = res = 0.0
    for idx in np.array_split(np.arange(vs.size), min(n_bins, vs.size)):
        if not idx.size:
            continue
        bw = ws[idx].sum()
        if bw <= 0:
            continue
        vm = float((ws[idx] * vs[idx]).sum() / bw)
        ym = float((ws[idx] * ys[idx]).sum() / bw)
        rel += (bw / W) * (vm - ym) ** 2
        res += (bw / W) * (ym - pbar) ** 2
    br = wbrier(w, v, y)
    return {"brier": _r(br), "REL": _r(rel), "RES": _r(res), "UNC_base_rate_cap": _r(unc),
            "base_rate_reweighted": _r(pbar),
            "skill_score_vs_base_rate": _r(1 - br / unc) if unc > 1e-12 else None}


out = {"run": RUN, "method": "inverse-capture-probability weighting to the played population",
       "steps": {}}
for step in STEPS:
    root = os.path.join(RUN, "eval_traces", f"step_{step}")
    if not os.path.isdir(root):
        continue
    man = json.load(open(os.path.join(root, "eval_manifest.json")))
    sel = man["selection"]["opponents"]
    sess = ProbeSession(root)
    rows = []
    for b in sess.battles():
        ov = sess.battle_overview(b["id"])
        meta = ov["meta"]
        res = (meta.get("result") or "").upper()
        turns = meta.get("turns")
        if res == "WIN":
            cls, y = "WIN", 1.0
        elif turns is not None and turns >= MAX_TURNS:
            cls, y = "TIMEOUT_as_LOSS", None
        else:
            cls, y = "LOSS_decisive", 0.0
        opp = b["opponent"]
        rate = sel.get(opp, {}).get("capture_rate_win" if res == "WIN" else "capture_rate_loss")
        if not rate:
            continue
        inv = ov["invocations"]
        v = np.array([d["value"] if d["value"] is not None else np.nan for d in inv], float)
        td = np.array([d["td_residual"] if d["td_residual"] is not None else np.nan for d in inv], float)
        tn = np.array([d["turn"] if d["turn"] is not None else -1 for d in inv], float)
        rows.append({"w": 1.0 / float(rate), "opp": opp, "cls": cls, "y": y,
                     "stratum": "bot" if opp in BOTS else "pool_sentinel",
                     "v": v, "td": td, "turn": tn})

    def td_units(sel_f, bk=None, vlo=None, vhi=None):
        u = []
        for r in rows:
            if not sel_f(r):
                continue
            m = np.isfinite(r["td"])
            if bk:
                m &= np.array([bucket(int(t)) == bk for t in r["turn"]])
            if vlo is not None:
                m &= (r["v"] >= vlo) & (r["v"] < vhi)
            if m.any():
                u.append((r["w"], r["td"][m]))
        return u

    def cal_units(sel_f, bk=None):
        u = []
        for r in rows:
            if r["y"] is None or not sel_f(r):
                continue
            m = np.isfinite(r["v"])
            if bk:
                m &= np.array([bucket(int(t)) == bk for t in r["turn"]])
            if m.any():
                u.append((r["w"], r["v"][m], np.full(int(m.sum()), r["y"])))
        return u

    def cal(sel_f, bk=None, seed=1):
        u = cal_units(sel_f, bk)
        if not u:
            return {"n": 0}
        W = np.concatenate([np.full(x[1].size, x[0]) for x in u])
        V = np.concatenate([x[1] for x in u])
        Y = np.concatenate([x[2] for x in u])
        return {"n_decisions": int(V.size), "n_battles": len(u),
                "bias_V_minus_y": wboot(u, wbias, seed=seed),
                "brier": wboot(u, wbrier, seed=seed + 1),
                "overdispersion": wboot(u, wover, seed=seed + 2),
                "murphy": wmurphy(W, V, Y)}

    def term(sel_f, seed=9):
        u, sgn, wl = [], [], []
        for r in rows:
            if r["y"] is None or not sel_f(r):
                continue
            vv = r["v"][np.isfinite(r["v"])]
            if vv.size:
                u.append((r["w"], np.array([abs(vv[-1] - r["y"])])))
                sgn.append((r["w"], vv[-1] - r["y"]))
                wl.append((r["w"], vv[-1]))
        o = wboot(u, wmean, seed=seed)
        tw = sum(w for w, _ in sgn)
        o["signed_mean_V_minus_y"] = _r(sum(w * x for w, x in sgn) / tw) if tw else None
        o["V_last_mean"] = _r(sum(w * x for w, x in wl) / tw) if tw else None
        return o

    allf = (lambda r: True)
    nw = sum(r["w"] for r in rows if r["cls"] == "WIN")
    nt = sum(r["w"] for r in rows if r["y"] is not None)
    out["steps"][str(step)] = {
        "n_battles_traced": len(rows),
        "effective_population_battles": _r(sum(r["w"] for r in rows), 1),
        "captured_win_fraction": _r(sum(1 for r in rows if r["cls"] == "WIN") / len(rows)),
        "reweighted_win_rate": _r(nw / nt) if nt else None,
        "td_overall": wboot(td_units(allf), wmean, seed=0),
        "td_by_bucket": {bk: wboot(td_units(allf, bk=bk), wmean, seed=i) for i, bk in enumerate(BUCKETS)},
        "td_by_v_level": {f"[{lo:.2f},{hi:.2f})": wboot(td_units(allf, vlo=lo, vhi=hi), wmean, seed=50 + i)
                          for i, (lo, hi) in enumerate(VLEVELS)},
        "td_by_v_level_bot": {f"[{lo:.2f},{hi:.2f})":
                              wboot(td_units(lambda r: r["stratum"] == "bot", vlo=lo, vhi=hi), wmean, seed=60 + i)
                              for i, (lo, hi) in enumerate(VLEVELS)},
        "td_by_stratum": {s: wboot(td_units(lambda r, s=s: r["stratum"] == s), wmean, seed=70 + i)
                          for i, s in enumerate(("bot", "pool_sentinel"))},
        "calibration_overall": cal(allf),
        "calibration_by_bucket": {bk: cal(allf, bk=bk, seed=100 + i) for i, bk in enumerate(BUCKETS)},
        "calibration_by_stratum": {s: cal(lambda r, s=s: r["stratum"] == s, seed=110 + i)
                                   for i, s in enumerate(("bot", "pool_sentinel"))},
        "calibration_by_stratum_bucket": {
            s: {bk: cal(lambda r, s=s: r["stratum"] == s, bk=bk, seed=120 + i) for i, bk in enumerate(BUCKETS)}
            for s in ("bot", "pool_sentinel")},
        "terminal_error": {k: term(fn) for k, fn in (
            ("all", allf), ("WIN", lambda r: r["cls"] == "WIN"),
            ("LOSS_decisive", lambda r: r["cls"] == "LOSS_decisive"),
            ("bot", lambda r: r["stratum"] == "bot"),
            ("pool_sentinel", lambda r: r["stratum"] == "pool_sentinel"))},
    }
    st = out["steps"][str(step)]
    print(f"step {step}: traced {st['n_battles_traced']} -> pop {st['effective_population_battles']}, "
          f"win {st['captured_win_fraction']} -> {st['reweighted_win_rate']}; "
          f"td {st['td_overall']['est']} [{st['td_overall']['ci_lo']},{st['td_overall']['ci_hi']}]; "
          f"term {st['terminal_error']['all']['est']}", flush=True)

p = os.path.join(OUT, "bootstrap_consistency_ipw.json")
json.dump(out, open(p, "w"), indent=1)
print("wrote", p, os.path.getsize(p), flush=True)
