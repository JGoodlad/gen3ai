"""TEST 3 - BOOTSTRAP CONSISTENCY on ai_v12_02_winprob_critic (win-prob critic, gamma=1).

MODEL-FREE: reads each trace's recorded per-decision V(s) and reward through
ProbeSession.battle_overview (no checkpoint). Under --critic winprob with gamma=1 and
terminal-only reward the off-terminal reward is exactly 0.0, so
    delta_t = r_t + gamma*V(s_{t+1}) - V(s_t) == V(s_{t+1}) - V(s_t)
and the realized return G(s) is the terminal WIN INDICATOR (0/1) at every t.

Folds:
 (a) TD residual by turn bucket / result / opponent stratum, AND -- the decisive
     instrument -- E[delta | V-level], which is 0 at every level iff the critic is a
     consistent (martingale) bootstrap.
 (b) Calibration of V against the terminal win indicator: Brier + the MURPHY
     decomposition (REL - RES + UNC), the base-rate cap (UNC) and its skill score,
     bias, slope, EV and OVERDISPERSION (Brier / E[V(1-V)]).
 (c) Terminal |V(s_T) - outcome|, and the last turn V crossed 0.5 vs the outcome.

Every interval is a CLUSTER BOOTSTRAP OVER BATTLES -- never pooled decisions (decisions
inside one battle share the same label y and are strongly dependent).
"""
import json, os, sys
import numpy as np
from collections import defaultdict

RUN = "/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic"
OUT = "/home/goodlad/.claude/jobs/9ab51de6/tmp/bootstrap_75M"
STEPS = [int(s) for s in sys.argv[1:]] or [50000016, 60000000, 70000032, 74000016]
MAX_TURNS = 250
NBOOT = 2000
BOTS = {"random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"}
BUCKETS = ("early(<=10)", "mid(11-24)", "late(>=25)")
VLEVELS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]

from main.prober.session import ProbeSession                        # noqa: E402
from main.prober.session.stats import (_calibration_stats,          # noqa: E402
                                       _reliability_curve)


def bucket(t):
    if t is None:
        return "unknown"
    return "early(<=10)" if t <= 10 else ("mid(11-24)" if t < 25 else "late(>=25)")


def _r(x, n=5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def cluster_boot(groups, stat, n_boot=NBOOT, seed=0):
    """`groups` = one array (or tuple of aligned arrays) per BATTLE. `stat` maps the
    concatenated resample to a scalar. Resamples BATTLES with replacement."""
    groups = [g for g in groups if len(g[0]) > 0]
    if not groups:
        return {"est": None, "ci_lo": None, "ci_hi": None, "n_battles": 0, "n": 0}
    ncol = len(groups[0])
    cat = [np.concatenate([g[c] for g in groups]) for c in range(ncol)]
    est = stat(*cat)
    rng = np.random.default_rng(seed)
    k = len(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        cols = [np.concatenate([groups[i][c] for i in pick]) for c in range(ncol)]
        s = stat(*cols)
        if s is not None and np.isfinite(s):
            boots.append(s)
    if not boots:
        return {"est": _r(est), "ci_lo": None, "ci_hi": None, "n_battles": k, "n": int(cat[0].size)}
    b = np.sort(np.asarray(boots))
    return {"est": _r(est), "ci_lo": _r(b[int(0.025 * b.size)]),
            "ci_hi": _r(b[min(b.size - 1, int(0.975 * b.size))]),
            "n_battles": k, "n": int(cat[0].size)}


def dist(v):
    if len(v) == 0:
        return {}
    s = np.sort(np.asarray(v, dtype=float))
    k5 = max(1, int(0.05 * s.size))
    return {"n": int(s.size), "mean": _r(s.mean()), "sd": _r(s.std()),
            "p1": _r(np.quantile(s, .01)), "p5": _r(np.quantile(s, .05)),
            "median": _r(np.median(s)), "p95": _r(np.quantile(s, .95)),
            "p99": _r(np.quantile(s, .99)), "min": _r(s[0]), "max": _r(s[-1]),
            "cvar05_left": _r(s[:k5].mean()), "cvar05_right": _r(s[-k5:].mean()),
            "frac_negative": _r(float((s < 0).mean()))}


def sign_runs(seq):
    """Run structure of the residual SIGNS inside one battle. Under a zero-mean
    martingale the signs are ~iid -> mean run length ~2.0; longer runs mean the residual
    DRIFTS (persistent mis-valuation) rather than fluctuating."""
    seq = np.asarray(seq, dtype=float)
    sg = np.sign(seq[seq != 0])
    if sg.size < 2:
        return None
    brk = np.flatnonzero(sg[1:] != sg[:-1])
    runs = np.diff(np.concatenate(([-1], brk, [sg.size - 1])))
    m = seq.mean()
    den = float(((seq - m) ** 2).sum())
    acf1 = float(((seq[:-1] - m) * (seq[1:] - m)).sum() / den) if den > 1e-18 and seq.size > 1 else None
    return {"mean_run": float(runs.mean()), "max_run": int(runs.max()),
            "acf1": acf1, "pos_frac": float((sg > 0).mean())}


# ---- the statistics the bootstrap resamples -------------------------------------
def s_mean(x, *rest):
    return float(x.mean()) if x.size else None


def brier(v, y):
    return float(((v - y) ** 2).mean()) if v.size else None


def overdisp(v, y):
    """Realized squared error over the variance the stated probability CLAIMS.
    >1 = the head is over-confident (more surprises than its own p allows)."""
    den = float((v * (1.0 - v)).mean())
    return (float(((v - y) ** 2).mean()) / den) if den > 1e-12 else None


def bias(v, y):
    return float((v - y).mean()) if v.size else None


def murphy(v, y, n_bins=10):
    """Brier = REL - RES + UNC on equal-count V-bins. REL=0 is perfect calibration;
    RES is resolution (skill at separating); UNC = pbar(1-pbar) is the BASE-RATE CAP."""
    if v.size == 0:
        return None
    pbar = float(y.mean())
    unc = pbar * (1.0 - pbar)
    order = np.argsort(v, kind="stable")
    vs, ys = v[order], y[order]
    rel = res = 0.0
    for idx in np.array_split(np.arange(vs.size), min(max(1, n_bins), vs.size)):
        if idx.size == 0:
            continue
        w = idx.size / vs.size
        rel += w * (vs[idx].mean() - ys[idx].mean()) ** 2
        res += w * (ys[idx].mean() - pbar) ** 2
    return {"brier": brier(v, y), "REL": rel, "RES": res, "UNC_base_rate_cap": unc,
            "base_rate": pbar, "skill_score_vs_base_rate": (1.0 - brier(v, y) / unc) if unc > 1e-12 else None}


def main():
    out = {"run": RUN, "generated": "test 3 BOOTSTRAP CONSISTENCY", "steps": {},
           "definitions": {
               "delta": "r_t + gamma*V(s_{t+1}) - V(s_t); gamma read from the run's metadata",
               "off_terminal_reward": "asserted 0.0 -> delta == dV exactly",
               "G": "terminal WIN INDICATOR (1 win / 0 decisive loss) at gamma=1",
               "buckets": "early <=10, mid 11-24, late >=25 (game turn of the decision)",
               "timeout_rule": f"meta.turns >= {MAX_TURNS} => TIMEOUT-as-LOSS (pre-draw-bucket tree)",
               "strata": "bot = the 9 scripted bots; pool_sentinel = sentinel_0..4 (GREEDY-vs-STOCHASTIC handicap)",
               "overdispersion": "E[(V-y)^2] / E[V(1-V)]; 1.0 = the head's own stated confidence",
               "murphy": "Brier = REL - RES + UNC over 10 equal-count V-bins; UNC = base-rate cap",
               "ci": f"{NBOOT}-resample CLUSTER bootstrap over BATTLES",
               "telescoping_caveat": ("sum_t delta_t = V(s_T) - V(s_0) inside a battle, so a "
                                      "pooled residual mean is dominated by the WIN/LOSS MIX of the "
                                      "captured sample. E[delta | V-level] is the confound-free read."),
           }}

    for step in STEPS:
        root = os.path.join(RUN, "eval_traces", f"step_{step}")
        if not os.path.isdir(root):
            out["steps"][str(step)] = {"error": "no such trace dir"}
            continue
        sess = ProbeSession(root)
        gamma = sess._gamma
        rows = []
        for b in sess.battles():
            ov = sess.battle_overview(b["id"])
            meta = ov["meta"]
            res = (meta.get("result") or "").upper()
            turns = meta.get("turns")
            timeout = turns is not None and turns >= MAX_TURNS
            cls = "WIN" if res == "WIN" else (
                ("TIMEOUT_as_LOSS" if timeout else "LOSS_decisive") if res == "LOSS" else (res or "UNKNOWN"))
            opp = b.get("opponent") or "?"
            stratum = "bot" if opp in BOTS else ("pool_sentinel" if opp.startswith("sentinel") else "other")
            inv = ov["invocations"]
            rows.append({
                "id": b["id"], "short": ov["short_id"], "opp": opp, "stratum": stratum,
                "result": res, "cls": cls, "turns": turns,
                "turn": np.array([(d["turn"] if d["turn"] is not None else -1) for d in inv], float),
                "v": np.array([(d["value"] if d["value"] is not None else np.nan) for d in inv], float),
                "td": np.array([(d["td_residual"] if d["td_residual"] is not None else np.nan) for d in inv], float),
                "dv": np.array([(d["delta_v"] if d["delta_v"] is not None else np.nan) for d in inv], float),
                "r": np.array([(d["reward_total"] if d["reward_total"] is not None else np.nan) for d in inv], float),
            })

        # ---- registered invariants, asserted not assumed ----------------------
        td_dv_viol = int(sum(int(np.nansum(np.abs(r["td"] - r["dv"]) > 1e-9)) for r in rows))
        offterm_nonzero = int(sum(int(np.nansum(np.abs(r["r"][:-1]) > 1e-12)) for r in rows if r["r"].size > 1))
        term_rewards = defaultdict(list)
        for r in rows:
            if r["r"].size:
                term_rewards[r["cls"]].append(float(r["r"][-1]))

        def y_of(r):
            return 1.0 if r["cls"] == "WIN" else (0.0 if r["cls"] == "LOSS_decisive" else None)

        # ---- (a) TD residual folds -------------------------------------------
        def td_groups(sel, bk=None, vlo=None, vhi=None):
            g = []
            for r in rows:
                if not sel(r):
                    continue
                m = np.isfinite(r["td"])
                if bk is not None:
                    m &= np.array([bucket(int(t)) == bk for t in r["turn"]])
                if vlo is not None:
                    m &= (r["v"] >= vlo) & (r["v"] < vhi)
                if m.any():
                    g.append((r["td"][m],))
            return g

        def td_fold(sel, bk=None, vlo=None, vhi=None, seed=0):
            g = td_groups(sel, bk, vlo, vhi)
            res = cluster_boot(g, s_mean, seed=seed)
            res["dist"] = dist(np.concatenate([x[0] for x in g])) if g else {}
            sr = [sign_runs(x[0]) for x in g]
            sr = [x for x in sr if x]
            if sr:
                ac = [x["acf1"] for x in sr if x["acf1"] is not None]
                res["sign_runs"] = {"mean_run_len": _r(np.mean([x["mean_run"] for x in sr])),
                                    "max_run_len": int(max(x["max_run"] for x in sr)),
                                    "mean_acf1": _r(np.mean(ac)) if ac else None,
                                    "mean_pos_frac": _r(np.mean([x["pos_frac"] for x in sr])),
                                    "n_battles_scored": len(sr), "iid_reference_mean_run": 2.0}
            return res

        # ---- (b) calibration folds -------------------------------------------
        def cal_groups(sel, bk=None):
            g = []
            for r in rows:
                y = y_of(r)
                if y is None or not sel(r):
                    continue
                m = np.isfinite(r["v"])
                if bk is not None:
                    m &= np.array([bucket(int(t)) == bk for t in r["turn"]])
                if m.any():
                    g.append((r["v"][m], np.full(int(m.sum()), y)))
            return g

        def cal_fold(sel, bk=None, seed=1):
            g = cal_groups(sel, bk)
            if not g:
                return {"n": 0}
            V = np.concatenate([x[0] for x in g])
            Y = np.concatenate([x[1] for x in g])
            o = {"n_decisions": int(V.size), "n_battles": len(g),
                 "bias_V_minus_y": cluster_boot(g, bias, seed=seed),
                 "brier": cluster_boot(g, brier, seed=seed + 1),
                 "overdispersion": cluster_boot(g, overdisp, seed=seed + 2),
                 "murphy": {k: _r(v) for k, v in murphy(V, Y).items()},
                 "stats": {k: _r(v) if isinstance(v, float) else v
                           for k, v in _calibration_stats(V, Y).items()},
                 "reliability_bins": [{k: _r(v) for k, v in b.items()}
                                      for b in _reliability_curve(V, Y, 10)]}
            return o

        allsel = (lambda r: True)
        st = {
            "gamma": gamma, "critic_currency": sess.critic_currency(),
            "n_battles": len(rows),
            "n_decisions": int(sum(r["v"].size for r in rows)),
            "n_deltas": int(sum(int(np.isfinite(r["td"]).sum()) for r in rows)),
            "invariants": {
                "td_minus_dv_violations": td_dv_viol,
                "nonzero_OFF_terminal_reward_decisions": offterm_nonzero,
                "terminal_reward_by_class": {k: sorted(set(v)) for k, v in term_rewards.items()},
            },
            "counts_by_class": {c: sum(1 for r in rows if r["cls"] == c)
                                for c in sorted({r["cls"] for r in rows})},
            "counts_by_stratum": {s: sum(1 for r in rows if r["stratum"] == s)
                                  for s in sorted({r["stratum"] for r in rows})},
            "td_overall": td_fold(allsel),
            "td_by_bucket": {bk: td_fold(allsel, bk=bk, seed=i) for i, bk in enumerate(BUCKETS)},
            "td_by_result": {c: td_fold(lambda r, c=c: r["cls"] == c, seed=10 + i)
                             for i, c in enumerate(("WIN", "LOSS_decisive", "TIMEOUT_as_LOSS"))},
            "td_by_result_bucket": {c: {bk: td_fold(lambda r, c=c: r["cls"] == c, bk=bk, seed=20 + i)
                                        for i, bk in enumerate(BUCKETS)}
                                    for c in ("WIN", "LOSS_decisive", "TIMEOUT_as_LOSS")},
            "td_by_stratum": {s: td_fold(lambda r, s=s: r["stratum"] == s, seed=30 + i)
                              for i, s in enumerate(("bot", "pool_sentinel"))},
            "td_by_stratum_bucket": {s: {bk: td_fold(lambda r, s=s: r["stratum"] == s, bk=bk, seed=40 + i)
                                         for i, bk in enumerate(BUCKETS)}
                                     for s in ("bot", "pool_sentinel")},
            # THE DECISIVE INSTRUMENT: E[delta | V-level]. Zero at every level <=> a
            # CONSISTENT bootstrap. Systematically negative at high V <=> V drifts DOWN
            # from confident states = WITHIN-GAME DRIFT.
            "td_by_v_level": {f"[{lo:.2f},{hi:.2f})": td_fold(allsel, vlo=lo, vhi=hi, seed=50 + i)
                              for i, (lo, hi) in enumerate(VLEVELS)},
            "td_by_v_level_bot": {f"[{lo:.2f},{hi:.2f})":
                                  td_fold(lambda r: r["stratum"] == "bot", vlo=lo, vhi=hi, seed=60 + i)
                                  for i, (lo, hi) in enumerate(VLEVELS)},
            "calibration_overall": cal_fold(allsel),
            "calibration_by_bucket": {bk: cal_fold(allsel, bk=bk, seed=100 + i)
                                      for i, bk in enumerate(BUCKETS)},
            "calibration_by_stratum": {s: cal_fold(lambda r, s=s: r["stratum"] == s, seed=110 + i)
                                       for i, s in enumerate(("bot", "pool_sentinel"))},
            "calibration_by_stratum_bucket": {
                s: {bk: cal_fold(lambda r, s=s: r["stratum"] == s, bk=bk, seed=120 + i)
                    for i, bk in enumerate(BUCKETS)} for s in ("bot", "pool_sentinel")},
        }

        # ---- (c) terminal anchor + the 0.5 crossing --------------------------
        def terminal(sel):
            g, signed, per = [], [], []
            for r in rows:
                y = y_of(r)
                if y is None or not sel(r):
                    continue
                vv = r["v"][np.isfinite(r["v"])]
                if not vv.size:
                    continue
                g.append((np.array([abs(vv[-1] - y)]),))
                signed.append(vv[-1] - y)
                per.append(float(vv[-1]))
            o = cluster_boot(g, s_mean, seed=200)
            o["signed_mean_V_minus_y"] = _r(np.mean(signed)) if signed else None
            o["V_last_mean"] = _r(np.mean(per)) if per else None
            o["dist_abs_err"] = dist([x[0][0] for x in g]) if g else {}
            return o

        def crossings(sel):
            last_cross, final_side_ok, maxv, maxv_turn, never = [], [], [], [], 0
            for r in rows:
                y = y_of(r)
                if y is None or not sel(r):
                    continue
                m = np.isfinite(r["v"])
                v, t = r["v"][m], r["turn"][m]
                if v.size < 2:
                    continue
                above = v >= 0.5
                idx = np.flatnonzero(above[1:] != above[:-1])
                if idx.size:
                    last_cross.append(float(t[idx[-1] + 1]))
                else:
                    never += 1
                final_side_ok.append(1.0 if (above[-1] == (y == 1.0)) else 0.0)
                j = int(np.argmax(v))
                maxv.append(float(v[j]))
                maxv_turn.append(float(t[j]))
            return {"n": len(final_side_ok),
                    "n_never_crossed": never,
                    "last_cross_turn": dist(last_cross) if last_cross else {},
                    "final_side_matches_outcome_frac": _r(np.mean(final_side_ok)) if final_side_ok else None,
                    "max_V_in_battle": dist(maxv) if maxv else {},
                    "turn_of_max_V": dist(maxv_turn) if maxv_turn else {}}

        st["terminal_error"] = {
            "all": terminal(allsel),
            "WIN": terminal(lambda r: r["cls"] == "WIN"),
            "LOSS_decisive": terminal(lambda r: r["cls"] == "LOSS_decisive"),
            "bot": terminal(lambda r: r["stratum"] == "bot"),
            "pool_sentinel": terminal(lambda r: r["stratum"] == "pool_sentinel"),
        }
        st["crossings"] = {
            "all": crossings(allsel),
            "WIN": crossings(lambda r: r["cls"] == "WIN"),
            "LOSS_decisive": crossings(lambda r: r["cls"] == "LOSS_decisive"),
            "bot_LOSS_decisive": crossings(lambda r: r["stratum"] == "bot" and r["cls"] == "LOSS_decisive"),
        }
        st["per_opponent"] = {
            o: {"n_battles": sum(1 for r in rows if r["opp"] == o),
                "td": td_fold(lambda r, o=o: r["opp"] == o, seed=300),
                "terminal": terminal(lambda r, o=o: r["opp"] == o)}
            for o in sorted({r["opp"] for r in rows})}

        out["steps"][str(step)] = st
        print(f"step {step}: {st['n_battles']}b {st['n_deltas']}d  "
              f"td={st['td_overall']['est']} [{st['td_overall']['ci_lo']},{st['td_overall']['ci_hi']}]  "
              f"late={st['td_by_bucket']['late(>=25)']['est']}  "
              f"term|V-y|={st['terminal_error']['all']['est']}  "
              f"viol={td_dv_viol}/{offterm_nonzero}", flush=True)

    p = os.path.join(OUT, "bootstrap_consistency.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, os.path.getsize(p), "bytes", flush=True)


main()
