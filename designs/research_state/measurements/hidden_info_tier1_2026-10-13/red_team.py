"""THE RED TEAM's re-read of Tier 1 (`design_hidden_info_binding.md` §13.2, §16.10; `red_team.md` in this folder).

READ-ONLY and PURE: numpy over the FROZEN blinded table (sha checked by `unblind_t1.load`), the arm key, the archived
world rows, the pilot's Phase 1 truth rows and the blind replica's per-state IMPs. No model, no engine, no playout.
Every interval uses the registered read's own bootstrap (`unblind_t1.Frame`: battle-clustered, 2,000 draws, seed
20261011, pooled within checkpoint), so each number here is paired with the author's on the same draws.

What it computes (each a section of `red_team.json`; written by the red team AFTER reading the author's and the third
party's results — every read here is POST HOC and labelled so; it changes no registered verdict):

  A  reproduction: G, the four shares and R3's two paired contrasts (must equal `result_t1.json`)
  B  R3 on the ABSOLUTE scale (its sign needs no G, so no F): the s-side IMP(F+U) - IMP(F+M) and the n-side
     IMP(O-M) - IMP(O-U) on every state that holds them; the EXCLUDED states (no F / F+M / F+U) characterised;
     imputations (MAR, the replica's own sigma_M with sigma_U := 0, worst case "F+M recovers the whole oracle and
     F+U nothing") and the TIPPING value of the imputed contrast
  C  TIE-FREE scorers: G, s_M, n_M, s_U, n_U and R3 under LEAF-K8 with ties broken (i) uniformly at random (the exact
     expectation), (ii) by the same source's one-ply CRITIC, (iii) by its TRUNC-4-K8; and the continuous TRUNC-4-K8,
     CRITIC, LEAF-K4 / K1 (sanity: LEAF-K8 / TRUNC-4-K8 / CRITIC re-derived here equal the frozen table's)
  D  multiplicity: R3's one-sided bootstrap tail per side, and the largest confidence level at which R3 still holds
  E  information quantity: per state the never-seen mons and the seen mons' unrevealed move slots; the value per
     never-seen MON, per never-seen FACT and per hidden MOVE SLOT; R3 by k bin and by subgroup (checkpoint, opponent
     class, traced outcome)
  F  the never-seen SPECIES actually held by each arm's worlds, counted from the TEAMS (immune to the mis-keyed
     `facts` records, `8702a56f`), and an honest-draw expectation for F (a U-class leak check the R-GIGO audit never
     ran); the U arms' TEAM TEXT against the bank's true teams (F+U's never-seen slots, O-U's seen slots, team size)
  G  C4 (the frontier at k = 6), C5 (FN against F under every scorer; the M8 curve's quantisation and the M8 curve on
     the continuous TRUNC-4-K8 leaf), C6 (POOL's recovery by whether its worlds ARE the true team), C3 (R-EVAL's
     near-boundary contrast)
  H  R3 by the OPPONENT POPULATION the state's game was played against (scripted bots vs the network's own frozen
     snapshots, "sentinels"; §10.4 registers the opponent class as a condition axis, for s_M only): shares, the
     paired bot - sentinel difference raw and k-adjusted, every bot family per checkpoint
  I  the third class R = the seen mons' items + abilities + spreads: telling R in three contexts, and the full 2 x 2
     of M and U with R TRUE (two-point Shapley values and the interaction), from P (= O-M-U)
  J  the referee's own noise: R3 with the frozen picks judged on each half of the 24 truth seeds

    PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 \
      <env>/bin/python3 -I designs/research_state/measurements/hidden_info_tier1_2026-10-13/red_team.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402
import unblind_t1 as U  # noqa: E402

REP = T.MEAS / "hidden_info_replica_2026-10-11"
OUT = T.HERE / "red_team.json"
TIE = D.TIE
LEAF = "LEAF-K8"


def S(est, vec, n=None, level=95.0):
    return ST.summary(est, vec, n, level)


def lo(v, level=95.0):
    return ST.ci(v, level)[0]


def hi(v, level=95.0):
    return ST.ci(v, level)[1]


# ------------------------------------------------------------------------------------------------
# per-source arrays in POLICY order (index 0 = the policy's top1)
# ------------------------------------------------------------------------------------------------

def truth_arrays_all() -> Dict[str, dict]:
    """{id: {"W": [A, 8], "X4": [A, 8], "crit": [A, 8]}} of the TRUE world (Phase 1 leaf seeds 0..7), policy order."""
    out = {}
    for cp in T.CPS:
        meta = D.stage_meta(cp)
        r1 = {**D.phase1_rows(cp, "registered"), **D.phase1_rows(cp, "supp")}
        for did, m in meta.items():
            r = r1.get(did)
            if r is None or not r.get("ok"):
                continue
            A = D.truth_arrays(r, m)
            o = np.array(A["order"])
            out[did] = {"W": A["W"][o][:, :8], "X4": A["Xd"][4][o][:, :8], "crit": A["crit"][o]}
    return out


def _first_max(s: np.ndarray) -> int:
    return int(np.argmax(s >= s.max() - TIE))


def scorer_imps(A: dict, Tp: np.ndarray) -> Dict[str, float]:
    """IMP of one source at one state under every scorer of section C."""
    leaf = A["W"].astype(np.float64).mean(1)
    x4 = A["X4"].astype(np.float64).mean(1)
    cr = A["crit"].astype(np.float64).mean(1)
    tied = leaf >= leaf.max() - TIE
    out = {"LEAF-K8": Tp[_first_max(leaf)] - Tp[0],
           "LEAF-K4": Tp[_first_max(A["W"][:, :4].astype(np.float64).mean(1))] - Tp[0],
           "LEAF-K1": Tp[_first_max(A["W"][:, :1].astype(np.float64).mean(1))] - Tp[0],
           "TRUNC-4-K8": Tp[_first_max(x4)] - Tp[0],
           "CRITIC": Tp[_first_max(cr)] - Tp[0],
           "LEAF-K8|RAND": float(Tp[tied].mean() - Tp[0])}
    for nm, sec in (("LEAF-K8|CRITIC", cr), ("LEAF-K8|TRUNC4", x4)):
        s2 = np.where(tied, sec, -np.inf)
        out[nm] = Tp[_first_max(s2)] - Tp[0]
    out["_tied"] = float(tied.sum() > 1)
    return {k: float(v) for k, v in out.items()}


SCORERS_C = ["LEAF-K8", "LEAF-K8|RAND", "LEAF-K8|CRITIC", "LEAF-K8|TRUNC4", "TRUNC-4-K8", "CRITIC", "LEAF-K4",
             "LEAF-K1"]


class Imps:
    """imp[(label, scorer)] → [n] over a frame's states (NaN where the source is missing)."""

    def __init__(self, F: U.Frame, name: Dict[str, str], truth: Dict[str, dict], labels):
        self.F, self.name = F, name
        self.v: Dict[tuple, np.ndarray] = {}
        self.tied: Dict[str, np.ndarray] = {}
        self.check = {}
        for lab in labels:
            code = name.get(lab, lab)
            arr = {z: np.full(F.n, np.nan) for z in SCORERS_C}
            tie = np.full(F.n, np.nan)
            mism = {z: 0 for z in ("LEAF-K8", "TRUNC-4-K8", "CRITIC")}
            for i, r in enumerate(F.recs):
                if lab == "O":
                    A = truth.get(r["id"])
                else:
                    w = r["w"].get(code)
                    A = None if w is None else {"W": w["W"], "X4": w["X4"], "crit": w["crit"]}
                if A is None or not r["src"].get(code, {}).get("ok"):
                    continue
                sc = scorer_imps(A, r["Tp"])
                for z in SCORERS_C:
                    arr[z][i] = sc[z]
                tie[i] = sc["_tied"]
                for z in mism:          # the frozen table's IMP must be re-derived exactly
                    if abs(sc[z] - r["src"][code]["imp"][z]) > 1e-6:
                        mism[z] += 1
            for z in SCORERS_C:
                self.v[(lab, z)] = arr[z]
            self.tied[lab] = tie
            self.check[lab] = mism

    def __call__(self, lab, z=LEAF):
        return self.v[(lab, z)]


def ok_of(*xs):
    m = np.ones(len(xs[0]), bool)
    for x in xs:
        m &= np.isfinite(x)
    return m


def shares(F: U.Frame, I: Imps, z: str) -> dict:
    """G, s/n for M and U, R3's two paired contrasts and the one-sided bootstrap tails, under scorer z."""
    o, f = I("O", z), I("F", z)
    fm, fu, om, ou = I("F+M", z), I("F+U", z), I("O-M", z), I("O-U", z)
    bt = F.bt
    base = ok_of(o, f)
    g = bt.mean(o - f, base)
    res = {"G": S(*g, int(base.sum())), "IMP_O": S(*bt.mean(o, base)), "IMP_F": S(*bt.mean(f, base))}
    for X, fx, ox in (("M", fm, om), ("U", fu, ou)):
        ms, mn = base & ok_of(fx), base & ok_of(ox)
        res[f"s_{X}"] = S(*bt.ratio(fx - f, o - f, ms), int(ms.sum()))
        res[f"n_{X}"] = S(*bt.ratio(o - ox, o - f, mn), int(mn.sum()))
    ms = ok_of(o, f, fm, fu)
    mn = ok_of(o, f, om, ou)
    ds = bt.ratio(fu - fm, o - f, ms)
    dn = bt.ratio(om - ou, o - f, mn)
    s_m = bt.ratio(fm - f, o - f, base & ok_of(fm))
    n_m = bt.ratio(o - om, o - f, base & ok_of(om))
    res["s_U_minus_s_M"] = S(*ds, int(ms.sum()))
    res["n_U_minus_n_M"] = S(*dn, int(mn.sum()))
    res["R3"] = bool(ST.gt(lo(ds[1]), 0.0) and ST.gt(lo(dn[1]), 0.0))
    res["tail_s"] = float(np.mean(ds[1][np.isfinite(ds[1])] <= 0))
    res["tail_n"] = float(np.mean(dn[1][np.isfinite(dn[1])] <= 0))
    # the M-only cascade if R3 is set aside (§12.3: R1, then NECESSARY-NOT-SUFFICIENT, then SHARED)
    if ST.lt(hi(s_m[1]), 0.25) and ST.lt(hi(n_m[1]), 0.25):
        casc = "REFUTED (R1)"
    elif ST.ge(lo(n_m[1]), 0.25) and ST.lt(hi(s_m[1]), 0.25):
        casc = "NECESSARY-NOT-SUFFICIENT"
    elif ST.gt(lo(s_m[1]), 0.0) or ST.gt(lo(n_m[1]), 0.0):
        casc = "SHARED"
    else:
        casc = "NOT-DETECTED"
    res["M_cascade_without_R3"] = casc
    res["B3"] = bool(ST.ge(s_m[0], 0.5) and ST.ge(lo(s_m[1]), 0.25))
    res["B4"] = bool(ST.ge(n_m[0], 0.5) and ST.ge(lo(n_m[1]), 0.25))
    return res


def max_level(vec) -> float:
    """The largest two-sided percentile level L (in %) at which the lower bound of ``vec`` is still > 0."""
    v = np.asarray(vec)
    v = v[np.isfinite(v)]
    tail = float(np.mean(v <= 0))
    return round(100.0 * (1.0 - 2.0 * tail), 2)


# ------------------------------------------------------------------------------------------------
# B: R3 without the F requirement, the exclusions, imputations
# ------------------------------------------------------------------------------------------------

def replica_imps() -> Dict[str, dict]:
    raw = json.loads((REP / "imp_per_state.json").read_text())
    out = {}
    for cp, d in raw.items():
        for did, s in d.items():
            dd = ast.literal_eval(s) if isinstance(s, str) else s       # the replica stores a dict's repr
            out[did] = dd
    return out


def section_b(F: U.Frame, I: Imps, rep: Dict[str, dict]) -> dict:
    bt = F.bt
    o, f = I("O"), I("F")
    fm, fu, om, ou = I("F+M"), I("F+U"), I("O-M"), I("O-U")
    ds, dn = fu - fm, om - ou
    out = {}
    m_reg = ok_of(o, f, fm, fu)
    m_abs = ok_of(fm, fu)
    out["s_side_abs"] = {"with_F_required (registered read)": S(*bt.mean(ds, m_reg), int(m_reg.sum())),
                         "F+U_and_F+M_only": S(*bt.mean(ds, m_abs), int(m_abs.sum()))}
    n_reg = ok_of(o, f, om, ou)
    n_abs = ok_of(om, ou)
    out["n_side_abs"] = {"with_F_required (registered read)": S(*bt.mean(dn, n_reg), int(n_reg.sum())),
                         "O-M_and_O-U_only": S(*bt.mean(dn, n_abs), int(n_abs.sum()))}
    exc = ~ok_of(f)
    out["excluded_F"] = {
        "n": int(exc.sum()),
        "k_hist_excluded": {b: int(sum(1 for i in np.where(exc)[0] if F.recs[i]["k_bin"] == b)) for b in T.K_BINS},
        "k_hist_included": {b: int(sum(1 for i in np.where(~exc)[0] if F.recs[i]["k_bin"] == b)) for b in T.K_BINS},
        "IMP_O": {"excluded": S(*bt.mean(o, exc)), "included": S(*bt.mean(o, ~exc))},
        "nu_M = O - (O-M)": {"excluded": S(*bt.mean(o - om, exc & ok_of(om))),
                             "included": S(*bt.mean(o - om, ~exc & ok_of(om)))},
        "nu_U = O - (O-U)": {"excluded": S(*bt.mean(o - ou, exc & ok_of(ou))),
                             "included": S(*bt.mean(o - ou, ~exc & ok_of(ou)))},
        "n-side (O-M) - (O-U)": {"excluded": S(*bt.mean(dn, exc & n_abs)), "included": S(*bt.mean(dn, ~exc & n_abs))},
        "F+M_ok_among_excluded": int((exc & ok_of(fm)).sum()), "F+U_ok_among_excluded": int((exc & ok_of(fu)).sum()),
    }
    # replica rows on the states the original could not fit
    rep_fm = np.array([rep.get(r["id"], {}).get("F+M|LEAF-K8", np.nan) for r in F.recs], dtype=np.float64)
    rep_f = np.array([rep.get(r["id"], {}).get("F|LEAF-K8", np.nan) for r in F.recs], dtype=np.float64)
    rep_om = np.array([rep.get(r["id"], {}).get("O-M|LEAF-K8", np.nan) for r in F.recs], dtype=np.float64)
    rep_o = np.array([rep.get(r["id"], {}).get("O|LEAF-K8", np.nan) for r in F.recs], dtype=np.float64)
    miss_s = ok_of(o) & ~m_abs                      # states the s-side lacks (no F+U or no F+M)
    have_rep = miss_s & ok_of(rep_fm, rep_f, rep_o)
    out["replica_on_missing"] = {
        "states_missing_s_side": int(miss_s.sum()), "of_which_replica_has_F_and_F+M": int(have_rep.sum()),
        "replica_sigma_M there": S(*bt.mean(rep_fm - rep_f, have_rep)),
        "replica_G there": S(*bt.mean(rep_o - rep_f, have_rep)),
        "replica_nu_M there": S(*bt.mean(rep_o - rep_om, have_rep & ok_of(rep_om))),
        "original_nu_M there": S(*bt.mean(o - om, have_rep & ok_of(om))),
        "original_nu_U there": S(*bt.mean(o - ou, have_rep & ok_of(ou))),
        "original_n-side there": S(*bt.mean(dn, have_rep & n_abs)),
    }
    mar = float(np.nansum(ds[m_abs]) / m_abs.sum())
    imps = {}
    for nm, fill in (("MAR (the included mean)", np.where(miss_s, mar, np.nan)),
                     ("replica sigma_M, sigma_U := 0 (else MAR)",
                      np.where(have_rep, -(rep_fm - rep_f), np.where(miss_s, mar, np.nan))),
                     ("replica sigma_M, sigma_U := 0 (else WORST)",
                      np.where(have_rep, -(rep_fm - rep_f), np.where(miss_s, -np.maximum(o, 0.0), np.nan))),
                     ("WORST: F+M recovers the oracle, F+U nothing (-max(IMP_O, 0))",
                      np.where(miss_s, -np.maximum(o, 0.0), np.nan))):
        x = np.where(m_abs, ds, fill)
        m = ok_of(x)
        e, v = bt.mean(x, m)
        imps[nm] = {**S(e, v, int(m.sum())), "lower_gt_0": bool(ST.gt(lo(v), 0.0))}
    # tipping point: the constant imputed value c on every missing state at which the lower bound reaches 0
    def low_at(c):
        x = np.where(m_abs, ds, np.where(miss_s, c, np.nan))
        return lo(bt.mean(x, ok_of(x))[1])
    a, b = -1.0, mar
    for _ in range(40):
        mid = (a + b) / 2
        if low_at(mid) > 0:
            b = mid
        else:
            a = mid
    imps["tipping_value"] = {"c": round(b, 4), "note": "every missing state given IMP(F+U)-IMP(F+M) = c makes the "
                             "s-side lower bound 0; compare the included states' mean and the per-state spread",
                             "included_mean": round(mar, 4),
                             "included_per_state_sd": round(float(np.nanstd(ds[m_abs])), 4)}
    out["s_side_imputations"] = imps
    return out


# ------------------------------------------------------------------------------------------------
# E: information quantity, R3 by subgroup
# ------------------------------------------------------------------------------------------------

def section_e(Freg: U.Frame, Iall: Imps, Fall: U.Frame, name: Dict[str, str]) -> dict:
    out = {}
    bt = Fall.bt
    o, f = Iall("O"), Iall("F")
    fm, fu, om, ou = Iall("F+M"), Iall("F+U"), Iall("O-M"), Iall("O-U")
    n_never = np.array([6 - r["k"] for r in Fall.recs], dtype=np.float64)
    # the seen mons' unrevealed move slots: O-U tells them all, so its provenance counts every one (×8 worlds)
    slots = np.array([(r["hits"].get(name["O-U"]) or [np.nan] * 4)[1] / 8.0 for r in Fall.recs], dtype=np.float64)
    fhits = np.array([(r["hits"].get(name["F"]) or [np.nan] * 4)[0] / 8.0 for r in Fall.recs], dtype=np.float64)
    m = ok_of(o, om, ou, slots)
    out["per_state_counts (all Tier-1S)"] = {
        "never_seen_mons_mean": round(float(n_never[m].mean()), 3),
        "unrevealed_seen_move_slots_mean": round(float(slots[m].mean()), 3),
        "F_true_hidden_moves_held_mean": round(float(np.nanmean(fhits[m])), 3)}
    nu_u, nu_m = o - ou, o - om
    out["value_per_unit (nu, all Tier-1S)"] = {
        "nu_U per never-seen MON": S(*bt.ratio(nu_u, n_never, m)),
        "nu_U per never-seen FACT (x8: species, 4 moves, item, ability, spread)": S(*bt.ratio(nu_u, 8 * n_never, m)),
        "nu_M per hidden MOVE SLOT": S(*bt.ratio(nu_m, slots, m)),
        "nu_M per move slot F gets WRONG": S(*bt.ratio(nu_m, slots - fhits, m & ok_of(fhits))),
        "per-FACT ratio U/M": S(*_ratio_of_ratios(bt, nu_u, 8 * n_never, nu_m, slots, m)),
    }
    ms = ok_of(o, f, fm, fu, slots)
    out["value_per_unit (sigma, all Tier-1S)"] = {
        "sigma_U per never-seen MON": S(*bt.ratio(fu - f, n_never, ms)),
        "sigma_U per never-seen FACT": S(*bt.ratio(fu - f, 8 * n_never, ms)),
        "sigma_M per hidden MOVE SLOT": S(*bt.ratio(fm - f, slots, ms)),
        "per-FACT ratio U/M": S(*_ratio_of_ratios(bt, fu - f, 8 * n_never, fm - f, slots, ms)),
    }
    # R3's two numerators and the absolute class effects by k bin (k = 6: U empty)
    byk = {}
    for b in T.K_BINS:
        sel = np.array([r["k_bin"] == b for r in Fall.recs])
        ms_b, mn_b = sel & ok_of(fm, fu), sel & ok_of(om, ou)
        byk[b] = {"n": int(sel.sum()),
                  "never_seen_mons": round(float(n_never[sel].mean()), 2),
                  "move_slots": round(float(np.nanmean(slots[sel])), 2),
                  "sigma_U - sigma_M": S(*bt.mean(fu - fm, ms_b), int(ms_b.sum())),
                  "nu_U - nu_M": S(*bt.mean(om - ou, mn_b), int(mn_b.sum())),
                  "nu_M": S(*bt.mean(o - om, mn_b)), "nu_U": S(*bt.mean(o - ou, mn_b)),
                  "sigma_M": S(*bt.mean(fm - f, sel & ok_of(fm, f))), "sigma_U": S(*bt.mean(fu - f, sel & ok_of(fu, f)))}
    out["by_k (all Tier-1S)"] = byk
    # subgroup robustness on the registered 600 (absolute numerators: their sign is R3's)
    oR, fmR, fuR, omR, ouR = (U_imp(Freg, name, x) for x in ("O", "F+M", "F+U", "O-M", "O-U"))
    groups = {"end2": lambda r: r["cp"] == "end2", "prod": lambda r: r["cp"] == "prod",
              "opp:bot": lambda r: not r["opp_class"].startswith("sentinel"),
              "opp:sentinel (own snapshots)": lambda r: r["opp_class"].startswith("sentinel"),
              "traced win": lambda r: r["outcome"].startswith("win"),
              "traced loss": lambda r: not r["outcome"].startswith("win")}
    sg = {}
    for gname, pred in groups.items():
        sel = Freg.sel(pred)
        ms_g, mn_g = sel & ok_of(fmR, fuR), sel & ok_of(omR, ouR)
        sg[gname] = {"n": int(sel.sum()), "sigma_U - sigma_M": S(*Freg.bt.mean(fuR - fmR, ms_g)),
                     "nu_U - nu_M": S(*Freg.bt.mean(omR - ouR, mn_g))}
    out["subgroups (registered 600)"] = sg
    return out


def U_imp(F: U.Frame, name, lab, z=LEAF):
    return F.imp(name.get(lab, lab), z)


def _ratio_of_ratios(bt, a, da, b, db, m):
    def sums(x):
        return bt._sum(np.asarray(x, dtype=np.float64), m)
    A, DA, B, DB = sums(a), sums(da), sums(b), sums(db)
    est = (A.sum() / DA.sum()) / (B.sum() / DB.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        vec = ((bt.C @ A) / (bt.C @ DA)) / ((bt.C @ B) / (bt.C @ DB))
    return float(est), vec


# ------------------------------------------------------------------------------------------------
# F: never-seen species held, from the TEAMS
# ------------------------------------------------------------------------------------------------

def _species(team: str):
    out = []
    for e in team.split("]"):
        f = e.split("|")
        out.append("".join(ch for ch in (f[1] or f[0]).lower() if ch.isalnum()))
    return out


def worlds_rows(cp: str, code: str) -> Dict[str, dict]:
    return T.read_dir(T.cp_dir(cp) / "worlds" / code, "worlds_*.jsonl")


def section_f(name: Dict[str, str]) -> dict:
    """Per arm: the share of never-seen slots whose world mon is a TRUE never-seen species (counted from the world
    teams; the true never-seen set = the O-M worlds' never-seen species, which O-M keeps true), and for F the honest
    expectation under the builder's own draw (sequential, ∝ the mean Smogon teammate weight of the revealed species,
    no species twice; Monte Carlo, 400 per state) — a U-class leak check."""
    from main.value_search.world_draws import Prior
    P = Prior()
    res = {}
    rng = np.random.default_rng(20261015)
    exp_hits, obs_hits_f, nslots_f = 0.0, 0, 0
    per_arm = {lab: [0, 0, 0] for lab in ("F", "F+M", "FN", "O-U", "F+U", "POOL", "POOLSETS")}
    pool_rows = []
    for cp in T.CPS:
        om = worlds_rows(cp, name["O-M"])
        rows = {lab: worlds_rows(cp, name[lab]) for lab in per_arm}
        for did, r0 in om.items():
            if not r0.get("ok"):
                continue
            revealed = set(r0["revealed"])
            true_sp = _species(r0["teams"][0])
            true_never = [s for s in true_sp if s not in revealed]
            rp = rows["POOL"].get(did)
            if rp is not None and rp.get("ok"):                 # every state, k = 6 included
                pm = (rp.get("extra") or {}).get("pool", {})
                pool_rows.append({"id": did, "true_drawn": int(sum(pm.get("true_drawn", []))),
                                  "distinct": int(pm.get("passed_distinct_teams", 0))})
            if not true_never:
                continue
            for lab, rr in rows.items():
                r = rr.get(did)
                if r is None or not r.get("ok"):
                    continue
                for tw in r["teams"]:
                    sp = _species(tw)
                    held = sum(1 for s in sp if s not in revealed and s in true_never)
                    per_arm[lab][0] += held
                    per_arm[lab][1] += len(true_never)
                per_arm[lab][2] += 1
                if lab == "F":
                    obs_hits_f += sum(sum(1 for s in _species(tw) if s not in revealed and s in true_never)
                                      for tw in r["teams"])
                    nslots_f += len(true_never) * len(r["teams"])
                    # the honest expectation: the builder's draw, re-simulated
                    cand = [s for s in P.pool if s not in revealed]
                    mates = [P.mates(x) for x in sorted(revealed)]
                    w = np.array([np.mean([mt.get(s, 0.0) for mt in mates]) for s in cand]) if mates else \
                        np.zeros(len(cand))
                    if w.sum() <= 0:
                        w = np.array([P.usage.get(s, 0.0) for s in cand])
                    tn = set(true_never)
                    istrue = np.array([s in tn for s in cand])
                    n = len(true_never)
                    tot = 0.0
                    for _ in range(400):
                        ww = w.copy()
                        h = 0
                        for _k in range(n):
                            p = ww / ww.sum()
                            c = rng.choice(len(cand), p=p)
                            h += int(istrue[c])
                            ww[c] = 0.0
                        tot += h
                    exp_hits += tot / 400.0 * len(r["teams"])
    for lab, (h, n, st) in per_arm.items():
        res[lab] = {"true_never_seen_species_held_share": round(h / n, 4) if n else None, "states": st}
    res["F_leak_check"] = {"observed_held": obs_hits_f, "expected_under_honest_draw": round(exp_hits, 1),
                           "slots": nslots_f, "observed_share": round(obs_hits_f / nslots_f, 4),
                           "expected_share": round(exp_hits / nslots_f, 4),
                           "note": "expected = the builder's sequential teammate-weighted draw WITHOUT the gate's "
                                   "acceptance (clause rejections, the playout check); a gap is the gate's conditioning"}
    res["_pool_rows"] = pool_rows
    return res


def section_f_teams(name: Dict[str, str]) -> dict:
    """The U arms' construction, from the team TEXT against the bank's TRUE team (registered states): F+U's never-seen
    slots ARE the true entries at the true slots, O-U's seen slots ARE the true entries (the gender field aside: the pad
    / placement names it), and the true team size."""
    import replica_compare as RC

    def norm(e: str) -> str:
        f = e.split("|")
        if len(f) > 7:
            f[7] = ""
        return "|".join(f)

    c: Dict[str, int] = {}

    def inc(k, v=1):
        c[k] = c.get(k, 0) + int(v)

    examples = []
    for cp in T.CPS:
        truth = RC.bank_truth(cp)
        fu = T.read_dir(T.cp_dir(cp) / "worlds" / name["F+U"], "worlds_registered_*.jsonl")
        ou = T.read_dir(T.cp_dir(cp) / "worlds" / name["O-U"], "worlds_registered_*.jsonl")
        for did, tt in truth.items():
            te = tt.split("]")
            inc(f"true_team_size_{len(te)}")
            for lab, rows in (("F+U", fu), ("O-U", ou)):
                r = rows.get(did)
                if r is None or not r.get("ok"):
                    continue
                rev = set(r["revealed"])
                for tw in r["teams"]:
                    we = tw.split("]")
                    if len(we) != len(te):
                        inc(f"{lab}_team_size_mismatch")
                        continue
                    for a, b in zip(te, we):
                        seen = _species(a)[0] in rev
                        if lab == "F+U" and not seen:
                            inc("F+U never-seen slots")
                            inc("F+U never-seen slots = the true entry", norm(a) == norm(b))
                        elif lab == "O-U" and seen:
                            inc("O-U seen slots")
                            eq = norm(a) == norm(b)
                            inc("O-U seen slots = the true entry", eq)
                            if not eq and len(examples) < 3:
                                examples.append({"id": did, "true": a, "world": b})
    return {"counts": c, "O-U seen mismatches (examples)": examples}


# ------------------------------------------------------------------------------------------------
# G: C3, C4, C5, C6
# ------------------------------------------------------------------------------------------------

def section_g(fr_reg, fr_all, name, Ireg: Imps, Iall: Imps, pool_rows) -> dict:
    out = {}
    P, A = fr_reg["pooled"], fr_all["pooled"]
    # C5: FN against F under every scorer (all Tier-1S states, where FN ran)
    fnv = {}
    for z in SCORERS_C:
        fn, f = Iall("FN", z), Iall("F", z)
        m = ok_of(fn, f)
        fnv[z] = S(*A.bt.mean(fn - f, m), int(m.sum()))
    out["C5_FN_minus_F"] = fnv
    # the M8 curve's QUANTISATION: per state, the smallest weight p on the TRUE world's leaf that can move the honest
    # LEAF-K8 argmax at all, beyond the tie step (a K = 8 leaf moves in steps of 1/16)
    pmin = []
    for i, r in enumerate(A.recs):
        w = r["w"].get(name["F"])
        if w is None:
            continue
        hf = w["W"].astype(np.float64).mean(1)
        a0 = _first_max(hf)
        tr = TRUTH.get(r["id"])
        if tr is None:
            continue
        tl = tr["W"].astype(np.float64).mean(1)
        best = np.inf
        for b in range(len(hf)):
            if b == a0:
                continue
            dh = hf[a0] - hf[b]          # >= 0
            dt = tl[b] - tl[a0]
            if dt <= 0:
                continue
            if dh <= TIE:
                p = 0.0
            else:
                p = dh / (dh + dt)
            best = min(best, p)
        pmin.append(best)
    pm = np.array(pmin)
    fin = pm[np.isfinite(pm)]
    out["C5_M8_quantisation"] = {
        "states": int(len(pm)), "truth_cannot_move_pick": int((~np.isfinite(pm)).sum()),
        "p_min = 0 (tied: the tie step)": int((fin == 0).sum()),
        "p_min in (0, 0.1]": int(((fin > 0) & (fin <= 0.1 + 1e-9)).sum()),
        "p_min in (0.1, 0.2]": int(((fin > 0.1 + 1e-9) & (fin <= 0.2 + 1e-9)).sum()),
        "p_min in (0.2, 0.35]": int(((fin > 0.2 + 1e-9) & (fin <= 0.35 + 1e-9)).sum()),
        "p_min > 0.35": int((fin > 0.35 + 1e-9).sum()),
        "most common nonzero p_min": sorted([[round(float(v), 4), int(c)] for v, c in
                                             zip(*np.unique(np.round(fin[fin > 0], 4), return_counts=True))],
                                            key=lambda x: -x[1])[:8]}
    # the M8 curve under the CONTINUOUS TRUNC-4-K8 leaf (no tie step), fine grid
    grid = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    cur = np.full((A.n, len(grid)), np.nan)
    for i, r in enumerate(A.recs):
        w = r["w"].get(name["F"])
        tr = TRUTH.get(r["id"])
        if w is None or tr is None:
            continue
        hf = w["X4"].astype(np.float64).mean(1)
        tl = tr["X4"].astype(np.float64).mean(1)
        for j, p in enumerate(grid):
            cur[i, j] = r["Tp"][_first_max(p * tl + (1 - p) * hf)] - r["Tp"][0]
    mm = np.isfinite(cur[:, 0])
    out["C5_M8_TRUNC4_curve"] = {f"{p:g}": S(*A.bt.mean(cur[:, j], mm)) for j, p in enumerate(grid)}
    # C4: the k = 6 bucket of R-STAGE-SEARCH, per checkpoint and pooled, at 95 % and 99.6 %
    c4 = {}
    for cp in list(T.CPS) + ["pooled"]:
        Fx = fr_all[cp]
        m = Fx.ok(name["F"]) & Fx.sel(lambda r: r["k_bin"] == "6")
        d = Fx.bt.mean(Fx.imp(name["F"], "TRUNC-4-K8") - Fx.imp(name["F"], "CRITIC"), m)
        t4 = Fx.bt.mean(Fx.imp(name["F"], "TRUNC-4-K8"), m)
        c4[cp] = {"n": int(m.sum()), "TRUNC4_minus_CRITIC": S(*d), "TRUNC4_minus_CRITIC_99.6": S(*d, level=99.6),
                  "TRUNC4": S(*t4), "TRUNC4_minus_CRITIC_level_at_which_lower_is_0": max_level(d[1])}
    out["C4_k6"] = c4
    # C3: R-EVAL's F contrast, per checkpoint, and the level at which it would read EVALUATOR-BOUND
    h = name["HE@F"]
    c3 = {}
    for cp in list(T.CPS) + ["pooled"]:
        Fx = fr_reg[cp]
        m = Fx.ok(h)
        d = Fx.bt.mean(Fx.imp(h, "HE-TRUNC-4-K8") - Fx.imp(h, "VT-TRUNC-4-K8"), m)
        c3[cp] = {"n": int(m.sum()), "HE-TRUNC-4-K8 - TRUNC-4-K8 (F)": S(*d),
                  "one-sided tail <= +0.0005": float(np.mean(d[1] <= 0.0005)),
                  "level at which lower is 0": max_level(d[1])}
    out["C3_R-EVAL"] = c3
    # C6: POOL's recovery by whether its worlds are the TRUE team
    td = {pr["id"]: pr for pr in pool_rows}
    o, f, pl = Ireg("O"), Ireg("F"), Ireg("POOL")
    allt = P.sel(lambda r: td.get(r["id"], {}).get("true_drawn") == 8)
    sm = P.sel(lambda r: 0 < td.get(r["id"], {}).get("true_drawn", -1) < 8)
    nonet = P.sel(lambda r: td.get(r["id"], {}).get("true_drawn") == 0)
    base = ok_of(o, f, pl)
    c6 = {"states": int(base.sum()),
          "share_all_8_worlds_true_team": round(float((allt & base).sum() / base.sum()), 4),
          "share_some_true": round(float((sm & base).sum() / base.sum()), 4),
          "share_no_true": round(float((nonet & base).sum() / base.sum()), 4),
          "distinct_passed_teams_median": float(np.median([pr["distinct"] for pr in pool_rows])) if pool_rows else None}
    for nm, sel in (("all 8 true", allt), ("some true", sm), ("none true", nonet)):
        m = base & sel
        c6[f"r_POOL | {nm}"] = S(*P.bt.ratio(pl - f, o - f, m), int(m.sum()))
        c6[f"POOL - F | {nm}"] = S(*P.bt.mean(pl - f, m))
    out["C6_POOL"] = c6
    return out


# ------------------------------------------------------------------------------------------------
# H: the OPPONENT POPULATION the state came from (a registered condition axis, §10.4; read here for R3, post hoc)
# ------------------------------------------------------------------------------------------------

def family(r: dict) -> str:
    c = r["opp_class"]
    if c.startswith("sentinel"):
        return "sentinel"
    for b in U.BOTS:
        if c.startswith(b):
            return b
    return c


def section_h(fr_reg, fr_all, name) -> dict:
    out = {}
    for which, fr in (("registered", fr_reg), ("all Tier-1S", fr_all)):
        P = fr["pooled"]
        o, f = P.imp("O"), P.imp(name["F"])
        fm, fu, om, ou = (P.imp(name[x]) for x in ("F+M", "F+U", "O-M", "O-U"))
        ms, mn = ok_of(fu, fm), ok_of(om, ou)
        sent = P.sel(lambda r: family(r) == "sentinel")
        res = {}
        for g, sel in (("bot (scripted opponents)", ~sent), ("sentinel (the network's own snapshots)", sent)):
            base = sel & ok_of(o, f)
            d = {"n": int(sel.sum()), "mean_k": round(float(np.mean([r["k"] for r, s in zip(P.recs, sel) if s])), 2),
                 "G": S(*P.bt.mean(o - f, base))}
            for X, fx, ox in (("M", fm, om), ("U", fu, ou)):
                d[f"s_{X}"] = S(*P.bt.ratio(fx - f, o - f, base & ok_of(fx)))
                d[f"n_{X}"] = S(*P.bt.ratio(o - ox, o - f, base & ok_of(ox)))
            dsx = P.bt.ratio(fu - fm, o - f, base & ms)
            dnx = P.bt.ratio(om - ou, o - f, base & mn)
            d["s_U_minus_s_M"], d["n_U_minus_n_M"] = S(*dsx), S(*dnx)
            d["R3"] = bool(ST.gt(lo(dsx[1]), 0.0) and ST.gt(lo(dnx[1]), 0.0))
            res[g] = d
        for lab, x, m in (("s-side IMP(F+U) - IMP(F+M)", fu - fm, ms), ("n-side IMP(O-M) - IMP(O-U)", om - ou, mn)):
            eb, vb = P.bt.mean(x, m & ~sent)
            es, vs = P.bt.mean(x, m & sent)
            est, vec, wsum = 0.0, 0.0, 0
            for kb in T.K_BINS:
                sel = m & P.sel(lambda r, kb=kb: r["k_bin"] == kb)
                e1, v1 = P.bt.mean(x, sel & ~sent)
                e2, v2 = P.bt.mean(x, sel & sent)
                if np.isfinite(e1) and np.isfinite(e2):
                    est += sel.sum() * (e1 - e2)
                    vec = vec + sel.sum() * (v1 - v2)
                    wsum += sel.sum()
            res[f"bot - sentinel, {lab}"] = {"raw": S(eb - es, vb - vs), "k-adjusted": S(est / wsum, vec / wsum)}
        fams = {}
        for fam in sorted({family(r) for r in P.recs}):
            sel = P.sel(lambda r, fam=fam: family(r) == fam)
            fams[fam] = {"n": int(sel.sum()), "s-side": S(*P.bt.mean(fu - fm, sel & ms)),
                         "n-side": S(*P.bt.mean(om - ou, sel & mn)),
                         "nu_M": S(*P.bt.mean(o - om, sel & mn)), "nu_U": S(*P.bt.mean(o - ou, sel & mn))}
            for cp in T.CPS:
                selc = sel & P.sel(lambda r, cp=cp: r["cp"] == cp)
                fams[fam][cp] = {"n": int(selc.sum()), "s-side": S(*P.bt.mean(fu - fm, selc & ms)),
                                 "n-side": S(*P.bt.mean(om - ou, selc & mn))}
        res["by_family"] = fams
        # is it the information VOLUME? the per-state numerators regressed on the never-seen count, the seen mons'
        # unrevealed move slots and the bot indicator (intercept first)
        k = np.array([r["k"] for r in P.recs], dtype=np.float64)
        slots = np.array([(r["hits"].get(name["O-U"]) or [np.nan] * 4)[1] / 8.0 for r in P.recs])
        Z = np.column_stack([6 - k, slots, (~sent).astype(float)])
        reg = {}
        for lab, y, m in (("n-side", om - ou, mn), ("s-side", fu - fm, ms)):
            beta, bs = P.bt.ols(y, Z, m & np.isfinite(slots))
            reg[lab] = {nm: {"coef": round(float(beta[j]), 5),
                             "ci": [round(float(x), 5) for x in np.nanpercentile(bs[:, j], [2.5, 97.5])]}
                        for j, nm in enumerate(("intercept", "never_seen_mons", "move_slots", "bot"))}
        res["volume_adjusted_regression"] = reg
        per_slot = {}
        for kk in range(1, 7):
            row = {}
            for g, gs in (("bot", ~sent), ("sentinel", sent)):
                sel = gs & (k == kk) & mn & np.isfinite(slots)
                if sel.sum():
                    row[g] = {"n": int(sel.sum()), "move_slots": round(float(slots[sel].mean()), 2),
                              "nu_M per slot": round(float((o - om)[sel].mean() / slots[sel].mean()), 4),
                              "nu_U per never-seen mon": (round(float((o - ou)[sel].mean() / (6 - kk)), 4)
                                                          if kk < 6 else None)}
            per_slot[str(kk)] = row
        res["per_unit_by_k"] = per_slot
        out[which] = res
    return out


# ------------------------------------------------------------------------------------------------
# I: the third class R = the seen mons' items, abilities and spreads (I + A + S), and the R-true 2 x 2
# ------------------------------------------------------------------------------------------------

def section_i(fr_reg, name) -> dict:
    """With R TRUE the four corners of the M x U square exist: P (= O-M-U, the pilot's v2 base), O-U (M true), O-M
    (U true), O. Telling R is read in three contexts: P - F (nothing else true), (O-M) - (F+U) (U true),
    (O-U) - (F+M) (M true). Descriptive; P is the pilot's own v2 builder (registered states only)."""
    P = fr_reg["pooled"]
    o, f, p = P.imp("O"), P.imp(name["F"]), P.imp("P")
    fm, fu, om, ou = (P.imp(name[x]) for x in ("F+M", "F+U", "O-M", "O-U"))
    m = ok_of(o, f, p, fm, fu, om, ou)
    bt = P.bt
    g = bt.mean(o - f, m)
    out = {"states": int(m.sum()), "G": S(*g),
           "R told | nothing else (P - F)": S(*bt.mean(p - f, m)),
           "R told | U true ((O-M) - (F+U))": S(*bt.mean(om - fu, m)),
           "R told | M true ((O-U) - (F+M))": S(*bt.mean(ou - fm, m)),
           "R told | M and U true (O - (F+M+U))": "not run (F+M+U is not a Tier-1 arm)"}
    phi_m = 0.5 * ((ou - p) + (o - om))
    phi_u = 0.5 * ((om - p) + (o - ou))
    inter = o - om - ou + p
    out["R-true 2x2 (base P)"] = {"phi_M": S(*bt.mean(phi_m, m)), "phi_U": S(*bt.mean(phi_u, m)),
                                  "phi_U - phi_M": S(*bt.mean(phi_u - phi_m, m)),
                                  "interaction O - (O-M) - (O-U) + P": S(*bt.mean(inter, m)),
                                  "O - P (the gap with R true)": S(*bt.mean(o - p, m))}
    return out


# ------------------------------------------------------------------------------------------------
# J: the REFEREE's own noise: R3 judged on each HALF of the 24 truth seeds (16..27 / 28..39)
# ------------------------------------------------------------------------------------------------

def section_j(fr_reg, name) -> dict:
    """The picks are fixed (the frozen table's, in Phase 1's action order); only the referee changes: T_A = the mean
    outcome over truth seeds 16..27, T_B over 28..39. A verdict that holds on one half and not the other rests on
    referee noise."""
    P = fr_reg["pooled"]
    halves = {}
    for cp in T.CPS:
        meta = D.stage_meta(cp)
        r1 = D.phase1_rows(cp, "registered")
        for did, r in r1.items():
            if not r.get("ok") or did not in meta:
                continue
            acts = sorted(r["out"], key=int)
            Ob = np.array([[D.DEC[ch] for ch in r["out"][a]] for a in acts])
            tr = D.w(Ob[:, T.N_LEAF:T.N_LEAF + T.N_TRUTH])
            lg = np.array([meta[did]["logits"][int(a)] for a in acts])
            top1 = sorted(range(len(acts)), key=lambda i: (-lg[i], int(acts[i])))[0]
            halves[did] = (tr[:, :12].mean(1), tr[:, 12:].mean(1), top1)
    out = {}
    labs = {"O": "O", "F": name["F"], "F+M": name["F+M"], "F+U": name["F+U"], "O-M": name["O-M"], "O-U": name["O-U"]}
    for h in (0, 1):
        imp = {}
        for lab, code in labs.items():
            x = np.full(P.n, np.nan)
            for i, r in enumerate(P.recs):
                s = r["src"].get(code, {})
                if not s.get("ok") or r["id"] not in halves:
                    continue
                Th, top1 = halves[r["id"]][h], halves[r["id"]][2]
                x[i] = Th[s["pick"][LEAF]] - Th[top1]
            imp[lab] = x
        o, f = imp["O"], imp["F"]
        ms, mn = ok_of(o, f, imp["F+M"], imp["F+U"]), ok_of(o, f, imp["O-M"], imp["O-U"])
        ds = P.bt.ratio(imp["F+U"] - imp["F+M"], o - f, ms)
        dn = P.bt.ratio(imp["O-M"] - imp["O-U"], o - f, mn)
        out["truth seeds 16-27" if h == 0 else "truth seeds 28-39"] = {
            "G": S(*P.bt.mean(o - f, ok_of(o, f))),
            "s_M": S(*P.bt.ratio(imp["F+M"] - f, o - f, ok_of(o, f, imp["F+M"]))),
            "n_M": S(*P.bt.ratio(o - imp["O-M"], o - f, ok_of(o, f, imp["O-M"]))),
            "s_U": S(*P.bt.ratio(imp["F+U"] - f, o - f, ok_of(o, f, imp["F+U"]))),
            "n_U": S(*P.bt.ratio(o - imp["O-U"], o - f, ok_of(o, f, imp["O-U"]))),
            "s_U_minus_s_M": S(*ds), "n_U_minus_n_M": S(*dn),
            "R3": bool(ST.gt(lo(ds[1]), 0.0) and ST.gt(lo(dn[1]), 0.0))}
    return out


TRUTH: Dict[str, dict] = {}


def main():
    global TRUTH
    tab, name, sha = U.load()
    fr_reg, fr_all = U.frames(tab, "registered"), U.frames(tab, "all")
    TRUTH = truth_arrays_all()
    labs = ["O", "F", "F+M", "F+U", "O-M", "O-U", "FN", "POOL"]
    Ireg = Imps(fr_reg["pooled"], name, TRUTH, labs)
    Iall = Imps(fr_all["pooled"], name, TRUTH, ["O", "F", "F+M", "F+U", "O-M", "O-U", "FN"])
    res = {"status": "RED TEAM, POST HOC (read after the author's and the third party's results); no registered "
                     "verdict changes", "table_sha256": sha,
           "rederivation_mismatches (must be 0)": {"registered": Ireg.check, "all": Iall.check}}
    P = fr_reg["pooled"]
    # A + C: every scorer, registered 600 pooled, and per checkpoint
    res["C_scorers"] = {}
    for z in SCORERS_C:
        res["C_scorers"][z] = shares(P, Ireg, z)
        res["C_scorers"][z]["R3_max_level_s"] = max_level(
            P.bt.ratio(Ireg("F+U", z) - Ireg("F+M", z), Ireg("O", z) - Ireg("F", z),
                       ok_of(Ireg("O", z), Ireg("F", z), Ireg("F+M", z), Ireg("F+U", z)))[1])
        res["C_scorers"][z]["R3_max_level_n"] = max_level(
            P.bt.ratio(Ireg("O-M", z) - Ireg("O-U", z), Ireg("O", z) - Ireg("F", z),
                       ok_of(Ireg("O", z), Ireg("F", z), Ireg("O-M", z), Ireg("O-U", z)))[1])
    res["tied_share (registered, LEAF-K8 argmax tied)"] = {
        lab: S(*P.bt.mean(Ireg.tied[lab], ok_of(Ireg.tied[lab]))) for lab in labs}
    per_cp = {}
    for cp in T.CPS:
        Ic = Imps(fr_reg[cp], name, TRUTH, ["O", "F", "F+M", "F+U", "O-M", "O-U"])
        per_cp[cp] = {z: {k: v for k, v in shares(fr_reg[cp], Ic, z).items()
                          if k in ("G", "s_M", "n_M", "s_U", "n_U", "s_U_minus_s_M", "n_U_minus_n_M", "R3")}
                      for z in ("LEAF-K8", "LEAF-K8|RAND", "LEAF-K8|CRITIC", "TRUNC-4-K8")}
    res["C_per_cp"] = per_cp
    # B
    res["B_exclusions"] = section_b(P, Ireg, replica_imps())
    # E
    res["E_information"] = section_e(P, Iall, fr_all["pooled"], name)
    # F
    sf = section_f(name)
    pool_rows = sf.pop("_pool_rows")
    res["F_species_held"] = sf
    res["F_u_arm_team_text"] = section_f_teams(name)
    # G
    res["G_other_claims"] = section_g(fr_reg, fr_all, name, Ireg, Iall, pool_rows)
    res["H_opponent_population"] = section_h(fr_reg, fr_all, name)
    res["I_items_abilities_spreads"] = section_i(fr_reg, name)
    res["J_referee_halves"] = section_j(fr_reg, name)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({z: {k: res["C_scorers"][z][k] for k in ("G", "s_M", "n_M", "s_U", "n_U", "s_U_minus_s_M",
                                                               "n_U_minus_n_M", "R3")}
                      for z in SCORERS_C}, indent=1))


if __name__ == "__main__":
    main()
