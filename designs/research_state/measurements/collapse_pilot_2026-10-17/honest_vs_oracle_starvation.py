"""Can a DEPLOYABLE (honest-world) one-ply rollout read DETECT action starvation, compared with the oracle read?

POST HOC, DESCRIPTIVE re-analysis of data that already exists (the value-discrimination pilot's Phase 1 truth and the
hidden-information binding study's Tier 1 arm F). Nothing here was registered; no new playouts; pure numpy.

STARVATION (the definitions of ``main.policy_spectrum.truth``, reused through ``turn_truth``): an action is NEAR-BEST
when its paired gap to the best action is <= EPS = 0.1 (outcomes on the +-1 scale); DOMINATED when gap - 1.96 SE > EPS;
a DECISIVE turn has >= 1 dominated action; a STARVED pair is (turn, near-best action with policy mass pi < 0.01); a
STARVED turn is a decisive turn with >= 1 starved pair. pi = softmax of the root logits over the legal actions.

INPUTS (read-only):
- truth / oracle (TRUE world): ``~/gen3ai_archive/value_disc_pilot/<cp>/rows/shard_*.jsonl`` (registered 600) and
  ``~/gen3ai_archive/hidden_info_tier1/<cp>/rows_supp/shard_*.jsonl`` (the Tier-1S supplement); ``out[a]`` = 40 chars
  '+'/'-'/'0' (searched side's win / loss / tie-or-truncated), seeds 0..15 = LEAF, 16..39 = held-out TRUTH
  (``common.py``); ``crit[a]["q"]`` = 8 one-ply critic values on seeds 40..47, ALREADY on +-1 (``2 V - 1`` or a
  terminal +-1/0, ``run_playouts.py``).
- the policy: ``states_<cp>.json`` (pilot) / ``states_supp_<cp>.json`` (Tier 1): 11 root ``logits`` and ``top``.
- arm F (fully honest worlds) = code T12 (``arm_key.json``, label "F", spec "off"; sha256 checked against the sealed
  ``arm_key.sha256``): ``~/gen3ai_archive/hidden_info_tier1/<cp>/rows/T12/rows_*.jsonl``; ``out[a]`` = 8 chars, world j
  on dice seed j (the same dice as oracle leaf seed j); ``crit[a]["q"]`` = the honest one-ply critic, world j on seed
  40 + j, +-1. A row with ok = False is a builder exclusion (counted).

ESTIMATORS (each its own near-best set: within EPS of ITS best): ORACLE_K8 / ORACLE_K16 (leaf seeds 0..7 / 0..15),
HONEST_F_K8 (the 8 F worlds), CRITIC_ORACLE, CRITIC_F (mean one-ply q); BASE_ALL = the CHANCE baseline (every
action near-best: flags every pi < 0.01 pair, so its precision is the base rate). Flagged pair = pi < 0.01 AND
E-near-best.
Compared on the states that have BOTH a truth row and an F row (``common``); the truth alone is also reported on every
truth row (``truth_all``). Sets: ``registered`` (the pilot's 600 per checkpoint, primary) and ``all`` (+ supplement).

RULE 8: a (state, action) pair whose pi is within 1e-6 of 0.01, or whose truth gap / dominance margin or any estimator
gap is within 1e-9 of EPS, is EXCLUDED from every pair count and turn flag (counted). Bootstrap: battle-clustered
(battle = the id up to the first '#'), 2,000 draws, seed 20261011, percentile 95 %; pooled resamples battles WITHIN
each checkpoint. Values on the +-1 scale (win probability = half); ``imp_wp`` is the pilot's win-probability IMP.

    cd <worktree> && PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 5-7,13-15 \\
      <env>/bin/python3 designs/research_state/measurements/collapse_pilot_2026-10-17/honest_vs_oracle_starvation.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from main.policy_spectrum.truth import EPS, STARVE, Z, turn_truth

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
PILOT_DIR = MEAS / "value_disc_pilot_2026-10-11"
T1_DIR = MEAS / "hidden_info_tier1_2026-10-13"
ARCH = Path.home() / "gen3ai_archive"
OUT = HERE / "honest_vs_oracle_starvation.json"
CPS = ("end2", "prod")
N_LEAF, N_TRUTH, J = 16, 24, 8
DEC = {"+": 1.0, "-": -1.0, "0": 0.0}
PI_EDGE, GAP_EDGE, TIE = 1e-6, 1e-9, 1e-6
B, SEED = 2000, 20261011
EST = ("ORACLE_K8", "ORACLE_K16", "HONEST_F_K8", "CRITIC_ORACLE", "CRITIC_F", "BASE_ALL")


def jrows(paths: list[Path], audit: dict[str, int]) -> dict[str, dict]:
    """{id: LAST row} (``t1_common.read_dir``'s rule); duplicates counted and checked for agreement."""
    out: dict[str, dict] = {}
    for p in sorted(paths):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                audit["torn_lines"] += 1
                continue
            prev = out.get(r["id"])
            if prev is not None:
                audit["duplicate_rows"] += 1
                if (prev.get("ok"), prev.get("out"), prev.get("crit")) != (r.get("ok"), r.get("out"), r.get("crit")):
                    audit["duplicates_disagree"] += 1
            out[r["id"]] = r
    return out


def f_code() -> str:
    key = json.loads((T1_DIR / "arm_key.json").read_text())
    sealed = (T1_DIR / "arm_key.sha256").read_text().split()[0]
    if hashlib.sha256((T1_DIR / "arm_key.json").read_bytes()).hexdigest() != sealed:
        raise SystemExit("arm_key.json does not match its sealed sha256")
    labels = sorted(c["label"] for c in key["codes"].values())
    rank = sorted(labels, key=lambda lab: hashlib.sha256((key["salt"] + lab).encode()).hexdigest())
    code = f"T{rank.index('F') + 1:02d}"
    if key["codes"][code]["label"] != "F" or key["codes"][code]["spec"] != "off":
        raise SystemExit(f"code scheme does not reproduce F = {code}")
    return code


def outcomes(row: dict, acts: list[str], n: int) -> np.ndarray:
    o = np.array([[DEC[ch] for ch in row["out"][a]] for a in acts], dtype=np.float64)
    if o.shape[1] != n:
        raise RuntimeError(f"{row['id']}: {o.shape[1]} outcomes, not {n}")
    return o


def crit_mean(row: dict, acts: list[str]) -> np.ndarray:
    q = np.array([row["crit"][a]["q"] for a in acts], dtype=np.float64)
    if q.shape[1] != 8 or np.isnan(q).any() or np.abs(q).max() > 1.0 + 1e-9:
        raise RuntimeError(f"{row['id']}: critic q not 8 values on +-1")
    return q.mean(1)


def near_of(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = v.max() - v
    return g <= EPS, np.abs(g - EPS) < GAP_EDGE


def state_rec(cp: str, did: str, meta: dict, tr: dict, fr: dict | None, audit: dict[str, int]) -> dict:
    acts = sorted(tr["out"], key=int)
    lg = np.array([meta["logits"][int(a)] for a in acts], dtype=np.float64)
    order = sorted(range(len(acts)), key=lambda i: (-lg[i], int(acts[i])))
    if int(acts[order[0]]) != int(meta["top"][0]):
        raise RuntimeError(f"{did}: top1 {acts[order[0]]} != registered {meta['top'][0]}")
    z = np.exp(lg - lg.max())
    pi = z / z.sum()
    O = outcomes(tr, acts, N_LEAF + N_TRUTH)
    Tt = O[:, N_LEAF:]
    t = turn_truth({"outcomes": {a: Tt[i].tolist() for i, a in enumerate(acts)}}, EPS)
    if [int(a) for a in acts] != t["actions"]:
        raise RuntimeError(f"{did}: action order")
    excl = (np.abs(pi - STARVE) < PI_EDGE) | (np.abs(t["gap"] - EPS) < GAP_EDGE) \
        | (np.abs(t["gap"] - Z * t["se"] - EPS) < GAP_EDGE)
    est: dict[str, np.ndarray] = {"ORACLE_K8": O[:, :8].mean(1), "ORACLE_K16": O[:, :16].mean(1),
                                  "CRITIC_ORACLE": crit_mean(tr, acts), "BASE_ALL": np.zeros(len(acts))}
    if fr is not None:
        if sorted(fr["out"], key=int) != acts:
            audit["f_action_set_mismatch"] += 1
            fr = None
        else:
            est["HONEST_F_K8"] = outcomes(fr, acts, J).mean(1)
            est["CRITIC_F"] = crit_mean(fr, acts)
    near_e = {}
    for k, v in est.items():
        n, e = near_of(v)
        near_e[k] = n
        excl |= e
    audit["pairs_excluded_rule8"] += int(excl.sum())
    keep = ~excl
    starve = pi < STARVE
    halves = []
    for lo, hi in ((0, 12), (12, 24)):
        th = turn_truth({"outcomes": {a: Tt[i, lo:hi].tolist() for i, a in enumerate(acts)}}, EPS)
        halves.append(th["near"])
    return {"cp": cp, "id": did, "battle": f"{cp}:{did.split('#')[0]}", "acts": acts, "order": order, "pi": pi,
            "keep": keep, "starve": starve, "T": t["value"], "near": t["near"], "decisive": bool(t["dominated"][keep].any()),
            "est": est, "near_e": near_e, "has_f": fr is not None, "halves": halves,
            "se8": O[:, :8].std(1, ddof=1) / np.sqrt(8), "se16": O[:, :16].std(1, ddof=1) / np.sqrt(16),
            "se24": Tt.std(1, ddof=1) / np.sqrt(N_TRUTH),
            "seF8": (outcomes(fr, acts, J).std(1, ddof=1) / np.sqrt(J)) if fr is not None else None}


def load(audit: dict[str, int]) -> list[dict]:
    code = f_code()
    audit["f_code"] = int(code[1:])
    recs: list[dict] = []
    for cp in CPS:
        reg = json.loads((PILOT_DIR / f"states_{cp}.json").read_text())["states"]
        sup = json.loads((T1_DIR / f"states_supp_{cp}.json").read_text())["states"]
        truth = jrows(list((ARCH / "value_disc_pilot" / cp / "rows").glob("shard_*.jsonl")), audit)
        truth_s = jrows(list((ARCH / "hidden_info_tier1" / cp / "rows_supp").glob("shard_*.jsonl")), audit)
        f_rows = jrows(list((ARCH / "hidden_info_tier1" / cp / "rows" / code).glob("rows_*.jsonl")), audit)
        if set(reg) & set(sup):
            raise RuntimeError(f"{cp}: a supplement id is also registered")
        for name, states, tr_src in (("registered", reg, truth), ("supp", sup, truth_s)):
            for did in sorted(states):
                tr = tr_src.get(did)
                if tr is None or not tr.get("ok"):
                    audit[f"{cp}_{name}_truth_missing_or_refused"] += 1
                    continue
                fr = f_rows.get(did)
                if fr is None:
                    audit[f"{cp}_{name}_f_missing"] += 1
                elif not fr.get("ok"):
                    audit[f"{cp}_{name}_f_{'excluded' if fr.get('excluded') else 'refused'}"] += 1
                    fr = None
                rec = state_rec(cp, did, states[did], tr, fr, audit)
                rec["set"] = name
                recs.append(rec)
    return recs


class Boot:
    """Battle-clustered percentile bootstrap of a ratio of sums; battles resampled within each checkpoint."""

    def __init__(self, recs: list[dict]):
        self.ub = sorted({r["battle"] for r in recs})
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[r["battle"]] for r in recs])
        rng = np.random.default_rng(SEED)
        self.C = np.zeros((B, len(self.ub)))
        for cp in CPS:
            cols = np.array([i for i, b in enumerate(self.ub) if b.startswith(cp + ":")])
            if len(cols) == 0:
                continue
            draws = cols[rng.integers(0, len(cols), size=(B, len(cols)))]
            for d in range(B):
                np.add.at(self.C[d], draws[d], 1.0)

    def ratio(self, num, den) -> dict:
        num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
        nb = len(self.ub)
        S, N = np.bincount(self.bi, num, nb), np.bincount(self.bi, den, nb)
        if N.sum() <= 0:
            return {"mean": None, "ci": None, "den": 0.0}
        bn, bd = self.C @ S, self.C @ N
        ok = bd > 0
        lo, hi = np.percentile(bn[ok] / bd[ok], [2.5, 97.5])
        return {"mean": S.sum() / N.sum(), "ci": [lo, hi], "den": float(N.sum())}

    def mean(self, x, sel) -> dict:
        sel = np.asarray(sel, dtype=np.float64)
        return self.ratio(np.where(sel > 0, x, 0.0), sel)


def pick(v: np.ndarray, order: list[int], allowed: np.ndarray | None = None) -> int:
    cand = [i for i in order if allowed is None or allowed[i]]
    m = max(v[i] for i in cand)
    return next(i for i in cand if v[i] >= m - TIE)


def analyse(recs: list[dict], truth_only: bool = False) -> dict:
    bt = Boot(recs)
    n = len(recs)
    res: dict = {"n_states": n, "n_battles": len(bt.ub)}
    kp = [r["keep"] for r in recs]
    tsp = [r["near"] & r["starve"] & k for r, k in zip(recs, kp)]                     # truth-starved pairs (any turn)
    dec = np.array([r["decisive"] for r in recs], dtype=float)
    tsp_d = [s & r["decisive"] for s, r in zip(tsp, recs)]                             # ... on decisive turns
    starved_turn = np.array([s.any() for s in tsp_d], dtype=float)
    res["truth"] = {"decisive_share": bt.ratio(dec, np.ones(n)),
                    "starved_share_of_decisive": bt.ratio(starved_turn, dec),
                    "starved_share_of_all": bt.ratio(starved_turn, np.ones(n)),
                    "near_starved_any_turn_share": bt.ratio([s.any() for s in tsp], np.ones(n)),
                    "mean_near_best_per_turn": float(np.mean([(r["near"] & k).sum() for r, k in zip(recs, kp)])),
                    "mean_legal": float(np.mean([len(r["acts"]) for r in recs])),
                    "starved_pairs": int(sum(s.sum() for s in tsp_d)),
                    "starved_pairs_any_turn": int(sum(s.sum() for s in tsp))}
    T1 = np.array([r["T"][r["order"][0]] for r in recs])
    res["truth"]["oracle_upper_imp"] = bt.mean(np.array([r["T"].max() for r in recs]) - T1, np.ones(n))
    if truth_only:
        return res
    res["estimators"] = {}
    for e in EST:
        fl = [r["near_e"][e] & r["starve"] & k for r, k in zip(recs, kp)]
        any_fl = np.array([f.any() for f in fl], dtype=float)
        tp = [(f & s).sum() for f, s in zip(fl, tsp)]
        tp_d = [(f & s).sum() for f, s in zip(fl, tsp_d)]
        fl_d = [(f.sum() if r["decisive"] else 0) for f, r in zip(fl, recs)]
        rho, rsel = np.zeros(n), np.zeros(n)
        for i, r in enumerate(recs):
            v = r["est"][e]
            if len(v) >= 3 and np.ptp(v) > 0 and np.ptp(r["T"]) > 0:
                rho[i], rsel[i] = spearmanr(v, r["T"]).statistic, 1.0
        gain, imp = np.zeros(n), np.zeros(n)
        for i, (r, f) in enumerate(zip(recs, fl)):
            imp[i] = r["T"][pick(r["est"][e], r["order"])] - T1[i]
            if f.any():
                gain[i] = r["T"][pick(r["est"][e], r["order"], f)] - T1[i]
        res["estimators"][e] = {
            "flagged_turn_share_all": bt.ratio(any_fl, np.ones(n)),
            "flagged_turn_share_truth_decisive": bt.ratio(any_fl * dec, dec),
            "flagged_pairs": int(sum(f.sum() for f in fl)),
            "precision_any_turn": bt.ratio(tp, [f.sum() for f in fl]),
            "recall_any_turn": bt.ratio(tp, [s.sum() for s in tsp]),
            "precision_truth_decisive": bt.ratio(tp_d, fl_d),
            "recall_truth_decisive": bt.ratio(tp_d, [s.sum() for s in tsp_d]),
            "turn_flag_recall_starved_turns": bt.ratio(any_fl * starved_turn, starved_turn),
            "turn_flag_precision_vs_starved_turns": bt.ratio(any_fl * starved_turn, any_fl),
            "spearman_within_turn": bt.mean(rho, rsel),
            "value_flagged_move_vs_top1": bt.mean(gain, any_fl),
            "imp_argmax_vs_top1": bt.mean(imp, np.ones(n)),
            "imp_wp": bt.mean(imp / 2.0, np.ones(n)),
            "mean_near_best_per_turn": float(np.mean([(r["near_e"][e] & k).sum() for r, k in zip(recs, kp)]))}
    jac, a_fl, b_fl, ab = [], [], [], []
    for r, k in zip(recs, kp):
        na, nb_ = r["halves"]
        na, nb_ = na & k, nb_ & k
        jac.append((na & nb_).sum() / max(1, (na | nb_).sum()))
        fa, fb = na & r["starve"], nb_ & r["starve"]
        a_fl.append(fa.sum())
        b_fl.append(fb.sum())
        ab.append((fa & fb).sum())
    cat = np.concatenate
    res["noise_floor"] = {
        "mean_action_se_oracle_K8": float(cat([r["se8"] for r in recs]).mean()),
        "mean_action_se_oracle_K16": float(cat([r["se16"] for r in recs]).mean()),
        "mean_action_se_truth_K24": float(cat([r["se24"] for r in recs]).mean()),
        "mean_action_se_F_K8": float(cat([r["seF8"] for r in recs]).mean()),
        "split_half_near_best_jaccard": bt.mean(np.array(jac), np.ones(n)),
        "split_half_starved_pair_agreement_a_given_b": bt.ratio(ab, b_fl),
        "split_half_starved_pair_agreement_b_given_a": bt.ratio(ab, a_fl),
        "near_best_granularity": {"K8": 1.0 / 8, "K16": 1.0 / 16, "K24": 1.0 / 24,
                                  "note": "outcome means step by 1/K on +-1 (a tie scores 0): at K = 8 the step "
                                          "0.125 > EPS, so 'within 0.1 of the best' == 'tied with the best'"}}
    return res


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    audit: dict[str, int] = Counter({k: 0 for k in ("torn_lines", "duplicate_rows", "duplicates_disagree",
                                                    "f_action_set_mismatch")})
    recs = load(audit)
    out: dict = {"label": "POST HOC, DESCRIPTIVE (not registered)", "eps": EPS, "starve_mass": STARVE,
                 "scale": "+-1 outcome (win prob = half); imp_wp in win probability",
                 "audit": dict(audit), "sets": {}}
    for set_name, sets in (("registered", ("registered",)), ("all", ("registered", "supp"))):
        base = [r for r in recs if r["set"] in sets]
        blk: dict = {}
        for scope in CPS + ("pooled",):
            rs = [r for r in base if scope == "pooled" or r["cp"] == scope]
            common = [r for r in rs if r["has_f"]]
            truth_all = analyse(rs, truth_only=True)["truth"]
            truth_all["n_states"] = len(rs)
            blk[scope] = {"truth_all": truth_all, "common": analyse(common)}
        out["sets"][set_name] = blk
    OUT.write_text(json.dumps(rnd(out), indent=1, sort_keys=True) + "\n")
    p = out["sets"]["registered"]["pooled"]["common"]
    print(json.dumps(rnd({"n": p["n_states"], "truth": p["truth"]["starved_share_of_decisive"],
                          "audit": dict(audit)})))


if __name__ == "__main__":
    main()
