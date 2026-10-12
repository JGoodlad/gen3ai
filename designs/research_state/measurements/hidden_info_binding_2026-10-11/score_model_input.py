"""THE MODEL-INPUT AXIS READ, BLINDED (brief section 4): every table is computed BY ARM CODE. This script never reads
``arm_key.json`` (``unblind_model_input.py`` applies the labels afterwards). Pure numpy, no model:

    python -I score_model_input.py <cp> [<cp> ...] [--out result_model_input_blinded.json] [--partial]
           [--raw-root DIR] [--p1-root DIR] [--states-dir DIR] [--pool end2,prod]

The registered read is ``score_model_input.py end2 prod end2_s2002`` (pooled: end2 + prod, the default ``--pool``).

Inputs: the runner's raw rows (``<raw-root>/<cp>/raw/raw_*.jsonl``), Phase 1's rows (``<p1-root>/<states cp>/rows/``;
the truth T and the registered policy order) and ``states_<states cp>.json`` (strata, battles, capture rates).
``end2_s2002`` reads END2's states, Phase 1 rows and truth: its T(a) is the end2-vs-end2 continuation's truth.

Per state, legal actions a (Phase 1's ``out`` keys; the raw row's root mask must agree):
- T(a) = the mean win-scale outcome over Phase 1 seeds 16..39 (``score.state_arrays``' ``truth``); its integer form
  (the sum of +1 / -1 / 0 over the 24 seeds) decides exact ties.
- pi_0 = softmax of BASE's root logits over the legal actions (BASE = the un-hooked forward; TRUE@0.00 is one of the
  coded arms and is bit-identical to it by the runner's teeth); top1_0 = its argmax, ties to the lower action index.
- per code c: pi_c, top1_c likewise; KL(pi_c || pi_0) (masked, nats); FLIP = 1[top1_c != top1_0];
  IMP_POLICY(c) = T(top1_c) - T(top1_0).
- the one-ply critic under c: Qhat_c(a) = the win-scale mean over the 8 one-ply seeds of q, q = 2 V_c - 1 on a captured
  successor and the branch's end value on a terminal one; IMP_CRITIC(c) = T(argmax Qhat_c) - T(top1_0), argmax ties
  (|d| < 1e-6) toward BASE's policy order (``score.pick``). BASE is reported as a reference row (IMP_POLICY 0).
- PAIRWISE(c): over sibling pairs with T not tied (integer sums differ), the share ordered like T by Qhat_c (a Qhat tie
  scores 1/2); a state with no untied pair is left out. TOP12(c): the same on BASE's (top1, top2) pair only.
- ECE(c): Qhat_c(a) against T(a) over every (state, action), 10 equal-width bins on Qhat (floor(10 Qhat), 1.0 -> bin 9),
  sum_b |sum Qhat - sum T| / N.
Contrasts (paired, per state): IMP_POLICY and IMP_CRITIC for every unordered code pair (``Axx-Ayy`` = Axx minus Ayy)
and every code minus BASE (IMP_CRITIC). Strata: unrevealed opponent slots 0-1 / 2-3 / 4-5 (``states``' ``unrevealed``)
and the number of TRUE-targeted move slots at the root (the runner's state-level ``n_true_target_moves``) 0 / 1-4 /
5-8 / 9+. CIs: ``score.Boot`` (battle-clustered, 2,000 draws, seed 20261011, percentile 95 %); raw (unweighted) is the
primary read, ``score.reweights`` (the natural outcome rate) secondary.
TEETH recorded in the result (the read continues): ``top1_mismatch``, on end2 / prod only (their BASE is the registered
network on its own states), the ids whose BASE root top1 differs from ``states``' registered ``top[0]`` (a FINDING,
printed loudly); ``identity``, per code, max |KL|, the FLIP count and the largest per-state |IMP - IMP(BASE)| for the
policy and the critic (the unblinder requires all four to be exactly 0 on TRUE@0.00).
POOLED (``--pool``; the registered rules' pooled read, `unblind_model_input.py`): every per-code metric and contrast
(no ECE, no strata) over the listed checkpoints' states together, state-weighted, on a STRATIFIED cluster bootstrap
(`PooledBoot`: battles resampled within each checkpoint, checkpoint k on its own stream default_rng([20261011, k + 1])).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
sys.path.insert(0, str(PILOT))
import score as PS  # noqa: E402  (the pilot's: state_arrays, Boot, reweights, pick, TIE, B, SEED)

ACT = 11
N_LEAF, N_TRUTH = 16, 24
STATES_OF = {"end2_s2002": "end2"}
DEC = {"+": 1, "-": -1, "0": 0}
UNREV = (("unrev_0-1", 0, 1), ("unrev_2-3", 2, 3), ("unrev_4-5", 4, 5))
TGT = (("tgt_0", 0, 0), ("tgt_1-4", 1, 4), ("tgt_5-8", 5, 8), ("tgt_9+", 9, 10 ** 9))
BASE = "BASE"


def read_jsonl(path: Path):
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load(cp, raw_root, p1_root, states_dir, partial):
    scp = STATES_OF.get(cp, cp)
    st = json.loads((Path(states_dir) / f"states_{scp}.json").read_text())
    raw = {}
    for p in sorted((Path(raw_root) / cp / "raw").glob("raw_*.jsonl")):
        for r in read_jsonl(p):
            raw[r["id"]] = r
    missing = sorted(set(st["states"]) - set(raw))
    if missing and not partial:
        raise SystemExit(f"{cp}: {len(missing)} registered states have no raw row (--partial for a progress read)")
    want = {i for i, r in raw.items() if r.get("ok")}
    p1 = {}
    for p in sorted((Path(p1_root) / scp / "rows").glob("shard_*.jsonl")):
        for r in read_jsonl(p):
            if r.get("id") in want and r.get("ok"):
                p1[r["id"]] = r
    return st, raw, p1, missing


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def order_of(lg, acts):
    return sorted(range(len(acts)), key=lambda i: (-lg[i], int(acts[i])))


def qhat(row, succ, acts, n_seeds=8):
    """Per legal action: the win-scale mean of q over the one-ply seeds (captured: 2V - 1; terminal: the end value)."""
    pos = {a: i for i, a in enumerate(acts)}
    acc = np.zeros(len(acts))
    cnt = np.zeros(len(acts), dtype=int)
    j = 0
    for act, _seed, src, qend in row["branches"]:
        if qend is None:
            q = 2.0 * float(succ[j]) - 1.0
            j += 1
        else:
            q = float(qend)
        i = pos[str(act)]
        acc[i] += q
        cnt[i] += 1
    if j != len(succ):
        raise RuntimeError(f"{row['id']}: {len(succ)} successor values for {j} captured branches")
    if (cnt != n_seeds).any():
        raise RuntimeError(f"{row['id']}: one-ply seeds per action {cnt.tolist()} != {n_seeds}")
    return (acc / cnt + 1.0) / 2.0


def pair_acc(q, tint, pairs):
    """The share of ``pairs`` (i, j) with untied integer truth that ``q`` orders like the truth (a q tie = 1/2)."""
    s, n = 0.0, 0
    for i, j in pairs:
        if tint[i] == tint[j]:
            continue
        n += 1
        d = q[i] - q[j]
        if abs(d) < PS.TIE:
            s += 0.5
        elif (d > 0) == (tint[i] > tint[j]):
            s += 1.0
    return (s / n) if n else float("nan")


def per_state(r1, e, raw, codes):
    S = PS.state_arrays(r1, e)
    acts = S["acts"]
    legal = [i for i in range(ACT) if raw["mask"][i]]
    if sorted(int(a) for a in acts) != legal:
        raise ValueError("mask")
    T = S["truth"]
    tint = np.array([sum(DEC[ch] for ch in r1["out"][a][N_LEAF:N_LEAF + N_TRUTH]) for a in acts])
    ai = [int(a) for a in acts]
    lg0 = np.array([raw["base"]["root"][k] for k in ai], dtype=np.float64)
    pi0 = softmax(lg0)
    o0 = order_of(lg0, acts)
    t1 = o0[0]
    pairs = list(itertools.combinations(range(len(acts)), 2))
    out = {"T": T, "tint": tint, "top12": (o0[0], o0[1]) if len(o0) > 1 else None, "base_top1": ai[t1]}
    rows = {}
    for c in [BASE] + codes:
        src = raw["base"] if c == BASE else raw["arms"][c]
        lg = np.array([src["root"][k] for k in ai], dtype=np.float64)
        pi = softmax(lg)
        tc = order_of(lg, acts)[0]
        q = qhat(raw, src["succ"], acts)
        pk = PS.pick(q, o0)
        top12 = (pair_acc(q, tint, [out["top12"]]) if out["top12"] is not None else float("nan"))
        rows[c] = {
            "KL": float(np.sum(pi * (np.log(pi) - np.log(pi0)))),
            "FLIP": float(tc != t1),
            "IMP_POLICY": float(T[tc] - T[t1]),
            "IMP_CRITIC": float(T[pk] - T[t1]),
            "PAIRWISE": pair_acc(q, tint, pairs),
            "TOP12": top12,
            "q": q,
        }
    out["rows"] = rows
    return out


METRICS = ("KL", "FLIP", "IMP_POLICY", "IMP_CRITIC", "PAIRWISE", "TOP12")


def ece_boot(bt, q, t, bidx, wt):
    """ECE and its battle-clustered percentile CI from per-battle bin sums (the same bootstrap draws as ``bt``)."""
    nb = len(bt.ub)
    b = np.clip(np.floor(q * 10.0).astype(int), 0, 9)
    N = np.zeros((nb, 10))
    SQ = np.zeros((nb, 10))
    ST = np.zeros((nb, 10))
    np.add.at(N, (bidx, b), wt)
    np.add.at(SQ, (bidx, b), wt * q)
    np.add.at(ST, (bidx, b), wt * t)
    est = float(np.abs(SQ.sum(0) - ST.sum(0)).sum() / N.sum())
    bn, bq, btt = bt.C @ N, bt.C @ SQ, bt.C @ ST
    den = bn.sum(1)
    ok = den > 0                      # a draw with none of the selected states (a sparse stratum) is skipped
    bs = np.abs(bq - btt)[ok].sum(1) / den[ok]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": round(est, 5), "ci": [round(float(lo), 5), round(float(hi), 5)], "n": int(len(q))}


def safe_stat(bt, x, sel, weighted):
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool)
    m = m & np.isfinite(x)
    if m.sum() == 0:
        return {"mean": None, "ci": None, "n": 0}
    return bt.stat(np.where(np.isfinite(x), x, 0.0), sel=m, weighted=weighted)


def tables(bt, P, codes, sel, ece_data):
    """Per code (and BASE): every metric raw + re-weighted; the contrasts; ECE."""
    per = {}
    for c in [BASE] + codes:
        per[c] = {"raw": {}, "reweighted": {}}
        for k in METRICS:
            per[c]["raw"][k] = safe_stat(bt, P[c][k], sel, False)
            per[c]["reweighted"][k] = safe_stat(bt, P[c][k], sel, True)
        if ece_data is not None:
            q, t, bidx, wst, sidx = ece_data[c]
            m = np.ones(len(q), bool) if sel is None else np.asarray(sel, bool)[sidx]
            if m.any():
                per[c]["raw"]["ECE"] = ece_boot(bt, q[m], t[m], bidx[m], np.ones(int(m.sum())))
                per[c]["reweighted"]["ECE"] = ece_boot(bt, q[m], t[m], bidx[m], wst[m])
    con = {"IMP_POLICY": {}, "IMP_CRITIC": {}}
    for k in con:
        for a, b in itertools.combinations(codes, 2):
            d = np.asarray(P[a][k]) - np.asarray(P[b][k])
            con[k][f"{a}-{b}"] = {"raw": safe_stat(bt, d, sel, False), "reweighted": safe_stat(bt, d, sel, True)}
    for a in codes:
        d = np.asarray(P[a]["IMP_CRITIC"]) - np.asarray(P[BASE]["IMP_CRITIC"])
        con["IMP_CRITIC"][f"{a}-{BASE}"] = {"raw": safe_stat(bt, d, sel, False),
                                            "reweighted": safe_stat(bt, d, sel, True)}
    return per, con


def read_cp(cp, raw_root, p1_root, states_dir, partial):
    st, raw, p1, missing = load(cp, raw_root, p1_root, states_dir, partial)
    E = st["states"]
    refused = sorted(i for i, r in raw.items() if not r.get("ok"))
    ids = sorted(i for i, r in raw.items() if r.get("ok") and i in E)
    no_p1 = [i for i in ids if i not in p1]
    ids = [i for i in ids if i in p1]
    codes = sorted(raw[ids[0]]["arms"]) if ids else []
    for i in ids:
        if sorted(raw[i]["arms"]) != codes:
            raise SystemExit(f"{cp}: {i} carries codes {sorted(raw[i]['arms'])}, not {codes}")
    P = {c: {k: [] for k in METRICS} for c in [BASE] + codes}
    ece_q = {c: [] for c in [BASE] + codes}
    ece_t, ece_s = [], []
    kept, mask_bad, top1_bad = [], [], []
    own_states = cp not in STATES_OF          # end2 / prod: BASE is the registered network on its own states
    for i in ids:
        try:
            ps = per_state(p1[i], E[i], raw[i], codes)
        except ValueError:
            mask_bad.append(i)
            continue
        if own_states and ps["base_top1"] != int(E[i]["top"][0]):
            top1_bad.append(i)
        kept.append(i)
        si = len(kept) - 1
        for c in [BASE] + codes:
            for k in METRICS:
                P[c][k].append(ps["rows"][c][k])
            ece_q[c].append(ps["rows"][c]["q"])
        ece_t.append(ps["T"])
        ece_s.append(np.full(len(ps["T"]), si))
    e = [E[i] for i in kept]
    battles = [x["battle"] for x in e]
    wts = PS.reweights(st, e)
    bt = PS.Boot(battles, wts)
    res = {"cp": cp, "states_from": STATES_OF.get(cp, cp), "n_states": len(kept), "n_battles": len(set(battles)),
           "missing": len(missing), "refused": refused, "no_phase1_row": no_p1, "mask_mismatch": mask_bad,
           "codes": codes, "top1_checked": own_states, "top1_mismatch": top1_bad}
    if top1_bad:
        print(f"!!! FINDING [{cp}]: BASE's root top1 differs from the registered top1 on {len(top1_bad)} state(s): "
              f"{top1_bad[:10]}{' ...' if len(top1_bad) > 10 else ''} (the read continues)",
              file=sys.stderr, flush=True)
    # the identity teeth's inputs (`unblind_model_input.py` checks the TRUE@0.00 code against them): per code, the
    # largest |KL|, the FLIP count, and the largest per-state |IMP - IMP(BASE)| for the policy and the critic
    res["identity"] = {c: {"max_abs_kl": float(np.max(np.abs(P[c]["KL"]))) if kept else None,
                           "flip_count": int(np.sum(P[c]["FLIP"])) if kept else None,
                           "max_abs_d_imp_policy": float(np.max(np.abs(np.subtract(P[c]["IMP_POLICY"],
                                                                                   P[BASE]["IMP_POLICY"]))))
                           if kept else None,
                           "max_abs_d_imp_critic": float(np.max(np.abs(np.subtract(P[c]["IMP_CRITIC"],
                                                                                   P[BASE]["IMP_CRITIC"]))))
                           if kept else None}
                       for c in codes}
    internal = {"battles": battles, "wts": wts, "P": P, "codes": codes}
    if not kept:
        return res, internal
    sidx = np.concatenate(ece_s)
    bidx = bt.bi[sidx]
    tt = np.concatenate(ece_t)
    wst = np.asarray(wts, dtype=np.float64)[sidx]
    ece_data = {c: (np.concatenate(ece_q[c]), tt, bidx, wst, sidx) for c in [BASE] + codes}
    per, con = tables(bt, P, codes, None, ece_data)
    res["per_code"], res["contrasts"] = per, con
    strata = {}
    for lab, lo, hi in UNREV:
        strata[lab] = np.array([lo <= x["unrevealed"] <= hi for x in e])
    ntg = [int(raw[i]["n_true_target_moves"]) for i in kept]
    for lab, lo, hi in TGT:
        strata[lab] = np.array([lo <= n <= hi for n in ntg])
    res["strata"] = {}
    for lab, m in strata.items():
        p_, c_ = tables(bt, P, codes, m, ece_data)
        res["strata"][lab] = {"n": int(m.sum()), "per_code": p_, "contrasts": c_}
    # descriptive coverage per code (root targets), no outcome involved
    cov = {}
    for c in codes:
        rs = [raw[i]["arms"][c]["root_stats"] for i in kept]
        cov[c] = {k: round(float(np.mean([x.get(k, 0) for x in rs])), 4)
                  for k in ("slots", "moves", "short", "hyp_slots", "hyp_matched", "never_seen", "never_seen_covered")}
        cov[c]["succ_moves_per_state"] = round(float(np.mean([raw[i]["arms"][c]["succ_moves"] for i in kept])), 3)
    res["coverage"] = cov
    res["mean_legal"] = round(float(np.mean([len(t) for t in ece_t])), 3)
    return res, internal


class PooledBoot:
    """``score.Boot``'s interface over several checkpoints' states CONCATENATED (in ``parts`` order): the bootstrap
    resamples battles WITHIN each checkpoint (a stratified cluster bootstrap; checkpoint k draws from its own stream
    ``default_rng([SEED, k + 1])``, so the strata are independent), and every estimate is the state-weighted ratio
    sum_k S_k / sum_k N_k (the same draws for every statistic, so contrasts stay paired)."""

    def __init__(self, parts):
        self.parts = []
        off = 0
        for k, (battles, weights) in enumerate(parts):
            ub = sorted(set(battles))
            ix = {b: i for i, b in enumerate(ub)}
            bi = np.array([ix[b] for b in battles])
            rng = np.random.default_rng([PS.SEED, k + 1])
            nb = len(ub)
            draws = rng.integers(0, nb, size=(PS.B, nb))
            C = np.zeros((PS.B, nb))
            for r in range(PS.B):
                np.add.at(C[r], draws[r], 1.0)
            self.parts.append((off, off + len(battles), bi, C, np.asarray(weights, dtype=np.float64), nb))
            off += len(battles)
        self.n = off

    def stat(self, x, sel=None, weighted=False):
        x = np.asarray(x, dtype=np.float64)
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool)
        S_tot = N_tot = 0.0
        num = np.zeros(PS.B)
        den = np.zeros(PS.B)
        for lo_, hi_, bi, C, wt, nb in self.parts:
            xs, ms = x[lo_:hi_], m[lo_:hi_]
            w = wt if weighted else np.ones(hi_ - lo_)
            S = np.bincount(bi[ms], weights=(xs * w)[ms], minlength=nb)
            N = np.bincount(bi[ms], weights=w[ms], minlength=nb)
            S_tot += S.sum()
            N_tot += N.sum()
            num += C @ S
            den += C @ N
        ok = den > 0
        lo, hi = np.percentile(num[ok] / den[ok], [2.5, 97.5])
        return {"mean": round(float(S_tot / N_tot), 5), "ci": [round(float(lo), 5), round(float(hi), 5)],
                "n": int(m.sum())}


def pooled(name, internals):
    """The rules' pooled read over several checkpoints (same codes): every per-code metric and contrast (no ECE, no
    strata), on the stratified bootstrap of `PooledBoot`."""
    codes = internals[0]["codes"]
    for it in internals[1:]:
        if it["codes"] != codes:
            raise SystemExit(f"pool {name}: the checkpoints carry different codes")
    pb = PooledBoot([(it["battles"], it["wts"]) for it in internals])
    P = {c: {k: [v for it in internals for v in it["P"][c][k]] for k in METRICS} for c in [BASE] + codes}
    per, con = tables(pb, P, codes, None, None)
    return {"pool": name, "n_states": pb.n, "n_battles": sum(p[5] for p in pb.parts), "codes": codes,
            "per_code": per, "contrasts": con}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cps", nargs="+")
    ap.add_argument("--out", default=str(HERE / "result_model_input_blinded.json"))
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--raw-root", default=str(Path.home() / "gen3ai_archive" / "hidden_info_binding"))
    ap.add_argument("--p1-root", default=str(Path.home() / "gen3ai_archive" / "value_disc_pilot"))
    ap.add_argument("--states-dir", default=str(PILOT))
    ap.add_argument("--pool", default="end2,prod",
                    help="checkpoints pooled for the rules (all must be among the scored ones; '' = none)")
    a = ap.parse_args(argv)
    done = {c: read_cp(c, a.raw_root, a.p1_root, a.states_dir, a.partial) for c in a.cps}
    out = {"blinded": True, "partial": bool(a.partial),
           "config": {"bootstrap_draws": PS.B, "seed": PS.SEED, "tie": PS.TIE,
                      "truth_seeds": [N_LEAF, N_LEAF + N_TRUTH], "ece_bins": 10, "pool": a.pool,
                      "strata": {"unrevealed": UNREV, "true_target_moves": TGT}},
           "cps": {c: r for c, (r, _) in done.items()}}
    pool = [c for c in a.pool.split(",") if c]
    if pool:
        if not set(pool) <= set(done):
            raise SystemExit(f"--pool {pool}: not every pooled checkpoint is scored ({sorted(done)})")
        out["pooled"] = pooled("+".join(pool), [done[c][1] for c in pool])
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return out


if __name__ == "__main__":
    main()
