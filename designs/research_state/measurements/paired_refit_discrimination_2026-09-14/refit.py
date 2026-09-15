"""The PAIRED REFIT — does a pairwise RANKING term give the win-prob head sibling discrimination?

    python3 refit.py --forks <dir> --snapshot <zip> --tree <eval_traces/step_N> --out <dir>

Reads the fork dataset `forks.py` wrote, forwards the FROZEN trunk over every branch's SUCCESSOR
state to get `value_pooled` (the win head's literal input) and the ORIGINAL head's V, then fits the
head two ways and scores both plus the original on held-out forks.

WHAT IS FROZEN AND WHAT MOVES. Only `WinProbHead`'s four tensors move: LayerNorm(128) ->
Linear(128,128) -> ReLU -> Linear(128,1). The trunk is not touched and is not even in the graph —
`value_pooled` is computed once, offline, and the fit is a 4-tensor MLP on a [N,128] matrix. Both
arms are WARM-STARTED from the checkpoint's own head, so arm (i) at step 0 IS the original head
and every movement is attributable to the data or to the loss.

  (i)  BCE on each branch's own outcome                                    -- the CONTROL refit
  (ii) the same BCE + `-log sigma(z_a - z_b)` over the branch pairs inside a fork, where branch a
       won and b lost, ZERO weight on tied pairs, coefficient swept over {0.1, 0.3, 1.0} and
       chosen on the VALIDATION split before the held-out read is taken once.

THE SPLIT IS BY BATTLE, not by fork. Two forks of the same battle share a prefix and often a
successor neighbourhood; splitting by fork would leak. Held-out battles are disjoint from training
and validation battles.

THE READ. The headline is PAIRWISE ACCURACY on held-out NON-TIED pairs: does sign(V_a - V_b) agree
with sign(outcome_a - outcome_b)? Its CI is a bootstrap over FORKS, never over pairs — three pairs
inside one fork share a prefix and are not three observations. Beside it: the separation
(mean |V_a - V_b| on non-tied vs tied pairs), calibration (ECE over 15 equal-mass bins, and Brier)
so ranking is not bought with calibration, and the CONDITIONING GUARD — the turns-4-10 decode of V
to the opponent's CLASS (sentinel vs scripted bot) on the recorded eval frame, the row
`cond.opp_class_auc.t4_10` is computed from.

A branch that hit the 250-turn stall cap is EXCLUDED (its win/loss is decided by SEAT), and so is
a branch with no successor state (the line ended at the divergence turn).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `from forks import ...`

BRANCHES = ("top1", "top2", "rand")


# ---------------------------------------------------------------- stats helpers
def auc(scores, labels) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged."""
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(p, y, bins=15) -> float:
    """Expected calibration error over EQUAL-MASS bins (equal-width bins on a head whose mass sits
    in one corner report a number about the corner)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    tot = 0.0
    for k in range(bins):
        m = (p > qs[k]) & (p <= qs[k + 1])
        if m.sum() == 0:
            continue
        tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def boot_ci(fn, fork_ids, n_boot=2000, seed=7):
    """Bootstrap over FORKS: resample fork ids with replacement, recompute."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(fork_ids)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = fn(pick)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


# ---------------------------------------------------------------- data
def load_forks(fork_dir):
    rows, succ, mask = [], [], []
    off = 0
    for rp in sorted(glob.glob(os.path.join(fork_dir, "forks_s*.jsonl"))):
        tag = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
        sp = os.path.join(fork_dir, f"succ_{tag}.npy")
        mp = os.path.join(fork_dir, f"succmask_{tag}.npy")
        if not os.path.exists(sp):
            continue
        S = np.load(sp)
        M = np.load(mp) if os.path.exists(mp) else np.zeros((len(S), 11), np.int8)
        n_ok = 0
        for line in open(rp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break                       # a torn last line on a killed shard
            keep = True
            for b in BRANCHES:
                s = r["branches"][b].get("succ")
                if s is not None and s >= len(S):
                    keep = False            # written after the last npy flush
            if not keep:
                continue
            for b in BRANCHES:
                s = r["branches"][b].get("succ")
                r["branches"][b]["succ"] = None if s is None else s + off
            rows.append(r)
            n_ok += 1
        succ.append(S)
        mask.append(M)
        off += len(S)
        print(f"  {tag}: {n_ok} forks, {len(S)} successor states", flush=True)
    return rows, np.concatenate(succ, 0), np.concatenate(mask, 0)


def build_table(rows, pooled, V0):
    """(branch table, pair table). A branch row survives only if it has a successor, a binary
    outcome and did NOT hit the stall cap."""
    b_fork, b_idx, b_y, b_battle, b_name = [], [], [], [], []
    for f, r in enumerate(rows):
        for name in BRANCHES:
            br = r["branches"][name]
            if br["succ"] is None or br["capped"] or br["outcome"] not in ("win", "loss"):
                continue
            b_fork.append(f)
            b_idx.append(br["succ"])
            b_y.append(1.0 if br["outcome"] == "win" else 0.0)
            b_battle.append(r["base"])
            b_name.append(name)
    b_fork = np.asarray(b_fork)
    b_idx = np.asarray(b_idx)
    b_y = np.asarray(b_y, np.float32)
    # pairs, within a fork
    pa, pb, py, pf = [], [], [], []
    by_fork = defaultdict(list)
    for k, f in enumerate(b_fork):
        by_fork[f].append(k)
    for f, ks in by_fork.items():
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                i, j = ks[x], ks[y]
                pa.append(i)
                pb.append(j)
                py.append(1.0 if b_y[i] > b_y[j] else (0.0 if b_y[i] < b_y[j] else 0.5))
                pf.append(f)
    return (dict(fork=b_fork, idx=b_idx, y=b_y, battle=np.asarray(b_battle), name=np.asarray(b_name),
                 pooled=pooled[b_idx], V0=V0[b_idx]),
            dict(a=np.asarray(pa), b=np.asarray(pb), y=np.asarray(py, np.float32),
                 fork=np.asarray(pf)))


# ---------------------------------------------------------------- the fit
def head_V(head, P):
    import torch
    with torch.no_grad():
        return torch.sigmoid(head(torch.as_tensor(P)).reshape(-1)).numpy()


# ---------------------------------------------------------------- the read
def read_head(V, tab, pairs, sub_pairs, label):
    """Every registered meter for one head, on one set of pairs."""
    a, b, y, pf = pairs["a"][sub_pairs], pairs["b"][sub_pairs], pairs["y"][sub_pairs], pairs["fork"][sub_pairs]
    nt = y != 0.5
    d = V[a] - V[b]
    agree = np.where(d > 0, y, np.where(d < 0, 1.0 - y, 0.5))[nt]
    pf_nt = pf[nt]

    def acc_for(fork_ids):
        m = np.isin(pf_nt, fork_ids)
        return float(agree[m].mean()) if m.sum() else None

    lo, hi = boot_ci(acc_for, pf_nt)
    sep_nt = float(np.abs(d[nt]).mean()) if nt.sum() else float("nan")
    sep_t = float(np.abs(d[~nt]).mean()) if (~nt).sum() else float("nan")
    rows = np.unique(np.concatenate([a, b]))
    return {"head": label, "n_pairs": int(len(y)), "n_nontied": int(nt.sum()),
            "pairwise_acc": float(agree.mean()) if nt.sum() else float("nan"),
            "pairwise_acc_ci": [lo, hi],
            "sep_nontied": sep_nt, "sep_tied": sep_t,
            "sep_ratio": float(sep_nt / sep_t) if sep_t else float("nan"),
            "ece": ece(V[rows], tab["y"][rows]),
            "brier": float(np.mean((V[rows] - tab["y"][rows]) ** 2)),
            "mean_V": float(V[rows].mean()), "base_rate": float(tab["y"][rows].mean())}


def paired_delta(Va, Vb, pairs, sub_pairs, seed=11):
    """Delta in pairwise accuracy between two heads, with its OWN bootstrap CI over forks."""
    a, b, y, pf = pairs["a"][sub_pairs], pairs["b"][sub_pairs], pairs["y"][sub_pairs], pairs["fork"][sub_pairs]
    nt = y != 0.5
    def ag(V):
        d = V[a] - V[b]
        return np.where(d > 0, y, np.where(d < 0, 1.0 - y, 0.5))[nt]
    ga, gb = ag(Va), ag(Vb)
    pf_nt = pf[nt]

    def dfn(fork_ids):
        m = np.isin(pf_nt, fork_ids)
        return float(ga[m].mean() - gb[m].mean()) if m.sum() else None

    lo, hi = boot_ci(dfn, pf_nt, seed=seed)
    return {"delta": float(ga.mean() - gb.mean()), "ci": [lo, hi],
            "detected": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- conditioning guard
def conditioning_frame(tree, model, n_per_opp=120, lo=4, hi=10, seed=3):
    """turns-4-10 recorded decisions, half from SENTINELS and half from BOTS, with their pooled
    vectors — the frame `cond.opp_class_auc.t4_10` is computed on."""
    import torch
    rng = np.random.default_rng(seed)
    plan = json.load(open(os.path.join(tree, "plan.json")))
    kinds = {it["key"]: it["kind"] for it in plan["items"]}
    P, cls = [], []
    for opp, kind in kinds.items():
        d = os.path.join(tree, opp)
        files = [f for f in sorted(os.listdir(d)) if f.endswith("_summary.json")]
        rng.shuffle(files)
        got = 0
        for f in files:
            if got >= n_per_opp:
                break
            base = os.path.join(d, f[: -len("_summary.json")])
            try:
                summ = json.load(open(base + "_summary.json"))
                npz = np.load(base + "_states.npz")
            except Exception:  # noqa: BLE001
                continue
            obs = np.asarray(npz["obs"], np.float32)
            mk = np.asarray(npz["action_mask"]).astype(np.float32)[:, :11]
            keep = [k for k, iv in enumerate(summ.get("invocations", []))
                    if lo <= int(iv.get("turn", 0)) <= hi and k < len(obs)]
            if not keep:
                continue
            k = int(rng.choice(keep))
            d_in = {"observation": torch.as_tensor(obs[k:k + 1]),
                    "action_mask": torch.as_tensor(mk[k:k + 1])}
            with torch.no_grad():
                model.policy.get_distribution(d_in)
                P.append(model.policy.features_extractor.stash.value_pooled.numpy()[0].copy())
            # label 1 = a SCRIPTED BOT. That orientation is not cosmetic: the published
            # `cond.opp_class_auc.t4_10` (0.723 / 0.730 for the online head, N-curve) is on this
            # sign, because a sentinel is the harder opponent and V is LOWER against it. Scored
            # the other way the same head reads 0.277 / 0.270 and looks like an anti-detector.
            cls.append(0 if kind == "sentinel" else 1)
            got += 1
    return np.asarray(P, np.float32), np.asarray(cls)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--coefs", default="0.1,0.3,1.0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--cond-n", type=int, default=120)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print("[refit] loading forks", flush=True)
    rows, succ, smask = load_forks(args.forks)
    print(f"[refit] {len(rows)} forks, {len(succ)} successor states", flush=True)

    from forks import load_model, score_batch  # noqa: E402  (same directory)
    model = load_model(args.snapshot)
    pooled = np.empty((len(succ), 128), np.float32)
    V0 = np.empty(len(succ), np.float64)
    for i in range(0, len(succ), 256):
        j = min(len(succ), i + 256)
        pooled[i:j], V0[i:j] = score_batch(model, succ[i:j], smask[i:j])
    print(f"[refit] forwarded successors ({time.time()-t0:.0f}s)", flush=True)

    tab, pairs = build_table(rows, pooled, V0)
    print(f"[refit] branch rows={len(tab['y'])} pairs={len(pairs['y'])} "
          f"non-tied={int((pairs['y']!=0.5).sum())}", flush=True)

    # --- split by BATTLE -------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    battles = np.unique(tab["battle"])
    rng.shuffle(battles)
    n_test = max(1, int(round(0.20 * len(battles))))
    n_val = max(1, int(round(0.15 * len(battles))))
    test_b = set(battles[:n_test])
    val_b = set(battles[n_test:n_test + n_val])
    is_test = np.array([b in test_b for b in tab["battle"]])
    is_val = np.array([b in val_b for b in tab["battle"]])
    is_tr = ~(is_test | is_val)
    tr_idx = np.flatnonzero(is_tr)
    va_idx = np.flatnonzero(is_val)
    te_idx = np.flatnonzero(is_test)
    in_set = lambda idx_set: np.array([  # noqa: E731
        (a in idx_set and b in idx_set) for a, b in zip(pairs["a"], pairs["b"])])
    p_tr = np.flatnonzero(in_set(set(tr_idx.tolist())))
    p_va = np.flatnonzero(in_set(set(va_idx.tolist())))
    p_te = np.flatnonzero(in_set(set(te_idx.tolist())))
    print(f"[refit] battles {len(battles)} -> train {len(tr_idx)} / val {len(va_idx)} / "
          f"test {len(te_idx)} branch rows; test pairs {len(p_te)} "
          f"(non-tied {int((pairs['y'][p_te]!=0.5).sum())})", flush=True)

    Ptr, Ytr = tab["pooled"], tab["y"]   # indexed by tr_idx/va_idx/te_idx below
    orig_sd = {k: v.detach().clone()
               for k, v in model.policy.features_extractor.win_head.state_dict().items()}

    fitlog = []
    # The fit may only ever see TRAIN pairs; validation is scored on the full table's val pairs.
    tr_pairs = {"a": pairs["a"][p_tr], "b": pairs["b"][p_tr], "y": pairs["y"][p_tr],
                "fork": pairs["fork"][p_tr]}

    def fit2(coef, seed):
        import torch as _t
        from agents.model.aux_value_heads import WinProbHead
        _t.manual_seed(seed)
        head = WinProbHead(); head.load_state_dict(orig_sd)
        opt = _t.optim.Adam(head.parameters(), lr=args.lr)
        Pt, Yt = _t.as_tensor(Ptr), _t.as_tensor(Ytr)
        r = np.random.default_rng(seed)
        bce = _t.nn.BCEWithLogitsLoss()
        ntr = np.flatnonzero(tr_pairs["y"] != 0.5)
        nva = p_va[pairs["y"][p_va] != 0.5]
        best, best_sd, since, step = float("inf"), None, 0, 0
        for step in range(1, args.steps + 1):
            sel = tr_idx[r.integers(0, len(tr_idx), size=min(args.batch, len(tr_idx)))]
            z = head(Pt[sel]).reshape(-1)
            loss = bce(z, Yt[sel])
            if coef > 0 and len(ntr):
                ps = ntr[r.integers(0, len(ntr), size=min(args.batch, len(ntr)))]
                za = head(Pt[tr_pairs["a"][ps]]).reshape(-1)
                zb = head(Pt[tr_pairs["b"][ps]]).reshape(-1)
                lab = _t.as_tensor(tr_pairs["y"][ps])
                d = _t.where(lab > 0.5, za - zb, zb - za)
                loss = loss + coef * _t.nn.functional.softplus(-d).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                with _t.no_grad():
                    v = float(bce(head(Pt[va_idx]).reshape(-1), Yt[va_idx]))
                    if coef > 0 and len(nva):
                        za = head(Pt[pairs["a"][nva]]).reshape(-1)
                        zb = head(Pt[pairs["b"][nva]]).reshape(-1)
                        lab = _t.as_tensor(pairs["y"][nva])
                        dd = _t.where(lab > 0.5, za - zb, zb - za)
                        v += coef * float(_t.nn.functional.softplus(-dd).mean())
                if v < best - 1e-6:
                    best, since = v, 0
                    best_sd = {k: t.detach().clone() for k, t in head.state_dict().items()}
                else:
                    since += 1
                    if since >= 40:
                        break
        if best_sd is not None:
            head.load_state_dict(best_sd)
        head.eval()
        fitlog.append({"rank_coef": coef, "steps_run": step, "best_val": best})
        return head

    heads = {"original": None}
    V = {"original": V0}
    ctrl = fit2(0.0, args.seed)
    heads["control_bce"] = ctrl
    V["control_bce"] = head_V(ctrl, tab["pooled"])

    # sweep the ranking coefficient; PICK ON VALIDATION
    best_coef, best_va, best_head = None, -1.0, None
    sweep = {}
    for c in [float(x) for x in args.coefs.split(",")]:
        h = fit2(c, args.seed)
        v = head_V(h, tab["pooled"])
        r_va = read_head(v, tab, pairs, p_va, f"rank@{c}")
        sweep[str(c)] = {"val": r_va}
        heads[f"rank_{c}"] = h
        V[f"rank_{c}"] = v
        print(f"  rank coef {c}: VAL pairwise acc {r_va['pairwise_acc']:.4f} "
              f"(n_nontied {r_va['n_nontied']})", flush=True)
        if r_va["pairwise_acc"] > best_va:
            best_va, best_coef, best_head = r_va["pairwise_acc"], c, h

    # --- the held-out read, taken ONCE -------------------------------------------------------
    report = {"n_forks": len(rows), "n_branch_rows": int(len(tab["y"])),
              "n_pairs": int(len(pairs["y"])),
              "n_nontied_pairs": int((pairs["y"] != 0.5).sum()),
              "split": {"battles": int(len(battles)), "train_rows": int(len(tr_idx)),
                        "val_rows": int(len(va_idx)), "test_rows": int(len(te_idx)),
                        "test_pairs": int(len(p_te)),
                        "test_nontied": int((pairs["y"][p_te] != 0.5).sum())},
              "best_coef": best_coef, "sweep_val": {k: v["val"] for k, v in sweep.items()},
              "fitlog": fitlog, "test": {}, "deltas": {}}
    for k in ["original", "control_bce"] + [f"rank_{c}" for c in
                                            [float(x) for x in args.coefs.split(",")]]:
        report["test"][k] = read_head(V[k], tab, pairs, p_te, k)
        print(f"  TEST {k}: pairwise {report['test'][k]['pairwise_acc']:.4f} "
              f"{report['test'][k]['pairwise_acc_ci']} sep_ratio "
              f"{report['test'][k]['sep_ratio']:.3f} ECE {report['test'][k]['ece']:.4f}", flush=True)
    bk = f"rank_{best_coef}"
    report["deltas"]["best_rank_minus_control"] = paired_delta(V[bk], V["control_bce"], pairs, p_te)
    report["deltas"]["best_rank_minus_original"] = paired_delta(V[bk], V["original"], pairs, p_te)
    report["deltas"]["control_minus_original"] = paired_delta(V["control_bce"], V["original"],
                                                              pairs, p_te)

    # --- the blind-spot rate (a property of the POLICY) ---------------------------------------
    bs_n = bs_hit = 0
    for r in rows:
        o = {b: r["branches"][b] for b in BRANCHES}
        if any(o[b]["capped"] or o[b]["outcome"] not in ("win", "loss") for b in BRANCHES):
            continue
        bs_n += 1
        if o["rand"]["outcome"] == "win" and o["top1"]["outcome"] == "loss" \
                and o["top2"]["outcome"] == "loss":
            bs_hit += 1
    report["blind_spot"] = {"n_complete_forks": bs_n, "rand_beats_both": bs_hit,
                            "rate": (bs_hit / bs_n) if bs_n else None}
    # branch-level outcome table
    report["branch_outcomes"] = {
        b: {"n": int((tab["name"] == b).sum()),
            "win_rate": float(tab["y"][tab["name"] == b].mean()) if (tab["name"] == b).sum() else None}
        for b in BRANCHES}

    # --- the conditioning guard ---------------------------------------------------------------
    print("[refit] conditioning guard frame...", flush=True)
    Pc, cls = conditioning_frame(args.tree, model, n_per_opp=args.cond_n)
    guard = {}
    for k in ["original", "control_bce", bk]:
        v = V0 if k == "original" else None
        vv = (head_V(heads[k], Pc) if k != "original"
              else head_V(model.policy.features_extractor.win_head, Pc))
        a = auc(vv, cls)
        guard[k] = {"opp_class_auc_t4_10": a, "distance_from_null": abs(a - 0.5),
                    "orientation": "label 1 = scripted bot (the published sign)",
                    "n": int(len(cls)), "n_bot": int(np.sum(cls == 1))}
        print(f"  guard {k}: opp_class AUC t4-10 = {guard[k]['opp_class_auc_t4_10']:.4f}",
              flush=True)
    report["conditioning_guard"] = guard

    # --- save ---------------------------------------------------------------------------------
    import torch as _t
    for k, h in heads.items():
        if h is not None:
            _t.save({"win_head": h.state_dict(), "rank_coef": k}, os.path.join(args.out, f"head_{k}.pt"))
    _t.save({"win_head": orig_sd}, os.path.join(args.out, "head_original.pt"))
    report["wall_s"] = time.time() - t0
    json.dump(report, open(os.path.join(args.out, "refit_report.json"), "w"), indent=1)
    print(f"[refit] DONE in {report['wall_s']:.0f}s -> {args.out}/refit_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
