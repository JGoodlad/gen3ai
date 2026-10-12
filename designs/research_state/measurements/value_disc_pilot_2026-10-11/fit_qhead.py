"""PHASE 3, step 4 (Amendment 7): fit the OFFLINE per-action Q head on the FROZEN end2 checkpoint's root features
(``qfeatures.py``) against ORACLE playout labels on the TRAINING states (``run_qlabels.py``), then act by its argmax on
the registered EVALUATION states (the 600 Phase-1 states, disjoint battles) with NO rollout and NO world at decision
time, scored against Phase 1's TRUE-world truth (seeds 16–39).

THE RECIPE (registered in Amendment 7; nothing here is tuned on an evaluation state):
- label y(s, a) = the mean of the K_Q = 2 oracle playout outcomes on the win scale (1 win, ½ tie / truncated, 0 loss);
- per action a, input = a FAMILY projection of the action's own entity (move k: the request-ordered move token ⊕ its
  op cells; switch j: our team token j ⊕ its switch cells; Struggle: a learned vector) + a projection of the critic's
  ``value_pooled`` + a projection of the pointer head's per-action hidden layer + a projection of (logit_a, log π_a,
  V); ReLU → Linear(256, 256) → ReLU → Linear(256, 1) = q_a. Inputs standardized on the FIT split;
- loss = the per-state CENTRED squared error over the legal actions, mean_a ((q_a − q̄) − (y_a − ȳ))², averaged over
  states (the head learns the action's ADVANTAGE: what an argmax reads);
- AdamW lr 1e-3, weight decay 1e-4, 256 states per batch, ≤ 60 epochs, early stopping (patience 8) on the VALIDATION
  split's loss (15 % of the training BATTLES by ``sha256("vdp_qsplit_v1:" + battle)``), best epoch restored;
- an ENSEMBLE of 5 seeds (torch seeds 0–4); the score is the members' mean centred q; â = its argmax over the legal
  actions (a tie within 1e-6 breaks toward the policy's order, as every Phase-1 scorer).

    python fit_qhead.py end2 [--fracs 0.25,0.5,1.0] [--out-json result_qhead.json] [--out-md result_qhead.md]
    (CPU, torch; under taskset / nice / mem_cap like every pilot job)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S1  # noqa: E402
import score_det as SD  # noqa: E402

ARCH = S1.ARCH
HERE = Path(__file__).resolve().parent
ACT = 11
SW, MV = slice(0, 6), slice(6, 10)
SEEDS = (0, 1, 2, 3, 4)
HID = 256
SPLIT_SALT = "vdp_qsplit_v1:"
VAL_SHARE = 0.15
DECW = {"+": 1.0, "-": 0.0, "0": 0.5}


def load_feats(cp, bank):
    z = np.load(ARCH / cp / f"qfeat_{bank}.npz")
    return {k: z[k] for k in z.files}


def load_labels(cp):
    lab = {}
    for p in sorted((ARCH / cp / "qrows").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                lab[r["id"]] = r["out"]
    return lab


def label_matrix(F, lab):
    """[N, 11] y (NaN off the legal set), and the kept row index (states with a label whose actions = the mask)."""
    keep, Y = [], []
    for i, did in enumerate(F["ids"]):
        out = lab.get(str(did))
        if out is None:
            continue
        legal = set(np.flatnonzero(F["mask"][i]).tolist())
        if {int(a) for a in out} != legal:
            raise SystemExit(f"{did}: labelled actions {sorted(out)} != the mask's legal set {sorted(legal)}")
        y = np.full(ACT, np.nan)
        for a, s in out.items():
            y[int(a)] = np.mean([DECW[ch] for ch in s])
        keep.append(i)
        Y.append(y)
    return np.array(keep, dtype=int), np.array(Y)


def subset(F, idx):
    return {k: v[idx] for k, v in F.items()}


def standardizers(F):
    """Per-feature mean / sd on the FIT split (entity blocks over every action slot of their family)."""
    st = {}
    for k in ("pooled", "mtok", "mcell", "team", "scell", "hid"):
        x = F[k].reshape(-1, F[k].shape[-1]).astype(np.float64)
        st[k] = (x.mean(0), x.std(0) + 1e-6)
    lg = np.where(F["mask"].astype(bool), F["logits"], np.nan)
    lp = lg - np.log(np.nansum(np.exp(lg - np.nanmax(lg, 1, keepdims=True)), 1, keepdims=True)) - np.nanmax(lg, 1, keepdims=True)
    sc = np.stack([lg, lp, np.repeat(F["v"][:, None], ACT, 1)], -1).reshape(-1, 3)
    sc = sc[~np.isnan(sc).any(1)]
    st["scal"] = (sc.mean(0), sc.std(0) + 1e-6)
    return st


def tensors(F, st):
    import torch

    def nz(k):
        m, s = st[k]
        return torch.tensor(((F[k] - m) / s).astype(np.float32))
    mask = F["mask"].astype(bool)
    lg = np.where(mask, F["logits"], -np.inf)
    mx = lg.max(1, keepdims=True)
    lp = lg - mx - np.log(np.exp(lg - mx).sum(1, keepdims=True))
    sc = np.stack([np.where(mask, F["logits"], 0.0), np.where(mask, lp, 0.0), np.repeat(F["v"][:, None], ACT, 1)], -1)
    m, s = st["scal"]
    return {"pooled": nz("pooled"), "mtok": nz("mtok"), "mcell": nz("mcell"), "team": nz("team"),
            "scell": nz("scell"), "hid": nz("hid"), "scal": torch.tensor(((sc - m) / s).astype(np.float32)),
            "mask": torch.tensor(mask)}


def build(F, seed):
    import torch

    torch.manual_seed(seed)
    D, Cm, Cs, H = F["mtok"].shape[-1], F["mcell"].shape[-1], F["scell"].shape[-1], F["hid"].shape[-1]

    class QHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mv = torch.nn.Linear(D + Cm, HID)
            self.sw = torch.nn.Linear(D + Cs, HID)
            self.st = torch.nn.Parameter(torch.zeros(HID))
            self.ctx = torch.nn.Linear(F["pooled"].shape[-1], HID)
            self.hid = torch.nn.Linear(H, HID)
            self.sc = torch.nn.Linear(3, HID)
            self.l2 = torch.nn.Linear(HID, HID)
            self.out = torch.nn.Linear(HID, 1)

        def forward(self, b):
            ent = torch.cat([self.sw(torch.cat([b["team"], b["scell"]], -1)),
                             self.mv(torch.cat([b["mtok"], b["mcell"]], -1)),
                             self.st.expand(b["team"].shape[0], 1, HID)], 1)           # [B,11,HID]
            z = torch.relu(ent + self.ctx(b["pooled"])[:, None, :] + self.hid(b["hid"]) + self.sc(b["scal"]))
            return self.out(torch.relu(self.l2(z))).squeeze(-1)                      # [B,11]
    return QHead()


def centred(q, mask):
    import torch

    m = mask.float()
    return (q - (q * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m


def loss_fn(q, y, mask):
    import torch

    m = mask.float()
    d = centred(q, mask) - centred(y, mask)
    return ((d ** 2).sum(1) / m.sum(1)).mean()


def fit(T, Y, fit_idx, val_idx, seed, max_epochs=60, patience=8, bs=256):
    import torch

    model = build({k: v.numpy() for k, v in T.items() if k != "mask"}, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Yt = torch.tensor(np.nan_to_num(Y, nan=0.0).astype(np.float32))
    take = lambda idx: ({k: v[idx] for k, v in T.items()}, Yt[idx])  # noqa: E731
    vb, vy = take(torch.tensor(val_idx))
    best, best_state, bad, hist = float("inf"), None, 0, []
    g = torch.Generator().manual_seed(1000 + seed)
    for ep in range(max_epochs):
        model.train()
        perm = torch.tensor(fit_idx)[torch.randperm(len(fit_idx), generator=g)]
        for i in range(0, len(perm), bs):
            b, y = take(perm[i:i + bs])
            opt.zero_grad()
            loss_fn(model(b), y, b["mask"]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(vb), vy, vb["mask"]))
        hist.append(round(vl, 6))
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val": round(best, 6), "epochs": len(hist), "val_curve": hist}


def ensemble_scores(models, T):
    import torch

    with torch.no_grad():
        return np.mean([centred(m(T), T["mask"]).numpy() for m in models], 0)


def split(battles):
    ub = sorted(set(battles), key=lambda b: hashlib.sha256((SPLIT_SALT + b).encode()).hexdigest())
    val_b = set(ub[: int(round(VAL_SHARE * len(ub)))])
    return val_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cp")
    ap.add_argument("--fracs", default="0.25,0.5,1.0")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fit-only", action="store_true", help="fit and report the validation R² only (no evaluation)")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    cp = a.cp
    Ftr_all = load_feats(cp, "qbank")
    lab = load_labels(cp)
    keep, Y = label_matrix(Ftr_all, lab)
    Ftr = subset(Ftr_all, keep)
    battles = [str(b) for b in Ftr["battle"]]
    val_b = split(battles)
    is_val = np.array([b in val_b for b in battles])
    st = standardizers(subset(Ftr, np.flatnonzero(~is_val)))
    T = tensors(Ftr, st)
    # the fit-split battles in hash order: the label-size curve takes their first share
    fit_b = sorted({b for b in battles if b not in val_b},
                   key=lambda b: hashlib.sha256(("vdp_qfrac_v1:" + b).encode()).hexdigest())
    val_idx = np.flatnonzero(is_val)
    # ---- the evaluation states (registered sample; never seen by any fit)
    Fev = load_feats(cp, "bank")
    if set(map(str, Fev["battle"])) & set(battles):
        raise SystemExit("an evaluation battle is in the training set")
    Tev = tensors(Fev, st)
    ev_ids = [str(x) for x in Fev["ids"]]
    res = {"cp": cp, "train_states": int(len(keep)), "train_battles": len(set(battles)),
           "val_battles": len(val_b), "unlabelled_train_states": int(len(Ftr_all["ids"]) - len(keep)),
           "label_k": 2, "fits": {}, "imp": {}, "contrasts": {}, "ranking": {}, "val_r2": {}}
    picks = {}
    fracs = [float(x) for x in a.fracs.split(",")]
    for frac in fracs:
        fb = set(fit_b[: int(round(frac * len(fit_b)))])
        fit_idx = np.flatnonzero(np.array([b in fb for b in battles]))
        models, info = [], []
        for s in SEEDS:
            m, inf = fit(T, Y, fit_idx, val_idx, s)
            models.append(m)
            info.append(inf)
        # validation R² of the centred labels (the ensemble vs predicting 0 advantage)
        qv = ensemble_scores(models, {k: v[val_idx] for k, v in T.items()})
        mv = Ftr["mask"][val_idx].astype(bool)
        yv = np.nan_to_num(Y[val_idx])
        yc = (yv - (yv * mv).sum(1, keepdims=True) / mv.sum(1, keepdims=True)) * mv
        sse = (((qv - yc) * mv) ** 2).sum()
        sst = ((yc * mv) ** 2).sum()
        res["val_r2"][str(frac)] = round(float(1 - sse / sst), 5)
        res["fits"][str(frac)] = {"fit_states": int(len(fit_idx)), "members": info}
        if a.fit_only:      # a training-data diagnostic: reads NO evaluation outcome
            print(json.dumps({"frac": frac, "val_r2": res["val_r2"][str(frac)],
                              "members": [(m["best_val"], m["epochs"]) for m in info]}), flush=True)
            continue
        q = ensemble_scores(models, Tev)
        picks[str(frac)] = q
    if a.fit_only:
        return
    st_doc, r1, _, _ = S1.load(cp, True)
    rd = SD.det_rows(cp)
    E = st_doc["states"]
    # ---- score: IMP against the TRUE-world truth, beside the critic scorers on the same states
    rows_imp, ids_ok = {}, []
    rank = {}
    for i, did in enumerate(ev_ids):
        r = r1.get(did)
        if not (r and r.get("ok")):
            continue
        S = S1.state_arrays(r, E[did])
        acts = [int(x) for x in S["acts"]]
        T_, t1, order = S["truth"], S["top1"], S["order"]
        d1, _ = S1.decisions(S)
        x = {"ORACLE CRITIC": T_[d1["ARGMAX|CRITIC"]] - T_[t1], "ORACLE LEAF-K8": T_[d1["ARGMAX|LEAF-K8"]] - T_[t1]}
        for frac, q in picks.items():
            x[f"QHEAD@{frac}"] = T_[S1.pick(q[i][acts], order)] - T_[t1]
        x["A-HEAD (ride-along)"] = T_[S1.pick(Fev["adv"][i][acts].astype(np.float64), order)] - T_[t1]
        if rd.get(did, {}).get("ok"):
            _, Sd = SD.det_arrays(r, rd[did], E[did])
            n16 = S1.N_LEAF
            S1.N_LEAF = Sd["W"].shape[1]
            try:
                dd, _ = S1.decisions(Sd)
            finally:
                S1.N_LEAF = n16
            x["HONEST CRITIC"] = T_[dd["ARGMAX|CRITIC"]] - T_[t1]
            x["HONEST TRUNC-4-K4"] = T_[dd["ARGMAX|TRUNC-4-K4"]] - T_[t1]
            x["HONEST LEAF-K8"] = T_[dd["ARGMAX|LEAF-K8"]] - T_[t1]
            hc = Sd["crit"]
        else:
            hc = None
        rows_imp[did] = x
        ids_ok.append(did)
        # within-state ranking quality vs the truth: Spearman ρ over the legal actions (descriptive)
        def rho(sc):
            from scipy.stats import spearmanr
            v = spearmanr(sc, T_).correlation
            return None if v is None or np.isnan(v) else float(v)
        rank.setdefault("QHEAD@1.0", []).append(rho(picks[str(fracs[-1])][i][acts]))
        rank.setdefault("POLICY logits", []).append(rho(S["lg"]))
        rank.setdefault("ORACLE CRITIC", []).append(rho(S["crit"]))
        if hc is not None:
            rank.setdefault("HONEST CRITIC", []).append(rho(hc))
    for k, v in rank.items():
        vv = [x for x in v if x is not None]
        res["ranking"][k] = {"mean_spearman": round(float(np.mean(vv)), 4), "n": len(vv)}
    e = [E[i] for i in ids_ok]
    bt = S1.Boot([x["battle"] for x in e], [1.0] * len(e))
    keys = sorted({k for x in rows_imp.values() for k in x})
    for k in keys:
        sel = np.array([k in rows_imp[i] for i in ids_ok])
        arr = np.array([rows_imp[i].get(k, 0.0) for i in ids_ok])
        res["imp"][k] = bt.stat(arr, sel=sel)
    common = np.array(["HONEST CRITIC" in rows_imp[i] for i in ids_ok])
    full = str(fracs[-1])
    get = lambda k: np.array([rows_imp[i].get(k, 0.0) for i in ids_ok])  # noqa: E731
    res["n_states"], res["n_common_with_honest"] = len(ids_ok), int(common.sum())
    res["contrasts"]["R-QHEAD: QHEAD - policy (all states)"] = bt.stat(get(f"QHEAD@{full}"))
    res["contrasts"]["R-QHEAD-CRITIC: QHEAD - HONEST CRITIC (common)"] = bt.stat(
        get(f"QHEAD@{full}") - get("HONEST CRITIC"), sel=common)
    res["contrasts"]["QHEAD - HONEST TRUNC-4-K4 (common)"] = bt.stat(get(f"QHEAD@{full}") - get("HONEST TRUNC-4-K4"), sel=common)
    res["contrasts"]["QHEAD - ORACLE CRITIC"] = bt.stat(get(f"QHEAD@{full}") - get("ORACLE CRITIC"))
    res["contrasts"]["QHEAD - A-HEAD"] = bt.stat(get(f"QHEAD@{full}") - get("A-HEAD (ride-along)"))
    switch = {k: round(float(np.mean([rows_imp[i][k] != 0.0 for i in ids_ok if k in rows_imp[i]])), 4) for k in keys}
    res["share_nonzero_imp"] = switch
    c = res["contrasts"]
    lo, hi = c["R-QHEAD-CRITIC: QHEAD - HONEST CRITIC (common)"]["ci"]
    res["verdicts"] = {
        "R-QHEAD": S1.verdict3(c["R-QHEAD: QHEAD - policy (all states)"]["ci"]),
        "R-QHEAD-CRITIC": ("AHEAD" if lo > S1.EDGE else "BEHIND" if hi < -S1.EDGE
                           else "EQUIVALENT" if (lo > -0.01 and hi < 0.01) else "NOT DETECTED")}
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(md(res) + "\n")
    print(json.dumps({k: res[k] for k in ("verdicts", "contrasts", "val_r2", "n_states")}, indent=1))


def md(r):
    f = lambda d: f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"  # noqa: E731
    L = [f"# Value-discrimination pilot, PHASE 3 (offline per-action Q head): {r['cp']}", "",
         f"Training: {r['train_states']} states / {r['train_battles']} battles (oracle labels, K = {r['label_k']}; "
         f"validation {r['val_battles']} battles). Evaluation: {r['n_states']} registered states "
         f"({r['n_common_with_honest']} with Phase 2's honest rows), TRUE-world truth seeds 16–39.", "",
         "Verdicts: " + ", ".join(f"{k} **{v}**" for k, v in r["verdicts"].items()), "",
         "| scorer (argmax, no rollout at decision time unless named) | IMP [95 %] | n | share of states moved off top1 |",
         "|---|---|---|---|"]
    for k, d in r["imp"].items():
        L.append(f"| {k} | {f(d)} | {d['n']} | {r['share_nonzero_imp'].get(k, 0):.3f} |")
    L += ["", "| contrast | Δ [95 %] |", "|---|---|"]
    for k, d in r["contrasts"].items():
        L.append(f"| {k} | {f(d)} |")
    L += ["", "| fit share | fit states | validation R² (centred labels) | member epochs |", "|---|---|---|---|"]
    for k, d in r["fits"].items():
        L.append(f"| {k} | {d['fit_states']} | {r['val_r2'][k]:+.4f} | {[m['epochs'] for m in d['members']]} |")
    L += ["", "| scorer | mean within-state Spearman ρ vs truth | n |", "|---|---|---|"]
    for k, d in r["ranking"].items():
        L.append(f"| {k} | {d['mean_spearman']:+.4f} | {d['n']} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
