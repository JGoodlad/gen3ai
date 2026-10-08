"""FOLLOW-UP T4 (README §0.7): does fixed_mass's opponent-intent head (the flat pointer) inflate P(opponent
stays) when our policy stays in? Extraction + read in one; run AT THE PIN via ``run_pin.sh``:

    run_pin.sh alpha_t4.py --out <dir> --ckpt <zip>=<label> [...]     # the 8 fixed_mass finals

Writes ``alpha_rows/<label>.npz`` (resumable) and ``alpha_t4.json`` / ``alpha_t4.md``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

TIE_EPS = 1e-6
BOUND_EPS = 1e-9
M4 = 0.02
BOOT_SEED = 20261007
SWITCH_BASE = 1000
NO_EVENT = -1


def forward(zip_path, br, threads=4, batch=256):
    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.flat_intent import other_species_col, slot_col
    from main.belief_roles.forward import arm_of, load_strict
    from main.policy_spectrum.reader import inference_globals

    out = {k: np.full(br.n, np.nan) for k in ("p_sw", "margin")}
    out["top1"] = np.full(br.n, -1, dtype=np.int64)
    out["sw_live"] = np.zeros(br.n, dtype=bool)
    with inference_globals(threads), torch.no_grad():
        model = load_strict(zip_path)
        if arm_of(model) != "fixed_mass":
            raise ValueError(f"{zip_path}: T4 is fixed_mass only")
        fe = model.policy.features_extractor
        for i in range(0, br.n, batch):
            sl = slice(i, i + batch)
            rows = torch.tensor(br.rows[sl]); mk = torch.tensor(br.masks[sl])
            ob = {"observation": rows, "action_mask": mk.float()}
            ob.update(zero_extra_obs(fe, batch=rows.shape[0]))
            logits = model.policy.get_distribution(ob).distribution.logits.double()
            pr = torch.softmax(logits.masked_fill(~mk, -math.inf), -1)
            t2 = pr.topk(2, -1)
            out["top1"][sl] = t2.indices[:, 0].numpy()
            out["margin"][sl] = (t2.values[:, 0] - t2.values[:, 1]).numpy()
            st = fe.stash
            fi = st.flat_intent
            live = fi.live.bool()
            P = torch.softmax(st.flat_intent_logits.double().masked_fill(~live, -math.inf), -1)
            K = int(fi.k)
            cols = list(range(slot_col(K), slot_col(K) + 6)) + [other_species_col(K)]
            out["p_sw"][sl] = P[:, cols].sum(-1).numpy()
            out["sw_live"][sl] = live[:, cols].any(-1).numpy()
        del model
    return out


def tci(x):
    from scipy import stats
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(x.mean()), "lo": float(x.mean() - h), "hi": float(x.mean() + h),
            "per_seed": [float(v) for v in x]}


def main():
    from main.belief_roles.__main__ import default_bank
    from main.belief_roles.bank_rows import bank_rows_of
    from main.policy_spectrum.bank import load_bank

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "alpha_rows").mkdir(parents=True, exist_ok=True)
    bank = load_bank(default_bank())
    br = bank_rows_of(bank, workers=2)
    print(f"[bank] {br.n} rows; obs as recorded {br.gate['obs_as_recorded']}", flush=True)
    labs = []
    for spec in a.ckpt:
        z, lab = spec.rsplit("=", 1)
        labs.append(lab)
        f = out / "alpha_rows" / f"{lab}.npz"
        if not f.exists():
            np.savez_compressed(f, **forward(Path(z), br))
            print(f"[done] {lab}", flush=True)
    free = np.array([d["kind"] == "free" for d in bank.decisions])
    base = free & br.masks[:, 6:11].any(1) & br.masks[:, 0:6].any(1) & (br.event != NO_EVENT)
    y = (br.event >= SWITCH_BASE).astype(float)
    bi = br.battle_index
    nb = int(bi.max()) + 1
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float) for _ in range(a.boot)])
    R = {lab: dict(np.load(out / "alpha_rows" / f"{lab}.npz")) for lab in labs}

    def ok_of(r):
        return base & r["sw_live"] & (r["margin"] >= TIE_EPS)

    def gap(v, ok, g1, w=None):
        m1, m0 = ok & g1, ok & ~g1
        if w is None:
            return v[m1].mean() - v[m0].mean()
        f = lambda m: (w @ np.bincount(bi, np.where(m, v, 0), nb)) / (w @ np.bincount(bi, m.astype(float), nb))  # noqa: E731
        return f(m1) - f(m0)

    ok_all = np.zeros(br.n, bool)
    for r in R.values():
        ok_all |= ok_of(r)
    base_rate = float(y[ok_all].mean())
    attack = np.zeros((len(labs), br.n), bool)
    for j, lab in enumerate(labs):
        t1 = R[lab]["top1"]
        attack[j] = np.array([t1[g] >= 6 and t1[g] < 10 and d["cats"].get(str(int(t1[g]))) == "attack"
                              for g, d in enumerate(bank.decisions)])
    res = {"base_rate": base_rate, "labels": labs}
    cols = {"arm": [], "prior": [], "delta": [], "attack_vs_switch_arm": [], "cross_policy_arm": [],
            "p_sw_stay": [], "y_stay": [], "p_sw_switch": [], "y_switch": [], "n_stay": [], "n_switch": []}
    bo = np.zeros(a.boot)
    for j, lab in enumerate(labs):
        r = R[lab]
        ok = ok_of(r)
        stay = r["top1"] >= 6
        e = r["p_sw"] - y
        ep = base_rate - y
        da, dp = gap(e, ok, stay), gap(ep, ok, stay)
        cols["arm"].append(da); cols["prior"].append(dp); cols["delta"].append(da - dp)
        bo += gap(e, ok, stay, W) / len(labs)
        sw = ok & ~stay
        att = ok & attack[j]
        cols["attack_vs_switch_arm"].append(e[att].mean() - e[sw].mean())
        cross = []
        for t in labs:
            if t == lab:
                continue
            ok2 = ok & ok_of(R[t])
            cross.append(gap(e, ok2, R[t]["top1"] >= 6))
        cols["cross_policy_arm"].append(float(np.mean(cross)))
        for k, m in (("stay", ok & stay), ("switch", sw)):
            cols[f"p_sw_{k}"].append(r["p_sw"][m].mean()); cols[f"y_{k}"].append(y[m].mean())
            cols[f"n_{k}"].append(int(m.sum()))
    for k, v in cols.items():
        res[k] = tci(v) if not k.startswith("n_") else v
    lo, hi = np.percentile(bo, [2.5, 97.5])
    res["arm"]["boot_lo"], res["arm"]["boot_hi"] = float(lo), float(hi)
    A = res["arm"]
    mag_ok = abs(A["mean"]) - M4 > BOUND_EPS
    if A["mean"] < 0 and A["hi"] < -BOUND_EPS and hi < -BOUND_EPS and mag_ok:
        v = "BENDS"
    elif A["mean"] > 0 and A["lo"] > BOUND_EPS and lo > BOUND_EPS and mag_ok:
        v = "ANTI-self-serving"
    else:
        v = "NOT DETECTED"
    res["verdict"] = v
    (out / "alpha_t4.json").write_text(json.dumps(res, indent=1))

    def c(x):
        return f"{x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
    md = ["# Follow-up T4: fixed_mass intent head, P(opp switch) − truth, STAY − SWITCH (generated by `alpha_t4.py`)", "",
          f"Base rate (opp switches, eligible rows): {base_rate:.4f}. Rows per seed STAY {cols['n_stay']} / SWITCH {cols['n_switch']}.", "",
          "| quantity | across-seed mean [t95] |", "|---|---|",
          f"| **Δ4_arm** (verdict; boot95 [{lo:+.4f}, {hi:+.4f}]) | **{c(res['arm'])}** |",
          f"| Δ4_prior (constant base rate = −Δy) | {c(res['prior'])} |",
          f"| arm − prior (= Δ of P_sw) | {c(res['delta'])} |",
          f"| ATTACK − SWITCH, arm (descriptive) | {c(res['attack_vs_switch_arm'])} |",
          f"| split by another fm seed's policy (mean of 7) | {c(res['cross_policy_arm'])} |",
          f"| P_sw / truth when we STAY | {res['p_sw_stay']['mean']:.4f} / {res['y_stay']['mean']:.4f} |",
          f"| P_sw / truth when we SWITCH | {res['p_sw_switch']['mean']:.4f} / {res['y_switch']['mean']:.4f} |",
          "", f"**Verdict: {v}** (rule §0.7: Δ4_arm < 0, t95 and boot95 below 0, |Δ| > 0.02)"]
    (out / "alpha_t4.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
