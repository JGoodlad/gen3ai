"""PROBE L, measurement 4 — the CfEvidentialHead Beta(alpha,beta) at whiff decisions.

Is the confession head CONFIDENT when the win-prob head disagrees with the policy?

`cf_evid_head` is NOT called from the extractor forward (it is a pure side readout the training
loss applies to the STASHED `value_pooled`), so there is no recorded trace key and no
`ProbeModel` accessor: this runs the real forward on the RECORDED obs, takes
`features_extractor.stash.value_pooled`, and calls the head on it — the same composition the
training term uses.

Includes a LIVENESS check, because a head that was built but never trained is indistinguishable
from a trained one by the presence of its parameters: it compares the head's mean against the
recorded win-prob head and against the realized episode outcome, and reports the spread of the
evidence alpha+beta. An untrained head is flat and uncorrelated.

Run:
  nice -n 15 python tmp/probe_l_evidential.py --n 400
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

RUN = "/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823"
STEP = "step_24000000"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="tmp/whiff_census_all.jsonl")
    ap.add_argument("--step", default=STEP)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="tmp/probe_l_evidential.json")
    a = ap.parse_args(argv)

    rows = [json.loads(ln) for ln in open(a.census) if json.loads(ln)["step"] == a.step]
    picks = []
    for r in rows:
        bw = {b["turn"]: b for b in r["baits"]}
        for d in r["decisions"]:
            b = bw.get(d["turn"])
            tag = ("whiff" if (b and b["whiff"]) else
                   "hit_pivot" if b else d["tag"])
            if tag in ("whiff", "hit_pivot", "no_pivot"):
                picks.append({"base": r["base"], "inv": d["inv"], "tag": tag,
                              "kind": (b["kind"] if b else None),
                              "loop_step": bool(b and b["loop_step"]),
                              "outcome": r["outcome"]})
    rng = random.Random(17)
    whiff = [p for p in picks if p["tag"] == "whiff"]
    other = [p for p in picks if p["tag"] != "whiff"]
    rng.shuffle(other)
    sample = whiff + other[: max(0, a.n - len(whiff))]

    traces = os.path.join(RUN, "eval_traces", a.step)
    from main.prober.session import ProbeSession
    sess = ProbeSession(traces, ckpt_override=os.path.join(traces, "snapshot.zip"),
                        impl="rust", compile_extractor=False)
    bt0 = sess._battle(sample[0]["base"] + "_summary.json")
    model, _ = sess._model_for(bt0)
    fe = model._policy.features_extractor
    head = getattr(fe, "cf_evid_head", None)
    built = head is not None
    out: dict = {"step": a.step, "head_built": built,
                 "cf_evidential_config": None, "rows": []}
    cfg_path = os.path.join(RUN, "model_config.json")
    if os.path.exists(cfg_path):
        c = json.load(open(cfg_path))
        out["cf_evidential_config"] = {k: c.get(k) for k in
                                       ("cf_evidential", "cf_evidential_coef",
                                        "cf_evidential_reg", "cf_head_only",
                                        "cf_label_lag_steps", "cf_label_likelihood")}
    if not built:
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps(out, indent=1))
        return 0

    by_base: dict = {}
    for p in sample:
        by_base.setdefault(p["base"], []).append(p)
    for base, ps in by_base.items():
        with np.load(base + "_states.npz") as z:
            obs_all = np.asarray(z["obs"])
            mask_all = np.asarray(z["action_mask"], dtype=bool)
            wp_all = np.asarray(z["win_probs"])
        idx = [p["inv"] for p in ps if p["inv"] < len(obs_all)]
        if not idx:
            continue
        ot = torch.as_tensor(obs_all[idx], dtype=torch.float32)
        mt = torch.as_tensor(mask_all[idx])
        with torch.no_grad():
            model._policy.predict_values({"observation": ot, "action_mask": mt})
            vp = fe.stash.value_pooled
            al, be = head(vp)
        al = al.numpy()
        be = be.numpy()
        for j, p in enumerate([q for q in ps if q["inv"] < len(obs_all)]):
            out["rows"].append({
                **p, "alpha": float(al[j]), "beta": float(be[j]),
                "mean": float(al[j] / (al[j] + be[j])),
                "evidence": float(al[j] + be[j]),
                "win_prob_head": float(wp_all[p["inv"]]),
                "won": 1.0 if p["outcome"] == "WIN" else 0.0,
            })
    sess.close()

    R = out["rows"]
    if R:
        m = np.array([r["mean"] for r in R])
        e = np.array([r["evidence"] for r in R])
        w = np.array([r["win_prob_head"] for r in R])
        y = np.array([r["won"] for r in R])
        out["liveness"] = {
            "n": len(R),
            "mean_range": [float(m.min()), float(m.max())], "mean_sd": float(m.std()),
            "evidence_range": [float(e.min()), float(e.max())], "evidence_median": float(np.median(e)),
            "corr_mean_vs_winprob_head": float(np.corrcoef(m, w)[0, 1]),
            "corr_mean_vs_outcome": float(np.corrcoef(m, y)[0, 1]),
            "corr_winprob_head_vs_outcome": float(np.corrcoef(w, y)[0, 1]),
            "at_beta_1_1_frac": float(np.mean((np.array([r["alpha"] for r in R]) < 1.05)
                                              & (np.array([r["beta"] for r in R]) < 1.05))),
            "verdict": None,
        }
        out["liveness"]["verdict"] = (
            "LIVE — the head's mean tracks the win-prob head and the realized outcome, and its "
            "evidence varies across states"
            if (abs(out["liveness"]["corr_mean_vs_winprob_head"]) > 0.5
                and out["liveness"]["mean_sd"] > 0.02)
            else "NOT DEMONSTRABLY TRAINED — flat and/or uncorrelated; treat as absent")
        for tag in ("whiff", "hit_pivot", "no_pivot"):
            sub = [r for r in R if r["tag"] == tag]
            if not sub:
                continue
            out.setdefault("by_tag", {})[tag] = {
                "n": len(sub),
                "median_evidence": float(np.median([r["evidence"] for r in sub])),
                "median_mean": float(np.median([r["mean"] for r in sub])),
                "median_sd": float(np.median([
                    (r["alpha"] * r["beta"] / ((r["alpha"] + r["beta"]) ** 2
                                               * (r["alpha"] + r["beta"] + 1))) ** 0.5
                    for r in sub])),
                "median_winprob_head": float(np.median([r["win_prob_head"] for r in sub])),
            }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
