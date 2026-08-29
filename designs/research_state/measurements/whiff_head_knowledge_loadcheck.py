"""PROBE L — load-path ACID TEST + lookahead cost probe.

Two jobs, both prerequisites for the real sweep:

1. **Acid test**: the step dir's `snapshot.zip` is the EXACT model that wrote those traces, so the
   reloaded policy must reproduce the recorded action probabilities at a recorded decision. If it
   does not, every model read downstream is about a different network. Reported as max |Δp| over
   the legal actions of N sampled decisions.
2. **Cost probe**: wall-clock for one `lookahead` (CRN only) and for one with `n_seeds=R`, which is
   what sizes the whole sweep.

Run:
  nice -n 15 python tmp/probe_l_costcheck.py
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("POKESIM_SIM_BRIDGE_BIN",
                      "/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge")
os.environ.setdefault("POKESIM_SEARCH_DRIVER_BIN",
                      "/home/goodlad/dev/gen3ai/src/rust_sim/target/release/search_driver")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

RUN = "/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="tmp/whiff_census_s24.jsonl")
    ap.add_argument("--step", default="step_24000000")
    ap.add_argument("--n-acid", type=int, default=6)
    ap.add_argument("--n-cost", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args(argv)

    rows = [json.loads(ln) for ln in open(a.census)]
    picks = []
    for r in rows:
        for b in r["baits"]:
            if b["whiff"] and b["kind"] == "immune" and b["inv"] is not None:
                picks.append((r["base"], b))
    print(f"population: {len(picks)} immune whiffs")

    traces = os.path.join(RUN, "eval_traces", a.step)
    ckpt = os.path.join(traces, "snapshot.zip")
    from main.prober.session import ProbeSession
    t0 = time.time()
    sess = ProbeSession(traces, ckpt_override=ckpt, impl="rust", compile_extractor=False)
    print(f"session up in {time.time()-t0:.1f}s")

    # ---- 1. ACID TEST: reloaded policy vs the recorded action probabilities -------------------
    acid = []
    for base, b in picks[: a.n_acid]:
        bid = base + "_summary.json"
        bt = sess._battle(bid)
        model, ck = sess._model_for(bt)
        summary = sess._summary(bt)
        with np.load(base + "_states.npz") as z:
            obs = np.asarray(z["obs"][b["inv"]])
            mask = np.asarray(z["action_mask"][b["inv"]], dtype=bool)
            rec_logits = np.asarray(z["logits"][b["inv"]])
            rec_v = float(z["values"][b["inv"]])
            rec_wp = float(z["win_probs"][b["inv"]])
        probs, logits = model.action_dist(obs, mask)
        d_log = float(np.max(np.abs(logits - rec_logits)))
        d_v = abs(model.value(obs, mask) - rec_v)
        _wp = model.win_prob_at(obs, mask)
        d_wp = abs(_wp - rec_wp) if _wp is not None else None
        inv = summary["invocations"][b["inv"]]
        rec = {k: float(str(v.get("prob", "0")).rstrip("%")) / 100.0
               for k, v in (inv.get("actions") or {}).items() if isinstance(v, dict)}
        # the summary keys actions by LABEL; map label -> index via the same helper lookahead uses
        from main.prober.falsifier import _label_of
        deltas = []
        for i in range(len(mask)):
            if not mask[i]:
                continue
            lab = _label_of(inv, i)
            if lab in rec:
                deltas.append(abs(float(probs[i]) - rec[lab]))
        acid.append({"base": os.path.basename(base), "inv": b["inv"], "ckpt_tier": ck.tier, "ckpt": os.path.basename(ck.path or "?"),
                     "n_legal": int(mask.sum()), "n_matched": len(deltas),
                     "max_abs_dp": round(max(deltas), 5) if deltas else None,
                     "max_abs_dlogit": round(d_log, 6),
                     "d_value": round(d_v, 6),
                     "d_win_prob": (round(d_wp, 6) if d_wp is not None else None)})
        print("ACID", acid[-1])

    # ---- 2. COST PROBE -----------------------------------------------------------------------
    cost = []
    for base, b in picks[: a.n_cost]:
        bid = base + "_summary.json"
        t0 = time.time()
        la = sess.lookahead(bid, inv=b["inv"])
        t_crn = time.time() - t0
        n_cand = len(la["candidates"])
        wp = [c["win_prob_crn"] for c in la["candidates"]]
        t0 = time.time()
        la2 = sess.lookahead(bid, inv=b["inv"], n_seeds=a.seeds)
        t_seed = time.time() - t0
        cost.append({"inv": b["inv"], "n_cand": n_cand,
                     "t_crn_s": round(t_crn, 2), f"t_{a.seeds}seeds_s": round(t_seed, 2),
                     "win_probs": wp,
                     "chosen": la["chosen"]["label"], "census_move": b["move"],
                     "best_alt": la["best_alternative"], "best_dv": la["best_delta_v"],
                     "n_evaluated_seeded": [c["n_evaluated"] for c in la2["candidates"]]})
        print("COST", json.dumps(cost[-1]))
    sess.close()
    print(json.dumps({"acid": acid, "cost": cost}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
