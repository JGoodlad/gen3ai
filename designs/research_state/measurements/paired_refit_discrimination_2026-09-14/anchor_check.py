"""THE ANCHOR — does a fork's PREFIX actually reproduce the battle it claims to fork?

    python3 anchor_check.py --tree <eval_traces/step_N> --opponents sentinel_2,... [-n 40]

A `divergence_turn=None` replay is the correctness oracle of the whole fork design: every choice on
both sides is scripted from the record and NOTHING is played by a policy, so it must reproduce the
RECORDED winner exactly, and no side may run out of script (`script_exhausted` empty). If that
fails, the three branches of a fork are not three continuations of one position and every paired
number in this directory is void — which is why this is an ANCHOR run before the read, not a
diagnostic after it (the rule `cf_audit` states: label trust before map trust).

This is the registered prefix check's (a) half. Its (b) half — that the three branches SHARE the
dice — is structural (one `>start` seed, no `post_t_seed` reseed) and is asserted empirically by
`forks.py --determinism-check-every`, which re-runs one branch and refuses on a disagreement.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--opponents", default="sentinel_2,sentinel_1,sentinel_0")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from poke_env.ps_client import LocalhostServerConfiguration
    from agents.observation.state_encoder import load_mappings
    from main.prober.replay import build_opponent, build_trainee
    from utils.bridge.counterfactual import replay_counterfactual as _run_one
    from utils.bridge.reconstruction import ReconstructionRecord
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from forks import load_model

    plan = json.load(open(os.path.join(args.tree, "plan.json")))
    sent = {it["key"]: it.get("path") for it in plan["items"] if it.get("kind") == "sentinel"}
    opponents = [o for o in args.opponents.split(",") if o]
    model = load_model(args.snapshot)
    opp_models = {o: load_model(sent[o]) for o in opponents}
    mappings = load_mappings()

    cands = []
    for opp in opponents:
        d = os.path.join(args.tree, opp)
        for f in sorted(os.listdir(d)):
            if f.endswith("_summary.json") and not f.startswith("draw_"):
                cands.append((opp, os.path.join(d, f[: -len("_summary.json")])))
    random.Random(args.seed).shuffle(cands)

    ok = bad = exh = 0
    fails = []
    for opp, base in cands[: args.n]:
        rec = ReconstructionRecord.load(base + "_reconstruction.json")
        summ = json.load(open(base + "_summary.json"))
        want = ((summ.get("meta") or {}).get("result") or "").lower()
        trainee = build_trainee(model, rec, mappings, LocalhostServerConfiguration, tag="anc")
        opponent, _ = build_opponent("", rec, model, mappings, LocalhostServerConfiguration,
                                     opponent_ckpt=sent[opp], opp_model=opp_models[opp],
                                     opponent_source="ckpt", opponent_stochastic=False, tag="anc")
        res = _run_one(rec, trainee=trainee, opponent=opponent, divergence_turn=None,
                       substitute_choice=None, seed=rec.start_options().get("seed"),
                       impl=args.impl)
        got = res["outcome"]
        if res.get("script_exhausted"):
            exh += 1
        if got == want:
            ok += 1
        else:
            bad += 1
            fails.append({"base": os.path.basename(base), "want": want, "got": got,
                          "exhausted": res.get("script_exhausted")})
    out = {"n": ok + bad, "reproduced": ok, "mismatched": bad, "script_exhausted": exh,
           "rate": ok / max(1, ok + bad), "failures": fails[:10],
           "verdict": "ANCHOR PASSES" if (bad == 0 and exh == 0) else "🚨 ANCHOR FAILS — forks void"}
    print(json.dumps(out, indent=1))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
