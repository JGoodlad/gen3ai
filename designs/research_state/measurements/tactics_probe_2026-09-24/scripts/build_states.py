"""Build + PROVE every registered state, then read the three models at it.

For each state (seed from find_seeds' registry): play the scripted prefix with the continuation players
(G0, the model under the trainee seat), capture at turn T: the trainee's obs + action mask (exactly what the
live policy would see), the legal action -> sim token map, and a snapshot from BOTH players' own battles.
The snapshot must pass the state's predicates again (a state that fails is REFUSED, not measured).
Then, for G0 / B / C: policy probabilities (T = 1, masked), critic V, win-prob head, and for family L the
opp-active move belief on the lure / standard move, plus the model's own Smogon prior buffer for them.

Also asserts: for every lure, the PRE-lure and PRE-std obs are byte-identical (the two worlds are
indistinguishable before the reveal), and the three models load with no arch drift.

    python build_states.py --seeds <seeds.json> --out <archive dir> --manifest <repo out/states.json>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib  # noqa: E402
import states as ST  # noqa: E402
from snap import check, snapshot  # noqa: E402


def record_for(spec, seed, regime):
    cmds = spec.commands()
    if regime == "premise":
        cmds = cmds + [(tlib.OTHER[spec.our_side], spec.premise)]
    return tlib.make_record(tlib.pack(spec.p1), tlib.pack(spec.p2), seed, cmds, spec.our_side)


def capture_token(spec):
    """A surely-legal substitute for the capture line (the pin's install_scripted_prefix takes a STRING):
    the first move of our active mon at T."""
    from poke_env.data.normalize import to_id_str
    sets = spec.p1 if spec.our_side == "p1" else spec.p2
    want = spec.expect.get(f"{spec.our_side}_active")
    for st in sets:
        mon = tlib.parse_set(st)
        if want is None or to_id_str(mon.species or mon.nickname) == want:
            return "move " + to_id_str(mon.moves[0])
    raise ValueError(spec.sid)


def capture(spec, seed, g0):
    rec = record_for(spec, seed, "live")
    trainee = tlib.make_rl(g0, prefix="BSt", policy_seed=1)
    opp = tlib.make_rl(g0, prefix="BSo", policy_seed=2)
    box = {}

    def hook(battle, obs_dict, choice):
        if int(battle.turn) != spec.T or "obs" in box or obs_dict is None:
            return
        box["obs"] = np.asarray(obs_dict["observation"], dtype=np.float32).copy()
        box["mask"] = np.asarray(obs_dict["action_mask"], dtype=np.float32).copy()
        toks = {}
        for idx in np.flatnonzero(box["mask"]):
            toks[int(idx)] = trainee.action_to_order(int(idx), battle).message[len("/choose "):]
        box["tokens"] = toks
        ob = opp.battles.get(battle.battle_tag)
        b1, b2 = (battle, ob) if spec.our_side == "p1" else (ob, battle)
        box["snap"] = snapshot(b1, b2)
        box["capture_choice"] = choice

    res = tlib.run_line(rec, trainee=trainee, opponent=opp, T=spec.T, substitute=capture_token(spec), hook=hook,
                        post_t_seed=tlib.mint_seed(f"build:{spec.sid}"), keep_sink=False)
    box["line"] = {k: res[k] for k in ("outcome", "turns", "capped", "our_exhausted", "opp_exhausted")}
    return box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True, help="archive dir (obs .npy)")
    ap.add_argument("--manifest", required=True, help="repo json (no obs)")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    reg = json.load(open(a.seeds))
    os.makedirs(os.path.join(a.out, "obs"), exist_ok=True)
    from agents.observation.state_encoder import load_mappings
    mv = load_mappings()["moves"]
    models = {t: tlib.load_model(t) for t in tlib.MODELS}
    man = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {"states": {}}
    man["models"] = {t: {"run": tlib.MODELS[t], "ckpt": tlib.CKPT, "sha256": tlib.sha256_file(tlib.model_path(t))}
                     for t in tlib.MODELS}
    man["pin"] = "6eb9c776"
    only = set(x for x in a.only.split(",") if x)
    for spec in ST.all_states():
        if only and spec.sid not in only:
            continue
        if spec.sid not in reg:
            print(f"!! {spec.sid}: no registered seed; skipped", flush=True)
            continue
        seed = reg[spec.sid]["seed"]
        t0 = time.time()
        box = capture(spec, seed, models["G0"])
        bad = check(box["snap"], spec.expect) if "snap" in box else ["no capture at T"]
        if "obs" not in box:
            bad.append("no obs at T")
        entry = {"sid": spec.sid, "family": spec.family, "our_side": spec.our_side, "T": spec.T,
                 "seed": seed, "seed_k": reg[spec.sid]["k"], "premise": spec.premise, "factors": spec.factors,
                 "commands": spec.commands(), "p1_team": tlib.pack(spec.p1), "p2_team": tlib.pack(spec.p2),
                 "snapshot": box.get("snap"), "refused": bad, "tokens": box.get("tokens"),
                 "capture_line": box.get("line"), "wall_s": round(time.time() - t0, 1)}
        if not bad:
            np.save(os.path.join(a.out, "obs", f"{spec.sid}.npy"), box["obs"])
            np.save(os.path.join(a.out, "obs", f"{spec.sid}.mask.npy"), box["mask"])
            nums = None
            if spec.family == "L":
                nums = [mv[spec.factors["lure_move"]]["num"], mv[spec.factors["std_move"]]["num"],
                        mv["hiddenpower"]["num"]]
                entry["belief_nums"] = {"lure": nums[0], "std": nums[1], "hiddenpower": nums[2]}
            entry["readout"] = {t: tlib.readout(m, box["obs"], box["mask"], nums) for t, m in models.items()}
            if spec.family == "L":
                fe = models["G0"].policy.features_extractor
                sp = load_mappings()["species"][spec.factors["species"]]["num"]
                import torch
                pr = torch.sigmoid(fe.move_belief.move_prior_logits[sp]).detach().numpy()
                entry["model_prior_buffer"] = {"lure": float(pr[nums[0]]), "std": float(pr[nums[1]])}
                pri = json.load(open(os.path.join(os.getcwd(), "data/pokemon/gen3_move_priors.json")))
                row = pri[spec.factors["species"]]
                row = row.get("moves", row) if isinstance(row, dict) else row
                entry["smogon_prior"] = {"lure": float(row.get(spec.factors["lure_move"], 0.0)),
                                         "std": float(row.get(spec.factors["std_move"], 0.0))}
        man["states"][spec.sid] = entry
        # model readouts are deliberately NOT printed: the build runs before PREDICTION.md is committed.
        print(f"{'OK' if not bad else 'REFUSED'} {spec.sid} ({entry['wall_s']}s) bad={bad} tokens={box.get('tokens')}",
              flush=True)
        json.dump(man, open(a.manifest, "w"), indent=1)
    # the lure worlds must be indistinguishable before the reveal
    ident = {}
    for lid in sorted({s.factors.get("lure") for s in ST.all_states() if s.family == "L"}):
        pa = os.path.join(a.out, "obs", f"{lid}_pre_lure.npy")
        pb = os.path.join(a.out, "obs", f"{lid}_pre_std.npy")
        if os.path.exists(pa) and os.path.exists(pb):
            x, y = np.load(pa), np.load(pb)
            ident[lid] = {"identical": bool(np.array_equal(x, y)), "n_diff": int((x != y).sum())}
    man["lure_pre_obs_identity"] = ident
    print("lure pre obs identity:", ident, flush=True)
    json.dump(man, open(a.manifest, "w"), indent=1)


if __name__ == "__main__":
    main()
