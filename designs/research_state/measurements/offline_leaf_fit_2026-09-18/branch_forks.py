"""Build the ALPHAGO-STYLE BRANCHED dataset — one uniformly-random move at a uniformly-random
depth, then the frozen policy to the terminal.

    python3 branch_forks.py --tree <eval_traces/step_N> --snapshot <snapshot.zip> --out <dir> \
        --opponents sentinel_0 --shard i/W [--forks-per-battle 2]

WHAT THIS IS, and how it differs from `paired_refit_discrimination_2026-09-14/forks.py`.

That builder selects CONTESTED decisions (the bottom 40 % of the policy's top-2 masked-logit gap)
and branches into `top1` / `top2` / `rand`. It is the right instrument for MEASURING sibling
discrimination, and it is used VERBATIM BY PATH for this read's held-out set.

This builder is its TRAINING counterpart, and it is AlphaGo's value-network recipe rather than a
contested-state selector:

  * the decision is drawn UNIFORMLY over the battle's eligible `move_selection` decisions
    (turn in [--min-turn, --max-turn], >= --min-legal legal actions) — NOT on the policy's gap,
    NOT on |V - 0.5|, NOT on the outcome. The selector reads no head and no logit.
  * ONE uniformly-random legal move is played at that decision, drawn from `legal \\ {top1}` with a
    DECISION-KEYED rng. Excluding `top1` is deliberate and is the one deviation from "uniformly
    random legal": it guarantees the two siblings are DIFFERENT actions, which is what the paired
    fit (b) and the pairwise ranking term need. Without it a draw that lands on `top1` would
    produce a degenerate pair.
  * the SIBLING is the frozen policy's own `top1` from the same pre-fork state, under the SAME
    dice (common random numbers). Two branches, two labels, one decorrelated position per game
    (or per `--forks-per-battle` positions per game).

EVERYTHING ELSE IS forks.py's, IMPORTED FROM IT rather than copied: `load_model`, `score_batch`,
`SuccessorCapture`, `run_branch`, `battle_files` and `_sha`. In particular the CRN discipline is
untouched — `replay_counterfactual(seed=<the record's own resolved seed>, post_t_seed=None)` hands
the sim ONE dice stream for the whole line, our side plays GREEDY, and the opponent is a SENTINEL
reloaded from the exact snapshot the manifest names, played GREEDY. `--determinism-check-every`
re-runs one branch and REFUSES a disagreement, so the pairing is asserted rather than assumed.

A branch that hit the 250-turn stall-forfeit cap is a DRAW AT CAP decided by SEAT, and is EXCLUDED
rather than scored 0.5. A branch that ENDS at the divergence turn has no successor (``succ: null``).

Nothing is written under ``models/``; CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np

# The 2026-09-14 builder is the instrument; everything shared with it is IMPORTED, never copied,
# so this dataset cannot drift from the one the reference levels are measured on.
_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "paired_refit_discrimination_2026-09-14")
sys.path.insert(0, os.path.abspath(_REC))

BRANCHES = ("top1", "rand")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opponents", default="sentinel_0")
    ap.add_argument("--shard", default="0/1", help="i/W — disjoint battle slices")
    ap.add_argument("--battles", type=int, default=0)
    ap.add_argument("--max-forks", type=int, default=0)
    ap.add_argument("--min-legal", type=int, default=2,
                    help="2, not 3: a uniform draw needs only ONE alternative to top1")
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument("--max-turn", type=int, default=60,
                    help="a fork much later than this is, on a recorded draw, already inside a "
                         "stall loop; the cap rate is reported either way")
    ap.add_argument("--exclude-draws", type=int, default=1)
    ap.add_argument("--forks-per-battle", type=int, default=2)
    ap.add_argument("--impl", default="rust", choices=["node", "rust"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--determinism-check-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from agents.observation.state_encoder import load_mappings
    from agents.training.obs_materializer import materialize_from_record
    from utils.bridge.reconstruction import ReconstructionRecord
    from forks import _sha, battle_files, load_model, run_branch   # noqa: E402

    os.makedirs(args.out, exist_ok=True)
    i, w = (int(x) for x in args.shard.split("/"))
    tag = f"s{i}"
    rows_path = os.path.join(args.out, f"forks_{tag}.jsonl")
    succ_path = os.path.join(args.out, f"succ_{tag}.npy")
    t0 = time.time()

    plan = json.load(open(os.path.join(args.tree, "plan.json")))
    sent = {it["key"]: it.get("path") for it in plan["items"] if it.get("kind") == "sentinel"}
    opponents = [o for o in args.opponents.split(",") if o]
    for o in opponents:
        if o not in sent:
            raise SystemExit(f"REFUSED: {o} is not a SENTINEL in this tree's plan. Only a "
                             f"sentinel is reloadable EXACTLY and was recorded GREEDY.")
    man = json.load(open(os.path.join(args.tree, "eval_manifest.json")))
    if not man.get("generated_by", {}).get("eval_sentinel_greedy"):
        raise SystemExit("REFUSED: this tree's sentinels were NOT greedy — reloading them greedy "
                         "would not be the recorded regime.")

    play_model = load_model(args.snapshot)
    opp_models = {o: load_model(sent[o]) for o in opponents}
    mappings = load_mappings()

    files = battle_files(args.tree, opponents)
    files = files[i::w]
    if args.battles:
        files = files[: args.battles]
    print(f"[branch {tag}] {len(files)} battles, opponents={opponents}, impl={args.impl}",
          flush=True)

    # ---- pass 1: the UNIFORM depth draw -----------------------------------------------------
    picked = []
    n_elig_total = n_battles_used = 0
    for n_b, (opp, base) in enumerate(files):
        try:
            summ = json.load(open(base + "_summary.json"))
            npz = np.load(base + "_states.npz")
            obs = np.asarray(npz["obs"], dtype=np.float32)
            masks = np.asarray(npz["action_mask"]).astype(np.int8)
            if masks.shape[1] > 11:
                masks = masks[:, :11]
        except Exception as e:  # noqa: BLE001
            print(f"  skip {base}: {type(e).__name__} {e}", flush=True)
            continue
        if args.exclude_draws and ((summ.get("meta") or {}).get("result") or "").lower() == "draw":
            continue
        invs = summ.get("invocations", [])
        elig = [k for k, iv in enumerate(invs)
                if iv.get("phase") == "move_selection"
                and args.min_turn <= int(iv.get("turn", 0)) <= args.max_turn
                and k < len(obs) and int(masks[k].sum()) >= args.min_legal]
        if not elig:
            continue
        n_elig_total += len(elig)
        n_battles_used += 1
        # UNIFORM over the eligible decisions, battle-keyed so the draw is reproducible and
        # independent of shard geometry. `U(1, remaining)` in the registration, realised as a
        # uniform draw over this battle's eligible decision INDICES.
        r = random.Random(_sha(base, args.seed))
        take = r.sample(elig, min(args.forks_per_battle, len(elig)))
        rec_lg = np.asarray(npz["logits"], dtype=np.float64)
        rec_v = np.asarray(npz["win_probs"], dtype=np.float64).reshape(-1)
        for k in sorted(take):
            row = rec_lg[k].copy()
            row[masks[k] == 0] = -np.inf
            order = np.argsort(-row)
            a1, a2 = int(order[0]), int(order[1])
            picked.append({"opp": opp, "base": base, "inv": int(k),
                           "turn": int(invs[k]["turn"]),
                           "gap": float(row[a1] - row[a2]),
                           "legal": np.flatnonzero(masks[k]).astype(int).tolist(),
                           "top1": a1, "V": float(rec_v[k]),
                           "absV": abs(float(rec_v[k]) - 0.5),
                           "n_elig": len(elig), "depth_rank": int(elig.index(k)),
                           "result": ((summ.get("meta") or {}).get("result") or "").lower()})
        if (n_b + 1) % 200 == 0:
            print(f"  pass1 {n_b+1}/{len(files)} battles, {len(picked)} picks "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if not picked:
        raise SystemExit("REFUSED: no eligible decisions")
    # The BATTLE order is shuffled so a shard stopped early by the clock returns a random subset
    # of battles rather than an alphabetical one (this tree's filenames sort draw_ < loss_ < win_,
    # so an alphabetical prefix would be a sample of LOSSES — the defect the 2026-09-14 builder's
    # own comment records catching in its smoke).
    by_battle = {}
    for c in picked:
        by_battle.setdefault(c["base"], []).append(c)
    rng = random.Random(args.seed + i)
    order = list(by_battle)
    rng.shuffle(order)
    picked = [c for b in order for c in sorted(by_battle[b], key=lambda c: c["inv"])]
    if args.max_forks:
        picked = picked[: args.max_forks]
    print(f"[branch {tag}] forking {len(picked)} decisions over {len(by_battle)} battles "
          f"(mean eligible/battle {n_elig_total/max(1,n_battles_used):.1f})", flush=True)

    # ---- pass 2: the branches ---------------------------------------------------------------
    succ_vecs, stats = [], {"forks": 0, "errors": 0, "no_successor": 0, "capped": 0,
                            "det_checks": 0, "det_fail": 0, "no_alt": 0}
    fout = open(rows_path, "a")
    recs = {}
    for n, c in enumerate(picked):
        try:
            rec = recs.get(c["base"])
            if rec is None:
                rec = ReconstructionRecord.load(c["base"] + "_reconstruction.json")
                recs = {c["base"]: rec}
            npz = np.load(c["base"] + "_states.npz")
            actions = np.asarray(npz["actions"], dtype=int)
            trace = materialize_from_record(rec, actions=actions, mappings=mappings,
                                            map_actions_at=c["inv"],
                                            stop_after_decision=c["inv"], impl=args.impl)
            legal_in_trace = sorted(trace.action_choices or {})
            others = [a for a in legal_in_trace if a != c["top1"]]
            if not others:
                stats["no_alt"] += 1
                continue
            r_act = others[_sha(c["base"], c["inv"], args.seed, "uniform") % len(others)]
            acts = {"top1": c["top1"], "rand": int(r_act)}
            br = {}
            for name in BRANCHES:
                o, capped, turns, sobs, smask = run_branch(
                    rec, trace, c["turn"], acts[name], play_model=play_model,
                    opp_model=opp_models[c["opp"]], opp_ckpt=sent[c["opp"]], mappings=mappings,
                    impl=args.impl, tag=f"{i}_{n}_{name}")
                br[name] = {"action": int(acts[name]), "outcome": o, "capped": capped,
                            "turns": turns, "succ": None}
                if capped:
                    stats["capped"] += 1
                if sobs is not None:
                    br[name]["succ"] = len(succ_vecs)
                    succ_vecs.append((sobs, smask))
                else:
                    stats["no_successor"] += 1
            if args.determinism_check_every and (n % args.determinism_check_every == 0):
                o2, _, _, _, _ = run_branch(rec, trace, c["turn"], acts["top1"],
                                            play_model=play_model,
                                            opp_model=opp_models[c["opp"]],
                                            opp_ckpt=sent[c["opp"]], mappings=mappings,
                                            impl=args.impl, tag=f"{i}_{n}_det")
                stats["det_checks"] += 1
                if o2 != br["top1"]["outcome"]:
                    stats["det_fail"] += 1
                    print(f"  🚨 DETERMINISM FAIL at {c['base']}#{c['inv']}: "
                          f"{br['top1']['outcome']} vs {o2}", flush=True)
            row = {"opp": c["opp"], "base": os.path.basename(c["base"]), "inv": c["inv"],
                   "turn": c["turn"], "gap": c["gap"], "V_rec": c["V"], "absV": c["absV"],
                   "n_legal": len(c["legal"]), "n_elig": c["n_elig"],
                   "depth_rank": c["depth_rank"],
                   "depth_frac": (c["depth_rank"] / max(1, c["n_elig"] - 1)) if c["n_elig"] > 1
                                 else 0.0,
                   "recorded_result": c["result"], "branches": br, "shard": i}
            fout.write(json.dumps(row) + "\n")
            stats["forks"] += 1
            if stats["forks"] % 20 == 0:
                fout.flush()
                np.save(succ_path, np.array([v[0] for v in succ_vecs], dtype=np.float32))
                np.save(succ_path.replace("succ_", "succmask_"),
                        np.array([v[1] for v in succ_vecs], dtype=np.int8))
                el = time.time() - t0
                print(f"  [{tag}] {stats['forks']} forks ({el:.0f}s, "
                      f"{el/max(1,stats['forks']):.1f}s/fork) {stats}", flush=True)
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  ERR {c['base']}#{c['inv']}: {type(e).__name__} {e}", flush=True)
                traceback.print_exc()
    fout.close()
    np.save(succ_path, np.array([v[0] for v in succ_vecs], dtype=np.float32))
    np.save(succ_path.replace("succ_", "succmask_"),
            np.array([v[1] for v in succ_vecs], dtype=np.int8))
    meta = {"shard": args.shard, "tree": args.tree, "snapshot": args.snapshot,
            "opponents": opponents, "sentinel_paths": {o: sent[o] for o in opponents},
            "selector": "UNIFORM over eligible move_selection decisions; rand ~ U(legal \\ top1)",
            "n_picked": len(picked), "n_battles_used": n_battles_used,
            "mean_eligible_per_battle": n_elig_total / max(1, n_battles_used),
            "stats": stats, "wall_s": time.time() - t0, "impl": args.impl, "seed": args.seed,
            "min_legal": args.min_legal, "min_turn": args.min_turn, "max_turn": args.max_turn,
            "forks_per_battle": args.forks_per_battle}
    json.dump(meta, open(os.path.join(args.out, f"meta_{tag}.json"), "w"), indent=1)
    print(f"[branch {tag}] DONE {json.dumps(meta['stats'])} in {meta['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
