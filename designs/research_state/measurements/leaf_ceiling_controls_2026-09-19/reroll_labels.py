"""CONTROL 2 — ROLLOUT-AVERAGED LABELS on the banked forks: re-roll every branch K times with FRESH
post-divergence dice and use the MEAN outcome as the label.

    python3 reroll_labels.py --forks <banked dir> --snapshot <zip> --out <dir> --shard i/W \
        --k 8 --sample 1000 --seed 20260919

WHAT IT IS FOR. Every published level on this instrument is scored against a SINGLE-rollout outcome:
one realization of the dice from the successor. If the `top1|top2` column's 0.545 is a LABEL-NOISE
floor rather than a representation floor, then averaging the label must raise every scorer's
accuracy on it; if it is the GAME — the policy's two best moves being genuinely near-tied — then
averaging raises nothing and the single-vs-averaged ORDER agreement is high.

THE DICE. ``replay_counterfactual(post_t_seed=s)`` swaps the sim PRNG at the START of the divergence
turn, so the PREFIX keeps the recorded dice (the pre-fork state is IDENTICAL across re-rolls — that
is what makes the K rollouts re-rolls of the SAME state) and everything from the fork on is
resampled. Both sides stay GREEDY, so the sim PRNG is the only randomness in the line.

CRN IS WITHIN A RE-ROLL INDEX. For re-roll k, the three branches of a fork share ONE seed
``s_k`` — so the k-th realization of `top1` and the k-th of `top2` differ in exactly the action,
exactly as the banked single-rollout pair does. Across k the seeds differ. The K seeds are derived
from the fork's own identity (``battle:inv:seed``), so a re-run of this script reproduces them, and
``fresh_seeds(4)`` is a strict prefix of ``fresh_seeds(8)`` — a smaller K is a SUB-SAMPLE of the
same dice rather than a different experiment.

THE K SEEDS ARE INDEPENDENT OF THE BANKED LABEL, deliberately. The banked outcome was taken on the
record's own realized stream (``post_t_seed=None``); none of the K re-rolls reproduces it. So the
single-vs-averaged order-disagreement rate this produces is an honest estimate of how much the
single label wanders, rather than a rate deflated by the two labels sharing a realization.

A CAPPED line (the 250-turn stall forfeit) is scored **0.5**, not win/loss: at the cap both sides
forfeit and the winner is an artifact of forfeit ordering. That is `cf_producer.rollout_outcome_score`
and it is imported, not re-implemented. The per-branch capped COUNT is recorded so a reader can see
how much of a label is cap.

NESTED FORK-OUTER / K-INNER, so an early stop yields FEWER FORKS AT FULL K rather than all forks at
a ragged K — a ragged K would weight forks unequally in the mean without saying so. Forks are drawn
in a seeded shuffle, so a prefix is a random subset.

CPU only. Nothing is written under ``models/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import traceback

import numpy as np

BRANCHES = ("top1", "top2", "rand")
_MEAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MEAS, "paired_refit_discrimination_2026-09-14"))


def run_branch(record, trace, turn, action, *, play_model, opp_model, opp_ckpt, mappings, impl, tag,
               post_t_seed):
    """One re-rolled branch, played to a terminal on FRESH post-divergence dice."""
    from poke_env.ps_client import LocalhostServerConfiguration
    from main.prober.replay import build_opponent, build_trainee
    from utils.bridge.counterfactual import replay_counterfactual as _run_one

    choice_map = trace.action_choices or {}
    if int(action) not in choice_map:
        return None
    trainee = build_trainee(play_model, record, mappings, LocalhostServerConfiguration, tag=tag)
    opponent, _src = build_opponent(
        "", record, play_model, mappings, LocalhostServerConfiguration,
        opponent_ckpt=opp_ckpt, opp_model=opp_model, opponent_source="ckpt",
        opponent_stochastic=False, tag=tag)
    return _run_one(record, trainee=trainee, opponent=opponent, divergence_turn=int(turn),
                    substitute_choice=choice_map[int(action)],
                    seed=record.start_options().get("seed"), post_t_seed=post_t_seed, impl=impl)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/W — which forks_s*.jsonl slice")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sample", type=int, default=1000, help="forks over the WHOLE set; this shard "
                                                             "takes its proportional share")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--impl", default="rust", choices=["node", "rust"])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from agents.observation.state_encoder import load_mappings
    from agents.training.cf_producer import rollout_outcome_score
    from agents.training.obs_materializer import materialize_from_record
    from main.prober.falsifier import fresh_seeds
    from utils.bridge.reconstruction import ReconstructionRecord
    from forks import load_model                       # by path, unmodified

    os.makedirs(args.out, exist_ok=True)
    i, w = (int(x) for x in args.shard.split("/"))
    t0 = time.time()

    paths = sorted(glob.glob(os.path.join(args.forks, "forks_s*.jsonl")))
    if not (0 <= i < len(paths)) or w != len(paths):
        raise SystemExit(f"REFUSED: --shard {args.shard} does not name one of {len(paths)} banked "
                         f"shard files")
    rp = paths[i]
    tag = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
    S = np.load(os.path.join(args.forks, f"succ_{tag}.npy"))
    rows = []
    for line in open(rp):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        if any((r["branches"][b].get("succ") is not None
                and r["branches"][b]["succ"] >= len(S)) for b in BRANCHES):
            continue
        rows.append(r)

    # ELIGIBLE: all three branches have a successor. Selection does NOT read any outcome — reading
    # it would make the averaged-label read partly a measurement of the selector.
    elig = [n for n, r in enumerate(rows)
            if all(r["branches"][b].get("succ") is not None for b in BRANCHES)]
    rng = random.Random(args.seed + i)
    rng.shuffle(elig)
    take = max(1, round(args.sample / w))
    elig = elig[:take]
    print(f"[reroll {tag}] {len(rows)} banked forks, {len(elig)} sampled (target {take}), "
          f"K={args.k}", flush=True)

    bmeta = json.load(open(os.path.join(args.forks, f"meta_{tag}.json")))
    tree = bmeta["tree"]
    plan = json.load(open(os.path.join(tree, "plan.json")))
    sent = {it["key"]: it.get("path") for it in plan["items"] if it.get("kind") == "sentinel"}

    play_model = load_model(args.snapshot)
    opp_models = {}
    mappings = load_mappings()

    out_path = os.path.join(args.out, f"reroll_{tag}.jsonl")
    done = set()
    if os.path.exists(out_path):                      # resumable: skip forks already complete
        for line in open(out_path):
            try:
                done.add(json.loads(line)["fork_line"])
            except Exception:                          # noqa: BLE001 — a torn last line
                break
    fout = open(out_path, "a")
    stats = {"forks": 0, "rollouts": 0, "errors": 0, "capped": 0, "skipped_done": 0}
    for n in elig:
        if n in done:
            stats["skipped_done"] += 1
            continue
        r = rows[n]
        try:
            base = os.path.join(tree, r["opp"], r["base"])
            rec = ReconstructionRecord.load(base + "_reconstruction.json")
            npz = np.load(base + "_states.npz")
            actions = np.asarray(npz["actions"], dtype=int)
            trace = materialize_from_record(rec, actions=actions, mappings=mappings,
                                            map_actions_at=r["inv"],
                                            stop_after_decision=r["inv"], impl=args.impl)
            if r["opp"] not in opp_models:
                opp_models[r["opp"]] = load_model(sent[r["opp"]])
            # ONE seed per re-roll index, SHARED by the three branches — the CRN pairing.
            seeds = fresh_seeds(args.k, salt=f"{r['base']}:{r['inv']}:{args.seed}")
            per = {b: {"scores": [], "outcomes": [], "capped": 0, "turns": []} for b in BRANCHES}
            ok = True
            for k, s in enumerate(seeds):
                for b in BRANCHES:
                    res = run_branch(rec, trace, r["turn"], r["branches"][b]["action"],
                                     play_model=play_model, opp_model=opp_models[r["opp"]],
                                     opp_ckpt=sent[r["opp"]], mappings=mappings, impl=args.impl,
                                     tag=f"r{i}_{n}_{k}_{b}", post_t_seed=s)
                    if res is None:
                        ok = False
                        break
                    stats["rollouts"] += 1
                    per[b]["scores"].append(float(rollout_outcome_score(res)))
                    per[b]["outcomes"].append(res.get("outcome"))
                    per[b]["turns"].append(int(res.get("turns") or 0))
                    if res.get("capped"):
                        per[b]["capped"] += 1
                        stats["capped"] += 1
                if not ok:
                    break
            if not ok:
                stats["errors"] += 1
                continue
            fout.write(json.dumps({
                "shard": tag, "fork_line": n, "battle": r["base"], "inv": r["inv"],
                "turn": r["turn"], "k": args.k, "seed_salt": f"{r['base']}:{r['inv']}:{args.seed}",
                "succ": {b: int(r["branches"][b]["succ"]) for b in BRANCHES},
                "banked_outcome": {b: r["branches"][b]["outcome"] for b in BRANCHES},
                "banked_capped": {b: bool(r["branches"][b]["capped"]) for b in BRANCHES},
                "mean": {b: float(np.mean(per[b]["scores"])) for b in BRANCHES},
                "scores": {b: per[b]["scores"] for b in BRANCHES},
                "capped_k": {b: per[b]["capped"] for b in BRANCHES},
                "mean_turns": {b: float(np.mean(per[b]["turns"])) for b in BRANCHES},
            }) + "\n")
            stats["forks"] += 1
            if stats["forks"] % 5 == 0:
                fout.flush()
                el = time.time() - t0
                print(f"  [{tag}] {stats['forks']}/{len(elig)} forks ({el:.0f}s, "
                      f"{el/max(1,stats['forks']):.1f}s/fork) {stats}", flush=True)
        except Exception as e:                        # noqa: BLE001
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  ERR {r['base']}#{r['inv']}: {type(e).__name__} {e}", flush=True)
                traceback.print_exc()
    fout.close()
    meta = {"shard": args.shard, "tag": tag, "k": args.k, "sample_target": take,
            "n_eligible_sampled": len(elig), "stats": stats, "wall_s": time.time() - t0,
            "seed": args.seed, "impl": args.impl, "forks_dir": args.forks, "tree": tree}
    json.dump(meta, open(os.path.join(args.out, f"rerollmeta_{tag}.json"), "w"), indent=1)
    print(f"[reroll {tag}] DONE {json.dumps(stats)} in {meta['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
