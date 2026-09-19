"""JOB 1 — THE K-CURVE's dice: 16 FRESH rollouts per branch on the EXACT forks the 2026-09-19
control already banked 8 for.

    python3 reroll_extend.py --forks <banked forks dir> --banked <reroll dir> --snapshot <zip> \
        --out <dir> --shard i/6 --k-start 8 --k-end 24 --verify 4

WHY A SECOND SCRIPT AND NOT `--k 24`. `reroll_labels.py --k 24` would re-run re-rolls 0-7, which are
already on disk — 24,048 rollouts of pure repetition. ``fresh_seeds(n, salt)`` is
``sha256(f"{salt}:{i}")`` for ``i < n``, so ``fresh_seeds(24)`` is a strict EXTENSION of
``fresh_seeds(8)``: indices 8..23 are new dice, indices 0..7 are the banked ones unchanged. This
script draws the extension only.

THE SPLIT THAT MAKES THE CURVE A MATCHED-NOISE READ.

* **THE LABEL is re-rolls 0-7** — the BANKED K' = 8 mean, one label set for EVERY K, so the whole
  curve is read against the same target and the levels are comparable to each other and to the
  head. (`leaf_ceiling_controls_2026-09-19` §5.1 read exactly this label: the head is 0.5801 pooled
  / 0.5610 on `top1|top2` against it.)
* **THE LEAF is re-rolls 8-23** — dice the label has never seen. So `ROLLOUT_K` is an INDEPENDENT
  estimate of the very same successors, and whatever it reads is what a value function of
  K-rollout quality reads on this label. No modelling, no ceiling arithmetic, no tie-rate
  assumption: two scorers, one label, one pair set.
* **NESTED ACROSS K, declared.** `ROLLOUT_K` = mean of re-rolls 8..8+K-1, so a smaller K is a strict
  SUB-SAMPLE of the larger one's dice. The levels are therefore positively correlated — which is
  the right design for reading a monotone DOSE curve (the K-to-K difference is paired and carries
  no between-K dice noise) and is wrong for treating two levels as independent samples. Every CI is
  still bootstrapped over FORKS.

CRN WITHIN A RE-ROLL INDEX is inherited unchanged: for re-roll k the three branches share one seed,
so the k-th `top1` and the k-th `top2` differ in exactly the action.

🚨 **THE JOIN IS ASSERTED, NOT ASSUMED.** The fork list is read from the BANKED rows rather than
re-derived from the sampler, and every row's recomputed ``seed_salt`` must equal the banked row's —
a mismatch means the new dice belong to a different fork and the row is REFUSED. ``--verify N``
additionally re-runs re-roll index 0 for the first N forks of the shard and requires the score to
reproduce the banked one EXACTLY: that is a CRN reproduction check of the whole harness (model
load, sentinel resolution, replay, reseed, scoring), not of the arithmetic above it.

A capped line scores 0.5 (`cf_producer.rollout_outcome_score`, imported). FORK-OUTER / K-INNER and
resumable on ``fork_line``, so a wall-clock stop yields FEWER FORKS AT FULL K rather than all forks
at a ragged K. CPU only; nothing is written under ``models/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

BRANCHES = ("top1", "top2", "rand")
_MEAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MEAS, "paired_refit_discrimination_2026-09-14"))
sys.path.insert(0, os.path.join(_MEAS, "leaf_ceiling_controls_2026-09-19"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--banked", required=True, help="the 2026-09-19 reroll dir (reroll_s*.jsonl)")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/6")
    ap.add_argument("--k-start", type=int, default=8)
    ap.add_argument("--k-end", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260919, help="MUST match the banked salt")
    ap.add_argument("--verify", type=int, default=4, help="forks whose re-roll 0 is re-run and "
                                                          "required to reproduce the banked score")
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
    from reroll_labels import run_branch               # by path, unmodified — the SAME rollout

    os.makedirs(args.out, exist_ok=True)
    i, w = (int(x) for x in args.shard.split("/"))
    t0 = time.time()

    paths = sorted(glob.glob(os.path.join(args.forks, "forks_s*.jsonl")))
    if not (0 <= i < len(paths)) or w != len(paths):
        raise SystemExit(f"REFUSED: --shard {args.shard} does not name one of {len(paths)} shards")
    rp = paths[i]
    tag = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
    S = np.load(os.path.join(args.forks, f"succ_{tag}.npy"), mmap_mode="r")
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

    # THE FORK LIST IS THE BANKED ONE. Not re-derived from the sampler: re-deriving would make the
    # join depend on reproducing a shuffle, and a silent drift there would pair fresh dice with the
    # wrong fork's label.
    banked = {}
    bp = os.path.join(args.banked, f"reroll_{tag}.jsonl")
    for line in open(bp):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            break
        banked[int(d["fork_line"])] = d
    order = sorted(banked)
    print(f"[ext {tag}] {len(rows)} banked forks in shard, {len(order)} with banked re-rolls, "
          f"k=[{args.k_start},{args.k_end})", flush=True)

    bmeta = json.load(open(os.path.join(args.forks, f"meta_{tag}.json")))
    tree = bmeta["tree"]
    plan = json.load(open(os.path.join(tree, "plan.json")))
    sent = {it["key"]: it.get("path") for it in plan["items"] if it.get("kind") == "sentinel"}

    play_model = load_model(args.snapshot)
    opp_models = {}
    mappings = load_mappings()

    out_path = os.path.join(args.out, f"rerollx_{tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["fork_line"])
            except Exception:                          # noqa: BLE001 — a torn last line
                break
    fout = open(out_path, "a")
    stats = {"forks": 0, "rollouts": 0, "errors": 0, "capped": 0, "skipped_done": 0,
             "salt_mismatch": 0, "verify_ok": 0, "verify_bad": 0}
    n_verified = 0
    for n in order:
        if n in done:
            stats["skipped_done"] += 1
            continue
        r = rows[n]
        bk = banked[n]
        salt = f"{r['base']}:{r['inv']}:{args.seed}"
        if salt != bk.get("seed_salt"):
            stats["salt_mismatch"] += 1
            print(f"  REFUSED fork_line {n}: salt {salt!r} != banked {bk.get('seed_salt')!r}",
                  flush=True)
            continue
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
            seeds = fresh_seeds(args.k_end, salt=salt)
            ks = list(range(args.k_start, args.k_end))
            if n_verified < args.verify:
                ks = [0] + ks                          # re-roll 0 must reproduce the banked score
            per = {b: {"scores": [], "capped": 0, "turns": []} for b in BRANCHES}
            vscore = {}
            ok = True
            for k in ks:
                for b in BRANCHES:
                    res = run_branch(rec, trace, r["turn"], r["branches"][b]["action"],
                                     play_model=play_model, opp_model=opp_models[r["opp"]],
                                     opp_ckpt=sent[r["opp"]], mappings=mappings, impl=args.impl,
                                     # the VERIFY rollout reuses the BANKED run's own
                                     # player tag, so re-roll 0 is a bit-for-bit
                                     # reproduction attempt and not a near-miss.
                                     tag=(f"r{i}_{n}_{k}_{b}" if k == 0 else
                                          f"x{i}_{n}_{k}_{b}"), post_t_seed=seeds[k])
                    if res is None:
                        ok = False
                        break
                    stats["rollouts"] += 1
                    sc = float(rollout_outcome_score(res))
                    if k == 0:
                        vscore[b] = sc
                        continue
                    per[b]["scores"].append(sc)
                    per[b]["turns"].append(int(res.get("turns") or 0))
                    if res.get("capped"):
                        per[b]["capped"] += 1
                        stats["capped"] += 1
                if not ok:
                    break
            if not ok:
                stats["errors"] += 1
                continue
            verify = None
            if vscore:
                n_verified += 1
                verify = {b: [vscore[b], bk["scores"][b][0]] for b in BRANCHES}
                good = all(abs(vscore[b] - bk["scores"][b][0]) < 1e-9 for b in BRANCHES)
                stats["verify_ok" if good else "verify_bad"] += 1
                print(f"  [verify {tag}#{n}] CRN re-roll 0 "
                      f"{'REPRODUCED' if good else '🚨 MISMATCH'} {verify}", flush=True)
            fout.write(json.dumps({
                "shard": tag, "fork_line": n, "battle": r["base"], "inv": r["inv"],
                "turn": r["turn"], "k_start": args.k_start, "k_end": args.k_end,
                "seed_salt": salt,
                "succ": {b: int(r["branches"][b]["succ"]) for b in BRANCHES},
                "scores": {b: per[b]["scores"] for b in BRANCHES},
                "capped_k": {b: per[b]["capped"] for b in BRANCHES},
                "mean_turns": {b: float(np.mean(per[b]["turns"])) for b in BRANCHES},
                "verify": verify,
            }) + "\n")
            stats["forks"] += 1
            if stats["forks"] % 5 == 0:
                fout.flush()
                el = time.time() - t0
                print(f"  [{tag}] {stats['forks']}/{len(order)} forks ({el:.0f}s, "
                      f"{el/max(1,stats['forks']):.1f}s/fork) {stats}", flush=True)
        except Exception as e:                        # noqa: BLE001
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  ERR {r['base']}#{r['inv']}: {type(e).__name__} {e}", flush=True)
                traceback.print_exc()
    fout.close()
    meta = {"shard": args.shard, "tag": tag, "k_start": args.k_start, "k_end": args.k_end,
            "n_banked": len(order), "stats": stats, "wall_s": time.time() - t0,
            "seed": args.seed, "impl": args.impl, "forks_dir": args.forks,
            "banked_dir": args.banked, "tree": tree}
    json.dump(meta, open(os.path.join(args.out, f"rerollxmeta_{tag}.json"), "w"), indent=1)
    print(f"[ext {tag}] DONE {json.dumps(stats)} in {meta['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
