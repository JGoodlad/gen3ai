"""Build the PAIRED SIBLING fork dataset — three branches per contested decision, common dice.

    python3 forks.py --tree <eval_traces/step_N> --snapshot <snapshot.zip> --out <dir> \
        --opponents sentinel_2,sentinel_1 --shard i/W [--battles N] [--max-forks N]

WHAT IT PRODUCES, and why each choice is the way it is.

One row per FORK (a recorded decision), carrying THREE branches:

  top1  the frozen policy's argmax at that decision
  top2  its runner-up
  rand  one uniformly-random legal alternative, drawn from legal \ {top1, top2} with a
        DECISION-KEYED rng (reproducible, and independent of the policy's own ordering — a
        probability-ordered draw would spend the branch on what the policy already covers)

All three replay the recorded prefix to the divergence turn and then play LIVE to a terminal.

COMMON RANDOM NUMBERS, and how they are actually obtained.
`replay_counterfactual(seed=<the record's own resolved >start seed>, post_t_seed=None)` hands the
sim ONE dice stream for the whole line, so the three branches differ in exactly one thing: the
action at turn T. That is the pairing. Two further sources of randomness would break it and are
closed rather than noted:

  * OUR side plays GREEDY (``RLPlayer(stochastic=False)``), so no torch draw enters.
  * the OPPONENT is a SENTINEL reloaded from the exact snapshot the manifest names, played
    GREEDY — which is the regime it was RECORDED in (this tree's manifest has
    ``eval_sentinel_greedy: True``). `cf_q_labels` records a residual policy-draw variance its
    temperature-1.0 rollouts cannot remove; at temperature 0 on both sides it does not exist.

`--determinism-check` re-runs one branch of each fork a second time and REFUSES the shard if the
two runs disagree — the pairing is asserted, not assumed.

CONTESTED SELECTION (declared in PREDICTION.md, and it does NOT read the head being refit):
a ``move_selection`` decision at turn >= 2 with >= 3 legal actions whose POLICY top-2 masked-logit
gap is below ``--gap-quantile`` of that quantity over the shard's own candidates. |V-0.5| is
RECORDED per decision but never selects — selecting on V would make the held-out read partly a
measurement of the selector.

THE SUCCESSOR STATE is the quantity the refit is about, so it is captured rather than re-derived:
the trainee's ``choose_move`` is wrapped BEFORE the scripted prefix is installed, so the wrapper
becomes the live policy the prefix defers to, and on the FIRST live call it takes the obs with the
tracker's own snapshot/restore discipline (the same one `RLPlayer` uses for a stale re-decide) so
the capture leaves no phantom turn in the history. A branch that ENDS at the divergence turn has
no successor and is recorded with ``succ: null``.

A branch that reaches the 250-turn stall-forfeit cap is a DRAW AT CAP: its win/loss is decided by
SEAT, not by the position (`_battle_outcome`), so it is EXCLUDED, never scored 0.5.

Nothing is written under ``models/``; CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from typing import Optional

import numpy as np

BRANCHES = ("top1", "top2", "rand")


def _sha(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def load_model(path, device="cpu"):
    from sb3_contrib import MaskablePPO
    from main.prober.model import sanitized_load_custom_objects
    custom_objects, _ = sanitized_load_custom_objects(path, device)
    m = MaskablePPO.load(path, env=None, device=device, custom_objects=custom_objects)
    m.policy.set_training_mode(False)
    for mod in m.policy.modules():
        if hasattr(mod, "_debugger"):
            mod._debugger = None
    return m


def score_batch(model, obs, masks):
    """(value_pooled[B,128], V[B]) from ONE forward. The recorded ``logits``/``win_probs`` in a
    trace's ``states.npz`` are used for pass 1 instead (they reproduce this forward exactly — the
    N-curve's frozen-forward QC measured max |V_fwd - V_rec| = 2.2e-06 on this checkpoint), so
    this is called only on SUCCESSOR states, which no recording contains."""
    import torch
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    d = {"observation": torch.as_tensor(obs),
         "action_mask": torch.as_tensor(np.asarray(masks, dtype=np.float32))}
    ex = model.policy.features_extractor
    with torch.no_grad():
        model.policy.get_distribution(d)
        pooled = ex.stash.value_pooled
        wpl = ex.stash.win_prob_logits
        if wpl is None:
            raise SystemExit("REFUSED: checkpoint stashes no win_prob_logits")
        v = torch.sigmoid(wpl.reshape(-1))
    return pooled.numpy().copy(), v.numpy().copy()


class SuccessorCapture:
    """Wrap a player's ``choose_move`` so the FIRST LIVE call records its obs."""

    def __init__(self, player):
        self.player = player
        self.obs = None
        self.mask = None
        self._orig = player.choose_move

        def wrapped(battle):
            if self.obs is None:
                try:
                    tracker = player._get_tracker(battle)
                    snap = tracker.snapshot()
                    d = player.embed_battle(battle)
                    tracker.restore(snap)
                    m = np.asarray(d["action_mask"]).reshape(-1)
                    if int(m.sum()) > 0:
                        self.obs = np.asarray(d["observation"], dtype=np.float32)
                        self.mask = m.astype(np.int8)
                except Exception:  # noqa: BLE001 — a capture failure must not kill the rollout
                    self.obs = self.obs
            return self._orig(battle)

        player.choose_move = wrapped


def run_branch(record, trace, turn, action, *, play_model, opp_model, opp_ckpt, mappings, impl,
               tag):
    """One branch: substitute ``action`` at ``turn``, play to a terminal on the RECORD's own dice.

    Returns ``(outcome_str, capped, turns, successor_obs_or_None, successor_mask_or_None)``."""
    from poke_env.ps_client import LocalhostServerConfiguration
    from main.prober.replay import build_opponent, build_trainee
    from utils.bridge.counterfactual import replay_counterfactual as _run_one

    choice_map = trace.action_choices or {}
    if int(action) not in choice_map:
        return None, False, 0, None, None
    trainee = build_trainee(play_model, record, mappings, LocalhostServerConfiguration, tag=tag)
    opponent, _src = build_opponent(
        "", record, play_model, mappings, LocalhostServerConfiguration,
        opponent_ckpt=opp_ckpt, opp_model=opp_model, opponent_source="ckpt",
        opponent_stochastic=False, tag=tag)
    cap = SuccessorCapture(trainee)
    res = _run_one(record, trainee=trainee, opponent=opponent, divergence_turn=int(turn),
                   substitute_choice=choice_map[int(action)], seed=record.start_options().get("seed"),
                   post_t_seed=None, impl=impl)
    return res["outcome"], bool(res.get("capped")), int(res.get("turns") or 0), cap.obs, cap.mask


def battle_files(tree, opponents):
    out = []
    for opp in opponents:
        d = os.path.join(tree, opp)
        if not os.path.isdir(d):
            continue
        for s in sorted(os.listdir(d)):
            if not s.endswith("_summary.json"):
                continue
            base = os.path.join(d, s[: -len("_summary.json")])
            if os.path.exists(base + "_reconstruction.json") and os.path.exists(base + "_states.npz"):
                out.append((opp, base))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opponents", default="sentinel_2,sentinel_1,sentinel_0")
    ap.add_argument("--shard", default="0/1", help="i/W — disjoint battle slices")
    ap.add_argument("--battles", type=int, default=0, help="cap on battles this shard scans")
    ap.add_argument("--max-forks", type=int, default=0)
    ap.add_argument("--gap-quantile", type=float, default=0.40)
    ap.add_argument("--min-legal", type=int, default=3)
    ap.add_argument("--min-turn", type=int, default=2)
    ap.add_argument("--max-turn", type=int, default=40,
                    help="a fork later than this is near-terminal and, on a recorded DRAW, already "
                         "inside a stall loop: the smoke put two forks at turns 178/183 and five of "
                         "their six branches hit the 250-turn cap")
    ap.add_argument("--exclude-draws", type=int, default=1,
                    help="skip battles whose RECORDED result is a draw (they are stall-loop games; "
                         "a draw at the cap is decided by SEAT, not by the position)")
    ap.add_argument("--forks-per-battle", type=int, default=2)
    ap.add_argument("--impl", default="rust", choices=["node", "rust"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--determinism-check-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from agents.observation.state_encoder import load_mappings
    from agents.training.obs_materializer import materialize_from_record
    from utils.bridge.reconstruction import ReconstructionRecord

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
            raise SystemExit(f"REFUSED: {o} is not a SENTINEL in this tree's plan. Only a sentinel "
                             f"is reloadable EXACTLY and was recorded GREEDY; a bot opponent is "
                             f"rebuilt from code whose internal RNG this pairing does not control.")
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
    print(f"[forks {tag}] {len(files)} battles, opponents={opponents}, impl={args.impl}", flush=True)

    # ---- pass 1: candidate decisions + the shard's own gap threshold -------------------------
    cands = []   # (opp, base, inv_index, turn, gap, legal_tuple, top1, top2, V, |V-.5|)
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
        keep = [k for k, iv in enumerate(invs)
                if iv.get("phase") == "move_selection"
                and args.min_turn <= int(iv.get("turn", 0)) <= args.max_turn
                and k < len(obs) and int(masks[k].sum()) >= args.min_legal]
        if not keep:
            continue
        rec_lg = np.asarray(npz["logits"], dtype=np.float64)
        rec_v = np.asarray(npz["win_probs"], dtype=np.float64).reshape(-1)
        for k in keep:
            legal = np.flatnonzero(masks[k]).astype(int)
            row = rec_lg[k].copy()
            row[masks[k] == 0] = -np.inf
            order = np.argsort(-row)
            a1, a2 = int(order[0]), int(order[1])
            gap = float(row[a1] - row[a2])
            cands.append({"opp": opp, "base": base, "inv": int(k), "turn": int(invs[k]["turn"]),
                          "gap": gap, "legal": legal.tolist(), "top1": a1, "top2": a2,
                          "V": float(rec_v[k]), "absV": abs(float(rec_v[k]) - 0.5),
                          "result": ((summ.get("meta") or {}).get("result") or "").lower()})
        if (n_b + 1) % 100 == 0:
            print(f"  pass1 {n_b+1}/{len(files)} battles, {len(cands)} candidates "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if not cands:
        raise SystemExit("REFUSED: no candidate decisions")
    thr = float(np.quantile([c["gap"] for c in cands], args.gap_quantile))
    sel = [c for c in cands if c["gap"] <= thr]
    print(f"[forks {tag}] candidates={len(cands)} gap_thr={thr:.4f} selected={len(sel)} "
          f"(rate {len(sel)/len(cands):.3f})", flush=True)

    # at most --forks-per-battle per battle, the TIGHTEST gaps first (a battle's forks share a
    # prefix, so many from one battle buy less than the same count spread over battles)
    by_battle = {}
    for c in sorted(sel, key=lambda c: c["gap"]):
        by_battle.setdefault(c["base"], []).append(c)
    # The BATTLE order is shuffled and each battle's forks stay together. That buys both things
    # at once: one reconstruction-record load per battle (locality), and an UNBIASED PREFIX — a
    # shard stopped early by the clock returns a random subset of battles rather than an
    # alphabetical one, and this tree's filenames sort `draw_* < loss_* < win_*`, so an
    # alphabetical prefix would be a sample of LOSSES. (Caught in the 20-battle smoke: all 12
    # forks came from loss battles.)
    rng = random.Random(args.seed + i)
    order = list(by_battle)
    rng.shuffle(order)
    picked = []
    for b in order:
        picked.extend(sorted(by_battle[b][: args.forks_per_battle], key=lambda c: c["inv"]))
    if args.max_forks:
        picked = picked[: args.max_forks]
    print(f"[forks {tag}] forking {len(picked)} decisions over {len(by_battle)} battles", flush=True)

    # ---- pass 2: the forks ------------------------------------------------------------------
    succ_vecs, out_rows = [], []
    stats = {"forks": 0, "errors": 0, "no_successor": 0, "capped": 0, "det_checks": 0,
             "det_fail": 0, "no_rand": 0}
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
            others = [a for a in legal_in_trace if a not in (c["top1"], c["top2"])]
            if not others:
                stats["no_rand"] += 1
                continue
            r_act = others[_sha(c["base"], c["inv"], args.seed) % len(others)]
            acts = {"top1": c["top1"], "top2": c["top2"], "rand": int(r_act)}
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
                                            play_model=play_model, opp_model=opp_models[c["opp"]],
                                            opp_ckpt=sent[c["opp"]], mappings=mappings,
                                            impl=args.impl, tag=f"{i}_{n}_det")
                stats["det_checks"] += 1
                if o2 != br["top1"]["outcome"]:
                    stats["det_fail"] += 1
                    print(f"  🚨 DETERMINISM FAIL at {c['base']}#{c['inv']}: "
                          f"{br['top1']['outcome']} vs {o2}", flush=True)
            row = {"opp": c["opp"], "base": os.path.basename(c["base"]), "inv": c["inv"],
                   "turn": c["turn"], "gap": c["gap"], "V_rec": c["V"], "absV": c["absV"],
                   "n_legal": len(c["legal"]), "recorded_result": c["result"],
                   "branches": br, "shard": i}
            out_rows.append(row)
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
            "gap_threshold": thr, "gap_quantile": args.gap_quantile,
            "n_candidates": len(cands), "n_selected": len(sel), "selection_rate":
                len(sel) / len(cands), "n_picked": len(picked), "stats": stats,
            "wall_s": time.time() - t0, "impl": args.impl, "seed": args.seed,
            "min_legal": args.min_legal, "min_turn": args.min_turn,
            "forks_per_battle": args.forks_per_battle}
    json.dump(meta, open(os.path.join(args.out, f"meta_{tag}.json"), "w"), indent=1)
    print(f"[forks {tag}] DONE {json.dumps(meta['stats'])} in {meta['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
