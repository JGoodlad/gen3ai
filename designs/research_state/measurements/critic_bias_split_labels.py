"""PROBE G — per-decision CRITIC vector vs CRN-paired Monte-Carlo LABEL vector.

For each sampled recorded decision, for EVERY legal root action:
  C[a] = the critic's one-ply read  (lookahead: materialize the CRN successor, read the
         win-prob head at s')  -- win-prob units, [0,1]
  L[a] = a tight Monte-Carlo value label (replay_counterfactual: substitute a, play the rest
         LIVE, trainee greedy vs the RELOADED real opponent, R rollouts to a terminal
         win/loss) -- win-prob units, [0,1]

The R post-divergence dice seeds are derived from `<battle_tag>:<inv>:cf` with NO action in the
salt, so every sibling action at one decision is rolled on the SAME seed list: common random
numbers, which is the whole point (the paired difference L[a]-L[b] is what we need resolved).

Run:
  nice -n 15 python tmp/probe_g_labels.py --shard 0 --shards 2 --rollouts 8 --decisions 120
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import json
import os
import random
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

from agents.training.cf_audit import build_frame  # noqa: E402
from main.prober.falsifier import anchor_deltas  # noqa: E402
import main.prober.replay as _rp  # noqa: E402

# SPLIT-HALF CRN BLOCKS. `replay_counterfactual` derives its R post-divergence dice seeds from
# `<battle_tag>:<inv>:cf` with NO action in the salt -- that is the common-random-numbers coupling
# across sibling actions, and it is exactly what we want. But it also means two calls with R/2
# would replay the SAME dice, so there would be no way to measure the label noise floor.
# This wrapper appends a per-BLOCK tag to the salt: within one block every action still shares one
# seed list (CRN preserved), while block A and block B are independent draws. The two blocks
# average to the full-R label, and their DIFFERENCE measures the noise -- including whatever
# variance the CRN coupling already removed, which no closed-form binomial floor can know.
_ORIG_FRESH_SEEDS = _rp.fresh_seeds
_BLOCK = {"tag": ""}


def _blocked_fresh_seeds(n, salt):
    return _ORIG_FRESH_SEEDS(n, salt + _BLOCK["tag"])


_rp.fresh_seeds = _blocked_fresh_seeds


MIN_LEGAL = 3


def enrich(frame, gamma: float):
    """Attach n_legal + |TD delta| to every frame row, reading each trace's npz ONCE."""
    import collections
    by_battle = collections.defaultdict(list)
    for d in frame:
        by_battle[d.battle].append(d)
    rows = []
    for base, ds in by_battle.items():
        with np.load(base + "_states.npz") as z:
            mask = np.asarray(z["action_mask"], dtype=bool)
            npz = {k: z[k] for k in z.files}
        with open(base + "_summary.json") as f:
            summ = json.load(f)
        try:
            deltas = anchor_deltas(summ, npz, gamma=gamma)
        except (ValueError, KeyError):
            deltas = {}
        for d in ds:
            n_legal = int(mask[d.inv].sum())
            rows.append({
                "battle": d.battle, "short": d.short, "opponent": d.opponent,
                "opp_class": d.opp_class, "outcome": d.outcome, "inv": d.inv, "turn": d.turn,
                "win_prob": d.win_prob, "value": d.value, "action": d.action,
                "move_rank": d.move_rank, "n_moves": d.n_moves, "n_legal": n_legal,
                "td_delta": float(deltas[d.inv]) if d.inv in deltas else None,
            })
    return rows


def sample(rows, n, *, seed, pivotal_frac=0.5):
    """Half PIVOTAL (largest |TD delta|), half ORDINARY, at most 3 per battle, >=MIN_LEGAL actions."""
    elig = [r for r in rows if r["n_legal"] >= MIN_LEGAL and r["td_delta"] is not None]
    if not elig:
        return []
    d = np.asarray([abs(r["td_delta"]) for r in elig])
    cut = float(np.quantile(d, 0.85))
    piv = [r for r in elig if abs(r["td_delta"]) >= cut]
    ord_ = [r for r in elig if abs(r["td_delta"]) < cut]
    rng = random.Random(seed)
    rng.shuffle(piv)
    rng.shuffle(ord_)
    out, per_battle = [], {}
    want_p = int(round(n * pivotal_frac))
    for pool, tag, want in ((piv, "pivotal", want_p), (ord_, "ordinary", n - want_p)):
        got = 0
        for r in pool:
            if got >= want:
                break
            if per_battle.get(r["battle"], 0) >= 3:
                continue
            per_battle[r["battle"]] = per_battle.get(r["battle"], 0) + 1
            out.append(dict(r, stratum=tag))
            got += 1
    rng.shuffle(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823")
    ap.add_argument("--step", default="step_24000000")
    ap.add_argument("--decisions", type=int, default=120)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--out", default="tmp/probe_g_out")
    ap.add_argument("--frame-only", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=1e9,
                    help="stop starting new decisions after this many wall-clock minutes")
    a = ap.parse_args(argv)

    traces = os.path.join(a.run, "eval_traces", a.step)
    ckpt = os.path.join(traces, "snapshot.zip")            # the EXACT model that wrote the traces
    frame, skipped = build_frame(traces)
    rows = enrich(frame, gamma=0.99)
    picked = sample(rows, a.decisions, seed=a.seed)

    if a.frame_only:
        import collections
        nl = collections.Counter(r["n_legal"] for r in rows)
        print(json.dumps({
            "traces": traces, "frame": len(rows),
            "battles": len({r["battle"] for r in rows}), "skipped": dict(skipped),
            "n_legal_hist": dict(sorted(nl.items())),
            "eligible_ge3": sum(1 for r in rows if r["n_legal"] >= MIN_LEGAL),
            "picked": len(picked),
            "picked_actions": sum(r["n_legal"] for r in picked),
            "picked_strata": dict(collections.Counter(r["stratum"] for r in picked)),
        }, indent=2))
        return 0

    mine = [r for i, r in enumerate(picked) if i % a.shards == a.shard]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"shard{a.shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done.add((r["battle"], r["inv"]))
                except ValueError:
                    pass

    from main.prober.session import ProbeSession
    from agents.training.cf_audit import sentinel_snapshots
    sess = ProbeSession(traces, ckpt_override=ckpt, impl=a.impl, compile_extractor=True)
    # Pin the REAL sentinel weights (cf_audit does the same): the fallback is the trainee's own
    # model standing in for the opponent, which is a different opponent distribution.
    snaps = sentinel_snapshots(a.run)
    t_start = time.time()
    n_ok = 0
    with open(out_path, "a", buffering=1) as fh:
        for k, r in enumerate(mine):
            if (r["battle"], r["inv"]) in done:
                continue
            if (time.time() - t_start) / 60.0 > a.deadline_min:
                print(f"[shard{a.shard}] DEADLINE — stopping after {n_ok} decisions", flush=True)
                break
            bid = r["battle"] + "_summary.json"
            opp_ckpt = snaps.get(r["opponent"])
            t0 = time.time()
            rec = dict(r, ckpt=ckpt, rollouts=a.rollouts, impl=a.impl)
            try:
                la = sess.lookahead(bid, inv=r["inv"])
            except Exception as e:                                        # noqa: BLE001
                rec["error"] = f"lookahead:{type(e).__name__}:{e}"
                fh.write(json.dumps(rec) + "\n")
                print(f"[shard{a.shard}] {k+1}/{len(mine)} {r['short']}#{r['inv']} ERR {rec['error'][:120]}",
                      flush=True)
                continue
            rec["critic"] = [
                {"action": c["action"], "label": c["label"], "choice": c["choice"],
                 "is_chosen": c["is_chosen"], "win_prob": c["win_prob_crn"],
                 "value": c["value_crn"], "terminal": c["terminal"],
                 "terminal_frac": c["terminal_frac"]}
                for c in la["candidates"]
            ]
            rec["recorded_next_value"] = la.get("recorded_next_value")
            rec["lookahead_recorded_value"] = la.get("recorded_value")
            half = max(1, a.rollouts // 2)
            labels = []
            for c in rec["critic"]:
                row = {"action": int(c["action"])}
                try:
                    for tag in ("A", "B"):
                        _BLOCK["tag"] = f":blk{tag}"
                        cf = sess.replay_counterfactual(bid, r["inv"], int(c["action"]),
                                                        n_rollouts=half,
                                                        opponent_ckpt=opp_ckpt)
                        row[f"wins_{tag}"] = cf["wins"]
                        row[f"n_{tag}"] = cf["n_rollouts"]
                        row[f"outcomes_{tag}"] = cf["outcomes"]
                        row["opponent_source"] = cf["opponent_source"]
                    row["wins"] = row["wins_A"] + row["wins_B"]
                    row["n"] = row["n_A"] + row["n_B"]
                    row["win_rate"] = row["wins"] / row["n"]
                except Exception as e:                                    # noqa: BLE001
                    row["error"] = f"{type(e).__name__}:{e}"[:300]
                finally:
                    _BLOCK["tag"] = ""
                labels.append(row)
            rec["labels"] = labels
            rec["secs"] = round(time.time() - t0, 2)
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            ok = sum(1 for x in labels if "error" not in x)
            print(f"[shard{a.shard}] {k+1}/{len(mine)} {r['short']}#{r['inv']} "
                  f"legal={r['n_legal']} labelled={ok}/{len(labels)} {rec['secs']}s "
                  f"(elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)
    sess.close()
    print(f"[shard{a.shard}] done: {n_ok} decisions in {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
