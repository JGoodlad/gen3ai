"""PROBE L, stage 2 — the one-ply WIN-PROB read at whiff decisions, and at MATCHED CONTROLS.

For every sampled decision, sweep every legal action one ply (opponent plays its RECORDED move,
CRN on the realized dice) and read the model's **win-prob head** and scalar **V** at each
successor. The question is whether, AT DECISION TIME, the head ranked a non-whiff alternative
above the whiff the policy fired.

Two things make this a measurement rather than a number:

* **CONTROLS.** Probe G measured that the one-ply win-prob head beats the played action GENERALLY
  (+0.0219, 35% agreement) — so "the head prefers an alternative on X% of whiffs" is meaningless
  without the base rate. Every whiff decision is therefore accompanied by matched controls drawn
  from the same battles: `hit_pivot` (they pivoted, we moved in, it CONNECTED) and `no_pivot`.
* **A MEASURED DICE FLOOR.** The headline is the CRN line (one realized dice stream). `--seeds R`
  re-rolls each action on R further INDEPENDENT dice streams and reads the win-prob head on each,
  so the CRN margin can be quoted against the spread it actually has. `win_prob_crn` is the only
  win-prob `lookahead` returns, so the per-seed values are captured by wrapping the model's
  `value()` — the one call made per (action, seed) — and the reconstruction is VERIFIED against
  the returned `win_prob_crn` per decision (`wp_map_verified`); an unverified decision keeps its
  CRN row and is excluded from the floor.

Run:
  nice -n 15 python tmp/probe_l_sweep.py --shard 0 --shards 2 --seeds 6
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

RUN = "/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823"
MIN_LEGAL = 3          # a 2-action decision has almost no ranking to get right


class WPRecorder:
    """Delegate every ProbeModel call, and record the win-prob head beside every `value()`.

    `lookahead_decision` calls `model.value(succ.obs, succ.mask)` exactly ONCE per surviving
    (action, seed) arm, in `for a in cand: for s in seed_list` order with `"original"` LAST. That
    is the only hook needed to get a per-seed win-prob out of the shipped sweep without forking it;
    the ordering assumption is then CHECKED, per decision, against the `win_prob_crn` the sweep
    itself returns."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: "list[tuple[float, float | None]]" = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def value(self, obs, mask):
        v = self._inner.value(obs, mask)
        self.calls.append((float(v), self._inner.win_prob_at(obs, mask)))
        return v


def attach_wp(rows, calls, cand_order):
    """Split the flat call log into per-action blocks and return {action: [wp per evaluated seed]}.

    Returns (per_action, verified). `verified` is True when every action whose CRN arm survived has
    its LAST recorded win-prob equal to the sweep's own `win_prob_crn` — i.e. the ordering
    assumption held."""
    by_action = {int(r["action"]): r for r in rows}
    out, i, verified = {}, 0, True
    for a in cand_order:
        n = int(by_action[a]["n_evaluated"])
        block = calls[i:i + n]
        i += n
        out[a] = [wp for (_v, wp) in block]
        crn = by_action[a]["win_prob_crn"]
        if crn is not None:
            if not block or block[-1][1] is None or abs(block[-1][1] - crn) > 1e-4:
                verified = False
    if i != len(calls):
        verified = False
    return out, verified


def build_population(census_path, *, seed, control_ratio=1.0):
    """Whiff decisions + matched controls from the SAME battles.

    A control is drawn from the battles that produced a whiff, so the population and its baseline
    share the game length / winning-position confounds the bait hunt registered."""
    rows = [json.loads(ln) for ln in open(census_path)]
    whiffs, pool_hit, pool_none = [], [], []
    for r in rows:
        if not r["has_recon"] or not r["has_actions_array"]:
            continue
        loop_ord = {}
        for g in r["loops"]:
            for k, t in enumerate(sorted(g["turns"])):
                loop_ord[(g["move"], g["arrival"], t)] = k + 1     # 1-based click ordinal
        has_whiff = any(b["whiff"] and b["inv"] is not None for b in r["baits"])
        for b in r["baits"]:
            if b["whiff"] and b["inv"] is not None:
                whiffs.append({
                    "base": r["base"], "opponent": r["opponent"], "step": r["step"],
                    "outcome": r["outcome"], "inv": b["inv"], "turn": b["turn"],
                    "tag": "whiff", "kind": b["kind"], "move": b["move"],
                    "arrival": b["arrival"], "loop_step": b["loop_step"],
                    "reclick": b["reclick"],
                    "click_ordinal": loop_ord.get((b["move"], b["arrival"], b["turn"])),
                    "chosen_prob": b["chosen_prob"], "delta_v": b["delta_v"],
                    "delta_win_prob": b["delta_win_prob"],
                })
        if not has_whiff:
            continue
        for d in r["decisions"]:
            if (d["n_legal"] or 0) < MIN_LEGAL:
                continue
            row = {"base": r["base"], "opponent": r["opponent"], "step": r["step"],
                   "outcome": r["outcome"], "inv": d["inv"], "turn": d["turn"],
                   "tag": d["tag"], "kind": None, "move": None, "arrival": None,
                   "loop_step": False, "reclick": False, "click_ordinal": None,
                   "chosen_prob": d["chosen_prob"], "delta_v": None, "delta_win_prob": None}
            if d["tag"] == "hit_pivot":
                pool_hit.append(row)
            elif d["tag"] == "no_pivot":
                pool_none.append(row)
    rng = random.Random(seed)
    rng.shuffle(pool_hit)
    rng.shuffle(pool_none)
    n = int(round(len(whiffs) * control_ratio))
    return whiffs + pool_hit[:n] + pool_none[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="tmp/whiff_census_all.jsonl")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--rng", type=int, default=11)
    ap.add_argument("--control-ratio", type=float, default=1.0)
    ap.add_argument("--out", default="tmp/probe_l_out")
    ap.add_argument("--deadline-min", type=float, default=1e9)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    pop = build_population(a.census, seed=a.rng, control_ratio=a.control_ratio)
    # SHUFFLED, deterministically. The population is stratified by step and tag, so a run that has
    # to stop early on a busy box must leave a RANDOM subsample behind, not "every early step and
    # no late one" — a partial sweep in trace order would be a different experiment.
    pop.sort(key=lambda r: (r["step"], r["base"], r["inv"], r["tag"]))
    random.Random(a.rng + 1).shuffle(pop)
    if a.dry_run:
        import collections
        print(json.dumps({
            "n": len(pop),
            "by_tag": dict(collections.Counter(r["tag"] for r in pop)),
            "by_kind": dict(collections.Counter(r["kind"] for r in pop if r["kind"])),
            "by_step": dict(collections.Counter(r["step"] for r in pop)),
            "reclicks": sum(1 for r in pop if r["reclick"]),
            "loop_steps": sum(1 for r in pop if r["loop_step"]),
        }, indent=2))
        return 0

    mine = [r for i, r in enumerate(pop) if i % a.shards == a.shard]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"shard{a.shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done.add((r["base"], r["inv"]))
                except ValueError:
                    pass

    from main.prober.session import ProbeSession
    from main.prober.lookahead import lookahead_decision
    from utils.bridge.reconstruction import ReconstructionRecord

    sessions: dict = {}
    t_start = time.time()
    n_ok = 0
    with open(out_path, "a", buffering=1) as fh:
        for k, r in enumerate(mine):
            if (r["base"], r["inv"]) in done:
                continue
            if (time.time() - t_start) / 60.0 > a.deadline_min:
                print(f"[shard{a.shard}] DEADLINE after {n_ok}", flush=True)
                break
            step = r["step"]
            if step not in sessions:
                traces = os.path.join(RUN, "eval_traces", step)
                sessions[step] = ProbeSession(
                    traces, ckpt_override=os.path.join(traces, "snapshot.zip"),
                    impl="rust", compile_extractor=False)
            sess = sessions[step]
            bid = r["base"] + "_summary.json"
            rec = dict(r)
            t0 = time.time()
            try:
                bt = sess._battle(bid)
                model, _ck = sess._model_for(bt)
                summary, npz = sess._summary(bt), sess._npz(bt)
                record = ReconstructionRecord.load(r["base"] + "_reconstruction.json")
                wrapped = WPRecorder(model)
                la = lookahead_decision(wrapped, record, summary, npz, int(r["inv"]),
                                        n_seeds=a.seeds, impl="rust")
                cand_order = sorted(int(c["action"]) for c in la["candidates"])
                # cand order is choice_map insertion order == ascending legal index
                per_action, verified = attach_wp(la["candidates"], wrapped.calls, cand_order)
                rec["wp_map_verified"] = bool(verified)
                rec["candidates"] = [
                    {"action": int(c["action"]), "label": c["label"], "choice": c["choice"],
                     "is_chosen": bool(c["is_chosen"]),
                     "win_prob_crn": c["win_prob_crn"], "value_crn": c["value_crn"],
                     "value_mean": c["value_mean"], "value_std": c["value_std"],
                     "n_evaluated": c["n_evaluated"], "terminal": c["terminal"],
                     "terminal_frac": c["terminal_frac"],
                     "wp_seeds": per_action.get(int(c["action"]))}
                    for c in la["candidates"]]
                rec["chosen_action"] = la["chosen"]["action"]
                rec["chosen_label"] = la["chosen"]["label"]
                rec["recorded_value"] = la["recorded_value"]
                rec["recorded_next_value"] = la["recorded_next_value"]
                # the policy's own recorded distribution at this decision (free, from the trace)
                inv = summary["invocations"][int(r["inv"])]
                rec["policy_probs"] = {kk: vv.get("prob") for kk, vv in
                                       (inv.get("actions") or {}).items() if isinstance(vv, dict)}
                rec["alpha"] = (inv.get("opp_intent") or {}).get("alpha")
                rec["beta"] = ((inv.get("opp_intent") or {}).get("beta") or [])[:3]
                with np.load(r["base"] + "_states.npz") as z:
                    rec["wp_s"] = float(z["win_probs"][int(r["inv"])])
                    rec["v_s"] = float(z["values"][int(r["inv"])])
                    rec["n_legal"] = int(np.asarray(z["action_mask"][int(r["inv"])]).sum())
            except Exception as e:                                     # noqa: BLE001
                rec["error"] = f"{type(e).__name__}:{e}"[:300]
            rec["secs"] = round(time.time() - t0, 2)
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            if n_ok % 25 == 0 or "error" in rec:
                print(f"[shard{a.shard}] {k+1}/{len(mine)} {r['tag']} {rec['secs']}s "
                      f"elapsed={(time.time()-t_start)/60:.1f}m err={rec.get('error','')[:80]}",
                      flush=True)
    for s in sessions.values():
        s.close()
    print(f"[shard{a.shard}] DONE {n_ok} in {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
