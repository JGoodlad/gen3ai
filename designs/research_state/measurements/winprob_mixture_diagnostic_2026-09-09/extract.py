"""Extract a tidy per-STATE table from arm A's last 4 eval trace cycles.

One row per traced decision point:
  cycle, opponent, opp_class (bot|sentinel), battle, y (battle outcome), turn,
  V (the critic's win prob at that state), w (inverse capture-rate weight),
  team (trainee team id), strength (opponent Elo on the bot-anchored scale).

Selection: the trace quota PREFERS LOSSES (manifest selection_schema 1). Every
statistic downstream is Horvitz-Thompson reweighted by capture_rate_win /
capture_rate_loss. A cycle without a manifest would be SELECTION UNKNOWN and is
refused rather than silently pooled.

Read-only over models/. Nothing is written under models/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

CYCLES = ["step_50000016", "step_60000000", "step_70000032", "step_74000016"]
BOTS = ["random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"]


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def bot_strengths(repo_root):
    d = _load_json(os.path.join(repo_root, "data", "gen3_bot_elo_anchors.json"))
    return dict(d["ratings"])


def sentinel_strengths(run_dir, ladder_path=None):
    """(cycle_step) -> {sentinel_k: (snapshot_step, rating, true_wr_from_eval_row)}.

    The positional map sentinel_k -> eval_results.jsonl's k-th `sentinels` entry is
    VERIFIED downstream against the manifest's own battles_won for that opponent;
    a mismatch refuses the cycle rather than guessing.
    """
    # DEFAULT is the ALL-STEPS refit (refit_ladder.py), not the committed ladder.json: the
    # committed file rates only 36M-74M (pool grooming deleted the early snapshots) and would
    # leave 7 of this read's 20 sentinel cells unrated, AND it folded the eval cycles'
    # greedy-vs-stochastic sentinel edges, which reads high on the newest nodes.
    lp = ladder_path or os.path.join(run_dir, "snapshot_ladder", "ladder.json")
    ladder = _load_json(lp)["ratings"]
    out = {}
    with open(os.path.join(run_dir, "eval_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            cyc = {}
            for k, s in enumerate(r.get("sentinels") or []):
                step = str(s["step"])
                cyc[f"sentinel_{k}"] = (int(s["step"]),
                                        ladder.get(step),
                                        float(s["win_rate"]))
            out[int(r["step"])] = cyc
    return out


def team_id(summary):
    sp = sorted(p.get("species", "?") for p in (summary.get("teams") or {}).get("ours") or [])
    if not sp:
        return "unknown"
    return hashlib.sha1("|".join(sp).encode()).hexdigest()[:10]


def extract(run_dir, repo_root, out_path, ladder_path=None):
    bot_elo = bot_strengths(repo_root)
    sent = sentinel_strengths(run_dir, ladder_path)
    rows = []
    meta = {"cycles": {}, "refusals": []}

    for cyc_name in CYCLES:
        cdir = os.path.join(run_dir, "eval_traces", cyc_name)
        mpath = os.path.join(cdir, "eval_manifest.json")
        if not os.path.exists(mpath):
            meta["refusals"].append(f"{cyc_name}: NO MANIFEST -> SELECTION UNKNOWN")
            continue
        man = _load_json(mpath)
        step = int(man["step"])
        sel = (man.get("selection") or {}).get("opponents") or {}
        if not sel or man.get("selection_schema") != 1:
            meta["refusals"].append(f"{cyc_name}: selection block absent/unknown schema")
            continue
        cyc_sent = sent.get(step, {})
        cyc_meta = {"step": step, "opponents": {}, "sentinel_map_verified": {}}

        for opp in man["opponents"]:
            odir = os.path.join(cdir, opp)
            s = sel.get(opp)
            if s is None:
                meta["refusals"].append(f"{cyc_name}/{opp}: no selection row")
                continue
            played, won = int(s["battles_played"]), int(s["battles_won"])
            true_wr = won / played
            crw, crl = s.get("capture_rate_win"), s.get("capture_rate_loss")

            if opp in bot_elo:
                strength, opp_class, snap_step = bot_elo[opp], "bot", None
            else:
                info = cyc_sent.get(opp)
                if info is None:
                    meta["refusals"].append(f"{cyc_name}/{opp}: unmapped sentinel")
                    continue
                snap_step, rating, ev_wr = info
                # VERIFY the positional map against the manifest's own win count.
                ok = abs(ev_wr - true_wr) < 1e-9
                cyc_meta["sentinel_map_verified"][opp] = {
                    "snapshot_step": snap_step, "eval_row_wr": ev_wr,
                    "manifest_wr": true_wr, "match": bool(ok), "ladder_rating": rating}
                if not ok:
                    meta["refusals"].append(
                        f"{cyc_name}/{opp}: sentinel map MISMATCH {ev_wr} vs {true_wr}")
                    continue
                if rating is None:
                    meta["refusals"].append(f"{cyc_name}/{opp}: snapshot {snap_step} not in ladder")
                    continue
                strength, opp_class = rating, "sentinel"

            n_b = 0
            for fn in sorted(os.listdir(odir)):
                if not fn.endswith("_states.npz"):
                    continue
                base = fn[: -len("_states.npz")]
                sp = os.path.join(odir, base + "_summary.json")
                if not os.path.exists(sp):
                    continue
                summ = _load_json(sp)
                res = (summ.get("meta") or {}).get("result")
                if res not in ("WIN", "LOSS"):
                    continue
                y = 1.0 if res == "WIN" else 0.0
                cr = crw if y == 1.0 else crl
                if not cr:            # 0 or None -> this outcome class was never played
                    continue
                w = 1.0 / float(cr)
                z = np.load(os.path.join(odir, fn))
                V, wp, hs = z["values"], z["win_probs"], z["has_state"]
                invs = summ.get("invocations") or []
                if len(invs) != len(V):
                    meta["refusals"].append(f"{cyc_name}/{opp}/{base}: npz/inv length mismatch")
                    continue
                vmax = float(np.max(np.abs(V - wp))) if len(V) else 0.0
                turns = np.array([int(i.get("turn", -1)) for i in invs])
                keep = (hs == 1) & (turns > 0)
                tid = team_id(summ)
                for v, t in zip(np.asarray(wp)[keep].tolist(), turns[keep].tolist()):
                    rows.append((step, opp, opp_class, base, y, t, float(v), w, tid,
                                 float(strength), true_wr, vmax))
                n_b += 1
            cyc_meta["opponents"][opp] = {
                "class": opp_class, "strength": float(strength), "snapshot_step": snap_step,
                "battles_played": played, "battles_won": won, "true_win_rate": true_wr,
                "capture_rate_win": crw, "capture_rate_loss": crl,
                "traces_written": int(s["traces_written"]), "traces_won": int(s["traces_won"]),
                "battles_loaded": n_b}
        meta["cycles"][cyc_name] = cyc_meta

    dt = np.dtype([("cycle", "i8"), ("opponent", "U24"), ("opp_class", "U10"),
                   ("battle", "U48"), ("y", "f8"), ("turn", "i8"), ("V", "f8"),
                   ("w", "f8"), ("team", "U12"), ("strength", "f8"),
                   ("true_wr", "f8"), ("vmax", "f8")])
    arr = np.array(rows, dtype=dt)
    np.save(out_path, arr)
    meta["sentinel_ladder"] = ladder_path or "COMMITTED ladder.json"
    meta["n_states"] = int(arr.size)
    meta["n_battles"] = int(len(set(zip(arr["cycle"].tolist(), arr["opponent"].tolist(),
                                        arr["battle"].tolist()))))
    meta["max_abs_values_minus_winprobs"] = float(arr["vmax"].max()) if arr.size else None
    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return arr, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic")
    ap.add_argument("--repo", default="/home/goodlad/dev/gen3ai")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", default=None,
                    help="ratings json for the SENTINEL strength axis (default: committed "
                         "ladder.json; pass refit_ladder.py's output for full coverage)")
    a = ap.parse_args()
    arr, meta = extract(a.run, a.repo, a.out, a.ladder)
    print(f"states={meta['n_states']} battles={meta['n_battles']} "
          f"max|values-win_probs|={meta['max_abs_values_minus_winprobs']}")
    for r in meta["refusals"]:
        print("REFUSED:", r, file=sys.stderr)
    for c, cm in meta["cycles"].items():
        bad = [k for k, v in cm["sentinel_map_verified"].items() if not v["match"]]
        print(f"{c}: opponents={len(cm['opponents'])} sentinel_map_mismatches={bad}")
