"""Extract the per-STATE table AND the frozen model's features for one substrate.

For every traced decision point of the named cycles this writes
  meta.npy      structured: cycle, opponent, opp_class, battle, y, turn, w, team,
                strength, true_wr, V_rec (the RECORDED win prob), V_fwd (re-forwarded)
  raw.npy       [N, 2501] float32   the observation vector as recorded
  pi.npy        [N, 512]  float32   the POLICY/shared trunk features  (extract_features[0])
  vf.npy        [N, 512]  float32   the VALUE-path features           (extract_features[1])
  pooled.npy    [N, 128]  float32   `stash.value_pooled` — the win head's LITERAL input

ONE frozen checkpoint forwards every state, so features, V and the trunk all come from the
same weights even where the state was traced at an earlier cycle. `V_rec` is kept beside
`V_fwd` precisely so the EXACT cycle (the one whose snapshot this is) can be used as a
faithfulness QC on all the others.

Why those four feature sets and not others — from `projection.py` and `extractor_forward.py`:
  * `extract_features` returns `(pi, vf)`, the two post-projection head inputs (512 each).
  * `vf_combined IS value_pooled` before the projection, and the win head reads
    `stash.value_pooled` directly: `V = sigmoid(win_prob_head(value_pooled))`. So `pooled` is
    ONE MLP away from V — the sharpest possible "the head has the information" probe.

CPU only, `torch.no_grad`, small batches, nothing written under models/. Selection: every
cycle must carry an `eval_manifest.json` at a KNOWN schema; a cycle without one is REFUSED
(SELECTION UNKNOWN), never silently pooled. Draws are excluded (no binary outcome) and
counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

BOTS = ("random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2")


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def bot_strengths(repo):
    return dict(_load_json(os.path.join(repo, "data", "gen3_bot_elo_anchors.json"))["ratings"])


def sentinel_strengths(run_dir, ladder_path):
    """(cycle_step) -> {sentinel_k: (snapshot_step, rating, eval_row_win_rate)}.

    The positional map `sentinel_k` -> the k-th `sentinels` entry of the cycle's
    `eval_results.jsonl` row is VERIFIED downstream against the manifest's own
    `battles_won`; a mismatch refuses the cell rather than guessing.
    """
    ratings = _load_json(ladder_path)["ratings"]
    out = {}
    with open(os.path.join(run_dir, "eval_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["step"])] = {
                f"sentinel_{k}": (int(s["step"]), ratings.get(str(s["step"])),
                                  float(s["win_rate"]))
                for k, s in enumerate(r.get("sentinels") or [])}
    return out


def team_id(summary):
    sp = sorted(p.get("species", "?") for p in (summary.get("teams") or {}).get("ours") or [])
    return hashlib.sha1("|".join(sp).encode()).hexdigest()[:10] if sp else "unknown"


# ---------------------------------------------------------------------------
# The frozen forward. ProbeModel does the loading (arch-drift diagnosis, the dropped-kwarg
# recovery, the ObservationDebugger silence, and the --value-true-team refusal at `_pin`);
# this only widens `_value_pooled_batch` to keep `pi` and the win-prob logit as well.
# ---------------------------------------------------------------------------
def forward_batch(model, obs, masks):
    import torch
    ex = model._policy.features_extractor
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    model._check_obs_dim(obs)
    ot = torch.as_tensor(obs)
    mt = torch.as_tensor(np.asarray(masks, dtype=np.float32))
    with torch.no_grad():
        pi, vf = model._policy.extract_features(model._pin(ot, mt))
        pooled = ex.stash.value_pooled
        wpl = ex.stash.win_prob_logits
        v = torch.sigmoid(wpl.reshape(-1)) if wpl is not None else None
    return (pi.numpy(), vf.numpy(), pooled.numpy(),
            None if v is None else v.numpy())


def run(run_dir, cycles, snapshot, repo, ladder_path, out_dir, batch=64):
    import torch
    torch.set_num_threads(2)
    from agents.training.trace_selection import KNOWN_SELECTION_SCHEMAS
    from main.prober.model import ProbeModel

    os.makedirs(out_dir, exist_ok=True)
    model = ProbeModel.load(snapshot, device="cpu")
    bot_elo = bot_strengths(repo)
    sent = sentinel_strengths(run_dir, ladder_path)

    meta_rows, obs_chunks = [], []
    info = {"run": run_dir, "snapshot": snapshot, "cycles": {}, "refusals": [],
            "dropped_kwargs": list(model.dropped_kwargs), "n_draw_excluded": 0}

    for cyc in cycles:
        cdir = os.path.join(run_dir, "eval_traces", cyc)
        mpath = os.path.join(cdir, "eval_manifest.json")
        if not os.path.exists(mpath):
            info["refusals"].append(f"{cyc}: NO MANIFEST -> SELECTION UNKNOWN")
            continue
        man = _load_json(mpath)
        if man.get("selection_schema") not in KNOWN_SELECTION_SCHEMAS:
            info["refusals"].append(f"{cyc}: selection schema {man.get('selection_schema')!r} unknown")
            continue
        step = int(man["step"])
        sel = (man.get("selection") or {}).get("opponents") or {}
        if not sel:
            info["refusals"].append(f"{cyc}: no selection block")
            continue
        cyc_sent = sent.get(step, {})
        cmeta = {"step": step, "opponents": {}, "sentinel_map_verified": {}}

        for opp in man["opponents"]:
            odir = os.path.join(cdir, opp)
            s = sel.get(opp)
            if s is None:
                info["refusals"].append(f"{cyc}/{opp}: no selection row")
                continue
            played, won = int(s["battles_played"]), int(s["battles_won"])
            true_wr = won / played
            crw, crl = s.get("capture_rate_win"), s.get("capture_rate_loss")
            if opp in bot_elo:
                strength, klass, snap_step = bot_elo[opp], "bot", None
            else:
                got = cyc_sent.get(opp)
                if got is None:
                    info["refusals"].append(f"{cyc}/{opp}: unmapped sentinel")
                    continue
                snap_step, rating, ev_wr = got
                ok = abs(ev_wr - true_wr) < 1e-9
                cmeta["sentinel_map_verified"][opp] = {
                    "snapshot_step": snap_step, "eval_row_wr": ev_wr, "manifest_wr": true_wr,
                    "match": bool(ok), "ladder_rating": rating}
                if not ok:
                    info["refusals"].append(f"{cyc}/{opp}: sentinel map MISMATCH")
                    continue
                if rating is None:
                    info["refusals"].append(f"{cyc}/{opp}: snapshot {snap_step} unrated")
                    continue
                strength, klass = rating, "sentinel"

            n_b = 0
            for fn in sorted(os.listdir(odir)):
                if not fn.endswith("_states.npz"):
                    continue
                base = fn[: -len("_states.npz")]
                spath = os.path.join(odir, base + "_summary.json")
                if not os.path.exists(spath):
                    continue
                summ = _load_json(spath)
                res = (summ.get("meta") or {}).get("result")
                if res == "DRAW":
                    info["n_draw_excluded"] += 1
                    continue
                if res not in ("WIN", "LOSS"):
                    continue
                y = 1.0 if res == "WIN" else 0.0
                cr = crw if y == 1.0 else crl
                if not cr:                       # 0 or None: that outcome class was never played
                    continue
                z = np.load(os.path.join(odir, fn))
                invs = summ.get("invocations") or []
                V, hs, om = z["obs"], z["has_state"], z["action_mask"]
                if len(invs) != len(z["values"]):
                    info["refusals"].append(f"{cyc}/{opp}/{base}: npz/inv length mismatch")
                    continue
                turns = np.array([int(i.get("turn", -1)) for i in invs])
                keep = (hs == 1) & (turns > 0)
                if not keep.any():
                    continue
                tid = team_id(summ)
                obs_chunks.append((V[keep], om[keep]))
                for t, vr in zip(turns[keep].tolist(), np.asarray(z["win_probs"])[keep].tolist()):
                    meta_rows.append((step, opp, klass, f"{step}|{opp}|{base}", y, t,
                                      1.0 / float(cr), tid, float(strength), true_wr,
                                      float(vr), np.nan))
                n_b += 1
            cmeta["opponents"][opp] = {
                "class": klass, "strength": float(strength), "snapshot_step": snap_step,
                "battles_played": played, "battles_won": won, "true_win_rate": true_wr,
                "capture_rate_win": crw, "capture_rate_loss": crl,
                "traces_written": int(s["traces_written"]), "battles_loaded": n_b}
        info["cycles"][cyc] = cmeta

    dt = np.dtype([("cycle", "i8"), ("opponent", "U24"), ("opp_class", "U10"),
                   ("battle", "U80"), ("y", "f8"), ("turn", "i8"), ("w", "f8"),
                   ("team", "U12"), ("strength", "f8"), ("true_wr", "f8"),
                   ("V_rec", "f8"), ("V_fwd", "f8")])
    meta = np.array(meta_rows, dtype=dt)
    obs = np.concatenate([c[0] for c in obs_chunks], axis=0)
    masks = np.concatenate([c[1] for c in obs_chunks], axis=0)
    assert len(obs) == len(meta), (len(obs), len(meta))

    n = len(meta)
    pi = np.empty((n, 512), np.float32)
    vf = np.empty((n, 512), np.float32)
    pooled = np.empty((n, 128), np.float32)
    vfwd = np.empty(n, np.float64)
    for i in range(0, n, batch):
        j = min(n, i + batch)
        p, v, po, vv = forward_batch(model, obs[i:j], masks[i:j])
        if p.shape[1] != pi.shape[1] or v.shape[1] != vf.shape[1] or po.shape[1] != pooled.shape[1]:
            raise SystemExit(f"REFUSED: unexpected feature widths pi={p.shape} vf={v.shape} "
                             f"pooled={po.shape} — the probe's feature-set map is stale.")
        pi[i:j], vf[i:j], pooled[i:j] = p, v, po
        if vv is None:
            raise SystemExit("REFUSED: this checkpoint stashes no win_prob_logits — V is not the "
                             "win-prob head here, so the 'does V rank opponents' decoder would be "
                             "measuring a different quantity.")
        vfwd[i:j] = vv
        if (i // batch) % 50 == 0:
            print(f"  forward {j}/{n}", flush=True)
    meta["V_fwd"] = vfwd

    # QC — on the EXACT cycle (the snapshot's own step) the re-forward must reproduce the
    # recorded win prob. Anywhere else it legitimately differs (different weights).
    exact_step = None
    for cyc, cm in info["cycles"].items():
        if os.path.realpath(os.path.join(run_dir, "eval_traces", cyc, "snapshot.zip")) == \
                os.path.realpath(snapshot):
            exact_step = cm["step"]
    info["exact_cycle_step"] = exact_step
    if exact_step is not None:
        m = meta["cycle"] == exact_step
        info["qc_exact_max_abs_Vfwd_minus_Vrec"] = float(
            np.max(np.abs(meta["V_fwd"][m] - meta["V_rec"][m]))) if m.any() else None
    info["qc_offcycle_mean_abs_Vfwd_minus_Vrec"] = float(
        np.mean(np.abs(meta["V_fwd"] - meta["V_rec"])))

    np.save(os.path.join(out_dir, "meta.npy"), meta)
    np.save(os.path.join(out_dir, "raw.npy"), obs)
    np.save(os.path.join(out_dir, "pi.npy"), pi)
    np.save(os.path.join(out_dir, "vf.npy"), vf)
    np.save(os.path.join(out_dir, "pooled.npy"), pooled)
    info["n_states"] = int(n)
    info["n_battles"] = int(len(set(meta["battle"].tolist())))
    info["n_teams"] = int(len(set(meta["team"].tolist())))
    with open(os.path.join(out_dir, "extract_meta.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--cycles", required=True, help="comma-separated step_ dir names")
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default="/home/goodlad/dev/gen3ai")
    a = ap.parse_args()
    info = run(a.run, a.cycles.split(","), a.snapshot, a.repo, a.ladder, a.out_dir)
    print(f"states={info['n_states']} battles={info['n_battles']} teams={info['n_teams']} "
          f"draws_excluded={info['n_draw_excluded']}")
    print(f"QC exact max|V_fwd-V_rec| = {info.get('qc_exact_max_abs_Vfwd_minus_Vrec')} "
          f"(off-cycle mean {info['qc_offcycle_mean_abs_Vfwd_minus_Vrec']:.4f})")
    for r in info["refusals"]:
        print("REFUSED:", r, file=sys.stderr)
