"""Extract ONE STATIONARY per-state table + the frozen `value_pooled` for one substrate.

The N-curve needs the opposite of what the head refit had: not four cycles of a MOVING policy
pooled together, but MANY battles from ONE frozen policy. `eval_trace_gen` was run three times
against the SAME checkpoint (`step_10000032`) with three different seeds, so the three trees are
independent draws from one stationary distribution — the snapshot.zip files are byte-identical
(md5 asserted below) and every cycle's manifest names the same `checkpoint_sha`.

  cycle_400  seed 20260909  400 games x 12 opponents
  cycle_800  seed 20260910  800 games x 12 opponents
  cycle_800b seed 20260911  800 games x 12 opponents        => ~24,000 battles, ~700k states

This writes
  meta.npy     structured: cycle, opponent, opp_class, battle, y, turn, w, team, strength,
               true_wr, n_games, V_rec, V_fwd
  pooled.npy   [N, 128] float32 — `stash.value_pooled`, the win head's LITERAL input

and NOTHING else: `raw`/`pi`/`vf` are not needed here (the probe read already decided the
representation question) and at 700k states they would be 8 GB.

🚨 THE THREE TREES MUST BE ONE POLICY OR THE MEASUREMENT IS NOT ABOUT STATIONARITY. The md5 of
every `snapshot.zip` is compared and a mismatch REFUSES the substrate; the per-cycle manifest
`checkpoint_sha` is recorded beside it. The per-cycle overall win rate is recorded too, because
"the same weights" and "the same difficulty" are different claims and the reader needs both.

HT weights: every cell of every cycle is FULL CAPTURE (`capture_rate_win`/`_loss` == 1.0, or None
where that outcome class was never played — `random` is never lost to), so every weight is 1.0.
The IPW machinery is kept rather than deleted so the pipeline is the head refit's, and the fact
that the weights are all 1.0 is an assertion here, not an assumption.

CPU only, torch with `--threads` threads, small batches, nothing written under models/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

BOTS = ("random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2")


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def _md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def team_id(summary):
    sp = sorted(p.get("species", "?") for p in (summary.get("teams") or {}).get("ours") or [])
    return hashlib.sha1("|".join(sp).encode()).hexdigest()[:10] if sp else "unknown"


def forward_batch(model, obs, masks):
    import torch
    ex = model._policy.features_extractor
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    model._check_obs_dim(obs)
    ot = torch.as_tensor(obs)
    mt = torch.as_tensor(np.asarray(masks, dtype=np.float32))
    with torch.no_grad():
        model._policy.extract_features(model._pin(ot, mt))
        pooled = ex.stash.value_pooled
        wpl = ex.stash.win_prob_logits
        if wpl is None:
            raise SystemExit("REFUSED: this checkpoint stashes no win_prob_logits — V is not the "
                             "win-prob head here.")
        v = torch.sigmoid(wpl.reshape(-1))
    return pooled.numpy(), v.numpy()


def run(trees, snapshot, out_dir, threads=2, batch=256, flush_every=30_000):
    import torch
    torch.set_num_threads(threads)
    from agents.training.trace_selection import KNOWN_SELECTION_SCHEMAS
    from main.prober.model import ProbeModel

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── stationarity gate ────────────────────────────────────────────────────
    snap_md5 = _md5(snapshot)
    info = {"snapshot": snapshot, "snapshot_md5": snap_md5, "trees": {}, "refusals": [],
            "n_draw_excluded": 0, "capture_rates_seen": []}
    for tag, cdir in trees.items():
        s = os.path.join(cdir, "snapshot.zip")
        m = _md5(s)
        if m != snap_md5:
            raise SystemExit(f"REFUSED: {tag}'s snapshot.zip md5 {m} != {snap_md5}. The three "
                             f"trees are NOT one frozen policy and the dataset is not stationary.")

    model = ProbeModel.load(snapshot, device="cpu")
    info["dropped_kwargs"] = list(model.dropped_kwargs)

    meta_rows = []
    pooled_parts, vfwd_parts = [], []
    pend_obs, pend_mask, n_pend = [], [], 0

    def _flush():
        nonlocal pend_obs, pend_mask, n_pend
        if not n_pend:
            return
        obs = np.concatenate(pend_obs, axis=0)
        mk = np.concatenate(pend_mask, axis=0)
        po = np.empty((len(obs), 128), np.float32)
        vv = np.empty(len(obs), np.float64)
        for i in range(0, len(obs), batch):
            j = min(len(obs), i + batch)
            p, v = forward_batch(model, obs[i:j], mk[i:j])
            if p.shape[1] != 128:
                raise SystemExit(f"REFUSED: value_pooled width {p.shape[1]} != 128 — the head "
                                 f"refit's feature map is stale.")
            po[i:j], vv[i:j] = p, v
        pooled_parts.append(po)
        vfwd_parts.append(vv)
        pend_obs, pend_mask, n_pend = [], [], 0
        print(f"  forwarded {sum(len(x) for x in pooled_parts)} states "
              f"({time.time() - t0:.0f}s)", flush=True)

    for tag, cdir in trees.items():
        man = _load_json(os.path.join(cdir, "eval_manifest.json"))
        if man.get("selection_schema") not in KNOWN_SELECTION_SCHEMAS:
            raise SystemExit(f"REFUSED: {tag} selection schema {man.get('selection_schema')!r} "
                             f"unknown -> SELECTION UNKNOWN.")
        sel = (man.get("selection") or {}).get("opponents") or {}
        if not sel:
            raise SystemExit(f"REFUSED: {tag} has no selection block.")
        g = man.get("generated_by") or {}
        tinfo = {"step": int(man["step"]), "seed": g.get("seed"),
                 "checkpoint_sha": g.get("checkpoint_sha"), "n_games": man.get("n_games"),
                 "sentinel_steps": g.get("sentinel_steps"), "opponents": {}}
        cyc_code = int(_load_json(os.path.join(cdir, "eval_manifest.json"))["step"])
        # the CYCLE code distinguishes the three trees; they share a step, so the seed is the key
        cyc_code = int(g.get("seed"))

        for opp in man["opponents"]:
            odir = os.path.join(cdir, opp)
            s = sel.get(opp)
            if s is None:
                info["refusals"].append(f"{tag}/{opp}: no selection row")
                continue
            played, won = int(s["battles_played"]), int(s["battles_won"])
            drawn = int(s.get("battles_drawn", 0))
            true_wr = won / played
            crw, crl = s.get("capture_rate_win"), s.get("capture_rate_loss")
            for cr in (crw, crl):
                if cr is not None and abs(cr - 1.0) > 1e-9:
                    info["capture_rates_seen"].append([tag, opp, cr])
            klass = "bot" if opp in BOTS else "sentinel"
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
                if not cr:
                    continue
                z = np.load(os.path.join(odir, fn))
                invs = summ.get("invocations") or []
                if len(invs) != len(z["values"]):
                    info["refusals"].append(f"{tag}/{opp}/{base}: npz/inv length mismatch")
                    continue
                turns = np.array([int(i.get("turn", -1)) for i in invs])
                keep = (z["has_state"] == 1) & (turns > 0)
                if not keep.any():
                    continue
                tid = team_id(summ)
                pend_obs.append(z["obs"][keep])
                pend_mask.append(z["action_mask"][keep])
                n_pend += int(keep.sum())
                for t, vr in zip(turns[keep].tolist(),
                                 np.asarray(z["win_probs"])[keep].tolist()):
                    meta_rows.append((cyc_code, opp, klass, f"{tag}|{opp}|{base}", y, t,
                                      1.0 / float(cr), tid, float(played - drawn), true_wr,
                                      float(played), float(vr), np.nan))
                n_b += 1
                if n_pend >= flush_every:
                    _flush()
            tinfo["opponents"][opp] = {
                "class": klass, "battles_played": played, "battles_won": won,
                "battles_drawn": drawn, "true_win_rate": true_wr,
                "capture_rate_win": crw, "capture_rate_loss": crl, "battles_loaded": n_b}
        tinfo["overall_true_wr"] = (sum(v["battles_won"] for v in tinfo["opponents"].values())
                                    / sum(v["battles_played"] for v in tinfo["opponents"].values()))
        info["trees"][tag] = tinfo
        print(f"[{tag}] done, {len(meta_rows)} states so far ({time.time() - t0:.0f}s)", flush=True)
    _flush()

    dt = np.dtype([("cycle", "i8"), ("opponent", "U24"), ("opp_class", "U10"),
                   ("battle", "U90"), ("y", "f8"), ("turn", "i8"), ("w", "f8"),
                   ("team", "U12"), ("strength", "f8"), ("true_wr", "f8"),
                   ("n_games", "f8"), ("V_rec", "f8"), ("V_fwd", "f8")])
    meta = np.array(meta_rows, dtype=dt)
    pooled = np.concatenate(pooled_parts, axis=0)
    vfwd = np.concatenate(vfwd_parts, axis=0)
    if len(pooled) != len(meta):
        raise SystemExit(f"REFUSED: {len(pooled)} forwarded states != {len(meta)} meta rows.")
    meta["V_fwd"] = vfwd

    # ── IPW gate: full capture means every weight is exactly 1.0. Assert, do not assume.
    if not np.allclose(meta["w"], 1.0):
        raise SystemExit("REFUSED: capture rates are not all 1.0 — the HT weights are not "
                         "trivial and every claim about 'full capture' in the README is false.")

    # ── QC: the frozen re-forward must reproduce the RECORDED win prob. Here it is the EXACT
    # cycle for all three trees (one snapshot, three seeds), so this is a hard check, not the
    # head refit's off-cycle probe.
    info["qc_max_abs_Vfwd_minus_Vrec"] = float(np.max(np.abs(meta["V_fwd"] - meta["V_rec"])))
    if info["qc_max_abs_Vfwd_minus_Vrec"] > 1e-3:
        raise SystemExit(f"REFUSED: frozen forward does not reproduce the recorded win prob "
                         f"(max |d| = {info['qc_max_abs_Vfwd_minus_Vrec']}).")

    np.save(os.path.join(out_dir, "meta.npy"), meta)
    np.save(os.path.join(out_dir, "pooled.npy"), pooled)
    info["n_states"] = int(len(meta))
    info["n_battles"] = int(len(set(meta["battle"].tolist())))
    info["n_teams"] = int(len(set(meta["team"].tolist())))
    info["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "extract_meta.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append", required=True,
                    help="tag=/path/to/eval_traces/step_XXXX (repeat)")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    trees = dict(t.split("=", 1) for t in a.tree)
    info = run(trees, a.snapshot, a.out_dir, a.threads, a.batch)
    print(f"states={info['n_states']} battles={info['n_battles']} teams={info['n_teams']} "
          f"draws_excluded={info['n_draw_excluded']} "
          f"QC max|Vfwd-Vrec|={info['qc_max_abs_Vfwd_minus_Vrec']:.2e} "
          f"({info['seconds']:.0f}s)")
    for r in info["refusals"]:
        print("REFUSED:", r, file=sys.stderr)
