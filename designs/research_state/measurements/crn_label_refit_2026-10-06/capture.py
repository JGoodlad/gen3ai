"""One-ply successor capture (README §2): for every turn of the subsets, every legal action x the
first 8 of truth's dice seeds, advanced to the banked side's NEXT decision with ``qhat.one_ply``
(the opponent answered by the blob checkpoint's greedy choice: truth's construction). For each
captured successor it stores the frozen trunk's ``value_pooled`` (128) and BASE's V = sigmoid(win
logit); a branch that ended first stores its terminal value.

Durable + resumable: per chunk of turns ONE ``cap/<chunk>.npz`` (written to a temp name, fsync'd,
renamed) and then one fsync'd index row per turn in ``cap/index.jsonl`` naming that file. A turn
counts as done only when its index row is on disk and its file exists.

    python capture.py --subset rows/subset_train.json --subset rows/subset_held.json \
        --workers 2 --threads 3
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CKPT = Path("/home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1001/final_model.zip")
N_SEEDS = 8
ACT = 11
D = 128


def make_forward(model):
    """rows, masks -> [N, 11 + 1 + 128]: policy logits, V = sigmoid(win logit), value_pooled. ONE
    extractor pass; the caller serializes calls (the extractor keeps per-forward state)."""
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs

    pol = model.policy
    fe = pol.features_extractor
    dev = next(pol.parameters()).device

    def fwd(rows: np.ndarray, masks: np.ndarray, batch: int = 1024) -> np.ndarray:
        out = []
        for i in range(0, len(rows), batch):
            mb = th.tensor(masks[i:i + batch].astype(np.float32), device=dev)
            ob = {"observation": th.tensor(rows[i:i + batch], device=dev), "action_mask": mb}
            ob.update(zero_extra_obs(fe, batch=len(mb), device=dev))
            with th.no_grad():
                pi_f, vf_f = pol.extract_features(ob)
                lat_pi = pol.mlp_extractor.forward_actor(pi_f)
                lat_vf = pol.mlp_extractor.forward_critic(vf_f)
                v = pol._critic_value(lat_vf).reshape(-1, 1)
                logits = pol._get_action_dist_from_latent(lat_pi).distribution.logits
                vp = fe.stash.value_pooled
                # V must be exactly sigmoid(win_head(value_pooled)): the refit's head reads the same tap
                v2 = th.sigmoid(fe.win_head(vp)).reshape(-1, 1)
                if not th.allclose(v, v2, atol=1e-6, rtol=0):
                    raise RuntimeError(f"V != sigmoid(win_head(value_pooled)): max |d| {(v - v2).abs().max().item()}")
            out.append(th.cat([logits, v, vp], dim=1).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    return fwd


def capture_turn(bank, did, fwd, core):
    from main.policy_spectrum.qhat import QhatError, greedy_of, one_ply
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env.successors import SuccessorsError

    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    if b.commands[at][1] != d["played"]:
        raise QhatError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")

    def greedy_answer(rows, masks, j):
        return greedy_of(fwd(rows, masks)[:, :ACT + 1], masks)

    base = {"id": did, "battle": d["battle"], "side": side, "at": at, "n_seeds": N_SEEDS}
    try:
        r = one_ply(core, to_log(b), at, side, turn_seeds(did, N_SEEDS), greedy_answer)
    except (SuccessorsError, QhatError) as exc:
        return dict(base, ok=False, error=str(exc)[:500]), None
    want = {str(k): v for k, v in sorted((int(a), t) for a, t in d["tokens"].items())}
    if r["tokens"] != want:
        return dict(base, ok=False, error=f"root tokens {r['tokens']} != banked"), None
    from main.policy_spectrum.qhat import end_value

    idx = [i for i, br in enumerate(r["branches"]) if br["row"] is not None]
    feats = np.zeros((0, ACT + 1 + D), dtype=np.float32)
    if idx:
        feats = fwd(np.stack([r["branches"][i]["row"] for i in idx]),
                    np.stack([r["branches"][i]["mask"] for i in idx]).astype(bool))
    pos = {i: k for k, i in enumerate(idx)}
    acts: dict = {}
    for i, br in enumerate(r["branches"]):
        a = acts.setdefault(str(br["action"]), {"seed": [], "term": [], "fi": [], "v": []})
        a["seed"].append(int(br["seed"]))
        if br["row"] is None:
            a["term"].append(end_value(br["end"], side))
            a["fi"].append(-1)
            a["v"].append(None)
        else:
            a["term"].append(None)
            a["fi"].append(pos[i])
            a["v"].append(round(float(feats[pos[i], ACT]), 7))
    for k, a in acts.items():
        if a["seed"] != list(range(N_SEEDS)):
            return dict(base, ok=False, error=f"action {k}: seeds {a['seed']}"), None
        del a["seed"]
    return dict(base, ok=True, other_open=r["other_open"], actions=acts), feats[:, ACT + 1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", action="append", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default=str(HERE / "rows" / "cap"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import inference_globals, load_checkpoint
    from main.policy_spectrum.truth import MicroBatcher
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    bank = load_bank(HERE.parent / "m5_laneS" / "bank_v1")
    ids = []
    for s in a.subset:
        sub = json.loads(Path(s).read_text())
        if sub["bank"] != bank.manifest["content_sha256"]:
            raise SystemExit(f"{s}: drawn from a different bank")
        ids += sub["ids"]
    if a.limit:
        ids = ids[: a.limit]
    done = set()
    if index.exists():
        for line in index.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("file") is None or (out / r["file"]).exists():
                    done.add(r["id"])
    todo = [i for i in ids if i not in done]
    print(f"[capture] {len(done)} turns done, {len(todo)} to go", flush=True)
    lib = F.load(F.default_path("release"))
    local = threading.local()
    cores = []
    lk = threading.Lock()
    t0 = time.monotonic()
    with inference_globals(a.threads):
        model = load_checkpoint(CKPT)
        check_winprob(model)
        fwd = MicroBatcher(make_forward(model))

        def one(did):
            core = getattr(local, "core", None)
            if core is None:
                core = local.core = SearchCore(lib=lib)
                with lk:
                    cores.append(core)
            return capture_turn(bank, did, fwd, core)

        try:
            with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex, open(index, "a") as fi:
                for c0 in range(0, len(todo), a.chunk):
                    part = todo[c0:c0 + a.chunk]
                    res = list(ex.map(one, part))
                    blocks, off = [], 0
                    name = f"c{int(time.time() * 1000)}_{os.getpid()}.npz"
                    for row, f in res:
                        if row["ok"]:
                            row["file"] = name
                            row["off"] = off
                            off += len(f)
                            blocks.append(f)
                        else:
                            row["file"] = None
                            print(f"[capture] REFUSED {row['id']}: {row['error'][:200]}", flush=True)
                    if blocks:
                        tmp = out / (name + ".tmp")
                        with open(tmp, "wb") as fh:
                            np.savez_compressed(fh, feats=np.concatenate(blocks).astype(np.float32))
                            fh.flush()
                            os.fsync(fh.fileno())
                        os.replace(tmp, out / name)
                    for row, _ in res:
                        fi.write(json.dumps(row, sort_keys=True) + "\n")
                    fi.flush()
                    os.fsync(fi.fileno())
                    el = time.monotonic() - t0
                    n = c0 + len(part)
                    print(f"[capture] {n}/{len(todo)} turns, {el / 60:.1f} min, "
                          f"eta {el / n * (len(todo) - n) / 60:.1f} min", flush=True)
        finally:
            for c in cores:
                c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
