"""The fits and the scorers' per-action values (README §2).

Reads the capture (``rows/cap/``), the truth rows (TRAIN at S = 8, HELD-OUT at S = 32) and writes
``rows/scores.npz`` + ``rows/fit_report.json``: for every HELD-OUT action, BASE, REFIT-K{1,4,8}
(fit seed 0, and seeds 1-2 for the spread), LEAF-K{1,4,8} and its truth outcomes (seeds 8-31).

    python refit.py [--threads 3]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows"
CKPT = Path("/home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1001/final_model.zip")
KS = (1, 4, 8)
FIT_SEEDS = (0, 1, 2)
N_FEAT = 8
LR, WD, BS, EPOCHS, PATIENCE = 1e-3, 1e-4, 256, 300, 20
SAL = "crn_label_refit_2026-10-06"


def outcomes_of(row: dict) -> dict:
    """{action: np.array of per-seed outcomes} for a truth row (full or compact form)."""
    out = {}
    for a, o in row["outcomes"].items():
        if isinstance(o, str):
            o = [{"+": 1.0, "-": -1.0, "0": 0.0}[c] for c in o]
        out[a] = np.asarray(o, dtype=np.float64)
    return out


def load_truth(path: Path, s: int) -> dict:
    """The full JSONL rows (the run's own output), else the committed compact ``.compact.jsonl.gz``."""
    from main.policy_spectrum.truth import load_rows

    if not path.exists():
        path = path.with_name(path.name.replace(".jsonl", ".compact.jsonl.gz"))
    return {r["id"]: r for r in load_rows(path) if len(r.get("seeds", [])) == s}


def load_capture(cap: Path):
    idx = {}
    for line in (cap / "index.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            idx[r["id"]] = r
    files = {}
    for r in idx.values():
        if r.get("file") and r["file"] not in files:
            files[r["file"]] = np.load(cap / r["file"])["feats"]
    return idx, files


def build(ids, idx, files, truth, split_of):
    """One record per (turn, action) with capture AND truth: F [8, 128] (zeros where ended), E [8]
    (1 = ended), T [8] terminal value, V [8] BASE's V, O [S] outcomes."""
    recs = []
    for did in ids:
        c, t = idx.get(did), truth.get(did)
        if c is None or t is None or not c.get("ok") or not t.get("ok"):
            continue
        feats = files[c["file"]]
        outs = outcomes_of(t)
        if set(outs) != set(c["actions"]):
            raise RuntimeError(f"{did}: truth actions {sorted(outs)} != capture {sorted(c['actions'])}")
        for a, ca in c["actions"].items():
            F = np.zeros((N_FEAT, 128), dtype=np.float32)
            E = np.zeros(N_FEAT, dtype=np.float32)
            T = np.zeros(N_FEAT, dtype=np.float32)
            V = np.full(N_FEAT, np.nan, dtype=np.float64)
            for j in range(N_FEAT):
                if ca["fi"][j] < 0:
                    E[j], T[j] = 1.0, ca["term"][j]
                else:
                    F[j] = feats[c["off"] + ca["fi"][j]]
                    V[j] = ca["v"][j]
            recs.append({"id": did, "battle": c["battle"], "action": int(a), "split": split_of(c["battle"]),
                         "F": F, "E": E, "T": T, "V": V, "O": outs[a]})
    return recs


def q_base(r) -> float:
    q = np.where(r["E"] > 0, r["T"], 2.0 * np.nan_to_num(r["V"]) - 1.0)
    return float(q.mean())


def fit(train, val, k: int, seed: int, head0):
    import torch as th

    th.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def tens(rs):
        return (th.tensor(np.stack([r["F"] for r in rs])), th.tensor(np.stack([r["E"] for r in rs])),
                th.tensor(np.stack([r["T"] for r in rs])),
                th.tensor(np.array([(r["O"][:k].mean() + 1.0) / 2.0 for r in rs], dtype=np.float32)))

    def q01(h, F, E, T):
        p = th.sigmoid(h(F.reshape(-1, 128)).reshape(F.shape[0], N_FEAT))
        q = th.where(E > 0, (T + 1.0) / 2.0, p)
        return q.mean(1).clamp(1e-6, 1 - 1e-6)

    def bce(q, y):
        return -(y * th.log(q) + (1 - y) * th.log(1 - q)).mean()

    tr, va = tens(train), tens(val)
    h = copy.deepcopy(head0)
    for p in h.parameters():
        p.requires_grad_(True)
    opt = th.optim.Adam(h.parameters(), lr=LR, weight_decay=WD)
    with th.no_grad():
        best = (float(bce(q01(h, *va[:3]), va[3])), -1, copy.deepcopy(h.state_dict()))
    init_val = best[0]
    bad = 0
    n = len(train)
    for ep in range(EPOCHS):
        h.train()
        perm = rng.permutation(n)
        for i in range(0, n, BS):
            b = th.tensor(perm[i:i + BS])
            loss = bce(q01(h, tr[0][b], tr[1][b], tr[2][b]), tr[3][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        h.eval()
        with th.no_grad():
            vl = float(bce(q01(h, *va[:3]), va[3]))
        if vl < best[0] - 1e-6:
            best, bad = (vl, ep, copy.deepcopy(h.state_dict())), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    h.load_state_dict(best[2])
    h.eval()
    return h, {"k": k, "seed": seed, "val_bce_init": init_val, "val_bce_best": best[0],
               "best_epoch": best[1], "epochs_run": ep + 1}


def score_head(h, rs) -> np.ndarray:
    import torch as th

    with th.no_grad():
        F = th.tensor(np.stack([r["F"] for r in rs]))
        E = th.tensor(np.stack([r["E"] for r in rs]))
        T = th.tensor(np.stack([r["T"] for r in rs]))
        p = th.sigmoid(h(F.reshape(-1, 128)).reshape(F.shape[0], N_FEAT))
        q = th.where(E > 0, T, 2.0 * p - 1.0).mean(1)
    return q.numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="PLUMBING ONLY: every outcome is FABRICATED at random "
                    "(the real truth is never read); writes rows/smoke_*")
    a = ap.parse_args()
    import hashlib

    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    def split_of(b):
        if int(hashlib.sha256(f"{SAL}:{b}".encode()).hexdigest()[0], 16) % 2 == 0:
            return "held"
        return "val" if int(hashlib.sha256(f"{SAL}:val:{b}".encode()).hexdigest()[0], 16) <= 2 else "train"

    tr_ids = json.loads((ROWS / "subset_train.json").read_text())["ids"]
    he_ids = json.loads((ROWS / "subset_held.json").read_text())["ids"]
    idx, files = load_capture(ROWS / "cap")
    if a.smoke:
        rng = np.random.default_rng(0)
        held_set = set(he_ids)
        truth = {}
        for i, c in idx.items():
            s_ = 32 if i in held_set else 8
            outs = {k: rng.choice([-1.0, 0.0, 1.0], size=s_, p=[.45, .1, .45]) for k in c["actions"]}
            for k, ca in c["actions"].items():
                for j in range(N_FEAT):
                    if ca["term"][j] is not None:
                        outs[k][j] = ca["term"][j]
            truth[i] = {"ok": True, "outcomes": {k: list(v) for k, v in outs.items()}}
    else:
        truth = load_truth(ROWS / "truth_train_S8.jsonl", 8)
        truth.update(load_truth(ROWS / "truth_held_S32.jsonl", 32))
    recs = build(tr_ids + he_ids, idx, files, truth, split_of)
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "val"]
    held = [r for r in recs if r["split"] == "held"]
    for r in held:
        if len(r["O"]) != 32:
            raise RuntimeError(f"held-out {r['id']} has {len(r['O'])} seeds")
    # CRN alignment: a capture branch that ENDED within one ply must end the same way as truth
    # playout j under the same seed (the capture's successor IS truth's state otherwise)
    ended = [(r["T"][j], r["O"][j]) for r in recs for j in range(N_FEAT) if r["E"][j] > 0]
    agree = sum(1 for t, o in ended if t == o)
    crn = {"ended_branches": len(ended), "agree": agree}
    print(f"[refit] CRN alignment on ended branches: {agree}/{len(ended)}", flush=True)
    if ended and agree / len(ended) < 0.99:
        raise RuntimeError(f"capture and truth are not on the same dice: {crn}")
    report = {"n_actions": {"train": len(train), "val": len(val), "held": len(held)},
              "n_turns": {s: len({r["id"] for r in recs if r["split"] == s}) for s in ("train", "val", "held")},
              "refused_capture": sorted(i for i, c in idx.items() if not c.get("ok")),
              "refused_truth": sorted(i for i, t in truth.items() if not t.get("ok")),
              "crn_alignment": crn,
              "fits": []}
    out = {"ids": np.array([r["id"] for r in held]), "battle": np.array([r["battle"] for r in held]),
           "action": np.array([r["action"] for r in held]),
           "truth": np.stack([r["O"][8:] for r in held]), "leafseeds": np.stack([r["O"][:8] for r in held]),
           "BASE": np.array([q_base(r) for r in held])}
    for k in KS:
        out[f"LEAF_K{k}"] = np.array([r["O"][:k].mean() for r in held])
    with inference_globals(a.threads):
        model = load_checkpoint(CKPT)
        head0 = copy.deepcopy(model.policy.features_extractor.win_head).float()
        del model
        # the warm-started head reproduces BASE exactly on the held-out actions (the tap is right)
        chk = score_head(head0, held)
        report["base_reproduction_max_abs"] = float(np.abs(chk - out["BASE"]).max())
        if report["base_reproduction_max_abs"] > 1e-5:
            raise RuntimeError(f"win_head on stored value_pooled != stored V: {report['base_reproduction_max_abs']}")
        for k in KS:
            for s in FIT_SEEDS:
                h, info = fit(train, val, k, s, head0)
                report["fits"].append(info)
                out[f"REFIT_K{k}_s{s}"] = score_head(h, held)
                print(f"[refit] K={k} seed={s}: {info}", flush=True)
    pre = "smoke_" if a.smoke else ""
    np.savez_compressed(ROWS / f"{pre}scores.npz", **out)
    (ROWS / f"{pre}fit_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"[refit] wrote scores for {len(held)} held-out actions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
