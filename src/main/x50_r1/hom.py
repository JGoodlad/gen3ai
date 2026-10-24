"""THE HUMAN OPPONENT MODEL (HOM; ``design_x50_rung1b_human_opponent_model.md`` §4): X26's network fine-tuned from P*
on the choices of human players rated >= 1500, used ONLY as search's opponent-choice likelihood judge. The policy never
reads it; it is never a run, a baseline or a file under ``models/``.

    python -m main.x50_r1.hom compact --parts <archive>/train --out <archive>/hom/rows
    python -m main.x50_r1.hom train --rows <archive>/hom/rows --init <P* zip> --out <archive>/hom --budget-min 40
    python -m main.x50_r1.hom export --rows <archive>/hom/rows --init <P* zip> --out <archive>/hom

* ``compact`` keeps the training extraction's rows the HOM reads (:func:`eligible`: a completed row, mode 2; an own
  action, (b)'s label; >= 2 legal actions; the acting human rated >= 1500) and splits them by battle: VALIDATION iff
  ``sha256(battle id) mod 50 == 0`` (the log's file stem), else TRAIN. Train rows go to shards of the extraction's
  order (``SHARD`` rows each); ``rows_manifest.json`` holds every shard's sha256, the source parts' sha256 and the
  counts, and ITS sha256 is the rows' identity in the sidecar.
* ``train`` (fixed by the registration): the cross-entropy of the masked policy softmax at the human's action; every
  parameter the policy logits depend on trains, the value head gets no gradient (it is not in the loss); AdamW,
  lr 1e-4 cosine to 1e-5 over 3 epochs, weight decay 1e-5, batch 512, gradient-norm clip 0.5, fp32 on the CPU, seed 51.
  Validation NLL at step 0 (P* itself) and every quarter epoch; the kept model is the lowest; stop after 2 consecutive
  rises. The network runs in eval mode (its dropouts are p = 0; LayerNorm only), so the trained function is the one
  the judge evaluates. RESUMABLE: the batch plan is a pure function of the shard sizes and the seed, and ``state.pt``
  (model, optimiser, step, validation history) is written atomically every few minutes and at the budget's end.
* ``export`` writes the kept model in P*'s checkpoint format (P*'s zip with ``policy.pth`` replaced), the run's
  ``model_config.json`` beside it (the loader's lookup), and the sidecar: init P* and its sha256, the rows manifest's
  sha256, the seed, the commit, the validation history, the HOM zip's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RATING = 1500
VAL_MOD = 50
SHARD = 16384
SEED = 51
EPOCHS = 3
BATCH = 512
LR_MAX, LR_MIN = 1e-4, 1e-5
WEIGHT_DECAY = 1e-5
CLIP = 0.5
EVALS_PER_EPOCH = 4
RISES_TO_STOP = 2
BUFFER_SHARDS = 8
THREADS = 6
MODE_COMPLETED = 2
SAVE_EVERY_S = 600.0


class HomError(RuntimeError):
    """A rows manifest, checkpoint or state that does not match what it claims (a defect: stop)."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def is_val(battle_id: str) -> bool:
    """The validation split (design §3): ``sha256(battle id) mod 50 == 0``, the battle id being the log's stem."""
    return int(hashlib.sha256(battle_id.encode()).hexdigest(), 16) % VAL_MOD == 0


def eligible(z: Dict[str, np.ndarray]) -> np.ndarray:
    """The rows the HOM reads (module docstring)."""
    return ((z["mode"] == MODE_COMPLETED) & (z["own_action"] >= 0) & (z["n_legal"] >= 2)
            & (z["rating_viewer"] >= RATING))


def battle_key(battle_id: str) -> int:
    """A battle's int64 cluster key (the bootstrap's and the split's unit)."""
    return int(hashlib.sha256(battle_id.encode()).hexdigest()[:15], 16)


# ------------------------------------------------------------------------------------------------ compact

def _part_rows(js: Path) -> Optional[Dict[str, np.ndarray]]:
    meta = json.loads(js.read_text())
    npz = js.with_suffix(".npz")
    z = np.load(npz)
    if "mode" not in z.files:
        return None
    small = {k: z[k] for k in ("mode", "own_action", "n_legal", "rating_viewer", "battle", "forced", "seat")}
    sel = eligible(small)
    if not sel.any():
        return None
    first = int(js.stem.rsplit("_", 1)[1]) * int(meta["part_size"])
    ids = [Path(meta["battles"][int(b) - first]).stem for b in small["battle"][sel]]
    rows = z["rows"][sel]
    return {"rows": rows.astype(np.float32), "mask": z["mask"][sel].astype(np.int8),
            "act": small["own_action"][sel].astype(np.int8), "forced": small["forced"][sel].astype(bool),
            "seat": small["seat"][sel].astype(np.int8), "rating": small["rating_viewer"][sel].astype(np.int32),
            "bkey": np.array([battle_key(i) for i in ids], np.int64),
            "val": np.array([is_val(i) for i in ids], bool)}


FIELDS = ("rows", "mask", "act", "forced", "seat", "rating", "bkey")


def _write(out: Path, name: str, buf: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    arr = {f: np.concatenate([b[f] for b in buf]) for f in FIELDS}
    p = out / f"{name}.npz"
    tmp = out / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **arr)  # type: ignore[arg-type]
    os.replace(tmp, p)
    return {"name": name, "rows": int(len(arr["act"])), "sha256": sha256_file(p),
            "battles": int(len(np.unique(arr["bkey"])))}


def compact(parts: Path, out: Path) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    jss = sorted(parts.glob("humans_*.json"))
    src = []
    counts: Dict[str, int] = {}
    shards: List[Dict[str, Any]] = []
    buf: List[Dict[str, np.ndarray]] = []
    val: List[Dict[str, np.ndarray]] = []
    n_buf = 0
    for js in jss:
        meta = json.loads(js.read_text())
        for k, v in meta["counts"].items():
            counts[k] = counts.get(k, 0) + int(v)
        src.append({"part": js.stem, "json_sha256": sha256_file(js), "npz_sha256": sha256_file(js.with_suffix(".npz")),
                    "commit": meta.get("commit"), "exclude_teams": meta.get("exclude_teams"),
                    "rows": meta.get("rows")})
        r = _part_rows(js)
        if r is None:
            continue
        v = r.pop("val")
        val.append({f: r[f][v] for f in FIELDS})
        buf.append({f: r[f][~v] for f in FIELDS})
        n_buf += int((~v).sum())
        if n_buf >= SHARD:
            shards.append(_write(out, f"train_{len(shards):04d}", buf))
            buf, n_buf = [], 0
    if buf and n_buf:
        shards.append(_write(out, f"train_{len(shards):04d}", buf))
    vrec = _write(out, "val", val)
    if any(s["commit"] != src[0]["commit"] for s in src) or any(not s["exclude_teams"] for s in src):
        raise HomError("the training parts do not share one commit, or a part ran without the team-memory exclusion")
    man = {"schema": "x50_r1b_hom_rows_v1", "filter": {"mode": MODE_COMPLETED, "own_action": ">= 0",
                                                        "n_legal": ">= 2", "rating_viewer": f">= {RATING}"},
           "split": f"validation iff sha256(battle id) mod {VAL_MOD} == 0", "parts": len(src), "sources": src,
           "extraction_counts": counts, "train_shards": shards, "val": vrec,
           "train_rows": int(sum(s["rows"] for s in shards)), "val_rows": vrec["rows"]}
    (out / "rows_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def load_manifest(rows: Path) -> Tuple[Dict[str, Any], str]:
    p = rows / "rows_manifest.json"
    return json.loads(p.read_text()), sha256_file(p)


# ------------------------------------------------------------------------------------------------ the plan

def plan(sizes: List[int]) -> List[Tuple[int, int, List[int], int]]:
    """The whole run's buffers in order: (epoch, buffer index, shard ids, batches). Pure in the sizes and SEED."""
    out = []
    for e in range(EPOCHS):
        order = np.random.default_rng([SEED, e]).permutation(len(sizes)).tolist()
        for b in range(0, len(order), BUFFER_SHARDS):
            ids = order[b:b + BUFFER_SHARDS]
            n = sum(sizes[i] for i in ids)
            out.append((e, b // BUFFER_SHARDS, ids, int(math.ceil(n / BATCH))))
    return out


def eval_steps(pl: List[Tuple[int, int, List[int], int]]) -> List[int]:
    """The validation steps: every quarter epoch (in optimiser steps), the epoch's end included."""
    per = [0] * EPOCHS
    for e, _, _, nb in pl:
        per[e] += nb
    out, base = [], 0
    for e in range(EPOCHS):
        out.extend(base + int(round(per[e] * q / EVALS_PER_EPOCH)) for q in range(1, EVALS_PER_EPOCH + 1))
        base += per[e]
    return out


def lr_at(step: int, total: int) -> float:
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1)))


def stop_now(hist: List[float]) -> bool:
    """Two consecutive rises of the validation NLL."""
    if len(hist) < RISES_TO_STOP + 1:
        return False
    tail = hist[-(RISES_TO_STOP + 1):]
    return all(tail[i + 1] > tail[i] for i in range(RISES_TO_STOP))


# ------------------------------------------------------------------------------------------------ the network

def policy_logits_t(pol, x, m):
    """The policy's 11 raw logits as a differentiable tensor (``qhat.logits_and_values``'s actor wiring)."""
    from agents.model.extra_obs_keys import zero_extra_obs

    ob = {"observation": x, "action_mask": m}
    ob.update(zero_extra_obs(pol.features_extractor, batch=int(m.shape[0]), device=x.device))
    pi_f, _ = pol.extract_features(ob)
    return pol._pointer_logits(pol.mlp_extractor.forward_actor(pi_f))


def masked_nll_t(logits, mask, act):
    import torch as th

    lg = logits.float().masked_fill(~mask.bool(), float("-inf"))
    return -th.log_softmax(lg, dim=-1).gather(1, act.long().view(-1, 1)).squeeze(1)


def val_nll(model, V: Dict[str, np.ndarray], batch: int = 1024) -> float:
    """Mean validation NLL in float64 (the judge's masked softmax: ``opp_likelihood.log_probs``)."""
    from main.policy_spectrum.qhat import logits_and_values
    from main.value_search.opp_likelihood import log_probs

    out = logits_and_values(model, V["rows"], V["mask"].astype(np.uint8), batch=batch)
    lp, _ = log_probs(out, V["mask"].astype(np.uint8), V["act"].astype(np.int64))
    return float(-lp.mean())


def _load_shard(rows: Path, rec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    p = rows / f"{rec['name']}.npz"
    if sha256_file(p) != rec["sha256"]:
        raise HomError(f"{p}: sha256 differs from the rows manifest")
    z = np.load(p)
    return {f: z[f] for f in FIELDS}


def _atomic_save(obj: Any, p: Path) -> None:
    import torch as th

    tmp = p.with_suffix(".tmp")
    th.save(obj, tmp)
    os.replace(tmp, p)


def train(rows: Path, init: Path, out: Path, budget_min: float, threads: int = THREADS,
          limit_steps: Optional[int] = None) -> int:
    """One resumable chunk of the registered training (module docstring). 0 = finished, 3 = budget reached."""
    import torch as th

    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    out.mkdir(parents=True, exist_ok=True)
    man, man_sha = load_manifest(rows)
    sizes = [int(s["rows"]) for s in man["train_shards"]]
    pl = plan(sizes)
    total = sum(nb for *_, nb in pl)
    evs = eval_steps(pl)
    t0 = time.monotonic()
    with inference_globals(threads):
        th.manual_seed(SEED)
        model = load_checkpoint(init)
        pol = model.policy
        pol.eval()
        params = [p for p in pol.parameters() if p.requires_grad]
        opt = th.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        sp = out / "state.pt"
        st: Dict[str, Any] = {"step": 0, "hist": [], "hist_steps": [], "best": None, "done": False,
                              "rows_manifest_sha256": man_sha, "init_sha256": sha256_file(init), "total": total}
        if sp.exists():
            st = th.load(sp, weights_only=False)
            if st["rows_manifest_sha256"] != man_sha or st["init_sha256"] != sha256_file(init):
                raise HomError("state.pt was trained on other rows or another init")
            pol.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
        if st["done"]:
            return 0
        V = _load_shard(rows, man["val"])

        def save() -> None:
            st["model"] = pol.state_dict()
            st["opt"] = opt.state_dict()
            _atomic_save(st, sp)

        def evaluate() -> bool:
            v = val_nll(model, V)
            st["hist"].append(v)
            st["hist_steps"].append(st["step"])
            if st["best"] is None or v < min(st["hist"][:-1] or [math.inf]):
                st["best"] = st["step"]
                _atomic_save(pol.state_dict(), out / "best.pt")
            print(f"[hom] eval step {st['step']}/{total}: val NLL {v:.5f} (best @ {st['best']})", flush=True)
            if stop_now(st["hist"]):
                print("[hom] early stop: two consecutive rises", flush=True)
                return True
            return False

        if not st["hist"]:
            if evaluate():
                st["done"] = True
            save()
        last_save = time.monotonic()
        step = 0
        for (e, b, ids, nb) in pl:
            if step + nb <= st["step"]:
                step += nb
                continue
            buf = [_load_shard(rows, man["train_shards"][i]) for i in ids]
            B = {f: np.concatenate([x[f] for x in buf]) for f in ("rows", "mask", "act")}
            del buf
            perm = np.random.default_rng([SEED, e, b]).permutation(len(B["act"]))
            for j in range(nb):
                if step < st["step"]:
                    step += 1
                    continue
                if st["done"]:
                    break
                ix = perm[j * BATCH:(j + 1) * BATCH]
                x = th.from_numpy(B["rows"][ix])
                m = th.from_numpy(B["mask"][ix].astype(np.float32))
                a = th.from_numpy(B["act"][ix].astype(np.int64))
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total)
                loss = masked_nll_t(policy_logits_t(pol, x, m), m, a).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(params, CLIP)
                opt.step()
                step += 1
                st["step"] = step
                if step % 50 == 0:
                    print(f"[hom] step {step}/{total} epoch {e} loss {float(loss):.4f} "
                          f"{(time.monotonic() - t0) / 60:.1f} min", flush=True)
                if step in evs:
                    if evaluate():
                        st["done"] = True
                if st["done"] or step == total:
                    st["done"] = True
                    save()
                    return 0
                if time.monotonic() - last_save > SAVE_EVERY_S:
                    save()
                    last_save = time.monotonic()
                if (limit_steps is not None and step >= limit_steps) or \
                        (budget_min and (time.monotonic() - t0) / 60 > budget_min):
                    save()
                    print(f"[hom] budget: stopping at step {step}; resume to continue", flush=True)
                    return 3
            if st["done"]:
                break
        st["done"] = True
        save()
    return 0


# ------------------------------------------------------------------------------------------------ export

def export(rows: Path, init: Path, out: Path, config: Path) -> Dict[str, Any]:
    """The kept model in P*'s checkpoint format, its config, and the sidecar (module docstring)."""
    import torch as th

    from utils.git import get_git_hash

    st = th.load(out / "state.pt", weights_only=False)
    if not st["done"]:
        raise HomError("the training has not finished")
    best = th.load(out / "best.pt", weights_only=False)
    dst = out / "model"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(config, dst / "model_config.json")
    zp = dst / "hom_x50r1b.zip"
    buf = io.BytesIO()
    th.save(best, buf)
    with zipfile.ZipFile(init) as zi, zipfile.ZipFile(zp.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as zo:
        for info in zi.infolist():
            data = buf.getvalue() if info.filename == "policy.pth" else zi.read(info.filename)
            zo.writestr(info, data)
    os.replace(zp.with_suffix(".tmp"), zp)
    _, man_sha = load_manifest(rows)
    side = {"schema": "x50_r1b_hom_sidecar_v1", "what": "the human opponent model (X50 rung 1b use (b)): "
            "search's opponent-choice likelihood judge only; never a run, a baseline or a policy",
            "init": str(init), "init_sha256": sha256_file(init), "rows_manifest_sha256": man_sha,
            "seed": SEED, "commit": get_git_hash(), "recipe": {
                "loss": "masked policy softmax cross-entropy at the human's action", "optimizer": "AdamW",
                "lr": f"{LR_MAX} cosine to {LR_MIN} over {EPOCHS} epochs", "weight_decay": WEIGHT_DECAY,
                "batch": BATCH, "clip": CLIP, "dtype": "fp32", "device": "cpu",
                "validation": f"step 0 and every 1/{EVALS_PER_EPOCH} epoch; kept = lowest; stop after "
                              f"{RISES_TO_STOP} consecutive rises"},
            "steps_planned": st["total"], "steps_run": st["step"], "kept_step": st["best"],
            "val_hist": [[int(s), float(v)] for s, v in zip(st["hist_steps"], st["hist"])],
            "hom_zip": str(zp), "hom_sha256": sha256_file(zp),
            "note": "policy.optimizer.pth, data and pytorch_variables.pth are P*'s, unchanged; the value head is "
                    "P*'s (no gradient); the belief and intent heads are not calibrated and nothing may read them"}
    (dst / "hom_sidecar.json").write_text(json.dumps(side, indent=1))
    return side


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.hom", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compact")
    c.add_argument("--parts", required=True)
    c.add_argument("--out", required=True)
    for name in ("train", "export"):
        t = sub.add_parser(name)
        t.add_argument("--rows", required=True)
        t.add_argument("--init", required=True)
        t.add_argument("--out", required=True)
        if name == "train":
            t.add_argument("--budget-min", type=float, default=40.0)
            t.add_argument("--threads", type=int, default=THREADS)
        else:
            t.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "compact":
        man = compact(Path(a.parts), Path(a.out))
        print(json.dumps({k: man[k] for k in ("parts", "train_rows", "val_rows")}), flush=True)
        return 0
    if a.cmd == "train":
        return train(Path(a.rows), Path(a.init), Path(a.out), a.budget_min, a.threads)
    side = export(Path(a.rows), Path(a.init), Path(a.out), Path(a.config))
    print(json.dumps({k: side[k] for k in ("hom_sha256", "kept_step", "steps_run")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
