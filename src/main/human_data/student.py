r"""X59-S's MATCHED STUDENTS: our architecture from the same random init per seed, taught one label source each.

    python -m main.human_data.student train  --arm human|pstar|pstar0|selfplay --rows <rows> [--labels <labels>] \
        --registration <results/registration.json> --seed 5901 --out <dir>/s_h_5901 [--epochs 3] [--budget-min 40] \
        [--device cpu|cuda]
    python -m main.human_data.student export --out <dir>/s_h_5901

The three arms (``x59s_matched_students_2026-10-25/PREREG.md``):

* ``human`` (S-H): the human rowset's rows, the HUMAN's action as the label;
* ``pstar`` (S-P): the SAME rows, P\*'s sample RESTRICTED to the human's consistent set as the label (a
  "P\*-restricted-to-human-support" teacher, PREREG Amendment 2 M2: not P\* as an independent teacher);
* ``pstar0`` (S-P0, descriptive): the SAME rows, P\*'s UNRESTRICTED sample over the legal actions
  (``derive``'s ``labels_pstar0``);
* ``selfplay`` (S-S): P\*'s own self-play rows (``teacher selfplay``), P\*'s sampled action as the label.

A hidden-move row (plan §3.2) is trained on its SET label in S-H and S-P alike: the loss is, for every row,
``logsumexp(logits over the legal actions) - logsumexp(logits over the TARGET set)``, the target set being the one
label or, on a hidden row, the legal moves. One recipe for every arm, fixed before any student trained:

* the network is ``main.fresh_checkpoint.build_fresh_model(seed)`` (``--arch production``'s network, P\*'s
  architecture without its ride-along diagnostics); the same seed gives the same initial bytes in every arm (its
  sha256 is recorded); every parameter the policy logits depend on trains, the value head gets no gradient;
* AdamW (the production optimiser kwargs: weight decay 1e-5, eps 1e-5), lr linear warm-up over :data:`WARMUP`
  steps to :data:`LR_MAX`, cosine to :data:`LR_MIN`; batch :data:`BATCH`; gradient-norm clip :data:`CLIP`; fp32;
  the network in eval mode (it has no dropout);
* a FIXED step budget, ``epochs x ceil(train rows / BATCH)``, no early stopping (it would de-match the arms); the
  batch order is :mod:`main.x50_r1.hom`'s plan keyed by the seed, so S-H and S-P see the SAME batches;
* validation NLL every quarter epoch is DESCRIPTIVE (nothing stops on it); at the end, every student is read on the
  common yardsticks: the human validation rows' human labels, and the self-play validation rows' P\* labels.

RESUMABLE: ``state.pt`` is written atomically every few minutes and at the budget's end. ``export`` writes the final
model through the project's own save path (``MaskablePPO.save`` + ``save_model_snapshot``) and checks that the
reloaded checkpoint gives the in-memory logits bit for bit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.human_data import rowset as RS

ARMS = ("human", "pstar", "pstar0", "selfplay")
LABEL_ARMS = ("pstar", "pstar0")
#: the registration key of each arm's rows and labels (``checks register``; PREREG Amendment 2, REVIEW E m4)
REG_ROWS = {"human": "rows_manifest_sha256", "pstar": "rows_manifest_sha256", "pstar0": "rows_manifest_sha256",
            "selfplay": "selfplay_manifest_sha256"}
REG_LABELS = {"pstar": "labels_manifest_sha256", "pstar0": "labels0_manifest_sha256"}
#: the learned-nothing guard (Amendment 2, REVIEW E M4): the final validation NLL on the arm's OWN labels must be at
#: least this far below the uniform NLL over the legal actions, else the student is VOID (a bug or a recipe failure)
LEARNED_MARGIN = 0.1
BATCH = 512
LR_MAX, LR_MIN = 3e-4, 3e-5
WARMUP = 1000
CLIP = 0.5
EVALS_PER_EPOCH = 4
BUFFER_SHARDS = 8
THREADS = 2          # intra-op threads: 6 on the shared agent cores oversubscribed them 60x (measured)
SAVE_EVERY_S = 600.0
LOSS_EVERY = 100     # the training loss is recorded every LOSS_EVERY steps (the CUDA pre-flight's parity points)
TRAIN_FIELDS = ("rows", "mask", "act", "hidden", "moveset", "cons", "rid")


class StudentError(RuntimeError):
    """A student's inputs or state that do not match what they claim (a defect: stop)."""


def plan(sizes: List[int], seed: int, epochs: int) -> List[Tuple[int, int, List[int], int]]:
    """(epoch, buffer, shard ids, batches) for the whole run: pure in the sizes, the seed and the epochs."""
    out = []
    for e in range(epochs):
        order = np.random.default_rng([seed, e]).permutation(len(sizes)).tolist()
        for b in range(0, len(order), BUFFER_SHARDS):
            ids = order[b:b + BUFFER_SHARDS]
            n = sum(sizes[i] for i in ids)
            out.append((e, b // BUFFER_SHARDS, ids, int(math.ceil(n / BATCH))))
    return out


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    t = min(step - WARMUP, total - WARMUP) / max(total - WARMUP, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * t))


def targets(arm: str, z: Dict[str, np.ndarray], lab: Optional[np.ndarray]) -> np.ndarray:
    """``[n, 11]`` int8: the TARGET set of every row (module docstring)."""
    n = len(z["act"])
    act = z["act"] if arm not in LABEL_ARMS else lab
    if act is None:
        raise StudentError(f"the {arm} arm needs its label column")
    hid = z["hidden"].astype(bool)
    if arm in LABEL_ARMS:
        if ((act < 0) != hid).any():
            raise StudentError("a P* label column whose hidden rows differ from the rowset's")
        allowed = z["cons"] if arm == "pstar" else z["mask"]
        if (allowed[np.nonzero(~hid)[0], act[~hid]] != 1).any():
            raise StudentError("VOID: a P* label is an action the log rules out (outside the consistent set)"
                               if arm == "pstar" else "VOID: a P* label outside the legal actions")
    if ((act < 0) & ~hid).any():
        raise StudentError("an unlabelled row that is not a hidden move")
    t = np.zeros((n, RS.N_ACT), np.int8)
    li = np.nonzero(~hid)[0]
    t[li, act[li]] = 1
    t[hid] = z["moveset"][hid]
    if ((t == 1) & (z["mask"] == 0)).any() or (t.sum(1) == 0).any():
        raise StudentError("a target outside the legal mask, or an empty target")
    return t


def set_nll_t(logits, legal, target):
    """``logsumexp(legal) - logsumexp(target)`` per row (the module docstring's loss)."""
    import torch as th

    lg = logits.float()
    a = th.logsumexp(lg.masked_fill(~legal.bool(), float("-inf")), dim=-1)
    b = th.logsumexp(lg.masked_fill(~target.bool(), float("-inf")), dim=-1)
    return a - b


def _shard(rows: Path, rec: Dict[str, Any], arm: str, labels: Optional[Path]) -> Dict[str, np.ndarray]:
    z = RS.load_shard(rows, rec, TRAIN_FIELDS)
    lab = None
    if arm in LABEL_ARMS:
        assert labels is not None
        lr = json.loads((labels / f"{rec['name']}.json").read_text())
        if lr["rows_sha256"] != rec["sha256"]:
            raise StudentError(f"{rec['name']}: the label column was made on other rows")
        p = labels / f"{rec['name']}.npz"
        if RS.S_sha(p) != lr["sha256"]:
            raise StudentError(f"{p}: sha256 differs from its label record")
        lz = np.load(p)
        if not np.array_equal(lz["rid"], z["rid"]):
            raise StudentError(f"{rec['name']}: the label column's row ids differ from the rows'")
        lab = lz["act_p"]
    return {"rows": z["rows"], "mask": z["mask"], "target": targets(arm, z, lab)}


def uniform_nll(V: Dict[str, np.ndarray]) -> float:
    """The set-NLL of the uniform policy over the legal actions: mean log(|legal| / |target|)."""
    return float(np.mean(np.log(V["mask"].sum(1).astype(np.float64)) - np.log(V["target"].sum(1).astype(np.float64))))


def check_registration(reg: Dict[str, Any], arm: str, rows_sha: str, labels_sha: Optional[str]) -> None:
    """VOID 1 enforced (REVIEW E m4): the arm's rows and labels are the registered ones, or the trainer refuses."""
    if not reg.get("pass"):
        raise StudentError("the registration did not pass")
    if reg.get(REG_ROWS[arm]) != rows_sha:
        raise StudentError(f"VOID: the {arm} rows manifest {rows_sha[:12]} is not the registered one")
    if arm in REG_LABELS and reg.get(REG_LABELS[arm]) != labels_sha:
        raise StudentError(f"VOID: the {arm} labels manifest is not the registered one")


def precision_flags() -> Dict[str, Any]:
    """What fp32 means on this device (Amendment 2, M7 check 3): TF32 off for matmul and cuDNN, 'highest'."""
    import torch as th

    return {"matmul_allow_tf32": bool(th.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(th.backends.cudnn.allow_tf32),
            "float32_matmul_precision": th.get_float32_matmul_precision()}


def _state_digest(sd: Dict[str, Any]) -> str:
    import torch as th

    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(th.as_tensor(sd[k]).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _atomic_save(obj: Any, p: Path) -> None:
    import torch as th

    tmp = p.with_suffix(".tmp")
    th.save(obj, tmp)
    os.replace(tmp, p)


def eval_nll(pol, V: Dict[str, np.ndarray], device: str, batch: int = 1024) -> float:
    """Mean set-NLL over ``V`` (float64), no gradient."""
    import torch as th

    from main.x50_r1.hom import policy_logits_t

    tot, n = 0.0, 0
    with th.no_grad():
        for i in range(0, len(V["target"]), batch):
            x = th.from_numpy(V["rows"][i:i + batch]).to(device)
            m = th.from_numpy(V["mask"][i:i + batch].astype(np.float32)).to(device)
            t = th.from_numpy(V["target"][i:i + batch].astype(np.float32)).to(device)
            ll = set_nll_t(policy_logits_t(pol, x, m), m, t)
            tot += float(ll.double().sum())
            n += len(ll)
    return tot / max(n, 1)


def train(arm: str, rows: Path, labels: Optional[Path], seed: int, out: Path, epochs: int, budget_min: float,
          threads: int = THREADS, device: str = "cpu", limit_steps: Optional[int] = None,
          registration: Optional[Dict[str, Any]] = None) -> int:
    """One resumable chunk of the registered recipe (module docstring). 0 = finished, 3 = budget reached."""
    import torch as th

    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals
    from main.x50_r1.hom import policy_logits_t

    if arm not in ARMS:
        raise StudentError(f"arm {arm!r} is not one of {ARMS}")
    out.mkdir(parents=True, exist_ok=True)
    man, man_sha = RS.load_manifest(rows)
    lab_sha = RS.S_sha(labels / "labels_manifest.json") if arm in LABEL_ARMS and labels is not None else None
    if registration is not None:
        check_registration(registration, arm, man_sha, lab_sha)
    sizes = [int(s["rows"]) for s in man["train_shards"]]
    pl = plan(sizes, seed, epochs)
    total = sum(nb for *_, nb in pl)
    per_epoch = total // epochs
    evs = {int(round(per_epoch * (e + q / EVALS_PER_EPOCH))) for e in range(epochs) for q in range(1, 5)}
    t0 = time.monotonic()
    with inference_globals(threads, device):
        model, _, _ = build_fresh_model(seed)
        pol = model.policy.to(device)
        pol.eval()
        init_digest = _state_digest(pol.state_dict())
        params = [p for p in pol.parameters() if p.requires_grad]
        opt = th.optim.AdamW(params, lr=LR_MAX, weight_decay=1e-5, eps=1e-5)
        sp = out / "state.pt"
        ident = {"arm": arm, "seed": seed, "epochs": epochs, "rows_manifest_sha256": man_sha,
                 "labels_manifest_sha256": lab_sha, "init_sha256": init_digest, "total": total}
        st: Dict[str, Any] = {**ident, "step": 0, "hist": [], "hist_steps": [], "done": False, "train_s": 0.0}
        if sp.exists():
            st = th.load(sp, weights_only=False, map_location=device)
            if any(st.get(k) != v for k, v in ident.items()):
                raise StudentError(f"{sp} was trained under another arm / rows / labels / seed / init / budget")
            pol.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            st["resumed_opt_devices"] = sorted({str(v.device) for sv in opt.state.values() for k, v in sv.items()
                                                if k != "step" and th.is_tensor(v)})
        if st["done"]:
            return 0
        V = _shard(rows, man["val"], arm, labels)
        st.setdefault("uniform_val_nll", uniform_nll(V))
        st.setdefault("device_info", {"device": device, "torch": th.__version__,
                                      "cuda_device": th.cuda.get_device_name(0) if device.startswith("cuda") else None})

        def save() -> None:
            st["model"] = pol.state_dict()
            st["opt"] = opt.state_dict()
            _atomic_save(st, sp)

        def evaluate() -> None:
            v = eval_nll(pol, V, device)
            st["hist"].append(v)
            st["hist_steps"].append(st["step"])
            print(f"[student] {arm} seed {seed} step {st['step']}/{total}: val NLL {v:.5f}", flush=True)

        if not st["hist"]:
            evaluate()
            save()
        last_save = time.monotonic()
        step = 0
        for (e, b, ids, nb) in pl:
            if step + nb <= st["step"]:
                step += nb
                continue
            buf = [_shard(rows, man["train_shards"][i], arm, labels) for i in ids]
            B = {f: np.concatenate([x[f] for x in buf]) for f in ("rows", "mask", "target")}
            del buf
            perm = np.random.default_rng([seed, e, b]).permutation(len(B["target"]))
            for j in range(nb):
                if step < st["step"]:
                    step += 1
                    continue
                ts = time.monotonic()
                ix = perm[j * BATCH:(j + 1) * BATCH]
                x = th.from_numpy(B["rows"][ix]).to(device)
                m = th.from_numpy(B["mask"][ix].astype(np.float32)).to(device)
                t = th.from_numpy(B["target"][ix].astype(np.float32)).to(device)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total)
                loss = set_nll_t(policy_logits_t(pol, x, m), m, t).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(params, CLIP)
                opt.step()
                step += 1
                st["step"] = step
                st["train_s"] += time.monotonic() - ts
                if step % LOSS_EVERY == 0:
                    st.setdefault("loss_at", {})[str(step)] = float(loss.detach())
                    st.setdefault("precision", precision_flags())
                if step % 100 == 0:
                    pf = precision_flags()
                    print(f"[student] {arm} seed {seed} step {step}/{total} epoch {e} loss {float(loss):.4f} "
                          f"{st['train_s'] / step * 1000:.0f} ms/step tf32 {pf['matmul_allow_tf32']}/"
                          f"{pf['cudnn_allow_tf32']} {pf['float32_matmul_precision']}", flush=True)
                if step in evs:
                    evaluate()
                if step == total:
                    st["done"] = True
                    save()
                    return 0
                if time.monotonic() - last_save > SAVE_EVERY_S:
                    save()
                    last_save = time.monotonic()
                if (limit_steps is not None and step >= limit_steps) or \
                        (budget_min and (time.monotonic() - t0) / 60 > budget_min):
                    save()
                    print(f"[student] budget: stopping at step {step}; resume to continue", flush=True)
                    return 3
        st["done"] = True
        save()
    return 0


def export(out: Path, yardsticks: Optional[Dict[str, Tuple[Path, str]]] = None) -> Dict[str, Any]:
    """The final student in the project's checkpoint format (module docstring), its sidecar, and the yardsticks:
    ``{name: (rows dir, arm whose labels to read)}`` scored on each one's validation shard."""
    import torch as th

    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import save_model_snapshot
    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits
    from utils.git import get_git_hash

    st = th.load(out / "state.pt", weights_only=False, map_location="cpu")
    if not st["done"]:
        raise StudentError("the training has not finished")
    dst = out / "model"
    dst.mkdir(parents=True, exist_ok=True)
    with inference_globals(THREADS):
        model, args, pk = build_fresh_model(int(st["seed"]))
        model.policy.load_state_dict(st["model"])
        model.policy.eval()
        name = f"student_{st['arm']}_{st['seed']}"
        model.save(str(dst / name))
        version = ModelVersion.from_layout_and_policy_kwargs(
            pk["features_extractor_kwargs"]["layout"], pk, vf_coef=args.vf_coef,
            move_belief_coef=args.move_belief_coef, spread_belief_coef=args.spread_belief_coef,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            hp_type_belief_coef=args.hp_type_belief_coef, item_belief_coef=args.item_belief_coef)
        save_model_snapshot(str(dst), version, git_hash=get_git_hash())
        zp = dst / f"{name}.zip"
        back = load_checkpoint(zp)
        man, _ = RS.load_manifest(Path(_rows_of(st, out)))
        prec = man["val"] if man["val"]["rows"] else man["train_shards"][0]
        probe = RS.load_shard(Path(_rows_of(st, out)), prec, ("rows", "mask"))
        a = policy_logits(model, probe["rows"][:256], probe["mask"][:256].astype(np.uint8))
        b = policy_logits(back, probe["rows"][:256], probe["mask"][:256].astype(np.uint8))
        if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            raise StudentError("VOID: the reloaded checkpoint's logits differ from the trained student's")
        from main.anchors.core_side import OurSideState, load_policy

        osd = OurSideState()
        c = policy_logits(load_policy(str(zp), "cpu", "bare", osd), probe["rows"][:256],
                          probe["mask"][:256].astype(np.uint8))
        if osd.model_loader != "bare" or not np.array_equal(a.view(np.uint32), c.view(np.uint32)):
            raise StudentError("VOID: the anchors harness's bare load does not reproduce the student's logits")
        ys: Dict[str, float] = {}
        for yname, (yrows, yarm) in (yardsticks or {}).items():
            yman, _ = RS.load_manifest(yrows)
            V = _shard(yrows, yman["val"], yarm, None)
            ys[yname] = eval_nll(model.policy, V, "cpu") if len(V["target"]) else float("nan")
    buf = io.BytesIO()
    th.save(st["model"], buf)
    side = {"schema": "gen3_x59s_student_v1", "arm": st["arm"], "seed": st["seed"], "epochs": st["epochs"],
            "rows_manifest_sha256": st["rows_manifest_sha256"], "labels_manifest_sha256": st["labels_manifest_sha256"],
            "init_sha256": st["init_sha256"], "final_sha256": _state_digest(st["model"]),
            "steps": st["step"], "train_s": st["train_s"],
            "val_hist": [[int(s), float(v)] for s, v in zip(st["hist_steps"], st["hist"])],
            "recipe": {"loss": "logsumexp(legal) - logsumexp(target set)", "optimizer": "AdamW wd 1e-5 eps 1e-5",
                       "lr": f"warm-up {WARMUP} to {LR_MAX}, cosine to {LR_MIN}", "batch": BATCH, "clip": CLIP,
                       "dtype": "fp32"},
            "zip": str(zp), "zip_sha256": RS.S_sha(zp), "commit": get_git_hash(), "yardsticks": ys,
            "learned_guard": learned_guard(st), "device_info": st.get("device_info"),
            "reload_logits_bit_equal": True, "anchors_bare_load_bit_equal": True}
    (dst / "student_sidecar.json").write_text(json.dumps(side, indent=1))
    return side


def learned_guard(st: Dict[str, Any]) -> Dict[str, Any]:
    """Amendment 2's learned-nothing VOID guard (:data:`LEARNED_MARGIN`) on the final own-label validation NLL."""
    final = float(st["hist"][-1]) if st.get("hist") else float("nan")
    uni = float(st.get("uniform_val_nll", float("nan")))
    return {"final_val_nll": final, "uniform_val_nll": uni, "margin": LEARNED_MARGIN,
            "pass": bool(np.isfinite(final) and np.isfinite(uni) and final <= uni - LEARNED_MARGIN - 1e-9)}


def _rows_of(st: Dict[str, Any], out: Path) -> str:
    p = out / "rows_path.txt"
    if not p.exists():
        raise StudentError(f"{p} is missing (train writes it)")
    return p.read_text().strip()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.student", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--rows", required=True)
    t.add_argument("--labels", default=None)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--budget-min", type=float, default=40.0)
    t.add_argument("--threads", type=int, default=THREADS)
    t.add_argument("--device", default="cpu")
    t.add_argument("--limit-steps", type=int, default=None)
    t.add_argument("--registration", required=True, help="results/registration.json (VOID 1, enforced)")
    x = sub.add_parser("export")
    x.add_argument("--human-rows", default=None, help="yardstick: the human rowset (its validation rows)")
    x.add_argument("--selfplay-rows", default=None, help="yardstick: the S-S rows (their validation rows)")
    for p in (t, x):
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    try:
        RS._guard_out(out)
        if a.cmd == "train":
            if (a.arm in LABEL_ARMS) != (a.labels is not None):
                raise StudentError(f"--labels is required for, and only for, the {LABEL_ARMS} arms")
            reg = json.loads(Path(a.registration).read_text())
            out.mkdir(parents=True, exist_ok=True)
            (out / "rows_path.txt").write_text(str(Path(a.rows).resolve()) + "\n")
            return train(a.arm, Path(a.rows), Path(a.labels) if a.labels else None, a.seed, out, a.epochs,
                         a.budget_min, a.threads, a.device, a.limit_steps, registration=reg)
        ys = {}
        if a.human_rows:
            ys["human_val_human_labels"] = (Path(a.human_rows), "human")
        if a.selfplay_rows:
            ys["selfplay_val_pstar_labels"] = (Path(a.selfplay_rows), "selfplay")
        side = export(out, ys)
        print(json.dumps({k: side[k] for k in ("arm", "seed", "zip_sha256", "yardsticks")}), flush=True)
        return 0
    except (StudentError, RS.RowsetError) as exc:
        print(f"[student] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
