r"""B2, THE FINE-TUNE (PREREG §3): one P\* student per panel fold, trained offline toward S′'s piKL targets on the two
TRAINING panels' label rows with a KL anchor to P\* on their anchor rows; the inner validation picks among two learning
rates and sixteen checkpoints each, blind to the read panel.

    python -m main.x65.student train --bank <bank> --bank-sha256 <registered> --fold heldout72_hpfix --lr 1e-5 \
        --init <P* zip> --out <dir>/s_h [--device cuda] [--budget-min 50]
    python -m main.x65.student pick --out <dir>/s_h

**The folds** (:data:`FOLDS`): a student is named by its READ panel and trains on the other two; its training seed is
fixed (S_H 6511, S_F 6512, S_P 6513). No row of the read panel is loaded, and :func:`split` refuses a training or
inner-validation row whose panel, or whose ``deal``, is the read panel's.

**The recipe** (registered, nothing here is a knob): start from P\*'s weights and architecture; the loss is
``mean over label rows KL(π′ ‖ π_θ) + β · mean over anchor rows KL(π_P* ‖ π_θ)``, β = 1, both over the legal actions,
π_θ the policy's masked log-softmax (``main.x50_r1.hom.policy_logits_t``), π_P\* computed here from P\*'s own bytes
(a frozen copy, once, before the first step); every decision row is an anchor row (a label row too, when S′ searched
it). A step takes 256 label rows and 256 anchor rows, each from that epoch's permutation (``default_rng([seed, 0,
epoch])`` for the label rows, ``[seed, 1, epoch]`` for the anchor rows), so each is drawn without replacement within
an epoch; an epoch is ``ceil(train label rows / 256)`` steps (the last label batch is the remainder), 4 epochs.
AdamW (weight decay 1e-5, eps 1e-5), lr linear warm-up over 200 steps then constant, gradient-norm clip 0.5, fp32
(TF32 off on CUDA), eval mode. Only the parameters the policy logits depend on take gradient (found by one backward
through ``policy_logits_t``; the rest, the value head's own among them, are frozen: ``requires_grad = False``).

**The inner validation**: the training panels' rows whose deal is inner-validation (``bank.inner_val``) are withheld
from training; at every quarter epoch (16 checkpoints per learning rate) the same L is computed on them in float64,
and the checkpoint's policy weights are saved. ``pick`` takes, over both learning rates' 32 checkpoints, the lowest L,
ties to the lower learning rate and then the earlier checkpoint, and records all 32 values.

RESUMABLE: ``state.pt`` (model, optimiser, step, the L history) is written atomically at every checkpoint and every
few minutes; a state trained under another bank, init, fold, learning rate or plan is refused. The bank must be the
REGISTERED one (``--bank-sha256``, the pin amendment's): ``bank.check`` refuses any other, or a VOID bank.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# the X59-S student path's own helpers (one copy): the policy digest, the atomic state write, the precision record
from main.human_data.student import _atomic_save, _state_digest, precision_flags
from main.x65 import bank as BK

#: read panel -> (student, training seed)
FOLDS = {"heldout72_hpfix": ("S_H", 6511), "forum72": ("S_F", 6512), "p72": ("S_P", 6513)}
LRS = (1e-5, 3e-5)
EPOCHS = 4
BATCH_LABEL = 256
BATCH_ANCHOR = 256
BETA = 1.0
WARMUP = 200
WEIGHT_DECAY = 1e-5
EPS = 1e-5
CLIP = 0.5
EVALS_PER_EPOCH = 4
THREADS = 2
SAVE_EVERY_S = 600.0
#: P\*'s bytes (PREREG abbreviations)
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
FIELDS = ("obs", "mask", "label", "pi_prime", "inner_val", "deal")


class StudentError(RuntimeError):
    """A student's inputs or state that do not match what they claim (a defect: stop)."""


def train_panels(fold: str) -> tuple[str, ...]:
    if fold not in FOLDS:
        raise StudentError(f"fold {fold!r}: one of {sorted(FOLDS)}")
    return tuple(p for p in BK.U_PANELS if p != fold)


def split(z: dict[str, np.ndarray], fold: str, read_deals: Sequence[str]) -> dict[str, np.ndarray]:
    """Index arrays into ``z`` (the training panels' rows): ``train_label`` / ``train_anchor`` / ``val_label`` /
    ``val_anchor``. REFUSES a row of the read panel ``fold`` or one whose deal is a read-panel deal (PREREG §3, §10:
    a student whose inner validation contains a read panel's deal is VOID)."""
    read_ix = BK.U_PANELS.index(fold)
    if (z["panel"] == read_ix).any():
        raise StudentError(f"VOID: a row of the read panel {fold} among {fold}'s training rows")
    leaked = set(np.unique(z["deal"]).tolist()) & set(read_deals)
    if leaked:
        raise StudentError(f"VOID: {len(leaked)} read-panel deal(s) among the training panels' rows")
    iv = z["inner_val"].astype(bool)
    lab = z["label"].astype(bool)
    out = {"train_label": np.flatnonzero(lab & ~iv), "train_anchor": np.flatnonzero(~iv),
           "val_label": np.flatnonzero(lab & iv), "val_anchor": np.flatnonzero(iv)}
    if not all(len(v) for v in out.values()):
        raise StudentError(f"an empty split: { {k: len(v) for k, v in out.items()} }")
    return out


def plan(n_label: int, n_anchor: int, seed: int, epochs: int = EPOCHS) -> tuple[int, list[int]]:
    """``(steps per epoch, the checkpoint steps)``: every quarter epoch, the run's end included."""
    if n_anchor < BATCH_ANCHOR:
        raise StudentError(f"{n_anchor} anchor rows: fewer than one anchor batch")
    per = math.ceil(n_label / BATCH_LABEL)
    cks = [round(per * (e + q / EVALS_PER_EPOCH)) for e in range(epochs) for q in range(1, EVALS_PER_EPOCH + 1)]
    if cks[0] < 1 or any(b <= a for a, b in itertools.pairwise(cks)):
        raise StudentError(f"{n_label} label rows: too few for {len(cks)} distinct checkpoints")
    return per, cks


def batch_at(step: int, per: int, n_label: int, n_anchor: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(label positions, anchor positions)`` of optimiser step ``step``: pure in the step, the sizes and the seed."""
    e, j = divmod(step, per)
    lp = np.random.default_rng([seed, 0, e]).permutation(n_label)
    ap = np.random.default_rng([seed, 1, e]).permutation(n_anchor)
    return lp[j * BATCH_LABEL:(j + 1) * BATCH_LABEL], ap[j * BATCH_ANCHOR:(j + 1) * BATCH_ANCHOR]


def lr_at(step: int, lr: float) -> float:
    return lr * (step + 1) / WARMUP if step < WARMUP else lr


def masked_log_softmax_t(logits, mask):
    import torch as th

    return th.log_softmax(logits.float().masked_fill(~mask.bool(), float("-inf")), dim=-1)


def kl_t(target, logp, mask):
    """KL(target ‖ exp(logp)) per row over the legal actions (``target`` 0 off them)."""
    import torch as th

    t = target.masked_fill(~mask.bool(), 0.0)
    lp = logp.masked_fill(~mask.bool(), 0.0)
    return (th.xlogy(t, t) - t * lp).sum(-1)


def logit_path_params(pol, x, m) -> list[str]:
    """The names of the parameters the policy logits depend on (one backward through ``policy_logits_t``)."""
    from main.x50_r1.hom import policy_logits_t

    for p in pol.parameters():
        p.grad = None
    policy_logits_t(pol, x, m).float().sum().backward()
    names = [n for n, p in pol.named_parameters() if p.grad is not None]
    for p in pol.parameters():
        p.grad = None
    return names


def sha256_file(p: Path) -> str:
    return BK.sha256_file(p)


def teacher_logp(model, obs: np.ndarray, mask: np.ndarray, device: str, batch: int = 1024) -> np.ndarray:
    """P\\*'s masked log-softmax on every row, float32 (no gradient; the frozen copy)."""
    import torch as th

    from main.x50_r1.hom import policy_logits_t

    out = []
    with th.no_grad():
        for i in range(0, len(obs), batch):
            x = th.from_numpy(obs[i:i + batch]).to(device)
            m = th.from_numpy(mask[i:i + batch].astype(np.float32)).to(device)
            out.append(masked_log_softmax_t(policy_logits_t(model.policy, x, m), m).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def loss_t(pol, z, tlp, il, ia, device: str):
    """``(L, label KL mean, anchor KL mean)`` on label rows ``il`` and anchor rows ``ia`` (tensors, with gradient)."""
    import torch as th

    from main.x50_r1.hom import policy_logits_t

    def part(ix, target):
        x = th.from_numpy(z["obs"][ix]).to(device)
        m = th.from_numpy(z["mask"][ix].astype(np.float32)).to(device)
        lp = masked_log_softmax_t(policy_logits_t(pol, x, m), m)
        return kl_t(th.from_numpy(target).to(device), lp, m)

    kl_lab = part(il, z["pi_prime"][il].astype(np.float32)).mean()
    tgt = np.exp(tlp[ia].astype(np.float64)).astype(np.float32) * (z["mask"][ia] == 1)
    kl_anc = part(ia, tgt).mean()
    return kl_lab + BETA * kl_anc, kl_lab, kl_anc


def eval_L(pol, z, tlp, val_label: np.ndarray, val_anchor: np.ndarray, device: str, batch: int = 1024) -> dict[str, float]:
    """The inner-validation L (float64 means over ALL inner-validation label and anchor rows), no gradient."""
    import torch as th

    from main.x50_r1.hom import policy_logits_t

    def mean_kl(ix, target_of) -> float:
        tot = 0.0
        with th.no_grad():
            for i in range(0, len(ix), batch):
                b = ix[i:i + batch]
                x = th.from_numpy(z["obs"][b]).to(device)
                m = th.from_numpy(z["mask"][b].astype(np.float32)).to(device)
                lp = masked_log_softmax_t(policy_logits_t(pol, x, m), m).double()
                t = th.from_numpy(target_of(b)).to(device)
                tot += float(kl_t(t, lp, m).sum())
        return tot / len(ix)

    lab = mean_kl(val_label, lambda b: z["pi_prime"][b].astype(np.float64))
    anc = mean_kl(val_anchor, lambda b: np.exp(tlp[b].astype(np.float64)) * (z["mask"][b] == 1))
    return {"L": lab + BETA * anc, "kl_label": lab, "kl_anchor": anc}


def load_fold(bank: Path, fold: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    z = BK.load_rows(bank, train_panels(fold), FIELDS)
    read = BK.load_rows(bank, [fold], ("deal",))
    return z, split(z, fold, np.unique(read["deal"]).tolist())


def train(bank: Path, bank_sha256: str, fold: str, lr: float, init: Path, out: Path, device: str = "cpu",
          budget_min: float = 0.0, threads: int = THREADS, limit_steps: int | None = None,
          init_sha256: str = PSTAR_SHA256, epochs: int = EPOCHS) -> int:
    """One resumable chunk of the registered fine-tune of ``fold`` at learning rate ``lr`` (module docstring).
    0 = finished, 3 = budget reached."""
    import torch as th

    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    if lr not in LRS:
        raise StudentError(f"learning rate {lr!r} is not one of the registered {LRS}")
    bk = BK.check(bank, bank_sha256)
    if sha256_file(init) != init_sha256:
        raise StudentError(f"{init}: not P*'s bytes {init_sha256[:12]}")
    name, seed = FOLDS[fold]
    d = out / f"lr{lr:g}"
    d.mkdir(parents=True, exist_ok=True)
    z, ix = load_fold(bank, fold)
    per, cks = plan(len(ix["train_label"]), len(ix["train_anchor"]), seed, epochs)
    total = per * epochs
    ident = {"student": name, "fold": fold, "train_panels": list(train_panels(fold)), "seed": seed, "lr": lr,
             "bank_manifest_sha256": bk["manifest_sha256"], "init_sha256": init_sha256, "epochs": epochs,
             "per_epoch": per, "total": total, "checkpoints": cks,
             "sizes": {k: len(v) for k, v in ix.items()}}
    t0 = time.monotonic()
    with inference_globals(threads, device):
        th.manual_seed(seed)
        model = load_checkpoint(init, device=device)
        teacher = load_checkpoint(init, device=device)
        pol = model.policy
        pol.eval()
        teacher.policy.eval()
        tlp = teacher_logp(teacher, z["obs"], z["mask"], device)
        del teacher
        probe = ix["train_anchor"][:64]
        names = logit_path_params(pol, th.from_numpy(z["obs"][probe]).to(device),
                                  th.from_numpy(z["mask"][probe].astype(np.float32)).to(device))
        keep = set(names)
        for n, p in pol.named_parameters():
            p.requires_grad_(n in keep)
        params = [p for n, p in pol.named_parameters() if n in keep]
        frozen = [n for n, _ in pol.named_parameters() if n not in keep]
        opt = th.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY, eps=EPS)
        sp = d / "state.pt"
        st: dict[str, Any] = {**ident, "step": 0, "hist": [], "done": False, "train_s": 0.0,
                              "trainable": names, "frozen": frozen}
        if sp.exists():
            st = th.load(sp, weights_only=False, map_location=device)
            if any(st.get(k) != v for k, v in ident.items()):
                raise StudentError(f"{sp} was trained under another bank / init / fold / lr / plan")
            if st["trainable"] != names:
                raise StudentError(f"{sp}: another trainable parameter set")
            pol.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
        if st["done"]:
            return 0

        def save() -> None:
            st["model"] = pol.state_dict()
            st["opt"] = opt.state_dict()
            _atomic_save(st, sp)

        if not st["hist"]:
            st["hist"].append({"step": 0, "checkpoint": None, **eval_L(pol, z, tlp, ix["val_label"],
                                                                         ix["val_anchor"], device)})
            st["precision"] = precision_flags()
            st["device_info"] = {"device": device, "torch": th.__version__,
                                 "cuda_device": th.cuda.get_device_name(0) if device.startswith("cuda") else None}
            save()
        last = time.monotonic()
        while st["step"] < total:
            step = st["step"]
            ts = time.monotonic()
            lp_, ap_ = batch_at(step, per, len(ix["train_label"]), len(ix["train_anchor"]), seed)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr)
            loss, kl_lab, kl_anc = loss_t(pol, z, tlp, ix["train_label"][lp_], ix["train_anchor"][ap_], device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            st["step"] = step + 1
            st["train_s"] += time.monotonic() - ts
            if st["step"] % 50 == 0:
                st.setdefault("loss_at", {})[str(st["step"])] = [float(loss), float(kl_lab), float(kl_anc)]
                print(f"[x65 student] {name} lr {lr:g} step {st['step']}/{total} L {float(loss):.5f} "
                      f"(label {float(kl_lab):.5f}, anchor {float(kl_anc):.5f})", flush=True)
            if st["step"] in cks:
                k = cks.index(st["step"])
                ev = eval_L(pol, z, tlp, ix["val_label"], ix["val_anchor"], device)
                ck = d / f"ckpt_{k:02d}.pt"
                _atomic_save({kk: v.detach().cpu() for kk, v in pol.state_dict().items()}, ck)
                st["hist"].append({"step": st["step"], "checkpoint": k, "path": ck.name,
                                   "policy_sha256": _state_digest(pol.state_dict()), **ev})
                print(f"[x65 student] {name} lr {lr:g} checkpoint {k} step {st['step']}: inner-val L {ev['L']:.6f}",
                      flush=True)
                if st["step"] == total:
                    st["done"] = True
                save()
                last = time.monotonic()
                if st["done"]:
                    break
            if time.monotonic() - last > SAVE_EVERY_S:
                save()
                last = time.monotonic()
            if (limit_steps is not None and st["step"] >= limit_steps) or \
                    (budget_min and (time.monotonic() - t0) / 60 > budget_min):
                save()
                print(f"[x65 student] budget: stopping at step {st['step']}; resume to continue", flush=True)
                return 3
        st["done"] = True
        save()
    return 0


def candidates(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 32 candidates (both learning rates' 16 checkpoints) with their inner-validation L."""
    import torch as th

    rows = []
    ident = None
    for lr in LRS:
        sp = out / f"lr{lr:g}" / "state.pt"
        if not sp.exists():
            raise StudentError(f"{sp}: missing")
        st = th.load(sp, weights_only=False, map_location="cpu")
        if not st["done"]:
            raise StudentError(f"{sp}: the fine-tune has not finished")
        mine = {k: st[k] for k in ("student", "fold", "seed", "bank_manifest_sha256", "init_sha256", "epochs")}
        if ident is not None and mine != ident:
            raise StudentError("the two learning rates were trained under different folds / banks / inits")
        ident = mine
        for h in st["hist"]:
            if h["checkpoint"] is not None:
                rows.append({"lr": lr, "checkpoint": int(h["checkpoint"]), "step": int(h["step"]),
                             "L": float(h["L"]), "kl_label": float(h["kl_label"]), "kl_anchor": float(h["kl_anchor"]),
                             "path": str(Path(f"lr{lr:g}") / h["path"]), "policy_sha256": h["policy_sha256"]})
    want = len(LRS) * EPOCHS * EVALS_PER_EPOCH
    if len(rows) != want:
        raise StudentError(f"{len(rows)} candidates, the registration has {want}")
    return rows, ident


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The lowest inner-validation L; ties to the lower learning rate, then the earlier checkpoint."""
    return min(rows, key=lambda r: (r["L"], r["lr"], r["checkpoint"]))


def pick(out: Path) -> dict[str, Any]:
    rows, ident = candidates(out)
    best = choose(rows)
    doc = {"schema": "gen3_x65_pick_v1", **ident, "candidates": rows, "picked": best,
           "rule": "lowest inner-validation L over 2 learning rates x 16 checkpoints; ties to the lower lr, then the "
                   "earlier checkpoint"}
    (out / "pick.json").write_text(json.dumps(doc, indent=1))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x65.student", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--bank", required=True)
    t.add_argument("--bank-sha256", required=True, help="the REGISTERED bank manifest sha256 (the pin amendment's)")
    t.add_argument("--fold", required=True, choices=sorted(FOLDS), help="the READ panel")
    t.add_argument("--lr", type=float, required=True, choices=LRS)
    t.add_argument("--init", required=True, help="P*'s zip")
    t.add_argument("--device", default="cpu")
    t.add_argument("--budget-min", type=float, default=50.0)
    t.add_argument("--threads", type=int, default=THREADS)
    p = sub.add_parser("pick")
    for q in (t, p):
        q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "train":
            return train(Path(a.bank), a.bank_sha256, a.fold, a.lr, Path(a.init), Path(a.out), a.device,
                         a.budget_min, a.threads)
        doc = pick(Path(a.out))
        print(json.dumps(doc["picked"]), flush=True)
        return 0
    except (StudentError, BK.BankError) as exc:
        print(f"[x65 student] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
