r"""THE HUMAN-REPLAY CLONE (R9 of the 2026-11-09 independent review): our architecture cloned on a human rowset TO
CONVERGENCE, the model ``--human-data init:<model>@<dose>`` and ``opp:<model>@<share>`` name.

    python -m main.human_data.clone train    --rows <rows> --seed 5901 --out <dir> [--device cpu|cuda] \
        [--budget-min 50] [--accept-layout <the rowset's layout hash, >= 12 hex>]
    python -m main.human_data.clone export   --out <dir> [--id <model id>]
    python -m main.human_data.clone register --out <dir> [--status measurement]

WHY IT IS NOT THE X59-S STUDENT. ``main.human_data.student`` trains a FIXED 3 epochs (7,041 steps) so its arms stay
matched; its human-label students were still improving when the schedule ended (held-out NLL 1.194 → 1.185 → 1.181
over the last three reads of seed 5901), so X59-S is a null on SHORT fresh-student cloning. This trainer keeps the
student's network, loss, optimiser, batch, clip, precision and batch plan (it imports them) and changes three things:

* **A held-out split with no battle shared with training.** The rowset validates by ACCOUNT, so most validation
  battles also have their other seat in the training rows (X59-S REVIEW G: 71 % of the validation rows). Every
  training row whose battle key is a validation battle's is DROPPED (:func:`split_plan`), and the validation
  battles are halved by the rank of their key: the STOP half drives the schedule and the stop, the REPORT half is
  read once, on the exported checkpoint, and is the number to quote.
* **A plateau schedule that ends in a stop** (:class:`Recipe`). After the linear warm-up the learning rate sits on a
  ladder (3e-4, 1.5e-4, 7.5e-5, 3.75e-5). The STOP half is read every quarter epoch (action NLL and top-1). A read
  IMPROVES when its NLL is at least 0.001 nats under the reference; four reads in a row without one (a full epoch)
  step the ladder down, and a plateau on the last rung stops the run as CONVERGED. A cap of 40 epochs stops it as
  BUDGET, which is recorded as NOT converged.
* **The best weights are the model.** The checkpoint exported is the one with the lowest STOP-half NLL, not the last.

THE DATA-QUALITY FILTER is the rowset's own registered one (``hr_spec_v1.r2``: rated, a result, more than ten
turns, the acting player at 1300 or above, the held-out panels excluded, the quitter's tail and the flagged accounts
dropped; ``designs/endstate/human_data_architecture.md`` §4), plus two guards here: the rows' layout hash must be the
code's (:func:`check_layout`; a mismatch is refused unless the operator names the rowset's hash with
``--accept-layout``, and the waiver is recorded), and the battle-disjoint split above.

``export`` writes the best weights through the student's own export (the project's save path, the reload and the
anchors bare load bit-checked), reads both held-out halves on the exported checkpoint, and writes
``human_model.json``: the ``utils.human_models`` manifest, with the TRAINED tensors (those whose bytes differ from
the seed's fresh build), which are what ``init`` copies. ``register`` installs the directory under
``utils.paths.human_models_dir()`` and writes the committed stub. A run made under any override of the recipe
(``--smoke-*``) is a SMOKE model: it registers only as ``smoke`` and only a ``--debug`` run may use it.

RESUMABLE like the student: ``state.pt`` is rewritten atomically every few minutes; ``train`` returns 3 when its
``--budget-min`` ran out (run it again) and 0 when the recipe stopped.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.human_data import rowset as RS
from main.human_data import student as ST

ARM = "human"                      # the student arm whose targets and export this trainer reuses
SPLIT_FIELDS = ST.TRAIN_FIELDS + ("bkey",)
MIN_ACCEPT = 12                    # hex digits of the layout hash an --accept-layout must name


class CloneError(RuntimeError):
    """A clone's inputs or state that do not match what they claim (a defect: stop)."""


@dataclasses.dataclass(frozen=True)
class Recipe:
    """The convergence recipe (module docstring). The defaults ARE the registered recipe; any other value makes
    the model a SMOKE model."""

    max_epochs: int = 40
    evals_per_epoch: int = 4
    patience: int = 4                                   # reads without an improvement before the ladder steps
    min_delta: float = 1e-3                             # nats of STOP-half NLL
    warmup: int = ST.WARMUP
    lr_ladder: Tuple[float, ...] = (3e-4, 1.5e-4, 7.5e-5, 3.75e-5)
    train_shards: Optional[int] = None                  # smoke: only the first N training shards
    held_out_rows: Optional[int] = None                 # smoke: at most N rows of each held-out half

    @property
    def smoke(self) -> bool:
        return self != REGISTERED

    def as_json(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["lr_ladder"] = list(self.lr_ladder)
        d.update({"batch": ST.BATCH, "clip": ST.CLIP, "optimizer": "AdamW wd 1e-5 eps 1e-5", "dtype": "fp32",
                  "loss": "logsumexp(legal) - logsumexp(target set)", "smoke": self.smoke})
        return d


REGISTERED = Recipe()


def lr_at(step: int, rung: int, recipe: Recipe) -> float:
    return recipe.lr_ladder[rung] * min(1.0, (step + 1) / max(recipe.warmup, 1))


# ------------------------------------------------------------------------------------------------ the guards
def code_layout_hash() -> str:
    """The layout hash of the code's encoder, as ``rowset._code_identity`` records it in a rows manifest."""
    from main.policy_spectrum.bank import encoder_identity

    return hashlib.sha256(json.dumps(encoder_identity(), sort_keys=True).encode()).hexdigest()


def check_layout(man: Dict[str, Any], accept: Optional[str], code: Optional[str] = None) -> Dict[str, Any]:
    """The rows were encoded by THIS encoder, or the operator named the rowset's hash (a recorded waiver)."""
    code = code or code_layout_hash()
    have = man.get("layout_hash")
    if not have:
        raise CloneError("the rows manifest has no layout_hash: the rows' encoder cannot be checked")
    rec = {"rowset_layout_hash": have, "code_layout_hash": code, "match": have == code, "waived": False}
    if have == code:
        return rec
    if accept and len(accept) >= MIN_ACCEPT and have.startswith(accept):
        return {**rec, "waived": True, "accepted_prefix": accept}
    raise CloneError(f"the rows were encoded under layout {have[:16]}, the code's encoder is {code[:16]}: the "
                     "network would train on rows it will never see in play. Re-extract the rowset, or, if the "
                     "difference was MEASURED and accepted, name the rowset's hash: --accept-layout "
                     f"{have[:MIN_ACCEPT]}")


def split_plan(rows: Path, man: Dict[str, Any], man_sha: str, recipe: Recipe,
               cache: Optional[Path] = None) -> Dict[str, Any]:
    """The battle-disjoint split (module docstring): which training rows survive, and the two held-out halves.
    Pure in the rowset and the recipe's two smoke cuts; cached in ``cache`` keyed by the manifest's sha256."""
    key = {"rows_manifest_sha256": man_sha, "train_shards": recipe.train_shards, "held_out_rows": recipe.held_out_rows}
    if cache is not None and cache.exists():
        got = json.loads(cache.read_text())
        if all(got.get(k) == v for k, v in key.items()):
            return got
    vb = np.unique(RS.load_shard(rows, man["val"], ("bkey",))["bkey"])
    shards = man["train_shards"][:recipe.train_shards] if recipe.train_shards else man["train_shards"]
    kept = [int((~np.isin(RS.load_shard(rows, rec, ("bkey",))["bkey"], vb)).sum()) for rec in shards]
    total = sum(int(rec["rows"]) for rec in shards)
    if sum(kept) == 0:
        raise CloneError("every training row shares a battle with the validation rows: nothing to train on")
    out = {**key, "val_battles": int(len(vb)), "stop_battles": int(len(vb[0::2])), "report_battles":
           int(len(vb[1::2])), "val_bkeys_sha256": hashlib.sha256(np.ascontiguousarray(vb).tobytes()).hexdigest(),
           "kept": kept, "train_rows": int(sum(kept)), "dropped_rows": int(total - sum(kept)),
           "rule": "a training row whose battle key is a validation battle's is dropped; validation battles are "
                   "halved by the rank of their key (even = STOP, odd = REPORT)"}
    if cache is not None:
        cache.write_text(json.dumps(out, indent=1))
    return out


def _train_shard(rows: Path, rec: Dict[str, Any], vb: np.ndarray) -> Dict[str, np.ndarray]:
    z = RS.load_shard(rows, rec, SPLIT_FIELDS)
    keep = ~np.isin(z["bkey"], vb)
    return {"rows": z["rows"][keep], "mask": z["mask"][keep], "target": ST.targets(ARM, z, None)[keep]}


def held_out(rows: Path, man: Dict[str, Any], recipe: Recipe) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                                                                       np.ndarray]:
    """``(STOP half, REPORT half, the validation battle keys)``; each half is ``{"rows", "mask", "target"}``."""
    z = RS.load_shard(rows, man["val"], SPLIT_FIELDS)
    vb = np.unique(z["bkey"])
    t = ST.targets(ARM, z, None)
    halves = []
    for keys in (vb[0::2], vb[1::2]):
        ix = np.nonzero(np.isin(z["bkey"], keys))[0][:recipe.held_out_rows]
        halves.append({"rows": z["rows"][ix], "mask": z["mask"][ix], "target": t[ix]})
    if not len(halves[0]["target"]) or not len(halves[1]["target"]):
        raise CloneError("a held-out half is empty: the validation rows hold fewer than two battles")
    return halves[0], halves[1], vb


def read_held_out(pol, V: Dict[str, np.ndarray], device: str, batch: int = 1024) -> Dict[str, Any]:
    """The held-out read: mean set-NLL (float64) and top-1 on the rows whose target is ONE action (a hidden-move
    row's target is a set, so it has no top-1)."""
    import torch as th

    from main.x50_r1.hom import policy_logits_t

    tot, hit, one = 0.0, 0, 0
    with th.no_grad():
        for i in range(0, len(V["target"]), batch):
            x = th.from_numpy(V["rows"][i:i + batch]).to(device)
            m = th.from_numpy(V["mask"][i:i + batch].astype(np.float32)).to(device)
            t = th.from_numpy(V["target"][i:i + batch].astype(np.float32)).to(device)
            lg = policy_logits_t(pol, x, m)
            tot += float(ST.set_nll_t(lg, m, t).double().sum())
            single = t.sum(1) == 1
            pick = lg.float().masked_fill(~m.bool(), float("-inf")).argmax(1)
            hit += int((t.gather(1, pick.view(-1, 1)).squeeze(1).bool() & single).sum())
            one += int(single.sum())
    n = len(V["target"])
    return {"rows": n, "nll": tot / max(n, 1), "top1": hit / max(one, 1), "top1_rows": one,
            "uniform_nll": ST.uniform_nll(V)}


# ------------------------------------------------------------------------------------------------ the schedule
def advance(sched: Dict[str, Any], nll: float, recipe: Recipe) -> Optional[str]:
    """One held-out read into the plateau schedule ``{"rung", "bad", "ref"}`` (mutated). Returns ``"CONVERGED"``
    when the last rung plateaued, else None. Pure: the whole stop rule is this function."""
    if sched["ref"] is None or nll < sched["ref"] - recipe.min_delta:
        sched["ref"], sched["bad"] = nll, 0
        return None
    sched["bad"] += 1
    if sched["bad"] < recipe.patience:
        return None
    if sched["rung"] + 1 < len(recipe.lr_ladder):
        sched["rung"], sched["bad"] = sched["rung"] + 1, 0
        return None
    return "CONVERGED"


def train(rows: Path, seed: int, out: Path, recipe: Recipe = REGISTERED, budget_min: float = 0.0,
          threads: int = ST.THREADS, device: str = "cpu", accept_layout: Optional[str] = None,
          limit_steps: Optional[int] = None) -> int:
    """One resumable chunk of the recipe (module docstring). 0 = the recipe stopped, 3 = budget reached."""
    import torch as th

    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals
    from main.x50_r1.hom import policy_logits_t

    out.mkdir(parents=True, exist_ok=True)
    man, man_sha = RS.load_manifest(rows)
    layout = check_layout(man, accept_layout)
    split = split_plan(rows, man, man_sha, recipe, out / "split.json")
    shards = man["train_shards"][:len(split["kept"])]
    pl = ST.plan(split["kept"], seed, recipe.max_epochs)
    total = sum(nb for *_, nb in pl)
    per_epoch = total // recipe.max_epochs
    evs = {int(round(per_epoch * (e + q / recipe.evals_per_epoch)))
           for e in range(recipe.max_epochs) for q in range(1, recipe.evals_per_epoch + 1)} | {total}
    t0 = time.monotonic()
    with inference_globals(threads, device):
        model, _, _ = build_fresh_model(seed)
        pol = model.policy.to(device)
        pol.eval()
        init_digest = ST._state_digest(pol.state_dict())
        params = [p for p in pol.parameters() if p.requires_grad]
        opt = th.optim.AdamW(params, lr=recipe.lr_ladder[0], weight_decay=1e-5, eps=1e-5)
        sp = out / "state.pt"
        ident = {"arm": ARM, "seed": seed, "epochs": recipe.max_epochs, "rows_manifest_sha256": man_sha,
                 "labels_manifest_sha256": None, "init_sha256": init_digest, "total": total,
                 "recipe": recipe.as_json(), "rowset": man.get("rowset"), "layout": layout,
                 "val_bkeys_sha256": split["val_bkeys_sha256"]}
        st: Dict[str, Any] = {**ident, "schema": "gen3_human_clone_state_v1", "step": 0, "hist": [], "hist_steps": [],
                              "top1_hist": [], "lr_hist": [], "done": False, "train_s": 0.0, "stop": None,
                              "sched": {"rung": 0, "bad": 0, "ref": None}, "best": None, "per_epoch": per_epoch}
        if sp.exists():
            st = th.load(sp, weights_only=False, map_location=device)
            if any(st.get(k) != v for k, v in ident.items()):
                raise CloneError(f"{sp} was trained under other rows / seed / init / recipe / layout waiver")
            pol.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
        if st["done"]:
            return 0
        S, _, vb = held_out(rows, man, recipe)
        st.setdefault("uniform_val_nll", ST.uniform_nll(S))
        st.setdefault("device_info", {"device": device, "torch": th.__version__, "cuda_device":
                                      th.cuda.get_device_name(0) if device.startswith("cuda") else None})

        def save() -> None:
            st["model"] = pol.state_dict()
            st["opt"] = opt.state_dict()
            ST._atomic_save(st, sp)

        def read() -> Optional[str]:
            r = read_held_out(pol, S, device)
            st["hist"].append(r["nll"])
            st["hist_steps"].append(st["step"])
            st["top1_hist"].append(r["top1"])
            st["lr_hist"].append(lr_at(max(st["step"] - 1, 0), st["sched"]["rung"], recipe))
            if st["best"] is None or r["nll"] < st["best"]["nll"]:
                st["best"] = {"step": st["step"], "nll": r["nll"], "top1": r["top1"],
                              "model": {k: v.detach().clone() for k, v in pol.state_dict().items()}}
            before = st["sched"]["rung"]
            verdict = advance(st["sched"], r["nll"], recipe)
            print(f"[clone] seed {seed} step {st['step']}/{total} (epoch {st['step'] / max(per_epoch, 1):.2f}): STOP "
                  f"NLL {r['nll']:.5f} top-1 {r['top1']:.4f} best {st['best']['nll']:.5f}@{st['best']['step']} rung "
                  f"{before}->{st['sched']['rung']} bad {st['sched']['bad']}" + (f" {verdict}" if verdict else ""),
                  flush=True)
            return verdict

        def finish(reason: str) -> int:
            st["done"] = True
            st["stop"] = {"reason": reason, "converged": reason == "CONVERGED", "step": st["step"],
                          "epochs": st["step"] / max(per_epoch, 1), "rung": st["sched"]["rung"],
                          "best_step": st["best"]["step"], "best_stop_nll": st["best"]["nll"]}
            save()
            print(f"[clone] {reason} at step {st['step']} ({st['stop']['epochs']:.2f} epochs): best STOP NLL "
                  f"{st['best']['nll']:.5f} at step {st['best']['step']}", flush=True)
            return 0

        if not st["hist"]:
            read()
            save()
        last_save = time.monotonic()
        step = 0
        for (e, b, ids, nb) in pl:
            if step + nb <= st["step"]:
                step += nb
                continue
            buf = [_train_shard(rows, shards[i], vb) for i in ids]
            B = {f: np.concatenate([x[f] for x in buf]) for f in ("rows", "mask", "target")}
            del buf
            perm = np.random.default_rng([seed, e, b]).permutation(len(B["target"]))
            for j in range(nb):
                if step < st["step"]:
                    step += 1
                    continue
                ts = time.monotonic()
                ix = perm[j * ST.BATCH:(j + 1) * ST.BATCH]
                x = th.from_numpy(B["rows"][ix]).to(device)
                m = th.from_numpy(B["mask"][ix].astype(np.float32)).to(device)
                t = th.from_numpy(B["target"][ix].astype(np.float32)).to(device)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, st["sched"]["rung"], recipe)
                loss = ST.set_nll_t(policy_logits_t(pol, x, m), m, t).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(params, ST.CLIP)
                opt.step()
                step += 1
                st["step"] = step
                st["train_s"] += time.monotonic() - ts
                if step % ST.LOSS_EVERY == 0:
                    st.setdefault("loss_at", {})[str(step)] = float(loss.detach())
                    st.setdefault("precision", ST.precision_flags())
                    print(f"[clone] seed {seed} step {step}/{total} loss {float(loss):.4f} lr "
                          f"{opt.param_groups[0]['lr']:.2e} {st['train_s'] / step * 1000:.0f} ms/step", flush=True)
                if step in evs and read() == "CONVERGED":
                    return finish("CONVERGED")
                if step == total:
                    return finish("BUDGET")
                if time.monotonic() - last_save > ST.SAVE_EVERY_S:
                    save()
                    last_save = time.monotonic()
                if (limit_steps is not None and step >= limit_steps) or \
                        (budget_min and (time.monotonic() - t0) / 60 > budget_min):
                    save()
                    print(f"[clone] budget: stopping at step {step}; run train again to continue", flush=True)
                    return 3
        return finish("BUDGET")


# ------------------------------------------------------------------------------------------------ export, register
def default_id(rowset: str, seed: int, smoke: bool) -> str:
    return "hclone_" + "".join(c if c.isalnum() else "_" for c in rowset) + f"_s{seed}" + ("_smoke" if smoke else "")


def export(out: Path, model_id: Optional[str] = None) -> Dict[str, Any]:
    """The best weights as a checkpoint + ``human_model.json`` (module docstring). Returns the manifest."""
    import torch as th

    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals, load_checkpoint
    from utils import human_models as HM
    from utils.git import get_git_hash

    st = th.load(out / "state.pt", weights_only=False, map_location="cpu")
    if st.get("schema") != "gen3_human_clone_state_v1" or not st["done"]:
        raise CloneError("the clone's training has not stopped (run train until it returns 0)")
    recipe = Recipe(**{f.name: (tuple(st["recipe"][f.name]) if f.name == "lr_ladder" else st["recipe"][f.name])
                       for f in dataclasses.fields(Recipe)})
    rows = Path(ST._rows_of(st, out))
    man, man_sha = RS.load_manifest(rows)
    if man_sha != st["rows_manifest_sha256"]:
        raise CloneError("the rows at rows_path.txt are not the rows the clone trained on")
    mid = model_id or default_id(str(st["rowset"]), int(st["seed"]), recipe.smoke)
    if not mid.replace("_", "").replace("-", "").isalnum():
        raise CloneError(f"model id {mid!r}: letters, digits, '_' and '-' only (it becomes a label and a file name)")
    best = st["best"]
    bi = st["hist_steps"].index(best["step"])
    view = {**st, "model": best["model"], "step": best["step"], "hist": st["hist"][:bi + 1],
            "hist_steps": st["hist_steps"][:bi + 1]}
    side = ST.export(out, state=view, name=mid)
    if not side["learned_guard"]["pass"]:
        raise CloneError(f"VOID: the clone learned nothing (best STOP NLL {best['nll']:.4f} against the uniform "
                         f"{st['uniform_val_nll']:.4f})")
    zp = Path(side["zip"])
    with inference_globals(ST.THREADS):
        fresh = build_fresh_model(int(st["seed"]))[0].policy.state_dict()
        if ST._state_digest(fresh) != st["init_sha256"]:
            raise CloneError("the seed's fresh build is not the network the clone started from (another commit?): "
                             "the trained tensors cannot be told from the untouched ones")
        trained = sorted(k for k in fresh if not th.equal(fresh[k], best["model"][k]))
        S, R, _ = held_out(rows, man, recipe)
        pol = load_checkpoint(zp).policy
        reads = {"stop": read_held_out(pol, S, "cpu"), "report": read_held_out(pol, R, "cpu")}
    if not trained:
        raise CloneError("VOID: no tensor moved from the fresh build")
    manifest = {
        "schema": HM.SCHEMA, HM.HUMAN_DATA_KEY: True, "id": mid, "rowset": st["rowset"],
        "rows_manifest_sha256": st["rows_manifest_sha256"], "seed": int(st["seed"]), "zip": zp.name,
        "smoke": recipe.smoke, "recipe": recipe.as_json(), "stop": st["stop"], "held_out": reads,
        "split": json.loads((out / "split.json").read_text()), "layout": st["layout"],
        "init_sha256": st["init_sha256"], "trained_keys": trained,
        "trained_keys_sha256": HM.state_digest(best["model"], trained), "n_tensors": len(fresh),
        "train_s": st["train_s"], "device_info": st.get("device_info"), "commit": get_git_hash(),
        "val_hist": [[int(s), float(v), float(t), float(lr)] for s, v, t, lr in
                     zip(st["hist_steps"], st["hist"], st["top1_hist"], st["lr_hist"])]}
    (zp.parent / HM.MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
    return manifest


def register(out: Path, status: Optional[str] = None) -> Dict[str, Any]:
    """Install ``<out>/model`` under the human-model store and write its committed stub."""
    from utils import human_models as HM

    src = out / "model"
    got = HM.measure(src)
    dst = HM.model_dir(got["id"])
    if dst.exists():
        raise CloneError(f"{dst} exists: a registered model is never overwritten (export under another --id)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return {"id": got["id"], "dir": str(dst),
            **HM.register(dst, status or (HM.SMOKE_STATUS if got["smoke"] else "measurement"))}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.clone", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--rows", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--budget-min", type=float, default=50.0, help="stop this chunk after N minutes (0 = never)")
    t.add_argument("--threads", type=int, default=ST.THREADS)
    t.add_argument("--device", default="cpu")
    t.add_argument("--accept-layout", default=None, help="the rowset's layout hash (>= 12 hex): a recorded waiver")
    t.add_argument("--limit-steps", type=int, default=None)
    for f in dataclasses.fields(Recipe):
        if f.name != "lr_ladder":
            t.add_argument(f"--smoke-{f.name.replace('_', '-')}", dest=f"smoke_{f.name}", default=None,
                           type=float if f.name == "min_delta" else int,
                           help="OVERRIDES the registered recipe: the result is a SMOKE model")
    x = sub.add_parser("export")
    x.add_argument("--id", default=None)
    r = sub.add_parser("register")
    r.add_argument("--status", default=None)
    for p in (t, x, r):
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    try:
        RS._guard_out(out)
        if a.cmd == "train":
            over = {f.name: getattr(a, f"smoke_{f.name}") for f in dataclasses.fields(Recipe)
                    if f.name != "lr_ladder" and getattr(a, f"smoke_{f.name}") is not None}
            out.mkdir(parents=True, exist_ok=True)
            (out / "rows_path.txt").write_text(str(Path(a.rows).resolve()) + "\n")
            return train(Path(a.rows), a.seed, out, dataclasses.replace(REGISTERED, **over), a.budget_min, a.threads,
                         a.device, a.accept_layout, a.limit_steps)
        if a.cmd == "export":
            m = export(out, a.id)
            print(json.dumps({k: m[k] for k in ("id", "smoke", "stop", "held_out")}), flush=True)
            return 0
        print(json.dumps(register(out, a.status)), flush=True)
        return 0
    except (CloneError, ST.StudentError, RS.RowsetError, ValueError) as exc:
        print(f"[clone] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
