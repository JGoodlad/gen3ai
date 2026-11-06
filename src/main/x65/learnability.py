r"""X65-R2's LEARNABILITY PROBE (DESCRIPTIVE: no registration, no verdict; the design and its readings are
``designs/research_state/measurements/x65_learnability_probe_2026-11-06/README.md``). Can a student fit S′'s
departures from P\* on rows it trains on, and do they carry over to rows it never saw? Against the same question on a
LEARNABLE control target of the same departure rate.

    python -m main.x65.student probe …          # a trained arm (GPU): writes rows.npz + eval_s<step>.npz
    python -m main.x65.learnability dump --bank <bank> --bank-sha256 <registered> --fold <read panel> \
        --model <student zip> --out <dir>         # a model scored as it is (CPU): rows.npz + eval_final.npz
    python -m main.x65.learnability report --arm NAME=DIR [--arm …] --out <json> [--l2 <l2_exposure.json.gz>]

**The row sets** of an arm whose read panel is ``fold``: ``train`` (the label rows it trains on), ``val`` (the
training panels' inner-validation label rows: the same teams, deals it never saw) and ``read`` (every label row of
the read panel: teams it never saw; loaded for scoring only, never batched). On each row ``pick`` is the arm's label
action (S′'s ``played``; a control arm's: its target policy's masked argmax), ``base`` is P\*'s recorded greedy
``pi_argmax``, a DEPARTURE is ``pick != base``.

**The meters** (per set, per checkpoint; ``report``): D = the student's agreement with ``pick`` at departures,
K = at the other rows, Δagree = mean of 1[student = pick] − 1[base = pick] (the X65 gate's quantity), the student's own
departure rate and its PRECISION (agreement with ``pick`` where the student leaves ``base``; Δagree > 0 needs it
above about one half). A control arm is also read against S′'s ``played`` (``vs_sprime``): the precision an unrelated
policy change reaches on S′'s departures. The checkpoint read is
the one with the highest ``val`` Δagree (ties to the earlier), never chosen on ``read``; the last and the best-on-read
(optimistic, labelled so) are reported too. Intervals bootstrap DEALS (B = 2,000, ``default_rng(20261106)``).
Splits of the departures: the bank's per-row leak counters (``attribution_miss`` ≥ 1, ``rewritten_moves`` ≥ 1,
neither), S′'s q margin ``q[pick] − q[base]`` (bins :data:`Q_BINS`, half-open), and, with ``--l2``, the per-decision
persisted-dice exposure (sleep, confusion, a pending Future Sight).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from main.x65 import bank as BK
from main.x65 import student as ST

SCHEMA = "gen3_x65_learnability_v1"
TRAIN_FIELDS = ST.FIELDS + ("key", "played", "pi_argmax", "q", "leak_attr", "leak_rewr", "turn")
READ_FIELDS = ("obs", "mask", "label", "pi_prime", "inner_val", "deal", "key", "played", "pi_argmax", "q",
               "leak_attr", "leak_rewr", "turn")
META = ("key", "deal", "played", "pi_argmax", "mask", "leak_attr", "leak_rewr", "turn", "panel")
SETS = ("train", "val", "read")
Q_BINS = (0.03, 0.10)
B_BOOT = 2000
SEED = 20261106
CHUNK = 8192


# ------------------------------------------------------------------------------------------------ the arm's rows
def check_probe(probe: dict[str, Any], fold: str, lr: float) -> dict[str, Any]:
    """The probe's knobs, checked and normalised (the dict a resumed state must match)."""
    if not (isinstance(lr, float) and math.isfinite(lr) and lr > 0):
        raise ST.StudentError(f"probe learning rate {lr!r}: a positive float")
    beta = float(probe["beta"])
    if not (math.isfinite(beta) and beta >= 0):
        raise ST.StudentError(f"probe beta {beta!r}: finite and >= 0")
    epochs, evals = int(probe["epochs"]), int(probe["evals_per_epoch"])
    if epochs < 1 or evals < 1:
        raise ST.StudentError("probe epochs and evals per epoch: >= 1")
    allowed = ST.train_panels(fold)
    tp = list(probe.get("train_panels") or allowed)
    if not tp or not set(tp) <= set(allowed):
        raise ST.StudentError(f"probe train panels {tp}: a non-empty subset of {list(allowed)} (never the read panel)")
    tm = probe.get("target_model")
    return {"schema": SCHEMA, "name": str(probe["name"]), "beta": beta, "epochs": epochs, "evals_per_epoch": evals,
            "train_panels": [p for p in allowed if p in tp],
            "target_model": str(tm) if tm else None,
            "target_sha256": BK.sha256_file(Path(tm)) if tm else None}


def restrict(z: dict[str, np.ndarray], ix: dict[str, np.ndarray], panels: list[str]) -> dict[str, np.ndarray]:
    """``ix`` kept to the rows of ``panels``."""
    keep = np.isin(z["panel"], [BK.U_PANELS.index(p) for p in panels])
    out = {k: v[keep[v]] for k, v in ix.items()}
    if not all(len(v) for v in out.values()):
        raise ST.StudentError(f"an empty split on {panels}: { {k: len(v) for k, v in out.items()} }")
    return out


def read_rows(bank: Path, fold: str) -> dict[str, np.ndarray]:
    """The read panel's LABEL rows (the searched arm's verified searched decisions), for scoring only."""
    zr = BK.load_rows(bank, [fold], READ_FIELDS, labels=[BK.LABELS["searched"]])
    lab = zr["label"].astype(bool)
    return {k: v[lab] for k, v in zr.items()}


def masked_argmax(lp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.argmax(np.where(mask.astype(bool), lp.astype(np.float64), -np.inf), axis=1).astype(np.int8)


def _sets(z, ix, zr) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray]]:
    return {"train": (z, ix["train_label"]), "val": (z, ix["val_label"]), "read": (zr, np.arange(len(zr["key"])))}


def write_rows(d: Path, sets, init_argmax: dict[str, np.ndarray], target: dict[str, np.ndarray] | None,
               meta: dict[str, Any]) -> None:
    """``rows.npz`` (per set: the bank's fields, ``pick``, ``base``, P\\*'s argmax on the arm's own forward and S′'s
    q margin) and ``probe.json``."""
    arr: dict[str, np.ndarray] = {}
    for s, (zz, rows) in sets.items():
        for f in META:
            arr[f"{s}__{f}"] = zz[f][rows]
        base = zz["pi_argmax"][rows].astype(np.int8)
        pick = zz["played"][rows].astype(np.int8) if target is None else masked_argmax(target[s], zz["mask"][rows])
        q = zz["q"][rows]
        i = np.arange(len(rows))
        arr[f"{s}__pick"], arr[f"{s}__base"], arr[f"{s}__init_argmax"] = pick, base, init_argmax[s]
        arr[f"{s}__q_margin"] = np.nan_to_num(q[i, pick] - q[i, base], nan=0.0)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_npz(d / "rows.npz", arr)
    (d / "probe.json").write_text(json.dumps(meta, indent=1))


def _atomic_npz(path: Path, arr: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arr)  # type: ignore[arg-type]
    os.replace(tmp, path)


def _logp_rows(pol, obs: np.ndarray, mask: np.ndarray, rows: np.ndarray, device: str) -> np.ndarray:
    parts = [ST.policy_logp(pol, obs[rows[i:i + CHUNK]], mask[rows[i:i + CHUNK]], device)
             for i in range(0, len(rows), CHUNK)]
    return np.concatenate(parts, 0) if parts else np.zeros((0, BK.N_ACT), np.float32)


def prepare(bank: Path, fold: str, z, ix, tlp: np.ndarray, teacher, probe: dict[str, Any], d: Path, device: str):
    """A trained arm's scoring rows (written once), and, on a control arm, ``z["pi_prime"]`` REPLACED in memory by
    the target policy's masked softmax (the label rows' only target). Returns the sets ``write_eval`` scores."""
    zr = read_rows(bank, fold)
    sets = _sets(z, ix, zr)
    init = {s: masked_argmax(tlp[rows], zz["mask"][rows]) if s != "read" else None for s, (zz, rows) in sets.items()}
    init["read"] = masked_argmax(_logp_rows(teacher.policy, zr["obs"], zr["mask"], sets["read"][1], device),
                                 zr["mask"])
    target = None
    if probe["target_model"]:
        from main.policy_spectrum.reader import load_checkpoint

        tm = load_checkpoint(Path(probe["target_model"]), device=device)
        tm.policy.eval()
        tz = ST.teacher_logp(tm, z["obs"], z["mask"], device)
        target = {s: tz[rows] for s, (zz, rows) in sets.items() if s != "read"}
        target["read"] = _logp_rows(tm.policy, zr["obs"], zr["mask"], sets["read"][1], device)
        del tm
        z["pi_prime"] = np.where(z["mask"] == 1, np.exp(tz.astype(np.float64)), 0.0)
    write_rows(d, sets, init, target, {"kind": "trained", "fold": fold, **probe})
    return sets


def write_eval(d: Path, step: int, pol, sets, device: str) -> None:
    """``eval_s<step>.npz``: per set the policy's masked argmax (int8, from float32) and log-softmax (float16)."""
    arr = {}
    for s, (zz, rows) in sets.items():
        lp = _logp_rows(pol, zz["obs"], zz["mask"], rows, device)
        arr[f"{s}__argmax"] = masked_argmax(lp, zz["mask"][rows])
        arr[f"{s}__logp"] = lp.astype(np.float16)
    _atomic_npz(d / f"eval_s{step:06d}.npz", arr)


def dump(bank: Path, bank_sha256: str, fold: str, model: Path, out: Path, threads: int = 2) -> dict[str, Any]:
    """A model scored AS IT IS on a fold's three sets, through the gate's forward (``delivery._forward``)."""
    from main.x65 import delivery as DV

    BK.check(bank, bank_sha256)
    z, ix = ST.load_fold(bank, fold, TRAIN_FIELDS)
    zr = read_rows(bank, fold)
    sets = _sets(z, ix, zr)
    arr = {}
    for s, (zz, rows) in sets.items():
        mask = zz["mask"][rows]
        lg = DV._forward(model, zz["obs"][rows], mask, threads)[:, :BK.N_ACT].astype(np.float64)
        lg = np.where(mask.astype(bool), lg, -np.inf)
        lp = lg - lg.max(1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
        arr[f"{s}__argmax"] = masked_argmax(lp, mask)
        arr[f"{s}__logp"] = lp.astype(np.float16)
    init = {s: zz["pi_argmax"][rows].astype(np.int8) for s, (zz, rows) in sets.items()}   # the recorded greedy
    write_rows(out, sets, init, None, {"schema": SCHEMA, "kind": "dump", "fold": fold, "model": str(model),
                                       "model_sha256": BK.sha256_file(model), "init_argmax": "recorded"})
    _atomic_npz(out / "eval_final.npz", arr)
    return {s: len(r) for s, (_, r) in sets.items()}


# ------------------------------------------------------------------------------------------------ the meters
def meters(a: np.ndarray, pick: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    dep = pick != base
    hit = a == pick
    leave = a != base
    def mean(x, m):
        return float(x[m].mean()) if m.any() else None
    return {"n": int(len(a)), "departures": int(dep.sum()), "dep_rate": float(dep.mean()) if len(a) else None,
            "D": mean(hit, dep), "K": mean(hit, ~dep),
            "delta": float(hit.mean() - (base == pick).mean()) if len(a) else None,
            "student_dep_rate": float(leave.mean()) if len(a) else None, "precision": mean(hit, leave)}


def ratio_ci(deal: np.ndarray, num: np.ndarray, den: np.ndarray, B: int = B_BOOT, seed: int = SEED) -> list[float]:
    """``[estimate, 2.5 %, 97.5 %]`` of sum(num) / sum(den), deals resampled with replacement."""
    keys, inv = np.unique(deal, return_inverse=True)
    sn = np.bincount(inv, weights=num.astype(np.float64), minlength=len(keys))
    sd = np.bincount(inv, weights=den.astype(np.float64), minlength=len(keys))
    if sd.sum() == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        j = rng.integers(0, len(keys), len(keys))
        reps[b] = sn[j].sum() / max(sd[j].sum(), 1e-12)
    return [float(sn.sum() / sd.sum()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def intervals(a, pick, base, deal) -> dict[str, list[float]]:
    dep, hit = pick != base, (a == pick)
    ones = np.ones(len(a))
    return {"D": ratio_ci(deal, hit & dep, dep), "K": ratio_ci(deal, hit & ~dep, ~dep),
            "delta": ratio_ci(deal, hit.astype(np.int8) - (base == pick).astype(np.int8), ones)}


def splits(a, pick, base, r: dict[str, np.ndarray], s: str, l2: dict[str, Any] | None) -> dict[str, Any]:
    """The departures' agreement and the rows' Δagree, per bucket (reported, never decided)."""
    qm = r[f"{s}__q_margin"]
    attr, rewr = r[f"{s}__leak_attr"] >= 1, r[f"{s}__leak_rewr"] >= 1
    dep = pick != base                  # the q-margin bins hold departures only (a keep's margin is 0)
    b = {"attribution_miss": attr, "rewritten_moves": rewr, "neither_counter": ~(attr | rewr),
         f"departures_q_margin<{Q_BINS[0]}": dep & (qm < Q_BINS[0]),
         f"departures_{Q_BINS[0]}<=q_margin<{Q_BINS[1]}": dep & (qm >= Q_BINS[0]) & (qm < Q_BINS[1]),
         f"departures_q_margin>={Q_BINS[1]}": dep & (qm >= Q_BINS[1])}
    if l2 is not None:
        keys = r[f"{s}__key"]
        known = np.array([k in l2["scanned"] for k in keys])
        hitk = np.array([k in l2["exposed"] for k in keys])
        b["l2_exposed"], b["l2_not_exposed"], b["l2_unscanned"] = known & hitk, known & ~hitk, ~known
    out = {}
    for k, m in b.items():
        mm = meters(a[m], pick[m], base[m])
        out[k] = {f: mm[f] for f in ("n", "departures", "D", "K", "delta", "precision")}
    return out


def load_l2(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        doc = json.load(fh)
    return {"scanned": set(doc["scanned"]), "exposed": set(doc["exposed"])}


def _evals(d: Path) -> list[tuple[int | None, Path]]:
    out: list[tuple[int | None, Path]] = []
    for p in sorted(d.glob("eval_*.npz")):
        tag = p.stem.split("_", 1)[1]
        out.append((None if tag == "final" else int(tag[1:]), p))
    return sorted(out, key=lambda t: -1 if t[0] is None else t[0])


def report_arm(d: Path, l2: dict[str, Any] | None = None) -> dict[str, Any]:
    with np.load(d / "rows.npz") as z:
        r = {k: z[k] for k in z.files}
    meta = json.loads((d / "probe.json").read_text())
    evs = _evals(d)
    if not evs:
        raise ST.StudentError(f"{d}: no eval_*.npz")
    curve, loaded = [], {}
    for step, p in evs:
        with np.load(p) as z:
            loaded[step] = {s: z[f"{s}__argmax"] for s in SETS}
        c = {"step": step, **{s: meters(loaded[step][s], r[f"{s}__pick"], r[f"{s}__base"]) for s in SETS}}
        if meta.get("target_model"):        # a control arm, read against S′ too: the chance level of a policy change
            c["vs_sprime"] = {s: meters(loaded[step][s], r[f"{s}__played"], r[f"{s}__base"]) for s in SETS}
        curve.append(c)
    init = {s: meters(r[f"{s}__init_argmax"], r[f"{s}__pick"], r[f"{s}__base"]) for s in SETS}
    cand = [c for c in curve if c["step"] != 0]
    sel = max(cand, key=lambda c: (c["val"]["delta"], -(c["step"] or 0)))
    best_read = max(cand, key=lambda c: (c["read"]["delta"], -(c["step"] or 0)))
    at = {}
    for tag, c in (("selected_on_val", sel), ("last", cand[-1]), ("best_on_read_OPTIMISTIC", best_read)):
        a = loaded[c["step"]]
        at[tag] = {"step": c["step"],
                   **{s: {"meters": c[s], "ci": intervals(a[s], r[f"{s}__pick"], r[f"{s}__base"], r[f"{s}__deal"]),
                          "splits": splits(a[s], r[f"{s}__pick"], r[f"{s}__base"], r, s, l2)} for s in SETS}}
    return {"dir": str(d), "probe": meta, "init_path_check": init, "curve": curve, "at": at}


def report(arms: dict[str, Path], l2: Path | None = None) -> dict[str, Any]:
    l2d = load_l2(l2) if l2 else None
    return {"schema": SCHEMA, "descriptive": "X65-R2's learnability probe: no registration, no verdict",
            "q_bins": list(Q_BINS), "B": B_BOOT, "seed": SEED, "l2": str(l2) if l2 else None,
            "arms": {n: report_arm(d, l2d) for n, d in arms.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x65.learnability", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--bank", required=True)
    d.add_argument("--bank-sha256", required=True)
    d.add_argument("--fold", required=True, choices=sorted(ST.FOLDS))
    d.add_argument("--model", required=True)
    d.add_argument("--threads", type=int, default=2)
    r = sub.add_parser("report")
    r.add_argument("--arm", action="append", required=True, help="NAME=DIR")
    r.add_argument("--l2", default=None)
    for q in (d, r):
        q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "dump":
            print(json.dumps(dump(Path(a.bank), a.bank_sha256, a.fold, Path(a.model), Path(a.out), a.threads)),
                  flush=True)
            return 0
        arms = {}
        for spec in a.arm:
            name, _, path = spec.partition("=")
            if not name or not path:
                raise ST.StudentError(f"--arm {spec!r}: NAME=DIR")
            arms[name] = Path(path)
        doc = report(arms, Path(a.l2) if a.l2 else None)
        Path(a.out).write_text(json.dumps(doc, indent=1))
        for n, arm in doc["arms"].items():
            s = arm["at"]["selected_on_val"]
            print(json.dumps({"arm": n, "step": s["step"], **{k: {m: s[k]["meters"][m] for m in ("D", "K", "delta")}
                                                              for k in SETS}}), flush=True)
        return 0
    except (ST.StudentError, BK.BankError) as exc:
        print(f"[x65 learnability] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
