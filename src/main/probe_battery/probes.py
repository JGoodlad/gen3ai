"""Cross-validated LINEAR PROBES from captured representations to the battery's facts (`gen3_probe_battery_v1`).

The probe is ridge regression on standardised features (the standardisation fitted on each training fold only), the
ridge strength chosen INSIDE each training fold by generalised cross-validation (the closed-form leave-one-out error
over :data:`ALPHAS`), scored out-of-fold: R² for a continuous fact, ROC-AUC of the ridge score for a binary one.

* **Folds by BATTLE.** A battle's decisions share one fold; the fold is ``sha256(battle id) mod 5`` — a function of
  the id alone, so it never moves with row order, row count or a seed.
* **Control task** (Hewitt & Liang 2019, "Designing and Interpreting Probes with Control Tasks"): every fact gets a
  CONTROL label with the same marginal, assigned as a random function of the species of the mon the fact is about
  (:attr:`facts.Fact.about`; our active's for a fact about no mon), seeded by the fact's name. A probe can score on the
  control only by memorising species identity, so **selectivity = real − control** says how much of the read is the
  fact rather than "which mon is this".
* A fact is scored at a site only where it has :data:`MIN_ROWS` rows and, if binary, :data:`MIN_MINORITY` of each
  class — otherwise ``None`` (too rare to read), never a number from a handful of rows.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ALPHAS = (1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)
N_FOLDS = 5
MIN_ROWS = 300
MIN_MINORITY = 40
DEPTH_NAMES = ("in", "L1", "L2", "L3", "L4")


# ------------------------------------------------------------------------------------------------ folds
def battle_folds(battles: Sequence[str], n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold of each row: ``int(sha256(battle)[:8], 16) % n_folds`` — deterministic, by battle."""
    cache: Dict[str, int] = {}
    out = np.empty(len(battles), dtype=np.int64)
    for i, b in enumerate(battles):
        if b not in cache:
            cache[b] = int(hashlib.sha256(str(b).encode()).hexdigest()[:8], 16) % n_folds
        out[i] = cache[b]
    return out


# --------------------------------------------------------------------------------------------- control
def control_labels(y: np.ndarray, valid: np.ndarray, key: np.ndarray, kind: str, seed_name: str) -> np.ndarray:
    """The control task's labels: each species key gets ONE value drawn from the fact's empirical marginal (a
    continuous fact: a valid row's value; a binary fact: a Bernoulli draw at the base rate), by a generator seeded by
    ``sha256(seed_name)``. Rows with no key (−1) are their own key."""
    rng = np.random.default_rng(int(hashlib.sha256(seed_name.encode()).hexdigest()[:16], 16))
    pool = y[valid]
    out = np.full_like(y, np.nan, dtype=np.float64)
    if len(pool) == 0:
        return out
    uniq = np.unique(key[valid])
    if kind == "b":
        p = float(np.mean(pool))
        table = {int(k): float(rng.random() < p) for k in uniq}
    else:
        table = {int(k): float(pool[rng.integers(0, len(pool))]) for k in uniq}
    out[valid] = [table[int(k)] for k in key[valid]]
    return out


def species_lookup(y: np.ndarray, valid: np.ndarray, key: np.ndarray, folds: np.ndarray, kind: str) -> float:
    """How much of a fact the SPECIES alone gives: out-of-fold (by battle) prediction of ``y`` by the training folds'
    mean of ``y`` per species key (unseen key → the training mean). A property of the bank, not of a model."""
    yy, kk, ff = y[valid], key[valid], folds[valid]
    pred = np.zeros(len(yy))
    for k in np.unique(ff):
        tr, te = ff != k, ff == k
        if tr.sum() == 0:
            continue
        m = float(yy[tr].mean())
        sums: Dict[int, List[float]] = {}
        for a, b in zip(kk[tr], yy[tr]):
            sums.setdefault(int(a), []).append(float(b))
        tab = {a: (sum(v) + m) / (len(v) + 1) for a, v in sums.items()}      # one pseudo-count of the mean
        pred[te] = [tab.get(int(a), m) for a in kk[te]]
    return score(kind, yy, pred)


# ------------------------------------------------------------------------------------------------- metrics
def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss <= 0 else 1.0 - float(np.sum((y - p) ** 2)) / ss


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC by ranks (ties get their average rank)."""
    from scipy.stats import rankdata

    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(kind: str, y: np.ndarray, p: np.ndarray) -> float:
    return r2(y, p) if kind == "c" else auc(y, p)


# --------------------------------------------------------------------------------------------- the ridge
def ridge_oof(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, alphas: Sequence[float] = ALPHAS) -> np.ndarray:
    """Out-of-fold ridge predictions of every column of ``Y`` [n, t] from ``X`` [n, d] (rows already selected).
    Per fold: standardise on the training rows, eigen-decompose the training Gram matrix once, choose α per target
    by the closed-form leave-one-out (GCV-exact) error on the training rows, predict the held-out fold."""
    n, d = X.shape
    Y = Y.reshape(n, -1)
    out = np.zeros_like(Y, dtype=np.float64)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if tr.sum() < 10 or te.sum() == 0:
            out[te] = np.nan
            continue
        Xtr = X[tr].astype(np.float64)
        mu = Xtr.mean(0)
        sd = Xtr.std(0)
        # a (near-)CONSTANT training column carries nothing and, scaled by a tiny sd, would blow a held-out row up:
        # it is zeroed in both folds (a random network's critic pool has such columns)
        dead = sd < 1e-4
        sd[dead] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (X[te].astype(np.float64) - mu) / sd
        Xtr[:, dead] = 0.0
        Xte[:, dead] = 0.0
        lam, Q = np.linalg.eigh(Xtr.T @ Xtr)
        lam = np.clip(lam, 0.0, None)
        Z = Xtr @ Q
        Z2 = Z * Z
        ym = Y[tr].mean(0)
        Yc = Y[tr] - ym
        B = Z.T @ Yc                                    # d x t
        ntr = Xtr.shape[0]
        best = np.full(Y.shape[1], np.inf)
        coef = np.zeros((d, Y.shape[1]))
        for a in alphas:
            w = 1.0 / (lam + a)
            C = B * w[:, None]
            H = Z2 @ w + 1.0 / ntr                       # the hat diagonal (with the intercept)
            R = (Yc - Z @ C) / np.clip(1.0 - H, 1e-6, None)[:, None]
            g = np.mean(R * R, 0)
            better = g < best
            best[better] = g[better]
            coef[:, better] = C[:, better]
        out[te] = (Xte @ Q) @ coef + ym
    return out


# ---------------------------------------------------------------------------------------------- the slots
def slots_of(cap: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    """``[(slot name, site, getter)]`` — every token site at every depth, the move seats concatenated, the policy
    state (``PI``) and the critic pool (``VF``). Identical arrays (legacy's three board sites are ONE seat) are
    named separately and probed once."""
    sites = [str(s) for s in cap["sites"]]
    tok = cap["tok"]
    n_depth = tok.shape[2]
    out: List[Tuple[str, str, Any]] = []
    for di in range(n_depth):
        dn = DEPTH_NAMES[di]
        for si, s in enumerate(sites):
            if s in ("M1", "M2", "M3"):
                continue
            out.append((f"{s}@{dn}", s, (lambda si=si, di=di: tok[:, si, di])))
        mi = [sites.index(m) for m in ("M0", "M1", "M2", "M3")]
        out.append((f"Mcat@{dn}", "Mcat", (lambda mi=mi, di=di: tok[:, mi, di].reshape(len(tok), -1))))
    out.append(("PI", "PI", lambda: cap["pi"]))
    out.append(("VF", "VF", lambda: cap["vf"]))
    return out


def site_valid(cap: Dict[str, Any], site: str) -> np.ndarray:
    n = len(cap["probs"])
    if site in ("OA", "TA", "OB"):
        return np.asarray(cap[site]) >= 0
    if site in ("M0", "Mcat"):
        return np.asarray(cap["OA"]) >= 0
    return np.ones(n, dtype=bool)


def control_key(fact_about: str, keys: Dict[str, np.ndarray]) -> np.ndarray:
    return keys["OA" if fact_about == "-" else fact_about]


def probe_capture(cap: Dict[str, Any], facts: Sequence[Any], vals: np.ndarray, valid: np.ndarray,
                  keys: Dict[str, np.ndarray], folds: np.ndarray, only_slots: Optional[Sequence[str]] = None
                  ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{slot: {fact: {"n", "metric", "ctrl"} | None}}`` for one capture."""
    ctrl = {f.name: control_labels(vals[:, j].astype(np.float64), valid[:, j], control_key(f.about, keys), f.kind,
                                   f"control:{f.name}") for j, f in enumerate(facts)}
    res: Dict[str, Dict[str, Dict[str, Any]]] = {}
    done: Dict[Tuple[int, str], Dict[str, Dict[str, Any]]] = {}
    for slot, site, get in slots_of(cap):
        if only_slots is not None and slot not in only_slots:
            continue
        X = get()
        sv = site_valid(cap, site)
        # legacy's three board sites are ONE seat: an array already probed (same bytes, same rows) is reused
        ident = (hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(), np.packbits(sv).tobytes())
        if ident in done:
            res[slot] = done[ident]
            continue
        groups: Dict[bytes, List[int]] = {}
        for j, f in enumerate(facts):
            m = valid[:, j] & sv
            yy = vals[m, j]
            if m.sum() < MIN_ROWS or (f.kind == "b" and min((yy > 0.5).sum(), (yy <= 0.5).sum()) < MIN_MINORITY):
                continue
            groups.setdefault(np.packbits(m).tobytes(), []).append(j)
        out: Dict[str, Dict[str, Any]] = {f.name: None for f in facts}  # type: ignore[misc]
        for _, js in groups.items():
            m = valid[:, js[0]] & sv
            Xm = np.asarray(X[m], dtype=np.float32)
            Y = np.stack([vals[m, j].astype(np.float64) for j in js] +
                         [ctrl[facts[j].name][m] for j in js], 1)
            P = ridge_oof(Xm, Y, folds[m])
            for t, j in enumerate(js):
                f = facts[j]
                y = Y[:, t]
                out[f.name] = {"n": int(m.sum()), "metric": round(score(f.kind, y, P[:, t]), 4),
                               "ctrl": round(score(f.kind, Y[:, len(js) + t], P[:, len(js) + t]), 4)}
        res[slot] = out
        done[ident] = out
    return res


# ------------------------------------------------------------------------------------------------- driver
def facts_for_bank(bank: Path, cache_dir: Path):
    """Extract the facts once per bank (cached as ``facts.npz`` + ``facts.json`` in ``cache_dir``)."""
    from main.probe_battery import facts as F
    from main.probe_battery.bank import load_decisions

    decs = load_decisions(bank)
    facts, vals, valid, keys = F.extract(decs)
    battles = [d["game"] for d in decs]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "facts.npz", vals=vals, valid=valid, **{f"key_{k}": v for k, v in keys.items()})
    folds = battle_folds(battles)
    meta = []
    for j, f in enumerate(facts):
        v = valid[:, j]
        y = vals[:, j].astype(np.float64)
        ok = v.sum() >= MIN_ROWS and (f.kind == "c" or min((y[v] > 0.5).sum(), (y[v] <= 0.5).sum()) >= MIN_MINORITY)
        meta.append({"name": f.name, "family": f.family, "kind": f.kind, "about": f.about, "sites": list(f.sites),
                     "desc": f.desc, "n_valid": int(v.sum()), "base": float(np.nanmean(y)) if v.any() else None,
                     "species_lookup": round(species_lookup(y, v, control_key(f.about, keys), folds, f.kind), 4)
                     if ok else None})
    (cache_dir / "facts.json").write_text(json.dumps(meta, indent=1))
    return facts, vals, valid, keys, battles, decs


def probe_all(*, bank: Path, caps: Path, labels: Optional[Sequence[str]], out: Path, threads: int = 4) -> None:
    from main.probe_battery.bank import refuse_models_output

    refuse_models_output(out)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    out.mkdir(parents=True, exist_ok=True)
    facts, vals, valid, keys, battles, _ = facts_for_bank(bank, out)
    folds = battle_folds(battles)
    names = labels or sorted(p.stem for p in Path(caps).glob("*.npz"))
    for lab in names:
        dst = out / f"{lab}.json"
        if dst.exists():
            print(f"[probe_battery probe] {lab}: done already", flush=True)
            continue
        cap = dict(np.load(Path(caps) / f"{lab}.npz"))
        if len(cap["probs"]) != len(vals):
            raise RuntimeError(f"{lab}: {len(cap['probs'])} captured rows vs {len(vals)} bank decisions")
        res = probe_capture(cap, facts, vals, valid, keys, folds)
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps({"label": lab, "n_layers": int(cap["n_layers"]), "slots": res}, indent=0))
        os.replace(tmp, dst)
        print(f"[probe_battery probe] {lab}: {len(res)} slots", flush=True)
