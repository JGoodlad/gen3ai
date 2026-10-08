"""The RND INPUT-CHOICE measurement (meter (v)): novelty over the raw OBSERVATION or over the trunk's
detached FEATURES (`value_pooled`)?

`agents.model.ridealong_heads.RndNovelty` reads the observation, because trunk features DRIFT as the
trunk trains, so a feature-space novelty mixes "rarely seen" with "the representation moved". This
module measures both halves of that argument offline, on CPU, with fixed seeds:

* **Split — battle-level, by SOURCE (never row-level).** TRAIN = every bank decision from the
  N0@36M…N0@66M eval cycles (:data:`TRAIN_SOURCES`); HELD-OUT = everything else (N0@74M, C_fix,
  K2, K3, and K2 against the round-0 exploiter). Every battle belongs to exactly one source, so no
  battle straddles the split. HELD-OUT rows are labelled UNSEEN-TEAM when their (banked side's) team
  hash never occurs in TRAIN, and EXPLOITER by the bank's ``opp_class`` (the exploiter class is
  held-out only). The two labels are confounded (most exploiter rows are unseen teams), so each is
  also read inside the other's complement.
* **Predictors.** The heads' own ``RndNovelty`` class — the same frozen target and predictor shapes,
  the same per-dimension normalisation (fitted on TRAIN only) and ±5 clip — once over the 2845-dim
  observation (obs-RND) and once over the 128-dim ``value_pooled`` of a checkpoint (feature-RND). Adam
  at the ride-along learner's rate, minibatch 256, shuffles from a fixed NumPy seed; read after each
  epoch count in :data:`EPOCHS_READ` (one training run).
* **Drift.** Feature-RND trained on checkpoint A's features, scored on the SAME states through a
  LATER checkpoint B of the same lineage: whatever its novelty rises by is the representation
  moving, not the states. Obs-RND's input does not depend on the checkpoint, so its drift is zero by
  construction (its scores are bit-identical under any checkpoint).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from main.ridealong_read.meters import cluster_boot_auroc

TRAIN_SOURCES = ("N0@36M", "N0@46M", "N0@56M", "N0@66M")
EPOCHS_READ = (3, 10, 30)
BATCH = 256
SEED = 20260930


def split(decisions: Sequence[dict], train_sources: Sequence[str] = TRAIN_SOURCES) -> Dict[str, np.ndarray]:
    """Boolean row masks in bank order: ``train``, ``heldout``, ``unseen_team`` (held-out rows whose
    team never occurs in train), ``exploiter`` (held-out rows vs the exploiter class)."""
    src = np.array([d["source"] for d in decisions])
    train = np.isin(src, list(train_sources))
    heldout = ~train
    teams = np.array([d["team"] for d in decisions])
    seen = set(teams[train].tolist())
    unseen = heldout & ~np.isin(teams, list(seen))
    expl = heldout & (np.array([d["opp_class"] for d in decisions]) == "exploiter")
    battles = np.array([d["battle"] for d in decisions])
    leak = set(battles[train].tolist()) & set(battles[heldout].tolist())
    if leak:
        raise ValueError(f"{len(leak)} battles straddle the train / held-out split (first: {sorted(leak)[0]})")
    return {"train": train, "heldout": heldout, "unseen_team": unseen, "exploiter": expl}


def make_rnd(in_dim: int) -> Any:
    """A fresh ``RndNovelty`` built on a PRIVATE RNG (its constructor seeds torch itself)."""
    import torch

    from agents.model.ridealong_heads import RndNovelty

    with torch.random.fork_rng(devices=[]):
        return RndNovelty(int(in_dim))


def rnd_errors(rnd: Any, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(rnd.error(torch.from_numpy(np.ascontiguousarray(x[i:i + batch], dtype=np.float32))).numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def train_and_read(x_train: np.ndarray, read_fn: Callable[[Any], Any], epochs_read: Sequence[int] = EPOCHS_READ,
                   seed: int = SEED, batch: int = BATCH, lr: Optional[float] = None) -> dict:
    """Fit obs statistics on ``x_train`` and train a fresh predictor on it; after each epoch count in
    ``epochs_read`` call ``read_fn(rnd)`` and keep its result under that count. Returns
    ``{"<epochs>": read_fn(...), "train_loss": {...}}`` plus the module itself under ``"_rnd"``."""
    import torch

    from agents.training.instrumented_ppo.ridealong_terms import RIDEALONG_EPS, RIDEALONG_LR

    rnd = make_rnd(x_train.shape[1])
    rnd.update_obs_stats(torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)))
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=RIDEALONG_LR if lr is None else lr,
                           eps=RIDEALONG_EPS)
    rng = np.random.default_rng(seed)
    out: dict = {"train_loss": {}}
    last = max(epochs_read)
    for ep in range(1, last + 1):
        perm = rng.permutation(len(x_train))
        tot = 0.0
        for i in range(0, len(perm), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x_train[perm[i:i + batch]], dtype=np.float32))
            loss = rnd.error(xb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)
        if ep in epochs_read:
            out["train_loss"][str(ep)] = round(tot / len(perm), 6)
            out[str(ep)] = read_fn(rnd)
    out["_rnd"] = rnd
    return out


def renormalised(rnd: Any, x_train: np.ndarray) -> Any:
    """A copy of a trained ``rnd`` whose input statistics are RE-FITTED on ``x_train`` (the predictor
    and target untouched) — the drift control that removes a pure shift / rescale of the features,
    as a live running normalisation would partly track."""
    import copy

    import torch

    r = copy.deepcopy(rnd)
    r.obs_mean.zero_()
    r.obs_var.fill_(1.0)
    r.obs_count.zero_()
    r.update_obs_stats(torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)))
    return r


def _au(score: np.ndarray, label: np.ndarray, cl: np.ndarray) -> dict:
    from agents.training.instrumented_ppo.ridealong_terms import rank_auroc

    a = rank_auroc(score, label)
    return {"auroc": None if a is None else round(float(a), 4),
            "ci": cluster_boot_auroc(score, label, cl), "n_pos": int(label.sum()),
            "n_neg": int((~label).sum())}


def novelty_reads(err: np.ndarray, sp: Mapping[str, np.ndarray], battles: np.ndarray) -> dict:
    """Novelty AUROCs on HELD-OUT rows (``err`` over all rows, bank order): unseen team vs seen
    team (all held-out, and without the exploiter rows), exploiter vs others (all held-out, and
    inside seen teams only), and held-out-seen-team vs TRAIN rows (how much the predictor learned
    the training rows themselves). Plus the mean error per group."""
    ho = sp["heldout"]
    un, ex, tr = sp["unseen_team"], sp["exploiter"], sp["train"]
    seen_ho = ho & ~un

    def sub(mask: np.ndarray, pos: np.ndarray) -> dict:
        return _au(err[mask], pos[mask], battles[mask])

    both = np.concatenate([np.flatnonzero(seen_ho), np.flatnonzero(tr)])
    lab = np.r_[np.ones(seen_ho.sum(), bool), np.zeros(tr.sum(), bool)]
    return {
        "unseen_team_vs_seen": sub(ho, un),
        "unseen_team_vs_seen_no_exploiter": sub(ho & ~ex, un),
        "exploiter_vs_other": sub(ho, ex),
        "exploiter_vs_other_seen_teams": sub(seen_ho, ex),
        "heldout_seen_vs_train": _au(err[both], lab, battles[both]),
        "mean_err": {k: round(float(err[m].mean()), 6) if m.any() else None
                     for k, m in (("train", tr), ("heldout_seen_team", seen_ho),
                                  ("heldout_unseen_team", un), ("exploiter", ex))},
    }


def drift_reads(err_a: np.ndarray, err_b: np.ndarray, sp: Mapping[str, np.ndarray],
                battles: np.ndarray) -> dict:
    """Feature-RND trained on A's features: the SAME states scored through A and through B.

    ``inflation_*``: mean / median error ratio B ÷ A (1.0 = no drift); ``frac_above_a_p90``: the
    share of states whose B error exceeds A's own 90th percentile (0.10 = no drift);
    ``drift_auroc``: how separable (state via A) and (same state via B) are by the novelty score
    alone (0.5 = none) — each on held-out rows and on the TRAIN rows the predictor fitted."""
    out: dict = {}
    for name, m in (("heldout", sp["heldout"]), ("train", sp["train"])):
        a, b = err_a[m], err_b[m]
        p90 = float(np.quantile(a, 0.9))
        sc = np.concatenate([a, b])
        lab = np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)]
        cl = np.concatenate([battles[m], battles[m]])
        out[name] = {"inflation_mean": round(float(b.mean() / a.mean()), 4),
                     "inflation_median": round(float(np.median(b) / np.median(a)), 4),
                     "frac_above_a_p90": round(float((b > p90).mean()), 4),
                     "drift_auroc": _au(sc, lab, cl)}
    return out


__all__ = ["TRAIN_SOURCES", "EPOCHS_READ", "split", "make_rnd", "rnd_errors", "train_and_read",
           "renormalised",
           "novelty_reads", "drift_reads"]
