"""winprob_finetune — the OFFLINE, HEAD-ONLY repair of the win-probability head.

The consumer half of the harvest pipeline. ``main.harvest`` mines mid/late-game decision states
out of eval traces and labels each with ``k`` wins of ``n`` terminal-adjudicated Monte-Carlo
rollouts (schema: :mod:`agents.training.harvest_schema`). This module reads those rows and fits
:class:`agents.model.aux_value_heads.WinProbHead` — and NOTHING else — against them.

    python -m agents.training.winprob_finetune <harvest_dir_or_shard> --subject <ckpt.zip>

Why the head-only rung is the FIRST one
---------------------------------------
The G0 bias map convicted the scalar win-prob head of a RESOLUTION defect: it puts states in the
same confidence bin that do not belong in the same bin. There are two families of repair and they
are not equally safe. Re-training the TRUNK on win-prob labels changes ``value_pooled``, which
every other readout and the critic's own projection consume — a change that can only be judged by
a full generation, costs days, and risks the policy. Re-fitting the HEAD changes one small MLP that
feeds nothing forward (:class:`WinProbHead` is a leak-safe SIDE readout: its logit is stashed and
read by the aux loss and the offline prober, never concatenated into pi/vf). So the head-only fit
is the rung whose downside is bounded by construction, and it answers a question the trunk rung
needs answered first: *how much of the head's error is the head's, and how much is already gone
from ``value_pooled``?* If a well-fit head still cannot resolve the states, the information is not
in the trunk summary and the trunk rung is licensed. If it can, the trunk was never the problem.

TWO PHASES, and the split is what makes trunk-frozen STRUCTURAL
---------------------------------------------------------------
``WinProbHead``'s only input is ``value_pooled`` — a [B, D_MODEL] vector the frozen trunk produces.
So the fit does not need the trunk in the loop at all:

* **Phase 1 (precompute)** — one ``no_grad`` forward of the frozen extractor over every labelled
  obs, caching ``value_pooled``. Runs through :meth:`ProbeModel._value_pooled_batch`, the same code
  path the prober's cf readouts use, so the numbers here are comparable with the live ``cf/*``
  scalars rather than a second derivation of them.
* **Phase 2 (fit)** — train only the head's parameters on the cached vectors. The trunk is not
  merely frozen, it is ABSENT: it is not in the optimizer's param groups, it is not in the graph,
  and there is nothing for a gradient to flow into. It is also the cheap half — a few-thousand-row
  MLP fit is seconds, where re-forwarding the extractor every epoch would be the whole cost.

The explicit ``requires_grad``/``.grad`` assertions in the test file are kept anyway. The split is
an optimization, and an optimization is the kind of thing someone eventually fuses.

The likelihood: BINOMIAL over counts, never BCE on the rate
-----------------------------------------------------------
A harvest label is ``k`` of ``n``, not a bit. With ``q = sigmoid(z)``::

    NLL_i = -[ k_i * log q_i + (n_i - k_i) * log(1 - q_i) ]
    loss  = SUM_i w_i * NLL_i  /  SUM_i w_i * n_i

which is exactly ``n_i`` per-observation cross-entropies, so a 32-rollout label pulls 32x a
1-rollout one. That is not an emphasis choice — it is the likelihood of the data, and collapsing
the row to BCE on ``k/n`` would throw away precisely the precision that makes a k/n label worth
more than an outcome bit. At ``n == 1`` it reduces EXACTLY to BCE (pinned by a test, and
cross-checked against the live trainer's :func:`agents.training.cf_terms.cf_binomial_nll`, which
uses the same ``SUM NLL / SUM n`` normalization so a number here is comparable with one there).
Computed through ``softplus``: ``-log sigmoid(z) = softplus(-z)``, stable where ``log(sigmoid(z))``
underflows.

Turn-slice re-weighting
-----------------------
The harvest targets the stall tail, but "targets" is not "balances": a late slice can still hold a
tenth the rows of an early one, and an unweighted likelihood would let the early states buy the
head's capacity. Each row is weighted by the INVERSE FREQUENCY of its turn slice, normalized to
MEAN 1 so ``--lr`` means the same thing on every dataset. The slice edges are a declared, versioned
constant (:data:`SLICE_EDGES` / :data:`SLICE_VERSION`) written into every checkpoint, because a
re-weighting whose consumer cannot see the bins is a distribution-shift confound wearing a number.
``--slice-reweight none`` turns it off (all-ones), which is the honest control arm.

The re-weighting is an OPTIMIZER device, so the reported val NLL is the PLAIN one (no slice
weights) and the best epoch is chosen by it. Re-weighting the meter as well would make the meter
agree with the device by construction.

Battle-level holdout
--------------------
Decisions inside one battle are not independent — they share a team draw, an opponent, and most of
a trajectory. A state-level split therefore puts near-copies of a val state in train, and the
resulting val NLL measures memorisation. :func:`split_by_battle` splits on ``battle_tag`` and
:func:`assert_battle_disjoint` refuses a split whose sides share one. State-level leakage is the
failure mode that would make the whole meter a lie, so it is an assertion, not a convention.

Grafting the fitted head back
-----------------------------
Two paths, both provided, both verified after writing:

* :func:`apply_head` grafts the head into an ALREADY-LOADED :class:`ProbeModel`, in place. This is
  the documented meter path — it needs no re-save, works on any subject the prober can load, and
  is what a scoring script should call.
* :func:`graft_into_checkpoint` writes a full ``MaskablePPO``-loadable ``.zip`` with the new head
  in it, so the fine-tuned subject can be scored by the ORDINARY ``ProbeModel.load`` path with no
  special-casing at all. It reloads what it wrote and asserts every head tensor is bitwise equal
  before returning. This is the CLI default; ``--no-emit-zip`` skips it.

A row whose obs cannot be reproduced is COUNTED, never dropped
--------------------------------------------------------------
:func:`harvest_schema.load_obs` raises on a digest mismatch — the array on disk is not the array
that was labelled. Those rows are counted per-reason and printed. A silent drop would let a
harvest whose traces were rewritten under it train on half the data and report a clean run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training.harvest_schema import load_obs, read_dir

# ---------------------------------------------------------------------------------------------
# The turn-slice contract
# ---------------------------------------------------------------------------------------------

#: Turn-slice edges. ``len(SLICE_EDGES) + 1`` bins: ``(-inf, 60) [60, 80) ... [170, 250) [250, inf)``.
#: Chosen around the stall tail the harvest targets — MAX_TURNS is 250 and the forfeit deadline sits
#: exactly there, so the last bin is "at or past the cap" and is its own regime.
SLICE_EDGES: Tuple[int, ...] = (60, 80, 100, 130, 170, 250)

#: Bump when :data:`SLICE_EDGES` changes. Written into every checkpoint and every report, so a
#: consumer can tell two runs' weights apart instead of assuming they share a frame.
SLICE_VERSION = "winprob_slices_v1"

#: The re-weighting modes ``--slice-reweight`` accepts.
SLICE_MODES = ("inverse", "none")


def slice_index(turns: "np.ndarray | Sequence[int]") -> np.ndarray:
    """Map game turns to slice indices in ``[0, len(SLICE_EDGES)]`` (``np.searchsorted``, right-open)."""
    t = np.asarray(turns)
    return np.searchsorted(np.asarray(SLICE_EDGES), t, side="right").astype(np.int64)


def slice_weights(turns: "np.ndarray | Sequence[int]", mode: str = "inverse") -> np.ndarray:
    """Per-row weights, MEAN EXACTLY 1.

    ``inverse``: ``w_i = N / (B * count(bin_i))`` where ``B`` is the number of NON-EMPTY bins. Every
    non-empty bin then carries total weight ``N / B`` regardless of its size, so a slice with a
    tenth the rows gets ten times the per-row weight — and the mean is 1 by construction, which is
    what keeps ``--lr`` comparable across datasets.

    ``none``: all ones (the control arm).
    """
    if mode not in SLICE_MODES:
        raise ValueError(f"unknown slice-reweight mode {mode!r} (expected one of {SLICE_MODES})")
    t = np.asarray(turns)
    n = int(t.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if mode == "none":
        return np.ones(n, dtype=np.float64)
    idx = slice_index(t)
    counts = np.bincount(idx, minlength=len(SLICE_EDGES) + 1).astype(np.float64)
    n_nonempty = float((counts > 0).sum())
    w = n / (n_nonempty * counts[idx])
    # Exact mean-1 up to float error: the closed form above is already mean 1, but a final rescale
    # makes it mean-1 BITWISE for any float rounding, which the test asserts.
    return w * (n / float(w.sum()))


# ---------------------------------------------------------------------------------------------
# The likelihood and its noise floor
# ---------------------------------------------------------------------------------------------

def binomial_nll(logits: Any, wins: Any, n_rollouts: Any, weights: Any = None) -> Any:
    """``SUM w*NLL / SUM w*n`` — the weighted binomial NLL, in nats PER ROLLOUT.

    ``logits`` [B] or [B, 1]; ``wins`` (k) and ``n_rollouts`` (n) [B]. ``weights`` [B] or ``None``
    (which is all-ones and reduces the formula to ``SUM NLL / SUM n``, matching
    :func:`agents.training.cf_terms.cf_binomial_nll` exactly).

    Stable form: ``-log sigmoid(z) = softplus(-z)`` and ``-log(1 - sigmoid(z)) = softplus(z)``.
    """
    import torch
    import torch.nn.functional as F

    z = torch.as_tensor(logits).reshape(-1)
    k = torch.as_tensor(wins).reshape(-1).to(z.dtype)
    n = torch.as_tensor(n_rollouts).reshape(-1).to(z.dtype).clamp(min=1.0)
    k = torch.minimum(k.clamp(min=0.0), n)
    per_row = k * F.softplus(-z) + (n - k) * F.softplus(z)
    if weights is None:
        w = torch.ones_like(per_row)
    else:
        w = torch.as_tensor(weights).reshape(-1).to(z.dtype)
    return (w * per_row).sum() / (w * n).sum().clamp(min=1e-12)


def label_noise_variance(wins: np.ndarray, n_rollouts: np.ndarray) -> float:
    """The mean SAMPLING VARIANCE of the labels themselves — the floor no head can go below.

    For one row, ``phat(1-phat)/(n-1)`` is EXACTLY unbiased for ``p(1-p)/n``, the variance of an
    n-rollout mean (``E[phat(1-phat)] = p(1-p)(n-1)/n``). This is the same identity ``cf_audit``
    subtracts to get ``sd_true_excess``, reused rather than re-derived.

    Rows with ``n < 2`` carry no information about their own variance and are EXCLUDED; ``nan``
    when no row qualifies (never 0, which would read as "the labels are exact").
    """
    k = np.asarray(wins, dtype=np.float64)
    n = np.asarray(n_rollouts, dtype=np.float64)
    ok = n >= 2
    if not ok.any():
        return float("nan")
    p = k[ok] / n[ok]
    return float(np.mean(p * (1.0 - p) / (n[ok] - 1.0)))


def label_noise_sd(wins: np.ndarray, n_rollouts: np.ndarray) -> float:
    """``sqrt`` of :func:`label_noise_variance` — the label's own sd, in probability units."""
    v = label_noise_variance(wins, n_rollouts)
    return float("nan") if math.isnan(v) else math.sqrt(v)


# ---------------------------------------------------------------------------------------------
# The battle-level holdout
# ---------------------------------------------------------------------------------------------

def split_by_battle(rows: Sequence[dict], frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Split ``rows`` into ``(train, val)`` on ``battle_tag``, never on state.

    Battles are shuffled with a seeded RNG and assigned to val until the val ROW count reaches
    ``frac`` of the total, so the returned fraction is approximate in rows and exact in battles.
    The result is checked by :func:`assert_battle_disjoint` before it is returned.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"holdout frac must be in [0, 1), got {frac}")
    by_tag: Dict[str, List[dict]] = {}
    for r in rows:
        by_tag.setdefault(str(r["battle_tag"]), []).append(r)
    tags = sorted(by_tag)
    rng = np.random.default_rng(seed)
    rng.shuffle(tags)
    target = int(round(frac * len(rows)))
    val: List[dict] = []
    val_tags: set = set()
    for t in tags:
        if len(val) >= target:
            break
        val.extend(by_tag[t])
        val_tags.add(t)
    train = [r for t in tags if t not in val_tags for r in by_tag[t]]
    assert_battle_disjoint(train, val)
    return train, val


def assert_battle_disjoint(train: Sequence[dict], val: Sequence[dict]) -> None:
    """Raise ``ValueError`` if one battle's states appear on both sides of a split.

    This is the guard, not a comment: a state-level splitter puts near-copies of a val state in
    train (same team draw, same opponent, most of the same trajectory), and the val NLL then
    measures memorisation while looking exactly like a measurement of generalisation.
    """
    a = {str(r["battle_tag"]) for r in train}
    b = {str(r["battle_tag"]) for r in val}
    shared = sorted(a & b)
    if shared:
        raise ValueError(
            f"battle-level holdout VIOLATED: {len(shared)} battle_tag(s) appear in both splits "
            f"(e.g. {shared[:3]}) — this is a state-level split, and its val NLL would measure "
            "memorisation. Split with split_by_battle().")


# ---------------------------------------------------------------------------------------------
# Phase 1 — precompute value_pooled
# ---------------------------------------------------------------------------------------------

@dataclass
class PooledDataset:
    """Cached ``value_pooled`` plus everything the fit needs. Phase 1's whole output."""

    x: np.ndarray            # [N, D_MODEL] float32 — the frozen trunk's summary
    wins: np.ndarray         # [N] int64 — k
    n_rollouts: np.ndarray   # [N] int64 — n
    turn: np.ndarray         # [N] int64
    battle_tag: List[str]    # [N]
    weight: np.ndarray       # [N] float64 — the slice weights (set by set_weights)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def rate(self) -> np.ndarray:
        """``k/n`` — the empirical win rate per row."""
        return self.wins.astype(np.float64) / np.maximum(self.n_rollouts, 1)

    def set_weights(self, mode: str) -> "PooledDataset":
        self.weight = slice_weights(self.turn, mode)
        return self


@dataclass
class RejectReport:
    """Why rows did not make it in. Printed; never silent."""

    loaded: int = 0
    digest_mismatch: int = 0
    obs_dim_mismatch: int = 0
    other: int = 0
    examples: List[str] = field(default_factory=list)

    @property
    def rejected(self) -> int:
        return self.digest_mismatch + self.obs_dim_mismatch + self.other

    def note(self, kind: str, msg: str) -> None:
        setattr(self, kind, getattr(self, kind) + 1)
        if len(self.examples) < 5:
            self.examples.append(f"[{kind}] {msg}")


PooledFn = Callable[[np.ndarray], np.ndarray]


def probe_pooled_fn(probe: Any) -> PooledFn:
    """A ``[B, obs_dim] -> [B, D_MODEL]`` callable over :meth:`ProbeModel._value_pooled_batch`.

    Deliberately the PRIVATE method rather than a re-derivation: it is the shared preamble every cf
    readout in the prober rides, and a second implementation here would be a second opinion about
    the mask fallback, the obs-dim check and the stash key — the exact drift that would make these
    offline numbers stop being comparable with the live ``cf/*`` scalars.
    """
    def _fn(obs: np.ndarray) -> np.ndarray:
        pooled, _vf = probe._value_pooled_batch(np.asarray(obs, dtype=np.float32))
        if pooled is None:
            raise RuntimeError(
                "the subject checkpoint's extractor published no value_pooled — this is not a "
                "model the win-prob head can be fine-tuned on")
        return np.asarray(pooled.detach().cpu().numpy(), dtype=np.float32)
    return _fn


def precompute_value_pooled(pooled_fn: PooledFn, rows: Sequence[dict], *,
                            models_root: Optional[str] = None,
                            batch_size: int = 64,
                            slice_mode: str = "inverse",
                            log: Callable[[str], None] = print,
                            ) -> Tuple[PooledDataset, RejectReport]:
    """PHASE 1: resolve every row's obs and cache the frozen trunk's ``value_pooled`` for it.

    One ``no_grad`` pass (``pooled_fn`` owns the ``no_grad``), batched. Rows whose obs cannot be
    reproduced are counted in the returned :class:`RejectReport` and excluded — the digest check in
    :func:`harvest_schema.load_obs` is what makes "the array I loaded is the array that was
    labelled" a fact rather than a belief.
    """
    rep = RejectReport()
    npz_cache: Dict[str, np.ndarray] = {}
    obs_list: List[np.ndarray] = []
    kept: List[dict] = []
    for r in rows:
        try:
            obs = load_obs(r, models_root=models_root, npz_cache=npz_cache)
        except ValueError as exc:
            # load_obs' own message already names the row, so re-prefixing it would double the id.
            kind = "digest_mismatch" if "digest mismatch" in str(exc) else "other"
            rep.note(kind, str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 — a broken shard must be counted, not fatal
            rep.note("other", f"{r.get('battle_tag')}#{r.get('decision_idx')}: {exc}")
            continue
        obs_list.append(obs)
        kept.append(r)

    xs: List[np.ndarray] = []
    ok_rows: List[dict] = []
    for i in range(0, len(obs_list), max(1, batch_size)):
        chunk = obs_list[i:i + batch_size]
        chunk_rows = kept[i:i + batch_size]
        try:
            xs.append(pooled_fn(np.stack(chunk)))
            ok_rows.extend(chunk_rows)
        except Exception as exc:  # noqa: BLE001
            # Fall back to per-row so ONE bad obs does not cost a whole batch, and so the reason
            # lands on the row it belongs to.
            for obs, r in zip(chunk, chunk_rows):
                try:
                    xs.append(pooled_fn(obs[None, :]))
                    ok_rows.append(r)
                except Exception as inner:  # noqa: BLE001
                    kind = "obs_dim_mismatch" if "dim" in str(inner).lower() else "other"
                    rep.note(kind, f"{r.get('battle_tag')}#{r.get('decision_idx')}: {inner}")
            del exc

    rep.loaded = len(ok_rows)
    x = np.concatenate(xs, axis=0).astype(np.float32) if xs else np.zeros((0, 0), dtype=np.float32)
    ds = PooledDataset(
        x=x,
        wins=np.asarray([int(r["n_wins"]) for r in ok_rows], dtype=np.int64),
        n_rollouts=np.asarray([int(r["n_rollouts"]) for r in ok_rows], dtype=np.int64),
        turn=np.asarray([int(r["turn"]) for r in ok_rows], dtype=np.int64),
        battle_tag=[str(r["battle_tag"]) for r in ok_rows],
        weight=np.ones(len(ok_rows), dtype=np.float64),
    ).set_weights(slice_mode)
    log(f"phase 1: {rep.loaded} rows pooled, {rep.rejected} rejected "
        f"(digest {rep.digest_mismatch}, obs-dim {rep.obs_dim_mismatch}, other {rep.other})")
    for e in rep.examples:
        log(f"  {e}")
    return ds, rep


def subset(ds: PooledDataset, idx: np.ndarray, slice_mode: str) -> PooledDataset:
    """A view of ``ds`` at row indices ``idx``, with weights RECOMPUTED on the subset.

    Recomputed rather than sliced, because inverse-frequency weights are a property of the
    population you are optimizing over: carrying the full set's counts into a split would make the
    train weights describe rows the optimizer never sees.
    """
    i = np.asarray(idx, dtype=np.int64)
    return PooledDataset(
        x=ds.x[i],
        wins=ds.wins[i],
        n_rollouts=ds.n_rollouts[i],
        turn=ds.turn[i],
        battle_tag=[ds.battle_tag[int(j)] for j in i],
        weight=np.ones(len(i), dtype=np.float64),
    ).set_weights(slice_mode)


def split_pooled(ds: PooledDataset, frac: float, seed: int,
                 slice_mode: str = "inverse") -> Tuple[PooledDataset, PooledDataset]:
    """:func:`split_by_battle`, applied to a :class:`PooledDataset` by index."""
    rows = [{"battle_tag": t, "_i": i} for i, t in enumerate(ds.battle_tag)]
    tr, va = split_by_battle(rows, frac, seed)
    return (subset(ds, np.asarray([r["_i"] for r in tr], dtype=np.int64), slice_mode),
            subset(ds, np.asarray([r["_i"] for r in va], dtype=np.int64), slice_mode))


# ---------------------------------------------------------------------------------------------
# Phase 2 — the fit
# ---------------------------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Everything that defines the fit. Written verbatim into every checkpoint."""

    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 1024
    weight_decay: float = 0.0
    seed: int = 0
    slice_reweight: str = "inverse"
    holdout_frac: float = 0.2
    anchor_coef: float = 0.3


@dataclass
class EpochMetrics:
    """One epoch's line. ``val_nll`` is the PLAIN (un-re-weighted) held-out likelihood."""

    epoch: int
    train_nll: float
    train_nll_weighted: float
    val_nll: float
    val_nll_weighted: float
    val_brier: float
    val_brier_floor: float
    val_brier_excess: float
    val_ece: float
    val_mean_pred: float
    val_mean_label: float


@dataclass
class FitResult:
    best_epoch: int
    best_val_nll: float
    history: List[EpochMetrics]
    out_dir: str
    label_noise_sd_train: float
    label_noise_sd_val: float


def _logits(head: Any, x: Any) -> Any:
    return head(x).reshape(-1)


def evaluate(head: Any, ds: PooledDataset) -> Dict[str, float]:
    """Held-out diagnostics: plain + weighted NLL, Brier vs its own binomial floor, ECE.

    ``val_brier`` is the n-weighted mean of ``(phi - k/n)^2``; ``val_brier_floor`` is
    :func:`label_noise_variance`, which is EXACTLY the part of that square the label's own dice
    contribute. ``val_brier_excess = max(0, brier - floor)`` is therefore the part that is the
    head's, and it is the number to quote.
    """
    import torch

    if len(ds) == 0:
        return {k: float("nan") for k in
                ("nll", "nll_weighted", "brier", "brier_floor", "brier_excess", "ece",
                 "mean_pred", "mean_label")}
    with torch.no_grad():
        z = _logits(head, torch.as_tensor(ds.x))
        nll = float(binomial_nll(z, ds.wins, ds.n_rollouts).item())
        nll_w = float(binomial_nll(z, ds.wins, ds.n_rollouts, ds.weight).item())
        phi = torch.sigmoid(z).cpu().numpy().astype(np.float64)
    n = ds.n_rollouts.astype(np.float64)
    rate = ds.rate
    brier = float(np.sum(n * (phi - rate) ** 2) / max(n.sum(), 1e-12))
    floor = label_noise_variance(ds.wins, ds.n_rollouts)
    excess = float("nan") if math.isnan(floor) else max(0.0, brier - floor)
    return {
        "nll": nll, "nll_weighted": nll_w,
        "brier": brier, "brier_floor": floor, "brier_excess": excess,
        "ece": calibration_ece(phi, ds.wins, ds.n_rollouts),
        "mean_pred": float(np.mean(phi)), "mean_label": float(np.mean(rate)),
    }


def calibration_bins(phi: np.ndarray, wins: np.ndarray, n_rollouts: np.ndarray,
                     nbins: int = 10) -> List[Dict[str, float]]:
    """Reliability table over ``nbins`` equal-width bins of the predicted probability.

    Each bin reports its predicted mean, its ROLLOUT-weighted realized rate, and the row/rollout
    counts — rollout-weighted because a 32-rollout label is 32 observations of that bin.
    """
    phi = np.asarray(phi, dtype=np.float64)
    k = np.asarray(wins, dtype=np.float64)
    n = np.asarray(n_rollouts, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.searchsorted(edges, phi, side="right") - 1, 0, nbins - 1)
    out: List[Dict[str, float]] = []
    for b in range(nbins):
        m = idx == b
        if not m.any():
            continue
        out.append({
            "bin": float(b), "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "rows": float(m.sum()), "rollouts": float(n[m].sum()),
            "pred": float(np.average(phi[m], weights=n[m])),
            "actual": float(k[m].sum() / max(n[m].sum(), 1e-12)),
        })
    return out


def calibration_ece(phi: np.ndarray, wins: np.ndarray, n_rollouts: np.ndarray,
                    nbins: int = 10) -> float:
    """Expected calibration error — rollout-weighted ``|pred - actual|`` over the reliability bins."""
    bins = calibration_bins(phi, wins, n_rollouts, nbins)
    total = sum(b["rollouts"] for b in bins)
    if total <= 0:
        return float("nan")
    return float(sum(b["rollouts"] * abs(b["pred"] - b["actual"]) for b in bins) / total)


def _epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    """The epoch's shuffle, derived from ``(seed, epoch)`` alone.

    Deriving it rather than consuming a running RNG is what makes ``--resume`` reproduce the
    uninterrupted run BITWISE: epoch ``e``'s batches do not depend on how many epochs ran in this
    process. The checkpoint still records the torch/numpy/python RNG states (a fit that later grows
    a stochastic component will need them), but the fit does not depend on restoring them.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)


def _rng_state() -> Dict[str, Any]:
    import torch
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def build_head(state_dict: Optional[Dict[str, Any]] = None) -> Any:
    """A fresh :class:`WinProbHead`, optionally warm-started from the subject's existing weights.

    Warm-starting is the default in the CLI: the fine-tune is a REPAIR of a trained head, and
    starting from random weights would answer a different question (how well can this trunk summary
    be decoded at all) than the one asked (can this head be fixed).
    """
    from agents.model.aux_value_heads import WinProbHead
    head = WinProbHead()
    if state_dict is not None:
        head.load_state_dict(state_dict)
    return head


def fit_head(head: Any, train: PooledDataset, val: PooledDataset, cfg: FitConfig, *,
             out_dir: str, subject_ckpt: str = "", resume: bool = False,
             log: Callable[[str], None] = print) -> FitResult:
    """PHASE 2: fit ONLY ``head``'s parameters on the cached ``value_pooled``.

    The trunk is not frozen by a flag — it is not present. ``head.parameters()`` is the entire
    optimizer param group and ``train.x`` is a plain tensor with no graph behind it, so there is
    nothing a gradient could reach.

    Writes ``head_last.pt`` (resumable) every epoch and ``head_best.pt`` at the best PLAIN val NLL.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    # Capture the anchor BEFORE anything can modify the head — in particular before the resume
    # branch below loads a partially-trained state dict. The anchor is the SUBJECT's original
    # function, not "wherever the fit had got to", and taking it after a resume would both change
    # the objective mid-run and make a resumed fit diverge from an uninterrupted one (which is
    # exactly how this was caught: `test_resume_reproduces_the_uninterrupted_run_bitwise`).
    with torch.no_grad():
        z_anchor = _logits(head, torch.as_tensor(train.x)).detach().clone()

    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    _assert_head_only(head, opt)

    start_epoch = 0
    history: List[EpochMetrics] = []
    best_epoch, best_nll = -1, float("inf")
    last_path = os.path.join(out_dir, "head_last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        history = [EpochMetrics(**h) for h in ck.get("history", [])]
        best_epoch, best_nll = int(ck.get("best_epoch", -1)), float(ck.get("best_val_nll", math.inf))
        log(f"resumed from {last_path} at epoch {start_epoch}")

    xt = torch.as_tensor(train.x)
    kt = torch.as_tensor(train.wins)
    nt = torch.as_tensor(train.n_rollouts)
    wt = torch.as_tensor(train.weight, dtype=torch.float32)

    # THE ANCHOR — a trust region on the head's own output, in LOGIT space.
    #
    # Earned by two metered pilots. A harvest is a PRIORITIZED sample: it is selected for states
    # where the head is wrong, so its label mean is far from the population's. Fitting a
    # 6-parameter head on it with nothing holding it back collapses the head toward that sample
    # mean and destroys calibration everywhere else — measured on held-out battles, and the damage
    # scaled with the offset: a sample mean of 0.621 moved the long-WIN control's phi_T by -0.376,
    # a sample mean of 0.427 moved it by -0.165. Same sign, same mechanism, half the dose.
    #
    # `anchor_coef * mean((z - z0)^2)` where `z0` is the SUBJECT's logit on the same cached
    # `value_pooled`, captured once before the first step. The head may move where the labels are
    # strong and is pulled back where they are not, which is exactly the asymmetry a biased sample
    # requires. It is a per-EXAMPLE penalty rather than a weight penalty on purpose: the quantity
    # that must not drift is the head's FUNCTION on the real state distribution, and an L2 on six
    # parameters says nothing about that.
    #
    # DEFAULT 0.3 — ON. Shipping 0.0 would ship a setting MEASURED to be destructive, which is the
    # one thing a default must not be. The sweep, metered on held-out battles (pilot 2's labels,
    # long-WIN control `phi_T` is the falsification):
    #
    #     anchor  0.0 -> control phi_T -0.165 SIG, long-loss detect_le05 -0.326 SIG   REGRESSION
    #     anchor  0.3 -> control phi_T -0.033,     every categorical metric UNCHANGED, and the cap
    #                    class's phi_T moves the RIGHT way (-0.109) — safe, not yet beneficial
    #     anchor  1.0 -> best epoch 0: the anchor dominates and the fit does not move at all
    #
    # So 0.3 is the largest dose that still lets the labels speak. `--anchor-coef 0` opts out and
    # reproduces the pilots exactly.

    for epoch in range(start_epoch, cfg.epochs):
        head.train()
        perm = _epoch_permutation(len(train), cfg.seed, epoch)
        for i in range(0, len(perm), max(1, cfg.batch_size)):
            sel = torch.as_tensor(perm[i:i + cfg.batch_size])
            opt.zero_grad(set_to_none=True)
            z = _logits(head, xt[sel])
            loss = binomial_nll(z, kt[sel], nt[sel], wt[sel])
            if cfg.anchor_coef > 0.0:
                loss = loss + cfg.anchor_coef * ((z - z_anchor[sel]) ** 2).mean()
            loss.backward()
            opt.step()
        head.eval()
        tr = evaluate(head, train)
        va = evaluate(head, val)
        m = EpochMetrics(
            epoch=epoch,
            train_nll=tr["nll"], train_nll_weighted=tr["nll_weighted"],
            val_nll=va["nll"], val_nll_weighted=va["nll_weighted"],
            val_brier=va["brier"], val_brier_floor=va["brier_floor"],
            val_brier_excess=va["brier_excess"], val_ece=va["ece"],
            val_mean_pred=va["mean_pred"], val_mean_label=va["mean_label"])
        history.append(m)
        improved = not math.isnan(m.val_nll) and m.val_nll < best_nll
        if improved:
            best_epoch, best_nll = epoch, m.val_nll
        _save(os.path.join(out_dir, "head_last.pt"), head, opt, epoch, cfg, subject_ckpt,
              history, best_epoch, best_nll)
        if improved:
            _save(os.path.join(out_dir, "head_best.pt"), head, opt, epoch, cfg, subject_ckpt,
                  history, best_epoch, best_nll)
        log(f"epoch {epoch:3d}  train_nll {m.train_nll:.5f}  val_nll {m.val_nll:.5f}"
            f"  val_brier {m.val_brier:.5f} (floor {m.val_brier_floor:.5f},"
            f" excess {m.val_brier_excess:.5f})  ece {m.val_ece:.4f}"
            f"  pred {m.val_mean_pred:.3f} vs label {m.val_mean_label:.3f}"
            f"{'  <- best' if improved else ''}")

    return FitResult(
        best_epoch=best_epoch, best_val_nll=best_nll, history=history, out_dir=out_dir,
        label_noise_sd_train=label_noise_sd(train.wins, train.n_rollouts),
        label_noise_sd_val=label_noise_sd(val.wins, val.n_rollouts))


def _assert_head_only(head: Any, opt: Any) -> None:
    """Refuse an optimizer that can reach anything but ``head``. The trunk-frozen guard, asserted."""
    allowed = {id(p) for p in head.parameters()}
    for gi, group in enumerate(opt.param_groups):
        for p in group["params"]:
            if id(p) not in allowed:
                raise ValueError(
                    f"winprob_finetune: optimizer param group {gi} holds a parameter that is not "
                    f"the win-prob head's (shape {tuple(p.shape)}). This fit is HEAD-ONLY by "
                    "contract — a trunk parameter in the optimizer would silently re-train the "
                    "shared value_pooled every other readout consumes.")


def _save(path: str, head: Any, opt: Any, epoch: int, cfg: FitConfig, subject_ckpt: str,
          history: Sequence[EpochMetrics], best_epoch: int, best_val_nll: float) -> None:
    import torch
    tmp = path + ".tmp"
    torch.save({
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
        "rng_state": _rng_state(),
        "config": asdict(cfg),
        "slice_version": SLICE_VERSION,
        "slice_edges": list(SLICE_EDGES),
        "subject_ckpt": subject_ckpt,
        "history": [asdict(h) for h in history],
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
    }, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# Grafting the fitted head back
# ---------------------------------------------------------------------------------------------

def apply_head(probe_model: Any, head_ckpt_path: str, *, strict_subject: bool = False) -> Any:
    """Graft a fitted head into an ALREADY-LOADED ``ProbeModel``, IN PLACE. **The meter path.**

    Returns the ``probe_model`` for chaining. After this call every ``ProbeModel`` read that routes
    through ``win_head`` — :meth:`win_prob_at`, the cf twin readouts, the prober's ``/analyze``
    win-prob row — reports the FINE-TUNED head, with an otherwise byte-identical trunk.

    ``strict_subject=True`` raises when the checkpoint was fitted against a different subject; the
    default warns to stderr instead, because scoring a fitted head on a sibling checkpoint is a
    legitimate (and interesting) transfer question.
    """
    import torch

    ex = getattr(getattr(probe_model, "_policy", None), "features_extractor", None)
    head = getattr(ex, "win_head", None) if ex is not None else None
    if head is None:
        raise ValueError(
            "this checkpoint carries no win_head (--win-prob-mode none, or the extractor did not "
            "load) — there is nothing to graft a fine-tuned win-prob head into")
    ck = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
    fitted_on = str(ck.get("subject_ckpt") or "")
    if strict_subject and fitted_on and getattr(probe_model, "_subject_ckpt", None) not in (None, fitted_on):
        raise ValueError(f"head was fitted on {fitted_on!r}, not this subject")
    if fitted_on:
        print(f"[winprob_finetune] grafting head fitted on {fitted_on} "
              f"(epoch {ck.get('epoch')}, slices {ck.get('slice_version')})", file=sys.stderr)
    head.load_state_dict(ck["head_state_dict"])
    head.eval()
    return probe_model


def graft_into_checkpoint(subject_zip: str, head_ckpt_path: str, out_zip: str,
                          device: str = "cpu") -> str:
    """Write a full ``MaskablePPO``-loadable ``.zip`` = the subject with the fitted head in it.

    Reloads what it wrote and asserts every head tensor is BITWISE equal to the fitted one before
    returning — a graft that silently did not take would be indistinguishable from a fine-tune that
    did not help. The subject's ``model_config.json`` / ``metadata.json`` siblings are copied next
    to the output so ``load_model_snapshot`` (which searches the zip's dir and its parent) can
    version-check the result exactly as it does the subject.
    """
    import shutil

    import torch
    from sb3_contrib import MaskablePPO

    from main.prober.model import sanitized_load_custom_objects

    custom_objects, _dropped = sanitized_load_custom_objects(subject_zip, device)
    model = MaskablePPO.load(subject_zip, device=device, custom_objects=custom_objects)
    head = getattr(model.policy.features_extractor, "win_head", None)
    if head is None:
        raise ValueError("subject checkpoint carries no win_head — nothing to graft")
    ck = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
    want = {k: v.clone() for k, v in ck["head_state_dict"].items()}
    head.load_state_dict(want)

    os.makedirs(os.path.dirname(os.path.abspath(out_zip)) or ".", exist_ok=True)
    model.save(out_zip)
    for sib in ("model_config.json", "metadata.json"):
        src = _find_sibling(subject_zip, sib)
        if src:
            shutil.copyfile(src, os.path.join(os.path.dirname(os.path.abspath(out_zip)), sib))

    co2, _ = sanitized_load_custom_objects(out_zip, device)
    reloaded = MaskablePPO.load(out_zip, device=device, custom_objects=co2)
    got = reloaded.policy.features_extractor.win_head.state_dict()
    for k, v in want.items():
        if not torch.equal(got[k].cpu(), v.cpu()):
            raise RuntimeError(
                f"graft VERIFICATION FAILED: reloaded {out_zip} disagrees with the fitted head at "
                f"{k!r} — the saved checkpoint does not carry the fine-tune")
    return out_zip


def _find_sibling(ckpt_path: str, name: str) -> Optional[str]:
    """``name`` beside the checkpoint, else one level up (the run root) — snapshot.py's search."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for cand in (os.path.join(d, name), os.path.join(os.path.dirname(d), name)):
        if os.path.exists(cand):
            return cand
    return None


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agents.training.winprob_finetune",
        description="Offline HEAD-ONLY fine-tune of the win-probability head on harvest labels.")
    p.add_argument("harvest", help="a harvest directory or a single .jsonl.gz shard")
    p.add_argument("--subject", required=True, help="the checkpoint .zip whose head is repaired")
    p.add_argument("--out", default=None, help="output dir (default <harvest>/winprob_finetune)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--anchor-coef", type=float, default=0.3,
                   help="trust-region pull toward the SUBJECT's original head, in logit space "
                        "(mean squared logit delta). DEFAULTS ON: a harvest is a PRIORITIZED "
                        "sample, and an unconstrained fit collapses the head toward the sample "
                        "mean and wrecks calibration on the true population — measured on two "
                        "metered pilots. `--anchor-coef 0` opts out and reproduces them.")
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="approximate VAL fraction; split is by battle_tag, never by state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/head_last.pt")
    p.add_argument("--slice-reweight", choices=SLICE_MODES, default="inverse",
                   help=f"turn-slice inverse-frequency weights ({SLICE_VERSION}) or none")
    p.add_argument("--models-root", default=None,
                   help="root the rows' relative obs_npz resolve against "
                        "(default utils.paths.main_models_dir())")
    p.add_argument("--fresh-head", action="store_true",
                   help="random-init the head instead of warm-starting from the subject's")
    p.add_argument("--no-emit-zip", action="store_true",
                   help="skip writing the grafted full checkpoint (apply_head still works)")
    p.add_argument("--device", default="cpu")
    return p


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    args = build_parser().parse_args(argv)

    import torch
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from main.prober.model import ProbeModel
    from utils.paths import main_models_dir

    models_root = args.models_root or main_models_dir()
    out_dir = args.out or os.path.join(
        args.harvest if os.path.isdir(args.harvest) else os.path.dirname(args.harvest),
        "winprob_finetune")
    os.makedirs(out_dir, exist_ok=True)

    rows = read_dir(args.harvest)
    print(f"harvest: {len(rows)} rows / {len({r['battle_tag'] for r in rows})} battles "
          f"from {args.harvest}", flush=True)
    if not rows:
        print("winprob_finetune: no rows — nothing to fit", file=sys.stderr)
        return 2

    print(f"loading subject {args.subject} on {args.device} ...", flush=True)
    probe = ProbeModel.load(args.subject, device=args.device)
    ds, rep = precompute_value_pooled(
        probe_pooled_fn(probe), rows, models_root=models_root,
        batch_size=max(1, args.batch_size), slice_mode=args.slice_reweight)
    if len(ds) == 0:
        print("winprob_finetune: every row was rejected — refusing to fit", file=sys.stderr)
        return 2

    train, val = split_pooled(ds, args.holdout_frac, args.seed, args.slice_reweight)
    print(f"split: train {len(train)} rows / {len(set(train.battle_tag))} battles"
          f" | val {len(val)} rows / {len(set(val.battle_tag))} battles", flush=True)
    print(f"label noise sd: train {label_noise_sd(train.wins, train.n_rollouts):.4f}"
          f" | val {label_noise_sd(val.wins, val.n_rollouts):.4f}"
          f"  (the floor no head can beat)", flush=True)

    subject_head = getattr(probe._policy.features_extractor, "win_head", None)
    warm = None if (args.fresh_head or subject_head is None) else subject_head.state_dict()
    head = build_head(warm)
    if warm is not None:
        base = evaluate(head, val)
        print(f"subject head, before the fit: val_nll {base['nll']:.5f}"
              f"  brier {base['brier']:.5f} (excess {base['brier_excess']:.5f})"
              f"  ece {base['ece']:.4f}", flush=True)

    cfg = FitConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    weight_decay=args.weight_decay, seed=args.seed,
                    slice_reweight=args.slice_reweight, holdout_frac=args.holdout_frac,
                    anchor_coef=args.anchor_coef)
    res = fit_head(head, train, val, cfg, out_dir=out_dir, subject_ckpt=args.subject,
                   resume=args.resume)
    print(f"best epoch {res.best_epoch} at val_nll {res.best_val_nll:.5f} "
          f"-> {os.path.join(out_dir, 'head_best.pt')}", flush=True)

    best_path = os.path.join(out_dir, "head_best.pt")
    best_bins: List[Dict[str, float]] = []
    if os.path.exists(best_path):
        best_head = build_head(torch.load(best_path, map_location="cpu",
                                          weights_only=False)["head_state_dict"])
        best_head.eval()
        with torch.no_grad():
            phi = torch.sigmoid(_logits(best_head, torch.as_tensor(val.x))).numpy()
        best_bins = calibration_bins(phi, val.wins, val.n_rollouts)
        print("val reliability (best epoch) — bin | rows | rollouts | pred | actual", flush=True)
        for b in best_bins:
            print(f"  [{b['lo']:.1f},{b['hi']:.1f})  {int(b['rows']):5d}  {int(b['rollouts']):7d}"
                  f"  {b['pred']:.3f}  {b['actual']:.3f}", flush=True)

    zip_path = None
    if not args.no_emit_zip:
        try:
            zip_path = graft_into_checkpoint(
                args.subject, os.path.join(out_dir, "head_best.pt"),
                os.path.join(out_dir, "subject_winprob_finetuned.zip"), device=args.device)
            print(f"grafted checkpoint verified: {zip_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 — the head weights are already safe on disk
            print(f"winprob_finetune: could not emit a grafted zip ({exc}); use "
                  f"apply_head(probe, {os.path.join(out_dir, 'head_best.pt')!r})", file=sys.stderr)

    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump({
            "subject": args.subject, "harvest": args.harvest,
            "config": asdict(cfg), "slice_version": SLICE_VERSION,
            "slice_edges": list(SLICE_EDGES),
            "rows": {"loaded": rep.loaded, "rejected": rep.rejected,
                     "digest_mismatch": rep.digest_mismatch,
                     "obs_dim_mismatch": rep.obs_dim_mismatch, "other": rep.other,
                     "examples": rep.examples},
            "train_rows": len(train), "val_rows": len(val),
            "train_battles": len(set(train.battle_tag)), "val_battles": len(set(val.battle_tag)),
            "label_noise_sd_train": res.label_noise_sd_train,
            "label_noise_sd_val": res.label_noise_sd_val,
            "best_epoch": res.best_epoch, "best_val_nll": res.best_val_nll,
            "history": [asdict(h) for h in res.history],
            "val_calibration_best_epoch": best_bins,
            "grafted_zip": zip_path,
        }, fh, indent=2)
    print(f"report -> {os.path.join(out_dir, 'report.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
