"""The RIDE-ALONG step (`gen3_ridealong_heads_v1`) — the heads' own update, OUTSIDE the PPO fold.

`agents.model.ridealong_heads` holds the heads and their losses; this mixin runs them inside
`train()` without touching what PPO learns:

  * `_ridealong_update` runs RIGHT AFTER the minibatch's `evaluate_actions` forward. It snapshots the
    forward's stashes through `RideAlongBatch.detached` (THE stop-grad seam), computes the heads'
    losses, backpropagates them into the heads ALONE, clips the heads' own gradient, steps the
    heads' OWN Adam, and sets their gradients back to None — all before PPO's loss is even
    assembled. So `loss`, PPO's `clip_grad_norm_` (whose total norm would otherwise include the
    heads' gradients and rescale the trunk's), the noise-scale probes
    never see a ride-along gradient, and nothing the heads do consumes a random number.
  * `RideAlongAccumulator` folds the per-minibatch readouts into the `ridealong/*` TB family. The
    RANK meters (AUROC, Spearman, error by decile — does uncertainty predict V's actual error?) need
    the whole rollout. The heads train and are read on epoch 0 only (`RIDEALONG_EPOCHS`): one
    policy, every row once, each scored before the step that trains on it.

ONE HOST READ PER MICRO-BATCH (T25 next lever 1, 2026-10-10). The step used to make ~120 device reads
per epoch-0 micro-batch — each a host sync on CUDA, the first draining the deep queue — which kept the eager
tail from overlapping the GPU (window C: epoch 0 ran 85.6 % GPU-busy against ~95 % later). Now: the RND
statistics branch on host MIRRORS (`ridealong_heads._HostMirror`), the label guards are device GATES
(`RideAlongHeads.losses_gated`), the meters are DEFERRED snapshots folded once past the ride-along epoch
(`RideAlongAccumulator`), and `_ridealong_step` reads every clip norm and gate in ONE transfer. Every number
logged and every weight stepped is bit-identical to the old path (`ridealong_host_sync_test.py` runs HEAD's
code as the oracle and counts the syncs; a read coming back fails it).

THE RND VARIANTS (`gen3_ridealong_rnd_variants_v1`, v127) step right after the four heads, each on
its OWN Adam (rate RIDEALONG_LR × its declared `lr_mult`), its OWN gradient clip and its OWN
fail-closed switch (a non-finite variant stops alone; the four heads and PPO carry on). The `decay`
variant's pull toward init runs once per PPO update, before that update's first step
(`RideAlongHeads.begin_update_`; the per-`train()` accumulator marks the boundary). Each variant
logs `ridealong/rndv_<name>_*`: loss, error mean / median / IQR / relative spread (IQR ÷ median, the
SATURATION series), z by opponent class, the V-error meters, and — for base and the observation
variants — `*_ident_ratio`, the error on the batch's block chimeras ÷ the error on its real rows
(the IDENTIFICATION monitor: it stays high while a predictor learns only what it visits).

The optimizers are the LEARNER's, not the model's, and they are ACQUIRED AT STARTUP (the declared
lifecycle, owner 2026-09-28; the K8 inventory flagged the old lazy build): `_setup_model` ends with
`_ridealong_acquire`, which builds the four heads' Adam and every variant's Adam with their state
PRE-ALLOCATED (`preallocate_adam_state`: identical to torch's lazy first-step init, so the steps are
bit-identical). A steady-state step only READS them: there is NO lazy build path. A step that
finds an optimizer missing (or bound to other heads) raises `RideAlongLifecycleViolation`, the same
typed refusal K6.1's freeze guard gives any post-startup acquisition. `ridealong_update_test`'s
lifecycle tests pin this: the raise, and stable optimizer, state and buffer identities over real updates. The optimizers are not in the checkpoint, so a restart resumes the heads'
weights with a fresh Adam state (bias-corrected, so the first steps are ~lr-sized). Declared, not
hidden: the heads observe, and a restart every few hours costs them a few noisy steps and PPO
nothing. The heads' step is K8's candidate compile region R-ride (`ridealong_step`, train/grad ×
micro-batch, epoch 0, with its own optimizer region); it stays EAGER for now.
"""
from __future__ import annotations

import math
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from agents.training.instrumented_ppo.host_reads import HostReadQueue
from agents.training.lifecycle_decl import startup_builder

#: The heads' own Adam. Upstream Adam defaults except eps (SB3's 1e-5); the rate is PPO's order of
#: magnitude, fixed — the heads' learning speed is not an experimental lever of the baseline.
RIDEALONG_LR = 3e-4
RIDEALONG_EPS = 1e-5
#: How many of PPO's epochs the heads TRAIN on. ONE: each rollout row is seen once, like any
#: streaming learner on an on-policy stream. Measured on the GPU learner benchmark (2026-09-30, arm C's
#: real 98,304-row buffer): training on all 10 epochs cost +8.9 s on a 67.0 s update (+13 %), far
#: past the design's ~2 % budget for an instrument (`design_q_head.md` §8). One pass MEASURED +0.59 s =
#: 0.88 % (`ridealong_step_benchmark.py`; X26's PREREGISTRATION.md "Overhead"). Every meter was already epoch 0's.
RIDEALONG_EPOCHS = 1
#: The heads' own gradient clip (their norm only).
RIDEALONG_MAX_GRAD_NORM = 1.0
#: "V is on the wrong side": the error a disagreement / novelty AUROC is read against. |V − z| > 0.5
#: is exactly "V's rounded prediction was the wrong outcome".
ERR_THRESHOLD = 0.5
#: An action the policy STARVES: π below this (UNDERSTANDING §4.4's starvation read uses 1 %).
STARVED_PI = 0.01


def rank_auroc(score: np.ndarray, label: np.ndarray) -> Optional[float]:
    """AUROC of ``score`` for the binary ``label`` (Mann–Whitney, average ranks for ties); None when
    either class is empty."""
    from scipy.stats import rankdata

    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(spearmanr(x, y).statistic)


def error_by_decile(score: np.ndarray, err: np.ndarray) -> List[float]:
    """Mean ``err`` in each decile of ``score`` (lowest decile first)."""
    order = np.argsort(score, kind="stable")
    return [float(err[c].mean()) for c in np.array_split(order, 10) if len(c)]


def uncertainty_meters(prefix: str, score: np.ndarray, err: np.ndarray) -> Dict[str, float]:
    """Does ``score`` (a disagreement or a novelty) predict V's actual error ``err`` = |V − z|?"""
    out: Dict[str, float] = {}
    au = rank_auroc(score, err > ERR_THRESHOLD)
    if au is not None:
        out[f"{prefix}_auroc_err"] = au
    rho = spearman(score, err)
    if rho is not None:
        out[f"{prefix}_spearman_err"] = rho
    if len(score) >= 10:
        dec = error_by_decile(score, err)
        out[f"{prefix}_err_bottom_decile"] = dec[0]
        out[f"{prefix}_err_top_decile"] = dec[-1]
    return out


def rnd_prefixes(out: Dict[str, Any]) -> List[str]:
    """The RND keys a readout carries: ``rnd`` (base) and ``rndv_<name>`` per variant, each with
    both ``<key>_err`` and ``<key>_z``."""
    keys = ["rnd"] + sorted(k[: -len("_err")] for k in out
                            if k.startswith("rndv_") and k.endswith("_err"))
    return [k for k in keys if f"{k}_err" in out and f"{k}_z" in out]


#: The `out` keys `_observe_now` reads by PRESENCE only. A snapshot keeps the key and no tensor (None), so a
#: future edit that reads their DATA raises instead of reading a stale or empty stand-in.
_PRESENCE_ONLY = ("adv", "opp")
#: The batch fields `_observe_now` reads (`b.<field>`); a snapshot carries exactly these.
_OBSERVE_BATCH_FIELDS = ("values", "win_target", "win_mask", "actions", "advantages", "legal", "logits",
                         "pi", "flat_target")


class RideAlongAccumulator:
    """Per-train() sink for the `ridealong/*` family.

    **The meters are DEFERRED and READ ONCE** (T25 next lever 1). `observe` / `observe_identification` used
    to read the device ~100 times per micro-batch (every `float(t)`, every `bool(t.any())`, every boolean-mask
    index, every `.cpu()` of a column): on CUDA each is a host sync, and together they kept the eager
    ride-along tail from overlapping the GPU queue (window C: epoch 0 ran 85.6 % GPU-busy against ~95 % in
    later epochs). They now only SNAPSHOT the detached tensors they will read (device copies, no read).
    `flush()` then folds every snapshot, in call order: the statistics are computed on the device, each
    scalar is QUEUED (`host_reads.HostReadQueue`) with the host arithmetic that consumes it, and the whole
    fold is read back in ONE transfer. What stays data-dependent is only the compaction of a masked set (one
    `nonzero` per mask, shared by every statistic that uses it). Every logged number is the value it was, bit
    for bit: the same ops on the same inputs, the same float32 values, the same host arithmetic in the same
    order. The learner flushes at the first micro-batch past the ride-along epoch
    (`RideAlongTerms._ridealong_update`), and `metrics()` flushes whatever is left. Read the accumulator only
    through `metrics()` / after `flush()`."""

    def __init__(self) -> None:
        self.scalars: Dict[str, List[float]] = {}
        self.cols: Dict[str, List[np.ndarray]] = {}
        #: Set by the learner's first ride-along step of this `train()` call: the PPO-update boundary
        #: the `decay` variant's pull keys on (one accumulator per update).
        self.update_begun = False
        #: [sum of real-row error, sum of chimera-row error] per RND key (`rnd` / `rndv_<name>`).
        self.ident: Dict[str, List[float]] = {}
        #: Snapshots not yet folded, in call order: ("observe", args) / ("ident", args).
        self._pending: List[Tuple[str, Tuple[Any, ...]]] = []
        #: The fold's read queue while `flush()` runs (None otherwise: `col` then reads at once).
        self._fold_q: Optional[HostReadQueue] = None

    def add(self, key: str, v: float) -> None:
        if v == v:                                    # a NaN is REPORTED by omission, never logged
            self.scalars.setdefault(key, []).append(float(v))

    def col(self, key: str, arr: th.Tensor) -> None:
        """Append a per-row column (host float32). During a fold the copy rides the fold's one transfer."""
        flat = arr.detach().float()
        q = self._fold_q
        if q is None:
            self.cols.setdefault(key, []).append(flat.cpu().numpy())
            return
        shape = tuple(flat.shape)
        q.push(flat, lambda a: self.cols.setdefault(key, []).append(a.reshape(shape).copy()))

    def observe(self, out: Dict[str, th.Tensor], b: Any, losses: Dict[str, th.Tensor],
                opp_class: Optional[th.Tensor], epoch: int) -> None:
        """One micro-batch's readout, DEFERRED: snapshot now (no device read), fold at `flush()`.
        The batch's fields and ``opp_class`` are COPIED (they are views of the rollout buffer, which a
        staged device-batch mode refills); ``out`` and ``losses`` are fresh tensors of this step."""
        snap_out: Dict[str, Optional[th.Tensor]] = {
            k: (None if k in _PRESENCE_ONLY else v.detach()) for k, v in out.items()}
        snap_b = types.SimpleNamespace(**{
            f: (None if getattr(b, f, None) is None else getattr(b, f).detach().clone())
            for f in _OBSERVE_BATCH_FIELDS})
        snap_loss = {k: v.detach() for k, v in losses.items()}
        snap_opp = None if opp_class is None else opp_class.detach().clone()
        self._pending.append(("observe", (snap_out, snap_b, snap_loss, snap_opp)))

    def discard_losses(self, keys: Tuple[str, ...]) -> None:
        """Take ``keys`` out of the NEWEST pending `observe` snapshot's losses. A head with no usable
        label in a micro-batch is not a loss there (`losses` omits its key, so HEAD never logged a
        ``<key>_loss`` for it); the step learns that from the label gates it reads anyway, after the
        snapshot was taken."""
        if not keys:
            return
        for kind, args in reversed(self._pending):
            if kind == "observe":
                for k in keys:
                    args[2].pop(k, None)
                return

    def observe_identification(self, out: Dict[str, th.Tensor], chim: Dict[str, th.Tensor]) -> None:
        """Fold one minibatch's identification probe: the real rows' error (``out``) and the same
        predictors' error on the rows' block chimeras (``chim``), summed per RND key. DEFERRED like
        `observe`."""
        pairs = [(pre, out[f"{pre}_err"].detach(), ec.detach()) for pre, ec in chim.items()
                 if out.get(f"{pre}_err") is not None]
        self._pending.append(("ident", (pairs,)))

    def flush(self) -> None:
        """Fold every pending snapshot, in call order, and read the whole fold back in ONE transfer.
        The ONLY place the meters read the device."""
        pending, self._pending = self._pending, []
        if not pending:
            return
        rq = HostReadQueue()
        self._fold_q = rq
        try:
            for kind, args in pending:
                if kind == "observe":
                    self._observe_now(*args, rq)
                else:
                    self._observe_identification_now(*args, rq)
            rq.drain()
        finally:
            self._fold_q = None

    def _observe_now(self, out: Dict[str, Any], b: Any, losses: Dict[str, th.Tensor],
                     opp_class: Optional[th.Tensor], rq: HostReadQueue) -> None:
        """One micro-batch's meters: HEAD's body, with every ``add(key, float(t))`` queued on ``rq`` (the
        host arithmetic after a read rides in its callback) and every masked set compacted by ONE shared
        ``nonzero`` + ``index_select`` (the elements `t[mask]` selects, in its order). An empty selection's
        mean is NaN, which `add` omits — what HEAD's ``if mask.any()`` guards did."""
        from agents.model.opp_intent import INTENT_IGNORE, OPP_CLASS_NAMES

        add = self.add

        def read(t: th.Tensor, apply: Callable[[np.ndarray], None]) -> None:
            rq.push(t.detach().float().reshape(-1), apply)

        def scalar(key: str, t: th.Tensor) -> None:                    # add(key, float(t))
            read(t, lambda a: add(key, float(a[0])))

        for k, v in losses.items():
            scalar(f"{k}_loss", v)
        with th.no_grad():
            v_ = b.values.float().reshape(-1) if b.values is not None else None
            z = b.win_target.float().reshape(-1) if b.win_target is not None else None
            zm = (b.win_mask.float().reshape(-1) > 0) if b.win_mask is not None else None
            zi = zm.nonzero(as_tuple=True)[0] if zm is not None else None      # the rows `x[zm]` selects
            if "ens_std" in out:
                scalar("ens_disagreement_mean", out["ens_std"].mean())
                if z is not None and zi is not None:
                    p = th.sigmoid(out["ens_logits"]).index_select(0, zi)
                    zs = z.index_select(0, zi)
                    brier = (p - zs[:, None]).pow(2).mean(0)
                    read(brier, lambda a: [add(f"ens_brier_head_{j}", float(x)) for j, x in enumerate(a)])
                    scalar("ens_brier_mean_member", (out["ens_p"].index_select(0, zi) - zs).pow(2).mean())
                    if v_ is not None:
                        scalar("ens_brier_v", (v_.index_select(0, zi) - zs).pow(2).mean())
            # base (`rnd`) and every RND variant (`rndv_<name>`): the same per-row statistics.
            class_rows: Dict[int, th.Tensor] = {}
            oc = opp_class.long().reshape(-1) if opp_class is not None else None
            for pre in rnd_prefixes(out):
                scalar(f"{pre}_err_mean", out[f"{pre}_err"].mean())
                scalar(f"{pre}_z_mean", out[f"{pre}_z"].mean())
                # every row's RAW error: the update's median / IQR (the saturation series)
                self.col(f"{pre}_err_raw", out[f"{pre}_err"])
                if oc is not None:
                    names = OPP_CLASS_NAMES.items() if isinstance(OPP_CLASS_NAMES, dict) else \
                        enumerate(OPP_CLASS_NAMES)
                    for code, name in names:
                        if int(code) not in class_rows:
                            class_rows[int(code)] = (oc == int(code)).nonzero(as_tuple=True)[0]
                        scalar(f"{pre}_z_{name}", out[f"{pre}_z"].index_select(0, class_rows[int(code)]).mean())
            if "adv" in out and b.actions is not None and b.advantages is not None:
                a = b.actions.long().reshape(-1)
                y = b.advantages.float().reshape(-1)
                pred = out["adv_mean"].gather(-1, a[:, None]).squeeze(-1)

                def explained_var(h: np.ndarray) -> None:
                    var_y = float(h[0])
                    if var_y > 0:
                        add("adv_explained_var", 1.0 - float(h[1]) / var_y)
                read(th.stack([y.var(unbiased=False), (y - pred).var(unbiased=False)]), explained_var)
                if b.legal is not None and b.logits is not None and b.pi is not None:
                    legal = b.legal.bool()
                    n_legal = legal.float().sum(-1, keepdim=True).clamp(min=1.0)
                    lg = th.where(legal, b.logits.float(), th.zeros_like(b.logits.float()))
                    am = th.where(legal, out["adv_mean"], th.zeros_like(out["adv_mean"]))
                    lc = th.where(legal, lg - lg.sum(-1, keepdim=True) / n_legal, th.zeros_like(lg))
                    ac = th.where(legal, am - am.sum(-1, keepdim=True) / n_legal, th.zeros_like(am))

                    def corr(h: np.ndarray) -> None:
                        den = float(h[0])
                        if den > 0:
                            add("adv_corr_logit", float(h[1]) / den)
                    read(th.stack([lc.pow(2).sum().sqrt() * ac.pow(2).sum().sqrt(), (lc * ac).sum()]), corr)
                    std = out["adv_std"]
                    starved = legal & (b.pi < STARVED_PI)
                    fed = legal & (b.pi >= STARVED_PI)
                    scalar("adv_std_starved", std[starved].mean())
                    scalar("adv_std_fed", std[fed].mean())
                    if v_ is not None:
                        q = v_[:, None] + out["adv_mean"]
                        scalar("q_out_of_range", ((q < 0) | (q > 1))[legal].float().mean())
            if "opp" in out and b.flat_target is not None:
                # X5 U4: the share of rows B trains on — OTHER rows included (a belief miss is a label)
                scalar("opp_label_rate", (b.flat_target != INTENT_IGNORE).float().mean())
            if v_ is not None and z is not None and zi is not None:
                err = (v_ - z).abs()
                for key in ("ens_logit_std", *(f"{p}_z" for p in rnd_prefixes(out))):
                    if key in out:
                        self.col(key, out[key].index_select(0, zi))
                # V's own binary entropy, so the reader of `ens_*_err` can tell whether the
                # disagreement adds anything beyond V's own uncertainty (the level confound).
                pv = v_.index_select(0, zi).clamp(1e-6, 1 - 1e-6)
                self.col("v_entropy", -(pv * pv.log() + (1 - pv) * (1 - pv).log()))
                self.col("v_err", err.index_select(0, zi))

    def _observe_identification_now(self, pairs: List[Tuple[str, th.Tensor, th.Tensor]],
                                    rq: HostReadQueue) -> None:
        for pre, er, ec in pairs:
            acc = self.ident.setdefault(pre, [0.0, 0.0])

            def fold(h: np.ndarray, acc: List[float] = acc) -> None:
                acc[0] += float(h[0])
                acc[1] += float(h[1])
            rq.push(th.stack([er.detach().float().sum(), ec.detach().float().sum()]), fold)

    def metrics(self) -> Dict[str, float]:
        self.flush()
        out = {k: float(np.mean(v)) for k, v in self.scalars.items() if v}
        cols = {k: np.concatenate(v) for k, v in self.cols.items() if v}
        err = cols.get("v_err")
        if err is not None and len(err):
            if "ens_logit_std" in cols:
                out.update(uncertainty_meters("ens", cols["ens_logit_std"], err))
            if "v_entropy" in cols:
                out.update(uncertainty_meters("ref_v_entropy", cols["v_entropy"], err))
            for key in [k for k in cols if k.endswith("_z") and k.startswith("rnd")]:
                pre = key[:-2]
                out.update(uncertainty_meters(pre, cols[key], err))
                out[f"{pre}_z_p90"] = float(np.quantile(cols[key], 0.9))
        for key in [k for k in cols if k.endswith("_err_raw")]:
            # SATURATION (X26 amendment (c)): the RAW error's median and spread over every row this
            # update scored BEFORE training on it. A predictor that has saturated reads every fresh
            # row as familiar: the spread collapses. `rel_spread` (IQR ÷ median) is scale-free.
            pre = key[: -len("_err_raw")]
            e = cols[key]
            if len(e):
                q25, q50, q75 = (float(x) for x in np.quantile(e, [0.25, 0.5, 0.75]))
                out[f"{pre}_err_median"] = q50
                out[f"{pre}_err_iqr"] = q75 - q25
                if q50 > 0:
                    out[f"{pre}_err_rel_spread"] = (q75 - q25) / q50
        for pre, (real, chim) in self.ident.items():
            if real > 0:
                out[f"{pre}_ident_ratio"] = chim / real
        return out


def preallocate_adam_state(opt: th.optim.Optimizer) -> None:
    """Allocate every parameter's Adam state NOW, exactly as torch's lazy first-step init would.

    One step on ZERO gradients does torch's own `_init_group` (whatever the torch version, fused
    or not): the moments are created as zeros and stay zeros (β·0 + (1 − β)·0), the parameter
    update is exactly 0 (a zero first moment over a positive denominator), and the step counter is
    then reset to 0. So the state afterwards is exactly what the first real step would have created
    at entry, and every later step is bit-identical to a lazily-initialised optimizer's
    (`ridealong_update_test` pins it). The gradients go back to None."""
    params = [p for g in opt.param_groups for p in g["params"]]
    with th.no_grad():
        for p in params:
            p.grad = th.zeros_like(p)
        opt.step()
        for p in params:
            p.grad = None
            st = opt.state.get(p)
            if st and "step" in st:
                step = st["step"]
                if th.is_tensor(step):
                    step.zero_()
                else:
                    st["step"] = 0


class RideAlongLifecycleViolation(RuntimeError):
    """A ride-along step found no optimizer acquired at startup for the heads it is stepping. The
    declared lifecycle has no lazy fallback (owner 2026-09-28; K6.1's freeze guard): acquire with
    `_ridealong_acquire()` — `_setup_model` does, and so must any tool that swaps `policy.ridealong`."""


@startup_builder
def _adam(params: List[th.nn.Parameter], lr: float) -> th.optim.Optimizer:
    """The heads' Adam, its state allocated NOW (startup only — `_ridealong_acquire`)."""
    # fused on CUDA: one kernel for the whole step instead of several per parameter tensor (the heads
    # are many small tensors, so the per-tensor launches were the cost).
    fused = bool(params) and all(p.is_cuda for p in params)
    opt = th.optim.Adam(params, lr=lr, eps=RIDEALONG_EPS, fused=fused or None)
    preallocate_adam_state(opt)
    return opt


class RideAlongTerms:
    """Mixin: the heads' own step and their `ridealong/*` export."""

    def _ridealong_heads(self) -> Any:
        return getattr(self.policy, "ridealong", None)   # type: ignore[attr-defined]

    def _setup_model(self) -> None:
        """SB3's model setup (the policy is built and on its device), then the ride-along heads'
        STARTUP ACQUISITION. Runs on a fresh build and on `load` (whose `set_parameters` copies into
        the same parameter objects afterwards, so the optimizers stay bound)."""
        super()._setup_model()   # type: ignore[misc]
        self._ridealong_acquire()

    def _ridealong_acquire(self) -> None:
        """THE acquisition site: the four heads' optimizer and every RND variant's, with their Adam
        state pre-allocated, for the heads the policy carries NOW. Tooling that swaps the heads (the
        benchmarks) calls it again explicitly; the training step never builds anything."""
        heads = self._ridealong_heads()
        self._ridealong_opt = None
        self._ridealong_opt_owner = None
        self._ridealong_vopts: Dict[str, th.optim.Optimizer] = {}
        self._ridealong_vopts_owner = None
        self._ridealong_variant_disabled: set = set()
        if heads is None:
            return
        self._ridealong_opt = _adam(heads.trainable_parameters(), RIDEALONG_LR)
        self._ridealong_opt_owner = heads
        self._ridealong_vopts = {n: self._variant_adam(heads, n) for n in heads.variant_names()}
        self._ridealong_vopts_owner = heads

    @staticmethod
    def _variant_adam(heads: Any, name: str) -> th.optim.Optimizer:
        from agents.model.ridealong_heads import RND_VARIANT_BY_NAME

        return _adam(heads.variant_parameters(name),
                     RIDEALONG_LR * RND_VARIANT_BY_NAME[name].lr_mult)

    def _ridealong_optimizer(self, heads: Any) -> th.optim.Optimizer:
        """The four heads' optimizer, acquired at startup. READ ONLY: nothing is built here."""
        opt = getattr(self, "_ridealong_opt", None)
        if opt is None or getattr(self, "_ridealong_opt_owner", None) is not heads:
            raise RideAlongLifecycleViolation(
                "the ride-along heads' optimizer was not acquired at startup for these heads — call "
                "_ridealong_acquire() after building or swapping policy.ridealong")
        return opt

    def _ridealong_update(self, rollout_data: Any, values: th.Tensor, actions: th.Tensor,
                          epoch: int, acc: RideAlongAccumulator) -> None:
        """One ride-along step on this minibatch's forward. A no-op (one attribute read) when the
        policy has no heads. **One host read on the micro-batch path** (T25 next lever 1): the RND
        statistics branch on host mirrors, the meters are deferred snapshots (`RideAlongAccumulator`),
        and the finiteness verdicts, the heads' label gates and the logged norms are read together in
        `_ridealong_step`'s single transfer. The first micro-batch PAST the ride-along epoch folds the
        deferred meters (`acc.flush()`)."""
        heads = self._ridealong_heads()
        if epoch >= RIDEALONG_EPOCHS:
            acc.flush()
            return
        if heads is None or getattr(self, "_ridealong_disabled", False):
            return
        from agents.model.ridealong_heads import RideAlongBatch

        fe = self.policy.features_extractor                  # type: ignore[attr-defined]
        obs = rollout_data.observations
        dist = getattr(self.policy, "_last_pi_distribution", None)   # type: ignore[attr-defined]
        pi = logits = legal = None
        if dist is not None:
            pi = dist.distribution.probs
            legal = pi > 0
            logits = dist.distribution.logits
        opp_on = float(getattr(self, "opp_intent_coef", 0.0) or 0.0) > 0.0   # labels ALIGNED
        # X5 U4: B reads the FLAT pointer — its logits, its candidate ids and the label
        # column from THE target function the intent loss uses (so B and the loss name one column).
        _fi = getattr(fe, "last_flat_intent", None) if opp_on else None
        _flat_kw: Dict[str, Any] = {}
        if _fi is not None and "opp_action_kind" in obs:
            from agents.model.flat_intent import flat_intent_targets
            _ft, _ = flat_intent_targets(_fi, obs["opp_action_kind"], obs["opp_action_num"],
                                         obs["opp_switch_slot"], obs["opp_switch_species"])
            _flat_kw = dict(flat_logits=fe.last_flat_intent_logits, flat_ids=_fi.cand_ids,
                            flat_target=_ft)
        b = RideAlongBatch.detached(
            obs=obs["observation"], pooled=fe.last_value_pooled,
            pointer=tuple(fe.last_pointer_inputs) if fe.last_pointer_inputs is not None else None,
            pi=pi, logits=logits, legal=legal, values=values,
            actions=actions, advantages=rollout_data.advantages,
            win_target=obs.get("win_target"), win_mask=obs.get("win_mask"),
            **_flat_kw)
        if b.pooled is None:
            return
        rnd = getattr(heads, "rnd", None)
        variants = getattr(heads, "rnd_variants", None)
        dead = getattr(self, "_ridealong_variant_disabled", None) or set()
        if not acc.update_begun:
            # The PPO-update boundary: the `decay` variant's once-per-update pull toward its init.
            acc.update_begun = True
            if hasattr(heads, "begin_update_"):
                heads.begin_update_(skip=tuple(dead))
        if rnd is not None and epoch == 0:
            # Burda et al.: normalise with statistics that include this batch, then score it
            # BEFORE the predictor sees it — epoch 0 is each row's first (and only) RND visit.
            rnd.update_obs_stats(b.obs)
            if variants is not None and "feat" in variants and "feat" not in dead:
                variants["feat"].update_obs_stats(b.pooled)      # its OWN feature normalisation
        out = heads.readout(b, skip_variants=tuple(dead)) if variants is not None \
            else heads.readout(b)
        losses, gates = heads.losses_gated(b, out, train_rnd=(epoch == 0))
        vlosses = heads.variant_losses(out) if variants is not None else {}
        if rnd is not None and epoch == 0 and "rnd_err" in out:
            rnd.update_err_stats(out["rnd_err"])
            # z against statistics that INCLUDE this batch (the first batch of a run would
            # otherwise be scored against the placeholder mean 0 / variance 1).
            out["rnd_z"] = rnd.zscore(out["rnd_err"])
            for n, vmod in (variants.items() if variants is not None else ()):
                if f"rndv_{n}_err" in out:
                    vmod.update_err_stats(out[f"rndv_{n}_err"])
                    out[f"rndv_{n}_z"] = vmod.zscore(out[f"rndv_{n}_err"])
        # The READ is epoch 0's: one policy over every rollout row once, and — for the losses —
        # each row scored BEFORE the heads trained on it this call. Later epochs only train, so the
        # meters cost host syncs once per row, not n_epochs times — and since T25 next lever 1 not even
        # once per micro-batch: `observe` snapshots, `acc.flush()` reads.
        if epoch == 0:
            acc.observe(out, b, {**losses, **vlosses}, obs.get("opp_class"), epoch)
            if variants is not None:
                # The identification probe (no grad, no RNG): the same predictors on the batch's
                # block chimeras, scored BEFORE this step trains on the real rows.
                chim = heads.identification_errors(b.obs)
                acc.observe_identification(out, {k: v for k, v in chim.items()
                                                 if k[len("rndv_"):] not in dead})
        acc.discard_losses(self._ridealong_step(heads, losses, gates, vlosses, epoch, acc))

    def _ridealong_variant_optimizer(self, heads: Any, name: str) -> th.optim.Optimizer:
        """The variant's OWN Adam (rate RIDEALONG_LR × its declared `lr_mult`), acquired at startup.
        READ ONLY: nothing is built here."""
        opts: Optional[Dict[str, th.optim.Optimizer]] = getattr(self, "_ridealong_vopts", None)
        if (opts is None or getattr(self, "_ridealong_vopts_owner", None) is not heads
                or name not in opts):
            raise RideAlongLifecycleViolation(
                f"RND variant {name!r}'s optimizer was not acquired at startup for these heads — "
                "call _ridealong_acquire() after building or swapping policy.ridealong")
        return opts[name]

    def _ridealong_step(self, heads: Any, losses: Dict[str, th.Tensor], gates: Dict[str, th.Tensor],
                        vlosses: Dict[str, th.Tensor], epoch: int, acc: RideAlongAccumulator
                        ) -> Tuple[str, ...]:
        """The RND variants' steps and the four heads' step, behind ONE device->host transfer.

        (a) Enqueue, reading nothing: the variants' backward (ONE over the SUM of their losses: the
        parameter sets are disjoint and every input is detached, so each predictor receives exactly its
        own loss's gradient) and per-variant clip; then the four heads' backward and clip.
        (b) ONE read — every clip's total norm and every label gate (`losses_gated`).
        (c) Per variant, then the heads: a non-finite norm is never stepped (fail-closed, below); a
        finite one logs its norm, steps its own Adam, and sets its grads back to None.

        Nothing irreversible happens before (b): no Adam step, no flag. So the two verdicts that need a
        host value — "is this norm finite" and "does this head have a usable label this batch" — cost one
        transfer instead of ~10 syncs, and the host enqueues the whole eager tail before it first waits
        on the GPU. The old order stepped the variants before the heads' backward; the two parameter sets
        and their graphs are disjoint (nothing in either reads the other), so the result is the same.

        Returns the loss keys whose gate was False (a head with no usable label in this micro-batch:
        not a loss here), for the caller to take out of the snapshot of the meters."""
        from agents.training.batched_reads import read_floats

        names = [k[len("rndv_"):] for k in vlosses]
        vopts = {n: self._ridealong_variant_optimizer(heads, n) for n in names}
        dead = getattr(self, "_ridealong_variant_disabled", None)
        if dead is None:
            dead = set()
            self._ridealong_variant_disabled = dead
        # (a) enqueue
        norms: List[th.Tensor] = []
        if vlosses:
            for o in vopts.values():
                o.zero_grad(set_to_none=True)
            th.stack(list(vlosses.values())).sum().backward()
            norms += [th.nn.utils.clip_grad_norm_(heads.variant_parameters(n), RIDEALONG_MAX_GRAD_NORM)
                      for n in names]
        opt = params = None
        if losses:
            opt = self._ridealong_optimizer(heads)
            params = heads.trainable_parameters()
            opt.zero_grad(set_to_none=True)
            # The graph is kept only when a gate may overrule this backward (the rare path below).
            th.stack(list(losses.values())).sum().backward(retain_graph=bool(gates))
            norms.append(th.nn.utils.clip_grad_norm_(params, RIDEALONG_MAX_GRAD_NORM))
        if not norms:
            return ()
        gate_keys = [k for k in losses if k in gates]
        # (b) the one read
        vals = read_floats(norms + [gates[k].float() for k in gate_keys])
        # (c) the variants
        for i, n in enumerate(names):
            g = vals[i]
            if not math.isfinite(g):
                # FAIL-CLOSED FOR THIS VARIANT ONLY: never stepped, stopped for the rest of the
                # process, loudly; the four heads, the other variants and PPO carry on.
                vopts[n].zero_grad(set_to_none=True)
                dead.add(n)
                print(f"🛑 [RIDE-ALONG] RND variant {n!r}: non-finite loss/gradient (grad norm "
                      f"{g}, loss {float(vlosses['rndv_' + n].detach())}) — DISABLED for "
                      "the rest of this process; the other heads and PPO are unaffected.", flush=True)
                continue
            if epoch == 0:
                acc.add(f"rndv_{n}_grad_norm", g)
            vopts[n].step()
            vopts[n].zero_grad(set_to_none=True)
        if not losses or opt is None or params is None:
            return ()
        # (c) the four heads
        g = vals[len(names)]
        gate_vals = dict(zip(gate_keys, vals[len(names) + 1:]))
        valid = {k: v for k, v in losses.items() if gate_vals.get(k, 1.0) > 0.0}
        overruled = tuple(k for k in losses if k not in valid)
        if overruled:
            # RARE: a head had no usable label in this whole micro-batch, so it is NOT a loss (the key
            # `losses` omits) and its parameters must see no gradient at all (Adam skips a None grad; a
            # zero one would still move them). Redo the backward over the valid keys, off the retained
            # graph, with the host verdicts the old path made. Nothing was stepped above.
            opt.zero_grad(set_to_none=True)
            if not valid:
                return overruled
            th.stack(list(valid.values())).sum().backward()
            g = float(th.nn.utils.clip_grad_norm_(params, RIDEALONG_MAX_GRAD_NORM))
        if not math.isfinite(g):
            # FAIL-CLOSED FOR THE HEADS, NEVER FOR THE RUN. A non-finite ride-along loss or
            # gradient is never stepped (the heads' weights stay finite, so no parameter scan
            # can trip on them) and the heads STOP for the rest of this process, loudly. The
            # run is not killed: the heads observe, and PPO never saw their numbers.
            opt.zero_grad(set_to_none=True)
            self._ridealong_disabled = True
            print(f"🛑 [RIDE-ALONG] non-finite loss/gradient (grad norm {g}, losses "
                  f"{ {k: float(v.detach()) for k, v in valid.items()} }) — the ride-along "
                  "heads are DISABLED for the rest of this process; PPO is unaffected.",
                  flush=True)
            return overruled
        if epoch == 0:
            acc.add("grad_norm", g)
        opt.step()
        # Back to None, so no later reader of `policy.parameters()` in this minibatch — PPO's
        # clip, the noise probes — can see a ride-along gradient.
        opt.zero_grad(set_to_none=True)
        return overruled

    def _record_ridealong_metrics(self, acc: RideAlongAccumulator) -> None:
        if self._ridealong_heads() is None:
            return
        # 1 once the heads have disabled themselves on a non-finite step — a gap in every other
        # `ridealong/*` series then has its cause on the dashboard.
        self.logger.record("ridealong/disabled",   # type: ignore[attr-defined]
                           float(bool(getattr(self, "_ridealong_disabled", False))))
        dead = getattr(self, "_ridealong_variant_disabled", None) or set()
        for n in getattr(self._ridealong_heads(), "variant_names", lambda: ())():
            self.logger.record(f"ridealong/rndv_{n}_disabled",   # type: ignore[attr-defined]
                               float(n in dead))
        for k, v in acc.metrics().items():
            self.logger.record(f"ridealong/{k}", v)   # type: ignore[attr-defined]
