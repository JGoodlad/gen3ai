"""The RIDE-ALONG step (`gen3_ridealong_heads_v1`) — the heads' own update, OUTSIDE the PPO fold.

`agents.model.ridealong_heads` holds the heads and their losses; this mixin runs them inside
`train()` without touching what PPO learns:

  * `_ridealong_update` runs RIGHT AFTER the minibatch's `evaluate_actions` forward. It snapshots the
    forward's stashes through `RideAlongBatch.detached` (THE stop-grad seam), computes the heads'
    losses, backpropagates them into the heads ALONE, clips the heads' own gradient, steps the
    heads' OWN Adam, and sets their gradients back to None — all before PPO's loss is even
    assembled. So `loss`, PPO's `clip_grad_norm_` (whose total norm would otherwise include the
    heads' gradients and rescale the trunk's), the noise-scale probes and the distill projector
    never see a ride-along gradient, and nothing the heads do consumes a random number.
  * `RideAlongAccumulator` folds the per-minibatch readouts into the `ridealong/*` TB family. The
    RANK meters (AUROC, Spearman, error by decile — does uncertainty predict V's actual error?) need
    the whole rollout. The heads train and are read on epoch 0 only (`RIDEALONG_EPOCHS`): one
    policy, every row once, each scored before the step that trains on it.

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

from typing import Any, Dict, List, Optional

import numpy as np
import torch as th

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


class RideAlongAccumulator:
    """Per-train() sink for the `ridealong/*` family."""

    def __init__(self) -> None:
        self.scalars: Dict[str, List[float]] = {}
        self.cols: Dict[str, List[np.ndarray]] = {}
        #: Set by the learner's first ride-along step of this `train()` call: the PPO-update boundary
        #: the `decay` variant's pull keys on (one accumulator per update).
        self.update_begun = False
        #: [sum of real-row error, sum of chimera-row error] per RND key (`rnd` / `rndv_<name>`).
        self.ident: Dict[str, List[float]] = {}

    def add(self, key: str, v: float) -> None:
        if v == v:                                    # a NaN is REPORTED by omission, never logged
            self.scalars.setdefault(key, []).append(float(v))

    def col(self, key: str, arr: th.Tensor) -> None:
        self.cols.setdefault(key, []).append(arr.detach().float().cpu().numpy())

    def observe(self, out: Dict[str, th.Tensor], b: Any, losses: Dict[str, th.Tensor],
                opp_class: Optional[th.Tensor], epoch: int) -> None:
        from agents.model.opp_intent import OPP_CLASS_NAMES

        for k, v in losses.items():
            self.add(f"{k}_loss", float(v.detach()))
        with th.no_grad():
            v_ = b.values.float().reshape(-1) if b.values is not None else None
            z = b.win_target.float().reshape(-1) if b.win_target is not None else None
            zm = (b.win_mask.float().reshape(-1) > 0) if b.win_mask is not None else None
            if "ens_std" in out:
                self.add("ens_disagreement_mean", float(out["ens_std"].mean()))
                if z is not None and zm is not None and bool(zm.any()):
                    p = th.sigmoid(out["ens_logits"])[zm]
                    brier = (p - z[zm][:, None]).pow(2).mean(0)
                    for j, bj in enumerate(brier.tolist()):
                        self.add(f"ens_brier_head_{j}", bj)
                    self.add("ens_brier_mean_member", float(
                        (out["ens_p"][zm] - z[zm]).pow(2).mean()))
                    if v_ is not None:
                        self.add("ens_brier_v", float((v_[zm] - z[zm]).pow(2).mean()))
            # base (`rnd`) and every RND variant (`rndv_<name>`): the same per-row statistics.
            for pre in rnd_prefixes(out):
                self.add(f"{pre}_err_mean", float(out[f"{pre}_err"].mean()))
                self.add(f"{pre}_z_mean", float(out[f"{pre}_z"].mean()))
                # every row's RAW error: the update's median / IQR (the saturation series)
                self.col(f"{pre}_err_raw", out[f"{pre}_err"])
                if opp_class is not None:
                    oc = opp_class.long().reshape(-1)
                    names = OPP_CLASS_NAMES.items() if isinstance(OPP_CLASS_NAMES, dict) else \
                        enumerate(OPP_CLASS_NAMES)
                    for code, name in names:
                        sel = oc == int(code)
                        if bool(sel.any()):
                            self.add(f"{pre}_z_{name}", float(out[f"{pre}_z"][sel].mean()))
            if "adv" in out and b.actions is not None and b.advantages is not None:
                a = b.actions.long().reshape(-1)
                y = b.advantages.float().reshape(-1)
                pred = out["adv_mean"].gather(-1, a[:, None]).squeeze(-1)
                var_y = float(y.var(unbiased=False))
                if var_y > 0:
                    self.add("adv_explained_var", 1.0 - float((y - pred).var(unbiased=False)) / var_y)
                if b.legal is not None and b.logits is not None and b.pi is not None:
                    legal = b.legal.bool()
                    n_legal = legal.float().sum(-1, keepdim=True).clamp(min=1.0)
                    lg = th.where(legal, b.logits.float(), th.zeros_like(b.logits.float()))
                    am = th.where(legal, out["adv_mean"], th.zeros_like(out["adv_mean"]))
                    lc = th.where(legal, lg - lg.sum(-1, keepdim=True) / n_legal, th.zeros_like(lg))
                    ac = th.where(legal, am - am.sum(-1, keepdim=True) / n_legal, th.zeros_like(am))
                    den = float(lc.pow(2).sum().sqrt() * ac.pow(2).sum().sqrt())
                    if den > 0:
                        self.add("adv_corr_logit", float((lc * ac).sum()) / den)
                    std = out["adv_std"]
                    starved = legal & (b.pi < STARVED_PI)
                    fed = legal & (b.pi >= STARVED_PI)
                    if bool(starved.any()):
                        self.add("adv_std_starved", float(std[starved].mean()))
                    if bool(fed.any()):
                        self.add("adv_std_fed", float(std[fed].mean()))
                    if v_ is not None:
                        q = v_[:, None] + out["adv_mean"]
                        self.add("q_out_of_range", float(((q < 0) | (q > 1))[legal].float().mean()))
            if ("opp" in out and b.opp_kind is not None and b.opp_num is not None
                    and b.alpha_seat_nums is not None):
                from agents.model.opp_intent import INTENT_IGNORE, match_seats_to_move_num
                idx = match_seats_to_move_num(b.alpha_seat_nums, b.opp_num.long().reshape(-1),
                                              b.opp_kind.long().reshape(-1),
                                              int(b.alpha_seat_nums.shape[-1]))
                # the share of rows B could train on: the opponent's action named in α's support
                self.add("opp_label_rate", float((idx != INTENT_IGNORE).float().mean()))
            if v_ is not None and z is not None and zm is not None:
                err = (v_ - z).abs()
                for key in ("ens_logit_std", *(f"{p}_z" for p in rnd_prefixes(out))):
                    if key in out:
                        self.col(key, out[key][zm])
                # V's own binary entropy, so the reader of `ens_*_err` can tell whether the
                # disagreement adds anything beyond V's own uncertainty (the level confound).
                pv = v_[zm].clamp(1e-6, 1 - 1e-6)
                self.col("v_entropy", -(pv * pv.log() + (1 - pv) * (1 - pv).log()))
                self.col("v_err", err[zm])

    def metrics(self) -> Dict[str, float]:
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

    def observe_identification(self, out: Dict[str, th.Tensor],
                               chim: Dict[str, th.Tensor]) -> None:
        """Fold one minibatch's identification probe: the real rows' error (``out``) and the same
        predictors' error on the rows' block chimeras (``chim``), summed per RND key."""
        for pre, ec in chim.items():
            er = out.get(f"{pre}_err")
            if er is None:
                continue
            acc = self.ident.setdefault(pre, [0.0, 0.0])
            acc[0] += float(er.detach().float().sum())
            acc[1] += float(ec.detach().float().sum())


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
        policy has no heads."""
        heads = self._ridealong_heads()
        if heads is None or epoch >= RIDEALONG_EPOCHS or getattr(self, "_ridealong_disabled", False):
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
        b = RideAlongBatch.detached(
            obs=obs["observation"], pooled=fe.last_value_pooled,
            pointer=tuple(fe.last_pointer_inputs) if fe.last_pointer_inputs is not None else None,
            pi=pi, logits=logits, legal=legal, values=values,
            actions=actions, advantages=rollout_data.advantages,
            win_target=obs.get("win_target"), win_mask=obs.get("win_mask"),
            alpha_logits=fe.last_alpha_logits if opp_on else None,
            alpha_seat_nums=fe.last_alpha_seat_nums if opp_on else None,
            opp_kind=obs.get("opp_action_kind") if opp_on else None,
            opp_num=obs.get("opp_action_num") if opp_on else None)
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
        losses = heads.losses(b, out, train_rnd=(epoch == 0))
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
        # meters cost host syncs once per row, not n_epochs times.
        if epoch == 0:
            acc.observe(out, b, {**losses, **vlosses}, obs.get("opp_class"), epoch)
            if variants is not None:
                # The identification probe (no grad, no RNG): the same predictors on the batch's
                # block chimeras, scored BEFORE this step trains on the real rows.
                chim = heads.identification_errors(b.obs)
                acc.observe_identification(out, {k: v for k, v in chim.items()
                                                 if k[len("rndv_"):] not in dead})
        if vlosses:
            self._ridealong_variant_step(heads, vlosses, epoch, acc)
        if losses:
            opt = self._ridealong_optimizer(heads)
            params = heads.trainable_parameters()
            opt.zero_grad(set_to_none=True)
            th.stack(list(losses.values())).sum().backward()
            gn = th.nn.utils.clip_grad_norm_(params, RIDEALONG_MAX_GRAD_NORM)
            if not bool(th.isfinite(gn)):
                # FAIL-CLOSED FOR THE HEADS, NEVER FOR THE RUN. A non-finite ride-along loss or
                # gradient is never stepped (the heads' weights stay finite, so no parameter scan
                # can trip on them) and the heads STOP for the rest of this process, loudly. The
                # run is not killed: the heads observe, and PPO never saw their numbers.
                opt.zero_grad(set_to_none=True)
                self._ridealong_disabled = True
                print(f"🛑 [RIDE-ALONG] non-finite loss/gradient (grad norm {float(gn)}, losses "
                      f"{ {k: float(v.detach()) for k, v in losses.items()} }) — the ride-along "
                      "heads are DISABLED for the rest of this process; PPO is unaffected.",
                      flush=True)
                return
            if epoch == 0:
                acc.add("grad_norm", float(gn))
            opt.step()
            # Back to None, so no later reader of `policy.parameters()` in this minibatch — PPO's
            # clip, the noise probes, the distill projector — can see a ride-along gradient.
            opt.zero_grad(set_to_none=True)

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

    def _ridealong_variant_step(self, heads: Any, vlosses: Dict[str, th.Tensor], epoch: int,
                                acc: RideAlongAccumulator) -> None:
        """Every RND variant's own step. One backward over the SUM of the variants' losses (their
        parameter sets are disjoint and every input is detached, so each predictor receives exactly
        its own loss's gradient), then per variant: its own clip, its own finiteness check, its own
        Adam step, its gradients back to None."""
        names = [k[len("rndv_"):] for k in vlosses]
        opts = {n: self._ridealong_variant_optimizer(heads, n) for n in names}
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        th.stack(list(vlosses.values())).sum().backward()
        dead = getattr(self, "_ridealong_variant_disabled", None)
        if dead is None:
            dead = set()
            self._ridealong_variant_disabled = dead
        for n in names:
            gn = th.nn.utils.clip_grad_norm_(heads.variant_parameters(n), RIDEALONG_MAX_GRAD_NORM)
            if not bool(th.isfinite(gn)):
                # FAIL-CLOSED FOR THIS VARIANT ONLY: never stepped, stopped for the rest of the
                # process, loudly; the four heads, the other variants and PPO carry on.
                opts[n].zero_grad(set_to_none=True)
                dead.add(n)
                print(f"🛑 [RIDE-ALONG] RND variant {n!r}: non-finite loss/gradient (grad norm "
                      f"{float(gn)}, loss {float(vlosses['rndv_' + n].detach())}) — DISABLED for "
                      "the rest of this process; the other heads and PPO are unaffected.", flush=True)
                continue
            if epoch == 0:
                acc.add(f"rndv_{n}_grad_norm", float(gn))
            opts[n].step()
            opts[n].zero_grad(set_to_none=True)

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
