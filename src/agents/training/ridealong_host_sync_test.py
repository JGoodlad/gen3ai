"""The ride-along heads' micro-batch path makes ONE host read (T25 next lever 1), and changes NOTHING it computes.

Window C (`designs/research_state/measurements/perf_phase_2026-10-10/`) found the update's host-blocked
time in one site, `RndNovelty.update_obs_stats`'s ``obs_count.item()`` — the FIRST of ~100 device reads
the ride-along tail made per epoch-0 micro-batch (the rest: `normalise`, `update_err_stats`, the two label
guards in `losses`, every ``float(t)`` / ``bool(t.any())`` / boolean-mask index in
`RideAlongAccumulator.observe`, the finiteness verdicts). Each is a host sync on CUDA; the first drains the
deep queue and the rest serialise the eager tail behind the GPU. They are now:

  * RND: host MIRRORS of ``obs_count`` / ``err_seen`` (`ridealong_heads._HostMirror`);
  * the label guards: device gates read inside the step's one transfer (`losses_gated`);
  * the meters: DEFERRED snapshots folded once past the ride-along epoch (`RideAlongAccumulator.flush`);
  * the verdicts: every clip norm and gate in ONE transfer (`RideAlongTerms._ridealong_step`).

THE ORACLE is HEAD's code, kept in this file (`_legacy_*`): the three RND methods, `_ridealong_update`'s
tail, the variant step and the accumulator's two bodies, VERBATIM except that the meters are folded at once
(`_legacy_observe`: the per-micro-batch reads). The tests run the SAME rollout through the oracle and the new path and compare, with `==`
(no tolerance — nothing here is allowed an ulp): every head and buffer, both Adam states, every logged
`ridealong/*` scalar, the PPO policy / optimizer / scalars and the RNG after a real `train()`. They also
count the syncs with `HostSyncTrace` and FAIL if a per-micro-batch read comes back: the oracle shows the
instrument sees ~100 per micro-batch where the new path makes one.
"""
from __future__ import annotations

import contextlib
import copy
import types

import numpy as np
import pytest
import torch as th

from agents.model.ridealong_heads import RND_ERR_EMA, RND_OBS_CLIP, RndNovelty, RndObsVariant
from agents.training import learner_golden as L
from agents.training.host_sync_trace import HostSyncTrace, optimizer_host_tensors
from agents.training.instrumented_ppo.ridealong_terms import (RIDEALONG_EPOCHS,
                                                              RIDEALONG_MAX_GRAD_NORM,
                                                              RideAlongAccumulator, RideAlongTerms)

#: Fold sites: a sync attributed to one of these functions is the DEFERRED meters' fold, not the micro-batch path.
_FOLD_BODY_FNS = ("flush", "_observe_now", "_observe_identification_now", "col")
#: ... and the fold's ONE transfer (`HostReadQueue.drain`, which the learner's own optimizer-step reads also use).
_FOLD_FNS = (*_FOLD_BODY_FNS, "drain")


# ======================================================================================== the oracle
def _legacy_update_obs_stats(self, obs):
    """HEAD's `RndNovelty.update_obs_stats` (``obs_count.item()`` — a device read per call)."""
    x = obs.detach().float()
    n_b = float(x.shape[0])
    if n_b == 0:
        return
    m_b = x.mean(0)
    v_b = x.var(0, unbiased=False)
    n_a = float(self.obs_count.item())
    n = n_a + n_b
    delta = m_b - self.obs_mean
    self.obs_mean.add_(delta * (n_b / n))
    self.obs_var.copy_((self.obs_var * n_a + v_b * n_b + delta * delta * (n_a * n_b / n)) / n)
    self.obs_count.fill_(n)


def _legacy_normalise(self, obs):
    """HEAD's `RndNovelty.normalise`."""
    if float(self.obs_count.item()) == 0.0:
        x = obs.float()
    else:
        x = (obs.float() - self.obs_mean) / th.sqrt(self.obs_var + 1e-8)
    return x.clamp(-RND_OBS_CLIP, RND_OBS_CLIP)


def _legacy_update_err_stats(self, err):
    """HEAD's `_RndErrStats.update_err_stats` (``err_seen.item()`` — a device read per call)."""
    with th.no_grad():
        e = err.detach().float()
        if e.numel() == 0:
            return
        m, v = e.mean(), e.var(unbiased=False)
        if not bool(self.err_seen.item()):
            self.err_mean.copy_(m)
            self.err_var.copy_(th.clamp(v, min=1e-12))
            self.err_seen.fill_(True)
            return
        self.err_mean.mul_(1 - RND_ERR_EMA).add_(RND_ERR_EMA * m)
        self.err_var.mul_(1 - RND_ERR_EMA).add_(RND_ERR_EMA * v)


@contextlib.contextmanager
def head_rnd_methods(monkeypatch):
    """HEAD's RND statistics methods, installed on the classes (a class attribute: every instance, and
    `RndObsVariant`'s inherited `update_err_stats`, reads them) for the duration of the block."""
    from agents.model.ridealong_heads import _RndErrStats

    with monkeypatch.context() as m:
        m.setattr(RndNovelty, "update_obs_stats", _legacy_update_obs_stats)
        m.setattr(RndNovelty, "normalise", _legacy_normalise)
        m.setattr(_RndErrStats, "update_err_stats", _legacy_update_err_stats)
        yield


def _legacy_observe(acc, out, b, losses, opp_class):
    """HEAD's `RideAlongAccumulator.observe` body, VERBATIM (`self` -> `acc`): ~100 device reads per call."""
    from agents.model.opp_intent import OPP_CLASS_NAMES
    from agents.training.instrumented_ppo.ridealong_terms import STARVED_PI, rnd_prefixes

    for k, v in losses.items():
        acc.add(f"{k}_loss", float(v.detach()))
    with th.no_grad():
        v_ = b.values.float().reshape(-1) if b.values is not None else None
        z = b.win_target.float().reshape(-1) if b.win_target is not None else None
        zm = (b.win_mask.float().reshape(-1) > 0) if b.win_mask is not None else None
        if "ens_std" in out:
            acc.add("ens_disagreement_mean", float(out["ens_std"].mean()))
            if z is not None and zm is not None and bool(zm.any()):
                p = th.sigmoid(out["ens_logits"])[zm]
                brier = (p - z[zm][:, None]).pow(2).mean(0)
                for j, bj in enumerate(brier.tolist()):
                    acc.add(f"ens_brier_head_{j}", bj)
                acc.add("ens_brier_mean_member", float(
                    (out["ens_p"][zm] - z[zm]).pow(2).mean()))
                if v_ is not None:
                    acc.add("ens_brier_v", float((v_[zm] - z[zm]).pow(2).mean()))
        for pre in rnd_prefixes(out):
            acc.add(f"{pre}_err_mean", float(out[f"{pre}_err"].mean()))
            acc.add(f"{pre}_z_mean", float(out[f"{pre}_z"].mean()))
            acc.col(f"{pre}_err_raw", out[f"{pre}_err"])
            if opp_class is not None:
                oc = opp_class.long().reshape(-1)
                names = OPP_CLASS_NAMES.items() if isinstance(OPP_CLASS_NAMES, dict) else \
                    enumerate(OPP_CLASS_NAMES)
                for code, name in names:
                    sel = oc == int(code)
                    if bool(sel.any()):
                        acc.add(f"{pre}_z_{name}", float(out[f"{pre}_z"][sel].mean()))
        if "adv" in out and b.actions is not None and b.advantages is not None:
            a = b.actions.long().reshape(-1)
            y = b.advantages.float().reshape(-1)
            pred = out["adv_mean"].gather(-1, a[:, None]).squeeze(-1)
            var_y = float(y.var(unbiased=False))
            if var_y > 0:
                acc.add("adv_explained_var", 1.0 - float((y - pred).var(unbiased=False)) / var_y)
            if b.legal is not None and b.logits is not None and b.pi is not None:
                legal = b.legal.bool()
                n_legal = legal.float().sum(-1, keepdim=True).clamp(min=1.0)
                lg = th.where(legal, b.logits.float(), th.zeros_like(b.logits.float()))
                am = th.where(legal, out["adv_mean"], th.zeros_like(out["adv_mean"]))
                lc = th.where(legal, lg - lg.sum(-1, keepdim=True) / n_legal, th.zeros_like(lg))
                ac = th.where(legal, am - am.sum(-1, keepdim=True) / n_legal, th.zeros_like(am))
                den = float(lc.pow(2).sum().sqrt() * ac.pow(2).sum().sqrt())
                if den > 0:
                    acc.add("adv_corr_logit", float((lc * ac).sum()) / den)
                std = out["adv_std"]
                starved = legal & (b.pi < STARVED_PI)
                fed = legal & (b.pi >= STARVED_PI)
                if bool(starved.any()):
                    acc.add("adv_std_starved", float(std[starved].mean()))
                if bool(fed.any()):
                    acc.add("adv_std_fed", float(std[fed].mean()))
                if v_ is not None:
                    q = v_[:, None] + out["adv_mean"]
                    acc.add("q_out_of_range", float(((q < 0) | (q > 1))[legal].float().mean()))
        if "opp" in out and b.flat_target is not None:
            from agents.model.opp_intent import INTENT_IGNORE
            acc.add("opp_label_rate", float((b.flat_target != INTENT_IGNORE).float().mean()))
        if v_ is not None and z is not None and zm is not None:
            err = (v_ - z).abs()
            for key in ("ens_logit_std", *(f"{p}_z" for p in rnd_prefixes(out))):
                if key in out:
                    acc.col(key, out[key][zm])
            pv = v_[zm].clamp(1e-6, 1 - 1e-6)
            acc.col("v_entropy", -(pv * pv.log() + (1 - pv) * (1 - pv).log()))
            acc.col("v_err", err[zm])


def _legacy_observe_identification(acc, out, chim):
    """HEAD's `RideAlongAccumulator.observe_identification`, VERBATIM."""
    for pre, ec in chim.items():
        er = out.get(f"{pre}_err")
        if er is None:
            continue
        a = acc.ident.setdefault(pre, [0.0, 0.0])
        a[0] += float(er.detach().float().sum())
        a[1] += float(ec.detach().float().sum())


def _legacy_ridealong_update(self, rollout_data, values, actions, epoch, acc):
    """HEAD's `_ridealong_update`: per-micro-batch host reads throughout (guards, meters, verdicts)."""
    heads = self._ridealong_heads()
    if heads is None or epoch >= RIDEALONG_EPOCHS or getattr(self, "_ridealong_disabled", False):
        return
    from agents.model.ridealong_heads import RideAlongBatch

    fe = self.policy.features_extractor
    obs = rollout_data.observations
    dist = getattr(self.policy, "_last_pi_distribution", None)
    pi = logits = legal = None
    if dist is not None:
        pi = dist.distribution.probs
        legal = pi > 0
        logits = dist.distribution.logits
    opp_on = float(getattr(self, "opp_intent_coef", 0.0) or 0.0) > 0.0
    _fi = getattr(fe, "last_flat_intent", None) if opp_on else None
    _flat_kw = {}
    if _fi is not None and "opp_action_kind" in obs:
        from agents.model.flat_intent import flat_intent_targets
        _ft, _ = flat_intent_targets(_fi, obs["opp_action_kind"], obs["opp_action_num"],
                                     obs["opp_switch_slot"], obs["opp_switch_species"])
        _flat_kw = dict(flat_logits=fe.last_flat_intent_logits, flat_ids=_fi.cand_ids, flat_target=_ft)
    b = RideAlongBatch.detached(
        obs=obs["observation"], pooled=fe.last_value_pooled,
        pointer=tuple(fe.last_pointer_inputs) if fe.last_pointer_inputs is not None else None,
        pi=pi, logits=logits, legal=legal, values=values,
        actions=actions, advantages=rollout_data.advantages,
        win_target=obs.get("win_target"), win_mask=obs.get("win_mask"), **_flat_kw)
    if b.pooled is None:
        return
    rnd = getattr(heads, "rnd", None)
    variants = getattr(heads, "rnd_variants", None)
    dead = getattr(self, "_ridealong_variant_disabled", None) or set()
    if not acc.update_begun:
        acc.update_begun = True
        if hasattr(heads, "begin_update_"):
            heads.begin_update_(skip=tuple(dead))
    if rnd is not None and epoch == 0:
        rnd.update_obs_stats(b.obs)
        if variants is not None and "feat" in variants and "feat" not in dead:
            variants["feat"].update_obs_stats(b.pooled)
    out = heads.readout(b, skip_variants=tuple(dead)) if variants is not None else heads.readout(b)
    losses = heads.losses(b, out, train_rnd=(epoch == 0))
    vlosses = heads.variant_losses(out) if variants is not None else {}
    if rnd is not None and epoch == 0 and "rnd_err" in out:
        rnd.update_err_stats(out["rnd_err"])
        out["rnd_z"] = rnd.zscore(out["rnd_err"])
        for n, vmod in (variants.items() if variants is not None else ()):
            if f"rndv_{n}_err" in out:
                vmod.update_err_stats(out[f"rndv_{n}_err"])
                out[f"rndv_{n}_z"] = vmod.zscore(out[f"rndv_{n}_err"])
    if epoch == 0:
        _legacy_observe(acc, out, b, {**losses, **vlosses}, obs.get("opp_class"))
        if variants is not None:
            chim = heads.identification_errors(b.obs)
            _legacy_observe_identification(acc, out, {k: v for k, v in chim.items()
                                                      if k[len("rndv_"):] not in dead})
    if vlosses:
        _legacy_variant_step(self, heads, vlosses, epoch, acc)
    if losses:
        opt = self._ridealong_optimizer(heads)
        params = heads.trainable_parameters()
        opt.zero_grad(set_to_none=True)
        th.stack(list(losses.values())).sum().backward()
        gn = th.nn.utils.clip_grad_norm_(params, RIDEALONG_MAX_GRAD_NORM)
        if not bool(th.isfinite(gn)):
            opt.zero_grad(set_to_none=True)
            self._ridealong_disabled = True
            print(f"🛑 [RIDE-ALONG] non-finite loss/gradient (grad norm {float(gn)}, losses "
                  f"{ {k: float(v.detach()) for k, v in losses.items()} }) — the ride-along "
                  "heads are DISABLED for the rest of this process; PPO is unaffected.", flush=True)
            return
        if epoch == 0:
            acc.add("grad_norm", float(gn))
        opt.step()
        opt.zero_grad(set_to_none=True)


def _legacy_variant_step(self, heads, vlosses, epoch, acc):
    """HEAD's `_ridealong_variant_step`."""
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


class _NewLearner(RideAlongTerms):
    def __init__(self, policy):
        self.policy = policy
        self.opp_intent_coef = 0.05
        self.logged = {}
        self.logger = types.SimpleNamespace(record=lambda k, v: self.logged.__setitem__(k, v))


class _OldLearner(_NewLearner):
    _ridealong_update = _legacy_ridealong_update


# ================================================================================ small comparisons
_INT_OF = {th.float32: th.int32, th.float64: th.int64, th.float16: th.int16, th.bfloat16: th.int16}


def _same(a, b, what=""):
    """BITWISE equality of two tensors (a NaN equals the same NaN; -0.0 does not equal 0.0)."""
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    if a.dtype in _INT_OF:
        a, b = a.contiguous().view(_INT_OF[a.dtype]), b.contiguous().view(_INT_OF[b.dtype])
    assert th.equal(a, b), what


def _same_state(a, b, what=""):
    assert a.keys() == b.keys(), what
    for k in a:
        _same(a[k], b[k], f"{what}{k}")


def _same_opt(oa, ob, what=""):
    sa, sb = oa.state_dict(), ob.state_dict()
    assert sa["state"].keys() == sb["state"].keys(), what
    for i in sa["state"]:
        for k, v in sa["state"][i].items():
            if th.is_tensor(v):
                _same(v, sb["state"][i][k], f"{what}opt[{i}].{k}")
            else:
                assert v == sb["state"][i][k], (what, i, k)


def _same_metrics(ma, mb, what=""):
    assert ma.keys() == mb.keys(), (what, sorted(set(ma) ^ set(mb)))
    for k, v in ma.items():
        assert v == mb[k] or (v != v and mb[k] != mb[k]), (what, k, v, mb[k])    # exact: no tolerance


# ===================================================== A. the host mirrors == the device branches
def _twin_rnd(dim=6):
    with th.random.fork_rng(devices=[]):
        a = RndNovelty(dim)
    return a, copy.deepcopy(a)


def _assert_rnd_equal(a, b, what):
    for name in ("obs_mean", "obs_var", "obs_count", "err_mean", "err_var", "err_seen"):
        _same(getattr(a, name), getattr(b, name), f"{what}: {name}")


def test_RND_mirrors_equal_the_device_branches_bit_for_bit_across_outside_writes():
    """The mirrors answer ``count == 0`` and ``err_seen`` from the host; HEAD answered with a device read.
    Same buffers, same normalised rows, after every kind of write the mirror must notice: the module's
    own update, an outside ``zero_`` (`main.ridealong_read.rnd_choice` does it), a state-dict load, a
    deepcopy, and an empty batch."""
    a, b = _twin_rnd()
    g = th.Generator().manual_seed(7)
    batches = [th.randn(n, 6, generator=g) * 3 + 1 for n in (5, 17, 40, 9)]
    errs = [(bt ** 2).sum(-1) for bt in batches]

    def step(i, tag):
        a.update_obs_stats(batches[i])
        _legacy_update_obs_stats(b, batches[i])
        _same(a.normalise(batches[i]), _legacy_normalise(b, batches[i]), f"{tag}: normalise")
        a.update_err_stats(errs[i])
        _legacy_update_err_stats(b, errs[i])
        _assert_rnd_equal(a, b, tag)

    _same(a.normalise(batches[0]), _legacy_normalise(b, batches[0]), "count 0: the raw clipped rows")
    step(0, "first batch (count 0, err unseen)")
    step(1, "second batch")
    a.update_obs_stats(th.zeros(0, 6))
    _legacy_update_obs_stats(b, th.zeros(0, 6))
    a.update_err_stats(th.zeros(0))
    _legacy_update_err_stats(b, th.zeros(0))
    _assert_rnd_equal(a, b, "empty batches change nothing")
    a.obs_count.zero_()                     # an OUTSIDE write: the mirror must see it and re-read
    b.obs_count.zero_()
    a.err_seen.fill_(False)
    b.err_seen.fill_(False)
    _same(a.normalise(batches[2]), _legacy_normalise(b, batches[2]), "after an outside zero_")
    step(2, "after an outside zero_ / fill_")
    c = copy.deepcopy(a)                    # a copy must read ITS OWN buffer, never the original's mirror
    d = copy.deepcopy(b)
    c.obs_count.zero_()
    d.obs_count.zero_()
    c.update_obs_stats(batches[3])
    _legacy_update_obs_stats(d, batches[3])
    _assert_rnd_equal(c, d, "a deepcopy, then zero_ on the copy")
    assert float(a.obs_count) == float(b.obs_count) != 0.0, "the original is untouched"
    other, _ = _twin_rnd()
    other.update_obs_stats(batches[0] * 2)
    other.update_err_stats(errs[0] * 5)
    sd = other.state_dict()
    a.load_state_dict(sd)
    b.load_state_dict(sd)
    step(3, "after load_state_dict")
    a.to(th.float64)                        # a dtype move REPLACES the buffers: new tensors, same values
    b.to(th.float64)
    a.update_obs_stats(batches[0].double())
    _legacy_update_obs_stats(b, batches[0].double())
    for name in ("obs_mean", "obs_var", "obs_count"):
        _same(getattr(a, name), getattr(b, name), f"after .to(float64): {name}")


def test_RND_stat_updates_read_no_device_scalar_after_the_first_use():
    """The instrument's teeth AND the fix: HEAD's methods sync once per call each; the mirrors sync once
    in a lifetime (the first use) and never again — fails the moment an ``.item()`` comes back."""
    rnd, _ = _twin_rnd()
    var = RndObsVariant(rnd_variant_decl(), copy.deepcopy(rnd.predictor))
    x = th.randn(32, 6)
    err = x.pow(2).sum(-1)
    for m in (rnd, var):                    # priming: the one read of each mirror
        m.update_err_stats(err)
    rnd.update_obs_stats(x)
    with HostSyncTrace() as tr:
        for _ in range(3):
            rnd.update_obs_stats(x)
            rnd.normalise(x)
            rnd.update_err_stats(err)
            var.update_err_stats(err)
    assert tr.total == 0, tr.table()
    twin, _ = _twin_rnd()
    with HostSyncTrace() as tl:
        _legacy_update_obs_stats(twin, x)
        _legacy_normalise(twin, x)
        _legacy_update_err_stats(twin, err)
    assert [e.op for e in tl.events] == ["item", "item", "item"], tl.table()    # the oracle is sync-ful


def rnd_variant_decl():
    from agents.model.ridealong_heads import RND_VARIANT_BY_NAME

    return RND_VARIANT_BY_NAME["fast"]


# =========================== B. the whole micro-batch path == HEAD's, on a real policy
def _micro_batches(pol, layout, n_micro, rows_per=8, *, poison=None, no_labels=()):
    """``n_micro`` forwards of the committed REAL-obs fixture rows (a rolling slice), each with the
    learner's inputs. ``no_labels``: micro-batch indices whose win labels are ALL masked out (the rare
    path: the V ensemble has no usable label); ``poison``: an index whose observation holds a NaN."""
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.training.instrumented_ppo_ridealong_terms_test import _forward

    rows, masks = load_parity_rows(layout["total_dim"])
    out = []
    for i in range(n_micro):
        sl = [(i * 3 + j) % len(rows) for j in range(rows_per)]
        rd, values, actions = _forward(pol, rows[sl].copy(), masks[sl])
        rd.observations["opp_class"] = (th.arange(rows_per) % 2).float()[:, None]    # two classes present
        if i in no_labels:
            rd.observations["win_mask"] = th.zeros_like(rd.observations["win_mask"])
        if i == poison:
            bad = rd.observations["observation"].clone()
            bad[0, 0] = float("nan")
            rd.observations["observation"] = bad
        out.append((rd, values, actions))
    return out


def _run_arm(old, monkeypatch, ridealong, *, n_micro=4, **kw):
    """Build a policy (seeded) and the OLD (HEAD) or NEW learner on it, drive ``n_micro`` epoch-0
    micro-batches and the epoch-1 call that folds the meters; return everything the arm leaves behind."""
    from agents.model.ridealong_heads_test import _policy

    with head_rnd_methods(monkeypatch) if old else contextlib.nullcontext():
        pol, layout = _policy(ridealong)
        lr = (_OldLearner if old else _NewLearner)(pol)
        lr._ridealong_acquire()
        acc = RideAlongAccumulator()
        for rd, values, actions in _micro_batches(pol, layout, n_micro, **kw):
            lr._ridealong_update(rd, values, actions, epoch=0, acc=acc)
        lr._ridealong_update(*_micro_batches(pol, layout, 1)[0], epoch=1, acc=acc)
        lr._record_ridealong_metrics(acc)
        return dict(pol=pol, lr=lr, metrics=acc.metrics(), logged=dict(lr.logged),
                    state={k: v.detach().clone() for k, v in pol.ridealong.state_dict().items()},
                    rng=th.get_rng_state().clone())


def _compare_arms(old, new):
    _same_state(old["state"], new["state"], "ride-along state: ")
    _same_opt(old["lr"]._ridealong_opt, new["lr"]._ridealong_opt, "heads' Adam: ")
    assert old["lr"]._ridealong_vopts.keys() == new["lr"]._ridealong_vopts.keys()
    for n in old["lr"]._ridealong_vopts:
        _same_opt(old["lr"]._ridealong_vopts[n], new["lr"]._ridealong_vopts[n], f"variant {n}'s Adam: ")
    _same_metrics(old["metrics"], new["metrics"], "ridealong metrics: ")
    _same_metrics(old["logged"], new["logged"], "logged: ")
    assert getattr(old["lr"], "_ridealong_disabled", False) == getattr(new["lr"], "_ridealong_disabled", False)
    assert (getattr(old["lr"], "_ridealong_variant_disabled", set())
            == getattr(new["lr"], "_ridealong_variant_disabled", set()))
    _same(old["rng"], new["rng"], "the torch RNG")


ON = {"ridealong_ensemble": 3, "ridealong_rnd": True, "ridealong_adv": 2, "ridealong_opp": 2,
      "ridealong_rnd_variants": "all"}


@pytest.mark.parametrize("case", ["clean", "no_win_labels_in_one_micro_batch", "nan_observation"])
def test_the_new_micro_batch_path_equals_HEADs_bit_for_bit(case, monkeypatch):
    """Heads, buffers (the RND running statistics), both Adam states, every `ridealong/*` scalar, the
    disabled flags and the RNG: IDENTICAL to HEAD's path, bit for bit, on the all-heads + all-variants
    surface. `no_win_labels` drives the RARE path (the V ensemble has no usable label in one whole
    micro-batch: its key is omitted and its parameters see no gradient at all); `nan_observation` the
    fail-closed path (the four heads and three variants disabled, nothing stepped)."""
    kw = {"no_win_labels_in_one_micro_batch": dict(no_labels=(1,)),
          "nan_observation": dict(poison=2)}.get(case, {})
    old = _run_arm(True, monkeypatch, ON, **kw)
    new = _run_arm(False, monkeypatch, ON, **kw)
    _compare_arms(old, new)
    assert new["metrics"]                       # not vacuous: the meters folded
    if case == "nan_observation":
        assert new["lr"]._ridealong_disabled
        assert new["lr"]._ridealong_variant_disabled == {"fast", "decay", "small"}     # feat reads a finite input
    else:
        assert not getattr(new["lr"], "_ridealong_disabled", False)
        assert not getattr(new["lr"], "_ridealong_variant_disabled", None)
        assert new["metrics"]["ens_loss"] > 0.0 and new["metrics"]["grad_norm"] > 0.0


def test_the_RARE_path_really_skips_the_ensemble_for_that_micro_batch(monkeypatch):
    """The case above is not vacuous: with the win labels masked in micro-batch 1, the `ens_loss` series
    holds 3 values, not 4, and the ensemble's parameters took no step in that micro-batch — in HEAD and
    in the new path alike (Adam skips a None grad; a zero one would still move them)."""
    from agents.model.ridealong_heads_test import _policy

    got = {}
    for name, old in (("old", True), ("new", False)):
        with head_rnd_methods(monkeypatch) if old else contextlib.nullcontext():
            pol, layout = _policy(ON)
            lr = (_OldLearner if old else _NewLearner)(pol)
            lr._ridealong_acquire()
            acc = RideAlongAccumulator()
            moved = []
            for rd, values, actions in _micro_batches(pol, layout, 4, no_labels=(1,)):
                before = {k: v.clone() for k, v in pol.ridealong.state_dict().items() if k.startswith("ensemble.")}
                lr._ridealong_update(rd, values, actions, epoch=0, acc=acc)
                after = pol.ridealong.state_dict()
                moved.append(any(not th.equal(after[k], v) for k, v in before.items() if ".members." in k))
            acc.flush()
            got[name] = (len(acc.scalars["ens_loss"]), moved)
    assert got["old"] == got["new"] == (3, [True, False, True, True]), got


def test_a_head_with_NO_label_at_all_takes_no_step_and_logs_no_grad_norm(monkeypatch):
    """`losses` empty (the ensemble-only surface, every label masked): nothing is stepped and no
    ``grad_norm`` is logged — HEAD's ``if losses:`` guard — though the new path has already enqueued a
    backward on the ungated loss before it learns that."""
    one = {"ridealong_ensemble": 3}
    old = _run_arm(True, monkeypatch, one, n_micro=2, no_labels=(0, 1))
    new = _run_arm(False, monkeypatch, one, n_micro=2, no_labels=(0, 1))
    _compare_arms(old, new)
    assert "grad_norm" not in new["metrics"] and "ens_loss" not in new["metrics"], new["metrics"]
    assert all(p.grad is None for p in new["pol"].ridealong.parameters())


# ============================ C. the sync counts, with teeth
def _trace_micro_batches(old, monkeypatch, n_micro=3):
    """The host reads of each epoch-0 micro-batch after a priming one, and of the meters' fold."""
    from agents.model.ridealong_heads_test import _policy

    with head_rnd_methods(monkeypatch) if old else contextlib.nullcontext():
        pol, layout = _policy(ON)
        lr = (_OldLearner if old else _NewLearner)(pol)
        lr._ridealong_acquire()
        acc = RideAlongAccumulator()
        mbs = _micro_batches(pol, layout, n_micro + 1)
        lr._ridealong_update(*mbs[0], epoch=0, acc=acc)             # priming: the mirrors' one read each
        hosts = optimizer_host_tensors(lr._ridealong_opt)           # the CPU Adam's host step counters
        for o in lr._ridealong_vopts.values():
            hosts += optimizer_host_tensors(o)
        per_micro = []
        for rd, values, actions in mbs[1:]:
            with HostSyncTrace(host=hosts) as tr:
                lr._ridealong_update(rd, values, actions, epoch=0, acc=acc)
            per_micro.append(tr)
        with HostSyncTrace(host=hosts) as fold:
            acc.flush()
        return per_micro, fold, acc


def test_a_ride_along_micro_batch_makes_ONE_host_read_and_HEAD_made_over_a_hundred(monkeypatch):
    old, _, _ = _trace_micro_batches(True, monkeypatch)
    new, fold, acc = _trace_micro_batches(False, monkeypatch)
    for tr in new:
        assert tr.total == 1, f"{tr.total} host reads in one ride-along micro-batch:\n{tr.table()}"
        assert tr.events[0].op == "cpu" and tr.events[0].fn == "read_floats", tr.table()
    # the instrument's teeth: HEAD's path, the same inputs, the same tracer
    assert all(tr.total > 100 for tr in old), [tr.total for tr in old]
    # the deferred meters are read in the FOLD, off the micro-batch path: ONE transfer, plus one `nonzero` per
    # mask the statistics compact by (the rows, the opponent classes, the starved / fed / legal sets) — a
    # handful per micro-batch where HEAD's body made over a hundred
    assert 1 <= fold.total <= 10 * (len(new) + 1) + 1 and all(e.fn in _FOLD_FNS for e in fold.events), fold.table()
    assert sum(e.fn == "drain" for e in fold.events) == 1, fold.table()
    assert not acc._pending


def test_on_a_real_train_every_meter_read_comes_after_the_last_micro_batch_read():
    """The structural claim on the golden learner's ``train()`` with every head on: exactly ONE
    ride-along transfer per epoch-0 micro-batch, the mirrors' priming reads before them, and the whole
    fold of the meters AFTER the last of them — at epoch 1's first micro-batch, never inside epoch 0."""
    model = _golden_with_heads()
    hosts = optimizer_host_tensors(model.policy.optimizer) + optimizer_host_tensors(model._ridealong_opt)
    for o in model._ridealong_vopts.values():
        hosts += optimizer_host_tensors(o)
    with L._one_thread():
        np.random.seed(L.UPDATE_SEED)
        th.manual_seed(L.UPDATE_SEED)
        with HostSyncTrace(host=hosts) as tr:
            model.train()
    micros = -(-model.n_steps * model.n_envs // model.batch_size)         # micro-batches in epoch 0
    fold = [i for i, e in enumerate(tr.events) if e.fn in _FOLD_BODY_FNS]
    verdicts = [i for i, e in enumerate(tr.events) if e.fn == "read_floats"]
    primes = [i for i, e in enumerate(tr.events) if e.fn == "get"]
    drains = [e for e in tr.events if e.fn == "drain"]
    steps = model.n_epochs * -(-micros // model.grad_accum_steps)           # optimizer steps of the update
    assert fold and len(verdicts) == micros, (len(verdicts), micros, tr.table())
    assert len(fold) <= 10 * micros, tr.table()            # the whole fold: a handful per micro-batch ...
    assert len(drains) == steps + 1, tr.table()            # ... plus ONE transfer beside the learner's own
    assert len(primes) <= 7, tr.table()                  # base + feat counts, base + four variants' err_seen
    assert max(verdicts) < min(fold) and max(primes) < min(fold), tr.table()
    # no ride-along file reads the device outside the fold except the mirrors' priming (`get`)
    stray = [e for e in tr.events if "ridealong" in e.site and e.fn not in _FOLD_BODY_FNS and e.fn != "get"]
    assert not stray, tr.table()


def _golden_with_heads(plain=True):
    from agents.model.ridealong_heads import RND_VARIANTS, RideAlongSpec, build_ridealong

    model = L.build_learner()
    L.load_buffer_into(model)
    if plain:                               # not a diagnostics update (`diagnostics_cadence`)
        model.diagnostics_every = 10
        model._diagnostics_ran_in_process = True
        model.num_timesteps = model.n_steps * model.n_envs * 3
    fe = model.policy.features_extractor
    obs_dim = int(model.policy.observation_space["observation"].shape[0])
    model.policy.ridealong = build_ridealong(
        fe, obs_dim=obs_dim, spec=RideAlongSpec(ensemble=5, rnd=True, adv=5, opp=5,
                                                rnd_variants=RND_VARIANTS,
                                                opp_flat_k=RideAlongSpec.from_extractor(fe).opp_flat_k)
    ).to(model.device)
    model._ridealong_disabled = False
    model._ridealong_acquire()
    return model


# ====================== D. a real train(): the PPO side and the ride-along side, end to end
def test_a_real_update_is_BIT_IDENTICAL_to_HEADs_ride_along_and_PPO_side(monkeypatch):
    """``InstrumentedMaskablePPO.train()`` on the production surface (the committed real-obs rows, the
    flat-pointer labels, B ON, every variant): HEAD's ride-along path (and RND methods) installed on the
    classes vs the new one. After one update the policy, trunk and V parameters, PPO's optimizer state, the
    RNG, every ride-along tensor, both Adam states and EVERY logged scalar (the `ridealong/*` ones
    included) are the SAME BITS. This is the proof that moving the host reads moves nothing the update
    computes or logs."""
    from agents.training import ridealong_update_test as T
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    with head_rnd_methods(monkeypatch), monkeypatch.context() as m:
        m.setattr(InstrumentedMaskablePPO, "_ridealong_update", _legacy_ridealong_update)
        old = T._build(T.RIDEALONG_ON)
    new = T._build(T.RIDEALONG_ON)
    assert sum(t.startswith("ridealong/") for t in new["logged"]) > 90 and len(new["logged"]) > 200
    _same_state(old["post"], new["post"], "policy: ")
    assert old["opt"].keys() == new["opt"].keys() and old["opt"]
    for k in old["opt"]:
        _same(old["opt"][k], new["opt"][k], f"PPO optimizer {k}")
    _same_metrics({k: v for k, v in old["logged"].items() if not k.endswith("_ms")},
                  {k: v for k, v in new["logged"].items() if not k.endswith("_ms")}, "logged: ")
    _same_opt(old["model"]._ridealong_opt, new["model"]._ridealong_opt, "heads' Adam: ")
    for n in old["model"]._ridealong_vopts:
        _same_opt(old["model"]._ridealong_vopts[n], new["model"]._ridealong_vopts[n], f"variant {n}'s Adam: ")
    _same(old["rng"][0], new["rng"][0], "torch RNG")
    assert old["rng"][2] == new["rng"][2]
    assert np.array_equal(old["rng"][1][1], new["rng"][1][1])


# ============================================================= E. the meters are a faithful DEFERRAL
def test_the_deferred_meters_read_the_tensors_as_they_were_when_observed(monkeypatch):
    """A snapshot is a COPY of what the learner's buffers hold at that moment: refilling a buffer in place
    (the staged device-batch mode's prefetch thread does) AFTER ``observe`` must not change the meter.
    The same step folded at once on a twin policy is what the deferred fold must read."""
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.ridealong_heads_test import _policy
    from agents.training.instrumented_ppo_ridealong_terms_test import _forward

    pol, layout = _policy(ON)
    rows, masks = load_parity_rows(layout["total_dim"])
    lr = _NewLearner(pol)
    lr._ridealong_acquire()
    rd, values, actions = _forward(pol, rows[:8].copy(), masks[:8])
    rd.observations["opp_class"] = (th.arange(8) % 2).float()[:, None]
    acc, ref = RideAlongAccumulator(), RideAlongAccumulator()
    lr._ridealong_update(rd, values, actions, epoch=0, acc=acc)
    assert acc._pending and "ens_loss" not in acc.scalars, "observe must not fold at once"
    with head_rnd_methods(monkeypatch):
        twin, _ = _policy(ON)
        lt = _OldLearner(twin)
        lt._ridealong_acquire()
        rd2, v2, a2 = _forward(twin, rows[:8].copy(), masks[:8])
        rd2.observations["opp_class"] = (th.arange(8) % 2).float()[:, None]
        lt._ridealong_update(rd2, v2, a2, epoch=0, acc=ref)
    for t in (rd.advantages, rd.observations["win_target"], rd.observations["win_mask"],
              rd.observations["opp_class"], actions):          # clobber every buffer a snapshot could alias
        t.zero_()
    lr._ridealong_update(rd, values, actions, epoch=1, acc=acc)    # the fold: the first call past the epoch
    assert not acc._pending
    _same_metrics(ref.metrics(), acc.metrics(), "deferred vs immediate: ")


def _synthetic_readout(case, n=24, k=3, seed=0):
    """A hand-built readout / batch / losses / classes for the meters, with the EDGE the case names:
    ``masked`` (no win-labelled row), ``one_class`` (the other opponent classes are empty), ``const_adv``
    (zero advantage variance), ``all_fed`` (no starved action), ``zero_corr`` (a zero advantage mean: the
    correlation's denominator is 0), ``few_rows`` (the win mask keeps three rows), ``normal``."""
    from agents.model.opp_intent import INTENT_IGNORE

    g = th.Generator().manual_seed(seed)
    r = lambda *shape: th.randn(*shape, generator=g)                         # noqa: E731
    legal = th.rand(n, 11, generator=g) > 0.3
    legal[:, 0] = True
    logits = r(n, 11)
    pi = th.softmax(th.where(legal, logits * 3, th.full_like(logits, -1e9)), dim=-1)
    if case == "all_fed":
        pi = th.where(legal, th.full_like(pi, 0.2), th.zeros_like(pi))
    mask = th.ones(n)
    if case == "masked":
        mask = th.zeros(n)
    if case == "few_rows":
        mask = th.zeros(n)
        mask[[1, 5, 9]] = 1.0
    out = {"ens_std": r(n).abs(), "ens_logits": r(n, k), "ens_p": th.rand(n, generator=g),
           "ens_logit_std": r(n).abs(), "rnd_err": r(n).abs(), "rnd_z": r(n),
           "rndv_fast_err": r(n).abs(), "rndv_fast_z": r(n),
           "adv": r(n, k, 11), "adv_mean": r(n, 11) * (0.0 if case == "zero_corr" else 1.0),
           "adv_std": r(n, 11).abs(), "opp": r(n, k, 5)}
    adv = r(n) if case != "const_adv" else th.full((n,), 0.25)
    b = types.SimpleNamespace(
        values=th.rand(n, generator=g), win_target=(th.rand(n, generator=g) > 0.5).float(),
        win_mask=mask, actions=th.randint(0, 11, (n,), generator=g), advantages=adv,
        legal=legal, logits=logits, pi=pi,
        flat_target=th.where(th.rand(n, generator=g) > 0.4, th.randint(0, 5, (n,), generator=g),
                             th.full((n,), INTENT_IGNORE)))
    classes = th.zeros(n, 1) if case == "one_class" else th.randint(0, 4, (n, 1), generator=g).float()
    losses = {"ens": r(()).abs(), "rnd": r(()).abs(), "rndv_fast": r(()).abs()}
    return out, b, losses, classes


@pytest.mark.parametrize("case", ["normal", "masked", "one_class", "const_adv", "all_fed", "zero_corr",
                                  "few_rows"])
def test_the_ONE_TRANSFER_fold_equals_HEADs_per_read_body_on_every_edge(case):
    """The fold's rewrite (every ``float(t)`` queued, every masked set compacted by one shared ``nonzero``)
    against HEAD's body, on inputs that hit each guard HEAD had: an empty selection (its ``if mask.any()``
    skip is the new NaN-omission), zero variance, a zero denominator. Scalars, the per-row columns and the
    identification sums are equal as exact floats / arrays, over three micro-batches into one accumulator."""
    old, new = RideAlongAccumulator(), RideAlongAccumulator()
    for i in range(3):
        out, b, losses, classes = _synthetic_readout(case, seed=i)
        _legacy_observe(old, out, b, losses, classes)
        new.observe(out, b, losses, classes, 0)
        chim = {"rnd": out["rnd_err"] * 2, "rndv_fast": out["rndv_fast_err"] * 0.5}
        _legacy_observe_identification(old, out, chim)
        new.observe_identification(out, chim)
    assert len(new._pending) == 6
    _same_metrics(old.metrics(), new.metrics(), f"{case}: ")
    assert old.scalars == new.scalars, case                       # the raw per-micro-batch lists, in order
    assert old.cols.keys() == new.cols.keys()
    for key in old.cols:
        assert len(old.cols[key]) == len(new.cols[key])
        for x, y in zip(old.cols[key], new.cols[key]):
            assert x.dtype == y.dtype == np.float32 and x.shape == y.shape and x.tobytes() == y.tobytes(), key
    assert old.ident == new.ident
    if case == "masked":
        assert "ens_brier_v" not in new.scalars and "v_err" in new.cols       # the guard's skip, by NaN omission
    if case == "one_class":
        assert "rnd_z_" + _class_name(0) in new.scalars and "rnd_z_" + _class_name(1) not in new.scalars
    if case == "const_adv":
        assert "adv_explained_var" not in new.scalars
    if case == "zero_corr":
        assert "adv_corr_logit" not in new.scalars
    if case == "all_fed":
        assert "adv_std_starved" not in new.scalars and "adv_std_fed" in new.scalars
    if case == "normal":
        assert {"ens_brier_v", "adv_explained_var", "adv_corr_logit", "adv_std_starved", "adv_std_fed",
                "q_out_of_range", "opp_label_rate"} <= new.scalars.keys()


def _class_name(code):
    from agents.model.opp_intent import OPP_CLASS_NAMES

    return dict(OPP_CLASS_NAMES)[code] if isinstance(OPP_CLASS_NAMES, dict) else OPP_CLASS_NAMES[code]


def test_metrics_folds_whatever_is_pending_and_a_snapshot_holds_no_graph():
    acc = RideAlongAccumulator()
    p = th.nn.Parameter(th.ones(4))
    out = {"rnd_err": p * 2.0, "rnd_z": p * 3.0, "adv": p, "opp": p}
    b = types.SimpleNamespace(values=None, win_target=None, win_mask=None, actions=None, advantages=None,
                              legal=None, logits=None, pi=None, flat_target=None)
    acc.observe(out, b, {"rnd": (p * 2.0).mean()}, None, 0)
    (_kind, (snap_out, _b, snap_loss, _opp)), = acc._pending
    assert all(v is None or not v.requires_grad for v in snap_out.values()) and snap_out["adv"] is None
    assert all(not v.requires_grad for v in snap_loss.values())
    m = acc.metrics()
    assert m["rnd_loss"] == 2.0 and m["rnd_err_mean"] == 2.0 and m["rnd_z_mean"] == 3.0
    assert not acc._pending and acc.metrics() == m            # idempotent: nothing is folded twice
