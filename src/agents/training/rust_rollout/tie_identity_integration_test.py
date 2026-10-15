"""K9(b) tie IDENTITY on real Rust-collector rows, at every arm (`gen3_behaviour_tie_identity_v1`, 2026-10-05).

The rule: a selection tie between candidates whose selected payload is bit-identical is NOT a hazard (its
margin is the gap to the nearest DISTINCT candidate, `selection_sites.Rule.payload`), and at a fresh run's
first update (zero-init action scorers) no tie can reach log pi at all (`consistency.selection_free`).

On a seeded complete-game rollout per arm (CPU, T2 eager, p2 a seeded random policy, seeded pool teams):

* the payload rule CLEARS rows the rule before excluded, and resolving each cleared tie the OTHER way (the
  argmax flipped to the tied candidate) leaves the full masked log-probs BIT-IDENTICAL — the declaration is
  complete on real rows — while flipping a still-excluded (distinct) tie MOVES them (the flip has teeth);
* a planted 3e-4 behaviour mismatch on a judged row is FATAL at every arm, at trained-like (perturbed) and
  at fresh weights; at fresh weights no row is excluded.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch as th
from torch.overrides import TorchFunctionMode

from agents.model import selection_sites as SS
from agents.training.rust_rollout import consistency as K
from agents.training.rust_rollout import tie_margins as TM

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: The production surface (X5's fixed-mass hypothesis tokens — the blob arm was DELETED at the version break)
#: and the two oracle-reveal arms on top of it.
ARMS: Dict[str, Dict[str, str]] = {"fixed_mass": {}, "oracle_species": {"oracle_reveal": "species"},
                                   "oracle_full": {"oracle_reveal": "full"}}
#: The seeds are a FIXED DRAW (a collected test wants the same battles every run). The excluded share of a 512-row,
#: perturbed-weights rollout is CLUSTERED (a tie at a mon's move-order cut persists for a whole game, so ~16 games decide
#: it) and swings with the draw: at 1001 the production surface's ``oracle_species`` arm read 16.6 % (> the 15 % ceiling,
#: the probe is FATAL by design), the same arm over run seeds 1001-1005 reads 16.6 / 1.6 / 3.3 / 3.1 / 6.8 % on the
#: production surface and 5.7 / 5.1 / 8.0 / 2.7 / 7.8 % on the pre-adoption (legacy) one -- the same distribution, so
#: the draw, not the end state, was the cause. At 1002 the three arms read 2.1 / 1.6 / 0.2 % (fixed_mass / oracle_species /
#: oracle_full). A real run's healthy share is 2.7 % (mean over the eight end-state seeds' updates, max 6.8 %).
N_ENVS, N_STEPS, RUN_SEED, MODEL_SEED = 8, 64, 1002, 1002
#: The declared PAYLOAD sites, read once at import (the rule-before arm strips them from `SS.MARGIN`).
_PAYLOAD_RULES = {k: r for k, r in SS.MARGIN.items() if r.payload}


def _rollout(arm: str, fresh: bool, **surface: Any) -> Any:
    """A seeded complete-game rollout of ``arm`` on the PRODUCTION surface, with ``surface`` (resolved training-namespace
    keys) overridden on top: the one lever a test needs, e.g. ``op_reduction="max"`` (see ``max_reduction``)."""
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl, trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.train.production_args import production_args
    from utils.torch_state_guard import single_thread_build

    TK.build_selfcheck()
    args = production_args()
    for k, v in {**ARMS[arm], **surface}.items():
        setattr(args, k, v)
    obs, act = trainee_spaces(args)
    decl = RustEnvDecl(n_envs=N_ENVS, threads=2, front="ffi", profile="selfcheck", n_steps=N_STEPS,
                       micro_batch=N_STEPS, device="cpu", backend="eager", run_seed=RUN_SEED, gamma=1.0,
                       gae_lambda=0.8, oracle_reveal=ARMS[arm].get("oracle_reveal", "off"))
    p2 = TK.RandomP2(5)
    env = RustVecEnv(n_envs=N_ENVS, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    with single_thread_build():
        model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS * N_ENVS, seed=MODEL_SEED,
                               perturb_seed=1234, policy_args=args, perturb_keyed=True)
    if fresh:                       # a fresh run's first update: the zero-init action scorers
        with th.no_grad():
            for name in K._SCORERS:
                getattr(model.policy.pointer_head, name).weight.zero_()
    try:
        col = env.startup(model)
        assert col.collect(model, TK.NullCallback(), model.rollout_buffer)
    finally:
        env.close()
    model.behaviour_check = "fatal"
    model._current_progress_remaining = 1.0
    return model


class _ArgmaxFlip(TorchFunctionMode):
    """At the PAYLOAD argmax calls, in call order: resolve the argmax of the elements in ``flip[c]`` to the
    best OTHER candidate (the tied one) — what T2 would select had a rounding error crossed the tie."""

    def __init__(self, flip: List[np.ndarray]) -> None:
        super().__init__()
        self.flip, self.c, self.changed = flip, 0, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if _payload_site(func, args, TM._caller_frame()) is None:
            return out
        sel = self.flip[self.c]
        self.c += 1
        x = args[0]
        taken = th.zeros_like(x, dtype=th.bool).scatter_(-1, out, True)
        alt = th.where(taken, th.full_like(x, -float("inf")), x).argmax(dim=-1, keepdim=True)
        m = th.as_tensor(sel, device=x.device).reshape(out.shape)
        self.changed += int((m & (alt != out)).sum())
        return th.where(m, alt, out)


def _payload_site(func: Any, args: Tuple[Any, ...], frame: Any) -> Any:
    """(rule, frame, site) when ``func`` is an argmax at a declared PAYLOAD site, else None (``frame`` is
    `tie_margins._caller_frame()` taken IN the mode's ``__torch_function__``)."""
    if getattr(func, "__name__", "") != "argmax" or not args or not args[0].is_floating_point():
        return None
    if frame is None:
        return None
    where = (frame.f_code.co_filename.rsplit("/", 1)[-1][:-3], int(frame.f_lineno))
    res = SS.resolve(where[0], where[1], "sel")
    rule = None if res is None else _PAYLOAD_RULES.get((where[0], res.src))
    if rule is None:
        return None
    return rule, frame, f"{where[0]}.py:{where[1]} argmax"


def _forward(model: Any, mode: Any) -> np.ndarray:
    from stable_baselines3.common.utils import obs_as_tensor

    buf = model.rollout_buffer
    n = buf.log_probs.size
    obs = {k: v.reshape((n,) + v.shape[2:]) for k, v in buf.observations.items()}
    model.policy.set_training_mode(True)
    with th.no_grad(), mode:
        model.policy.evaluate_actions(obs_as_tensor(obs, "cpu"), th.as_tensor(buf.actions.reshape(-1)).long(),
                                      action_masks=th.as_tensor(buf.action_masks.reshape(n, -1)))
    return K._stashed_logp(model.policy)


class _Elements(TM.TieMargins):
    """The recorder, also keeping every PAYLOAD argmax call's per-element margin (in call order) under
    the payload rule (``payload=True``) or the rule before (``False``)."""

    def __init__(self, rows: int, payload: bool) -> None:
        super().__init__(rows)
        self.payload = payload
        self.elem: List[th.Tensor] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = super().__torch_function__(func, types, args, kwargs)
        hit = _payload_site(func, args, TM._caller_frame())
        if hit is not None:
            rule, frame, site = hit
            pay = TM.payload_tensors(rule, frame, args[0], site) if self.payload else None
            self.elem.append(TM.site_margin(rule, "argmax", args, kwargs or {}, pay, out))
        return out


def _without_payloads() -> Dict[Any, Any]:
    """The rule BEFORE `gen3_behaviour_tie_identity_v1`: no payload identity (undone by `_restore`)."""
    saved = dict(SS.MARGIN)
    for k, r in list(SS.MARGIN.items()):
        if r.payload:
            SS.MARGIN[k] = r._replace(payload=())
    SS.line_map.cache_clear()
    return saved


def _restore(saved: Dict[Any, Any]) -> None:
    SS.MARGIN.clear()
    SS.MARGIN.update(saved)
    SS.line_map.cache_clear()


@pytest.fixture(scope="module")
def oracle_full():
    return _rollout("oracle_full", fresh=False)


@pytest.fixture(scope="module")
def max_reduction():
    """The ``oracle_full`` arm on the ``--op-reduction max`` surface: the ONLY one whose forward reaches the declared
    PAYLOAD argmaxes (`damage_op`'s dominant-move accuracy / provenance). The production reduction, ``principled``
    (since the END STATE's adoption, 2026-10-12), has no argmax and no max, so the payload rule has nothing to clear
    there (`test_the_production_reduction_reaches_no_payload_argmax`); the rule is the `max` reduction's, kept until
    that path is deleted (`designs/ops/legacy_deletion_manifest_2026-10-12.md`, lane OP)."""
    return _rollout("oracle_full", fresh=False, op_reduction="max")


def test_the_production_reduction_reaches_no_payload_argmax(oracle_full):
    """The adopted truth the test below leans on: the principled op reduction (production) reaches NO declared payload
    argmax, so no argmax tie can reach log pi there at all."""
    model = oracle_full
    assert model.policy.features_extractor.op_reduction == "principled"
    rec = _Elements(model.rollout_buffer.log_probs.size, payload=True)
    _forward(model, rec)
    rec.check()
    assert not rec.elem, "a payload argmax ran under the principled reduction: its tie rule is live again"


def test_a_payload_identical_tie_is_cleared_and_resolving_it_the_other_way_changes_nothing(max_reduction):
    model = max_reduction
    assert model.policy.features_extractor.op_reduction == "max"        # the surface that HAS payload argmaxes
    n = model.rollout_buffer.log_probs.size
    eps = K.FP32_TIE_EPS
    new = _Elements(n, payload=True)
    base = _forward(model, new)
    new.check()
    saved = _without_payloads()
    try:
        old = _Elements(n, payload=False)
        _forward(model, old)
    finally:
        _restore(saved)
    cleared_rows = (old.margin < eps) & ~(new.margin < eps)
    assert cleared_rows.sum() >= 2, "the seeded rollout must hold rows the payload rule clears"
    assert not ((new.margin < eps) & ~(old.margin < eps)).any(), "identity never excludes a row the rule before judged"
    assert len(new.elem) == len(old.elem) >= 2
    cleared = [((go < eps) & (gn >= eps)).cpu().numpy() for go, gn in zip(old.elem, new.elem)]
    distinct = [((go < eps) & (gn < eps)).cpu().numpy() for go, gn in zip(old.elem, new.elem)]
    fm = _ArgmaxFlip(cleared)
    flipped = _forward(model, fm)
    assert fm.changed >= 2, "the flip must actually move the selection on the cleared ties"
    assert np.array_equal(flipped, base, equal_nan=True), "a cleared tie MOVED log pi: the payload is incomplete"
    # teeth: flipping the ties the payload rule still EXCLUDES (distinct payloads) moves log pi
    fd = _ArgmaxFlip(distinct)
    moved = _forward(model, fd)
    assert fd.changed >= 1 and not np.array_equal(moved, base, equal_nan=True)


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("fresh", [False, True], ids=["perturbed", "zero_scorers"])
def test_a_planted_behaviour_mismatch_on_a_judged_row_is_FATAL_at_every_arm(arm, fresh, oracle_full):
    model = oracle_full if (arm, fresh) == ("oracle_full", False) else _rollout(arm, fresh)
    assert K.selection_free(model.policy) == fresh
    out = K.behaviour_probe(model)                     # the unmodified buffer passes
    assert out["behaviour/max_abs_dlogp_judged"] < K.BEHAVIOUR_BAR
    assert out["behaviour/excluded_frac"] < K.FP32_EXCLUDED_CEILING
    if fresh:
        assert out["behaviour/rows_excluded"] == 0.0
    # plant on a JUDGED row (the probe's own margins: the recorder's forward on its chosen rows)
    buf = model.rollout_buffer
    rng = np.random.default_rng([int(model.seed or 0), int(model.num_timesteps)])
    f = K.choose_rows(np.zeros(buf.log_probs.shape, np.int64), model.batch_size, rng)
    t, e = f // buf.n_envs, f % buf.n_envs
    g, _s = TM.selection_gaps(model.policy, {k: v[t, e] for k, v in buf.observations.items()}, buf.actions[t, e],
                              buf.action_masks[t, e], "cpu")
    i = int(np.argmax(g))
    assert g[i] >= K.FP32_TIE_EPS
    keep = buf.log_probs[t[i], e[i]].copy()
    buf.log_probs[t[i], e[i]] += 3e-4
    try:
        with pytest.raises(K.BehaviourMismatch, match=r"max 0\.0003 NOT < 0\.0001"):
            K.behaviour_probe(model)
    finally:
        buf.log_probs[t[i], e[i]] = keep


# ---------------------------------------------------------------- gen3_behaviour_tie_consumed_v1 (2026-10-06)
# The X5 sort site counts a tie only where the CALLER reads the order (`hypothesis_set.stable_order`'s
# ``consumed``): the op's per-mon move orders (`build_op_roster`, `other_roster`) are read as a SET before
# each cut, so only the pair straddling a cut is a boundary.

_SET_CALLERS = ("build_op_roster", "other_roster")


class _PermuteSetPrefix(TorchFunctionMode):
    """At every X5 sort issued by a SET-reading caller: a seeded random permutation of the first ``K``
    positions of each order (``swap_boundary``: swap positions K−1 and K instead — the teeth)."""

    def __init__(self, K: int, seed: int, swap_boundary: bool = False) -> None:
        super().__init__()
        self.K, self.g, self.swap, self.calls = K, th.Generator().manual_seed(seed), swap_boundary, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") != "argsort":
            return out
        f = TM._caller_frame()
        if f is None or f.f_code.co_name != "stable_order" or f.f_back.f_code.co_name not in _SET_CALLERS:
            return out
        self.calls += 1
        flat = out.clone().reshape(-1, out.shape[-1])
        if self.swap:
            flat[:, [self.K - 1, self.K]] = flat[:, [self.K, self.K - 1]]
        else:
            p = th.argsort(th.rand(flat.shape[0], self.K, generator=self.g), dim=-1)
            flat[:, :self.K] = flat[:, :self.K].gather(-1, p)
        return flat.reshape(out.shape)


def _without_consumed() -> Dict[Any, Any]:
    """The rule BEFORE `gen3_behaviour_tie_consumed_v1`: every pair of the sort head (undone by `_restore`)."""
    saved = dict(SS.MARGIN)
    for k, r in list(SS.MARGIN.items()):
        if r.consumed:
            SS.MARGIN[k] = r._replace(consumed="")
    SS.line_map.cache_clear()
    return saved


@pytest.fixture(scope="module")
def fixed_mass():
    return _rollout("fixed_mass", fresh=False)


#: What reordering a set-read prefix may move a fp32 log pi by: float-sum ROUNDING and nothing else. The principled op
#: reduction (production since the END STATE's adoption, 2026-10-12) reads each mon's first K seats through alpha-weighted
#: SUMS, and a permuted prefix re-associates them -- a few ulp of a log pi (measured <= 9.5e-7 over ten production
#: rollouts: the three arms, run seeds 1001-1005), where the `max` reduction's maxima are order-free and were BIT-IDENTICAL. The bar is a
#: tenth of the K9(b) judgement bar (`K.BEHAVIOUR_BAR`, 1e-4): 10x above that noise, 10x below the line a tie must cross
#: to matter, and the cut's own swap (below) must clear the judgement bar itself.
SET_ORDER_ROUNDING = K.BEHAVIOUR_BAR / 10


def _max_abs_dlogp(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a - b| over the entries both hold a real log-prob for; the MASKED entries (the -1e8 fill) must coincide."""
    assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
    real = a > -1e7                                   # False at a NaN and at the mask's fill
    assert np.array_equal(real, b > -1e7), "the action mask moved: a permutation reached the mask"
    assert np.array_equal(a[~real], b[~real], equal_nan=True)
    return float(np.abs(a[real] - b[real]).max())


def test_a_set_read_prefix_is_order_free_on_real_rows_and_its_cut_is_not(fixed_mass):
    """The declaration's premise, on every row: permuting the first K = the cut of EVERY per-mon order
    moves the full masked log-probs by float-sum rounding at most (`SET_ORDER_ROUNDING`; so a tie inside the set cannot
    move log pi), while swapping the pair across the cut moves them past the K9(b) judgement bar (the cut stays a
    boundary)."""
    model = fixed_mass
    ext = model.policy.features_extractor
    cuts = {int(ext.consequence_topk), int(ext.entity_topk_seats)}
    assert len(cuts) == 1, "the production cuts coincide; a split cut needs a permutation per cut"
    K_cut = cuts.pop()
    base = _forward(model, contextlib.nullcontext())
    for seed in (1, 2):
        pm = _PermuteSetPrefix(K_cut, seed)
        d = _max_abs_dlogp(_forward(model, pm), base)
        assert d < SET_ORDER_ROUNDING, f"a set-read prefix's ORDER moved log pi by {d:.3g} (rounding bar {SET_ORDER_ROUNDING:g})"
        assert pm.calls >= 2
    sw = _PermuteSetPrefix(K_cut, 0, swap_boundary=True)
    moved = _max_abs_dlogp(_forward(model, sw), base)
    assert sw.calls >= 2 and moved >= K.BEHAVIOUR_BAR, f"the cut's swap moved log pi by only {moved:.3g}"


def test_the_consumed_rule_clears_set_internal_ties_and_never_excludes_a_new_row(fixed_mass):
    model = fixed_mass
    n = model.rollout_buffer.log_probs.size
    eps = K.FP32_TIE_EPS
    new = TM.TieMargins(n)
    _forward(model, new)
    new.check()
    saved = _without_consumed()
    try:
        old = TM.TieMargins(n)
        _forward(model, old)
    finally:
        _restore(saved)
    assert not ((new.margin < eps) & ~(old.margin < eps)).any(), "the declaration never excludes a judged row"
    assert ((old.margin < eps) & ~(new.margin < eps)).sum() >= 2, \
        "the seeded rollout must hold rows whose only tie is inside a set-read prefix"
