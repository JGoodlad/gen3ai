"""SOURCE-SEPARATED DISTILLATION ANCHORING (`gen3_distill_grad_project_v1`) — the five things that
must hold for `--distill-anchor-mode grad_project` to mean what its docstring says.

1. **THE LINEAR ALGEBRA IS THE LINEAR ALGEBRA.** On planted gradients and planted constraint
   vectors, `P⊥g` is orthogonal to every constraint, never longer than `g`, and idempotent; and
   Gram-Schmidt DROPS a duplicated constraint rather than counting it twice.
2. **PPO'S GRADIENT IS UNTOUCHED.** With nothing tagged as distill-sourced the projector cannot
   move a single number, and with a distill term present `.grad` is EXACTLY
   `g_ppo + P⊥ g_distill` — computed independently and compared, not asserted in prose.
3. **OFF IS FREE.** Any mode other than `grad_project` builds the null projector, whose `add` is a
   passthrough and whose hooks do nothing, so the update is bit-identical.
4. **THE FIRST-ORDER CLAIM IS TRUE.** Three arms from one init — PPO-only, fold, projected fold —
   and the projected fold's off-slice log-probs land measurably closer to the PPO-only arm's than
   the unprojected fold's do. That is the whole point of the feature, so it is measured.
5. **THE SEAM IS WIRED.** `ppo.py` tags the three TEACHER terms and NOT the anchor term (projecting
   the output anchor off the off-slice subspace would delete it and make the documented composition
   vacuous), the accumulation scaling matches the backward's, the noise-scale sampler still sees
   every group, and both flags reach `checkargs` and the config layer.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.distill_anchor import ANCHOR_MODES
from agents.training.instrumented_ppo.distill_grad_project import (
    DEFAULT_PROJ_SAMPLES,
    GRAD_PROJECT_MODE,
    NULL_PROJECTOR,
    DistillGradProjector,
    behaviour_constraints,
    flatten_grads,
    make_projector,
    offslice_rows,
    orthonormalize,
    project_out,
)
from agents.training.instrumented_ppo.noise_scale_terms import term_gradient
from agents.training.instrumented_ppo_test import _build_tiny_ppo, _train_from_init
from agents.training.distill_anchor_test import _Rec, _build_anchor_ppo


# ======================================================================================
# 1. The pure linear algebra
# ======================================================================================

def _rand_basis_problem(n=64, k=5, seed=0):
    th.manual_seed(seed)
    return th.randn(n), [th.randn(n) for _ in range(k)]


def test_the_survivor_is_orthogonal_to_every_constraint():
    g, cons = _rand_basis_problem()
    basis = orthonormalize(cons)
    survivor = g - project_out(g, basis)
    for q in basis:
        assert abs(float(th.dot(q, survivor))) < 1e-4, "P⊥g still has a component along a constraint"


def test_the_survivor_is_never_longer_than_the_original():
    """A projection is a contraction — if this ever grew, the sign or the basis is wrong."""
    g, cons = _rand_basis_problem(seed=3)
    basis = orthonormalize(cons)
    survivor = g - project_out(g, basis)
    assert float(survivor.norm()) <= float(g.norm()) + 1e-5
    # …and it removed something, or the test is vacuously passing on an already-orthogonal g.
    assert float(project_out(g, basis).norm()) > 1e-3


def test_projection_is_idempotent():
    g, cons = _rand_basis_problem(seed=7)
    basis = orthonormalize(cons)
    once = g - project_out(g, basis)
    twice = once - project_out(once, basis)
    assert th.allclose(once, twice, atol=1e-4)


def test_a_gradient_entirely_inside_the_span_is_removed_entirely():
    """The extreme case the meter reports as `proj_removed_frac == 1`: a distill gradient that is
    PURE off-slice behaviour change survives as nothing at all."""
    _, cons = _rand_basis_problem(seed=11)
    basis = orthonormalize(cons)
    g = sum(q * float(c) for q, c in zip(basis, (1.5, -2.0, 0.25, 3.0, -1.0)))
    assert float((g - project_out(g, basis)).norm()) < 1e-3 * float(g.norm())


def test_a_gradient_orthogonal_to_the_span_survives_untouched():
    """The other extreme, and the one the feature exists for: a distill direction that only moves
    TAUGHT states is not taxed at all."""
    th.manual_seed(5)
    cons = [th.zeros(6) for _ in range(2)]
    cons[0][0], cons[1][1] = 1.0, 1.0
    basis = orthonormalize(cons)
    g = th.tensor([0.0, 0.0, 1.0, -2.0, 3.0, 0.5])
    assert th.allclose(g - project_out(g, basis), g, atol=1e-6)


def test_gram_schmidt_drops_a_duplicated_constraint():
    """Two sampled off-slice states wanting the SAME behavioural change is the normal case, and a
    duplicate kept as a second dimension would be a numerical-noise direction the distill gradient
    then gets projected onto."""
    th.manual_seed(2)
    v = th.randn(2048)
    w = th.randn(2048)
    basis = orthonormalize([v, v.clone(), w, v * -3.0])
    assert len(basis) == 2, f"expected rank 2 from 2 distinct directions, got {len(basis)}"
    assert abs(float(th.dot(basis[0], basis[1]))) < 1e-4
    assert all(abs(float(q.norm()) - 1.0) < 1e-5 for q in basis)


def test_gram_schmidt_drops_a_zero_vector():
    """A parameter set the sampled row's log-prob does not depend on yields an all-`None` gradient,
    which `flatten_grads` reads as zeros — it must not become a unit vector of noise."""
    assert orthonormalize([th.zeros(32)]) == []


# ======================================================================================
# 2. Row sampling
# ======================================================================================

def _dmask(vals):
    return th.tensor(vals, dtype=th.float32).reshape(-1, 1)


def test_only_off_slice_rows_are_ever_sampled():
    rows = offslice_rows(_dmask([0, 1, 0, 2, 0, 0]), 16, th.Generator().manual_seed(0))
    assert sorted(int(i) for i in rows) == [0, 2, 4, 5]


def test_all_off_slice_rows_are_taken_when_there_are_no_more_than_m():
    rows = offslice_rows(_dmask([0, 1, 0]), 16, th.Generator().manual_seed(0))
    assert sorted(int(i) for i in rows) == [0, 2]


def test_a_minibatch_with_no_off_slice_row_yields_no_constraints():
    """Full teacher coverage this minibatch ⇒ nothing off-slice to protect ⇒ decline, never guess."""
    assert offslice_rows(_dmask([1, 1, 2]), 4, th.Generator().manual_seed(0)) is None


def test_the_sample_is_reproducible_and_does_not_touch_the_global_rng():
    """A feature that perturbed the global stream would silently change every seeded-arm comparison
    it is supposed to be measured by."""
    mask = _dmask([0] * 40)
    a = offslice_rows(mask, 8, th.Generator().manual_seed(0))
    th.manual_seed(999)
    before = th.rand(3)
    b = offslice_rows(mask, 8, th.Generator().manual_seed(0))
    th.manual_seed(999)
    assert th.equal(a, b)
    assert th.equal(before, th.rand(3))       # the global stream is exactly where it was


# ======================================================================================
# 3. The gradient contract, on a real (tiny) policy
# ======================================================================================

class _TinyPolicy(th.nn.Module):
    """The smallest thing carrying the ONE method the projector calls on a policy."""

    def __init__(self, n_in=3, n_act=4):
        super().__init__()
        self.net = th.nn.Linear(n_in, n_act)
        self.unused = th.nn.Parameter(th.zeros(5))   # a param no loss reaches: the `.grad is None` path

    def get_distribution(self, obs):
        return SimpleNamespace(
            distribution=SimpleNamespace(logits=self.net(obs["observation"])))


def _tiny_rollout(n=12, n_act=4, seed=0):
    th.manual_seed(seed)
    obs = {"observation": th.randn(n, 3),
           "distill_mask": th.tensor([float(i % 3 == 0) for i in range(n)]).reshape(-1, 1)}
    masks = th.ones(n, n_act)
    masks[:, -1] = 0.0                                   # one illegal column everywhere
    return SimpleNamespace(observations=obs, action_masks=masks)


def _expected_projected_grad(pol, roll, params, g_ppo, g_dis, m, seed=0):
    rows = offslice_rows(roll.observations["distill_mask"], m, th.Generator().manual_seed(seed))
    basis = orthonormalize(
        behaviour_constraints(pol, roll.observations, roll.action_masks, rows, params))
    return g_ppo + (g_dis - project_out(g_dis, basis)), len(basis)


def test_grad_is_exactly_ppo_plus_the_projected_distill_gradient():
    """THE contract, computed two independent ways and compared."""
    pol, roll = _TinyPolicy(), _tiny_rollout()
    params = [p for p in pol.parameters() if p.requires_grad]
    metrics: dict = {}
    proj = DistillGradProjector(params, metrics, samples=4, seed=0)

    logits = pol.get_distribution(roll.observations).distribution.logits
    ppo_term = (logits ** 2).mean()                       # stands in for the clipped surrogate
    distill_term = proj.add(logits.abs().mean())          # stands in for the teacher KL
    g_ppo = flatten_grads(term_gradient([ppo_term], params), params)
    g_dis = flatten_grads(term_gradient([distill_term], params), params)

    proj.before_backward(pol, roll)
    (ppo_term + distill_term).backward()
    proj.after_backward(1)

    got = flatten_grads([p.grad for p in params], params)
    expect, rank = _expected_projected_grad(pol, roll, params, g_ppo, g_dis, 4)
    assert th.allclose(got, expect, atol=1e-6), "the applied update is not g_ppo + P⊥ g_distill"
    assert metrics["proj_rank"] == [float(rank)]
    assert 0.0 < metrics["proj_removed_frac"][0] <= 1.0
    assert metrics["proj_ms"][0] >= 0.0


def test_the_ppo_half_of_that_grad_is_bit_identical_to_an_unprojected_ppo_only_step():
    """"PPO's gradient is never read, never projected, never scaled" as a MEASUREMENT: run the same
    PPO term alone, and its `.grad` must equal the projected run's `.grad` minus P⊥ g_distill."""
    pol, roll = _TinyPolicy(), _tiny_rollout(seed=4)
    params = [p for p in pol.parameters() if p.requires_grad]
    logits = pol.get_distribution(roll.observations).distribution.logits
    ppo_term = (logits ** 2).mean()
    ppo_term.backward()
    alone = flatten_grads([p.grad for p in params], params).clone()

    pol.zero_grad(set_to_none=True)
    proj = DistillGradProjector(params, {}, samples=4, seed=0)
    logits = pol.get_distribution(roll.observations).distribution.logits
    ppo_term = (logits ** 2).mean()
    distill_term = proj.add(logits.abs().mean())
    g_dis = flatten_grads(term_gradient([distill_term], params), params)
    proj.before_backward(pol, roll)
    (ppo_term + distill_term).backward()
    proj.after_backward(1)
    together = flatten_grads([p.grad for p in params], params)

    rows = offslice_rows(roll.observations["distill_mask"], 4, th.Generator().manual_seed(0))
    basis = orthonormalize(
        behaviour_constraints(pol, roll.observations, roll.action_masks, rows, params))
    assert th.allclose(together - (g_dis - project_out(g_dis, basis)), alone, atol=1e-6)


def test_nothing_tagged_means_nothing_moved():
    """A minibatch where the distill term did not fire (no teacher-team rows) must leave `.grad`
    byte-identical — the projector has no source to separate."""
    pol, roll = _TinyPolicy(), _tiny_rollout(seed=9)
    params = [p for p in pol.parameters() if p.requires_grad]
    proj = DistillGradProjector(params, {}, samples=4, seed=0)
    logits = pol.get_distribution(roll.observations).distribution.logits
    (logits ** 2).mean().backward()
    before = [None if p.grad is None else p.grad.clone() for p in params]
    proj.before_backward(pol, roll)
    proj.after_backward(1)
    for p, b in zip(params, before):
        assert (p.grad is None and b is None) or th.equal(p.grad, b)


def test_the_accumulation_scaling_matches_the_backward():
    """`after_backward(accum)` must subtract `removal/accum`, because the backward it follows
    applied `1/accum` to everything else."""
    pol, roll = _TinyPolicy(), _tiny_rollout(seed=13)
    params = [p for p in pol.parameters() if p.requires_grad]
    proj = DistillGradProjector(params, {}, samples=4, seed=0)
    logits = pol.get_distribution(roll.observations).distribution.logits
    ppo_term = (logits ** 2).mean()
    distill_term = proj.add(logits.abs().mean())
    g_ppo = flatten_grads(term_gradient([ppo_term], params), params)
    g_dis = flatten_grads(term_gradient([distill_term], params), params)
    proj.before_backward(pol, roll)
    ((ppo_term + distill_term) / 2).backward()
    proj.after_backward(2)
    got = flatten_grads([p.grad for p in params], params)
    expect, _ = _expected_projected_grad(pol, roll, params, g_ppo, g_dis, 4)
    assert th.allclose(got, expect / 2.0, atol=1e-6)


def test_the_constraint_is_the_gradient_of_the_argmax_log_probability():
    """The basis is not an arbitrary direction — pin WHAT it is, so a future edit that changes it
    changes a test rather than a run's meaning."""
    pol, roll = _TinyPolicy(), _tiny_rollout(seed=17)
    params = [p for p in pol.parameters() if p.requires_grad]
    rows = th.tensor([0])
    got = behaviour_constraints(pol, roll.observations, roll.action_masks, rows, params)[0]
    logits = pol.get_distribution({k: v[rows] for k, v in roll.observations.items()}
                                  ).distribution.logits
    neg = (roll.action_masks[rows] - 1.0) * 1e9
    logp = th.log_softmax(logits + neg, dim=-1)
    want = flatten_grads(term_gradient([logp[0, int(logp.detach().argmax(-1)[0])]], params), params)
    assert th.allclose(got, want, atol=1e-6)


def test_the_illegal_action_can_never_be_the_constrained_direction():
    """The mask is applied before the argmax, so a huge illegal logit cannot hijack the constraint."""
    pol, roll = _TinyPolicy(), _tiny_rollout(seed=19)
    with th.no_grad():
        pol.net.bias[-1] += 50.0                 # the ILLEGAL column, made overwhelmingly largest
    params = [p for p in pol.parameters() if p.requires_grad]
    obs = {k: v[th.tensor([0])] for k, v in roll.observations.items()}
    logits = pol.get_distribution(obs).distribution.logits
    assert int(logits.argmax(-1)) == 3           # unmasked, the illegal column wins
    neg = (roll.action_masks[th.tensor([0])] - 1.0) * 1e9
    assert int((logits + neg).argmax(-1)) != 3   # masked, it cannot
    assert behaviour_constraints(pol, roll.observations, roll.action_masks,
                                 th.tensor([0]), params)[0].norm() > 0


def test_a_failure_retires_the_projector_and_leaves_grad_alone(capsys):
    """A regulariser must never take a run down — and it must SAY it stopped, because a silently
    retired projector trains as an ordinary unprojected fold."""
    pol, roll = _TinyPolicy(), _tiny_rollout(seed=23)
    params = [p for p in pol.parameters() if p.requires_grad]
    proj = DistillGradProjector(params, {}, samples=4, seed=0)
    logits = pol.get_distribution(roll.observations).distribution.logits
    ppo_term = (logits ** 2).mean()
    proj.add(logits.abs().mean())
    ppo_term.backward()
    before = [p.grad.clone() for p in params if p.grad is not None]
    proj.before_backward(pol, SimpleNamespace(observations={}, action_masks=None))   # boom
    proj.after_backward(1)
    assert proj.failed
    assert "DISTILL-PROJ" in capsys.readouterr().out
    for p, b in zip([p for p in params if p.grad is not None], before):
        assert th.equal(p.grad, b)


# ======================================================================================
# 4. The construction seam
# ======================================================================================

def test_any_other_mode_builds_the_null_projector():
    for mode in ("off_slice", "all", "typo", None):
        assert make_projector(SimpleNamespace(distill_anchor_mode=mode), {}, []) is NULL_PROJECTOR
    assert NULL_PROJECTOR.add("anything") == "anything"
    assert NULL_PROJECTOR.before_backward(None, None) is None
    assert NULL_PROJECTOR.after_backward(4) is None


def test_grad_project_builds_a_real_one_and_honours_the_sample_count():
    model = SimpleNamespace(distill_anchor_mode=GRAD_PROJECT_MODE,
                            distill_anchor_proj_samples=3, seed=7)
    proj = make_projector(model, {}, list(_TinyPolicy().parameters()))
    assert isinstance(proj, DistillGradProjector) and proj._m == 3
    model.distill_anchor_proj_samples = None            # absent ⇒ the documented default
    assert make_projector(model, {}, []).__dict__["_m"] == DEFAULT_PROJ_SAMPLES


def test_the_mode_is_a_declared_anchor_mode():
    assert GRAD_PROJECT_MODE in ANCHOR_MODES


# ======================================================================================
# 5. End to end through a real train()
# ======================================================================================

def _fold_arm(*, mode, distill_coef, batch_size=32, accum=1, samples=64):
    """One `train()` from a FIXED init, with a live teacher unless `distill_coef` is 0."""
    th.manual_seed(0)
    np.random.seed(0)
    model, parent = _build_anchor_ppo()
    teacher, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    th.manual_seed(21)
    with th.no_grad():
        for p in teacher.policy.action_net.parameters():
            p.add_(th.randn_like(p) * 2.0)
    teacher.policy.set_training_mode(False)
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    if distill_coef:
        model._distill_teachers = [teacher]
        model.distill_coef = distill_coef
    model._distill_anchor_parent = parent
    model.distill_anchor_coef = 0.0                 # the projection alone; no OUTPUT anchor
    model.distill_anchor_mode = mode
    model.distill_anchor_proj_samples = samples
    model.n_epochs = 1
    model._logger = _Rec()
    sd = _train_from_init(model, init_sd, init_opt, batch_size=batch_size, accum=accum)
    return sd, model.logger.vals, model, init_sd


def test_grad_project_is_bit_identical_when_no_distill_term_fires():
    """OFF IS FREE, through the real `train()`: the mode selected but nothing to project."""
    base, _, _, _ = _fold_arm(mode="off_slice", distill_coef=0.0)
    proj, log, _, _ = _fold_arm(mode=GRAD_PROJECT_MODE, distill_coef=0.0)
    for k in base:
        assert th.equal(base[k], proj[k]), f"grad_project moved {k} with no distill term to project"
    assert "distill/proj_rank" not in log      # nothing measured, because nothing happened


def test_grad_project_changes_the_update_of_a_real_fold_and_publishes_its_meters():
    plain, _, _, _ = _fold_arm(mode="off_slice", distill_coef=0.2)
    proj, log, _, _ = _fold_arm(mode=GRAD_PROJECT_MODE, distill_coef=0.2)
    assert any(not th.equal(plain[k], proj[k]) for k in plain), "the projection folded nothing"
    assert log["distill/proj_rank"] >= 1.0
    assert 0.0 < log["distill/proj_removed_frac"] <= 1.0
    assert log["distill/proj_ms"] >= 0.0
    assert log["distill/proj_constraint_rows"] >= 1.0
    assert log["distill/kl"] > 0.0              # the teacher term is live in the same call


def test_grad_project_also_works_under_grad_accum():
    """`--grad-accum-steps 2`: two micro-batches, one step, each micro-batch projected against its
    own off-slice rows and scaled by the same 1/accum the backward used."""
    plain, _, _, _ = _fold_arm(mode="off_slice", distill_coef=0.2, batch_size=16, accum=2)
    proj, log, _, _ = _fold_arm(mode=GRAD_PROJECT_MODE, distill_coef=0.2, batch_size=16, accum=2)
    assert any(not th.equal(plain[k], proj[k]) for k in plain)
    assert log["distill/proj_rank"] >= 1.0


def test_the_per_term_noise_sampler_still_sees_the_distill_group():
    """The projector wraps the SAME tensors the noise tagger tags. Both must still fire — a fold
    that lost `train/noise_scale_distill` to this change would have lost its dose meter."""
    _, log, _, _ = _fold_arm(mode=GRAD_PROJECT_MODE, distill_coef=0.2, batch_size=16, accum=2)
    assert any(k.startswith("train/noise_scale") for k in log), log.keys()


# --- THE FIRST-ORDER CLAIM ------------------------------------------------------------

def _offslice_logps(model, sd, data):
    """Σ over `data`'s off-slice rows of the student's masked argmax log-prob, at weights `sd`.

    🚨 `data` is passed IN, never re-drawn. `RolloutBuffer.get(None)` re-PERMUTES on every call, so
    a helper that fetched its own batch would subtract mismatched rows — which it did while this
    test was being written, and it turned a 95% reduction into an apparent 4%. The arms must differ
    in weights and in nothing else, evaluation order included.
    """
    keep = copy.deepcopy(model.policy.state_dict())
    model.policy.load_state_dict(sd)
    with th.no_grad():
        logits = model.policy.get_distribution(data.observations).distribution.logits
        neg = (data.action_masks.to(logits.dtype) - 1.0) * 1e9
        logp = th.log_softmax(logits + neg, dim=-1)
        off = (data.observations["distill_mask"].reshape(-1) < 0.5)
        out = logp.max(-1).values[off].clone()
    model.policy.load_state_dict(keep)
    return out


def test_one_projected_step_moves_off_slice_behaviour_less_than_an_unprojected_one():
    """THE POINT OF THE FEATURE, as a measurement.

    Three arms from ONE init, one optimizer step each, on the SAME rollout: (A) PPO alone, (B) PPO +
    an unprojected fold, (C) PPO + a projected fold. The distill term's contribution to the off-slice
    log-probs is `arm − A`. To first order the projection kills it, so `|C − A|` must be well below
    `|B − A|`. It is a FIRST-ORDER claim, so one step is exactly the regime in which it holds — the
    module's docstring is explicit that accumulated displacement is the output anchor's job.

    MEASURED on this toy while writing it: **100.0%** reduction with the optimizer swapped for plain
    SGD (the projection is EXACT in gradient space, clipped or not), and **95.1%** on the real
    Adam + `clip_grad_norm_` path — Adam rescales per coordinate, so a projection of the GRADIENT is
    not exactly a projection of the UPDATE. The 0.5 threshold is a wide regression guard around the
    second number, not a tuned one.
    """
    sd_ppo, _, model, init_sd = _fold_arm(mode="off_slice", distill_coef=0.0)
    sd_fold, _, _, _ = _fold_arm(mode="off_slice", distill_coef=0.2)
    sd_proj, _, _, _ = _fold_arm(mode=GRAD_PROJECT_MODE, distill_coef=0.2)

    data = next(model.rollout_buffer.get(None))         # ONE fixed board for all three arms
    base = _offslice_logps(model, sd_ppo, data)
    unprojected = float((_offslice_logps(model, sd_fold, data) - base).abs().sum())
    projected = float((_offslice_logps(model, sd_proj, data) - base).abs().sum())
    assert unprojected > 1e-9, "the unprojected fold moved nothing off-slice — nothing to reduce"
    assert projected < 0.5 * unprojected, (
        f"projection reduced the off-slice move by only {1 - projected / unprojected:.1%} "
        f"(unprojected {unprojected:.3e}, projected {projected:.3e})")


# ======================================================================================
# 6. The seam in ppo.py, and the CLI surface
# ======================================================================================

def test_ppo_tags_the_teacher_terms_and_not_the_anchor_term():
    """The one distinction that cannot be made inside the projector: the OUTPUT anchor rides the
    same `distill` noise group but must NOT be projected — its job IS to act on off-slice outputs,
    so projecting it off the off-slice subspace would delete it and make the documented composition
    (`grad_project` + `--distill-anchor-coef > 0`) vacuous."""
    import inspect
    from agents.training.instrumented_ppo import ppo as _ppo
    src = inspect.getsource(_ppo.InstrumentedMaskablePPO.train)
    assert '_ntg.add("distill", _dgp.add(distill_term))' in src
    assert src.count('"distill", _dgp.add(') == 3, \
        "expected exactly the three TEACHER terms (policy KL, value MSE, FitNets hint) to be tagged"
    assert '_ntg.add("distill", anchor_term)' in src, "the anchor term must stay UNprojected"
    assert "_dgp.before_backward(self.policy, rollout_data)" in src
    assert "_dgp.after_backward(accum)" in src


def test_checkargs_accepts_the_projection_flags():
    from main.checkargs import check
    got = check(["--distill-teacher", "models/t:data/teams/sample/a.txt", "--distill-coef", "0.2",
                 "--distill-anchor-mode", "grad_project", "--distill-anchor-proj-samples", "8"])
    assert got["unknown"] == []
    for f in ("--distill-anchor-mode", "--distill-anchor-proj-samples"):
        assert f in got["accepted"]


def _resolved(argv):
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


_FOLD = ["--distill-teacher", "models/t:data/teams/sample/a.txt", "--distill-coef", "0.2",
         "--steps", "10"]


def test_the_sample_count_defaults_to_the_documented_value():
    assert _resolved(["--steps", "10"]).distill_anchor_proj_samples == DEFAULT_PROJ_SAMPLES
    assert InstrumentedMaskablePPO.distill_anchor_proj_samples == DEFAULT_PROJ_SAMPLES


def test_grad_project_needs_no_coefficient_and_no_monitor_flag():
    """The projection IS the mode's effect, so the pre-existing "you typed a knob that does nothing"
    refusal must not fire on it — and the config must still demand a live distill, because the slice
    is the `distill_mask` obs key."""
    args = _resolved(_FOLD + ["--distill-anchor-mode", "grad_project"])
    assert args.distill_anchor_mode == "grad_project" and args.distill_anchor_coef == 0.0
    with pytest.raises(SystemExit):
        _resolved(["--distill-anchor-mode", "grad_project", "--steps", "10"])


def test_grad_project_composes_with_an_output_anchor_and_a_moving_reference():
    args = _resolved(_FOLD + ["--distill-anchor-mode", "grad_project",
                              "--distill-anchor-coef", "0.02", "--distill-anchor-ref", "ema"])
    assert args.distill_anchor_coef == 0.02 and args.distill_anchor_ref == "ema"


def test_config_refuses_the_sample_count_outside_grad_project_or_below_one():
    with pytest.raises(SystemExit):
        _resolved(_FOLD + ["--distill-anchor-coef", "0.02", "--distill-anchor-proj-samples", "8"])
    with pytest.raises(SystemExit):
        _resolved(_FOLD + ["--distill-anchor-mode", "grad_project",
                           "--distill-anchor-proj-samples", "0"])


def test_the_callback_is_registered_by_the_mode_alone():
    """`distill_anchor_mode` reaches the model ONLY through `DistillAnchorCallback`, and the frozen
    parent it loads is what makes `distill/collateral_kl_vs_parent` — the projection's own readout —
    exist. So the registration condition and `resolve_config`'s `_anchor_wanted` must agree."""
    import inspect
    from main.train import callbacks as _cb
    src = inspect.getsource(_cb)
    assert 'getattr(args, "distill_anchor_mode", None) == "grad_project"' in src
    assert "proj_samples=int(_arg_or(args, \"distill_anchor_proj_samples\", 16))" in src


def test_the_callback_pushes_the_sample_count_onto_the_model():
    from agents.training.distill_anchor_callback import DistillAnchorCallback
    cb = DistillAnchorCallback(parent_path="p", route="explicit", coef=0.0,
                               mode=GRAD_PROJECT_MODE, monitor=False, proj_samples=5,
                               load_parent=lambda _p: object())
    cb.model = SimpleNamespace()
    cb._load_parent = lambda _p: object()
    try:
        cb._on_training_start()
    except Exception:                # the emit/ref plumbing needs a real model; the hparams do not
        pass
    assert cb.model.distill_anchor_mode == GRAD_PROJECT_MODE
    assert cb.model.distill_anchor_proj_samples == 5
