"""K9(a) THE LEARNER GOLDEN on X5's fixed-mass hypothesis tokens (X5 U6; design §6, §8.3) — the production
surface since the X5 VERSION BREAK (config v144), whose entry this golden's default slot holds VERBATIM.

`learner_golden_test` holds the generic properties (reproduces, has teeth, never records); this file proves
the X5 update is pinned AND that the pin is about the right thing — correct, not merely new:

* REPRODUCES: one eager fp32 ``train()`` of the seeded learner (name-keyed perturbation) on the committed
  buffer reproduces every recorded hash — initial (whole and per group), post-update (whole and per group)
  — and every pinned loss EXACTLY; with K9(b) ON (``fatal``) it still does, and the behaviour probe passes
  with its rule-8 excluded share under the ceiling.
* NOT VACUOUS: the X5 loss keys are logged and nonzero, α / β's are absent, every X5 group moved, and the
  deleted blob path's groups (the retired α / β heads, BeliefSlots) are absent.
* fp64: the fp32 micro-step agrees with the same arithmetic at fp64 within the declared tolerances,
  and the recorded fp64 reference reproduces; an fp32-only defect FAILS it.
* INDEPENDENT fp64 numpy references of the construction on the golden rows (Σπ = k, the root-find,
  OTHER's masked bias iff its tail is empty).
* TEETH: a bias in the hypothesis builder's construction, the log-π key bias's sign, OTHER's column, the
  flat pointer's mask — each FAILS the golden naming X5 groups, never the INIT; a plant on the RETIRED α
  head (the deleted blob path's intent readout) leaves it byte-identical — the head is never called.
* ISOLATION at golden level: with the set BCE's logits detached δ_θ does not move at all (it learns
  from the presence BCE alone, M10); the B ride-along on the flat pointer is bit-identical to learning.

Each check FAILS on revert of what it pins (the planted variants above are the reverts, run).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch as th

from agents.training import learner_golden as L
from agents.training import learner_golden_fp64 as F64

#: X5's own parameter groups (depth-2 names, `group_sha256`).
X5_GROUPS = ("features_extractor.hypothesis_builder", "features_extractor.flat_intent_head")
#: The deleted blob path's groups: never in the state_dict (α / β retired by the policy, BeliefSlots discarded).
BLOB_ONLY_GROUPS = ("features_extractor.alpha_head", "features_extractor.beta_head",
                    "features_extractor.belief_slots")
DELTA_PREFIX = "features_extractor.hypothesis_builder.delta_"


def _delta(model):
    return {n: p.detach().clone() for n, p in model.policy.named_parameters() if n.startswith(DELTA_PREFIX)}


@pytest.fixture(scope="module")
def golden():
    """The golden, with K9(b) ON: (diffs, fingerprint, behaviour read, δ_θ before / after, model)."""
    entry = L.entry()
    model = L.build_learner()
    model.behaviour_check = "fatal"
    d0 = _delta(model)
    diffs, now = L.check(model)
    beh = {k[len("behaviour/"):]: float(v) for k, v in model.logger.name_to_value.items()
           if k.startswith("behaviour/")}
    return dict(entry=entry, diffs=diffs, now=now, beh=beh, d0=d0, d1=_delta(model))


def test_the_fixed_mass_update_reproduces_its_golden_exactly_with_K9b_on(golden):
    assert not golden["diffs"], (
        "the fixed_mass learner golden MOVED:\n  " + "\n  ".join(golden["diffs"])
        + "\nIf intended: python -m agents.training.learner_golden record --reason \"...\"")


def test_K9b_passes_on_the_fixed_mass_entry_with_the_excluded_share_under_its_ceiling(golden):
    from agents.training.rust_rollout import consistency as K
    beh = golden["beh"]
    assert beh["rows_current"] == min(L.GOLDEN_OVERRIDES["batch_size"], L.N_STEPS * L.N_ENVS)
    assert beh["max_abs_dlogp_judged"] < 1e-5, beh       # measured 4.8e-7 (T2 eager vs the learner)
    assert 0.0 <= beh["excluded_frac"] < K.FP32_EXCLUDED_CEILING, beh
    rec = golden["entry"]["behaviour"]
    assert rec["excluded_frac"] == beh["excluded_frac"] and rec["rows_judged"] == beh["rows_judged"]


def test_the_fixed_mass_entry_is_not_vacuous(golden):
    now, entry = golden["now"], golden["entry"]
    for k in ("opp_intent/flat_loss", "belief/set_aux_loss", "win_prob/loss", "belief/move_loss",
              "train/policy_gradient_loss", "train/entropy_loss"):
        assert k in now["losses"] and now["losses"][k] != 0.0, k
    assert not any(k in now["losses"] for k in ("opp_intent/alpha_loss", "opp_intent/beta_loss"))
    for g in X5_GROUPS:
        assert entry["init_group_sha256"][g] != entry["group_sha256"][g], f"{g} did not move in the update"
    assert not any(g in entry["group_sha256"] for g in BLOB_ONLY_GROUPS)
    # δ_θ (the presence BCE's only parameters) moved in the real update
    assert any(not th.equal(golden["d0"][n], golden["d1"][n]) for n in golden["d0"])


def test_the_buffer_covers_every_x5_case_design_6_4():
    """design §6.4: the buffer must hold >= `MIN_CASE_ROWS` rows of every X5 case — OTHER_species live and
    dead, OTHER_move live and dead (an active with four revealed moves), a hypothesis — or the golden pins
    a branch it never ran. The counts are the recorded ones."""
    cov = L.coverage()
    assert cov == L.entry()["coverage"]
    thin = {k: v for k, v in cov.items() if k != "near_tie_rows" and v < L.MIN_CASE_ROWS}
    assert not thin, thin


# ------------------------------------------------------------------------------------------- fp64
def test_the_fp32_micro_step_agrees_with_fp64_within_the_declared_tolerances_and_reproduces():
    entry = L.entry()
    ref = F64.reference(L.build_learner(), L.BUFFER_PATH)
    assert F64.violations(ref) == [], F64.violations(ref)
    assert ref["grads_absent"] == [] and len(ref["grads"]) == len(F64.KEY_GRADS)
    rec = entry["fp64_reference"]
    assert rec["tolerances"] == ref["tolerances"], "the tolerances changed — re-record deliberately"
    # one torch build, one thread: both precisions reproduce exactly (a rounding-level wobble is a change)
    assert ref["terms_fp32"] == rec["terms_fp32"] and ref["terms_fp64"] == rec["terms_fp64"]
    assert ref["rows_excluded"] == rec["rows_excluded"] and ref["rows_judged"] >= 0.9 * L.N_STEPS * L.N_ENVS
    for k in ("species_belief_set", "opp_intent", "win_prob", "policy", "entropy"):
        assert k in ref["terms_fp64"], k


def test_the_fp64_reference_has_teeth_an_fp32_only_defect_fails_it(monkeypatch):
    """A defect that exists only at fp32 (a 1e-3 shift of the win-prob logits when they are float32) moves
    the fp32 term away from fp64 by far more than rounding."""
    from agents.training.instrumented_ppo import micro_step as MS
    orig = MS.win_prob_terms

    def fp32_only(logits, target, mask, margin):
        if logits is not None and logits.dtype == th.float32:
            logits = logits + 1e-3
        return orig(logits, target, mask, margin)

    monkeypatch.setattr(MS, "win_prob_terms", fp32_only)
    bad = F64.violations(F64.reference(L.build_learner(), L.BUFFER_PATH))
    assert any(v.startswith("term win_prob") for v in bad), bad


def test_independent_fp64_numpy_references_of_the_construction_on_the_golden_rows():
    """§6.3 on the golden's rows at the seeded init (δ_θ perturbed, so π is not the prior): Σπ = k for the
    species group and Σπ = 4 − r for the active's moves; π equals a DIRECT fp64 numpy root-find of
    Σσ(a + τ) = k; OTHER's log-mass is the masked −1e9 iff its tail is empty."""
    from agents.model.hypothesis_set import MASKED_LOG_PRESENCE
    model = L.build_learner()
    data = L.load_buffer_into(model, L.BUFFER_PATH)
    n = L.N_STEPS * L.N_ENVS
    obs = {k[4:]: th.as_tensor(v.reshape(n, *v.shape[2:])) for k, v in data.items() if k.startswith("obs:")}
    fe = model.policy.features_extractor
    with th.no_grad(), L._one_thread():
        fe(obs)
    hs = fe.last_hypothesis
    for pres in (hs.species, hs.moves.presence):                       # k = 6 − r, and k_m = 4 − r
        pi = pres.pi.double().numpy()
        live = pres.live.numpy()
        k = pres.k.numpy().astype(np.float64)
        assert live.sum() >= 2
        assert np.abs(pi.sum(-1)[live] - k[live]).max() < 1e-4        # fp32 construction, 64 iterations
        a = pres.logits.detach().double().numpy()
        cand = pres.cand.numpy()
        for b in np.flatnonzero(live):                                 # a direct fp64 bisection, numpy only
            x = a[b][cand[b]]
            lo, hi = -60.0, 60.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if (1.0 / (1.0 + np.exp(-(x + mid)))).sum() > k[b] else (mid, hi)
            ref = 1.0 / (1.0 + np.exp(-(x + 0.5 * (lo + hi))))
            assert np.abs(ref - pi[b][cand[b]]).max() < 1e-5, b
    om = hs.other_log_mass.numpy()
    olive = hs.other_live.numpy()
    assert (om == MASKED_LOG_PRESENCE).tolist() == (~olive).tolist()
    assert 2 <= olive.sum() < n, "the buffer must hold both a live and a dead OTHER_species"
    mo = hs.moves
    assert (mo.other_log_mass.numpy() == MASKED_LOG_PRESENCE).tolist() == (~mo.other_live.numpy()).tolist()


# ------------------------------------------------------------------------------------------ teeth
def _plant_tau_bias(monkeypatch):
    """A bias in the hypothesis builder: the fixed-size construction's τ shifted (Σπ ≠ k)."""
    from agents.model import hypothesis_set as HS
    orig = HS.fixed_size_tau
    monkeypatch.setattr(HS, "fixed_size_tau", lambda *a, **k: orig(*a, **k) + 1e-2)


def _plant_logpi_sign(monkeypatch):
    """The log-π KEY bias with its sign flipped on every live key."""
    from agents.model import extractor_forward as EF
    orig = EF.key_log_presence

    def flipped(*a, **k):
        lp = orig(*a, **k)
        return th.where(lp > -1e8, -lp, lp)

    monkeypatch.setattr(EF, "key_log_presence", flipped)


def _plant_other_column(monkeypatch):
    """OTHER's cells read at the NEXT opponent column instead of OTHER's hidden slot."""
    from agents.model import extractor_forward as EF
    orig = EF.other_column
    monkeypatch.setattr(EF, "other_column", lambda cells, col, axis: orig(cells, (col + 1) % 6, axis))


def _plant_flat_mask(monkeypatch):
    """The flat pointer's mask with OTHER_move's column masked off on every row."""
    from agents.model import extractor_forward as EF
    from agents.model.flat_intent import other_move_col
    orig = EF.flat_candidates

    def masked(*a, **k):
        tok, fi = orig(*a, **k)
        live = fi.live.clone()
        live[:, other_move_col(fi.k)] = False
        return tok, dataclasses.replace(fi, live=live)

    monkeypatch.setattr(EF, "flat_candidates", masked)


PLANTS = {"tau_bias": _plant_tau_bias, "logpi_sign": _plant_logpi_sign, "other_column": _plant_other_column,
          "flat_mask": _plant_flat_mask}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_TEETH_each_planted_x5_perturbation_fails_the_golden(plant, monkeypatch):
    PLANTS[plant](monkeypatch)
    diffs, _now = L.check()
    assert diffs, f"the planted {plant} perturbation did not move the golden"
    assert not any(d.startswith("the INIT moved") for d in diffs), "a forward plant must not move the init"
    assert any(d.startswith("post_params_sha256") for d in diffs), diffs
    moved = next(d for d in diffs if d.startswith("parameter groups whose post-update bytes moved"))
    assert any(g in moved for g in X5_GROUPS), moved


# -------------------------------------------------------------------------------------- isolation
def test_delta_theta_learns_from_the_set_BCE_alone_at_golden_level(monkeypatch):
    """M10 / U2's isolation, through the REAL update: detach the logits every set BCE reads (the presence
    BCE and BeliefHead's) and δ_θ's bytes do not move at all — no policy, critic, intent or other belief
    term reaches it. (Unplanted, δ_θ moves: `test_the_fixed_mass_entry_is_not_vacuous`.)"""
    from agents.model import hypothesis_set as HS
    orig = HS.set_bce
    monkeypatch.setattr(HS, "set_bce", lambda logits, *a, **k: orig(logits.detach(), *a, **k))
    model = L.build_learner()
    d0 = _delta(model)
    L.compute(model)
    d1 = _delta(model)
    assert d0 and all(th.equal(d0[n], d1[n]) for n in d0), [n for n in d0 if not th.equal(d0[n], d1[n])]


def test_the_B_ride_along_on_the_flat_pointer_is_bit_identical_to_learning_at_golden_level():
    """U4's proof on the golden: production + B (`--ridealong-opp 2`) — every recorded group's initial and
    post-update bytes and every pinned loss equal the entry's; B's own groups and loss are the only additions,
    and B trained."""
    from main.train.production_args import production_args
    entry = L.entry()
    args = production_args()
    args.ridealong_opp = 2
    model = L.build_learner(args=args)
    assert model.policy.ridealong is not None and model.policy.ridealong.opp is not None
    now = L.compute(model)
    extra = sorted(set(now["group_sha256"]) - set(entry["group_sha256"]))
    assert extra and all(g.startswith("ridealong") for g in extra), extra
    for key in ("init_group_sha256", "group_sha256"):
        bad = [g for g in entry[key] if now[key].get(g) != entry[key][g]]
        assert bad == [], (key, bad)
    for k, v in entry["losses"].items():
        assert now["losses"].get(k) == v, k
    assert all(k.startswith("ridealong/") for k in set(now["losses"]) - set(entry["losses"]))
    assert any(now["init_group_sha256"][g] != now["group_sha256"][g] for g in extra), "B did not train"
