"""gen3_parity_perturb_ladder_v1 — T2's parity gate on FLAT weights: informative, never vacuous, never
a crash loop.

THE DEFECT (2026-09-30, cutover prep, ``~/gen3ai_archive/cutover_prep/fresh3``): a small CPU run's
win-prob critic COLLAPSED within minutes (the trainee lost ~97% of its games; the win head saturated
at logit ≈ −8…−10), so V varied 6e-6 across the fixture rows. The gate's single seeded perturbation
(scale 0.05) only lifted that to 9.6e-5 — under the 1e-4 bar — so the slot was REFUSED as vacuous at
the trainee's load, and again at every restart's startup (the resume loads the same weights).

THE FIX: a vacuous comparison climbs a DECLARED ladder of (scale, seed) rungs — scale capped at 0.1,
where the GPU's compiled-vs-eager noise sits 200x under the bars, more seeds instead of larger
scales (in place, restored bit-exactly) — and is judged at the FIRST informative rung; none ⇒ it
still refuses. Every verdict records its PATH. NaN / Inf weights are refused before any slot is
touched.

The COLLAPSED fixture reproduces fresh3's shape on a fresh production policy (seed 0): the win
head's final bias −9 and its weight ×0.01 — vacuous on V at the real weights and the first rung; on
the gate's 8/7/2/1-row fills (CPU) the 8-row bucket turns informative at rung (0.05, 3) and the 2-row
bucket at (0.1, 0) (measured 2026-10-03 after gen3_nonformula_damage_v1 moved the fixture rows' op
features; before it, (0.1, 3)) — WHICH rung is first is a property of the fixture's features, so no
test below depends on it. The DEEP fixture (bias −12) is beyond every rung and
must be refused. Each test names what its revert does.
"""
from __future__ import annotations

import copy

import pytest
import torch

from agents.inference.service import (
    InferenceService, NonFiniteWeights, ParityFailure, ServiceSpec, SlotGroupSpec, VacuousParity,
    policy_reference,
)
from agents.inference.service.parity import fixture_rows, judge
from agents.model.parity_probe import PERTURB_LADDER, PERTURB_MAX_SCALE, PERTURB_SCALE, PERTURB_SEED
from utils.torch_state_guard import torch_globals


#: The path label of the ladder's FIRST rung — the fresh-weights scale at the base seed.
_FIRST_RUNG = f"perturbed seed={PERTURB_SEED} scale={PERTURB_SCALE:g}"


@pytest.fixture(scope="module")
def fresh_policy():
    """An UNPERTURBED fresh production policy — exactly what a fresh launch serves."""
    from main.fresh_checkpoint import build_fresh_model
    with torch_globals(num_threads=2):
        yield build_fresh_model(3)[0].policy.eval()


def _collapse(policy, bias: float, gain: float = 0.01):
    """A COPY of ``policy`` whose win-prob critic is saturated (fresh3's shape): V ≈ sigmoid(bias)
    on every row, so V's across-row spread sits far under the 1e-4 bar."""
    p = copy.deepcopy(policy)
    head = p.features_extractor.win_head.net[3]
    with torch.no_grad():
        head.bias.fill_(bias)
        head.weight.mul_(gain)
    return p


@pytest.fixture(scope="module")
def collapsed_policy():
    """A COLLAPSED critic the capped ladder CAN judge: fresh seed 0, win logit ≈ −9 on every row.
    Measured (CPU, the gate's 8/7/2/1-row fills): vacuous on V through rung (0.1, 2), informative
    at (0.1, 3) — so it exercises the climb past the first scale and past the first seeds."""
    from main.fresh_checkpoint import build_fresh_model
    with torch_globals(num_threads=2):
        yield _collapse(build_fresh_model(0)[0].policy.eval(), -9.0)


@pytest.fixture(scope="module")
def deep_collapsed_policy(fresh_policy):
    """A critic saturated BEYOND what any rung at scale <= `PERTURB_MAX_SCALE` can move (win logit
    ≈ −12): every rung is vacuous on V, so the gate must REFUSE it."""
    return _collapse(fresh_policy, -12.0)


def _service(template, *, n_slots=1, buckets=(2, 8), **kw):
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", n_slots, template),), device="cpu",
                       backend="eager", buckets=buckets, **kw)
    return InferenceService(spec).startup()


#: How much the planted value-coupling defect AMPLIFIES each row's logit deviation from the batch mean.
#: Chosen so the catch is DETERMINISTIC at ANY informative rung (standing rule 8), not at whichever rung
#: the fixture happens to reach: a rung is informative iff V's across-row spread s > the 1e-4 bar, and
#: some row deviates from the mean by >= s/2, so the defect moves that row's V by >= (C - 1)·s/2 (V is
#: linear in the logit on a saturated head). C = 4 ⇒ >= 1.5·s > 1.5 × the bar. (C = 2 — a DOUBLING —
#: moves it by only >= s/2, which an informative rung can pass: on 2026-10-03 the first informative
#: rungs read 5.8e-5 and 9.0e-5 < 1e-4, and the slot was SERVED. That gap is the gate's, reported as a
#: finding: "informative" (spread > bar) does not imply a coupling defect smaller than ~2x the spread
#: is visible.)
_COUPLING_GAIN = 4.0


def _value_coupling_bug(real):
    """A served path that AMPLIFIES (x `_COUPLING_GAIN`) each row's win-prob logit deviation from the
    batch mean — a batch-coupled value miscompile (a wrong reduction across rows). On a collapsed
    critic every row has the same logit, so V barely moves (invisible to the real-weights comparison);
    on ANY informative rung the rows differ and V moves by more than 1.5x the 1e-4 bar."""
    def decide(module, obs, mask):
        logp, value, greedy = real(module, obs, mask)
        lg = torch.logit(value.clamp(1e-12, 1 - 1e-12))
        mean = lg.mean()
        return logp, torch.sigmoid(mean + _COUPLING_GAIN * (lg - mean)), greedy
    return decide


def _state(policy):
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


# ---------------------------------------------------------------- the premise


def test_PREMISE_the_collapsed_critic_is_vacuous_on_V_even_at_the_first_rung(collapsed_policy):
    """If this flips, the collapsed fixture no longer exercises the ladder — rebuild it, do not delete
    the tests below."""
    from agents.model.parity_probe import perturbed_parameters, spread
    obs, mask = fixture_rows(collapsed_policy.observation_space["observation"].shape[0], 8)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    _, v = policy_reference(collapsed_policy, o, m)
    assert spread("value", v) < 1e-4
    with perturbed_parameters(collapsed_policy, scale=PERTURB_SCALE):
        _, v1 = policy_reference(collapsed_policy, o, m)
    assert spread("value", v1) < 1e-4, "the first rung alone must NOT be enough for this fixture"
    assert PERTURB_LADDER[0] == (PERTURB_SCALE, 0), "the ladder's first rung is the fresh-weights one"
    assert max(sc for sc, _ in PERTURB_LADDER) <= PERTURB_MAX_SCALE


# ---------------------------------------------------------------- the fix


def test_a_FRESH_production_slot_passes_startup_through_the_first_rung_and_records_the_path(
        fresh_policy):
    """What a fresh `--arch production` launch does at T2 startup. Revert the perturbed path ⇒
    `VacuousParity` (the fresh pointer head's log-probs are constant per row)."""
    before = _state(fresh_policy)
    svc = _service(fresh_policy)
    paths = svc.stats()["parity_paths"]
    assert "real" not in paths, f"a fresh policy is vacuous on its own — no plain real verdict: {paths}"
    assert paths.get(_FIRST_RUNG, 0) > 0, paths
    assert paths.get("real (vacuity waived)", 0) == paths[_FIRST_RUNG], paths
    after = svc.groups[0].policies[0].state_dict()
    assert all(torch.equal(before[k], after[k].cpu()) for k in before), "weights not restored"


def test_a_COLLAPSED_critic_passes_startup_AND_load_by_climbing_the_ladder(fresh_policy,
                                                                          collapsed_policy):
    """fresh3's refusal, fixed. Revert the ladder to the single 0.05 rung ⇒ `VacuousParity` here —
    at startup AND at the load that killed fresh3 (a collapsed trainee copied into a live slot)."""
    before = _state(collapsed_policy)
    svc = _service(collapsed_policy)
    rungs = [r.path for r in svc.startup_reports if r.path.startswith("perturbed")]
    assert rungs and all(p != _FIRST_RUNG for p in rungs), \
        f"the collapsed critic must be judged ABOVE the first rung: {rungs}"
    after = svc.groups[0].policies[0].state_dict()
    assert all(torch.equal(before[k], after[k].cpu()) for k in before), "weights not restored"

    live = _service(fresh_policy)                  # a healthy service; then the collapsed trainee
    before_paths = dict(live.stats()["parity_paths"])
    rep = live.load(0, collapsed_policy, "trainee:v2")
    assert rep.path == "real (vacuity waived)" and live.state == "FROZEN"
    grew = {k for k, v in live.stats()["parity_paths"].items() if v > before_paths.get(k, 0)}
    rungs = {k for k in grew if k.startswith("perturbed")}
    assert rungs and _FIRST_RUNG not in rungs, f"the load must climb above the first rung: {grew}"


def test_the_vacuity_guard_still_REFUSES_when_the_perturbed_path_is_disabled(fresh_policy):
    """An EMPTY ladder disables the perturbed path: a flat slot is REFUSED, never passed. Revert the
    guard (let the real-weights verdict pass vacuously) ⇒ this startup succeeds."""
    with pytest.raises(VacuousParity, match=r"EMPTY — the perturbed path is disabled"):
        _service(fresh_policy, perturb_ladder=())


def test_a_DEEPLY_saturated_critic_is_REFUSED_by_the_full_ladder_never_passed(deep_collapsed_policy):
    """The capped ladder's honest limit: a critic no rung at scale <= 0.1 can move is REFUSED (a
    `FATAL_CONFIG` exit for the trainer), with every rung's spread named — never passed."""
    with pytest.raises(VacuousParity, match=r"every rung.*scale 0\.1 seed\+7: value spread"):
        _service(deep_collapsed_policy)


def test_a_ladder_whose_every_rung_is_vacuous_REFUSES_naming_every_rung(collapsed_policy):
    """Fail-closed at the top of the ladder: a declared ladder that never moves V (here one rung at
    the fresh scale) refuses, naming each rung's spread."""
    with pytest.raises(VacuousParity, match=r"every rung.*real weights: .*value spread.*scale 0\.05 seed\+0: "
                                            r"value spread"):
        _service(collapsed_policy, perturb_ladder=((PERTURB_SCALE, 0),))


def test_a_ladder_is_validated_at_declaration():
    for bad in (((0.1, 0), (0.05, 0)), ((0.0, 0),), ((-0.1, 0),), ((float("inf"), 0),),
                ((0.1, 0), (0.1, 0)), ((0.1, -1),), ((PERTURB_MAX_SCALE * 2, 0),), (0.1,)):
        with pytest.raises(ValueError, match="perturb_ladder"):
            ServiceSpec(groups=(SlotGroupSpec("g", 1, object()),), device="cpu", backend="eager",
                        perturb_ladder=bad).validate()


# ---------------------------------------------------------------- the teeth


def test_TEETH_a_value_miscompile_invisible_on_the_collapsed_critic_is_CAUGHT_on_a_higher_rung(
        monkeypatch, collapsed_policy):
    """The defect a flat critic hides: the served path couples the rows' value logits. On the REAL
    collapsed weights it moves V by < 1e-4 (the waived real-weights check passes it — asserted); the
    ladder's informative rung REFUSES it with a `ParityFailure` that is NOT a vacuity refusal.
    Revert the ladder ⇒ `VacuousParity` instead (the defect is never judged); revert the perturbed
    path entirely and waive the guard ⇒ the defect passes."""
    import agents.inference.service.engine as engine_mod
    bug = _value_coupling_bug(engine_mod.decide)

    # (a) invisible on the real weights: the vacuity-waived comparison PASSES the buggy output.
    obs, mask = fixture_rows(collapsed_policy.observation_space["observation"].shape[0], 8)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    from agents.inference.service.decision import DecisionModule
    served = bug(DecisionModule(collapsed_policy).eval(), o, m)
    judge(where="real, waived", policy=collapsed_policy, obs=o, mask=m, served=served,
          allow_vacuous=True)

    # (b) the service's gate catches it on the ladder.
    monkeypatch.setattr(engine_mod, "decide", bug)
    with pytest.raises(ParityFailure, match=r"perturbed seed=\d+ scale=") as ei:
        _service(collapsed_policy)
    assert not isinstance(ei.value, VacuousParity), f"judged vacuous, not caught: {ei.value}"
    assert f"[{_FIRST_RUNG}]" not in str(ei.value), "caught above the first rung"


# ---------------------------------------------------------------- broken weights


def test_NaN_weights_are_REFUSED_at_load_before_the_slot_is_touched(fresh_policy):
    svc = _service(fresh_policy)
    held = _state(svc.groups[0].policies[0])
    broken = copy.deepcopy(fresh_policy)
    with torch.no_grad():
        broken.features_extractor.win_head.net[1].weight[0, 0] = float("nan")
    with pytest.raises(NonFiniteWeights, match="NaN / Inf"):
        svc.load(0, broken, "trainee:nan")
    now = svc.groups[0].policies[0].state_dict()
    assert all(torch.equal(held[k], now[k]) for k in held), "a refused load touched the slot"
    assert svc.state == "FROZEN", "broken INPUT is refused, the service is not poisoned"


def test_NaN_weights_are_REFUSED_as_a_startup_template(fresh_policy):
    broken = copy.deepcopy(fresh_policy)
    with torch.no_grad():
        broken.features_extractor.win_head.net[3].bias.fill_(float("inf"))
    with pytest.raises(NonFiniteWeights, match="template"):
        _service(broken)
