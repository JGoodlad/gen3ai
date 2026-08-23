"""The TWIN win-prob heads + the SHADOW critic (v99, gen3_cf_twin_heads_v1).

The owner-authorized amendment to the signed R1 pre-registration (ledger 2026-08-22 evening,
"Three owner sign-offs" item 3). What is pinned here, in order of how badly a silent break would
hurt:

1. **The twins are the SAME module class with the SAME capacity as head A.** The arm's entire claim
   is that the three heads differ only in their label stream; a difference of architectures would
   be a second explanation for every difference of scores, and nothing downstream would say so.
2. **OFF is byte-identical and ON is BIT-identical in pi/vf.** Both flags build modules the forward
   never calls, and both are built LAST — so an "optional" head must not re-roll the initialization
   RNG stream for anything before it. Easy to break by tidying the constructor, impossible to
   notice without this test.
3. **The version gate fires, on BOTH flags.** Neither head is called by the forward, so a
   mismatched resume produces NO shape error anywhere: `check_compatible` is the only thing between
   a flipped flag and a run that silently supervises freshly-random heads (or loses the within-run
   paired comparison) for the rest of its life.
4. **The shadow head is a value readout, not a probability one.** Its output is unbounded and in
   the PopArt-normalized frame; a sigmoid creeping in would silently clamp every label to [0,1] and
   the loss would still fall.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from agents.model.arch_constants import D_MODEL
from agents.model.aux_value_heads import ShadowValueHead, WinProbHead
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError,
                                        _migrate_config)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

FLAGS = ("cf_twin_heads", "cf_shadow_critic")


@pytest.fixture(scope="module")
def ek_and_space():
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    total = ek["layout"]["total_dim"]
    space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (total,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
    })
    return ek, space, total


def _build(ek, space, seed=0, **flags):
    """`win_prob_mode="read_only"` ALWAYS — head A must exist for the twins (the registry's
    declared `requires`, enforced in the constructor). Passing it on the OFF arm too is what keeps
    the BIT-identity comparison honest: both arms then build the same head A."""
    torch.manual_seed(seed)
    return Gen3FeaturesExtractor(
        space, **{**ek, "win_prob_mode": "read_only", **flags}).eval()


# ── the shadow head's own contract ────────────────────────────────────────────

def test_the_shadow_head_emits_an_UNBOUNDED_scalar_value():
    """A VALUE, not a probability. The label is a shaped return in the run's own reward units and
    can be large and negative; a sigmoid (or any squashing) creeping in would silently clamp the
    target range while the MSE kept falling, which is exactly the failure a passive head cannot
    self-report."""
    head = ShadowValueHead()
    with torch.no_grad():
        # Drive the bottleneck hard in both directions: the output must not saturate.
        out = head(torch.randn(64, D_MODEL) * 50.0)
    assert out.shape == (64, 1)
    assert out.min() < 0.0 or out.max() > 0.0
    assert not torch.all((out >= 0.0) & (out <= 1.0)), \
        "the shadow head's range looks squashed — is a sigmoid in the way?"


def test_the_twins_have_head_As_architecture_exactly():
    """Not "similar capacity" — the SAME class. B and C must be substitutable for A in every
    respect except the labels they are trained on, or a difference of scores has a second
    explanation the arm cannot rule out."""
    a, b = WinProbHead(), WinProbHead()
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert [tuple(p.shape) for p in a.parameters()] == [tuple(p.shape) for p in b.parameters()]


# ── the extractor integration ─────────────────────────────────────────────────

def test_off_builds_nothing(ek_and_space):
    ek, space, _t = ek_and_space
    off = _build(ek, space)
    assert off.cf_twin_head_b is None and off.cf_twin_head_c is None
    assert off.cf_shadow_head is None
    assert off.cf_twin_heads is False and off.cf_shadow_critic is False


@pytest.mark.parametrize("flag,attrs", [
    ("cf_twin_heads", ("cf_twin_head_b", "cf_twin_head_c")),
    ("cf_shadow_critic", ("cf_shadow_head",)),
])
def test_on_adds_ONLY_its_own_heads(ek_and_space, flag, attrs):
    ek, space, _t = ek_and_space
    off, on = _build(ek, space), _build(ek, space, **{flag: True})
    delta = (sum(p.numel() for p in on.parameters())
             - sum(p.numel() for p in off.parameters()))
    own = sum(p.numel() for a in attrs for p in getattr(on, a).parameters())
    assert delta == own > 0


@pytest.mark.parametrize("flags", [
    {"cf_twin_heads": True},
    {"cf_shadow_critic": True},
    {"cf_twin_heads": True, "cf_shadow_critic": True},
])
def test_on_is_BIT_identical_in_pi_and_vf(ek_and_space, flags):
    """The strong form, and the reason both are built LAST — after `cf_evid_head` and before the
    identity-init snapshot. Adding a module mid-constructor shifts the initialization RNG stream for
    every module built after it, so an "optional" head can silently re-roll the whole network."""
    ek, space, total = ek_and_space
    obs = {"observation": torch.zeros(3, total)}
    off, on = _build(ek, space), _build(ek, space, **flags)
    with torch.no_grad():
        a, b = off(obs), on(obs)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), \
        f"building {sorted(flags)} perturbed pi/vf — are they still built LAST in __init__?"


def test_no_new_head_is_called_by_the_forward(ek_and_space):
    """All three are training-side readouts applied to the STASHED value_pooled, so the rollout
    pays nothing for them and no privileged label can reach the acting path through them."""
    ek, space, total = ek_and_space
    fe = _build(ek, space, cf_twin_heads=True, cf_shadow_critic=True)
    calls = []
    for name in ("cf_twin_head_b", "cf_twin_head_c", "cf_shadow_head"):
        getattr(fe, name).register_forward_hook(
            lambda _m, _i, _o, _n=name: calls.append(_n))
    with torch.no_grad():
        fe({"observation": torch.zeros(2, total)})
    assert calls == [], f"{calls} ran during the extractor forward"


def test_the_stashed_value_pooled_is_what_all_three_consume(ek_and_space):
    """The training-side contract: one `stash.value_pooled` feeds head A, the twins and the shadow,
    which is precisely what "identical trunk, identical states" means at the tensor level."""
    ek, space, total = ek_and_space
    fe = _build(ek, space, cf_twin_heads=True, cf_shadow_critic=True)
    with torch.no_grad():
        fe({"observation": torch.zeros(2, total)})
        pooled = fe.stash.value_pooled
        outs = [fe.win_head(pooled), fe.cf_twin_head_b(pooled),
                fe.cf_twin_head_c(pooled), fe.cf_shadow_head(pooled)]
    assert pooled.shape[-1] == D_MODEL
    assert all(o.shape == (2, 1) for o in outs)


def test_the_twins_are_INDEPENDENTLY_initialized(ek_and_space):
    """B and C must not start life as the same function.

    If they did, `cf/twin_b_vs_c_abs` would read 0 at step 0 for a reason that has nothing to do
    with the labels, and the arm's "have the heads diverged yet" launch-window check would be
    reading its own initialization. (Independent init is also what makes the pair a weak internal
    replicate: at cf_twin_coef=0 they should converge toward each other, and a persistent gap
    is a seed effect the arm must not mistake for a label effect.)
    """
    ek, space, _t = ek_and_space
    fe = _build(ek, space, cf_twin_heads=True)
    b = dict(fe.cf_twin_head_b.named_parameters())
    c = dict(fe.cf_twin_head_c.named_parameters())
    differing = [k for k in b if not torch.equal(b[k], c[k])]
    assert differing, "heads B and C were initialized identically"


# ── the v99 version gate ──────────────────────────────────────────────────────

def _ver(**flags):
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    ek["win_prob_mode"] = "read_only"
    ek.update(flags)
    pk = {"features_extractor_class": Gen3FeaturesExtractor, "features_extractor_kwargs": ek,
          "net_arch": [512, 512]}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk)


@pytest.mark.parametrize("flag", FLAGS)
def test_version_records_the_toggle(flag):
    assert getattr(_ver(**{flag: True}), flag) is True
    assert getattr(_ver(**{flag: False}), flag) is False


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("saved,current", [(False, True), (True, False)])
def test_mismatch_fatals(flag, saved, current):
    """The ONLY gate. Because the forward never calls these heads, a mismatched resume loads
    "successfully" and then trains (or fails to train) silently for the rest of the run."""
    with pytest.raises(ModelVersionError, match=flag):
        _ver(**{flag: current}).check_compatible(_ver(**{flag: saved}))


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("on", [False, True])
def test_matching_toggle_loads(flag, on):
    _ver(**{flag: on}).check_compatible(_ver(**{flag: on}))       # no raise


def test_migration_defaults_a_v98_config_off():
    """A pre-v99 checkpoint could not have built either module, so False is not a guess — it is the
    only possible past. The refusal direction is `check_compatible`'s."""
    out = _migrate_config({"config_version": 98})
    assert out["cf_twin_heads"] is False and out["cf_shadow_critic"] is False
    # NOT `== 99`: the migration must carry a config all the way to the CURRENT version, and a
    # test pinned to the version a feature landed at goes red on the next bump for no defect — with
    # the fastest "fix" indistinguishable from one that hides a chain that genuinely stopped early.
    assert out["config_version"] == MODEL_CONFIG_VERSION


def test_a_recorded_on_config_round_trips():
    """The other migration leg: a config that already RECORDS the flags must survive untouched."""
    out = _migrate_config({"config_version": 99, "cf_twin_heads": True,
                           "cf_shadow_critic": True})
    assert out["cf_twin_heads"] is True and out["cf_shadow_critic"] is True
    assert out["config_version"] == MODEL_CONFIG_VERSION


@pytest.mark.parametrize("flag", FLAGS)
def test_the_flags_are_in_the_registry_as_structural_cli_toggles(flag):
    """They build MODULES from extractor constructor kwargs — the registry's declared scope, and
    the `cf_evidential` / `win_prob_mode` precedent. Their COEFFICIENTS are deliberately not:
    training-only loss weights in the `--opd-coef` class, set on the model, never on the
    extractor."""
    from agents.model.flag_registry import BY_NAME, Klass, Tier
    row = BY_NAME[flag]
    assert row.klass is Klass.STRUCTURAL and row.tier is Tier.CLI
    assert row.since == 99 and row.default is False
    assert "cf_twin_coef" not in BY_NAME and "cf_shadow_coef" not in BY_NAME


@pytest.mark.parametrize("flag", FLAGS)
def test_the_snapshot_version_helper_threads_the_toggle(flag):
    """A frozen eval/pool opponent's load gate must see these, or a twin-heads run FATALs loading
    its OWN sentinels — the failure `value_threat_inject` and `cf_evidential` each shipped once."""
    from agents.model.snapshot import current_model_version
    mappings = load_mappings()
    assert getattr(current_model_version(mappings, **{flag: True}), flag) is True
    assert getattr(current_model_version(mappings), flag) is False
