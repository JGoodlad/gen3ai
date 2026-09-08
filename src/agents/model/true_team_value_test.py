"""`TrueTeamValueReadout` — the PRIVILEGED critic channel's model half (`gen3_value_true_team_v1`).

The load-bearing claim of the whole arm is **the policy cannot see this channel**. It is asserted
here the only way that means anything: with a LARGE random weight in the route, not merely at
init — a zero-init module is trivially invisible, and "vf-only" has to hold for the trained network
the arm is actually read on.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import D_MODEL, TTV_K
from agents.model.extractor_ctx import slice_pokemon_categoricals
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.true_team_value import TrueTeamValueReadout, pokemon_id_columns
from agents.observation.constants import (
    POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.true_team import TRUE_TEAM_KEY


def _layout():
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def _block(batch=3, seed=0):
    """A block whose categorical columns hold LEGAL ids (a random float would floor to 0 and hide
    an out-of-range lookup) and whose rows are all marked present."""
    g = torch.Generator().manual_seed(seed)
    b = torch.rand(batch, TEAM_SIZE, POKEMON_FULL_DIM, generator=g)
    b[:, :, POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
    return b


def _build(seed=7, **kw):
    m = load_mappings()
    layout = Gen3ObservationEncoder(m).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(seed)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=m, **kw), layout


# ---------------------------------------------------------------- the id-column derivation
def test_id_columns_are_exactly_what_the_slicer_reads():
    """The raw half of a row must have every embedding-id column zeroed (a raw dex num must never
    reach a Linear). The column set is PROBED out of `slice_pokemon_categoricals` rather than
    restated, so this test's job is to prove the probe answers with real columns and that the
    slicer agrees when they are cleared."""
    layout = _layout()
    cols = pokemon_id_columns(layout)
    assert cols and len(cols) == len(set(cols))
    assert all(0 <= c < POKEMON_FULL_DIM for c in cols)
    # Clearing exactly those columns makes every id the slicer reports zero — which is the property
    # the readout relies on when it hands the remaining raw row to a Linear.
    blk = _block(1)
    blk[:, :, list(cols)] = 0.0
    ids = slice_pokemon_categoricals(blk, layout)
    for key, val in ids.items():
        if key.endswith("_ids"):
            assert int(val.abs().max()) == 0, f"{key} survived the id-column clear"


def test_the_species_column_is_among_them():
    """A probe that silently returned nothing would pass the test above vacuously."""
    layout = _layout()
    cols = set(pokemon_id_columns(layout))
    sp = layout['pokemon']['species']
    assert sp['offset'] + sp['layout']['species_id']['offset'] in cols


# ---------------------------------------------------------------- the readout's own contracts
def test_output_shape_and_zero_init():
    r = TrueTeamValueReadout(_layout())
    assert float(r.out_proj.weight.abs().max()) == 0.0
    assert float(r.out_proj.bias.abs().max()) == 0.0
    out = r(_block(3), _build()[0].embeddings)
    assert out.shape == (3, D_MODEL)
    assert float(out.abs().max()) == 0.0, "zero-init must add EXACTLY zero at cold start"
    assert r.last_att.shape == (3, TTV_K, TEAM_SIZE)


def test_the_pool_is_permutation_invariant_over_mons():
    """Slot order on this channel is `species num ascending` and means nothing, so an
    order-sensitive readout would be learning an artifact of the builder's sort."""
    r = TrueTeamValueReadout(_layout())
    torch.nn.init.normal_(r.out_proj.weight, std=1.0)
    emb = _build()[0].embeddings
    b = _block(2, seed=3)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        a = r(b, emb)
        c = r(b[:, perm, :], emb)
    assert torch.allclose(a, c, atol=1e-5), (a - c).abs().max()


def test_absent_rows_are_masked_out_of_the_pool():
    r = TrueTeamValueReadout(_layout())
    torch.nn.init.normal_(r.out_proj.weight, std=1.0)
    emb = _build()[0].embeddings
    b = _block(2, seed=5)
    b[:, 4:, :] = 0.0                       # two ABSENT slots (species_known 0)
    poisoned = b.clone()
    # A LEGAL-but-different id everywhere: the readout embeds every row before masking (masking is
    # in the attention, not in the lookup), so the poison stays inside the narrowest embedding axis
    # rather than testing bounds-checking, which is a different question.
    poisoned[:, 4:, :] = 3.0
    poisoned[:, 4:, POKEMON_SPECIES_KNOWN_OFFSET] = 0.0
    with torch.no_grad():
        assert torch.allclose(r(b, emb), r(poisoned, emb), atol=1e-5), (
            "a masked row changed the pool — absent slots are leaking into the summary")


def test_an_all_absent_block_pools_without_a_nan():
    """The all-zero block is what a ladder game supplies. It must produce a finite value, not a
    NaN that would silently poison every downstream critic read."""
    r = TrueTeamValueReadout(_layout())
    torch.nn.init.normal_(r.out_proj.weight, std=1.0)
    out = r(torch.zeros(2, TEAM_SIZE, POKEMON_FULL_DIM), _build()[0].embeddings)
    assert torch.isfinite(out).all()


def test_a_wrong_shaped_block_raises_naming_the_builder():
    r = TrueTeamValueReadout(_layout())
    with pytest.raises(ValueError, match="build_true_team_block"):
        r(torch.zeros(2, TEAM_SIZE + 1, POKEMON_FULL_DIM), _build()[0].embeddings)


# ---------------------------------------------------------------- THE arm-defining property
def test_the_policy_path_has_no_gradient_route_from_the_privileged_key():
    """Perturb the key: pi logits BIT-identical, vf moves. Asserted at a LARGE random weight, so
    the claim is about the trained network and not about zero-init."""
    fe, layout = _build(value_true_team=True)
    fe.eval()
    with torch.no_grad():
        fe.true_team_value.out_proj.weight.normal_(std=1.0)
        fe.true_team_value.out_proj.bias.normal_(std=1.0)
    obs_vec = torch.rand(4, layout["total_dim"])
    a = {"observation": obs_vec, TRUE_TEAM_KEY: _block(4, seed=1)}
    b = {"observation": obs_vec, TRUE_TEAM_KEY: _block(4, seed=2)}
    with torch.no_grad():
        pi_a, vf_a = fe(a)
        pi_b, vf_b = fe(b)
    assert torch.equal(pi_a, pi_b), "the PRIVILEGED key reached the POLICY features"
    assert not torch.equal(vf_a, vf_b), "the privileged key did not move the VALUE features"


def test_no_gradient_reaches_the_route_from_a_policy_only_loss():
    """The structural counterpart: backprop from pi alone must leave the route's projection with
    no gradient at all. `torch.equal` above proves the FORWARD is independent; this proves the
    BACKWARD is, which is what stops a policy loss from training the privileged summary."""
    fe, layout = _build(value_true_team=True)
    fe.train()
    obs = {"observation": torch.rand(3, layout["total_dim"]), TRUE_TEAM_KEY: _block(3)}
    pi, vf = fe(obs)
    pi.sum().backward()
    g = fe.true_team_value.out_proj.weight.grad
    assert g is None or float(g.abs().max()) == 0.0


def test_the_route_raises_rather_than_skipping_when_the_key_is_absent():
    """A silent skip is indistinguishable from a route that learned nothing — the gen-12
    dead-tail bug the `_value_pooled_routes` seam exists to prevent."""
    fe, layout = _build(value_true_team=True)
    fe.eval()
    with pytest.raises(RuntimeError, match="opp_true_team"):
        fe({"observation": torch.rand(2, layout["total_dim"])})


def test_off_builds_nothing_and_needs_no_key():
    fe, layout = _build()
    assert fe.true_team_value is None and fe.value_true_team is False
    pi, vf = fe({"observation": torch.rand(2, layout["total_dim"])})
    assert pi.shape[0] == 2 and vf.shape[0] == 2


def test_off_is_bit_identical_to_the_pre_flag_baseline_under_one_seed():
    """The append-never-insert claim: the readout is built LAST, so turning the flag OFF must
    leave every earlier module's init RNG draw — and therefore the whole forward — untouched."""
    fe_off, layout = _build(seed=11)
    fe_on, _ = _build(seed=11, value_true_team=True)
    obs_vec = torch.rand(2, layout["total_dim"])
    fe_off.eval(); fe_on.eval()
    with torch.no_grad():
        pi_off, vf_off = fe_off({"observation": obs_vec})
        pi_on, vf_on = fe_on({"observation": obs_vec,
                              TRUE_TEAM_KEY: torch.zeros(2, TEAM_SIZE, POKEMON_FULL_DIM)})
    assert torch.equal(pi_off, pi_on)
    assert torch.equal(vf_off, vf_on), "ON at zero-init is not bit-identical to OFF"
