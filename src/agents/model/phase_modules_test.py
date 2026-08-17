"""Per-phase unit tests for the decomposed Gen3FeaturesExtractor.

These exercise each phase module in isolation — something the monolithic
forward_internal made impossible. ObsUnpack slices are asserted exactly;
CLSPool and ProjectionAssembler run on a hand-built ExtractorContext with
only the fields they consume, with no need to drive a full forward pass.
"""
import dataclasses

import numpy as np
import gymnasium as gym
import torch

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    ExtractorContext,
    Embeddings,
    HiddenOppBeliefPool,
    locate_active_slot,
    turn_delta_embed_dim,
    ROLE_TOKEN_SIZE,
    D_MODEL,
    N_HISTORY_TURNS,
)
import pytest
from agents.observation.constants import POKEMON_FULL_DIM, TEAM_SIZE
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.action.constants import ACTION_SPACE_SIZE


def _make_model(**extractor_kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(layout["total_dim"],), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings, **extractor_kwargs)
    model.eval()
    return model, layout


def _zeros(layout):
    return torch.zeros(1, layout["total_dim"])


def _dummy_ctx(**overrides) -> ExtractorContext:
    """Build an ExtractorContext with every field None, overriding only what a phase reads.
    Lets CLSPool/ProjectionAssembler be tested without a real observation."""
    vals = {f.name: None for f in dataclasses.fields(ExtractorContext)}
    vals.update(overrides)
    # Derive the combined key-mask the value-CLS pool reads, exactly as ObsUnpack does, so callers
    # that set only the per-side masks still exercise the real masking path.
    if vals.get("all_fainted") is None and vals.get("fainted_mask_ours") is not None \
            and vals.get("fainted_mask_opp") is not None:
        vals["all_fainted"] = torch.cat([vals["fainted_mask_ours"], vals["fainted_mask_opp"]], dim=1)
    return ExtractorContext(**vals)


# ---------------------------------------------------------------------------
# locate_active_slot (pure helper)
# ---------------------------------------------------------------------------

def test_locate_active_slot_picks_set_flag():
    flags = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert locate_active_slot(flags).tolist() == [2, 0]


def test_locate_active_slot_falls_back_to_zero():
    flags = torch.zeros(1, 6)
    assert locate_active_slot(flags).tolist() == [0]


# ---------------------------------------------------------------------------
# ObsUnpack — exact slice extraction
# ---------------------------------------------------------------------------

def test_obsunpack_species_id_slice_is_exact():
    model, layout = _make_model()
    our_start = layout["parts"]["our_team"]["start"]
    species_info = layout["pokemon"]["species"]
    species_idx = species_info["offset"] + species_info["layout"]["species_id"]["offset"]

    obs = _zeros(layout)
    obs[0, our_start + 1 * POKEMON_FULL_DIM + species_idx] = 135.0  # our slot 1 → Jolteon
    ctx = model.unpack({"observation": obs})
    assert ctx.species_ids[0, 1].item() == 135
    assert ctx.species_ids[0, 0].item() == 0


def test_obsunpack_history_span_is_exact():
    model, layout = _make_model()
    hist_start = layout["base_dim"] + ACTION_SPACE_SIZE
    obs = _zeros(layout)
    obs[0, hist_start : hist_start + TURN_DELTA_DIM] = 1.0  # fill the oldest slot
    ctx = model.unpack({"observation": obs})
    assert ctx.turn_history_raw.shape == (1, N_HISTORY_TURNS * TURN_DELTA_DIM)
    assert torch.allclose(ctx.turn_history_raw[0, :TURN_DELTA_DIM], torch.ones(TURN_DELTA_DIM))
    assert ctx.turn_history_raw[0, TURN_DELTA_DIM:].abs().sum() == 0


def test_obsunpack_active_idx_and_fainted_mask():
    model, layout = _make_model()
    our_start = layout["parts"]["our_team"]["start"]
    hp_off = layout["pokemon"]["hp"]["offset"]

    obs = _zeros(layout)
    # Active flag (last dim of slot) on our slot 2, with HP > 0 so it isn't fainted.
    obs[0, our_start + 2 * POKEMON_FULL_DIM + (POKEMON_FULL_DIM - 1)] = 1.0
    obs[0, our_start + 2 * POKEMON_FULL_DIM + hp_off] = 0.9
    ctx = model.unpack({"observation": obs})

    assert ctx.our_active_idx[0].item() == 2
    # Active slot is never masked; an HP=0 bench slot is fainted.
    assert bool(ctx.fainted_mask_ours[0, 2]) is False
    assert bool(ctx.fainted_mask_ours[0, 0]) is True


def _opp_three_slot_obs(layout):
    """Obs with opp slot 0 = revealed active, slot 1 = revealed-fainted (species_known=1, hp=0),
    slot 2 = unrevealed (all-zero → species_known=0, hp=0). Slots 3-5 stay unrevealed too."""
    from agents.observation.constants import POKEMON_SPECIES_KNOWN_OFFSET
    opp_start = layout["parts"]["opp_team"]["start"]
    hp_off = layout["pokemon"]["hp"]["offset"]
    obs = _zeros(layout)
    base0 = opp_start + 0 * POKEMON_FULL_DIM
    obs[0, base0 + (POKEMON_FULL_DIM - 1)] = 1.0   # active flag on opp slot 0
    obs[0, base0 + hp_off] = 0.8                   # alive
    obs[0, base0 + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
    base1 = opp_start + 1 * POKEMON_FULL_DIM
    obs[0, base1 + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0  # revealed but hp stays 0 → fainted
    return obs


def test_unrevealed_opp_masked_by_default():
    """Baseline: an unrevealed opp slot (species_known=0, hp=0) is key-masked exactly like a
    revealed-fainted one — the species_known bit is discarded by the mask."""
    model, layout = _make_model()  # attend_unrevealed_opponents=False
    ctx = model.unpack({"observation": _opp_three_slot_obs(layout)})
    assert bool(ctx.fainted_mask_opp[0, 0]) is False   # active — never masked
    assert bool(ctx.fainted_mask_opp[0, 1]) is True    # revealed-fainted — masked
    assert bool(ctx.fainted_mask_opp[0, 2]) is True    # unrevealed — masked (baseline)
    assert ctx.fainted_mask_opp[0, 3:].all().item() is True  # other unrevealed slots masked


def test_unrevealed_opp_attendable_when_flag_on():
    """Flag on: unrevealed opp slots (species_known=0) stay ATTENDABLE; only revealed-fainted
    slots (species_known=1, hp=0) remain masked. Active is always unmasked."""
    model, layout = _make_model(attend_unrevealed_opponents=True)
    ctx = model.unpack({"observation": _opp_three_slot_obs(layout)})
    assert bool(ctx.fainted_mask_opp[0, 0]) is False   # active — never masked
    assert bool(ctx.fainted_mask_opp[0, 1]) is True    # revealed-fainted — STILL masked
    assert bool(ctx.fainted_mask_opp[0, 2]) is False   # unrevealed — now attendable
    assert ctx.fainted_mask_opp[0, 3:].any().item() is False  # all unrevealed slots attendable


def test_unrevealed_flag_active_force_unmask_prevents_nan_row():
    """No-attention-NaN invariant (v8): the opp ACTIVE slot is force-unmasked even when it is itself
    revealed-fainted (hp=0). On an all-revealed-fainted board with the flag ON, EVERY slot satisfies
    `hp==0 & species_known>0.5` — so the force-unmask of the active slot is the ONLY thing keeping the
    key-padding row from being all-True (which would NaN nn.MultiheadAttention). Both assertions are
    load-bearing on that force-unmask line; without it slot 0 stays masked and the whole row is True."""
    from agents.observation.constants import POKEMON_SPECIES_KNOWN_OFFSET
    model, layout = _make_model(attend_unrevealed_opponents=True)
    opp_start = layout["parts"]["opp_team"]["start"]
    obs = _zeros(layout)
    for i in range(TEAM_SIZE):                          # all 6 opp slots revealed + fainted (hp stays 0)
        obs[0, opp_start + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
    obs[0, opp_start + 0 * POKEMON_FULL_DIM + (POKEMON_FULL_DIM - 1)] = 1.0   # slot 0 = active
    ctx = model.unpack({"observation": obs})
    assert bool(ctx.fainted_mask_opp[0, 0]) is False        # active force-unmasked despite hp==0
    assert ctx.fainted_mask_opp[0].all().item() is False    # ⇒ no all-True row ⇒ no attention NaN


# ---------------------------------------------------------------------------
# HiddenOppBeliefPool — opponent-belief CLS / K query tokens (B2)
# ---------------------------------------------------------------------------

def _all_team_and_mask(B, n_masked_per_side=3):
    """[B, 12, D_MODEL] team memory + a [B, 12] key-mask with the two actives (slots 0, TEAM_SIZE)
    always unmasked — the force-unmask invariant ObsUnpack guarantees."""
    all_team_out = torch.randn(B, 2 * TEAM_SIZE, D_MODEL)
    all_fainted = torch.zeros(B, 2 * TEAM_SIZE, dtype=torch.bool)
    for s in range(1, 1 + n_masked_per_side):                 # mask some of our + opp bench
        all_fainted[:, s] = True
        all_fainted[:, TEAM_SIZE + s] = True
    return all_team_out, all_fainted


@pytest.mark.parametrize("k", [1, 4])
def test_hidden_opp_belief_pool_shape_and_finite(k):
    pool = HiddenOppBeliefPool(k).eval()
    B = 3
    all_team_out, all_fainted = _all_team_and_mask(B)
    out = pool(all_team_out, all_fainted, B)
    assert out.shape == (B, k * D_MODEL)
    assert torch.isfinite(out).all()


def test_hidden_opp_belief_queries_are_distinct():
    """The whole point of K queries (vs identical zero-slots): they are non-identical BY
    CONSTRUCTION, so they can break the permutation-symmetry collapse."""
    pool = HiddenOppBeliefPool(4)
    q = pool.queries[0]                                       # [4, D_MODEL]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(q[i], q[j])


def test_hidden_opp_belief_no_nan_when_only_actives_unmasked():
    """The genuine NaN-risk board: every bench slot masked, only the two actives open. The pool
    must stay finite (memory_key_padding_mask never all-True per row)."""
    pool = HiddenOppBeliefPool(2).eval()
    B = 2
    all_team_out = torch.randn(B, 2 * TEAM_SIZE, D_MODEL)
    all_fainted = torch.ones(B, 2 * TEAM_SIZE, dtype=torch.bool)
    all_fainted[:, 0] = False                                 # our active
    all_fainted[:, TEAM_SIZE] = False                         # opp active
    out = pool(all_team_out, all_fainted, B)
    assert torch.isfinite(out).all()


def test_hidden_opp_belief_pool_rejects_bad_k():
    with pytest.raises(ValueError, match="k >= 1"):
        HiddenOppBeliefPool(0)


def test_opp_belief_cls_k_zero_is_baseline():
    """k=0 (default): no belief module, baseline projection dims — the current state."""
    base, _ = _make_model()
    assert base.opp_belief_cls_k == 0 and base.hidden_opp_belief is None
    assert "hidden_opp_belief.queries" not in dict(base.named_parameters())


@pytest.mark.parametrize("k", [1, 4])
def test_opp_belief_cls_k_grows_projection_by_k_times_dmodel(k):
    base, _ = _make_model()
    on, _ = _make_model(attend_unrevealed_opponents=True, opp_belief_cls_k=k)
    assert on.hidden_opp_belief is not None and on.hidden_opp_belief.k == k
    # Belief feeds BOTH heads, so both projection inputs grow by exactly k*D_MODEL.
    assert on.projection_input_dim == base.projection_input_dim + k * D_MODEL
    assert on.value_projection_input_dim == base.value_projection_input_dim + k * D_MODEL


def test_opp_belief_cls_k_requires_unmask():
    """k>0 reads the unrevealed opp slots; with the unmask flag off those slots are key-masked, so
    building the belief is incoherent → hard error at construction."""
    with pytest.raises(ValueError, match="attend_unrevealed_opponents"):
        _make_model(attend_unrevealed_opponents=False, opp_belief_cls_k=1)


def test_opp_belief_cls_k_negative_rejected():
    with pytest.raises(ValueError, match="opp_belief_cls_k must be >= 0"):
        _make_model(attend_unrevealed_opponents=True, opp_belief_cls_k=-1)


def test_opp_belief_cls_k_forward_finite_both_heads():
    on, layout = _make_model(attend_unrevealed_opponents=True, opp_belief_cls_k=3)
    pi, vf = on({"observation": _zeros(layout)})
    assert pi.shape[0] == 1 and vf.shape[0] == 1
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()


def test_embeddings_hp_soft_type_is_weighted_lookup():
    model, _ = _make_model()
    emb = model.embeddings
    probs = torch.zeros(1, 1, 16)
    probs[0, 0, 3] = 1.0  # one-hot on candidate type 3
    soft = emb.hp_soft_type(probs)                                   # [1, 1, type_dim]
    expected = emb.type_embedding(emb.hp_type_idx_map[3]).view(1, 1, -1)
    assert torch.allclose(soft, expected, atol=1e-6)


def test_embeddings_delta_slot_width_matches_history_proj():
    model, layout = _make_model()
    out = model.embeddings.embed_delta_slot(torch.zeros(2, TURN_DELTA_DIM))
    assert out.shape == (2, turn_delta_embed_dim(layout))
    assert out.shape[1] == model.team_transformer.history_proj.in_features


def test_embeddings_hp_type_map_matches_type_encoder_order():
    from agents.observation.types import TypeEncoder
    from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
    model, _ = _make_model()
    expected = [TypeEncoder.TYPE_TO_IDX[t.name] for t in HIDDEN_POWER_TYPE_ORDER]
    assert model.embeddings.hp_type_idx_map.tolist() == expected


# ---------------------------------------------------------------------------
# PokemonEncoder
# ---------------------------------------------------------------------------

def test_pokemon_encoder_role_token_shape():
    model, layout = _make_model()
    ctx = model.unpack({"observation": _zeros(layout)})
    role = model.pokemon_encoder(ctx, model.embeddings)
    assert role.shape == (1, 2 * TEAM_SIZE, ROLE_TOKEN_SIZE)


def test_pokemon_encoder_move_self_attn_wired():
    model, layout = _make_model()
    ctx = model.unpack({"observation": torch.rand(1, layout["total_dim"]) * 4.0})
    before = model.pokemon_encoder(ctx, model.embeddings)
    for p in model.pokemon_encoder.move_self_attn.parameters():
        p.data.zero_()
    after = model.pokemon_encoder(ctx, model.embeddings)
    assert not torch.allclose(before, after)


# ---------------------------------------------------------------------------
# TeamTransformer
# ---------------------------------------------------------------------------

def test_team_transformer_output_shapes():
    model, layout = _make_model()
    ctx = model.unpack({"observation": torch.rand(1, layout["total_dim"]) * 4.0})
    role = model.pokemon_encoder(ctx, model.embeddings)
    # gen3_entity_move_seats_v1: the forward returns a third element — the refined extra entity
    # seats (None when no `extra` is passed, as here).
    our_out, their_out, extra_out = model.team_transformer(role, ctx, model.embeddings)
    assert our_out.shape == (1, TEAM_SIZE, D_MODEL)
    assert their_out.shape == (1, TEAM_SIZE, D_MODEL)
    assert extra_out is None


# ---------------------------------------------------------------------------
# CLSPool — true isolation on a synthetic context
# ---------------------------------------------------------------------------

def test_clspool_active_refined_is_exact_gather():
    model, _ = _make_model()
    B = 3
    our_out = torch.randn(B, TEAM_SIZE, D_MODEL)
    their_out = torch.randn(B, TEAM_SIZE, D_MODEL)
    our_active_idx = torch.tensor([0, 4, 2])
    ctx = _dummy_ctx(
        batch_size=B, device=torch.device("cpu"),
        fainted_mask_ours=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        fainted_mask_opp=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        our_active_idx=our_active_idx,
    )
    _, _, our_active_refined, _ = model.cls_pool(our_out, their_out, ctx)
    expected = our_out[torch.arange(B), our_active_idx]
    assert torch.allclose(our_active_refined, expected)


def test_clspool_ignores_fainted_keyed_slot():
    """Changing a fainted (key-masked) team slot must not move the pooled output."""
    model, _ = _make_model()
    B = 1
    base = torch.randn(B, TEAM_SIZE, D_MODEL)
    fainted = torch.zeros(B, TEAM_SIZE, dtype=torch.bool)
    fainted[0, 5] = True                       # slot 5 fainted/masked
    ctx = _dummy_ctx(
        batch_size=B, device=torch.device("cpu"),
        fainted_mask_ours=fainted,
        fainted_mask_opp=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        our_active_idx=torch.tensor([0]),      # active slot 0, unmasked
    )
    their = torch.randn(B, TEAM_SIZE, D_MODEL)
    pooled_a, _, _, _ = model.cls_pool(base, their, ctx)
    mutated = base.clone()
    mutated[0, 5] = torch.randn(D_MODEL)       # perturb only the masked slot
    pooled_b, _, _, _ = model.cls_pool(mutated, their, ctx)
    assert torch.allclose(pooled_a, pooled_b, atol=1e-6)


def test_clspool_value_pool_shape_and_masking():
    """The value pool returns a [B, D_MODEL] summary and ignores fainted (key-masked)
    slots on EITHER side — it attends over all 12 team tokens."""
    model, _ = _make_model()
    B = 1
    our = torch.randn(B, TEAM_SIZE, D_MODEL)
    their = torch.randn(B, TEAM_SIZE, D_MODEL)
    fainted_ours = torch.zeros(B, TEAM_SIZE, dtype=torch.bool)
    fainted_opp = torch.zeros(B, TEAM_SIZE, dtype=torch.bool)
    fainted_opp[0, 3] = True                    # an opponent slot is fainted/masked
    ctx = _dummy_ctx(
        batch_size=B, device=torch.device("cpu"),
        fainted_mask_ours=fainted_ours, fainted_mask_opp=fainted_opp,
        our_active_idx=torch.tensor([0]),
    )
    _, _, _, value_a = model.cls_pool(our, their, ctx)
    assert value_a.shape == (B, D_MODEL)
    # Perturbing only the masked opponent slot must not move the value summary.
    mutated = their.clone()
    mutated[0, 3] = torch.randn(D_MODEL)
    _, _, _, value_b = model.cls_pool(our, mutated, ctx)
    assert torch.allclose(value_a, value_b, atol=1e-6)


def test_clspool_value_query_is_wired():
    """Zeroing the value_cls query must change the value pool — proves it's queried."""
    model, _ = _make_model()
    B = 1
    our = torch.randn(B, TEAM_SIZE, D_MODEL)
    their = torch.randn(B, TEAM_SIZE, D_MODEL)
    ctx = _dummy_ctx(
        batch_size=B, device=torch.device("cpu"),
        fainted_mask_ours=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        fainted_mask_opp=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        our_active_idx=torch.tensor([0]),
    )
    _, _, _, value_before = model.cls_pool(our, their, ctx)
    model.cls_pool.value_cls.data.zero_()
    _, _, _, value_after = model.cls_pool(our, their, ctx)
    assert not torch.allclose(value_before, value_after)


# ---------------------------------------------------------------------------
# ProjectionAssembler — true isolation
# ---------------------------------------------------------------------------

def test_projection_assembler_width_and_passthrough():
    model, layout = _make_model()
    active_ctx_dim = layout["active_context_dim"]
    B, K = 2, 7
    our_pool = torch.randn(B, D_MODEL)
    their_pool = torch.randn(B, D_MODEL)
    active = torch.randn(B, D_MODEL)
    value_pool = torch.randn(B, D_MODEL)
    non_matchup = torch.randn(B, K)
    ctx = _dummy_ctx(
        our_ctx_raw=torch.randn(B, active_ctx_dim),
        opp_ctx_raw=torch.randn(B, active_ctx_dim),
        non_matchup_rest=non_matchup,
    )
    pi_combined, vf_combined = model.assembler(our_pool, their_pool, active, value_pool, ctx)
    # gen3_ctx_dedup_v1: the encoded active-ctx pair is DELETED from both heads (its content
    # rides the active tokens via the E2 injection + the global token).
    # Policy input: our/their pools + active + non-matchup tail.
    pi_width = 3 * D_MODEL + K
    # Value input: value pool + non-matchup tail (no team pools / active).
    vf_width = D_MODEL + K
    assert pi_combined.shape == (B, pi_width)
    assert vf_combined.shape == (B, vf_width)
    # The non-matchup tail is concatenated verbatim into both heads.
    assert torch.allclose(pi_combined[:, -K:], non_matchup)
    assert torch.allclose(vf_combined[:, -K:], non_matchup)


def test_projection_assembler_concat_is_dead_and_seeds_ride_vf_only():
    """gen3_no_concat_v1 (v61): the flat damage block NEVER enters pi, and enters vf only
    through the multi-seed readout (k*dim, computed from the per-mon rows — not verbatim).
    A no-op assembler (seed_per_mon=0, the op-less config) ignores the block entirely."""
    from agents.model.arch_constants import VALUE_SEED_K, VALUE_SEED_DIM
    from agents.observation.constants import TEAM_SIZE
    model, layout = _make_model()
    active_ctx_dim = layout["active_context_dim"]
    B, K = 2, 7
    per_mon = 12
    args = (torch.randn(B, D_MODEL), torch.randn(B, D_MODEL),
            torch.randn(B, D_MODEL), torch.randn(B, D_MODEL))
    hp_active = torch.zeros(B, 2 * TEAM_SIZE, 3)
    hp_active[:, :, 0] = 1.0                                    # everyone alive
    ctx = _dummy_ctx(
        our_ctx_raw=torch.randn(B, active_ctx_dim),
        opp_ctx_raw=torch.randn(B, active_ctx_dim),
        non_matchup_rest=torch.randn(B, K),
        hp_and_active=hp_active,
    )
    from agents.model.features_extractor import ProjectionAssembler
    asm = ProjectionAssembler(layout, seed_per_mon=per_mon)
    pi0, vf0 = asm(*args, ctx)
    rows = torch.randn(B, TEAM_SIZE, per_mon)                   # the typed incoming-rows view
    pi1, vf1 = asm(*args, ctx, seed_rows=rows)
    assert pi1.shape[1] == pi0.shape[1], "the rows must NEVER widen pi (the concat is dead)"
    assert vf1.shape[1] - vf0.shape[1] == VALUE_SEED_K * VALUE_SEED_DIM
    assert torch.equal(pi1, pi0), "pi must be bit-identical with and without the rows"
    # And an op-less assembler ignores the rows on vf too.
    asm0 = ProjectionAssembler(layout, seed_per_mon=0)
    pi2, vf2 = asm0(*args, ctx, seed_rows=rows)
    pi3, vf3 = asm0(*args, ctx)
    assert torch.equal(pi2, pi3) and torch.equal(vf2, vf3)
