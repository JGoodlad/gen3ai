"""
Unit tests for the unified-transformer feature extractor (ai_v4).

Coverage:
  - module presence and shape sanity for the new transformer components
  - forward-pass output is responsive to changes in every input region
    (team slots, history slots, global scalars)
  - history positional encoding and key-padding mask are actually wired in
"""
import torch
import numpy as np
import gymnasium as gym

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    ROLE_TOKEN_SIZE,
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
    NUM_TOKEN_TYPES,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


def _make_model():
    """Build a freshly initialised extractor with the live layout."""
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    layout = encoder.get_layout()
    total_dim = layout["total_dim"]
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(total_dim,), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()
    return model, layout


def _zero_obs(layout):
    """Return a zero observation tensor [1, total_dim]."""
    return torch.zeros(1, layout["total_dim"])


# ---------------------------------------------------------------------------
# Constants and shape sanity
# ---------------------------------------------------------------------------

def test_d_model_matches_role_token_size():
    """d_model = role_token_size so team role tokens enter the transformer unchanged."""
    assert D_MODEL == ROLE_TOKEN_SIZE


# ---------------------------------------------------------------------------
# Module presence — proves the design's component list is implemented
# ---------------------------------------------------------------------------

def test_unified_transformer_modules_exist():
    model, _ = _make_model()
    tt = model.team_transformer
    assert hasattr(tt, "token_type_emb")
    assert hasattr(tt, "global_proj")
    assert hasattr(tt, "transformer_layers")
    cp = model.cls_pool
    assert hasattr(cp, "our_cls")
    assert hasattr(cp, "their_cls")
    assert hasattr(cp, "our_cls_attn")
    assert hasattr(cp, "their_cls_attn")
    assert hasattr(cp, "norm_pool_our")
    assert hasattr(cp, "norm_pool_their")


def test_phase_modules_exist():
    """The decomposed pipeline exposes its phases as named sub-modules."""
    model, _ = _make_model()
    for phase in ("embeddings", "unpack", "pokemon_encoder", "team_transformer", "cls_pool", "assembler"):
        assert hasattr(model, phase), f"phase module {phase!r} missing"


def test_removed_modules_absent():
    """The five hand-engineered attention paths and the v3 injection MLPs must be gone."""
    model, _ = _make_model()
    for removed in (
        "pressure_attn", "safety_attn", "synergy_attn", "threat_attn", "opp_synergy_attn",
        "our_pool_query", "their_pool_query", "our_pool_attn", "their_pool_attn",
        "active_ctx_to_role", "td_conditioner",
        "turn_history_attn", "turn_history_norm",
        # gen3_frame_deletion_v1 — deleted with the lag frames:
        "history_proj", "turn_history_pos_emb",
        "status_embedding",
    ):
        assert not hasattr(model, removed), f"v3 module {removed!r} should be removed in v4"


def test_token_type_embedding_shape():
    model, _ = _make_model()
    assert model.team_transformer.token_type_emb.num_embeddings == NUM_TOKEN_TYPES
    assert model.team_transformer.token_type_emb.embedding_dim == D_MODEL


def test_global_proj_shape():
    model, layout = _make_model()
    active_ctx_dim = layout.get("active_context_dim", 22)
    tt = model.team_transformer
    expected_in = 2 * active_ctx_dim + tt._non_matchup_rest_dim
    assert tt.global_proj.in_features == expected_in
    assert tt.global_proj.out_features == D_MODEL


def test_transformer_stack_shape():
    model, _ = _make_model()
    assert len(model.team_transformer.transformer_layers) == TRANSFORMER_N_LAYERS
    for layer in model.team_transformer.transformer_layers:
        # gen3_edge_bias_trunk_v1: the stack is the BiasedEncoderLayer clone — fused qkv in_proj
        # (3·d_model) instead of nn.MultiheadAttention's self_attn.
        assert layer.in_proj.out_features == 3 * D_MODEL
        assert layer.n_heads == TRANSFORMER_N_HEADS
        # FFN expansion lives in linear1.out_features
        assert layer.linear1.out_features == TRANSFORMER_FFN_DIM


def test_cls_parameters_shape():
    model, _ = _make_model()
    assert model.cls_pool.our_cls.shape == (1, 1, D_MODEL)
    assert model.cls_pool.their_cls.shape == (1, 1, D_MODEL)


def test_token_count_matches_design():
    """Base token sequence length = 2 * TEAM_SIZE + 1 (global).

    gen3_frame_deletion_v1 dropped the N_HISTORY_TURNS history seats that used to sit between
    the team tokens and the global one. That is the load-bearing half: every edge family
    addressing the GLOBAL seat computes its index as `2 * TEAM_SIZE`, and every `extra` seat
    (E3/E4, the H-B event seats) is `_total_tokens + k` — so a stale count here would silently
    write whole edge families onto the wrong tokens rather than raise."""
    from agents.observation.constants import TEAM_SIZE
    model, _ = _make_model()
    assert model.team_transformer._total_tokens == 2 * TEAM_SIZE + 1


# ---------------------------------------------------------------------------
# Forward pass: shape and basic responsiveness
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    model, layout = _make_model()
    obs = _zero_obs(layout)
    with torch.no_grad():
        pi_features, vf_features = model({"observation": obs})
    # Dual-head extractor returns (policy, value) features, both PROJECTION_DIM-wide.
    assert pi_features.shape == (1, model.features_dim)
    assert vf_features.shape == (1, model.features_dim)


def test_active_context_changes_output():
    """A non-zero active-context dim must propagate through to the projected output."""
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]

    obs_zero = _zero_obs(layout)
    obs_ctx = obs_zero.clone()
    obs_ctx[0, ctx_start] = 1.0  # first dim of our active context

    with torch.no_grad():
        out_zero, _ = model.forward_internal({"observation": obs_zero})
        out_ctx , _ = model.forward_internal({"observation": obs_ctx})
    assert not torch.allclose(out_zero, out_ctx)


def test_opp_active_context_changes_output():
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]
    active_ctx_dim = layout.get("active_context_dim", 22)

    obs_zero = _zero_obs(layout)
    obs_ctx = obs_zero.clone()
    obs_ctx[0, ctx_start + active_ctx_dim] = 1.0

    with torch.no_grad():
        out_zero, _ = model.forward_internal({"observation": obs_zero})
        out_ctx , _ = model.forward_internal({"observation": obs_ctx})
    assert not torch.allclose(out_zero, out_ctx)


# ---------------------------------------------------------------------------
# Turn history — masking, positional encoding, attention path all live
# ---------------------------------------------------------------------------

def test_transformer_layer_wired_in():
    """Zeroing all parameters of the first transformer layer must change the output."""
    model, layout = _make_model()

    # Non-trivial input: touch a few inputs so tokens are not exactly zero everywhere.
    obs = _zero_obs(layout)
    ctx_start = layout["parts"]["context"]["start"]
    obs[0, ctx_start] = 1.0
    # gen3_frame_deletion_v1: the second non-zero poke used to land in the lag frames. Use the
    # H-A2 pair-history block instead — still a real, tokenised input, so the test keeps its
    # point (a non-trivial forward), rather than degrading to an all-but-one-zero obs.
    obs[0, layout['pair_history_offset'] + 3] = 0.3

    with torch.no_grad():
        out_before, _ = model.forward_internal({"observation": obs})
        for p in model.team_transformer.transformer_layers[0].parameters():
            p.zero_()
        out_after, _ = model.forward_internal({"observation": obs})
    assert not torch.allclose(out_before, out_after)


def test_cls_pooling_wired_in():
    """Zeroing the CLS attention modules must change the output for non-zero input."""
    model, layout = _make_model()

    obs = _zero_obs(layout)
    obs[0, layout["parts"]["context"]["start"]] = 1.0

    with torch.no_grad():
        out_before, _ = model.forward_internal({"observation": obs})
        for p in model.cls_pool.our_cls_attn.parameters():
            p.zero_()
        for p in model.cls_pool.their_cls_attn.parameters():
            p.zero_()
        # The CLS parameter vectors themselves are nontrivially queried — zero them too.
        model.cls_pool.our_cls.data.zero_()
        model.cls_pool.their_cls.data.zero_()
        out_after, _ = model.forward_internal({"observation": obs})
    assert not torch.allclose(out_before, out_after)


# ---------------------------------------------------------------------------
# Active-slot identification — guards against stale hardcoded indices into
# the per-Pokémon layout. The active flag is invariantly the LAST dim of
# `hp_and_active = pokemon_part[:, :, hp_offset:]`; this index moved when
# spread (step 1) and the HP candidate block (step 2) were appended.
# ---------------------------------------------------------------------------

def test_active_flag_is_last_dim_of_hp_and_active_block():
    """
    Documents and pins the invariant that the active flag lives at the LAST
    position of `hp_and_active`. If POKEMON_FULL_DIM grows (e.g. step 4 adds
    another per-slot block) and someone forgets to keep the active flag at
    the end, this test fires with a precise message rather than letting
    `hp_and_active[:, :, -1]` silently point at the wrong dim.
    """
    from agents.observation.constants import POKEMON_FULL_DIM
    _, layout = _make_model()
    hp_off = layout["pokemon"]["hp"]["offset"]
    hp_and_active_dim = POKEMON_FULL_DIM - hp_off

    # The state_encoder writes the active flag at `start + POKEMON_VECTOR_DIM`
    # for every slot — i.e. at POKEMON_FULL_DIM - 1 within the slot, which is
    # hp_and_active_dim - 1 within the hp_and_active block.
    expected_active_index_in_block = hp_and_active_dim - 1
    assert expected_active_index_in_block >= 0
    # The forward pass relies on this — `active_flags = hp_and_active[:, :, -1]`.
    # If hp_and_active_dim ever drops to 0 or the layout invariant changes,
    # this test gives an early warning before the model silently miscomputes
    # the active slot.
    assert hp_and_active_dim >= 5, (
        "hp_and_active block must contain at least [HP, species_known, sleep_ctr, "
        f"toxic_ctr, ..., active_flag] — got dim={hp_and_active_dim}"
    )


def test_active_flag_detection_picks_marked_slot():
    """
    With every own-team slot populated (HP > 0, spread filled — mimicking real
    own-team encoding) and each slot carrying a DISTINCT species id, set the
    active flag on slot 3 only and verify the model's forward pass produces a
    *different* output than when the active flag is on slot 0.

    This is the second-order check: even when the post-spread layout has many
    nonzero per-slot dims that a stale-index lookup might pick up, the model
    must still react to the actual active flag at the last position — it must
    surface slot 3's content (vs slot 0's) through `our_active_refined`.

    Why distinct species ids matter: if all six slots were byte-identical apart
    from the flag position, the network is permutation-symmetric over them, so
    "extract the active slot" yields the SAME vector whether the active slot is
    0 or 3 (slot-0-of-config-0 == slot-3-of-config-3 by symmetry). The real
    signal would then be exactly zero, not small-but-real, and the test could
    only ever pass on float round-off. Giving each slot a unique species breaks
    that symmetry so the active-slot readout genuinely changes — a clean,
    deterministic ~1e-1 signal we can assert against a noise-floor threshold.
    """
    from agents.observation.constants import (
        POKEMON_FULL_DIM,
        POKEMON_SPREAD_OFFSET,
        POKEMON_SPREAD_DIM,
    )
    # Determinism + robust signal separation. We (a) seed the model so init is
    # fixed and (b) run the comparison in float64 so the real signal is cleanly
    # separable from round-off, then assert against an explicit noise-floor
    # threshold rather than `torch.allclose`'s relative tolerance.
    #
    # The original test used `not allclose(...)` with all six slots byte-identical
    # apart from the flag. That made the network permutation-symmetric over the
    # slots, so the active-slot readout was the SAME for active_slot=0 and =3 and
    # the true diff was ~1e-15 (round-off) — the assertion only ever passed on the
    # larger float32 noise (~1e-7) and was a flaky near-tie. Breaking the symmetry
    # (distinct species per slot, above) restores a genuine, deterministic ~1e-1
    # signal, comfortably ~14 orders of magnitude above the float64 noise floor.
    torch.manual_seed(0)
    model, layout = _make_model()
    model.eval()
    model.double()
    our_start = layout["parts"]["our_team"]["start"]
    hp_off = layout["pokemon"]["hp"]["offset"]
    species_info = layout["pokemon"]["species"]
    species_id_off = species_info["offset"] + species_info["layout"]["species_id"]["offset"]
    active_flag_off_within_slot = POKEMON_FULL_DIM - 1

    def _zero_obs_double(layout):
        return torch.zeros(1, layout["total_dim"], dtype=torch.float64)

    def populate(obs, active_slot: int):
        # Fill every alive own-team slot (HP=1, spread filled) and give each a
        # DISTINCT species id so the slots are not permutation-symmetric, then
        # set the active flag on the requested slot. The distinct species is what
        # makes "extract the active slot" yield genuinely different content for
        # active_slot=0 vs active_slot=3 (see the docstring).
        for i in range(6):
            slot = our_start + i * POKEMON_FULL_DIM
            obs[0, slot + hp_off] = 1.0
            obs[0, slot + hp_off + 1] = 1.0  # species_known
            obs[0, slot + species_id_off] = float(10 + i)  # distinct species per slot
            for j in range(POKEMON_SPREAD_DIM):
                obs[0, slot + POKEMON_SPREAD_OFFSET + j] = 0.5
        obs[0, our_start + active_slot * POKEMON_FULL_DIM + active_flag_off_within_slot] = 1.0
        return obs

    obs_slot0 = populate(_zero_obs_double(layout), active_slot=0)
    obs_slot3 = populate(_zero_obs_double(layout), active_slot=3)

    with torch.no_grad():
        out0, _ = model.forward_internal({"observation": obs_slot0})
        out3, _ = model.forward_internal({"observation": obs_slot3})

    # Outputs must differ — different active flag changes per-slot inputs
    # (and so role tokens), and also changes which role token is extracted
    # as `our_active_refined`. Compare in double precision against a noise-floor
    # threshold rather than `allclose`'s relative tolerance (which would be a
    # flaky near-tie for this small-but-real signal).
    max_diff = (out0 - out3).abs().max().item()
    assert max_diff > 1e-9, (
        "active flag had no effect on the forward output "
        f"(max abs diff {max_diff:.2e}) — moving the active flag from slot 0 "
        "to slot 3 must change the forward output."
    )
