"""Tests for the Hidden-Power-aware wiring inside Gen3FeaturesExtractor:

- The weighted type embedding for HP move slots (one-hot collapses to a direct
  type_embedding lookup).
- The hp_probs feature appended to move_network input (zero for non-HP slots).
- The matchup validity mask (move_known AND opp_species_known) appended to
  move_network input alongside the existing matchup scalars.
"""
import numpy as np
import torch
import pytest
from gymnasium import spaces

from poke_env.battle.pokemon_type import PokemonType

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    MOVE_NET_HIDDEN,
)
from agents.observation.constants import (
    POKEMON_FULL_DIM,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
    POKEMON_MOVES_OFFSET,
    MOVE_SLOT_DIM,
    TEAM_SIZE,
    OFFSET_OUR_TEAM,
    OFFSET_OPP_TEAM,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER


@pytest.fixture(scope="module")
def extractor():
    mappings = load_mappings()
    enc = Gen3ObservationEncoder(mappings)
    obs_space = spaces.Dict({
        'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(enc.dimension,), dtype=np.float32),
        'action_mask': spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),
    })
    fe = Gen3FeaturesExtractor(obs_space, layout=enc.get_layout(), mappings=mappings)
    fe.eval()
    return fe, enc


def test_hp_type_idx_map_matches_type_encoder(extractor):
    """The buffer that drives the weighted type embedding must map each of the
    16 HP candidate slots to the same type_id that the encoder writes into a
    typed-HP move slot. Otherwise the one-hot collapse property breaks."""
    fe, _ = extractor
    expected = [TypeEncoder.TYPE_TO_IDX[t.name] for t in HIDDEN_POWER_TYPE_ORDER]
    assert fe.hp_type_idx_map.tolist() == expected


def test_one_hot_hp_distribution_collapses_to_direct_lookup(extractor):
    """A one-hot HP distribution at index i must produce the same soft embedding
    as type_embedding(hp_type_idx_map[i]). This is the property that lets own
    team's HP path stay identical to the pre-change behaviour."""
    fe, _ = extractor
    for i, t in enumerate(HIDDEN_POWER_TYPE_ORDER):
        one_hot = torch.zeros(16)
        one_hot[i] = 1.0
        soft = one_hot @ fe.type_embedding(fe.hp_type_idx_map)
        direct = fe.type_embedding(torch.tensor(TypeEncoder.TYPE_TO_IDX[t.name]))
        assert torch.allclose(soft, direct, atol=1e-6), f"mismatch for {t.name}"


def _make_zero_obs(enc):
    obs = np.zeros(enc.dimension, dtype=np.float32)
    return obs


def _set_hp_slot(obs, team_offset, slot_idx, move_num, type_idx, hp_probs):
    """Write a single Pokémon slot's HP move + hp_probs block."""
    pk_start = team_offset + slot_idx * POKEMON_FULL_DIM
    # First move slot
    move_slot_start = pk_start + POKEMON_MOVES_OFFSET
    obs[move_slot_start + 0] = float(move_num)  # move id
    obs[move_slot_start + 4] = float(type_idx)  # type id
    obs[move_slot_start + 6] = 1.0              # known flag
    # hp_probs at offset 80 within the Pokémon vector
    obs[pk_start + POKEMON_HP_PROBS_OFFSET : pk_start + POKEMON_HP_PROBS_OFFSET + 16] = hp_probs
    obs[pk_start + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0


def test_hp_slot_detection_only_fires_on_canonical_move_num(extractor):
    """The is_hp_slot mask must trigger on move_num == HIDDEN_POWER_MOVE_NUM and
    nowhere else. We verify by building two obs vectors that differ only in the
    move_num of one slot — HP-num vs a non-HP num — and confirming the
    embedded_move_types path produces different outputs only when HP is set."""
    fe, enc = extractor
    obs = _make_zero_obs(enc)
    # Slot with HP-num and one-hot Ice
    ice_idx = HIDDEN_POWER_TYPE_ORDER.index(PokemonType.ICE)
    probs = np.zeros(16, dtype=np.float32)
    probs[ice_idx] = 1.0
    _set_hp_slot(obs, OFFSET_OUR_TEAM, 0, HIDDEN_POWER_MOVE_NUM, type_idx=0, hp_probs=probs)

    # Compare to a "fake HP" scenario where move_num is something else (Surf=57)
    # but hp_probs are still set — the extractor should ignore the probs and use
    # the raw type embedding (which is type_id=0 since we wrote 0). The forward
    # outputs must differ between these two scenarios.
    obs_no_hp = obs.copy()
    pk_start = OFFSET_OUR_TEAM + 0 * POKEMON_FULL_DIM
    obs_no_hp[pk_start + POKEMON_MOVES_OFFSET + 0] = 57.0  # Surf

    x_hp     = {'observation': torch.tensor(obs)[None],     'action_mask': torch.ones(1, 11)}
    x_no_hp  = {'observation': torch.tensor(obs_no_hp)[None], 'action_mask': torch.ones(1, 11)}
    with torch.no_grad():
        out_hp    = fe(x_hp)
        out_no_hp = fe(x_no_hp)
    # Outputs must differ — the HP-num path engaged the soft type embedding.
    assert not torch.allclose(out_hp, out_no_hp)


def test_matchup_validity_distinguishes_unknown_from_immune(extractor):
    """When an opponent slot is unrevealed (species_known=0), the matchup
    validity bit for that cell must be 0 regardless of the matchup value.
    When the opp is revealed (species_known=1) but the type chart yields 0×,
    validity stays 1 with matchup=0.

    We can't read move_features directly, but we can verify the forward output
    *differs* between two obs that share matchup=0 but differ in opp species_known.
    """
    fe, enc = extractor
    base = _make_zero_obs(enc)
    # Our team mon 0 has a Ground move targeted at opp 0
    pk_start = OFFSET_OUR_TEAM + 0 * POKEMON_FULL_DIM
    move_slot = pk_start + POKEMON_MOVES_OFFSET
    base[move_slot + 0] = 89.0     # earthquake
    base[move_slot + 4] = float(TypeEncoder.TYPE_TO_IDX["GROUND"])
    base[move_slot + 6] = 1.0      # known
    base[pk_start + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0

    # All opp slots stay 0 — their species_known = 0 — so matchup_validity for
    # opp 0 is 0. We don't write any matchup values; both scenarios have all
    # zeros there. The forward should still differentiate the two cases through
    # the validity bit (matchup_validity is appended to move_network input).
    obs_unrevealed = base.copy()

    # Now flip species_known=1 on opp 0 with no other changes — matchup stays
    # 0 (no real matchup data set) but validity goes to 1.
    obs_revealed = base.copy()
    opp_pk_start = OFFSET_OPP_TEAM + 0 * POKEMON_FULL_DIM
    obs_revealed[opp_pk_start + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0

    x_un  = {'observation': torch.tensor(obs_unrevealed)[None], 'action_mask': torch.ones(1, 11)}
    x_rev = {'observation': torch.tensor(obs_revealed)[None],   'action_mask': torch.ones(1, 11)}
    with torch.no_grad():
        out_un  = fe(x_un)
        out_rev = fe(x_rev)
    # Must differ — the validity bit feeds the move_network differently in the
    # two scenarios, which propagates to the projection output.
    assert not torch.allclose(out_un, out_rev)


def test_forward_runs_with_hp_probs_in_obs(extractor):
    """Smoke check: an obs containing populated hp_probs and an HP slot does not
    NaN, inf, or crash the forward pass."""
    fe, enc = extractor
    obs = _make_zero_obs(enc)
    # Two HP scenarios: own one-hot, opp distribution
    own_probs = np.zeros(16, dtype=np.float32)
    own_probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.GRASS)] = 1.0
    _set_hp_slot(
        obs, OFFSET_OUR_TEAM, 0, HIDDEN_POWER_MOVE_NUM,
        type_idx=TypeEncoder.TYPE_TO_IDX["GRASS"], hp_probs=own_probs,
    )

    opp_probs = np.zeros(16, dtype=np.float32)
    opp_probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.ICE)] = 0.6
    opp_probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.FIRE)] = 0.4
    _set_hp_slot(
        obs, OFFSET_OPP_TEAM, 0, HIDDEN_POWER_MOVE_NUM,
        type_idx=0, hp_probs=opp_probs,
    )

    x = {'observation': torch.tensor(obs)[None], 'action_mask': torch.ones(1, 11)}
    with torch.no_grad():
        out = fe(x)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 512)
