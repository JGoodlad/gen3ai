"""Unit tests for gen3_unrevealed_outgoing_prior_v1 — the item-4 GIGO fix: outgoing damage priced
against UNREVEALED opponent slots reads an EXPECTED-LATENT defender (the Species-Clause-filtered
gen3ou usage prior marginalized through the v36 tables) instead of zeros.

The claims (designs/ai_v9/design_conditional_opponent_cells.md §4.1):
  * the unrevealed marginal == the usage prior with every revealed species zeroed + renormalized
    (Species Clause), and a revealed species carries ZERO mass;
  * the unrevealed D1 cell == a direct recomputation from the op's own tables (E[mult] / E[def] /
    E[maxhp] through the gen3 formula) — non-zero high/low, P(KO) NULLED, `revealed` channel 0;
  * REVEALED columns are byte-identical to the pre-fix kernel (flipping slots to unrevealed changes
    ONLY those slots' columns);
  * an unrevealed slot is FORCED-ALIVE: the obs hp placeholder (0) does not zero its cells.
"""
import inspect

import gymnasium as gym
import numpy as np
import torch

from agents import gen3_data
from agents.model.damage_op import (
    POKEMON_CONDITION_OFFSET, _DMG_CHIP_CAP, _DMG_ROLL_MIN, _SB_DEF,
)
from agents.model.damage_tables import _T2I
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.observation.pokemon import POKEMON_SPREAD_OFFSET
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

_D1_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="both",
                   damage_op=True, damage_outgoing=True, move_latent=True,
                   move_prior_fusion=True,
                   entity_topk_seats=5, edge_bias_families="d1")

_BODYSLAM = gen3_data.moves.get("bodyslam")
_SNORLAX = gen3_data.species.get("snorlax")
_TTAR = gen3_data.species.get("tyranitar")
_SKARM = gen3_data.species.get("skarmory")


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=2, seed=71):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


def _ctx(fe, seed=71):
    obs = _obs(seed=seed)
    with torch.no_grad():
        fe(obs)
    return fe.unpack(obs)


def _pin_scenario(ctx):
    """Force the canonical §4.1 board: OUR active = neutral-spread Snorlax at slot 0 with only
    Body Slam legal (physical STAB, 100% acc, no CB/burn/boosts/screens/weather); opp slot 0 =
    revealed alive Tyranitar (the active); opp slots 1-5 UNREVEALED (species 0, types 0, hp 0)."""
    B = ctx.batch_size
    ar = torch.arange(B)
    ctx.our_active_idx[:] = 0
    ctx.opp_active_local[:] = 0
    ctx.species_ids[:, 0] = _SNORLAX.num
    ctx.type1_ids[:, 0] = _T2I["NORMAL"]
    ctx.type2_ids[:, 0] = 0
    ctx.item_ids[:, 0] = 0
    sp = POKEMON_SPREAD_OFFSET
    ctx.pokemon_part[:, 0, sp:sp + 18] = 0.0
    ctx.pokemon_part[:, 0, sp:sp + 6] = 1.0                      # IV 31
    ctx.pokemon_part[:, 0, sp + 12] = 1.0                        # spread_known
    ctx.pokemon_part[:, 0, sp + 13:sp + 18] = 1.0                # neutral nature
    ctx.pokemon_part[:, 0, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
    ctx.hp_and_active[:, 0, 0] = 1.0
    ctx.our_active_req_move_ids[:, :] = 0
    ctx.our_active_req_move_ids[:, 0] = _BODYSLAM.num
    ctx.our_active_req_move_type_ids[:, :] = 0
    ctx.our_active_req_move_type_ids[:, 0] = _T2I["NORMAL"]
    ctx.our_active_req_move_legal[:, :] = 0.0
    ctx.our_active_req_move_legal[:, 0] = 1.0
    ctx.our_ctx_raw[:, :] = 0.0
    ctx.opp_ctx_raw[:, :] = 0.0
    ctx.screen_feature[:, :] = 0.0
    ctx.weather_feature[:, :] = 0.0
    # opp side: slot 0 revealed TTar active; 1-5 hidden placeholders.
    ctx.species_ids[:, TEAM_SIZE:] = 0
    ctx.type1_ids[:, TEAM_SIZE:] = 0
    ctx.type2_ids[:, TEAM_SIZE:] = 0
    ctx.ability1_ids[:, TEAM_SIZE:] = 0
    ctx.hp_and_active[:, TEAM_SIZE:, :] = 0.0
    ctx.species_ids[:, TEAM_SIZE] = _TTAR.num
    ctx.type1_ids[:, TEAM_SIZE] = _T2I["ROCK"]
    ctx.type2_ids[:, TEAM_SIZE] = _T2I["DARK"]
    ctx.hp_and_active[:, TEAM_SIZE, 0] = 1.0
    ctx.hp_and_active[:, TEAM_SIZE, -1] = 1.0                    # opp active flag
    ctx.pokemon_part[:, TEAM_SIZE, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
    ctx.opp_believed_mask[:, :] = True
    ctx.opp_believed_mask[:, 0] = False
    return ar


def _direct_marginal(op):
    """The hand recomputation of the Species-Clause marginal for the pinned board: the usage
    prior with Tyranitar (the one revealed opp species) zeroed, renormalized."""
    p = op.SPECIES_USAGE_PRIOR.clone()
    p[_TTAR.num] = 0.0
    return p / p.sum()


def test_unrevealed_marginal_is_the_species_clause_prior():
    fe = _make(**_D1_TOGGLES).eval()
    ctx = _ctx(fe)
    _pin_scenario(ctx)
    with torch.no_grad():
        sp = fe.damage_op.unrevealed_species_probs(ctx)          # [B,S] — ONE marginal per battle
    # The prior path returns [B, n_species]: the Species-Clause marginal is identical for all
    # six slots, and the [B,6,S] expand mis-vectorized under Inductor (the compile-precedent
    # note on the method) — consumers broadcast the RESULTS. A learned override keeps [B,6,S].
    assert sp.dim() == 2
    expect = _direct_marginal(fe.damage_op)
    assert torch.allclose(sp, expect.expand(ctx.batch_size, -1), atol=1e-6)
    # Species Clause: the revealed species carries ZERO mass; the marginal still sums to 1.
    assert float(sp[:, _TTAR.num].abs().max()) == 0.0
    assert torch.allclose(sp.sum(-1), torch.ones(ctx.batch_size), atol=1e-5)
    # The sentinel species (num 0) never carries mass.
    assert float(sp[..., 0].abs().max()) == 0.0
    # A dominant non-revealed species keeps real mass (the prior is not the flat floor).
    assert float(sp[0, _SNORLAX.num]) > 0.01


def test_unrevealed_cell_matches_direct_table_recompute():
    """Pin the physics: the unrevealed D1 cell equals an independent recomputation from the op's
    own tables — Body Slam (neutral Snorlax, STAB 1.5) into the E[mult]/E[def]/E[maxhp] marginal
    defender. pko NULLED, revealed channel 0."""
    fe = _make(**_D1_TOGGLES).eval()
    ctx = _ctx(fe)
    _pin_scenario(ctx)
    op = fe.damage_op
    with torch.no_grad():
        cells = op.pairwise_outgoing(ctx)                        # [B,4,6,6]
    p = _direct_marginal(op)
    eps = 1e-6
    atk = 2.0 * float(_SNORLAX.base_stats["atk"]) + 31.0 + 5.0   # neutral spread, no CB/burn/boost
    bp = float(op.MOVE_BP[_BODYSLAM.num])
    e_def = float(p @ op.SPECIES_SPREAD_PRIOR[:, _SB_DEF, 0])
    e_maxhp = 2.0 * float(p @ op.BASE_STATS[:, 0]) + 31.0 + 110.0
    e_mult = float(p @ op.SPECIES_EXP_MULT[:, _T2I["NORMAL"]])
    core = 42.0 * bp * atk / (e_def + eps) / 50.0 + 2.0
    dmg = core * 1.5 * e_mult * 0.925                            # STAB, no weather/screens
    inv = 1.0 / (e_maxhp + eps)
    want_high = min(dmg * inv, _DMG_CHIP_CAP)
    want_low = min(_DMG_ROLL_MIN * dmg * inv, _DMG_CHIP_CAP)
    for j in range(1, TEAM_SIZE):
        low, high, crit, pko, mult, revealed = (float(x) for x in cells[0, 0, j])
        assert high > 0.0 and low > 0.0, "the §4.1 guard: expected-latent cells must be non-zero"
        assert abs(high - want_high) < 1e-4, f"slot {j}: high {high} != recompute {want_high}"
        assert abs(low - want_low) < 1e-4
        assert abs(mult - e_mult) < 1e-5, "the type_mult channel must carry E[mult]"
        assert pko == 0.0, "P(KO) must stay NULLED for an unrevealed slot"
        assert revealed == 0.0, "the certainty channel must stay 0 (magnitudes, not epistemics)"
    # The revealed TTar column stays a real revealed read (revealed bit 1, chart eff not E[mult]).
    assert float(cells[0, 0, 0, 5]) == 1.0
    # Non-legal move rows stay zero (ch 5 = the per-slot revealed bit, expanded over every move row).
    assert float(cells[:, 1:, :, :5].abs().sum()) == 0.0


def test_revealed_columns_byte_identical_when_slots_flip_unrevealed():
    fe = _make(**_D1_TOGGLES).eval()
    ctx = _ctx(fe, seed=83)
    # a fully-revealed random board with a guaranteed-live slot 3 and one legal damaging move
    ctx.our_active_req_move_ids[:, 0] = _BODYSLAM.num
    ctx.our_active_req_move_type_ids[:, 0] = _T2I["NORMAL"]
    ctx.our_active_req_move_legal[:, 0] = 1.0
    ctx.hp_and_active[:, TEAM_SIZE + 3, 0] = 0.7
    ctx.hp_and_active[:, TEAM_SIZE + 4, 0] = 0.7
    ctx.opp_believed_mask[:, :] = False
    with torch.no_grad():
        a = fe.damage_op.pairwise_outgoing(ctx)
    ctx.opp_believed_mask[:, 3] = True
    ctx.opp_believed_mask[:, 4] = True
    with torch.no_grad():
        b = fe.damage_op.pairwise_outgoing(ctx)
    keep = [0, 1, 2, 5]
    assert torch.equal(a[:, :, keep, :], b[:, :, keep, :]), \
        "flipping slots 3/4 unrevealed must change ONLY those slots' columns"
    assert not torch.equal(a, b), "the flipped slots must actually change"
    # the flipped slots: revealed channel + pko forced to 0
    assert float(b[:, :, 3:5, 5].abs().sum()) == 0.0
    assert float(b[:, :, 3:5, 3].abs().sum()) == 0.0


def test_forced_alive_obs_hp_zero_still_priced():
    """An unrevealed slot's obs hp is a 0 placeholder — the expected-latent read must force it
    alive (full-HP switch-in), and must IGNORE whatever hp value the obs happens to hold."""
    fe = _make(**_D1_TOGGLES).eval()
    ctx = _ctx(fe)
    _pin_scenario(ctx)                                           # hidden slots already at hp 0
    with torch.no_grad():
        at_zero = fe.damage_op.pairwise_outgoing(ctx)
    assert float(at_zero[0, 0, 1:, 1].min()) > 0.0, "hp-0 unrevealed slots must still be priced"
    ctx.hp_and_active[:, TEAM_SIZE + 1:, 0] = 0.5                # a nonsense placeholder value
    with torch.no_grad():
        at_half = fe.damage_op.pairwise_outgoing(ctx)
    assert torch.equal(at_zero, at_half), "the obs hp placeholder must not leak into hidden slots"


def test_species_probs_override_wins():
    """The learned-belief seam: a one-hot species_probs replaces the usage prior entirely —
    the type_mult channel reads that species' own expected multiplier."""
    fe = _make(**_D1_TOGGLES).eval()
    ctx = _ctx(fe)
    _pin_scenario(ctx)
    op = fe.damage_op
    B = ctx.batch_size
    one_hot = torch.zeros(B, TEAM_SIZE, op.SPECIES_USAGE_PRIOR.shape[0])
    one_hot[:, :, _SKARM.num] = 1.0
    with torch.no_grad():
        cells = op.pairwise_outgoing(ctx, species_probs=one_hot)
    want = float(op.SPECIES_EXP_MULT[_SKARM.num, _T2I["NORMAL"]])   # Steel resist ≈ 0.5
    assert want < 0.75, "fixture: Normal into Skarmory must read a resist"
    for j in range(1, TEAM_SIZE):
        assert abs(float(cells[0, 0, j, 4]) - want) < 1e-5
