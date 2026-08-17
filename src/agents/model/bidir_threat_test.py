"""Unit tests for the DamageOperator's discrete OUTGOING + STATUS kernels
(gen3_bidir_threat_trunk_v1 / gen3_status_trunk_v1) and `threat_prob_outspeed`.

    (E[mult] via SPECIES_EXP_MULT, P(KO) nulled) when `species_probs` is supplied
  * `discrete_incoming_status` / `discrete_outgoing_status` — per-defender
    [P(major), P(immobilize)] (the physics the S1/S3 edge families deliver)
  * `threat_prob_outspeed` — uncertainty-aware P(outspeed) (÷ believed speed std)

Correctness is exercised on hand-built ctxs (controllable revealed/unrevealed slots + belief),
so the kernels are pinned independently of which consumer happens to read them.
"""
import types

import torch

from agents import gen3_data
from agents.model.features_extractor import (
    DamageOperator, TEAM_SIZE, _DMG_SPEED_SCALE,
    _DMG_STATUS_REFINE,
)
from agents.observation.constants import (
    POKEMON_SPREAD_OFFSET, POKEMON_FULL_DIM, ACTIVE_CONTEXT_DIM)
from agents.observation.types import TypeEncoder
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_T2I = TypeEncoder.TYPE_TO_IDX


def _op():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return DamageOperator(layout, prob_outspeed=False), layout


def _op_prob():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return DamageOperator(layout, prob_outspeed=True), layout


def _num(species_id):
    return gen3_data.species.get(species_id).num


def _ctx_out(*, our_species, our_t1, our_t2, our_moves, our_move_types, move_mask,
             opp_defenders, believed, our_alive=True, B=1):
    """ctx for discrete_outgoing. our active = slot 0; opp 6 = slots TEAM_SIZE.. .
    opp_defenders: list of 6 (species_num, t1, t2); believed: list of 6 bool (True = UNREVEALED).
    The opp ACTIVE is slot TEAM_SIZE+0 (its active flag set). Spread = IV31/EV0/neutral, full HP."""
    n = 2 * TEAM_SIZE
    species = torch.zeros(B, n, dtype=torch.long)
    t1 = torch.zeros(B, n, dtype=torch.long)
    t2 = torch.zeros(B, n, dtype=torch.long)
    species[:, 0] = our_species
    t1[:, 0] = our_t1
    t2[:, 0] = our_t2
    for i, (num, a, b) in enumerate(opp_defenders):
        species[:, TEAM_SIZE + i] = num
        t1[:, TEAM_SIZE + i] = a
        t2[:, TEAM_SIZE + i] = b
    hp = torch.zeros(B, n, POKEMON_FULL_DIM)
    hp[:, 0, 0] = 1.0 if our_alive else 0.0                 # our active HP
    hp[:, TEAM_SIZE, -1] = 1.0                               # opp active flag (slot 0)
    # revealed opp slots full HP; unrevealed slots left 0 (the kernel force-revives them)
    for i, bel in enumerate(believed):
        hp[:, TEAM_SIZE + i, 0] = 0.0 if bel else 1.0
    sp = torch.zeros(B, n, POKEMON_FULL_DIM)
    spv = sp[:, :, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]
    spv[..., 0:6] = 1.0      # IV 31
    spv[..., 13:18] = 1.0    # neutral nature
    amid = torch.zeros(B, n, 4, dtype=torch.long)
    amty = torch.zeros(B, n, 4, dtype=torch.long)
    for k, (mid, mty) in enumerate(zip(our_moves, our_move_types)):
        amid[:, 0, k] = mid
        amty[:, 0, k] = mty
    return types.SimpleNamespace(
        batch_size=B, device=torch.device("cpu"),
        our_active_idx=torch.zeros(B, dtype=torch.long),
        opp_active_local=torch.zeros(B, dtype=torch.long),
        species_ids=species, type1_ids=t1, type2_ids=t2, ability1_ids=torch.zeros(B, n, dtype=torch.long),
        item_ids=torch.zeros(B, n, dtype=torch.long),
        hp_and_active=hp, pokemon_part=sp,
        all_move_ids=amid, all_move_type_ids=amty,
        move_mask=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        # gen3_op_move_align_v1: the op reads the request-ordered slice; the test writes the active's
        # moves to slot 0 in request order, so these ARE slot 0's moves; legal == the request-order mask.
        our_active_req_move_ids=amid[:, 0, :],
        our_active_req_move_type_ids=amty[:, 0, :],
        our_active_req_move_legal=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        screen_feature=torch.zeros(B, 8),
        our_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM), opp_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM),
        weather_feature=torch.zeros(B, 7),
        opp_believed_mask=torch.tensor([[bool(x) for x in believed]] * B),
    )


def _eq():
    return gen3_data.moves.get("earthquake")   # Ground, physical, BP 100


# --------------------------------------------------------------------- #3 probabilistic outspeed
def test_prob_outspeed_bounds_and_differs():
    op_fixed, _ = _op()
    op_prob, _ = _op_prob()
    our = torch.tensor([200.0, 100.0, 300.0])
    opp = torch.tensor([180.0, 180.0, 180.0])
    std = torch.tensor([40.0, 40.0, 40.0])
    p_fixed = op_fixed._p_outspeed(our, opp, std)            # ignores std (prob_outspeed off)
    p_prob = op_prob._p_outspeed(our, opp, std)
    assert torch.all((p_prob >= 0.0) & (p_prob <= 1.0))
    # fixed path == legacy sigmoid over the gap / fixed scale
    assert torch.allclose(p_fixed, torch.sigmoid((our - opp) / _DMG_SPEED_SCALE))
    # uncertainty-aware path differs (wider std → softer toward 0.5 for the same gap)
    assert not torch.allclose(p_prob, p_fixed)
    # monotonic in our speed, and ~0.5 when speeds tie
    assert p_prob[2] > p_prob[0] > p_prob[1]
    tie = op_prob._p_outspeed(torch.tensor([180.0]), torch.tensor([180.0]), torch.tensor([40.0]))
    assert abs(tie.item() - 0.5) < 1e-5


# ------------------------------------------------------------------------ gradient flow
def _one_hot_species(layout, slot_species, B=1):
    """species_probs [B,6,n_species] one-hot per slot on the given species nums (None = uniform-ish 0)."""
    n_sp = layout["max_species"]
    p = torch.zeros(B, TEAM_SIZE, n_sp)
    for i, num in enumerate(slot_species):
        if num is not None:
            p[:, i, num] = 1.0
    return p


def test_threat_grad_flows_to_a_consumer_and_the_species_belief():
    """The expected-latent defender read is differentiable in P(species): a backward through a
    consumer projection reaches BOTH the projection's weight AND the species probs — so a loss
    reading it sharpens the species belief, the whole point of the expected-latent read. Pinned
    on the LIVE consumer (`_outgoing_matrix`'s unrevealed pricing,
    gen3_unrevealed_outgoing_prior_v1) since the standalone `discrete_outgoing` kernel was
    deleted with the refine loop's other orphans (`gen3_op_dead_kernel_cleanup_v1`)."""
    op, layout = _op()
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])] + [(0, 0, 0)] * 5,
                   believed=[False, True, True, True, True, True])
    sp = _one_hot_species(layout, [None, _num("magneton")] + [None] * 4).clone().requires_grad_(True)
    out = op._outgoing_matrix(ctx, species_probs=sp)             # [1, _DMG_OMX]
    proj = torch.nn.Linear(out.shape[1], 128)                    # stand-in consumer (non-zero init)
    proj(out).sum().backward()
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum().item() > 0.0
    assert sp.grad is not None and sp.grad.abs().sum().item() > 0.0   # gradient reaches P(species)


# ===================================================================== v37 status-landing into the trunk
def _ctx_status(*, our_defenders, opp_defenders, our_moves, our_move_types, move_mask,
                opp_believed=None, B=1):
    """ctx for the status kernels. our_defenders / opp_defenders = list of 6 (species,t1,t2). our active =
    slot 0 (its 4 status moves); opp active = slot TEAM_SIZE. No conditions / no sub / hp_probs zero."""
    n = 2 * TEAM_SIZE
    species = torch.zeros(B, n, dtype=torch.long)
    t1 = torch.zeros(B, n, dtype=torch.long); t2 = torch.zeros(B, n, dtype=torch.long)
    for i, (num, a, b) in enumerate(our_defenders):
        species[:, i] = num; t1[:, i] = a; t2[:, i] = b
    for i, (num, a, b) in enumerate(opp_defenders):
        species[:, TEAM_SIZE + i] = num; t1[:, TEAM_SIZE + i] = a; t2[:, TEAM_SIZE + i] = b
    hp = torch.zeros(B, n, POKEMON_FULL_DIM)
    hp[:, :TEAM_SIZE, 0] = 1.0                       # our mons alive
    hp[:, TEAM_SIZE, -1] = 1.0                        # opp active flag (slot 0)
    believed = opp_believed if opp_believed is not None else [False] * 6
    for i, bel in enumerate(believed):
        hp[:, TEAM_SIZE + i, 0] = 0.0 if bel else 1.0
    sp = torch.zeros(B, n, POKEMON_FULL_DIM)
    spv = sp[:, :, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]; spv[..., 0:6] = 1.0; spv[..., 13:18] = 1.0
    amid = torch.zeros(B, n, 4, dtype=torch.long); amty = torch.zeros(B, n, 4, dtype=torch.long)
    for k, (mid, mty) in enumerate(zip(our_moves, our_move_types)):
        amid[:, 0, k] = mid; amty[:, 0, k] = mty
    return types.SimpleNamespace(
        batch_size=B, device=torch.device("cpu"),
        our_active_idx=torch.zeros(B, dtype=torch.long), opp_active_local=torch.zeros(B, dtype=torch.long),
        species_ids=species, type1_ids=t1, type2_ids=t2, ability1_ids=torch.zeros(B, n, dtype=torch.long),
        item_ids=torch.zeros(B, n, dtype=torch.long), hp_and_active=hp, pokemon_part=sp,
        all_move_ids=amid, all_move_type_ids=amty,
        move_mask=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        # gen3_op_move_align_v1: request-ordered slice (== slot 0's moves, the test's request order).
        our_active_req_move_ids=amid[:, 0, :],
        our_active_req_move_type_ids=amty[:, 0, :],
        our_active_req_move_legal=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        hp_probs=torch.zeros(B, n, 16), opp_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM),
        opp_believed_mask=torch.tensor([[bool(x) for x in believed]] * B),
    )


def _belief_on(move_num, layout, B=1):
    """move_belief_logits [B,6,n_moves] putting ~all mass on `move_num` at the opp active (slot 0)."""
    lg = torch.full((B, TEAM_SIZE, layout["max_moves"]), -10.0)
    lg[:, 0, move_num] = 10.0
    return lg


def test_incoming_status_type_immunity_and_split():
    """#INCOMING: opp believed Thunder Wave (paralysis). Our GROUND mon reads 0 (Ground immune to T-Wave);
    a Water mon reads major>0 AND immobilize>0 (paralysis is action-denying)."""
    op, layout = _op()
    twave = gen3_data.moves.get("thunderwave").num
    ctx = _ctx_status(our_defenders=[(_num("flygon"), _T2I["GROUND"], _T2I["DRAGON"]),    # Ground → T-Wave immune
                                     (_num("starmie"), _T2I["WATER"], _T2I["PSYCHIC"])] + [(0, 0, 0)] * 4,
                      opp_defenders=[(_num("zapdos"), _T2I["ELECTRIC"], _T2I["FLYING"])] + [(0, 0, 0)] * 5,
                      our_moves=[0, 0, 0, 0], our_move_types=[0, 0, 0, 0], move_mask=[0, 0, 0, 0])
    out = op.discrete_incoming_status(ctx, _belief_on(twave, layout))     # [1,6,2]
    assert out.shape == (1, TEAM_SIZE, _DMG_STATUS_REFINE)
    assert out[0, 0, 0].item() == 0.0 and out[0, 0, 1].item() == 0.0      # Ground: immune (major & immob 0)
    assert out[0, 1, 0].item() > 0.0 and out[0, 1, 1].item() > 0.0        # Water: para lands → major & immob >0


def test_incoming_status_toxic_is_major_not_immobilize():
    """Toxic on a non-immune mon: major>0 but immobilize==0 (poison chips, doesn't deny the action)."""
    op, layout = _op()
    toxic = gen3_data.moves.get("toxic").num
    ctx = _ctx_status(our_defenders=[(_num("starmie"), _T2I["WATER"], _T2I["PSYCHIC"])] + [(0, 0, 0)] * 5,
                      opp_defenders=[(_num("zapdos"), _T2I["ELECTRIC"], _T2I["FLYING"])] + [(0, 0, 0)] * 5,
                      our_moves=[0, 0, 0, 0], our_move_types=[0, 0, 0, 0], move_mask=[0, 0, 0, 0])
    out = op.discrete_incoming_status(ctx, _belief_on(toxic, layout))
    assert out[0, 0, 0].item() > 0.0       # major (toxic lands on Water/Psychic)
    assert out[0, 0, 1].item() == 0.0      # NOT immobilize


def test_outgoing_status_type_immunity_and_revealed_gate():
    """#OUTGOING: our Thunder Wave vs the opp active. A Ground opp reads 0 (immune); a Water opp reads >0.
    An UNREVEALED opp slot is zeroed (revealed-gated)."""
    op, layout = _op()
    twave = gen3_data.moves.get("thunderwave").num
    # opp active = Ground (immune); opp bench slot 1 = Water but UNREVEALED
    ctx = _ctx_status(our_defenders=[(_num("zapdos"), _T2I["ELECTRIC"], _T2I["FLYING"])] + [(0, 0, 0)] * 5,
                      opp_defenders=[(_num("flygon"), _T2I["GROUND"], _T2I["DRAGON"]),
                                     (_num("starmie"), _T2I["WATER"], _T2I["PSYCHIC"])] + [(0, 0, 0)] * 4,
                      our_moves=[twave, 0, 0, 0], our_move_types=[_T2I["ELECTRIC"], 0, 0, 0],
                      move_mask=[1, 0, 0, 0], opp_believed=[False, True, False, False, False, False])
    out = op.discrete_outgoing_status(ctx)         # [1,6,2]
    assert out[0, 0, 0].item() == 0.0              # Ground opp active: T-Wave immune
    assert out[0, 1, :].abs().sum().item() == 0.0  # Water bench but UNREVEALED → revealed-gated to 0
    # now reveal the Water bench mon → it should read para landing
    ctx2 = _ctx_status(our_defenders=[(_num("zapdos"), _T2I["ELECTRIC"], _T2I["FLYING"])] + [(0, 0, 0)] * 5,
                       opp_defenders=[(_num("flygon"), _T2I["GROUND"], _T2I["DRAGON"]),
                                      (_num("starmie"), _T2I["WATER"], _T2I["PSYCHIC"])] + [(0, 0, 0)] * 4,
                       our_moves=[twave, 0, 0, 0], our_move_types=[_T2I["ELECTRIC"], 0, 0, 0],
                       move_mask=[1, 0, 0, 0], opp_believed=[False, False, False, False, False, False])
    out2 = op.discrete_outgoing_status(ctx2)
    assert out2[0, 1, 0].item() > 0.0 and out2[0, 1, 1].item() > 0.0   # revealed Water: para lands


def test_status_grad_flows():
    """A backward through outgoing_status/incoming_status reaches both consumer projections (a trunk
    consumer can learn to value the computed status fact)."""
    op, layout = _op()
    twave = gen3_data.moves.get("thunderwave").num
    ctx = _ctx_status(our_defenders=[(_num("zapdos"), _T2I["ELECTRIC"], _T2I["FLYING"])] + [(0, 0, 0)] * 5,
                      opp_defenders=[(_num("starmie"), _T2I["WATER"], _T2I["PSYCHIC"])] + [(0, 0, 0)] * 5,
                      our_moves=[twave, 0, 0, 0], our_move_types=[_T2I["ELECTRIC"], 0, 0, 0], move_mask=[1, 0, 0, 0])
    proj_in = torch.nn.Linear(_DMG_STATUS_REFINE, 128)
    proj_out = torch.nn.Linear(_DMG_STATUS_REFINE, 128)
    fi = op.discrete_incoming_status(ctx, _belief_on(twave, layout))
    fo = op.discrete_outgoing_status(ctx)
    (proj_in(fi).sum() + proj_out(fo).sum()).backward()
    assert proj_in.weight.grad.abs().sum().item() > 0.0
    assert proj_out.weight.grad.abs().sum().item() > 0.0
