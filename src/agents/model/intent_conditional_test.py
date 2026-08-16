"""gen3_intent_conditional_v1 (v85) — the remaining α-conditioned mechanic cells' gates.

What must hold (design_conditional_execution.md §§3.1/3.6/3.7/3.8 + the house rules):
  * Counter's value is the α-weighted PHYSICAL-damaging sum and nothing else — a special or
    status read feeds Mirror Coat / neither ("there is no safe Counter");
  * flinch is worthless into a switch — the (1−α_SWITCH) term the raw chance was missing;
  * Explosion's p_executes reads the Protect-family seats; its into-switch mass is α_SWITCH;
  * Pursuit follows the PORT-VERIFIED rule (departing target, ×2 never-miss) — the trigger is
    α_SWITCH and the bonus scales with the slot's own outgoing damage; no β anywhere;
  * seat-permutation invariance, the width fail-loud, zero-init on the projection;
  * OFF builds nothing; the v85 version machinery holds.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import INTENT_COND_MOVE_DIM
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.intent_conditional import IntentConditionalMoveCell
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_ON_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_matrices_outgoing=True,
    damage_topk_k=6, entity_topk_seats=6, opp_intent=True, intent_conditional=True,
)

_COUNTER, _MC, _EXPL, _SD, _PURSUIT = 68, 243, 153, 120, 228


def _build(**kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(7)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs)
    fe.eval()
    return fe, layout


def _obs(layout, b=3):
    torch.manual_seed(11)
    return {"observation": torch.rand(b, layout["total_dim"])}


def _identity_cell():
    m = IntentConditionalMoveCell(13)
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(13))
    return m


def _args(K=3, *, alpha_logits=None, high=None, is_phys=None, topk_nums=None,
          out_high=None, p_outspeed=1.0, sec_flinch=0.0, req_ids=(0, 0, 0, 0),
          protect_odds=0.5, beta_logits=None, out_pko=None):
    """Hand-built forward args for one batch row."""
    cells = torch.zeros(1, TEAM_SIZE, K, 6)
    cells[..., 1] = torch.tensor(high if high is not None else [0.4] * K)
    cells[..., 5] = torch.tensor(is_phys if is_phys is not None else [1.0] * K)
    gate = torch.ones(1, TEAM_SIZE, 1)
    active = torch.zeros(1, dtype=torch.long)
    lg = alpha_logits if alpha_logits is not None else torch.zeros(1, K + 1)
    nums = torch.tensor([topk_nums if topk_nums is not None else [89] * K])
    oh = torch.tensor([out_high if out_high is not None else [0.3] * 4])
    return (lg, cells, gate, active, nums, oh,
            torch.full((1, 1), float(p_outspeed)),
            torch.full((1, 4), float(sec_flinch)),
            torch.tensor([list(req_ids)]),
            torch.full((1, 1), float(protect_odds)),
            beta_logits if beta_logits is not None else torch.full((1, TEAM_SIZE), float("-inf")),
            out_pko if out_pko is not None else torch.zeros(1, 4, TEAM_SIZE),
            torch.zeros(1, dtype=torch.long))


def test_counter_reads_only_the_physical_damaging_sum():
    m = _identity_cell()
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0                                    # α ≈ all on seat 0
    # seat 0 physical, 0.6 damage → counter return 0.6, mc 0
    out = m(*_args(alpha_logits=lg, high=[0.6, 0.2, 0.2], is_phys=[1, 0, 0],
                   req_ids=(_COUNTER, _MC, 0, 0)))
    assert torch.allclose(out[0, 0, 0], torch.tensor(0.6), atol=1e-3)   # counter slot
    assert float(out[0, 1, 1].abs()) < 1e-3                              # mc slot: no special read
    assert torch.allclose(out[0, 0, 2], torch.tensor(1.0), atol=1e-3)   # cat match ≈ 1
    # α flipped to the SPECIAL seat: counter dies, mirror coat lives
    lg2 = torch.zeros(1, 4)
    lg2[:, 1] = 10.0
    out2 = m(*_args(alpha_logits=lg2, high=[0.6, 0.5, 0.2], is_phys=[1, 0, 0],
                    req_ids=(_COUNTER, _MC, 0, 0)))
    assert float(out2[0, 0, 0].abs()) < 1e-3
    assert torch.allclose(out2[0, 1, 1], torch.tensor(0.5), atol=1e-3)


def test_a_status_read_feeds_neither_counter_nor_mirror_coat():
    """high == 0 (status/immune) is not a damaging candidate in either category."""
    m = _identity_cell()
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0
    out = m(*_args(alpha_logits=lg, high=[0.0, 0.4, 0.4], is_phys=[0, 1, 1],
                   req_ids=(_COUNTER, _MC, 0, 0)))
    assert float(out[0, :2, :3].abs().max()) < 1e-3


def test_flinch_is_worthless_into_a_switch():
    m = _identity_cell()
    stay = m(*_args(alpha_logits=_switchless(3), p_outspeed=0.8, sec_flinch=0.3))
    certain_switch = torch.zeros(1, 4)
    certain_switch[:, -1] = 20.0
    switch = m(*_args(alpha_logits=certain_switch, p_outspeed=0.8, sec_flinch=0.3))
    assert torch.allclose(stay[..., 3], torch.full((1, 4), 0.8 * 0.3), atol=1e-3)   # α_SWITCH ≈ 0
    assert float(switch[..., 3].abs().max()) < 1e-3


def _switchless(K):
    lg = torch.full((1, K + 1), 0.0)
    lg[:, -1] = -20.0                                  # α_SWITCH ≈ 0, uniform over K seats
    return lg


def test_explosion_reads_the_protect_family_and_the_switch_mass():
    m = _identity_cell()
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0                                    # α ≈ all on seat 0 = Protect
    out = m(*_args(alpha_logits=lg, topk_nums=[182, 89, 89],
                   req_ids=(_EXPL, _SD, 0, 0)))
    assert float(out[0, 0, 4]) < 1e-3                  # p_executes ≈ 0 into a certain Protect
    assert float(out[0, 1, 4]) < 1e-3                  # selfdestruct gated identically
    certain_switch = torch.zeros(1, 4)
    certain_switch[:, -1] = 20.0
    out2 = m(*_args(alpha_logits=certain_switch, topk_nums=[182, 89, 89],
                    req_ids=(_EXPL, 0, 0, 0)))
    assert torch.allclose(out2[0, 0, 4], torch.tensor(1.0), atol=1e-3)  # switch ⇒ it lands
    assert torch.allclose(out2[0, 0, 5], torch.tensor(1.0), atol=1e-3)  # …on an arrival


def test_pursuit_trigger_and_bonus_ride_alpha_switch_only():
    m = _identity_cell()
    certain_switch = torch.zeros(1, 4)
    certain_switch[:, -1] = 20.0
    out = m(*_args(alpha_logits=certain_switch, out_high=[0.35, 0, 0, 0],
                   req_ids=(_PURSUIT, 0, 0, 0)))
    assert torch.allclose(out[0, 0, 6], torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(out[0, 0, 7], torch.tensor(0.35), atol=1e-3)
    out2 = m(*_args(alpha_logits=_switchless(3), out_high=[0.35, 0, 0, 0],
                    req_ids=(_PURSUIT, 0, 0, 0)))
    assert float(out2[0, 0, 6:].abs().max()) < 1e-3


def test_seat_permutation_invariance():
    torch.manual_seed(5)
    K = 4
    m = _identity_cell()
    cells = torch.rand(1, TEAM_SIZE, K, 6)
    gate = torch.ones(1, TEAM_SIZE, 1)
    active = torch.zeros(1, dtype=torch.long)
    nums = torch.randint(1, 300, (1, K))
    lg = torch.randn(1, K + 1)
    rest = (torch.rand(1, 4), torch.rand(1, 1), torch.rand(1, 4),
            torch.tensor([[_COUNTER, _MC, _EXPL, _PURSUIT]]), torch.rand(1, 1),
            torch.randn(1, TEAM_SIZE), torch.rand(1, 4, TEAM_SIZE),
            torch.zeros(1, dtype=torch.long))
    perm = torch.randperm(K)
    lg_p = torch.cat([lg[:, perm], lg[:, -1:]], dim=1)
    a = m(lg, cells, gate, active, nums, *rest)
    b = m(lg_p, cells[:, :, perm], gate, active, nums[:, perm], *rest)
    assert torch.allclose(a, b, atol=1e-6)


def test_axis_width_mismatch_raises():
    m = IntentConditionalMoveCell(INTENT_COND_MOVE_DIM)
    args = list(_args(K=3))
    args[0] = torch.zeros(1, 6)                        # 5 seats vs 3 candidate columns
    with pytest.raises(ValueError, match="SAME axis"):
        m(*args)


def test_zero_init_contributes_nothing():
    m = IntentConditionalMoveCell(INTENT_COND_MOVE_DIM)
    assert float(m.proj.weight.abs().max()) == 0.0
    out = m(*_args(req_ids=(_COUNTER, _MC, _EXPL, _PURSUIT), sec_flinch=0.3))
    assert float(out.abs().max()) == 0.0


def test_protect_carries_the_avoided_quantities_decorrelated():
    """Step 5 (§3.3): c4 carried the p_success multiplier and omitted the quantity — here the
    α-weighted avoided damage, the mechanical odds, and the α status mass ride separately."""
    m = _identity_cell()
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0                                    # α ≈ all on seat 0
    # seat 0 = a damaging 0.6 hit → avoided damage 0.6, status mass 0
    out = m(*_args(alpha_logits=lg, high=[0.6, 0.0, 0.0], topk_nums=[89, 92, 92],
                   req_ids=(182, 197, 0, 0), protect_odds=0.25))
    assert torch.allclose(out[0, 0, 8], torch.tensor(0.6), atol=1e-3)   # protect: dmg avoided
    assert torch.allclose(out[0, 1, 8], torch.tensor(0.6), atol=1e-3)   # detect gated too
    assert torch.allclose(out[0, 0, 9], torch.tensor(0.25), atol=1e-6)  # the mechanical odds
    assert float(out[0, 0, 10].abs()) < 1e-3                            # no status mass
    # α flipped to seat 1 = Toxic (a STATUS move): dmg avoided 0, status mass 1
    lg2 = torch.zeros(1, 4)
    lg2[:, 1] = 10.0
    out2 = m(*_args(alpha_logits=lg2, high=[0.6, 0.0, 0.0], topk_nums=[89, 92, 92],
                    req_ids=(182, 0, 0, 0)))
    assert float(out2[0, 0, 8].abs()) < 1e-3
    assert torch.allclose(out2[0, 0, 10], torch.tensor(1.0), atol=1e-3)
    # a non-protect slot's channels stay zero; ENDURE (203) is deliberately not in this gate
    out3 = m(*_args(alpha_logits=lg, high=[0.6, 0, 0], req_ids=(203, 89, 0, 0)))
    assert float(out3[..., 8:].abs().max()) < 1e-6


def test_status_table_types_by_data_not_by_damage():
    """An immune damaging seat (high == 0) must NOT read as a status seat."""
    m = _identity_cell()
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0
    # seat 0 = Earthquake (num 89, DAMAGING) with high 0 (immune defender)
    out = m(*_args(alpha_logits=lg, high=[0.0, 0.0, 0.0], topk_nums=[89, 89, 89],
                   req_ids=(182, 0, 0, 0)))
    assert float(out[0, 0, 10].abs()) < 1e-3           # not status — just immune


def test_magic_coat_reads_only_the_oracle_verified_reflectable_set():
    """§3.12 + the G0 oracle: foe-targeting status (Toxic 92) is reflectable; Spikes (191,
    side-targeting) and a damaging move are NOT — the exact sim-verified boundary."""
    m = _identity_cell()
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0
    mcoat = 277
    hit = m(*_args(alpha_logits=lg, topk_nums=[92, 89, 89], req_ids=(mcoat, 0, 0, 0)))
    assert torch.allclose(hit[0, 0, 11], torch.tensor(1.0), atol=1e-3)   # Toxic bounces
    spikes = m(*_args(alpha_logits=lg, topk_nums=[191, 89, 89], req_ids=(mcoat, 0, 0, 0)))
    assert float(spikes[0, 0, 11].abs()) < 1e-3                          # Spikes does NOT
    dmg = m(*_args(alpha_logits=lg, topk_nums=[89, 89, 89], req_ids=(mcoat, 0, 0, 0)))
    assert float(dmg[0, 0, 11].abs()) < 1e-3


def test_boom_trade_ko_follows_the_branch_target():
    """§3.1's β half: on a certain STAY the trade KO reads the ACTIVE's pko; on a certain
    SWITCH it reads the β-weighted ARRIVAL's — the target differs by branch, which is the
    whole reason β had to be published."""
    m = _identity_cell()
    pko = torch.zeros(1, 4, TEAM_SIZE)
    pko[0, 0, 0] = 0.9                                 # boom slot vs their active (local 0)
    pko[0, 0, 3] = 0.2                                 # boom slot vs bench mon 3
    beta = torch.full((1, TEAM_SIZE), float("-inf"))
    beta[0, 3] = 0.0                                   # β certain on bench mon 3
    stay = torch.zeros(1, 4)
    stay[:, 0] = 20.0                                  # α ≈ all on a move seat
    out = m(*_args(alpha_logits=stay, req_ids=(_EXPL, 0, 0, 0), beta_logits=beta, out_pko=pko))
    assert torch.allclose(out[0, 0, 12], torch.tensor(0.9), atol=1e-3)
    switch = torch.zeros(1, 4)
    switch[:, -1] = 20.0                               # α ≈ all on SWITCH
    out2 = m(*_args(alpha_logits=switch, req_ids=(_EXPL, 0, 0, 0), beta_logits=beta, out_pko=pko))
    assert torch.allclose(out2[0, 0, 12], torch.tensor(0.2), atol=1e-3)
    # no legal switch-in (all -inf): the arrival term contributes exactly 0, never NaN
    out3 = m(*_args(alpha_logits=switch, req_ids=(_EXPL, 0, 0, 0), out_pko=pko))
    assert torch.isfinite(out3).all()
    assert float(out3[0, 0, 12].abs()) < 1e-3


# ------------------------------------------------------------------- extractor wiring


def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**{**_ON_KWARGS, "intent_conditional": False})
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.intent_conditional is None
    assert not any("intent_conditional" in k for k in fe_off.state_dict())
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim + INTENT_COND_MOVE_DIM
    assert fe_on.projection.in_features == fe_off.projection.in_features


def test_on_forward_runs_and_is_in_the_identity_sweep():
    fe, layout = _build(**_ON_KWARGS)
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape
    assert "intent_conditional.proj" in fe._identity_init_zeroed


def test_requires_outgoing():
    with pytest.raises(ValueError, match="damage_outgoing"):
        _build(**{**_ON_KWARGS, "damage_outgoing": False,
                  "damage_matrices_outgoing": False})
    with pytest.raises(ValueError, match="damage_matrices_outgoing"):
        _build(**{**_ON_KWARGS, "damage_matrices_outgoing": False})


def test_full_intent_stack_builds_and_runs():
    """v77 + v84 + v85 all on — the three move-cell wideners must coexist, and the discovery
    forward must size every one (the ede5a88 class, pointer edition)."""
    fe, layout = _build(**{**_ON_KWARGS, "intent_move_cell": True, "intent_threshold": True,
                           "opp_belief_slots": True, "intent_value_reduce": True,
                           "value_entity_pool": True})
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape


# ------------------------------------------------------------------- version machinery


def test_migration_defaults_off():
    migrated = _migrate_config({"config_version": 84})
    assert migrated["intent_conditional"] is False
    assert migrated["config_version"] >= 85
    assert MODEL_CONFIG_VERSION >= 85


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, intent_conditional=True)
    with pytest.raises(ModelVersionError, match="intent_conditional"):
        a.check_compatible(b)
