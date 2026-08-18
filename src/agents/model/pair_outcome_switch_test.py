"""gen3_pair_outcome_switch_v1 (v94) — the SWITCH-cell half's gates.

Phase A delivered the α-reduced unified outcome row to the pointer MOVE cells as context. This is
the delivery `design_pair_reduction.md` §2.1 says the decision actually needs, at the sink it names:

  > The decision *"they will click Will-O-Wisp, so bring the Natural Cure mon"* is made at the
  > **switch logit**. The switch logit's per-action cell contains **no status information at all**.

What must hold:

  * **Contract W survives the per-defender generalization.** `reduce_pair_in_all` produces six rows
    from ONE α; defect D3 (a per-defender α — Skarmory's row assuming "they click Rock Slide" while
    Blissey's assumes "Thunderbolt") stays a SHAPE error, and the planted violation proves it.
  * **The two reducers never drift.** The per-defender reduction at our active row must equal
    Phase A's `reduce_pair_in` exactly, because they are the same contract with a different gather.
  * **`spin_denied` is the conjunction it claims**, on exact arithmetic, and reads 0 when any of
    its three factors is 0 (in particular: with no hazards on their side, denying a spin is worth
    nothing).
  * **The §2.1 case, through the REAL op**: two of OUR mons facing a believed Toxic, identical in
    every damage coordinate, separated by `tempo_cost` in their OWN switch rows.
  * **OFF is byte-identical**; **ON is identity-at-init on a real `MaskablePPO` policy** (M1).
  * The v94 version machinery + the delivery-graph edges.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import (
    PAIR_OUTCOME_SWITCH_DIM, _PAIR_OUTCOME_RAW, _PAIR_OUTCOME_SWITCH_RAW,
)
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.pair_outcome import (
    GHOST_TYPE_IDX, PAIR_OUTCOME_COORDS, PAIR_OUTCOME_IDX, PAIR_OUTCOME_SWITCH_COORDS,
    PAIR_OUTCOME_SWITCH_IDX, PairOutcomeSwitchCell, pair_alpha, pair_alpha_full, rapid_spin_num,
    reduce_pair_in, reduce_pair_in_all,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_BASE_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6,
)
_ON_KWARGS = {**_BASE_KWARGS, "pair_outcome_switch": True}


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


# --------------------------------------------------------------- the coordinate CONTRACT itself


def test_the_switch_coordinate_table_is_the_reduced_row_plus_its_own_extras():
    assert PAIR_OUTCOME_SWITCH_COORDS[:_PAIR_OUTCOME_RAW] == PAIR_OUTCOME_COORDS
    assert PAIR_OUTCOME_SWITCH_COORDS[_PAIR_OUTCOME_RAW:] == ("spin_denied",)
    assert len(PAIR_OUTCOME_SWITCH_COORDS) == _PAIR_OUTCOME_SWITCH_RAW == PAIR_OUTCOME_SWITCH_DIM
    # the reduced row's indices are UNCHANGED by the extension — a consumer that learned
    # PAIR_OUTCOME_IDX in Phase A reads the same columns here.
    for name, i in PAIR_OUTCOME_IDX.items():
        assert PAIR_OUTCOME_SWITCH_IDX[name] == i


def test_the_ghost_type_id_is_resolved_from_the_encoder_not_written_as_a_number():
    """A hard-coded `14` would keep passing after a TypeEncoder renumbering while pricing the wrong
    type as a spinblocker — silently, since the coordinate would still be well-formed."""
    from agents.observation.types import TypeEncoder
    assert GHOST_TYPE_IDX == TypeEncoder.TYPE_TO_IDX["GHOST"]


def test_rapid_spin_resolves_from_the_data_facade():
    from agents import gen3_data
    assert rapid_spin_num() == int(gen3_data.moves.get("rapidspin").num) > 0


# ------------------------------------------------- Contract W, generalized to EVERY defender


def _grid(B=2, K=3, J=TEAM_SIZE, F=_PAIR_OUTCOME_RAW):
    torch.manual_seed(5)
    return torch.rand(B, J, K, F)


def test_reduce_all_is_the_alpha_weighted_sum_at_every_defender():
    K = 3
    pair_in = _grid(K=K)
    gate = torch.ones(2, TEAM_SIZE, 1)
    alpha = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rows = reduce_pair_in_all(alpha, pair_in, gate)
    assert rows.shape == (2, TEAM_SIZE, _PAIR_OUTCOME_RAW)
    for j in range(TEAM_SIZE):
        assert torch.allclose(rows[0, j], pair_in[0, j, 0])
        assert torch.allclose(rows[1, j], pair_in[1, j, 2])


def test_the_two_reducers_agree_at_our_active_row():
    """Phase A's `reduce_pair_in` and Phase B's `reduce_pair_in_all` are ONE contract with two
    gathers. If they ever disagree, the move cell and the switch cell would describe two different
    opponents on the same turn — and nothing about the shapes would say so."""
    K = 4
    pair_in = _grid(B=3, K=K)
    gate = torch.rand(3, TEAM_SIZE, 1)
    active = torch.tensor([0, 2, 5])
    alpha = torch.softmax(torch.randn(3, K), dim=-1)
    one = reduce_pair_in(alpha, pair_in, gate, active)
    allr = reduce_pair_in_all(alpha, pair_in, gate)
    ar = torch.arange(3)
    assert torch.allclose(one, allr[ar, active], atol=1e-6)


def test_a_PER_DEFENDER_alpha_is_a_SHAPE_ERROR_not_a_thing_a_test_hunts_for():
    """THE load-bearing gate (design_pair_reduction.md §2, defect D3).

    The flat block's max is independent per defender, so the "winning" opponent move differs down
    the column — the profile of six mons is an opponent playing six different moves at once. This
    is the phase that produces six rows, so it is the phase where the defect could return; the
    contract kills it by SIGNATURE (α has no J axis), and this plants the violation to prove the
    structure is real rather than merely intended."""
    K, B = 3, 2
    pair_in = _grid(B=B, K=K)
    gate = torch.ones(B, TEAM_SIZE, 1)
    # D3, literally: each defender picks its own worst-case seat.
    per_defender_alpha = torch.zeros(B, TEAM_SIZE, K)
    per_defender_alpha.scatter_(2, pair_in[..., PAIR_OUTCOME_IDX["high"]].argmax(-1, keepdim=True),
                                1.0)
    with pytest.raises(RuntimeError):
        reduce_pair_in_all(per_defender_alpha, pair_in, gate)


def test_reduce_all_seat_width_mismatch_raises():
    with pytest.raises(ValueError, match="SAME axis"):
        reduce_pair_in_all(torch.rand(1, 8), _grid(B=1, K=3), torch.ones(1, TEAM_SIZE, 1))


def test_reduce_all_is_equivariant_in_OUR_team_axis():
    """The switch logits are scored per token, so permuting our team must permute the rows and
    nothing else — α is untouched because it has no defender index."""
    K = 3
    pair_in = _grid(B=1, K=K)
    gate = torch.rand(1, TEAM_SIZE, 1)
    alpha = torch.softmax(torch.randn(1, K), dim=-1)
    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    a = reduce_pair_in_all(alpha, pair_in, gate)[:, perm]
    b = reduce_pair_in_all(alpha, pair_in[:, perm], gate[:, perm])
    assert torch.allclose(a, b, atol=1e-6)


def test_pair_alpha_full_splits_the_publication_without_reassigning_masked_mass():
    """`a_stay` is `Σ_k α_k`, NOT `1 − a_switch`. The two differ exactly when a seat is masked, and
    the sum form is the honest one: an unmodeled seat's usage mass is simply not spent."""
    lg = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    alpha, a_switch, a_stay = pair_alpha_full(lg, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(alpha, torch.tensor([[0.1, 0.2, 0.0]]), atol=1e-6)
    assert float(a_switch) == pytest.approx(0.4, abs=1e-6)
    assert float(a_stay) == pytest.approx(0.3, abs=1e-6), "masked mass was renormalized back in"
    assert float(a_stay + a_switch) < 1.0


def test_pair_alpha_full_is_stop_grad():
    lg = torch.zeros(1, 4, requires_grad=True)
    alpha, a_switch, a_stay = pair_alpha_full(lg)
    for t in (alpha, a_switch, a_stay):
        assert not t.requires_grad, "a policy-side consumer opened a PPO route into alpha_head"


# ------------------------------------------------------------------------------- the switch cell


_SPIN = rapid_spin_num()


def _cell_readable(cell):
    """Set the projection to the identity so the test reads the RAW coordinates. (The shipped
    weight is zero, which is what `test_..._zero_init` asserts separately.)"""
    with torch.no_grad():
        cell.proj.weight.copy_(torch.eye(_PAIR_OUTCOME_SWITCH_RAW))
        cell.proj.bias.zero_()
    return cell


def _switch_case(hazards=2.0 / 3.0, ghost_slots=(2,), spin_seat=1):
    cell = _cell_readable(PairOutcomeSwitchCell(PAIR_OUTCOME_SWITCH_DIM))
    rows = torch.arange(1 * TEAM_SIZE * _PAIR_OUTCOME_RAW, dtype=torch.float32).reshape(
        1, TEAM_SIZE, _PAIR_OUTCOME_RAW)
    alpha = torch.tensor([[0.2, 0.5, 0.1]])
    nums = torch.tensor([[11, 22, 33]])
    nums[0, spin_seat] = _SPIN
    t1 = torch.zeros(1, TEAM_SIZE, dtype=torch.long)
    t2 = torch.zeros(1, TEAM_SIZE, dtype=torch.long)
    for j in ghost_slots:
        t1[0, j] = GHOST_TYPE_IDX
    out = cell(rows, alpha, nums, t1, t2, torch.tensor([[hazards]]))
    return out, rows


def test_switch_cell_passes_each_mons_OWN_row_through_and_appends_spin_denied():
    out, rows = _switch_case()
    assert out.shape == (1, TEAM_SIZE, PAIR_OUTCOME_SWITCH_DIM)
    assert torch.allclose(out[..., :_PAIR_OUTCOME_RAW], rows), (
        "mon j's switch cell must carry mon j's OWN reduced row — this is per-DEFENDER content, "
        "not the broadcast context Phase A delivers")
    got = out[0, :, PAIR_OUTCOME_SWITCH_IDX["spin_denied"]]
    # EXACT arithmetic: is_ghost_j (0/1) × α on the Rapid Spin seat (0.5) × their-side hazards (2/3)
    want = torch.tensor([0.0, 0.0, 0.5 * 2.0 / 3.0, 0.0, 0.0, 0.0])
    assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_spin_denied_is_zero_when_any_one_of_its_three_factors_is():
    """A conjunction, and the STAKE is what turns a fact into a value: with no Spikes on their side
    a Ghost switch-in denies nothing, and the coordinate must say 0 rather than "Gengar is good"."""
    i = PAIR_OUTCOME_SWITCH_IDX["spin_denied"]
    assert float(_switch_case(hazards=0.0)[0][0, :, i].abs().max()) == 0.0
    assert float(_switch_case(ghost_slots=())[0][0, :, i].abs().max()) == 0.0
    # ...and no believed Rapid Spin at all (the seat is some other move)
    cell = _cell_readable(PairOutcomeSwitchCell(PAIR_OUTCOME_SWITCH_DIM))
    out = cell(torch.zeros(1, TEAM_SIZE, _PAIR_OUTCOME_RAW), torch.tensor([[0.4, 0.4, 0.2]]),
               torch.tensor([[11, 22, 33]]),
               torch.full((1, TEAM_SIZE), GHOST_TYPE_IDX, dtype=torch.long),
               torch.zeros(1, TEAM_SIZE, dtype=torch.long), torch.ones(1, 1))
    assert float(out[0, :, i].abs().max()) == 0.0


def test_spin_denied_reads_the_SECOND_type_slot_too():
    """Gengar is Ghost/Poison and Sableye is Dark/Ghost — a predicate that only looked at type1
    would price exactly half the gen-3 spinblockers at zero."""
    i = PAIR_OUTCOME_SWITCH_IDX["spin_denied"]
    cell = _cell_readable(PairOutcomeSwitchCell(PAIR_OUTCOME_SWITCH_DIM))
    t1 = torch.zeros(1, TEAM_SIZE, dtype=torch.long)
    t2 = torch.zeros(1, TEAM_SIZE, dtype=torch.long)
    t2[0, 4] = GHOST_TYPE_IDX
    out = cell(torch.zeros(1, TEAM_SIZE, _PAIR_OUTCOME_RAW), torch.tensor([[0.0, 0.5, 0.0]]),
               torch.tensor([[11, _SPIN, 33]]), t1, t2, torch.ones(1, 1))
    assert float(out[0, 4, i]) == pytest.approx(0.5, abs=1e-6)


def test_switch_cell_is_zero_init():
    cell = PairOutcomeSwitchCell(PAIR_OUTCOME_SWITCH_DIM)
    assert float(cell.proj.weight.abs().max()) == 0.0
    assert float(cell.proj.bias.abs().max()) == 0.0
    out = cell(torch.rand(2, TEAM_SIZE, _PAIR_OUTCOME_RAW), torch.rand(2, 3),
               torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, TEAM_SIZE, dtype=torch.long),
               torch.zeros(2, TEAM_SIZE, dtype=torch.long), torch.rand(2, 1))
    assert out.shape == (2, TEAM_SIZE, PAIR_OUTCOME_SWITCH_DIM)
    assert float(out.abs().max()) == 0.0


def test_switch_cell_refuses_a_drifted_row_width_and_a_drifted_seat_axis():
    cell = PairOutcomeSwitchCell(PAIR_OUTCOME_SWITCH_DIM)
    with pytest.raises(ValueError, match="drifted"):
        cell(torch.rand(1, TEAM_SIZE, _PAIR_OUTCOME_RAW + 1), torch.rand(1, 3),
             torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, TEAM_SIZE, dtype=torch.long),
             torch.zeros(1, TEAM_SIZE, dtype=torch.long), torch.rand(1, 1))
    with pytest.raises(ValueError, match="SAME axis"):
        cell(torch.rand(1, TEAM_SIZE, _PAIR_OUTCOME_RAW), torch.rand(1, 3),
             torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, TEAM_SIZE, dtype=torch.long),
             torch.zeros(1, TEAM_SIZE, dtype=torch.long), torch.rand(1, 1))


# ---------------------------------------------- §9a through the REAL op: the §2.1 canonical case


def _real_op(K=6):
    from agents.model import damage_op_test as DT
    op, layout = DT._op_and_layout_topk(K)
    op.stash_pair_cells = True
    op.stash_pair_outcome = True
    return op, layout, DT


def test_g0_the_switch_cell_separates_two_mons_a_damage_read_calls_identical():
    """§2.1's canonical case, on the axis Phase B owns: TWO OF OUR MONS, one switch decision.

    Both face a believed Toxic. Both take EXACTLY zero damage from it and show the SAME `p_tox` —
    so every one of the ten damage numbers, and even the status probability, ties them. One
    carries Refresh; the other does not. `tempo_cost` in each mon's OWN reduced row is the only
    thing that separates them, and before this phase that row never reached a switch logit at all.

    (The Phase-B-era limitation note that stood here — "the tempo source is the mon's own cure
    MOVESET, NOT the Natural Cure ABILITY" — is CLOSED by v95 `gen3_status_economy_v1`. The ability
    and the bench cleric are now undo paths; the case below still uses Refresh vs nothing, so it
    reads the same numbers it always did, and the ability's own case is the next test.)
    """
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    op, layout, DT = _real_op()
    tox = DT._move_num("toxic")
    # both mons Normal-typed (no Poison/Steel immunity), identical stats — only the moveset differs
    ctx = DT._topk_ctx(op, defenders=[(242, T["NORMAL"], T["NORMAL"])] * 2 + [(0, 0, 0)] * 4)
    ctx.all_move_ids[0, 0, 0] = DT._move_num("refresh")     # our mon 0 can undo it
    ctx.all_move_ids[0, 1, 0] = DT._move_num("surf")        # our mon 1 cannot
    op(ctx, DT._logits_moves(layout["max_moves"], [tox]), None, DT._synth_latent(layout))

    # α one-hot on the Toxic seat: the claim under test is "GIVEN they click Toxic, the two mons'
    # switch rows differ", so the belief is pinned rather than sampled. (The α LADDER — publication
    # vs the R1 rung — is gated separately; mixing the two here would let a 4.5e-5 leak from the
    # other top-K seats masquerade as damage and defeat the exact-zero assertion.)
    k_tox = op.last_topk_idx[0].tolist().index(tox)
    alpha = torch.zeros_like(op.last_topk_w)
    alpha[:, k_tox] = 1.0
    rows = reduce_pair_in_all(alpha, op.last_pair_in, op.last_pair_gate)[0]      # [6, F]
    cure, nocure = rows[0], rows[1]
    for name in ("low", "high", "crit", "ko_ramp"):
        i = PAIR_OUTCOME_IDX[name]
        assert float(cure[i]) == 0.0 and float(nocure[i]) == 0.0, (
            f"{name}: Toxic deals no damage — this is the premise of the whole argument")
    i_tox = PAIR_OUTCOME_IDX["p_tox"]
    assert float(cure[i_tox]) > 0.0
    assert float(cure[i_tox]) == pytest.approx(float(nocure[i_tox]), abs=1e-6), (
        "the two mons are identical to the status PROBABILITY too — only the undo cost differs")
    i_tempo = PAIR_OUTCOME_IDX["tempo_cost"]
    assert float(cure[i_tempo]) > 0.0
    assert float(nocure[i_tempo]) == 0.0


def test_g0_bring_the_natural_cure_mon_is_finally_a_switch_cell_read():
    """§2.1's literal sentence — *"they will click Will-O-Wisp, so bring the Natural Cure mon"* —
    end to end, and the ONE case that needed both phases plus v95.

    Two of our mons, identical species, identical typing, identical HP, identical (empty) movesets.
    One has Natural Cure. Under a believed Will-O-Wisp every damage coordinate is 0.0 for both
    (a burn deals none), `p_brn` is identical for both, and `neutralization` is identical for both
    (the ability changes DURATION, not the per-turn rate — the deliberate limit). `tempo_cost` in
    each mon's OWN switch row is the only separator, and it exists at all only because v95 taught
    the coordinate to read the ability: before it, both read 0.0."""
    from agents import gen3_data
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    op, layout, DT = _real_op()
    wow = DT._move_num("willowisp")
    ctx = DT._topk_ctx(op, defenders=[(242, T["NORMAL"], T["NORMAL"])] * 2 + [(0, 0, 0)] * 4)
    ctx.ability1_ids[0, 0] = int(gen3_data.abilities.get("naturalcure").num)
    op(ctx, DT._logits_moves(layout["max_moves"], [wow]), None, DT._synth_latent(layout))
    k = op.last_topk_idx[0].tolist().index(wow)
    alpha = torch.zeros_like(op.last_topk_w)
    alpha[:, k] = 1.0
    rows = reduce_pair_in_all(alpha, op.last_pair_in, op.last_pair_gate)[0]
    nc, plain = rows[0], rows[1]
    for name in ("low", "high", "crit", "ko_ramp", "p_brn", "neutralization"):
        i = PAIR_OUTCOME_IDX[name]
        assert float(nc[i]) == pytest.approx(float(plain[i]), abs=1e-6), (
            f"{name} must NOT move — the ability is an undo PATH, not a landing or severity change")
    assert float(nc[PAIR_OUTCOME_IDX["p_brn"]]) > 0.0, "the burn must actually land"
    i_tempo = PAIR_OUTCOME_IDX["tempo_cost"]
    assert float(plain[i_tempo]) == 0.0
    assert float(nc[i_tempo]) == pytest.approx(float(nc[PAIR_OUTCOME_IDX["p_brn"]]), rel=1e-6), (
        "one switch = one of our turns, so the whole coordinate is P(burn) x 1.0")


def test_g0_each_defenders_row_is_its_own_and_the_reduction_is_still_one_alpha():
    """The per-defender delivery must not become a per-defender BELIEF. Two of our mons with
    different type immunities: their rows differ (that is the point), but each is the SAME α
    contracted against that mon's own column."""
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    op, layout, DT = _real_op()
    tw = DT._move_num("thunderwave")
    DT._topk_ctx  # (documented helper)
    ctx = DT._topk_ctx(op, defenders=[(260, T["WATER"], T["GROUND"]),          # Ground: immune
                                      (242, T["NORMAL"], T["NORMAL"])] + [(0, 0, 0)] * 4)
    op(ctx, DT._logits_moves(layout["max_moves"], [tw]), None, DT._synth_latent(layout))
    alpha = pair_alpha(None, op.last_topk_w, op.last_pair_seat_live)  # the shipped R1 rung
    rows = reduce_pair_in_all(alpha, op.last_pair_in, op.last_pair_gate)[0]
    i = PAIR_OUTCOME_IDX["p_par"]
    assert float(rows[0, i]) == 0.0, "Ground is immune to Thunder Wave"
    assert float(rows[1, i]) > 0.0
    # and each row IS Σ_k α_k · pair_in[j,k,:] — one distribution, six contractions
    manual = torch.einsum("k,jkf->jf", alpha[0], op.last_pair_in[0]) * op.last_pair_gate[0]
    assert torch.allclose(rows, manual, atol=1e-6)


# ------------------------------------------------------------------------------ extractor wiring


def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**_BASE_KWARGS)
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.pair_outcome_switch is None and fe_on.pair_outcome_switch is not None
    assert not any("pair_outcome_switch" in k for k in fe_off.state_dict())
    assert (fe_on.pointer_switch_cell_dim
            == fe_off.pointer_switch_cell_dim + PAIR_OUTCOME_SWITCH_DIM)
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim, (
        "the switch half must not touch the MOVE cell — that is Phase A's sink")
    assert fe_on.projection.in_features == fe_off.projection.in_features
    assert fe_on.value_projection.in_features == fe_off.value_projection.in_features


def test_off_is_byte_identical():
    fe_off, layout = _build(**_BASE_KWARGS)
    obs = _obs(layout)
    pi, vf = fe_off(obs)
    pi2, vf2 = fe_off(obs)
    assert torch.equal(pi, pi2) and torch.equal(vf, vf2)
    assert not any("pair_outcome_switch" in k for k in fe_off.state_dict())
    assert fe_off.damage_op.stash_pair_outcome is False
    assert fe_off.damage_op.last_pair_in is None


def test_on_forward_runs_and_contributes_exactly_zero_at_init():
    fe, layout = _build(**_ON_KWARGS)
    fe(_obs(layout))
    cells = fe.last_pointer_inputs.switch_cells
    assert cells.shape[2] == fe.pointer_switch_cell_dim
    assert float(cells[..., -PAIR_OUTCOME_SWITCH_DIM:].abs().max()) == 0.0
    assert "pair_outcome_switch.proj" in fe._identity_init_zeroed


def test_on_is_independently_enableable_of_the_MOVE_half_and_of_the_intent_head():
    """The two pair-outcome flags deliver ONE tensor to TWO sinks. Coupling them would make a
    measured result unattributable to a sink, so each must build and run alone — and each must
    reach α through the same ladder (publication, else the R1 belief_mean rung)."""
    fe, layout = _build(**_ON_KWARGS)
    assert fe.pair_outcome_move is None and fe.alpha_head is None
    fe(_obs(layout))
    assert fe.damage_op.last_pair_in is not None
    both, layout2 = _build(**_BASE_KWARGS, pair_outcome_cell=True, pair_outcome_switch=True,
                           opp_intent=True)
    pi, vf = both(_obs(layout2))
    assert both.last_pointer_inputs.move_cells.shape[2] == both.pointer_move_cell_dim
    assert both.last_pointer_inputs.switch_cells.shape[2] == both.pointer_switch_cell_dim


def test_requires_damage_op():
    with pytest.raises(ValueError, match="damage_op"):
        _build(**{**_ON_KWARGS, "damage_op": False, "damage_outgoing": False,
                  "damage_matrices_incoming": False, "damage_topk_k": 0})


def test_missing_topk_stash_fails_loud_rather_than_contributing_zeros():
    with pytest.raises((RuntimeError, ValueError)):
        fe, layout = _build(**{**_ON_KWARGS, "damage_matrices_incoming": False,
                               "damage_topk_k": 0})
        fe(_obs(layout))


# ------------------------------------------------------- identity-at-init on a REAL policy (M1)


def test_zero_init_survives_a_real_MaskablePPO_build():
    """Ledger M1: SB3's `_build()` orthogonally re-initialises every extractor Linear, so a
    zero-init asserted on a directly-built module is not an invariant."""
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(damage_matrices_incoming=True, damage_topk_k=6,
                                  entity_topk_seats=6, move_latent=True,
                                  pair_outcome_switch=True)
    fe = model.policy.features_extractor
    assert fe.pair_outcome_switch is not None
    assert float(fe.pair_outcome_switch.proj.weight.abs().max()) == 0.0
    assert float(fe.pair_outcome_switch.proj.bias.abs().max()) == 0.0


# ------------------------------------------------------------------------------ version machinery


def test_migration_defaults_the_flag_off():
    out = _migrate_config({"config_version": 93})
    assert out["pair_outcome_switch"] is False
    assert out["config_version"] == MODEL_CONFIG_VERSION >= 94


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, pair_outcome_switch=True)
    with pytest.raises(ModelVersionError, match="pair_outcome_switch"):
        a.check_compatible(b)


def test_the_flag_round_trips_through_the_snapshot_kwargs():
    from agents.model.snapshot import current_model_version
    v = current_model_version(load_mappings(), pair_outcome_switch=True)
    assert v.pair_outcome_switch is True
    assert current_model_version(load_mappings()).pair_outcome_switch is False


# -------------------------------------------------------------------------------- delivery graph


def test_the_module_is_drawn_with_edges_when_it_is_ON():
    import json
    import os
    import tempfile

    from agents.model.delivery_graph import (
        MODULE_GRAPH_TOKENS, _DEFAULT_CONFIG, build_extractor, build_graph, module_coverage)
    assert MODULE_GRAPH_TOKENS["pair_outcome_switch"] == ("PairOutcomeSwitchCell",)
    cfg = json.load(open(_DEFAULT_CONFIG))
    cfg["pair_outcome_switch"] = True
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        graph = build_graph(path)
        fe = build_extractor(path)[0]
    vias = " ".join(str(e.get("via", "")) for e in graph["edges"])
    assert "PairOutcomeSwitchCell" in vias, "the module is ON and draws no edge"
    # the FIRST alpha route to a switch logit — worth asserting explicitly, since every other
    # alpha consumer lands on the move cells or the value tail
    assert any(str(e["dst"]).startswith("pointer.switch_logit")
               and "alpha" in str(e.get("via", "")) for e in graph["edges"])
    assert not module_coverage(fe, graph)
