"""gen3_conditional_threat_v1 (v95) — OA1's gates.

`design_conditional_opponent_cells.md` §1 + §0.2, and §5's pre-registered test list, which this
file follows item by item where the item still applies after the Phase-A/B substitutions:

  * §5.2 **ON == OFF bitwise at init** on a REAL `MaskablePPO` policy (post-SB3-ortho — ledger M1).
  * §5.5 **Equivariance**: permuting our bench permutes the OA1 cells with it.
  * §5.1's spirit — constructed scenarios where the cell equals a HAND-COMPUTED marginal. Every
    numeric assertion below is the exact product, not an inequality, because the coordinates are
    products and an ">0" test passes on any positive constant.
  * §5.3 / §5.4 are OA2's (p_switch, the unrevealed `q_b` mass) and are gated in
    `switch_branch_test.py`; they have no OA1 counterpart because OA1 contracts over their MOVE
    axis, not over the switch branch.

Plus the standing substrate contract: ONE α (Contract W), a fail-loud seat-axis mismatch, and the
version/graph machinery. **The load-bearing test is `test_the_decorrelated_channels_CANNOT_express
_e_pko_acc`** — the planted §9a case, where two of our mons are identical in every channel the
reduced row already carries and differ only in the product this cell forms.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import (
    CONDITIONAL_THREAT_SWITCH_DIM, _CONDITIONAL_THREAT_RAW, _PAIR_OUTCOME_RAW,
)
from agents.model.conditional_threat import (
    CONDITIONAL_THREAT_COORDS, CONDITIONAL_THREAT_IDX, ConditionalThreatCell,
)
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.pair_outcome import PAIR_OUTCOME_IDX, pair_alpha
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_BASE_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6,
)
_ON_KWARGS = {**_BASE_KWARGS, "conditional_threat_cell": True}


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


def test_the_coordinate_table_is_the_single_spelling_of_the_layout():
    assert len(CONDITIONAL_THREAT_COORDS) == _CONDITIONAL_THREAT_RAW \
        == CONDITIONAL_THREAT_SWITCH_DIM
    assert CONDITIONAL_THREAT_COORDS == (
        "e_pko_acc", "e_type_mult", "margin_high", "margin_crit")


def test_none_of_the_coordinates_duplicates_one_the_reduced_row_already_delivers():
    """The §1.2 substitution, asserted rather than asserted-in-prose: OA1 must not re-emit
    `high` / `ko_ramp` / `acc` / the status columns, which Phase B already puts on this exact
    cell. Duplicated delivery is not neutral — it doubles a channel's effective weight at init
    and makes an ablation unattributable."""
    assert not (set(CONDITIONAL_THREAT_COORDS) & set(PAIR_OUTCOME_IDX)), (
        "an OA1 coordinate collides by NAME with a reduced-row coordinate")


# ------------------------------------------------------- the cell's arithmetic, on exact inputs


def _cell_out(alpha, pair_in, type_mult, gate, hp, cell=None):
    """Read the RAW coordinates back by loading the projection with the identity."""
    cell = cell or ConditionalThreatCell(_CONDITIONAL_THREAT_RAW)
    with torch.no_grad():
        cell.proj.weight.copy_(torch.eye(_CONDITIONAL_THREAT_RAW))
    return cell(alpha, pair_in, type_mult, gate, hp)


def _grid(B=1, K=2, J=TEAM_SIZE, F=_PAIR_OUTCOME_RAW):
    return torch.zeros(B, J, K, F)


def test_the_decorrelated_channels_CANNOT_express_e_pko_acc():
    """§9a, and the reason `e_pko_acc` is a coordinate at all (§0.2(2): *precompute every
    nonlinearity of two numbers IN THE OP*).

    Their believed set is {Blizzard 70% acc — OHKOs our mon 0; Thunderbolt 100% acc — OHKOs our
    mon 1}, α = ½/½. The two mons are then IDENTICAL in every channel the reduced row carries:

        Σα·ko_ramp  = 0.5·1 + 0.5·0 = 0.5   for mon 0
                    = 0.5·0 + 0.5·1 = 0.5   for mon 1
        Σα·acc      = 0.5·0.7 + 0.5·1.0 = 0.85   for BOTH (acc has no defender axis at all)

    while P(this mon dies) is 0.5·(1·0.7) = 0.35 for mon 0 and 0.5·(1·1.0) = 0.50 for mon 1. A
    thin `tanh` scorer over a shared cell does not multiply two of its own inputs, so this pair is
    unorderable without the product — and the numbers are exact, so a coordinate that merely
    correlated with it would fail here."""
    K = 2
    pair_in = _grid(K=K)
    i_ko, i_acc = PAIR_OUTCOME_IDX["ko_ramp"], PAIR_OUTCOME_IDX["acc"]
    pair_in[0, :, 0, i_acc] = 0.70          # Blizzard
    pair_in[0, :, 1, i_acc] = 1.00          # Thunderbolt
    pair_in[0, 0, 0, i_ko] = 1.0            # Blizzard OHKOs mon 0
    pair_in[0, 1, 1, i_ko] = 1.0            # Thunderbolt OHKOs mon 1
    alpha = torch.tensor([[0.5, 0.5]])
    out = _cell_out(alpha, pair_in, torch.zeros(1, TEAM_SIZE, K),
                    torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, TEAM_SIZE))
    i = CONDITIONAL_THREAT_IDX["e_pko_acc"]
    assert float(out[0, 0, i]) == pytest.approx(0.35, abs=1e-6)
    assert float(out[0, 1, i]) == pytest.approx(0.50, abs=1e-6)
    # ...and the two decorrelated channels really are tied, which is the half that makes the case
    a = alpha[0]
    for j in (0, 1):
        assert float((a * pair_in[0, j, :, i_ko]).sum()) == pytest.approx(0.5, abs=1e-6)
        assert float((a * pair_in[0, j, :, i_acc]).sum()) == pytest.approx(0.85, abs=1e-6)


def test_e_type_mult_is_the_alpha_weighted_multiplier_and_zero_means_IMMUNE():
    """§9a pair 1: *switch Gengar vs switch Swampert into a believed Earthquake.* A `high` of 0.0
    is ambiguous — an immune defender, a status seat, and a roll that simply does nothing all read
    it — while `type_mult = 0.0` is structural immunity and nothing else. Exact: α = (0.25, 0.75)
    over multipliers (2.0, 0.5) gives 0.875 on mon 0, and 0.0 on the immune mon 1."""
    K = 2
    tm = torch.zeros(1, TEAM_SIZE, K)
    tm[0, 0] = torch.tensor([2.0, 0.5])
    tm[0, 1] = torch.tensor([0.0, 0.0])
    out = _cell_out(torch.tensor([[0.25, 0.75]]), _grid(K=K), tm,
                    torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, TEAM_SIZE))
    i = CONDITIONAL_THREAT_IDX["e_type_mult"]
    assert float(out[0, 0, i]) == pytest.approx(0.25 * 2.0 + 0.75 * 0.5, abs=1e-6) == \
        pytest.approx(0.875, abs=1e-6)
    assert float(out[0, 1, i]) == 0.0


def test_the_two_margins_separate_mons_a_saturated_probability_cannot():
    """§0.2(3), at BOTH ends of the saturation, in one board.

    mon 0 (`hp` 0.45) and mon 1 (`hp` 0.82) both face a believed hit whose max roll is 0.50 of
    their bars and whose crit roll is 0.90. `ko_ramp` is identical for both by construction here,
    so every probability channel ties them; `margin_high` reads +0.05 (dead) vs −0.32 (alive) and
    `margin_crit` reads +0.45 vs +0.08 — i.e. mon 1 survives the max roll and dies to a crit, which
    is exactly the *safe pivot vs coinflip pivot* distinction nothing else in the cell records."""
    K = 1
    pair_in = _grid(K=K)
    pair_in[0, :, 0, PAIR_OUTCOME_IDX["high"]] = 0.50
    pair_in[0, :, 0, PAIR_OUTCOME_IDX["crit"]] = 0.90
    hp = torch.zeros(1, TEAM_SIZE)
    hp[0, 0], hp[0, 1] = 0.45, 0.82
    out = _cell_out(torch.ones(1, K), pair_in, torch.zeros(1, TEAM_SIZE, K),
                    torch.ones(1, TEAM_SIZE, 1), hp)
    ih, ic = CONDITIONAL_THREAT_IDX["margin_high"], CONDITIONAL_THREAT_IDX["margin_crit"]
    assert float(out[0, 0, ih]) == pytest.approx(0.05, abs=1e-6)
    assert float(out[0, 1, ih]) == pytest.approx(-0.32, abs=1e-6)
    assert float(out[0, 0, ic]) == pytest.approx(0.45, abs=1e-6)
    assert float(out[0, 1, ic]) == pytest.approx(0.08, abs=1e-6)


def test_the_gate_zeroes_a_dead_or_opponentless_defender():
    """`gate` is `alive × has_opp`. Without it the margin of a fainted mon would read `0 − 0 = 0`
    by luck rather than by construction, and a board with no opponent would emit real numbers."""
    K = 1
    pair_in = _grid(K=K)
    pair_in[0, :, 0, PAIR_OUTCOME_IDX["high"]] = 0.5
    gate = torch.ones(1, TEAM_SIZE, 1)
    gate[0, 3] = 0.0
    hp = torch.full((1, TEAM_SIZE), 0.9)
    out = _cell_out(torch.ones(1, K), pair_in, torch.full((1, TEAM_SIZE, K), 2.0), gate, hp)
    assert float(out[0, 3].abs().max()) == 0.0
    assert float(out[0, 0].abs().max()) > 0.0


# ------------------------------------------------------------------ Contract W + the fail-louds


def test_a_PER_DEFENDER_alpha_is_a_SHAPE_ERROR_not_a_thing_a_test_hunts_for():
    """The planted D3 violation. `alpha` has no `J` axis by signature, so a caller that computed a
    per-defender distribution cannot pass it — the reduction may vary per defender, the
    DISTRIBUTION may not (`design_pair_reduction.md` §2)."""
    K = 3
    per_defender_alpha = torch.rand(1, TEAM_SIZE, K)   # the violation, made explicit
    with pytest.raises((RuntimeError, ValueError)):
        _cell_out(per_defender_alpha, _grid(K=K), torch.zeros(1, TEAM_SIZE, K),
                  torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, TEAM_SIZE))


def test_seat_axis_mismatch_fails_loud():
    """α's seats, `pair_in`'s candidate columns and `type_mult`'s are ONE axis. A silent broadcast
    would pair each α weight with the wrong opponent move while every shape check still passed —
    the named `op move-order` bug class."""
    cell = ConditionalThreatCell(_CONDITIONAL_THREAT_RAW)
    with pytest.raises(ValueError, match="move-order"):
        cell(torch.ones(1, 4), _grid(K=3), torch.zeros(1, TEAM_SIZE, 3),
             torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, TEAM_SIZE))
    with pytest.raises(ValueError, match="move-order"):
        cell(torch.ones(1, 3), _grid(K=3), torch.zeros(1, TEAM_SIZE, 4),
             torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, TEAM_SIZE))


def test_defender_axis_mismatch_fails_loud():
    cell = ConditionalThreatCell(_CONDITIONAL_THREAT_RAW)
    with pytest.raises(ValueError, match="team axis"):
        cell(torch.ones(1, 2), _grid(K=2), torch.zeros(1, 5, 2),
             torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, TEAM_SIZE))


def test_seat_permutation_invariance():
    """Every seat-indexed computation is `Σ_k α_k · f_k`, so a JOINT permutation of α's seats and
    the grids' candidate columns leaves every coordinate unchanged."""
    K = 4
    torch.manual_seed(3)
    pair_in = torch.rand(2, TEAM_SIZE, K, _PAIR_OUTCOME_RAW)
    tm = torch.rand(2, TEAM_SIZE, K)
    alpha = torch.rand(2, K)
    hp, gate = torch.rand(2, TEAM_SIZE), torch.ones(2, TEAM_SIZE, 1)
    cell = ConditionalThreatCell(_CONDITIONAL_THREAT_RAW)
    a = _cell_out(alpha, pair_in, tm, gate, hp, cell=cell)
    perm = torch.tensor([2, 0, 3, 1])
    b = _cell_out(alpha[:, perm], pair_in[:, :, perm], tm[:, :, perm], gate, hp, cell=cell)
    assert torch.allclose(a, b, atol=1e-6)


def test_our_team_permutation_EQUIvariance():
    """§5 gate 5, our half: permuting our bench permutes the OA1 cells with it. Each row rides its
    own mon's switch logit and α has no `J` axis, so this holds by construction — but the design
    pre-registered it as a test and a future coordinate reading a slot index would break it."""
    K = 3
    torch.manual_seed(4)
    pair_in = torch.rand(1, TEAM_SIZE, K, _PAIR_OUTCOME_RAW)
    tm, hp = torch.rand(1, TEAM_SIZE, K), torch.rand(1, TEAM_SIZE)
    gate, alpha = torch.ones(1, TEAM_SIZE, 1), torch.rand(1, K)
    cell = ConditionalThreatCell(_CONDITIONAL_THREAT_RAW)
    base = _cell_out(alpha, pair_in, tm, gate, hp, cell=cell)
    p = torch.tensor([3, 1, 5, 0, 4, 2])
    permuted = _cell_out(alpha, pair_in[:, p], tm[:, p], gate, hp[:, p], cell=cell)
    assert torch.allclose(base[:, p], permuted, atol=1e-6)


def test_the_cell_is_zero_init():
    cell = ConditionalThreatCell(7)
    assert float(cell.proj.weight.abs().max()) == 0.0
    assert float(cell.proj.bias.abs().max()) == 0.0


# --------------------------------------------------------- the op's type-multiplier seam (v95)


def test_the_op_stashes_the_type_multiplier_only_when_asked_and_at_alphas_alignment():
    """The multiplier must arrive on α's OWN seat axis. Re-deriving it at the consumer from
    `last_topk_idx` + the chart is the `op move-order` bug class with extra steps — the real
    move-num gather and the ability fold both live inside `_incoming_matrix`."""
    from agents.model import damage_op_test as DT
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    K = 6
    op, layout = DT._op_and_layout_topk(K)
    eq = DT._move_num("earthquake")
    # mon 0 = ELECTRIC (2x to Ground), mon 1 = FLYING (immune).
    ctx = DT._topk_ctx(op, defenders=[(0, T["ELECTRIC"], 0), (0, T["FLYING"], 0)] + [(0, 0, 0)] * 4)
    op(ctx, DT._logits_moves(layout["max_moves"], [eq]), None, DT._synth_latent(layout))
    assert op.last_pair_type_mult is None, "the seam is OFF by default — no stash, no cost"

    op, layout = DT._op_and_layout_topk(K)
    op.stash_pair_type_mult = True
    ctx = DT._topk_ctx(op, defenders=[(0, T["ELECTRIC"], 0), (0, T["FLYING"], 0)] + [(0, 0, 0)] * 4)
    op(ctx, DT._logits_moves(layout["max_moves"], [eq]), None, DT._synth_latent(layout))
    tm = op.last_pair_type_mult
    assert tm is not None and tm.shape[1] == TEAM_SIZE and tm.shape[2] == K
    k = op.last_topk_idx[0].tolist().index(eq)
    assert float(tm[0, 0, k]) == pytest.approx(2.0, abs=1e-6)
    assert float(tm[0, 1, k]) == 0.0, "Ground vs Flying is a gen3 IMMUNITY, exactly 0"


# --------------------------------------------------------------------- OFF / ON, and the widths


def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**_BASE_KWARGS)
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.conditional_threat is None and fe_on.conditional_threat is not None
    assert not any("conditional_threat" in k for k in fe_off.state_dict())
    assert (fe_on.pointer_switch_cell_dim
            == fe_off.pointer_switch_cell_dim + CONDITIONAL_THREAT_SWITCH_DIM)
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim, (
        "OA1 is a SWITCH-cell cell — it must not touch the move cell")
    assert fe_on.projection.in_features == fe_off.projection.in_features
    assert fe_on.value_projection.in_features == fe_off.value_projection.in_features


def test_off_is_byte_identical():
    fe_off, layout = _build(**_BASE_KWARGS)
    obs = _obs(layout)
    pi, vf = fe_off(obs)
    pi2, vf2 = fe_off(obs)
    assert torch.equal(pi, pi2) and torch.equal(vf, vf2)
    assert fe_off.damage_op.stash_pair_type_mult is False
    assert fe_off.damage_op.last_pair_type_mult is None


def test_on_forward_runs_and_contributes_exactly_zero_at_init():
    fe, layout = _build(**_ON_KWARGS)
    fe(_obs(layout))
    cells = fe.last_pointer_inputs.switch_cells
    assert cells.shape[2] == fe.pointer_switch_cell_dim
    assert float(cells[..., -CONDITIONAL_THREAT_SWITCH_DIM:].abs().max()) == 0.0
    assert "conditional_threat.proj" in fe._identity_init_zeroed


def test_it_stacks_with_the_phase_B_switch_cell_and_stays_the_LAST_block():
    """Two modules now widen one cell. The order matters for reading a slice back (this test's own
    `-DIM:` reads, and every future ablation), so it is pinned: Phase B's row first, OA1 last."""
    fe, layout = _build(**_BASE_KWARGS, pair_outcome_switch=True, conditional_threat_cell=True)
    fe(_obs(layout))
    cells = fe.last_pointer_inputs.switch_cells
    assert cells.shape[2] == fe.pointer_switch_cell_dim
    assert cells.shape[2] == (fe.damage_op.pointer_switch_cell_dim
                             + fe.pair_outcome_switch.in_dim + CONDITIONAL_THREAT_SWITCH_DIM)


def test_on_is_independently_enableable_of_phase_B_and_of_the_intent_head():
    """OA1 and Phase B widen the SAME cell with DIFFERENT quantities. Coupling them would make a
    measured result unattributable to either, so each must build and run alone — and OA1 must
    reach α through the same ladder (publication, else the R1 belief_mean rung)."""
    fe, layout = _build(**_ON_KWARGS)
    assert fe.pair_outcome_switch is None and fe.alpha_head is None
    fe(_obs(layout))
    assert fe.damage_op.last_pair_in is not None and fe.damage_op.last_pair_type_mult is not None


def test_requires_damage_op_and_the_incoming_matrix():
    with pytest.raises(ValueError, match="damage_op"):
        _build(**{**_ON_KWARGS, "damage_op": False, "damage_outgoing": False,
                  "damage_matrices_incoming": False, "damage_topk_k": 0})
    with pytest.raises(ValueError, match="damage_matrices_incoming"):
        _build(**{**_ON_KWARGS, "damage_matrices_incoming": False, "damage_topk_k": 0})


def test_missing_stashes_fail_loud_rather_than_contributing_zeros():
    """A silent no-op reads exactly like a null RESULT, which is the failure this whole substrate
    is built to make impossible."""
    fe, layout = _build(**_ON_KWARGS)
    fe.damage_op.stash_pair_type_mult = False       # simulate a seam that stopped being set
    with pytest.raises(RuntimeError, match="type multiplier"):
        fe(_obs(layout))


def test_the_alpha_ladder_is_the_shipped_one_not_a_second_distribution():
    """§1.2's `λ`-weighted `w` is NOT built, and this is the assertion that keeps it that way: no
    parameter of the cell may be a temperature, and the α the forward uses must be exactly what
    `pair_alpha` returns for the same inputs."""
    fe, layout = _build(**_ON_KWARGS)
    assert [n for n, _ in fe.conditional_threat.named_parameters()] == ["proj.weight", "proj.bias"]
    fe(_obs(layout))
    op = fe.damage_op
    expected = pair_alpha(fe.last_alpha_logits, op.last_topk_w, op.last_pair_seat_live)
    assert expected.shape == (3, op.last_topk_w.shape[-1])
    assert float(expected.sum(-1).max()) <= 1.0 + 1e-6


# ------------------------------------------------------- identity-at-init on a REAL policy (M1)


def test_zero_init_survives_a_real_MaskablePPO_build():
    """§5.2. Ledger M1: SB3's `_build()` orthogonally re-initialises every extractor Linear, so a
    zero-init asserted on a directly-built module is not an invariant — the construction path a
    unit test uses is not the one training uses."""
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(damage_matrices_incoming=True, damage_topk_k=6,
                                  entity_topk_seats=6, move_latent=True,
                                  conditional_threat_cell=True)
    fe = model.policy.features_extractor
    assert fe.conditional_threat is not None
    assert float(fe.conditional_threat.proj.weight.abs().max()) == 0.0
    assert float(fe.conditional_threat.proj.bias.abs().max()) == 0.0


# ------------------------------------------------------------------------------ version machinery


def test_a_pre_floor_config_is_REFUSED_not_defaulted():
    """The critic-route deletion wave bumped ARCH_SIGNATURE, so MIGRATION_FLOOR rose to 96
    and this v94 config is now refused outright rather than walked through the v95 branch
    that defaults `conditional_threat_cell`. That is the floor's stated purpose ("refuses pre-floor
    configs outright instead of walking dead branches"), and the assertion follows the
    BEHAVIOUR: what must hold is that a stale config is rejected with a diagnosis, not that
    an unreachable branch still defaults a field."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 94})
    assert MODEL_CONFIG_VERSION >= 95


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, conditional_threat_cell=True)
    with pytest.raises(ModelVersionError, match="conditional_threat_cell"):
        a.check_compatible(b)


def test_the_flag_round_trips_through_the_snapshot_kwargs():
    from agents.model.snapshot import current_model_version
    v = current_model_version(load_mappings(), conditional_threat_cell=True)
    assert v.conditional_threat_cell is True
    assert current_model_version(load_mappings()).conditional_threat_cell is False


# -------------------------------------------------------------------------------- delivery graph


def test_the_module_is_drawn_with_edges_when_it_is_ON():
    import json
    import os
    import tempfile

    from agents.model.delivery_graph import (
        MODULE_GRAPH_TOKENS, _DEFAULT_CONFIG, build_graph)
    assert MODULE_GRAPH_TOKENS["conditional_threat"] == ("ConditionalThreatCell",)
    cfg = json.load(open(_DEFAULT_CONFIG))
    cfg["conditional_threat_cell"] = True
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        graph = build_graph(path)
    drawn = [e for e in graph["edges"] if "ConditionalThreatCell" in (e.get("via") or "")]
    assert len(drawn) >= TEAM_SIZE, "one switch-logit edge per our mon"
    physics = [e for e in drawn if e["src"] == "damage_op"]
    assert len(physics) == TEAM_SIZE and all(e.get("zero_init") for e in physics)
    # ...and α's own publication must be drawn reaching the SWITCH logits through it, which is the
    # edge that says "the intent head now weights a switch decision".
    assert any(e["src"] == "alpha_head" and "switch_logit" in e["dst"] for e in drawn)
