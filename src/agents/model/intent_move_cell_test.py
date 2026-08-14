"""gen3_intent_move_cell_v1 (G3) — the POLICY-side alpha consumer's gates.

What must hold (design_conditional_execution.md §6 G3 + the house rules):
  * OFF is the default and builds NOTHING — no module, no extra dims, no state_dict keys.
  * ON contributes EXACTLY zero to every pointer logit at init, asserted on a REAL
    MaskablePPO-built policy (the M1/SB3-ortho-clobber lesson: a bare-extractor assertion is
    not an invariant), and the path is proven LIVE by perturbing the projection.
  * The alpha weighting is invariant under a joint permutation of (alpha's move seats, the
    per-candidate operand columns) — the axis is content-addressed, not positional.
  * An axis-width mismatch and a missing-alpha forward FAIL LOUD (the `op move-order` class).
  * Under belief_grad_mode="label_only" no pointer-logit gradient reaches alpha_head's
    parameters through this path (the v75 publish boundary).
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import INTENT_MOVE_CELL_DIM
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.intent_move_cell import IntentMoveCell
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_ON_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6, opp_intent=True, intent_move_cell=True,
)


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


_TRACES = ("/home/goodlad/dev/gen3ai/models/ai_v9_10_gen9_intent_distcritic_0813/"
           "eval_traces/**/*_states.npz")


def _real_obs(n=8):
    """Real eval-trace states — random [0,1) obs carry no valid belief seats, so alpha's mass
    collapses onto SWITCH and every alpha-carried gradient/operand is structurally zero. The
    gradient-flow tests need boards where alpha has something to point at."""
    from agents.model.audit_states import collect_states
    try:
        obs, _, _ = collect_states([_TRACES], n, seed=0)
    except FileNotFoundError:
        pytest.skip("no gen-9 eval traces on this machine (models/ lives in the main checkout)")
    return {"observation": torch.from_numpy(obs)}


# ------------------------------------------------------------------- OFF builds nothing
def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**{**_ON_KWARGS, "intent_move_cell": False})
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.intent_move_cell is None
    assert fe_on.intent_move_cell is not None
    assert (fe_on.pointer_move_cell_dim - fe_off.pointer_move_cell_dim
            == INTENT_MOVE_CELL_DIM)
    off_keys = {k for k in fe_off.state_dict() if "intent_move_cell" in k}
    on_keys = {k for k in fe_on.state_dict() if "intent_move_cell" in k}
    assert off_keys == set() and on_keys == {"intent_move_cell.proj.weight",
                                            "intent_move_cell.proj.bias"}


# ------------------------------------------------- ON is identity at init, on a REAL policy
def test_on_is_identity_at_init_on_a_real_maskableppo_policy():
    """The zero-init projection must SURVIVE SB3's ortho pass (restore_identity_init captures
    it by observation — ledger M1), asserted on a policy built through the SAME path training
    uses. A bare-extractor assertion is not an invariant."""
    pytest.importorskip("sb3_contrib")
    from agents.model.identity_init_test import _build_real_policy
    model, _enc = _build_real_policy(**_ON_KWARGS)
    fe = model.policy.features_extractor
    assert fe.intent_move_cell is not None
    assert torch.all(fe.intent_move_cell.proj.weight == 0), \
        "SB3 ortho-init clobbered the zero-init projection (M1 guard regression)"
    assert torch.all(fe.intent_move_cell.proj.bias == 0)


def test_on_at_init_logits_match_zeroed_cells_and_perturbed_projection_moves_them():
    fe, layout = _build(**_ON_KWARGS)
    obs = _obs(layout)
    with torch.no_grad():
        fe.forward(obs)
        tok_req, move_valid, team_out, mcells, scells = fe.last_pointer_inputs
        # at init the appended block is exactly zero
        assert torch.all(mcells[..., -INTENT_MOVE_CELL_DIM:] == 0)
        # perturb the projection BIAS -> the appended block must move even when the raw
        # operands are zero on this board (the concat wiring itself is what's under test)
        torch.nn.init.constant_(fe.intent_move_cell.proj.bias, 0.37)
        fe.forward(obs)
        mcells2 = fe.last_pointer_inputs[3]
        assert torch.all(mcells2[..., -INTENT_MOVE_CELL_DIM:] != 0), \
            "the appended cells never reached the pointer stash — the path is decorative"


# ------------------------------------------------------------- permutation invariance + fail-loud
def test_alpha_weighting_is_seat_permutation_invariant():
    torch.manual_seed(3)
    m = IntentMoveCell(INTENT_MOVE_CELL_DIM)
    torch.nn.init.normal_(m.proj.weight, std=0.5)
    B, K = 4, 6
    alpha_logits = torch.randn(B, K + 1)
    base = torch.rand(B, 4, 4)
    d_burn, d_slp = -torch.rand(B, K), -torch.rand(B, K)
    is_brn, is_slp = (torch.rand(B, 4) > 0.5).float(), (torch.rand(B, 4) > 0.5).float()
    out = m(alpha_logits, base, d_burn, d_slp, is_brn, is_slp)
    perm = torch.randperm(K)
    alpha_p = torch.cat([alpha_logits[:, perm], alpha_logits[:, -1:]], dim=1)
    out_p = m(alpha_p, base, d_burn[:, perm], d_slp[:, perm], is_brn, is_slp)
    assert torch.allclose(out, out_p, atol=1e-6), \
        "a joint seat permutation changed the alpha-expectation — the axis is positional"


def test_axis_width_mismatch_fails_loud():
    m = IntentMoveCell(INTENT_MOVE_CELL_DIM)
    with pytest.raises(ValueError, match="SAME axis"):
        m(torch.randn(2, 7), torch.rand(2, 4, 4), torch.rand(2, 5), torch.rand(2, 6),
          torch.ones(2, 4), torch.ones(2, 4))


def test_switch_mass_shrinks_the_conditional_terms_toward_zero():
    """The unrenormalized contract: as alpha_SWITCH -> 1, e_burn/e_slp/alpha_stay -> 0 —
    'they are leaving' must not be priced as 'they attack'."""
    m = IntentMoveCell(INTENT_MOVE_CELL_DIM)
    torch.nn.init.normal_(m.proj.weight, std=0.5)
    B, K = 2, 6
    base = torch.rand(B, 4, 4)
    d_burn, d_slp = -torch.ones(B, K), -torch.ones(B, K)
    ones = torch.ones(B, 4)
    stay = torch.zeros(B, K + 1); stay[:, 0] = 20.0          # all mass on a move seat
    leave = torch.zeros(B, K + 1); leave[:, -1] = 20.0       # all mass on SWITCH
    out_stay = m(stay, base, d_burn, d_slp, ones, ones)
    out_leave = m(leave, base, d_burn, d_slp, ones, ones)
    # under all-SWITCH mass the alpha-carried channels vanish; the raw base channels remain,
    # so compare against the same module with alpha-channels forced to zero
    zero = torch.zeros(B, K + 1); zero[:, -1] = 1e9
    assert torch.allclose(out_leave, m(zero, base, d_burn, d_slp, ones, ones), atol=1e-5)
    assert not torch.allclose(out_stay, out_leave, atol=1e-4)


# --------------------------------------------------------------- label_only publish boundary
def test_label_only_cuts_the_ppo_route_into_alpha_head():
    fe, layout = _build(**{**_ON_KWARGS, "belief_grad_mode": "label_only"})
    fe.train()
    torch.nn.init.normal_(fe.intent_move_cell.proj.weight, std=0.5)  # make the path carry signal
    obs = _real_obs()
    fe.forward(obs)
    _tok, _mv, _team, mcells, _sc = fe.last_pointer_inputs
    loss = mcells[..., -INTENT_MOVE_CELL_DIM:].sum()
    loss.backward()
    alpha_grads = [p.grad for p in fe.alpha_head.parameters() if p.grad is not None]
    assert all(g.abs().max() == 0 for g in alpha_grads if g is not None) or not alpha_grads, \
        "label_only must stop the pointer-path gradient at alpha's publication boundary"


def test_shaping_mode_lets_gradient_reach_alpha_head():
    # Real states: random obs have no valid belief seats, so alpha's mass sits on SWITCH and
    # the gradient through the alpha-expectation is structurally zero regardless of the mode.
    fe, layout = _build(**{**_ON_KWARGS, "belief_grad_mode": "shaping"})
    fe.train()
    torch.nn.init.normal_(fe.intent_move_cell.proj.weight, std=0.5)
    obs = _real_obs()
    fe.forward(obs)
    mcells = fe.last_pointer_inputs[3]
    mcells[..., -INTENT_MOVE_CELL_DIM:].sum().backward()
    got = any(p.grad is not None and p.grad.abs().max() > 0
              for p in fe.alpha_head.parameters())
    assert got, "under shaping the alpha head should be trainable through the live path"


# --------------------------------------------------------------------------- requirements
def test_requires_opp_intent_and_damage_op():
    with pytest.raises(ValueError, match="opp_intent"):
        _build(**{**_ON_KWARGS, "opp_intent": False})
    with pytest.raises(ValueError, match="damage_op"):
        _build(**{**_ON_KWARGS, "damage_op": False, "damage_outgoing": False,
                  "damage_matrices_incoming": False, "damage_topk_k": 0})
