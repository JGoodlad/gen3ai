"""Unit tests for gen3_belief_grad_mode_v1 — the `--belief-grad-mode {shaping, detached}` knob.

`detached` makes the STATE-prediction belief heads (move / spread / hp-type / the species-moves-latent
aux) READ a stop-grad trunk, so neither their supervised loss nor the op/policy gradient through them can
reshape the shared trunk — while the belief is still computed, reinjected into the forward, and consumed
by the op. The two load-bearing invariants:
  (1) detach() changes the BACKWARD graph only, so the forward VALUES are bit-identical to shaping;
  (2) a belief-derived loss reshapes the trunk under `shaping` but NOT under `detached`, while the belief
      HEAD itself still trains in both modes (it stays fully "in the system").
"""
import gymnasium as gym
import numpy as np
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mp = load_mappings()
_layout = Gen3ObservationEncoder(_mp).get_layout()
# A config with all four state-prediction belief heads live (move + spread + hp-type + the aux head).
_CFG = dict(attend_unrevealed_opponents=True, opp_belief_slots=True, move_belief_mode="both",
            spread_belief=True, damage_op=True, hp_type_belief_mode="learned")


def _model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mp, **kw)


def _obs(n):
    return {"observation": torch.rand(n, _layout["total_dim"])}


def _grad_mass(model, substr_any):
    return sum(p.grad.abs().sum().item()
               for n, p in model.named_parameters()
               if p.grad is not None and any(s in n for s in substr_any))


def test_construct_and_flag_set():
    sh = _model(**_CFG, belief_grad_mode="shaping")
    dt = _model(**_CFG, belief_grad_mode="detached")
    assert sh.belief_grad_mode == "shaping" and not sh._belief_detach
    assert dt.belief_grad_mode == "detached" and dt._belief_detach
    # the per-head flag is stamped on every constructed state-prediction head
    for h in (dt.move_belief, dt.spread_belief, dt.hp_type_belief_head, dt.belief_head):
        assert h is not None and h.detach_read is True
    for h in (sh.move_belief, sh.spread_belief, sh.hp_type_belief_head, sh.belief_head):
        assert h is not None and h.detach_read is False


def test_invalid_mode_raises():
    try:
        _model(**_CFG, belief_grad_mode="bogus")
    except ValueError:
        return
    assert False, "expected ValueError on an invalid belief_grad_mode"


def test_detached_forward_value_identical_to_shaping():
    """detach() touches the backward graph, not forward VALUES — so toggling the mode on the SAME model
    (same weights) leaves the forward output bit-identical. Proves detached is a pure gradient change."""
    m = _model(**_CFG)
    obs = _obs(3)
    m.eval()
    with torch.no_grad():
        pi0, vf0 = m.forward_internal(obs)
        for h in (m.move_belief, m.spread_belief, m.hp_type_belief_head, m.belief_head):
            h.detach_read = True
        pi1, vf1 = m.forward_internal(obs)
    assert torch.allclose(pi0, pi1, atol=1e-6) and torch.allclose(vf0, vf1, atol=1e-6)


def test_detached_cuts_trunk_gradient_but_head_still_trains():
    """A belief-derived loss reshapes the trunk under shaping but NOT under detached; the belief head
    trains in BOTH modes (computed + reinjected + consumed — fully in the system, just trunk-isolated)."""
    obs = _obs(4)
    trunk = ("team_transformer", "pokemon_encoder")

    sh = _model(**_CFG, belief_grad_mode="shaping")
    sh.forward_internal(obs)
    (sh.last_move_belief_logits.float().sum()).backward()
    sh_trunk = _grad_mass(sh, trunk)
    sh_head = _grad_mass(sh, ("move_belief.move_head",))

    dt = _model(**_CFG, belief_grad_mode="detached")
    dt.forward_internal(obs)
    (dt.last_move_belief_logits.float().sum()).backward()
    dt_trunk = _grad_mass(dt, trunk)
    dt_head = _grad_mass(dt, ("move_belief.move_head",))

    assert sh_trunk > 0.0, "shaping: the move-belief loss should reshape the trunk"
    assert dt_trunk == 0.0, "detached: the move-belief loss must NOT reach the trunk"
    assert sh_head > 0.0 and dt_head > 0.0, "the belief head must train in BOTH modes"


def test_detached_spread_and_aux_also_trunk_isolated():
    """The spread-belief output + the species-aux logits don't reach the trunk under detached; the aux
    species head still trains. (The spread head's own training is exercised directly below — under _CFG's
    random obs the opp species resolve to the sentinel whose spread-std≈0, so stat_head's grad is ~0 here
    for reasons unrelated to detach; asserting it on this path would be vacuous.)"""
    obs = _obs(4)
    trunk = ("team_transformer", "pokemon_encoder")
    dt = _model(**_CFG, belief_grad_mode="detached")
    dt.forward_internal(obs)
    loss = dt.last_spread_belief.float().sum() + dt.last_belief_logits["species"].float().sum()
    loss.backward()
    assert _grad_mass(dt, trunk) == 0.0                              # no trunk gradient from the belief
    assert _grad_mass(dt, ("belief_head.species_head",)) > 0.0       # the aux head still trains


def test_detached_spread_head_trains_but_input_is_isolated():
    """Direct SpreadBelief test with a REAL revealed species (non-degenerate std): under detach_read the
    stat_head still gets a gradient from the believed-stat output, while the trunk INPUT is isolated."""
    from agents.model.features_extractor import SpreadBelief, TEAM_SIZE
    from agents import gen3_data
    sb = SpreadBelief(_layout["max_species"])
    sb.detach_read = True
    tok = torch.randn(2, TEAM_SIZE, 128, requires_grad=True)
    species = torch.full((2, TEAM_SIZE), gen3_data.species.get("tyranitar").num, dtype=torch.long)
    _, believed, _, _ = sb(tok, torch.ones(2, TEAM_SIZE, dtype=torch.bool), species)
    believed.float().sum().backward()
    assert sb.stat_head.weight.grad is not None and sb.stat_head.weight.grad.abs().sum() > 0   # head trains
    assert tok.grad is None or tok.grad.abs().sum() == 0                                        # trunk isolated


def test_detached_preserves_normal_trunk_training():
    """The dual invariant: the reinject WRITE keeps the LIVE token identity term, so the policy/value loss
    STILL shapes the trunk under detached — only the BELIEF gradient is cut, not normal training. Guards a
    future refactor that accidentally detaches the write (which would silently sever trunk training)."""
    obs = _obs(4)
    dt = _model(**_CFG, belief_grad_mode="detached")
    pi, vf = dt.forward_internal(obs)
    (pi.float().sum() + vf.float().sum()).backward()
    assert _grad_mass(dt, ("team_transformer",)) > 0.0    # normal trunk training intact under detached


def test_set_belief_grad_mode_updates_all_three_places():
    """The runtime setter (the --allow-belief-grad-mode-change migration fix) must flip the extractor
    attr, _belief_detach, AND every belief head's detach_read — SB3 load reconstructs the extractor
    from the ZIP's saved kwargs, so without this post-load application the migration is a silent
    no-op (grad/*_norm_shared stays 0 under a requested 'shaping')."""
    from types import SimpleNamespace
    from agents.model.features_extractor import Gen3FeaturesExtractor

    fe = Gen3FeaturesExtractor.__new__(Gen3FeaturesExtractor)
    heads = [SimpleNamespace(detach_read=True) for _ in range(3)]
    fe.move_belief, fe.spread_belief, fe.hp_type_belief_head, fe.belief_head = (
        heads[0], heads[1], None, heads[2])
    fe.belief_grad_mode = "detached"
    fe._belief_detach = True

    fe.set_belief_grad_mode("shaping")
    assert fe.belief_grad_mode == "shaping" and fe._belief_detach is False
    assert all(h.detach_read is False for h in heads)

    fe.set_belief_grad_mode("detached")
    assert fe._belief_detach is True and all(h.detach_read is True for h in heads)

    import pytest
    with pytest.raises(ValueError):
        fe.set_belief_grad_mode("bogus")
