"""Unit tests for the FROZEN pre-attention move belief (`move_belief_single_compute`, v47).

The claim under test: the belief is computed EXACTLY ONCE per forward (pre-attention, at the prefuse
reinjection) and the between-layers refine kernels REUSE that posterior instead of re-reading the
move-belief head off the being-enriched tokens. So:

    belief ONCE  ->  physics ONCE  ->  N attention layers that CANNOT revise the belief.

Pins: the forward-behavior nature (no new modules), the dependency guard (requires prefuse), that the
head is genuinely called ONE fewer time, that the refine physics consumes the SAME tensor the
reinjection produced, that it is a REAL forward change vs the per-round re-read (not a no-op), that
OFF is byte-for-byte, that gradient still reaches the head through the frozen path, and the
version/migration gate.
"""
import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()

# The toggle only has meaning with a refine loop to feed: prefuse + damage_op + refine rounds.
_BASE = dict(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
             move_belief_prefuse=True, damage_op=True, damage_refine_rounds=2)


def _make_model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mappings, **kw)


def _obs(batch=4, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(batch, _layout["total_dim"], generator=g)}


# --------------------------------------------------------------------------- structure / guards
def test_single_compute_adds_no_modules():
    """Forward-behavior only: same MoveBelief params, so state_dict keys + projection widths match."""
    on = _make_model(**_BASE, move_belief_single_compute=True)
    off = _make_model(**_BASE, move_belief_single_compute=False)
    assert on.move_belief_single_compute is True and off.move_belief_single_compute is False
    assert set(on.state_dict()) == set(off.state_dict())
    assert on.projection_input_dim == off.projection_input_dim
    assert on.value_projection_input_dim == off.value_projection_input_dim


def test_single_compute_requires_prefuse():
    """Without prefuse the only belief is POST-transformer, so there is nothing to reuse — the flag
    would silently be a no-op. Fail loud instead."""
    kw = dict(_BASE)
    kw["move_belief_prefuse"] = False
    with pytest.raises(ValueError, match="move_belief_single_compute"):
        _make_model(**kw, move_belief_single_compute=True)
    _make_model(**_BASE, move_belief_single_compute=True)   # valid with prefuse on


# --------------------------------------------------------------------------- the core claim
def test_belief_head_called_exactly_once_per_forward():
    """THE headline property. With 2 refine rounds the default computes the belief 3× (prefuse + one
    re-read per round); frozen computes it ONCE."""
    def _count(single):
        m = _make_model(**_BASE, move_belief_single_compute=single).eval()
        calls = [0]
        real = m.move_belief.move_logits

        def counted(*a, **kw):
            calls[0] += 1
            return real(*a, **kw)
        m.move_belief.move_logits = counted
        with torch.no_grad():
            m(_obs())
        return calls[0]

    frozen, per_round = _count(True), _count(False)
    assert frozen == 1, f"frozen belief should call the head exactly once, got {frozen}"
    assert per_round == 1 + _BASE["damage_refine_rounds"], per_round
    assert frozen < per_round


def test_refine_consumes_the_same_tensor_the_reinjection_produced():
    """"Once" must mean the SAME posterior object reaches the physics — not merely a second call that
    happens to agree. Capture what discrete_incoming was handed and assert identity with the stash."""
    m = _make_model(**_BASE, move_belief_single_compute=True).eval()
    seen = []
    real = m.damage_op.discrete_incoming

    def spy(ctx, logits, cand=None):
        seen.append(logits)
        return real(ctx, logits, cand)
    m.damage_op.discrete_incoming = spy
    with torch.no_grad():
        m(_obs())
    assert seen, "the refine loop never ran"
    # every round got the identical tensor, and it is the stashed pre-attention posterior
    assert all(t is seen[0] for t in seen)
    assert seen[0] is m.last_move_belief_logits


def _simulate_trained(model):
    """At cold start this toggle is a no-op for TWO independent reasons, both deliberate:

      1. under `--move-prior-fusion`, `MoveBelief.move_head` is ZERO-init so the posterior == the
         Smogon prior — token-INDEPENDENT, so re-reading it off enriched tokens returns the same
         thing the pre-attention read did;
      2. `refine_proj` (and its status/outgoing siblings) are ZERO-init (identity-at-init), so the
         refine injection is multiplied by zero regardless.

    Both must train away from zero before frozen-vs-per-round can differ at all. Perturb both so the
    forward test measures the MECHANISM rather than the initialization."""
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        head = model.move_belief.move_head
        head.weight.copy_(torch.rand(head.weight.shape, generator=g) * 0.1 - 0.05)
        for name in ("refine_proj", "status_in_proj", "outgoing_proj", "status_out_proj"):
            proj = getattr(model, name, None)
            if proj is not None:
                proj.weight.copy_(torch.rand(proj.weight.shape, generator=g) * 0.5 - 0.25)


def test_frozen_belief_changes_the_forward():
    """A real computational change, not a reorder: re-reading the belief off the ENRICHED tokens
    yields a different posterior (hence different physics) than the frozen pre-attention one.
    Measured with the belief head + refine projection trained away from zero (see
    _simulate_trained) — at cold start the toggle is provably inert, which the sibling test pins."""
    torch.manual_seed(0)
    on = _make_model(**_BASE, move_belief_single_compute=True).eval()
    off = _make_model(**_BASE, move_belief_single_compute=False).eval()
    off.load_state_dict(on.state_dict())          # identical weights ⇒ any delta is the toggle
    _simulate_trained(on)
    _simulate_trained(off)                        # same perturbation (same seed) on both
    with torch.no_grad():
        pi_on, vf_on = on(_obs())
        pi_off, vf_off = off(_obs())
    assert not torch.allclose(pi_on, pi_off, atol=1e-6), "frozen vs per-round belief must differ"
    assert torch.isfinite(pi_on).all() and torch.isfinite(vf_on).all()


def test_identity_at_init_forward_equals_per_round():
    """The flip side, pinned deliberately: untouched, frozen and per-round are BYTE-IDENTICAL — the
    zero-init `move_head` (under prior fusion) makes the posterior token-independent, and the
    zero-init `refine_proj` multiplies the injection by zero. So enabling the flag is risk-free at
    step 0 and can only diverge as those paths learn. If this test ever fails, one of those two
    zero-inits changed and the toggle's cold-start guarantee is gone."""
    torch.manual_seed(0)
    on = _make_model(**_BASE, move_belief_single_compute=True).eval()
    off = _make_model(**_BASE, move_belief_single_compute=False).eval()
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        pi_on, vf_on = on(_obs())
        pi_off, vf_off = off(_obs())
    assert torch.equal(pi_on, pi_off) and torch.equal(vf_on, vf_off)


def test_off_is_byte_identical_to_the_untoggled_model():
    """OFF must reproduce the v46 forward exactly (the flag defaults off; no silent drift)."""
    torch.manual_seed(0)
    a = _make_model(**_BASE, move_belief_single_compute=False).eval()
    torch.manual_seed(0)
    b = _make_model(**_BASE).eval()                # flag omitted entirely
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        pi_a, vf_a = a(_obs())
        pi_b, vf_b = b(_obs())
    assert torch.equal(pi_a, pi_b) and torch.equal(vf_a, vf_b)


def test_grad_reaches_the_move_head_through_the_frozen_path():
    """Freezing must not detach: the stash is live, so the refine physics gradient still trains the
    belief head (one posterior, one gradient path)."""
    m = _make_model(**_BASE, move_belief_single_compute=True).train()
    pi, vf = m(_obs())
    (pi.sum() + vf.sum()).backward()
    grads = [p.grad for p in m.move_belief.parameters() if p.grad is not None]
    assert grads, "no gradient reached the move-belief head"
    assert any(g.abs().sum() > 0 for g in grads)


def test_single_round_gives_belief_once_physics_once_two_attention_rounds():
    """The configuration the design targets: refine_rounds=1 fires the callback only before layer 0
    (pre-attention), so both transformer layers then run over frozen physics."""
    kw = dict(_BASE)
    kw["damage_refine_rounds"] = 1
    m = _make_model(**kw, move_belief_single_compute=True).eval()
    rounds, belief_calls = [], [0]
    real_inc, real_logits = m.damage_op.discrete_incoming, m.move_belief.move_logits

    def spy_inc(ctx, logits, cand=None):
        rounds.append(1)
        return real_inc(ctx, logits, cand)

    def spy_logits(*a, **kw_):
        belief_calls[0] += 1
        return real_logits(*a, **kw_)
    m.damage_op.discrete_incoming, m.move_belief.move_logits = spy_inc, spy_logits
    with torch.no_grad():
        m(_obs())
    assert belief_calls[0] == 1, "belief must be computed once"
    assert len(rounds) == 1, "physics must be computed once"


# --------------------------------------------------------------------------- versioning
def _mv(single):
    policy_kwargs = {"features_extractor_kwargs": {"move_belief_single_compute": single}}
    return ModelVersion.from_layout_and_policy_kwargs(_layout, policy_kwargs)


def test_version_gate_and_migration():
    assert MODEL_CONFIG_VERSION >= 47
    # a resume that flips the toggle is a FATAL forward mismatch, both directions
    on, off = _mv(True), _mv(False)
    with pytest.raises(ModelVersionError, match="move_belief_single_compute"):
        on.check_compatible(off)
    with pytest.raises(ModelVersionError, match="move_belief_single_compute"):
        off.check_compatible(on)
    on.check_compatible(_mv(True))                # like-for-like passes
    # an old config migrates to the byte-identical default
    migrated = _migrate_config({"config_version": 46})
    assert migrated["move_belief_single_compute"] is False
    assert migrated["config_version"] >= 47
