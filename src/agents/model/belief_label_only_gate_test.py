"""THE gate for `--belief-grad-mode label_only` (gen3_belief_label_only_v1).

`label_only` makes a promise about a gradient that must NOT exist: **no policy/value gradient reaches
a belief head's parameters**, so the belief is trained by its supervised labels alone. That is a
negative claim over a fan-out — `last_move_belief_logits` alone has eleven forward consumers — so it
cannot be maintained by remembering to detach at each one. The design puts ONE stop-grad at each
head's publish boundary; **this file is what makes that design checkable**, and it is deliberately
written so that a consumer added tomorrow either stays cut or fails here.

Three things it does that a naive version would get wrong:

1. **It builds a REAL `MaskablePPO`.** Ledger M1: SB3's `_build()` orthogonally re-initialises every
   Linear in the extractor, so an invariant asserted on a bare extractor is asserted on a
   construction path production never uses.
2. **It backprops from the ACTION LOGITS, not just `pi_features`.** Under `gen3_pointer_native_v1`
   there is no flat `action_net` — the logits come from the `PointerNativeActionHead` over the
   extractor's `last_pointer_inputs` stash, whose op cells are belief-derived. A test that summed
   `pi_features` would miss that entire route and pass while the real PPO loss still trained the
   heads.
3. **It asserts the SUPERVISED direction too.** A stop-grad that also severed the label path would
   satisfy every "no PPO gradient" assertion perfectly while training nothing at all — and the loss
   value would look completely normal, because the loss is still computed.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.features_extractor import (BELIEF_GRAD_MODES, _BELIEF_SUPERVISION_KEYS,
                                             Gen3FeaturesExtractor)
from agents.model.model_version import _BELIEF_GRAD_MODE_EFFECT
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

# Every forward-consumed belief head ON at once — the configuration whose routes the cut must sever.
_CFG = dict(
    attend_unrevealed_opponents=True, opp_belief_slots=True,
    move_belief_mode="both", move_prior_fusion=True, move_latent=True,
    damage_op=True, damage_outgoing=True, spread_belief=True,
)

# The parameter prefixes of the heads `label_only` protects. `belief_head` is NOT here: it is a pure
# readout with no forward path, so it is covered by the separate structural assertion below.
_PROTECTED = ("move_belief.move_head", "spread_belief.stat_head", "spread_belief.nature_head",
              "spread_belief.ev_head", "hp_type_belief_head.type_head")

# Consumer-side adapters that must KEEP training under label_only — PPO is their ONLY gradient
# source, so detaching at the wrong point (after the matmul instead of at the logits) freezes them.
_ADAPTERS = ("move_belief.reinject", "spread_belief.reinject", "hp_type_belief_head.reinject_proj")


class _Env(gym.Env):
    def __init__(self, dim):
        self.observation_space = gym.spaces.Dict(
            {"observation": gym.spaces.Box(0.0, 1.0, (dim,), np.float32)})
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self._dim = dim

    def reset(self, **kw):
        return {"observation": np.zeros(self._dim, np.float32)}, {}

    def step(self, a):
        return {"observation": np.zeros(self._dim, np.float32)}, 0.0, True, False, {}

    def action_masks(self):
        return np.ones(ACTION_SPACE_SIZE, bool)


def _build_real_policy(**overrides):
    """Construct through the SAME path training uses — MaskablePPO -> ActorCriticPolicy._build()."""
    enc = Gen3ObservationEncoder(load_mappings())
    ek = enc.get_features_extractor_kwargs()
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {**ek, **{k: v for k, v in {**_CFG, **overrides}.items() if k in sig}}
    torch.manual_seed(0)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64])},
    )
    return model, enc


def _obs(enc, n=4):
    torch.manual_seed(1)
    return {"observation": torch.rand(n, enc.dimension)}


def _grad_mass(model, prefixes):
    return sum(p.grad.abs().sum().item()
               for n, p in model.policy.features_extractor.named_parameters()
               if p.grad is not None and any(n.startswith(s) for s in prefixes))


def _ppo_backward(model, enc):
    """Backprop from EVERYTHING the PPO objective actually differentiates.

    The value half and — crucially — the ACTION LOGITS, which under the pointer-native head do NOT
    come from `pi_features`. `evaluate_actions` is the real `train()` entry point, so this exercises
    the identical graph the PPO loss does.
    """
    model.policy.zero_grad(set_to_none=True)
    obs = _obs(enc)
    actions = torch.zeros(obs["observation"].shape[0], dtype=torch.long)
    masks = np.ones((obs["observation"].shape[0], ACTION_SPACE_SIZE), bool)
    values, log_prob, entropy = model.policy.evaluate_actions(obs, actions, action_masks=masks)
    (values.sum() + log_prob.sum() + entropy.sum()).backward()


# --------------------------------------------------------------------------------------------
# The claim itself
# --------------------------------------------------------------------------------------------

def test_ppo_gradient_reaches_the_belief_heads_under_shaping():
    """The control. Without this, the test below could pass because the route never existed."""
    model, enc = _build_real_policy(belief_grad_mode="shaping")
    _ppo_backward(model, enc)
    assert _grad_mass(model, _PROTECTED) > 0.0, (
        "no PPO gradient reached the belief heads even under `shaping` — the config built no "
        "forward-consumed belief head, so the label_only assertion below would be vacuous."
    )


def test_no_ppo_gradient_reaches_the_belief_heads_under_label_only():
    """THE gate: the policy/value objective must not train a belief head's parameters.

    Fails if any consumer route is left live — including one added later, since the cut is at the
    publish boundary rather than at the consumers.
    """
    model, enc = _build_real_policy(belief_grad_mode="label_only")
    _ppo_backward(model, enc)
    fe = model.policy.features_extractor
    live = {n: p.grad.abs().sum().item()
            for n, p in fe.named_parameters()
            if p.grad is not None and any(n.startswith(s) for s in _PROTECTED)
            and p.grad.abs().sum().item() > 0.0}
    assert not live, (
        f"label_only leaked policy/value gradient into {len(live)} belief-head parameter(s): "
        f"{sorted(live)}. A forward consumer is reading a LIVE belief tensor — publish it through "
        "`_publish_belief` (or, for a head's own reinjection, honour `publish_detach`)."
    )


def test_label_only_still_trains_the_consumer_side_adapters():
    """The reinjection adapters have NO supervised loss — PPO is their only gradient source.

    This is the assertion that forces the detach to sit on the LOGITS rather than on `soft_emb`:
    detaching after the matmul would zero these (and the shared embedding tables) silently.
    """
    model, enc = _build_real_policy(belief_grad_mode="label_only")
    _ppo_backward(model, enc)
    for name in _ADAPTERS:
        assert _grad_mass(model, (name,)) > 0.0, (
            f"{name} received no gradient under label_only. The detach is too far downstream — it "
            "must cut the belief LOGITS, not the tensor the adapter consumes."
        )


def test_label_only_still_trains_the_shared_trunk():
    """`label_only` cuts route C, not route D. Normal policy training must be untouched."""
    model, enc = _build_real_policy(belief_grad_mode="label_only")
    _ppo_backward(model, enc)
    assert _grad_mass(model, ("team_transformer", "pokemon_encoder")) > 0.0


@pytest.mark.parametrize("key,prefix", [
    ("move_belief_logits", "move_belief.move_head"),
    ("hp_type_logits", "hp_type_belief_head.type_head"),
])
def test_every_belief_loss_still_trains_its_head(key, prefix):
    """The SUPERVISED direction — the failure a "no PPO gradient" test cannot see.

    A cut that also severed the label path would pass every assertion above while training nothing,
    and the loss value would look perfectly healthy because it is still computed. So each
    supervision view must deposit gradient on its own head.
    """
    model, enc = _build_real_policy(belief_grad_mode="label_only")
    fe = model.policy.features_extractor
    model.policy.zero_grad(set_to_none=True)
    fe(_obs(enc))
    t = fe.belief_supervision(key)
    assert t is not None, f"{key} was not registered by the forward"
    t.float().sum().backward()
    assert _grad_mass(model, (prefix,)) > 0.0, (
        f"the supervised loss for {key} reaches no parameter of {prefix} — the label path was cut "
        "along with the policy path, so this head now trains on NOTHING."
    )


def test_spread_label_path_is_live_while_its_reinject_is_cut():
    """`SpreadBelief` gets its own direct test — the policy-level fixture CANNOT express it.

    Two degeneracies make a whole-policy assertion vacuous here, and both bit while writing this:
    a random obs yields an unknown species whose prior std is 0 (so `believed = mean + delta*std`
    carries no gradient in ANY mode), and summing the module's LayerNorm'd output is a CONSTANT
    (`sum(LayerNorm(x))` is independent of x), so the reinject route reads as dead even under
    `shaping`. Hence a REAL species and a random projection — and both directions asserted, since
    each of those degeneracies would have made the cut look like it worked when nothing was tested.
    """
    from agents import gen3_data
    from agents.model.features_extractor import TEAM_SIZE, SpreadBelief

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    species = torch.full((2, TEAM_SIZE), gen3_data.species.get("tyranitar").num, dtype=torch.long)
    mask = torch.ones(2, TEAM_SIZE, dtype=torch.bool)
    torch.manual_seed(0)
    proj = torch.randn(2, TEAM_SIZE, 128)          # non-degenerate readout of the enriched tokens

    def _routes(publish_detach):
        sb = SpreadBelief(layout["max_species"])
        sb.publish_detach = publish_detach
        torch.nn.init.zeros_(sb.stat_head.weight)  # the production cold start (posterior == prior)
        torch.nn.init.zeros_(sb.stat_head.bias)
        tok = torch.randn(2, TEAM_SIZE, 128)
        _, believed, _, _ = sb(tok, mask, species)
        believed.float().sum().backward()          # the SUPERVISED loss reads `believed`
        label = sb.stat_head.weight.grad.abs().sum().item()
        sb.zero_grad(set_to_none=True)
        enriched, _, _, _ = sb(tok, mask, species)
        (enriched * proj).sum().backward()         # the POLICY reads the enriched TOKENS
        g = sb.stat_head.weight.grad                # None once cut — autograd never reaches it
        return label, (0.0 if g is None else g.abs().sum().item())

    label_on, ppo_on = _routes(False)
    label_cut, ppo_cut = _routes(True)
    assert ppo_on > 0.0, "the policy route to stat_head does not exist even when live — vacuous test"
    assert ppo_cut == 0.0, f"label_only left the spread reinject route live ({ppo_cut})"
    assert label_on > 0.0 and label_cut > 0.0, "the supervised route to stat_head was cut too"


def test_supervision_view_is_live_while_the_stash_is_published():
    """The two must be the same VALUE and differ only in the graph — that is what makes the forward
    bit-identical while the backward changes."""
    model, enc = _build_real_policy(belief_grad_mode="label_only")
    fe = model.policy.features_extractor
    fe(_obs(enc))
    live, published = fe.belief_supervision("move_belief_logits"), fe.last_move_belief_logits
    assert live.requires_grad and not published.requires_grad
    assert torch.equal(live, published)


def test_shaping_and_detached_publish_the_identical_object():
    """No mode but label_only may introduce a copy — otherwise "byte-identical" is a claim about a
    different tensor than the one consumers read."""
    for mode in ("shaping", "detached"):
        model, enc = _build_real_policy(belief_grad_mode=mode)
        fe = model.policy.features_extractor
        fe(_obs(enc))
        assert fe.belief_supervision("move_belief_logits") is fe.last_move_belief_logits


def test_forward_is_bit_identical_across_all_three_modes():
    """`detach()` changes the graph, never a value — so the same weights must produce the same
    forward in every mode. This is what makes the flip weight-safe on a converged checkpoint."""
    model, enc = _build_real_policy(belief_grad_mode="shaping")
    fe, obs = model.policy.features_extractor, _obs(enc)
    with torch.no_grad():
        ref = [t.clone() for t in fe(obs)]
    for mode in ("detached", "label_only"):
        fe.set_belief_grad_mode(mode)
        with torch.no_grad():
            got = fe(obs)
        for a, b in zip(ref, got):
            assert torch.equal(a, b), f"forward changed under {mode}"


# --------------------------------------------------------------------------------------------
# The heads that need no cut — asserted, not assumed
# --------------------------------------------------------------------------------------------

def test_readout_heads_are_structurally_label_only_in_every_mode():
    """`BeliefHead` & co. are side readouts whose output never re-enters the forward, so no policy
    gradient can reach them in ANY mode and `label_only` is a no-op for them.

    Asserting it is the point: if one ever starts feeding forward, it silently rejoins the PPO
    objective — and being outside `_BELIEF_SUPERVISION_KEYS`, nothing else would notice.
    """
    for mode in BELIEF_GRAD_MODES:
        model, enc = _build_real_policy(belief_grad_mode=mode)
        _ppo_backward(model, enc)
        mass = _grad_mass(model, ("belief_head.",))
        assert mass == 0.0, (
            f"belief_head received {mass} of policy/value gradient under {mode!r} — it is no longer "
            "a pure readout. It now needs a publish boundary like the other belief heads."
        )


# --------------------------------------------------------------------------------------------
# Contracts that keep the machinery honest
# --------------------------------------------------------------------------------------------

def test_alpha_is_cut_under_intent_value_reduce():
    """`AlphaIntentHead` is the head that would have been missed.

    It is a pure readout UNTIL `--intent-value-reduce`, which appends an alpha-weighted threat term
    to the CRITIC half — so that flag, and nothing about alpha itself, is what puts the value
    gradient on `alpha_head`. Both directions are asserted on the same config, because "cut" is only
    meaningful next to a live control.
    """
    # intent_value_reduce gathers the op's held pair cells onto alpha's seat axis, so it needs the
    # top-K candidate index — i.e. the incoming matrix (which needs move_latent, already in _CFG).
    intent = dict(opp_intent=True, entity_topk_seats=6, intent_value_reduce=True,
                  damage_topk_k=6, damage_matrices_incoming=True)

    def _alpha_grad(mode):
        model, enc = _build_real_policy(belief_grad_mode=mode, **intent)
        # `IntentValueReduce.proj` is ZERO-INIT, so at init no gradient reaches alpha through it and
        # BOTH arms would read 0 — the route is closed at step 0 and opens as proj trains. Gate the
        # trained regime against a large random projection, not just at init (the standing rule for
        # the zero-init couplings here); otherwise "cut" would be indistinguishable from "not yet
        # connected" and this test would pass no matter what label_only did.
        with torch.no_grad():
            model.policy.features_extractor.intent_value_reduce.proj.weight.normal_(0.0, 0.5)
        _ppo_backward(model, enc)
        return _grad_mass(model, ("alpha_head",)), model, enc

    live, _, _ = _alpha_grad("shaping")
    assert live > 0.0, (
        "no value gradient reached alpha_head even under shaping — --intent-value-reduce did not "
        "build the route, so the cut asserted below would be vacuous."
    )

    cut, model, enc = _alpha_grad("label_only")
    assert cut == 0.0, (
        f"label_only left the intent-value-reduce route to alpha_head live ({cut} vs {live} under "
        "shaping) — the critic can still train the intent head."
    )

    # ...and its own supervised loss must still reach it.
    fe = model.policy.features_extractor
    model.policy.zero_grad(set_to_none=True)
    fe(_obs(enc))
    fe.belief_supervision("alpha_logits").float().sum().backward()
    assert _grad_mass(model, ("alpha_head",)) > 0.0, "the intent loss no longer trains alpha_head"


def test_supervision_keys_are_exactly_what_the_forward_registers():
    """A key in the frozenset but never registered is a loss silently reading `None`; one registered
    but absent from the set cannot be read at all (the accessor raises)."""
    model, enc = _build_real_policy(belief_grad_mode="label_only")
    fe = model.policy.features_extractor
    fe(_obs(enc))
    registered = set(fe._belief_supervision)
    missing = _BELIEF_SUPERVISION_KEYS - registered
    # alpha/beta_logits need --opp-intent-coef, which this config does not build.
    assert missing <= {"alpha_logits", "beta_logits"}, \
        f"declared but never registered: {sorted(missing)}"
    assert not registered - _BELIEF_SUPERVISION_KEYS, (
        f"registered but undeclared (unreadable via belief_supervision): "
        f"{sorted(registered - _BELIEF_SUPERVISION_KEYS)}")


def test_unknown_supervision_key_raises():
    model, _ = _build_real_policy(belief_grad_mode="label_only")
    with pytest.raises(KeyError, match="unknown belief supervision key"):
        model.policy.features_extractor.belief_supervision("no_such_belief")


def test_every_mode_has_a_migration_notice():
    """`check_belief_grad_mode` prints a per-mode effect line; a fourth mode must not silently fall
    back to the generic string."""
    assert set(_BELIEF_GRAD_MODE_EFFECT) == set(BELIEF_GRAD_MODES)


def test_invalid_mode_is_rejected_at_both_entry_points():
    enc = Gen3ObservationEncoder(load_mappings())
    with pytest.raises(ValueError, match="belief_grad_mode must be one of"):
        _build_real_policy(belief_grad_mode="bogus")
    model, _ = _build_real_policy(belief_grad_mode="shaping")
    with pytest.raises(ValueError, match="belief_grad_mode must be one of"):
        model.policy.features_extractor.set_belief_grad_mode("bogus")
    assert enc is not None
