"""The PER-ACTION win-prob head (v107, `gen3_q_winprob_head_v1`).

What is pinned here, in order of how badly a silent break would hurt:

1. **OFF is byte-identical and ON is BIT-identical in pi/vf.** `none` builds no module at all;
   `read_only` builds one whose every input is detached and whose only output is a stash. Both
   claims die the moment someone "tidies" the constructor and the head stops being built LAST —
   which re-rolls the initialization RNG stream for nothing, because it is already last, and would
   re-roll it for everything if it were not.
2. **The column order IS the action space.** A Q head whose index `a` did not mean action `a`
   would be the order-mismatch bug class with no shape error anywhere to catch it — every label
   would train the wrong column and every metric would look healthy.
3. **The zero-init cold start is a total tie.** P(win|s,a) = 0.5 exactly, for every action, on a
   REAL `MaskablePPO`-built policy — because SB3's ortho-init clobbers zero-inits inside the
   extractor and a claim asserted only on a bare module is a claim about a construction path
   training does not use (`gen3_identity_init_guard_v1`).
4. **It cannot perturb the trunk at any coefficient.** There is no `shaping` mode; the detachment
   is inside the forward, so this is a structural property and is tested as one.
5. **Equivariance within a family.** One shared scorer means permuting our team permutes the six
   switch Q values with it. A head that memorised "slot 0" would be worthless as a search leaf.
6. **The version gate fires.** The head's only output is a stash, so a flipped flag produces no
   shape error: `check_compatible` is the only thing between a resume and a run that silently
   stops training (or starts supervising a freshly random) head.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from agents.action.constants import (ACTION_SPACE_SIZE, MOVE_START, N_MOVE_SLOTS, N_SWITCH_SLOTS,
                                     STRUGGLE, SWITCH_START)
from agents.model.arch_constants import D_MODEL
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError,
                                        _migrate_config)
from agents.model.q_winprob_head import Q_WINPROB_MODES, QWinProbHead
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_MC, _SC = 3, 5           # arbitrary nonzero pointer-cell widths for the standalone module tests


@pytest.fixture(scope="module")
def ek_and_space():
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    total = ek["layout"]["total_dim"]
    space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (total,), np.float32),
        "action_mask": spaces.Box(0, 1, (ACTION_SPACE_SIZE,), np.int8),
    })
    return ek, space, total


def _build(ek, space, seed=0, **flags):
    torch.manual_seed(seed)
    return Gen3FeaturesExtractor(space, **{**ek, **flags}).eval()


def _head(seed=0, **kw):
    torch.manual_seed(seed)
    return QWinProbHead(move_token_dim=D_MODEL, d_model=D_MODEL, ctx_dim=D_MODEL,
                        move_cell_dim=_MC, switch_cell_dim=_SC, **kw)


def _inputs(B=4, move_valid=None, team=None, generator=None):
    g = generator
    r = (lambda *s: torch.randn(*s, generator=g)) if g is not None else torch.randn
    return dict(
        ctx_vec=r(B, D_MODEL),
        move_tokens_req=r(B, N_MOVE_SLOTS, D_MODEL),
        move_valid=torch.ones(B, N_MOVE_SLOTS) if move_valid is None else move_valid,
        team_tokens=r(B, N_SWITCH_SLOTS, D_MODEL) if team is None else team,
        move_cells=r(B, N_MOVE_SLOTS, _MC),
        switch_cells=r(B, N_SWITCH_SLOTS, _SC),
    )


# ── the module's own contract ─────────────────────────────────────────────────

def test_the_output_is_the_action_space_in_order():
    """Index `a` IS action `a`. The three families land in `[switch x6, move x4, struggle]`, which
    is what makes a label row's `action` field, the policy's logits and the action mask the same
    index — the whole interface, and unrepresentable to check any other way than by construction."""
    head = _head()
    out = head(**_inputs())
    assert out.shape[-1] == ACTION_SPACE_SIZE
    assert (SWITCH_START, N_SWITCH_SLOTS, MOVE_START, N_MOVE_SLOTS, STRUGGLE) == (0, 6, 6, 4, 10)
    # A move slot marked invalid must zero exactly ITS column and no other. That pins the layout
    # from the inside: a head whose move block sat anywhere else would move the wrong zero.
    mv = torch.ones(2, N_MOVE_SLOTS)
    mv[:, 1] = 0.0
    torch.manual_seed(1)
    trained = _head()
    torch.nn.init.normal_(trained.q_score.weight)     # break the zero-init so 0 is informative
    torch.nn.init.normal_(trained.q_score.bias)
    got = trained(**_inputs(B=2, move_valid=mv))
    assert torch.all(got[:, MOVE_START + 1] == 0.0)
    assert torch.all(got[:, MOVE_START] != 0.0)


def test_zero_init_makes_every_action_exactly_one_half():
    """The honest cold start: a head that has seen no label ranks nothing. EXACT equality, not
    approximate — the scorer's weight AND bias are zeroed, so the logits are literally 0."""
    probs = torch.sigmoid(_head()(**_inputs(B=8)))
    assert torch.equal(probs, torch.full_like(probs, 0.5))


def test_one_shared_scorer_not_three():
    """The parameter sharing IS the equivariance claim. Three per-family scorers (the pointer
    head's shape, where they are correct because each family's logit has its own semantics) would
    let the Q head learn a different value function per family from the same evidence."""
    head = _head()
    scorers = [n for n, _ in head.named_parameters() if "q_score" in n]
    assert sorted(scorers) == ["q_score.bias", "q_score.weight"]


def test_switch_scores_are_EQUIVARIANT_under_permuting_our_team():
    """Permute our team tokens ⇒ the six switch Q values permute with them. A head that memorised
    "slot 0 is usually best" from an ordering that means nothing would be useless as a search leaf,
    and it is exactly what a flat `Linear(ctx, 11)` learns."""
    g = torch.Generator().manual_seed(7)
    head = _head(seed=3)
    # A zero-init scorer is trivially equivariant, so break it first: the property has to hold for
    # a head that actually discriminates.
    torch.nn.init.normal_(head.q_score.weight, std=0.5)
    torch.nn.init.normal_(head.q_score.bias, std=0.5)
    base = _inputs(B=2, generator=g)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    swapped = dict(base)
    swapped["team_tokens"] = base["team_tokens"][:, perm]
    swapped["switch_cells"] = base["switch_cells"][:, perm]
    a = head(**base)[:, SWITCH_START:N_SWITCH_SLOTS]
    b = head(**swapped)[:, SWITCH_START:N_SWITCH_SLOTS]
    assert torch.allclose(a[:, perm], b, atol=1e-6)


def test_mode_none_is_not_a_constructible_head_and_a_typo_raises(ek_and_space):
    ek, space, _t = ek_and_space
    assert Q_WINPROB_MODES == ("none", "read_only")
    with pytest.raises(ValueError, match="q_winprob_mode"):
        _build(ek, space, q_winprob_mode="shaping")


# ── the extractor integration ─────────────────────────────────────────────────

def test_off_builds_nothing(ek_and_space):
    ek, space, _t = ek_and_space
    off = _build(ek, space)
    assert off.q_winprob_head is None and off.q_winprob_mode == "none"
    assert off.last_q_winprob_logits is None


def test_on_adds_ONLY_its_own_parameters(ek_and_space):
    ek, space, _t = ek_and_space
    off, on = _build(ek, space), _build(ek, space, q_winprob_mode="read_only")
    delta = (sum(p.numel() for p in on.parameters())
             - sum(p.numel() for p in off.parameters()))
    own = sum(p.numel() for p in on.q_winprob_head.parameters())
    assert delta == own > 0


def test_the_state_dict_KEY_CENSUS_moves_only_by_the_head(ek_and_space):
    """The census, both directions: OFF has no `q_winprob_head.*` key at all, ON adds exactly that
    prefix and touches nothing else. A head that leaked a key elsewhere (or renamed a sibling's)
    would break every resume in the generation with an opaque torch error."""
    ek, space, _t = ek_and_space
    off, on = _build(ek, space), _build(ek, space, q_winprob_mode="read_only")
    off_keys, on_keys = set(off.state_dict()), set(on.state_dict())
    assert not any(k.startswith("q_winprob_head.") for k in off_keys)
    added = on_keys - off_keys
    assert added and all(k.startswith("q_winprob_head.") for k in added)
    assert not (off_keys - on_keys), "building the Q head REMOVED a state_dict key"


def test_off_is_BYTE_identical_and_ON_is_BIT_identical_in_pi_and_vf(ek_and_space):
    """The established checksum pattern. OFF vs OFF is the control (the harness itself must be
    deterministic); OFF vs ON is the claim — and it is the strong form, because the head is built
    LAST and therefore re-rolls no earlier module's initialization draw."""
    ek, space, total = ek_and_space
    obs = {"observation": torch.zeros(3, total)}
    a, b = _build(ek, space), _build(ek, space)
    on = _build(ek, space, q_winprob_mode="read_only")
    with torch.no_grad():
        pa, pb, pon = a(obs), b(obs), on(obs)
    assert all(torch.equal(x, y) for x, y in zip(pa, pb)), "the harness is not deterministic"
    assert all(torch.equal(x, y) for x, y in zip(pa, pon)), \
        "building the Q head perturbed pi/vf — is it still built LAST in __init__?"


def test_APPEND_NEVER_INSERT_every_prior_module_draws_the_same_init(ek_and_space):
    """The append-never-insert rule, asserted on the PARAMETERS rather than on the output.

    SB3 restores optimizer state POSITIONALLY (the ai_v6_13 "128 vs 5" crash), and a module
    inserted mid-constructor shifts every subsequent module's initialization RNG draw. pi/vf
    byte-identity is strong evidence but not proof — a shifted draw could in principle produce the
    same forward on a zero observation. This compares every PRIOR parameter tensor directly, and
    additionally pins that the prior parameters keep their ORDER, which is the thing the optimizer
    actually indexes by.
    """
    ek, space, _t = ek_and_space
    off, on = _build(ek, space), _build(ek, space, q_winprob_mode="read_only")
    off_named = list(off.named_parameters())
    on_named = [(n, p) for n, p in on.named_parameters() if not n.startswith("q_winprob_head.")]
    assert [n for n, _ in off_named] == [n for n, _ in on_named], \
        "the Q head changed the parameter ORDER of the modules before it"
    differing = [n for (n, a), (_m, b) in zip(off_named, on_named) if not torch.equal(a, b)]
    assert not differing, (
        f"{len(differing)} prior parameter(s) drew a different init when the Q head was built "
        f"(first few: {differing[:5]}) — it must be APPENDED last, never inserted")


def test_the_forward_STASHES_the_logits_and_feeds_them_nowhere(ek_and_space):
    """The stash/read seam. The head IS called by the forward (unlike the four cf readouts), so the
    contract is not "never runs" but "runs and publishes only": the logits land on
    `last_q_winprob_logits`, the stash is replaced every forward, and pi/vf never see it."""
    ek, space, total = ek_and_space
    fe = _build(ek, space, q_winprob_mode="read_only")
    calls = []
    fe.q_winprob_head.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        fe({"observation": torch.zeros(2, total)})
    assert calls == [1], "the Q head must run exactly once per forward"
    logits = fe.last_q_winprob_logits
    assert logits is not None and logits.shape == (2, ACTION_SPACE_SIZE)
    assert logits is fe.stash.q_winprob_logits
    # Replaced, never carried: a stale batch's Q row is unrepresentable.
    with torch.no_grad():
        fe({"observation": torch.zeros(5, total)})
    assert fe.last_q_winprob_logits.shape == (5, ACTION_SPACE_SIZE)


def test_read_only_cannot_reach_the_trunk_at_any_weight(ek_and_space):
    """No `shaping` mode exists, and the detachment is INSIDE the forward — so this is structural.
    Backprop a large arbitrary function of the Q logits and assert not one trunk parameter takes a
    gradient. (The head's OWN parameters must still take one, or the term would train nothing —
    both halves are checked, because either failure alone is silent.)"""
    ek, space, total = ek_and_space
    fe = _build(ek, space, q_winprob_mode="read_only")
    fe.zero_grad(set_to_none=True)
    fe({"observation": torch.zeros(2, total)})
    (fe.last_q_winprob_logits.sum() * 1000.0).backward()
    trunk_with_grad = [n for n, p in fe.named_parameters()
                       if not n.startswith("q_winprob_head.") and p.grad is not None
                       and bool(p.grad.abs().sum() > 0)]
    assert not trunk_with_grad, f"the Q head reached the trunk: {trunk_with_grad[:5]}"
    head_with_grad = [n for n, p in fe.q_winprob_head.named_parameters()
                      if p.grad is not None and bool(p.grad.abs().sum() > 0)]
    assert head_with_grad, "the Q head's own parameters took no gradient — it trains nothing"


def test_the_head_scores_the_SAME_tokens_the_pointer_head_scores(ek_and_space):
    """The amortization claim's precondition. If the Q head read anything other than the pointer
    stash, "one forward, eleven win probs from the action tokens" would be a different sentence —
    and the per-action semantics would have no anchor."""
    ek, space, total = ek_and_space
    fe = _build(ek, space, q_winprob_mode="read_only")
    with torch.no_grad():
        fe({"observation": torch.zeros(2, total)})
        pi_in = fe.last_pointer_inputs
        replay = fe.q_winprob_head(
            fe.stash.value_pooled, pi_in.move_tokens, pi_in.move_valid,
            pi_in.team_tokens, pi_in.move_cells, pi_in.switch_cells)
    assert torch.equal(replay, fe.last_q_winprob_logits)


def test_the_pointer_CELL_WIDTHS_are_respected(ek_and_space):
    """The head is built LAST for a SECOND reason: it sizes its projections from the pointer cell
    widths, which every op / intent / pair-outcome module widens. A stale width would be a shape
    error at the first forward of a flag combination nobody ran locally."""
    ek, space, _t = ek_and_space
    fe = _build(ek, space, q_winprob_mode="read_only")
    assert fe.q_winprob_head.move_cell_dim == fe.pointer_move_cell_dim
    assert fe.q_winprob_head.switch_cell_dim == fe.pointer_switch_cell_dim


def test_the_cold_start_TIE_survives_a_REAL_MaskablePPO_build():
    """The init-state sanity, on the construction path TRAINING actually uses.

    `gen3_identity_init_guard_v1`: SB3's `_build()` orthogonally re-initialises every `nn.Linear`
    inside the extractor, so "zero-init ⇒ uniform" asserted on a bare module is a claim about a
    path production does not take — measured max|W| before that guard existed was 0.19-0.47. The
    guard captures its protected set BY OBSERVATION at the end of `__init__`, so `q_score` is
    covered automatically; this test is what proves the automatic coverage actually reached it,
    because "it should be picked up" is exactly the assumption that class of bug is made of.
    """
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.policy import Gen3DualHeadMaskablePolicy

    enc = Gen3ObservationEncoder(load_mappings())
    dim = enc.dimension

    class _Env(gym.Env):
        observation_space = gym.spaces.Dict(
            {"observation": gym.spaces.Box(0.0, 1.0, (dim,), np.float32)})
        action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        def reset(self, **kw):
            return {"observation": np.zeros(dim, np.float32)}, {}

        def step(self, a):
            return {"observation": np.zeros(dim, np.float32)}, 0.0, True, False, {}

        def action_masks(self):
            return np.ones(ACTION_SPACE_SIZE, bool)

    torch.manual_seed(0)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env()]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": {
                **enc.get_features_extractor_kwargs(), "q_winprob_mode": "read_only"},
            "net_arch": dict(pi=[64], vf=[64])},
    )
    fe = model.policy.features_extractor
    assert not fe.q_winprob_head.q_score.weight.any(), \
        "SB3's ortho-init clobbered the Q scorer — restore_identity_init did not cover it"
    with torch.no_grad():
        fe({"observation": torch.zeros(2, dim)})
    probs = torch.sigmoid(fe.last_q_winprob_logits)
    assert torch.equal(probs, torch.full_like(probs, 0.5))


# ── the v107 version gate ─────────────────────────────────────────────────────

def _ver(**flags):
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    ek.update(flags)
    pk = {"features_extractor_class": Gen3FeaturesExtractor, "features_extractor_kwargs": ek,
          "net_arch": [512, 512]}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk)


def test_version_records_the_toggle():
    assert _ver(q_winprob_mode="read_only").q_winprob_mode == "read_only"
    assert _ver().q_winprob_mode == "none"


def test_check_compatible_REJECTS_a_flipped_mode():
    """The only gate there is. The head's output is a stash, so nothing downstream would fail: a
    resume that dropped the flag would load "successfully" and quietly stop training the head."""
    off, on = _ver(), _ver(q_winprob_mode="read_only")
    with pytest.raises(ModelVersionError, match="q_winprob_mode"):
        on.check_compatible(off)
    with pytest.raises(ModelVersionError, match="q_winprob_mode"):
        off.check_compatible(on)
    on.check_compatible(_ver(q_winprob_mode="read_only"))          # matching: no raise


def test_a_pre_v107_config_migrates_to_OFF():
    """Defaulting rather than refusing is correct here for the v98 reason: a pre-v107 checkpoint
    COULD not have built the head, so 'none' is not a guess, it is the only possible past."""
    data = _migrate_config({"config_version": 106})
    assert data["q_winprob_mode"] == "none"
    assert data["q_winprob_coef"] == 0.0 and data["q_winprob_onpolicy_coef"] == 0.0
    assert data["config_version"] == MODEL_CONFIG_VERSION
    # A RECORDED value migrates untouched — the migration supplies an absent field, never
    # overwrites a present one.
    assert _migrate_config({"config_version": 106,
                            "q_winprob_mode": "read_only"})["q_winprob_mode"] == "read_only"
