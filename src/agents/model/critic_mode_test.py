"""The WIN-PROB CRITIC MODE — `gen3_winprob_critic_mode_v1`.

Three halves, and each test names the failure it guards against rather than restating the code:

1. **the ROUTE** — `Gen3DualHeadMaskablePolicy._critic_value` reads the win-prob head, in [0,1],
   with NO fallback and no PopArt, on a REAL `MaskablePPO`-built policy. Built that way for
   `gen3_identity_init_guard_v1`'s reason: an invariant asserted only on a directly-constructed
   module is an invariant about a construction path training does not use.
2. **the VERSION GATE** — `critic` is recorded, string-compared on resume, and a pre-v109 config
   migrates to `shaped`. The gate matters more than usual here: BOTH routes return a `[B,1]`
   float tensor, so a flipped mode produces no shape error anywhere and `check_compatible` is the
   only thing standing between a resume and a run that predicts a different quantity in silence.
3. **the REWARD** — the win-INDICATOR terminal, and the re-armable anti-stall tilt (design B4).

The OFF-path byte-identity claims live with the surfaces they are about:
`src/main/critic_mode_config_test.py` (the flags and the refusals) and
`src/agents/training/instrumented_ppo_winprob_critic_test.py` (the PPO update).
"""
from __future__ import annotations

import inspect
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.critic_mode import (
    CRITIC_DEFAULT, CRITIC_MODES, CRITIC_SHAPED, CRITIC_WINPROB, is_winprob)
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


# --------------------------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------------------------

def test_the_default_is_still_shaped():
    """The DEFAULT FLIP is a separate commit; this one only makes the mode EXIST.

    Flipping it here would silently change every flagless run's critic, its reward composition and
    its `check_compatible` verdict — without the `ARCH_SIGNATURE` bump that forces the fresh
    weights a probability critic cannot be warm-started into."""
    assert CRITIC_DEFAULT == CRITIC_SHAPED == "shaped"
    assert CRITIC_MODES == ("shaped", "winprob")


@pytest.mark.parametrize("value,expected", [
    ("winprob", True), (CRITIC_WINPROB, True),
    ("shaped", False), (None, False), ("", False), ("WINPROB", False),
])
def test_is_winprob_is_the_one_predicate(value, expected):
    """Every site reads the mode through this, never a bare `== "winprob"`, so a
    `getattr(obj, "critic", "shaped")` read answers identically everywhere."""
    assert is_winprob(value) is expected


def test_critic_mode_imports_no_torch():
    """`main.checkargs` promises not to import torch and reads the legal set from this module."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, agents.model.critic_mode; print('torch' in sys.modules)"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "importing critic_mode pulled torch in"


# --------------------------------------------------------------------------------------------
# the ROUTE, through the path training actually builds
# --------------------------------------------------------------------------------------------

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


def _real_policy(critic="shaped", win_prob_mode="shaping", seed=0, **policy_kw):
    """MaskablePPO -> ActorCriticPolicy._build() — the construction training uses."""
    enc = Gen3ObservationEncoder(load_mappings())
    ek = enc.get_features_extractor_kwargs()
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {**ek, **{k: v for k, v in {"win_prob_mode": win_prob_mode}.items() if k in sig}}
    torch.manual_seed(seed)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64]),
                       "critic": critic, **policy_kw},
    )
    return model.policy, enc


def _obs(enc, batch=4):
    return {"observation": torch.zeros(batch, enc.dimension, dtype=torch.float32)}


def test_the_winprob_critic_IS_the_heads_sigmoid_and_lives_in_the_unit_interval():
    """The identity every downstream consumer assumes — GAE, the search leaf, every calibration
    instrument. Asserted against the head's own stash rather than a range check alone, because a
    small net's `value_net` could land inside [0,1] by luck."""
    p, enc = _real_policy(critic="winprob")
    with torch.no_grad():
        v = p.predict_values(_obs(enc))
    assert v.shape == (4, 1), "the winprob route must keep value_net's [B,1] shape"
    assert bool(((v >= 0.0) & (v <= 1.0)).all()), f"V left [0,1]: {v.reshape(-1).tolist()}"
    logits = p.features_extractor.last_win_prob_logits
    assert torch.equal(v, torch.sigmoid(logits.reshape(-1, 1)))


def test_the_shaped_critic_is_value_net_even_when_the_win_prob_head_exists():
    """The CONTROL for the test above. Without it, a build where BOTH modes routed to the head
    would pass the winprob assertions and nothing would notice."""
    p, enc = _real_policy(critic="shaped", seed=3)
    obs = _obs(enc, batch=3)
    with torch.no_grad():
        v = p.predict_values(obs)
        direct = p.value_net(p.mlp_extractor.forward_critic(p.extract_features(obs)[1]))
    assert torch.equal(v, direct)


def test_winprob_has_NO_fallback_when_the_head_is_absent():
    """The v89 orphaned-route class: `value_net` is in no loss graph under this critic, so falling
    back to it would be a critic the training loop believes in and nothing updates."""
    p, enc = _real_policy(critic="winprob")
    obs = _obs(enc, batch=2)
    p.predict_values(obs)                      # warm the stash so the failure is the head, not it
    p.features_extractor.win_head = None
    with pytest.raises(RuntimeError, match="critic='winprob'"):
        p.predict_values(obs)


def test_winprob_refuses_popart_at_construction():
    """The launch path refuses it (`combination_checks`); this is the last line of defence for a
    policy built directly, where `_denorm` would take V straight out of [0,1]."""
    with pytest.raises(ValueError, match="use_popart"):
        _real_policy(critic="winprob", use_popart=True)


def test_an_unknown_critic_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown critic"):
        _real_policy(critic="winprb")


def test_every_value_site_routes_through_critic_value():
    """`forward`, `evaluate_actions` and `predict_values` must agree, or PPO's epoch recompute
    would disagree with the rollout it is scoring."""
    p, enc = _real_policy(critic="winprob")
    obs = _obs(enc, batch=2)
    masks = np.ones((2, ACTION_SPACE_SIZE), dtype=bool)
    with torch.no_grad():
        actions, v_fwd, _ = p.forward(obs, action_masks=masks)
        v_pred = p.predict_values(obs)
        v_eval, _, _ = p.evaluate_actions(obs, actions, action_masks=torch.as_tensor(masks))
    assert torch.equal(v_fwd, v_pred)
    assert torch.equal(v_eval.reshape(-1), v_pred.reshape(-1))


# --------------------------------------------------------------------------------------------
# the VERSION GATE
# --------------------------------------------------------------------------------------------

def _version(critic: str):
    from agents.model.snapshot import current_model_version
    return current_model_version(load_mappings(), critic=critic)


def test_the_critic_is_recorded_and_gated():
    from agents.model.model_version import ModelVersionError
    saved, live = _version("shaped"), _version("winprob")
    assert saved.critic == "shaped" and live.critic == "winprob"
    import json
    assert json.loads(saved.to_json())["critic"] == "shaped", \
        "the mode must reach model_config.json"
    with pytest.raises(ModelVersionError, match="critic mismatch"):
        live.check_compatible(saved)
    saved.check_compatible(saved)


def test_a_pre_v109_config_migrates_to_todays_behaviour():
    """Not a guess: none of the three fields existed before v109, so these are the only possible
    past. The REFUSAL direction belongs to check_compatible / check_reward_config."""
    from agents.model.model_version import MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config
    out = _migrate_config({"config_version": 108})
    assert out["critic"] == "shaped"
    assert out["terminal_indicator"] is False
    assert out["no_progress_tax_armed"] is False
    assert out["config_version"] == 109 == MODEL_CONFIG_VERSION


def test_a_recorded_critic_survives_the_migration():
    from agents.model.model_version.migrations import _migrate_config
    assert _migrate_config({"config_version": 108, "critic": "winprob"})["critic"] == "winprob"


def test_no_arch_signature_bump_at_v109():
    """`shaped` is the default and builds no module and moves no state_dict key, so v109 must NOT
    bump the signature — the bump belongs to the DEFAULT FLIP, where it forces fresh weights."""
    from agents.model.model_version import ARCH_SIGNATURE
    assert ARCH_SIGNATURE == "gen3_critic_route_wave_v1", (
        "v109 is a byte-identical-when-off mode; if the signature moved, it moved for another "
        "reason and this pin should be updated deliberately alongside it.")


# --------------------------------------------------------------------------------------------
# the REWARD — the win INDICATOR, and the re-armed tilt
# --------------------------------------------------------------------------------------------

def test_the_indicator_terminal_pays_zero_on_every_non_win():
    """Loss, pre-cap tie and 250-turn timeout alike, because the critic is sigmoid(logit) in
    [0,1] and V(s) == E[return] only holds when the return is `victory_value * 1{win}`."""
    from agents.training.reward_manager import RewardConfig
    from agents.training.reward_terminal_test_support import terminal_reward
    cfg = RewardConfig(terminal_indicator=True, victory_value=1.0, hand_shaping=False)
    assert terminal_reward(cfg, "win") == pytest.approx(1.0)
    assert terminal_reward(cfg, "loss") == pytest.approx(0.0)
    assert terminal_reward(cfg, "tie") == pytest.approx(0.0)
    assert terminal_reward(cfg, "timeout") == pytest.approx(0.0)


def test_the_default_terminal_is_unchanged():
    """OFF byte-identity for the reward: the historical ±30 with the −35 stall ordering."""
    from agents.training.reward_manager import RewardConfig
    from agents.training.reward_terminal_test_support import terminal_reward
    cfg = RewardConfig()
    assert cfg.terminal_indicator is False and cfg.no_progress_tax_armed is False
    assert terminal_reward(cfg, "win") == pytest.approx(30.0)
    assert terminal_reward(cfg, "loss") == pytest.approx(-30.0)
    assert terminal_reward(cfg, "tie") == pytest.approx(-30.0)
    assert terminal_reward(cfg, "timeout") == pytest.approx(-35.0)


def test_arm_no_progress_tax_re_arms_ONLY_that_term():
    """Design gap B4. `--no-hand-shaping` zeroes the WHOLE BIAS class; this re-arms the anti-stall
    tilt without reviving the other 24, which is the entire reason it is its own flag."""
    from agents.training.reward_manager import RewardConfig, reward_class_composition
    plain = reward_class_composition(RewardConfig(hand_shaping=False))
    armed = reward_class_composition(RewardConfig(hand_shaping=False, no_progress_tax_armed=True))
    assert plain["bias_terms"] == []
    assert armed["bias_terms"] == ["no_progress_tax"]
    assert armed["pbrs_terms"] == plain["pbrs_terms"] == []
    assert armed["terminal"] == plain["terminal"] == 1


def test_arming_is_a_no_op_with_hand_shaping_ON():
    """The term is already reachable there, so the flag must change nothing — otherwise it is a
    second, hidden gate on a term that already has one."""
    from agents.training.reward_manager import RewardConfig, reward_class_composition
    assert (reward_class_composition(RewardConfig())
            == reward_class_composition(RewardConfig(no_progress_tax_armed=True)))


def test_the_two_reward_fields_are_resume_immutable_and_name_a_real_flag():
    from agents.model.model_version.constants import _REWARD_FIELD_FLAGS, _REWARD_IMMUTABLE_FIELDS
    for name, flag in (("terminal_indicator", "--terminal-indicator"),
                       ("no_progress_tax_armed", "--arm-no-progress-tax")):
        assert name in _REWARD_IMMUTABLE_FIELDS, f"{name} is not value-checked on resume"
        assert _REWARD_IMMUTABLE_FIELDS[name] is False, "the default must be today's behaviour"
        assert _REWARD_FIELD_FLAGS[name] == flag, "the resume message must name a real flag"


def test_the_reward_defaults_track_the_dataclass():
    """The `reward_defaults_test` rule: a divergence would make an ABSENT field mean one thing to
    the reward and another to the version record."""
    from agents.model.model_version.constants import _REWARD_IMMUTABLE_FIELDS
    from agents.training.reward_manager import RewardConfig
    rc = RewardConfig()
    for name in ("terminal_indicator", "no_progress_tax_armed"):
        assert getattr(rc, name) == _REWARD_IMMUTABLE_FIELDS[name]


def test_the_gamma_default_is_the_pbrs_constant_not_a_retyped_number():
    """A second copy of 0.9999 is a second place for the PBRS invariance premise to break."""
    import main.train.config as cfg
    from agents.training.reward_weights import PBRS_GAMMA
    assert '_resolve("gamma", _PBRS_GAMMA_DEFAULT)' in open(cfg.__file__).read()
    assert PBRS_GAMMA == 0.9999


# --------------------------------------------------------------------------------------------
# the DRAW BRANCH (design §3.2 / gap B9) and the `popart is None` paths (gap B2)
# --------------------------------------------------------------------------------------------

class _Battle:
    def __init__(self, won):
        self.won = won


class _EnvStub:
    def __init__(self, won):
        self.battle1 = _Battle(won)


def _win_outcome(won):
    """The wrapper's terminal label branch, exercised on its own inputs.

    `MaskableAgentWrapper.step` is a long method over a real env, so the branch is reproduced here
    against the ONE thing it reads — `battle1.won`, a TRI-STATE. Kept in step with the source by
    `test_the_draw_branch_matches_the_wrapper_source` below, which reads the real code."""
    b = _EnvStub(won).battle1
    outcome = getattr(b, "won", None)
    if outcome is True:
        return 1.0, False
    if outcome is False:
        return 0.0, False
    return 0.0, True


@pytest.mark.parametrize("won,expect_y,expect_draw", [
    (True, 1.0, False), (False, 0.0, False), (None, 0.0, True),
])
def test_a_draw_is_SCORED_as_a_not_win_and_COUNTED(won, expect_y, expect_draw):
    """Design §3.2, option (a), made an explicit branch instead of a boolean fall-through.

    The load-bearing half is the third row: `won is None` — a draw or the 250-turn timeout — is
    scored `y = 0` and **flagged**, never masked out. Masking would leave that episode's ~250
    decisions with no learning signal at all, and they are the decisions that most need one."""
    y, is_draw = _win_outcome(won)
    assert y == expect_y
    assert is_draw is expect_draw


def test_the_draw_branch_matches_the_wrapper_source():
    """The stub above is only worth having if it still describes the real code: pin the three-way
    branch and the `win_draw` publication in `MaskableAgentWrapper.step`."""
    import agents.training.wrappers as w
    src = open(w.__file__).read()
    assert 'if _outcome is True:' in src and 'elif _outcome is False:' in src
    assert 'info["win_draw"] = float(is_draw)' in src
    assert 'won = 1.0 if (b is not None and b.won is True) else 0.0' not in src, (
        "the boolean fall-through is back — a draw would again be a not-win BY ACCIDENT rather "
        "than by decision, and nothing would count it")


def test_the_draw_rate_is_published_from_the_terminal_scan():
    """A decision that is not stated is a decision nobody can audit. `signal/draw_rate` is what
    makes the §3.2 choice's FREQUENCY visible — and it is a PRIMARY endpoint on a `winprob` arm,
    where a [0,1] critic has given up the terminal's anti-stall ordering."""
    import agents.training.signal_callback as sc
    src = open(sc.__file__).read()
    assert 'self.logger.record("signal/draw_rate"' in src
    assert 'info.get("win_draw", 0.0)' in src, (
        "the rate must read the wrapper's published flag, not re-derive a draw from win_outcome — "
        "a loss and a draw are the SAME win_outcome by construction")


@pytest.mark.parametrize("site,needle", [
    # gap B2: every place that divides by / normalizes through PopArt must already have a
    # `popart is None` branch, because `--critic winprob` REFUSES PopArt — so on that arm all
    # three run their None path on every call, permanently, rather than as an edge case.
    ("agents.training.instrumented_ppo.aux_terms",
     'float(popart.sigma) if popart is not None else 1.0'),
    ("agents.training.cf_terms",
     'popart.normalize(b.mc_return) if popart is not None else b.mc_return'),
    ("agents.training.cf_terms",
     'popart.denormalize(pred) if popart is not None else pred'),
])
def test_every_popart_path_has_a_None_branch(site, needle):
    """The AUDIT half of gap B2, as an executable pin rather than a note.

    `td_aux`'s `/ popart.sigma`, `cf_shadow`'s normalize and its de-normalized readback are the
    three sites the design names. All three already branch — sigma = 1.0 and the identity map,
    which is exactly right when the target is already a probability — and the point of pinning it
    is that under `--critic winprob` these stop being the rare path and become the ONLY path."""
    import importlib
    mod = importlib.import_module(site)
    assert needle in open(mod.__file__).read(), f"{site} lost its `popart is None` branch"


def test_the_value_loss_from_se_needs_no_popart_branch():
    """The third site the design lists, and the honest answer is that it never had one to lose:
    `_value_loss_from_se` takes the per-sample squared errors ALREADY in whatever space the
    caller chose, so PopArt is the CALLER's business (`ppo.py`'s three-way branch). Recorded so
    the B2 audit reads as complete rather than as two-of-three."""
    import inspect
    from agents.training.instrumented_ppo.value_terms import ValueTerms
    src = inspect.getsource(ValueTerms._value_loss_from_se)
    assert "popart" not in src
