"""The `opp_class` label's route: wrapper -> env -> obs key -> rollout buffer -> `train()`.

`opp_class` (`gen3_opp_class_v1`) shipped as a METRICS key — it is what splits every `opp_intent/*`
number by opponent kind. `--intent-label-bot-weight` (`gen3_intent_label_bot_weight_v1`) makes it
LOAD-BEARING: it now decides how much each alpha/beta label is trained on, so a break in this chain
stops being a mislabelled dashboard and becomes wrong supervision. Nothing covered the chain, so
these are the gates it never had.

The property that matters at each hop is the PAIRING, not the presence: row i's `opp_class` must
still be row i's opponent after the label shift and after `get()`'s shuffle. A key that survives
but decouples from its label would weight the wrong rows and look completely healthy.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from agents.model.opp_intent import OPP_CLASS_BOT, OPP_CLASS_NAMES
from agents.training.opp_intent_labels import align_labels_to_predictions
from agents.training.wrappers import MaskableAgentWrapper


# ── the two class tables must agree (they are hand-mirrored across packages) ───────────────

def test_the_model_side_class_table_mirrors_the_wrapper():
    """`opp_intent.OPP_CLASS_NAMES` is a hand copy kept in the model package so it does not import
    the training package. A drift here silently renames every stratified metric AND mis-targets
    the label weight."""
    assert OPP_CLASS_NAMES == {
        MaskableAgentWrapper.OPP_CLASS_BOT: "bot",
        MaskableAgentWrapper.OPP_CLASS_POOL: "pool",
        MaskableAgentWrapper.OPP_CLASS_STABLE: "stable",
        MaskableAgentWrapper.OPP_CLASS_EXPLOITER: "exploiter",
    }


def test_the_weighted_class_is_the_bot_class():
    """The label weight discounts exactly the class the wrapper calls a bot."""
    assert OPP_CLASS_BOT == MaskableAgentWrapper.OPP_CLASS_BOT


# ── hop 1: the wrapper tags the episode and pushes it onto the env ─────────────────────────

def _stub_env():
    env = MagicMock()
    env.agent1.username = "a1"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    return env


def _wrapper(**kw):
    env = _stub_env()
    return MaskableAgentWrapper(env, heuristic_opponents=[MagicMock()], **kw), env


def test_a_bot_episode_tags_the_bot_class():
    w, _ = _wrapper(self_play_fraction=0.0)
    w._select_episode_opponent()
    assert w._opponent_class == MaskableAgentWrapper.OPP_CLASS_BOT


def test_a_pool_episode_tags_the_pool_class():
    pool = MagicMock()
    pool.is_empty.return_value = False
    pool.sample.return_value = MagicMock()
    pool.load_model.return_value = "M"
    w, _ = _wrapper(self_play_fraction=1.0, pool=pool, pool_player=MagicMock())
    w._select_episode_opponent()
    assert w._opponent_class == MaskableAgentWrapper.OPP_CLASS_POOL


def test_an_exploiter_episode_tags_the_exploiter_class():
    w, _ = _wrapper(exploiter_player=MagicMock())
    w._select_episode_opponent()
    assert w._opponent_class == MaskableAgentWrapper.OPP_CLASS_EXPLOITER


def test_reset_pushes_the_class_down_to_the_env_that_owns_the_obs():
    """The env builds the obs, so the tag has to reach it — and at RESET, so a label can never
    describe the PREVIOUS opponent."""
    from unittest.mock import patch

    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    w, env = _wrapper(exploiter_player=MagicMock())
    # Stub only the SUPER call (it needs a live battle); the push under test is the wrapper's own.
    with patch.object(SingleAgentWrapper, "reset", return_value=({}, {})):
        w.reset()
    assert env._opponent_class == MaskableAgentWrapper.OPP_CLASS_EXPLOITER


# ── hop 2: the env emits it beside the intent labels ───────────────────────────────────────

class _EnvStub:
    """The minimum `Gen3Env._opp_intent_labels` reads. `_intent_delta=None` takes the
    zero-label branch, which is the case that matters here: even a masked label must carry an
    honest class, since the class is what the WEIGHT keys on."""

    _intent_delta = None
    _opp_slot_map_prev: dict = {}
    _species_num: dict = {}


def _labels(cls_code=None):
    from agents.training.gen3_env import Gen3Env
    stub = _EnvStub()
    if cls_code is not None:
        stub._opponent_class = cls_code
    return Gen3Env._opp_intent_labels(stub)


@pytest.mark.parametrize("code", sorted(OPP_CLASS_NAMES))
def test_the_env_emits_the_class_the_wrapper_set(code):
    out = _labels(code)
    assert out["opp_class"].dtype == np.int64
    assert out["opp_class"].shape == (1,)
    assert int(out["opp_class"][0]) == code


def test_an_unwrapped_env_reports_bot_rather_than_crashing():
    """A bare env has no opponent rotation. 0 is the honest default — and it is the WEIGHTED
    class, so the fallback is conservative (train less), never optimistic."""
    assert int(_labels()["opp_class"][0]) == MaskableAgentWrapper.OPP_CLASS_BOT


def test_the_class_rides_the_same_dict_as_the_intent_labels():
    """One dict, one emission point — so the class cannot describe a different decision than the
    label it qualifies."""
    out = _labels(1)
    assert {"opp_action_kind", "opp_action_num", "opp_switch_slot",
            "opp_switch_species", "opp_class"} <= set(out)


# ── hop 3: the one-ahead shift moves it with everything else ───────────────────────────────

def test_the_shift_carries_opp_class_with_the_label_it_qualifies():
    """`train()` shifts every intent key back one row before `get()`. `opp_class` is CONSTANT
    within an episode so the shift is a semantic no-op — but it must still be applied, or a
    reader has to remember which keys were shifted and which were not (the asymmetry that
    produced the `opp_switch_species` bug). Here the classes DIFFER per row, which is what makes
    the assertion able to fail."""
    col = np.array([[0], [1], [2], [3]], dtype=np.int64).reshape(4, 1, 1)   # [n_steps, n_envs, 1]
    starts = np.zeros((4, 1), dtype=np.float32)
    shifted = align_labels_to_predictions(col, starts, 0)
    assert shifted[:, 0, 0].tolist() == [1, 2, 3, 0]      # last row has no successor -> fill


def test_the_shift_drops_a_pair_that_spans_an_episode_boundary():
    """The fill value for `opp_class` is 0 (=bot). A dropped pair's alpha label is
    KIND_UNKNOWN, so the row is MASKED and never scored — the class it carries is irrelevant by
    construction, and this pins that it is the pair that is dropped, not just the label."""
    col = np.array([[1], [1], [1], [1]], dtype=np.int64).reshape(4, 1, 1)
    starts = np.zeros((4, 1), dtype=np.float32)
    starts[2, 0] = 1.0                                    # row 2 begins a new episode
    shifted = align_labels_to_predictions(col, starts, 0)
    assert shifted[1, 0, 0] == 0                          # row 1's successor is another battle


# ── hop 4: the rollout buffer keeps it paired through the shuffle ──────────────────────────

def test_the_buffer_shuffle_keeps_opp_class_paired_with_its_label():
    """THE plumbing property. `get()` returns a random permutation; `opp_class` and the alpha
    label are separate arrays, so nothing but riding the SAME obs dict keeps them together. If
    they ever decoupled, the weight would land on the wrong rows and every metric would still
    read normally."""
    from gymnasium import spaces
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

    n_steps, n_envs = 8, 3
    space = spaces.Dict({
        "observation": spaces.Box(low=-1e4, high=1e4, shape=(1,), dtype=np.float32),
        "opp_action_num": spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int64),
        "opp_class": spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64),
    })
    buf = MaskableDictRolloutBuffer(n_steps, space, spaces.Discrete(2), n_envs=n_envs)

    rng = np.random.default_rng(0)
    for t in range(n_steps):
        cls = rng.integers(0, 4, size=n_envs)
        obs = {
            "observation": np.zeros((n_envs, 1), dtype=np.float32),
            # The INVARIANT under test, encoded in the data: num == 100 * class.
            "opp_action_num": (100 * cls).reshape(n_envs, 1).astype(np.int64),
            "opp_class": cls.reshape(n_envs, 1).astype(np.int64),
        }
        buf.add(obs, np.zeros((n_envs,), dtype=np.int64), np.zeros(n_envs, dtype=np.float32),
                np.zeros(n_envs, dtype=np.float32), torch.zeros(n_envs), torch.zeros(n_envs),
                action_masks=np.ones((n_envs, 2), dtype=np.int8))

    seen = 0
    for batch in buf.get(batch_size=5):
        num = batch.observations["opp_action_num"].reshape(-1).long()
        cls = batch.observations["opp_class"].reshape(-1).long()
        assert torch.equal(num, 100 * cls)
        seen += len(cls)
    assert seen == n_steps * n_envs


# ── hop 5: the PPO loop hands the configured weight to the loss ────────────────────────────

def test_the_trainer_defaults_the_weight_to_one():
    """A model built without the flag must present 1.0, since `train()` reads the attribute."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    assert InstrumentedMaskablePPO.intent_label_bot_weight == 1.0


def test_the_train_loop_passes_the_configured_weight_to_intent_losses():
    """Source-level pin of the ONE call site: the weight must be read off the model, not
    hardcoded. A silently-dropped kwarg is a flag that does nothing while looking wired."""
    import inspect

    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert "intent_losses(" in src
    assert 'bot_label_weight=float(\n' in src or "bot_label_weight=" in src
    assert 'getattr(self, "intent_label_bot_weight", 1.0)' in src


def test_only_the_intent_loss_takes_the_weight():
    """The BeliefBank rows (species / move / item / spread / nature-EV / HP-type) are TEAM truth —
    valid whoever pilots the team — so opponent class must never reach them. Pinned as a source
    fact because the failure would be a quiet loss of valid labels."""
    import inspect

    from agents.training import belief_bank
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    assert "bot_label_weight" not in inspect.getsource(belief_bank)
    assert "opp_class" not in inspect.getsource(belief_bank)
    assert inspect.getsource(InstrumentedMaskablePPO.train).count("bot_label_weight") == 1
