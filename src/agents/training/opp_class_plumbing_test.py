"""The `opp_class` label's CONSUMER side: obs key -> one-ahead shift -> rollout buffer -> `train()`.

`opp_class` (`gen3_opp_class_v1`) shipped as a METRICS key — it is what splits every `opp_intent/*`
number by opponent kind. `--intent-label-bot-weight` (`gen3_intent_label_bot_weight_v1`) makes it
LOAD-BEARING: it decides how much each alpha/beta label is trained on, so a break in this chain
stops being a mislabelled dashboard and becomes wrong supervision.

The property that matters at each hop is the PAIRING, not the presence: row i's `opp_class` must
still be row i's opponent after the label shift and after `get()`'s shuffle. A key that survives
but decouples from its label would weight the wrong rows and look completely healthy.

(The PRODUCER hops — the Python wrapper tagging the episode, `Gen3Env` emitting the key — were
deleted with the Python env core, deletion pass U3. The Rust collector writes the key from the
declared class table; its column is pinned by `rust_rollout/store_test` and `trainee_spaces_test`.)
"""

import numpy as np
import torch

from agents.model.opp_intent import OPP_CLASS_BOT, OPP_CLASS_NAMES
from agents.training.opp_intent_labels import align_labels_to_predictions


# ── the two class tables must agree (they are hand-mirrored across packages) ───────────────

def test_the_model_side_class_table_mirrors_the_declared_one():
    """`opp_intent.OPP_CLASS_NAMES` is a hand copy kept in the model package so it does not import
    the training package. A drift here silently renames every stratified metric AND mis-targets
    the label weight."""
    from agents.training import opponent_classes as oc

    assert OPP_CLASS_NAMES == dict(oc.OPP_CLASS_NAMES)


def test_the_weighted_class_is_the_bot_class():
    """The label weight discounts exactly the class the collector calls a bot."""
    from agents.training import opponent_classes as oc

    assert OPP_CLASS_BOT == oc.OPP_CLASS_BOT


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
    from agents.training.rollout_buffer import RolloutBuffer

    n_steps, n_envs = 8, 3
    space = spaces.Dict({
        "observation": spaces.Box(low=-1e4, high=1e4, shape=(1,), dtype=np.float32),
        "opp_action_num": spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int64),
        "opp_class": spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64),
    })
    buf = RolloutBuffer(n_steps, space, spaces.Discrete(2), n_envs=n_envs)

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


def test_the_train_loop_passes_the_configured_weight_to_the_intent_fold():
    """Source-level pin of the ONE call site: the weight must be read off the model, not
    hardcoded. A silently-dropped kwarg is a flag that does nothing while looking wired."""
    import inspect

    # K8: the intent block is the static `intent_fold` inside region R1 (`micro_step`); its weight is a
    # static of the region, read off the model once per call by `TrainSetup._micro_static`.
    from agents.training.instrumented_ppo import micro_step as ms
    from agents.training.instrumented_ppo.train_setup import TrainSetup

    r1 = inspect.getsource(ms.micro_step)
    assert "intent_fold(fe, obs" in r1 and "bot_label_weight=st.bot_label_weight" in r1
    assert 'getattr(self, "intent_label_bot_weight", 1.0)' in inspect.getsource(TrainSetup._micro_static)


def test_only_the_intent_loss_takes_the_weight():
    """The BeliefBank rows (species / move / item / spread / nature-EV / HP-type) are TEAM truth —
    valid whoever pilots the team — so opponent class must never reach them. Pinned as a source
    fact because the failure would be a quiet loss of valid labels."""
    import inspect

    from agents.training import belief_bank
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    assert "bot_label_weight" not in inspect.getsource(belief_bank)
    assert "opp_class" not in inspect.getsource(belief_bank)
    from agents.training import belief_bank_static
    from agents.training.instrumented_ppo import micro_step as ms
    assert "bot_label_weight" not in inspect.getsource(belief_bank_static)
    assert "opp_class" not in inspect.getsource(belief_bank_static)
    assert "bot_label_weight" not in inspect.getsource(InstrumentedMaskablePPO.train)
    r1 = inspect.getsource(ms.micro_step)
    assert r1.count("bot_label_weight=st.bot_label_weight") == 1          # the intent fold's call
    assert r1.count("bot_label_weight") == 2                              # ... and nowhere else
