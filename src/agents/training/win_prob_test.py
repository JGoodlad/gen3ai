"""Win-probability loss + the MC-label callback (the future-outcome plumbing)."""

from types import SimpleNamespace

import numpy as np
import torch

from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.win_prob_callback import WinProbLabelCallback


# ── _win_prob_loss ──────────────────────────────────────────────────────────────

def test_loss_masks_unknown_transitions():
    """A masked transition (in-progress episode) must be EXCLUDED — even one whose unmasked loss
    would be huge — so the head is never trained toward a fabricated label."""
    logits = torch.tensor([[2.0], [-2.0], [0.0], [5.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])      # 4th unknown
    loss, m = InstrumentedMaskablePPO._win_prob_loss(logits, target, mask)
    assert abs(m["coverage"] - 0.75) < 1e-6
    assert abs(m["pred_mean"] - 0.5) < 1e-6          # mean over the 3 known (excludes the 0.993 of #4)
    assert abs(m["label_mean"] - 2 / 3) < 1e-6
    # loss == mean BCE over the 3 known, the 4th must not appear
    import torch.nn.functional as F
    expect = F.binary_cross_entropy_with_logits(logits[:3].reshape(-1), target[:3].reshape(-1))
    assert abs(float(loss) - float(expect)) < 1e-6


def test_loss_none_guards():
    z = torch.zeros(3, 1)
    assert InstrumentedMaskablePPO._win_prob_loss(None, z, z) is None
    assert InstrumentedMaskablePPO._win_prob_loss(z, None, z) is None
    assert InstrumentedMaskablePPO._win_prob_loss(z, z, torch.zeros(3, 1)) is None  # all masked


def test_loss_grad_flows_to_logits():
    logits = torch.zeros(4, 1, requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones(4, 1)
    loss, _ = InstrumentedMaskablePPO._win_prob_loss(logits, target, mask)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


# ── WinProbLabelCallback MC-fill ─────────────────────────────────────────────────

def _make_cb(n_steps, n_envs, episode_starts):
    buf = SimpleNamespace(
        observations={
            "win_target": np.zeros((n_steps, n_envs, 1), np.float32),
            "win_mask": np.zeros((n_steps, n_envs, 1), np.float32),
        },
        episode_starts=np.asarray(episode_starts, np.float32),
        pos=0,
    )
    model = SimpleNamespace(n_steps=n_steps, n_envs=n_envs, _async_rollout=False, rollout_buffer=buf)
    cb = WinProbLabelCallback()
    cb.model = model
    return cb, model, buf


def test_mc_fill_propagates_outcome_and_masks_inprogress():
    # 1 env, 5 steps: episode A = steps 0,1,2 (WIN at step 2); episode B = steps 3,4 (in progress).
    es = np.array([[1.0], [0.0], [0.0], [1.0], [0.0]])     # starts at step 0 and step 3
    cb, model, buf = _make_cb(5, 1, es)
    cb._on_rollout_start()
    model._win_terminal_scratch[2, 0] = 1.0               # WIN terminal at step 2
    cb._on_rollout_end()
    np.testing.assert_array_equal(buf.observations["win_target"][:, 0, 0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(buf.observations["win_mask"][:, 0, 0], [1, 1, 1, 0, 0])


def test_mc_fill_two_complete_episodes():
    # episode A = 0,1 (LOSS at 1); episode B = 2,3,4 (WIN at 4). All known.
    es = np.array([[1.0], [0.0], [1.0], [0.0], [0.0]])
    cb, model, buf = _make_cb(5, 1, es)
    cb._on_rollout_start()
    model._win_terminal_scratch[1, 0] = 0.0               # loss
    model._win_terminal_scratch[4, 0] = 1.0               # win
    cb._on_rollout_end()
    np.testing.assert_array_equal(buf.observations["win_target"][:, 0, 0], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(buf.observations["win_mask"][:, 0, 0], [1, 1, 1, 1, 1])


def test_mc_fill_no_terminal_all_masked():
    es = np.array([[1.0], [0.0], [0.0]])                   # one in-progress episode, no done
    cb, model, buf = _make_cb(3, 1, es)
    cb._on_rollout_start()
    cb._on_rollout_end()
    np.testing.assert_array_equal(buf.observations["win_mask"][:, 0, 0], [0, 0, 0])


def test_sync_capture_writes_scratch_at_buffer_pos():
    cb, model, buf = _make_cb(4, 2, np.zeros((4, 2, 1)))
    cb._on_rollout_start()
    buf.pos = 2
    cb.locals = {"rollout_buffer": buf, "infos": [{"win_outcome": 1.0}, {}], "dones": [True, False]}
    cb._on_step()
    assert model._win_terminal_scratch[2, 0] == 1.0       # env0 done+outcome → captured at pos
    assert np.isnan(model._win_terminal_scratch[2, 1])    # env1 not done → untouched


def test_rlplayer_win_prob_reads_stashed_logit():
    """RLPlayer._win_prob (the trace-capture seam): sigmoid of the extractor's stashed
    last_win_prob_logits; None when the head is off (--win-prob-mode none)."""
    from agents.inference.player import RLPlayer
    fe = SimpleNamespace(last_win_prob_logits=torch.tensor([[2.0]]))
    fake = SimpleNamespace(model=SimpleNamespace(policy=SimpleNamespace(features_extractor=fe)))
    assert abs(RLPlayer._win_prob(fake) - float(torch.sigmoid(torch.tensor(2.0)))) < 1e-6
    fe.last_win_prob_logits = None
    assert RLPlayer._win_prob(fake) is None


def test_sync_capture_skipped_under_async():
    cb, model, buf = _make_cb(4, 1, np.zeros((4, 1, 1)))
    model._async_rollout = True
    cb._on_rollout_start()
    buf.pos = 1
    cb.locals = {"rollout_buffer": buf, "infos": [{"win_outcome": 1.0}], "dones": [True]}
    cb._on_step()
    assert np.isnan(model._win_terminal_scratch[1, 0])     # async path captures inline, not here
