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


def test_loss_margin_stratifies_and_scores_skill():
    """With the material margin, the loss adds closeness-stratified Brier/acc + a skill-vs-material
    score — the 'value on close games, beyond counting mons' signal the aggregate Brier hides."""
    # 2 blowouts (|margin| 0.9, head trivially right) + 2 close (|margin| 0.05, head still right).
    logits = torch.tensor([[4.0], [-4.0], [1.0], [-1.0]])   # P ≈ .98, .02, .73, .27
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones(4, 1)
    margin = torch.tensor([[0.9], [-0.9], [0.05], [-0.05]])
    _, m = InstrumentedMaskablePPO._win_prob_loss(logits, target, mask, margin)
    assert "brier_contested" in m and "skill_vs_material" in m
    assert abs(m["contested_frac"] - 0.5) < 1e-6          # 2 of 4 are |margin|<0.25
    assert abs(m["acc_contested"] - 1.0) < 1e-6           # both close calls predicted correctly
    assert m["brier_contested"] < 0.25                    # below a 50/50 game's no-skill floor → real skill
    assert m["skill_vs_material"] > 0.0                   # beats the material-only baseline
    assert abs(m["contested_label_mean"] - 0.5) < 1e-6    # the close band is genuinely even


def test_loss_no_margin_omits_stratified():
    """Without the margin (old config / margin absent), only the aggregate metrics are reported."""
    logits = torch.zeros(2, 1)
    _, m = InstrumentedMaskablePPO._win_prob_loss(logits, torch.tensor([[1.0], [0.0]]), torch.ones(2, 1), None)
    assert "brier" in m and "brier_contested" not in m and "skill_vs_material" not in m


def test_a_FLAT_margin_is_treated_as_absent():
    """`gen3_tb_relevance_v1`: a margin with no SPREAD cannot stratify, so the six tags it would
    produce are copies of their pooled siblings plus two constants — worse than publishing
    nothing, because all six READ AS MEASUREMENTS. This is the exact shape the pre-fix win-prob
    arm shipped (`win_margin` pinned at 0.0 by `_fold_material_pbrs`'s early return)."""
    logits = torch.tensor([[2.0], [-2.0], [1.0], [-1.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    _, m = InstrumentedMaskablePPO._win_prob_loss(
        logits, target, torch.ones(4, 1), torch.zeros(4, 1))
    assert "contested_frac" not in m and "skill_vs_material" not in m


def test_a_REAL_margin_spread_selects_a_STRICT_SUBSET():
    """`gen3_obs_margin_unconditional_v1` — the consumer half of the fix, driven by margins the
    REAL reward manager publishes under the win-prob composition rather than by hand-typed floats.

    Pre-fix, every one of these read 0.0, so `|margin| < tau` was always true: `contested_frac`
    was a flat 1.0, every `*_contested` tag was a byte-identical copy of its pooled sibling, and
    `P_mat` was a constant 0.5 — `skill_vs_material` scored the head against a coin flip."""
    from agents.training.progress_clock import ProgressClock
    from agents.training.reward_manager import Gen3RewardManager, RewardConfig
    from agents.training.reward_test_fakes import _Battle, _delta, _full_team_live

    winprob = RewardConfig(hand_shaping=False, terminal_indicator=True,
                           victory_value=1.0, draw_penalty=0.0)
    margins = []
    for ours, opp in ((6, 1), (6, 2), (4, 4), (3, 3), (2, 6), (1, 6)):
        mgr = Gen3RewardManager(config=winprob, progress_clock=ProgressClock())
        mgr.process_turn_reward(_Battle(_full_team_live(our_alive=ours, opp_alive=opp), turn=5),
                                _delta())
        margins.append([mgr._last_material_margin])

    margin = torch.tensor(margins)
    n = margin.shape[0]
    # A board-derived spread, not a hand-built one — the fix's whole point.
    assert float(margin.max() - margin.min()) > 0.0, "the reward manager published a flat margin"

    logits = torch.zeros(n, 1)
    target = (margin > 0).float()
    _, m = InstrumentedMaskablePPO._win_prob_loss(logits, target, torch.ones(n, 1), margin)

    assert "contested_frac" in m, "the contested family is missing on a real spread"
    assert 0.0 < m["contested_frac"] < 1.0, (
        f"the contested split selected {m['contested_frac']:.3f} of the batch — a STRICT SUBSET "
        "is what makes it a split rather than a copy of the pooled metrics")
    # The material baseline is no longer the degenerate constant 0.5, so the skill score is a
    # comparison against 'count the mons' rather than against a coin flip.
    assert abs(m["brier_material"] - 0.25) > 1e-6, "P_mat collapsed to the constant 0.5"


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
