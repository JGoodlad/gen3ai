"""THE ROLLOUT-LEVEL SLICE N and the LEARNER-LEVEL CHECK (M5 Lane G's gate; ``parity.py`` states the
contract and the declared bars).

COMMIT (routine): 4 envs x 48 decisions x 2 windows on pool teams — the learner's buffer from the Rust
collector (window mode) equal to today's Python path on the same games (every observation key, label,
action, mask, reward, episode start and win label EXACT; values / log-probs / GAE within their float
bars), the replay's keyed draw reproducing every recorded action, and one production update on each
buffer landing on the same weights. Teeth: one Python label cell moved fails the slice; the comparator
refuses an exact field off by one bit and a float field past its bar.

MILESTONE (``slow``; verdict in ``designs/ops/slow_tier_status.json``): 8 envs x 128 x 3 windows on the
pool and on the LADDER corpus.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.training.rust_rollout import parity as PA

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def built():
    from agents.training.rust_rollout import testkit as TK

    TK.build_selfcheck()


def _assert_clean(out, *, rows, learner=True):
    assert not out["divergences"], out["divergences"]
    assert out["rows_compared"] == rows
    assert out["win_mask_rows"] > 0, "no game ended inside a window: the win-label path was not compared"
    assert out["episode_starts"] > out["n_envs"], "no env reached a second game"
    for k, bar in PA.FLOAT_FIELDS:
        assert out["max_abs"][k] <= bar, (k, out["max_abs"][k])
    assert all(v == 0 for v in out["lifecycle"].values()), out["lifecycle"]
    if learner:
        L = out["learner"]
        assert L["pass"], L
        assert L["losses_compared"] > 100 and L["param_moved_by_update"] > 1e-5, L
        assert L["worst_loss_over_allowance"] <= 1.0, L


def test_commit_tier_the_learners_buffer_equals_todays_python_path(built):
    out = PA.run(n_envs=4, n_steps=48, windows=2, learner=True)
    _assert_clean(out, rows=4 * 48 * 2)


def test_the_slice_has_teeth_on_a_label(built, monkeypatch):
    """A Python label that differs in ONE cell must fail the rollout slice (bytes, per key)."""
    from agents.training import gen3_env

    real = gen3_env.Gen3Env._item_labels

    def perturbed(self, *a):
        out = real(self, *a)
        out["item_label"] = out["item_label"].copy()
        out["item_label"].reshape(-1)[-1] += 1
        return out

    monkeypatch.setattr(gen3_env.Gen3Env, "_item_labels", perturbed)
    out = PA.run(n_envs=2, n_steps=16, windows=1, learner=False)
    assert any(k.endswith("obs:item_label") for k in out["divergences"]), out["divergences"]


def test_the_comparator_refuses_one_bit_and_a_float_past_its_bar():
    a = {"actions": np.zeros((4, 2, 1), np.int64), "values": np.zeros((4, 2), np.float32),
         "obs:observation": np.zeros((4, 2, 3), np.float32)}
    b = {k: v.copy() for k, v in a.items()}
    assert PA.compare([a], [b])["divergences"] == {}
    b["obs:observation"][1, 0, 2] = np.float32(1e-30)
    b["values"][0, 1] = PA.FLOAT_BAR * 2
    div = PA.compare([a], [b])["divergences"]
    assert "w0:obs:observation" in div and "w0:values" in div
    b2 = {k: v.copy() for k, v in a.items()}
    b2["values"][0, 1] = PA.FLOAT_BAR / 2
    assert PA.compare([a], [b2])["divergences"] == {}


@pytest.mark.slow
@pytest.mark.parametrize("teams", ["pool", "ladder"])
def test_milestone_the_learners_buffer_equals_todays_python_path(built, teams):
    out = PA.run(n_envs=8, n_steps=128, windows=3, learner=True, teams=teams, team_offset=11)
    _assert_clean(out, rows=8 * 128 * 3)


def test_one_near_boundary_threshold_everywhere():
    from agents.training import keyed_draw as KD
    from agents.training.rust_env_opponents_parity import NEAR_TIE_FACTOR

    assert PA.NEAR == NEAR_TIE_FACTOR * PA.FLOAT_BAR == KD.NEAR_MARGIN


def test_a_trainee_flip_is_recorded_with_both_margins_aligned_and_judged_by_the_margin_rule():
    """The replay's forward: a draw that differs from the recorded action is a FLIP — recorded with both
    sides' margins, the RECORDED action played on — and ``judge_flips`` at the gate's bar makes a flip
    whose larger margin is >= 2 x FLOAT_BAR fatal and one below it a counted tie."""
    from types import SimpleNamespace

    import torch as th

    from agents.training import keyed_draw as KD
    from agents.training.rust_env_opponents_parity import judge_flips

    logits = th.tensor([[0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]] * 2)

    class _Dist:
        distribution = SimpleNamespace(logits=logits)

        def apply_masking(self, m):
            pass

        def log_prob(self, a):
            return th.zeros(a.shape[0])

    ident = SimpleNamespace(forward_actor=lambda x: x, forward_critic=lambda x: x)
    pol = SimpleNamespace(extract_features=lambda o: (o, o), mlp_extractor=ident,
                          _critic_value=lambda x: th.zeros(x.shape[0]), _get_action_dist_from_latent=lambda _l: _Dist(),
                          action_space=SimpleNamespace(shape=()))
    venv = SimpleNamespace(envs=[SimpleNamespace(env=SimpleNamespace(episode=0, decision=0)) for _ in range(2)])
    mask = np.ones((2, 11), dtype=bool)
    lp = th.log_softmax(logits, -1).numpy()
    u = KD.keyed_uniforms(17, KD.STREAM_TRAINEE, np.arange(2), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64))
    drawn, _m = KD.keyed_actions(lp, u)
    other = [(int(a) + 1) % 3 for a in drawn]
    scripts = [PA.EnvScript(p1=[[(int(drawn[0]), 0.3)]]), PA.EnvScript(p1=[[(other[1], 0.5)]])]
    kf = PA._KeyedForward(pol, venv, 17, scripts)
    actions, _v, _lp = kf(th.zeros(2, 4), action_masks=mask)
    assert actions.tolist() == [int(drawn[0]), other[1]]          # env 1 plays the RECORDED action on
    assert len(kf.flips) == 1 and kf.flips[0]["recorded"] == other[1] and kf.flips[0]["margin"] >= 0.5
    ties, fatal = judge_flips({"flips": kf.flips}, PA.FLOAT_BAR)
    assert (ties, len(fatal)) == (0, 1)
    tie = dict(kf.flips[0], margin=0.9 * PA.NEAR)
    assert judge_flips({"flips": [tie]}, PA.FLOAT_BAR)[0] == 1
