"""`gen3_winprob_pbrs_v1` — the win-prob PBRS reward shaping (ai_v12 route 1).

What each group pins, and why it is the thing worth pinning:

* **TELESCOPING** — the invariance theorem's content, as arithmetic. Over a COMPLETE episode the
  γ-discounted shaping sum must equal exactly ``−coef·φ(s_0)``: a constant per start state, which is
  why a miscalibrated φ cannot change the optimal policy. If the terminal convention is implemented
  wrong, this identity breaks and nothing else in the system notices — the run just trains on a
  reward stream that is no longer policy-invariant.
* **TRUNCATION** — the classic PBRS bug in this family: forcing φ(s′)=0 at a *buffer boundary*
  charges the policy a large phantom penalty for the rollout ending. A separate test, because the
  two cases look identical in the array and differ only in `dones`.
* **DETACHMENT** — a revert-catcher. The φ read must run under `no_grad` and reach numpy detached;
  the test asserts grad is DISABLED inside the forward, so deleting the `no_grad` fails it.
* **OFF byte-identity** — at coef 0 nothing is touched and the module is not even imported.
* **PopArt / GAE order** — the shaping must land in RAW reward space, before the returns PopArt
  reads at the top of ``train()``.
* **The config gates** — for a training-only coefficient the `parser.error` is the ONLY gate there
  is, and the "no head to read" case is the invisible-regression class.

Run:
    python -m pytest src/agents/training/winprob_pbrs_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer

from agents.training.winprob_pbrs import (
    WinProbPbrsError,
    apply_winprob_pbrs,
    buffer_potentials,
    episode_shaping_sum,
    pbrs_shaping,
    successor_potential,
)

GAMMA = 0.99
COEF = 0.25


# ──────────────────────────────────────────────────────────────────────────────────────────────
# Fakes: the smallest thing that is still a REAL rollout buffer (so GAE is the genuine SB3 code).
# ──────────────────────────────────────────────────────────────────────────────────────────────

class _FakeExtractor:
    def __init__(self):
        self.last_win_prob_logits = None


class _FakePolicy:
    """A policy whose `predict_values` stashes a win-prob logit derived from the obs, exactly as the
    real extractor does. `phi_fn` maps the observation's first scalar to a logit."""

    def __init__(self, phi_fn=None, param=None):
        self.features_extractor = _FakeExtractor()
        self._phi_fn = phi_fn or (lambda x: x)
        self._param = param
        self.grad_enabled_seen = []

    def predict_values(self, obs):
        self.grad_enabled_seen.append(th.is_grad_enabled())
        x = obs["observation"][:, 0].to(th.float32)
        logits = self._phi_fn(x).reshape(-1, 1)
        if self._param is not None:            # a real graph, so a missing detach would be caught
            logits = logits * self._param
        self.features_extractor.last_win_prob_logits = logits
        return th.zeros((x.shape[0], 1), dtype=th.float32)


class _FakeModel:
    def __init__(self, buf, policy, coef=COEF, gamma=GAMMA):
        self.policy = policy
        self.device = th.device("cpu")
        self.gamma = gamma
        self.win_prob_pbrs_coef = coef
        self.rollout_buffer = buf
        self._last_obs = None
        self._last_episode_starts = None


def _make_buffer(n_steps, n_envs, obs_dim=1):
    obs_space = spaces.Dict({"observation": spaces.Box(-10.0, 10.0, (obs_dim,), np.float32)})
    buf = DictRolloutBuffer(n_steps, obs_space, spaces.Discrete(2),
                            device="cpu", gae_lambda=1.0, gamma=GAMMA, n_envs=n_envs)
    buf.pos, buf.full = n_steps, True
    buf.observations["observation"][:] = 0.0
    buf.actions[:] = 0
    buf.rewards[:] = 0.0
    buf.values[:] = 0.0
    buf.log_probs[:] = 0.0
    buf.episode_starts[:] = 0.0
    return buf


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 1. TELESCOPING — the invariance identity
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_a_complete_episodes_discounted_shaping_sum_is_exactly_minus_coef_times_phi_of_its_start():
    """The whole theorem, as one equality. Two episodes in one column, both complete."""
    n_steps = 8
    phi = np.array([[0.1], [0.9], [0.4], [0.7], [0.2], [0.55], [0.35], [0.8]], dtype=np.float64)
    es = np.zeros((n_steps, 1), dtype=np.float64)
    es[0, 0] = 1.0                     # episode A: rows 0..3
    es[4, 0] = 1.0                     # episode B: rows 4..7 (still running at the boundary)
    phi_next = successor_potential(phi, es, np.array([0.0]), np.array([1.0]))
    shaping = pbrs_shaping(phi, phi_next, GAMMA, COEF)

    # episode A is complete (row 4 starts a new one) -> the identity must hold exactly
    eps = episode_shaping_sum(shaping, es, GAMMA)
    assert len(eps) == 1, "only the FIRST episode both starts and ends inside the buffer"
    start, end, total = eps[0]
    assert (start, end) == (0, 3)
    assert total == pytest.approx(-COEF * phi[0, 0], abs=1e-12)


def test_the_identity_holds_for_every_complete_episode_across_many_random_layouts():
    """Not one hand-picked case: 40 random episode layouts, every complete episode checked."""
    rng = np.random.default_rng(20260829)
    for _ in range(40):
        n_steps, n_envs = int(rng.integers(4, 30)), int(rng.integers(1, 4))
        phi = rng.random((n_steps, n_envs))
        es = (rng.random((n_steps, n_envs)) < 0.25).astype(np.float64)
        es[0, :] = 1.0
        last_dones = (rng.random(n_envs) < 0.5).astype(np.float64)
        phi_next = successor_potential(phi, es, rng.random(n_envs), last_dones)
        shaping = pbrs_shaping(phi, phi_next, GAMMA, COEF)
        for e in range(n_envs):
            for start, _end, total in episode_shaping_sum(shaping, es, GAMMA, env_index=e):
                assert total == pytest.approx(-COEF * phi[start, e], abs=1e-9)


def test_a_terminal_row_gets_a_ZERO_successor_potential_and_the_row_before_it_does_not():
    phi = np.array([[0.2], [0.6], [0.9]], dtype=np.float64)
    es = np.array([[1.0], [0.0], [1.0]], dtype=np.float64)   # row 1 ENDED its episode (row 2 starts one)
    phi_next = successor_potential(phi, es, np.array([0.44]), np.array([0.0]))
    assert phi_next[0, 0] == pytest.approx(0.6), "row 0 continues -> the successor is row 1's phi"
    assert phi_next[1, 0] == 0.0, "row 1 terminated -> phi(s') := 0 by convention"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 2. TRUNCATION — the buffer boundary is NOT a terminal
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_an_episode_still_running_at_the_buffer_boundary_BOOTSTRAPS_instead_of_being_zeroed():
    """The classic PBRS bug: a phantom -coef*phi penalty for the rollout merely ending."""
    phi = np.array([[0.3], [0.8]], dtype=np.float64)
    es = np.array([[1.0], [0.0]], dtype=np.float64)
    boot = np.array([0.77])
    running = successor_potential(phi, es, boot, np.array([0.0]))     # episode continues
    ended = successor_potential(phi, es, boot, np.array([1.0]))       # episode ended on the last row
    assert running[-1, 0] == pytest.approx(0.77), "truncation bootstraps phi(s_T)"
    assert ended[-1, 0] == 0.0, "a real terminal zeroes it"
    # And the discounted sums differ by exactly the bootstrap term — the shaping the bug would eat.
    sr = pbrs_shaping(phi, running, GAMMA, COEF)[:, 0]
    se = pbrs_shaping(phi, ended, GAMMA, COEF)[:, 0]
    disc = np.array([1.0, GAMMA])
    assert float((sr * disc).sum() - (se * disc).sum()) == pytest.approx(
        COEF * GAMMA * GAMMA * 0.77, abs=1e-12)


def test_the_terminal_test_reads_the_NEXT_rows_episode_start_not_the_current_rows():
    """A revert-catcher for the one indexing mistake this code can make.

    SB3's own GAE uses ``1 - episode_starts[step + 1]`` for `next_non_terminal`; if the shaping
    used ``episode_starts[step]`` instead, the two notions of "terminal" would silently disagree
    and the advantages would be built on a different episode segmentation than the shaping.
    """
    phi = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
    es = np.array([[1.0], [0.0], [1.0]], dtype=np.float64)
    pn = successor_potential(phi, es, np.array([0.0]), np.array([1.0]))
    # Under the CORRECT rule row 1 is the terminal. Under the off-by-one rule row 0 would be.
    assert pn[0, 0] != 0.0 and pn[1, 0] == 0.0


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 3. DETACHMENT — a test that FAILS if a gradient can flow through the potential
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_phi_is_read_with_gradients_DISABLED_and_reaches_the_buffer_detached():
    """Deleting the `no_grad` in `_forward_phi` / `apply_winprob_pbrs` fails this test.

    Two independent assertions, because either alone is escapable: (a) grad was disabled at every
    forward, and (b) the parameter the logit was built from received no gradient and the buffer
    holds plain numpy.
    """
    buf = _make_buffer(4, 2)
    buf.observations["observation"][:, :, 0] = 0.5
    param = th.nn.Parameter(th.tensor(1.0))
    pol = _FakePolicy(param=param)
    model = _FakeModel(buf, pol)
    model._last_obs = {"observation": np.full((2, 1), 0.5, dtype=np.float32)}
    model._last_episode_starts = np.zeros(2, dtype=np.float32)

    apply_winprob_pbrs(model, buf)

    assert pol.grad_enabled_seen and not any(pol.grad_enabled_seen), (
        "every phi forward must run under torch.no_grad() — a graph built here would keep the whole "
        "rollout's activations alive and open a path from the policy loss into the potential")
    assert param.grad is None
    assert isinstance(buf.rewards, np.ndarray) and not hasattr(buf.rewards, "grad_fn")


def test_a_requires_grad_logit_still_reaches_numpy_rather_than_raising():
    """`.numpy()` on a grad-tracking tensor RAISES. That this passes is the detach, proven."""
    logits = (th.ones(3, 1) * th.nn.Parameter(th.tensor(2.0)))
    assert logits.requires_grad
    from agents.training.winprob_pbrs import _phi_from_logits
    phi = _phi_from_logits(logits, "test")
    assert phi.shape == (3,) and np.all((phi > 0) & (phi < 1))


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 4. OFF byte-identity
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_at_coef_zero_nothing_in_the_buffer_moves():
    buf = _make_buffer(4, 2)
    rng = np.random.default_rng(7)
    buf.rewards[:] = rng.random(buf.rewards.shape)
    buf.values[:] = rng.random(buf.values.shape)
    buf.compute_returns_and_advantage(last_values=th.zeros(2, 1), dones=np.zeros(2))
    before = (buf.rewards.copy(), buf.returns.copy(), buf.advantages.copy())

    model = _FakeModel(buf, _FakePolicy(), coef=0.0)
    assert apply_winprob_pbrs(model, buf) == {}
    for a, b in zip(before, (buf.rewards, buf.returns, buf.advantages)):
        assert np.array_equal(a, b)


def test_collect_rollouts_does_not_even_IMPORT_the_shaping_when_the_coef_is_zero():
    """Source contract: the guard is on the coefficient and the import is LOCAL to the branch.

    A module-level import would make an OFF run pay for a feature it does not use, and — more to
    the point here — would make "off is byte-identical" a claim about a code path that still ran.
    """
    from agents.training.instrumented_ppo.ppo import InstrumentedMaskablePPO
    src = inspect.getsource(InstrumentedMaskablePPO.collect_rollouts)
    assert 'getattr(self, "win_prob_pbrs_coef", 0.0) or 0.0) != 0.0' in src
    assert "from agents.training.winprob_pbrs import apply_winprob_pbrs" in src, (
        "the import must live INSIDE the non-zero branch")
    # and it must not be a module-level import of the ppo module
    import agents.training.instrumented_ppo.ppo as ppo_mod
    assert "winprob_pbrs" not in inspect.getsource(ppo_mod).split("class InstrumentedMaskablePPO")[0]


def test_the_class_default_is_off():
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    assert InstrumentedMaskablePPO.win_prob_pbrs_coef == 0.0


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 5. THE LIVE PATH — GAE recomputation, raw reward space, and the metrics
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _live_case(coef=COEF, n_steps=6, n_envs=2, seed=11):
    buf = _make_buffer(n_steps, n_envs)
    rng = np.random.default_rng(seed)
    buf.observations["observation"][:, :, 0] = rng.random((n_steps, n_envs)).astype(np.float32)
    buf.rewards[:] = rng.random((n_steps, n_envs)).astype(np.float32)
    buf.values[:] = rng.random((n_steps, n_envs)).astype(np.float32)
    buf.episode_starts[0, :] = 1.0
    buf.episode_starts[3, :] = 1.0
    model = _FakeModel(buf, _FakePolicy(), coef=coef)
    model._last_obs = {"observation": rng.random((n_envs, 1)).astype(np.float32)}
    model._last_episode_starts = np.zeros(n_envs, dtype=np.float32)
    return model, buf


def test_the_shaping_lands_in_RAW_rewards_and_the_returns_are_recomputed_from_them():
    model, buf = _live_case()
    raw_before = buf.rewards.copy()
    buf.compute_returns_and_advantage(last_values=th.zeros(2, 1), dones=np.zeros(2))
    returns_before = buf.returns.copy()

    metrics = apply_winprob_pbrs(model, buf)

    delta = buf.rewards - raw_before
    phi = buffer_potentials(model, buf)
    # The last row's successor is the bootstrap φ(s_T) from `_last_obs`, which this reconstruction
    # deliberately does not model — so the interior rows are compared exactly and the last is not.
    phi_next = successor_potential(phi, buf.episode_starts, np.zeros(2), np.zeros(2))
    # The shaping is visible in RAW reward space (PopArt has not run — it reads `returns` in train()).
    assert not np.allclose(delta, 0.0)
    assert not np.allclose(buf.returns, returns_before), "GAE must be re-run on the shaped stream"
    # and the interior rows match the pure arithmetic exactly (the last row uses the bootstrap)
    expected = pbrs_shaping(phi, phi_next, GAMMA, COEF)
    assert np.allclose(delta[:-1], expected[:-1], atol=1e-5)
    assert set(metrics) == {"shaping_mean", "shaping_absmean", "phi_mean", "reward_share"}
    assert metrics["reward_share"] > 0.0


def test_reward_share_is_quoted_against_the_UNSHAPED_stream():
    """A share measured against the shaped rewards would flatter itself as the coefficient rises —
    the metric exists to say when a coefficient has replaced the outcome signal, so its denominator
    has to be the signal being replaced."""
    model, buf = _live_case(coef=0.5)
    raw_absmean = float(np.abs(buf.rewards).mean())
    m = apply_winprob_pbrs(model, buf)
    assert m["reward_share"] == pytest.approx(m["shaping_absmean"] / raw_absmean, rel=1e-9)


def test_doubling_the_coefficient_doubles_the_shaping():
    m1 = apply_winprob_pbrs(*_live_case(coef=0.1))
    m2 = apply_winprob_pbrs(*_live_case(coef=0.2))
    assert m2["shaping_absmean"] == pytest.approx(2.0 * m1["shaping_absmean"], rel=1e-6)


def test_a_missing_win_prob_head_is_a_LOUD_refusal_not_a_silent_no_op():
    """The invisible-regression class: a shaping term the operator believes is on, doing nothing."""
    class _Headless(_FakePolicy):
        def predict_values(self, obs):
            self.features_extractor.last_win_prob_logits = None
            return th.zeros((obs["observation"].shape[0], 1))

    buf = _make_buffer(3, 1)
    model = _FakeModel(buf, _Headless())
    model._last_obs = {"observation": np.zeros((1, 1), dtype=np.float32)}
    model._last_episode_starts = np.zeros(1, dtype=np.float32)
    with pytest.raises(WinProbPbrsError, match="win-prob-mode"):
        apply_winprob_pbrs(model, buf)


def test_the_potentials_read_every_row_of_the_buffer_even_across_chunk_boundaries():
    buf = _make_buffer(9, 3)
    buf.observations["observation"][:, :, 0] = np.arange(27, dtype=np.float32).reshape(9, 3) / 27.0
    model = _FakeModel(buf, _FakePolicy())
    small = buffer_potentials(model, buf, chunk=4)     # forces 7 chunks over 27 rows
    big = buffer_potentials(model, buf, chunk=1000)
    assert small.shape == (9, 3)
    assert np.allclose(small, big), "chunking must not reorder or drop rows"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 6. THE TB SCALARS + the config gates
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_train_records_the_pbrs_scalars_under_the_train_prefix():
    from agents.training.instrumented_ppo.ppo import InstrumentedMaskablePPO
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert 'self.logger.record(f"train/pbrs_{_pk}", float(_pv))' in src
    assert "if self._pbrs_metrics:" in src


def test_the_hparam_is_forwarded_on_BOTH_model_build_paths():
    """One table row, applied by one function to both the fresh and resume paths — the whole point
    of `_TRAINING_HPARAMS` is that a coefficient cannot be wired into one branch only."""
    from main.train.model_build import _TRAINING_HPARAMS
    assert any(name == "win_prob_pbrs_coef" for name, _ in _TRAINING_HPARAMS)


def test_the_coefficient_is_recorded_on_model_version_for_provenance():
    from agents.model.model_version import ModelVersion
    assert "win_prob_pbrs_coef" in {f.name for f in __import__("dataclasses").fields(ModelVersion)}


def test_a_pre_v104_config_migrates_to_the_off_default_rather_than_refusing():
    """0.0 is not a guess about the past — the flag did not exist, so no run could have used it.
    A REFUSAL here would make every archived checkpoint unreadable for a coefficient none of them
    carried."""
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config
    assert MODEL_CONFIG_VERSION == 104
    out = _migrate_config({"config_version": 103})
    assert out["win_prob_pbrs_coef"] == 0.0
    assert out["config_version"] == 104
    # a recorded value migrates UNTOUCHED
    out2 = _migrate_config({"config_version": 103, "win_prob_pbrs_coef": 0.3})
    assert out2["win_prob_pbrs_coef"] == 0.3


@pytest.mark.parametrize("argv,needle", [
    (["--win-prob-pbrs-coef", "-0.1"], "must be >= 0"),
    (["--win-prob-pbrs-coef", "0.1", "--win-prob-mode", "none"], "requires --win-prob-mode"),
])
def test_the_config_gates_refuse_the_two_ways_this_flag_can_be_wrong(argv, needle, capsys):
    from main.train.parser import build_parser
    from main.train.config import resolve_config
    parser = build_parser()
    args = parser.parse_args(["--steps", "1", "--debug", *argv])
    with pytest.raises(SystemExit):
        resolve_config(args, parser)
    assert needle in capsys.readouterr().err


def test_a_positive_coefficient_with_a_real_mode_is_accepted():
    from main.train.parser import build_parser
    parser = build_parser()
    args = parser.parse_args(["--steps", "1", "--win-prob-pbrs-coef", "0.1",
                              "--win-prob-mode", "shaping"])
    assert args.win_prob_pbrs_coef == 0.1
