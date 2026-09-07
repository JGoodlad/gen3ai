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
    # …and not a module-level import of the module that OWNS `collect_rollouts`. Resolved from the
    # function rather than named, so the check follows the method if it ever moves again — naming
    # `ppo` would leave this passing while testing a module that no longer holds the branch.
    owner = inspect.getmodule(InstrumentedMaskablePPO.collect_rollouts)
    assert "winprob_pbrs" not in inspect.getsource(owner).split("class ")[0]


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
    # `_FakeModel` sets no terminal scale, so the two sparsity-proof companions stay absent — a
    # denominator that was never supplied must not be invented.
    assert set(metrics) == {"shaping_mean", "shaping_absmean", "phi_mean", "reward_share",
                            "episode_dose_n"}
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
    from agents.training.instrumented_ppo import ppo as ppo_mod
    # The whole TRAIN STEP — these scalars are recorded in `metrics_export._record_term_metrics`.
    src = ppo_mod.train_step_source()
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
    # The migration chain must carry a pre-v104 config all the way to HEAD, not merely to 104 —
    # pinning the live constant here would make every later bump a false failure in this file.
    # `>=` because later versions keep landing above this one; what this test owns is that the
    # v104 STEP still injects the off default and stamps at least v104, not that v104 is the tip.
    assert MODEL_CONFIG_VERSION >= 104
    out = _migrate_config({"config_version": 103})
    assert out["win_prob_pbrs_coef"] == 0.0
    assert out["config_version"] == MODEL_CONFIG_VERSION
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


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 8. FROZEN φ (`--win-prob-pbrs-source`, gen3_winprob_pbrs_source_v1)
#
# WHY THIS EXISTS. The invariance theorem at the top of this file assumes φ is a FIXED function of
# state. Ours is a head inside the network being trained, so exact invariance holds only WITHIN a
# rollout and degrades across them. A frozen source removes the caveat entirely — which makes the
# correctness bar unusually crisp: pointed at a run's OWN current checkpoint, the frozen path must
# produce BIT-IDENTICAL shaping to the live path. Anything less means the two forwards differ, and
# a φ that differs from the head it claims to be is a silently wrong potential.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_with_no_source_phi_comes_from_the_LIVE_model_which_is_the_v104_behaviour():
    from agents.training.winprob_pbrs import phi_model
    model, _ = _live_case()
    assert phi_model(model) is model
    model._winprob_phi_source = None          # the attribute EXISTING but None must not change it
    assert phi_model(model) is model


def test_a_frozen_source_supplies_phi_for_EVERY_buffer_row():
    """The routing, at its most visible: a source whose φ is a constant makes φ constant, whatever
    the live head would have said about the same observations."""
    from agents.training.winprob_pbrs import phi_model
    model, buf = _live_case()
    live_phi = buffer_potentials(model, buf)
    model._winprob_phi_source = _FakeModel(buf, _FakePolicy(phi_fn=lambda x: th.zeros_like(x)))
    assert phi_model(model) is model._winprob_phi_source
    frozen_phi = buffer_potentials(model, buf)
    assert np.allclose(frozen_phi, 0.5)                 # sigmoid(0)
    assert not np.allclose(live_phi, frozen_phi)


def test_the_BOOTSTRAP_potential_comes_from_the_source_too():
    """The last row's successor is φ(s_T). A frozen φ on the buffer rows and a LIVE φ on the
    bootstrap would break the telescoping at every truncation boundary — the classic half-fix."""
    model, buf = _live_case()
    model._last_episode_starts = np.zeros(2, dtype=np.float32)   # both episodes still running
    model._winprob_phi_source = _FakeModel(buf, _FakePolicy(phi_fn=lambda x: th.zeros_like(x)))
    raw_before = buf.rewards.copy()
    apply_winprob_pbrs(model, buf)
    delta = (buf.rewards - raw_before)[-1]
    # φ(s) == φ(s′) == 0.5 everywhere under the constant source ⇒ shaping = coef·(γ−1)·0.5
    assert np.allclose(delta, COEF * (GAMMA - 1.0) * 0.5, atol=1e-6)


def test_the_GAE_BOOTSTRAP_VALUE_still_comes_from_the_LIVE_critic():
    """φ and `last_values` are different quantities that happened to share one forward. The value
    is the collector's own bootstrap and must stay the LIVE critic's, or the recomputed advantages
    stop being the shaped-stream counterpart of the ones collection produced."""
    class _Counting(_FakePolicy):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.calls = 0

        def predict_values(self, obs):
            self.calls += 1
            return super().predict_values(obs)

    model, buf = _live_case()
    live_pol, src_pol = _Counting(), _Counting(phi_fn=lambda x: th.zeros_like(x))
    model.policy = live_pol
    model._winprob_phi_source = _FakeModel(buf, src_pol)
    apply_winprob_pbrs(model, buf)
    assert live_pol.calls == 1, "the live model may be forwarded ONLY for the GAE bootstrap value"
    assert src_pol.calls >= 2, "the source must serve every buffer chunk AND the bootstrap φ"


def test_a_foreign_obs_space_is_filtered_to_the_keys_the_source_KNOWS():
    """A prior-generation φ is the point of the flag (that is where a MATURE potential lives), and
    such a checkpoint has an older Dict obs space. SB3's `preprocess_obs` iterates the keys it is
    handed against the space, so an extra key is a KeyError — the same filter the exploiter-distill
    teachers use in `train()`."""
    from agents.training.winprob_pbrs import _phi_obs
    class _Sp:
        spaces = {"observation": None}
    class _Net:
        observation_space = _Sp()
    out = _phi_obs(_Net(), {"observation": np.zeros((2, 1)), "distill_mask": np.zeros((2, 1))})
    assert set(out) == {"observation"}
    # A φ network with no declared space (the fakes above) is passed through untouched.
    assert set(_phi_obs(object(), {"a": 1, "b": 2})) == {"a", "b"}


def test_the_frozen_source_is_NEVER_pickled_into_our_checkpoint():
    """The `_distill_teacher` genre exactly: a full frozen FOREIGN model. Saving it would embed
    another run's weights in every checkpoint of this one — and `--win-prob-pbrs-source` is
    inherited on a flagless resume precisely so it is re-loaded from its own path instead."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    class _Bare(InstrumentedMaskablePPO):        # the real MRO, no env / no policy construction
        def __init__(self):
            pass

    assert "_winprob_phi_source" in _Bare()._excluded_save_params()


# ── 8b. THE IDENTITY TEST, on a REAL Gen3 policy through the REAL loader ───────────────────────

def _real_gen3_ppo():
    """A real `InstrumentedMaskablePPO` on the real `Gen3FeaturesExtractor` with a win-prob head.

    The fakes above pin the ROUTING; only a real extractor can pin that a frozen source reproduces
    the live φ, because that claim is about `CLSPool.value_cls -> WinProbHead` running over the
    frozen trunk's own `value_pooled`. ~1.5 s on CPU, no battles, no data beyond the mappings.
    """
    import inspect
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agents.action.constants import ACTION_SPACE_SIZE
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    class _Env(gym.Env):
        def __init__(self, dim):
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(0.0, 1.0, (dim,), np.float32),
                "action_mask": spaces.Box(0, 1, (ACTION_SPACE_SIZE,), np.int8)})
            self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
            self._d = dim

        def _o(self):
            return {"observation": np.zeros(self._d, np.float32),
                    "action_mask": np.ones(ACTION_SPACE_SIZE, np.int8)}

        def reset(self, **kw):
            return self._o(), {}

        def step(self, a):
            return self._o(), 0.0, True, False, {}

        def action_masks(self):
            return np.ones(ACTION_SPACE_SIZE, bool)

    maps = load_mappings()
    enc = Gen3ObservationEncoder(maps)
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in {**enc.get_features_extractor_kwargs(),
                            "win_prob_mode": "read_only"}.items() if k in sig}
    th.manual_seed(0)
    model = InstrumentedMaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=8, batch_size=8, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64])})
    return model, maps, enc


def _real_buffer(model, enc, n_steps=4, n_envs=2, seed=0):
    from stable_baselines3.common.buffers import DictRolloutBuffer
    buf = DictRolloutBuffer(n_steps, model.observation_space, model.action_space,
                            device="cpu", n_envs=n_envs)
    buf.pos, buf.full = n_steps, True
    rng = np.random.default_rng(seed)
    buf.observations["observation"][:] = rng.random(
        (n_steps, n_envs, enc.dimension)).astype(np.float32)
    buf.observations["action_mask"][:] = 1
    buf.rewards[:] = 0.0
    buf.episode_starts[:] = 0.0
    buf.episode_starts[0, :] = 1.0
    return buf


def test_a_frozen_source_that_IS_our_own_checkpoint_reproduces_live_phi_BIT_FOR_BIT(tmp_path):
    """THE correctness check, and it is as strong as this claim can be made.

    Save the live model, reload it through the REAL `load_foreign_opponent` loader `model_build`
    uses, attach it as the frozen source — and every φ must come back bit-identical. A head-only
    shortcut (running the frozen head over the LIVE trunk's `value_pooled`) fails this, and so does
    any obs-key or eval-mode discrepancy between the two forwards.
    """
    import dataclasses
    import json
    from agents.model.snapshot import current_model_version, load_foreign_opponent

    model, maps, enc = _real_gen3_ppo()
    buf = _real_buffer(model, enc)
    live_phi = buffer_potentials(model, buf)

    mv = current_model_version(maps, win_prob_mode="read_only")
    zip_path = tmp_path / "ckpt.zip"
    model.save(str(zip_path))
    (tmp_path / "model_config.json").write_text(json.dumps(dataclasses.asdict(mv)))

    source, foreign_v = load_foreign_opponent(str(zip_path), current_version=mv, device="cpu")
    source.policy.set_training_mode(False)
    assert foreign_v.arch_signature == mv.arch_signature
    model._winprob_phi_source = source

    frozen_phi = buffer_potentials(model, buf)
    assert np.array_equal(live_phi, frozen_phi), (
        "a frozen source that is our own checkpoint must give the identical potential; "
        f"max|Δ| = {np.abs(live_phi - frozen_phi).max()}")


def test_the_frozen_potential_does_NOT_move_when_the_live_network_does(tmp_path):
    """The property the flag exists to buy, and the anti-vacuity half of the test above: the
    identity must come from the frozen source genuinely being read, not from the source being
    ignored. Drift the live weights and the frozen φ must not budge while the live φ does."""
    import copy
    model, _maps, enc = _real_gen3_ppo()
    buf = _real_buffer(model, enc)
    live_before = buffer_potentials(model, buf)

    model._winprob_phi_source = copy.deepcopy(model)
    model._winprob_phi_source.policy.set_training_mode(False)
    frozen_before = buffer_potentials(model, buf)

    with th.no_grad():
        for p in model.policy.parameters():
            p.add_(0.05)

    assert np.array_equal(frozen_before, buffer_potentials(model, buf)), "the frozen φ drifted"
    model._winprob_phi_source = None
    assert not np.array_equal(live_before, buffer_potentials(model, buf)), (
        "the live φ did not move, so this test proved nothing about freezing")


# ── 8c. `--compile-trainer` × the frozen extractor ────────────────────────────────────────────
#
# ⚠️ HONEST SCOPE. `compile_trainer_extractor` REFUSES a non-cuda device by design, so the real
# Inductor path cannot run in this (CPU-only) tier. What is exercised here is the seam that makes
# the interaction safe: the compile patches ONE bound method on the LIVE policy and knows nothing
# about the source, and the source's own forward is a different object that keeps serving φ.
# WHAT REMAINS UNEXERCISED: a real `torch.compile` on CUDA with a frozen source attached, and
# whether compiling the SOURCE would pay for itself (it is deliberately left eager — it runs once
# per rollout, not per minibatch, so a second Inductor graph would buy a warm-up and nothing else).

def test_the_trainer_compile_patches_only_the_LIVE_policy_and_never_the_source():
    """A source-level statement of scope: the compile module addresses `model.policy` and has no
    knowledge of `_winprob_phi_source`, so it cannot reach the frozen network by construction."""
    import inspect as _inspect
    from agents.model import compile_trainer
    src = _inspect.getsource(compile_trainer)
    assert "_winprob_phi_source" not in src
    assert "policy.features_extractor" in src


def test_patching_the_live_extractors_forward_leaves_the_frozen_source_serving_phi():
    """The behavioural half, with the compile's *effect* (a replaced bound `fe.forward`) simulated
    on CPU: after the live extractor's forward is swapped for a poisoned one, φ must still come out
    of the frozen source, unchanged."""
    import copy
    model, _maps, enc = _real_gen3_ppo()
    buf = _real_buffer(model, enc)
    model._winprob_phi_source = copy.deepcopy(model)
    model._winprob_phi_source.policy.set_training_mode(False)
    frozen_before = buffer_potentials(model, buf)

    live_fe = model.policy.features_extractor
    src_fe = model._winprob_phi_source.policy.features_extractor
    assert live_fe is not src_fe
    def _poisoned(*a, **k):
        raise AssertionError("the LIVE extractor must not be forwarded for φ when a source is set")
    live_fe.forward = _poisoned                       # exactly what the compile does: bind a new fn

    assert np.array_equal(frozen_before, buffer_potentials(model, buf))
    # ...and the source's own forward was never touched by the patch.
    assert getattr(src_fe, "forward").__func__ is type(src_fe).forward


# ── 8d. THE FLAG: gates, provenance, resume inheritance ───────────────────────────────────────

def test_the_source_flag_requires_a_positive_coefficient():
    """A source with no coefficient loads a whole extra network, forwards it once per rollout, and
    multiplies the result by zero — the same invisible-no-op class the coef/mode gate guards."""
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    parser = build_parser()
    args = parser.parse_args(["--steps", "1", "--debug", "--win-prob-pbrs-source", "x.zip"])
    with pytest.raises(SystemExit):
        resolve_config(args, parser)


def test_the_source_is_recorded_on_model_version_for_provenance():
    """A clean-world run is uninterpretable if the identity of its frozen potential is not pinned."""
    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, win_prob_pbrs_coef=0.5,
        win_prob_pbrs_source="models/rev1/checkpoints/c.zip")
    assert v.win_prob_pbrs_source == "models/rev1/checkpoints/c.zip"
    assert ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}).win_prob_pbrs_source is None


def test_the_source_is_INHERITED_on_a_flagless_resume():
    """It rides with the coefficient. A resume that silently reverted to live-φ would change the
    objective mid-run — approximate invariance instead of exact — with nothing saying so."""
    from main.train import config as _cfg
    src = inspect.getsource(_cfg)
    assert '_resolve("win_prob_pbrs_source"' in src


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 9. THE SIZING COMPANION — a meter that survives a SPARSE reward stream (probe N §7.5)
#
# `reward_share`'s denominator is the unshaped stream's own mean |reward|. That is the right
# question on a dense stream and a broken one on the stream this lever was built for: under
# `--no-hand-shaping` the unshaped reward is TERMINAL-ONLY, so the denominator is exactly 0 on a
# rollout with no episode end and is "±V ÷ episode length" otherwise — a meter that moves with the
# episode length rather than with the coefficient. The companions divide by the run's terminal
# magnitude instead, which is a constant.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _sparse_case(coef=COEF, scale=1.0, n_steps=6, n_envs=2, seed=11, terminal_rows=(2, 5)):
    """A CLEAN-WORLD-shaped rollout: every reward zero except a ±`scale` terminal on episode ends."""
    model, buf = _live_case(coef=coef, n_steps=n_steps, n_envs=n_envs, seed=seed)
    buf.rewards[:] = 0.0
    for r in terminal_rows:
        if r < n_steps:
            buf.rewards[r, :] = scale         # the ±1 terminal, on the row whose episode ended
    model.win_prob_pbrs_terminal_scale = scale
    return model, buf


def test_the_companion_reports_a_SIZE_where_reward_share_can_only_report_NaN():
    """The failure this companion exists for. A terminal-only stream with no episode end inside the
    rollout has NO unshaped magnitude to divide by: `reward_share` is NaN there (R1's F3 fixed the
    old, worse `0.0`), which is honest but is not a SIZE. `terminal_share` still reports one."""
    model, buf = _sparse_case(terminal_rows=())        # no terminal lands inside this rollout
    m = apply_winprob_pbrs(model, buf)
    assert np.isnan(m["reward_share"]), "undefined must read undefined, never 0.0"
    assert m["shaping_absmean"] > 0.0, "the shaping itself is very much non-zero"
    assert m["terminal_share"] > 0.0, "and the companion still reports its size"


def test_the_companions_are_present_whenever_a_terminal_scale_is_known():
    model, buf = _sparse_case()
    m = apply_winprob_pbrs(model, buf)
    assert {"terminal_share", "episode_dose", "episode_dose_n"} <= set(m)
    assert m["episode_dose_n"] == 2.0        # rows 0-2 and 3-5 complete in each of 2 env columns


def test_no_terminal_scale_means_no_companion_rather_than_a_made_up_denominator():
    model, buf = _live_case()
    assert not hasattr(model, "win_prob_pbrs_terminal_scale")
    m = apply_winprob_pbrs(model, buf)
    assert "terminal_share" not in m and "episode_dose" not in m


def test_terminal_share_is_INDEPENDENT_of_episode_length_where_reward_share_is_not():
    """The defect, as a measurement. Same coefficient, same φ, two rollouts differing only in how
    many terminals land inside them: `reward_share` moves by a factor of ~2 (its denominator is the
    terminal mass spread over the rows), `terminal_share` does not move at all."""
    m_dense = apply_winprob_pbrs(*_sparse_case(terminal_rows=(2, 5)))
    m_rare = apply_winprob_pbrs(*_sparse_case(terminal_rows=(5,)))
    assert m_dense["terminal_share"] == pytest.approx(m_rare["terminal_share"], rel=1e-9)
    assert m_dense["reward_share"] < 0.75 * m_rare["reward_share"]


def test_episode_dose_is_the_TELESCOPED_budget_and_equals_coef_times_phi_of_the_start():
    """The identity is what makes this a *sizing* number rather than a summary statistic: a complete
    episode's discounted shaping sum is exactly −coef·φ(s_0), so the meter reads the shaping's whole
    per-episode budget priced against one win."""
    coef, scale = 0.3, 1.0
    model, buf = _sparse_case(coef=coef, scale=scale)
    phi = buffer_potentials(model, buf)
    m = apply_winprob_pbrs(model, buf)
    # Only the episode starting at row 0 both STARTS and ENDS inside this buffer (the row-3 one is
    # still running at the boundary and is deliberately not counted — its sum is not yet its budget).
    assert m["episode_dose"] == pytest.approx(coef * float(np.mean(phi[0, :])) / scale, rel=1e-6)


def test_episode_dose_scales_with_the_coefficient_which_is_the_whole_point_of_a_ladder():
    m1 = apply_winprob_pbrs(*_sparse_case(coef=0.1))
    m3 = apply_winprob_pbrs(*_sparse_case(coef=0.3))
    assert m3["episode_dose"] == pytest.approx(3.0 * m1["episode_dose"], rel=1e-6)


def test_a_bigger_terminal_makes_the_SAME_shaping_a_smaller_share_of_a_win():
    """±30 vs ±1 is exactly the re-sizing the clean world performs, and the meter has to see it —
    otherwise a coefficient carried over from the ±30 era reads 'fine' at 1/30th of its dose."""
    m1 = apply_winprob_pbrs(*_sparse_case(coef=0.3, scale=1.0))
    m30 = apply_winprob_pbrs(*_sparse_case(coef=0.3, scale=30.0))
    assert m30["episode_dose"] == pytest.approx(m1["episode_dose"] / 30.0, rel=1e-6)


def test_episode_dose_reports_its_SAMPLE_and_is_absent_when_there_is_none():
    """A dose averaged over zero episodes is not a dose. `episode_dose_n` is always emitted so a
    reader can tell 'this rollout had no complete episode' from 'the dose is small'."""
    model, buf = _sparse_case(n_steps=6, terminal_rows=())
    buf.episode_starts[:] = 0.0                        # one long episode, complete in neither end
    buf.episode_starts[0, :] = 1.0
    m = apply_winprob_pbrs(model, buf)
    assert m["episode_dose_n"] == 0.0
    assert "episode_dose" not in m
    assert "terminal_share" in m, "the per-step companion needs no episodes at all"


def test_the_denominator_is_derived_from_victory_value_on_BOTH_build_paths():
    """It is a DERIVED attribute, not a `_TRAINING_HPARAMS` row (the arg has a different name), so
    the guard has to be that the one function both build paths call sets it."""
    from main.train import model_build
    src = inspect.getsource(model_build.apply_training_hparams)
    assert "model.win_prob_pbrs_terminal_scale" in src
    assert 'getattr(args, "victory_value"' in src
    assert model_build.apply_training_hparams.__name__ in inspect.getsource(model_build)


def test_the_class_default_terminal_scale_is_a_no_op():
    """A smoke, a unit test or a frozen opponent that never sets it must emit no companion at all
    rather than divide by a fictitious 30."""
    from agents.training.instrumented_ppo.hparams import PpoHyperparameters
    assert PpoHyperparameters.win_prob_pbrs_terminal_scale == 0.0


def test_episode_dose_pools_every_env_column():
    """`episode_shaping_sum` is per-column by design; the dose is a property of the ROLLOUT."""
    from agents.training.winprob_pbrs import episode_dose as _dose
    shaping = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    es = np.zeros((4, 2)); es[0, :] = 1.0; es[2, :] = 1.0
    mean_abs, n = _dose(shaping, es, gamma=1.0)
    # One COMPLETE episode per column (rows 0-1); the row-2 one is still running at the boundary.
    assert n == 2, "both columns contribute — a per-column reading would say 1"
    assert mean_abs == pytest.approx((2.0 + 4.0) / 2.0)
