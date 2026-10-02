"""The FORK ARM's rollout hook, end to end on a synthetic buffer (`gen3_fork_v1`).

`ForkArmCallback._on_rollout_end` is the one place the selector, the fan-out, the row assembly and
the buffer surgery meet, and the properties worth pinning are the ones that make it SAFE to leave
on: it never FABRICATES a row, a mis-wired arm is FATAL_CONFIG rather than a print-and-continue,
an arm that injects nothing for its declared floor of rollouts is FATAL_SUPPLY
(`gen3_supply_guard_v2`), and at `--fork-fraction 0` it does not even ASK the driver.

The branch continuations are stubbed — playing real ones is the smoke's job, and `fork_crn_sim_test`
is what puts the bridge under test. What is real here is every array the callback touches.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

from agents.action.constants import MOVE_START
from agents.training.fork_arm import PG_MASK_KEY
from agents.training.fork_buffer import fork_buffer_class
from agents.training.fork_callback import ForkArmCallback
from agents.training.lever_supply import LeverConfigError, LeverStarvedError
from main.exit_codes import exit_code_for

D, A = 5, 11
N_STEPS, N_ENVS = 6, 2


class _Policy:
    _critic_mode = "winprob"
    training = False

    def set_training_mode(self, mode):
        self.training = bool(mode)

    def get_distribution(self, d):
        n = int(d["observation"].shape[0])
        # A deterministic, CONTESTED-looking logit field: every legal action within 0.01 of its
        # neighbour, so the top-2 gap is small and the quantile has something to cut.
        logits = th.arange(A, dtype=th.float32).reshape(1, A).repeat(n, 1) * 0.01
        return type("D", (), {"distribution": type("C", (), {"logits": logits})()})()

    def predict_values(self, d):
        return th.full((int(d["observation"].shape[0]), 1), 0.5)


class _Model:
    n_steps, n_envs = N_STEPS, N_ENVS
    seed, num_timesteps = 7, 1000
    gamma, gae_lambda = 1.0, 0.95
    device = "cpu"
    fork_fraction = 0.5           # ask for a lot; the eligible set is what bounds it
    fork_branches = 3
    fork_contested_gap = 1.0
    fork_contested_absv = 0.0
    fork_max_per_battle = 1
    fork_crn = "dice_and_draws"

    def __init__(self, buf):
        self.rollout_buffer = buf
        self.policy = _Policy()
        self._win_handle_keys = np.empty((N_STEPS, N_ENVS), dtype=object)
        self._win_handle_turns = np.full((N_STEPS, N_ENVS), 10, dtype=np.int64)
        for t in range(N_STEPS):
            for e in range(N_ENVS):
                self._win_handle_keys[t, e] = f"{t}_{e}"
        self._fork_metrics = None

    def save(self, path):
        open(path, "wb").write(b"stub")


def _space(with_key=True):
    d = {"observation": spaces.Box(-1, 1, (D,), np.float32),
         "action_mask": spaces.Box(0, 1, (A,), np.float32),
         "win_mask": spaces.Box(0, 1, (1,), np.float32)}
    if with_key:
        d[PG_MASK_KEY] = spaces.Box(0, 1, (1,), np.float32)
    return spaces.Dict(d)


def _buffer(cls, with_key=True):
    buf = cls(N_STEPS, _space(with_key), spaces.Discrete(A), gae_lambda=0.95, gamma=1.0,
              n_envs=N_ENVS)
    am = np.zeros((N_ENVS, A), np.float32)
    am[:, MOVE_START:MOVE_START + 4] = 1.0
    for t in range(N_STEPS):
        obs = {"observation": np.zeros((N_ENVS, D), np.float32), "action_mask": am.copy(),
               "win_mask": np.ones((N_ENVS, 1), np.float32)}
        if with_key:
            obs[PG_MASK_KEY] = np.ones((N_ENVS, 1), np.float32)
        buf.add(obs, np.zeros((N_ENVS, 1)), np.zeros(N_ENVS, np.float32),
                np.array([1.0 if t == 0 else 0.0] * N_ENVS, np.float32),
                th.zeros(N_ENVS, 1), th.zeros(N_ENVS),
                action_masks=am.copy())
    buf.compute_returns_and_advantage(th.zeros(N_ENVS, 1), np.ones(N_ENVS, bool))
    return buf


def _cb(model, tmp_path, records=True):
    cb = ForkArmCallback(records_dir=str(tmp_path) if records else None, impl="rust")
    cb.model = model
    return cb


def _stub_branches(monkeypatch, tmp_path, n_rows=3):
    """Stand in for the worker fan-out: two forks, three branches each, real row arrays."""
    calls = {"n": 0}

    def fake_play_forks(*, model, forks, impl, crn, workers=None, log=print):
        calls["n"] += 1
        calls["crn"] = crn
        calls["forks"] = list(forks)
        out = []
        for i, fk in enumerate(forks):
            branches = {}
            for j, (name, act) in enumerate(sorted((fk["actions"] or {}).items())):
                branches[name] = {
                    "outcome": float(j % 2), "capped": False, "turns": 20, "decisions": 12.0,
                    "obs": np.zeros((n_rows, D), np.float32),
                    "mask": np.ones((n_rows, A), np.int8),
                    "action": np.full(n_rows, int(act), dtype=np.int64),
                }
            out.append({"id": i, "branches": branches})
        return out, {"seconds": 0.5, "branches": 6.0, "branches_capped": 0.0,
                     "rows": float(len(forks) * 3 * n_rows)}

    monkeypatch.setattr("agents.training.fork_driver.play_forks", fake_play_forks)
    monkeypatch.setattr("agents.training.fork_driver._score",
                        lambda model, obs, masks, actions: (
                            np.linspace(0.1, 0.9, len(obs)).astype(np.float32),
                            np.full(len(obs), -1.0, np.float32)))
    monkeypatch.setattr("agents.training.cf_records.index_records",
                        lambda d: {f"{t}_{e}": str(tmp_path / "rec.json")
                                   for t in range(N_STEPS) for e in range(N_ENVS)})
    return calls


# ── OFF ──────────────────────────────────────────────────────────────────────────────────────
def test_OFF_never_touches_the_buffer_and_never_asks_the_driver(monkeypatch, tmp_path):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    m.fork_fraction = 0.0
    calls = _stub_branches(monkeypatch, tmp_path)
    cb = _cb(m, tmp_path)
    cb._on_rollout_start()
    cb._on_rollout_end()
    assert calls["n"] == 0
    assert m.rollout_buffer.n_fork_rows == 0
    assert m._fork_metrics is None


def test_a_non_winprob_critic_is_inert_even_with_the_flag_on(monkeypatch, tmp_path):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    m.policy._critic_mode = "shaped"
    calls = _stub_branches(monkeypatch, tmp_path)
    cb = _cb(m, tmp_path)
    cb._on_rollout_end()
    assert calls["n"] == 0 and m.rollout_buffer.n_fork_rows == 0


def test_no_records_ring_is_FATAL_CONFIG_not_a_print(monkeypatch, tmp_path):
    """gen3_supply_guard_v2: it used to print once and fork nothing for the rest of the run."""
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    cb = _cb(m, tmp_path, records=False)
    with pytest.raises(LeverConfigError) as ei:
        cb._on_rollout_end()
    assert "--cf-records" in str(ei.value) and exit_code_for(ei.value) == 3


# ── the mis-wirings are FATAL_CONFIG (they used to DISABLE the arm and keep training) ──────────
def test_a_plain_buffer_is_FATAL_CONFIG(monkeypatch, tmp_path):
    m = _Model(_buffer(MaskableDictRolloutBuffer))
    _stub_branches(monkeypatch, tmp_path)
    cb = _cb(m, tmp_path)
    with pytest.raises(LeverConfigError) as ei:
        cb._on_rollout_end()
    assert "ForkRolloutBuffer" in str(ei.value) and exit_code_for(ei.value) == 3


def test_an_unfillable_obs_key_is_FATAL_CONFIG_by_name_and_the_buffer_is_untouched(monkeypatch,
                                                                                   tmp_path):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    buf = _buffer(Fork)
    buf.observations["some_new_key"] = np.zeros((N_STEPS, N_ENVS, 1), np.float32)
    m = _Model(buf)
    _stub_branches(monkeypatch, tmp_path)
    cb = _cb(m, tmp_path)
    with pytest.raises(LeverConfigError) as ei:
        cb._on_rollout_end()
    assert "REFUSED" in str(ei.value) and "some_new_key" in str(ei.value)
    assert buf.n_fork_rows == 0


def test_a_failing_pass_leaves_the_buffer_exactly_as_collection_made_it(monkeypatch, tmp_path,
                                                                       capsys):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    _stub_branches(monkeypatch, tmp_path)

    def boom(**kw):
        raise RuntimeError("the bridge wedged")

    monkeypatch.setattr("agents.training.fork_driver.play_forks", boom)
    cb = _cb(m, tmp_path)
    cb._on_rollout_end()
    assert "pass failed" in capsys.readouterr().out
    assert m.rollout_buffer.n_fork_rows == 0


def test_an_arm_that_injects_NOTHING_for_its_floor_of_rollouts_is_FATAL_SUPPLY(monkeypatch,
                                                                              tmp_path):
    """One failed pass is "no forks this rollout"; every pass failing is a run that never forked."""
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    _stub_branches(monkeypatch, tmp_path)

    def boom(**kw):
        raise RuntimeError("the bridge wedged")

    monkeypatch.setattr("agents.training.fork_driver.play_forks", boom)
    cb = ForkArmCallback(records_dir=str(tmp_path), impl="rust", starve_cycles=3,
                         emit=lambda _m: None)
    cb.model = m
    cb._on_rollout_end()
    cb._on_rollout_end()
    with pytest.raises(LeverStarvedError) as ei:
        cb._on_rollout_end()
    assert exit_code_for(ei.value) == 5 and "the bridge wedged" in str(ei.value)
    assert m.rollout_buffer.n_fork_rows == 0


def test_a_pass_that_injects_rows_resets_the_streak(monkeypatch, tmp_path):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    _stub_branches(monkeypatch, tmp_path, n_rows=2)
    cb = ForkArmCallback(records_dir=str(tmp_path), impl="rust", starve_cycles=2,
                         emit=lambda _m: None)
    cb.model = m
    cb._guard.streak = 1
    cb._on_rollout_start()
    cb._on_rollout_end()
    assert cb._guard.streak == 0 and cb._guard.total == m.rollout_buffer.n_fork_rows > 0


# ── a full pass ──────────────────────────────────────────────────────────────────────────────
def test_a_full_pass_injects_rows_masks_the_fork_steps_and_publishes_the_meters(monkeypatch,
                                                                                tmp_path):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    buf = _buffer(Fork)
    m = _Model(buf)
    calls = _stub_branches(monkeypatch, tmp_path, n_rows=2)
    cb = _cb(m, tmp_path)
    cb._on_rollout_start()
    cb._on_rollout_end()

    assert calls["n"] == 1
    assert calls["crn"] == "dice_and_draws"
    assert calls["forks"], "the selector picked nothing on an all-eligible buffer"
    for fk in calls["forks"]:
        assert set(fk["actions"]) == {"top1", "top2", "rand"}
        assert fk["turn"] == 10 and fk["record"].endswith("rec.json")

    n_forks = len(calls["forks"])
    assert buf.n_fork_rows == n_forks * 3 * 2            # forks x branches x rows
    masks = buf._fork_rows["observations"][PG_MASK_KEY].reshape(-1)
    assert int((masks == 0.0).sum()) == n_forks * 3      # one fork step per branch

    m_ = m._fork_metrics
    assert m_ is not None
    assert m_["forks"] == n_forks and m_["branches"] == 3.0 and m_["crn_draws"] == 1.0
    assert m_["injected_rows"] == buf.n_fork_rows
    assert 0.0 < m_["branch_share"] < 1.0
    assert m_["sim_steps_share"] == pytest.approx(n_forks * 3 * 12.0 / (N_STEPS * N_ENVS))
    assert m_["dropped_forks"] == 0.0
    assert not np.isnan(m_["pairwise_acc"]), "outcomes differ across branches, so pairs exist"


def test_the_row_budget_drops_a_fork_WHOLE_and_says_so(monkeypatch, tmp_path, capsys):
    """Never partially: a half-injected fork would put one branch of a sibling pair in the
    objective and not the other, which is the one asymmetry the arm exists to avoid."""
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    buf = _buffer(Fork)
    m = _Model(buf)
    _stub_branches(monkeypatch, tmp_path, n_rows=3)      # 2 forks x 3 x 3 = 18 > the 12-row budget
    cb = _cb(m, tmp_path)
    cb._on_rollout_end()
    assert "dropped WHOLE at the row budget" in capsys.readouterr().out
    assert m._fork_metrics["dropped_forks"] == 1.0
    assert buf.n_fork_rows % (3 * 3) == 0, "a whole number of FORKS survived"


def test_one_fork_per_episode_slice_by_default(monkeypatch, tmp_path):
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    calls = _stub_branches(monkeypatch, tmp_path)
    cb = _cb(m, tmp_path)
    cb._on_rollout_end()
    envs = [fk["row"][1] for fk in calls["forks"]]
    assert len(envs) == len(set(envs)), "one episode per env in this buffer, so one fork per env"


def test_the_row_budget_bounds_the_NEXT_rollouts_ASK_not_only_its_rows(monkeypatch, tmp_path,
                                                                       capsys):
    """🚨 A fork dropped at the budget has ALREADY BEEN PLAYED, so the row cap alone bounds MEMORY
    and not COST. The previous rollout's MEASURED rows-per-fork is what stops the next one paying
    for branches it will throw away."""
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    calls = _stub_branches(monkeypatch, tmp_path, n_rows=3)
    cb = _cb(m, tmp_path)
    cb._on_rollout_end()                                  # pays the drop, once
    first = len(calls["forks"])
    assert m._fork_metrics["dropped_forks"] > 0
    assert cb._rows_per_fork == pytest.approx(9.0)        # 3 branches x 3 rows

    m.rollout_buffer = _buffer(Fork)                      # a fresh rollout, same shape
    cb._on_rollout_end()
    assert len(calls["forks"]) < first, "the second ask was bounded by the measured size"
    assert m._fork_metrics["dropped_forks"] == 0.0
    assert m._fork_metrics["rows_per_fork"] > 0.0


def test_the_metrics_are_cleared_at_every_rollout_start(monkeypatch, tmp_path):
    """A stale `fork/*` family would read as a live measurement of a rollout that did not produce
    it — the discipline every metric family in this tree keeps."""
    Fork = fork_buffer_class(MaskableDictRolloutBuffer)
    m = _Model(_buffer(Fork))
    _stub_branches(monkeypatch, tmp_path)
    cb = _cb(m, tmp_path)
    cb._on_rollout_end()
    assert m._fork_metrics is not None
    cb._on_rollout_start()
    assert m._fork_metrics is None
