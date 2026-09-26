"""`--policy-gae-lambda`, `--matmul-precision` and the per-epoch PPO diagnostics.

Three training-side changes, each defaulting to the behaviour every run had before it existed:

* **`--policy-gae-lambda`** (`gen3_policy_gae_lambda_v1`, config v123) — the PPO policy's GAE λ,
  until now HARDCODED to 0.80 at both `model_build` sites. Default 0.80, so the advantages are
  bit-identical; recorded on `ModelVersion` and inherited on a flagless resume.
* **`--matmul-precision {highest,high}`** (`gen3_matmul_precision_v1`) — `highest` is PyTorch's own
  default and calls NOTHING; `high` enables TF32 in the trainer process. Stamped at launch and
  recorded in `metadata.json`.
* **`train/approx_kl_epoch_<k>` · `train/clip_fraction_epoch_<k>`** (`gen3_ppo_per_epoch_diag_v1`) —
  one pair per epoch the update actually ran, folded from the numbers the loop already computes.

Every test here fails if its change is reverted.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch as th

from main.train.parser import build_parser

_DEFAULT = 0.80


def _resolved(argv):
    from main.train.config import resolve_config
    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


# ── --policy-gae-lambda: the flag surface ────────────────────────────────────────────────────
def test_the_argparse_default_is_None_so_the_resolve_line_is_reachable():
    p = build_parser()
    assert p.parse_args([]).policy_gae_lambda is None
    assert p.parse_args(["--policy-gae-lambda", "0.95"]).policy_gae_lambda == 0.95
    assert p.parse_args(["--policy_gae_lambda", "1"]).policy_gae_lambda == 1.0


def test_the_help_distinguishes_it_from_the_critics_win_prob_lambda():
    helps = {a.dest: a.help for a in build_parser()._actions}
    assert "--win-prob-lambda" in helps["policy_gae_lambda"]
    assert "0.80" in helps["policy_gae_lambda"]


def test_a_fresh_run_resolves_to_the_old_hardcoded_value_and_a_typed_value_wins():
    assert _resolved([]).policy_gae_lambda == _DEFAULT
    assert _resolved(["--policy-gae-lambda", "0.95"]).policy_gae_lambda == 0.95


def test_the_range_check_refuses_outside_zero_to_one():
    for bad in ("1.01", "-0.1"):
        with pytest.raises(SystemExit):
            _resolved(["--policy-gae-lambda", bad])


def test_a_flagless_resume_INHERITS_the_recorded_value_and_a_typed_one_overrides_it():
    from agents.model.model_version import ModelVersion
    from main.train.config import inherit_saved_flag
    saved = ModelVersion.__new__(ModelVersion)
    saved.policy_gae_lambda = 0.95
    args = build_parser().parse_args([])
    assert inherit_saved_flag(args, saved, "policy_gae_lambda", _DEFAULT) is True
    assert args.policy_gae_lambda == 0.95
    args = build_parser().parse_args(["--policy-gae-lambda", "0.9"])
    assert inherit_saved_flag(args, saved, "policy_gae_lambda", _DEFAULT) is False
    assert args.policy_gae_lambda == 0.9
    import main.train.config as cfg
    assert '_resolve("policy_gae_lambda", 0.80)' in inspect.getsource(cfg)


def test_it_is_a_RECORDED_ModelVersion_field_and_an_old_config_migrates_to_0_80():
    from agents.model.model_version import ModelVersion
    from agents.model.model_version.construct import ModelVersionConstruction
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config
    assert ModelVersion.__dataclass_fields__["policy_gae_lambda"].default == _DEFAULT
    fn = ModelVersionConstruction.from_layout_and_policy_kwargs
    assert inspect.signature(fn).parameters["policy_gae_lambda"].default == _DEFAULT
    assert "policy_gae_lambda=float(policy_gae_lambda)" in inspect.getsource(fn)
    assert MODEL_CONFIG_VERSION >= 123
    out = _migrate_config({"config_version": 122})
    assert out["policy_gae_lambda"] == _DEFAULT and out["config_version"] >= 123
    kept = _migrate_config({"config_version": 122, "policy_gae_lambda": 0.95})
    assert kept["policy_gae_lambda"] == 0.95


def _gae_lambda_sites(src: str):
    """Every `gae_lambda` a `model_build` statement sets — the ctor kwarg and the resume assign."""
    sites = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.keyword) and node.arg == "gae_lambda":
            sites.append(ast.unparse(node.value))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "gae_lambda":
                    sites.append(ast.unparse(node.value))
    return sites


def test_both_model_build_sites_read_the_flag_and_neither_hardcodes_0_80():
    """The fresh ctor and the resume assignment were the two hardcoded 0.80s. A site that reverts
    to a literal would make the flag silently inert on that path."""
    import main.train.model_build as mb
    src = inspect.getsource(mb)
    sites = _gae_lambda_sites(src)
    assert len(sites) == 2, sites
    for s in sites:
        assert "args.policy_gae_lambda" in s, sites
    # ...and both ModelVersion constructions RECORD it, plus the save-path one in lifecycle.
    assert src.count("policy_gae_lambda=args.policy_gae_lambda") == 2
    import main.train.lifecycle as lc
    assert "policy_gae_lambda=float(model.gae_lambda)" in inspect.getsource(lc)


# ── --policy-gae-lambda: the advantages on a hand-built buffer ───────────────────────────────
_REWARDS = np.array([0.0, 1.0, -0.5, 0.0, 2.0, 0.0], dtype=np.float32)
_VALUES = np.array([0.3, 0.1, -0.2, 0.4, 0.0, 0.5], dtype=np.float32)
_STARTS = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)   # an episode boundary at t=4
_LAST_V, _LAST_DONE, _GAMMA = 0.25, 0.0, 0.99


def _hand_gae(lam: float) -> np.ndarray:
    """GAE written out by hand: δ_t = r_t + γ·V(t+1)·(1-done) − V(t), A_t = δ_t + γλ(1-done)·A_{t+1}."""
    adv = np.zeros_like(_REWARDS, dtype=np.float64)
    last = 0.0
    for t in reversed(range(len(_REWARDS))):
        if t == len(_REWARDS) - 1:
            nonterm, v_next = 1.0 - _LAST_DONE, _LAST_V
        else:
            nonterm, v_next = 1.0 - _STARTS[t + 1], float(_VALUES[t + 1])
        delta = float(_REWARDS[t]) + _GAMMA * v_next * nonterm - float(_VALUES[t])
        last = delta + _GAMMA * lam * nonterm * last
        adv[t] = last
    return adv


def _buffer_advantages(lam: float) -> np.ndarray:
    """The advantages the REAL rollout buffer class computes at this λ over the fixed rollout."""
    from gymnasium import spaces
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    buf = MaskableRolloutBuffer(len(_REWARDS), spaces.Box(-1, 1, (1,), np.float32),
                                spaces.Discrete(2), device="cpu", gamma=_GAMMA, gae_lambda=lam,
                                n_envs=1)
    for t in range(len(_REWARDS)):
        buf.add(np.zeros((1, 1), np.float32), np.zeros((1, 1)), _REWARDS[t:t + 1],
                _STARTS[t:t + 1], th.tensor([_VALUES[t]]), th.zeros(1),
                action_masks=np.ones((1, 2), np.float32))
    buf.compute_returns_and_advantage(last_values=th.tensor([_LAST_V]),
                                      dones=np.array([_LAST_DONE]))
    return buf.advantages[:, 0].copy()


def test_the_DEFAULT_path_gives_advantages_bit_identical_to_the_old_hardcoded_0_80():
    lam = _resolved([]).policy_gae_lambda
    got = _buffer_advantages(lam)
    assert np.array_equal(got, _buffer_advantages(0.80)), "default λ moved the advantages"
    np.testing.assert_allclose(got, _hand_gae(0.80), rtol=1e-6, atol=1e-6)


def test_0_95_changes_the_advantages_exactly_as_GAE_says():
    lam = _resolved(["--policy-gae-lambda", "0.95"]).policy_gae_lambda
    got = _buffer_advantages(lam)
    np.testing.assert_allclose(got, _hand_gae(0.95), rtol=1e-6, atol=1e-6)
    base = _buffer_advantages(0.80)
    moved = ~np.isclose(got, base)
    # A step whose NEXT step is terminal-cut (t=3, t=5) has A = δ at any λ; every other one moves.
    assert moved.tolist() == [True, True, True, False, True, False], (got, base)


# ── --matmul-precision ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def _restore_precision():
    before = th.get_float32_matmul_precision()
    yield
    th.set_float32_matmul_precision(before)


def test_the_precision_flag_parses_refuses_and_defaults_to_highest():
    p = build_parser()
    assert p.parse_args([]).matmul_precision == "highest"
    assert p.parse_args(["--matmul-precision", "high"]).matmul_precision == "high"
    with pytest.raises(SystemExit):
        p.parse_args(["--matmul-precision", "medium"])


def test_highest_calls_nothing_and_stamps_highest(monkeypatch, capsys, _restore_precision):
    import main.train.config as cfg
    calls = []
    monkeypatch.setattr(th, "set_float32_matmul_precision", lambda v: calls.append(v))
    assert cfg.apply_matmul_precision(build_parser().parse_args([])) == "highest"
    assert calls == []
    assert "🧮 [MATMUL PRECISION] highest" in capsys.readouterr().out


def test_high_sets_TF32_in_this_process_and_stamps_it(capsys, _restore_precision):
    import main.train.config as cfg
    th.set_float32_matmul_precision("highest")
    got = cfg.apply_matmul_precision(build_parser().parse_args(["--matmul-precision", "high"]))
    assert got == "high" == th.get_float32_matmul_precision()
    assert "🧮 [MATMUL PRECISION] high" in capsys.readouterr().out


def test_resolve_config_applies_and_stamps_it(capsys, _restore_precision):
    th.set_float32_matmul_precision("highest")
    _resolved(["--matmul-precision", "high"])
    assert th.get_float32_matmul_precision() == "high"
    assert "🧮 [MATMUL PRECISION] high" in capsys.readouterr().out


def test_metadata_records_the_realized_precision(_restore_precision):
    from main.train import run_io
    th.set_float32_matmul_precision("high")
    assert run_io._matmul_precision() == "high"
    assert '"matmul_precision": _matmul_precision()' in inspect.getsource(run_io._model_hparams)


# ── per-epoch PPO diagnostics ────────────────────────────────────────────────────────────────
def _tiny_ppo(n_epochs, target_kl=None):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.instrumented_ppo_test import _CounterDictEnv
    venv = DummyVecEnv([(lambda: _CounterDictEnv()) for _ in range(2)])
    return InstrumentedMaskablePPO("MultiInputPolicy", venv, n_steps=8, batch_size=4,
                                   n_epochs=n_epochs, learning_rate=3e-3, target_kl=target_kl,
                                   device="cpu", seed=0)


def _epoch_rows(logged, family):
    return sorted((k for k in logged if k.startswith(f"train/{family}_epoch_")),
                  key=lambda k: int(k.rsplit("_", 1)[1]))


def test_per_epoch_scalars_appear_once_per_epoch_and_REUSE_the_loop_numbers():
    model = _tiny_ppo(n_epochs=3)
    model.learn(total_timesteps=16)                # one rollout (8 steps × 2 envs), one train()
    logged = model.logger.name_to_value
    kl = _epoch_rows(logged, "approx_kl")
    cf = _epoch_rows(logged, "clip_fraction")
    assert kl == [f"train/approx_kl_epoch_{k}" for k in range(3)]
    assert cf == [f"train/clip_fraction_epoch_{k}" for k in range(3)]
    # The SAME numbers the stock tags fold: stock `approx_kl` is the LAST epoch's mean, and stock
    # `clip_fraction` pools every minibatch — equal-size epochs make that the mean of the epochs.
    assert logged[kl[-1]] == pytest.approx(float(logged["train/approx_kl"]), rel=1e-6)
    assert np.mean([logged[k] for k in cf]) == pytest.approx(
        float(logged["train/clip_fraction"]), rel=1e-6)
    assert logged[kl[0]] < logged[kl[-1]]          # epoch 0 starts ON the rollout policy


def test_a_KL_early_stop_records_only_the_epochs_that_ran():
    model = _tiny_ppo(n_epochs=5, target_kl=1e-12)   # trips on the first minibatch past epoch 0's
    model.learn(total_timesteps=16)
    logged = model.logger.name_to_value
    n = len(_epoch_rows(logged, "approx_kl"))
    assert 1 <= n < 5
    assert len(_epoch_rows(logged, "clip_fraction")) == n
