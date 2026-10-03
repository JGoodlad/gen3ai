"""P10-D — the compile sentinel's torch check runs WHERE A LEARNER COMPILES, and nowhere else.

The defect (P10 review, confirmed): `train_rl_agent` -> `_declare_compile_cache` ->
`compile_cache.cache_stamp` imported `compile_control` for `config_row_hash`, and that import ran
`verify_torch_internals()`. So EVERY trainer — `--debug` and `--no-compile-trainer` included —
died at startup with an uncaught `CompileSentinelError` on any torch but the recorded
`2.8.0+cu126`, instead of a non-compile run working and a compile run exiting FATAL_CONFIG.

These tests hold the fixed contract:

* the cache stamp never imports the adapter (`compile_config` holds the row and its hash);
* importing the adapter checks nothing — a fresh interpreter on an UNRECORDED torch imports it,
  stamps a cache and reads the T2 service's graph counter, and only `control()` refuses;
* the compile path's two entry points (`_maybe_compile_trainer`'s preflight and
  `_arm_compile_sentinel`) turn a failing check into `FATAL_CONFIG` naming the reason;
* the non-compile path never calls the check.

Each of the first three fails on a revert of the fix.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from agents.model import compile_control as cc
from main.exit_codes import TrainExitCode

_DRIFT = "P10-D test: the recorded torch internals DRIFTED"


def _drifted(version=None):
    raise cc.CompileSentinelError(_DRIFT)


@pytest.fixture
def drifted_torch(monkeypatch):
    """The sentinel's internals check fails, through the symbol the adapter reads at call time."""
    monkeypatch.setattr(cc, "verify_torch_internals", _drifted)
    monkeypatch.setattr(cc, "_CONTROL", None)        # so `control()` CONSTRUCTS (and so checks)


def test_the_cache_stamp_never_imports_the_compile_adapter(monkeypatch):
    """`None` in `sys.modules` makes any import of the adapter raise: the stamp must not need it."""
    from agents.model import compile_cache as CC
    from agents.model.compile_config import config_row_hash
    monkeypatch.setitem(sys.modules, "agents.model.compile_control", None)
    stamp = CC.cache_stamp()
    assert stamp["config_row_sha"] == config_row_hash()


def test_importing_the_adapter_on_an_UNRECORDED_torch_checks_nothing_and_control_refuses():
    """A fresh interpreter whose torch reports a version with no recorded row: the import, the
    cache stamp and the T2 service's graph counter all work; constructing the sentinel refuses."""
    code = textwrap.dedent("""
        import torch
        torch.__version__ = "2.8.0+cu999"
        import agents.model.compile_control as cc
        from agents.model import compile_cache
        from agents.inference.service.service import _dynamo_graphs
        compile_cache.cache_stamp()
        _dynamo_graphs()
        try:
            cc.control()
        except cc.CompileSentinelError as exc:
            print("REFUSED:", str(exc).splitlines()[0])
        else:
            print("NOT REFUSED")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "REFUSED:" in proc.stdout and "does not support torch '2.8.0+cu999'" in proc.stdout, (
        proc.stdout, proc.stderr[-2000:])


def test_the_compile_PREFLIGHT_exits_FATAL_CONFIG_naming_the_drift(drifted_torch, capsys):
    from main.train.lifecycle import _maybe_compile_trainer
    args = SimpleNamespace(compile_trainer=True, n_steps=4, n_envs=1, batch_size=4)
    with pytest.raises(SystemExit) as ei:
        _maybe_compile_trainer(SimpleNamespace(policy=None), args)
    assert ei.value.code == TrainExitCode.FATAL_CONFIG
    assert _DRIFT in capsys.readouterr().err


def test_arming_the_sentinel_exits_FATAL_CONFIG_naming_the_drift(drifted_torch, capsys):
    from main.train.lifecycle import _arm_compile_sentinel
    model = SimpleNamespace(policy=SimpleNamespace(features_extractor=object()),
                            _micro_static=object(), n_envs=1, batch_size=4)
    with pytest.raises(SystemExit) as ei:
        _arm_compile_sentinel(model, SimpleNamespace(compile_trainer=True, n_envs=1))
    assert ei.value.code == TrainExitCode.FATAL_CONFIG
    assert _DRIFT in capsys.readouterr().err
    assert cc._CONTROL is None                        # nothing was installed


def test_the_NON_compile_path_never_runs_the_check(drifted_torch):
    from agents.inference.service.service import _dynamo_graphs
    from agents.model import compile_cache as CC
    from main.train.lifecycle import _arm_compile_sentinel, _maybe_compile_trainer
    args = SimpleNamespace(compile_trainer=False, n_steps=4, n_envs=1, batch_size=4)
    _maybe_compile_trainer(SimpleNamespace(policy=None), args)
    _arm_compile_sentinel(SimpleNamespace(), args)
    CC.cache_stamp()
    _dynamo_graphs()
    assert cc._CONTROL is None
