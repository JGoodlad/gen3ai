"""The torch floor (`utils.torch_floor`, deletion pass K1): HEAD refuses torch < 2.8, and the trainer
refuses FIRST. Fails if the refusal or its call in the trainer's `main()` is reverted."""
from __future__ import annotations

import asyncio

import pytest

from utils import torch_floor as TF


def test_the_floor_accepts_2_8_and_later_and_refuses_older_or_unreadable():
    assert TF.refusal("2.8.0+cu126") is None
    assert TF.refusal("2.9.1") is None
    assert TF.refusal("3.0.0") is None
    for old in ("2.5.1+cu121", "2.7.1", "1.13.0"):
        why = TF.refusal(old)
        assert why is not None and old in why and "PINNED" in why, why
    assert TF.refusal("not-a-version") is not None


def test_a_missing_torch_is_refused_never_assumed_new_enough(monkeypatch):
    monkeypatch.setattr(TF, "installed_torch", lambda: None)
    assert TF.refusal() is not None


def test_the_installed_torch_passes():
    assert TF.refusal() is None, "the test interpreter is gen3ai_torch28"


def test_the_TRAINER_exits_FATAL_CONFIG_on_torch_2_5_1_before_anything_else(monkeypatch, capsys):
    """`train_rl_agent.main()` on a 2.5.1 interpreter stops at once with FATAL_CONFIG, naming the
    pinned-resume route — before any argv is parsed or any directory is created."""
    from main.exit_codes import TrainExitCode
    from main import train_rl_agent

    monkeypatch.setattr(TF, "installed_torch", lambda: "2.5.1+cu121")
    monkeypatch.setattr("sys.argv", ["train_rl_agent.py", "--this-flag-does-not-exist"])
    with pytest.raises(SystemExit) as ei:
        asyncio.run(train_rl_agent.main())
    assert ei.value.code == int(TrainExitCode.FATAL_CONFIG)
    err = capsys.readouterr().err
    assert "[TorchFloor] FATAL" in err and "2.5.1+cu121" in err
