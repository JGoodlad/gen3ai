"""Pins for the trainer's `--env-core` wiring (M5 Lane G, unit 2): python stays the DEFAULT and
untouched; the collector flags resolve to their stated defaults only after the refusal sweep; the
floor roster maps onto Lane F's ported bots; a restart's run seed differs from the first segment's;
every save records the env core."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from main.train.parser import build_parser
from main.train.rust_env_setup import _bot_names, resolve_env_core_args, segment_seed


def _args(*argv):
    return build_parser().parse_args(["--steps", "1", *argv])


def test_rust_is_the_default_and_every_collector_flag_is_untyped():
    a = _args()
    assert a.env_core == "rust"           # the bare-argv flip (deletion pass D2, 2026-10-02)
    for d in ("rollout_trigger", "rollout_target_samples", "rust_env_front", "version_pinning",
              "opponent_sampling", "behaviour_check", "trainee_slots"):
        assert getattr(a, d) is None, d
    resolve_env_core_args(a)
    assert a.behaviour_check == "fatal"        # K9(b) on both cores (the python gate costs one read)


def test_the_rust_core_resolves_to_the_stated_defaults():
    a = _args("--env-core", "rust")
    resolve_env_core_args(a)
    assert (a.rollout_trigger, a.rust_env_front, a.rust_env_profile, a.version_pinning, a.trainee_slots,
            a.behaviour_check, a.rollout_target_samples, a.opponent_sampling) == (
        "complete_game", "proc", "release", "off", 1, "fatal", 0, "keyed")
    b = _args("--env-core", "rust", "--version-pinning", "per_game")
    resolve_env_core_args(b)
    assert b.trainee_slots == 3
    c = _args("--env-core", "rust", "--behaviour-check", "warn", "--trainee-slots", "4")
    resolve_env_core_args(c)
    assert (c.behaviour_check, c.trainee_slots) == ("warn", 4)


def test_the_training_floor_roster_maps_onto_ported_bots():
    from utils.rust_env import bot_inventory as BI

    classes = []
    for row in BI.ROWS:
        if "train" not in row.used_by:
            continue
        mod, _, name = row.cls.rpartition(".")
        classes.append(getattr(importlib.import_module(mod), name))
    names = _bot_names(classes)
    assert names == [r.name for r in BI.ROWS if "train" in r.used_by] and len(names) == 8
    from agents.baitbot import make_baitbot_class

    assert _bot_names([make_baitbot_class(0.6)]) == ["baitbot"]


def test_an_unported_bot_is_refused_by_name():
    class Nope:  # not in the inventory
        pass

    with pytest.raises(RuntimeError, match="no ported Rust bot"):
        _bot_names([Nope])


def test_a_restart_segment_draws_a_different_run_seed():
    assert segment_seed(1001, 0) != segment_seed(1001, 3_000_000)
    assert segment_seed(1001, 0) == segment_seed(1001, 0)
    assert 0 <= segment_seed(7, 5) < 2 ** 47


def test_every_save_records_the_env_core():
    from main.train.run_io import _model_hparams

    class _Opt:
        param_groups = [{"lr": 1e-4, "weight_decay": 0.0}]

    m = SimpleNamespace(policy=SimpleNamespace(optimizer=_Opt()), batch_size=64, n_steps=8,
                        clip_range=lambda _p: 0.15, clip_range_vf=None, n_epochs=5, ent_coef=0.05,
                        vf_coef=0.5, gamma=1.0, gae_lambda=0.8, max_grad_norm=0.5, learning_rate=1e-4)
    out = _model_hparams(m)
    assert out["env_core"] == {"env_core": "rust"}   # no collector stamp: the only core there is
    m._env_core_stamp = {"env_core": "rust", "front": "proc", "summary": "x"}
    assert _model_hparams(m)["env_core"] == {"env_core": "rust", "front": "proc"}


def test_startup_builds_this_checkouts_env_core_before_anything_else(monkeypatch):
    """F-LG-6: the launcher's PINNED worktree is a fresh checkout with no ``src/rust_env/target``, so every
    ``--env-core rust`` launch through ``main.launcher`` died ~10 s in (``rust_env_proc does not exist``).
    ``build_rust_vec_env`` now builds THIS checkout's core, for the declared profile, first."""
    import utils.rust_env.build as B
    from main.train.rust_env_setup import build_rust_vec_env

    class _Built(Exception):
        pass

    seen = []

    def _fake(profile, *, emit=None):
        seen.append(profile)
        raise _Built(profile)

    monkeypatch.setattr(B, "ensure_built", _fake)
    for argv, profile in ((("--env-core", "rust"), "release"),
                          (("--env-core", "rust", "--rust-env-profile", "selfcheck"), "selfcheck")):
        with pytest.raises(_Built):
            build_rust_vec_env(_args(*argv), mappings=None, trainee_teambuilder=None, opponent_teambuilder=None,
                               opponent_classes=[], bot_weights=None, fixed_opponents=[], exploiter_entry=None,
                               snapshot_dir=None, opponent_version=None, self_play_fraction=0.0, n_envs=4,
                               emit=lambda _m: None)
    assert seen == ["release", "selfcheck"]


def test_a_resume_that_switches_env_core_is_announced_never_silent(tmp_path):
    """An UNTYPED `--env-core` on a `--model` launch INHERITS the checkpoint's core (the M5 switch,
    `resolve_env_core_default`); a PYTHON-ERA checkpoint moves onto rust, and `env_core_switch_line` names
    it (the trainer emits it at startup; `--dry-run` prints it)."""
    import json

    from main.train.rust_env_setup import env_core_switch_line, recorded_env_core

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "checkpoint_512_steps.zip"
    ckpt.write_bytes(b"")
    (run / "checkpoints" / "checkpoint_512_steps.json").write_text(json.dumps({"env_core": {"env_core": "rust"}}))
    (run / "metadata.json").write_text(json.dumps({"env_core": {"env_core": "python"}}))
    assert recorded_env_core(str(ckpt)) == "rust"             # the sidecar is the checkpoint's own record
    from main.train.rust_env_setup import resolve_env_core_default
    untyped = _args("--model", str(ckpt))
    assert resolve_env_core_default(untyped) is not None and untyped.env_core == "rust"
    assert env_core_switch_line(untyped) is None              # inherited: no switch
    assert env_core_switch_line(_args("--model", str(ckpt), "--env-core", "rust")) is None
    assert env_core_switch_line(_args()) is None              # a fresh run has nothing to switch from
    (run / "checkpoints" / "checkpoint_512_steps.json").unlink()
    assert recorded_env_core(str(ckpt)) == "python"           # falls back to the run's metadata.json
    py_line = env_core_switch_line(_args("--model", str(ckpt)))
    assert py_line and "CORE SWITCH" in py_line               # a python-era record moves onto rust: announced
    import inspect

    import main.launcher.dry_run as dr
    import main.train.config as cfg
    assert "env_core_switch_line(args)" in inspect.getsource(cfg.resolve_config)
    assert "env_core_switch_line(ns)" in inspect.getsource(dr)
