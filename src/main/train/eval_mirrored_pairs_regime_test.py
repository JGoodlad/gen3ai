"""T17 `gen3_mirrored_pairs_v1` — the in-loop eval's PAIRING REGIME is resolved, recorded and inherited
exactly like `eval_sentinel_greedy`, and DEFAULTS OFF (the M5 sizing arms are compared across it)."""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace


def _resolved(argv, saved=None, monkeypatch=None):
    from main.train import config as cfg_mod
    from main.train_rl_agent import build_parser

    parser = build_parser()
    if saved is not None:
        monkeypatch.setattr(cfg_mod, "_load_saved_version", lambda path: saved)
        argv = ["--model", "models/parent/checkpoints/checkpoint_10_steps.zip"] + argv
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        args = parser.parse_args(["--steps", "100", "--allow-nonproduction-arch"] + argv)
        cfg_mod.resolve_config(args, parser)
    return args


def test_a_fresh_run_is_UNMIRRORED_and_unset_is_not_off():
    from main.train_rl_agent import build_parser

    assert build_parser().parse_args([]).eval_mirrored_pairs is None
    args = _resolved([])
    assert args.eval_mirrored_pairs is False and args.eval_mirrored_pairs_source == "default"


def test_the_flag_turns_it_on():
    args = _resolved(["--eval-mirrored-pairs"])
    assert args.eval_mirrored_pairs is True and args.eval_mirrored_pairs_source == "argv"


def test_a_flagless_resume_keeps_the_recorded_regime(monkeypatch):
    saved = SimpleNamespace(eval_mirrored_pairs=True)
    args = _resolved([], saved=saved, monkeypatch=monkeypatch)
    assert args.eval_mirrored_pairs is True and args.eval_mirrored_pairs_source == "inherited"
    args = _resolved(["--no-eval-mirrored-pairs"], saved=saved, monkeypatch=monkeypatch)
    assert args.eval_mirrored_pairs is False and args.eval_mirrored_pairs_source == "argv"


def test_a_pre_v128_config_migrates_UNMIRRORED_and_the_field_is_recorded():
    from agents.model.model_version import migrations
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.fields import ModelVersionFields

    assert MODEL_CONFIG_VERSION >= 128
    assert ModelVersionFields.eval_mirrored_pairs is False
    data = {"config_version": 127}
    out = migrations._migrate_config(data)
    assert out["eval_mirrored_pairs"] is False and out["config_version"] >= 128
