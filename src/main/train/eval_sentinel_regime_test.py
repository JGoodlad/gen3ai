"""`gen3_eval_sentinel_greedy_default_v1` — the EVAL OPPONENT REGIME and the gate it derives.

Three things are pinned here, and each one is a defect this repo has already paid for.

1. **The DEFAULT is greedy + symmetric.** `--eval-sentinel-greedy` was ON for 49 runs (v5.5–v8)
   and was dropped, unrecorded, at the v9 launch; every run since measured its pool sentinels
   against a temperature-1.0 opponent drawing from a different team distribution than the trainee,
   worth **+8.9 pp [+7.0, +10.7]** to the trainee on the same frozen pair the dense ladder plays
   symmetrically. The flip restores the v8 convention.

2. **RESUME SAFETY (rule of evidence 15: a windowed statistic never crosses an opponent-regime
   boundary).** The flag's argparse default is `None`, so a FLAGLESS resume INHERITS the regime its
   checkpoint recorded and a v9-era run resumed on this code stays stochastic. The failure this
   forbids is silent by construction: `win_rate_vs_pool` would simply step down ~9 pp mid-run with
   nothing in any metric, log line or sidecar saying the opponent had changed.

3. **The promotion gate FOLLOWS the regime, and the three branches are ORDERED.** Explicit argv
   wins; a regime typed on THIS argv re-derives the gate (inheriting 0.65 into a freshly-greedy run
   would freeze the pool, which is the failure the auto-lowering exists to prevent); otherwise the
   checkpoint's own gate is inherited.
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest

from agents.training.snapshot_pool import (
    EVAL_SENTINEL_GREEDY_DEFAULT, PROMOTE_THRESHOLD_GREEDY, PROMOTE_THRESHOLD_STOCHASTIC,
    promote_threshold_default)


def _resolved(argv, saved=None, monkeypatch=None):
    """Run the LAUNCH path's own `resolve_config` over `argv`, with `saved` standing in for the
    checkpoint's recorded `model_config.json`. Validated by EXECUTING the resolution rather than
    reading the branches — the repo's standing preference, and the reason `checkargs` calls
    `inherit_saved_flag` instead of re-implementing it."""
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


def _saved(greedy, threshold):
    """A stand-in for the parent's `ModelVersion`. `inherit_saved_flag` reads it by `getattr`, so
    the attribute set is the whole contract — the same duck-typing `checkargs` relies on."""
    return SimpleNamespace(eval_sentinel_greedy=greedy, promote_threshold=threshold)


# ---------------------------------------------------------------- 1. the fresh-run default

def test_a_fresh_run_with_no_flag_is_GREEDY():
    args = _resolved([])
    assert args.eval_sentinel_greedy is True
    assert args.eval_sentinel_greedy_source == "default"
    assert args.promote_threshold == PROMOTE_THRESHOLD_GREEDY


def test_the_module_constant_is_the_only_definition_of_the_default():
    """No second copy of the boolean: the parser reads the constant, so a future flip is one edit."""
    from main.train_rl_agent import build_parser
    assert EVAL_SENTINEL_GREEDY_DEFAULT is True
    assert build_parser().parse_args([]).eval_sentinel_greedy is None   # unset ≠ off
    assert promote_threshold_default(True) == PROMOTE_THRESHOLD_GREEDY
    assert promote_threshold_default(False) == PROMOTE_THRESHOLD_STOCHASTIC


def test_the_negation_still_turns_it_off_and_the_gate_follows():
    args = _resolved(["--no-eval-sentinel-greedy"])
    assert args.eval_sentinel_greedy is False
    assert args.eval_sentinel_greedy_source == "argv"
    assert args.promote_threshold == PROMOTE_THRESHOLD_STOCHASTIC


# ---------------------------------------------------------------- 2. resume safety

def test_a_flagless_resume_of_a_STOCHASTIC_run_stays_stochastic(monkeypatch):
    """THE regression this file exists for. Without inheritance the resume would read the new
    greedy default and `win_rate_vs_pool` would step ~9 pp mid-run, silently."""
    args = _resolved([], saved=_saved(False, PROMOTE_THRESHOLD_STOCHASTIC), monkeypatch=monkeypatch)
    assert args.eval_sentinel_greedy is False
    assert args.eval_sentinel_greedy_source == "inherited"
    assert args.promote_threshold == PROMOTE_THRESHOLD_STOCHASTIC
    assert args.promote_threshold_source == "inherited"


def test_a_flagless_resume_of_a_GREEDY_run_stays_greedy(monkeypatch):
    args = _resolved([], saved=_saved(True, PROMOTE_THRESHOLD_GREEDY), monkeypatch=monkeypatch)
    assert args.eval_sentinel_greedy is True
    assert args.eval_sentinel_greedy_source == "inherited"
    assert args.promote_threshold == PROMOTE_THRESHOLD_GREEDY


def test_an_inherited_EXPLICIT_threshold_survives_a_flagless_resume(monkeypatch):
    """A parent that typed `--promote-threshold 0.7` keeps it — the gate is inherited, not
    re-derived, whenever the regime itself was inherited."""
    args = _resolved([], saved=_saved(False, 0.7), monkeypatch=monkeypatch)
    assert args.promote_threshold == pytest.approx(0.7)
    assert args.promote_threshold_source == "inherited"


def test_typing_the_regime_on_a_resume_RE_DERIVES_the_gate(monkeypatch):
    """The operator moved the regime deliberately, so the gate must move with it — inheriting the
    parent's 0.65 into a now-greedy run would freeze the pool."""
    args = _resolved(["--eval-sentinel-greedy"],
                     saved=_saved(False, PROMOTE_THRESHOLD_STOCHASTIC), monkeypatch=monkeypatch)
    assert args.eval_sentinel_greedy is True
    assert args.eval_sentinel_greedy_source == "argv"
    assert args.promote_threshold == PROMOTE_THRESHOLD_GREEDY
    assert args.promote_threshold_source == "default"


def test_an_explicit_threshold_beats_everything(monkeypatch):
    args = _resolved(["--promote-threshold", "0.42", "--eval-sentinel-greedy"],
                     saved=_saved(False, PROMOTE_THRESHOLD_STOCHASTIC), monkeypatch=monkeypatch)
    assert args.promote_threshold == pytest.approx(0.42)
    assert args.promote_threshold_source == "argv"


# ---------------------------------------------------------------- 3. it is RECORDED

@pytest.mark.parametrize("greedy,threshold", [(True, 0.55), (False, 0.65)])
def test_the_regime_round_trips_through_model_config_json(tmp_path, greedy, threshold):
    """`metadata.json:cli_args` is overwritten by every resuming process, so `model_config.json` is
    the only durable record — and the only place `inherit_saved_flag` can read the regime back."""
    import json

    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion

    ver = ModelVersion.from_layout_and_policy_kwargs(
        _LAYOUT, _POLICY_KWARGS, eval_sentinel_greedy=greedy, promote_threshold=threshold)
    assert ver.eval_sentinel_greedy is greedy and ver.promote_threshold == threshold
    path = tmp_path / "model_config.json"
    path.write_text(ver.to_json())
    back = ModelVersion.from_json_file(str(path))
    assert back.eval_sentinel_greedy is greedy and back.promote_threshold == threshold
    assert json.loads(path.read_text())["config_version"] == MODEL_CONFIG_VERSION


def test_a_pre_v112_config_migrates_to_STOCHASTIC_and_its_own_gate():
    """Not a guess: the argparse default was False from the v9 launch until this bump, so every
    config the migration can reach (floor 96) trained stochastic unless its launch typed the flag.
    The gate is then DERIVED by the rule that produced it, not pinned to one number."""
    from agents.model.model_version import _migrate_config

    out = _migrate_config({"config_version": 111})
    assert out["eval_sentinel_greedy"] is False
    assert out["promote_threshold"] == PROMOTE_THRESHOLD_STOCHASTIC

    greedy = _migrate_config({"config_version": 111, "eval_sentinel_greedy": True})
    assert greedy["promote_threshold"] == PROMOTE_THRESHOLD_GREEDY


def test_neither_field_is_GATED_by_check_compatible():
    """A frozen eval/pool/distill opponent runs no eval cycle at all, so gating it on the regime
    would be a false rejection that breaks league play."""
    from agents.model.model_version import ModelVersion

    a = ModelVersion.from_layout_and_policy_kwargs(
        _LAYOUT, _POLICY_KWARGS, eval_sentinel_greedy=True, promote_threshold=0.55)
    b = ModelVersion.from_layout_and_policy_kwargs(
        _LAYOUT, _POLICY_KWARGS, eval_sentinel_greedy=False, promote_threshold=0.65)
    a.check_compatible(b)      # must not raise


@pytest.fixture(scope="module", autouse=True)
def _layout_and_kwargs():
    """Built once — the encoder is the expensive part of this file."""
    global _LAYOUT, _POLICY_KWARGS
    from agents.model.features_extractor import NET_ARCH
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    enc = Gen3ObservationEncoder(load_mappings())
    _LAYOUT = enc.get_layout()
    _POLICY_KWARGS = {"net_arch": NET_ARCH, "features_extractor_kwargs":
                      enc.get_features_extractor_kwargs()}


_LAYOUT: dict = {}
_POLICY_KWARGS: dict = {}


# ---------------------------------------------------------------- 4. the two SURFACES agree

def _parent_tree(tmp_path, greedy, threshold):
    """A minimal fork parent on disk: a run dir with a `model_config.json` and a checkpoint zip
    path `checkargs` can resolve a parent config from."""
    from agents.model.model_version import ModelVersion

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ver = ModelVersion.from_layout_and_policy_kwargs(
        _LAYOUT, _POLICY_KWARGS, eval_sentinel_greedy=greedy, promote_threshold=threshold)
    (run / "model_config.json").write_text(ver.to_json())
    ckpt = run / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.write_bytes(b"")
    return str(ckpt)


@pytest.mark.parametrize("argv,expect", [
    ([], (False, 0.65)),                              # flagless FORK inherits the parent's regime
    (["--eval-sentinel-greedy"], (True, 0.55)),       # typed regime RE-DERIVES the gate
    (["--eval-sentinel-greedy", "--promote-threshold", "0.6"], (True, 0.6)),
])
def test_checkargs_resolves_the_regime_exactly_as_the_launch_does(tmp_path, argv, expect):
    """`main.checkargs` printing an effective config the launch does not produce is the defect
    class this repo names in `combination_checks`' own docstring. `--promote-threshold` is the one
    that would drift: it is NOT a plain inherited flag — a TYPED regime re-derives it — so a blanket
    inheritance sweep would report the parent's 0.65 against a launch that resolves 0.55.
    """
    from main.checkargs import resolve_against_parent

    ckpt = _parent_tree(tmp_path, False, PROMOTE_THRESHOLD_STOCHASTIC)
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        via_checkargs = resolve_against_parent(["--steps", "100", "--model", ckpt] + argv)["ns"]
    launched = _resolved(argv, saved=_ParentStub(ckpt), monkeypatch=_DirectMonkey())

    assert (via_checkargs.eval_sentinel_greedy, via_checkargs.promote_threshold) == expect
    assert (launched.eval_sentinel_greedy, launched.promote_threshold) == expect
    assert via_checkargs.eval_sentinel_greedy_source == launched.eval_sentinel_greedy_source
    assert via_checkargs.promote_threshold_source == launched.promote_threshold_source


class _ParentStub:
    """The saved `ModelVersion` the launch path would load for `_parent_tree`'s checkpoint."""

    def __init__(self, ckpt):
        import json
        import os

        data = json.load(open(os.path.join(os.path.dirname(os.path.dirname(ckpt)),
                                           "model_config.json")))
        self.eval_sentinel_greedy = data["eval_sentinel_greedy"]
        self.promote_threshold = data["promote_threshold"]


class _DirectMonkey:
    """`_resolved`'s tiny monkeypatch surface, usable outside a pytest fixture."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __del__(self):
        for obj, name, old in self._undo:
            setattr(obj, name, old)
