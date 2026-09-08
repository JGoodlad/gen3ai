"""`gen3_teacher_scan_limit_flag_v1` (config v113) — `--teacher-scan-limit`, its record, its resume.

The number was **60, hard-coded in `SearchTeacherCallback.__init__`, reachable by no flag** — while
being both halves of the thing that matters about the search-teacher's selection phase:

* its **COST** — each scanned loss trace is falsify-gated through the re-roll driver at ~3 s, so a
  cycle spent ~100 s scanning at 60. That cost used to be paid ON THE TRAINING STEP, which is the
  defect the same commit fixes by moving selection into a child.
* its **SUPPLY** — the crater pool a cycle's candidates are drawn from. Two runs at 10 and at 200
  are not the same experiment, which is why the value is RECORDED rather than merely forwarded.

What is pinned here is the `capacity_telemetry`/`eval_sentinel_greedy` contract applied to it: the
argparse default is `None` so an unset flag INHERITS, an explicit value always wins, the value
round-trips through `model_config.json`, a pre-v113 config migrates to 60 (a record, not a guess —
nothing could set it), and `check_compatible` never looks at it (a frozen eval/pool/distill opponent
runs no teacher cycle at all).
"""
from __future__ import annotations

import contextlib
import io
import json

import pytest

#: The callback's own default, and the value every pre-flag run therefore ran at. Read from the
#: signature rather than retyped: a second copy of the number is a second place for it to drift.
def _callback_default() -> int:
    import inspect

    from agents.training.teacher.callback import SearchTeacherCallback
    return inspect.signature(SearchTeacherCallback.__init__).parameters["scan_limit"].default


def _resolved(argv, saved=None, monkeypatch=None):
    """The LAUNCH path's own `resolve_config` over `argv`, `saved` standing in for the checkpoint's
    recorded `model_config.json`. Validated by EXECUTING the resolution, not by reading branches."""
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


def _saved(scan_limit):
    from types import SimpleNamespace
    return SimpleNamespace(teacher_scan_limit=scan_limit)


def test_the_flags_default_is_the_value_the_callback_hard_coded():
    """A fresh flagless run keeps the historical scan width exactly."""
    assert _callback_default() == 60
    assert _resolved([]).teacher_scan_limit == 60


def test_an_unset_flag_is_None_at_the_parser_so_it_can_INHERIT():
    """`unset` must be distinguishable from `typed the default` — that is the whole mechanism."""
    from main.train_rl_agent import build_parser
    assert build_parser().parse_args([]).teacher_scan_limit is None


def test_an_explicit_value_wins_on_a_fresh_run():
    assert _resolved(["--teacher-scan-limit", "9"]).teacher_scan_limit == 9


def test_a_flagless_resume_INHERITS_the_recorded_scan_width(monkeypatch):
    """THE regression. The launcher re-invokes the ORIGINAL argv every few hours; a run launched at
    12 whose restart silently reset to 60 would draw its corrections from a five-times-larger crater
    pool, and pay five times the selection latency, with nothing on disk saying the diet changed."""
    args = _resolved([], saved=_saved(12), monkeypatch=monkeypatch)
    assert args.teacher_scan_limit == 12


def test_a_typed_value_beats_the_recorded_one_on_a_resume(monkeypatch):
    args = _resolved(["--teacher-scan-limit", "200"], saved=_saved(12), monkeypatch=monkeypatch)
    assert args.teacher_scan_limit == 200


def test_the_callback_is_CONSTRUCTED_with_the_resolved_value(monkeypatch, tmp_path):
    """The flag is worth nothing if `build_callbacks` does not thread it — and it did not exist to
    be threaded until now, which is exactly how a hard-coded 60 survived."""
    from agents.training.teacher.callback import SearchTeacherCallback

    seen = {}
    real_init = SearchTeacherCallback.__init__

    def _spy(self, run_dir, **kw):
        seen.update(kw)
        real_init(self, run_dir, **kw)

    monkeypatch.setattr(SearchTeacherCallback, "__init__", _spy)
    args = _resolved(["--search-teacher", "--teacher-scan-limit", "7"])
    cb = SearchTeacherCallback(str(tmp_path), freq_steps=1000,
                               scan_limit=args.teacher_scan_limit)
    assert seen["scan_limit"] == 7 and cb.scan_limit == 7


def test_it_round_trips_through_model_config_json(tmp_path):
    """`metadata.json:cli_args` is overwritten by every resuming process, so `model_config.json` is
    the only durable record — and the only place `inherit_saved_flag` can read it back."""
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion

    ver = ModelVersion.from_layout_and_policy_kwargs(
        _LAYOUT, _POLICY_KWARGS, teacher_scan_limit=17)
    assert ver.teacher_scan_limit == 17
    path = tmp_path / "model_config.json"
    path.write_text(ver.to_json())
    assert ModelVersion.from_json_file(str(path)).teacher_scan_limit == 17
    assert json.loads(path.read_text())["config_version"] == MODEL_CONFIG_VERSION


def test_a_pre_v113_config_migrates_to_the_hard_coded_60():
    """Not a guess: no run could set anything else, because nothing could set it."""
    from agents.model.model_version import _migrate_config

    out = _migrate_config({"config_version": 112})
    assert out["teacher_scan_limit"] == _callback_default()
    assert out["config_version"] >= 113


def test_it_is_NOT_gated_by_check_compatible():
    """A frozen eval/pool/distill opponent runs no teacher cycle, so gating it on the scan width
    would be a false rejection that breaks league play."""
    from agents.model.model_version import ModelVersion

    a = ModelVersion.from_layout_and_policy_kwargs(_LAYOUT, _POLICY_KWARGS, teacher_scan_limit=5)
    b = ModelVersion.from_layout_and_policy_kwargs(_LAYOUT, _POLICY_KWARGS, teacher_scan_limit=500)
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
