"""CLI gates for the counterfactual label-plumbing flags (gen3_cf_label_plumbing_v1).

These flags are TRAINING-ONLY — not in `agents.model.flag_registry`, not on `ModelVersion`, not in
`check_compatible` — exactly like `--opd-coef`. Nothing about them is weight-shape relevant, so
there is no version gate to catch a bad combination: **these parser checks ARE the only gate**,
which is why they get a test of their own.

The refusals run `train_rl_agent.py` as a subprocess (~2 s each) because the checks live inside
`main()`, which installs a global `sys.excepthook` and `os._exit` handlers — not something to
invoke in-process from a test runner.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from main.train_rl_agent import build_parser

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_TRAIN = _REPO / "src" / "main" / "train_rl_agent.py"


def _run(*flags):
    proc = subprocess.run(
        [sys.executable, str(_TRAIN), "--steps", "1", *flags],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
    )
    return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------------------
# defaults + parsing (in-process, free)
# --------------------------------------------------------------------------------------
def test_defaults_are_all_off():
    """A command that names none of the flags must be indistinguishable from today: no record tap,
    no label buffer, no loss term."""
    a = build_parser().parse_args(["--steps", "1"])
    assert a.cf_records is False
    assert a.cf_winprob_coef == 0.0
    assert a.cf_label_lag_steps == 150_000
    assert a.cf_records_keep == 512


def test_head_only_defaults_true_and_takes_both_negation_forms():
    """`--cf-head-only` defaults TRUE — the design's safe R1 stage — and the OPT-OUT is what has to
    be typed, in either of the two spellings this parser supports."""
    p = build_parser()
    assert p.parse_args(["--steps", "1"]).cf_head_only is True
    assert p.parse_args(["--no-cf-head-only"]).cf_head_only is False
    assert p.parse_args(["--cf-head-only", "false"]).cf_head_only is False
    assert p.parse_args(["--cf-head-only=false"]).cf_head_only is False
    assert p.parse_args(["--cf-head-only"]).cf_head_only is True


def test_underscore_aliases_resolve():
    p = build_parser()
    a = p.parse_args(["--cf_winprob_coef", "0.25", "--cf_label_lag_steps", "7", "--cf_records"])
    assert (a.cf_winprob_coef, a.cf_label_lag_steps, a.cf_records) == (0.25, 7, True)


# --------------------------------------------------------------------------------------
# the refusals (subprocess)
# --------------------------------------------------------------------------------------
def test_a_live_coef_without_a_win_prob_head_is_refused():
    """`--win-prob-mode none` does not BUILD a WinProbHead, so a live coefficient would fold
    nothing for a whole run. Refuse at the CLI rather than train a silent no-op."""
    rc, out = _run("--cf-winprob-coef", "0.5")
    assert rc != 0
    assert "--cf-winprob-coef > 0 requires --win-prob-mode" in out


def test_a_negative_coef_is_refused():
    rc, out = _run("--cf-winprob-coef", "-1")
    assert rc != 0 and "--cf-winprob-coef must be >= 0" in out


def test_a_negative_lag_bound_is_refused():
    rc, out = _run("--cf-label-lag-steps", "-5")
    assert rc != 0 and "--cf-label-lag-steps must be >= 0" in out


def test_the_record_tap_without_a_bridge_is_refused():
    """The record is a bridge `__RECON__` frame; a websocket run emits none, so `--cf-records`
    there is a silent no-op — the exact shape of bug this repo keeps paying for."""
    rc, out = _run("--cf-records", "--use-bridge", "off")
    assert rc != 0 and "--cf-records requires the in-process bridge" in out


def test_checkargs_accepts_the_whole_family():
    """`python -m main.checkargs` must not report the new flags as stale — it is what an operator
    runs before relaunching a recorded command."""
    proc = subprocess.run(
        [sys.executable, "-m", "main.checkargs", "--argv",
         "--steps 1 --cf-records --cf-records-keep 8 --cf-winprob-coef 0.5 "
         "--no-cf-head-only --cf-label-lag-steps 1000"],
        capture_output=True, text=True, timeout=300, cwd=str(_REPO),
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unrecognized                   : 0" in proc.stdout
