"""CLI gates for the counterfactual label-plumbing flags (gen3_cf_label_plumbing_v1).

These flags are TRAINING-ONLY — not in `agents.model.flag_registry`, not in `check_compatible`.
Nothing about them is weight-shape relevant, so there is no version gate to catch a bad
combination: **these parser checks ARE the only gate**, which is why they get a test of their own.

Since `gen3_cf_coef_provenance_v1` (config v100) they ARE recorded on `ModelVersion` and inherited
on a flagless resume, so every argparse default here is **`None`** and the OFF value is supplied by
`resolve_config`'s `_resolve`. That is why the default tests below assert TWO things: `None` at
parse time (without which `_resolve` can never fire) and the OFF value after resolution (the
behaviour a fresh run actually gets). Asserting only the second would pass with the defaults back
in argparse and the inheritance silently dead again; the recording half is
`agents/model/cf_coef_provenance_test.py`.

The refusals run `train_rl_agent.py` as a subprocess (~2 s each) because the checks live inside
`main()`, which installs a global `sys.excepthook` and `os._exit` handlers — not something to
invoke in-process from a test runner.
"""
import os
import subprocess
import sys

import pytest

from main.train_rl_agent import build_parser
from utils.paths import repo_root, src_path

pytestmark = pytest.mark.integration

_REPO = repo_root()
_TRAIN = src_path("main", "train_rl_agent.py")


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
def _fresh(*flags):
    """Args as a FRESH run (no `--model`) actually sees them — parsed AND resolved.

    `_resolve` is where the OFF value now comes from, so a test that stops at `parse_args` is
    testing the sentinel rather than the behaviour."""
    from main.train.config import resolve_config

    p = build_parser()
    a = p.parse_args(["--steps", "1", *flags])
    resolve_config(a, p)
    return a


def test_defaults_are_all_off():
    """A command that names none of the flags must be indistinguishable from today: no record tap,
    no label buffer, no loss term."""
    raw = build_parser().parse_args(["--steps", "1"])
    assert (raw.cf_records, raw.cf_winprob_coef, raw.cf_label_lag_steps, raw.cf_records_keep) \
        == (None, None, None, None), "argparse must leave the sentinel for `_resolve` to fill"
    a = _fresh()
    assert a.cf_records is False
    assert a.cf_winprob_coef == 0.0
    assert a.cf_label_lag_steps == 150_000
    assert a.cf_records_keep == 512


def test_head_only_defaults_true_and_takes_both_negation_forms():
    """`--cf-head-only` defaults TRUE — the design's safe R1 stage — and the OPT-OUT is what has to
    be typed, in either of the two spellings this parser supports."""
    p = build_parser()
    assert p.parse_args(["--steps", "1"]).cf_head_only is None   # the `_resolve` sentinel
    assert _fresh().cf_head_only is True
    assert p.parse_args(["--no-cf-head-only"]).cf_head_only is False
    assert p.parse_args(["--cf-head-only", "false"]).cf_head_only is False
    assert p.parse_args(["--cf-head-only=false"]).cf_head_only is False
    assert p.parse_args(["--cf-head-only"]).cf_head_only is True


def test_underscore_aliases_resolve():
    p = build_parser()
    a = p.parse_args(["--cf_winprob_coef", "0.25", "--cf_label_lag_steps", "7", "--cf_records"])
    assert (a.cf_winprob_coef, a.cf_label_lag_steps, a.cf_records) == (0.25, 7, True)


# --------------------------------------------------------------------------------------
# gen3_cf_binomial_likelihood_v1 + gen3_cf_evidential_head_v1
# --------------------------------------------------------------------------------------
def test_the_likelihood_defaults_to_binomial():
    """The DEFAULT is the correct likelihood, not the legacy one.

    `--cf-winprob-coef` has never been live in a production run, so there is no trained behaviour to
    preserve and no reason to default to the flat BCE that treats an R=16 label as one observation.
    'bce' stays as the explicit A/B arm.
    """
    assert build_parser().parse_args(["--steps", "1"]).cf_label_likelihood is None
    assert _fresh().cf_label_likelihood == "binomial"
    assert build_parser().parse_args(["--cf-label-likelihood", "bce"]).cf_label_likelihood == "bce"


def test_an_unknown_likelihood_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--cf-label-likelihood", "poisson"])


def test_evidential_defaults_are_off():
    """The structural head is OFF and its coefficient is 0 — a command that names neither is
    indistinguishable from today, in the state_dict and in the loss."""
    a = _fresh()
    assert a.cf_evidential is False
    assert a.cf_evidential_coef == 0.0
    assert a.cf_evidential_reg == 1e-3


def test_evidential_takes_both_negation_forms_and_the_underscore_aliases():
    p = build_parser()
    assert p.parse_args(["--cf-evidential"]).cf_evidential is True
    assert p.parse_args(["--no-cf-evidential"]).cf_evidential is False
    assert p.parse_args(["--cf-evidential=false"]).cf_evidential is False
    a = p.parse_args(["--cf_evidential", "--cf_evidential_coef", "0.2",
                      "--cf_evidential_reg", "0.01"])
    assert (a.cf_evidential, a.cf_evidential_coef, a.cf_evidential_reg) == (True, 0.2, 0.01)


def test_the_structural_flag_is_in_the_registry_but_its_coefficients_are_not():
    """The scope call, written down where a reader will look for it.

    `--cf-evidential` builds a MODULE from an extractor constructor kwarg — the registry's declared
    scope, and the win_prob_mode / value_dist_mode precedent — so it is a `cli`/`structural` row and
    is version-gated. The two coefficients are loss weights set on the MODEL and never reach the
    extractor: the `--opd-coef` class, deliberately absent.
    """
    from agents.model.flag_registry import BY_NAME
    assert "cf_evidential" in BY_NAME
    for absent in ("cf_evidential_coef", "cf_evidential_reg", "cf_label_likelihood",
                   "cf_winprob_coef", "cf_head_only", "cf_records"):
        assert absent not in BY_NAME, f"{absent} is training-only and must stay out of the registry"


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


def test_an_evidential_coef_without_the_head_is_refused():
    """`--cf-evidential` BUILDS the head, and it is structural (version-gated), so it cannot be
    added mid-run to rescue a live coefficient. A silent no-op here would cost a whole run AND
    FATAL the resume that tried to fix it."""
    rc, out = _run("--cf-evidential-coef", "0.5")
    assert rc != 0
    assert "--cf-evidential-coef > 0 requires --cf-evidential" in out


def test_a_negative_evidential_coef_is_refused():
    rc, out = _run("--cf-evidential-coef", "-1")
    assert rc != 0 and "--cf-evidential-coef must be >= 0" in out


def test_a_negative_evidential_reg_is_refused():
    rc, out = _run("--cf-evidential-reg", "-0.5")
    assert rc != 0 and "--cf-evidential-reg must be >= 0" in out


# --------------------------------------------------------------------------------------
# TWIN HEADS + SHADOW CRITIC (gen3_cf_twin_heads_v1, v99)
# --------------------------------------------------------------------------------------
def test_twin_and_shadow_defaults_are_off():
    a = _fresh()
    assert a.cf_twin_heads is False and a.cf_twin_coef == 0.0
    assert a.cf_shadow_critic is False and a.cf_shadow_coef == 0.0


def test_twin_and_shadow_take_both_negation_forms_and_the_underscore_aliases():
    P = build_parser()
    assert P.parse_args(["--cf-twin-heads"]).cf_twin_heads is True
    assert P.parse_args(["--no-cf-twin-heads"]).cf_twin_heads is False
    assert P.parse_args(["--cf_twin_heads"]).cf_twin_heads is True
    assert P.parse_args(["--cf-shadow-critic"]).cf_shadow_critic is True
    assert P.parse_args(["--no-cf-shadow-critic"]).cf_shadow_critic is False
    assert P.parse_args(["--cf_shadow_coef", "0.25"]).cf_shadow_coef == 0.25


def test_the_two_structural_flags_are_in_the_registry_but_their_coefficients_are_not():
    """Same scope call as `--cf-evidential`: the flags build MODULES from extractor constructor
    kwargs, the coefficients are loss weights set on the MODEL."""
    from agents.model.flag_registry import BY_NAME
    assert "cf_twin_heads" in BY_NAME and "cf_shadow_critic" in BY_NAME
    for absent in ("cf_twin_coef", "cf_shadow_coef"):
        assert absent not in BY_NAME, f"{absent} is training-only and must stay out of the registry"


def test_a_twin_coef_without_the_twin_heads_is_refused():
    """The heads are a state_dict change (v99, version-gated), so they cannot be added mid-run to
    rescue a live coefficient — the mistake would cost a whole run AND FATAL the resume that tried
    to fix it."""
    rc, out = _run("--cf-twin-coef", "0.1", "--win-prob-mode", "read_only")
    assert rc != 0
    assert "--cf-twin-coef > 0 requires --cf-twin-heads" in out


def test_twin_heads_without_a_win_prob_head_is_refused():
    """Heads B and C MIRROR head A's on-policy BCE, and head A is `win_head`. With
    `--win-prob-mode none` there is no head A, so the arm's control arm — the whole point of the
    amendment — would silently not exist."""
    rc, out = _run("--cf-twin-heads", "--win-prob-mode", "none")
    assert rc != 0
    assert "--cf-twin-heads requires --win-prob-mode" in out


def test_a_shadow_coef_without_the_shadow_head_is_refused():
    rc, out = _run("--cf-shadow-coef", "0.1")
    assert rc != 0
    assert "--cf-shadow-coef > 0 requires --cf-shadow-critic" in out


@pytest.mark.parametrize("flag", ["--cf-twin-coef", "--cf-shadow-coef"])
def test_negative_twin_shadow_coefficients_are_refused(flag):
    rc, out = _run(flag, "-1")
    assert rc != 0 and f"{flag} must be >= 0" in out


def test_checkargs_accepts_the_whole_family():
    """`python -m main.checkargs` must not report the new flags as stale — it is what an operator
    runs before relaunching a recorded command.

    The argv must also LAUNCH: since `gen3_combination_checks_complete_v1` (2026-09-06) checkargs
    runs every value-conditional refusal the launch path runs, so this test's original argv —
    `--cf-label-lag-steps 1000` against the default checkpoint interval — was correctly refused by
    the CF duty-cycle FATAL_CONFIG (0.1% vs the 25% floor). The lag is widened to a launchable
    value; the assertion this test exists for is the `unrecognized : 0` line."""
    proc = subprocess.run(
        [sys.executable, "-m", "main.checkargs", "--argv",
         "--steps 1 --cf-records --cf-records-keep 8 --cf-winprob-coef 0.5 "
         "--no-cf-head-only --cf-label-lag-steps 400000 --cf-label-likelihood binomial "
         "--cf-evidential --cf-evidential-coef 0.1 --cf-evidential-reg 0.001 "
         "--win-prob-mode read_only --cf-twin-heads --cf-twin-coef 0.1 "
         "--cf-shadow-critic --cf-shadow-coef 0.5"],
        capture_output=True, text=True, timeout=300, cwd=str(_REPO),
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unrecognized                   : 0" in proc.stdout
