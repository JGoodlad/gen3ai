"""`--canary-shift-max` (T37, `gen3_canary_shift_max_v1`, config v158): the flag's surface, its resolution through a fresh
launch, a same-run resume, a launcher restart and a fork, and the invariant that makes it safe — it moves ONLY the compile
gate's / canary's judgement, never training.

WHY IT EXISTS. The X26 headroom watch (`designs/endstate/design_x26_readiness.md` §4.2) raises the canary's margin
constant `compile_regions.R1_LOGPROB_SHIFT_MAX` (0.18) when the compiled bf16 shift passes it. The launcher REFUSES to move
a live run's code pin at a restart (readiness F28), so a code change cannot reach X26; this flag carries the same raise
through a SAME-RUN RESUME instead. The judgement half (the rule lines, the canary records, the margin) is
`agents/model/canary_shift_max_gate_test.py`; here is the plumbing.

Every assertion below fails on the revert it names.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from main.train.derived_toggle_resume_test import _resolve, _saved_version
from main.train.parser import build_parser

_FRESH = ["--arch", "production", "--steps", "1000"]


# ── the parser: closed, raise-only ──────────────────────────────────────────────────────────────
def test_the_default_is_None_and_a_typed_number_parses():
    assert build_parser().parse_args([]).canary_shift_max is None
    assert build_parser().parse_args(["--canary-shift-max", "0.5"]).canary_shift_max == 0.5
    assert build_parser().parse_args(["--canary_shift_max", "0.18"]).canary_shift_max == 0.18   # the default: allowed


@pytest.mark.parametrize("bad", ["0.179", "0", "-1", "nan", "inf", "abc", ""])
def test_a_value_below_the_constant_or_not_a_finite_number_is_REFUSED_typed(bad, capsys):
    """THE RAISE-ONLY RULE: a lower bar has no measured basis (0.18 is the largest healthy shift measured). The refusal is
    an argparse error that names the flag and the reason, not a silently clamped value."""
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--canary-shift-max", bad])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--canary-shift-max" in err and ("BELOW the code's R1_LOGPROB_SHIFT_MAX" in err or "not a number" in err
                                            or "not finite" in err), err


def test_the_parser_accepts_no_abbreviation_of_it():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--canary-shift", "0.5"])


def test_the_launcher_forwards_the_flag_verbatim_and_claims_none_of_it():
    """The launcher owns NO trainer flag (`compile_flag_forwarding_test`'s contract): a same-run restart that re-passes the
    argv must reach the child with the flag and its value intact."""
    from main.launcher.checkpoint import _strip_launcher_args, resume_child_args
    from main.launcher.run import build_launcher_parser

    argv = ["--restart-interval-hours", "6", "--steps", "1000", "--canary-shift-max", "0.5", "--device", "cuda"]
    assert _strip_launcher_args(argv) == ["--steps", "1000", "--canary-shift-max", "0.5", "--device", "cuda"]
    known, unknown = build_launcher_parser().parse_known_args(["--canary-shift-max", "0.5", "--steps", "1000"])
    assert unknown[:2] == ["--canary-shift-max", "0.5"] and not hasattr(known, "canary_shift_max")
    child, _ = resume_child_args(["--steps", "1000", "--canary-shift-max", "0.5"], "/x/ck.zip", "/x")
    assert child[child.index("--canary-shift-max") + 1] == "0.5"


# ── resolution: a fresh launch, a same-run resume, a launcher restart, a fork ───────────────────
def _run(tmp_path, version):
    from main.train.derived_toggle_resume_test import _run_dir
    from main.train.recipe_testkit import recorded_cli_args
    return _run_dir(tmp_path, version, cli_args=recorded_cli_args())


def test_a_FRESH_launch_resolves_to_None_and_records_the_typed_value(capsys):
    plain = _resolve(list(_FRESH))
    assert plain.canary_shift_max is None
    assert _saved_version(plain).canary_shift_max is None
    assert "[CANARY SHIFT MAX]" not in capsys.readouterr().out          # nothing to say about the default
    typed = _resolve(_FRESH + ["--canary-shift-max", "0.3"])
    assert typed.canary_shift_max == 0.3
    assert _saved_version(typed).canary_shift_max == 0.3                # model_config.json records it
    assert "[CANARY SHIFT MAX] 0.3" in capsys.readouterr().out          # and the launch says so


def test_a_same_run_RESUME_may_RAISE_it_and_the_typed_value_WINS_and_is_recorded(tmp_path):
    """THE FEATURE: X26 launched with nothing typed; at a planned restart the operator resumes the SAME run with the raised
    value. The typed value reaches the child's namespace AND the ModelVersion its next save records."""
    fresh = _resolve(list(_FRESH))
    run, ckpt = _run(tmp_path, _saved_version(fresh))
    resumed = _resolve(["--model", ckpt, "--run-dir", run, "--steps", "2000", "--canary-shift-max", "0.4"])
    assert resumed.canary_shift_max == 0.4
    assert _saved_version(resumed).canary_shift_max == 0.4
    # …and a typed value beats a RECORDED one (a second raise at a later restart)
    run2, ckpt2 = _run(tmp_path / "again", dataclasses.replace(_saved_version(resumed)))
    raised = _resolve(["--model", ckpt2, "--run-dir", run2, "--steps", "2000", "--canary-shift-max", "0.9"])
    assert raised.canary_shift_max == 0.9


def test_an_UNTYPED_restart_and_a_flagless_resume_INHERIT_the_recorded_value(tmp_path):
    """Once typed, the raise must SURVIVE: the launcher's 6-hourly restart re-passes its argv (which may be the pre-raise
    one) and a crash restart re-launches the same, so the recorded value is the only thing that keeps the bar raised."""
    from main.launcher.checkpoint import resume_child_args
    fresh = _resolve(list(_FRESH))
    run, ckpt = _run(tmp_path, dataclasses.replace(_saved_version(fresh), canary_shift_max=0.4))
    restart_argv, stripped = resume_child_args(list(_FRESH), ckpt, run)
    assert stripped and "--canary-shift-max" not in restart_argv                 # the pre-raise argv
    restarted = _resolve(restart_argv)
    assert restarted.canary_shift_max == 0.4
    assert _saved_version(restarted).canary_shift_max == 0.4                       # a second restart keeps it too
    flagless = _resolve(["--model", ckpt, "--run-dir", run, "--steps", "2000"])
    assert flagless.canary_shift_max == 0.4


def test_a_FORK_inherits_the_parents_value(tmp_path):
    fresh = _resolve(list(_FRESH))
    parent_run, ckpt = _run(tmp_path, dataclasses.replace(_saved_version(fresh), canary_shift_max=0.4))
    fork = _resolve(["--model", ckpt, "--run-name", "child_fork", "--steps", "2000"])
    assert fork.canary_shift_max == 0.4
    assert str(parent_run) not in str(getattr(fork, "run_dir", "") or "")            # a different run: the PARENT's record
    # a fork may also RAISE it, and the typed value wins there too
    assert _resolve(["--model", ckpt, "--run-name", "child_fork2", "--steps", "2000",
                     "--canary-shift-max", "1.5"]).canary_shift_max == 1.5


def test_an_INHERITED_value_below_the_constant_is_refused_too(tmp_path, capsys):
    """The code's constant may rise after a run recorded its value (a deliberate, measured change). A recorded value
    below it must not become a lower bar by inheritance: refused at the launch, naming the flag."""
    fresh = _resolve(list(_FRESH))
    _run_dir, ckpt = _run(tmp_path, dataclasses.replace(_saved_version(fresh), canary_shift_max=0.1))
    with pytest.raises(SystemExit) as e:
        _resolve(["--model", ckpt, "--run-name", "child", "--steps", "2000"])
    assert e.value.code == 2
    assert "BELOW the code's R1_LOGPROB_SHIFT_MAX" in capsys.readouterr().err


def test_checkargs_reports_the_inherited_value(tmp_path):
    """`python -m main.checkargs` reads the SAME inheritance rule, so a would-be launch shows what the gate will judge by."""
    from main.checkargs import check
    from main.checkargs_test import _parent_run
    ckpt, _ = _parent_run(tmp_path, canary_shift_max=0.4)
    res = check(["--model", ckpt, "--steps", "1000", "--run-name", "child"])
    assert res["resolution"]["inherited"]["canary_shift_max"] == 0.4
    typed = check(["--model", ckpt, "--steps", "1000", "--run-name", "child", "--canary-shift-max", "0.9"])
    assert "canary_shift_max" not in typed["resolution"]["inherited"]


# ── a typed value needs the gate it moves (only a TYPED one) ─────────────────────────────────────
def test_a_TYPED_value_on_a_launch_with_no_compiled_cuda_learner_is_refused_but_an_inherited_one_is_not():
    from main.train.combination_checks import failing_checks
    name = "canary_shift_max_needs_the_compiled_cuda_learner"
    a = build_parser().parse_args(["--steps", "100", "--canary-shift-max", "0.4", "--device", "cpu"])
    a._explicit_flags = frozenset(d for d, v in vars(a).items() if v is not None)
    assert name in [c.name for c in failing_checks(a)]
    inherited = build_parser().parse_args(["--steps", "100", "--device", "cpu"])
    inherited._explicit_flags = frozenset(d for d, v in vars(inherited).items() if v is not None)
    inherited.canary_shift_max = 0.4                       # what the sweep fills from the parent: not TYPED
    assert name not in [c.name for c in failing_checks(inherited)]
    ok = build_parser().parse_args(["--steps", "100", "--canary-shift-max", "0.4", "--device", "cuda"])
    ok._explicit_flags = frozenset(d for d, v in vars(ok).items() if v is not None)
    assert name not in [c.name for c in failing_checks(ok)]


# ── recorded at both save sites ─────────────────────────────────────────────────────────────────
def test_the_value_is_recorded_by_both_launch_save_sites_and_reaches_the_model():
    import main.train.model_build as mb

    src = inspect.getsource(mb)
    assert src.count("canary_shift_max=args.canary_shift_max") == 2           # the resume build and the fresh build
    assert "model._canary_shift_max = getattr(args, \"canary_shift_max\", None)" in inspect.getsource(
        mb.apply_training_hparams)


def test_the_model_attribute_is_never_pickled():
    """A checkpoint's pickle must not carry a process's margin input into one that did not ask (`_learner_cudagraphs`'s rule):
    the run's record of it is model_config.json."""
    from agents.training import learner_golden as LG
    assert "_canary_shift_max" in LG.build_learner()._excluded_save_params()


# ── THE INVARIANT: it affects ONLY the gate's judgement, never training ─────────────────────────
def _fingerprint(value):
    from agents.training import learner_golden as LG
    from main.train.production_args import production_args
    args = production_args()
    args.canary_shift_max = value
    model = LG.build_learner(args=args)
    assert model._canary_shift_max == value       # PRECONDITION (fails if apply_training_hparams stops handing it on)
    return LG.compute(model)


def test_one_update_is_BIT_IDENTICAL_with_the_flag_at_two_values():
    """Same seeds, same rows, one `train()`: the post-update weights (every parameter's bytes), the per-group hashes and
    every pinned loss / KL / clip scalar are identical with the margin input at the code's default, at a RAISED value
    and at an absurd one. A change that fed the flag into a loss, a bar, the optimizer or a seed would move these."""
    base = _fingerprint(None)
    for v in (0.72, 25.0):
        assert _fingerprint(v) == base, f"--canary-shift-max {v} moved the update"


def test_the_fingerprint_has_teeth_a_real_hyperparameter_change_moves_it():
    """The control for the test above: the SAME harness distinguishes a different learning-rate-class change, so
    'identical' is a finding, not a blind spot."""
    from agents.training import learner_golden as LG
    from main.train.production_args import production_args
    base = _fingerprint(None)
    args = production_args()
    model = LG.build_learner(args=args)
    model.ent_coef = float(model.ent_coef) * 2.0
    assert LG.compute(model)["post_params_sha256"] != base["post_params_sha256"]
