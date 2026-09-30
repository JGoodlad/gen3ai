"""A T2 PARITY REFUSAL stops the launcher; it is never restarted into the same refusal.

Observed 2026-09-30 (``~/gen3ai_archive/cutover_prep/fresh3``): the inference service refused the
trainee's slot load with ``VacuousParity`` (a collapsed win-prob critic). It exited ``CRASH`` (1), the
launcher resumed ``final_model_exception.zip`` — the SAME weights — and T2's startup gate refused them
again, three times, until the rapid-crash circuit breaker gave up. The verdict is a deterministic
function of (code, weights, the committed fixture), so a restart can only replay it. Now:

* ``exit_codes.exit_code_for`` maps ``ParityFailure`` (and ``VacuousParity``) to ``FATAL_CONFIG`` (3)
  and T2's ``NonFiniteWeights`` to ``FATAL_NONFINITE`` (4), by NAME along the MRO and cause chain;
* the launcher gives up on either code at once — one child, the code propagated — and surfaces the
  service's exception line.

Every test here fails on revert of the part it names.
"""
from agents.inference.service.spec import NonFiniteWeights, ParityFailure, VacuousParity
from main.exit_codes import TrainExitCode, exit_code_for
from main.launcher.nonfinite_exit_test import _drive
from main.launcher.run import _fatal_config_reason


def _fresh3_chain() -> BaseException:
    """The exception the fresh3 trainer died on, rebuilt with the same shape: the perturbed pass's
    `VacuousParity` raised while handling the real pass's, each FROM a compile-gate vacuity error."""
    class VacuousCompileParityError(RuntimeError):       # stands in for the compile gate's class
        pass
    try:
        try:
            try:
                raise VacuousCompileParityError("value spread 6.11e-06 <= bar 0.0001")
            except VacuousCompileParityError as e:
                raise VacuousParity("eager group=pool0 slot=24 bucket=8 rows=8: VACUOUS") from e
        except VacuousParity:
            try:
                raise VacuousCompileParityError("value spread 9.58e-05 <= bar 0.0001")
            except VacuousCompileParityError as e2:
                raise VacuousParity("... [perturbed seed=20260929 scale=0.05] rows=7: VACUOUS") from e2
    except VacuousParity as top:
        return top
    raise AssertionError("unreachable")


def test_a_parity_refusal_maps_to_FATAL_CONFIG_and_broken_weights_to_FATAL_NONFINITE():
    assert exit_code_for(VacuousParity("x")) == int(TrainExitCode.FATAL_CONFIG) == 3
    assert exit_code_for(ParityFailure("greedy action differs")) == 3
    assert exit_code_for(_fresh3_chain()) == 3
    assert exit_code_for(NonFiniteWeights("nan in slot")) == int(TrainExitCode.FATAL_NONFINITE) == 4
    try:
        try:
            raise NonFiniteWeights("inf")
        except NonFiniteWeights as inner:
            raise RuntimeError("collector.after_update") from inner
    except RuntimeError as outer:
        assert exit_code_for(outer) == 4
    assert exit_code_for(RuntimeError("an ordinary crash")) == 1, "control: everything else restarts"


def test_the_launcher_surfaces_the_services_refusal_line():
    lines = ["Traceback (most recent call last):",
             "agents.inference.service.spec.VacuousParity: eager group=pool0 slot=0 bucket=8: VACUOUS"]
    reason = _fatal_config_reason(int(TrainExitCode.FATAL_CONFIG), lines)
    assert reason and "VacuousParity" in reason[-1], reason
    reason4 = _fatal_config_reason(int(TrainExitCode.FATAL_NONFINITE),
                                   ["agents.inference.service.spec.NonFiniteWeights: slot group 'pool0' load"])
    assert reason4 and "NonFiniteWeights" in reason4[-1], reason4


def test_a_repeated_startup_parity_refusal_is_never_restarted(tmp_path):
    rc = exit_code_for(_fresh3_chain())
    code, spawned, events = _drive(tmp_path, rc)
    assert (code, spawned) == (3, 1), f"the launcher re-spawned into the same refusal ({code}, {spawned})"
    assert any("Fatal config error — will NOT restart" in e for e in events), events


def test_non_finite_weights_are_never_restarted(tmp_path):
    code, spawned, _ = _drive(tmp_path, exit_code_for(NonFiniteWeights("nan")))
    assert (code, spawned) == (4, 1)
