"""`--compile-trainer` DEFAULTS to AUTO — and the default itself is the contract.

Owner decision (2026-08-17): the compile flags exist as FALLBACKS, not as opt-ins. A run should
compile unless someone has a reason it must not. (Deletion pass U3 / R6 deleted the other half,
`--compile-opponents` and its preload / strict siblings, with the Python env core whose workers
compiled the opponents; the Rust core's opponents forward through the inference service.)

`--compile-trainer` REFUSES a non-cuda device by design (the CPU backward provably does not
lower — Inductor's C++ backend asserts on the damage op's atomic_add scatter). A flat `True`
default would therefore turn every CPU invocation that works today — the `--debug` smoke, a
laptop, CI — into a `FATAL_CONFIG` exit. So its default is AUTO: conditioned on the resolved
device, never softening the refusal an explicit flag still gets.

`--debug` is excluded outright, even with an explicit `--device cuda`: a smoke exists to prove the
pipeline in about a minute, and a multi-minute Inductor compile defeats it.
"""

import pytest

from main.train_rl_agent import (build_parser, resolve_compile_trainer_auto,
                                 resolve_compile_trainer_default)

_DIVIDES = dict(n_steps=2048, n_envs=48, batch_size=4096)      # 98304 = 24 x 4096, exactly


def _args(argv):
    return build_parser().parse_args(list(argv))


# --------------------------------------------------------------------------- the parser defaults

def test_compile_trainer_parses_as_unset_by_default():
    assert _args([]).compile_trainer is None
    assert _args(["--compile-trainer"]).compile_trainer is True
    assert _args(["--no-compile-trainer"]).compile_trainer is False


# --------------------------------------------------- the trainer default is CONDITIONED ON DEVICE

@pytest.mark.parametrize("device", ["cuda", "cuda:0", "CUDA"])
def test_trainer_default_is_on_for_cuda(device):
    assert resolve_compile_trainer_default(device, debug=False) is True


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_trainer_default_is_off_for_a_non_cuda_device(device):
    """The load-bearing half. --compile-trainer RAISES on a non-cuda device, so a default that
    said True here would convert a working CPU command into a FATAL_CONFIG exit."""
    assert resolve_compile_trainer_default(device, debug=False) is False


def test_trainer_default_follows_auto_to_whatever_the_box_has():
    assert resolve_compile_trainer_default("auto", debug=False, cuda_available=lambda: True) is True
    assert resolve_compile_trainer_default("auto", debug=False, cuda_available=lambda: False) is False


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_debug_is_never_touched_by_the_default(device):
    """A --debug smoke stays a minute-long CPU pipeline check on EVERY device spelling — including
    an explicit --device cuda, which would otherwise take a CUDA context and a multi-minute compile
    from whatever production run owns the card."""
    assert resolve_compile_trainer_default(device, debug=True) is False


def test_a_probe_that_raises_resolves_to_off():
    """`torch.cuda.is_available()` can throw on a broken driver. A default must not be the thing
    that crashes a launch, and OFF is the safe side (a missed 1.75x, not a refusal)."""
    def _boom():
        raise RuntimeError("no driver")
    assert resolve_compile_trainer_default("auto", debug=False, cuda_available=_boom) is False


def test_explicit_flag_beats_the_auto_default_in_both_directions():
    """The resolution only ever fills a None — pinned at the parser boundary so a future edit that
    starts overwriting an explicit value fails here."""
    assert _args(["--compile-trainer"]).compile_trainer is True
    assert _args(["--no-compile-trainer"]).compile_trainer is False
    assert _args(["--compile-trainer", "--device", "cpu"]).compile_trainer is True, (
        "an EXPLICIT --compile-trainer on cpu must survive parsing and reach the fail-loud "
        "refusal in preflight_compile_trainer — the default is conditioned, the contract is not")


# ------------------------------------- the auto default YIELDS to a config that cannot take it

def _auto(**over):
    kw = dict(device="cuda", debug=False, **_DIVIDES)
    kw.update(over)
    return resolve_compile_trainer_auto(**kw)


def test_auto_is_on_for_a_healthy_cuda_config():
    assert _auto() == (True, None)


def test_auto_yields_to_a_rollout_that_does_not_divide_the_batch():
    """The other refusal, and the more common one — an arbitrary --batch-size is not required to
    divide n_steps*n_envs, and no run today is broken by that."""
    enabled, why = _auto(batch_size=5000)
    assert enabled is False
    assert why and "remainder" in why


def test_the_yield_is_never_silent():
    """A default that quietly declined would be indistinguishable from one that applied — the
    caller emits this reason at startup, so the string has to be present and explanatory."""
    _, why = _auto(batch_size=5000)
    assert why and len(why) > 40 and "--compile-trainer" in why


def test_a_cpu_config_short_circuits_before_the_shape_check():
    """Device first: an unstable-shape CPU config is off for the device reason, and must not
    report a shape reason it never got to (the two would be confusing to read together)."""
    assert _auto(device="cpu", batch_size=5000) == (False, None)


def test_auto_judges_the_REAL_update_size_not_n_steps_times_n_envs():
    """F9 (P10-E): the update is `--rollout-target-samples` rows (the production recipe sets it
    independently of --n-steps / --n-envs), so the auto default must judge THAT. Before, a valid run
    (target divides, n_steps*n_envs does not) had `--compile-trainer` QUIETLY left off, and an invalid one
    (the product divides, the target does not) turned it on only to die at the first buffer build.
    Fails on revert (the check goes back to n_steps x n_envs)."""
    assert 1000 * 48 % 4096 != 0 and 98_304 % 4096 == 0
    assert _auto(n_steps=1000, n_envs=48, batch_size=4096, rollout_target_samples=98_304) == (True, None)
    enabled, why = _auto(rollout_target_samples=100_000)
    assert enabled is False and why and "100,000 rows" in why, why
    # an untyped / zero target is today's default (n_steps x n_envs)
    assert _auto(rollout_target_samples=None) == (True, None)
    assert _auto(rollout_target_samples=0, batch_size=5000)[0] is False


def test_the_explicit_preflight_judges_the_real_update_size_with_the_models_own_batch(capsys):
    """An EXPLICIT --compile-trainer reaches `check_shape_stability` through `_maybe_compile_trainer`,
    with the same real update size, and with the MODEL's n_steps / batch_size when it has them (a resume
    restores the checkpoint's; the argv's are inert there). The valid config gets PAST the shape check (to
    the next refusal, the stand-in model's missing extractor); the invalid ones stop AT it."""
    from types import SimpleNamespace

    from main.train.lifecycle import _maybe_compile_trainer

    def run(model, **over):
        args = SimpleNamespace(compile_trainer=True, n_steps=1000, n_envs=48, batch_size=4096,
                               rollout_target_samples=98_304)
        for k, v in over.items():
            setattr(args, k, v)
        with pytest.raises(SystemExit):
            _maybe_compile_trainer(model, args)
        return capsys.readouterr().err
    assert "no `features_extractor`" in run(SimpleNamespace(policy=None))        # past the shape check
    err = run(SimpleNamespace(policy=None), n_steps=2048, rollout_target_samples=100_000)
    assert "divides evenly" in err and "100,000 rows" in err
    # the model's own batch size is the one the collector will use
    err = run(SimpleNamespace(policy=None, batch_size=5000), rollout_target_samples=0, n_steps=2048)
    assert "divides evenly" in err and "--batch-size 5000" in err


def test_an_explicit_flag_never_reaches_the_auto_path():
    """The refusal is preserved where it belongs. An explicit --compile-trainer parses to True and
    `_maybe_compile_trainer` calls `check_shape_stability` directly, so an impossible ask still
    exits FATAL_CONFIG — the yielding above applies to the DEFAULT only."""
    assert _args(["--compile-trainer", "--batch-size", "5000"]).compile_trainer is True


# ---------------------------------------------------------------- these stay RUNTIME, not version

def test_compile_flags_are_not_architecture_flags():
    """Runtime perf knobs: never recorded in model_config.json, never version-gated. Default-ON
    changes WHEN they apply (a flagless resume now gets them) but not WHAT they are."""
    from agents.model.model_version import ModelVersion
    fields = set(getattr(ModelVersion, "__dataclass_fields__", {}))
    for name in ("compile_trainer",):
        assert name not in fields, (
            f"{name} became a recorded ModelVersion field — it is a runtime knob, and recording it "
            "would make a resume with a different value a version mismatch")
