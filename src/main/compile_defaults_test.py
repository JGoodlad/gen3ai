"""The three `torch.compile` flags DEFAULT ON — and the defaults themselves are the contract.

Owner decision (2026-08-17): the compile flags exist as FALLBACKS, not as opt-ins. A run should
compile unless someone has a reason it must not. That inverts three defaults, and an inverted
default is exactly the kind of change that is invisible once landed — nothing fails, the run just
quietly takes (or stops taking) a 1.75x/6.5x path. So each default is pinned here by VALUE, beside
its opt-out.

The one asymmetry, and the reason this file exists rather than one `assert default is True`:

    `--compile-trainer` REFUSES a non-cuda device by design (the CPU backward provably does not
    lower — Inductor's C++ backend asserts on the damage op's atomic_add scatter). A flat `True`
    default would therefore turn every CPU invocation that works today — the `--debug` smoke, a
    laptop, CI — into a `FATAL_CONFIG` exit. So its default is AUTO: conditioned on the resolved
    device, never softening the refusal an explicit flag still gets.

`--debug` is excluded outright, even with an explicit `--device cuda`: a smoke exists to prove the
pipeline in about a minute, and a multi-minute Inductor compile defeats it.
"""

import pytest

from main.train_rl_agent import (build_parser, resolve_compile_opponents_preload,
                                 resolve_compile_trainer_auto, resolve_compile_trainer_default)

_DIVIDES = dict(n_steps=2048, n_envs=48, batch_size=4096)      # 98304 = 24 x 4096, exactly


def _args(argv):
    return build_parser().parse_args(list(argv))


# --------------------------------------------------------------------------- the parser defaults

def test_compile_opponents_defaults_on():
    assert _args([]).compile_opponents is True


def test_compile_opponents_opt_out_is_no_compile_opponents():
    assert _args(["--no-compile-opponents"]).compile_opponents is False


def test_compile_opponents_also_takes_an_explicit_value():
    """BoolFlag's value form — the convention every other tri-state toggle here uses."""
    assert _args(["--compile-opponents", "false"]).compile_opponents is False
    assert _args(["--compile-opponents=false"]).compile_opponents is False
    assert _args(["--compile-opponents"]).compile_opponents is True


def test_compile_opponents_strict_stays_opt_in():
    """Default-ON is about the compile, NOT about the failure mode. Falling back to eager on a
    failed compile IS the fallback the default wants; --compile-opponents-strict stays something
    you ask for."""
    assert _args([]).compile_opponents_strict is False
    assert _args(["--compile-opponents-strict"]).compile_opponents_strict is True


def test_preload_parses_as_unset_by_default():
    """None, not False — the parser records "nobody said", and `resolve_compile_opponents_preload`
    turns that into "follow --compile-opponents". Keeping the two apart is what lets an EXPLICIT
    preload alongside --no-compile-opponents still be an error."""
    assert _args([]).compile_opponents_preload is None
    assert _args(["--compile-opponents-preload"]).compile_opponents_preload is True
    assert _args(["--no-compile-opponents-preload"]).compile_opponents_preload is False


def test_compile_trainer_parses_as_unset_by_default():
    assert _args([]).compile_trainer is None
    assert _args(["--compile-trainer"]).compile_trainer is True
    assert _args(["--no-compile-trainer"]).compile_trainer is False


# ------------------------------------------------------------------ preload follows its opponent

def test_preload_follows_compile_opponents_when_unset():
    assert resolve_compile_opponents_preload(None, True) is True
    assert resolve_compile_opponents_preload(None, False) is False


def test_no_compile_opponents_alone_is_not_a_usage_error():
    """THE regression this resolution exists for. `--no-compile-opponents` is the documented
    fallback; if the preload defaulted to a flat True, that single flag would hit the
    "preload requires opponents" error and the fallback would be unusable."""
    assert resolve_compile_opponents_preload(None, False) is False


def test_explicit_preload_without_opponents_still_errors():
    with pytest.raises(ValueError, match="requires --compile-opponents"):
        resolve_compile_opponents_preload(True, False)


def test_explicit_preload_off_survives_an_on_opponent_compile():
    """The narrow middle case: keep the per-worker compile, drop the forkserver graph sharing."""
    assert resolve_compile_opponents_preload(False, True) is False


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
        "refusal in compile_trainer_extractor — the default is conditioned, the contract is not")


# ------------------------------------- the auto default YIELDS to a config that cannot take it

def _auto(**over):
    kw = dict(device="cuda", debug=False, async_rollout=False, **_DIVIDES)
    kw.update(over)
    return resolve_compile_trainer_auto(**kw)


def test_auto_is_on_for_a_healthy_cuda_config():
    assert _auto() == (True, None)


def test_auto_yields_to_async_rollout_instead_of_refusing_to_launch():
    """THE hazard a device-only default would have shipped. `check_shape_stability` REFUSES
    --async-rollout outright, so a flat device-conditioned default would turn every working
    `--async-rollout --device cuda` command into a FATAL_CONFIG exit at startup."""
    enabled, why = _auto(async_rollout=True)
    assert enabled is False
    assert why and "async-rollout" in why


def test_auto_yields_to_a_rollout_that_does_not_divide_the_batch():
    """The other refusal, and the more common one — an arbitrary --batch-size is not required to
    divide n_steps*n_envs, and no run today is broken by that."""
    enabled, why = _auto(batch_size=5000)
    assert enabled is False
    assert why and "remainder" in why


def test_the_yield_is_never_silent():
    """A default that quietly declined would be indistinguishable from one that applied — the
    caller emits this reason at startup, so the string has to be present and explanatory."""
    _, why = _auto(async_rollout=True)
    assert why and len(why) > 40 and "--compile-trainer" in why


def test_a_cpu_config_short_circuits_before_the_shape_check():
    """Device first: an unstable-shape CPU config is off for the device reason, and must not
    report a shape reason it never got to (the two would be confusing to read together)."""
    assert _auto(device="cpu", async_rollout=True) == (False, None)


def test_an_explicit_flag_never_reaches_the_auto_path():
    """The refusal is preserved where it belongs. An explicit --compile-trainer parses to True and
    `_maybe_compile_trainer` calls `check_shape_stability` directly, so an impossible ask still
    exits FATAL_CONFIG — the yielding above applies to the DEFAULT only."""
    assert _args(["--compile-trainer", "--async-rollout"]).compile_trainer is True


# ---------------------------------------------------------------- these stay RUNTIME, not version

def test_compile_flags_are_not_architecture_flags():
    """Runtime perf knobs: never recorded in model_config.json, never version-gated. Default-ON
    changes WHEN they apply (a flagless resume now gets them) but not WHAT they are."""
    from agents.model.model_version import ModelVersion
    fields = set(getattr(ModelVersion, "__dataclass_fields__", {}))
    for name in ("compile_opponents", "compile_trainer", "compile_opponents_preload"):
        assert name not in fields, (
            f"{name} became a recorded ModelVersion field — it is a runtime knob, and recording it "
            "would make a resume with a different value a version mismatch")
