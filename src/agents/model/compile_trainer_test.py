"""Tests for `--compile-trainer` — the GPU-compiled LEARNER (`agents.model.compile_trainer`).

Every rule this flag enforces guards a failure that would otherwise be SILENT. That is the whole
reason the flag is fail-loud: a fallback to eager costs ~1.75x forever and the run keeps training
correctly, so nothing in any metric would surface it. "We did not actually compile" must therefore
be unreachable quietly, and these tests are what make that true.

The verdicts are PURE FUNCTIONS (`check_speedup`, `check_numerics`, `resolve_device`) precisely so
they can be tested on any box. A contract that can only be exercised on a machine with a free GPU is
a contract that gets exercised rarely — which is how the CUDA cells in `extractor_compiles_test.py`
end up skipped on this box most of the time, and why they say so loudly when they do.

One CUDA test covers the property that would corrupt a RUN rather than merely slow it: a compiled
callable leaking into the saved `state_dict`. It skips (naming the reason) when the GPU is hidden or
busy.
"""
import pytest
import torch

# Captured BEFORE any monkeypatch swaps torch.zeros out, so a patched test can still make a tensor.
_real_zeros = torch.zeros

from agents.model.compile_trainer import (CompileTrainerError, check_numerics, check_shape_stability,
                                          check_speedup, compile_trainer_extractor, resolve_device)
from agents.model.extractor_compiles_test import _cuda_skip_reason

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None,
                                reason=_cuda_skip_reason() or "")


class _FakeFE(torch.nn.Module):
    """Stand-in with the two things the helper reads: parameters (for the device) and a width.

    Deliberately not the real extractor — these are control-flow tests, and building the real one
    would make them slow enough that nobody runs them.
    """

    def __init__(self, device="cpu", obs_dim=32):
        super().__init__()
        self.lin = torch.nn.Linear(obs_dim, 8)
        self.obs_dim = obs_dim
        self.to(device)

    def forward(self, obs):
        h = self.lin(obs["observation"])
        return h, h


def _is_original(fe, orig) -> bool:
    """Is `fe.forward` still the module's own method?

    NOT `fe.forward is orig`: attribute access on a method descriptor builds a NEW bound-method
    object every time, so `fe.forward is fe.forward` is already False and that assertion would fail
    even for a completely untouched module. Compare the underlying function instead — a compiled
    callable has no `__func__`, so this distinguishes exactly the case we care about.
    """
    return getattr(fe.forward, "__func__", None) is orig.__func__


def _model(device="cpu", obs_dim=32):
    m = torch.nn.Module()
    m.policy = torch.nn.Module()
    m.policy.features_extractor = _FakeFE(device, obs_dim)
    return m


# --------------------------------------------------------------------------- the no-op


def test_disabled_is_a_true_noop():
    """OFF must not touch the model at all. An off run has to be byte-identical, and returning
    before anything is even read is the cheapest way to guarantee that."""
    m = _model()
    fe = m.policy.features_extractor
    before = fe.forward
    assert compile_trainer_extractor(m, False) is None
    assert _is_original(fe, before)
    assert "forward" not in vars(fe), "OFF must not install an instance attribute at all"


# --------------------------------------------------------------------------- pure verdicts


def test_a_compile_that_is_not_faster_is_refused():
    """Parity means the graph fragmented or the backend fell back per-frame. The measured figure for
    this arch is ~1.75x, so ~1.00x is a defect, not an acceptable outcome."""
    with pytest.raises(CompileTrainerError) as e:
        check_speedup(100.0, 100.0)
    assert "NOT faster" in str(e.value)
    assert "graph break" in str(e.value), "the error must say what to go LOOK at"


def test_a_marginal_compile_is_refused_but_a_real_one_passes():
    with pytest.raises(CompileTrainerError):
        check_speedup(100.0, 99.0)              # 1.01x — under the 1.05x tripwire
    assert check_speedup(155.1, 88.5) == pytest.approx(1.7525, abs=1e-3)   # the measured figure


def test_a_zero_or_negative_time_cannot_pass_as_infinite_speedup():
    """A degenerate timing must FAIL rather than divide its way to a pass."""
    with pytest.raises(CompileTrainerError):
        check_speedup(100.0, 0.0)


def test_numerics_drift_is_refused():
    """A faster wrong model is not a win."""
    check_numerics(0.0)
    check_numerics(9e-5)                        # under tolerance: cuBLAS reduction-order noise
    with pytest.raises(CompileTrainerError) as e:
        check_numerics(1e-3)
    assert "DISAGREES" in str(e.value)


def test_nan_drift_is_refused():
    """`not (nan < tol)` is True, so NaN must fail — written that way on purpose; `nan > tol` would
    be False and a NaN would sail through as 'fine'."""
    with pytest.raises(CompileTrainerError):
        check_numerics(float("nan"))


# --------------------------------------------------------------------------- the refusals


def test_cpu_is_refused_with_the_measured_reason():
    """CPU is rejected up front rather than attempted.

    Not conservatism: `extractor_compiles_test.test_cpu_backward_still_does_not_compile` pins that
    the CPU backward genuinely does not lower (Inductor's C++ backend asserts on the damage op's
    atomic_add scatter). Failing at startup with that reason beats a backend traceback ten minutes
    into a run."""
    with pytest.raises(CompileTrainerError) as e:
        compile_trainer_extractor(_model("cpu"), True)
    msg = str(e.value)
    assert "requires CUDA" in msg
    assert "atomic_add" in msg, "the refusal must NAME the measured reason, not just say no"
    assert "--compile-opponents" in msg, "and must point at the flag that DOES work on CPU"


def test_missing_extractor_is_refused():
    m = torch.nn.Module()
    m.policy = torch.nn.Module()
    with pytest.raises(CompileTrainerError, match="no `features_extractor`"):
        compile_trainer_extractor(m, True)


def test_unknown_obs_width_is_refused_rather_than_guessed(monkeypatch):
    """If the width cannot be found the compile cannot be VALIDATED, and an unvalidated compile is
    exactly the silent-regression case this flag exists to prevent."""
    m = _model("cpu")
    del m.policy.features_extractor.obs_dim
    monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                        lambda fe: torch.device("cuda"))     # past the device gate, not around it
    with pytest.raises(CompileTrainerError) as e:
        compile_trainer_extractor(m, True)
    assert "could not determine" in str(e.value)


def test_a_failing_compile_is_fatal_and_leaves_the_model_untouched(monkeypatch):
    """THE contract, and the asymmetry with the opponent path.

    `maybe_compile_extractor` warns and falls back to eager; here that would be an invisible ~1.75x
    regression — the run trains correctly and just produces ~38% fewer steps/hour, forever. So this
    raises, AND it must put `fe.forward` back exactly as it found it.
    """
    m = _model("cpu")
    fe = m.policy.features_extractor
    orig = fe.forward
    monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                        lambda f: torch.device("cuda"))
    monkeypatch.setattr("agents.model.compile_trainer._time_steps", lambda *a, **k: 10.0)
    monkeypatch.setattr("agents.model.compile_trainer.torch.zeros",
                        lambda *a, **k: _real_zeros(2, 32))      # no real cuda alloc
    monkeypatch.setattr("agents.model.compile_trainer.torch.compile",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend exploded")))
    with pytest.raises(CompileTrainerError) as e:
        compile_trainer_extractor(m, True, batch=2)
    assert _is_original(fe, orig), "a failed compile must leave the model exactly as it was"
    msg = str(e.value)
    assert "FAILED to compile" in msg and "backend exploded" in msg
    assert "bisect" in msg, "the error should point at the ONE-op precedent, not just report failure"


def test_a_rejected_compile_is_uninstalled_not_left_running(monkeypatch):
    """A compile that passes `torch.compile` but fails a VERDICT must also be reverted — otherwise
    the process would keep running a callable we just declared unacceptable."""
    m = _model("cpu")
    fe = m.policy.features_extractor
    orig = fe.forward
    monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                        lambda f: torch.device("cuda"))
    monkeypatch.setattr("agents.model.compile_trainer.torch.zeros",
                        lambda *a, **k: _real_zeros(2, 32))
    monkeypatch.setattr("agents.model.compile_trainer.torch.compile", lambda f, **k: f)
    monkeypatch.setattr("agents.model.compile_trainer._time_steps",
                        lambda *a, **k: 10.0)                # identical arms => 1.00x, rejected
    with pytest.raises(CompileTrainerError, match="NOT faster"):
        compile_trainer_extractor(m, True, batch=2)
    assert _is_original(fe, orig)


def test_resolve_device_reads_the_models_real_device():
    assert resolve_device(_FakeFE("cpu")).type == "cpu"


def test_validation_uses_a_small_batch_NOT_the_models_batch_size(monkeypatch):
    """A REGRESSION TEST, and the regression was mine.

    For one afternoon this validated at `model.batch_size`, reasoning that measuring at the shape
    training uses is more honest. It broke startup: at batch 4096 the real trainer dies with
    `CUDA error: invalid configuration argument` inside the compiled graph — while the same arch
    compiles fine at 4096 in isolation, so it is an interaction with the live trainer process. A
    running gen-10 refused to relaunch.

    The validation answers "did the compile WORK", which a small batch answers just as well. The
    honesty problem was never the batch — it was printing a batch-64 ratio as if it were the
    production figure — and that is fixed by NAMING the shape (asserted below).
    """
    seen = {}
    m = _model("cpu")
    m.batch_size = 4096
    monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                        lambda f: torch.device("cuda"))
    monkeypatch.setattr("agents.model.compile_trainer.torch.zeros",
                        lambda *a, **k: seen.setdefault("shape", a) and None or _real_zeros(2, 32))
    monkeypatch.setattr("agents.model.compile_trainer._time_steps", lambda *a, **k: 10.0)
    monkeypatch.setattr("agents.model.compile_trainer.torch.compile", lambda f, **k: f)
    with pytest.raises(CompileTrainerError):        # 1.00x -> refused; we only want the shape
        compile_trainer_extractor(m, True)
    assert seen["shape"][0] == 64, (
        f"validated at {seen['shape'][0]}; must be the small fixed batch, not model.batch_size "
        "(that combination is what broke a live launch)")


def test_an_explicit_batch_still_wins():
    """The CUDA property test passes batch=8 deliberately; the caller must stay in control."""
    seen = {}
    m = _model("cpu")
    assert compile_trainer_extractor(m, False, batch=8) is None      # off short-circuits


# --------------------------------------------------------------- shape stability (the real hazard)


def _stable(**kw):
    base = dict(n_steps=2048, n_envs=48, batch_size=4096, async_rollout=False)   # gen-10's config
    base.update(kw)
    return base


def test_the_production_config_is_accepted():
    """gen-10: 2048*48 = 98304 = 24 x 4096 exactly. Two shapes total, well under cache_size_limit."""
    check_shape_stability(**_stable())


def test_async_rollout_is_refused():
    """The async collector forwards whichever envs are READY, so the batch VARIES by construction —
    an unbounded shape set, which exhausts dynamo's cache and drops to eager SILENTLY."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(async_rollout=True))
    msg = str(e.value)
    assert "--async-rollout" in msg and "SILENTLY" in msg
    assert "+14%" in msg and "+62%" in msg, "the error should let you pick, with the measured numbers"


def test_a_remainder_minibatch_is_refused_with_a_concrete_suggestion():
    """A remainder is a THIRD shape replayed every epoch, for no benefit. The error must not leave
    the reader doing arithmetic — it names a batch size that actually divides."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(batch_size=5000))
    msg = str(e.value)
    assert "remainder" in msg
    import re
    m = re.search(r"e\.g\. (\d+)", msg)
    assert m, f"no concrete suggestion in: {msg}"
    suggested = int(m.group(1))
    assert (2048 * 48) % suggested == 0, f"suggested {suggested} does not divide the rollout"
    assert suggested <= 5000


def test_the_suggestion_is_the_LARGEST_divisor_that_fits():
    """Suggesting 1 would be technically correct and useless."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(batch_size=5000))
    import re
    assert int(re.search(r"e\.g\. (\d+)", str(e.value)).group(1)) == 4096


def test_a_zero_batch_size_does_not_divide_by_zero():
    check_shape_stability(**_stable(batch_size=0))     # unknown -> not our call to police


# --------------------------------------------------------------------------- the CUDA property


@_skip_cuda
def test_compiled_learner_leaves_the_state_dict_and_a_save_reload_intact():
    """The one failure that would corrupt a RUN rather than slow it.

    `torch.compile(module)` returns an `OptimizedModule` and prefixes every `state_dict` key with
    `_orig_mod.`. `save_model_snapshot` -> `model.save()` writes `policy.state_dict()`, so those keys
    would land in every checkpoint of the run and nothing else could load them. Patching the BOUND
    `fe.forward` avoids that — this asserts it on the real extractor, on the real device.
    """
    import io

    from agents.model.extractor_compiles_test import _build_production_extractor
    fe, layout = _build_production_extractor()
    fe = fe.cuda()
    fe.obs_dim = layout["total_dim"]
    before = set(fe.state_dict().keys())

    m = torch.nn.Module()
    m.policy = torch.nn.Module()
    m.policy.features_extractor = fe

    speedup = compile_trainer_extractor(m, True, batch=8)
    assert speedup is not None and speedup > 1.0

    after = set(fe.state_dict().keys())
    assert after == before, (
        "compiling changed the state_dict keys — a checkpoint written now would be unloadable. "
        f"added={sorted(after - before)[:5]} removed={sorted(before - after)[:5]}")
    assert not any(k.startswith("_orig_mod.") for k in after)

    buf = io.BytesIO()
    torch.save(fe.state_dict(), buf)
    buf.seek(0)
    assert set(torch.load(buf, map_location="cuda", weights_only=True).keys()) == before

@_skip_cuda
def test_every_production_shape_agrees_with_eager():
    """THE correctness gate, and the gap that made it necessary.

    `torch.compile` compiles LAZILY PER SHAPE. The startup validation can only afford a small batch
    (validating at the train batch needs more GPU memory than training does — it runs eager AND
    compiled in one process, plus Inductor's workspace), so the graphs production actually trains
    with are compiled at shapes the startup check never touches:

        batch n_envs     the rollout forward
        batch batch_size the train step

    If a shape-specific kernel were wrong, nothing at startup would see it and the run would train on
    quietly corrupt features — the exact GIGO this flag exists to prevent. So the per-shape agreement
    is asserted HERE, where a GPU is free and memory is not contended.

    Measured against the live gen-10 config on REAL observations off the rust bridge (a separate
    one-off, since a bridge battle does not belong in the unit suite): batch 48 -> 9.5e-07,
    batch 64 -> 7.2e-07, batch 4096 -> 3.6e-06, against a value scale of 2.111. Float32 rounding,
    not a wrong kernel.
    """
    from agents.model.extractor_compiles_test import _build_production_extractor
    fe, layout = _build_production_extractor()
    fe = fe.cuda().eval()
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    compiled = torch.compile(fe.forward)

    worst = {}
    for batch in (1, 48, 512):          # inference, rollout, and a train-shaped minibatch
        obs = {"observation": torch.zeros(batch, layout["total_dim"], device="cuda")}
        with torch.no_grad():
            e_pi, e_vf = fe(obs)
            c_pi, c_vf = compiled(obs)
        torch.cuda.synchronize()
        worst[batch] = max(float((e_pi - c_pi).abs().max()), float((e_vf - c_vf).abs().max()))

    bad = {b: d for b, d in worst.items() if not (d < 1e-4)}
    assert not bad, (
        f"a compiled PRODUCTION shape disagrees with eager: {bad} (all shapes: {worst}). A faster "
        "wrong model is not a win — this is the silent-corruption case, not a performance nit.")
