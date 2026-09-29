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

import numpy as np


def _fake_parity_obs(rows=2, width=32, seen=None):
    """Stand-in for `compile_trainer._parity_obs` for the 32-wide `_FakeFE` (the committed REAL-obs
    fixture is the production width, so a control-flow test must supply its own rows)."""
    def _obs(obs_dim, batch, device):
        if seen is not None:
            seen["batch"] = batch
        g = torch.Generator().manual_seed(0)
        return ({"observation": torch.rand(rows, width, generator=g)},
                np.ones((rows, 11), dtype=bool))
    return _obs

from agents.model.compile_trainer import (_MAX_NUMERIC_DRIFT, _TF32_EPS, _TF32_K, CompileTrainerError,
                                          check_numerics, check_shape_stability, check_speedup,
                                          compile_trainer_extractor, fp32_reference, parity_verdict,
                                          resolve_device)
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


def test_fp32_rule_is_UNCHANGED_by_the_precision_aware_gate():
    """gen3_tf32_parity_gate_v1 must not weaken the default path: at 'highest' the rule is the same
    1e-4 on max|compiled - eager|, and a reference — even a wildly wrong one — is never consulted
    (so it cannot LOOSEN the fp32 bar)."""
    assert _MAX_NUMERIC_DRIFT == 1e-4
    check_numerics(9e-5, precision="highest")
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        check_numerics(1e-3, precision="highest", eager_err=1.0)   # eager_err is IGNORED at fp32
    eager = (torch.zeros(4, 8), torch.zeros(4, 3))
    bogus_ref = (torch.full((4, 8), 5.0), torch.full((4, 3), 5.0))
    ok = parity_verdict(eager=eager, compiled=(eager[0] + 5e-5, eager[1]), reference=bogus_ref,
                        precision="highest")
    assert "fp32" in ok and "< 0.0001" in ok
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        parity_verdict(eager=eager, compiled=(eager[0] + 2e-4, eager[1]), reference=bogus_ref,
                       precision="highest")


def test_tf32_rule_is_relative_to_what_eager_tf32_itself_pays():
    """At 'high' both arms are measured against an fp32 reference; compiled passes iff
    e_comp <= K*e_eager + EPS. The T32 launch's 7.62e-03 compiled-vs-eager is the kind of number
    this admits; a disagreement well past K x eager's own rounding is refused."""
    ref = (torch.zeros(4, 8), torch.zeros(4, 3))
    eager = (ref[0] + 3e-3, ref[1])                       # eager TF32 rounding: e_eager = 3e-3
    ok = parity_verdict(eager=eager, compiled=(ref[0] - 5e-3, ref[1]), reference=ref,
                        precision="high")                 # |c-e| = 8e-3 > 1e-4, but e_comp 5e-3 ok
    assert "'high'" in ok and "TF32" in ok and "e_eager" in ok and "e_comp" in ok
    bar = _TF32_K * 3e-3 + _TF32_EPS
    with pytest.raises(CompileTrainerError, match="DISAGREES") as e:
        parity_verdict(eager=eager, compiled=(ref[0], ref[1] + 1.5 * bar), reference=ref,
                       precision="high")
    assert "fp32" in str(e.value) and "wrong kernel" in str(e.value)


def test_tf32_rule_is_never_looser_than_fp32_when_eager_is_exact():
    """e_eager == 0 (nothing TF32 touched) collapses the bar to EPS == the fp32 tolerance."""
    assert _TF32_EPS == _MAX_NUMERIC_DRIFT
    check_numerics(9e-5, precision="high", eager_err=0.0)
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        check_numerics(2e-4, precision="high", eager_err=0.0)


def test_tf32_rule_refuses_without_a_reference_and_on_nan():
    """No fp32 reference => no claim; NaN anywhere => FAIL, never a pass."""
    t = (torch.zeros(2, 2), torch.zeros(2, 1))
    with pytest.raises(CompileTrainerError, match="no fp32 reference"):
        parity_verdict(eager=t, compiled=t, reference=None, precision="high")
    for bad_eager in (None, float("nan"), float("inf"), -1.0):
        with pytest.raises(CompileTrainerError):
            check_numerics(0.0, precision="high", eager_err=bad_eager)
    with pytest.raises(CompileTrainerError):
        check_numerics(float("nan"), precision="high", eager_err=1e-3)


@pytest.fixture
def _restore_matmul_precision():
    prev = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(prev)


def test_fp32_reference_runs_at_highest_and_restores_the_callers_precision(
        _restore_matmul_precision):
    """The reference forward must see 'highest', and the trainer's precision must come back —
    INCLUDING when the forward raises (else a TF32 argv would silently train at fp32, or vice versa)."""
    torch.set_float32_matmul_precision("high")
    seen = []

    def fn(obs):
        seen.append(torch.get_float32_matmul_precision())
        return obs["observation"], obs["observation"]

    fp32_reference(fn, {"observation": torch.ones(2, 3)})
    assert seen == ["highest"] and torch.get_float32_matmul_precision() == "high"

    def boom(obs):
        raise RuntimeError("forward exploded")

    with pytest.raises(RuntimeError):
        fp32_reference(boom, {"observation": torch.ones(2, 3)})
    assert torch.get_float32_matmul_precision() == "high"


def test_a_broken_compiled_graph_still_fails_the_full_gate_at_tf32(monkeypatch,
                                                                   _restore_matmul_precision):
    """The WIRING, through `compile_trainer_extractor` itself, at precision 'high': a compiled
    callable that DROPS a term (here: the bias) must be refused and uninstalled, and a correct one
    must pass with the rule printed. CPU stand-in for the CUDA cell below (TF32 is CUDA-only, so
    on CPU e_eager == 0 and the bar is EPS — the case above)."""
    torch.set_float32_matmul_precision("high")

    def _setup(compiled_factory):
        m = _model("cpu")
        fe = m.policy.features_extractor
        with torch.no_grad():
            fe.lin.bias.fill_(0.5)                     # a term that REACHES the output at zero obs
        monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                            lambda f: torch.device("cuda"))
        monkeypatch.setattr("agents.model.compile_trainer._parity_obs", _fake_parity_obs())
        times = iter([20.0, 10.0])                     # eager then compiled: a real 2x
        monkeypatch.setattr("agents.model.compile_trainer._time_steps",
                            lambda *a, **k: next(times))
        monkeypatch.setattr("agents.model.compile_trainer.torch.compile",
                            lambda f, **k: compiled_factory(fe))
        return m, fe, fe.forward

    def _dropped_bias(fe):
        def fwd(obs):
            h = obs["observation"] @ fe.lin.weight.T      # the bias term is DROPPED
            return h, h
        return fwd

    m, fe, orig = _setup(_dropped_bias)
    with pytest.raises(CompileTrainerError, match="DISAGREES") as e:
        compile_trainer_extractor(m, True, batch=2)
    assert "'high'" in str(e.value) and "e_eager" in str(e.value)
    assert _is_original(fe, orig), "a rejected compile must be uninstalled"

    lines = []
    m, fe, orig = _setup(lambda fe: fe.forward)          # a CORRECT "compile"
    assert compile_trainer_extractor(m, True, batch=2, emit=lines.append) == 2.0
    assert any("parity PASS" in ln and "'high'" in ln and "e_comp" in ln for ln in lines), lines

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
    monkeypatch.setattr("agents.model.compile_trainer._parity_obs", _fake_parity_obs())
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
    monkeypatch.setattr("agents.model.compile_trainer._parity_obs", _fake_parity_obs())
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
    monkeypatch.setattr("agents.model.compile_trainer._parity_obs", _fake_parity_obs(seen=seen))
    monkeypatch.setattr("agents.model.compile_trainer._time_steps", lambda *a, **k: 10.0)
    monkeypatch.setattr("agents.model.compile_trainer.torch.compile", lambda f, **k: f)
    with pytest.raises(CompileTrainerError):        # 1.00x -> refused; we only want the shape
        compile_trainer_extractor(m, True)
    assert seen["batch"] == 64, (
        f"validated at {seen['batch']}; must be the small fixed batch, not model.batch_size "
        "(that combination is what broke a live launch)")


def test_an_explicit_batch_still_wins():
    """The CUDA property test passes batch=8 deliberately; the caller must stay in control."""
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

    It runs on the committed REAL-obs fixture (`compile_parity_fixture`), not zeros: on 2026-09-28
    the all-zero probe read 4.8e-7 while real rows were off by 7.65 (the single-graph CUDA
    miscompile `team_transformer`'s gen3_inductor_trunk_split_v1 note describes).
    """
    from agents.model.extractor_compiles_test import _build_production_extractor
    fe, layout = _build_production_extractor()
    fe = fe.cuda().eval()
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    compiled = torch.compile(fe.forward)

    from agents.model.compile_parity_fixture import load_parity_rows
    rows, _ = load_parity_rows(layout["total_dim"])       # REAL rows — zeros hid a 7.65 miscompile
    worst = {}
    for batch in (1, 48, 512):          # inference, rollout, and a train-shaped minibatch
        idx = np.arange(batch) % len(rows)
        obs = {"observation": torch.as_tensor(rows[idx], device="cuda")}
        with torch.no_grad():
            e_pi, e_vf = fe(obs)
            c_pi, c_vf = compiled(obs)
        torch.cuda.synchronize()
        worst[batch] = max(float((e_pi - c_pi).abs().max()), float((e_vf - c_vf).abs().max()))

    bad = {b: d for b, d in worst.items() if not (d < 1e-4)}
    assert not bad, (
        f"a compiled PRODUCTION shape disagrees with eager: {bad} (all shapes: {worst}). A faster "
        "wrong model is not a win — this is the silent-corruption case, not a performance nit.")


@_skip_cuda
def test_tf32_gate_passes_the_real_extractor_and_refuses_a_broken_graph(monkeypatch,
                                                                        _restore_matmul_precision):
    """gen3_tf32_parity_gate_v1 on the REAL production extractor at `--matmul-precision high`.

    The case that motivated the rule: ai_v14_04_lbat_t32 died at startup on a compiled-vs-eager
    max|delta| of 7.62e-03 under TF32 — TF32 rounding in two different kernel sets, not a wrong
    graph. So (1) the full gate must PASS a correct compile at TF32, and (2) a genuinely wrong
    compiled graph — the same extractor with one term dropped — must still FAIL at TF32. SKIPS
    (naming why) when the GPU is hidden or another process holds it; a skip is NOT a pass.
    """
    import copy

    from agents.model.extractor_compiles_test import _build_production_extractor
    fe, layout = _build_production_extractor()
    fe = fe.cuda()
    fe.obs_dim = layout["total_dim"]
    fe_again = copy.deepcopy(fe)                 # an identical learner for the broken-graph arm
    torch.set_float32_matmul_precision("high")

    def _as_model(extractor):
        m = torch.nn.Module()
        m.policy = torch.nn.Module()
        m.policy.features_extractor = extractor
        return m

    torch._dynamo.reset()
    lines = []
    assert compile_trainer_extractor(_as_model(fe), True, batch=8, emit=lines.append) is not None
    assert any("parity PASS" in ln and "'high'" in ln for ln in lines), lines
    assert torch.get_float32_matmul_precision() == "high", "the fp32 reference must restore it"

    # A WRONG graph over the SAME parameters (so its gradient lands where eager's does, and the
    # refusal is about the numbers, not about a detached copy): the pi projection's bias DROPPED.
    import torch.nn.functional as F

    def _dropped_bias_forward(obs):
        pi_c, vf_c = type(fe_again).forward_internal(fe_again, obs)
        pi = F.linear(fe_again.pre_proj_norm(pi_c), fe_again.projection.weight)   # no bias
        vf = fe_again.value_projection(fe_again.value_pre_norm(vf_c))
        return fe_again.activation(pi), fe_again.activation(vf)

    torch._dynamo.reset()
    real_compile = torch.compile
    monkeypatch.setattr("agents.model.compile_trainer.torch.compile",
                        lambda f, **k: real_compile(_dropped_bias_forward))
    with pytest.raises(CompileTrainerError, match="DISAGREES") as e:
        compile_trainer_extractor(_as_model(fe_again), True, batch=8)
    assert "features" in str(e.value), "caught at the forward, not only by the gradient"


# ------------------------------------------------ gen3_compile_parity_real_obs_v1: the fixture + verdicts


def test_the_committed_real_obs_fixture_matches_the_live_layout():
    """A layout change makes the fixture STALE, and a stale fixture must fail HERE (the routine
    gate), not at a launch. Regenerate with `python -m agents.model.compile_parity_fixture --write`."""
    from agents.model.compile_parity_fixture import N_ROWS, load_parity_rows
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    width = Gen3ObservationEncoder(load_mappings()).get_layout()["total_dim"]
    rows, mask = load_parity_rows(width)
    assert rows.shape == (N_ROWS, width) and rows.dtype == np.float32
    assert mask.shape[0] == N_ROWS and mask.any(axis=1).all()
    assert (rows != 0).any(axis=1).all(), "every fixture row must be a REAL state, never a zero row"


def test_a_stale_or_missing_fixture_REFUSES_rather_than_falling_back(monkeypatch, tmp_path):
    """The fallback (zeros) is exactly the probe that hid the miscompile — so there is none."""
    from agents.model import compile_parity_fixture as cpf
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    width = Gen3ObservationEncoder(load_mappings()).get_layout()["total_dim"]
    with pytest.raises(cpf.ParityFixtureError, match="STALE"):
        cpf.load_parity_rows(width + 1)
    monkeypatch.setattr(cpf, "FIXTURE_PATH", tmp_path / "absent.npz")
    with pytest.raises(cpf.ParityFixtureError, match="missing"):
        cpf.load_parity_rows(width)


def test_decision_level_bars_catch_a_value_or_policy_shift_that_features_alone_miss():
    """The gate reads what the rollout and PPO read: masked legal log-probs and V, each at its
    own fp32 bar."""
    from agents.model.compile_trainer import _FP32_TOL, decision_verdicts
    # INFORMATIVE rows (gen3_fresh_parity_probe_v1): a quantity constant across rows / within
    # every row's legal entries is REFUSED as vacuous — `parity_probe_test` pins that.
    g = torch.Generator().manual_seed(0)
    lp = torch.zeros(4, 11)
    lp[:, :3] = torch.log_softmax(torch.randn(4, 3, generator=g), dim=-1)
    base = {"features": torch.randn(4, 8, generator=g), "legal_logprob": lp,
            "value": torch.linspace(0.2, 0.8, 4)}
    ok = decision_verdicts(eager=base, compiled={k: v.clone() for k, v in base.items()},
                           precision="highest")
    assert len(ok) == 3
    bad_v = {k: v.clone() for k, v in base.items()}
    bad_v["value"] += 2 * _FP32_TOL["value"]
    with pytest.raises(CompileTrainerError, match="value"):
        decision_verdicts(eager=base, compiled=bad_v, precision="highest")
    bad_p = {k: v.clone() for k, v in base.items()}
    bad_p["legal_logprob"][0, 3] -= 2 * _FP32_TOL["legal_logprob"]
    with pytest.raises(CompileTrainerError, match="legal_logprob"):
        decision_verdicts(eager=base, compiled=bad_p, precision="highest")
    with pytest.raises(CompileTrainerError, match="no fp32 reference"):
        decision_verdicts(eager=base, compiled=base, precision="high")


def test_train_graph_gradient_cosine_is_gated():
    """The single-graph miscompile's gradient had cosine 0.778 to eager's — PPO stepping the wrong
    way. The train verdict must refuse that at fp32, and apply the relative rule at TF32."""
    from agents.model.compile_trainer import _MIN_GRAD_COSINE, train_verdict
    g = torch.randn(1000, generator=torch.Generator().manual_seed(0))
    f = torch.randn(2, 4, generator=torch.Generator().manual_seed(2))   # informative rows
    assert "grad cosine" in train_verdict(eager={"features": f, "grad": g},
                                          compiled={"features": f, "grad": g.clone()},
                                          precision="highest")
    skew = g + 0.6 * torch.randn(1000, generator=torch.Generator().manual_seed(1))
    with pytest.raises(CompileTrainerError, match="gradient DISAGREES"):
        train_verdict(eager={"features": f, "grad": g}, compiled={"features": f, "grad": skew},
                      precision="highest")
    assert _MIN_GRAD_COSINE == 0.9999
    ref = {"features": f, "grad": g}
    tf32_eager = {"features": f + 1e-3, "grad": g + 1e-3 * torch.randn(1000)}
    with pytest.raises(CompileTrainerError, match="gradient DISAGREES"):
        train_verdict(eager=tf32_eager, compiled={"features": f + 1e-3, "grad": skew},
                      reference=ref, precision="high")
    assert "1-cos" in train_verdict(eager=tf32_eager, compiled=tf32_eager, reference=ref,
                                    precision="high")


@_skip_cuda
def test_REVERTING_the_trunk_split_FAILS_the_real_obs_gate(monkeypatch):
    """gen3_inductor_trunk_split_v1 — THE revert-must-fail pin. The single CUDA Inductor graph of
    the production extractor miscompiles on REAL observations (measured 2026-09-28: pi_features
    off by up to 7.65, argmax agreement 70.9%, gradient cosine 0.778 on ai_v14_01_base). With the
    split the real-obs gate passes; with the split OFF it must refuse — on torch 2.5.1. On a torch
    in `_SPLIT_NOT_NEEDED_ON` (Lane K1: 2.8.0+cu126) the split defaults OFF and the unsplit gate
    must PASS. SKIPS (naming why) when the
    GPU is hidden or busy — a skip is NOT a pass."""
    import agents.model.team_transformer as tt
    from agents.model.extractor_compiles_test import _build_production_extractor

    def _run(split: bool):
        monkeypatch.setattr(tt, "_CUDA_TRUNK_SPLIT", split)
        torch._dynamo.reset()
        fe, layout = _build_production_extractor()
        fe = fe.cuda()
        fe.obs_dim = layout["total_dim"]
        m = torch.nn.Module()
        m.policy = torch.nn.Module()
        m.policy.features_extractor = fe
        return compile_trainer_extractor(m, True)

    if torch.__version__ in tt._SPLIT_NOT_NEEDED_ON:
        # Lane K1: on this torch the UNSPLIT graph is correct — the split defaults OFF, and the
        # real-obs gate must PASS without it (the verification that licensed turning it off).
        assert tt._CUDA_TRUNK_SPLIT is False
        assert _run(False) is not None
        return
    assert tt._CUDA_TRUNK_SPLIT is True               # every other torch (2.5.1 included): ON
    assert _run(True) is not None
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        _run(False)

