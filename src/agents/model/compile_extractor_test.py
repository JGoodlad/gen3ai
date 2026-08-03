"""Unit tests for `maybe_compile_extractor` — the `--compile-extractor` runtime perf knob.

It is a PERF knob, so its contract is mostly about what it must NOT do: never crash a run, never
silently claim a win it didn't get, never change the state_dict, never create a CUDA context in a
CPU worker (the 2026-06-30 OOM: compiling a CPU model in a CUDA-visible process took ~252 MiB of
card, ×48 workers).

The measured payoff it exists to deliver (recorded here so a regression has a number to fail
against): B=1 CPU forward 6.37 -> 0.98 ms (6.5x) on the literal production arch, and a
production-scale training A/B at n_envs=48 of 498 -> 653 marginal fps (+31%, disjoint ranges over 6
samples per arm). NOTE: that fps A/B predates the `species_posterior` fix, so it was measured on the
PARTIALLY compiled graph (3.6x per forward, not 6.5x) — the end-to-end number should now be at
least that good.
"""
import types

import pytest
import torch

from agents.model import snapshot as S


class _FE(torch.nn.Module):
    """Minimal stand-in with the attributes the helper touches."""

    def __init__(self, cost_ms=0.0):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.layout = {"total_dim": 8}
        self._debugger = None
        self._cost = cost_ms

    def disable_observation_debugger(self):
        had = self._debugger is not None
        self._debugger = None
        return had

    def forward(self, obs):
        if self._cost:
            import time
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) * 1e3 < self._cost:
                pass
        return self.lin(obs["observation"])


def _model(fe, device="cpu"):
    return types.SimpleNamespace(policy=types.SimpleNamespace(features_extractor=fe), device=device)


def _is_eager(fe):
    """`fe.forward` builds a NEW bound-method object on every access, so `is` never matches. What
    actually distinguishes eager from patched is whether the INSTANCE holds an override at all, and
    whether the callable is still the class's own function."""
    return "forward" not in fe.__dict__ or getattr(fe.forward, "__func__", None) is _FE.forward


def _fast_compile(fe):
    """A stand-in compile that is genuinely faster than the instrumented eager forward."""
    return lambda fn: (lambda obs: fe.lin(obs["observation"]))


def test_disabled_is_a_no_op():
    fe = _FE()
    assert S.maybe_compile_extractor(_model(fe), False) is False
    assert _is_eager(fe)


def test_missing_extractor_is_tolerated():
    """Some frozen opponents are loaded through paths that don't expose a policy — must not raise."""
    assert S.maybe_compile_extractor(types.SimpleNamespace(policy=None), True) is False
    assert S.maybe_compile_extractor(types.SimpleNamespace(), True) is False


def test_reverts_when_compile_is_not_faster(monkeypatch):
    """Compiling can LOSE — the June 2026 attempt measured 0.70x, because dynamo overhead exceeded
    the fusion win on a fragmented graph. Measure, then keep or revert; never assume."""
    fe = _FE()

    def slow_compile(fn):
        def wrapper(obs):
            import time
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) * 1e3 < 1.0:
                pass
            return fn(obs)
        return wrapper

    monkeypatch.setattr(torch, "compile", slow_compile)
    assert S.maybe_compile_extractor(_model(fe), True, label="slow") is False
    assert _is_eager(fe), "a non-winning compile must be REVERTED to the eager forward"


def test_keeps_it_when_compile_is_faster(monkeypatch):
    fe = _FE(cost_ms=2.0)
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    assert S.maybe_compile_extractor(_model(fe), True, label="fast") is True
    assert not _is_eager(fe)


def test_exception_is_swallowed_and_forward_restored(monkeypatch):
    """A perf knob must never take a run down."""
    fe = _FE()

    def boom(fn):
        raise RuntimeError("inductor exploded")

    monkeypatch.setattr(torch, "compile", boom)
    assert S.maybe_compile_extractor(_model(fe), True, label="boom") is False
    assert _is_eager(fe)


def test_does_not_set_global_suppress_errors(monkeypatch):
    """REGRESSION. The helper used to set `torch._dynamo.config.suppress_errors = True` globally to
    work around ONE uncompilable op (see `species_posterior_compiles_test.py`). That turned every
    OTHER backend failure — anywhere in the process, forever — into a silent per-frame eager
    fallback. The op is fixed; the global suppression must not come back."""
    fe = _FE(cost_ms=2.0)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    S.maybe_compile_extractor(_model(fe), True)
    assert torch._dynamo.config.suppress_errors is False


def test_late_failure_falls_back_to_eager_instead_of_raising(monkeypatch):
    """`torch.compile` guards on input properties, so a shape it has not seen can trigger a fresh
    trace at CALL time — long after the load-time try/except returned. That must degrade THIS model
    to eager, not kill a 3-hour run. This is the scoped replacement for global suppress_errors."""
    fe = _FE(cost_ms=2.0)
    calls = {"n": 0}

    # Load-time does 1 forced compile + _TIMING_WARMUP + _TIMING_REPS calls; start failing after all
    # of those so this exercises the LATE path rather than the load-time try/except.
    _load_time_calls = 1 + S._TIMING_WARMUP + S._TIMING_REPS

    def flaky(_fn):
        def wrapper(obs):
            calls["n"] += 1
            if calls["n"] > _load_time_calls:
                raise RuntimeError("late guard failure")
            return fe.lin(obs["observation"])
        return wrapper

    monkeypatch.setattr(torch, "compile", flaky)
    assert S.maybe_compile_extractor(_model(fe), True, label="flaky") is True
    obs = {"observation": torch.zeros(1, 8)}
    for _ in range(4):
        out = fe.forward(obs)                     # must NOT raise
    assert out.shape == (1, 8)


def test_validation_is_paid_once_per_process(monkeypatch):
    """Consumers that load models in a LOOP (the search-teacher worker rebuilds its opponent every
    iteration) must not re-pay ~15 eager forwards for an answer that cannot have changed —
    `torch.compile` keys on the code object, so the second model in a process has the same answer."""
    monkeypatch.setattr(S, "_COMPILE_VALIDATED", False)
    calls = {"n": 0}
    fe1 = _FE(cost_ms=2.0)

    def counting_compile(_fn):
        calls["n"] += 1
        return lambda obs: fe1.lin(obs["observation"])

    monkeypatch.setattr(torch, "compile", counting_compile)
    assert S.maybe_compile_extractor(_model(fe1), True, label="first") is True
    assert S._COMPILE_VALIDATED is True

    # A second model: still compiled, but the eager/compiled timing is skipped.
    fe2 = _FE(cost_ms=2.0)
    fe2_forwards = {"n": 0}
    orig_forward = fe2.forward

    def counted_forward(obs):
        fe2_forwards["n"] += 1
        return orig_forward(obs)

    fe2.forward = counted_forward
    monkeypatch.setattr(torch, "compile", lambda _fn: (lambda obs: fe1.lin(obs["observation"])))
    assert S.maybe_compile_extractor(_model(fe2), True, label="second") is True
    assert fe2_forwards["n"] == 0, (
        f"the second compile re-ran {fe2_forwards['n']} eager timing forwards; validation should be "
        f"once per process"
    )


def test_the_reuse_path_still_logs(monkeypatch, capsys):
    """A silent success is indistinguishable from 'never ran' in a run log. That bit us for real:
    the eval-worker OPPONENT compile appeared to be missing, and only instrumenting the call proved
    it had fired — because the reuse path returned without printing."""
    monkeypatch.setattr(S, "_COMPILE_VALIDATED", True)      # pretend an earlier model validated
    fe = _FE(cost_ms=2.0)
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    assert S.maybe_compile_extractor(_model(fe), True, label="eval-opp:x.zip") is True
    out = capsys.readouterr().out
    assert "eval-opp:x.zip" in out and "ON" in out, out


def test_grad_enabled_calls_route_to_eager(monkeypatch):
    """The compiled artifact is INFERENCE-only: under grad, dynamo hands the graph to AOTAutograd,
    which must lower the BACKWARD too, and Inductor's CPU backward codegen fails on this model's
    scatter/index_add. Frozen opponents always run under no_grad, but the PROBER backprops through
    the same extractor for saliency — so grad-enabled calls must take the eager path."""
    fe = _FE(cost_ms=2.0)
    used = {"compiled": 0}

    def counting(_fn):
        def wrapper(obs):
            used["compiled"] += 1
            return fe.lin(obs["observation"])
        return wrapper

    monkeypatch.setattr(torch, "compile", counting)
    assert S.maybe_compile_extractor(_model(fe), True, label="grad") is True
    obs = {"observation": torch.zeros(1, 8)}
    before = used["compiled"]
    with torch.no_grad():
        fe.forward(obs)
    assert used["compiled"] == before + 1, "no_grad must use the compiled path"
    with torch.enable_grad():
        out = fe.forward(obs)
    assert used["compiled"] == before + 1, "grad-enabled must NOT use the compiled path"
    assert out.requires_grad, "the eager path must still build a graph for saliency"


def test_hide_cuda_true_hides_the_device(monkeypatch):
    """The 48x CUDA-context OOM guard: an env worker must hide the GPU BEFORE inductor runs."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fe = _FE(cost_ms=2.0)
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    S.maybe_compile_extractor(_model(fe), True, label="worker", hide_cuda=True)
    import os
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "an env worker must hide the GPU before compiling — otherwise inductor creates a ~252 MiB "
        "context per worker and 48 of them OOM the card."
    )


def test_hide_cuda_false_leaves_the_device_alone(monkeypatch):
    """The LEARNER needs its GPU. hide_cuda is now the caller's EXPLICIT declaration rather than
    something inferred from `torch.cuda.is_initialized()`, which was only correct by accident of the
    call sites and would have blinded the learner the first time this was called from the parent."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    fe = _FE(cost_ms=2.0)
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    S.maybe_compile_extractor(_model(fe, device="cuda"), True, label="learner", hide_cuda=False)
    import os
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"


def test_hide_cuda_refuses_when_cuda_already_initialised(monkeypatch):
    """Mis-declared caller: hiding the device after a context exists buys nothing, so refuse loudly
    rather than compile and silently pay for a context."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    fe = _FE(cost_ms=2.0)
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    assert S.maybe_compile_extractor(_model(fe), True, label="bad", hide_cuda=True) is False
    assert _is_eager(fe)


def test_patches_the_bound_method_not_the_module(monkeypatch):
    """`torch.compile(module)` prefixes every state_dict key with `_orig_mod.` and breaks resume."""
    fe = _FE(cost_ms=2.0)
    keys_before = set(fe.state_dict())
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    S.maybe_compile_extractor(_model(fe), True)
    assert set(fe.state_dict()) == keys_before
    assert not any(k.startswith("_orig_mod") for k in fe.state_dict())


def test_drops_the_observation_debugger(monkeypatch):
    """The debugger runs numpy asserts inside forward; dynamo dies creating a guard on it. The
    helper calls the extractor's own method rather than assigning to a private attribute."""
    fe = _FE(cost_ms=2.0)
    fe._debugger = object()
    monkeypatch.setattr(torch, "compile", _fast_compile(fe))
    S.maybe_compile_extractor(_model(fe), True, label="dbg")
    assert fe._debugger is None


def test_cache_dir_is_not_an_import_side_effect():
    """`snapshot.py` is imported by the prober, eval workers and offline tooling that never compile.
    The cache dir is set when a compile actually happens, not at import."""
    import os
    monkey = os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    try:
        import importlib
        importlib.reload(S)
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ, (
            "importing snapshot.py must not mutate the environment"
        )
        assert S._inductor_cache_dir() == S.DEFAULT_INDUCTOR_CACHE_DIR
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == S.DEFAULT_INDUCTOR_CACHE_DIR
    finally:
        if monkey is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = monkey
