"""Unit tests for `maybe_compile_extractor` — the `--compile-opponents` runtime perf knob.

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
import random
import statistics
import types

import pytest
import torch

from agents.model import compile_opponents as S


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Every test gets a virgin module: no validated compile carried in from a neighbour, no quorum
    verdicts pooled across tests, no inherited tally directory. Without this the file's verdicts
    depend on collection ORDER — which is the same class of defect the measurement fix is about."""
    monkeypatch.setattr(S, "_COMPILE_VALIDATED", False)
    monkeypatch.setattr(S, "_LOCAL_TALLY", {"reverts": 0, "total": 0})
    monkeypatch.delenv(S.COMPILE_QUORUM_ENV, raising=False)


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
    work around ONE uncompilable op (see `extractor_compiles_test.py`). That turned every
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

    # Load-time does 1 forced compile + the warm-up + every timed sample; start failing after all of
    # those so this exercises the LATE path rather than the load-time try/except.
    _load_time_calls = 1 + S._TIMING_WARMUP + S._TIMING_SAMPLES * S._TIMING_REPS

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


# ── The measurement under the floor (2026-08-24) ──────────────────────────────────────────────
#
# THE RECORDED PRODUCTION REGIME. Same checkpoint, same box, n=48 env workers compiling at once.
# The EAGER arm alone spread 14.94-115.71 ms (7.7x) and the compiled arm 2.08-17.90 ms; the healthy
# answer is 6.3x median across 48/48 workers, yet one worker scored 0.78x (FATAL under strict, a
# dead launch), another landed at EXACTLY the 1.05x floor, and the successful run's own spread ran
# to 47.8x.
#
# The mechanism reproduced below is the one the fix targets: the old gate timed the arms BACK TO
# BACK, so a load regime that drifts across the measurement window — 47 sibling workers each
# running a ~30 s compile — lands unequally on the two arms. Each arm's aggregate is a min, so it
# reports the best moment IN ITS OWN BLOCK, and the two blocks are different moments. A drift of D
# over the window therefore multiplies the reported ratio by roughly D^-0.5.
#
# That model is not decoration: at D = 64 it produces **0.77x** and at D = 1/64 **51x**, which
# bracket the two recorded extremes (0.78x and 47.8x) without being fitted to them.
_TRUE_EAGER_MS = 14.94                       # the recorded floor of the eager spread
_TRUE_SPEEDUP = 6.3                          # the 48/48 median
_TRUE_COMPILED_MS = _TRUE_EAGER_MS / _TRUE_SPEEDUP


def _tagged(tag):
    def fn(_obs):
        return None
    fn.tag = tag
    return fn


class _DriftingBox:
    """One worker's measurement window on a box whose load drifts by `drift` from start to end.

    Costs are a pure function of the CALL INDEX, so the only thing that distinguishes the old
    measurement design from the new one is WHEN each arm's calls happen — which is exactly the
    variable under test.
    """

    def __init__(self, drift, n_calls, rng=None):
        self.drift, self.n_calls, self.rng, self.i = drift, n_calls, rng, 0

    def _load(self, i):
        jitter = self.rng.uniform(1.0, 1.15) if self.rng else 1.0
        return (self.drift ** (i / max(self.n_calls - 1, 1))) * jitter

    def sample(self, fn, _obs, reps=S._TIMING_REPS):
        base = _TRUE_EAGER_MS if fn.tag == "eager" else _TRUE_COMPILED_MS
        best = float("inf")
        for _ in range(reps):
            best = min(best, base * self._load(self.i))
            self.i += 1
        return best

    def warm(self, fn, obs, calls=S._TIMING_WARMUP):
        self.i += calls


def _old_gate_ratio(box):
    """The OLD decision logic, verbatim: warm + min-of-12 on eager, THEN warm + min-of-12 on
    compiled, then one ratio. It no longer exists in the module, so it lives here."""
    eager, compiled = _tagged("eager"), _tagged("compiled")
    box.warm(eager, None)
    eager_ms = box.sample(eager, None, reps=12)
    box.warm(compiled, None)
    comp_ms = box.sample(compiled, None, reps=12)
    return eager_ms / comp_ms


class TestTheProductionRegime:
    """Feed the recorded regime through both decision logics and compare their verdicts."""

    def test_old_logic_flips_run_to_run_and_the_new_one_does_not(self, monkeypatch):
        rng = random.Random(20260824)
        old_verdicts, old_ratios = set(), []
        new_verdicts, new_ratios = set(), []
        new_calls = 2 * S._TIMING_WARMUP + 2 * S._TIMING_SAMPLES * S._TIMING_REPS
        old_calls = 2 * (3 + 12)

        for _ in range(400):
            drift = 64.0 ** rng.uniform(-1.0, 1.0)    # the storm building, or clearing, or neither

            r = _old_gate_ratio(_DriftingBox(drift, old_calls, random.Random(rng.random())))
            old_ratios.append(r)
            old_verdicts.add(r >= S._MIN_COMPILE_SPEEDUP)

            box = _DriftingBox(drift, new_calls, random.Random(rng.random()))
            monkeypatch.setattr(S, "_time_forward", box.sample)
            monkeypatch.setattr(S, "_warm_arm", box.warm)
            eager, comp = S._measure_arms(_tagged("eager"), _tagged("compiled"), obs=None)
            r = statistics.median(eager) / statistics.median(comp)
            new_ratios.append(r)
            new_verdicts.add(r >= S._MIN_COMPILE_SPEEDUP)

        # 1. The regime reproduces the defect — the old verdict is not a property of the model.
        assert old_verdicts == {True, False}, (
            f"the old gate must flip over this regime; ratios {min(old_ratios):.2f}-"
            f"{max(old_ratios):.2f}x"
        )
        assert min(old_ratios) < 1.0 < 47.0 < max(old_ratios), (
            f"and it must reach the recorded extremes (0.78x and 47.8x were both real): "
            f"{min(old_ratios):.2f}-{max(old_ratios):.2f}x"
        )
        # 2. The new one does not, on the SAME drifts.
        assert new_verdicts == {True}, (
            f"median-of-{S._TIMING_SAMPLES}, alternated, must be stable across the same regime; got "
            f"both verdicts with ratios {min(new_ratios):.2f}-{max(new_ratios):.2f}x"
        )
        # 3. Quantify it rather than just asserting the verdict.
        old_spread = max(old_ratios) / min(old_ratios)
        new_spread = max(new_ratios) / min(new_ratios)
        assert new_spread < old_spread / 10, (
            f"ratio spread over the same regime: old {old_spread:.1f}x, new {new_spread:.1f}x"
        )
        assert 0.5 < statistics.median(new_ratios) / _TRUE_SPEEDUP < 2.0, (
            f"and it must stay near the TRUE {_TRUE_SPEEDUP}x, not merely be stable: "
            f"{statistics.median(new_ratios):.2f}x"
        )

    def _losing_ratios(self, monkeypatch, true_speedup, drift_decades, n=200, seed=7):
        rng = random.Random(seed)
        n_calls = 2 * S._TIMING_WARMUP + 2 * S._TIMING_SAMPLES * S._TIMING_REPS
        out = []
        for _ in range(n):
            drift = drift_decades ** rng.uniform(-1.0, 1.0)
            box = _DriftingBox(drift, n_calls, random.Random(rng.random()))
            box_sample = box.sample
            penalty = _TRUE_SPEEDUP / true_speedup

            def losing(fn, obs, reps=S._TIMING_REPS, _s=box_sample, _p=penalty):
                ms = _s(fn, obs, reps)
                return ms if fn.tag == "eager" else ms * _p
            monkeypatch.setattr(S, "_time_forward", losing)
            monkeypatch.setattr(S, "_warm_arm", box.warm)
            eager, comp = S._measure_arms(_tagged("eager"), _tagged("compiled"), obs=None)
            out.append(statistics.median(eager) / statistics.median(comp))
        return out

    def test_a_genuinely_losing_compile_is_still_reverted(self, monkeypatch):
        """The fix must not be a disabled gate: June measured 0.70x for real, and that must still
        lose. On an ordinarily busy box (drift within 8x across the window) it loses every time."""
        ratios = self._losing_ratios(monkeypatch, true_speedup=0.70, drift_decades=8.0)
        assert max(ratios) < S._MIN_COMPILE_SPEEDUP, (
            f"a 0.70x compile read as high as {max(ratios):.2f}x")

    def test_the_residual_drift_BIAS_is_bounded_but_real(self, monkeypatch):
        """HONEST LIMIT, pinned rather than glossed. Alternation cancels most of a drifting regime
        but not all of it: each arm's five samples sit at slightly different moments, so a residual
        bias of order drift^0.1 survives. Under a hostile 64x drift that is ~1.5x — enough to carry
        a marginally-losing 0.70x compile up to the floor in ~2% of draws, and the reason a
        below-floor reading is the QUORUM's business rather than one worker's."""
        ratios = self._losing_ratios(monkeypatch, true_speedup=0.70, drift_decades=64.0)
        reverted = sum(r < S._MIN_COMPILE_SPEEDUP for r in ratios) / len(ratios)
        assert reverted > 0.95, f"only {reverted:.0%} of a 0.70x compile's readings reverted"
        assert statistics.median(ratios) < 0.9, statistics.median(ratios)
        assert max(ratios) < _TRUE_SPEEDUP * 0.5, (
            f"the residual bias must stay a small factor, not a rescue: max {max(ratios):.2f}x")

    def test_a_badly_losing_compile_never_survives_any_drift(self, monkeypatch):
        """The bias is bounded, so a compile that loses by a real margin cannot be carried over the
        floor by ANY drift the regime produces."""
        ratios = self._losing_ratios(monkeypatch, true_speedup=0.40, drift_decades=64.0)
        assert max(ratios) < S._MIN_COMPILE_SPEEDUP, max(ratios)

    def test_both_arms_are_warmed_before_either_is_timed(self, monkeypatch):
        """The old gate timed eager FIRST and compiled SECOND, so the eager arm paid every
        first-touch cost the compiled arm then measured warm. Whatever the warm-up is, both arms
        must have had exactly the same amount of it when the first timed sample is taken."""
        calls = {"eager": 0, "compiled": 0}
        snapshots = []

        def counting(tag):
            def fn(_obs):
                calls[tag] += 1
            fn.tag = tag
            return fn

        def sample(fn, _obs, reps=None):
            snapshots.append(dict(calls))
            return 1.0
        monkeypatch.setattr(S, "_time_forward", sample)

        S._measure_arms(counting("eager"), counting("compiled"), obs=None)
        assert snapshots[0] == {"eager": S._TIMING_WARMUP, "compiled": S._TIMING_WARMUP}

    def test_the_arms_are_alternated_and_the_order_flips(self, monkeypatch):
        """Interleaving is what makes a drifting load regime hit both arms alike. The per-round
        order also flips so neither arm permanently owns the post-other-arm position."""
        order = []

        def sample(fn, _obs, reps=None):
            order.append(fn.tag)
            return 1.0
        monkeypatch.setattr(S, "_time_forward", sample)
        monkeypatch.setattr(S, "_warm_arm", lambda *a, **k: None)

        S._measure_arms(_tagged("eager"), _tagged("compiled"), obs=None)
        assert order[:4] == ["eager", "compiled", "compiled", "eager"], order
        assert order.count("eager") == order.count("compiled") == S._TIMING_SAMPLES


class TestTheQuorum:
    """`--compile-opponents-strict` killed three launches because ONE worker's ratio was fatal.
    Below-floor is now a warning with its numbers; fatal needs a systemic fraction to agree."""

    @staticmethod
    def _slow_compile(fn):
        def wrapper(obs):
            import time as _t
            t0 = _t.perf_counter()
            while (_t.perf_counter() - t0) * 1e3 < 0.5:
                pass
            return fn(obs)
        return wrapper

    def _armed(self, monkeypatch, tmp_path, ok=0, revert=0):
        monkeypatch.setenv(S.COMPILE_QUORUM_ENV, str(tmp_path))
        for i in range(ok):
            (tmp_path / f"pre{i}.ok").write_text("")
        for i in range(revert):
            (tmp_path / f"pre{i}.revert").write_text("")
        monkeypatch.setattr(torch, "compile", self._slow_compile)

    def test_one_worker_below_the_floor_is_a_warning_not_a_launch_killer(
            self, monkeypatch, tmp_path, capsys):
        self._armed(monkeypatch, tmp_path, ok=20)
        fe = _FE()
        assert S.maybe_compile_extractor(_model(fe), True, label="w7", strict=True) is False
        assert _is_eager(fe), "the reading still reverts THIS opponent to eager"
        err = capsys.readouterr().err
        assert "REVERTED" in err and "Quorum so far: 1/21" in err, err

    def test_a_systemic_revert_is_still_fatal_under_strict(self, monkeypatch, tmp_path):
        """The flag must keep its point: an invisible ~6.5x regression across the fleet should stop
        the run at startup rather than show up in the FPS graph a day later."""
        self._armed(monkeypatch, tmp_path, revert=3)
        with pytest.raises(S.CompileExtractorError) as exc:
            S.maybe_compile_extractor(_model(_FE()), True, label="w7", strict=True)
        assert "systemic" in str(exc.value)

    def test_the_first_readings_can_never_be_fatal(self, monkeypatch, tmp_path):
        """A prefix tally has to survive being read when it holds one entry. The worker that reports
        first would otherwise be 1/1 = 100% below the floor and kill the run by itself."""
        self._armed(monkeypatch, tmp_path)
        assert S.maybe_compile_extractor(_model(_FE()), True, label="w0", strict=True) is False

    def test_a_compile_that_ERRORS_is_still_fatal_immediately(self, monkeypatch, tmp_path):
        """The quorum covers the TIMING verdict only. A backend crash is a fact, not a reading."""
        monkeypatch.setenv(S.COMPILE_QUORUM_ENV, str(tmp_path))

        def boom(_fn):
            raise RuntimeError("inductor exploded")
        monkeypatch.setattr(torch, "compile", boom)
        with pytest.raises(S.CompileExtractorError):
            S.maybe_compile_extractor(_model(_FE()), True, label="w0", strict=True)

    @pytest.mark.parametrize("reverts,total,fatal", [
        (1, 1, False), (3, 3, False),               # below _QUORUM_MIN_REPORTS: decides nothing
        (1, 4, False),                              # EXACTLY 25% — the boundary must not trip
        (2, 4, True), (13, 48, True), (12, 48, False),
    ])
    def test_the_fatal_condition(self, reverts, total, fatal):
        assert S._quorum_is_fatal(reverts, total) is fatal

    def test_the_tally_is_shared_across_processes_through_the_environment(self, tmp_path, monkeypatch):
        """One file per verdict, no lock: the count a worker reads is every verdict written before
        it looked. That is a PREFIX estimate and is documented as one."""
        monkeypatch.setenv(S.COMPILE_QUORUM_ENV, str(tmp_path))
        assert S._record_verdict(reverted=False) == (0, 1)
        assert S._record_verdict(reverted=True) == (1, 2)
        assert S._record_verdict(reverted=True) == (2, 3)

    def test_an_unwritable_tally_falls_back_to_the_process_local_one(self, monkeypatch):
        """Bookkeeping must never be the thing that fails a compile."""
        monkeypatch.setenv(S.COMPILE_QUORUM_ENV, "/proc/definitely/not/writable")
        assert S._record_verdict(reverted=True) == (1, 1)
        assert S._record_verdict(reverted=False) == (1, 2)

    def test_arming_publishes_a_fresh_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv(S.COMPILE_QUORUM_ENV, raising=False)
        d = S.arm_compile_quorum(str(tmp_path))
        S._record_verdict(reverted=True)
        assert len(list(__import__("os").listdir(d))) == 1
        S.arm_compile_quorum(str(tmp_path))           # a restart counts fresh, not protected by history
        assert len(list(__import__("os").listdir(d))) == 0


def test_the_revert_message_reports_measurements_and_asserts_no_cause(monkeypatch, capsys):
    """REGRESSION. The message used to claim 'the graph is probably fragmented' — a cause a ratio of
    two timings structurally cannot distinguish from a busy box, and it sent three separate launch
    investigations after the wrong thing. Report what was measured; the reader diagnoses."""
    fe = _FE()

    def slow(fn):
        def wrapper(obs):
            import time as _t
            t0 = _t.perf_counter()
            while (_t.perf_counter() - t0) * 1e3 < 0.5:
                pass
            return fn(obs)
        return wrapper

    monkeypatch.setattr(torch, "compile", slow)
    assert S.maybe_compile_extractor(_model(fe), True, label="noisy") is False
    err = capsys.readouterr().err
    assert "fragmented" not in err, err
    assert "median eager" in err and "median compiled" in err
    assert f"{S._MIN_COMPILE_SPEEDUP:.2f}x floor" in err
    assert err.count(",") >= 2 * (S._TIMING_SAMPLES - 1), (
        f"both arms' full sample series must be printed, not just the medians: {err}"
    )


def test_the_warmup_obs_uses_the_LIVE_call_signature(monkeypatch):
    """Dynamo guards on a dict's KEY SET and on dtype as hard as it guards on shape, and every real
    opponent call arrives through `policy.get_distribution` with `observation` + a preprocessed
    float32 `action_mask`. Warming with one key leaves the first LIVE decision to re-trace the whole
    extractor — 19.5 s, measured in the cf producer (53870dd)."""
    obs = S._compile_warmup_obs(_FE())
    assert sorted(obs) == ["action_mask", "observation"]
    assert obs["observation"].shape == (1, 8) and obs["observation"].dtype is torch.float32
    assert obs["action_mask"].shape == (1, 11) and obs["action_mask"].dtype is torch.float32


def test_cache_dir_is_not_an_import_side_effect():
    """`compile_opponents.py` is imported (via `snapshot.py`'s re-export) by the prober, eval
    workers and offline tooling that never compile. The cache dir is set when a compile actually
    happens, not at import."""
    import os
    monkey = os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    try:
        import importlib
        importlib.reload(S)
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ, (
            "importing compile_opponents.py must not mutate the environment"
        )
        assert S._inductor_cache_dir() == S.DEFAULT_INDUCTOR_CACHE_DIR
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == S.DEFAULT_INDUCTOR_CACHE_DIR
    finally:
        if monkey is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = monkey
