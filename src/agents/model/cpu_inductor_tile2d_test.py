"""`gen3_cpu_inductor_scalar_v1` (2026-10-15): the CPU Inductor tile2d codegen bug and the CPU-only option that
steps around it.

WHAT BROKE. Since the end state became production (`ed58a3ad`) `compiled_first_update_test`'s CPU cell died in g++
with `error: 'tmp8' was not declared in this scope` inside a `transpose_mxn` call, from a `slow` test that the
routine gate does not run. The end-state graph's move-resolution family reads rows at the active slot
(`x[arange(B), idx]`, e.g. `o.pair_type_mult[ar, o.our_active]`), and torch 2.8.0's Inductor CPU backend, when it
picks a 2-D TILED vector kernel for such a gather beside a [B] reduction over a strided column, emits the gather's
per-row index variable BEFORE it is declared (`CppTile2DKernel.need_vec_transpose` does not exclude an indirect
index). Trigger: B >= 16 and K >= 16 (both at least the 16 fp32 lanes of AVX-512); `designs/training/compile_flags.md`
"The CPU Inductor tile2d codegen bug" has the kernel-level account and the upstream issues.

THE SMALLEST REPRO is `_repro` below: a gather at a per-row index, a strided-column reduction over K, the gather
stored. THE STEP-AROUND is `compile_regions.CPU_INDUCTOR_OPTIONS` (`cpp.simdlen = 1`: scalar C++ kernels, no
tiling), applied to a CPU learner on the default Inductor backend only; a CUDA learner compiles to Triton, never
reaches the CPP codegen, and keeps its kwargs byte for byte.

Each test names what it pins and fails on a revert of it.
"""
from __future__ import annotations

import os
import types

import pytest
import torch

from agents.model import compile_regions as cr

B, K = 16, 18                       # the failing shape: B and K both >= 16 fp32 lanes

_skip_compile = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1",
                                   reason="GEN3AI_SKIP_COMPILE_TESTS=1")


def _repro(x: torch.Tensor, idx: torch.Tensor, q: torch.Tensor):
    """x [B,6,K] · idx [B] long · q [B,K,6]. `torch.arange` is built INSIDE the graph (a captured tensor makes
    the batch coordinate an indirect load too, which is a path that works)."""
    g = x[torch.arange(B), idx]                     # [B,K]: the row at each batch row's own slot
    return (g * q[..., 5]).sum(-1), g               # a [B] reduction over K of a strided column, and the gather stored


def _inputs():
    gen = torch.Generator().manual_seed(0)
    return (torch.rand(B, 6, K, generator=gen), torch.randint(0, 6, (B,), generator=gen),
            torch.rand(B, K, 6, generator=gen))


def _compile(options):
    torch._dynamo.reset()
    kw = {"backend": "inductor", "fullgraph": True}
    if options:
        kw["options"] = dict(options)
    return torch.compile(_repro, **kw)


@pytest.fixture
def one_thread(restore_torch_globals):
    """A test's CPU compile and run use ONE torch thread (designs/ops/testing.md "One torch thread for the CPU plays")."""
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch._dynamo.reset()


# ------------------------------------------------------------------------------ the option, as data
def test_the_cpu_option_is_a_real_inductor_config_of_this_torch():
    import torch._inductor.config as ic
    for key, val in cr.CPU_INDUCTOR_OPTIONS.items():
        obj = ic
        for part in key.split("."):
            assert hasattr(obj, part), f"{key} is not an Inductor config of torch {torch.__version__}"
            obj = getattr(obj, part)
        assert val is not None and (obj is None or type(obj) is type(val)), (key, obj, val)


@pytest.mark.parametrize("device,backend,expect", [
    ("cpu", None, cr.CPU_INDUCTOR_OPTIONS),
    ("cpu", "inductor", cr.CPU_INDUCTOR_OPTIONS),
    ("cuda", None, {}),                             # the production device: NO option, the compile unchanged
    ("cuda:0", "inductor", {}),
    ("cpu", "eager", {}),                           # a stand-in backend takes no Inductor options
    ("cpu", "aot_eager", {}),
])
def test_only_a_cpu_learner_on_the_default_backend_gets_the_option(device, backend, expect):
    model = types.SimpleNamespace(device=torch.device(device))      # `torch.device("cuda")` never touches the card
    assert cr.cpu_inductor_options(model, backend) == expect


def test_the_option_falls_back_to_the_policys_parameters_when_the_learner_has_no_device():
    policy = torch.nn.Linear(2, 2)
    assert cr.learner_device_type(types.SimpleNamespace(policy=policy)) == "cpu"


# ------------------------------------------------------------------------------ the wiring in install
@pytest.mark.parametrize("device,expect", [("cpu", {"options": cr.CPU_INDUCTOR_OPTIONS}), ("cuda", {})])
def test_install_hands_R1_the_cpu_option_and_a_cuda_learner_nothing(monkeypatch, device, expect):
    """CPU: R1's `torch.compile` carries `CPU_INDUCTOR_OPTIONS`. CUDA: exactly the production kwargs
    (`fullgraph`, `dynamic`), no `options` — byte for byte what it was before this fix."""
    from agents.model import compile_control as cc
    from agents.training import learner_golden as LG
    seen = []
    real = torch.compile

    def spy(fn, **kw):
        seen.append(dict(kw))
        return real(fn, fullgraph=True, dynamic=False, backend="eager")

    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    try:
        monkeypatch.setattr(torch, "compile", spy)
        model.device = torch.device(device)
        cc.control().install()
        cr.install(model)
        assert seen == [{"fullgraph": True, "dynamic": False, **expect}]
    finally:
        monkeypatch.undo()
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.reset()


# ------------------------------------------------------------------------------ the typed refusal's wording
@pytest.mark.parametrize("message,names_codegen", [
    ("InductorError: CppCompileError: C++ compile error ... 'tmp8' was not declared in this scope", True),
    ("Unsupported: call to a method dynamo cannot trace (a real graph break)", False),
])
def test_a_backend_codegen_failure_is_not_reported_as_a_graph_break(message, names_codegen):
    """`_fatal_compile_errors` turns a compile failure inside a declared region into the typed
    `CompileTrainerError`; an Inductor C++ codegen error said only "does not compile as one graph", which sent a
    reader to look for a graph break. It now says, in addition, that the backend failed and where to read."""
    from torch._dynamo.exc import TorchDynamoException
    from agents.model.compile_trainer import CompileTrainerError
    with pytest.raises(CompileTrainerError) as e:
        with cr._fatal_compile_errors():
            raise TorchDynamoException(message)
    text = str(e.value)
    assert "does not compile as one graph" in text                    # the existing wording stays
    assert ("NOT a graph break" in text) is names_codegen, text[-500:]


# ------------------------------------------------------------------------------ the real compile
@pytest.mark.slow  # ship gate (2026-10-19): 21 s beside a live run
@_skip_compile
def test_the_smallest_repro_compiles_and_matches_eager_with_the_cpu_option(one_thread):
    """Compiles (no `was not declared`), vectorisation and tiling are OFF in the emitted C++, and the numbers are
    eager's to fp32 rounding. Fails if the option is dropped (the compile dies in g++)."""
    from torch._inductor.utils import run_and_get_cpp_code
    args = _inputs()
    out, code = run_and_get_cpp_code(_compile(cr.CPU_INDUCTOR_OPTIONS), *args)
    ref = _repro(*args)
    assert "transpose_mxn<" not in code and "Vectorized<float>" not in code, "the option did not turn the CPP vector path off"
    assert torch.equal(out[1], ref[1]), "the gathered rows are an exact copy"
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-6)


@_skip_compile
@pytest.mark.skipif(not torch.__version__.startswith("2.8.0"),
                    reason="the bug is recorded on torch 2.8.0 (the env pins it); another build is a measurement of its own")
def test_the_bug_is_still_in_this_torch_so_the_option_is_still_needed(one_thread):
    """A TRIPWIRE, deterministic per torch build: without the option the repro still dies in g++ on 2.8.0.
    If this FAILS the compile worked: torch fixed the tile2d codegen, so retire `CPU_INDUCTOR_OPTIONS` (and this
    file's tripwire) in the same commit that moves the pin — do not keep a workaround for a bug that is gone."""
    with pytest.raises(Exception) as exc:
        _compile({})(*_inputs())
    text = str(exc.value)
    assert "was not declared in this scope" in text and "transpose_mxn" in text, text[:600]
