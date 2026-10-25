"""A LIMITATION PIN, on the CPU: Triton 3.4.0 mis-compiles one kernel shape at ``XBLOCK = 16`` (2026-10-25).

THE 2026-10-25 RED. `extractor_compiles_test.py::test_production_arch_compiles_on_cuda` failed on the card at
B = 2 with `InductorError ... make_ttgir ... RuntimeError: PassManager::run failed`. Driving Inductor's Triton
codegen on the CPU (`designs/research_state/measurements/triton_xblock16_2026-10-25/`) found the cause: 2 of
the forward's 656 kernels at B = 2 (0 at B = 1, 4, 8, 64 and 256) fail Triton's `TritonGPURemoveLayoutConversions`
pass ("operand #1 does not dominate this use") for every autotune config with ``XBLOCK = 16`` and compile for
every other. They are the speed mixture's fp64 cumsum (`damage_op_speed._opp_speed_mix`, a LOOPED reduction
over the 512-speed lattice) with the ``[B,6]`` Beat-Up / max-HP pointwise epilogue fused in, which makes that
epilogue ``[XBLOCK, 1]``-shaped; ``XBLOCK = 16`` is chosen only when the ``6B`` rows round to 16, i.e. B = 2.
No default launch compiles below T2's smallest bucket (8), so the CUDA inference cell runs there now.

The kernel below is that epilogue, cut down by line deletion from the generated one with the failure kept
(the cumsum loop itself is not needed). Compiling for sm_86 needs no GPU (`triton.compile` with a
`GPUTarget`), so this pin runs in the routine tier. It FAILS IF THE BUG LIFTS: then record the new Triton
version as lifted below, and the CUDA cell may go back to `_BATCH`.
"""
import pytest
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

# Does the XBLOCK-16 kernel FAIL Triton's pass pipeline? Per EXACT triton version; an unknown one FAILS the pin.
# Measured 2026-10-25: 3.4.0 (torch 2.8, both envs) fails; 3.1.0 (torch 2.5.1, `gen3ai_stable`) compiles it.
_XBLOCK16_FAILS = {"3.4.0": True, "3.1.0": False}


@triton.jit
def remat_kernel(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr4, out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x3 = xindex
    tmp36 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp40 = 6 + x0
    tmp43 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp44 = tmp40 < tmp43
    tmp46 = tl.full([XBLOCK, 1], 1, tl.int1)
    tmp49 = tl.load(in_ptr1 + (6*x1 + (x0)), tmp46 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp56 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp58 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp59 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp61 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp63 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp65 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp66 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp67 = tl.load(in_ptr0 + (122*(6 + x0) + 1472*x1), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.load(in_ptr2 + (122*(x0) + 732*x1), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tl.load(in_ptr0 + (732 + 122*(x0) + 1472*x1), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tl.where(tmp49, tmp68, tmp69)
    tmp71 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp72 = tl.where(tmp46, tmp70, tmp71)
    tmp73 = tl.where(tmp44, tmp67, tmp72)
    tmp74 = tmp73.to(tl.int64)
    tmp76 = tl.full([XBLOCK, 1], 1, tl.int64)
    tmp77 = tmp74 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp74)
    tmp80 = tl.load(in_ptr4 + (6*tmp78), xmask, eviction_policy='evict_last')
    tmp81 = tmp80 * tmp61
    tmp82 = tmp81 + tmp63
    tmp83 = tmp82 + tmp65
    tmp84 = tl.where(tmp36, tmp66, tmp83)
    tmp90 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp91 = tl.load(in_ptr4 + (2 + 6*tmp78), xmask, eviction_policy='evict_last')
    tmp92 = triton_helpers.maximum(tmp91, tmp58)
    tmp93 = tl.where(tmp36, tmp90, tmp92)
    tmp94 = tmp59 * tmp84
    tmp108 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tmp109 = tmp94 - tmp108
    tmp110 = triton_helpers.maximum(tmp109, tmp56)
    tl.store(in_out_ptr0 + (x3), tmp84, xmask)
    tl.store(in_out_ptr1 + (x3), tmp93, xmask)
    tl.store(out_ptr3 + (x3), tmp110, xmask)


_SIG = {"in_out_ptr0": "*fp32", "in_out_ptr1": "*fp32", "in_ptr0": "*fp32", "in_ptr1": "*i1", "in_ptr2": "*fp32",
        "in_ptr4": "*fp32", "out_ptr3": "*fp32", "xnumel": "i32", "XBLOCK": "constexpr"}


def _compile(xblock: int, num_warps: int):
    return triton.compile(ASTSource(remat_kernel, _SIG, {"XBLOCK": xblock}), target=GPUTarget("cuda", 86, 32),
                          options={"num_warps": num_warps, "num_stages": 1})


@pytest.fixture(autouse=True)
def _private_triton_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


@pytest.mark.parametrize("xblock,num_warps", [(8, 2), (8, 16), (32, 2), (64, 8)])
def test_every_other_xblock_compiles_for_sm86(xblock, num_warps):
    """The control: the same kernel at the block sizes B = 1, 4, 8 and up autotune to compiles to a cubin."""
    assert "cubin" in _compile(xblock, num_warps).asm


@pytest.mark.parametrize("num_warps", [2, 8])
def test_xblock_16_fails_remove_layout_conversions(num_warps, capfd):
    """The two warp counts the B = 2 configs used. Asserts the REASON (the dominance verifier error), not only
    that something raised."""
    fails = _XBLOCK16_FAILS.get(triton.__version__)
    assert fails is not None, (f"no recorded XBLOCK-16 expectation for triton {triton.__version__}: run this test "
                               f"and record the answer in _XBLOCK16_FAILS")
    try:
        _compile(16, num_warps)
    except RuntimeError as exc:
        assert fails, f"triton {triton.__version__} is recorded as fixed, yet XBLOCK 16 failed: {exc}"
        assert "PassManager::run failed" in str(exc)
        assert "does not dominate this use" in capfd.readouterr().err
        return
    assert not fails, (f"the XBLOCK-16 kernel now COMPILES on triton {triton.__version__}: the bug lifted. Record "
                       f"{triton.__version__!r}: False in _XBLOCK16_FAILS; extractor_compiles_test's CUDA inference "
                       f"cell may then run at _BATCH again.")
