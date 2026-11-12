"""A LIMITATION PIN, on the CPU: at ``XBLOCK = 16`` Triton 3.4.0 fails one kernel only when OUR spread is read
through a hypothesis context's cat (2026-11-11).

THE 2026-11-11 RED. `extractor_compiles_test.py::test_production_arch_compiles_forward_and_backward_on_cuda` failed
on the card at B = 2 with `make_ttgir ... RuntimeError: PassManager::run failed`. Driving Inductor's Triton codegen
on the CPU found 1 failing kernel of the 1,059 the forward + backward emits at B = 2 (0 of 1,070 at B = 8): the
non-formula damage kinds' roll (`damage_kinds.nonformula_rolls`, a LOOPED reduction over the 400 moves) with OUR
``[B, 6]`` max-HP chain (`damage_kinds.our_max_hp`) fused in as its ``[XBLOCK, 1]`` prologue. It is the bug
`triton_xblock16_remat_test.py` pins (`TritonGPURemoveLayoutConversions`: "operand #0 does not dominate this use"),
reached through a different kernel.

WHAT DECIDES IT AT 16. The prologue read OUR IVs and EVs as ``ctx.pokemon_part[:, :TEAM_SIZE, ...]`` on a
HYPOTHESIS context, whose block is ``cat([ours, where(hidden, dex_row, theirs)])``. Inductor lowers that cat
pointwise, so each read is three masked loads and a select chain, with the dead opponent branch kept. With those
reads the kernel fails at the two XBLOCK-16 configs below; with OUR rows loaded directly it compiles at all three.
Nothing inside the max-HP arithmetic changes that (reassociating the sum, or storing a different partial, was
tried on the generated kernel), so the fix is WHERE the spread is read: `damage_kinds.our_team_spread` reads the
REAL context's rows (`extractor_ctx.our_rows`), numerically identical (`our_rows_spelling_test.py`).

WHERE IT FAILED, AND WHAT THE FIX COVERS. Before the fix the generated kernel failed 2 of 3 configs at ``XBLOCK``
16, 2 of 3 at 32 and all 3 at 64, and the forward + backward graph formed it at B = 2 AND at B = 4 (both red on the
CPU screen; at B = 6 Inductor left the prologue out of the loop). At the fix the generated kernel (B = 2, 4
and 6) compiles at all nine of those configs, and every screened graph has 0 failing kernels (forward + backward
at B = 2, 3, 4, 5, 6, 8; the no-grad forward at B = 2, 3, 4, 8, 16).

WHAT THIS KERNEL IS, AND IS NOT. It is the PRE-FIX generated kernel cut down by hand with the failure kept: the
prologue with both spellings behind one constexpr, a stored partial of the stat sum, and the one loop statement
that needs the chain (a per-battle table lookup whose index Inductor broadcasts to ``[XBLOCK, R0_BLOCK]``, divided
by the max HP). ``THROUGH_CAT`` changes the two reads and NOTHING else, which is what shows the read decides it at
16. It is therefore NOT the kernel Inductor emits at the fix, which also stores a different set of partials: with
the stored partial kept, this kernel still fails at 32 and 64 with the direct read, and without it (``STORE``
false) it compiles everywhere in both spellings. So the direct read is necessary here, not a guarantee: a new
`PassManager::run failed` is this class again (a stored prologue value whose chain feeds a looped reduction);
start from `designs/training/compile_flags.md` ("Triton RemoveLayoutConversions").

Compiling for sm_86 needs no GPU. Every cell FAILS IF ITS OUTCOME CHANGES: record the new Triton version in
`_FAILS`; if the bug has lifted, `our_team_spread` is free to read through the context again and the CUDA
inference cell may go back to `_BATCH`.
"""
import pytest
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

# (XBLOCK, R0_BLOCK, num_warps): the configs Inductor autotunes this reduction over, by the row count's power of 2
# (`torch._inductor.runtime.triton_heuristics._reduction_configs`, torch 2.8.0).
_AT_16 = [(16, 4, 8), (16, 8, 2), (16, 64, 8)]            # 6B in (8, 16]: B = 2
_BELOW = [(1, 512, 4), (8, 8, 2), (8, 512, 16)]           # every row count autotunes the first and the last
_ABOVE = [(32, 8, 2), (64, 8, 4)]                         # 6B in (16, 32] and above

# Does the config FAIL Triton's pass pipeline? {triton version: {through the cat?: the failing configs}}. Per EXACT
# version; an unknown one FAILS the pin. Measured 2026-11-11: 3.4.0 (torch 2.8) as below; 3.1.0 (torch 2.5.1,
# `gen3ai_stable`) compiles every cell.
_FAILS = {
    "3.4.0": {True: {(16, 8, 2), (16, 64, 8), (32, 8, 2), (64, 8, 4)}, False: {(32, 8, 2), (64, 8, 4)}},
    "3.1.0": {True: set(), False: set()},
}


@triton.jit
def nonformula_kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel,
                      XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, THROUGH_CAT: tl.constexpr,
                      STORE: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x2 = xindex
    x0 = (xindex % 6)
    x1 = xindex // 6
    tmp0 = tl.load(in_ptr0 + (6*x2), xmask, eviction_policy='evict_last')
    if THROUGH_CAT:
        # ctx.pokemon_part[:, :TEAM_SIZE, iv] and [..., ev] on a hypothesis context: the cat, lowered pointwise
        tmp6 = tl.full([1, 1], 6, tl.int64)
        tmp7 = x0 < tmp6
        tmp9 = x0 >= tmp6
        tmp12 = tl.load(in_ptr2 + (6*x1 + ((-6) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp8 = tl.load(in_ptr1 + (71 + 122*(x0) + 1472*x1), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (71 + 122*((-6) + x0) + 732*x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr1 + (803 + 122*((-6) + x0) + 1472*x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.where(tmp12, tmp13, tmp14)
        tmp17 = tl.where(tmp9, tmp15, tl.full(tmp15.shape, 0.0, tmp15.dtype))
        tmp18 = tl.where(tmp7, tmp8, tmp17)
        tmp23 = tl.load(in_ptr1 + (77 + 122*(x0) + 1472*x1), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (77 + 122*((-6) + x0) + 732*x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr1 + (809 + 122*((-6) + x0) + 1472*x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.where(tmp12, tmp24, tmp25)
        tmp28 = tl.where(tmp9, tmp26, tl.full(tmp26.shape, 0.0, tmp26.dtype))
        tmp29 = tl.where(tmp7, tmp23, tmp28)
    else:
        # our_team_spread(ctx): the same two cells off the real context's rows
        tmp18 = tl.load(in_ptr1 + (71 + 122*(x0) + 1472*x1), xmask, eviction_policy='evict_last')
        tmp29 = tl.load(in_ptr1 + (77 + 122*(x0) + 1472*x1), xmask, eviction_policy='evict_last')
    tmp22 = tmp0 * 2.0 + libdevice.nearbyint(tmp18 * 31.0)
    tmp36 = tmp22 + libdevice.floor(libdevice.nearbyint(tmp29 * 252.0) * 0.25)
    tmp53 = tl.where(tmp0 == 1.0, 1.0, tmp36 + 110.0)
    tmp58 = tmp53 + 1e-06
    if STORE:
        tl.store(out_ptr0 + (x2), tmp36, xmask)      # the partial the pre-fix kernel kept for another consumer
    tmp118 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        tmp107 = tl.full([XBLOCK, R0_BLOCK], 400, tl.int32)
        tmp121 = tl.where(tmp118 < 0, tmp118 + tmp107, tmp118)
        tmp123 = tl.load(in_ptr5 + (tmp121), xmask, eviction_policy='evict_last')
        tmp129 = (tmp123 / tmp58)
        tl.store(out_ptr1 + (r0_index + 400*x2), tmp129, r0_mask & xmask)


_SIG = {"in_ptr0": "*fp32", "in_ptr1": "*fp32", "in_ptr2": "*i1", "in_ptr3": "*fp32", "in_ptr4": "*i64",
        "in_ptr5": "*fp32", "out_ptr0": "*fp32", "out_ptr1": "*fp32", "xnumel": "i32", "r0_numel": "i32",
        "XBLOCK": "constexpr", "R0_BLOCK": "constexpr", "THROUGH_CAT": "constexpr", "STORE": "constexpr"}


def _compile(cfg, through_cat: bool, store: bool = True):
    xblock, r0_block, num_warps = cfg
    consts = {"XBLOCK": xblock, "R0_BLOCK": r0_block, "THROUGH_CAT": through_cat, "STORE": store}
    return triton.compile(ASTSource(nonformula_kernel, _SIG, consts), target=GPUTarget("cuda", 86, 32),
                          options={"num_warps": num_warps, "num_stages": 1})


@pytest.fixture(autouse=True)
def _private_triton_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


def _check(cfg, through_cat: bool, capfd) -> None:
    """Compile one cell and hold it to the recorded outcome, asserting the REASON of a failure (the dominance
    verifier error), not only that something raised."""
    table = _FAILS.get(triton.__version__)
    assert table is not None, (f"no recorded expectation for triton {triton.__version__}: run this file and record "
                               f"the outcomes in _FAILS")
    fails = cfg in table[through_cat]
    spelling = "read through the cat" if through_cat else "read directly"
    try:
        asm = _compile(cfg, through_cat).asm
    except RuntimeError as exc:
        assert fails, (f"triton {triton.__version__}: {cfg}, {spelling}, is recorded as compiling, yet it failed: "
                       f"{exc}")
        assert "PassManager::run failed" in str(exc)
        assert "does not dominate this use" in capfd.readouterr().err
        return
    assert "cubin" in asm
    assert not fails, (f"triton {triton.__version__}: {cfg}, {spelling}, now COMPILES: the bug lifted here. Record "
                       f"it in _FAILS and re-read this file's docstring for what that frees.")


@pytest.mark.parametrize("cfg", _AT_16)
def test_the_direct_read_compiles_at_xblock_16(cfg, capfd):
    """THE FIX's cell: with OUR rows loaded directly, every XBLOCK-16 config compiles to a cubin."""
    _check(cfg, False, capfd)


@pytest.mark.parametrize("cfg", _AT_16)
def test_the_read_through_the_cat_fails_at_xblock_16(cfg, capfd):
    """The red: two of the three XBLOCK-16 configs fail on the masked reads (the third compiles either way)."""
    _check(cfg, True, capfd)


@pytest.mark.parametrize("through_cat", [True, False])
@pytest.mark.parametrize("cfg", _BELOW)
def test_smaller_blocks_compile_in_both_spellings(cfg, through_cat, capfd):
    """The control: the kernel is a valid one, and the read's spelling is irrelevant below 16."""
    _check(cfg, through_cat, capfd)


@pytest.mark.parametrize("through_cat", [True, False])
@pytest.mark.parametrize("cfg", _ABOVE)
def test_larger_blocks_fail_in_both_spellings(cfg, through_cat, capfd):
    """A sentinel on Triton, NOT the limit of the fix: this cut-down kernel keeps the pre-fix kernel's stored
    partial, and with it XBLOCK 32 and 64 fail whichever way OUR rows are read (the kernel generated at the fix
    stores other partials and compiles at these configs: the module docstring)."""
    _check(cfg, through_cat, capfd)


@pytest.mark.parametrize("through_cat", [True, False])
@pytest.mark.parametrize("cfg", [(16, 8, 2), (64, 8, 4)])
def test_without_the_stored_partial_every_cell_compiles(cfg, through_cat):
    """The other half of the failing shape: drop the prologue's store and the same cells compile in both spellings
    on every recorded Triton, so neither the masked reads nor the block size fails alone."""
    assert triton.__version__ in _FAILS, f"no recorded expectation for triton {triton.__version__}"
    assert "cubin" in _compile(cfg, through_cat, store=False).asm
