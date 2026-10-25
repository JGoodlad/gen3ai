# The B = 2 CUDA inference compile red: Triton 3.4.0 at XBLOCK 16 (2026-10-25)

**Question.** `extractor_compiles_test.py::test_production_arch_compiles_on_cuda` (the production extractor's no-grad
forward compiled for CUDA at B = 2) failed in the 2026-10-25 GPU window at `51c95abc` / `c0d0c971` with
`InductorError ... _worker_compile_triton ... make_ttgir ... RuntimeError: PassManager::run failed`, after 173.7 s
(window log `~/.claude/jobs/x26r/window_c_fail_27.txt`). Which kernel, which commit, and is a launch at risk? Answered
without a GPU lease.

**Method (CPU only, `CUDA_VISIBLE_DEVICES=""`, no CUDA context).** `fake_cuda_codegen.py` runs dynamo and AOTAutograd
on the CPU module (no-grad, the test's perturbed production extractor), rewrites the post-AOT aten graph's devices to
cuda, lifts its five non-scalar constants to inputs, and lowers it with `compile_fx_inner` under faked CUDA device
properties (RTX 3080 Ti, sm_86); every Triton kernel is precompiled in-process (`triton.compile` with a target needs no
device) and each failure recorded. `kernel_configs.py` recompiles one generated kernel per autotune config. Caveat: the
aten graph comes from the CPU trace (CPU decompositions; `_native_multi_head_attention`'s shape inferred by running it
on the CPU), so it can differ from the card's in places; the card confirmation below checks the two that matter.

**Results** (torch 2.8.0+cu126, triton 3.4.0; outputs in `~/gen3ai_archive/triton_crash_2026-10-25/b<B>/`):

| B | kernels compiled | failing |
|---|---|---|
| 1 | 650 | 33, a DIFFERENT class (below) |
| 2 | 656 | **2**: `triton_red_fused__to_copy_add_clamp_cumsum_index_mul_ones_like_reciprocal_sub_where_269` / `_270` |
| 2, counterfactual (`--speed-cum-mm`: the speed mixture's cumsum as a triangular matmul) | 659 | **0** |
| 4 | 656 | 0 |
| 8 (T2's smallest default bucket) | 656 | 0 |
| 64 | 656 | 0 |
| 256 | 642 | 0 |

- Each failing kernel is the speed mixture's fp64 cumsum (`damage_op_speed._opp_speed_mix`, a LOOPED scan over the
  512-speed lattice, `w_12` / `w_13` = `SPEED_MIX[species]`) with the `[B,6]` Beat-Up / max-HP pointwise epilogue
  (`damage_kinds.beatup_base_def`'s harmonic defence, the opponent max-HP formula) fused in. Per config
  (`kernel_configs.py`): the three with `XBLOCK = 16` (warps 2 / 8 / 8) FAIL, `XBLOCK` 1 and 8 compile. At B = 8 the
  same kernel autotunes at `XBLOCK` 1 / 8 / 64 and all compile.
- The failing pass is `TritonGPURemoveLayoutConversions` (`MLIR_ENABLE_DUMP=1`: the last pass dumped), the verifier
  error `operand #1 does not dominate this use`. Deleting the scan loop, the `device_assert`s, the `debug_barrier`s, or
  making the fp64 fp32 changes nothing: the `[XBLOCK, 1]`-shaped epilogue alone at `XBLOCK = 16` fails. Line deletion
  cut it to the 45-line kernel pinned in `src/agents/model/triton_xblock16_remat_test.py` (fails at `XBLOCK` 16 for
  2 and 8 warps; compiles at 4 / 8 / 32 / 64). triton 3.1.0 (torch 2.5.1, `gen3ai_stable`) compiles it at 16: a Triton
  regression.
- **Commit.** The epilogue ops predate the last pass (`336ddd27a`, 10-03); the cumsum that makes them a 2-D fused
  epilogue came with `50b034a73` (10-07, the discrete speed mixture, behind `--speed-physics on`) and reached the test's
  graph at `ed58a3ad6` (10-12, `production_config.json` takes `speed_physics: on`); `38a21c0fc` (10-15) made it
  unconditional. Not `fe16d565` (move resolution): none of the failing kernels' source nodes come from it.
- **B = 1 is a second, unrelated torch 2.8 / triton 3.4 failure**: 33 kernels with `xnumel = 1` (specialised to a
  constexpr) die in Triton's FRONTEND, `tl.broadcast_to(<constant index>, [XBLOCK])` raising `AttributeError:
  'constexpr_type' object has no attribute 'is_block'`. No path compiles the extractor at B = 1 on CUDA (T2's buckets
  are >= 8; `maybe_compile_extractor`'s B = 1 compile is CPU, `hide_cuda=True` at its one CUDA-capable caller, the
  prober), so it is recorded, not fixed. UNVERIFIED on the card.

**Launch risk: none.** T2 compiles the served forward only at its buckets, `>= MIN_BUCKET = 8` by default
(`agents.training.rust_rollout.build`); R1 compiles a train-mode forward + backward, whose B = 2 cell passed in the
same window; the live `rb_x26c_s2601` (forward code identical to HEAD) brought T2 up at (8, 64, 256). A typed
`--t2-buckets` with 2 would fail LOUD at T2 startup.

**Card confirmation (2026-10-25, window 2).** The red reproduced on the card with `TORCH_LOGS=output_code` (1 failed
in 181 s, `~/gen3ai_archive/extractor_crash_2026-10-25/extractor_stdout_stderr.txt`). The card's generated
`..._where_269` and `..._where_270` are statement-for-statement identical to this harness's (`size_hints` x 16,
`xnumel = 12`, `r0_numel = 512`). The worker-pool traceback names no kernel; these are the only two of the harness's
656 that fail, so they are the card's failing kernel (the card generated 658 definitions; the two extra are not
identified).
