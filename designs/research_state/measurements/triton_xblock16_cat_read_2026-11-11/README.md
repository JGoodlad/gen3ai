# The B = 2 CUDA forward + backward compile red: Triton 3.4.0 at XBLOCK 16, a second kernel (2026-11-11)

**Question.** `extractor_compiles_test.py::test_production_arch_compiles_forward_and_backward_on_cuda` (the production
extractor's train-mode forward + backward compiled for CUDA at B = 2) failed twice, deterministically, on the card at
`aee363bc9` with `InductorError ... make_ttgir ... RuntimeError: PassManager::run failed` (torch 2.8.0+cu126,
triton 3.4.0; `~/.cache/gen3ai/b1_derisk_2026-11-11/results/rerun_extractor_fwdbwd/`). It passed at `96f661e90` on
2026-11-05. Which kernel, which commit, what fixes it without changing a number, and is a launch at risk? Answered
without a GPU lease.

**Method (CPU only, `CUDA_VISIBLE_DEVICES=""`, no CUDA context).** `fake_cuda_codegen_fwdbwd.py` is the 2026-10-25
record's harness (`../triton_xblock16_2026-10-25/`) extended to the backward graph: dynamo and AOTAutograd run on the
CPU module, each post-AOT aten graph (forward, then backward) has its devices rewritten to cuda and is lowered by
`compile_fx_inner` under faked CUDA device properties (RTX 3080 Ti, sm_86), and every Triton kernel is precompiled
in-process (`triton.compile` with a target needs no device). `triton_diag_plugin.py` records each failing kernel
(source, `ASTSource` inputs, every autotune config recompiled alone); the same file is a pytest plugin for a GPU run.
`kernel_configs.py` recompiles one saved kernel per config. One screen is 7 to 10 minutes at nice 19:

    PYTHONPATH=src python3 designs/research_state/measurements/triton_xblock16_cat_read_2026-11-11/fake_cuda_codegen_fwdbwd.py \
        --batch 2 --mode train --out DIR          # --mode nograd: the inference cell's graph

Caveats, as on 10-25: the aten graph comes from the CPU trace, so it can differ from the card's in places; the harness
covers the extractor's graphs only (never T2's full policy graph or the learner's bf16 R1); it is a diagnostic, not a
gate. Do not delete a screen's `--out` caches while another screen is running (one screen was spoiled that way here
and re-run).

**Results** (torch 2.8.0+cu126, triton 3.4.0; "failing" counts kernels, `PassManager::run failed` unless noted; logs in
`~/.cache/gen3ai/triton_fwdbwd_2026-11-11/cpu_screens/`):

| checkout | graph | B | kernels compiled | failing |
|---|---|---|---|---|
| `96f661e90` (the last card pass) | forward + backward | 2 | 1,061 | 0 |
| `83272a8f9` | forward + backward | 2 | 1,061 | 0 |
| `fc699953a` (the parent) | forward + backward | 2 | 1,061 | 0 |
| **`7a0153b46`** (`gen3_stat_ev_points_v1`) | forward + backward | 2 | 1,063 | **1** |
| `e112cbcbf` | forward + backward | 2 | 1,059 | **1** (`failing_kernel_b2.txt`) |
| `e112cbcbf` | forward + backward | 8 | 1,070 | 0 |
| `1fe7283f7` (still before the fix) | forward + backward | 4 | 1,063 | **1**, the same kernel at `XBLOCK` 32: (32, 8, 2) and (32, 64, 16) |
| `1fe7283f7` | forward + backward | 6 | 1,064 | 0 (the roll kernel is formed WITHOUT the max-HP prologue) |
| `e112cbcbf` | no-grad forward | 2 / 16 | 650 / 650 | 0 / 0 |
| `1fe7283f7` | no-grad forward | 1 | 650 | 33, a DIFFERENT class (Triton's frontend, `is_block`; the 10-25 record) |
| **the fix** | forward + backward | 2 / 3 / 4 / 5 / 6 / 8 | 1,058 / 1,054 / 1,062 / 1,065 / 1,063 / 1,069 | **0** at each |
| **the fix** | no-grad forward | 2 / 3 / 4 / 8 / 16 | 649 / 649 / 649 / 649 / 649 | **0** at each |
| the fix | no-grad forward | 1 | 645 | 34, the same frontend class, none `PassManager` |

- **The kernel** is `triton_red_fused__to_copy_abs_add_bitwise_or_ceil_clamp_..._sum_where_405`
  (`failing_kernel_b2.txt`, its `ASTSource` inputs in `failing_kernel_b2.meta.json`): the non-formula damage kinds' roll
  (`damage_kinds.nonformula_rolls`, a LOOPED reduction over the 400 moves, `r0_numel` 400) with OUR `[B,6]` max-HP
  chain (`damage_kinds.our_max_hp`) fused in as its `[XBLOCK, 1]` prologue. Of its five autotune configs (XBLOCK,
  R0_BLOCK, warps), (16, 8, 2) and (16, 64, 8) fail `TritonGPURemoveLayoutConversions` ("operand #0 does not dominate
  this use"); (1, 512, 4), (8, 512, 16) and (16, 4, 8) compile. `python3 kernel_configs.py failing_kernel_b2.txt
  failing_kernel_b2.meta.json` replays it. It is the 10-25 bug, reached through a different kernel.
- **Commit.** `7a0153b46` put the EV floor into OUR stat chain; its parent has 0 failing kernels. The chain's reads
  were already through the cat; the added floor is what tipped this kernel's fused shape into the failing one.
- **What decides it at 16.** The prologue read OUR IV and EV as ``ctx.pokemon_part[:, :TEAM_SIZE, ...]`` on a
  HYPOTHESIS context, whose block is ``cat([ours, where(hidden, dex_row, theirs)])``. Inductor lowers the cat
  pointwise: each read is three masked loads and a select chain. Hand-editing the generated kernel so the two reads
  load OUR rows directly makes both failing configs compile. No rewrite inside the max-HP arithmetic did (the sum
  reassociated, a different partial stored, the Shedinja select moved): the fix has to sit at the READ.
- **Per config: the generated kernels.** `kernel_configs.py` over the nine `XBLOCK` 16 / 32 / 64 autotune configs:

  | (XBLOCK, R0_BLOCK, warps) | before the fix (`failing_kernel_b2.txt`; the B = 4 kernel agrees) | at the fix (`fixed_kernel_b2.txt`; the B = 4 and B = 6 kernels agree) |
  |---|---|---|
  | (16, 4, 8), (32, 4, 8) | compiles | compiles |
  | (16, 8, 2), (16, 64, 8), (32, 8, 2), (32, 64, 16), (64, 4, 8), (64, 8, 4), (64, 64, 16) | **FAILS** | compiles |

  So the red was never B = 2 alone: before the fix the graph formed this kernel at B = 2 and at B = 4 (both fail), and
  not at B = 6 (B = 8: 0 failing, its kernels not kept). At the fix it forms the fused kernel at each of B = 2 to 6
  (B = 8 not inspected), and every config compiles. `XBLOCK`
  16 / 32 / 64 are autotuned when the `6B` rows round to them (B = 2 / 3 to 5 / 6 to 8).
- **Per config: the hand-cut kernel** (`src/agents/model/triton_xblock16_cat_read_test.py`: the pre-fix kernel cut
  down, the two reads behind one switch and nothing else changed):

  | (XBLOCK, R0_BLOCK, warps) | read through the cat | read directly | either, the stored partial dropped |
  |---|---|---|---|
  | (1, 512, 4), (8, 4, 8), (8, 8, 2), (8, 64, 4), (8, 512, 16), (16, 4, 8), (32, 4, 8) | compiles | compiles | compiles |
  | (16, 8, 2), (16, 64, 8) | **FAILS** | compiles | compiles |
  | (32, 8, 2), (32, 64, 16), (64, 4, 8), (64, 8, 4), (64, 64, 16) | FAILS | FAILS | compiles |

  It matches the pre-fix generated kernel in its through-the-cat column. Its direct column is NOT the kernel at the
  fix: it keeps the pre-fix kernel's stored partial of the stat sum, which the kernel generated at the fix does not
  store. The failing shape is therefore a stored prologue value whose chain feeds the loop, made worse by the masked
  reads; the direct read is necessary and, for the graphs screened, sufficient, but it is not a guarantee for a new
  kernel of this shape. Triton 3.1.0 (torch 2.5.1, `gen3ai_stable`) compiles every cell: a Triton regression.
- **The fix.** `damage_kinds.our_team_spread(ctx)` reads OUR rows off the REAL context (`extractor_ctx.our_rows`; a
  hypothesis context carries `real`). Bit-identical: `src/agents/model/our_rows_spelling_test.py` swaps the old
  spelling back at all five sites and compares 168 tensors (915,888 elements: the damage op's block, the enriched
  opponent tokens, both stashes, the extractor's outputs) on golden states where 30 of 96 opponent slots are
  hypothesis slots; a control shows the comparison sees a changed spread.
- **Is a launch at risk?** No. T2 compiles at buckets `>= MIN_BUCKET = 8` by default, R1 at `batch_size` rows; before
  the fix the kernel was formed only in the forward + backward graph at B = 2 and B = 4 (the no-grad forward at
  B = 2 and 16 had 0 failing kernels). T2's own policy graph was not screened at any bucket.

**On the card: OPEN.** Nothing here ran on a GPU. `~/.cache/gen3ai/triton_fwdbwd_2026-11-11/gpu_confirm.sh` is the one
combined run (the cell as written at `aee363bc9` with the plugin, to confirm the kernel's identity; the cell at the
fix, which must pass; the batch-8 inference sibling at the fix).
