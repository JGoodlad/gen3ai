# The CPU Inductor tile2d codegen bug: raw reads (2026-10-15)

Provenance for `designs/training/compile_flags.md` "The CPU Inductor tile2d codegen bug". Every script is standalone
(no repo import; each sets `CUDA_VISIBLE_DEVICES=""` and asserts `not torch.cuda.is_available()`), run on the CPU box
under `nice -n 19 taskset -c 5-7,13-15`, `OMP_NUM_THREADS=1`, torch 2.8.0+cu126, g++ 15, AVX-512 (`-D CPU_CAPABILITY_AVX512`
in the failing g++ command line). Outputs below are the verbatim lines each printed.

The failing kernel itself was read from a full repro of `compiled_first_update_test`'s CPU cell (`_compiled_first_update
("cpu", gate=False)`) with a persistent Inductor cache (`.main.cpp`, 3,681 lines; the error at line 2003) and, for the
op that feeds it, a second run under `TORCH_COMPILE_DEBUG=1` (`ir_post_fusion.txt`: the 25-node fused group; `op3858` =
`buf3858[b,k] = buf3799[b, buf21[b], k] > 0`, buf3799 = the `[16, 6, 18]` `pair_type_mult` concat). Those dumps are 15-60 MB
and are not committed.

## `ingredients.py` — which ingredients the smallest repro needs (B = 16, K = 18, no options)

```
m1: FAIL  UNDECLARED tmp8      gather + strided-column [B] reduction + the gather stored   <- the 4-line repro
m2: FAIL  UNDECLARED tmp8      two reductions
m3: FAIL  UNDECLARED tmp8      the gather not stored
m4: OK    d=9.5e-07            a CONTIGUOUS [B,K] operand instead of the strided column
m5: FAIL  UNDECLARED tmp8      the bool gather (x[ar, act] > 0) as in move_facts
m6: OK    d=0.0                no reduction (pointwise only)
```

## `move_facts_shaped.py` — `arange` built in the graph (v1-v3 FAIL `tmp8`); a CAPTURED arange tensor (an earlier
variant of the same script, `ar` global) compiled OK: the batch coordinate was then indirect as well and the load stayed
untransposed.

## `shapes.py` — the shape threshold of m1

```
B=16 K=18: FAIL   B=8 K=18: OK    B=15 K=18: OK     B=17 K=18: FAIL   B=32 K=18: FAIL   B=64 K=18: FAIL
B=16 K=8: OK      B=16 K=16: FAIL B=16 K=17: FAIL   B=16 K=32: FAIL   B=256 K=18: FAIL  B=256 K=64: FAIL
```

## `config_candidates.py` — Inductor configs on m1

```
base: FAIL tmp8      vec_off (cpp.vec_isa_ok=False): OK d=4.8e-07, transpose_mxn=0, vec_loads=0
simdlen1 (cpp.simdlen=1): OK d=4.8e-07, transpose_mxn=0, vec_loads=0     heur_off (cpp.enable_tiling_heuristics=False): FAIL tmp8
maxfusion1 (max_fusion_size=1): OK d=4.8e-07, transpose_mxn=2, vec_loads=7 (unfused, still tiled)
```

and through `torch.compile(options={"cpp.simdlen": 1})`: `OK d=4.8e-07 transpose_mxn=0 vec_loads=0` (the path
`compile_regions.install` uses). (`no_fuse_pw_red` in the script guards a config attribute that this torch does not have;
its row ran the base config and is not a measurement.)

## `respellings.py` — would respelling the gather (option b) help?

```
eager: one-hot == index bitwise: True | gather == index bitwise: True
index: FAIL UNDECLARED    gather: FAIL UNDECLARED    one-hot: OK
```

## Standalone `move_facts` / `switch_facts`

`MoveResolutionOps` hand-built at B = 16, K = 18 (the `move_resolution_test.ops` builder, expanded, perturbed): the REAL
`move_resolution.move_facts` compiled on CPU, forward `OK max|d|=2.09e-07 (37 s)` and forward + backward `OK
max|d|=1.79e-07 (75 s)` — the failing fusion needs R1's neighbours.

## The full R1 read (the test itself)

| run | result |
|---|---|
| `compiled_first_update_test` CPU cell at `87e493c3` (the sweep) | FAIL, `CompileTrainerError` over `CppCompileError`, 337.9 s |
| the same cell, standalone with a persistent cache | FAIL in 254 s |
| `compiled_first_update_test -m slow` at the fix (the code as first committed) | 1 passed, 1 skipped (the CUDA cell) in 277.6 s |
