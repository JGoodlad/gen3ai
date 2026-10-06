# X5's hypothesis-token encoding, split exactly and gathered (2026-10-05, `gen3_x5_hyp_gather_v1`)

The cost cut proposed after the X5 cost ablation (`../x5_cost_ablation_2026-10-04/` §3: the hypothesis
`PokemonEncoder` pass was the largest single X5 piece). Under the GPU lease (owner `x5-perf-encode`). Tags:
**MEASURED** unless marked. Base `c7b4d03e`.

- **Box.** RTX 3080 Ti (12 GiB), torch 2.8.0+cu126, Triton 3.4.0, fp32 'highest', desktop STOPPED (the T23 rule;
  `nvidia-smi` read 33 MiB used before the first job).

## Verdict

| | before (`c7b4d03e`) | after | budget | verdict |
|---|---|---|---|---|
| hypothesis encoding, ms per micro-batch (compiled fwd+bwd, B = 2,048) | 10.09 | 4.38 | — | **−57 %** |
| X5 on the extractor harness (vs blob 72.17 ms) | +20.72 ms (+28.7 %) | +15.02 ms (+20.8 %) | — | −5.71 ms |
| **`train_ms`** (fixed_mass regime A vs blob, quiet medians) | 49.16 s = **+19.5 %** | 46.39 s = **+12.8 %** | ≤ +5 % | **OVER (2.6×)** |
| **T2 service per host step**, regime A (flush + GPU wait) | 31.80 ms = **+46.1 %** | 31.68 ms = **+45.6 %** | ≤ +3 % | **OVER** |
| **D-6 headroom** (`UpdateFit`, regime A, X26 heads) | 1,106 MiB | **1,192 MiB** | ≥ 1,024 | **PASS** (168 MiB margin) |

The brief's premise is FALSE, so the cut is about half of the pass, not most of it (§1). Two of the three budget
lines still fail; T2's X5 cost is not in this encoder at all (§3).

## 1. The premise: is a hypothesis token a function of its species alone? NO

`scripts/premise.py` (CPU, the production fixed_mass learner with the gates' seeded perturbation, 1,024 real rows
from the Rust collector — the cost ablation's `gen_real_obs.py`), `results/premise/premise_fp32_B1024.json`:

| reading | value |
|---|---|
| hidden slots holding a hypothesis | 752 (of 6,144 opponent slots: 12 %) |
| species seen in ≥ 2 rows: tokens bitwise equal across rows | **0 of 18**; max \|Δ\| **1.06** |
| the same, with the encoder's row-level inputs held at row 0's values | **18 of 18 bitwise equal**, max \|Δ\| 0.0 |
| a hypothesis slot that is the opponent's ACTIVE | 0 of 752 |
| row-level inputs that vary across these rows | clock, weather, fainted, hazards, the opponent active context (screens did not vary here) |

So the encoding depends on exactly these ROW-level inputs besides the species: the clock / weather / fainted /
hazard features (the move network's per-move context) and, with the screens, the role encoder's broadcast global
context, plus the active-context scatter (zero for a hypothesis, which is never the active). They pass through the
move network's nonlinearity and its within-mon self-attention, so no per-species encoding exists past the first layer.

**The exact split.** Each row-level input enters only through the FIRST `Linear` of the move network and of the
role encoder, and a Linear is additive over input columns: `W·[x_s; x_r] + b = (W_s·x_s + b) + W_r·x_r`.
`agents/model/hypothesis_encode.py`:

- **species half**, once per forward over the 400-row dex table (`species_table`): every species-only column through
  each first Linear, the row-level columns zero → `move_pre [S,4,96]`, `role_pre [S,256]`; gathered by the
  hypothesis species. When B·6 < 400 (T2's buckets 8 and 64) the slots' own dex rows are encoded instead (fewer
  rows than the table; a static branch, B is fixed per compiled graph).
- **row half**, once per row: the four context features against their move-network columns, the five global features
  and the opponent active context (scattered to its slot) against their role-encoder columns, no bias.
- **the rest per OPPONENT slot only** (6, not 12): ReLU, the second Linear, the move self-attention + norm, the
  processed moves' role columns, ReLU, the second Linear. Unchanged ops.

The per-row pass (`PokemonEncoder.forward` on `hypothesis_ctx`) stays the definition; the tests compare against it.

## 2. Exactness

| comparison (every hypothesis slot) | fp32 max \|Δ\| | fp64 max \|Δ\| | file |
|---|---|---|---|
| 1,024 real rows (table branch) | 2.86e-6 (token values up to 5.84; rel 4.9e-7) | 6.2e-15 | `results/premise/equiv_*_B1024.json` |
| 8 real rows (per-slot branch) | 1.91e-6 | 4.4e-15 | `results/premise/equiv_*_B8.json` |

**Declared bound: 2e-5 absolute at fp32, 1e-12 at fp64** (`hypothesis_encode_test`); the gradients of every encoder
and embedding parameter agree at fp64 within 1e-12 × scale, on every opponent slot including a planted active one.

**The K9 fixed_mass golden moved (reassociation) and is re-recorded**, by the three-part method of
`../move_prior_golden_2026-10-04/`:

| step | what | result |
|---|---|---|
| A | the recorded golden is the parent's | it is `c7b4d03e`'s committed entry (`learner_golden_fixed_mass_test` green on main) |
| B | this tree with the per-row pass patched back in (`scripts/gproof.py --perrow`, measurement only) vs the recorded golden | **99 / 99 fields identical** (`results/golden/B_final_perrow.json`) |
| C | this tree as built | init params identical; post `c1ab0fbf…` (was `50659234…`); every loss moves ≤ 6.7e-8 (`results/golden/fix_final.json`); the fp64 reference in `learner_golden_fixed_mass_test` passes |

So the update moves ONLY through the hypothesis encoding, and only at fp32 rounding. The blob entry is untouched.

## 3. Cost

**Extractor harness** (`scripts/ablate2.py` = the cost ablation's `ablate.py` unchanged + three measurement-only arms;
B = 2,048, compiled fullgraph fwd + bwd, median of 5 × 20, replicate 2 in reversed order; `results/ablation/`):

| arm | ms per micro-batch (r1, r2) | peak alloc MiB | T2 replay ms at B = 8 / 64 / 256 |
|---|---|---|---|
| blob | 72.28, 72.05 | 3,906 | 1.210 / 1.908 / 4.148 |
| `fm_perrow` (BEFORE: the per-row pass patched back in) | 92.81, 92.97 | 4,286 | 1.986 / 2.864 / 6.090 |
| `fm` (AFTER, table branch at B = 2,048) | 87.20, 87.17 (final tree 86.83) | 4,098 | 1.995 / 2.850 / 5.855 (final tree) |
| `fm_nohyp` (the FLOOR: hypothesis tokens dead, Inductor drops the encoding) | 82.81, 82.80 | 4,013 | 1.885 / 2.700 / 5.656 |

The encoding's own cost (arm − floor): **train 10.09 → 4.38 ms (−57 %)**; T2 per forward 0.10 → 0.11 (B = 8),
0.16 → 0.15 (64), 0.43 → 0.20 (256). Profiler (3 steps, `prof_*.json`): what remains is GEMM +3.0 ms (the skinny
per-slot second layers and their weight gradients), pointwise +1.0, 158 more kernel launches than the floor. The
gather's backward does not appear in the top 30 kernels.

**Launches** (the nanfix e2e's argv: `--arch production`, the X26 heads, `--belief-tokens fixed_mass`,
`--behaviour-check warn`, eval off, a seeded 20-snapshot pool built from the F-XC-5 unit's `fmB5` checkpoint
(regime A); `drive.sh` stops it by PID after 12 update rows; measurement-only `PROF_FIT_HEADROOM_MIB=256`,
`PROF_SLOT_LOAD_TOL_MIB=8` — F-XC-3 stands; quiet updates only, dry / first / canary excluded):

| launch | tree | quiet updates | `train_ms` median | T2 flush + wait ms / host step | `UpdateFit` headroom | steady peak alloc | startup |
|---|---|---|---|---|---|---|---|
| `fmA_nanfix` (BEFORE; banked by the nanfix unit, `../x5_fxc4_nanfix_2026-10-05/e2e/`) | `c7b4d03e` | 9 | 49.16 s | 1.95 + 29.85 = 31.80 | 1,106 MiB | 5,319 MiB | 801 s |
| **`fmA_gather`** (AFTER) | this | 9 | **46.39 s** | 1.91 + 29.78 = **31.68** | **1,192 MiB** | 5,131 MiB | 724 s |
| `blobB` (same-day blob) | this | 6 (2 contended, excluded) | **41.12 s** | 0.56 + 4.35 = 4.91 (regime B) | 2,874 MiB | 4,843 MiB | 389 s |

- K9(b) recorded 0 violations in every update of both fixed_mass launches, so neither `train_ms` carries the warn
  path's scan; the update-10 canary PASSED in both; no NaN.
- **Blob regime-A T2 is CARRIED, not same-day:** 21.76 ms from the cost ablation (`889add9d`). Blob's T2 compiled code
  is byte-identical since (this unit's and the nanfix unit's hash `4156f57a6ac8…`) and blob's regime-B T2 today reads
  4.91 ms against the ablation's 4.90 (time-boxed: no second blob launch). **UNVERIFIED** as a same-day pair.
- `train_ms` compares fixed_mass regime A with blob regime B; the ablation measured blob's update regime-independent
  (−0.2 %).

## 4. Blob is byte-identical (`scripts/identity.sh`, the nanfix unit's `codehash.py` / `compare_dumps.py`)

FX and AOT caches off; per arm and region, dynamo's graph code + every Inductor output module, at HEAD (the three
changed non-test source files put back) and at the fix:

| arm · region | modules | identical | combined sha |
|---|---|---|---|
| **blob · R1** | 32 / 32 | **YES** | `dee5ac1f16a8…` (= the nanfix unit's) |
| **blob · T2** | 3 / 3 | **YES** | `4156f57a6ac8…` (= the nanfix unit's) |
| fixed_mass · R1 | 16 / 16 | no (teeth) | `e3d6677f…` → `689688d4…`; compiled gradient finite at both |
| fixed_mass · T2 | 3 / 3 | no (teeth) | `b4820a86…` → `583277c3…` |

The blob K9 golden passes; `compile_regions_fixed_mass_cuda_test` (GPU tier) PASSES at the fix (148 s,
`results/cuda_test.log`).

## Tests that fail on revert

- `src/agents/model/hypothesis_encode_test.py::test_the_fixed_mass_forward_gathers_once_and_runs_the_encoder_once`
  — at HEAD it fails (`results/identity/unit_tests_at_head.log`; the symbol is absent there, so it fails at the
  patch; a revert that kept the import and restored the per-row pass fails the count: 2 encoder passes).
- the same file: blob never reaches the path; the values on both static branches (fp32 ≤ 2e-5, fp64 ≤ 1e-12); every
  slot incl. a planted active one, values and gradients at fp64 (these compare the module against the per-row pass,
  so they fail on any drift between the two).
- `learner_golden_fixed_mass_test` (re-recorded: the pre-change entry fails on this tree, the new one on the old).
- `compile_regions_fixed_mass_cuda_test` (GPU tier): compiled fixed_mass R1 finite + the real gate — green.

## Findings (standing rule 7)

- **F-HG-1 (premise).** A hypothesis token is NOT a function of its species: the clock, weather, fainted, hazard and
  screen features enter it (max spread 1.06 across rows for one species). The exact split cuts 57 % of the encoding,
  not all of it.
- **F-HG-2 (budget).** `train_ms` +12.8 % (≤ +5 %) and T2 +45.6 % (≤ +3 %) still FAIL. T2's X5 cost is not here: the
  encoding is 0.1–0.2 ms of the 0.8–1.7 ms X5 adds per T2 forward; the ablation's residual + OTHER pass own it.
- **F-HG-3 (D-6 now passes, and passed at HEAD with the desktop stopped).** 1,106 MiB at `c7b4d03e`, 1,192 here,
  against the cost ablation's 432 (`889add9d`). That ablation's desktop state is not recorded; `gnome-shell` held
  811 MiB on this card before T23 (UNVERIFIED as the cause of the difference). The margin is 168 MiB.
- **F-HG-4 (further lever, not exact under a static cap).** Only ~12 % of opponent slots hold a hypothesis on real
  rows, but static shapes make every per-slot op run on all 6. Compacting needs dynamic shapes or a capped buffer with
  an overflow path.
- **F-HG-5 (determinism).** The gather's backward accumulates with atomics in the compiled code (fp32 sums in
  nondeterministic order), the same class as every embedding lookup already in the graph.
- **F-HG-6 (small buckets).** The per-slot branch for B·6 < 400 gives no measurable T2 gain at B = 8 / 64 (one
  replicate; T2 is launch-bound there).
- **F-HG-7 (comparators).** The BEFORE launch is the nanfix unit's (same argv, pool template, box state, a few hours
  earlier); the blob regime-A T2 comparator is carried from `889add9d` (§3).
- **F-HG-8 (ARCH_SIGNATURE).** Not bumped: the network computes the same function (fp32 reassociation), so every
  fixed_mass checkpoint loads and means the same.

## Reproduce

Scripts in `scripts/` (written for this unit's worktree and scratch paths; edit `W`). `gpu_run.sh` = lease token +
`scripts/ops/gpu_lock.sh` + `scripts/ops/mem_cap.sh`. CPU: `premise.py`, `equiv.py <B> <dtype>`, `gproof.py
[--perrow]`. GPU: `ablate_seq.sh` (blob / fm_perrow / fm, two replicates), `ablate2.py --arm fm_nohyp`, `identity.sh`,
`launches.sh` (needs the cost ablation's `drive.sh`, `phase_probe_run.py`, `cpu_sampler.py`, `build_pool.py`,
`read_run.py` in the scratch dir).
