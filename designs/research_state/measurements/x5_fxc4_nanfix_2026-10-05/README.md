# F-XC-4 fixed: fixed_mass's compiled R1 backward is finite (2026-10-05, `gen3_fm_index_max_v1`)

**Verdict: FIXED for fixed_mass, blob untouched.** The NaN compiled gradient under `--belief-tokens
fixed_mass` came from `amax`'s backward over a tensor Inductor RECOMPUTES in the backward kernel. Its
mask `x == amax` compares the recomputed `x` with the max saved by the forward kernel, and Triton's
FMA contraction rounded the recompute differently, so on some rows no element matched: tie count 0,
gradient 0/0 = NaN. Under fixed_mass the ten affected channel maxima are now taken at the detached
argmax (`damage_op.max_by_index`): the same value, a scatter backward, no float equality. Tags:
**MEASURED** unless marked. Follows `../x5_fxc4_compile_gate_2026-10-05/` (the gate fixes, the
`remove_noop_ops` bisection).

- **Box.** RTX 3080 Ti, torch 2.8.0+cu126, Triton 3.4.0, fp32 'highest', desktop stopped. Base `84f8569f`.
- **Learner.** The K9 golden learners (`learner_golden.build_learner()` / `build_arm_learner("fixed_mass")`),
  CUDA, R1's gate rows (`compile_regions.r1_batch`), B = 2,048; "fresh" = `PERTURB_SCALE` 0.

## 1. Where the NaN enters (`scripts/nan_kernel.py`)

Every Triton kernel call in the generated wrapper is bracketed by a host hook that counts NaN / inf in
each tensor argument before and after the call (NaN decides; the forward's `-inf` masks are legitimate).
The hook is emitted at CODEGEN, after scheduling and fusion, so the kernels are the real compile's.

| reading | value | file |
|---|---|---|
| first NaN-writing kernel | `triton_red_fused_…_logsumexp_…_where_239`, a BACKWARD reduction kernel, x = 12,288 (B × 6 defenders), r = 400 (the candidate moves) | `results/nan_kernel_fm.json`, `results/first_nan_kernel_239_triton.py` |
| its inputs before the call | NaN 0, inf 0 in every argument | same |
| its output | `buf1111` [2048, 6, 400]: 128 NaN | same |
| what it does | loop 1 RECOMPUTES the incoming direction's rolls (`eff`, `inv_d`, `dmg_cb`, `ko_cb`, `high_cb`, …) and counts `x == saved_amax` for the ten channel maxima (`phys/spec_{low,high,crit,pko}`, `phys_high_cb`, `phys_pko_cb`); loop 2 divides each upstream gradient by its count (`tmp246 = tmp244 / tmp245`, …) | the kernel source |
| the same graph with Triton's FMA contraction OFF (`TRITON_DEFAULT_FP_FUSION=0`, diagnostic) | **0 non-finite parameters** | `results/nan_kernel_fm_fma_off.json` |
| the fix (same harness) | **0 non-finite parameters** | `results/nan_kernel_fm_fix.json` |

The forward kernel computes the same chain in another fusion and takes the max in registers; the
recompute's `a·b + c` chains contract into FMAs differently there (the multiply feeding `dmg_cb - cur_hp`
has a different number of uses in each kernel), so the bits differ and the count can be 0 on a row.
`amax`'s backward then gives `0/0`. Why the 1,974th `remove_noop_ops` removal mattered: it changed which
nodes the partitioner recomputed. The full candidate axis (400) exists only under fixed_mass, which
prices every move (`damage_candidate_k` must be 0); blob prices the top-6. **UNVERIFIED:** which exact
instruction differs (no SASS diff was taken); the FMA-off reading is what pins the class.

## 2. The fix and its scope

`damage_op.max_by_index(x)` = `torch.gather(x, -1, x.detach().argmax(-1, keepdim=True)).squeeze(-1)`,
used for the ten maxima ONLY in the `fixed_moves is not None` branch of the incoming direction
(`fixed_moves` is non-None only when the X5 hypothesis builder runs). Its argmax is K9(b)'s `MAX_VALUE`
EXACT site: the index is read only by a gather of its own operand, so the value is `amax`'s in every
forward (`selection_sites_test` pins the gather).

- **Value:** bit-identical to `amax` (CPU tests on the K9 golden rows: the fixed_mass forward with
  `max_by_index` equals the `amax` spelling bit for bit).
- **Gradient:** identical off ties; on an exact tie the whole gradient goes to the FIRST maximal element
  (`amax` splits it evenly). The fixed_mass K9 golden (initial and post-update bytes, every pinned loss)
  did NOT move.
- **Blob:** never calls it (a CPU test counts the calls: blob 0, fixed_mass 10 per forward).

## 3. Blob is byte-identical (`scripts/codehash.py`, `scripts/compare_dumps.py`, `scripts/identity.sh`)

FX and AOT caches OFF, so every artifact is generated in the process. Per arm and region: dynamo's FX
graph code (`gm.code`) and every Inductor output module (wrapper + Triton kernel sources). R1 =
`micro_step` forward + backward on the gate rows; T2 = the inference service's `decide` over a slot's
`DecisionModule` (the `graph` backend's callable) at bucket 64. HEAD = the six changed non-test source
files put back to `84f8569f` in the same worktree (restored after), so paths are identical. Two COMMENT
normalisations only: the per-process compile temp dir in `# kernel path:` lines, and the host address
ending a `_tensor_constantN = None  # …` line.

| arm · region | modules (HEAD / fix) | identical | combined sha (both) |
|---|---|---|---|
| **blob · R1** | 32 / 32 (1 dynamo graph + 31 Inductor) | **YES** | `dee5ac1f16a8…` |
| **blob · T2** | 3 / 3 | **YES** | `4156f57a6ac8…` |
| fixed_mass · R1 (teeth) | 17 / 16 | no (12 / 11 differ) | — |
| fixed_mass · T2 (teeth) | 3 / 3 | no (3 / 3 differ) | — |

`results/blob_identity_head_vs_fix.json`, `results/fixed_mass_head_vs_fix.json`, `results/codehash_*.{log,json}`
(the raw, un-normalised hashes; `R1_grad_finite` false for fixed_mass at HEAD, true at the fix). The blob
K9 golden passes at the fix.

## 4. CUDA: the gate passes, and still bites

| check | result | file |
|---|---|---|
| the real startup gate (`gate_regions`), 4 FRESH processes (2 fresh-init: the live + seeded-perturbation rungs; 2 golden-perturbed weights: the trained rung) | **4 / 4 PASS**; compiled gradient finite on every rep; cosine 1 − 1e-13 (live), 1 − 3.4e-10 (perturbed); per-parameter rel err max 1.37e-5 / 1.34e-5 (fresh, bar 0.1004), 1.16e-5 / 1.31e-5 (seeded rung, bar 0.00988), 8.78e-6 / 7.84e-6 (trained, bar 0.00988) | `results/gate_fix_fm_*.txt` |
| `compile_regions_fixed_mass_cuda_test` at the fix (finite + the real gate + a planted NaN named `NonFiniteGateArmError` + a planted 10 % gradient miscompile FATAL, on the real compiled graph) | PASS, 143 s | `results/cuda_test_at_fix.log` |
| the same test with the source at `84f8569f` | **FAIL**: "the compiled R1 gradient is NON-FINITE" | `results/cuda_test_at_head.log` |

## 5. End to end: a fixed_mass regime-A run, compiled

`e2e/e2e.sh` (the cost ablation's argv: `--arch production`, the X26 heads, `--belief-tokens fixed_mass`,
`--behaviour-check warn`, eval off; a seeded 20-snapshot pool built from the F-XC-5 unit's `fmB5`
checkpoint — regime A; `--compile-trainer` ON by default; `drive.sh` stops it by PID after 12 update
rows). Measurement-only overrides `PROF_FIT_HEADROOM_MIB=256`, `PROF_SLOT_LOAD_TOL_MIB=8` (F-XC-2 /
F-XC-3 stand); **no** `PROF_LOG_KEY_MAX` — the production stdout table.

| check | result |
|---|---|
| startup gate (in the launch) | **PASS**: fresh rung loss rel 0, grad cosine 1.000000, per-parameter 2.28e-5 over 147 parameters (bar 0.1004); seeded-perturbation rung 4.11e-5 over 218 (bar 0.00988) |
| update-10 canary | **PASS** (`e2e/canary_verdicts.jsonl`: update 10, step 990,874): loss rel 0, cosine 1.000000, per-parameter 1.56e-5 |
| logger dumps | **11** stdout tables (the dry update's + 10), every `opp_intent/flat_switch_tgt_top1{,_bot,_pool}` row rendered; no crash |
| NaN | none in any loss / gradient-norm row; the trainer ran 11 real updates and stopped on the driver's SIGTERM |
| K9(b), now against the COMPILED learner | excluded share 0 (update 1, selection-free), then 0.078–0.105 on updates 2–11 (ceiling 0.15); max \|Δ log π\| 7.2e-7 on judged rows, 3.6e-7 on excluded |

The F-XC-5 read's deviation (1) — the learner eager — is gone: the same share reads under the
compiled learner. GPU time: 26 min (launch to stop). ⚠️ `train/contested_rel_gap_b*` printed `nan` on
some updates; the same keys printed `nan` in the F-XC-5 eager run at `84f8569f` (`fmA5b.log`), so it is
pre-existing and is not a gradient (it is a diagnostic table row; **UNVERIFIED** whether it is an
empty-bucket NaN by design).

## Tests that fail on revert

- `src/agents/model/compile_regions_fixed_mass_cuda_test.py` (GPU tier, `slow`): §4.
- `src/agents/model/damage_op_index_max_test.py` (CPU): fixed_mass selects by index 10× per forward
  (0 on revert); blob never does; the value is `amax`'s bit for bit; the tie rule.
- `src/agents/model/selection_sites_test.py::test_every_max_value_sites_index_is_read_only_by_a_gather_of_its_own_operand`.
- `src/agents/model/flat_intent_test.py::test_every_flat_fold_key_of_every_opponent_class_renders_uniquely_in_the_stdout_table`
  (the logger collision: `ValueError` on revert).
- `src/agents/model/compile_regions_independence_test.py::test_the_CANARY_names_a_dependent_pair_as_such_never_as_a_CONFIRMED_disagreement`.

## Reproduce

`scripts/gpu_run.sh <GB> <timeout_s> <log> <script> [args]` (lease token, `scripts/ops/gpu_lock.sh`,
`scripts/ops/mem_cap.sh`; written for this unit's worktree — edit `W`).

```bash
gpu_run.sh 24 1800 k.log scripts/nan_kernel.py fixed_mass out_prefix            # first NaN kernel
TRITON_DEFAULT_FP_FUSION=0 gpu_run.sh 24 1800 k.log scripts/nan_kernel.py fixed_mass out_prefix
scripts/gate4.sh fix                                                             # 4 fresh gate processes
NO_TEST=1 scripts/identity.sh                                                    # code hashes, fix vs HEAD
python3 scripts/compare_dumps.py <dump_head_blob> <dump_fix_blob> out.json
```
