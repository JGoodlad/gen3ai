# K9(b) deterministic tie exclusion — the measurement (2026-10-01)

The numbers behind `gen3_behaviour_tie_exclusion_v1` (`designs/training/learner_gates.md` § (b), the
fp32 rule), re-derived by `consistency_test::test_the_fp32_rule_numbers_are_derived_from_the_banked_measurement`.

| file | what |
|---|---|
| `sweep.py` | the GPU driver: the tail sweep's setup (`../k9_behaviour_tail/`) with every row's TIE MARGIN, the forward-variant margin comparison, `--faults`, `--cost`, `--precision high` |
| `derive.py` | the DECLARED criteria (epsilon = round_up_125(10 × R); the ceiling rule) and the verification → `result.json` |
| `result.json` | epsilon, ceiling, rounding scale, verification, the margin-decade table, the epsilon grid, faults, cost; the TF32 block |
| `fills_fp32.jsonl`, `fills_tf32.jsonl`, `extras_fp32.jsonl` | the per-fill summaries the derivation reads (the per-row `fill_<n>.npz` arrays are in `~/gen3ai_archive/k6_k8/k9excl/`) |

**Setup.** A2's 4.0M checkpoint and its 2 snapshots (`~/gen3ai_archive/k6_k8/k9tail/a2`), the production
rust path (T2 graph backend, buckets 8 / 48, 7 lanes, N = 48, fills of 98,304 rows), nothing training
(every row current), the learner's probe forward eager / train mode / grad on, 1,024-row chunks. torch
2.8.0+cu126, RTX 3080 Ti. fp32: 36 fills, 3,538,944 rows. TF32 (`high`, T2 and the learner both): 12 fills,
1,179,648 rows. GPU holds ≤ 18 min under `scripts/ops/gpu_lock.sh`.

## fp32 — the rule is deterministic

| quantity | value |
|---|---|
| rounding scale R (largest \|m_learner − m_variant\|, relative margin, 73,728 rows × every declared site) | **1.44e-5** (eval/no-grad bucket 48: 3.6e-6; bucket 8: 4.8e-6; a few-ulp weight jitter: 1.44e-5) |
| epsilon = round_up_125(10 × R) | **2e-4** |
| excluded share | **3.70 %** (largest 1,024-row block 6.25 %; 2.0 % dominant-move argmax exact ties, 0.6 % fixed damage == current HP, the rest candidate top-k near-ties) |
| ceiling (≥ 3 × pooled, ≥ 1.5 × largest block) | **0.15** |
| judged rows over the 1e-4 bar | **0 of 3,407,893**; judged max \|Δ\| 2.1e-5 (4.7× headroom) |
| all rows over the bar | 4, at margins ≤ 4.2e-7 — every real flip excluded, ~480× under epsilon |
| planted faults (the real probe at epsilon) | one-step-stale weights, ONE wrong-action row (margin 0.128), one env's obs/mask column shifted: **all FATAL on the first update**; the unmodified buffer passes |
| cost | +27.7 ms on a 62.6 ms probe forward of 2,048 rows (~0.07 % of an update) |
| excluded share at epsilon by weights (CPU, 4,096 of A2's rows) | fresh 4.2 %, A2 4.0M 3.9 %, N0 final (75M) 4.2 % |

**UNVERIFIED:** R is measured against eager T2-like variants and a weight jitter, not T2's own compiled
intermediate values (a `TorchFunctionMode` cannot see inside a compiled graph); the end-to-end check
against the real T2 is that every one of the 4 flips sat far under epsilon. Shares are on A2's states.

## TF32 — exclusion cannot make it deterministic, and its current gate fails by chance

- The margins' rounding scale at TF32 is **9.6e-3** relative (bucket 48; bucket 8 3.7e-3; jitter
  2.5e-3), ~700× fp32. Epsilon at the same safety factor is 0.1 and excludes **99.6 %** of rows; even
  1e-2 excludes 51 %. A judged TF32 row's continuous noise is itself ~1e-2.
- 🚨 **FINDING: the existing TF32 gate (p99 < 3.6e-3 single-shot, max < 0.071 FATAL on 4 consecutive)
  false-FATALs on the Rust core with trained weights.** Over 3,600 random 1,024-row healthy probes the
  p99 exceeds 3.6e-3 on **36 %** (median 3.5e-3) and the max exceeds 0.071 on **17 %**: the p99 would
  FATAL within the first few updates, and the max's k = 4 rule expects ~8 false FATALs per 10k updates
  (k = 6: 0.2; k = 8: 0.006). 211 of 1,179,648 rows exceed 0.071 (largest 1.22); p99.9 7.3e-3. The TF32
  bars were measured on the python core with a fresh perturbed learner (`../k9_behaviour_bar_2026-09-30/`).
- **Decision (orchestrator, 2026-10-01):** the TF32 bars are NOT re-tuned; a TF32 run on the Rust core
  with a FATAL `--behaviour-check` is REFUSED at launch (`env_core_rust_tf32_behaviour_check_fatal`,
  FATAL_CONFIG), naming the two ways out (`--behaviour-check warn`, or re-measure the TF32 gate on the
  Rust core). Production (and X26) run fp32.
- **The owner RETIRED TF32 (2026-10-01):** `--matmul-precision high` and every TF32-only gate path leave
  with the deletion pass; these numbers are the evidence for that retirement. The refusal guards the
  window until the deletion lands.

## Reproduce

    scripts/ops/gpu_lock.sh timeout 1080 env PYTHONPATH=src python sweep.py --ckpt <a2 ckpt> --pool <a2 snapshots> \
        --out <dir> --fills 36 [--faults --eps 2e-4 --cost] [--precision high]
    python derive.py --fp32 <dir> --tf32 <tf32 dir> --out result.json
