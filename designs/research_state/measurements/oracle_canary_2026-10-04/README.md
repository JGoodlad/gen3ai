# The oracle canary FATAL: fp32 noise on an unmoved zero-init head (2026-10-04)

**Verdict: NOISE, not a miscompile.** Tag: MEASURED on the CPU, with the CUDA confirmation DEFERRED.
The FATAL came from `rb_x5ab_oracle_sp_s1001` (pin `401873b5`, `--oracle-reveal species`, the
production blob arch with the X26 ride-along heads) at update 10. The canary read 1.40e-2 on
`features_extractor.belief_head.species_head.weight` against the trained bar of 9.88e-3.

## What happened

- **The species head never trained.** In the run's `final_model_canary_fatal.zip` the whole
  `belief_head` is bit-exactly at its init: `species_head.weight` and `.bias` are 0.0 (zero-init
  under the species-prior fusion), `moves_head.bias` is 0.0, and `norm` is at identity. Under the
  oracle the belief labels are all PAD, so the head gets NO gradient in training. In the blob seed's
  checkpoint at 1.0M steps, the head's weight norm is 2.50.
- **The canary still supervised it.** The canary's rows are the K9 golden's, encoded oracle-off, so
  their labels are real. The canary therefore judged a FRESH-weights gradient (the zero head's) at
  the TRAINED bar, because `weights_regime` classifies the whole model, which had trained.
- **1.40e-2 is the fresh regime's own reading for this parameter.** At this run's startup gate on
  fresh weights it read 8.39e-3. The blob seeds read 6.9e-3 to 9.5e-3 there. The matched-noise
  control (`../k6_k8/r1_noise/`) read 1.0e-2 to 2.5e-2 on four fresh seeds, inside CPU eager's own
  1.4e-2 to 3.1e-2. At update 10 on the blob seeds, after the head had moved, the worst parameter
  read 1.2e-5 to 1.9e-5.

## Measurements (CPU, torch 2.8.0, fp32 'highest', B = 2048, the canary's rows)

Each row compares the head's gradient across arms. `x64` is CPU eager fp32 vs float64. `c64` is
CPU-compiled vs float64. `gate_cpu` is CPU-compiled vs CPU eager.

| weight state | species-head gradient norm (fp64) | fraction of top param | x64 | c64 | gate_cpu |
|---|---|---|---|---|---|
| oracle fatal, live (head = 0) | 0.191 | 0.075 | **1.51e-2** | 4.4e-7 | 1.51e-2 |
| oracle fatal, only the unmoved params perturbed (the new rung) | 0.198 | 0.077 | **4.1e-7** (max over 198 params 1.2e-5) | — | — |
| oracle fatal, every param perturbed | 0.193 | 0.136 | 4.3e-7 | 2.3e-2 † | 2.3e-2 † |
| blob s1001 @ 1.0M, live | 0.172 | 0.104 | 4.4e-7 | 2.9e-1 † | 2.9e-1 † |

† CPU Inductor carries a KNOWN, unresolved belief-path deviation (`../k6_k8/r1_noise/` README,
"CPU Inductor"). Production never compiles on the CPU, so these cells say nothing about the CUDA
graph.

What the numbers show:

- **The gradient is not near zero.** It is 75x above the per-parameter floor in norm, and as a
  sum over (row, slot) terms it is well conditioned (‖|d|ᵀ|x|‖ / ‖dᵀx‖ = 1.39).
- **The error is fp32 eager's own.** On the zero head, CPU eager fp32 errs 1.51e-2 against float64,
  where the compiled CPU arm errs 4.4e-7. Moving the head off zero, by training or by the rung's
  perturbation, takes eager to about 4e-7.
- **Structure (`structure.py`).** The head's input x agrees with float64 to 9.5e-7. The error is all
  in dL/dlogits: 100% of its squared error sits in 5 species columns (376, 385, 135, 373, 251), each
  wrong by about its own norm. On the 2,048 labelled terms, the median per-entry relative error is
  2.4e-7. A pure-prior posterior (the delta at 0) is sharply peaked on a few species, which is the
  softmax-backward cancellation setting. Any movement of the delta softens the peaks.

## The fix (`gen3_r1_unmoved_param_v1`)

The fix is `compile_regions.unmoved_parameters` plus `r1_rungs`, shared by the startup gate and the
canary.

- **Trigger.** A judged parameter whose every element is exactly 0.0. The test is categorical, with
  no tolerance, so no input sits within a rounding error of its boundary.
- **Live rung.** The parameter is judged at the FRESH bar (0.1004) on the live weights.
- **Unmoved rung.** R1 runs again with ONLY those parameters moved off zero, using name-keyed seeded
  noise (the ladder's first rung, restored bit-exactly). EVERY parameter is judged at the TRAINED
  bar there.

A miscompile on the head's path between the two bars is caught by the unmoved rung
(`compile_regions_unmoved_test`).

## Deferred: GPU checks

1. Run the R1 gate on CUDA on `final_model_canary_fatal.zip`, both rungs, and record
   compiled-vs-float64 per parameter. This is the CUDA twin of the table above, and it is the check
   that the unmoved rung reads under the trained bar on the real graph.
2. Watch the first canary (update 10) of the oracle relaunch at the fix commit.

## Follow-up: the INIT rule (`gen3_r1_unmoved_init_v1`)

The zero rule above cannot see a dead parameter whose init is not zero. This follow-up generalises
"unmoved" to bit-identical to the run's fresh-build init record, or exactly 0.0.

- **The FATAL checkpoint against its rebuilt init** (seed 1001, from `original_command`, under
  `single_thread_build`; `unmoved_init_species_fatal.json`). 17 of 387 parameters are bit-identical
  to the rebuilt init.
  - Ten are the SB3 value tower, which is dead in every arm: the blob seed has the same ten at
    15.0M (`unmoved_init_blob_s1001_final.json`).
  - Seven are the oracle's: `belief_head.{species_head, moves_head, norm}.{weight, bias}` and
    `belief_slots.unknown_slot_emb`. The zero rule saw four of them.
- **The rungs' CPU noise floor** (`unmoved_init_rungs_cpu_B2048.jsonl`).
  - Live weights: `moves_head.weight` reads 9.4e-4 and `unknown_slot_emb` 6.4e-4 (eager vs float64).
  - The init rule's unmoved rung: each of the seven reads at most 6.5e-7, and the worst of 199
    judged parameters reads 1.7e-5.
- **Oracle levels** (`unmoved_init_levels_fresh.json`). The fresh builds at `off`, `species` and
  `full` are bit-identical, with the same 387 parameters. So `full`'s unmoved set is decided by its
  no-op losses, which are `species`'s.

## Files

- `unmoved_init.py`: `checkpoint <run_dir> <zip> <out>` lists the parameters bit-identical to the
  rebuilt init; `predict <out>` compares the fresh builds across oracle levels.
- `unmoved_init_rungs.py`: per-parameter eager-vs-float64 on the live rung and the init rule's rung.

- `repro.py`: the matched-noise driver. Usage:
  `repro.py <out.jsonl> <B> [compile|eager] [live,all,unmoved] [states]`.
- `structure.py`: the per-element decomposition.
- `results_cpu_B2048.jsonl`: the rows above.
