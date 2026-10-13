# CUDA build validation: torch 2.8.0+cu126 vs +cu128

Results: `/home/goodlad/.claude/jobs/4b7f33cf/tmp/cudaval/designs/research_state/measurements/cuda_build_2026-10-12/results/`. A benchmark warns, never stretches: the load average and the other GPU processes are in `summary.json`.

## (a)+(c) Update cells (the local baseline for the cloud ladder)

| config | build | gate | update s (= train_ms/1000) | marginal epoch s | per-micro ms | peak MiB | compile+gate+prewarm s | spread | torch |
|---|---|---|---|---|---|---|---|---|---|
| E_fp32 | cu126 | PASS | 53.47 | 5.228 | 108.9 | 6718 | 273 | 0.1% | 2.8.0+cu126 |
| E_fp32 | cu128 | PASS | 52.99 | 5.16 | 107.5 | 6718 | 274 | 0.1% | 2.8.0+cu128 |
| **E_fp32** | cu128 / cu126 | | **0.991** | 0.987 | | 1.000 | | | |
| E_bf16 | cu126 | PASS | 41.24 | 4.015 | 83.64 | 6436 | 254 | 0.0% | 2.8.0+cu126 |
| E_bf16 | cu128 | PASS | 41.3 | 4.007 | 83.48 | 6436 | 273 | 0.3% | 2.8.0+cu128 |
| **E_bf16** | cu128 / cu126 | | **1.001** | 0.998 | | 1.000 | | | |
| P_fp32 | cu126 | PASS | 45.51 | 4.431 | 92.31 | 6582 | 189 | 0.1% | 2.8.0+cu126 |
| P_fp32 | cu128 | PASS | 45.58 | 4.437 | 92.44 | 6582 | 205 | 0.1% | 2.8.0+cu128 |
| **P_fp32** | cu128 / cu126 | | **1.001** | 1.001 | | 1.000 | | | |
| P_bf16 | cu126 | PASS | 37.75 | 3.653 | 76.11 | 6076 | 203 | 0.1% | 2.8.0+cu126 |
| P_bf16 | cu128 | PASS | 37.72 | 3.65 | 76.04 | 6076 | 203 | 0.1% | 2.8.0+cu128 |
| **P_bf16** | cu128 / cu126 | | **0.999** | 0.999 | | 1.000 | | | |

## (b) Numerics: same weights, same batch, eager (cross-build difference read against the same-build noise)

| prec | valid | loss cu126 | loss cu128 | loss rel diff | grad rel-L2 | cosine | bitwise equal | worst parameter | same-build noise (grad rel-L2) |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | yes | 0.7407341 | 0.7407341 | 0 | 1.4e-09 | 1.000000000 | False | features_extractor.spread_belief.nature_head.weight (1.43e-08) | 2.36e-09 |
| bf16 | yes | 0.741915762 | 0.741915762 | 0 | 2.11e-09 | 1.000000000 | False | features_extractor.hp_type_belief_head.type_head.weight (2.7e-08) | 2.16e-09 |
