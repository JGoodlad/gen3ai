# R1's per-parameter gradient: the matched-noise control (2026-10-01)

**Verdict: the R1 gate's FATAL on a resume of arm C was fp32 noise, not a miscompile.** Tag:
MEASURED, n = 25 weight states. That FATAL came from the sizing study's fp32 `--compile-trainer`
resume, which read 2.47e-3 on `history_events.itemtr_emb.weight` against a bar of 1e-3.

- On arm C's final weights, the CUDA EAGER arm is the inaccurate one. Against float64 its error is
  2.47e-3; the compiled gradient's is 1.7e-5.
- Across 25 weight states, the compiled gradient never erred against float64 by more than 2× the
  worse of CUDA eager's and CPU eager's own fp32 error. The one exception is the 17
  hidden-opp-belief decoder parameters of N0 at 9.5M steps, which read ≤ 5.5e-4 (below).
- The gate now selects its bar by WEIGHT REGIME. Each bar is 4× the largest healthy reading in that
  regime: **fresh 0.100, trained 9.9e-3** (`compile_regions.R1_PARAM_BAR`).

**FINDING.** Under the old bar (1e-3, the extractor gate's FRESH bar from its 64-row probe loss),
a FRESH fp32 launch at the production micro-batch read 1.0e-2 to 2.5e-2. It would also have FATAL'd,
so the X26 baseline was blocked as well. The in-run canary's R1 check had the same bar and would
have FATAL'd every fp32 run at update 100.

## Method (`noise.py`)

- **Setup:** torch 2.8.0+cu126, RTX 3080 Ti, fp32 matmul precision `highest`, B = 2048.
- **Batch:** the gate's own rows: `compile_regions.r1_batch`, the K9 learner golden's 64 real
  labelled rows tiled.
- **Learner:** the production-surface learner (`learner_golden.build_learner`), with each weight
  state loaded strictly.

For one micro-step, R1's gradient over every policy parameter is computed by five arms:

| arm | what |
|---|---|
| R64 | eager `micro_step` in float64 on the CPU. The model is an fp32 program, so this arm runs under two measurement-only patches: sb3's `preprocess_obs` keeps float64, and `torch.Tensor.float` means "the default dtype". |
| E32 | eager fp32 on CUDA |
| C32 | the compiled R1 (Inductor, `fullgraph=True, dynamic=False`) on CUDA |
| X32 | eager fp32 on the CPU (another correct fp32 implementation) |
| aot | `backend="aot_eager"`: the same traced and decomposed graph, run eagerly |

Each comparison is per parameter, using the gate's own relative error `||a - b|| / ||b||` over
the parameters above the gate's floor:

- `gate` = C32 vs E32 (what the gate reads);
- `c64` = C32 vs R64;
- `e64` = E32 vs R64;
- `x64` = X32 vs R64.

Two controls: deterministic algorithms on vs off changes eager by ≤ 1e-6, and a compiled repeat
by ≤ 6e-6. aot_eager is bit-equal to eager in every state, so any compiled difference comes from
Inductor's codegen, not from the traced graph.

The weight states:

| group | states |
|---|---|
| fresh (unperturbed) | 4 seeds |
| perturbed-fresh | the golden's perturbation at 3 seeds |
| trained (18) | `ai_v14_01_base` at 2.4M / 4.8M / 9.5M / 21.7M / 45.8M / 72.3M and final; `ai_v14_02_lbat_ctrl` (arm C) at 75.5M / 78.0M / 82.6M and final; the finals of `ai_v14_03_lbat_e5`, `ai_v14_05_lbat_l95`, `ai_v14_06_lbat_ctrl_fix`, `ai_v14_07_g0p_k2` and `ai_v14_08_g0p_k3`; E5 and L95 at 75.5M |

## Results (`results_cuda_B2048.json`; worst parameter of each state by the gate's reading)

| weight state | regime | gate (compiled vs CUDA eager) | worst parameter | compiled vs fp64 | CUDA eager vs fp64 | CPU eager vs fp64 | params outside 2x the eager envelope |
|---|---|---|---|---|---|---|---|
| fresh | fresh | 2.37e-02 | `belief_head.species_head.weight` | 2.4e-02 | 5.5e-07 | 2.8e-02 | 0 |
| C_final | trained | 2.47e-03 | `history_events.itemtr_emb.weight` | 1.7e-05 | 2.5e-03 | 1.8e-05 | 0 |
| golden | trained | 2.28e-05 | `history_events.cant_emb.weight` | 2.3e-05 | 1.2e-06 | 3.2e-05 | 0 |
| N0_final | trained | 1.82e-05 | `history_events.stat_emb.weight` | 1.7e-05 | 2.4e-06 | 1.7e-05 | 0 |
| C_mid | trained | 2.81e-05 | `conditional_threat.proj.bias` | 3.6e-06 | 2.5e-05 | 9.5e-06 | 0 |
| E5_final | trained | 1.95e-05 | `history_events.stat_emb.weight` | 2.0e-05 | 3.1e-06 | 1.6e-05 | 0 |
| L95_final | trained | 1.31e-03 | `edge_bias.s3_map.weight` | 1.3e-03 | 1.0e-05 | 1.3e-03 | 0 |
| fresh_s1 | fresh | 2.51e-02 | `belief_head.species_head.weight` | 2.5e-02 | 4.9e-07 | 3.1e-02 | 0 |
| fresh_s2 | fresh | 1.01e-02 | `belief_head.species_head.weight` | 1.0e-02 | 5.0e-07 | 1.4e-02 | 0 |
| fresh_s3 | fresh | 1.28e-02 | `belief_head.species_head.weight` | 1.3e-02 | 4.2e-07 | 1.8e-02 | 0 |
| golden_s1 | trained | 1.02e-05 | `history_events.cant_emb.weight` | 9.6e-06 | 1.8e-06 | 1.4e-05 | 0 |
| golden_s2 | trained | 1.86e-05 | `history_events.stat_emb.weight` | 1.9e-05 | 1.1e-06 | 1.7e-05 | 0 |
| 01_base@2400000 | trained | 1.34e-05 | `history_events.status_emb.weight` | 1.5e-05 | 1.6e-06 | 1.4e-05 | 0 |
| 01_base@4800000 | trained | 1.15e-05 | `history_events.cant_emb.weight` | 1.2e-05 | 1.6e-06 | 1.0e-05 | 0 |
| 01_base@9477888 | trained | 5.51e-04 | `hidden_opp_belief.decoder.linear1.weight` | 5.5e-04 | 1.8e-06 | 1.4e-06 | 17 |
| 01_base@21667584 | trained | 1.96e-05 | `history_events.status_emb.weight` | 2.1e-05 | 1.8e-06 | 2.0e-05 | 0 |
| 01_base@45752064 | trained | 1.68e-05 | `history_events.itemtr_emb.weight` | 1.0e-04 | 9.9e-05 | 1.1e-05 | 0 |
| 01_base@72334848 | trained | 1.83e-05 | `edge_bias.t_map.weight` | 5.2e-05 | 3.4e-05 | 3.6e-05 | 0 |
| 02_lbat_ctrl@78006048 | trained | 1.55e-05 | `history_events.denial_emb.weight` | 1.6e-05 | 4.8e-06 | 1.7e-05 | 0 |
| 02_lbat_ctrl@82609344 | trained | 1.97e-05 | `intent_conditional.proj.weight` | 1.7e-05 | 3.2e-06 | 2.0e-05 | 0 |
| 06_lbat_ctrl_fix final | trained | 1.88e-05 | `history_events.cant_emb.weight` | 2.3e-05 | 7.6e-06 | 1.3e-05 | 0 |
| 07_g0p_k2 final | trained | 2.26e-05 | `history_events.itemtr_emb.weight` | 2.2e-05 | 3.9e-06 | 2.7e-05 | 0 |
| 08_g0p_k3 final | trained | 3.43e-05 | `edge_bias.t_map.weight` | 3.5e-05 | 6.9e-05 | 2.5e-05 | 0 |
| 03_lbat_e5@75505968 | trained | 1.81e-05 | `history_events.stat_emb.weight` | 1.6e-05 | 5.1e-06 | 1.5e-05 | 0 |
| 05_lbat_l95@75505968 | trained | 2.13e-05 | `edge_bias.d2_map.bias` | 9.6e-06 | 1.6e-05 | 4.9e-06 | 0 |

`fresh` is seed 0 (unperturbed); `golden` is the K9 golden's perturbed-fresh learner, and
`golden_s1` / `golden_s2` are other perturbation seeds.

- The fresh worst parameter is `belief_head.species_head.weight` in all four seeds. The compiled
  error there (1.0–2.5e-2) sits inside CPU eager's own (1.4–3.1e-2). CUDA eager happens to be
  accurate on that parameter (~5e-7).
- The three trained outliers (C final, L95 final, N0 @ 9.5M) are single ill-conditioned
  parameters, or one transformer block. The other 15 trained states read ≤ 3.5e-5.

## The one state outside the eager envelope: N0 at 9.5M

Here 17 parameters read 1.6e-4 to 5.5e-4 against float64: every parameter of
`hidden_opp_belief.decoder` (a `TransformerDecoderLayer`) plus `edge_bias.{x,g}_map.bias`.
CUDA eager reads ~1.5e-6 on them and CPU eager ~1.3e-6.

- aot_eager is exact, so the deviation is Inductor's.
- The magnitude is 18× under the trained bar. It is deterministic (compiled repeat 0).
- The attention-kernel probe (`sdpa_probe.py`) result is below.

## The fp64-REFERENCED gate — weighed, deferred (`designs/ops/TASK_BACKLOG.md` T16)

- **Cost:** one fp64 R1 forward + backward at B = 2048 takes ~11 s on the CPU (`time64.py`), so
  time is not the obstacle.
- **Obstacle 1:** the reference arm needs the two monkeypatches above until the forward is
  dtype-generic.
- **Obstacle 2:** CUDA eager alone is no envelope. On L95's final, `edge_bias.s3_map` reads
  compiled 1.3e-3, CUDA eager 1.0e-5 and CPU eager 1.3e-3 against float64. With CUDA eager as the
  envelope, a rule like "compiled ≤ k × eager's error" would false-FATAL there and on every fresh
  seed.

## CPU Inductor (a FINDING outside production)

On the CPU (`--compile-trainer` refuses a CPU device, so this is never a production path), the
golden learner at B = 64 gave:

- a loss 1.35e-3 relative off eager (0.71571 vs 0.71474);
- belief-path parameter gradients up to 21% off (`belief_slots.unknown_slot_emb`), where eager fp32
  is within 1e-6 of float64.

aot_eager is exact there too, so the deviation is CPU Inductor's codegen. Unresolved.
