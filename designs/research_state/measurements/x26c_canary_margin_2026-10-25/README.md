# X26 continuation canary u1000: a near-miss on the clip margin from row conditioning, not bf16 drift (2026-10-25)

**Run.** `rb_x26c_s2601`, the X26 continuation (a fork of `rb_x26_s2601` at 380M). Pin `f4b25238`, bf16 trunk with
decisions in fp32, `--compile-trainer`.

**Event.** Update 1000, 478,310,831 steps. The canary PASSED, but only just, on the clip-margin rule
(`compile_regions._clip_margin_rule`).
- The compiled arm's max |Δ log π| from the fp32 reference was **0.7124 against M = 0.72** (4 × shift 0.18,
  default). Its worst row was fixture row 50.
- Eager bf16's max was 0.224, on row 31.
- The same-precision gap |compiled − eager| was 0.657, on row 50.
- No clip branch switched.
- The worst per-parameter ratio was 3.31 (`edge_bias.c2_map.weight`, K = 12).

Every earlier canary of the run read compiled 0.13–0.33 and eager 0.17–0.38, with a different worst row nearly every
time (fixture rows 20, 21, 17, 16, 20, 25, 47, 4, 20, 55, then 50). Source: `canary_verdicts.jsonl`.

## Verdict: (P), the function's own per-row conditioning tail. Not (D), growing bf16 drift.

**A row the fp32 function is itself sensitive on moved a long way under the compiled kernels' rounding. The bf16 trunk's
error is flat from 180M to 479M. Keep the precision. The default margin is now inside the healthy tail, so a restart
FATAL on this rule would be a false alarm.**

Mechanism, refined: it is NOT an output near-tie. Log-softmax is 2-Lipschitz in the logits' sup-norm, so a small top-2
margin cannot amplify |Δ log π|. Two things amplify it: rows where the trunk's gain is high, and jumps at discrete ops
inside the trunk.

## The read (CPU, eager, at the pin; 64 golden rows, B = 64 so batch row = fixture row)

The arms are E32 (the trunk under `fp32_scope`, the canary's reference) and Ebf (the bf16 trunk, eager). Both run the
`micro_step` forward (`evaluate_actions_functional`). The dither null is 16 seeds of a half-bf16-ulp weight dither
(relative 2^-9, `diag_bf16_canary.dithered`). The full per-row data is in `results/read_<ckpt>.json`; the table comes
from `summarise_reads.py results`.

| ckpt | max \|Δlogp\| Ebf (row) | trunk latent rel. err, median / max | corr(\|Δlogp\|, 1/top-2), Spearman | fp32-alone dither max (row) | row 50: \|Δlogp\|, top-2, entropy, p(taken) | row 31: same | legal-logit spread, median |
|---|---|---|---|---|---|---|---|
| 100M | 0.074 (20) | 0.86 % / 2.5 % | −0.09 | 0.49 (20) | 0.007, 1.63, 1.50, 0.008 | 0.008, 0.47, 1.53, 0.13 | 4.58 |
| 180M | 0.119 (11) | 1.28 % / 3.7 % | −0.24 | 0.43 (11) | 0.014, 1.13, 1.38, 0.006 | 0.021, 3.02, 0.52, 0.04 | 4.85 |
| 230M | 0.194 (21) | 1.23 % / 4.6 % | −0.07 | 0.36 (21) | 0.003, 0.28, 1.76, 0.001 | 0.013, 0.97, 1.32, 0.20 | 4.64 |
| 280M | 0.129 (33) | 1.15 % / 3.7 % | −0.24 | 0.48 (11) | 0.015, 0.43, 1.69, 0.002 | 0.010, 0.27, 1.72, 0.14 | 5.30 |
| 330M | 0.123 (19) | 1.22 % / 3.1 % | −0.16 | **1.18** (62) | 0.057, 0.44, 1.69, 0.007 | 0.018, 0.08, 1.69, 0.19 | 5.26 |
| 380M (fork) | 0.211 (8) | 1.23 % / 3.8 % | −0.08 | **1.23** (51) | 0.021, 0.79, 1.34, 0.0002 | 0.024, 0.79, 1.42, 0.22 | 5.29 |
| 400M | 0.154 (47) | 1.27 % / 6.2 % | −0.28 | 0.50 (58) | 0.010, 0.24, 1.65, 0.0004 | 0.021, 1.22, 1.44, 0.10 | 5.70 |
| 430M | 0.141 (47) | 1.13 % / 3.7 % | −0.27 | **0.80** (20) | 0.034, 1.50, 1.33, 0.001 | 0.043, 1.18, 1.28, 0.05 | 5.75 |
| 470M | 0.214 (1) | 1.11 % / 4.2 % | −0.30 | **0.99** (11) | 0.023, 1.06, 1.09, 0.001 | 0.123, 1.39, 0.95, 0.04 | 6.14 |
| **478M** (≈ 3 updates before u1000) | 0.359 (35) | 1.12 % / 3.9 % | −0.16 | **0.83** (35) | 0.127, **0.073**, 1.52, **0.0003** | 0.108, 1.63, 1.22, 0.12 | 5.75 |
| 479M (after u1000) | 0.345 (25) | 1.12 % / 3.1 % | −0.32 | **0.90** (58) | 0.021, 0.40, 1.56, 0.0008 | 0.034, 0.46, 1.50, 0.08 | 5.86 |

**What kills (D).**
- The bf16 trunk's relative error at the trunk output is flat across 380M of training:
  - `latent_pi` median 1.1–1.3 % from 180M to 479M;
  - `pi_features` median 0.5–0.7 % at every checkpoint;
  - the move tokens, team tokens and cells the pointer head reads stay in the same band (`results/*.json`).
- No discrete op's output differs between E32 and Ebf on any row at any checkpoint (`trace_E32_vs_Ebf`).
- On CUDA the canary's own records agree:
  - eager bf16's max |Δlogp| is 0.17–0.38 at every canary from u10 to u1000;
  - the bf16 drift statistics (`drift_grad` 0.10–0.29, `drift_loss` ~2e-4) do not trend.

**What (P) looks like here: the conditioning of the fp32 function, not a near-tie.**
- **The output near-tie is refuted.** The correlation of |Δlogp| with 1/top-2 margin is −0.07 to −0.32 at every
  checkpoint. Row 50's top-2 margin at 478M is small (0.073), but its TAKEN action is rank 8 of 9 with p = 3e-4. A small
  top-2 margin cannot amplify the taken action's log-prob (Lipschitz bound above).
- **The bf16 error lands where the fp32 function is itself sensitive.** Across rows, Ebf's |Δlogp| correlates with
  each row's fp32-alone dither sensitivity at Spearman 0.67–0.80, and with the legal-logit error range at 0.64–0.80. It
  also correlates with 1/p(taken) at 0.45–0.65: a rarely chosen action's logit sits deep in the tail.
- **What grows is the fp32 function's own sensitivity.** Under a dither the size of bf16 rounding, the fp32 function
  ALONE, with no bf16 arithmetic and no compiler, moves some row's log-prob by:
  - 0.36–0.49 at 100–280M;
  - 0.80–1.23 at 330–479M.

  That is above the 0.72 margin. The policy sharpens slowly at the same time: the median legal-logit spread goes from
  4.6 to 5.9 and the median `latent_pi` norm from 17.8 to 18.7.
- **The tail has two kinds of row** (`eps_scaling.py`, `results/eps_scaling.json`: the worst seed re-run at relative
  amplitudes 2^-9 down to 2^-21, fp32 only):

  | kind | rows | how \|Δlogp\| scales | sites changed at 2^-9 | gain |
  |---|---|---|---|---|
  | SMOOTH HIGH-GAIN | 380M rows 51 / 41, 478M row 35, 479M rows 58 / 29 / 35 | linear in the amplitude (×4 per 2 bits) | — | 300–600 log-prob per unit relative weight perturbation, so 0.6–1.2 at 2^-9 |
  | JUMP | 479M row 50: 0.47 at 2^-9 against 0.007 at 2^-11; 478M rows 11 / 62; 380M row 13 | flat at 0.5–0.8 well above the linear prediction | 4–10 (`damage_op_pairwise.py:459/460/470/697/791/975`, `index_max.py:28` argmax, `damage_kinds.py:119`) | — |

  The jump rows are the piecewise-discontinuity class of `../k9_behaviour_tail/`. Row 50 is a jump row at 479M and
  smooth at 478M (gain ~160). The two sites `hypothesis_set.py:197/200` change under every dither on every row at every
  amplitude. They are exact ties in the weights that any dither breaks, and they move nothing.
- **CUDA compiled's 0.71 on row 50 sits inside this family.** The function's own response to a rounding-sized change
  reaches 0.8–1.2 on its worst rows at these weights, and row 50 moves by up to 0.27 (478M) and 0.51 (479M) under the
  CPU nulls.

**CPU vs the CUDA compiled measurement.**
- CPU eager bf16's max (0.36 at 478M) is about 1.6× CUDA eager's 0.224 at u1000, the same order. On row 50 CPU Ebf reads
  0.127. CUDA eager's row-50 value is not recorded; the triangle inequality bounds it to [0.055, 0.224].
- CUDA compiled's 0.71 on row 50 is about 5.6× CPU eager's value on that row, and 2.6× the row's 16-seed CPU null at
  478.0M. That null is 0.27; at 479M it is 0.51.
- The u1000 weights are 478.31M (about 3 updates past the 478.0M checkpoint), and they were not dumped because the
  canary passed. The exact CUDA reproduction is therefore UNVERIFIED (F1).
- The CPU Inductor compiled read, per row, is below.

**CPU Inductor compiled, per fixture row, at the canary's B = 2048** (`margin_read.py --compiled`,
`results/compiled_cpu_478M.json`; the production `compile_regions.install`, relocation as the canary: 608 rows):

| 478M, B = 2048 | max (row) | median | row 50 | row 31 |
|---|---|---|---|---|
| Cbf vs E32 | 0.379 (35) | 0.017 | 0.089 | 0.139 |
| Ebf vs E32 | 0.325 (35) | 0.012 | 0.012 | 0.048 |
| Cbf vs Ebf (the same-precision gap) | 0.263 (58) | 0.013 | 0.077 | 0.092 |

- E32 at B = 2048 equals E32 at B = 64 to 2.9e-5, and Cbf is identical across the tiles of a row (spread 0).
- But **eager bf16's per-row value depends on the batch shape.** Row 50 reads 0.127 at B = 64 and 0.012 at B = 2048: the
  same weights, the same precision, a different GEMM blocking.
- So any change of the bf16 rounding path is itself a rounding-sized perturbation that moves rows by 0.1–0.3. The
  compiled arm is one more such path.
- CPU Inductor does not reproduce CUDA's 0.71 on row 50. The CUDA value stays UNVERIFIED (F1).

## Recommendation

1. **No precision change.** Nothing in the read shows bf16 error growing in the trunk. The bf16 error tracks the fp32
   function's own conditioning row by row.
2. **The default margin M = 0.72 is no longer a safe bar for the COMPILED arm on these weights.** The fp32 function's
   own half-ulp tail already exceeds it (0.80–1.23 from 330M on). A healthy compiled arm past 0.72 is refused as
   "compiled DISAGREES" (`_clip_margin_rule`): only the yardstick arms get the M' re-run.
   - The worst row rotates and sits in one half of the fixture. Row 50 is in slice 1, so slice 2 cannot reproduce it.
   - So a trip reads UNCONFIRMED, and a second consecutive trip is FATAL.
   - A FATAL on this rule at the restart would be a **false alarm**.
   - The in-pin remedy is a same-run resume with `--canary-shift-max 0.3` (M = 1.2; the value already ruled for X34 and
     the toy, within the 0.5 ceiling). Note that the CPU fp32 null's single worst read, 1.23 (380M, row 51), sits just
     above even 1.2.
3. **The real fix is a per-row margin from a dithered null** (folded into the TECH_DEBT canary row). Judge the compiled
   arm's per-row shift against `YARDSTICK_HEADROOM` × that row's own fp32 / eager dither spread, not against one global
   M from a CPU census taken at 62M. This is the same "one draw against a fixed bar" class as the per-parameter rule
   (u200 read). It needs the CUDA census first (`gpu_replay.sh`).

## FINDINGS

- **F1 (UNVERIFIED on CUDA).** The CPU arms are not the run's Triton / cuBLAS kernels, and the u1000 weights (478.31M)
  were never saved because a PASS dumps nothing.
  - `gpu_replay.sh` is WRITTEN, NOT RUN. It is lease-gated and re-reads 478M / 479M on CUDA: eager E32 / Ebf per row,
    the dither null, and the production compiled arm per fixture row at B = 2048.
  - It closes whether CUDA compiled moves row 50 by about 0.7 at 478.0M and whether CUDA's own fp32 null on row 50
    reaches it.
- **F2.** The compiled arm has no M' relief. Its 0.71 against 0.72 is a 1 % headroom on weights whose own fp32 tail is
  0.8–1.2. Expect clip-margin trips on this run's remaining canaries and on any X26-lineage fork past about 330M.
  Neither the confirmation (the worst row is in one half) nor the consecutive rule tells a conditioning tail from a
  miscompile.
- **F3.** `R1_LOGPROB_SHIFT_MAX` 0.18 was derived on a CPU eager census at 1–62M of a toy run. Its own comment says to
  raise it when the compiled shift exceeds it. On X26 weights the CPU eager max alone reaches 0.36 (478M), already
  2× the shift.
- **F4 (UNVERIFIED, a hypothesis).** The run's K9(b) max (6.3e-4 against C0's 2.3e-4, ceiling 1e-3) may be the same
  conditioning growth, here between the T2 rollout forward and the learner's. The fp32 null's tail grew about 2× from
  280M to 478M. Not read here.
- **F5.** The discrete-op trace reshapes each op's output to [B, -1]. Ops whose leading dim is not a multiple of B (1
  of 1,820 per forward) are not compared per row. The "smallest margins" list ignores exact zeros, which are structural
  (padding compared with 0).
- **F6.** The CPU reads run at B = 64, not the canary's 2048.
  - E32 agrees across the two shapes to 2.9e-5.
  - Eager bf16's per-row values do NOT agree: row 50 reads 0.127 against 0.012 (above).
  - So a CPU bf16 per-row number is one draw of its rounding path, the same caveat as the dither null.
- **F7.** The 479M compiled read was REFUSED by the trainer's fork-recipe resolution: "--vf-coef: the checkpoint's
  model_config.json".
  - The same checkpoint's eager read had resolved earlier in this session, so the run dir's config moved between the two
    reads (the live run's restart, UNVERIFIED).
  - A re-read of any `rb_x26c_s2601` checkpoint from now on may need the recipe typed.

## Files and how to run

Run from a checkout of the pin, with `cwd=<pin>`, `PYTHONPATH=<pin>/src` and `CUDA_VISIBLE_DEVICES=""`, on the agent
cores at nice 19 under `mem_cap.sh` (peak < 1 GB eager).
- `margin_read.py --ckpt L=<zip> … --out D --dither 16 [--compiled]` takes about 13 s per checkpoint eager.
- `eps_scaling.py --ckpt L=<zip> … --top 3 --rows 50 --out eps.json` runs the amplitude scan.
- `summarise_reads.py D` prints the table.
- `gpu_replay.sh <pin> <out>` runs on the GPU, under a lease only.

The checkpoints are the run dirs' `checkpoint_<N>_steps.zip`. 478M is `rb_x26c_s2601/checkpoints/
checkpoint_478000208_steps.zip`, sha256 `b2653cf3…`, identical to `~/gen3ai_archive/x26c_canary_u1000_2026-10-25/`.
Raw outputs are mirrored in `~/gen3ai_archive/x26c_canary_margin/`.
