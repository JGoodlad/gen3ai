# X26's bf16 LOSS drift at the startup gate (2026-10-17)

**Question.** At the 121.14M restart (03:35), X26's (`rb_x26_s2601`, pin `9c033489`) R1 startup gate read a bf16 loss
drift of 6.68e-3, 48 % of the 0.014 bar. Every earlier restart had read about 1e-4. The first canary 10 updates later read
4.86e-4. Why, will it cross the bar, and what is the response?
Answer and design: `designs/training/compile_flags.md` "bf16 drift on a sharpening policy".

**How.**
- Everything ran on the CPU, eager fp32 (the gate's reference, `fp32_scope`) against eager bf16, on the K9 golden rows
  (slice 0, all 64), relocated exactly as the gate does (`compile_regions.clip_relocation` at the run's margin).
- X26's checkpoints were loaded strict at this checkout. `9c033489` → HEAD changes nothing under `src/agents/model`,
  `instrumented_ppo` or the golden buffer.
- The gate's own weights (`final_model_interrupted.zip`, 121,140,589 steps) were copied out before the next restart
  overwrote them. sha256 `c30a1f77378b2bcf36cdf81562acc0eb9d61aee5545c5bc62245d11c064a8268`.
- B = 64 untiled. B = 2,048 agrees to 0.3 %: 6.29e-3 against 6.31e-3, with 608 / 2,048 rows relocated, as on CUDA. CUDA
  read 6.68e-3.
- Rule 5b: nice 19, cores 5–7 / 13–15, one thread, one job at a time, no GPU.

**Scripts** (`scripts/`; run from the checkout root as `nice -n 19 taskset -c 5-7,13-15 scripts/run.sh scripts/<x>.py …`):

| script | reads |
|---|---|
| `drift.py --ckpts … --batch 64 [--attrib] [--dither N] [--shifts …]` | per checkpoint: the drift, per-term and per-row decomposition, the sensitivity S and s_pg, the golden-row ("raw") reading, the grad drift, the per-round attribution (one round bf16 at a time), round activations, the shift_max sweep, the dithered null |
| `rows.py <ckpts> N out.json` | the top rows of the loss drift. For each: A, ρ, whether it was relocated, the bf16 shift, and the fp32 FUNCTION's own move under N half-ulp dithers (its conditioning) |
| `scale.py <ckpts> <rows>` | per row, \|Δ log π\| of fp32 under dithers from 2^-20 to 2^-7. Linear means smooth ill-conditioning; flat means a near-tie |
| `newrule.py <ckpts> <B>` | **THE PRE-FLIGHT**, through production's own functions: the relocated-row drift (what X26's PINNED gate judges) and the golden-row drift (`gen3_r1_loss_on_golden_rows_v1`) |
| `null28.py <ckpts> N out.json` | the null at shift_max 0.18 and at 0.28 (the staged T37 raise), relocated and golden-row readings |

**Results** (`results/`): `checkpoint_reads.json` (series A: 2M–121M every 5M; B: 100M–121.14M fine-grained; C: 114M–123M
with the golden-row null), `rows.json`, `null28.json`.

| read | value |
|---|---|
| null through 113M (476 draws) | max 3.6e-3; 0 over 0.007 |
| null from 114M, relocated (204 + 170 draws) | 3 / 204 and 2 / 170 over 0.014; max 2.05e-2 |
| the same draws, golden rows | 0 / 170 over 0.014; max 7.7e-3 (55 %) |
| at shift_max 0.28, relocated | 7 / 170 over 0.014; max 3.4e-2; 118M undithered 0.01397 |
| row 9's share of the drift | 88–100 % at 117M, 118M, 119M, 121.14M |
| row 9's fp32 sensitivity at 2^-9 | 0.016 (110M) → 0.03–0.07 (118M–121.14M); linear in the dither scale |
| grad drift | max 0.226 over 680 draws (33 % of 0.68) |

Every probability here is CPU and UNVERIFIED on CUDA.
