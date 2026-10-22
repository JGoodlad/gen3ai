# X35 A1 s3503 canary u200 UNCONFIRMED: a one-row, near-floor parameter, not a compile defect (2026-10-22)

**Run.** `rb_x35_a1arch_s3503`. Pin `f4b25238`, bf16 trunk, `--compile-trainer`, production X35 arm A1.

**Event.** Update 200, 299,668,782 steps. This is the **gradient** rule, not the clip margin. The bf16 per-parameter
rule refused 2 parameters. The worst was `features_extractor.edge_bias.c2_map.weight` at ratio 52.5: compiled 1.456
against eager 0.0278. `edge_bias.s1_map.weight` read compiled 1.53 against eager 0.0324.
- The disagreement reproduced on slice 2 (rows 0–31, ratio 49.5) and not on slice 1 (rows 32–63).
- The clip-margin row was healthy: compiled 0.2315, eager 0.1164 (≤ M = 0.72), 0 switches, no yardstick event. It was
  the brief's "ratio 1.99, gap 0.115".
- The run ended about 400K steps later with no further canary.
- The canary wrote its weight dump (`canary_u200_unconfirmed_policy.pt`, sha256 `4ecce673…`), and this read uses it.

## Verdict: FIXTURE class (UNRESOLVED only for the CUDA kernels)

**This is a parameter whose gradient is tiny, carried by one fixture row, and ill-conditioned even in fp32, judged
against a luckily small eager yardstick. It is not a compile defect.**

1. **What `edge_bias.s1_map` / `c2_map` are.** They are two of EdgeBias's 17 zero-init `Linear(cell → 2·n_heads = 8)`
   maps.
   - Each writes per-pair, per-head attention-logit biases at the (E3 own-move seat, opp-mon seat) block and its
     transpose.
   - S1 (cell 2): `[land, land·immob]` of OUR status move against each opp mon.
   - C2 (cell 7): that status move's consequence world (`is_status, land, Δ outspeed, Δ phys hit, Δ residual, sleep`).
   - Both cells are non-zero only on a row whose active has a status move in a seat.
2. **The gradient is small and ONE row carries it** (`result_eager.json` → `attribution`).
   - Of the 64 fixture rows, only **18 and 22** carry a C2 cell. S1 adds rows 9, 56 and 60.
   - **Row 22 carries 92 % (c2) and 96–99 % (s1) of the parameter's gradient.** It is a RELOCATED row: A < 0, its
     old log-prob moved 2M inside the active side, so ρ ≈ 3.6.
   - The reference gradient's norm is **0.67 % (c2) and 0.38 % (s1) of the largest parameter's** on slice 0. The judged
     floor is 0.1 %, so both sit 4–7× above it.
   - **Slice 1 has NO row with a C2 cell.** c2_map has zero gradient there. s1_map reads 1.2e-4 of the top, below the
     floor. Neither is judged on slice 1, so "reproduced on 1 of 2 halves" was **structurally certain** for these
     parameters. It is not evidence for or against a defect. This is the gap `compile_flags.md` T36 §3 names: a
     parameter only one half judges can never confirm.
3. **The fp32 gradient of these parameters is this ill-conditioned** (`result_dither.json`, slice 2). Six seeded weight
   dithers at half a bf16 ulp (relative 2^-9), read in fp32 alone, with no bf16 arithmetic and no compiler.

   | parameter | fp32 dither vs fp32: median | fp32 dither vs fp32: max | eager bf16 dithered: median | eager bf16 dithered: max |
   |---|---|---|---|---|
   | c2_map.weight | 0.23 | **1.08** | 0.55 | 0.94 |
   | s1_map.weight | 0.25 | **1.14** | 0.58 | 1.02 |

   CUDA compiled's 1.46 / 1.53 is the same order as the function's own response to a half-ulp change in the weights.
   **CUDA eager's 0.028 / 0.032 was the outlier.** It is the denominator of the ratio, and a draw from the low tail.
   (`compile_flags.md` T36 row (iii) found the same bimodal yardstick on the pointer head.)
4. **The compiled graph is right, and compiled matches eager at the same precision** (`result_compiled.json`, CPU
   Inductor: the production `compile_regions.install`, C++ codegen).
   - **Compiled fp32 vs eager fp32** on an fp32 learner reads **5.5e-5 / 6.0e-5** relative on c2 / s1. The worst
     edge-bias parameter reads 1.4e-4, and the global gradient 2.6e-5. No structural miscompile touches this module.
   - **Compiled bf16 vs eager bf16**, both against fp32, on c2_map:

     | slice | compiled bf16 | eager bf16 |
     |---|---|---|
     | 2 | 0.345 | 0.350 |
     | 0 | 0.336 | 0.345 |

     s1_map tracks the same way.
   - The canary's own rule on these weights **PASSES**: worst ratio 2.71 (slice 2) and 4.34 (slice 0), with neither
     edge-bias parameter near the top. The relocation matches the run's exactly: 736 / 768 relocated rows on slices 0 / 2.
   - The bf16 error in both arms runs ANTI-parallel to row 22's own contribution (cos −0.88). bf16 shrinks one row's
     contribution; it is not a wrong path.
5. **The CUDA-side pattern.** In A1 arms, CUDA compiled read > 100 % on c2_map before:
   - s3502 at u100: 1.21 against eager 0.12, ratio 9.9, a pass;
   - s3503 at u200: 1.46.

   On CPU both bf16 arms read ~0.35. CPU cannot read the run's Triton kernels (finding F1). By the dither null, the
   CUDA magnitude is in family.

## A1 s3503's final: fit to enter G1 and A1's read (the orchestrator rules)

- The compiled graph is correct on this module (fp32: 6e-5).
- Every global check passed at u200: drift, envelope, the loss on the golden's rows, the clip margin and its 0
  switches. u10 and u100 passed.
- The flagged parameters carry < 1 % of the largest gradient, on one fixture row.
- Even in the worst reading, a 150 % error in c2_map / s1_map's gradient on THIS state is a small perturbation to two
  8-row maps for at most the ~400K steps to the end. That does not change an arm-end read.
- K9(b) arm-end passed (`8a3c4bba5`).

## The canary gap (filed as a TECH_DEBT row; not built here)

The per-parameter bf16 rule divides by ONE eager draw. On a parameter carried by 1–2 fixture rows near the floor, the
ratio is noise over noise:
- at u200, fp32 alone spans 0.0–1.1 under dither;
- eager read 0.028.

**No small fix with a test that fails on revert is honest here.** Any fix moves a bar: a per-parameter dither null for
the yardstick, a "carried-by-≥ N rows" judged set, or a higher floor. That needs a CUDA census first, as with the 1.25
headroom. Filed as P2 in `TECH_DEBT_BACKLOG.md` §2(b).

**For a pin that cannot take code:** a third such event on an A1 arm is two consecutive unconfirmed canaries, which is
FATAL. The live remedy is the orchestrator's call. This read says: diagnose from the dump, as here, before a
rollback.

## Files

- `replay_u200.py` runs the read. Run it from the pin's checkout (`f4b25238`) with `PYTHONPATH=<pin>/src` and
  `CUDA_VISIBLE_DEVICES=""`.
- `result_eager.json`: slices 0 / 2 / 1, E32 and Ebf, with per-row attribution.
- `result_compiled.json`: slices 2 / 0, plus Cbf and C32 (CPU Inductor) and the canary verdict replayed.
- `result_dither.json`: slice 2, the 6-seed half-ulp null.

The inputs are the run's dump (above) and `checkpoints/checkpoint_300000184_steps.zip`, sha256 `3c2e5d71…`, which
supplies the config only; the weights are strict-loaded from the dump.

Command, on the agent cores at nice 19:
`mem_cap.sh 24 python replay_u200.py --zip <ckpt> --pt <dump> --out r.json [--compiled] [--dither N] [--slices 2,0]`.
It takes 928 s eager-only for 3 slices, 1,691 s compiled for 2 slices, and 1,726 s for the dither.

## FINDINGS

- **F1 (UNVERIFIED on CUDA).** CPU Inductor and CPU bf16 GEMMs are not the run's Triton / cuBLAS kernels.
  - CPU eager bf16 reads 0.35 where CUDA eager read 0.028, so the CPU replay cannot reproduce the CUDA ratio itself.
  - A leased CUDA replay would close it: `diag_bf16_canary`'s Cbf / C32 / Ebf on this dump, slice 2.
  - The closing test is C32 vs E32 on c2_map at ~1e-5. If C32 is off on CUDA, it is a Triton-path defect.
- **F2.** The halves confirmation cannot confirm a parameter only one half judges. Here that was certain, and the
  next canary would have been FATAL had the run continued. `compile_flags.md` documents it as intended (the
  consecutive rule); it is still a FATAL a fixture tail can reach.
- **F3.** The same edge-bias tail recurs across production arms: s3502 u100 c2_map ratio 9.9, R0 s3531 u200 11.9,
  C0 s3503 u200 11.1, all just under K = 12 (`canary_verdicts.jsonl` census). The bf16 per-parameter rule has a thin
  margin on X35's surface.
- **F4.** The brief's "0.54 margin" does not match the run: the record says `margin` 0.72 (`canary_shift_max` 0.18,
  default).
