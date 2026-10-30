# Why the compile canary refuses on slice 2 only (2026-10-29)

**Question.** Production-lineage canaries keep refusing R1's bf16 rule on the confirmation slice 2 and passing slice 1.
Is that the compiler (a real numerics difference, which training would have too, since it trains compiled), the
canary's statistic, or the slicing?

**Short answer.**

- The refusing parameter's gradient on slices 0 and 2 is carried by one to three **first-half rows**: 9, 18, 22, 24.
- The fp32 function itself moves that gradient, under the canary's half-ulp weight dither, by about as much as the
  compiled arm does.
- Eager bf16, the ratio's denominator, sits 5-50x **below** that null.
- So the ratio is mostly a lucky denominator on a single-row-carried, ill-conditioned parameter.
- **Hypothesis (i), real compiled numerics, is RULED OUT at these 5 states (CONDITIONING, the GPU replay of
  2026-10-30).** Compiled CUDA bf16's median distance over 16 dithered seeds sits at 0.09-0.55 of the fp32 null's
  maximum on every refusing parameter (below, "The GPU read").
- **The fix is built for new pins** (`gen3_canary_param_null_v1` below, "The fix"): the per-parameter bar divides
  by max(eager, the parameter's own 16-seed fp32 dither null, floor) instead of one eager draw.

## What the canary does (code at main `d78700253`, the same at the X35 / X63 pins)

- **A slice** is a choice of the K9 learner golden's 64 committed rows (16 steps x 4 envs), tiled to the run's
  B = 2,048 (`compile_trainer.fixture_index`).
  - Slice 0: all 64 rows, 32 copies each. This is the scheduled check and the startup gate.
  - Slice 2: rows 0-31, 64 copies each.
  - Slice 1: rows 32-63.
  - The rows are the same for every run and every update. "Slice 2" always means the same 32 states.
- **Relocation** (`compile_regions.clip_relocation`) runs one fp32 eager forward per read, on the current weights.
  - A row whose log-ratio is within the margin M of the clip bound that switches its gradient gets its old log-prob
    moved, so that its ratio sits 2M inside the active side: rho = 0.85 e^(2M) for A < 0, 1.15 e^(-2M) for A > 0.
  - At M = 0.72 a relocated A < 0 row reads rho = 3.59. At M = 1.2 (stage 2, `--canary-shift-max 0.3`) it reads
    rho = 9.37.
  - So its policy weight |A| rho grows 4-11x. Golden row 9 has the largest |A| (3.2) and becomes coefficient 30 at
    1.2.
- **Confirmation** (`compile_canary.run`). A scheduled (slice 0) refusal re-runs the whole check on
  `FIXTURE_HALVES = (2, 1)`, in that order.
  - "Reproduced on slice(s) [2] of [2, 1]" means the first half refused again and the second half passed.
  - It is UNCONFIRMED and training continues.
  - A refusal at the NEXT scheduled canary is FATAL: two consecutive disagreements, each unconfirmed.

## The events (every production-lineage record at margin 0.72; 212 checks over 62 runs)

| run, update | rule that refused | slice 0 | slice 2 | slice 1 | refusing parameter: compiled vs eager (ratio) |
|---|---|---|---|---|---|
| X48 kffn s3502 u200 | per-parameter | refused | refused | pass | edge_bias.c2_map.weight 1.27 vs 0.055 (23.2) |
| A1-arch s3503 u200 | per-parameter | refused | refused | pass | edge_bias.c2_map.weight 1.46 vs 0.028 (52.5) |
| N1-P s3505 u100 | per-parameter | refused | refused | pass | flat_intent_head.hidden.weight 0.120 vs 0.0059 (floor 0.0099: 12.2) |
| N1-P s3501 u200 | per-parameter | refused | pass | pass | hp_type_belief_head.norm.bias 0.403 vs 0.032 (12.8) |
| X26 s2601 u600 | clip margin (compiled shift 0.76 > 0.72) | refused | refused | pass | — |
| D1-wide s3505 u100 | bf16 DRIFT (eager vs fp32, the yardstick) | refused | refused | pass | global gradient 1.23 > 0.68 |
| X26c @482M, stage-2 replay at 1.2 | per-parameter | pass at 0.996 of K | refused 3/3 | pass | spread_belief.nature_head.weight 1.73 vs 0.136 (12.7) |

- Six of seven events are slice 2, and none is slice 1.
- The pattern crosses three rules: the per-parameter rule, the clip margin and the yardstick's drift. The drift
  refusal is EAGER bf16 against fp32, where the compiler takes no part.
- The per-canary disagreement rate at 0.72 is 6 / 166 = 3.6 %.

## The CPU reads (`results/summary_cpu.json`)

The reads, all on CPU with the card hidden:

- `slice_conditioning.py`: per slice, the per-row policy coefficients and each parameter's per-row exposure, on
  relocated and raw rows.
- Its `--dither 16` mode: the fp32 gradient's own half-ulp weight-dither null.
- Its `--loo` mode: leave-one-row-out.
- `row_null.py`: the per-row log-prob null.
- `tie_hook.py`: exact ties at sort and argmax cuts.

Exposure S of a row is ||c_i grad log pi_i|| / ||g_p||. To first order, it is the share of the parameter's gradient
that row carries.

| state | refusing parameter | carrier rows (S) on slice 2 | slice 1 | fp32 dither null on the fired slice (median / max) | compiled read | eager read / null median |
|---|---|---|---|---|---|---|
| X48 u200 | edge_bias.s1 / c2_map.weight | 22 (0.80), 18 (0.58) | **no gradient** (not judged) | 0.52 / 1.78 | 1.29-1.33 | 0.11 |
| A1-arch u200 | edge_bias.s1 / c2_map.weight | 22 (0.96-1.03) | **no gradient** | 0.35-0.41 / 0.97-1.03 | 1.46-1.53 | 0.08 |
| N1-P s3505 u100 | flat_intent_head.hidden.* | diffuse; policy part 0 (supervised intent term); LOO top row 17 (0.27) | judged, null 0.04 | 0.108 / 0.178 (slice 2); 0.081 / 0.128 (slice 0) | 0.178 / 0.120 | 0.02-0.07 |
| N1-P s3501 u200 | hp_type_belief_head.norm.bias | 24 (0.74; 0.97 on slice 0) | spread (top 0.13) | 0.50 / 1.44 (slice 0) | 0.403 | 0.06 |
| X26c 482M @1.2 | spread_belief.nature_head.weight | 9 (1.00), rho 9.37, coefficient 30 | row 32 (0.98), |g| 4.5x smaller | 1.00 / 2.38 | 1.73 | 0.14 |

What the CPU reads establish:

1. **Slice 0 and slice 2 judge the same row.**
   - The carrier sits in rows 0-31, so slice 0's gradient of the parameter is slice 2's, at half the weight. The
     per-parameter numbers on 0 and 2 agree to 1-10 % (X26c 482M: 11.95 and 12.75).
   - So "reproduced on [2]" is the expected signature of ANY refusal on these parameters.
   - On X48 and A1-arch, slice 1 carries no gradient of the refusing parameters at all. It can never confirm them, so
     the consecutive rule is the only route to a FATAL.
2. **The denominator is the outlier.** Eager bf16's distance is 2-14 % of the fp32 function's own median move under
   rounding-sized noise.
   - The K = 12 x eager bar is therefore no bar on these parameters.
   - This is the structure of T36's pointer-head bimodality, without the clip.
   - It belongs to the same class as TECH_DEBT's "one eager draw as the denominator" row.
3. **Relocation is not necessary, but at 1.2 it amplifies.**
   - The raw rows show the same carriers: X48 row 18 at S 1.00, N1-P s3501 row 24 at S 1.02, A1-arch row 22 at
     S 1.0-1.4.
   - At M = 1.2, X26c's nature_head gradient grows 6x and becomes 100 % row 9. At 1.2 the median judged parameter is
     carried by a single row (median max-S 1.00 on slices 0 and 2, against 0.37-0.63 raw).
4. **Not a tie-break.** The tie hook finds no learned-value exact tie at an argsort, argmax or top-K cut. The
   argmax ties are on 0/1 match masks, which are structural.
5. **The forward is rounding-sensitive in both halves.** The per-row fp32 log-prob null reaches 0.4-0.8 on rows in
   both halves (`results/row_null_*.json`), so the first-half asymmetry is about which parameters those rows carry.
   It is not a property of the forward alone.

## Hypotheses, ranked

1. **The statistic, not the compiler.** On a parameter whose judged gradient one row carries, the per-parameter ratio
   reads that row's rounding sensitivity, and eager bf16 lands far inside it. Supported in all five states.
2. **Why slice 2: the fixed fixture.** The carriers of these families live in rows 0-31, and slice 0 shares them.
   The two-halves confirmation is independent by rows, but the consecutive rule is not.
3. **(i) Real compiled numerics. RULED OUT at these 5 states (CONDITIONING; the GPU replay, 2026-10-30).** Under the
   pre-declared rule the compiled median over 16 dithered seeds is 0.09-0.55 of the fp32 null's maximum on every
   refusing parameter, so compiled CUDA bf16 computes these gradients inside the fp32 function's own rounding move.
   - A structural miscompile was already excluded on A1-arch's state by the 10-22 CPU Inductor read
     ([`../x35_a1_s3503_canary_u200_2026-10-22/`](../x35_a1_s3503_canary_u200_2026-10-22/README.md)): compiled fp32
     equals eager fp32 to 6e-5. The GPU replay reaches what that read could not, the Triton kernels' bf16 rounding.
   - **At the dump's own weights** A1-arch's compiled distance is 1.38-1.42 x the null's maximum (1.46-1.61 against
     1.05-1.14), and `mon_hazard_proj.weight` on its slice 2 1.60 x. At the 16 dithered weights around it the compiled
     median is 0.27, a quarter of the maximum. So the dump (kept BECAUSE the canary refused there) is the selected
     extreme of a rounding-sensitive neighbourhood, not a place where compiled leaves the function. The rule's median
     was declared before the read for exactly this reason.
4. **A relocation artefact.** Not the cause (the raw rows show the same carriers), but an amplifier at 1.2. That is
   stage 2's INCONCLUSIVE and a hazard for a `--canary-shift-max 0.3` resume.
5. **A tie-break in a discrete selection.** Not supported by the hook.

## The GPU read (`python -m main.ops.canary_slice_read`; the `slice2-replay` window of `main.ops.gpu_fill_window`)

**Ran 2026-10-30 14:46-15:19, every step rc 0; window verdict `RULED OUT AT THESE STATES`, every state
CONDITIONING.** Outputs: `~/gen3ai_archive/slice2_replay_2026-10-30/slice2_replay/` (`verdict.json`, and per state
`canary.json`, `dither.json`, `meta.json`, `verdict.json`; sha256s in
`src/agents/model/testdata/canary_param_null_basis.json`). CUDA, torch 2.8.0+cu126, B = 2,048, the canary replay valid
on every state.

| state | refusing parameter (slices 0 / 2) | compiled median | null max | compiled / null max | eager median | canary ratio at the dump (old) | with the null (new) |
|---|---|---|---|---|---|---|---|
| a X48 s3502 u200 | `edge_bias.s1_map / c2_map.weight` | 0.650-0.681 | 2.06-2.16 | 0.32 | 0.14 | 21.5-23.3 | 0.62 |
| b N1-P s3505 u100 | `flat_intent_head.hidden.weight / bias` | 0.063-0.084 | 0.116-0.178 | 0.46-0.55 | 0.05 | 12.2-18.0 | 0.94-1.00 |
| c N1-P s3501 u200 | `hp_type_belief_head.norm.bias` | 0.233-0.298 | 1.18-1.43 | 0.20-0.21 | 0.33-0.40 | 12.3-13.1 | 0.28-0.29 |
| d X26c @482M, margin 1.2 | `spread_belief.nature_head.weight` | 0.282-0.297 | 3.17-3.25 | 0.09 | 0.33 | 11.7-11.8 (no refusal in this process) | — |
| e A1-arch s3503 u200 | `edge_bias.s1_map / c2_map.weight` | 0.259-0.275 | 1.05-1.14 | 0.24-0.25 | 0.24-0.26 | 43.8-52.4 | 1.38-1.42 |

- Medians and maxima are over the 16 dithered seeds; "at the dump" is the canary at the dump's own weights, whose
  refusals also named `hp_type_belief_head.type_head.bias` (c, 12.3 -> 0.28) and `mon_hazard_proj.weight` (e slice 2,
  14.1 -> 1.60).
- **The eager draw is the lottery, read directly on CUDA.** The old statistic per dithered seed spans 0.19-27.9 on
  X48's targets (12-25 % of seeds above K = 12) and up to 23.5 on X26c's; over all 62,880 dithered reads (every judged
  parameter, slice and seed) it exceeds 12 on 42, the new statistic on none (maximum 9.18).
- **FINDING (census, not a target):** on X48's slice 2, at dither seeds 2-3, the compiled arm alone moves the value
  path's gradients (`cls_pool.value_cls_attn.*`, `value_entity_pool.*`, `win_head.net.*`) to 0.02-0.13 relative,
  against medians of about 0.003 and nulls at the 0.0099 floor (slice 0 reaches 0.04 at the same seeds, slice 1 0.016).
  That reads 9.18 x the floor under the old and the new statistic alike: under K, unchanged by the fix, and the
  census's largest new-statistic read. Unexplained; no state refused on it.
- Its d state did not refuse in this process (11.7-11.8 against stage 2's 12.7 at the same anchor); its target was
  judged by the dither read regardless.

One process per state:

- the canary on slices 0, 2 and 1, twice (determinism);
- the canary once per carrier row set: slice 2 without its carrier, and slice 1 with it;
- the dither read: 16 seeds x slices 0 / 2 / 1, each seed giving the fp32 null d(E32_s, E32_0) and the canary's
  two distances at the seed's weights.

The verdict rule is pre-declared in the module docstring:

- a target is REAL iff the median compiled distance is above the null's maximum (1 % band);
- it is CONDITIONING iff the median is below it;
- hypothesis (i) is LIVE iff any state reads REAL.

## The fix (`gen3_canary_param_null_v1`, new pins only; `src/agents/model/canary_param_null.py`)

The canary's bf16 per-parameter bar is K x max(eager bf16's error, the parameter's fp32 dither null, the floor), K
kept at 12. The null is this read's own: the fp32 reference arm on the check's own rows at 16 half-ulp-dithered weights
(`diag_bf16_canary.dithered`, the same draws), maximum distance from the reference at the live weights. Because the
bar is never lower than the old one, the null runs only when the eager bar alone would refuse. Records name it
(`canary_rule_version` `per_row_stage1_param_null`). Its basis is this read
(`extract_param_null_basis.py` -> `src/agents/model/testdata/canary_param_null_basis.json`); the design and its limits are
`designs/training/compile_flags.md` "Slice-2-only refusals".

The states are `gpu_fill_window.SLICE2_STATES`, (a)-(e) above: X48, N1-P s3505, N1-P s3501, X26c 482M at 1.2, and
A1-arch. A1-arch's dump was archived for it, with checksums, to `~/gen3ai_archive/a1arch_s3503_canary_u200_unconfirmed_2026-10-22/`.
