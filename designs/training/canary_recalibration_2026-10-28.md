# Canary recalibration from the W0 census (2026-10-28)

**Status: RULED (2026-10-28): stage 1 implemented** (`gen3_canary_row_bar_v1`, `src/agents/model/canary_row_bar.py`;
the rules as built are `compile_flags.md` "The per-row clip margin, STAGE 1"). The orchestrator accepted stage 1: the
per-row bar (k 6.0 over a 16-seed per-row dithered null, floor 0.02, each row's bar capped at 0.72, the relocation
scalar), the bf16 loss envelope K 5.4 -> 9.2, the per-parameter rule kept at 12 with no edge_bias exemption, and the
family bars (C0 2.64 / X47 2.40) kept as descriptive tags for runs on the global M and dropped on per-row runs (a per-row
FATAL HOLDs in `canary_decide`). **Stage 2** (cap 1.2) needs a leased read later, not now. Every run at a pin before the
implementing commit (every X35-frame arm and X63-C at `f4b25238`) keeps the global M and K 5.4; every canary record
names its regime (`canary_rule_version`). The basis the code re-derives k and K_loss from is
`src/agents/model/testdata/canary_stage1_basis.json` (`extract_stage1_basis.py` beside the backtest, from the census and
the banked records); `canary_row_bar_test` backtests the rule on it. The DRAFT below is kept as the ruling's evidence.

The backtest is
`designs/research_state/measurements/canary_recalibration_2026-10-28/backtest.py` (CPU, read-only, seconds; its
docstring holds the command). Its outputs are `results/backtest.json` (the proposal) and `results/backtest_cap072.json`
(stage 1, the bar capped at M) beside it. Every number below comes from those files unless it names another source.

The canary itself is `compile_flags.md` ("The bf16 canary FATAL on trained weights" through "The canary-FATAL resume
rule and its family bars"). The per-row design is `../research_state/measurements/canary_row_null_design_2026-10-25/`.
The two debt rows this answers are in `../ops/TECH_DEBT_BACKLOG.md`: "The canary-FATAL resume rule's C0 bar 2.64…" and
"The canary's bf16 per-parameter rule divides by ONE eager draw…". The C0 loss-envelope row is answered too.

## The rulings asked for (all four ruled as proposed for stage 1, 2026-10-28)

| question | proposal | the evidence in one line |
|---|---|---|
| which arm and floor set the per-row k | **k = 6.0**: 4 x the healthy maximum (1.479) of the canary's own two bf16 arms at B = 2,048 on CUDA (compiled and eager, pooled), at **floor 0.02** | the compiled arm alone gives 4.7; eager's tail on these rows is evidence about any bf16 arm; floors 0.01 and 0.02 differ only on rows whose null is pure rounding |
| the seeds | **K = 16** (the shadow's `ROW_NULL_SEEDS`) | K = 8 under-samples the null (k would be 13); K = 32 gives k 5.7 but costs about 4.7 s against the shadow's 5 s cap |
| the false-FATAL rate on the census | **0 of 20** checkpoints refuse: in sample, leaving each checkpoint out, and across lineages (X26 to X26c and back) | a row-local refusal cannot confirm on two disjoint halves, so a FATAL needs two consecutive refusals; at the 0-of-20 rule-of-three bound that is at most about 2.3 % per consecutive pair |
| switch the live canary from M to the per-row k? | **Yes, at a pin, never mid-run, in two stages**: stage 1 caps every row's bar at M = 0.72; stage 2 raises the cap to 1.2 after one leased read | every live arm runs at pin `f4b25238`, which the launcher will not move; stage 1 only tightens the bar |
| the family bars (C0 2.64, X47 2.40) | **C0: CONFIRM** 2.64 for runs on the global M, with production provenance; **X47: unchanged** (toy only); **DROP both** for runs on the per-row rule | 3 of 91 healthy X35 passes read above 2.64 (maximum 5.0); raising the bar to cover them would resume most real disagreements |
| the bf16 loss envelope (C0 final) | **re-derive K_loss = 9.2** (4 x 2.283, the maximum over 159 production checks) in place of 5.4 (4 x 1.35 over window D's 20 reads) | the C0 final's slice-1 refusal (1.036 of today's bar) reads 0.61 of the new one; no production check exceeds either bar |
| the per-parameter 12x rule (edge_bias) | **KEEP 12**; fix the statistic (a per-parameter dithered null), never the number and never an exemption by name | R1's own convention on production data would put the bar at 48, which disarms the rule; both large events are `edge_bias.c2_map` states that did not confirm |

## What the canary judges today

Every 100 updates the canary runs region R1 (the compiled learner micro-step) against eager on the K9 golden's 64 rows,
tiled to B = 2,048, under the bf16 rule. These are the quantities it judges, with each bar and where it came from:

- **The clip margin.** Each arm's largest |Δ log π| from the fp32 reference must be at most M = 0.72 (4 x 0.18, from a
  CPU census of a toy run). Rows near their PPO clip boundary are first moved 2 x M inside the active side
  (`relocate_clip_boundary`, T36).
- **The drift.** Eager bf16 against fp32: the loss within 0.014 and the gradient within 0.68 (relative L2).
- **The envelope.** Compiled against fp32 must stay within K x max(eager's own distance, a floor): the loss at K 5.4
  with floor 2^-9, the gradient at K 6.4. Each judged parameter's ratio must be at most K 12.

All of these bars come from window D (2026-10-10: 20 reads, 4 x the healthy maximum). The family bars of
`main.ops.canary_decide` come from the toy runs.

A scheduled refusal re-runs the check on the two disjoint halves (rows 0-31 and rows 32-63). If both halves refuse,
the canary is FATAL. If only one does, the verdict is UNCONFIRMED. Two consecutive scheduled refusals are FATAL.

## The data

- **The W0 CUDA census** (`~/.claude/jobs/x35/w0_2026-10-28/census/`, `gpu_census.sh`; RTX 3080 Ti, torch
  2.8.0+cu126) covers 20 X26-lineage checkpoints: X26 at 100-380M, X26c at 390-482M, and the X26c u1000 neighbours at
  478.0M and 479M. For each of the 64 fixture rows it holds a 32-seed fp32 dither null (relative 2^-9, on CUDA), plus
  the compiled (Cbf) and eager (Ebf) bf16 arms at B = 2,048 on CUDA. The old global rule refuses none of the 1,280
  row-reads.
- **The banked records** (`models/<run>/canary_verdicts.jsonl`, read-only) cover three groups:
  - 33 X35 runs: 95 scheduled checks plus 8 confirmation halves, 91 PASS and 4 UNCONFIRMED;
  - the P\* lineage (rb_x26, rb_x26c): 54 scheduled checks plus 2 halves, 53 PASS and 1 UNCONFIRMED;
  - 7 toy runs (51 checks), kept for reference only.

  The three archived copies (`~/gen3ai_archive/`, 17 lines) are all duplicates of `models/` lines.
- **The X-CHECK anchors** (`~/.claude/jobs/x35/w0_2026-10-28/xcheck/`, `main.ops.canary_dump_replay --anchor`) are
  four states, read on three slices each, 5 bit-identical repeats. They are the out-of-sample test for the loss
  envelope.

**FINDING 1: no banked record carries `row_null` or `bar_fractions`.** Every X35, X48 and X63-C run is pinned at
`f4b25238`, and N1-P is pinned at `5f8f6cfa`. Both pins predate the shadow null (`4fa4e5360`) and the near-margin dump
(`0dcf0ea4c`). The design's plan was to set k from "an era of production canaries". That cannot happen at these pins,
so the W0 census is the only CUDA basis for k.

## 1. The per-row bar: arm, floor, seeds, cap

**The rule.** Row i of the judged (compiled) arm fails iff its shift is greater than

    b_i = min(k x max(null_i, f), C)

where null_i is the row's K-seed fp32 dither null, computed at check time on the canary's own weights (exactly what
the shadow records). Relocation stays one scalar, at C.

**k by arm, seeds and floor** (4 x the healthy maximum, rounded up to 2 significant digits, R1's convention):

| null seeds K | floor | compiled only | eager only | compiled + eager (proposed) | + eager at B = 64 |
|---|---|---|---|---|---|
| 8 | 0.02 | 6.2 | 13 | 13 | 13 |
| **16** | **0.02** | 4.7 | 6.0 | **6.0** (1.479 at X26 380M, row 63) | 7.0 |
| 16 | 0.05 | 4.3 | 5.7 | 5.7 | 7.0 |
| 32 | 0.02 | 3.6 | 5.7 | 5.7 | 7.0 |

- **The arm.** The compiled and eager bf16 arms differ only by Inductor's fusion rounding. Same-precision compiled
  against eager reads 4-6 x tighter than either arm against fp32 (the X34 u1700 read). Twenty checkpoints are a thin
  sample, and eager's tail on these rows (1.48) tells us what a healthy bf16 arm can do on a row the compiled arm
  happened not to hit. The compiled arm already ran 2.7 x a CPU null on row 50 at u1000. So pool the two B = 2,048
  arms. Leave out the B = 64 eager read: the canary never runs that batch.
- **The floor, 0.02** (the shadow's `ROW_NULL_FLOOR`). Floors 0.01 and 0.02 differ only on the 190 row-reads whose null
  is under 0.01. bf16 moves those rows by at most 0.0103, so the maximum at floor 0.01 (1.034 at K = 32) is a 0.010
  shift over a 0.008 null: rounding, not signal. At floor 0.02, 19 % of row-reads sit on the floor; floor 0.05 would
  loosen the 29 % whose 16-seed null is under 0.05, for nothing.
- **The seeds, K = 16.** The null's maximum over K seeds is heavy-tailed. Eight seeds reach a median 0.72-0.92 of the
  32-seed maximum, which inflates k to 13. Sixteen seeds reach 0.92-1.00 (`derive_k_census.txt`). A K-seed null
  costs K forwards: K = 16 measured **2.35 s on the card** (`census/cost_cuda.json`; noise drawn on the CPU side, so it
  is an upper bound), and K = 32 would be about 4.7 s against the shadow's 5 s cap (`ROW_NULL_CAP_S`), which risks a
  skip. K = 16 is also what the shadow already computes.
- **The cap, C.** Standing rule 8 needs the relocation margin to be at least every arm's bar. Relocating each row by
  its own b_i would move A < 0 rows up to 2 x 7 log-prob inside the active side, a ratio of about e^14. A single row
  would then carry the gradient rule's inputs (T36 already multiplies such a row's weight by 4.2 at M). A scalar cap
  keeps `relocate_clip_boundary`'s one-scalar signature and the gradient rule's fixture, which closes the design's
  open question 4 by construction.
  - **Stage 1: C = 0.72 (= M).** This is strictly tighter than today, and the relocation is unchanged.
  - **Stage 2: C = 1.2,** the ruled `--canary-shift-max 0.3` margin. It also clears the tail false alarm that
    motivated the design. Before it binds it needs one leased read: the gradient rule on rows relocated at 1.2
    (FINDING 5).
- **The yardsticks stay on the scalar rule.** Eager and the fp32 reference keep M (or C) and the M′ re-run. Only the
  judged arm gets per-row bars; the design's rule 4 is not adopted, because per-row M′ bookkeeping would add no teeth.

## 2. The false-FATAL rate on the census, and the teeth

| test | refusing checkpoints | both halves | consecutive pairs |
|---|---|---|---|
| in sample (k 6.0, K 16, f 0.02; stage 1 or stage 2) | 0 / 20 | 0 | 0 |
| leave one checkpoint out (k re-derived from the other 19: 5.7-6.0) | 0 / 20 | - | - |
| derived on X26 (k 6.0), judged on X26c | 0 / 14 | - | - |
| derived on X26c (k 5.7), judged on X26 | 0 / 6 | - | - |
| k 6.0 with K = 8 / 16 / 32 seed nulls | 0 / 0 / 0 | - | - |

- **The rate.** A refusal carried by one row can never confirm on two disjoint halves, so a per-row FATAL needs two
  consecutive refusals. With 0 of 20, the 95 % bound is 15 % per canary and about 2.3 % per consecutive pair. The 107
  production proxy reads below (0 refusing) bound it at about 2.8 % per canary on their own.
- **The teeth (share of the 1,280 row-reads whose bar is below a planted shift):**

  | shift | today | stage 1 | stage 2 |
  |---|---|---|---|
  | 0.2 | 0 % | 24 % | 24 % |
  | 0.3 | 0 % | 29 % | 29 % |
  | 0.5 | 0 % | 41 % | 41 % |
  | 0.72 | 0 % | 55 % | 55 % |
  | 1.0 | 100 % | 100 % | 68 % |

  Stage 2 trades the 24 % of rows that sit at its cap: a shift in (0.72, 1.2] confined to a high-null row passes the
  margin. The gradient rules still see a real miscompile. This is the design's open question 5, for the
  orchestrator.
- **The X26c u1000 near miss** (compiled 0.712 on row 50, 1 % under M). Its exact neighbours' 16-seed nulls on row 50
  are 0.267 (478.0M) and 0.472 (479M), so b = min(1.6, C). Under stage 1 it reads 0.99 of the bar, the same as today.
  Under stage 2 it reads 0.59.
- **The X26 u600 UNCONFIRMED** (compiled 0.76 on a row in 0-31; the record predates `worst_fixture_row`). Stage 1
  refuses it as today. Stage 2 clears it iff that row's null is at least 0.127, which half of rows 0-31 meet at 380M.
  Which row it was is UNVERIFIED.
- **A row's null swings 10x between adjacent checkpoints.** Row 50 reads 0.073 at 470M, 0.47 at 479M and 0.077 at
  482M. So the null must be the canary's own, computed at check time, which it is. Every borrowed null below is
  indicative only.

## 3. The backtest on the banked records

**The proposed rules on the X35 and P\* lineage records:** the per-row margin (stage 2, proxy null), K_loss 9.2,
per-parameter 12, the global M kept for the yardsticks.

| group | scheduled checks | refused today (by quantity) | refused under the proposal | FATAL paths, today / proposed |
|---|---|---|---|---|
| X35 (33 runs) | 95 | 4: per-parameter 3 (12.8, 23.2, 52.5), eager drift gradient 1 | the same 4 (the per-row proxy adds 0; no loss refusal either way) | 0 / 0 (62 consecutive pairs) |
| P\* lineage | 54 | 1: clip margin (X26 u600, 0.76) | 0 or 1 (stage 2 clears it iff its row's null is at least 0.127) | 0 / 0 |
| toy (reference) | 51 | 1: clip margin (X47 s3602 u400, compiled 0.91) | not read (no proxy null for toy weights) | 0 / - |

- **How the per-row margin was read on the records.** A record names only its worst row, so this is a lower bound on
  refusals. Each record's compiled worst row was judged against that fixture row's census null (the K9 golden buffer
  is the same blob, `99c67b80`, at both pins and at HEAD):
  - at the nearest lineage checkpoint: 0 of 95 X35 and 0 of 12 P\* refusals;
  - at the smaller of the two checkpoints that bracket the record's step: 0 and 0;
  - at that row's smallest null over all 20 checkpoints: 31 and 6.

  The last line is not a usable bound, because the nulls swing 10x (§2).
- **Near-margin records** (at least 0.9 of any bar, computed offline as `compile_canary.bar_fractions` does):
  - X35: the per-parameter ratio at C0 s3503 u200 (0.93), R0 s3531 u200 (0.99) and SPRT s3501 u100 (0.91), plus the
    four UNCONFIRMED records;
  - P\*: X26c u1000 on the clip margin (0.99).

  None of the PASS records dumped weights (FINDING 2).

## 4. The bf16 loss envelope (the C0 final)

**The facts.** On slice 1 the C0 s3501 final's compiled loss distance is 1.09e-2. Eager's 1.20e-3 is under the floor
(1.95e-3), so the bar is 5.4 x 1.95e-3 = 1.05e-2, and the read is 1.036 of it, in 5 of 5 repeats. Slices 0 and 2 pass.
Its ratio envelope / max(eager, floor) is 5.59. The largest across the 159 production checks is 2.283 (X35, every
slice), and window D's healthy maximum was 1.35 over 20 reads. None of window D's reads was a production state at
280-480M.

**The proposal: K_loss = 4 x 2.283 = 9.2.** This is R1's convention, run on the production sample that window D did
not have.

- No production check exceeds 5.4 or 9.2.
- The 12 anchor slices are out of sample. The C0 final's slice 1 reads 0.61 of the new bar, and every other anchor
  slice reads at most 0.27.
- The teeth: a floor-limited read's bar rises from 1.05e-2 to 1.80e-2 (relative loss), and the DRIFT bar 0.014 is
  unchanged.
- Implementation: `compile_regions_bf16_test` pins `BF16_ENVELOPE_K` to `derive_bf16_bars.py` on window D's reads, so
  the change must commit its new basis (this script's records basis) with the constant
  (`../research_state/measurements/perf_phase_2026-10-10/scripts/derive_bf16_bars.py` is window D's).

**The alternative: keep 5.4.** The C0 final's refusal cannot reach FATAL on its own. Slice 0, the scheduled check,
passes, and slice 2 passes, so neither the two-halves path nor the consecutive path can complete. It can, however,
complete a two-halves FATAL for an unrelated refusal that hits slice 2: the X48 dump's shape, where slices 0 and 2
refused on edge_bias and slice 1 passed. That is why the bar is moved rather than left.

**FINDING 3: the production basis is thin on halves.** Only 10 of the 159 checks are confirmation halves (maximum
1.41), and the C0 anchor's 5.59 is a half. A half has 32 unique rows, so one heavy row weighs twice as much.

## 5. The per-parameter 12x rule and the edge_bias event

**KEEP 12.** These are the production facts:

- Healthy X35 passes reach 11.89, 11.13 and 10.92.
- Three scheduled checks refused, and none confirmed:
  - N1-P s3501 u200, 12.8 on `hp_type_belief_head.norm.bias`;
  - X48 s3502 u200, 23.2 on `edge_bias.c2_map.weight`;
  - A1 s3503 u200, 52.5 on `edge_bias.c2_map.weight`.
- `edge_bias` holds the worst parameter on 32 of the 95 X35 scheduled checks.
- The X-CHECK shows the X48 event belongs to the state, not to a graph class. It reproduces on its dump (26.6 x) and on
  neither X48's own final (8.0) nor X26c @480M (2.4). The C0 final is edge_bias-clean (8.67).

**Why not move the number or carve out edge_bias:**

- R1's convention (4 x the healthy maximum) on production data gives 48, or 210 if the UNCONFIRMED states count as
  healthy. Either disarms the rule.
- The defect is the statistic. Its denominator is one eager draw, and on a parameter whose gradient one or two rows
  carry, that draw is a lottery.
- An exemption by name for edge_bias would exempt the family where a real miscompile on those parameters would also
  show.

**What protects a run meanwhile is the disjoint halves.** Across all causes, X35's scheduled refusal rate is 4 of 95,
about 0.2 % per consecutive pair if refusals are independent; 0 of 62 pairs were observed.

**The fix is the TECH_DEBT row's first half:** a per-parameter dithered null, or a guard that judges a parameter only
when at least N rows carry its gradient. That needs a census W0 did not take (eager-bf16 per-parameter gradients under
the dither); its scope is that row's.

## 6. The family bars

`canary_decide` resumes a FATAL only if compiled / eager max |Δ log π| is at or below the family bar. That ratio
divides two maxima taken on different rows, and eager's maximum alone ranges from 0.096 to 0.58 across production.

| group | healthy PASS maximum | PASS records above 2.64 |
|---|---|---|
| X35 | 5.00 (D3 s3504 u200: 0.479 / 0.096) | 3 of 91 |
| P\* lineage | 3.18 (X26c u1000) | 1 of 53 |
| toy C0 + X47 | 2.30 | 0 of 50 |

**C0: CONFIRM 2.64** for every run on the global M, which is every run pinned at `f4b25238`, with this production
provenance. It errs to HOLD on about 3 % of healthy states, and a HOLD costs an operator look, never a wrong resume.
The alternative, 1.25 x 5.00 = 6.3, would resume a both-halves margin FATAL with compiled up to 6.3 x eager (0.6-1.2
at a typical eager 0.1-0.2), which is most real disagreements.

**X47: unchanged.** No production arm takes it.

**DROP both for runs on the per-row rule.** That rule already compares the compiled arm with its row's own null, so a
per-row FATAL means compiled exceeded 6 x the tail on two consecutive checks. `canary_decide` then HOLDs, and the
operator replays the dump (`main.ops.canary_dump_replay --dump`).

## 7. When to switch

**Only at a pin, never mid-run.** A canary rule is code. It reaches a run only through the run's pin, and the
launcher refuses to move a live run's pin (`worktree.resolve_pin`). Every live and queued X35-frame run (C0, X48,
X63-C) is at `f4b25238` and finishes on the global M, with the confirmed C0 bar and the `--canary-shift-max 0.3`
resume.

The first run launched at a pin carrying the change takes it, fresh or as a fork. The change at that pin:

- **Stage 1:** the per-row bar (k 6.0, K 16, floor 0.02, cap 0.72), binding. The global-M verdict is recorded beside it
  and never judged (the design's rule 5). It also carries K_loss 9.2, and `canary_decide` HOLDs on a per-row FATAL.
- **Stage 2:** the cap goes to 1.2 after the leased read in FINDING 5 passes. That read is the gradient rule on rows
  relocated at 1.2 over the census checkpoints.

**PREREG, stage 2's read (registered 2026-10-29, before any number exists).**

- **What it decides:** whether `ROW_BAR_CAP` may go from 0.72 to 1.2 at the next pin. Nothing else moves on it.
- **The read:** the window `canary-stage2` of `main.ops.gpu_fill_window` (its module docstring has the commands).
  - Each of the 20 census checkpoints (§ The data, the exact zips the census read) is replayed with the canary's own
    check, `canary_dump_replay --anchor … --canary-shift-max 0.3 --repeats 3 --batch 2048`, on CUDA.
  - `--canary-shift-max 0.3` relocates the rows at 4 x 0.3 = 1.2.
  - Each read covers slices 0, 2 and 1.
  - Outputs go to `~/gen3ai_archive/gpufill_2026-10-29/canary_stage2/<run>_<at>/`, and the verdict to `verdict.json`.
- **The margin rule in this read is the scalar 1.2.** That is the cap, which a row whose null cannot bind falls back
  to. The per-row tightening below the cap is the census's own result (§2: 0 of 20 at stage 2). This read adds the
  part the census could not see: the gradient, loss and per-parameter rules on rows relocated 2 x 1.2 inside the
  active side.
- **The rule** (`stage2_verdict`, applied by the window's CPU `verdict` step):
  - **REJECT** if any checkpoint's canary refuses its scheduled slice (0) in at least 2 of the 3 repeats: a
    reproducible refusal on a healthy state.
  - **ADOPT** if all 20 checkpoints read complete, self-valid and at margin 1.2, with zero refusals on any slice and
    no per-parameter ratio within 0.1 % of K = 12 (standing rule 8).
  - **INCONCLUSIVE** otherwise: an isolated refusal, a boundary read, or a missing or invalid checkpoint.
  - Only ADOPT moves the cap. REJECT and INCONCLUSIVE keep 0.72.
  - An INCONCLUSIVE's follow-up is the orchestrator's call, for example the refusing checkpoint at the run's own
    margin as a control.
- **Cost:** about 45 min on the card, UNVERIFIED. Each checkpoint pays one Inductor compile (85 s in the X-CHECK
  anchor read), then about 10 s per repeat. The step bound is 900 s and the window's budget 90 min.

## 8. Findings (hazards and what is UNVERIFIED)

1. **No production record carries `row_null` or `bar_fractions`** (§ The data). The shadow's data starts only at the
   next pin.
2. **The TECH_DEBT row's "sharper X-CHECK" has no input.** It wants every near-margin dump on a P\*-lineage run, but
   no P\*-lineage pin carries the near-margin dump, so only the UNCONFIRMED records left weights. Two of those are
   archived: N1-P u200 and X48 u200. The nine near-margin records in §3 are what it would have replayed.
3. **The loss basis is thin on halves** (§4).
4. **The per-row backtest on records is a proxy.** It borrows a null and sees only the worst row. Only the census is
   exact.
5. **The gradient rule at relocation 1.2 is unread.** Production ran at M 1.2-1.4 only on resumes (one P\* record at
   1.4 passed). `canary_dump_replay` could not take a margin override, because anchor mode took the run's own
   `--canary-shift-max`. It now takes `--canary-shift-max` in anchor mode (2026-10-29; dump mode refuses it). A CPU
   run on X26 380M read margin 1.200 with 960 of 2,048 rows relocated. The read is §7's PREREG.
6. **Outside this unit's closed list:** D1 s3505 u100 refused on the eager DRIFT gradient (1.23 against 0.68, a
   window-D bar from a healthy maximum of 0.169). Healthy X35 passes reach 0.52, which is 0.76 of that bar. That bar
   wants the same production re-derivation, by its own ruling.
7. **The seeds are not the shadow's.** The census null was drawn with CPU-side noise (`margin_read`), while the shadow
   draws on the device. The distribution is the same and the draws are not. The CUDA cost (2.35 s at K 16) is an
   upper bound for the same reason.

## Decision record

| date | decision | why |
|---|---|---|
| 2026-10-28 | DRAFT written: per-row k 6.0 (K 16, floor 0.02, scalar cap 0.72, then 1.2), K_loss 9.2, per-parameter 12 kept, C0 2.64 confirmed for global-M runs and dropped on per-row runs, switch only at a new pin | W0's CUDA census (20 checkpoints) and the 159 banked production checks; awaiting the orchestrator's ruling |
| 2026-10-28 | RULED: stage 1 accepted as proposed; stage 2 (cap 1.2) deferred to a later leased read. BUILT behind the pin boundary: the judged arm only, the canary only (the startup gate keeps the scalar margin), the fallback at the cap when the null cannot bind, `canary_rule_version` on every record, `canary_dump_replay` judging a dump by its record's own bars. The frozen `backtest.py` refuses at the new pin by design (its `check_constants` pins the pre-ruling bars); the committed basis replaces it for CI | the orchestrator's ruling; `canary_row_bar_test` re-derives k and K_loss and backtests the census |
| 2026-10-29 | Stage 2's read REGISTERED (§7 PREREG) before any number; `canary_dump_replay --canary-shift-max` (anchor mode) built for it; the read packaged as the `canary-stage2` window of `main.ops.gpu_fill_window` | FINDING 5: the replay had no margin override |
| 2026-10-29 | Stage 2's read: **INCONCLUSIVE**, so the cap stays 0.72 (`~/gen3ai_archive/gpufill_2026-10-29/canary_stage2/verdict.json`). 19 of 20 checkpoints were clean; X26c @482M refused slice 2 in 3 of 3 on `spread_belief.nature_head.weight` (12.7) and read 11.95 (0.996 of K) on slice 0. Diagnosed in `../research_state/measurements/canary_slice2_2026-10-29/`: relocated at 1.2, golden row 9 (rho 9.37) carries 100 % of that gradient, and the fp32 dither null reaches 2.4 | the registered rule; the window budget (90 min against a measured 100) left the last two steps and the verdict to be run by hand |
