# The X14 ON arm's K9(b) flag: diagnosis (2026-10-21)

**Verdict: neither (a) nor (b). The flag was a reader artifact, and the two rows at or over 1e-4 are (c), the
healthy fp32 tail.** The X14 ON arm's K9(b) arm-end read is PASS once the artifact is removed. Nothing here
argues against overlap.

Inputs: the battery's own artifacts, read-only. They are `cells/x14_{off,on}.json`, both arms' TensorBoard and
`behaviour_violations.jsonl` with the dumped rows and snapshot weights (`~/gen3ai_archive/x14_ab_runs_2026-10-21/`),
and X26's serial run (`models/rb_x26_s2601`). The replay ran on the CPU (`CUDA_VISIBLE_DEVICES=""`) at a checkout
whose `src/` equals the battery pin `f4b0a74f2` outside tests. Reproduce with `x14_k9b_diag.py`; the numbers are in
`x14_k9b_diag.json`.

## 1. What actually fired

The read said `VOID`, `fired_families ['behaviour']`. The family that fired is **not** the snapshot family and
**not** the 1e-4 rows:

- **`behaviour` (the learner's own probe) fired on 13 NaN p99 rows.** Under overlap, the learner's probe has **0 current
  rows** on every update after each process's first (`behaviour/rows_current` is 0 on 12 distinct steps). For
  that case `consistency.behaviour_probe` wrote `p99_abs_dlogp_current` and `max_abs_dlogp_current` as **NaN** to
  mean "empty". X35 AM-6's reader (`main.x35.batch.judge_k9b`) reads any non-finite value as a non-finite |Δ| and
  VOIDs. So every overlap arm would have VOIDed this way, at any weights. The step `381,481,870` is listed twice
  because the final dump writes the last update's rows at the step of the one before (`loop._final_dump`): two real
  updates, not a duplicate (corrected 2026-10-21: 15 rows are 15 updates on 14 steps, and the K9(b) reader counts rows).
- **`behaviour_snapshot` (the check that judges the stale rows) PASSED AM-6**: judged max 1.26e-4 against the 1e-3
  ceiling, and p99 max 3.13e-5 against 1e-4. The line "2 judged rows ≥ 1e-4" is a printed COUNT
  (`n_judged_at_or_over_1e-4`), not a ceiling. The 1e-4 bar is the in-run `fatal` gate's. Production arms don't run
  that gate: they run `warn` under AM-6's ceilings (X26 Amendment 3; `design_compute_efficiency.md` F14).
- **x14_off is clean** (verified): `behaviour` PASS, judged max 9.73e-5, p99 max 2.89e-5, 0 rows ≥ 1e-4.

With the NaN-at-zero-rows placeholder excluded, the ON arm reads:

| family | judged max | p99 max | AM-6 verdict |
|---|---|---|---|
| `behaviour_snapshot` | 1.26e-4 | 3.13e-5 | **PASS** |
| `behaviour` (2 finite updates) | 7.25e-5 | 3.05e-5 | PASS (a near-vacuous family on an overlap arm) |

## 2. The two rows at or over 1e-4

| update | step | rows played at | learner at | max \|Δ\| | over the bar by | next judged row | p99 | judged / excluded |
|---|---|---|---|---|---|---|---|---|
| 38,730 | 381,285,164 | v11 (snapshot) | v12 | 1.066e-4 | +6.6 % | 6.97e-5 | 2.65e-5 | 974 / 50 |
| 38,750 | 381,481,870 | v13 (snapshot) | v14 | 1.261e-4 | +26.1 % | 3.87e-5 | 2.81e-5 | 956 / 68 |

In each case one isolated row sits over the bar. Each update's full-buffer scan (4,096 current rows) found **0**
rows over the bar (scan maxima 9.69e-5 and 5.79e-5).

**Why not (b), a check that assumes no lag.** The check already judges the stale rows against the weights that
played them: the `where` field reads "the BEHAVIOUR SNAPSHOT the overlapped rollout played", v11 against a learner at
v12. A no-lag check would compare one-version-old rows with the newer weights. X26 measured that gap at mean
|π/μ − 1| = 0.106 at age 1, and the bar's planted one-update-stale fault reads 0.108. That is about 1,000× these
values, and the bulk of the rows would move with it. Here the p99 is 2.65e-5 and 2.81e-5, the same as the OFF arm's
2.48e-5 median. No lag-aware rule is needed, so none is proposed. The IS correction (`proximal_transform`) is the
objective's business, not K9(b)'s.

**Why not (a), weights that did not act.**

- A stale or raced weight copy moves whole flushes, at ≥ 2e-2 (the smallest planted stale fault). These are single
  rows at about 1e-4.
- The dumped snapshot weights (`behaviour_violation_u*_policy.pt`, saved inside the swap) reproduce T2's stored μ
  on the **CPU**, a third fp32 program. For the scan's worst 32 rows, |T2 − CPU| has a median of 3.06e-5 / 2.96e-5
  and a max of 9.3e-5 / 9.5e-5. Rows from v12 or v14 would sit about 0.1 away. So the snapshot is the version that
  acted, confirmed independently of the CUDA forwards.
- The process-wide state of the T2-attribution finding 6 cannot reach these values. T2 serves through the `graph`
  backend, which replays captured CUDA graphs whose kernels were fixed at capture. T2's frozen guard checks the fp32
  matmul precision after every flush, and the run did not poison. No training code sets the precision; the only
  setter is `utils.torch_state_guard`, a scoped restore used by tests and benches. The probe runs on the main
  thread with the worker joined.

**Why (c).**

- **The violators are rounding-sensitive rows.** A ~1-ulp jitter of every weight (8 seeded CPU forwards) moves
  their log π by 7.5e-5 and 5.8e-5, which is 75 % and 58 % of the bar. They rank 30th and 27th of 32 against the
  scan's worst rows (median 3.7e-5 / 3.4e-5). Between fp32 programs, 1e-4 is ordinary on such rows: even the
  learner's eager CUDA recompute and the CPU disagree by up to 1.03e-4 on the scan rows.
- On both violators, T2's graph is the program that sits apart: |T2 − CPU| is 8.8e-5 and 1.33e-4, against
  |learner − CPU| of 1.9e-5 and 6.4e-6.
- **The serial production run shows the same tail.** X26 (serial, `warn`) logged 159 violations up to 380M. Over
  the 1,000 updates before the fork, **12.6 %** of updates had a judged max ≥ 1e-4 (median 6.08e-5). Over the last
  200, the share was **25.5 %** (median 7.02e-5). The ON arm's 2 of 13 has P = 0.50 under the 1,000-update rate.
  The ON arm's per-update maxima are indistinguishable from X26's last 200 (Mann-Whitney p = 0.37).
- **Rule 8 does not apply to these rows.** Both rows are judged. Their tie margins (7.3e-3, and 4.6e-4 at
  `hypothesis_set.py:284 argsort`) are at or above `FP32_TIE_EPS` = 2e-4, so neither sits within a rounding error of
  a discrete cutoff. Both also sit 8× under AM-6's 1e-3 ceiling. It is the 1e-4 `fatal` bar, derived on A2's 4.0M
  checkpoint (judged max 2.1e-5 over 3.4M rows), that this checkpoint's healthy tail outgrew. That is already known
  and is why X26 and X35 run `warn` (Amendment 3: about 110 rounding stops by 500M).

**FINDING, unexplained, at the 1e-5 level.** At the same pin, the ON arm's per-update judged max runs higher than
the OFF arm's: median 7.5e-5 vs 5.65e-5, Mann-Whitney p = 0.006 over 13 vs 14 updates. Yet ON is in line with X26's
serial last 200 (p = 0.37), and it is OFF that sits low against them (p = 0.008; 0 of 14 at X26's recent 25.5 %
rate: P = 0.016). So which arm moved is not settled. The shift is 3 orders
of magnitude under any wrong-weights signature, and a CPU replay cannot place it. A candidate, unverified: the T2
bucket mix under a concurrent learner. The learning A/B should log both arms' per-update maxima and re-test this on
more updates.

## 3. The fix (built, `gen3_k9b_no_row_no_tag_v1`)

`consistency.behaviour_probe` now writes `max_abs_dlogp_current` / `p99_abs_dlogp_current` only when there are
current rows. Absence means nothing was judged, as with `max_abs_dlogp_judged`. A real NaN on a current row still
writes NaN and still FATALs. The test is
`consistency_test::test_an_update_with_no_current_rows_writes_no_current_row_statistic_never_a_NaN`, which FAILS on
revert: it asserts the tags are absent and that AM-6's judge reads such a series as INCONCLUSIVE, never VOID. A
serial run is unchanged, since every update has ≥ 1,024 current rows. The archived battery TensorBoard keeps its
NaN rows; §1 is its corrected read.

## 4. Decision for the orchestrator: before an X14 learning A/B is registered

1. **Land this fix** (shipped with this note). Without it, AM-6 VOIDs every ON arm.
2. **The ON arm's arm-end K9(b) must REQUIRE `behaviour_snapshot/*`.** `main.x35.batch.k9b_read` reads only
   `behaviour/*`. After the fix, an overlap arm's `behaviour/*` holds only one or two serial updates per process, so
   it would PASS near-vacuously. This is TECH_DEBT §2(b), the "X35 AM-6's arm-end K9(b) reader cannot judge an
   `--overlap-collect` arm" row, and it blocks registration. The battery's `x14_read.K9B_REQUIRED` already holds the
   rule. Key it on the run's recorded `overlap_collect`. **Built 2026-10-21**: `main.x35.batch.judge_k9b(..., overlap=)` /
   `k9b_read` (`batch_test.py`; TECH_DEBT §3).
3. **Both arms type `--behaviour-check warn` under AM-6's ceilings** (judged max < 1e-3, p99 < 1e-4, no non-finite
   value). That is the production regime, unchanged. `fatal` would stop either arm by rounding: X26's recent rate is
   about 1 update in 4.
4. **K9(b) needs no lag-aware rule.** The snapshot already judges the stale rows against the weights that played
   them, under the same bar and tie rule.
5. **Register the §2 FINDING as a secondary read** (per-update judged-max distribution, ON vs OFF). It is not a gate.
