# Opponent-checkpoint isolation cell — greedy / team set M / rev-1 FINAL (25M) opponent (2026-09-01)

**Pre-registered (chain.sh, 2026-09-01 13:48, before any game was played):** the greedy / set M
cell of `greedy_meter_vs_composition_2026-09-01.md` used rev-1's **24M snapshot** as the opponent;
probe Q used rev-1's **25M `final_model.zip`**. This cell re-runs the greedy / set M contrast against
the 25M opponent, so its delta from the 24M cell isolates the opponent-checkpoint axis EXACTLY,
holding policy regime and team set fixed. Read: |delta| < ~1.5pp ⇒ the 3.62pp "composition +
opponent checkpoint" column of that record is (almost all) TEAM COMPOSITION; large ⇒ the
checkpoint carries it.

**Instrument:** `arm_25M.py` is the verbatim untaught-8 meter (`axis_split_untaught_arm.py`) with
the target path pointed at `final_model.zip`; n = 200/team fixed pre-data, seeds `1000 + slice
index`, rust bridge, greedy (`stochastic=False`), `GEN3AI_TIMEOUT_SCALE=12`, niced 15, concurrency
3. Both arms sequential; R2ACTION arm took 2 attempts (the first was interrupted by the
documented per-team resume — byte-equivalent). 3,200 new battles. Inputs and logs in
`oppckpt_isolation_inputs/`.

## Result

| arm (vs rev-1 @25M, greedy, set M) | wins / games | WR |
|---|---|---|
| `REV1FIN` (rev-1 final, the mirror) | 898 / 1600 | 0.5613 |
| `R2ACTION` (rev-2 fold) | 838 / 1600 | 0.5238 |
| **hop R2ACTION − REV1FIN** | | **−3.75pp [−7.2, −0.3], z = −2.13** (binomial) |

Against the 24M-opponent cell: **−3.44pp [−6.19, −0.31]**. **Opponent-checkpoint delta =
−0.31pp** — inside the pre-registered ~1.5pp bar by a factor of five.

Per team (hop, pp): 24M cell → 25M cell: `U_61590463` −7.0 → **+6.0** · `U_90b94599` −4.0 → −9.0 ·
`U_92832108` −2.0 → −2.5 · `U_9909f2e9` −2.0 → −4.0 · `U_9d5f8458` −9.0 → −13.0 · `U_ce35b736` −7.5
→ −7.0 · `U_dbf81d8e` +5.5 → 0.0 · `U_f7ba5702` −1.5 → −0.5. Sign agreement 6/8 (one flip, one to
zero); pooled magnitude agrees to 0.3pp. Direction negative on 6 of 8 teams.

## Reading

1. **The opponent checkpoint (24M vs 25M) is NOT an axis that matters** for the untaught-8
   meter: ~0.3pp, a tenth of the effect and a tenth of the replicate floor. The 3.62pp
   "composition (+ opponent ckpt)" column in `greedy_meter_vs_composition_2026-09-01.md` is
   **team-set composition**. Untaught numbers still carry the three-part stamp, but the checkpoint
   part can be read as inert between adjacent snapshots of the same run.
2. **The greedy / set M hop replicates**: −3.44 and −3.75 on two independent draws of 3,200
   battles. That is a real, small, negative rev-2 effect under the greedy regime — and it still
   sits inside M9's **4.19pp** untaught replicate floor. Nothing here rescues or convicts the
   robbery; it removes one confound from the record.
3. Per-team effects at n=200 are noise-dominated (one team flips sign by 13pp between two
   opponent snapshots that differ by 1M steps) — the pooled read is the only one to quote.

**Meter stamp:** greedy · opponent rev-1 `final_model.zip` (25M) · team set M.
