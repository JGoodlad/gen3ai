# X26 (70M) vs the toy at its 100M cap: INTERIM, DESCRIPTIVE (registration)

**Registered 2026-10-16 19:15 PDT, before any game of this family.** Owner's question: *"is production stronger than the toy?"*
at the current X26 point.

**What this is, and is not.**
- **DESCRIPTIVE and INTERIM.** No verdict rule, no §3.6 class is read from it.
- **It is NOT R-T2.** R-T2 ([`design_x26_readiness.md`](../../../endstate/design_x26_readiness.md) §3.5) is pre-registered for X26's
  PLATEAU checkpoint against each toy's 100M cap; this plays X26's 70M checkpoint, a LIVE run still climbing. Nothing here is G_p(cap).
- The toy is at its cap and still climbing there too (§3.5 amendment); X26 at 70M is far from its plateau, so the gap read
  here is biased toward the toy relative to R-T2.

**Cells** (production is the PLAYER, so each cell's score is production's):
- `models/rb_x26_s2601/checkpoints/checkpoint_70000008_steps.zip` (read-only, a live run's file) ×
  `rb_toy_ent_s4001b/final_model.zip`, `rb_toy_ent_s4002/final_model.zip`, `rb_toy_ent_s4003/final_model.zip`.
- 500 mirrored pairs per cell (1,000 games each), `main.h2h play-many`, 50-pair batches, CPU, `--torch-threads 1`,
  `--oracle-reveal-mode off`, family `xvt_interim` (`gen3_eval_protocol_v1_h2h`), the toy-crosses invocation
  ([`../toy_crosses_2026-10-16/run_crosses.sh`](../toy_crosses_2026-10-16/run_crosses.sh)).
- Rule 5b: `nice -n 19`, `taskset -c 5-7,13-15`, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`, `mem_cap.sh 16`.

**Load check** ([`plan_cells.py`](plan_cells.py), strict loader, historical kwargs, CPU, at main `d5880c87`'s `src/`): all four load.

**The statistic.** Pooled production score − 50 pp (a draw counts ½) = the mean of the three per-cell scores − 50. The interval
takes the three TOY SEEDS as the units: one-sample t over their three values (df 2), 90 % and 95 %. This is the toy-crosses
`rect_stat.py` taken to its one-column limit (X26 is a single production checkpoint, so there is no column variance):
**production's own seed variance is not in the interval.**
