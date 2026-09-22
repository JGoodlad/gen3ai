# Best-response gap

`gap = win rate − 0.5`; the population loop is working iff the gap FALLS round over round.

* **stat**: `pooled`
* **regime**: greedy-vs-greedy (EVAL regime, main.eval_worker FIXED branch)

> 🚨 **UNMATCHED** — printed under `--allow-unmatched`. Confounds:
>
> * DOSE MISMATCH — ai_v13_05_exploit_big5starmie vs ai_v13_13_exploit5_offense: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_05_exploit_big5starmie vs ai_v13_14_exploit5_balance: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_05_exploit_big5starmie vs ai_v13_15_exploit5_stall: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_06_exploit_ddtar_spikes vs ai_v13_13_exploit5_offense: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_06_exploit_ddtar_spikes vs ai_v13_14_exploit5_balance: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_06_exploit_ddtar_spikes vs ai_v13_15_exploit5_stall: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_10_exploit_stall vs ai_v13_13_exploit5_offense: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_10_exploit_stall vs ai_v13_14_exploit5_balance: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
> * DOSE MISMATCH — ai_v13_10_exploit_stall vs ai_v13_15_exploit5_stall: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)

## Round 1 — target `ai_v13_02_flywheel_winprob` @75,005,952

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| balance | 1/5 | `ai_v13_05_exploit_big5starmie` | 0.7400 | 0.6975 | 400 | **+19.75** | [+15.08, +24.05] |
| offense | 1/5 | `ai_v13_06_exploit_ddtar_spikes` | 0.7400 | 0.7375 | 400 | **+23.75** | [+19.23, +27.82] |
| stall | 1/5 | `ai_v13_10_exploit_stall` | 0.7400 | 0.7225 | 400 | **+22.25** | [+17.67, +26.41] |

## Round 2 — target `ai_v13_12_plateau` @95,158,272

budget 8,060,928 steps · dose 8.392e-09

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| balance | 5/5 | `ai_v13_14_exploit5_balance` | 0.5300 | 0.5250 | 400 | **+2.50** | [-2.39, +7.35] |
| offense | 5/5 | `ai_v13_13_exploit5_offense` | 0.7000 | 0.6575 | 400 | **+15.75** | [+10.97, +20.23] |
| stall | 5/5 | `ai_v13_15_exploit5_stall` | 0.4500 | 0.4550 | 400 | **-4.50** | [-9.31, +0.40] |

> ⚠ **CAVEAT:** round 1 -> 2, archetype balance: the TEAMSET SIZE changed (1/5 -> 5/5). The two gaps are best responses inside DIFFERENT subgame restrictions, so the delta confounds 'the loop absorbed it' with 'the later exploiter had a harder job'. `main.exploitability`'s caveat 2 is the same one.

> ⚠ **CAVEAT:** round 1 -> 2, archetype offense: the TEAMSET SIZE changed (1/5 -> 5/5). The two gaps are best responses inside DIFFERENT subgame restrictions, so the delta confounds 'the loop absorbed it' with 'the later exploiter had a harder job'. `main.exploitability`'s caveat 2 is the same one.

> ⚠ **CAVEAT:** round 1 -> 2, archetype stall: the TEAMSET SIZE changed (1/5 -> 5/5). The two gaps are best responses inside DIFFERENT subgame restrictions, so the delta confounds 'the loop absorbed it' with 'the later exploiter had a harder job'. `main.exploitability`'s caveat 2 is the same one.

## Δ round 2 − round 1 (paired on archetype)

| archetype | gap r1 | gap r2 | Δ pp | 95% CI pp |
|---|---:|---:|---:|---|
| balance | +19.75 | +2.50 | **-17.25** | [-23.76, -10.52] |
| offense | +23.75 | +15.75 | **-8.00** | [-14.28, -1.63] |
| stall | +22.25 | -4.50 | **-26.75** | [-33.11, -20.04] |

**MEAN Δ over 3 archetype(s):** **-17.33 pp** [-26.67, -8.00]

**VERDICT: THE GAP FELL — the generalist is absorbing its best responders**
