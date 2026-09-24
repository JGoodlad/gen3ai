# Best-response gap

`gap = win rate − 0.5`; the population loop is working iff the gap FALLS round over round.

* **stat**: `pooled`
* **regime**: greedy-vs-greedy (EVAL regime, main.eval_worker FIXED branch)

## Round 1 — target `ai_v13_12_plateau` @95,158,272

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_18_teach5_offense_hidose` | 0.6600 | 0.6400 | 400 | **+14.00** | [+9.18, +18.55] |

## Round 2 — target `ai_v13_23_popr1_ctrl` @103,219,200

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_25_popr1_read_ctrl` | 0.6900 | 0.6700 | 400 | **+17.00** | [+12.25, +21.43] |

## Round 3 — target `ai_v13_22_popr1_loop` @103,219,200

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_24_popr1_read_loop` | 0.6000 | 0.5700 | 400 | **+7.00** | [+2.10, +11.76] |

## Δ round 2 − round 1 (paired on archetype)

| archetype | gap r1 | gap r2 | Δ pp | 95% CI pp |
|---|---:|---:|---:|---|
| offense | +14.00 | +17.00 | **+3.00** | [-3.58, +9.54] |

**MEAN Δ over 1 archetype(s):** —

**VERDICT: UNREADABLE — fewer than two archetypes are present in both rounds**

## Δ round 3 − round 2 (paired on archetype)

| archetype | gap r2 | gap r3 | Δ pp | 95% CI pp |
|---|---:|---:|---:|---|
| offense | +17.00 | +7.00 | **-10.00** | [-16.60, -3.27] |

**MEAN Δ over 1 archetype(s):** —

**VERDICT: UNREADABLE — fewer than two archetypes are present in both rounds**
