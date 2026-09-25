# Best-response gap

`gap = win rate − 0.5`; the population loop is working iff the gap FALLS round over round.

* **stat**: `pooled`
* **regime**: greedy-vs-greedy (EVAL regime, main.eval_worker FIXED branch)

## Round 1 — target `ai_v13_12_plateau` @95,158,272

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_18_teach5_offense_hidose` | 0.6600 | 0.6400 | 400 | **+14.00** | [+9.18, +18.55] |

## Round 2 — target `ai_v13_22_popr1_loop` @103,219,200

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_24_popr1_read_loop` | 0.6000 | 0.5700 | 400 | **+7.00** | [+2.10, +11.76] |

## Round 3 — target `ai_v13_27_popr2_loop` @111,280,128

budget 8,060,928 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_29_popr2_read_loop` | 0.5500 | 0.5250 | 400 | **+2.50** | [-2.39, +7.35] |

## Δ round 2 − round 1 (paired on archetype)

| archetype | gap r1 | gap r2 | Δ pp | 95% CI pp |
|---|---:|---:|---:|---|
| offense | +14.00 | +7.00 | **-7.00** | [-13.68, -0.23] |

**MEAN Δ over 1 archetype(s):** —

**VERDICT: UNREADABLE — fewer than two archetypes are present in both rounds**

## Δ round 3 − round 2 (paired on archetype)

| archetype | gap r2 | gap r3 | Δ pp | 95% CI pp |
|---|---:|---:|---:|---|
| offense | +7.00 | +2.50 | **-4.50** | [-11.33, +2.39] |

**MEAN Δ over 1 archetype(s):** —

**VERDICT: UNREADABLE — fewer than two archetypes are present in both rounds**
