# Best-response gap

`gap = win rate − 0.5`; the population loop is working iff the gap FALLS round over round.

* **stat**: `pooled`
* **regime**: greedy-vs-greedy (EVAL regime, main.eval_worker FIXED branch)

## Round 2 — target `ai_v13_23_popr1_ctrl` @103,219,200

budget 4,030,464 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_32_popr1_read_ctrl_ext` | 0.6800 | 0.6350 | 200 | **+13.50** | [+6.63, +19.86] |

## Round 3 — target `ai_v13_22_popr1_loop` @103,219,200

budget 4,030,464 steps · dose 3.815e-08

| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---|
| offense | 5/5 | `ai_v13_31_popr1_read_loop_ext` | 0.5800 | 0.6000 | 200 | **+10.00** | [+3.08, +16.54] |

## Δ round 3 − round 2 (paired on archetype)

| archetype | gap r2 | gap r3 | Δ pp | 95% CI pp |
|---|---:|---:|---:|---|
| offense | +13.50 | +10.00 | **-3.50** | [-12.90, +5.98] |

**MEAN Δ over 1 archetype(s):** —

**VERDICT: UNREADABLE — fewer than two archetypes are present in both rounds**
