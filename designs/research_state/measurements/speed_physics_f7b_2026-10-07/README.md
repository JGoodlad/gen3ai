# F7b speed physics — real-battle calibration of P(we act first), 2026-10-07

**What.** `--speed-physics on` (`gen3_speed_physics_v1`, config v143) vs `off` (the production logistic
`sigmoid(Δspeed / 15)`), read on REAL battles: the M5 Lane S bank (`m5_laneS/bank_v1`, 580 trained-policy battles,
inputs only) replayed through the Rust core (`core_events --trackers --obs`), no poke-env. Both arms are read from
ONE model (the flag has no parameters): the op's PRE-gain P(our active acts before theirs at equal priority).

**Truth.** A turn qualifies when both players chose a MOVE; who acted first = the first `|move|` (no `[from]`),
`|cant|`, or confusion / Attract `-activate` line of either active after the decision. Both viewers' rows are read.
Unequal priorities test the bracket (`p_seat_first`, deterministic); equal priorities give the reliability table.

**Command** (CPU, ~6.8 GB peak, under `scripts/ops/mem_cap.sh 16`):

    python src/agents/model/speed_physics_bridge_integration_test.py --json calibration_coldstart.json
    python src/agents/model/speed_physics_bridge_integration_test.py \
        --ckpt models/rb_x5ab_blob_s1007/final_model.zip --json calibration_rb_x5ab_blob_s1007.json

Deterministic: a committed bank, a deterministic replay, CPU forwards. Code: this commit (base `f0d673fd`).

## Result — cold-start beliefs (a fresh production extractor: spread / item beliefs = their Smogon priors)

27,026 rows · 23,598 equal-priority · 3,428 unequal-priority (**0 contradicted**) · 200 turns skipped (Struggle /
recharge / a move outside the dex) · 0 rows with a true Quick Claw holder.

| bin | off n | off mean P | off observed | on n | on mean P | on observed |
|---|---|---|---|---|---|---|
| [0.00,0.05) | 8866 | 0.005 | 0.004 | 9012 | 0.003 | 0.005 |
| [0.05,0.20) | 1269 | 0.120 | 0.094 | 1116 | 0.115 | 0.052 |
| [0.20,0.35) | 871 | 0.263 | 0.326 | 937 | 0.282 | 0.340 |
| [0.35,0.50) | 748 | 0.445 | 0.282 | 700 | 0.445 | 0.333 |
| [0.50,0.65) | 696 | 0.592 | 0.501 | 627 | 0.584 | 0.628 |
| [0.65,0.80) | 887 | 0.724 | 0.724 | 829 | 0.727 | 0.593 |
| [0.80,0.95) | 1355 | 0.889 | 0.941 | 914 | 0.881 | 0.887 |
| [0.95,1.00] | 8906 | 0.995 | 0.997 | 9463 | 0.997 | 0.998 |

| arm | Brier | log loss | ECE | float32-certain and wrong |
|---|---|---|---|---|
| off (logistic) | 0.0425 | 0.1341 | 0.0157 | 0 |
| on (physics) | 0.0406 | 0.1367 | 0.0160 | 1 |

## Result — a trained blob arm's learned beliefs (`rb_x5ab_blob_s1007`, final)

| arm | Brier | log loss | ECE | float32-certain and wrong |
|---|---|---|---|---|
| off | 0.0369 | 0.1207 | 0.0237 | 0 |
| on | 0.0368 | 0.1308 | 0.0180 | 1 |

(Full tables: the two JSON files.)

## Read

- **Not clearly better calibrated.** `on` lowers the Brier score slightly and is sharper (more rows in the two
  saturated bins: 18,475 vs 17,772 cold-start), but its log loss is WORSE on both belief sources and its middle
  bins are no better aligned. No interval was computed: this is a descriptive read, not an A/B.
- **The one certain-and-wrong row** (`ai_v14_07_g0p_k2/step_90000000/heuristic2/win_s0_001`, turn 15): a Timid
  252-Speed Blissey (true speed 229) against our Swampert (214); the Smogon prior believes Blissey's speed at
  148 ± 5.3, so the Gaussian puts the true value 15 σ out and P rounds to exactly 1.0 in float32. The Gaussian tail
  under-covers a spread the Smogon prior never saw — a named residual; the logistic, insensitive to σ, read 1.0
  only at a gap above ~250.
- **Quick Claw**: banned in gen3ou (owner + Showdown master, 2026-10-07); the bank holds no true holder, so the
  format gate is not exercised by real data here.

## Re-read, 2026-10-07 (later): the Gaussian REPLACED by the discrete Smogon spreads mixture (`gen3_speed_mixture_v1`)

**What changed** (under `--speed-physics on` only; `off` untouched): their speed is no longer a Gaussian around
the spread belief's mean with the Smogon prior's σ, but the species' DISCRETE distribution over the Speed STAT
from EVERY chaos spread (`gen3_data.priors.all_spreads`, no top-25 cut): P(we first) = Σ_v w_v (1[ours > f(v)] +
½ 1[ours = f(v)]), each support point v through the same exact stage / paralysis arithmetic as ours. The learned
spread belief is NOT read by the speed sites any more (the mixture is the prior alone), so both belief sources
give the SAME `on` numbers.

**Command** (one collection, both reads; CPU, 6.75 GB peak, 28 min, under `scripts/ops/mem_cap.sh 16`): the
module's `collect` once, then `check(all)` and `check(all, ckpt)` — `calibration_coldstart_mixture.json`,
`calibration_rb_x5ab_blob_s1007_mixture.json`. The cold-start Gaussian was re-run at the pre-change commit
(`26131c0c`) and reproduced the table above EXACTLY; the trained-arm Gaussian row is the committed JSON (made at
`d02aded6`'s base — its `off` row differs from today's by ≤ 0.0002, so the code moved slightly between).

| beliefs | arm | Brier | log loss | ECE | certain and wrong |
|---|---|---|---|---|---|
| cold-start | off (logistic) | 0.0425 | 0.1341 | 0.0157 | 0 |
| cold-start | Gaussian (replaced) | 0.0406 | 0.1367 | 0.0160 | 1 |
| cold-start | **mixture** | 0.0440 | **0.1308** | 0.0184 | **0** |
| `rb_x5ab_blob_s1007` | off (logistic) | 0.0369 | 0.1208 | 0.0235 | 0 |
| `rb_x5ab_blob_s1007` | Gaussian (replaced; committed JSON) | 0.0368 | 0.1308 | 0.0180 | 1 |
| `rb_x5ab_blob_s1007` | **mixture** | 0.0440 | 0.1308 | 0.0184 | **0** |

Reliability (mixture; identical for both belief sources): [0.00,0.05) n 9,387 P 0.002 obs 0.003 · [0.05,0.20)
583 · 0.111 · 0.060 · [0.20,0.35) 475 · 0.275 · 0.219 · [0.35,0.50) 762 · 0.411 · 0.377 · [0.50,0.65) 664 · 0.568
· 0.599 · [0.65,0.80) 1,146 · 0.722 · 0.595 · [0.80,0.95) 1,195 · 0.865 · 0.737 · [0.95,1.00] 9,386 · 0.997 ·
1.000. Priority bracket: 3,428 rows, 0 contradicted.

**Read.** The mixture fixes the tail: the Blissey row is no longer certain-and-wrong (0 such rows), and the
cold-start log loss is the best of the three (0.1308 vs 0.1341 off, 0.1367 Gaussian). It is NOT better
calibrated overall: Brier and ECE are the worst of the three, and the upper-middle bins are OVER-confident
([0.65,0.95): P ~0.72–0.87 against 0.60–0.74 observed) — the Smogon usage mixture says "this species is usually
slower", while the battles' sets (trained-policy teams from the pool) are faster than usage more often than
usage says. On the trained arm the mixture discards what the learned belief knew (log loss equal to the
Gaussian's, Brier 0.0440 vs 0.0368). Descriptive, no interval. The next lever is conditioning the mixture on
the battle (an observed move order prunes the support), not a different prior.
