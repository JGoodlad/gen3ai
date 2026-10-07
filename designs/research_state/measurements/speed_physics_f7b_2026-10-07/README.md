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
