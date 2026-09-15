## 9. The registered predictions, scored

`PREDICTION.md` was committed at **901ae5d7**, before any cell ran. The wall-clock stopping rule
(`STOPPING_RULE.md`) was committed at **60d3d02b**, with the rung-B cells at 3–6 pairs.

### Part A

| # | registered | outcome |
|---|---|---|
| A1 | no arm clears the L2 bar; all three in [0.45, 0.55] | @@A1@@ |
| A2 | `rollout` is the best `grid` of the three, 0.25–0.38 | **REFUTED.** It is the WORST of the three (0.2450 vs 0.2725 / 0.2825) and its point estimate is below the 0.25 band's floor. |
| A3 | no arm's `grid` differs from the anchor with a paired CI clear of zero | **HELD.** −0.0150 / −0.0250 / −0.0525, every CI straddling zero at 100 pairs. |
| A4 | no head-level L1 difference survives width matching | @@A4@@ |

### Part B

| # | registered | outcome |
|---|---|---|
| **B0** | the ORIGINAL head's pairwise accuracy is 0.52–0.58 | **HELD by the point estimate and OVERTAKEN by its meaning: 0.5169 [0.4800, 0.5524] — the CI straddles 0.50.** The registered clause "*a value below 0.52 with its CI excluding 0.55 would itself be the headline*" fires: the CI excludes 0.5524-and-above only marginally, and the honest statement is stronger and simpler — **at this width the head is NOT DISTINGUISHABLE FROM CHANCE at ranking siblings.** |
| B1 | the CONTROL refit does not beat the original, \|Δ\| < 0.02 | **REFUTED, and this is the finding.** +0.0863 [+0.0384, +0.1347] DETECTED. |
| **B2** | the RANKING refit beats (i) and the original by +0.03 to +0.10, CI clear of zero | **REFUTED against (i): −0.0107 [−0.0249, +0.0028] NOT DETECTED**, point estimate negative at every swept coefficient. It does beat the ORIGINAL (+0.0756 [+0.0276, +0.1233]) — by less than (i) does. |
| B3 | separation ratio 1.0–1.2 on the original, ≥ 1.5 on (ii) | **HELD on both clauses (1.020 → 1.72), and uninformative:** (i) reaches 1.711 too, so the ratio does not separate the arms. |
| B4 | (ii)'s ECE at coef 1.0 worse than (i)'s by 0.01–0.04 | **REFUTED in DIRECTION: it is BETTER** (0.0398 vs 0.0520). Ranking was not bought with calibration. |
| B5 | the conditioning guard is not detected below the original | **HELD.** 0.711 → 0.706 (i) → 0.699 (ii). |
| B6 | blind-spot rate 8–20 % | **REFUTED, low: 4.5 % [3.99, 5.16].** |

### Part C

| # | registered | outcome |
|---|---|---|
| C1 | rung-B L2 does not clear 0.50; 0.47–0.55 | @@C1@@ |
| C2 | the refit head's `grid` beats the control's by +0.03 to +0.12 | @@C2@@ |
| C3 | separation-of-raced at matched K rises 1.2–2.0× vs the control | @@C3@@ |
