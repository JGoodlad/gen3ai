# `ent05` vs the three controls — the free conditioning read (2026-09-12)

**Verdict: NOT DETECTED at the registered standard; an UPWARD CANDIDATE.** Ledger 2026-09-12 · *READ · `ent05`'s conditioning row*.

Arm `ai_v12_28_ladder_ent05@10000032` (`ctrl10M`'s argv with only `--ent-coef 0.05`, seed 1001, pin `f3502568`). Two offline 800-game draws (`hp_ent05.sh`, log beside it): `hp800/` seed 20260910 with `hp800_floor_v6.json` (class AUC t4–10 floor 0.0219); `hp800b/` seed 20260911 with `hp800b_floor_v6.json` (0.0245). Registered before the read: PASS either direction = the Δ CI on `cond.opp_class_auc.t4_10` clears ±0.022 at both draws against all three controls.

| draw | control | arm | control | Δ [CI] | label |
|---|---|---|---|---|---|
| hp800 | `ctrl10M` | 0.7348 | 0.7109 | +0.0239 [+0.0094, +0.0379] | NOT DETECTED |
| hp800 | `ctrl10M_b` | 0.7348 | 0.7064 | +0.0284 [+0.0141, +0.0433] | NOT DETECTED |
| hp800 | `ctrl10M_c` | 0.7348 | 0.6775 | **+0.0573 [+0.0429, +0.0726]** | DETECTED |
| hp800b | `ctrl10M` | 0.7272 | 0.7130 | +0.0143 [−0.0007, +0.0287] | WITHIN FLOOR |
| hp800b | `ctrl10M_b` | 0.7272 | 0.6919 | +0.0353 [+0.0201, +0.0508] | NOT DETECTED |
| hp800b | `ctrl10M_c` | 0.7272 | 0.6907 | +0.0365 [+0.0222, +0.0506] | NOT DETECTED (lower bound 0.0222 < 0.0245) |

Six of six points positive (mean +0.033), four of six CIs clear zero, one of six clears the floor. `t1_3` +0.011 to +0.047 (three of six DETECTED). `gate.resolution.all` within floor / n.d. on every pair; `gate.resolution.bot` DETECTED up on draw 2 vs `ctrl10M` and `_b` (+0.011 / +0.012), within floor on draw 1. Crossing 4,128,768 on both sides of every pair. Each `vs_*/` holds `critic_read.md`, `critic_read.json` and `cond_readout.json` where written; the ARM column's bracket on the class-AUC rows is not the arm's own CI (backlog row 2026-09-12).
