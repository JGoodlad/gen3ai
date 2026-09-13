# `vf025` vs the three controls — the registered read (2026-09-13)

**Verdict: NOT DETECTED at the registered standard (PASS = the Δ CI on `cond.opp_class_auc.t4_10` clears +0.022 UPWARD at both draws vs all three controls). The lever is NOT monotone: a quarter of the control's coefficient leaves the row where 0.5 leaves it, while 1.5 strips it.** Ledger 2026-09-13 · *VERDICT · `vf025` NOT DETECTED*.

Arm `ai_v12_29_ladder_vf025@10000032` (`ctrl10M`'s argv with only `--vf-coef 0.25`, seed 1001, pin `f3502568`). Two offline 800-game draws (`hp_vf025.sh`, log beside it): `hp800/` seed 20260910 (floor 0.0219), `hp800b/` seed 20260911 (floor 0.0245).

| draw | control | arm | control | Δ [CI] | label |
|---|---|---|---|---|---|
| hp800 | `ctrl10M` | 0.6914 | 0.7101 | −0.0187 [−0.0334, −0.0041] | WITHIN FLOOR |
| hp800 | `ctrl10M_b` | 0.6914 | 0.7072 | −0.0158 [−0.0303, −0.0010] | WITHIN FLOOR |
| hp800 | `ctrl10M_c` | 0.6914 | 0.6885 | +0.0029 [−0.0115, +0.0184] | WITHIN FLOOR |
| hp800b | `ctrl10M` | 0.6788 | 0.7129 | −0.0341 [−0.0489, −0.0193] | NOT DETECTED |
| hp800b | `ctrl10M_b` | 0.6788 | 0.6928 | −0.0140 [−0.0295, +0.0012] | WITHIN FLOOR |
| hp800b | `ctrl10M_c` | 0.6788 | 0.6876 | −0.0088 [−0.0230, +0.0060] | WITHIN FLOOR |

Five of six within floor, one NOT DETECTED (clears zero downward, not the floor); mean −0.015. `t1_3` −0.002 to −0.030 (two DETECTED downward on draw 2). Gate rows within floor / n.d. on every pair. Crossing 4,128,768 on both sides of every pair. Delivery row (Training Run, `879cd051`): `grad/value_policy_logratio` median-20 = 0.095× at 0.25, 0.243–0.266× at 0.5 (three seeds), 0.84–1.23× at 1.5 — the dose was delivered; the response is super-linear in `vf_coef`.
