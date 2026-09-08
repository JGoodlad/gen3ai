---
name: project_stall_recovery_analysis
description: "Forensic split of the E_OTHER value-cliff residual (run_20260601 @122M) — recovery-reset is small/clean, tempo-swing is the real E_OTHER majority; Suicune-Rest is the discriminator"
metadata: 
  node_type: memory
  type: project
  originSessionId: c092d34c-288c-467a-a45d-c3d160e22838
---

Investigated the "stall/recovery-reset" value-cliff losses (the E_OTHER bucket the incoming-damage
feature does NOT fix) for `run_20260601_193826` @122M, and **the eyeball hypothesis was wrong on
magnitude** — measured, don't assume.

**Measured (extend `designs/ai_v5/falsifier_cliff_attribution.py` → `falsifier_recovery_attribution.py`):**
- Decisive turning points (V>5, ΔV<−15): incoming-addressable 56%, **E_OTHER 34%**. Of E_OTHER:
  **RECOVERY_RESET only ~3% of TPs (9% of E_OTHER); TEMPO/switch-matchup swing ~31% of TPs (91% of
  E_OTHER).** Raw-cliff level: 53% of E_OTHER is ALREADY_LOSING (V<0, downstream — exclude).
- TEMPO_SWING is heterogeneous (73% blank-coverage, ~half switch-driven, ~no our-HP loss) → **mostly
  the incoming-damage feature's residual** (never-blank + speed bit), NOT a fresh single-feature
  category. Hand it back to that design to measure.
- **Recovery-reset is the only clean, orthogonal, single-belief-fixable residual.** Battle-level:
  ~10% of losses are recovery-stalls. **The discriminator is Suicune-Rest: healed-against 127× in
  losses vs 2× in wins** (Blissey appears in both). Mechanism: **Rest cures the Toxic clock AND fully
  heals → unbreakable for our stall team; Softboiled does NOT cure status → Blissey stays Toxic-able.**
  This is the "Rest-resets-a-won-Toxic-stall" pathology from [[project_loss_analysis_run20260531_v2]].

**Prober (exact tier):** critic prices a 20%-HP Toxic'd(counter 5) Suicune at **+27.6 (won)**, then
Rest → V crashes −54; our whole 6-mon stall team lost to one Suicune. Reward bug still live:
`_compute_futile_attack_penalty` (reward_manager.py:715) skips Rest turns + single-turn only → no
net-progress accounting over the chip→heal cycle.

**Design:** `designs/ai_v5/design_stall_recovery_obs.md` — 4 opp-active scalars (recovery_rate,
**cures_status**=the discriminator, known, net_chip_margin) on the `non_matchup_rest` lane + windowed
futile reward; **fold into the incoming-damage retrain** (shared move-usage prior, `_estimate_damage_
fraction`+fixed-damage branch, ARCH bump) — 3% scope doesn't justify a solo retrain. Builds on
[[project_loss_analysis_run20260601]]. 4-lens reviewed (gameplay/ML-arch/data-perf/red-team), folded in.
