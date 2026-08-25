# End-of-run battery — `ai_v9_34_tick1_0824`

## 1. Ladder (tail-4, matched-count convention)
- current: **2012.2** ± 29.9
- reference: 2110.0 ± 29.6
- **INFERIOR** (Δ -97.8, CI [-139.9, -55.7]; rule: delta >= -15.0 and CI-low > -40.0)
- current hodge: spine 889 ELO · width 45 raw / 36 null / **26 excess** ELO (p=0.0149) · cyclic 0.9% (null-adj) · 0 significant 3-cycles / 100 triangles
- reference hodge: spine 980 ELO · width 61 raw / 36 null / **49 excess** ELO (p=0.005) · cyclic 4.0% (null-adj) · 5 significant 3-cycles / 814 triangles
  - cycle: 16.0M > 24.0M > 22.0M > 16.0M (curl +162 ELO, z=3.25)
  - cycle: 14.0M > 18.0M > 16.0M > 14.0M (curl +154 ELO, z=3.09)
  - cycle: 12.0M > 16.0M > 14.0M > 12.0M (curl +144 ELO, z=2.89)

## 2. Critic routes + edge families
- `threat`: **DELETION_CANDIDATE** (|dV| 13% of all_off, flips 0.00%)
- `entity_pool`: **KEEP** (|dV| 108% of all_off, flips 0.00%)
- `nmr`: **READ**
- `event_seats`: **READ**
- family `h`: **ALIVE** (|dV| 0.147 vs median 0.1011)

## 3. Awareness + coverage (vs gen-10 baselines)
- blind_loss_fraction: **WORSE** (0.092 vs 0.072)
- median_lead_time: **WORSE** (4.0 vs 7.0)
- cap_aware_ge_bar_fraction: **IMPROVED** (1.0 vs 0.5)
- coverage80: **WORSE** (0.3745 vs 0.44)
- pit_mean: **IMPROVED** (0.4129 vs 0.396)

## 4. Mechanic usage (G2 — conditional execution)
- focuspunch: pick 1.5% of 1446 (mean prob 4.3%) — ref 6.7%
- substitute: pick 13.1% of 3135 (mean prob 15.2%) — ref 5.2%
- endeavor: pick 13.0% of 154 (mean prob 11.8%) — ref 5.7%
- explosion: pick 26.8% of 3131 (mean prob 25.8%) — ref 24.5%
- selfdestruct: pick 13.7% of 1900 (mean prob 13.4%) — ref 20.2%
- pursuit: pick 12.2% of 328 (mean prob 14.4%) — ref 5.6%

*Verdicts are decision-support against the pre-registered runbook rules; the runbooks remain the registration of record.*
