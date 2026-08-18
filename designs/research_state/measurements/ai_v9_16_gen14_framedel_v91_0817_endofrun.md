> ⚠️ **This is `main.endofrun`'s RAW tool output, not the verdict record.** Its §1 line comes from
> `main.elo`'s SPARSE fit (`step_elo` calls `elo_cli.analyze`), which the runbook's §0 designates as
> **ORIENTATION and never the verdict** — it reads Δ −14.1, CI [−53.3, +25.1] = INCONCLUSIVE. The
> pre-registered headline artifact is the DENSE `snapshot_ladder/ladder.json` tail-4, which after
> the authorised 4× tie-break reads **Δ −38.30, CI [−55.04, −21.56] = INFERIOR**.
> The verdict record is [`gen14_endofrun_battery.json`](gen14_endofrun_battery.json); this file is
> kept because §0 requires the disagreement between the two artifacts to be reported, and that is
> only checkable if both are on disk.

# End-of-run battery — `ai_v9_16_gen14_framedel_v91_0817`

## 1. Ladder (tail-4, matched-count convention)
- current: **2042.8** ± 27.5
- reference: 2056.8 ± 28.0
- **INCONCLUSIVE** (Δ -14.1, CI [-53.3, 25.1]; rule: delta >= -15.0 and CI-low > -40.0)

## 2. Critic routes + edge families
- `seed`: **DELETION_CANDIDATE** (|dV| 0% of all_off, flips 0.00%)
- `threat`: **KEEP** (|dV| 22% of all_off, flips 0.00%)
- `hidden_opp_vf`: **DELETION_CANDIDATE** (|dV| 0% of all_off, flips 0.00%)
- `intent_reduce`: **DELETION_CANDIDATE** (|dV| 6% of all_off, flips 0.00%)
- `entity_pool`: **KEEP** (|dV| 96% of all_off, flips 0.00%)
- `nmr`: **READ**
- `event_seats`: **READ**
- `hidden_opp_pi`: **READ**
- `hidden_opp_both`: **READ**
- family `h`: **ALIVE** (|dV| 0.1769 vs median 0.0534)

## 3. Awareness + coverage (vs gen-10 baselines)
- blind_loss_fraction: **WORSE** (0.138 vs 0.072)
- median_lead_time: **WORSE** (5.0 vs 7.0)
- cap_aware_ge_bar_fraction: **IMPROVED** (0.765 vs 0.5)
- coverage80: **WORSE** (0.3931 vs 0.44)
- pit_mean: **WORSE** (0.3465 vs 0.396)

## 4. Mechanic usage (G2 — conditional execution)
- focuspunch: pick 3.4% of 2625 (mean prob 4.9%) — ref 5.1%
- substitute: pick 3.8% of 4717 (mean prob 6.0%) — ref 0.1%
- endure: pick 0.0% of 64 (mean prob 0.2%) — ref 0.0%
- destinybond: pick 11.2% of 80 (mean prob 8.2%) — ref 23.3%
- endeavor: pick 8.3% of 96 (mean prob 9.6%) — ref 4.7%
- counter: pick 11.5% of 1409 (mean prob 14.1%) — ref 3.7%
- explosion: pick 6.3% of 12719 (mean prob 7.3%) — ref 5.2%
- selfdestruct: pick 5.0% of 1506 (mean prob 7.6%) — ref 3.1%
- pursuit: pick 7.0% of 3914 (mean prob 9.2%) — ref 4.2%
- protect: pick 20.6% of 13633 (mean prob 17.9%) — ref 27.3%

*Verdicts are decision-support against the pre-registered runbook rules; the runbooks remain the registration of record.*
