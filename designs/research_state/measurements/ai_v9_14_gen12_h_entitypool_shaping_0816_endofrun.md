# End-of-run battery — `ai_v9_14_gen12_h_entitypool_shaping_0816`

## 1. Ladder (tail-4, matched-count convention)
- current: **2066.4** ± 28.2
- reference: 2091.2 ± 29.2
- **INCONCLUSIVE** (Δ -24.8, CI [-65.4, 15.8]; rule: delta >= -15.0 and CI-low > -40.0)

## 2. Critic routes + edge families
- needs_pinned_tree: ModelVersionError: Stable opponent total_dim mismatch: opponent=2921, current=3529 (arch_signature matched — the opponent's model_config.json looks hand-edited).
    - `git worktree add /tmp/endofrun-pinned ede5a887ea056b6f3d2f56719baeecdccc5b1634`
    - `copy src/main/endofrun.py + src/agents/model/critic_route_audit.py into the pinned tree if it predates them`
    - `PYTHONPATH=src python -m main.endofrun /home/goodlad/dev/gen3ai/models/ai_v9_14_gen12_h_entitypool_shaping_0816 --skip elo,awareness`

## 3. Awareness + coverage (vs gen-10 baselines)
- blind_loss_fraction: **IMPROVED** (0.054 vs 0.072)
- median_lead_time: **UNCHANGED** (7.0 vs 7.0)
- cap_aware_ge_bar_fraction: **IMPROVED** (0.8 vs 0.5)
- coverage80: **IMPROVED** (0.4652 vs 0.44)
- pit_mean: **UNCHANGED** (0.3924 vs 0.396)

## 4. Mechanic usage (G2 — conditional execution)
- focuspunch: pick 5.4% of 2394 (mean prob 6.5%) — ref 3.3%
- substitute: pick 1.9% of 5179 (mean prob 4.7%) — ref 2.7%
- endure: pick 0.0% of 95 (mean prob 0.5%) — ref 0.0%
- destinybond: pick 15.8% of 133 (mean prob 13.2%) — ref 5.8%
- endeavor: pick 4.4% of 91 (mean prob 7.8%) — ref 12.7%
- counter: pick 6.0% of 1108 (mean prob 9.3%) — ref 13.8%
- explosion: pick 5.5% of 11896 (mean prob 7.0%) — ref 7.1%
- selfdestruct: pick 7.0% of 1378 (mean prob 8.3%) — ref 7.2%
- pursuit: pick 3.9% of 4466 (mean prob 7.3%) — ref 6.1%
- protect: pick 24.5% of 13825 (mean prob 21.0%) — ref 26.8%

*Verdicts are decision-support against the pre-registered runbook rules; the runbooks remain the registration of record.*
