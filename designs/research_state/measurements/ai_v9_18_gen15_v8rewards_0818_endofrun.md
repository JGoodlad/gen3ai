# End-of-run battery — `ai_v9_18_gen15_v8rewards_0818`

## 1. Ladder (tail-4, matched-count convention)
- current: **2094.4** ± 29.0
- reference: 2042.8 ± 27.5
- **NON_INFERIOR** (Δ 51.6, CI [11.6, 91.6]; rule: delta >= -15.0 and CI-low > -40.0)

## 2. Critic routes + edge families
- needs_pinned_tree: ModelVersionError: config_version 95 is a PRE-GENERATION checkpoint: it predates v96, the first MODEL_CONFIG_VERSION stamped with the current ARCH_SIGNATURE ('gen3_critic_route_wave_v1'). A checkpoint from an earlier generation cannot be loaded by this code — its weights were trained against an architecture this codebase no longer contains, and no config migration can bridge that.
To re-probe it, use the git_hash recorded in the checkpoint's own metadata.json (git checkout <git_hash> and probe from there — the prober prints exactly this diagnosis, with the hash, for archived runs).
    - `git worktree add /tmp/endofrun-pinned ff1daaef52c3e01fa12f3a46d6b4f50b03551e28`
    - `copy src/main/endofrun.py + src/agents/model/critic_route_audit.py into the pinned tree if it predates them`
    - `PYTHONPATH=src python -m main.endofrun models/ai_v9_18_gen15_v8rewards_0818 --skip elo,awareness`

## 3. Awareness + coverage (vs gen-10 baselines)
- blind_loss_fraction: **IMPROVED** (0.047 vs 0.072)
- median_lead_time: **UNCHANGED** (7.0 vs 7.0)
- cap_aware_ge_bar_fraction: **IMPROVED** (0.571 vs 0.5)
- coverage80: **WORSE** (0.4245 vs 0.44)
- pit_mean: **WORSE** (0.3788 vs 0.396)

## 4. Mechanic usage (G2 — conditional execution)
- focuspunch: pick 6.2% of 2124 (mean prob 7.9%) — ref 3.4%
- substitute: pick 3.4% of 4157 (mean prob 6.6%) — ref 3.8%
- endure: pick 0.0% of 41 (mean prob 4.9%) — ref 0.0%
- destinybond: pick 9.7% of 124 (mean prob 13.7%) — ref 11.2%
- endeavor: pick 11.8% of 76 (mean prob 14.3%) — ref 8.3%
- counter: pick 42.3% of 823 (mean prob 33.8%) — ref 11.5%
- explosion: pick 18.5% of 6075 (mean prob 17.6%) — ref 6.3%
- selfdestruct: pick 13.1% of 1249 (mean prob 13.1%) — ref 5.0%
- pursuit: pick 3.7% of 2968 (mean prob 7.4%) — ref 7.0%
- protect: pick 37.9% of 11106 (mean prob 29.1%) — ref 20.6%

*Verdicts are decision-support against the pre-registered runbook rules; the runbooks remain the registration of record.*
