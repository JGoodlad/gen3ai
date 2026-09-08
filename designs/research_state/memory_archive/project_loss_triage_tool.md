---
name: project_loss_triage_tool
description: "prober `triage` loss-attribution tool (built 2026-06-09) ranks failure LEVERS across a run; finding = critic_blindspot is"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7d3c1d7-0aa2-4875-9183-6e9491acdea9
---

Built 2026-06-09: `python -m main.prober.query triage <run_dir> [--step N] [--opponent X]`
(also `ProbeSession.triage()`). Model-free loss-attribution: categorizes every loss's
single worst-ΔV turning point into `engine.LOSS_TAXONOMY` (the one place to extend), ranks
categories by `est_recoverable_winrate_pct` = mean over the 9 fixed **bot** opponents of
`loss_rate(opp) × category_share(opp)` (upper bound). Each category names a LEVER
(obs / reward / policy / critic-capacity / upstream / measurement). Reads true win-rates from
`eval_results.jsonl` (falls back to raw loss volume if absent). 82 prober tests pass (+14 new).

**Taxonomy split logic** (the lever-naming insight): deaths split on the belief signal —
belief **under-read** a healthy death = OBS (`surprise_ohko`); belief **fired** + a pivot
existed = REWARD/POLICY (`ignored_threat_death`, the under-switch target); no pivot = UPSTREAM
(`doomed_already`); already-fainted = forced replacement, look back a turn (`post_faint_replacement`).
No-death value craters split on the critic's **pre-cliff value SIGN** (scale-invariant):
V(s)>0 (thought it was WINNING then craters) = `critic_blindspot` (CRITIC CAPACITY / missing-obs);
V(s)≤0 (already knew) = `positional_grind` (upstream/material). `attrition_death` = worn-down
mid-belief deaths.

**Finding (ai_v5_6_stable_N_0608 @46M AND ai_v5_5_popart_50m_0607 @52M — cross-run STABLE):**
- `critic_blindspot` ~28-29% = **#1 lever → value capacity** (more transformer layers / the
  value-dedicated readout). The critic confidently rates ~30% of losses as winning right before
  they crater. Strongest signal for the "how many transformer layers" question.
- `positional_grind` ~24% (upstream/material) + `attrition_death` ~16% = the hard residual.
- `ignored_threat_death` ~11-12% = **reward/policy lever** (under-switching) — confirms the
  standing thesis that under-switching is a POLICY problem, NOT obs. Pairs with
  [[project_incoming_damage_outcome]] (obs is wired-in + critic-read; policy under-switches).
- `surprise_ohko` ~7-8% = **obs lever is SMALL** — obs is largely doing its job; not the main lever.
- `post_faint_replacement` ~7-9% (measurement), `greedy_setup` ~2-5%, `stall_timeout` ~1%.

⚠️ The naive "critic_blindspot #1 → add transformer layers" read was **CORRECTED** by an
adversarial deep-dive at 66M — see [[project_plateau_diagnosis_2026_06_09]]: the aggregate critic
is healthy (EV 0.77), much of critic_blindspot is irreducible (hidden last mon + RNG), and the real
defect is critic TAIL-calibration, not blind depth. Treat triage recov% as an UPPER BOUND; verify
each lever adversarially before acting (the high-V craters are a non-representative subsample of the
bucket). See [[project_markovian_reward_design]], [[project_popart]]. Self-improvement loop next:
port `triage` to the prober TUI (per [[feedback_prober_self_improvement]]). NOT committed (awaiting /gen3ai-ship).
