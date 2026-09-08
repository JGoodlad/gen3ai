---
name: project_value_dist_head_status
description: "Distributional value head (v29, read_only) is healthy/calibrated in ai_v6_11; BUT [-5,5] support tail-clips → widen vmax before shaping/Phase-B"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — the ValueDistHead is DELETED under the ai_v12 win-prob critic. Preserved verbatim; nothing below is current.

2026-06-19 (ai_v6_11_typed_hp_0619 @ ~9.8M steps, read-only TB/prober analysis, adversarially verified). The
v29 `ValueDistHead` (`--value-dist-mode read_only --value-dist-bins 51 --value-dist-vmin -5 --value-dist-vmax 5`)
is behaving exactly as Phase-A intends: HL-Gauss CE loss is LIVE (not staged); `ce 4.69→2.22`, `mae 0.77→0.38`,
`std 2.69→0.51`, converged ~1-1.4M then flat. Calibrated (`pit_mean 0.498`; STRENGTHENED — `vd_std 0.523 ≈
√(1−EV)·ret_z_std 0.513`, 2% match = honest residual-variance read, not a spike hiding behind pit=0.5; entropy
2.21 nats > kernel floor ~1.13). Risk-free: `grad/value_dist_{share,cosine,norm}` EXACTLY 0.0 at all 99 points
(`value_pooled.detach()`), other heads nonzero so the metric is alive. "Learning is trivial (sigma-shrink)"
REFUTED: normalized target std ~1 throughout, sigma GROWS 5→15 while ce falls.

**THE TAIL-CLIP CAVEAT — flagged @9.8M, RESOLVED itself by 26.4M (re-reviewed 2026-06-20).** Support `[-5,5]` is in
PopArt-NORMALIZED units (Δ=0.2). @9.8M I worried the edge atoms were clipping (the body half-width was ~5σ, +5 exceeded
86% of batches). By 26.4M it's a NON-ISSUE: **PopArt sigma grew 5.1→14.5**, which renormalized returns so the ±2σ body is
now only ~2.0 units wide — comfortably inside [-5,5], ~0 body mass at the edges. The +5/−5 crossings that still register
(+5 in 16/20, −5 in 14/20 of recent steps) are **single-episode `return_abs_max` tail OUTLIERS** (raw 50–118) that the
edge atom is DESIGNED to absorb — not body mass. Calibration intact: `pit_mean` pinned 0.500, `mae` flat ~0.36 (not rising),
`ce` 2.20. And the sharper point: read_only ⇒ `grad/value_dist_share`=0 always, so it could NEVER have corrupted pi/vf
regardless. STILL TRUE: if you graduate to `--value-dist-mode shaping` or Phase-B (dist critic replaces scalar `value_net`),
widen `vmax` on the fresh run so the shaping signal isn't biased by an extreme tail — but for the current read_only
diagnostic this is fully benign.

**NEXT-LEVER read (2026-06-20, ai_v6_11 @34M, loss-crater analysis).** ai_v6_11 plateaued ~1880 ELO = GENUINE
convergence (pool growing/promoting on cadence — not a treadmill). Crater bracket: ~43% LUCK (risk-neutral PPO eating
dice tails) / ~50% NEUTRAL / only ~7% PROVEN policy mistakes; top failure lever = positional_grind ~42% (slow losses
the critic already knew, V≤0 — skill gap, not an obs/critic miss); surprise_ohko ~19% (obs coverage). I initially named "turn THIS read_only value_dist head into a SHAPING objective (tail-weighted / distributional value
loss)" as the top plateau lever. **CORRECTION (same-day deeper grind diagnosis, [[project_positional_grind_decomposition]]):
the tail-weighted CRITIC angle is FALSIFIED for strong-opp grind** — the loss residuals are SUB-GAUSSIAN (tail-dom 0.33,
exkurt −0.89, research_state kill K1): there is NO fat tail to re-weight; the apparent "fat tail" was outcome-conditioning
+ a Pressure-stall reward artifact. So distributional/tail-critic is NOT the grind lever, and ai_v6_11 already runs
`--value-tail-weight 0.3` anyway. The risk-sensitive POLICY angle (CVaR objective) is technically un-killed, but don't lead
with the critic side. The REAL grind levers are PFSP league + EXPLOITER and offline teacher/BC — see
[[project_positional_grind_decomposition]]. value_dist stays valuable as the calibrated read_only DIAGNOSTIC; if ever
promoted to shaping, widen vmax first (above). Pairs with [[project_plateau_diagnosis_2026_06_09]] + [[project_model_frontier_roadmap]].

**ACTIONABLE:** `--value-dist-vmin/vmax` are RESUME-IMMUTABLE (version-checked, `check_value_dist`). Doesn't matter for
read_only. BUT before graduating to `--value-dist-mode shaping` (objective shapes the trunk) or Phase-B (dist critic
replaces scalar `value_net`), WIDEN vmax on the FRESH run (e.g. 7–8), else the shaping signal inherits the inward-biased
tail. Prober has a first-class view: `python -m main.prober.query analyze <full_*_summary.json> <idx>` → `value_dist`
block (probs/support/mean_real/std/entropy/bimodality); TUI Summary panel renders the histogram (`⑂ bimodal` >0.35).
Code: `ValueDistHead` features_extractor.py:1129; loss instrumented_ppo.py:312-361,1064-1079; capture battle_recorder.py:162.
Pairs w/ [[project_model_frontier_roadmap]] (item 3: distributional critic).
