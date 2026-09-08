---
name: project-distributional-critic-verdict
description: "Distributional value critic = quantified NO for the strong-opponent ceiling (residuals sub-Gaussian, no tail to re-weight). Damage-magnitude feature also dropped. Verified via adversarial workflow 2026-06-12."
metadata: 
  node_type: memory
  type: project
  originSessionId: 6d3cc0b1-c1f5-48ff-b361-e28b6d0f519f
---

> **Archived 2026-09-08** — verdict NO; the value_dist head is DELETED under the ai_v12 win-prob critic. Preserved verbatim; nothing below is current.

**Distributional value critic — DROP (quantified, adversarially verified 2026-06-12 on ai_v5_11).** The hoped-for fat critic tail is an OUTCOME-CONDITIONING artifact: conditioning residual e=R−V on meta.result=LOSS mechanically drags MC return below the mean-over-outcomes V (ext-LOSS bias −22.5 MIRRORS ext-WIN +24.8; unconditioned bias only −4.7). Once you stop conditioning on outcome, strong-opp (ext+pool) residuals are **sub-Gaussian** (tail-domination 0.33 ≈ 0.31 Gaussian baseline, excess kurtosis −0.89). No heavy tail → nothing for a distributional/tail-weighted loss to re-weight. The only heavy tail is the γ=0.9999 PP-stall self-play battles = a reward/horizon artifact (the full-HP heal loops, see [[project_floor_leak_critic_selfko]]). NOTE: trace EV (full-MC return-to-go, ~0.11) is NOT comparable to TB explained_variance (~0.74 bootstrapped GAE) — different quantities; values ARE PopArt-denormalized (units verified). One real-but-small surviving signal: V is **under-spread** (std V 10.9 vs R 28.2) — a representation issue a distributional head doesn't fix.

**Corrected damage feature — also DROP.** The damage probe r²≈0.08 IS partly a measurement artifact (62% structural zeros from opp switch/status/miss — the user's insight was right). BUT conditioning doesn't rescue it: condition on the model's OWN belief>0.15 and r² collapses 0.071→0.012 (~17% of a low ~0.07 aleatoric ceiling), rank-corr ~0 above the floor, top decile over-predicts 5×. Belief is SATURATED above its floor (predicts will-I-take-ANY-damage, not magnitude). The residual is which-move/opponent-action uncertainty, not a marginal damage scalar.

**Method that worked:** ultracode Workflow = 3 parallel measurements → 3 INDEPENDENT adversarial refutations → synthesis. All 3 measurements came back `weakened` (each had a reproduced selection-bias). The adversarial pass is what saved us from 3 false starts — do this for every "should we build X" call. rev rabbit-hole pivot: rebalanced run ai_v5_11_tail2 (vf_coef 0.5→0.25) value_share landed ~0.6 (target 0.45, not fully) + ELO 1828@22M > ai_v5_10's 1796 (rebalance helped); but the REAL lever turned out to be the floor leak, not any of these. See [[project_plateau_diagnosis_2026_06_09]].
