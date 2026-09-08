---
name: project_l3_oracle_grind_l4
description: "L3 opponent-action VoI oracle KILLED the forward-model head for the strong-opp grind; the L1/L2/L3 tree is closed → grind is L4 (skill ceiling), only league/teacher/accept remain"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6d3cc0b1-c1f5-48ff-b361-e28b6d0f519f
---

> **Archived 2026-09-08** — the durable rule (the amortizability gate, never search ON the model) lives in feedback_research_state. Preserved verbatim; nothing below is current.

2026-06-12. After the grind workflow ([[project_model_frontier_roadmap]], the H4/H5 frontier fill), the
owner picked the **L3 opponent-action ORACLE** as the cheap falsify-before-build test for a forward-model
/ opponent-action aux head. **Result: KILLED (verified).**

**Method (reusable — falsify ANY "give the model info X" feature before building it):** the feature's
benefit is bounded above by the **VALUE OF INFORMATION** of what it surfaces. For the opp-action head, VoI
= `E_o[max_a U(a,o)] − max_a E_o[U(a,o)]` over the re-rolled OUR-action × OPP-action material grid
(`reroll_turn` fixes both sides' actions; `material_margin` scores). Instrument:
`models/saved_work/opp_action_voi_oracle.py` (spawn-safe ProcessPool, in-process bridge, no server).

**Numbers (run ai_v5_11_tail2_53m_0611, strong opps = sentinel_0..4 + ext_*, competitive-phase decisions,
n=66 loss / 58 win):** VoI median **loss 0.031 ≈ win 0.032 mon** (Mann-Whitney **p=0.53**, bootstrap CI on
the diff straddles 0); restricted to the opp ATTACKING (realistic) ~**0.009 mon** (52% of loss decisions
≤0.01). A perfect opp-action predictor would change material by ~3% of one mon, and that sliver is the SAME
in wins — so the head cannot move the grind. **Verified** by an independent adversarial agent
(`voi_verify_*.py`): re-derived the grid by hand (44 distinct robust-actions, VoI≥0 100% — NOT an
A4-style hardcoded-constant bug); horizon stress-test held (deterministic + terminal-clamp re-scoring
moved the median 0.0270→0.0274, ~40× below the kill-premature threshold); the loss≈win null is the guard
against the one residual caveat (a one-turn re-roll can't fully exclude arbitrary-depth anticipation, but
if it mattered *in losses* the proxy would skew toward losses — it shows zero skew).

**Strategic upshot — the L1–L4 decision tree is CLOSED for the strong-opp grind:** L1 obs ❌ (no
accumulation channel discriminates, all AUC ~0.5), L2 critic ❌ (mean calibrated once the eval-quota WR
confound is defeated — the A4 all-0.48-sentinel bug; true WRs 0.55/0.55/0.62/0.64/0.76 collapse the
residual), **L3 anticipation ❌ (this)**. The dominant mass is diffuse global out-play = **L4 / genuine
multi-turn play strength (a skill ceiling)**, ~50–64% lost-from-turn-1. On-model search is owner-ruled-out
([[project_model_frontier_roadmap]] — search is teacher/offline only). So the strong-opp axis has only
THREE moves left, all bigger structural bets: **PFSP/exploiter league** (training-dist pressure, upside
gated by a cheap league-vs-uniform A/B on the structurally-lost fraction), an **offline search-teacher**
distilled into the net, or **accept the plateau** and ship the cheap **bot-side EXPLOIT portfolio** (H1
self-KO SHIPPED + H2 type-mismatch + H3 surprise-OHKO which is now strong-specific-DEAD = bot-only).

This **confirms the explore→exploit posture shift** (the amortizability gate + build-bar the owner added
to `designs/research_state/`): there is no cheap feature lever for strong opponents — the blunder AND the
feature veins are both mined out. Full record: `designs/research_state/levers/strong_opp_grind.md` (H4),
ledger K9. See [[feedback_research_state]] for the SOT-maintenance protocol.

**CORRECTION (2026-07-03, owner-directed — STOP citing "~2/3 / 50-64% auto-loss / uncoachable team-draw").**
The "~50-64% lost-from-turn-1" above is NOT a measured objective-unwinnability figure and must not be used as a
specialization/coaching CEILING. Two reasons: (1) it was read largely off turn-1 V / win-prob, which is
POLICY-CONDITIONAL — V is low because THIS policy expects to lose, mathematically indistinguishable from "the
policy PILOTS that team/archetype worse" (see [[project_archetype_competence_gradient]] which flags this exact
confound). (2) The only DEFENSIBLE loss-attribution numbers are ~6% proven-mistake + ~30% luck; the ~64%
residual is UNATTRIBUTED = UNKNOWN, NOT proven-unwinnable ([[project_plateau_research_2026_06_25]] called the
ceiling "UNDER-IDENTIFIED"). Owner's domain point (correct): gen3 OU is a MATURE, BALANCED meta — well-built
teams beat each other ~evenly with SKILL deciding; there is NO large class of subpar teams, let alone 2/3. A
thin tail of genuinely-skewed matchups exists; that's it. IMPLICATION: most losses are COACHABLE piloting, so
per-archetype SPECIALIZATION (routed policy + isolated-specialist distillation + DRO) has FAR MORE headroom
than the retracted "uncoachable" story implied. The VoI kill above still stands (knowing the opp ACTION ≈0.03
mon) — that is separate from and does NOT imply team-matchup auto-loss.

**SCOPE — what this does NOT kill (don't over-claim):** the VoI oracle measured MATERIAL value vs the
SELF-PLAY POOL + ext opponents (sentinels/ext). It does NOT say the opp-action / forward-model head is
useless everywhere — it survives as (a) the **world-model SCAFFOLD** for ai_v6 (roadmap #2), (b) a
**bot-side** signal (H3 surprise-OHKO fires on bots), and (c) potentially a **vs-HUMAN** lever:
[[project_human_agreement_probe]] found the policy UNDER-SWITCHES vs strong humans (16% vs 30%) — a
DIFFERENT opponent distribution than the self-play grind tested here, and a behavioural/style gap, not a
material-VoI one. The kill is precise: *knowing the opp's action does not improve material outcome in the
self-play-pool grind.* Human-ladder transfer + the world-model scaffold value are untested by this probe.
