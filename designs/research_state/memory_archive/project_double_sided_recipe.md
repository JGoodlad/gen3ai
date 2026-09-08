---
name: project_double_sided_recipe
description: "distill + stable-opp on the SAME exploiter = offense HELD + defense RECOVERED, no interference (ai_v7_19)"
metadata: 
  node_type: memory
  type: project
  originSessionId: f1a71c92-5431-47c7-a6e2-b2b0c1e359c4
---

> **Archived 2026-09-08** — ai_v7-era distill+stable-opponent recipe; the flywheel account is restated in UNDERSTANDING.md §2. Preserved verbatim; nothing below is current.

The DOUBLE-SIDED exploiter recipe WORKS: use each exploiter as BOTH a distill TEACHER (offense: learn
to pilot its team) AND a stable-OPPONENT (defense: learn to beat it) in ONE run. Validated in
**ai_v7_19** (forked _18@158.3M + the 4 exploiters as `--stable-opponents`, ~4.3M combined steps):
- **OFFENSE HELD**: Δoff vs distill-only _18 = **+0.025** (no team dropped) → the two objectives do NOT
  interfere. Offense shapes team-k-PILOTING states, defense shapes POOL-vs-k states — different input
  distributions, so complementary not conflicting.
- **DEFENSE RECOVERED — LOCKED** (2:47 AM night check, run's own greedy eval, 7 cycles 160M→172M / ~14M
  combined steps): rose on **ALL 4** exploiters, TSS 0.22→0.40 (+0.18), trap 0.36→0.46 (+0.10), cmpass
  0.34→0.50 (+0.16), stall 0.23→0.40 (+0.17); mean ~+0.15, monotone through noise, STILL CLIMBING at
  172M (not plateaued). Offense held the whole way (agree 0.73–0.79, bots ~0.90). (The earlier offline
  N=40 Δdef +0.106 @4.3M was a noisy floor; the 7-point run-eval trend confirms it.) Fully-confirmed WIN.

**Root cause it fixes** (defense-decompose eval, _02→_14→_18): training-AGAINST built big defense
(Δtrain +0.35 on the 3 trained-against exploiters; stall control ~0 → SPECIFIC, not general
improvement), but distillation-only (_18) had NO exploiters as opponents → it ERODED that defense
(Δdist NEGATIVE, TSS −0.15 = H2 forgetting). Fix = **keep the teacher in the opponent pool** (AlphaStar
league principle). This is the per-cluster LEAGUE building block: one exploiter teaches AND hardens.

Recipe: `--distill-teacher <exploiters> + --stable-opponents <same> --stable-opponent-selfplay-share 0.4
--stable-opponent-pfsp`. Self-instruments: `eval/win_rate_vs_ext_<exp>` = defense, `distill/agree_rate` =
offense. See [[project_exploiter_no_team_advantage]], [[project_tss_specialist_poc]].
