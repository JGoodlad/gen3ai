---
name: project_value_distill_fitnet
description: "scalar value-distill CRYSTALLIZES the critic (rank↓); FitNets hint-distill (--distill-value-feat-coef) is the fix, A/B live as ai_v7_21"
metadata:
  node_type: memory
  type: project
  originSessionId: f1a71c92-5431-47c7-a6e2-b2b0c1e359c4
---

> **Archived 2026-09-08** — ai_v7-era value-distill A/B; era closed, and no gen fold ever ran v8_14's literal value-feat coef. Preserved verbatim; nothing below is current.

Exploiter VALUE distillation has two forms, both requiring `--distill-coef>0` (the policy KL makes
the teacher's value the right target — V^π is policy-relative):

1. **SCALAR** `--distill-value-coef` (`gen3_exploiter_value_distill_v1`) — `MSE(V_teacher, V_student)`
   in the PopArt frame. **CRYSTALLIZES the critic** (confirmed A/B, ai_v7_20 forked _14, 148→153M):
   `distill/value_mse` falls (0.125→0.041) BUT `rank/value_cls_pr` 4.15→3.62, `effrank` 9.24→7.96,
   `n90` 12→10 — a scalar target has ~1 dim of info so the value-CLS pool collapses onto it. The
   shared trunk rank stays flat (~49 effrank) → it's the READOUT that crystallizes, not the trunk.

2. **FITNETS** `--distill-value-feat-coef` (`gen3_exploiter_value_feat_distill_v1`, SHIPPED a6ae04f) —
   the "hint" fix: `1 − cos(value_pooled_student, value_pooled_teacher)` on the teacher-team states,
   matching the INTERMEDIATE 128-dim value-CLS pool (`features_extractor.last_value_pooled`, stashed
   every forward) instead of the collapsed scalar → the trunk inherits the teacher's per-team value
   STRUCTURE. **COSINE not MSE** — chosen from the geometry analysis (`tmp/fitnet_analysis.py`): the 4
   teachers' value subspaces are low-rank (PR ~3–5 even for specialists), COMPLEMENTARY (TSS orthogonal
   0.04–0.07, collective effRank ~12), NON-competing (all pull-cosines positive); student/teacher are
   common-ancestor forks (all from _14) so bases are ~shared → regressor-free cosine is meaningful. The
   smoke's per-teacher `distill/tK_value_feat_cos` MATCHED the analysis (TSS=0.915 dist ≈ orthogonal).

**Why:** the double-sided recipe [[project_double_sided_recipe]] transfers the teacher's MOVES (policy
KL) but the student pilots the teacher team with its OWN amortized ~4-dim critic — it never gets the
per-team VALUE understanding. Scalar distill was the naive fix; it crystallizes. FitNets is the hint fix.

**How to apply:** the A/B is `_20` (scalar, control, STOPPED @152.67M) vs **`ai_v7_21_fitnet_valuefeat_ab_0717`**
(FitNets, LIVE, launched ~09:52 PDT 2026-07-17, `--distill-coef 0.1 --distill-value-feat-coef 0.5`, no
scalar). READOUT = does `rank/value_cls_pr` PRESERVE/RAISE (stay ≥~4) vs _20's drop to 3.6, while
`distill/value_feat_cos` falls. Read via `tmp/value_rank_compare.py` / TB `rank/value_cls_*`. Both terms
COMPOSE (can run scalar+feature together). Distributional-value distill (teacher's ValueDistHead) is the
next follow-on (enables archetype-token/FiLM conditioning — the v8 capstone).
