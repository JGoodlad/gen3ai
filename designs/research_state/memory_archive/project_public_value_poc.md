---
name: project_public_value_poc
description: "Public-info value function POC on human replays: aggregate value real+cheap (0.73 AUC), but identity/structure (rich feats/MLP/transformer) did NOT beat it + overfit — data-limited; run the cheap transfer test not a fancier model"
metadata:
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — v43 pubval; owner verdict "did nothing" (recorded in project_damage_op_block_audit K9/K10) and the era is closed. Preserved verbatim; nothing below is current.

**2026-07-05. Public-information value function — SCRATCH EXPLORED (CPU), NOT built.** Full record:
`designs/ai_v8/design_public_info_value.md`. Scripts in `tmp/` (uncommitted): pubval_experiment.py (crude),
pubval_richobs.py (rich), pubval_transformer.py (per-mon+move transformer).

**Motivation:** the value function is the hub/limiter (defensive AUC≈0.50 → advantage≈0 on positional
decisions → generic play). A public-info value `V_pub(public board)→P(win)`, trained offline on human replays,
is the value-INDEPENDENT exogenous signal to break the bootstrap; intended use = shared-trunk AUX TARGET (RL
head still learns V^π, only trunk features shared; NEVER wire V_pub into GAE — it's V^human not V^π). Corpus:
`/home/goodlad/dev/gen3ai/replays/showdown/gen3ou/<date>/*.log` = **164,230** gen3ou rated ladder logs
(Showdown protocol; ratings in |player| lines; |win| = outcome). `data/replays` is EMPTY — corpus under
`replays/`. Parse public board per |turn|, both perspectives (mirror), label by terminal outcome, split
held-out BY GAME.

**Results (test AUC, split by game):** material-only logistic ~0.70; **crude aggregates (17 feats: mon-count,
HP, hazards, status, boosts) logistic = 0.734 = BEST** (MLP also 0.734); rich identity (122 feats:
species-base-stats/types/type-matchups) logistic 0.733 (**NO gain**), MLP 0.689 test / **0.914 train**
(overfit); **transformer over per-mon+move tokens (learned species+move embeddings): peak 0.699 @ep3 →
declines to 0.638** (overfits). Leakage guard CLEAN (turn-1 AUC≈0.51 → ~0.84 late), calibrated
([0.8,1.0)→0.89), beats material +0.04.

**VERDICT (honest):** (1) aggregate public value is REAL, cheap, validated — cheapest model wins. (2)
identity/STRUCTURE did NOT beat aggregates at scratch scale + overfits within-game correlation. (3) KEY CAVEAT
= DATA-LIMITED (7–20k games; ~758 embeddings can't stabilize; full corpus 164k=20× would shrink overfit —
scratch can't rule out that a regularized full-corpus transformer helps). Pokémon's irreducible variance
(dice+hidden teams) likely caps PUBLIC prediction ~0.74–0.80 regardless.

**RECOMMENDATION:** don't chase raw AUC>0.74 with a fancier model. Build the CHEAP aggregate V_pub as a
shared-trunk AUX TARGET and run the ONE test that matters: does the critic's DEFENSIVE-AUC-by-style climb
(positional-discrimination TRANSFER, reuse [[project_archetype_competence_gradient]] calibration-by-style)?
That's nearly free and doesn't hinge on beating 0.74. Full-corpus transformer = bigger bet, follow-on ONLY if
the cheap version shows transfer. Standalone, aggregate V_pub is immediately useful as a universal position
evaluator for the prober/curriculum. Relates to the value-function-as-hub thesis in
`designs/learning/pbs_value_functions_and_search.md`; the exogenous-data lever from
[[project_plateau_research_2026_06_25]] (Metamon).

**2026-07-08 — BUILT (v43 `gen3_pubval_aux_v1`, worktree, NOT shipped — awaiting user review).** The
recommendation above implemented end-to-end, user-requested ("aux loss head from public replays flowing back
into the shared trunk... a toy example of what is learnable"). PIECES: (1) `agents/training/pubval.py` — ONE
shared 17-feature definition (`PubSide`/`features()`) used by BOTH the corpus parser (`parse_replay_log`,
improved POC parser: faint-clears-status, -cureteam, -setboost/-clearallboost/-copyboost) and the live fold
(`pub_side_from_live` over LiveView) → parity structural; `PubValModel` frozen-artifact loader (fail-loud).
(2) `pubval_calibration.py` CLI → **`data/gen3_pubval.json`: FULL corpus 170,769 games / 8.58M positions,
held-out-by-game AUC 0.7343, turn-1 0.500 leakage-clean, calibrated** (refuses to write if turn-1 drifts).
(3) `PubValHead` (WinProbHead subclass) off value_pooled; tri-state `--pubval-mode {none,read_only,shaping}`
(v43 check_compatible string gate, resume-immutable; OFF byte-identical) + training-only `--pubval-coef` (0.1).
(4) Gen3Env emits `pubval_target`/`pubval_mask` per decision (REAL present-state value, no callback);
`_pubval_loss` masked soft-BCE; `pubval/*` metrics (watch **mae**→0, bce floors at target entropy) +
`grad/pubval_share`. VERIFIED: 3144-test suite green; **parity fuzz 3306 checks/30 bridge battles ALL EXACT +
e2e target==artifact 0 mismatches** (caught 2 real bugs: PubValHead class-swallow of ValueDistHead; capture
hook must install BEFORE attach_bridge_transport — bound-handler capture → added anti-vacuous-run guard);
--debug shaping smoke: roundtrip PASSED, pubval/coverage 1, target_mean 0.202 (human V_pub correctly prices a
losing random-policy board), grad/pubval_share 0.017. NEXT (after review + /gen3ai-ship): a run with
--pubval-mode shaping; acceptance = defensive-AUC-by-style transfer + WR/ELO vs a pubval-off control.

**2026-07-08 — SHIPPED 5f62009 + LAUNCHED ai_v7_09_tss_bots_pubval_0708** (tmux win 0,
`/tmp/launch_ai_v7_09.sh`, worktree pinned 5f620095): ai_v7_08's EXACT bots-only TSS config +
`--pubval-mode shaping --pubval-coef 0.1` — the direct A/B vs the collapsed ai_v7_08 control (stopped
@56.8M, bot-WR flat ~7% its whole life; ckpt banked). THE QUESTION: does the dense human credit-assignment
signal un-stick single-team-from-scratch learning? Watch: bot-WR curve vs ai_v7_08's flat 5–9% band (any
sustained climb = the signal matters), `pubval/mae` falling (the head fits), `grad/pubval_share` small,
entropy NOT collapsing to −0.9 (ai_v7_08's collapse signature), vs random >75%. If ai_v7_09 ALSO collapses
identically → pubval-shaping alone doesn't rescue the no-bootstrap-floor pathology (still informative:
the ceiling is exploration/credit deeper than a dense value prior) → next arm = diverse-teams + pubval.
