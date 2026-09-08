---
name: project_belief_shaping_experiment
description: "Q1/Q2 belief-imagination experiment — ai_v7_03 belief-grad-mode shaping (LAUNCHED), then move-belief pre-fusion (Q2, queued)"
metadata:
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — ai_v7/ai_v8-era belief-grad-mode arms; era closed, --belief-grad-mode semantics live in src/agents/training/CLAUDE.md. Preserved verbatim; nothing below is current.

**BELIEF-IMAGINATION EXPERIMENT (2026-06-30, LAUNCHED).** Goal: make the model genuinely IMAGINE the
opponent's hidden state (unseen mons/moves/spread) and reason over it — a TRUNK/attention property, not a
head property. Design discussion established: the privileged-label supervision + differentiable op train the
belief HEAD, but only the belief→trunk SHAPING gradient makes the REPRESENTATION encode hidden identity;
"reasoning over the imagined mon" needs that gradient. **⚠️ CORRECTED 2026-07-25 (ledger P3): the "~3-5 effective dims" figure below is the `value_cls` READOUT series, NOT the trunk.** `rank_metrics.py` emits FOUR series (trunk/value_cls/policy/vf_feat); measured on ai_v8_03: **trunk 24->35 of 128 and RISING**, policy ~48-58 of 512, value_cls ~3.3, vf_feat ~2.7. And rank-3 on the value side is APPROPRIATE, not pathological (the critic emits ONE scalar; outcome AUC 0.833 from value_pooled vs 0.835 from the policy's 384 dims). So "capacity isn't the constraint" still holds for the trunk, but NOT via this number. The rank probe (the value_cls readout runs in ~3-5 effective
dims) says capacity ISN'T the constraint → the original rationale for `--belief-grad-mode detached`
(protect policy capacity) is WEAK → worth turning shaping back on. See [[project_damage_op_block_audit]]
(pipeline flow) + [[project_plateau_research_2026_06_25]] (detached run had belief at chance, win=critic-shaping).

**Q1 = `ai_v7_03_belief_shape_0630` (LAUNCHED tmux window 0, fresh from-scratch).** IDENTICAL to
ai_v7_02_critic_shape_0627's original_command EXCEPT `--belief-grad-mode shaping` (was `detached`) + new
run-name, no --model. belief_grad_mode is RESUME-IMMUTABLE → must be fresh. Confirmed at launch:
belief_grad_mode=shaping, config_version 42, arch_sig gen3_opp_hp_typed_candidates_v1, damage_op=True,
refine_rounds=2. Launch script `/tmp/launch_ai_v7_03.sh`. **Success criteria (user):** at least as good as
ai_v7_02 (matched-step WR/ELO non-regression) AND the belief metrics climb OFF chance
(`belief/species_acc_above_chance`, `belief/latent_cos>chance`, `move_recall` — detached had them at
~chance 0.004-0.013). RISK: shaping reintroduces belief↔policy gradient interference → watch for WR
regression vs ai_v7_02's matched-step curve. Checks scheduled 21:56 (+30m) + 03:26 Jul-1 (+6h) — SESSION-ONLY
crons (durable didn't persist), but the TRAINING is a detached tmux launcher (6h auto-restart) so it survives
my session regardless; only the auto-check-ins are session-bound.

**Q1 6h/~12M-STEP RESULT (2026-07-01, PASSING — clean positive):** matched-step vs ai_v7_02 (detached),
both FRESH runs so directly comparable (GOTCHA: ai_v7_02's LATEST tb file only covers ~104M; must read
its EARLY tb files / eval_results.jsonl for the ~12M row, else you falsely compare shp@10M vs det@104M).
Matched ~12M: **ELO shp 1822@10M vs det 1806@12M** (Q1 AHEAD despite fewer steps); **bot-WR 0.805 vs
0.7825** (+2.25pp); pool-WR 0.565=0.565 (promotion-gate-pinned). BELIEF = decisive: **species_acc>chance
0.35 (shp) vs 0.12 (det)** same 11.8M step (~3×); **latent_cos>chance 0.066 (shp) vs 0.0002 (det, at
chance)** — shp climbs MONOTONIC 0→0.066 over 12M (fixes the user's "latent cosine seems worse" concern);
move_recall 0.643 vs 0.625. INTERFERENCE = benign: aux_share moderated 0.70(30min)→0.347, value_policy_
logratio −0.27 (not blown), policy_share 0.42 (still dominant), expl_var 0.73. So capacity-not-constrained
held: belief shapes the trunk AND policy is ≥ detached. CAVEAT: 12M is early (ai_v7_02 peaked 92-107M) —
the real test is whether the belief-accuracy win WIDENS the ELO gap at the ceiling (the "better belief helps
more once move-selection is good" hypothesis) or just matches. Keep Q1 running to confirm the gap holds.

**Q1 ~28-30M-STEP UPDATE (2026-07-01, REVERSAL — the 12M lead evaporated):** the early lead I reported at
6h was the early phase; by 28M detached (ai_v7_02) has PULLED AHEAD on policy metrics. Matched ELO gap
(shp − det): +32@6M/10M → −40@14M → −1@22M → −38@26M → **−52@28M** (widening, ~2× the ±19-26 CI). bot-WR
REVERSED too: shp +2.3pp@12M → **−5pp@28M** (det 0.839 vs shp 0.788); pool-WR det 0.629 vs shp 0.598. Yet
BELIEF stays decisively better under shaping AND keeps climbing: species_acc>chance **0.431 (shp) vs 0.149
(det)** ~3×; latent_cos>chance **0.132 vs 0.0011**; move_recall 0.707 vs 0.657. So this is the classic
LEARNS≠HELPS: shaping makes the belief IMAGINE far better (mechanism 100% confirmed) but that accuracy does
NOT translate to policy — it slightly HURTS (aux_share higher 0.373 vs 0.211; policy modestly behind;
value_policy_logratio −0.242 vs −0.141). = EVIDENCE FOR the detach decision / the belief↔policy interference
is real & modest, AND against the near-term "accurate belief helps" thesis (a late payoff at a much stronger
policy isn't ruled out, but the trend at 28M is AGAINST, widening). DECISION POINT: Q1 no longer meets the
"≥ ai_v7_02" bar. Options: (a) watch to ~40-50M to confirm gap is real/widening not noise, (b) kill Q1 +
resume ai_v7_02 (0.92/1998, 107.4M), (c) try a MIDDLE ground (partial shaping coef, or belief→OP detach
[keep op-as-feature] instead of belief→trunk detach). CORRECTED my 6h "passing" read.

**WHY shaping underperforms — DIAGNOSED (2026-07-01, TB probe @28M).** Tested the user's 3 hypotheses:
- **H1 opposing gradients: REFUTED.** Every belief-head's `grad/*_policy_cosine` on the shared trunk is ~0
  (species +0.005, move −0.005, latent +0.002, all ±0.05-0.08) — belief gradients are ORTHOGONAL to policy,
  NOT anti-aligned. (detached reads exactly 0 — its belief grad to trunk is stop-grad'd, confirms the metric.)
- **H2 slow/transient convergence: REFUTED.** aux_share fell 0.88→0.35 by 20M then PLATEAUED ~0.37 (not
  decaying to 0), and the ELO gap WIDENS — persistent allocation, not a settling transient.
- **H3 behavioral over-caution: only PARTIALLY supported after a PROPER test** (CORRECTS an earlier over-claim
  — I first quoted setup_sweep WR −0.10 / ELO −52, but those were SINGLE eval-cycle snapshots at exactly 28M
  = cherry-picked noise). PAIRED test across 17 matched eval cycles (Wilcoxon signed-rank, shp−det):
  win_rate_vs_setup_sweep mean −0.015 MEDIAN 0.000 **p=0.57 ns**; all matchup WRs ns (p 0.28-0.92); **ELO mean
  −11.7 p=0.12 ns** (leans − but within noise). The ONLY statistically robust behavioral diff: **self-play
  ep_len +0.76 turns, p<0.0001** (n=346) = a small (~2-3%) but real PASSIVITY signature — NO proven WR cost.
  So at ~30M it is a WASH on performance (not the "behind by 52 ELO" I wrongly reported), with a confirmed
  tiny passivity shift + decisively better belief. The user's skepticism ("hard to believe accuracy lowers
  ability") is VINDICATED — the data does NOT show a significant ability decrease. POWER CAVEAT: 17 cycles
  detect a LARGE regression but not a ~10-ELO one → rules out "clearly worse", can't prove "exactly equal".
  Direct switch-rate battle probe (wf_04080a1e) DIED on a session usage limit (resets 7:10pm PT) — decision-
  level caution measurement still PENDING. Net: gradients orthogonal (H1 dead) + belief far more accurate +
  mild passivity + no significant WR change ⇒ the earlier "reversal" was largely eval noise.

**Q2 (QUEUED, after Q1 holds) = move-belief PRE-FUSION** (`--move-belief-prefuse`): inject the move belief
into the opp ROLE tokens BEFORE the transformer so believed moves co-refine through attention (today only
species/BeliefSlots is pre-transformer; moves/spread/HP are grafted POST). There is NO prefuse for spread/HP
yet — a possible Q3+. Downside of pre-fusing more: amplifies belief→trunk gradient interference (only bites
under shaping — which is why Q1 must come first) + couples board reasoning to belief quality (mitigated by
the existing small/zero-init reinject).

**ai_v7_02 BASELINE — STOPPED (SIGTERM, clean) at step 107,417,700** (`final_model_interrupted.zip` +
checkpoint_106685764). Fully RESUMABLE if Q1 underperforms (one GPU → one run at a time; had to stop it to
free window 0 per user). It hit 0.92 WR / 1998 ELO — the all-time best, critic-shaping win.

**Q3 (2026-07-21, LIVE) — the IN-RUN flip on ai_v8_03 @240.2M.** Shipped
`--allow-belief-grad-mode-change` (1842d22, the intentional-migration escape hatch on the v41
gate) and flipped the CONVERGED ai_v8_03 lineage detached→shaping at a restart (film-accum
4→6 rode along; 200-game evals active since 238M → ±0.069 CIs). The 240M eval is the LAST
detached baseline. Rationale vs Q1: the "late payoff at a much stronger policy" branch left
open in the Q1 record + the ~19% inert belief→physics machinery + crystallization pressure
(PR(vf) 2.85 falling). GATES: belief/species_acc + latent_cos + move_recall OFF chance within
~5-10M (mechanism); the ~1990-2018 ELO band breaks UP vs sideways (payoff); rank/vf_feat_pr
rises (crystallization); ep_len watched for the +0.76-turn passivity signature. Outcome decides
next_run_plan items 3/5 (refine rounds + threat channels keep-or-strip).
