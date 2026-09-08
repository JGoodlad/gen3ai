---
name: project_plateau_research_2026_06_25
description: "Why the ~1990-ELO Gen3 self-play plateau: 9-agent research verdict (converged FLAT fixed point of THIS recipe; v1 exploiter null was basin-locked → worthless; ceiling under-identified; Metamon existence-proof) + the v2 basin-escape exploiter LAUNCHED"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — superseded by designs/research_state/UNDERSTANDING.md §1-§2 (the era map and the flywheel account). Preserved verbatim; nothing below is current.

2026-06-25. A 9-agent workflow (`wf_b8097d8a-b9f`; 7 lenses + synthesis + adversarial critique, ~1M tokens)
investigated WHY the agent plateaus (~1990 ELO / ~90% bot-WR, flat for tens of M steps). The adversarial pass
MATERIALLY corrected the synthesis — record both.

**PROVEN mechanism:** converged to a FLAT fixed point of THIS self-play RECIPE (adversary downgraded the
synthesis's "property of the GAME" → "property of the RECIPE": the 3 compared runs share self-play + reward/PBRS
+ pool mechanism + hparam family + team pool, not just architecture; data can't separate game- from
recipe-intrinsic, and the recommended fixes ARE recipe changes). Every PPO throttle measured OPEN — KL ~0.012
mid-band (0% violations), LR floating idle off both rails, entropy healthy ~1.07 nats / ~2.9 eff actions,
effective batch right at the McCandlish critical (noise_ratio ~0.92) — yet ELO/bot-WR/pool-WR flat. IDENTICAL
fingerprint across 3 archs (ai_v6_13 / v6_11 / v5_6: KL 0.010-0.012, EV 0.76-0.78, poolWR ~0.60). The optimizer
is FINE; the OBJECTIVE went flat → NO knob (KL / LR / entropy / batch / critic) breaks it. Firmest finding,
bit-for-bit reproduced from raw TB.

**CORRECTION (my earlier read was WRONG):** the v1 exploiter null (`win_rate_vs_ext` ~0.49-0.58, mean 0.52,
decayed to 0.49 by 28M steps) is WORTHLESS as ceiling evidence. Adversary confirmed it was BASIN-LOCKED:
byte-for-byte the parent recipe (ent 0.05, clip 0.10, defensive_entropy 1.0), init FROM the 1963 target, at the
parent's CONVERGED entropy (1.07 nats) from step 1 → it polished a COPY of the target for 28M steps, never
searched a structurally different attack. It proves ONLY "no CHEAP locally-reachable exploit." My 5am/midday
"confirms the matchup ceiling" reads were OVERCONFIDENT.

**Ceiling-vs-addressable = UNDER-IDENTIFIED (don't trust point estimates).** Killer confound the adversary
caught: eval traces save a HARD CAP of 8 wins/cycle but losses UNCAPPED → trace-derived WR 0.45 vs TRUE 0.90 →
EVERY trace-based loss statistic is ~10× loss-enriched. So the synthesis's "55-75% ceiling" / "15-35%
structurally-unwinnable, NOT ⅔" do NOT survive as numbers — report as wide, UPWARD-biased brackets (no Gen3 team
preview makes turn-1 outcome-AUC 0.659 a LOWER bound on true draw-determinism; the causal replay-counterfactual
stream is only N=14). Only TWO defendable numbers: **~6% of loss-crater mass is a PROVEN fixable mistake**
(falsify-scan, volume-weighted: bot losses 19.9% but self-play losses only 3.4%) and **~30% is irreducible
LUCK**. Critic is NOT the bottleneck (EV ~0.77 flat-healthy, PIT 0.500, win-prob skill_vs_material 0.30; the fat
td-tail IS real — partially corrects prior "no tail" memory — but it's aleatoric KO/crit/freeze reward jumps a
MEAN critic can't remove, so "mean / tail-weighted value loss is dead as a WR lever" STANDS). REUSABLE GOTCHA:
the **8-win trace quota** biases every prober loss analysis (triage recov%, falsify-scan brackets, calibration
bias E[V-G]=+8.5 is a selection artifact — calibrated critic, −17 on wins vs +26 on losses) — always note it.

**HOPE / existence proof (the strongest argument against "hard ceiling, stop"):** Metamon (arXiv 2504.04395) —
OUR EXACT domain (Showdown, early gens, random teams, no preview) — names our symptom verbatim ("self-play
overfit to recognizing its own playstyle, inconsistent vs real players") and broke the SAME plateau to
top-10%-human via OFFLINE RL on human replays + procedural TEAM diversity (~50% → ~80% GXE). So the aggregate
plateau is liftable by a REGIME change (league / teacher / diversity), NOT more self-play. AlphaStar league +
PFSP + exploiter RESETS is the cyclic-Nash fix; a single exploiter is the known-degenerate case.

**DECISIVE experiment — LAUNCHED 2026-06-25 as `ai_v6_13_outgoing_dmg_0620_exploiter_v2` in tmux 0:0** (cancelled
v1 first — one 12GB GPU can't fit two): a basin-ESCAPING exploiter. Init from the WEAKER parent CHECKPOINT
`checkpoint_4788069_steps.zip` (4.79M, ELO ~1680, ~280 below the 1963 target); `--ent-coef 0.05→0.13`,
`--clip-range 0.10→0.2`, dropped `--defensive-entropy-boost`; TARGET unchanged (frozen 1963 best_model, via
`--exploiter` + `--stable-opponents models/ai_v6_13_outgoing_dmg_0620`). VERDICT rule on
`eval/win_rate_vs_ext_ai_v6_13_outgoing_dmg_0620`: separates **>0.6** → a real exploitable hole → build the
multi-exploiter LEAGUE (payoff-matrix runner is the unbuilt linchpin); ALSO plateaus **~0.50** from a
genuinely-different attacker → ceiling much closer to proven → stop exploiters, run the search-teacher.
(Minimax opponent-Q reward shaping = riskier add-on — we expose state-value V, not a Q over opp actions —
deferred.) **LAUNCH GOTCHA discovered:** a POOL SNAPSHOT as `--model` FATALs — the pool `model_config.json` is
defaults-y (`opp_belief_aux_coef` 0 / `vf_coef` 0.5) → arch mismatch vs the belief-ON flags ("flips the belief
head → will FATAL on load"); a parent CHECKPOINT (in `checkpoints/`) resolves its config from the belief-ON run
root → loads clean. Use checkpoints, not pool snapshots, as exploiter init.

**STOP (falsified/spent — do NOT re-propose):** optimizer-knob tuning (every throttle open), critic mean /
tail-weighted value, the opp-action aux head ([[project_l3_oracle_grind_l4]] / [[project_opp_action_head_falsified]],
killed twice), the damage-magnitude scalar (r²→0.012), blind transformer depth / funnel / D_MODEL-shrink (trunk
runs ~3-5 eff dims), pool refresh, switch-bias / temperature / BC-for-under-switching (outgrown; corpus rated
below the student → regression-to-mean), turn-1 / team-draw features, running more v1-config exploiter steps, and
headlining triage recov% or critic_headroom_upper_bound (both upper bounds biased UP).

**Other real lever (gated):** the offline SEARCH-TEACHER (selective Expert Iteration; AWR-distills ONLY
Wilson-CI-confirmed-better search corrections → dodges the BC regression trap) — [[project_search_teacher]],
BUILT-not-RUN in a worktree, unshipped; targets the addressable thrown-late slice but MUST clear a `coef=0` A/B
(the project's recurring "learns ≠ helps" trap) before any WR claim.

**NEXT RUN — SHIPPED + LAUNCHED 2026-06-26 as `ai_v7_01_teacher_0626`** (fresh self-play, new arch generation,
tmux 0:0, single GPU, --use-showdown-bridge). Bundled config: (1) **detached belief** `--belief-grad-mode detached`
(soundest — the rank-probe-grounded change); (2) **history N=10→7** (`N_HISTORY_TURNS`; obs dim 3469→**2992**;
RETRAIN-class — SHIPPED commit **96c0ea0**, MODEL_CONFIG_VERSION 41→**42**, golden fixture regenerated; chose 7 not
5, the research-blessed cut); (3) **dropped the v31 reattend layer** (omitted `--damage-reattend`; the body is only
2 layers — do NOT cut it or it silently kills `--damage-refine-rounds 2`); (4) **distillation REMOVED** (shipped
**dc58306**, 1894 deletions). Owner BUNDLED (un-attributable, accepted; obs-dim change forces fresh anyway). Kept
`--damage-refine-rounds 2` + the self-play PFSP setup (`--pfsp-scale 2.5 --pool-spread --n-sentinels 10`); NO
`--stable-opponents`. CAVEAT: detached belief cuts the gradient the refine loop exists to provide
(`opp_tokens.detach()`), so refine-rounds-2 earns less (kept anyway for the forward physics enrichment).
**OOM ROOT CAUSE — RESOLVED 2026-06-27 (my earlier "detach inflates GPU memory" hypothesis was WRONG, disproven):**
NOTHING we added increased memory. The OOM was simply an **8192 micro-batch sitting at the 11.63GiB card cap** — a
margin the PARENT shared. PROOF (3 independent): (1) **flag diff** — vs the parent (`ai_v6_13_outgoing_dmg_0620`,
which ran the IDENTICAL `--batch-size 8192 --grad-accum-steps 2` micro-batch for tens of M steps on this same card
and NEVER logged a CUDA OOM), the new run only ADDED `--belief-grad-mode detached` (REDUCES backward) + CPU-side
self-play knobs (`--pfsp-scale/--pool-spread/--n-sentinels`, zero GPU) and REMOVED `--damage-reattend` (the 3rd
attention layer) + shrank history 10→7 — every arch delta makes it LIGHTER; both used `--grad-checkpointing`. (2)
**CPU saved-tensors byte-sum** @B=8192: new 3.41GB retained vs parent 4.15GB → new forward graph is SMALLER. (3)
**GPU `max_memory_allocated()` fwd+bwd peak** (`/tmp/oom_peak_probe.py`, tiny-batch B=32/64/128 extrapolated, both
on current 7-turn code, grad-ckpt ON): NEW (reattend=F, detached) **10.45GB** vs PARENT-arch (reattend=T, shaping)
**10.93GB** at B=8192 — new peaks ~0.5GB LOWER (and real parent ran 10-turn → even higher). Slope ≈1.3GB/1000 rows;
at 4096 rows ≈5.2GB extractor peak (matches the live ~6.2GB). So at 8192 the EXTRACTOR ALONE peaks ~10.5–11GB; add
PPO heads+grads+optimizer temps and BOTH arches cross 11.6GB — 8192 was always on the edge, parent survived by luck/
marginally-emptier card, new tipped over (environmental/transient, not architectural). FIX (CONFIRMED, live): `--batch-size
8192→4096` + `--grad-accum-steps 2→4` (effective batch UNCHANGED 16384; per-row slope ⇒ peak ~halved → **6.2GB**) +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. OOM'd dir moved aside to `_oom1`. LESSON: with this arch on a
~12GB card, **8192 micro-batch is over the line — use ≤4096 + grad-accum to recover the effective batch.** Detach /
refine / reattend are NOT memory regressions.
**SEARCH-TEACHER timing — owner's call: enable when `win_rate_vs_bots` hits ~70%** (≈10% below the ~90% plateau —
turn it on at the leading edge of convergence to nudge toward the global optimum, not after it's fully settled).
NOT from step 0 (CPU-saturated box: the teacher's search workers steal rollout CPU + a fresh policy has no plateau
to attack). Enable via a plain RESUME (stop launcher → relaunch `--model <latest ckpt>` + `--search-teacher
--search-teacher-coef 0.1 --teacher-persistent`, NO `--run-name` to dodge the resolve_launch_run_dir bug).
**ENABLED 2026-06-27 ~00:42** at step ~10.6M / bot-WR 0.78 (crossed 70% at 6M; first 3-hourly check caught it):
resumed from `final_model_interrupted.zip`, teacher flag live, **GPU stayed 6.06GB** (teacher search workers are
CPU-side → no new OOM), no crash. CRON-WRITING GOTCHA: a `grep -o '--search-teacher' | head -1 && echo ON`
test FALSE-POSITIVES — `head -1` always exits 0, so the `&&` always fires; gate on grep's OWN exit status
(`grep -q ... && echo ON || echo OFF`) instead. Later 3-hourly checks (03:33/06:33/09:33) now monitor whether the
teacher distils confirmed corrections + pushes past the self-play plateau (the coef-0-vs-0.1 honesty gate is the
real proof; "learns≠helps"). The
exploiter line (v1 basin-locked, v2 plateaued below parity with bounded specialization) was concluded — no
decisive exploit; pivoted to the teacher per this plan.

**THROUGHPUT finding:** the self-play OPPONENT forward + its obs-build run on CPU in the SubprocVecEnv workers
(trainee is on GPU, which is ~86% idle); so cutting a transformer layer buys ZERO throughput (GPU-side). The real
"GPU the opponent" fix = a centralized batched inference server (UNBUILT, significant). The built CPU-cheapening
lever was opponent-distillation — which the owner had me **REMOVE entirely** this session (deemed dead weight):
the whole `src/agents/training/distill/` package + 18 wiring files (selfplay_callback/snapshot_pool/wrappers/
launcher app+format/train_rl_agent), 1894 deletions, **staged in the bridge-cse worktree at base 9ca7c56, 3058
unit tests GREEN (verified twice), NOT yet shipped** (awaiting /gen3ai-ship). Search-teacher's own AWR
"policy-distillation" + the frozen-opponent `check_compatible` comments were KEPT (different sense of "distill").

**SEARCH-TEACHER usage (commit 9ca7c56, on main):** selective ExIt — depth-2/beam-3 beam search (clone via
serializeBattle) finds Wilson-CI-CONFIRMED-better corrections on falsify-proven loss craters, distils A* into the
POLICY via an AWR aux loss (teaches policy not critic — value-only was rejected as it biases GAE). Enable:
`--search-teacher --search-teacher-coef 0.1 --teacher-persistent`. GOTCHAS: `--search-teacher` is store_true +
NOT resume-inherited (re-pass EVERY relaunch); `--search-teacher-coef` defaults 0 = no distill (must set nonzero);
the per-cycle opponent resolver only resolves self-play SENTINELS + bots (exploiter `ext_` targets → unresolved →
persistent mode effectively required); ~13s/correction CPU.

**TEACHER-CEILING DIAGNOSIS + lever ranking (2026-06-27, ai_v7_01_teacher_0626 @32M, measured + 3-lens adversarial panel
that VERIFIED claims against code):** the search-teacher CONVERGED (agree 0.87→0.98, share 0.71→0.025, bot-WR/ELO FLAT
in the 0.78/1820 band across 12M+ steps since teacher-on) because its addressable bucket is TINY: model-free triage (214
losses) + falsify/calibration crater-mass bracket → **policy_reducible (proven action-fixable = the teacher's ENTIRE
ceiling) = 6.9%**; aleatoric 14.5% + lost_position ~15% ≈ 30% uncoachable; **critic_overvalued ~61%** (selection-inflated
UPPER bound, but the dominant residual) — i.e. the #1 recoverable triage bucket is **critic_blindspot (26.6%, critic rated
losing positions as winning)**, then **positional_grind (33.6%, matchup-lost/team-draw, wp_even=0.50 = half never favored)**.
So ~60% of losses are NOT policy-action mistakes the teacher can fix. mean_adv=0.51 proves the surviving corrections ARE
real (0%→51% confirmed rollout), so the machinery works — the POOL is small. LEVER RANK (referee, GPU-week-grade):
**(1) FIX the post-faint window-centering bug — the user's "choose a turn earlier", and it's a REAL code defect:**
`selection.py` L101-104 expands the window BACKWARD-only (`range(1,window+1)` t-back) with ZERO forced-turn detection, so
`post_faint_replacement` (6.5%/2.17%-recov) is searched at a NO-CHOICE forced-replacement turn while the real decision sits
1-2 plies upstream; fixing it recovers that bucket AND de-confounds the ~61% critic upper bound every lever reasons from.
Cheapest, selection-only, can't corrupt training. (2) TEAM DIVERSITY (cheap half of the regime change — attacks the 33.6%
grind/team-draw at its source). (3) CRITIC SHAPING A/B (value-dist/win-prob read_only→shaping, widen vmin/vmax beyond
±8 [memory: widen vmax first], the dominant residual + synergy: the better_line beam is critic-GUIDED [`_keep_beam` sorts
by V] so a sharper critic also improves teacher proposals — BUT confirm is ground-truth-independent + agree=0.98 chokes the
synergy; risks a flat-stable run; fresh-run only). **(4) DEEPER/WIDER SEARCH = the user's "search further ahead" = NOT worth
it standalone** — bounded by agree=0.98/share=0.025 (deeper beam mostly RE-FINDS already-agreed lines → already_known rises;
tighter Wilson + upstream both SHRINK yield; extra confirm cost on a saturated box). depth/beam/window are HARDCODED
(produce.py L31; only confirm_rollouts is CLI). Run depth-4 ONLY as a kill-switched falsification probe AFTER the centering
fix (null = teacher exhausted → pivot to team diversity). (5) full Metamon offline-human-replay = most expensive, gate hard.
**CHEAP NEXT STEP (read-only, spare CPU, :8001 untouched):** patch selection's forced-turn re-centering, re-run triage/
falsify-scan OFFLINE over saved traces → measure how much post_faint re-buckets into policy_reducible + how much the 61%
critic UB shrinks; same patched candidates depth-2-vs-4 in pure-selection mode → does depth-4 surface NEW confirmable
corrections or re-find agreed ones. Decides whether to spend ANY GPU-weeks on the teacher BEFORE committing.

**MEASURED 2026-06-27 (stopped the live run, ran the offline experiment with the full machine free; both user levers
EMPIRICALLY REFUTED — DO NOT spend GPU on either):** (A) **"choose a turn earlier" — NOT real:** over 214 latest-step
losses the worst-ΔV crater is a switch-only FORCED post-faint turn in **0.0%** of cases (the 6.5% post_faint craters are
non-`move_selection`-phase invs that `selection.py` ALREADY filters at L111-112) → the panel's #1 pick (reasoned from code,
not data) is wrong; the falsify gate + window=2 already handle the upstream-cause case. (B) **"search further ahead"
(depth 2→4) — measured WASH:** mirrored `search_teacher_worker` (select_candidates falsify_gate=True → warm SearchSession →
produce_correction at depth 2 AND 4, frozen snap = final_model_interrupted 33.4M, 30 candidates): depth-2 yield **17/30
(57%)** vs depth-4 **16/30 (53%)** — net DOWN; depth-4 found 2 the other missed but LOST 3 (incl. a 0.875 + a 0.75); mean
confirmed advantage 0.412 vs 0.430 (identical). The ok↔gate_failed flips are **8-rollout CONFIRM NOISE** (wide Wilson CI
near the margin), not search depth → depth-4 = 2× compute for no yield gain. KEY POSITIVE: the teacher MACHINERY WORKS —
57% of falsify-gated craters produce a CONFIRMED 0%→41% correction at depth 2 (mean_adv 0.41), gate_failed 37% (falsify
over-flags, confirm correctly rejects). So the bottleneck is NOT search power (supply is fine) — it's that the addressable
decisions are RARE (~7% of crater mass, worst-case loss states) AND the live policy already plays the found A* ~98% of the
time (near-zero gradient, share 0.025) AND corrections are vs the EXACT opponent on rare states (poor generalization). The
ONLY minor teacher knob with a basis: **confirm_rollouts 8→24** (stabilizes the noisy gate, the cheap CLI flag) — but it
does NOT change the structural ceiling. CONCLUSION: stop tuning the teacher; the data points at the CRITIC (#1 bucket) +
team diversity (the uncoachable third). ai_v7_01 stopped, resumes from `checkpoint_33392145_steps.zip` (33.4M).

**LAUNCHED 2026-06-27 `ai_v7_02_critic_shape_0627` (tmux 0:0, FRESH critic-shaping run, the chosen lever):** the
ai_v7 base command with these changes — (1) **win-prob + value-dist read_only→SHAPING** (both heads now backprop into the
shared trunk; coef **0.2** each [smoke at 0.5 gave ~94% combined grad-share = too hot]; coefs are RESUME-MUTABLE so tunable
live — watch `grad/win_prob_share` + `grad/value_dist_share`, target a real-but-minority pull); (2) **value support ±8→±12**
(PopArt-normalized units; HL-Gauss edge-bin absorbs overflow so it's resolution not safety; widened for the fresh-run early
transient); (3) **+`--spread-belief-nature --spread-belief-nature-marginalize`** (the v40 generative nature/EV head + op
nature-marginalization — already shipped, just ENABLED; "add support for nature marginalization" = enable, NOT new code);
(4) **TEACHER OFF** (already absent in the base). FRESH (no --model), `--sync-to-main` pinned ee8f2b92. GPU-SAFETY: shaping
backprops the trunk TWICE (policy/value + aux heads) → batch-4096 hit 9.6GB/2.3 free (≈ the OOM'd footprint) → cut to
**`--batch-size 2048 --grad-accum-steps 8`** (same eff batch 16384) → **5.1GB/6.8 free**, FPS 1027 unchanged (CPU-bound, grad-
accum is free). Smoke PASSED (roundtrip + value_dist_share 0.46 / win_prob_share 0.48 / natureev_nature_acc 0.40 all live, no
NaN). A/B = compare its bot-WR/ELO/calibration(PIT, brier) trajectory vs ai_v7_01's plateau (0.78/1820) at matched steps; the
THESIS to confirm: critic-shaping cuts the critic_blindspot 26.6% bucket → fewer thrown-from-winning losses → WR breaks the
band. WATCH: shaping shares not dominating, value_dist PIT→0.5 (calibrated), no OOM, nature_acc rising + largest_bias→0.
**PROGRESS 2026-06-28 @16M — FIRST POSITIVE SIGNAL, plateau may be breaking:** trajectory went BEHIND early (shaping
tax: 6M bot-WR 0.632 vs ai_v7 0.717) → caught up by 10M (0.765 vs 0.780) → **PULLED ABOVE at 14-16M: bot-WR 0.830/0.824
vs ai_v7's 0.766/0.771 (+0.05-0.06), ELO 1848/1850 vs 1808/1811 (+40)**. 0.83 bot-WR is ABOVE the 0.78 band AND above
ai_v7's all-time best (0.809/ELO 1855) — reached EARLIER. Mechanism CONFIRMED engaged: value_dist PIT=0.500 (perfect
calibration), EV 0.74-0.76 (rising), brier 0.147. Shaping shares a healthy MINORITY (win_prob 0.08-0.20, value_dist
0.10-0.18, combined ~0.25 — coefs 0.2 well-tuned, the earlier 0.36 value_dist creep resolved by 11M). Nature belief
learning (nature_acc 0.71, largest_bias -38.8→-22). GPU steady ~5GB, clean 6h restarts. CAVEAT: 2 cycles above the band
≠ proof; ai_v7 didn't plateau until ~30M, so the DECISIVE test is holding/extending above 0.78/1820 through 30M+.
Strongest evidence yet the plateau is beatable — and it VALIDATES the loss-attribution diagnosis (critic_blindspot 26.6%
was the #1 lever, NOT the search-teacher).
**CORRECTION 2026-06-28 (owner: "compare against ai_v6_13, the BEST run; latent cosine seems worse"):** my "breaking the
plateau" framing was WRONG-BASELINED. The TRUE best is **ai_v6_13_outgoing_dmg_0620 = bot-WR 0.900 / ELO 1967 @104M**;
ai_v7_01 (0.809) was itself a REGRESSION, not "the plateau." At MATCHED steps ai_v7_02 is merely TIED with ai_v6_13 through
20M (both ~0.84; ai_v6_13 20M:0.821, ai_v7_02 0.840) — NOT beating it. The real bar is ai_v6_13's LONG CLIMB 0.82(20M)→
0.868(40M)→0.899(104M); ai_v7_02 must replicate that, unproven. **DETACHED-BELIEF REGRESSION CONFIRMED (the user's latent-
cosine catch):** ai_v6_13 used `belief_grad_mode=SHAPING`, the whole ai_v7 line uses `detached` → `belief/latent_cosine_above_chance`
**0.316 → 0.001 (AT CHANCE)**, `species_acc_above_chance` 0.68 → 0.15 (~5× worse), move_recall 0.75→0.64. Mechanism: detached
reads a STOP-GRAD trunk → the trunk never co-adapts to make hidden mons distinguishable → the belief heads collapse to baseline.
The v41 detach was meant to kill belief↔policy INTERFERENCE but empirically killed the BELIEF ITSELF. HYPOTHESIS: ai_v6_13's
0.82→0.90 climb leaned on its strong hidden-team belief (fewer surprise-OHKO/matchup losses in the long grind) → ai_v7_02's
crippled belief likely CAPS it below 0.90. CONFOUND: ai_v7 also dropped reattend + 10→7 history, so the 0.90-vs-0.84 ceiling
isn't purely the detach — but the detach is the one change with a CONFIRMED quality regression. **OWNER DECISION: keep ai_v7_02
running (re-baselined vs ai_v6_13), watch for the detach-cap — does it climb past 0.84 toward 0.90 or STALL ~0.84.** If it
stalls by ~40M, the next run = **ai_v7_03 = critic-shaping + belief SHAPING** (revert the detach; best-of-both — ai_v6_13's
strong belief ⊕ the proven critic-shaping win; belief_grad_mode is resume-immutable → must be fresh). Cron re-baselined to
ai_v6_13 matched-step + the stall trigger.
**UPDATE 2026-06-28/29 @40M — PASSED the decisive test, detach-cap NOT manifesting:** ai_v6_13's signature climb was
0.836(36M)→**0.868(40M)**; ai_v7_02 made the SAME move to **0.866(40M)** (ELO 1922 ≈ v6's 1926). MEAN WR since 30M is
IDENTICAL: v7_02 0.850 vs ai_v6_13 0.850 — the two runs track near-perfectly through 40M. So the acute detach-cap fear
(v7_02 stalls ~0.84 while v6_13 pulls to 0.868) DID NOT happen. KEY INSIGHT: v7_02 matches the BEST run DESPITE the at-chance
latent belief (still 0.003) → the CRITIC-SHAPING is fully COMPENSATING for the crippled belief (validates the lever). Shaping
stayed a healthy minority throughout (shares ~0.1, never needed a coef cut), pit_mean rock-solid 0.500, nature marg working
(largest_bias −38.8→−20). Remaining watch = the LONG tail 60M→104M (ai_v6_13's final push 0.866→0.899). STRENGTHENS the
ai_v7_03 case: critic-shaping (matches 0.90-pace alone) + belief SHAPING (ai_v6_13's strong belief) could EXCEED 0.90.
**@46M — now AHEAD of the best run:** v7_02 WR **0.882** / ELO **1959** vs ai_v6_13's 0.875/1938 at matched 46M; mean WR
since 30M edges ahead (0.857 vs 0.855); ELO 1959 within 8 of ai_v6_13's ALL-TIME peak (1967) at <HALF the steps (46M vs
104M). Clean sustained climb 0.853→0.882, detach-cap hypothesis now UNLIKELY (critic-shaping matches/edges the best despite
latent belief at chance 0.004). Verdict upgraded to ON-TRACK-TO-BEAT-0.90; remaining test = sustain to 0.90+ over the long
tail 60→104M.
**@58M — SURGING, detach-cap REFUTED:** after a one-cycle dip to 0.853@50M (oscillation, recovered), v7_02 surged to WR
**0.895** / ELO **1959** @58M — within 0.005 of ai_v6_13's ALL-TIME peak (0.900/1967), reached at ~56% of the steps (58M
vs 104M), clearly ahead at matched steps (+0.027 WR; mean 40M+ 0.870 vs 0.862). The crippled at-chance latent belief
(0.006) is CONCLUSIVELY not capping it — critic-shaping carries the run to/past the best, FASTER. Healthy throughout
(pit 0.500, shaping shares ~0.1, nature_acc 0.78 / largest_bias −18 best yet). Next 1-2 cycles: does it cross 0.90 =
beat the all-time best. CONCLUSION SO FAR: critic-shaping (the loss-attribution-diagnosed lever) WORKS; detached belief
did NOT cap bot-WR. ai_v7_03 (critic-shaping + belief shaping) could push even higher / help vs-human where belief matters.
**@62M — BEAT the all-time best on ELO:** v7_02 peak ELO **1968 > ai_v6_13's lifetime peak 1967**, reached @60M vs 104M
(~58% of the steps); WR peak 0.895 (oscillating high-0.88s, band shifted up to ~0.87-0.90; 62M cycle a normal pullback to
0.868). Mean WR 40M+ 0.872 vs 0.863. CRITIC-SHAPING IS A CONFIRMED WIN — surpasses the previous best faster, detach-cap dead.
WR not yet SUSTAINED >0.90 (peaked 0.895), ~40M steps still to run. Healthy throughout (pit 0.500, shares ~0.1).
**@92M — BROKE 0.90, DECISIVELY BEAT the all-time best (the plateau is BROKEN):** after oscillating high-0.88 for ~30M (60-88M,
dead-even with ai_v6_13), v7_02 stepped UP hard at 90-92M — WR **0.915→0.920**, ELO **1982→1998** (two consecutive cycles, not
noise). At 92M: **WR 0.920 / ELO 1998 vs ai_v6_13's LIFETIME best 0.900/1967 → +0.020 WR / +31 ELO, reached at 92M vs 104M, still
climbing.** Mean WR 60M+ now clearly leads (0.884 vs 0.879). DETACH-CAP FULLY DEAD: critic-shaping not only matched but EXCEEDED
the strong-belief ai_v6_13 DESPITE latent belief at chance (0.013). DEFINITIVE: the loss-attribution diagnosis was RIGHT — the
CRITIC (critic_blindspot 26.6%) was the lever, NOT the search-teacher. Only at 92M of a 300M target → lots of runway left. The
combined recipe (critic-shaping + nature marg, teacher off, detached belief, batch 2048/8) is the new BEST. ai_v7_03 (add belief
SHAPING) could go even higher / help vs-human where hidden-team belief matters.

Links: [[project_exploiter_league_tooling]] · [[project_pfsp_phase1_built]] · [[project_positional_grind_decomposition]]
· [[project_plateau_diagnosis_2026_06_09]] · [[project_l3_oracle_grind_l4]] · [[project_search_teacher]]
· [[project_value_dist_head_status]].

**RIDER (2026-08-23, ledger e38f029):** the Metamon framing needs precision — VERIFIED against
arXiv 2504.04395 v2: in gen3ou their BC = 35 GXE, offline RL = 42, final 64 (90.1st pct) via
SELF-PLAY on top. Human replays were the bootstrap + opponent-diversity source, not the strength
lever; gen3ou is their WEAKEST gen. Parsed dataset published (cc-by-nc-4.0, gen3ou 2.69 GB).
Feasibility memo: designs/research_state/metamon_replay_feasibility.md (~18–30 d lift; hindsight
own-team imputation is the risk — we can meter it with a degraded bridge battle, they couldn't).
