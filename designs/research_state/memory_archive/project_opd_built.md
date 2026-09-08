---
name: project_opd_built
description: "On-policy self-distillation (OPD, --opd-coef): KL-to-π' aux loss extending the search-teacher — BUILT + verified, NOT shipped"
metadata:
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — ai_v7-era on-policy self-distillation, verdict NEUTRAL; era closed. Preserved verbatim; nothing below is current.

> **⚠️ DEAD CLAIM — DO NOT CITE THE "~⅔ MATCHUP-LOST / TEAM-DRAW" NUMBER (owner-corrected 2026-07-24,
> the THIRD time).** The attribution was **CIRCULAR**: recoverability was judged by our OWN
> policy/critic (and shallow search over that critic), so "unwinnable / uncoachable team-draw" only
> ever meant **"unwinnable by a policy of THIS strength"** — which is precisely the thing we are trying
> to improve. The falsifying test is simple and decisive: **a strong human beats our bot easily on the
> same teams**, so those games are not team-draw. Treat that bucket as **HEADROOM, not a floor**, and
> never use it to argue a lever is not worth building. (Prior corrections:
> [[project_l3_oracle_grind_l4]] 2026-07-03, [[project_exploiter_no_team_advantage]] 2026-07-14 — the
> equal-pilot mirror already showed the exploiter's wins were exploitation, not team advantage.)



**STOPPED 2026-07-03 @135.9M steps (clean SIGTERM, final_model_interrupted.zip, resumable) to free win-0/GPU
for the TSS+Starmie specialist PoC ([[project_tss_specialist_poc]]). OPD VERDICT = NEUTRAL-to-slightly-below
ai_v7_02 baseline (agree_rate flat ~0.90-0.96, ELO ~1951 vs 1978-1998, bot-WR ~0.884 vs 0.90-0.92;
grad/opd_share ~0.42 + slightly-negative policy_cosine = mild gradient-interference fingerprint). The 2
crashes/ files were STALE June-30 ai_v7_02 carry-overs from the folder copy, NOT OPD crashes.**

**ON-POLICY SELF-DISTILLATION — SHIPPED (4fed4bd, 2026-07-02) + LAUNCHED as ai_v7_04_opd_selfdistill_0702
(tmux win 0).** Ship also added the CPU-ONLY TEACHER fix: both SearchTeacherCallback worker-spawn sites set
`worker_env["CUDA_VISIBLE_DEVICES"]=""` so the search/confirm workers never init a CUDA context on the
learner's card (the OOM class that killed the opponent-compile run). VERIFIED live: opd_coef=0.5 recorded,
GPU compute-apps = ONLY the learner (4 teacher-worker PIDs absent from nvidia-smi), 0 FATAL, episodes WIN
(continues from the strong model). RUN SETUP: stopped ai_v7_03 (belief-shaping, clean SIGTERM, resumable);
COPIED the best-model folder ai_v7_02_critic_shape_0627 → ai_v7_04_opd_selfdistill_0702 (full cp -a, ai_v7_02
PRESERVED) so the OPD run inherits the 20-snapshot pool + resumes ckpt_106685764 IN the copied dir (no
--run-name → continues in place; --sync-to-main pulls 4fed4bd). Launcher script /tmp/launch_ai_v7_04.sh.
Teacher = --teacher-persistent (samples the copied pool = selves = "ourselves as the opponent"). Watch:
opd/agree_rate↑, grad/opd_share few-%, teacher/yield, WR/ELO vs ai_v7_02's 0.92/1998. A/B CONTROL (if wanted
later) = same ckpt w/o OPD (/tmp/launch_opd_control.sh); currently running the treatment only (1 GPU).

**(build detail, unchanged) ON-POLICY SELF-DISTILLATION BUILT (2026-07-02, worktree bridge-cse).**
OPD = extend the shipped SEARCH-TEACHER (which was on main after all — commit 9ca7c56, my "not shipped"
memory was STALE) by distilling the FULL improved distribution π' via `--opd-coef · KL(π' ‖ π_student)`
instead of only the single verified-better action A* (the AWR `_searchteacher_loss`). π' is built from the
beam's per-action **`backup`** values in `out["candidates"]` (better_line.py) — softmax((v−max_legal)/opd_beta)
over legal actions, completed-Q floor (min legal / baseline) for unsearched legal slots, 0 illegal, L1-normed;
terminals map win/loss/tie→+1e3/−1e3/0. Reuses selection/search/confirm/CorrectionBuffer/workers WHOLESALE.
Training-only coef (mirrors `search_teacher_coef` EXACTLY): NOT in ModelVersion/check_compatible/any check_*,
coef 0 byte-identical → BOTH A/B arms resume ai_v7_02 with zero FATAL risk. "Ourselves as the opponent" =
`--teacher-persistent` samples the run's own snapshot pool (ai_v7_02 has 20). Requires `--search-teacher`
(parser.error guard).

**12 files:** buffer.py (`Correction.pi_target` field + to_tensors), produce.py (`_build_pi_target` +
build_pi_target/opd_beta params), search_teacher_{worker,persistent_worker}.py + teacher/callback.py (thread
opd_build_pi_target/opd_beta; pi_target rides the shard .npz, NaN row=None), instrumented_ppo.py (`opd_coef`/
`opd_beta` attrs + static `_opd_loss` masked-forward-KL + train() fold sampling the SAME buffer + `aux_probe_
terms['opd']` + `opd/*` metrics), train_rl_agent.py (`--opd-coef`/`--opd-beta` _resolve+_model_hparams+both
build paths + guard + callback threading), format.py (opd/* + grad/opd_* TUI labels), CLAUDE.md + 3 test files.
VERIFIED: 3075 unit tests pass (+15 OPD incl. OFF-byte-identical param-update guards), CLI+guard work,
serverless --debug smoke (roundtrip PASSED, Training complete, no crash). Metrics: opd/{kl,pi_target_entropy,
agree_rate}, grad/opd_share+policy_cosine, teacher/yield.

**A/B (staged, NOT launched):** control=`/tmp/launch_opd_control.sh` (continue ai_v7_02 ckpt 106685764, no OPD),
treat=`/tmp/launch_opd_treat.sh` (+ --search-teacher --teacher-persistent --opd-coef 0.5). BOTH need OPD
SHIPPED to main first (launcher --sync-to-main pulls origin/main; OPD isn't there yet). One GPU: ai_v7_03 is
currently running it — decide stop/2nd-GPU/queue. Watch: opd/agree_rate↑ + grad/opd_share few-% + WR/ELO vs
ai_v7_02's matched-step curve. HONESTY GATE: search finding a better line ≠ helps (the same gate as
~~[RETRACTED — circular, see the banner above]~~ [[project_search_teacher]]); ~⅔ grind losses uncoachable team-draw.

**DESCOPED follow-ups (the knowledge-quality levers from the design chat):** a `--teacher-self-only` flag
(persistent samples pool+bots today → distilling bot-blunder-punishing; gate to pool selves), opponent-action-
quality weighting (down-weight corrections whose Δwin rode an opp blunder), best-response/multi-opponent
confirm. Design note: designs/learning/on_policy_self_distillation.md.
