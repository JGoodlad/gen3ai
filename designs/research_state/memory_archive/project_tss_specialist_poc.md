---
name: project_tss_specialist_poc
description: "TSS specialist arc — DOUBLE CORRECTION 2026-07-08 (read the bottom blocks FIRST): (1) the eval 'plateau' verdicts on 05–08 were OOD artifacts (eval_worker hardcoded the default teambuilder — specialists measured on random teams); (2) the training-side reads were INFLATED by stochastic opponents/sampled-self (07's '95% vs target' → 27% greedy-greedy; 08's '100% vs bots' → 55.8%). TRUE standings: 07 = 65.5% bots / 27% vs-02 vs the generalist pilot's 93.5% / 55% — specialists learned a lot, mastered nothing, diversity wins by ~30pts. Only the greedy in-distribution remeasure is a clean yardstick."
metadata:
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — ai_v7-era single-team specialist arc, double-corrected; era closed. Preserved verbatim; nothing below is current.

**2026-07-03. Single-team specialist PoC — SHIPPED (49aeb1e) + LAUNCHED (ai_v7_05_tss_specialist_0703, tmux
win 0).** Question the PoC answers (owner): "before archetypes, can a model be REALLY GOOD with just ONE
team?" — a from-scratch specialist on the CANONICAL classic TSS+Starmie ("big five + Starmie" = TTar / Skarmory
/ Blissey / Swampert / Gengar / Starmie) vs our current-best generalist + the heuristic bots. Owner waived the
best-response-overfit guard ("I don't care if it beats bot idiosyncrasies — I'll make 10 specialized opponents
later") and REQUIRED from-scratch ("do not poison its mind with generalist ideas" → NO --model, random init).

**SHIPPED code (49aeb1e, 4 files, OFF byte-identical):** `--trainee-team <file>` pins ONLY the TRAINEE's team
pool to one Showdown-export team (single-team `Gen3Teambuilder([str])`, validates on construction; OPPONENTS
keep the full diverse pool) + `--exploiter-keep-bots` / `--exploiter-bot-fraction` (default 0.5) mixes the
heuristic-bot floor BACK IN alongside the fixed `--exploiter` target per episode (else the target is the sole
opponent). `data/teams/specialist/tss_starmie.txt` = BYTE-IDENTICAL to sample `f6229d2c867e21d6.txt` (the only
sample team with all 6). `wrappers.py` `_select_episode_opponent`: `if keep_bots and rng<frac:
_pick_floor_opponent() else exploiter_player`. Tests: wrappers_test 3 (mix + frac 0/1 edges).

**RUN (ai_v7_05, `/tmp/launch_ai_v7_05.sh`):** from scratch (no --model), IDENTICAL v42 arch to ai_v7_02
(config_version 42, gen3_opp_hp_typed_candidates_v1, obs 2992, 7 hist) so specialist-vs-generalist is clean;
opponents = frozen `models/ai_v7_02_critic_shape_0627` (--exploiter) 50/50 with bots; --ent-coef 0.05 (owner:
"the only lever we really have is the entropy lever"); serverless (--use-showdown-bridge, --async-rollout),
--sync-to-main (pulls 49aeb1e). Replaced ai_v7_04 (OPD, STOPPED @135.9M — see [[project_opd_built]]) in win 0.
Verified healthy: metadata cli_args all correct (trainee_team/exploiter/keep_bots=True/frac=0.5/self_play=False/
model=None), single learner on GPU, 10+ episodes, 0 FATAL, first episodes LOSS 0v3 (expected from scratch).

**VERDICT METRIC — ADDED (owner-approved restart, ~15:17).** First launch had NO `--stable-opponents` and
exploiter mode does NOT auto-eval the target → no direct specialist-vs-generalist number. Owner: "include the
win rate against the labeled opponent." RESTARTED (stopped launcher, renamed the <15-min from-scratch dir →
`ai_v7_05_tss_specialist_0703_aborted_noeval` for a clean fresh start, relaunched win 0) WITH
`--stable-opponents models/ai_v7_02_critic_shape_0627` → produces `eval/win_rate_vs_ext_ai_v7_02_critic_shape_0627`
+ `eval/elo_vs_ext_...`, the direct "does the specialist BEAT the strong generalist AT this team" verdict.
EVAL-ONLY here (self_play=False → NOT injected into training, zero contamination; the documented exploiter-verdict
pattern, see [[project_archetype_competence_gradient]]). Verified applied (metadata cli_args stable_opponents set),
single learner on GPU, episodes completing, 0 FATAL.

**Success read (when it has trained):** specialist WR-at-TSS climbs well ABOVE the generalist ai_v7_02's WR at
the SAME team (deferred CONTROL: measure ai_v7_02's own WR piloting TSS+Starmie) → competence is COACHABLE via
specialization → greenlights the "jack-of-all-trades AND expert" build (routed policy + isolated-specialist
distillation + DRO, after the Rust sim). Flat/below → matchup-bound. This is the per-team instance of the
archetype-specialist decisive test in [[project_archetype_competence_gradient]]; pairs with
[[project_plateau_research_2026_06_25]] (only a REGIME change, not more self-play, raises the ceiling).

**2026-07-06 UPDATE — ai_v7_05 STOPPED (plateaued), ai_v7_06 LAUNCHED with exploiter TEMPERATURE ANNEALING.**
ai_v7_05 never took off: bot-WR flat ~0.16 the whole 60M→112M window; ELO peaked ~1341@102M then DECLINED to
~1258@108M (rolling over); 0/8 bots >0.5; ext-vs-ai_v7_02 flat ~0 → confirmed plateau (the from-scratch trainee
never got a foothold vs the strong frozen target). SIGTERM'd the launcher clean (saved checkpoint_111279963).
Hypothesis: a from-scratch trainee vs a STRONG frozen target is crushed every game → PPO advantage ~0 → no
learning signal.

SHIPPED `gen3_exploiter_temp_anneal_v1` (commit a0f3047): anneal the EXPLOITER target's SAMPLING TEMPERATURE over
training — a difficulty curriculum via opponent STOCHASTICITY (not by swapping opponents). `--exploiter-temp-start`
(hot → flatter logits → weaker/noisier so the trainee wins some games early) linearly → `--exploiter-temp-end`
(default 1.0 = the target's true distribution) over `--exploiter-temp-anneal-frac` of --steps, held after.
`ExploiterTempAnnealCallback` computes the temp from `_current_progress_remaining` each rollout (sync+async both
fire on_rollout_start) and pushes it via `env_method("set_exploiter_temperature", T)` (the set_self_play_target
idiom); `MaskableAgentWrapper.set_exploiter_temperature` sets `RLPlayer._temperature` (read fresh each
choose_move). Training-only (NOT version-locked, forwarded verbatim on resume — anneal continues from the resumed
step), OFF byte-identical (registered only when --exploiter-temp-start set), metric `train/exploiter_temp`. Files:
`exploiter_temp_callback.py` (+test), wrappers.py setter (+test), train_rl_agent.py (3 flags+validation+register),
launcher/format.py (TUI label), training/CLAUDE.md. Tests 10 new + 1166-suite green + --debug exploiter smoke
(anneal 2→1 confirmed).

LAUNCHED **ai_v7_06_tss_temp_anneal_0706** (tmux win 0, `/tmp/launch_ai_v7_06.sh`): IDENTICAL to ai_v7_05's config
(same v42 arch, --exploiter ai_v7_02 + --exploiter-keep-bots 0.5 + --stable-opponents ai_v7_02 verdict +
--trainee-team tss_starmie + --ent-coef 0.05, from scratch, --sync-to-main pins the isolated worktree to a0f3047)
PLUS `--exploiter-temp-start 2.0 --exploiter-temp-end 1.0 --exploiter-temp-anneal-frac 0.2` (target anneals 2→1
over the first 60M of 300M steps, held at 1.0 after). Verified healthy: worktree pinned 787f996d, 660 fps,
`exploiter_temp=2` live in metrics, EXPLOITER emit confirms "temp 2→1 annealed over 20%", temp flags in metadata
cli_args (resume-safe). WATCH: `eval/win_rate_vs_ext_ai_v7_02_critic_shape_0627` + ELO climbing off the ai_v7_05
plateau floor as the temp ramps, and `train/exploiter_temp` tracking the anneal. Hyperparams (2.0/1.0/0.2) are a
first guess — easily changed on a relaunch.

**2026-07-06 — DYNAMIC WR-RATCHET BUILT (uncommitted, NOT shipped/run).** Diagnosed WHY ai_v7_06 climbs
~5× slower than ai_v7_02's curve (bot-WR @12M: 15% vs 78%): ai_v7_02 was `--self-play` (opponents grow
WITH it → always near 50% WR → max advantage signal), whereas ai_v7_06 is an exploiter vs a FROZEN
1983-ELO wall + 50% bots — half its games are ~unwinnable (advantage≈0) AND the fixed temp ~1.8–2.0
barely dents a 1983 policy for a from-scratch net, so the target half is near-dead signal. (bot-WR is also
a SIDE metric for an exploiter — its real objective is beating the target, still 0% vs greedy.) So the
fixed start is too weak — vindicating the user's dynamic-schedule instinct. BUILT the fix (user: "do a+c"
= keep ai_v7_06 running as the fixed baseline + implement the ratchet): `--exploiter-temp-mode ratchet` +
`ExploiterTempRatchetCallback` — start the target near-trivial (`--exploiter-temp-start` HIGH, e.g. 5.0)
and ratchet temp DOWN (`*=--exploiter-temp-ratchet-factor` 0.9, floored at end) ONLY when the trainee's
measured TRAINING WR vs the target (NOT the greedy eval WR) clears `--exploiter-temp-ratchet-wr` (0.55)
over a window of `--exploiter-temp-ratchet-games` (500) target-games. ONE-WAY (never raises temp → no
comfort-trap, mirrors stable-mastery). Signal: wrapper counts per-episode target outcomes
(`_record_exploiter_outcome`/`exploiter_winrate_totals`, bots excluded); callback diffs cumulative totals
via env_method each on_rollout_end. RESUME-SAFE (temp persisted to `<run>/exploiter_temp_state.json`,
restored on restart). Metrics `train/exploiter_{temp,target_wr,temp_ratchets}`. 7 files
(wrappers.py + exploiter_temp_callback.py + train_rl_agent.py + format.py + 2 tests + CLAUDE.md), 94+1179
tests green, --debug ratchet smoke fired ratchets 5.0→4.05→3.65 + wrote the state file. NOT shipped
(awaiting /gen3ai-ship) and NOT launched (1 GPU busy with the ai_v7_06 fixed baseline) — run as the A/B
contender once shipped + a GPU frees. Launch = ai_v7_06 config but `--run-name ai_v7_07_tss_temp_ratchet_*`
+ `--exploiter-temp-start 5.0 --exploiter-temp-mode ratchet` (defaults for the rest).

**2026-07-07 — ai_v7_06 fixed baseline PLATEAUED → SHIPPED ratchet + LAUNCHED ai_v7_07 (the A/B).**
ai_v7_06 (fixed 2→1 anneal) confirmed a plateau: bot-WR FLAT ~13–16% for 20M steps (12M→32M, no trend),
ELO flat ~1290 (below ai_v7_05's ~1341 peak), vs-target ~1%. Diagnosis held: the fixed start 2.0 never
gave a foothold, and the clock kept annealing the target HARDER (temp 1.5 @32M, →1.0 @60M) regardless of
whether the trainee earned it → the early high-temp window passed with no breakout, remaining anneal only
tightens the wall. SHIPPED the ratchet (commit **04f7409**). STOPPED ai_v7_06 clean @32.47M (baseline
eval_results.jsonl + checkpoints PRESERVED for the A/B — new dir, no clobber). LAUNCHED
**ai_v7_07_tss_temp_ratchet_0707** (tmux win 0, `/tmp/launch_ai_v7_07.sh`): from scratch, worktree pinned
04f74090 (has the ratchet), config = ai_v7_06 + `--exploiter-temp-start 5.0 --exploiter-temp-end 1.0
--exploiter-temp-mode ratchet` (defaults wr 0.55 / factor 0.9 / games 500). Verified: EXPLOITER emit "temp
5→1 WR-RATCHETED", cli_args all correct (resume-safe), training started. WATCH: `train/exploiter_temp`
ratcheting DOWN from 5.0 (+ `exploiter_target_wr` near 0.55, `exploiter_temp_ratchets` climbing) as the
trainee earns each level, and — the verdict — `eval/win_rate_vs_ext_ai_v7_02` + ELO climbing OFF the ~1290
fixed-arm plateau (ideally faster bot-WR too, since early games now yield real advantage signal). The
RUNNING run is now ai_v7_07 (ai_v7_06 = the stopped fixed baseline).

**2026-07-07 — ai_v7_07 ratchet ALSO plateaued → RATCHET VERDICT: curriculum is NOT the lever. Launched
ai_v7_08 BOTS-ONLY control.** ai_v7_07 (dynamic ratchet) ran the temp ALL the way 5.0→1.0 (16 ratchets,
state {"temp":1.0}) — mechanically it worked (trainee cleared each level) — but plateaued at bot-WR ~14% /
ELO ~1286 / vs-target 1%, IDENTICAL to ai_v7_05 (plain exploiter) and ai_v7_06 (fixed anneal). **THREE
different opponent curricula → the SAME ~1290 ceiling** ⇒ opponent-difficulty is NOT the bottleneck; the
common factor is the single-team + exploiter regime. Stopped ai_v7_07 @18.6M (ckpt saved).

DIAGNOSTIC on the way (user caught it via TB): the belief metrics EXPOSED the single-team degeneracy. `hptype_acc`
hit EXACTLY 1.0 by 2M on ai_v7_07 while the mature generalist ai_v7_02 CAPS at 0.80 — impossible as a genuine
belief (HP-type is NOT deterministic: 30/69 pool species run split HP, Celebi 8 types). Root cause: HP-type is
UNPREDICTABLE from species (ai_v7_07 cold-starts at 0.24, BELOW the prior), so reaching 1.0 = the head READING
the already-OBSERVED type via `hp_probs` (effectiveness-narrowed, carried in the opp token), which the NARROW
specialist matchup pins for ~all slots. NOT the same-team bug (opponents ARE the full 719 pool — `species_acc`
~0.26, code = `Gen3Teambuilder(all_teams)` unconditional), NOT a privileged leak (label never in forward, op
gets the right type), but the METRIC is degenerate here + signals LOW STATE DIVERSITY (why species_acc is HIGHER
than ai_v7_02's 0.18 too) — a plausible plateau contributor.

LAUNCHED **ai_v7_08_tss_bots_0707** (tmux win 0, `/tmp/launch_ai_v7_08.sh`, ZERO code — bots-only IS the default
when no --self-play/--exploiter): from scratch, `--trainee-team tss` vs BOTS ONLY (strip the strong-target
distraction that wasted ~half the games), same v42 arch, `--stable-opponents ai_v7_02` eval-only ref, pinned
04f74090. TESTS: with the strong-opponent drag removed, does bot-WR climb toward the generalist's ~0.89 curve
(⇒ team learnable, opponent regime WAS the plateau cause), or cap ~14% (⇒ TSS matchup-limited vs the bot roster
OR 1-team pathology)? Config verified (exploiter None / self_play False / trainee_team tss), training started.

**2026-07-08 ★ THE GREAT CORRECTION — every 05–09 eval verdict was an INSTRUMENTATION ARTIFACT.**
`eval_worker._run` HARDCODED the default full-pool trainee teambuilder (`--trainee-team` never threaded into
the worker cfg) → every specialist eval (bot-WR "plateau ~7–16%", "vs-target 1%", declining ELO, "vs random
73%") measured the model piloting RANDOM teams it never trained on — pure OOD. The IN-DISTRIBUTION truth
(training-side metrics, which ARE TSS): **ai_v7_07 ratchet-exploiter beat the FULL-strength (temp 1.0)
ai_v7_02 target 94–98% with TSS from ~6M to its 18.6M stop** (`train/exploiter_target_wr`; temp hit 1.0 by
4M) — the exploiter question is ANSWERED YES, from scratch, via the temp ratchet, ckpt banked
(checkpoint_18609109). **ai_v7_08 hit training-WR ~1.00 vs the bots by ~6M** (`win_prob/label_mean`) — it
MASTERED bots-only, then SATURATED (p→1 ⇒ advantage→0: the "collapse signature" — entropy −0.87, EV 0.93 —
was a CONVERGED WINNER committing, NOT pessimism; my basin story was inverted). **ai_v7_09 (pubval) same
mastery ramp, ~1.00 by 5M** — pubval didn't change the bot ramp (never credit-limited); its test needs a
harder task. REVISED LESSONS: (1) single-team-from-scratch learning WORKS (fast); diversity buys
GENERALIZATION + valid metrics, not learnability. (2) The real signal problem was SATURATION (top of
p(1−p)), not starvation — specialists need harder opponents after ~6M. (3) Declining OOD eval = increasing
specialization (narrowing random-team piloting), expected. FIX BUILT (worktree, NOT shipped — needs
/gen3ai-ship): `trainee_team_str` threaded train_rl_agent → BOTH eval callbacks → worker cfg →
`eval_worker._build_trainee_tb` (pin > default; None = byte-identical default); tests
`src/main/eval_worker_test.py` + 104 callback tests green; ai_v7_09 (--sync-to-main) picks it up at its
next 6h restart once shipped. IN-DISTRIBUTION REMEASURE of banked ckpts (07 TSS vs greedy-02 200g; 07+08
TSS vs bots 50g/bot) running via `tmp/specialist_remeasure.py` → `/tmp/specialist_remeasure_results.json`.
CAVEAT on 07's 95%: vs the STOCHASTIC temp-1.0 target in training; the greedy-greedy remeasure is the clean
yardstick (mirror anchor: same-pilot TSS vs pool = 55%).

**REMEASURE RESULTS (same day, greedy trainee, `/tmp/specialist_remeasure_results.json`): BOTH mirrors were
distorted, opposite directions.** 07 (TSS, ckpt 18.6M): **vs GREEDY ai_v7_02 = 27.0%** (54/200; eval said 1%,
training said 95%) + **vs bots 65.5%** (eval said 14%). 08 (TSS, ckpt 56.8M): **vs bots 55.8%** (eval said 7%,
training said ~100%). So: (a) the OOD eval UNDERSTATED by 4–27× — the specialists learned a LOT; (b) the
training-side reads OVERSTATED — stochastic (temp-1.0) opponents + sampled-self + step-weighted label_mean
inflate massively (95→27 vs the target = mostly exploiting the stochastic target's noise, which evaporates vs
greedy; 100→56 vs deterministic bots = sampled-vs-argmax + possible greedy-vs-deterministic degenerate-loop
losses). TRUE STANDINGS vs the same-pilot anchors: 07 = 65.5% bots / 27% vs-02 vs the generalist's 93.5% / 55%
⇒ single-team training UNDERPERFORMS the generalist pilot by ~28pts on both axes (real, but nothing like the
80pt eval story). NUGGET: 07 (exploiter-trained, 18.6M) BEATS 08 (bots-trained, 56.8M) ON BOTS 65.5 vs 55.8 —
training vs the strong diverse target transferred better than training on the bots themselves (opponent
diversity/strength is a nutrient too). LESSONS LOCKED: (1) trust NEITHER stochastic-opponent training WR NOR
un-pinned eval — the greedy in-distribution remeasure is the only clean yardstick (now automated by the eval
fix); (2) the exploiter "95%" was noise-farming, NOT answered-yes — tempered to 27% vs greedy (below the 55%
team edge); (3) diversity still wins, deficit ~30pts not ~80.

**2026-07-09 ★★ ROOT CAUSE FOUND + FIXED — THE TRAINING MIRROR (bug B, the final piece).** The
win-attribution probe (`tmp/win_attribution_probe.py` — exact training path: Gen3Env+bridge+
MaskableAgentWrapper+bots, 08-ckpt sampled, protocol |win| lines as ground truth) showed ALL THREE witnesses
(info['win_outcome'], battle1.won, protocol) agreeing at **1.000** — the metric was HONEST; the battles were
genuinely trivial. WHY: **PokeEnv passes its single `team=` kwarg to BOTH internal _EnvPlayers, and the
per-episode opponent Players are pure decision-functions over battle2 (agent2 does the networking) → agent2's
`_team` decides the opponent's REAL team → `--trainee-team` silently pinned the OPPONENTS to TSS too.**
Protocol dump confirmed: p2 brought Skarmory/Blissey/TTar/Swampert/Gengar/Starmie every episode. Every
specialist run (05–09) trained on a TSS MIRROR vs bot pilots driving an expertise-gated stall team = trivially
beatable → genuine ~100% training WRs, fake curriculum. The 07 "exploiter" trained vs 02-piloting-TSS-only
(mirror), hence 95% training → 27% vs 02-on-pool. FIX: `Gen3Env(opponent_team=…)` post-init seam (the
_battle_class injection pattern; agent2._team = opponent builder; None = pre-fix both-sides), threaded
unconditionally from the factory. VALIDATED: probe 1.000 (mirror) → **0.483** (fixed) == fixed-eval ~55% ==
offline 51.5%, 0 witness disagreements; gen3_env_test.py pins the seam; 1134 training tests green. Docs:
training/CLAUDE.md post-mortem covers BOTH bugs. STATUS: fix in worktree NOT shipped (needs /gen3ai-ship);
**ai_v7_09 still RUNNING ON THE MIRROR** (its pinned worktree predates the fix) — its training is compromised;
recommend stop + ship + rerun on the fixed distribution. FINAL LESSON STACK: bug A = eval OOD teams (shipped
182bb4d); bug B = training mirror (this fix); factor C = stochastic-target noise-farming (~20pts, minor). AND:
"single-team-from-scratch works" is UN-TESTED — no run ever actually trained TSS-vs-diverse-field; the
original diversity question is technically still open. USER DECISION: stay on the from-scratch-exploiter
question (do NOT drift to the composite yet) — restart the experiment on fixed code.

**2026-07-09 pm — SHIP 0f16bcd WAS BROKEN (my factory edit = SyntaxError: kwarg before the positional
`mappings` in the Gen3Env call; nothing imports train_rl_agent so 1134 tests missed it). ai_v7_10 launch
crashed 3× at startup; ALSO exposed launcher bug #4: crash-restart on a fresh no-checkpoint run resolved
"latest checkpoint" to models/_goldens/ai_v3 (discovery not run-dir-scoped — P1 in the scope doc). HOTFIX in
worktree (kwarg moved after positionals) + NEW `src/main/syntax_test.py` (repo-wide ast.parse gate, 392
modules — fails on the broken tree) + argparse + factory smoke verified; AWAITING /gen3ai-ship, then launch
`/tmp/launch_ai_v7_10.sh` = ai_v7_10_tss_exploiter_fixed_0709 (from scratch, exploiter ai_v7_02 + ratchet
5→1 + keep-bots 0.5 + pubval shaping + TSS; the FIRST honest run: real opponent teams + pinned-team eval).
EXPECT: the ratchet holds high temps much longer than 07's noise-farmed 16-by-4M; anchors 93.5%/55%.
SCOPED (user-requested): `designs/ai_v8/design_matchup_config.md` — MatchupSpec single source (teams/mix/
play-modes/eval), realized-matchup fuzz, startup echo, regime tags, controllers-key-on-eval, launcher
discovery hardening; P0 ≈ 1 day.

**2026-07-09 eve — MatchupSpec P0 SHIPPED (59837aa) + ai_v7_10 LAUNCHED (tmux win 0).** One commit carries:
(1) the 0f16bcd SyntaxError hotfix; (2) `src/agents/training/matchup_spec.py` — TeamSource
(pool/default_biased/pinned/pin_biased, `build()` the ONLY builder constructor, legacy byte-parity pinned) +
PlayMode + MatchupSpec (`from_args` = the one CLI mapping; `eval_trainee_teams` defaults to `trainee_teams` —
the OOD fix structural; `to_dict`/`spec_hash` stamped into metadata cli_args `_matchup_spec{,_hash}` as the
measurement-regime tag; `summary_lines()` = the 🧭 [MATCHUP <hash>] startup echo); train_rl_agent builds both
teambuilders FROM the spec (sides independent BY CONSTRUCTION); (3)
`poke_env_gaps/matchup_realized_fuzz_test.py` — the permanent mirror-catcher (real bridge battles: trainee ==
declared pin, opponent != pin, rosters vary; PASS 8/8 distinct); (4) `src/main/syntax_test.py` (392-module
ast.parse gate); (5) launcher `find_latest_checkpoint` STRICTLY run-scoped (latest.txt → run glob → None,
NEVER the global models/ fallback that reported the ai_v3 golden). Gates: 3555 unit + 492 targeted + fuzz +
debug smoke (echo `🧭 [MATCHUP a7146263ed]`, roundtrip PASSED, metadata stamp verified). LAUNCHED
**ai_v7_10_tss_exploiter_fixed_0709** (`/tmp/launch_ai_v7_10.sh`, from scratch, --sync-to-main pins 59837aa):
exploiter ai_v7_02 + ratchet 5→1 + keep-bots 0.5 + pubval shaping 0.1 + TSS pin — the FIRST honest
single-team-vs-diverse-field run (real opponent teams + pinned-team eval). WATCH: ratchet should hold high
temps far longer than 07's noise-farmed 16-by-4M; anchors 93.5% (bots) / 55% (vs-02 team edge); verdict metric
`eval/win_rate_vs_ext_ai_v7_02_critic_shape_0627`.

**2026-07-10 am — ai_v7_10 @18M (~14h): the honest run WORKS, dramatically.** In-distribution greedy eval:
vs bots 0.60@2M → 0.91@4M → **~0.99 from 8M** (ABOVE the 93.5 generalist-pilot anchor); vs GREEDY frozen
ai_v7_02 **75%@16M → 84%@18M** (verdict metric — vs the 55% team-edge anchor and 07's true 27%); ELO 2344±105
(bot-anchored, TSS-pilot REGIME — not comparable to 02's pool-pilot 1998). Ratchet ran 5→1 (16 ratchets) fast
— but this time the WR signal was honest and the post-ratchet full-strength target is beaten greedy-greedy.
Health: hptype_acc 0.82 (mirror's degenerate 1.0 GONE ⇒ real state diversity), pubval mae 0.030 live under
shaping (pred 0.674≈target), value_dist PIT 0.499, contested_label_mean 0.80 (labels not pinned at 1). One
clean scheduled restart survived (run-scoped discovery + ratchet state + spec re-echo all correct on resume).
READ: (a) from-scratch single-team exploiter training is VIABLE on the honest distribution — the program's
core question answered YES; (b) the specialist out-pilots the generalist at TSS by ~30pts. CAVEAT: crushing a
FROZEN target is the expected exploiter outcome (Nash non-robustness), NOT "generally stronger than 02" —
next questions: does 02+pool-sentinels stay beaten (league fold-back), and does the 99%-vs-bots saturation
(p→1 advantage starvation) cap further growth.

**2026-07-10 pm — FOLD-BACK SHIPPED (d072d9d) + GPU ROTATED to the pubval A/B control.** User: "build the
exploiter functionality back into the core" → exploiters are now first-class core opponents: (1)
**per-opponent pinned teams** — `FixedOpponentEntry.team_str` read from the opponent run's
metadata `cli_args.trainee_team` (`_read_trainee_pin`, FAIL-LOUD on missing file / MatchupSpec `pin_sha`
mismatch); training: `MaskableAgentWrapper._apply_opponent_team` switches `env.agent2._team` PER EPISODE
(pool restored on unpinned episodes, agent2 untouched when no pins → byte-identical); eval:
`eval_worker._fixed_opponent_tb` measures the opponent ON its pin. (2) **exploiter auto-eval**
(`register_exploiter_for_eval`, parity Proposal A, dedup by zip/label). Guards:
`opponent_pin_fuzz_test.py` (bridge PASS 10/10 — pinned episodes field EXACTLY the pin), 19 unit tests,
suite 3574, smoke vs the pinned ai_v7_10 target end-to-end (pin + auto-reg emits fired). **ai_v7_10
STOPPED clean @23.4M / ELO 2402±123** (final_model_interrupted + checkpoint_23350277; run-scoped exit
summary correct). **ai_v7_11_tss_exploiter_nopubval LAUNCHED** (tmux win 0, pinned d072d9d, NO
--sync-to-main for A/B stability): identical to ai_v7_10 MINUS pubval — SAME matchup hash 2eb76f8465
(pubval isn't a matchup property → one measurement regime). A/B compare at MATCHED steps: time-to-99%
bots (10: ~8M), vs-target 75%@16M/84%@18M, ELO@18M 2344. Gotcha fixed pre-launch: sed had left a
blank-line break in the launch script's continuation chain (would have dropped every exploiter/team
flag). NEXT league step: the double-trap (Magneton+Dugtrio) exploiter — needs only a team file; its
fold-back now transfers the trap team automatically.

**2026-07-11 — pubval A/B = NULL + league member #2 (TRAPPER) SHIPPED (1453ff4) + LAUNCHED.** pubval
verdict (ai_v7_10 vs ai_v7_11 at matched steps, TB `win_rate_vs_ext_ai_v7_02`): mean Δ +0.032 over 11
cycles (inside 100-game noise ±0.046, single-seed → unresolvable), one early-window lean (4M 0.24 vs
0.12) that evaporated by 6M, ELO control-favored late, −2% throughput (466 vs 475 fps). ⇒ NOT the
warm-start win; **do NOT put pubval in the v8 recipe** (revisit only as read_only diagnostic). Stopped
ai_v7_11 @26.4M. USER CONSTRAINT (adopted, now ENFORCED): "only ever use an exploiter as a subset of the
SAMPLE teams" (the 32 vetted, not the ~687 bulk `other`). Built `team_archetypes.py` (pace class + style
tags, artifact `data/teams/gen3_team_archetypes.json`, keyed by strip-normalized sha) — NONE of the 32
sample teams are Mag+Dug double-trap (all 9 were `other`); 13 sample teams have a single trapper. Built
the SAMPLE-ONLY guard (`matchup_spec.validate_exploiter_trainee_is_sample` → FATAL, e2e-verified). Picked
the strongest of the 5 MAGNETON sample teams by ai_v7_02-pilot WR: **b08909ab04 MagGross**
(Salamence/Metagross/Suicune/Claydol/Blissey/Magneton, 97.5% vs bots) → `trap_magneton.txt` (Magneton
traps Skarmory = loss bucket #1). LAUNCHED **ai_v7_12_trap_exploiter_0711** (tmux win 0, pinned 1453ff4,
no-pubval recipe + ratchet 0.55 + --sync-to-main): the SECOND league exploiter, attacking ai_v7_02 via a
DIFFERENT weakness than TSS. GOTCHA (locked): a pinned team MUST be committed before a launcher run — the
launcher runs from an ISOLATED worktree pinned to a git commit, so the first launch FATAL'd on the
uncommitted team file (need /gen3ai-ship first). WATCH: `win_rate_vs_ext_ai_v7_02` climbing like the TSS
exploiter's 75%@16M/84%@18M. Suite 3596 green.

**2026-07-11 pm — TRAPPER CONVERGED + league member #3 (CM-PASS) SHIPPED (3d3227e) + LAUNCHED.** ai_v7_12
trapper result: cracked ai_v7_02 ~2× FASTER than TSS early (vs-02 0.57@4M/0.77@6M vs TSS 0.24/0.35) but
PLATEAUED at ~0.74 vs the full-strength target since 10M (temp hit 1.0/16 ratchets by 8.7M); TSS caught up
by 14M (both ~0.75). KEY FINDING: **both independent specialists top out at ~75% vs ai_v7_02, not higher**
— a shared ~25% resistance floor (consistent with the "⅔ losses draw/matchup-dependent" theme); ai_v7_02
has real holes AND real resistance through 2 axes. Stopped ai_v7_12 @15.6M/ELO 2380 (converged, checkpoint
banked). LEAGUE MEMBER #3: CM-pass Celebi exploiter — team `41357610f2` (Zapdos/Metagross/Celebi/TTar/
Swampert/Charizard, Celebi=CM-BP passer, 95% ai_v7_02-pilot) → `cm_pass_celebi.txt`. NOTE (data-quality):
initial loose filter (CM+BP anywhere on team) over-counted 3 teams; the CORRECT per-mon filter (one mon
CMs AND baton-passes) found exactly ONE true CM-pass sample team (the Celebi one). LAUNCHED
**ai_v7_13_cmpass_exploiter_0711** (tmux win 0, pinned 3d3227e, same no-pubval+ratchet recipe): THIRD
distinct attack axis (boost-passing offense). BIG-5 exploiter idea DEFERRED: only 1 sample team has the
full big-5 core (= the TSS team, already done) — a multi-team big-5 exploiter needs the UNBUILT multi-team
pin (`--trainee-team-prob`/team-list) + more big-5 sample teams. WATCH: does CM-pass hit the same ~75%
ceiling (→ league-retrain ready) or higher/lower.

**2026-07-12 — v7 LEAGUE CAPSTONE LAUNCHED (ai_v7_14) + idempotent-fork fix SHIPPED (818b56b).**
CM-pass converged ~0.76-0.81 vs ai_v7_02 (stopped @25.7M/ELO 2235) → ALL THREE exploiters land in the
~0.75-0.85 band (TSS/trap/cmpass), confirming ai_v7_02's shared resistance floor. CAPSTONE = fork ai_v7_02
(its 106.7M ckpt, full v43 self-play + PFSP 2.5 config) + the 3 exploiters as --stable-opponents @ raised
0.35 share; fold-back gives each its OWN team (verified: TSS=tss_starmie, trap=trap_magneton,
cmpass=cm_pass_celebi). Guard out of scope (self-play, not --exploiter). SHIPPED the idempotent-fork fix
(user's explicit ask, 818b56b): `checkpoint.resolve_fork_resume_model` + `run._prepare_session` swap — a
re-launched fork RESUMES its own checkpoint instead of re-copying the source (the launcher-PROCESS-restart
gap; the internal 6h loop was already idempotent). Clobber-FATAL now only for target-with-no-checkpoint. 4
launcher tests + 3607 suite green. LAUNCHED **ai_v7_14_league_capstone_0712** (tmux win 0, pinned 818b56b,
--sync-to-main). ★ THE CAPSTONE VERDICT IS THE FOLLOW-UP: after the generalist masters the 3 (watch
win_rate_vs_ext_ai_v7_10/12/13 → 0.80, win_rate_vs_bots NOT regressing = no whack-a-mole), train a FRESH
from-scratch exploiter vs hardened ai_v7_14 — if it caps BELOW ~75-85%, robustness measurably improved (the
v7 headline). GOTCHA locked: a pinned team / fork source must be COMMITTED before a launcher run (isolated
worktree). User out 24h.

**2026-07-12 pm — capstone STALLED at flat 0.35 share → DYNAMIC PFSP shipped (c27d843) + relaunched.**
ai_v7_14 (flat 0.35 stable share, ~12% exposure each of 3 exploiters) hardening FLATTENED at ~0.30 vs the
exploiters over 108M→120M (~8h): quick early jump 0.20→0.30 then oscillating flat, NOT climbing to 0.80.
vs-bots held ~0.90 (no whack-a-mole — the pool safeguard worked). Diagnosis: low exposure + the 65%
self-play pool pulling back to the button-mashing average-optimum faster than the thin flat exploiter slice
pushes out. FIX shipped (c27d843, user asked "make it dynamic"): **`--stable-opponent-pfsp`** — within the
capped stable slice, `_pick_stable` weights by (1−win_rate) floored 0.05 (spend budget on the axis it's
losing WORST; fades as mastered); `set_stable_win_rates` push each cycle (mirrors pool PFSP). TOTAL
pool-vs-stable share UNCHANGED → reporting/anti-drift-test unaffected; OFF byte-identical. Also shipped the
idempotent-fork fix EARLIER (818b56b): a re-launched fork RESUMES its own checkpoint (resolve_fork_resume_model)
not re-copy the source — so relaunching ai_v7_14 continues its 119M ckpt. RELAUNCHED ai_v7_14 (idempotent
resume from checkpoint_119821971) with `--stable-opponent-pfsp --stable-opponent-selfplay-share 0.5` (raised
+ dynamic). 3611 suite + pfsp debug smoke green. WATCH (20:00 update): do the exploiter WRs finally CLIMB
past ~0.30 toward 0.80, and does vs-bots hold. If STILL flat even at 0.5 dynamic → strong evidence the
frozen-league-in-self-play mechanism can't break the attractor → the offline-RL-on-human-ACTIONS regime
change (NOT pubval, which was value-only) is the real lever. **20:00 RESULT — clean 24h-unattended day
(108M→124M, 0 crashes/0 restarts, idempotent resume from 119.8M ckpt worked perfectly, ~394 FPS):**
vs-bots HELD ~0.91 all day (baseline 0.923, no whack-a-mole — the safeguard holds), ELO flat ~1988
(1953–2038 band). Exploiter closing PARTIAL + UNEVEN over the day — **trap 0.17→0.46** (the real climber),
cmpass 0.24→0.47 (net-up but noisy), **TSS stuck ~0.20→0.29** (the attrition axis stays hardest); NONE near
0.80 mastery, best ~0.47 (coin flip). Dynamic-PFSP effect (120M→124M, only 2 cycles) TOO EARLY: trap
0.40→0.46, cmpass 0.36→0.47, TSS 0.30→0.29 — still inside the day's noise band, needs overnight cycles to
separate signal. Leaning toward the standing finding (shared resistance floor; regime change is the real
lever) but not yet decisive at 0.5 dynamic. **Provenance diligence SHIPPED f7d33e4** (user
audit request): eval_results.jsonl rows stamp matchup_hash + carry externals (the vs-target verdict
now survives append-only); metadata matchup_history era log + resume ⚠️ MATCHUP-DRIFT warn+diff
(warn-not-fatal); eval_manifest records the regime (trainee/opponent pin shas); checkpoint sidecars
stamp matchup_hash. Note: ai_v7_10/11 both PREDATE these stamps (symmetric for the A/B).
DEFERRED follow-up (user's idea, NOT built): `--trainee-team-prob 0.5` — trainee plays the target team 50% +
random-pool 50% (the existing `Gen3Teambuilder(all_teams, bias_teams=[tss], bias_prob=)` mechanism, ~5-line flag)
to KEEP the state-diversity the belief/value systems need WHILE specializing — the "robust TSS model" build IF
the bots-only control shows the team is learnable. Consider fine-tuning ai_v7_02 (fast) vs from-scratch for that.

**2026-07-07 — TSS TEAM CEILING MEASURED: 93.5% — the team is NOT the problem; ai_v7_02 already IS a
top-tier TSS pilot.** CPU-only bridge eval (`tmp/tss_ceiling_eval.py`, mirrors eval_worker's BOT path exactly:
greedy EvalRLPlayer + trained reward, 50 games/bot): frozen ai_v7_02 best_model PILOTING tss_starmie vs the 9
eval bots = **aggregate 93.5% (374/400)**, per-bot 88–98% (aggressive 88 lowest, staller 98, random 100) —
ABOVE ai_v7_02's own-pool ~89.5%. Results: `/tmp/tss_ceiling_eval_results.json`. KILLS the "TSS is
matchup-limited vs the bot roster" hypothesis: the specialist plateau (~14–16%) is 100% a
LEARNABILITY/REGIME failure (93.5 vs 14 = the team's value is skill-conditional; from-scratch can't
bootstrap it — no random-play floor, longest credit chains, critic blind on positional play). IMPLICATIONS:
(1) ai_v7_08's calibrated target = ~93%, not 89%; if it caps ~14% bots-only, single-team-from-scratch is
confirmed pathological. (2) The generalist learned to pilot TSS at 93.5% WITHOUT ever being pinned to it —
diverse training built the skill that transfers INTO the team; so the fastest route to the user's "model
that plays one team well" is FINE-TUNE ai_v7_02 (already a 93.5% TSS pilot vs bots) with ~50% TSS bias +
self-play, aiming to exceed it vs STRONG opponents (bots are saturated as a specialist metric).

**MIRROR anchor (same day): TSS vs the diverse pool, ai_v7_02 GREEDY BOTH SIDES = 55.0% (110/200, ±7)**
(`tmp/tss_mirror_eval.py`, bridge, 226s CPU; `/tmp/tss_mirror_eval_results.json`). USER'S FRAMING (correct,
adopted): the team's excellence is the PRIOR/measuring stick, not the hypothesis — an excellent team floors
at ~50% vs the field, so ANY substantial shortfall measures the LEARNING SYSTEM, never the team. 55% confirms
it (at/above field-average; and it's a LOWER bound — the pilot is measurably style-biased on balance/defensive
play, so true team value ≥55%). THE TWO ANCHORS: vs bots 93.5%, vs strong diverse field 55%. ⇒ the specialists'
~1% vs the target = a ~54-point pure LEARNABILITY deficit: the current from-scratch single-team regime cannot
learn to play this TYPE of team meaningfully. A true specialist's success bar = EXCEED 55% vs ai_v7_02
(beat the generalist at its own team); ai_v7_08 (bots-only, target now ~93%) isolates whether
single-team-from-scratch learns at all.
