---
name: project_incoming_damage_outcome
description: "incoming-damage/OHKO belief feature — first-run forensic outcome (run_20260606_204351 @44M): wired-in + critic-read + calibrated, but policy under-switches → next lever is reward/training, not obs. v2 (2026-06-07) recalibrated the obs tail (crit+tail+coverage); under-switch still the bottleneck"
metadata: 
  node_type: memory
  type: project
  originSessionId: 16b80e1f-9743-4946-884d-4f13f0a66e57
---

> **Archived 2026-09-08** — ai_v5-era obs-block forensics; era closed. Preserved verbatim; nothing below is current.

First run WITH the incoming-damage / OHKO belief obs block (`gen3_incoming_damage_v1`, obs 3390):
`run_20260606_204351`, probed at 44M steps (win_rate_vs_bots 0.759, ELO 1815±22 — both artifact-verified).

**The feature works (3 independent confirmations):** (1) win-rate ~+8pts above baseline at equal
steps — but that magnitude is from the user's TB chart, NOT artifact-reproducible (no baseline
traces on disk); confirm via ablation (the belief-toggle flags exist — [[project_belief_toggle_flags]]).
(2) The CRITIC reads it: value-gradient saliency on `incoming_damage(33)` ≈ 5.1× the per-dim overall
mean (vs 0.30× for the old `their_matchups(144)` it supersedes); survives a density-artifact attack
(dominates per-nonzero-dim too). (3) Calibrated: across 733 direct-hit deaths of our active, median
`active_pko` 0.51 (0.0 at non-deaths), AUC ~0.79.

**But the bottleneck migrated obs→POLICY (the headline).** The policy switch-probability *inverts*
vs the belief: 37% switch-mass when P(KO)<0.2 → 18% (median 7%) when P(KO)≥0.8. Among 589 decisions
with P(KO)≥0.6 AND a legal switch, it stayed in and our mon died ~252× (82%); after a bench-HP
cross-check excluding good-trades/forced/no-safe-pivot (~39%), **~200 are genuine under-switch
mistakes**, threshold-stable, across all opponents. Correction to a too-hasty earlier read: it is NOT
flat — under-switch improves ~10-15pt over 36M (≈92% early → ≈80% late) but PLATEAUS high, far too
slow/high to rely on training alone. The inversion survives early-game/≥2-legal-switch controls.

**Calibration tail (17% of direct deaths read P(KO)<0.25):** reconciled — 12% near-OHKOs priced for
heavy chip (exp≥0.5) but P(KO) too timid (85th-pct stat tail + per-hit misses high-Atk/crit/roll
OHKOs), 7% candidate-absent (unrevealed SE coverage below the 0.12 prior floor / variable-power
Return / Hidden Power), **81% moderate ~30%-exp 2-3HKOs where per-hit P(KO) is CORRECT and the mon
died cumulatively** — a per-hit-vs-cumulative-HP limitation, arguably more a policy-reasoning gap.

**Next levers (ranked):** 1) [reward/training] target the policy under-switch — obs is filled +
critic reads it, so more obs won't teach a policy that ignores a correct salient signal; shape against
staying-and-fainting when an alive low-pko pivot exists. 2) [obs, cheap] make P(KO) less timid on the
near-OHKO tail (higher stat quantile / max-Atk+crit). 3) [obs] widen candidates (HP@60BP, per-channel
SE floor). 4) [validation] ablate via belief-toggle to causally confirm the win-rate lift.

**UPDATE 2026-06-07 — levers #2 + #3 DONE as `gen3_incoming_damage_v2`** (ARCH bumped; same 33 dims/obs
3390, values only → retrain-class; golden regenerated; obs-build bench +6.6% calls/encode, under the
10% gate). FIX 1 (P(KO) timidity): gen3 crit term `_CRIT_P`=1/16 (×2, screen-ignoring) folded into
`p_ko` + offensive-stat tail quantile 0.85→0.95 (KO rides the tail, expected-chip re-normalises to the
MEAN so it's unchanged — the clean lever). FIX 2 (silent zeros): revealed bare `hiddenpower` expands
into per-type candidates (~70 BP, typed from the HP tracker's narrowed dist / Smogon HP prior — tracker
now threaded into `encode_block`); Return/Frustration priced at 102 BP; prior floor/cap widened
0.12→0.05, 4→6 (per-defender max over p_in_set·P(KO) is the type-eff gate). Measured A/B on identical
fresh bridge states (vs aggressive bot, reproduces the 17% baseline tail): **frac of direct-hit deaths
reading P(KO)<0.05 ≈ halved at HP≥80 (25%→13%)**, <0.25 tail 17→15%, mean P(KO)-at-death up every HP
bucket; common case NOT over-inflated (mean +0.022 ≈ the crit mass). The 0.25 tail moves only modestly
because crit/tail correctly lift true-OHKOs from ~0 to ~0.1 (calibrated belief, not worst-case calc) —
the <0.05 floor is the right metric. **Lever #1 (policy under-switch, reward/training) is STILL the
main bottleneck** — v2 fills the obs better but won't move the plateau on its own. Files:
incoming_damage.py (crit), incoming_damage_encoder.py (tail/floor/cap/Return/HP/tracker). NOT yet run.
Continues [[project_loss_analysis_run20260601]] (the pre-feature under-switch/tail-blindness pathology);
relates to [[project_outgoing_damage_design]] and [[project_popart]].

**Prober was extended (repeatable):** `decode_incoming_belief` + `value_saliency` (critic-grad) +
`scan` now reports per-cliff `incoming_active_pko`/`max_pko`/`active_outspeed`; CLI+TUI parity, tests.
(Uncommitted in worktree xenodochial-varahamihira-6a16d0 as of 2026-06-07.)

**UPDATE 2026-06-07 — v2 RUN OUTCOME, adversarially verified (`run_20260607_102632`, fresh, ~28M, arch
gen3_incoming_damage_v2 + belief-PBRS switch shaping 7483dd1 + self-KO fix 4c82bbd + vf_coef 0.5→0.05 +
vf-clip off; 15-agent workflow with a refute-phase + V1 control).** The OBS half is the only real win,
and it is NOT new to this run (V1 had it). Verified: belief is real damage (gen3 formula + 16-roll
P(KO) + crit + 0.95 stat tail; reaches policy+value via non_matchup_rest + global token), critic-read
(value-saliency ~5–6× per-dim mean), calibration IMPROVED (active_pko separates stay-and-die 0.63 vs
survive 0.16, AUC **0.86** vs V1 0.84). **The NEW-specific fixes did NOT help (REFUTED):**
(1) **belief-PBRS switch shaping did NOT fix under-switching** — over 18.5k decoded decisions switch-mass
STILL inverts at the danger endpoint (pko<0.2→0.218, mid-peak 0.362, pko≥0.8→**0.191**); stay-and-die
among pko≥0.6+legal-switch **61.3% NEW vs 61.1% V1** (indistinguishable). The "82%→48.8% improvement"
was a METHOD MISMATCH (48.8% strict-same-turn == V1's 49.3% same method; the 82% was never reproduced
consistently). Shaping terms DO fire → "fired but didn't fix." Likely cause: **PBRS is potential-based →
telescopes to ~0 net → policy-invariant by construction**, so it can't overpower a converged wrong
preference; the memo's intended *bias* (non-telescoping) penalty was implemented as PBRS instead. Next
lever: a BIAS/Markovian stay-and-die penalty (see [[project_markovian_reward_design]]) or much larger
weight, not more PBRS. (2) **self-KO reward fix**: logic correct (faint_ours=-3.25, no finishing_blow on
self-faint, 33/33 tests) but BEHAVIOR frequency ~unchanged (1.66% NEW vs 1.93% PRIOR) — and 4c82bbd was
already in PRIOR, not a differentiator. (3) **vf_coef 0.05**: critic NOT under-fit (EV 0.81, value_loss
90 == V1/PRIOR at 10× the coef) BUT value STILL swamps the shared trunk (value_share 0.975,
value_policy_logratio 1.56 ≈ **36× value/policy grad**) — scale-driven → **PopArt is the actual fix**
([[project_popart]]). **Cross-run control (key method point): V1 (`run_20260606_204351`, has OHKO obs,
lacks NEW fixes) is step-matched-INDISTINGUISHABLE from NEW** (28M: 0.720 vs 0.706 win-vs-bots, ELO 1763
vs 1760) and V1 EXCEEDS NEW's plateau at 48M (0.805 / ELO 1844) → NEW's per-step gain over the pre-OHKO
PRIOR is attributable to the **obs block (in V1 too)**, not the new fixes. **Growing td_resid_tails
(−9.5→−14.5) = benign VALUE-SCALE artifact** (CONFIRMED): scale-normalized |tail|/value_pred_std FELL
22%, held flat ~0.72; EV healthy; critic slightly better per-unit-scale. **Bottom line: the #1
bottleneck (policy under-switch) is unchanged — obs is filled+calibrated+critic-read, so the lever is
reward-MAGNITUDE / a bias term / training-side, NOT more obs and NOT PBRS.**

**UPDATE 2026-06-07 — the bias lever is BUILT** (worktree claude/sweet-keller-51cf69 off main `adc0fe4`,
NOT shipped/run). `--switch-bias-weight` (default 0.0=OFF, reward-only → ARCH unchanged
gen3_incoming_damage_v2, `MODEL_CONFIG_VERSION 4→5`, resume-immutable via check_reward_config). Two
BIAS-class terms on the Step-5 registry: `stay_risk_tax=max(−w·risk,−2.0)` (tax STAYING into high
P(KO)·(1−outspeed) when a safe pivot exists) + `escape_risk_bonus=w·0.5·risk` (escape-to-safety,
asymmetric). Red-team-hardened gates: never tax a TRAPPED stay (`_cur_can_switch` from decision-time
`ctx.mask`), an RNG fizzle (`our_failed_to_move`), a KO'ing stay (`opp_fainted`), or a forced stay
(`_prev_safe_pivot`: a non-fainted bench mon with raw P(KO)≤0.35); escape needs the safe pivot too
(kills the rotation farm). BIAS-class → rides `--bias-additivity`, so weight fixed @ λ=1 vs λ=0 is the
causal A/B. Kept `pbrs_belief` (free credit aid). Verified: 1069 unit tests + 16 new TestSwitchBias +
bridge smoke (round-trip cfg-v5 PASS) + 1-agent red-team (SHIP-WITH-FIXES → all applied). Docs:
`design_reward_switching.md §7`, `impl_step6_switch_bias_lever.md`, training/+model CLAUDE.md. Recommended
run: fresh `--switch-bias-weight 1.5 --bias-additivity 1.0`; success = P(KO)≥0.8 switch-mass no longer
below the low bin + stay-and-die@(pko≥0.6,legal) ≪ 61%, no pivot-spam. NOT yet run.

**UPDATE 2026-06-08 — threat-response REFRAME from the `ai_v5_5_popart_N_0607` review (@32–36M; PopArt
run, switch-bias OFF; multi-agent workflow + adversarial verify; deterministic scripts in /tmp).** Two
corrections to the picture above:
- **Under-switching IMPROVED in volume + the inversion is GONE.** Switch-rate rose to ~22–27% (vs the
  predecessor `ai_v5_4_pbrs_opp_threat`'s ~14% at matched 34M; ~2×). Unconditioned corr(switch-mass,pko)
  ≈+0.03 looks flat, but that's a CONFOUND — pko is conditional on the opp getting to hit. Conditioned on
  **slower** (active_outspeed<0.5, threat is real): corr **+0.12**, switch-rate climbs monotonically
  **22%→38%** as pko rises (right direction, magnitude still too small). Conditioned on **faster**:
  correctly flat ~20% (the policy correctly ignores a non-firing threat). So the prior "switch-mass
  inverts" is no longer true here — it's weak-but-correctly-signed.
- **The BIGGER failure is the belief UNDER-READING a surprise OHKO (NEW, majority).** Across all
  healthy-mon (HP≥80%) single-turn OHKO deaths, the belief **fired (pko≥0.7) only ~10–20%**, but
  **under-read (pko<0.3) ~53–61%** (vs aggressive_v2 53%, aggressive 61%, heuristic2 58%, sentinels ~55%).
  Cause: the OHKO comes from a **just-switched-in / unrevealed** attacker, and the belief prices only
  REVEALED moves (the `their_matchups`/incoming block is ~0 for an unrevealed moveset). So "policy ignores
  a FIRED belief" (the headline above) is the MINORITY (~10–20%); the majority is "belief never warned."
  → the fix for the majority is obs COVERAGE of unrevealed/surprise threats (species-prior or
  learnable-from-embedding), NOT a reward term.
- **`doomed_stay_tax` design (workflow-produced, NOT built)** targets only the minority (belief-fired,
  policy-ignored). A narrow BIAS term: gate = slower (outspeed<0.5) ∧ active_pko≥0.85 ∧ a safe pivot
  exists (min non-fainted bench pko ≤0.35); tax = max(−w·(active_pko−min_bench_pko), −3.5), keyed on the
  decision-time belief snapshot (reuses `_belief_potential_and_risk` / `_prev_safe_pivot` already in
  reward_manager). **BIAS not PBRS deliberately** — PBRS (`pbrs_belief`) was already falsified
  (run_20260607_102632, telescopes→policy-invariant). Distinct from the existing blanket
  `--switch-bias-weight` lever (conflated risk·(1−outspeed), fires across the whole band → would
  over-tax the already-correct moderate regime). Forensic detail: the genuine clean blunders are the
  slow-frail-vs-revenge-killer cases (Jolteon/Salamence vs TTar at full HP, switch-mass 0.05–0.17, died).
  See [[project_anti_stall_fix]] (shipped this session) and [[project_markovian_reward_design]].
