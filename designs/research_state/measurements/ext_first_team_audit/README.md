# F-LH-13 audit — which banked verdicts leaned on a FIRST-TEAM-ONLY `ext_` eval rate (2026-09-30)

> **VERDICT.** No DECISION changes. The population loop's PRIMARY reads (round 1 Δ −10.00, round 2
> Δ2 −8.25, both NOT DETECTED) never touched the defect: every exploiter target in the archive is
> an unpinned generalist. Both MANIPULATION CHECKS, the "did the loop generalist absorb the
> specialists it was shown" gates that separate branch N+ from branch M, were affected. They were
> re-measured over all five of each specialist's teams, and both still read **ABSORBED**:
> - round 1 (M1): **+9.33 pp [+3.77, +14.81]**, against the banked team-0-only +8.50;
> - round 2 (M2): **+6.67 pp [+2.09, +11.20]**, against the banked +8.67.
>
> Two DESCRIPTORS do NOT survive the all-teams read:
> - round 2's "+17.0 against the new specialist RB" becomes **+5.33 [−2.58, +13.16]** (NOT DETECTED);
> - round 1's "+9.50 vs `ai_v13_13`" becomes **+2.67 [−5.18, +10.47]**.
>
> Absorption is concentrated on arm A in both rounds (+16.0, +12.7). Team 0 is **harder** for the
> generalists than teams 1–4, by 4–11 pp, so every banked absolute "generalist vs specialist"
> `ext_` level reads low. Owner decision 09-25 (no round 3 on the old lineage; the loop carries
> into the new lineage) is **unchanged**.

## 1. The defect, and its window

From `b13b30b2` (2026-07-24, the multi-team fold-back fix) until `40b37a34`
(the F-LH-13 BOUNDARY ledger entry, which landed while this audit ran), the live Python eval behaved as follows:
- **What was built.** Both callbacks built a FIXED opponent's `EvalItem` with
  `team_str=f.get("team_str")` and never `team_strs`. The call sites are `eval_callback.py`
  ~L1570 and `selfplay_callback.py` ~L431.
- **What was played.** `main.eval_worker._fixed_opponent_tb` honours `team_strs` but never
  received it. A multi-team specialist was therefore EVAL-measured piloting its **pin index 0
  only**, while in TRAINING it sampled among all its pinned teams.
- **What was recorded.** `eval_manifest.json` `opponent_pins` recorded that one team (since `40b37a34` it records every pinned team).

Before `b13b30b2` a multi-team opponent piloted the full POOL in both training and eval, which is
the older defect that commit fixed. Every affected run launched at or after it.

## 2. Exposure, re-derived (`exposure/exposure.py`, `exposure/exposure_rows.json`)

`read_recorded_trainee_teams` was run over every `--stable-opponents` / `--exploiter` target of
every run in `models/`. 130 runs have a target. **20 have a multi-team target, all through
`--stable-opponents`.** That matches the brief's list exactly. **No `--exploiter` target anywhere
in the archive recorded a pin**: every exploiter's target is a generalist, including
`ai_v14_09_r0_offense_a`. So `main.best_response_gap`'s series, which is each exploiter run's own
vs-target cell (the exploiter's multi-team TRAINEE pin was always honoured), is unaffected across
the whole archive.

**Which team was actually measured** (the inventory sub-audit's `inventory/first_team_map.py` and
`raw_sha_map.py`, confirmed against each traced game's `input_log`). Every `ext_` trace used
exactly one team, pin index 0:

| Opponent (teams) | Team measured |
|---|---|
| `ai_v8_09_pool10_exploiter_0723` (10) | `564b9be3ae` (semi-stall) |
| `ai_v8_06_semistall_3team_exploiter_0722` (3) | **the same `564b9be3ae`**: two "different" teachers, one team |
| `ai_v8_13_defensive10_exploiter_0725` (10) | `9278913bce` (stall) |
| `ai_v9_31_tock1_k4_0824`, the 4-team hyper-offense teacher (4) | `fffd943e9e` |
| `ai_v9_32_tock1b_rain_0824`, the 3-team rain teacher (3) | `01cb64e16c` |
| `ai_v13_13` / `ai_v13_18` / `ai_v13_24`, the three 5-team offense specialists (5 each) | **all three `data/teams/sample/9eb3abdc52876a63.txt`**: Tyranitar · Skarmory · Celebi · Starmie · Salamence · Metagross |

⚠️ One team, **THREE fingerprints**:
- `eval_manifest.json`'s `_sha` hashes the text-mode, unstripped export (`c5676d264e`);
- `team_archetypes.team_sha` hashes the LF-normalized, stripped text (`f36747ae7e`);
- the untaught meter's TeamSlice hashes the raw bytes, `pin_sha` `6212de2e8c`, with a
  CRLF-stripped `team_sha` of `436b8a807b`.

Strip AND normalize before joining any two of them.

## 3. The affected-verdict inventory

Classes:
- **UNAFFECTED:** the claim did not use the series, or the opponent was single-team or unpinned.
- **AFFECTED, BOUNDED:** the direction or size can be argued.
- **AFFECTED, UNKNOWN.**
- **RE-MEASURED:** section 4.

| Run(s) | Claim | Pointer | Metric | Class | Decision-bearing? |
|---|---|---|---|---|---|
| RB/RC, RB2/RC2, RB+/RC+ (the population loop's readers), A/A2, era-1/era-2 exploiters | loop PRIMARY Δ −10.00 [−16.60, −3.27] (r1), Δ2 −8.25 [−15.01, −1.38] (r2), the convergence side-check, the 09-22 archive read | ledger L21329, L21387, L21399, L20729 | exploiter's vs-target `ext_` cell; the target is an unpinned generalist | **UNAFFECTED** | yes: branch N+, count 2 of 3 |
| `ai_v13_22` B / `ai_v13_23` C (round 1: loop generalist / no-exploiter control) | **M1 = +8.50 [+1.76, +15.13] ABSORBED** | L21329; `population_loop_r1_2026-09-23/read/manipulation_check.json` | B − C over `ext_` vs A and `ai_v13_13`, team 0 only | AFFECTED (matched-arm contrast on one of 5 teams) → **RE-MEASURED: HOLDS** | **yes**: ABSORBED separates branch N+ from branch M ("the loop did not engage") |
| same | per-specialist descriptors: vs A +7.50, vs `ai_v13_13` +9.50 | r1 read §1 | same | AFFECTED → **RE-MEASURED**: A **+16.00 [+8.15, +23.57]**, `ai_v13_13` **+2.67 [−5.18, +10.47]** | no |
| same | "G0's own rate 0.360 / 0.3425; C sits at them" | r1 read §1 | G0's rate is from the exploiters' series (all 5 teams); C's is team 0 | AFFECTED, BOUNDED: a cross-population comparison; team 0 reads ~11 pp low for C1 (§4) | no |
| `ai_v13_27` B2 / `ai_v13_28` C2 (round 2: loop / control) | **M2 = +8.67 [+3.06, +14.19] ABSORBED** | L21387; `population_loop_r2_2026-09-24/read/manipulation_check.json` | B2 − C2 over `ext_` vs A, `ai_v13_13`, RB; team 0 | AFFECTED → **RE-MEASURED: HOLDS** | **yes** (same gate) |
| same | **"+17.0 [+7.3, +26.3] against the new specialist RB"** | L21387; UNDERSTANDING §3 | same, the RB cell | AFFECTED → **RE-MEASURED: NOT DETECTED, +5.33 [−2.58, +13.16]** | no, but it is quoted as evidence the loop absorbs NEW specialists |
| B, B2 | training mix among specialists (stable-opponent PFSP ON at share 0.40; weight `max(0.05, 1 − EMA(ext_wr))`) | `selfplay_callback._push_stable_mastered`; `wrappers._pick_stable` | team-0 `ext_` rates steered which specialist B trained against | AFFECTED, BOUNDED: the total specialist share is fixed at 0.40; B's rates vs A and `ai_v13_13` were near-equal (≤ 0.03 apart, weights ≈ 1:1); in B2, RB's higher team-0 rate (0.53–0.59) down-weighted RB about 0.44 : 0.57. Mastery never fired (bar 1.01). C/C2 at share 0.0 are untouched | no |
| `ai_v8_14_distill3_0725` | D1 ✅ "head-to-head vs the teachers 0.228 → 0.36" | ledger L59; memory `project_multiteam_distill_payoff.md` | mean of the three `ext_` rates | rise: AFFECTED, BOUNDED (a within-run trend on fixed teams, but only TWO distinct teams, one counted twice, and the 0.228 start is the fold's first eval, not the parent); the absolute "0.36 ≈ 55 % of the way to parity": AFFECTED, UNKNOWN | weakly: one of three legs of D1; the other two are separate-games meters (ELO 1986 → 2055, piloting 0.438 → 0.710), so D1 cannot flip. **Not re-measured** (FINDING 4) |
| `ai_v8_14` | the +69 audit's "piloting SOLID … + head-to-head 0.228 → 0.36" | L5477–5478 | same series, as corroboration | AFFECTED, BOUNDED | weakly |
| `ai_v8_14`, `v8rep_p1_{A,B,C}`, `v8rep_p2loss_{A,B,C}` (the v8 replications, loss on / loss off) | gift reproduces (+4.56 pp); loss not the carrier (+4.92 pp); v8 gift = transient hump; ELO | L10379ff, L10976–11060, L10729 | the era untaught meter; direct games; bot-anchored ELO | UNAFFECTED | yes |
| `ai_v9_34` (tick-1, the first flywheel fold) | graded INFERIOR: ladder −97.8, piloting −4.0 pp, self-exploitability +11.8 pp | L4437 | `ladder.json`; piloting and admission games | UNAFFECTED | yes |
| `ai_v9_34` | mechanism = representation-rank collapse (capacity battery) | L4437 | capacity battery on the run's own eval traces; 16.5 % of states from first-team-only `ext_` games | AFFECTED (state mix only), BOUNDED: fdB (no stable opponents) collapsed too, and fdC (same exposure) did not | yes (the mechanism) |
| `ai_v9_37` (tick-1 dose extension) | R1 dose read | `measurements/r1_dose_read_ai_v9_37.json` | critic-calibration strata labelled `ext_…` | AFFECTED (a stratum label means "team 0 only"), BOUNDED | no |
| `ai_v9_38` fdA / `ai_v9_40` fdC (the factorial arms: coefficient 0.3 with stable opponents / 0.0 with) | loss convicted, ecology exonerated | L4526 | per-team piloting on 9 teams; capacity; retention | UNAFFECTED (capacity as above) | yes |
| `ai_v13_17`–`_20` (the K-ladder folds, top-1/3/11 target widths + the share-matched K11) | wider targets cost more; share-match 61 % | L21163, L21219 | untaught 8; offense-slice games; `grad/distill_share` | UNAFFECTED | yes |
| `ai_v13_33_core_burnin` (the Rust-core burn-in) | 0 refusals, memory flat, bots 93.0 % | L21395ff | ops meters, bots | UNAFFECTED (stable share 0.0) | yes (GO) |

Readers in `src/` that consume `ext_` keys:
- display only: `eval/win_rate_vs_external`, `eval/elo_vs_ext_*`, the launcher TUI;
- excluded: `main.policy_spectrum`;
- no `ext_` read at all: `main.exploitability`, `main.scaffolding_gauge`, `main.tb_curate`,
  `main.ops.g7_report`.

## 4. The re-measure — the two manipulation checks over ALL teams

Pre-registered in [`PREREG.md`](PREREG.md), in its own commit (author time 2026-09-30 14:20:49 PT) before any battle — the first ran at 14:25; its sha256
`fd92291a…` is stamped on every row. Setup:
- **Pilot:** the specialist, on each of its 5 pinned teams.
- **Opponent:** the generalist, on the pool.
- **Regime:** greedy vs greedy, the `ext_` series' own.
- **Tree:** the arms' own pin `6eb9c776`, with its own release `sim_bridge`, CPU only, nice 19.
- **Size:** 60 battles per (generalist, specialist, team), with B and C seeded identically. That
  is 3,000 battles, 0 timeouts, ~20 min on 6 workers.
- **Rows:** `remeasure/rows.jsonl.gz`. Aggregate: `remeasure/result.json`. Driver:
  `remeasure/driver.py`, run from `~/gen3ai_archive/ext_audit/remeasure/` with PREREG.md beside it.

| Read | Banked (team 0 only) | **All 5 teams** | Verdict |
|---|---|---|---|
| **M1** (round 1, B1 − C1, snapshot 100,000,032) | +8.50 [+1.76, +15.13], n 400/arm | **+9.33 [+3.77, +14.81]**, n 600/arm (0.455 vs 0.362); cluster bootstrap [+2.67, +15.67] | **HOLDS: ABSORBED** |
| **M2** (round 2, B2 − C2, snapshot 110,000,016) | +8.67 [+3.06, +14.19], n 600/arm | **+6.67 [+2.09, +11.20]**, n 900/arm (0.462 vs 0.396); cluster bootstrap [+2.00, +11.44] | **HOLDS: ABSORBED** |
| r1 vs A (arm A, the 1.78×-dose offense exploiter of G0) | +7.50 [−2.03, +16.85] | **+16.00 [+8.15, +23.57]** | larger |
| r1 vs `ai_v13_13` (the first offense exploiter of G0) | +9.50 [+0.00, +18.76] | **+2.67 [−5.18, +10.47]** | NOT DETECTED |
| r2 vs A | +3.50 [−6.10, +13.01] | **+12.67 [+4.72, +20.39]** | larger |
| r2 vs `ai_v13_13` | +5.50 [−4.17, +15.02] | **+2.00 [−5.84, +9.81]** | not detected either way |
| **r2 vs RB** (round 1's reader of B, the NEW specialist) | **+17.0 [+7.3, +26.3]** | **+5.33 [−2.58, +13.16]** | **NOT DETECTED at the all-teams level** |

**Team 0 is not representative.** It is harder for every generalist. Each generalist's rate on
team 0 minus its rate on teams 1–4:

| Generalist | team 0 − teams 1–4 |
|---|---|
| B1 | −4.79 [−14.33, +5.18] |
| C1 | **−10.83 [−19.28, −1.25]** |
| B2 | **−9.86 [−17.57, −1.73]** |
| C2 | −4.31 [−11.89, +3.75] |

So a banked absolute "generalist vs specialist" `ext_` level understates the rate over the full
team set by ~5–10 pp. The per-team B − C contrast also varies. In round 1 it ranged +18.33
(team 2, `0026abf84d`) to −5.00 (team 4, `d252e25433`); team 0 was near the top (+14.17). In
round 2 team 0 was near the bottom (+2.22). The single-team M therefore did not have a fixed
sign of bias.

**Reproduction check** (the re-measured team-0 cell against the banked same-cycle row;
consistency, not replay, because of deviations D1, D2 and D4):
- B1: −2.33 [−13.24, +8.84], PASS.
- C1: −10.50 [−20.47, +0.24], PASS.
- C2: −0.22, PASS.
- **B2: −10.33 [−19.16, −1.15], FAIL at nominal 95 %.** It does not fail under a Bonferroni
  correction for the four checks, and 3 of 4 re-measured cells read LOW. That is consistent with
  D2: the live eval's generalist drew with a 10 % tilt toward the curated sample teams, and the
  re-measure drew from the flat pool. A LOW B2 makes the re-measured M2 CONSERVATIVE, so ABSORBED
  holds a fortiori. It is still reported, not waved away (FINDING 3).

## 5. What this changes

- **Branch assignments: unchanged.** Round 1 is N+ and round 2 is N+ FINAL, 2 of 3 toward the
  stopping rule. Primary unaffected, and both manipulation gates hold over all teams.
- **The owner's 09-25 decision** (NO round 3 on the old lineage; the loop carries into the new
  lineage with `--eval-battles 200` readers and a pooled multi-round read): **unchanged**. It
  rested on power, and power is a property of the primary reads, which are unaffected.
- **Claim standing that changed** (UNDERSTANDING §3, the population-loop bullet):
  - "round 2 +17.0 against the new specialist" is withdrawn → +5.33 [−2.58, +13.16], NOT DETECTED;
  - the absorption numbers become the all-teams M1 and M2;
  - absorption is concentrated on arm A.
- **For the NEW lineage's loop:** its manipulation check reads `ext_` series, so it must run on a
  post-`40b37a34` eval (or the Rust eval core, which plays all pinned teams). Any `ext_` number
  from the old window is not comparable with one after it (the 2026-09-30 BOUNDARY ledger entry).

## 6. FINDINGS

1. **Training-side exposure, beyond the eval number.** Stable-opponent PFSP and the mastery push
   read the `ext_` rate. So in every affected run with `--stable-opponent-pfsp` and a nonzero
   share, WHICH specialist got trained against was steered by team-0 rates. Those runs are
   `ai_v8_14`, the six v8rep arms, tick-1, dosext, fdA, fdC, B and B2. The steering is bounded:
   weights spanned at most ~1.8 : 1 (v8/v9) and ~1.3 : 1 (B2); the total specialist share was
   fixed; mastery never fired. The v8rep arms reproduce v8_14's steering faithfully (same pin),
   so the replication comparison is not confounded.
2. The capacity battery samples each run's own eval traces, so its "ext" states are team-0-only
   (16.5 % of tick-1's). Bounded, as argued in §3.
3. The B2 reproduction check failed at nominal 95 % (§4). Its direction makes M2 conservative.
   It is not resolved.
4. **Not re-measured: D1's head-to-head leg** (`ai_v8_14`). It is not decision-bearing, and it
   needs the era checkout at `b13b30b2` (obs 2992). Recipe if wanted:
   - `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` vs
     `ai_v8_14_distill3_0725/final_model_interrupted.zip`;
   - against the three teachers' `best_model/best_model.zip` on all 23 pinned teams (22 unique);
   - ~40 games per team per checkpoint, greedy vs greedy, teams and seeds paired.
5. Not verified from code: the tick-era `probes/arm_piloting.py` and `tmp/pool10_perteam_eval.py`
   no longer exist. They are classed as separate-games meters from the ledger's own description.
6. Memory notes are not edited here. The orchestrator may want to update
   `project_multiteam_distill_payoff.md` (D1's "0.228 → 0.36" is two distinct teams) and any note
   quoting "+17.0 vs RB".
