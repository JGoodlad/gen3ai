# REVIEW G: data quality and selection bias in the X59-S read (2026-10-27)

**What it reviews:** [`PREREG.md`](PREREG.md) with Amendments 1, 2, 2b, 3, 4, 4b and 4c, and FR-2's registered
record (`~/gen3ai_archive/x59s/read/fr2/fr2.json`, class NO-ARTEFACT). It also reads the floor references
(`read/floor_refs.json`), the students' sidecars (`read/students/students.json`) and the rows and manifests
under `~/gen3ai_archive/x59s/`.

**Context:**
- [`human_data_architecture.md`](../../../endstate/human_data_architecture.md);
- [`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md) §2.2 (FR-2) and §6.2 (which
  panels are clean);
- the replay archive's selection-bias check ([`replay_archive_bias_2026-10-21`](../replay_archive_bias_2026-10-21/README.md));
- [REVIEW E](REVIEW_E.md) and [REVIEW F](../human_replay_quality_2026-10-25/REVIEW_F_x60.md) (`e41beafe`).

**The question:** what besides data-source quality could Δ_data (S-H − S-S) measure? This is the data and
selection-bias review, one of the two independent adversarial reviews the human-data rule requires. The
experimental and causal review is separate.

**Method:**
- CPU only, at nice 19 on cores 5-7,13-15, every job under `mem_cap.sh` in the short lane.
- Read-only on the archive, the corpus, `models/` and the students. No GPU, no game, no training.
- No pilot or Kakuna unit result was opened. The sealed `forum216b` was not read: the panel checks use H, F, P72 and
  the retired heldout72 only.
- One script sits beside this file, [`review_g_checks.py`](review_g_checks.py). Its output is
  [`results/review_g_checks.json`](results/review_g_checks.json). The PREREG and its amendments were not edited.

**Abbreviations:**
- P\* = X26's plateau checkpoint. H / F / P72 = `heldout72_hpfix` / `forum72` / `p72`; U = their equal-weight mean.
- C-GAME = the completion of a seat's hidden facts from the Smogon prior.
- "Consistent set" = the legal actions the whole log leaves possible for the actor.
- "Golden" = the sha256 of `golden_obs_fixture.json`, which moves with every deliberate encoder change.
- pp = percentage points; nats = natural-log units of NLL (negative log-likelihood).

## Verdict: READY-WITH-FIXES. No BLOCKER; four MAJORs, two of them already answered by Amendment 5. G-M1 and G-M4 need an amendment before any contrast is computed

The core matching holds. The arms have equal rows, steps (7,041) and epochs (3). One `init_sha256` is used per
seed, and S-H and S-P see the same batches. S-S's rows are P\*'s own labels (sha `814823c5…`). The account split
cannot be dodged by name variants. Team-level panel leakage is small. FR-2's exclusion was decided blind, and it
could not have changed FR-2's class.

What is NOT matched is how the arms' data was made. Four things bias Δ_data, and they pull in opposite directions:
- **Against S-H:**
  - FR-2 cannot see the drawn-move signature it was put there to catch (G-M1).
  - S-H's and S-P's rows were encoded by an older observation encoder than the one S-S trained on and every game
    plays on (G-M2).
  - The panel exclusion leaves S-H a forfeit-heavy, endgame-poor slice that S-S's rule does not reproduce (G-M4).
- **For S-H:** two of U's three panels share the human corpus's population or its fill tables (G-M3).

The read can keep playing. None of these touches a game. Each fix is an amendment or a CPU-only read. It must be
registered before the full record is read; today only the floor boolean is released.

**Status against Amendment 5** (`aaeabbd46`, landed while this review ran; it answers REVIEW H and says REVIEW G is
folded in by a further amendment):
- **G-M2 is ANSWERED by A5.7.** It measured the re-encode share this review proposed: 51 of 8,556 rows (0.60 %)
  differ, all in `context.volatiles`, under its 1.0 % trigger (`results/m3_reencode.json`). This review's 8.3 % is
  an upper bound on rows near a start, not a count. G-M2 stays a registered limit.
- **G-M3 is ANSWERED by A5.4** (POPULATION-SPECIFIC on forum72's point estimate), the same rule this review asks for.
- **G-M4 is PARTLY answered.** A5.6 scopes the closure to "a behaviour-cloning target for a fresh network". It does
  not name the forfeit-heavy, endgame-poor slice, and no length split is registered.
- **G-M1 is OPEN.** No amendment touches FR-2's gate. A5.9 m14's twin gap of S-H_k against S-S_k carries the same
  sign confound.

**The amendment still owed:** G-M1's Δ_restriction gate, G-M4's slice wording and its length split, and the MINORs'
reporting lines.

## 1. Findings at a glance

| id | sev | one line |
|---|---|---|
| G-M1 | MAJOR | FR-2 cannot detect the drawn-move signature. C-GAME completes from the WHOLE game's reveals, so in FR-2's completed rows a drawn option is never the label, and a model that learned the signature gets a SMALLER twin gap. The registered NO-ARTEFACT (D = −0.149) is what that predicts. The closure gate (A2.5) is therefore open, unguarded. |
| G-M2 | MAJOR → answered (A5.7: 0.60 % of rows differ) | Only S-H and S-P train on an encoder different from the play encoder. S-H's and S-P's rows predate `d8d674b99` (golden `9a6f947b`, layout hash `fbc5c674`). S-S's rows and the play pin share golden `95e24c32`. A silently ending volatile occurs in 12.1 % of kept battles (≤ 8.3 % of rows); no VOID guard compares the arms' encoders. |
| G-M3 | MAJOR → answered (A5.4) | U is two-thirds population-contaminated in S-H's favour. H is ladder sides with usage fills (C-GAME's own construction); P72 is drawn from C-GAME's own tables; only F is clean (Q-H13; X60 took F alone for this). Both meters average all three, and KAKUNA-SPECIFIC does not cover it. |
| G-M4 | MAJOR (A5.6 partly) | The panel exclusion selects trajectories (forfeits, few reveals), and S-S's pass-rate-matched rule cannot. S-H rows: 44 a battle against 82; median turn 15 against 19; 7.6 % against 10.3 % at turn ≥ 50; 2.9 % against 6.6 % with no switch legal; 50.6 % from forfeits against 0 %. A SELF-PLAY-DATA-BETTER partly reads "endgame coverage". |
| G-m1 | MINOR | 71 % of S-H's validation rows (21,318 / 29,904; 755 of 1,007 battles) come from battles whose other seat is in TRAIN; S-S has 0. The human-validation yardsticks are optimistic for S-H, S-P and S-P0. No rule reads them. |
| G-m2 | MINOR | The val NLL ordering (S-H 1.18 < S-P0 1.25 < S-S 1.28 < S-P 1.29) is teacher entropy and set labels, not data quality. S-P0 and S-S sit the SAME 0.278 nats above their teacher's entropy. |
| G-m3 | MINOR | S-S's `forced` flag is never set (0.0 against 8.5 %), though 9.3 % of its rows are switch-only. A2.11 m7's profile must read the mask, not the flag. |
| G-m4 | MINOR | The Kakuna-family audit's drop is a "P\* fits badly" selection: it removed the 1.73 % of rows least like P\*, nudging S-H's labels toward P\*. |
| G-m5 | MINOR | S-S's states come from the pre-parity engine (`d02550c0d`; moves 325 / 369, 16 BF classes); S-H's come from real Showdown; play is post-parity. The mechanics drift is asymmetric toward S-H, and tiny. |
| G-m6 | MINOR (A5.9 m13 names the regime) | The pin predates `b0441623e`, so Kakuna's forum72 Gyarados plays Frustration at base power 1 in every X59-S F game. This is equal across arms (twinned), but G-M3's F-only sensitivity reads exactly that regime. |
| G-m7 | MINOR | The interval holds student-seed variance only. Each arm has ONE dataset (S-S: one self-play seed, 5910), so dataset variance is outside it. |
| G-m8 | MINOR | S-S is P\* distillation by design: "self-play data" means the plateau policy's own games and choices, not the RL rollout data that trains P\*. The owner's answer must say so. |

## 2. MAJORs

### G-M1 (MAJOR): FR-2's registered class cannot detect the signature it gates

**The registered claim.** A2.5 says S-H's and S-P's own-side inputs carry C-GAME draws that are never a target.
"FR-2 is read before Δ_data is reported … if FR-2 reads ARTEFACT, a SELF-PLAY-DATA-BETTER closes nothing." FR-2
read **NO-ARTEFACT**:

| model | twin gap = NLL(completed) − NLL(true) | 95 % CI |
|---|---:|---|
| the HOM (trained on C-GAME rows) | +0.069 | [+0.062, +0.076] |
| P\* (trained on true rows) | +0.218 | [+0.200, +0.235] |
| **D = gap(HOM) − gap(P\*)** | **−0.149** | **[−0.163, −0.134]** |

Counts: 2,091 battles; 62,415 paired decisions; 33 battles and 2,860 rows excluded.

**Why the sign cannot detect the signature:**
- **C-GAME conditions on the whole game.** `main.x50_r1.complete` says: "Every own fact the log REVEALS by the
  battle's end is kept … every other fact is drawn". It also redraws until the team explains every labelled action.
  So in FR-2's mode-2 rows, as in the students' rows, a drawn move or bench mon is never what the actor chose.
- **What a learned signature does to the twin gap.** Suppose a model has learned the signature: "options shaped
  like a fill are not chosen". On the completed rows that knowledge is correct, so NLL(completed) falls. On the true
  rows the same suppression lands on real options, so NLL(true) rises. Both effects shrink the HOM's gap.
- **So the signature predicts D < 0.** Plain in-distribution robustness (the HOM trained on completed rows, so they
  suit it) predicts the same sign. ARTEFACT needs D > +0.05, the opposite direction.
- **Where the rule came from.** That direction answers rung 1b's question ("completion costs the HOM more than P\*,
  so HOM-based Kakuna scores are lower bounds", lever doc §2.2). It was carried into A2.5 unchanged. The
  `read_fr2` docstring's "does a model trained on C-GAME rows learn the completion artefact?" is not what its
  class measures.

**What the data does say (descriptive, from this review's `nll` check, on the human validation rows):**
- **The signature is learned, and it is small.**

  | arm | mass outside the consistent set (labelled rows) |
  |---|---:|
  | S-H | 0.142 |
  | S-P | 0.145 |
  | S-P0 | 0.176 |
  | S-S | 0.174 |

  - Two arms' labels never name an option outside the consistent set (S-H, S-P); two arms' labels do (S-P0, S-S).
  - The first two put about 3 pp less probability there.
  - S-S has never seen a completed row, yet it agrees with S-P0. So the gap comes from the labels' restriction, not
    from the human states or the fill.
- **The HOM's own gap is positive.** True inputs still help a completion-trained model on average. That is weak
  reassurance: it is not a bound on a targeted behaviour such as under-using a not-yet-used move or a
  not-yet-revealed bench mon.
- **A bound in NLL terms.** On the S-S validation rows (true inputs, P\*'s labels), S-P0 scores 1.327 against
  S-S's 1.281 (five-seed means, the sidecars). Human states plus the fill in the inputs cost a P\*-taught student
  about 0.05 nats there.

**Fix (an amendment before the verdict is ACTED on; no new GPU):**
1. State that FR-2's class answers the rung-1b question. A NO-ARTEFACT does not certify the absence of the
   signature.
2. Replace the closure gate with the falsifier the design already has: **Δ_restriction = S-P0 − S-P**. The two arms
   share their rows and their teacher, and differ exactly in whether the labels carry the signature. The amended
   gate: a SELF-PLAY-DATA-BETTER closes nothing until Δ_restriction is read on U with §4's interval, and nothing if
   its LB > 0.
3. That moves S-P0's U units (A4.5 step 6, about 20 lane-h) ahead of the closure, not ahead of the verdict.
4. A cheaper descriptive companion, from the read's own game records: per arm, the rate at which a student's chosen
   action is a move it has not used yet this game, or a switch to a mon not yet revealed. Compare it with its own
   training labels' rate. An S-H-only deficit is the signature in play.

### G-M2 (MAJOR): S-H and S-P trained on an older encoder than S-S and the read

**Evidence (manifests and git, verified):**

| rows | made at | golden | layout hash |
|---|---|---|---|
| `hr_spec_v1.r1` → r2 (S-H, S-P, S-P0) | `da36173c9` (10-25 13:52, sources dirty vs commit) | `9a6f947b…` | `fbc5c674` |
| `ss_rows_v2` (S-S) | `d02550c0d` (10-25 22:10) | `95e24c32…` | `b9d868ec` |
| the play pin `0534e90a6` (every game) | | `95e24c32…` | |

- Each layout hash is the sha256 of `{golden, obs_dim 2845, gen3_mon_tied_gain_v1}`; this review recomputed both.
- Between the two builds landed `d8d674b99` (`gen3_volatile_silent_ends_v1`). The reader now clears Endure, Snatch
  and Helping Hand at `|turn|`; Destiny Bond, Grudge and Rage at the holder's next move; and `trapped` when the
  trapper leaves. Before it, the cell stayed at 1.
- REVIEW F verified r2's encoder for X60's side pin (`f4b25238`, the same golden). That is correct there, and it is
  the mismatch here: X59-S plays at `0534e90a6`.

**Size, on the 27,586 kept logs (`logs`):**
- 3,346 battles (12.1 %) start such a volatile: Endure 1,692 lines, Destiny Bond 1,576, `trapped` 1,392, Snatch 42.
- **Upper bound on affected rows:** 102,433 of r2's 1,231,442 rows (8.3 %) sit at or after the first start in
  their battle. The stale cell is one input on a subset of these.
- Pool self-play meets these volatiles far less often: 33 of 2,306 R1 traces (1.4 %), and the golden moved on 36 of
  991 decisions.

**Direction:**
- Only S-H and S-P (and S-P0) meet a train/play encoder shift. This costs Δ_data and Δ_states. Δ_labels is matched.
- Nothing in §7 catches it: VOID 1 compares rows manifests, and VOID 3 compares inits and steps.

**Fix:**
1. **Registered limit (amendment):** a SELF-PLAY-DATA-BETTER cites it as a cost to S-H only.
2. **A cheap measurement (minutes of CPU):** re-extract a seeded sample of 200 kept battles at the play pin
   (the `hom_repro` check's machinery, at the new encoder). Report the share of r2 rows and cells that differ.
3. If more than 2 % of rows differ, record that Δ_data carries an encoder term of unknown sign and size. Do not
   retrain: about 6 GPU-h would change no route.

### G-M3 (MAJOR): two of U's three panels share the human data's population or its fill

**Evidence:**
- The project's own Q-H13 and lever doc §6.2 were written after this PREREG registered (X60 redesign, 10-26):
  - "**H** … its own population and construction (C-GAME completes ladder sides exactly as `hl_05_26` was built)";
  - "**P72** … adjacent: the BC rows' own sides carry the same draws";
  - "**F** … different population".
- X60 made F alone its primary for exactly this reason. X59-S's primary is U = (H + F + P72) / 3, on Kakuna AND on
  the in-house meter.
- **The guard does not reach it.** KAKUNA-SPECIFIC (A2.10) asks only that the in-house meter agree, and the
  in-house meter averages the same three panels.
- **The mechanism is concrete.** At play the student pilots the panel team.
  - On H and P72, that team's sets are usage or Smogon draws, which is what S-H's and S-P's own-side training inputs
    look like (C-GAME fills).
  - On F they are real human sets.
- **Mirror of G-M1.** The fill signature should cost S-H most on F and least on H and P72. So the per-panel split
  separates the two biases.

**Team-level leakage is small (`logs`, passed):**
- Every SEEN mon is counted, not only those with a revealed move as the rowset rule counts.
- Kept actor sides fitting ≥ 4 mons of one scored-panel team: 1,798 of 44,947 seats, 2.55 % of rows. ≥ 5: 171
  seats, 0.20 % of rows. 6 of 6: 14 seats, 0.01 % of rows.
- The opponent sides are alike (2.78 % / 0.25 %).
- A team share of ≥ 4 is mostly shared OU cores, so this is population overlap, not a copied team.

**Fix (an amendment before any contrast is computed):**
1. Report Δ_data and Δ_labels on F alone, with §4's interval, as a registered sensitivity.
2. A HUMAN-DATA-BETTER (or HUMAN-LABELS-BETTER) on U whose F-only point estimate is ≤ 0 (a tie within 1e-9
   counts) is reported as **LADDER-POPULATION-SPECIFIC**. Like KAKUNA-SPECIFIC, it opens no lever beyond X60's own
   F-primary test.
3. F alone carries a third of the games, so its interval is wide. The qualifier is therefore a point-sign rule, not
   a test.

### G-M4 (MAJOR): the panel exclusion selects trajectories, and S-S's rule cannot match that

**Evidence (`profile`, `split`; the TRAIN rows of each arm):**

| | S-H (r2) | S-S |
|---|---:|---:|
| battles / rows a battle | 27,056 / 44.4 | 14,709 / 81.7 |
| turn quantiles 10 / 25 / 50 / 75 / 90 | 3 / 7 / 15 / 26 / 43 | 4 / 10 / 19 / 31 / 50 |
| rows at turn ≥ 30 / ≥ 50 | 20.6 % / 7.6 % | 27.2 % / 10.3 % |
| rows with no legal switch (last mon, trapped, locked) | 2.9 % | 6.6 % |
| rows from forfeit / timer games | 50.6 % / 2.9 % | 0 / 0 |

- The human rule keeps battles whose reveal is SMALL. Kept against excluded: 64.1 % against 49.3 % forfeit or
  timer; 5.14 against 5.66 mons seen; 51.9 % against 74.6 % of sides seen whole.
- A2.1's S-S rule simulates reveals on teams before the game and matches the pass rate (16.8 % each). It cannot
  select on how a game goes, because every S-S game plays out to a KO.
- So S-H holds fewer late-game and last-mon decisions **because of our filter**, not because of what humans teach.
- The spectator corpus is itself forfeit-heavy. The replay archive's check found spectator days at 53–56 %
  forfeits, against 34–46 % in uploaded replays. The exclusion pushes that further.

**Direction.** Against S-H in long games. §8.2 names the slice ("forfeit-heavy"), but A2.7's closure text (1300+
ladder data at this mix and volume) does not.

**Fix:**
1. **Amendment:** the closure text names the slice ("the panel-distant, forfeit-heavy 1300+ slice").
2. **Descriptive, from the read's own records, CPU-free:** each arm's U rate split by game length (turn terciles of
   the twin games). If S-H's deficit against S-S concentrates in the long tercile, the report says that Δ_data is
   partly endgame coverage.

## 3. MINORs

- **G-m1, the validation split leaks at the battle level.**
  - `key64 mod 50` splits by ACCOUNT, so the TRAIN opponent of a VALIDATION account contributes the same battle's
    other seat: 21,318 of 29,904 VALIDATION rows (71 %), in 755 of 1,007 battles. S-S splits by game: 0.
  - The human-validation yardsticks of S-H, S-P and S-P0 are therefore optimistic against S-S's.
  - The only rule on a yardstick (A2.4's H-BC gate) reads the S-S validation rows, which are clean.
  - **Fix:** report the human yardstick on the 252 clean battles too.
- **G-m2, the val NLL ordering is not a data-quality ordering (`nll`).** Each arm's own-label NLL is measured on its
  own rows against its own teacher.

  | arm | own-label NLL | teacher floor | excess over the floor |
  |---|---:|---:|---:|
  | S-S | 1.277 | 0.998 (P\*'s entropy on S-S states) | 0.278 |
  | S-P0 | 1.253 | 0.975 (P\*'s entropy on human states, plus the hidden rows' set-NLL) | **0.278** |
  | S-P | 1.299 | 0.839 | 0.460 (its labels use whole-game look-ahead the input cannot see) |
  | S-H | 1.181 (labelled rows 1.260; the 8.0 % hidden rows 0.266) | unknown | — |

  - S-P0 and S-S learn their teacher equally well. Their order is teacher entropy plus S-P0's set-label rows.
  - S-H's low number is partly its hidden rows: about −0.08 nats.
  - **Fix:** never quote the ordering as evidence. Report each NLL beside its floor.
- **G-m3, S-S's forced-switch flag is never set.** The flag reads 0.0 % in S-S against 8.5 % in S-H. Switch-only
  rows are 9.3 % in S-S and 8.4 % in S-H, so the decisions exist; only the flag is missing. **Fix:** A2.11 m7's
  profile reads the forced share from the mask.
- **G-m4, the audit's drop is a P\*-misfit selection.**
  - S = LL_Kakuna − LL_P\* flagged 11 accounts because P\* fits them badly (Amendment 2b), not because Kakuna fits
    them well.
  - Dropping them removes the 1.73 % of rows least like P\*. That nudges S-H's labels toward P\* and shrinks
    |Δ_labels| a little.
  - **Fix:** a one-line registered limit.
- **G-m5, engine drift is asymmetric toward S-H.**
  - S-S played at `d02550c0d`, before full parity (B1–B11, BF: moves 325 → 369, OU-visible classes).
  - S-H's states are real Showdown, and every game plays post-parity at `0534e90a6`.
  - S-H's training mechanics are nearer the play mechanics. The classes are rare (REVIEW F: most ≤ 0.01 % of fielded
    mons).
  - **Fix:** a registered limit.
- **G-m6, the forum72 regime.** The pin predates `b0441623e`, so Kakuna's `sft431` Gyarados plays Frustration at
  BP 1 in every F game. Twinning makes this equal across arms, so it does not bias Δ. It is the regime of G-M3's
  F-only sensitivity and of the floor references (random 0.000, aggressive 0.046). **Fix:** name it beside any
  F-only number.
- **G-m7, one dataset an arm.** The interval holds student-seed variance only. S-S is one self-play draw (seed 5910,
  15,069 games); S-H is one fixed corpus. **Fix:** say so beside σ̂. A second S-S seed would cost about 4 lane-h of CPU for
  the rows, plus 5 students (about 2 GPU-h).
- **G-m8, S-S is P\* distillation.**
  - It is P\*'s T = 1 policy on P\*-vs-P\* states (manifest: P\* sha `814823c5…`; P\* re-scores its labels at its
    entropy, 0.993 against 0.998).
  - This is the best case for self-play data at this volume: the plateau teacher on its own distribution.
  - **Fix:** the owner's sentence says "our plateau policy's own games and choices".

## 4. What I checked that passed

| check | result |
|---|---|
| **FR-2's exclusion was decided blind** | Yes, by the file times: the exclusion list was written at 03:57:52 and committed at 03:57:57 (`c3ca93787`, `e652875c6`). The first FR-2 forward cache was written at 05:55, `fr2.json` at 06:04. The cache holds ONLY chunks tagged with the list's sha (`x1c6f1ef8`): no unexcluded read exists. The list is a protocol pattern, not an outcome. |
| **The exclusion cannot have moved the class** | It drops 2,860 rows of both modes, about 1,430 decisions (2.2 % of the paired set). For D's upper bound (−0.134) to reach +0.05, those decisions would need a D of about +8 nats each, against NLLs of 1.1–1.6. |
| **Arm matching** | Every student ran 7,041 steps over 3 epochs. S-H, S-P and S-P0 share rows `2a413c85…`; S-S has `3f81f71c…`, with equal TRAIN / VAL counts and shard sizes. There is one `init_sha256` per seed. No arm's validation NLL rose over its last epoch. |
| **The account id** | `toid(name)` comes before the salted hash (`textpass.acct`), so a capitalisation or spacing variant cannot straddle the split. |
| **Dates** | The corpus runs 2026-05-18..09-20; held-out dates are dropped. The Smogon fill window (2025-05..2026-04) precedes it, so the fill is not fitted on these games. |
| **Duplicates and win-trading** | 0 battle ids appear twice. Of 17,361 two-seat battles: no account pair meets ≥ 10 times (maximum 9); 13 pairs meet ≥ 5 times (80 battles); 25 groups share accounts, turns and ending. The 763 one-seat groups `logs` counts are one player's unrelated games. |
| **The S-S labels are P\*'s** | The manifest's P\* sha equals the registered one, and the recorded label's NLL under P\* equals P\*'s entropy (0.993 against 0.998). |
| **The yardstick code** | My set-NLL reproduces the sidecars exactly (S-H 1.18084, S-P 1.35885, S-S 1.41617 on human labels), so G-m2's decomposition is the trainer's own measure. |

## 5. FINDINGS (standing rule 7)

1. **Not run:**
   - G-M2's re-encode sample: the 8.3 % is an upper bound on rows near a start, not a count of differing cells.
   - G-M1's play-time behavioural rate, which needs the games.
   - G-M4's split by length, which needs the record.
   - The residual of the Kakuna audit's power against SmallRL- or LargeIL-like bots (Amendment 2b's own limit).
2. **The `logs` check's ≥ 4-fit count includes mons with no revealed move**, which fit trivially by species. It is an
   upper bound on population overlap, not a team-copy count.
3. **The `nll` student numbers are seed 5901 only** (one student an arm). The sidecars give the other seeds'
   yardsticks, which agree within 0.02.
4. **Every job queued for a heavy slot** (up to 20 min) behind a live training run (`rb_x35_lrfloor_s3502`) and the
   running reads. None ran outside the short lane or off the agent cores.
5. **I did not contact the X59-S operator**, and I did not open any pilot, floor-pilot or Kakuna unit of a student.
