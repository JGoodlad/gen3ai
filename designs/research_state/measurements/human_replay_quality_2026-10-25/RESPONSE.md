# RESPONSE to reviews A, B and C (2026-10-25)

**What this is.** The disposition of every finding of [`REVIEW_A_data.md`](REVIEW_A_data.md) (data / bias / leakage),
[`REVIEW_B_experiments.md`](REVIEW_B_experiments.md) (experiments / causal) and
[`REVIEW_C_reasoning.md`](REVIEW_C_reasoning.md) (reasoning), against the draft of `14ae4203`. Each finding is FIXED,
REJECTED (with the reason), or DEFERRED (it needs something outside this unit: a download, or the owner's word).

**What was revised.**
- The architecture doc: [`../../../endstate/human_data_architecture.md`](../../../endstate/human_data_architecture.md),
  "the arch doc".
- The lever plan: [`../../../endstate/design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md),
  "the plan".
- This card: [`README.md`](README.md), `hrq.py`, `results/results.json`.
- The registry: [`../../human_replay_snapshots.json`](../../human_replay_snapshots.json).
- A new static gate: [`src/human_data_doc_gate_test.py`](../../../../src/human_data_doc_gate_test.py).
- The power script: [`plan_power.py`](plan_power.py), whose output is `results/plan_power.json`.

Both docs are a REVISED DRAFT pending re-review.

**Counts.**

| | BLOCKER + MAJOR | MINOR |
|---|---|---|
| fixed | 22 | 27 |
| fixed in part, with a remainder DEFERRED to an approval | 2 (A-M7, B-M5) | 0 |
| rejected | 0 | 0 |

Parts of four fixes were declined, each with its reason: A-M2's HOM re-fit, B-B1 fix 3 (the λ mix), B-M6 fix 3
(Foul Play) and B-m10's tier-A re-cut.

## Measured in this revision (CPU, on R-b1's cached forwards and the re-hashed snapshots)

| read | result | answers |
|---|---|---|
| HOM − P\* on accounts never seen on the train dates | **−0.503 [−0.542, −0.467] nats** (22.5 % of the ≥ 1500 held-out rows); seen accounts −0.605; the difference +0.102 [+0.060, +0.143] | A-M2: learnability is mostly not account memory (≈ 17 % is) |
| P\* NLL, full-six fits ≤ 2 minus fits ≥ 4 | **+0.064 [−0.016, +0.149]** (the original all-sides cut: +0.024 [−0.043, +0.097]) | A-M1: uninformative, as A said |
| drift beyond an account-disjoint floor | WITHIN 0.047–0.064 by month; EXCESS 0.005–0.011 per consecutive month, positive in 19–20 of 20 splits | A-m1: slow drift is real but ≈ 4–8× smaller than "0.04 a month" implied |
| extraction from the manifests | 376,410 + 16,490 logs re-hashed, 0 mismatches; `results.json` stamped | A-M5 |
| the durable copy of `hr_spec_v1` | 376,410 members re-hashed after writing; 297 MB; sha256 `315a9d1b…f09c` | A-M6 |

## BLOCKERs and MAJORs

| finding | disposition | where |
|---|---|---|
| **A-B1** H-TEAMS's guard recalls ≈ 1 % of copies after the fill | **FIXED.** The exclusion runs on REVEALED facts BEFORE C-GAME, over both seats and every held-out panel including `forum216b`. The exact-set guard stays as a second net. The regression test (a verbatim panel team's log is excluded; it fails on revert) is specified for the rowset build, because no export code exists yet | the plan §3.1, §7 |
| **A-B2** ≥ 4-fit exclusion vs the panels' ≤ 2-fit definition | **FIXED with A's option (a):** ≥ 3 fits, counting only mons with a revealed move. Its cost is stated: it removes 70.7 % of ≥ 1500 battles. The claim that F + P72 are "populations no human lever trains on" is dropped. Option (b) is rejected because B measured its per-team dose-response read as unpowered (7.5 pp MDE) | the plan §3.1, §0; the card §D; the arch doc Q-H6 |
| **A-M1** the familiarity read cannot carry "removes the offline evidence" | **FIXED.** The claim is withdrawn and the read is called "uninformative" everywhere. A full-six re-cut with a joint CI on the difference was added (+0.064 [−0.016, +0.149]). It steers nothing | the card §E; the arch doc fact 4; the plan §2.0; `UNDERSTANDING.md` §4.8 |
| **A-M2** held-out dates share accounts | **FIXED in part.** The account-disjoint re-read was done: −0.503 nats on unseen accounts. Every future split is by a salted account hash. *Declined:* re-fitting the HOM on an account-disjoint validation split. X59 no longer decides anything, so nothing rests on the HOM's early stopping; the re-fit comes with the first rowset | the plan §2.0, §3.3; the card §E |
| **A-M3** hidden choices are missing not at random | **FIXED (design).** Hidden turns keep a set label (−log Σ over legal moves); Pursuit-on-switch windows are excluded. A's corrected switch rates are in the card §B | the plan §3.2; the card §B |
| **A-M4** Kakuna-family play may be in the teacher data | **FIXED (design).** A per-account likelihood audit (Kakuna vs the HOM; flag above a human p99) feeds a registered sensitivity in X59-S and X60. Not run (≈ 1 agent-day + 6 lane-h) | the plan §3.4 |
| **A-M5** nothing makes a result use the snapshot | **FIXED (built).** `hrq.py extract --snapshot` reads the manifest, checks its digest against the registry, re-hashes every log and checks the population header. It REFUSES on any mismatch (tests in `hrq_test.py`). Caches and `results.json` carry `{snapshot_id, manifest_sha256, extraction_commit, hrq_sha256}`. The consumer-side check in `agents.info_levers.human_data` stays PROPOSED | `hrq.py`; the arch doc §5 |
| **A-M6** `hr_spec_v1` exists once | **FIXED.** The durable copy is `~/gen3ai_archive/human_snapshots/hr_spec_v1/logs.tar.zst`, beside its manifest and `ARCHIVE.json`, every member verified and read-only. Both locations are in the registry. The original was only read | the registry; the arch doc §5 |
| **A-M7** the spectator's selection is concrete and unmeasured | **FIXED in the text; the measurement is DEFERRED.** (b) is no longer called "the reference ladder sample"; its known mechanics are in the taxonomy; Q-H4 says what would answer it; A-4's falsifier is quantified. The coverage read (`search.json` ids, Smogon monthly counts) needs pulls the owner has not approved | the arch doc §4, Q-H4; the plan §8 |
| **B-B1** X59's SUPPORTS rewards forgetting; power 0.12 | **FIXED.** X59 is withdrawn as a decision rule, and X59-S, the matched-student contest, replaces it. Its power: P(LB > 0) 0.84 at +4 pp with 2,400 U games per student, 0.025 at 0, and 0.71 jointly with the replication. The HOM's Δ_U survives as a one-sided descriptive read at 1,600 a cell (B's fix 2). *Declined:* fix 3, the λ logit mix. It needs a two-network agent that `main.anchors` is UNVERIFIED to host, and it would only be read after X59-S | the plan §2.4, §2.5 |
| **B-B2** X60 cannot read NULL | **FIXED.** The primary moves to F + P72 (M1) at **630 games per (seed final, panel)**, more than the ≥ 420 asked. That gives NULL 0.58 at Δ = 0 and GAIN 0.77 at +2.5 pp, equal to B's 420-game three-panel frame. At 420 on two panels NULL would be only 0.29. A seed rule applies before look 1, and every final class has an action | the plan §6; `plan_power.py` |
| **B-M1** no clean U panel | **FIXED.** The wording is corrected: F and P72 are less near, not clean. GAIN now needs both F's and P72's point estimates > 0, a heterogeneity guard. A proximity report is added, H becomes a positive control, and `forum216b` carries the clean claim | the plan §3.1, §6; the arch doc Q-H13 |
| **B-M2** the controls do not isolate "human" | **FIXED.** Two contrasts are registered, H-TEAMS − D1 and H-BC − P-BC (the P\*-label placebo, conditional on an H-BC GAIN), with power 0.61 / 0.92 at +2 / +3 pp. A null-dose identity test is added | the plan §6 |
| **B-M3** no positive control, no dose axis | **FIXED.** H is split in halves: H-p (practised) has power 0.90 at +5 pp, and H-u is the transfer read. The 10M checkpoints are read too, and an offline data-scaling curve (25 / 50 / 100 % of the rowset) runs before any GPU. A-8's falsifier now reads on H-p | the plan §6, §8 |
| **B-M4** the ranking | **FIXED.** "The only lever" is struck. The levers are re-ranked: the reads, then H-TEAMS (with D1), then H-BC or H-INIT, then H-KL. H-INIT is added. Literature support for H-BC is withdrawn | the plan §4 |
| **B-M5** the owner's questions are under-served | **FIXED in part; the downloads are DEFERRED.** The first reads are moved to the front: SmallRL (pinned, zero build), the trace twin, LargeRL / LargeIL, then X59-S. HB0 is subsumed by X59-S's S-H arm. LargeRL / LargeIL need two downloads, awaiting approval | the arch doc §7.1; the plan §2 |
| **B-M6** X59's instrument is unchecked | **FIXED.** The trace twin is first read 2, with a 0.05-nat bar. The drawn-move signature is declared and matched across X59-S's arms. In-house head-to-head serves as the non-human-trained meter in X59-S. *Declined:* Foul Play as a further meter. The head-to-head is faster at higher n, and Foul Play adds cost without a question it alone answers | the plan §2.2, §2.4, §3.5 |
| **C-F1** "no human data enters today" is false | **FIXED.** The headline is replaced by a STATUS box and a five-row INVENTORY. The pool's human-built teams were verified: 719 teams = `data/teams/others/` (five PokePaste builders) + `sample/` + `specialist/` | the arch doc §0, §2 |
| **C-F2** no enforcing gate | **FIXED.** `src/human_data_doc_gate_test.py` runs in the static tier (static-gate count 17 → 18). It holds T1 (snapshots vs registry, and every `hr_…` token registered), T2 (build lockstep with `deleted_flags.md`), T3 (pinned numbers from `results.json`), T4 (production off) and T5 (rowsets vs panels). Each check has a planted-mismatch meta test | `src/`; `docs/DEVELOPING.md` |
| **C-F3** snapshot id ≠ rows | **FIXED.** The chain is SNAPSHOT → ROWSET (`<snapshot>.r<n>`, rows manifest sha) → MODEL (`hom_<rowset>`). The registry has `rowsets` / `models` sections (empty), and a new panel retires a rowset (T5) | the arch doc §5; the registry |
| **C-F4** the complexity budget is broken | **FIXED.** `--human-data-combined-consent` is dropped, and so is its `deleted_flags.md` row. `judge` leaves the grammar. One clause, one parameter; every other knob lives in the rowset or the pre-registration | the arch doc §6; `deleted_flags.md` |
| **C-F5** IS and WILL mixed | **FIXED.** A tense rule is stated. Everything unbuilt is marked PROPOSED, and the box line `switch --human-data : NOT BUILT` is pinned by T2 | the arch doc header, §6 |
| **C-F6** under-defined full-information slot; no upload intake | **FIXED.** The taxonomy is two axes (who played × what we know), with (c1) submitted replays and (c2) team sheets. Rule: opponent truth is never an input. A (c1) intake is PROPOSED. Registry rows carry `actor_side` / `opp_side` | the arch doc §3, §4 |
| **C-F7** first page is mechanism, not status | **FIXED.** The status box comes first, then a result → data table, then the mental model with fact 4 | the arch doc §0, §1, §3 |

## MINORs

| finding | disposition |
|---|---|
| A-m1 drift floor ignores accounts | FIXED: an account-disjoint WITHIN / CROSS / EXCESS read is computed; the alarm is recalibrated (excess > 0.016, positive in every split); the reference values are in the registry (`drift_reference`) |
| A-m2 rating mismatch | FIXED: card FINDING 3 is restated as resolved |
| A-m3 forfeit tails | FIXED (rule): the quitter's last 2 decisions are dropped (the plan §3.5) |
| A-m4 stale fill prior | FIXED (design): a lag-matched cost read comes before any lever reads (the plan §3.5); the card §B states the staleness |
| A-m5 circular calibration | FIXED: belief / intent calibration ON vs OFF is an X60 guard |
| A-m6 Choice Band mask | FIXED (design): its share is reported and those masks marked uncertain |
| A-m7 claims and code | FIXED: the row estimate is restated (≈ 0.5M after the exclusion, ESTIMATE); the "lower bound" is scoped to full-six sides; the machine-speed meter is caveated; the priors rule is stated once (the arch doc §2); `judge`'s surfaces are moot (dropped); `hrq.py` uses `utils.paths.repo_path`; the registration claim is qualified; the archive's manifest is its definition |
| B-m1 X59 arithmetic | FIXED: X59 is withdrawn, and the descriptive Δ_U re-plays P\* on new unit seeds |
| B-m2 Q-H9 has no teeth | FIXED: Q-H9 is redefined as matched contrasts (X59-S's Δ_S; H-TEAMS − D1; H-BC − P-BC) |
| B-m3 unregistered discretion | FIXED: TRADE is accepted iff Δ̂ ≥ 2 × \|Δ̂_P\| and the goal (a) guard holds; UNDECIDED leads to the next lever |
| B-m4 goal (a) unguarded | FIXED: Kakuna home72 Wilson LB > 0.50 |
| B-m5 C0 reuse check | FIXED: it now also checks `data/` hashes, the harness, the reference population, the sim binary and the torch / CUDA / driver versions |
| B-m6 AM-5 under-covers | FIXED: an upper-quantile φ |
| B-m7 falsifiers | FIXED: each one is an interval statement, with power where it exists |
| B-m8 the flag does not name the rows | FIXED: rowset ids (C-F3) |
| B-m9 H-BC's dose | FIXED: a held-out human NLL on account-disjoint validation; the VOID checks move to the pre-launch smoke |
| B-m10 §E misclassification; re-read held-out days | FIXED in part: the full-six re-cut is done, and the held-out days are no longer called "untouched". *Declined:* the tier-A re-cut (acting mon with all four moves seen). The proxy is now withdrawn as uninformative, so a sharper version of it would steer nothing |
| B-m11 drift alarm vs n; behaviour | FIXED: the alarm is relative to the snapshot's own account floor at matched size; a behavioural meter (the HOM's NLL by month) is PROPOSED |
| B-m12 base checkpoint | FIXED: a scope row; an ADOPT at P\* is re-read at production |
| B m-lit | FIXED: the literature support for H-BC is struck; the rest is marked UNVERIFIED |
| C-F8 missing questions | FIXED: Q-H13 CONTAMINATION, Q-H14 PROVENANCE and Q-H15 TEACHER QUALITY are added; Q-H9 and Q-H11 are separated; Q-H4 is stated as unanswerable with what would answer it; Q-H1 covers legality |
| C-F9 the archive's training status | FIXED: DROP, and what would reopen it |
| C-F10 one home per fact; inheritance | FIXED: the arch doc §6 owns the grammar; info_levers §13 is a pointer; resume and fork REFUSE without a typed clause |
| C-F11 the card copied into the arch doc | FIXED: §5.1 is removed; pinned headline numbers stay in the status box |
| C-F12 small defects | FIXED: 3,497 / day, 365 / day (160–714); the archive-skew read is linked by folder; the dangling "§3.9" is gone |
| C-F13 account names | FIXED: committed artifacts carry salted ids (`hrq.acct`); the salt and the name map are in `~/gen3ai_archive/human_snapshots/`. Older commits still hold the names, and that history is the owner's call |
| C-F14 drift constants | FIXED: `drift_reference` is in `hr_spec_v1`'s registry row |

## FINDINGS (standing rule 7)

1. **The durable copy covers `hr_spec_v1` only.** `hr_arch_2026-10-24` (206 MB) still exists once, in the gitignored
   corpus. Copying it is the same `hrq.py archive` command; it was out of this unit's list.
2. **The re-run changed one archive row.** The old cache held 79 logs beyond the archive manifest. Re-extracting from
   the manifest changed only the archive's 2026-10 row (9,548 → 9,462 sides) and §A's archive count (16,569 →
   16,490). No spectator number changed.
3. **Older commits still contain account names**: `14ae4203` and `742decb2`, in `results.json`, the card README and
   REVIEW A. The two raw replay fixtures under `src/main/x50_r1/testdata/` carry player names as replays do, and they
   were left as they are.
4. **Not run:** the first reads, the trace twin, the Kakuna-family audit, the coverage read and X59-S. All are designs.
   Downloads awaiting approval: Metamon `LargeRL` and `LargeIL`; the `search.json` coverage pulls; Smogon's monthly
   battle counts for 2026-05..09.
5. **The ≥ 3-fit exclusion's dose leaves out `forum216b`.** It is sealed, and this unit did not read it. The realised
   share is recorded when a rowset is built.

---

# D. RESPONSE to REVIEW D, the adversarial re-review (2026-10-25)

**What this is.** The disposition of every finding of [`REVIEW_D_rereview.md`](REVIEW_D_rereview.md) (verdict
READY-WITH-FIXES; X60 NOT-READY). Each is FIXED (with where), FIXED BY DECLARATION (the plan states the limit and
routes nothing through it), or DEFERRED (with the reason). X59-S is rebuilt on these fixes and pre-registered in
[`../x59s_matched_students_2026-10-25/PREREG.md`](../x59s_matched_students_2026-10-25/PREREG.md); its code is
`src/main/human_data/` (snapshot reader, rowset builder, P\* labels and self-play rows, the student trainer).

| finding | disposition | where |
|---|---|---|
| **D-B1** X60's estimator / power mismatch; the seed rule fires at a true seed SD of 0 | **X60 marked NOT-READY**, with the reason and the four fixes it needs before registration (one AM-5 multiplier implemented in `main.x35.d1`; `plan_power` re-run with it, ≈ 1,000 games per cell under t; a seed bar 10 finals can resolve, simulated at SD 0 and 1.5 pp; the 8-seed classifier costed; the contrasts widened alike). X60 is not on the path: it runs only after X59-S routes a lever | the plan §6 (the NOT-READY box) |
| **D-M1** `hrq.py extract` HUNG on a tampered log or a header mismatch | **FIXED (code).** `SnapshotError` is now an ordinary `Exception` in the new shared reader `src/main/human_data/snapshots.py` (hrq.py imports it), so a worker's refusal is re-raised in the parent; `hrq.py main` prints `REFUSED: …` and exits 1; a reused cached day is re-hashed too (D-M1 fix 3). Tests (`src/main/human_data/snapshots_test.py`, review D's tamper harness on synthetic logs, run in the gate): the pool propagates within a bounded `get`, and the real CLI refuses a one-byte change (fresh and reused cache) and a wrong header, under a subprocess timeout. Reverting the class to `SystemExit` fails 3 of the 4 tests (the CLI test reports the 90-s hang) | `snapshots.py`, `hrq.py`, the test |
| **D-M2** X59-S is not label-matched (P\* can name a drawn move, a drawn bench mon, or a switch on a turn the log proves was a move) | **FIXED (built).** Every rowset row carries its CONSISTENT set: the legal actions the log leaves possible for the human (a move this mon was seen to use, a switch to a mon that was seen). S-P's label is one P\* sample from that set, renormalised, with matched noise keyed by the rowset digest and row id. A hidden-move row gets the SAME set label in S-H and S-P (no P\* sample). A row whose human label falls outside its consistent set (a Transform / Mimic slot) is dropped in every arm. VOID guards: the label writer refuses a P\* label outside the set, and the trainer re-checks it (tests). FR-2 leaves X59-S's critical path | `rowset.consistent_mask`, `teacher.label`, `student.targets`; the PREREG §2 |
| **D-M3** one seed per arm; an unsourced 0.30 base; a pre-exclusion volume | **FIXED.** Five training seeds per arm (5901–5905, the same initial bytes for a seed in every arm). The registered interval adds a seed term: the binomial part plus 2 × max(1.5 pp, the measured seed SD)² / 5, falling back to a Welch t across seeds when the measured seed SD's one-sided 95 % lower bound exceeds 1.5 pp. Power is simulated over seed SD 0–3 pp and base 0.15 / 0.30 / 0.50, and the games are sized at the binomial worst case p = 0.5, so no guessed base enters a decision (`../x59s_matched_students_2026-10-25/power.py`): P(LB > 0) at +4 pp is 0.94 / 0.89 / 0.81 / 0.67 at a seed SD of 0 / 1 / 1.5 / 2 pp, with false positives ≤ 0.028 at Δ = 0. The base is MEASURED by the registered floor pilot (200 games per student) before the full read. The volume is the real post-exclusion rowset (`hr_spec_v1.r1`), not 6.9M: its count is in the PREREG | the PREREG §4; `power.py` |
| **D-M4** the owner asked "human DATA vs self-play DATA"; S-S was optional | **FIXED.** S-S (our architecture trained the same way on P\*'s own self-play states and sampled labels, the same row count) is MANDATORY. **Δ_data = S-H − S-S answers the owner (Q-H9)**, registered with its own rule beside Δ_labels = S-H − S-P (routes H-BC); the states part S-P − S-S is reported | the PREREG §3, §5; the plan §2.4; the arch doc §7.1 |
| **D-M5** H-INIT forks a HOM fitted on rows the new rule forbids | **FIXED by dropping H-INIT** from the ranked plan. The existing HOM is barred by name from every lever and every routing rule: it stays a measurement artefact, and its descriptive Δ_U is labelled CONTAMINATED and routes nothing. H-INIT, or an H-KL `kl:<model>`, may return only on a `hom_<rowset>` re-fit on a §3 rowset with account-disjoint early stopping, as a new registration. Its compute is small; its cost is VOLUME: the compliant ≥ 1500 rows are `hr_spec_v1.r1`'s `rating_ge1500_rows` (its manifest), against the HOM's 1.76M | the plan §2.5, §4 |
| **D-M6** no non-human-trained meter on the unfamiliar panels | **FIXED.** X60 (when registered) carries an in-house U read (ON − OFF on F + P72 against the frozen reference population through `main.h2h`) as a GAIN guard; a GAIN only Kakuna sees is reported "Kakuna-specific". In X59-S the in-house head-to-heads (S-H vs S-S, S-H vs S-P on H, F, P72) are always reported, and decide under the floor fallback | the plan §6; the PREREG §5 |
| **D-M7** what the exclusion keeps is a different population | **FIXED.** The kept / excluded profile is on page one of the plan (§0) and in §3.1, and every rowset's manifest records it (`rowset scan`, `rows_manifest.json` `panel_exclusion_profile`, `row_profile`). **Registered choice:** X59-S's rows FILTER battles of ≤ 10 turns (the information floor; it also removes every walk-over) and KEEP forfeits beyond that with the quitter's last 2 decisions dropped; the row share by ending is recorded. X59-S's absolute levels are read as this slice's | `rowset.py`; the plan §0, §3.1; the PREREG §2 |
| **D-M8** page one still misleads on "is there evidence it helps?" | **FIXED.** The status box says `evidence that human data helps goal (b) : none yet, either way`, the `open` line is in words, the switch values are spelled out, and the 15.9 % line is relabelled as D-m10 asks | the arch doc §0; the gate's key |
| **B-m7 / D-m4** four falsifiers were point statements | **FIXED.** Every falsifier in the plan §8 is an interval statement with a boundary rule (a bound within 1e-9 of its bar takes the undecided class), FR-2's included | the plan §2.2, §8 |

**MINORs.**

| finding | disposition |
|---|---|
| D-m1 the gate's blind spots | FIXED in part: `status_box` now REFUSES a duplicate key (a stale line before the real one), with a planted-duplicate meta test. DEFERRED: pinning the qualitative lines and the numbers repeated outside the box (each needs a source the gate can read; listed for the next gate change) |
| D-m2 `extraction_commit` misleads | FIXED: `hrq.py`'s stamp records `hrq_dirty`; the rowset records a `source_sha256` over every file that makes a row, and refuses to compact parts of different code |
| D-m3 the durable copy's threat model | DEFERRED (owner's call): the copy and the salt sit on the same disk; `extract --from-durable` is not built. Stated as a FINDING |
| D-m5 X59-S's volume and cost | FIXED with D-M3 (the real rowset's count; the measured CPU cost per step) |
| D-m6 the H-p exception | FIXED: a battle joins ON's rows only if EVERY one of its ≥ 3-fit matches is to an H-p team |
| D-m7 the audit's HOM bias | FIXED (design): the Kakuna-family audit scores against P\*, not the HOM |
| D-m8 the contrasts' seed term | FIXED with D-B1 (X60 NOT-READY until the contrasts carry it) |
| D-m9 RESPONSE wording | FIXED here: B-B2's "630" is the same 1,260 games per final as B's frame on two panels; A-B2's 70.7 % is for H, F and P72 |
| D-m10 the 15.9 % box line | FIXED: relabelled `>= 1500 full-six sides unfamiliar to the pool (<= 2 fits; a lower bound)` |

**FINDINGS (standing rule 7).**
1. **The exclusion with `forum216b` included removes 83.2 %** of the candidate battles (a ≥ 1300 non-bot seat, ≥ 11
   turns), against REVIEW A's 70.7 % without it (`forum216b` is 216 teams). Its content was read for the exclusion
   only (`heldout_panels.panel_team_texts`, which the seal allows); it was not played.
2. **The rowset's teacher is mostly 1300–1499.** On the smoke date 93 % of ≥ 1300 rows came from 1300–1499 actors;
   the realised share is in the rowset manifest. The card ranks that stratum "the human-only comparator only".
3. **The extraction at HEAD reproduces the HOM's.** On the overlapping battles of a 300-log smoke, 1,196 / 1,196 rows
   are byte-equal to the rung-1b extraction (`2bbdffa7`) with equal labels, though the live reader binary differs;
   on the built rowset, 8,878 / 8,878 rows of 200 seeded battles (X59-S's VOID 8, PASS).
4. **No CPU pilot student was trained.** The registered recipe costs ≈ 2.5 s a step at 2 threads on the shared agent
   cores (≈ 10 lane-h a student at 3 epochs), above the brief's 6 lane-h; the floor pilot in the GPU window measures
   the base instead.
5. **Torch at 6 threads on the agent cores ran 60× slower** than at 2 (160 vs 2.6 ms a row) under the live run's
   contention; the new tools default to 2 threads.
6. **The durable copy and the salt** share one disk (D-m3), unchanged.

---

# E. RESPONSE to REVIEW E, the adversarial review of the X59-S registration (2026-10-25)

[REVIEW E](../x59s_matched_students_2026-10-25/REVIEW_E.md) (`d02550c0`) read X59-S NOT-READY: two BLOCKERs, eight
MAJORs, seven MINORs. Every fix is in the PREREG's **Amendment 2** (rules, registered before the audit's flags, the
regenerated S-S rows or any training) and **Amendment 2b** (the inputs as built). No GPU was used.

| finding | disposition |
|---|---|
| **E-B1** S-S is held to a looser panel-proximity standard than S-H | **FIXED.** S-S's team pair is redrawn before each game while either side's SIMULATED human reveal is near a held-out panel team (the human rule's matcher, panels incl. `forum216b`, threshold), from a reveal profile of 35,554 candidate human sides; every game's pair is recorded; S-S regenerated (seed 5910). **Departure:** the profile is the CANDIDATE sides', not the kept ones' (the kept sides are selected for revealing little, so they would under-exclude). Measured on the pool before generation: 16.6 % of pairs pass, against the 16.8 % of human battles the rule kept. Also matched: S-S drops games of < 11 turns (E-m7). Δ_data per panel is reported |
| **E-B2** the Kakuna-family sensitivity cannot be read as registered; the audit misses 78 % of the rows | **FIXED, the preferred way:** the audit runs FIRST, on every ≥ 1300 account with ≥ 10 kept games (83.1 % of TRAIN rows; REVIEW E proposed ≥ 20, 67.9 %; a seeded 10 games an account keeps the cost flat), score Kakuna − P\* log-likelihood, flag above the chat-active reference's p99, a POWER control on Kakuna's own captured seats. Flagged rows > 1.0 % of TRAIN rows ⇒ the flagged accounts leave the rowset before training. §8.4's "sensitivity re-read" and its retrain are retired |
| **E-M1** the quitter's degraded play runs past the 2 dropped decisions | **FIXED:** rows with `0 ≤ quit_dist ≤ 9` leave (the derived rowset); S-P's labels carry over unchanged; S-S is cut to the new counts. The ≤ 10-turn filter stays |
| **E-M2** S-P is P\* restricted to the human's whole-game support | **FIXED.** Named "P\*-restricted-to-human-support" with REVIEW E's numbers as its definition; OWN-LABELS-BETTER is conservative; HUMAN-LABELS-BETTER is reported against the restricted teacher and routes H-BC only if the S-P students' yardstick on S-S validation is below the S-H students' (Welch, UB < 0). **Departure:** REVIEW E proposed "within a registered margin of the S-S students'"; no margin can be set before data, and S-S's states differ, so the same-states falsifier against S-H replaces it. **S-P0 ADDED** (descriptive): P\*'s unrestricted sample from the stored logits (labels free), 5 students ≈ 1 GPU-h after the primary 15 |
| **E-M3** the drawn-move signature is not matched against S-S | **DOCUMENTED + GATED:** a registered limit expected to cost S-H; FR-2 is read before Δ_data is reported (the read code refuses without it); ARTEFACT ⇒ a SELF-PLAY-DATA-BETTER closes nothing. FR-2 is NOT run in this unit (≈ 1 lane-h + 0.2 agent-day): a FINDING |
| **E-M4** no UNINFORMATIVE outcome; no floor reference | **FIXED:** `bot:random` and `bot:aggressive` (max-damage) vs Kakuna on the pilot twins (240 games each, CPU, before the floor pilot); UNINFORMATIVE = every arm ≤ the better floor + 5 pp, or the fallback fires (closes and routes nothing); a learned-nothing VOID guard (final own-label validation NLL ≥ 0.1 nat below uniform) |
| **E-M5** the closure speaks for human data in general | **FIXED:** closure scoped to "1300+ ladder data, at this mix and volume"; the conditional ≥ 1500 follow-up (S-H≥1500, S-H-mix, S-S cut at matched rows, 5 seeds each, in-house; ≈ 0.7 GPU-h) is registered, firing when Δ_data is neither HUMAN-DATA-BETTER nor UNINFORMATIVE |
| **E-M6** NOT-RESOLVED merges "no difference" and "no information" | **FIXED:** NO-DIFFERENCE-AT-4PP (the whole 95 % interval inside ±4 pp, the MDE), with its own action, in code (`x59s_read.classify`) |
| **E-M7** the CUDA path has no pass criteria | **FIXED:** `cuda_preflight.py` implements the 8 checks (CPU-vs-CUDA parity, TF32 off, resume, export, cost, learning by step 1,000); `window.py` continues a student only on exit 3, runs each chunk under `gpu_lock.sh mem_cap.sh 8`, primary students first. Tested CPU-vs-CPU only: a FINDING until the first GPU minutes |
| **E-M8** no rule when Kakuna and the in-house meter disagree | **FIXED:** KAKUNA-SPECIFIC (in code, `x59s_read.verdict`); the in-house head-to-heads play for every read |
| E-m1 … E-m7 | **FIXED as reporting rules** (Amendment 2 A2.11); E-m4 (the trainer refuses unregistered inputs) is enforced in code (`student --registration`); E-m7 also changes S-S (the < 11-turn drop) |

**Amendment 2b (the inputs as built).** The audit has POWER (Kakuna's own seats: 158 / 158 pseudo-accounts above
the human p99) and found no Kakuna-like account; its registered rule still fired (11 accounts, 1.73 % of r1's TRAIN
rows, flagged because P\* fits them badly) and they were dropped as registered. `hr_spec_v1.r2`: 1,201,538 / 29,904
rows (−108,740 quitter-tail and −20,531 flagged TRAIN rows); 7,041 steps a student. S-S regenerated: 16.8 % of pool
pairs passed, matching the human rule's 16.8 %. `registration_a2.json` PASS. S-H, S-P, S-S and S-P0 are READY to
train (≈ 3.6 GPU-h, the CUDA pre-flight first).
