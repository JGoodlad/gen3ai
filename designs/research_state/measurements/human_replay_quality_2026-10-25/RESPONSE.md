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
