# REVIEW A: data, bias and leakage (adversarial), of the human-data DRAFT at 14ae4203

**Under review:** [`human_data_architecture.md`](../../../endstate/human_data_architecture.md),
[`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md), this directory's data card
([`README.md`](README.md), `hrq.py`, `results/results.json`) and the registry
[`human_replay_snapshots.json`](../../human_replay_snapshots.json).

**Lens:** train / eval leakage, selection, labels, distribution shift, snapshots and the priors rule. The causal and
experimental design is REVIEW B's.

**Method:**
- CPU only, rule 5b cores, every heavy step under `mem_cap.sh`.
- Read-only on the corpus, the hrq cache, the X50 archive and the panels. The sealed forum216b was never read.
- Three scripts, committed beside this file:
  - `review_a_checks.py`: the cache-based checks, ≈ 2 min, 5.7 GB peak;
  - `review_a_mnar.py`: a 6,000-log text pass;
  - `review_a_guard_sim.py`: the contamination-guard simulation.
- Every number below is their printed output unless it is tagged otherwise.

**Abbreviations:**
- H = heldout72_hpfix, F = forum72, P72 = the procedural panel.
- "Fit" = `memory._fits`: a mon whose revealed moves, item and ability all sit in one panel set.
- MNAR = missing not at random.

## Verdict: SOUND-WITH-FIXES, and NOT BUILDABLE until B1 and B2 are fixed

**What holds.** The card's measured facts reproduce: the snapshot digests, a re-hash sample, the population split,
the bot share, the month TVDs and the side-level floor. The snapshots really are the X50 inputs, and the spectator
and archive populations are cleanly separated.

**What fails.**
- The plan's leakage controls are UNSOUND as specified. H-TEAMS's panel guard catches ≈ 1 % of the copies it exists
  to catch (B1). The ≥ 4-fit exclusion keeps 3-fit neighbours of the held-out panels in ≈ 48 % of all ≥ 1500 battles, ≈ 75 % of
  the battles it keeps (B2).
- Four claims the plan leans on do not survive a check:
  - "no familiarity gradient" (M1);
  - "the held-out dates are held out" (M2);
  - "the labels are unbiased because the chosen action is real" (M3);
  - "real, slow drift" (m1).

| # | severity | finding | one-line fix |
|---|---|---|---|
| B1 | **BLOCKER** | H-TEAMS's guard (`contamination`, exact mon sets ≥ 4) recalls ≈ 1 % of exact human copies of a panel team after C-GAME completion | run the FIT rule on the REVEALED facts before completion; add a test that a panel team's own log is caught |
| B2 | **BLOCKER** | The exclusion (≥ 4 fits) is looser than the panels' own "unfamiliar" region (≤ 2 fits). 3-fit neighbours of a held-out team remain in 48 % of all ≥ 1500 battles, so U stops being unfamiliar | register ONE consistent rule (≥ 3, or ≥ 4 plus a registered per-team dose-response read), and state the cost: ≥ 3 removes 84 % |
| M1 | MAJOR | The familiarity read (§E) cannot show "no gradient": its strata are mostly unseen teams, the completion homogenises them, it has no equivalence bar, and the metric has no A-1 prediction | withdraw "removes the offline evidence"; re-cut on full-six sides with a delta CI, or drop the proxy |
| M2 | MAJOR | The held-out dates share 79 % of ≥ 1500 accounts and 57 % of (account, species sextet) with TRAIN; the HOM's 2 % validation is by battle id | an account-disjoint held-out split; re-read Δ(HOM − P\*) on unseen accounts |
| M3 | MAJOR | Hidden choices are MNAR. Dropping `cant` and fainted-before-acting rows doubles the labelled switch rate of a sleeping mon (0.56 vs 0.26) and of a frozen one (0.68 vs 0.29) | keep those rows with a set-valued label ("some move"); a `cant` proves a move was chosen |
| M4 | MAJOR | Kakuna-family play in the corpus cannot be excluded by a name or volume rule, and X59 / X60 score against Kakuna | a per-account policy-likelihood audit under Kakuna, plus a registered sensitivity arm |
| M5 | MAJOR | Snapshot citation is not enforced anywhere: `results.json` carries no snapshot id; `hrq extract` reads date folders, not the manifest, with no hash check; X50's rows carry neither | stamp the id and manifest digest; extract FROM the manifest; refuse on a hash or header mismatch |
| M6 | MAJOR | `hr_spec_v1`'s bytes exist once: in the main checkout's gitignored `replays/`. The collector is deleted and the archive holds ≈ 1/10 of these games | make the zstd tar now (≈ 0.6 GB, ESTIMATE), sha256 it in the registry, keep it off the checkout |
| M7 | MAJOR | The spectator's selection is concrete, non-stationary and unmeasured: public rooms only, a 30 s poll, a concurrency cap of 20, a reaper added mid-window, > 1 h rooms dropped. Smogon's counts suggest ≈ 50 % coverage | measure coverage by month and by game length before calling (b) "the reference" |
| m1 | MINOR | The drift floor ignores account clustering. The account-split floor is 0.050, not 0.009, so month-to-month 0.038–0.049 is not "4–5× the floor" | recompute the floor by account; recalibrate the alarm |
| m2 | MINOR | Card FINDING 3 (the 2 % rating mismatch) is benign: every mismatch is an unrated row (None vs 0) | restate it as resolved |
| m3 | MINOR | Forfeit games are less predictable (P\* 1.796 vs 1.681 nats); the quitting side's last decisions are likely low-effort | drop or flag the quitter's last k decisions; register k |
| m4 | MINOR | The fill prior is Smogon 2025-05..2026-04 (1–17 months stale). Its TVD to ≥ 1500 usage grows 0.070 → 0.092 over the window | a lag-matched prior, or measure the cost |
| m5 | MINOR | H-TEAMS opponents' hidden sets are Smogon draws, and the network's belief priors are the same Smogon tables: a circular calibration | read belief-head calibration ON vs OFF in X60 |
| m6 | MINOR | Choice Band is never revealed in gen-3 logs, so an unrevealed band leaves a 4-move mask on a locked mon (label right, mask wrong) | report the share; mark those rows' masks uncertain |
| m7 | MINOR | Small claim and code errors: the 1.7M row estimate; the "lower bound" scope; the machine-speed meter; the rules text; `judge` vs "eval and play honour only off"; `hrq.py:32`'s `parents[4]`; registration not restated | see below |

---

## BLOCKERS

### B1. H-TEAMS's panel guard cannot see a completed copy of a panel team

**Evidence.**
- The plan (§4.2) excludes completed human teams with `main.heldout_panels.contamination`, "0 hits at startup".
  - That function is `hits()` over `Reference.max_shared`.
  - It counts mon sets matching on species plus the SORTED MOVE TUPLE (`src/main/heldout_panels.py:225-239`;
    `src/main/anchors/team_overlap.py:159,191`).
- A C-GAME team keeps only the revealed moves, ≈ 1.9 of 4 at ≥ 1500 (card §B), and draws the rest from Smogon.
- `review_a_guard_sim.py` simulates a human piloting a panel team EXACTLY:
  - each move is revealed with p = 0.475;
  - the rest are drawn from `agents.gen3_data.priors.moves`;
  - all six mons are seen (an upper bound on the guard's recall);
  - 20 draws per team.

| panel | guard recall (≥ 4 exact sets) | exact mons of 6 |
|---|---:|---:|
| H | **1.0 %** | 1.05 |
| F | **1.1 %** | 1.09 |
| P72 | **0.4 %** | 0.99 |
| (at p = 0.75 per move) | 24–27 % | 2.6–2.7 |

**Consequence.**
- "0 hits at startup" is VACUOUS: the `--team-mix` training guard passes a corpus that holds human copies of H / F /
  P72 teams.
- §5.1 makes H-TEAMS the likely FIRST build ("UNCLEAR … routes to H-TEAMS first").

**Fix.**
- Exclude at EXPORT on the log's revealed facts, with `memory._fits` over both seats, BEFORE completion. This is
  the H-BC rule, widened to every held-out panel including forum216b, whose content `panel_team_texts` already serves
  to exclusion code.
- Keep the exact-set guard as a second net.
- Add a regression test that fails on revert: a log in which a side plays a panel team must be excluded.

### B2. The exclusion threshold contradicts the panels' own definition of "unfamiliar"

**Evidence.** The panels' region is "≤ 2 mons fitting any pool team's sets" (card §D: H 98.6 %, F 77.8 %, P72
100 %). The exclusion drops only ≥ 4 fits (plan §4.1; `memory.py:21` `NEAR = 4`). Over the 56,073 TRAIN-date battles
with a ≥ 1500 side (`E_panel_exclusion`):

| a side fits one held-out team (H, F or P72) with … | share of battles | share of turns |
|---|---:|---:|
| ≥ 4 mons (excluded by the plan) | 36.4 % (H 26.5, F 18.2, P72 4.2) | 33.8 % |
| ≥ 3 mons | **84.4 %** (H 72.1, F 67.2, P72 48.9) | 83.9 % |
| ≥ 3 mons, counting only mons with a revealed MOVE | 70.7 % | — |
| ≤ 3 fits but ≥ 5 shared species | 24.3 % | — |
| ≥ 4 only through species-only reveals (≤ 1 move-revealed fit) | 0.15 % | — |

**What the numbers say.**
- About 48 % of all battles (84.4 − 36.4) are KEPT by the ≥ 4 rule while a side carries a 3-set core of a held-out
  team.
- By the panels' own yardstick, a team becomes "familiar" when the training teams hold a 3-fit neighbour. After H-BC
  or H-TEAMS, most of H, F and P72 stop being unfamiliar.
- Species-only matches are NOT the problem: 0.15 %. The radius is.
- **H is not X59's clean cell either.** The HOM excluded only H at ≥ 4, and H's 3-fit neighbours sit in 72 % of the
  ≥ 1500 battles.
- **H's own sets are partly invented.** `hl_05_26` "fills replays … with teams predicted from … usage stats" (Metamon
  README, "High Ladder May '26"). A human's later reveals of the same team can therefore contradict a filled set,
  so `_fits` under-counts copies.
- hl_05_26 draws on replays up to May 2026, so its source battles may lie inside `hr_spec_v1`'s first week. That is
  UNVERIFIABLE: the team files carry no replay ids.

**Fix.** Choose one before X59 is read as A-1 evidence or X60 is registered.
- **Option (a):** exclude at ≥ 3 move-revealed fits. This is consistent, and it keeps ≈ 16–30 % of the rows. State
  that dose.
- **Option (b):** keep ≥ 4 and make A-7 a REGISTERED secondary with a rule:
  - Δ_U per panel team, regressed on that team's count of 3-fit human neighbours in the training rows;
  - a positive slope means memory.
  - Report F's FAR stratum (thinly covered) separately.

  Either way, §5.2's claim that F + P72 are "populations no human lever trains on" is false at the 3-fit radius.
  Drop it.

---

## MAJORS

### M1. The familiarity read (card §E) cannot carry "removes the offline evidence for A-1"

**The strata are mostly unseen teams** (`D_familiarity_contamination`, held-out ≥ 1500 sides):
- of "species overlap ≤ 3 (unfamiliar team)", 216 of 249 sides (87 %) never showed all six mons, and 170 (68 %)
  showed ≤ 4;
- of "set fits ≤ 2", 186 of 367 (51 %) were not fully seen. An unseen mon cannot fit, so for these sides "≤ 2" is
  not an upper bound;
- §D restricted itself to full-six sides; §E (`hrq.py:401-461`) did not.

**Other problems:**
- **The completion homogenises the teams.** P\* reads the C-GAME own side, whose unrevealed ~52 % is Smogon's most
  likely sets. An odd team, once filled, looks standard. This shrinks the contrast that is being measured.
- **No equivalence was registered.** The cells give separate CIs ([1.706, 1.826] vs [1.709, 1.767]) and no CI on
  their DIFFERENCE. Overlapping intervals are not equivalence (memory: "equivalence needs the delta's CI inside the
  bar"). The point estimate (+0.024) also leans the A-1 way, and the species cut leans the other way.
- **The metric has no A-1 prediction.** NLL against humans rises wherever P\* disagrees with humans. On pool-like
  teams P\* plays its own self-play style with sets it knows, so it can disagree there as much as on unfamiliar teams,
  for the opposite reason. A flat profile is predicted with A-1 true or false.

**Fix.**
- Withdraw "removes the offline evidence" in the plan's §0, §2.1 and FINDING 1, and in the architecture doc's
  Q-H7. Say "uninformative".
- If the proxy is kept: full-six sides only, a battle-clustered CI on the stratum difference, a registered bar, and
  the opponent team's familiarity as a second axis.

### M2. "Held-out dates" are not held out in accounts or teams

**Evidence** (`I_heldout_overlap`):
- of 2,086 held-out ≥ 1500 sides, **1,655 (79 %)** are accounts seen in TRAIN;
- of the 1,467 full-six sides, **836 (57 %)** repeat the same account with the same species sextet;
- the HOM's VALIDATION split is `sha256(battle id) mod 50` inside the train dates (rung-1b design §3), so its
  early-stopping and "what volume buys" curve (card §C) are in-account too.

**Consequence.** These numbers include account and team memory:
- Δ(HOM − P\*) of −0.57 to −0.62 ("learnable everywhere");
- the R-b2 judge read;
- the "uniform" strata.

**Fix.** Hold out ACCOUNTS: hash the account id for both the validation and the held-out split, then re-read
Δ(HOM − P\*) and §E on accounts never seen in training. This needs CPU only, on R-b1's cached forwards plus a
re-validation.

### M3. Hidden choices are missing NOT at random, and the bias lands on the switch policy

**Evidence.** In `src/main/x50_r1/own.py:571-613`, a `cant` and a faint before acting both give
`OwnAction("none")`, which becomes label −1, which drops the row. `review_a_mnar.py` (6,000 seeded TRAIN-date logs,
≥ 1500 sides, each turn decision classed by its first own event) measures the effect:

| active's status at the turn | P(switch), labelled rows only | P(switch), with the hidden rows counted as stay-in | stay-in decisions hidden |
|---|---:|---:|---:|
| asleep | **0.558** | **0.260** | 72 % |
| frozen | 0.675 | 0.288 | 81 % |
| paralysed | 0.296 | 0.216 | 35 % |
| healthy | 0.242 | 0.227 | 8 % (KO'd before acting: 11,155 vs 2,102 `cant`) |
| all | 0.250 | 0.228 | 12 % |

The paralysis line checks the parser: 2,753 `cant par` against 8,056 moves is 25.5 %, gen 3's full-paralysis rate.

**Why these rows are partly labelled.** In gen 3 a switch executes before the status check and before the foe's
move, so a `cant` or an own KO before acting PROVES the human chose a move rather than a switch. Dropping those rows
teaches H-BC, the HOM and HB0 that a sleeping mon is switched out 2.1× as often as humans do. It also teaches that
"stay in" is rarer exactly where staying in got the mon KO'd.

**Fix.**
- Keep these rows with a SET label: the loss is −log Σ over the legal move actions of p(a).
- Exclude the windows where the foe's Pursuit KO'd a switching mon.
- Report the corrected switch rate by status in the card.

### M4. Kakuna-family play may sit in the teacher data, and the meter is Kakuna

**Evidence.**
- Metamon's README: its agents rate "90–99th percentile against human players"; "KakunaEnsemble" was "#1 on the
  human Showdown ladder"; "we have become a lot more cautious about laddering".
- Its public eval agents `PastaMittens` (the Kakuna router family) and `Exeggcute` can be run by anyone, under any
  name.
- The card itself found an account named like a Metamon public agent (`acct_9f2ef7808db6`), and a 24-hour fleet of
  generated names (`acct_64c8a8570c27`, `acct_bd194687c6bd`, `acct_5aed467f14fc`, `acct_cc4b77376004`,
  `acct_b1887901ed8f`). (Account names replaced by the card's salted ids in the 2026-10-25 revision, REVIEW C F13.)
- The rule flags by name, volume (≥ 100 games a day) or hours (≥ 20).
  - 31 unflagged accounts have ≥ 100 games, zero chat, and ≥ 50 games or ≥ 14 active hours in one day. They hold
    1.3 % of ≥ 1500 TRAIN sides (`A_zero_chat_heavy_unflagged_*`).
  - A fleet that plays 40 games a day per account passes unflagged.
- Flagged accounts chat (median 0.056 lines a game), so the rule mostly catches human grinders. That is a
  false-positive cost of ≈ 5.6 % of ≥ 1500 sides (reproduced: 0.0560).

**Consequence.** H-BC or the HOM imitating Kakuna's own decisions, then scored against Kakuna, measures familiarity
with the opponent's policy, not human skill on unfamiliar teams. A-6's hand audit of 50 random accounts cannot find
a bot that plays like a strong human.

**Fix.**
- Score every ≥ 1500 account with ≥ 50 TRAIN games by the mean log-likelihood of its labelled actions under Kakuna's
  policy (CPU) against the HOM's. Flag the accounts above the human distribution's p99.
- Register an X60 sensitivity arm, or a re-read, without them.
- The same audit applied to the HOM bears on X59.

### M5. The snapshot exists, but nothing makes a result use it

**What checks out (`G_*`):**
- the registry digests equal the manifests' (e7d491e7…f3b1, ad940108…f7a2);
- a 400-file re-hash of `hr_spec_v1` gives 0 mismatches;
- the snapshot's paths equal the spectator manifest's;
- all 376,410 X50 battle paths (368,997 train + 7,413 held out) are in it;
- every `hr_spec_v1` file starts `|init|battle` (376,410 / 376,410) and every archive file does not (16,490 / 16,490).

**What does not:**
- `results.json` has no snapshot id or digest (its keys: schema, corpus, files, …).
- `hrq extract` walks DATE FOLDERS on disk (`hrq.py:200-206`), labels "spec" by path membership, never hashes a
  file, and REUSES any cached day by existence (`hrq.py:174`). The card's cache already mixes in the 79 logs of the
  partial 10-25 folder (README, last line).
- X50's part JSONs carry absolute paths, a commit and the reader's hash, but no snapshot id and no per-log sha. The
  plan's X60 VOID stamp `rows_sha` proves which ROWS were read, not which LOGS made them.
- A run can therefore silently read a different corpus state today, and the "re-hash check is to build".

**Fix (before any consumer is built).**
- `extract` reads the snapshot's TSV, re-hashes each log, checks the population's header signature (`|init|battle`
  for the spectator; none for the archive), and REFUSES on any mismatch.
- Every derived artefact (cache, part, shard, team corpus, `results.json`) stamps
  `{snapshot_id, manifest_sha256, extraction commit, layout hash}`.
- The consumer refuses a stamp the registry does not hold.

### M6. The reference dataset has one copy

**Evidence.**
- The logs live only under the main checkout's `replays/`, which is gitignored (`.gitignore:31`).
- `~/gen3ai_archive` holds manifests, not bytes.
- The collector was deleted on 2026-10-08 (`27f150724`).
- The archive holds about a tenth of these games ([`designs/tools/replay_corpus.md`](../../../tools/replay_corpus.md)).

So a `git clean -fdx` in main, or a disk loss, destroys `hr_spec_v1` beyond recovery. A sha256 manifest DETECTS a
change; it does not PRESERVE.

**Fix.** Build the zstd tar the docs already promise (§3, §7.2) now, put it under `~/gen3ai_archive`, and record its
sha256 in the registry row.

### M7. The spectator's selection is concrete and unmeasured, and it changed during the window

**Evidence.** The deleted daemon (`git show 27f150724^:src/poke_env/spectator/spectator_client.py`; `collect_replays.py`
deleted in `5d85d8a20`):
- polls `/query roomlist gen3ou` every 30 s;
- Showdown's `getBattles` skips `room.settings.isPrivate` (`deps/pokemon-showdown/server/rooms.ts:1548`), so
  private and hidden battles are NEVER sampled;
- watches at most 20 rooms (the `--max-concurrent` default; the flag the run actually used is unrecorded, and
  `5d85d8a20` notes that the deleted `collect_replays.py`'s newest output was 2026-08-02, so what collected
  08-18..09-23 is undocumented);
- drops rooms that are stale for 10 min or watched for more than 1 h, unsaved ("Abandoned rooms are not saved");
- gained its stuck-room reaper on 2026-07-05 (`5d77790d9`), which changes the capture process mid-window.

**Measured** (`B_*`):
- the captured games' overlap runs p50 10, p90 17, p99 22 (battle intervals, not watch intervals): consistent
  with a cap near saturation at peak hours, not proof of it;
- the daily median falls 4,141 (Jun) → 3,408 (Jul) → 2,997 (Sep);
- Smogon's gen3ou-1500 window (2025-05..2026-04) counts 2,533,370 battles, ≈ 6,900 a day, against the spectator's
  ≈ 3,500. That is ≈ 50 % coverage IF 2026's volume matched (ESTIMATE);
- on 05-22, 45 of 51 UPLOADED games were spectated (corpus doc). Long, high-rated games were therefore well covered,
  which means the miss falls elsewhere.

**Consequence.** (b) is "public rooms, at the daemon's capacity, under a changing reaper". Its month-to-month numbers
(card §F) mix capture with meta. A-4 is currently unfalsifiable as worded: the live stream is gone.

**Fix (CPU, through the approved proxy, download only).**
- Sample `search.json` uploaded ids for ~30 spectator dates and measure P(spectated) by turns, rating, hour and
  month.
- Get Smogon's monthly `number of battles` for 2026-05..09; whether that fetch is covered by the proxy grant is the
  owner's call.

---

## MINORS

- **m1 (drift floor).**
  - July's split-half floor is **0.0082 by side** (max 0.0096; the card's 0.009 reproduces) and **0.050 by account**
    (0.043–0.059; `C_july_split_half_floor_by_account`).
  - The month-to-month TVDs reproduce: 0.044, 0.041, 0.038, 0.049. They sit inside the account-clustered floor, so
    "real, slow drift" is not shown, and the 0.074 alarm is 1.25–1.5× noise.
  - Calibrate the alarm on the clustered floor.
  - The ≥ 1500 first-vs-last month TVD is 0.103 on smaller samples, also unfloored.
- **m2 (the rating mismatch).** All 7,601 mismatched held-out rows are `rating_viewer ≤ 0` with no header rating. 0
  equal the other seat; 0 change the ≥ 1500 stratum (`F_rating_mismatch`). Card FINDING 3 is resolved: the join is
  right.
- **m3 (forfeits).**
  - The card's own §E: forfeit / timer games 1.796 (P\*) and 1.185 (HOM) against KO games 1.681 and 1.120. The
    quitting side's last decisions are worse data.
  - The walk-over filter (≤ 3 turns) is not that.
  - Register "drop the quitting side's last k = 2 decisions" (or weight them), and report NLL by decisions-before-quit.
  - HB0-ORL's outcome reward also inherits rage-quits as losses.
- **m4 (the stale fill prior).**
  - `data/pokemon/gen3_smogon_stats.json` is `window_months` 2025-05..2026-04, cutoff 1500.
  - Its species TVD to ≥ 1500 spectator usage is 0.070, 0.059, 0.058, 0.080, 0.092 (05..09). C-GAME fills a 09-20
    battle from a meta 5–17 months older. F-1b-4's own-team cost (+0.019 nats) does not cover this.
  - Measure it with a lag-matched prior on a held-out slice.
- **m5 (circular calibration under H-TEAMS).** About 52 % of an H-TEAMS opponent's move slots, and every spread, are
  Smogon draws, and the observation's belief priors are Smogon. Self-play against such teams rewards a belief that is
  exactly calibrated to Smogon, which real opponents (Kakuna's panels, humans) are not. Add the belief / intent heads'
  calibration on the Kakuna cells, ON vs OFF, to X60's guards.
- **m6 (the Choice Band mask).**
  - Choice Band is never announced in a gen-3 log. If the completion draws a non-band item for a banded mon, the mask
    offers four moves to a locked mon.
  - The reverse is redrawn: `complete.py` redraws on an illegal label. That makes the drawn item a function of FUTURE
    actions (it keeps a band only if the mon never changed moves), which is a small future-information path inside
    F-1b-4.
  - Report the share of informative rows whose active used one move throughout its stay with no item revealed.
- **m7 (claims and code).**
  - **The row estimate.** H-BC's "≈ 1.7M rows" ignores that widening the exclusion from H to H, F and P72 raises it
    from 26.5 % to 36.4 % of ≥ 1500 battles, and that bots take another 5.6 %. That gives ≈ 1.4–1.5M (ESTIMATE),
    and ≈ 0.35–0.65M under B2(a).
  - **The "lower bound".** The 15.9 % holds for FULL-SIX sides only (68–71 % of sides, longer games); say so where
    it is quoted as "≥ 1500 human sides".
  - **The machine-speed meter.** "Machine-speed games ≤ 0.02 %" can only see bot-vs-bot games, because a bot waits
    for its human opponent. Do not place it as evidence about bots.
  - **The priors rule, stated twice.** Root `CLAUDE.md` lets the network read what "traces to … ladder replays". The
    architecture doc forbids any human-derived feature or table. Reconcile them in one place, or a future
    `tools/` replay table passes one rule and breaks the other.
  - **`judge`.** The plan (§7.1) says "eval and play honour only `off`", while `judge` is a SEARCH consumer that runs
    at play time. Say which surfaces may run `judge`.
  - **`hrq.py:32`.** It hand-writes `Path(__file__).resolve().parents[4]`, against root's paths rule; use
    `utils.paths.repo_path`.
  - **The registration claim.** `hrq.py`'s docstring says its thresholds are "restated" in the plan's §2. They are
    not, and the code arrived in one commit with its results, so "fixed before the analysis ran" cannot be checked.
    Say so.
  - **The archive rule.** `hr_arch_2026-10-24`'s rule ("as on disk 2026-10-25") is not reproducible, because late
    uploads land in old date folders. The manifest is the definition, so say that.

## What I could not verify (standing rule 7)

- **Selection.** Whether any `hr_spec_v1` account IS a Metamon or Kakuna agent (M4 is a risk, not a measurement), and
  the spectator's true coverage (M7's 50 % is an ESTIMATE from a different year).
- **Provenance.** hl_05_26's source-replay dates and ids: whether H's source battles are in `hr_spec_v1`.
- **The MNAR replication.** M3's table is a text pass that mirrors `own.py`'s rule (`cant` or own faint ⇒ no label).
  It was not re-run through the X50 extractor itself, so the exact X50 shares may differ slightly.
- **The guard simulation.** B1 refills moves by Smogon marginals without C-GAME's teammate and legality conditioning.
  The real recall could differ by a few points; it cannot approach the ≥ 90 % a guard needs.
- **Forum216b.** It was not read. The excluded share near it is unmeasured.
