# HUMAN DATA in the system: what is in use, the sources, the snapshots, the switch, and what we know

**Status: ALWAYS-CURRENT (audited 2026-10-27 against everything since 10-25).** Eight adversarial reviews shaped the
human-data work: A (data / bias / leakage), B (experiments / causal), C (reasoning) and D (the re-review) on this doc
and the lever plan; E (the X59-S registration), G (X59-S: data and selection bias) and H (X59-S: causal) on the
matched-student contest; F on X60's redesign. Dispositions: [`RESPONSE.md`](../research_state/measurements/human_replay_quality_2026-10-25/RESPONSE.md)
for A to E; the X59-S [`PREREG.md`](../research_state/measurements/x59s_matched_students_2026-10-25/PREREG.md)'s
amendments for E, G and H; F's are owed (Q-H11). The doc is **ALWAYS-CURRENT**: every change that touches human data
updates it and its Decision record in the same commit, and
[`src/human_data_doc_gate_test.py`](../../src/human_data_doc_gate_test.py) pins its status box, snapshot table and
switch state to the registry, the card and the code. The plan and its pre-registrations are the explicit-only
[`design_human_replay_lever.md`](design_human_replay_lever.md); the measured data card is
[`../research_state/measurements/human_replay_quality_2026-10-25/`](../research_state/measurements/human_replay_quality_2026-10-25/README.md).

**Tense rule.** A sentence in the present tense describes something that exists today. Everything not built is marked
**PROPOSED**.

## 0. STATUS (pinned by `src/human_data_doc_gate_test.py`)

```
STATUS
  human decisions in training / eval / play     : X59-S's 5 S-H student networks only (offline) / none / none
  human-built teams in training                 : yes (the 719-team pool; --team-mix corpora when typed)
  replay-derived teams in eval                  : yes (heldout72_hpfix = Metamon hl_05_26)
  human-trained models                          : Kakuna, SmallRL, LargeRL, LargeIL, PastaMittens (eval anchors); the HOM and X59-S's 15 human-row students S-H / S-P / S-P0 (offline only)
  human aggregates                              : Smogon usage / sets / spreads (every prior; the C-GAME fill)
  switch --human-data                           : NOT BUILT
  snapshot hr_spec_v1                           : 376,410 logs; durable copy yes
  snapshot hr_arch_2026-10-24                   : 16,490 logs; durable copy yes
  rowsets / models registered                   : 2 / 0
  rowset in use (X59-S)                         : hr_spec_v1.r2, 1,201,538 train / 29,904 validation rows
  informative rows, actor >= 1500               : 1.80M
  >= 1500 full-six sides unfamiliar to the pool (<= 2 fits; a lower bound) : 15.9 %
  HOM - P* NLL on accounts unseen in training   : -0.503 nats
  human-only models vs P* (Q-H10)               : P* wins all three (U 0.666 / 0.631 / 0.790)
  fill artefact (FR-2)                          : NO-ARTEFACT, which gates nothing (Q-H5)
  human vs self-play data (Q-H9, X59-S)         : SELF-PLAY-DATA-BETTER, Δ_data -7.28 pp [-10.29, -4.28]; in-house -9.06; closure pending Δ_restriction
  evidence that human data helps goal (b)       : none; at matched volume human rows teach a fresh network LESS than our self-play rows
  switch values (PROPOSED)                      : bc = an extra loss term | teams = more teams in the pool | kl = a pull toward a human model
  open                                          : whether X59-S's closure stands (Δ_restriction, playing); rating at fixed slice (the ≥ 1500 follow-up, registered); does ONE human lever raise P* (X60: REVIEW F's fixes owed); how the spectator stream was selected (Q-H4)
```

**What the box says, in words.** No human DECISION reaches a production run, an evaluation or play today. The one
exception is an experiment: on 2026-10-26 X59-S trained 20 small offline student networks to answer whether human
data beats our self-play data (§7.1): 5 learn human decisions (S-H), 10 learn P\*'s choices on human states (S-P,
S-P0), 5 learn self-play (S-S). None is a run under `models/` and none seeds one. Human data is nevertheless already
in the system in four other forms (§2): the teams we train on were built by people, one evaluation panel was
extracted from ladder replays, our external meters were trained on human replays, and every prior is Smogon's
aggregate of human ladder games. The switch that would let human decisions train the policy is not built. **There is
no evidence yet, either way, that human data would help on unfamiliar teams:** the 15.9 % line says human teams
REACH that region (coverage), the −0.503 line says human choices are LEARNABLE, and upstream's human-only models all
lose to our P\* (Q-H10). **X59-S has now read inside our network, and the answer is no at this volume** (2026-10-30):
a fresh student taught by 1300+ ladder players' games and choices plays unfamiliar teams 7.28 pp [4.28, 10.29] WORSE
against Kakuna than one taught by our plateau policy's own games and choices, the in-house meter agrees (−9.06 pp), and
the human labels lose to P\*'s on the same human states (−4.50 pp). H-BC is dropped; whether the read closes this slice
of human data waits on one registered guard (Δ_restriction, §7.1).

### 0.1 The named questions at a glance

One line each; the numbers and the mechanisms are in §7. **Status:** *answered* = measured and reviewed;
*in flight* = a registered read is running; *open* = unmeasured, or a decision not yet made. **Evidence** = the
commit that landed the measurement (`git show <sha>`).

| id | the question, in plain words | the answer now | status | evidence |
|---|---|---|---|---|
| **Q-H1** COLLECTION | How do we get more human games, at what rate, legally? | Archive: ≈ 365 a day by the daily timer (skewed). Spectating gave 3,497 a day until 09-23; reviving it, or backfilling the archive to 2020 (≈ 0.5M games, ESTIMATE), needs the owner's word. | answered (rates); open (the decision is the owner's) | `431b1fdf7` |
| **Q-H2** SNAPSHOTS | Which data exactly did a result use? | Every result cites a frozen, versioned id (§5): 2 snapshots (every log re-hashed on every read; a mismatch REFUSES) and 2 rowsets (their manifest digests are checked before any student trains). | answered, built | `431b1fdf7`, `e73c8d4bd`, `a1370436f` |
| **Q-H3** QUALITY | Are the logs parseable, honest and free of bots? | Yes under one rule set: 0 parse errors, 4 duplicate bodies, ≈ 6 % of sides from rule-flagged accounts. The Kakuna-family audit found no account that plays like Kakuna; its registered rule still dropped 11 accounts (1.73 % of rows). | answered | `431b1fdf7`, `a1370436f` |
| **Q-H4** SELECTION | Is the sample the ladder? | Archive: NO, SKEWED (+35 to +128 Elo, 26–41 % longer games, 10–20 pp fewer forfeits). Spectator: unmeasured (≈ 50 % of games, ESTIMATE), and forfeit-heavy (53–56 % of games end in a forfeit, against 34–46 % in uploads). | answered (archive); open (spectator) | `fd7f46243`, `431b1fdf7` |
| **Q-H5** LABELS / FILL | How often is the chosen action known, and how much of a row is invented? | Known for 90.3 % of decisions; ≈ 52 % of the actor's own move slots are Smogon draws. FR-2 (the fill-artefact check): filling hurts the HOM less than it hurts P\* (NO-ARTEFACT, D −0.149 nats [−0.163, −0.134]), but that test cannot see "a drawn option is never the label", so it gates nothing. | answered, with that limit | `431b1fdf7`, `e652875c6`, `eadacd559`, `b27226c3a` |
| **Q-H6** COVERAGE | Do human teams reach the unfamiliar-team region? | Partly: ≥ 15.9 % of full-six ≥ 1500 human sides. The panel-exclusion rule keeps 16.8 % of candidate battles, a forfeit-heavy slice (X59-S's rows: 50.6 % from forfeits, median turn 15 against self-play's 19). | answered | `431b1fdf7`, `a1370436f`, `b27226c3a` |
| **Q-H7** LEARNABILITY | Can a model learn human choices? | Yes: the human opponent model (HOM, P\* fine-tuned on human choices) beats P\* by 0.503 nats [0.467, 0.542] on accounts it never saw. Learnable is not the same as useful (Q-H9). | answered | `92af1f121`, `431b1fdf7` |
| **Q-H8** DRIFT | Has the meta or the population moved? | Slow and small (0.005–0.011 a month beyond account noise); no new snapshot has been cut, so the alarm has not been read since. | answered (for `hr_spec_v1`'s months) | `431b1fdf7` |
| **Q-H9** HUMAN vs SELF-PLAY DATA | Is human data better than our own self-play data, at matched volume and recipe? | **NO: SELF-PLAY-DATA-BETTER.** Δ_data = S-H − S-S −7.28 pp [−10.29, −4.28] on U vs Kakuna (S-H 0.320, S-S 0.393), in-house −9.06 [−10.98, −7.14], F alone −8.80; Δ_labels OWN-LABELS-BETTER −4.50 [−7.42, −1.58] (H-BC dropped). The closure ("1300+ ladder data in the panel-distant, forfeit-heavy slice, at this mix and volume, as a BC target for a fresh network at 3 epochs") is pending Δ_restriction (playing); the ≥ 1500 follow-up fires | answered (closure pending) | X59-S README; record `~/gen3ai_archive/x59s/read/full/record.json` |
| **Q-H10** HUMAN-ONLY MODEL | Does a model trained only on human data beat ours? | NO. All three of Metamon's human-only models lose to P\*: U 0.666 / 0.631 / 0.790 (small RL / large RL / large imitation); offline RL beats imitation on the same data by ≈ 16 pp. These are upstream's recipes, not ours. In our architecture (X59-S, secondary, cross-engine) a human-only student sits 16.7 pp under P\*'s X51 twins, the self-play student 9.4 | answered | `4d8d9eef8`, `df35d1bb0`, X59-S README |
| **Q-H11** DOES A LEVER HELP? | Does ONE human clause added to P\* raise its win rate against Kakuna on human-built teams it never trained on? | Unmeasured. X60 is redesigned with both routes pre-written; REVIEW F: READY-WITH-FIXES, 0 BLOCKERs and 9 MAJORs, none yet applied. The headline "GAIN 0.80 at +3 pp" is the zero-seed-noise cell: 0.70 at a 1.0 pp floor, 0.27 at the measured upper bound of 2.75 pp. | open | `e1512c361`, `e41beafe1` |
| **Q-H12** TOGGLE INTEGRITY | With every lever off, is the system unchanged? | Trivially yes: the switch is not built. REVIEW F wants one throwing guard now: a human-data team corpus must not enter through `--team-mix` unannounced. | answered (trivially); the guard is open | `e41beafe1` |
| **Q-H13** CONTAMINATION | Which meters already carry human data, and what does a lever's gain on them mean? | Every panel is team-clean; the POPULATION differs: H (replay-built) is the human levers' own, P72 is fill-adjacent, F (forum-built) is the cleanest. No panel is shown Kakuna-familiar. Kakuna's packer bug (a Gyarados playing Frustration at power 1 on F) is fixed, so F reads before 10-27 are a different regime. | answered | `4f555e577`, `b0441623e`, `431b1fdf7` |
| **Q-H14** PROVENANCE | Given a checkpoint, did any ancestor train on human data? | No production run did. The only networks of ours trained on human rows are the HOM and X59-S's 15 human-row students, offline and outside `models/`. Lineage tooling is PROPOSED with the switch. | answered (today); the tooling is open | `431b1fdf7` |
| **Q-H15** TEACHER QUALITY | How strong is the population we would imitate? | Upstream's human-only models lose to P\* and sit 14–30 pp under Kakuna. Inside our architecture the 1300+ slice (77.5 % rated 1300–1499) is a worse cloning target than P\*: human labels lose to P\*'s on the same states by 4.50 pp [1.58, 7.42] (X59-S Δ_labels). The ≥ 1500 stratum at fixed slice and volume: the follow-up, registered | answered (1300+ slice); follow-up (≥ 1500) | `4d8d9eef8`, `df35d1bb0`, X59-S README |

**Names used below.** P\* = our plateau policy (`rb_x26_s2601` at 280M steps); Kakuna = the Metamon agent (checkpoint 34) we play
as goal (b)'s opponent, itself pretrained on human replays. H / F / P72 = the three unfamiliar-team panels (teams extracted from ladder
replays / built by forum members / procedural); U = their equal-weight mean; home72 = our own sample teams, which are in
the pool. HOM = the human opponent model. C-GAME = the completion of a seat's hidden facts from Smogon's prior.
X59-S = the matched-student contest; its students are **S-H** (taught by humans: human states, human choices),
**S-P** (taught by P\* on the SAME human states, P\*'s choice restricted to what the log allows the human), **S-S**
(taught by our self-play: P\*'s own games and choices) and **S-P0** (S-P without the restriction). X60 = the planned
test of adding ONE human clause to P\*; **H-BC** = an extra loss on human choices (behaviour cloning); **H-TEAMS** =
human-built teams added to the self-play pool. FR-1 / FR-3 = P\* against Metamon's human-only models; FR-2 = the
fill-artefact check. pp = percentage points; nats = natural-log units of prediction loss (lower predicts better).

## 1. Which result read which data

| result | data | id it cites | where |
|---|---|---|---|
| X50 R1, R0, R-b1, R-b2 | spectator logs | `hr_spec_v1` (retroactively: the reads ran on exactly its paths; train ≤ 09-20, held out 09-21..23) | `design_x50_rung1b_human_opponent_model.md` |
| the HOM (human opponent model) | spectator, actor ≥ 1500, heldout72 ≥ 4-fit exclusion | `hr_spec_v1`; its rowset predates the rowset registry (unregistered); **barred from every lever and routing rule** (its rows kept 3-set cores of the panels; REVIEW D M5); since 10-27 also FR-2's instrument | `~/gen3ai_archive/x50_r1b/hom/` |
| this data card | spectator + archive | `hr_spec_v1`, `hr_arch_2026-10-24`, stamped in `results.json` | the card |
| the archive-skew read | archive pulls before the registry | `none` (predates; cite the folder) | [`replay_archive_bias_2026-10-21`](../research_state/measurements/replay_archive_bias_2026-10-21/README.md) |
| X59-S: the Kakuna-family audit, the 20 students, the reads | spectator rows | rowset `hr_spec_v1.r2` (rows manifest `2a413c85…`; the audit reads its parent `hr_spec_v1.r1`); S-S's rows are self-play, no human | the X59-S PREREG |
| FR-2 (the fill-artefact check) | our own eval traces, the sentinel's and bots' seats, read by the HOM and P\* | no snapshot (no human rows read); the HOM is `hr_spec_v1`'s | `results/fr2.json` beside the PREREG |
| FR-1 / FR-3 (human-only models vs P\*) | no human rows of ours; the opponents were trained upstream on human replays | `none` | [`human_only_vs_pstar_2026-10-25`](../research_state/measurements/human_only_vs_pstar_2026-10-25/README.md) |
| every self-play, Kakuna and panel result | no human decisions | `none` (read §2 for what IS human in them) | — |

## 2. The inventory: every human-derived thing in use today

| kind | in use today | where | its switch | what it means for a result |
|---|---|---|---|---|
| human-BUILT teams | **training**; **evaluation** | the 719-team pool = PokePaste collections of five community builders (`data/teams/others/`; four are listed in its `sources.json`) + Smogon sample teams (`data/teams/sample/`); D1's forum corpus (`src/utils/team_corpora/w_forum_v1/`) only when `--team-mix` is typed; the `forum72` panel (72 forum-built teams) | none for the pool (it IS the baseline); `--team-mix` | self-play pilots human-designed teams; "no human data" never meant "no human artefacts" |
| REPLAY-derived teams | **evaluation** | `heldout72_hpfix` = Metamon's `hl_05_26`: teams extracted from ladder replays, hidden sets filled from usage | none | a human-data lever trains on H's POPULATION: a gain on H alone is "learned the ladder", not generalisation (Q-H13) |
| human AGGREGATES | **every prior** | Smogon usage / sets / teammates / spreads (`data/pokemon/`), read through `agents.gen3_data`; the C-GAME fill; P72's generator; the procedural corpora built from them (`n1_proc_v1`, X61, when `--team-mix` names it) | none | allowed by the root priors rule; P72 and every C-GAME fill draw from the SAME tables (Q-H13) |
| human-TRAINED models | **evaluation**; offline | Kakuna (Superkazam: self-play + 4M human replays), `metamon:SmallRL`, `LargeRL` (human-only offline RL) and `LargeIL` (human-only imitation), PastaMittens (a Kakuna-family router) in `designs/ops/anchors.json`; the HOM and X59-S's S-H / S-P / S-P0 students, which learn from human rows (offline measurement only) | none | Kakuna is human-pretrained, so it is not a neutral meter for human-likeness (B-M6) |
| human DECISIONS (rows) | **offline only**: the HOM (10-24) and X59-S's 5 S-H students (10-26); none in a run, an evaluation or play | `hr_spec_v1`, `hr_arch_2026-10-24`; rowset `hr_spec_v1.r2` | `--human-data` (**NOT BUILT**); the student trainer reads one registered rowset and nothing else reads it | the rest of this doc |

**The priors rule, stated once.** The root `CLAUDE.md` rule governs what the network READS: it traces to Smogon,
ground-truth labels or ladder replays, and a team pool never ships as a prior. The human-data switch adds nothing the
network reads: replay rows enter only as a TRAINING SIGNAL (a label in a loss, a team to play) or as a measurement.
A replay-derived TABLE the network would read is a separate decision under the root rule, never a value of this switch.
Every hidden fact a replay lacks is filled from Smogon only, never from our pool or from a later reveal of the opponent.

## 3. The mental model

```
  WHO PLAYED ↓ / WHAT WE KNOW →   public log only           the actor's team true          both teams true
  humans (ladder)                 (a) archive               (c1) a submitted replay +       (c2) tournament replays
                                  (b) spectator                  the submitter's paste           + posted team sheets
  our agent vs bots / Kakuna      —                         —                               (c-ours) eval traces
  our agent vs itself             —                         —                               (d) self-play

        │ (a),(b): the actor's hidden facts are FILLED from Smogon (C-GAME); the chosen action is real
        ▼
  SNAPSHOT (frozen logs) ──► ROWSET (filtered rows / teams) ──► MODEL (e.g. a HOM)
        each id registered in designs/research_state/human_replay_snapshots.json (§5)
        ▼
  ONE switch per run, OFF by default (PROPOSED): bc:<rowset> | teams:<rowset> | kl:<model>     (§6)
  eval and play never read human decisions
```

**Four facts carry most of the reasoning.**
1. **Information level decides how much of a row is invented.** A self-play row is exact. In a replay row the ACTOR's
   own side is largely invented: at ≥ 1500 a mon shows ≈ 1.9 of its 4 moves, its EVs never. The chosen action is real.
2. **Selection decides what population a source is.** The archive is the uploaders' choice (SKEWED); the spectator
   stream covered public rooms at the daemon's capacity (≈ 50 % of ladder games, ESTIMATE) and is forfeit-heavy; our
   panel exclusion then keeps a further forfeit-heavy slice (50.6 % of X59-S's rows); self-play is our pool only.
3. **Coverage decides whether a source can help goal (b).** Self-play never pilots a team outside the pool; ≥ 15.9 %
   of full-six ≥ 1500 human sides sit in the region the unfamiliar panels occupy. Most of the near-panel region must be
   EXCLUDED from training to keep the panels unfamiliar (the plan §3.1).
4. **There is no offline evidence either way** that human labels carry the unfamiliar-team skill goal (b) lacks. The
   one proxy (P\*'s human-prediction NLL by team familiarity) is UNINFORMATIVE: it predicts the same flat profile
   whether that hypothesis is true or false (REVIEW A M1). Upstream's human-only models lose to P\* (Q-H10), which
   compares recipes, not data. X59-S compares data: SELF-PLAY-DATA-BETTER (§7.1).

## 4. The source taxonomy: one data card per source

**Rule beside the table:** a known OPPONENT team never enters the actor's observation; it is a label or a measurement
only (an oracle leak otherwise, the reveal spec's territory). A source may never be pooled with another.

| source (version id) | actor side / opponent side | selection (who chose that this game is in our data) | volume | use |
|---|---|---|---|---|
| **(a) ARCHIVE**: public uploads through the owner's proxy (`hr_arch_2026-10-24`) | public / public | the UPLOADER, after the game: +35 to +128 Elo, 26–41 % longer, 10–20 pp fewer forfeits (34–46 % of games end in one, against 53–56 % on the spectator's days); one account up to 10 % of a window | 365 games / day (160–714) | drift and collection reads only; **DROP for training** (reopened only by a registered coverage read and its own snapshot) |
| **(b) SPECTATOR**: games the deleted daemon watched, 2026-05-18 → 09-23 (`hr_spec_v1`) | public / public | public rooms only (private never listed), a 30 s poll, ≤ 20 rooms at once, rooms > 1 h dropped, a reaper added 07-05: ≈ 50 % of ladder games (ESTIMATE), unmeasured by length / rating / month | 3,497 games / day while it ran | the nearest ladder sample we hold, NOT a verified ladder sample (Q-H4); the source of every row any lever or student uses |
| **(c1) SUBMITTED**: a replay a person gives us with their own team paste (PROPOSED intake) | true / public | the submitter (opt-in; consent recorded) | owner and friends: tens of games | the fill error on HUMAN teams (A-2's test); never a policy-lever source until registered |
| **(c2) TEAM SHEETS**: tournament replays matched to posted teams (PROPOSED) | true / true | the tournament | unknown | the fill-error gold standard; opponent truth is a label only |
| **(c-ours) EVAL TRACES**: our games vs bots / Kakuna (`models/<run>/eval_traces/`) | true / true | ours (pool, panels) | per eval | measures the fill error on OUR teams (R0: +0.012 to +0.019 nats); FR-2 reads the sentinel's and the bots' seats |
| **(d) SELF-PLAY** | true / true | the 719-team pool | unbounded | the baseline; the familiarity premium ≈ 11 pp (X51); X59-S's S-S rows (self-play under the human panel rule) |

### 4.1 Entry points: where each source enters the code today

| source | enters through | toggle |
|---|---|---|
| (a) archive | `tools/replay_corpus_downloader/` (`gen3ai-replay-daily.timer`, through the owner's proxy: download only; [`replay_corpus.md`](../tools/replay_corpus.md)) into the gitignored `replays/showdown/gen3ou/` | none; a read cites `hr_arch_2026-10-24` |
| (b) spectator | no collector today (the daemon is deleted; reviving it needs the owner's word); its 376,410 logs sit in the same corpus | none |
| any snapshot → rows | `src/main/human_data/snapshots.py` verifies the snapshot; `python -m main.human_data.rowset scan \| extract \| compact` makes a rowset; `src/main/human_data/derive.py` derives r2 from r1; `src/main/human_data/audit.py` is the Kakuna-family audit | the rowset id (`hr_spec_v1.r<n>`) |
| rows → a model | `python -m main.human_data.student train \| export` inside a GPU lease (`src/main/human_data/window.py`); reads: `src/main/human_data/x59s_read.py`, `python -m main.human_data.x59s_ops`, `python -m main.reads`; the ≥ 1500 follow-up's rows, arm-chain window and in-house read: `python -m main.human_data.followup` (X59-S PREREG Amendment 7) | none: it never touches a run |
| (c-ours) eval traces | `python -m main.x50_r1.run` (spectator views and rows) and `python -m main.x50_r1.read_fr2` | none |
| (d) self-play, as X59-S's comparison | `python -m main.human_data.ss_rule` (the panel rule on self-play team draws) | none |
| human teams into training | `--team-mix` corpora (`src/utils/team_corpora/`: D1's forum corpus today; `n1_hrep_v1` PROPOSED) | `--team-mix`; REVIEW F F-M7: one corpus must not have two entry points (§9 item 6) |
| a human loss on a run | **none**: `--human-data` is NOT BUILT and `agents.info_levers` has no human-data module | `--human-data` (PROPOSED) |
| human-trained opponents | `python -m main.anchors` and `python -m main.reads`'s anchor lane (Kakuna, Metamon's models; [`EXTERNAL_ANCHORS_SOP.md`](../ops/EXTERNAL_ANCHORS_SOP.md)) | none |

**The (c1) intake (PROPOSED).** A person drops a replay plus their paste into `~/gen3ai_archive/human_snapshots/intake/`
with a consent line; a cut freezes the folder as snapshot `hr_sub_<tag>` (`actor_side: true`, `opp_side: public`);
the account is stored only as a salted hash; a per-account cap applies; it is never pooled with (a) or (b).

## 5. Snapshots → rowsets → models

```
  SNAPSHOT  raw logs, frozen                hr_spec_v1                     registry "snapshots" (exists)
     │  + the filter spec (rating floor, bot rule, walk-over bar, per-account cap, the panel-exclusion RULE and SET,
     │    the hidden-turn label rule, the account split) + C-GAME commit and seed + extraction commit + layout hash
     ▼
  ROWSET    training rows or teams          <snapshot>.r<n>  (rows manifest sha256)   registry "rowsets" (hr_spec_v1.r1, .r2)
     │  a fine-tune or a from-scratch fit on the rowset
     ▼
  MODEL     e.g. a HOM                      hom_<rowset>  (checkpoint sha256)         registry "models" (empty)
  the switch names the most-derived id it uses: bc:<rowset>@<coef> | teams:<rowset>@<share> | kl:<model>@<coef>
```

A citation names the most-derived id it used; the registry resolves the chain down to the snapshot. **A new panel
registration RETIRES every rowset whose exclusion set it changes** (each rowset row records its panel set; the gate
checks it against `designs/research_state/heldout_panels.json`).

**The snapshots** (registry [`human_replay_snapshots.json`](../research_state/human_replay_snapshots.json); manifests of
(relative path, sha256, bytes) per log in `~/gen3ai_archive/human_replay_snapshots/`):

| id | population | files | dates (Los Angeles) | manifest sha256 | durable copy | status | use |
|---|---|---:|---|---|---|---|---|
| `hr_spec_v1` | spectator | 376,410 | 2026-05-18 → 2026-09-23 | `e7d491e7f7a1b98093360ece7929eca0d2880c0ffc5706244edee4700935f3b1` | yes | reference | X50 R1 / R0 / R-b1 / R-b2, this card, X59-S; TRAIN ≤ 09-20, HELD OUT 09-21..23 |
| `hr_arch_2026-10-24` | archive | 16,490 | 2026-06-16 → 2026-10-24 | `ad940108978dd6a52a72cf2d637cc108c652d3eaa5c673346e8351c0f1f2f7a2` | yes | measurement | drift and collection reads only |

- **The manifest IS the definition.** Late uploads land in old date folders, so "as on disk" is not reproducible.
- **Verification is one reader**, `src/main/human_data/snapshots.py`: it reads the manifest, checks its digest and row
  count against the registry, re-hashes every log and checks the population's header, and REFUSES on any mismatch
  with an ordinary exception, so a refusal inside a worker pool ends the job with a non-zero exit instead of hanging
  (REVIEW D M1). The card's `hrq.py extract` (whose cache and `results.json` carry `{snapshot_id, manifest_sha256,
  extraction_commit, hrq_dirty, hrq_sha256}`; a reused cached day is re-hashed) and the rowset builder both use it.
  **PROPOSED:** the same check inside `agents.info_levers.human_data`, refusing any consumer whose stamp the registry
  does not hold.
- **The rowset builder** (`python -m main.human_data.rowset scan | extract | compact`) applies the plan's §3 rules in
  order, every battle-level one on the log before C-GAME fills anything, and writes a rows manifest (every shard's
  sha256, the counts of every rule, the kept / excluded population profile, the account and rating profile of the
  rows, and a `source_sha256` over every file that makes a row); compact refuses parts of different code. **The
  rowsets:**

  | id | snapshot | rule | rows (train / validation) | rows manifest sha256 | status | use |
  |---|---|---|---:|---|---|---|
  | `hr_spec_v1.r1` | `hr_spec_v1` | TRAIN dates, rated, ≥ 11 turns, no duplicate, no bot, the ≥ 3-fit exclusion over all five held-out panels, actor ≥ 1300, ≥ 2 legal, hidden moves as set labels, the consistent set per row, quitter's last 2 decisions dropped, account-disjoint validation | 1,330,809 / 32,102 (8.2 % hidden-move rows; 78 % from 1300–1499 actors) | `e9e0df4224b589798dc507d005c60bcaa8ede9744e1e7403901664899e8fe492` | measurement | superseded for training by r2; the Kakuna-family audit's universe |
  | `hr_spec_v1.r2` | `hr_spec_v1` | r1 minus the quitter's last 10 decisions (a forfeit or timer loser's play degrades near the end) and minus the 11 accounts the audit's registered rule flagged (1.73 % of r1's rows) | 1,201,538 / 29,904 (7.7 % hidden-move rows; 77.5 % from 1300–1499 actors; 270,260 rows at ≥ 1500; endings: 45.9 % KO, 50.6 % forfeit, 2.9 % timer, 0.6 % tie) | `2a413c85e2109553b4f930d85f369625a5daf77a7e1103afeb01e962ec0332b8` | measurement | X59-S: the S-H, S-P and S-P0 rows; S-S's row counts |
- **No student is a registered model.** X59-S's 20 students are offline measurement networks (their digests are in
  the read's `students.json` under `~/gen3ai_archive/x59s/`); the model registry stays empty until a human-trained
  model is used by a lever.
- **The durable copy.** Each snapshot's bytes lived once, in the main checkout's gitignored `replays/`. Each has a copy,
  `~/gen3ai_archive/human_snapshots/<id>/logs.tar.zst` beside its manifest (`hrq.py archive`), every member re-hashed
  after writing; both locations and the archive's sha256 are in the registry row. The originals were only read.
- **Two live at most:** one `reference`, one `candidate`; others are `measurement` or `retired` (the gate counts).
- **A new snapshot** is cut on the drift alarm (§8) or at +50 % usable volume; it is a new treatment, never a replicate.
- **Accounts in committed artifacts are salted hashes** (`hrq.acct`); the salt and the name map stay in
  `~/gen3ai_archive/human_snapshots/`.

## 6. The switch (PROPOSED, NOT BUILT)

**The grammar, the one place it is defined:** `--human-data off` | `bc:<rowset>@<coef>` | `teams:<rowset>@<share>` |
`kl:<model>@<coef>`.

| place | today | with a lever on (PROPOSED) |
|---|---|---|
| the observation, inputs, priors | no human decisions; priors Smogon | unchanged, always |
| the PPO loss | self-play | `bc`: + a BC term on a rowset's rows; `kl`: + a KL to a registered model on self-play states |
| the self-play team pool | the 719-team pool (+ `--team-mix` corpora) | `teams`: + a rowset's completed, panel-excluded ladder teams |
| search | our network as the judge | nothing: `judge` (the HOM as search's opponent likelihood) left the grammar when R-b2 read HOM-NONE; reopening it is a new registration |
| eval, the panels, Kakuna | no human decisions | no human decisions, ever |
| offline measurement | X50 reads, the card, drift, X59-S's students | any read citing its snapshot / rowset / model id |

- **The complexity budget.** ONE clause per run; the clause's ONE parameter; every other knob (the filters, a
  `coef(t)` schedule, a human-minibatch share, a per-account cap) lives in the ROWSET or in a pre-registration, never
  on the command line. Two levers at once is a new registered lever value replacing a row, never a second flag. At most
  three lever heads (`bc`, `teams`, `kl`); a new one replaces a row with a decision-record entry.
- **Provenance (PROPOSED).** The resolved config, `model_config.json`, `metadata.json` and the immutable lineage carry
  the clause and its hash (`sha256("human_data_v1:" + canonical)[:16]`). **Resuming or forking a run whose lineage has
  a human-data clause REFUSES unless `--human-data` is typed** (the clause or `off`), so nothing inherits or drops a
  lever silently. `main.lineage` shows the first human-data ancestor; `main.baselines set` refuses one without a reason.
- **One entry point per corpus (PROPOSED guard, REVIEW F F-M7).** `--team-mix` refuses a corpus whose manifest says
  `"human_data": true` unless the argv also carries the matching `--human-data teams:` clause (or the switch is the
  only way in), with a test that fails on revert; written before either arm launches. Not built.
- **What it is NOT.** It reveals nothing about the battle being played and is never combined with `--oracle-reveal`.
- **Off means off (PROPOSED gates).** With `off`, the training smoke runs under an `open()` audit that fails on any path
  under the corpus, a snapshot or rowset store; `learner_golden` and the played-bits sentinel are unchanged by the build;
  a null-dose ON build (coefficient or share 0) reproduces OFF's played bits.
- **Production:** `designs/production_config.json` has no `human_data` key today; when built it resolves `off`.

## 7. The questions and their current answers

§0.1 has the one-line answers, statuses and evidence commits; this section holds the numbers and the mechanisms.

### 7.1 The owner's two questions (Q-H9, Q-H10) and the instrument check (FR-2)

**Q-H10 HUMAN-ONLY MODEL: does a model trained only on human data beat ours? ANSWERED, NO.** P\* against each of
Metamon's three human-only models, greedy against greedy, 1,600 games each, every game twinning X51's P\* against
Kakuna game of the same panel, seed and half (240 / 240 units counted):

| opponent | P\*'s win rate on U [95 %] | verdict | home72 | weaker than Kakuna on U by |
|---|---|---|---|---|
| `SmallRL` (14M, offline RL) | 0.666 [0.639, 0.692] | OURS-BEATS | 0.785 | +0.179 [+0.140, +0.218] |
| `LargeRL` (195M, offline RL) | 0.631 [0.603, 0.658] | OURS-BEATS | 0.700 | +0.144 [+0.105, +0.183] |
| `LargeIL` (194M, imitation) | 0.790 [0.766, 0.812] | OURS-BEATS | 0.785 | +0.303 [+0.266, +0.339] |

Offline RL beats imitation on the same human data by 0.159 [0.123, 0.195]. The three unfamiliar panels are level within
≈ 4 pp against each of them, so P\*'s P72 deficit against Kakuna (0.433) is something Kakuna exploits, not a general
weakness. **It cannot say** that human data loses inside OUR network: these are upstream's recipes, which differ from
P\* in architecture and scale as well as data ([`human_only_vs_pstar_2026-10-25`](../research_state/measurements/human_only_vs_pstar_2026-10-25/README.md);
`4d8d9eef8`, `df35d1bb0`).

**FR-2, the fill-artefact check (Q-H5's instrument): READ, NO-ARTEFACT; it gates nothing.** The HOM trained on filled
rows, P\* on true ones. FR-2 measures how much worse each predicts a seat's real choices when its hidden facts are
filled from Smogon than when they are true, on the sentinel's and the bots' seats of our eval traces (2,091 battles,
62,415 decisions; 33 battles with a silently ending volatile left out by a rule registered before the read, Amendment
4c). The fill costs the HOM 0.069 nats and P\* 0.218: **D = −0.149 [−0.163, −0.134]** against an ARTEFACT bar of +0.05.
REVIEW G's finding (answered by Amendment 6): the completion uses the whole game's reveals, so a model that had learned
"a drawn option is never the label" would ALSO show a smaller gap. The test answers rung 1b's question (does completion
cost the HOM more than P\*) and certifies nothing about that signature; the closure's guard is Δ_restriction below
(`results/fr2.json`; `e652875c6`, `eadacd559`).

**Q-H9 HUMAN vs SELF-PLAY DATA: is human data better than our self-play data? READ 2026-10-30 (X59-S): NO, SELF-PLAY-DATA-BETTER; the closure is pending Δ_restriction.** Students of our
architecture, five seeds each, one initial network a seed, the same 7,041 steps: **S-H** (human states, human choices),
**S-P** (the SAME states, P\*'s choice restricted to what the log allows the human), **S-S** (P\*'s own self-play games
and choices: a distillation of our plateau policy, the best case for self-play data at this volume, not the RL rollout
data that trains P\*) and **S-P0** (S-P without the restriction; descriptive). **Δ_data = S-H − S-S answers the
owner's question**; Δ_labels = S-H − S-P asks whether a human teaches better than P\* on human states (it routes
H-BC); S-P − S-S is the states part (it routes nothing). Each student plays Kakuna on the three unfamiliar panels and
its S-S and S-P twins in-house, the meter no human-trained agent shapes.

**The verdict** ([README](../research_state/measurements/x59s_matched_students_2026-10-25/README.md); record
`~/gen3ai_archive/x59s/read/full/record.json`, COMPLETE, 901 Kakuna units on one engine):
- **Δ_data SELF-PLAY-DATA-BETTER, −7.28 pp [−10.29, −4.28]** (S-H 0.320, S-S 0.393 on U vs Kakuna); the in-house meter
  agrees (−9.06 [−10.98, −7.14]), and so does F alone (−8.80 [−12.25, −5.35]).
- **Δ_labels OWN-LABELS-BETTER, −4.50 pp [−7.42, −1.58]** (S-P 0.365; in-house −5.23): **H-BC is dropped.**
- Δ_states UNCLEAR, −2.78 [−5.68, +0.12]. σ̂ 1.47 pp. S-H fitted; re-encode 0.60 %; the deficit is not concentrated in
  long games (terciles −7.0 / −10.2 / −4.6 pp).
- **The closure waits on Δ_restriction** (S-P0 − S-P, 300 units, playing): a lower bound ≤ 0 closes the slice as
  quoted below; > 0 closes nothing. The ≥ 1500 follow-up (rating at fixed slice and volume) is registered next.

How it was read (2026-10-27):
- **Inputs.** Human rows `hr_spec_v1.r2`; S-S's rows regenerated under the human panel rule (a simulated human reveal,
  redrawn while near a panel: 16.8 % of pool team pairs pass, as 16.8 % of human candidate battles did; 15,069 games).
  Registered and checked (`registration_a2.json`: PASS).
- **Training.** The GPU pre-flight HELD the first window on 10-26 (the loader returned the policy in training mode, a
  ≤ 5e-7 logit difference; fixed for every reader, `141f36f7f`); after the re-run passed, the 20 students (15 primary +
  5 S-P0) trained in 453 min of wall (PASS 23:04), against 3–4.5 h estimated: three at once ran 2.4× slower each
  (`ops/TECH_DEBT_BACKLOG.md` P3).
- **The read.** One engine, one pin (`89c2c2501`, tag `x59s-play-89c2c250`, a commit off main; Amendment 6), through `python -m main.reads`. The identity unit
  reproduces X51's P\* game for game (20 / 20). Floor references: a random player 0 / 240, a max-damage player 11 / 240
  (4.6 %). The floor pilot (181 units) found every contrast informative; the rest (720 units) and the in-house
  head-to-heads (30 cells × 400 games) followed, all blind until the full record.
- **Input checks.** A 200-battle seeded extraction reproduces the rowset byte for byte (8,878 / 8,878 rows). S-H and S-P
  trained on rows made by an older encoder than S-S's and the play tree's (REVIEW G G-M2); re-extracted at the pin,
  51 of 8,556 rows differ (0.60 %, all in one input group, volatile effects), under the 1.0 % trigger: a registered
  limit that costs S-H only (`dd1299109`).
- **The rules, in plain words** (code before any result; Amendments 4 to 6):
  - A contrast is UNINFORMATIVE (it reads nothing and routes "H-TEAMS first") when both arms' pilot win rates are at
    most 9.6 %, five points above the better floor reference. The rest of the Kakuna read plays only if Δ_data or
    Δ_labels is informative.
  - An interval is the wider of the registered normal interval and a Welch interval on the contrast's own 10
    students; "neither better by more than 4 pp" needs the Kakuna AND the in-house interval inside ±4 pp.
  - A human win on U opens a lever only if the forum-built panel alone agrees (else POPULATION-SPECIFIC) and the
    in-house meter agrees (else KAKUNA-SPECIFIC); a self-play win is checked against the in-house meter the same way.
  - A self-play win closes human data ONLY as "1300+ ladder data in the panel-distant, forfeit-heavy slice (50.6 % of
    its rows from forfeits; median turn 15 against self-play's 19), at this mix and volume, as a behaviour-cloning
    target for a fresh network at 3 epochs", and only if S-H fitted its data, the re-encode share stays under 1.0 %, and
    Δ_restriction = S-P0 − S-P (what "a drawn option is never the label" costs; same rows, same teacher) has been read
    and its lower bound is not above 0.
  - HUMAN-LABELS-BETTER routes H-BC only if S-P learned its teacher (a midpoint test).
  - Reported, routing nothing: Δ_data by game-length tercile (is a deficit just endgame coverage?), per-panel Δs, and
    the student-to-P\* gap (Q-H10 in our architecture; across two Rust engines, so secondary).
- **Predictions** (guesses, read by no rule): every student far below P\* (arm means 0.10–0.25 against Kakuna); Δ_data
  ≈ −1 pp (−6 to +4). The smallest effect seen at 80 % power is ≈ 4 pp for a seed noise of at most 1.5 pp, so
  NOT-RESOLVED is a live outcome.
- **Reviews.** E (NOT-READY: 2 BLOCKERs, 8 MAJORs) → Amendment 2 before any training. H (the design answers a narrower
  question than Q-H9; 1 BLOCKER, 6 MAJORs) → Amendment 5 (`aaeabbd46`). G (READY-WITH-FIXES: 0 BLOCKERs, 4 MAJORs) →
  Amendments 5 and 6 (`62dd53cb5`). Limits that stay: one dataset an arm and student-seed variance only; the read's
  forum72 plays Kakuna under the legacy pack (Q-H13), so no later F read is set against X59-S's F numbers.

### 7.2 The rest

| id | question | answer now | snapshot |
|---|---|---|---|
| **Q-H1 COLLECTION** | How do we get more, at what rate, legally? | the spectator's collector is deleted and reviving it needs the owner's word (the proxy grant is "download only", and spectating joins rooms); the proxy's 1 request / s is not the bound (≈ 43k replays / day would fit), the upload volume is; an archive backfill to 2020 is not authorised | `hr_arch_2026-10-24` |
| **Q-H2 SNAPSHOTS** | Which data exactly did a result use? | two snapshots frozen and registered (§5); the card's extraction re-hashes against the manifest and stamps its results; one durable copy each; two rowsets chained to `hr_spec_v1` | both |
| **Q-H3 QUALITY** | Parse, duplicates, bots, walk-overs? | 0 text-pass errors; one rule set; C-GAME completes 99.39 % of seats; 4 duplicate bodies; 61 rule-flagged accounts ≈ 6 % of sides (heavy human grinders among them); walk-overs 1.0–5.1 % of sides; half of all games end in a forfeit. **The Kakuna-family audit** (X59-S Amendment 2b; ≥ 1300 accounts with ≥ 10 kept games, 83 % of `hr_spec_v1.r1`'s rows): no audited account plays like Kakuna (best human Kakuna log-likelihood −1.81 a decision against ≥ −1.46 for every Kakuna pseudo-account); the audit has power against Kakuna and PastaMittens, not against SmallRL- or LargeIL-like bots; its registered rule still dropped 11 accounts (1.73 % of rows, flagged because P\* fits them badly) from `hr_spec_v1.r2`. REVIEW G adds: the human validation split leaks at the battle level (71 % of S-H's validation rows share a battle with a TRAIN seat), so the human yardsticks are optimistic and no rule reads them | `hr_spec_v1` |
| **Q-H4 SELECTION** | Is the sample the ladder? | the archive: NO, far outside the pre-registered bars and the drift control in all three windows (+35 / +101 / +128 Elo, 6–11 pp fewer rated games too); a few uploaders supply 2–10 % of all sides, and "did the uploader win" is not recoverable from a log. The spectator stream: unanswerable from data we hold (coverage ≈ 50 %, ESTIMATE; unmeasured by length, rating and month). What would answer it: `search.json` uploaded ids on ~30 spectator dates, P(spectated) by turns / rating / hour (needs the owner's approval for the pulls) | both |
| **Q-H5 LABELS / FILL** | How often is the chosen action known; how much is invented; does a model learn the invention? | known for 90.3 % of completed decisions; the hidden 12 % are missing NOT at random (a sleeping mon's labelled switch rate doubles), so the plan keeps them with a set label; ≈ 52 % of own move slots and every spread are Smogon draws, from a prior 1–17 months older than the games; a drawn move is never the label (a structural signature). FR-2 (§7.1): NO-ARTEFACT, which certifies nothing about that signature. Descriptively the students learn it, a little: the probability they put outside the moves the log allows is 0.142 (S-H) and 0.145 (S-P) against 0.174 (S-S) and 0.176 (S-P0) (REVIEW G, seed 5901) | `hr_spec_v1` |
| **Q-H6 COVERAGE** | Do human teams reach the unfamiliar region? | partly: ≥ 15.9 % of FULL-SIX ≥ 1500 human sides; humans pilot 3-set cores of most H and F teams, so the panel exclusion (≥ 3 move-revealed fits, before the fill) removes ≈ 71 % of ≥ 1500 battles without `forum216b` and 83.2 % of the candidate battles with it (a ≥ 1300 seat, ≥ 11 turns). What it KEEPS is a different population: 64 % forfeits or timer losses (49 % among the excluded), 52 % full-six sides (75 %); it keeps the pool-unfamiliar region better than proportionally (REVIEW D M7). In X59-S's training rows: 44 rows a battle against self-play's 82, median turn 15 against 19, 2.9 % of rows with no legal switch against 6.6 %, 50.6 % from forfeits against none: a self-play win partly reads "endgame coverage" (REVIEW G G-M4; the length split is registered) | `hr_spec_v1` |
| **Q-H7 LEARNABILITY** | Can a model learn human choices? | yes: the HOM (X26's network fine-tuned from P\* on human choices of actors ≥ 1500; sha256 `76e29a01…`; barred from every lever and routing rule, and closed as search's judge by R-b2's HOM-NONE) beats P\* by ≈ 0.6 nats in every cut, and by 0.503 nats [0.467, 0.542] on accounts it never played on the train dates (0.605 on seen accounts: ≈ 17 % of the gain is account memory); the familiarity proxy is uninformative (fact 4) | `hr_spec_v1` |
| **Q-H8 DRIFT** | Has the meta / population / rating moved? | slow and small: consecutive months' species mix differs by 0.038–0.049 TVD, most of which two account-disjoint halves of ONE month also show (0.047–0.064); the EXCESS beyond that floor is 0.005–0.011 a month, positive in 19–20 of 20 splits; the distance to the pool and panels is flat (0.19–0.20 to the pool); the mean side rating stays 1,258–1,287; the archive swings 0.09–0.23 (uploaders, not meta) | `hr_spec_v1` |
| **Q-H11 DOES A LEVER HELP?** | Does ONE human-data clause added to P\* raise its win rate vs Kakuna on human-built teams it never trained on, without costing home72, and is the gain the human part? | unmeasured. X60 is REDESIGNED (the plan §6) with both routes written before X59-S was read: X60-B (H-BC against a P\*-label placebo) if X59-S routes "H-BC first" (it did NOT: OWN-LABELS-BETTER, 2026-10-30), none if it closes human data (pending Δ_restriction), X60-T (H-TEAMS against D1's forum corpus) otherwise. Primary: F alone against Kakuna (1,200 games a final, 5 seeds, the plain-continuation control shared); a gain needs the in-house meter (never trained on human data) to agree. **REVIEW F (READY-WITH-FIXES, 0 BLOCKERs, 9 MAJORs; none applied yet).** Four gate any build: the route is not frozen (the firing PREREG is written after X59-S's numbers and routes on prose strings); Kakuna's engine drifts under the imported control cells; the human team corpus is not checked at the side pin (the packer there mis-packs Frustration and Hidden Power IVs); one corpus has two entry points. Five gate the firing route's PREREG: the power headline, X60-B's delivery logic (its most damning null lands in "closes nothing"), the KAKUNA-SPECIFIC guard (it passes a Kakuna-only gain half the time), the CPU cost (X60-T's Kakuna games alone ≈ 70–95 h of wall), and X60-B's teacher (77.5 % rated 1300–1499). **Power, restated:** a +3 pp gain is found with probability 0.80 only at zero seed noise; 0.70 at the registered 1.0 pp floor, 0.43 at 2 pp, 0.27 at the measured upper bound of 2.75 pp. The packer fix (Q-H13) adds one rule: F games played after it cannot pool with C0's banked F cells | `hr_spec_v1` (rowset r2, or the route's new `hr_spec_v1.r<n>`) |
| **Q-H12 TOGGLE INTEGRITY** | With every lever off, is the system unchanged? | trivially yes today (nothing is built); the gates of §6 are PROPOSED, and so is the `--team-mix` guard (REVIEW F F-M7) | — |
| **Q-H13 CONTAMINATION** | Which meters already carry human data, and what does a lever's gain on them mean? | Every panel is TEAM-clean once the ≥ 3-fit exclusion runs; POPULATION is what differs. H is the human levers' own population and construction (ladder sides with usage fills): a gain there alone is "learned the ladder". P72 is fill-adjacent (its generator draws from the Smogon tables C-GAME fills with). F (human-built forum teams) is the cleanest open panel, so X60's primary is F alone. Kakuna is human-pretrained, so a lever's Kakuna gain counts only if the in-house meter against our own reference population agrees. The sealed `forum216b` carries the clean claim at X54. **No panel is shown Kakuna-familiar** (Kakuna's replay data, up to 2025-08, predates H's source; H's own date window is UNVERIFIED; ladder near-copies, ≥ 4 shared sets: H 4.2 %, F 14 %, P72 0 %, home72 47 %; [`goal_b_panel_validity_2026-10-26`](../research_state/measurements/goal_b_panel_validity_2026-10-26/README.md)). **The Kakuna packer fix (`b0441623e`, 10-27):** Metamon's peer packed teams with an upstream packer that differs from Showdown's reading of a paste. Of the panels here only F (forum72) changed: its `sft431` Gyarados (`Happiness: 0`) played Frustration at base power 1, not 102 (`away` and two pool teams changed too; H, P72 and home72 pack identically). Every earlier Kakuna F read, X59-S's included, faced that small handicap on 1 team in 72, equal across a read's arms, so contrasts inside one read do not move; but a new F read is its own regime and an import of banked F rows across the fix REFUSES (`designs/tools/reads.md` §4) | — |
| **Q-H14 PROVENANCE** | Given a checkpoint, did any ancestor train on human data, and on which rowset? | today: no run can have (nothing is built). The HOM and X59-S's 15 human-row students (S-H, S-P, S-P0) trained on `hr_spec_v1`'s rows but are offline and outside `models/`; none seeds a run. After the switch is built, `main.lineage` (PROPOSED) | — |
| **Q-H15 TEACHER QUALITY** | How strong is the population we would imitate, against P\* and Kakuna? | upstream's human-only policies all lose to P\* (U 0.63–0.79 for P\*) and are 14–30 pp weaker than Kakuna; imitation is the weakest (LargeIL), offline RL the strongest (LargeRL). X59-S's rows are 77.5 % rated 1300–1499 (the card ranks that stratum "the comparator only"); a ≥ 1500 follow-up (S-H≥1500 against a volume-matched mix) fired (Δ_data read SELF-PLAY-DATA-BETTER; 270,260 such rows). Human labels inside our architecture lose to P\*'s on the same states: X59-S Δ_labels −4.50 pp [−7.42, −1.58] (OWN-LABELS-BETTER) | `hr_spec_v1.r2` |

## 8. The drift policy

- **The meter:** species-usage TVD of full-six rated sides, month against month, read against an ACCOUNT-disjoint floor
  (REVIEW A m1: repeat players make a side-level split-half floor of 0.009 far too small). Players are split into halves
  by a keyed hash, 20 times; WITHIN = TVD of two account-disjoint halves of one month; CROSS = the same across two
  months; EXCESS = CROSS − WITHIN at the same sample size. Plus the monthly mean side rating and the ≥ 1500 share.
- **`hr_spec_v1`'s reference values:** WITHIN 0.047–0.064 by month and EXCESS 0.005–0.011 per consecutive month (means of 20 splits; positive in 19–20 of 20) (registry `drift_reference`).
- **The alarm (for a new snapshot):** a new month's EXCESS over the snapshot's last month positive in every split with a mean above 0.016 (1.5 × the largest within-`hr_spec_v1` mean excess, 0.0108); or a mean-rating move > 50 Elo between same-population months.
  The floor scales with sample size, so a new snapshot is compared at its own size, never against a fixed constant.
- **PROPOSED: a behavioural meter** (species TVD cannot see set or play drift): the HOM's NLL on each new month's
  ≥ 1500 rows against its held-out value.
- **What drift invalidates:** a verdict on snapshot S speaks for S's months; under an alarm a human-trained lever's
  verdict is re-read on the new snapshot before it steers anything. Results that read no human decisions are untouched.
- **Between populations** (archive vs spectator) the distance is selection, never drift.

## 9. Where this is heading (all PROPOSED unless marked)

1. **X59-S** (READ 2026-10-30: SELF-PLAY-DATA-BETTER, OWN-LABELS-BETTER): Δ_restriction decides the closure; the
   ≥ 1500 follow-up (≈ 1.5 GPU-h, in-house read) asks whether rating changes it at fixed slice and volume. H-BC is
   dropped, so X60-B is off the table; X60-T stands only if the closure does not (§7.1).
2. The switch and the consumer-side snapshot check (≈ 1.5 agent-days), only once a first read routes a lever to a
   build; REVIEW F's `--team-mix` guard is written before either arm that uses a human team corpus launches.
3. X60 (REDESIGNED; REVIEW F's fixes owed before a build): H-BC (with the P\*-label placebo) or H-TEAMS (with the D1
   contrast), per X59-S (the plan §6).
4. A live spectator collector on the Rust live reader: read-only spectating needs the owner's word; not designed here.
   It would also answer Q-H4's open half.
5. The (c1) intake, the one full-information human source within reach.
6. `n1_hrep_v1`, a corpus of Hidden-Power-repaired REPLAY teams for N1's breadth (X61 PREREG §10; X53-N1's arm
   N1H): its own `--team-mix` toggle, its snapshot id and `"human_data": true` in its manifest, the ≥ 3
   move-revealed-fit exclusion before the fill. It is the corpus H-TEAMS needs too (the plan §6.4), so it is built
   once, under one rowset id. **OPEN:** one corpus then has two entry points (`--team-mix n1_hrep_v1` and
   `--human-data teams:<rowset>@<share>`); REVIEW F F-M7 asks for the throwing guard (§6) and one spec (the snapshot,
   the account cap, the audit's drop) before the build, and for the corpus to pass the side pin's soak and pack
   identity. The procedural `n1_proc_v1` is Smogon aggregates, not this.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **This doc CREATED (DRAFT)**: the owner's requirement of an always-current architecture for human data, split from the lever design (explicit-only) | the four-source taxonomy; snapshots frozen by manifest sha256; one flag registered in `agents.info_levers`; Q-H1..Q-H12; a split-half-floored drift policy | one doc for architecture and experiment; pooling (a) and (b); per-lever flags | the card; X50 R-b1 / R-b2 |
| 2026-10-25 | **REVISED after REVIEW A / B / C** (REVISED DRAFT, pending re-review) | a status box pinned by a new static gate; the INVENTORY of human data already in use (the old headline "no human data enters" was false); the taxonomy on two axes with the (c1) intake; the snapshot → rowset → model chain; ONE grammar here (judge and the consent flag dropped); fork / resume refuse without a typed clause; the owner's two questions as first reads; the account-clustered drift floor; snapshot extraction from the manifest, a durable copy, salted account ids; future things PROPOSED | keeping "no human data enters" with a footnote (it answers the contamination question wrongly); a rowset id hidden behind the snapshot id (two runs citing one snapshot could train on different rows); `judge` kept "closed" inside the grammar (replace, don't append) | `RESPONSE.md`; `REVIEW_A_data.md`, `REVIEW_B_experiments.md`, `REVIEW_C_reasoning.md` |
| 2026-10-25 | **FR-1 and FR-3 REGISTERED; `hr_arch_2026-10-24` copied** | P\* vs SmallRL, LargeRL, LargeIL on H, F, P72, home72 (400 each, X51's twins, greedy); LargeRL / LargeIL pinned at hub `ac8e88e7` ckpt 40; the archive snapshot's durable copy (16,490 members re-hashed) | re-running an identity unit (the strongest read showed it at the same commit); a sampling regime (SOP rule 2) | `human_only_vs_pstar_2026-10-25/PREREG.md`; the registry row |
| 2026-10-25 | **REVIEW D answered; ALWAYS-CURRENT** | page one says there is no evidence yet either way, spells out the `open` line and the switch values, relabels the coverage line (a lower bound against the pool); Q-H9 is the owner's question as asked (Δ_data = S-H − S-S, the self-play arm mandatory) and X59-S is built and registered; the existing HOM is barred from levers and routing; Q-H6 states what the exclusion keeps; a snapshot is read through `src/main/human_data/snapshots.py` (a refusal never hangs); the first rowset `hr_spec_v1.r1` is registered | keeping H-INIT on the existing HOM (its rows kept panel cores); reading Q-H9 as labels only (it holds the states fixed) | `RESPONSE.md` §D; `REVIEW_D_rereview.md` |
| 2026-10-25 | **N1's replay-team corpus specified as a separate human-data toggle** (training-lever builder, X61) | `n1_hrep_v1` (PROPOSED, §9 item 6): its own corpus and `--team-mix` toggle, one registered snapshot cited, the ≥ 3 move-revealed-fit exclusion before the fill; the procedural `n1_proc_v1` listed under human AGGREGATES (§2) | mixing replay teams into the procedural corpus (one toggle for two sources) | `x61_n1_breadth_2026-10-25/PREREG.md` §10 |
| 2026-10-25 | **FR-1 READ: OURS-BEATS SmallRL**; FR-3 held behind it (CPU-bound), then resumed | Q-H10's answer for a 14M human-only model is no; the 200M question stays open | — | `human_only_vs_pstar_2026-10-25/README.md` |
| 2026-10-25 | **REVIEW E answered (X59-S NOT-READY → Amendment 2)** | before any training: S-S faces the human panel rule (a simulated human reveal, redrawn while near); the Kakuna-family audit runs first and drops flagged accounts above 1.0 % of rows; the quitter's last 10 decisions leave; S-P named "P\*-restricted-to-human-support" with an unrestricted S-P0; floor references, UNINFORMATIVE, NO-DIFFERENCE-AT-4PP, KAKUNA-SPECIFIC; a CUDA pre-flight and an exit-3-only window; a conditional ≥ 1500 follow-up | the review's kept-side reveal profile (it under-excludes); a margin against S-S for the H-BC gate (no margin can be set before data); retraining after the audit | `RESPONSE.md` §E; `REVIEW_E.md`; the PREREG's Amendment 2 |
| 2026-10-26 | **X59-S inputs rebuilt (Amendment 2b); `hr_spec_v1.r2` registered** | the audit ran first (POWER; drop rule fired on 11 accounts, 1.73 %, none Kakuna-like); r2 = r1 minus the quitter's last 10 decisions and those accounts; S-S regenerated (16.8 % of pool pairs pass the panel rule, as 16.8 % of human battles did); READY to train | waiving the drop after reading that the flagged accounts are not Kakuna-like (a post-hoc deviation from a registered rule) | the PREREG's Amendment 2b; `results/kakuna_audit.json`; `results/registration_a2.json` |
| 2026-10-26 | **FR-3 READ: OURS-BEATS LargeRL and LargeIL** | Q-H10 answered NO for every human-only model Metamon publishes; human data's value inside our network rests on X59-S | — | `human_only_vs_pstar_2026-10-25/README.md` |
| 2026-10-26 | **X60 REDESIGNED (pending review); Q-H11 and Q-H13 restated** | Q-H11 names the conditional A/B (X60-B / none / X60-T by X59-S's verdict, F alone, the in-house agreement rule); Q-H13 splits contamination into team (settled by the exclusion) and population (H own, P72 fill-adjacent, F cleanest open); `n1_hrep_v1` and H-TEAMS are one corpus, built once (its two entry points left OPEN, §9 item 6) | two corpora for one population; F + P72 as a human lever's primary | the plan §6; `x60_redesign_2026-10-26/power.py` |
| 2026-10-26 | **X59-S TRAINED (20 students, window PASS 23:04); the box's "no human decision in training" corrected** | the pre-flight's export FAIL (the loader read the policy in training mode) fixed for every reader, the whole pre-flight re-run, a 453-minute window; the box now reads "X59-S's 5 S-H student networks only (offline)"; §2 and Q-H14 say the students sit outside `models/` and seed nothing | leaving "none / none / none" (false from this moment); registering the students as models (that registry is for models a lever uses) | the PREREG's Amendments 3 and 4; `141f36f7f` |
| 2026-10-27 | **X59-S's read machinery and rules registered, all before any result** (Amendments 4, 4b, 4c, 5, 6) | one play tree and engine (pin last moved to `89c2c2501`); FR-2 READ NO-ARTEFACT, with the battle exclusion for silently ending volatiles; REVIEW H: a per-contrast floor (0.0958), the wider-of-two interval, the in-house interval for "no difference", POPULATION-SPECIFIC and two-way KAKUNA-SPECIFIC, the closure's scope and fit gate, the re-encode check (0.60 %), a midpoint H-BC gate; REVIEW G: Δ_restriction as the closure's guard instead of FR-2, the slice named in the closure, a length split | keeping the 0.15 floor bar (it would call reads UNINFORMATIVE that Kakuna resolves); FR-2 as the closure's guard (it cannot see the signature it gates) | the PREREG; `eadacd559`, `0470d273a`, `dd1299109`, `b27226c3a` |
| 2026-10-27 | **REVIEW F recorded; Q-H11's power restated; the Kakuna packer fix enters Q-H13** | the headline GAIN 0.80 at +3 pp is the zero-noise cell, quoted as a range (0.70 at the 1.0 pp floor); X60's nine MAJORs listed, four gating any build; F reads before `b0441623e` are a legacy-pack regime and cannot pool with later ones | quoting 0.80 "at the measured noise"; treating the packer fix as invisible to human-team panels | `REVIEW_F_x60.md` (`e41beafe1`); `b0441623e`; `designs/tools/reads.md` §4 |
| 2026-10-27 | **This doc AUDITED against everything since 10-25** | the Q-H table at a glance with a status and an evidence commit per question; one data card per source with its entry point; rowset r2 in §5 and the students in §2 and §5; Q-H4 gains the spectator's forfeit share; Q-H5 gains FR-2 and its limit; the X59-S window's measured wall time replaces its estimate; stale figures in other docs reported, not edited | appending a changelog to page one (replace, don't append) | this commit |
| 2026-10-30 | **X59-S READ: SELF-PLAY-DATA-BETTER (Q-H9) and OWN-LABELS-BETTER; H-BC dropped** | page one, Q-H9, Q-H10 (the student-to-P\* gap, secondary) and Q-H15 answered from the full record; the closure of the 1300+ slice pending Δ_restriction; the ≥ 1500 follow-up registered next; X60-B off the table | closing human data before Δ_restriction is read (A6.1 makes it the guard) | X59-S README; `results/full_record_decision.json` |
