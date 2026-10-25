# HUMAN DATA in the system: what is in use, the sources, the snapshots, the switch, and what we know

**Status: REVISED DRAFT — pending re-review (2026-10-25).** This revises the draft of `14ae4203` after three adversarial
reviews: A (data / bias / leakage), B (experiments / causal) and C (reasoning). Every finding's disposition is in
[`RESPONSE.md`](../research_state/measurements/human_replay_quality_2026-10-25/RESPONSE.md). Once re-reviewed the doc is
**ALWAYS-CURRENT**: every change that touches human data updates it and its Decision record in the same commit, and
[`src/human_data_doc_gate_test.py`](../../src/human_data_doc_gate_test.py) pins its status box, snapshot table and
switch state to the registry, the card and the code. The plan and its pre-registrations are the explicit-only
[`design_human_replay_lever.md`](design_human_replay_lever.md); the measured data card is
[`../research_state/measurements/human_replay_quality_2026-10-25/`](../research_state/measurements/human_replay_quality_2026-10-25/README.md).

**Tense rule.** A sentence in the present tense describes something that exists today. Everything not built is marked
**PROPOSED**.

## 0. STATUS (pinned by `src/human_data_doc_gate_test.py`)

```
STATUS
  human decisions in training / eval / play     : none / none / none
  human-built teams in training                 : yes (the 719-team pool; --team-mix corpora when typed)
  replay-derived teams in eval                  : yes (heldout72_hpfix = Metamon hl_05_26)
  human-trained models                          : Kakuna, SmallRL, LargeRL, LargeIL, PastaMittens (eval anchors); the HOM (offline only)
  human aggregates                              : Smogon usage / sets / spreads (every prior; the C-GAME fill)
  switch --human-data                           : NOT BUILT
  snapshot hr_spec_v1                           : 376,410 logs; durable copy yes
  snapshot hr_arch_2026-10-24                   : 16,490 logs; durable copy yes
  rowsets / models registered                   : 0 / 0
  informative rows, actor >= 1500               : 1.80M
  human sides in the panels' region (>= 1500)   : 15.9 %
  HOM - P* NLL on accounts unseen in training   : -0.503 nats
  open                                          : Q-H10 first read (P* vs SmallRL); the trace twin; X59-S
```

**What the box says, in words.** No human DECISION reaches training, evaluation or play today. Human data is
nevertheless already in the system in four other forms (§2): the teams we train on were built by people, one
evaluation panel was extracted from ladder replays, our external meters were trained on human replays, and every prior
is Smogon's aggregate of human ladder games. The switch that would let human decisions train the policy is not built.

## 1. Which result read which data

| result | data | id it cites | where |
|---|---|---|---|
| X50 R1, R0, R-b1, R-b2 | spectator logs | `hr_spec_v1` (retroactively: the reads ran on exactly its paths; train ≤ 09-20, held out 09-21..23) | `design_x50_rung1b_human_opponent_model.md` |
| the HOM (human opponent model) | spectator, actor ≥ 1500, heldout72 ≥ 4-fit exclusion | `hr_spec_v1`; its rowset predates the rowset registry (unregistered) | `~/gen3ai_archive/x50_r1b/hom/` |
| this data card | spectator + archive | `hr_spec_v1`, `hr_arch_2026-10-24`, stamped in `results.json` | the card |
| the archive-skew read | archive pulls before the registry | `none` (predates; cite the folder) | [`replay_archive_bias_2026-10-21`](../research_state/measurements/replay_archive_bias_2026-10-21/README.md) |
| every self-play, Kakuna and panel result | no human decisions | `none` (read §2 for what IS human in them) | — |

## 2. The inventory: every human-derived thing in use today

| kind | in use today | where | its switch | what it means for a result |
|---|---|---|---|---|
| human-BUILT teams | **training** | the 719-team pool = PokePaste collections of five community builders (`data/teams/others/`; four are listed in its `sources.json`) + Smogon sample teams (`data/teams/sample/`); D1's forum corpus (`src/utils/team_corpora/w_forum_v1/`) only when `--team-mix` is typed | none for the pool (it IS the baseline); `--team-mix` | self-play pilots human-designed teams; "no human data" never meant "no human artefacts" |
| REPLAY-derived teams | **evaluation** | `heldout72_hpfix` = Metamon's `hl_05_26`: teams extracted from ladder replays, hidden sets filled from usage | none | a human-data lever trains on H's POPULATION: a gain on H alone is "learned the ladder", not generalisation (Q-H13) |
| human AGGREGATES | **every prior** | Smogon usage / sets / teammates / spreads (`data/pokemon/`), read through `agents.gen3_data`; the C-GAME fill; P72's generator | none | allowed by the root priors rule; P72 and every C-GAME fill draw from the SAME tables (Q-H13) |
| human-TRAINED models | **evaluation**; offline | Kakuna (Superkazam: self-play + 4M human replays), `metamon:SmallRL`, `LargeRL` (human-only offline RL) and `LargeIL` (human-only imitation), PastaMittens (a Kakuna-family router) in `designs/ops/anchors.json`; the HOM (offline measurement only) | none | Kakuna is human-pretrained, so it is not a neutral meter for human-likeness (B-M6) |
| human DECISIONS (rows) | **none** in training / eval / play | `hr_spec_v1`, `hr_arch_2026-10-24`: offline measurement only | `--human-data` (**NOT BUILT**) | the rest of this doc |

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
   stream covered public rooms at the daemon's capacity (≈ 50 % of ladder games, ESTIMATE); self-play is our pool only.
3. **Coverage decides whether a source can help goal (b).** Self-play never pilots a team outside the pool; ≥ 15.9 %
   of full-six ≥ 1500 human sides sit in the region the unfamiliar panels occupy. Most of the near-panel region must be
   EXCLUDED from training to keep the panels unfamiliar (the plan §3.1).
4. **There is no offline evidence either way** that human labels carry the unfamiliar-team skill goal (b) lacks. The
   one proxy (P\*'s human-prediction NLL by team familiarity) is UNINFORMATIVE: it predicts the same flat profile
   whether that hypothesis is true or false (REVIEW A M1). The first reads (§7.1) and X59-S test it.

## 4. The source taxonomy

**Rule beside the table:** a known OPPONENT team never enters the actor's observation; it is a label or a measurement
only (an oracle leak otherwise, the reveal spec's territory). A source may never be pooled with another.

| source | exists? | actor side / opponent side | selection (who chose that this game is in our data) | volume | use |
|---|---|---|---|---|---|
| **(a) ARCHIVE**: public uploads through the owner's proxy (`tools/replay_corpus_downloader`) | yes: `hr_arch_2026-10-24` | public / public | the UPLOADER, after the game: +35 to +128 Elo, longer, fewer forfeits; one account 10 % of a window | 365 games / day (160–714) | drift and collection reads only; **DROP for training** (reopened only by a registered coverage read and its own snapshot) |
| **(b) SPECTATOR**: games the deleted daemon watched (2026-05-18 → 09-23) | yes: `hr_spec_v1`; no collector today | public / public | public rooms only (private never listed), a 30 s poll, ≤ 20 rooms at once, rooms > 1 h dropped, a reaper added 07-05: ≈ 50 % of ladder games (ESTIMATE), unmeasured by length / rating / month | 3,497 games / day while it ran | the nearest ladder sample we hold, NOT a verified ladder sample (Q-H4); the source of every lever's rows |
| **(c1) SUBMITTED**: a replay a person gives us with their own team paste (PROPOSED intake) | no | true / public | the submitter (opt-in; consent recorded) | owner and friends: tens of games | the fill error on HUMAN teams (A-2's test); never a policy-lever source until registered |
| **(c2) TEAM SHEETS**: tournament replays matched to posted teams | no | true / true | the tournament | unknown | the fill-error gold standard; opponent truth is a label only |
| **(c-ours) EVAL TRACES**: our games vs bots / Kakuna | yes (`models/<run>/eval_traces/`) | true / true | ours (pool, panels) | per eval | measures the fill error on OUR teams (R0: +0.012 to +0.019 nats) |
| **(d) SELF-PLAY** | always | true / true | the 719-team pool | unbounded | the baseline; the familiarity premium ≈ 11 pp (X51) |

**The (c1) intake (PROPOSED).** A person drops a replay plus their paste into `~/gen3ai_archive/human_snapshots/intake/`
with a consent line; a cut freezes the folder as snapshot `hr_sub_<tag>` (`actor_side: true`, `opp_side: public`);
the account is stored only as a salted hash; a per-account cap applies; it is never pooled with (a) or (b).

## 5. Snapshots → rowsets → models

```
  SNAPSHOT  raw logs, frozen                hr_spec_v1                     registry "snapshots" (exists)
     │  + the filter spec (rating floor, bot rule, walk-over bar, per-account cap, the panel-exclusion RULE and SET,
     │    the hidden-turn label rule, the account split) + C-GAME commit and seed + extraction commit + layout hash
     ▼
  ROWSET    training rows or teams          <snapshot>.r<n>  (rows manifest sha256)   registry "rowsets" (empty)
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
| `hr_spec_v1` | spectator | 376,410 | 2026-05-18 → 2026-09-23 | `e7d491e7f7a1b98093360ece7929eca0d2880c0ffc5706244edee4700935f3b1` | yes | reference | X50 R1 / R0 / R-b1 / R-b2, this card; TRAIN ≤ 09-20, HELD OUT 09-21..23 |
| `hr_arch_2026-10-24` | archive | 16,490 | 2026-06-16 → 2026-10-24 | `ad940108978dd6a52a72cf2d637cc108c652d3eaa5c673346e8351c0f1f2f7a2` | yes | measurement | drift and collection reads only |

- **The manifest IS the definition.** Late uploads land in old date folders, so "as on disk" is not reproducible.
- **Verification exists for the card's code.** `hrq.py extract --snapshot <id>` reads the manifest, checks its digest
  against the registry, re-hashes every log and checks the population's header, and REFUSES on any mismatch; the cache
  and `results.json` carry `{snapshot_id, manifest_sha256, extraction_commit, hrq_sha256}`. **PROPOSED:** the same
  check inside `agents.info_levers.human_data`, refusing any consumer whose stamp the registry does not hold.
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
| offline measurement | X50 reads, the card, drift | any read citing its snapshot / rowset / model id |

- **The complexity budget.** ONE clause per run; the clause's ONE parameter; every other knob (the filters, a
  `coef(t)` schedule, a human-minibatch share, a per-account cap) lives in the ROWSET or in a pre-registration, never
  on the command line. Two levers at once is a new registered lever value replacing a row, never a second flag. At most
  three lever heads (`bc`, `teams`, `kl`); a new one replaces a row with a decision-record entry.
- **Provenance (PROPOSED).** The resolved config, `model_config.json`, `metadata.json` and the immutable lineage carry
  the clause and its hash (`sha256("human_data_v1:" + canonical)[:16]`). **Resuming or forking a run whose lineage has
  a human-data clause REFUSES unless `--human-data` is typed** (the clause or `off`), so nothing inherits or drops a
  lever silently. `main.lineage` shows the first human-data ancestor; `main.baselines set` refuses one without a reason.
- **What it is NOT.** It reveals nothing about the battle being played and is never combined with `--oracle-reveal`.
- **Off means off (PROPOSED gates).** With `off`, the training smoke runs under an `open()` audit that fails on any path
  under the corpus, a snapshot or rowset store; `learner_golden` and the played-bits sentinel are unchanged by the build;
  a null-dose ON build (coefficient or share 0) reproduces OFF's played bits.
- **Production:** `designs/production_config.json` has no `human_data` key today; when built it resolves `off`.

## 7. The questions and their current answers

### 7.1 First reads: the owner's two questions

| order | question | the read | cost | answer now |
|---|---|---|---|---|
| 1 | **Q-H10 HUMAN-ONLY MODEL:** does a model trained only on human data beat ours? | P\* vs `metamon:SmallRL` (human-only offline RL, pinned) on H, F, P72, home72, 400 games each | ≈ 1.3 CPU lane-h, no build | REGISTERED and playing ([`human_only_vs_pstar_2026-10-25`](../research_state/measurements/human_only_vs_pstar_2026-10-25/PREREG.md)); Metamon's own table puts its human-only models far below its self-play-trained ones (README, UNVERIFIED for our cells) |
| 2 | instrument check | the HOM's TRACE TWIN: NLL on C-GAME-completed vs true rows of the same decisions, minus P\*'s | ≈ 1 lane-h + ≈ 0.2 agent-day | unread (registered in rung 1b, never run) |
| 3 | Q-H10 at scale | P\* vs `LargeRL` and `LargeIL` (200M, human-only) | ≈ 4.5 lane-h each | pinned (hub `ac8e88e7`, ckpt 40, sha256 in the registration); REGISTERED and playing ([`human_only_vs_pstar_2026-10-25`](../research_state/measurements/human_only_vs_pstar_2026-10-25/PREREG.md)) |
| 4 | **Q-H9 HUMAN vs SELF-PLAY DATA:** on the same states, do human labels teach better than ours? | X59-S, the matched-student contest (S-H human labels vs S-P P\*'s sampled labels); S-H vs P\* is also Q-H10 in our architecture | ≈ 0.5–1 agent-day + 25–35 CPU lane-h (or ≈ 2 GPU-h + 20 lane-h) | unmeasured |

### 7.2 The rest

| id | question | answer now | snapshot |
|---|---|---|---|
| **Q-H1 COLLECTION** | How do we get more, at what rate, legally? | the spectator stream gave 3,497 games / day and its collector is deleted; reviving it needs the owner's word (the proxy grant is "download only", and spectating joins rooms); the archive timer gives 365 / day, SKEWED; the proxy's 1 request / s is not the bound (≈ 43k replays / day would fit), the upload volume is; an archive backfill to 2020 (≈ 0.5M games, ESTIMATE) is not authorised | `hr_arch_2026-10-24` |
| **Q-H2 SNAPSHOTS** | Which data exactly did a result use? | two snapshots frozen and registered (§5); the card's extraction re-hashes against the manifest and stamps its results; one durable copy | both |
| **Q-H3 QUALITY** | Parse, duplicates, bots, walk-overs? | 0 text-pass errors; one rule set; C-GAME completes 99.39 % of seats; 4 duplicate bodies; 61 rule-flagged accounts ≈ 6 % of sides (heavy human grinders among them); walk-overs 1.0–5.1 % of sides; half of all games end in a forfeit | `hr_spec_v1` |
| **Q-H4 SELECTION** | Is the sample the ladder? | unanswerable from data we hold: the archive is SKEWED; the spectator stream's coverage is ≈ 50 % (ESTIMATE) and unmeasured by length, rating and month. What would answer it: `search.json` uploaded ids on ~30 spectator dates, P(spectated) by turns / rating / hour (needs the owner's approval for the pulls) | both |
| **Q-H5 LABELS / FILL** | How often is the chosen action known; how much is invented? | known for 90.3 % of completed decisions; the hidden 12 % are missing NOT at random (a sleeping mon's labelled switch rate doubles), so the plan keeps them with a set label; ≈ 52 % of own move slots and every spread are Smogon draws, from a prior 1–17 months older than the games; a drawn move is never the label (a structural signature) | `hr_spec_v1` |
| **Q-H6 COVERAGE** | Do human teams reach the unfamiliar region? | partly: ≥ 15.9 % of FULL-SIX ≥ 1500 human sides; humans pilot 3-set cores of most H and F teams, so the panel exclusion (≥ 3 move-revealed fits, before the fill) removes ≈ 71 % of ≥ 1500 battles (`forum216b` not counted) | `hr_spec_v1` |
| **Q-H7 LEARNABILITY** | Can a model learn human choices? | yes: the HOM beats P\* by ≈ 0.6 nats in every cut, and by 0.503 nats [0.467, 0.542] on accounts it never played on the train dates (0.605 on seen accounts: ≈ 17 % of the gain is account memory); the familiarity proxy is uninformative (fact 4) | `hr_spec_v1` |
| **Q-H8 DRIFT** | Has the meta / population / rating moved? | slow and small: consecutive months' species mix differs by 0.038–0.049 TVD, most of which two account-disjoint halves of ONE month also show (0.047–0.064); the EXCESS beyond that floor is 0.005–0.011 a month, positive in 19–20 of 20 splits; the distance to the pool and panels is flat (0.19–0.20 to the pool); the mean side rating stays 1,258–1,287; the archive swings 0.09–0.23 (uploaders, not meta) | `hr_spec_v1` |
| **Q-H11 DOES A LEVER HELP?** | Does a lever raise U vs Kakuna without costing home72? | unmeasured; X60 (fixed per REVIEW B) | — |
| **Q-H12 TOGGLE INTEGRITY** | With every lever off, is the system unchanged? | trivially yes today (nothing is built); the gates of §6 are PROPOSED | — |
| **Q-H13 CONTAMINATION** | Which meters already carry human data, and what does a lever's gain on them mean? | H is replay-derived; P72 is drawn from the fill's own Smogon tables; F is human-built with human cores near most teams; Kakuna is human-pretrained. No unfamiliar panel is clean; the sealed `forum216b` is the cleanest claim | — |
| **Q-H14 PROVENANCE** | Given a checkpoint, did any ancestor train on human data, and on which rowset? | today: no run can have (nothing is built); after the build, `main.lineage` (PROPOSED) | — |
| **Q-H15 TEACHER QUALITY** | How strong is the population we would imitate, against P\* and Kakuna? | unmeasured; the first reads 1, 3 and X59-S bear on it | — |

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

## 9. Where this is heading (all PROPOSED)

1. The first reads of §7.1, in order (CPU): FR-1 and FR-3 are playing; the trace twin and X59-S follow.
2. The switch and the consumer-side snapshot check (≈ 1 agent-day), only once a first read routes a lever to a build.
3. H-TEAMS (with the D1 contrast) or H-BC / H-INIT, per X59-S, read in X60 as REVIEW B fixed it.
4. A live spectator collector on the Rust live reader: read-only spectating needs the owner's word; not designed here.
5. The (c1) intake, the one full-information human source within reach.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **This doc CREATED (DRAFT)**: the owner's requirement of an always-current architecture for human data, split from the lever design (explicit-only) | the four-source taxonomy; snapshots frozen by manifest sha256; one flag registered in `agents.info_levers`; Q-H1..Q-H12; a split-half-floored drift policy | one doc for architecture and experiment; pooling (a) and (b); per-lever flags | the card; X50 R-b1 / R-b2 |
| 2026-10-25 | **REVISED after REVIEW A / B / C** (REVISED DRAFT, pending re-review) | a status box pinned by a new static gate; the INVENTORY of human data already in use (the old headline "no human data enters" was false); the taxonomy on two axes with the (c1) intake; the snapshot → rowset → model chain; ONE grammar here (judge and the consent flag dropped); fork / resume refuse without a typed clause; the owner's two questions as first reads; the account-clustered drift floor; snapshot extraction from the manifest, a durable copy, salted account ids; future things PROPOSED | keeping "no human data enters" with a footnote (it answers the contamination question wrongly); a rowset id hidden behind the snapshot id (two runs citing one snapshot could train on different rows); `judge` kept "closed" inside the grammar (replace, don't append) | `RESPONSE.md`; `REVIEW_A_data.md`, `REVIEW_B_experiments.md`, `REVIEW_C_reasoning.md` |
| 2026-10-25 | **FR-1 and FR-3 REGISTERED; `hr_arch_2026-10-24` copied** | P\* vs SmallRL, LargeRL, LargeIL on H, F, P72, home72 (400 each, X51's twins, greedy); LargeRL / LargeIL pinned at hub `ac8e88e7` ckpt 40; the archive snapshot's durable copy (16,490 members re-hashed) | re-running an identity unit (the strongest read showed it at the same commit); a sampling regime (SOP rule 2) | `human_only_vs_pstar_2026-10-25/PREREG.md`; the registry row |
