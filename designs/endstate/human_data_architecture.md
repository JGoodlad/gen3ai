# HUMAN DATA in the system: the sources, the snapshots, the switch, and what we know

**Status (2026-10-25): DRAFT until the two adversarial reviews (data / bias / leakage; experimental / causal) land;
then ALWAYS-CURRENT** (the endstate rule: it describes the system as it is and where it is heading, and every change
that touches human data updates it and its Decision record in the same commit). **Today no human data enters
training, eval or play.** The levers below are designed, not built. The experiment plan and its pre-registration are
the explicit-only [`design_human_replay_lever.md`](design_human_replay_lever.md); the measured data card is
[`../research_state/measurements/human_replay_quality_2026-10-25/`](../research_state/measurements/human_replay_quality_2026-10-25/README.md).

**The rules it serves.**
- **The owner's grant (2026-10-25):** human replays MAY train the policy and the search, ONLY as a toggleable lever,
  OFF by default; the system always trains and plays with no human data, and nothing depends on it silently;
  data-quality analysis comes first. It supersedes the 10-23 scope (the human model as search's judge only).
- **The priors rule (unchanged):** what the network READS (observation features, tables, `data/` priors) traces to
  Smogon, ground-truth labels or ladder replays; a team pool measures structure and never ships as a prior. Every
  hidden fact a human source lacks is filled from SMOGON only, never from our pool or a later reveal of the opponent.
  Human data enters as a TRAINING SIGNAL (labels in a loss, teams to play) or as a measurement, never as an input.

---

## 1. The mental model, on one page

```
                 ┌──────────── what the source saw ────────────┐
  SOURCE         │ own side            │ opponent side          │  selection          │ volume
  (d) self-play  │ FULL                │ FULL (we hold both)    │ ours (the pool)     │ unbounded
  (c) full-info  │ FULL                │ FULL or partial        │ opt-in / ours       │ (c-ours) eval traces only; (c-human) none
  (b) spectator  │ what play revealed  │ what play revealed     │ finished rooms      │ ≈ 3,450 games/day while it ran
  (a) archive    │ what play revealed  │ what play revealed     │ the UPLOADER        │ ≈ 250-450 games/day
                 └──────────────────────────────────────────────┘
        │ (a),(b): the hidden part is FILLED from Smogon (C-GAME), checked against the log
        ▼
  ROWS (obs, legal mask, the chosen action, the outcome)  ── snapshot id ──►  a LEVER, one per run, OFF by default
        │                                                                     ├─ bc     the loss  (training)
        │                                                                     ├─ kl     the loss  (training)
  TEAMS (completed, legal, panel-excluded)  ── snapshot id ──────────────────►├─ teams  the pool  (training)
                                                                              └─ judge  search's opponent likelihood
  eval and play: never read human data (the panels are frozen team sets; Kakuna is an external anchor)
```

**Three facts carry most of the reasoning.**
1. **Information level decides how much of a row is invented.** A self-play row is exact. A replay row's OWN side is
   largely invented: at ≥ 1500 a mon shows ≈ 1.9 of its 4 moves on average, EVs never; the completion fills the rest from
   Smogon. The CHOSEN ACTION is real in every source.
2. **Selection decides what population a source is.** The archive is not the ladder (SKEWED, fd7f4624); the spectator
   stream is the nearest thing we have; self-play is our pool only.
3. **Coverage decides whether a source can help goal (b).** Self-play never pilots a team outside the 719-team pool;
   human sides do: at least 15.9 % of ≥ 1500 human sides sit in the region the unfamiliar panels occupy (≤ 2 mons fitting any pool team's sets), and humans pilot cores near most H and F teams hundreds of times.

---

## 2. The source taxonomy

| source | exists? | observed | hidden | how the hidden part is filled | selection | bias risk |
|---|---|---|---|---|---|---|
| **(a) ARCHIVE** — uploaded public replays, through the owner's proxy (`tools/replay_corpus_downloader`) | YES: snapshot `hr_arch_2026-10-24` (16,490 logs) | both sides' public protocol from turn 0 to the end: what each side's play revealed; ratings; the outcome; post-game rating lines | every unrevealed move, item, ability; all EVs / IVs / natures; the uploader | C-GAME (the log's reveals kept, the rest drawn from Smogon's usage, teammates, spreads; legality- and log-checked) | the UPLOADER chooses, after the game: +35 to +128 Elo, 26–41 % longer, 10–20 pp fewer forfeits; a few accounts supply 2–10 % of sides | **HIGH**: never pooled with (b); never a ladder sample; per-account cap; used for drift and collection reads only, training only as its own registered snapshot |
| **(b) SPECTATOR** — battles watched live (the deleted poke-env daemon, 2026-05-18 → 09-23) | YES: snapshot `hr_spec_v1` (376,410 logs); no live collector today | the same public protocol as (a), for every finished room the daemon joined | as (a) | as (a) | finished rooms only; which rooms the daemon joined is unmeasured; the ladder's rating mix | **MEDIUM**: the reference sample; its coverage of the ladder is unverified (A-4); bots ≈ 6 % of ≥ 1500 sides by the rule |
| **(c-ours) FULL-INFO, our own non-self games** — eval traces vs bots, Kakuna cells where we install both teams | YES for eval traces (`models/<run>/eval_traces/`); Kakuna cells' logs: whether they are kept is UNVERIFIED | both teams in full (we built them); every choice | nothing | none needed | ours: our pool and panels; the opponents are bots or Kakuna, not humans | the players are not humans; used to MEASURE the fill error (R0's trace twin: the completion's cost +0.019 nats on our teams) |
| **(c-human) FULL-INFO, human** — team sheets, opt-in players, tournament replays matched to posted teams | **NO** | — | — | — | — | its SLOT: the gold standard for the fill error on HUMAN teams (A-2) and for team-sheet BC; a snapshot `hr_full_*` with `own_side: true`, no C-GAME |
| **(d) SELF-PLAY** — the baseline | ALWAYS | everything, both sides | nothing | none | the 719-team pool (plus a `--team-mix` corpus where registered) | the familiarity premium: ≈ 11 pp home72 vs unfamiliar panels vs Kakuna (X51) |

**Information levels, as a ladder:** (d) and (c) give the TRUE own side and both teams; (b) and (a) give the public
log only. A lever that reads rows must say which level it was trained at; a model trained on (a) / (b) rows has seen
Smogon-completed own sides that the real player never had in that form (C-GAME's declared leak, F-1b-4).

---

## 3. Snapshots and versioning

- **A snapshot is immutable and named.** Id `hr_<population>_<tag>`; a manifest of (relative path, sha256, bytes) per
  log, gzip'd, kept in `~/gen3ai_archive/human_replay_snapshots/` (never the repo); a committed registry row in
  `designs/research_state/human_replay_snapshots.json` (COMMITTED 2026-10-25 with the two rows below): id, population,
  files, bytes, dates, manifest sha256, the drawing rule, `frozen_since`. The consumers' re-hash check is to build.
- **Frozen 2026-10-25:**

| id | population | files | dates (Los Angeles) | manifest sha256 | use |
|---|---|---:|---|---|---|
| `hr_spec_v1` | (b) spectator, the 2026-09-23 manifest | 376,410 | 2026-05-18 → 2026-09-23 | `e7d491e7f7a1b98093360ece7929eca0d2880c0ffc5706244edee4700935f3b1` | the reference: X50 R1 / R0 / R-b1 / R-b2 and this card; TRAIN ≤ 09-20, HELD OUT 09-21..23 |
| `hr_arch_2026-10-24` | (a) archive, every non-manifest log on disk through 2026-10-24 | 16,490 | 2026-06-16 → 2026-10-24 | `ad940108978dd6a52a72cf2d637cc108c652d3eaa5c673346e8351c0f1f2f7a2` | drift and collection reads only |

- **Verification:** the corpus is append-only (`sync.py` never rewrites an id); a consumer re-hashes its snapshot's
  files and REFUSES on any mismatch. A zstd-compressed tar of a snapshot's logs makes it independent of the live corpus.
- **Derived data** (C-GAME rows, BC shards, team corpora) carries the snapshot id, the extraction commit and the
  encoder's layout hash; a consumer with another layout REFUSES (an observation change means a re-extraction, ≈ 5 CPU-h).
- **Citation:** every result, data card and run config names its snapshot id, or `none`. The X50 reads predate the
  registry; they ran on exactly `hr_spec_v1`'s files (the manifest's paths), so they cite it retroactively.
- **A new snapshot** is cut only on a trigger (§6) or when usable volume grows ≥ 50 %; at most two are live (the frozen
  reference and the next candidate).

---

## 4. The switch, and where each source enters

| place | today | with a lever on (designed, not built) |
|---|---|---|
| the observation / inputs / priors | (d) only; priors Smogon | unchanged, always |
| the PPO loss | (d) | `bc`: + a BC term on (a)/(b) rows; `kl`: + a KL to the HOM on (d) states |
| the self-play team pool | the 719-team pool (+ registered corpora) | `teams`: + completed human ladder teams, panel-excluded |
| search's opponent likelihood | our network as the judge | `judge`: the HOM (R-b2: HOM-NONE, so it stays off) |
| eval, the panels, Kakuna | no human data | no human data, ever (the panels are frozen; their near neighbours are excluded from every source) |
| offline measurement | X50 (`hr_spec_v1`), drift reads | any registered read, citing its snapshot |

**The one flag:** `--human-data off` (the default and what `--arch production` resolves) or ONE clause
`<lever>:<snapshot>@<param>`. It is a spec family registered in `agents.info_levers`
([`design_information_levers.md`](design_information_levers.md) §13), recorded in `model_config.json`, `metadata.json`
and the immutable lineage, never inherited silently by a resume, and a human-data ancestor is refused as a named
baseline without a stated reason. **Off means off:** a routine gate runs the training smoke under an `open()` audit
that fails on any path in the corpus, the human-row archive or a snapshot directory.

---

## 5. The named questions and their current answers

| id | question | answer now | measured on | where |
|---|---|---|---|---|
| **Q-H1 COLLECTION** | How do we get more, at what rate, legally? | the spectator stream gave ≈ 3,500 games / day (≈ 17k informative ≥ 1500 rows / day) and its collector is deleted; the archive timer gives ≈ 365 uploaded games / day (≈ 4–6k ≥ 1500 rows / day, SKEWED); the 1 request / s proxy floor is not the bound (≈ 43k replays / day would fit), the upload volume is; a historical archive backfill to 2020 (≈ 0.5M games, ESTIMATE) is the one large lever and is not authorised | `hr_arch_2026-10-24` | the card §C |
| **Q-H2 SNAPSHOTS** | Which data exactly did a result use? | `hr_spec_v1` and `hr_arch_2026-10-24` frozen and registered (§3); the consumers' re-hash check is to build | both | §3 |
| **Q-H3 QUALITY** | Does it parse; duplicates, bots, walk-overs? | 0 text-pass errors; 100 % gen3 OU singles under one rule set; C-GAME completes 99.39 % of seats; 4 duplicate bodies; 61 suspected-bot accounts ≈ 6 % of sides per stratum (rule); walk-overs 1.0–5.1 % of sides; about half of all games end in a forfeit at every rating | `hr_spec_v1` | the card §A |
| **Q-H4 SELECTION** | Is the sample the ladder? | the archive is SKEWED (fd7f4624); the spectator stream is the reference; its own coverage is unmeasured | both | fd7f4624 |
| **Q-H5 LABELS / FILL** | How often is the chosen action known; how much of a row is invented? | the chosen action is known for 90.3 % of completed decisions (15.18M of 16.81M); ≥ 1500 gives 1.80M informative rows; ≈ 52 % of own move slots and every spread are Smogon draws; the completion's cost is +0.012 to +0.019 nats where the truth is known (our teams), UNVERIFIED on human teams | `hr_spec_v1` train dates | the card §B |
| **Q-H6 COVERAGE** | Do human teams reach the unfamiliar region? | partly: ≥ 15.9 % of ≥ 1500 human sides (a lower bound) sit where the panels sit (≤ 2 mons fitting any pool team; H 98.6 %, F 77.8 %, P72 100 %); humans are nearer the panels' species mix than our pool is (TVD 0.125 to H vs the pool's 0.170); P72 and forum72's FAR teams are thinly covered (median 6 and 11 human sides fitting ≥ 4 sets) | `hr_spec_v1` train dates, ≥ 1500 | the card §D |
| **Q-H7 LEARNABILITY** | Can a model learn human choices, by stratum? | yes, uniformly: the HOM beats P\* by 0.57–0.62 nats in every cut (rating, team familiarity, turn, ending); **P\* predicts humans no worse on unfamiliar teams than on pool-like ones** (1.762 vs 1.738 nats, overlapping), so the offline proxy gives no support to a human-label mechanism for goal (b) | `hr_spec_v1` held-out dates | the card §E |
| **Q-H8 DRIFT** | Has the meta / population / rating moved? | slow real drift: species TVD 0.038–0.049 month to month against a split-half floor of 0.009; the distance to our pool (0.19–0.20) and to each panel is flat over five months; the mean side rating stays 1,258–1,287; the archive swings 0.10–0.23 (uploaders, not meta) | both | the card §F |
| **Q-H9 BETTER THAN SELF-PLAY?** | For goal (b), is a human row worth more than a self-play row? | unmeasured; the definition and the proxies are the lever doc's §2; X59 / X60 answer it | — | lever doc |
| **Q-H10 HUMAN-ONLY MODEL** | Does a model trained only on humans beat ours, or Kakuna? | unmeasured; HB0 is registered as the comparator | — | lever doc §4.3 |
| **Q-H11 DOES IT HELP?** | Does a lever raise U vs Kakuna without costing home72? | unmeasured; X60 | — | lever doc §5 |
| **Q-H12 TOGGLE INTEGRITY** | With every lever off, is the system unchanged; can a run depend on human data silently? | unbuilt (the gates of §4) | — | §4 |

### 5.1 The data card of `hr_spec_v1` (headline; the full card is the measurement README)

| stratum (actor's rating) | informative rows (train dates) | games ending in forfeit | walk-over ≤ 3 turns | suspected-bot sides | full team seen | verdict |
|---|---:|---:|---:|---:|---:|---|
| 1500–1599 / 1600–1699 / 1700+ | 1.26M / 0.44M / 0.09M | 49.9 / 49.2 / 50.8 % | 1.6 / 1.2 / 1.0 % | 6.1 / 5.9 / 5.7 % | 68–71 % | **USE-WITH-FILTER** (the policy levers' rows) |
| 1300–1499 | 5.14M | 52.2 % | 2.0 % | 5.2 % | 63 % | **USE-WITH-FILTER, the human-only comparator only** |
| < 1300 | 7.70M | 57.6 % | 5.1 % | 5.4 % | 50 % | **DROP** for every lever |
| unrated | 0.34M | 28.1 % | 4.0 % | 1.2 % | 64 % | **DROP** |
| suspected bots; walk-overs; duplicates; panel-near battles | — | — | — | — | — | **DROP** |
| the archive population | (not extracted) | 34–46 % (fd7f4624) | — | — | — | **DROP for training; USE for drift / collection** |

The filters: drop the 61 rule-flagged accounts, every battle lost by forfeit / timer by turn 3, the 4 duplicate
bodies, and every battle where a side has ≥ 4 mons fitting one team of any registered panel (forum216b included).
Learnability (the HOM's held-out gain over P\*, −0.57 to −0.62 nats) is the same in every stratum. Volume and format:
376,410 logs, 0 parse errors, one rule set, 99.39 % of seats completed.

---

## 6. The drift policy

- **The meter:** the species-usage total-variation distance (TVD) of full-six rated sides, month against month and
  against each panel and the pool, read against a SPLIT-HALF FLOOR (two random halves of one month: what sampling alone
  makes) (0.009 at ≈ 59k sides a half); plus the monthly mean side rating and the ≥ 1500 share.
- **The alarm (to apply to a new snapshot):** a month-to-month TVD above 0.074 (1.5 × the largest within-`hr_spec_v1`
  month-to-month TVD), or a mean-rating shift > 50 Elo between same-population months.
- **What drift invalidates:** a result measured on snapshot S speaks for S's months. Under an alarm, a human-trained
  lever's verdict is re-read on the new snapshot before it steers anything; a panel whose TVD to the newest snapshot
  rises by more than the alarm stops being called "the ladder population" (H is the one that claims it). Results that
  read no human data (self-play, Kakuna on frozen panels) are untouched by corpus drift.
- **Between populations** (archive vs spectator) the distance is selection, not drift: never read it as a meta change.

---

## 7. Where this is heading

1. The toggle and the consumers' snapshot check (the first build; the lever doc §7).
2. X59 (the HOM vs Kakuna, CPU) decides whether `bc` or `teams` is built first; X60 reads it against Kakuna on U.
3. A live spectator collector (Rust live reader; no poke-env) would restore (b)'s volume at ≈ 3,450 games / day,
   lawful only through the owner's proxy and the read-only spectator protocol (no play, no chat); it needs the owner's
   word and is not designed here.
4. (c-human) has a slot and no source; it would be the fill-error gold standard.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **This doc CREATED (DRAFT)**: the owner's requirement of an always-current architecture for human data, split from the lever design (explicit-only) | the four-source taxonomy by information level and selection; snapshots `hr_spec_v1` and `hr_arch_2026-10-24` frozen by manifest sha256; one flag `--human-data` registered in `agents.info_levers`; Q-H1..Q-H12 with their answers; a split-half-floored drift policy | one doc for architecture and experiment (the experiment doc is explicit-only and goes stale by design); pooling (a) and (b) (SKEWED); per-lever flags (the complexity budget) | the card; fd7f4624; X50 R-b1 / R-b2 |
