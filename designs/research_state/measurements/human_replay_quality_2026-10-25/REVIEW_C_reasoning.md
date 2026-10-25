# REVIEW C: is the human-data architecture easy to reason with, and will it stay true? (2026-10-25)

**Under review:** commit `14ae4203` (DRAFT): [`designs/endstate/human_data_architecture.md`](../../../endstate/human_data_architecture.md)
("the arch doc"), its rows in `designs/endstate/README.md` (35, 36) and the root `CLAUDE.md` map, and how it relates
to [`design_information_levers.md`](../../../endstate/design_information_levers.md) §13 and
[`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md) ("the lever doc").
**Lens:** the owner's 2026-10-25 requirement: an easy way to reason about human data "that doesn't require knowing
every implementation detail", in an always-current architecture covering uploaded replays, ones we see online, and any
full-information source. **Method:** a cold read; cross-checks against the registry
(`designs/research_state/human_replay_snapshots.json`), the data card (this folder's `README.md`, `results/results.json`),
`data/teams/`, `production_config.json`, the X50 rung-1b doc and the freshness gate (`src/claude_md_freshness_gate_test.py`).
Docs only; nothing run, nothing edited under review. The two other reviews (data / bias / leakage; experimental /
causal) own their lenses; this one does not re-litigate the numbers.

**Verdict: ACCEPT WITH CHANGES — not yet "easy to reason with", and nothing keeps it true.** The bones are right: one
switch, a snapshot id, a source table by information level and selection, named questions with answers. But the first
page scores **4.5 / 10** on a cold read (the full doc 7 / 10), its headline "no human data enters training, eval or
play" is wrong in a way a reader will act on, the "one snapshot id per lever" model hides the layer where results
actually differ, and no gate ties a single sentence to the code or the registry. Two BLOCKERs, five MAJORs. Both
BLOCKERs are text plus one cheap static test (§4); nothing needs a re-measurement.

---

## 1. The cold-read test

A reader with no implementation knowledge reads ONLY the first page (the status header, "The rules it serves", §1:
lines 1–48), then the whole doc. Score: 1 = answered correctly, 0.5 = partly or only with inference, 0 = not answered
or answered WRONGLY (a misleading answer scores 0).

| # | question | first page alone | score | whole doc | score |
|---|---|---|---:|---|---:|
| 1 | What human data do we HAVE? | rates only ("≈ 3,450 games/day while it ran", "≈ 250–450/day"); no holdings, no snapshot ids | 0.5 | §2–§3: `hr_spec_v1` 376,410 logs, `hr_arch_2026-10-24` 16,490, both frozen | 1 |
| 2 | Is any human data in training TODAY? | "Today no human data enters training, eval or play" | 0.5 | same sentence; wrong as stated (F1): human-BUILT teams are in the pool (`data/teams/others/` are PokePaste teams of named players), Smogon usage (an aggregate of human ladder games) is every prior | 0.5 |
| 3 | Which results depend on which snapshot? | nothing | 0 | §3 "use" column: X50 R1 / R0 / R-b1 / R-b2 and the card on `hr_spec_v1`; drift on both. The archive-skew read (`fd7f4624`, a bare SHA) and the HOM checkpoint are not mapped | 0.5 |
| 4 | What does each toggle value do? | diagram: `bc` "the loss", `kl` "the loss", `teams` "the pool", `judge` "search's opponent likelihood": `bc` and `kl` indistinguishable | 0.5 | §4 table: clear | 1 |
| 5 | Can I type `--human-data` now? | "The levers below are designed, not built" | 1 | §4's "The one flag" paragraph is in the PRESENT tense ("what `--arch production` resolves", "a routine gate runs the smoke under an `open()` audit"): none exists (F5) | 0.5 |
| 6 | What is UNKNOWN? | nothing listed | 0 | §5: Q-H9..Q-H12 unmeasured; A-4 (spectator coverage); the fill error on human teams | 1 |
| 7 | Can an eval result be contaminated by human data? | "eval and play: never read human data (the panels are frozen team sets; Kakuna is an external anchor)" | 0 | §4: "no human data, ever". WRONG: heldout72_hpfix is replay-derived (the lever doc's FINDING 3), Kakuna was trained on human replays; a reader concludes there is no contamination path | 0 |
| 8 | How much of a replay row is real? | fact 1: own side largely invented (≈ 1.9 of 4 moves seen, EVs never); the chosen action is real | 1 | §2 + Q-H5 | 1 |
| 9 | Can the archive train a lever? | fact 2: SKEWED, not the ladder; training use not stated | 0.5 | CONTRADICTORY: §2 (a) "training only as its own registered snapshot" vs §5.1 "DROP for training" (F9) | 0.5 |
| 10 | Is there evidence human data will help goal (b)? | fact 3 (coverage) reads as YES | 0 | Q-H7: "the offline proxy gives no support"; the one negative result is absent from page 1, so a page-1 reader is over-optimistic | 1 |
| | **total** | | **4.5 / 10** | | **7 / 10** |

**Reading.** The first page is a good MECHANISM page (facts 1–2 and the diagram) and a poor STATUS page. The
questions a reasoner asks first (what is true today, what depends on what, what we don't know, what we found) are
on pages 3–5 or nowhere. Q7 is the dangerous one: the doc answers it confidently and wrongly.

---

## 2. Findings

### BLOCKER

**F1 (BLOCKER) — "Today no human data enters training, eval or play" is false as written, and the taxonomy has no
lane for the human data already in the system.** The doc's taxonomy covers human DECISIONS in battle logs. Human data
reaches the system today by three other routes, and a reasoner needs all of them to judge contamination and
"what depends on human data":

| kind of human artifact | in the system today | where | the doc says |
|---|---|---|---|
| human-BUILT teams | YES, training | the self-play pool: `data/teams/others/{giraffe, johnnyg2, mcmegan, schmuck_nick, yak_attack}` (PokePaste, `tools/others_team_downloader`), the Smogon samples; D1's forum `--team-mix` corpus | (d) "selection: ours (the pool)" |
| REPLAY-derived teams | YES, eval | panel heldout72_hpfix = Metamon's hl_05_26, teams extracted from ladder replays (`hp_repair.py` exists for exactly them) | "the panels are frozen team sets" |
| human AGGREGATES | YES, every prior | Smogon usage / sets / teammates / spreads in `data/`; C-GAME's fill | "priors Smogon" (true, but Smogon IS the human ladder, aggregated) |
| human-TRAINED models | YES, eval + offline | Kakuna (Superkazam: 4M human replays); the HOM (P\* fine-tuned on `hr_spec_v1` ≥ 1500, built 2026-10-24) | "Kakuna is an external anchor"; the HOM only via Q-H7 |
| human DECISIONS (rows) | NO (offline measurement only) | `hr_spec_v1` / `hr_arch_2026-10-24` | the whole doc |

**Why it blocks:** the owner's point is reasoning without implementation detail; the doc's own headline gives the
wrong answer to "can this eval be contaminated by human data?" (cold-read Q7) and to "does production already depend
on human data?" (Q2). The lever doc's FINDING 3 ("a human-data lever trains on [H's] POPULATION") is exactly this
interaction, and it is invisible from the architecture doc. **Fix:** replace the headline with "Today no human
DECISION data and no replay-derived team trains, evaluates or plays. Human-built teams (the pool), Smogon's aggregates
(every prior), one replay-derived panel (H) and two human-trained models (Kakuna, the HOM) are already in the system;
§2.0 lists them." Add §2.0, the table above (five rows), as the FIRST table of §2. The four-source taxonomy then
becomes the "human decisions" row expanded.

**F2 (BLOCKER) — nothing keeps the doc true.** It is always-current by declaration only. Checked:
- the freshness gate checks paths, flags and modules; it passes `--human-data` only because `designs/deleted_flags.md`
  lists it as PROPOSED, and checks no claim;
- no test reads `human_replay_snapshots.json` (grep: only the card README names it); the §3 table's ids, file counts
  and manifest SHAs are hand copies;
- no test ties the lever list (`bc`, `teams`, `kl`, `judge`) or the grammar to code, because there is no code; when
  `agents.info_levers.human_data` lands, nothing forces "designed, not built" off the page;
- the arch doc copies the card's headline numbers (§5, §5.1) — principle 4 ("one home per fact … the one exception is a
  copy that a test pins") is broken by every one of them.
**Fix:** the static test of §4 (≈ 80 lines, < 0.1 s, the static tier). Without it the doc decays at the first build,
which is the next unit the lever doc schedules.

### MAJOR

**F3 (MAJOR) — "one flag + one snapshot id per lever" hides the layer that decides which ROWS a run saw.** Two runs
citing `bc:hr_spec_v1@0.1` can train on different data: the rows also depend on the rating floor, the bot rule, the
walk-over bar, the per-account cap, the panel-exclusion set (it CHANGES whenever a panel is registered: forum216b is
already in it), the C-GAME completion code and seed, the extraction commit and the encoder layout hash. The lever
doc's §7.1 `record()` does carry a "rows manifest sha256", but the arch doc's mental model (the diagram, §3, §4) says
"snapshot id", so a reader reasons at the wrong grain. And `kl` / `judge` do not consume a snapshot at all: they
consume a MODEL (the HOM), itself trained on one rowset. **Fix:** make the id chain explicit, three levels, each
registered and each named in the clause:

```
  SNAPSHOT (raw logs, frozen)        hr_spec_v1                       ← the registry, today
     │  filters + panel-exclusion set + C-GAME version + extraction commit + layout hash
     ▼
  ROWSET  (training rows / teams)    hr_spec_v1.r1  (rows manifest sha256)       ← NEW registry section
     │  a fine-tune of P* on the rowset
     ▼
  MODEL   (HOM)                      hom_spec_v1.r1 (checkpoint sha256)          ← for kl / judge
  clause:  bc:<rowset>@<coef> | teams:<rowset>@<share> | kl:<model>@<coef> | judge:<model>
```

A citation names the most-derived id it used; the registry resolves the chain down to the snapshot. Rule: a new panel
registration RETIRES every rowset whose exclusion set it changes (a test can check: each rowset row lists the panel
set it excluded; the panel registry's set must equal it or the rowset is `retired`).

**F4 (MAJOR) — the complexity budget is already broken in the lever doc.** (i) Lever doc §6 adds a second flag,
`--human-data-combined-consent`, also registered PROPOSED in `deleted_flags.md`; the arch doc says "the one flag …
ONE clause". (ii) "At most three registered levers plus `judge`" is four values, and `judge` is CLOSED (R-b2
HOM-NONE): replace-don't-append says a closed lever leaves the grammar until reopened. (iii) The grammar
`<lever>:<snapshot>@<param>` does not fit `judge:<snap>` (no param). (iv) The creep sites the budget does not name:
the filters (F3), the `coef(t)` schedule and the 25 % human-minibatch share in §4.1, the H-TEAMS per-account cap, and
AWR, already queued as "H-BC's variant". **Fix:** the budget names its knobs: ONE clause; the clause's ONE param; every
other knob lives in the ROWSET (F3) or in a pre-registration, never on the command line. Drop the consent flag (a
combination is a new registered lever value, e.g. `bc+teams:…`, replacing a row, or nothing). Drop `judge` from the
live grammar (the decision record keeps it).

**F5 (MAJOR) — IS and WILL are mixed in an always-current doc.** §4's "The one flag" paragraph, the endstate README
row 35 ("registered in `agents.info_levers`") and info_levers §13 state an unbuilt flag, its production default, its
lineage refusal and an `open()` audit gate in the present tense. Q-H12 says "unbuilt". `production_config.json` has no
`human_data` key. A reader cannot tell which provenance guarantees exist today (none do: a run CAN depend on human data
silently today, because nothing would record it — e.g. a hand-built team corpus from replays passed by `--team-mix`).
**Fix:** every §4 sentence carries BUILT / NOT BUILT; the status box (F7) holds one line `switch: NOT BUILT`, which the
test of §4 flips in lockstep with the module.

**F6 (MAJOR) — the full-information slot is under-defined, and "uploaded replays" is read one way only.** (i) The
taxonomy mixes axes: (a) / (b) are acquisition CHANNELS, (c) is an INFORMATION LEVEL, (d) is a PLAYER. (ii) (c-human)
says opponent "FULL or partial" without saying what a lever may do with a known opponent team: if the opponent's
hidden facts are known, using them anywhere in the actor's observation is an oracle leak (the reveal spec's territory);
the only honest uses are the ACTOR's own side (fill-free rows) and measuring the fill error. (iii) The owner's
"uploaded replays" may mean replays a person UPLOADS TO US (the owner's or a friend's game, with their team paste):
that is the one concrete (c-human) source available now, and the doc has no intake route for it (where it lands, its
snapshot id, consent, the per-account cap). **Fix:** two axes plus one attribute:

| who played ↓ / what we know → | public log only | the actor's team true | both teams true |
|---|---|---|---|
| humans (ladder) | (a) archive, (b) spectator | (c1) a submitted replay + the submitter's paste; opt-in players | (c2) tournament replays matched to posted sheets |
| our agent vs bots / Kakuna | — | — | (c-ours) eval traces |
| our agent vs itself | — | — | (d) self-play |

Attribute columns: SELECTION (who chose that this game exists in our data) and VOLUME. Rule beside it: a known opponent
team never enters the actor's observation; it is a label or a measurement only. Registry rows carry `actor_side: true |
public` and `opp_side: true | public` (the lever doc proposes only `own_side`). A (c1) intake is a registered snapshot
`hr_sub_<tag>` cut from a drop folder, never pooled with (a) / (b).

**F7 (MAJOR) — the first page answers mechanism, not status (cold read 4.5 / 10).** **Fix:** open §1 with a STATUS
box (≤ 10 lines, every value pinned by the test of §4 or pointing at its generator), then a RESULT → DATA table, then
the existing diagram and three facts, with the negative finding added as fact 4:

```
STATUS (pinned by src/human_data_doc_gate_test.py)
  human decisions in training / eval / play : none / none / none
  human-built teams in training             : yes (the pool; --team-mix corpora)        → §2.0
  replay-derived teams in eval              : heldout72_hpfix (Metamon hl_05_26)        → §2.0
  human-trained models                      : Kakuna (eval anchor), the HOM (offline)   → §2.0
  the switch --human-data                   : NOT BUILT
  snapshots                                 : hr_spec_v1 (376,410 logs), hr_arch_2026-10-24 (16,490)
  the negative result                       : P* predicts humans as well on unfamiliar teams as on pool-like ones
  open                                      : does any lever help vs Kakuna on U? (X59, then X60); the fill error on human teams
```

| result | data | id |
|---|---|---|
| X50 R1, R0, R-b1, R-b2; the data card | spectator | `hr_spec_v1` (train ≤ 09-20, held out 09-21..23) |
| the HOM | spectator, ≥ 1500, filters of rung 1b | `hr_spec_v1` → (rowset, F3) |
| archive skew (`replay_archive_bias_2026-10-21`) | archive pulls before the registry | none (predates; cite the folder, not a SHA) |
| drift (card §F) | both | `hr_spec_v1`, `hr_arch_2026-10-24` |
| every self-play / Kakuna result | none | `none` |

### MINOR

**F8 (MINOR) — Q-H1..Q-H12 cover the owner's list but miss three and blur one.** Owner's list → collection Q-H1,
snapshots Q-H2, drift Q-H8, human vs self-play Q-H9, human-only model Q-H10: all present. Missing:
- **Q-H13 CONTAMINATION:** which eval panels and anchors already carry human data, and what does a lever's gain on
  them mean? (F1; the lever doc's FINDING 3 is its only answer, outside the always-current doc.)
- **Q-H14 PROVENANCE:** given a checkpoint, did any ancestor train on human data, and on which rowset? (Answer today:
  unanswerable; after the build: `main.lineage`.) Q-H12 asks whether it CAN happen silently, not how to READ it.
- **Q-H15 TEACHER QUALITY:** how strong is the population we imitate, relative to P\* and Kakuna? (A-3's basis; HB0
  and X59 bear on it.)
Blurred: **Q-H9 and Q-H11 overlap** ("better than self-play" vs "does it help"), both answered "X60". Sharpen Q-H9 to
the EXCHANGE RATE (how many self-play steps is one human row worth at matched GPU budget) or merge it into Q-H11.
Unanswerable as posed: **Q-H4** ("is the sample the ladder") cannot be answered without a revived collector (A-4); say
so in its answer and name what would answer it. **Q-H1** says "legally" and its answer never addresses legality (the
proxy grant is "download only"; joining rooms needs the owner): answer it in one clause.

**F9 (MINOR) — the archive's training status contradicts itself.** §2 (a) "training only as its own registered
snapshot" vs §5.1 "DROP for training". Pick one (the card's DROP), and say what would re-open it.

**F10 (MINOR) — one home per fact is broken three ways.** The clause grammar, the hash prefix and the provenance rules
live in the arch doc §4, info_levers §13 AND the lever doc §6–§7.1; the inheritance rule already differs between them
(arch doc: "never inherited silently by a resume"; lever doc §7.1: "a fork inherits `off` unless typed", while root
`CLAUDE.md` says a fork inherits every untyped flag — so forking an ON run silently drops the lever). **Fix:** the arch
doc owns the grammar and the rules; §13 keeps two lines and a pointer; the lever doc points. Make the inheritance rule
symmetric and loud: resuming or forking a run whose lineage has a human-data clause REFUSES unless `--human-data` is
typed (the clause or `off`).

**F11 (MINOR) — §5.1 copies the card's table into an always-current doc.** Measurement tables belong to the measurement
(principle 2, 4). Keep three pinned headline numbers in the status box and point at the card.

**F12 (MINOR) — small accuracy and linking defects.** The spectator rate is ≈ 3,450 (§1, §7), ≈ 3,500 (Q-H1) and
3,497 (the card); the archive rate "≈ 250–450 / day" (§1) vs 365 (Q-H1, range 160–714). `fd7f4624` is a bare commit
SHA (link `replay_archive_bias_2026-10-21/README.md`). The lever doc §6 cites "the drift alarm (§3.9)", a section
that does not exist (the alarm is the arch doc §6; the freshness gate's section check skips a bare `§`).

**F13 (MINOR, outside the lens; a hazard per rule 7) — committed account names labelled "suspected bot".**
`results/results.json` (`accounts.flagged`) and the card README name 61 public Showdown accounts as suspected bots by a
heuristic the card itself says catches "heavy human grinders it cannot tell from scripts". The per-account cap needs
ids, not names: consider a salted hash in committed artifacts. FINDING for the data / bias review and the owner; not
changed here.

**F14 (MINOR) — the drift alarm's constants have no home.** 0.074 and the 0.009 floor are derived from `hr_spec_v1`;
put them in that snapshot's registry row (`drift_reference`) so a new snapshot is compared by a tool, not by prose.

---

## 3. Will it stay current? What forces an update today

| a change in … | what forces the doc to change today | after the test of §4 |
|---|---|---|
| the snapshot registry (a new / retired id) | nothing | T1 fails |
| the switch being built | nothing (the freshness gate stays green: the flag resolves either way) | T2 fails until the status box and §4 say BUILT and `deleted_flags.md` drops the PROPOSED rows |
| the lever list / grammar in code | nothing | T2 compares the module's tuple to the doc's grammar line |
| production's default | nothing | T4 |
| the card's numbers | nothing | T3 pins the status box to `results.json` / the registry |
| a new panel (exclusion set) | nothing | T5 (once rowsets exist) |
| a deleted path or flag the doc names | the freshness gate (checks 1–3) | unchanged |

---

## 4. The enforcing test (proposed: `src/human_data_doc_gate_test.py`, static tier, `pytest.mark.static`)

Cheap (reads three JSON files and two markdown files; no torch import). Each check names the file to fix.

- **T1 snapshots ↔ registry.** Parse the arch doc §3 table rows (`` | `hr_…` | … | ``). The set of ids equals the
  registry's `snapshots` keys; per id, `files` and `manifest_sha256` equal the registry's. Every `hr_[a-z0-9_.-]+` token
  in `designs/endstate/*.md` and `UNDERSTANDING.md` is a registered id, or a placeholder containing `<` or `*`. At most
  two ids carry `status` `reference` / `candidate` (the "two live" rule; add the field).
- **T2 build-state lockstep.** `spec = importlib.util.find_spec("agents.info_levers.human_data")`. If `spec is None`:
  the status box line reads `NOT BUILT`, and `--human-data` is a PROPOSED row of `designs/deleted_flags.md`. If not
  None: the line reads `BUILT`, no PROPOSED row remains, and the module's `LEVERS` (the canonical clause heads) equal the
  lever set parsed from the arch doc's single grammar line.
- **T3 pinned headline numbers.** The status box's counts (376,410; 16,490) equal the registry; any card figure kept in
  the status box equals `results/results.json` at its key (a small dict in the test: figure → JSON path).
- **T4 production is off.** Pre-build: `production_config.json` has no `human_data` key (the doc's "what `--arch
  production` resolves" must then be marked NOT BUILT). Post-build: its resolved `human_data == "off"`.
- **T5 (with rowsets, F3).** Every live rowset's recorded panel-exclusion set equals the current panel registry's set;
  otherwise the rowset is `retired`.
- **A meta test per check** (a planted mismatch fails), the freshness gate's pattern, so no check is vacuous
  (`test_stub_vacuity_gate_test.py` would demand it anyway).

Cheaper alternative if a new static file is unwanted: T1 and T2 as two functions inside the freshness gate (it already
owns "docs vs tree"); the static-gate count does not move.

---

## 5. Relation to the neighbouring docs

- **`design_information_levers.md` §13:** consistent on the grammar and the hash; it restates the arch doc (F10).
  Its "What it is NOT" bullet (no hidden fact revealed; never combined with `--oracle-reveal`) is the clearest
  statement of the boundary in either doc: move it into the arch doc §4 and point from §13.
- **The X50 rung-1b doc:** its §1.3 scope (HOM as judge only) is superseded and the arch doc says so; its §3 rule
  "[archive pulls] never enter" agrees with the card's DROP and disagrees with the arch doc §2 (a) (F9). R-b2's
  HOM-NONE argues for `judge` leaving the live grammar (F4).
- **The lever doc:** properly explicit-only and pointing at the arch doc; its §6 budget and §7.1 inheritance rule are
  where the contradictions sit (F4, F10). Its FINDING 3 belongs in the arch doc's always-current view (F1).

---

## 6. FINDINGS (standing rule 7)

1. Nothing was run: the review is a read of docs, the registry, `results.json`, `data/teams/` listings and
   `production_config.json`. The cold-read scores are this reviewer's judgement against the stated rubric.
2. That `data/teams/others/` teams are in the self-play pool today is inferred from their location and the downloader;
   the pool's exact composition (the 719) was not re-derived here. **UNVERIFIED** in detail; the F1 conclusion holds if
   any pool team is human-built, which the Smogon samples alone guarantee.
3. F13 (committed account names) is outside this lens and is reported, not acted on.
4. The two adversarial reviews the arch doc names are separate units; this review neither substitutes for nor reads
   them.
