# X50 S0: the human-replay census on the Rust reader (2026-10-20)

**What this is.** The zero-GPU scoping read of X50
([`../../../endstate/design_bot_ideas_screen.md`](../../../endstate/design_bot_ideas_screen.md) §2.4), re-measuring
ai_v11's 2026-08-18 census ([`../../../ai_v11/design_human_replay_objectives.md`](../../../ai_v11/design_human_replay_objectives.md)
§5) on the whole corpus with today's Rust reader, plus the count ai_v11 never made: how many OPPONENT-INTENT labels
the corpus yields. No GPU, no model, no training. Read-only on `data/` and on the corpus.

**Verdict (S0, the registered floor): FEASIBLE.** ai_v11's floor was at least 50k admitted tier-A decisions at
rating 1500+. The corpus has **489,973** (458,354 of them also hazard-clean), about 10x the floor. **But no volume buys
a complete own side.** Every own mon showing 4 moves happens on **0.03 %** of decisions, and EVs on none. So every
behaviour-cloning input still has to be COMPLETED.

**The intent use case is the clean one.** At 1500+ there are 2.44M viewer turn decisions. Of the opponent's actions
on them:
- **86.0 %** are labelable as the opponent's actual choice (2.10M labels; 44.8 % of the decisions, 1.10M, have a
  target that was already revealed);
- **69.4 %** are labelable under the production label rules (1.69M; 36.6 % already revealed).

The read covers all 376,410 logs (109 date folders, 2026-05-18 → 2026-09-23) and 752,820 seat reads at HEAD
`dfeb3d92d`. It took 0.58 worker-pool hours of compute and 4 h 10 min of wall, most of that waiting in the
heavy-slot queue. Every table is in [`results/census.md`](results/census.md) (generated); the numbers below are
copied from it.

## Results

### The reader
**Zero refusals and zero encoder failures** over 752,820 seat reads and 19,804,295 per-turn encoder probes. The
one classified refusal is the known spectator-only renamed-seat replay (gate (b), 2026-10-07; its end view fails
too). ai_v11 lost 3.15 % of 1500+ sides to `UnknownVolatileError` on the deleted Python reader; that loss is gone.

### The acting side's set, revealed by game end (decisions = turn decisions + forced replacements / Baton Pass targets)

| band (acting side) | decisions | full team | tier A (+ acting mon 4 moves) | + every mon 4 moves | + every item | + every ability |
|---|---|---|---|---|---|---|
| < 1300 | 10,580,747 | 69.17 % | 1,504,833 (14.22 %) | 0.03 % | 0.03 % | 0.02 % |
| 1300–1499 | 7,553,141 | 76.99 % | 1,201,038 (15.90 %) | 0.02 % | 0.01 % | 0.00 % |
| **1500+** | **2,727,459** | **80.16 %** | **489,973 (17.96 %)** | 792 (0.03 %) | 792 (0.03 %) | 0 |
| 1700+ (⊂ 1500+) | 142,810 | 80.07 % | 29,714 (20.81 %) | 0 | 0 | 0 |
| all (incl. unrated) | 21,346,129 | 73.60 % | 3,287,279 (15.40 %) | 0.03 % | 0.03 % | 0.01 % |

Per own mon at 1500+ (437,565 mons):

| quantity | share of mons |
|---|---|
| 0 / 1 / 2 / 3 / 4 moves shown | 9.2 % / 30.0 % / 30.6 % / 21.2 % / 9.0 % |
| item known | **69.7 %** |
| ability known, any route | 65.8 % |
| mons of a one-ability species (known from the species alone) | 72.8 % |
| ability known on a two-ability species | 0.29 % |

Full team revealed by side: 68.6 % of 1500+ sides (all bands: 56.5 %).

**Selection bias** (tier-A share, lost vs won, at 1500+): 18.24 % vs 17.53 %, a ratio of **1.04** (all bands: 1.08).
**Mask-hazard proxy** at 1500+: **2.48 %** of turn decisions. The parts are a trapper 1.91 %, Taunt 0.32 %, Mean
Look family 0.13 %, a Choice Band holder that already moved 0.05 %, and the rest under 0.05 % each. The proxy
undercounts.

### The opponent-intent labels (band = the LABELLED human)

| band | viewer turn decisions | CHOICE labelled | of which target already revealed | PROD labelled | of which revealed | Hidden Power (unlabelable) |
|---|---|---|---|---|---|---|
| < 1300 | 9,445,403 | 84.83 % | 39.79 % | 68.23 % | 32.19 % | 4.20 % |
| 1300–1499 | 6,730,868 | 85.30 % | 42.05 % | 68.41 % | 34.02 % | 4.34 % |
| **1500+** | **2,443,031** | **85.99 % (2,100,663)** | **44.82 % (1,095,000)** | **69.35 % (1,694,178)** | **36.59 % (893,821)** | 4.30 % |
| 1700+ | 129,557 | 87.41 % | 48.40 % | 71.61 % | 40.24 % | 3.95 % |
| all | 19,051,444 | 85.15 % | 41.32 % | 68.43 % | 33.47 % | 4.25 % |

At 1500+ the 2.44M opponent turn actions break down as follows (CHOICE):
- **moves 63.2 %**: 31.5 % already revealed, 31.7 % a first use, plus Hidden Power 4.3 %;
- **switches 22.7 %**: 13.4 % to a revealed mon, 9.3 % to a mon never seen before;
- a `cant` that names the move 0.15 %.
- **Not labelable (14.0 %)**: Hidden Power 4.3 %, fainted before acting 5.4 %, a hidden `cant` (sleep, paralysis,
  flinch, freeze) 3.2 %, and no action 1.1 % (forfeits and the end of the game).

PROD masks 16.6 pp more than CHOICE. That is 11.9 % of decisions whose window closes on the viewer's own forced
switch, and 11.4 % where the opponent's switch co-occurs with a move, faint or drag (mostly a KO and its replacement).
The two overlap with each other and with the hidden choices, so the shares do not add up to the difference.

### Against ai_v11 (2026-08-18, Python reader, 263k logs, ≈ 1,142 sides at 1500+)

| quantity | ai_v11 | this census | why they differ |
|---|---|---|---|
| sides that fail to parse | 3.15 % | **0** (1 classified spectator case in 752,820) | the Rust reader |
| own item known | **3.93 %** of own mons | **69.7 %** (1500+) | the Rust reader knows items revealed by `[from] item:` lines (overwhelmingly Leftovers). The 3.93 % was a property of the deleted reader (F-X50-2) |
| own ability known | 7.86 % | 65.8 % (any route); 0.29 % on a two-ability species | the Rust view counts a one-ability species as known |
| mons at 4/4 moves | 8.64 % | 9.02 % | agrees |
| sides with all 6 revealed | 67.2 % | 68.6 % | agrees |
| tier A share | 16.70 % | 17.96 % | close; the decision definitions differ (ai_v11 counted the first voluntary action per turn; this counts every turn decision plus the forced ones) |
| tier A at 1500+, count | ≈ 259k (extrapolated) | **489,973** (counted) | a counted whole corpus, 43 % larger, and the wider decision definition |
| loss enrichment of tier A | 1.29 | **1.04** | ai_v11's n was 1,142 sides |
| α-label coverage | 92.04 % "paired" turns | 85.99 % CHOICE / 69.35 % PROD | ai_v11 counted turns where both sides had a reconstructed decision, not whether the opponent's action is a usable label |

## Recommendation for the owner's Q1 (re-open the human-replay design?)

**Owner, 2026-10-20: rung 1 re-opened (labels only).** Its design and the registered first read:
[`../../../endstate/design_x50_rung1_intent_labels.md`](../../../endstate/design_x50_rung1_intent_labels.md); F-X50-5's request
synthesiser is built there.

**Re-open it narrowly: rung 1 only, the human-opponent intent labels. Keep the behaviour-cloning warm start
(rung 3, X50 S1) closed for now.**

The label side is clean and dense. At 1500+:
- 86 % of the opponent's turn actions are observable choices: 2.1M labels, half of them on targets the head can
  address deterministically;
- the opponent half of the input is in its native distribution;
- the per-turn read leaks nothing, provided the opponent's set is never completed from the future.

The behaviour-cloning case fails on a different axis than volume:
- tier A clears ai_v11's floor about 10x;
- but a decision whose WHOLE own side is known essentially does not exist: every mon at 4 moves on 0.03 %, and EVs
  never;
- so every BC row is a completion, and S1 would measure the completion as much as the human play.

The next rung-1 step is still zero-GPU: ai_v11's P0.2 / P0.3 offline probe. Score the current intent head on
held-out human decisions against the Smogon prior. That needs a build first, which is also B-BC's core: a per-side
request synthesiser for spectator replays (F-X50-5), and an own-side override of the fabricated spread (F-X50-4).
Train on human labels only if that probe reads something.


## Method

**Corpus.** The project's own download of public gen3ou replays, `<main>/replays/showdown/gen3ou/<date>/*.log`
(gitignored; 109 date folders, 2026-05-18 → 2026-09-23). Every log, no sampling.

**Three passes per replay, in `census.py`:**

1. **The Rust reader, both seats** (`main.live.replay_scan.scan_one`: `live_reader`, the chain training's rows come
   from; the live client's line filter; one write per turn; an encoder PROBE after every turn). It counts refusals
   and encoder failures. This is gate (b) of the live reader, re-run at today's HEAD.
2. **The Rust end view, both seats** (`core_events --present-stream`: the seat's `present()` view after the whole
   log). For each side, the view's OWN team says which mons were revealed, and per mon the moves, item (held or
   consumed) and ability the reader knows by game end. An ability counts as known if the reader has it from any
   source; the species-has-one-ability share is reported beside it.
3. **A text timeline** over the same filtered lines. Per side per turn it records the human's action (move, switch,
   a `cant` that names the move, a hidden `cant`, a locked turn, fainted before acting, no action), the forced
   decisions (faint replacements, Baton Pass targets), a mask-hazard proxy, and the opponent-intent label. The reader
   exposes no per-turn action record for a spectator stream (no `|request|`, so it never "decides"), so this pass
   is text. It is held to the reader: every move it attributes to a mon must be in that mon's end-of-game moveset
   in the Rust view, and every switch target must be in the team (section 1 of `results/census.md`).

**Definitions.**
- A **decision** is one per side per turn (`td`; the reader's live session opens one per `|request|`, and every turn
  sends one to each side) plus each forced replacement or Baton Pass target (`fd`).
- The **acting side** is the side that decides. Its rating (the `|player|` line, pre-battle) sets the band:
  `<1300`, `1300-1499`, `1500+`, with `1700+` a subset of `1500+`, plus `unrated`.
- **Tier A** is ai_v11's definition: all of the acting side's mons revealed by game end, and the acting mon showed 4
  moves. The stricter columns are nested: every mon 4 moves, then every item, then every ability. EVs, IVs and
  nature are never in a replay.
- **The mask-hazard proxy** (ai_v11 P0.4, per turn decision) flags a decision where a synthesised legal set would
  be over-permissive. The flags are: a trapper on the other side (Arena Trap, Shadow Tag or Magnet Pull, by species,
  type and Levitate); Mean Look, Block or Spider Web (`-activate|trapped`); a partial trap; Ingrain; Taunt, Encore,
  Disable or Torment active; a locked turn (recharge, charge, rampage); a Choice Band holder (known by game end) that
  already moved. It UNDERCOUNTS: 0 PP and Choice Bands that were never revealed are invisible.
- **The intent label** goes on each viewer turn decision, and it is the OPPONENT's action that turn. The band is the
  LABELLED human's rating. Two semantics:
  - **CHOICE** is the opponent's observable turn choice: a move, a switch, or a `cant` that names the move.
  - **PROD** is the production rule set (`agents/training/opp_intent_labels.py` and
    `rust_sim/src/trackers/mod.rs::IntentLabel::build`), APPROXIMATED on the text. It masks: a window that closes on
    the viewer's own forced switch (the viewer's mon fainted, or it Baton Passed), a switch that co-occurs with a
    move, faint, drag or replacement, an Encore override, and every hidden choice. A rampage continuation
    (`[from]lockedmove`) is labelled MOVE, because a move resolved.
  - `_rev` means the move or switch target was already revealed before the decision, so alpha's seat or beta's slot
    exists deterministically. `_new` means its first reveal, so alpha can match it only if the belief put that move
    in its top-K seats (`alpha_mask_rate`, which needs a model forward; unmeasured here). `MOVE_hp` is Hidden Power,
    which is NOT counted as labelled: training resolves it to the TYPED num from the truth team, and a replay never
    states the type.

**Reproduce.**
```bash
cargo build --release --bin live_reader --bin core_events --manifest-path src/rust_sim/Cargo.toml
X50_DEADLINE_EPOCH=$(( $(date +%s) + 8*3600 )) setsid nohup timeout 30000 \
  designs/research_state/measurements/x50_replay_census_2026-10-20/run_census.sh > run.log 2>&1 &
python designs/research_state/measurements/x50_replay_census_2026-10-20/census.py aggregate \
  --shards designs/research_state/measurements/x50_replay_census_2026-10-20/shards \
  --out designs/research_state/measurements/x50_replay_census_2026-10-20/results
```
`run_census.sh` is resumable (a date with a shard is skipped). It holds ONE heavy slot per chunk of 20 dates, each
chunk bounded at 45 min of work, at `nice -n 19` on the agent cores. It stops at the registered epoch: this run
registered **8 h** from 04:11 PDT, so 12:11 PDT. The shards (`shards/<date>.json.gz`, per-band counters) are
committed, so `aggregate` reproduces `results/` without the corpus.

| file | what |
|---|---|
| `census.py` | the three passes and the counters (`run`, `aggregate`, `dates`) |
| `census_report.py` | the readout: counters → `results/census.json` + `results/census.md` |
| `run_census.sh` | the detached, chunked, resumable driver with the registered stop |
| `shards/` | one gzipped counter file per date folder |
| `results/census.md` | every table (generated) |
| `results/census.json` | counters, derived numbers, meta |
| `run.log` | the driver's log |

## Leakage: what the model would see, what the label needs

The pipeline this census implies feeds a replay to the reader TURN BY TURN from the viewer's seat, encodes the
viewer's observation at its decision, and takes the label from the next turn's lines. That is training's own
shift-by-one: the intent label at decision *t* is the opponent's action in the window that *t* opens.

| | in live play | from a replay, read turn by turn | leak? |
|---|---|---|---|
| **opponent side** (the labelled human) | what the protocol revealed so far | exactly the same: the reader folds only the lines fed so far (ai_v11's O10) | **none**, as long as nothing from later turns is injected. Never complete the opponent's set with its end-of-game reveals: a `_new` label's move would then already sit in a revealed seat, and a `SWITCH_new` target would already be a revealed slot. That would put the label in the input |
| **viewer's own side** | the full team sheet: 6 mons, 4 moves, item, ability, the exact spread and stats, and the `\|request\|` (legal set, PP, trapped) | only what the viewer's own mons have revealed so far, and **no request at all**. A spectator chain never decides | not a leak but **out of distribution**. The encoder writes a FABRICATED own spread (IVs 31, EVs 0, nature serious) and asserts it as known (`spread_known`; `rust_sim/src/encoder/slot.rs::spread`). That is ai_v11's O3, still present on the Rust reader |
| **viewer's own side, completed with its end-of-game reveals** (ai_v11's turn-1 injection) | - | content-faithful (the player knew its own team), but still partial (§3: tier A is a minority; every mon at 4 moves ≈ 0) | **a selection leak**: whether a mon or move is in the injected set depends on the game's future (a mon that never came in is absent; §4 measures the outcome skew). It is a weak leak of game length and result into the input. The intent head's input reads our side too (alpha reads our outgoing physics, ai_v11 §3 rung 1), so it is exposed to this |
| **the label** (the opponent's action at *t*) | the env core's delta, from the server's truth | the opponent's lines in turn *t* | none, by construction. Two gaps: a TYPED Hidden Power label cannot be made (the replay never states the type; training resolves it from the truth team), and a `cant` that hides the choice (sleep, paralysis, flinch) has no label |
| **room / meta lines** (rating, names, chat, timers, `\|win\|`) | - | dropped by the live client's declared filter, or never encoded; the read stops at `\|win\|` | none found |

**What a model needs that a replay cannot give.** Our observation always carries our full sets and a server request.
A replay gives the acting side's set only as far as the game revealed it. It never gives a request, and it never
gives an EV spread. So every decision row needs (a) a SYNTHESISED request (a legal set; §5 measures how often it
would be over-permissive), and (b) a COMPLETED own side. Under the owner's rule the completed inputs are read by the
network, so they must be Smogon-derived; the labels are not affected by that rule.


## FINDINGS (standing rule 7)

- **F-X50-1 (the reader, PASS).** The Rust reader reads the whole corpus from both seats with no refusal and no
  encoder failure (one classified spectator-only case). It is the same result as gate (b) on 2026-10-07, now at
  `dfeb3d92d`.
- **F-X50-2 (ai_v11's item number is OVERTURNED).** "Own item known on 3.93 %" was a property of the deleted Python
  reader: the Rust reader knows the item on 63.7 % of own mons (69.7 % at 1500+). Why the old reader missed them
  (most likely it did not carry `[from] item:` reveals into its injected team) is **UNVERIFIED**: the code is
  deleted, so this census did not inspect it. `designs/research_state/README.md`'s human-data row quoted 3.93 %; it
  now points here.
- **F-X50-3 (no complete own side at any volume).** All six own mons at 4 moves: 0.03 % of decisions. Two-ability
  species' abilities: 0.29 % of mons. EVs / IVs / nature: never. The observation encodes every own mon's set, so a
  behaviour-cloning row needs the non-acting mons completed, even on tier A.
- **F-X50-4 (HAZARD: the fabricated own spread is still there).** A spectator reading asserts `spread_known` on
  its own side, and the encoder writes IVs 31, EVs 0, nature serious as KNOWN (`rust_sim/src/encoder/slot.rs`
  `spread`, reached through `present/view.rs`'s `spread_known: own`). This is ai_v11's O3, carried into the Rust
  reader. Any replay pipeline (BC or intent) must override it with a Smogon-derived completion, or the intent head's
  outgoing-physics inputs are computed on fabricated stats.
- **F-X50-5 (a build gap: no decision rows from a replay).** The reader decides only at a `|request|`, and a
  spectator replay has none. So nothing of ours turns a replay into decision rows, and the production intent
  labeller (the trackers fold at decisions) cannot run on one. Both rungs need a per-side REQUEST SYNTHESISER: the
  core of B-BC, and of any rung-1 pipeline. This census's PROD column is therefore a text approximation of the
  production rules, **UNVERIFIED** against the production labeller.
- **F-X50-6 (Hidden Power labels).** 4.3 % of opponent turn actions are Hidden Power. Training labels them with
  the TYPED num from the truth team, and a replay never states the type. They must be masked, or given a declared
  bare class; never a guessed type.
- **F-X50-7 (the production rules cost 16.6 pp of labels on replays).** Mostly from windows that close on the
  viewer's own forced switch (11.9 %) and from switch co-occurrence (11.4 %, mostly a KO and its replacement). In
  those turns the opponent's own turn choice is still observable in the log. Whether an offline rung-1 pipeline
  keeps the production rules (matching training) or the choice rule (more labels) is a design decision for its
  pre-registration.
- **F-X50-8 (selection bias is weak).** Tier A is loss-enriched by 1.04 at 1500+, against ai_v11's 1.29 (n = 1,142
  sides). It still needs outcome-balanced weights for any outcome-labelled objective.
- **F-X50-9 (the mask hazard is small per decision).** ai_v11's 54 % of games with a trapper on the field becomes
  2.5 % of 1500+ turn decisions under any proxy flag. The proxy UNDERCOUNTS: 0 PP and unrevealed Choice Bands are
  invisible to it.
- **F-X50-10 (the text pass is held to the reader).**
  - 99.82 % of the moves the timeline attributes to a mon are in that mon's end-of-game moveset in the Rust view.
    The 23,724 misses (0.18 %) are, in a 1,000-replay sample, Struggle, Transform copies and a handful of
    Umbreon lines that were not root-caused.
  - 100 % of switch targets are in the team.
  - 54 timeline anomaly lines in 376,410 replays.
- **F-X50-11 (HAZARD, ops: heavy-slot FIFO bypass).** `~/.cache/gen3ai/ws_stage1/runner/` runs a FROZEN pre-FIFO
  copy of `utils/heavy_slots.py` (no `ticket` in it) in a loop (`loop_s1.sh`). Each of its chunks is `timeout 3h`,
  against the 45-min policy. It re-took slot 0 at least three times, about 07:00, 07:30 and 07:40 PDT, while this
  census was at the head of a 13-waiter queue. This census waited 2 h 15 min for its second slot. Not touched;
  reported for the orchestrator.
- **F-X50-12 (minor, UNVERIFIED).** The reader's `ability_public` did not fire on Tyranitar's
  `|-weather|…|[from] ability: Sand Stream|[of] …` line, which was seen in a sample. This is harmless in gen-3 OU,
  where the weather setters have one ability. Whether other `[from] ability:` paths also skip the flag was not
  checked.
- **F-X50-13 (the corpus is stale).** The newest date folder is 2026-09-23: the replay collector appears to have
  stopped again (ai_v11's D4 again, 27 days at this census).
- **F-X50-14 (thin top band).** 1700+ is 142,810 decisions (29,714 tier A; 129,557 intent decisions). A
  "strong-human only" read is a 1500+ read.
- **F-X50-15 (process).**
  - `data/` was only read: `gen3_species.json` for types, and `gen3_ability_priors.json` for Levitate and the
    one-ability species.
  - The corpus was only read.
  - No GPU (`CUDA_VISIBLE_DEVICES=""`), no `models/`.
  - The driver was restarted once by its own PID group, to run every remaining date as one ≤ 45-min chunk instead of
    re-queuing at the back per 20 dates. It is recorded in `run.log`.

