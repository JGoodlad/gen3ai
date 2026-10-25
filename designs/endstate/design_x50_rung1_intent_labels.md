# X50 rung 1: human opponent-action labels for the intent head (design + the registered first read)

**Status (2026-10-20):** DESIGN + REGISTRATION; extractor BUILT; **R1 READ: GAP** (Δ NLL(H − S) +0.188 nats
[+0.164, +0.213]; a switch gap; [`the read`](../research_state/measurements/x50_r1_intent_read_2026-10-20/README.md)). **The training-use design** (R1's next step) **is [`design_x50_rung1b_human_opponent_model.md`](design_x50_rung1b_human_opponent_model.md)** (2026-10-23, pre-registered; READ by 2026-10-24: R0 FIXED, the HOM built, R-b1 LEARNED, **R-b2 HOM-NONE**, the lever CLOSED for search at P\*; R-a1, the offline refit of THIS head, is not read): F-R1-1 fixed first by a Smogon completion of the own side, then a search-only human opponent model, then this head refit offline. The owner re-opened the human-replay ladder at **rung 1 only**
(2026-10-20): opponent-action labels from real ladder replays, for the network's opponent-intent prediction. Rungs 2+
(anything beyond labels: behaviour cloning, a human prior) stay closed. This doc is the census's follow-up
([`x50_replay_census_2026-10-20`](../research_state/measurements/x50_replay_census_2026-10-20/README.md), its
recommendation), and the X50 row of [`design_bot_ideas_screen.md`](design_bot_ideas_screen.md) §2.4. Results land in
[`x50_r1_intent_read_2026-10-20`](../research_state/measurements/x50_r1_intent_read_2026-10-20/README.md).

**The rule this unit lives under.** Ladder replays are an allowed source of TRAINING LABELS and EVALUATION DATA.
Nothing derived here ships as a PRIOR the network reads at play, and nothing biases toward the pool. Every input
the network reads in this pipeline is the production encoder's reading of the log's public lines.

## 1. What a label is

**A label is the production intent label, made by the production labeller, on a replay.** At each decision of the
VIEWER (the seat we read the log from), the label is what the OPPONENT then did in the window that decision opens,
under training's DECISION-vs-CONSEQUENCE rules (`agents/training/opp_intent_labels.py`'s table; the Rust
`trackers::IntentLabel::build`, folded by `SideTrackers::decide` at the viewer's NEXT decision, the shift-by-one
`align_labels_to_predictions` makes in training):

| the window shows | label |
|---|---|
| the opponent's move resolved (a called move: the CALLER) | MOVE, the move's num |
| a voluntary switch | SWITCH, the revealed slot or the species num |
| a phaze, a drag, a faint replacement, a window that closes on the viewer's own forced switch, an Encore override, a hidden `cant` (sleep, paralysis, flinch, freeze), no action | UNKNOWN (masked) |

**Into OUR action space as the head sees it.** The flat pointer's column is resolved against the model's OWN
candidates at that row by `agents.model.flat_intent.flat_intent_targets`, the function training's loss uses: a move in
the active's K seats → its seat; a move beyond them → OTHER_move; a revealed switch-in → its slot; a hidden switch-in →
the hypothesis slot holding its species, else OTHER_species; a choice outside every candidate's support → UNMODELED
(masked, counted). So a label names the same column it would name in training.

**Hidden Power.** A replay never states the type; training resolves a bare `hiddenpower` to the typed num from the
truth team (`rust_env/src/labels/intent.rs`). Here a bare Hidden Power label stays num 237 and is EXCLUDED from the
primary read (its own declared stratum, F-X50-6); it is never given a guessed type.

## 2. Hidden state: when a label is valid

A label is valid only where the viewer's information state at the decision is reconstructed from the log's public
lines, and nothing later.

- **The opponent side (the labelled human) is exact.** The reader folds only the lines up to the decision point.
  Nothing from a later line, and no end-of-game reveal, ever enters the row. Our live play sees exactly these lines.
- **The viewer's own side is PUBLIC-ONLY, never completed.** A replay carries no `|request|`, so the reader is
  OPENED with no team, and its own side is what its own mons revealed so far, plus the reader's fabricated own spread
  (IVs 31, EVs 0, nature serious, asserted as known; F-X50-4). Completing it from the end of the game would be a
  selection leak (the census's leakage table). This is OUT OF DISTRIBUTION for the head, not a leak; §5's control
  measures its cost on battles where the full side is known.
- **The decision itself is synthesised** (F-X50-5, the request synthesiser). At each viewer decision point (§3) the
  extractor feeds the reader ONE synthesised `|request|` line built by the Rust reader from its own current reading
  (`side_reader::spectator_request`): the own active's REVEALED moves at their read PP, a switch to each revealed
  alive benched mon, `forceSwitch` on a forced decision, and the flag `"spectator":true`. The flag makes the board
  take the request's LEGALITY only: no team update, no HP, no item, no ability, no PP sync (the request carries no
  information the public lines did not). Where nothing would be legal (the lead before it moved, alone), Struggle is
  the declared placeholder so the decision still opens and the trackers fold every turn; such rows are flagged.
- **The guard (the extractor's prefix guard).** For a row to be valid, a FRESH reader fed only the lines before the
  decision point plus that synthesised request must produce a byte-identical row. The extractor runs it on ONE
  decision of every seat of every battle (chosen by a hash of the battle and seat), and on every decision in the
  tests (`src/main/x50_r1/extract_test.py`, which also holds a leaky row it must refuse). A
  mismatch is a refusal, never a skip; it is what fails if the extractor ever folds a line past the decision.

## 3. Decision points on a spectator log

The viewer gets a decision exactly where Showdown sends it a non-`wait` request:

1. **turn**: right after each `|turn|N` line;
2. **forced**: the viewer's replacement after its active fainted, or its Baton Pass resolved. The decision sits at the
   START of the chunk that carries the replacement `|switch|`: back from that switch line to the chunk's `|t:|` line,
   and before the bare `|` that opens it (a double KO's two switches share one chunk). A replacement with no new
   chunk since its trigger was chosen BEFORE (gen 2-4: a switch continues after a Pursuit KO): no decision.

No decision at the opponent's forced switch (the viewer's request is `wait`), and none after `|win|` / `|tie|`.
**Validated on our own battles:** on X26's eval traces, which carry both seats' real streams, the rule must
reproduce the trainee's real non-`wait` request positions exactly (`points_test.py`, and a refusal the extractor
raises on every trace it reads). A spectator view of a trace is the two seat streams ALIGNED (HP figures folded): a
paired line is taken from the stream of the side it does NOT name (the public version), and a line one seat holds
alone (its `[silent]` ability, its `|error|`, a `|-hint|`) is dropped. Checked on 405 traces before the build landed:
405 / 405 point sequences equal, 405 / 405 label sequences equal between the spectator and full reads, and 404 / 405
full reads byte-equal to the rows the trainee played on (the one: a 250-turn draw whose last row the trace omits).

## 4. Population, rating filter, held-out dates

- **Population: the spectator manifest only** (`replays/showdown/gen3ou/SPECTATOR_MANIFEST_2026-09-23.txt`, 376,410
  logs), unselected. The archive pulls since 2026-10-20 are UPLOADED replays (about a tenth of battles, selection
  UNVERIFIED; `designs/tools/replay_corpus.md`) and stay out until a bias check clears them.
- **Held-out evaluation dates: 2026-09-21, 2026-09-22, 2026-09-23** (the manifest's last three folders). RESERVED: no
  later training use of human labels may read them. Training-use dates, if any: 2026-05-18 → 2026-09-20.
- **Rating filter (primary): the LABELLED human's pre-battle rating ≥ 1500** (the `|player|` line). Why: it is the
  census's band (2.44M viewer decisions, 1.69M labels), the owner's own play is 1400–1600, and low-rated choices are
  noisier targets than the head should chase. All rated bands are reported as a secondary; unrated logs are excluded.
- **Both seats** of each log are read, each labelling the other.

## 5. The registered read (R1, CPU, before any training use)

**Question.** Does X26's intent head predict human opponents as well as it predicts the opponents it was trained
against? If not, human labels carry signal the self-play labels lack.

**Model.** `models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip`, read-only, CPU, one forward per row
(`main.policy_spectrum.reader.load_checkpoint`; the flat pointer's `last_flat_intent_logits` / `last_flat_intent`).

**Populations, every one through the SAME spectator pipeline:**

| | source | labelled side |
|---|---|---|
| **H** humans | the held-out dates, labelled rating ≥ 1500 | either seat |
| **S** own snapshots | X26's eval traces, cycles with step ≥ 360,000,000, opponents `sentinel_*` (pool snapshots, stochastic T = 1) | the sentinel |
| **B** bots | the same cycles, the eight archetype bots (not `random`) | the bot |
| **S-full / B-full** (control) | the same S / B battles, read from the TRAINEE's own stream with its real requests and packed team: the training row | as S / B |

**Rows.** Supervised rows only: a flat target that is not masked, Hidden Power excluded, the battle's last decision
dropped (no successor). Every eligible row of every population is read; no cap. Expected sizes (from the census and
the trace manifests): H ≈ 40k, S and B ≈ 20k each.

**Metrics** (per population): NLL of the flat softmax at the target (nats), top-1 accuracy, and two floors: uniform
over the live candidates, and PRESENCE-ONLY (softmax of the detached log-presence the head adds, i.e. the belief,
which starts from the Smogon prior). Strata: move vs switch label, target class (seat / OTHER_move / slot /
OTHER_species), the placeholder flag. 95 % CIs by a battle-cluster bootstrap (2,000 resamples, seed 50).

**Decision rule.** Δ = NLL(H) − NLL(S), both spectator.
- **GAP** (human labels carry signal): the 95 % CI of Δ is entirely above +0.05 nats. Next: a training-use design
  (human labels as an added intent-loss source on the training dates), registered separately; rung 1 stays labels.
- **NO GAP**: the CI lies entirely inside ±0.05 nats. The head already transfers; human labels add little to the
  head's own accuracy. Rung 1 closes at this read unless search's opponent model wants them.
- **INCONCLUSIVE** otherwise.
- Reported beside it, never a decision input: Δ against B; the pipeline penalty NLL(S) − NLL(S-full) and the same for
  B (how far the spectator rows sit from the training rows); top-1 for all; the all-bands human row.

**Known biases, declared before the read.** The traces are LOSS-ENRICHED by design (10 losses, 5 draws, 5 wins per
opponent per cycle); humans are unselected. The teams differ (our pool vs human teams), which is part of what "human
labels" means here. S's and B's boards come from X26's own play; H's from two humans.

## 6. The extractor (build)

`src/main/x50_r1/` (CPU, the Rust live reader only, no Node, no poke-env):
- `points.py` (decision points), `spectate.py` (a spectator view of a trace), `extract.py` (logs → rows: the reader
  session, the synthesised decisions, the production label from `LABEL`, the prefix guard);
- the Rust side: `live_reader`'s `SPECDECIDE` and `LABEL` commands, `SideReader::spectator_request`, and the
  `"spectator"` request branch of `BoardReading::parse_request_val` / `core_events::reading`;
- `read.py`: the registered R1 read (one CPU forward per row, `flat_intent_targets`, the battle-cluster bootstrap,
  `read.rule`), driven by the measurement's `run_r1.sh`;
- `run.py`: deterministic (sorted inputs, fixed seeds), resumable (a shard per date or trace cycle is skipped when
  present), chunked ≤ 45 min per heavy slot under `scripts/ops/mem_cap.sh` at rule 5b's priority;
- output under `/home/goodlad/gen3ai_archive/x50_r1_intent/` (never `data/`): one `npz` per shard (rows as float32,
  the label `(kind, num, slot, species)`, the coordinates `(battle, seat, decision, turn, kind)`, ratings, flags)
  and a `manifest.json` with the commit, the binary's hash and the counts.

## 7. Risks

- **The fabricated own spread and the public-only own side** move the input off the training distribution
  (F-X50-4). Mitigated for the read by running every population through the same pipeline and measuring the
  penalty on S / B; not fixed. A training use needs it fixed (a Smogon completion of the own side) or re-measured.
- **The synthesised legality** (revealed moves only, read PP, the Struggle placeholder) feeds the board block's
  requested-move cells: off distribution in the same way.
- **The production labeller on a spectator chain** is new ground: the decision-point check on traces (§3) and the
  S-full / S label agreement are its validation.

## Decision record

| date | what | rejected | evidence |
|---|---|---|---|
| 2026-10-20 | **CREATED + R1 REGISTERED** (X50 rung-1 worker, the owner's rung-1 approval). Label = the production label by the production labeller on synthesised spectator decisions, resolved by `flat_intent_targets`; own side public-only; spectator manifest only; held-out 09-21..09-23; primary ≥ 1500; R1's rule on Δ NLL(H − S) at ±0.05 nats | completing the own side from the game's end (a selection leak); the census's text labeller (an approximation of the production rules); the archive pulls (upload selection unverified); a typed guess for Hidden Power; a training run in this unit | the census README; `opp_intent_labels.py`; `trackers/mod.rs`; `flat_intent.py`; X26's eval-trace manifests |
| 2026-10-20 | **EXTRACTOR BUILT** | One decision guarded per seat (not a 1-in-16 battle sample: broader, cheaper). Two point-rule refinements found on traces before landing: the Pursuit-continuation switch is no decision; a log with no `\|t:\|` mark is refused. The trace merge aligns rather than zips (private `[silent]` / `\|error\|` / `\|-hint\|` lines). A battle whose trainee stream holds a refused choice (`\|error\|`, 7 % of traces) keeps its spectator read and skips only the full control (its retry request has no public twin) | zipping the two seat streams (fails on private lines); placing a forced point on every post-faint switch | `extract_test.py` (hand-read golden labels, the guard and its leak, spectator = full labels, full = the trainee's rows); `points_test.py` |
| 2026-10-20 | **R1 READ: GAP** | X26's head on humans ≥ 1500: NLL 1.937 against 1.748 on its own snapshots, through the same pipeline; Δ +0.188 [+0.164, +0.213]. The gap is in switches (+0.39 nats; moves +0.07). The pipeline costs 0.23 nats on our own battles (F-R1-1). Next: a training-use design, registered separately; dates 09-21..23 stay reserved | reading H − B as the decision input (bots switch on 6 % of labels) | `x50_r1_intent_read_2026-10-20/` |
| 2026-10-23 | **TRAINING-USE DESIGN REGISTERED** ([`design_x50_rung1b_human_opponent_model.md`](design_x50_rung1b_human_opponent_model.md)). §7's first risk is answered there by completing the own side from the end-of-game reveals plus Smogon draws (C-GAME), with its declared leak and true-row decisive reads; this doc's "never completed from the game's end" stays the rule for R1's read | a prefix-only completion (≈ 50 CPU-h, and an actor's first-use move becomes illegal) | that design §2 |
| 2026-10-25 | **Status line: the training-use design is READ (docs-health pass)** | The status called the rung-1b design "pre-registered, nothing read". Its R0, R-b1 and R-b2 are read (2026-10-23 / 24); R-a1 (this head refit offline) is not | — | `design_x50_rung1b_human_opponent_model.md` Status and Decision record; ledger `L28352`, `L28403`, `L28436` |
