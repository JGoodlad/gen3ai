# THE INFORMATION LEVERS: one spec for giving training, search and the model's inputs graded extra information

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-12): the SPEC, the LOG SOURCE, the SEARCH WORLD BUILDER (honest by default), the MODEL-INPUT
INJECTION, the GENDER PAD and the AUDIT SUITE are BUILT; the TRAINING OBSERVATION's spec flag and the diagnostic-fork
consent are BUILDING (§6).** Owner, 2026-10-12: "I want a single dried
out implementation for giving both search and training additional information that would not generally be available
so that we can understand what the ceilings are and what is our constraint and how much value there would be in
improving determinization." It serves the hidden-information study ([`design_hidden_info_binding.md`](design_hidden_info_binding.md))
and the value-discrimination pilot ([`design_value_discrimination.md`](design_value_discrimination.md)).

---

## 0. Summary, in plain language

A real player does not know the opponent's hidden facts: which Pokémon it has not shown yet, their moves, items,
abilities and EV spreads, a shown Pokémon's exact HP, how many turns of sleep are left. To find out WHICH of those
facts limits us, and how much a better guess would be worth, we hand some of them to our system on purpose and
measure the difference. Before this doc, three separate pieces of code did that three different ways: the training
observation (`--oracle-reveal off|species|full`), the search world builder (which kept some facts true by accident),
and a hook the hidden-information study wrote into its measurement folder. They could not be compared, and one of
them leaked.

Now there is ONE description of "what extra information, how much of it, and whether it is true", the
**information-lever spec**, and every consumer reads it:

1. **The training observation**: the network is told the facts in its input, during training and its eval.
2. **The search world builder**: the guessed opponent teams search plays out in hold the facts.
3. **The model-input injection**: a frozen network's own belief about the facts is pushed toward them.

Every fact is a **lever** with a **mass** (how confident: 0.95 = "high mass but not 100 %", the owner's dose) and a
**source** (the truth; a confident WRONG but plausible fact, the placebo; or the honest belief's own best guess, an
honest sharpening). By default every consumer is **honest**: it is given nothing a real player would not have.

**How to read ceilings with it** (§11): the oracle (every lever, mass 1) is the information ceiling; honest is today;
the gap for one class of fact (reveal only it, or hide only it) is what improving our guesses about that class could
buy at most; the dose (mass 0.25 … 1) says how much a partly better guess buys; the placebo says whether it is the
content or just the confidence.

---

## 1. THE SPEC (BUILT: `agents.info_levers.spec`)

### 1.1 The taxonomy

| class | the hidden fact | parts |
|---|---|---|
| `species` | the identity of a never-seen mon (a seen mon's species is public) | unseen |
| `moves` | the move slots play has not revealed | seen, unseen |
| `item` | the held item, until play names it | seen, unseen |
| `ability` | the ability, until play names it (two-ability species) | seen, unseen |
| `spread` | nature, EVs, IVs (→ exact stats, Hidden Power type) | seen, unseen |
| `hp` | the exact HP inside the displayed 1 % bucket | seen |
| `counters` | hidden durations: sleep / confusion / Encore / Disable / lock turns | seen |

`seen` = the opponent mons the viewer has seen switch or be dragged in; `unseen` = the rest. A never-seen mon is at
full HP with no counters, so `hp` / `counters` have no unseen part. An **unseen lever applies where the consumer's mon
at that slot IS a true opponent mon** (its species is right: told by a `species` lever, or a draw / hypothesis that
happens to be right): the obs's tail slot of a told species, the world's never-seen slot holding a true species, the
injection's X5 hypothesis slot whose species is a true never-seen one.

**The Hidden Power belongs to TWO classes, split (the blind replica's note, 2026-10-12).** In gen 3 its type is set
by the IVs, so `moves` and `spread` overlap there. The split every consumer follows: **`moves` tells THAT a set holds
a Hidden Power** (one untyped slot); **`spread` tells its TYPE** (the IVs that carry it). The search builder screens
the type with the spread (§7.2); the injection pushes the presence with the moves lever and the type with the spread
lever (§8); the observation writes a Hidden Power told without the spread BARE (§6). `full` tells both, so its bytes
are X32's.

### 1.2 A lever: (class, part, mass, source)

- **mass p ∈ (0, 1]**, a whole multiple of 1/10,000 (exact text, the same integer in Rust). The consumer's belief about
  the fact becomes **p·δ(target) + (1 − p)·(its honest belief)**. p = 1 is the oracle for that fact; a mass-0 lever is
  no lever (the honest belief) and is dropped from the canonical text.
- **source**: `truth` (the target is the true fact); `placebo` (an equally confident WRONG but plausible fact: drawn
  ∝ the honest prior over the candidates EXCLUDING the truth and everything play revealed; deterministic per
  decision / episode); `map` (the honest belief's own mode: an honest sharpening — no truth is read).

### 1.3 Text, hash, record

A spec is a **preset name** (§5) or a comma list of clauses `class[:scope][@mass][~source]` (defaults: the class's
widest scope, mass 1, `truth`). The **canonical text** — what the Rust core parses and the hash is taken over — puts
clauses in class order with the scope explicit (`both` when a class's seen and unseen parts are identical), a mass
below 1 as `@` + up to four decimals, a non-truth source as `~source`; the empty spec is `off`. The **label** is the
X32 level name (`off` / `species` / `full`) when the spec IS that level — so `model_config.json`'s `oracle_reveal`
records exactly what it recorded before — else the canonical text. **Hash** = the first 16 hex of
`sha256("info_reveal_v1:" + canonical)` (pinned for `off` / `species` / `full` / `F+M` by `spec_test.py`).
`InfoReveal.record()` is the block a run or a measurement records (label, canonical, hash, levers).

Examples: `full` = `species:unseen,moves:both,item:both,ability:both,spread:both`; the move dose at 0.9 =
`moves:seen@0.9`; the wrong-moves placebo = `moves:seen@0.9~placebo`; everything but the moves of seen mons =
`O-M` = `species:unseen,moves:unseen,item:both,ability:both,spread:both,hp:seen,counters:seen`.

---

## 2. THE ONE LOG SOURCE (BUILT: `agents.info_levers.revealed`)

What play has revealed is derived from the viewer's own PROTOCOL stream (the battle log it was sent), never from its
observation rows (the v1 world builder's row gate missed gender tokens, Focus Punch's tightening line and a Beat Up
hit: `design_hidden_info_binding.md` §16.4). Per opponent mon seen: its OWN moves named by play (a `|move|` line not
CALLED from another set — `[from] Assist` / `Snatch` / `Magic Coat` / `Metronome` / `Mirror Move` are not its own;
`Sleep Talk` / `Pursuit` / a locked move are — Focus Punch's `-singleturn`, the move a `cant` line says it chose;
never one used while transformed or copied by Mimic; never Struggle); its ORIGINAL item (`[from] item:` on its own
line, `-enditem`, `-activate|…|item:`, Recycle, or the item our mon received from it by Thief / Trick — before any
transfer); its ORIGINAL ability (an `-ability` announce, Trace on either side, `[from] ability:` where the line's form
makes the owner certain: Showdown's `heal` writes `[of]` = the attacker for Volt / Water Absorb, `damage` /
`trySetStatus` / weather write `[of]` = the holder).

**Conservative by design.** A fact the parser misses is only drawn by the search builder and then forced back to the
truth by the protocol gate (inefficient, never wrong); a fact it CLAIMED that play did not reveal would be kept true
(a leak). So an ambiguous owner reveals nothing and is counted (`Reveals.ambiguous`).

**One definition, two readers.** Python consumers (the builder, the injection, the audits) read this module; the
Rust encoder's reading parses the same stream for the observation. The LOG-CONSISTENCY audit (§10) checks they agree
on every decision of real battles.

---

## 3. THE CONSUMERS AND WHAT EACH HONOURS (`agents.info_levers.consumers`)

A lever a consumer cannot honour is **refused by name** before anything is built (`Consumer.validate`); a fact a
consumer can only ever give TRUE is its named **residual**, reported beside every honest spec it runs.

| fact | (a) training observation | (b) search world builder | (c) model-input injection |
|---|---|---|---|
| `species:unseen` | truth (a tail slot per told mon) | truth / placebo / map | refused (the X5 hypotheses are the network's own guesses) |
| `moves:seen` | truth | truth / placebo / map | truth / placebo / map (the typed move posterior) |
| `moves:unseen` | truth (needs `species:unseen`) | truth / placebo / map | truth (a hypothesis slot whose species is a true never-seen one) |
| `item:seen` | truth | truth / placebo / map | truth / placebo / map (the item posterior: the op's P(Choice Band)) |
| `item:unseen` | truth (needs `species:unseen`) | truth / placebo / map | refused |
| `ability:seen` / `:unseen` | truth (unseen needs `species:unseen`) | truth / placebo / map | refused (no ability belief input) |
| `spread:seen` | truth | truth / placebo / map | truth (the believed stats) |
| `spread:unseen` | truth (needs `species:unseen`) | truth / placebo / map | refused |
| `hp:seen` | refused (no exact-HP cell; a layout change) | truth (the world must reproduce the true exact HP) | refused |
| `counters:seen` | refused (no hidden-duration cell) | truth — **RESIDUAL**: always the truth's (§7.3) | refused |
| mass p | a per-(episode, side, mon, class) Bernoulli: the observation can only state a fact or not | a per-world draw from the mixture | the mixed marginals, written |

---

## 4. Status of each consumer

- **(a) The training observation** (§6): BUILDING. Today's `--oracle-reveal off|species|full` keeps its exact
  behaviour and bytes; the flag will accept any spec the OBS consumer honours.
- **(b) The search world builder** (§7): BUILT — honest by default; the pilot's builder is the `pilot` preset.
- **(c) The model-input injection** (§8): BUILT (`agents.info_levers.inject`), lifted from the study's `inject.py` into `src/` and driven by the spec.
- **The gender pad** (§9): BUILT (the engine marker `98146ddb`, wired into the builder).
- **The audit suite** (§10): BUILT (`main.info_audit`); its read is §10.3.

---

## 5. PRESETS, and how each study arm maps to one (`spec.PRESETS`)

| preset | canonical text | the arm it is |
|---|---|---|
| `off` = `F` | `off` | honest: production; the study's FULLY honest base F |
| `species` | `species:unseen` | X32 `--oracle-reveal species` (byte-identical observation); the study's F+Usp |
| `full` | `species:unseen,moves:both,item:both,ability:both,spread:both` | X32 `--oracle-reveal full` (byte-identical observation) |
| `oracle` = `O` | `full` + `hp:seen,counters:seen` | everything: the information ceiling (search: the TRUE battle) |
| `pilot` | `item:seen,ability:seen,spread:seen,hp:seen,counters:seen` | the value-discrimination pilot's world builder (v1 / v2): P |
| `F+M` / `F+U` / `F+I` / `F+A` / `F+S` | F plus the study class | §6.2 SUFFICIENCY ("reveal only X") |
| `O-M` / `O-U` / `O-I` / `O-A` / `O-S` | O minus the study class | §6.2 NECESSITY ("reveal all but X") |
| `D25` `D50` `D75` `D90` | `moves:seen@p` | §6.4 the move DOSE |
| `W50` `W90` `W100` | `moves:seen@p~placebo` | §6.4 the wrong-moves PLACEBO |
| `MAP90` `MAP100` | `moves:seen@p~map` | §6.4 the honest SHARPENING |
| `TRUE@p` / `PLACEBO@p` / `SELFMAP@p` / `TRUEHYP@p` | `moves:seen@p`, `…~placebo`, `…~map`, `moves:both@p` | §7.2 the MODEL-INPUT arms (injection) |
| `ki_moves` `ki_items` `ki_spreads` `ki_wrongmoves` | `moves:seen@0.95`, `item:seen@0.95`, `spread:seen@0.95`, `moves:seen@0.95~placebo` | §8.3 the knock-in levers at the study's mass 0.95 |

The study's five classes (§6.1 there) are `M` = `moves:seen`; `U` = `species:unseen` + every set class's unseen part;
`I` = `item:seen`; `A` = `ability:seen`; `S` = `spread:seen` + `hp:seen` (a seen mon's spread determines the exact HP
its replayed history leaves). `counters` belongs to no study class: only O tells it. The POOL / POOLSETS arms (a
different PRIOR, not extra information) and the hand evaluator (a different leaf) are not levers.

---

## 6. (a) THE TRAINING OBSERVATION and the DIAGNOSTIC-FORK consent (BUILDING)

**The flag.** `--oracle-reveal` accepts any spec the OBS consumer honours; `off` / `species` / `full` keep their exact
behaviour. Symmetric as today (each side's chain is told the other side's team at the run's spec; scripted bots read
the view, not the row); the in-loop eval core plays at the run's RECORDED spec; `main.h2h`'s per-side reveal takes
the recorded label. Recorded in `model_config.json` (`oracle_reveal` = the label) and `metadata.json` (the
`InfoReveal.record()` block).

**Mass in the observation.** The observation can only state a fact or not: a lever of mass p tells its fact for a
whole episode with probability p, decided per (episode seed, side, mon, class) by a hash, so the episode-averaged
belief is p·truth + (1 − p)·the network's own. A never-seen mon's tail slot exists only where its species is told.

**The fork rule (to build).** `--oracle-reveal` is RESUME-IMMUTABLE (a same-run resume that flips it is refused). The
knock-in forks need a reveal on a FORK of a non-reveal checkpoint, so a fork may change it only with an explicit,
recorded DIAGNOSTIC-FORK consent: never a silent flip, refused on a same-run resume and for a production lineage, and
stamped in the child's `metadata.json` lineage. The fork arm (`--fork-fraction > 0`, the collector's counterfactual
successor rows, encoded WITHOUT the reveal) stays refused with any non-`off` spec.

---

## 7. (b) THE SEARCH WORLD BUILDER, honest by default (BUILT: `main.value_search.worlds`, `world_draws`, `world_screen`)

`WorldSampler(reveal=spec)` (default `off`: HONEST); `sample(log, at, side, J, key, screen_cache=…)`. `SearchConfig(oracle=
True)` stays the cheap way to play the TRUE battle; the `oracle` preset through the builder reproduces it exactly (the
audit's `oracle_identity`).

### 7.1 The leaks it fixed (before = the pilot's builder, the `pilot` preset; after = `off`; measured, §10.3)

The pilot's builder (v1 / v2) kept every SEEN opponent mon's item, ability and spread TRUE — hence its exact stats, its
Hidden Power type and the exact HP its history leaves — because they shape the observed history and a joint redraw
almost never passes the gate. Honest by default, each of them is DRAWN from the Smogon prior and conditioned on the log
by the same byte-equal protocol + row gate; what play revealed (`agents.info_levers.revealed`) stays the truth's.

Truth-match rate of each fact in the worlds built for 8 committed-bank decisions × 4 worlds (`python -m main.info_audit`; a revealed fact is the truth's under both and is not counted here):

| fact | the pilot's builder (`pilot`) | the honest builder (`off`) |
|---|---|---|
| a seen mon's unrevealed item | 1.000 (told, n 48) | 0.750 (drawn, n 48) |
| its unrevealed ability | 1.000 (told, n 96) | 1.000 (drawn, n 96) |
| its spread (nature + EVs + IVs, exactly) | 1.000 (told, n 124) | 0.032 (drawn, n 124) |
| its unrevealed move slots (all of them right) | 0.113 (drawn, n 124) | 0.097 (drawn, n 124) |
| a never-seen slot's species | 0.000 (drawn, n 68) | 0.000 (drawn, n 68) |
| a never-seen slot's item (when its species is right) | 0.044 (drawn, n 68) | 0.044 (drawn, n 68) |
| a seen mon's EXACT HP at the root equals the truth's | 124 / 124 | 70 / 124 |

Under `pilot` the seen mons' item, ability and spread are TOLD (the truth, every world); under `off` they are drawn and match the truth only as often as the log forces it (Leftovers on a mon whose healing showed; the one ability most species have) or the prior happens to.

### 7.2 What each fact of a world is

- **Revealed by play** (the ONE log source): the truth's. A reveal the source names that the true set does not hold is
  the parser's error: counted (`leak["reveal_mismatch"]`) and drawn instead.
- **A lever fires** (a categorical fact: with probability = its mass, decided once per world and mon; the MOVES lever is
  no coin — its mass is the presence mixture of every candidate move, §1.2): its target — the truth, a placebo
  (drawn ∝ the prior excluding the truth; for a seen mon's item / ability / spread, one the log allows), or the map
  (the prior's mode; for a seen mon's item / ability / spread, the most probable state the screen found the log to
  allow).
- **Otherwise drawn**, honestly: a NEVER-SEEN slot's species ∝ the Smogon teammate co-occurrence with the revealed
  species, then its whole set (moves by the conditional-Poisson presence draw, item, ability, spread); a SEEN mon's
  unrevealed moves by the same presence draw (`mixture.mix_presence` + `cp_sample`: p = 0 is the honest draw exactly,
  the study's §6.4 dose); a seen mon's item, ability and spread by the SCREEN (§7.3).
- **Hidden Power**: its TYPE and whether a set carries one are SPREAD facts (the IVs: they move the exact stats the log
  pins by a point). The screen draws them; the world's moves then hold a Hidden Power of exactly that type (or none);
  a revealed bare Hidden Power's type is drawn the same way, and a past choice of it is rendered as the world's typed
  move. A seen mon's moves that match the truth count modulo the Hidden Power's type. The state is (type, IV
  PATTERN): any of the type's 30 / 31 patterns (`packed.hp_iv_masks`; the canonical one, the most 31s, at `HP_CANON`
  0.8, the others sharing the rest), because pool sets use non-canonical ones (the blind replica's note) and a
  pattern moves the exact stats by a point, which the log pins. IVs below 30 are outside the support (§7.5).
- **Genders** (§9): the never-seen mons are placed so the turn-0 gender draws fall on exactly the true team's draw
  slots; with the engine's pad any of them may sit on a draw slot.
- **Legality**: a world breaking a gen3ou rule the true team does not break is rejected (`clause`); the never-seen
  draws and the genders are checked BEFORE the screen (cheap), each seen mon's set after its moves are drawn (a banned
  set — e.g. a Baton Passer drawn a Salac Berry — is redrawn up to `MOVE_REDRAWS` times), the whole team last.
- **The playout tool opens it** (the blind replica's finding 3): a world that passed the gate is also opened by
  `rust_env_playout_open`, the search's own replay, before it is returned. The gate's `core_events` replay accepts
  some choice logs the playout tool refuses (a numeric `move k`; a Struggle the world does not force); such a world is
  rejected (`fails["playout_refused"]`, the refusal named), never handed to a search that cannot play it. The audit's
  `playout_open` re-checks every returned world; the planted numeric choice is gated green by `core_events` and
  refused here (`info_audit_integration_test.py`).

### 7.3 The SCREEN: honest posterior draws of a seen mon's item, ability and spread (`world_screen`)

- **The honest prior.** Item and ability ∝ the species' Smogon usage. The spread (nature, EVs) ∝ (1 − ε)·J + ε·N·Π M:
  J the species' chaos spread table (every spread, not the top-25 cut, which covers only ~49 % of the pool's true
  spreads: the blind replica's measure), N its nature marginal, M each stat's EV
  marginal floored by a uniform over 0, 4, …, 252 (`EPS` 0.25, `FLOOR` 0.15), an EV total over 510 impossible. J alone
  is a coarse histogram: a pool spread Smogon never lists (a 16-HP-EV / 252-Def Vaporeon) has no consistent J
  candidate, and an honest player then believes the EVs that fit; the smooth term gives every legal spread mass. The
  Hidden Power: P(the set holds one) and P(type | one) from the moves prior.
- **One sampler per (seen mon, constraint)** — the constraint being the facts a lever told, the Hidden Power rule, an
  exact-HP requirement — serves every world of the decision, and, through `screen_cache`, every spec built at it: arms
  that differ only in other levers (the move dose, the placebo, the map) take the SAME seen-mon draws (common random
  numbers for their paired contrasts) and pay the screen once.
- **The screening world** holds every OTHER opponent mon at its true facts (with each observable event depending on one
  opponent mon's hidden facts — singles damage, speed order, item and ability triggers — the draw is then from P(facts
  of mon k | everything observed) and the true background leaks nothing); the combined world must pass the STRICT
  gate again (the `joint` count is the factorisation's measured residual).
- **The sampler**: (1) REJECTION — independent prior draws, each replayed; an accepted one is an exact posterior draw.
  (2) When the prior rarely hits the log, a CHAIN: a search phase climbs `divergence_score` (the number of leading
  protocol lines that match the truth's, plus a closeness of the first differing HP number) with moves DIRECTED by the
  first difference (`divergence_hint`: our mon's HP after their hit → their power, Atk / SpA, a Choice Band; their HP
  after ours → their bulk, HP / Def / SpD; a two-sided order → their Speed, signed by who acted first), restarts after
  `STALL` flat steps (from the best state shaken, or a fresh prior draw); once the log fits, Metropolis-Hastings on the
  prior × the gate (symmetric local moves at the prior ratio, independence draws at ratio 1), burnt in and thinned. No
  step reads the truth. Screening replays are LENIENT (`CHOOSEIF`: a recorded choice the diverged world no longer asks
  for is skipped, so a wrong candidate still yields the protocol its score reads); the world returned is STRICT.
  A sampler that never fits the log in `MAX_STEPS` makes the decision unbuildable (a `WorldError`: an EXCLUSION).
- **Legality inside the screen**: when a seen mon's moves are TOLD (a mass-1 truth lever), a drawn item or ability
  that breaks a clause with them is refused without a replay; never for drawn moves (it would filter by the truth's
  unrevealed moves).

### 7.4 Cost and exclusions (measured, §10.3)

On 8 committed-bank decisions × 4 worlds, the box under a live training run (`nice -n 19`, three agent cores), one screen shared by every spec of a decision:

| spec | built | excluded | screening replays | seconds |
|---|---|---|---|---|
| `D50` | 8 | 0 | 32 | 4 |
| `F+M` | 8 | 0 | 1754 | 85 |
| `W90` | 8 | 0 | 134 | 8 |
| `off` | 8 | 0 | 3537 | 172 |
| `oracle` | 8 | 0 | 8 | 5 |
| `pilot` | 8 | 0 | 63 | 7 |

No decision was excluded.

A wider read under `off` alone (40 decisions × 2 worlds, turns 3–40, after the playout-tool check and the Hidden
Power IV patterns landed; `audit_off_40.json`): **39 built, 1 EXCLUDED** — a Tyranitar on a pool spread (184 HP / 104
Atk / 220 Spe, Adamant; its IVs inside the support) the chain did not fit in `MAX_STEPS` — every check PASS
(`playout_open`: 0 of 78 worlds refused; `gender_independence`: 0 gender-count rejections), 1,298 s contended.

The honest screen is the cost: a seen mon the prior fits is a few replays; one whose log pins a spread far from Smogon's table (a pool spread, a Choice Band at 0.5 % prior) takes the chain tens of steps × `SEARCH_WIDTH` replays. Specs that share the screen (`D50`, `W90`, …) cost almost nothing more.

### 7.5 The residuals (named)

- **Hidden counters** (`counters`, the builder's declared residual): the world replays the prefix on the TRUE battle
  seed, so a duration rolled BEFORE the root (sleep, confusion, Encore, a lock) is the truth's in every world (the
  study's A5 caveat). Hiding it needs a root-state resample in the playout core (not built).
- **The factorisation** of §7.3: measured by the `joint` rejections, never assumed.
- **Party-wide effects** (Beat Up's hit sizes read the party's base Attack, never-seen mons included): the log really
  does constrain the never-seen species there; the joint gate conditions on it by rejection, correct but slow.
- **A never-seen ratio'd mon on a true draw slot takes the truth's gender coin** (the draw's value, not its count):
  unobservable before the root and not a fact of the taxonomy.
- **The honest spread prior is a choice** (`EPS`, `FLOOR`): Smogon-derived (its joint table and marginals) with a
  uniform floor; it shapes the posterior only where the log leaves the spread loose.
- **Dice-silent facts under SEED REUSE** (the blind replica's finding 2; `SEARCH.partial`, reported in every world's
  `leak["partial"]`). The prefix is replayed on the TRUE seed, so a world must spend the truth's dice; a hidden fact
  that spends a die WITHOUT printing a line is then visible to the gate though no honest player sees it. Verified in
  `deps/pokemon-showdown`: King's Rock (a 10 % flinch roll added to most damaging moves, per hit), Focus Band (a
  1-in-10 roll on EVERY move damage its holder takes, before the lethal check), the contact abilities Static, Flame
  Body, Poison Point, Effect Spore, Cute Charm (a roll per contact hit taken), Shed Skin (a roll per statused turn),
  and a SPEED TIE (speedSort shuffles a tied group: a die only when a tie exists). Not one: Quick Claw (gen 3 rolls
  it battle-wide every turn). **Measured** (`main.info_audit`'s `seed_reuse`: COUNTERFACTUAL FLIPS of the true team,
  one hidden fact of one seen mon changed and gated exactly as a world is, against a passing unflipped control; 169
  committed-bank decisions, turns 3–40, 1,662 replays; `research_state/measurements/info_levers_2026-10-12/
  seed_reuse_200.json`):

  | fact | flips | the gate rejects | what that removes |
  |---|---|---|---|
  | a seen mon's Speed ±1 point (Spe EVs only: at ±1 only a tie changes anything) | 674 (decision, seen mon) | 34 (5.0 %), in 34 of 169 decisions (20.1 %): 6 both neighbours (the truth at a tie: its exact Speed pinned), 28 one (a neighbour would tie) | the honest posterior's mass on those Speed values |
  | a hidden item → King's Rock / Focus Band (truth not one; a neutral-item swap accepted) | 318 (159 mons) | 125 (39 %) | the prior's mass on them: 0.0075 summed, 4.7·10⁻⁵ per mon |
  | a hidden item that IS King's Rock / Focus Band → a neutral item | 0 | — | none of the bank's 1,386 pool mons (231 teams) holds one |
  | a hidden ability ↔ its dice-rolling alternative | 0 | — | no seen mon in the corpus had one unrevealed |

  So the item and ability legs are negligible on this pool; the SPEED-TIE leg is the real one: a fifth of decisions
  have a seen mon whose Speed the gate pins a point tighter than honest play could. Removing it needs a playout core
  that replays the prefix on fresh dice constrained to the printed outcomes (not built).
- **IVs outside the screen's support** (measured on the bank's 231 teams, 1,386 mons): of 609 Hidden Power sets, 146
  use the canonical 30 / 31 pattern, 253 another 30 / 31 pattern (both in the support), **210 (34 %) IVs below 30**
  (mostly a minimised Attack, e.g. 31 / 2 / 31 / 30 / 31 / 31 for Hidden Power Fire); of 777 other sets, 258 (33 %)
  are not all 31 (mostly 0 Attack IVs). The screen draws neither: where the log pins such a stat (a Rapid Spin, a
  confusion hit) no world fits and the decision is EXCLUDED (counted), never misbuilt; where it does not, the world
  holds a different but log-consistent value. An IV prior beyond 30 / 31 has no Smogon source (a judgment for the
  owner, `design_hand_computed_features.md`'s fact-vs-judgment test).

---

## 8. (c) THE MODEL-INPUT INJECTION (BUILT: `agents.info_levers.inject`)

The study's hook (`measurements/hidden_info_binding_2026-10-11/inject.py`, its §7.1) lifted into `src/` at HEAD and
driven by the spec (`Injector(extractor)`; `inj.set(spec, decision_id, TrueTeam.of(opponent_packed_team))`; every
forward inside `inj.installed()` is injected). For a FROZEN network, offline (the study's model-input axis, probes);
nothing in training reads it.

- **Moves** (`moves:seen` truth / placebo / map, `moves:unseen` truth): it wraps the extractor INSTANCE's
  `_typed_hp_posterior` (the narrowest point every consumer of the move belief reads: the published
  `last_move_belief_logits`, the active's move group, the reinjection, the damage op, the X5 roster, the E4 seats,
  the edges and cells) and rewrites each targeted opponent slot's typed posterior q' = (1 − p)·σ(ℓ) + p·1[m ∈
  target]; the revealed moves, the bookkeeping typeless Hidden Power and (once revealed) all 16 typed Hidden Power
  channels keep their bytes; the clip never moves a channel against the injection; the Hidden Power presence and type
  posterior stay consistent. A TRUE Hidden Power is no typed target (§1.1): the moves lever pushes its PRESENCE,
  presence' = (1 − p)·presence + p, and leaves the network's type posterior; a `spread:seen` lever pushes its TYPE,
  type' = (1 − p_s)·type + p_s·δ(true type); the 16 typed channels become presence'·type'. The placebo is 4 − r moves drawn ∝ the network's own belief excluding the truth and the
  pinned channels (deterministic per decision and slot); map is its own top 4 − r. An unseen lever applies on an X5
  hypothesis slot whose hypothesised species is a true never-seen one (the study's TRUE-HYP).
- **Item** (`item:seen`): a forward hook on `item_belief_head`: on a seen slot whose item the observation does not
  state, the posterior becomes (1 − p)·softmax + p·one-hot(target) (its consumers: the op's P(Choice Band) /
  P(Quick Claw)).
- **Spread** (`spread:seen`, truth): a forward hook on `spread_belief`: the believed stats become (1 − p)·believed +
  p·the true L100 stats (`priors.gen3_stat`, exact), and the head's token reinjection is recomputed from them. **Not
  covered (named):** the nature posterior the op marginalises under `--spread-belief-nature-marginalize`.
- **Teeth** (`inject_integration_test.py`, a production-config extractor on a real committed-bank battle): `off` /
  the empty spec bit-identical to the unhooked forward; every targeted true move ≥ its mass; pinned channels
  byte-equal; a placebo never a true move and deterministic per decision; map = the network's own top candidates; the
  item and spread levers reach their consumers exactly; refusals by name; a planted off-by-one target is caught.
- **Training-time injection** (the hidden-info study's §8 `--belief-oracle`, the label keys in the rollout) is NOT
  built. The knock-in levers are offered on the OBSERVATION channel instead (§6: the `ki_*` presets with the
  diagnostic-fork consent); which channel the study's forks take is the study's call.

---

## 9. THE GENDER PAD (BUILT: the engine `98146ddb`, the builder's wiring)

The engine draws one gender at team construction for every mon of a gender-ratio species whose packed set names none,
so the pre-root dice depend on which never-seen slots draw. The v2 builder placed ratio'd never-seen mons on the true
draw slots, so a world needed at least as many as the truth drew (m ≥ r) and the gate conditioned the never-seen
species on that count (accepted P(m = r) 0.67 vs the honest 0.57; `design_value_discrimination.md` Amendment 8). The
engine's PAD — a packed gender `M*` / `F*` / `N*` keeps the named gender and still spends the one turn-0 draw
(`utils.team_packing.GENDER_DRAW_PAD`; `designs/rust_sim/module_map.md`, the `team.rs` row; the marker-free path is
Showdown's exactly, its Rust tests and differential / fuzz rungs green) — lets ANY never-seen mon sit on a true draw
slot. HEAD's `place_genders` uses it: no world is rejected on its gender count any more, so that conditioning is gone
by construction (the audit's `genders` check holds every world's draw slots to the truth's; `gender_independence`
fails any world rejected on its gender count, and its PLANT — the builder with the pad switched off, at a decision
where the pilot's builder needed it — must record such rejections: the blind replica's finding 1, seed reuse through
the turn-0 gender dice, is closed by the pad and caught without it). A padded team is a SEARCH
world only: it never reaches Node's validator or a live server (Showdown would read `M*` as unnamed).

---

## 10. THE AUDIT SUITE (BUILT: `main.info_audit`)

ONE suite over all three consumers; every check passes or fails DETERMINISTICALLY (fixed committed-bank battles,
fixed seeds; a statistical bound only where its planted defect sits far beyond it).

### 10.1 The checks

| consumer | check | what it catches | where |
|---|---|---|---|
| search | `template_identity` | the choice template rendered against the TRUE team reproduces the true prefix (a rewritten slot one off is invisible to a replay when the choice never executed) | `checks.py` |
| search | `oracle_identity` | the `oracle` preset builds exactly the true team and prefix | `checks.py` |
| search | `regate` | every returned world replays STRICTLY to the searched side's byte-equal rows and protocol (+ the seen mons' exact-HP match, descriptive) | `checks.py` |
| search | `legality` | no world breaks a gen3ou rule the true team does not | `checks.py` |
| search | `genders` | the never-seen slots that draw (or pad) are exactly the true team's draw slots | `checks.py` |
| search | `gender_independence` | no world was rejected on its turn-0 gender-draw count (seed reuse through the gender dice; the pad closes it) | `checks.py` |
| search | `playout_open` | every returned world opens in the playout tool (`rust_env_playout_open`), not only in the gate's `core_events` replay | `checks.py` |
| search | `seed_reuse` | DESCRIPTIVE: the dice-silent residual's size by counterfactual flips (§7.5); fails only if its unflipped control does | `checks.py` |
| search | `provenance` | every fact is revealed / told / drawn exactly as the spec says (told without a lever, a mass-1 truth lever not the truth, a revealed fact not the truth, a placebo equal to the truth) | `checks.py` |
| search | `leak:moves:seen:drawn` | a seen mon's DRAWN moves match the truth no more often than their exact conditional-Poisson inclusion predicts (z ≤ 4) | `checks.Leakage` |
| search | `dose:moves:seen:told` | a TOLD move lever hits the truth exactly as often as its mixture says (\|z\| ≤ 4): the dose is the registered one, neither diluted nor sharpened | `checks.Leakage` |
| search | `leak:{item,ability}:seen:proposals` | the screen's independent proposals match the truth no more often than the prior's own mass on it (z ≤ 4) | `checks.Leakage` |
| log source | `log_consistency` | the Python source of what play revealed against the Rust reading the observation encodes, every decision: a fact the source claims that the reading does not show (the builder's leak direction) fails | `checks.py` |
| injection | reach / pins / placebo / map / item / spread / `off` bit-identity | §8's teeth | `agents/info_levers/inject_integration_test.py` |
| observation | `off` / `species` / `full` byte-identity, the declared-cell guard, label alignment, the mass Bernoulli | §6 | `rust_env/tests/oracle_reveal_test.rs`, `utils/rust_env/oracle_reveal_integration_test.py` |

**Planted defects (each must flip its check):** a template slot rendered one off (`template_identity`); a seen mon's
item swapped in a returned world (`regate`); a never-seen mon's gender flipped (`genders`); a world that copies a seen
mon's true moves but reports a draw (`leak:moves`, z > 6 against an honest z < 2.5); a screen whose item proposals peek
(`leak:item` proposals; the synthetic unit test, since a prior that already puts most mass on the true item gives the
real-battle version little power); a source that claims a move play never named (`log_consistency`); an injection one
move channel off (the reach check); provenance violations of each kind (unit); the builder without the gender pad
(`gender_independence`); a numeric choice token `core_events` gates green (`playout_open`'s refusal); a control gate
that rejects everything (`seed_reuse`); an injection's Hidden Power type moved by the moves lever (§1.1; the
injection's `the_hidden_power_type_is_a_spread_fact` test). Plus the hidden-info study's six A1–A3
plants and its gender plants, which these generalise.

### 10.2 The tiers

- **Routine gate:** the unit checks (`checks_test.py`, `spec_test.py`, `revealed_test.py`, `mixture_test.py`,
  `worlds_test.py`) and two early-turn decisions under `oracle` / `pilot` / `off` / `F+M` with the teeth
  (`info_audit_integration_test.py`); the injection's teeth.
- **Slow tier:** twelve decisions under ten specs (`info_audit_slow_test.py`).
- **By hand:** `python -m main.info_audit --n N --worlds J --specs … --out report.json` (one JSON report: every
  verdict, the per-spec cost and exclusions, the per-class truth-match tables).

### 10.3 The read (2026-10-12)

8 committed-bank decisions × 4 worlds, specs `D50`, `F+M`, `W90`, `off`, `oracle`, `pilot`: **every check PASSES**.

| check | verdict | counts |
|---|---|---|
| `dose:moves:seen:told[D50]` | PASS | expected 299.16, slots 396, true_hits 290, z -1.12 |
| `dose:moves:seen:told[F+M]` | PASS | expected 396.0, slots 396, true_hits 396, z 0.0 |
| `dose:moves:seen:told[W90]` | PASS | expected 25.97, slots 396, true_hits 24, z -0.46 |
| `dose:moves:seen:told[oracle]` | PASS | expected 396.0, slots 396, true_hits 396, z 0.0 |
| `genders[D50]` | PASS | mismatches 0, worlds 32 |
| `genders[F+M]` | PASS | mismatches 0, worlds 32 |
| `genders[W90]` | PASS | mismatches 0, worlds 32 |
| `genders[off]` | PASS | mismatches 0, worlds 32 |
| `genders[oracle]` | PASS | mismatches 0, worlds 32 |
| `genders[pilot]` | PASS | mismatches 0, worlds 32 |
| `leak:ability:seen:proposals[D50]` | PASS | expected 162.99, proposals 172, true_hits 164, z 0.34 |
| `leak:ability:seen:proposals[F+M]` | PASS | expected 159.42, proposals 168, true_hits 159, z -0.15 |
| `leak:ability:seen:proposals[W90]` | PASS | expected 210.66, proposals 221, true_hits 210, z -0.21 |
| `leak:ability:seen:proposals[off]` | PASS | expected 162.99, proposals 172, true_hits 164, z 0.34 |
| `leak:item:seen:proposals[D50]` | PASS | expected 58.5, proposals 82, true_hits 58, z -0.12 |
| `leak:item:seen:proposals[F+M]` | PASS | expected 60.42, proposals 84, true_hits 59, z -0.35 |
| `leak:item:seen:proposals[W90]` | PASS | expected 66.04, proposals 99, true_hits 66, z -0.01 |
| `leak:item:seen:proposals[off]` | PASS | expected 58.5, proposals 82, true_hits 58, z -0.12 |
| `leak:moves:seen:drawn[off]` | PASS | expected 203.66, slots 396, true_hits 202, z -0.18 |
| `leak:moves:seen:drawn[pilot]` | PASS | expected 223.04, slots 396, true_hits 218, z -0.55 |
| `legality[D50]` | PASS | violations 0, worlds 32 |
| `legality[F+M]` | PASS | violations 0, worlds 32 |
| `legality[W90]` | PASS | violations 0, worlds 32 |
| `legality[off]` | PASS | violations 0, worlds 32 |
| `legality[oracle]` | PASS | violations 0, worlds 32 |
| `legality[pilot]` | PASS | violations 0, worlds 32 |
| `log_consistency` | PASS | decisions 98, items_agree 139, items_obs_only 0, items_python_only 0, mons 246, moves_agree 217, moves_obs_only 0, moves_python_only 0 |
| `oracle_identity` | PASS | mismatches 0, worlds 32 |
| `provenance[D50]` | PASS | facts 768, violations 0 |
| `provenance[F+M]` | PASS | facts 768, violations 0 |
| `provenance[W90]` | PASS | facts 768, violations 0 |
| `provenance[off]` | PASS | facts 768, violations 0 |
| `provenance[oracle]` | PASS | facts 768, violations 0 |
| `provenance[pilot]` | PASS | facts 768, violations 0 |
| `regate[D50]` | PASS | failures 0, seen_exact_hp_equal 70, seen_mons 124, worlds 32 |
| `regate[F+M]` | PASS | failures 0, seen_exact_hp_equal 69, seen_mons 124, worlds 32 |
| `regate[W90]` | PASS | failures 0, seen_exact_hp_equal 70, seen_mons 124, worlds 32 |
| `regate[off]` | PASS | failures 0, seen_exact_hp_equal 70, seen_mons 124, worlds 32 |
| `regate[oracle]` | PASS | failures 0, seen_exact_hp_equal 124, seen_mons 124, worlds 32 |
| `regate[pilot]` | PASS | failures 0, seen_exact_hp_equal 124, seen_mons 124, worlds 32 |
| `template_identity` | PASS | commands 184, mismatches 0 |

---

## 11. HOW TO READ CEILINGS WITH IT

For a decision-level score IMP (the study's: win probability per decision over the policy's pick, LEAF-K8):

- **The information ceiling** = IMP under `oracle`. **Today** = IMP under `off`. Their difference G is everything
  hidden information costs this searcher at this budget (part of it no honest player can recover: §2.2 item 6 there).
- **The value of improving determinization for class X** is bounded by the sufficiency gap IMP(`F+X`) − IMP(`F`) and
  the necessity gap IMP(`O`) − IMP(`O-X`); when the two disagree the classes interact.
- **How much a partly better belief buys** is the dose curve IMP(`moves:seen@p`) over p: a steep start says a small
  belief improvement pays; a curve flat until p ≈ 1 says only near-certainty pays.
- **Content or confidence?** IMP(`moves:seen@p`) − IMP(`moves:seen@p~placebo`) at the same p. And IMP(`…~map`) − IMP(`off`)
  is what an honest sharpening of today's belief buys.
- **For the network** (training observation, injection): the same arms read in strength (pp) or IMP of the policy's
  own pick; a lever the network is told but does not convert is a USABILITY limit, not an information one.

---

## 12. Open risks

- **The screen's factorisation** (§7.3: each seen mon screened with the others at their true facts) is measured by
  the strict joint gate's rejections, not proven; a party-wide effect (Beat Up) is the known coupling.
- **The chain is an approximate posterior sampler.** Rejection draws are exact; a chain started by the directed search
  is Metropolis-Hastings after a short burn-in (`BURN` 6, `THIN` 2): a tightly pinned mon's draws may be correlated
  across worlds. No step reads the truth; the leak reads (§10) check the proposals and the drawn moves, and a spread
  that matches the truth EXACTLY stays rare (3.2 % in §10.3).
- **The proposal check's power** is low where the prior already puts most of its mass on the true item (Leftovers);
  the moves check is the strong one. Neither replaces the independent re-implementation the study registered (§13.1).
- **Cost.** The honest screen dominates: ≈ 28 s per decision at 4 worlds on this contended box (§7.4), the hardest
  decisions minutes. The study's F arm over 600 states × 2 checkpoints is hours of CPU, shared by every arm that keeps
  the seen mons' item / ability / spread hidden (`screen_cache`). A decision the search cannot fit in `MAX_STEPS` is an
  EXCLUSION, counted, never a silent leak.
- **The honest spread prior is a choice** (`EPS`, `FLOOR`, §7.3): Smogon-derived with a uniform floor.
- **Hidden counters** stay the builder's residual (§7.5) until a root-state resample exists.
- **Seed reuse** leaks dice-silent facts (§7.5, measured): speed ties in 20 % of decisions; King's Rock / Focus Band
  and the contact abilities negligible on this pool. Same cure as the counters (a fresh-dice prefix replay).
- **IVs below 30** (a third of pool sets) are outside the screen's support (§7.5): an exclusion risk where the log
  pins the stat.
- The observation cannot state an exact HP, a hidden duration, a placebo or a map fact (§3): the knock-in forks'
  wrong-moves arm needs a placebo in the observation, which needs a prior sampler on the Rust side (not built).

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-12 | **This doc CREATED; the SPEC and the LOG SOURCE BUILT** (owner: "a single dried out implementation for giving both search and training additional information … so that we can understand what the ceilings are and what is our constraint and how much value there would be in improving determinization") | One declarative `InfoReveal` (seven classes × seen / unseen parts, a mass and a source per lever; presets for the X32 levels, the oracle, the pilot's builder and every study arm; canonical text + hash; per-consumer support tables that REFUSE what a consumer cannot honour and NAME its residual leaks); one protocol-derived source of what play revealed, conservative by design | three separate mechanisms (the status quo: incomparable, one leaked); a soft "known = p" cell in the observation (an input the network never sees in play; a Bernoulli keeps every row a possible observation); the observation rows as the reveal source (they miss protocol-only reveals: the study's A2) | `design_hidden_info_binding.md` §2.2, §16.4; `design_value_discrimination.md` Amendment 8 |
| 2026-10-12 | **The SEARCH WORLD BUILDER made HONEST BY DEFAULT, the GENDER PAD wired, the AUDIT SUITE built** | `WorldSampler(reveal=…)`, default `off`: a seen mon's item, ability, spread (its IVs: the Hidden Power type and presence) and so its exact HP are DRAWN and screened against the log by a per-(seen mon, constraint) sampler — prior rejection, then a chain whose search is directed by the first differing line and scored on lenient replays, Metropolis-Hastings once the log fits — shared across a decision's worlds and specs; the pilot's builder is the `pilot` preset; any never-seen mon may sit on a true gender-draw slot (the engine pad); one audit suite (`main.info_audit`) over all three consumers, read on 8 committed-bank decisions × 6 specs: 41 checks PASS (the dose of the told move levers exact), no exclusion, a seen mon's item 1.000 → 0.750, exact spread 1.000 → 0.032, exact HP 124/124 → 70/124 | keeping the pilot's builder as the default (a search not honest by default measures the wrong thing whenever a caller forgets); joint rejection over every seen mon (hopeless: a log pins each attacker's stats to a window); a pure Smogon spread table as the prior (a pool spread outside it has no consistent candidate: the Vaporeon 16 HP / 252 Def case); starting the chain at the truth (a leak until it mixes); filtering a drawn item by the truth's unrevealed moves (a leak) | §7, §9, §10.3; `research_state/measurements/info_levers_2026-10-12/` |
| 2026-10-12 | **The blind replica's three findings answered** (orchestrator relay) | (1) seed reuse through the turn-0 GENDER dice: closed by the engine pad, a `gender_independence` check and its plant (the builder without the pad); (2) DICE-SILENT facts: seed reuse kept, the leak NAMED (`SEARCH.partial`) and MEASURED by counterfactual flips (speed ties 5.0 % of seen mons / 20 % of decisions; King's Rock, Focus Band, contact abilities negligible on this pool); (3) every world OPENED by the playout tool before it is returned, a `playout_open` check and a planted numeric choice. Plus: the Hidden Power's TYPE is a SPREAD fact, its presence a MOVES fact, in all three consumers (§1.1); the screen's Hidden Power state spans every 30 / 31 IV pattern | avoiding seed reuse (needs a playout core that replays on fresh dice constrained to the printed outcomes — not built); refusing `moves` without `spread` on Hidden Power sets (the observation writes it bare instead, byte-safe); an IV prior below 30 (no Smogon source: the owner's call) | §1.1, §7.2, §7.5, §9, §10.1; `seed_reuse_200.json` |
