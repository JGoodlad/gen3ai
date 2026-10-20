# THE WORLD SAMPLER AND DETERMINIZATION: putting weight on the opponent's true hidden team

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this doc
updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-19, latest): F1 TIER A BUILT, behind `WorldSampler(dice="rolls")`, default OFF (§3.3.1, §7 F1).**
Each damage roll of a world's prefix replay may take any roll that prints what the searched side saw, the dice stream
kept aligned, and each world carries a likelihood weight. Measured before building: today's screen rejects 9.0 % of
its candidates that tier A accepts (2.7× the acceptances, 37 of 40 random states). On 60 rebuilt states it cuts the
exclusions from 20 to 6 (3 / 40 → 0 / 40 random; 17 / 20 → 6 / 20 on states the stored builds excluded) and halves the
build's cost; the seen mons' fact marginals barely move. Descriptive builds only; no arm re-run.

**Status (2026-10-19): DESIGN, plus stage 0 of the first experiment READ.** Stage 0 is a zero-playout screen of every
candidate sampler on the existing banks. It is post hoc and descriptive, and its numbers are in §6.1. Nothing is built,
and stage 1 (with playouts) is written so it can be registered (§6.2). Two owner questions are open (§8): may a sampler
fitted on real teams run at search time, and may a live search exclude a decision. Every claim here inherits the
PROVISIONAL tag of the reads it rests on (`search_ledger.md` §8 rule 3).

**Vocabulary.**
- A **world** (a "determinization") is ONE fully specified guess of everything we cannot see: the opponent's six
  Pokémon with their moves, items, abilities and spreads (nature, EVs, IVs), plus the hidden battle state.
- The **sampler** is what produces the worlds. **Determinization** is the whole process: belief → concrete worlds that
  are consistent with everything we saw.
- **IMP** is decision improvement: the win probability gained per decision by playing the search's pick instead of the
  policy's own top choice, judged on held-out truth playouts (`search_ledger.md` §6.2).
- **O** = search in the TRUE world (the oracle); **F** = search in the fully honest worlds we build today;
  **G = IMP_O − IMP_F** is what hidden information costs search.
- **U** = the opponent's never-seen Pokémon (identity and sets); **M** = the unrevealed moves of the opponent Pokémon we
  have seen.
- **LEAF-K8** = the mean result of 8 worlds, each played to the end; **TRUNC-4-K8** = 8 worlds played 4 turns, then the
  critic.

---

## 0. Summary, in plain language

Search would work if its guessed worlds were close enough to the truth. The measurements say how close: the search's
average needs about a fifth to a third of its weight on the TRUE world before it beats the policy at the 8-rollout leaf
(`search_ledger.md` §5.1), and the network's own guesses are worth about 1 % today, at 25M steps and at 225M alike. The
gap is not closing with training. The bots that now top Showdown's doubles ladder get near-exact worlds for free
(team preview, open team sheets, or a published set list); gen-3 OU has none of those. So the piece that binds search
here is the **world sampler**.

This doc defines what a better sampler means (§2), then how a belief becomes concrete worlds consistent with every fact
play revealed, what our builder already enforces and what it gets wrong (§3). It ranks six candidate samplers (§4),
updates the search's shape to the top bots' one-turn simultaneous-move matrix in its correct hidden-information form
(§5), and gives a cheap first experiment with a decision rule (§6).

The zero-playout screen (§6.1) already changes the picture:
- **The network already knows the never-seen SPECIES.**
  - At X26 225M its belief puts 0.31 of its mass on the exact set of never-seen Pokémon (on our training-pool
    opponents), against 0.02 for the Smogon-based draws search uses today.
  - That share ROSE from 0.21 at 25M while the belief's value stayed at 1 %.
- **It is the SETS that are missing.**
  - A world gets every seen Pokémon's hidden moves right 1 % of the time under Smogon, 2 % under the network, and at
    most 3.5 % under any candidate.
  - So a world right about the never-seen species AND the moves is drawn at most a quarter of a percent of the time,
    about 100× short of the target. That is why a species belief this sharp is still worth only 1 % of the true world.
- **One honest world in twenty WAS an impossible team (F2, FIXED 2026-10-19).** Drawing each move slot independently
  gave sets Showdown's own validator rejects (egg-move and event combinations), in a third of the states. The builder
  now redraws every set until the validator accepts it (0 illegal worlds in a 316-state rebuild, no new exclusion).
- **The true-dice replay was over-sharp (F1, tier A BUILT 2026-10-19, default OFF).** A world had to reproduce every
  printed damage number with the truth's own damage roll. With `dice="rolls"` any of the 16 rolls that prints the same
  thing will do. Before building, the measurement: 9 % of the candidates today's screen rejected fit on another roll.
  On 60 rebuilt states the exclusions fall from 20 to 6, and the worlds' facts barely move (§3.3.1).
- **So the first target is the moves of the seen Pokémon, kept consistent with the rest of the team**: real teams and
  sets (team-level and set-level models), the legality of move combinations, and the opponent's own choices as
  evidence. A better species guess alone will not do it. Stage 1 (§6.2) then asks the open question: do worlds that are
  PARTLY right carry the value instead?

---

## 1. Why the sampler is the binding piece (the evidence, by pointer)

| evidence | what it says | where |
|---|---|---|
| honest search never beats the policy, 25M → 225M | IMP_F −0.012 / −0.022 / −0.004 / −0.011; G_F +0.075 → +0.053 (change NOT DETECTED, and the decline is the oracle's edge falling) | `design_hidden_info_binding.md` §12.13.7; `search_ledger.md` §1 |
| the network's belief is worth ≈ 0.01 of the true world, and does not rise | TODAY 0.00 / 0.01 / 0.00 / 0.01 against P-STAR 0.20–0.30 at LEAF-K8 | same; §5.1 |
| the gap is hidden TEAM information | G +0.070 (Tier 1), +0.072 (E2), robust to the honest builder | `design_hidden_info_binding.md` §16.8, §16.12 |
| the never-seen Pokémon carry more than the moves, and the moves are NECESSARY | E2: U-FIRST (Δs +0.017, Δn +0.011); n_M 0.58 (B4 holds): a sampler that got U right would still lose more than half the gap with wrong moves | §16.12 |
| identifying the team pays; set priors alone do not | POOL (the closed pool's teams consistent with the reveals) recovers r = 0.80 of G; POOLSETS (the pool's per-species set frequencies) −0.02 | §16.8 |
| information is a search lever, not a policy lever | knock-in forks NULL at +3M | §8.7 |
| the top doubles bots average 16 sampled worlds per cell of a one-turn matrix, from near-exact information | team preview / open team sheets; randbats bots use the published set list | `../research_state/literature/showdown_bots_landscape_2026-10-18.md` §2, §5 |

The POOL row and the doubles-bots row together are the argument: what pays is knowing WHICH team, as a whole. A sampler is the
only place that knowledge can enter search.

---

## 2. THE TARGET, defined

"A better world" is measured at four levels, cheapest first. Levels 1–3 need no playout; level 4 is the verdict.

| level | meter | definition | today (F) | target | instrument |
|---|---|---|---|---|---|
| 1. likelihood of the truth | **log p(U\*)** | log-probability that a world's never-seen species SET equals the truth's (exact: the sum over the h! draw orders) | −6.43 | higher is better; a proper scoring rule, comparable across samplers | `measurements/world_sampler_screen_2026-10-19/screen_samplers.py` |
| | **p_exact(U\*)** and **hit** | the same as a probability, and the expected share of never-seen slots holding a TRUE never-seen species (set-based, the study's `match`) | p_exact 0.020, hit 0.149 (Tier 1's set-based hit 0.18) | — | same; agrees with the stored F worlds (§6.1 GIGO) |
| 2. top-k recall | **recall@h** | the share of true never-seen species among the h species the sampler rates most likely (h = the slots to fill) | 0.35 | — | same |
| | **team recall@8** | P(at least one of 8 worlds holds the exact never-seen species set) = 1 − (1 − p_exact)^8, per state | 0.12 (the network at 225M: 0.53) | — | derived from the rows |
| 3. exact-world mass | **q_SM** | P(a world is exact on the two binding classes: U's species AND every seen mon's unrevealed moves). It is the unit of the dose curve's p, restricted to what binds (items, abilities and spreads add ≈ 0.003 alone, `search_ledger.md` §5.2) | < 0.0001 | **≥ 0.2 at LEAF-K8**; the continuous TRUNC-4-K8 leaf covers half its rise by p ≈ 0.15 | same |
| 4. downstream value | **TODAY_X** | the p-equivalent: the weight on the true world at which the M8 dose curve equals sampler X's own IMP (value units) | 0.00–0.01 (FN, X26) | **≥ P-STAR** (0.20–0.30 at LEAF-K8 along X26) | `score_blind.dk_family`, `posthoc_mat.py` |
| | **G_X, r_X** | G_X = IMP_O − IMP_X; recoverability r_X = (IMP_X − IMP_F) / G_F | G_F +0.053 (X26 225M) | r_X ≥ 0.25 is MATERIAL; IMP_X > 0 with its lower bound clear is search ENABLED | Tier 1's `t1_data` + the maturity reader |

**How the levels connect.** A world exact on U and M is the true world up to items, abilities and spreads, so q_SM is a
near-LOWER bound on TODAY in the dose curve's own unit (partly right worlds add a little: the random-honest-world
placebo M5W is worth ≈ 0, `search_ledger.md` §5.1). Stage 1 (§6.2) checks this calibration. Two cautions:
- **p\* is the 8-rollout leaf's resolution as much as a belief need** (the red team, `search_ledger.md` §5.1). Every
  target is quoted at both leaves.
- **These banks' opponents are our own TRAINING-POOL teams.** Every level-1 to level-3 number for the network's belief
  is optimistic for the ladder (§7 F3, F4).

---

## 3. DETERMINIZATION: from a belief to concrete worlds consistent with every observed fact

### 3.1 What a world must specify

A world is the battle's input log with the opponent's team replaced, played from the root on a fresh dice seed. To be
playable it must fix:
- the opponent's six species and their slots;
- each mon's moves, item, ability, nature, EVs, IVs (and so its exact stats and Hidden Power type), gender and level;
- the hidden battle state at the root:
  - each seen mon's exact HP (we see a percentage);
  - its PP;
  - the hidden counters (sleep, confusion, Encore, Disable and lock durations rolled before the root);
- the dice.

A world is CONSISTENT when, replayed to the root, it reproduces every protocol line and every observation row our side
saw. It is HONEST when it was drawn without reading anything our side could not see.

### 3.2 The facts, and how each is honoured today

Today's builder (`main.value_search.worlds` / `world_draws` / `world_screen`; `design_information_levers.md` §7) works in
three steps:
1. it keeps what play revealed (the ONE log source, `agents.info_levers.revealed`);
2. it draws the rest from the Smogon prior;
3. it keeps a world only if a replay of the whole prefix is BYTE-EQUAL to the truth's, for the searched side's protocol
   and rows.

| fact (owner's list, 2026-10-19) | what it constrains | how it is enforced today | honest? | the gap |
|---|---|---|---|---|
| **revealed species, moves, items** | identity; a set's known slots; a held item | kept TRUE from the log source: own `|move|` lines (not called moves, not Transform / Mimic copies, never Struggle), `-item`, `-enditem`, `[from] item:`, Trick / Thief transfers. Conservative: an ambiguous owner reveals nothing, so the fact is drawn and the gate forces it back | yes | none known |
| **damage ranges** → EVs, nature, Hidden Power type | the attacker's Atk / SpA and the defender's HP / Def / SpD, given the board (boosts, items, weather, screens, crit) | the replay gate: our mon's exact HP after their hit; their HP percentage after ours; the screen draws the seen mon's spread and keeps it only if the replay matches | **Default (`dice="seed"`): NO, too sharp.** The prefix replays on the TRUE dice seed, so a world must reproduce the observed damage with the truth's own damage ROLL. An honest player knows only that the damage fell inside the 16-roll range (× the printed crit flag). **`dice="rolls"` (F1 tier A, BUILT 2026-10-19, default OFF): YES for the roll's value**: any roll that prints what the searched side saw, the world weighted by Π k / 16 | **F1** (§7): MEASURED and PARTLY RESOLVED (§3.3.1): today's screen rejects 9 % of its candidates that fit on another roll; tier A cuts the exclusions 20 → 6 of 60. The crit roll and the other dice remain the truth's (tier B) |
| **speed order** | Speed inequalities (priority bracket, paralysis, Quick Claw, which rolls battle-wide each turn in gen 3) | the replay gate (who moved first is printed) | partly: a speed TIE spends a die without printing a line, so under the true seed the gate pins the tie | MEASURED: a seen mon's Speed ±1 is rejected for 5.0 % of seen mons, in 20.1 % of decisions (`design_information_levers.md` §7.5) |
| **abilities** | the ability of a two-ability species | announced → kept; otherwise drawn ∝ Smogon, and the gate rejects one whose effect would have printed (Intimidate on entry, an immunity line) | partly: a dice-rolling ability that did NOT trigger (Static, Flame Body, Effect Spore, Cute Charm, Shed Skin) is visible to the gate through the dice | MEASURED: negligible on this pool |
| **PP use** | the number of uses of each revealed move; Struggle implies no usable move | the replay consumes PP exactly as the truth did; a never-used move has full PP; `playout_open` refuses a Struggle the world does not force | yes (the opponent's PP is never shown to us in gen 3; only our count of uses) | a Struggle / Taunt / Disable / Choice-lock history is a strong constraint on the UNREVEALED moves, honoured by rejection only (rare, unmeasured) |
| **Smogon legality** | team clauses; bans; learnsets; event and egg-move combinations; EV total ≤ 510 | `team_legality` clauses (a world that breaks a rule the truth does not is rejected); only engine-runnable moves and items are drawn; and, since 2026-10-19, every drawn set and returned world passes Showdown's `TeamValidator('gen3ou')` (rejection: a set is redrawn until accepted; the truth must pass the same rule or the build raises) | **yes** (F2 RESOLVED) | none known. Before: 5.2 % of stored F worlds illegal, in 32 % of states (§6.1, §7 F2) |
| exact HP of a seen mon | from its spread and the replayed history | follows from the screened spread | as honest as the spread (see damage) | 70 / 124 seen mons' exact HP equal the truth's (`design_information_levers.md` §7.1) |
| **hidden counters** | sleep / confusion / Encore / Disable / lock turns left | NOT drawn: rolled before the root on the true seed, so every world holds the truth's | **no** (the builder's declared residual) | cure: a root-state resample (not built) |
| gender | the turn-0 gender draws | the engine's gender pad places the drawn never-seen mons on the true draw slots | yes (one residual coin, unobservable) | — |
| **opponent's choices** | what a player who held world w would have done | **not used** | — | candidate (f), §4 |

### 3.3 How consistency is checked: rejection, constraint propagation, and the hybrid we should build

**Today: rejection against the engine.** A proposed world is replayed through the Rust core and compared byte for byte.
- **Its strength:** the ENGINE is the specification. Every mechanic is right by construction: Beat Up's party-wide hits,
  PP, abilities, and the Counter / Hidden Power rule of §3.4. No hand-written mechanics means no new GIGO class.
- **Its three weaknesses:**
  1. **Acceptance collapses when many facts bind together.** The builder therefore factorises: each seen mon is
     screened with the others at their TRUE facts, then the joint world is re-gated (150 + 111 + 164 joint rejections on
     the three banks of §6.1). A per-mon chain (a directed search, then Metropolis-Hastings) handles the hard cases.
     When the chain fails in 200 steps, the decision is EXCLUDED (§3.5).
  2. **It needs the true dice to make the replay deterministic.** That makes the posterior too sharp (damage rolls,
     speed ties, non-triggering dice abilities: §3.2). The damage-roll leg is cured behind `dice="rolls"` (tier A,
     §3.3.1); the others need tier B.
  3. **Cost.** About 28 s per decision at 4 worlds on a contended box; the hardest decisions take minutes.

**Constraint propagation** derives the feasible set of each hidden fact from the log analytically: an interval on the
attacker's stat from each damage event under ANY of the 16 rolls; Speed inequalities from the move order; item and
ability equalities and exclusions (no Leftovers heal at the end of a turn on a damaged, unknocked mon ⇒ not Leftovers).
It then samples the prior restricted to that set. It is fast, it is dice-honest by construction, and it is what a human
player does with a damage calculator. Its danger is re-implementing mechanics: one wrong rule silently removes the
TRUTH from the feasible set.

**The hybrid (recommended): PROPOSE → PROPAGATE → VERIFY, with dice marginalised.**
1. **Propose** from the sampler (§4).
2. **Propagate** cheap, conservative constraints to prune proposals before any replay. "Conservative" means a rule may
   only remove what is certainly impossible.
3. **Verify** by an engine replay whose dice are CONSTRAINED to the printed outcomes, instead of reusing the truth's
   seed. At each random event, the replay accepts any outcome that reproduces the printed line and multiplies the
   world's weight by that outcome's probability: for a damage roll, the share of the 16 rolls that print the observed
   number; for a crit, a miss or a secondary effect, the printed branch. This is the cure the levers doc names and has
   not built ("a playout core that replays the prefix on fresh dice constrained to the printed outcomes",
   `design_information_levers.md` §7.5). Implemented in the Rust core, it removes F1, the speed-tie leak and the
   dice-ability leak at once. It also returns a LIKELIHOOD weight per world in place of a 0 / 1 verdict. The search can
   use that weight directly (§3.6).
4. **A throwing guard, per the owner's GIGO rule:** the TRUTH must pass every propagation rule on every bank decision.
   A rule that excludes the truth anywhere is a defect and raises; it is never an exclusion. That is the test that
   catches a mis-ported mechanic such as §3.4's. The legality rule (F2, BUILT 2026-10-19) is the first rule that
   carries it: `world_legality.Legality.require_true_team` raises before any world is drawn.

#### 3.3.1 What the VERIFY step needs to accept any dice outcome consistent with the log (F1; tier A BUILT 2026-10-19 behind `dice="rolls"`, default OFF; tier B not built)

**Why today's replay is too sharp.** The world replays the prefix on the TRUTH's seed. The Rust core's dice are one
`Prng` stream (`src/rust_sim/src/prng/mod.rs`), consumed in a fixed order, so the world's k-th draw IS the truth's k-th
draw. A seen mon's hit then reproduces the printed HP only with the truth's own damage ROLL. An honest player knows only
that the roll was one of 16 (and the printed crit flag). The screen keeps the spreads that fit ONE roll, not the
spreads that fit ANY roll: its posterior on Atk / SpA / HP / Def / SpD is narrower than honest, and a mon whose spread
fits only another roll is an exclusion (§3.5 cause 3).

**What the engine already has.** `Prng` is `Clone` (the clone-and-branch search primitive); the damage function builds
the full 16-roll table (`damage.rs`, `rolls[r]`); every draw is counted (`POKESIM_PRNG_TRACE`, `TRACE_DRAW_N`).

**Two tiers, cheapest first.**

| tier | what changes | cures | does not cure | cost (UNVERIFIED) |
|---|---|---|---|---|
| **A: the damage roll's VALUE, stream kept aligned** | At each damage-roll draw the PRNG call is consumed as today (so every later die stays the truth's and the stream never desyncs), but its VALUE is replaced by a roll from the CONSISTENT set: the rolls whose damage prints the observed HP line (our mon: the exact HP from our request; theirs: the shown fraction). The world's weight is multiplied by k / 16 (k consistent rolls), and k = 0 rejects. Built as a per-draw override list (keyed by draw index) fed by a first pass that records each damage site's 16 candidates and its target | the damage leg of F1 (the main one for spreads), and the exclusions it causes | speed ties, dice abilities that did not trigger, crit-rate differences, the hidden counters (F6): all still the truth's dice | ≈ 1–2 agent-days: an override channel in the core + two replays per candidate |
| **B: constrained dice (§3.3's VERIFY)** | Every draw site is TYPED (damage roll, crit, accuracy, secondary, speed tie, Quick Claw, sleep / confusion turns, ability procs). A printed outcome is FORCED, weight × its prior probability; an outcome the protocol never prints (a silent speed tie, a Static that did not trigger, a pre-root sleep counter) is SAMPLED from its prior. Lookahead uses `Prng`'s clone: at a site, branch the outcomes to the next printed line and keep the consistent ones (sequential importance sampling: one sampled ∝ prior, the weight × their mass). The verdict becomes a log-LIKELIHOOD per world | F1 whole, the speed-tie and dice-ability leaks (§7.5 of the levers doc), and F6 (counters drawn, not inherited) | nothing in this list; it changes the screen from 0 / 1 to weighted | several agent-days: typed sites across the core, an in-core resolver fed the viewer's line stream, the gate in Python weighted |

**The guards either tier must carry.**
- The truth-passes guard: the TRUE team has weight > 0 at every bank decision, and with every site forced to the
  truth's realized draw the constrained replay is BYTE-identical to today's seeded replay (the oracle identity; run
  in the emission self-check build).
- A Counter / Mirror Coat / Hidden Power regression (§3.4) for any rule that reads a move's category.
- The weight is a likelihood, not a probability of the world: a search that uses it must normalise over the J worlds.

**Measured before the builder was wired** (`research_state/measurements/world_rolls_f1a_2026-10-19/`, step 1). The
measurement used the engine itself rather than an analytic 16-roll check: every gate replay today's builder made
(`dice="seed"`, the studies' call: J 8, 60 tries) was SHADOWED by the same replay with tier A's match control, both
judged by the builder's byte gate. On 40 random stage-0 states:
- today's screen ran 22,132 candidate replays and accepted 1,084; **tier A accepts 2,978 of the same candidates
  (2.7×)**: 1,894 of today's 21,048 rejections (9.0 %), in 37 of the 40 states;
- a rescued candidate took 1.65 rolls off the stream on average; there are 15 damage-roll sites per prefix on average;
- **nothing today accepts is rejected by tier A** (0 of 27,236 screen and joint replays, 3 excluded states included):
  the control reproduces today's trajectory whenever the stream's roll is consistent.

Not negligible, so tier A was built.

**Tier A AS BUILT (2026-10-19).**
- **The Rust core** (`src/rust_sim/src/rolls.rs`, `gen3_roll_control_v1`; contract and gates:
  `designs/rust_sim/search_and_replay_drivers.md` "The damage-roll control"). Every damage roll (the single hit, a
  multi-hit strike, a confusion self-hit, Beat Up, a Future Sight cast, a Jump Kick crash) goes through ONE site that
  draws the `random(16)` exactly as before and returns the roll used. Unarmed (every training, eval, live and gate
  path) nothing changes. Three modes:
  - RECORD (the truth's pass) records, per site, its key (kind, sides, move; Hidden Power's type folded) and what the
    searched side SAW: our mon's exact HP, the opponent's percent, a Substitute held / broke;
  - MATCH (a world's pass) takes the stream's roll when it prints the target's sight, else the consistent roll nearest
    it, and adds log(k / 16) to the world's weight; an empty set makes the world infeasible (the gate rejects it);
  - FORCE replays a table `[[site, roll], …]`, the trajectory a match took.
  A site whose sight a post-roll modifier can change is PINNED (the stream's roll, factor 1 / 16): Beat Up, a Future
  Sight cast and a Jump Kick crash always; a Focus Band holder, Endure or False Swipe when some roll is lethal. On the
  60 states no site was pinned.
- **The builder** (`main.value_search.worlds`, `WorldSampler(dice="rolls")`; `dice="seed"` is today and the
  default). The truth's replay records the targets once per decision; every screen and joint gate replay matches
  against them; each returned world carries `Worlds.log_w` (Σ log k / 16) and `Worlds.rolls` (its force table, also
  its log's `"rolls"`). The playout tool (`rust_env_playout_open`'s `log.rolls`) and the info audit's regate replay
  that table, so search plays the world the gate accepted. The shared screen caches are keyed by the dice mode.
- **The guards, as built.** THE TRUTH-PASSES GUARD: per decision, the true team's own match replay must pass the
  gate on exactly its own rolls (no forced site) with a finite weight, else `RollMatchError` raises (a defect, never
  an exclusion; planted in `world_rolls_integration_test.py`). THE IDENTITY: record, a force of every site's own
  roll and a self-match are byte-identical to the unarmed replay (`tests/rolls_test.rs` and the Python test). THE
  ENGINE IS THE ORACLE for the consistent set: each of the 16 rolls forced and read off the viewer's stream gives the
  same set and the same k the match computes. The F2 legality rule is untouched (every set and world still passes
  the validator; the truth-team legality guard runs first).

**The re-measure (step 5): the builder rebuilt with `dice="seed"` and `dice="rolls"` on the same 60 states** (the 40
random ones and 20 the STORED builds excluded; descriptive, builds only; `result_f1a.json`):

| | `seed` (today) | `rolls` (tier A) |
|---|---|---|
| excluded, 40 random states | 3 (7.5 %) | **0** |
| excluded, 20 stored-excluded states | 17 | **6** (12 recovered, 1 newly lost) |
| excluded, all 60 | 20 | **6** (15 recovered, 1 lost) |
| exclusion causes | 19 screen chains, 1 tries | 5 screen chains, 1 tries |
| defects (a guard raised) | 0 | 0 |
| wall time per state (mean, contended cores) | 96 s | **42 s** |
| screening replays per state (39 states ok in both) | 457 | 262 |

On the 39 states both modes built (312 worlds each):

| | `seed` | `rolls` |
|---|---|---|
| a seen mon's item / ability / spread equal to the truth's | 0.953 / 0.989 / 0.056 | 0.951 / 0.980 / 0.047 |
| a seen mon's EXACT HP at the root equal to the truth's | 0.601 | 0.588 |
| distinct spreads per seen mon across the 8 worlds | 2.21 | 2.65 |
| never-seen species marginal, TV seed vs rolls | — | 0.0005 |
| the info audit's regate passes | 39 / 39 | 39 / 39 |

Under `rolls`, 91 % of worlds took at least one roll off the stream (4.8 sites on average). The weights are mild: the
effective sample size of the normalised weights is 7.3 of 8 on average (minimum 2.1; 2 of 54 states below 4).

**Reading.** Tier A widens the honest posterior on spreads (more distinct spreads, slightly fewer exact matches with
the truth: the true-dice replay was optimistic) and removes most of the exclusions, at less than half the cost, while
the fact marginals barely move. It is a correctness and robustness change, not a lift lever: the sets (§0) are still
what the sampler lacks. Worlds built with `dice="rolls"` are not comparable with `dice="seed"` worlds; no study arm
was re-run, and every study in flight keeps its own pin.

### 3.4 A checked edge case every propagation rule must honour: Counter, Mirror Coat and Hidden Power (gen 3)

In gen 3, **Counter reflects ANY Hidden Power, whatever its type, and Mirror Coat NEVER does**, even when the Hidden
Power's type is special. This is verified at source:
- `deps/pokemon-showdown/data/mods/gen3/moves.ts`: Counter's `onDamage` fires on `effect.category === 'Physical' ||
  effect.id === 'hiddenpower'` (line 176); Mirror Coat's on `effect.category === 'Special' && effect.id !==
  'hiddenpower'` (line 420);
- a typed Hidden Power id normalises to `hiddenpower` (`sim/pokemon.ts:1617`);
- our Rust port and `agents/model/move_resolution.py` (its docstring: "Hidden Power: always Counter, never Mirror
  Coat") match.

What it means for determinization:
- A propagation rule "Counter succeeded against their Hidden Power ⇒ its type is physical" would be WRONG. Counter
  succeeds against Hidden Power Fire too. The success says nothing about the Hidden Power's type.
- The same holds for "their Mirror Coat failed against our Hidden Power ⇒ our hit was physical".
- The engine-replay gate gets this right by construction. A hand-written propagation layer must carry it as a named rule
  with a regression test that fails on revert (a Countered Hidden Power Fire must not exclude HP Fire spreads), and the
  truth-passes guard of §3.3 catches it if it is missed.
- More generally, **a category inferred from a move's TYPE is not a gen-3 fact** for Hidden Power. Every rule that
  reads a move's category must go through the engine's own category function.

### 3.5 What today's builders enforce and miss: the exclusions are a symptom

**The exclusion rate, measured on the three banks of §6.1** (the F rows; counts only, no outcome read):

| bank | states | EXCLUDED |
|---|---|---|
| maturity interim, LEAN | 320 | 33 (10.3 %) |
| E2 end2 | 600 | 40 (6.7 %) |
| E2 prod | 600 | 52 (8.7 %) |
| **pooled** | **1,520** | **125 (8.2 %)** |

Of the 125, **121 are a seen mon whose screen found no item / ability / spread in 200 chain steps**, and 4 had too few
worlds within the try limit (the joint gate or a clause). The seen mons the screen fails on are the pool's workhorses
with custom spreads: Tyranitar 23, Zapdos 17, Swampert 14, Jirachi 13, Metagross 12, Celebi 9. Three causes are known:
1. a spread far from Smogon's table (the smooth floor covers it, slowly);
2. IVs below 30 that the log pins (a minimised Attack seen through a confusion hit; the owner's 2026-10-12 ruling keeps
   such decisions excluded);
3. **the true-dice replay**, which demands the observed damage from the truth's own roll. That is the narrowest
   possible window. **Tier A (`dice="rolls"`, §3.3.1) widens it: on 60 rebuilt states the exclusions fall 20 → 6,
   and on 20 states the stored builds excluded, 17 → 6.** The rest are causes 1 and 2 and chain-path failures.

Why this matters beyond the 10 %:
- **The exclusions are not missing at random** (on the states only the blind replica could fit, n_M is 1.17,
  `search_ledger.md` §6.1). Every honest read is biased toward the decisions the builder can fit.
- **A LIVE search cannot exclude a decision.** It must always return worlds. An in-play sampler needs a FALLBACK: the
  dice-marginalised check first, then a widened prior support, then approximate worlds that carry their likelihood
  weight. Owner question Q2 (§8).

**What the builders miss, in order of expected harm to search:**

| # | gap | class | evidence |
|---|---|---|---|
| 1 | the seen mons' unrevealed moves are drawn from Smogon's per-species MARGINALS, independently of each other, the teammates, the item and the spread | **M** (necessary, n_M 0.58) | §6.1: one seen mon's hidden moves all right 0.12; every seen mon's 0.011 |
| 2 | the never-seen species are drawn ∝ the MEAN teammate share of the seen species, not jointly; the drawn mates do not condition the next draw | U | §6.1: p_exact(U\*) 0.020 (the joint (b) 0.045) |
| 3 | the opponent's own choices are not used as evidence | M, U | unmeasured; candidate (f) |
| 4 | the true-dice replay over-sharpens spreads and causes exclusions | I / S (+0.019 once U is known) | F1, PARTLY RESOLVED: tier A (`dice="rolls"`, default OFF) cuts exclusions 20 → 6 of 60 (§3.3.1); the crit and other dice remain the truth's |
| 5 | the hidden counters are the truth's | a leak (optimistic) | the builder's residual |
| 6 | ~~no full-validator legality check~~ RESOLVED 2026-10-19 | M (impossible sets) | F2: 5.2 % of worlds, 32 % of states before; 0 after (§7) |

### 3.6 How determinization meets search

**Per-world solves averaged (perfect-information Monte Carlo, PIMC).** The one-ply search today plays ONE action for all
worlds: it averages each action's value over the worlds and picks the best. That is sound at the root: strategy fusion,
choosing as if we knew the world, does not happen there because the choice is shared. Inside a playout, our side's
continuation acts on its own simulated observation, which is honest within the world. The known costs:
- **non-locality**: the value of a later state depends on beliefs the playout does not carry (Frank & Basin 1998; Long
  et al. 2010, cited and verified in `design_value_discrimination.md` §14.1);
- **the modelled opponent**: the opponent replies greedily in each world.

**The top VGC bots' one-turn matrix** averages each cell (our action × their action) over the sampled worlds and solves
the matrix. That is valid when every world gives the opponent the SAME actions, which open team sheets guarantee. In
gen-3 OU the opponent's moves and switch-ins DIFFER by world, so a column "their action b" means different things in
different worlds. §5 gives the correct form, in which each world is an opponent TYPE. Solving each world's matrix
separately and averaging OUR strategies is the wrong way: Foul Play's 2019 "nash" mode did that, and its author called it
wrong (`design_value_discrimination.md` §14.4).

**Information-set approaches** (information-set Monte Carlo tree search, IS-MCTS; ReBeL's public belief states) keep one
tree over what WE know and re-determinize per iteration. They remove strategy fusion in a deep tree. At ONE ply, which
is the owner's scope rule, the one-turn game in which the opponent's type is the world IS the information-set solution of
that turn. A deeper information-set search is outside the scope rule and is the belief-size wall of
`design_value_discrimination.md` §14.2.

**Weighted worlds connect the belief to search.** The dose curve says the search needs WEIGHT on the true world. Worlds
need not count equally:
- the dice-marginalised check returns a likelihood per world (§3.3);
- the opponent's past choices give another (candidate (f));
- a team-level model gives a prior per world.

The search's average, and the matrix's world mixture, should use the normalised product of these weights. Uniform
weights are today's special case.

---

## 4. CANDIDATE SAMPLERS, ranked

Ranked by expected lift per cost on the target of §2. Every number is from §6.1 (stage 0, zero playouts, post hoc) unless
stated.

### (c) A TEAM-LEVEL model fitted on real teams: rank 1

- **Method.** A world's opponent team is a real team CONSISTENT with what play revealed: it holds every seen species,
  and each seen species' set holds its revealed moves. It is drawn from a corpus, backing off to (b) when the corpus
  has few consistent teams, with λ = n / (n + α) for n consistent teams (α = 1 in stage 0, which proved over-confident;
  stage 1 fits α on a held-out population, §6.2). The draw takes the never-seen mons' species and
  WHOLE sets, and the seen mons' unrevealed moves, from the corpus team. The seen mons' item, ability and spread stay
  with F's honest screen, so a corpus team never has to fit the log's damage numbers. This is Tier 1's POOL arm
  (`measurements/hidden_info_tier1_2026-10-13/pool_worlds.py`) with an OPEN corpus and a backoff.
- **Data.**
  - The forum collection: `utils.team_corpora` `w_forum_v1`, 2,424 teams (2,415 when the screen ran, before the crash-class lift) with the pool's exact copies removed (the full
    collection is 5,330, `measurements/smogon_forum_teams_v2_2026-10-18/`).
  - The ladder-replay corpus: `utils.ladder_corpus`, 22,813 `hl_05_26` teams. Their unrevealed parts were filled by
    Metamon's reconstruction, by a method UNVERIFIED here.
- **Why rank 1.** It is the only candidate with a measured analogue that pays: POOL, the same method on a CLOSED
  population that contains the truth, recovers r = 0.80 of G. When all 8 POOL worlds were the true team (70 % of
  states), r was 1.00; when none was (11 %), r was 0.14 (`design_hidden_info_binding.md` §16.10 C6). And in stage 0 it is
  the best OPEN-population source on BOTH binding classes: the ladder corpus for the species (6× F's exact-set mass),
  the forum's per-species sets for the moves (2× F's one-mon rate). Its open-population ceiling is far below POOL's
  (below), so rank 1 is a statement about direction, not about reaching p\*.
- **Expected lift.** Bounded by coverage: how often the opponent's team, or a near copy, is in the corpus.
  - On the LADDER: 18 % of `hl_05_26` teams share all six species with a sample team, but only 0.2 % share four or more
    full sets (`design_diversity_teams_opponents.md` §0). Expect real lift on the species and little on the sets.
  - **Stage 0 on our banks (§6.1)** splits it into a species part and a set part:
    - **Species.** The ladder corpus holds the exact never-seen species set for 38 % of states, and raises the exact-set
      mass six-fold over F (0.020 → 0.125); the forum corpus holds it for 14 % with near copies of the truth removed.
    - **Sets, per species.** The forum's sets of each species double the chance a seen mon's hidden moves are all right
      (0.12 → 0.26).
    - **Whole-team identification** (a corpus team equal to the truth on species AND moves) almost never happens in an
      open corpus: exact-world mass 0.0005 (forum STRICT) to 0.0024 (ladder).
  - So in an open population, (c) is a species model plus a set model, not POOL's team identification. Its lift is
    probably far below POOL's 0.80, and stage 1 measures it.
- **Cost.** Stage 0: done (≈ 5 CPU-min). Stage 1: a measurement driver around `pool_worlds.py` (no harness build), then
  about 35–75 CPU-s per state per arm (§6.2).
- **The rule question: is a sampler fitted on forum teams a "prior"?** The owner's rule (2026-08-15) is "ALL priors must
  be Smogon-derived; only the MODEL gets bias against the pool". Its test is INPUT vs EXPERIENCE: anything the network
  READS must trace to Smogon usage, to ground-truth labels, or to ladder replays.
  - A search-time sampler is not an input the network reads: no table enters the observation or `data/`. The network
    only sees the simulated battles, exactly as it sees today's Smogon-drawn worlds. That argues it is a SEARCH-TIME
    component, like the pool-posterior measurement arms the hidden-info study already runs (whose own status is that
    study's open §17 Q2).
  - But it IS a prior in the Bayesian sense: it decides which opponents search believes it faces.
  - The forum collection overlaps our training pool (711 of the 719 pool teams are in it verbatim), and X38 (D1) would
    make `w_forum_v1` TRAINING data. Fitted on that, the sampler would believe the ladder looks like our training
    teams, which is exactly the harm the rule exists to prevent.
  - The ladder-replay corpus is different: "ladder replays" is a source the rule already names.

  **Decision: AMBIGUOUS. It is owner question Q1 (§8).** Until it is answered:
  - (c) runs only as an OFFLINE MEASUREMENT;
  - it never enters `data/`, the network's input or its belief prior;
  - it is never fitted on the population it is evaluated on (the pool, or any `--team-mix` corpus a run trained on);
  - its ladder value is read on a ladder-derived population, never on the pool.

### (f) The opponent's own choices as evidence: rank 2 (a NEW candidate)

- **Method.** Weight each world by how likely the opponent's OBSERVED choices were if world w were true:
  w_weight ∝ Π_t π_opp(observed action_t | the opponent's view at t in world w).
  - π_opp is an opponent model. Our own network acting from the opponent's seat is the cheapest choice; a
    behaviour-cloned human model (Metamon-like) is the right one for the ladder.
  - This is the inference a player makes: "if it had Earthquake it would have used it on my Magneton by now".
  - It is the one honest source of evidence about UNREVEALED MOVES. An unused move leaves no protocol trace, but the
    choices made instead of it do.
- **Data.** None new. Computing the weights needs one forward per past opponent decision per world: the gate's replay
  must also emit the opponent side's rows.
- **Expected lift.** It targets M, the class §6.1 shows is missing. UNVERIFIED; no analogue has been measured.
  Re-weighting only 8 F worlds cannot put weight on a truth none of them holds, so it needs a larger proposal set
  (e.g. build 64 and resample 8 by weight), or it is paired with (c).
- **Hazards.**
  - On the states from games against our own snapshots, the opponent model nearly IS the opponent: optimistic.
  - On the scripted-bot states it is mis-specified.
  - So it must be read per opponent stratum, and only a human-model version speaks for the ladder.
  - It is a likelihood from a model, not a hand-computed input the network reads, so the priors rule does not bind it.
    The fact-vs-judgment test does not apply either: it is a search-time weight.
- **Cost.** About 1 agent-day for a measurement driver (both sides' rows in the replay, the weights, a resample).
  Stage 1c (§6.2), after 1a.

### (b) Smogon teammates, jointly (naive Bayes over the teammate table): rank 3

- **Method.** Each never-seen slot is drawn ∝ u(s) · Π_r lift_r(s) over every seen AND every already-drawn species r,
  with lift_r(s) = (t_r(s) + 0.05·ū(s)) / ū(s). Here t_r is Smogon's teammate share and u the usage. The team then
  hangs together, where F's rule averages over the seen species alone.
- **Data.** `gen3_teammate_priors.json`, the one species × species joint Smogon publishes. It is allowed by the rule
  today.
- **Stage 0.** Species hit 0.149 → 0.211, exact-set mass 0.020 → 0.045, and the BEST mean log-likelihood of the truth
  of any open-population candidate (−5.69 against F's −6.43; §6.1).
- **Expected lift.** Small. The network's far sharper species belief (0.31 exact-set mass) bought at most +0.008 (FN −
  F at TRUNC-4-K8, a boundary result) and nothing at LEAF-K8. Species alone do not pay while the sets are drawn
  independently. Smogon publishes no move-pair joint, so (b) cannot fix M.
- **Cost.** A weight function swapped in `world_draws.never_seen_species`. It is the cheapest rule-safe improvement, and
  the honest BASELINE any corpus model must beat.

### (e) Set-level inference from observed facts: rank 4 (a CORRECTNESS requirement, not a lift lever)

- **Method.**
  - **First, the cheapest fact and the one measured broken: the set must be LEGAL. BUILT 2026-10-19 (F2):** every set
    is drawn from the prior conditioned on Showdown's validator accepting it, by rejection through the validator itself
    (no table; §7 F2), and every returned world is re-validated. Before, 5.2 % of worlds were illegal. The impossible
    combinations' mass went back to possible ones (Tyranitar Pursuit 0.22 → 0.16 of its sets).
  - Then §3.3's dice-marginalised check and conservative propagation over damage, speed, item and ability events.
- **What it buys.** For the SEEN mons' items, abilities and spreads, today's screen already performs this inference
  (sharper than honest, F1). Making it honest WIDENS the posterior (value small: I / A / S ≈ 0.003 alone, but +0.019
  once U is known, `search_ledger.md` §5.2). It also removes most of the 10 % exclusions. For MOVES, facts say little:
  an unused move leaves no trace, except Struggle / PP, a Choice lock, and Encore or Disable naming a move.
- **Cost.** The constrained-dice replay is a Rust core build (several agent-days, UNVERIFIED). It is required before any
  LIVE use (no leak, no exclusion), and not before stage 1.

### (d) The network's own belief heads: rank 5 (measured; why it is worth ≈ 0.01)

- **What it is.** FN: never-seen species drawn ∝ the X5 species presence, seen mons' moves ∝ the network's per-mon move
  presence. The species presence is a marginal, drawn sequentially without conditioning on the drawn mates.
- **What stage 0 shows (§6.1).**
  - Its species belief is SHARP on these banks, and it MATURES: the exact never-seen species set gets 0.21 of its mass
    at 25M, 0.29 at 100M, 0.30 at 175M and 0.31 at 225M; its recall@h is 0.67.
  - Its move belief is not: every seen mon's hidden moves are right in 0.022–0.025 of its worlds at every anchor
    (F 0.011). The 15M read agrees: a world's hidden moves right 0.513 vs Smogon's 0.508.
- **Why it is worth ≈ 0.01 of the true world anyway** (each a measured link, the mechanism UNVERIFIED as a whole):
  1. **The unit.** TODAY is weight on the EXACT true world in value units. FN's worlds are exact on the species often,
     but on the seen mons' moves and the never-seen mons' sets almost never, so their exact-world mass is near 0
     (§6.1 q_SM).
  2. **Partial correctness is worth little.** A random honest world given the true world's weight is worth ≈ 0 (M5W);
     the network's species-right worlds bought at most +0.008 to +0.017 at any scorer (at 15M).
  3. **U is not sufficient.** Even TRUE never-seen mons with their true sets close only 0.31–0.40 of G (s_U), while
     hiding only the moves opens 0.42–0.58 (n_M).
  4. **It is a closed-pool belief.** The banks' opponents are the pool teams X26 trained against for 230M steps. Off the
     pool, the pre-era species head was WORSE than the Smogon prior (−1.28 nats on ladder teams,
     `design_diversity_teams_opponents.md`). Its sharpness here may be memorisation of 719 teams. Whether X26's head
     transfers off the pool is UNVERIFIED.
- **Lift if fixed.** A joint team head or a set head would need a training change. The knock-in forks say information is
  not a policy lever at this maturity, so this path is ranked last for SEARCH. As a SOURCE of species inside (c) or (f)
  (network species, corpus-consistent sets), its in-pool sharpness would read optimistic: measurement only.

### (a) Today's F: the BASELINE

- **Method.** Species ∝ the mean Smogon teammate share of the seen species (usage when it is zero); every mon's moves by
  the conditional-Poisson presence draw from Smogon's per-species marginals; never-seen items, abilities and spreads
  from Smogon; the seen mons' items, abilities and spreads by the screen.
- **Numbers.** Stage 0 reproduces what F's stored worlds actually held (§6.1 GIGO), so it is the measured baseline:
  - species hit 0.149, exact never-seen species set 0.020;
  - one seen mon's hidden moves all right 0.123, every seen mon's 0.011;
  - exact-world mass < 0.0001;
  - 5.2 % of its stored worlds are illegal (F2). F at HEAD (since 2026-10-19) is legal: 0 illegal worlds.
- **Cost.** About 4 CPU-s per state to build (network-free; the maturity battery's cost table), more when the screen is
  hard (§3.5).

---

## 5. THE SEARCH FORMULATION: a one-turn simultaneous-move matrix, in its hidden-information form

**What changes, and why.** Today's search is a best response to ONE modelled reply: the opponent's root choice is the
network's greedy action in each world. The bots now at the top of Showdown (mikumiku37, Wolfe_Alt6767; Jaxcalibur in
randbats; every figure self-reported) solve the turn as a SIMULTANEOUS-move game instead:
- top-k actions per side;
- each cell averaged over sampled worlds and dice;
- a mixed strategy by regret matching; Jaxcalibur uses pUCT with regret-mixed edge sampling
  (`showdown_bots_landscape_2026-10-18.md` §2, §3).

`design_value_discrimination.md` §14.4 recorded the idea and its soundness limits on 2026-10-11. This section is its
gen-3 form and its place in the harness. It is DESIGN ONLY: the owner's one-ply scope rule holds, and nothing here is
wired into training or play.

**The formulation (a one-turn Bayesian game: each world is an opponent TYPE).**
- Worlds w_1 … w_J, with weights π_j (uniform today; §3.6's likelihoods later).
- Our rows A: every legal action, at most 9 in gen-3 singles (four moves, five switches). The pilot found value in
  low-ranked moves (`search_ledger.md` §2 W4), so our side gets no top-k cut.
- The opponent's columns B_j, PER WORLD: its top-k_B actions under the network's policy on the OPPONENT's own root
  observation in world j, plus its greedy action. They differ by world: different moves, different switch-ins.
- Cells: V_j(a, b) = the mean over R dice seeds of the leaf after the FORCED joint first turn (a, b). The leaf is the
  critic after that one turn, or TRUNC-4.
- **Solve** max_σ Σ_j π_j · min_{b ∈ B_j} Σ_a σ_a V_j(a, b): one σ for us, because we cannot tell the worlds apart, and a
  best response per world for the opponent, who knows its own team.
  - It is a small linear program: k_A + J variables and Σ_j |B_j| constraints. An exact solve takes microseconds.
  - Equivalently, regret matching with our regrets on the π-weighted payoff and one opponent regret vector per world.
  - Solving each world's matrix separately and averaging σ is the strategy-fusion error of Foul Play's 2019 "nash" mode;
    averaging cells over worlds under a SHARED column label is valid only when every world has the same opponent
    actions (open team sheets).
- **Special cases:** k_B = 1 (the greedy reply) is today's search. "π_j on one world" is the oracle matrix.

**Interaction with the sampler.** In the Bayesian matrix the opponent best-responds in EVERY sampled world. A wrong world
that gives the opponent a counter it does not hold makes σ over-cautious. So the matrix NEEDS a better sampler more than
the best response does: today's best response only averages over the worlds. Two mitigations:
- weighted worlds (§3.6);
- a magnet: σ regularised toward the policy prior (`design_value_discrimination.md` §14.3, MMD).

**Where it fits in the harness, and what changes** (`src/main/value_search/`, offline only):

| piece | today | the change |
|---|---|---|
| `search.py` `search_decision` | uniform or halving allocation over OUR actions; one greedy opponent reply per playout | a new `allocation="matrix"` (k_B, R, leaf), returning σ, its value and the per-world opponent replies |
| `playout.py` / Rust `rust_env_playout_open` | `actions` = our first actions only; the opponent's root action is the forward's greedy choice | **a FORCED opponent root action** per request (`[a, b]` pairs), a small Rust FFI change. A depth-1 version needs NO Rust change: `utils.rust_env.successors.Successors.expand_many` already takes `{node_id, p1_action, p2_action, seed}` arms with rows |
| the opponent's root row per world | not computed | the world replay emits the OPPONENT's obs row at the root (`core_events --obs` for the other side), so its top-k_B is computed in its own world |
| `rules.py` | Gumbel / argmax picks, completed-Q targets | `solve_bayes_matrix(V, π)` (an LP plus regret matching, pure numpy), deterministic |
| the referee | IMP = T(pick) − T(top-1) against the TRUE world's GREEDY continuation | keep it (comparable). ADD a best-response referee: the true world's own matrix (every legal opponent action × truth seeds 16–39), scoring σ by its worst case against the true opponent's best reply (IMP-BR). ADD games: head-to-head against the raw policy, and `main.best_response_gap` against an exploiter |

🚨 **A hazard for whoever builds it: never append the joint first turn to the world's LOG and replay it.** The prefix
replays on the TRUE seed, so that turn would resolve on the realized dice. That is the `seed="original"` clairvoyance
(`utils/bridge/search_session.py`'s warning). The forced action must be a branch with a fresh seed.

**What to expect.** Against the scripted bots and the greedy referee a mixed strategy can only LOSE a little: a best
response to a fixed opponent is already optimal (`design_value_discrimination.md` §14.4). The matrix pays against
ADAPTIVE opponents: people, exploiters, the ladder. So its read is IMP-BR and the exploitability meters, not IMP. It
ranks AFTER the sampler: with worlds worth 1 % of the truth, a matrix over them solves the wrong game.

**Cost.** About 1 agent-day (depth-1 on `expand_many`, no Rust change) to 2 agent-days (the forced-opponent playout
API) for the harness. Per decision, about 4× today's TRUNC-4 search at k_B = 4 and R = 1 (§14.4's estimate: ≈ 550 rows
against ≈ 260).

---

## 6. THE FIRST EXPERIMENT (CPU only)

### 6.1 Stage 0: the zero-playout screen (READ 2026-10-19; post hoc, descriptive)

**What.** `measurements/world_sampler_screen_2026-10-19/screen_samplers.py` scores each sampler's truth weight on the
existing banks. It reads only INPUTS of the studies that own them, never a playout outcome or an IMP:
- each state's revealed opponent information;
- the true opponent team;
- the stored F and FN world records;
- the network's belief rows.

**States.**
- The maturity battery's INTERIM bank, LEAN 320 ids only, X26 at 230M–234M, half roster bots and half X26's own
  snapshots. The bank's FULL remainder is being built by the running plateau step and is NOT read.
- E2's two banks: end2 and prod at 15M, own-snapshot games only.
- The 1,395 states with an OK F world in all.

All three banks' opponents are TRAINING-POOL teams.

**Cost.** About 5 CPU-min per pass (7–11 wall-min under `nice -n 19 taskset -c 5-7,13-15`, `CUDA_VISIBLE_DEVICES=""`, one
process), plus ≈ 3 wall-min for the validator check (12,555 teams). The bulk rows are in `result_screen.rows.jsonl.gz`.

**GIGO first (all pass).**
- Re-implemented exactly, F's rule reproduces what F's STORED worlds held: species hit 0.149 analytic vs 0.149 stored,
  on 1,161 states.
- The network's rule does too: 0.480 / 0.485, 0.550 / 0.548, 0.559 / 0.559 and 0.565 / 0.575 at the four anchors.
- 0 states with a seen species missing from the true team; 0 never-seen species outside Smogon's 216-species pool.

**Results** (pooled; mean [95 %, battle-clustered bootstrap, 2,000 draws, seed 20261019]). Species meters are on the
1,161 states with a never-seen mon (k < 6). The network's are on the 234 maturity LEAN states with a belief row.

*U, the never-seen species set:*

| sampler | hit (share of slots with a true species) | p_exact(U\*) | mean log p(U\*) | recall@h | a consistent corpus team holds U\* |
|---|---|---|---|---|---|
| (a) F, today | 0.149 [0.144, 0.154] | 0.020 [0.017, 0.022] | −6.43 | 0.35 | — |
| (b) joint Smogon teammates | 0.211 [0.202, 0.218] | 0.045 [0.040, 0.051] | **−5.69** | 0.37 | — |
| (c) forum corpus | 0.250 [0.239, 0.261] | 0.068 [0.059, 0.077] | −6.55 | 0.39 | 16 % |
| (c) forum, STRICT (no near copy of the true team) | 0.244 [0.234, 0.254] | 0.062 [0.054, 0.070] | −6.68 | 0.38 | 14 % |
| (c) ladder corpus | 0.305 [0.290, 0.319] | 0.125 [0.109, 0.140] | −5.99 | 0.44 | 38 % |
| (d) the network, X26 25M | 0.480 [0.451, 0.508] | 0.212 [0.173, 0.252] | −3.82 | 0.62 | — |
| (d) the network, X26 100M / 175M | 0.550 / 0.559 | 0.288 / 0.295 | −3.36 / −3.29 | 0.66 / 0.65 | — |
| (d) the network, X26 225M | **0.565** [0.533, 0.598] | **0.307** [0.260, 0.355] | **−3.27** | **0.67** | — |

*M, the seen mons' unrevealed moves (states with at least one seen mon holding a hidden slot):*

| sampler | P(one seen mon's hidden moves all right) | P(EVERY seen mon's hidden moves right) |
|---|---|---|
| (a) F, today (its stored worlds) | 0.123 [0.116, 0.130] | 0.011 [0.007, 0.015] |
| (d) the network, 225M (its stored FN worlds) | — | 0.024 [0.010, 0.039] |
| (c) forum corpus, team-joint | — | 0.035 [0.029, 0.041] |
| (c) ladder corpus, team-joint | — | 0.016 [0.012, 0.020] |
| per-species SET prior, forum (the sets of that species holding the revealed moves; λ = n / (n + 1) over F) | **0.256** [0.247, 0.264] | 0.027 [0.022, 0.032] |
| per-species SET prior, ladder | 0.136 [0.130, 0.142] | 0.011 [0.008, 0.014] |

The forum's per-species sets are optimistic on these banks: the pool's own sets recur in other forum teams (F3).

*The exact-world mass q_SM (U's species AND every seen mon's hidden moves exact):*

| sampler | mean q_SM | share of states with q_SM ≥ 0.2 |
|---|---|---|
| (a) F | < 0.0001 | 0 |
| (b) | 0.0001 | 0 |
| (c) forum / forum STRICT / ladder | 0.0023 / 0.0005 / 0.0024 | ≤ 0.5 % |
| (d) the network, 225M | 0.0005 [0, 0.0013] | 0 |
| the best combination in hand: the network's species (225M) × the forum's per-species sets (independence assumed) | 0.0017 [0.0010, 0.0027] | — |

**What exact worlds would take.** A typical state has about two seen mons with hidden slots (F: 0.123 for one, 0.011
for all, so ≈ 2.2 mons). With the network's 0.31 on the species, q_SM = 0.2 needs every seen mon's hidden moves right
about 65 % of the time, about 0.8 per mon. The best honest source in hand gives 0.26 per mon.

*Legality of today's worlds (F2, `validate_worlds.py`, `result_validate.json`).* Showdown's `TeamValidator('gen3ou')`
was run on F's stored world teams:
- **581 of 11,160 worlds (5.2 %) are ILLEGAL**, and 445 of the 1,395 states (32 %) hold at least one.
- The control passes: the TRUE teams, 0 of 1,395 invalid.
- The causes are move COMBINATIONS that per-slot draws cannot see:
  - egg-move combinations no father can pass on (Tyranitar Pursuit + Dragon Dance, Skarmory Whirlwind + Drill Peck,
    Blissey Aromatherapy + Heal Bell);
  - event-move incompatibilities (Blissey Wish + Aromatherapy, Shedinja Slash + Baton Pass);
  - a few Hidden Power IV and EV errors (28 and 3 messages).

**What stage 0 says** (descriptive, post hoc; the ranking it implies is §4's):
1. **The network already identifies the never-seen species, and that identification MATURED while its value did not.**
   Its exact-set mass rose from 0.21 at 25M to 0.31 at 225M (separate intervals [0.17, 0.25] and [0.26, 0.36]), on
   the same bank where the registered TODAY stayed at 0.00–0.01 and BELIEF-RISES read NOT DETECTED
   (`design_hidden_info_binding.md` §12.13.7). A sharper species guess is therefore not what search lacks. Caveat
   (F4): these are the pool teams X26 trained against.
2. **The MOVES are the bottleneck of the exact world.** No candidate gets every seen mon's hidden moves right more than
   3.5 % of the time. Real teams' per-species SETS double the one-mon rate (0.12 → 0.26), and that is still a third of
   the ≈ 0.8 an exact world needs. With the network's species, a world exact on U and M is drawn ≈ 0.05 % of the time
   (0.17 % with the forum's sets), about 100–400× short of 0.2.
   - **No honest source in hand reaches the exact-world target.** This is the format difference the doubles bots do
     not face (open team sheets hand them the moves).
   - It is a statement in the dose curve's EXACT-world unit. Whether partly right worlds carry the value is stage 1's
     question; nothing here measures value.
3. **Open-population team models help the species, not the sets.** The ladder corpus holds the exact never-seen species
   set for 38 % of states and raises p_exact six-fold over F. Its team-joint sets do not raise the moves' exactness
   (0.016). The forum corpus, with near copies of the truth removed, covers 14 %.
4. **The corpus backoff as specified is over-confident.** With λ = n / (n + 1), (c) wins on mean p_exact but loses to
   (b) on mean log p: when the corpus misses, the truth keeps only (1 − λ) of (b)'s mass. Stage 1 fixes λ on a held-out
   population before any read (§6.2).
5. **One in three states carries an illegal world.** Impossible sets in the worlds are a determinization defect,
   independent of which sampler is used (§3.2, F2). FIXED 2026-10-19 (§7 F2): the read below.

*After the fix (F2, `measurements/world_legality_f2_2026-10-19/`, `result_f2.json`; world BUILDS only, no playout).*
F was rebuilt at HEAD with the rule (`on`) on a uniform random 316 of the 1,520 stage-0 states (e2_end2 138, e2_prod
105, mat_lean 73; the contended box stopped the rest). Without the rule (`off`), it was rebuilt on the 113 states
where the rule rejected any draw, plus controls.

| | stored F (the studies' pins) | F at HEAD, `on` |
|---|---|---|
| illegal worlds (Showdown's validator) | 130 / 2,288 (5.7 %) | **0 / 2,272** |
| states holding an illegal world | 95 / 316 (30 %) | **0** |
| excluded states | 30 (9.5 %) | 32 (10.1 %) |
| builder defects (a guard raised) | 0 | 0 |

- **The rule's own effect on exclusions is nil.** On the 113 touched states, `on` excludes 5 and `off` 6: 0 newly
  excluded, 1 rescued. (The rejected draws spent rng, so that state's later tries differed.) The stored-vs-HEAD gap
  of 30 → 32 is the OTHER builder change since the pins: the one Hidden Power fill of 2026-10-19
  (`design_information_levers.md`, Decision record). `off` and the stored rows agree on 128 / 132 exclusion
  verdicts, and on 0 / 132 world texts: every world holding a Hidden Power moved by an IV point.
- **The rule spends no randomness until it rejects.** 19 / 19 control states (no rejected draw) rebuilt `off` are
  byte-identical to `on`. Where `off` was rebuilt, 127 of its 856 worlds are illegal, so `on` changed what it had to.
- **Cost.** 193 rejected draws over 316 states (`illegal_set`). Each validation is a few milliseconds through one
  persistent Node child, memoised per set. The per-state build cost with and without the rule was not compared under
  equal load (**UNVERIFIED**: expected negligible beside the screen's replays).
- **The world distribution.** The species marginal is unmoved (total variation 0.0028 over 13,248 sets on the 276
  states OK in both). The pairs the validator forbids are gone:
  - Tyranitar Dragon Dance + Pursuit: 90 / 1,177 sets → 0 / 1,172;
  - Blissey Wish + Aromatherapy: 7 → 0;
  - Skarmory Whirlwind + Drill Peck: 5 → 0;
  - Gengar Will-O-Wisp + Perish Song: 6 → 0;
  - Salamence Dragon Dance + Wish: 9 → 0.

  Their mass went to legal moves: Tyranitar's Pursuit rate fell from 0.222 to 0.163 of its sets. The other large move
  shifts are Hidden Power TYPES on small-n species (Regice, n ≈ 60), confounded with the Hidden Power fill change.
- **Worlds built before 2026-10-19 are NOT comparable with worlds built after**, for both reasons: the legality rule
  and the Hidden Power fill. Every stored honest read (Tier 1, E2, the maturity battery at its pins) used the
  illegal-capable builder.

**Not read here, and why.** The PLATEAU bank's states are not read: they belong to the plateau step's pending
registered read (`design_hidden_info_binding.md` §12.13.7), so reading even their inputs before it would muddy it.

### 6.2 Stage 1: the sampler ablation with playouts (DESIGNED, ready to register; not run)

**Question.** Does any candidate sampler put enough weight on the truth to make honest search pay, and does stage 0's
exact-world mass predict it?

**Bank and anchor.**
- The maturity interim LEAN states that F could build (287, the series' common states) at **X26 225M**. That is the
  last LEAN anchor, its O / F / FN rows are banked, at the pin `9c033489`, and it is network-free for every new arm.
- Stage 1b replicates on E2's 1,108 OK states only if stage 1a meets its trigger.

**Arms** (each 8 worlds, every legal action, LEAF-K8 / TRUNC-4-K8 / CRITIC, world j on dice seed j as F, the same
referee rows):

| arm | worlds |
|---|---|
| **F_legal** (the baseline of record) | F at HEAD (F2 BUILT 2026-10-19: every set redrawn until Showdown's validator accepts it). F_legal − F (F's banked worlds) is itself a read: what the illegal worlds cost IMP, confounded with the Hidden Power fill change of the same day unless F is also rebuilt at HEAD without the rule (`world_legality_f2_2026-10-19/rebuild_worlds.py --mode off`) |
| **B** | (b): never-seen species by joint Smogon teammates; everything else F's |
| **C_forum_strict** | (c) on `w_forum_v1` ABC, minus every team sharing ≥ 4 mon sets with the state's true team; backoff to B |
| **C_ladder** | (c) on the `hl_05_26` ladder corpus; backoff to B |
| **C_pool** (calibration) | POOL itself on these states: the closed-population CEILING of team identification (Tier 1's arm, r 0.80 at 15M) |
| **NS** (the best combination in hand; optimistic, F4) | never-seen species from the network's 225M belief (FN's rule); every mon's moves from the forum's per-species sets, conditioned on the revealed moves, with F's per-mon rate as the backoff; everything else F's |
| F, FN, O | banked; not re-run |

**The corpus backoff is fixed BEFORE the read, on a population that is not the bank.** Stage 0 found λ = n / (n + 1)
over-confident (§6.1 point 4). Stage 1 uses λ = n / (n + α), with α the maximiser of the never-seen species set's
log-likelihood under a leave-one-team-out fit on the LADDER corpus with simulated reveals: k uniform in 1–5 seen mons,
each with a uniform random subset of its moves revealed. The fitted α, the fit's script and its sha are recorded in the
registration. No bank state enters the fit.

The seen mons' item, ability and spread come from F's own screen in every arm (common random numbers, and the corpus
never has to fit damage). **Every new arm's worlds pass Showdown's validator** (an illegal draw is redrawn, counted):
corpus sets are legal by construction and per-slot draws are not, so without this filter part of a corpus arm's lift
would be F2's defect. Every world passes F's gate, the world guard, and the leakage check (no C_forum_strict world
shares ≥ 4 mon sets with the true team). An arm-state with no world is an exclusion, counted, and read on the common
states.

**Registered reads** (rule 8: a bound within ±0.0005 of its boundary is not clear; battle-clustered bootstrap, 2,000
draws, seed 20261019):
- **Δ_X = IMP_X − IMP_F_legal, at LEAF-K8 AND TRUNC-4-K8** (primary).
  - X **PAYS** iff the lower 95 % bound > +0.0005 at both scorers.
  - **MATERIAL** iff it PAYS and r_X = Δ_X / G_F_legal has a lower bound ≥ 0.25.
  - **NULL** iff the whole 95 % interval of Δ_X lies inside ±0.01 at LEAF-K8.
  - Otherwise NOT DETECTED.
- **SEARCH-ENABLED_X** iff lower95[IMP_X] > +0.0005 at LEAF-K8 (honest search beats the policy) AND lower95[IMP_X −
  IMP_CRITIC,X] > +0.0005 at TRUNC-4-K8 (it beats its own critic search). This is the frontier rule's content, pooled.
- **TODAY_X** on the 225M dose curve, beside q_SM,X from stage 0. The proxy is CALIBRATED iff TODAY_X's interval
  contains q_SM,X for every arm.
- **C_pool − C_forum_strict** = what an open corpus loses against team identification on a closed pool.
- Per opponent stratum (bots / own snapshots), descriptive.

**Power.** Between-arm IMP differences on the same states carry about ±0.010 at 1,200 states (E2's Δs), so about
±0.021 on 287. Stage 1a therefore detects only a search-sized lift: the dose curve's p = 0.2 is worth ≈ +0.03 at
LEAF-K8. A NOT DETECTED at 1a does not exclude a small lift; 1b's pooled ≈ 1,400 states reach ≈ ±0.010.

**Trigger for 1b:** any arm with Δ_X ≥ +0.010 (the point estimate) at either scorer, or any C or NS arm
SEARCH-ENABLED.

**The question stage 0 sharpened: does partial correctness carry value?** Stage 0 says no honest source reaches the
exact-world target. If NS or C_ladder nonetheless reads PAYS while its q_SM stays ≈ 0.002, worlds that are partly
right carry the value. The sampler's target is then decision-relevant correctness, not exactness, and §2's level 3 is
the wrong proxy. If nothing pays except C_pool (team identification), exactness IS the requirement, and the honest path
to search runs through information the format does not give us: the opponent's choices (f), or a population whose
teams are public.

**What each result means:**
- **C_ladder or C_forum_strict MATERIAL:** a team-level model is the path. Next come owner Q1, the dice-marginalised
  check (e), and a ladder-population read.
- **Only C_pool pays:** team identification works but our corpora do not cover the opponents. The lever is coverage or
  (f).
- **Nothing pays, C_pool included at 225M:** information is not the limit at this leaf. Re-read with the matrix (§5) and
  the evaluator.
- **B PAYS:** the Smogon-only joint goes into F as the new baseline.

**Cost (measured unit costs, the maturity pilot's table: F playouts 12.7 CPU-s per state at 225M, F build 3.8 CPU-s):**

| arm | build (CPU-s per state) | playouts | 287 states |
|---|---|---|---|
| F_legal | ≈ 4 (the builder's own validation is ms per set, measured in the F2 rebuild) | ≈ 13 | ≈ 1.5 CPU-h |
| B | ≈ 4 | ≈ 13 | ≈ 1.4 CPU-h |
| C_forum_strict, C_ladder | ≈ 20–60 each (UNVERIFIED: POOL-style gating of up to `MAX_GATED` candidates) | ≈ 13 each | ≈ 5–12 CPU-h |
| C_pool | ≈ 20–60 | ≈ 13 | ≈ 2.5–6 CPU-h |
| NS | ≈ 5 (FN's rule + a set table) | ≈ 13 | ≈ 1.5 CPU-h |
| the α fit on the ladder corpus | — | — | minutes |
| **stage 1a total** | | | **≈ 12–24 CPU-h**, one slot (rule 5b; the maturity queue holds the other two) |
| stage 1b (E2, 1,108 states, end2 + prod pins) | | | ≈ 40–75 CPU-h, conditional |
| stage 1c: (f) re-weighting over 64 proposed worlds | + the opponent-side replay and forwards | resample to 8, ≈ 13 | ≈ 1 agent-day of driver + ≈ 10 CPU-h, after 1a |

**Build needed:** one measurement driver beside `pool_worlds.py` (`load_pool` swapped for a corpus, a backoff to B,
the strict filter), and B's weight function: measurement-folder code. The validator redraw is BUILT into the builder
itself (F2, 2026-10-19, `world_legality`): every arm through `WorldSampler` gets it, and a corpus arm's worlds pass
`world_guard.check_legal`.

---

## 7. HAZARDS AND FINDINGS

- **F1, the true-dice replay over-sharpens the honest posterior: PARTLY RESOLVED 2026-10-19 (tier A BUILT, default
  OFF; §3.3.1).** The prefix replays on the TRUE seed, so a world had to reproduce the observed damage with the truth's
  own roll. MEASURED: today's screen rejects 9.0 % of its candidates that fit on another roll (2.7× the acceptances, 37
  of 40 random states). Tier A (`WorldSampler(dice="rolls")`, the Rust core's damage-roll control) replaces only the
  roll's VALUE, the stream aligned, weight Π k / 16: exclusions 20 → 6 of 60 rebuilt states, cost halved, fact
  marginals barely moved. **What remains:**
  - **Tier B is not built:** the crit roll, speed ties, a dice item / ability that did not trigger and the pre-root
    counters are still the truth's dice.
  - **The default is still `seed`.** Switching it changes every honest read and is the next study's call (stage 1).
  - **Search does not read the weights yet.** It averages its worlds uniformly; the worlds are draws from the prior
    restricted to the consistent set, so the posterior needs the weights (mild here: ESS 7.3 of 8).
  - **The choice inside the consistent set prefers the stream's roll**, so a matched world carries the truth's hidden
    exact HP after a site whenever the truth's roll fits it (the same class as today's true-dice leak). The weight is
    the likelihood along the path taken, not marginalised over the set's other rolls.
  - **A pinned site's factor (1 / 16) is a lower bound**, and pinning on a world's hidden Focus Band is world-dependent
    (no site was pinned on the 60 states).
  - **One state built under `seed` failed under `rolls`** (a Salamence screen chain, 200 steps). Its every
    seed-accepted candidate is also accepted under `rolls` (checked by shadow): the chain's path differs, a
    robustness gap of the chain, not of the control.
  - **A world rebuilt from its TEAM alone loses its rolls.** Every consumer must carry `Worlds.rolls` / the log's
    `"rolls"` (the playout tool and the info audit do).
- **F2, the honest builder made ILLEGAL worlds: RESOLVED 2026-10-19** (`main.value_search.world_legality`; the read in
  §6.1, after the fix).
  - **Before:** worlds passed the format clauses and the engine-runnable checks, never Showdown's
    `TeamValidator('gen3ou')`. **581 of 11,160 stored F worlds (5.2 %) failed it, in 445 of 1,395 states (32 %).**
    The true teams: 0 of 1,395. The causes were egg- and event-move COMBINATIONS that independent per-slot draws
    cannot see (all but 31 of the 587 failing set instances), plus 28 with a stale Hidden Power type field (below)
    and 3 with 0 EVs (a rule the validator holds; such draws are now redrawn too).
  - **The method: rejection through the validator itself, not a derived table.** A table would be a second copy of
    the breeding / event rules that can drift. Rejection is EXACT for the conditional draw (the prior restricted to
    legal sets and renormalised) and costs milliseconds per set. Details:
    - every never-seen mon's whole set is redrawn until accepted (`SET_REDRAWS` 25; its lever coins flipped once);
    - a seen mon's unrevealed moves are redrawn after the screen (`MOVE_REDRAWS` 25);
    - a never-seen set the gender placement made illegal, and a world team the validator rejects, are redrawn;
    - the throwing guard re-validates every returned world (`world_guard.check_legal`);
    - the audit's `legality` check counts `showdown_invalid`.

    The species and the screened spreads are NOT re-weighted by their sets' legality rate: that rate is an artefact
    of the independent draw, not evidence about the species.
  - **The true-team guard (§3.3 point 4):** the TRUTH must pass the same rule, or the build raises `LegalityRuleError`
    before any draw (a defect, never an exclusion). It is tested with a deliberately broken rule.
  - **A second defect found and fixed with it: a STALE Hidden Power type field.** Three pool sets name their type in
    the packed `misc` field (Salamence `Flying`, Salamence `Grass`, Tyranitar `Bug`). A seen mon's world kept the
    truth's field beside its own drawn IVs. Showdown's battle would play the FIELD's type
    (`pokemon.ts`: `hpType = set.hpType || ivs`), while our Rust core plays the IVs'. The core's request names the IVs'
    type, so search was unaffected; only the text and Showdown's verdict were. The entry now drops a field that
    disagrees with its IVs (`world_screen.clear_stale_hp_type`).
  - **After** (316 random states rebuilt): 0 illegal worlds, 0 new exclusions caused by the rule, and the species
    marginal unmoved. Its effect on IMP is UNMEASURED: stage 1's F_legal − F reads it (§6.2).
  - **Not comparable:** worlds built before 2026-10-19 and worlds built after differ (this rule and the same day's
    Hidden Power fill). The studies in flight run on their own pins (the maturity queue `826bee89` / `198fa91e`,
    X37b) and keep the old builder.
  - **A new runtime dependency:** the world builder now needs `node` and the built submodule (the validator), still
    under Node's one runtime job, team validation. A LIVE search would carry it to play time.
- **F3, forum-pool leakage.** 711 of the 719 training-pool teams appear verbatim in the forum collection, and 1,569 of
  its 5,330 teams share all six species with a pool team. `w_forum_v1` removes exact copies only. Any corpus sampler read
  on pool opponents must drop near copies of the TRUE team (stage 0 / 1's STRICT filter) and report the leak.
  - If X38 trains on `w_forum_v1`, every later run's eval opponents include forum teams. A forum sampler evaluated there
    would then hold the truth by construction.
- **F4, the banks are pool-opponent banks.** The network's species belief is sharp on them because X26 trained on those
  teams for 230M steps. Off-pool, the pre-era head was worse than Smogon. No level-1 to level-3 number here speaks for
  the ladder. The ladder read needs states from games against `hl_05_26` or heldout teams; those banks do not exist yet.
- **F5, the exclusions bias every honest read and are fatal live.** 125 of 1,520 bank states (8.2 %), 121 of them a
  seen mon's screen failing. A live search cannot exclude a decision (owner Q2).
- **F6, the hidden counters are the truth's in every world** (the builder's residual). They are optimistic for F and
  every candidate alike, so they are not a bias between samplers.
- **F7, the ladder corpus's never-seen parts are a reconstruction.** Metamon filled what a replay did not reveal, by a
  method UNVERIFIED here, possibly from usage statistics, which would make C_ladder partly a Smogon sampler.
- **F8, read-only use of running artifacts.** Stage 0 read the maturity bank's F world file while the plateau step was
  appending to it. It read the LEAN ids only (all built before the read of 2026-10-19), skipped torn lines, and wrote
  nothing there.
- **The Counter / Mirror Coat / Hidden Power rule (§3.4)** is a named test for any propagation layer.

---

## 8. Questions for the owner

1. **Q1. Is a search-time sampler fitted on real teams a "prior"?**
   - (A) Forbidden: Smogon usage only, so (b) is the ceiling.
   - (B) Allowed at SEARCH time only (never the network's input, `data/` or its belief prior), on a corpus disjoint from
     any training data. Ladder replays are already a named source.
   - (C) Allowed at search time on any public corpus, training data included.

   The doc recommends (B): it keeps the rule's purpose (search never assumes the ladder looks like our training teams)
   and lets team identification be tested. Until answered, (c) is offline measurement only.
2. **Q2. May a LIVE search exclude a decision?** If not, the sampler must always return worlds. That means the
   dice-marginalised check and a fallback to approximate weighted worlds, and the honest builder's ≈ 10 % exclusions
   become a build requirement.
3. **Q3. Is a model of the opponent's choices allowed as evidence (candidate (f))?** Our own network as the opponent
   model is pool-biased experience. A behaviour-cloned human model would need human replay data (Metamon's), a new
   dependency.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-19 | **This doc CREATED** (orchestrator brief after the maturity battery's first LEAN read; the owner added DETERMINIZATION the same day) | One always-current doc for the world sampler and determinization. The TARGET in four levels (the truth's likelihood, recall, the exact-world mass q_SM, and the existing value meters TODAY / G / r); the facts a world must honour and how each is enforced today (rejection on a true-dice replay); PROPOSE → PROPAGATE → VERIFY with dice marginalised and a truth-passes guard; the Counter / Mirror Coat / Hidden Power rule as a named test; six candidates ranked (c) team-level corpus model > (f) opponent-choice likelihood > (b) joint Smogon teammates > (e) fact-based inference (correctness) > (d) the network's belief, against the baseline (a); the one-turn matrix in its Bayesian (opponent-type-per-world) form; stage 0 (zero playouts) READ and stage 1 designed | a species-first programme (stage 0: the network already holds the species; the sets are missing); the doubles bots' shared-column matrix (invalid when the opponent's moves differ by world); Foul Play's per-world Nash average (strategy fusion); pure constraint propagation (re-implementing mechanics is a GIGO class, §3.4); fitting any sampler on the evaluated population | §1's pointers; `measurements/world_sampler_screen_2026-10-19/`; `deps/pokemon-showdown/data/mods/gen3/moves.ts` Counter / Mirror Coat; `design_value_discrimination.md` §14.4; `showdown_bots_landscape_2026-10-18.md` |
| 2026-10-19 | **F2 RESOLVED: the honest builder's worlds are LEGAL** (orchestrator brief; `main.value_search.world_legality`) | Rejection through Showdown's pinned `TeamValidator('gen3ou')` (one persistent Node child, `utils/bridge/validate_stream.js`, memoised): each never-seen set and each seen mon's unrevealed moves are redrawn until accepted, so the draw is the prior conditioned on legality; gender-made and team-level failures are redrawn; the TRUE team must pass the same rule or `LegalityRuleError` raises; `world_guard.check_legal` re-validates every returned world; the audit's `legality` reads `showdown_invalid`; the stale Hidden Power type field of three pool sets is dropped from a seen mon's world. Read on 316 random stage-0 states: 0 / 2,272 illegal worlds (stored 130 / 2,288), 0 new exclusions caused by the rule (5 vs 6 on the 113 touched states), species marginal unmoved (TV 0.0028); the F1 verify step's two tiers written down (§3.3.1), not built | a per-species incompatible-combination table derived from the validator (the doc's first proposal: a second copy of the breeding / event rules that can drift, and only a pairwise approximation of a triple or event rule); re-weighting the species or the screened spread by their sets' legality rate (an artefact of the independent draw); whole-world rejection (the same artefact on the species) | §6.1 (after the fix), §7 F2, §3.3.1; `measurements/world_legality_f2_2026-10-19/` (`result_f2.json`); `world_legality_test.py`, `world_legality_integration_test.py`, `info_audit_integration_test.py` |
| 2026-10-19 | **F1 TIER A BUILT, default OFF** (orchestrator brief; `pokesim::rolls`, `WorldSampler(dice="rolls")`) | MEASURED FIRST by shadowing every gate replay of today's builder with the engine's match control (not an analytic 16-roll check: the engine is the oracle): 9.0 % of the rejected candidates fit on another roll, 2.7× the acceptances. Built: one damage-roll site in the Rust core (draw unchanged; record / match / force; pins where a post-roll modifier could change the sight); the builder's option with the truth-passes guard (`RollMatchError`), per-world `log_w` and `rolls`, the playout tool and the regate replaying the table. Re-measured on 60 states: exclusions 20 → 6, cost 96 → 42 s, marginals barely moved, regate 39 / 39 (§3.3.1) | building tier B now (typed draw sites everywhere; several agent-days); a uniform pick inside the consistent set (would break the truth-passes identity: the truth's own trajectory would not be reproduced); flipping the default (changes every honest read; stage 1's call); feeding the weights into the MH chain (the prior × indicator chain plus world weights is already the correct importance form) | §3.3.1, §3.5, §7 F1; `measurements/world_rolls_f1a_2026-10-19/` (`result_f1a.json`); `src/rust_sim/tests/rolls_test.rs`, `src/rust_env/tests/search_rolls_test.rs`, `world_rolls_integration_test.py` |
