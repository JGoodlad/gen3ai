# THE WORLD SAMPLER AND DETERMINIZATION: putting weight on the opponent's true hidden team

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this doc
updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-21, latest): STAGE 1c READ (§6.3): NEARER BUT NOT PAYING.** The weighted proposals hold slightly more
of the truth's binding hidden facts:
- 128 honest and 128 real-team corpus worlds per state, weighted by the damage-roll weight × the opponent-choice
  likelihood;
- ΔH +0.007 / +0.019 / +0.013;
- SELF-PLAY-ONLY; the exact-world mass ≈ 0.

Honest one-ply search over the resampled 8, with L1's piKL anchor, is NOT-DETECTED against the policy on every arm
(Holm: none pays). With stage 1 (exactness carries the value; only POOL's team identification pays) and L3's off-pool
read (no likelihood evidence against non-self opponents), the opponent-choice route buys no search gain here.
**Since (2026-10-24, checked 2026-10-25):** X50's R-b2 tested a human opponent model as the likelihood's judge on the same off-pool bank and read HOM-NONE (bot AUC 0.459 [0.406, 0.510]; `search_ledger.md` Status), so the route stays closed for search at P\*; search is PAUSED.

**Status (2026-10-21): STAGE 1 READ (§6.2): ONLY TEAM IDENTIFICATION PAYS.** On the 287 maturity LEAN states
at X26 225M, POOL (the training pool's consistent teams) beats today's honest worlds by +0.038 [+0.022, +0.055] per
decision at the 8-rollout leaf and recovers 0.72 [0.53, 0.92] of the oracle gap (PAYS, MATERIAL). Every
open-population sampler is NOT DETECTED against F, with points from −0.008 to +0.002 and intervals near ±0.015. That
covers joint Smogon teammates, a ladder-corpus team model, the forum corpus without near copies, and the network's
species with the forum's sets. The network-species arm makes the truth's facts far likelier and still gains nothing,
so partly right worlds do not carry the value: exactness is the requirement at this leaf. No arm is SEARCH-ENABLED.
The exact-world mass is NOT a calibrated proxy (POOL: 0.82 exact, worth p ≈ 0.40). F2's legality rule and tier A's
dice move IMP undetectably. The 1b trigger fires on POOL alone. PROVISIONAL (pool opponents, 225M).

**Status (2026-10-20): STAGE 1c's PHASE A READ (§6.3).** 128 honest F and 128 C_ladder worlds per state,
weighted by the damage-roll weight × the opponent-choice likelihood, hold worlds NEARER the truth than today's 8 on all
three primaries (ΔH +0.007 / +0.019 / +0.013), but none MATERIAL, all PROVISIONAL-POOL and SELF-PLAY-ONLY (bots
NOT-DETECTED). The corpus brings the never-seen species; the likelihood adds +0.008 among F worlds and nothing among
corpus worlds. The exact-world mass stays ≈ 0. Phase B (playouts) RUNNING.

**Status (2026-10-19): STAGE 1 REGISTERED and RUNNING (§6.2; READ 2026-10-21, above).** The sampler ablation with playouts is
pre-registered (`measurements/world_sampler_stage1_2026-10-19/PREREGISTRATION.md`) on the 287 maturity LEAN states at
X26 225M: B, C_ladder, C_forum_strict, C_pool and NS against F_legal (F at HEAD, F2's rule on), plus F_off (the legality
rule alone) and F_rolls (F1 tier A, read uniform and likelihood-weighted) as secondaries. The corpus backoff's α is
fitted on the ladder corpus (89.1); the playouts run at X26's pin + the damage-roll control, so a rolls world plays as
its gate accepted it. A detached runner crawls one bounded chunk at a time to all 287 states (the 24 h stop was lifted
before any read, A2). No number yet. Search can now read the worlds' weights (`SearchConfig(weights="likelihood")`, default uniform).

**Status (2026-10-19): F1 TIER A BUILT, behind `WorldSampler(dice="rolls")`, default OFF (§3.3.1, §7 F1).**
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
- **The opponent's own choices recognise the truth (L3, 2026-10-20; PROVISIONAL-POOL).** The network, seated as the
  opponent, ranks the true world above an honest one 69 % of the time. That pooled figure is carried by the snapshot
  opponents (80 %, greedy earlier copies of the same network); against scripted bots it is 59 %, WEAK on its own. It
  was read on pool opponents only, so pool familiarity is not yet ruled out (§4 (f)). Today's honest worlds hold
  nothing near the truth for it to pick.
- **So the first target is the moves of the seen Pokémon, kept consistent with the rest of the team**: real teams and
  sets (team-level and set-level models), the legality of move combinations, and the opponent's own choices as
  evidence. A better species guess alone will not do it.
- **Stage 1 (with playouts, READ 2026-10-21, §6.2) answers the open question: partly right worlds do NOT carry the
  value.** Only identifying the opponent's whole team pays: POOL, on a closed pool, recovers 0.72 of the gap. The
  open-population candidates (the Smogon joint, two real-team corpora, the network's species with real sets) read no
  gain over F, although the network-species arm holds the true never-seen species set in 44 % of its worlds and
  makes the truth's facts far likelier. At the 8-rollout leaf exactness is the requirement, and no honest source in
  hand reaches it.

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
| **opponent's choices** | what a player who held world w would have done | **not used by the builder**; scored on request by `main.value_search.opp_likelihood` (OFF by default) | yes (the opponent's own seat; unobserved and root-round choices excluded) | candidate (f), §4. MEASURED by L3 (2026-10-20): the truth out-ranks honest worlds, AUC 0.692 |

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
| 3 | the opponent's own choices are not used as evidence | M, U | L3 (2026-10-20): they identify the truth (AUC 0.692, WORKS), but re-weighting today's 8 worlds moves the moves' match only +0.005; candidate (f) |
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
  - The ladder-replay corpus: `utils.ladder_corpus`, 22,862 `hl_05_26` teams. Their unrevealed parts were filled by
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
- **Expected lift.** It targets M, the class §6.1 shows is missing. **MEASURED by L3 (2026-10-20, zero playouts; see
  the BUILT bullet below):** a strong JUDGE, but no lift on today's proposals.
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
- **BUILT 2026-10-20, OFF by default** (`main.value_search.opp_likelihood`; nothing in the sampler or the search calls
  it). `world_logliks(root, logs, forward)` replays each world's prefix (its damage rolls included) and scores the
  OPPONENT's own rows (its seat: its world team, our team as it saw it) at every past decision whose result the searched
  side saw. Two kinds of decision are excluded and counted: a choice the template rewrote to a SLOT (unobserved: a leak if
  scored), and the opponent's root-round choice when it chose first (pending: decided by replay). Guards raise when a
  world's searched-side rows differ from the truth's (e.g. replayed without its rolls) or a token is not one legal
  choice. A familiarity control (`conf`, the network's confidence in its own top choice) rides along. Measured by L3
  (`measurements/search_l3_2026-10-20/`, pre-registered `4d149799`).
- **L3 READ 2026-10-20 (WORKS; X26 225M as the opponent model; 283 stage-1 states; MEASURED, PROVISIONAL).**
  - **The truth vs today's honest worlds:** the TRUE world out-ranks an honest world (`dice="rolls"`, legality on) with
    AUC 0.692 [0.655, 0.728]. Against X26's own snapshots 0.796, against scripted bots 0.588.
  - **Not familiarity:** the confidence-only control reads 0.481.
  - **The truth's weight is COUNTERFACTUAL, not the dose curve's p:** inserted among the 8 honest worlds, the truth
    would take 0.249 of the likelihood weight (snapshots 0.360; uniform 0.111). This is a different quantity from the
    dose curve's p (`search_ledger.md` §5.1), which is the true world's weight INSIDE the worlds a search plays. No
    honest proposal set holds the truth: its exact-world mass is ≤ 0.0025 per draw (§6.1). p_truth also falls as
    candidates are added: among 64 or 128 proposals it would be far below a quarter unless the truth dominated them
    all. So 0.249 does not say search would reach the dose curve's 0.2–0.35. The quantity that does is the near- or
    exact-world mass after the resample (stage 1c's nearness meters, §6.3). (L3 review F3, 2026-10-20.)
  - **Carried by the snapshot stratum (review F1):** the pooled AUC is the equal-weight average of the snapshot
    stratum (0.796) and the bot stratum (0.588 [0.536, 0.641], WEAK on its own). The snapshot opponents are greedy
    earlier X26 snapshots, so there the opponent model nearly IS the opponent. By L3's own WEAK rule the bot stratum
    alone would not have built 1c (its lift_moves is −0.005 [−0.019, +0.007]). A ladder human is neither, and the bot
    figure is the more defensible prior (UNVERIFIED: no off-pool or human bank exists).
  - **PROVISIONAL-POOL (review F2, F5):** one pre-plateau anchor (225M; X26 plateaued near 280M) and one bank whose
    opponents all play training-pool teams. The confidence control shows the network is not SURER on the truth's rows.
    It does not show that the network does not PREDICT BETTER on a team it trained against.
  - **The familiarity counter-test (READ 2026-10-20, pre-registered `b39c9cdc`; NOT-DETECTED):** a pool DECOY (another
    pool team consistent with every reveal, not a near copy) in the truth's place. On the 82 of 283 states that hold
    one, decoys beat honest worlds clearly (AUC 0.575 [0.517, 0.631]), and the truth beats its decoys only
    inconclusively (0.578 [0.489, 0.664]). About half of L3's separation there is pool-ness (or real-team coherence,
    not separated). Only the snapshot stratum shows the truth above its decoys (0.639); against bots the decoys match
    the truth. So L3 stays PROVISIONAL-POOL, and the off-pool bank is required
    (`measurements/search_l3_2026-10-20/counter_test/`).
  - **No lift on today's proposals:** re-weighting today's 8 honest worlds moves the share of the seen mons' correct
    unrevealed moves only +0.005 [−0.004, +0.013] (+0.014 [+0.005, +0.024] against snapshots); exact move sets and
    never-seen species are not detected.

  **Consequence:** stage 1c pairs this likelihood with a proposal set that can hold near-true worlds (64 F proposals
  resampled to 8, or (c)'s real teams as proposals), never with today's 8 alone. **REGISTERED 2026-10-20 (§6.3).**

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

### 6.2 Stage 1: the sampler ablation with playouts (REGISTERED 2026-10-19; READ 2026-10-21)

**Registered** in `measurements/world_sampler_stage1_2026-10-19/PREREGISTRATION.md`, which fixes what this section
leaves open and adds the day's builds. In short:
- **F_legal** (F at HEAD) is the baseline of record. **F_off** (HEAD with the legality rule off) separates the rule
  from the Hidden Power fill, so F_legal − F_off is the legality rule's own effect. **F_rolls** reads tier A,
  uniform and likelihood-weighted. All three are SECONDARY.
- **α = 89.1**, fitted leave-one-team-out on the ladder corpus. NS keeps stage 0's λ_s = n_s / (n_s + 1).
- A seen mon's Hidden Power inclusion and type follow F_legal's screen.
- C_pool uses the pool **as X26's pin packed it**: the Hidden Power refactor re-packs 34 of the 719 teams (§7 F9).
- A fifth verdict, **HURTS**, is added.
- Each arm's TODAY is checked against the exact-world mass of **its own** worlds.
- The read is made once, at `QUEUE_DONE`, on all 287 states. The registered 24 h stop was lifted by amendment A2
  before any outcome was read.
- **Amendment A1 (L2, registered before any outcome was read; SECONDARY).** It asks whether the per-state gain tracks
  each arm's exact-world mass or its CALIBRATION: LL8, the KT-smoothed factorised log-likelihood of the true hidden
  facts under the arm's 8 worlds. It is read within arms (Spearman, pooled) and between arms (with and without
  C_pool), with BOTH / EXACT-ONLY / LL-ONLY / NEITHER. The literature review's point: "only C_pool pays" cannot tell
  the two apart.

**THE READ (2026-10-21).** It ran on all 287 states (248 battles; 0 parked), under the registered rules after the
GIGO audit passed. Amendment A3 corrected a mis-specified leak tooth before any outcome was read: as written it
fired on an honest F world, while the corpus filter's integrity check and the near-copy excess over F are clean.
Data: `measurements/world_sampler_stage1_2026-10-19/` (`result_s1.json`, `gigo_s1.json`). Each Δ is IMP_X − IMP_F_legal
on the states both built. Battle-clustered 95 %, seed 20261019.

| arm | IMP, LEAF-K8 | **Δ LEAF-K8** | Δ TRUNC-4-K8 | verdict | r = Δ / G | q_SM (own worlds) | TODAY (p on the truth) |
|---|---|---|---|---|---|---|---|
| F_legal (baseline; G +0.053 [+0.035, +0.072]) | −0.010 [−0.025, +0.005] | — | — | — | — | 0.000 | — |
| B (joint Smogon teammates) | −0.014 [−0.029, +0.001] | −0.004 [−0.019, +0.011] | −0.007 [−0.020, +0.006] | NOT DETECTED | −0.08 [−0.46, +0.19] | 0.000 | 0.00 [0, 0.05] |
| C_ladder (ladder-corpus teams, α 89) | −0.008 [−0.026, +0.009] | +0.002 [−0.013, +0.018] | −0.001 [−0.017, +0.014] | NOT DETECTED | +0.04 [−0.29, +0.31] | 0.000 | 0.01 [0, 0.19] |
| C_forum_strict (forum teams, no near copy) | −0.018 [−0.034, −0.003] | −0.008 [−0.023, +0.007] | −0.003 [−0.018, +0.012] | NOT DETECTED | −0.15 [−0.54, +0.12] | 0.000 | 0.00 [0, 0.03] |
| NS (network species × forum sets) | −0.014 [−0.031, +0.002] | −0.004 [−0.023, +0.014] | +0.008 [−0.008, +0.024] | NOT DETECTED | −0.08 [−0.54, +0.25] | 0.001 | 0.00 [0, 0.16] |
| **C_pool (POOL, the closed pool)** | **+0.027 [+0.010, +0.045]** | **+0.038 [+0.022, +0.055]** | **+0.023 [+0.006, +0.041]** | **PAYS, MATERIAL** | **0.72 [0.53, 0.92]** | 0.822 | 0.40 [0.27, 0.57] |
| O (the truth, banked) | +0.042 [+0.024, +0.061] | | | | | 1 | 1 |

- **No arm is SEARCH-ENABLED.** POOL's honest search beats the policy at LEAF-K8, but its 4-turn leaf does not clear
  its own critic search: +0.016 [−0.001, +0.034].
- **NOT CALIBRATED.** POOL's TODAY interval [0.27, 0.57] excludes its own exact-world mass, 0.82. Every open arm's TODAY
  contains its q ≈ 0.
- **C_pool − C_forum_strict** is +0.046 [+0.029, +0.063] at LEAF-K8 and +0.026 [+0.008, +0.045] at TRUNC-4. An open
  corpus loses almost all of what team identification on a closed pool buys.
- **Per opponent stratum** (descriptive, Δ LEAF-K8): POOL pays on both, +0.027 [+0.006, +0.050] on the bots and +0.049
  [+0.025, +0.076] on own snapshots. C_ladder: +0.014 [−0.010, +0.040] on snapshots, −0.009 on bots.
- **The 1b trigger fires**, on C_pool alone (Δ ≥ +0.010 at both scorers). No C or NS arm is SEARCH-ENABLED. Stage 1b
  (E2's 1,108 states) is not run by this unit.
- **2026-10-21, orchestrator ruling: stage 1b is NOT run.** C_pool is the closed-population ceiling, and knowing the
  training pool is not available against real opponents; the off-pool L3 read (`82f90eb3`) shows the opponent-choice
  likelihood works only when our network IS the opponent. 1b would sharpen a number no real opponent lets us use.

*Secondary contrasts, the same rule* (none clear):

| contrast | Δ LEAF-K8 | Δ TRUNC-4-K8 | verdict |
|---|---|---|---|
| F_legal − F (every builder change since `826bee89`) | +0.003 [−0.007, +0.013] | +0.012 [+0.001, +0.023] | NOT DETECTED |
| **F_legal − F_off (F2's legality rule alone)** | +0.005 [−0.001, +0.011] | +0.005 [−0.004, +0.014] | NOT DETECTED |
| F_off − F (the Hidden Power fill and the rest) | −0.002 [−0.011, +0.007] | +0.007 [−0.001, +0.015] | NOT DETECTED |
| F_rolls − F_legal (tier A, uniform) | +0.003 [−0.012, +0.018] | −0.007 [−0.019, +0.005] | NOT DETECTED |
| F_rolls_w − F_legal (tier A, likelihood-weighted) | −0.003 [−0.018, +0.011] | −0.010 [−0.021, +0.001] | NOT DETECTED |
| F_rolls_w − F_rolls (the weighting alone) | −0.006 [−0.015, +0.001] | −0.003 [−0.007, +0.001] | NOT DETECTED |

*Amendment A1 (L2), exactness vs calibration:*
- **Within arms the verdict is NEITHER, NOT SEPARATED.** Exactness is identically 0 on every open arm, so its
  correlation is undefined. The pooled correlation of the gain with calibration (LL8) is −0.06, interval [−0.14,
  +0.02].
- **Between arms at LEAF-K8**, with POOL included, the verdict is **EXACT-ONLY** (ρ exact 0.47 [0.13, 0.80]; LL8 0.27
  [−0.14, 0.82]). Without POOL it is **NEITHER** (LL8 −0.17 [−0.83, +0.71]). At TRUNC-4 (descriptive) it is BOTH with
  POOL and NEITHER without (LL8 +0.77 [−0.26, +0.94]).
- **The telling arm is NS.** Its worlds are far better calibrated than F's (+0.31 per hidden fact in LL8; +3.3 nats on
  the never-seen species set), and its gain is −0.004 at LEAF-K8 (+0.008 at TRUNC-4).

*World meters (descriptive):*

| meter | F | NS | C_ladder | C_forum_strict | C_pool |
|---|---|---|---|---|---|
| the never-seen species set exact (k = 6 counts as exact) | 0.20 | 0.44 | 0.24 | 0.22 | 0.83 |
| every seen mon's move set exact | 0.018 | 0.028 | 0.015 | 0.021 | 0.87 |

- **The corpus arms' backoff, λ.**
  - Ladder: mean 0.29; a consistent team exists on 68 % of states; 2.6 corpus worlds per state.
  - Forum strict: 0.14; 53 %; 1.2.
- **F_rolls:** weight ESS 7.4 of 8.
- **Exclusions:** 7 / 287 on every F-screen arm, 4 for F_rolls, 0 for POOL. No defect in any arm.

**Reading.**
1. **Only team identification pays.** POOL recovers 0.72 of the gap at 225M (Tier 1's POOL: 0.80 at 15M), on both
   opponent strata. Every open-population sampler reads NOT DETECTED against F, with points from −0.008 to +0.002.
   None is search-sized (the dose curve's p = 0.2 is worth ≈ +0.03 at this leaf), but none is NULL either: the
   intervals reach about ±0.015.
2. **Partly right worlds do not carry the value** (the question of this section). NS holds the true never-seen species
   set in 44 % of its worlds (F: 20 %), and its facts are far likelier, yet it gains nothing at the 8-rollout leaf.
   By this section's key, **exactness IS the requirement at this leaf.** The honest path runs through the opponent's
   choices (f, §6.3) or a population whose teams are public. L2 agrees within its power: EXACT-ONLY between arms,
   NEITHER where exactness is ≈ 0.
3. **The exact-world mass is not a calibrated proxy of value.** POOL's worlds are exact 82 % of the time but worth
   p ≈ 0.40 on the dose curve. A post-hoc account, DESCRIPTIVE: POOL's exactness is nearly all-or-nothing per state
   (it identifies the team or it does not), whereas the dose curve mixes the truth into EVERY state. Across states,
   a mixture of the concave curve's ends falls below the curve at the average p. **UNVERIFIED** as the whole cause.
4. **Today's two builds are correctness changes, not value levers.** Neither F2's legality rule nor tier A's dice
   moves IMP detectably. The likelihood-weighted read is no better than the uniform one, and tier A's gain is the
   exclusions: 4 against 7.
5. **For search**, honest sampling from open populations does not approach the 0.2–0.3 of weight on the truth that
   the dose curve asks: the best open arm's TODAY is 0.01, upper bound 0.19. Only knowing which team, on a closed
   pool, reaches p ≈ 0.4. PROVISIONAL: pool opponents, X26 225M, the interim bank.

The design as written before the registration:

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
| stage 1c: (f) re-weighting over 64 proposed worlds | + the opponent-side replay and forwards (BUILT: `opp_likelihood`; L3 measured ≈ 1 s per 9 worlds once built) | resample to 8, ≈ 13 | REGISTERED 2026-10-20 (§6.3): 64 F + 64 C_ladder proposals per state, the resample BUILT (`opp_likelihood.resample`) |

**Build needed:** one measurement driver beside `pool_worlds.py` (`load_pool` swapped for a corpus, a backoff to B,
the strict filter), and B's weight function: measurement-folder code. The validator redraw is BUILT into the builder
itself (F2, 2026-10-19, `world_legality`): every arm through `WorldSampler` gets it, and a corpus arm's worlds pass
`world_guard.check_legal`.

### 6.3 Stage 1c: more or team-level proposals, resampled by the opponent's own choices (READ 2026-10-21: NEARER BUT NOT PAYING)

**Registered** in `measurements/search_1c_2026-10-20/PREREGISTRATION.md` (`c9bdc928`; the read script and the facts meter
by sha256), as L3's registered consequence, with **AMENDMENT A** (`a144482b`, §11 there: the orchestrator's ruling on an
independent review, before any registered state was built).
- **The setup.** On stage 1's 287 states at X26 225M, under `dice="rolls"` with the legality rule on, each state gets
  two PROPOSAL SETS on one screen cache:
  - **128 honest F worlds**;
  - **128 C_ladder worlds** (stage 1's corpus sampler, α 89.1, unchanged).

  Each proposal is weighted by its damage-roll weight × the opponent-choice likelihood (L3's, X26 225M as the
  opponent). A search's 8 worlds are drawn by systematic resampling (`opp_likelihood.resample`, BUILT with this unit,
  OFF by default; `SearchConfig(proposals=N)` turns it on in a search).
- **The arms:**
  - F8, the first 8 F worlds uniform: the baseline of record;
  - **F128L, C128L, M256L** (the 128 F, the 128 corpus, and their union, weighted): PRIMARY;
  - the nested 64-world arms, the dice-only F128d / C128d, and C8 (the corpus alone) as controls.
- **The amendment.** Every corpus arm is read STRICT: proposals sharing ≥ 4 whole mon sets with the pool truth are
  weighted 0 (F3).
  - The likelihood's credit is read only on F128L − F128d and C128L − C128d.
  - A permuted-L placebo must not read NEARER, or the read stops.
  - Any positive is PROVISIONAL-POOL (until an off-pool bank and the plateau checkpoint replicate it), and
    SELF-PLAY-ONLY when only the snapshot stratum carries it.
- **Phase A, the rule:** the primary is **ΔH**, the change in the share of the truth's BINDING hidden facts the worlds
  hold (each never-seen species and each seen mon's unrevealed move), against F8. NEARER iff its lower bound > +0.0005;
  MATERIAL > 0.0505; DOSE iff the exact-world mass's lower bound > 0.1005.

**PHASE A READ (2026-10-20; 285 states; zero playouts; MEASURED, PROVISIONAL-POOL).** All gates pass (0 defects, 2
exclusions, 6 / 6 deterministic rechecks, 2,280 / 2,280 resamples recomputed).

| arm | ΔH vs F8 [95 %] | verdict | bot / snapshot | the exact-world mass q_SM |
|---|---|---|---|---|
| F128L | +0.007 [+0.001, +0.014] | NEARER, not MATERIAL | −0.006 / +0.021 | 0.0000 |
| C128L (STRICT) | +0.019 [+0.012, +0.027] | NEARER, not MATERIAL | −0.000 / +0.038 | 0.0001 |
| M256L | +0.013 [+0.007, +0.019] | NEARER, not MATERIAL | −0.003 / +0.030 | 0.0000 |

All three are labelled **PROVISIONAL-POOL, SELF-PLAY-ONLY**.
- **The corpus brings the never-seen species by itself.** C8 − F8 is +0.017 on H and +0.095 on the species share; the
  exact never-seen set rises 0.017 → 0.065.
- **The likelihood's own selection is small:**
  - +0.008 [+0.002, +0.013] among the F proposals (NEARER, snapshot-only; the every-move-exact share 0.011 → 0.016);
  - +0.005 [−0.000, +0.011] among the corpus ones (NOT-DETECTED);
  - against scripted bots it pulls the corpus worlds AWAY from the truth (−0.008).
- **More proposals buy nothing:** 128 − 64 ≈ 0 on every arm. The likelihood keeps an ESS of ≈ 18–20 of 128.
- **A1's leak is small:** 11 % of states hold a near copy, and the likelihood puts 0.0018 of its weight on them. **The
  placebos read NULL.**
- **What it means for the sampler (descriptive):**
  - The judge (L3) finds slightly nearer worlds where the opponent is our own network, and none against bots.
  - The exact-world mass stays ≈ 0 in every arm, two orders of magnitude short of the dose curve's ≈ 0.2.
  - What binds is still proposals that hold the truth's team. The likelihood cannot conjure them from 128–256 honest or
    corpus draws.
  - The bar (any primary NEARER) is met. **Phase B** plays F8, the three primaries, C8, F128d and C128d with L1's
    fitted piKL anchor (RUNNING; stop 2026-10-21 21:00 PDT).

**PHASE B READ (2026-10-21; 285 states × 7 arms; MEASURED, PROVISIONAL-POOL).** The identity tooth passed 12 / 12, and
there were 0 refused worlds; every played world is the registered one. With L1's fitted piKL anchor, IMP over the
policy at LEAF-K8 / TRUNC-4-K8:

| arm | LEAF-K8 | TRUNC-4-K8 |
|---|---|---|
| F8 | +0.001 | −0.002 |
| F128L | −0.001 | −0.001 |
| **C128L** | **+0.007** [−0.003, +0.017] | +0.002 |
| M256L | −0.002 | −0.005 |
| F128d | −0.002 | −0.002 |
| C128d | +0.001 | −0.004 |
| C8 | −0.001 | +0.007 |

- Every played arm is NOT-DETECTED, in each stratum. Holm rejects none, and none beats F8.
- NOT-DETECTED means no search-sized (≈ +0.03) gain; it licenses no re-run on this bank.
- **The unit: NEARER BUT NOT PAYING.**
- **Read with stage 1 (§6.2):**
  - Partly nearer worlds carry no value at this leaf; only exact team identification does.
  - The likelihood's selection exists only against our own network (L3 off-pool: bots NO-EVIDENCE).
  - So (f) with our network as the opponent model is not a search lever here. Its only remaining use is as the weight
    on a proposal set that ALREADY holds the truth's team, for example a behaviour-cloned opponent model with a
    covering corpus. Neither exists for the ladder.

Detail: `measurements/search_1c_2026-10-20/README.md` §5 (phase A), §6 (phase B), §7 (findings);
`result_1c_a.json`, `result_1c_b.json`.

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
  - **The default is still `seed`.** Switching it changes every honest read and was stage 1's call. Stage 1 read no
    IMP effect (F_rolls − F_legal +0.003 [−0.012, +0.018]; weighted −0.003; the weighting alone −0.006 [−0.015,
    +0.001]): tier A is a correctness change (exclusions 7 → 4 of 287), not a value lever.
  - **Search reads the weights only on request** (BUILT 2026-10-19 for stage 1): `SearchConfig(weights="likelihood")`
    averages each action's leaves by the self-normalised `rules.world_weights(Worlds.log_w)`; the default stays
    uniform, and halving refuses it. The worlds are draws from the prior restricted to the consistent set, so the
    posterior needs the weights (mild here: ESS 7.3 of 8). Stage 1 reads F_rolls both ways.
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
    marginal unmoved. **Its effect on IMP (stage 1, §6.2):** F_legal − F_off (the rule alone) +0.005 [−0.001, +0.011]
    at LEAF-K8, NOT DETECTED; every builder change since the banked F together +0.003 [−0.007, +0.013]. The banked
    honest reads did not move detectably for the illegal worlds.
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
  the ladder. The ladder read needs states from games against `hl_05_26` or heldout teams. **The first such bank exists
  (2026-10-20, `measurements/search_offpool_bank_2026-10-20/`):** 320 states of P\* against its plateau roster, every
  opponent on a `heldout72_hpfix` team, proven disjoint from every pool file at X26's pin. On it, the opponent-choice
  likelihood (§4 (f)) replicates against our own snapshots (0.794) and shows nothing against bots (0.472). An off-pool
  ladder decoy gets no lift over F's worlds (0.503), so the pool decoys' lift was pool memory. This doc's level-1 to
  level-3 numbers have not been re-read on it.
- **F5, the exclusions bias every honest read and are fatal live.** 125 of 1,520 bank states (8.2 %), 121 of them a
  seen mon's screen failing. A live search cannot exclude a decision (owner Q2).
- **F6, the hidden counters are the truth's in every world** (the builder's residual). They are optimistic for F and
  every candidate alike, so they are not a bias between samplers.
- **F7, the ladder corpus's never-seen parts are a reconstruction.** Metamon filled what a replay did not reveal, by a
  method UNVERIFIED here, possibly from usage statistics, which would make C_ladder partly a Smogon sampler.
- **F8, read-only use of running artifacts.** Stage 0 read the maturity bank's F world file while the plateau step was
  appending to it. It read the LEAN ids only (all built before the read of 2026-10-19), skipped torn lines, and wrote
  nothing there.
- **F9, Tier 1's POOL at HEAD is not the pool the banks' opponents came from** (found 2026-10-19, building stage 1).
  The Hidden Power refactor (`c9493053b`) and the Ghost fill fix (`f20526571`) changed the IVs the trainer's packing
  path fills for a typed Hidden Power. So HEAD packs 34 of the 719 pool teams with other 30 / 31 patterns than X26's
  pin did (pool sha256 `d8f708e5…` at HEAD against `4083c86b…` at the pins; Ghost 30 Def / 30 SpD against 30 Atk /
  30 SpD). POOL gates a candidate WHOLE, so at HEAD a true team can fail the gate by a stat point and the ceiling
  loses the truth. Stage 1 loads the pin's packing (`pool_pin.py`) and checks every true team is in it. Any later POOL
  read on a pinned bank must do the same.
- **F10, the exact-world mass q_SM is not a calibrated proxy of value** (stage 1, NOT CALIBRATED). POOL's worlds are
  exact 82 % of the time and worth p ≈ 0.40 [0.27, 0.57] on the dose curve. §2's level 3 maps to level 4 only within a
  state; across states that are identified all-or-nothing it overstates value. Post hoc: the dose curve is concave
  (§6.2 reading 3).
- **F11, a world-level leak tooth cannot tell a leak from a good honest draw** (stage 1, amendment A3). "No world
  shares ≥ 4 mon sets with the true team" fires on F's own worlds once most of a team is revealed: the one state it
  fired on had k = 6, no corpus world, and F_legal's identical share. Test a corpus sampler's leakage on the corpus
  teams it draws, and against F's share on the same state.
- **F12, a long CPU measurement meets the heavy-job slots** (stage 1, 2026-10-19 to 21). A slot is held per bounded
  chunk (≤ ~45 min), never for a whole run (orchestrator, 2026-10-20). The runner must import the CURRENT
  `heavy_slots` (the FIFO ticket queue, `96f011dc`), or its waits jump the queue. A detached loop that takes one
  `mem_cap.sh --slot` per chunk (`loop_s1.sh`) does both. Stage 1 also ran BLOCKING for a day by the orchestrator's
  grant.
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
| 2026-10-19 | **STAGE 1 REGISTERED and LAUNCHED** (orchestrator brief; `measurements/world_sampler_stage1_2026-10-19/`) | §6.2's arms on the 287 maturity LEAN states at X26 225M. **F_legal** (F at HEAD) is the baseline of record. The brief's confound and factor arms are SECONDARY: **F_off** (the legality rule off) and **F_rolls** (tier A, uniform and likelihood-weighted through the new `SearchConfig(weights="likelihood")`). α = 89.1 is fitted on the ladder corpus alone. Every arm but C_pool is F's builder with its departures swapped in for one sample, on one shared screen cache: CRN, with a backoff world exactly B's. A seen mon's Hidden Power is the screen's. C_pool uses the pin's pool packing (F9). The playouts run at X26's pin + the cherry-picked roll control, and its identity tooth passed 12 / 12. One heavy job runs at a time; the read is made once, on the done prefix at the stop | stage 0's α = 1 backoff (over-confident, §6.1 point 4); playing at HEAD (the network's obs and τ solve changed since the pin, so arms would not be comparable with the banked F / FN / O); dropping the rolls factor (the default switch is stage 1's call); a mid-run interim verdict (two looks); the forum per-species sets strict-filtered for NS (stage 0's NS configuration kept, flagged optimistic) | `PREREGISTRATION.md`, `alpha_fit.json`; `play_s1.py`'s identity mode (`identity.json` in the archive); `search_weights_test.py` |
| 2026-10-20 | **Candidate (f)'s likelihood BUILT, default OFF** (orchestrator brief, L3 of the literature review; `main.value_search.opp_likelihood`) | A reusable function, not a sampler mode: the opponent's own rows from each world's prefix replay, scored by any forward; unobserved (slot-rewritten) and pending (root-round) choices excluded and counted; throwing guards on the searched side's rows and the tokens; tests that fail on revert of the seat and of the pending rule | scoring the slot-rewritten choices with the world's rendered token (the slot is the truth's: a leak); scoring the searched side's rows (the opponent model must see what the opponent saw); a `SearchConfig` weighting mode now (stage 1c's job, after L3's read) | `measurements/search_l3_2026-10-20/PREREGISTRATION.md` |
| 2026-10-20 | **L3 READ: WORKS** (`measurements/search_l3_2026-10-20/`, pre-registered `4d149799`) | §0, §3.2 and §3.5 rows, §4 (f) (the result and its consequence), §6.2's stage-1c cost row. The opponent's choices JUDGE worlds well (AUC 0.692; 0.796 snapshots, 0.588 bots; not familiarity, control 0.481; the truth's weight 0.249 of 9), but re-weighting today's 8 honest worlds moves the moves' match only +0.005, so stage 1c needs a bigger or team-level proposal set | (f) ranked first now (its value depends on the proposals: it multiplies (c), it does not replace it); stage 1c as a re-weighting of today's 8 worlds; quoting the snapshot stratum as a ladder number | L3 README §2–§4, `result_l3.json` |
| 2026-10-20 | **STAGE 1c REGISTERED; the importance resample BUILT, default OFF** (orchestrator brief, L3's registered consequence; `measurements/search_1c_2026-10-20/`; `opp_likelihood.resample`, `Worlds.select`, `SearchConfig.proposals`) | §6.3: 64 F + 64 C_ladder proposals per state under `dice="rolls"`, weighted by dice × the opponent-choice likelihood, systematic resampling to 8 (each kept world with its own rolls; the weights spent, so a search over them is uniform); phase A's zero-playout nearness read (ΔH primary, NEARER / MATERIAL / DOSE) gates phase B's playouts with L1's fitted piKL anchor | re-weighting today's 8 (L3: nothing near the truth); top-8 by weight (a mode, not a posterior sample); multinomial resampling (more variance than systematic); a playout of every arm regardless of phase A (the brief's registered bar); the corpus arm under `seed` dice (the brief: `rolls` everywhere); a properly weighted mixture for M128L (the components' consistent mass is not estimated: declared) | `measurements/search_1c_2026-10-20/PREREGISTRATION.md`; `opp_likelihood_test.py`, `worlds_test.py`, `search_weights_test.py`, `world_rolls_integration_test.py` |
| 2026-10-20 | **L3 CORRECTED after its adversarial review** (`measurements/search_l3_2026-10-20/REVIEW.md`, F1–F3, F5) | §0 (the L3 bullet) and §4 (f) (the L3 read): p_truth 0.249 is a COUNTERFACTUAL (the truth inserted among 8 honest worlds), not the dose curve's p; the pooled WORKS is carried by the snapshot stratum (bots WEAK alone); the read is PROVISIONAL-POOL, with the familiarity counter-test named | "0.249 is the dose curve's p\* range" (two different quantities; the quantity search needs is the near-world mass after the resample, stage 1c's meters) | REVIEW.md F1–F3, F5 |
| 2026-10-20 | **STAGE 1c AMENDMENT A registered** (the orchestrator's ruling on the independent review, `measurements/search_l3_2026-10-20/REVIEW.md` §3; PREREGISTRATION §11) | §6.3: the STRICT corpus read (`near4` weighted 0; F3's leak class), the L-vs-dice attribution co-primaries, the permuted-L placebo stop, PROVISIONAL-POOL / SELF-PLAY-ONLY, Holm, NOT-DETECTED = no search-sized gain, the ESS split; the runner (not yet past its first slot wait) stopped and relaunched with the same stop | a corpus-team-level near-copy flag in place of the per-world `near4` (it would need a driver change; the ruling's flag is conservative: it also zeroes rare honest near-exact worlds, declared, with the honest base rate reported) | PREREGISTRATION §11 |
| 2026-10-20 | **L3 FAMILIARITY COUNTER-TEST READ: NOT-DETECTED, POOL-LIFT** (`measurements/search_l3_2026-10-20/counter_test/`, pre-registered `b39c9cdc`) | §4 (f): pool decoys beat honest worlds (0.575 [0.517, 0.631]); the truth vs its decoys is inconclusive (0.578 [0.489, 0.664]); L3 stays PROVISIONAL-POOL, the off-pool bank required | treating L3 as truth-specific evidence on pool banks | counter_test README, `result_decoy.json` |
| 2026-10-20 | **STAGE 1c PHASE A READ** (`measurements/search_1c_2026-10-20/` README §5, `result_1c_a.json`) | Status; §6.3 rewritten (it still named the pre-registration draft's 64 + 64 proposals and its arm names; the registered N is 128 + 128) with the read: all primaries NEARER, none MATERIAL, PROVISIONAL-POOL and SELF-PLAY-ONLY; the corpus carries the species, the likelihood +0.008 among F worlds only; the exact-world mass ≈ 0; phase B running | ranking (f) above (c) on this read (the likelihood adds little to the corpus); calling the route closed before phase B's playouts and the off-pool replication | README §5 |
| 2026-10-20 | **THE OFF-POOL BANK EXISTS; L3 off-pool SELF-PLAY-ONLY, the pool decoy lift was MEMORY** (`measurements/search_offpool_bank_2026-10-20/`, pre-registered `7642df0f6`) | F4 (the bank now exists; what it showed for §4 (f)) | quoting a pool-bank likelihood or corpus-arm number as a ladder number; reading the pool decoys' lift as coherence | that folder's README, `result_offpool.json` |
| 2026-10-21 | **STAGE 1 READ: only team identification pays** (`measurements/world_sampler_stage1_2026-10-19/result_s1.json`; registered, amendments A1–A3 before any outcome was read) | §6.2's read on all 287 states. C_pool (POOL) PAYS and is MATERIAL: Δ +0.038 [+0.022, +0.055], r 0.72. B, C_ladder, C_forum_strict and NS are NOT DETECTED, and none is SEARCH-ENABLED. NOT CALIBRATED (F10). L2 reads EXACT-ONLY between arms with POOL and NEITHER without. The secondaries (F2's rule, tier A uniform and weighted) are NOT DETECTED. The 1b trigger fires on POOL alone. Status, §0, §6.2, §7 F1 / F2 / F10–F12 updated | reading partial worlds as the path (NS's far better calibration bought nothing); calling the open arms NULL (their intervals reach ±0.015); flipping the dice default to `rolls` on value (no effect read; it stays a correctness option); quoting q_SM as a value proxy (POOL 0.82 vs TODAY 0.40) | `PREREGISTRATION.md` §9–§11; `gigo_s1.json` (PASS with A3); `read_s1_test.py` (14 planted) |
| 2026-10-21 | **STAGE 1c READ: NEARER BUT NOT PAYING** (`measurements/search_1c_2026-10-20/` README §6) | Status; §6.3 (phase B's table and the unit; read with stage 1 and L3 off-pool) | ranking (f) as a sampler for the ladder (its evidence is self-play-only, and nearer-not-exact worlds do not pay at this leaf); a re-run on this bank (A6); quoting phase A's nearness as value | README §6 |
| 2026-10-25 | **Status line checked against R-b2 (docs-health pass)** | The 2026-10-21 reads stand; the status gained one sentence: the human-opponent-model route to the opponent-choice likelihood was tested 2026-10-24 and read HOM-NONE, and search is PAUSED | — | `search_ledger.md` Status (2026-10-24); `design_x50_rung1b_human_opponent_model.md` §8.4 |
