# WHICH HIDDEN INFORMATION BINDS OUR DECISIONS: an adversarial study of "the opponent's moves are the binding constraint"

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-11): DESIGN + PRE-REGISTRATION (§12, landed `e25a5240` before any arm was run or read); the CHEAP
parts RUN AND READ (§16; Amendment 1 after the audits, §12.12).** The cheap parts: the sampler audits (§5), the model-input
axis on banked states (§7), the registered descriptive reads of the pilot's existing rows (§11), and the maturity
meters (§10.1–§10.2). The playout-world factorial (§6) runs after the closing test frees the box (≈ Mon 2026-10-12
09:00–10:00). The GPU knock-in forks (§8) are QUEUED by the orchestrator (owner-delegated GPU scheduling) after the
closing test's verdict branch and the bf16 strength A/B, ≈ mid-week. **No conclusion of this study may
steer research until the independent replication (§13.1) agrees, the red team (§13.2) has attacked it, and, for
anything bigger than cheap reversible work, the plateau replication (§10.5) has confirmed it.**

Measurement folder: [`../research_state/measurements/hidden_info_binding_2026-10-11/`](../research_state/measurements/hidden_info_binding_2026-10-11/).
The pilot this study interrogates: [`design_value_discrimination.md`](design_value_discrimination.md) §7.10–§7.12.
**Every arm of this study is a preset of the ONE information-lever spec** (2026-10-12,
[`design_information_levers.md`](design_information_levers.md) §5: F = `off`, O = `oracle`, F+X / O−X, the dose,
placebo and MAP families, the model-input arms `TRUE@p` …, the knock-in levers `ki_*`): the training observation, the
search world builder and the model-input injection all read it, and its audit suite runs against all three.

---

## 0. Summary, in plain language

**The claim under test.** The value-discrimination pilot found that a search which plays each candidate action out
to the end in the TRUE battle (it knows the opponent's real team: the "oracle" world) picks a better action than our
policy by about **7 points of win probability per decision**, while the same search in "fair" worlds (opponent
teams guessed from Smogon usage, kept only if they reproduce everything our side has seen) gains **nothing**. The
pilot attributed the gap to the opponent's hidden MOVES. If that is right, the next research path is move inference
(a better belief about the opponent's moves, for the network and for search). That is a multi-week commitment, so
the owner asked for the claim to be reproduced AND attacked: "give high mass to the known moves, but don't
necessarily make them 100% and see if that changes how the model behaves … many tests, many countering tests, and
then still feeling good about it."

**What this doc does.**
1. It states the claim precisely, in three separable parts (§1): WHICH hidden fact carries the gap (attribution),
   whether a better but still honest belief could recover it (recoverability), and whether the NETWORK itself would
   decide better if told (usability).
2. It audits the existing evidence for and against (§2). Two findings reframe the question before any new run:
   - By construction, the pilot's fair worlds kept every SEEN opponent Pokémon's item, ability, nature, EVs and IVs
     TRUE. So the pilot's gap could only ever have come from (a) the unseen moves of seen Pokémon and (b) the
     never-seen Pokémon as a whole. Spreads, items and abilities were never in the comparison. A fully honest world
     has a gap the pilot never measured.
   - Three pieces of prior evidence cut AGAINST "moves bind": a network TRAINED with the opponent's full set
     revealed was only +2.2 pp stronger than without, and the set (moves, item, spread) added nothing over the
     species (−0.26 pp [−1.92, +1.40], the X5 oracle arms); Foul Play, which guesses sets from Smogon usage alone,
     beat our 75M policy, and swapping its better set source for Smogon's changed nothing (+0.000 [−0.15, +0.15],
     n = 80); and the eval opponents play OUR team pool, not Smogon's sets, so a low move-hit rate may partly measure
     how un-Smogon-like the pool is.
3. It lists the RIVAL explanations (§3), each with the result that would kill it: items, abilities, spreads, exact
   HP, hidden counters, the never-seen Pokémon, a sampler bug, statistics, information no honest belief could have,
   the Smogon-vs-pool mismatch, our evaluator rather than our information, search width, and a still-maturing policy.
4. It designs four axes of tests: the sampler AUDITS (§5); the PLAYOUT-WORLD factorial (§6: reveal only X, reveal
   all but X, a dose-response of move knowledge, a wrong-moves placebo, an honest sharpening control, the Bayes
   posterior of our own pool, Foul Play's hand-written evaluator as a second leaf); the MODEL-INPUT axis (§7: hand the
   network's move belief the true moves with 60–100 % mass and see whether its choices and its critic get better);
   and the TRAINING axis (§8, GPU, queued ≈ mid-week: can a network LEARN to use the moves if it is given them?).
5. It pre-registers deterministic decision rules (§12) for "moves are binding", "refuted", "shared", "transient",
   and the other verdicts, with the power each needs, a blinded analysis, an independent re-implementation (§13.1),
   a red team (§13.2), a maturity guard against the "v8 gift" trap and a plateau replication (§10).
6. It says what each verdict would mean for the roadmap and what each path costs (§14), and prices the compute
   (§15): ≈ 15 CPU-hours now; ≈ 270 CPU-hours for the factorial, its replication and maturity arms on the free box
   after the closing test (≈ 28–32 wall-hours, or ≈ 6–8 with a GPU-batched playout harness), ≈ 40 more for the
   plateau replication later; ≈ 14 GPU-hours for the knock-in forks.

**The cheap results so far (§16, read 2026-10-11 after the registration):**
- The pilot's honest search keeps about HALF the oracle's edge when all six opponent Pokémon have been seen and
  NOTHING when one is still unseen: the never-seen Pokémon and the seen ones' hidden moves BOTH carry the gap (§16.1).
- Our pool's sets are as predictable from Smogon usage as real human sets; even sets drawn from the prior itself are
  no easier to guess: the move-guessing difficulty is intrinsic, not a pool artefact (§16.2).
- At 15M both policies are still changing fast and their beliefs still improving: every reading is PROVISIONAL (§16.3).
- The world builder holds every seen Pokémon's hidden state exactly and the post-root dice are clean, but its check is
  weaker than our own battle log in ≈ 4 % of worlds and 2–3 % of its teams break gen-3 OU clauses: R-GIGO FAILS by its
  rule; dropping every affected state leaves the gap unchanged; the fixes gate the factorial (§16.4).
- Told the true moves at 95 % confidence, the network changes one decision in eight and decides NO better (≤ +0.5
  points per decision vs the oracle search's +7): R-NET and CRITIC-USES NOT DETECTED (§16.5).
- **Honest confidence that the moves are THE binding constraint: ≈ 15 %**; that hidden TEAM information binds search:
  ≈ 85 %. No multi-week move-inference programme on this evidence (§16.6).

---

## 1. The claim, stated precisely

### 1.1 Notation (the pilot's, unchanged)

- A STATE is one of the pilot's 600 registered decisions per checkpoint (`rb_es_end2_s2001`, `rb_es_prod_s2001`;
  own eval traces, stratified by the policy's top-2 gap; `value_disc_pilot_2026-10-11/README.md` §2).
- The REFEREE T(a) is the mean win-scale outcome of action a over 24 truth seeds (16–39), played in the TRUE world
  by the checkpoint greedy on both sides. No scorer ever reads a truth seed.
- **IMP_W(scorer)** = the mean over states of T(â) − T(top1): how much better (win probability per decision) the
  scorer's argmax over ALL legal actions is than the policy's own pick, when the scorer's playouts run in world
  kind W. Battle-clustered bootstrap, 2,000 draws, `default_rng(20261011)`, every contrast paired.
- **LEAF-K8**: the mean outcome of 8 playouts to the end per action (8 worlds × one dice seed each in a fair arm; 8
  dice seeds in the oracle). **TRUNC-4-K8**: the critic's value 4 turns later, averaged the same way. **CRITIC**: the
  critic's one-ply value of the successor.
- World kinds: **O** = the oracle (the true battle); **P** = the pilot's Phase 2 fair world (Smogon prior, the
  named leaks); **F** = the FULLY honest world (§6.1, new).

### 1.2 The claim in three parts (they can come apart)

| part | statement | the axis that tests it |
|---|---|---|
| **C1 ATTRIBUTION** | Of the gap G = IMP_O(LEAF-K8) − IMP_F(LEAF-K8), the opponent's hidden MOVES carry the majority, both as the thing that SUFFICES to close it (reveal only the moves) and the thing that is NECESSARY (reveal everything but the moves) | §6 playout-world factorial |
| **C2 RECOVERABILITY** | A better but still HONEST belief about the moves would recover a material part of G (it is belief error, not information no honest player can have) | §6.4 dose, MAP and the pool posterior |
| **C3 USABILITY** | The NETWORK itself (policy and critic) would decide better if its move belief were right: it lacks the information, rather than being unable to use it | §7 model-input axis (frozen); §8 training axis (forks) |

"Moves are THE binding constraint", the owner's phrase, needs C1 and at least one of C2 / C3. C1 alone says the
information matters to search; it does not say any affordable work can buy it back.

### 1.3 Why it is a huge call

The two research paths it would choose between are weeks apart: move inference (belief training signals,
set-level priors, opponent modelling, belief-sampled search) versus the alternatives (a better evaluator, value
labels from oracle playouts (`design_value_discrimination.md` §13.8), information-set search, more width). A wrong
attribution would spend the next research block on the wrong one. Hence the bar: every rival named, each with a
kill test, an independent re-implementation, a red team, and a plateau replication before a multi-week commitment.

---

## 2. The evidence so far, audited

### 2.1 What the pilot measured (both closing-test seeds 2001; `design_value_discrimination.md` §7.11–§7.12)

| read | end2 | prod |
|---|---|---|
| IMP_O LEAF-K8 (oracle) | +0.081 | +0.071 |
| IMP_P LEAF-K8 (Smogon worlds) | +0.007 [−0.008, +0.022] | +0.004 [−0.010, +0.019] |
| R-INFO (oracle − fair, LEAF-K8) | **+0.072 [+0.057, +0.087]** | **+0.067 [+0.053, +0.080]** |
| fair LEAF-K8 with 0–1 opponent slots unrevealed vs oracle | +0.019 vs +0.072 | +0.014 vs +0.062 |
| R-BELIEF (worlds whose never-seen SPECIES come from the network's belief) | +0.001 [−0.014, +0.015] | −0.003 [−0.016, +0.009] |
| a world's re-drawn move slot = the true move / a world's SET holds a true unseen move | 16.6 % / 45 % | 16.0 % / 45 % |

### 2.2 What that evidence can and cannot support (the adversarial audit)

1. **The pilot's fair world was not fully fair, and the leak decides what it could attribute.** `determinize.py`
   kept every SEEN opponent mon's species, item, ability, nature, EVs and IVs TRUE (they shaped the observed
   history, so the byte-equal gate would reject most redraws). In the pilot's world therefore only three things
   differ from the truth: the unseen move slots of seen mons (≈ 10.2 per world: 45,640 slots over 4,464 end2
   worlds, `move_accuracy_read.json`), every fact of the never-seen slots (≈ 2.1 per world), and the counted leaks.
   Consequences: (a) the pilot's gap contains NO spread, item, ability or exact-HP component of a seen mon, so it
   cannot rule those in or out; (b) the fully honest gap G (seen mons' item / ability / spread also guessed) is
   UNMEASURED and may be larger; (c) within the pilot's design the only rival to "seen mons' moves" is "the
   never-seen mons", and the 0–1-unrevealed stratum is the read that separates them (§11 D1 splits it into 0 and 1).
2. **"With nearly all species seen, fair keeps ~¼" is a pooled 0–1 stratum, POST-HOC and descriptive**, with no CI
   reported on the ratio. One unrevealed slot is a whole hidden set (species, four moves, item, spread).
3. **The 16 % is a slot-POSITION metric**; the game-relevant set recall is 45 % (the network's belief: 51 % / 48 %).
4. **The species-belief null (R-BELIEF) is consistent with "species don't matter" and with "the right species with
   Smogon sets is still the wrong mon".** It does not by itself point at moves.
5. **The referee rewards matching the referee's world, including its continuation.** The oracle leaf is the
   referee's own estimator on disjoint dice: it knows the opponent's team AND, through it, exactly how the greedy
   opponent continuation will play. Part of "information" may be "a model of this particular opponent policy".
   §6.7 adds a second referee with a different continuation.
6. **The gap is a value-of-information (VOI) quantity, partly IRREDUCIBLE.** Even an exact posterior over the
   opponent's team leaves argmax_a E_w[Q(w, a)] ≠ argmax_a Q(s_true, a): no honest player can close that part.
   The pilot cannot separate VOI under the exact posterior from posterior ERROR; only the second is a research path.
   §6.4's pool-posterior arm (the Bayes posterior of the distribution the eval opponents actually draw from) does.

### 2.3 Prior evidence AGAINST "moves bind" (owner and orchestrator, 2026-10-11)

| evidence | what it says | how this design discriminates |
|---|---|---|
| **X5 oracle reference arms** (`design_x5_belief_tokens.md` §7.6–§7.7; UNDERSTANDING "Perfect information about the opponent's team buys little strength"): networks TRAINED 15M with the opponent's species (+2.44 pp [+0.31, +4.57]) or whole set (+2.18 pp [+0.53, +3.84]) revealed in the observation, one-sided clairvoyance, vs blob | For an AMORTIZED policy (no search), knowing the set added nothing over the species (−0.26 [−1.92, +1.40]): moves do NOT bind the trained policy at this budget | It is about the POLICY, not search. Three readings: (i) information × search interaction (moves matter only to an explicit rollout); (ii) the network cannot use the information (shortcut learning; the caveat registered then); (iii) a different architecture / era. §7 (frozen network told the truth) and §8 (forks trained with it) separate (i) from (ii) on the CURRENT architecture |
| **Foul Play** (`foulplay_axes_and_frontend_validation_2026-09-16/`; UNDERSTANDING L659–L666): MCTS on a hand-written evaluator, sets from Smogon usage only, beat our 75M policy (our win rate 0.388 [0.288, 0.497], ~1.4M visits); its knowledge delta at matched width (full team datasets vs Smogon-only sets) +0.000 [−0.15, +0.15], n = 80 | Weakly against "moves bind" at huge search width; or WIDTH SUBSTITUTES for knowledge (many determinized worlds average to a robust choice); or its EVALUATOR is the edge | Three separators: R-WIDTH (§6.6: does fair-world IMP climb with the number of worlds at fixed knowledge?); the DOSE (§6.4: does it climb with knowledge at fixed width?); R-EVAL (§6.5: Foul Play's own evaluator as our leaf, in oracle AND fair worlds). The n = 80 CI (±15 pp) is too wide to weigh heavily; the per-decision dose is ~10× more precise |
| **The POOL-vs-PRIOR confound**: the fair sampler draws Smogon sets, the opponents play our ~719-team pool | A low hit rate may measure the pool's un-Smogon-ness, not the game's difficulty; on the ladder the prior may fit far better | §6.4's POOLSETS arm (worlds drawn from the POOL's own per-species set distribution) and the descriptive read `poolprior_read.json` (§11 D9: the Smogon prior's hit rates on pool sets vs real human sets) |

---

## 3. The competing hypotheses, each with the test that would KILL it

| id | hypothesis (an explanation of G) | its kill test (registered in §12) | first cheap read |
|---|---|---|---|
| **H-MOVES** | the seen opponent mons' unseen MOVES carry most of G | R-MOVES REFUTED (§12.3): both shares' upper bounds < 0.25; or the wrong-moves placebo buys as much as the truth; or another class clearly dominates | §11 D2–D3 (G by hidden move slots); §7 |
| **H-UNSEEN** | the never-seen mons (species + whole set) carry it | s_U and n_U upper < 0.25 | §11 D1 (0 vs 1 unrevealed) |
| **H-ITEMS** | unrevealed items (Choice Band, unused berries, Leftovers at full HP) | s_I and n_I upper < 0.25 | — |
| **H-ABILITIES** | unrevealed abilities (two-ability species only) | s_A and n_A upper < 0.25 | — |
| **H-SPREADS** | natures / EVs / IVs → exact stats, speed order, KO thresholds | s_S and n_S upper < 0.25 | the F sampler's gate acceptance on spreads (§6.1): if observations pin the spread, it is not really hidden |
| **H-EXACT-HP** | the true integer HP inside the displayed 1 % bucket | in the PILOT's worlds: A2 (§5) shows a seen mon's exact HP equal to the truth ⇒ zero component by construction; in F worlds exact HP is nested in S (replay determines it), so s_S bounds it; a state-edit knock-out is DEFERRED unless s_S lower > 0.25 | §5 A2 |
| **H-COUNTERS** | hidden counters and volatiles (sleep / Toxic counters, PP, choice lock, Substitute HP, Encore / Disable turns, Wish, Future Sight) | A2: zero counter / volatile differences beyond those CAUSED by a re-drawn move (those belong to M) | §5 A2 |
| **H-ARTIFACT** | the fair worlds are mis-built though byte-consistent (GIGO) | R-GIGO PASS (§12.1): A1, A2, A3, A5 pass; each audit's teeth shown on a planted defect; the independent re-implementation agrees (§13.1) | §5 (running now) |
| **H-STAT** | selection (the 3.5 % without 8 worlds), the loss-enriched quota, a few state types carrying the gap | D6 worst-case bound keeps R-INFO's lower bound > 0.04; the reweighted G within ±0.02 of raw; D5's split-half concentration (if the top 10 % of states carry > 80 % of G, the claim is restated for that state type) | §11 D5–D7 |
| **H-VOI** | G is mostly information NO honest belief can have (irreducible) | r_POOL lower ≥ 0.5 kills it (the Bayes posterior of the eval distribution recovers half) | — |
| **H-PRIOR-MISMATCH** | G is partly our Smogon prior's mismatch to the pool, not the game | r_POOLSETS upper < 0.10 AND D9's recall difference (human − pool) CI includes 0 or favours the pool | §11 D9 |
| **H-EVALUATOR** | our critic / leaf, not our information, is the binding constraint | R-EVAL (§12.6): the hand evaluator is NOT better than our critic in fair worlds (upper < +0.005) and collapses oracle → fair like ours | — |
| **H-WIDTH** | more fair worlds substitute for knowledge (Foul Play's reading) | R-WIDTH WIDTH-FLAT (§12.7) | — |
| **H-HORIZON** | G is a full-length-rollout phenomenon (wrong worlds compound over ~26 turns) | G at TRUNC-4-K8 ≥ ½ G at LEAF-K8 (the pilot: about half on end2) | §11 D8 |
| **H-REFEREE** | part of G is knowing the referee's own opponent continuation | G under the second referee (§6.7) ≥ 0.75 of G under the first | — |
| **H-TRANSIENT** | G (or the moves' share of it) is a feature of a still-improving 15M policy / belief: the "v8 gift" trap | the maturity guard (§10.4) does not fire, and the plateau replication CONFIRMS (§10.5) | §10.1–§10.2 |

---

## 4. Design principles (owner, 2026-10-11)

- **Correctness over economy.** The study outranks other agent work for quota. No arm is cut for cost if it is
  needed to kill a rival; anything that cannot change a decision is cut.
- **Register first, then read.** This section's commit lands before any arm of the study is read. The analysis
  scripts are committed before the arm outputs exist; arms are written under OPAQUE CODES and unblinded only after
  the blinded tables are frozen (§12.10).
- **Independent replication of the instrument** (§13.1) and **a red team** (§13.2) before any conclusion steers.
- **Provisional at 15M** (§10.5): a conclusion unlocks cheap reversible work only, until a plateaued checkpoint
  replicates it.
- **CPU constraints:** until the closing test ends (≈ Mon 09:00), cores 5–7 and their SMT siblings, ≤ 3 workers for
  this study, `nice -n 19`, `mem_cap`; then the full box minus what the next GPU run needs. No GPU in this unit.

---

## 5. The sampler AUDITS (H-ARTIFACT; running in this unit)

Each audit writes `audit_<name>_<cp>.json` with counts, PASS / FAIL and up to 10 examples per failure.

| id | audit | PASS |
|---|---|---|
| **A1** | BUILDER IDENTITY through the builder: ≥ 150 states per checkpoint, every "world" = the TRUE team pushed through `build_worlds`' command-template rendering (slot rewrites, switch rewrites) and gate; the rendered commands replay to the same observation rows for BOTH sides; `run_det_playouts.one_state` reproduces Phase 1's outcomes on dice seeds 0–7 for every legal action (the pilot checked 12 states, bypassing the template) | 0 mismatches |
| **A2** | FIELD DIFF AT THE ROOT: for every played world, the OPPONENT's own observation row (its exact HP, PP, item, ability, stats, status and counters, boosts, volatiles) in the world vs the truth, cell by cell, named through the pin's layout; classified by (block, field, slot, seen / never-seen). Expected: a seen mon's re-drawn move slots (identity, move features, that slot's PP); anything on a never-seen mon; the opponent's own last-action cells where a past unseen choice was rewritten (counted separately) | 0 UNEXPECTED cells (a seen mon's exact HP, status, counters, boosts, volatiles, item, ability, stats; the field; the opponent's view of our side) |
| **A3** | RE-GATE DETERMINISM: 100 archived worlds per checkpoint replayed again | 100 % byte-equal |
| **A4** | LEGALITY of world teams: Species Clause, ≤ 1 Hidden Power, no engine-banned move or item, learnset legality, Hidden Power type = the IVs' | counted; > 1 % of world teams in violation ⇒ FINDING, fixed before §6 |
| **A5** | ROOT DICE: from the Rust source at both pins, whether the dice after the root depend ONLY on the playout seed (not on the replayed prefix's PRNG state, which could differ by world); a direct check if not | the dice depend on the seed only, or the check shows equality |
| **A6** | THE GATE'S CONDITIONING (descriptive): the re-drawn moves' marginal vs the prior; why 3.5 % of states failed to build 8 worlds | — |

**Teeth.** Each of A1–A3 is re-run once on a PLANTED defect (e.g. a world whose seen mon's item is swapped; a
template that renders one slot index off) and must FAIL; a check that cannot fail is not a check.

---

## 6. The PLAYOUT-WORLD axis: what does SEARCH need to know? (the factorial; runs after the closing test)

### 6.1 Attribute classes and the three bases

The opponent's hidden facts at a root are PARTITIONED into five classes, so every hidden fact belongs to exactly one:

| class | the facts | revealed when |
|---|---|---|
| **M** | the unseen move slots of SEEN mons (and the TYPE of a revealed bare Hidden Power) | the move is used in the searched side's view |
| **U** | every fact of a NEVER-SEEN slot: species, moves, item, ability, spread | it switches in (then its facts move to M / I / A / S) |
| **I** | the unrevealed item of a seen mon | `-item`, `-enditem`, `[from] item:` naming it |
| **A** | the unrevealed ability of a seen mon (two-ability species only) | `-ability`, `[from] ability:` |
| **S** | the nature, EVs and IVs of a seen mon (hence exact stats, speed, KO thresholds and its exact HP) | never directly; constrained by observed damage and speed order |

- **O** (oracle): every class TRUE. Phase 1's rows, no new playouts.
- **F** (fully honest, NEW sampler): every class SAMPLED from the Smogon prior and gate-conditioned. M by a
  conditional-Poisson fixed-mass draw from the prior renormalised over the unrevealed candidates (`cp_sample`, so the
  dose family's p = 0 IS F exactly); I, A, S of seen mons per mon: each candidate in the prior's support is screened
  by a replay with the other mons at their F draws, then the world's combination passes the full gate (≤ 60 tries;
  a state enters with 8 worlds). States without 8 F worlds are excluded and COUNTED; if > 10 % of states are
  excluded, every F-based share is reported on the included states AND the P-based shares (§6.2) become co-primary.
  The gate's acceptance per class is itself a read: a spread the observations pin down is not hidden from an honest
  player.
- **P** (the pilot's Phase 2): M and U sampled; I, A, S of seen mons TRUE. In this partition P = O − I − A − S... of
  the HIDDEN classes, P samples exactly M + U.

### 6.2 The arms (each: 8 worlds per state, all legal actions, LEAF-K1/K4/K8, TRUNC-d-K for d ∈ {1, 2, 4}, the honest one-ply CRITIC on the worlds, Phase 1's dice seeds and referee)

| code family | arm | what it answers |
|---|---|---|
| base | **F** | the fully honest IMP |
| SUFFICIENCY ("reveal ONLY X", from F) | **F+M**, **F+U**, **F+I**, **F+A**, **F+S**; **F+Usp** (the never-seen SPECIES only, their sets sampled) | is X alone enough to close G? |
| NECESSITY ("reveal ALL BUT X", from O) | **O−M**, **O−U**, **O−I**, **O−A**, **O−S** | does hiding X alone open G? |
| DOSE (§6.4) | **D25, D50, D75, D90** (the true unseen moves get presence mass p, the prior 1 − p; p = 0 is F, p = 1 is F+M) | how sharply must the move belief improve? |
| PLACEBO | **W50, W90, W100**: equally confident WRONG but plausible moves | does any confident move set help, or the true one? |
| HONEST SHARPENING | **MAP90, MAP100**: the prior's own top-(4 − r) moves at mass p | does reducing world diversity help on its own (an honest option)? |
| POSTERIORS | **POOL**: worlds drawn from the training-pool teams consistent with the observation (gate-conditioned); **POOLSETS**: F with each species' moves / item / ability / spread drawn from the POOL's own per-species distribution instead of Smogon's | how much of G is recoverable by the Bayes posterior of the eval distribution, and how much is Smogon's mismatch to the pool? MEASUREMENTS only: the pool never ships as a prior (owner's rule, 2026-08-15) |
| HAND EVALUATOR (§6.5) | **HE** scorers on O, P and F worlds | is our evaluator or our information binding? |
| WIDTH (§6.6) | **P16**: the pilot's world kind with all 16 built worlds played | does width substitute for knowledge? |
| REFEREE 2 (§6.7) | a second truth with a different opponent continuation, on 300 states | is part of G a model of the referee's opponent? |

The pilot's Phase 2c ceiling arm (2c-T: sampler C with every seen mon's TRUE set, run by the pilot's own agent) is
close to **O−U** with the never-seen slots drawn from the network's belief. It is read under ITS registration
(`design_value_discrimination.md` §7.10 Amendment 6). Here it is corroboration only, with one registered prediction
made before it is read: **under H-MOVES, (CT − B) / (oracle − B) at LEAF-K8 ≥ 0.5** on both seeds.

### 6.3 The attribution arithmetic

- G = IMP_O − IMP_F (LEAF-K8 primary; TRUNC-4-K8 secondary).
- Sufficiency share s_X = (IMP_{F+X} − IMP_F) / G; necessity share n_X = (IMP_O − IMP_{O−X}) / G; ratios of means on
  the same bootstrap draws. A share is computed only when G's lower bound > 0.02 (else "no gap to apportion").
- With interactions s_X ≠ n_X; both are reported, and Σ_X s_X, Σ_X n_X say how additive the classes are. A class that
  is necessary but not sufficient (high n, low s) is an INTERACTION; the follow-up pairs it with the class of the
  next-largest n (F+M+X).
- Pooled over checkpoints: Σ Δ / Σ G with states weighted equally, the bootstrap resampling battles within each
  checkpoint.

### 6.4 The dose-response, the placebo, honest sharpening and the posteriors

- **Mixture.** For a seen mon with r revealed moves and true unseen set T (|T| = 4 − r), each unrevealed candidate m
  gets presence π'_m = p·1[m ∈ T] + (1 − p)·π_prior(m), π_prior the Smogon inclusion probability renormalised to
  mass 4 − r over the candidates (so Σ π' = 4 − r), then a conditional-Poisson fixed-size draw (`cp_sample`), ≤ 1
  Hidden Power, then the gate. Every other class: F's draws. This is the owner's "high mass to the known moves, but
  not 100 %".
- **PLACEBO W_p:** the same with T replaced by a WRONG set W (|W| = 4 − r) drawn ∝ π_prior over the candidates
  EXCLUDING T (deterministic per state and mon: `sha256("hib_wrong_v1:" + id + ":" + slot)`). Equally confident,
  plausible, wrong.
- **MAP_p:** T replaced by the prior's own top-(4 − r) candidates. Honest (no peeking); it isolates "fewer, sharper
  worlds" from "the right worlds".
- **POOL:** a world's opponent team = a pool team drawn uniformly among those whose species include every revealed
  species and whose sets contain every revealed move, then the gate (the true team is in the pool, so POOL can be
  the oracle when the observations identify the team). **POOLSETS:** F's structure, the pool's per-species set
  frequencies instead of Smogon's.
- Recoverability r_X = (IMP_X − IMP_F) / G for X ∈ {POOL, POOLSETS, MAP100, D25…D90}.

### 6.5 The HAND EVALUATOR arm (backlog X3; owner, 2026-10-11): evaluator-bound or information-bound?

Foul Play's leaf, poke-engine's `gen3/evaluate.rs` (MIT, © 2024 pmariglia; the licence notice travels with the
port): per mon alive +30, HP fraction × 100, status penalties (freeze −40, sleep / paralysis / burn −25 with burn
scaled by the physical-move count, Toxic −30, poison −10, Guts / Marvel Scale exceptions), an item +10, a low-PP
penalty, a floor at 0, Spikes layers −12 / −16 / −25 on grounded mons, Leech Seed −30, Substitute +75, confusion
−20, the active's boosts by a step table × 30 / 15, Reflect / Light Screen +20 PER TURN LEFT (a fresh screen +100: the source keeps both as turn counters; corrected 2026-10-11 by the port's author); side one minus side two.

- **Port:** a pure function over the SEARCHED side's observation row (its own exact HP and set; the opponent's HP
  bucket, status, boosts, revealed moves / PP and item state; hazards, screens, volatiles), so it reads exactly the
  information our critic reads in each world: the comparison is evaluator against evaluator, never evaluator
  against information. A full-state variant (the world's internal state) is secondary and only if the Rust state is
  reachable at the pins without a build. **BUILT 2026-10-11:** `src/main/hidden_info_study/hand_eval.py` (HEAD; 73
  tests: every term, a side-swap symmetry, and 905 real Rust-core decisions checked term by term against an
  independent protocol tracker with 0 mismatches, each layout field's one-cell shift caught); the pinned twin
  `hand_eval_pin.py` loads the same source by path (the layouts at both pins resolve identically); the driver
  `hand_eval_playouts.py --world O|P|F` re-plays to T + 4 and records the hand eval and V at every searched-side
  decision (its critic traces reproduce the pilot's archived ones exactly). Named deviations, each forced by
  reading the one-sided row (the module docstring): a never-seen opponent slot scores full HP, held item, grounded;
  an unknown opponent item scores held; only KNOWN move slots get the low-PP term; an unrevealed ability is the
  prior's mode; burn's halving uses base stats when the spread is unknown. Its leaf scale for MIX-type scorers is
  Foul Play's own (σ(0.0125·(h − h_root)), 1 / 0 at a terminal); argmax scorers need no scale.
- **Scorers:** HE-1PLY (the one-ply successors, seeds 40–47) and HE-TRUNC-d-K (d ∈ {1, 2, 4}, K ∈ {4, 8}: the hand
  eval at the searched side's first decision d turns on), in O, P and F worlds, beside CRITIC and TRUNC-d-K. The
  truncated playouts are re-played with a driver that records the eval (Phase 1 / 2 stored V only): k ≤ 4 turns, ≈
  1/7 of a full playout's cost.

### 6.6 WIDTH at fixed knowledge (Foul Play's alternative reading)

P16: the pilot's 16 built worlds per state all played (the pilot played 8). R-WIDTH reads IMP_P(LEAF-K16) −
IMP_P(LEAF-K4); the dose reads knowledge at fixed width. If both rise, width can substitute for knowledge; if only
the dose rises, knowledge is the lever and Foul Play's edge must lie in its evaluator or search shape (§6.5).

### 6.7 A second referee

On 300 states (every 2nd registered state of each checkpoint), 24 new truth seeds per action with a DIFFERENT
opponent continuation: the checkpoint at temperature 1 (sampled, seeded) on the opponent side only. Every arm's
picks are fixed; only T changes. G2 / G and the shares under both referees.

### 6.8 Replication across checkpoints

end2 and prod (seed 2001, both with Phase 1 truth), and a THIRD: `rb_es_end2_s2002` (same pin as end2) on its OWN 600
states, which needs its own Phase 1 (≈ 4 CPU-h) and Phase 2 rows.

---

## 7. The MODEL-INPUT axis: does the NETWORK use the moves if told? (running in this unit)

### 7.1 Where the truth goes in

`_apply_move_belief` (`src/agents/model/extractor_forward.py`, T0) publishes ONE typed move posterior,
`last_move_belief_logits` [B, 6, 400] (independent per-move sigmoids, Hidden Power typed at nums 355–370), and every
consumer reads it: the move-identity reinjection into the opponent role tokens, the opponent active's move group (E4
seats, the intent pointer's operands), the X5 op roster's fixed-mass `move_w` that prices every opponent mon as an
ATTACKER in the damage operator, the edges and the cells (ARCHITECTURE.md §2.1 item 3, the X5 paragraph). The hook
rewrites that posterior for each SEEN opponent slot: q' = (1 − p)·σ(ℓ) + p·1[m ∈ target], ℓ' = logit(clip(q')),
revealed moves untouched; the observation is untouched. Production (`--move-set-closure off`) and end2 (`on`) then
close it with their own readers. Teeth: p = 0 is BIT-IDENTICAL to the unhooked forward; at p = 1 the published
posterior and the roster's `move_w` put ≥ 0.99 of the slot's unrevealed mass on the target.

### 7.2 Arms (per state; coded opaquely, §12.10)

| arm | target | doses p |
|---|---|---|
| **TRUE** | the true unseen moves of each seen mon | 0 (identity), 0.6, 0.8, 0.95, 1.0 |
| **PLACEBO** | 4 − r WRONG moves per mon, drawn ∝ the network's own belief excluding the true ones | 0.8, 0.95, 1.0 |
| **SELFMAP** | the network's own top-(4 − r) moves (an honest sharpening) | 0.8, 0.95, 1.0 |
| **TRUE-HYP** (secondary) | TRUE, plus a hypothesis slot whose hypothesis species IS a true never-seen species gets that species' true moves | 0.95, 1.0 |

Registered secondary arms if their hooks prove clean at the pins: ITEM (the item posterior → the true item for seen
mons with an unrevealed item; its consumer is the op's P(Choice Band)) and SPREAD (the believed stats → the true
stats). The OBSERVATION-channel reveal (`--oracle-reveal full` re-encoded at the pins) is DEFERRED: the pins'
offline encoders do not take it, and a network never trained on revealed rows reads them out of distribution (the X5
both-sided cells: −8.1 pp).

### 7.3 Measures (per checkpoint: end2, prod, and `rb_es_end2_s2002` on end2's states)

- **Policy shift:** mean masked KL(π_arm ‖ π_0) and the top-1 FLIP rate.
- **The policy's own pick against the referee:** IMP_POL(arm) = mean[T(top1_arm) − T(top1_0)]. Free: T exists for
  every legal action.
- **The critic:** the one-ply successors (seeds 40–47, the TRUE world, the opponent answered by the UN-injected
  network) regenerated once (teeth: p = 0 reproduces Phase 1's critic values within 1e-5), forwarded under every arm
  with the injection recomputed at each successor; IMP_CRIT(arm), the pairwise sibling accuracy on non-tied pairs,
  top1 | top2 accuracy, the ECE of the per-action value against T.
- Strata: unrevealed slots (0–1 / 2–3 / 4–5), the number of targeted move slots.
- For s2002 on end2's states, the referee is end2_s2001's continuation (stated; a third network's pick scored by
  another's referee).

### 7.4 How it reads

| pattern | meaning |
|---|---|
| TRUE improves IMP_POL / IMP_CRIT, PLACEBO does not (or hurts), SELFMAP does not | the network USES move information it is given: it LACKS it. Better beliefs help the policy directly |
| TRUE moves the policy (flips ≥ 2 %) but IMP ≤ 0 | it reacts without benefit: it cannot USE the information (or reads confident beliefs out of distribution) |
| nothing moves (flips < 2 %, KL < 0.01) | the move channel barely reaches decisions |
| TRUE ≈ SELFMAP ≈ PLACEBO | it reacts to CONFIDENCE, not content |

**The frozen-network caveat.** A frozen network never saw confident correct move beliefs often in training; a null
here does not show that a network TRAINED with good beliefs would not gain. That is §8's question, and the X5 oracle
arms' +0 for the set over the species is its prior evidence.

---

## 8. The TRAINING axis: KNOCK-IN FORKS (GPU; QUEUED by the orchestrator 2026-10-11 after the closing test's verdict branch and the bf16 strength A/B, ≈ mid-week, CONDITIONAL on §8.6)

**Question:** can the network LEARN to use each attribute when it is given it, and how much strength is each
worth? The frozen axis (§7) cannot answer the first half; the X5 oracle arms answered it for the OBSERVATION channel
(trained from scratch, blob architecture, +2.18 pp for the whole set, +0 over species).

### 8.1 The mechanism: `--belief-oracle` (to build; ≈ 2–2.5 agent-days)

- **Flags:** `--belief-oracle {off, moves, items, spreads, wrong_moves}` (default `off`) and `--belief-oracle-mass p`
  (default 0.95). Class `resume_immutable` (no module, no weight: like `--oracle-reveal`), recorded in
  `model_config.json` and `metadata.json`, refused if a same-run resume flips it, ALLOWED to differ from the parent on
  a genuine fork (that is its use), printed by `checkargs` and the launch banner.
- **What it does:** in `_apply_move_belief` / `_spread_hp_damage`, for the opponent's SEEN slots only, the published
  posteriors are rewritten from the training-only LABEL keys the Rust env core already emits (ARCHITECTURE.md §7):
  `moves` mixes the typed move posterior with the true set from `known_moves` (§7.1's rule: q' = (1 − p)·σ(ℓ) +
  p·1[true], revealed moves untouched); `items` mixes the item posterior with `item_label`'s one-hot (consumers: the
  op's P(Choice Band)); `spreads` mixes the believed stats with the true stats from `belief_spread`; `wrong_moves`
  is `moves` with a deterministic WRONG set (per battle, slot and episode seed: 4 − r moves drawn ∝ the network's
  own detached posterior excluding the true and revealed ones). Whether `known_moves` carries the full TRUE set (not
  only the revealed moves) is verified first, against the packed team, by a test; if it does not, the build adds a
  `true_moves` label key.
- **Leak discipline:** the flag feeds a LABEL to the forward on purpose. Off is BYTE-INERT (a parity test on the K9
  learner golden and the production / end-state graph identities). The offline tools (`main.anchors`, `main.play`,
  `main.belief_roles`, `main.policy_spectrum`) REFUSE a non-off checkpoint; `main.h2h` plays one ONE-SIDED, with the
  label keys emitted for the told side only (a per-side emission, the `--oracle-reveal-mode one_sided` pattern;
  +0.5 agent-day of the build).
- **STARTUP GATE (G-HIB, in the first update; a refusal is `FATAL_CONFIG` and no data counts):** on the update's first
  minibatch, (a) the forward with the flag at mass 0 is BIT-IDENTICAL to the flag-off forward; (b) at the arm's mass,
  on every targeted slot, the published posterior's σ on each true (or, for `wrong_moves`, each wrong) move is ≥ p
  and the roster's fixed-mass `move_w` puts ≥ p − 0.01 of the unrevealed mass on the target; (c) the share of
  opponent slots targeted is > 0 and is logged. TensorBoard rides `hib/targeted_slot_share`, `hib/mass_on_target`
  and `hib/label_coverage` every rollout.

### 8.2 The pin

**P_hib** = the commit that lands §8.1, with this precondition checked on CPU before any fork (the closing test's
identity method, `design_endstate_closing_test.md` §2.3): with the flag OFF, the parent loads and the end-state
arm's graph / state / outputs hashes and the K9 learner golden at P_hib equal those at the parent's pin `62085750`.
If they do not, the flag is built as a minimal patch on `62085750` instead, tagged `hib-forks-pin`, and that tag is
P_hib.

### 8.3 The arms, the parent and the argvs

- **Parent:** the closing test's ADOPTED arm's seed 2001 final (the pilot's convention): on PASS
  `models/rb_es_end2_s2001/final_model.zip` (15,047,835 steps, `lr_now` 3.6e-4, the KL controller adaptive), on FAIL
  `models/rb_es_prod_s2001/final_model.zip` (`--arch production`, its own `lr_now` read with `python -m main.dose`).
- **Arms × seeds (5 × 3 = 15 forks):** `moves`, `items`, `spreads`, `wrong_moves` (placebo, mass 0.95) and the
  **NO-CHANGE control** (`--belief-oracle off`; the v8 guard: continued training is never credited to a lever).
  Seeds 3101–3103 for every arm (the same seeds across arms, so arm × seed cells are paired). Run names
  `rb_hib_<arm>_s310N`, `<arm>` ∈ {moves, items, spreads, wrongmoves, ctrl}.
- **Budget:** +3M steps each (`--steps` is the ABSOLUTE target: 18,047,835 for the end2 parent). The SAME `--fork-lr`
  for every arm (the parent's final 3.6e-4; the controller stays adaptive, the recipe's), so every arm starts at one
  dose; `python -m main.dose` reads each fork's realised dose and a dose ratio outside [0.8, 1.25] vs the control is
  a FINDING stamped on the read.
- **The argv (end2 parent, the `moves` arm, seed 3101; the others change `--belief-oracle`, `--seed`, `--run-name`):**

```bash
cd /home/goodlad/dev/gen3ai && PYTHONPATH=src python -m main.launcher --restart-interval-hours 6 --pin-commit <P_hib> \
  --model models/rb_es_end2_s2001/final_model.zip --arch endstate --steps 18047835 --seed 3101 \
  --fork-lr 0.00036 --belief-oracle moves --belief-oracle-mass 0.95 \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all \
  --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda --run-name rb_hib_moves_s3101
```

  The control: the same with `--belief-oracle off` (or the flag omitted) and `--run-name rb_hib_ctrl_s3101`. Each argv
  is validated at P_hib before launch: `python -m main.checkargs --argv "…"` (the arch surface inherited from the
  parent, the typed `--belief-oracle` the only lever) and `python -m main.launcher --dry-run …` (FORK, `+3,000,000
  steps`, the pool auto-seeded from the parent). The training session launches; the orchestrator grants the lease.

### 8.4 The evaluation

- **Primary cross:** for each lever arm L, its 3 seeds × the control's 3 seeds, mirrored pairs on the pool teams,
  1,000 pairs (2,000 games) per cell, ONE-SIDED (L told through its label keys; the control as trained), the X5 A/B's
  cross statistic and interval (`design_x5_belief_tokens.md` §7.4; seeds as the units): **Δ_L** in pp.
- **Honest-play cross (secondary):** L with the flag OFF at eval vs the control: did learning WITH the information
  change its honest play (a dependence cost or a transfer)?
- **The v8 read:** the control vs its parent (3 seeds × the parent, mirrored): what +3M of plain training buys.
- **Meters per fork:** policy drift vs the parent; the belief maturity read (§10.2) of its own heads; §7's model-input
  axis on the registered states with its own injection (IMP_POL(TRUE 0.95) of a network trained to read the truth vs
  the frozen parent's).
- **Cost:** end state ≈ 0.21 GPU-h per 1M steps (`design_endstate_closing_test.md` §7: 3.0–3.2 h per 15M) ⇒ ≈ 0.85
  GPU-h per 3M fork with its start-up; 15 forks ≈ 13 GPU-h; the crosses (4 × 9 cells + 4 honest + 3 v8 cells ≈ 90,000
  games, at the closing test's ≈ 1.2 GPU-h per 128,000) ≈ 1 GPU-h; **≈ 14 GPU-h**. The extension of §8.5 adds ≈ 6 GPU-h.

### 8.5 Pre-registered rules for the forks (rule 8: a bound within ±0.05 pp of a boundary is not clear)

- **K-VALID:** every fork passed G-HIB; the placebo is not a gain: Δ_WRONG upper ≤ +0.05 pp is NOT required, but if
  **Δ_WRONG lower > +0.05 pp** every lever verdict below is VOID (a "told" gain that a wrong set also buys is not the
  content).
- **K-L-WORTH (per lever L):** Δ_L lower > +0.05 pp AND Δ_L − Δ_WRONG (paired by seed cell) lower > +0.05 pp: the
  network LEARNS to turn attribute L into strength.
- **K-L-NULL:** Δ_L's CI inside ±2.0 pp (the closing test's margin): knowing L buys no strength worth a programme.
- **K-L-NOT-DETECTED:** otherwise. **Extension (registered):** if Δ_MOVES is NOT DETECTED with a CI wider than ±2.5
  pp, the `moves` arm and the control extend to +6M (same seeds, ≈ 6 GPU-h) BEFORE the verdict is read.
- **K-RANK:** Δ_MOVES − Δ_L (paired cells) for L ∈ {items, spreads}: MOVES-FIRST if each lower > +0.05 pp.
- **Reading with the frozen axes:** K-MOVES-WORTH with NET-NOT-DETECTED (§12.8) = the network CAN learn to use the
  moves but does not with today's belief (a belief-quality path); K-MOVES-NULL with MOVES-BINDING (§12.3) = moves
  matter to SEARCH, not to an amortized policy (the X5 reading, now on the current architecture).
- **Prediction (from the X5 arms):** K-MOVES NOT DETECTED or NULL; K-ITEMS and K-SPREADS NULL; Δ_WRONG ≤ 0.

### 8.6 The condition for launching

The forks launch (in the orchestrator's queue slot) unless the frozen reads make them moot: they are CANCELLED only if
R-GIGO FAILS (the instrument must be fixed first) — every other frozen outcome leaves the training-level question
open, so they run.

---

## 9. The GAME-LEVEL check (PLANNED only)

A per-decision improvement is not a win rate. After the factorial, if a verdict says knowledge pays for search: a
mirrored head-to-head battery on one checkpoint between a FAIR-search player (TRUNC-4-K4 argmax over F worlds), a
MOVES-search player (the same over F+M worlds: the search told the seen mons' moves) and the POLICY (no search),
with the ORACLE-search player as a cheating ceiling. ~1,000 mirrored games per pair; one honest search decision costs
≈ 3 CPU-s (TRUNC-4-K4 in worlds, UNVERIFIED for the end state), ≈ 30 decisions per side per game ⇒ ≈ 200 CPU-s per
game with one searching side ⇒ ≈ 55 CPU-h per pair on CPU (a GPU-batched forward would cut it several-fold).
Build: a search player in `main.h2h` over `src/main/value_search/` (≈ 1 agent-day). Read: the moves-search − fair-search
win-rate difference with a mirrored-pair CI; its relation to the per-decision dose.

---

## 10. MATURITY and ADVERSARIAL ROBUSTNESS (owner, 2026-10-11)

The owner's precedent: the "v8 gift" looked like a lever's effect but was the policy still improving in a
non-plateaued region. At 15M the policy and its belief may still be moving.

### 10.1 How fast is the policy still changing? (running in this unit)

`main.policy_drift` (collect → watch `--source both` → report) over `rb_es_end2_s2001` and `rb_es_prod_s2001`'s
checkpoints, every 1M to 15M: the masked KL and greedy flip rate per 1M steps vs the previous checkpoint and the
10M-back reference. Output: `policy_drift_summary.json`. Descriptive meter.

### 10.2 BELIEF MATURITY (running in this unit)

The network's OWN belief accuracy on the 600 registered states at end2 1M / 3M / 5M / 7M / 10M / 12M / 15M (and
prod 5M / 10M / 15M): move expected set recall (Σ of the fixed-mass `move_w` on the true unseen moves / (4 − r)) and
top-(4 − r) recall, vs the Smogon prior's; the item posterior's P(true item) and top-1; the species presence mass on
true hidden species. Output `belief_maturity.json`. **M-BELIEF:** IMPROVING if the 15M − 10M move-recall difference's
lower bound > 0; FLAT if its CI lies inside ±0.01; else UNRESOLVED. (The owner calls a belief that learns the pool a
success milestone; it is also exactly the mechanism by which a moves gap could shrink with training.)

### 10.3 The TRAJECTORY arm (with the factorial)

`rb_es_end2_s2001` at 5M, 10M and 15M, each as the checkpoint CP (leaf continuation, critic and referee all that
snapshot's), on the SAME 600 registered states: Phase 1 truth at 5M and 10M (≈ 4 CPU-h each); the core arms F, F+M,
O−M, D50, D90, W90; the model-input axis TRUE 0.95 / 1.0, PLACEBO 0.95, SELFMAP 0.95. **M-STABLE** on s_M(15M) −
s_M(5M): STABLE if the CI lies inside ±0.25; SHIFTING-DOWN if upper < 0; SHIFTING-UP if lower > 0; else UNRESOLVED.

### 10.4 ADVERSARIAL conditions, and the maturity guard

- **Off-distribution teams:** a new bank of 300 states from CP greedy self-play on the `away` team set (20 teams,
  `designs/ops/anchors.json`), each side on away teams (the belief prior is weaker there); Phase 1 + F, F+M, O−M,
  D90, W90 on it.
- **Opponent class:** the registered states' traces were played against scripted bots (`aggressive`, `heuristic`,
  `random`, `setup_sweep`, `staller` and their v2s) and frozen sentinels; s_M per class (the playout continuation is
  CP on both sides either way). A self-play bank is part of the away bank above.
- **Game phase:** turn 2–10 / 11–25 / 26–40.
- **Long-term-planning states:** tagged at the root from the observation through the pin's layout: hazards up on
  either side; a boosted mon on either side; a setup move legal to the searched side. s_M on tagged vs untagged.
- **CONDITION-ROBUST** if B3 (s_M point ≥ 0.5) holds in every condition with ≥ 150 states; else
  **CONDITION-DEPENDENT**, naming the condition.
- **THE V8 GUARD (registered):** if M-BELIEF is IMPROVING and either M-STABLE is SHIFTING-DOWN or G(15M) − G(5M) has
  upper < 0, a MOVES-BINDING verdict is DOWNGRADED to **TRANSIENT**: no steering until the plateau read (§10.5).

### 10.5 THE SCOPE RULE (registered)

Every verdict read at 15M is labelled **PROVISIONAL "at 15M, current architecture"**.
- **It MAY unlock** cheap, reversible work: meters (the belief-accuracy meter on TensorBoard), sampler improvements,
  builds of ≤ 2 agent-days, and the §8 forks if the owner queues them.
- **It may NOT unlock** a multi-week research commitment (a move-inference programme, a new belief architecture, a
  belief-sampled search build-out) until the core arms replicate on a **PLATEAUED** checkpoint: the X26 deep plateau
  run's checkpoint once X26's own registered plateau criterion fires, AND `main.policy_drift`'s per-1M masked KL and
  flip rate each ≤ ½ of `rb_es_end2_s2001`'s 10–15M means over three consecutive 1M steps.
- **Re-run at plateau:** A1, A2, A5; Phase 1 truth on 600 of the plateau run's own states; F, F+M, O−M, the strongest
  rival class's F+X / O−X, D50, D90, W90; the model-input axis (TRUE 0.95 / 1.0, PLACEBO 0.95, SELFMAP 0.95).
- **CONFIRM:** B3, B4, B6, B7 of §12.3 hold at plateau AND s_M(plateau) − s_M(15M) has a CI inside ±0.25.
  **REVOKE:** at plateau s_M upper < 0.25, or n_M upper < 0.25, or R-PLACEBO-EQUIVALENT fires. Otherwise
  **PROVISIONAL STANDS** (unresolved), and the commitment stays locked.

---

## 11. Registered DESCRIPTIVE reads of the pilot's existing rows (running in this unit; no verdict)

All on the pilot's Phase 1 (oracle) and Phase 2 / 2b (fair) rows, both seeds, LEAF-K8 unless stated; the per-state
quantity g(s) = [T(â_O) − T(top1)] − [T(â_P) − T(top1)]; battle-clustered bootstrap CIs; raw and outcome-reweighted.

| id | read |
|---|---|
| **D1** | G_P by unrevealed opponent slots 0 / 1 / 2 / 3 / 4–5 SEPARATELY (the pilot pooled 0–1) |
| **D2** | G_P by the total unseen move slots of SEEN mons (0–2 / 3–5 / 6–8 / 9+), overall and within 0–1 unrevealed |
| **D3** | G_P by the opponent ACTIVE's unseen move slots (0 / 1 / 2 / 3–4) |
| **D4** | G_P by phase (2–10 / 11–25 / 26–40), by the trace's opponent class (bot / sentinel), by the trace's outcome |
| **D5** | concentration, split-half: rank states by g on truth seeds 16–27, measure the share of G (on seeds 28–39) carried by the top 10 % / 25 % |
| **D6** | the excluded 3.5 %: their oracle IMP vs the included; the worst-case R-INFO if each excluded state's fair IMP equalled its oracle IMP |
| **D7** | G_P by the kind of the oracle's pick (move / switch) and whether the oracle and fair picks differ |
| **D8** | H-HORIZON: G at TRUNC-4-K8 and TRUNC-4-K4 over G at LEAF-K8 |
| **D9** | the POOL-vs-PRIOR confound: the Smogon prior's move / item / ability / spread hit rates on (a) the training pool's sets, (b) real human sets (human-built teams; ladder replays, censored to revealed moves, with the same censoring applied to the pool), (c) the 1,200 opponent teams met in the pilot's states; bootstrap over teams (`poolprior_read.json`) |

**Pre-stated expectations under H-MOVES** (so a pattern can embarrass it): D2 increasing in hidden move slots
within 0–1 unrevealed; D3 increasing in the active's hidden slots; D1's "0 unrevealed" stratum keeping a G_P of at
least half the all-states G_P.

---

## 12. PRE-REGISTERED DECISION RULES (rule 8: deterministic; a bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear)

### 12.1 R-GIGO (the gate on everything that uses the fair-world builder)

**PASS** iff A1 has 0 mismatches, A2 has 0 unexpected cells, A3 is 100 %, A5 shows post-root dice depend only on the
seed (or its check shows equality), and each audit FAILS on its planted defect. **FAIL** otherwise: the pilot's
honest numbers (Phase 2 / 2b / 2c) and every arm built on the sampler are VOID until the defect is fixed and re-run;
§6 does not start. A4 violations above 1 % are fixed before §6 regardless.

### 12.2 R-REPLICATE (the independent implementation, §13.1)

For every state and arm both implementations ran: the world sets agree in DISTRIBUTION (the per-class marginals'
total-variation distance < 0.02 on the pooled draws) and the IMP tables agree: every registered contrast's point
estimates differ by < 0.005 and every verdict of §12.3–§12.9 is the same. A disagreement is a FINDING resolved
before any verdict is reported as more than "computed".

### 12.3 R-MOVES (the headline; on POOLED shares over the three checkpoints, LEAF-K8)

**MOVES-BINDING** iff ALL of:
- **B1** R-GIGO PASS and R-REPLICATE agree;
- **B2** G's lower bound > 0.02;
- **B3** sufficiency: s_M point ≥ 0.50 and lower ≥ 0.25;
- **B4** necessity: n_M point ≥ 0.50 and lower ≥ 0.25;
- **B5** dominance: s_M ≥ s_X and n_M ≥ n_X (points) for every other class X;
- **B6** dose: IMP_{D90} − IMP_F lower > +0.0005, and the paired-bootstrap OLS slope of IMP on p over p ∈ {0, 0.25,
  0.5, 0.75, 0.9, 1.0} has lower > 0;
- **B7** placebo: IMP_{F+M@0.9} − IMP_{W90} lower > +0.0005, and IMP_{W90} − IMP_F upper < +0.01;
- **B8** replication: on EACH checkpoint separately, IMP_{F+M} − IMP_F lower > +0.0005 and IMP_O − IMP_{O−M} lower >
  +0.0005.

**MOVES-REFUTED** iff ANY of:
- **R1** s_M upper < 0.25 AND n_M upper < 0.25;
- **R2 (R-PLACEBO-EQUIVALENT)** the CI of IMP_{D90} − IMP_{W90} lies inside ±0.01;
- **R3** some class X dominates: s_X − s_M lower > 0 AND n_X − n_M lower > 0 (paired).

**MOVES-NECESSARY-NOT-SUFFICIENT:** n_M lower ≥ 0.25 and s_M upper < 0.25 (an interaction: the follow-up runs F+M+X
for the X with the largest n_X).
**MOVES-SHARED:** none of the above, and s_M lower > 0 or n_M lower > 0 (a real but not majority or not unique share).
**MOVES-NOT-DETECTED:** otherwise.
Descriptive flag **NON-ADDITIVE:** Σ_X s_X upper < 0.5 (the classes' separate knock-ins do not account for the gap).

Every verdict is then modified by §10: **TRANSIENT** (the v8 guard), **CONDITION-DEPENDENT**, and the PROVISIONAL
label until the plateau CONFIRMS.

### 12.4 R-RIVALS (one verdict per rival class X ∈ {U, I, A, S}, same thresholds)

**X-BINDING** iff B2–B5 and B8 hold with X in place of M (no dose or placebo for X). **X-REFUTED** iff s_X upper <
0.25 and n_X upper < 0.25. Otherwise **X-SHARED** / **X-NOT-DETECTED** as for M.

### 12.5 R-RECOVER (C2, pooled)

- **BELIEF-RECOVERABLE (on this distribution):** r_POOL lower ≥ 0.5.
- **LARGELY-IRREDUCIBLE:** r_POOL upper < 0.25.
- **PARTIAL:** otherwise.
- **PRIOR-MISMATCH:** r_POOLSETS lower > 0.25 (a material part of G is the Smogon prior's mismatch to our pool).
- **HONEST-SHARPENING-HELPS:** IMP_{MAP100} − IMP_F lower > +0.0005.
- Descriptive **p½** = the smallest registered dose with r_p point ≥ 0.5.

### 12.6 R-EVAL (evaluator-bound vs information-bound, §6.5)

- **EVALUATOR-BOUND:** in fair worlds (F, and P), IMP(HE-TRUNC-4-K8) − IMP(TRUNC-4-K8) lower > +0.0005, OR
  IMP(HE-1PLY) − IMP(CRITIC) lower > +0.0005: a hand-written evaluator beats our critic honestly, so our critic is a
  binding constraint whatever the moves say.
- **INFORMATION-BOUND (support):** the hand eval's own gap g_HE = IMP_O(HE-TRUNC-4-K8) − IMP_F(HE-TRUNC-4-K8) has
  lower > +0.0005 and g_HE / g_CRIT (our TRUNC-4-K8's gap) point ≥ 0.5.
- Both may hold. **EVALUATOR-NOT-BINDING:** the EVALUATOR-BOUND contrasts' upper < +0.005.

### 12.7 R-WIDTH

**WIDTH-HELPS:** IMP_P(LEAF-K16) − IMP_P(LEAF-K4) lower > +0.0005. **WIDTH-FLAT:** upper < +0.01. Else UNRESOLVED.

### 12.8 R-NET (C3, the model-input axis; per checkpoint, then pooled over end2 + prod; s2002 as replication)

At the primary dose p = 0.95 (1.0 secondary):
- **NET-USES:** IMP_POL(TRUE) lower > +0.0005 AND IMP_POL(TRUE) − IMP_POL(PLACEBO) lower > +0.0005.
- **NET-IGNORES:** TRUE's flip rate < 0.02 AND its mean KL < 0.01 nats.
- **NET-MISUSES:** flip rate ≥ 0.02 and IMP_POL(TRUE) upper < −0.0005.
- **NET-CONFIDENCE-ONLY:** flip rate ≥ 0.02, and the CIs of IMP_POL(TRUE) − IMP_POL(SELFMAP) and IMP_POL(TRUE) −
  IMP_POL(PLACEBO) both lie inside ±0.005.
- **NET-NOT-DETECTED:** otherwise.
- **CRITIC-USES:** IMP_CRIT(TRUE) − IMP_CRIT(p = 0) lower > +0.0005 AND IMP_CRIT(TRUE) − IMP_CRIT(PLACEBO) lower >
  +0.0005; **CRITIC-NOT-DETECTED** otherwise. Sibling accuracy, top1 | top2 accuracy and ECE deltas are descriptive.
- Why 2 % and 0.01 nats: at 600 states 2 % is 12 flipped decisions; fewer cannot move IMP_POL beyond its ±0.01 CI.

### 12.9 The maturity rules

M-BELIEF (§10.2), M-STABLE (§10.3), the V8 GUARD and CONDITION-ROBUST (§10.4), CONFIRM / REVOKE (§10.5), as
written there.

### 12.10 Power, blinding and the size rule

- **Power** (from the pilot's measured contrast widths: R-INFO ±0.015, R-BELIEF ±0.0145 at ≈ 580 states): a paired
  two-arm contrast's 95 % half-width ≈ 0.015 per checkpoint, ≈ 0.0105 pooled over two, ≈ 0.0087 over three. With G ≈
  0.07 a share's half-width at s = 0.5 is ≈ ±0.24 per checkpoint, ±0.17 over two, ±0.14 over three. B3 / B4 are
  reachable pooled when the true share is ≳ 0.45; R1 when it is ≲ 0.10.
- **Size rule:** after the Tier-1 arms (§15), if the pooled s_M or n_M half-width exceeds 0.20, 600 more states per
  checkpoint are added (drawn by the pilot's procedure from the earlier eval-trace cycles not used, Phase 1 truth
  built for them) BEFORE any verdict is read.
- **Blinding:** every arm's raw outputs carry only an opaque code (`arm_key.json`: code = the rank of
  `sha256("hib_arm_v1:" + label)`); the scoring scripts are committed before the first arm output exists; the
  blinded tables are frozen (committed) before `unblind_*.py` maps codes to labels and applies these rules as code.
  No stratum, exclusion or threshold changes after unblinding; any change is a disclosed amendment with both reads
  reported.

### 12.11 Predictions (written before any arm of this study was read)

- R-GIGO: PASS (the pilot's 12-state identity held; A2 is the one that could surprise).
- R-MOVES: **MOVES-SHARED** is my modal prediction (≈ 45 %), MOVES-BINDING ≈ 30 %, NECESSARY-NOT-SUFFICIENT ≈ 10 %,
  REFUTED ≈ 10 %, NOT-DETECTED ≈ 5 %. Reason: the pilot's stratum read points at moves, but the fully honest base adds
  spreads and items the pilot never hid, and the never-seen mons carry their own share.
- G (fully honest) larger than the pilot's R-INFO by +0.005 to +0.02.
- R-RECOVER: PARTIAL; PRIOR-MISMATCH not detected; HONEST-SHARPENING-HELPS not detected.
- R-EVAL: EVALUATOR-NOT-BINDING in fair worlds, INFORMATION-BOUND support (the hand eval also collapses).
- R-WIDTH: WIDTH-FLAT (the pilot's fair K-curve: −0.001 / +0.004 / +0.007 at K = 1 / 4 / 8).
- R-NET: NET-NOT-DETECTED or NET-CONFIDENCE-ONLY for the policy; CRITIC-NOT-DETECTED (the X5 oracle arms' +0 for the
  set over the species).
- M-BELIEF IMPROVING; M-STABLE STABLE.

### 12.12 Amendments (dated; each says what had been read)

**Amendment 1 (2026-10-11, after the audits A1–A6 and the descriptive reads D1–D9 were read; no arm of §6 / §7 had
been read).** The audits found the row gate weaker than the searched side's protocol (≈ 4 % of worlds), a
construction-draw leak (A6), and clause-illegal world teams (A4); R-GIGO FAILS (§16.4). Changes: (a) every §6 sampler
(F and its derivatives, P re-built, POOL, POOLSETS, the dose / placebo / MAP families) gates on the searched side's
protocol bytes AND rows, parses every public line naming an opponent move as a reveal, rejects gen3ou-clause-illegal
teams, and draws never-seen genders independently of species; (b) the pilot's Phase 2 base P is RE-BUILT with the
fixed builder and re-played (≈ 2 × 2.6 CPU-h) before any share uses it, and the old P rows are reported beside it;
(c) §13.1's hand-over spec gains the same three requirements; (d) R-GIGO is re-read on the fixed builder, with the
teeth. Nothing else in §12 changes.

---

## 13. Independent replication and the red team (owner, 2026-10-11: "a GIGO in the instrument is the most likely way this goes wrong")

### 13.1 The independent implementation (a separate agent, blind to this study's code and results)

The orchestrator dispatches it with ONLY the following spec, the registered states and the pin checkouts.

**Spec for the second implementation (hand-over text).**
1. *Inputs:* for each checkpoint, the registered states (`value_disc_pilot_2026-10-11/states_<cp>.json`) and their
   bank (`~/gen3ai_archive/value_disc_pilot/<cp>/bank`: each decision's battle, side, command index; the battle's
   packed teams, seed and command log). The Phase 1 truth rows are an INPUT for the referee only.
2. *The referee:* T(a) = mean over truth seeds 16–39 of the win-scale outcome (win 1, loss 0, tie or truncation ½)
   of playing action a at the root then the checkpoint GREEDY on both sides to the end, in the TRUE battle. Re-derive
   it from the rows' outcome strings yourself; do not reuse any scoring code.
3. *What the searched side has seen at the root:* from its OWN protocol stream up to the root: the opponent species
   that switched or were dragged in; for each, the moves it was seen to use; items / abilities named by `-item`,
   `-enditem`, `-ability` or a `[from] item:` / `[from] ability:` tag.
4. *A fair world (class F of §6.1):* the opponent team with every unrevealed fact re-drawn from the Smogon prior
   (`agents.gen3_data.priors` at the pin), the never-seen slots whole (species ∝ the teammate prior given the
   revealed species), and ACCEPTED only if replaying the world's command log to the root gives the searched side
   byte-equal observation rows at every one of its decisions AND a byte-equal protocol stream (everything that side
   was sent), with no gen3ou clause broken (Amendment 1). A move is "seen" if any public line names it for that mon,
   not only `|move|`. Past opponent choices that name a re-drawn move or a never-seen species must be rendered so the
   world can execute them (decide how; document it).
5. *The playouts:* every legal action × 8 worlds, world j with dice seed j of `truth.turn_seeds(id, 48)`, the
   checkpoint greedy on both sides to the end; IMP of LEAF-K8 = T(argmax of the world-mean outcome) − T(top1), ties
   toward the policy's order.
6. *Also:* class P (the pilot's Phase 2 base) and F+M (F with the true seen moves), so that G_P, G and s_M can be
   compared with this study's.
7. *Deliverable:* per state and arm, the 8 world teams, the outcomes, the IMP; the tables of §12.3's B2–B4 for M.
8. *Forbidden:* reading `determinize.py`, `src/main/value_search/worlds.py`, this study's scripts or any result.

### 13.2 The red team (after the results; the orchestrator's dispatch, at the highest tier)

A fresh reviewer briefed with this doc, the raw rows and the scripts, NOT the author's write-up, with the mandate to
break the conclusion. Its checklist:
1. re-derive every verdict from the raw rows with its own code;
2. GIGO: plant defects and confirm the audits catch them; probe the samplers' corners (Hidden Power, formes, a
   Transform user, Baton Pass, a choice lock carried across a re-drawn slot);
3. attribution: interactions (request F+M+X), the other scorers (TRUNC-4, the hand eval), the second referee, half
   the truth seeds;
4. statistics: per-checkpoint heterogeneity, the exclusions, the loss-enriched quota, multiplicity across the many
   descriptive reads;
5. meaning: the X5 oracle arms, Foul Play, the pool-vs-prior confound, the maturity guard;
6. write its section into this doc: each attack and the data that answers it, or the downgrade.

**A conclusion may steer research** (anything beyond §10.5's cheap unlocks) only when R-REPLICATE agrees, the red
team's section is in this doc with every attack answered or the verdict downgraded, and the owner has read it.

---

## 14. What each conclusion would MEAN for the roadmap, and what each path costs

| verdict pattern | meaning | the path | its cost |
|---|---|---|---|
| MOVES-BINDING + BELIEF-RECOVERABLE + NET-USES (and CONFIRMED at plateau) | knowing the moves pays for search AND the policy, and a better honest belief can buy much of it | MOVE INFERENCE: stronger move-belief supervision (weight, set-level / combinatorial targets), pool-aware learning (the model may learn the pool; priors stay Smogon), belief-sampled search worlds; the §8 MOVES fork as the training-level check | weeks: 2–4 agent-weeks of builds + GPU A/Bs (≈ 25–50 GPU-h) |
| MOVES-BINDING + LARGELY-IRREDUCIBLE | the information matters but no honest belief can have it | stop investing in move inference beyond cheap wins; information-set-aware search (robust actions across worlds, the per-turn matrix game, `design_value_discrimination.md` §14.4), oracle playouts as per-action VALUE labels (§13.8) | the §13.8 / §14 builds (≈ 1–3 agent-weeks) |
| MOVES-BINDING + NET-IGNORES or NET-MISUSES | search could use the moves, the network cannot | an ARCHITECTURE question first: route the move belief into decisions (the op's attacker pricing, the intent pointer); re-read the X5 oracle-full null as a network limit; the §8 fork decides whether training fixes it | §8 (≈ 14 GPU-h) then an architecture A/B |
| PRIOR-MISMATCH | part of the gap is Smogon's mismatch to our pool; on the ladder it may be smaller | measure on ladder-like teams (the away bank, human teams) before investing; the training pool's realism becomes a research question | small (a re-run on another bank, ≈ 1 day CPU) |
| S-BINDING (spreads) | exact stats / speed / KO thresholds bind | spread inference from observed damage and move order (an exact damage-calc posterior; the obs constrains it); `design_hand_computed_features.md` exact-fact rule | ≈ 1–2 agent-weeks |
| I-BINDING (items) | Choice Band and unused berries bind | item inference (cheap: the item belief exists; P(CB) from damage) | days |
| U-BINDING (the never-seen mons) | team identification binds | the species belief (R-BELIEF was null: the network's species belief does not help search today), the pool posterior as the measurement | ≈ 1–2 agent-weeks |
| EVALUATOR-BOUND | our critic, not our information, is the constraint | value-side work: oracle-label per-action Q heads (Phase 3), a sibling-level trunk signal, the hand eval as a leaf now (X3) | ≈ 1–2 agent-weeks |
| WIDTH-HELPS | width substitutes for knowledge | cheaper per-world search (truncated leaves, the GPU-batched harness, S1 pairing) | ≈ 1 agent-week |
| MOVES-SHARED / NON-ADDITIVE / NOT-DETECTED | no single lever | general belief quality and the evaluator in parallel; search is not the next big step | — |
| TRANSIENT (the v8 guard) | the gap is moving with training | nothing multi-week; re-read on X26 at plateau | the plateau replication (≈ 40 CPU-h) |
| R-GIGO FAIL | the pilot's honest numbers are void | fix the sampler, re-run Phase 2 and the study | ≈ 1–2 days |

---

## 15. The COMPUTE PLAN

**Unit costs (measured):** a fair-world playout to the end 0.245 CPU-s (Phase 2, prod: 14.5 CPU-s per state for 8
worlds × all actions; no cross-world row dedup); world building 1.2 CPU-s per state for 16 Smogon worlds (F's
per-mon screening is estimated at ≤ 5 CPU-s per state, UNVERIFIED); an oracle Phase 1 ≈ 4 CPU-h per checkpoint.
**Per fair arm per checkpoint ≈ 600 × 14.5 s + worlds + the honest critic ≈ 2.6 CPU-h.**

| tier | contents | CPU-h |
|---|---|---|
| **now (this unit)** | audits A1–A6; the model-input axis (17 arms × 3 checkpoints, root + one-ply successors); policy drift; belief maturity; descriptive reads; the pool-vs-prior read | ≈ 15 |
| **Tier 1 (decisive)** | on end2 and prod: F, F+M, O−M, D50, D90, W90, MAP100, POOLSETS, POOL (9 arms × 2) | ≈ 47 |
| **Tier 2 (the rivals and the curve)** | F+U, F+I, F+A, F+S, F+Usp, O−U, O−I, O−A, O−S (9 × 2); D25, D75, W50, W100, MAP90 (5 × 2); P16 (2 × ≈ 2.6); the second referee (2 × ≈ 2); the hand eval: truncated re-plays in O, P, F (3 × 2 × 1.3–4.8, measured on the contended box) | ≈ 95–105 |
| **Tier 3 (replication and maturity)** | the third checkpoint: Phase 1 + Tier-1 arms (≈ 27); the trajectory at 5M / 10M: Phase 1 + 6 arms each (≈ 2 × 20); the away bank: build + Phase 1 + 5 arms (≈ 18) | ≈ 85 |
| **the independent implementation** | F, P, F+M on 2 checkpoints | ≈ 16 |
| **plateau replication (later, on X26)** | §10.5's list | ≈ 40 |
| **total before the plateau** | | **≈ 260–270** |

- **On 3 physical cores (6 logical, now):** Tier 1 alone ≈ 10–12 wall-hours; everything ≈ 50+ wall-hours. Only the
  "now" tier runs on them.
- **On the full box after the closing test** (16 logical; ≈ 12 workers at `nice -n 19` beside whatever the next GPU
  run needs; ≈ 9 effective cores): Tier 1 ≈ 5 h, Tiers 1–3 + the replication ≈ 26–28 wall-hours.
- **A GPU-batched playout harness** (every pending row of many states in one forward through the inference service):
  the fair playout's 0.245 CPU-s is mostly the CPU forward of the end-state network; with the forward on the GPU a
  playout costs its sim + encode, plausibly 0.02–0.05 CPU-s (UNVERIFIED: measure on 20 states first), i.e. a 5–10×
  cut: Tiers 1–3 in ≈ 4–6 wall-hours plus ≈ 2–4 GPU-hours. **Build ≈ 1 agent-day. Recommended IF Tiers 2–3 are run in
  full** (it pays back after ≈ 40 CPU-h); for Tier 1 alone, CPU suffices. The GPU window is the owner's queue call.
- **Order after the closing test:** R-GIGO and the independent implementation's F / P first (they gate everything);
  then Tier 1; read Tier 1 (blinded); the size rule; then Tiers 2–3; then the red team.

---

## 16. RESULTS SO FAR (the cheap parts; filled when read, each read after this registration landed)

### 16.1 The registered descriptive reads D1–D8 (read 2026-10-11, after the registration `e25a5240`; `descriptive_reads.json`, `descriptive_reads_det_b.json`)

**Plain language.** The pilot's honest search loses its whole edge as soon as ONE opponent Pokémon is still unseen;
when all six have been seen it keeps about HALF of the oracle's edge. So the never-seen Pokémon carry a large part of
the gap, and the seen Pokémon's hidden moves carry the rest. The pilot's "with nearly all species seen, fair keeps
only ~¼" pooled two very different groups.

Reproduction check: G (all states) is +0.072 [+0.057, +0.087] (end2) / +0.066 [+0.053, +0.080] (prod): the pilot's
R-INFO exactly. Pooled over both seeds (1,157 states; g = oracle − fair LEAF-K8 improvement):

| read | stratum | n | G [95 %] | oracle IMP | fair IMP |
|---|---|---|---|---|---|
| **D1** unrevealed opponent slots | **0** (all six seen) | 229 | **+0.032 [+0.015, +0.051]** | +0.066 | **+0.035** |
| | **1** | 224 | **+0.070 [+0.046, +0.096]** | +0.068 | **−0.002** |
| | 2 | 209 | +0.094 [+0.066, +0.123] | +0.097 | +0.003 |
| | 3 | 183 | +0.074 [+0.051, +0.099] | +0.063 | −0.011 |
| | 4–5 | 312 | +0.077 [+0.058, +0.096] | +0.077 | −0.000 |
| **D3** the opponent active's unseen move slots | 0 / 1 / 2 / 3–4 | 16 / 103 / 311 / 727 | +0.049 / +0.046 / +0.060 / **+0.077** | | |
| **D7** the oracle's pick | a SWITCH / a move | 400 / 757 | **+0.112** / +0.046 (end2 +0.130 / +0.043; prod +0.096 / +0.050) | | |
| **D8** the gap at a 4-turn leaf, over the full leaf's | TRUNC-4-K8 / TRUNC-4-K4 / CRITIC | | 0.38 [0.19, 0.61] / 0.49 / 0.19 (end2); 0.54 [0.33, 0.76] / 0.43 / 0.16 (prod) | | |

- **D2** (the seen mons' total unseen move slots): flat overall (+0.069 / +0.082 / +0.075 / +0.065 at 0–2 / 3–5 / 6–8
  / 9+); within 0–1 unrevealed every state is in the 9+ bin, so the registered H-MOVES expectation "D2 rises within 0–1
  unrevealed" COULD NOT BE READ with the registered bins (a design miss, named). POST-HOC finer bins within 0
  unrevealed (`posthoc_reads.json`, labelled post-hoc, no verdict): 11–14 unseen slots G +0.001 [−0.009, +0.011] (n
  61; the fair leaf matches the oracle) vs 15+ G +0.043 [+0.020, +0.068] (n 168): when enough of the seen mons' moves
  are revealed, the all-seen gap vanishes. That is the H-MOVES signature inside the all-seen stratum, confounded with
  game progress.
- **D3** rises with the active's hidden moves on end2 (+0.025 → +0.088) and is flat on prod (+0.069 / +0.068 / +0.066):
  mixed. Within 0 unrevealed (post-hoc) it is flat (+0.022 / +0.031 / +0.034).
- **D4:** no difference by opponent class (bot +0.066 / +0.065 vs snapshot +0.079 / +0.068); larger in traced LOSSES
  (+0.079 / +0.084) than wins (+0.065 / +0.048); smaller late (turns 26–40: +0.051 / +0.027).
- **D5** concentration (split-half: ranked on truth seeds 16–27, measured on 28–39): the top 10 % of states carry
  0.54 [0.41, 0.66] / 0.45 [0.33, 0.58] of G, the top 25 % 0.87 / 0.85. Concentrated, but below the registered 80 %-in-
  the-top-10 % restatement bar.
- **D6** the excluded 3.5 % have a HIGHER oracle IMP (+0.130 / +0.103 vs +0.079 / +0.070); counting them with g = 0
  (worst case) R-INFO is still +0.069 [+0.055, +0.085] / +0.064 [+0.051, +0.077]: H-STAT's exclusion arm is KILLED
  (lower bound > 0.04). Outcome-reweighted G +0.065 / +0.055 (within 0.02 of raw): the loss-enrichment arm is killed.
- **D7:** the gap lives disproportionately in SWITCH decisions (≈ 60 % of G on end2 from a third of the states).
  Which Pokémon to bring in depends on what the opponent can do and who else it has; consistent with moves AND with
  the never-seen mons.
- **D8 (H-HORIZON):** 40–55 % of the full-leaf gap is already present at a 4-turn leaf; the registered kill bar
  (≥ ½) straddles: neither killed nor supported.
- Phase 2b's belief-species worlds give the same pattern (`descriptive_reads_det_b.json`: D1 0 / 1 / 2 unrevealed G
  +0.032 / +0.068 / +0.097).

**Caveat (added after §16.4):** these reads use the pilot's Phase 2 worlds, which R-GIGO marks formally VOID until the
fixed builder re-builds them; the post-hoc sensitivity in §16.4 finds G and the D1 split unchanged when every state
touched by a known defect is dropped.

**What D1–D8 do to the hypotheses (descriptive, no verdict):** H-UNSEEN is very much alive (one unseen mon wipes the
honest gain); H-MOVES is alive inside the all-seen stratum (about half the oracle's edge there, vanishing when enough
moves are revealed, post-hoc); H-STAT's exclusion and reweighting arms are killed; H-HORIZON is unresolved. The
pilot's sentence "it is the MOVES, not the species" is not supported as stated: the never-seen mons matter at least
as much in this base, and the factorial (§6) must apportion U against M.

### 16.2 D9, the POOL-vs-PRIOR confound (read 2026-10-11; `poolprior_read.json`)

**Plain language.** Our pool's sets are about as predictable from Smogon usage as real human sets are, so the
pilot's low move-hit rate is NOT a pool artefact: guessing an opponent's hidden moves is genuinely hard in gen-3 OU.
Even if every opponent's set were DRAWN from the Smogon prior itself, a guessed world would hold only about 40 % of
the hidden moves: that is the prior's own spread, the game's intrinsic hidden information.

- **Method check:** the read reproduces the pilot's numbers on its own 1,200 states (slot hit 0.159 / 0.161, set
  recall 0.445 / 0.447).
- **The prior-as-truth ceiling:** sets drawn from the Smogon prior are predicted NO better than the pool's (slot hit
  with 2 revealed 0.193 vs pool 0.197; set hit 0.380 vs 0.389; top-4 recall with none revealed 0.609 vs 0.650).
- **Real human sets** (a 40,000-log sample of 376,410 public gen-3 OU replays, 2026-05-18 → 09-23; 32,183 mons that
  revealed all four moves): top-4 recall human − pool −0.029 [−0.036, −0.021] (the pool is the MORE predictable);
  censored per revealed move +0.007 [+0.004, +0.011] usage-weighted (≥ 1500-rated players about the same). Items:
  P(true | species) 0.802 pool vs 0.801 Metamon; abilities 0.918 vs 0.906.
- **Spreads** are where the pool is less Smogon-like (exact-spread probability 0.049 vs the curated sample teams'
  0.066); no honest human reference exists (replays never reveal spreads; the Metamon corpus fills them from usage).
- **Teammates:** the prior predicts the pool's teammates BETTER than human teams' (mean rank of a true teammate 9.75
  vs 15.8).
- **For the hypotheses:** H-PRIOR-MISMATCH's D9 half is met against it (for moves and items the difference CI favours
  the pool); the POOLSETS arm (§6.4) completes the kill test. For C2: a population prior is near its own ceiling, so
  any recoverable part of G must come from game-conditioned inference (the network's belief, §16.3) or team
  identification (POOL), which the ladder does not offer.

### 16.3 MATURITY: policy drift and belief maturity, 1M–15M (read 2026-10-11; `policy_drift_summary.json`, `belief_maturity.json`)

**Plain language.** At 15M both policies are still changing fast and their beliefs are still learning: per million
steps each policy moves ≈ 0.1 nats of KL and changes its top action on 15–22 % of probe states, with no slowdown from
5–10M to 10–15M (part of it is back-and-forth). These checkpoints are NOT plateaued, which is exactly the owner's
v8 worry: every verdict here is PROVISIONAL "at 15M".

| meter | end2 | prod |
|---|---|---|
| policy drift per 1M, 5–10M: KL mean / flip rate | 0.107 / 14.3 % | 0.095 / 22.1 % |
| policy drift per 1M, 10–15M: KL mean / flip rate | 0.097 / 15.2 % | 0.104 / 22.3 % |
| 15M vs ≈ 10M earlier: KL / flip (≈ 1.0 nats if the per-1M steps added up) | 0.64 / 52 % | 0.71 / 47 % |
| move belief, expected recall at 15M (Δ 10M→15M) | 0.459 (+0.017 [+0.015, +0.020]) | 0.419 (+0.006 [+0.004, +0.008]) |
| move belief vs the Smogon prior at 15M: expected / top-k recall | +0.034 / +0.022 | −0.004 / +0.041 |
| item belief P(true item) at 15M (Δ; vs Smogon) | 0.779 (+0.001; +0.105) | 0.770 (+0.015; +0.105) |
| species belief, mass on true hidden species (Δ) | 0.441 (+0.032) | 0.470 (+0.030) |

- **M-BELIEF = IMPROVING** on both seeds (the 15M − 10M move-recall lower bounds +0.015 / +0.004 > 0): the v8 guard's
  first condition holds; whether it fires waits for M-STABLE (the trajectory arm, §10.3).
- The network's MOVE belief barely beats the Smogon prior (end2 +0.03, prod ≈ 0 in expected recall), consistent with
  D9: per-game move inference is where any C2 gain would have to come from, and it is small today.
- **The plateau yardstick (§10.5) is now numeric:** a plateaued checkpoint needs a per-1M KL ≤ 0.049 and a flip rate
  ≤ 7.6 % (half of end2's 10–15M means) over three consecutive 1M steps.
- FINDINGS: the pin checkouts have no Showdown build, so the drift collector's team validation was redirected to the
  main checkout's byte-identical `validate_team.js` (`policy_drift_scripts/`); the two runs' probe sets differ (each
  from its own 4M policy), so drift compares within a run only; "item revealed" is approximated from protocol tags.

### 16.4 The sampler AUDITS and R-GIGO (read 2026-10-11; `audit_*_<cp>.json`, bulk `~/gen3ai_archive/hidden_info_binding/<cp>/audit/`)

**Plain language.** The world builder holds exactly the right hidden state for every Pokémon our side has seen, and
the dice after the decision depend only on the playout seed, as designed. But its consistency check compares our
OBSERVATION ROWS, which is weaker than what our side actually saw: about 4 % of the played worlds disagree with our
own battle log in small ways (a Pokémon's gender shown on switch-in, a Focus Punch "tightening" line, Beat Up hit
sizes), and 2–3 % of world teams break gen-3 OU clauses (Speed Pass, One Boost Passer, Accuracy Trap). By the
registered rule that is a FAIL. The defects are small and pull in both directions; dropping every affected state
leaves the pilot's gap unchanged.

| audit | end2 | prod | verdict |
|---|---|---|---|
| **A1** builder identity (the TRUE team through the builder: gate on try 1, rendered prefix replays identically for both sides, Phase 1 outcomes on seeds 0–7) | 150 states / 1,100 actions, 0 mismatches | 150 / 1,101, 0 | PASS |
| **A2** field diff at the root (the opponent's own row + its last request, every played world of det and det_b, ≈ 18,450 worlds) | 1 + 1 states with UNEXPECTED cells | 1 + 1 (3 + 1 cells) | **FAIL** (registered: 0 unexpected) |
| … of which: every SEEN mon's exact HP, status, stats, item, ability, kept moves and their PP | 0 differences (≈ 17–18k mon comparisons per run) | 0 | the hidden state that matters is clean |
| … the UNEXPECTED cells | the event-window MAGNITUDE of a past hit on a seen mon off by 1 HP (326/331 vs 325/331): a never-seen slot of a different gender class changes the turn-0 gender draws (`rust_sim/src/turn/driver.rs:691`) and shifts the pre-root dice | same | — |
| … worlds whose SEARCHED side's PROTOCOL differs from the truth (not caught by the row gate) | 224 + 245 (70 + 65 states): gender tokens in switch details; Focus Punch `-singleturn` (the mon fainted by the root in every end2 case); one Beat Up hit size | 162 + 150 (46 + 44 states) | FINDING |
| **A3** re-gate determinism (100 worlds) | 100 % | 100 % | PASS |
| **A4** legality of world teams | 0 Species Clause / multi-HP / engine-banned / HP-IV / EV violations; **≈ 2–3 % of teams break gen3ou clauses** (end2 det 116 of 4,624); learnability and abilities UNCHECKABLE (the pin's learnset lacks pre-evolution and egg moves: true teams "fail" it ≈ 2.2 times each; no ability lists) | prod det 127 of 4,632 | **FAIL** (> 1 % clause-illegal ⇒ fixed before §6) |
| **A5** root dice: post-root dice depend ONLY on the playout seed (`search/playout.rs:372` → `game.rs:154-158` → `bridge.rs:1858-1861` replaces `st.prng` wholesale; a two-battle-seed control: 90 / 90 branches identical, the no-reseed control differs in every pair; a bench-mon swap leaves the root turn identical 61 / 61, 67 / 67) | PASS | PASS | PASS (caveat: durations rolled BEFORE the root, e.g. sleep, carry over from the prefix) |
| **A6** the gate's conditioning (descriptive) | re-drawn seen-mon moves are effectively UNconditioned (KL 0.012 = the matched-noise floor); the gate LEAKS the never-seen slots' gender class: accepted worlds match the true count of random-gender mons 98.2 % vs 53 % ungated (toward the truth) | 97.5 % vs 51 % | — |
| teeth (planted defects, 10 states per checkpoint; `audit_teeth_<cp>.json`) | A1: a kept seen mon's item swapped 10/10; a rewritten move slot rendered OFF BY ONE was invisible to the replay check in 9/10 states (a rewritten choice is one never executed, so its slot leaves no trace): a TEXT-level invariant was added (every rendered rewritten token equals the original command up to `toid`, counts equal the builder's leak counts; 0 violations on all 600 states, 187 moves + 3 switches), after which the plant is caught 10/10. A2: a seen mon's item swapped, its EVs changed: caught in every world (80 / 80). A3: an opponent move command changed, a switch retargeted: 10/10 (mostly by core refusal) | same (191 moves + 1 switch) | 6 / 6 plants caught after the A1 fix |

**R-GIGO = FAIL** (A2's unexpected cells; A4's clause-illegal teams; the teeth now pass). Under §12.1 the pilot's
honest-world numbers are formally VOID until a fixed sampler re-runs them, and the factorial does not start on the
current builder. **POST-HOC sensitivity** (`gigo_sensitivity.py`, labelled, no verdict): dropping every state with
ANY protocol-inconsistent world or unexpected cell (end2 71 of 578, prod 47 of 579) leaves G at +0.071 [+0.055, +0.087]
/ +0.068 [+0.055, +0.083] (was +0.072 / +0.066) and the D1 split unchanged (0 unrevealed: +0.039 / +0.018; 1: +0.066 /
+0.077). The defects are mixed in sign: the gender leak pulls worlds toward the truth (helps honest search), an
unparsed Focus Punch reveal and clause-illegal teams make worlds worse. They are far too small to create or erase a
7-point gap, so the pilot's qualitative reading survives; its numbers are re-confirmed only by the fixed sampler.

**The fixes, now preconditions of §6 (Amendment 1, §12.12):** (1) gate on the searched side's PROTOCOL bytes as well as
its rows (gender tokens included, or genders fixed per slot independently of the species draw so the gate no longer
conditions on construction draws: the A6 leak); (2) parse every public line that names an opponent's move outside
`|move|` (Focus Punch `-singleturn`, Taunt / Disable `cant`, Encore / Disable `-start`) as a reveal; (3) reject world
teams that break a gen3ou clause (`agents.gen3_data.format_spec`); (4) the independent implementation (§13.1) is built
to the fixed spec. ≈ 0.5 agent-day.

**UPDATE 2026-10-11 23:15 (the pilot's builder owner; `design_value_discrimination.md` §7.10 Amendment 8): the FIXED
builder (v2) is built and R-GIGO re-read on its worlds = PASS** (these audits re-run with `HIB_AUDIT_DIRS` /
`HIB_AUDIT_TAG=_v2`, files `audit_*_<cp>_v2.json`): A1 0 mismatches, A2 0 unexpected cells and 0 protocol-inconsistent
worlds (was 469 / 312), A3 100 %, A4 0 clause / ban violations in sampled parts (was ≈ 2.5 %), teeth 6 / 6 plus two
gender plants 20 / 20. v2 also fixes a defect these audits did not name: a past opponent move choice is now credited
to the mon that made it (v1 kept a choice by name when another mon had the move seen, so worlds whose actual chooser
lacked it were refused). The gender leak is REDUCED, not removed: a world still needs at least as many gender-ratio
never-seen mons as the true draws (accepted P(m = r) 0.67 / 0.64 vs the proposal's conditional 0.57; v1 0.98 / 0.97);
only an engine-side draw pad removes it. Detail: `measurements/value_disc_pilot_2026-10-11/README.md` §13. The v2
worlds of every honest arm are `~/gen3ai_archive/value_disc_pilot/<cp>/det2*`. **2026-10-12: every arm is PLAYED on
v2** (rows `det2*/rows_*.jsonl`; the pilot's Phase 2 base P re-built and re-played is `det2`, Amendment 1 (b)): the
pilot's honest verdicts hold (G on v2 = oracle − A′ LEAF-K8 +0.070 / +0.065, was +0.072 / +0.067), and Phase 2c's
true-seen-moves arm (`det2_ct`: the seen mons' TRUE sets on belief-species worlds; NOT this study's F+M, whose base hides more) closes ½ (end2) / ⅕ (prod) of the gap over
the belief-species worlds (`value_disc_pilot` README §14).

### 16.5 The MODEL-INPUT axis: R-NET and CRITIC-USES (read 2026-10-11; `result_model_input_blinded.json` frozen at `8df724e0` before unblinding, `result_model_input.json`)

**Plain language.** Told the TRUE moves of every opponent Pokémon it has seen, with 95 % confidence, the network
changes its choice in about one decision in eight, but the changed choices are NOT better: +0.1 points of win
probability per decision [−0.3, +0.5], against the oracle search's +7. A confidently WRONG set of moves changes even
more decisions and does not help either; it makes the critic's one-ply search about a point worse, so the critic does
read move content, but the true content does not improve it. The network, as trained, does not turn knowledge of the
moves into better decisions.

Validity: 600 states × 3 checkpoints, 0 refused; the regenerated one-ply successors reproduce Phase 1's critic values
(worst 4.6e-6 / 1.1e-6); the base critic's one-ply IMP reproduces Phase 1's (+0.026 / +0.012); the hook at p = 0 is
an exact identity on every state (identity_check PASS); 0 top-1 mismatches against the registered picks. Disclosure:
the arm key was written by the harness agent and seen by the author before the blinded result existed; the blinding
here protects the frozen scripts and rules, not the author's knowledge of the codes.

| arm (pooled end2 + prod, 1,200 states) | flip rate | KL (nats) | IMP of the policy's pick | IMP of the one-ply critic search | critic top1 \| top2 accuracy |
|---|---|---|---|---|---|
| base (the network as trained) | — | — | 0 | +0.019 [+0.011, +0.028] | 0.583 |
| TRUE p = 0.6 | 8.4 % | 0.029 | +0.004 [+0.001, +0.008] | +0.017 | 0.594 |
| TRUE p = 0.8 | 11.4 % | 0.049 | +0.002 [−0.001, +0.007] | +0.017 | 0.603 |
| **TRUE p = 0.95 (primary)** | **13.3 %** | 0.067 | **+0.001 [−0.003, +0.005]** | +0.017 | 0.600 |
| TRUE p = 1.0 | 13.8 % | 0.074 | +0.002 [−0.003, +0.007] | +0.018 | 0.601 |
| PLACEBO (wrong moves) p = 0.95 | 19.6 % | 0.159 | −0.003 [−0.008, +0.003] | +0.011 [+0.003, +0.020] | 0.572 |
| SELFMAP (its own top moves) p = 0.95 | 10.3 % | 0.037 | −0.002 [−0.006, +0.002] | +0.021 | 0.593 |
| TRUE-HYP p = 0.95 (also the true never-seen mons' moves where a hypothesis names one) | 13.9 % | 0.068 | +0.003 [−0.002, +0.008] | +0.017 | 0.600 |

| registered contrast (p = 0.95) | end2 | prod | pooled | end2_s2002 (replication) |
|---|---|---|---|---|
| IMP_POL(TRUE) | +0.003 [−0.003, +0.009] | −0.001 [−0.007, +0.006] | +0.001 [−0.003, +0.005] | +0.010 [+0.004, +0.016] |
| IMP_POL(TRUE) − IMP_POL(PLACEBO) | +0.004 [−0.004, +0.012] | +0.003 [−0.005, +0.010] | +0.004 [−0.002, +0.009] | +0.002 [−0.007, +0.010] |
| IMP_POL(TRUE) − IMP_POL(SELFMAP) | +0.006 [+0.002, +0.011] | −0.000 [−0.006, +0.007] | +0.003 [−0.001, +0.007] | +0.004 [−0.001, +0.010] |
| IMP_CRIT(TRUE) − IMP_CRIT(base) | −0.003 [−0.009, +0.003] | −0.001 [−0.008, +0.006] | −0.002 [−0.007, +0.003] | −0.001 [−0.007, +0.005] |
| IMP_CRIT(TRUE) − IMP_CRIT(PLACEBO) | +0.007 [−0.002, +0.018] | +0.005 [−0.004, +0.014] | +0.006 [−0.000, +0.013] | −0.010 [−0.021, +0.001] |
| **R-NET** | **NOT DETECTED** | **NOT DETECTED** | **NOT DETECTED** | NOT DETECTED |
| **CRITIC-USES** | **NOT DETECTED** | **NOT DETECTED** | **NOT DETECTED** | NOT DETECTED |

- **Neither IGNORES nor MISUSES nor CONFIDENCE-ONLY:** the network reacts (flips 12–15 %, well above 2 %), its picks
  do not get worse, and the CIs are too wide to call TRUE ≡ PLACEBO ≡ SELFMAP inside ±0.005. The pooled upper bound
  says perfect seen-mon move knowledge is worth at most ≈ +0.5 points per decision to the frozen policy (+0.65 at
  p = 1): under a tenth of the oracle search's edge.
- **The replication network's positive TRUE read is not content:** on end2's states `rb_es_end2_s2002`'s pick gains
  +0.010 under TRUE but also +0.008 under the PLACEBO and +0.005 under SELFMAP (its own picks scored by end2's
  referee: any nudge toward end2's choices pays). The placebo control did its job.
- **Descriptive:** the smallest dose (0.6) is the only TRUE dose clear of 0 (+0.004 [+0.001, +0.008]); the effect
  does not rise with confidence. The PLACEBO costs the critic search ≈ 0.8 points pooled (+0.019 → +0.011; at p =
  1.0 TRUE − PLACEBO +0.009 [+0.002, +0.016]): the critic reads move content, and true content does not help it.
- **For the claim:** C3 (usability) is NOT supported on the frozen network, in line with the X5 oracle arms (a
  network TRAINED with the whole set gained nothing over species). Whether a network can LEARN to use the moves is
  §8's question (the forks, queued).

### 16.6 Where the claim stands after the cheap reads, and the author's honest confidence

| part | evidence so far | standing |
|---|---|---|
| **C1 attribution (search)** | D1: with all six mons seen, honest search keeps half the oracle's edge, and the remaining half shrinks to ≈ 0 when enough of the seen mons' moves are revealed (post-hoc); ONE unseen mon wipes the honest edge. The pilot never hid a seen mon's item, ability or spread | **shared**: the seen mons' moves carry a real part (≈ 0.03 of ≈ 0.07 where everything else is known), the never-seen mons at least as much; items / abilities / spreads unmeasured. The factorial decides |
| **C2 recoverability** | D9: Smogon sets are as guessable as human sets, and sets drawn from the prior itself are no more guessable (the difficulty is intrinsic); the network's own move belief beats Smogon by +0.03 (end2) / 0 (prod) and improves slowly | **unlikely on the ladder** for moves; on our pool, team identification (POOL) is untested |
| **C3 usability (network)** | §16.5: told the true moves at 95 %, the frozen network's decisions are no better (≤ +0.5 points), its critic no better; X5: a trained oracle-set network +0 over species | **not supported** for the frozen network; the forks test learning |
| maturity | policies still move ≈ 0.1 nats per 1M with 15–22 % flips; the beliefs still improve | every reading PROVISIONAL "at 15M" |
| instrument | R-GIGO FAILS by its rule (≈ 4 % of worlds protocol-inconsistent; 2–3 % clause-illegal teams); the known defects cannot move G (post-hoc) | fix before the factorial (Amendment 1) |

**Honest confidence (the author, 2026-10-11, before any factorial arm):** that the opponent's hidden MOVES are THE
binding constraint in the owner's sense (C1 with a majority share AND a payable path through C2 or C3): **≈ 15 %**.
That hidden information about the opponent's TEAM binds honest SEARCH: ≈ 85 % (the pilot's R-INFO, robust to the
audits). That the moves are a major SHARE of it (≥ ¼) in the factorial: ≈ 60 %; the majority share by both sufficiency
and necessity (MOVES-BINDING): ≈ 25 %. What would raise it: F+M closing ≥ half of the fully honest gap with a null
placebo, and the MOVES fork learning to use them. What would kill it: O−M costing < ¼ of G, or the placebo buying as
much. **Recommendation for the roadmap now:** do NOT open a multi-week move-inference programme on this evidence; run
the factorial's Tier 1 with the fixed sampler after the closing test (≈ 50 CPU-h), the knock-in forks as queued, and
the hand evaluator's arm (R-EVAL) beside it. (The pilot's Phase 3, read the same evening: a per-action value head on
oracle labels does not help a FROZEN body at K = 2 labels, R-QHEAD NOT DETECTED −0.004, 83 % of the label spread dice
noise; `design_value_discrimination.md` §7.13. The value side stays open only with a trained body or a far larger
label budget, so the evaluator question, R-EVAL, matters more, not less.)

---

## 17. Open questions for the owner

1. ~~The §8 knock-in forks: queue them?~~ QUEUED by the orchestrator 2026-10-11 (≈ mid-week, after the closing
   test's verdict branch and the bf16 A/B); the launch-ready design is §8.
2. **POOL and POOLSETS** use the training pool as a MEASUREMENT of the eval opponents' posterior. The owner's rule
   forbids the pool as a shipped prior; does it also forbid it as a measurement arm? (The study assumes not: nothing
   it computes reaches a network or a search at play time.)
3. **The full factorial vs Tier 1 only**, and the GPU-batched harness (≈ 1 agent-day + ≈ 2–4 GPU-h).
4. **The hand-evaluator build (≈ 1 agent-day)** now, so its arm is ready for the post-verdict window?

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-11 | **This doc CREATED and PRE-REGISTERED at the owner's request** ("reproduce knowing the moves as being the critical link … and challenge that … give high mass to the known moves, but don't necessarily make them 100% … its own end state doc, full analysis, many tests, many countering tests"), with the orchestrator's additions the same day (maturity / adversarial robustness and the scope rule; the pool-vs-prior confound, belief maturity, blinding, the game-level check, the knock-in forks; Foul Play's evaluator as a scorer and the Foul Play / X5 counter-evidence; correctness over economy, the independent implementation and the red team) | The claim split into ATTRIBUTION / RECOVERABILITY / USABILITY; a five-class PARTITION of the hidden facts (M, U, I, A, S) with a FULLY honest base F, sufficiency and necessity arms, a move dose with a wrong-moves placebo and an honest MAP control, the pool posterior and pool sets as recoverability measurements, the hand evaluator, a width arm and a second referee; a frozen-network model-input axis (the true moves injected into the ONE published move posterior at 60–100 % mass, placebo and self-MAP controls); deterministic rules with pooled shares and per-checkpoint replication; a v8 guard and a plateau CONFIRM / REVOKE; blinded codes; the independent implementation and the red team as preconditions | attributing within the pilot's leaked base only (it could not see spreads, items or abilities); a 100 %-only reveal (the owner asked for high but not total mass); injecting into the OBSERVATION (out of distribution for a network trained without reveals: the X5 both-sided −8.1 pp); a full-factorial 2⁵ design (32 arms per checkpoint; the one-at-a-time sufficiency and necessity arms plus a targeted interaction follow-up answer the registered question at a sixth of the cost); dropping rival arms for cost (owner: no arm that kills a rival is cut) | §2 (the pilot's construction, `determinize.py`; the X5 oracle arms; the Foul Play axes); `move_accuracy_read.json`; owner and orchestrator 2026-10-11 |
| 2026-10-11 | **The knock-in forks QUEUED (the orchestrator, under the owner's GPU-scheduling delegation) and designed to launch-ready detail** | §8: `--belief-oracle` from the training-only label keys with a startup gate, the pin precondition, the parent (the closing test's adopted seed-2001 final), 5 arms × 3 seeds × +3M with a no-change control and a wrong-moves placebo, the one-sided cross, the K-rules, ≈ 14 GPU-h | an observation-channel reveal (out of distribution for a network trained without it); training from scratch (the X5 arms did that; a fork asks the cheaper question "can it learn to use it from here") | orchestrator 2026-10-11 |
| 2026-10-11 | **The hand evaluator BUILT** (§6.5) | `src/main/hidden_info_study/hand_eval.py` over the searched side's row (so it reads exactly the critic's information), its pinned twin and truncated-playout driver; screens corrected to Foul Play's per-turn counters | a full-state evaluator (more information than the critic: it would conflate evaluator with information) | 73 tests incl. 905 real decisions vs an independent tracker |
| 2026-10-11 | **The cheap parts READ (§16) and AMENDMENT 1 (§12.12)** | D1–D8: one unseen opponent mon wipes the honest edge, all six seen keeps half (the never-seen mons and the seen moves share the gap); D9: the move-guessing difficulty is intrinsic (not a pool artefact); maturity: not plateaued at 15M; R-GIGO FAILS by its rule (protocol-inconsistent worlds, clause-illegal teams) with the gap robust to dropping them, so the sampler fixes become preconditions of the factorial and the pilot's base P is re-built; R-NET and CRITIC-USES NOT DETECTED (the frozen network does not use true move knowledge). The author's confidence that the moves are THE binding constraint: ≈ 15 %. Recommendation: no move-inference programme yet; Tier 1 with the fixed sampler after the closing test, the forks as queued | treating the pilot's honest numbers as unaffected by the audit (the rule says VOID until re-run; the post-hoc sensitivity is reported, not substituted); reading the replication network's positive TRUE read as content (its placebo bought as much) | `research_state/measurements/hidden_info_binding_2026-10-11/`; ledger 2026-10-11 |
| 2026-10-12 | **The study's arms become presets of the ONE information-lever spec** (owner: one DRY implementation for training, search and the model's inputs) | `design_information_levers.md`: every §6 / §7 / §8 arm is a named preset (`spec.PRESETS`), its five-class partition is `spec.STUDY_CLASSES` (S = a seen mon's spread AND exact HP; `counters` told by O alone); the world builder, the training observation and the injection consume the same spec; nothing registered here changes | keeping each arm as bespoke code in the measurement folder (three incomparable mechanisms; the builder leaked) | `design_information_levers.md` §5 |
