# design — ONE BELIEF: the unified posterior, the one reduction, and the one delivery contract

**Status: REFRAME (owner-requested 2026-08-13). Not a build order yet — a restructuring of three
existing docs onto one spine.** It supersedes nothing on disk; it says how
`design_opponent_intent.md`, `design_conditional_opponent_cells.md` and
`design_conditional_execution.md` are three views of one object, and what is actually missing.

---

## 0. The complaint, stated precisely

> *"I feel like we keep dancing around this issue."*

The dancing is structural, not a failure of attention. **The belief system is factored by WHAT IS
HIDDEN. It should be factored by WHAT THE DECISION NEEDS.**

Today, hiddenness is partitioned into five independent subsystems, each with its own module, its
own privileged label key, its own loss, its own masking convention, and its own consumer wiring:

| what is hidden | module | label key(s) | consumer |
|---|---|---|---|
| which mon occupies an unrevealed slot | `BeliefSlots` + `BeliefHead` | `belief_species` | slot tokens (pre-transformer) |
| which moves a mon holds | `MoveBelief` | `belief_moves`, `known_moves` | reinjected into the opp token; `w` into the op |
| the EV/nature spread | `SpreadBelief` | `belief_spread`, `belief_nature`, `belief_ev` | `sb` stats into the op |
| the Hidden Power type | `HPTypeBelief` | `hp_type_label` | composed into the typed channels |
| what they will DO this turn | `α` / `β` (v67) | `opp_action_*` | **none** |

Five heads. Eight label keys. **No shared object.** So:

* "Reason uniformly over unrevealed mons and unrevealed move slots" has no place to live — they are
  different subsystems with different shapes, different masks and different failure modes.
* Every integration conversation stalls on *"consumed by what?"*, because there is nothing to consume.
* Each new hidden quantity costs a full vertical slice: head + label + loss + mask + wiring + gates.
* The **couplings are thrown away** (§1) and then partially re-invented downstream.

The three consumer docs are not competing proposals. They are three readouts of one missing object.

---

## 0b. THE TIERING — and why α must be strictly downstream

**Owner, 2026-08-13.** The three stages are not peers running in parallel. They are **tiers with a
strict dependency order**, and collapsing them is precisely the "dancing" this doc exists to stop:

| tier | question | object | may read |
|---|---|---|---|
| **T0 — RESOLVE** | *what is actually on the board?* | the hidden-state posterior — species, moves, spread, item, HP type | the obs + evidence only |
| **T1 — REASON** | *what follows from that board?* | `pair[a,k,:]` — the physics: damage, KO, status, neutralisation, tempo | T0's output |
| **T2 — DECIDE** | *what should I do, given the near-equilibrium we are in?* | `α`/`β` → the reduction → the action | T0 **and** T1's outputs |

**α is a T2 object and can never be a T0 one.** Predicting "what will they click" *before* resolving
what they HAVE is predicting intent over an unknown action set — a fifth parallel belief head, which
is exactly the failure mode §0 describes. α must consume the resolved board and its consequences.

**The current code already satisfies this, by construction rather than by contract.** α reads
`_seat_out[:, 4:4+K]` — the E4 threat seats, whose header carries the belief weight `w` *and* the
physics (`damage_op.py:1378`) — plus the pooled board. So its INPUTS are already tier-correct. Two
things are missing:

1. **The tiering is accidental, not enforced.** Nothing prevents a future head from reading α at T0,
   or from computing an intent-like quantity off raw tokens. It should be a stated contract with a
   test, the way leak-safety is.
2. **The loop is open.** α's OUTPUT reaches nothing (§2.2). A T2 object that no decision consumes is
   a measurement, not an architecture.

### 0b.1 "Near equilibrium" — what the third tier actually is

The owner's framing — *"a decision on what we should do this turn given the near-equilibrium of
states we think we're in"* — is the honest description of T2, and it is game-theoretic, not
predictive. Both sides choose simultaneously; each choice is best-response to a belief about the
other. That is a fixed point, not a forward computation.

**We deliberately do not solve it at inference.** α depends on the BOARD (which includes our stance)
and never on our *realized action* — reading our own policy logits would be level-3 reasoning and
would create a forward-pass cycle. The acyclic approximation is: board → α → our choice; and the
fixed point is found by **self-play training**, because our opponent is us. That is what makes a
one-ply α coherent rather than naive: we are not claiming to predict a rational adversary's exact
choice, we are claiming a calibrated distribution over it, trained against the same policy.

The `near-` is load-bearing and should stay in the vocabulary. It is why α is graded by
*calibration* (does the distribution match what they did) rather than by *accuracy alone*, and why
its null result is interpretable: a poorly-predicted opponent is a fact about the equilibrium, not
necessarily a broken head.

---

## 1. Stage 1 — BELIEF: one posterior over SETS, not five marginals

### 1.1 The factorization error

In gen 3 the latent object is **not** "a species" plus "some moves" plus "a spread". It is a **SET** —
a discrete, named, jointly-distributed configuration. Choice-Band Tyranitar, Dragon-Dance Tyranitar
and the Sand-trap variant are *different objects*: different moves, different spread, different
behaviour, different answer to every question the policy asks.

The current design predicts four **independent marginals** and re-composes them multiplicatively
where it must (`compose_typed_hp` = `P(HP present) · P(HP type)`). That discards the couplings:

* **species → moveset.** The learnset gate (v65) is a HARD constraint we now apply, but usage
  *correlation* is softer and richer: a Skarmory that has shown Spikes is far more likely to hold
  Whirlwind than Drill Peck.
* **moveset → spread.** A revealed Choice Band implies a very different EV spread than a revealed
  Calm Mind. `SpreadBelief` cannot see the move belief at all.
* **move → move.** Sets are archetypes. Seeing Substitute raises Focus Punch enormously — the
  SubPunch set is one object. Four independent Bernoullis cannot represent that.

**This is not an abstract elegance argument.** It is why `move_recall` sits at 0.58 and why
`spread_largest_bias` needed a structural fix rather than more supervision: each head is estimating
a marginal of a distribution whose *structure* lives in the correlations it was denied.

### 1.2 ⚠️ CORRECTION — we do not have set priors. We have CONDITIONALS.

**Owner, 2026-08-13, and it invalidates the naive form of §1.2 below.** The first draft of this doc
proposed a posterior over an enumerated SET vocabulary. That assumes a **prior over sets**, and we do
not have one. Smogon publishes per-species **attribute frequencies** — `P(move | species)`,
`P(spread | species)`, `P(item | species)`, `P(ability | species)` — not the joint
`P(move₁,move₂,move₃,move₄, spread, item | species)`. `data/pokemon/` consumes exactly those
marginals today (`gen3_ability_priors`, `gen3_hidden_power_priors`).

**This has a consequence that reverses the criticism in §1.1: given ONLY marginals, the
maximum-entropy joint IS the independent product.** The current five-head design is therefore not a
modeling error — it is the *correct* answer under the information actually available. Anything that
adds structure must justify it with a real constraint, not with elegance.

So the upgrade is **not** "switch to a set posterior". It is:

> **Keep a factorized belief, and add the couplings we can actually measure as CONSTRAINTS —
> pairwise conditionals — with zero-coupling reducing exactly to today's independent product.**

Three sources of genuine conditional information, in descending order of confidence:

1. **In-battle evidence (free, exact, already flowing).** Every reveal is a hard conditioning event.
   The learnset gate (v65) is the extreme case: a hard `P(move | species) = 0`. This is where most
   of the real information is, and it costs nothing.
2. **The 719-team pool — the only genuinely SET-LEVEL data we own.** Each team file is a joint
   sample: real movesets with real spreads on real species, already fingerprinted by `team_sha` and
   archetype-labelled. Pairwise co-occurrence `P(move_j | move_i, species)` is directly countable
   from it. **Caveat, stated up front: it is small (719 teams) and biased** — tournament/sample
   teams, which is our *training distribution* but not the ladder's.
3. **Smogon co-occurrence, IF it exists at the needed granularity — UNVERIFIED.** Usage stats
   sometimes carry teammate and moveset breakdowns. Whether they give per-species move-pair
   conditionals must be checked before being relied on (§1.3).

The natural object is then a **pairwise-coupled factorization** (a small MRF over the move axis,
conditioned on species) rather than a categorical over sets: it is fit from co-occurrence, it is
exactly the independent product when couplings are zero, and it represents the thing marginals
cannot — *SubPunch is one object; seeing Substitute should raise Focus Punch.* Byte-identical
off-state, strict generalization on: this project's usual shape.

### 1.2b The superseded framing (kept because the projection identity still holds)

> ~~`P(set | evidence)` — one posterior over a discrete SET vocabulary, per opponent slot.~~

Every current head becomes a **marginal by projection**, not a separate network:

```
P(species)        = Σ_sets P(set) · 1[set.species = s]
P(move m present) = Σ_sets P(set) · 1[m ∈ set.moves]
E[spread]         = Σ_sets P(set) · set.spread
P(HP type t)      = Σ_sets P(set) · 1[set.hp_type = t]
```

Consequences that fall out for free rather than being engineered:

* **Unrevealed mon and unrevealed move slot become the same question.** A fully hidden slot is a
  posterior over all sets; a revealed Skarmory with two moves shown is the same posterior
  *conditioned*. There is no longer a "hidden mon path" and a "revealed mon path" — only more or
  less evidence. That is precisely the uniformity being asked for.
* **One masking convention.** Today each head masks differently and each mask rate means something
  different. With one posterior there is one question: how much probability mass is this slot's
  posterior spread over?
* **Evidence is monotone and legible.** Every reveal (a move used, an item consumed, damage taken
  that implies bulk) is a *likelihood update on one distribution*, not five uncoordinated ones.
* **Legality is already unconditional** (v65), so the support is real by construction.

### 1.3 What must be verified before this is real

**⚠️ The set vocabulary is a DATA question and it is UNVERIFIED.** This design assumes Smogon's
gen3ou statistics can be read at set granularity (a species' *joint* move/spread/item modes), not
merely as independent per-attribute frequencies. `data/pokemon/gen3_smogon_stats.json` is currently
consumed as marginals (`gen3_ability_priors`, `gen3_hidden_power_priors`). **If the raw stats carry
only marginals, the joint must be reconstructed** — from the 719-team pool (which IS set-level and
already fingerprinted by `team_sha`), or by clustering. That reconstruction is step 0 and nothing
below is buildable without it.

---

## 2. Stage 2 — REASONING: one pair tensor, one reduction, and where α actually lives

### 2.1 The object

For each of our actions `a` and each of their actions `k`, the rules define an outcome:

```
pair[a, k, :]   # damage, KO probability, status landed, neutralization, tempo, ...
```

This is not new — `design_op_tensors.md` types it and `pair_in[k, j, :]` exists today. What is new
is the claim that **everything downstream is a READOUT of this one tensor**, not a separate feature:

| question | readout |
|---|---|
| how hard do they hit my mon j | reduce over their axis, take the damage channels |
| will Focus Punch execute | `Σ_k α_k · 1[pair[·,k].damage_to_me = 0]` |
| is Substitute safe | `Σ_k α_k · 1[pair[·,k].damage < 25% maxhp]` — a **threshold**, needs a second moment |
| should I Explode | trade value, zeroed on their Protect, target `β`-weighted on switch |
| is Counter live | `Σ_k α_k · 1[category match] · 2·damage` |

`design_conditional_execution.md`'s twelve mechanics are twelve *readouts*, not twelve features.
That is why its §3.0 could collapse five of them into one `p_thresh(τ, ⋛)` operator — they were
never separate to begin with.

### 2.2 α is not a feature. α is the reduction weight.

**This is the reframe's sharpest point.** α is currently built (v67) as an aux head with **no
consumer**, which reads as an unfinished feature. It is not. It is the **missing argument to an
operator that already exists**:

```python
# damage_op.py::_chan_max — THE arity-2 → arity-1 reduction site
#   hard_max            (today — incoherent: a different argmax per channel AND per defender)
#   belief_weighted_mean (the un-maxed marginal)
#   conditional(λ)       (OA1)
#   learned_attention(k) (PV)
```

Reducing over their move axis **requires a distribution over that axis**. Today we substitute
`hard_max`, which is why the reduction is incoherent (up to nine different opponent moves can
describe one defender) and why hedging is unreachable (`max` cannot produce a second moment).

So: **α exists to be the `how=` of a reduction that is already the single site.** Not a new path —
an argument. That is also why `design_pair_reduction.md`'s G1 null was near-guaranteed: it tested
the reduction *without* a distribution to reduce by, on damage-only cells.

### 2.3 What "our stance" means here

The owner's phrasing — *"considered both our stance and what we expect the opponent to do"* — is
exactly the pair tensor's two axes. Our stance is the `a` axis (which is why the policy reads it
per-action through the pointer head); their expected action is the `k` axis, weighted by α. A
reduction that keeps `a` and collapses `k` is precisely "what happens if I do this, given what I
think they'll do."

**The constraint that keeps this honest** (from `design_opponent_intent.md`): α may depend on the
BOARD, never on our realized action — otherwise it becomes per-action and reintroduces the D3
decorrelation defect. Both sides anticipate; the fixed point is found by self-play training, never
solved at inference.

---

## 3. Stage 3 — DELIVERY: one contract, two pools

Today delivery is four mechanisms with different expressiveness, and the policy/critic asymmetry is
**accidental** rather than designed:

| route | carries | reaches |
|---|---|---|
| edge biases (15 families) | a softmax-normalised RATIO | both heads |
| pointer cells | a per-action ABSOLUTE | **policy only** |
| `MultiSeedValueReadout` | a convex combination of six rows | **critic only** |
| `--value-threat-inject` (v64) | a per-entity ABSOLUTE | **critic only** |

Two of those four exist because the other two could not reach the head that needed them. That is the
signature of a missing contract, and it is measurable: gen-6 and gen-7 both showed the seed readout
capping at ~1 effective direction, which is what forced v64 into existence.

**Unified:** the reduction (§2) emits per-`(entity, action)` rows. The **policy** pools over
entities per action — that is the pointer head, unchanged. The **critic** pools over everything —
that is attention pooling, which is permutation-invariant. Same object, two pools, one contract.

Both are equivariant by construction: α has no defender index (Contract W), so it is invariant under
permuting their moves; the row rides its own entity's token, so it is equivariant under permuting
ours.

---

## 4. What this reframe actually changes

**Nothing about the physics.** The `DamageOperator`, the edge families and the pointer head all
stay. This is a re-factorization of *belief* and a naming of *where the reduction's missing argument
comes from* — not a rewrite of the model.

| today | unified |
|---|---|
| 5 belief modules, 8 label keys | 1 posterior; the five heads become projections |
| 4 masking conventions | 1 (how spread is this slot's posterior) |
| "hidden mon" and "revealed mon" are different code paths | one path, different evidence |
| α is a head with no consumer | α is the reduction's `how=` |
| 12 conditional mechanics = 12 features | 12 readouts of one pair tensor |
| 4 delivery routes, 2 of them compensating | 1 contract, 2 pools |

---

## 5. Honesty — what this does NOT promise

* **It is not a strength argument.** Gen-8 enabled the entire belief stack, the beliefs *learned*
  (`species_acc_above_chance` 0.67, `move_recall` 0.19→0.58, `spread_largest_bias` −26→−13) and
  anchored ELO **fell** (tail-4: gen-4 2057.8 / gen-5 2038.4 / gen-8 2016.5, matched steps). The
  honest case here is **coherence and consumability** — one object that can finally be reduced,
  read out and delivered — not a predicted ELO jump. If it is sold as a strength lever it will look
  like a failure for the same reason gen-8 did.
* **The prior null stands.** An opponent-action aux head was falsified on value-of-information
  (~0.03). This is a different use — a weighting *inside* the physics, gradient riding the damage —
  but the null should be stated, not stepped around.
* **One-ply only.** "They are saving Explosion for my Celebi" stays out of reach. Each leg of a
  two-turn plan can be priced; the plan cannot be represented.
* **Delivery ≠ behaviour.** An oracle move-belief once flipped 19.3% of actions but moved switch
  mass by +0.019. Gate the architecture claim and the behaviour claim separately.
* **The set vocabulary may not exist at the needed granularity** (§1.3). That is the load-bearing
  unknown and it is a *data* question, answerable offline in an afternoon.

---

## 6. The order this implies

Deliberately not a build order — a dependency order, cheapest-decisive-first:

| # | step | needs a run? | decides |
|---|---|---|---|
| 0 | **Do usable CONDITIONALS exist?** (revised — there is no set prior, §1.2.) Count pairwise move co-occurrence per species on the 719-team pool, and check whether Smogon's stats carry any joint/teammate structure at all. | no | whether §1 has any input |
| 1 | **Does the coupling carry information?** Measure `P(move_j \| move_i, species)` against the independent marginal. **This is THE decisive number**: if the lift is small, the MaxEnt argument (§1.2) says the current independent product is already right and the whole refactor is elegance without payoff. Report the lift on the pairs that matter (SubPunch, CB-vs-DD, Spikes/Whirlwind), not just the mean. | no | whether ANY of §1 is worth building |
| 2 | **G3** from `design_conditional_execution.md` — one family (`c2`) re-delivered through the move cell with α. | no | whether the consequence line is alive AT ALL |
| 3 | α's own gate — `alpha_acc_move` vs its `argmax(w)` baseline (both now logged, gen-9). | in flight | whether α beats the free guess |
| 4 | the reduction's `how=` — swap `hard_max` for an α-weighted rung at the ONE call site. | no | §2 |
| 5 | the delivery contract (§3) | retrain | §3 |

**Steps 0–2 are all offline and all cheap, and any of them can kill the line before a generation is
spent.** Step 1 is the one that decides whether this reframe is a real architecture or a tidier way
of drawing the same picture — and I do not know the answer to it yet.

---

## See also

* `design_opponent_intent.md` — where α/β come from (the §2.2 argument re-homes it)
* `design_pair_reduction.md` — Contract W, the second-moment argument, the ONE reduction site
* `design_conditional_execution.md` — the twelve readouts of §2.1
* `design_conditional_opponent_cells.md` — OA1/OA2 and the critic-route question of §3
* `designs/learning/entity_tokens_biases_pointers.md` — the sorting rule this obeys
