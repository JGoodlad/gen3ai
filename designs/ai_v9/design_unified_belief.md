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
| what they will DO this turn | `α` / `β` (v68 in code; earlier prose said v67) | `opp_action_*` | ~~none~~ **[UPDATE 2026-08-14: `intent_value_reduce` (v74) — α-weighted pair-cell rows into vf; the policy-side consumer stays open]** |

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
2. **The loop is open.** **[UPDATE 2026-08-14: half-closed — v74 `gen3_intent_value_reduce_v1`
   feeds `Σ_k α_k · pair_in[k,·,:]` to the CRITIC (vf-only, zero-init concat; live in gen-9).
   The POLICY-side consumer (`design_conditional_execution.md`'s move-cell route) remains the
   open half.]** α's OUTPUT reaches nothing (§2.2). A T2 object that no decision consumes is
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
3. **Smogon co-occurrence — VERIFIED 2026-08-15** (the chaos JSON, e.g.
   `stats/2026-07/chaos/gen3ou-1500.json`; our `tools/smogon_stats_downloader` already merges 12
   months of exactly these files into `gen3_smogon_stats.json`): per-species `Moves` are
   **marginals only** — chaos carries NO within-species move-pair joint. What it DOES carry as
   joints: **`Teammates`** (species×species co-occurrence, 2.5M battles — now derived as
   `gen3_teammate_priors.json` / `gen3_data.priors.teammates`, the hidden-team belief's coupling
   prior) and **`Spreads`** (full nature+EV strings — a per-species joint over the spread axes,
   already consumed as `gen3_spread_priors.json`).

**⚠️ OWNER RULE (2026-08-15): priors are always SMOGON-based, never pool-based.** The 719-team
pool may *measure* coupling (source 2 above — it is the only set-level joint we own), but a
measured pool statistic never ships as a prior. Consequence for this design: species↔species and
spread couplings have a shippable Smogon source; **move↔move couplings do not** — they stay with
in-battle evidence (source 1) and whatever the network learns, and §1's pairwise-MRF idea is
bounded to structure a Smogon-priored or evidence-driven quantity, not a pool-fit one.

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

**ANSWERED 2026-08-15 (was: the load-bearing UNVERIFIED).** Smogon's chaos stats carry NO
set-level move joint (§1.2 source 3) — move sets exist in Smogon only as marginals. The
couplings are nonetheless REAL: measured on the 719-team pool
(`designs/research_state/measurements/belief_coupling_lift.json`,
`tmp/belief_coupling_lift.py` — statistic `T = Σ freq·|log2 lift|` vs a marginal-preserving
bipartite-re-deal null), every large species shows **T at 2.7–4.0× its null, p≈0**
(tyranitar 3.97×, salamence 3.36×, swampert 3.16×, celebi 2.77×, jirachi 2.67×). The dominant
structure is **EXCLUSION** (slot competition: crunch↔dragondance lift 0.007 — the CB/DD split;
dragonclaw↔hiddenpowerflying 0.009; hydropump↔surf 0.01) with real positive archetype pairs
(tyranitar sub→focuspunch lift 6.1, charizard 4.4). So §1.1's criticism stands *as physics* —
but under the owner rule above, the shippable coupling prior is limited to what Smogon carries
(teammates, spreads); move-pair structure must come from in-battle evidence or be learned,
never fit on the pool.

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

> **[UPDATE 2026-08-15] The CRITIC half of this section is BUILT**: v80
> `gen3_unified_value_readout_v1` (`--value-entity-pool`, opt-in, zero-init, vf-only — one
> attention pool over the 12 team tokens + the op's incoming rows, with its own `entity_pool`
> arm in `critic_route_audit`). The policy half needs no build — the pointer head IS the
> per-action pool, as this section says. What remains of §3 is the ADOPTION: the gen-11 audit
> condemns (or spares) the bolt-on routes below, and the enabling generation swaps them for
> the contract in one config change.

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
| 0 | ~~Do usable CONDITIONALS exist?~~ **DONE 2026-08-15** (§1.3): Smogon = move marginals + the `Teammates` joint (now `gen3_teammate_priors.json`) + full `Spreads`; the pool is the only move-pair joint and is measurement-only (owner rule). | no | whether §1 has any input |
| 1 | ~~Does the coupling carry information?~~ **DONE 2026-08-15 — YES** (§1.3): T at 2.7–4.0× the marginal-preserving null, p≈0, on every n≥200 species; dominated by slot-EXCLUSION plus real archetype pairs (subpunch 6.1×/4.4×; the named Skarmory spikes→whirlwind is uninformative — spikes is in 203/203 sets, lift 1.0 by construction). The MaxEnt defense of the independent product is REFUTED as physics; the shippable-prior path is constrained to Smogon sources per the owner rule. | no | whether ANY of §1 is worth building |
| 2 | **G3** from `design_conditional_execution.md` — one family (`c2`) re-delivered through the move cell with α. | no | whether the consequence line is alive AT ALL |
| 3 | α's own gate — `alpha_acc_move` vs its `argmax(w)` baseline (both now logged, gen-9). | in flight | whether α beats the free guess |
| 4 | the reduction's `how=` — swap `hard_max` for an α-weighted rung at the ONE call site. **Still open** (both HEAD-side α consumers exist — v74 critic, v77 policy, gen-11 trains both — but the op's internal reduction remains R0 `hard_max`; deprioritized behind the entity end-state). | no | §2 |
| 5 | the delivery contract (§3) — **critic half BUILT** (v80 `--value-entity-pool`, opt-in); adoption waits on the gen-11 critic-route audit | retrain | §3 |

**Steps 0–2 are all offline and all cheap, and any of them can kill the line before a generation is
spent.** Steps 0–1 are answered (2026-08-15): the coupling is real physics, and the prior budget
for expressing it is Smogon-shaped — teammates + spreads carry shippable joints, move-pairs do
not. The next decisive step is therefore G3 (step 2, built v77, unrun) plus wiring the teammate
prior into the hidden-team belief's T0 posterior — the first consumer of the one coupling we may
actually ship.

---

## See also

* `design_opponent_intent.md` — where α/β come from (the §2.2 argument re-homes it)
* `design_pair_reduction.md` — Contract W, the second-moment argument, the ONE reduction site
* `design_conditional_execution.md` — the twelve readouts of §2.1
* `design_conditional_opponent_cells.md` — OA1/OA2 and the critic-route question of §3
* `designs/learning/entity_tokens_biases_pointers.md` — the sorting rule this obeys
