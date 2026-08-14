# design — OPPONENT INTENT: from pair physics to "what will they do", and from that to our answer

> **[STATE 2026-08-14]** α/β are **BUILT** (v68 `gen3_opp_intent_v1` — supervised pointer heads,
> canonical-id matching, mask-rate diagnostics, the whole render path to the prober) and α is
> **CONSUMED on the critic side** (v74 `gen3_intent_value_reduce_v1`). The schedule this doc
> pre-registered happened: gen-8 (foundation: beliefs + threat-inject, launched 2026-08-11) then
> gen-9 (α + the distributional critic, launched 2026-08-13, live). The OUTGOING/policy-side
> consumer (`design_conditional_execution.md`) remains open. See `designs/CHANGELOG.md`.

> **What this document is.** The build for one sentence the model cannot currently express:
> *"they are likely to click **this**, so **this** is my answer."*
>
> **Relationship to existing docs.** `design_pair_reduction.md` specifies the *operator* that
> collapses their move axis and proves the coherence contract; this doc supplies the two things
> that operator was missing — a **distribution worth weighting by** (`α`) and an **outcome vector
> worth weighting** (the unified `pair_in`). The pair-reduction ladder is component 3 here.
> `design_conditional_opponent_cells.md`'s OA1/OA2 are the query-conditioned extension, out of
> scope until this lands.

---

## 0. Goal

**Make the opponent's intent a first-class, supervised, physically-grounded quantity that the
reduction consumes — so that anticipation stops being something the network must infer from an
unweighted outer product and becomes something it is given.**

The chain, and what each link needs:

```
complete pair physics  →  α: what they'd want to do  →  weight the physics by α  →  compare per action
   (ONE tensor)            (a supervised head)           (Contract W)                (pointer logits)
```

The reason the chain closes: **our incoming grid IS their outgoing grid.** `d3` is
"their believed move × our mons" — precisely the tensor *they* would consult to choose a move. So
`α` is not new information that must be sourced; it is a **readout of physics we already compute.**
That is why component 1 comes first and why it makes component 2 cheap.

---

## 1. Why the current system cannot express this

Three facts compose into a structural impossibility, not a weakness.

**(a) An edge cannot carry it.** An edge bias is softmax-normalised — it writes a **ratio within its
row**, never an absolute (`ARCHITECTURE.md` §5.3). And it is indexed by `(entity, entity)`, while a
distribution over their moves is a vector over *one* axis with no natural cell. The single place a
belief rides today is `d3`, where `w` is smuggled in as a per-cell channel `[high, pko, eff,
is_phys, w]`, broadcast along our-mon axis.

**(b) The comparison happens somewhere else.** "Best response" is
`argmax_a Σ_o P(o) · U(a, o)`. The `argmax_a` happens at the **pointer logits**, so the expectation
must be formed **per action, before the logit** — i.e. inside the per-action cell. Edges deliver the
raw `U(a,o)` outer product and can do nothing else with it.

**(c) The one legal home contains a max.** The per-action cell is filled by a hard `amax` over an
unweighted grid (`damage_op.py:534`, `_chan_max(how="hard_max")`).

**The measurement that confirms this is a real gap, not a theoretical one** — gen-4 end-of-run edge
audit (25M, stratified, n=6000, `gen4_edge_family_audit_25M.json`):

| what it carries | family | flips |
|---|---|---|
| **our** bench offense → their active | `d2` | **19.25%** |
| **our** active's moves → their mons | `d1` | **12.17%** |
| **their believed move → our mons** | **`d3`** | **0.63%** |
| **their status → our mons** | `s3` | 0.40% |

`d3` carries their believed move, its damage to each of our mons, its category, and our belief in
it — the core defensive read — and it reads **0.63% at end of training, down from 1.9% at gen-3's
9.6M.** It is not growing into usefulness. The system is overwhelmingly offensive, and §1(a)–(c)
say why: the offensive half reaches per-action cells as absolutes, the anticipatory half is routed
through a channel that cannot express what it needs to say.

### 1.1 The three-part object

| | needed | we have |
|---|---|---|
| **1. a distribution over their action** | `α` — **usage**: will they click it, into this active | `w` — **presence**: does the move exist in their set |
| **2. an outcome vector worth weighting** | damage **and** status, neutralization, tempo | damage |
| **3. the weighting done per-action, before the logit** | `Σ_o α_o · φ(...)` | `max` |

**Missing any one makes the other two useless.** That is why the capability feels absent everywhere
rather than weak somewhere.

**And it explains the G1 null without appeal to anything else.** G1 FINAL (n=299, 5 seeds) tested
**part 3 in isolation** — reducer rungs, on damage-only cells, with `w` substituted for `α`.
Weighting a damage-only outcome vector by a better distribution cannot help when the decisions that
turn on anticipation turn on non-damage consequences. That null was close to guaranteed by
construction. (Separately, its 2800-dim SKYLINE was fit with ~239 training rows at L2 = 1e-3 and is
likely variance-limited — see §9.1. Two independent reasons not to read it as "the grid is
exhausted.")

### 1.2 This is NOT the falsified opponent-action head

The opp-action aux head was falsified two ways (`project_opp_action_head_falsified`,
`project_l3_oracle_grind_l4`) and that verdict stands **for what it tested**: a head that predicts
their action as a **side output**, emitting a label and changing nothing downstream. It re-encoded
information the trunk already had (opp_switches 0.89/0.90, move-TYPE 0.93/0.96).

`α` is not a prediction; it is a **weighting operator inside the reduction**. Identical supervision,
completely different position in the graph — one produces a label, the other changes what the
pointer cell *contains*. The falsification measured the value of *knowing*; this design is about the
value of *applying*. (See `design_pair_reduction.md` §10.8, which retracts the VoI-ceiling reading
that would otherwise appear to bound this work.)

---

## 2. The contract — what is fixed regardless of implementation

| | |
|---|---|
| **`α` may NOT depend on our realized action.** | Gen3 is simultaneous-move with no team preview. Allowing `α_j` lets Skarmory's row assume Rock Slide while Blissey's assumes Thunderbolt — one opponent clicking several moves at once. |
| **`α` MAY depend on our policy's incentives.** | They cannot see our choice but they can infer our *intent*. See §3 — this is the load-bearing reconciliation and it costs no signature change. |
| **One `α` per decision, shared across every channel and every defender.** | The coherence contract from `design_pair_reduction.md` §3.1. |
| **`Σα = 1, α ≥ 0`** | Keeps the reduced output a convex combination of HP fractions ⇒ units, range and auditability survive. |
| **The label never enters the forward.** | `α`'s supervision rides a training-only obs key, guarded by the existing graph invariant (§5.4). |
| **No fixed point is solved at inference.** | Training finds it. No iterated best response, no search — the owner's hard constraint holds (`research_state/README.md`). |

---

## 3. RECONCILIATION 1 — both sides anticipate, and the fixed point

**The objection.** They cannot see our switch, but they can *infer* what we would want to do. So
there is a distribution on both sides, each conditioned on the other, and the correct object is a
fixed point. Does that break the contract, and does it need a solver?

**It does not break the contract, and the reason is a type argument.**

`α` must not depend on our **realized** action `a` — that would be observing a simultaneous choice.
But `α` **may** depend on our **policy** `π(· | board)`, because a policy is not an observation of
our choice; it is a property of the position. And `π(· | board)` is a function of the board. So:

> **`α = f(board, their options)` is already the correct form.** The equilibrium reasoning lives
> *inside* the function, not as an extra argument. Level-2 reasoning is a property of *how `α` is
> computed*, not of the signature.

### 3.1 The implementation consequence — `α` must see OUR physics

This is the concrete part, and it is the change this reconciliation actually forces:

> **`α`'s input must include our own OUTGOING physics** — what our active threatens (`d1`'s grid)
> and what our bench offers (`d2`'s grid) — not only their incoming grid.

To learn *"in boards like this, opponents click the switch-punishing move,"* `α` has to see that our
active is outmatched and our bench holds an obvious answer. That is exactly what "they can infer
what we might want to do" means, made computable. Without our offensive grid in scope, `α` can only
learn *"which of their moves is individually strongest"* — a level-1 read, and largely what `w`
already gives.

**Why physics and not our policy logits.** Reading `π` directly would be level-3 and is tempting,
but it creates a **cycle**: `π` depends on the reduction, the reduction depends on `α`, `α` would
depend on `π`. The pointer logits are produced at the *end* of the chain; `α` is needed in the
middle. Feeding physics keeps the forward pass a DAG. (A cheap early "our-intent" head supervised by
our own final policy is the level-3 version — recorded in §10 as deliberately **not** taken.)

### 3.2 Who solves the fixed point

**Self-play does, during training.** The opponent literally is a copy of us, so `α` supervised
against what the opponent actually did converges toward a mutually-consistent pair. An equilibrium
is just a pair of distributions and a network can represent a distribution — what it cannot do is
*solve* for one at runtime. Training does the solving; inference reads the answer off.

This is the same claim `design_pair_reduction.md` §4.2 makes, now with the mechanism attached.

### 3.3 The honest risks of that answer

**Non-stationarity — `α` chases a moving target.** Wiring `α` into the policy tightens the
self-play feedback loop: `α` predicts the opponent, the opponent is us, we adapt to `α`, `α`'s
target moves. Mitigations, in order of preference:

1. **Train `α` against the pool distribution, not only the latest self** — the snapshot pool and
   PFSP machinery already exist and this is what they are for.
2. **Stop-gradient the RL loss out of `α` initially** (§4.4) so `α` optimizes only its supervised
   objective and cannot be dragged by a shifting policy.

**Amortization — `α` learns the pool's habits, not gen3's.** Whatever distribution we train
against is what `α` encodes, so it will be wrong against off-distribution opponents (ladder humans)
until retrained. This is the amortization gap in a new place, and it is **not** fixable here; it is
a limitation to record, and an argument for the human-replay corpus as an eventual `α` validation
set rather than a training set.

**Level-k is bounded in practice.** Feeding our physics buys level-2. Self-play pushes the pair
toward the fixed point over training. Neither yields true level-∞ reasoning, and no claim here
should be read as doing so.

---

## 4. RECONCILIATION 2 — the DISCRETE constraint, and what the aux loss targets

**The owner constraint (2026-08-11), and it is binding on everything below:**

> **The model must always pick among the discrete states the belief holds. It may never invent a
> move.** Interpretation is the reason: an output that can name something not in the belief cannot
> be checked against anything.

**The objection this has to survive.** The E4 threat seats are the **top-K of a belief**, so (i) the
move they actually clicked may not be among them, and (ii) seat index 3 means a different move on
turn 5 than on turn 12. Cross-entropy against a seat *index* supervises a moving target with a
possibly-absent label.

The constraint turns out to make the design **smaller**, not larger — it deletes the learned
property head an earlier draft proposed (§4.6).

### 4.1 The distribution is over the SEATS, plus SWITCH

```
α ∈ Δ^(K+1)        [ believed move seat 1 … seat K ,  SWITCH ]
```

Every unit of mass names a discrete, inspectable object — *"Gengar's Will-O-Wisp"*, *"they pivot"*.
There is **no `UNKNOWN` slot** (an earlier draft had one; §4.6 records why it was cut).

**This is strictly MORE interpretable than today.** The block's current `provenance` channel reports
one move — the argmax — and nothing about the alternatives. A discrete `α` prints the whole
weighting per decision:

```
Will-O-Wisp 0.52   Thunderbolt 0.31   switch 0.11   Explosion 0.06
```

and it makes the reduction **attributable**: *"this switch cell reads 43% incoming because α put
0.52 on Will-O-Wisp"* is a sentence that can be checked against what they actually did.

### 4.2 Target construction — ONE rule, both axes

> **If we can't name it, we don't train on it.** Hard target when the belief holds it; **masked**
> otherwise; the mask rate is a first-class diagnostic and the gate on belief work (§4.5).

| | what happened | target |
|---|---|---|
| 1 | they **switched** | `SWITCH` (hard). Always available, belief-free. |
| 2 | move **held by a seat** | that seat (hard). Matched by **canonical id**, never by index. |
| 3 | move **not held by any seat** | **masked**, rate logged |

**Matching is by identity, never by stored index**, because seats are re-derived from the belief
every decision. Same order-invariance discipline as the hidden-team belief head's Hungarian
matching (`belief_labels_fuzz_test.py`).

**Why masking and not a soft target over property-similar seats** (owner, 2026-08-11 — an earlier
draft of this section proposed exactly that, and it was wrong):

- **Masking yields a well-defined object.** `α` learns `P(seat k | they used a move we modeled)`,
  which is the honest reading of a distribution over seats — if a move is not in the seats,
  `pair_in` has no cell for it and there is nothing to weight. A soft target instead yields the
  seat distribution *plus* redistributed mass from moves we failed to model: harder to state,
  harder to check.
- **A soft target injects the belief's failure mode as a bias.** The belief's blind spots are not
  random, so smeared mass would systematically over-weight whichever categories it misses — a
  distortion caused by the belief's weakness, living inside `α`, and invisible in `α`'s accuracy.
- **It needs a similarity metric with no principled setting.** Category and type? Why not base
  power, priority, secondary chance? A free hyperparameter directly on an
  interpretability-critical path.
- **Consistency.** `β` masks its belief misses (§4.3). One rule on both axes, or an unjustified
  asymmetry.

**The division of labour this buys:** `α`/`β` own *which of the things we believe*; the belief head
owns *whether we believe the right things*. Two heads, two failure modes, two separate
measurements — instead of one head quietly absorbing the other's errors.

**Hidden Power — and note this removes the motivating example for soft targets.** We already use the
privileged type as a *label*, in the right place: `hp_type_belief_coef = 0.05`, consumed in
production, hptype_acc ≈ 0.91. For `α`, match on whatever representation the seat carries (typed num
355–370 if the belief typed it, bare 237 otherwise) and let the HP-type head own type accuracy —
double-penalising `α` for the HP head's error conflates two failures. **Under canonical-id matching
a seat holding bare `hiddenpower` MATCHES a used HP Ice**, so HP was never a belief-miss case at all.
What remains for case 3 is genuinely-unmodelled tech moves — a belief-coverage problem, and one that
a cleverer label would hide rather than fix.

### 4.3 `β` — switch to WHOM

`SWITCH` alone is not actionable: bringing Blissey and bringing Skarmory invert which of our mons is
useful. So the switch branch factors.

```
β ∈ Δ^(≤6)     over their team slots — gated to ALIVE and NOT-ACTIVE (exact, observable),
                and in v1 additionally to REVEALED
P(their action) = P(SWITCH) · β(mon)  +  Σ_k α_k
```

Discrete and named, same as `α`: `β` prints as *"Blissey 0.60, Skarmory 0.30, Tyranitar 0.10."*

**Targets:**

| | what happened | target |
|---|---|---|
| 1 | switched to a **revealed** mon | that slot (hard) |
| 2 | switched to a **previously-unrevealed** mon | **masked in v1**; rate logged |
| 3 | they did not switch | no `β` target — `β` is conditional on `SWITCH` |

**Case 2 is the honest v1 limitation, and masking is defensible on when it bites.** Early game most
switches are to unrevealed mons; late game — where endgame positioning is decided and where the
pivot question actually matters — the team is largely revealed. So v1 trains `β` exactly where `β`
earns its keep, and the mask rate measures what is being given up.

**The upgrade path is already built and never run.** The hidden-team belief (**B1**, ledger:
in-place unknown-mon slot tokens + Hungarian species aux head, learns immediately at acc 0.08–0.16
vs ~0.003 chance, `--opp-belief-aux-coef`, **BUILT 2026-06-13, UNMEASURED**) gives the unrevealed
slots a *species posterior* — which makes them named entities, so case 2 becomes a soft target by
posterior instead of a mask. **B1 is therefore the named prerequisite for `β`'s unrevealed half**,
exactly as `move_belief_coef` is for `α` (§4.5). Turn it on only if the mask rate says the
unrevealed-switch case is costing enough to matter.

### 4.4 What `β` unlocks — and why it retro-justifies the (bench × bench) family

`β` is consumed in three places, and the third is the one that matters most:

1. **`α`'s `SWITCH` mass scales this turn's incoming damage down.** A turn they spend switching is a
   turn they do not attack — a fact the current block cannot represent at all.
2. **`β` weights `d4`** (their bench's threat to us) into an expectation over who actually arrives,
   instead of pricing all six slots equally.
3. **`β` weights our OFFENSE against whoever arrives** — and this is what makes the missing
   `(our bench × their bench)` grid actionable. That grid alone is an unweighted outer product;
   with `β` it answers the real pivot question: *"if I bring Skarmory in and they pivot to Blissey,
   is Skarmory still doing anything?"*

That last point resolves a ranking question left open elsewhere: the `d2` phys/spec split and the
`d5` bench×bench family are **not** independent cheap wins to be scheduled separately — they are
the grid `β` needs in order to be useful, and `β` is what makes them worth having.

### 4.5 THE BELIEF STACK — everything `α` rests on

Confining `α` to the seats means **`α`'s ceiling is belief quality.** So the whole stack becomes a
gated dependency rather than background context. Audited against `designs/production_config.json`
and the live code, 2026-08-11:

| # | belief | predicts | mechanism | in production | SUPERVISED? |
|---|---|---|---|---|---|
| **B-move** | which moves the revealed mons still hold | learned head, Smogon prior at init | **ON** (`move_belief_mode: revealed`) | ❌ **NO** — `move_belief_coef` **0.0** |
| **B-hptype** | which of the 16 typed Hidden Powers | learned head | **ON** (`hp_belief_mode: composed`) | ✅ 0.05, acc ≈ 0.91 |
| **B-spread** | their EVs/nature ⇒ the 5 derived stats | — | ❌ **OFF** | n/a |
| **B-team** | species of the ~3 unrevealed slots | Hungarian head (**B1**) | ❌ **OFF** — BUILT 2026-06-13, never run | n/a |
| **B-latent** | identity in role-token space | SimSiam | ❌ OFF | n/a |
| **B-item** | P(opp active holds Choice Band) | **STATIC species usage prior**, collapses to 0/1 on reveal | **ON** | **not learnable — a lookup** |
| **B-ability** | which ability | **STATIC Smogon per-species prior** | **ON** (obs) | **not learnable — a lookup** |

**Read the last column.** Of seven belief legs, **exactly one is supervised** (B-hptype). Two are
static lookups that cannot improve with training. Three are off. And **the one this entire design
rests on — B-move — runs unsupervised**, shaped only by the Smogon prior at init plus the RL
gradient (`belief_grad_mode: shaping`) and a 0.05 latent term.

#### The two that block this design

**B-move — a coefficient, and nothing killed it.** `known_moves` (the revealed mons' full privileged
movesets, direct BCE) is **already emitted and already plumbed**; `ARCHITECTURE.md` §7 lists it as
"emitted; **unconsumed**". The flag is `--move-belief-coef`, **training-only, not version-locked,
resume-mutable**. No ledger or changelog entry records it being turned off after a failure — it was
simply never turned on. House scale for belief aux coefficients is 0.05.

**B-spread — and this one is a PHYSICS DEFECT, not a missing signal.** With it off, the op prices
every opponent's offense from a hardcoded assumption (`damage_op.py:1727`):

```
atk_j = (2.0 * a_base[..., _BS_ATK] + off_const) * 1.1        # de-timid
```

— 252 EVs and a boosting nature, **applied uniformly to every mon on their team**, at nine sites.
So `pair_in`'s incoming damage numbers, the foundation everything downstream weights, are computed
against a **fictional maximally-invested opponent**. The over-estimate is **not uniform**: it scales
with base stats, so it distorts the *relative* threat ordering across their team, not merely the
level. `--spread-belief` replaces the constants with believed stats;
`--spread-belief-nature` (v40) additionally fixes an order-statistic bias where a point-estimate head
over-estimates whichever stat carries the largest EV investment.

**Consequence for this design, stated plainly: B-spread is a correctness fix to component 1, not a
third belief leg to stack.** Weighting distorted physics with a better `α` inherits the distortion.

**Operationally they are very different**, and the difference sets the schedule:

| | flag | class | can it join a running generation? |
|---|---|---|---|
| B-move | `--move-belief-coef` | **training-only, resume-mutable** | yes, on a resume |
| B-spread | `--spread-belief` (+`-coef`, `-nature`) | **STRUCTURAL, version-checked, FRESH-ONLY** | **no** |
| B-team (B1) | `--opp-belief-aux-coef` | structural slots | no |

#### The gate — measure before enabling

**G2a, and it needs nothing but existing traces:** *how often does the belief's top-K actually
contain the move they clicked, and how often is a switch to a revealed mon?* Those two rates are the
ceilings on `α` and `β`. If coverage is already high, supervising B-move buys little and the
generation slot is better spent on B-spread; if it is low, we enable with a measured prize.

**Ordering: measure coverage → fix the belief (B-move coefficient, B-spread structurally, B-team if
`β`'s mask rate warrants) → build `α`.** Enabling these because they are available is not the same
as enabling them because we know what they are worth.

### 4.6 Rejected: the `UNKNOWN` slot, and the learned property head

Both appeared in earlier drafts of this document. Recorded with their causes of death so they are
not re-proposed.

**`UNKNOWN` (an escape-hatch slot for belief misses) — CUT.** Three reasons: (a) **it is
undirected** — "something I did not model is coming" does not tell you to prefer Celebi over
Swampert, and there is no outcome vector for an unmodelled move, so no expectation can be taken over
it; (b) **it is either constant or rare** — a bad belief makes it high everywhere (no per-state
signal), a good belief makes it vanishingly rare (nothing to learn from); (c) **it conflates belief
coverage with usage prediction** in one head, making a null uninterpretable. Its one legitimate use
— modulating *trust*, interpolating between the α-expectation and the hard max as coverage degrades
— is a second-order refinement, not a foundation. It also violates the discrete constraint.

**A learned property head (predict category/type/switch directly, belief-free) — CUT.** It was
attractive precisely because it removed the belief dependency, but its output is not a member of the
belief's discrete state set, so it fails the constraint: the model could imply a move it does not
hold. **And with soft targets cut (§4.2), properties do not survive in the design at all** — not as
an output, not as a label. The move's category and type are read from `gen3_data` only to *describe*
a seat, never to construct a target.

*Note the v63 convergence:* the instinct to "let a latent carry the residual and penalise
degeneracy" runs into the gen-6 result that **a repulsion penalty buys spread, not multiplicity**
(every VICReg term moved; effective rank stayed 1.05). The positive fix there was giving each unit a
*job*. Here the seats are the job — supervised discrete structure rather than penalised latent
structure.

### 4.7 Gradient policy — supervision only, at first

`α` is used by the reduction *and* trained by cross-entropy, so gradients could flow from both. The
RL loss would be free to drag `α` away from being an honest predictor toward being a merely useful
weighting.

**Ship with the RL path stop-gradiented, so `α` is trained by supervision alone.** Two reasons, both
about being able to learn something:

- `α`'s accuracy and calibration become **clean measurements** of a quantity with ground truth, not
  entangled with policy improvement.
- If the whole design fails, stop-grad tells us *which half* failed — a bad `α`, or a good `α`
  wired into a reduction that cannot use it. Without it, a null is uninterpretable.

Releasing the stop-grad is a separate, later arm.

### 4.8 Leak safety

The label is a **future observation relative to the decision** — we supervise with what they did
*after* we chose. That is ordinary supervised learning, not a leak, provided the label never enters
the forward. Precedent exists: `win_target` is explicitly "a **future** label"
(`ARCHITECTURE.md` §7).

The discipline is already built and must be reused verbatim:

- ride a **training-only obs key** (`opp_action_target` / `opp_action_mask`), emitted only by the
  trainee `Gen3Env` — eval and self-play opponents use `RLPlayer`, which never constructs them;
- `ObsUnpack.forward` reads only `obs["observation"]`, so no privileged key can reach `pi`/`vf`;
- the **graph invariant** `delivery_graph_test.test_no_aux_edge_reaches_the_forward` must cover the
  new edge;
- a bridge fuzz test in the `belief_labels_fuzz_test.py` mould (real battles, intercept the
  protocol, assert the label equals what the opponent actually did **and** that zeroing it does not
  move a single logit).

**Buffer-cost note:** §7 records that four privileged keys already ride the rollout buffer with only
one consumed. Emit `opp_action_target` **gated on its own coefficient**, not on a neighbouring mode
flag, so this design does not add a fifth unconsumed key.

---

## 5. The components

### 5.1 Component 1 — one `pair_in` tensor, every currency

**The prerequisite and today's blocker.** The physics of (their move `k` → our mon `j`) is computed
in **two separate functions with two separate reductions**: damage in the incoming block, status in
`discrete_incoming_status`. **One `α` cannot weight two tensors.** Unification is not cleanup; it is
what makes everything downstream possible.

```
pair_in[k, j] = [ phys_low, phys_high, crit, pko, acc,     ← damage (have)
                  p_status_land, p_immobilize,              ← status (exists, wrong home)
                  neutralization,                           ← NEW
                  tempo_cost ]                              ← NEW
```

**`neutralization`** — the fraction of this mon's future contribution destroyed *without* a KO.
Burn on a physical attacker, paralysis on a sweeper. Without it, Swampert reads 0.0 damage against
both branches of a Gengar carrying Will-O-Wisp and Thunderbolt (immune to one, and burn deals no
damage), so damage-only scoring picks it forever and the hedge is unreachable.

**`tempo_cost`** — turns of our clock spent undoing it. Milotic **Refreshes** the burn away, so
`neutralization` correctly reads ≈0 — but it costs a turn. Nothing in ten damage numbers plus speed
encodes that, so "absorbs it" and "absorbs it and falls a turn behind a setup sweeper" are currently
the same state.

**Deliberately out of scope: physics mutation.** Marvel Scale means burning Milotic multiplies its
Def by 1.5 and moves *every subsequent number in the matrix*. That is a statement about the successor
state, not an outcome coordinate. A one-ply reduction can learn "burn into Milotic is fine"; it
cannot represent "and here is the new matrix."

**Note the receiver is fully observed.** `j` indexes *our* six mons, so ability, moveset and stats
are exact. All the uncertainty is on the sender axis, and the belief head has already run by the
time the op consumes it. There is no marginalisation question on the receiver side.

### 5.2 Component 2 — the intent head

**One head, one softmax, discrete support** (§4.1). It scores each believed seat plus a `SWITCH`
option; `β` is a second head over their team slots, conditional on `SWITCH`.

```
α = softmax_over_{K seats + SWITCH} ( g( pair_in[k,·] , their_move_id[k] ,
                                          OUR outgoing physics , board ) )

β = softmax_over_{alive, non-active, revealed slots} ( h( d4[m,·] , opp_slot_token[m] ,
                                          OUR outgoing physics , board ) )
```

Four properties, each load-bearing and each traceable to a constraint above:

- **`g` is shared over `k`** ⇒ equivariant in their move axis; a seat's score depends on the seat's
  content, never on its position. Same for `h` over `m`.
- **Defender-independent** (§2) — neither head may see our candidate action, only the board.
- **Our outgoing physics is in scope** (§3.1) — the change RECONCILIATION 1 forced, and what
  separates a level-2 read ("they will punish the switch I want to make") from a level-1 one
  ("their strongest move").
- **The support is discrete and named** (§4.1) — the output is directly printable and directly
  falsifiable against the event log.

**`SWITCH` is the highest-value single slot and the cheapest to learn**: always observable,
belief-free, and it flips the whole decision — at `P(SWITCH) = 0.7` this turn's expected incoming
damage is ~0.3× its attack-conditional value, and setup or a slow pivot is often correct against a
hit that is not coming. Nothing in the model represents that today.

### 5.3 Component 3 — the reduction that consumes it

```
cell[j] = Σ_k α_k · pair_in[k, j, :]
```

Contract W from `design_pair_reduction.md` §3.1, now with something worth weighting. Two properties
come free:

- `Σα = 1, α ≥ 0` ⇒ output stays an HP fraction, in `[0,1]` — so it remains auditable against the
  sim and lands in the responsive region of the pointer head's `tanh` rather than the saturated
  tails.
- Have `φ` emit second-order terms and the same weighted sum yields `E[o]` and `E[o²]`, so `ρ` can
  form `Var = E[o²] − E[o]²`. **A learned combination of mean and spread is a learned risk
  attitude** — hedging becomes expressible, one ply, no search. `max` cannot produce a second
  moment.

`R0 hard_max` stays shipped beside it. The measured regret is asymmetric: adding a statistic costs
dims, removing one can cost a generation.

---

## 6. Gates

| # | gate | needs a run? |
|---|---|---|
| **G0** | Component 1 is **byte-identical** while the reduction still maxes damage-only | no |
| **G1** | **Physics oracle extended** — the constructed single-turn scenarios that already prove the damage channels against the sim (`damage_op_probe_fuzz_test`) must cover `p_status_land`, `neutralization`, `tempo_cost`. *This is what "robust" means operationally: every channel falsifiable against the simulator, not asserted.* | no |
| **G2a** | **SEAT COVERAGE — runs FIRST, needs no head at all.** On existing traces: how often does the belief's top-K contain the move they clicked, and how often is a switch to a revealed mon? These two rates are the ceilings on `α` and `β` respectively (§4.5). Poor coverage ⇒ fix the belief (turn on `move_belief_coef`) / turn on **B1** before building anything. | no |
| **G2b** | **THE DECISIVE EARLY GATE — does `α` beat `w` at predicting their actual click, and `β` beat the alive-bench base rate?** Offline, on existing traces. If not, **stop**: the rest of the chain has nothing to stand on. Same shape as the hidden-team belief pre-build probe, which cleared at +7pp recall / +8–10pp top-1 before anyone built the head. | no |
| **G3** | **`α` and `β` are CALIBRATED** — reliability curves via the prober's `calibration` verb. Accurate-but-overconfident produces worse hedging than `w`, because the whole point is to preserve spread. | no |
| **G3b** | **INTERPRETABILITY (the owner constraint, made executable).** Every decision must render `α`/`β` as a ranked list of **named** moves/mons summing to 1, with **no mass on anything the belief does not hold** — asserted as a test, not just displayed. Surfaced per decision in the prober. | no |
| **G4** | **Leak-free** — the graph invariant + the new bridge fuzz (§4.8) | no |
| **G5** | **Identity at init** — `α` zero-init to `normalize(w)` ⇒ the whole design is byte-identical when first switched on. Assert on a **real `MaskablePPO`-built policy**, not a bare module (ledger **M1**: SB3's ortho-init destroyed every zero-init in the extractor for the entire pre-2026-08-01 history). | no |
| **G6** | **Bake-off re-run** with components 1+2 present — the honest version of the test G1-FINAL ran without them | no |
| **G7** | **Cost** — no measurable regression on the compiled B=1 CPU path (`--compile-extractor`, currently 0.976 ms), a shipped 6.53× lever | no |
| **G8** | **Acceptance** — anchored ELO generation-vs-generation, plus a behavioural readout: switch choices become sensitive to their *likely* action, not their *possible* actions | yes |

**G0–G7 need no training run.** The whole chain is decidable offline beside a live generation, and
three separate gates can kill it before anything expensive happens.

---

## 7. Build order

| # | step | gate |
|---|---|---|
| **0a** | **Measure seat coverage on existing traces** — no head, no training, no GPU. The cheapest thing in this document and it gates everything. | **G2a** |
| 0b | **The belief generation (§4.5).** `--move-belief-coef 0.05` (resume-mutable) **+ `--spread-belief` / `--spread-belief-coef` / `--spread-belief-nature` (STRUCTURAL ⇒ fresh run required)**; **B1** only if 0a's unrevealed-switch rate warrants it. B-spread is the physics fix and must land here — a better `α` over de-timid physics inherits the distortion. | coverage re-measured; `belief/spread_*` (mae, largest_bias→0) |
| 0c | Fix the G1-FINAL skyline (§9.1), re-run with the new coordinates | discriminates "grid exhausted" vs "wrong currency" |
| 1 | Unify `pair_in`; add `neutralization` + `tempo_cost` | G0, G1 |
| 2 | Label plumbing — `opp_action_target` / `_mask`; seat matching by canonical id; mask on belief miss | G4 |
| 3 | `α` head (K seats + `SWITCH`) + supervision, stop-grad from RL | **G2b**, G3, G3b |
| 4 | Wire `α` into the reduction (`how=intent`), zero-init to `normalize(w)` | G5, G6 |
| 5 | `β` over their mons; wire into `d4` **and** the (bench × bench) offense grid (§4.4) | G2b/G3 analogue |
| 6 | Ship at a generation boundary | G7, G8 |

**Retrain-class** — new cell widths and new heads ⇒ `MODEL_CONFIG_VERSION` bump and a fresh lineage.
Gen-7 is spoken for by the v63 quantile arm; this is gen-8 material at the earliest.

---

## 7a. REVIEW (2026-08-11) — scope gap, one alternative hypothesis, and a fallback arm

Four notes from a review of this doc against the live measurements. None contradicts the design;
one names something it does not cover.

### 7a.1 ⚠️ THIS IS A POLICY-SIDE DESIGN — the measured regression is CRITIC-side

The whole chain terminates at the **pointer logits**, which are policy-only. Meanwhile the
critic's deficit is now MEASURED, and it is separate:

| run | dense `ladder_elo` @24M |
|---|---|
| gen-4 (concat alive) | **2081 ± 11** |
| gen-5 (no concat)    | **2037 ± 11** |

The sparse `eval/elo` at ±30 could not resolve this and reported "parity"; the dense ladder can, and
supersedes it. What replaced the concat for the critic — `MultiSeedValueReadout` — has measured at
**~1 effective direction** under two structurally different pressures (VICReg repulsion, gen-6;
per-seed quantile assignment, gen-7: centered PR 0.846 vs 0.835, identical).

#### ⚠️ Amendment (2026-08-11) — "cost ~44 Elo" is SUGGESTIVE, not established

Read back from the persisted `snapshot_ladder/ladder.json` of both runs (converged, all pairs —
55/55 and 66/66). Three corrections, in increasing order of importance:

**The ±11 above is the STANDARD ERROR, not the CI.** `se` = 10.70 (gen-4) and 10.10 (gen-5), so the
95% intervals are **[2059.6, 2101.6]** and **[2017.6, 2057.2]** — disjoint by **2.4 Elo**. Marginal,
not comfortable. Quoting ±11 makes the separation read far more decisively than the data supports.

**The trajectories INTERLEAVE, and the whole gap is one endpoint:**

| step | gen-4 | gen-5 |
|---|---|---|
| 14M | 2021.1 | **2025.2** |
| 16M | 2030.2 | **2036.6** |
| 18M | 2049.3 | 2029.9 |
| 20M | 2050.9 | 2050.0 |
| 22M | 2050.3 | 2036.2 |
| **24M** | **2080.6** | 2037.4 |

gen-5 is *ahead* at 14M and 16M and dead level at 20M. **The entire 43-point gap comes from gen-4's
final checkpoint jumping +30 while gen-5's stayed flat.** Selecting the endpoint from two
trajectories that cross repeatedly is exactly the outcome-conditioning the ledger's own honesty
gates name — a single-point comparison here cannot separate "the concat deletion cost Elo" from
"gen-4's last snapshot was a good draw."

**The cheap disambiguation** (no new battles for the first): fit both over their last 3–4
checkpoints instead of the endpoint, and/or ladder gen-4's 22M directly against gen-5's 24M.

**What this does NOT weaken.** §7a.1's conclusion — *this is a policy-side design and the critic
deficit is separate* — **does not rest on the Elo number at all.** The seed readout measuring ~1
effective direction under two independent pressures is a solid, self-standing, critic-side finding.
So the scope gap is real on structural grounds and **§7a.2 remains the right response whether the
concat cost 43 Elo or 0.**

**So `α` does not, on its own, repair the critic.** The two deficits are complementary:
the policy cannot express *"they'll click this, so this is my answer"*; the critic lost its
magnitude window. Do not let this doc be read as answering both. **§7a.2 is the resolution** — the
same `α`-reduced object serves both heads, and it is the first critic route that is equivariant in
BOTH axes.

### 7a.2 The critic route that falls out of this design (and why it is equivariant)

`cell[j] = Σ_k α_k · pair_in[k, j, :]` is a **per-entity, magnitude-carrying, coherent** row. Send
it to the two consumers by their physical channel (`design_conditional_opponent_cells.md` §0.1):

* **policy** → the pointer switch cell (per-action absolute at the logit) — this doc's component 3;
* **critic** → **token content on our mon `j`'s token**, which `value_cls` then pools.

Equivariance, per axis:

| permute | why it holds |
|---|---|
| **their moves (`k`)** | `α` = softmax over seats with `g` SHARED over `k` ⇒ `α` permutes with the seats ⇒ `Σ_k α_k·pair_in[k,j]` is **INVARIANT** |
| **our mons (`j`)** | the row is computed per `j` and injected onto mon `j`'s own token ⇒ **EQUIVARIANT** (it rides the entity) |
| **the readout** | `value_cls` is softmax attention over tokens ⇒ its output is **permutation-INVARIANT** |

No positional axis is introduced anywhere — unlike the deleted concat, which was a slot-ordered
flatten, and unlike a widened seat (§6 item 7a of the OA doc), where the within-seat axis stays
positional. It also needs **no seeds**, which matters given the two-generation seed-collapse result.

**Two honest limits.** (1) This restores *conditional per-mon threat*, NOT **cross-pair joint**
reasoning (*"their Ice Beam threatens three of my mons"*) — that still needs PV or pair-token
promotion, and the §2b.4 coverage probe is what decides whether the critic misses it. (2) The
`in_permon` anti-pattern (`design_conditional_opponent_cells.md` §0.4) still stands **against an
incoherent max collapse**; a coherent `α`-weighted contraction is a different object (a conditional
expectation, the quantity a pivot decision actually needs) — but "a collapse cannot deliver the axis
it collapsed" remains true for the joint questions.

**This can be tested BEFORE `α` exists.** Substitute `α := normalize(w)` (the ladder's R1
`belief_mean`, already implemented in `agents/model/pair_reduce.py`) and inject that row as token
content. That separates the two claims cleanly: **R1-into-tokens tests the DELIVERY channel; `α`
later upgrades the DISTRIBUTION.** A null on the first is about plumbing; a null on the second is
about intent.

### 7a.2b THE BUILD (gen-8 arm) — `--value-threat-inject`, and where it goes

Measured first (G2a, gen-5 traces, 400 battles / 16410 decisions, `tmp/g2a_seat_coverage.py`):
**α's ceiling is 89.3%** (the K=6 seats already hold the clicked move in 1101/1233 nameable-move
decisions) — the discrete constraint is affordable. **β's ceiling is 53.6%**: switches are **24.7%
of ALL decisions**, but 46.4% of them bring a previously-unseen mon, which v1 masks. So B1 is what
β v1 is waiting on, and the {ATTACK, SWITCH} form (§7a.4) is attractive on its own merits.

**Where the injection goes, and why not the obvious place.** `prefuse_proj` already writes op rows
onto tokens PRE-transformer — and gen-5 had it and did not come out ahead, so "inject earlier" is
not the fix. Two things differ here: the CONTENT is a coherent α-weighted contraction rather than
`in_permon`'s incoherent per-defender max, and the PLACEMENT is **post-transformer, immediately
before the value pool**, so the trunk cannot wash it out:

```
row_j        = Σ_k α_k · pair_in[k, j, :]        # α := normalize(w) in v1 (the shipped R1 rung)
aug_j        = our_token_j + W_inj(row_j)        # W_inj ZERO-INIT, shared over j
value_pooled = attend(value_cls, [aug_our_tokens, their_tokens])   # VF ONLY
```

**The policy path keeps the UNAUGMENTED tokens**, so `pi` is bit-identical ON vs OFF — not just at
init but for any `W_inj`. That is the strongest available statement that this arm changes the
critic and nothing else, and it is a test, not a claim (§7a.2c).

v1 deliberately uses `α := normalize(w)`: it separates **DELIVERY** (does a coherent magnitude row
reaching the critic un-mixed help?) from **DISTRIBUTION** (does a supervised `α` beat `w`?). Both
are needed; conflating them makes a null uninterpretable.

### 7a.2c Gates for the injection arm

| # | gate | needs a run? |
|---|---|---|
| **V0** | ON == OFF **bitwise** at init on a REAL `MaskablePPO`-built policy (zero-init `W_inj`; ledger **M1** — SB3's ortho pass destroys extractor zero-inits, so `restore_identity_init()` must cover it) | no |
| **V1** | **`pi` is bit-identical ON vs OFF for ARBITRARY `W_inj`** — the vf-only claim, asserted rather than assumed | no |
| **V2** | **Equivariance**: permute our six mons ⇒ `value_pooled` **invariant** (permutation-invariant pooling over per-entity rows) | no |
| **V3** | Version gate: flipping the flag FATALs on resume (structural, state_dict-changing) | no |
| **V4** | Cost: no measurable regression on the compiled B=1 path (a shipped 6.53× lever) | no |
| **V5** | Acceptance: dense **`ladder_elo`**, fit over the **last 3–4 checkpoints** rather than the endpoint (per §7a.1's amendment — the gen-4/gen-5 trajectories interleave and the 43-point gap is one endpoint, so a single-point target would re-commit the same error). Compare against gen-4's tail, and treat 'recovering the ~44' as a hypothesis under test, **not** the established baseline | yes |

### 7a.3 An alternative hypothesis for `d3` = 0.63% that this doc does not raise

§1 reads `d3`'s decay as a CHANNEL failure (an edge carries a ratio, not a magnitude). Plausible —
but a channel carrying *distorted content* would also read low, and §4.5 establishes that the
content **is** distorted: every opponent's offense is priced as 252 EV × 1.1 nature, uniformly, at
nine sites, with an error that scales with base stats and therefore corrupts the RELATIVE threat
ordering across their team — precisely what `d3` exists to convey.

So there are two live explanations for the same number. **B-spread (step 0b) prices them cheaply and
is already first in the build order** — so re-measure the `d3` family flip rate after the belief
generation, BEFORE concluding the channel was the problem.

#### Refinement (2026-08-11) — they are NOT mutually exclusive, and the design does not fork on it

An earlier phrasing of this note framed the two as alternatives implying different next moves
(*channel* ⇒ build `α`; *content* ⇒ fix B-spread and `d3` may recover). That over-reads it.

**The channel argument is STRUCTURAL and holds at any content quality.** An edge bias writes a
softmax-normalised ratio within its row; it cannot deliver a per-action absolute *however accurate
the numbers being ratio-ed are* (`ARCHITECTURE.md` §5.3, §1(a) here). Fixing the de-timid defect
makes `d3` carry a **correct** ratio — it does not make an edge able to carry a magnitude, and it
does not move the `argmax_a` computation off the pointer logits (§1(b)).

So both can be true simultaneously, and almost certainly are:

| | claim | fixed by |
|---|---|---|
| **content** | `d3`'s numbers are distorted (de-timid) ⇒ any channel carrying them reads low | B-spread, step 0b |
| **channel** | an edge cannot deliver a per-action absolute ⇒ the anticipatory read has no adequate route | the per-action cell — components 1–3 |

**What the post-B-spread re-measurement actually buys is a MAGNITUDE, not a verdict:** it says *how
much* of the 0.63% was content. A large recovery means the de-timid defect was suppressing a channel
that works better than we thought; a small one means the channel was the binding constraint. Neither
outcome removes the need for the per-action route, and **neither should be treated as a fork in the
build order** — step 0b runs first either way, and its `d3` reading calibrates expectations for
components 1–3 rather than deciding whether to build them.

### 7a.4 The coverage-risk fallback: `α` over {ATTACK, SWITCH} only

§8 names low seat coverage (G2a) as a way this dies. The degenerate version is immune to it and
should be **pre-registered now rather than improvised under pressure**:

> `α ∈ Δ²` over **{ATTACK, SWITCH}** — one scalar, **belief-free**, always observable from the
> event log, zero mask rate, no B-move prerequisite.

§4.4(1) and §5.2 already say this slot carries the largest single effect (at `P(SWITCH)=0.7` this
turn's expected incoming damage is ~0.3× its attack-conditional value, and nothing in the model
represents that today). If G2a returns poor coverage, ship this instead of stalling on the belief.

### 7a.5 Schedule honesty — this is TWO generations, not one

"G0–G7 need no training run" is true of the GATES, but build step **0b** does: `--spread-belief`
is STRUCTURAL and fresh-only. So the realistic shape is **gen-8 = the foundation generation**
(B-spread/de-timid + `--move-belief-coef 0.05`) and **gen-9 = `α`**. Worth stating plainly so this
is not planned as "all offline until one big run."

---

## 8. What could kill this, honestly

**`α` ≈ `w` (the G2b risk, and the one I would watch).** In gen3 OU the answer to "what will they
click" often genuinely *is* "their strongest damaging move," which `w`-weighted damage already
approximates. If so, `α` is a small correction and the chain's value collapses to component 1's new
coordinates. **That is a real finding cheaply bought** — which is exactly why G2b gates step 4.

**Seat coverage is too low to support a discrete `α` (the G2a risk, and the constraint's real
cost).** Confining `α` to the belief's seats buys interpretability and pays for it with a ceiling:
if the top-K rarely holds what they clicked, most samples are masked and `α` trains on a thin,
possibly-unrepresentative slice. **The design deliberately does NOT paper over this** — an earlier
draft's property-similar soft targets would have kept the sample count up while hiding the very
rate that tells us the belief needs work (§4.2). **This is why G2a runs first and needs nothing but
existing traces**, and why
`move_belief_coef` (currently `0.0`, label already emitted and plumbed) is a precondition rather
than an adjacent project. A low coverage rate does not kill the design; it redirects the next
increment to the belief head, which is a cheaper fix than anything downstream.

**Non-stationarity destabilises training** (§3.3). Mitigated by pool-mixture training and the
stop-grad; if it still bites, `α` can be frozen periodically like a target network.

**The reduction still cannot use it.** Possible even with `α` good and currencies present — in which
case the bottleneck is delivery (the pointer cell), not representation, and the next move is
`design_conditional_opponent_cells.md`'s per-action route rather than a better statistic.

---

## 9. Open questions

**9.1 The G1-FINAL skyline is probably underpowered, and it is cited widely.** It was fit with LBFGS
on a linear model with L2 = 1e-3 and an 80/20 split — at n=299 that is **~239 training rows against
2800 parameters** for the SKYLINE arm, while the R0 arm (tens of dims) is well-conditioned. The
asymmetry runs the wrong way: the skyline is penalised by its own dimensionality, so 0.413 may
under-state the grid's linear information and make R0 look artificially close to a ceiling. G1 v1
recorded this caveat explicitly ("overfit-limited … cannot support a 'no headroom' claim"); the
FINAL entry reads as though it lapsed, though n went 101 → 299 and the 2800 dims did not move.
**Before either reading is trusted:** tune the L2 by cross-validation, or PCA to ~100–200
components, or raise n. Sanity check: a skyline that cannot beat a strict *subset* of its own input
is measuring its own fit.

**9.2 Should `α` be per-decision or per-turn?** A forced switch and a normal turn are different
decision types. Probably per-decision, but unverified.

**9.3 Does `β` need `α`'s conditioning?** "Which mon they bring" plausibly depends on what they
expect us to do next, recursing the §3 argument one level. Deferred.

**9.4 The level-3 option** — a cheap early "our-intent" head supervised by our own final policy,
letting `α` read our *policy* rather than our *physics*. Strictly more expressive, and strictly more
circular. Named here as deliberately **not** taken (§3.1).

**9.5 `α` against off-distribution opponents.** It amortizes whatever we train against. The human
replay corpus is the natural *validation* set; using it as a training set is a different project.

---

## 10. Provenance

| claim | source |
|---|---|
| `d3` 0.63% / `d2` 19.25% / `d1` 12.17% at gen-4 end-of-run, stratified | `research_state/measurements/gen4_edge_family_audit_25M.json` |
| `d3` was 1.9% at gen-3 9.6M (i.e. it fell) | `ARCHITECTURE.md` §5.4 |
| an edge carries a softmax-normalised ratio, not a magnitude | `ARCHITECTURE.md` §5.3 |
| the edge-family cell contents (`d3` = `[high, pko, eff, is_phys, w]`) | `ARCHITECTURE.md` §5.1 |
| the reduction is one named call site with a `how` knob | `damage_op.py:534` `_chan_max(..., how="hard_max")` |
| status is absent from the pointer switch cell; only `s3` (a ratio) carries it in production | `design_pair_reduction.md` §2.1 |
| G1 FINAL n=299: R0 0.403±0.034 · R1 0.423±0.063 · SKYLINE 0.413±0.037 | `research_state/ledger.md` |
| the G1 probe's fit (LBFGS, L2 1e-3, 80/20) | `tmp/g1_bakeoff.py:175-195` (read 2026-08-11) |
| opponent moves + switches are recoverable with side attribution and delegation-awareness | `agents/battle/battle_event.py` — `EventKind.MOVE`, `side`, `move_id`, `delegating_move_id` |
| the training-only-key pattern, and `win_target` as precedent for a FUTURE label | `ARCHITECTURE.md` §7 |
| no aux edge may reach the forward | `delivery_graph_test.test_no_aux_edge_reaches_the_forward` |
| Hungarian / order-invariant belief supervision precedent | `belief_labels_fuzz_test.py`; `project_hidden_team_belief_built` |
| typed HP nums 355–370 vs the bare 237 for unrevealed | `gen3_typed_hidden_power_ids_v1`, root `CLAUDE.md` → Data Dependencies |
| SB3 ortho-init destroys extractor zero-inits | ledger **M1** |
| the opp-action head was falsified as a SIDE OUTPUT | `project_opp_action_head_falsified`, `project_l3_oracle_grind_l4`; retraction of the VoI-ceiling reading in `design_pair_reduction.md` §10.8 |
| no search on the model (hard constraint) | `research_state/README.md` → amortizability gate |
| **the DISCRETE constraint** — the model must always pick among the belief's discrete states and may never invent a move; interpretability is the reason | **owner, 2026-08-11** |
| the two reconciliations (§3, §4) | owner, 2026-08-11 |
| **the dense-ladder Elo values and their `se`** — gen-4 @24M **2080.6 / se 10.70**, gen-5 @24M **2037.4 / se 10.10**, both converged (55/55 and 66/66 frozen pairs) | read back from `<run>/snapshot_ladder/ladder.json` for both runs, 2026-08-11 (`agents/training/snapshot_ladder.py`) — the §7a.1 amendment's `[2059.6, 2101.6]` vs `[2017.6, 2057.2]` at 1.96·se, and the per-checkpoint trajectory table |
| `move_belief_coef` = **0.0** in production (belief head runs, `known_moves` emitted + plumbed, BCE **unconsumed**); `move_belief_latent_coef` 0.05, `hp_type_belief_coef` 0.05 on | `designs/production_config.json` (read 2026-08-11); `ARCHITECTURE.md` §7 |
| the full **belief stack** table (§4.5) — 7 legs, exactly 1 supervised, 2 unlearnable static lookups, 3 off | `designs/production_config.json`; `damage_op.py:2870-2877` (`p_cb` = `SPECIES_CB_PRIOR`, collapses to 0/1 on reveal); `agents/observation/abilities.py` + `reactive.py` (Smogon per-species ability priors); `gen3_ability_priors.json` |
| **the de-timid physics defect** — opp offense priced as 252 EV × 1.1 nature uniformly, at 9 sites | `damage_op.py:1727` (`atk_j = (2.0·a_base[ATK] + off_const) * 1.1  # de-timid`), also `:332`, `:1489`, `:1498`, `:1714`, `:1882`, `:2127`; `:118` names `--spread-belief` as the replacement |
| `--move-belief-coef` is **training-only / resume-mutable**; `--spread-belief` is **STRUCTURAL, version-checked, fresh-only** | `train_rl_agent.py:1007-1012`, `:1558-1596`, `:2094-2105`, `:3832` |
| no ledger/changelog entry records `move_belief_coef` being disabled after a failure | searched `CHANGELOG.md`, `research_state/ledger.md` (2026-08-11) |
| B1 hidden-team belief: BUILT 2026-06-13, **never run**, acc 0.08–0.16 vs ~0.003 chance | `research_state/ledger.md` B1; `levers/hidden_team_belief.md` |
| a repulsion penalty buys SPREAD, not MULTIPLICITY (⇒ supervised discrete structure over penalised latent structure) | gen-6 measurement; `ec32c93` (v63 per-seed quantile assignment) |

## See also

- `design_pair_reduction.md` — the reduction operator, the coherence contract, the ladder (component 3)
- `design_conditional_opponent_cells.md` — OA1/OA2, the query-conditioned successor
- `design_op_tensors.md` §3.2 — `REDUCE(pair_in, over=MOVE_AXIS, how=…)`
- [[marginalization_and_uncertainty]] — convex combinations; why Contract W preserves units
- [[shortcut_learning_and_feature_delivery]] — never collapse an axis you must choose along
- [[entity_tokens_biases_pointers]] §6.9 — invariance vs equivariance; the rank-*h* trade
