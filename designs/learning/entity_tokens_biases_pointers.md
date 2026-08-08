# Entity-based modeling — tokens, edge biases, pointer heads

> **What this is.** A durable explainer for the concept cluster the ai_v9 generation is built
> on: what "entity-based" (a.k.a. entity-centric / object-centric / relational) modeling *is*,
> where it came from, why it beats a flat MLP, how a computed per-pair quantity like **expected
> damage** gets delivered, and how **history** is represented once positions stop being
> identities. Intuitive first, then technical, no code. Grounded in our `PokemonEncoder` /
> `TeamTransformer`, the `DamageOperator`, the v51 `PointerNativeActionHead`, and the 7×159
> `TurnDelta` block.

---

## TL;DR

- **The one idea:** *positions are not identities.* A flat vector where dim 217 means "mon 3's
  HP" forces the network to memorize a map and re-learn "HP" six times. An entity model
  represents the state as a **set of things** carrying attributes, processed by **one shared
  function** — so meaning comes from content, not from slot.
- **Lineage:** it is the same weight-sharing-under-a-symmetry idea that made CNNs work (1989),
  generalized from translation to **permutation**. GNNs (2005–09) → Interaction/Relation Nets
  (2016–17) → Deep Sets/PointNet (2017, the theory) → Transformers (2017, the vehicle) →
  AlphaStar/OpenAI Five (2019, at scale) → AlphaFold2 (2021, edge biases done right).
- **Four pieces of vocabulary.** **Entities** (the things). **Tokens** (fixed-width seats at
  the attention table + a type embedding). **Edge biases** (externally computed per-pair terms
  added to attention scores — AlphaFold's trick; costs zero token dims). **Pointer heads**
  (action logits read from the token of the entity they select).
- **Why better, concretely:** weight sharing multiplies effective data (12 mon slots → 12× the
  samples for one encoder); invariance shrinks the hypothesis class (a real generalization
  bound, not vibes); and whole **bug classes become unrepresentable** (the sorted-vs-request
  ordering defect, F2's unreachable bench token).
- **The cost, honestly:** equivariant models are strictly **less expressive** by construction,
  attention is O(n²) in tokens, and *we measured that structure is not automatically a win* —
  physics-into-the-trunk was NULL 3-for-3 (ledger K9/K10) while the crude flat head-concat
  carried the policy's largest dependency (P1).
- **Expected damage lives on an EDGE**, because it is a property of the (move, defender, board)
  **triple** — an *activation* recomputed every forward, never a weight. The move token carries
  only invariants (BP/type/category/accuracy); randomness rides as a distribution summary
  `[low, high, crit, pko]`.
- **History follows the same sorting rule:** most of it is Markovian residue that belongs as
  **state on the entity**; the sequential remainder becomes **its own token type with recency
  as content**. Recurrence is RULED OUT — not for plumbing reasons, but because it breaks the
  invariant the whole forensic stack rests on (obs = a pure function of the event log).
- **The sorting rule is a COMPOSITION CONTRACT, not a filing system** (Part 6). It partitions
  facts by *arity* and *certainty* — both properties of the fact, never of the model — so every
  new fact lands in exactly one place with no cross-cutting rewiring, and families compose as
  ingredients (G's per-mon HP ledger became C4's Protect consequence for ~free). Adding a family
  touches four lines: a kernel, a cell width, a placement branch, a gate string.
- **Edges ROUTE; they never carry payload.** A softmax row is shift-invariant, so a bias can
  only express *contrast within a row* — never a level, never a magnitude past the point the
  row saturates. Magnitude reaches the decision through the **pointer cells** (exact, per-action)
  and the critic through the **op concat**. That is why we run three routes, and why the first
  real family audit (gen-1 @9.6M) reads **outgoing dominant, incoming near-decorative** — a
  replication of the ledger's P1 shape, with three falsifiable explanations in §6.2.
- **The heads are a severe funnel** (§6.5): 35 seats → three pooled 128-vectors for the policy and
  **one** for the critic. The op concat is the only un-pooled route for *both*; the pointer head is
  a second un-pooled route for the *policy only*. v51 made the policy entity-native and left the
  critic reading one microphone — which is exactly why D2 carries the largest measured `|ΔV|`.
- **Search is not what the edges do** (§6.6). C edges are a depth-1, opponent-static, pure-function
  calculator; search advances the world and takes a maximum. The entity work makes search cheaper to
  *aim* (equivariant candidate generation, C-deltas as a pruning layer before the expensive clone) —
  and the no-recurrence decision is what makes clone-and-branch search legal at all. The honest
  correction: in a simultaneous-move imperfect-information game the object is an **equilibrium
  strategy, not a best path**.
- **Equivariance and conditioning are two halves of one decision** (§6.7): *share weights where a
  symmetry is real, un-share along axes where it is false.* Edge biases and FiLM are the same
  hypernetwork shape at different **clock speeds** (computed-per-forward vs learned-per-battle);
  LoRA would attach to the shared functions entity design manufactures — but the conditioning ladder
  is under two independent measured nulls, so that stays an open, not a plan.

---

## Part 0 — The altitude view (read this before anything else)

If you remember one sentence: **the entire generation is one refactor — "stop letting POSITION carry
meaning" — applied to four different axes.**

The old model saw the battle as one long list of numbers where meaning came from *where* a number sat:
dim 217 means "mon 3's HP", logit row 7 means "switch to slot 1", the second 159-dim block means "two
turns ago". Every one of those is a **convention the network has to memorise**, separately, per
position. The new model sees a set of **things** with seats at a table, relationships **computed** by a
calculator and whispered into the conversation, and decisions **read off the thing being chosen**.

Three ideas, and they are the same idea three times:

| | the move | the axis it de-positions |
|---|---|---|
| **Things get seats** (tokens) | one shared encoder over every entity; identity comes from content | team slots, move slots |
| **Relationships get computed and whispered** (edge biases) | we *know* the damage formula — compute it, deliver it as "these two matter to each other" | the pairwise structure that had no home at all |
| **Decisions are read off the chooser** (pointer head) | the logit for "switch to Zapdos" is computed from *Zapdos's own token* | action indices |
| *(still to come)* **Time gets content, not slots** | "three turns ago" becomes a feature, not a weight range | the history block |

```
        BEFORE                                   AFTER
   ┌──────────────────────┐            ┌────┐┌────┐┌────┐┌────┐┌────┐
   │ 2925 numbers in a    │            │mon ││mon ││move││thrt││glob│   ← things, seats
   │ row; meaning = index │    ──►     └─┬──┘└─┬──┘└─┬──┘└─┬──┘└─┬──┘
   │                      │              └──┬──┴─────┴──┬──┴─────┘
   │ one Linear → 11      │            attention, biased by computed physics
   │ logits; meaning=row  │                 │
   └──────────────────────┘            score each action FROM its own thing
```

**What it buys.** Weight sharing (one mon-encoder trained by all 12 slots ⇒ ~12× the data per
parameter). Generalisation (you cannot overfit to a distinction you cannot represent). And whole bug
classes become *impossible to write* rather than *guarded against*.

**What it costs.** Strictly less expressive by construction; a genuinely position-specific convention
now has to be promoted to explicit content; and the action head can no longer form arbitrary
relationships *between* candidate actions — that burden moves upstream into attention.

**Where we are.** Stage 0 (pointer head) ✅ · Stage 1 (move/threat/tail seats) ✅ · Stage 2 (fifteen
edge families, incl. the whole C consequence set) ✅ · Stage 3 (one declarative schema; the flat
vector dies) — schema view + generator half in · **history: rung 1 shipped** (`gen3_entity_recency_v1`,
obs 2889 → 2925), and Part 4 explains why the rest is deliberately last rather than procrastinated.

Parts 1–5 are the detailed version; Part 6 is why the pieces compose.

---

## Part 1 — Where entity-based design came from

### The intuitive story

Deep learning's first real win was **weight sharing under a symmetry**. A CNN never learns
"edge detector at pixel (14,7)" — it learns *one* detector and slides it. That is translation
**equivariance**: shift the input, the output shifts with it. The field then spent thirty years
rediscovering that the same trick applies to **things** rather than pixels.

The pressure came from a specific failure. Fodor & Pylyshyn's 1988 **systematicity** critique
of connectionism (if you understand "John loves Mary" you should get "Mary loves John" for
free — nets didn't) and the related **binding problem** (how do you represent "the *red* square
is *left of* the *blue* triangle" in one fixed vector without the attributes smearing?).
Smolensky's tensor-product representations (1990) were the first serious answer and were
unwieldy. The practical answers arrived in waves:

| Era | Idea | Contribution |
|---|---|---|
| 1990s | RAAM (Pollack); recursive nets over trees (Goller & Küchler; later Socher) | structure-shaped computation, not fixed-shape |
| 2005–09 | **Graph Neural Networks** (Gori; Scarselli) | message passing over nodes/edges as a general primitive |
| 2016–17 | **Interaction Networks** (Battaglia); Neural Physics Engine (Chang); **Relation Networks** (Santoro) | learn *pairwise* functions; generalize across object counts |
| 2017 | **Deep Sets** (Zaheer); **PointNet** (Qi) | the theory: any permutation-invariant function is `ρ(Σ φ(xᵢ))` — pooling over per-item MLPs is *universal*, not a hack |
| 2017 | **Transformer** (Vaswani) | the vehicle: a set-to-set function, one shared weight set, content-based routing |
| 2018 | Battaglia et al., *Relational inductive biases, deep learning, and graph networks* | the manifesto that named the field |
| 2018–19 | **Deep Relational RL** (Zambaldi); **AlphaStar**; **OpenAI Five** | entity list → transformer → pointer-style selection, at scale, in games |
| 2021 | **AlphaFold2** | the *pair representation* biasing attention — the precedent for our Stage-2 edges |
| 2021 | **Geometric Deep Learning** (Bronstein, Bruna, Cohen, Veličković) | the unification: CNN/GNN/Transformer/DeepSets differ only in the symmetry group |

The move that is easy to miss: **the Transformer was not invented as a language model.** It is
a permutation-equivariant set function with positional encodings bolted on so it can pretend to
handle sequences. Strip those off, add a per-entity **type embedding**, and you have the
canonical entity encoder. Most of "entity-based modeling" is exactly that substitution.

The RL-specific ancestor of our action head is **Pointer Networks** (Vinyals 2015), which
solved "how do you output a permutation of a variable-length input?" by making the output
distribution *be* an attention distribution over the inputs. AlphaStar used it to select which
unit to command out of hundreds.

### The technical statement

You are choosing an **inductive bias** by choosing a symmetry group `G` and demanding the
function respect it:

- **Invariance** `f(g·x) = f(x)` — a board's *value* doesn't change if you renumber your team.
- **Equivariance** `f(g·x) = g·f(x)` — renumber your team and the **switch logits renumber with
  it**.

For sets, `G = Sₙ` (the symmetric group). Deep Sets characterizes the invariant case exactly as
`ρ(Σ φ(xᵢ))`; attention gives the equivariant case with content-dependent routing.

Three mechanisms turn that into measurable payoff:

1. **Weight sharing → effective data multiplication.** One shared `PokemonEncoder` over 12
   slots means every gradient step supplies 12× the samples for that function. A flat MLP
   splits its data across 12 disjoint weight blocks.
2. **Hypothesis-space reduction → a generalization bound.** Restricting to `G`-invariant
   functions shrinks the class roughly by the orbit size; there are clean results quantifying a
   strict generalization gain for invariant models in the linear/kernel setting (Elesedy &
   Zaidi 2021; related bounds in Bietti et al.). Intuition: you cannot overfit to a distinction
   you cannot represent.
3. **Bug classes become unrepresentable.** If move logit *k* is *computed from* the entity at
   request-slot *k*, a sorted-vs-request misalignment cannot occur — not "is guarded against."
   We maintained `src/agents/action/ordering_integrity.py` purely to police that class.

**The honest cost.** Equivariance buys sample efficiency and correctness, **not** expressiveness
— an equivariant model is strictly *less* expressive by construction. That is the point, and it
is also the failure mode: if the true function is not equivariant, you have excluded it.
Attention is O(n²) in token count, sets require masking machinery, and a large enough flat MLP
with enough data will simply memorize the map.

---

## Part 2 — The four pieces of vocabulary

- **Entities** — the state is a SET of things, each with an attribute bundle. We are *half*
  entity-based already (shared `PokemonEncoder` over 12 slots → `TeamTransformer`); moves were
  the un-entity part, dissolved into their mon's vector by an MLP — describable but not
  *addressable* ("no seat = the network cannot think ABOUT Rock Slide, only about a
  Tyranitar-that-has-Rock-Slide").
- **Tokens** — the seat at the table: one fixed-width vector per entity + a **type embedding**
  saying what kind of thing it is. Fixed width lets one set of attention weights process any
  mix of types. Attention is the table conversation: each seat decides who is relevant (query·key)
  and absorbs summaries (values). Current trunk: 12 mon seats + 2 CLS note-takers (one pi, one
  vf) **plus a global token and, since v54, the E3/E4 move seats**.
- **Edge biases** — attention normally LEARNS relevance from scratch: `logit(i,j) = qᵢ·kⱼ`. A
  bias ADDS a computed term: `logit(i,j) = qᵢ·kⱼ + b_ij`. A trusted expert (the exact damage
  calc) whispering *"these two — pay attention."* Costs **zero token dimensions** (it modifies
  the attention matrix, not the tokens) and converts sample scarcity into a prior: the model is
  TOLD the 4× move matters on turn one and spends its samples on what to DO about it.
  Precedent: T5 relative-position biases, ALiBi, and above all AlphaFold2's pair biases.
- **Pointer heads** — a flat head maps a pooled summary → 11 logits where slot 7 means "move 2"
  by convention. A pointer head scores each entity's token: `logit("use move k") = f(context,
  move-k's token)`. Falls out: alignment by construction, weight sharing across actions (one
  scoring function, not 11 independent rows), and per-move physics flowing into its **own**
  logit. Lineage: Pointer Networks; AlphaStar unit selection.

### One turn through the machine

Choosing between Rock Slide and switching Swampert: entities get seats; the calculator stamps
edges (Rock Slide→their Zapdos hot: 4×/likely KO; their believed HP-Grass→Swampert: warning);
attention confers (Swampert's token absorbs the threat, Rock Slide's absorbs "their last bird");
the pointer head scores each move/bench token directly — Rock Slide's logit rises because ITS
OWN token carries the hot edge plus composed context. **Physics computes the edges, attention
holds the conversation, pointers choose from the seat that owns the decision.**

---

## Part 3 — Where the expected damage lives

### The sorting rule

**Damage is not a property of a move.** Rock Slide doesn't "do 140" — it does 140 *to that
Zapdos, at those boosts, under sand, with that item, at this HP*. It is a property of the
**(move, defender, board) TRIPLE**. Hence the rule that governs every ai_v9 placement decision:

| Kind of fact | Home |
|---|---|
| pair-varying (damage, status landing, speed order) | **edge** — an activation, recomputed every forward |
| entity-invariant (BP, type, category, accuracy, priority) | **token** — a feature on the entity |
| probabilistic (rolls, crits, accuracy) | **distribution summary** — `[low, high, crit, pko]`, `pko = acc·P(KO|hit)` |
| future-facing (tempo, information value, plans) | **attention** — learned, never tabled |

The edge being an **activation, not a weight** is the load-bearing part: when Reflect goes up,
every affected edge updates *this forward pass*, no learning and no staleness. A purely learned
representation would have to re-derive it from evidence scattered across separate tokens.

### The three delivery routes

1. **Concatenate at the readout — SHIPPED (v51).** `DamageOperator.pointer_cells` slices the
   flat damage block into per-action cells: move cell *k* → `[low, high, crit, pko, p_land,
   known, sec×10]`; switch cell *j* → its incoming per-defender row + the Choice-Band
   conditional tail + (under `--damage-matrices-outgoing-all`) its OAX attacker row. The
   pointer head concatenates those onto the entity token before scoring. Two properties worth
   copying: the **op owns its own layout** (consumers never hardcode an offset; offsets are
   pinned against `decode_damage_block`), and the cells are a *pure slice of the same tensor*
   the projection heads consume, so the two routes cannot disagree on a value.
2. **Bias the attention with it — SHIPPED (v56 `gen3_edge_bias_trunk_v1` + the D2/S1/S3 slice).**
   Small learned map: per-pair
   cell → per-head additive scalar (one head can attend by KO-range, another by expected chip),
   with the full cell available as edge features where marginals need it. Requires a custom MHA
   (stock `TransformerEncoderLayer` takes no per-pair float bias). **Kernel proven on our
   stack** (`src/agents/model/entity_spike_benchmark.py`, 2026-08-03): SDPA with an additive
   float mask matches a float64 reference at max|Δ| 1.2e-7, `bias=0` reproduces `bias=None`
   exactly, and it compiles `fullgraph=True` with **zero graph breaks** — non-negotiable, since
   the compiled-opponent path is a shipped 6.5× lever and a break would be an invisible
   regression.
3. **Make the model learn it.** We deliberately do not. (See `[[objective_richness_and_representation]]`
   and the provide-vs-learn principle: give the model raw KNOWN facts; don't make it
   rediscover arithmetic.)

### The deeper principle: the differentiable expert

The `DamageOperator` is not a learned damage predictor — it is the **actual Gen 3 damage
formula written in PyTorch**, fuzz-validated against constructed Showdown probes
(`damage_op_probe_fuzz_test.py`, one modifier per scenario: type/STAB/SE/resist/4×/immunity/
Thick Fat/Choice Band/item/boosts/burn/screens/weather). Two reasons to bother, and the second
is the good one:

1. **It is a known function.** Capacity and samples spent re-deriving it are wasted.
2. **The gradient flows backward through it into the belief.** The op consumes the *believed*
   opponent moveset from `MoveBelief`; the policy loss therefore reaches the belief head via
   the damage. The physics is a differentiable, correct, hand-written layer that converts a
   *policy* error into a *belief* error — supervision the BCE aux cannot give.

That pattern — **exact known mechanics as a differentiable in-graph operator sitting between a
learned belief and a learned policy** — is the most transferable idea in this codebase, and it
generalizes to any domain with a known simulator component over unknown latent state.
See `[[marginalization_and_uncertainty]]` for why the operator must *marginalize* rather than
mean-field, and what that costs.

### Status, stat, and field moves in this world

The edge bias never encoded "damage" — it encodes **computed mechanical consequence**; damage
was merely the first consequence priced. Every move class gets edges; only the content differs:

- **Status moves** (Toxic/T-Wave/Spore/WoW) are pairwise like damage: the v27/v37 landing
  physics (type/ability/Sleep-Clause/Sub immunities, `[P(major), P(immobilize)]`) becomes the
  bias on the status-move↔defender edge — plus computable CONSEQUENCES: para → the speed-order
  FLIPS it causes; burn → the delta to their outgoing table; Toxic → the HP schedule. Learned
  residue: WHICH status target matters this game (attention's contextual selection).
- **Stat moves** (SD/CM/Curse) are **self-loops carrying hypothetical worlds**: run the damage
  kernel once at boosted stats and write the DELTA on edges to each opposing mon ("SD flips EQ
  vs Swampert 3HKO→2HKO; vs Skarmory nothing"). Setup value = a computed table-diff. Learned
  residue: temporal risk (is the setup turn safe; phaze exposure) — composition, attention's job.
- **Field/side moves** (Spikes/screens/weather/Roar/Recover/Protect) edge to the GLOBAL/side
  token: Spikes = the chip schedule on their grounded bench entries; Reflect = a halving delta
  on my incoming table; Roar = "drags a random bench entry through hazards," priced from
  existing tables; Recover flips their pko cells; Protect rides the computed stall odds.
- **Status CONDITIONS are attributes, not entities.** Current burn, sleep-wake belief, protect
  counter stay as columns on the mon's identity/condition token. *Things get seats; conditions
  of things get columns.*
- **Protect** (the extreme temporal case): the token carries identity + the COMPUTED success
  odds (the gen3 100/50/25/12.5 floored-doubling counter, `gen3_protect_odds_v1` re-homed); a
  self-loop edge carries the computed TURN LEDGER (Toxic/sand/Leftovers/Leech ticks + the Wish
  resolve — deterministic schedules; for stall teams this ledger IS the win condition);
  attention prices the residue — TEMPO (their free turn) and INFORMATION (Protect-scouting),
  never a table.

Migration note: almost no NEW computation. v24 secondary chances, v26 physics, v27 landing, v37
split, wish/protect wiring are already fuzz-validated; the work is **re-homing** them from flat
positional blocks onto the structure matching their shape.

---

## Part 4 — History under the same rule

### The two defects, and they are the ones v51 just fixed for actions

The current history block is 7 turns × 159 dims of `TurnDelta` frames
(`N_HISTORY_TURNS`, v42). It is:

- **Positional in time** — turn *t−3* is a fixed weight range, so the same event at a different
  lag lands on different weights and must be learned separately. (Identical disease to "logit
  row 7 means move 2.")
- **Entity-blind** — raw embedded ids, disconnected from the tokens representing *those same
  entities* in the trunk. The Zapdos in history and the Zapdos in the team block are, to the
  network, unrelated objects.

### The intuition: you need the BALANCE, not the bank statement

In a POMDP, history's job is to supply a **sufficient statistic** — enough of the past to act
correctly *now*. You don't need every transaction; you need the balance.

And the striking thing is how much of Pokémon history genuinely **compiles into a balance**. Toxic
stage, sleep-wake belief, protect counter, pending Wish, choice-lock, revealed movesets, PP, boosts —
every one of those is "history, already summed". This project has been building them piecemeal for a
year without naming the pattern. What *doesn't* compile is the residue: **tendencies** — patterns in
the transactions rather than the total ("they Protect on my Toxic turns", "they're pacing a PP war").
That residue is real, and it is small.

So the design question is not "how do we replay the past?" It is: **which balances are we still not
keeping, and how much genuinely-sequential residue is left over?**

### The right frame: history completes the belief state

In a POMDP, history's job is to supply a **sufficient statistic**, not to be replayed. And most
of the content is **Markovian residue that belongs as state on the entity** — which this project
has been doing piecemeal for a year without naming the pattern: sleep-wake belief, protect
counter, pending Wish, choice-lock state, revealed movesets are all *history compiled into
state*. The irreducibly sequential remainder (opponent tendencies, PP-war pacing, momentum) is
small.

**The decided ladder** (`designs/ai_v9/design_generation_roadmap.md` §4), in landing order:

1. **Per-entity recency features — SHIPPED** (`gen3_entity_recency_v1`, obs 2889 → 2925): three
   per-mon dims at `POKEMON_RECENCY_OFFSET` — `[turns_since_seen, turns_since_acted,
   turns_since_was_hit]` — log-saturated over a 10-turn cap, both sides (all three derive from
   public protocol), sourced from the same `EpisodeTracker` as the progress clock. **TURN-ANCHORED**
   (`cur_turn − event_turn`) rather than tick-then-reset: the fuzz caught the counter form reading
   differently across forced-switch turns, so anchoring on event turns makes it
   processing-order invariant. Gate: `poke_env_gaps/recency_fuzz_test.py` (encoded scalars ==
   an independent full-log recount). Cheap, entity-native, and the event-sourced fold already
   computed all of it — which is why it was rung 1.
2. **A short window of turn/event TOKENS** for the residue, entering attention with a **recency
   embedding** instead of a fixed slot. Equivariant in the token dimension, trivially
   variable-length; the policy can now *query* patterns ("they Protect on Toxic turns") rather
   than have them smeared across a weight range.
3. **Entity-LINKED event tokens** — the end state, gated on an attention-usage audit. Each event
   becomes a token whose actor/move fields ARE the same embeddings as the live entity tokens,
   so history becomes **edges between past events and present entities**. The `BattleEvent` log
   is already the validated source of truth; this is a re-encoding of trusted data, not new
   state tracking.

The general lesson for any entity-centric model: **history is either compiled into entity
state, or it becomes its own entity type with recency as content.** Sequence position becomes
content, exactly as team-slot position did.

### One fact, walked up all four rungs

Take a single concrete thing the model should notice: *"this Blissey has now Protected on two of my
Toxic turns."* Here is what each rung can represent.

```
RUNG 0 — today (7 × 159 positional frames)
  turn t−1 frame: [... opp_move_id = Protect ...]      ← a fixed weight range
  turn t−3 frame: [... opp_move_id = Protect ...]      ← a DIFFERENT fixed weight range
  ✗ "Protect at lag 1" and "Protect at lag 3" are separate weights, learned separately
  ✗ neither is connected to the Blissey token, or to its protect counter
```
```
RUNG 1 — recency ON the entity                                  ✅ SHIPPED (obs 2889 → 2925)
  Blissey's token: [... turns_since_seen, turns_since_acted, turns_since_was_hit,
                       protect_counter = 2 ...]
  ✓ the fact now lives on the thing it is about
  ✓ C4 already reads that counter → the Protect-consequence edge prices it automatically
  ✗ still cannot express the CONDITIONAL ("...on my Toxic turns")
```
```
RUNG 2 — turn tokens, recency as CONTENT     (the sequential residue gets seats)
  seat: [what happened, how long ago = 1]
  seat: [what happened, how long ago = 3]
  ✓ the same pattern at lag 2 or lag 6 hits the SAME weights
  ✓ attention can QUERY: "find turns where I used Toxic" — the conditional becomes expressible
  ✗ the Protect in history and the Blissey in the lineup are still unrelated objects
```
```
RUNG 3 — entity-LINKED event tokens         (gated on an attention-usage audit)
  event token: [actor = ←the SAME Blissey embedding, move = Protect, when = t−1]
  ✓ history becomes EDGES between past events and present entities
  ✓ "THIS mon Protects when I status it" is representable — a per-opponent tendency
```

The escalation is the whole design in miniature: **rung 1 is a balance, rungs 2–3 are statements**,
and you only pay for statements once you can show the balances didn't already cover it. That is also
why the ordering is not arbitrary — rung 1 is nearly free and takes most of the value, so it goes
first and *sets the bar* the later rungs have to beat.

### Recurrence is RULED OUT — and the reason is a tooling invariant

Not because recurrent-PPO plumbing is annoying. Because it breaks the invariant the entire
forensic stack rests on: **observations are a pure function of the replayed event log.**
Reconstruction, reroll-parity, clone-search, and the obs-roundtrip fuzz (offline obs == live
obs, bit-for-bit) all assume the obs can be rebuilt offline; a hidden state threaded across
decisions forfeits every one of them — you could no longer take a saved trace and ask "what if
it had picked a different move on turn 14."

Generalizable point, and one that doesn't appear in coursework: **an architectural option can be
vetoed by a debuggability invariant.** Given how much of this project's progress came from the
prober, that trade reads as correct.

**Deferral, then rung 1 (2026-08-03 → 2026-08-07):** history was deferred past Stages 1–2 — the
7×159 block is a self-contained obs slice with its own encoder path, and Stages 1–2 change the model
side only, so it rode along unchanged (attribution discipline: one fewer simultaneous change in the
riskiest stages). **Rung 1 has now landed** (`gen3_entity_recency_v1`), which is the cheap
sufficient-statistic half — balances, not statements. The 7×159 positional block is still there and
still unchanged; the forcing point for it remains Stage 3, whose minimal port is 7 opaque "history
tokens" through a per-type input projection, preserving every signal without deciding rungs 2–3.
Those wait on the attention-usage audit, exactly as the edge families waited on theirs.

---

## Part 5 — Where this lives in our architecture

- **Already entity-based:** `PokemonEncoder` (one shared per-mon encoder over 12 slots) →
  `TeamTransformer` (12 mon tokens + 2 CLS pools, pi and vf) — textbook Deep Sets → attention.
- **Stage 0 SHIPPED — v51 `gen3_pointer_native_v1`** (`f25e708`, 2026-08-03): the flat
  `action_net` is **deleted** (a raising stub takes its slot; the optimizer is rebuilt) and
  `PointerNativeActionHead` IS the action head. Move logit *k* ← the REQUEST-slot-*k* move token
  ⊕ its op cells; switch logit *j* ← our-team token *j* ⊕ its incoming/CB/OAX cells; struggle ←
  the context; ctx = `latent_pi`, so the op block / beliefs / FiLM condition every score. Zero-init
  scorers built AFTER SB3's ortho-init ⇒ cold start is uniform-over-legal. It structurally
  dissolved **F2** (switch logits read from a permutation-*invariant* CLS pool, so a bench mon's
  token could never reach its own logit — the information was destroyed by the pooling) and the
  ordering bug class. No flag, no off state: the cross-era break rides the `ARCH_SIGNATURE` bump.
- **Stage 1 SHIPPED — v54 `gen3_entity_move_seats_v1`:** moves are attention citizens. **E3**
  (unconditional) — our active's 4 request-ordered move tokens, permuted ONCE pre-transformer by
  move-num identity and projected to `d_model`, enter the trunk as seats appended after the global
  token (existing absolute slices position-stable); **the pointer head now reads the REFINED E3
  seats**, so its move tokens are board-aware. **E4** (`--entity-topk-seats K`) — the opp active's
  top-K believed threat moves as seats `[latent ⊕ w ⊕ acc ⊕ is_phys]`, sourced from the op's
  `refine_candidates(k=K)` (one candidate definition shared with the refine kernels; idx detached,
  `w` differentiable). Measured B=1: E4 K=5 = **+0.18 ms** on a ~3.1 ms prefuse forward — the spike
  predicted +0.19. **E5 (the tail-threat seats) SHIPPED at v57** (`gen3_entity_tail_seats_v1`,
  `entity_tail_seats`): 6 per-opp-mon seats summarising the beyond-top-K tail of that mon's composed
  posterior, cell `[p_tail, worst_phys, worst_spec, revealed]`, K shared with the E4 seats so there is
  ONE truncation definition. They reuse `TOKEN_TYPE_THEIR_THREAT` + a learned `tail_marker` rather than
  growing the token-type table (which would break loading in-generation checkpoints into newer code) and
  are appended LAST so the pointer stash's E3 slice is untouched. This is the truncation insurance the
  bimodal-miss finding in [[shortcut_learning_and_feature_delivery]] asked for.
- **Stage 2 SHIPPED — v56 `gen3_edge_bias_trunk_v1`, now FIFTEEN families:** the encoder
  stack is now `BiasedEncoderLayer` (the spike-proven clone taking an additive per-pair per-head
  float bias; the key-padding mask rides the same tensor as a −1e9 addend). `EdgeBias` delivers
  fifteen families behind `--edge-bias-families` — **D1** (our active's 4 moves × the opp's 6 mons,
  the v34 kernel), **D3** (the opp's top-K believed moves × our 6 mons, over the pre-collapse
  `_incoming_rolls`), **D2** (every our-mon's best offense vs the opp active, the v39 switch-in
  kernel move-collapsed), **D4** (the missing quadrant — the opp BENCH's own top-K threats vs our
  6 mons), **S1/S3** (status-landing both directions, the v27/v37 kernels gaining
  `per_pair` branches), **V** (`pairwise_speed` — P(our mon *i* outspeeds opp mon *j*) for
  EVERY (i,j) pair: our real spread vs the believed spread, public paralysis ×0.25 folded both
  sides, the uncertainty-aware sigmoid over the believed per-species speed STD; v1 convention is
  **no stage boosts either side**, the coarse-signal contract — the active's live boost stays the
  incoming block's job), **T** (gen3 trapping — Shadow Tag / Arena Trap / Magnet Pull, both
  directions, ours exact and theirs revealed-exact-else-prior), **X** (entry/exit costs —
  Spikes chip × grounded, Pursuit exposure, at the (mon, GLOBAL seat) pairs), **G** (the per-mon
  end-of-turn HP ledger at the same (mon, GLOBAL) route), and the whole **C consequence set** —
  **C4** Protect (at the (Protect E3 seat, GLOBAL) pair), **C1/C1b** setup-move post-boost deltas,
  **C3** recovery-flip, **C2** status-landing consequence (all three at the (E3 seat, opp-mon)
  route, behind D1/S1), and **C5** Baton Pass (the first family on the (E3 seat, OUR-mon) route —
  the receiver axis). All maps **zero-init ⇒ families ON is bitwise-identical to OFF at init**,
  test-pinned. Measured B=1: D1+D3 = **+0.63 ms** on a ~3.5 ms forward. Note the family set grows
  *without* a version bump — the string gate catches any mismatch, and `'d'` stays a FROZEN
  `d1,d3` alias so a saved config never silently grows maps.
- **The verdict instrument SHIPPED — `agents.model.edge_ablation_audit`:** for a trained checkpoint
  it zeroes each family's map in turn (zero bias == that family absent — identity-at-init in reverse)
  and re-measures masked KL / argmax-flip rate / |ΔV| over REAL eval-trace decision states. There is
  deliberately no random-obs mode: off-distribution vectors would understate every number. It also
  carries **`concat` / `concat_cells` arms** (zero the op block at the ProjectionAssembler only —
  the exact deletion counterfactual), which is what turned the op-concat question from *deferred*
  into *answered*. Both reads are in §6.2.
- **Stage 3 groundwork SHIPPED — `agents.observation.schema`:** ONE declarative module naming every
  contiguous block of the flat observation, with a **tiling proof** (blocks are ordered, gap-free,
  and sum to `total_dim`; children tile their parent) that throws on any violation. It starts as a
  validated *view* derived from the live `get_layout()`, so it cannot drift while both exist; the
  second half (generating packer / unpacker / spaces FROM it, and re-homing the flat vector into
  entity blocks) builds on that.
- **Physics delivery today — and the concat deletion is now REFUTED, not deferred.** The op rides
  *three* routes: the flat post-pool concat, the v51 per-action pointer cells, and the edge biases.
  The plan was to delete the concat once the edges had a home. **The audit says don't:** its
  `concat` arm (zero the 807-dim block at the ProjectionAssembler only — edges, prefuse injection
  and pointer cells all left intact, i.e. the exact deletion counterfactual) reads **KL 0.482 /
  35.5% flips / |ΔV| 7.45** on gen-1 and **0.537 / 33.1% / |ΔV| 7.44** on gen-2's full stack —
  **larger than switching the entire edge system off** (0.330/26.9%/2.51 and 0.491/31.5%). The
  edges did **not** absorb the concat's role; they added on top of it. Twice measured, on two
  generations. The concat stays.
- **Feasibility, MEASURED** (`entity_spike_benchmark.py`, threads=1, idle box): the 2-layer
  production-shape trunk *with* the bias map at B=1 CPU is 0.183 ms at n=14 tokens → 0.374 ms at
  n=50 — **1.57×/2.05× growth where quadratic predicts 6.6×/12.8×** (the B=1 opponent forward is
  dispatch-bound, not tensor-size-bound). Absolute verdict: **a full ~50-seat entity trunk costs
  +0.19 ms on a ~4.6 ms forward (~4%)**. The spike held on delivery — E4 K=5 measured +0.18 ms,
  D1+D3 +0.63 ms. Honest caveat: the B=256 learner proxy grows faster (3.5× at n=50), but that
  runs on GPU. The predicted *net refund* has **not** landed yet: it was premised on Stage 2
  deleting the ~2.4 ms op flat sweep, and the head-concat deliberately still stands.
- **The measured caution.** Entity structure is **not** automatically a win here: physics
  injected into the TRUNK measured NULL 3-for-3 (ledger K9/K10) while the flat HEAD-concat
  carried the policy's largest measured dependency (P1). Stage 2's edge biases are a bet that
  those nulls were about *delivery-as-residual-injection*, not about attention-over-physics per
  se. Mitigations: the pointer head keeps a direct lossless physics→logit route regardless, and
  a per-family bias-ablation audit decides what stays.

---

---

## Part 6 — Compositionality: why a dozen pieces became one system

Between 2026-08-03 and 2026-08-04 this generation landed a pointer-native action head, two move-seat
families, tail seats, **fifteen** edge-bias families, an ablation audit, and a declarative obs schema.
Built separately, largely in isolation, they compose. That is not luck and it is not good taste — it
is a consequence of the sorting rule being an *algebra* rather than a convention. This part makes
that claim precise.

### 6.1 The sorting rule as an algebra

The rule again:

| kind of fact | home |
|---|---|
| entity-invariant (BP, type, species, item) | **token feature** |
| pair-varying (damage, status landing, speed order, trapping) | **edge** |
| probabilistic (rolls, crits, accuracy) | **distribution summary** (`[low, high, crit, pko]`) |
| future-facing (tempo, information value, plans) | **attention** (learned, never tabled) |
| *(the fifth, easy to miss)* a MODIFIER of an existing computation | **inside the kernel**, not a home of its own |

#### The intuition: it is database normalisation

If you have written a schema, you already know this rule under a different name. **Arity is the
primary key.** "Rock Slide has base power 75" is keyed by `(move)`. "Rock Slide does 0.78 maxhp to
Zapdos" is keyed by `(move, defender)`. Storing the second fact in a `(move)`-keyed table is a
**lossy denormalisation** — you are forced to pick an aggregate (a max, a mean, "vs the active
only"), and the aggregate is the information you threw away. Nobody would defend that in a database;
we defended it in a neural network for two years because the loss was invisible.

Under that reading the four homes are just the key shapes, and **attention is the join**:

```
   arity 1  →  a column on the entity's row        (token feature)
   arity 2  →  its own table, keyed by both        (edge)
   uncertain → the column is a distribution        ([low, high, crit, pko])
   not a fact at all, a judgement → no table       (attention learns it)
```

Three properties make it a contract rather than a filing cabinet.

**(1) It is a partition, and the sorting key is a property of the FACT.** The home is decided by
*arity* (how many entities does this fact range over: one, two, or the whole board?) and *certainty*
(is it a number, a distribution, or a judgement?). Both are properties of the fact itself, not of the
current model. So there is never a placement negotiation, never a "where should this go" design
meeting — you read the fact's arity off its definition. "Rock Slide has base power 75" ranges over one
entity → token. "Rock Slide does 0.78 maxhp to Zapdos" ranges over two → edge. "…78% of the time,
0.42 chance to KO" → distribution summary in the cell. "Is this the turn to spend my sleep" → nobody's
table; attention.

**(2) Closure under composition.** Facts that share a home compose by ordinary arithmetic on cells,
*before* delivery — no new plumbing, no new validation surface. **The worked example: G → C4.**

- **G** (`DamageOperator.pairwise_schedule`) is the per-mon end-of-turn HP ledger: a `[B,6,4]` cell
  `[leftovers, weather_chip, status_tick, leech]` in signed maxhp fractions, for both sides,
  delivered at the (mon, GLOBAL seat) pairs. It answers "what does this mon bleed or heal per turn
  while it stands there".
- **C4** (`pairwise_protect`) is the first C-family member: "if I click Protect this turn, what do I
  bank for free?" — a `[B,4,4]` cell `[is_protect_k, p_success, net_ours, net_theirs]` at the
  (E3 move seat *k*, GLOBAL seat) pairs, gated to the request slots that actually ARE
  Protect/Detect/Endure.
- **The implementation of C4 calls `pairwise_schedule` and sums the two actives' rows.** That is the
  whole physics. Because a successful Protect *is* the turn with the combat term deleted: their Toxic
  ramps, our Leftovers ticks, and nothing else happens. C4 cost one call, a sum, a `p_success` read
  (the existing `gen3_protect_odds_v1` obs scalar), and an `is_protect` gate.

The composition was only available because **G was written as a pure function returning a CELL, not
as a bias write.** If `pairwise_schedule` had been "add the ledger to the bias tensor", C4 would have
had to re-derive the schedule. That gives the design rule that keeps the algebra closed:

> **Kernels return cells. Families deliver cells. Never fuse the two.**

The coupling is honest in both directions: C4 inherits G's v1 approximations (Toxic charged a flat
−1/8 because the stage ramp needs an E2 counter). One fix improves both families — which is the
contract paying a second time.

```
        G  (pairwise_schedule)                     ── a PURE function returning a CELL
        ├─ leftovers   +1/16
        ├─ weather     −1/16   (type-gated)
        ├─ status      −1/8
        └─ leech       −1/8
             │                    │
             │ delivered as-is    │ REUSED as an ingredient
             ▼                    ▼
   g_map → bias at            sum the two ACTIVES' rows
   (mon, GLOBAL)                  │
                                  ▼
                          C4 (pairwise_protect)
                          [is_protect, p_success, net_ours, net_theirs]
                                  │
                          c4_map → bias at (Protect E3 seat, GLOBAL)
```

**(3) Locality of change.** Adding a family touches exactly four things: a kernel method that returns
the cell, one entry in `_EDGE_FAMILIES` declaring its width, one branch in `EdgeBias.forward` naming
the index block, and the family's letter in the gate string. **No projection width changes, no
`ModelVersion` field, no `ARCH_SIGNATURE` bump, no alignment guard.** Contrast the pre-generation
treadmill, where a new physics fact meant a new flat block → wider projections → a version field → a
migration → an offset-alignment test. The families are also *independently ablatable* by construction,
because each owns exactly one zero-initialisable map — which is what made the audit instrument
(§6.2) a 200-line script rather than a research project.

#### What breaks if you violate the rule

Take the sharpest violation: **put a pair-varying fact on a token.** Say we put "expected outgoing
damage" on the move token instead of on a (move, defender) edge. Four failures, escalating:

1. **You are forced to choose an aggregation, and the aggregation is a lie.** Damage varies over six
   defenders; a token has no defender axis. So you collapse — a max, a mean, or "vs the active only."
   We have the measurement: the op's opp-active-level collapses (believed-effect, incoming secondary)
   carried **1.2% and 0.1%** of the zero-op dependence ceiling while the per-move, per-defender form
   in the incoming matrix carried **15.4%** (ledger P1/P4). The head does not use a summary that has
   discarded the axis it must choose along.
2. **It produces a concrete, measured blindness.** The v34→v39 arc *is* this bug: the outgoing block
   priced only the current active as attacker, so on a **forced switch** (active fainted) it zeroed
   and the policy chose its switch-in **blind to offense**. That defect existed precisely because a
   two-endpoint fact had been pinned to one endpoint.
3. **It re-introduces position through the back door.** A feature "vs the mon in slot *j*" is indexed
   by slot, and slots are not identities — a different Pokémon occupies slot *j* every battle. You
   have just handed a permutation-equivariant model a positional feature to overfit.
4. **It corrupts the encoder's weight sharing.** Token features are computed by the shared
   `PokemonEncoder` over 12 slots — that sharing is where the 12× effective-data multiplication comes
   from. Making a token feature board-dependent means the "invariant" encoder now consumes board
   context per slot, and the sharing argument degrades toward twelve special cases.

The **reverse** violation — putting an entity-invariant fact on an edge — is merely wasteful, not
wrong: the value is replicated R×C times, costs compute, and dilutes the map's gradient across pairs
that all carry an identical number. Worth naming the asymmetry: *the rule is a hard constraint in one
direction and an efficiency rule in the other.*

The other two violations have their own notes. Collapsing a **probabilistic** fact to its mean is the
Jensen error — see [[marginalization_and_uncertainty]] (why the operator marginalises P(KO) over the
believed nature rather than evaluating at E[nature]). Tabling a **future-facing** fact is the
provide-vs-learn violation: hand-coding tempo or "when to spend the sleep" bakes in a prior the model
would otherwise discover, and the anti-stall / switch-bias reward terms are the standing cautionary
tale.

### 6.2 Routing vs payload — one concrete link, then the honest limits

Attention is not a bus. This is the single most load-bearing thing to understand about the edge
route, and it is easy to get backwards.

#### The intuition first

An edge bias is a **seating chart**, not a message. It tells Zapdos *"talk to that one"*; it does not
tell Zapdos *what was said*. The content is whatever the Rock Slide seat already carries — the bias
only decides how loudly Zapdos hears it relative to everyone else at the table.

Equivalently, if you prefer search engines: the bias is the **ranking function**, the value vector is
the **document**. Reranking never edits the document.

#### The seat layout you are writing biases into

```
 seat  0 ──────── 5  6 ──────── 11  12 ──────── 18   19    20 ── 23  24 ── 24+K  … +6
      ┌─────────────┬──────────────┬──────────────┬──────┬─────────┬──────────┬─────────┐
      │  OUR 6 mons │ THEIR 6 mons │  history ×7  │ GLB  │  E3 ×4  │  E4 ×K   │ E5 ×6   │
      │   (E1/E2)   │   (E1/E2)    │  (TurnDelta) │ (E7) │our moves│their topK│their tail│
      └─────────────┴──────────────┴──────────────┴──────┴─────────┴──────────┴─────────┘
       ←──────── position-stable ABSOLUTE slices ────────→  ←── appended AFTER, so every
                                                                existing slice is untouched
```

That "appended after" discipline is why v54/v57 could add seats without touching the refine
callback's tail concat, the pointer stash's E3 slice, or any family's index math.

#### The fifteen families, drawn as the blocks they occupy

Every family is a rectangle in the (from-seat × to-seat) grid. Seeing them laid out is the fastest
way to grasp that the sorting rule *tiles a space* rather than accreting features:

```
              to →   OUR 6        THEIR 6         GLOBAL   E3 (4)     E4 (K)
   from ↓          ┌────────────┬───────────────┬────────┬──────────┬──────────┐
   OUR 6 mons      │            │ D2 D4 V  T    │  X  G  │   C5ᵀ    │   D3ᵀ    │
   THEIR 6 mons    │D2ᵀD4ᵀVᵀ Tᵀ │               │  X  G  │D1ᵀS1ᵀC1ᵀ │          │
                   │            │               │        │ C2ᵀ C3ᵀ  │          │
   GLOBAL          │   X  G     │   X  G        │        │   C4ᵀ    │          │
   E3 our moves    │    C5      │ D1 S1 C1 C2 C3│   C4   │          │          │
   E4 their moves  │  D3  S3    │               │        │          │          │
                   └────────────┴───────────────┴────────┴──────────┴──────────┘

   D1 our moves→their mons      D2 our mons' best offense→their ACTIVE
   D3 their believed→our mons   D4 their BENCH threats→our mons
   S1/S3 status landing, both directions        V speed  T trapping
   X entry/exit costs           G end-of-turn ledger
   C1/C1b setup-move post-boost deltas   C2 status-landing consequence
   C3 recovery flip   C4 Protect consequence   C5 Baton Pass (receiver axis)
   (ᵀ = the transpose block; each family owns a head-set per direction)
```

Three things jump out. The **mon↔mon block is crowded** (four damage/speed/trap families share it)
because that is where the game's pairwise structure actually lives. The **GLOBAL column is the escape
hatch** for board-level facts with no natural second entity — X, G and C4 all use it, which is what
let C4 attach a *move-level* consequence to a *board-level* ledger. And the **(E3 seat, opp-mon)
route is where the C family piled in**: D1 says what a move does *now*, and C1/C2/C3 sit behind it
saying what the board looks like *after*. Same endpoints, different cell — which is the composition
contract paying off, not a coincidence.

#### The forward pass, concretely

Board: their Tyranitar is active, our Zapdos is on the bench, `--entity-topk-seats 5`.

1. **Seat construction (E4).** The move belief runs *pre*-transformer (`--move-belief-prefuse` +
   `--damage-op-prefuse`), so a composed posterior over the opponent's moveset exists before any
   attention. `DamageOperator.refine_candidates(k=K)` picks the K most-believed candidates from it —
   typed Hidden Power already scattered onto its 16 real move-nums, so HP-Ice and HP-Rock are separate
   candidates. Rock Slide is candidate *c*. Its seat is
   `threat_seat_proj([move latent(32) ⊕ w ⊕ accuracy ⊕ is_phys])` → `d_model`, appended after the
   global token. The **index is detached** (selection isn't differentiable), the **belief weight `w`
   is not** — so the policy gradient reaches the move belief through the seat's own content.
2. **Cell computation (D3).** `pairwise_incoming` runs the pre-collapse `_incoming_rolls` physics at
   **exactly the same detached candidate selection the seats stashed** (`EntityMoveSeats.last_cand`).
   This single-sourcing is a correctness requirement, not tidiness: if the two selections could
   disagree, you would get a bias *about Rock Slide* landing on the row of a seat that *encodes
   Earthquake*, and nothing would ever fail loudly. That is the GIGO class this project treats as
   drop-everything.
   The cell at (c = Rock Slide, i = Zapdos) is `[high, pko, eff, is_phys, w]` — say
   `[0.78, 0.42, 2.0, 1.0, 0.83]` (Zapdos is Flying; Rock is 2×).
3. **Map to a bias.** `d3_map` is a **zero-init** `Linear(5 → 2·n_heads)` — one head-set for the
   (seat → mon) direction, one for (mon → seat), because "how much should the threat token listen to
   the defender" and the reverse are different questions. After training, suppose head 3 emits +1.4 in
   the (mon → seat) direction for this cell.
4. **What actually flows along the link.** *Not the cell.* The bias enters only the **pre-softmax
   attention logit**: `logit(Zapdos, RockSlide-seat) = q_Zapdos · k_RockSlide + 1.4`. What flows is
   `softmax(...) · v_RockSlide-seat` — the **value vector of the Rock Slide seat**, i.e. the projected
   move latent ⊕ belief ⊕ accuracy ⊕ is_phys. The bias raises the *weight* on that value. **The number
   0.78 is never transported anywhere.**
5. **What Zapdos's refined token encodes afterward.** Its own encoded identity plus a weighted mixture
   of what it attended to, with the Rock Slide seat over-weighted relative to what content similarity
   alone would have produced. In words: *"a Zapdos, on a board where a physical, ~85%-accurate,
   strongly-believed Rock-type move is the salient thing pointed at me."* A **contextualised identity**,
   not a number.
6. **How the pointer head reads it.** The switch logit for Zapdos is
   `switch_score(ReLU(switch_proj([our_team_out[Zapdos] ⊕ Zapdos's op switch cells])) + ctx_proj(latent_pi))`.
   `our_team_out[Zapdos]` is the refined token from step 5, so the edge's contribution arrives as a
   **shift in that token**. The literal magnitude — 0.78 maxhp, the CB-conditional tail, the OAX
   attacker row — arrives separately, through the **switch cells**, which are a lossless slice of the
   same op tensor the projection heads consume.

So: **the edge decided what Zapdos's token is about; the cells decided what the numbers are.**

```
   D3 CELL  [0.78, 0.42, 2.0, 1.0, 0.83]
        │
        │  d3_map: Linear(5 → 2·4 heads), ZERO-init
        ▼
      +1.4 ─────────────► added to ONE attention LOGIT, pre-softmax
                                    │
        q_Zapdos · k_RockSlide + 1.4 ──softmax──► weight w
                                                     │
                                        w · v_RockSlide-seat   ◄── THIS is what flows:
                                                     │             the SEAT's value vector
                                                     ▼             (latent ⊕ belief ⊕ acc ⊕ is_phys)
                                        refined Zapdos token
                                                     │
                              ┌──────────────────────┴──────────────────────┐
                              ▼                                             ▼
                   pointer switch logit j                    pooled into our_pool / value_pooled
                              ▲
                              │
                   op SWITCH CELLS ── the number 0.78 arrives HERE, never via the edge
```

#### What a softmax weight cannot carry

- **Levels.** A softmax row is shift-invariant: add the same constant to every entry of a row and
  nothing changes. An edge bias can therefore express only **contrast within a row**. "Everything on
  this board threatens me severely" is *invisible* to an edge; "this one most" is exactly what it
  encodes. A family whose cell is near-constant across a row's columns produces no effect at all,
  however large it is.
- **Magnitude past the routing decision.** Once a row has saturated onto one column, more bias buys
  nothing and the gradient into the map vanishes. It is a soft argmax, not a scalar channel.
- **Valence.** The bias says *who to listen to*; whether that's good news or bad must live in the
  attended token's value vector.
- **Anything on a masked pair.** A fainted or unrevealed endpoint's row is masked out, so a fact whose
  only home is that pair silently disappears. (This is why D1 reads zero gradient on random
  observations — its gates see no revealed opponent. Correct behaviour, alarming-looking test output.)

#### The three routes, and which gap each closes

| what the decision needs | op head-concat | pointer cells | edge biases |
|---|---|---|---|
| exact magnitude at the right logit | pooled + position-flat; head must learn the alignment | ✅ exact, per-action, lossless | ✗ — routing only |
| relevance ("which of these matters now") | ✗ | ✗ (each action scored in isolation) | ✅ — this is the whole job |
| physics visible to the **critic** | ✅ **the only direct route** (the pointer head is policy-only) | ✗ | indirect, via refined tokens → CLS pools |
| cross-entity composition *before* pooling | ✗ (post-pool by construction) | ✗ | ✅ |

That third row is the one people miss, and it predicts something the audit then found.

#### The audit reads: two generations, and the story changed with training

`edge_ablation_audit`, per-family map-zeroing on real eval-trace states. Three reads now exist, and
the mid-training one was misleading in an instructive way.

**Gen-1 @9.6M (2048 states, PRELIMINARY) → gen-1 @40M (4000 states, COMPLETE):**

| family | @9.6M KL | @40M KL | @40M flips | note |
|---|---|---|---|---|
| **d1** our moves → their mons | 0.059 | **0.145** | 13.6% | |
| **d2** our mons' offense → their active | 0.057 | **0.187** | 19.1% | \|ΔV\| 1.66 — largest critic edge dependence |
| s1 our status → their mons | 0.002 | 0.0061 | | |
| v speed, mon↔mon | 0.002 | 0.0096 | 5.0% | \|ΔV\| 0.53 — the value head reads speed |
| d3 their believed moves → our mons | 0.0009 | 0.0021 | | still near-decorative |
| s3 their believed status → our mons | 0.00007 | 0.0005 | | still near-decorative |
| **all off** | 0.124 | **0.330** | **26.9%** | \|ΔV\| 2.51 |

**The headline correction: the edges became LOAD-BEARING with training** — every family grew 2–3×,
and the whole system went from 16.6% to 26.9% argmax flips. Reading the 9.6M snapshot as a verdict
would have been wrong; it was a *snapshot of an untrained map*.

**Gen-2 @40M (all eleven launch families trained, 4000 states):** d2 dominant (KL **0.312** / 23.8%
flips / \|ΔV\| 2.18), then d1 (0.108/10.1%), v (0.012/6.3%), d4 (0.004/2.1%), s1 (0.0096/1.9%), t
(0.0018/1.8%). **Near-decorative at 40M: d3, s3, x, g and c4** (c4 ≈ 0.00002 — the Protect edge
never got used). All-off = **0.491 / 31.5%**.

Same outgoing-dominant shape as ledger P1, now on a third independent architecture. And d2 — a
*bench-offense* family — carrying the largest \|ΔV\| is exactly what the route table predicts: the
critic has no pointer head, so an edge is one of only two ways board physics reaches it.

**Owner decision (2026-08-06): KEEP ALL FAMILIES.** The decorative ones encode strategy-critical
mechanics a 40M self-play run may simply not have discovered — Protect×Toxic timing, Explosion
consequence play, entry/exit costs — and they are the substrate exploiters and longer runs need.
"Unused at 40M" is not "useless"; cut A/Bs can come later.

#### What happened to the three explanations

The mid-training asymmetry got three candidate stories. Training partly adjudicated them:

**E-c (belief immaturity) — PARTLY CONFIRMED, and it explains the wrong thing.** Every family grew
2–3× from 9.6M to 40M, so "the maps were still near zero" was real. But the *incoming* families grew
too and stayed near-decorative in absolute terms (d3 0.0009 → 0.0021). So immaturity explains the
**level** at 9.6M, not the **asymmetry**.

**E-a (redundancy) — now the leading story, and the concat arm supports it.** Incoming physics has a
lossless per-candidate route into exactly the logit it should drive (the pointer switch cells), and
the concat arm shows the flat block is *still* carrying more than the entire edge system. The edges
did not absorb it; they added on top. Marginal ablation of one redundant route under-reads by
construction. *Test still open:* joint ablation — zero `d3_map` AND the op's incoming concat rows AND
the pointer switch cells, and see whether the joint effect vastly exceeds the sum of marginals.

**E-b (dilution) — untested.** Stratify the audit by decision type (switch legal and in the top-2;
high-entropy; forced-switch). Still cheap, still worth doing.

**A GIGO caveat on the `v` rows, found 2026-08-06 (v58).** `pairwise_speed` and `pairwise_boost`'s
outspeed read **stat index 4 — Special Defense — as "speed"** (bare-integer indexing across two stat
layouts). Both trained generations' V edge therefore priced *bulk* as speed. So the v rows (5.0% and
6.3% flips) mean "**the v route carries signal**", not "the speed physics is validated". Fixed with
named indices and a discriminating regression test; gen-3 trains on true physics. A good reminder
that an ablation measures *the channel*, not *the correctness of what's in it*.

**The standing caveat:** masked KL measures **USE, not VALUE** ("learns ≠ helps"). A heavily-used
family could be leading the policy astray; a decorative one could be insurance that only pays in rare
states — which is E5's whole rationale, and the reasoning behind keeping the decorative families.

### 6.3 Why equivariance shrinks the hypothesis space

Part 1 gave the abstract argument (invariance shrinks the function class ≈ by the orbit size). Here is
what it buys in this codebase, in parameters.

**One scorer vs eleven positional rows.** The deleted flat head was `Linear(512 → 11)` = 5,632
parameters arranged as **eleven independent rows**. Row 7 ("switch to slot 1") receives gradient only
when that action's advantage is nonzero, and has to independently rediscover what slot 1 means —
eleven separate learning problems sharing only an input. The pointer head applies **one** switch scorer
to all six team tokens: every switch decision anywhere trains the same function (~6× the effective
samples per parameter), and the function is indexed by **content** — that mon's token ⊕ its cells —
never by slot.

**One bias map vs a per-pair table.** Family V spans 36 mon-pairs. The unshared alternative is a table
of 36 × 2 directions × `n_heads` learned scalars, each seeing only its own pair's gradient *and*
indexed by slot — i.e. learning noise, because slot *j* holds a different Pokémon in every battle. The
shared `Linear(cell → 2·n_heads)` is a function of the **cell**, so "a 2× physical hit with 40% KO
odds is worth attending to" is learned once and applies to every pair, every battle, every team. The
model **literally cannot express** "pair (2,5) is special", which is why it cannot overfit to it.

**Bug classes that became unrepresentable** (not "guarded against" — *inexpressible*):

- **The ordering class.** The extractor reads moves sorted-by-id; the action space is request-ordered.
  The permutation now happens **once, by move-num identity** (`_request_order_move_tokens`), and every
  downstream consumer — the E3 seats, the pointer head's move logits, D1's rows, S1's rows, C4's
  Protect gate — reads that one ordering. `agents/action/ordering_integrity.py` existed solely to
  police a class that no longer has a representation.
- **F2.** Switch logits used to be read from a permutation-*invariant* CLS pool. The pooling had
  **destroyed** the per-slot information, so a bench mon's own token could never reach its own logit
  — no amount of training could fix it. The pointer head reads `our_team_out[j]` directly.
- **Bias misplacement.** Every family writes at contiguous, documented index ranges, pinned by a
  placement-only-at-documented-pairs test. "The physics landed on the wrong entity" is a test failure,
  not a silent wrong answer.

The general shape: **a class of alignment defects became type errors.**

**What we paid.** Four honest costs:

1. **Genuine positional conventions are now inexpressible.** If a slot really did carry meaning, a
   flat head learns it for free; we must promote it to *content* on the token. That is the discipline
   — but it is a restriction, and when a real asymmetry exists and we forget to encode it, the model
   cannot represent it at all.
2. **The head is SEPARABLE given the context.** Each action is scored independently from its own
   entity plus a shared `latent_pi`. So relations *between candidate actions* — "switch to Zapdos only
   because Skarmory is also alive and would rather come in later" — cannot be formed at the head. They
   must be resolved upstream in the trunk and arrive inside `latent_pi` or inside the token. A flat
   head over a rich latent forms arbitrary cross-logit correlations in its hidden layer for free. We
   traded head expressiveness for correctness and sample efficiency, and pushed the composition burden
   onto attention.
3. **Cold start forfeits the positional prior.** Zero-init scorers ⇒ uniform-over-legal. A flat head's
   random rows are a (bad, but nonzero) prior that sometimes speeds early learning; "action index 6 is
   often good" is a real shortcut we gave up. Correct, but not free.
4. **The standing cost from Part 1**: an equivariant model is strictly less expressive. If the true
   function is not equivariant, we have excluded it.

### 6.4 The hypothetical-world trick — how the C family got built in days

The C family (consequence deltas) looked like the hardest piece — "price what happens if I set up"
sounds like search. It wasn't, and the evidence is that **the entire family shipped in one sweep**
(C1, C1b, C2, C3, C5 landing alongside the earlier C4) rather than as the multi-month design lift it
was budgeted as. The reason is a property of the `DamageOperator` worth defending on purpose.

**The damage kernel is a pure function.** `_rolls` / `_damage_rolls` map
(attacker stats + boosts, defender stats + boosts, move attributes, field) → rolls. No hidden state,
no mutation, no learned parameters, no ordering dependence. Therefore:

> "What if I Swords Dance first?" is not a simulation. It is the same call with
> `atk_boost += 2`.

One input changed, one extra kernel call, the same fuzz-validated physics (the constructed-probe
oracle covers it, 22/22), and still differentiable — so the gradient continues to reach the belief.
Compare what a simulator would need: mutate a battle state, advance a turn, resolve both sides'
actions, re-encode an observation, re-run the network. C1 needs none of it, because the **only** thing
the hypothetical changes is an *input to a pure function*. **Purity is what makes counterfactuals
cheap.**

That is not one trick, it is the whole family:

| | the hypothetical | the perturbed input | |
|---|---|---|---|
| **C1/C1b** stat move | "if I Swords Dance" (and the incoming halves: Iron Defense / Amnesia / Curse) | attacker/defender boost stage | ✅ |
| **C2** status | "if this T-Wave / WoW / Toxic *lands*" | speed ×0.25 / phys Atk ×0.5 / the HP schedule | ✅ |
| **C3** recovery | "if they Recover" (and Rest's 2 lost turns) | defender current HP, sleep counter | ✅ |
| **C4** Protect | "if I Protect" | *no re-run at all* — G's ledger with the combat term deleted | ✅ |
| **C5** Baton Pass | "if I pass these boosts" | recipient's stats with the passed boost stages | ✅ |

Five names, one idea — which is exactly why they arrived together rather than one per quarter.

**Why the cell must be a DELTA.** A C1 cell could carry post-boost damage; it should carry
`post − pre`. Two reasons, both already established elsewhere in the operator:

1. **Decorrelation.** D1 already delivers the pre-boost cell at exactly the same (move, defender)
   pair. An absolute post-boost cell would be nearly collinear with it, and the head would spend
   capacity learning a subtraction to recover the quantity that actually matters. This is the same
   reasoning that put the v28 Choice-Band tail *beside* the modal line instead of blending it in, and
   that split the crit term off as a delta rather than folding it into the mean.
2. **The decision is a threshold crossing on the delta.** "Does Swords Dance flip Earthquake vs
   Swampert from a 3HKO to a 2HKO?" is a nonlinear function of the *change*, not of either level. The
   operator should do the nonlinear part so the ReLU head can stay additive — the same principle that
   makes `pko = accuracy · P(KO | hit)` a computed product rather than two factors handed over
   separately.

**Where it hits its ceiling.** C edges are a **depth-1, breadth-1, opponent-static** differentiable
lookahead. Be precise about each limit:

- **The opponent is frozen.** C1 prices what Swords Dance does to *my* damage table. It does not price
  them switching to a resist, or KO'ing me during the setup turn. The setup **risk** is deliberately
  assigned to attention — it is a future-facing fact, and the sorting rule says so.
- **No sequencing.** Every C edge perturbs the *current* board. "SD, then EQ, then they Roar it away"
  is composition over time; that needs a tree.
- **No maximisation over replies.** Search's value comes largely from taking a max over the
  opponent's best response. A C edge takes no max over anything.
- **The state never advances.** No damage applied, no faint, no switch, no new board.

So the honest framing, and the division of labour with the search line: **C edges amortise the cheap,
exactly-computable part of one-ply lookahead into the forward pass at the cost of roughly one kernel
call. What they cannot do is precisely what search is for.** The `better-line` CRN-anchored beam and
the search-as-teacher / OPD work run offline and expensively and produce *targets*; the C edges make
the arithmetic free so search does not have to spend depth re-deriving it. See
[[on_policy_self_distillation]] and [[pbs_value_functions_and_search]].

One consequence worth stating as a rule: **any convenience that gives the damage kernel hidden state
or in-place mutation kills the entire C family.** The purity is load-bearing, not stylistic.

### 6.5 How all of it reaches the two heads — the funnel

Everything above happens in a 35-seat trunk. The heads are two 512-dim vectors. Between them is a
**severe bottleneck**, and understanding its exact shape explains most of the measured results in
this note.

#### The intuition: a mixing desk

Think of the 35 seats as 35 musicians. The heads cannot hear the musicians — they hear a **mix**.

- The **policy** gets three microphones over the room: `our_pool`, `their_pool`, and a close mic on
  our active mon. Plus — since v51 — **individual channel strips** for the 6 bench mons and the 4
  moves, wired straight to the pointer head, bypassing the mix entirely.
- The **critic** gets **one** microphone (`value_pooled`, a single learned query over all 12 mon
  tokens). No channel strips. The pointer head is policy-only.
- The **op block** is a DI box: 807 dims soldered directly to the master bus on *both* heads,
  bypassing the room mics altogether.

That picture is the whole architecture's information economy, and it predicts the audit.

#### The literal plumbing

```
                    35 refined seats  (our 6 · their 6 · history 7 · GLOBAL · E3 · E4 · E5)
                                   │
          ┌────────────────────────┼─────────────────────────────┐
          │  POOLED (lossy)        │                             │  UNPOOLED (lossless)
          ▼                        ▼                             ▼
   our_cls   → our_pool   128   value_cls → value_pooled 128    our_team_out[j]  (per bench mon)
   their_cls → their_pool 128                                    E3 refined seats (per move)
   our_active_refined     128                                              │
          │                        │                                       │
          ▼                        ▼                                       │
  pi_combined =                vf_combined =                               │
   [our_pool 128                 [value_pooled 128                         │
    their_pool 128                our_ctx_enc  32                          │
    our_active   128              opp_ctx_enc  32                          │
    our_ctx_enc   32              non_matchup_rest                         │
    opp_ctx_enc   32              (our_active if --value-active-readout)    │
    non_matchup_rest              (hidden_opp_belief)                      │
    (hidden_opp_belief)           ══ damage_block 807 ══]  ◄── DI box      │
    ══ damage_block 807 ══]  ◄── DI box                                    │
          │                        │                                       │
   pre_proj_norm (LN)        value_pre_norm (LN)                           │
   projection  → 512         value_projection → 512                        │
   FiLM: h·(1+Δγ_pi(z))+Δβ   FiLM: h·(1+Δγ_vf(z))+Δβ    ◄── z_arch         │
   ReLU                      ReLU                                          │
          │                        │                                       │
   mlp_extractor              mlp_extractor                                │
   .forward_actor             .forward_critic                              │
          │                        │                                       │
      latent_pi                latent_vf                                   │
          │                        │                                       │
          │                   value_net  (or ValueDistHead → E[Z])         │
          │                        │                                       │
          │                   PopArt denorm → V(s)                         │
          │                                                                │
          └──────────► POINTER HEAD ◄────────────────────────────────────── ┘
                       ctx = latent_pi
                       + op move cells  [low,high,crit,pko,p_land,known,sec×10]
                       + op switch cells [incoming row ⊕ CB tail ⊕ OAX row]
                                │
                          11 action logits
```

#### Five things this diagram tells you

1. **The edges never add a dimension anywhere.** They change *what the pools contain* and *what
   `our_team_out[j]` encodes*. Every projection width is identical with families on or off — which is
   why growing the family set is not a version bump.
2. **The op block is the only route that skips pooling for BOTH heads.** That is the mechanical
   explanation of ledger P1: zeroing it costs masked KL 0.9385 because it is the one un-compressed
   channel the critic has at all.
3. **v51 made the POLICY entity-native; the critic was not touched.** The pointer head reads
   per-token, pre-pool state. The critic still reads one 128-dim summary. So the policy has two
   lossless routes (pointer cells + per-token reads) and the critic has one (the concat).
4. **That asymmetry predicts the audit's `|ΔV|` results — and the concat arm confirmed it hard.**
   D2, a bench-offense family, carries the largest edge-side critic dependence for exactly this
   reason. And when the concat arm was finally run, **the critic was hit ~3× harder by removing the
   concat (\|ΔV\| 7.45) than by removing every edge family combined (2.51)**. That is this diagram,
   measured: the policy has alternatives and the critic does not.
5. **FiLM sits at the very end**, after both projections, before both ReLUs — downstream of every
   phase including the op concat, so it composes with every toggle and modulates the *finished* head
   features rather than the contested trunk.

#### The honest counterweight on the critic's narrow readout

It is tempting to read point 3 as "the critic is starved — widen it." **That specific lever is
already dead** (ledger P3). A representation probe measured `value_cls` effective rank at 3–4 versus
the policy's 30–40 on the same body — but also measured `value_pooled` predicting the episode outcome
at **AUC 0.833** against the policy's 384 dims at **0.835**. A critic emitting one scalar does not
need thirty dimensions; rank ~3 is *appropriate*, not pathological, and v45's 51-bin distributional
target correctly did not move it. "Widen or delete the value pool" is refuted.

What is *not* refuted, and is a genuinely different claim: the critic has no **per-candidate** readout.
`value_pooled` answers "who is winning this board." It cannot answer "what is this board worth *if I
send in Zapdos*" from any entity's own token. That is the pointer-head question asked on the value
side, and it remains open — flagged here as an open question with a real prior against the naive
version, not as a plan.

### 6.6 What search looks like from here

The entity graph is not a search engine, and §6.4 was explicit that C edges structurally cannot do
what search does. So what *would* search look like, and what does the entity work change about it?

#### The intuition: calculator vs sparring

A C edge answers *"how hard would I hit if I were stronger?"* — a calculator question, answerable
without anything happening. Search answers *"what happens if we actually fight?"* — which requires
the world to advance, the opponent to respond, and someone to take a maximum. The entity work makes
the calculator free. It does not make the sparring free; it makes it **cheaper to aim**.

#### The shape we already have

`better-line` (shipped, in the prober) is a **shallow CRN-anchored beam over the critic**:

```
                          state at turn T
                                │
        ┌───────────────┬───────┴────────┬────────────────┐   ← branch on OUR action
       SD              EQ              switch Zapdos    switch Skarm
        │               │                 │                 │
   ═══ the CHEAP layer: pure-kernel cells, no clone, no sim ═══
    C1 delta        D1 cell          D2 / X cell       D2 / X cell
        │               │                 │                 │
        └────── prune to the beam width using those ─────────┘
                        │
                 CLONE the sim state   ◄── the EXPENSIVE op (serializeBattle,
                        │                   `utils/bridge/search_session.py`)
                 opponent replies      ◄── RECORDED at the divergence ply;
                        │                  the reloaded policy at interior plies
                 advance one ply, re-encode obs, re-run the model
                        │
                 read V(s′)  ──►  ΔV / ΔP(win) per ply
                        │
                 (optional) --confirm-rollouts N  → play to a real win/loss
```

Three properties of that design are worth internalising because they are what make search *sound*
rather than merely expensive:

- **CRN (common random numbers).** Branches share a dice stream, so a difference between two lines is
  attributable to the *decision*, not to variance. Without it you need orders more rollouts to see
  anything.
- **The obs-purity invariant is what makes cloning possible at all.** Part 4 ruled out recurrence
  because observations must remain a pure function of the replayed event log. That decision — taken
  for debuggability — is precisely what lets you materialise a cloned mid-battle state and rebuild a
  valid observation for it. **A recurrent model would have no legal search at all.** This is the
  clearest case in the codebase of a debuggability invariant paying a capability dividend.
- **The critic is the leaf evaluator**, so search quality is bounded by critic quality. Deepening a
  tree over a mis-calibrated critic amplifies its errors rather than fixing them.

#### What the entity generation changes about search

1. **Candidate generation becomes structured and free.** The pointer head already emits a score *per
   entity*. Branching is then "take the top-k pointer logits" (or Gumbel-top-k for an unbiased
   sample), and because the head is **equivariant**, that prior transfers across permutations — the
   same branching quality on a team you have never seen in that slot order.
2. **The C family becomes a pruning layer.** Cloning is the expensive operation; a pure-kernel delta
   is ~one kernel call. So the C edges let you rank candidates *before* paying for a single clone.
   This is the concrete sense in which "C edges make the cheap part of search free": they raise the
   quality of the actions you spend clones on.
3. **The cost regime flips.** A B=1 forward is dispatch-bound (~0.44 µs per aten call, which is why
   the v49 candidate-axis cap bought +0.3% at B=1 and +63.5% at B=256). Search *batches nodes*, so it
   runs in the regime where tensor-size optimisations actually pay. Levers that look pointless for the
   rollout opponent become the right levers for a search.
4. **The search's output is a distribution, and distributions are what OPD wants.** A beam that
   returns an improved *policy* (not just an argmax) is a teacher target; see
   [[on_policy_self_distillation]] for the KL(π′‖π) upgrade over the current AWR-toward-`A*`.

#### The limit that matters most, stated plainly

**"Selecting the best possible path" is the wrong objective in this game**, and it is worth being
precise about why. Pokémon is **simultaneous-move and imperfect-information**. Two consequences:

- There is no "best path" against an opponent who is also choosing. A max over our actions against a
  *fixed* opponent policy produces a line that is **exploitable** — you have solved a different game.
  The correct object is an equilibrium strategy at the node (a *distribution*), not a sequence.
- The state is not a sufficient statistic; the belief is part of it. Sound depth-limited search in
  this class needs a public-belief-state formulation and a value function consistent with it.

Both are the subject of [[pbs_value_functions_and_search]] — including why our beam's "opponent
recorded at the divergence ply" is a deliberate, honest approximation (it answers *"could this have
gone better against what actually happened"*, a counterfactual-attribution question) rather than a
claim about equilibrium play. Keeping those two questions separate is the discipline.

### 6.7 Entity structure vs FiLM/LoRA — two orthogonal factorisations

These get conflated because both are answers to "one shared function is averaging conflicting
things." They act on different axes.

> **Entity structure factorises the INPUT by a symmetry.**
> **Modulation (FiLM / LoRA) factorises the PARAMETERS by a context.**

#### The intuition: the orchestra and the tone knob

Entity design is the **wiring** — which instrument feeds which channel, and the fact that one channel
strip works for any violinist. FiLM is the **tone knob on the shared amplifier** — the same amp,
adjusted for the room. You would not fix a miswired stage by turning the treble up, and you would not
fix a room's acoustics by re-soldering the patch bay.

#### The one design question that decides which you need

For any axis along which behaviour differs, ask: **is this a genuine symmetry, or a genuine mode?**

- *"A Pokémon is a Pokémon"* — the encoder should be the same function for slot 1 and slot 5. Genuine
  symmetry → **share weights** (equivariance). Sharing here multiplies data 12×.
- *"A stall team and a hyper-offense team want different policies from the same board"* — there is no
  symmetry mapping one to the other. Genuine mode → **un-share along that axis** (conditioning).
  Sharing here makes per-team gradients cancel — the measured amortization gap.

Equivariance and conditioning are therefore not rivals; they are the two halves of one decision.
**Share where a symmetry is real; condition where it is false.** Every architecture choice in this
project is one of those two moves.

#### The unification worth carrying: they are the same mechanism at different clock speeds

An edge bias is `logit += Linear(cell)`. FiLM is `h = h·(1 + Linear(z)) + Linear(z)`. Both are **a
small network generating a modulation for a bigger computation** — hypernetwork-shaped. What differs
is the *source* of the modulator and how fast it changes:

| mechanism | modulator comes from | what it modulates | changes every… | learned? |
|---|---|---|---|---|
| **edge bias** | a computed physics cell (a *known function* of the board) | attention logits, per pair per head | forward, per pair | the map is; the cell is not |
| **pointer cells** | a slice of the op tensor | the entity's own score input | forward, per action | no — lossless passthrough |
| **FiLM(z_arch)** | a learned DeepSets latent over our team | head features, post-projection pre-ReLU | battle (team-static) | fully |
| **LoRA (hypothetical)** | any context code | a shared function's *weights*, low-rank | whatever you key it on | fully |

Reading the table top to bottom is reading a **clock-speed axis**: physics changes every turn, team
archetype changes every battle. Fast, certain, computable facts go through the fast, computed
channels; slow, uncertain, learned modes go through the slow, learned channel. Putting a fast fact on
a slow channel means it is stale; putting a slow mode on a fast channel means it has to be
re-inferred every forward from evidence scattered across tokens.

#### Where LoRA would attach in an entity world — and the standing prior against it

There is a real synergy worth naming: **entity design manufactures heavily-shared functions, and
heavily-shared functions are exactly what low-rank conditioning attaches to.** In this architecture
the natural sites, in order of appeal:

1. **The pointer scorers.** One switch scorer runs over 6 mons; one move scorer over 4 moves. A rank-1
   modulation here means *"how does THIS archetype weigh a switch"* — conditioning a genuinely shared
   decision function rather than a positional row.
2. **The edge-bias maps.** Each is a single `Linear(cell → 2·n_heads)`. A per-archetype rank-1 delta
   would say *"a stall team cares about the G ledger; a hyper-offense team cares about D1"* — cheap,
   interpretable, and precisely aligned with what archetypes actually differ on.
3. **The `PokemonEncoder` atom MLP.** Probably the *wrong* site: that is the genuine symmetry, and
   conditioning it trades away the data multiplication that makes it work.

**But do not build any of that on the old theory.** The measured result stands: a **free, unconstrained
per-team LUT did not close the N=20 exploiter gap** (+0.024, CI [−0.016, +0.064] — a decisive positive
is ruled out), and the z-clustered arm was a *second independent null*. Conditioning-signal theory is
unsupported on two orthogonal manipulations, and the follow-up 2×2 found **count dominates
conditioning** (N 20→10 worth +0.077 significant; conditioning +0.027 n.s.). See
[[amortization_gap_and_conditioning]], [[conditioning_architectures]], and the LUT/count memories.

The one input the entity generation genuinely changes: **every one of those nulls was measured on a
flat positional head.** A modulation applied to an *equivariant per-entity scorer* is conditioning a
different surface — "how to weigh threats" instead of "what slot 3 means." That is a reason to keep
the question open, not a reason to reopen the ladder. Per the standing rule, any revival needs a cheap
pre-build probe and an argument for why it is not a fourth instance of the same null.

### 6.8 Quiz — could you design the next family?

A complete answer names **four things**: (a) the **home** for each fact (token / edge / distribution
summary / attention / kernel-internal modifier), (b) both **endpoints** if it is an edge, (c) the
**cell** contents, (d) the **gate** (what it requires to be buildable) and the one **GIGO risk** you
would guard.

**Q1 — the Wish-passing edge.** Gen3 Wish heals the *slot occupant's* maxhp/2 at the end of the turn
after cast, slot-keyed (it survives switch, faint, and Roar-phazing). Today it is two flat reactive
scalars (`wish_floating`, `gen3_wish_wired_v1`). Design the entity-native replacement, such that the
policy can price *"Wish now, then pivot Zapdos in to catch it."* Sort each of these: **the wish is
pending**; **the heal fraction**; **which bench mon benefits most**; **whether the pivot turn is
safe**. For whatever becomes an edge, name both endpoints, the cell, the gate, and the GIGO risk.

**Q2 — the phaze-exposure edge.** Roar / Whirlwind drags in a **random** non-active teammate; the
entity-graph inventory parks it under X. Design it. The trap is in the endpoints — the drag target is
a random member of a *set*, not a specific entity. Give endpoints, cell, and gate; then explain why
the naive placement *"put expected phaze chip on the Roar move token"* violates the sorting rule, and
name the measurable cost it would produce.

**Q3 — sorting discrimination.** Give the home of each: (i) Explosion halves the defender's Defense;
(ii) our side has 2 Spikes layers up; (iii) this opponent has clicked Protect two turns running;
(iv) trading our Starmie for their Tyranitar is a good trade in this matchup. Then: **exactly one of
the four is a MODIFIER** — it belongs inside an existing kernel rather than in any of the four homes.
Which, and what is the general test that identifies that category?

<details>
<summary><b>Answer sketches</b> (try first — the value is in the sorting, not the trivia)</summary>

**Q1.** *Pending* is a condition of a thing → an **E2 column** on the mon holding the slot (and a side
summary on E6); it is entity-invariant given the board. *The heal fraction* is a **constant ≈0.5** by
gen3's own rule (the recipient's own maxhp/2) — this is why the current scalar is GIGO-proof, and it
means it does **not** need an edge at all. *Which bench mon benefits most* is the genuinely
pair-varying fact and the reason to build anything: the beneficiary depends on the (wish, candidate
recipient) pair via the candidate's *current* HP deficit and its incoming threat table. So: an edge at
**(the Wish-holding mon's slot / the side seat, each of our 6 mon seats)**, cell something like
`[resolves_this_turn, hp_deficit_capped_at_half, pko_flips]` — where `pko_flips` is the C3-shaped
recomputation of which incoming P(KO) cells fall to ~0 after the heal, i.e. it reuses the pure-kernel
trick. *Whether the pivot turn is safe* is future-facing → **attention**, never tabled. **Gate:**
requires `damage_op` for the pko recomputation, and the Wish reconstruction from the event log (poke-env
tracks none of it). **GIGO risk:** the slot-keying — a naive implementation keys the heal to the *mon*
rather than the *slot*, which silently misprices exactly the pivot case the feature exists for; guard
with the existing wish fuzz that checks every real resolve was flagged pending the turn before.

**Q2.** The drag target is a uniform draw over the eligible bench, so the edge's other endpoint cannot
be a specific mon. Two defensible shapes: (a) an edge at **(the phazing move seat, the GLOBAL/side
seat)** carrying an *expectation over the set* — cell `[p_phaze_lands, E[entry chip over eligible
bench], boosts_reset_value, n_eligible]` (the X-family route, which G and C4 already use for
board-level facts); or (b) an edge at **(the phaze move seat, each of our 6 mon seats)** carrying that
mon's *own* `[eligible, entry_chip_if_dragged, 1/n_eligible]` — strictly more informative, and it
keeps the per-mon axis instead of collapsing it, at 6× the cells. Prefer (b) by the axis rule; (a) is
the cheap first cut. **Gate:** `damage_op` (entry chip needs the hazard/grounded fold X already
computes) + the phaze move being in the belief for the incoming direction. **Why the naive placement
fails:** "expected phaze chip" put on the Roar *move token* collapses the defender axis — the same
error as pricing outgoing damage vs the active only, whose measured cost was the forced-switch
offense blindness that v39 had to fix. The policy would know "Roar chips someone" without knowing
*whom*, which is precisely the choice it is being asked to make.

**Q3.** (i) is the **modifier** — it belongs inside `_rolls` as a defense multiplier, exactly like
STAB, burn, screens, and the type chart, none of which are separate edges. (ii) is a **side/E6 token
feature** that feeds the X family's `entry_chip` cell — the hazard *count* is entity-invariant; the
*chip it causes to a particular mon* is the pair-varying part and already lives on an edge. (iii) is a
**condition of a thing** → an E2 column (the protect/endure consecutive counter), which is exactly
where C4's `p_success` is sourced from. (iv) is **future-facing** → attention; it is a judgement about
plans and win conditions, and tabling it would be the provide-vs-learn violation.
**The general test for a modifier:** it is a *multiplier or override inside an existing computation*
rather than a fact about entities. If removing it changes a number the kernel already produces, but
adding it introduces no new (entity) or (entity, entity) fact, it is kernel-internal. The tell is
that it has no natural cell of its own.

</details>

---

## Synthesis

The through-line from a plain deep net to this architecture is one question: **where does
information live, and what has to be re-learned per position?** A flat MLP answers "everywhere,
and everything." Entity-based design answers with a discipline:

- per-thing facts → **tokens**
- pairwise facts → **edges** (recomputed activations, never weights)
- decisions → attached to **the entity they select**
- known mechanics → computed by a **differentiable operator**, injected at the level that
  describes them
- history → **compiled into entity state**, or its own token type with recency as content
- attention's job is to **route and compose**, never to be a calculator

The day-to-day payoff is not primarily accuracy — it is that a new physics fact becomes *a new
edge feature* instead of a new flat block + wider projections + a version bump + an alignment
guard. Under this discipline the accretion treadmill dies.

And the deeper payoff, the one Part 6 is about: the discipline is **compositional**. Because the
sorting key is a property of the fact (its arity and its certainty) rather than a property of the
model, every new fact has exactly one home and nothing else has to move. Because kernels return
*cells* and families merely *deliver* them, a family is reusable as another family's ingredient —
which is how the per-mon HP ledger became the Protect-consequence edge for the cost of one call and
a sum. Because the physics kernel is a **pure function**, every "what if" is a re-run with one input
changed rather than a simulator. Fifteen edge families, a pointer head, three seat families, an
ablation audit and a schema were built largely independently and fit together — not because they
were coordinated, but because each one was answering the same question the same way. That is what
makes it a system rather than a pile: **the pieces share a contract, not a plan.**

## Further reading (in the order worth doing it)

1. Battaglia et al. 2018, *Relational inductive biases, deep learning, and graph networks* — the
   manifesto; read first, it names everything.
2. Zaheer et al. 2017, *Deep Sets* — short, and the theorem is worth carrying.
3. Vinyals et al. 2015, *Pointer Networks* — ~8 pages; the direct ancestor of the v51 head.
4. Bronstein, Bruna, Cohen, Veličković 2021, *Geometric Deep Learning* — skim the blueprint
   chapter; it reframes CNN/GNN/Transformer/DeepSets as one construction.
5. AlphaStar (Vinyals et al. 2019, Nature) — the **architecture supplement** specifically, for
   the entity-list transformer + pointer action head at scale.
6. AlphaFold2 (Jumper et al. 2021) — the pair-representation section; edge biases done best.
7. Zambaldi et al. 2018, *Deep Relational RL* — the closest thing to a controlled experiment on
   whether relational structure helps RL.
8. `entity-gym` (Clemens Winter, GitHub) — small and readable; the only codebase I know that
   treats entity-structured *action* spaces as the primary abstraction.

## See also
- [[shortcut_learning_and_feature_delivery]] — **the companion decision rule**: when feeding a
  computed feature straight to the head is amortization (a plus) vs. a bottleneck you built
  (a minus); gradient starvation, the "never collapse a choice axis" rule, and the measured P1
  ablation showing the model *ignores* collapsed summaries when un-collapsed ones are present
- [[marginalization_and_uncertainty]] — why the operator marginalizes instead of mean-fielding,
  and how a net carries a distribution at all
- [[amortization_gap_and_conditioning]] — the FiLM family, signal-vs-storage, the conditioning ladder
- [[objective_richness_and_representation]] — what the trunk is actually asked to represent
- [[on_policy_self_distillation]] and [[pbs_value_functions_and_search]] — the other side of §6.4's
  division of labour: what real search buys that a depth-1 consequence edge structurally cannot
- `designs/ai_v9/design_generation_roadmap.md` — the operative staged plan (Stage 0–3, the E9
  history decision, the feasibility-spike results, the §3 Stage-2 audit read)
- `designs/ai_v9/design_entity_graph.md` — the entity/edge INVENTORY (E1–E9, D/S/C/V/T/X, the
  sorting rule §0, the nothing-lost audit §6, the open questions §7)
- `designs/ai_v9/design_pointer_action_head.md` §0 — the fresh-generation reset + v51 spec
- `src/agents/model/CLAUDE.md` — the current phase contract these stages supersede; the v51–v57
  versioning notes are the literal specs for everything in Part 6
- `src/agents/model/edge_ablation_audit.py` — the verdict instrument behind §6.2's numbers
- `src/agents/observation/schema.py` — Stage 3's tiling-proof view over the flat observation
