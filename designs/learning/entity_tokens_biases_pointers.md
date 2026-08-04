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

### The right frame: history completes the belief state

In a POMDP, history's job is to supply a **sufficient statistic**, not to be replayed. And most
of the content is **Markovian residue that belongs as state on the entity** — which this project
has been doing piecemeal for a year without naming the pattern: sleep-wake belief, protect
counter, pending Wish, choice-lock state, revealed movesets are all *history compiled into
state*. The irreducibly sequential remainder (opponent tendencies, PP-war pacing, momentum) is
small.

**The decided ladder** (`designs/ai_v9/design_generation_roadmap.md` §4), in landing order:

1. **Per-entity recency features** — last-move-used, damage-taken-last-appearance,
   last-seen-turn, folded onto the mon identity/condition tokens. Cheap, entity-native, and the
   event-sourced fold already computes all of it.
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

**Deferral (owner, 2026-08-03):** history is deferred past Stages 1–2 — the 7×159 block is a
self-contained obs slice with its own encoder path, and Stages 1–2 change the model side only,
so it rides along unchanged (attribution discipline: one fewer simultaneous change in the
riskiest stages). The forcing point is Stage 3; the minimal port there is 7 opaque "history
tokens" through a per-type input projection, which preserves every signal without deciding E9.

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
  predicted +0.19. **E5 (the tail-threat token) is still NOT shipped** — see the truncation
  corollary in [[shortcut_learning_and_feature_delivery]]; it remains the open insurance.
- **Stage 2 SHIPPED (2 slices) — v56 `gen3_edge_bias_trunk_v1` + the D2/S1/S3 slice:** the encoder
  stack is now `BiasedEncoderLayer` (the spike-proven clone taking an additive per-pair per-head
  float bias; the key-padding mask rides the same tensor as a −1e9 addend). `EdgeBias` delivers
  six families behind `--edge-bias-families` — **D1** (our active's 4 moves × the opp's 6 mons,
  the v34 kernel), **D3** (the opp's top-K believed moves × our 6 mons, over the pre-collapse
  `_incoming_rolls`), **D2** (every our-mon's best offense vs the opp active, the v39 switch-in
  kernel move-collapsed), **S1/S3** (status-landing both directions, the v27/v37 kernels gaining
  `per_pair` branches), and **V** (`pairwise_speed` — P(our mon *i* outspeeds opp mon *j*) for
  EVERY (i,j) pair: our real spread vs the believed spread, public paralysis ×0.25 folded both
  sides, the uncertainty-aware sigmoid over the believed per-species speed STD; v1 convention is
  **no stage boosts either side**, the coarse-signal contract — the active's live boost stays the
  incoming block's job). All maps **zero-init ⇒ families ON is bitwise-identical to OFF at
  init**, test-pinned. Measured B=1: D1+D3 = **+0.63 ms** on a ~3.5 ms forward. Note the family
  set grows *without* a version bump — the string gate catches any mismatch, and `'d'` stays a
  FROZEN `d1,d3` alias so a saved config never silently grows maps.
- **Physics delivery today:** the op head-concat is **NOT deleted** — deliberately, per the
  deprecation playbook (build the edge home first; the per-family bias-ablation audit decides
  deletion). So the op currently rides *three* routes: the flat post-pool concat, the v51
  per-action pointer cells, and the new edge biases.
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
- `designs/ai_v9/design_generation_roadmap.md` — the operative staged plan (Stage 0–3, the E9
  history decision, the feasibility-spike results)
- `designs/ai_v9/design_entity_graph.md` — the entity/edge INVENTORY (E1–E9, the nothing-lost audit)
- `designs/ai_v9/design_pointer_action_head.md` §0 — the fresh-generation reset + v51 spec
- `src/agents/model/CLAUDE.md` — the current phase contract these stages supersede
