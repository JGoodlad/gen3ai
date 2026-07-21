# Entities, tokens, edge biases, pointer heads — the vocabulary of the ai_v9 skeleton

**TL;DR:** Four ideas that are one idea unfolding — represent the game as *things*, and make
every part of the network respect thing-ness. **Entities**: the state is a set of discrete
things (mons, moves, weather) carrying attributes — meaning from content, not vector position.
**Tokens**: the fixed-width packaging that gives each entity a seat at the attention table (+ a
type embedding saying what kind of thing it is). **Edge biases**: externally computed terms
added to attention scores — the damage calculator stamping "this move↔mon pair matters" onto
the conversation (AlphaFold's pair-bias trick; zero dims consumed; converts sample scarcity
into a prior). **Pointer heads**: action logits read from the token of the entity they select
("point at Rock Slide") instead of a flat positional head — alignment by construction, one
shared scoring function, and the per-move physics reaches its own logit directly (what makes
the flat op head-concat deletable).

## Intuitive level

- **Entities** — today's obs is a 2992-dim vector where dim 217 means "mon 3's HP" by
  convention; the network memorizes the map. Entity representation: the state is a SET of
  things, each with its own attribute bundle. Payoffs: weight sharing (one mon-encoder learns
  from all 12 slots — 12× effective data), permutation invariance (team order is structurally
  noise), graceful variable counts. We are half-entity already (shared `PokemonEncoder` over 12
  slots); moves are the un-entity part — dissolved into their mon's vector, describable but not
  addressable ("no seat = the network cannot think ABOUT Rock Slide").
- **Tokens** — the seat at the table: one fixed-width vector per entity + a type embedding.
  Fixed width lets one set of attention weights process any mix of things. Attention = the
  table conversation: each seat decides who is relevant (query·key) and absorbs summaries
  (values). Current model: 12 mon seats + 2 CLS note-takers; the proposal adds move seats.
- **Edge biases** — attention normally LEARNS relevance from scratch; a bias ADDS a computed
  term: `logit(i,j) = q_i·k_j + b_ij`. A trusted expert (the exact damage calc) whispering
  "these two — pay attention." Costs zero dims (modifies the attention matrix, not tokens);
  sample-efficient (the model is TOLD the 4× move matters on turn one and spends samples on
  what to DO about it). Precedent: T5 relative-position biases, ALiBi, and above all
  AlphaFold's pair biases.
- **Pointer heads** — the flat action head maps a pooled summary → 11 logits where slot 7 means
  "move 2" by convention (defended with GIGO guards). A pointer head scores each entity's
  token: logit("use move k") = f(decision context, move-k's token output). Falls out:
  alignment by construction (the misalignment bug class becomes UNREPRESENTABLE), weight
  sharing across actions (one scoring function, not 11 rows), and per-move physics flowing
  into its own logit (replaces the "express lane" flat concat). Lineage: Pointer Networks
  (Vinyals); AlphaStar unit selection.

## One turn through the machine

Choosing between Rock Slide and switching Swampert: entities get seats; the calculator stamps
edges (Rock Slide→their Zapdos hot: 4×/likely KO; their believed HP-Grass→Swampert: warning);
attention confers (Swampert's token absorbs the threat, Rock Slide's absorbs "their last
bird"); the pointer head scores each move/bench token directly — Rock Slide's logit rises
because ITS OWN token carries the hot edge + composed context. **Physics computes the edges,
attention holds the conversation, pointers choose from the seat that owns the decision.**

## Status and stat moves in this world (owner question, 2026-07-21)

The edge bias never encoded "damage" — it encodes COMPUTED MECHANICAL CONSEQUENCE; damage was
just the first consequence priced. Every move class gets edges; only the bias content differs:

- **Status moves** (Toxic/T-Wave/Spore/WoW) are pairwise like damage: the v27/v37 landing
  physics (type/ability/Sleep-Clause/Sub immunities, [P(major), P(immobilize)]) becomes the
  bias on the status-move↔defender edge — plus computable CONSEQUENCES: para → the speed-order
  FLIPS it causes; burn → the delta to their outgoing table; Toxic → the HP schedule. Learned
  residue: WHICH status target matters this game (attention's contextual selection).
- **Stat moves** (SD/CM/Curse) are self-loops carrying HYPOTHETICAL worlds: run the damage
  kernel once at boosted stats and write the DELTA on edges to each opposing mon ("SD flips EQ
  vs Swampert 3HKO→2HKO; vs Skarmory nothing"). Setup value = a computed table-diff. Learned
  residue: temporal risk (is the setup turn safe; phaze exposure) — composition, attention's
  job.
- **Field/side moves** (Spikes/screens/weather/Roar/Recover/Protect) edge to the GLOBAL/side
  token: Spikes = the chip schedule on their grounded bench entries (composes with the
  their-bench threat quadrant); Reflect = a halving delta on my incoming table; Roar = "drags
  a random bench entry through hazards," priced from existing tables; Recover flips their pko
  cells; Protect rides the computed stall odds.
- **Status CONDITIONS are attributes, not entities**: current burn, sleep-wake belief, protect
  counter stay as columns on the mon's identity/condition token. Things get seats; conditions
  of things get columns.

- **Protect** (the extreme temporal case): token carries identity + the COMPUTED success odds
  (the gen3 100/50/25/12.5 floored-doubling counter — `gen3_protect_odds_v1` re-homed); a
  self-loop edge carries the computed TURN LEDGER (Toxic/sand/Leftovers/Leech ticks + the Wish
  resolve — all deterministic schedules; for stall teams this ledger IS the win condition);
  attention prices the residue — TEMPO cost (their free turn, composed over their tokens) and
  INFORMATION value (Protect-scouting — priced by the history-conditioned scouting-safe
  critic, never by a table).

**Where the number lives (owner: "each move does different damage").** Damage is a property of
the (move, defender, board) TRIPLE, which is exactly why it lives on edges: the move token
carries only invariants (BP/type/category/accuracy — the latent); the edge carries the
pair-specific outcome and is an ACTIVATION, not a weight — recomputed every forward from the
live board (boosts/screens/burn/HP/weather), so state changes update every affected edge
instantly. Within-pair randomness (rolls, crits, accuracy) rides as the distribution summary
`[low, high, crit, pko]` (pko = acc·P(KO|hit)). Per-head attention biases come from a tiny
learned map cell→scalar-per-head (one head can attend by KO-range, another by expected chip).
Rule: pair-varying → edge (recomputed); invariant → token; probabilistic → distribution
summary; future-facing (tempo/information) → attention.

Migration note: almost no NEW computation — v24 secondary chances, v27 landing, v37 split,
wish/protect wiring are already fuzz-validated; the work is re-homing them from flat positional
blocks onto the structure matching their shape.

## Where this lives in our architecture

- Current state: mon tokens exist (TeamTransformer over 12 + CLS pools); moves are MLP-folded
  (`PokemonEncoder` move-net); the op's outputs are a flat post-pool concat into both
  projection heads (no attention sees them; the accretion treadmill).
- The staged path: Form-A cross-attention over top-K move tokens (zero-init out-proj =
  identity-at-init, shippable) → Form B move tokens in the body + physics-as-edge-bias +
  pointer heads (fresh-run / ai_v9) → op head-concat deprecation via the `--unified-obs`
  mask-A/B playbook. Token budget sized by the top-K probe (K=16 owns 94% of channels; tail
  bound insures the rest). See `designs/ai_v8/next_run_plan.md`.

## Synthesis

The four concepts are a discipline for WHERE information lives: per-thing facts in tokens,
pairwise facts on edges, decisions attached to the entity they select — and computed physics
injected at exactly the level it describes (edges), never re-derived by attention (a router
and composer, not a calculator). Under this discipline the bolt-on pattern dies: a new physics
fact is a new edge feature, not a new flat block + wider projections + version bump.

## See also
- [[amortization_gap_and_conditioning]] — the FiLM family, signal-vs-storage, the conditioning ladder
- `designs/ai_v8/next_run_plan.md` — the staged Form-A/Form-B/deprecation items
- `src/agents/model/CLAUDE.md` — the current phase contract this would supersede
