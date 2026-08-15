# ARCHITECTURE.md — what is true NOW

**Scope: current state only.** History lives in [`CHANGELOG.md`](CHANGELOG.md); versioning
*mechanics* live in `src/agents/model/CLAUDE.md`. If a claim here disagrees with the code, the code
wins and this file is a bug — fix it in the same change.

**"Current" means the production configuration as HEAD resolves it.** Two objects, deliberately
distinct — a generation now turns over every ~2 days, so conflating them is how this file went
stale twice:

| | |
|---|---|
| Production run | `models/ai_v9_10_gen9_intent_distcritic_0813/` (gen-9, launched 2026-08-13) — `model_config.json` `config_version` **69**, `arch_signature` **`gen3_deadline_clock_v1`**; trains on its own pinned worktree |
| Code on HEAD | `MODEL_CONFIG_VERSION` / `ARCH_SIGNATURE` — **read them from `model_version.py`**, never from prose (at this writing: 77 / `gen3_ctx_dedup_v1`) |
| `designs/production_config.json` | the **gen-11** run's config (`ai_v9_13_gen11_labelonly_winprob_0815`) **carried forward to HEAD's schema** (the in-generation migration defaults applied by hand at each schema bump). It stops being a byte-identical run-config copy whenever HEAD's signature moves past the live run's, and exists so this file, the compile gate, the delivery graph and the viewer all derive from ONE real feature set |

Everything below describes what HEAD builds under `designs/production_config.json`. The
machine-derived tables are **generated** (`python -m agents.model.arch_tables`, pinned by
`arch_tables_test.py`) — regenerate them in the same commit as any architecture change; the prose
is hand-written and the generator never touches it.

**The companion artifact is [`architecture_graph.dot`](architecture_graph.dot)** — a *generated*
delivery digraph (seats and sinks as nodes; edges typed by what they physically carry: `bias` = a
softmax-normalised RATIO, `content` = an absolute as token content, `concat` = an absolute at the
head input, `cell` = an absolute per-action, `aux` = training-only and never in the forward). It is
produced by `src/agents/model/delivery_graph.py` and pinned by `delivery_graph_test.py` against a
committed JSON snapshot, so an architecture change that is not reflected in the graph fails a test
rather than quietly rotting. Regenerate both artifacts in the same commit:

```bash
export PYTHONPATH=$PYTHONPATH:src && python -m agents.model.delivery_graph \
    --dot designs/architecture_graph.dot \
    --json src/agents/model/delivery_graph_snapshot.json
```

**Conventions used in this file**
- No version numbers in prose. Blocks and families are named **structurally** — write
  `outgoing_matrix (our moves × opp mons)`, never "the v34 matrix". Where a version tag is
  genuinely needed it goes in a trailing parenthetical, never in the noun phrase.
- Every measured number carries checkpoint / step / state-count / date inline.
- A claim that could not be verified against code or the run config is marked **UNVERIFIED**.

---

## 1. Observation

One flat `float32` vector of **2669** dims, plus an 11-dim `action_mask`, delivered as a Dict obs.
Every number below comes from `agents/observation/constants.py` and
`Gen3ObservationEncoder.get_layout()`. **Never hardcode an offset — read the layout.**

### 1.1 Top-level blocks

| Block | Start | End | Dims | Constant |
|---|---|---|---|---|
| Our team — 6 × per-mon slot | 0 | 696 | 696 | `OFFSET_OUR_TEAM`, `6 × POKEMON_FULL_DIM` |
| Opp team — 6 × per-mon slot | 696 | 1392 | 696 | `OFFSET_OPP_TEAM` |
| Active context ×2 (ours, theirs) | 1392 | 1508 | 116 | `OFFSET_CONTEXT`, `2 × ACTIVE_CONTEXT_DIM` (58) |
| Global env | 1508 | 1528 | 20 | `OFFSET_GLOBAL`, `GLOBAL_ENV_DIM` |
| Board (reactive) | 1528 | 1545 | 17 | `OFFSET_REACTIVE`, `REACTIVE_DIM` |
| *(= `base_dim`)* | | 1545 | | |
| Prev-turn action mask | 1545 | 1556 | 11 | `ACTION_SPACE_SIZE` |
| Turn history — 7 × TurnDelta | 1556 | 2669 | 1113 | `N_HISTORY_TURNS` (7) × `TURN_DELTA_DIM` (159) |
| **Total** | | **2669** | | `Gen3ObservationEncoder.dimension` |

`gen3_entity_rehome_v1` (Stage 3): the two 144-dim matchup matrices and 6 of the 11 reactive
scalars are **deleted** — the D/V edge families compute a strict superset of the matchup signal
GPU-side, `active_status` was byte-redundant with the per-mon condition one-hot, and
`forced_struggle` is derivable (all-zero `active_req_moves` legal bits / the action mask).
`protect_odds`, `trapped` and `maybe_trapped` moved **onto the per-mon slots** (the facts ride
the entities they describe).

> The `OFFSET_*` trailing comments inside `constants.py` used to carry stale evaluated numbers
> (642 / 1284 / …). They are deleted (2026-08-14) — only the expressions remain, with a comment
> forbidding evaluated values there; read `get_layout()` for live offsets.

### 1.2 Per-Pokémon slot — 116 dims (`POKEMON_FULL_DIM`)

`POKEMON_VECTOR_DIM` is 113; `state_encoder` appends the two OUR-side trapping bits and then the
active flag → 116. The active flag stays the **last** dim of the slot on purpose — the model's
`hp_and_active[:, :, -1]` convention is load-bearing (ObsUnpack / DamageOperator / entity seats).

| Field | Offset | Dims | Constant |
|---|---|---|---|
| species id + 6 base stats | 0 | 7 | `POKEMON_SPECIES_OFFSET` |
| item `[id, known, consumed]` | 7 | 3 | `POKEMON_ITEMS_OFFSET` |
| type ids ×2 | 10 | 2 | `POKEMON_TYPES_OFFSET` |
| ability `[id1, id2, dominance, known]` | 12 | 4 | `POKEMON_ABILITIES_OFFSET` |
| status one-hot | 16 | 7 | `POKEMON_CONDITION_OFFSET`, `CONDITION_DIM` |
| 4 × move slot (11 each) | 23 | 44 | `POKEMON_MOVES_OFFSET`, `MOVE_SLOT_DIM` |
| HP fraction | 67 | 1 | `POKEMON_HP_OFFSET` |
| species_known | 68 | 1 | `POKEMON_SPECIES_KNOWN_OFFSET` |
| sleep / toxic counters | 69 | 2 | `POKEMON_COUNTER_OFFSET`, `POKEMON_COUNTER_DIM` |
| spread (IV×6, EV×6, known, nature×5) | 71 | 18 | `POKEMON_SPREAD_OFFSET` |
| Hidden-Power block (`hp_revealed`, 16 type probs) | 89 | 17 | `POKEMON_HP_BLOCK_OFFSET` |
| sleep-wake belief | 106 | 3 | `POKEMON_SLEEP_BELIEF_OFFSET` |
| recency `[since_seen, since_acted, since_hit]` | 109 | 3 | `POKEMON_RECENCY_OFFSET` |
| protect-success odds | 112 | 1 | `POKEMON_PROTECT_OFFSET` |
| trapped (our active only) | 113 | 1 | `POKEMON_TRAPPED_OFFSET`, appended by `state_encoder` |
| maybe_trapped (our active only) | 114 | 1 | `POKEMON_MAYBE_TRAPPED_OFFSET`, appended |
| active flag | 115 | 1 | `POKEMON_ACTIVE_OFFSET`, appended (LAST — load-bearing) |

Move slot (11, `moves.py`): `[id, power/200, has_secondary, has_recoil, type_id, category, known,
current_pp, max_pp, accuracy, never_miss]`.

### 1.3 Board (reactive) block — 17 dims

**5 raw board scalars, then the request-ordered active moves.** Everything derived is gone
(`gen3_entity_rehome_v1`): the matchup matrices live GPU-side as D/V edges, and the per-entity
scalars moved to the mon slots (§1.2).

| Field | Offset in reactive | Dims |
|---|---|---|
| `fainted` (ours, theirs) | 0 | 2 |
| `turns_since_progress` | 2 | 1 |
| `wish_floating_our` | 3 | 1 |
| `wish_floating_opp` | 4 | 1 |
| `active_req_moves` — `[move_num ×4, type_id ×4, legal_now ×4]` | 5 | 12 |

`active_req_moves` is in **request-slot order** (slot *k* ↔ action logit 6+*k*) and is sliced
straight into `ExtractorContext`; it never enters the raw-scalar projection path. The per-mon move
block (§1.2) stays **sorted by id** because the role token concatenates the 4 encodings and is
therefore order-sensitive. Both orders are live simultaneously — that is the reason the pointer
head permutes by move-num identity (§3).

`non_matchup_rest` — the raw-scalar tail the global token and both projection heads read — is
`GLOBAL_ENV_DIM (20) + the 5 board scalars = 25` dims. It stops at the `active_req_moves` offset,
so the embedding-ID block is excluded from it by construction.

### 1.4 Global env — 20 dims

`weather` 7 · `hazards` (spikes ×2) 2 · `clock` **3** · `screens` 8. (`global_env.py`)

**The clock group is 3 scalars** (`CLOCK_DIM`), all sharing the `log(1 + MAX_TURNS)` denominator:

| idx | scalar | formula | where its resolution sits |
|---|---|---|---|
| 0 | log-ELAPSED | `log(1+turn) / log(1+MAX_TURNS)` | the OPENING — turns 1–50 are 58.6% of its range |
| 1 | remaining LINEAR | `(MAX_TURNS − turn) / MAX_TURNS` | uniform — the proportional budget left |
| 2 | log-REMAINING | `log(1 + MAX_TURNS − turn) / log(1+MAX_TURNS)` | the DEADLINE — the last 20 turns are 55.1% of its range |

`MAX_TURNS` (250) is **also the forfeit deadline**: `StallConfig.threshold` imports it, so the
turn the trainee actually loses on and the clock's normaliser cannot drift apart
(`global_env_test.py::test_max_turns_is_the_forfeit_deadline`). Remaining is clamped at 0, so an
over-cap turn saturates rather than going negative or NaN.

Why three and not one: value near a forfeit cap is a function of turns REMAINING, and the
log-elapsed scalar alone gave the last 20 turns **1.5%** of its range (per-turn sensitivity at
turn 249 is 125× lower than at turn 1). A critic cannot price a cliff it has no resolution on,
and that cliff is the link TD must fit FIRST before it can bootstrap value back down a 200-turn
episode. Both remaining forms are provided as raw facts; which one matters is the model's to
learn.

### 1.5 Active context — 58 dims per side

`BOOSTS_DIM` 14 + `VOLATILES_DIM` 44 (`gen3_effects.VOLATILE_DIM`, source-derived).

### 1.6 Turn history — 7 slots × 159 dims

Folded from the event log (`agents/battle/`). Per-slot layout, and the embedded-ID manifest that
routes raw ids to embedding tables, live in `src/agents/observation/CLAUDE.md`. The 7 comes from
`N_HISTORY_TURNS` in `arch_constants.py`.

---

## 2. Feature extractor — the production chain

`Gen3FeaturesExtractor` (`src/agents/model/features_extractor.py`), paired **mandatorily** with
`Gen3DualHeadMaskablePolicy` (`policy.py`) — the extractor returns a `(pi_features, vf_features)`
tuple, which stock SB3 policies cannot consume.

Modules actually built under the production config (`named_children()`) — GENERATED:

<!-- BEGIN GENERATED: modules -->
```
embeddings · unpack · pokemon_encoder · entity_seats · edge_bias · team_transformer · cls_pool ·
hidden_opp_belief · intent_value_reduce · intent_move_cell · t0_species_prior · belief_slots ·
belief_head · move_belief · spread_belief · hp_type_belief_head · damage_op · prefuse_proj ·
assembler · win_head · value_dist_head · pre_proj_norm · projection · value_pre_norm ·
value_projection · activation · alpha_head · beta_head
```

Notably **absent** (`None` on the instance): `pubval_head`.
<!-- END GENERATED: modules -->

### 2.1 Order of operations — the TIER ORDER, and the only order

The belief + physics stack runs **once, before attention**, on every config. There is no flag: the
forward resolves the game in the order the game resolves in, and the four tiers are an **asserted
invariant** (`tier_contract.py`, `tier_contract_test.py`) rather than a property of how the code
happens to be written.

| tier | question | modules |
|---|---|---|
| **T0 RESOLVE** | what is on the board? | `pokemon_encoder`, `belief_slots`, `move_belief`, `hp_type_belief_head`, `spread_belief` |
| **T1 REASON** | what follows from it? | `damage_op`, `entity_seats`, `edge_bias`, `team_transformer` |
| **T2 DECIDE** | what will they do, what are my moves worth? | `belief_head`, `cls_pool`, `alpha_head`, `beta_head` |
| **T3 DELIVER** | one contract, two pools | `hidden_opp_belief`, `assembler`, `win_head`, `pubval_head`, `value_dist_head` |

The contract asserts two things per forward: tier-declared entry points are entered in
**non-decreasing** tier order, and no entry point receives a tensor whose storage was produced by a
strictly **later** tier (checked across two forwards, so a stale stash counts). It is a check on
data flow, not on meaning — it cannot see a T0 leg *recomputing* something intent-like from raw
tokens. Every `nn.Module` child of the extractor must declare a tier or be listed as untiered, so a
new phase cannot escape it.

`BeliefSlots` (T0, injects unknown-mon tokens pre-trunk) and `BeliefHead` (T2, reads refined tokens)
are the one deliberate split: `BeliefHead` is a **training-only side readout** whose output is
stashed for the aux loss and never fed forward, which is exactly what its T2 declaration records.

The concrete steps:

1. **`ObsUnpack`** — slices the 2669-dim vector into `ExtractorContext` (~30 named tensors:
   per-mon blocks, categorical ids, active-slot indices, fainted key-masks,
   `our_active_req_move_{ids,type_ids,legal}`).
2. **`PokemonEncoder`** — per-move network (`MOVE_NET_HIDDEN` `[96,32]`, with the `MoveLatentEncoder`
   latent concatenated in) → within-mon move self-attention → role encoder
   (`ROLE_ENCODER_HIDDEN` `[256,128]`) → **12 × 128 role tokens**. The role input carries the
   E2 active-context injection: each side's 58-dim boosts+volatiles block scattered onto its
   ACTIVE mon's row, bench rows zero (the §6-audited entity home; the global-token/projection
   routes remain — additive delivery, pinned by `e2_ctx_injection_test.py`). Stashes
   `last_move_tokens` `[B,12,4,32]` (sorted-by-id) for the seats and the pointer head.
3. **`MoveBelief`** (T0) — reads the opp **role** tokens, predicts each opp slot's moveset,
   fuses the Smogon log-odds prior, pins revealed moves, and reinjects the soft-embedded moveset
   into the opp role tokens. Stash: `last_move_belief_logits` `[B,6,400]`.
   The prior buffer `[n_species, n_moves]` is **learnset-gated unconditionally**: a move the species
   cannot learn is `logit(1e-6)` (impossible), a legal move keeps its **true** Smogon usage (no rarity
   cap — a surprise tech is never zeroed), a legal move absent from the usage data gets the
   `move_candidate_floor` base, and a row about which nothing is known (national-dex num 0 — the
   unknown-species sentinel an unrevealed slot carries — or a dex gap) is the **flat floor**, never
   "no moves". Non-persistent, recomputed from `data/` at build.
4. **`HPTypeBelief.compose_typed_hp`** — inside the same step: rewrites the posterior so Hidden
   Power exists only at the 16 typed move-nums **355–370** (each `logit(presence · P(type))`) and
   the bare typeless 237 is driven to a finite `-30`. `Σ_t P(HP_t) == presence`, and presence is
   reveal-pinned, so a seen Hidden Power can never be believed away. Every downstream consumer
   (op, edges, seats, BCE, prober) reads this one typed posterior.
5. **`DamageOperator`** — the full 660-dim block (§4), computed on the pre-attention tokens.
6. **`prefuse_proj`** — the op's per-our-mon incoming rows `[B,6,12]` projected to `d_model` and
   **added** to our 6 role tokens. Zero-init ⇒ exactly 0 at init.
7. **`EntityMoveSeats`** — builds 16 extra seats (§2.3).
8. **Edge cells** — 15 per-pair physics tensors computed here, pre-transformer (§5).
9. **`TeamTransformer`** — 36 tokens, 2 `BiasedEncoderLayer`s, `d_model` 128, 4 heads, FFN 256,
   post-LN. One `[B,4,36,36]` float bias carries both the key-padding addend (`-1e9`) and every
   edge family; it is built once and shared by both layers.
10. **`CLSPool`** — three learned queries: `our_cls` over our 6 refined tokens, `their_cls` over
    theirs, `value_cls` over **all 12**. Also extracts `our_active_refined`.
11. **`ProjectionAssembler`** → `pre_proj_norm`/`projection`/ReLU (policy) and
    `value_pre_norm`/`value_projection`/ReLU (value), both emitting `PROJECTION_DIM` = 512.

### 2.2 Dims that flow between phases

| Constant | Value | File |
|---|---|---|
| `ROLE_TOKEN_SIZE` = `D_MODEL` | 128 | `arch_constants.py` |
| `PROJECTION_DIM` | 512 | " |
| `MOVE_NET_HIDDEN` | `[96, 32]` | " |
| `MOVE_LATENT_DIM` / `MOVE_LATENT_HIDDEN` | 32 / 64 | " |
| `ROLE_ENCODER_HIDDEN` | `[256, 128]` | " |
| `ACTIVE_CTX_HIDDEN` | `[64, 32]` | " |
| `POINTER_HIDDEN` | 64 | " |
| `TRANSFORMER_N_LAYERS` / `N_HEADS` / `FFN_DIM` | 2 / 4 / 256 | " |
| `N_HISTORY_TURNS` | 7 | " |
| `NET_ARCH` (SB3 mlp_extractor) | `[512, 512]` | " |

Embedding tables (`Embeddings`, registered exactly once, passed as a forward argument):
species 400×32, move 400×16, item 600×16, ability 100×16, type 20×16.

### 2.3 The 36-token sequence

| Seats | Index range | Token type | Content |
|---|---|---|---|
| our mons | 0–5 | `TOKEN_TYPE_OUR_TEAM` | role token (+ `prefuse_proj` incoming residual) |
| opp mons | 6–11 | `TOKEN_TYPE_THEIR_TEAM` | role token (+ move-belief reinjection) |
| history | 12–18 | `TOKEN_TYPE_HISTORY` | embedded TurnDelta + positional emb |
| global | 19 | `TOKEN_TYPE_GLOBAL` | `[our_ctx, opp_ctx, non_matchup_rest]` → `global_proj` |
| **E3** our active's moves | 20–23 | `TOKEN_TYPE_OUR_MOVE` | move token in **request order**, `move_seat_proj` 32→128 |
| **E4** opp threat moves | 24–29 | `TOKEN_TYPE_THEIR_THREAT` | `threat_seat_proj([latent(32), w, acc, is_phys])`, K = `entity_topk_seats` = 6 |
| **E5** tail threats | 30–35 | `TOKEN_TYPE_THEIR_THREAT` + `tail_marker` | per-opp-mon beyond-top-K residual `tail_proj([p_tail, worst_phys, worst_spec, revealed])` |

`entity_seats.n_seats` = 16 (4 + 6 + 6). Base seat count = `2·TEAM_SIZE + N_HISTORY_TURNS + 1` = 20,
so **every extra seat index is `20 + offset`** — that is what makes the base slices position-stable.
E5 deliberately reuses `TOKEN_TYPE_THEIR_THREAT` rather than adding a 7th token-type row (growing
the table changes every model's state_dict).

---

## 3. Heads — exactly what each consumes

This section is the canonical answer to "what does head X read". Widths verified by forward pass,
2026-08-08.

### 3.1 / 3.2 The head inputs — GENERATED

**The op head-concat is DEAD (`gen3_no_concat_v1`, v61)** — the flat block enters neither head;
the op reaches the policy via the pointer cells (lossless per-action), the prefuse token
injection, and the edge cells. **The active-ctx concat is DEAD too (`gen3_ctx_dedup_v1`, v76)** —
the per-side encoded ctx pair was duplicated delivery (the E2 injection carries each side's full
raw ctx block on its active token; the global token is a second route). `non_matchup_rest` stays:
its only token route is the global token, which no pool reads directly.

The exact concat composition and widths of both projections, under the production config on
HEAD, are generated below — never hand-edit inside the markers.

<!-- BEGIN GENERATED: head-inputs -->
**`pi_projection` — `Linear(1177, 512)`** (LayerNorm → Linear → ReLU). Input concat, in order:

| Part | Dims | Source |
|---|---|---|
| `our_team_pooled` | 128 | `CLSPool.our_cls` over our 6 refined tokens |
| `their_team_pooled` | 128 | `CLSPool.their_cls` over their 6 |
| `our_active_refined` | 128 | our active slot's refined token |
| `non_matchup_rest` | 25 | global env + board scalars (`_non_matchup_rest_dim`) |
| `hidden_opp_belief` | 768 | `HiddenOppBeliefPool` — k=6 × `D_MODEL` |
| **total** | **1177** | == `projection.in_features`, asserted at generation |

**`vf_projection` — `Linear(1241, 512)`** (LayerNorm → Linear → ReLU). Input concat, in order:

| Part | Dims | Source |
|---|---|---|
| `value_pooled` | 128 | `CLSPool.value_cls` over **all 12** team tokens |
| `non_matchup_rest` | 25 | shared with pi |
| `hidden_opp_belief` | 768 | `HiddenOppBeliefPool` — k=6 × `D_MODEL` |
| seed readout | 256 | `MultiSeedValueReadout` — k=4 × 64 over `OpTensors.incoming_rows` |
| intent reduce | 64 | `IntentValueReduce` — α-weighted pair cells, appended AFTER the assembler |
| **total** | **1241** | == `value_projection.in_features`, asserted at generation |
<!-- END GENERATED: head-inputs -->

The value head does **not** read `our_active_refined` (`value_active_readout` is off), and does not
read either team pool. Its board summary is `value_pooled` plus the **multi-seed window**: k=4
learned queries cross-attend (explicit softmax, dead mons key-masked) over the op's per-our-mon
incoming rows — the critic's magnitude read after the concat's death, MULTIPLICITY not width
(ledger P3 refuted width only). Every `train()` logs the `value_seeds/*` collapse contract
(`agents/model/seed_diagnostics.py`: query/output cosine, uncentered effective rank, the VICReg
variance target) with the VICReg trigger pre-registered in that module — the z_arch lesson,
mechanized. **The trigger FIRED on gen-5** (`value_seeds/out_effective_rank` 1.0 sustained
196k→15.7M steps — the k=4 outputs identical).

**Two pressures were then applied to those seeds, and BOTH are deleted (v78). The measurement is
why.** `--value-seed-vicreg-coef` (v62) was the repulsive one — a scale-relative
variance+**covariance** floor on the seed outputs. Gen-6 ran it and every term moved (std_rel
0.002 → 0.53, correlation → 0.19) while effective rank stayed **1.05**, because the deviations
occupy **less than one direction** (centered PR 0.846): seeds 0/1/2 kept near-identical attention
while seed 3 alone broke away. Repulsion buys spread, not multiplicity. `--seed-quantile-coef`
(v63) was the positive counterpart — seed k predicts **quantile τ_k** of the return through **one
shared** `Linear(dim,1)`, so k different predictions require k different seed reads. Gen-7 ran it
and it worked on its own terms (`quantile_crossing_rate` 0.456 → **0.000**, `quantile_spread`
0.007 → **1.016** at 10.6M steps — the four seeds do predict four ordered quantiles) yet
`out_effective_rank` reached only **1.157** against a ceiling of k=4.

**The structural reason is shared, which is what closed the line**: a **shared** readout constrains
only each seed output's component along its own weight vector, leaving every orthogonal direction
free. Seed MULTIPLICITY is therefore not the axis the critic was missing, and no coefficient on
either term reaches it — so both flags and both modules were deleted rather than retuned.
`seed_diagnostics.py`, the MEASUREMENT that produced these numbers, **stays** and still logs the
`value_seeds/*` contract every `train()`.

**That null is what `--value-threat-inject` (v64) responds to.**

`--value-threat-inject` takes the third route instead — **magnitude as token content, per entity**.
For each of OUR mons `j`, the op's α-weighted incoming row (`Σ_k α_k · pair_in[k, j, :]`, α = the R1
`belief_mean` rung, which the flag forces on because R0 `hard_max` builds no reducer) is projected
by ONE shared zero-init `Linear(13, 128)` and added to that mon's token on **the value pool's copy
only**. `value_cls` then pools augmented tokens; `our_cls`, `our_active_refined` and the pointer
head all read the untouched tensor, so `pi` is bit-identical at **any** weight — asserted against a
large random projection, not merely at init. The route is invariant under permuting their moves (α
is shared across defenders by Contract W), equivariant under permuting ours (the row rides mon `j`'s
token), and invariant at the pool — unlike the deleted flat concat, whose meaning was slot-ordered.
`W_inj` is covered by `restore_identity_init()` (ledger M1) and that is gated on a real
`MaskablePPO` build, not a bare extractor. Structural + version-checked, fresh runs only; OFF
(production) builds no module and leaves the op on `hard_max`. **v1 substitutes α := normalize(w),
a PRESENCE belief where the design wants a supervised USAGE belief** — deliberately, so a null
indicts the delivery route rather than the belief.

### 3.3 The action head is the pointer head — there is no flat `action_net`

`Gen3DualHeadMaskablePolicy._build` replaces SB3's flat `Linear(latent, 11)` with a **raising stub**
and rebuilds the optimizer; `PointerNativeActionHead` produces the logits, and
`_get_action_dist_from_latent` is the single funnel all three logit sites pass through.

Shared context for all three families: **`latent_pi`** — the policy tower's output, i.e. everything
in §3.1 after the mlp_extractor. So the op block, the beliefs, and any head-level modulation
condition every pointer score.

Output layout is `[switch ×6, move ×4, struggle]` (`agents/action/constants.py`).

| Logit | Entity token | Physics cells | Cell width |
|---|---|---|---|
| **move k** (logit 6+k) | the **refined E3 seat k** (`last_pointer_inputs[0]`, `[B,4,128]`) — post-attention, board-aware, already permuted sorted-by-id → **request** order by move-num identity | `[low, high, crit, pko, p_land, known, sec×7]` | **13** (`_PTR_MOVE_CELL`) |
| **switch j** | our-team token *j* (`our_team_out[:, j]`, `[B,6,128]`) — the same post-transformer token the CLS pools read | the incoming per-defender row (12) + `[phys_high_cb_j, pko_cb_j, p_cb]` | **15** (`_PTR_SWITCH_CELL_IN`) |
| **struggle** | none — context only | none | 0 |

Scoring: `tanh(proj(token ⊕ cells) + ctx_proj(latent_pi))` → a zero-init `Linear(64, 1)`.
Move logits are multiplied by `move_valid`, so an unresolved request slot contributes **exactly 0**
rather than a score computed from a zero token.

**What the switch logit does NOT see in this config:** the `outgoing_matrix_all` attacker row
(`[cells×16, p_outspeed_j, alive_j]`, 18 dims, `_PTR_SWITCH_CELL_OAX`). That row is gated on
`damage_matrices_outgoing_all`, which is **false** — so the switch cell is 15, not 33, and the
per-candidate **offense** read does not exist. The switch logit's physics is purely defensive
(what this mon takes on the switch-in) plus whatever the trunk carried into `our_team_out`. The
`d2` edge family (§5) is the only other route by which a bench mon's offense reaches its own token.

Secondary channel widths: `sec×7`, not `sec×10` — the outgoing block prices only the 7 secondary
columns an our-side gen3 move can inflict (`_OUT_SEC_COLS`; slp/psn/tox were dropped as structural
zeros). The `PointerNativeActionHead` docstring still says `sec×10`; the code is right (§8).

Position-equivariance is structural: one shared scorer per entity family, so permuting the team
permutes the logits, and a sorted-vs-request misalignment is unrepresentable at the logits.
Cold start: all three scorers are zero-init and built **after** SB3's ortho-init pass, so every
logit is exactly 0 at step 0 ⇒ uniform-over-legal.

**The pointer route is POLICY-ONLY.** `pointer_head` is reached solely through
`_get_action_dist_from_latent(latent_pi)`; every value path is
`forward_critic(vf_features) → _critic_value`, which never touches it. So the per-action `cell`
channel exists for the actor and **not** for the critic — the critic's op-physics routes are the
`MultiSeedValueReadout` window over the typed `incoming_rows`, `--value-threat-inject`'s token
content on the value pool's copy, and (when on) the `intent_value_reduce` term — all vf-only,
all reading the op through `OpTensors` views rather than flat offsets.

### 3.4 Side readouts

A side readout hangs off `value_pooled` AFTER the pools and stashes its logits; none of them ever
enters `pi` or `vf`, so none changes a projection width. Two are built in this config:

| head | flag | grad flow |
|---|---|---|
| `win_head` | `win_prob_mode` **`shaping`** (coef 0.05) | live `value_pooled` — the win objective also shapes the trunk (`read_only` would stop-grad it) |
| `value_dist_head` | `value_dist_mode` **`shaping`**, 51 atoms over [−12, +12] | live — and `value_from_dist` **true**, so this head IS the critic |
| `pubval_head` | `pubval_mode` `none` | not built |

**`value_from_dist` true makes the distributional head load-bearing rather than diagnostic**: GAE,
bootstrapping and deployment all read `E[Z]` (`policy._critic_value`), the HL-Gauss cross-entropy
is the primary value loss at `vf_coef` weight, and the scalar `value_net` freezes as a fallback.
So "the critic" in this config is the *categorical* head, not `value_net`. PopArt is still on
(`use_popart` true, which forces `--clip-range-vf none`).

Belief heads run under `belief_grad_mode` **`label_only`**: their outputs are published stop-grad
to every forward consumer, so no policy/value gradient reaches a belief head's parameters and the
belief is trained by its supervised labels alone. The heads' trunk READ stays live, so the label
loss still shapes the trunk — see `src/agents/model/CLAUDE.md` for the four-route table.

---

## 4. The `DamageOperator` output block

**`out_dim` = 660** under the production config. Layout is contiguous, in this order, and every
sub-block is appended after the previous one (so enabling a later one never moves an earlier
offset).

| # | Sub-block | Width | Present? | Gate |
|---|---|---|---|---|
| 1 | incoming per-mon — 6 × `[phys(low,high,crit,pko,acc), spec(…), p_outspeed, provenance]` | 6 × 12 = **72** | ✅ | `damage_op` |
| 2 | Choice-Band tail — `phys_high_cb ×6`, `pko_cb ×6`, `p_cb` | **13** | ✅ | `damage_op` |
| | *(1 + 2 = `incoming_dim` = **85**)* | | | |
| 3 | outgoing single-active — 4 moves × `[low,high,crit,pko]`, `p_outspeed`, 4 × 7 secondary | **45** | ✅ | `damage_outgoing` |
| 4 | status-landing — `P(lands) ×4`, `known ×4` | **8** | ✅ | `damage_outgoing` |
| 5 | `outgoing_matrix` — our 4 moves × opp 6 mons | **126** | ❌ **ABSENT** | `damage_matrices_outgoing` = **false** |
| 6 | `incoming_matrix` — K=6 headers (51 each) + 6 mons × 6 moves × 6-wide cells | 6×51 + 6×6×6 = **522** | ✅ | `damage_matrices_incoming` = true, K = `damage_topk_k` = 6 |
| 7 | `outgoing_attacker_matrix` (OAX) — our 6 mons × 4 moves + `p_outspeed ×6` + `alive ×6` | **108** | ❌ **ABSENT** | `damage_matrices_outgoing_all` = **false** |
| | **Total** | **85 + 45 + 8 + 522 = 660** | | |

The block passes through a learned per-channel `out_gain` (a Parameter, multiplicative only, so the
"no threat ⇒ exactly 0" gates stay clean) before it reaches the heads and before `pointer_cells`
slices it — so the pointer path and the flat concat can never disagree on a value.

**The pair-reduction rungs exist but are INERT in production** (`agents/model/pair_reduce.py`,
`design_pair_reduction.md` §8.1 steps 3–4): `DamageOperator(reduce_how=…)` — constructor-only, no
CLI flag, no config field — can build Contract-W/L reducers (R1 `belief_mean` / R2W `learned` /
R2L `deepsets_{sum,max}` / R3 `multi`) BESIDE the legacy per-channel hard max. The production
default `"hard_max"` builds **nothing**: no params, no state_dict keys, no forward work. A
non-default rung only stashes `last_reduced_extra` [B,6,extra_dim]; nothing consumes it — delivery
+ versioning is gen-6 work, gated on the §8.1 step-0 audit.

Also **not present anywhere** (deleted with the op block trim, not merely off): the opp-active
collapsed effect scalars, the opp-active collapsed incoming-secondary scalars, the outgoing
slp/psn/tox columns, and the lean top-K block. `damage_topk_k` now sizes the incoming matrix and
nothing else; `K > 0` without `damage_matrices_incoming` **raises** in both the extractor and the
op.

### 4.1 Per-block dependence — the current-config measurement

Source: [`research_state/measurements/gen3_op_block_dependence_6k.json`](research_state/measurements/gen3_op_block_dependence_6k.json)
— **this run's `checkpoint_9600000_steps.zip`, 6000 real eval states, 2026-08-07.** Method: zero
each sub-block as a contiguous slice of the op's output **at the `ProjectionAssembler` concat only**
(edges, the `prefuse_proj` injection and the pointer cells stay live) → masked KL against the
policy's own distribution. It answers *what does the HEAD still lean on*, not *what does the model
use*.

| Sub-block | Width | KL | Argmax flips | Shuffle-control KL |
|---|---|---|---|---|
| `FULL_CONCAT` (ceiling) | 660 | 0.2444 | 23.6% | 0.1818 |
| `incoming_matrix` | 522 | **0.2534** | **24.2%** | 0.1357 |
| outgoing single-active | 45 | 0.0176 | 5.4% | **0.0254** |
| incoming per-mon + CB | 85 | 0.0174 | 6.2% | 0.0158 |
| incoming per-mon | 72 | 0.0160 | 6.0% | 0.0130 |
| Choice-Band tail | 13 | 0.0013 | 1.4% | 0.0014 |
| status landing | 8 | 0.0006 | 0.8% | 0.0010 |

**Read the shuffle column first.** Shuffling a block across the batch preserves its marginal
statistics and destroys its state-specific content. For the outgoing single-active block, the
Choice-Band tail and status landing, the shuffle arm **meets or exceeds** the zero arm — at this n
those blocks show no dependence the probe can separate from noise. The one clean signal is the
`incoming_matrix`: zeroing it costs essentially the entire concat ceiling.

Two caveats that bound this: it is **mid-training** (9.6M of 40M, and edge dependence grew ~3× with
training in earlier generations), and it has not been re-run at end of run.

### 4.2 ⚠️ The older P1 table is NOT current — do not quote it

The frequently-cited per-block ablation table (`tmp/op_block_ablation_probe.py`, **2026-07-25**,
4000 real eval states, per-block zero → masked KL, ceiling 0.9385 = zeroing the whole op) was
measured on a **different model and a different config**, and §4.1 above supersedes it. Three things
make it non-transferable:

1. **It predates this generation entirely.** It was taken before the pointer-native action head
   existed, when the op block reached only a flat positional `action_net`. The current model routes
   the same numbers through per-action pointer cells *as well as* the concat, so "how much does the
   policy depend on block X" is a different question with a different mechanism.
2. **Two of the blocks it ranks do not exist here.**

| Block in that table | % of that run's ceiling | Exists in production config? |
|---|---|---|
| OUTGOING (per-action, un-collapsed) | 65.7% | ✅ yes (sub-block 3) |
| `outgoing_attacker_matrix` | 21.4% | ❌ **no** — `damage_matrices_outgoing_all` false |
| `incoming_matrix` (mon × move) | 15.4% | ✅ yes (sub-block 6) |
| incoming per-mon | 12.7% | ✅ yes (sub-block 1) |
| status-landing | 8.8% | ✅ yes (sub-block 4) |
| `outgoing_matrix` | 6.3% | ❌ **no** — `damage_matrices_outgoing` false |
| Choice-Band | 2.9% | ✅ yes (sub-block 2) |
| incoming effect (collapsed) | 1.2% | ❌ deleted from the code |
| incoming secondary (collapsed) | 0.1% | ❌ deleted from the code |

3. **Its headline is reversed by §4.1.** That table says the OUTGOING families dominate; the
   current-config measurement puts the outgoing single-active block at its own shuffle-control level
   and the `incoming_matrix` at the whole ceiling. Both cannot be true of the same model. The
   sub-block ordering is a fact about a model, not about the architecture.

Its raw output is not in version control anywhere (only the derived table in
`designs/learning/shortcut_learning_and_feature_delivery.md` survives), so it could not be archived
under `research_state/measurements/`.

---

## 5. Edge families — physics as attention bias

`edge_bias_families = "d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5"` — **all 15 families are on** in
the production config. Each maps its per-pair cell through a **zero-init**
`Linear(cell_width, 2 · n_heads)`: one head-set for `row→col`, one for `col→row`. Zero-init ⇒ the
whole edge system is bitwise-identical to `off` at initialisation.

Seat indices as in §2.3: our mons `[0:6]`, opp mons `[6:12]`, global `19`, E3 `[20:24]`,
E4 `[24:30]`, E5 `[30:36]`.

### 5.1 The from × to grid

| Family | Placed at (row, col) + transpose | Cell width | Cell contents |
|---|---|---|---|
| **d1** | E3 move seat *k* × opp mon *d* | 6 | `[low, high, crit, pko, type_mult, revealed]` — our active's move vs each opp mon |
| **s1** | E3 seat *k* × opp mon *d* | 2 | `[land, land·immob]` — will this status move land on that mon |
| **c1** | E3 seat *k* × opp mon *d* | 7 | `[is_boost, d_best_high, d_best_pko, d_outspeed, hp_cost, d_in_high, d_in_pko]` — post-setup deltas, offensive **and** defensive halves |
| **c2** | E3 seat *k* × opp mon *d* | 7 | `[is_status, land, d_their_outspeed, d_in_phys_high, d_sched, d_in_all_slp, e_slp_free_turns]` — what *landing* would do |
| **c3** | E3 seat *k* × opp mon *d* | 3 | `[is_recovery, d_in_pko, rest_sleep_turns]` — does healing beat their KO |
| **c5** | E3 seat *k* × **our** mon *j* | 4 | `[is_bp, d_best_high, d_best_pko, d_outspeed]` — Baton-Pass receiver axis |
| **c4** | E3 seat *k* × **global** (19) | 4 | `[is_protect, p_success, net_ours, net_theirs]` — the turn a successful Protect banks |
| **d3** | E4 threat seat *c* × our mon *i* | 5 | `[high, pko, eff, is_phys, w]` — their believed move vs each of our mons |
| **s3** | E4 seat *c* × our mon *i* | 3 | `[land, land·immob, w]` |
| **d2** | our mon *i* × opp **ACTIVE** (one-hot column) | 4 | `[best_high, best_pko, p_outspeed, alive]` — our bench's offense vs their active |
| **d4** | our mon *i* × opp mon *j* (active column pre-zeroed) | 4 | `[phys_high, spec_high, phys_pko, spec_pko]` — the opp **bench**'s believed threat |
| **v** | our mon *i* × opp mon *j* | 3 | `[p_outspeed, both_alive, revealed_j]` |
| **t** | our mon *i* × opp mon *j* | 2 | `[P(i traps j), P(j traps i)]` |
| **x** | each mon × **global** (both sides) | 4 | `[entry_chip, pursuit_p, pursuit_eff, grounded]` |
| **g** | each mon × **global** (both sides) | 4 | `[leftovers, weather_chip, status_tick, leech]` — signed maxhp fractions, Toxic at its ramped next tick |

**No family targets the E5 tail seats** — they are token content only.

### 5.2 Requirements (enforced at extractor build, `ValueError`)

| Families | Require |
|---|---|
| d1, s1, c1, c2 | `damage_op` **and** `damage_outgoing` |
| c3, c4, c5, d2, d4, g, t, v | `damage_op` |
| x | `damage_op` |
| d3, s3 | `entity_topk_seats > 0` (the bias rows *are* the E4 seats) |

All are satisfied in the production config.

### 5.3 What an edge can and cannot carry

Attention weights are softmax-**normalised**, so an edge bias moves *who attends to whom*: what it
writes is a **ratio within its row**, not an absolute magnitude ("53% of max HP"). The two channels
that can carry an absolute are **token content** (`prefuse_proj`) and **per-action pointer cells**
(§3.3). This is a capacity/conditioning argument, not an impossibility proof; the reasoning is in
`designs/learning/shortcut_learning_and_feature_delivery.md`.

### 5.4 Edge-family audit — the current-config measurement

Source: [`research_state/measurements/gen3_edge_family_audit_9p6M.json`](research_state/measurements/gen3_edge_family_audit_9p6M.json)
— **this run's `checkpoint_9600000_steps.zip`, 6000 eval-trace states, 2026-08-07**, produced by
`src/agents/model/edge_ablation_audit.py`. Each row zeroes one family's bias map and measures masked
KL / argmax flips / |dV| against the unablated policy.

| Family | KL | Argmax flips | \|dV\| |
|---|---|---|---|
| `d2` (bench offense → their active) | 0.0426 | 7.6% | 1.308 |
| `d1` (our moves → their mons) | 0.0345 | 6.0% | 0.274 |
| `v` (speed) | 0.0035 | 2.9% | 0.651 |
| `d3` (their threats → our mons) | 0.0013 | 1.9% | 0.141 |
| `d4` (their bench threats) | 0.0015 | 1.1% | 0.492 |
| `s3` | 0.0003 | 0.9% | 0.063 |
| `s1` | 0.0017 | 0.8% | 0.022 |
| `t` (trapping) | 0.0003 | 0.7% | 0.121 |
| `c1` (setup consequence) | 0.0002 | 0.4% | 0.015 |
| `c2` (status consequence) | 0.0007 | 0.4% | 0.017 |
| `x` (entry/exit) | 0.0000 | 0.3% | 0.036 |
| `c3` (recovery) | 0.0002 | 0.2% | 0.018 |
| `c5` (Baton Pass) | 0.0000 | 0.2% | 0.018 |
| `g` (end-of-turn ledger) | 0.0000 | 0.1% | 0.016 |
| `c4` (Protect) | 0.0000 | 0.1% | 0.001 |
| **all families off** | 0.1011 | **13.9%** | 1.857 |
| **op head-concat off** | 0.2444 | **23.6%** | 5.669 |
| **concat + pointer cells off** | 0.6369 | **37.8%** | 5.669 |

Two things to carry from this, both **mid-training (9.6M of 40M)** and provisional:

- **The OUTGOING damage families dominate** (`d2`, `d1`), the same ordering gen-1 and gen-2 showed;
  every consequence family (`c1`–`c5`) and the board-level `g`/`x` are at or below 0.4% flips so
  far. Edge dependence grew ~3× with training in earlier generations, so a low number here is not
  yet a verdict on a family.
- **The concat is not starved by the edges, and vice versa.** Zeroing the head concat flips more
  actions (23.6%) than zeroing the *entire* 15-family edge system (13.9%) — replicated in gen-1,
  gen-2 and gen-2.5 (`research_state/measurements/README.md` has the cross-run table). That is the
  expected result if the two carry different things: a bias is a softmax-normalised ratio, the
  concat is an absolute.

Earlier generations' audits are archived alongside for the cross-run comparison. **The gen-1 and
gen-2 `v` rows were measured on the speed-stat GIGO bug** (§8) and describe the buggy feature.

---

## 6. Flags — production value and status

The flag/status and loss-coefficient tables are GENERATED from `designs/production_config.json`
resolved against HEAD (`python -m agents.model.arch_tables`; drift pinned by
`arch_tables_test.py`) — a hand-derived version of this table went stale twice within one week of
generation turnover, and its stale rows are precisely what mis-briefed downstream readers.

**Status legend** — `ACTIVE`: on and doing work. `OFF`: not enabled. `INERT`: nominally set but
does nothing given another setting.

<!-- BEGIN GENERATED: flag-table -->
| Flag | Production value | Status |
|---|---|---|
| `attend_unrevealed_opponents` | `true` | ACTIVE |
| `belief_grad_mode` | `"label_only"` | ACTIVE |
| `consequence_topk` | `6` | ACTIVE |
| `damage_candidate_k` | `0` | OFF |
| `damage_matrices_incoming` | `true` | ACTIVE |
| `damage_matrices_outgoing` | `false` | OFF |
| `damage_matrices_outgoing_all` | `false` | OFF |
| `damage_op` | `true` | ACTIVE |
| `damage_outgoing` | `true` | ACTIVE |
| `damage_topk_k` | `6` | ACTIVE |
| `edge_bias_families` | `"d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5"` | ACTIVE |
| `entity_tail_seats` | `true` | ACTIVE |
| `entity_topk_seats` | `6` | ACTIVE |
| `hp_belief_mode` | `"composed"` | ACTIVE |
| `intent_move_cell` | `true` | ACTIVE |
| `intent_value_reduce` | `true` | ACTIVE |
| `move_belief_mode` | `"both"` | ACTIVE |
| `move_candidate_floor` | `0.02` | ACTIVE |
| `move_latent` | `true` | ACTIVE |
| `move_prior_fusion` | `true` | ACTIVE |
| `opp_belief_cls_k` | `6` | ACTIVE |
| `opp_belief_slots` | `true` | ACTIVE |
| `opp_intent` | `true` | ACTIVE |
| `opp_intent_grad_mode` | `"detached"` | ACTIVE |
| `pubval_mode` | `"none"` | OFF |
| `species_prior_fusion` | `true` | ACTIVE |
| `spread_belief` | `true` | ACTIVE |
| `spread_belief_nature` | `true` | ACTIVE |
| `t0_species_prior` | `true` | ACTIVE |
| `threat_prob_outspeed` | `false` | OFF |
| `value_active_readout` | `false` | OFF |
| `value_dist_bins` | `51` | ACTIVE |
| `value_dist_mode` | `"shaping"` | ACTIVE |
| `value_dist_vmax` | `12.0` | ACTIVE |
| `value_dist_vmin` | `-12.0` | ACTIVE |
| `value_threat_inject` | `true` | ACTIVE |
| `win_prob_mode` | `"shaping"` | ACTIVE |
| `hp_type_belief_coef` | `0.05` | ACTIVE |
| `move_belief_coef` | `0.05` | ACTIVE |
| `move_belief_latent_coef` | `0.05` | ACTIVE |
| `opp_belief_aux_coef` | `0.05` | ACTIVE |
| `pubval_coef` | `0.1` | INERT — no `pubval_head` |
| `spread_belief_coef` | `0.05` | ACTIVE |
| `value_dist_coef` | `1.0` | ACTIVE |
| `value_tail_weight` | `0.3` | ACTIVE |
| `vf_coef` | `0.5` | ACTIVE |
| `win_prob_coef` | `0.05` | ACTIVE |
<!-- END GENERATED: flag-table -->

### 6.3 Reward config (resume-immutable, `check_reward_config`)

`draw_penalty` −30.0 · `no_progress_penalty` 0.15 · `mat_alive_weight` 1.25 · `bias_additivity` 1.0
· `self_ko_hp_penalty` 0.0 · `switch_bias_weight` 0.0 · `bias_redesign` false ·
`drop_redundant_bias` false · `drop_switch_bias` false · `all_shaping_pbrs` false · `stall_pbrs`
false.

### 6.4 Runtime knobs (never versioned, must be re-passed on every resume)

`--compile-opponents` (on in this run) · `--grad-accum-steps 4` · `--grad-checkpointing` ·
`--async-rollout` · `--use-bridge {off,node,rust}` (this run: `node`). These do not appear in
`model_config.json` and are **not** inherited on resume.

---

## 7. Training-only obs keys — the leak-safety list

These are Dict-obs keys emitted by `Gen3Env` for supervision. **The forward reads only
`obs["observation"]`** (`ObsUnpack.forward`), so none of them can reach `pi`/`vf` or any pointer
logit. Declared conditionally, so a key absent from the space is simply not emitted.

| Key | Shape | Consumer | Emitted when | In production? |
|---|---|---|---|---|
| `belief_species` | int64 `[6]` | `BeliefHead` species CE | `opp_belief_aux_coef > 0` **or** `move_belief_mode != off` | ✅ **emitted** (via the move-belief clause) — but **unconsumed**: no `BeliefHead` exists |
| `belief_moves` | int64 `[6,4]` | `BeliefHead` moves BCE (Hungarian) | " | ✅ emitted, unconsumed |
| `known_moves` | int64 `[6,4]` | `MoveBelief` BCE | `move_belief_mode` ∈ {revealed, both} | ✅ emitted; **unconsumed** — `move_belief_coef` = 0.0 |
| `belief_spread` / `belief_spread_mask` | f32 `[6,5]` / `[6]` | `SpreadBelief` regression | `spread_belief` **and** `spread_belief_coef > 0` | ❌ |
| `belief_nature` / `belief_nature_mask` | int64 `[6]` / f32 `[6]` | nature CE | " | ❌ |
| `belief_ev` / `belief_ev_mask` | f32 `[6,5]` / `[6]` | EV smooth-L1 | " | ❌ |
| `hp_type_label` / `hp_type_mask` | int64 `[6]` / f32 `[6]` | HP-type CE | `move_belief_mode != off` **and** `hp_belief_mode == composed` **and** `hp_type_belief_coef > 0` | ✅ **emitted and consumed** |
| `win_target` / `win_mask` / `win_margin` | f32 `[1]` each | win-prob aux (MC outcome — a **future** label) | `win_prob_mode != none` | ❌ |
| `pubval_target` / `pubval_mask` | f32 `[1]` each | public-value aux | `pubval_mode != none` | ❌ |
| `defensive_opportunity` | f32 `[1]` | state-conditioned entropy boost | `--defensive-entropy-boost > 1.0` (default 1.0) | ❌ |
| `distill_mask` | f32 `[1]` | exploiter-distillation KL gate | `--distill-coef > 0` with teacher teams | ❌ |

So in production **four privileged keys ride the rollout buffer but only one is read**
(`hp_type_label`). `belief_species` / `belief_moves` / `known_moves` are emitted because the emit
gate keys on `move_belief_mode != off` while their losses key on separate coefficients that are
zero. That is buffer cost, not a leak — but it does mean "the label is emitted" is not evidence
that a belief is supervised.

Only the **trainee** `Gen3Env` emits any of these. Eval and self-play opponents play through
`RLPlayer`, which never constructs them.

Two side-channel stashes are also never fed forward: `last_belief_target_latent` (computed only
under `torch.is_grad_enabled()`) and `last_move_latent_table`. The pinned no-leak tests are
`belief_slots_test.test_latent_target_is_no_leak`,
`damage_op_test.test_op_is_leak_free_of_privileged_keys`, the bridge fuzz
`poke_env_gaps/belief_labels_fuzz_test.py`, and — as a **graph invariant** —
`delivery_graph_test.test_no_aux_edge_reaches_the_forward`, which asserts that no `aux` edge
terminates at `pi_projection`, `vf_projection`, or any pointer logit.

---

## 8. Known contradictions between the old prose and the code

> **Update 2026-08-14:** several entries below are FIXED by the ctx-dedup / OpTensors /
> generated-tables pass: this file's header and flag/head tables no longer hand-state config
> values (generated from `production_config.json`), the delivery graph no longer draws the dead
> op→head concat edges (it drew them for five days after the v61 deletion, pinned by its own
> test — the exact rot class it exists to prevent), and `designs/CLAUDE.md`'s state table was
> brought to gen-9/v76, and the `constants.py` stale offset comments are deleted. (The
> observation leaf's opening had already been fixed separately — item 2 below is resolved.)
> The list below is kept as found (2026-08-08) for the record.

Found while deriving this file (2026-08-08). Each is a place where a doc asserted something the
code does not do. None are fixed in `src/` by this pass — they are recorded so the next reader does
not re-derive them.

1. **Root `CLAUDE.md` reactive-block prose was two revisions stale.** It described "the 414-dim
   reactive block (**19 scalars**)" with `turns_since_progress` at `vec[14]`, protect-odds at
   `vec[15]`/`vec[16]`, wish at `vec[17]`/`vec[18]`. Real: **311 dims, 11 scalars**, progress at
   offset **6**, protect at **7**/**8**, wish at **9**/**10**. The summary table 60 lines above it
   was correct — the table and the prose contradicted each other in the same section.
2. **`src/agents/observation/CLAUDE.md` opens with "2889-dim"**; the live obs at audit time
   was **2925**; since `gen3_entity_rehome_v1` it was **2667**, and since
   `gen3_deadline_clock_v1` it is **2669** (the
   per-mon recency block added 12 × 3). Its per-block reference section then describes the
   pre-deletion 414-dim reactive layout and the 51-dim incoming-damage / 44-dim move-effect blocks
   as if present. Its own inline banner says to treat the deletion note as authoritative — i.e. the
   file tells you not to trust the rest of the file.
3. **`src/agents/model/CLAUDE.md` states `ARCH_SIGNATURE` in three places with two different
   values** (`gen3_op_block_trim_v1` in the phase-structure rules, `gen3_edge_bias_trunk_v1`
   later) and states `MODEL_CONFIG_VERSION` as 31/32/37/38/40/41/43/44/45/46/47/53/55/57 in
   different paragraphs. Live at audit time: **59** and `gen3_edge_bias_trunk_v1` (now **60** /
   `gen3_entity_rehome_v1`). It also contains a duplicated,
   partly-conflicting pair of paragraphs (two "`MODEL_CONFIG_VERSION` was **38** at v38" endings).
4. **Root `CLAUDE.md` claimed `MODEL_CONFIG_VERSION` = 57.** Live at audit time: **59** (58 = the
   speed-stat GIGO stamp, 59 = `consequence_topk`; now 60 = the re-home stamp). Neither v58 nor
   v59 was described anywhere in the root file.
5. **`src/agents/model/CLAUDE.md` describes `ObsUnpack` as peeling "the flat 3390-dim
   observation"** — obs-layout generations out of date (live is now 2669).
6. **`PointerNativeActionHead`'s docstring says the move cell is `[low,high,crit,pko,p_land,known,
   sec×10]`.** It is `sec×7` (`_PTR_MOVE_CELL` = 13, not 16) since the outgoing slp/psn/tox columns
   were dropped. `pointer_cells`' own docstring, 900 lines away, says 7 correctly.
7. **`constants.py`'s `OFFSET_*` trailing comments are stale** (`# 642`, `# 1284`, `# 1400`,
   `# 1418`, and "base dim = 1790, full obs = 3391"). The expressions are correct; only the
   comments are wrong — so anyone grepping for an offset by eye gets the wrong number.
8. **`designs/CLAUDE.md` named the active run as gen-2 `run_20260805_060807`.** The live run is
   gen-3 `run_20260807_135637_gen3`. Corrected in this pass.
9. **The speed-stat GIGO is fixed in code but its consequence is under-flagged in docs.**
   `pairwise_speed` (edge `v`) and `pairwise_boost`'s outspeed channel read stat index 4 — Special
   Defense — as "speed" through two trained generations. Fixed 2026-08-06 with named `_BS_*` /
   `_NAT_*` indices and a discriminating regression test. Consequence: **every `v`-edge and
   C1-outspeed number measured before 2026-08-06 describes the buggy feature**, including the
   gen-1 audit in §5.4.

---

## 9. Where to look next

| Question | File |
|---|---|
| **This document as a clickable digraph** — the 58 nodes / 487 edges above, hue-coded by what each channel physically carries, with a per-checkpoint measured-dependence overlay and a path filter (pick `vf_projection` to see exactly what the critic reads) | **https://model.g5d.io** (served live from the workstation checkout, so it is never a stale copy), or `designs/architecture_viewer.html` via `file://`. **Generated — never hand-edit it**: rebuild with `python -m agents.model.build_arch_viewer`, and `--check` fails if the committed artifact has drifted from the graph. |
| Obs-build performance gate (mandatory benchmark) + per-slot detail | `src/agents/observation/CLAUDE.md` |
| Phase contract, `ExtractorContext`, versioning playbook | `src/agents/model/CLAUDE.md` |
| How it got here — every version entry, verbatim | `designs/CHANGELOG.md` |
| Which `ai_vN` folder is relevant | `designs/CLAUDE.md` |
| Hypothesis status / what has been killed | `designs/research_state/ledger.md` |
| **The raw audit outputs behind every measured number here** | `designs/research_state/measurements/` (+ its README for how to read one) |
| Delivery-channel theory (edge vs content vs cell) | `designs/learning/shortcut_learning_and_feature_delivery.md` |
| Event-sourced battle layer + read-models | `src/agents/battle/CLAUDE.md` |
| Training loop, eval sharding, ELO | `src/agents/training/CLAUDE.md` |
