# ARCHITECTURE.md — what is true NOW

**Scope: current state only.** History lives in [`CHANGELOG.md`](CHANGELOG.md); versioning
*mechanics* live in `src/agents/model/CLAUDE.md`. If a claim here disagrees with the code, the code
wins and this file is a bug — fix it in the same change.

**"Current" means the production configuration as HEAD resolves it.** Two objects, deliberately
distinct — a generation now turns over every ~2 days, so conflating them is how this file went
stale twice:

| | |
|---|---|
| Production run | **`ai_v12_02_winprob_critic`** (the WIN-PROB CRITIC era, 2026-09-06) — `config_version` **109**, `arch_signature` **`gen3_critic_route_wave_v1`**. It is gen-17's architecture surface with the CRITIC swapped and nothing else: the substrate cells stay ON in the base (`pair_outcome_cell` / `pair_outcome_switch` / `switch_branch_cell` / `conditional_threat_cell`), `pair_value_route` stays OFF pending the C4 offline gate, and all 17 edge families, the entity seats, the event window and the belief stack are unchanged. The 13 rows that moved are the critic family alone — see §3.4 and §6. Its predecessor `models/ai_v9_21_gen17_pfspoff_0820/` (gen-17, v97) is what every §4/§5 measurement below was taken on |
| Code on HEAD | `MODEL_CONFIG_VERSION` / `ARCH_SIGNATURE` — **read them from `agents/model/model_version/constants.py`**, never from prose (at this writing: 109 / `gen3_critic_route_wave_v1`) |
| `designs/production_config.json` | the live run's config **carried forward to HEAD's schema** — a verbatim mirror of the production run's `model_config.json`, refreshed with `python -m agents.model.delivery_graph --sync-config <run>/model_config.json`, never hand-edited, and carrying its provenance in the sibling [`production_config.README.md`](production_config.README.md) (JSON has no comment syntax, so the record cannot live in the file). The `gen3_critic_route_wave_v1` **signature-bump window is CLOSED**: the production run records that signature, so the mirror tracks the RUN rather than the live code. (Inside such a window the two requirements pull in opposite directions — the compile gate needs the mirror to match live code, the drift gate needs it to mirror the newest run, and neither can be relaxed — so `arch_tables_test` DETECTS the window from the run's recorded signature and lets the mirror follow the code until a run at the new signature exists.) It exists so this file, the compile gate, the delivery graph and the viewer all derive from ONE real feature set |

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
rather than quietly rotting. The snapshot compares the graph against *itself*, which cannot see a
module nobody drew — so a second gate enumerates the extractor's parametered top-level modules and
fails unless each one is reachable in the graph or allowlisted with a reason
(`delivery_graph.MODULE_GRAPH_TOKENS` / `NON_DELIVERY_MODULES`, checked by
`test_every_parametered_module_is_reachable_in_the_graph` and by `--check`). Regenerate both
artifacts in the same commit:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python -m agents.model.delivery_graph \
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

One flat `float32` vector of **2501** dims, plus an 11-dim `action_mask`, delivered as a Dict obs.
Every number below comes from `agents/observation/constants.py` and
`Gen3ObservationEncoder.get_layout()`. **Never hardcode an offset — read the layout.**

### 1.1 Top-level blocks

| Block | Start | End | Dims | Constant |
|---|---|---|---|---|
| Our team — 6 × per-mon slot | 0 | 732 | 732 | `OFFSET_OUR_TEAM`, `6 × POKEMON_FULL_DIM` |
| Opp team — 6 × per-mon slot | 732 | 1464 | 732 | `OFFSET_OPP_TEAM` |
| Active context ×2 (ours, theirs) | 1464 | 1580 | 116 | `OFFSET_CONTEXT`, `2 × ACTIVE_CONTEXT_DIM` (58) |
| Global env | 1580 | 1600 | 20 | `OFFSET_GLOBAL`, `GLOBAL_ENV_DIM` |
| Board (reactive) | 1600 | 1617 | 17 | `OFFSET_REACTIVE`, `REACTIVE_DIM` |
| Pair history — 6×6×5 h[i,j] | 1617 | 1797 | 180 | `OFFSET_PAIR_HISTORY`, `PAIR_HISTORY_DIM` (`gen3_pair_history_v1`) |
| Event window — 32 × 22 event records | 1797 | 2501 | 704 | `OFFSET_EVENT_WINDOW`, `EVENT_WINDOW_DIM` (`gen3_event_window_v1`) |
| **Total** *(= `base_dim`)* | | **2501** | | `Gen3ObservationEncoder.dimension` |

The event window is the LAST block: `total_dim == base_dim`, and the encoder's output IS the
observation. There is no appended tail — `Gen3Env.embed_battle` returns `encode(...)` unchanged.

**The event window** (Tier H-B, `gen3_event_window_v1`): the last 32 decision-relevant EVENTS as
typed 22-column records — type id · actor/target species + side · move id · attributed
`hp_delta` · outcome/crit/effectiveness · `we_first` · status id · log-saturated recency ·
forced-window phase tag · valid · cant-reason id · faint-cause id · item-transition id — folded by `EpisodeTracker.EventWindowTracker` from PUBLIC
protocol events (seq-idempotent), most-recent LAST with zero-padding at the front. Ids are
embedding ids; **no Linear reads the block raw** — its only consumer is the opt-in
`history_events` event-seat encoder (§ flag table). The columns are documented at
`agents/observation/constants.py` (`EVENT_TOKEN_DIM`).

`gen3_entity_rehome_v1` (Stage 3): the two 144-dim matchup matrices and 6 of the 11 reactive
scalars are **deleted** — the D/V edge families compute a strict superset of the matchup signal
GPU-side, `active_status` was byte-redundant with the per-mon condition one-hot, and
`forced_struggle` is derivable (all-zero `active_req_moves` legal bits / the action mask).
`protect_odds`, `trapped` and `maybe_trapped` moved **onto the per-mon slots** (the facts ride
the entities they describe).

> The `OFFSET_*` trailing comments inside `constants.py` used to carry stale evaluated numbers
> (642 / 1284 / …). They are deleted (2026-08-14) — only the expressions remain, with a comment
> forbidding evaluated values there; read `get_layout()` for live offsets.

### 1.2 Per-Pokémon slot — 122 dims (`POKEMON_FULL_DIM`)

`POKEMON_VECTOR_DIM` is 119; `state_encoder` appends the two OUR-side trapping bits and then the
active flag → 122. The active flag stays the **last** dim of the slot on purpose — the model's
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
| last action `[move_id, was_switch, hit, miss, fail, crit]` (active only) | 113 | 6 | `POKEMON_LAST_ACTION_OFFSET` (`gen3_pair_history_v1` — the id is embedding-routed, its raw column zeroed at the slice) |
| trapped (our active only) | 119 | 1 | `POKEMON_TRAPPED_OFFSET`, appended by `state_encoder` |
| maybe_trapped (our active only) | 120 | 1 | `POKEMON_MAYBE_TRAPPED_OFFSET`, appended |
| active flag | 121 | 1 | `POKEMON_ACTIVE_OFFSET`, appended (LAST — load-bearing) |

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

### 1.6 What happened last turn

Carried by the **event window** (§1.1), not by lag frames. The 7 × 159 TurnDelta frames and the
11-dim prev-turn action mask are DELETED; `TurnDelta` itself survives as the reward manager's
per-decision input and as the α/β intent label source, but it no longer has an obs encoding.

Every fact the frames delivered has an event-window column — move id, outcome, crit,
effectiveness, status applied/cured, boosts, switch-ins, forced-switch phase, move order —
with one addition and one accepted loss:

- **`cant_id` (column 19) was ADDED for the deletion.** "This mon could not move, and why" (full
  paralysis / sleep / flinch / recharge) had NO event-window column: `EventKind.CANT` was in the
  battle event log with its reason and the TurnDelta fold read it, but the window emitted no row.
  It now emits `EVENT_T_CANT` with the reason as a 1-based id into `gen3_effects.CANT_REASONS`
  (0 = not a cant row). It has its own column rather than riding `status_id` — the two are
  mutually exclusive by `type_id`, so overloading would encode compactly and read wrongly.
- **`faint_cause_id` (column 20) and `item_transition` (column 21) were ADDED**
  (`gen3_event_semantics_v1`), closing the other two gaps. The FAINT row now carries WHY a mon
  died — 1..8 into `turn_view.FAINT_CAUSE_VOCAB`, via the same `_classify_faint_cause` the
  TurnDelta fold uses, so the two cannot drift. The sequence-makes-it-inferable argument only
  ever covered {attack, recoil, selfko}: weather, status, hazard and Leech Seed deaths emit NO
  preceding event, because residual damage is not an event. `item_transition` is an ENUM, not a
  consumed flag — gen3 has three item-GONE routes (consumed berries/herbs · REMOVED by Knock
  Off, permanent in ADV · SWAPPED by Trick/Thief/Covet) and one flag would leave the conflation
  half-alive.
- **`our_attempted_switch_spec` is LOST, knowingly.** When a switch is refused while trapped, the
  window records that it happened (`EVENT_T_SWITCH_REJECTED`) but not WHICH bench mon was aimed
  at. That is structural, not an omission: `Gen3Battle.record_choice_rejected` documents that the
  attempted target "is not on the wire and is recovered at fold time from the action index", and
  this window folds from events alone. Trappedness itself still reaches the model through the
  per-mon slots (`gen3_entity_rehome_v1`).

Per-slot layout of the event record, and the embedded-ID manifest that routes raw ids to
embedding tables, live in `src/agents/observation/CLAUDE.md`.

---

## 2. Feature extractor — the production chain

`Gen3FeaturesExtractor` (`src/agents/model/features_extractor.py`), paired **mandatorily** with
`Gen3DualHeadMaskablePolicy` (`policy.py`) — the extractor returns a `(pi_features, vf_features)`
tuple, which stock SB3 policies cannot consume.

Modules actually built under the production config (`named_children()`) — GENERATED:

<!-- BEGIN GENERATED: modules -->
```
embeddings · unpack · pokemon_encoder · entity_seats · edge_bias · team_transformer · cls_pool ·
hidden_opp_belief · intent_move_cell · intent_threshold_move · intent_conditional ·
pair_outcome_move · pair_outcome_switch · switch_branch · conditional_threat · t0_species_prior ·
belief_slots · belief_head · move_belief · spread_belief · hp_type_belief_head ·
item_belief_head · damage_op · prefuse_proj · assembler · win_head · value_entity_pool ·
history_events · pre_proj_norm · projection · value_pre_norm · value_projection · activation ·
alpha_head · beta_head
```

Notably **absent** (`None` on the instance): `value_dist_head`.
<!-- END GENERATED: modules -->

### 2.1 Order of operations — the TIER ORDER, and the only order

The belief + physics stack runs **once, before attention**, on every config. There is no flag: the
forward resolves the game in the order the game resolves in, and the four tiers are an **asserted
invariant** (`tier_contract.py`, `tier_contract_test.py`) rather than a property of how the code
happens to be written.

| tier | question | modules |
|---|---|---|
| **T0 RESOLVE** | what is on the board? | `pokemon_encoder`, `belief_slots`, `move_belief`, `hp_type_belief_head`, `spread_belief`, `item_belief_head` |
| **T1 REASON** | what follows from it? | `damage_op`, `entity_seats`, `edge_bias`, `team_transformer` |
| **T2 DECIDE** | what will they do, what are my moves worth? | `belief_head`, `cls_pool`, `alpha_head`, `beta_head`, `intent_threshold_move` / `intent_conditional` / `pair_outcome_move` / `pair_outcome_switch` / `switch_branch` / `conditional_threat`; `cls_pool` additionally owns the two token-content critic injections (`value_threat_proj`, and `pair_value_proj` opt-in/off) |
| **T3 DELIVER** | one contract, two pools | `hidden_opp_belief`, `assembler`, `win_head`, `value_dist_head` |

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

1. **`ObsUnpack`** — slices the 2501-dim vector into `ExtractorContext` (~30 named tensors:
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
   *(Also live: **`ItemBelief`** — `gen3_item_belief_v1`, `--item-belief` — resolves
   each opp slot's hidden item as a posterior over item nums, Smogon usage prior ⊕ zero-init trunk
   delta; cold start == prior. Published leak-mode-aware (`last_item_logits`); the op's p_cb
   unrevealed branch consumes P(Choice Band) from the publication instead of the static
   `SPECIES_CB_PRIOR` scalar — the revealed 0/1 exactness gate is unchanged. Supervised as the
   BeliefBank's seventh row.)*
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
| `NET_ARCH` (SB3 mlp_extractor) | `[512, 512]` | " |

Embedding tables (`Embeddings`, registered exactly once, passed as a forward argument):
species 400×32, move 400×16, item 600×16, ability 100×16, type 20×16.

### 2.3 The 29-token sequence

| Seats | Index range | Token type | Content |
|---|---|---|---|
| our mons | 0–5 | `TOKEN_TYPE_OUR_TEAM` | role token (+ `prefuse_proj` incoming residual) |
| opp mons | 6–11 | `TOKEN_TYPE_THEIR_TEAM` | role token (+ move-belief reinjection) |
| global | 12 | `TOKEN_TYPE_GLOBAL` | `[our_ctx, opp_ctx, non_matchup_rest]` → `global_proj` |
| **E3** our active's moves | 13–16 | `TOKEN_TYPE_OUR_MOVE` | move token in **request order**, `move_seat_proj` 32→128 |
| **E4** opp threat moves | 17–22 | `TOKEN_TYPE_THEIR_THREAT` | `threat_seat_proj([latent(32), w, acc, is_phys])`, K = `entity_topk_seats` = 6 |
| **E5** tail threats | 23–28 | `TOKEN_TYPE_THEIR_THREAT` + `tail_marker` | per-opp-mon beyond-top-K residual `tail_proj([p_tail, worst_phys, worst_spec, revealed])` |

There are **no `TOKEN_TYPE_HISTORY` seats in the base sequence** — the seven of them went with the
lag frames (`gen3_frame_deletion_v1`), which is what took the sequence from 36 tokens to 29 and
shifted every extra seat down by seven. `TOKEN_TYPE_HISTORY` itself survives in the token-type
table and is what an opt-in event seat takes.

`entity_seats.n_seats` = 16 (4 + 6 + 6). Base seat count = `2·TEAM_SIZE + 1` = 13 (the
`N_HISTORY_TURNS` history seats went with the lag frames), so **every extra seat index is
`13 + offset`** — that is what makes the base slices position-stable.
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
| `hidden_opp_belief` | 768 | `HiddenOppBeliefPool` — k=6 × `D_MODEL` (POLICY only — the vf half read dV 0.0000 and was deleted) |
| **total** | **1177** | == `projection.in_features`, asserted at generation |

**`vf_projection` — `Linear(128, 512)`** (LayerNorm → Linear → ReLU). Input concat, in order:

| Part | Dims | Source |
|---|---|---|
| `value_pooled` | 128 | `CLSPool.value_cls` over **all 12** team tokens |
| **total** | **128** | == `value_projection.in_features`, asserted at generation |
<!-- END GENERATED: head-inputs -->

**Every value route INJECTS into `value_pooled`** (v89 `gen3_value_pooled_routes_v1`): the
routes below add a zero-init `D_MODEL` contribution to the tensor the dist-head critic actually
reads (and `vf_parts[0]`, so the scalar critic sees the same wiring). The old post-assembler
vf-concat delivery was structurally bypassed by `--value-from-dist` — gen-12 proof:
`value_entity_pool.out_proj` and the then-live α-reduce projection bit-exact ZERO after 25M
steps, while `value_threat_proj` (the one `value_pooled` route) trained to 0.117. The
gradient-connectivity guard (`value_route_gradient_test.py`) backprops the critic through every
registered route each suite run, under BOTH critic parameterizations.

**The seam has ONE member** (v96 `gen3_critic_route_wave_v1`). Four of its five original routes
were deleted on measured dependence against a 0.39 dV bar: `intent_value_reduce` 0.3176 and
`value_clock` 0.2169 (both re-audited at 2× sample first), `intent_threshold`'s p_KO vf half
0.155/0.136, and `value_intent` 0.156. `value_intent`'s **re-entry condition survives its
deletion**: any future α/β-to-critic proposal passes the C4-style offline gate FIRST (ledger C6 —
the delivery line is EXHAUSTED). The seam itself is kept generic at one entry because its value is
covering the NEXT route on the day it is written.

**ON in production: `value_entity_pool`** (v80, `UnifiedValueReadout` — Stage-3 T3-DELIVER of
`design_unified_belief.md` §3). ONE attention pool over the critic's entity-row set (the 12
post-transformer team tokens + the op's 6 per-our-mon incoming rows, each projected to
`UVR_DIM`=64 with a per-source type embedding, `UVR_K`=4 queries, explicit NaN-safe softmax)
adds its zero-init `D_MODEL` output into `value_pooled` — the policy is untouched at
any weight. It was the designed SUCCESSOR contract of the bolt-on vf routes, it ran ALONGSIDE
them for two generations so the `critic_route_audit` could price them against each other on one
trained run, and **it WON**: gen-14 read it at dV **5.490 = 97% of `all_off` 5.635**, against
`threat` 1.0686 and every other route below 0.32. The succession is complete — the seed readout,
the `nmr` vf concat and the hidden-opp vf half are all deleted at v96, and this pool plus
`--value-threat-inject` are what the critic reads. **`value_entity_pool_full`** (v82, **ON**)
completes the row set — +the refined GLOBAL token and +the hidden-opp belief queries — which is
where the content of the two deleted concats now lives (its own flag/shape: a v80-shape pool
keeps loading under `full=False`).

**ON in production: `intent_threshold`** (v84, `gen3_intent_threshold_v1` — the §3.0
threshold operator of `design_conditional_execution.md`). `threshold_probs` contracts the op's
per-candidate pair cells with the published α into `p_KO` / `p_sub_broken` / `p_fp_broken`, and
ONE zero-init projection delivers five mechanic channels (Focus Punch / Substitute / Endure /
Destiny Bond / Endeavor) plus `p_KO` as per-slot context through the pointer MOVE cell
(+`INTENT_THRESH_MOVE_DIM`). ⚠️ **v84 also built a SECOND consumer — the `[p_KO, …]` vf route,
the ledger-H1 payoff — and v96 DELETED it** (dV 0.155 on gen-13, 0.136 on gen-14, against a 0.39
bar). What H1 asked for stands; this delivery of it did not, and any successor owes the C4-style
offline gate first. The POLICY half is a per-action pointer channel measured in KL/flips rather
than |dV|, was never part of that verdict, and is pinned by `intent_threshold_test.py`.
**`intent_conditional`** (v85) is its sibling: the Counter/Mirror-Coat category sums,
flinch's `(1−α_SWITCH)` term, Explosion's execute/into-switch facts and Pursuit's ×2 never-miss
doubling trigger (the port-verified departing-target rule — no β), one more zero-init block on
the move cell (+`INTENT_COND_MOVE_DIM`). Enabling either is a gen-13+ decision gated on
gen-12's `intent_move_cell` audit (the G3 verdict); the pre-build G2 usage baseline is
`measurements/gen12_mechanic_usage_baseline.json` (Endure 0.0% / Sub 0.9% / Counter 5.6%).

**LIVE (ON in the gen-17 base): `pair_outcome_cell`** (v93, `gen3_pair_outcome_v1` — component 1 + 3 of
`design_opponent_intent.md`, §2.1/§9a of `design_pair_reduction.md`). It closes a **currency**
failure, not a reduction failure: incoming status reaches the policy only through the `s3` edge
family, i.e. as a softmax-normalised RATIO, so "35% of my HP" and "80% chance of burn" never meet
in one vector and no reducer can trade them. When on, the op builds ONE **`pair_in[their believed
seat k, our mon j, :]`** of width `_PAIR_OUTCOME_RAW` = **14** — the six existing damage channels
(`[low, high, crit, ko_ramp, acc, is_phys]`) concatenated with eight new ones:

| # | coordinate | source |
|---|---|---|
| 6-11 | `p_par p_brn p_frz p_slp p_psn p_tox` | `_incoming_status_lands` (the per-pivot immunity physics, unchanged) SPLIT by the seat's status IDENTITY — `MOVE_STATUS_IDENT` for a dedicated status move (read from the raw `status_inflicted`, so **tox and psn stay apart** where `MOVE_STATUS_CAT` folds them), `MOVE_SECONDARY`'s L1-normalised major prefix for a damaging move's secondary |
| 12 | `neutralization` | fraction of this mon's per-turn contribution destroyed WITHOUT a KO: burn → `0.5·base_atk/(base_atk+base_spa)`, paralysis → `0.25 + 0.75·Δp_outspeed` (the op's OWN outspeed logistic re-evaluated at ×0.25 speed), freeze/sleep → 1.0, psn/tox → the 1/8 and 1/16 residual ticks. Every scalar is a gen3 RULE; no tuned prior |
| 13 | `tempo_cost` | `P(any major status) × undo_turns(j)`, where `undo_turns` is the **CHEAPEST available undo path**: 1 turn for a cure MOVE (`MOVE_CURES_SELF_STATUS`: Refresh / Heal Bell / Aromatherapy), **1 for the Natural Cure ABILITY** (the status is shed on switch-out and a switch consumes exactly one of our actions), the op's own `rest_sleep_noeb` (**2**) for Rest, **2 for the bench-CLERIC path** (switch to an ALIVE teammate carrying a party-wide Heal Bell / Aromatherapy, then click it), else **0**. `0` means *no path exists* — never *the path is free* — which is why Natural Cure is priced at its literal switch. Every input is OUR mon's (moveset, ability, HP), so all of it is exact and no marginalisation arises on this axis. `neutralization` deliberately does NOT read the ability: it is a per-TURN rate and Natural Cure changes DURATION, which this reduction refuses to model without a rule to source a number from |

ONE α over the move axis then reduces it — **Contract W**: α has no defender axis and no channel
axis, so the flat block's nine-independent-maxima incoherence (D2) and a per-defender α (D3) are
**shape errors** here rather than properties under test. α is the softmax of the PUBLISHED α
logits, **move slice only, unrenormalized and stop-grad** (a high `α_SWITCH` correctly shrinks
every coordinate toward zero; the detach is unconditional, so no PPO→`alpha_head` route depends on
a training flag). **With `--opp-intent` OFF it falls back to the shipped R1 `belief_mean` rung
(α := w/Σw)**, which is what makes the flag independently enableable — the DELIVERY claim is
testable apart from the DISTRIBUTION claim. A seat closed by the meaningful-K gate is MASKED, its
mass not reassigned. The reduced row for our ACTIVE defender rides every move cell as decorrelated
context through a zero-init `Linear(14, 14)`.

Known limits of the coordinate table, named rather than approximated: status DURATION (which is
also why the Natural Cure ability rides `tempo_cost` and not `neutralization`), physics mutation
(Marvel Scale), and a held berry's auto-cure.

**LIVE (ON in the gen-17 base): `pair_outcome_switch`** (v94, `gen3_pair_outcome_switch_v1`). The same
reduction, at **every** defender (`Σ_k α_k · pair_in[k, j, :]`), delivered to mon *j*'s own pointer
**SWITCH** cell through a second zero-init projection. This is the sink `design_pair_reduction.md`
§2.1 traced the defect to: the switch cell carries ten damage numbers, one speed number, two
belief-mass numbers and **no status coordinate in any currency**, so *"they will click Will-O-Wisp,
so bring the Natural Cure mon"* is unrepresentable there. It is the **first module to widen the
switch cell**. One α still serves all six rows — D3 (a per-defender α) stays a shape error — and
the module is equivariant in our team axis by construction. One extra per-defender coordinate rides
with the row, `spin_denied` = `is_ghost(our mon j) · Σ_k α_k·is_rapidspin(k) · their_side_hazards`
(the defensive half of the Pursuit mirror: a gen-3 Rapid Spin fails outright against a Ghost, so a
Ghost switch-in is hazard insurance; the stake is what makes it a value rather than a fact).
Requires `damage_op`, **not** `pair_outcome_cell` — the two deliver one tensor to two sinks and
coupling them would make a measured result unattributable. `PAIR_OUTCOME_SWITCH_DIM` = **15**.

**LIVE (ON in the gen-17 base): `conditional_threat_cell`** (v95, `gen3_conditional_threat_v1` —
`design_conditional_opponent_cells.md` §1's **OA1**, the defensive pivot). The **second** module to
widen the pointer SWITCH cell, and it carries exactly the quantities the α-reduced outcome row
structurally cannot. Four coordinates, all `Σ_k α_k · f(k, j)` against the same one α:

| # | coordinate | meaning |
|---|---|---|
| 0 | `e_pko_acc` | `Σ_k α_k · ko_ramp(k,j) · acc(k)` — §0.2(2)'s rule (*precompute every nonlinearity of two numbers IN THE OP*). `ko_ramp` and `acc` ride the reduced row DECORRELATED and a thin `tanh` scorer does not multiply two of its own inputs; two of our mons can be identical in `Σα·ko_ramp` AND in `Σα·acc` while their true P(dies) differ |
| 1 | `e_type_mult` | `Σ_k α_k · type_mult(k,j)` — the one cell channel NOT divided by the defender's own bulk, so a structural immunity (`0.0`) reads apart from an incidental zero and the read survives the mon's own HP moving |
| 2-3 | `margin_high` `margin_crit` | `Σ_k α_k · high(k,j) − hp_frac(j)` and the same on the crit roll (§0.2(3): *probabilities SATURATE; ship the MARGIN too*; `> 0` ⇒ dead). They separate two mons a saturated `pko` cannot — at the bottom (*both survive; by how much?*) and at the top (*both die; does a low roll save one?*) — and the crit margin is the *safe pivot vs coinflip pivot* distinction |

Three of §1.2's five clauses are **superseded and were substituted rather than built** (the table is
in `conditional_threat.py`): its `λ`-weighted `w = softmax(λ·threat + log belief)` is NOT built —
`pair_alpha` is the shipped distribution and a second one would be a second α; `high` / `pko` /
`status_lands` are already delivered by `pair_outcome_switch`, and `status_lands = Σ_s p_s` is
additionally barred by §9a's derivability rule; and §1.3's *"also turn on
`--damage-matrices-outgoing-all`"* is **VOID**, that flag having been deleted at v88. The op
stashes the per-(defender, seat) `type_mult` at α's own seat alignment behind a new seam
(`stash_pair_type_mult`) rather than letting a consumer re-derive it — the `op move-order` bug class
with extra steps. Requires `damage_op` + `damage_matrices_incoming`, **not** `opp_intent` (the R1
`belief_mean` fallback is MEANINGFUL here: every coordinate is a *what lands on me if they attack*
contraction, so the missing SWITCH mass correctly shrinks it) and **not** `pair_outcome_switch` (two
quantities, one sink, attributable separately). `CONDITIONAL_THREAT_SWITCH_DIM` = **4**.

**LIVE (ON in the gen-17 base): `switch_branch_cell`** (v94, `gen3_switch_branch_v1` —
`design_conditional_opponent_cells.md` §2's OA2, plus two owner-specified mechanics of the same
shape). Everything in it is `Σ over their options of (usage probability) × (a property of the
option)`, contracted over the branch in which they **switch**. Gen-3 is simultaneous-move, so
`P(they switch)` is ONE scalar for the turn (§2.1); the CONSEQUENCE is per-move, because switches
resolve first and our move lands on the arrival, which **β** names. Nine coordinates on the move
cell:

| # | coordinate | meaning |
|---|---|---|
| 0-2 | `e_high_switch` `e_pko_switch` `e_mult_switch` | `Σ_j β_j · omx[k, j, ·]` — the SWITCH branch of our own move, from the outgoing matrix. §2.3's rule is followed literally: the branches ship DECORRELATED (the stay branch already rides the op's move cell), never the collapsed `(1−p)·stay + p·switch` |
| 3 | `wasted_ko` | `pko_stay(k) · α_SWITCH` — §2.3's named interaction, *"don't click the KO into the obvious switch"* |
| 4 | `a_switch` | the ONE per-turn switch scalar, broadcast over all four slots |
| 5-6 | `p_spin_blocked` `spin_value_lost` | `is_ghost(their active)·a_stay + α_SWITCH·Σ_j β_j·P(slot j is Ghost)`, gated to the Rapid Spin request slot, and that probability × our-side hazards. **The Pursuit mirror**: v85's Pursuit is `α_SWITCH` against a property of the DEPARTING mon with positive valence and no β (the sim strikes before the switch resolves); this is `α_SWITCH` through β against a property of the ARRIVING mon with negative valence (Rapid Spin resolves after). `P(slot is Ghost)` is leak-free — revealed types where revealed, the hidden-team species posterior through `SPECIES_IS_GHOST` where not |
| 7-8 | `protect_attack_mass` `protect_blocked_mass` | `Σ_k α_k · is_damaging(k)` gated to Protect/Detect, and that × the obs `p_success` decay. The `c4` successor: Protect's cell carried the mechanical decay and never asked *will they attack*. Decorrelated from v85's `e_dmg_avoided`, which is a MAGNITUDE where this is a MASS — they come apart in both directions (a believed Spore has mass and no magnitude; a 4×-resisted Hidden Power the reverse). `is_damaging` is typed from the data facade, so an immune damaging move cannot masquerade as a status move |

α and β are read from the PUBLICATIONS and **stop-grad unconditionally**. This flag **requires
`opp_intent` with no fallback**, and the asymmetry with the pair-outcome pair is substantive: the
R1 `belief_mean` rung is a PRESENCE belief over their MOVES and carries no switch class, so
`α_SWITCH` would be identically 0 and every coordinate would assert *"they never switch"* — a
claim, not an absence. §4.1's hard prerequisite for OA2 is **CLOSED**
(`gen3_unrevealed_outgoing_prior_v1` prices an unrevealed arrival against the expected-latent
defender); the one residue, stated rather than hidden, is that **`pko` stays NULLED at unrevealed
slots** by the op's owner rule, so `e_pko_switch` is deflated in proportion to β's hidden mass
while `e_high_switch` carries the magnitude there. `SWITCH_BRANCH_MOVE_DIM` = **9**. Not modelled:
Rapid Spin also clears Leech Seed and partial-trap from its user, and a Ghost KO'd on the switch-in
denies nothing.

Route availability is **width-neutral by construction** (additive injection changes no
projection width), so the old ede5a88 discovery-sizing bug class — a fall-through branch hiding
a vf part from the forward that sized `value_pre_norm` — is unrepresentable. Both projection
input widths are **static arithmetic** (`compute_projection_widths`, `gen3_static_widths_v1`;
the construction-time discovery forward is deleted): pi = 3·D_MODEL + the `non_matchup_rest`
scalar tail + k·D_MODEL (hidden-opp belief pool, `opp_belief_cls_k`, POLICY side only);
**vf = D_MODEL, a constant no flag can move**. `projection_width_test.py` verifies the arithmetic
against a real forward per flag combo. Any NEW value route goes through `_value_pooled_routes`
(the registry the gradient guard iterates) — never a new vf-concat part; there is no vf concat.

**`vf_combined` IS `value_pooled`** (v96 `gen3_critic_route_wave_v1`). The value head reads one
tensor and nothing else: not `our_active_refined` (the `value_active_readout` toggle was deleted
at v88), not either team pool, not the `non_matchup_rest` scalar tail, not the hidden-opp belief.
The whole post-assembler vf tail was retired on measurement — its three members read dV **0.0000**
on gen-14 at n=12,391 (`nmr` and `hidden_opp_vf`) and **0.0000 bit-exact on two consecutive
audits** (the seed window). Since `--value-from-dist`'s critic reads `value_pooled` directly, this
also makes the v89/M2 orphaned-vf-branch class **unrepresentable** rather than merely fixed: both
critic parameterizations now read the same tensor, and there is no second path for either to
bypass. Every critic enrichment is an additive injection into `value_pooled` (the v89 seam) or a
token-content injection on the value pool's local copy inside `CLSPool`.

**The v61 multi-seed window is DELETED with the rest of that tail.** It was k=4 learned queries
cross-attending over the op's per-our-mon incoming rows — MULTIPLICITY rather than width (ledger
P3 refuted width only) — and it carried the `value_seeds/*` collapse contract, which went with it.
**The trigger it existed to catch FIRED on gen-5** (`out_effective_rank` 1.0 sustained
196k→15.7M steps — the k=4 outputs identical), and the two pressures then applied to it are the
reason the line closed:

**Both pressures are deleted (v78). The measurement is why.**
`--value-seed-vicreg-coef` (v62) was the repulsive one — a scale-relative
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
either term reaches it — so both flags and both modules were deleted rather than retuned, and the
READOUT itself followed at v96 once two end-of-run audits read its dependence at 0.0000.

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

**Available but OFF: `pair_value_route`** (v95, `gen3_pair_value_route_v1` —
`design_opponent_intent.md` §7a(2)'s **PV**). The SAME token-content mechanism carrying a DIFFERENT
object: Phase A's **unified** `pair_in` row (`PAIR_VALUE_ROUTE_DIM` = `_PAIR_OUTCOME_RAW` = **14**),
whose last eight coordinates are the six status identities, `neutralization` and `tempo_cost`. This
is the first per-entity route by which the CRITIC reads that currency at all — incoming status
otherwise reaches vf only as the `s3` edge family's softmax-normalised **RATIO**
(`design_pair_reduction.md` §2.1). It is a SECOND zero-init `Linear(14, 128)` on the same local copy
inside `CLSPool`, so the two injections stack additively and independently and vf-only holds at any
weight for both.

**It is NOT in the `_value_pooled_routes` seam, on structure rather than taste.** A seam route
yields one `[B, D_MODEL]` vector added AFTER pooling, so it would have to collapse the `J` axis
itself — and the only equivariant collapse is a sum, which cannot tell *one mon about to lose 90% of
its bar* from *six mons losing 15% each*. Token content does not collapse: the row rides the token
that also carries the mon's identity, HP and typing, and `value_cls`'s attention decides the
weighting (§2b.2 — *you can only preserve an axis you have output slots for*; here the tokens ARE
the slots). Cost of that choice: the seam's gradient guard does not cover it by construction, so
`value_route_gradient_test.py` was extended with a dedicated cell for **both** token-content
injections under both critic parameterizations.

⚠️ **α here is the R1 `belief_mean` rung UNCONDITIONALLY — ORDERING, not preference.** `value_cls`
pools at T2 *before* the α/β heads are scored, so the publication does not exist yet; it is not a
fallback that fires when a head is absent, and the gate asserts the injected rows are byte-identical
with `--opp-intent` ON. §7a(2) pre-registers exactly this substitution as the way to test the
DELIVERY claim apart from the DISTRIBUTION claim.

⚠️ **The C4 RE-ENTRY CONDITION governs ENABLING it, not building it**: *any α/β-critic route may be
BUILT opt-in but its ENABLING owes the C4-style offline gate first.* Ledger **C6** failed
2026-08-17 with route liveness PROVEN — all five v89 routes trained off zero and `entity_pool`
carried decisively (dV 6.28 = 110% of all-off), yet the critic's stall-loss over-confidence did not
move (gen-13 confident-band gap +0.358, CI [0.23, 0.50]) — and the delivery line was declared
EXHAUSTED — and the critic-route deletion wave then executed the four route deletions that verdict
licensed. Requires `damage_op`. Width-neutral (additive), so the version gate is the ONLY thing
that rejects a mismatched resume. `critic_route_audit` carries a **`pair_value` arm** (and includes
it in `all_off`), so that gate is runnable the moment a checkpoint carries the route.

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
| **switch j** | our-team token *j* (`our_team_out[:, j]`, `[B,6,128]`) — the same post-transformer token the CLS pools read | the incoming per-defender row (12) + `[phys_high_cb_j, pko_cb_j, p_cb]` | **15** (`_PTR_SWITCH_CELL_IN`), +15 under `pair_outcome_switch`, +4 under `conditional_threat_cell` |
| **struggle** | none — context only | none | 0 |

The move cell WIDENS under the opt-in α cells, each appending its own zero-init block:
`intent_move_cell` (+`INTENT_MOVE_CELL_DIM`), `intent_threshold` (+`INTENT_THRESH_MOVE_DIM`),
`intent_conditional` (+`INTENT_COND_MOVE_DIM`), `pair_outcome_cell`
(+`PAIR_OUTCOME_MOVE_DIM` = 14) and `switch_branch_cell` (+`SWITCH_BRANCH_MOVE_DIM` = 9).
`pointer_move_cell_dim` is the single sum the policy sizes the move scorer's `in_features` from —
a missing block narrows the `Linear` rather than silently feeding it zeros at a learned weight.
The **switch** cell likewise widens under `pair_outcome_switch`
(+`PAIR_OUTCOME_SWITCH_DIM` = 15) and `conditional_threat_cell`
(+`CONDITIONAL_THREAT_SWITCH_DIM` = 4), summed by `pointer_switch_cell_dim` and appended in that
order; until v94 nothing widened it at all.

Scoring: `tanh(proj(token ⊕ cells) + ctx_proj(latent_pi))` → a zero-init `Linear(64, 1)`.
Move logits are multiplied by `move_valid`, so an unresolved request slot contributes **exactly 0**
rather than a score computed from a zero token.

**What the switch logit does NOT see** (with every v94 flag off): a per-candidate **offense** read,
and — the defect `design_pair_reduction.md` §2.1 names — any **status** coordinate in any currency
(`pair_outcome_switch` closes the second one; `conditional_threat_cell` adds the conditional-threat
coordinates that row cannot carry). The OAX attacker row
(`damage_matrices_outgoing_all`) was deleted with its flag (v88 `gen3_dead_flag_purge_v1` — never
enabled in a gen-8+ run), so the flags-off switch cell is 15 dims and its physics is purely defensive
(what this mon takes on the switch-in) plus whatever the trunk carried into `our_team_out`. **In the
gen-17 production config both flags are ON, so the switch cell is 15 + 15 + 4 = 34** and both gaps
above are closed. The
`d2` edge family (§5) — whose engine is the same `_outgoing_attacker_matrix` kernel — is the route
by which a bench mon's offense reaches its own token.

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
entity pool's injection into `value_pooled` (dV 5.490 — 97% of the whole critic route joint),
`--value-threat-inject`'s token content on the value pool's copy (1.0686), and, when on, PV's
unified-row sibling beside it — all vf-only, all reading the op through `OpTensors` views rather
than flat offsets.

### 3.4 Side readouts

A side readout hangs off `value_pooled` AFTER the pools and stashes its logits; none of them ever
enters `pi` or `vf`, so none changes a projection width. **One is built in this config, and it is
the critic:**

| head | flag | grad flow |
|---|---|---|
| `win_head` | `win_prob_mode` **`shaping`**, `win_prob_coef` **1.0** | live `value_pooled` — the win objective also shapes the trunk (`read_only` would stop-grad it) |

#### WHICH readout is the critic — `--critic {shaped,winprob}`

`policy._critic_value` has a MODE, and the production config is on the second of two. **This is
the ONLY axis on which this generation differs from gen-17** — the trunk, the seats, the edge
families, the pointer cells and the belief stack are that run's, unchanged.

| `--critic` | `V(s)` is | trained by | reward stream | PopArt | `gamma` |
|---|---|---|---|---|---|
| `shaped` (the argparse default) | `value_net`, or the distributional `E[Z]` under `value_from_dist` | the MSE / HL-Gauss CE at `vf_coef`, in PopArt-normalized units | 1 TERMINAL + 7 PBRS + 1 BIAS, ±30 with a −35 timeout | on | 0.9999 |
| **`winprob`** (**this config**) | `sigmoid(win_head logit)` ∈ **[0, 1]** | the win-prob head's **BCE against the terminal WIN INDICATOR**, at `vf_coef` **0.5** | the TERMINAL **WIN INDICATOR** alone — `+victory_value` (**1.0**) on a win, `0.0` on a loss, a tie and a 250-turn timeout alike | **absent** (`use_popart` false, refused here) | **1.0** |

The critic and the return are the same quantity by construction: at `--victory-value 1.0` the
undiscounted return from any state is exactly `1{win}`, so **`V(s) = P(win | s)` with no
approximation term**. `value_net` is in no loss graph (its scalar term is dropped exactly as it is
under `value_from_dist`), the BCE joins the **`value`** noise-scale group rather than `aux`, and
`--win-prob-coef` is refused as a separate weight — one critic, one coefficient, so the 1.0 in the
table above is the `_resolve` default of a flag this mode does not let you type.

**`value_dist_head` is not built here** (`value_dist_mode` `none`, `value_dist_bins` 0), and
`--value-dist-mode` is refused under this critic — the A2 consumer census found ~15 sites gating on
the mode STRING rather than on `value_dist_head is None`, so a config that left the string set while
skipping the build would report a distributional loss that was never computed. `value_dist_coef`
stays recorded at 1.0 and §6 marks it `INERT — no value_dist_head`. **Both critic-side enrichment
routes SURVIVE the swap** (`value_entity_pool` / `value_entity_pool_full` / `value_threat_inject`,
all still true): they inject additively into `value_pooled`, which is exactly what the win head
reads, so they enrich a probability critic the same way they enriched a scalar one.

Under `--critic shaped` the mode builds the other route instead: `value_dist_head` (a categorical
head over `value_dist_bins` atoms spanning `[value_dist_vmin, value_dist_vmax]`, trained by an
HL-Gauss cross-entropy at `vf_coef`), which becomes the critic itself whenever `value_from_dist` is
true, with `value_net` frozen as its fallback; PopArt normalizes the value targets (and forces
`--clip-range-vf none`); the reward is the 1 TERMINAL + 7 PBRS + 1 BIAS composition at γ 0.9999;
and `win_head` demotes to an auxiliary readout weighted by `--win-prob-coef`. That mode is
STRUCTURAL — it selects a different set of heads to carry the value — so `critic` is recorded in
`model_config.json` and string-compared by `check_compatible`. It carries **no `ARCH_SIGNATURE`
bump**: a flipped mode produces no shape error anywhere (both routes return `[B,1]`), which is
exactly why the recorded-and-compared field is the whole safety.

⚠️ **A critic bounded in [0,1] cannot represent "a timeout is worse than a loss."** The `−35 < −30`
ordering `--draw-penalty` exists to set is not merely unused here, it is unrepresentable — so
`--draw-penalty` is REFUSED at any non-zero value, and the anti-stall pressure comes from the obs
deadline clock (§1.4) plus `--arm-no-progress-tax`, which re-arms `no_progress_tax` alone under
`--no-hand-shaping` without reviving the other 24 BIAS terms. **Stall rate and mean episode length
are PRIMARY endpoints, not monitored ones.** The 250-turn cap, forfeits and ties are TERMINAL under
this mode rather than SB3 truncations — as truncations at γ = 1 the bootstrapped `γ·V(s_last)` made
every timeout's TD error identically zero, so the critic could not see them at all.

Three flags are IMPLIED by `--critic winprob` (`--win-prob-mode shaping`, `--gamma 1.0`,
`--no-use-popart`) because their argparse default is the `None` sentinel, so "unset" is
representable and an implication can never overwrite a typed value. Four are REQUIRED and named by
their own refusal (`--no-hand-shaping`, `--terminal-indicator`, `--victory-value 1.0`,
`--draw-penalty 0`) because theirs are concrete, so an implication could not be told apart from an
overwrite. `resolve_critic_mode` runs BEFORE the resume-inheritance sweep, so a fork of a `shaped`
parent cannot inherit that parent's `use_popart` / `win_prob_mode` and break the mode with a value
nobody typed. Design of record:
[`designs/ai_v12/design_winprob_only_critic.md`](ai_v12/design_winprob_only_critic.md).

The `--win-prob-pbrs-*` family is **refused under this critic, not deleted**: with `V ≡ φ`,
`coef·(γφ(s′) − φ(s))` IS the TD residual GAE already turns into the advantage, so the SELF-φ route
would add the advantage to the reward and take the advantage of that. The FROZEN-φ route
(`--win-prob-pbrs-frozen <run|zip>`, a boolean by presence — φ is already in the value currency, so
the coefficient is exactly 1.0 and never a knob) is DEFERRED rather than judged wrong, and
`agents/training/winprob_pbrs.py` is intact.

**One more readout EXISTS in the code and is OFF here — `q_winprob_mode` (`QWinProbHead`).** It is
the only member of this family that does not hang off `value_pooled` alone: it scores each of the
eleven actions from the token of the entity that action selects — the SAME per-action tokens the
pointer head scores (`stash.pointer_inputs`) — with `value_pooled` as its context, and stashes
`last_q_winprob_logits [B, 11]`. One forward, eleven `P(win | s, a)`; the point is to amortize the
eleven simulator re-rolls a per-action win probability otherwise costs (ledger 229e9f1 / 5edbd05).
It is also the only readout here with no `shaping` value: every input is detached inside the
forward, so `pi`/`vf` are bit-identical whenever it is built and `grad/q_winprob_share` is 0 by
construction. **LATENT — not enabled in any run.** The head is a state_dict delta gated by
`check_compatible`, and its two training coefficients (`--q-winprob-coef`, the per-action
counterfactual likelihood; `--q-winprob-onpolicy-coef`, the weak and biased taken-action fallback)
default to 0, so nothing about it is live until a run turns it on. §6's table carries it as
`q_winprob_mode` `"none"` / OFF.

Belief heads run under `belief_grad_mode` **`shaping`** (production mirror `belief_grad_mode:
"shaping"`; §6's table carries it ACTIVE): all four routes are live — the label loss trains the
head AND shapes the shared trunk through the head's read, and the PPO loss reaches the head's
parameters through the reinject write. The opponent-INTENT head is the exception and runs
**`detached`** (`opp_intent_grad_mode: "detached"`): its trunk read is stop-grad, so the intent
labels cannot reshape the trunk. `label_only` (publish stop-grad; labels alone train the head) is
a built, non-default mode — see `src/agents/model/CLAUDE.md` for the four-route table.

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
— **`models/run_20260807_135637_gen3/checkpoints/checkpoint_9600000_steps.zip`, 6000 real eval
states, 2026-08-07.** ⚠️ That is gen-3, an EARLIER generation — not the production run. The
architecture surface it measured is the same family, but the numbers are a fact about that model.
Method: zero
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
| `outgoing_attacker_matrix` | 21.4% | ❌ **no** — the OAX flat block is deleted (v88); the kernel survives as `d2`'s engine |
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
| **h** | our mon *i* × opp mon *j* | 5 | `[switch_ins, attacks, status_clicks, shared_field_turns, pairing_recency]` — obs-fed pair-history TENDENCIES (`gen3_pair_history_v1`; EpisodeTracker-folded, log-saturated; **IN the production families string** since gen-12 — the one family whose cell the GPU cannot recompute, since it IS compiled battle history) |
| **r** | event seat *e* (the LAST-N tokens) × mon *m* (all 12) | 2 | `[is_actor, is_target]` — STRUCTURAL reference edges (`gen3_event_ref_edges_v1`, Tier H-C): event *e*'s recorded actor/target IS mon *m* (species-num equality, side-gated against mirror false-links; `_event_reference_cells`, pure). **Not in the production string** — requires `--history-events` (the seats are the rows) |
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
| r | `history_events` (the bias rows *are* the H-B event seats) |

All are satisfied in the production config, whose string carries every family including `h` and `r`
(`r`'s requirement holds because `history_events` is ON).

### 5.3 What an edge can and cannot carry

Attention weights are softmax-**normalised**, so an edge bias moves *who attends to whom*: what it
writes is a **ratio within its row**, not an absolute magnitude ("53% of max HP"). The two channels
that can carry an absolute are **token content** (`prefuse_proj`) and **per-action pointer cells**
(§3.3). This is a capacity/conditioning argument, not an impossibility proof; the reasoning is in
`designs/learning/shortcut_learning_and_feature_delivery.md`.

### 5.4 Edge-family audit — an EARLIER generation's measurement

Source: [`research_state/measurements/gen3_edge_family_audit_9p6M.json`](research_state/measurements/gen3_edge_family_audit_9p6M.json)
— **`models/run_20260807_135637_gen3/checkpoints/checkpoint_9600000_steps.zip`, 6000 eval-trace
states, 2026-08-07** (gen-3, NOT the production run), produced by
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
| `belief_grad_mode` | `"shaping"` | ACTIVE |
| `cf_evidential` | `false` | OFF |
| `cf_shadow_critic` | `false` | OFF |
| `cf_twin_heads` | `false` | OFF |
| `conditional_threat_cell` | `true` | ACTIVE |
| `consequence_topk` | `6` | ACTIVE |
| `damage_candidate_k` | `0` | OFF |
| `damage_matrices_incoming` | `true` | ACTIVE |
| `damage_matrices_outgoing` | `true` | ACTIVE |
| `damage_op` | `true` | ACTIVE |
| `damage_outgoing` | `true` | ACTIVE |
| `damage_topk_k` | `6` | ACTIVE |
| `edge_bias_families` | `"d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r"` | ACTIVE |
| `entity_tail_seats` | `true` | ACTIVE |
| `entity_topk_seats` | `6` | ACTIVE |
| `history_events` | `true` | ACTIVE |
| `hp_belief_mode` | `"composed"` | ACTIVE |
| `intent_conditional` | `true` | ACTIVE |
| `intent_move_cell` | `true` | ACTIVE |
| `intent_threshold` | `true` | ACTIVE |
| `item_belief` | `true` | ACTIVE |
| `move_belief_mode` | `"both"` | ACTIVE |
| `move_candidate_floor` | `0.02` | ACTIVE |
| `move_latent` | `true` | ACTIVE |
| `move_prior_fusion` | `true` | ACTIVE |
| `op_believed_lean` | `true` | ACTIVE |
| `op_drop_renders` | `true` | ACTIVE |
| `opp_belief_cls_k` | `6` | ACTIVE |
| `opp_belief_slots` | `true` | ACTIVE |
| `opp_intent` | `true` | ACTIVE |
| `opp_intent_grad_mode` | `"detached"` | ACTIVE |
| `pair_outcome_cell` | `true` | ACTIVE |
| `pair_outcome_switch` | `true` | ACTIVE |
| `pair_value_route` | `false` | OFF |
| `q_winprob_mode` | `"none"` | OFF |
| `species_prior_fusion` | `true` | ACTIVE |
| `spread_belief` | `true` | ACTIVE |
| `spread_belief_nature` | `true` | ACTIVE |
| `switch_branch_cell` | `true` | ACTIVE |
| `t0_species_prior` | `true` | ACTIVE |
| `value_dist_bins` | `0` | OFF |
| `value_dist_mode` | `"none"` | OFF |
| `value_dist_vmax` | `0.0` | OFF |
| `value_dist_vmin` | `0.0` | OFF |
| `value_entity_pool` | `true` | ACTIVE |
| `value_entity_pool_full` | `true` | ACTIVE |
| `value_threat_inject` | `true` | ACTIVE |
| `win_prob_mode` | `"shaping"` | ACTIVE |
| `cf_evidential_coef` | `0.0` | OFF |
| `cf_evidential_reg` | `0.001` | INERT — no `cf_evid_head` |
| `cf_shadow_coef` | `0.0` | OFF |
| `cf_twin_coef` | `0.0` | OFF |
| `cf_winprob_coef` | `0.0` | INERT — coef 0, `win_head` built |
| `hp_type_belief_coef` | `0.05` | ACTIVE |
| `intent_label_bot_weight` | `0.25` | ACTIVE |
| `item_belief_coef` | `0.05` | ACTIVE |
| `move_belief_coef` | `0.05` | ACTIVE |
| `move_belief_latent_coef` | `0.05` | ACTIVE |
| `opp_belief_aux_coef` | `0.05` | ACTIVE |
| `policy_grad_coef` | `1.0` | ACTIVE |
| `q_winprob_coef` | `0.0` | OFF |
| `q_winprob_onpolicy_coef` | `0.0` | OFF |
| `spread_belief_coef` | `0.05` | ACTIVE |
| `td_aux_coef` | `0.0` | OFF |
| `value_dist_coef` | `1.0` | INERT — no `value_dist_head` |
| `value_tail_weight` | `0.0` | OFF |
| `vf_coef` | `0.5` | ACTIVE |
| `win_prob_coef` | `1.0` | ACTIVE |
| `win_prob_pbrs_coef` | `0.0` | INERT — coef 0, `win_head` built |
<!-- END GENERATED: flag-table -->

### 6.3 Reward config (resume-immutable, `check_reward_config`)

**The production reward is ONE TERMINAL TERM.** `hand_shaping` **false** · `terminal_indicator`
**true** · `victory_value` **1.0** · `draw_penalty` **0.0** · `no_progress_tax_armed` **false** ·
γ **1.0**. The composition is **1 TERMINAL (`win_loss`) + 0 PBRS + 0 BIAS**: `+1.0` on a win, `0.0`
on a loss, a draw and a 250-turn timeout alike.

🚨 **Three fields are recorded `true` and are INERT — read the composition, never these.**
`all_shaping_pbrs` **true**, `pbrs_material` **true** and `pbrs_belief` **true** are all in
`model_config.json` and none of them emits a term: `--no-hand-shaping` zeroes the whole shaping
surface and `--terminal-indicator` replaces the terminal itself, so the flags describe a composition
that is not built. They are recorded because `RewardConfig` is resume-immutable and every field must
round-trip — they are *the values a resume must re-pass*, not a description of the objective. The
authority is `metadata.json`'s `reward_composition` block (0 pbrs terms, 0 bias terms), which a
launch also prints: `train_rl_agent` emits `[Reward] composition: …`
(`reward_composition.format_reward_composition`) and records the census
(`reward_composition.reward_class_composition`). *(A frozen-φ provenance fix is in flight that will
make the three read as resolved; until it lands the composition block is the honest reading.)*

The remaining fields are the DEFAULTS, recorded and inert for the same reason: `no_progress_penalty`
0.15 · `mat_alive_weight` 1.25 · `bias_additivity` 1.0 · `self_ko_hp_penalty` 0.0 ·
`switch_bias_weight` 0.0 · `bias_redesign` false · `drop_redundant_bias` false ·
`drop_switch_bias` false · `stall_pbrs` false.

Under `--critic shaped` the same fields resolve to a real composition — **1 TERMINAL + 7 PBRS +
1 BIAS (`no_progress_tax`)** at `all_shaping_pbrs` true and a −35 `draw_penalty`, every non-stall
shaping term a telescoping potential and the anti-stall tilt the single acknowledged objective bias;
`--no-all-shaping-pbrs` is its fallback and a different objective rather than a smaller one
(2 potentials and **26 additive BIAS terms**, with `no_progress_tax` itself disarmed, its clock
charge gating on `bias_redesign OR all_shaping_pbrs`).

`stall_pbrs` stays off deliberately. Turning it on additionally zeroes `no_progress_tax` and folds
Φ_progress instead — the zero-BIAS destination, but a separate single-variable step, since the
stall tilt carries a documented stall-regression risk.

**Resume:** every field here is enforced on the training-resume path only. A run recorded under the
pre-2026-08-18 defaults (`all_shaping_pbrs` false, `draw_penalty` −30.0 — every `ai_v9_*` run
through gen-14) therefore FATALs on a flagless resume and must re-pass
`--no-all-shaping-pbrs --draw-penalty -30.0`; the error names the flags. Frozen eval / pool /
distill opponents are unaffected — `check_compatible` excludes reward fields, because their forward
never reads the reward.

### 6.4 Runtime knobs (never versioned, must be re-passed on every resume)

`--use-bridge rust` (serverless) · `--compile-opponents` + `--compile-opponents-preload` +
`--compile-trainer` (all ON by default) · `--grad-accum-steps` at whatever `--batch-size` the run
uses · `--grad-checkpointing` · `--async-rollout`. These do not appear in `model_config.json` and
are **not** inherited on resume — with the compile flags defaulting ON it is the OPT-OUT that must
be re-passed each launch, not the flag.

⚠️ **`--gamma` is not in `model_config.json` either.** `--critic winprob` implies γ 1.0 and the
resume path restores the checkpoint's own γ, so the value in force is visible in `metadata.json`'s
`cli_args` and in the startup lines — not in the mirror, and therefore not in §6's table.

---

## 7. Training-only obs keys — the leak-safety list

These are Dict-obs keys emitted by `Gen3Env` for supervision. **The forward reads only
`obs["observation"]`** (`ObsUnpack.forward`), so none of them can reach `pi`/`vf` or any pointer
logit. Declared conditionally, so a key absent from the space is simply not emitted.

| Key | Shape | Consumer | Emitted when | In production? |
|---|---|---|---|---|
| `belief_species` | int64 `[6]` | `BeliefHead` species CE | `opp_belief_aux_coef > 0` **or** `move_belief_mode != off` | ✅ emitted and consumed (`opp_belief_aux_coef` 0.05) |
| `belief_moves` | int64 `[6,4]` | `BeliefHead` moves BCE (Hungarian) | " | ✅ emitted and consumed |
| `known_moves` | int64 `[6,4]` | `MoveBelief` BCE | `move_belief_mode` ∈ {revealed, both} | ✅ emitted and consumed (`move_belief_coef` 0.05) |
| `belief_spread` / `belief_spread_mask` | f32 `[6,5]` / `[6]` | `SpreadBelief` regression | `spread_belief` **and** `spread_belief_coef > 0` | ✅ emitted and consumed |
| `belief_nature` / `belief_nature_mask` | int64 `[6]` / f32 `[6]` | nature CE | " | ✅ (`spread_belief_nature` true) |
| `belief_ev` / `belief_ev_mask` | f32 `[6,5]` / `[6]` | EV smooth-L1 | " | ✅ |
| `hp_type_label` / `hp_type_mask` | int64 `[6]` / f32 `[6]` | HP-type CE | `move_belief_mode != off` **and** `hp_belief_mode == composed` **and** `hp_type_belief_coef > 0` | ✅ **emitted and consumed** |
| `item_label` / `item_mask` | int64 `[6]` / f32 `[6]` | item CE (`gen3_item_belief_v1`) | `item_belief` **and** `item_belief_coef > 0` | ✅ emitted and consumed (`item_belief_coef` 0.05) |
| `win_target` / `win_mask` / `win_margin` | f32 `[1]` each | the win-prob head's BCE — under `--critic winprob` **the value loss itself** (MC outcome, a **future** label back-filled by `WinProbLabelCallback`) | `win_prob_mode != none` | ✅ **emitted and consumed — this is the critic's target** |
| `defensive_opportunity` | f32 `[1]` | state-conditioned entropy boost | `--defensive-entropy-boost > 1.0` (default 1.0) | ❌ |
| `bait_opportunity` | f32 `[1]` | state-conditioned entropy boost (bait) | `--bait-entropy-boost > 1.0` (default 1.0) | ❌ |
| `distill_mask` | f32 `[1]` | exploiter-distillation KL gate | `--distill-coef > 0` with teacher teams | ❌ |

🚨 **In this config a privileged key is no longer merely auxiliary — `win_target` IS the critic's
training target.** The leak-safety property is unchanged and is exactly what makes that safe: the
forward reads `obs["observation"]` alone, so a future outcome can label the value head without ever
being visible to it at decision time. But the consequence for reasoning is real — "training-only"
now means "not in the forward", never "not load-bearing".

Every belief label above is both emitted AND consumed here (all six supervised coefficients are at
0.05), so the whole emitted set is read. **Do not infer supervision from emission**, though: the
emit gates and the loss coefficients are separate conditions, and a config that drops a coefficient
to 0 keeps paying the buffer cost while training nothing — which reads identically in every metric.
`--defensive-entropy-boost`, `--bait-entropy-boost` and `--distill-coef` are off, so their three
keys are not emitted at all.

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
   `gen3_deadline_clock_v1` it was **2669** (the
   per-mon recency block added 12 × 3). ⚠️ **Every obs dim in this section is AS-FOUND in 2026-08;
   live is 2501** — see §1. Its per-block reference section then describes the
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
   observation"** — obs-layout generations out of date (2669 at audit time; **2501 live**).
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
| **This document as a clickable digraph** — the **120 nodes / 1103 edges** above (counted 2026-08-23 from `delivery_graph_snapshot.json`, which the viewer is built from — read it there rather than trusting this cell), hue-coded by what each channel physically carries, with a per-checkpoint measured-dependence overlay and a path filter (pick `vf_projection` to see exactly what the critic reads) | **https://model.g5d.io** (served live from the workstation checkout, so it is never a stale copy), or `designs/architecture_viewer.html` via `file://`. **Generated — never hand-edit it**: rebuild with `python -m agents.model.build_arch_viewer`, and `--check` fails if the committed artifact has drifted from the graph. |
| Obs-build performance gate (mandatory benchmark) + per-slot detail | `src/agents/observation/CLAUDE.md` |
| Phase contract, `ExtractorContext`, versioning playbook | `src/agents/model/CLAUDE.md` |
| How it got here — every version entry, verbatim | `designs/CHANGELOG.md` |
| Which `ai_vN` folder is relevant | `designs/CLAUDE.md` |
| Hypothesis status / what has been killed | `designs/research_state/ledger.md` |
| **The raw audit outputs behind every measured number here** | `designs/research_state/measurements/` (+ its README for how to read one) |
| Delivery-channel theory (edge vs content vs cell) | `designs/learning/shortcut_learning_and_feature_delivery.md` |
| Event-sourced battle layer + read-models | `src/agents/battle/CLAUDE.md` |
| Training loop, eval sharding, ELO | `src/agents/training/CLAUDE.md` |
