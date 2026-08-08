# ARCHITECTURE.md — what is true NOW

**Scope: current state only.** History lives in [`CHANGELOG.md`](CHANGELOG.md); versioning
*mechanics* live in `src/agents/model/CLAUDE.md`. If a claim here disagrees with the code, the code
wins and this file is a bug — fix it in the same change.

**"Current" means the production configuration**, which is the live gen-3 self-play run:

| | |
|---|---|
| Run | `models/run_20260807_135637_gen3/` |
| Started | 2026-08-07, git `e60a1e1` |
| Config | `model_config.json` (`config_version` **59**, `arch_signature` **`gen3_edge_bias_trunk_v1`**) |
| Invocation | `command.txt` (also `metadata.json` → `original_command`) |
| Progress at time of writing | step 32,000,016 of 40,000,000; bots 0.911, anchored ELO 2094 (`metadata.json` → `latest_eval`, 2026-08-08) |

Everything below was derived on **2026-08-08** by instantiating that config against the code, not
by reading prose. Regenerate with `tmp/arch_ground_truth.py` (a probe, not a test).

**Conventions used in this file**
- No version numbers in prose. Blocks and families are named structurally
  (`outgoing_matrix`, not "v34's matrix"). Where a version tag is genuinely needed it appears in a
  trailing parenthetical only.
- Every measured number carries checkpoint / step / state-count / date inline.
- A claim that could not be verified against code or the run config is marked **UNVERIFIED**.

---

## 1. Observation

One flat `float32` vector of **2925** dims, plus an 11-dim `action_mask`, delivered as a Dict obs.
Every number below comes from `agents/observation/constants.py` and
`Gen3ObservationEncoder.get_layout()`. **Never hardcode an offset — read the layout.**

### 1.1 Top-level blocks

| Block | Start | End | Dims | Constant |
|---|---|---|---|---|
| Our team — 6 × per-mon slot | 0 | 678 | 678 | `OFFSET_OUR_TEAM`, `6 × POKEMON_FULL_DIM` |
| Opp team — 6 × per-mon slot | 678 | 1356 | 678 | `OFFSET_OPP_TEAM` |
| Active context ×2 (ours, theirs) | 1356 | 1472 | 116 | `OFFSET_CONTEXT`, `2 × ACTIVE_CONTEXT_DIM` (58) |
| Global env | 1472 | 1490 | 18 | `OFFSET_GLOBAL`, `GLOBAL_ENV_DIM` |
| Reactive | 1490 | 1801 | 311 | `OFFSET_REACTIVE`, `REACTIVE_DIM` |
| *(= `base_dim`)* | | 1801 | | |
| Prev-turn action mask | 1801 | 1812 | 11 | `ACTION_SPACE_SIZE` |
| Turn history — 7 × TurnDelta | 1812 | 2925 | 1113 | `N_HISTORY_TURNS` (7) × `TURN_DELTA_DIM` (159) |
| **Total** | | **2925** | | `Gen3ObservationEncoder.dimension` |

> The `OFFSET_*` values in the comments inside `constants.py` (642 / 1284 / 1400 / 1418) are stale
> arithmetic from an older per-mon width. The **expressions** are correct and are what runs; only
> the trailing `# 642`-style comments are wrong. Listed in §8.

### 1.2 Per-Pokémon slot — 113 dims (`POKEMON_FULL_DIM`)

`POKEMON_VECTOR_DIM` is 112; `state_encoder` appends 1 active flag → 113.

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
| active flag | 112 | 1 | appended by `state_encoder` |

Move slot (11, `moves.py`): `[id, power/200, has_secondary, has_recoil, type_id, category, known,
current_pp, max_pp, accuracy, never_miss]`.

### 1.3 Reactive block — 311 dims

**11 scalars, then the matchups, then the request-ordered active moves.** The 11 scalar indices are
current; anything citing `vec[14]`–`vec[18]` is describing a layout deleted with the CPU damage
blocks (see §8).

| Field | Offset in reactive | Dims |
|---|---|---|
| `fainted` (ours, theirs) | 0 | 2 |
| `active_status` | 2 | 1 |
| `forced_struggle` | 3 | 1 |
| `trapped` | 4 | 1 |
| `maybe_trapped` | 5 | 1 |
| `turns_since_progress` | 6 | 1 |
| `protect_odds_our` | 7 | 1 |
| `protect_odds_opp` | 8 | 1 |
| `wish_floating_our` | 9 | 1 |
| `wish_floating_opp` | 10 | 1 |
| `our_matchups` | 11 | 144 |
| `their_matchups` | 155 | 144 |
| `active_req_moves` — `[move_num ×4, type_id ×4, legal_now ×4]` | 299 | 12 |

`active_req_moves` is in **request-slot order** (slot *k* ↔ action logit 6+*k*) and is sliced
straight into `ExtractorContext`; it never enters the raw-scalar projection path. The per-mon move
block (§1.2) stays **sorted by id** because the role token concatenates the 4 encodings and is
therefore order-sensitive. Both orders are live simultaneously — that is the reason the pointer
head permutes by move-num identity (§3).

`non_matchup_rest` — the raw-scalar tail the global token and both projection heads read — is
`GLOBAL_ENV_DIM (18) + the 11 reactive scalars = 29` dims. It stops at the matchup offset, so the
matchups and `active_req_moves` are excluded from it by construction.

### 1.4 Global env — 18 dims

`weather` 7 · `hazards` (spikes ×2) 2 · `clock` 1 · `screens` 8. (`global_env.py`)

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

Modules actually built under the production config (`named_children()`, verified 2026-08-08):

```
embeddings · unpack · pokemon_encoder · entity_seats · edge_bias · team_transformer ·
cls_pool · move_belief · hp_type_belief_head · damage_op · prefuse_proj · assembler ·
pre_proj_norm · projection · value_pre_norm · value_projection · activation
```

Notably **absent** (their flags are off): `belief_slots`, `belief_head`, `spread_belief`,
`hidden_opp_belief`, `refine_proj`, `outgoing_proj`, `status_in_proj` / `status_out_proj`,
`reattend_*`, `zarch_encoder` / `film_*`, `win_head`, `pubval_head`, `value_dist_head`.

### 2.1 Order of operations

Because `damage_op_prefuse` is on, the belief + physics stack runs **once, before attention**:

1. **`ObsUnpack`** — slices the 2925-dim vector into `ExtractorContext` (~30 named tensors:
   per-mon blocks, categorical ids, matchups, active-slot indices, fainted key-masks,
   `our_active_req_move_{ids,type_ids,legal}`).
2. **`PokemonEncoder`** — per-move network (`MOVE_NET_HIDDEN` `[96,32]`, with the `MoveLatentEncoder`
   latent concatenated in) → within-mon move self-attention → role encoder
   (`ROLE_ENCODER_HIDDEN` `[256,128]`) → **12 × 128 role tokens**. Stashes
   `last_move_tokens` `[B,12,4,32]` (sorted-by-id) for the seats and the pointer head.
3. **`MoveBelief`** (pre-fuse) — reads the opp **role** tokens, predicts each opp slot's moveset,
   fuses the Smogon log-odds prior, pins revealed moves, and reinjects the soft-embedded moveset
   into the opp role tokens. Stash: `last_move_belief_logits` `[B,6,400]`.
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

There is **no between-layers refine loop** in this config (`damage_refine_rounds` = 0, mutually
exclusive with the pre-fuse). Consequences: §6.

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

### 3.1 `pi_projection` — 1137 → 512

`Linear(1137, 512)` after `pre_proj_norm` (LayerNorm), then ReLU. Input concat, in order:

| Part | Dims | Source |
|---|---|---|
| `our_team_pooled` | 128 | `CLSPool.our_cls` over our 6 refined tokens |
| `their_team_pooled` | 128 | `CLSPool.their_cls` over their 6 refined tokens |
| `our_active_refined` | 128 | our active slot's refined token |
| `active_ctx_enc` (ours) | 32 | `active_ctx_encoder` (shared with vf) |
| `active_ctx_enc` (theirs) | 32 | " |
| `non_matchup_rest` | 29 | global env 18 + reactive scalars 11 |
| **damage block** | **660** | `DamageOperator` output, post-gain (§4) |
| **total** | **1137** | |

### 3.2 `vf_projection` — 881 → 512

| Part | Dims | Source |
|---|---|---|
| `value_pooled` | 128 | `CLSPool.value_cls` over **all 12** team tokens |
| `active_ctx_enc` ×2 | 64 | shared with pi |
| `non_matchup_rest` | 29 | |
| **damage block** | **660** | the **same** tensor pi reads |
| **total** | **881** | |

The value head does **not** read `our_active_refined` (`value_active_readout` is off), and does not
read either team pool. Its only board summary is `value_pooled`.

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

### 3.4 Side readouts

None are built in this config (`win_prob_mode`, `pubval_mode`, `value_dist_mode` all `none`;
`value_from_dist` false). The critic is the scalar `value_net`, PopArt-normalized
(`use_popart` true, `--clip-range-vf none`).

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

Also **not present anywhere** (deleted with the op block trim, not merely off): the opp-active
collapsed effect scalars, the opp-active collapsed incoming-secondary scalars, the outgoing
slp/psn/tox columns, and the lean top-K block. `damage_topk_k` now sizes the incoming matrix and
nothing else; `K > 0` without `damage_matrices_incoming` **raises** in both the extractor and the
op.

### 4.1 ⚠️ Per-block dependence measurements — read the config before quoting

The frequently-cited per-block ablation table (`tmp/op_block_ablation_probe.py`, **2026-07-25**,
4000 real eval states, per-block zero → masked KL, ceiling 0.9385 = zeroing the whole op) was
measured on a **different model and a different config**. Two things make it non-transferable:

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

**There is no current-config equivalent of this table.** The nearest measurement is the per-family
edge audit (§5.4), which is a different instrument (edges, not concat sub-blocks) and was taken on
gen-1 mid-training. Producing a per-block concat ablation on the gen-3 snapshot is open work.

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
| x | `damage_op` **and** `damage_op_prefuse` |
| d3, s3 | `entity_topk_seats > 0` (the bias rows *are* the E4 seats) |

All are satisfied in the production config.

### 5.3 What an edge can and cannot carry

Attention weights are softmax-**normalised**, so an edge bias moves *who attends to whom*: what it
writes is a **ratio within its row**, not an absolute magnitude ("53% of max HP"). The two channels
that can carry an absolute are **token content** (`prefuse_proj`) and **per-action pointer cells**
(§3.3). This is a capacity/conditioning argument, not an impossibility proof; the reasoning is in
`designs/learning/shortcut_learning_and_feature_delivery.md`.

### 5.4 Edge-family audit — provenance

The only per-family dependence numbers in the repo are **gen-1 @9.6M of 40M steps, 2048 eval-trace
states, 2026-08-04, PRELIMINARY/mid-training** (`edge_audit_9p6M.json`, produced by
`src/agents/model/edge_ablation_audit.py`): d1 KL 0.059 / 8.2% action flips; d2 KL 0.057 / 10.3%
flips / |dV| 1.62; d3 0.0009; s3 0.00007; s1 0.002; v 0.002 / 3.1% flips; all families off = KL
0.124 / 16.6% flips.

**Do not read these as current.** That run had 6 families, not 15 (t, x, g, c1–c5, E5 and the K=6
default all postdate it), it was 24% through training, and its `v` family was measured on the
speed-stat GIGO bug (§8). A gen-1 end-of-run audit (`edge_audit_40M.json`, all-off = 26.9% flips)
exists per `designs/CLAUDE.md`; **UNVERIFIED:** neither JSON is present in this worktree, so the
per-family breakdown at 40M could not be checked, and no gen-3 audit has been run.

---

## 6. Flags — production value and status

Read from `models/run_20260807_135637_gen3/model_config.json` + `command.txt`, 2026-08-08.

**Status legend** — `ACTIVE`: on and doing work. `OFF`: not enabled. `INERT`: nominally set but
does nothing given another setting. `UNREACHABLE`: cannot be enabled in this config — turning it on
raises at build time.

### 6.1 Architecture toggles

| Flag / config field | Production value | Status |
|---|---|---|
| `damage_op` | true | ACTIVE — the 660-dim block, both heads + pointer cells |
| `damage_op_prefuse` | true | ACTIVE — the whole belief+physics stack runs once, pre-attention |
| `damage_outgoing` | true | ACTIVE — outgoing block + status landing; also gates the pointer **move** cells and edges d1/s1/c1/c2 |
| `move_belief_mode` | `"revealed"` | ACTIVE — predicts a *seen* mon's unrevealed moves |
| `move_belief_prefuse` | true | ACTIVE — required by `damage_op_prefuse` |
| `move_prior_fusion` | true | ACTIVE — posterior = Smogon log-odds prior ⊕ learned delta, revealed pinned |
| `move_latent` | true | ACTIVE — `MoveLatentEncoder`; required by `damage_matrices_incoming` |
| `hp_belief_mode` | `"composed"` | ACTIVE — the factorised presence × type head |
| `damage_matrices_incoming` | true | ACTIVE — 522 of the op's 660 dims |
| `damage_topk_k` | 6 | ACTIVE — sizes the incoming matrix (and nothing else) |
| `consequence_topk` | 6 | ACTIVE — k_cand for c1b/c2/c3, k_bench for d4 |
| `entity_topk_seats` | 6 | ACTIVE — E4 seats; also the precondition for edges d3/s3 |
| `entity_tail_seats` | true | ACTIVE — E5 seats |
| `edge_bias_families` | all 15 | ACTIVE (§5) |
| `attend_unrevealed_opponents` | true | ACTIVE — hidden opp slots stay attendable |
| `use_popart` | true | ACTIVE — with the mandatory `--clip-range-vf none` |
| `belief_grad_mode` | `"shaping"` | ACTIVE — belief heads read the live trunk |
| **`move_belief_single_compute`** | **true** | **INERT** — its only effect is to make the between-layers refine callback reuse the frozen posterior. `damage_refine_rounds` = 0 ⇒ **no callback is built** ⇒ nothing reads it. Under the pre-fuse the belief is computed once by construction. |
| `damage_matrices_outgoing` | false | OFF — the 126-dim `outgoing_matrix` does not exist |
| `damage_matrices_outgoing_all` | false | OFF — the 108-dim OAX block does not exist; the **pointer switch cell is 15 wide, not 33** (§3.3) |
| `damage_refine_rounds` | 0 | OFF — mutually exclusive with `damage_op_prefuse` |
| `damage_candidate_k` | 0 | OFF — the full candidate sweep, no truncation |
| `damage_reattend` | false | OFF |
| `opp_belief_aux_coef` | 0.0 | OFF ⇒ `opp_belief_slots` false ⇒ **no `BeliefHead`, no species posterior** |
| `opp_belief_latent` | false | OFF |
| `opp_belief_cls_k` | 0 | OFF |
| `spread_belief` | false | OFF — the op prices opponent stats with its hand-coded de-timid / neutral-0-EV constants, not a learned belief |
| `spread_belief_nature` | false | OFF |
| `spread_belief_nature_marginalize` | false | UNREACHABLE — requires `spread_belief_nature` |
| **`threat_refine_outgoing`** | false | **UNREACHABLE** — hard-requires `damage_refine_rounds > 0`, which is mutually exclusive with `damage_op_prefuse`. Setting it raises `ValueError` at extractor build. |
| **`threat_unrevealed_outgoing`** | false | **UNREACHABLE** — requires `threat_refine_outgoing` (above), *and* a `BeliefHead` for `species_posterior`, which does not exist (`opp_belief_aux_coef` = 0). Two independent blockers. |
| `threat_prob_outspeed` | false | OFF — `p_outspeed` uses the fixed logistic scale, not the believed-speed std |
| **`threat_status_refine`** | false | **UNREACHABLE** — same refine-loop dependency as `threat_refine_outgoing` |
| `win_prob_mode` | `"none"` | OFF |
| `pubval_mode` | `"none"` | OFF |
| `value_dist_mode` / `value_dist_bins` | `"none"` / 0 | OFF |
| `value_from_dist` | false | OFF — the critic is the scalar `value_net` |
| `value_active_readout` | false | OFF — vf does not read `our_active_refined` |
| `zarch_film` / `zarch_dim` / `zarch_lut` | `"off"` / 0 / `"off"` | OFF — no team-archetype conditioning |
| `move_candidate_floor` | 0.0 | OFF — the legacy flat prior floor; the learnset legality gate is not applied |

**No flag exists for the pointer-native action head.** It is unconditional — there is no off state,
and there is no flat `action_net` to fall back to.

### 6.2 Training-loss coefficients (recorded, not weight-shape)

| Field | Value | Effect |
|---|---|---|
| `hp_type_belief_coef` | 0.05 | ACTIVE — CE on the HP-type posterior |
| `move_belief_latent_coef` | 0.05 | ACTIVE — cosine latent grading + VICReg |
| **`move_belief_coef`** | **0.0** | **INERT** — the per-move BCE is off. The move belief is trained **only** by the damage-operator / edge / seat gradients flowing back through `w`, plus the HP-type CE. |
| `spread_belief_coef` | 0.0 | INERT — no `SpreadBelief` module exists anyway |
| `opp_belief_aux_coef` / `opp_belief_latent_coef` | 0.0 | INERT — no `BeliefHead` |
| `win_prob_coef` / `pubval_coef` / `value_dist_coef` | 1.0 / 0.1 / 1.0 | INERT — the corresponding heads are not built |
| `vf_coef` | 0.5 | ACTIVE — resume-immutable |
| `value_tail_weight` | 0.0 | OFF — plain MSE value loss |
| `zarch_recon_coef` / `zarch_vicreg_coef` | 1.0 / 0.1 | INERT — no `ZArchEncoder` |

### 6.3 Reward config (resume-immutable, `check_reward_config`)

`draw_penalty` −30.0 · `no_progress_penalty` 0.15 · `mat_alive_weight` 1.25 · `bias_additivity` 1.0
· `self_ko_hp_penalty` 0.0 · `switch_bias_weight` 0.0 · `bias_redesign` false ·
`drop_redundant_bias` false · `drop_switch_bias` false · `all_shaping_pbrs` false · `stall_pbrs`
false.

### 6.4 Runtime knobs (never versioned, must be re-passed on every resume)

`--compile-extractor` (on in this run) · `--grad-accum-steps 4` · `--grad-checkpointing` ·
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
| `belief_target_slots` | f32 `[6,113]` | SimSiam latent belief | `opp_belief_latent_coef > 0` | ❌ |
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
`damage_op_test.test_op_is_leak_free_of_privileged_keys`, and the bridge fuzz
`poke_env_gaps/belief_labels_fuzz_test.py`.

---

## 8. Known contradictions between the old prose and the code

Found while deriving this file (2026-08-08). Each is a place where a doc asserted something the
code does not do. None are fixed in `src/` by this pass — they are recorded so the next reader does
not re-derive them.

1. **Root `CLAUDE.md` reactive-block prose was two revisions stale.** It described "the 414-dim
   reactive block (**19 scalars**)" with `turns_since_progress` at `vec[14]`, protect-odds at
   `vec[15]`/`vec[16]`, wish at `vec[17]`/`vec[18]`. Real: **311 dims, 11 scalars**, progress at
   offset **6**, protect at **7**/**8**, wish at **9**/**10**. The summary table 60 lines above it
   was correct — the table and the prose contradicted each other in the same section.
2. **`src/agents/observation/CLAUDE.md` opens with "2889-dim"**; the live obs is **2925** (the
   per-mon recency block added 12 × 3). Its per-block reference section then describes the
   pre-deletion 414-dim reactive layout and the 51-dim incoming-damage / 44-dim move-effect blocks
   as if present. Its own inline banner says to treat the deletion note as authoritative — i.e. the
   file tells you not to trust the rest of the file.
3. **`src/agents/model/CLAUDE.md` states `ARCH_SIGNATURE` in three places with two different
   values** (`gen3_op_block_trim_v1` in the phase-structure rules, `gen3_edge_bias_trunk_v1`
   later) and states `MODEL_CONFIG_VERSION` as 31/32/37/38/40/41/43/44/45/46/47/53/55/57 in
   different paragraphs. Live: **59** and `gen3_edge_bias_trunk_v1`. It also contains a duplicated,
   partly-conflicting pair of paragraphs (two "`MODEL_CONFIG_VERSION` was **38** at v38" endings).
4. **Root `CLAUDE.md` claimed `MODEL_CONFIG_VERSION` = 57.** Live: **59** (58 = the speed-stat GIGO
   stamp, 59 = `consequence_topk`). Neither v58 nor v59 was described anywhere in the root file.
5. **`src/agents/model/CLAUDE.md` describes `ObsUnpack` as peeling "the flat 3390-dim
   observation"** — three obs-layout generations out of date (live 2925).
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
| Obs-build performance gate (mandatory benchmark) + per-slot detail | `src/agents/observation/CLAUDE.md` |
| Phase contract, `ExtractorContext`, versioning playbook | `src/agents/model/CLAUDE.md` |
| How it got here — every version entry, verbatim | `designs/CHANGELOG.md` |
| Which `ai_vN` folder is relevant | `designs/CLAUDE.md` |
| Hypothesis status / what has been killed | `designs/research_state/ledger.md` |
| Delivery-channel theory (edge vs content vs cell) | `designs/learning/shortcut_learning_and_feature_delivery.md` |
| Event-sourced battle layer + read-models | `src/agents/battle/CLAUDE.md` |
| Training loop, eval sharding, ELO | `src/agents/training/CLAUDE.md` |
