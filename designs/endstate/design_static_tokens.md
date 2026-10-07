# Static Pokémon tokens: identity from a table, battle state beside it (`--token-encoding static`)

**Status: DESIGN + BUILD STAGES 1 AND 2 (2026-10-06), behind `--token-encoding {legacy,static}`, default
`legacy`. Production is `legacy`. Nothing has been trained on `static`; every strength claim below is a
hypothesis for the screen in §8.** Stage 1 (`gen3_static_tokens_v1`, config v139: the per-mon encoder, S + D,
and the X5 table gather) takes board context OUT of the per-mon tokens. Stage 2 (`gen3_static_board_v1`, config
v140, §4: the three board tokens SIDE ×2 + FIELD, the edge retargets, the readers, and the per-mon op content
on both sides, from the entity-coverage audit) gives that context its new home. **`static` is now buildable
as the screen's arm; the screen (§8) is DRAFTED, NOT REGISTERED** (the orchestrator registers it before any
seed launches), and the GPU checks of §10 are deferred. This is the architecture audit's lead hypothesis
([`design_arch_audit.md`](design_arch_audit.md) §4 L1, with F4 and F14 folded in), widened by the owner
on 2026-10-06: *"I want item, ability, moves and stats all static if possible; let the attention and the
damage op do the heavy lifting."* [`../ARCHITECTURE.md`](../ARCHITECTURE.md) states the model as it is;
this doc states the static arm, why, and how it will be judged. Tags: **MEASURED** (a number with its
source), **ESTIMATED** (arithmetic from measured parts), **UNVERIFIED**.

---

## 0. Summary

Today every Pokémon's token is built by `PokemonEncoder` from its own row AND from five board facts
(clock, weather, faint counts, Spikes, screens) that each of the 12 encoders reads in its first layers
(audit F4). So a token is not a function of *who the Pokémon is*: two identical Skarmory on two rows get
different tokens because the weather differs. That has two costs: the trunk cannot decide which tokens
need which context (it is decided by construction), and an X5 hypothesis token (a guessed species for a
hidden slot) cannot be a table read, which is most of X5's encoding cost (X5 §3.6:
`gen3_x5_hyp_gather_v1` cut it 10.09 → 4.38 ms per micro-batch, not to ≈ 0).

Under `static`, a Pokémon's token is the SUM of two parts:

```
token(mon) = S(species, set)  +  D(mon's own battle state)
```

- **S, the static identity**, reads only what the mon IS: species, base and actual stats, types, item,
  ability, the four moves pooled as a SET, the Hidden Power belief, and the reveal bits. For our team
  every field is exact. For the opponent the same construction reads what is revealed, with the Smogon
  spread prior standing in for the unknown stats.
- **D, the dynamic state**, reads only what is happening TO this mon: HP, status and its counters, the
  sleep-wake belief, recency, protect odds, its last action, the trap bits, the active flag, the
  item-consumed bit, its side's boosts and volatiles when it is the active, and its moves' PP and
  legality.
- **No board fact enters either part.** Clock, weather, faint counts, hazards and screens reach a token
  only through the trunk's attention and the damage operator's physics. Their home is §4, from the
  entity-coverage audit ([`design_entity_coverage_audit.md`](design_entity_coverage_audit.md)): three
  board tokens (OUR SIDE and THEIR SIDE through one side-relative projection, and FIELD) replace the
  global token, and every mon on BOTH sides gets the operator's per-mon AMOUNTS as content (stage 2,
  built: §4.1).

Consequence: an X5 hypothesis token (a species with its dex row, full HP, no status) is a pure function
of the species, so it is a TABLE computed once per forward over the 400-row dex table and gathered:
exact, not split.

---

## 1. Which per-mon field is static, dynamic or board

Every field of the 122-dim per-mon slot (ARCHITECTURE §1.2), plus the per-mon inputs `PokemonEncoder`
reads from outside the slot. "Legacy" says where it goes today; every legacy route reads all of it
through the same two MLPs (the move network and the role encoder).

| field (slot offset) | class | static-mode home | why |
|---|---|---|---|
| species id (0) | STATIC | S: species embedding | identity |
| base stats (1–6) | STATIC | S: into the ACTUAL stats (below), not fed raw | identity; the stats are the fact the physics uses |
| item id, known (7, 8) | STATIC | S: item embedding + known bit | part of the set; the id survives consumption (`items.py`: a consumed item keeps its id with `consumed` = 1) |
| item consumed (9) | DYNAMIC | D | the owner's "item-consumed / changed flag" |
| type ids (10, 11) | STATIC | S | identity |
| ability ids, dominance, known (12–15) | STATIC | S | part of the set (the prior over two abilities for an unrevealed opponent) |
| status one-hot (16–22) | DYNAMIC | D | battle state |
| per move: id, power, secondary, recoil, type, category, known, max PP, accuracy, never-miss | STATIC | S: the move network per move, then the SET pool | the set |
| per move: current PP | DYNAMIC | D (per move) | battle state |
| our active's move legality (`active_req_moves` legal bits; legacy's "move validity") | DYNAMIC | D (per move) | changes turn by turn (Taunt, Disable, Choice lock, PP) |
| HP fraction (67) | DYNAMIC | D | battle state |
| species_known (68) | STATIC | S | a reveal bit: what we know about the identity |
| sleep / toxic counters (69, 70) | DYNAMIC | D | battle state |
| spread IV × 6, EV × 6, known, nature × 5 (71–88) | STATIC | S: into the ACTUAL stats | the set |
| Hidden Power block: revealed + 16 type probabilities (89–105) | STATIC | S | a belief about the SET (the IVs fix the HP type); it changes only on observation, like a revealed move |
| sleep-wake belief (106–108) | DYNAMIC | D | battle state |
| recency since seen / acted / hit (109–111) | DYNAMIC | D | battle state |
| protect-success odds (112) | DYNAMIC | D | battle state |
| last action: move id (embedded) + 5 outcome bits (113–118) | DYNAMIC | D | battle state |
| trapped, maybe-trapped (119, 120) | DYNAMIC | D | battle state |
| active flag (121) | DYNAMIC | D | battle state |
| own side's active context: boosts 14 + volatiles 46 (active only; the legacy "E2 injection") | DYNAMIC | D (zero on the bench) | the active mon's own state (the owner lists boosts and volatiles as per-mon dynamic) |
| clock 3, weather 7, faint counts 2, Spikes 2 | BOARD | **removed** from the per-mon path | legacy's move network context, read by all 48 move encodings |
| screens 8 | BOARD | **removed** | legacy's role-encoder broadcast context (with the four above) |

**Actual stats (the owner's "nature / EVs / IVs → actual stats").** S reads the six level-100 stats, not
base stats + a raw spread, divided by `STATIC_STAT_SCALE` (500):

- **our team** (`spread_known` = 1): the smooth gen-3 formula on the observed spread, the same one the
  damage operator uses for our mons: `HP = 2B + IV + EV/4 + 110`, `X = (2B + IV + EV/4 + 5) × nature`.
  Its uncertainty column is 0.
- **the opponent** (`spread_known` = 0): the Smogon usage-weighted mean and standard deviation of each
  realised stat (`gen3_data.priors.spreads`, the source of the operator's `SPECIES_SPREAD_PRIOR`, here
  with HP added), as a non-persistent buffer built from `data/` at construction. No new committed
  table; `data/` is untouched.

S's stat input is therefore 13 numbers: six means, six standard deviations, `spread_known`.

**The move SET (F14).** Each of the four moves goes through the move network (now with NO context
columns), the within-mon move self-attention (permutation-equivariant: no positions) and its norm, and
the four results are SUMMED into one 32-wide vector for the role network. Sum pooling after a per-element
network is the Deep Sets construction (Zaheer et al., *Deep Sets*, NeurIPS 2017), and a self-attention
block before the pool is the Set Transformer's encoder (Lee et al., ICML 2019). Sum (not mean) keeps the
count of known moves, which matters for an opponent with 1–4 revealed. Legacy concatenates the four in
sorted-by-id order into an order-sensitive Linear (deterministic, not symmetric: audit F14).

**ONE canonical move order, end to end.** Inside the encoder every per-move tensor is in the per-mon slot
order (sorted by id). The SET pool is a sum, so it reads no order at all, and it reads NO legality: legality
is per-decision DYNAMIC state and lives only in D. Our active's legality arrives in REQUEST order and is
matched to its slots by MOVE-NUM IDENTITY (`StaticTokenEncoder.move_legality`), never by position; the
pointer head and the E3 seats leave this order through the existing identity permutation
(`_request_order_move_tokens`). A position-based write is exactly legacy's F-ST-1 (§11).

## 2. The opponent's unknown set: prior, belief, hypothesis, reveal

| state of a fact | what S reads | where the learned belief enters |
|---|---|---|
| unrevealed species (a hidden slot, blob arm) | the empty row (species 0) | `BeliefSlots` replaces the token, as today |
| unrevealed species (fixed_mass arm) | the HYPOTHESIS species' dex row | S + D of the dex row, gathered from a 400-row table (§5) |
| revealed species, unrevealed moves / item / ability | species + the dex row's ability prior; move and item ids 0 (unknown) | `MoveBelief` reinjects the believed moveset into the opponent token after the encoder (unchanged); `ItemBelief`, `SpreadBelief` and `HPTypeBelief` feed the damage operator (unchanged) |
| unknown spread | the Smogon prior's per-species stat mean ± std | `SpreadBelief`'s learned correction feeds the operator (unchanged) |
| a revealed move / item / ability | the revealed id overrides the unknown | the move belief pins revealed moves (unchanged) |
| a consumed or knocked-off item | the item id stays; D's consumed bit flips | — |

So "the same construction gives the prior, the belief fills it, reveals override it" holds as follows.
The construction is S. The prior is the Smogon stat prior and the dex row's ability prior. The learned
beliefs fill the token through the existing post-encoder reinjection (moves) and fill the physics through
the operator (item, spread, HP type). A revealed fact replaces the unknown id in the row S reads.

**What S does NOT read: the learned spread / item belief.** Those heads read the role tokens S produces
(T0, after the encoder), so feeding their outputs back into S would be circular. They reach the model
through the operator, as today. A design that let S read a believed set would need the beliefs computed
before the encoder from a token-free input; not proposed.

## 3. How the dynamic state joins: ADDED

`token = S + D`: one 128-wide sum, the same convention as a transformer's token + position embedding
(Vaswani et al., *Attention Is All You Need*, 2017). The trunk's attention and FFN then mix identity and
state.

- **D's input** (per mon, 134 dims): HP, the status one-hot, the two counters, the sleep-wake belief,
  recency, protect odds, the last action (its move embedded through the shared move table + the five
  outcome bits), the trap bits, the active flag, the item-consumed bit, its side's 60-dim active context
  (zero on the bench), and a 32-wide pool of the per-move dynamic state:
  `Σ_k ReLU(W [m_k ; pp_k ; legal_k])`, where `m_k` is move k's STATIC token. Reading `m_k` ties a PP
  count to the move it belongs to. The pool is a sum, so it is permutation-invariant too.
- **D's network:** `Linear(134 → 128) → ReLU → Linear(128 → 128)`.
- **The per-move tokens the pointer head and the E3 seats read** are `m_k + Linear(2 → 32)[pp_k ; legal_k]`:
  the static move token plus its own state. They no longer carry the clock, weather or HP the legacy move
  network mixed in.

**Alternatives considered.**
- *A second token per mon* (12 more trunk tokens): the attention matrix grows (61 → 73 tokens, ≈ 1.4×),
  and option B of the audit (every per-mon state its own token) was judged over-reach at 128 dims. Not
  chosen.
- *Gated (FiLM-style) `S ⊙ (1 + g(D)) + b(D)`*: lets HP scale the identity multiplicatively, but a
  multiplicative join is the conditioning family the project measured as count-dominated (ai_v8's FiLM
  results, `project_count_dominates_conditioning`). The trunk's FFN can form any product it needs from a
  sum. Kept as a follow-up arm only if the screen says the additive join is short.

## 4. Board context: three board tokens + per-mon op content (stage 2, BUILT: §4.1)

**Decided by the entity-coverage audit** ([`design_entity_coverage_audit.md`](design_entity_coverage_audit.md),
`f32a9f4b`, 2026-10-06; its §6.1 layout and B1 + B2 are part of THIS arm, not separate levers). Once board
context leaves the per-mon encoder, three kinds of fact lose their route if the single global token stays
(audit §4):
- **A1, counts:** a fainted mon is a MASKED key, and a softmax over the alive keys is an average that
  cannot count, so the alive / fainted / revealed counts need an explicit home;
- **A2, mine vs theirs:** the global token packs every side fact as an (ours, theirs) pair, and the per-mon
  encoder has no side input (today it infers the side from `spread_known`);
- **A3, amounts:** an edge bias only reweights a softmax row, so the Spikes chip on entry, the end-of-turn
  ledger and our damage to THEIR mons can reach a token only as CONTENT; today `prefuse_proj` writes the
  op's incoming rows onto OUR six tokens only.

**The layout (stage 2):**

| seats | today (61) | static arm (63; 64 under X5) |
|---|---|---|
| 0–11 | 12 mon tokens (board broadcast inside) | 12 tokens = S + D (stage 1) + OPC, the op-derived per-mon content, BOTH sides |
| 12 | GLOBAL `[our_ctx 60, opp_ctx 60, non_matchup_rest 25]` | **OUR SIDE** |
| 13 | E3 starts | **THEIR SIDE** (the same `side_proj`; the token type tells them apart) |
| 14 | — | **FIELD** |
| then | E3 (4), E4 (6), E5 (6), [OTHER under X5], events (32) | unchanged, shifted by 2 (base = 2·TEAM_SIZE + 3) |

- **SIDE content** (side-relative, one projection for both sides): Spikes layers; Reflect, Light Screen,
  Safeguard, Mist presence; Wish pending; alive, fainted and revealed counts; Sleep Clause used. (Turns left
  on the screens, Future Sight on the target side and Freeze Clause are new OBSERVATION facts: the audit's
  B5 / B6, a separate later lever, NOT in this arm.)
- **FIELD content:** the weather one-hot (7, incl. permanence as the obs encodes it), the three clock
  scalars, turns since progress.
- **The active contexts** (boosts + volatiles) leave the board token entirely: they are D on the active mon
  (stage 1, already built).
- **Edges:** `x` retargets from the global seat to the mon's OWN SIDE token; `g` and `c4` to FIELD. No other
  family changes. (An optional zero-init membership family `m`, mon → its side token, is a cheap guard only.)
- **Readers:** the critic's `value_entity_pool_full` reads the three refined board tokens where it reads the
  refined global token today; the head-only `non_matchup_rest` route into the policy projection is DELETED
  (F4's fourth route).
- **OPC, the per-mon op content (audit B2, a REQUIREMENT of the arm):** each mon, on BOTH sides, gets a
  zero-init projection of: its incoming rows (today `prefuse_proj`, ours only), its Spikes chip on entry and
  its end-of-turn ledger (the `x` / `g` cells' amounts), and for their mons what our active does to them (a
  `d1`-style outgoing row). Without it a null on the screen indicts the missing amount route, not attention.

**Interaction with F2 (`--policy-readout {tower,trunk}`, landed the same day, config v138).** `trunk` retires
the flat policy tower and reads the decision context through `PolicyStateQuery`, one query over every refined
trunk token including the global token. Under `static`, stage 2's three board tokens change that key set, and
"delete the policy's board bypass" applies only to `tower` (under `trunk` there is no projection to delete
from). Stage 2 builds and tests `static` × {`tower`, `trunk`}; the screen runs on whichever readout is adopted.

**Why three tokens, not one or ten** (audit §6.2): one global token fails A1 / A2 / A4 (a 145-dim
bottleneck into one key); one token per condition would add ~10 mostly-zero seats to a trunk whose cost
binds (X5's +16.7 % `train_ms`). A condition is an attribute of a side, not an entity.

**Out of this arm (audit §6.3):** the five new observation facts (B5–B9: screen turns, Future Sight on the
target, elapsed volatile turns, the opponent's Choice-lock and disabled move, two-turn charge, what the
opponent has seen of us) are ONE separately flagged enrichment lever screened after this one; the
status-landing physics fix (B4: Safeguard, Sleep Clause incoming, our Substitute, Freeze Clause, Yawn)
belongs to the F11 move-resolution unit.

**Known reach loss that stage 2 does not restore (a FINDING):** the T0 belief heads (`MoveBelief`,
`SpreadBelief`, `ItemBelief`, `HPTypeBelief`) read the PRE-trunk opponent tokens, so under `static` they no
longer see weather or the clock (weather as a team-archetype cue: a rain team predicting Thunder). The board
tokens are refined in the trunk, after T0. X5's δ_θ is unaffected: it projects the raw context fields
itself (`our_ctx_raw`, `opp_ctx_raw`, `non_matchup_rest`, read off `ExtractorContext`, not off the global
token), so stage 2's token change does not touch it.

### 4.1 As built (stage 2, `gen3_static_board_v1`, config v140, 2026-10-06)

All of it under `token_encoding == "static"` only; `legacy` builds and runs exactly what it did (§10's identity
proof). Module: `agents/model/board_tokens.py`.

- **Seats.** `TeamTransformer(layout, token_encoding)`: under static the base is 15 (`_total_tokens`), the board
  seats are (12, 13, 14) (`board_seats`; legacy's are (12, 12, 12), so the edge code path is shared), and the
  token-type table has 9 rows (`TOKEN_TYPE_OUR_SIDE` 6, `TOKEN_TYPE_THEIR_SIDE` 7, `TOKEN_TYPE_FIELD` 8; legacy's
  stays 6 rows, `TOKEN_TYPE_GLOBAL`'s row is unused under static). `global_proj` is not built under static;
  `_global_token_input_dim` stays (X5's δ_θ projects the raw global fields itself). Every seat consumer reads
  `_total_tokens`; the one literal (`key_log_presence`'s length, `2·TEAM_SIZE + 1 + seats`) now reads it too.
- **SIDE content** (`side_features`, [B, 2, 10], row 0 ours, row 1 theirs, the same columns, `SIDE_FACTS`): Spikes
  layers /3; Reflect, Light Screen, Safeguard, Mist presence; Wish pending; alive count /6 (ours: HP > 0; theirs:
  `opp_addressable`, an unrevealed mon is alive); fainted count /6; revealed count /6 (Σ species_known); Sleep
  Clause used (a non-Rest sleeper on that side: the op's `nonrest_sleep` read, for both sides, as a clamp of 0/1
  observation bits). One `side_proj` = `Linear(10, 128)` for both rows; the token type tells them apart.
- **FIELD content** (`field_features`, 11): the weather block (one-hot 5 + permanent + turns left), the 3 clock
  scalars, turns since progress. `field_proj` = `Linear(11, 128)`.
- **Edges.** `EdgeBias.forward(…, board_seats=…)`: `x` writes (our mons, 12) and (their mons, 13), `g` and `c4`
  write to 14. No other family moves; the `r` family's "event seats are the final N tokens" contract holds.
- **Readers.** `value_entity_pool` (`full`) takes `board_rows` [B, 3, 128] (source tag 3, never masked) in place
  of `global_row`; `ProjectionAssembler` drops `non_matchup_rest` under static (`board_bypass`), and
  `compute_projection_widths(…, token_encoding)` says so (pi 1177 → 1152 on the production surface);
  `PolicyStateQuery` (`trunk`) keys on `TeamTransformer.board_rows()` (the 3 refined board tokens, in the trunk's
  own seat order, so the per-key log-presence stays aligned).
- **OPC** (`OpContent`, root attribute `op_content`, tier T1, zero-init, protected by `restore_identity_init`):
  `amount_proj` = `Linear(8, 128)` on every mon of BOTH sides over its `x` cell [entry_chip, pursuit_p,
  pursuit_eff, grounded] ⊕ its `g` cell [leftovers, weather_chip, status_tick, leech]; `outgoing_proj` =
  `Linear(24, 128)` on each of THEIR mons over the `d1` cells of our four request-order moves on it (built iff
  `--damage-outgoing`). Our mons' incoming rows keep riding `prefuse_proj`. The cells are the edge families' own
  tensors when a family built them this forward (production: all three), else the same kernels on the same
  context. Added to the mon tokens after the op, just before the trunk.
- **Both readouts** build and run under static (`tower` and `trunk`), on both belief arms (`blob`, `fixed_mass`);
  the four `--debug` smokes are in §10.



In the fixed_mass arm a hidden slot holds a hypothesis species whose per-mon row is its DEX ROW
(`hypothesis_dex_rows.json`: the species' stats, types and ability prior, full HP, no status, never seen,
not active). Under `static` the whole token (S + D) of such a row reads nothing outside the row: no
board column, and no active-context scatter because a hidden slot is never the active. So:

- **once per forward**, the full static encoder runs over the dex table (S = 400 rows) → `[400, 128]`;
- **per (row, hidden slot)**, the token is a gather by the hypothesis species, then `hypothesis_marker`
  is added as today.

For T2's small batches (B · 6 < 400) the slots' own dex rows are encoded instead of the whole table (the
same static choice `hypothesis_encode` makes). The gather IS the per-row computation (the same function
on the same row), so the two agree to fp32 rounding of a different matmul batch shape; the test pins it.
The legacy split (`gathered_hypothesis_tokens`) is not used under `static`: there is nothing to split.

## 6. Parameters and compute vs today (stage 1; stage 2 in §6.2)

Production surface, CPU census (`learner_golden.build_learner`, `production_args()` ± the flag, 2026-10-06,
this commit). MACs are counted analytically from the built layer shapes (multiply-accumulates per decision
row, 12 mons × 4 moves).

| | legacy | static | Δ |
|---|---|---|---|
| per-mon encoder parameters | **161,536** (MEASURED) | **135,520** (MEASURED) | −26,016 (−16.1 %) |
| whole policy parameters | 3,065,882 (MEASURED) | 3,039,866 (MEASURED) | −0.85 % |
| move network (per move) | `Linear(105→96)→ReLU→Linear(96→32)` | `Linear(88→96)→ReLU→Linear(96→32)`: no clock / weather / faint / Spikes / HP / current-PP / legality columns | |
| role / S network | `Linear(410→256)→ReLU→Linear(256→128)` | `Linear(178→256)→ReLU→Linear(256→128)`: the 4 moves pooled to 32 (was 128 concatenated); no 60-dim active context, no 22-dim board broadcast, no dynamic columns | |
| D network | — | `Linear(134→128)→ReLU→Linear(128→128)`, plus `Linear(2→32)` (per-move state onto the move token) and `Linear(34→32)` (the PP pool) | +34,912 |
| encoder MACs per row, excluding the parts both share (move latent ≈ 0.28 M, move self-attention ≈ 0.21 M) | 2.28 M (move net 0.63 M, role 1.65 M) | 1.95 M (move net 0.55 M, S 0.94 M, D 0.40 M, per-move state 0.06 M) | −0.33 M MACs ≈ −0.66 MFLOP ≈ **−1.2 % of the 56.65 MFLOP forward** (ESTIMATED from shapes) |
| X5 hypothesis encoding, B = 2,048 | per (row, hidden slot) the split's remainder ≈ 94 k MACs × 6 ≈ 0.57 M MACs per row ≈ 1.16 G MACs per micro-batch; MEASURED 4.38 ms compiled fwd + bwd (`x5_hyp_gather_2026-10-05`) | the whole encoder over the 400-row dex table once ≈ 0.08 G MACs per forward + a gather, independent of B | ≈ **14× fewer MACs** (ESTIMATED); **the GPU time is DEFERRED** |
| X5 hypothesis encoding, T2's B = 8 bucket | ≈ 4.5 M MACs | the 48 slots' own rows ≈ 9.7 M MACs | ≈ 2× more at the smallest bucket (ESTIMATED; dispatch-bound there, X5 §3.6) |

The per-row cost falls little: our 6 mons and the revealed opponents are still encoded per row, because a
stateless forward has no per-battle cache. **S of a revealed mon is a pure function of its row's set
fields**, so an inference-tier cache keyed by those fields could skip it on every later decision of a
battle; not built (a T2 optimisation, its own unit). Stage 2's cost is §6.2 (MEASURED).

### 6.1 As built (stage 1, 2026-10-06)

- `agents/model/static_tokens.py`: `StaticTokenEncoder` (S, D, the per-move state, `parts()` for the
  tests), `static_hypothesis_tokens` / `encode_dex_rows` (the X5 table gather), `TOKEN_ENCODING_MODES`.
- `agents/model/belief_tables.py`: `build_static_stat_prior` (the 6-stat prior; its 5 non-HP columns ARE the
  operator's `SPECIES_SPREAD_PRIOR`, pinned by a test).
- `arch_constants.py`: `STATIC_STAT_SCALE` 500, `STATIC_DYN_HIDDEN` 128, `STATIC_MOVE_POOL_DIM` 32.
- The flag `--token-encoding {legacy,static}` on the five registry surfaces, `check_compatible`, a
  `_migrate_config` step (pre-v139 → `legacy`), config v139, no `ARCH_SIGNATURE` bump while both encodings
  build; the production mirror records `legacy`.
- Our active's move LEGALITY is matched to its sorted move slots by MOVE-NUM IDENTITY (`move_legality`).
  Legacy writes the request-order legality bits onto the sorted-by-id slots BY POSITION (a FINDING, §11).
- `selection_sites`: `static_tokens` is a declared forward module (two OBS reads, five integer ops).

### 6.2 Stage 2's cost (MEASURED, CPU, 2026-10-06)

Parameters: `tmp`-scratch census on `learner_golden.build_learner(production_args() ± the flags)`. FLOPs:
`torch.utils.flop_counter.FlopCounterMode` over one extractor forward (grad enabled, so the training-only latent
table is counted in every arm alike) on the first 64 rows of the K9 golden buffer, per row. Production surface
(blob, all 17 edge families on).

| | legacy tower | static tower | legacy trunk | static trunk |
|---|---|---|---|---|
| whole policy parameters | 3,065,882 | **3,016,008** (−49,874, −1.63 %) | 1,991,528 | **1,954,504** (−37,024, −1.86 %) |
| per-mon encoder | 161,536 | 135,520 | 161,536 | 135,520 |
| board projection(s) | `global_proj` 18,688 | `side_proj` + `field_proj` 2,944 | 18,688 | 2,944 |
| token-type table | 768 | 1,152 | 768 | 1,152 |
| op content | — | 4,352 | — | 4,352 |
| tower input width | 1177 | **1152** | (retired) | (retired) |
| trunk tokens | 61 | **63** (64 under X5) | 61 | 63 |
| extractor forward, MFLOP / row | 54.27 | **54.94** (+0.68, +1.25 %) | 57.48 | **58.32** (+0.83, +1.45 %) |
| of which the trunk | 35.87 | 37.15 (+1.27, +3.55 %) | 35.87 | 37.15 |

Stage 2 against stage 1's static (3,039,866): −23,858 parameters (the board projections −15,744, the tower's
input −12,850, the type table +384, the op content +4,352). The two extra trunk tokens cost +1.27 MFLOP per row,
about what stage 1's encoder saved, so `static` costs ≈ +1.3–1.5 % of the extractor's FLOPs net. The op content
reuses the `x` / `g` / `d1` cells the edge families already compute on the production surface (0 extra kernel
calls). **The GPU wall cost (`train_ms`, the T2 flush, `UpdateFit` headroom) is DEFERRED** (§10).

## 7. One lever? No: one bundle, by the owner's decision

The arm changes, at once: **F4** (board context out of the per-mon encoder), **F14** (the move set pooled
symmetrically), the **stats made static** (actual stats; the Smogon prior for the opponent), the **join**
(S + D instead of one MLP over everything), and, from the coverage audit, the **board layout** (three board
tokens, edges retargeted, the policy's board bypass deleted) and the **per-mon op content** on both sides. The
owner bundled the first four on 2026-10-06 (arch audit Decision record, "STATIC-TOKEN rebuild ... GO, spec
widened"; F14 folded "split if the combined screen disappoints"); the coverage audit makes the last two
requirements of the same arm (its B1 / B2), since without them a null indicts a missing route. So:

- a **pass** (equivalent or better) adopts the bundle; no single piece is credited;
- a **failure** is split in this order, each its own arm against legacy: (1) F4 + the board layout alone
  (legacy's encoder, its board columns zeroed, with the three board tokens), (2) F14 alone (legacy with a sum
  pool), (3) the stats construction alone. The order follows what is expected to cost most (F4 removes a
  route the model used; F14 and the stats re-encode the same facts).

## 8. The screen it will get

The plateau-first funnel (owner 2026-10-06, `project_plateau_first_funnel`): cheap equal-budget SCREENS
kill broken or clearly-worse ideas; the leading candidate gets a deep run to plateau later. This is a
SCREEN, one of the "behaviour changes, each screened at short equal budget" before the deep run (X26).

- **Arms:** `static` vs `legacy` at the SAME commit, both on the adopted belief arm (fixed_mass, per the
  owner's pre-committed X5 rule; both modes build so the screen can run either), the production recipe,
  equal step budget (the X5 A/B's 15M), **three seeds each** (the X5 replicate design).
- **Meter:** the mirrored head-to-head `main.h2h` at matched steps, and the h2h CROSS (`play-many`,
  each static seed against each legacy seed: 3 × 3 cells); secondary: the matched-wall-time read (X5
  Amendment 5's precedent), `train_ms`, the T2 flush, the critic's sibling discrimination and the belief
  purpose metrics.
- **Decision rule (to pre-register with the owner before the first seed):** adopt `static` on
  EQUIVALENCE (the delta's own CI inside a bar the owner sets, e.g. ±3 pp) or better; a pass also buys
  the X5 encoding cost win. A failure splits per §7.
- **Before the GPU:** the deferred checks of §10.

### 8.1 The screen's pre-registration — DRAFT, NOT REGISTERED (2026-10-06)

🚨 **A draft. Nothing here binds until the orchestrator registers it (with the owner's δ) before the first seed
launches; a launch from this block without that registration is not the screen.** It follows the X5 A/B's
registered design ([`design_x5_belief_tokens.md`](design_x5_belief_tokens.md) §7.4) wherever the two questions
are the same, so the two reads are comparable.

**Arms and order.**
- Two arms, `--token-encoding legacy` and `--token-encoding static`, at ONE commit P_st through the flag, on the
  production recipe (`--arch production`), the SAME belief arm and the SAME policy readout in both: the ones
  production records at registration (`belief_tokens`, `policy_readout` in `production_config.json`; if X5 is
  adopted first, `fixed_mass`; if F2's `trunk` is adopted first, `trunk`). The X26 ride-along heads as the X5
  A/B ran them. `--snapshot-ladder-games 0` in every run (X5 Amendment 1's reason).
- 15M steps, checkpoints every 1M from 10M (the speed rule needs them). **Three seeds per arm** (1001, 1002,
  1003; ids for bookkeeping only: seeds do not pair runs, X5 §7.4). Order alternates by seed (L1 S1 S2 L2 L3 S3).
- **Preconditions** (a run that breaks one is INCONCLUSIVE, replaced by the next seed id, never dropped): one
  commit for all six; `metadata.json` `init_num_threads` equal across all six; `data/` frozen while any arm is
  live; the `legacy` arm's K9 golden entries byte-identical to the production golden at P_st (this commit's
  identity proof, re-checked at P_st); the deferred GPU checks of §10 passed at P_st (the R1 compiled gate on
  `static`, a `--compile-trainer` smoke, the cost read inside budget).

**Primary meter: the mirrored head-to-head CROSS** (`python -m main.h2h play-many`, the Rust eval core). Each
static seed's 15M snapshot plays each legacy seed's: 3 × 3 = 9 cells, **1,000 mirrored pairs per cell** (≈ 2,000
games; meter SE ≈ 1.1 pp per cell). h_ij = static_i's win rate against legacy_j, in pp. The player keeps seat
p1 and the mirror swaps the teams, as `main.h2h` does.

**Statistic** (X5 §7.4's): Δ̂ = mean_ij(h_ij) − 50; row means R_i, column means C_j; V̂ = (s²_R + s²_C)/3 on
**df = 4**. t_NI = (Δ̂ + δ)/√V̂ (non-inferiority), t_SUP = Δ̂/√V̂ (superiority), and the two-sided 90 % interval
Δ̂ ± 2.132 √V̂ (the TOST equivalence interval at α = 0.05 per side).

**Margin δ: the OWNER SETS IT at registration.** The draft carries **δ = 3.5 pp** (≈ 24 Elo at 50 %), X5's margin,
so a pass means the same thing in both screens. One-sided α = 0.05, critical t₄ = **2.132**. Rule 8: a t within
1e-9 of 2.132 is NOT a crossing.

| outcome | rule | what follows |
|---|---|---|
| **INCONCLUSIVE** | an input is incomplete or invalid: a run short of 15M or out of restarts, a broken precondition, a cell with < 1,000 completed pairs, or timeouts > 25 % of attempted battles | repair and re-read; never interpreted |
| **BETTER** | t_SUP ≥ 2.132 | ADOPT `static` (subject to the speed rule) |
| **EQUIVALENT** | the 90 % interval lies inside (−δ, +δ) (the delta's own CI inside the bar) | ADOPT (subject to the speed rule) |
| **NON-INFERIOR** | t_NI ≥ 2.132 and neither of the above | ADOPT (subject to the speed rule); reported as non-inferior, never "equivalent" |
| **INFERIOR** | the upper one-sided 95 % bound Δ̂ + 2.132 √V̂ < −δ | legacy stays; the failure is SPLIT per §7 (F4 + board layout, then F14, then the stats construction), each its own arm against legacy |
| **NOT DETECTED** | none of the above (the interval straddles −δ) | legacy stays; back to the owner with the read: add seeds (the X5 extension: +2 per arm, a second look on a pre-declared boundary) or run the §7 split |

**Speed rule** (X5 §7.4's, unchanged): s = (median update-cycle wall of `static`) / (median of `legacy`) − 1,
pooled over each arm's seeds; s ≤ 5 % adopts on the rule above; 5 % < s ≤ 15 % ALSO requires non-inferiority at
MATCHED WALL-CLOCK (the same cross, each static run at its 1M checkpoint at or below 15M/(1 + s)); s > 15 % stops
before any meter read, as an optimisation unit. §6.2 predicts s small (+1.3–1.5 % extractor FLOPs; UNVERIFIED on
the GPU).

**Secondary (REPORTED, never gated):** `train_ms`, the T2 flush, the outside panel (frozen pool, bots, SmallRL)
with X5's HARM flag (static worse than legacy by > 2δ on the panel point estimate ⇒ the owner is told before
adoption), the critic's sibling discrimination, and, on a `fixed_mass` screen, the belief purpose metrics through
`main.belief_roles` (F-ST-2 predicts the T0 belief heads lose the board context: their reads are the place it
would show).

**Power — a FINDING for the registration.** ESTIMATED (noncentral t, not simulated), at Δ = 0 with three seeds per
arm and δ = 3.5: **0.40 / 0.27 / 0.18** at per-run SD σ = 2.5 / 3.43 / 4.8 pp (X5 §7.1's σ). δ = 5 pp: 0.62 /
0.43 / 0.28. Five seeds at δ = 3.5: 0.63 / 0.43 / 0.28. **Three seeds detect only a large harm**; a NOT DETECTED
read is the likely outcome when `static` is truly equal. The orchestrator chooses at registration between (a) this
single look as a HARM screen (an INFERIOR read kills or splits; anything else goes to the deep-run decision with
the read), (b) a wider δ, or (c) X5's sequential extension (looks at 3 / 5 / 8 seeds, O'Brien–Fleming boundaries,
≈ 41 GPU-h ceiling).

## 9. Literature relied on

- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov, Smola, *Deep Sets*, NeurIPS 2017: a function of a
  set is a sum-pool of per-element features followed by a network; sum keeps the cardinality.
- Lee, Lee, Kim, Kosiorek, Choi, Teh, *Set Transformer*, ICML 2019: self-attention among the elements
  before pooling (our within-mon move self-attention is one such block).
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017: additive composition of token and position
  embeddings, the precedent for S + D.
- Vinyals et al., AlphaStar, *Nature* 2019: per-entity embeddings into a transformer, global scalars in a
  separate stream. **UNVERIFIED:** whether AlphaStar's unit embeddings exclude global state the way this
  design does.

## 10. Build, tests, deferred checks

**Stage 1 tests that fail on revert** (`src/agents/model/static_tokens_test.py`, 12 tests; each mutation
below was applied and FAILED the named tests, then restored):

| mutation | fails |
|---|---|
| the four moves weighted by slot before the sum (order-sensitive, legacy's concat in effect) | the permutation-invariance test |
| D dropped from the token | the dynamic-state test |
| a board scalar (the clock) fed into the per-mon encoder | the board test, the identity-anywhere test, the gather test |
| HP fed into S | the identity-anywhere test |
| the dex-table gather encoding the rows as an ACTIVE mon | both gather tests (fp32 and fp64) |

Plus: `legacy` is the default and builds `PokemonEncoder`; the flag is recorded and read back; the opponent
stat prior's non-HP columns equal the operator's; our actual stats are the level-100 formula. The legacy
identity is the existing goldens' (K9 both entries, the compiled-region goldens, the obs goldens) at this
commit, unchanged.

**Stage 2 tests that fail on revert** (`src/agents/model/static_board_tokens_test.py`, 16 tests on real
golden-buffer rows through the golden learner's build, plus `projection_width_test.py`'s two static combos and its
static production case). Each mutation below was applied and FAILED the named tests, then restored:

| mutation | fails |
|---|---|
| THEIR side token reads OUR Spikes column | side relativity (swapping the sides' content must swap the two side tokens' content exactly, through one projection) |
| the alive and fainted counts dropped from the side content | side relativity, counts-reach-their-own-side, the refined-board / critic test |
| the op content added to OUR mons only | THEIR Spikes reach their mons only through `op_content` (with its control: at zero init NOTHING pre-trunk reads their Spikes), the every-mon test (a mon's trunk input moves EXACTLY where its `x` / `g` / `d1` cells are non-zero, and the outgoing cells never reach our mons) |
| the op content not injected | the same two |
| `x` written to OUR side's seat for their mons too | the edge-retarget test (static case) |
| `g` written to OUR side's seat | the edge-retarget test (static case) |
| `tower` keeps the `non_matchup_rest` bypass under static | the tower-bypass test (width 1177 − 25 and a NaN board tail must not reach `pi`), and the width mismatch fails the forward tests |
| `trunk`'s state query keys on the FIELD row three times | the trunk test (the board keys must BE the three refined board tokens, unmasked, in the trunk's seat order) |
| the critic's pool reads one board row | the refined-board / critic test (`board_rows` [B, 3, 128], no `global_row`) |

Plus: `legacy` builds no board token, no op content and keeps the bypass; `static` builds 15 base seats and a
9-row type table; the op content's zero init survives a REAL `MaskablePPO` build (`identity_init_test`'s path);
the static per-mon encoder reads none of the board block (all of `non_matchup_rest` + the five board slices
randomised ⇒ bit-identical tokens); `OpContent` refuses a missing `d1` cell; a pre-v140 `static` config is
refused and a pre-v140 `legacy` one stamps through.

**The legacy identity proof (stage 2).** At this commit, unchanged and green: the K9 learner golden BOTH entries
(`learner_golden_test.py` blob, `learner_golden_fixed_mass_test.py`), the compiled-region golden rows
(`compile_regions_golden_rows_test.py`), the obs goldens (no observation change), and the stage-1 static tests.
The legacy code path differs only by (a) the edge writes reading their seat from `board_seats` = (12, 12, 12)
(the same slice as the old literal), (b) the global token's type embedding added after the team tokens' (an
independent sum), (c) `key_log_presence`'s length read from `_total_tokens` (= 13, the old literal's value), and
(d) the assembler's `non_matchup_rest` appended to the same list; none changes an operand or an order of
accumulation, and no legacy module is constructed in a different order.

**`--debug` smokes** (CPU, the small-recipe workaround: `--allow-nonproduction-recipe --rollout-target-samples
3072 --batch-size 512 --grad-accum-steps 1 --n-epochs 2`, launched through a runpy wrapper so the trainer's file
name is in no argv), `static` × {`blob`, `fixed_mass`} × {`tower`, `trunk`}: `fixed_mass` × both readouts EXIT 0
(the round-trip smoke, four updates, the learner freeze's 8 checks, `Training complete`); `blob` × both readouts
stop at their third update on K9(b)'s excluded-share CEILING (F-ST-5: every row within fp32; the same smoke under
`legacy` passes at 0.13), and with `--behaviour-check warn` both EXIT 0 (four updates, 8 checks, `Training
complete`).

**DEFERRED to a GPU lease (stage 2 adds):** whether `static` × `blob` trips K9(b) at production sizing (F-ST-5);
the cost of the two extra trunk tokens in `train_ms` and the T2 flush; the compiled region / T2 graph with the
board tokens and the op content.

**DEFERRED to a GPU lease (both stages):** the compiled (Inductor) forward and backward on CUDA (the R1
startup gate, `--compile-trainer`), the T2 compiled graph, the cost read (`train_ms`, T2 flush, `UpdateFit`
headroom) against legacy at the same commit, and the X5 hypothesis-encoding time under `static`.

## 11. Findings

- **F-ST-1 (legacy, a reading defect, UNFIXED here because legacy must stay byte-identical):**
  `PokemonEncoder` writes our active's legality bits (`our_active_req_move_legal`, REQUEST order) onto its
  move slots, which are SORTED BY ID, by POSITION. Wherever the request order differs from the id order, a
  move's "legal now" input belongs to another move. The pointer head and the operator use the identity
  permutation (`_request_order_move_tokens`), so the action logits' masking is correct; only the move
  network's validity INPUT is misaligned. Its comment calls the input "correctly aligned to the slot it
  gates". `static` matches by identity. **MEASURED** on the K9 golden buffers (real Rust-collector rows,
  2026-10-06): the two orders differ on 60 of 60 blob rows with a request (53 of 55 fixed_mass), but the
  legality bits are almost always all-ones, so a move reads a WRONG legality bit on 2 of 64 blob rows (1 of
  64 fixed_mass): the rows where some move is illegal (a Choice lock, Taunt, Disable, no PP). Small, and
  exactly the rows where legality matters.
- **F-ST-2:** under `static` the T0 belief heads lose the board context (§4, last paragraph).
- **F-ST-3:** the opponent's stats in S are the Smogon PRIOR, never the learned spread belief (§2: circular);
  the belief reaches the physics only.
- **F-ST-4:** the item-CHANGED fact (Trick / Knock Off) has no per-mon observation column; S follows the
  changed item id and D's bit is `consumed` only. The event window carries the transition.
- **F-ST-5 (stage 2, K9(b), MEASURED on the CPU `--debug` smokes, 2026-10-06):** `static` × `blob` trips K9(b)'s
  excluded-share CEILING at its third update, deterministically (the rerun reproduces it): 0.179 (`tower`) / 0.234
  (`trunk`) of the probe's current-version rows sit within the 2e-4 relative margin of a cutoff, against the 0.15
  ceiling; the full buffer reads 0.086 / 0.092. No row is wrong: every judged row is within fp32 (max |Δ log π|
  2.4e-7). The `legacy` control on the same recipe passes, but close to the ceiling (0.127 / 0.129 / 0.083);
  `static` × `fixed_mass` reads 0.01–0.03 (it has no such cut). The site is the E5 tail cut
  (`pointer_head.py` `w_all.topk(K)` / its `>=`): at the dumped update-4 weights the near-ties are between two
  hidden slots' BP-0 candidates (Protect 182 vs Toxic 92, w ≈ 0.1577 on three hidden slots of one row), where the
  cut's outputs are inert to the swap (both scores are 0, so `worst_*` cannot move, and `p_tail` moves only by the
  candidates' own gap). With `--behaviour-check warn` all four static smokes complete (§10). Not fixed here (K9(b)
  machinery, out of this unit's scope): the class fix is an identity clearance for the E5 cut (two candidates
  with equal BP × accuracy × category payload are no tie), the `gen3_behaviour_tie_identity_v1` / `_consumed_v1`
  precedent. Whether production sizing trips it is UNVERIFIED (a GPU check); the screen on `fixed_mass` does not
  read the site.
- **F-ST-6 (stage 2):** the two side tokens share one projection, so a fact that is NOT symmetric between the sides
  is still one column read for both: "revealed count" is always 6/6 on our side (a constant there), and alive =
  hp > 0 on ours but `opp_addressable` (unrevealed = alive) on theirs. Both are the true value of the same fact
  from our viewpoint; what the OPPONENT has seen of us (audit B7) is a separate new observation fact.
- **F-ST-7 (the screen, §8.1):** three seeds per arm give an ESTIMATED power of only 0.27 (σ = 3.43 pp, δ = 3.5 pp)
  to show non-inferiority when `static` is truly equal; the registration must choose a harm screen, a wider δ or
  X5's sequential extension.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-06 | Scope of "static" | species + set (item, ability, moves as a set, stats from the spread) + the reveal bits + the Hidden Power belief; dynamic per-mon state (HP, status, counters, sleep belief, recency, protect, last action, trap, active, item-consumed, own boosts and volatiles, PP and legality) separate | static species only (the audit's narrower L1); every state its own token (audit option B) | owner 2026-10-06 (audit Decision record) |
| 2026-10-06 | The join | ADDED: `S + D` | a second token per mon (+12 trunk tokens); FiLM gating | §3 |
| 2026-10-06 | The move set | the move network → within-mon self-attention → SUM | mean (loses the count of known moves); attention-pool (more parameters, no measured need) | §1, Deep Sets |
| 2026-10-06 | Opponent stats | the Smogon usage-weighted mean + std of each realised L100 stat, built from `data/` at construction | the learned `SpreadBelief` (circular: it reads S's tokens); base stats only (not "actual stats") | §2 |
| 2026-10-06 | Board context | out of the per-mon path. Its home, from the entity-coverage audit: OUR SIDE + THEIR SIDE (one side-relative projection) + FIELD replacing the global token, `x` → own side, `g` / `c4` → field, the critic reads all three, the policy's board bypass deleted, and the operator's per-mon amounts as content on BOTH sides; all part of THIS arm (stage 2) | one global token (fails counts, side relativity, amounts: audit §4); a token per condition (trunk cost); the audit's new obs facts B5–B9 inside the arm (a separate later lever); the status-landing fix B4 here (F11's unit) | §4; `design_entity_coverage_audit.md` (`f32a9f4b`); coordinator 2026-10-06 |
| 2026-10-06 | Build staging | ship stage 1 (S + D, the gather, the flag) now, stage 2 (§4) next, by handoff; `static` is NOT screen-ready until stage 2 lands | building both in one unit (it would cross the parallel F11 / F2 / K9(b) edits to the trunk and the readers) | the brief's "first coherent slice" rule |
| 2026-10-06 | Legality onto sorted move slots | matched by MOVE-NUM IDENTITY under `static` | legacy's positional write (F-ST-1) | §11 |
| 2026-10-06 | Observation layout | UNCHANGED: everything S and D read is already in the 2761-dim observation | an obs change (would carry the obs benchmark, Rust encoder parity and goldens) | §1 |
| 2026-10-06 | Stage 2 token types | a 9-row type table under static (OUR SIDE 6, THEIR SIDE 7, FIELD 8); legacy's stays 6 rows | re-using the GLOBAL row for FIELD (a type id whose meaning depends on the encoding); one type for both sides (side relativity would then need a learned side input) | §4.1; legacy byte identity |
| 2026-10-06 | Stage 2 SIDE content | the 10 side-relative facts of the CURRENT observation (Spikes, 4 screen presences, Wish, alive / fainted / revealed counts, Sleep Clause used = a non-Rest sleeper on the side); alive from the per-mon slots (unrevealed = alive), Sleep Clause the op's own read | the audit's turns-left / Future Sight / Freeze Clause columns (new obs facts, B5 / B6: a separate lever); alive = 1 − fainted (a linear duplicate) | §4.1; brief "no new obs facts" |
| 2026-10-06 | Stage 2 OPC shape | two zero-init projections: `amount_proj` (x ⊕ g, 8 → 128) shared by BOTH sides, `outgoing_proj` (our 4 request-order moves' d1 cells, 24 → 128) on THEIR mons; our incoming rows stay on `prefuse_proj` | one projection over a zero-padded union per side (a side-specific padding the model must learn to ignore); a reduction of d1 over our moves (loses which move does what, and the pointer reads per-move) | §4 A3; audit B2 |
| 2026-10-06 | Stage 2 readers × readout | `tower`: the `non_matchup_rest` concat deleted (pi 1177 → 1152); `trunk`: the state query keys on the three refined board tokens; the critic's `full` pool takes the three board rows under source tag 3 | keeping `non_matchup_rest` under `trunk`'s absent tower (nothing to delete there); a new source tag per board token (the trunk's type embedding already tells them apart) | §4.1; F2 interaction |
| 2026-10-06 | Stage 2 versioning | config v140, no field: a pre-v140 `static` record (stage 1's layout) is REFUSED, `legacy` stamps through; no ARCH_SIGNATURE bump while both encodings build | migrating a stage-1 static checkpoint (no home for `global_proj`; nothing was trained on it) | `designs/model/versioning.md`'s playbook |
| 2026-10-06 | The screen | DRAFTED in §8.1 (static vs legacy, one commit, 15M, 3 seeds each, the mirrored h2h 3 × 3 cross, X5's cross statistic, δ for the owner, BETTER / EQUIVALENT / NON-INFERIOR adopt, INFERIOR splits per §7); NOT registered | registering it from the build agent (the orchestrator registers before any seed) | the brief |
