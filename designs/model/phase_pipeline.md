# The phase pipeline — the chain, the tiers, and what each phase does

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

> 🚨 **[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §2.1 is the live statement of the chain and the
> tier table.** This doc holds the reasoning around it. **STALE:** step 2 below says the observation
> is 2,667 dims (`gen3_entity_rehome_v1`); the live figure is **2845** — `ARCHITECTURE.md` §1, read
> from `Gen3ObservationEncoder.get_layout()`. The text is preserved as it was written.

## The chain, and the four tiers

`forward_internal` is decomposed into phase `nn.Module`s, chained by a thin orchestrator, in ONE
order — the TIER ORDER (`gen3_tiered_pipeline_v1`). There is no placement flag and no second chain:

`ObsUnpack` → `PokemonEncoder` → `[HypothesisBuilder?]` → `[MoveBelief?]` → `[SpreadBelief?]` →
`[HPTypeBelief?]` → `[DamageOperator?]` → `prefuse_proj` residual → `EntityMoveSeats` → edge cells →
`TeamTransformer` → `[BeliefHead?]` → `CLSPool` → `[FlatIntentHead?]` → `[side readouts?]` →
`ProjectionAssembler`, then ONE root head for the policy (`pre_proj_norm`/`projection` → `ReLU`); the value
half is `value_pooled` itself, with no projection (architecture audit F1, the version break part 2).

Grouped into the four tiers the contract asserts:

| tier | question | modules |
|---|---|---|
| **T0 RESOLVE** | what is on the board? | `pokemon_encoder` (`PokemonEncoder`; under `--token-encoding static` it is `StaticTokenEncoder`, S + D with no board input, and the X5 hypothesis tokens are its dex-table encoding gathered, `static_tokens.py`), `t0_species_prior`, `move_belief`, `hp_type_belief_head`, `spread_belief`, `item_belief_head` (opt-in), `hypothesis_builder` (X5, built with the opponent-belief family — the only belief representation since the X5 version break, v144: its SPECIES half runs before `move_belief` — the hidden opponent slots then take THE `pokemon_encoder`'s token for their hypothesis's dex row + `hypothesis_marker` (computed by `hypothesis_encode.py`: the encoder's species half once over the dex table, gathered, + its row-level half — the exact split; `hypothesis_tokens.py`; the blob path's `belief_slots` token is deleted) and the T0 belief heads read them with their species — its move group after `move_belief`; OTHER_species enters the trunk as one extra seat and every opponent key carries log π in the trunk and the class-E pools; the active's move group gives the E4 seats, the op's seat axis and its class-M weights) |
| **T1 REASON** | what follows from it? | `damage_op`, `entity_seats`, `history_events` (H-B event seats, opt-in), `edge_bias`, `team_transformer` |
| **T2 DECIDE** | what will they do, what are my moves worth? | `belief_head`, `cls_pool` (which also owns the two token-content critic injections), `flat_intent_head` (X5's flat opponent pointer; the blob α / β heads are DELETED, not constructed), `intent_threshold_move` / `intent_conditional` / `pair_outcome_move` / `pair_outcome_switch` / `switch_branch` / `conditional_threat` (opt-in) |
| **T3 DELIVER** | one contract, two pools | `hidden_opp_belief`, `assembler`, `win_head`, `policy_query` (`--policy-readout trunk` only, audit F2) |

**The ordering is an ASSERTED INVARIANT, not a convention** — `tier_contract.py` declares a tier per
module and `tier_contract_test.py` runs a real forward under instrumentation, checking (a) tier
entries are non-decreasing within a forward and (b) no entry point receives a tensor whose STORAGE
was produced by a strictly later tier (checked over two forwards, so a stale stash counts; keyed on
storage so `.detach()` and views do not hide it). Every `nn.Module` child must declare a tier or be
listed in `UNTIERED_CHILDREN`, so a new phase cannot escape the contract by omission. Both checks
are proved falsifiable by planted-violation tests. **What it cannot catch:** a T0 leg *recomputing*
something intent-like from raw tokens — that is a semantic judgement, and the contract is a
data-flow check. It buys that such a head could not then be FED the real α, and could not run out
of order.

## The T0/T1 belief + physics stack

**`t0_species_prior` (v72) is the case that shows why the tier map earns its keep.** The SAME
team-composition species belief exists in two places, and the tier is the entire difference between
useful and unreachable. At T2, inside `BeliefHead`, it is a training-only readout the physics cannot
see — which is why the `DamageOperator` priced every unrevealed opponent from the static
`SPECIES_USAGE_PRIOR` frequency table for as long as it did, with the op's own `species_probs`
override sitting unused since the day it was added. Declared T0, the same computation is a resolve
step the op consumes directly. The math lives once, in `t0_species.species_team_prior_logits`, which
`BeliefHead.species_prior_logits` also calls. When the flag is on, the belief is resolved ONCE in
`forward_internal` and the same tensor goes to all three unrevealed-defender sites (the op block,
the `d1` cells, `pairwise_boost`) — the gate asserts tensor identity, because two
equal-but-separately-computed tensors is exactly how the "bias and concat can never disagree"
invariant stops holding without anything failing. Parameter-free, so OFF is byte-identical and the
version check is the only thing that can reject a mid-run flip.

`HypothesisBuilder`/`BeliefHead` are built only when `opp_belief_slots` (`--opp-belief-aux-coef>0`),
`MoveBelief` only when `move_belief_mode != off`, `DamageOperator` only when `damage_op` (which requires
`move_belief_mode` revealed/both); with all off the chain is the baseline `ObsUnpack →
PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler` byte-for-byte. X5's hypothesis tokens fill
the un-revealed opp slots with the encoded dex row of their hypothesis species *before* the transformer (so
the belief is refined in-lineup; the blob path's learned `BeliefSlots` token is deleted, v144); `BeliefHead` reads the refined opp tokens *after* the transformer and stashes the
species/moves aux logits (a side readout — does NOT feed forward). **That T0/T2 split of the species
belief is deliberate and stays**: `BeliefHead` is a training-only side readout, not a second resolve
path, and its T2 declaration is what records the fact — if it ever started feeding a T0/T1 consumer
the provenance check would fail. `MoveBelief` predicts + **reinjects** the moveset into the opp
**role** tokens *before* the transformer (so the believed moves co-refine through attention, and every
T1 consumer reads one posterior computed once); `DamageOperator`
runs *after* `MoveBelief` and consumes its predicted-move logits to compute the believed-move incoming
damage to each of our mons. Its per-our-mon incoming rows are added to our role tokens through the
zero-init `prefuse_proj` (built whenever `damage_op` is), so attention reasons over the physics. **`MoveBelief`'s Smogon prior is LEGALITY-GATED unconditionally**
(`gen3_unconditional_move_legality_v1`, v65): `build_move_prior_logits` drives every
`(species, move)` the species cannot learn to `_ILLEGAL_PROB` 1e-6, so the belief can no longer
invent "this special attacker might be holding Explosion". Three cases, and keeping them apart is
the whole point — **`floor` is the LEGAL-UNOBSERVED base, never an on/off switch**:
| case | prior | meaning |
|---|---|---|
| species has a learnset, move NOT in it | `_ILLEGAL_PROB` 1e-6 | **impossible** |
| legal, absent from usage data | `floor` (`_PRIOR_FLOOR` 0.02) | unlikely but liftable by evidence |
| legal, with recorded usage | its TRUE Smogon rate | no rarity cap — a surprise tech survives |
| **no learnset at all** (unknown species / num 0) | `floor` everywhere | nothing known ⇒ everything stays POSSIBLE |
That last row is a correctness invariant, not a default: *"not known to be illegal"* must never
collapse into *"known to be illegal"*, or the belief asserts an unidentified opponent can do
nothing. `logit(0.02) = -3.89` vs `logit(1e-6) = -13.8` is a **9.92-nat** gap, and a floor at or
below `_MIN_PRIOR_FLOOR` (1e-3) is a hard `ValueError` — the collapse is unrepresentable rather
than merely unlikely, because a collapsed floor silently turns the legality gate into the rarity

prune that previously crippled surprise-move anticipation.

## `--belief-grad-mode` — what each mode actually cuts

`detached` stop-grads the head's trunk **read** (`detach_read`), so the belief cannot reshape the
trunk. It does **not** stop PPO training the heads — the reinject write stays live, deliberately
(`belief_grad_mode_test::test_detached_preserves_normal_trunk_training`). `label_only` stop-grads
the head's **output** at its publish boundary (`publish_detach` inside a head, `_publish_belief` on
the extractor), so the belief is trained by its labels alone while the policy still reads it. The
read stays live under `label_only`, because cutting B and C together leaves a probe on a trunk with
no incentive to encode hidden state — still feeding the policy. That combination is not offered.

### Scope, and why the mode is resume-immutable rather than weight-shape

Scope is the four heads with a forward path: `MoveBelief`, `SpreadBelief`, `HPTypeBelief`, and
X5's flat opponent pointer (`FlatIntentHead`, cut at its publication boundary — the deleted blob
`AlphaIntentHead`'s role; since the critic-route deletion wave took every α→vf route, α reaches the
objective only through the POLICY, via the pointer cells). `BeliefHead` is
structurally label-only in every mode — asserted in `belief_label_only_gate_test.py`, not assumed,
so a head that starts feeding forward fails a test instead of quietly rejoining the PPO objective.
🚨 **`WinProbHead` is NOT in that set under the win-prob critic (the only critic)**: there the head IS `_critic_value`,
so it feeds GAE, the value loss and (at `win_prob_mode shaping`, which the mode implies) the trunk.
Under the shaped critic (a pre-break checkpoint, run PINNED; not constructible at HEAD) it was label-only. The claim is mode-conditional, and
reading it as unconditional would say the production critic cannot reach the objective.

`detach()` is value-preserving ⇒ the forward is bit-identical in all three modes ⇒ this is a
resume-immutable training hparam (the `vf_coef` class), NOT weight-shape: no `ARCH_SIGNATURE` bump,
excluded from `check_compatible`, enforced resume-only by `check_belief_grad_mode`
(`--allow-belief-grad-mode-change` for an intentional migration). `BELIEF_GRAD_MODES` in
`features_extractor.py` is the single source for the legal set.

## Projection widths, and why `vf` is a constant `D_MODEL`

The embedding tables live in a shared `Embeddings` module passed as a forward argument to the
phases that need them, so they register exactly once. An immutable `ExtractorContext` produced
by `ObsUnpack` carries the ~30 unpacked tensors downstream, keeping each phase's signature
narrow. Both projection input dims are STATIC ARITHMETIC (`gen3_static_widths_v1`):
`compute_projection_widths(layout, opp_belief_cls_k=…)` in `features_extractor.py` mirrors
`ProjectionAssembler.forward`'s concat exactly. **`vf` is a CONSTANT `D_MODEL`** — the
critic-route deletion wave retired the whole post-assembler vf tail (the seed window; the
hidden-opp belief's vf half; the `non_matchup_rest` vf concat), so `vf_combined IS value_pooled`,
the same tensor the critic reads — the win head under the win-prob critic (the only critic). That is the structural cure for the v89/M2
orphaned-branch class rather than another instance of it: there is no second vf path left for a
critic parameterization to bypass. Only TWO inputs still move `pi`: the layout's
`non_matchup_rest` tail, and the hidden-opp belief pool (`k·D_MODEL`, **policy side only** — its
vf half read dV 0.0000 while its pi half flipped 39.6% of argmaxes, so the deletion had to be
per-head). Every other flag is width-neutral by construction: the v89 value routes inject
ADDITIVELY into `value_pooled`, and the intent cells widen the pointer stash, not pi/vf.

> 🚨 **The old construction-time DISCOVERY forward is DELETED — its job is now a TEST.**
> `__init__` used to measure the widths by running a dummy `forward_internal` with
> `_intent_reduce_discovering` zero-fill branches threaded through the runtime forward; that
> mechanism shipped the ede5a88 bug class (a discovery branch's early `return` hid every vf part
> appended below it — the critic was built 128 dims short and died on the first real forward,
> only when both flags met). `projection_width_test.py` is the old mechanism preserved as the
> new mechanism's verifier: it builds production / all-routes-on / minimal / targeted flag
> combos, runs a REAL forward each, and asserts the measured concat widths equal the arithmetic.

## Phase by phase

1. **`Embeddings`** — shared tables: species (32), move (16), item (16), ability (16), type (16,
   shared for Pokémon types, move types, and TurnDelta move/type IDs). Owns the Hidden Power
   soft-type blend (`hp_soft_type`) and the per-slot TurnDelta embedder (`embed_delta_slot`).
2. **`ObsUnpack`** (stateless) — peels the flat observation (2667 dims under
   `gen3_entity_rehome_v1`) into the named tensors of `ExtractorContext` via the declarative
   schema's validated slice map (`build_schema(layout).slices()` — the tiling proof runs at
   construction): per-Pokémon block + categorical IDs, the global/board feature slices, and
   (hoisted here) the active-slot indices + fainted key-masks used downstream.
3. **`PokemonEncoder`** — embeds + stitches the enriched per-Pokémon vector; runs the **shared
   move processor** (Linear→ReLU→Linear, `MOVE_NET_HIDDEN`) over every move slot (input:
   move/type embeddings, remnants, known flag, battle context, HP-candidate distribution, and
   prev-turn move validity — the CPU matchup ×6 / validity ×6 inputs are DELETED with their obs
   block, `gen3_entity_rehome_v1`), a
   **within-Pokémon move self-attention** (MHA 32-dim, 2 heads, + LayerNorm residual), then the
   **role encoder** (Linear→ReLU→Linear, `ROLE_ENCODER_HIDDEN`) → 12 × 128 role tokens. The role
   input carries the **E2 active-context injection** (gen3_entity_rehome_v1): each side's
   boosts+volatiles block scattered onto its ACTIVE mon's row (bench rows zero) — the entity owns
   its own ctx; the global-token/projection routes remain (additive). Pinned by
   `e2_ctx_injection_test.py`.
4. **`TeamTransformer`** — builds a **13**-token sequence (6 our-team + 6 their-team role tokens
   + 1 global token). `gen3_frame_deletion_v1` deleted the `N_HISTORY_TURNS` history seats along
   with the lag frames that fed them (20 → 13), and that count is load-bearing rather than
   cosmetic: every edge family addressing the GLOBAL seat indexes it as `2·TEAM_SIZE`, and every
   `extra` seat (E3/E4, the H-B event seats) as `_total_tokens + k`, so a stale count writes whole
   families onto the wrong tokens instead of raising. Adds token-type embeddings, and runs a
   `TRANSFORMER_N_LAYERS`-deep `nn.TransformerEncoderLayer` stack (d_model
   128, `TRANSFORMER_N_HEADS` heads, FFN `TRANSFORMER_FFN_DIM`, post-LN) under a key-padding mask
   that masks fainted team slots. The global token comes from the two active-contexts +
   non-matchup scalars; `embed_delta_slot` and `history_proj` are deleted.
   **Under `--token-encoding static` (`gen3_static_board_v1`, stage 2)** the global token is replaced by
   THREE board tokens — OUR SIDE (12) and THEIR SIDE (13) through ONE shared `side_proj` over side-relative
   content, FIELD (14) through `field_proj` (`board_tokens.py`) — so the base is 15 (`_total_tokens`), every
   extra seat shifts by 2, and the edge families read their board seat from `board_seats` ((12, 12, 12) under
   legacy: `x` → the mon's own side, `g` / `c4` → FIELD under static). Just before the trunk the per-mon
   OP CONTENT (`op_content`, T1, zero-init) adds the `x` ⊕ `g` amounts on every mon and our `d1` cells on
   their mons (a set function of our moves), and under `--mon-hazard-cost on` each mon's own side's Spikes layers
   + its switch-in cost (`static_facts.py`, the op's `spikes_entry`); under `--move-actor-state on` the E3 seats
   carry our active's HP + status (both zero-init, `gen3_static_port_v1`); under `--eot-residual on` every mon (either
   encoding) gets its end-of-turn HP change if it is on the field then (`eot_residual.py`, zero-init,
   `gen3_static_recovery_v1`). Readers take `TeamTransformer.board_rows()` ([B, 1, D] legacy, [B, 3, D] static).
   **`--trunk-layers N > 2`** appends `N − 2` identity-init PRE-LN rounds (`trunk_depth.IdentityInitRound`,
   `team_transformer.extra_rounds`) after the two post-LN layers, on the same shared bias.
   Returns the two refined team-token blocks. **Optional gradient checkpointing**: a runtime
   `grad_checkpointing` flag (set per run by `train_rl_agent.py --grad-checkpointing`, never
   saved/version-checked) runs these encoder layers under `torch.utils.checkpoint(...,
   use_reentrant=False)` during the backward-needing pass — **bit-exact** (dropout=0.0), trading
   one extra forward on the otherwise-idle GPU for the layers' ~5 GB of activation VRAM at
   batch 16384. A no-op under inference (gated on `torch.is_grad_enabled()`), so eval / the
   self-play opponent forward pay nothing.
5. **`CLSPool`** — one learned CLS query per side cross-attends over its 6 post-transformer team
   tokens (fainted slots key-masked) → a 128-dim pooled team token per side (+ LayerNorm). Also
   extracts `our_active_refined` = the transformer output of our active slot. A **third learned
   query, `value_cls`**, cross-attends over **all 12 team tokens** (both sides, fainted
   key-masked) → a 128-dim global `value_pooled` summary — a whole-board "who's winning" read for
   the critic, a different aggregation than the policy's our-active-centric pools.
5b. **`HiddenOppBeliefPool`** *(optional — built only when `--opp-belief-cls-k > 0`)* — **k** distinct
   learned query tokens run through a `TransformerDecoderLayer` (self-attention among the queries to
   coordinate + cross-attention to the 12 team tokens under the single-sourced `ctx.all_fainted`
   key-mask) → a `[B, k·D_MODEL]` hidden-opponent belief. `None` when `k=0`. See the v9 toggle note
   under *Model versioning* and `designs/ai_v5/design_offense_and_opponent_belief.md` §B2.
6. **`ProjectionAssembler`** — emits a `(pi_combined, vf_combined)` pair. Policy: `our_pool(128)
   + their_pool(128) + our_active_refined(128) + non_matchup_rest`. Value: `value_pooled(128)` alone
   (`vf_combined IS value_pooled`).
   **`gen3_ctx_dedup_v1`: the per-side encoded active contexts are DELETED from both heads** —
   they were duplicated delivery with a 1:1 entity-native replacement already live (the E2
   injection puts each side's FULL raw ctx block on its active token; the global token is a
   second route). `non_matchup_rest` stays: the global token is its only other route and no
   pool reads that token directly, so the concat is currently its one direct head path. **Under
   `--token-encoding static` the `non_matchup_rest` concat is DELETED** (`board_bypass`; pi width −25): the
   board reaches the policy only through the board tokens, the edges and the cells. When
   the hidden-opponent belief is on, its `[B, K·D_MODEL]` is appended to the POLICY half (last),
   widening the projection input by `k·D_MODEL`.
7. **Root head** — ONE `pre_proj_norm` (LayerNorm) → `projection` (Linear) → `ReLU` on `pi_combined`,
   emitting `PROJECTION_DIM`; the value half passes through untouched as `value_pooled` (`[B, D_MODEL]`,
   `vf_features_dim`). `Gen3DualHeadMaskablePolicy` builds an ACTOR-only `MlpExtractor` (`net_arch` = the
   actor widths, `vf=[]`) and feeds the policy half to `forward_actor`; the critic is
   `_critic_value(vf)` = `sigmoid(fe.last_win_prob_logits)` (the win-prob head on `value_pooled`), and no
   value tower exists (architecture audit F1).
8. **`--policy-readout trunk` (`gen3_policy_readout_trunk_v1`, config v138, audit F2; OFF in production)**
   — the policy half of steps 6–7 and SB3's actor branch are RETIRED: `PolicyStateQuery` (`pools.py`,
   T3) is one learned query, 4 heads, over every refined trunk token (our 6, their 6, the global token,
   the entity and event seats) plus `HiddenOppBeliefPool`'s K outputs, under the trunk's own key mask
   (+ its per-key log π under `fixed_mass`) → LayerNorm → `[B, D_MODEL]`, and that IS `pi_features`
   (`forward_internal` returns it in place of `pi_combined`; `forward` applies no projection).
   `pre_proj_norm` / `projection` and `mlp_extractor.policy_net` are still BUILT and see the policy's orthogonal
   re-init (SB3's order) (so no surviving module's initial bytes move), then `Gen3DualHeadMaskablePolicy._build` drops
   them (`ExtractorApi.retire_policy_tower`; `policy_net` becomes the empty `Sequential`, the identity)
   before the optimizer is made, and sizes the pointer head from `policy_ctx_dim` (D_MODEL) with
   `TRUNK_POINTER_HIDDEN` scorers. The query is built from `POLICY_QUERY_INIT_SEED` inside `fork_rng` out
   of `IsolatedLinear`s. The value half is unchanged. Gate: `policy_readout_test.py`.

---

## Moved from the leaf (2026-10-10)

The full text of the model leaf's sections on this topic, moved here when `src/agents/model/CLAUDE.md` was cut to rules, commands, map and hazards (the leaf keeps a one-line pointer to each). Always-current like the rest of this doc; where it overlaps an earlier section, the earlier section is the fuller statement.

### Phase module structure

**[`designs/ARCHITECTURE.md`](../ARCHITECTURE.md) §2.1 states the chain and the four
tiers as they are now.** Three rules about it live here:

- **The TIER ORDER is an ASSERTED INVARIANT, not a convention** (`gen3_tiered_pipeline_v1`).
  `tier_contract.py` declares a tier per module; `tier_contract_test.py` runs a real forward and
  checks tier entries are non-decreasing and that no entry point receives a tensor whose STORAGE was
  produced by a strictly later tier. **Every `nn.Module` child must declare a tier or be listed in
  `UNTIERED_CHILDREN`** — a new phase cannot escape by omission. It is a data-flow check, not a
  semantic one.
- **There is no placement flag and no second chain.** With every optional block off the chain is the
  baseline `ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler`
  byte-for-byte.
- 🚨 **`gen3_no_concat_v1` (v61): the op's flat block enters NEITHER projection.** The op reaches the
  policy through the pointer cells + the `prefuse_proj` injection + the edge cells, and the critic
  through `UnifiedValueReadout`. The TRUNK/concat route is DEAD; adding one back is a new
  architecture decision, not a restoration.

#### `--belief-grad-mode` — which arrow gets cut (`gen3_belief_grad_mode_v1` / `gen3_belief_label_only_v1`)

**The two non-default modes cut OPPOSITE arrows, and the flag name does not say so** — read this
table before reasoning about either. Four routes exist between a state-prediction belief head and
the rest of the network:

| | route | `shaping` | `detached` | `label_only` |
|---|---|---|---|---|
| A | label loss → belief head params | on | on | on |
| B | label loss → shared trunk (the head's READ) | on | **CUT** | on |
| C | PPO loss → belief head params (the WRITE) | on | on | **CUT** |
| D | PPO loss → shared trunk (normal training) | on | on | on |

**Two rules when touching this:**

1. **A supervised loss reads `belief_supervision(name)`, never the `last_*` attribute.** Under
   `label_only` the attribute is the stop-grad publication, so a loss reading it trains *nothing*
   — silently, since the loss value and every metric derived from it look normal. The accessor
   raises on an unknown key so a typo cannot degrade into that.
2. **Detach the LOGITS, never the matmul output.** `soft_emb = sigmoid(logits) @ move_embedding.weight`
   — detaching `logits` keeps `move_embedding.weight`'s gradient, detaching `soft_emb` kills it.
   That table also trains from `PokemonEncoder`, so the damage would be an invisible slowdown, not
   a dead parameter. Same shape in `HPTypeBelief.reinject` (`hp_soft_type` → `type_embedding`). The
   reinjection adapters have no supervised loss, so PPO is their ONLY gradient source.
#### Phase-by-phase data flow
The per-phase walkthrough and the static-width arithmetic:
[`designs/model/phase_pipeline.md`](phase_pipeline.md). Two things stay here.

> **When you add a width-contributing part**: extend `compute_projection_widths` in the same
> pass and add the flag to the sweep — a wrong width for any combo fails in the suite, not at a
> production launch. (A new additive `value_pooled` route needs no width change at all — see
> `_value_pooled_routes`, whose runtime RAISE guards stay.)

Rules to preserve:

- **Each phase owns its layers** (`move_network` lives under `pokemon_encoder`, `our_cls` under `cls_pool`, etc.). State_dict keys are therefore phase-prefixed.
- **`Embeddings` is the sole owner of the 5 embedding tables + `hp_type_idx_map`.** It is passed as a **forward argument** to `PokemonEncoder` and `TeamTransformer` — never stored as a child attribute on them — so the tables register exactly once. (The root exposes read-only `@property` forwarders like `model.type_embedding` for convenience; those add no state_dict keys.)
- **`ExtractorContext`** (frozen-by-convention dataclass) is the inter-phase contract: `ObsUnpack` produces it, downstream phases read from it. Add a field here rather than widening a phase's positional signature. Cross-phase values (active-slot indices, fainted masks, `hp_probs`) are computed once in `ObsUnpack` and carried on the context.
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version/constants.py`. **Read the live value there, not from prose.** Three cases people get wrong:
  - A **pure decomposition** still changes state_dict keys, so old checkpoints must fail loudly — bump it.
  - A forward-math change with **unchanged `out_dim` / projection widths** is not shape-caught by anything, so the signature bump is the ONLY thing that rejects a stale checkpoint. This is the case that has bitten most often.
  - **Re-sourcing or re-meaning an obs block** is retrain-class even when no individual dim moves (a constant fallback becoming a real value; a scalar's definition changing; a block moving from `available_moves` order to request-slot order). The long list of historical examples lives in `designs/CHANGELOG.md`.
- Per-phase unit tests live in `phase_modules_test.py` — `CLSPool` (incl. the `value_cls` pool) and `ProjectionAssembler` (which returns `(pi_combined, vf_combined)`) are tested on a hand-built `ExtractorContext` (`_dummy_ctx`) without a full forward pass. Prefer adding precise phase-level tests there.
### 🚨 X5's hypothesis tokens: every reduction over OPPONENT tokens declares its presence semantics

**X5 is the ONLY belief representation** (the X5 version break, config v144: `--belief-tokens` and its `blob` path
are DELETED — `model_version/version_break.py`, `designs/model/versioning.md`). The opponent-belief family
(`opp_belief_slots` / `opp_intent`) builds it whenever it is on and REFUSES a configuration missing one of its
requirements; there is no second belief path to keep byte-identical. The blob path's `BeliefSlots`,
`AlphaIntentHead` and `BetaSwitchHead` are DELETED (the version break's part 2 — not even constructed for their
init draws), so the init bytes moved once and the K9 golden is re-recorded at the end of the break. The general rule
stands: a constructor that draws from the global RNG is never added or deleted without re-recording what it moves.
A retired module leaves a plain `None` (`extractor_api.drop_child`), so a strict load REPORTS its keys as unexpected.

A hidden opponent slot holds a hypothesis at presence π < 1 and OTHER_species
holds the tail's mass (`designs/endstate/design_x5_belief_tokens.md` §3.5). A reduction that reads
them as whole mons is the "counts as a whole mon" bug, and it fails no shape check. **An
EXPECTATION-type reduction (softmax attention, a pool, a weighted sum) takes the per-key log π**
(`hypothesis_tokens.OppPresence` / `key_log_presence`; a FLOAT key mask in an `nn` pool, `−inf` on a
masked key) **and gets its I1 / I2 test in `hypothesis_tokens_test.py`** (π = 0 ≡ masked, bit-exact;
two copies at w/2 ≡ one at w). A max-type reduction is class M (presence-scaled max, §9 M2 = C). π is
DETACHED wherever it weights the policy or critic (M10). Every X5 read sits behind `hypothesis_builder is not
None` (the belief-off ablation surface builds none of it). A consumer that contracts
the FLAT POINTER's α / β (presence already inside them, through its log π bias) never multiplies π in
again; its I1 / I2 live beside it (the move-resolution family: `move_resolution_x5_test.py`, where OTHER
is PRICED — OTHER_move one seat per priority level, `move_resolution.split_other_move`; OTHER_species a
7th mon on the renormalised tail).

🚨 **A change to `PokemonEncoder`'s INPUT stitch must be mirrored in `hypothesis_encode.species_table` /
`_cols`** (`gen3_x5_hyp_gather_v1`): X5 computes the hypothesis tokens from the encoder's two first
Linears split by COLUMN BLOCK, so a new input column, a reordered block or a new ROW-level input (anything
not in the dex row) changes the split. `hypothesis_encode_test` compares it with the per-row pass at fp64
and fails on drift; a new row-level input belongs in the per-row half, never the species table.

🚨 **`--token-encoding static` (`static_tokens.py`): a per-mon input goes to S or D by its CLASS, and NO board
fact goes to either** (`designs/endstate/design_static_tokens.md` §1 is the field-by-field table). S must stay a
pure function of the mon's own SET fields (that is what makes an X5 hypothesis token a table gather); D reads
only the mon's own state. `static_tokens_test.py` fails when a board or dynamic field reaches S, when a board
field reaches the encoder at all, and when the move SET stops being permutation-invariant. A new per-mon
observation column must be classified there in the same pass, or `static` silently drops it. **A new sub-block of
the OBS-FACTS block** (the observation's last block, read only under `--obs-facts v1`) **is classified in
`obs_facts_inject.FACTS_TOKEN_CLASS`** (D: added to the mon's token after the encoder; SIDE: the static side board
token, never a mon) — `ObsFactsInject` refuses an unclassified one and `obs_facts_inject_test.py` fails on it.
**Under `static` the board's home is the three BOARD tokens** (`board_tokens.py`, `gen3_static_board_v1`): OUR
SIDE / THEIR SIDE through ONE `side_proj` over the SAME side-relative columns (`SIDE_FACTS`), FIELD, and the
per-mon `op_content` (the `x` ⊕ `g` amounts on every mon, our `d1` cells on theirs). A new board fact goes into
`SIDE_FACTS` (one column for BOTH sides, read for each side) or `FIELD_FACTS`, never into the per-mon encoder;
the base seat count is `TeamTransformer._total_tokens` (15 under static) and a board reader takes
`TeamTransformer.board_rows()` / `board_seats`, never `2·TEAM_SIZE` or the global token.
`static_board_tokens_test.py` fails on a side mix-up, a dropped count, a one-sided or missing op content, a
`x` / `g` / `c4` seat regression and a readout that still reads the global token or `non_matchup_rest`.
**No static weight is keyed by a POSITION** (`gen3_static_port_v1`, v145's rule): the type pair is summed, our moves'
outgoing cells are a set function, and a narrow per-mon / per-seat fact (`static_facts.py`) is added as token content
AFTER the encoder, never into S or D's MLP (a hidden slot's token is a dex-table gather). A fact that restates op
physics READS the op's own rule function (`spikes_entry`), never a second copy. `static_port_test.py` permutes our
team slots through a full forward and fails on a position-keyed weight.

**The op's opponent-MON axis under X5 reads `op.stash.x5` (an `OpRoster`), never a gate of
its own.** Every opponent-slot kernel in `damage_op*.py` takes "alive" from `roster.alive`
(`opp_addressable` — a hypothesis row reads HP 1.0, and no gate may depend on that), the per-mon
candidates from `roster.move_order` / `move_w`, and (in an OTHER-mode pass) the tail-averaged tables
through `_x5_avg` / `_x5_stab`. A NEW opponent-axis kernel must branch on `self.stash.x5` the same way,
get a "no hidden mon dropped" row in `x5_opp_mon_axis_test.py`, and — if OTHER should see it — an entry
in `EdgeBias.OTHER_FAMILIES` plus a call in `_other_edge_cells`. A per-(seat, mon) cell is "what this mon
does IF present": never multiply it by π (the trunk's key bias carries presence); only a reduction over
the MON axis is presence-scaled (`p_pur_vs_us`).
### ⚠️ Identity-at-init is NOT free — SB3 clobbers it (`gen3_identity_init_guard_v1`)

**Every `nn.Linear` you zero-initialise inside the feature extractor is orthogonally
re-initialised by SB3 when the policy is built.** `ActorCriticPolicy._build()` runs
`self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))`
(`stable_baselines3/common/policies.py:617-631`); `init_weights` re-inits every Linear/Conv2d it
finds, and `ortho_init` defaults **True**.

**The guard.** `Gen3FeaturesExtractor.restore_identity_init()` re-zeros them, and
`Gen3DualHeadMaskablePolicy.__init__` calls it after `super().__init__()` (by which point SB3 has
finished). The protected set is captured **by observation** at the end of `__init__` — any Linear
whose weight is all-zero once construction finishes was zero-init'd on purpose — rather than a
hand-kept list, so **a new zero-init module is protected automatically**. Embeddings (e.g.
the belief tables) are untouched by SB3 and need no guard.

**The rule this leaves you with:** an invariant asserted only in a unit test that builds the module
(or a bare extractor) **directly** is not an invariant — that construction path is not the one
training uses. Assert "byte-identical / identity-at-init / cold-start == prior" claims on a REAL
`MaskablePPO`-built policy. `identity_init_test.py` does exactly that, and fails 8/10 if the guard
is removed.
