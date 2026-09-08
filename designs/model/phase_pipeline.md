# The phase pipeline — the chain, the tiers, and what each phase does

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

> 🚨 **[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §2.1 is the live statement of the chain and the
> tier table.** This doc holds the reasoning around it. **STALE:** step 2 below says the observation
> is 2,667 dims (`gen3_entity_rehome_v1`); the live figure is **2501** — `ARCHITECTURE.md` §1, read
> from `Gen3ObservationEncoder.get_layout()`. The text is preserved as it was written.

## The chain, and the four tiers

`forward_internal` is decomposed into phase `nn.Module`s, chained by a thin orchestrator, in ONE
order — the TIER ORDER (`gen3_tiered_pipeline_v1`). There is no placement flag and no second chain:

`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `[MoveBelief?]` → `[SpreadBelief?]` →
`[HPTypeBelief?]` → `[DamageOperator?]` → `prefuse_proj` residual → `EntityMoveSeats` → edge cells →
`TeamTransformer` → `[BeliefHead?]` → `CLSPool` → `[α/β?]` → `[side readouts?]` →
`ProjectionAssembler`, then **two** root heads
(`pre_proj_norm`/`projection` for policy, `value_pre_norm`/`value_projection` for value), each → `ReLU`.

Grouped into the four tiers the contract asserts:

| tier | question | modules |
|---|---|---|
| **T0 RESOLVE** | what is on the board? | `pokemon_encoder`, `t0_species_prior`, `belief_slots`, `move_belief`, `hp_type_belief_head`, `spread_belief`, `item_belief_head` (opt-in) |
| **T1 REASON** | what follows from it? | `damage_op`, `entity_seats`, `history_events` (H-B event seats, opt-in), `edge_bias`, `team_transformer` |
| **T2 DECIDE** | what will they do, what are my moves worth? | `belief_head`, `cls_pool` (which also owns the two token-content critic injections), `alpha_head`, `beta_head`, `intent_threshold_move` / `intent_conditional` / `pair_outcome_move` / `pair_outcome_switch` / `switch_branch` / `conditional_threat` (opt-in) |
| **T3 DELIVER** | one contract, two pools | `hidden_opp_belief`, `assembler`, `win_head`, `value_dist_head` |

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

`BeliefSlots`/`BeliefHead` are built only when `opp_belief_slots` (`--opp-belief-aux-coef>0`),
`MoveBelief` only when `move_belief_mode != off`, `DamageOperator` only when `damage_op` (which requires
`move_belief_mode` revealed/both); with all off the chain is the baseline `ObsUnpack →
PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler` byte-for-byte. `BeliefSlots` swaps the
un-revealed opp role-tokens for learned unknown-mon tokens *before* the transformer (so the belief is
refined in-lineup); `BeliefHead` reads the refined opp tokens *after* the transformer and stashes the
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
`AlphaIntentHead` (published unconditionally, so enabling a consumer later cannot reopen the route
— and since the critic-route deletion wave took every α→vf route, α now reaches the objective only
through the POLICY, via the pointer cells). `BeliefHead`, `PubValHead` and `BetaSwitchHead` are
structurally label-only in every mode — asserted in `belief_label_only_gate_test.py`, not assumed,
so a head that starts feeding forward fails a test instead of quietly rejoining the PPO objective.
🚨 **`WinProbHead` is NOT in that set under `--critic winprob`**: there the head IS `_critic_value`,
so it feeds GAE, the value loss and (at `win_prob_mode shaping`, which the mode implies) the trunk.
Under `--critic shaped` it is label-only like the other three. The claim is mode-conditional, and
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
the same tensor every critic parameterization reads — the dist head under `--value-from-dist`, and
the win head under `--critic winprob`. That is the structural cure for the v89/M2
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
   + their_pool(128) + our_active_refined(128) + non_matchup_rest`. Value: `value_pooled(128) +
   non_matchup_rest` (+ the seed readout over the op's typed `incoming_rows` when the op is on).
   **`gen3_ctx_dedup_v1`: the per-side encoded active contexts are DELETED from both heads** —
   they were duplicated delivery with a 1:1 entity-native replacement already live (the E2
   injection puts each side's FULL raw ctx block on its active token; the global token is a
   second route). `non_matchup_rest` stays: the global token is its only other route and no
   pool reads that token directly, so the concat is currently its one direct head path. When
   the hidden-opponent belief is on, its `[B, K·D_MODEL]` is appended to **both** (last),
   widening each projection input by `k·D_MODEL`.
7. **Root heads** — two parallel `pre_proj_norm` (LayerNorm) → `projection` (Linear) → `ReLU`
   heads, one per `*_combined`, both emitting `PROJECTION_DIM`. SB3 sizes the shared
   `mlp_extractor` from `features_dim = PROJECTION_DIM`, then `Gen3DualHeadMaskablePolicy` feeds
   the policy half to `forward_actor` and the value half to `forward_critic`.
