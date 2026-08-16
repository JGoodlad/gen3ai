# Model Directory — Contributor Notes

## Architecture constants — single source of truth

All network dims are defined as module-level constants in **`arch_constants.py`** (relocated there
2026-08-01 so `damage_op.py` can read them without importing the extractor — that would be circular).
`features_extractor.py` **re-exports the whole block unchanged**, so it remains the documented import
surface and `from agents.model.features_extractor import D_MODEL` still resolves:

```python
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]
MOVE_LATENT_HIDDEN = 64      # MoveLatentEncoder MLP hidden
MOVE_LATENT_DIM = 32         # per-move latent dim (the similarity-grading space)
ROLE_ENCODER_HIDDEN = [256, 128]
```

**Change them in `arch_constants.py` and nowhere else.** The phase modules' `__init__` read from these constants; `ModelVersion` imports them so `model_config.json` always reflects the live values. Do not hardcode these numbers anywhere else in the codebase.

Embedding dims (`species_embedding_dim`, `move_embedding_dim`, etc.) live in `state_encoder.get_layout()` and flow through `features_extractor_kwargs` — same principle, different file.

**`role_input_dim` is not a module-level constant** — it is computed dynamically in `PokemonEncoder.__init__` from the layout fields and `MOVE_NET_HIDDEN`. You do not need to update it manually when dims change; it is derived correctly. The projection input dim is also auto-discovered via a dummy forward pass for the same reason.

## Phase module structure

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
| **T2 DECIDE** | what will they do, what are my moves worth? | `belief_head`, `cls_pool`, `alpha_head`, `beta_head`, `intent_threshold_move` / `intent_conditional` (opt-in) |
| **T3 DELIVER** | one contract, two pools | `hidden_opp_belief`, `assembler`, `win_head`, `pubval_head`, `value_dist_head`, `intent_threshold_value` / `value_clock_route` / `value_intent_route` (opt-in) |

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
prune that previously crippled surprise-move anticipation. **gen3_no_concat_v1 (v61): its flat block no longer enters either
projection** — the op reaches the policy via the pointer cells + prefuse injection + edge cells, and
the critic via the `MultiSeedValueReadout` (k=4×64 seed queries over the per-our-mon rows, vf-only,
with the `value_seeds/*` TB collapse contract logged every train() by `seed_diagnostics.py`, which
stays). **v80 built its designed SUCCESSOR, opt-in and OFF in production:**
`UnifiedValueReadout` (`--value-entity-pool`, `gen3_unified_value_readout_v1` — Stage-3
T3-DELIVER of `design_unified_belief.md` §3): ONE attention pool over the critic's entity rows
(the 12 team tokens + the op's per-our-mon incoming rows, per-source type embeddings, UVR_K=4
queries, zero-init out projection, vf-only after the assembler so pi is untouched at any
weight). The gen-11 `critic_route_audit` — which carries an `entity_pool` arm — adjudicates the
seed/threat routes; a condemned route's next generation enables this in the same config
(`value_entity_pool_test.py` pins the contract).
**TWO PRESSURES WERE APPLIED TO THOSE SEEDS AND BOTH ARE NOW DELETED (v78)** — the record is kept
because the finding is what closed the line, not the code. `--value-seed-vicreg-coef` (v62,
`seed_vicreg.py`) was the repulsive one: gen-6 satisfied every VICReg term while
`out_effective_rank` stayed 1.05, because the deviations occupied <1 direction (three seeds
identical, one breakaway). `--seed-quantile-coef` (v63, `seed_quantile.py`) was the positive
counterpart — seed k predicts quantile τ_k of the return through ONE SHARED Linear, so k different
predictions require k different seed reads — and gen-7 drove `crossing_rate` to 0.000 with
`quantile_spread` 1.016 (the seeds genuinely predict four ordered quantiles) while
`out_effective_rank` reached only **1.157 of k=4**, matching gen-6's centered PR 0.846 from the
opposite direction. **A SHARED readout can only constrain each seed's component along its own
weight vector; every orthogonal direction stays free**, so multiplicity is not the missing axis and
no coefficient reaches it — which is why both flags were deleted rather than retuned. The response is
**`--value-threat-inject` (v64, `value_threat_inject.py`)** — magnitude as TOKEN CONTENT rather
than as another readout seat: one shared zero-init `Linear(13, D_MODEL)` adds the op's α-weighted
incoming row for our mon j (α = the R1 `belief_mean` rung, which the flag forces on since R0
`hard_max` builds no reducer) to that mon's token on **the value pool's copy only**, inside
`CLSPool`. Keeping the augmented tensor a local is what makes "vf-only" structural rather than a
convention: `our_cls`, `our_active_refined` and the pointer head cannot reach it, so `pi` is
bit-identical at ANY weight — gated against a large random projection, not just at init. Equivariant
in both axes (α has no defender index by Contract W; the row rides mon j's token; attention pooling
is permutation-invariant). `W_inj` sits in the `restore_identity_init()` capture set (M1) and that
is gated on a REAL `MaskablePPO` build. Structural + version-checked, off = no module).
(`BeliefHead` also carried an asymmetric SimSiam **latent** predictor until v75, regressing each
believed slot toward the stop-grad `pokemon_encoder` role-token of the true hidden mon. It is DELETED —
it was never fed forward, its own role-geometry probe concluded decodable != helps, and it cost ~13% of
the train step. Predicting the opponent's unrevealed mons is unaffected: the species CE, the moves BCE
and the T0 species prior all remain. See `designs/CHANGELOG.md` for how these landed.)
A separate `WinProbHead` (`win_prob_mode != none`) reads `value_pooled` *after* the pools and stashes
a `last_win_prob_logits` [B,1] — another side readout (never in pi/vf, so projection dims are unchanged),
read by the win-prob aux loss + the prober. `read_only` feeds it a STOP-GRAD `value_pooled` (head trains
its own params only); `shaping` feeds it live (the win objective also shapes the trunk).

### `--belief-grad-mode` — which arrow gets cut (`gen3_belief_grad_mode_v1` / `gen3_belief_label_only_v1`)

**The two non-default modes cut OPPOSITE arrows, and the flag name does not say so** — read this
table before reasoning about either. Four routes exist between a state-prediction belief head and
the rest of the network:

| | route | `shaping` | `detached` | `label_only` |
|---|---|---|---|---|
| A | label loss → belief head params | on | on | on |
| B | label loss → shared trunk (the head's READ) | on | **CUT** | on |
| C | PPO loss → belief head params (the WRITE) | on | on | **CUT** |
| D | PPO loss → shared trunk (normal training) | on | on | on |

`detached` stop-grads the head's trunk **read** (`detach_read`), so the belief cannot reshape the
trunk. It does **not** stop PPO training the heads — the reinject write stays live, deliberately
(`belief_grad_mode_test::test_detached_preserves_normal_trunk_training`). `label_only` stop-grads
the head's **output** at its publish boundary (`publish_detach` inside a head, `_publish_belief` on
the extractor), so the belief is trained by its labels alone while the policy still reads it. The
read stays live under `label_only`, because cutting B and C together leaves a probe on a trunk with
no incentive to encode hidden state — still feeding the policy. That combination is not offered.

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

Scope is the four heads with a forward path: `MoveBelief`, `SpreadBelief`, `HPTypeBelief`, and
`AlphaIntentHead` (reachable only under `--intent-value-reduce`, published unconditionally so
enabling that flag later cannot reopen the route). `BeliefHead`, `WinProbHead`, `PubValHead`,
and `BetaSwitchHead` are structurally label-only in every mode — asserted in
`belief_label_only_gate_test.py`, not assumed, so a head that starts feeding forward fails a test
instead of quietly rejoining the PPO objective.

`detach()` is value-preserving ⇒ the forward is bit-identical in all three modes ⇒ this is a
resume-immutable training hparam (the `vf_coef` class), NOT weight-shape: no `ARCH_SIGNATURE` bump,
excluded from `check_compatible`, enforced resume-only by `check_belief_grad_mode`
(`--allow-belief-grad-mode-change` for an intentional migration). `BELIEF_GRAD_MODES` in
`features_extractor.py` is the single source for the legal set.

**Dual-head value readout (H4 / Option C).** The transformer body is shared, but the actor and
critic read it through independent paths. `CLSPool` holds a third query `value_cls` that attends
over all 12 team tokens to produce `value_pooled`; `ProjectionAssembler.forward` returns a
`(pi_combined, vf_combined)` pair; and the root `forward` returns a `(pi_features, vf_features)`
tuple. This extractor therefore **must** be paired with `Gen3DualHeadMaskablePolicy`
(`policy.py`), which keeps `share_features_extractor=True` (one body) and overrides `forward` /
`evaluate_actions` / `get_distribution` / `predict_values` to unpack the tuple and route each half
to `mlp_extractor.forward_actor` / `forward_critic`. A stock SB3 policy expects a single-tensor
extractor and will break — doubly so under the pointer-native action head (`gen3_pointer_native_v1`): the policy's `_build`
deletes the flat `action_net` and the action logits come from the `PointerNativeActionHead` over
the extractor's `last_pointer_inputs` stash (per-logit inputs: `designs/ARCHITECTURE.md` § Heads). The startup `_run_roundtrip_test` and the snapshot/feature tests all
unpack the tuple — keep that in mind when touching the extractor's return shape.

### Phase-by-phase data flow

The embedding tables live in a shared `Embeddings` module passed as a forward argument to the
phases that need them, so they register exactly once. An immutable `ExtractorContext` produced
by `ObsUnpack` carries the ~30 unpacked tensors downstream, keeping each phase's signature
narrow. Both projection input dims are auto-discovered via a dummy forward pass in `__init__`,
so they stay correct when the architecture changes with no manual update.

> 🚨 **The discovery forward must reach EVERY value part — so a construction-time width probe may
> only ever fall through, never `return`.** The tail of `forward_internal` appends optional parts
> to `vf_combined` in sequence (`intent_value_reduce`, then v80's `value_entity_pool`, then
> whatever comes next), and each has a discovery branch that contributes a correctly-shaped ZERO
> because its real operand does not exist yet. `intent_value_reduce`'s branch **returned the pair
> outright**, so every part appended below it was invisible to the very forward that sizes
> `value_pre_norm` — v80 landed underneath and the critic was built `UVR_OUT_DIM` (128) short,
> dying on the first real forward with `normalized_shape=[1241] … got [*, 1369]`. It fires only
> with BOTH flags on, so it was unreachable until the two met, and production wanted both on the
> next run. When you add a value part: append it at the tail, give it a fall-through discovery
> branch, and add a both-flags-on build to `value_entity_pool_test.py` — **every flag in that tail
> was individually tested and the intersection was not, which is the whole reason this shipped.**

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
4. **`TeamTransformer`** — builds a 20-token sequence (6 our-team + 6 their-team role tokens +
   `N_HISTORY_TURNS`=7 history tokens + 1 global token), adds token-type and history-positional
   embeddings, and runs a `TRANSFORMER_N_LAYERS`-deep `nn.TransformerEncoderLayer` stack (d_model
   128, `TRANSFORMER_N_HEADS` heads, FFN `TRANSFORMER_FFN_DIM`, post-LN) under a key-padding mask
   that masks fainted team slots and empty history slots. History tokens come from
   `embed_delta_slot`; the global token from the two active-contexts + non-matchup scalars.
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

Rules to preserve:

- **Each phase owns its layers** (`move_network` lives under `pokemon_encoder`, `our_cls` under `cls_pool`, etc.). State_dict keys are therefore phase-prefixed.
- **`Embeddings` is the sole owner of the 5 embedding tables + `hp_type_idx_map`.** It is passed as a **forward argument** to `PokemonEncoder` and `TeamTransformer` — never stored as a child attribute on them — so the tables register exactly once. (The root exposes read-only `@property` forwarders like `model.type_embedding` for convenience; those add no state_dict keys.)
- **`ExtractorContext`** (frozen-by-convention dataclass) is the inter-phase contract: `ObsUnpack` produces it, downstream phases read from it. Add a field here rather than widening a phase's positional signature. Cross-phase values (active-slot indices, fainted masks, `hp_probs`) are computed once in `ObsUnpack` and carried on the context.
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version.py`. **Read the live value there, not from prose.** Three cases people get wrong:
  - A **pure decomposition** still changes state_dict keys, so old checkpoints must fail loudly — bump it.
  - A forward-math change with **unchanged `out_dim` / projection widths** is not shape-caught by anything, so the signature bump is the ONLY thing that rejects a stale checkpoint. This is the case that has bitten most often.
  - **Re-sourcing or re-meaning an obs block** is retrain-class even when no individual dim moves (a constant fallback becoming a real value; a scalar's definition changing; a block moving from `available_moves` order to request-slot order). The long list of historical examples lives in `designs/CHANGELOG.md`.
- Per-phase unit tests live in `phase_modules_test.py` — `CLSPool` (incl. the `value_cls` pool) and `ProjectionAssembler` (which returns `(pi_combined, vf_combined)`) are tested on a hand-built `ExtractorContext` (`_dummy_ctx`) without a full forward pass. Prefer adding precise phase-level tests there.

## File layout (`gen3_damage_op_split_v1`, 2026-08-01)

`features_extractor.py` was ~4,700 lines, of which `DamageOperator` alone was **1,689 (39%)**. Split
into three, a **pure relocation** — same classes, same constants, same forward math:

| file | holds |
|---|---|
| `arch_constants.py` (37) | the architecture constants — the single source of truth for weight-shape dims |
| `damage_op.py` (2,102) | `DamageOperator` + its `_DMG_*` constants + `decode_damage_block` |
| `features_extractor.py` (2,922) | everything else; **re-exports all 89 moved names** |

No import cycle: `DamageOperator` touches the extractor only through `ctx: 'ExtractorContext'`, which
is a **string** forward-reference and so costs no runtime import. The re-export means every historical
path (`from agents.model.features_extractor import DamageOperator / decode_damage_block / _DMG_* /
_SB_ATK`) still resolves — the prober, `model_version`, `snapshot` and the tests all rely on that.

**The gate for a refactor claiming to change nothing is proof, not review:** byte-identity on pi/vf +
the raw op block (`tmp/damage_op_equiv_probe.py`), unchanged `state_dict` keys, the constructed-scenario
physics oracle (`damage_op_probe_fuzz_test.py`, 22/22), and the full suite. All four held.

## The op's flat layout has ONE slicer (`gen3_op_tensors_views_v1`)

`DamageOperator.tensors_from_block()` is the only place the flat block's offsets are walked; it
returns **`OpTensors`** — named zero-copy views (`incoming_rows`, the CB tail, the outgoing/status
groups, the opaque matrix renders). Every same-forward consumer reads a field off
`damage_op.last_tensors` (prefuse injection, the assembler's `seed_rows`) or goes through
`pointer_cells` (which itself now assembles from the views); **never re-derive an offset at a
consumer** — the layout walk raises if a region is added to the block without a view. The flat
block remains the serialization: `decode_damage_block` (the prober's human-readable mirror) and
`last_raw_block` still read it, and dropping it from the forward is `design_op_tensors.md` step 3
(retrain-class — it shrinks `out_gain`). Landed as a byte-identical refactor under the proof
bundle above, on 64 real gen-9 eval states across three config arms.

## ⚠️ One op's SPELLING is load-bearing for `torch.compile` (`gen3_species_posterior_spelling_v1`)

`BeliefHead.species_posterior` computes `P(species)` for the expected-latent defender. It is written
as **`log_softmax(...).exp()`, not `torch.softmax(...)`, and that is deliberate** — do not
"simplify" it.

`torch.softmax` over the last dim of the `[B,6,n_species]` logits lowers to a numerator buffer plus a
`[B,6,1]` denominator, and the Inductor **CPU** scheduler then trips `AssertionError: buf<N>` trying to
fuse the division. That single op was the reason `--compile-opponents` used to set
`torch._dynamo.config.suppress_errors = True`, which in turn meant the production config compiled only
partially (3.6× instead of 6.53×) and every other backend failure in the process went silent.

`tmp/softmax_variant_probe.py` measured the alternatives: `.contiguous()`, `.clone()`, a 2-D
reshape and a hand-rolled `exp / sum` **all still fail**; only the `log_softmax().exp()` factoring
lowers cleanly. It is mathematically identical and keeps the same max-subtraction stability (measured
max|Δ| vs eager 5.07e-07). Guarded by `extractor_compiles_test.py`, which owns the whole compile
matrix: the fast tests pin the math, and the compile cells run a real compile of the literal
production arch with suppression OFF (verified to fail if the old spelling returns) across
CPU/CUDA x forward/backward — `GEN3AI_SKIP_COMPILE_TESTS=1` opts out, `GEN3AI_TEST_ALLOW_GPU=1` is
needed for the CUDA cells (the root conftest hides the GPU from the suite). Repro:
`tmp/inductor_crash_repro.py`. Note the CPU **backward** does NOT lower — an `atomic_add` scatter
the C++ backend refuses — which is why the compiled-opponent artifact is inference-only.

**It is NOT "CPU cannot accumulate", and the distinction is the actionable part.** Inductor's C++
backend has THREE store kernels and two of them implement the mode: `CppKernel.store` emits
`atomic_add(&buf[i], v)`, `CppVecKernel.store` emits `atomic_add_vec<...>(...)`, and only
**`CppTile2DKernel.store`** carries the bare `assert mode is None`. `CppTile2DKernel` is the
2D-tiled/TRANSPOSED variant, chosen when the store's index pattern needs a transpose. So the
refusal is one missing case in one kernel variant, selected by memory LAYOUT — if a CPU training
compile ever mattered, the lever is to reshape the scatter so Inductor picks `CppVecKernel`, not to
wait upstream. Triton's `store()` has the case unconditionally (`tl.atomic_add(..., sem='relaxed')`),
which is why CUDA — the only compiled backward we actually run — is unaffected.

The op is an accumulate-scatter because it IS a backward: the gradient of a gather/index-select is
a scatter-ADD (indices may repeat, so writes must accumulate). Cut the gradient and it disappears.
**So the refusal is CONFIG-CONDITIONAL, and the pin now says so**: `--belief-grad-mode label_only`
publishes every belief output stop-grad, which deletes those backwards and lets the CPU backward
compile cleanly. Bisected 2026-08-15 — `shaping` REFUSED, `label_only` COMPILED, `win_prob_mode`
irrelevant. ⚠️ **Which gather is NOT pinned**: detaching the obvious candidate
(`damage_op.py`'s `w_all.gather(-1, topk_idx)`, whose shape matches the reported buffer exactly)
left the refusal in place, so there are several sites and the shape match was a coincidence. The
limitation test therefore builds at `belief_grad_mode="shaping"` explicitly, because production
moved to `label_only` at gen-11 and the pin would otherwise have gone green while testing nothing;
that is
pinned as a limitation test that fails if it ever lifts.

**The general lesson:** a backend that "can't compile our model" was one op, not a property of the
architecture. Before reaching for a global suppression flag, bisect to the op — see
`src/agents/training/CLAUDE.md` → Compiled CPU opponents.

## ⚠️ Identity-at-init is NOT free — SB3 clobbers it (`gen3_identity_init_guard_v1`)

**Every `nn.Linear` you zero-initialise inside the feature extractor is orthogonally
re-initialised by SB3 when the policy is built.** `ActorCriticPolicy._build()` runs
`self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))`
(`stable_baselines3/common/policies.py:617-631`); `init_weights` re-inits every Linear/Conv2d it
finds, and `ortho_init` defaults **True**.

Until 2026-08-01 this silently falsified the identity-at-init contract for **13** Linears in every
real training run — the zero-init physics projections (`prefuse_proj` and, in the configs of the
day, the between-layers refine loop's), `film_pi`/`film_vf`, plus the belief heads
(`MoveBelief.move_head`, `SpreadBelief.*`, `HPTypeBelief.type_head`) whose zero-init is what makes
the **cold-start posterior equal the Smogon prior**. Measured max|W| before the fix: 0.19–0.47. See `designs/research_state/ledger.md` → **M1**
for the standing caveat this puts on the K10 and D4 result families.

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

## The flag registry — one declaration, five surfaces (`flag_registry.py`)

**Add a model-relevant toggle by adding a `ModelFlag` row to `agents/model/flag_registry.py`, then
following where the tests send you.** That file is the single declaration of every extractor
architecture toggle and of the five hand-synced places each one has to appear:

| # | surface | what it buys | how it is kept honest |
|---|---|---|---|
| 1 | the `argparse` entry in `main.train_rl_agent` | a human can SET it | **validated** |
| 2 | the `_resolve("name", default)` line beside it | a **flagless** resume INHERITS it | **validated** |
| 3 | `extractor_arch.ARCH_ARG_KEYS` / `_DERIVED` / `FROZEN_ARCH_KWARGS` | it reaches the extractor | **generated** |
| 4 | `snapshot.current_model_version()`'s keyword (via `_run_arch_toggles` → `arch_toggles_from_args`) | an eval/self-play WORKER rebuilds the SAME gate | **generated** + validated |
| 5 | the `ModelVersion` dataclass field | it is RECORDED and version-GATED | **validated** |

`flag_registry_test.py` fails with a message **naming the missing site**, which is the whole point:
every historical failure in this class was silent. A toggle in `ARCH_ARG_KEYS` but not on
`ModelVersion` means a resume version-checks against an architecture it does not build; one with an
argparse entry but no `_resolve` line means a flagless resume reverts it to OFF. The test earned its
keep on the first run — it found three rows whose flag name is not `--<field>`: `--damage-topk`
writes `damage_topk_k`, and the `--damage-matrices` MODE flag desugars into both
`damage_matrices_*` bools. Those name their flag with `cli_name=` rather than being exempted.

**Read `designs/flag_registry.md`** for the current table (generated; `--check` is the gate).

### The three TIERS — a flag can lose its CLI entry without losing explicitness

A flag plays three independent roles — **SELECT** (choose it at launch), **RECORD** (write it into
`model_config.json`), **GATE** (refuse a mismatched resume). Only SELECT needs argparse; RECORD and
GATE live in `ModelVersion` and are reached whether or not argparse ever heard of the toggle.

| tier | argparse | `_resolve` | recorded + gated | reachable for an experiment |
|---|---|---|---|---|
| `cli` | yes | yes | yes | via the flag |
| `config_only` | **no** | **no** | yes | via the extractor **constructor kwarg** |
| `constructor_only` | no | no | no | via the constructor only (`pair_reduce`'s `reduce_how`) |

**A `config_only` toggle is FROZEN at its registry `default` for every CLI-launched run** — that is
the only value the CLI can now produce, so the default must be the value production actually wants.
Demote a toggle when it is *settled*: same value in every run, no live experiment. The extractor's
own constructor default is deliberately left alone, so the OFF baseline stays constructible for a
test or a probe; only the launch surface shrinks. `config_only_pattern_test.py` pins the contract
end to end (recorded in a fresh `model_config.json` · rejected on a mismatched resume · no argparse
entry). The three demoted at v78: `attend_unrevealed_opponents` (frozen **ON** — a hard prerequisite
of the whole belief stack since v16), `value_active_readout` and `damage_matrices_outgoing_all`
(frozen OFF — never enabled in a gen-8/9/10 run).

### The four CLASSES — which gate a mismatch gets

| class | a mismatch means | gate |
|---|---|---|
| `structural` | weights and/or the trained forward differ | `check_compatible` — runs on **every** load |
| `resume_immutable` | the forward is bit-identical; only TRAINING differs | a dedicated `check_*`, **resume path only** |
| `training_coef` | a loss weight moved | none; recorded for provenance |
| `runtime` | a perf knob moved | none; never recorded, never inherited on resume |

Getting this wrong hurts in **both** directions, so both are asserted: a `structural` toggle with no
`check_compatible` compare lets a resume silently flip the architecture, and a `resume_immutable`
toggle *inside* `check_compatible` makes a run FATAL while loading its own pool snapshots (that gate
runs on frozen eval/pool/distill opponents too, whose forward is identical regardless).

## Model versioning (`model_version.py`, `snapshot.py`)

Every model save writes the **run-level** `model_config.json` + `metadata.json` at the run root via `save_model_snapshot()`, plus a **per-checkpoint** `.json` sidecar beside each checkpoint `.zip` (`write_checkpoint_metadata`, derived from the zip path). Periodic + forced checkpoints `.zip` live in `<run>/checkpoints/` (so their sidecar lands there too); the run-level config/metadata stay one level up at the run root. Loading goes through `load_model_snapshot()`, which resolves the zip then searches **its dir AND its parent** for `model_config.json` (so the run-root config is found even when the zip is in `checkpoints/`; `load_foreign_opponent` does the same) and runs `check_compatible()` before `MaskablePPO.load()` — a mismatch fails fast with a clear error rather than silently loading bad weights. (`snapshot_history` keys + the `worktree.py` resume lookup stay BARE basenames, e.g. `checkpoint_123_steps.zip`, regardless of the subdir.)

**When you change an architecture constant:**
- `check_compatible()` catches the mismatch automatically — no extra steps needed
- Old models can't be loaded, which is correct (rapid iteration project)

**When you add an optional new feature** (new field with a sensible default):
1. Add the field to `ModelVersion` in `model_version.py`
2. Bump `MODEL_CONFIG_VERSION`
3. Add one `if version < N:` block in `_migrate_config()` with `data.setdefault(...)`

**When you make a structural change** (different forward pass, new layer type):
1. Change `ARCH_SIGNATURE` in `model_version.py` (e.g. `"gen3_attn_v1"` → `"gen3_lstm_v1"`)
2. Old models get a clear arch-family error on load

**⚠️ REORDERING a module's parameters silently breaks the optimizer on resume.** SB3/torch save+load
the Adam optimizer state **by parameter POSITION, not name**. So if a refactor changes the *order*
`named_parameters()` yields (e.g. building submodules in `__init__` in a different sequence — the
`gen3_nature_ev_belief_v1` bug, where `SpreadBelief.__init__` moved `reinject`/`norm` before
`stat_head`), a resume's **weights** still load fine (name-keyed `load_state_dict` → arch check PASSES)
but the **momentum** (`exp_avg`/`exp_avg_sq`) gets assigned to the WRONG params. It then crashes in
`AdamW.step()` ("size of tensor a (128) must match b (5)") the moment a misassigned param of a
different shape first gets a gradient — **data-dependently, so it can survive many steps**, and (until
the guard) the broad `except` in `train_rl_agent.py` masked it as a clean completion. Guard:
`train_rl_agent._validate_or_reset_optimizer_state(model, checkpoint_path)` runs on every resume and
**REMAPS the momentum to the current params BY NAME** — it reads the saved optimizer state + the saved
parameter NAME ORDER straight from the checkpoint zip (`policy.optimizer.pth` + `policy.pth`) and
rebuilds `opt.state` so each current param receives the momentum saved for its name, regardless of
registration order. So a reorder is **corrected**, not just caught: a **same-shape** reorder (which a
shape check CANNOT see and would silently scramble) now follows the name, and a name reused at a
different shape (or a genuinely new param) cleanly drops to fresh zero-init. **This means "append new
params LAST" is no longer load-bearing for optimizer correctness** — though still good hygiene. Falls
back to the legacy shape-only drop-all-momentum reset only if the zip can't be read (never crashes a
resume); no-op (momentum carried verbatim) on an aligned resume. Pinned by
`src/main/resume_optimizer_realign_test.py` (incl. the same-shape-reorder + zip-read cases).

**Resume-immutable training hparams (value-meaning, NOT weight-shape).** A hyperparameter can
be wrong-to-change-mid-run without changing any weight shape — `vf_coef` (`--vf-coef`) is the
first: it rescales the value head's gradient on the shared trunk, so a forgotten/typo'd flag on
resume would silently drift training. These are recorded on `ModelVersion` (→ `model_config.json`)
but **deliberately excluded from `check_compatible`** — that gates EVERY load, including the frozen
eval / self-play-pool / distill opponents, where the forward is identical regardless of the value
and a false rejection would break league play. Instead they get a dedicated check
(`ModelVersion.check_vf_coef`) invoked **only on the training-resume path** via
`load_model_snapshot(..., enforce_vf_coef=…)`; `train_rl_agent.py` FATALs on mismatch exactly like
an arch error. To add another such hparam, follow the optional-feature playbook above (field +
`MODEL_CONFIG_VERSION` bump + `_migrate_config` default) **plus** a dedicated `check_*` + an
`enforce_*` opt-in on `load_model_snapshot`, and leave it out of `_WEIGHT_FIELDS`.

The **reward-config** hparams are the same kind, bundled into one check: `bias_additivity`
(`--bias-additivity`), `mat_alive_weight` (`--mat-alive-weight`), `bias_redesign` (`--bias-redesign`),
`switch_bias_weight` (`--switch-bias-weight`, the belief-risk stay-into-KO BIAS lever, v5),
`draw_penalty` (`--draw-penalty`, the DRAW/250-turn-timeout terminal, v7 — default −30.0 = a tie scores
as a decisive loss; set lower to make a stall-to-cap strictly worse), `self_ko_hp_penalty`
(`--self-ko-hp-penalty`, the HP-scaled self-KO penalty — default 0.0 = OFF; >0 charges −w·hp when
our mon self-KOs via Explosion/Self-Destruct, since the symmetric material PBRS prices a healthy 1-for-1
trade at ~0 and the critic then over-values it), the de-bias cleanup pair `drop_redundant_bias` +
`drop_switch_bias` (`--drop-redundant-bias` / `--drop-switch-bias` — zero the audit-flagged
distorting BIAS terms: stall_tax + matchup_penalty redundant with the no-progress clock/`--draw-penalty`
and `pbrs_belief`; the hand-coded switch subsidy), and the **two end-state PBRS switches**
`all_shaping_pbrs` + `stall_pbrs` plus `no_progress_penalty` (`--all-shaping-pbrs` / `--stall-pbrs` /
`--no-progress-penalty`): `all_shaping_pbrs` = "everything but stall" — folds
Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status and **zeros every BIAS term except the anti-stall tilt
`no_progress_tax`** (so all non-stall shaping is policy-invariant; the bad turn-ramp `stall_tax` is
zeroed); `stall_pbrs` = "stall" — folds Φ_progress and zeros `no_progress_tax`+`stall_tax`. Run BOTH ⇒
the whole BIAS class is zero (TERMINAL + PBRS only); run only `all_shaping_pbrs` ⇒ keep the
`no_progress` stall tilt as the single acknowledged BIAS. `no_progress_penalty` is recorded+checked
because it is Φ_progress's weight. (`--all-shaping-pbrs` ALSO now folds the DEDICATED phaze-out-boosts PBRS
**`pbrs_roar`** Φ_roar = −`ROAR_BOOST_WEIGHT`(0.25)·Σmax(0,opp-active-boost) — NO separate flag/field, it
rides the existing `all_shaping_pbrs` toggle, stacking with the bundled `pbrs_opp_boosts` for stronger
proportional roar-out-boosts shaping; safe since both telescope to 0.) All are recorded on
`ModelVersion` and enforced on resume by **`check_reward_config`** (FATAL on drift, since they silently
shift the reward/objective), excluded from `check_compatible`. They are reward-VALUE changes — **no
`ARCH_SIGNATURE` bump** (the network/obs are unchanged) — so a fresh run is needed to measure them but
old checkpoints don't fail an arch check — a fresh run is needed to measure them.
The live `MODEL_CONFIG_VERSION` is in `model_version.py`; per-version entries are in `designs/CHANGELOG.md`.

**The per-version entries that used to live here have moved to `designs/CHANGELOG.md` §4**
(verbatim). They described what each of v6–v57 added, in parallel with the root `CLAUDE.md`'s own
version narrative — two records of the same history that had drifted out of agreement with each
other and with the code.

- **What the architecture IS right now** — obs layout, the phase chain under the production config,
  what each head consumes, the `DamageOperator` block, the edge families, the flag table with
  `INERT` markings: **`designs/ARCHITECTURE.md`**.
- **What each version changed**: `designs/CHANGELOG.md` (history — do not quote as current).
- **The live values**: `MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` in `model_version.py`. Read them
  there. This file deliberately no longer states them: a version number written into prose is stale
  the moment the next one lands, and quoting a stale one is how a v30 description got applied to a
  v59 model.

The mechanics above (what to bump when, the optimizer-reorder guard, the resume-immutable-hparam
playbook) are the durable part and stay here. When you add a toggle, follow those rules, then record
the entry in `CHANGELOG.md` and state the new truth in `ARCHITECTURE.md` — never narrate it here.

A startup smoke test (`_run_roundtrip_test` in `train_rl_agent.py`) saves to a temp dir and reloads before every `model.learn()` call — catches serialization issues immediately.

## PopArt value-target normalization (`popart.py`, `--use-popart`)

Opt-in (default off). The dual-head extractor shares one trunk; with γ≈0.9999 the returns run to
±hundreds, so the value MSE gradient **swamps** the shared trunk and the policy under-updates
(diagnosed by a large positive `grad/value_policy_logratio`, see `src/agents/training/CLAUDE.md`). PopArt fixes the value
*scale* adaptively: `PopArtNormalizer` keeps running `(mu, sigma)` of the value targets, the value
head outputs **normalized** values, and the PPO loss trains in normalized space — so the value
gradient stays O(1). The **POP** half rescales `value_net`'s weight+bias on every stats update so the
**de-normalized** prediction is unchanged (`W'=(σ_old/σ_new)·W`, `b'=(σ_old·b+μ_old−μ_new)/σ_new`),
making the stats update a no-op on the value function (no corruption — the failure mode of naive
running-std normalization). Pure/torch-only → unit-tested in `popart_test.py` (load-bearing test:
**POP invariance**, de-normalized outputs identical across a stats update).

- **Policy integration** (`policy.py`): `__init__` takes `use_popart` (from `policy_kwargs`) and
  builds `self.popart` **after** `super().__init__` (which builds `value_net`); the 3 value sites
  (`forward`/`evaluate_actions`/`predict_values`) wrap the output in `self._denorm(...)` so GAE /
  advantages / bootstrapping always see **real-unit** values. `popart` is `None` when off (identity
  `_denorm`). The `(mu, sigma)` buffers ride the policy state_dict → save/restore for free.
- **PPO loop** (`instrumented_ppo.py`): once per `train()` (before the epochs) `popart.update(returns,
  value_net)` advances the stats + POPs; the value loss becomes `MSE(normalize(returns),
  normalize(values))`. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors
  otherwise — a self-documenting config beats a silent override): clipping is unnecessary with value
  normalization (the literature finds it little/negative regardless), and since the value sites
  return *de-normalized* values an active clip would clip in un-normalized units (`clip_range_vf` vs
  σ) and cripple the critic.
- **Version-checked**: `ModelVersion.use_popart` is recorded in `model_config.json` (config v3) and
  `check_compatible` raises a dedicated error if a resume toggles it — the value head's
  parameterization differs, so it can't be flipped mid-run.
- **Diagnostics** (TB + TUI): `popart/mu` & `popart/sigma` (should track `train/return_mean` &
  `train/return_std`), `popart/value_weight_norm` (POP keeps it bounded). With PopArt on,
  `train/value_loss` is the *normalized* loss (≈O(1)) and `grad/value_policy_logratio` should fall toward ~0.
- `_DEFAULT_BETA` (EMA decay, 0.1) and `_SIGMA_FLOOR` (1e-2) are module constants in `popart.py`
  (the only flag is on/off). The POP rescale changes `value_net` outside the optimizer; momentum
  staleness is negligible because `σ_old/σ_new ≈ 1` each call (optimizer state intentionally not
  rescaled — the standard PopArt approximation).

## Where the canonical architecture lives

| Question | File |
|---|---|
| What the architecture **IS** right now — obs layout, the phase chain with the production config's flags resolved, what each head consumes, the `DamageOperator` block, the edge families, the flag table | **`designs/ARCHITECTURE.md`** |
| A machine-checked picture of it — seats and sinks, edges typed by what they physically carry | `designs/architecture_graph.dot` (generated by `delivery_graph.py`, pinned by `delivery_graph_test.py`) |
| **The same picture, interrogable** — path queries ("what does the critic see?"), the measured-dependence overlay across every audited checkpoint, a per-family bias selector, and per-token detail (what a seat can deliver to, and every bias family acting on it ranked by measured dependence). **Family codes are never shown bare** — every `d2` / `c1` / `s3` carries its one-line label, with the cell definition parsed out of `features_extractor.py`'s `_EDGE_*_CELL` block so it cannot drift (`FAMILY_LABEL` holds only the curated phrase; a family with no entry fails the tests) | **https://model.g5d.io** — served live by `--serve` (re-rendered from the checkout per request, so it cannot go stale), or **`designs/architecture_viewer.html`** via `file://`. **Dark by default**. Generated by `build_arch_viewer.py` from **real asset files** — `arch_viewer_assets/viewer.{html,css,js}`, not a string literal, so the JS is `node --check`ed by a test and the CSS is lintable; the server lives apart in `arch_viewer_serve.py`. Pinned by `build_arch_viewer_test.py`; regenerate with `python -m agents.model.build_arch_viewer` (`--check` is the staleness gate). `--vendor --out <path>` inlines cytoscape for a copy that needs no network at all (a separate output — the committed artifact stays CDN-linked so `--check` has one thing to compare against) |
| **Does the page actually render?** — the text tests never execute a line of its JavaScript, and a `#theme` deep link once painted every node in the wrong palette because cytoscape resolves the CSS variables once at construction | `build_arch_viewer_render_integration_test.py` — headless chrome reads back a `document.body.dataset` record (script completed, every node positioned, and the node fill **as cytoscape computed it**). Skips, naming which, when there is no browser or no network |
| The **phase CONTRACT** — what a phase may own, `ExtractorContext` / `Embeddings` rules, the versioning playbook | the "Phase-by-phase data flow" + "Model versioning" sections **above** |
| How each version got here | `designs/CHANGELOG.md` |

The split is deliberate: this file holds the **rules** a phase must follow (durable), and
`ARCHITECTURE.md` holds the **state** the model is currently in (changes every run). When you touch
`features_extractor.py`, update the contract here if a rule changed, and `ARCHITECTURE.md` if the
state did — then regenerate the delivery graph.

> `designs/ai_v3/README.md` holds an old Mermaid digraph + dimension table. It is a **frozen
> historical record** (1309-dim obs, the pre-unified-transformer attention paths) and is **NOT
> maintained** — do not update it for current-arch changes. It carries a banner saying so. It is
> also the reason `designs/architecture_graph.dot` is generated rather than drawn.

## Opponent intent — `α` / `β` (`opp_intent.py`, v67)

The build for one sentence the model could not express: *"they are likely to click **this**, so
**this** is my answer."* `--opp-intent-coef>0` adds two SUPERVISED pointer heads:

- **`α`** — a distribution over the opponent's K believed threat-move seats (the refined **E4**
  tokens) **plus a SWITCH option**. Seat k's logit is scored from seat k's own token through a
  SHARED scorer, so `α` is equivariant under permuting their moves; SWITCH is scored from board
  context alone (there is no per-seat object to point at — it is the "none of these" option).
- **`β`** — given a switch, which of their mons comes in. A pointer over their six team tokens,
  masked to alive-and-non-active: an illegal switch-in must be UNREPRESENTABLE, not merely
  unlikely, or the head spends capacity learning the rules.

**Why pointers and not a flat `Linear(ctx, K)`.** The flat form passes every shape test and then
learns "seat 0 is usually right" from the belief's own `w.topk` sort order — memorising exactly the
ordering `α` exists to correct. Equivariance is gated in both axes.

**Matching is by canonical id.** Seats permute every turn and are built by the model mid-forward, so
the env emits the opponent's move NUM and `match_seats_to_move_num` locates it at loss time. A
belief miss is MASKED and `opp_intent/alpha_mask_rate` is logged — that rate is the BELIEF's coverage
failure, and folding it into "α was wrong" would hide which component to fix.

**The label is for the PREVIOUS decision.** Their turn-t action is only observable while building the
obs for t+1, so `instrumented_ppo` shifts the label block back one row **before `get()` shuffles**
and drops any pair whose successor starts an episode (`align_labels_to_predictions`). Skipping that
drop splices one battle's first decision onto another's last board — invisible in every metric.

**Supervision only:** both heads read a DETACHED input, so a null indicts the head's predictive
power, not the policy. Structural + version-checked; requires `--entity-topk-seats>0` (fail-loud);
OFF builds neither head.

### 🚨 Reading `opp_intent/*` — take the `_pool` suffix, not the bare key

Every metric is emitted **pooled AND per opponent class** (`_bot` / `_pool` / `_stable` /
`_exploiter`, a class appearing only when it holds ≥2 supervised rows in the minibatch —
`OPP_CLASS_NAMES`, mirroring `MaskableAgentWrapper.OPP_CLASS_*`). **`_pool` — frozen selves — is
the one that measures the thing the head is for.** Against the random bot the optimal prediction is
uniform and the achievable gain is ~0 BY CONSTRUCTION; against a heuristic it is easy but models a
decision tree rather than a player. Measured on gen-11: bot info gain **0.124 nats** vs pool
**0.254**, with bot accuracy flat at ~0.50 all run.

**The bare key is a MIX, and the mix MOVES.** Supervised rows ran **100% bot at 2M and ~7% from 6M
on** (self-play competence-gating), so a pooled metric rises as the mix shifts toward the pool and
that rise is indistinguishable from the head improving. The pooled α accuracy at 2M read 0.580 —
which was a pure bot measurement; the pool figure at the same step was 0.296. Any trend that spans
the ramp is uninterpretable; a trend after ~6M happens to be safe, but read `_pool` and do not rely
on that.

The split covers **every** axis: the KIND decision both directions (`alpha_switch_recall` /
`_precision`, `alpha_move_kind_recall` / `_precision`), the move axis (`alpha_move_recall_top1` /
`_top2` against `alpha_move_baseline_argmax_w` — compared LIKE FOR LIKE, both "given they moved"),
the β pointer (`beta_recall_top1` / `_top2`, `beta_info_gain_nats`), and the switch-coverage matrix
(`beta_switch_to_revealed` / `_hidden_found` / `_hidden_missed`, which partition voluntary switches
and sum to 1, plus `beta_belief_miss_rate` over the rows that ASKED). It used to cover only
accuracy / info-gain / count, which left exactly the metrics a reader uses to LOCATE a deficit
pooled. `alpha_mask_rate` stays whole-batch: it is the BELIEF's coverage failure, and folding it
into "α was wrong" would hide which component to fix.

One computation serves both reads — `_alpha_subset_metrics` / `_beta_subset_metrics` /
`switch_coverage_metrics` take a row subset and a suffix — so a pooled and a stratified number can
never drift apart. `switch_coverage_metrics` is module-level rather than a closure in the PPO loop
because nothing tested that matrix at all, and a metric with no test can silently read zero.

**Interpretability is a first-class output, not a debug aid** (`render_alpha` → the trace's
`opp_intent` block): `α` as a ranked list of NAMED moves. The owner constraint is that the model may
only ever point at options it can name, and rendering is where that becomes checkable.

**The path from the head to a human is WHOLE**, and it is worth naming because for one commit it was
not — `RLPlayer._opp_intent` built the block and `BattleRecorder` never wrote it, so the payload was
computed on every decision and dropped on the floor:

`α`/`β` logits → `RLPlayer._opp_intent` (`render_alpha`, plus `belief_decode.top_species_per_slot`
to NAME `β`'s slots) → the summary invocation's `opp_intent` block → `engine.build_opp_intent` /
`opp_intent_text` → the prober's Summary **EXPECT** line, `analyze`'s `opp_intent`, and the web
replay's per-turn *expect* line (`src/main/prober/CLAUDE.md`). `β` names a slot by the model's OWN
species posterior — the same content-addressing its training target uses — so the head and the
sentence refer to one object.
