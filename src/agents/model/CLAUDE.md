# Model Directory — Contributor Notes

**This leaf is a CONTRACT and a HAZARD LIST** — the rules a change to the model must follow, the
footguns that have already fired, and a map. **[`designs/ARCHITECTURE.md`](../../../designs/ARCHITECTURE.md)
is the only document that states the model AS IT IS NOW**; read it first, and where it and this file
disagree, ARCHITECTURE.md wins. History: [`designs/CHANGELOG.md`](../../../designs/CHANGELOG.md).

The split is deliberate: this file holds the **rules** a phase must follow (durable), and
`ARCHITECTURE.md` holds the **state** the model is currently in (changes every run). When you touch
`features_extractor.py`, update the contract here if a rule changed, and `ARCHITECTURE.md` if the
state did — then regenerate the delivery graph.

## Where the detail is — the topic map

**`designs/model/` owns the detail and carries the same always-current obligation as this file —
update the topic doc in the same pass as the code.**

| I am about to touch… | Read |
|---|---|
| the phase chain, the tier order, the T0/T1 belief+physics stack, what a phase does | [`designs/model/phase_pipeline.md`](../../../designs/model/phase_pipeline.md) (`ARCHITECTURE.md` §2.1 first) |
| any readout off `value_pooled` — win-prob, the cf heads, `QWinProbHead` — or a critic delivery route | [`designs/model/readouts_and_value_routes.md`](../../../designs/model/readouts_and_value_routes.md) |
| which file a class lives in, the table layering, the extractor class chain | [`designs/model/file_layout.md`](../../../designs/model/file_layout.md) |
| a stash surface, or the `torch.compile` refusal | [`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md) |
| adding, demoting or deleting a model flag | [`designs/model/flag_registry_rules.md`](../../../designs/model/flag_registry_rules.md) + the GENERATED [`designs/flag_registry.md`](../../../designs/flag_registry.md) |
| `model_version/`, a config bump, a deleted kwarg, a resume-immutable hparam, `--critic`'s gate | [`designs/model/versioning.md`](../../../designs/model/versioning.md) |
| the opponent-intent readout (X5's flat pointer re-expressed as α / β), its metrics, or a new α consumer | [`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md) |
| the delivery graph, the architecture viewer, where the canonical architecture lives | [`designs/model/architecture_artifacts.md`](../../../designs/model/architecture_artifacts.md) |
| a type annotation, or the mypy gate's scope | [`designs/model/typing.md`](../../../designs/model/typing.md) |

Closed history — **do not update it, do not re-derive a plan from it**:
[`designs/research_state/claude_md_archive/model_leaf_history.md`](../../../designs/research_state/claude_md_archive/model_leaf_history.md).

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

**`role_input_dim` is not a module-level constant** — it is computed dynamically in `PokemonEncoder.__init__` from the layout fields and `MOVE_NET_HIDDEN`. You do not need to update it manually when dims change; it is derived correctly. The projection input dims are likewise derived — static arithmetic in `compute_projection_widths` (`gen3_static_widths_v1`), verified per flag combo by `projection_width_test.py`.

## Phase module structure

**[`designs/ARCHITECTURE.md`](../../../designs/ARCHITECTURE.md) §2.1 states the chain and the four
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

**Dual-head value readout (H4 / Option C).** The transformer body is shared, but the actor and
critic read it through independent paths. `CLSPool` holds a third query `value_cls` that attends
over all 12 team tokens to produce `value_pooled`; `ProjectionAssembler.forward` returns a
`(pi_combined, vf_combined)` pair; and the root `forward` returns a `(pi_features, value_pooled)`
tuple — the value half has NO projection (`vf_features_dim` = `D_MODEL`; architecture audit F1, the X5
version break's part 2). This extractor therefore **must** be paired with `Gen3DualHeadMaskablePolicy`
(`policy.py`), which keeps `share_features_extractor=True` (one body) and overrides `forward` /
`evaluate_actions` / `get_distribution` / `predict_values` to unpack the tuple: the policy half goes to
`mlp_extractor.forward_actor`, and the value is `_critic_value(vf)` = `sigmoid(fe.last_win_prob_logits)` (the
win-prob head on `value_pooled`, the only critic; `vf` is read only for its batch size). There is NO value
tower: the policy's `_build` does not call SB3's — it builds an ACTOR-only `MlpExtractor` (`vf=[]`), SB3's
orthogonal re-init (extractor, then mlp extractor, gain √2), the retire hooks, the pointer head and the
optimizer, and `action_net` / `value_net` are RAISING stubs (`_NoFlatActionNet` / `_NoValueNet`). A stock SB3
policy expects a single-tensor extractor and will break — doubly so under the pointer-native action head
(`gen3_pointer_native_v1`): the action logits come from the `PointerNativeActionHead` over
the extractor's `last_pointer_inputs` stash (per-logit inputs: `designs/ARCHITECTURE.md` § Heads). The startup `_run_roundtrip_test` and the snapshot/feature tests all
unpack the tuple — keep that in mind when touching the extractor's return shape.

**`value_pooled` is vf-only by construction.** `vf_combined` IS `value_pooled` and `pi_combined`
is a concat that never contains it, so anything injected into `value_pooled` leaves `pi` bit-identical
at ANY weight; the extractor's `_value_pooled_routes` seam has ONE member (`value_entity_pool`). The
privileged true-team route and the dense auxiliary head that once read off it are deleted.

**The DETACHED RIDE-ALONG heads live on the POLICY, not the extractor** (`gen3_ridealong_heads_v1`,
v126, `ridealong_heads.py`). The extractor only RECORDS `ridealong_{ensemble,rnd,adv,opp}` (so the
flag registry, the version gate and every worker rebuild see them); `Gen3DualHeadMaskablePolicy`
builds `policy.ridealong` in `__init__` AFTER `_build`. Three rules keep "the baseline learns exactly
what production learns" true, and each has a test that fails on revert: **(1)** build inside
`torch.random.fork_rng` from `RIDEALONG_INIT_SEED` — a module built from the global RNG shifts every
later draw (init, rollout sampling, minibatch shuffles); **(2)** never put them in `policy.optimizer`
or fold their loss into PPO's `loss` — PPO's global `clip_grad_norm_` would include their gradient
and rescale the trunk's; **(3)** every input goes through `RideAlongBatch.detached`. A frozen random
network (RND target, randomized prior) is a BUFFER (`freeze_to_buffers`), never a
`requires_grad=False` parameter. **The RND VARIANTS** (`ridealong_rnd_variants`, v127,
`RND_VARIANT_DECLS`) are built LAST in `RideAlongHeads`, each from its own private seed or a deep copy
of base's predictor, so adding one never changes another head's init. B reads X5's FLAT opponent
pointer (`FlatOppEffectEnsemble`; labels from `flat_intent.flat_intent_targets`) — still detached, pinned
bit-identical to learning by `ridealong_update_test`. They are excluded from
`trainable_parameters()`: each has its own optimizer (`variant_parameters(name)`). The observation
variants share base's target and normaliser, so they must never own copies of them. Detail: [`designs/model/readouts_and_value_routes.md`](../../../designs/model/readouts_and_value_routes.md).

### Phase-by-phase data flow
The per-phase walkthrough and the static-width arithmetic:
[`designs/model/phase_pipeline.md`](../../../designs/model/phase_pipeline.md). Two things stay here.

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

## File layout (one responsibility per file; phases split 2026-08-16)

**Every file, its one responsibility, and the split rounds that produced it:**
[`designs/model/file_layout.md`](../../../designs/model/file_layout.md). The map:

| you are looking for | file |
|---|---|
| the architecture constants | `arch_constants.py` |
| the extractor: `__init__` · the `last_*` surface · `forward_internal` · the class + `forward` | `extractor_build.py` · `extractor_api.py` · `extractor_forward.py` · `features_extractor.py` (the re-export HUB) |
| the phases | `extractor_ctx.py` · `encoders.py` · `team_transformer.py` · `pools.py` · `belief_heads.py` · `projection.py` |
| the op | `damage_op.py` · `index_max.py` (`max_by_index`, every gradient-path max — a leaf) · `damage_op_layout.py` · `damage_op_pairwise.py` · `damage_op_blocks.py` · `damage_op_speed.py` (`--speed-physics on`'s inputs) · `move_order.py` (THE move-order rule: priority bracket + speed physics) · `damage_kinds.py` (the non-formula damage + Beat Up's exact party terms every kernel applies) · `status_rules.py` (the incoming side / clause status rule the op and the move-resolution family share; whether a clause is in force is `agents.gen3_data.format_spec`'s call, never a constant) |
| the lookup tables, in LAYER order | `damage_tables.py` → `belief_tables.py` → `dex_ids.py` |
| the readouts and the critic routes | `aux_value_heads.py` · `q_winprob_head.py` · `value_readouts.py` · `value_threat_inject.py` |
| the pointer head and the per-action cells | `pointer_head.py` · `pair_outcome.py` · `switch_branch.py` · `conditional_threat.py` · the move-resolution family that replaces them under `--move-resolution on` (`move_resolution.py` · `move_resolution_rules.py` · `move_resolution_tables.py`) |
| versioning, snapshots, the compile path, the critic modes | `model_version/` · `snapshot.py` · `compile_opponents.py` · `critic_mode.py` |
| the DICT obs keys the forward reads beyond `observation` | `extra_obs_keys.py` |
| X5's dex-row table (a hypothesised opponent mon's per-mon obs row, per species; the generator, the loader, the committed artifact) | `hypothesis_dex_rows.py` + `hypothesis_dex_rows.json` |
| X5's T0 hypothesis builder (built with the opponent-belief family — the only belief representation: δ_θ, the fixed-size presence, the one stable ordering, OTHER, the active's move group; the set-BCE helpers) | `hypothesis_set.py` |
| X5's hypothesis-token ENCODING (`PokemonEncoder` split exactly at its two first Linears: the species half once over the dex table, gathered; the row-level half per row; the rest per opponent slot — the per-row pass on `hypothesis_ctx` stays the definition its test compares against) | `hypothesis_encode.py` |
| the STATIC per-mon encoder (`--token-encoding static`: S = the static identity from the set fields, D = the mon's own state, added, no board input; the X5 hypothesis tokens as the dex table encoded once and gathered) | `static_tokens.py` |
| the STATIC arm's BOARD (`--token-encoding static`, stage 2: the side-relative SIDE / FIELD content the three board tokens project, the seat tuples, the per-mon op content `OpContent`) | `board_tokens.py` |
| the OBS-FACTS block's consumer (`--obs-facts v1`: the zero-init `ObsFactsInject` — the facts as token content; `FACTS_TOKEN_CLASS` routes each sub-block, a SIDE fact to the static board's side tokens) | `obs_facts_inject.py` |
| X5's hypothesis TOKENS in the chain (the hypothesis context, the spliced tokens, the per-key log-presence, the class-E pools' float masks, the op's opponent-MON roster `OpRoster` + OTHER's averaged `other_roster`) | `hypothesis_tokens.py` |

🚨 **THE FORWARD HAS TWO PUBLIC SURFACES: the constructor signature, and the obs DICT's KEY SET.**
`forward` is normally a pure function of `obs["observation"]` — but a route may read a flag-gated
Dict key of its own and **RAISE** when it is absent (a silent skip reads exactly like a route that
learned nothing). The privileged true-team route's `opp_true_team` was the first (deleted; the registry is now EMPTY and
the mechanism + its drift gate stay). That mapping is DECLARED once
in `extra_obs_keys.py` as `(extractor attribute -> key, shape, canonical zero block)`, keyed on the
ATTRIBUTE the forward itself tests, and every synthetic-obs caller on a TRAINING path builds from it
via `zero_extra_obs` / `synthetic_obs`: `main/train/lifecycle.py`'s round-trip
smoke, `compile_trainer`, `compile_opponents`, `agents/training/churn_probe.py`'s `masked_action_probs`. **Never hand-build
`{"observation": zeros(1, D)}` on a path a run reaches** — that literal is what killed
`ai_v12_14_ladder_truevalue` two minutes into its launch, and `extra_obs_keys_test.py` reproduces
it plus an AST gate that fails on any obs key the table does not declare. The ~17 OFFLINE audit /
probe CLIs still hand-build (they fail at a terminal, not on the GPU) — see
[`designs/ops/TECH_DEBT_BACKLOG.md`](../../../designs/ops/TECH_DEBT_BACKLOG.md).

🚨 **The table layering only ever points DOWN** (`damage_tables` → `belief_tables` → `dex_ids`), and
**an import back closes a cycle Python resolves only for whichever module was imported first** — it
would work in the normal import order and raise in every other. Do not add one;
`belief_tables_test.py` AST-scans every up-edge and also fails if a name is DEFINED in two of the
three. Every one of these tables is `persistent=False`, so a relocation moves no `state_dict` key.

### The extractor CLASS is a base-class CHAIN (`gen3_extractor_class_split_v1`, 2026-08-23)
`ExtractorBuild` → `ExtractorApi` → `ExtractorForward` → `Gen3FeaturesExtractor`, one file each,
`features_extractor.py` holding the class and `forward`. Why inheritance rather than helper
functions, why `__init__` is not split further, and the module-GLOBAL patching hazard a test author
must know: [`designs/model/file_layout.md`](../../../designs/model/file_layout.md).

**⚠️ `forward` stays on `Gen3FeaturesExtractor`, and must.** Both compile flags patch the BOUND
`fe.forward`; the capacity probe calls `type(fe).forward` for a deliberately-EAGER pass; and
`instrumented_ppo_test` ASSIGNS `type(fe).forward` and restores it. An attribute defined on a base
would be SHADOWED by that assignment and the restore would leave the shadow in place forever.

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
bundle recorded in `designs/research_state/claude_md_archive/model_leaf_history.md`, on 64 real
gen-9 eval states across three config arms.

🚨 **`last_tensors` is POST-gain and is for PROJECTIONS only.** The op's learned `out_gain` (one scalar per
(region, channel) with NO position in its key — tied across request slots / move seats, the X5 version break's part 4,
and across our team slots / their mons, `gen3_mon_tied_gain_v1`; `damage_op_layout.out_gain_channel_keys`) is a
projection adapter, not physics. A consumer that uses an op value AS a probability or a
damage fraction (P(first), a roll, a secondary chance) reads it PRE-gain: a typed pre-gain stash, or the block's
channels through **`last_raw_tensors`** (live views; `last_raw_block` is the detached prober copy — never a
training-path input). `intent_conditional` and the move-resolution family do (part 5;
`x5_version_break_part45_test.py`). A new region of the block needs a key in `out_gain_channel_keys` (the op
raises at build if the key walk and `out_dim` disagree), and its key names the channel only — a request slot, seat,
team slot or listing position is arbitrary and is never part of it (`mon_tied_gain_test.py` permutes team slots).

**The op's SIDE VALUES and the EXTRACTOR's follow one contract** (`gen3_op_stashes_v1` /
`gen3_extractor_stashes_v1`): every per-forward stash lives in ONE dataclass the forward replaces at
ENTRY, **reads** go through the `last_*` properties and **writes** through `op.stash.<field>` /
`fe.stash.<field>` — writing a `last_*` name raises. Add the dataclass field with its shape comment,
never a bare `self.last_x = …`; each producer owns its own stash surface and a submodule never writes
into its parent's. Gate: `extractor_stashes_test.py`. Detail:
[`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md).

🚨 **THAT CONTRACT IS SINGLE-THREADED, AND `forward` IS NOT RE-ENTRANT** (`gen3_extractor_forward_guard_v1`).
"The forward replaces the stash at ENTRY" makes a stale read unrepresentable only while ONE forward
is in flight per extractor — true of training (each env worker is its own process) and false the
moment two threads share a model object, which `main.search_dividend`'s mirror does by design (the
searched side runs off POKE_LOOP; the other side, and every playoff rollout player, decides on it).
Measured 2026-09-22: two threads, one real extractor, 2,400 interleaved forwards ⇒ **1,063 failures
in seven classes**, including `ValueThreatInject shape mismatch: tokens (1, 6) vs rows (9, 6)`. ⚠️
**The crash is the lucky case** — same-batch-size forwards corrupt each other silently. A
thread-sharing caller installs `agents.model.forward_guard.install_forward_guard(extractor)` (or
`install_model_forward_guard(model)`) and holds the returned RE-ENTRANT lock across the forward
**and** the `last_*` reads that belong to it. The guard is **OPT-IN and absent by default** — kept
in a weak-keyed registry, never as a module attribute (an `RLock` there breaks
`copy.deepcopy(policy)`), and the unguarded path contains no context manager at all so the compiled
graph is unchanged. Gate: `forward_guard_test.py`.

## 🚨 Our active's moves live in TWO orders — cross them by IDENTITY only (`gen3_move_legality_by_id_v1`)

The per-mon move slots (`ctx.all_move_ids`, every encoder) are SORTED BY `Move.id` STRING; the request
block (`ctx.our_active_req_move_*`) and actions 6–9 are in REQUEST order. **Never apply a request-order
tensor to the per-mon slots by position, or a per-mon tensor to the request slots** — go through
`extractor_ctx.active_request_sorted_match` (or `active_move_legality_sorted`), the ONE rule the pointer
head, `PokemonEncoder` and the static encoder share. The positional form fired twice (the old prev-turn
mask; then `PokemonEncoder`'s legality from `bcdd868b` to this fix, wrong on 6.8 % of real move-bearing
decisions) and is silent whenever every move is legal. `move_legality_alignment_test.py` fails on it;
`agents/action/ordering_integrity.check_obs_move_order` RAISES on a served row that breaks the rule's
preconditions.

## 🚨 An OPPONENT's ability is revealed only by its `known` flag (`gen3_op_ability_known_v1`)

An unrevealed opponent's ability slot carries its species' TOP-1 Smogon-prior ability in `id1` with `known = 0`.
The op read `ability1_ids > 0` as "revealed" until 2026-10-07 and so asserted the guess as CERTAIN in every
damage / status / secondary kernel (Toxic "never landed" on an unrevealed Snorlax). **Read an opponent's ability
through `extractor_ctx.ability_known` / `revealed_ability1_ids` (the op's `opp_ability_view` + `_known_or_prior`)
and take the Smogon species marginal when it is unknown.** Status landing has ONE rule per direction
(`_outgoing_status_land`; `_incoming_dedicated_land` + `status_rules.incoming_status_mask`). Gates:
`ability_known_gate_test.py` (AST, allowlist EMPTY), `op_status_rules_test.py`; detail:
[`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md).

## 🚨 Every DISCRETE op in the forward is DECLARED (`selection_sites.py`, `gen3_behaviour_tie_exclusion_v1`)

The forward is piecewise-discontinuous: a `topk` / `argmax` / comparison whose operands sit within a
rounding error of the cutoff resolves differently in T2's compiled forward and the learner's eager one,
and log π jumps on that row. K9(b) EXCLUDES those rows by their TIE MARGIN and judges the rest
deterministically (`designs/training/learner_gates.md`), so it must know every such op. **A new
selection call, value-position comparison or float → int cast in a `FORWARD_MODULES` module must be
declared in `selection_sites.py`** — a `MARGIN` rule when its operand is a score (it moves with the
weights, or is float arithmetic a rounding error can push across the cutoff), an `EXACT` reason
otherwise (observation / table / integer / gathered at a declared selection). `selection_sites_test.py`
fails on an undeclared or stale entry, on a line mixing the two classes (the recorder resolves ops by
LINE — split it), and when a site declared EXACT moves under a few-ulp weight jitter; at run time an
undeclared op on a float operand is a typed FATAL at the first update. The keys are source text, so
editing a declared line means re-declaring it. 🚨 **An `argmax` MARGIN rule may declare its `payload`**
(`gen3_behaviour_tie_identity_v1`: the frame-local tensors its index gathers — a tie between candidates
whose payloads are bit-identical is then no tie). The payload must be EVERYTHING the index reaches:
`selection_sites_test` fails when the index is read anywhere but a `gather` of a declared payload, so a
new consumer of a payload site's index means extending the declaration (or dropping it). 🚨 **Every
`hypothesis_set.stable_order` caller declares how it READS the order** (`consumed`,
`gen3_behaviour_tie_consumed_v1`): a `SetCuts` (a set before each cut: the op's per-mon move orders), a long
count (in order up to it: the species order), or None (every pair: the move group). A new reader of an order
declared `SetCuts` that reads its positions IN ORDER makes the declaration unsound —
`tie_identity_integration_test` permutes every set prefix and requires log π bit-identical.

## 🚨 X5's hypothesis tokens: every reduction over OPPONENT tokens declares its presence semantics

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

**The op's opponent-MON axis under X5 reads `op.stash.x5` (an `OpRoster`), never a gate of
its own.** Every opponent-slot kernel in `damage_op*.py` takes "alive" from `roster.alive`
(`opp_addressable` — a hypothesis row reads HP 1.0, and no gate may depend on that), the per-mon
candidates from `roster.move_order` / `move_w`, and (in an OTHER-mode pass) the tail-averaged tables
through `_x5_avg` / `_x5_stab`. A NEW opponent-axis kernel must branch on `self.stash.x5` the same way,
get a "no hidden mon dropped" row in `x5_opp_mon_axis_test.py`, and — if OTHER should see it — an entry
in `EdgeBias.OTHER_FAMILIES` plus a call in `_other_edge_cells`. A per-(seat, mon) cell is "what this mon
does IF present": never multiply it by π (the trunk's key bias carries presence); only a reduction over
the MON axis is presence-scaled (`p_pur_vs_us`).

## ⚠️ One op's SPELLING is load-bearing for `torch.compile` (`gen3_species_posterior_spelling_v1`)

`BeliefHead.species_posterior` computes `P(species)` for the expected-latent defender. It is written
as **`log_softmax(...).exp()`, not `torch.softmax(...)`, and that is deliberate** — do not
"simplify" it.
`extractor_compiles_test.py` owns the compile matrix and pins the spelling (a real compile of the
production arch with suppression OFF; `GEN3AI_SKIP_COMPILE_TESTS=1` opts out, `GEN3AI_TEST_ALLOW_GPU=1`
for the CUDA cells). The Inductor diagnosis:
[`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md).

🚨 **A second load-bearing spelling: EVERY gradient-path value-reduction max is `max_by_index`, never `amax`**
(`index_max.max_by_index`, a LEAF module; `damage_op` re-exports it; architecture audit F6a, after
`gen3_fm_index_max_v1` / F-XC-4). `amax`'s backward divides by `Σ(x == amax)`; when Inductor RECOMPUTES
`x` in the backward kernel and Triton's FMA contraction rounds it differently from the forward kernel,
no element equals the saved max and the compiled gradient is 0/0 = NaN (CUDA, 41 parameters, 2026-10-05).
`max_by_index(x, dim, keepdim)` gathers at the detached argmax: the same value bit for bit, a scatter
backward, and on an exact tie the WHOLE gradient goes to the FIRST maximum (the declared convention). It covers
the op's ten incoming channel maxima in EVERY configuration (belief on or off), the pairwise kernels, the
status-landing maxima, the E5 tail's worst-phys/spec and `pair_reduce`'s inert pool; a NEW max over candidates on
a gradient path uses it. `amax` / `amin` are legal only OFF any gradient path (a comparison operand, a table
lookup, a constant, an observation indicator, a `no_grad` bracket, a diagnostic) — the list:
[`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md) "The op's MAXIMA". Its argmax is
K9(b)'s one `MAX_VALUE` EXACT site (`selection_sites`; the index may only gather its own operand —
`selection_sites_test` pins it). Detail: `designs/training/compile_flags.md`.

🚨 **A third: every FLOAT attention bias reaches SDPA through `dense_attn_bias`** (`gen3_dense_attn_bias_v1`,
F-ST-8). Under Inductor, a bias built by in-place slice writes (`EdgeBias._write_block`'s head-innermost
`m.permute(0, 3, 1, 2)`) is a FLEXIBLE buffer, and Inductor may lay it out head-innermost. Inductor's own SDPA
stride constraint passes it unfrozen whenever the key count is a multiple of 8. CUDA's efficient kernel then
raises "(*bias): last dimension must be contiguous". `--token-encoding static` (64 keys) died on this in the T2
service's first graph build; legacy (62 keys) took the padded-copy branch and never did. `dense_attn_bias` pins the
bias row-major (Inductor's `inductor_force_stride_order`; its identity backward is registered at import) and is
`.contiguous()` in eager. A NEW SDPA call with a float mask goes through it. `dense_attn_bias_test.py` fails when a
call site drops it. Its CUDA test reproduces the failure without the pin. A CPU compile cannot reproduce it,
because the constraint's CPU branch always requires the stride order.

🚨 **A forward reads PLAIN INTS precomputed in `__init__`, never a module attribute holding a sub-dict of `layout`**
(`gen3_static_layout_ints_v1`, F-ST-9). `ObsUnpack.layout` reaches the same object, so dynamo installs an
object-aliasing guard. On the `static` arm's CUDA launch that guard recompiled the learner's region after the
compile lock (`[CompileSentinel] FATAL` at update 1), although the two objects stayed identical.
`static_tokens_test.py` fails when the static encoder holds a layout container.

**The general lesson:** a backend that "can't compile our model" was one op, not a property of the
architecture. Before reaching for a global suppression flag, bisect to the op — see
`designs/training/compile_flags.md` → Compiled CPU opponents.

## ⚠️ Identity-at-init is NOT free — SB3 clobbers it (`gen3_identity_init_guard_v1`)

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

## The flag registry — read it BEFORE adding a toggle

**A model-relevant toggle is one `ModelFlag` row in `agents/model/flag_registry.py`, and five
hand-synced surfaces** — the `argparse` entry (**defaulting to `None`**, or its `_resolve` line is
dead code), the `_resolve("name", default)` line beside it, `extractor_arch.ARCH_ARG_KEYS`,
`snapshot.current_model_version()`'s keyword, and the `ModelVersion` field. `flag_registry_test.py`
fails with a message NAMING the missing site; every historical failure in this class was silent.
Four more rules govern the row, all of them enforced, none of them guessable:

- **Three TIERS** — `cli` / `config_only` / `constructor_only`. A `config_only` toggle is FROZEN at
  its registry `default` for every CLI-launched run, so that default must be what production wants.
- **Four CLASSES** — `structural` / `resume_immutable` / `training_coef` / `runtime` — decide which
  gate a mismatch gets, and getting it wrong hurts in BOTH directions.
- **`requires=`** is the dependency data. `requirement_closure(name)` gives the transitive set,
  `flag_requires_test.py` enforces it forward and in reverse, and `python -m main.checkargs` reads
  the graph so an unsatisfiable command is reported offline instead of crashing a launch.
- **Read [`designs/flag_registry.md`](../../../designs/flag_registry.md)** for the current table
  (GENERATED; `--check` is the gate).

All four in full, with the demotion history and the reachability gate that found five live flags in
the dead-`_resolve` state: [`designs/model/flag_registry_rules.md`](../../../designs/model/flag_registry_rules.md).

🚨 **`oracle_reveal` (v137) is the registry's one OBSERVATION-MODE row** — a `resume_immutable` flag with no module and no
weight: `--oracle-reveal {off,species,full}` makes the Rust encoder write the opponent's true species (`full`: the whole set) into the observation
(a DIAGNOSTIC for the X5 A/B's oracle arm, `designs/endstate/design_x5_belief_tokens.md` §7.6; never production). The
extractor only stores it. It is recorded, inherited by a flagless resume and refused on a flip (`check_oracle_reveal`),
kept off the ARCH surface and out of `check_compatible` by its class, and `oracle_reveal.py` is the reader a tool
uses to REFUSE such a checkpoint when it builds observations without the reveal (`main.anchors`, `main.play`,
`main.belief_roles`, `main.policy_spectrum`). `main.h2h` PLAYS one, at its recorded level, through its per-side reveal
(`--oracle-reveal-mode one_sided|both_sided`, `main.h2h.reveal`) and refuses one under its default `off`.

## Model versioning (`model_version/`, `snapshot.py`)

**`model_version` is a PACKAGE**; `__init__.py` is a pure re-export hub. What each module holds, what
a save writes, the two sanitizers and the recorded-`critic` version gate:
[`designs/model/versioning.md`](../../../designs/model/versioning.md). The playbooks are here.

**When you change an architecture constant:**
- `check_compatible()` catches the mismatch automatically — no extra steps needed
- Old models can't be loaded, which is correct (rapid iteration project)

**When you add an optional new feature** (new field with a sensible default):
1. Add the field to `ModelVersionFields` in `model_version/fields.py`
2. Bump `MODEL_CONFIG_VERSION`
3. Add one `if version < N:` block in `_migrate_config()` with `data.setdefault(...)`

🚨 **Every value `--arch production` writes must be a `ModelVersion` field**, including a DERIVED
row's enabling coefficient. A launcher restart strips `--arch` and inherits only what
`model_config.json` records. `opp_intent_coef` was not recorded until v125, and every such restart
FATALed. When the past value is unknown, the migration leaves the field `None` rather than invent it
(the v125 branch). Gate: `main/train/derived_toggle_resume_test.py`.

**When you make a structural change** (different forward pass, new layer type):
1. Change `ARCH_SIGNATURE` in `model_version/constants.py` (e.g. `"gen3_attn_v1"` → `"gen3_lstm_v1"`)
2. Old models get a clear arch-family error on load

🚨 **The X5 VERSION BREAK (config v144, `gen3_x5_version_break_v1`) RAISED `MIGRATION_FLOOR` to 144**: every
pre-break checkpoint (blob or fixed_mass) is refused at the floor with the belief-specific reason and runs PINNED
(`model_version/version_break.py` — `LAST_BLOB_COMMIT`, `check_post_break`, the pickled `belief_tokens` judgment).
The post-floor `if version < N` branches (v122–v143) are unreachable and left in place (legacy manifest R1 / L1).
**v145 (`gen3_mon_tied_gain_v1`) raised the floor to 145** (the op's `out_gain` tied across team slots): a v144 config
is refused with `version_break.v144_reason()` and runs pinned at or before `LAST_V144_COMMIT`.

**When you DELETE an extractor kwarg** — the case with no automatic gate, and the one this project
has silently got wrong five times. Every archived checkpoint keeps the deleted name pickled in
`policy_kwargs["features_extractor_kwargs"]`, and **SB3 rebuilds the extractor from the ZIP, not
from `model_config.json`** — so a deleted kwarg `TypeError`s every training resume, frozen pool
opponent and eval worker that touches such a checkpoint. `_migrate_config`'s `MIGRATION_FLOOR` does
NOT cover it: the pickled kwargs carry no `config_version` for a floor to apply to.

The judging procedure — `_DEAD_FEK_INERT` vs `_DEAD_FEK_JUDGED`, when `_migrate_config` needs a
matching entry, and `ctor_kwarg_snapshot_test.CTOR_KWARGS_V96` LAST — is in
[`designs/model/versioning.md`](../../../designs/model/versioning.md). Do all three, in that order.

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

**Resume-immutable training hparams (value-meaning, NOT weight-shape)** — `vf_coef`, the reward
fields — are recorded on `ModelVersion` but **deliberately excluded from `check_compatible`**, which
gates EVERY load including the frozen eval / pool opponents whose forward is identical
regardless. They get a dedicated `check_*` on the training-resume path only, and `train_rl_agent.py`
FATALs on a mismatch exactly like an arch error. To add one: field + `MODEL_CONFIG_VERSION` bump +
`_migrate_config` default, **plus** a dedicated `check_*` and an `enforce_*` opt-in on
`load_model_snapshot`, and leave it out of `_WEIGHT_FIELDS`. The family list and the 2026-08-18
default flip: [`designs/model/versioning.md`](../../../designs/model/versioning.md).

The live `MODEL_CONFIG_VERSION` is in `model_version/constants.py`; per-version entries are in `designs/CHANGELOG.md`.

- **What the architecture IS right now**: [`designs/ARCHITECTURE.md`](../../../designs/ARCHITECTURE.md).
- **What each version changed**: [`designs/CHANGELOG.md`](../../../designs/CHANGELOG.md) — history, do not quote as current.
- **The live values**: `MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` in `model_version/constants.py`. Read
  them there. This file deliberately states neither: a version number in prose is stale the moment the
  next one lands, and quoting a stale one is how a v30 description got applied to a v59 model.

The mechanics above (what to bump when, the optimizer-reorder guard, the resume-immutable-hparam
playbook) are the durable part and stay here. When you add a toggle, follow those rules, then record
the entry in `CHANGELOG.md` and state the new truth in `ARCHITECTURE.md` — never narrate it here.

A startup smoke test (`_run_roundtrip_test` in `train_rl_agent.py`) saves to a temp dir and reloads before every `model.learn()` call — catches serialization issues immediately.

## The CRITIC MODE (`critic_mode.py`, the recorded `critic` field {shaped,winprob}, v109)

`gen3_winprob_critic_mode_v1`. `critic` is a RECORDED field with two historical values, and
`agents/model/critic_mode.py` is the ONE declaration of the legal set; since the X5 version break's part 2
(architecture audit F1) the policy BUILDS only `winprob` — its `critic` kwarg defaults to `winprob` and any other
value is refused before anything is built (the scalar `value_net` a `shaped` critic read is deleted with the
whole SB3 value tower). That module is
deliberately **torch-free and import-light**: `main.checkargs` promises not to import torch and
needs the legal set to validate an argv offline. There is NO `--critic` flag any more (deletion pass P11b batch (b)): the win-prob critic is a CONSTANT of every trainer namespace, and `critic` survives as the FIELD `model_config.json` records, string-compared by `check_compatible` (an absent record still means `shaped`).

| recorded `critic` | `_critic_value` returns | at HEAD |
|---|---|---|
| `shaped` (what an ABSENT RECORD means — `CRITIC_UNRECORDED`, for a `ModelVersion` / `model_config.json` read) | `value_net(latent_vf)` (historical) | NOT constructible: the policy refuses it, and every such checkpoint is below `MIGRATION_FLOOR` 144 — run it PINNED to its own commit |
| **`winprob`** (the only critic; the policy kwarg's DEFAULT) | `sigmoid(fe.last_win_prob_logits)` in **[0,1]**, `[B,1]` | the only one built; no value tower exists |

**Read the mode through `is_winprob`, never a bare `== "winprob"`** — one spelling, one answer, and
a `getattr(obj, "critic", "shaped")` read answers correctly through it.

🚨 **The version gate matters more here than for a typical structural flag: BOTH routes return a
`[B,1]` float tensor**, so a flipped `critic` produces no shape error, no load failure and no metric
that changes name — the run simply predicts a different quantity for the rest of its life. The string
compare in `check_compatible` is the only thing standing between a resume and that. (No
`ARCH_SIGNATURE` bump at v109 nor at the default flip; the version break's floor now refuses every
`shaped` checkpoint anyway.)
[`designs/model/versioning.md`](../../../designs/model/versioning.md) has both in full, and how the
kwarg is threaded.

## DELETED: PopArt, the distributional value head, `value_from_dist` (deletion pass L1, v131)

`popart.py`, `ValueDistHead` (`--value-dist-*`), the `value_from_dist` critic route, the CVaR value-tail
weight and the win-prob aux-BCE coefficient were levers of the shaped critic and of the Python env core, and
every v121+ checkpoint recorded them OFF. They are DELETED (`designs/deleted_flags.md`). The recorded fields
are popped by `_migrate_config`; a checkpoint that recorded PopArt / the dist head / `value_from_dist` ON is
refused on every load, and a resume or fork of a run that recorded ANY retired lever ON is refused
(`model_version/retired_levers.py` — each later deletion unit appends its levers to that one table). A
checkpoint's PICKLED `use_popart` / `value_from_dist` (policy kwargs) and `value_dist_*` (extractor kwargs) are
stripped on load by `snapshot._DEAD_POLICY_KWARGS_JUDGED` / `_DEAD_FEK_*` — a BARE `MaskablePPO.load` of a
pre-deletion zip TypeErrors (`play.py` and the prober's loaders sanitize; `snapshot.historical_load_kwargs`
is the helper). Detail: [`designs/model/versioning.md`](../../../designs/model/versioning.md).

## Opponent intent — X5's FLAT POINTER, re-expressed as `α` / `β` (`flat_intent.py`; `opp_intent.py` keeps the shared constants)

The build for one sentence the model could not express: *"they are likely to click **this**, so
**this** is my answer."* `--opp-intent-coef>0` turns on the SUPERVISED opponent-intent readout, which since
the X5 version break (v144) is X5's **flat pointer** (`gen3_x5_flat_pointer_v1`): ONE softmax over the
opponent active's K move seats, OTHER_move, a switch to each of their six slots and OTHER_species, scored by
one shared scorer plus each candidate's DETACHED log π. Its consumers read its RE-EXPRESSION
(`fe.stash.flat_consumer_ops`, `flat_intent.FlatConsumerOps`): **`α`** over the K seats + OTHER_move + the
switch mass, **`β`** over the six slots + OTHER_species. The pointer's shared scorer is BIAS-FREE (one scorer
over ONE softmax: a bias is a common shift, audit F16b) — unlike the POLICY pointer head's three per-FAMILY
scorer biases, which are not common to every logit and stay. The blob path's `AlphaIntentHead` /
`BetaSwitchHead` are DELETED, and with them the `alpha_logits` / `beta_logits` / `alpha_seat_nums` stashes and the
`last_alpha_logits` / `last_beta_logits` properties; the trace reader (`agents/inference/player.py`'s
`_opp_intent`) and `main/search_dividend/alpha.py` read the flat pointer (`last_flat_intent_logits` /
`last_flat_intent`). Why pointers, matching by canonical id, and why the label is shifted back
one row before `get()` shuffles: [`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md).

**Supervision only:** the pointer reads a DETACHED input (`opp_intent_grad_mode` `detached`), so a null
indicts its predictive power, not the policy. Structural + version-checked; requires the whole belief family
(X5's requirements, refused at build); OFF builds none of it.

### 🚨 Reading `opp_intent/*` — take the `_pool` suffix, not the bare key

Every metric is emitted **pooled AND per opponent class** (`_bot` / `_pool` / `_stable` /
`_exploiter`, a class appearing only when it holds ≥2 supervised rows in the minibatch —
`OPP_CLASS_NAMES`, mirroring `agents.training.opponent_classes`). **`_pool` — frozen selves — is
the one that measures the thing the head is for.** Against the random bot the optimal prediction is
uniform and the achievable gain is ~0 BY CONSTRUCTION; against a heuristic it is easy but models a
decision tree rather than a player. Measured on gen-11: bot info gain **0.124 nats** vs pool
**0.254**, with bot accuracy flat at ~0.50 all run.

🚨 **The bare key is a MIX, and the mix MOVES** — supervised rows ran 100% bot at 2M and ~7% from 6M
on, so a pooled metric rises as the mix shifts and that rise is indistinguishable from the head
improving. Any trend spanning the ramp is uninterpretable. `flat_mask_rate` and `other_label_rate` (the flat
pointer's, `opp_intent/*`; the blob α's `alpha_mask_rate` left with it, v144) are the coverage reads: a choice the
pointer's named candidates miss is an OTHER label, supervised, never masked. Full metric inventory and the head→human path:
[`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md).

#### 🚨 How a `β` slot is NAMED — two branches, and conflating them produced a wrong conclusion

`β` points at a SLOT, so something has to name it, and **where the name comes from decides what the
row means** (`gen3_beta_revealed_naming_v1`):

| slot | named from | trace flag |
|---|---|---|
| already REVEALED on the board | `battle.opponent_team`, read through `ObservationEncoder.get_team_list` so obs slot `k` and `β` candidate `k` cannot drift apart | `"revealed": true` |
| still HIDDEN | the model's OWN species posterior (`belief_decode.top_species_per_slot`) — the same content-addressing `β`'s training target uses | `"revealed": false` |

**Traces already on disk cannot be repaired** (they baked the posterior name and do not carry the
board), so the read side attaches `engine.BELIEF_NAME_CAVEAT` to any candidate not flagged
`revealed` rather than inventing a replacement name. See `src/main/prober/CLAUDE.md`.

### The rules an α CONSUMER follows (`pair_outcome.py` is the current template)

Nine modules now contract α against the op's physics (ten with `MoveResolutionCell`, `--move-resolution on`, which retires seven of them) (listed in
[`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md)). They share four
conventions, and each exists because breaking it fails silently:

1. **T1 produces, T2 consumes.** α is scored from the E4 seats and the CLS pools, both DOWNSTREAM
   of the op — so the op cannot reduce by α, and every consumer runs at the pointer stash. A
   "swap `_chan_max`'s `how=`" plan is unbuildable for that reason, not for want of a knob.
2. **Read the flat pointer's re-expression (`FlatConsumerOps.alpha` / `.beta`, built from the
   PUBLICATION), never a raw stash** — and take the **UNRENORMALIZED move slice** (OTHER_move included as a
   PRICED seat). The missing SWITCH mass is the literally-correct statement that a switching opponent applies
   no outcome this turn; renormalizing asserts they attacked.
3. **Align by CONSTRUCTION and fail loud on a width mismatch.** α's seats and the op's top-K are
   the SAME axis (`intent_axis_alignment_test`); broadcasting a mismatch would pair each α weight
   with the wrong opponent move while every shape check still passed — the named `op move-order`
   bug class.
4. **Zero-init the projection**, and let `restore_identity_init` capture it by observation (M1).

`pair_outcome.py` adds two worth copying — **stop-grad α unconditionally** on a policy-side consumer
(resting on `label_only` makes the route's EXISTENCE a function of a TRAINING flag), and **give α a
documented FALLBACK only if the flag can stand alone**, saying loudly that the R1 `belief_mean` rung
is a different object from the publication.

**⚠️ "Give it a fallback" is not the same as "a fallback is meaningful", and v94 is where the two
came apart.** `SwitchBranchMoveCell` REFUSES one and requires `opp_intent` instead: every coordinate
it computes is conditioned on `α_SWITCH` or on β, and the R1 rung is a presence belief over their
MOVES with no switch class at all — so the fallback would set `α_SWITCH ≡ 0` and make every
coordinate assert *"they never switch."* **A flag whose fallback silently states something false is
worse than a flag that says it needs the head.** The test for whether to build one is not "can I
compute something", it is "is what I would compute an ABSENCE or a CLAIM". Same rule kills the
tempting `softmax` over an all-`-inf` β row: that yields a UNIFORM arrival distribution, which is a
claim; the shipped code gates it to exactly zero.

Two more rules, evidence in
[`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md): **before choosing an
α rung, locate the consumer in the tier chain** — one inside `CLSPool` pools BEFORE the α/β heads
exist, so its rung is fixed by ORDERING and is not a fallback; and **a critic-facing α consumer owes
its own gradient guard** — `value_route_gradient_test.py` covers the `_value_pooled_routes` seam by
construction, but the two token-content injections sit outside it by design, so extend that test in
the same pass as any critic-side enrichment.

---

## Static typing (mypy)

The model package is **type-checked, and the gate is ZERO errors**. New code in
`src/agents/model/` must pass it before it lands:

```bash
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m mypy src/agents/model   # must be clean
```

The mypy scope, the strictness set, and the annotation rules (shape comments stay; buffers declared
under `if TYPE_CHECKING:`; every `# type: ignore` carries a code):
[`designs/model/typing.md`](../../../designs/model/typing.md).
