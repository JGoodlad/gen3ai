# File layout — one responsibility per file

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The unabridged table

Three split rounds, all **pure relocations** — same classes, same constants, same forward math
(`gen3_damage_op_split_v1` 2026-08-01 carved out the op; the 2026-08-16 round carved the phase
modules out of the extractor and the layout out of the op; `gen3_extractor_class_split_v1`
2026-08-23 carved the orchestrator CLASS itself into the base-class chain below). The critic-route
deletion wave then
REMOVED two files rather than reshuffling any — `value_routes.py` (`ValueClockRoute` /
`ValueIntentRoute`) and `intent_value_reduce.py` — plus `seed_diagnostics.py`. **The four
surviving critic-side files stay as they are**: two distinct delivery MECHANISMS (the v89 seam
in `value_readouts.py`; the `CLSPool` token-content injection in `value_threat_inject.py`) plus the
two producers (`pair_outcome.py`, `conditional_threat.py`).
Merging any of them would put two mechanisms behind one filename, which is the property this
table exists to prevent:

| file | holds |
|---|---|
| `arch_constants.py` | the architecture constants — the single source of truth for weight-shape dims |
| `extractor_ctx.py` | `ExtractorContext`, `ObsUnpack`, `Embeddings`, token-type ids, obs helpers |
| `encoders.py` | `MoveLatentEncoder`, `PokemonEncoder` |
| `team_transformer.py` | `EdgeBias` (+ the `_EDGE_*_CELL` definitions), `BiasedEncoderLayer`, `TeamTransformer`, `EventSeats` |
| `pools.py` | `CLSPool`, `HiddenOppBeliefPool`, `PolicyStateQuery` + `POLICY_READOUT_MODES` (`--policy-readout trunk`, audit F2: the policy's state query over the refined trunk; private-seed `IsolatedLinear`s). Gate: `policy_readout_test.py` |
| `belief_heads.py` | `BeliefSlots`, `BeliefHead`, `MoveBelief`, `SpreadBelief`, `ItemBelief`, `HPTypeBelief`, `BELIEF_GRAD_MODES` |
| `q_winprob_head.py` | `QWinProbHead` — the per-action `P(win\|s,a)` shared scorer over the pointer head's own action tokens. The extractor-built per-action Q head (and `Q_WINPROB_MODES`) was deleted with the cf training half (deletion pass L4); the class survives ONLY as the scorer the detached ride-along A head (`ridealong_heads.AdvantageEnsemble`) is built from |
| `aux_value_heads.py` | `WinProbHead` only — the readout off `value_pooled` that is the critic under the win-prob critic (the only critic). (`ValueDistHead` went with the distributional value head, deletion pass L1; `CfEvidentialHead` and `ShadowValueHead` went with the cf training half, L4; the twin win-prob heads B/C reused `WinProbHead`.) |
| `pointer_head.py` | `EntityMoveSeats`, `PointerNativeActionHead`, request-slot alignment |
| `value_readouts.py` | `UnifiedValueReadout` (the critic's entity pool — the ONE `_value_pooled_routes` member) |
| `value_threat_inject.py` | `ValueThreatInject` — the v64 damage-summary row as TOKEN CONTENT on the value pool's local copy of our tokens, inside `CLSPool`. Not in the v89 seam by design (a post-pool route must collapse the J axis) |
| `damage_tables.py` | the DAMAGE/type/stat lookup buffers the op's physics reads — the type chart + ability multipliers, `build_damage_buffers`, the status-landing / trap / self-boost / recovery / sleep tables, the move-attribute + Choice-Band tables, and `DAMAGE_MODELS` — the declared non-formula damage model of every move (with its throwing guard). **Also the re-export HUB for `belief_tables` and `dex_ids`** |
| `damage_kinds.py` | the ONE place the kernels turn `DAMAGE_MODELS`' tables into damage — the attacker-HP effective BP (Flail / Eruption) the non-formula rolls (fixed / fraction-of-HP / Endeavor) and Beat Up's exact party terms (`beatup_*`: the side's Σ base Atk / healthy count, the target's base Def, the typeless type), `gen3_nonformula_damage_v1` · `gen3_beatup_exact_v1` |
| `status_rules.py` | the ONE incoming side / clause status rule (`incoming_status_mask`: our Safeguard, incoming Sleep Clause, Freeze Clause, our active's Substitute), shared by the op (`_incoming_status_mask`) and the move-resolution family, each rule with its Showdown file:line, `gen3_op_status_rules_v1` |
| `belief_tables.py` | the BELIEF-PRIOR bases a belief HEAD fuses with — the opponent spread prior, its generative nature/EV decomposition (`build_nature_mult` / `build_species_nature_prior` / `build_species_ev_prior` / `build_species_base_stats`, and the nature/EV LABEL `true_nature_ev_label` + its `SpreadLabelError` guard), the Hidden-Power TYPE prior, the ITEM prior, the per-species MOVE prior (`build_move_prior_logits` + `sanitize_historical_move_floor` + the `_PRIOR_FLOOR` / `_ILLEGAL_PROB` / `_MIN_PRIOR_FLOOR` triple) and the team-composition SPECIES prior (`build_species_cooccur_prior`, `SPECIES_CLAUSE_LOGIT`). See the note below the table |
| `dex_ids.py` | the dex-IDENTITY facts BOTH of those key on — `HIDDEN_POWER_NUM` + `_belief_num` + `_hp_typed_nums` (the Hidden-Power num identity) and `build_species_usage_prior` (the normalized gen3ou usage share per num). The BOTTOM layer: it imports neither of the two above |
| `hypothesis_dex_rows.py` (+ the committed `hypothesis_dex_rows.json`) | X5's DEX-ROW TABLE (`gen3_x5_dex_rows_v1`, U1): the 122-dim per-mon row THE observation encoder (the Rust `encoder::hypothesis::hypothesis_slot`) writes for a HYPOTHESISED opponent mon of each base-form species — the generator (`python -m agents.model.hypothesis_dex_rows --write`, through `core_events --dex-rows`; it computes no cell itself) and the loader (`load_hypothesis_dex_rows(max_species)` → a dense, num-indexed, read-only `[max_species, 122]` float32 array + `valid` mask + the declared cell classes). Not under `data/`, so a pin isolates it. Gates: `hypothesis_dex_rows_test.py` (shape vs the layout), `hypothesis_dex_rows_sim_test.py` (`sim`: the byte gate + the real-state cross-check). Consumer: `hypothesis_set.py` (U2) |
| `hypothesis_set.py` | X5's T0 HYPOTHESIS BUILDER (`gen3_x5_hypothesis_set_v1`, U2; built only under `--belief-tokens fixed_mass`): `fixed_size_tau` / `fixed_mass_presence` (the logistic fixed-size marginals, a 64-step bisection on a provable bracket, k = 0 / k = n structural), `stable_order` (THE selection order, ties to the lower num), `species_candidates` / `move_candidates` (V, structural), `HypothesisBuilder` (δ_θ, OTHER, the active's move group; private-seed `IsolatedLinear`s), the `HypothesisSet` / `MovePresence` stash types, `near_tie_rows` (§3.1's rule-8 exclusion), and the set-BCE helpers the `hidden_team_set` belief row calls (and `hypothesis_moves_targets`, which MoveBelief's unrevealed population uses under fixed_mass). Gate: `hypothesis_set_test.py` |
| `hypothesis_encode.py` | X5's hypothesis-token ENCODING (`gen3_x5_hyp_gather_v1`; fixed_mass only): `species_table` (the species half of `PokemonEncoder`'s two first Linears over the 400-row dex table, once per forward) + `gathered_hypothesis_tokens` (gather by hypothesis species + the row-level half — clock, weather, fainted, hazards, screens, the active-context scatter — then the rest of the encoder on the 6 opponent slots). The exact split of the per-row pass (`PokemonEncoder` on `hypothesis_ctx`, which stays the definition), up to fp32 reassociation. Gate: `hypothesis_encode_test.py` (the gathered path taken once and the encoder once per forward, blob never; fp64 ≤ 1e-12 / fp32 ≤ 2e-5 against the per-row pass; every slot incl. the active; the gradients) |
| `static_tokens.py` | `--token-encoding static`'s per-mon encoder (`gen3_static_tokens_v1`; `designs/endstate/design_static_tokens.md`): `StaticTokenEncoder` (S, the static identity from the set fields only, the moves summed as a set; D, the mon's own battle state; added; no board fact; a drop-in at `pokemon_encoder` with `PokemonEncoder`'s submodule names), `parts()` (S, D and the per-move tokens, for the tests), `move_legality` (our active's legality onto its sorted slots by identity), `encode_dex_rows` / `static_hypothesis_tokens` (X5's hypothesis tokens: the dex table encoded once and gathered, exact), `TOKEN_ENCODING_MODES`. Its opponent stat prior is `belief_tables.build_static_stat_prior`. Gate: `static_tokens_test.py` |
| `board_tokens.py` | `--token-encoding static`'s BOARD (`gen3_static_board_v1`, stage 2; `designs/endstate/design_static_tokens.md` §4): `side_features` / `field_features` (the side-relative SIDE content, both rows the same columns `SIDE_FACTS`, and the FIELD content `FIELD_FACTS`, read off the current observation + the real context's per-mon slots), `board_offsets`, the seat tuples `BOARD_SEATS_LEGACY` / `BOARD_SEATS_STATIC`, and `OpContent` (the zero-init per-mon op content: the `x` ⊕ `g` amounts on every mon, our `d1` cells on their mons). The three board TOKENS are built by `TeamTransformer.board_tokens` (`side_proj`, `field_proj`). Gate: `static_board_tokens_test.py` |
| `hypothesis_tokens.py` | X5's hypothesis TOKENS in the phase chain (`gen3_x5_belief_tokens_v1`, U3; fixed_mass only): `hypothesis_ctx` (hidden slots' rows → their dex rows, ids re-sliced by THE slicer, every mask the REAL one), `splice_hypothesis_tokens` (+ `hypothesis_marker`), `key_log_presence` (the trunk's per-key log π), `float_key_mask` + `OppPresence` (the class-E pools' input), `FixedMassMoves` / `fixed_mass_moves` (the opponent active's move axis: one order, the fixed-mass candidate weights, the extended seat axis ⊕ `mix` that prices a revealed Hidden Power as its typed mixture) + `other_move_cells` (the active's E5 seat as OTHER_move) + `other_u` (U4: OTHER_move's renormalised-tail pricing weights). Gates: `hypothesis_tokens_test.py` (I1 / I2 per class-E site, the M10 gradient path, the hypothesis-seat move supervision), `hypothesis_moves_test.py` (one order, the HP mixture, class M on moves, OTHER_move, the K < 4 refusal) |
| `flat_intent.py` | X5's FLAT opponent pointer (`gen3_x5_flat_pointer_v1`, U4; fixed_mass only — it replaces α / β there): `FlatIntentHead` (one shared scorer + the detached log-π bias over K seats · OTHER_move · six switch targets · OTHER_species), `flat_candidates` + `FlatIntentInputs` (the candidate tokens and the detached label-side stash), `flat_intent_targets` (the labels: a belief miss is an OTHER label), `compat_intent_logits` + `FlatConsumerOps` / `append_other` (the α consumers' re-expression, OTHER priced), `render_flat`. Gate: `flat_intent_test.py` |
| `damage_op_layout.py` | every `_DMG_*` offset/width constant, `OpTensors`, `decode_damage_block` — the block's shape contract |
| `damage_op.py` | `DamageOperator` (ctor, core roll math, pointer surface, forward) + `OpStashes` |
| `damage_op_pairwise.py` | `DamageOperatorPairwise` MIXIN — the 17 `pairwise_*` edge-family cell producers |
| `selection_sites.py` | K9(b)'s DECLARED inventory of every discrete op in the forward (topk / argmax / comparisons / float→int casts): MARGIN rule or EXACT reason, keyed by source; the AST scan behind it (`gen3_behaviour_tie_exclusion_v1`) |
| `damage_op_blocks.py` | `DamageOperatorBlocks` MIXIN — the outgoing/incoming/status flat-block builders (incl. the OAX kernel = d2's engine) |
| `damage_op_speed.py` | `DamageOperatorSpeed` MIXIN — `--speed-physics on` only (`gen3_speed_physics_v1`, v143, audit F7b): every op site's P(we act first) inputs — our six EXACT speeds, their six believed speeds (mean, spread), Quick Claw (format-gated) — and the one call into `move_order.p_first_same_priority` |
| `move_order.py` | THE gen-3 move-ORDER rule, torch-free: the priority bracket `p_seat_first` (the move-resolution family's and `intent_conditional`'s) and the within-bracket speed physics (Showdown's exact stat / stage / paralysis arithmetic, the belief's lattice integral, the coin-flip tie, Quick Claw and its gen3ou ban `quick_claw_live`), each with its Showdown source. Gates: `move_order_test.py`, `speed_physics_extractor_test.py`, `speed_physics_bridge_integration_test.py` (`sim` + `integration`: the Lane S battles through the Rust core, who acted first vs P(first); the priority bracket never contradicted, the saturated bins hold) |
| `switch_branch.py` | `gen3_switch_branch_v1` — OA2 (E[our move \| they SWITCH], β-contracted, kept DECORRELATED from the stay branch), the Rapid-Spin spinblock (the Pursuit mirror) and Protect's α-derived attack mass. `SWITCH_BRANCH_COORDS` is the contract; each coordinate's §9a admission answer is in the module docstring |
| `conditional_threat.py` | `gen3_conditional_threat_v1` — **OA1**, the conditional THREAT cell (the defensive pivot): the four α-contracted coordinates the reduced outcome row structurally cannot carry (`e_pko_acc`, `e_type_mult`, `margin_high`, `margin_crit`), on the pointer SWITCH cell. `CONDITIONAL_THREAT_COORDS` is the contract; the module docstring holds the **substitution table** for the three §1.2 clauses that are superseded (no `λ`, no re-emitted row coordinates, `--damage-matrices-outgoing-all` void) plus each coordinate's §9a admission answer |
| `move_resolution.py` | `gen3_move_resolution_v1` (v141, `--move-resolution on`; audit F11 §9) — the MOVE-RESOLUTION family: `MoveResolutionOps` (every operand, as tensors), the pure `move_facts` / `switch_facts` (P(each action resolves as stated) by the exact gen-3 rules, intent-weighted, plus the seven blocks' FACTS; the incoming-status corrections `incoming_status_correction`), `MoveResolutionCell` (two zero-init `IsolatedLinear`s) and `gather_ops` (reads one forward at the pointer stash). Gates: `move_resolution_test.py` (each rule on a constructed state; 46 mutations all caught), `move_resolution_extractor_test.py` (off builds nothing; the real-policy retirement; every other parameter's init bytes = production's; the edited-board gathers), `move_resolution_bridge_integration_test.py` (`sim` + `integration`): the banked M5 Lane S battles replayed through the Rust core, the family run on every played move's observation, the move resolved against that turn's protocol; a stated exact 0 is never contradicted (full bank: 31,942 played moves, 1,515 exact-zero claims, 0 contradicted) |
| `move_resolution_rules.py` | the family's gen-3 RULES, torch-free: the source-verified move-flag sets (`protect`, `bypasssub`, `sound`, `reflectable`, `defrost`, `failencore`, the two status moves that check type immunity), the kinds, the per-turn probabilities, the coordinate contracts — each with its Showdown file:line. (The move-ORDER rule `p_seat_first` moved to `move_order.py`, v142.) `data/` carries no move flags, so `move_resolution_rules_integration_test.py` re-derives every set from the gen-3 dex through node |
| `move_resolution_tables.py` | the family's construction-time TABLES (never the forward): per-move kinds / flags, per-species and per-ability immunity / type / named-ability marginals from the Smogon priors |
| `pair_outcome.py` | the UNIFIED per-pair OUTCOME VECTOR's contract — `PAIR_OUTCOME_COORDS` (the coordinate table, with each one's §9a admission answer), `pair_alpha` (the publication read + the R1 fallback), `reduce_pair_in` (Contract W's one line), `PairOutcomeMoveCell`, plus Phase B's `reduce_pair_in_all` (Contract W at EVERY defender), `pair_alpha_full` (the three-way α split a SWITCH-branch consumer needs) and `PairOutcomeSwitchCell` (the FIRST module to widen the pointer SWITCH cell). Its op-side producer is `DamageOperatorBlocks.pair_outcome_coords` |
| `extractor_stashes.py` | `ExtractorStashes` — the per-forward side-value container (`gen3_extractor_stashes_v1`) |
| `projection.py` | `compute_projection_widths` (the STATIC width arithmetic) + `ProjectionAssembler` (the concat it describes) |
| `extractor_build.py` | `ExtractorBuild` — `Gen3FeaturesExtractor.__init__`: every flag validation, every module, in construction ORDER |
| `extractor_api.py` | `ExtractorApi` — the `last_*` stash reads, the pointer-cell widths, the debugger/ortho-init/belief-grad-mode setters |
| `extractor_forward.py` | `ExtractorForward` — `forward_internal`, the T0/T1 belief+physics stack, `_value_pooled_routes` |
| `features_extractor.py` | the `Gen3FeaturesExtractor` class + `forward`; **the re-export HUB for every moved name** |
| `compile_opponents.py` | `maybe_compile_extractor(model, enabled, label, hide_cuda)` — the CPU compile of ONE frozen extractor, used by the offline readers (split out of `snapshot.py`; the trainer-side opponent compile, its strict mode and revert quorum were deleted in U3) |
| `compile_config.py` | the compile-config ROW `compile_control` pins per torch version (`COMPILE_CONFIG`: `donated_buffer=False` + the cache-key tag) and its hash, the hermetic cache stamp's third field — DATA only, no check at import (P10-D) |
| `compile_control.py` | `gen3_compile_sentinel_v1` — the ONLY runtime module that touches `torch._dynamo`: the gate → reset → prewarm → lock → stats phases, the cache-limit detector, the torch version table and the source-hash drift tripwire, run by `require_supported_torch` where a learner compiles — never at import (`designs/training/compile_flags.md`) |

## The table LAYERING — `damage_tables` → `belief_tables` → `dex_ids`

**`belief_tables.py` + `dex_ids.py` are a fourth split round, taken in two passes
(`gen3_belief_tables_split_v1` then `gen3_dex_ids_split_v1`, both 2026-09-06), and the LAYERING is
the part to remember.** `damage_tables.py` had reached 1,433 lines holding unrelated subjects. Pass
one moved the BELIEF PRIORS a head fuses with — the per-species Smogon distributions it predicts a
zero-init DELTA on top of (spread, nature+EV, Hidden-Power type, item) — into `belief_tables`. Pass
two moved the remaining two priors of that kind (the MOVE prior and the team-composition SPECIES
prior, ~350 lines) and took `damage_tables` from 1,226 to **930**, under the size gate's 1,000-line
TARGET. That second pass needed a THIRD module: those two builders read `HIDDEN_POWER_NUM`,
`_belief_num`, `_hp_typed_nums` and `build_species_usage_prior`, which **the op's physics also
reads** — so putting them in a belief module would have parked dex-identity facts the physics
depends on inside the beliefs. `dex_ids.py` is the neutral bottom layer that holds them instead.

The order is:

```
damage_tables  →  belief_tables  →  dex_ids          (and damage_tables → dex_ids)
```

**and it only ever points down.** Each higher layer is a real CONSUMER, which is what fixes the
direction rather than leaving it a convention: `build_damage_buffers` registers
`SPECIES_SPREAD_PRIOR`, `NATURE_MULT` and `SPECIES_USAGE_PRIOR` for the op and its typed-HP
expansion reads `HIDDEN_POWER_NUM` + `_hp_typed_nums`; `build_move_prior_logits` calls `_belief_num`
and `_hp_typed_nums`. An import back would close a cycle Python resolves **only for whichever module
was imported first** — so it would work in the normal import order and raise in every other. Do not
add one. `belief_tables_test.py` AST-scans every up-edge from a single declared `_LAYERS` tuple (so
a fourth module is one more entry, not one more test), plus a COLD-IMPORT test that runs a fresh
interpreter importing only `dex_ids` and asserts the two above it stay out of `sys.modules`. It also
fails if any name is DEFINED in two of the three — a floor re-declared rather than imported is the
same "a fix lands in one copy" hazard, and is exactly the shortcut a later split round is tempted by.

`damage_tables` **re-exports every moved name** (`# noqa: F401  (re-export)` inline, never a new
`ruff.toml` entry), so the ~20 historical `from agents.model.damage_tables import …` spellings in
`belief_heads`, `t0_species`, `extractor_build`, `snapshot`, the (deleted) Python env, `main.train.config`,
`flag_registry`, the prober and nine test modules still resolve — and the test asserts they resolve
to the SAME object as the owning module.

**No `state_dict` key moved and `ARCH_SIGNATURE` is untouched**: every one of these tables is
registered `persistent=False` by its owning head (derived from `data/`, recomputable, never a saved
weight), so they contribute zero keys and a relocation cannot move a key that does not exist.
Verified rather than asserted, both rounds — on a seeded production-config build all **236
`state_dict` entries and all 80 buffers are byte-identical** across the cut, and every moved
definition is executable-AST identical to its pre-cut self (docstrings stripped; each gains one
origin line). The probe is re-runnable against any baseline:
`designs/research_state/measurements/dex_ids_split_2026-09-06/equivalence_probe.py`.

The per-table SEMANTICS stay beside the head each prior feeds (`spread_belief_test.py` /
`hp_type_belief_test.py` / `item_belief_test.py` / `move_prior_fusion_test.py` /
`species_prior_fusion_test.py` / `damage_tables_test.py` — all of which reach their subject through
the `damage_tables` hub and so were untouched by either move); `belief_tables_test.py` holds only
what the SPLIT can break.

## The extractor CLASS chain

A third split round, and the one that took the last entry off the size ratchet's grandfathered list
(the tree now has **no** source file over 2,000 lines). `features_extractor.py` went 2,280 → **277**:

    ExtractorBuild(torch.nn.Module)   extractor_build.py    __init__
      └─ ExtractorApi                 extractor_api.py      the last_* surface + the setters
           └─ ExtractorForward        extractor_forward.py  forward_internal + the T0/T1 stack
                └─ Gen3FeaturesExtractor  features_extractor.py  the class, and `forward`

**Inheritance rather than helper functions, for four reasons that are each a constraint, not a
preference:**

1. **`state_dict` keys.** Moving CODE is free; moving where a sub-MODULE is ATTACHED is not. A base
   class changes no attribute PATH on the instance, so the 236 keys are byte-identical.
2. **The constructor SIGNATURE is a public surface.** SB3 builds the extractor as
   `features_extractor_class(observation_space, **features_extractor_kwargs)`, and ~10 sites read
   `inspect.signature(Gen3FeaturesExtractor.__init__).parameters` as the flag set (`ctor_kwarg_snapshot_test`, `config_only_pattern_test`, `delivery_graph`, …).
   An inherited `__init__` IS that function, so every one of them is unchanged.

> 🚨 **A second public surface of the forward: the obs DICT's key set.** `forward` is normally a
> pure function of `obs["observation"]`, but a route may read a flag-gated Dict key of its own
> (the deleted privileged true-team route's `opp_true_team` was the first; the registry is now empty) and RAISE when it is absent. That mapping is
> DECLARED in **`extra_obs_keys.py`** — `(extractor attribute -> key, shape, canonical zero block)`
> — and every synthetic-obs caller on a training path builds from it (`lifecycle._run_roundtrip_test`, `compile_trainer`, `compile_opponents`, `warmstart`). The enable
> condition is the ATTRIBUTE the forward itself tests, so the table cannot drift from the seam, and
> an AST gate over `extractor_forward` fails on an undeclared key. Hand-building
> `{"observation": zeros(1, D)}` is what killed `ai_v12_14_ladder_truevalue` two minutes into its
> launch; `extra_obs_keys_test.py` reproduces it.
3. **Every body keeps its `self.` spelling**, so the split is checkable as a pure relocation — all
   44 members are source-hash identical to the pre-split class.
4. **mypy needs no declarations.** Each mixin's `self.<attr>` resolves against the `__init__` that
   assigns it, because that `__init__` is an ANCESTOR rather than a sibling.

### `__init__` stays whole, the ctor lookup, the module-GLOBAL patch hazard, and the absent import cycle

**⚠️ `__init__` is deliberately NOT split further** — same reasoning as `instrumented_ppo.train()`.
Its checkable property is MODULE CONSTRUCTION ORDER: SB3 restores optimizer state POSITIONALLY (the
ai_v6_13 "128 vs 5" crash), so a module must be APPENDED, never inserted, and several comments in the
body say exactly where in that order they sit. That is only readable while it is one straight line.

**⚠️ A test that patches a module GLOBAL must name the module that HOLDS the caller.** Python
resolves a global at call time against the *defining* module's namespace, so
`agents.model.features_extractor.threshold_probs = fake` stopped reaching `forward_internal` the
moment the forward moved. `intent_threshold_test` caught it because its pin is `assert not equal`, so
a patch landing nowhere reads as "the cell is dead" — the same drift under an `assert equal` pin
would have passed for the wrong reason forever. It now resolves the module through
`inspect.getmodule(type(fe).forward_internal)`, which follows the code.

**⚠️ `flag_requires_test._guarded_raises` resolves the ctor from the FUNCTION**
(`Gen3FeaturesExtractor.__init__` + its `__qualname__`), never from
`inspect.getsourcefile(Gen3FeaturesExtractor)` — the class's file no longer holds an `__init__` at
all.

No import cycle: `DamageOperator` touches the extractor only through `ctx: 'ExtractorContext'`, which
is a **string** forward-reference and so costs no runtime import. The re-exports mean every historical
path (`from agents.model.features_extractor import DamageOperator / EdgeBias / decode_damage_block /
_DMG_* / _SB_ATK`, `from agents.model.snapshot import maybe_compile_extractor`) still resolves — the
prober, `model_version`, `snapshot` and the tests all rely on that. `Gen3FeaturesExtractor` itself
stays DEFINED in `features_extractor.py`: SB3 checkpoints pickle the class by its defining module.
