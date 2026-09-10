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
`ValueIntentRoute`) and `intent_value_reduce.py` — plus `seed_diagnostics.py`. **The five
surviving critic-side files stay as they are**: three distinct delivery MECHANISMS (the v89 seam
in `value_readouts.py`; the two `CLSPool` token-content injections in `value_threat_inject.py` and
`pair_value_route.py`) plus the two producers (`pair_outcome.py`, `conditional_threat.py`).
Merging any of them would put two mechanisms behind one filename, which is the property this
table exists to prevent:

| file | holds |
|---|---|
| `arch_constants.py` | the architecture constants — the single source of truth for weight-shape dims |
| `extractor_ctx.py` | `ExtractorContext`, `ObsUnpack`, `Embeddings`, token-type ids, obs helpers |
| `encoders.py` | `MoveLatentEncoder`, `PokemonEncoder` |
| `team_transformer.py` | `EdgeBias` (+ the `_EDGE_*_CELL` definitions), `BiasedEncoderLayer`, `TeamTransformer`, `EventSeats` |
| `pools.py` | `CLSPool`, `HiddenOppBeliefPool` |
| `belief_heads.py` | `BeliefSlots`, `BeliefHead`, `MoveBelief`, `SpreadBelief`, `ItemBelief`, `HPTypeBelief`, `BELIEF_GRAD_MODES` |
| `q_winprob_head.py` | `QWinProbHead` (v107) + `Q_WINPROB_MODES` — the PER-ACTION `P(win\|s,a)` readout over the pointer head's own action tokens. Its own file rather than a fifth entry in `aux_value_heads.py`, because that file's subject is "readouts off `value_pooled`" and this one's input is the pointer stash; `value_pooled` is only its CONTEXT |
| `aux_value_heads.py` | `WinProbHead`, `ValueDistHead`, `CfEvidentialHead` (v98), `ShadowValueHead` (v99 — the passive MC-grounded value twin behind `--cf-shadow-critic`; the twin WIN-PROB heads B/C reuse `WinProbHead` unchanged, which is the point: an architecture difference would be a second explanation for a score difference) (v98 — the EVIDENTIAL Beta posterior over P(win\|state); `softplus+1` ⇒ α,β ≥ 1 so the Beta stays unimodal and `Beta(1,1)` is reachable, plus the two closed forms the loss needs: the Beta-Binomial marginal NLL and `KL(·‖Beta(1,1))`. The ONE readout here with no `read_only`/`shaping` split — its input is detached UNCONDITIONALLY and the forward never calls it) |
| `pointer_head.py` | `EntityMoveSeats`, `PointerNativeActionHead`, request-slot alignment |
| `value_readouts.py` | `UnifiedValueReadout` (the critic's entity pool — the ONE `_value_pooled_routes` member) |
| `value_threat_inject.py` | `ValueThreatInject` — the v64 damage-summary row as TOKEN CONTENT on the value pool's local copy of our tokens, inside `CLSPool`. Not in the v89 seam by design (a post-pool route must collapse the J axis) |
| `damage_tables.py` | the DAMAGE/type/stat lookup buffers the op's physics reads — the type chart + ability multipliers, `build_damage_buffers`, the status-landing / trap / self-boost / recovery / sleep tables, the move-attribute + fixed-damage + Choice-Band tables. **Also the re-export HUB for `belief_tables` and `dex_ids`** |
| `belief_tables.py` | the BELIEF-PRIOR bases a belief HEAD fuses with — the opponent spread prior, its generative nature/EV decomposition (`build_nature_mult` / `build_species_nature_prior` / `build_species_ev_prior` / `build_species_base_stats` / `invert_nature_evs`), the Hidden-Power TYPE prior, the ITEM prior, the per-species MOVE prior (`build_move_prior_logits` + `sanitize_historical_move_floor` + the `_PRIOR_FLOOR` / `_ILLEGAL_PROB` / `_MIN_PRIOR_FLOOR` triple) and the team-composition SPECIES prior (`build_species_cooccur_prior`, `SPECIES_CLAUSE_LOGIT`). See the note below the table |
| `dex_ids.py` | the dex-IDENTITY facts BOTH of those key on — `HIDDEN_POWER_NUM` + `_belief_num` + `_hp_typed_nums` (the Hidden-Power num identity) and `build_species_usage_prior` (the normalized gen3ou usage share per num). The BOTTOM layer: it imports neither of the two above |
| `damage_op_layout.py` | every `_DMG_*` offset/width constant, `OpTensors`, `decode_damage_block` — the block's shape contract |
| `damage_op.py` | `DamageOperator` (ctor, core roll math, pointer surface, forward) + `OpStashes` |
| `damage_op_pairwise.py` | `DamageOperatorPairwise` MIXIN — the 17 `pairwise_*` edge-family cell producers |
| `damage_op_blocks.py` | `DamageOperatorBlocks` MIXIN — the outgoing/incoming/status flat-block builders (incl. the OAX kernel = d2's engine) |
| `switch_branch.py` | `gen3_switch_branch_v1` — OA2 (E[our move \| they SWITCH], β-contracted, kept DECORRELATED from the stay branch), the Rapid-Spin spinblock (the Pursuit mirror) and Protect's α-derived attack mass. `SWITCH_BRANCH_COORDS` is the contract; each coordinate's §9a admission answer is in the module docstring |
| `conditional_threat.py` | `gen3_conditional_threat_v1` — **OA1**, the conditional THREAT cell (the defensive pivot): the four α-contracted coordinates the reduced outcome row structurally cannot carry (`e_pko_acc`, `e_type_mult`, `margin_high`, `margin_crit`), on the pointer SWITCH cell. `CONDITIONAL_THREAT_COORDS` is the contract; the module docstring holds the **substitution table** for the three §1.2 clauses that are superseded (no `λ`, no re-emitted row coordinates, `--damage-matrices-outgoing-all` void) plus each coordinate's §9a admission answer |
| `pair_value_route.py` | `gen3_pair_value_route_v1` — **PV**, the pair-VALUE CRITIC route: Phase A's unified row as TOKEN CONTENT on our mon j's token, injected inside `CLSPool` on the value pool's copy. The docstring carries the **C4 re-entry condition**, why the v89 seam was rejected (a post-pool route must collapse the J axis), and why α is R1 by ORDERING |
| `pair_outcome.py` | the UNIFIED per-pair OUTCOME VECTOR's contract — `PAIR_OUTCOME_COORDS` (the coordinate table, with each one's §9a admission answer), `pair_alpha` (the publication read + the R1 fallback), `reduce_pair_in` (Contract W's one line), `PairOutcomeMoveCell`, plus Phase B's `reduce_pair_in_all` (Contract W at EVERY defender), `pair_alpha_full` (the three-way α split a SWITCH-branch consumer needs) and `PairOutcomeSwitchCell` (the FIRST module to widen the pointer SWITCH cell). Its op-side producer is `DamageOperatorBlocks.pair_outcome_coords` |
| `extractor_stashes.py` | `ExtractorStashes` — the per-forward side-value container (`gen3_extractor_stashes_v1`) |
| `projection.py` | `compute_projection_widths` (the STATIC width arithmetic) + `ProjectionAssembler` (the concat it describes) |
| `extractor_build.py` | `ExtractorBuild` — `Gen3FeaturesExtractor.__init__`: every flag validation, every module, in construction ORDER |
| `extractor_api.py` | `ExtractorApi` — the `last_*` stash reads, the pointer-cell widths, the debugger/ortho-init/belief-grad-mode setters |
| `extractor_forward.py` | `ExtractorForward` — `forward_internal`, the T0/T1 belief+physics stack, `_value_pooled_routes` |
| `features_extractor.py` | the `Gen3FeaturesExtractor` class + `forward`; **the re-export HUB for every moved name** |
| `compile_opponents.py` | `maybe_compile_extractor` — the CPU-opponent compile path (split out of `snapshot.py`) |

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
`belief_heads`, `t0_species`, `extractor_build`, `snapshot`, `gen3_env`, `main.train.config`,
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
   `inspect.signature(Gen3FeaturesExtractor.__init__).parameters` as the flag set (`compile_prewarm`,
   `compile_preload`, `ctor_kwarg_snapshot_test`, `config_only_pattern_test`, `delivery_graph`, …).
   An inherited `__init__` IS that function, so every one of them is unchanged.

> 🚨 **A second public surface of the forward: the obs DICT's key set.** `forward` is normally a
> pure function of `obs["observation"]`, but a route may read a flag-gated Dict key of its own
> (`--value-true-team`'s `opp_true_team` is the first) and RAISE when it is absent. That mapping is
> DECLARED in **`extra_obs_keys.py`** — `(extractor attribute -> key, shape, canonical zero block)`
> — and every synthetic-obs caller on a training path builds from it (`compile_preload`,
> `lifecycle._run_roundtrip_test`, `compile_trainer`, `compile_opponents`, `warmstart`). The enable
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
