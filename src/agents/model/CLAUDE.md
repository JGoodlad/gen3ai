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
| the α/β heads, their metrics, or a new α consumer | [`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md) |
| PopArt | [`designs/model/popart.md`](../../../designs/model/popart.md) |
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
`(pi_combined, vf_combined)` pair; and the root `forward` returns a `(pi_features, vf_features)`
tuple. This extractor therefore **must** be paired with `Gen3DualHeadMaskablePolicy`
(`policy.py`), which keeps `share_features_extractor=True` (one body) and overrides `forward` /
`evaluate_actions` / `get_distribution` / `predict_values` to unpack the tuple and route each half
to `mlp_extractor.forward_actor` / `forward_critic`. A stock SB3 policy expects a single-tensor
extractor and will break — doubly so under the pointer-native action head (`gen3_pointer_native_v1`): the policy's `_build`
deletes the flat `action_net` and the action logits come from the `PointerNativeActionHead` over
the extractor's `last_pointer_inputs` stash (per-logit inputs: `designs/ARCHITECTURE.md` § Heads). The startup `_run_roundtrip_test` and the snapshot/feature tests all
unpack the tuple — keep that in mind when touching the extractor's return shape.

**The one channel the policy path cannot see** (`--value-true-team`, v114,
`gen3_value_true_team_v1`). Because `vf_combined` IS `value_pooled` and `pi_combined` is a concat
that never contains it, anything injected into `value_pooled` is vf-only at ANY weight — which is
what makes the PRIVILEGED true-opponent-team route (the critic ladder's arm-5 ceiling probe) safe
to build at all. It reads the opponent's real party off a training-and-eval-only obs key
`opp_true_team` and is the only route in the tree that adds INFORMATION rather than re-reading the
shared observation. Detail — including why it AUGMENTS rather than replaces the belief-keyed opp
view, why it RAISES on a missing key, and why the prober's offline forwards REFUSE on such a
checkpoint — is in [`designs/model/readouts_and_value_routes.md`](../../../designs/model/readouts_and_value_routes.md).

**The SEVENTH readout off `value_pooled`** (`--win-prob-dense-aux`, v117, `gen3_dense_aux_v1`) — the
DENSE AUXILIARY head, the critic ladder's arm 9. `Linear(D_MODEL, 64) → ReLU → Linear(64, 25)`,
zero-init output, built LAST and **not called by the forward at all** (the `CfEvidentialHead`
contract), predicting the episode's END-OF-BATTLE facts: survival and final HP of all twelve slots,
plus the scaled turns-left. It breaks the cf readouts' pattern in exactly one place — its input is
**NOT detached** in the training term, because the whole point is gradient into the shared trunk
along the per-entity axes one terminal bit cannot carry. `pi`/`vf` are still bit-identical at any
weight in it (the forward never calls it), and `grad/dense_aux_share` is the read that separates
"the arm ran" from "the arm did what it was built to do". Targets, masks and the λ precedence:
[`designs/training/critic_and_value_losses.md`](../../../designs/training/critic_and_value_losses.md).

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
| the op | `damage_op.py` · `damage_op_layout.py` · `damage_op_pairwise.py` · `damage_op_blocks.py` |
| the lookup tables, in LAYER order | `damage_tables.py` → `belief_tables.py` → `dex_ids.py` |
| the readouts and the critic routes | `aux_value_heads.py` · `q_winprob_head.py` · `value_readouts.py` · `value_threat_inject.py` · `pair_value_route.py` |
| the pointer head and the per-action cells | `pointer_head.py` · `pair_outcome.py` · `switch_branch.py` · `conditional_threat.py` |
| versioning, snapshots, the compile path, the critic modes | `model_version/` · `snapshot.py` · `compile_opponents.py` · `critic_mode.py` · `popart.py` |
| the DICT obs keys the forward reads beyond `observation` | `extra_obs_keys.py` |

🚨 **THE FORWARD HAS TWO PUBLIC SURFACES: the constructor signature, and the obs DICT's KEY SET.**
`forward` is normally a pure function of `obs["observation"]` — but a route may read a flag-gated
Dict key of its own and **RAISE** when it is absent (a silent skip reads exactly like a route that
learned nothing). `--value-true-team`'s `opp_true_team` is the first. That mapping is DECLARED once
in `extra_obs_keys.py` as `(extractor attribute -> key, shape, canonical zero block)`, keyed on the
ATTRIBUTE the forward itself tests, and every synthetic-obs caller on a TRAINING path builds from it
via `zero_extra_obs` / `synthetic_obs`: `compile_preload`, `main/train/lifecycle.py`'s round-trip
smoke, `compile_trainer`, `compile_opponents`, `agents/training/warmstart.py`. **Never hand-build
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
`fe.forward`; `cf_terms` calls `type(fe).forward` for a deliberately-EAGER pass; and
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

**The op's SIDE VALUES and the EXTRACTOR's follow one contract** (`gen3_op_stashes_v1` /
`gen3_extractor_stashes_v1`): every per-forward stash lives in ONE dataclass the forward replaces at
ENTRY, **reads** go through the `last_*` properties and **writes** through `op.stash.<field>` /
`fe.stash.<field>` — writing a `last_*` name raises. Add the dataclass field with its shape comment,
never a bare `self.last_x = …`; each producer owns its own stash surface and a submodule never writes
into its parent's. Gate: `extractor_stashes_test.py`. Detail:
[`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md).

## ⚠️ One op's SPELLING is load-bearing for `torch.compile` (`gen3_species_posterior_spelling_v1`)

`BeliefHead.species_posterior` computes `P(species)` for the expected-latent defender. It is written
as **`log_softmax(...).exp()`, not `torch.softmax(...)`, and that is deliberate** — do not
"simplify" it.
`extractor_compiles_test.py` owns the compile matrix and pins the spelling (a real compile of the
production arch with suppression OFF; `GEN3AI_SKIP_COMPILE_TESTS=1` opts out, `GEN3AI_TEST_ALLOW_GPU=1`
for the CUDA cells). The Inductor diagnosis:
[`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md).

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

## Model versioning (`model_version/`, `snapshot.py`)

**`model_version` is a PACKAGE**; `__init__.py` is a pure re-export hub. What each module holds, what
a save writes, the two sanitizers and the `--critic` version gate:
[`designs/model/versioning.md`](../../../designs/model/versioning.md). The playbooks are here.

**When you change an architecture constant:**
- `check_compatible()` catches the mismatch automatically — no extra steps needed
- Old models can't be loaded, which is correct (rapid iteration project)

**When you add an optional new feature** (new field with a sensible default):
1. Add the field to `ModelVersionFields` in `model_version/fields.py`
2. Bump `MODEL_CONFIG_VERSION`
3. Add one `if version < N:` block in `_migrate_config()` with `data.setdefault(...)`

**When you make a structural change** (different forward pass, new layer type):
1. Change `ARCH_SIGNATURE` in `model_version/constants.py` (e.g. `"gen3_attn_v1"` → `"gen3_lstm_v1"`)
2. Old models get a clear arch-family error on load

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
gates EVERY load including the frozen eval / pool / distill opponents whose forward is identical
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

## The CRITIC MODE (`critic_mode.py`, `--critic {shaped,winprob}`, v109)

`gen3_winprob_critic_mode_v1`. `Gen3DualHeadMaskablePolicy._critic_value` chooses between two
readouts, and `agents/model/critic_mode.py` is the ONE declaration of the legal set. That module is
deliberately **torch-free and import-light**: `main.checkargs` promises not to import torch and
needs the legal set to validate an argv offline.

| `--critic` | `_critic_value` returns | `value_net` | PopArt |
|---|---|---|---|
| **`shaped`** (default) | `_denorm(value_net(latent_vf))`, or `_denorm(head.mean(logits))` under `value_from_dist` | trained | allowed |
| `winprob` | `sigmoid(fe.last_win_prob_logits)` in **[0,1]**, `[B,1]`, no `_denorm` | in NO loss graph | **refused at the constructor** |

**Read the mode through `is_winprob`, never a bare `== "winprob"`** — one spelling, one answer, and
a `getattr(obj, "critic", "shaped")` read answers correctly through it.

🚨 **The version gate matters more here than for a typical structural flag: BOTH routes return a
`[B,1]` float tensor**, so a flipped `critic` produces no shape error, no load failure and no metric
that changes name — the run simply predicts a different quantity for the rest of its life. The string
compare in `check_compatible` is the only thing standing between a resume and that. **No
`ARCH_SIGNATURE` bump at v109** — `shaped` is the default, so an untyped flag adds and removes
nothing; the bump belongs to the DEFAULT FLIP, where it is forced.
[`designs/model/versioning.md`](../../../designs/model/versioning.md) has both in full, and how the
kwarg is threaded.

## PopArt value-target normalization (`popart.py`, `--use-popart`)

Opt-in (default off, **refused outright under `--critic winprob`** — a bounded stationary Bernoulli
payoff has no scale to track). It tracks running `(mu, sigma)` of the value targets so the value
gradient on the SHARED trunk stays O(1), and the **POP** half rescales `value_net` on every stats
update so the de-normalized prediction is unchanged. `--use-popart` **requires an explicit
`--clip-range-vf none`**, and `use_popart` is version-checked — it cannot be flipped mid-run. Detail:
[`designs/model/popart.md`](../../../designs/model/popart.md).

## Opponent intent — `α` / `β` (`opp_intent.py`, v67)

The build for one sentence the model could not express: *"they are likely to click **this**, so
**this** is my answer."* `--opp-intent-coef>0` adds two SUPERVISED pointer heads:

- **`α`** — a pointer over the opponent's K believed threat-move seats **plus a SWITCH option**.
- **`β`** — given a switch, which of their mons comes in; masked to alive-and-non-active, because an
  illegal switch-in must be UNREPRESENTABLE rather than merely unlikely.

Why pointers and not a flat `Linear(ctx, K)`, matching by canonical id, and why the label is shifted
back one row before `get()` shuffles:
[`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md).

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

🚨 **The bare key is a MIX, and the mix MOVES** — supervised rows ran 100% bot at 2M and ~7% from 6M
on, so a pooled metric rises as the mix shifts and that rise is indistinguishable from the head
improving. Any trend spanning the ramp is uninterpretable. `alpha_mask_rate` stays whole-batch on
purpose: it is the BELIEF's coverage failure. Full metric inventory and the head→human path:
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

Nine modules now contract α against the op's physics (listed in
[`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md)). They share four
conventions, and each exists because breaking it fails silently:

1. **T1 produces, T2 consumes.** α is scored from the E4 seats and the CLS pools, both DOWNSTREAM
   of the op — so the op cannot reduce by α, and every consumer runs at the pointer stash. A
   "swap `_chan_max`'s `how=`" plan is unbuildable for that reason, not for want of a knob.
2. **Read `last_alpha_logits` (the PUBLICATION), never a raw stash** — and take the
   **UNRENORMALIZED move slice**. The missing SWITCH mass is the literally-correct statement that a
   switching opponent applies no outcome this turn; renormalizing asserts they attacked.
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
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m mypy src/agents/model   # must be clean
```

The mypy scope, the strictness set, and the annotation rules (shape comments stay; buffers declared
under `if TYPE_CHECKING:`; every `# type: ignore` carries a code):
[`designs/model/typing.md`](../../../designs/model/typing.md).
