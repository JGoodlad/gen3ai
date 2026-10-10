# Model Directory — Contributor Notes

**This leaf is a CONTRACT, a HAZARD LIST and a MAP.** [`designs/ARCHITECTURE.md`](../../../designs/ARCHITECTURE.md)
is the only document that states the model AS IT IS NOW — read it first; where it and this file disagree,
ARCHITECTURE.md wins. History: [`designs/CHANGELOG.md`](../../../designs/CHANGELOG.md). When you touch the model,
update the rule here if a rule changed, ARCHITECTURE.md if the state did, and the topic doc below in the same pass.

## Where the detail is — the topic map

**`designs/model/` owns the detail and is always-current like this file.** Each topic doc ends with a
"Moved from the leaf (2026-10-10)" section holding this leaf's former full text on that topic.

| I am about to touch… | Read |
|---|---|
| the phase chain, the tier order, `--belief-grad-mode`, X5's hypothesis tokens, the static encoder, identity-at-init | [`designs/model/phase_pipeline.md`](../../../designs/model/phase_pipeline.md) (`ARCHITECTURE.md` §2.1 first) |
| any readout off `value_pooled`, the dual-head split, a critic route, the ride-along heads | [`designs/model/readouts_and_value_routes.md`](../../../designs/model/readouts_and_value_routes.md) |
| which file a class lives in, the architecture constants, the table layering, the extractor class chain | [`designs/model/file_layout.md`](../../../designs/model/file_layout.md) |
| the op's slicer / gain / stash contract, move order, ability-known, selection sites, the compile spellings | [`designs/model/op_contracts.md`](../../../designs/model/op_contracts.md) |
| adding, demoting or deleting a model flag; `oracle_reveal` | [`designs/model/flag_registry_rules.md`](../../../designs/model/flag_registry_rules.md) + the GENERATED [`designs/flag_registry.md`](../../../designs/flag_registry.md) |
| `model_version/`, a config bump, a deleted kwarg, the optimizer realign, a resume-immutable hparam, the recorded `critic` | [`designs/model/versioning.md`](../../../designs/model/versioning.md) |
| the opponent-intent readout (X5's flat pointer as α / β), its metrics, an α consumer | [`designs/model/opponent_intent.md`](../../../designs/model/opponent_intent.md) |
| the delivery graph, the architecture viewer | [`designs/model/architecture_artifacts.md`](../../../designs/model/architecture_artifacts.md) |
| a type annotation, the mypy scope | [`designs/model/typing.md`](../../../designs/model/typing.md) |
| a HAND-COMPUTED feature (op row/kernel, pointer cell, prior table, board-token fact, reduction, critic route, derived obs column) | update its row in [`designs/endstate/design_hand_computed_features.md`](../../../designs/endstate/design_hand_computed_features.md) in the same commit |

Closed history — do not update, do not re-derive a plan from it:
[`designs/research_state/claude_md_archive/model_leaf_history.md`](../../../designs/research_state/claude_md_archive/model_leaf_history.md).

## Architecture constants — one source of truth

Network dims are module-level constants in **`arch_constants.py`** — change them there and nowhere else
(`features_extractor.py` re-exports the block; `ModelVersion` imports it so `model_config.json` records
the live values). Embedding dims come from the observation layout (`get_layout()`) via
`features_extractor_kwargs`. `role_input_dim` and the projection widths are DERIVED
(`compute_projection_widths`, verified per flag combo by `projection_width_test.py`): **a new
width-contributing part extends `compute_projection_widths` and the sweep in the same pass.**

## Phase rules

- **The TIER ORDER is an asserted invariant** (`tier_contract.py`, `tier_contract_test.py`): every
  `nn.Module` child declares a tier or is listed in `UNTIERED_CHILDREN`.
- **No placement flag, no second chain.** 🚨 **The op's flat block enters NEITHER projection**
  (`gen3_no_concat_v1`): the trunk/concat route is dead; re-adding one is a new architecture decision.
- **Each phase owns its layers** (state_dict keys are phase-prefixed). **`Embeddings` is the sole owner of
  the embedding tables**, passed as a forward ARGUMENT, never stored as a child. **`ExtractorContext`** is
  the inter-phase contract — add a field there rather than widening a positional signature.
- **Any change to the phase structure or forward math bumps `ARCH_SIGNATURE`** (`model_version/constants.py`;
  read the live value there). A pure decomposition still bumps it (keys move); a forward-math change with
  unchanged widths is caught by NOTHING else; re-sourcing or re-meaning an obs block is retrain-class.
- Per-phase unit tests: `phase_modules_test.py` (hand-built `ExtractorContext`, no full forward).

**`--belief-grad-mode`:** `detached` and `label_only` cut OPPOSITE arrows (table: `phase_pipeline.md`).
🚨 A supervised loss reads `belief_supervision(name)`, never the `last_*` attribute (under `label_only` that
is the stop-grad publication and the loss would train NOTHING, silently). 🚨 Detach the LOGITS, never the
matmul output (`sigmoid(logits) @ emb.weight` — detaching the product kills the table's gradient).

## Dual head, critic, ride-along heads

- The extractor returns `(pi_features, value_pooled)` and **must** be paired with
  `Gen3DualHeadMaskablePolicy` (`policy.py`). There is no value tower and no flat `action_net`
  (`action_net` / `value_net` are RAISING stubs); actions come from `PointerNativeActionHead`. The startup
  `_run_roundtrip_test` (`main/train/lifecycle.py`) and the snapshot tests unpack the tuple.
- **`value_pooled` is vf-only by construction** — anything injected there leaves `pi` bit-identical; the
  `_value_pooled_routes` seam is the one place to add such a route.
- **The critic is the win-prob critic, the only one** (`_critic_value` = `sigmoid(fe.last_win_prob_logits)`).
  `critic` survives only as a RECORDED field; `critic_mode.py` is torch-free (for `main.checkargs`).
  **Read it through `is_winprob`, never `== "winprob"`.** A flipped `critic` makes no shape error, so the
  string compare in `check_compatible` is the only gate. PopArt, the value-dist head and `value_from_dist`
  are DELETED (`designs/deleted_flags.md`; `model_version/retired_levers.py` refuses a run that recorded
  one ON). Detail: `versioning.md`.
- 🚨 **The DETACHED RIDE-ALONG heads live on the POLICY** (`ridealong_heads.py`), and three rules keep the
  baseline learning exactly what production learns, each with a test that fails on revert: build inside
  `torch.random.fork_rng` from `RIDEALONG_INIT_SEED`; never put them in `policy.optimizer` or PPO's loss;
  every input through `RideAlongBatch.detached`. A frozen random network is a BUFFER
  (`freeze_to_buffers`). New RND variants are built LAST from their own seed. Detail:
  `readouts_and_value_routes.md`.

## Hazards in the forward

🚨 **THE FORWARD HAS TWO PUBLIC SURFACES: the constructor signature and the obs DICT's KEY SET.** A route may
read a flag-gated Dict key and RAISE when absent; the mapping is DECLARED in `extra_obs_keys.py` (registry
currently EMPTY, mechanism kept). **Never hand-build `{"observation": zeros(1, D)}` on a path a run
reaches** — build from `zero_extra_obs` / `synthetic_obs` (that literal killed a launch two minutes in;
`extra_obs_keys_test.py` AST-gates it).

🚨 **The table layering only points DOWN** (`damage_tables` → `belief_tables` → `dex_ids`); an up-import is a
cycle that works only in one import order (`belief_tables_test.py`).

⚠️ **`forward` stays on `Gen3FeaturesExtractor`** (not a base of the `ExtractorBuild → ExtractorApi →
ExtractorForward → Gen3FeaturesExtractor` chain): the compile flags patch the bound `fe.forward` and tests
assign/restore `type(fe).forward`.

🚨 **The op's flat layout has ONE slicer** (`DamageOperator.tensors_from_block()` → `OpTensors`). Never
re-derive an offset at a consumer. 🚨 **`last_tensors` is POST-gain, for projections only**: a consumer using
an op value AS a probability or damage fraction reads it PRE-gain (`last_raw_tensors`; `last_raw_block` is the
detached prober copy, never a training input). A new block region needs a key in `out_gain_channel_keys`,
naming the channel only — never a slot, seat or position (`mon_tied_gain_test.py`).

**Stash contract** (`gen3_op_stashes_v1` / `gen3_extractor_stashes_v1`): every per-forward stash is a
dataclass field replaced at forward ENTRY; read through `last_*`, write through `op.stash.<f>` /
`fe.stash.<f>` (writing a `last_*` name raises); a submodule never writes its parent's stash
(`extractor_stashes_test.py`). 🚨 **`forward` is NOT re-entrant and there is NO guard**: a thread-sharing
caller gives each thread its OWN extractor or serializes forward + its `last_*` reads (two threads on one
extractor: 1,063 failures in 2,400 forwards, and same-batch-size races corrupt SILENTLY).

🚨 **Our active's moves live in TWO orders** — per-mon slots SORTED BY `Move.id`, the request block and
actions 6–9 in REQUEST order. Cross them by identity only (`extractor_ctx.active_request_sorted_match` /
`active_move_legality_sorted`); the positional form fired twice and is silent when every move is legal
(`move_legality_alignment_test.py`).

🚨 **An OPPONENT's ability is revealed only by its `known` flag** — an unrevealed slot carries the TOP-1
Smogon prior in `id1`. Read through `extractor_ctx.ability_known` / `revealed_ability1_ids` and take the
species marginal when unknown (`ability_known_gate_test.py`, allowlist EMPTY; `op_status_rules_test.py`).

🚨 **Every DISCRETE op in the forward is DECLARED in `selection_sites.py`** (`topk` / `argmax` / a value
comparison / a float→int cast in a `FORWARD_MODULES` module): `MARGIN` for a score operand, `EXACT` with a
reason otherwise. Keys are source text, so editing a declared line re-declares it. An `argmax` MARGIN rule's
`payload` must be EVERYTHING its index reaches; every `hypothesis_set.stable_order` caller declares how it
reads the order (`consumed`). Gates: `selection_sites_test.py`, `tie_identity_integration_test.py`; detail:
`op_contracts.md`.

## X5 hypothesis tokens and the static encoder

**X5 fixed_mass hypothesis tokens are the ONLY belief representation** (the blob path and `--belief-tokens`
are deleted; `model_version/version_break.py`). Every X5 read sits behind `hypothesis_builder is not None`.
A retired module leaves a plain `None` (`extractor_api.drop_child`); a constructor that draws from the global
RNG is never added or deleted without re-recording what it moves (the K9 golden).

- 🚨 **Every reduction over OPPONENT tokens declares its presence semantics** — a hidden slot is a hypothesis
  at presence π < 1, so reading it as a whole mon is the silent "counts as a whole mon" bug. An
  EXPECTATION-type reduction takes the per-key log π (`hypothesis_tokens.OppPresence` /
  `key_log_presence`) and gets its I1 / I2 test in `hypothesis_tokens_test.py`; a max-type one is
  presence-scaled. π is DETACHED wherever it weights policy or critic. A consumer of the flat pointer's
  α / β never multiplies π in again.
- 🚨 **A change to `PokemonEncoder`'s INPUT stitch is mirrored in `hypothesis_encode.species_table` / `_cols`**
  (`hypothesis_encode_test.py` compares against the per-row pass at fp64); a new row-level input goes in
  the per-row half, never the species table.
- **The op's opponent-MON axis under X5 reads `op.stash.x5` (an `OpRoster`)**: "alive" from `roster.alive`,
  candidates from `roster.move_order` / `move_w`. A new opponent-axis kernel branches the same way and gets
  a row in `x5_opp_mon_axis_test.py` (+ `EdgeBias.OTHER_FAMILIES` if OTHER should see it). A per-(seat, mon)
  cell is never multiplied by π.
- **The static per-mon encoder** (`--token-encoding static`, `static_tokens.py`; `--arch static_recovery`
  and `--arch endstate` likewise) is BUILT and OFF — NOT adopted; the legacy encoding stays until the
  end-state closing test. Its rules, enforced by `static_tokens_test.py`, `static_board_tokens_test.py`,
  `static_port_test.py`, `obs_facts_inject_test.py`: a per-mon input goes to S (pure function of the SET
  fields) or D (the mon's own state) by class, and NO board fact goes to either; a new per-mon obs column
  is classified there in the same pass, a new OBS-FACTS sub-block in `obs_facts_inject.FACTS_TOKEN_CLASS`;
  board facts go to `board_tokens.SIDE_FACTS` / `FIELD_FACTS`; no static weight is keyed by a position;
  a fact restating op physics READS the op's rule function. Field-by-field table:
  [`designs/endstate/design_static_tokens.md`](../../../designs/endstate/design_static_tokens.md) §1.

## `torch.compile`: load-bearing SPELLINGS

A backend that "can't compile our model" was one op, not the architecture — bisect to the op before any
global suppression. Gate: `extractor_compiles_test.py` (`GEN3AI_SKIP_COMPILE_TESTS=1` opts out; its CUDA
cells need a GPU lease). Diagnoses: `op_contracts.md`, `designs/training/compile_flags.md`.

- `BeliefHead.species_posterior` is **`log_softmax(...).exp()`, not `torch.softmax`** — do not simplify.
- 🚨 **Every gradient-path value-reduction max is `index_max.max_by_index`, never `amax`** (Inductor's
  recomputed backward made `amax`'s gradient 0/0 = NaN on CUDA). `amax` / `amin` only OFF a gradient path.
  **A max over THEIR believed candidates is spelled `op_reduction.believed_reduce(...)`** (else
  `--op-reduction principled` silently keeps your max); keep the expression VERBATIM — dynamo names graph
  nodes after locals.
- 🚨 **Every FLOAT attention bias reaches SDPA through `dense_attn_bias`** (`dense_attn_bias_test.py`).
- 🚨 **A forward reads PLAIN INTS precomputed in `__init__`, never a module attribute holding a sub-dict of
  `layout`** (an aliasing guard recompiled after the compile lock: `[CompileSentinel] FATAL`).

## Identity-at-init is NOT free — SB3 clobbers it

SB3's `_build()` orthogonally re-inits EVERY `nn.Linear` in the extractor. `restore_identity_init()` (called
by the policy after `super().__init__()`) re-zeros the Linears that were all-zero at the end of
construction — captured by observation, so a new zero-init module is protected automatically. 🚨 **Assert
any "identity-at-init / byte-identical / cold-start == prior" claim on a REAL `MaskablePPO`-built policy**,
never a bare module (`identity_init_test.py`).

## The flag registry — read it BEFORE adding a toggle

A model toggle is one `ModelFlag` row in `flag_registry.py` **plus five hand-synced surfaces**: the
`argparse` entry (**default `None`**, or its `_resolve` line is dead), the `_resolve("name", default)` line,
`extractor_arch.ARCH_ARG_KEYS`, `snapshot.current_model_version()`'s keyword, the `ModelVersion` field.
`flag_registry_test.py` names the missing site. Three TIERS (`cli` / `config_only` / `constructor_only`;
a `config_only` default is what every CLI run gets), four CLASSES (`structural` / `resume_immutable` /
`training_coef` / `runtime`), `requires=` as the dependency data (`flag_requires_test.py`; `main.checkargs`
reads it). `oracle_reveal` is the one OBSERVATION-MODE row (a diagnostic, never production). All of it:
`flag_registry_rules.md`; the current table: `designs/flag_registry.md` (GENERATED; `--check` is the gate).

## Model versioning — the playbooks

The live `MODEL_CONFIG_VERSION`, `ARCH_SIGNATURE` and `OBS_SEMANTICS_VERSION` are in
`model_version/constants.py`; `MIGRATION_FLOOR` is in `model_version/migrations.py`. **Never quote a version number in prose.** Full text: `versioning.md`.

- **Architecture constant changed:** `check_compatible()` catches it; old checkpoints correctly fail.
- **Optional new feature:** field in `ModelVersionFields` (`model_version/fields.py`) + bump
  `MODEL_CONFIG_VERSION` + one `if version < N:` `setdefault` block in `_migrate_config()`.
- **Structural change:** bump `ARCH_SIGNATURE`.
- 🚨 **Every value `--arch production` writes must be a `ModelVersion` field** (a launcher restart strips
  `--arch` and inherits only what `model_config.json` records; `main/train/derived_toggle_resume_test.py`).
- 🚨 **An obs cell's MEANING changes without a shape change:** set `OBS_SEMANTICS_VERSION` to the stamping
  config version, add its `OBS_SEMANTICS_REASON` clause, re-pin `obs_semantics_test.py`.
- 🚨 **DELETING an extractor kwarg** has no automatic gate: SB3 rebuilds the extractor from the ZIP's
  pickled kwargs, so the deleted name `TypeError`s every resume / pool opponent / eval worker. Judge it
  (`_DEAD_FEK_INERT` vs `_DEAD_FEK_JUDGED` in `snapshot.py`), add the matching `_migrate_config` entry, then
  update `ctor_kwarg_snapshot_test.CTOR_KWARGS_V96` LAST — the procedure is in `versioning.md`. A bare
  `MaskablePPO.load` of an old zip TypeErrors; loaders go through `snapshot.historical_load_kwargs`.
- **Parameter ORDER:** the optimizer state saves by position; `main/train/checkpoint_state._validate_or_reset_optimizer_state`
  remaps it BY NAME on every resume (`src/main/resume_optimizer_realign_test.py`). Appending new params
  LAST is hygiene, not load-bearing.
- **Resume-immutable hparams** (`vf_coef`, the reward fields) stay OUT of `check_compatible` and get a
  dedicated `check_*` on the resume path. The X5 version break raised `MIGRATION_FLOOR`: pre-floor
  checkpoints run PINNED (`model_version/version_break.py`).

## Opponent intent — X5's flat pointer as α / β

`--opp-intent-coef>0` turns on the supervised opponent-intent readout: ONE flat pointer over the opponent
active's move seats, OTHER_move, the six switch slots and OTHER_species (`flat_intent.py`); consumers read
its re-expression `fe.stash.flat_consumer_ops` (α / β). Supervision only (a DETACHED input). Detail and
metric inventory: `opponent_intent.md`.

- 🚨 **Read `opp_intent/*` by the `_pool` suffix, not the bare key** — the bare key is a MIX of opponent
  classes whose proportions MOVE over a run, so a rising pooled metric can be the mix, not the head.
- 🚨 **A `β` slot's name has two provenances** (revealed board vs the model's own posterior); traces flag it
  `"revealed"` and the read side attaches `BELIEF_NAME_CAVEAT` (`src/main/prober/CLAUDE.md`).
- **An α CONSUMER** (`pair_outcome.py` is the template): T1 produces, T2 consumes; read
  `FlatConsumerOps.alpha` / `.beta`, UNRENORMALIZED (OTHER_move a priced seat; the switch mass means "no
  outcome this turn"); align by construction and fail loud on a width mismatch
  (`intent_axis_alignment_test.py`); zero-init the projection; stop-grad α on a policy-side consumer.
  ⚠️ **A fallback that silently states something false is worse than refusing** (`SwitchBranchMoveCell`
  requires `opp_intent` rather than assert "they never switch"). A critic-facing α consumer extends
  `value_route_gradient_test.py` in the same pass.

## Static typing (mypy)

The model package is type-checked and the gate is **ZERO errors**:

```bash
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m mypy src/agents/model   # must be clean
```

Scope, strictness and annotation rules: [`designs/model/typing.md`](../../../designs/model/typing.md).
