# CLAUDE.md — the Model (`src/agents/model/`)

`designs/ARCHITECTURE.md` states the model as it is now: read it first; it wins any disagreement with this
file. This leaf holds the hazards to know before touching the model. **`designs/model/` owns the detail** and
changes in the same commit as the code; history is `designs/CHANGELOG.md`.

## Where the detail is

| I am about to touch… | Read |
|---|---|
| the phase chain, the tier order, `--belief-grad-mode`, X5's hypothesis tokens, the static encoder, identity-at-init | `designs/model/phase_pipeline.md` (`ARCHITECTURE.md` §2.1 first) |
| a readout off `value_pooled`, the dual head, a critic route, the ride-along heads | `designs/model/readouts_and_value_routes.md` |
| which file a class lives in, the architecture constants, table layering | `designs/model/file_layout.md` |
| the op's slicer / gain / stash contract, the move orders, ability-known, selection sites, compile spellings | `designs/model/op_contracts.md` |
| a model flag (add, demote, delete); `oracle_reveal` | `designs/model/flag_registry_rules.md` + the generated `designs/flag_registry.md` |
| `model_version/`, a config bump, a deleted kwarg, a resume-immutable hparam, the recorded `critic` | `designs/model/versioning.md` |
| the opponent-intent readout (α / β) and its metrics | `designs/model/opponent_intent.md` |
| the delivery graph, the architecture viewer | `designs/model/architecture_artifacts.md` |
| a type annotation, the mypy scope | `designs/model/typing.md` |
| a hand-computed feature (op row, pointer cell, prior table, board fact, reduction, derived obs column) | its row in `designs/endstate/design_hand_computed_features.md`, in the same commit |
| the S / D / board classification of a static-token input | `designs/endstate/design_static_tokens.md` §1 |

## Versioning — what to bump

The live `MODEL_CONFIG_VERSION`, `ARCH_SIGNATURE` and `OBS_SEMANTICS_VERSION` are in
`model_version/constants.py`; never quote one in prose. Playbooks: `designs/model/versioning.md`.

- 🚨 **A change to the phase structure or the forward math bumps `ARCH_SIGNATURE`**, even a pure decomposition
  (state_dict keys move). A forward-math change with unchanged widths is caught by nothing else.
- **A change in what an observation cell MEANS, with no shape change, raises `OBS_SEMANTICS_VERSION`**: append an
  `OBS_SEMANTICS_HISTORY` row (the config version the commit stamps + its clause) and re-pin `obs_semantics_test.py`.
  It does not bump `ARCH_SIGNATURE` and gates no resume; a reader acts on it through ONE gap policy
  (`main.prober.arch_status.decide`: every use below it runs badged; a row's `replay: refuse` blocks a trace replay). A
  model-internal input the obs golden cannot see is raised by hand.
- An optional new feature is a `ModelVersionFields` field, a `MODEL_CONFIG_VERSION` bump and one `_migrate_config`
  `setdefault` block. Every value `--arch production` writes must be a `ModelVersion` field (a restart inherits
  only what `model_config.json` records).
- Deleting an extractor kwarg breaks every load of an old zip, and no gate catches it: follow `versioning.md`. A
  lever the legacy deletion removes is judged by `model_version/adopted_levers.py`; code that builds the LEGACY
  network on purpose reads `main.train.arch_arms.legacy_production_config`, never typed flags.
- A `resume_immutable` flag the extractor carries is re-applied after every `--model` load
  (`resume_reapply.SETTERS`).
- 🚨 **A FORK grows the network only DECLARED** (`--grow FIELD=VALUE`, `growth.py`): a growable field is one
  `growth.GROWABLE` row with its own hooks, and the fork-time gate refuses anything not bit-identical to the parent
  (`versioning.md` "Declared growth at a fork").

## Hazards in the forward

- 🚨 **Whatever a trunk round computes lives INSIDE its class** (`trunk_kffn.TrunkRound`): the bf16 region is a
  class swap of the rounds, so a sibling module beside them silently runs fp32.
- A new width-contributing part extends `compute_projection_widths` and its sweep (`projection_width_test.py`).
- `Embeddings` alone owns the embedding tables; `ExtractorContext` is the inter-phase contract (add a field, do not
  widen a signature). The table modules import only downward (`damage_tables` → `belief_tables` → `dex_ids`).
- **Never hand-build an obs dict** (`{"observation": zeros(1, D)}`) on a reachable path: use `zero_extra_obs` /
  `synthetic_obs` (`extra_obs_keys_test.py`).
- **`forward` stays on `Gen3FeaturesExtractor`** (compile patches the bound method), and it is not re-entrant: a
  thread-sharing caller gives each thread its own extractor or serializes forward plus its `last_*` reads.
- The op's flat block has one slicer (`DamageOperator.tensors_from_block()`). `last_tensors` is post-gain, for
  projections only; a consumer reading an op value as a probability reads `last_raw_tensors`.
- 🚨 **Our active's moves live in two orders**: per-mon slots sorted by `Move.id`, the request block and actions 6–9
  in request order. Cross them by identity only (`extractor_ctx.active_request_sorted_match`); a positional crossing
  is silent whenever every move is legal (`move_legality_alignment_test.py`).
- An opponent's ability is revealed only by its `known` flag: read `extractor_ctx.ability_known`.
- **Every discrete op in the forward** (`topk`, `argmax`, a comparison, a float→int cast, a step function) is
  declared in `selection_sites.py` (`selection_sites_test.py`).
- **Every reduction over opponent tokens declares its presence semantics** (a hidden slot is a hypothesis at
  presence π < 1); π is detached wherever it weights policy or critic. A change to `PokemonEncoder`'s input stitch
  is mirrored in `hypothesis_encode.species_table`.
- A new per-mon input goes to S (a pure function of the set fields) or D (the mon's own state), and no board fact
  goes to either (`static_tokens_test.py` and its siblings).
- Detach belief LOGITS, never the product (`sigmoid(logits) @ emb.weight` detached kills the table's gradient). A
  belief loss reads `belief_supervision`: `src/agents/training/CLAUDE.md`.
- **Compile spellings are load-bearing**: `log_softmax(...).exp()` not `softmax` in `species_posterior`;
  `index_max.max_by_index`, never `amax`, on a gradient path; `op_reduction.believed_reduce` verbatim; float
  attention biases through `dense_attn_bias`; a forward reads plain ints precomputed in `__init__`. Bisect a
  compile failure to the op before any global suppression (`designs/model/op_contracts.md`).
- 🚨 **Every compile goes through `inductor_compile.compile_on`** (never a bare `torch.compile`; a scan fails the
  bypass): on CPU it adds the torch 2.8 tile2d step-around, without which the extractor dies in g++ at B >= 16
  (`designs/training/compile_flags.md` "The CPU Inductor tile2d codegen bug").

## Heads

- The extractor returns `(pi_features, vf_features)` and pairs only with `Gen3DualHeadMaskablePolicy`; `vf_features`
  is `value_pooled`, vf-only by construction (add a value route only in `_value_pooled_routes`).
- The win-prob critic is the only critic; read the recorded `critic` through `is_winprob`, never a string compare.
- The ride-along heads live on the policy: built inside `fork_rng` from `RIDEALONG_INIT_SEED`, never in
  `policy.optimizer`, every input through `RideAlongBatch.detached`.
- 🚨 **SB3's `_build()` re-inits every Linear; `restore_identity_init()` re-zeros them.** Assert any identity-at-init
  or byte-identical claim on a real `MaskablePPO`-built policy, never a bare module.

## mypy

`src/agents/model` and `src/agents/observation` share `mypy.ini` at zero errors (`mypy_gate_test.py`). Never loosen
`mypy.ini` to clear an error, since that de-tiers both packages; use a targeted `# type: ignore[code]` with a
reason. Detail: `designs/model/typing.md`.
