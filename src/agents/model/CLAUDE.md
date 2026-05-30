# Model Directory — Contributor Notes

## Architecture constants — single source of truth

All network dims are defined as module-level constants at the top of `features_extractor.py`:

```python
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]
ROLE_ENCODER_HIDDEN = [256, 128]
ACTIVE_CTX_HIDDEN = [64, 32]
```

**Change them here and nowhere else.** The phase modules' `__init__` read from these constants; `ModelVersion` imports them so `model_config.json` always reflects the live values. Do not hardcode these numbers anywhere else in the codebase.

Embedding dims (`species_embedding_dim`, `move_embedding_dim`, etc.) live in `state_encoder.get_layout()` and flow through `features_extractor_kwargs` — same principle, different file.

**`role_input_dim` is not a module-level constant** — it is computed dynamically in `PokemonEncoder.__init__` from the layout fields and `MOVE_NET_HIDDEN`. You do not need to update it manually when dims change; it is derived correctly. The projection input dim is also auto-discovered via a dummy forward pass for the same reason.

## Phase module structure

`forward_internal` is decomposed into phase `nn.Module`s, chained by a thin orchestrator:

`ObsUnpack` → `PokemonEncoder` → `TeamTransformer` → `CLSPool` → `ProjectionAssembler`,
then **two** root heads (`pre_proj_norm`/`projection` for policy, `value_pre_norm`/`value_projection`
for value), each → `ReLU`.

**Dual-head value readout (H4 / Option C).** The transformer body is shared, but the actor and
critic read it through independent paths. `CLSPool` holds a third query `value_cls` that attends
over all 12 team tokens to produce `value_pooled`; `ProjectionAssembler.forward` returns a
`(pi_combined, vf_combined)` pair; and the root `forward` returns a `(pi_features, vf_features)`
tuple. This extractor therefore **must** be paired with `Gen3DualHeadMaskablePolicy`
(`policy.py`), which keeps `share_features_extractor=True` (one body) and overrides `forward` /
`evaluate_actions` / `get_distribution` / `predict_values` to unpack the tuple and route each half
to `mlp_extractor.forward_actor` / `forward_critic`. A stock SB3 policy expects a single-tensor
extractor and will break. The startup `_run_roundtrip_test` and the snapshot/feature tests all
unpack the tuple — keep that in mind when touching the extractor's return shape.

Rules to preserve:

- **Each phase owns its layers** (`move_network` lives under `pokemon_encoder`, `our_cls` under `cls_pool`, etc.). State_dict keys are therefore phase-prefixed.
- **`Embeddings` is the sole owner of the 5 embedding tables + `hp_type_idx_map`.** It is passed as a **forward argument** to `PokemonEncoder` and `TeamTransformer` — never stored as a child attribute on them — so the tables register exactly once. (The root exposes read-only `@property` forwarders like `model.type_embedding` for convenience; those add no state_dict keys.)
- **`ExtractorContext`** (frozen-by-convention dataclass) is the inter-phase contract: `ObsUnpack` produces it, downstream phases read from it. Add a field here rather than widening a phase's positional signature. Cross-phase values (active-slot indices, fainted masks, `hp_probs`) are computed once in `ObsUnpack` and carried on the context.
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version.py` (current: `gen3_dual_value_v1`). Pure decompositions still change state_dict keys, so old checkpoints must fail loudly.
- Per-phase unit tests live in `phase_modules_test.py` — `CLSPool` (incl. the `value_cls` pool) and `ProjectionAssembler` (which returns `(pi_combined, vf_combined)`) are tested on a hand-built `ExtractorContext` (`_dummy_ctx`) without a full forward pass. Prefer adding precise phase-level tests there.

## Model versioning (`model_version.py`, `snapshot.py`)

Every model save writes `model_config.json` + `metadata.json` alongside the `.zip` via `save_model_snapshot()`. Loading goes through `load_model_snapshot()` which runs `check_compatible()` before `MaskablePPO.load()` — a mismatch fails fast with a clear error rather than silently loading bad weights.

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

A startup smoke test (`_run_roundtrip_test` in `train_rl_agent.py`) saves to a temp dir and reloads before every `model.learn()` call — catches serialization issues immediately.

## Keep the architecture digraph in sync

`designs/ai_v3/README.md` contains a Mermaid digraph and dimension reference table for the network.

**Update it whenever you change `features_extractor.py`**, specifically:

- Adding, removing, or resizing any layer (Linear dims, attention heads, embedding dims)
- Changing what gets concatenated into move processor input, role encoder input, or the final aggregation
- Changing the observation dimension (`base_dimension` or `dimension` in `state_encoder.py`) or `N_HISTORY_TURNS`
- Changing the unified transformer (token count, layers, heads, FFN dim) or the CLS pooling
- Changing the turn-history token embedding (slot count, embed dim) or the global-token composition
- Changing how `prev_mask`, move validity, or active-context routing works

> Note: the `designs/ai_v3` digraph predates the ai_v4 unified transformer and is stale on
> attention topology; the live high-level pipeline is the phase-module summary in the root
> `CLAUDE.md` "Feature Extractor Architecture" section.

The digraph is the fastest way for a new contributor (or Claude) to understand data flow. A stale digraph is worse than none.
