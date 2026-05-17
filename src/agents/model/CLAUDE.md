# Model Directory — Contributor Notes

## Architecture constants — single source of truth

All network dims are defined as module-level constants at the top of `features_extractor.py`:

```python
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [64, 32]
ROLE_ENCODER_HIDDEN = [256, 128]
ACTIVE_CTX_HIDDEN = [64, 32]
```

**Change them here and nowhere else.** `__init__` reads from these constants; `ModelVersion` imports them so `model_config.json` always reflects the live values. Do not hardcode these numbers anywhere else in the codebase.

Embedding dims (`species_embedding_dim`, `move_embedding_dim`, etc.) live in `state_encoder.get_layout()` and flow through `features_extractor_kwargs` — same principle, different file.

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
- Changing the observation dimension (`base_dimension` or `dimension` in `state_encoder.py`)
- Adding or removing an attention path (Pressure / Safety / Synergy)
- Changing how `prev_mask` or future turn-history features are routed

The digraph is the fastest way for a new contributor (or Claude) to understand data flow. A stale digraph is worse than none.
