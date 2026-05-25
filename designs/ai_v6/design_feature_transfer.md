# Design: Surgical Checkpoint Transfer for Architecture Changes

## Motivation

The current PPO model took 72+ hours to train. Expanding the species embedding (32→64)
or adding new per-Pokémon features (IV/EV/base stats for our own team) changes weight
shapes, causing `check_compatible()` to hard-fail and forcing a full retrain from scratch.

Surgical transfer loads a checkpoint into a new architecture by copying compatible weights
and zero-initializing new dimensions. The model plays identically to the source on day
one and gradually learns to use the new capacity during continued training.

---

## How It Works

**Zero-initializing new weight dimensions is safe.** New dims contribute zero to all
downstream activations, so model output is unchanged at transfer time. Training then
teaches the model what the new capacity means.

### Species embedding expansion (e.g. 32 → 64)

```
old: species_embedding.weight  [num_species, 32]
new: species_embedding.weight  [num_species, 64]
     → copy [:, :32] from old
     → zero-init [:, 32:]
```

The role encoder's first linear layer gains 32 new input columns (one species embedding
per slot). Same pattern: copy old columns, zero-init new columns. Everything else
(attention heads, projection, actor/critic) is unchanged in shape and copies directly.

### Adding new scalar features (IV/EV/base stats for our team)

New features appended to the per-Pokémon observation vector. Role encoder input dim grows
by N. Same pattern: copy old weight columns, zero-init new columns. All other layers copy
directly.

### What always transfers exactly

Move/item/ability/type embeddings (unchanged dims), all five attention heads, attention
pool, LayerNorms, projection, actor/critic heads.

---

## Planned Usage Sequence

```
Run 1 (current): species_embedding_dim=32, no IV/EV/stats  ← 72h already spent
    │
    ▼  --transfer-from
Run 2: species_embedding_dim=64                             ← ~10-15h to converge new capacity
    │
    ▼  --transfer-from
Run 3: species_embedding_dim=64 + IV/EV/base stats (own team only) ← ~10-15h
    │
    ▼  retrain from new checkpoint
Team completion model: slot_input_dim=96 (64+16+16)
```

Each step starts from a competent baseline rather than scratch. 72h is a one-time cost.

---

## Implementation Design

### New: `src/agents/model/transfer.py`

```python
def transfer_checkpoint(
    source_path: str,           # path to old .zip checkpoint
    target_model: MaskablePPO,  # freshly-constructed model with new architecture
    device: str = "cpu",
) -> dict:                      # {copied: [...], zeroed: [...], skipped: [...]}
```

Steps:
1. Extract `policy.pth` from source `.zip` (same approach as team completion model)
2. Get `target_model.policy.state_dict()`
3. For each key in source state dict:
   - Shapes match → direct copy
   - Target has more columns (input expansion) → copy source cols, new cols stay zero
   - Target has more rows (output expansion) → copy source rows, zero rest
   - Incompatible in any other way → skip with warning
4. `model.policy.load_state_dict(sd, strict=False)`
5. Return transfer summary

### New flag: `--transfer-from <path>` in `src/main/train_rl_agent.py`

Mutually exclusive with `--model`. Bypasses `check_compatible()` entirely — this is
intentional, the whole point is that the architecture has changed.

When present:
1. Build fresh model with current-code architecture
2. Call `transfer_checkpoint(args.transfer_from, model)`
3. Log transfer summary
4. Write `metadata.json` with `"transferred_from": args.transfer_from`
5. Run `_run_roundtrip_test()` smoke test
6. Proceed to `model.learn()` with a new timestamped run directory

### Launcher support in `src/main/launcher/`

**New `--transfer-from` flag** forwarded to `train_rl_agent.py`.

Key difference from `--model`:
- `--model` → resume existing run, pin worktree to checkpoint's original `git_hash`
- `--transfer-from` → new run, pin worktree to current HEAD

**Restart loop transition:** after the first child completes under `--transfer-from`,
subsequent restarts use `--model <latest checkpoint of new run>`. The launcher detects
this by checking whether the new run dir has any checkpoints after the first child exits.
This mirrors the existing `--model` restart logic — just bootstrapped differently.

---

## Asymmetry: IV/EV/Stats for Own Team Only

When adding IV/EV/base stats: we know these exactly for our own team but not the
opponent's. The observation encoder should zero those fields for opponent slots (the same
way `species_known=0` zeros unrevealed opponent Pokémon today). The model learns to use
the signal where it exists and ignore it where it doesn't.

---

## Verification

1. **Unit test for `transfer_checkpoint()`** — small synthetic model, fake checkpoint,
   assert old values preserved in overlapping region and new regions are zero
2. **Smoke test** — `--transfer-from <checkpoint> --debug --steps 10000`, confirm:
   - `[ModelVersion] Round-trip smoke test PASSED`
   - Transfer summary printed
   - New run dir created (not source run dir)
   - Win rate vs heuristic doesn't collapse
3. **Launcher restart** — verify first restart switches to `--model <new checkpoint>` not
   `--transfer-from` again
