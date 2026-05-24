# Implementation: Step 3 — Team Completion Model

This step builds a masked-slot prediction model that, given the opponent's revealed
Pokémon mid-game, samples plausible complete teams for the unrevealed slots. It is the
world-sampling step for MCTS: at the start of each search trajectory, sample one complete
team hypothesis from this model and run the trajectory in that fully-observed world.

## Motivation

At most points in a Gen 3 OU game, three to five of the opponent's six Pokémon are
unrevealed. Naive MCTS cannot simulate rollouts without knowing the opponent's full team.
The reference paper (for Gen 4 Random Battles) addressed this by calling the server's
team-generation procedure. For Gen 3 OU, teams are human-built sets, not procedurally
generated — there is no server procedure to call. Instead, we train a model that learns
the joint distribution over Gen 3 OU team compositions from the replay corpus and self-play
data, then sample from it at MCTS time.

Raw usage statistics give only marginal distributions and miss the joint structure that
defines real teambuilding: a Skarmory nearly always comes with Blissey; a Spikes-and-Roar
Skarmory implies a physical wall elsewhere; weather-clear teams don't run Tyranitar. A
learned model captures these correlations without hand-coding them.

---

## Data Sources

| Source | Size | Notes |
|--------|------|-------|
| Scraped ladder replays (`replays/showdown/1/`) | 18,869 battles → 37,940 team records | Parsed with `replay_parser.py` into `data/replay_teams.jsonl` |

Gen 3 OU has **no team preview** (introduced Gen 5), so Pokémon are revealed only on
switch-in. Ground truth is therefore partial: species is always known for revealed slots;
moves and items are only known if they appeared in the battle. Average team record has 5.02
Pokémon revealed, 1.82 moves known, and 62.7% item coverage (boosted by parsing `|-heal|`
and `|-boost|` lines with `[from] item:` annotations in addition to `|-item|`/`|-enditem|`).

The mask pattern is re-sampled per `__getitem__` call so each DataLoader epoch sees
different masking. Loss is computed only over masked slots and, within items/moves, only
where the value was revealed in the replay.

---

## Architecture

### Frozen Backbone

Only the **embedding tables** are loaded and frozen from the PPO checkpoint. The full role
encoder and move processor are not reused because they require battle context (weather,
hazards, boosts, etc.) that doesn't exist in a static team record.

Extracted from `policy.pth` inside the SB3 `.zip`:

| Component | Dims | Key in state dict |
|-----------|------|-------------------|
| Species embedding | `[num_species, 32]` | `features_extractor.species_embedding.weight` |
| Move embedding | `[num_moves_in_ckpt, 16]` | `features_extractor.move_embedding.weight` |
| Item embedding | `[num_items, 16]` | `features_extractor.item_embedding.weight` |

All three are set `requires_grad=False`. Ability is **not** used — Gen 3 battles don't
reveal abilities, so there is no training signal for an ability head.

Move embedding may have more rows than the dataset vocabulary (backbone was trained with
up to 400 move IDs; dataset has 355). The slot encoder slices `move_emb.weight[:M]` where
`M = dataset.num_moves` to avoid shape mismatches.

### Trainable Components (~438K params)

**Slot encoder** — instead of the full role encoder, a lightweight per-slot encoder takes
the concatenation of frozen embeddings:
```
concat(species_emb[32], move_pool[16], item_emb[16])  →  Linear(64→128) → ReLU
```
`move_pool` is the weighted mean of move embeddings using the multi-hot input as weights
(sum of embeddings for known moves, divided by number of known moves). Masked slots have
all-zero inputs; their encoder output is discarded and replaced by `mask_token`.

**Mask token** — learned 128-dim parameter, substituted at masked slot positions.

**Completion transformer** — `nn.TransformerEncoder` (2 layers, 4 heads, 128-dim,
256-dim FF, no positional encodings). Team order is arbitrary, so no positional
encodings are used. Padding slots are masked out via `src_key_padding_mask`.

**Output heads** (applied only to masked non-padding positions):
- `species_head`: `Linear(128 → num_species)` — cross-entropy
- `item_head`: `Linear(128 → num_items)` — cross-entropy (supervised only where item was revealed)
- `move_head`: `Linear(128 → num_moves)` — **BCEWithLogitsLoss against a multi-hot target**

### Why multi-label BCE for moves

Moves are a **set** — "which of the `num_moves` moves is in this Pokémon's moveset?" —
not an ordered tuple. A single multi-label head eliminates move-order dependence entirely.
At inference, take the top-4 by sigmoid probability. This is cleaner and avoids the
autoregressive or permutation-matching complexity of 4 separate CE heads.

### Training Objective

```
L = CE(species_logits, target_species)                              # always for masked slots
  + BCE(move_logits, target_move_multihot)                          # always for masked slots
  + CE(item_logits, target_item) × item_known_mask                 # only where item was revealed
```

---

## Training Infrastructure

Entry point: `src/main/train_team_completion.py`

```
# Fresh run
python -m main.train_team_completion --backbone models/.../checkpoint.zip

# Resume
python -m main.train_team_completion --run-dir models/team_prediction/run_<timestamp>
```

Run directories mirror the PPO layout under `models/team_prediction/run_<timestamp>/`:

| File | Contents |
|------|----------|
| `checkpoint_epoch_NNNN.pt` | `{epoch, model, optimizer, val_loss, val_top1}` |
| `best_model.pt` | Copy of lowest-val-loss checkpoint |
| `latest.txt` | Filename of most recent checkpoint (for resume) |
| `metadata.json` | Git hash, args, and `evals` dict (see below) |
| `model_config.json` | Architecture params for reload validation |
| `command.txt` | Exact CLI invocation (fresh runs only) |

`metadata.json` `evals` block written at each checkpoint save:
```json
"evals": {
  "loss": 1.2341,
  "species_loss": 0.8912,
  "item_loss": 0.3124,
  "move_loss": 0.2195,
  "species_top1": 0.184,
  "species_top5": 0.521,
  "item_top1": 0.342,
  "move_recall_at_4": 0.613,
  "species_top1_by_ctx": {"1": 0.11, "2": 0.19, "3": 0.26, "4": 0.31, "5": 0.35}
}
```

Default hyperparameters: `--epochs 200 --batch-size 64 --lr 1e-3 --val-split 0.1 --save-every 5`.
Training is CPU-bound (model is tiny at ~438K trainable params); bottleneck is DataLoader
building multi-hot tensors. Optimal batch size 512 on GPU runs ~200 epochs in ~5 min.

---

## MCTS Integration

At the start of each MCTS trajectory:

1. Encode revealed opponent slots through the frozen backbone → role tokens.
2. Fill unrevealed slots with the learned mask token.
3. Run completion transformer → per-slot softmax distributions.
4. Sample one complete team hypothesis: for each unrevealed slot, sample species/item/moves
   from the per-slot distributions independently.
5. This sampled hypothesis becomes the "true" opponent team for the duration of this trajectory.

Trajectories see different hypotheses, so the MCTS visit counts aggregate over the
uncertainty about the opponent's team. The action selected at the root is optimal in
expectation over plausible worlds.

**Stochastic vs. argmax**: always sample stochastically at trajectory time. Argmax
produces the same hypothesis every trajectory, eliminating the uncertainty averaging that
makes PIMC work.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/agents/training/team_completion/replay_parser.py` | Parse `.log` files → `TeamRecord` JSONL; handles switch/drag/move/-item/-enditem/-heal/-boost events; two-pass for winner resolution |
| `src/agents/training/team_completion/team_dataset.py` | `TeamCompletionDataset` with BERT-style random masking; `Mappings` loads ID tables from `state_encoder.load_mappings()` |
| `src/agents/model/team_completion_model.py` | `TeamCompletionModel`: frozen embeddings + slot encoder + transformer + 3 output heads; `model_config()` for serialization |
| `src/main/train_team_completion.py` | Training entry point with resume, TensorBoard, Rich progress, PPO-style run dirs |

---

## Evaluation Metrics

All metrics are computed on the 10% held-out val split and logged to TensorBoard under
`team_completion/` and written to `metadata.json["evals"]` at each checkpoint.

| Metric | Description |
|--------|-------------|
| `species_top1` | Fraction of masked slots where predicted species (argmax) matches ground truth |
| `species_top5` | Fraction where ground truth is in top-5 predictions |
| `species_top1_by_ctx` | `species_top1` grouped by number of visible context Pokémon (1–5); diagnoses whether the model uses context |
| `item_top1` | Top-1 accuracy on masked slots where item was revealed in the replay |
| `move_recall_at_4` | For each masked slot, fraction of revealed moves that appear in the top-4 predicted moves |
| `species_loss` / `item_loss` / `move_loss` | Per-component val losses |

Baselines for species_top1:
- Uniform random over ~387 species: ~0.3%
- Always predict Blissey (most common): ~8%
- Target: ≥25%

## Verification

1. **Qualitative checks** — given `[Skarmory]` as context, top-5 species predictions
   should include Blissey; given `[Tyranitar]` predictions should favour sand-team staples.

2. **Context sensitivity** — `species_top1_by_ctx` should increase monotonically with
   context size. Flat values indicate the model is ignoring revealed teammates.

3. **Sampling sanity** — 100 sampled completions given "Skarmory revealed" should show
   Blissey appearing in > 50% of samples.

4. **MCTS smoke test** — Step 4 debug run should confirm `sample_completion()` returns a
   valid 6-mon team and that teams vary across trajectories.

---

## Current Status

Pipeline is implemented and runnable. Training command:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.train_team_completion \
  --backbone models/<run>/checkpoint_<N>_steps.zip \
  --data data/replay_teams.jsonl \
  --epochs 200 --batch-size 512 --lr 1e-3
```

## Final State

Step 3 is complete when:
- Training has converged on ladder replay data (species_top1 ≥ 25%)
- `species_top1_by_ctx` increases with context size (model uses context)
- Qualitative checks pass (Skarmory→Blissey, Tyranitar→sand team)
- A `predict_top_k()` wrapper is fast enough for MCTS (< 5ms per call)

**Ready for Step 5: MCTS**

The completion model is the world-sampling oracle MCTS depends on. Once trained,
Step 4 wires `predict_top_k()` (or a sampling wrapper) into the trajectory loop.
