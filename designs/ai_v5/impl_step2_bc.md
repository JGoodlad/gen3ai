# Implementation: Step 2 — Behavioural Cloning

This step converts the raw replay logs collected in Step 1 into (observation, action)
pairs and pre-trains the PPO policy on them before handing off to RL fine-tuning.

## Motivation

RL from scratch requires the agent to stumble upon good play through exploration before
it can learn from it. In Gen 3 OU, good play involves non-obvious strategies (Spikes
pressure, prediction-based switches, sleep talk sequencing) that a random policy discovers
only after millions of transitions. Human replays provide a dense supervision signal over
these strategies at zero additional environment cost.

BC is not the end goal — RL will still fine-tune, and the BC policy may overfit to human
idiosyncrasies. But a BC-initialised policy trains faster and often reaches a higher
asymptote than a cold start, because RL exploration begins from a useful distribution
rather than uniform noise.

---

## Phase 1: Replay Parsing Pipeline

### Log Reader

Instantiate a `Battle` object and replay each line of the `.log` file through
`AbstractBattle.parse_message()`, exactly as poke-env's own client does. Two special
cases:
- `|win|username` → call `battle.won_by(username)` directly
- `|tie` → call `battle.tied()` directly

After each `|turn|N` line, the `Battle` object reflects the game state at the start of
turn N. This is when `Gen3ObservationEncoder.encode(battle)` should be called to capture
the observation the acting player would have seen.

### Action Label Extraction

The acting player's choice appears in the lines following `|turn|N` and before the next
`|turn|` or `|win|`/`|tie|`. Parse:
- `|move|PLAYER|MOVENAME|TARGET` → map `MOVENAME` to a move slot index via
  `Gen3ActionMapper`
- `|switch|PLAYER|SPECIES|HP` → map `SPECIES` to a switch slot index

The spectated log shows both players' choices, so both can be extracted. However, only the
**player-1 perspective** is used for BC (the observation is always encoded from player 1's
viewpoint). Player 2's actions can be used for data augmentation if the dataset is small.

### Mask Synthesis

The `|request|` JSON (which lists available moves with exact PP) is not present in
spectated logs. Two strategies:

**Option A — Skip unknown turns**: If PP data is unavailable and any move might be
depleted, skip the turn rather than risk an incorrect mask. Conservative; reduces dataset
size by up to 20% for long games where PP matters.

**Option B — Synthesise from move history**: Infer PP depletion by counting `|move|`
occurrences for each move across the game log. A move with ≥ `max_pp` uses is marked
depleted. Approximate but retains all turns.

Recommended: Option B for moves with `max_pp > 5` (most competitive moves); Option A
for PP-sensitive moves (Struggle edge cases). Mark the `prev_mask` block as all-ones when
uncertain — the model treats all-ones as "unknown" rather than "all valid."

### Output Format

Each parsed replay produces a list of `BCExample` records:

```python
@dataclass
class BCExample:
    obs: np.ndarray       # (OBS_DIM,) float32
    action: int           # index into the 11-action space
    mask: np.ndarray      # (11,) int8 — synthesised action mask
    battle_tag: str       # for traceability
    turn: int
```

Written to `data/bc/train.npz` and `data/bc/val.npz` (90/10 split by replay, not by
turn — prevents adjacent turns from leaking between splits).

---

## Phase 2: BC Pre-training

### Loss

Cross-entropy over the 11-action logits, masked to valid actions:

```python
logits = policy_network(obs)                  # (B, 11)
logits[~mask.bool()] = float('-inf')          # mask illegal actions
loss = F.cross_entropy(logits, action_labels) # (B,)
```

The PPO policy head is reused directly — no architecture change needed. The value head is
frozen during BC (we are only shaping the policy, not training the value estimator).

### Class Imbalance

Human players switch much less frequently than they use moves (~15% of actions are
switches). Without correction, BC over-represents the most common moves and produces a
policy that rarely switches. Fix: per-class weights inversely proportional to class
frequency, clipped to `max_weight=5.0` to avoid extreme upweighting of rare switches.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW, `lr=3e-4`, `weight_decay=1e-4` |
| Batch size | 512 |
| Epochs | 50 (early stopping on val loss) |
| Scheduler | Cosine annealing, `T_max=50` |
| Gradient clip | 1.0 |

### Stopping Criterion

Monitor `val/loss` each epoch. Stop when:
- Val loss has not decreased for 5 consecutive epochs (patience), or
- Val loss exceeds `1.05 × best_val_loss` (rising — model has started to overfit)

Do not optimise for val accuracy — BC accuracy is misleading because the model cannot
distinguish equally-good actions from wrong ones (human replays do not contain counterfactual
labels for the moves not chosen).

---

## Phase 3: RL Fine-tuning

Resume PPO training from the BC-pre-trained checkpoint. The BC policy provides a strong
starting distribution; RL explores from there and improves beyond human-imitable play.

Key concerns:
- **Entropy collapse**: BC can reduce action entropy significantly. Monitor
  `train/entropy_loss`; if it drops below −0.8 nats within the first 1M RL steps,
  temporarily increase `ent_coef` to 0.05.
- **Value head cold start**: the value head was frozen during BC. SB3's PPO will train it
  from scratch during RL fine-tuning. This is fine — the value head converges quickly once
  the policy is good — but expect noisier early returns than a warm-started value head.

Training command is identical to the standard v3/v4 training command with `--model` pointing
to the BC checkpoint. No new CLI flags needed.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/bc/log_reader.py` | Replays a `.log` file through poke-env, yields `(Battle, turn)` pairs |
| `src/agents/bc/action_extractor.py` | Parses move/switch labels from log lines, maps to action indices |
| `src/agents/bc/mask_synthesiser.py` | Infers action mask from move-use history |
| `src/agents/bc/dataset.py` | `BCDataset` — loads `.npz` files, applies class weights |
| `src/agents/bc/trainer.py` | BC training loop with early stopping, TensorBoard logging |
| `src/main/parse_replays.py` | CLI: `--replay-dir replays/showdown/1/ --out data/bc/` |
| `src/main/train_bc.py` | CLI: `--data data/bc/ --model output/bc_policy.zip --epochs 50` |

## Files to Modify

| File | Change |
|------|--------|
| `src/poke_env/battle/abstract_battle.py` | Expose `parse_message` cleanly for log replay (may already work) |

---

## Verification

1. **Log reader round-trip**: parse a known replay; confirm that `battle.team` and
   `battle.opponent_team` match the species visible in the log at each `|turn|N`.

2. **Action extractor**: for 10 hand-checked replays, confirm that the extracted
   (obs, action) pairs match what a human reading the log would expect. Pay particular
   attention to switch actions and to the turn after a faint.

3. **Dataset stats**: confirm ~85% move actions, ~15% switch actions in the raw dataset
   (before class weighting). Significant deviation suggests a parsing bug.

4. **BC training**: `val/loss` should decrease monotonically for at least the first 10
   epochs. A rising val loss from epoch 1 indicates a data pipeline problem (e.g.,
   observation mismatch between training and the replayed battle state).

5. **RL fine-tune**: at 100K RL steps from the BC init, win rate vs. `MaxDamagePlayer`
   should exceed the win rate achieved at 100K steps from a cold start (the v3 baseline).
   If not, the BC policy has not provided a useful prior.

---

## Final State

Step 2 is complete when:
- The `data/bc/` dataset contains ≥ 10K (obs, action) pairs from ≥ 500 unique replays
- BC val loss has converged and the checkpoint beats `MaxDamagePlayer` ≥ 70% in 100-game
  spot evaluation
- RL fine-tuning from the BC checkpoint reaches v3-equivalent win rate in ≤ 50% of the
  steps a cold start requires

**Ready for Step 3: Team Completion Model**

- The BC-then-RL checkpoint is the frozen backbone for the team completion model
- `data/bc/` replays also provide training data for team composition (both teams are
  fully revealed by game end)
