# ai_v4 System Vision: From Better Observation to Future Simulation

## The Problem: Hidden Information

Gen3 competitive Pokémon is a hidden-information game. At battle start you see nothing; your opponent reveals their team one Pokémon at a time as they switch in. Spreads (IVs, EVs, nature), abilities, and Hidden Power types are never transmitted by the protocol. A standard RL agent trained purely on what's visible learns to react — it can never plan around what it doesn't know.

The ai_v4 → ai_v6 roadmap is entirely about solving this problem in three layers, each enabling the next:

1. **Observe better** — extract maximum signal from what *is* visible
2. **Complete the picture** — predict the parts that aren't
3. **Simulate the future** — use the completed picture to look ahead

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: OBSERVE BETTER  (ai_v4)                                │
│                                                                 │
│  Unified transformer    team + turn history attend together     │
│  Spread inference       own IVs/EVs, opponent battle stats      │
│  HP type inference      effectiveness signals → candidate mask  │
│                                                                 │
│  Output: 12 role tokens (128D each), history- and spread-aware  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ role tokens
┌───────────────────────────────▼─────────────────────────────────┐
│  TIER 2: COMPLETE THE PICTURE  (ai_v6 Steps 1–4)                │
│                                                                 │
│  Replay collection      download ladder battle logs             │
│  Behavioural cloning    pre-train policy on human moves         │
│  Team completion model  predict species / moves / item /        │
│                         ability / HP type for unseen slots      │
│  Enrichment             ability head + HP type head             │
│                         trained on curated full-team data       │
│                                                                 │
│  Output: sampled complete opponent team hypothesis              │
└───────────────────────────────┬─────────────────────────────────┘
                                │ fully-specified opponent team
┌───────────────────────────────▼─────────────────────────────────┐
│  TIER 3: SIMULATE THE FUTURE  (ai_v6 Step 5)                    │
│                                                                 │
│  Sim bridge (Node.js)   fork Showdown battle state, step turns  │
│  PIMC                   sample K team hypotheses per trajectory  │
│  MCTS                   UCB tree search, neural policy + value  │
│  Action selection       argmax visit count at root              │
│                                                                 │
│  Output: action with the best expected return across            │
│          plausible opponent team hypotheses                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Each Tier Depends on the Previous

### Tier 1 → Tier 2

The team completion model takes **role tokens** as its input. A role token is a 128-dim vector that encodes everything known about one Pokémon slot (species, moves, items, stats, condition). Under ai_v3, role tokens were computed by five hand-crafted attention paths over the current battle snapshot — they never saw turn history. Under ai_v4's unified transformer, each role token is produced after attending over all other team slots *and* the last 10 turns of battle history simultaneously.

Concretely: if Gengar consumed a Salac Berry on turn 4, the unified transformer lets every role token attend to that event. When the team completion model later asks "what item does their Gengar likely have?", its input already encodes "Gengar's item was consumed" — a signal that would be invisible to an ai_v3 role token.

Spread inference adds another dimension: per-slot accumulated battle statistics (total damage dealt to that slot, physical vs. special hit counts, min HP reached, healing observed, speed-tier signals) appear directly in the observation. These narrow the space of plausible spreads for the completion model.

### Tier 2 → Tier 3

MCTS requires a **fully-specified game state** to simulate. Without knowing the opponent's full team, every fork in the simulation tree hits unknown information — what moves can they use? what is their HP stat? Rollouts with incomplete info are noisy and lead to poor Q estimates.

With team completion, we sample a complete opponent team hypothesis at trajectory start and run all rollouts in that fully-observed world. Different trajectories sample different hypotheses; averaging Q and N values across trajectories integrates over team uncertainty in a principled way (Partially Observable Monte Carlo, PIMC).

### The Compound Effect

Each improvement in Tier 1 propagates all the way through:

```
Better unified transformer
  → richer role tokens
  → more accurate team predictions
  → more accurate rollout game states
  → better Q estimates
  → better action selection
  → higher win rate
  → more self-play data
  → better team completion training
  → (loop)
```

---

## Tier 1: ai_v4 Architecture Improvements

Three parallel improvements ship together. See the individual docs in this folder for full detail.

### Unified Transformer (implemented — see `impl_step4_unified_transformer.md`)

Replaces five hand-crafted attention paths (pressure, safety, synergy, threat, opp-synergy) and an isolated turn-history tower with a single transformer over 23 tokens:

- 6 our-team role tokens
- 6 their-team role tokens
- 10 turn-history tokens (embedded TurnDeltas)
- 1 global context token

All tokens attend to all other tokens in each of 2 transformer layers (d_model=128, 4 heads, FFN=256). This is the same architecture already designed for the team completion model in ai_v6 — the two transformers stack: the PPO extractor produces history-informed role tokens, the team completion transformer processes them.

### Spread Inference (`design_spread_inference.md`)

Three signals added to the observation:

1. **Own IV/EV encoding** (13 dims): own team's exact stat investment from team selection, never changes mid-battle. Lets the model reason about its own damage output accurately.
2. **Opponent battle stats** (9 dims per opp slot): accumulated per-battle — cumulative damage received, physical/special hit counts, minimum HP%, passive healing observed, speed-tier fraction. These narrow the posterior over opponent spreads.
3. **Auxiliary prediction losses**: `Gen3MaskablePPO` runs a second backward pass per PPO update predicting EV tier, nature, and spread archetype from the role tokens. Forces the embeddings to encode spread-predictive signal.

### Hidden Power Type Inference (implemented — see `impl_step2_hidden_power.md`)

HP type is never transmitted by Showdown, but can be inferred from effectiveness observations (e.g., HP hits Starmie for "super effective" → must be Grass or Electric, not Ice). A 17-dim block per opponent slot encodes:

- 1 dim: `hp_revealed` flag (did we see HP used this battle?)
- 16 dims: probability mass over 16 HP types, initialized from species priors, updated from effectiveness constraints

The matchup matrix in `reactive.py` uses this posterior to compute accurate expected damage, replacing the incorrect assumption that HP is always Normal type.

**Combined obs vector after all ai_v4 changes: ~1573 dims** (up from 1309).

---

## Tier 2: Team Completion Model (ai_v6 Steps 1–4)

Design details in `designs/ai_v6/design_team_completion_detail.md` and impl steps 1–4.

### Architecture

```
Input: 6 role tokens for revealed slots (from frozen PPO backbone)
       Mask token substituted for unrevealed slots (learned 128D parameter)
       ↓
Slot encoder: concat(species_emb, move_pool, item_emb) → Linear(64→128) → ReLU
       ↓
Completion transformer: 2 layers, 4 heads, 128D, no positional encoding
       ↓
Output heads (per slot):
  species   → softmax(387)           cross-entropy
  item      → softmax(N_items)       cross-entropy, only where revealed
  moves ×4  → sigmoid(N_moves) each  multi-label BCE (unordered set)
  ability   → softmax(N_abilities)   cross-entropy, curated data only
  hp_type   → softmax(16)            cross-entropy, where HP in moveset
```

No positional encoding because team slot order is arbitrary — the transformer must be permutation-equivariant.

### Training data

- **Self-play team log**: `--team-log` flag writes `{our_team, opp_team, winner, n_turns}` JSONL during PPO training. Both teams fully observed from the env's perspective.
- **Ladder replays**: 18,869+ downloaded replays → partial labels (species always, avg 1.82 moves revealed, 62.7% item coverage).
- **Curated teams**: 770 full team specs → complete move/ability/HP type labels. Weighted up 5–10× to correct for label noise in replays.

BERT-style training: randomly mask 1–5 slots, predict only masked slots, sum loss over masked positions.

### Target accuracy

| Metric | Target |
|--------|--------|
| `species_top1` | ≥ 25% (vs. ~8% "always Blissey" baseline) |
| `species_top1_by_ctx` | monotonically increasing with more revealed slots |
| `ability_top1` | ≥ 70% (on curated records) |
| `hp_type_top1` | ≥ 50% (where HP in moveset) |
| `move_recall_at_4` | measurable improvement vs. Step 3 baseline |

### Backbone relationship

The frozen backbone loaded into the team completion model is the PPO checkpoint's embedding tables (species 32D → 64D after ai_v4, move 16D, item 16D). The role encoder is **not** reused for team completion because it requires live battle context (boosts, conditions, matchups). The slot encoder is a simpler, static mapping.

After ai_v4 ships a new PPO checkpoint, the team completion model should be retrained with the new embedding dimensions.

---

## Tier 3: MCTS + Sim Bridge (ai_v6 Step 5)

Design details in `designs/ai_v6/impl_step5_mcts.md` and `impl_step5_sim_bridge.md`.

### Why MCTS on top of PPO

PPO is trained with a finite rollout horizon (n_steps=2048 env steps ≈ ~10 turns per update). It learns good short-horizon reactions but systematically underestimates the value of slow-burn strategies — Toxic + Leftovers accumulation, sacrifice-to-set-up, or switching to bait a predicted move. MCTS lookahead explicitly evaluates these multi-turn sequences.

### PIMC: handling the remaining hidden info

Even with a team completion model, some uncertainty remains. PIMC (Partially Observable Monte Carlo) handles this correctly:

1. At trajectory start, sample one complete opponent team hypothesis from the completion model
2. Run all rollouts for this trajectory in that fully-observed world
3. Different trajectories sample different hypotheses
4. Q and N values are averaged across trajectories → Q implicitly weights by team plausibility

### Sim bridge

A persistent Node.js subprocess wraps Showdown's `BattleStream` and exposes a fork-step-free protocol:

```
new      → initialize BattleStream, replay move history to sync root
advance  → replay one more confirmed move pair onto root
fork     → deep-copy current battle state (cheap in-process)
step     → execute one turn in a fork, return observation + done flag
free     → discard a forked state
```

The key insight: `Battle.fromJSON()` makes in-process forks cheap. No serialization crosses IPC boundaries for forks — only the initial `new` and `advance` messages carry game state. This is what makes ≥1000 rollouts/turn feasible.

### Search phases

**Phase 1 — Action sampling** (ships first, simpler):
- K=3 rollouts per legal action (4–6 actions in Gen3)
- Estimate Q(root, a) = mean return over K rollouts, max_depth=3
- Pick a* = argmax Q
- ~18 rollouts total, fits comfortably in 1s

**Phase 2 — Full MCTS**:
- UCB tree policy: `Q[s,a] + α · P[s,a]^β · sqrt(M[s]) / (N[s,a]+1)`
- 20 parallel worker processes + 1 aggregator
- Each worker runs ~50 rollouts, returns delta Q/N to aggregator
- Time budget: 8s (leaving 2s for Showdown round-trip)
- Target: ≥1000 rollouts/turn
- Root action: argmax N (visit count more robust than Q at root)

---

## The Data Flywheel

```
PPO self-play training (--team-log flag)
         │
         │  writes data/team_logs/run_*.jsonl
         │  {our_team, opp_team, winner, n_turns}
         ▼
Team completion training (ai_v6 Steps 3–4)
         │
         │  improves species/move/item/ability/HP predictions
         ▼
More accurate PIMC opponent hypotheses
         │
         │  lower variance Q estimates from MCTS rollouts
         ▼
Better action selection → higher win rate
         │
         │  more diverse, higher-quality self-play games
         ▼
More useful team completion training data
         │
         └──────────────────────────────────────────────── (repeat)
```

Each lap of the flywheel compounds: a 2% improvement in species accuracy propagates into a reduction in rollout variance, which compounds into better Q estimates, which compounds into win rate improvement.

---

## Implementation Sequence

Steps are ordered by dependency. Each step unblocks the next.

| Step | Doc | Dependency |
|------|-----|------------|
| 1. **Unified transformer** ✅ | `impl_step4_unified_transformer.md` | None — start here |
| 2. **Spread inference** (Signal 1 ✅) | `impl_step1_spread_encoding.md` / `design_spread_inference.md` (Signals 2–3) | Can overlap with step 1 |
| 3. **HP type inference** ✅ | `impl_step2_hidden_power.md` | Can overlap with steps 1–2 |
| 4. **Train new PPO checkpoint** | — | Requires steps 1–3 complete |
| 5. **Replay collection daemon** | `ai_v6/impl_step1_replay_collection.md` | Can start any time |
| 6. **Team completion model** | `ai_v6/impl_step3_team_completion.md` | Requires step 4 checkpoint |
| 7. **Enrichment** | `ai_v6/impl_step4_team_completion_enrichment.md` | Requires step 6 |
| 8. **Sim bridge baby steps** | `ai_v6/impl_step5_sim_bridge.md` | Requires step 4 checkpoint |
| 9. **MCTS Phase 1** | `ai_v6/impl_step5_mcts.md` | Requires steps 7 + 8 |
| 10. **MCTS Phase 2** | `ai_v6/impl_step5_mcts.md` | Requires step 9 |

**Behavioural cloning** (`ai_v6/impl_step2_bc.md`) is orthogonal — it pre-trains the PPO policy on human moves before RL fine-tuning. Run it in parallel with steps 5–6 if replay data is available. It improves rollout quality in MCTS by giving the policy better priors.

---

## Cross-References

All design detail lives in the docs listed below. This document is the map; those are the territory.

**This folder (`designs/ai_v4/`)** — the Tier-1 designs below shipped; their as-built records
are the `impl_step*` docs (the standalone design docs were folded in and retired):
- `impl_step4_unified_transformer.md` — full transformer architecture, as built
- `design_spread_inference.md` — IV/EV encoding (Signal 1 shipped, see `impl_step1`); battle-stat
  accumulation + aux losses (Signals 2–3) remain future
- `impl_step2_hidden_power.md` — HP type candidate mask, update logic, encoding, as built

**`designs/ai_v6/`** (team completion + MCTS):
- `design_team_completion_detail.md` — full team completion system design
- `impl_step1_replay_collection.md` — ladder replay daemon
- `impl_step2_bc.md` — behavioural cloning pipeline
- `impl_step3_team_completion.md` — model architecture + training pipeline
- `impl_step4_team_completion_enrichment.md` — ability + HP type heads
- `impl_step5_sim_bridge.md` — Node.js sim bridge baby steps
- `impl_step5_mcts.md` — full MCTS design (Phase 1 + Phase 2)
