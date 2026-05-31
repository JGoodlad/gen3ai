# Gen3AI: Advanced Pokémon AI for Gen 3 OU

Reinforcement learning agent for Generation 3 Overused Pokémon battles, built on `poke-env` and a local Pokémon Showdown server.

## Project Goals

- Learn strategic play specific to ADV Gen 3: no physical/special split, Sandstream weather, Spikes/Rapid Spin, and high-stakes switching
- Train via PPO against a diverse opponent pool (random, heuristic, staller, aggressive, setup sweeper)
- Evaluate against progressively stronger opponents

---

## Environment Setup

Uses the **`gen3ai_stable` conda environment**. To create it from scratch:
```bash
conda env create -f environment.yml
```

To update an existing env after `environment.yml` changes:
```bash
conda env update -f environment.yml
```

Always prefix Python commands with:
```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

### Git Worktrees
When opening a new worktree, the `deps/pokemon-showdown` submodule is empty. Run:
```bash
git submodule update --init
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist deps/pokemon-showdown/dist
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules deps/pokemon-showdown/node_modules
```
Do **not** symlink the entire `deps/pokemon-showdown` directory — it breaks `git status` and VS Code git integration.

---

## Showdown Server

```bash
# Start (with performance flags) — port is a positional argument (no --port flag)
NODE_ENV=production node --turbo-fast-api-calls --max-old-space-size=6144 deps/pokemon-showdown/pokemon-showdown start --no-security 8000

# Or via npm (defaults to 8000; append an explicit port with --)
npm run showdown            # :8000
npm run showdown -- 8001    # :8001

# Stop cleanly (Ctrl+C orphans subprocesses — use this instead)
npm run stop                # stops :8000
npm run stop -- 8001        # stops :8001
```

The server runs on port 8000. Key config at `deps/pokemon-showdown/config/config.js` — subprocess counts (`simulator`, `network`) require a full restart; most other settings reload live.

To run a training server alongside a development server on 8000, start it on a separate port
(`npm run showdown -- 8001`) and point the trainer at it with `train_rl_agent.py --showdown-port 8001`
(forwarded by the launcher). The default is 8000; there is no environment variable.

---

## Testing

Three tiers of tests — run from the repo root:

| Pattern | Requires | Command |
|---|---|---|
| `*_test.py` | Nothing (pure unit tests) | See below |
| `*_integration_test.py` | Symlinked `deps/pokemon-showdown` Node bridge | See below |
| `*_e2e_test.py` | Live Showdown server on `localhost:8000` | Run directly as scripts |

### Unit tests only
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q
```

### Unit + integration
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

### E2E tests (requires running server)
```bash
# Start the server first, then:
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/gen3_env_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py
```

---

## Training

### Via launcher (recommended for long runs)

The `launcher` package wraps the training script with periodic restarts to reclaim memory fragmentation, a Rich TUI dashboard, and **git worktree isolation** — it pins the child process to the exact commit at launch so agent pushes to `main` can't affect a running session.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --steps 50000000 \
  --n-envs 96 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

Resume from a checkpoint (launcher reads the saved `git_hash` and pins to that commit):
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --model models/<run>/checkpoint_NNNN_steps.zip \
  --steps 50000000 \
  --device cuda
```

Key launcher flags: `--restart-interval-hours` (default 3, set 0 for one-shot), `--no-pin` (skip worktree isolation), `--sync-to-main` (when resuming, pin the worktree to the current HEAD instead of the checkpoint's original commit — useful for picking up UI/tooling fixes without discarding a checkpoint). All other flags pass through to `train_rl_agent.py`.

**TUI keys:** `r` restart now · `c` force checkpoint · `q` quit cleanly · `l` logs · `d` dashboard

### Direct (no restart loop)

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --steps 50000000 \
  --n-envs 96 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Debug mode
Single environment with full trace logging — no 96-env overhead:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug
```

Checkpoints save to `models/run_<timestamp>/` automatically. TensorBoard logs always write to `./tensorboard/` in the repo root.

### TensorBoard
```bash
cd ~/dev/gen3ai && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/tensorboard --logdir ./tensorboard/ --host 0.0.0.0 --port 6006
```

---

## Play / Evaluate

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running.

---

## Repository Structure

```
src/
  agents/
    action/          # Action masking, mapping, and fuzz tests
    inference/       # RLPlayer — loads a model checkpoint and battles
    model/           # Gen3FeaturesExtractor (PyTorch)
    observation/     # Observation encoders: species, moves, items, abilities,
                     #   active context, global env, reactive/matchups
    opponents/       # Scripted opponents: staller, aggressive, setup sweeper
    training/        # Gen3Env, reward manager, battle context, wrappers,
                     #   stall detection, replay recorder
  main/
    launcher/          # Restart loop + Rich TUI (preferred entry point)
                     #   checkpoint.py, worktree.py, child.py, input.py,
                     #   run.py, state.py, ui.py
    exit_codes.py      # TrainExitCode enum (COMPLETE/INTERRUPTED/CRASH)
    train_rl_agent.py  # Training script (also callable directly)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/             # Gen 3 utilities, team loader, teambuilder, logging
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # ADV OU sample teams pool
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs)
tensorboard/         # Training logs (always written here from any worktree)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
designs/             # Architecture design docs
tools/               # Data generation and team sync utilities
```

---

## Observation Vector (3299-dim float32)

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 107) | 642 | 0 |
| Opp team (6 × 107) | 642 | 642 |
| Active context ×2 (boosts + full volatiles, `VOLATILE_DIM`=44) | 116 | 1284 |
| Global env | 18 | 1400 |
| Reactive + matchups | 300 | 1418 |
| Prev-turn action mask | 11 | 1718 |
| Turn history (`N_HISTORY_TURNS`=10 × 157) | 1570 | 1729 |
| **Total** | **3299** | |

Per-Pokémon slot (107 dims): species ID + 6 base stats, item block (id + known + consumed, 3), 2 type IDs, ability ID + known, 7-dim status one-hot, 4 × 11-dim move slots, HP fraction, species_known flag, sleep/toxic counters (2), **spread block (18: IVs ×6 + EVs ×6 + spread_known + nature ×5)**, **Hidden Power candidate block (17)**, active flag. Own-team IVs/EVs/nature are recovered from the declared team by the poke-env fork's `backfill_teambuilder_spread` (gen3ou has no team preview, so poke-env never attaches the spread); opponent spread is all-zero with `spread_known=0`.

Move slot (11 dims): move ID, base power (/200), has_secondary, has_recoil, type ID, category (status/physical/special), known flag, current PP, max PP, accuracy, never_miss bit.

Global env (18 dims): weather block (7: one-hot + cause-aware permanence + turns-remaining), spikes ×2 (2), log-turn (1), per-side screens (8: Reflect / Light Screen / Safeguard / Mist × both sides).

Turn history — `N_HISTORY_TURNS`=10 TurnDelta slots of 157 dims each, **folded from the event log** (`Gen3Battle.events_since(cursor)` per decision window). Each slot carries both sides' move/type/species IDs (embedded) + outcomes (hit/miss/fail/crit), cant one-hots, boost and HP deltas, faint flags + multi-hot faint causes, status applied/cured transitions, item-used bits, and the move we attempted (even if it never fired). All zeros on the first turn of each episode.

---

## Model Architecture (`Gen3FeaturesExtractor`)

Decomposed into named phase modules: **`ObsUnpack` → `PokemonEncoder` → `TeamTransformer` → `CLSPool` → `ProjectionAssembler`**, then two root projection heads.

1. **Embedding lookups** — species (32-dim), move (16), item (16), ability (16), type (16, shared across Pokémon types, move types, and TurnDelta IDs)
2. **Shared move processor** — Linear→ReLU→Linear per move slot; includes per-move type matchup against all 6 opponents
3. **Within-Pokémon move self-attention** — MHA(32, 2 heads) + LayerNorm residual across the 4 move slots of each Pokémon
4. **Role encoder** — Linear→ReLU→Linear per Pokémon → 12 × 128 role tokens, with broadcasted global context and validity bits
5. **Unified transformer** — a 23-token sequence (6 our-team + 6 their-team role tokens + 10 turn-history tokens + 1 global token) with token-type and positional embeddings, run through a multi-layer `TransformerEncoder` under a fainted/empty key-padding mask. Replaces the old hand-crafted attention paths — every token attends to every other.
6. **CLS pooling** — one learned query per side pools its 6 team tokens → a 128-dim team token per side; a **third `value_cls` query** pools all 12 team tokens → a global value summary for the critic.
7. **Dual projection heads** — policy and value each get their own `pre_proj_norm` → `projection` → `ReLU`. `forward` returns a `(pi_features, vf_features)` tuple consumed by `Gen3DualHeadMaskablePolicy`, which routes each half to its own actor/critic MLP branch. The transformer body is shared; only the readout, projection, and critic MLP are independent (the **value-dedicated CLS readout**).

Both projection input dimensions are discovered via a dummy forward pass at init — no magic constants.

---

## Data Dependencies

Training requires JSON files in `data/pokemon/`:
- `gen3_species.json` — `{num, baseStats}`
- `gen3_moves.json` — `{num, basePower, type, hasSecondary, hasRecoil}`
- `gen3_items.json` — `{num}`
- `gen3_abilities.json` — `{num}`

---

*Built with love for the ADV community.*
