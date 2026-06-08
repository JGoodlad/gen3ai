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

### Fuzz tests (`*_fuzz_test.py`, no server — local BattleStream bridge)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/battle/event_log_fuzz_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_test.py
```

### E2E tests (`*_e2e_test.py`, requires running server)
```bash
# Start the server first (npm run showdown), then:
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/reward_invariants_e2e_test.py
```

---

## Training

### Via launcher (recommended for long runs)

The `launcher` package wraps the training script with periodic restarts to reclaim memory fragmentation, **crash auto-restart** (a self-crash relaunches from the last checkpoint after saving a per-crash `crashes/restart_err_<token>.txt`, bounded by a `--max-crash-restarts` circuit-breaker), a **Textual TUI dashboard** (built on the shared `src/main/tui/` base), and **git worktree isolation** — it pins the child process to the exact commit at launch so agent pushes to `main` can't affect a running session. A closed terminal (SIGHUP) or external `kill` is turned into a clean, checkpoint-saving shutdown.

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

Key launcher flags: `--restart-interval-hours` (default 3, set 0 for one-shot), `--max-crash-restarts` (default 3, consecutive rapid self-crashes to auto-restart through before giving up; 0 = unlimited), `--no-pin` (skip worktree isolation), `--sync-to-main` (when resuming, pin the worktree to the current HEAD instead of the checkpoint's original commit — useful for picking up UI/tooling fixes without discarding a checkpoint). All other flags pass through to `train_rl_agent.py`.

`python -m main.launcher.tui …` is a back-compat alias for the same command.

**TUI keys:** `l` logs · `e` events · `d` dashboard · `r` restart now · `c` force checkpoint · `p` plots · `s` status · `q` or Ctrl-C → confirm → `y`/`n` quit cleanly

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

### Self-play opponent distillation (`--distill-opponents`)

On a `--self-play` run you can distil the frozen pool opponents into a much cheaper network for
faster rollouts — the opponent forward is ~70% of env-worker CPU, so a faithful ~4.7–6.4× cheaper
opponent is roughly **+15–25% rollout throughput at near-zero quality cost**. Off by default; add
the flag to a self-play launch (directly or via the launcher):

```bash
... src/main/train_rl_agent.py --self-play --distill-opponents ...
```

It is **all-or-nothing** — the per-step `SubprocVecEnv` barrier means a single worker on the full
teacher straggles and gates the batch, so the pool is only ever 100% distilled or 100% full. An
idempotent reconcile loop handles this with no modes: enabling it **backfills the whole pool** (incl.
the sentinels) on the idle CPU, then **atomically switches** the pool to the cheap opponents; each new
promotion is distilled *before* it becomes an opponent. A fail-closed **gate** (fidelity + a greedy
head-to-head vs the teacher) validates each one, with capacity escalation + auto-revert on drift.
Distilled networks + their gate manifests land in `models/<run>/distilled/` (auto-cleaned as the pool
window slides); distilling runs in a non-blocking subprocess on `--eval-device` (CPU by default, no
GPU contention). The launcher TUI surfaces it live: a `⚗ distilled 100%` (green = speedup active) /
`⚗ distilling N%` (yellow = backfilling) badge, a `distill/*` metrics block, and Events-panel lines
for each gate result + the atomic full↔distilled switch — all on TensorBoard too. Full design +
empirical results: `designs/ai_v5/distill_integration.md`; module map:
`src/agents/training/distill/CLAUDE.md`.

Checkpoints save to `models/run_<timestamp>/` automatically, and that run's TensorBoard logs
live alongside them at `models/run_<timestamp>/tb/` (co-located, so a run is self-contained —
promoting it to a golden carries its curves along). The cwd-relative path lands in the main repo
even under the launcher's worktree pin.

### TensorBoard
Point `--logdir` at `models/` — TensorBoard recursively discovers every `models/*/tb/` (live runs
**and** `_goldens/`), each shown under its directory name, so current runs and reference goldens
compare side by side:
```bash
cd ~/dev/gen3ai && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/tensorboard --logdir ./models/ --host 0.0.0.0 --port 6006
```
For a curated, nicely-named subset, use `--logdir_spec`:
```bash
tensorboard --logdir_spec current:models/run_20260607_102632/tb,v3:models/_goldens/ai_v3_final_450m_step_05_26/tb
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
    training/        # Gen3Env, reward manager, battle snapshot + event-sourced
                     #   turn-delta fold, wrappers, stall detection,
                     #   eval + forensic-trace capture, self-play snapshot pool
  main/
    launcher/          # Restart loop + Textual TUI (preferred entry point)
                     #   checkpoint.py, worktree.py, child.py, input.py,
                     #   run.py, state.py, ui.py
    exit_codes.py      # TrainExitCode enum (COMPLETE/INTERRUPTED/CRASH)
    train_rl_agent.py  # Training script (also callable directly)
    eval_worker.py     # Subprocess bot-eval worker (frozen snapshot, CPU; spawned by PerOpponentEvalCallback)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/             # Gen 3 utilities, team loader, teambuilder, logging
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # ADV OU sample teams pool
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs); each holds its
                     #   own tb/ (TensorBoard logs) + _goldens/ (kept reference runs)
tensorboard/         # Legacy top-level TB tree (pre-move runs only; train_team_completion.py)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
designs/             # Architecture design docs
tools/               # Data generation and team sync utilities
```

---

## Observation Vector (3321-dim float32)

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 107) | 642 | 0 |
| Opp team (6 × 107) | 642 | 642 |
| Active context ×2 (boosts + full volatiles, `VOLATILE_DIM`=44) | 116 | 1284 |
| Global env | 18 | 1400 |
| Reactive + matchups | 302 | 1418 |
| Prev-turn action mask | 11 | 1720 |
| Turn history (`N_HISTORY_TURNS`=10 × 159) | 1590 | 1731 |
| **Total** | **3321** | |

Per-Pokémon slot (107 dims): species ID + 6 base stats, item block (id + known + consumed, 3), 2 type IDs, ability ID + known, 7-dim status one-hot, 4 × 11-dim move slots, HP fraction, species_known flag, sleep/toxic counters (2), **spread block (18: IVs ×6 + EVs ×6 + spread_known + nature ×5)**, **Hidden Power candidate block (17)**, active flag. Own-team IVs/EVs/nature are recovered from the declared team by the poke-env fork's `backfill_teambuilder_spread` (gen3ou has no team preview, so poke-env never attaches the spread); opponent spread is all-zero with `spread_known=0`.

Move slot (11 dims): move ID, base power (/200), has_secondary, has_recoil, type ID, category (status/physical/special), known flag, current PP, max PP, accuracy, never_miss bit.

Global env (18 dims): weather block (7: one-hot + cause-aware permanence + turns-remaining), spikes ×2 (2), log-turn (1), per-side screens (8: Reflect / Light Screen / Safeguard / Mist × both sides).

Reactive block (302 dims): 14 scalars (active-move power ×4 + multiplier ×4, fainted ×2, active-status, `forced_struggle`, **`trapped`**, **`maybe_trapped`**) then the two 144-dim matchup matrices. `trapped`/`maybe_trapped` (gen3_trapping_signals_v1) come from the server-authoritative `LegalActions` snapshot — `maybe_trapped` is the high-value one (switches stay legal, so it is the only way the model sees a possible Arena Trap / Shadow Tag / Magnet Pull before attempting a blind pivot).

Turn history — `N_HISTORY_TURNS`=10 TurnDelta slots of 159 dims each, **folded from the event log** (`Gen3Battle.events_since(cursor)` per decision window). Each slot carries both sides' move/type/species IDs (embedded) + outcomes (hit/miss/fail/crit), cant one-hots, boost and HP deltas, faint flags + multi-hot faint causes, status applied/cured transitions, item-used bits, the move we attempted (even if it never fired), and — gen3_trapping_signals_v1 — an `attempted_switch_rejected` bit + the attempted-switch species id for a pivot the server refused while trapped. All zeros on the first turn of each episode.

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
