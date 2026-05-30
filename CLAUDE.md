# CLAUDE.md — Gen3AI Project Guide

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

Architecture constants (embedding dims, layer sizes, etc.) are defined as module-level constants in `src/agents/model/features_extractor.py` — that is the single source of truth. When you change one, change it there and nowhere else. See [Model Versioning](#model-versioning).

## Documentation Maintenance

Keep docs in sync **automatically, as part of the same change** — no need to be asked:

- **Every `CLAUDE.md`** (root, `src/agents/model/`, `designs/`, anywhere): always current. If a change makes one stale, fix it in the same pass.
- **Every `README.md`** (including `designs/ai_v3/README.md`, the architecture digraph + dimension reference): always current. When you change dims, layout, obs/architecture, or anything a README documents, update it without being prompted.

**Do NOT auto-update other docs under `designs/`** — `impl_step*.md`, `design_*.md`, `todo.md`, etc. are explicit-only. Touch them only when the user asks (directly or via `/gen3ai-update-design-docs`). The lone exception is `CLAUDE.md` files inside `designs/`, which follow the always-current rule above.

## Git Workflow

This is a personal project — no pull requests needed. Work is pushed directly to `main`, but **all edits and commits must happen in a worktree or branch, never on the main checkout itself.**

Main must never be in a dirty state. The `/gen3ai-ship` skill is the only mechanism that lands code on main — it commits in the worktree branch and pushes:

```bash
git push origin <worktree-branch>:main
```

Never `git add` or `git commit` from `/home/goodlad/dev/gen3ai` directly.

**NEVER run `git add`, `git commit`, or `git push` unless the user's current message explicitly contains `/gen3ai-ship`.** Completing a task, writing tests, or any other finishing signal is NOT permission to commit. This applies even when the task feels "done". Only `/gen3ai-ship` authorises a commit.

---

## Python Environment

The project uses a dedicated conda environment, **not** `deps/venv`. Always prefix commands with the correct interpreter and `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

The conda env is `gen3ai_stable`. `deps/venv` exists but is outdated — ignore it.

---

## Git Worktree Setup

When opening a new git worktree (e.g. via Claude Code), the `deps/pokemon-showdown` submodule directory is created but left empty. Two steps are required before training or running tests:

**Step 1 — initialize the submodule** (gets source files, fixes VS Code git integration):
```bash
git submodule update --init
```

**Step 2 — symlink the build artifacts** (the submodule checkout has no compiled `dist/` or installed `node_modules/`, but the main repo already has them):
```bash
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist \
      deps/pokemon-showdown/dist
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules \
      deps/pokemon-showdown/node_modules
```

Both `dist/` and `node_modules/` are gitignored in pokemon-showdown, so these symlinks don't affect `git status`. Without step 2, training fails with `Cannot find module '.../dist/sim/index.js'`.

Do **not** symlink the entire `deps/pokemon-showdown` directory — git treats the submodule path as a symlink rather than a real checkout, which breaks `git status` and VS Code's git integration.

---

## Running Tests

### Test file naming conventions

| Pattern | Requires | Marker |
|---|---|---|
| `*_test.py` | Nothing — pure unit tests with mocks | — |
| `*_integration_test.py` | `deps/pokemon-showdown` Node bridge (no live server) | `@pytest.mark.integration` |
| `*_e2e_test.py` | Live Showdown server on localhost:8000 | `@pytest.mark.e2e` (scripts only, run directly) |

### Unit tests only (default)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q
```

### Unit + integration (requires symlinked deps/pokemon-showdown)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

### E2E tests (run directly as scripts, require a running server)
```bash
# Start server first: npm run showdown
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/battle/event_log_fuzz_e2e_test.py [n_battles]
```

### What "fuzz test" means in this project

**Fuzz tests are E2E tests that run real battles against the live Showdown server and validate observations or behaviour against the actual protocol stream.** They are NOT deterministic scenario tests with fixed inputs.

The canonical pattern (see `src/agents/training/poke_env_gaps/`):

1. Subclass `Player` and override `_handle_battle_message` to intercept raw Showdown protocol lines mid-battle.
2. Archive per-turn snapshots of the state you care about (e.g. which items were consumed, which moves were used).
3. In `choose_move()`, validate that the encoded observation vector matches what the archived protocol events say should be there.
4. Run N random battles; any validation failure raises immediately with a detailed error.

This catches poke-env parsing bugs and encoder gaps that unit tests with mocks cannot — the test exercises the real server → poke-env → encoder pipeline end to end. When asked to write a fuzz test, always follow this pattern rather than writing parametrized unit tests with hand-crafted mock objects.

---

## Smoke Test

Before a full training run, verify the full pipeline (env, reward, replay callback, stall detection, evaluation) with a quick debug run (~1 min):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

What to look for:
- `[ModelVersion] Round-trip smoke test PASSED` — serialization and reload are healthy (printed early, before training begins)
- `🏁 Episode Finished` lines appearing throughout — episodes completing and resetting
- `🎥 [REPLAY]` fires once early (step 1), then training continues — replay callback works
- `[STALL LOGGED]` may appear if a 250-turn game occurs — should be followed by another `🏁 Episode Finished`, not a hang
- `Win rate vs Random` and `Win rate vs Heuristic` printed at the end — evaluation ran

A hang after `[STALL LOGGED]` or a crash before "Training complete" indicates a regression in the env/stall/forfeit pipeline. A `[ModelVersion] FATAL` error at startup means the checkpoint was saved with a different architecture than the current code.

---

## Launcher (preferred for long runs)

`src/main/launcher/` wraps `train_rl_agent.py` with:
- **Periodic restarts** — kills and relaunches the child every N hours to reclaim pymalloc fragmentation; the child saves a checkpoint on SIGTERM and the launcher picks it up automatically
- **Worktree isolation** — at startup, creates a detached git worktree pinned to the current HEAD (or to the commit recorded in the checkpoint's `metadata.json` when resuming). Agent pushes to `main` never affect a running session
- **Rich TUI** — live dashboard showing metrics, FPS, restart countdown; `l` for logs, `r` to restart now, `c` for forced checkpoint, `q` to quit cleanly
- **Crash reporting** — child stdout/stderr is streamed live to `<run_dir>/launcher_child.log` (complete even if the child hard-`os._exit`s, bypassing Python cleanup) and held in a 5000-line in-memory scrollback. On a non-zero exit the last 100 lines are dumped to the terminal after the TUI closes; on *every* exit (crash, complete, quit) the full log path is printed and the file is finalized (the in-memory buffer is flushed to it as a fallback if streaming never started)

### Exit codes (`src/main/exit_codes.py`)

| Code | `TrainExitCode` | Meaning |
|------|----------------|---------|
| 0 | `COMPLETE` | All steps done — launcher stops |
| 15 | `INTERRUPTED` | SIGTERM received, checkpoint saved — launcher restarts |
| 1 | `CRASH` | Unhandled exception — launcher stops, crash log printed |

### Starting a fresh run via launcher
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Resuming from a checkpoint
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --model models/<run>/checkpoint_NNNN_steps.zip \
  --steps 15000000 \
  --device cuda
```

The checkpoint must have a `metadata.json` with a `git_hash` field (written automatically by `save_model_snapshot()`). The launcher pins the worktree to that exact commit so the resumed run uses the same code as the original.

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--restart-interval-hours` | `3.0` | Set to `0` for a single run with no restart |
| `--no-pin` | off | Skip worktree creation; run from the current source tree |
| `--sync-to-main` | off | When resuming from a checkpoint, pin the isolated worktree to the current HEAD instead of the checkpoint's original git hash. Use this to pick up UI or tooling fixes on `main` without discarding the checkpoint. |

All other flags are forwarded verbatim to `train_rl_agent.py`.

---

## Training

Run directly (no restart loop, no worktree isolation):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model <path/to/checkpoint.zip> \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

Omit `--model` to start a fresh run. Use `--debug` for a single env (DummyVecEnv). Use `--device cpu` on machines without a GPU.

Checkpoints are saved automatically. Models land in `models/run_<timestamp>/`.

---

## Playing / Evaluation

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running (see below).

---

## Showdown Server

Start the local Pokémon Showdown instance:

```bash
npm run showdown
```

Stop it:

```bash
npm run stop
```

The server must be running for any battles (training or play). It binds to port 8000 by default.

---

## Repository Structure

```
src/
  agents/
    model/           # Gen3FeaturesExtractor (PyTorch feature extractor)
    observation/     # Observation encoders (state_encoder, pokemon, moves, etc.)
    action/          # Action masking and mapping
    training/        # Callbacks and reward manager
    battle/          # Event-sourced battle layer (Gen3Battle, BattleEvent log, TurnView)
  main/
    launcher/          # Restart loop + Rich TUI (preferred for long runs)
                     #   checkpoint.py, worktree.py, child.py, input.py,
                     #   run.py, state.py, ui.py
    exit_codes.py      # TrainExitCode enum (COMPLETE=0, INTERRUPTED=15, CRASH=1)
    train_rl_agent.py  # Training entry point (also callable directly)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/
    git.py           # get_git_hash(), get_repo_root()
    (other utils)    # Hidden Power, teambuilder, team loader, logging
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # Downloaded sample teams (gen3ou pool)
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
```

---

## Observation Vector

The full observation is a **2734-dim float32 vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset | Notes |
|---|---|---|---|
| Our team (6 × 107) | 642 | 0 | base encoder |
| Opp team (6 × 107) | 642 | 642 | base encoder |
| Active context ×2 | 46 | 1284 | base encoder |
| Global env | 13 | 1330 | base encoder |
| Reactive + matchups | 300 | 1343 | base encoder |
| Prev-turn action mask | 11 | 1643 | appended by `gen3_env.embed_battle()` |
| Turn history (`N_HISTORY_TURNS` × 108) | 1080 | 1654 | appended by `gen3_env.embed_battle()`; oldest first (`N_HISTORY_TURNS = 10`) |

Per-Pokémon slot (107 dims): species ID + 6 base stats, item ID + known + consumed, 2 type IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 11-dim move slots, HP fraction, species_known flag, sleep_counter_norm, toxic_counter_norm, **spread block (18 dims)**, **HP-candidate block (17 dims)**, active flag. The item block is 3 dims: `[item_id, known, consumed]` — `consumed=1` when the item was spent this battle (Berry activated, Knock Off, Trick, etc.) and `item_id` retains the identity of the consumed item so the model knows what was lost. `species_known = 1.0` for all populated slots (own team and revealed opponent mons), `0.0` for unseen opponent slots. Sleep counter: `min(turns_slept, 4) / 4` (Gen 3 max 4 turns); toxic counter: `min(turns_poisoned, 8) / 8` (practical max before fainting with Leftovers).

Move slot (11 dims, layout in `src/agents/observation/moves.py`): move ID, base power (/200), has_secondary, has_recoil, type ID, category (0=status, 1=physical, 2=special), known flag, current PP (/MAX_PP), max PP (/MAX_PP), accuracy (raw% / 100), never_miss bit. Accuracy is split into a continuous scalar plus a categorical bit: never-miss moves carry accuracy=100 in the mapping → encode as `[1.0, 1]`, while a genuine 100%-accuracy move is `[1.0, 0]` — same scalar, distinguished only by the bit. A 100%-accuracy move can still miss into evasion (Double Team) or after Sand-Attack; a never-miss move (Swift, Aerial Ace, all status/self moves) bypasses the accuracy/evasion check entirely.

Spread block (18 dims, appended at offset 71 within each slot): IVs ×6 each/31 + EVs ×6 each/252 + spread_known (1.0 own, 0.0 opp) + nature modifiers ×5 [atk, def, spa, spd, spe] as raw floats (0.9/1.0/1.1). Opponent slots have all 18 dims as zeros; `spread_known=0` distinguishes "unknown opponent" from "own Pokémon with 0 EVs".

Global env (13 dims): weather one-hot (6), spikes ×2 (2), log-turn (1), our reflect (1), our light screen (1), opp reflect (1), opp light screen (1).

Each TurnDelta slot (108 dims, layout in `src/agents/observation/turn_delta_encoder.py`; all offsets computed from named `OFFSET_*` / `*_DIM` constants — never hardcode indices):

- **Base block (51 dims, indices 0–50)** — our/opp move features (5 each: raw move_id int, power_norm, has_secondary, has_recoil, raw type_id int), switched/failed flags, cant onehots (`CANT_DIM` = 11 ea: par/slp/frz/flinch/confusion/…), summed HP deltas, faint flags, opp_move_known, effectiveness onehots (4 ea), move-order (2).
- **Extended block (57 dims, indices 51–107, added in `gen3_unified_v2`, extended in `gen3_move_outcome_v1`)** — our/opp boost deltas (7 each, in BOOST_STATS order); `phase_is_forced_switch` (1, distinguishes half-turn replacement slots from full action-pair slots); our/opp `target_hp_delta` (1 each, HP delta on the named target of each side's damaging move); per-side **HP-level vectors** (6 each, end-of-turn HP for every team slot, giving the model the full HP trajectory across the window); our/opp **target_status onehots** (7 each, status of the named target AT MOVE-FIRE TIME — for Flash Fire-vs-frozen and sleep-talker reads); our/opp **move-outcome onehots** (3 each: `[hit, miss, fail]`, all-zero when the side switched / was prevented by cant / used no move); our/opp **move-crit** (1 each, orthogonal to outcome — a hit slot may also have crit=1); 6 raw species IDs — `our_actor` / `opp_actor` / `our_target` / `opp_target` / `our_switch_to` / `opp_switch_to` — embedded by the extractor through `species_embedding`.

The extractor embeds all `N_HISTORY_TURNS` slots identically through shared move (16) / type (16) / species (32) embedding tables: 4 raw move/type IDs → 4×16 + 6 raw species IDs → 6×32 + 78 pass-through scalars = 334-dim per slot. Positional encodings are added, one self-attention pass runs, and the last (most-recent) slot's output flows into the projection block. All zeros on the first turn of each episode.

Actor species resolution prefers `damaging_event.user_species` (protocol-truth) and falls back to `prev_active` for switches and non-damaging moves; target species comes from the OTHER side's `damaging_event.target_species`. Species ID 0 is the unknown sentinel.

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py` is decomposed into
named phase `nn.Module`s, each owning its layers, chained by a thin `forward_internal`
orchestrator. Data flows:

**`ObsUnpack` → `PokemonEncoder` → `TeamTransformer` → `CLSPool` → `ProjectionAssembler`**, then **two** root projection heads (policy + value), each `pre_proj_norm` → `projection` → `ReLU`. `forward` returns a `(pi_features, vf_features)` tuple — the transformer body is shared, but the actor and critic read it through independent CLS pools and projection heads (the **value-dedicated CLS readout**, H4 / Option C). This extractor is paired with `Gen3DualHeadMaskablePolicy` (`src/agents/model/policy.py`), which unpacks the tuple and routes each half to its own `mlp_extractor` branch; stock SB3 policies assume a single-tensor extractor and won't work.

The embedding tables live in a shared `Embeddings` module and are passed as a forward
argument to the phases that need them (Pokémon encoding and turn-history embedding), so
they register exactly once. An immutable `ExtractorContext` dataclass produced by
`ObsUnpack` carries the ~30 unpacked tensors to the downstream phases, keeping each
phase's forward signature narrow. See `src/agents/model/CLAUDE.md` for the phase contract.

1. **`Embeddings`** — shared tables: species (32), move (16), item (16), ability (16), type (16, shared for Pokémon types, move types, and TurnDelta move/type IDs). Owns the Hidden Power soft-type blend (`hp_soft_type`) and the per-slot TurnDelta embedder (`embed_delta_slot`).
2. **`ObsUnpack`** (stateless) — peels the flat 2734-dim observation into the named tensors of `ExtractorContext`: per-Pokémon block + categorical IDs, the global/reactive feature slices, the matchup matrices, and (hoisted here) the active-slot indices + fainted key-masks used downstream.
3. **`PokemonEncoder`** — embeds + stitches the enriched per-Pokémon vector; runs the **shared move processor** (Linear→ReLU→Linear, `MOVE_NET_HIDDEN`) over every move slot (input: move/type embeddings, remnants, known flag, battle context, per-move matchup ×6 + matchup-validity ×6, HP-candidate distribution, and prev-turn move validity), a **within-Pokémon move self-attention** (MHA 32-dim, 2 heads, + LayerNorm residual), then the **role encoder** (Linear→ReLU→Linear, `ROLE_ENCODER_HIDDEN`) → 12 × 128 role tokens.
4. **`TeamTransformer`** — builds a 23-token sequence (6 our-team + 6 their-team role tokens + `N_HISTORY_TURNS`=10 history tokens + 1 global token), adds token-type and history-positional embeddings, and runs a `TRANSFORMER_N_LAYERS`-deep `nn.TransformerEncoderLayer` stack (d_model 128, `TRANSFORMER_N_HEADS` heads, FFN `TRANSFORMER_FFN_DIM`, post-LN) under a key-padding mask that masks fainted team slots and empty history slots. History tokens come from `embed_delta_slot`; the global token from the two active-contexts + non-matchup scalars. Returns the two refined team-token blocks.
5. **`CLSPool`** — one learned CLS query per side cross-attends over its 6 post-transformer team tokens (fainted slots key-masked) → a 128-dim pooled team token per side (+ LayerNorm). Also extracts `our_active_refined` = the transformer output of our active slot. A **third learned query, `value_cls`**, cross-attends over **all 12 team tokens** (both sides, fainted key-masked) → a 128-dim global `value_pooled` summary — a whole-board "who's winning" read for the critic, a different aggregation than the policy's our-active-centric pools.
6. **`ProjectionAssembler`** — emits a `(pi_combined, vf_combined)` pair. Policy: `our_pool(128) + their_pool(128) + our_active_refined(128) + active_ctx_enc(32) + opp_ctx_enc(32) + non_matchup_rest`. Value: `value_pooled(128) + active_ctx_enc(32) + opp_ctx_enc(32) + non_matchup_rest`. `active_ctx_encoder` (Linear→ReLU→Linear, `ACTIVE_CTX_HIDDEN`) is shared by both heads — it encodes inputs, not the contested body representation.
7. **Root heads** — two parallel `pre_proj_norm` (LayerNorm) → `projection` (Linear) → `ReLU` heads, one per `*_combined`, both emitting `PROJECTION_DIM`. SB3 sizes the shared `mlp_extractor` from `features_dim = PROJECTION_DIM`, then `Gen3DualHeadMaskablePolicy` feeds the policy half to `forward_actor` and the value half to `forward_critic`.

Both projection input dimensions are discovered automatically via a dummy forward pass in
`__init__` (run through the assembled phases), so they stay correct when the architecture
changes without any manual update.

---

## Model Versioning

Every model save writes two files alongside the `.zip`:

- `model_config.json` — all weight-shape-relevant architecture params (embedding dims, layer sizes, obs dim, etc.)
- `metadata.json` — git hash, timestamp, SB3/Python versions

`load_model_snapshot()` in `src/agents/model/snapshot.py` checks these before calling `MaskablePPO.load()`. A mismatch causes a hard `[ModelVersion] FATAL` error at startup, not a silent wrong-output bug later.

### Key files

| File | Purpose |
|------|---------|
| `src/agents/model/features_extractor.py` | Module-level constants `ROLE_TOKEN_SIZE`, `PROJECTION_DIM`, `MOVE_NET_HIDDEN`, `ROLE_ENCODER_HIDDEN`, `ACTIVE_CTX_HIDDEN` — **single source of truth** for architecture dims |
| `src/agents/model/model_version.py` | `ModelVersion` dataclass, `MODEL_CONFIG_VERSION`, `ARCH_SIGNATURE`, `_migrate_config()` |
| `src/agents/model/snapshot.py` | `save_model_snapshot()` / `load_model_snapshot()` |
| `src/agents/model/snapshot_test.py` | Unit tests including a full save→load→forward-pass round-trip |

### When you change the architecture

**Changing a dim** (e.g. `ROLE_TOKEN_SIZE = 128 → 256`):
1. Update the constant in `features_extractor.py`
2. `check_compatible()` catches it automatically — old models can't load, which is correct

**Adding an optional feature** (e.g. dropout):
1. Add the field with a default to `ModelVersion` in `model_version.py`
2. Bump `MODEL_CONFIG_VERSION` (e.g. 1 → 2)
3. Add a migration in `_migrate_config()`: `if version < 2: data.setdefault("dropout_rate", 0.0); data["config_version"] = 2`
4. Decide if the field is weight-relevant (add to `_WEIGHT_FIELDS` in `check_compatible`) or advisory (skip)

**Structural change** (e.g. adding LSTM — different forward pass):
1. Change `ARCH_SIGNATURE = "gen3_lstm_v1"` in `model_version.py`
2. Old models fail with a clear arch-family error; no code needed to support them

### Startup smoke test

`_run_roundtrip_test()` in `train_rl_agent.py` runs automatically before `model.learn()` on every training start. It saves to a temp dir, reloads, and runs a zero forward pass. If serialization is broken it crashes in seconds rather than hours.

---

## Data Dependencies

Training requires JSON mapping files in `data/pokemon/`:
- `gen3_species.json` — species ID → `{num, baseStats}`
- `gen3_moves.json` — move ID → `{num, basePower, type, hasSecondary, hasRecoil}`
- `gen3_items.json` — item ID → `{num}`
- `gen3_abilities.json` — ability ID → `{num}`

These are loaded at startup and will raise `FileNotFoundError` / `ValueError` if missing or empty.

---

## Event-Sourced Battle Layer (`src/agents/battle/`, ai_v4)

poke-env is a **state tracker** — each `|...|` protocol line overwrites "current board"
fields. RL/reward/replay need the opposite: *what happened, in order*. The event-sourced
layer captures that without reimplementing poke-env (design:
`designs/ai_v4/design_event_sourced_battle.md`). **Status: steps 1–2 implemented** (event
log + completeness/conservation + ergonomic per-turn read model). The `TurnDelta`/reward
fold onto it (step 5) is not done yet — nothing in training consumes the log yet, so this
layer is currently additive and obs/reward-neutral.

- **`Gen3Battle(Battle)`** (`gen3_battle.py`) — subclasses poke-env's singles `Battle`.
  Its `parse_message` override classifies each line, calls `super().parse_message`
  (state tracking is **verbatim** poke-env), then appends a `BattleEvent` with
  attribution resolved **before** the line mutates state. State-equivalence with the
  classic `Battle` is structural (every line still flows through `super()`).
- **`BattleEvent` / `EventKind` / `MESSAGE_POLICY`** (`battle_event.py`) — the immutable,
  ordered schema and the completeness registry. Every protocol keyword poke-env can emit
  is classified `EVENT` / `STATE_ONLY` / `CONTROL` / `COSMETIC` / `UNSUPPORTED`; an
  unclassified or non-gen3 keyword **raises** (a deliberate tripwire). The conservation
  invariant (`Gen3Battle.assert_conservation()`) proves no line is silently dropped.
- **`TurnView`** (`turn_view.py`) — the **history** read surface ("what happened, in
  order"). Folds one turn's events into per-side intent (`move_id`, `switched`,
  `cant_reason`/`cant_move`, `crit`/`missed`/`failed`, `effectiveness`, `damaging_move`,
  `status_applied`/`status_cured`, `item_lost`/`item_gained`) + turn-level facts
  (`move_order`, `we_moved_first`, `both_attacked`, `someone_fainted`,
  `damage_on(species, side=…)`). This is what `TurnDelta` and the reward manager will
  read once step 5 lands, replacing the diff-based heuristics.
- **`LiveView` / `LiveSide` / `LivePokemon`** (`live_view.py`) — the **current-board** read
  surface ("what is true now"), built via `battle.live_view()`. An immutable snapshot of
  HP, status, boosts, revealed moves/item/ability, volatiles, hazards, weather, team
  sizes/reveal counts — holding **only primitives, no past-turn state** and no reference
  back to poke-env's `Pokemon`. A consumer literally cannot reach `last_move` through it,
  so current-state and history come from two disjoint, separately-fuzzed sources that
  can't drift. Opponent fields are reveal-gated (unknown item → `None`, only revealed
  moves listed; `ability` is `None` unless disclosed or uniquely inferable from species).
- **Injection seam:** `poke_env.player.Player.__init__` takes `battle_class=Battle`
  (default); our players pass `battle_class=Gen3Battle`. This is the only edit to the
  poke-env core besides the (unchanged) parser.

**Verification:** unit tests in `src/agents/battle/*_test.py` (schema, registry audit,
scripted parse + state-equivalence, TurnView fold). The spine is
`event_log_fuzz_e2e_test.py` — real `gen3ou` battles where both players run `Gen3Battle`;
it independently re-derives each turn from the intercepted raw protocol and asserts the
event log matches, plus conservation + event-kind coverage.

---

## Current State (End of Step 1)

Step 1 is complete. The pipeline is stable and all known correctness issues are resolved.

**What was hardened in Step 1:**

- `BattleContext` / `TurnDelta` / `SlotRegistry` — clean per-turn state, no battle object mutation
- `EpisodeTracker` — owns all per-episode mutable state; extension point for turn history
- `StallConfig` / `StallLogger` — stall detection extracted, shared by env and inference
- poke-env `env.py` — three lifecycle bugs fixed: stale battle queue, forfeit popup, stale `_choose_move()` hang
- Action mask — struggle double-enabling fixed; mask correctness audited and tested
- `RLPlayer` / `StatTrackingRLPlayer` — exception-safe `choose_move()`, stall detection on inference

**Ready for Step 2: Turn History + Action Mask as Features**

- `EpisodeTracker._history` holds the full episode; cap to `deque(maxlen=N)` for N-frame history
- Each `BattleContext` already carries `obs` and `mask` — no new fields needed
- Extend `Gen3ObservationEncoder` to accept `history: list[BattleContext]` alongside the current frame
