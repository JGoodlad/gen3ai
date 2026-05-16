# Implementation: Step 3 — Architecture Improvements Batch

This batch completes the architectural improvements identified in the
`network_review_2.md` review pass, fixes a training-infrastructure
reliability gap, and adds a reward-shaping signal observed during
early evaluation runs.

---

## 1. Network Architecture (N1–N6)

**`src/agents/model/features_extractor.py`**

### N1 — Attention Pool (permutation-equivariant team aggregation)

Replaced the slot-order team flatten with a **learned attention pool** per side.
One learned query vector (128-dim) attends over the 6 role tokens for each
side; fainted slots are key-masked so they don't contribute. The output is a
single 128-dim pooled token per side fed into the projection layer.

Before: `our_team_flat (6×128=768)` + `their_team_flat (768)` → order-sensitive, 1536 dims  
After: `our_pool (128)` + `their_pool (128)` → order-invariant, 256 dims

The team representation is now permutation-equivariant: swapping two Pokémon
in the team doesn't change the pool output (modulo attention weight reorder).
The large dim reduction also eases the projection layer's compression task.

### N2 — TurnDelta move/type IDs through shared embedding tables

The `TurnDeltaEncoder` previously normalized move and type IDs as scalars
(`id / max_id`). These are now stored as **raw ints** (float32, exact for
values < 2²⁴). The `Gen3FeaturesExtractor` routes them through the shared
`move_embedding` and `type_embedding` tables, producing a 89-dim embedded
block fed to the projection layer (replacing the 10-dim raw-scalar block).

This lets the model re-use learned move and type representations built from
the rest of the observation — the TurnDelta move embedding is identical to
the embedding used in the active move slots, so gradients are shared.

Projection input change: 502 → 562 (due to N1 dim reduction + N2 expansion).

### N3 — Opponent synergy attention (Path ⑤)

Added a 5th attention path: `opp_synergy_attn` — opponent bench attends to
itself, symmetric to the existing `synergy_attn` for our side. This lets the
network model opponent team cohesion (e.g. recognising a sand team or a
balanced core), not just individual threats.

### N4 — Within-Pokémon move self-attention

Added a `MultiheadAttention(32, 2 heads)` block with LayerNorm residual that
runs over the **4 move slots of each individual Pokémon** before those slots
are flattened into the role encoder input. This lets the role encoder see
"this mon has two physical moves and a recovery move" as a composed signal
rather than four independent scalars.

### N5 — Fainted-slot output masking for Safety and Threat paths

Fainted-slot output masking was already applied to the Synergy path. Extended
to also zero out Safety path outputs for our fainted slots and Threat / Opp
Synergy path outputs for opponent fainted slots. Prevents gradient flow
through dead-slot attention outputs.

### N6 — Pre-projection LayerNorm

Added a `LayerNorm` over the full projection input vector before the final
`Linear → ReLU`. The projection input combines several heterogeneous blocks:
embedding outputs (~±1), binary flags (0/1), HP fractions (0–1), and
TurnDelta HP deltas (±1). LayerNorm equalises scale differences so the
projection weights don't need to learn compensatory scale factors.

---

## 2. Observation: `species_known` Flag (S5)

**`src/agents/observation/pokemon.py`**, **`src/agents/observation/constants.py`**

Added a 1-dim `species_known` flag as the last field of each per-Pokémon slot:
- `1.0` for all populated slots (own team and revealed opponent Pokémon)
- `0.0` for unseen opponent slots

This lets the model cleanly distinguish "this slot is empty / not yet revealed"
from "this Pokémon is alive and known" vs "this Pokémon fainted". Previously
the model had to infer unknown slots from all-zeros species ID, which was
ambiguous with Pokémon that have a low assigned ID.

**Dimension changes:**
- `POKEMON_VECTOR_DIM`: 57 → 58
- `POKEMON_FULL_DIM`: 58 → 59
- Base obs dim: 1053 → 1065
- Total obs dim (including prev-mask and TurnDelta blocks): **1093 → 1105**

---

## 3. Training Infrastructure: Subprocess Watchdog

**`src/agents/training/watchdog.py`** (new)  
**`src/main/train_rl_agent.py`** (updated)

`SubprocVecEnv` workers that crash (e.g. on a `TimeoutError` during `reset()`)
leave the main training process hanging forever on a pipe `recv()`. Added a
daemon thread that polls all worker PIDs every second and calls `os._exit(1)`
the moment any exits with a non-zero code.

```python
def start_subprocess_watchdog(vec_env, label="env"):
    processes = getattr(vec_env, "processes", None)
    if not processes: return
    def _watch():
        while True:
            for p in processes:
                if not p.is_alive() and p.exitcode not in (0, None):
                    print(f"\n🛑 [{label}] Worker PID {p.pid} died (exitcode={p.exitcode}). Exiting.")
                    os._exit(1)
            time.sleep(1)
    threading.Thread(target=_watch, daemon=True).start()
```

The watchdog was subsequently extracted from `train_rl_agent.py` into a
standalone module at `src/agents/training/watchdog.py` to keep the training
script slim and make it reusable wherever `SubprocVecEnv` is used.

---

## 4. Trace Improvements: TurnDelta Human-Readable Display

**`src/agents/observation/turn_delta_encoder.py`**,
**`src/agents/model/features_extractor.py`**

`describe_vector()` now resolves the raw move/type IDs stored in the TurnDelta
block to human-readable strings:

Before: `our_move: pwr=95 sec=1 recoil=0`  
After: `our: surf [Water, 95bp, +eff]`

- Move num → move name (via `gen3_moves.json` reverse lookup)
- Type ID → type string (via the type list)
- HP deltas grouped on one line with faint indicators
- Unconfirmed opponent moves surfaced as `[?]`

---

## 5. Reward: Struggle Loop Penalty

**`src/agents/training/reward_manager.py`**

During early evaluation runs, Blissey vs Blissey mirror matches were
exhausting all PP and then looping in forced Struggle for 20–60 turns. The
existing stall tax (turn 200+) was irrelevant since PP depletion in a defensive
mirror occurs around turns 50–100.

Added a consecutive-struggle penalty:
- `-0.5` per turn once the agent has been stuck in Struggle for **3 or more
  consecutive turns** (`STRUGGLE_LOOP_THRESHOLD = 3`)
- Counter resets immediately on any non-struggle action — switching any
  Pokémon in is the intended escape; the model learns this instantly

Also added `struggle_turns` to per-episode logging:
```
🏁 Episode Finished | ... | Struggle:  0
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/model/features_extractor.py` | N1–N6: attention pool, move self-attn, opp synergy path, fainted masking, LayerNorm, TurnDelta embedding routing |
| `src/agents/observation/pokemon.py` | S5: `species_known` flag at `POKEMON_SPECIES_KNOWN_OFFSET` |
| `src/agents/observation/constants.py` | `POKEMON_VECTOR_DIM` 57→58, `POKEMON_FULL_DIM` 58→59, new offset constant |
| `src/agents/observation/turn_delta_encoder.py` | N2: raw move/type IDs instead of normalized scalars; trace display improvements |
| `src/agents/observation/pokemon_test.py` | Updated for new POKEMON_VECTOR_DIM |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` updated 1093 → 1105 |
| `src/agents/training/reward_manager.py` | Struggle loop constants, `_consecutive_struggle` / `struggle_turns` counters, penalty logic, logging |
| `src/agents/training/watchdog.py` | **New** — `start_subprocess_watchdog()` |
| `src/main/train_rl_agent.py` | Import and call `start_subprocess_watchdog`; watchdog logic removed (now in module) |
| `designs/ai_v3/README.md` | Full rewrite of Mermaid digraph and architecture tables |
| `designs/ai_v3/network_review_2.md` | Pruned retracted review items |
| `designs/ai_v3/todo.md` | Added Section 5 (turn-history memory: sliding window → GRU) |
| `CLAUDE.md` | Updated obs vector table and feature extractor architecture summary |

---

## What's Next

See `designs/ai_v3/todo.md`. Priority items:

1. **Status and stat-stage deltas in `TurnDelta`** — burn, paralysis,
   Calm Mind boosts not yet in reward or observation.
2. **Delegating move `last_move` gap** — Metronome/Nature Power/Assist
   produce `opp_move_known=False`; low priority for gen3ou sets.
3. **Turn-history memory** — sliding window of K TurnDelta blocks → GRU
   once the baseline is established.
