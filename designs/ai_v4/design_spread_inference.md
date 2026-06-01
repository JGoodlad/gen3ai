# Design: Spread Inference — IV/EV/Nature and Future Opponent Signals

> **Status: Signal 1 SHIPPED** (own-team IV/EV/nature encoding — see
> `impl_step1_spread_encoding.md`). **Signals 2–3 are FUTURE** (per-slot accumulated battle
> statistics; auxiliary spread-prediction losses) — not yet implemented. This doc is kept as
> the forward-looking design for those two.

Make the PPO model's per-slot representations encode enough information about each
Pokémon's spread that the team completion model (ai_v6 Steps 3–4) can predict EV tiers,
nature, speed tier, and Hidden Power type from frozen role tokens.

Three signals are planned. **Signal 1 is shipped** (`impl_step1_spread_encoding.md`). Signals 2
and 3 are deferred — likely implemented after the unified transformer lands and self-play is
running.

---

## Signal 1 — Raw IV/EV/Nature Encoding for Own Team ✅ DONE

*Implemented in `impl_step1_spread_encoding.md`. See that doc for exact constants,
offsets, and test coverage.*

### What shipped

An 18-dim spread block was added to every Pokémon slot, appended after the existing
61 dims:

| Field | Dims | Encoding | Own slots | Opp slots |
|-------|------|----------|-----------|-----------|
| IVs (HP, Atk, Def, SpA, SpD, Spe) | 6 | `/31 → [0, 1]` | actual from teambuilder | 0.0 |
| EVs (HP, Atk, Def, SpA, SpD, Spe) | 6 | `/252 → [0, 1]` | actual from teambuilder | 0.0 |
| `spread_known` | 1 | 1.0 own / 0.0 opp | 1.0 | 0.0 |
| Nature (Atk, Def, SpA, SpD, Spe) | 5 | raw floats 0.9/1.0/1.1 from `natures.json` | actual | 0.0 |

`POKEMON_VECTOR_DIM`: 61 → 79. `POKEMON_FULL_DIM`: 62 → 80. Obs dim: 1309 → 1525.
`ARCH_SIGNATURE`: `"gen3_spread_v1"`.

A poke-env bug was also fixed: `_update_from_teambuilder` previously dropped IVs/EVs/nature
for any Pokémon with all-zero EVs. The guard is removed — those fields are now stored
unconditionally.

### What this enables

The model has precise own-team spread data rather than relying on species embeddings to
implicitly encode typical spreads. The role encoder can interpret "our Blissey has 252 SpD
EVs" explicitly. Signal 1 also provides the ground truth that Signal 3's aux heads will
train against — the EV dims are already in the obs, no side-buffer needed for own team.

---

## Signal 2 — Per-Slot Accumulated Battle Statistics (TODO — deferred)

*Not yet implemented. Planned for after the unified transformer lands, since that
rewrite changes where and how per-slot information is encoded.*

### The gap

`TurnDelta.opp_hp_delta` tracks per-turn damage but there is no memory across turns.
When a bench Pokémon returns to the field, the model has no record of how much cumulative
damage it has taken, whether it has Leftovers recovery, or what its speed tier revealed
about its EVs.

### Planned: 9 new dims per slot, both sides

Running accumulators in `BattleContext` that persist across switch-outs:

| Field | Description |
|-------|-------------|
| `cumulative_dmg_taken` | Total HP% lost this battle |
| `n_physical_hits` | Physical moves that connected |
| `n_special_hits` | Special moves that connected |
| `min_hp_pct` | Lowest HP% ever observed |
| `cumulative_healed` | Total passive HP% recovery (Leftovers, Ingrain) |
| `n_turns_active` | Turns this slot has been active |
| `speed_win_frac` | Fraction of non-priority turns this slot moved first |
| `speed_loss_frac` | Fraction of non-priority turns this slot moved second |
| `hp_type` | Observed Hidden Power type index (0 = unknown, 1–17 = type) |

HP type for own team is computed once at episode start from known IVs using the Gen 3
formula. For opponents it is inferred when they use Hidden Power and the effectiveness
is observed.

`hp_type` would be embedded via the shared `type_embedding` table (16D, zero new
parameters); the other 8 fields are plain scalars.

**Dim impact when implemented**: `POKEMON_VECTOR_DIM` 79 → 97; `POKEMON_FULL_DIM` 80 → 98;
obs dim ~1525 → ~1741.

**Files that would change**: `constants.py`, `pokemon.py`, `battle_context.py`,
`EpisodeTracker`, `features_extractor.py` (hp_type embedding lookup).

---

## Signal 3 — Auxiliary Spread Prediction Losses (TODO — deferred)

*Not yet implemented. Depends on Signal 2 for accumulator data; also benefits from
self-play team logs being available for opponent archetype supervision.*

### Why the RL signal alone is insufficient

Even with Signals 1 and 2 in the inputs, PPO won't reliably encode spread structure in
the 128D role tokens unless the win-rate signal happens to correlate with it. Knowing
whether Salamence has 252 vs 200 Spe EVs rarely changes the optimal action directly.
Auxiliary supervised losses force the backbone to represent spread structure explicitly,
which pays off when role tokens are consumed by the team completion model.

### Planned approach

Subclass `MaskablePPO` with a thin override that runs a second backward pass after each
PPO update:

```python
class Gen3MaskablePPO(MaskablePPO):
    def train(self) -> None:
        super().train()          # SB3 PPO update — untouched
        if self._aux_heads and self.aux_coef > 0.0:
            self._run_aux_pass()
```

The rollout buffer is still populated after `train()` (SB3 resets it at the start of
the next `collect_rollouts()`), so the aux pass can iterate over it without changes to
SB3 internals.

### Planned heads

| Head | Architecture | Ground truth source |
|------|-------------|---------------------|
| EV tier (own team) | Linear(128→3) per stat × 6 slots | obs dims (already present via Signal 1) |
| Nature (own team) | Linear(128→25) per slot | `info["our_natures"]` side-buffer |
| Spread archetype (opp) | Linear(128→K) per slot | `info["opp_archetypes"]` from team log |

EV tier classes: `[0 EVs | 1–124 EVs | 125–252 EVs]`. Ground truth is read directly
from the EV dims already in the observation — no label pipeline needed for own team.

**Starting loss weights**: `aux_coef=0.05` on own-team heads. Enable opp archetype head
only when `--team-log` is active (self-play). If win rate vs. random drops >2% at 500K
steps, halve `aux_coef`.

**Files that would be created/changed**: `gen3_mpp.py` (new), `aux_spread_heads.py`
(new), `spread_label_buffer.py` (new), `train_rl_agent.py` (swap PPO class, add flag).

---

## Relationship to Unified Transformer

Signals 2 and 3 are both more valuable after the unified transformer ships. Under the
current architecture, role tokens are history-blind — the accumulator data from Signal 2
would be the only way the role encoder sees cross-turn patterns. Under the unified
transformer, role tokens already attend over turn history, so the accumulators become
a complementary compressed summary rather than the sole cross-turn signal.

Recommended sequence: **unified transformer → Signal 2 → Signal 3**.
