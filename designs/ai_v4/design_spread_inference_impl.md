# Implementation: Step 1 — Spread Inference

Add IV/EV/nature/HP-type signals to the PPO observation and training loop so the backbone
produces role tokens that encode opponent spread information. See full design rationale in
`designs/ai_v4/design_spread_inference.md`.

**Entry state:** `POKEMON_VECTOR_DIM = 61`, `POKEMON_FULL_DIM = 62`, obs dim = 1309.
No IV/EV data in obs. No per-slot battle accumulators. No auxiliary losses.

---

## Signal 1 — Raw IV/EV Encoding

### `src/agents/observation/constants.py`

```python
POKEMON_IV_OFFSET    = 61   # 6 dims: HP/Atk/Def/SpA/SpD/Spe, each /31
POKEMON_EV_OFFSET    = 67   # 6 dims: HP/Atk/Def/SpA/SpD/Spe, each /252
POKEMON_KNOWN_OFFSET = 73   # 1 dim: 1.0 if ivs/evs known, else 0.0
POKEMON_VECTOR_DIM   = 74   # was 61
POKEMON_FULL_DIM     = 75   # was 62  (active flag still appended by state_encoder)
```

All downstream offsets that derive from `POKEMON_FULL_DIM` update automatically
(`OFFSET_OPP_TEAM`, `OFFSET_CONTEXT`, etc.).

### `src/agents/observation/pokemon.py` — `PokemonEncoder.encode()`

Append after the existing 61 dims (before returning `vec`):

```python
ivs = mon.ivs   # list[int] | None — order: HP, Atk, Def, SpA, SpD, Spe
evs = mon.evs   # list[int] | None — only populated for own team by poke-env
if ivs is not None:
    vec[POKEMON_IV_OFFSET    : POKEMON_EV_OFFSET]    = [v / 31.0  for v in ivs]
    vec[POKEMON_EV_OFFSET    : POKEMON_KNOWN_OFFSET] = [v / 252.0 for v in evs]
    vec[POKEMON_KNOWN_OFFSET] = 1.0
# Opponent slots: np.zeros init already leaves dims 61-73 as 0.0
```

Update `get_layout()` to document the three new entries.

---

## Signal 2 — Per-Slot Accumulated Battle Statistics

### `src/agents/observation/constants.py` (continued)

```python
POKEMON_ACC_OFFSET   = 74   # 8 scalar accumulators (see below)
POKEMON_HPTYPE_OFFSET = 82  # 1 dim: raw HP type index (0 = unknown, 1-17 = type)
POKEMON_VECTOR_DIM   = 83   # was 74 after Signal 1
POKEMON_FULL_DIM     = 84   # was 75 after Signal 1
```

### `src/agents/training/battle_context.py` — `BattleContext`

Add two new `np.ndarray` fields shaped `(6,)`, one per side:

```python
our_acc: np.ndarray   # (6, 9) float32 — per-slot accumulators for our team
opp_acc: np.ndarray   # (6, 9) float32 — per-slot accumulators for opp team
```

Accumulator column layout (index within the 9-wide array):

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `cumulative_dmg_taken` | Total HP% lost this battle |
| 1 | `n_physical_hits` | Physical moves that connected |
| 2 | `n_special_hits` | Special moves that connected |
| 3 | `min_hp_pct` | Lowest HP% ever observed |
| 4 | `cumulative_healed` | Total passive HP% recovery (Leftovers etc.) |
| 5 | `n_turns_active` | Turns this slot has been active |
| 6 | `speed_win_frac` | Fraction of non-priority turns this slot moved first |
| 7 | `speed_loss_frac` | Fraction of non-priority turns this slot moved second |
| 8 | `hp_type` | Hidden Power type index (0 = unknown, 1–17 = type) |

`BattleContext.from_battle()` does not compute these — they are long-lived state
maintained by `EpisodeTracker` and passed in at construction time.

### `EpisodeTracker` (wherever it lives — `gen3_env.py` or `episode_tracker.py`)

Add `_our_acc` and `_opp_acc` arrays initialised to zeros on episode start. Update
each turn after the `TurnDelta` is built:

- **Damage taken**: use `delta.our_hp_delta` / `delta.opp_hp_delta` (per-slot arrays)
  to increment `cumulative_dmg_taken`; update `min_hp_pct` from current HP arrays.
- **Hit counts**: use `delta.our_move_category` / `delta.opp_move_category` to
  increment physical/special counters for the active slot.
- **Healing**: detect positive HP delta on a turn where the mon didn't switch in.
- **Speed**: use `delta.we_moved_first`; increment the appropriate slot's win/loss frac
  numerator; normalise at encode time as `wins / max(1, wins+losses)`.
- **HP type** (own team): set once at episode start from known IVs using the formula:
  ```
  idx = floor(15 * (iv_hp%2 + 2*iv_atk%2 + 4*iv_def%2 +
                     8*iv_spe%2 + 16*iv_spa%2 + 32*iv_spd%2) / 63) + 1
  ```
- **HP type** (opponents): after each TurnDelta where `opp_move == "hiddenpower"` and
  `our_effectiveness` is known, call `infer_hp_type(our_active_types, effectiveness)`.
  If unambiguous, store in `_opp_acc[slot, 8]`.

### `src/agents/observation/pokemon.py` — `PokemonEncoder.encode()`

After Signal 1 block, append the accumulator slice and HP type index:

```python
# acc is the (9,) row for this slot, passed in alongside mon
vec[POKEMON_ACC_OFFSET : POKEMON_HPTYPE_OFFSET] = acc[:8]   # 8 scalar accumulators
vec[POKEMON_HPTYPE_OFFSET] = acc[8]                          # hp_type raw index
```

`PokemonEncoder` needs to accept `acc: np.ndarray` as an additional argument (or the
full acc array + slot index). Update all call sites in `state_encoder.py`.

### `src/agents/model/features_extractor.py`

The `hp_type` index (0–17) is embedded via the existing shared `type_embedding` table.
Add a lookup immediately before the role encoder MLP:

```python
hp_type_idx = obs_block[..., POKEMON_HPTYPE_OFFSET].long()
hp_type_emb = self.type_embedding(hp_type_idx)   # (..., 16)
```

Concatenate `hp_type_emb` into the role encoder input alongside the other embedding
outputs. Role encoder input dim grows by 16 (type_embedding_dim).

> **Note:** `role_input_dim` is computed dynamically via dummy forward pass — no constant
> to update manually.

---

## Signal 3 — Auxiliary Spread Prediction Losses

### `src/agents/model/features_extractor.py` — refactor

Extract the role-token computation into a private helper:

```python
def _compute_role_tokens(self, obs: dict[str, Tensor]) -> Tensor:
    """Returns (B, 12, 128) role tokens after all 5 attention paths."""
    ...  # everything currently between obs parsing and the final projection

def forward(self, obs):
    role_tokens = self._compute_role_tokens(obs)
    ...  # projection, LayerNorm, ReLU — unchanged

def extract_role_tokens(self, obs: dict[str, Tensor]) -> Tensor:
    """Public hook for the aux pass. Returns (B, 12, 128)."""
    return self._compute_role_tokens(obs)
```

### `src/agents/training/gen3_mpp.py` (new file)

```python
class Gen3MaskablePPO(MaskablePPO):
    """Drop-in replacement for MaskablePPO with optional aux spread pass."""

    def __init__(self, *args, aux_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef
        self._aux_heads: SpreadAuxHeads | None = None
        self._aux_optimizer: torch.optim.Optimizer | None = None

    def attach_aux_heads(self, heads: "SpreadAuxHeads", lr: float = 3e-4) -> None:
        self._aux_heads = heads.to(self.device)
        self._aux_optimizer = torch.optim.AdamW(
            list(self.policy.features_extractor.parameters())
            + list(heads.parameters()),
            lr=lr,
        )

    def train(self) -> None:
        super().train()                          # SB3 PPO update — untouched
        if self._aux_heads and self.aux_coef > 0.0:
            self._run_aux_pass()

    def _run_aux_pass(self) -> None:
        self.policy.set_training_mode(True)
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            role_tokens = self.policy.features_extractor.extract_role_tokens(
                rollout_data.observations
            )
            loss = self.aux_coef * self._aux_heads.compute_loss(
                role_tokens, rollout_data.observations
            )
            self._aux_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._aux_optimizer.param_groups[0]["params"],
                self.max_grad_norm,
            )
            self._aux_optimizer.step()
```

### `src/agents/training/aux_spread_heads.py` (new file)

```python
class SpreadAuxHeads(nn.Module):
    """
    Lightweight heads that supervise the backbone's per-slot role tokens.

    Heads:
      ev_tier  — Linear(128 → 3) per stat × 6 own-team slots
                 classes: 0 EVs | 1-124 EVs | 125-252 EVs
                 ground truth: read from obs dims POKEMON_EV_OFFSET..POKEMON_KNOWN_OFFSET
      nature   — Linear(128 → 25) for each own-team slot
                 ground truth: passed via info["our_natures"] side-buffer
      archetype — Linear(128 → K) for each opp slot
                 ground truth: info["opp_archetypes"] side-buffer (self-play only)
    """
```

`compute_loss(role_tokens, obs)` reads EV ground truth directly from the obs tensor
(dims `POKEMON_EV_OFFSET:POKEMON_KNOWN_OFFSET` for own-team slots 0–5) and discretises
in-place. Nature and archetype labels are injected by the side-buffer (see below).

**Loss weights (defaults):**

| Head | `aux_coef` multiplier | When active |
|------|----------------------|------------|
| EV tier | 0.05 | always |
| Nature | 0.02 | always |
| Spread archetype (opp) | 0.02 | `--team-log` active |

### `src/agents/training/spread_label_buffer.py` (new file)

A thin callback that maintains a ring buffer of nature and spread-archetype labels
synced to `rollout_buffer.pos`. Reads `info["our_natures"]` and `info["opp_archetypes"]`
from `Gen3Env.reset()` / `step()` info dicts. Exposes `get(pos)` so `SpreadAuxHeads`
can look up labels by rollout position.

### `src/main/train_rl_agent.py`

- Replace `MaskablePPO(...)` with `Gen3MaskablePPO(..., aux_coef=args.aux_coef)`
- After model construction: `model.attach_aux_heads(SpreadAuxHeads(...), lr=3e-4)`
- Wire `SpreadLabelBuffer` callback alongside existing callbacks
- Add `--aux-coef` CLI flag (default `0.0` — off by default, enable explicitly)

### `data/pokemon/gen3_spread_archetypes.json` (offline script)

Run once on a populated `teams.jsonl` to cluster (species, EV spread) pairs by species
using k-means (k ≤ 8) on the 6-stat vector from `compute_raw_stats()`. Schema:

```json
{
  "salamence": [
    {"label": 0, "name": "offense", "evs": [4, 252, 0, 0, 0, 252], "nature": "jolly"},
    {"label": 1, "name": "mixed",   "evs": [4, 100, 0, 200, 0, 200], "nature": "naive"}
  ],
  ...
}
```

---

## Dimension Summary

| Constant | Before | After |
|----------|--------|-------|
| `POKEMON_VECTOR_DIM` | 61 | 83 |
| `POKEMON_FULL_DIM` | 62 | 84 |
| `OFFSET_OPP_TEAM` | 372 | 504 |
| `OFFSET_CONTEXT` | 744 | 1008 |
| `base_dimension` | 1103 | 1367 |
| `dimension` (full obs) | 1309 | 1573 |

`model_version.py` catches the obs dim change via `total_dim` — old checkpoints are
correctly rejected. Use `--transfer-from` if you want to load weights despite the shape
change (embedding tables and attention weights are unaffected).

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/observation/constants.py` | New offsets; `POKEMON_VECTOR_DIM` 61→83, `POKEMON_FULL_DIM` 62→84 |
| `src/agents/observation/pokemon.py` | Append IV/EV/flag + accumulator + hp_type index; update `get_layout()` |
| `src/agents/training/battle_context.py` | Add `our_acc`, `opp_acc` fields `(6,9) float32` |
| `EpisodeTracker` | Init acc arrays on reset; update per-turn; pass to `BattleContext` |
| `src/agents/model/features_extractor.py` | Extract `_compute_role_tokens()`; add `extract_role_tokens()`; hp_type embedding lookup |
| `src/agents/training/gen3_mpp.py` | **New**: `Gen3MaskablePPO` |
| `src/agents/training/aux_spread_heads.py` | **New**: `SpreadAuxHeads` |
| `src/agents/training/spread_label_buffer.py` | **New**: nature/archetype side-buffer callback |
| `src/main/train_rl_agent.py` | Swap to `Gen3MaskablePPO`; add `--aux-coef`; wire callback |
| `data/pokemon/gen3_spread_archetypes.json` | **New**: offline archetype clusters |
| `CLAUDE.md` | Update per-Pokémon dims table (61→83) and obs offset table |

---

## Verification

1. **Smoke test** — `train_rl_agent.py --debug --steps 10000`: `[ModelVersion]
   Round-trip smoke test PASSED`; no shape errors.

2. **IV/EV dims** — unit test: own-team slot has correct normalised IV/EV values and
   `hidden_stats_known = 1.0`; opponent slot has zeros for dims 61–73 and flag = 0.0.

3. **Accumulator sanity** — `--debug --steps 5000`: a Pokémon that took 3 special hits
   has `n_special_hits = 3` and `cumulative_dmg_taken` equals the sum of the HP%
   drops on those turns.

4. **HP type** — integration test with a known team (opponent Zapdos with HP Ice):
   after first Hidden Power use against a Grass-type active, `opp_acc[slot, 8]` = 11
   (ice type index).

5. **Aux loss** — `--aux-coef 0.05 --steps 200K`: TensorBoard `aux/ev_tier_loss` falls
   from ≈1.1 toward ≈0.3 within 50K steps (EVs are in the obs; head just learns to
   read through the role token).

6. **Win-rate stability** — aux loss must not degrade win rate vs. random by >2% at
   500K steps vs. the no-aux baseline. If it does, halve `aux_coef`.

---

## Status

- [ ] Signal 1: IV/EV dims in `constants.py` + `pokemon.py`
- [ ] Signal 2: `BattleContext` acc fields + `EpisodeTracker` update logic
- [ ] Signal 2: `PokemonEncoder` acc + hp_type encoding; `features_extractor` hp_type embedding
- [ ] Signal 3: `_compute_role_tokens` refactor + `extract_role_tokens`
- [ ] Signal 3: `Gen3MaskablePPO` + `SpreadAuxHeads` + `SpreadLabelBuffer`
- [ ] Signal 3: `train_rl_agent.py` wiring + `--aux-coef` flag
- [ ] Archetype clustering script + `gen3_spread_archetypes.json`
- [ ] CLAUDE.md updated
- [ ] Verification passing
