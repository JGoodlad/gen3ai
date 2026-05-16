# Implementation: Step 2 — Action Mask as Observation Feature

This step adds the previous turn's action mask to the observation vector and routes each mask
component to the network submodule where it is semantically meaningful.

## Motivation

MaskablePPO already uses the *current* turn's action mask to zero out illegal logits at
sampling time. That prevents the model from choosing illegal actions, but it provides no signal
during the forward pass itself — the policy and value networks still see a flat observation with
no information about what was available *last turn*.

Adding the previous turn's mask as an input feature gives the model:

- **Move availability history** — was this move disabled last turn (Choice lock, PP depletion)?
  That predicts whether it will still be available now.
- **Switch availability context** — which bench slots were healthy enough to switch to last turn?
  Useful for tracking passive damage accumulation.
- **Struggle context** — was the active mon in Struggle last turn? Strongly predicts it needs to
  come out.

The *previous* turn's mask (not the current one) is used because the current mask is already
handled by MaskablePPO's logit masking; adding it redundantly as an input would provide no new
gradient signal. The previous turn's mask is genuinely new information.

---

## Observation Vector Change

`Gen3ObservationEncoder` now distinguishes two dimensions:

| Property | Value | Meaning |
|---|---|---|
| `base_dimension` | 1021 | Raw encoder output from `encode()` |
| `dimension` | 1032 | Full obs fed to the network: base + 11-dim prev_mask |

The 11-dim prev_mask layout mirrors the action space:

| Indices | Meaning |
|---|---|
| 0–5 | Switch validity (slots 0-5 — switch to bench mon i) |
| 6–9 | Move validity (move slots 0-3) |
| 10 | Struggle validity |

### Where prev_mask comes from

**Training** (`Gen3Env.embed_battle()`):
```python
def embed_battle(self, battle):
    obs = self.observation_encoder.encode(battle)
    if battle is self.battle1 and not battle.finished:
        mask = Gen3ActionMasker.get_mask(battle).astype(np.int8)
        if mask.sum() > 0:
            self._tracker.record(battle, mask, obs)
    prev_mask = self._tracker.prev_mask if battle is self.battle1 else np.ones(11, dtype=np.float32)
    return np.concatenate([obs, prev_mask])
```

`EpisodeTracker.prev_mask` returns `history[-2].mask` if at least two turns have been recorded,
otherwise all-ones (turn 1 has no prior turn; all-ones is a neutral prior).

**Inference** (`Gen3Player.embed_battle()`):
```python
result = self.observation_encoder.get_observation(battle)
result["observation"] = np.concatenate([result["observation"], np.ones(11, dtype=np.float32)])
return result
```

Inference players have no episode history, so all-ones is the correct fallback.

---

## Network Changes

### Routing Philosophy

Each mask bit routes to the network component that most directly uses that information:

| Mask component | Routed to | Reason |
|---|---|---|
| `move_mask[0:4]` | Move processor input | Validity modulates how useful that move's features are |
| `switch_mask[0:6]` | Role encoder input (per-slot) | Validity modulates whether that mon is a live switch option |
| `struggle_mask` | Role encoder input (all slots) | "Struggle last turn" is a global context that affects switch decisions |

### Move Processor (`move_input_dim`: 55 → 56)

Each of the 4 move slots gets 1 validity bit from `prev_mask[6:10]`.

For our team (6 Pokémon): the move validity comes from the previous turn's mask.  
For the opponent's team (6 Pokémon): all-ones (we have no knowledge of their PP state).

```python
move_validity_ours = move_mask.unsqueeze(1).unsqueeze(3).expand(-1, TEAM_SIZE, -1, -1)  # [B, 6, 4, 1]
move_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 4, 1, device=x.device)           # [B, 6, 4, 1]
move_validity      = torch.cat([move_validity_ours, move_validity_opp], dim=1)          # [B, 12, 4, 1]
```

### Role Encoder (`role_input_dim`: 238 → 240)

Two additional dims appended per Pokémon token before the role encoder:

1. **`switch_valid` (1 dim)** — was this specific slot a valid switch target last turn?  
   Our team: from `prev_mask[0:6]`. Opponent team: all-ones.

2. **`struggle_from_prev` (1 dim)** — was struggle the only option last turn?  
   Broadcast identically to all 12 tokens (it's a global signal about the active mon).

```python
switch_validity_ours = switch_mask.unsqueeze(2)                                        # [B, 6, 1]
switch_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 1, device=x.device)          # [B, 6, 1]
switch_validity      = torch.cat([switch_validity_ours, switch_validity_opp], dim=1)  # [B, 12, 1]

struggle_from_prev   = struggle_mask.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1)       # [B, 12, 1]

pokemon_enriched_with_context = torch.cat([
    pokemon_enriched, context_broadcasted, switch_validity, struggle_from_prev
], dim=2)  # [B, 12, 240]
```

Note: `struggle_from_prev` is distinct from the `is_forced_struggle` flag already in the
reactive observation block. The reactive flag reflects the *current* PP state; `struggle_from_prev`
reflects what the mask showed *last turn*. Both are useful and non-redundant.

### `remaining_part` slice fix

`forward_internal()` previously sliced `remaining_part = x[:, ctx['start']:]`, which
unintentionally included the 11-dim prev_mask tail when obs was 1032 dims. This would have caused
the final aggregation (`torch.cat([..., remaining_part], dim=1)`) to feed raw mask bits into the
projection layer without any routing logic.

Fixed to: `remaining_part = x[:, ctx['start']:base_dim]` — stops before the prev_mask tail.

### Trace logging fix

`forward()` strips the prev_mask tail before passing to `describe_vector()`, which expects the
base 1021-dim encoder output:

```python
x_base = x[:, :self.layout['base_dim']]
self._print_deep_trace(x_base, pokemon_part, species_ids)
```

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/observation/state_encoder.py` | Added `base_dimension` property (1021); `dimension` now 1032; `encode()` allocates only `base_dimension`; `get_layout()` exposes `base_dim` and `prev_mask_dim: 11` |
| `src/agents/training/episode_tracker.py` | Added `prev_mask` property (returns `history[-2].mask` or all-ones) |
| `src/agents/training/gen3_env.py` | `embed_battle()` appends `_tracker.prev_mask` |
| `src/agents/inference/player.py` | `embed_battle()` appends all-ones prev_mask for inference players |
| `src/agents/model/features_extractor.py` | `move_input_dim` 55→56; `role_input_dim` 238→240; prev_mask slicing and routing in `forward_internal()`; `remaining_part` slice fixed; trace strips mask tail |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` 1021→1032; added `EXPECTED_BASE_DIM = 1021`; `encode()` tested against `base_dimension` |

---

## Tests

101 unit tests pass. The `test_encoder_and_features_extractor_are_compatible` integration test
exercises the full encoder → features_extractor forward pass with the new 1032-dim observation,
catching any obs-space/architecture mismatch before it surfaces at checkpoint load.

Smoke test (`--debug --steps 10000`) passes end-to-end: episodes complete, replay callback fires,
evaluation runs, matchup matrix prints correctly.

---

## Final State

The observation now carries 11 dims of prev_mask alongside the 1021 base features. Each mask
component flows to the most semantically relevant network component rather than entering as raw
features at the projection layer.

**Ready for Step 3: Turn History**

- `EpisodeTracker._history` is a plain list — cap to `deque(maxlen=N)` for N-frame history
- Each `BattleContext` carries `obs` and `mask` — no new fields needed
- `Gen3ObservationEncoder` would gain an `encode_with_history(history: list[BattleContext])` path
- The 11-dim prev_mask slot in the observation could expand to N × 11 if turn history is
  concatenated rather than encoded via an RNN or attention layer
