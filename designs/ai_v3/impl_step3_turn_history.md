# Implementation: Step 3 — N-Turn History with Attention

## Motivation

Replay analysis at ~99M steps found that ~30% of battles contain back-and-forth switching
(A→B→A→B within 3–4 turns). The bouncing tax (`-0.15`) is 3× smaller than the switch base
bonus (`+0.5`), so round-tripping still nets positive reward. Reward tweaks alone cannot fully
fix this because the model has no memory of what it just did — the observation was a
single-frame snapshot with only the immediately previous turn's delta.

Adding N-turn history gives the model temporal context: "I switched to this Pokémon last turn,
and the turn before that I switched away from it — I should commit."

The same context also helps with:
- Detecting stall spirals (same HP delta pattern for 3+ turns)
- Recognising momentum (we've KO'd two mons in a row — press the attack)
- Move-order trends (we've been outsped 3 turns running — speed matters here)

---

## Design Decisions

**What to encode:** The last N encoded TurnDeltas (39 raw dims each). TurnDelta captures moves
used, HP changes, switches, effectiveness, move order, and `our_cant_reason` /
`opp_cant_reason` (par/slp/frz/flinch/confusion — why each side failed to act). Status
conditions themselves are already fully visible in the current-frame base observation
(per-Pokémon condition block), so the history adds the *what-led-here* signal. Burn/toxic chip
appears as recurring negative `opp_hp_delta` across turns. This is sufficient for the
bounce/momentum use cases.

**N = 5:** Covers the A→B→A→B bounce in 3 turns, with two extra turns of context for momentum
signals. Adds 4 × 39 = 156 new raw dims to obs (replacing the single-TurnDelta slot). Large
enough to detect multi-turn patterns; small enough not to dominate the observation.

**Sequence layout:** Oldest turn at index 0, most recent at index N−1. Zero-padded at episode
start. Index N−1 carries the same data as the old single TurnDelta slot — no signal is lost.

**Network:** Embed all N raw deltas identically (reusing existing move/type embedding tables →
N × 99 dims), add learned positional embeddings, apply one Multi-Head Attention (MHA)
self-attention block — the same mechanism used by the existing Pressure/Safety/Synergy/Threat
paths over the Pokémon team tokens — and use the last position's refined output (99 dims) as
the attended representation. This *replaces* the single TurnDelta slot in the projection.
Projection size is auto-discovered via a dummy forward, so no magic constant needs updating.

**Multi-Head Attention (MHA):** MHA lets each turn attend to all others. The last position
(most recent turn) aggregates context from the full window via key/value matrices, producing a
single 99-dim vector that is both history-informed and most-recent-aligned. num_heads=3
(99/3=33 per head) to satisfy PyTorch's divisibility requirement.

**Why last-position output (not mean pool):** The most recent turn is the most
decision-relevant. Attention lets earlier turns inform it; the last query token aggregates
context from the full window.

---

## Observation Vector Change

Replace the single 39-dim TurnDelta block with a N × 39 history block:

| Block | Old dims | New dims | Offset |
|---|---|---|---|
| Base encoder | 1103 | 1103 | 0 |
| Prev-turn action mask | 11 | 11 | 1103 |
| **TurnDelta history (N=5)** | 39 | **195** | 1114 |
| **Total** | **1153** | **1309** | — |

Layout within history block (195 dims, offset 1114):
- `[0:39]`    — oldest available delta (zero-padded if episode < 5 turns old)
- `[39:78]`   — four turns ago
- `[78:117]`  — three turns ago
- `[117:156]` — previous turn
- `[156:195]` — current turn (same data as old single TurnDelta)

`get_layout()` gains:
```python
"turn_history_offset": 1114,
"turn_history_dim": 195,      # N * TURN_DELTA_DIM
"n_history_turns": 5,
```

---

## EpisodeTracker Changes

Added `_actions: list[int]` parallel to `_history` — `_actions[i]` is the action taken *from*
`_history[i]`. `record()` commits `_last_action` before appending the new context.

Added method:
```python
def prev_N_delta_vecs(self, n: int, encoder: TurnDeltaEncoder) -> np.ndarray:
    """Return (n, 39) array of encoded TurnDeltas, oldest-first, zero-padded."""
    result = np.zeros((n, encoder.dimension), dtype=np.float32)
    available = min(n, len(self._history) - 1, len(self._actions))
    for i in range(available):
        action = self._actions[-1 - i]
        ctx_prev = self._history[-2 - i]
        ctx_curr = self._history[-1 - i]
        delta = TurnDelta.build(ctx_prev, ctx_curr, action)
        result[n - 1 - i] = encoder.encode(delta)
    return result
```

---

## features_extractor.py Changes

Added constants: `N_HISTORY_TURNS = 5`.

Added in `__init__`:
```python
self._td_embed_dim = 2*move_emb + 2*type_emb + 35  # = 99
self.turn_history_pos_emb = nn.Embedding(N_HISTORY_TURNS, self._td_embed_dim)
self.turn_history_attn = nn.MultiheadAttention(embed_dim=99, num_heads=3, batch_first=True)
self.turn_history_norm = nn.LayerNorm(99)
```

N-turn embedding and attention in `forward_internal`:
```python
history_slots = turn_history_raw.view(batch_size, N_HISTORY_TURNS, TURN_DELTA_DIM)
embedded_slots = torch.stack([embed_delta_slot(history_slots[:, t, :])
                               for t in range(N_HISTORY_TURNS)], dim=1)  # [B, N, 99]
positions = torch.arange(N_HISTORY_TURNS, device=x.device)
embedded_slots = embedded_slots + self.turn_history_pos_emb(positions)
attn_out, _ = self.turn_history_attn(embedded_slots, embedded_slots, embedded_slots)
attended = self.turn_history_norm(embedded_slots + attn_out)
turn_delta_embedded = attended[:, -1, :]  # [B, 99]
```

---

## Inference (player.py)

`Gen3Player` now maintains a per-battle `EpisodeTracker` (keyed by `battle_tag`) and calls
`record()` + `prev_N_delta_vecs()` identically to the training env. `_battle_finished_callback`
cleans up both trackers. Eval win rates reflect the model's actual capability with full history.

---

## Model Versioning

Added `n_history_turns: int` field to `ModelVersion`. Bumped `MODEL_CONFIG_VERSION` to 2.
Migration: old models (N=1 single TurnDelta) default to `n_history_turns=1`. Field added to
`_WEIGHT_FIELDS` — positional embedding and attention weights are indexed by N.

---

## Files Modified

| File | Change |
|---|---|
| `src/agents/model/features_extractor.py` | `N_HISTORY_TURNS=5`; pos_emb/attn/norm modules; N-turn embedding+attention block |
| `src/agents/training/episode_tracker.py` | `_actions` list; `prev_N_delta_vecs(n, encoder)` |
| `src/agents/training/gen3_env.py` | Replace single delta with `history_vecs.flatten()` |
| `src/agents/observation/state_encoder.py` | `get_layout()` exposes turn history keys; `dimension` = 1309 |
| `src/agents/model/model_version.py` | `n_history_turns` field; version bump; migration |
| `src/agents/inference/player.py` | Per-battle tracker; full history in `embed_battle()` |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM = 1309` |
