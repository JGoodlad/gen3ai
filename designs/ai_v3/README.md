# Gen3AI Network Architecture

Feature extractor used by MaskablePPO. Takes a 1093-dim observation and produces 512-dim features for the policy and value heads.

## Data Flow Digraph

```mermaid
flowchart TD
    OBS["Observation · 1093-dim float32"]

    OBS --> BASE["Base obs · 1053-dim"]
    OBS --> PM["prev_mask · 11-dim\n(previous turn's action mask)"]
    OBS --> TD["TurnDelta block · 29-dim\n(last turn's transition signal)"]

    PM --> SW["switch_mask [0:6]\nbench slot validity"]
    PM --> MV["move_mask [6:10]\nmove slot validity"]
    PM --> STR["struggle_mask [10]"]

    BASE --> TEAM["Pokémon vectors\n[B, 12, 58]  our + opp team"]
    BASE --> REM["remaining_part\nactive_ctx · global · reactive"]

    TEAM --> EMB["Embedding Lookups\nspecies · 32\nmove · 16\nitem · 16\nability · 16\ntype · 16  shared"]
    REM --> ACT_RAW["active_context · 22-dim × 2\nboosts · stat stages · volatiles"]
    REM --> GLOBAL["global env · 13-dim\nturn · weather·6 · fainted·2\nspikes·2 · reflect·2 · screen·2"]
    REM --> REACT["reactive scalars · 12-dim\npow·4 mult·4 fainted·2\nstatus·1 struggle·1\n(hp/spikes removed — in per-mon and global)"]
    REM --> MATCH["matchup matrix · 288-dim\n[B, 12, 4, 6] type effectiveness\n(used by move processor only)"]

    MV --> MPIN
    EMB --> MPIN["Move Processor input · [B, 12, 4, 58]\nmove_emb·16 + type_emb·16\nremnants·6 + known·1\ncontext·12 + matchups·6 + validity·1\nremnants: power secondary recoil category cur_pp max_pp"]
    MATCH --> MPIN
    GLOBAL --> MPIN

    MPIN --> MP["Shared Move Processor\nLinear 58→64 → ReLU\nLinear 64→32\n(all 12 mons × 4 slots)"]
    MP --> PMOV["Processed moves · [B, 12, 128]\n4 slots × 32-dim"]

    EMB --> PE["pokemon_enriched · [B, 12, 241]\nspecies·32 + stats·6 + item_emb·16\nitem_known·1 + pk_types·32\nability_emb·16 + ability_known·1\ncondition·7 + moves·128 + hp+active·2"]
    PMOV --> PE

    GLOBAL --> RIN
    SW --> RIN
    STR --> RIN
    PE --> RIN["Role Encoder input · [B, 12, 259]\nenriched·241 + ctx·16\nswitch_valid·1 + struggle_prev·1\n(ctx=16: turn·1+weather·6+fainted·2+spikes·2+screens·4+struggle·1)"]

    RIN --> RE["Shared Role Encoder\nLinear 259→256 → ReLU\nLinear 256→128\n(all 12 mons)"]
    RE --> RT["Role tokens · [B, 12, 128]\n+ status embedding bias\n(our active / our bench / their active / their bench)"]

    RT --> OT["our_team · [B, 6, 128]"]
    RT --> TT["their_team · [B, 6, 128]"]

    OT --> PA["① Pressure\nour_active ← their_team\nMultiheadAttention + LayerNorm residual"]
    TT --> PA
    PA --> OAP["our_active_post_pressure\n[B, 1, 128]\nwritten back into our_team"]

    OAP --> OT2["our_team updated\n(active slot = post-Pressure)"]
    OT --> OT2

    OT2 --> SA["② Safety\nour_team ← their_active\nMultiheadAttention + LayerNorm residual"]
    TT --> SA
    SA --> OT3["our_team · [B, 6, 128]"]

    OT3 --> SYA["③ Synergy\nour_team ← our_team\nMultiheadAttention + LayerNorm residual"]
    SYA --> OT4["our_team final · [B, 6, 128]"]

    OAP --> THA["④ Threat\ntheir_team ← our_active_post_pressure\nMultiheadAttention + LayerNorm residual"]
    TT --> THA
    THA --> TT2["their_team final · [B, 6, 128]"]

    OT4 --> AGG
    TT2 --> AGG
    OT4 --> OAR["our_active_refined · [B, 128]\nextracted from our_team final\nat our_active_idx"]
    OAR --> AGG

    ACT_RAW --> ACTENC["Active Context Encoder · shared\nLinear 22→64 → ReLU → Linear 64→32\nour side + opp side"]
    TD --> TDENC["TurnDelta Encoder\nLinear 29→64 → ReLU → Linear 64→32"]
    TDENC --> AGG
    ACTENC --> AGG["Aggregation\ncat(\n  our_team·768\n  their_team·768\n  our_active_refined·128\n  our_ctx_enc·32\n  opp_ctx_enc·32\n  global+scalars·29\n  turn_delta_enc·32\n)"]
    GLOBAL --> AGG
    REACT --> AGG

    AGG --> PROJ["Projection\nLinear N→512 → ReLU\n(N auto-discovered via dummy forward)"]
    PROJ --> OUT["Features · [B, 512]\n→ policy head + value head"]
```

## Dimension Reference

| Layer | Input dim | Output dim | Notes |
|---|---|---|---|
| Move Processor | 58 | 32 | shared; run 12×4 times per forward pass |
| Role Encoder | 259 | 128 | shared; run 12 times per forward pass |
| Active Ctx Encoder | 22 | 32 | shared; run twice (our side + opp side) |
| TurnDelta Encoder | 29 | 32 | encodes last-turn signal before projection |
| Pressure Attn | 128 (Q), 128 (KV) | 128 | our_active queries their_team |
| Safety Attn | 128 (Q), 128 (KV) | 128 | our_team queries their_active |
| Synergy Attn | 128 (Q/K/V) | 128 | our_team self-attention |
| Threat Attn | 128 (Q), 128 (KV) | 128 | their_team queries our_active_post_pressure |
| Projection | N (auto) | 512 | N determined by dummy forward in `__init__` |

## Observation Layout

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 58) | 348 | 0 |
| Opp team (6 × 58) | 348 | 348 |
| Active context ×2 (boosts, volatiles) | 44 | 696 |
| Global env | 13 | 740 |
| Reactive scalars + matchup matrix | 300 | 753 |
| **prev_mask** | **11** | **1053** |
| **TurnDelta block** | **29** | **1064** |
| **Total** | **1093** | |

The matchup matrix (288 of the 300 reactive dims) is consumed by the move processor only — it is **not** fed to the projection directly. The TurnDelta block is appended by `gen3_env.embed_battle()` and fed directly into the projection aggregation.

## TurnDelta Block Layout (29 dims, offset 1064)

| Field | Dims | Notes |
|---|---|---|
| our_move features | 5 | id_norm, power/200, has_secondary, has_recoil, type_id_norm |
| opp_move features | 5 | same |
| our_switched | 1 | bool |
| opp_switched | 1 | bool |
| our_failed_to_move | 1 | bool |
| opp_failed_to_move | 1 | bool |
| our_cant_reason | 5 | one-hot: par / slp / frz / flinch / confusion |
| opp_cant_reason | 5 | same |
| our_hp_delta | 1 | sum of our HP delta vector (negative = damage taken) |
| opp_hp_delta | 1 | sum of opp HP delta vector |
| we_fainted | 1 | bool |
| opp_fainted | 1 | bool |
| opp_move_known | 1 | False on Explosion gap or first active turn |

All zeros on the first turn of each episode (`TurnDelta.empty()`). See `src/agents/observation/turn_delta_encoder.py`.

## Per-Pokémon Vector (58 dims)

| Field | Dims | Notes |
|---|---|---|
| Species ID | 1 | embedded → 32 |
| Base stats | 6 | hp/atk/def/spa/spd/spe |
| Item ID | 1 | embedded → 16 |
| Item known | 1 | |
| Type 1 ID | 1 | embedded → 16, concatenated (not summed) |
| Type 2 ID | 1 | embedded → 16 |
| Ability ID | 1 | embedded → 16 |
| Ability known | 1 | |
| Condition (status one-hot) | 7 | None, BRN, PAR, SLP, FRZ, PSN, TOX |
| Moves (4 × 9) | 36 | id, power, secondary, recoil, type_id, category, known, cur_pp, max_pp |
| HP fraction | 1 | |
| Active flag | 1 | |

## Attention Path Summary

| Path | Query | Keys/Values | Updates | Purpose |
|---|---|---|---|---|
| ① Pressure | our_active | their_team | our_active | What threatens our active right now? |
| ② Safety | our_team | their_active | our_team | Which of our bench can safely switch in? |
| ③ Synergy | our_team | our_team | our_team | Internal team role cohesion |
| ④ Threat | their_team | our_active | their_team | Which of their bench counters us most? |

Pressure result is written back into `our_team` before Safety/Synergy run, so all paths compose correctly. `our_active_refined` is extracted from the fully-composed `our_team` after all four paths.

## Key Files

| File | Role |
|---|---|
| `src/agents/model/features_extractor.py` | Network definition |
| `src/agents/observation/state_encoder.py` | Observation encoding; owns `base_dimension` (1053) and `dimension` (1093) |
| `src/agents/observation/turn_delta_encoder.py` | TurnDelta → 29-dim float32 block |
| `src/agents/training/battle_context.py` | `BattleContext` snapshot + `TurnDelta` diff |
| `src/agents/observation/constants.py` | Layout offsets and dimension constants |
| `src/agents/observation/reactive.py` | Reactive block encoder; `get_layout()` drives reactive offsets in the network |
| `designs/ai_v3/impl_step3_turn_transition_signal.md` | Design notes for TurnDelta + cant-move tracking |
| `designs/ai_v3/network_review.md` | Full architectural review with prioritised suggestions |
