# Gen3AI Network Architecture

Feature extractor used by MaskablePPO. Takes a 1105-dim observation and produces 512-dim features for the policy and value heads.

## Data Flow Digraph

```mermaid
flowchart TD
    OBS["Observation · 1105-dim float32"]

    OBS --> BASE["Base obs · 1065-dim"]
    OBS --> PM["prev_mask · 11-dim\n(previous turn's action mask)"]
    OBS --> TD["TurnDelta block · 29-dim\n(last turn's transition signal)"]

    PM --> SW["switch_mask [0:6]\nbench slot validity"]
    PM --> MV["move_mask [6:10]\nmove slot validity"]
    PM --> STR["struggle_mask [10]"]

    BASE --> TEAM["Pokémon vectors\n[B, 12, 59]  our + opp team"]
    BASE --> REM["remaining_part\nactive_ctx · global · reactive"]

    TEAM --> EMB["Embedding Lookups\nspecies · 32\nmove · 16\nitem · 16\nability · 16\ntype · 16  shared"]
    REM --> ACT_RAW["active_context · 23-dim × 2\nboosts · stat stages · volatiles"]
    REM --> GLOBAL["global env · 13-dim\nturn · weather·6 · fainted·2\nspikes·2 · reflect·2 · screen·2"]
    REM --> REACT["reactive scalars · 12-dim\npow·4 mult·4 fainted·2\nstatus·1 struggle·1\n(hp/spikes removed — in per-mon and global)"]
    REM --> MATCH["matchup matrix · 288-dim\n[B, 12, 4, 6] type effectiveness\n(used by move processor only)"]

    TD --> TDEMB["TurnDelta embedding\nour/opp move id → move_emb·16 each\nour/opp type id → type_emb·16 each\n+ 25 scalar remnants\n= 89-dim block"]
    EMB --> MPIN

    MV --> MPIN["Move Processor input · [B, 12, 4, 58]\nmove_emb·16 + type_emb·16\nremnants·6 + known·1\ncontext·12 + matchups·6 + validity·1"]
    MATCH --> MPIN
    GLOBAL --> MPIN

    MPIN --> MP["Shared Move Processor\nLinear 58→64 → ReLU\nLinear 64→32\n(all 12 mons × 4 slots)"]
    MP --> MSA["Move Self-Attention\nMHA 32-dim, 2 heads\nper-mon 4-slot self-attn + residual+LN\n(lets moves see each other)"]
    MSA --> PMOV["Processed moves · [B, 12, 128]\n4 slots × 32-dim"]

    EMB --> PE["pokemon_enriched · [B, 12, 242]\nspecies·32 + stats·6 + item_emb·16\nitem_known·1 + pk_types·32\nability_emb·16 + ability_known·1\ncondition·7 + moves·128\nhp+species_known+active·3"]
    PMOV --> PE

    GLOBAL --> RIN
    SW --> RIN
    STR --> RIN
    PE --> RIN["Role Encoder input · [B, 12, 260]\nenriched·242 + ctx·16\nswitch_valid·1 + struggle_prev·1\n(ctx=16: turn·1+weather·6+fainted·2+spikes·2+screens·4+struggle·1)"]

    RIN --> RE["Shared Role Encoder\nLinear 260→256 → ReLU\nLinear 256→128\n(all 12 mons)"]
    RE --> RT0["Role tokens · [B, 12, 128]"]

    ACT_RAW --> ACTINJ["Active Ctx → Role · shared\nLinear 23→64 → ReLU → Linear 64→128\n→ + injected into active slot only\n(bench mons have no boosts/volatiles)"]
    RT0 --> ACTINJ
    ACTINJ --> RT["Role tokens · [B, 12, 128]\n+ active-ctx bias at active slots\n+ status embedding bias\n(our active / our bench / their active / their bench)"]

    RT --> OT["our_team · [B, 6, 128]"]
    RT --> TT["their_team · [B, 6, 128]"]

    OT --> PA["① Pressure\nour_active ← their_team\nMHA + LayerNorm residual"]
    TT --> PA
    PA --> OAP["our_active_post_pressure\n[B, 1, 128]\nwritten back into our_team"]

    OAP --> OT2["our_team updated\n(active slot = post-Pressure)"]
    OT --> OT2

    OT2 --> SA["② Safety\nour_team ← their_active\nMHA + LayerNorm residual\nfainted queries zeroed"]
    TT --> SA
    SA --> OT3["our_team · [B, 6, 128]"]

    OT3 --> SYA["③ Synergy\nour_team ← our_team\nMHA (fainted key-masked)\n+ LayerNorm residual\nfainted queries zeroed"]
    SYA --> OT4["our_team final · [B, 6, 128]"]

    OAP --> THA["④ Threat\ntheir_team ← our_active_post_pressure\nMHA + LayerNorm residual\nfainted queries zeroed"]
    TT --> THA
    THA --> TT2["their_team · [B, 6, 128]"]

    TT2 --> OSY["⑤ Opp Synergy\ntheir_team ← their_team\nMHA (fainted key-masked)\n+ LayerNorm residual\nfainted queries zeroed"]
    OSY --> TT3["their_team final · [B, 6, 128]"]

    OT4 --> POOL["Attention Pool (per team)\nlearned query → MHA over 6 role tokens\nfainted slots key-masked\n→ one 128-dim token per side"]
    TT3 --> POOL
    OT4 --> OAR["our_active_refined · [B, 128]\nextracted from our_team final\nat our_active_idx"]
    OAR --> AGG

    POOL --> AGG["Aggregation\ncat(\n  our_pool·128\n  their_pool·128\n  our_active_refined·128\n  our_ctx_enc·32\n  opp_ctx_enc·32\n  global+scalars·29\n  turn_delta_emb·89\n)"]
    ACT_RAW --> ACTENC["Active Context Encoder · shared\nLinear 23→64 → ReLU → Linear 64→32\nour side + opp side (direct path)"]
    ACTENC --> AGG
    GLOBAL --> AGG
    REACT --> AGG
    TDEMB --> AGG

    AGG --> LN["Pre-projection LayerNorm\n(equalises per-block scales)"]
    LN --> PROJ["Projection\nLinear 562→512 → ReLU\n(562 auto-discovered via dummy forward)"]
    PROJ --> OUT["Features · [B, 512]\n→ policy head + value head"]
```

## Dimension Reference

| Layer | Input dim | Output dim | Notes |
|---|---|---|---|
| Move Processor | 58 | 32 | shared; run 12×4 times per forward pass |
| Move Self-Attention | 32 (Q/K/V) | 32 | per-mon 4-slot self-attn; run 12 times |
| Role Encoder | 260 | 128 | shared; run 12 times per forward pass |
| Active Ctx → Role (injection) | 23 | 128 | shared MLP; bias added to active slot's role token only, before all 5 attention paths |
| Active Ctx Encoder (direct) | 23 | 32 | shared; run twice (our side + opp side); appended to final projection input |
| Pressure Attn | 128 (Q), 128 (KV) | 128 | our_active queries their_team |
| Safety Attn | 128 (Q), 128 (KV) | 128 | our_team queries their_active; fainted queries zeroed |
| Synergy Attn | 128 (Q/K/V) | 128 | our_team self-attention; fainted keys+queries masked |
| Threat Attn | 128 (Q), 128 (KV) | 128 | their_team queries our_active_post_pressure; fainted queries zeroed |
| Opp Synergy Attn | 128 (Q/K/V) | 128 | their_team self-attention; fainted keys+queries masked |
| Our Pool Attn | 128 (Q), 128 (KV) | 128 | learned query over our_team (fainted key-masked) |
| Their Pool Attn | 128 (Q), 128 (KV) | 128 | learned query over their_team (fainted key-masked) |
| Pre-proj LayerNorm | 562 | 562 | equalises scales before projection |
| Projection | 562 | 512 | N auto-discovered via dummy forward in `__init__` |

## Observation Layout

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 59) | 354 | 0 |
| Opp team (6 × 59) | 354 | 354 |
| Active context ×2 (boosts, volatiles) | 44 | 708 |
| Global env | 13 | 752 |
| Reactive scalars + matchup matrix | 300 | 765 |
| **prev_mask** | **11** | **1065** |
| **TurnDelta block** | **29** | **1076** |
| **Total** | **1105** | |

The matchup matrix (288 of the 300 reactive dims) is consumed by the move processor only — it is **not** fed to the projection directly. The TurnDelta block is appended by `gen3_env.embed_battle()` and routed through the extractor's embedding tables (move/type IDs embedded; 25 scalars kept raw) producing a 89-dim embedded block at the projection.

## TurnDelta Block Layout (29 dims, offset 1076)

| Field | Dims | Notes |
|---|---|---|
| our_move_id | 1 | raw int (embedded → 16-dim in extractor) |
| our_power_norm | 1 | basePower / 200 |
| our_has_secondary | 1 | bool |
| our_has_recoil | 1 | bool |
| our_type_id | 1 | raw int (embedded → 16-dim in extractor) |
| opp_move_id | 1 | raw int (embedded → 16-dim in extractor) |
| opp_power_norm | 1 | |
| opp_has_secondary | 1 | |
| opp_has_recoil | 1 | |
| opp_type_id | 1 | raw int (embedded → 16-dim in extractor) |
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

After embedding in the extractor: 4 × 16-dim embeddings + 25 raw scalars = **89-dim** block fed into the projection aggregation.

All zeros on the first turn of each episode (`TurnDelta.empty()`). See `src/agents/observation/turn_delta_encoder.py`.

## Per-Pokémon Vector (59 dims)

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
| Species known | 1 | 1.0 if slot is populated; 0.0 if absent (unseen opp slot) |
| Active flag | 1 | appended by `state_encoder.py`, not `pokemon_encoder.py` |

## Attention Path Summary

| Path | Query | Keys/Values | Updates | Purpose |
|---|---|---|---|---|
| ① Pressure | our_active | their_team | our_active | What threatens our active right now? |
| ② Safety | our_team | their_active | our_team (fainted zeroed) | Which of our bench can safely switch in? |
| ③ Synergy | our_team | our_team | our_team (fainted zeroed) | Internal team role cohesion |
| ④ Threat | their_team | our_active | their_team (fainted zeroed) | Which of their bench counters us most? |
| ⑤ Opp Synergy | their_team | their_team | their_team (fainted zeroed) | Opponent team internal role cohesion |
| Pool (×2) | learned query | each team | 128-dim pool token | Permutation-equivariant team summary |

Pressure result is written back into `our_team` before Safety/Synergy run, so all paths compose correctly. `our_active_refined` is extracted from the fully-composed `our_team` after all paths.

All 5 paths operate on role tokens that have already received the **active-context injection** (boosts + volatile effects projected via `active_ctx_to_role`), so Safety can see "+2 Atk on their active", Threat can see "our active has Substitute up", etc.

## Key Files

| File | Role |
|---|---|
| `src/agents/model/features_extractor.py` | Network definition |
| `src/agents/observation/state_encoder.py` | Observation encoding; owns `base_dimension` (1065) and `dimension` (1105) |
| `src/agents/observation/turn_delta_encoder.py` | TurnDelta → 29-dim float32 block (move/type IDs as raw ints) |
| `src/agents/training/battle_context.py` | `BattleContext` snapshot + `TurnDelta` diff |
| `src/agents/observation/constants.py` | Layout offsets and dimension constants |
| `src/agents/observation/reactive.py` | Reactive block encoder; `get_layout()` drives reactive offsets in the network |
| `designs/ai_v3/impl_step3_turn_transition_signal.md` | Design notes for TurnDelta + cant-move tracking |
| `designs/ai_v3/network_review.md` | First architectural review |
| `designs/ai_v3/network_review_2.md` | Second architectural review (pruned of retracted items) |
