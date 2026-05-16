# Gen3AI Network Architecture

Feature extractor used by MaskablePPO. Takes a 1032-dim observation and produces 512-dim features for the policy and value heads.

## Data Flow Digraph

```mermaid
flowchart TD
    OBS["Observation · 1032-dim float32"]

    OBS --> BASE["Base obs · 1021-dim"]
    OBS --> PM["prev_mask · 11-dim\n(previous turn's action mask)"]

    PM --> SW["switch_mask\n[0:6] — bench slot validity"]
    PM --> MV["move_mask\n[6:10] — move slot validity"]
    PM --> STR["struggle_mask\n[10]"]

    BASE --> TEAM["Pokémon vectors\n[B, 12, 55]  our + opp team"]
    BASE --> REM["remaining_part\n context · global · reactive"]

    TEAM --> EMB["Embedding Lookups\nspecies · 32\nmove · 16\nitem · 16\nability · 16\ntype · 16  shared"]
    REM --> CTX["Global context · 12-dim\nturn · weather·6 · fainted·2\nspikes·2 · struggle·1"]
    REM --> MATCH["Matchup matrix\n[B, 12, 4, 6]\ntype effectiveness per move vs opp slot"]

    MV --> MPIN
    EMB --> MPIN["Move Processor input\n[B, 12, 4, 56]\nmove_emb·16 + type_emb·16\nremnants·4 + known·1\ncontext·12 + matchups·6\nvalidity·1"]
    MATCH --> MPIN

    MPIN --> MP["Shared Move Processor\nLinear 56→64 → ReLU\nLinear 64→32\n(shared across all 12 mons × 4 slots)"]
    MP --> PMOV["Processed moves · [B, 12, 128]\n4 slots × 32-dim"]

    EMB --> PE["pokemon_enriched · [B, 12, 226]\nspecies·32 + stats·6 + item_emb·16\nitem_known·1 + pk_types·16\nability_emb·16 + ability_known·1\ncondition·8 + moves·128 + hp+active·2"]
    PMOV --> PE

    CTX --> RIN
    SW --> RIN
    STR --> RIN
    PE --> RIN["Role Encoder input · [B, 12, 240]\nenriched·226 + ctx·12\nswitch_valid·1 + struggle_prev·1"]

    RIN --> RE["Shared Role Encoder\nLinear 240→256 → ReLU\nLinear 256→128\n(shared across all 12 mons)"]
    RE --> RT["Role tokens · [B, 12, 128]\n+ status embedding bias\n(our active / our bench / their active / their bench)"]

    RT --> OT["our_team · [B, 6, 128]"]
    RT --> TT["their_team · [B, 6, 128]"]

    OT --> PA["Pressure Attn\nour_active ← their_team\nMultiheadAttention + LayerNorm residual"]
    TT --> PA
    PA --> OAR["our_active_refined · [B, 1, 128]"]

    OT --> SA["Safety Attn\nour_team ← their_active\nMultiheadAttention + LayerNorm residual"]
    TT --> SA
    SA --> OT2["our_team · [B, 6, 128]"]

    OT2 --> SYA["Synergy Attn\nour_team ← our_team\nMultiheadAttention + LayerNorm residual"]
    SYA --> OT3["our_team final · [B, 6, 128]"]

    OT3 --> AGG["Aggregation\ncat( our_team·768, their_team·768,\n     our_active·128, remaining_raw )"]
    TT --> AGG
    OAR --> AGG
    REM --> AGG

    AGG --> PROJ["Projection\nLinear N→512 → ReLU\n(N discovered via dummy forward)"]
    PROJ --> OUT["Features · [B, 512]\n→ policy head + value head"]
```

## Dimension Reference

| Layer | Input dim | Output dim | Notes |
|---|---|---|---|
| Move Processor | 56 | 32 | shared; run 12×4 times per forward pass |
| Role Encoder | 240 | 128 | shared; run 12 times per forward pass |
| Pressure Attn | 128 (Q), 128 (KV) | 128 | our_active queries their_team |
| Safety Attn | 128 (Q), 128 (KV) | 128 | our_team queries their_active |
| Synergy Attn | 128 (Q/K/V) | 128 | our_team self-attention |
| Projection | N (auto) | 512 | N determined by dummy forward in `__init__` |

## Observation Layout

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 55) | 330 | 0 |
| Opp team (6 × 55) | 330 | 330 |
| Active context ×2 | 44 | 660 |
| Global env | 13 | 704 |
| Reactive + matchups | 304 | 717 |
| **prev_mask** | **11** | **1021** |
| **Total** | **1032** | |

## Key Files

| File | Role |
|---|---|
| `src/agents/model/features_extractor.py` | Network definition |
| `src/agents/observation/state_encoder.py` | Observation encoding; owns `base_dimension` (1021) and `dimension` (1032) |
| `src/agents/observation/constants.py` | Layout offsets and dimension constants |
| `designs/ai_v3/impl_step2_obs_features.md` | Design notes for the prev_mask feature |
