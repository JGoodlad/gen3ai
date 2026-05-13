# Step 4: Team-Wide Attention Architecture

This phase transitions the model from a simple "Active Matchup" focus to a comprehensive team-wide reasoning engine. We implement a multi-path attention system that allows the model to reason about pressure, safety, and internal synergy simultaneously.

## 🚀 Architectural Suggestions

### 1. Pre-Attention: Status Biases
Before any attention layers, we inject learnable biases into the **12 Role Tokens** to signify their current status in the battle.
*   **Categories**: `Our Active`, `Our Bench`, `Their Active`, `Their Bench`.
*   **Implementation**: `nn.Embedding(4, 128)` added to the tokens based on the `active_flags`.

### 2. Cross-Attention 1: "The Pressure" (with Residual)
*   **Goal**: Refine our active Pokémon's role token with knowledge of the pressure it exerts on the opponent's team.
*   **Query**: Our Active Role Token ($x_{oa}$).
*   **Key/Value**: Their Team Role Tokens.
*   **Residual**: $x_{oa} = x_{oa} + \text{PressureAttn}(x_{oa}, \text{TheirTeam})$
*   **Insight**: Ensures the model always has a direct path to our active Pokémon's core stats while "layering on" matchup pressure.

### 3. Cross-Attention 2: "The Safety" (with Residual)
*   **Goal**: Refine our team's role tokens with knowledge of how they handle the current opponent.
*   **Query**: Our Team Role Tokens ($x_{ot}$).
*   **Key/Value**: Their Active Role Token.
*   **Residual**: $x_{ot} = x_{ot} + \text{SafetyAttn}(x_{ot}, \text{TheirActive})$
*   **Insight**: Every member of our team (active and bench) is updated with a "safety profile" relative to the threat. This is the key "Switch Logic" engine.

### 4. Self-Attention: "Internal Synergy" (with Residual)
*   **Goal**: Allow our team members to "talk" to each other to understand overall strategy.
*   **Query/Key/Value**: Our Team Role Tokens ($x_{ot}$).
*   **Residual**: $x_{ot} = x_{ot} + \text{SynergyAttn}(x_{ot}, x_{ot})$
*   **Insight**: Helps the model realize synergistic states, such as "I have a Spiker and a Spinner ready," by allowing tokens to share context.

---

## 🛠️ Code Implementation (src/agents/model/features_extractor.py)

### Inside `__init__`:
```python
# 1. Status Biases (0: Our Active, 1: Our Bench, 2: Their Active, 3: Their Bench)
self.status_embedding = nn.Embedding(4, 128)

# 2. Multi-Path Attention Heads
self.pressure_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
self.safety_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
self.synergy_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

# 3. LayerNorm for Stability (Standard in Residual blocks)
self.norm1 = nn.LayerNorm(128)
self.norm2 = nn.LayerNorm(128)
self.norm3 = nn.LayerNorm(128)
```

### Inside `forward_internal`:
```python
def forward_internal(self, role_tokens, active_flags):
    # --- PRE-ATTENTION ---
    our_active_idx = torch.argmax(active_flags[:, 0:6], dim=1)
    opp_active_idx = torch.argmax(active_flags[:, 6:12], dim=1) + 6
    
    status_idx = torch.ones((batch_size, 12), device=role_tokens.device).long()
    status_idx[:, 0:6] = 1   # Our Bench
    status_idx[:, 6:12] = 3  # Their Bench
    status_idx[torch.arange(batch_size), our_active_idx] = 0 # Our Active
    status_idx[torch.arange(batch_size), opp_active_idx] = 2 # Their Active
    
    role_tokens += self.status_embedding(status_idx)

    # Split for logic
    our_team = role_tokens[:, 0:6, :]
    their_team = role_tokens[:, 6:12, :]
    our_active = role_tokens[torch.arange(batch_size), our_active_idx].unsqueeze(1)
    their_active = role_tokens[torch.arange(batch_size), opp_active_idx].unsqueeze(1)

    # --- ATTENTION PATHS WITH RESIDUALS ---

    # Path 1: The Pressure (Update Our Active with Opponent Context)
    pressure_delta, _ = self.pressure_attn(our_active, their_team, their_team)
    our_active = self.norm1(our_active + pressure_delta)
    
    # Path 2: The Safety (Update Our Team with Opponent Active Context)
    safety_delta, _ = self.safety_attn(our_team, their_active, their_active)
    our_team = self.norm2(our_team + safety_delta)
    
    # Path 3: Synergy (Update Our Team with Internal Context)
    synergy_delta, _ = self.synergy_attn(our_team, our_team, our_team)
    our_team = self.norm3(our_team + synergy_delta)

    # --- FINAL AGGREGATION ---
    # Re-stitch our team (which now has Safety and Synergy baked in)
    # and combine with the updated Active Pressure token
    our_team_flat = our_team.reshape(batch_size, -1)
    our_active_refined = our_active.squeeze(1)
    
    combined = torch.cat([our_team_flat, our_active_refined, remaining_part], dim=1)
    return self.activation(self.projection(combined))
```

## 📊 Why Residuals Matter
1.  **Gradient Flow**: Prevents vanishing gradients in deep attention stacks.
2.  **Feature Preservation**: The model can learn to ignore the attention (delta = 0) if the raw stats are sufficient for the current turn.
3.  **Bootstrapping**: Allows the model to start with a "good enough" policy based on raw stats while the complex attention logic gradually refines its strategy over millions of steps.
