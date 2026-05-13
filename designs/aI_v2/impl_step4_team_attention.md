# === 1. POKEMON SHOWDOWN CONFIG (deps/pokemon-showdown/config/config.js) ===
exports.subprocesses = {
    network: 1, 
    simulator: 16, 
    validator: 1,
    verifier: 1
};

# === 2. SHOWDOWN LAUNCH COMMAND ===
NODE_ENV=production node --turbo-fast-api-calls --max-old-space-size=4096 deps/pokemon-showdown/pokemon-showdown start --no-security

# === 3. PYTHON TRAINING COMMAND ===
python3 src/main/train_rl_agent.py --steps 13000000 --n-envs 32 --batch-size 16384 --n-epochs 10 --ent-coef 0.02 --n-steps 2048 --lr 0.0003 --device cuda --log-level periodic

# === 4. TRANSFORMER ENCODER (src/agents/model/features_extractor.py) ===
# Inside __init__:
encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
self.team_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
self.ownership_embedding = nn.Embedding(2, 128)
self.pos_embedding = nn.Parameter(torch.randn(1, 12, 128))

# Inside forward_internal:
def forward_internal(self, role_tokens):
    ownership_idx = torch.tensor([0,0,0,0,0,0,1,1,1,1,1,1], device=role_tokens.device)
    role_tokens += self.ownership_embedding(ownership_idx)
    role_tokens += self.pos_embedding
    return self.team_transformer(role_tokens)

# === 5. TARGETED SWITCHING ATTENTION (src/agents/model/features_extractor.py) ===
# Inside __init__:
self.switch_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
self.q_proj = nn.Linear(128, 128)
self.k_proj = nn.Linear(128, 128)
self.v_proj = nn.Linear(128, 128)

# Implementation:
def get_switch_context(self, role_tokens):
    opponent_active = role_tokens[:, 6:7, :] # Opponent Active
    self_bench = role_tokens[:, 1:6, :]      # Self Bench
    
    query = self.q_proj(opponent_active) 
    key = self.k_proj(self_bench)        
    value = self.v_proj(self_bench)      
    
    attn_output, attn_weights = self.switch_attention(query, key, value)
    return attn_output.squeeze(1), attn_weights.squeeze(1)
