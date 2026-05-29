# ai_v4: Dedicated Value-Head Readout

## Motivation

The unified-transformer extractor (`design_unified_transformer.md`) produces a **single 512-dim
feature vector** that feeds *both* the policy MLP and the value MLP. Under SB3 the feature extractor
(the whole 23-token transformer) is **shared** between actor and critic; only the small `[512, 512]`
MLP heads differ.

This couples two objectives that want different things from the representation:

- **Policy** wants *action-discriminative* features — my active mon's move options, what threatens it,
  what I can switch to. It's a local, "what do I do this turn" view, and it is naturally anchored to a
  concrete token (our active Pokémon).
- **Value** wants a *global position estimate* — material, hazards, win-condition health, "who is
  winning." It has **no single anchor token**; it is a property of the whole board.

Because the two objectives backprop into the same representation, value is squeezed. Observed symptom:
**`train/value_loss ≈ 90`** on the v4 run (high; `explained_variance ≈ 0.82`, with headroom). A shared
512-d readout forces value to extract its global signal from a vector optimized largely for action
discrimination.

This design gives value its **own readout path off the shared transformer body** — cheap, principled,
and staged.

### Connection to v5+ (why this matters now)

`designs/CLAUDE.md` records a hard prerequisite for v5 league/MCTS play: *"reward annealing ≥ 50%
complete so the value head learns win probability (needed for MCTS in v5)."* Wang (2024) got the
headline result (78.6% → 90.8% vs Heuristic) from **MCTS at inference**, and MCTS uses **V(s) to
evaluate leaf nodes**. A value head that fits returns well is therefore not a side-tweak — it is
**direct groundwork for the biggest untapped lever in the roadmap.** A better-decoupled value head also
tightens GAE advantages (`advantage = return − V(s)`), which lowers policy-gradient variance and speeds
all PPO learning.

---

## Two-step plan

| Step | Change | Decoupling | Cost | When |
|------|--------|-----------|------|------|
| **1** | **Value-dedicated CLS readout** — one learned query pools all 23 tokens for value | readout-level | ~+0.2M params, body runs **once** | first |
| **2** | **Per-head readout transformer** — replace the single query with a multi-query (PMA) readout block + FFN | readout-level (richer) | moderate, body still shared | only if value still underfits |

Both keep the expensive transformer **body** (embeddings → move processor → role encoder → 23-token
self-attention stack) **shared and run once**. They diverge only at the final readout. Splitting *late*
maximises shared representation learning while specialising the task-specific aggregation.

> **What these steps do NOT fix:** the shared body still receives gradients from both losses, so a
> representation tug-of-war at the *body* level persists. Steps 1–2 decouple the **readout** (how value
> *reads* the tokens), not the **representation** (how the tokens are *shaped*). If value still underfits
> after Step 2, that is the diagnostic signal that the body itself is the bottleneck — at which point the
> fallback is two separate feature extractors (`share_features_extractor=False` with independent bodies,
> ~2× extractor cost). We deliberately try the cheap, high-likelihood fix first.

---

## Step 1 — Value-dedicated CLS readout

### Architecture

The body and the **policy readout are unchanged**. We add a parallel value readout off the same
post-transformer `tokens` (`features_extractor.py:712-714`).

```
                          ┌─→ [policy readout, UNCHANGED]
                          │     our_cls / their_cls pools + our_active_refined + ctx + scalars
tokens[B,23,128] ─────────┤     → pre_proj_norm → projection(→512) → ReLU      ── pi_features[512]
(shared body, run once)   │
                          └─→ [value readout, NEW]
                                value_cls query ─cross-attn over ALL 23 tokens (key_padding_mask)→
                                → norm_pool_value → [B,128] ++ global scalars
                                → value_pre_norm → value_projection(→512) → ReLU ── vf_features[512]
```

**New modules** (add in `__init__`, alongside `our_cls`/`their_cls` at `features_extractor.py:272`):

```python
self.value_cls        = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
self.value_cls_attn   = nn.MultiheadAttention(D_MODEL, TRANSFORMER_N_HEADS, batch_first=True)
self.norm_pool_value  = nn.LayerNorm(D_MODEL)
# value readout input = pooled value token (128) ++ raw global scalars (non_matchup_rest)
self.value_pre_norm   = nn.LayerNorm(D_MODEL + non_matchup_dim)
self.value_projection = nn.Linear(D_MODEL + non_matchup_dim, PROJECTION_DIM)
```

**Forward** (insert right after the transformer stack at line 714; policy `combined` path unchanged):

```python
# --- value readout (over ALL 23 post-transformer tokens) ---
v_q = self.value_cls.expand(batch_size, -1, -1)
v_pool, _ = self.value_cls_attn(v_q, tokens, tokens, key_padding_mask=key_padding_mask)  # [B,1,128]
v_pool = self.norm_pool_value(v_pool).squeeze(1)                                          # [B,128]
value_combined = torch.cat([v_pool, non_matchup_rest], dim=1)
```

`forward` then projects both heads and concatenates (see SB3 integration below):

```python
pi_features = self.activation(self.projection(self.pre_proj_norm(combined)))             # [B,512]
vf_features = self.activation(self.value_projection(self.value_pre_norm(value_combined)))# [B,512]
return torch.cat([pi_features, vf_features], dim=-1)                                      # [B,1024]
```

### Why these choices

- **Value query attends over all 23 tokens** (both teams + history + global), not just the team slices.
  V(s) is a global "who's winning" estimate, so it should see the whole board. The policy readout stays
  deliberately *our-active-centric*. Reuse the existing `key_padding_mask` (`:705`) so fainted slots and
  empty-history slots are masked out.
- **Concatenate raw global scalars** (`non_matchup_rest`, built at `:686` — turn count, weather, hazard
  layers, screens). These are strongly value-relevant and shouldn't have to survive a round-trip through
  attention to reach the value head.
- **No `policy_cls`.** Action selection is anchored to a concrete token — `our_active_refined`
  (`:734`), the active mon's own transformer output, pulled by index and kept *un-pooled*. A single
  policy summary token would blur exactly the signal the policy most needs. The asymmetry is the point:
  policy has a focal token, value does not.

### SB3 integration (recommended: concat single tensor + split MLP extractor)

SB3 2.8.0's `MaskableActorCriticPolicy` assumes the feature extractor returns **one** tensor of width
`features_dim`. The least-invasive integration keeps `share_features_extractor=True` (body runs once),
returns the concatenated `[B, 1024]` tensor (`features_dim = 2 * PROJECTION_DIM`), and **splits inside
the MLP extractor** so the actor reads the first 512 and the critic the second 512:

```python
class Gen3SplitMlpExtractor(MlpExtractor):           # actor input 512, critic input 512
    def forward_actor(self, features):  return self.policy_net(features[..., :PROJECTION_DIM])
    def forward_critic(self, features): return self.value_net(features[..., PROJECTION_DIM:])
    def forward(self, features):        return self.forward_actor(features), self.forward_critic(features)

class Gen3DualHeadPolicy(MaskableActorCriticPolicy):
    def _build_mlp_extractor(self):                  # build with split input dims
        self.mlp_extractor = Gen3SplitMlpExtractor(PROJECTION_DIM, net_arch=NET_ARCH, ...)
```

This keeps a single extractor forward (cheap), a single-tensor save/load contract, and — importantly —
leaves `_run_roundtrip_test` (`train_rl_agent.py:177`, which compares
`reloaded.policy.features_extractor(dummy_obs)`) working unchanged on one `[B,1024]` tensor.

> **Alternative considered:** `share_features_extractor=False` + a tuple return `(pi_features,
> vf_features)`. SB3 2.8.0 supports the tuple convention natively (`extract_features` returns a tuple,
> `forward` calls `mlp_extractor.forward_actor/forward_critic`). Rejected as primary because the
> not-shared path instantiates **two** extractor modules (it would run the body twice unless aliased)
> and the round-trip test would need to handle a tuple. The concat approach is strictly cheaper and
> simpler here.

`policy_kwargs` then passes `policy=Gen3DualHeadPolicy`; `net_arch=[512, 512]` is unchanged (it now
describes each head's MLP, fed a 512-d slice).

---

## Step 2 — Per-head readout transformer (optional, only if value still underfits)

A single pooling query can do exactly **one weighted average** of the 23 tokens — it cannot *reason*.
If Step 1 narrows but does not close the value gap, upgrade the value readout from one query to a small
**cross-attention readout block** (the Set-Transformer PMA pattern, Lee et al. 2019):

```
value readout block:
  K_v learned seed queries  ─cross-attn over 23 tokens (key_padding_mask)→  [B, K_v, 128]
  → LayerNorm residual → FFN(128→256→128) → LayerNorm                       [B, K_v, 128]
  → flatten / mean over K_v  → ++ global scalars → value_projection(→512)
```

- `K_v` (e.g. 4) seed queries let value gather several distinct facets (offensive pressure, defensive
  backbone, hazard/clock state) instead of one blurred average — and the FFN gives it *multi-step*
  refinement, i.e. its own reasoning over the shared tokens.
- The **policy readout may optionally get the same treatment**, but it already has a focal token + two
  team pools, so Step 2 prioritises value.
- Cost is in the readout only — the body is still shared and run once. Much cheaper than duplicating the
  body.

Step 2 is **gated on evidence**: implement only if `value_loss` / `explained_variance` after Step 1 show
value is still the limiting factor (see Success criteria).

---

## Key Design Decisions

**Why not jump straight to separate bodies (`share_features_extractor=False`)?** That fully decouples
the representation but ~doubles the extractor's params and per-step compute. The likelier diagnosis here
is that value needs a different *aggregation* of a basically-sound board representation, which the
readout fix solves at a fraction of the cost. Separate bodies remain the documented fallback if Steps
1–2 don't move `value_loss`.

**Honest caveat on `value_loss ≈ 90`.** With `gamma = 0.9999`, `HP_VALUE = 2`, and `VICTORY_VALUE = 30`,
returns span tens-to-hundreds, so an RMSE of ~9.5 may partly reflect **reward scale**, not underfit. The
dedicated readout's win is largest if it's genuine underfit. This is cheap enough to just measure — the
round-trip + a short run will tell us.

**Value sees the global token + scalars directly.** Redundant-but-cheap: the global token already flows
through attention, but feeding the raw scalars to the value head too removes any dependence on the
attention preserving them.

---

## Model Versioning

| Field | Old | New |
|-------|-----|-----|
| `ARCH_SIGNATURE` | `"gen3_abilities_v2"` | `"gen3_dualvalue_v1"` |
| `MODEL_CONFIG_VERSION` | `2` | `2` (unchanged) |

This is a structural change (new value readout, dual-output forward, custom policy class), so bump
`ARCH_SIGNATURE` in `model_version.py:40`. All prior checkpoints then fail `check_compatible()` with a
clear arch-family error — correct for a rapid-iteration project, no migration needed. The observation
vector is **unchanged** (no new obs features), so encoder/layout work is nil.

---

## Implementation Checklist

### `src/agents/model/features_extractor.py`
- [ ] Add `value_cls`, `value_cls_attn`, `norm_pool_value`, `value_pre_norm`, `value_projection` to
      `__init__` (read `non_matchup_dim` from the same layout slice used to build `non_matchup_rest`).
- [ ] In `forward`/`forward_internal`: after the transformer stack, compute `value_combined`; project
      both heads; return `cat([pi_features, vf_features], dim=-1)` of width `2 * PROJECTION_DIM`.
- [ ] Set `self.features_dim = 2 * PROJECTION_DIM` (1024).

### New file `src/agents/model/dual_head_policy.py`
- [ ] `Gen3SplitMlpExtractor(MlpExtractor)` — `forward_actor`/`forward_critic`/`forward` slice the 1024
      into two 512 halves.
- [ ] `Gen3DualHeadPolicy(MaskableActorCriticPolicy)` — override `_build_mlp_extractor` to build the
      split extractor with input dim `PROJECTION_DIM`.

### `src/main/train_rl_agent.py`
- [ ] Pass `Gen3DualHeadPolicy` as the policy (replace `"MultiInputPolicy"` at `:847`, or register it),
      keeping `net_arch=[512, 512]` and the existing `features_extractor_kwargs`.
- [ ] Confirm `_run_roundtrip_test` passes on the `[B,1024]` single-tensor output (no change expected).

### `src/agents/model/model_version.py`
- [ ] `ARCH_SIGNATURE = "gen3_dualvalue_v1"`.

### Tests
- [ ] `features_extractor_test.py`: expect forward output width `2 * PROJECTION_DIM`; assert
      `pi`/`vf` halves are produced by different readout params (e.g. perturb `value_cls`, check only the
      second half changes).
- [ ] Add a policy test: split MLP extractor routes the correct half to actor vs critic.
- [ ] `pytest src/ -m "not integration and not e2e" -q`.

### Update digraph
- [ ] Per `src/agents/model/CLAUDE.md`, update the Mermaid digraph in `designs/ai_v3/README.md` — the
      readout/aggregation changed.

---

## Verification

```bash
export PYTHONPATH=$PYTHONPATH:src

# Forward-shape sanity
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -c "
import torch
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.model.features_extractor import Gen3FeaturesExtractor, PROJECTION_DIM
from utils.mappings import load_mappings
enc = Gen3ObservationEncoder(load_mappings())
ext = Gen3FeaturesExtractor(enc.get_features_extractor_kwargs()['observation_space'] if False else None) # build per existing test harness
print('features_dim should be', 2*PROJECTION_DIM)
"

# Smoke test (full pipeline, ~1 min) — must print [ModelVersion] Round-trip smoke test PASSED
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

### Success criteria (decides whether Step 2 runs)
Compare a short v4.1 run against the v4 baseline at matched steps:
- **`train/value_loss`** drops meaningfully (target: well below ~90) and/or **`explained_variance`**
  rises above ~0.85 → Step 1 worked; Step 2 likely unnecessary.
- Value metrics barely move but win rates are flat → value was reward-scale-bound, not underfit; do
  **not** invest in Step 2, revisit reward scaling instead.
- Value metrics improve but plateau below target → value needs *reasoning*, not just aggregation →
  proceed to **Step 2** (per-head readout transformer).
