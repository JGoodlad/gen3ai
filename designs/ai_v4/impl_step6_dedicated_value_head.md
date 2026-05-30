# Implementation: Step 6 — Value-Dedicated CLS Readout (Dual-Head Policy)

This step gives the **critic its own readout path off the shared transformer body**.
Until now the unified-transformer extractor produced a single 512-dim feature vector
that fed *both* the policy MLP and the value MLP — the whole 23-token transformer was
shared, and only the small `[512, 512]` MLP heads differed. Policy and value want
different things from that representation (policy: action-discriminative, anchored to
our active mon; value: a global "who's winning" estimate with no single anchor token),
so a single shared readout squeezed value. Observed symptom on the v4 run:
`train/value_loss ≈ 90` with `explained_variance ≈ 0.82` (headroom).

This implements **Step 1** of `design_dedicated_value_head.md`: a third learned CLS
query (`value_cls`) pools the team tokens into a value-specific summary, the assembler
emits a `(pi_combined, vf_combined)` pair, the extractor grows a second projection head,
and `forward` returns a `(pi_features, vf_features)` tuple consumed by a new custom
policy. The transformer **body is shared and runs once** — only the readout, projection,
and critic MLP branch are independent. The observation vector is **unchanged by this
step** (the obs dim is 2754 on the current tree — set by the move-attribution work, not
by this readout change), so there is zero encoder/layout work here. `ARCH_SIGNATURE`
bumps to `gen3_dual_value_v1`.

Primary themes: split *late* (decouple the readout, not the body), keep one body via
`share_features_extractor=True`, and route the two halves with a thin policy subclass
rather than a custom MLP-extractor slice. Step 2 (the PMA per-head readout transformer)
is **not** built — it is gated on whether Step 1 closes the value gap.

---

## Motivation

### The coupling

Under SB3 the feature extractor is shared between actor and critic by default. Both
losses backprop into the same 23-token representation:

- **Policy** wants *action-discriminative* features — my active mon's moves, what
  threatens it, what I can switch to. A local, "what do I do this turn" view, naturally
  anchored to a concrete token (`our_active_refined`).
- **Value** wants a *global position estimate* — material, hazards, win-condition
  health. It has **no single anchor token**; it is a property of the whole board.

`value_loss ≈ 90` is the symptom: value is forced to extract a global signal from a
vector optimised largely for action discrimination. A dedicated readout is cheap,
principled groundwork — it also tightens GAE advantages (`advantage = return − V(s)`),
and a value head that fits returns well is the **direct prerequisite for v5 MCTS**, which
uses `V(s)` to evaluate leaf nodes.

> **Honest caveat (carried from the design).** With `gamma = 0.9999`, `HP_VALUE = 2`,
> `VICTORY_VALUE = 30`, returns span tens-to-hundreds, so RMSE ≈ 9.5 may partly reflect
> **reward scale**, not underfit. The dedicated readout's win is largest if it's genuine
> underfit — cheap enough to just measure (success criteria are in the design doc).

---

## What Changed

### `value_cls` pool (`CLSPool`)

A third learned query is added alongside `our_cls`/`their_cls`:

```python
self.value_cls       = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
self.value_cls_attn  = nn.MultiheadAttention(D_MODEL, TRANSFORMER_N_HEADS, batch_first=True)
self.norm_pool_value = nn.LayerNorm(D_MODEL)
```

In `forward` it cross-attends over **both teams' 12 post-transformer tokens** (a combined
fainted key-mask) → a 128-dim `value_pooled` global summary, and `CLSPool` now returns a
4-tuple `(our_team_pooled, their_team_pooled, our_active_refined, value_pooled)`:

```python
all_team_out = torch.cat([our_team_out, their_team_out], dim=1)          # [B, 12, 128]
all_fainted  = torch.cat([ctx.fainted_mask_ours, ctx.fainted_mask_opp], dim=1)
value_cls_q  = self.value_cls.expand(batch_size, -1, -1)
value_pool_out, _ = self.value_cls_attn(value_cls_q, all_team_out, all_team_out,
                                        key_padding_mask=all_fainted)
value_pooled = self.norm_pool_value(value_pool_out).squeeze(1)           # [B, 128]
```

### Dual projection heads (`ProjectionAssembler` + root)

`ProjectionAssembler.forward` now returns a `(pi_combined, vf_combined)` pair. The policy
input is unchanged; the value input is the value pool plus the same per-side encoded
active contexts and the non-matchup scalar tail (turn, weather, hazards, screens):

```
pi_combined = our_pool(128) + their_pool(128) + our_active_refined(128)
              + our_ctx_enc(32) + opp_ctx_enc(32) + non_matchup_rest(25)   = 473
vf_combined = value_pooled(128) + our_ctx_enc(32) + opp_ctx_enc(32)
              + non_matchup_rest(25)                                        = 217
```

`active_ctx_encoder` is reused for both heads — it encodes *inputs*, not the contested
body representation. The root extractor discovers both input dims via the dummy forward
and builds two heads (`pre_proj_norm`/`projection` and `value_pre_norm`/`value_projection`),
each `→ PROJECTION_DIM (512)`. `forward` returns the tuple:

```python
pi_features = self.activation(self.projection(self.pre_proj_norm(pi_combined)))      # [B,512]
vf_features = self.activation(self.value_projection(self.value_pre_norm(vf_combined)))# [B,512]
return pi_features, vf_features
```

`features_dim` stays `PROJECTION_DIM = 512` (each head's width), which is what SB3 uses to
size the shared `mlp_extractor`.

### Custom policy (`Gen3DualHeadMaskablePolicy`, new `policy.py`)

A `MaskableMultiInputActorCriticPolicy` subclass keeps `share_features_extractor=True` —
so SB3 builds exactly **one** transformer body — and overrides the four methods that
consume features (`forward`, `evaluate_actions`, `get_distribution`, `predict_values`) to
unpack the `(pi_features, vf_features)` tuple and route each half to
`mlp_extractor.forward_actor` / `forward_critic`. Nothing else (action distribution, value
net, masking) changes. The extractor **must** be paired with this policy — a stock SB3
policy expects a single-tensor extractor and breaks. `train_rl_agent.py` passes the class
in place of the `"MultiInputPolicy"` string.

### Version bump

`ARCH_SIGNATURE = "gen3_dual_value_v1"` (`model_version.py`, v7 changelog entry). New
weights (value pool + second projection) and a tuple-returning forward make this
incompatible with v6 (`gen3_modular_v1`) checkpoints — `check_compatible()` rejects them
at load with a clear arch-family error, which is correct for this rapid-iteration project.
`MODEL_CONFIG_VERSION` is unchanged (no schema field added); the obs vector is unchanged,
so no migration is needed.

---

## Implementation Details

### Readout dims (obs and body unchanged)

| Path | Composition | Input dim | → head |
|---|---|---|---|
| Policy | our_pool 128 + their_pool 128 + our_active_refined 128 + ctx 32 + ctx 32 + non_matchup 25 | **473** | Linear → 512 |
| Value | value_pooled 128 + ctx 32 + ctx 32 + non_matchup 25 | **217** | Linear → 512 |

`obs total_dim = 2754` (this step does not touch it), `D_MODEL = 128`, `PROJECTION_DIM = 512`,
`features_dim = 512`. Both input dims are auto-discovered by the `__init__` dummy forward,
so they stay correct if upstream blocks resize.

### SB3 2.8.0 seam

With `share_features_extractor=True`, SB3's base `extract_features` returns whatever the
extractor returns — here the `(pi, vf)` tuple — so the four overridden consumers simply
unpack it. The single body is built once; `mlp_extractor` (from `net_arch=[512, 512]`,
unchanged) builds independent actor and critic branches fed a 512-d slice each;
`value_net` reads the critic branch. No `MlpExtractor` subclass and no second body.

---

## Divergences from the design

The design (`design_dedicated_value_head.md`) sketched Step 1; three implementation
choices departed from its **recommended** path, each deliberately:

1. **SB3 integration: tuple, not concat.** The design *recommended* returning a single
   `[B, 1024]` tensor and splitting inside a custom `Gen3SplitMlpExtractor`, and listed the
   `(pi, vf)` tuple as the *rejected alternative* (because the not-shared path would
   instantiate two bodies). We built the **tuple** — but neutralised that downside by
   keeping `share_features_extractor=True` (one body) and overriding the four consumers
   directly. This needs no custom MLP-extractor slicing and keeps `features_dim = 512`
   (not 1024). The round-trip and feature tests were updated to unpack the tuple rather
   than relying on a single `[B,1024]` tensor.

2. **Value query pools the 12 team tokens, not all 23.** The design had `value_cls` attend
   over all 23 tokens (teams + history + global). We pool over the **12 team tokens** only:
   history and global information has already flowed into the team tokens through the
   unified transformer, so the team tokens are a whole-board read, and this keeps the
   value pool's masking symmetric with the per-side pools (a simple `cat` of the two
   fainted masks). If value underfits, widening the value query back to 23 tokens is a
   cheap thing to try before Step 2.

3. **Value input includes the encoded active contexts.** The design fed the value head
   `value_pooled ++ non_matchup_rest`. We also concatenate `our_ctx_enc` and `opp_ctx_enc`
   (the per-side active-context encodings — boosts/volatiles), since those are strongly
   value-relevant and the encoder is already computed for the policy head.

Minor: the policy class lives in `policy.py` as `Gen3DualHeadMaskablePolicy` (design said
`dual_head_policy.py` / `Gen3DualHeadPolicy`); `ARCH_SIGNATURE` is `gen3_dual_value_v1`
(design said `gen3_dualvalue_v1`).

---

## Edge Cases

- **Fainted masking on the value pool.** The combined `[B, 12]` key-mask masks fainted
  slots on both sides; a phase test perturbs a masked opponent slot and asserts the value
  summary does not move.
- **Single shared body, not two.** `share_features_extractor=True` is load-bearing — it is
  why only one transformer runs. Flipping it to `False` (the Option-A fallback) would
  instantiate a second body and is explicitly *not* what this step does.
- **Save/load of the custom policy.** SB3 stores the policy class by reference; the
  snapshot round-trip reconstructs `Gen3DualHeadMaskablePolicy` and reproduces both
  `pi`/`vf` halves bit-for-bit (asserted in `snapshot_test.py`).
- **Inference path.** `model.predict()` flows through the overridden `get_distribution`
  (policy half only), so no inference code assumes a single-tensor extractor.

---

## Test Suite

### Unit tests

- `phase_modules_test.py` — `CLSPool`/`ProjectionAssembler` call sites updated to the new
  arities; new `test_clspool_value_pool_shape_and_masking` (value pool ignores a masked
  slot on either side) and `test_clspool_value_query_is_wired` (zeroing `value_cls` moves
  the value pool).
- `features_extractor_test.py`, `features_extractor_hp_test.py`,
  `state_encoder_test.py` — forward/`forward_internal` call sites unpack the `(pi, vf)`
  tuple; shape assertions check both heads.
- `snapshot_test.py` — round-trip built on `Gen3DualHeadMaskablePolicy`; both `pi` and
  `vf` features asserted reproduced across save/load.
- Full unit suite: **897 passed**, plus **18 integration** passing.

### PPO end-to-end (ad-hoc)

Constructed a real `InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, …)` on the live
layout and exercised `forward` / `evaluate_actions` / `predict_values`, a `learn(128)`
cycle, and a save/load reproduction of both heads — all pass.

### Startup smoke test

`train_rl_agent.py --debug --steps 10000` printed
`[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))`, completed episodes,
fired the replay callback, handled a 250-turn stall without hanging, and printed eval win
rates — the full pipeline runs on the dual-head policy.

---

## What This Enables

- A **value head that reads the board through its own lens** — a global pooling query plus
  direct global scalars — instead of sharing the policy's action-centric readout.
- **Decoupled critic gradients** at the readout/projection/MLP level: the actor and critic
  no longer fight over the final 512-d vector, which should tighten advantages and lower
  policy-gradient variance.
- The infrastructure for the **v5 MCTS value function** — a dedicated, better-fit `V(s)` is
  exactly what leaf-node evaluation consumes.

Whether to proceed to **Step 2** (per-head PMA readout transformer) is gated on the v4.1
run's `value_loss` / `explained_variance` versus the v4 baseline (success criteria in the
design doc). Step 2 is intentionally unbuilt here.

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/model/features_extractor.py` | `value_cls` pool in `CLSPool` (4-tuple return); `ProjectionAssembler` emits `(pi_combined, vf_combined)`; root grows `value_pre_norm`/`value_projection`; dual-dim discovery; `forward` returns `(pi_features, vf_features)` |
| `src/agents/model/policy.py` | **New** — `Gen3DualHeadMaskablePolicy`: one shared body, overrides `forward`/`evaluate_actions`/`get_distribution`/`predict_values` to route each tuple half |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE` → `gen3_dual_value_v1` + v7 changelog |
| `src/main/train_rl_agent.py` | Pass `Gen3DualHeadMaskablePolicy` (was `"MultiInputPolicy"`); round-trip test unpacks the tuple |
| `src/agents/model/phase_modules_test.py` | New value-pool tests; updated `CLSPool`/`ProjectionAssembler` arities |
| `src/agents/model/features_extractor_test.py` | Unpack `(pi, vf)` in forward/`forward_internal` sites |
| `src/agents/model/features_extractor_hp_test.py` | Unpack `(pi, vf)` in forward sites |
| `src/agents/model/snapshot_test.py` | Build on the dual-head policy; assert both heads reproduce |
| `src/agents/observation/state_encoder_test.py` | Compatibility test unpacks both heads |
| `CLAUDE.md`, `src/agents/model/CLAUDE.md`, `README.md` | Document the dual-head value readout |
