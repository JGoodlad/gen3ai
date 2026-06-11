# Model Directory — Contributor Notes

## Architecture constants — single source of truth

All network dims are defined as module-level constants at the top of `features_extractor.py`:

```python
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]
ROLE_ENCODER_HIDDEN = [256, 128]
ACTIVE_CTX_HIDDEN = [64, 32]
```

**Change them here and nowhere else.** The phase modules' `__init__` read from these constants; `ModelVersion` imports them so `model_config.json` always reflects the live values. Do not hardcode these numbers anywhere else in the codebase.

Embedding dims (`species_embedding_dim`, `move_embedding_dim`, etc.) live in `state_encoder.get_layout()` and flow through `features_extractor_kwargs` — same principle, different file.

**`role_input_dim` is not a module-level constant** — it is computed dynamically in `PokemonEncoder.__init__` from the layout fields and `MOVE_NET_HIDDEN`. You do not need to update it manually when dims change; it is derived correctly. The projection input dim is also auto-discovered via a dummy forward pass for the same reason.

## Phase module structure

`forward_internal` is decomposed into phase `nn.Module`s, chained by a thin orchestrator:

`ObsUnpack` → `PokemonEncoder` → `TeamTransformer` → `CLSPool` → `ProjectionAssembler`,
then **two** root heads (`pre_proj_norm`/`projection` for policy, `value_pre_norm`/`value_projection`
for value), each → `ReLU`.

**Dual-head value readout (H4 / Option C).** The transformer body is shared, but the actor and
critic read it through independent paths. `CLSPool` holds a third query `value_cls` that attends
over all 12 team tokens to produce `value_pooled`; `ProjectionAssembler.forward` returns a
`(pi_combined, vf_combined)` pair; and the root `forward` returns a `(pi_features, vf_features)`
tuple. This extractor therefore **must** be paired with `Gen3DualHeadMaskablePolicy`
(`policy.py`), which keeps `share_features_extractor=True` (one body) and overrides `forward` /
`evaluate_actions` / `get_distribution` / `predict_values` to unpack the tuple and route each half
to `mlp_extractor.forward_actor` / `forward_critic`. A stock SB3 policy expects a single-tensor
extractor and will break. The startup `_run_roundtrip_test` and the snapshot/feature tests all
unpack the tuple — keep that in mind when touching the extractor's return shape.

### Phase-by-phase data flow

The embedding tables live in a shared `Embeddings` module passed as a forward argument to the
phases that need them, so they register exactly once. An immutable `ExtractorContext` produced
by `ObsUnpack` carries the ~30 unpacked tensors downstream, keeping each phase's signature
narrow. Both projection input dims are auto-discovered via a dummy forward pass in `__init__`,
so they stay correct when the architecture changes with no manual update.

1. **`Embeddings`** — shared tables: species (32), move (16), item (16), ability (16), type (16,
   shared for Pokémon types, move types, and TurnDelta move/type IDs). Owns the Hidden Power
   soft-type blend (`hp_soft_type`) and the per-slot TurnDelta embedder (`embed_delta_slot`).
2. **`ObsUnpack`** (stateless) — peels the flat 3390-dim observation into the named tensors of
   `ExtractorContext`: per-Pokémon block + categorical IDs, the global/reactive feature slices,
   the matchup matrices, and (hoisted here) the active-slot indices + fainted key-masks used
   downstream.
3. **`PokemonEncoder`** — embeds + stitches the enriched per-Pokémon vector; runs the **shared
   move processor** (Linear→ReLU→Linear, `MOVE_NET_HIDDEN`) over every move slot (input:
   move/type embeddings, remnants, known flag, battle context, per-move matchup ×6 +
   matchup-validity ×6, HP-candidate distribution, and prev-turn move validity), a
   **within-Pokémon move self-attention** (MHA 32-dim, 2 heads, + LayerNorm residual), then the
   **role encoder** (Linear→ReLU→Linear, `ROLE_ENCODER_HIDDEN`) → 12 × 128 role tokens.
4. **`TeamTransformer`** — builds a 23-token sequence (6 our-team + 6 their-team role tokens +
   `N_HISTORY_TURNS`=10 history tokens + 1 global token), adds token-type and history-positional
   embeddings, and runs a `TRANSFORMER_N_LAYERS`-deep `nn.TransformerEncoderLayer` stack (d_model
   128, `TRANSFORMER_N_HEADS` heads, FFN `TRANSFORMER_FFN_DIM`, post-LN) under a key-padding mask
   that masks fainted team slots and empty history slots. History tokens come from
   `embed_delta_slot`; the global token from the two active-contexts + non-matchup scalars.
   Returns the two refined team-token blocks. **Optional gradient checkpointing**: a runtime
   `grad_checkpointing` flag (set per run by `train_rl_agent.py --grad-checkpointing`, never
   saved/version-checked) runs these encoder layers under `torch.utils.checkpoint(...,
   use_reentrant=False)` during the backward-needing pass — **bit-exact** (dropout=0.0), trading
   one extra forward on the otherwise-idle GPU for the layers' ~5 GB of activation VRAM at
   batch 16384. A no-op under inference (gated on `torch.is_grad_enabled()`), so eval / the
   self-play opponent forward pay nothing.
5. **`CLSPool`** — one learned CLS query per side cross-attends over its 6 post-transformer team
   tokens (fainted slots key-masked) → a 128-dim pooled team token per side (+ LayerNorm). Also
   extracts `our_active_refined` = the transformer output of our active slot. A **third learned
   query, `value_cls`**, cross-attends over **all 12 team tokens** (both sides, fainted
   key-masked) → a 128-dim global `value_pooled` summary — a whole-board "who's winning" read for
   the critic, a different aggregation than the policy's our-active-centric pools.
6. **`ProjectionAssembler`** — emits a `(pi_combined, vf_combined)` pair. Policy: `our_pool(128)
   + their_pool(128) + our_active_refined(128) + active_ctx_enc(32) + opp_ctx_enc(32) +
   non_matchup_rest`. Value: `value_pooled(128) + active_ctx_enc(32) + opp_ctx_enc(32) +
   non_matchup_rest`. `active_ctx_encoder` (Linear→ReLU→Linear, `ACTIVE_CTX_HIDDEN`) is shared by
   both heads — it encodes inputs, not the contested body representation.
7. **Root heads** — two parallel `pre_proj_norm` (LayerNorm) → `projection` (Linear) → `ReLU`
   heads, one per `*_combined`, both emitting `PROJECTION_DIM`. SB3 sizes the shared
   `mlp_extractor` from `features_dim = PROJECTION_DIM`, then `Gen3DualHeadMaskablePolicy` feeds
   the policy half to `forward_actor` and the value half to `forward_critic`.

Rules to preserve:

- **Each phase owns its layers** (`move_network` lives under `pokemon_encoder`, `our_cls` under `cls_pool`, etc.). State_dict keys are therefore phase-prefixed.
- **`Embeddings` is the sole owner of the 5 embedding tables + `hp_type_idx_map`.** It is passed as a **forward argument** to `PokemonEncoder` and `TeamTransformer` — never stored as a child attribute on them — so the tables register exactly once. (The root exposes read-only `@property` forwarders like `model.type_embedding` for convenience; those add no state_dict keys.)
- **`ExtractorContext`** (frozen-by-convention dataclass) is the inter-phase contract: `ObsUnpack` produces it, downstream phases read from it. Add a field here rather than widening a phase's positional signature. Cross-phase values (active-slot indices, fainted masks, `hp_probs`) are computed once in `ObsUnpack` and carried on the context.
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version.py` (current: `gen3_markovian_progress_v1` — the literal source of truth is `ARCH_SIGNATURE` in `model_version.py`; check it there, not this prose). Pure decompositions still change state_dict keys, so old checkpoints must fail loudly. Re-sourcing or re-meaning an obs block (e.g. own IV/EV/nature going from constant fallbacks to real values via the poke-env `backfill_teambuilder_spread` fix; the event-sourced TurnDelta fold + status/item transition history; routing the trapping signals — `trapped`/`maybe_trapped`/`attempted_switch_rejected` — into the obs; the action-aligned per-move effect block — `gen3_move_effects_v1`; the per-our-mon incoming-damage / OHKO belief block — `gen3_incoming_damage_v1`; **re-calibrating that belief's VALUES** — `gen3_incoming_damage_v2`, which added a gen3 crit term + raised the offensive-stat tail to de-timid P(KO), and widened the candidate set [revealed-HP typed expansion, Return/Frustration pricing, broader prior floor/cap] so the killing move isn't silently absent; same 33 dims, values only; or adding the `turns_since_progress` no-progress-clock scalar at `vec[14]` — `gen3_markovian_progress_v1`, obs dim 3390 → 3391) is likewise retrain-class even when individual dims are unchanged.
- Per-phase unit tests live in `phase_modules_test.py` — `CLSPool` (incl. the `value_cls` pool) and `ProjectionAssembler` (which returns `(pi_combined, vf_combined)`) are tested on a hand-built `ExtractorContext` (`_dummy_ctx`) without a full forward pass. Prefer adding precise phase-level tests there.

## Model versioning (`model_version.py`, `snapshot.py`)

Every model save writes `model_config.json` + `metadata.json` alongside the `.zip` via `save_model_snapshot()`. Loading goes through `load_model_snapshot()` which runs `check_compatible()` before `MaskablePPO.load()` — a mismatch fails fast with a clear error rather than silently loading bad weights.

**When you change an architecture constant:**
- `check_compatible()` catches the mismatch automatically — no extra steps needed
- Old models can't be loaded, which is correct (rapid iteration project)

**When you add an optional new feature** (new field with a sensible default):
1. Add the field to `ModelVersion` in `model_version.py`
2. Bump `MODEL_CONFIG_VERSION`
3. Add one `if version < N:` block in `_migrate_config()` with `data.setdefault(...)`

**When you make a structural change** (different forward pass, new layer type):
1. Change `ARCH_SIGNATURE` in `model_version.py` (e.g. `"gen3_attn_v1"` → `"gen3_lstm_v1"`)
2. Old models get a clear arch-family error on load

**Resume-immutable training hparams (value-meaning, NOT weight-shape).** A hyperparameter can
be wrong-to-change-mid-run without changing any weight shape — `vf_coef` (`--vf-coef`) is the
first: it rescales the value head's gradient on the shared trunk, so a forgotten/typo'd flag on
resume would silently drift training. These are recorded on `ModelVersion` (→ `model_config.json`)
but **deliberately excluded from `check_compatible`** — that gates EVERY load, including the frozen
eval / self-play-pool / distill opponents, where the forward is identical regardless of the value
and a false rejection would break league play. Instead they get a dedicated check
(`ModelVersion.check_vf_coef`) invoked **only on the training-resume path** via
`load_model_snapshot(..., enforce_vf_coef=…)`; `train_rl_agent.py` FATALs on mismatch exactly like
an arch error. To add another such hparam, follow the optional-feature playbook above (field +
`MODEL_CONFIG_VERSION` bump + `_migrate_config` default) **plus** a dedicated `check_*` + an
`enforce_*` opt-in on `load_model_snapshot`, and leave it out of `_WEIGHT_FIELDS`.

The **reward-config** hparams are the same kind, bundled into one check: `bias_additivity`
(`--bias-additivity`), `mat_alive_weight` (`--mat-alive-weight`), `bias_redesign` (`--bias-redesign`),
`switch_bias_weight` (`--switch-bias-weight`, the belief-risk stay-into-KO BIAS lever, v5), and
`draw_penalty` (`--draw-penalty`, the DRAW/250-turn-timeout terminal, v7 — default −30.0 = a tie scores
as a decisive loss; set lower to make a stall-to-cap strictly worse) are all recorded on `ModelVersion`
and enforced on resume by **`check_reward_config`** (FATAL on drift, since they silently shift the
reward/objective), excluded from `check_compatible`. They are reward-VALUE changes — **no
`ARCH_SIGNATURE` bump** (the network/obs are unchanged) — so a fresh run is needed to measure them but
old checkpoints don't fail an arch check. Current `MODEL_CONFIG_VERSION` = **8**.

**Feature toggle that changes the value-head STRUCTURE (e.g. `use_popart`, v6).** Distinct from the
value-meaning hparams above: PopArt adds normalized output + `mu/sigma` buffers, so a mismatch breaks
the state_dict on EVERY load (eval / pool / distill included). So it goes in **`check_compatible`**
(not a resume-only `check_*`) with a dedicated, tailored message (NOT `_WEIGHT_FIELDS`, whose message
is about shapes), plus the bool field + `MODEL_CONFIG_VERSION` bump + a `_migrate_config`
`setdefault(...)` default. It lands in `model_config.json` via `to_json`; a resume that flips it fails
loudly. The litmus test: **value-meaning → resume-only `check_*`; structural → `check_compatible`.**

**Behavioral toggle that changes the FORWARD pass but not the state_dict (e.g.
`attend_unrevealed_opponents`, v8).** A third category: `--attend-unrevealed-opponents` keeps the
opponent's still-hidden party (unrevealed mons — Gen 3 has no team preview, so unseen slots arrive as
all-zero `species_known=0, hp=0` placeholders) **attendable** in the transformer instead of
key-masking them identically to revealed-fainted mons. It flips a single line in `ObsUnpack.forward`
(`fainted_mask_opp &= species_known>0.5` when on), threaded via `Gen3FeaturesExtractor(…,
attend_unrevealed_opponents)` ← `features_extractor_kwargs`. The weights are **identical shape** (no
`_WEIGHT_FIELDS` change, no `ARCH_SIGNATURE` bump, no obs-layout change) — but the mask the policy AND
value trained under differs, so a mid-run flip would feed a different forward. Like PopArt it lives in
**`check_compatible`** (dedicated message); unlike PopArt the state_dict is byte-identical either way,
so it is NOT a loadability concern — just a train/eval-consistency one. Refined litmus test:
**value-meaning → resume-only `check_*`; structural OR forward-behavior → `check_compatible`.** Off by
default (clean A/B baseline). The active opp is always revealed + force-unmasked, so even with every
bench slot attendable no key-padding row is all-True (no attention NaN).

A startup smoke test (`_run_roundtrip_test` in `train_rl_agent.py`) saves to a temp dir and reloads before every `model.learn()` call — catches serialization issues immediately.

## PopArt value-target normalization (`popart.py`, `--use-popart`)

Opt-in (default off). The dual-head extractor shares one trunk; with γ≈0.9999 the returns run to
±hundreds, so the value MSE gradient **swamps** the shared trunk and the policy under-updates
(diagnosed by `grad/value_share`≈1, see `src/agents/training/CLAUDE.md`). PopArt fixes the value
*scale* adaptively: `PopArtNormalizer` keeps running `(mu, sigma)` of the value targets, the value
head outputs **normalized** values, and the PPO loss trains in normalized space — so the value
gradient stays O(1). The **POP** half rescales `value_net`'s weight+bias on every stats update so the
**de-normalized** prediction is unchanged (`W'=(σ_old/σ_new)·W`, `b'=(σ_old·b+μ_old−μ_new)/σ_new`),
making the stats update a no-op on the value function (no corruption — the failure mode of naive
running-std normalization). Pure/torch-only → unit-tested in `popart_test.py` (load-bearing test:
**POP invariance**, de-normalized outputs identical across a stats update).

- **Policy integration** (`policy.py`): `__init__` takes `use_popart` (from `policy_kwargs`) and
  builds `self.popart` **after** `super().__init__` (which builds `value_net`); the 3 value sites
  (`forward`/`evaluate_actions`/`predict_values`) wrap the output in `self._denorm(...)` so GAE /
  advantages / bootstrapping always see **real-unit** values. `popart` is `None` when off (identity
  `_denorm`). The `(mu, sigma)` buffers ride the policy state_dict → save/restore for free.
- **PPO loop** (`instrumented_ppo.py`): once per `train()` (before the epochs) `popart.update(returns,
  value_net)` advances the stats + POPs; the value loss becomes `MSE(normalize(returns),
  normalize(values))`. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors
  otherwise — a self-documenting config beats a silent override): clipping is unnecessary with value
  normalization (the literature finds it little/negative regardless), and since the value sites
  return *de-normalized* values an active clip would clip in un-normalized units (`clip_range_vf` vs
  σ) and cripple the critic.
- **Version-checked**: `ModelVersion.use_popart` is recorded in `model_config.json` (config v3) and
  `check_compatible` raises a dedicated error if a resume toggles it — the value head's
  parameterization differs, so it can't be flipped mid-run.
- **Diagnostics** (TB + TUI): `popart/mu` & `popart/sigma` (should track `train/return_mean` &
  `train/return_std`), `popart/value_weight_norm` (POP keeps it bounded). With PopArt on,
  `train/value_loss` is the *normalized* loss (≈O(1)) and `grad/value_share` should fall toward ~0.4.
- `_DEFAULT_BETA` (EMA decay, 0.1) and `_SIGMA_FLOOR` (1e-2) are module constants in `popart.py`
  (the only flag is on/off). The POP rescale changes `value_net` outside the optimizer; momentum
  staleness is negligible because `σ_old/σ_new ≈ 1` each call (optimizer state intentionally not
  rescaled — the standard PopArt approximation).

## Where the canonical architecture lives

The live, maintained description of the extractor is **the "Phase-by-phase data flow" section
above** plus the root `CLAUDE.md` "Feature Extractor Architecture" summary. Keep those two in
sync when you change `features_extractor.py` — layers, dims, the token sequence, the CLS
pooling, the turn-history embedding, or active-context routing.

> `designs/ai_v3/README.md` holds an old Mermaid digraph + dimension table. It is a **frozen
> ai_v3 historical record** (1309-dim obs, the pre-unified-transformer attention paths) and is
> **NOT maintained** — do not update it for current-arch changes. It carries a banner saying so.
> If a fresh visual digraph of the ai_v4 arch is ever wanted, add a new one rather than editing
> the frozen ai_v3 one.
