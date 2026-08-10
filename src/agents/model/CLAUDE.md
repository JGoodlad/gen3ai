# Model Directory — Contributor Notes

## Architecture constants — single source of truth

All network dims are defined as module-level constants in **`arch_constants.py`** (relocated there
2026-08-01 so `damage_op.py` can read them without importing the extractor — that would be circular).
`features_extractor.py` **re-exports the whole block unchanged**, so it remains the documented import
surface and `from agents.model.features_extractor import D_MODEL` still resolves:

```python
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]
MOVE_LATENT_HIDDEN = 64      # MoveLatentEncoder MLP hidden
MOVE_LATENT_DIM = 32         # per-move latent dim (the similarity-grading space)
ROLE_ENCODER_HIDDEN = [256, 128]
ACTIVE_CTX_HIDDEN = [64, 32]
```

**Change them in `arch_constants.py` and nowhere else.** The phase modules' `__init__` read from these constants; `ModelVersion` imports them so `model_config.json` always reflects the live values. Do not hardcode these numbers anywhere else in the codebase.

Embedding dims (`species_embedding_dim`, `move_embedding_dim`, etc.) live in `state_encoder.get_layout()` and flow through `features_extractor_kwargs` — same principle, different file.

**`role_input_dim` is not a module-level constant** — it is computed dynamically in `PokemonEncoder.__init__` from the layout fields and `MOVE_NET_HIDDEN`. You do not need to update it manually when dims change; it is derived correctly. The projection input dim is also auto-discovered via a dummy forward pass for the same reason.

## Phase module structure

`forward_internal` is decomposed into phase `nn.Module`s, chained by a thin orchestrator:

`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `[MoveBelief? (prefuse)]` →
`[SpreadBelief+HPTypeBelief+DamageOperator? (damage_op_prefuse)]` → `TeamTransformer` →
`[BeliefHead?]` → `[MoveBelief? (default, post)]` → `[SpreadBelief+HPTypeBelief+DamageOperator?
(default, post)]` → `CLSPool` → `[damage_reattend? → re-attend + RE-POOL]` → `ProjectionAssembler`,
then **two** root heads
(`pre_proj_norm`/`projection` for policy, `value_pre_norm`/`value_projection` for value), each → `ReLU`.

`BeliefSlots`/`BeliefHead` are built only when `opp_belief_slots` (`--opp-belief-aux-coef>0`),
`MoveBelief` only when `move_belief_mode != off`, `DamageOperator` only when `damage_op` (which requires
`move_belief_mode` revealed/both); with all off the chain is the baseline `ObsUnpack →
PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler` byte-for-byte. `BeliefSlots` swaps the
un-revealed opp role-tokens for learned unknown-mon tokens *before* the transformer (so the belief is
refined in-lineup); `BeliefHead` reads the refined opp tokens *after* the transformer and stashes the
species/moves aux logits (a side readout — does NOT feed forward); `MoveBelief` predicts + **reinjects**
the moveset into `their_team_out` *before* the CLS pools (so it DOES flow to the heads); `DamageOperator`
runs *after* `MoveBelief` and consumes its predicted-move logits to compute the believed-move incoming
damage to each of our mons. **gen3_no_concat_v1 (v61): its flat block no longer enters either
projection** — the op reaches the policy via the pointer cells + prefuse injection + edge cells, and
the critic via the `MultiSeedValueReadout` (k=4×64 seed queries over the per-our-mon rows, vf-only,
with the `seeds/*` TB collapse contract logged every train()).
Under `--opp-belief-latent-coef>0` `BeliefHead` ALSO carries an asymmetric SimSiam latent predictor
(the `latent` logits key) and `forward_internal` stashes a stop-grad `last_belief_target_latent` (the
`pokemon_encoder` role-tokens of the true hidden mons, from the training-only `belief_target_slots` obs
key) — also a side readout, never fed forward (leak-safe). See `designs/CHANGELOG.md` for how these landed.
A separate `WinProbHead` (`win_prob_mode != none`) reads `value_pooled` *after* the pools and stashes
a `last_win_prob_logits` [B,1] — another side readout (never in pi/vf, so projection dims are unchanged),
read by the win-prob aux loss + the prober. `read_only` feeds it a STOP-GRAD `value_pooled` (head trains
its own params only); `shaping` feeds it live (the win objective also shapes the trunk).

**Dual-head value readout (H4 / Option C).** The transformer body is shared, but the actor and
critic read it through independent paths. `CLSPool` holds a third query `value_cls` that attends
over all 12 team tokens to produce `value_pooled`; `ProjectionAssembler.forward` returns a
`(pi_combined, vf_combined)` pair; and the root `forward` returns a `(pi_features, vf_features)`
tuple. This extractor therefore **must** be paired with `Gen3DualHeadMaskablePolicy`
(`policy.py`), which keeps `share_features_extractor=True` (one body) and overrides `forward` /
`evaluate_actions` / `get_distribution` / `predict_values` to unpack the tuple and route each half
to `mlp_extractor.forward_actor` / `forward_critic`. A stock SB3 policy expects a single-tensor
extractor and will break — doubly so under the pointer-native action head (`gen3_pointer_native_v1`): the policy's `_build`
deletes the flat `action_net` and the action logits come from the `PointerNativeActionHead` over
the extractor's `last_pointer_inputs` stash (per-logit inputs: `designs/ARCHITECTURE.md` § Heads). The startup `_run_roundtrip_test` and the snapshot/feature tests all
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
2. **`ObsUnpack`** (stateless) — peels the flat observation (2667 dims under
   `gen3_entity_rehome_v1`) into the named tensors of `ExtractorContext` via the declarative
   schema's validated slice map (`build_schema(layout).slices()` — the tiling proof runs at
   construction): per-Pokémon block + categorical IDs, the global/board feature slices, and
   (hoisted here) the active-slot indices + fainted key-masks used downstream.
3. **`PokemonEncoder`** — embeds + stitches the enriched per-Pokémon vector; runs the **shared
   move processor** (Linear→ReLU→Linear, `MOVE_NET_HIDDEN`) over every move slot (input:
   move/type embeddings, remnants, known flag, battle context, HP-candidate distribution, and
   prev-turn move validity — the CPU matchup ×6 / validity ×6 inputs are DELETED with their obs
   block, `gen3_entity_rehome_v1`), a
   **within-Pokémon move self-attention** (MHA 32-dim, 2 heads, + LayerNorm residual), then the
   **role encoder** (Linear→ReLU→Linear, `ROLE_ENCODER_HIDDEN`) → 12 × 128 role tokens. The role
   input carries the **E2 active-context injection** (gen3_entity_rehome_v1): each side's
   boosts+volatiles block scattered onto its ACTIVE mon's row (bench rows zero) — the entity owns
   its own ctx; the global-token/projection routes remain (additive). Pinned by
   `e2_ctx_injection_test.py`.
4. **`TeamTransformer`** — builds a 20-token sequence (6 our-team + 6 their-team role tokens +
   `N_HISTORY_TURNS`=7 history tokens + 1 global token), adds token-type and history-positional
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
   self-play opponent forward pay nothing. **Optional iterative damage refinement**
   (`--damage-refine-rounds N`): a `between_layers(tokens, i)` callback runs BEFORE each of the first N
   encoder layers to recompute the DamageOperator's lean discrete incoming damage from the being-enriched
   opp tokens and inject it (via the extractor's zero-init `refine_proj`) onto our-mon token positions — so
   each layer attends over physics from the freshest belief. `None` (off) ⇒ the loop is byte-identical.
   (Built by the extractor; mutually exclusive with the pre-attention op — the production config runs the latter, so this loop does not exist there.)
5. **`CLSPool`** — one learned CLS query per side cross-attends over its 6 post-transformer team
   tokens (fainted slots key-masked) → a 128-dim pooled team token per side (+ LayerNorm). Also
   extracts `our_active_refined` = the transformer output of our active slot. A **third learned
   query, `value_cls`**, cross-attends over **all 12 team tokens** (both sides, fainted
   key-masked) → a 128-dim global `value_pooled` summary — a whole-board "who's winning" read for
   the critic, a different aggregation than the policy's our-active-centric pools.
5b. **`HiddenOppBeliefPool`** *(optional — built only when `--opp-belief-cls-k > 0`)* — **k** distinct
   learned query tokens run through a `TransformerDecoderLayer` (self-attention among the queries to
   coordinate + cross-attention to the 12 team tokens under the single-sourced `ctx.all_fainted`
   key-mask) → a `[B, k·D_MODEL]` hidden-opponent belief. `None` when `k=0`. See the v9 toggle note
   under *Model versioning* and `designs/ai_v5/design_offense_and_opponent_belief.md` §B2.
6. **`ProjectionAssembler`** — emits a `(pi_combined, vf_combined)` pair. Policy: `our_pool(128)
   + their_pool(128) + our_active_refined(128) + active_ctx_enc(32) + opp_ctx_enc(32) +
   non_matchup_rest`. Value: `value_pooled(128) + active_ctx_enc(32) + opp_ctx_enc(32) +
   non_matchup_rest`. When the hidden-opponent belief is on, its `[B, K·D_MODEL]` is appended to
   **both** (last), widening each projection input by `k·D_MODEL`. `active_ctx_encoder`
   (Linear→ReLU→Linear, `ACTIVE_CTX_HIDDEN`) is shared by both heads — it encodes inputs, not the
   contested body representation.
7. **Root heads** — two parallel `pre_proj_norm` (LayerNorm) → `projection` (Linear) → `ReLU`
   heads, one per `*_combined`, both emitting `PROJECTION_DIM`. SB3 sizes the shared
   `mlp_extractor` from `features_dim = PROJECTION_DIM`, then `Gen3DualHeadMaskablePolicy` feeds
   the policy half to `forward_actor` and the value half to `forward_critic`.

Rules to preserve:

- **Each phase owns its layers** (`move_network` lives under `pokemon_encoder`, `our_cls` under `cls_pool`, etc.). State_dict keys are therefore phase-prefixed.
- **`Embeddings` is the sole owner of the 5 embedding tables + `hp_type_idx_map`.** It is passed as a **forward argument** to `PokemonEncoder` and `TeamTransformer` — never stored as a child attribute on them — so the tables register exactly once. (The root exposes read-only `@property` forwarders like `model.type_embedding` for convenience; those add no state_dict keys.)
- **`ExtractorContext`** (frozen-by-convention dataclass) is the inter-phase contract: `ObsUnpack` produces it, downstream phases read from it. Add a field here rather than widening a phase's positional signature. Cross-phase values (active-slot indices, fainted masks, `hp_probs`) are computed once in `ObsUnpack` and carried on the context.
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version.py`. **Read the live value there, not from prose.** Three cases people get wrong:
  - A **pure decomposition** still changes state_dict keys, so old checkpoints must fail loudly — bump it.
  - A forward-math change with **unchanged `out_dim` / projection widths** is not shape-caught by anything, so the signature bump is the ONLY thing that rejects a stale checkpoint. This is the case that has bitten most often.
  - **Re-sourcing or re-meaning an obs block** is retrain-class even when no individual dim moves (a constant fallback becoming a real value; a scalar's definition changing; a block moving from `available_moves` order to request-slot order). The long list of historical examples lives in `designs/CHANGELOG.md`.
- Per-phase unit tests live in `phase_modules_test.py` — `CLSPool` (incl. the `value_cls` pool) and `ProjectionAssembler` (which returns `(pi_combined, vf_combined)`) are tested on a hand-built `ExtractorContext` (`_dummy_ctx`) without a full forward pass. Prefer adding precise phase-level tests there.

## File layout (`gen3_damage_op_split_v1`, 2026-08-01)

`features_extractor.py` was ~4,700 lines, of which `DamageOperator` alone was **1,689 (39%)**. Split
into three, a **pure relocation** — same classes, same constants, same forward math:

| file | holds |
|---|---|
| `arch_constants.py` (37) | the architecture constants — the single source of truth for weight-shape dims |
| `damage_op.py` (2,102) | `DamageOperator` + its `_DMG_*` constants + `decode_damage_block` |
| `features_extractor.py` (2,922) | everything else; **re-exports all 89 moved names** |

No import cycle: `DamageOperator` touches the extractor only through `ctx: 'ExtractorContext'`, which
is a **string** forward-reference and so costs no runtime import. The re-export means every historical
path (`from agents.model.features_extractor import DamageOperator / decode_damage_block / _DMG_* /
_SB_ATK`) still resolves — the prober, `model_version`, `snapshot` and the tests all rely on that.

**The gate for a refactor claiming to change nothing is proof, not review:** byte-identity on pi/vf +
the raw op block (`tmp/damage_op_equiv_probe.py`), unchanged `state_dict` keys, the constructed-scenario
physics oracle (`damage_op_probe_fuzz_test.py`, 22/22), and the full suite. All four held.

## ⚠️ One op's SPELLING is load-bearing for `torch.compile` (`gen3_species_posterior_spelling_v1`)

`BeliefHead.species_posterior` computes `P(species)` for the expected-latent defender
(`--threat-unrevealed-outgoing`). It is written as **`log_softmax(...).exp()`, not
`torch.softmax(...)`, and that is deliberate** — do not "simplify" it.

`torch.softmax` over the last dim of the `[B,6,n_species]` logits lowers to a numerator buffer plus a
`[B,6,1]` denominator, and the Inductor **CPU** scheduler then trips `AssertionError: buf<N>` trying to
fuse the division. That single op was the reason `--compile-extractor` used to set
`torch._dynamo.config.suppress_errors = True`, which in turn meant the production config compiled only
partially (3.6× instead of 6.53×) and every other backend failure in the process went silent.

`tmp/softmax_variant_probe.py` measured the alternatives: `.contiguous()`, `.clone()`, a 2-D
reshape and a hand-rolled `exp / sum` **all still fail**; only the `log_softmax().exp()` factoring
lowers cleanly. It is mathematically identical and keeps the same max-subtraction stability (measured
max|Δ| vs eager 5.07e-07). Guarded by `species_posterior_compiles_test.py` — the fast tests pin the
math, and `GEN3AI_COMPILE_TESTS=1` runs a real compile of the literal production arch with
suppression OFF (verified to fail if the old spelling returns). Repro: `tmp/inductor_crash_repro.py`.

**The general lesson:** a backend that "can't compile our model" was one op, not a property of the
architecture. Before reaching for a global suppression flag, bisect to the op — see
`src/agents/training/CLAUDE.md` → Compiled CPU opponents.

## ⚠️ Identity-at-init is NOT free — SB3 clobbers it (`gen3_identity_init_guard_v1`)

**Every `nn.Linear` you zero-initialise inside the feature extractor is orthogonally
re-initialised by SB3 when the policy is built.** `ActorCriticPolicy._build()` runs
`self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))`
(`stable_baselines3/common/policies.py:617-631`); `init_weights` re-inits every Linear/Conv2d it
finds, and `ortho_init` defaults **True**.

Until 2026-08-01 this silently falsified the identity-at-init contract for **13** Linears in every
real training run — `refine_proj`, `outgoing_proj`, `status_in_proj`/`status_out_proj`,
`film_pi`/`film_vf`, plus the belief heads (`MoveBelief.move_head`, `SpreadBelief.*`,
`HPTypeBelief.type_head`) whose zero-init is what makes the **cold-start posterior equal the Smogon
prior**. Measured max|W| before the fix: 0.19–0.47. See `designs/research_state/ledger.md` → **M1**
for the standing caveat this puts on the K10 and D4 result families.

**The guard.** `Gen3FeaturesExtractor.restore_identity_init()` re-zeros them, and
`Gen3DualHeadMaskablePolicy.__init__` calls it after `super().__init__()` (by which point SB3 has
finished). The protected set is captured **by observation** at the end of `__init__` — any Linear
whose weight is all-zero once construction finishes was zero-init'd on purpose — rather than a
hand-kept list, so **a new zero-init module is protected automatically**. Embeddings (e.g.
`zarch_lut_emb`) are untouched by SB3 and need no guard.

**The rule this leaves you with:** an invariant asserted only in a unit test that builds the module
(or a bare extractor) **directly** is not an invariant — that construction path is not the one
training uses. Assert "byte-identical / identity-at-init / cold-start == prior" claims on a REAL
`MaskablePPO`-built policy. `identity_init_test.py` does exactly that, and fails 8/10 if the guard
is removed.

## Model versioning (`model_version.py`, `snapshot.py`)

Every model save writes the **run-level** `model_config.json` + `metadata.json` at the run root via `save_model_snapshot()`, plus a **per-checkpoint** `.json` sidecar beside each checkpoint `.zip` (`write_checkpoint_metadata`, derived from the zip path). Periodic + forced checkpoints `.zip` live in `<run>/checkpoints/` (so their sidecar lands there too); the run-level config/metadata stay one level up at the run root. Loading goes through `load_model_snapshot()`, which resolves the zip then searches **its dir AND its parent** for `model_config.json` (so the run-root config is found even when the zip is in `checkpoints/`; `load_foreign_opponent` does the same) and runs `check_compatible()` before `MaskablePPO.load()` — a mismatch fails fast with a clear error rather than silently loading bad weights. (`snapshot_history` keys + the `worktree.py` resume lookup stay BARE basenames, e.g. `checkpoint_123_steps.zip`, regardless of the subdir.)

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

**⚠️ REORDERING a module's parameters silently breaks the optimizer on resume.** SB3/torch save+load
the Adam optimizer state **by parameter POSITION, not name**. So if a refactor changes the *order*
`named_parameters()` yields (e.g. building submodules in `__init__` in a different sequence — the
`gen3_nature_ev_belief_v1` bug, where `SpreadBelief.__init__` moved `reinject`/`norm` before
`stat_head`), a resume's **weights** still load fine (name-keyed `load_state_dict` → arch check PASSES)
but the **momentum** (`exp_avg`/`exp_avg_sq`) gets assigned to the WRONG params. It then crashes in
`AdamW.step()` ("size of tensor a (128) must match b (5)") the moment a misassigned param of a
different shape first gets a gradient — **data-dependently, so it can survive many steps**, and (until
the guard) the broad `except` in `train_rl_agent.py` masked it as a clean completion. Guard:
`train_rl_agent._validate_or_reset_optimizer_state(model, checkpoint_path)` runs on every resume and
**REMAPS the momentum to the current params BY NAME** — it reads the saved optimizer state + the saved
parameter NAME ORDER straight from the checkpoint zip (`policy.optimizer.pth` + `policy.pth`) and
rebuilds `opt.state` so each current param receives the momentum saved for its name, regardless of
registration order. So a reorder is **corrected**, not just caught: a **same-shape** reorder (which a
shape check CANNOT see and would silently scramble) now follows the name, and a name reused at a
different shape (or a genuinely new param) cleanly drops to fresh zero-init. **This means "append new
params LAST" is no longer load-bearing for optimizer correctness** — though still good hygiene. Falls
back to the legacy shape-only drop-all-momentum reset only if the zip can't be read (never crashes a
resume); no-op (momentum carried verbatim) on an aligned resume. Pinned by
`src/main/resume_optimizer_realign_test.py` (incl. the same-shape-reorder + zip-read cases).

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
`switch_bias_weight` (`--switch-bias-weight`, the belief-risk stay-into-KO BIAS lever, v5),
`draw_penalty` (`--draw-penalty`, the DRAW/250-turn-timeout terminal, v7 — default −30.0 = a tie scores
as a decisive loss; set lower to make a stall-to-cap strictly worse), `self_ko_hp_penalty`
(`--self-ko-hp-penalty`, the HP-scaled self-KO penalty — default 0.0 = OFF; >0 charges −w·hp when
our mon self-KOs via Explosion/Self-Destruct, since the symmetric material PBRS prices a healthy 1-for-1
trade at ~0 and the critic then over-values it), the de-bias cleanup pair `drop_redundant_bias` +
`drop_switch_bias` (`--drop-redundant-bias` / `--drop-switch-bias` — zero the audit-flagged
distorting BIAS terms: stall_tax + matchup_penalty redundant with the no-progress clock/`--draw-penalty`
and `pbrs_belief`; the hand-coded switch subsidy), and the **two end-state PBRS switches**
`all_shaping_pbrs` + `stall_pbrs` plus `no_progress_penalty` (`--all-shaping-pbrs` / `--stall-pbrs` /
`--no-progress-penalty`): `all_shaping_pbrs` = "everything but stall" — folds
Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status and **zeros every BIAS term except the anti-stall tilt
`no_progress_tax`** (so all non-stall shaping is policy-invariant; the bad turn-ramp `stall_tax` is
zeroed); `stall_pbrs` = "stall" — folds Φ_progress and zeros `no_progress_tax`+`stall_tax`. Run BOTH ⇒
the whole BIAS class is zero (TERMINAL + PBRS only); run only `all_shaping_pbrs` ⇒ keep the
`no_progress` stall tilt as the single acknowledged BIAS. `no_progress_penalty` is recorded+checked
because it is Φ_progress's weight. (`--all-shaping-pbrs` ALSO now folds the DEDICATED phaze-out-boosts PBRS
**`pbrs_roar`** Φ_roar = −`ROAR_BOOST_WEIGHT`(0.25)·Σmax(0,opp-active-boost) — NO separate flag/field, it
rides the existing `all_shaping_pbrs` toggle, stacking with the bundled `pbrs_opp_boosts` for stronger
proportional roar-out-boosts shaping; safe since both telescope to 0.) All are recorded on
`ModelVersion` and enforced on resume by **`check_reward_config`** (FATAL on drift, since they silently
shift the reward/objective), excluded from `check_compatible`. They are reward-VALUE changes — **no
`ARCH_SIGNATURE` bump** (the network/obs are unchanged) — so a fresh run is needed to measure them but
old checkpoints don't fail an arch check — a fresh run is needed to measure them.
The live `MODEL_CONFIG_VERSION` is in `model_version.py`; per-version entries are in `designs/CHANGELOG.md`.

**The per-version entries that used to live here have moved to `designs/CHANGELOG.md` §4**
(verbatim). They described what each of v6–v57 added, in parallel with the root `CLAUDE.md`'s own
version narrative — two records of the same history that had drifted out of agreement with each
other and with the code.

- **What the architecture IS right now** — obs layout, the phase chain under the production config,
  what each head consumes, the `DamageOperator` block, the edge families, the flag table with
  `INERT` markings: **`designs/ARCHITECTURE.md`**.
- **What each version changed**: `designs/CHANGELOG.md` (history — do not quote as current).
- **The live values**: `MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` in `model_version.py`. Read them
  there. This file deliberately no longer states them: a version number written into prose is stale
  the moment the next one lands, and quoting a stale one is how a v30 description got applied to a
  v59 model.

The mechanics above (what to bump when, the optimizer-reorder guard, the resume-immutable-hparam
playbook) are the durable part and stay here. When you add a toggle, follow those rules, then record
the entry in `CHANGELOG.md` and state the new truth in `ARCHITECTURE.md` — never narrate it here.

A startup smoke test (`_run_roundtrip_test` in `train_rl_agent.py`) saves to a temp dir and reloads before every `model.learn()` call — catches serialization issues immediately.

## PopArt value-target normalization (`popart.py`, `--use-popart`)

Opt-in (default off). The dual-head extractor shares one trunk; with γ≈0.9999 the returns run to
±hundreds, so the value MSE gradient **swamps** the shared trunk and the policy under-updates
(diagnosed by a large positive `grad/value_policy_logratio`, see `src/agents/training/CLAUDE.md`). PopArt fixes the value
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
  `train/value_loss` is the *normalized* loss (≈O(1)) and `grad/value_policy_logratio` should fall toward ~0.
- `_DEFAULT_BETA` (EMA decay, 0.1) and `_SIGMA_FLOOR` (1e-2) are module constants in `popart.py`
  (the only flag is on/off). The POP rescale changes `value_net` outside the optimizer; momentum
  staleness is negligible because `σ_old/σ_new ≈ 1` each call (optimizer state intentionally not
  rescaled — the standard PopArt approximation).

## Where the canonical architecture lives

| Question | File |
|---|---|
| What the architecture **IS** right now — obs layout, the phase chain with the production config's flags resolved, what each head consumes, the `DamageOperator` block, the edge families, the flag table | **`designs/ARCHITECTURE.md`** |
| A machine-checked picture of it — seats and sinks, edges typed by what they physically carry | `designs/architecture_graph.dot` (generated by `delivery_graph.py`, pinned by `delivery_graph_test.py`) |
| **The same picture, interrogable** — path queries ("what does the critic see?"), the measured-dependence overlay across every audited checkpoint, a per-family bias selector, and per-token detail (what a seat can deliver to, and every bias family acting on it ranked by measured dependence). **Family codes are never shown bare** — every `d2` / `c1` / `s3` carries its one-line label, with the cell definition parsed out of `features_extractor.py`'s `_EDGE_*_CELL` block so it cannot drift (`FAMILY_LABEL` holds only the curated phrase; a family with no entry fails the tests) | **https://model.g5d.io** — served live by `--serve` (re-rendered from the checkout per request, so it cannot go stale), or **`designs/architecture_viewer.html`** via `file://`. **Dark by default**. Generated by `build_arch_viewer.py` from **real asset files** — `arch_viewer_assets/viewer.{html,css,js}`, not a string literal, so the JS is `node --check`ed by a test and the CSS is lintable; the server lives apart in `arch_viewer_serve.py`. Pinned by `build_arch_viewer_test.py`; regenerate with `python -m agents.model.build_arch_viewer` (`--check` is the staleness gate). `--vendor --out <path>` inlines cytoscape for a copy that needs no network at all (a separate output — the committed artifact stays CDN-linked so `--check` has one thing to compare against) |
| **Does the page actually render?** — the text tests never execute a line of its JavaScript, and a `#theme` deep link once painted every node in the wrong palette because cytoscape resolves the CSS variables once at construction | `build_arch_viewer_render_integration_test.py` — headless chrome reads back a `document.body.dataset` record (script completed, every node positioned, and the node fill **as cytoscape computed it**). Skips, naming which, when there is no browser or no network |
| The **phase CONTRACT** — what a phase may own, `ExtractorContext` / `Embeddings` rules, the versioning playbook | the "Phase-by-phase data flow" + "Model versioning" sections **above** |
| How each version got here | `designs/CHANGELOG.md` |

The split is deliberate: this file holds the **rules** a phase must follow (durable), and
`ARCHITECTURE.md` holds the **state** the model is currently in (changes every run). When you touch
`features_extractor.py`, update the contract here if a rule changed, and `ARCHITECTURE.md` if the
state did — then regenerate the delivery graph.

> `designs/ai_v3/README.md` holds an old Mermaid digraph + dimension table. It is a **frozen
> historical record** (1309-dim obs, the pre-unified-transformer attention paths) and is **NOT
> maintained** — do not update it for current-arch changes. It carries a banner saying so. It is
> also the reason `designs/architecture_graph.dot` is generated rather than drawn.
