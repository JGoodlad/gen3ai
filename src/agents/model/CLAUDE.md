# Model Directory — Contributor Notes

## Architecture constants — single source of truth

All network dims are defined as module-level constants at the top of `features_extractor.py`:

```python
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]
MOVE_LATENT_HIDDEN = 64      # MoveLatentEncoder MLP hidden (v24, gen3_unified_move_system_v1)
MOVE_LATENT_DIM = 32         # per-move latent dim (the similarity-grading space)
ROLE_ENCODER_HIDDEN = [256, 128]
ACTIVE_CTX_HIDDEN = [64, 32]
```

**Change them here and nowhere else.** The phase modules' `__init__` read from these constants; `ModelVersion` imports them so `model_config.json` always reflects the live values. Do not hardcode these numbers anywhere else in the codebase.

Embedding dims (`species_embedding_dim`, `move_embedding_dim`, etc.) live in `state_encoder.get_layout()` and flow through `features_extractor_kwargs` — same principle, different file.

**`role_input_dim` is not a module-level constant** — it is computed dynamically in `PokemonEncoder.__init__` from the layout fields and `MOVE_NET_HIDDEN`. You do not need to update it manually when dims change; it is derived correctly. The projection input dim is also auto-discovered via a dummy forward pass for the same reason.

## Phase module structure

`forward_internal` is decomposed into phase `nn.Module`s, chained by a thin orchestrator:

`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `[MoveBelief? (prefuse)]` → `TeamTransformer` →
`[BeliefHead?]` → `[MoveBelief? (default, post)]` → `CLSPool` → `[DamageOperator?]` →
`[damage_reattend? → re-attend + RE-POOL]` → `ProjectionAssembler`, then **two** root heads
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
damage to each of our mons, **appended to both projection inputs** (it doesn't enter the token stream).
Under `--opp-belief-latent-coef>0` (v18) `BeliefHead` ALSO carries an asymmetric SimSiam latent predictor
(the `latent` logits key) and `forward_internal` stashes a stop-grad `last_belief_target_latent` (the
`pokemon_encoder` role-tokens of the true hidden mons, from the training-only `belief_target_slots` obs
key) — also a side readout, never fed forward (leak-safe). See the v16 / v17 / v18 / v19 versioning notes.
A separate `WinProbHead` (v22, `win_prob_mode != none`) reads `value_pooled` *after* the pools and stashes
a `last_win_prob_logits` [B,1] — another side readout (never in pi/vf, so projection dims are unchanged),
read by the win-prob aux loss + the prober. `read_only` feeds it a STOP-GRAD `value_pooled` (head trains
its own params only); `shaping` feeds it live (the win objective also shapes the trunk). See the v22 note.

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
   self-play opponent forward pay nothing. **Optional iterative damage refinement** (v31,
   `--damage-refine-rounds N`): a `between_layers(tokens, i)` callback runs BEFORE each of the first N
   encoder layers to recompute the DamageOperator's lean discrete incoming damage from the being-enriched
   opp tokens and inject it (via the extractor's zero-init `refine_proj`) onto our-mon token positions — so
   each layer attends over physics from the freshest belief. `None` (off) ⇒ the loop is byte-identical.
   (Built by the extractor; see the v31 versioning note.)
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
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version.py` (current: `gen3_opp_hp_typed_candidates_v1` — the DamageOperator now treats the opponent's Hidden Power as the 16 ORDINARY typed move-nums 355-370 [`C = n_moves`, no synthetic appended-16] with the bare 237 masked as the presence token; a forward-math change to the op [out_dim unchanged → not shape-caught] so it bumps the signature. The literal source of truth is `ARCH_SIGNATURE` in `model_version.py`; check it there, not this prose). Pure decompositions still change state_dict keys, so old checkpoints must fail loudly. Re-sourcing or re-meaning an obs block (e.g. own IV/EV/nature going from constant fallbacks to real values via the poke-env `backfill_teambuilder_spread` fix; the event-sourced TurnDelta fold + status/item transition history; routing the trapping signals — `trapped`/`maybe_trapped`/`attempted_switch_rejected` — into the obs; the action-aligned per-move effect block — `gen3_move_effects_v1`; the per-our-mon incoming-damage / OHKO belief block — `gen3_incoming_damage_v1`; **re-calibrating that belief's VALUES** — `gen3_incoming_damage_v2`, which added a gen3 crit term + raised the offensive-stat tail to de-timid P(KO), and widened the candidate set [revealed-HP typed expansion, Return/Frustration pricing, broader prior floor/cap] so the killing move isn't silently absent; same 33 dims, values only; or adding the `turns_since_progress` no-progress-clock scalar at `vec[14]` — `gen3_markovian_progress_v1`, obs dim 3390 → 3391; or **re-ordering** the per-move features (base power vec[0:4], type multiplier vec[4:8], the move-effect block) from `battle.available_moves` order to request-slot order so feature slot k aligns with action logit 6+k — `gen3_move_slot_align_v1`, same 3409 dims, VALUES only on the disabled-move / <4-move / no-opp-active cases, byte-identical otherwise; or adding the two **protect-success-odds** reactive scalars at `vec[15]`/`vec[16]` — `gen3_protect_odds_v1`, P(Protect succeeds NOW) per active mon from `LivePokemon.protect_counter`, obs dim 3409 → 3411; or adding the two static per-move status-cure bits — `cures_self_status` (Refresh) + `cures_team_status` (Heal Bell / Aromatherapy) — to the move-effect block so the head can connect a cure move to the per-mon status one-hots, `gen3_status_cure_moves_v1`, `MOVE_EFFECT_FEATURES` 9 → 11, obs dim 3411 → 3419; or adding the 3-dim per-mon SLEEP WAKE belief block — `sleep_is_deterministic` [Rest] + a COMPUTED `p_wake` (the verified gen3 sleep-RNG tables, opp time∈{2,3,4,5} / Rest time=3 / Early Bird halves, marginalising the opp Early-Bird prior) + `sleep_counter_reliable` — so the head reads the wake odds + Rest source poke-env can't expose instead of learning the sleep RNG, `gen3_sleep_wake_belief_v1`, `POKEMON_VECTOR_DIM` 106 → 109, obs dim 3419 → 3455; or RESERVING two reactive scalars at `vec[17]`/`vec[18]` for a pending-Wish "floating heal" signal (`gen3_wish_reserve_v1`, `REACTIVE_SCALAR_DIM` 17 → 19, obs dim 3455 → 3457) then WIRING them — the gen3 Wish (gen4-inherited) heals the slot mon's `maxhp/2` at the end of the turn after cast, slot-keyed, so the per-side scalar is the flat `WISH_HEAL_FRACTION` (≈0.5, GIGO-proof) when a wish cast last turn resolves this turn, reconstructed from the event log since poke-env doesn't track it, `gen3_wish_wired_v1`, a VALUES-only change, fuzz-calibrated vs the real sim; or **re-meaning the `turns_since_progress` no-progress-clock scalar** at `vec[14]` so a REST-LOOP — our active Rested earlier this episode, woke, and re-Rested without Sleep Talk — is classified a NO_OP stall instead of a free defensive heal (it now ADVANCES the clock + charges `no_progress_tax`; Sleep-Talk mons and winning residual rest-stalls stay exempt — and, folded into the SAME signature with NO separate bump (owner decision), a WASTED Refresh (`cures_self_status` used with `our_status_cured is None`) is likewise a NO_OP charged BEFORE the progress check, so it is taxed even when a winning Leech/Toxic residual would otherwise launder the turn into "progress"), `gen3_rest_loop_stall_v1`, a VALUES-only change in `progress_clock.py`, same obs dim 3457; or adding the request-ordered **active-req-moves** block — OUR active mon's 4 moves in `legal.move_slots` order `[move_num ×4, resolved_type_id ×4, legal_now ×4]` after the matchups, so the DamageOperator's OUTGOING per-move methods (`_outgoing_block`/`_status_landing`/`_outgoing_matrix`) read request order instead of the per-mon block's sorted-by-id order and their per-move output aligns with action logit 6+k — `gen3_op_move_align_v1`, `REACTIVE_DIM` 402 → 414, obs dim 3457 → 3469, a SHAPE change; or giving each TYPED Hidden Power its OWN distinct move num — `gen3_typed_hidden_power_ids_v1`, a VALUES-only change [same obs dim 3469, no weight-shape change: 355-370 are previously-unused move-embedding rows, max_moves=400]: OUR-side HP carries a distinct num (355-370) + real type in the obs/per-num tables so the extractor's `is_hp_slot == 237` no longer matches it [normal typed-move path; our outgoing HP priced right] and the history folds the distinct num, while the OPPONENT's HP stays bare 237 with ALL its belief machinery [the HP tracker, the hp_probs blend, the op's 237→16-candidate expansion, the move-belief PRIOR+LABELS via `damage_tables._belief_num` + `gen3_env._move_num`] folded to 237 — the known/unknown boundary; supersedes the `gen3_own_hp_typed_history_v1` hp_probs one-hot [reverted]) is likewise retrain-class even when individual dims are unchanged.
- Per-phase unit tests live in `phase_modules_test.py` — `CLSPool` (incl. the `value_cls` pool) and `ProjectionAssembler` (which returns `(pi_combined, vf_combined)`) are tested on a hand-built `ExtractorContext` (`_dummy_ctx`) without a full forward pass. Prefer adding precise phase-level tests there.

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
`named_parameters()` yields (e.g. building submodules in `__init__` in a different sequence — the v40
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
(`--self-ko-hp-penalty`, the HP-scaled self-KO penalty, v12 — default 0.0 = OFF; >0 charges −w·hp when
our mon self-KOs via Explosion/Self-Destruct, since the symmetric material PBRS prices a healthy 1-for-1
trade at ~0 and the critic then over-values it), the de-bias cleanup pair `drop_redundant_bias` +
`drop_switch_bias` (`--drop-redundant-bias` / `--drop-switch-bias`, v13 — zero the audit-flagged
distorting BIAS terms: stall_tax + matchup_penalty redundant with the no-progress clock/`--draw-penalty`
and `pbrs_belief`; the hand-coded switch subsidy), and the **two end-state PBRS switches**
`all_shaping_pbrs` + `stall_pbrs` plus `no_progress_penalty` (`--all-shaping-pbrs` / `--stall-pbrs` /
`--no-progress-penalty`, v14/v15): `all_shaping_pbrs` = "everything but stall" — folds
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
old checkpoints don't fail an arch check. Current `MODEL_CONFIG_VERSION` = **42** (v42 = the turn-history
depth cut `N_HISTORY_TURNS` 10 → 7, a retrain-class obs-dim change — total obs 3469 → 2992, caught by the
`total_dim`/`n_history_turns` weight-field check, NO `ARCH_SIGNATURE` bump; see the v42 note below. v41 = the
`gen3_belief_grad_mode_v1` belief-trunk-gradient knob, a resume-immutable training hparam — see the v41 note
below; the `pbrs_roar` PBRS
above added NO new version — it rides `all_shaping_pbrs`;
see the belief +
unified-damage + unified-move + spread-belief + op-physics + status-landing + choice-band + value-dist
+ topk-incoming + damage-reattend + move-prefuse + iterative-refinement + per-move-matrices
(outgoing v34 / incoming v35; bidirectional in-trunk threat v36; transposed outgoing v39) notes below for v16–v39).

**Two probe-driven V-tail levers (v10 structural, v11 resume-immutable).** A representation probe on a
real checkpoint found the **value head is partly blind to incoming KOs the policy head sees**
(VF→"our active faints this turn" AUC **0.79** vs PI→ **0.90**, ≈ the raw-obs-linear 0.77 — i.e. the
critic isn't using the trunk's nonlinear KO reasoning), and the **TD-residual tail is fat + barely
anticipated** (r²≈0.08). Two targeted fixes, both flag-guarded default-off (clean A/B), both with the
existing `eval/td_resid_tail` as the before/after metric:
- **v10 `value_active_readout`** (`--value-active-readout`) — STRUCTURAL toggle: the dual-head value
  readout pools the whole board (`value_pooled`) but DROPS `our_active_refined`, the active-mon token
  the policy reads. This routes it into the value projection (widening it by `D_MODEL`, value head
  only — policy untouched). Versioned like `use_popart`: `check_compatible`, no `ARCH_SIGNATURE` bump
  (OFF = baseline value head byte-for-byte). `ProjectionAssembler(value_active_readout=…)`.
- **v11 `value_tail_weight`** (`--value-tail-weight` β) — resume-immutable VALUE-meaning hparam (like
  `vf_coef`, NOT weight-shape): the value loss becomes `(1-β)·MSE + β·CVaR(worst ~10% squared errors)`
  in `instrumented_ppo._value_loss_from_se`, so the critic prioritises the big over-claim craters.
  β=0 = plain MSE (byte-identical). Symmetric in error sign → V stays unbiased (GAE advantages
  unaffected). Enforced ONLY on resume via `check_value_tail_weight` (excluded from `check_compatible`
  — a frozen opponent never runs the value loss); no `ARCH_SIGNATURE` bump.

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

**Structural toggle that changes the state_dict via a flag (e.g. `opp_belief_cls_k`, v9).** The
hidden-opponent belief (`--opp-belief-cls-k`) adds the `HiddenOppBeliefPool` module — **k** distinct
learned query tokens (DETR object-query style) that read the 12 post-transformer team tokens and
summarise the belief over the opponent's still-hidden party, feeding **both** projection heads (so both
projection inputs widen by `k·D_MODEL`). **One int flag, `k=0` = off** (the cleaner surface — `k=0` is
literally the baseline state, so there's no separate on/off bool). Same versioning class as `use_popart`
(a flag that changes the state_dict): recorded on `ModelVersion`, gated in **`check_compatible`** with a
dedicated message, `MODEL_CONFIG_VERSION` bump + a `_migrate_config` `setdefault(0)`. Because it's a
plain int, **every distinct value (including `0↔N`, i.e. adding/removing the module) is a weight-shape
mismatch**, so a *single unconditional* compare gates it — no on/off conditional. **No `ARCH_SIGNATURE`
bump** — `k=0` builds no module and reproduces the baseline arch byte-for-byte (auto-discovered
projection dims stay identical), so existing checkpoints still load. It **hard-requires
`attend_unrevealed_opponents`** when `k>0` (enforced both at the CLI via `parser.error` and in
`Gen3FeaturesExtractor.__init__` via `ValueError`): with the hidden slots masked the queries would read
a board with the hidden mons deleted. `k=1` is a single "hidden-opponent CLS" set-summary; `k>1` gives
distinct per-slot queries that coordinate (decoder self-attention) and specialise. **Caveat (by
design):** without a dedicated objective (B3 — species-ID / BYOL aux head) the RL gradient only weakly
shapes these queries; this is the *structure* those objectives later attach to. Full rationale:
`designs/ai_v5/design_offense_and_opponent_belief.md` §B2.

**In-place belief slots + the B3 aux objective (`opp_belief_slots` / `--opp-belief-aux-coef`, v16).**
The live evolution of the belief idea — supersedes the `opp_belief_cls_k` side-pool. Instead of
summarising the hidden party into K side query tokens (a readout), **`BeliefSlots` fills the
un-revealed opp team slots in-place** with `TEAM_SIZE` distinct learned "unknown-mon" tokens (the
believed mask is `ctx.opp_believed_mask = species_known<0.5`, single-sourced in `ObsUnpack`), BEFORE
the transformer — so the imagined mons sit *in the lineup*, are refined by the same 12-token
`TeamTransformer`, and are attended over by every readout (`their_cls`/`value_cls`/policy) as party
members. Distinct per-slot init breaks the permutation-collapse the same way the side-pool's queries
did, in-place. **`BeliefHead`** then aux-supervises the refined opp tokens — per believed slot it
predicts the hidden mon's **species (CE) + moves (multi-label BCE)** (role implicit via the predicted
species' own embeddings); the head returns a logits **dict** so a later BYOL/latent-matching target
swaps in cleanly. Logits are stashed at `features_extractor.last_belief_logits` each forward (None
when off) and consumed ONLY by the aux loss (`InstrumentedMaskablePPO._belief_aux_loss`, folded at
`opp_belief_aux_coef`) — privileged labels never enter the forward. The forward ALSO stashes
`features_extractor.last_opp_believed_mask` (`ctx.opp_believed_mask`, `[B,6]` bool — which opp slots
are un-revealed): a read-only side stash (no effect on the forward output → off-path stays
byte-identical) so eval/forensic tooling can decode the species head's per-slot prediction for exactly
the hidden slots. `RLPlayer._decode_belief` (`inference/player.py`) reads both at trace-capture time and
`inference/belief_decode.decode_species_belief` (the inverse of `observation/belief_labels` — logit
index == national-dex num) turns them into the per-hidden-slot top-k species the eval `summary.json`'s
per-decision `belief` block shows ("what does the policy think the unrevealed mons are?"). Two version fields:
`opp_belief_slots` (bool) is the **state_dict-changing arch toggle** — gated in `check_compatible`
like `opp_belief_cls_k`, OFF = baseline byte-for-byte (NO `ARCH_SIGNATURE` bump), hard-requires
`attend_unrevealed_opponents`; `opp_belief_aux_coef` (float) is a **training-only** loss weight (like
`ent_coef`) — recorded for provenance, NOT version-locked. `--opp-belief-aux-coef>0` is the single
enable signal (auto-sets `opp_belief_slots` + forces `--attend-unrevealed-opponents`). The privileged
labels (`belief_species`/`belief_moves`) are TRAINING-ONLY Dict-obs keys emitted by `Gen3Env`
(`emit_belief_labels`, sourced from `battle2.team`; builder in `agents.observation.belief_labels`).

**Move-belief REINJECTION (`move_belief_mode` / `--move-belief-mode`, v17).** Makes the predicted
moveset *flow into* the representation instead of being a dead-end readout (the "make it meaningful"
mechanism). When `move_belief_mode != "off"`, **`MoveBelief`** runs AFTER `BeliefHead` and BEFORE the
CLS pools: per opp slot it predicts the moveset (`move_head: D_MODEL→n_moves`), **soft-embeds** it
(`sigmoid(logits) @ move_embedding` — the expected-moveset embedding), projects it back to token space
(`reinject`, small-init so the enrichment starts ≈0), ADDs it as a residual to the slot token (gated to
the slots the mode selects), and LayerNorms. The enriched `their_team_out` then feeds the CLS pools, so
**both heads reason about the believed moves**. `mode` picks which slots are enriched + scored:
`revealed` = seen mons (predict their still-UNREVEALED moves — the defensible, surprise-OHKO lever),
`unrevealed` = hidden/believed slots (omniscient; REQUIRES `--opp-belief-aux-coef>0`, else the hidden
slots are empty placeholders), `both`. The revealed-vs-unrevealed axis is the defensible-vs-omniscient A/B. Logits stash at `features_extractor.last_move_belief_logits`
(None when off), consumed ONLY by `InstrumentedMaskablePPO._move_belief_loss` (folded at
`move_belief_coef`). `move_belief_mode` (str) is the **state_dict-changing arch toggle** — gated in
`check_compatible` (string compare), OFF = baseline byte-for-byte (NO `ARCH_SIGNATURE` bump),
hard-requires `attend_unrevealed_opponents`; `move_belief_coef` (float) is a **training-only** loss
weight, recorded but NOT version-locked. Labels: `known_moves` (revealed mons' FULL privileged movesets,
direct BCE) + the shared `belief_moves` (hidden slots, Hungarian) — TRAINING-ONLY Dict-obs keys from
`Gen3Env` (builder in `agents.observation.belief_labels`).

**LATENT belief — predict identity in role-token space (`opp_belief_latent` / `--opp-belief-latent-coef`,
v18).** The BYOL/SimSiam escalation of the species head: instead of (only) a hard species CE, regress
each believed slot's refined token toward the **stop-grad `pokemon_encoder` role-token of the TRUE
hidden mon** — graded identity supervision (a "similar wall" is less wrong) in the role geometry a
representation probe found the encoder amplifies ~7.5×. ON adds an **asymmetric predictor MLP** to
`BeliefHead` (the `latent` logits key); `forward_internal` runs the model's OWN `pokemon_encoder` over a
privileged 12-slot block `[live our-team, true hidden-opp-team]` (the believed opp slots' live matchups
are already neutral → a clean identity encode) under `no_grad` and stashes the opp-half role-tokens as
`last_belief_target_latent` (detached). The TARGET rides a **training-only `belief_target_slots` [6,107]
Dict-obs key** (`Gen3Env._build_belief_target_slots`: the fresh per-mon obs encode of each hidden mon at
its believed slot, the SAME `assign_hidden_to_slots` assignment as `belief_species`, per-battle cached) —
read ONLY by the loss, NEVER concatenated into pi/vf (leak-safe; pinned by
`belief_slots_test.test_latent_target_is_no_leak`). The loss (`_belief_aux_loss`, the latent term) is the
mean cosine distance over the **same species-CE Hungarian assignment** + a **VICReg variance floor** on
the predictions (collapse guard); `aux/belief_latent_std` is the NO-GO monitor (std→0 while cosine→1 =
collapse). The discrete species head stays as the **banked fallback**. `opp_belief_latent` (bool) is the
**state_dict-changing arch toggle** — gated in `check_compatible` (bool compare), OFF = byte-for-byte (NO
`ARCH_SIGNATURE` bump), hard-requires `opp_belief_slots`; `opp_belief_latent_coef` (float) is a
**training-only** loss weight (read back on a flagless resume, like `opp_belief_aux_coef`). The id-slicing
ObsUnpack shares with the privileged encode is the value-neutral module-level `slice_pokemon_categoricals`.
This is config v18.

**Differentiable damage operator (`damage_op` / `--damage-op`, v19).** "Compute the physics, learn the
belief" (`designs/ai_v6/design_differentiable_damage_op.md`): a fixed, **differentiable** gen3 damage
calculator run in the GPU forward, fed by the move belief's PREDICTED moves. `DamageOperator`
(`features_extractor.py`) runs AFTER `MoveBelief` and reads `last_move_belief_logits` for the opp ACTIVE
slot (`w = sigmoid`), computing the believed-move incoming damage to each of our 6 mons. Output (Stage B,
`out_dim = 6·_DMG_PER_MON + _DMG_EFFECT = 78`): per defender **12** features (the **3-roll + P(KO) +
accuracy** representation, `unified-damage`) `[phys_low, phys_high, phys_crit, phys_pko, phys_acc,
spec_low, spec_high, spec_crit, spec_pko, spec_acc, p_outspeed, provenance]` — per gen3 type channel, the
0.85-roll / max-roll / ×2-crit damage as a fraction of the defender's MAX HP (damage IF it lands), the
**accuracy-discounted** P(KO this turn) vs CURRENT HP (`pko = acc·P(KO|hit)`, the exact realized KO
probability — accuracy and the damage roll are independent events), and the dominant threat's `accuracy`.
`{pko, accuracy}` together parameterize the full miss/survive/KO outcome distribution with every product
PRE-COMPUTED in the operator — so the ReLU head reasons additively and never has to learn a multiplication
(the design rationale for the whole differentiable op). The roll physics is the shared role-parameterized
kernel `DamageOperator._damage_rolls` (reused by the outgoing / safe-switch directions; named offsets
`_DMG_IDX_*`). NOT modifier-for-modifier parity: the op applies
type/STAB/ability-immunity/screens/crit but **not yet** weather, burn, defender boost stages, or
fixed-damage/OHKO/HP-relative moves — those are documented v2 follow-ups; the learned-belief gradient
story holds without them) + **6** opp-active believed-EFFECT scalars `[recovery, status,
phaze, boost, hazard, protect]` — the status/utility-threat axis the damage-only CPU block never had,
computed as a belief-weighted **MAX** over the belief × per-move effect flags (`MOVE_EFFECT_FLAGS`; a
full-axis noisy-OR over ~400 moves saturated to ~1 from the floor alone). The chip/pko
aggregation is the same **HARD max** over the channel's believed candidates (= `incoming_damage`'s
max-over-candidates; differentiable via the argmax subgradient — NOT a low-temperature soft-max, which
diluted the true max ~17× over the ~400-candidate axis). `p_outspeed` is our mon's real speed vs the opp's
fast-tail speed (a per-mon point estimate; para/boosts deferred to v2). `provenance` = the belief weight of the
dominant believed move (1≈revealed, <1=guess). The `[B,54]` block is **appended to BOTH projection
inputs**. Differentiable in `w` → the gradient sharpens the move-belief head; replaces the CPU
`incoming_damage` block's FIXED usage-prior with the LEARNED belief. **Gradient honesty:** revealed
moves are pinned to a constant `_REVEAL_LOGIT` (under prior fusion) — that `torch.where` branch carries
NO gradient, and a pinned move already contributes its (certain) damage to the channel max. So the op's
gradient sharpens the belief **only on the opp active's still-UNREVEALED candidate moves** — i.e. it
teaches the head to predict the *unseen* move that would threaten a KO, exactly the surprise-OHKO lever
the move belief exists to capture (revealed moves are already certain, nothing to learn there). **Hidden Power** (all 17 variants
collide on num=237) is expanded into **16 typed candidates** (BP 70), weighted `P(present)·P(type)` —
presence from `w[237]`, type from the obs `hp_probs` — so HP Grass vs HP Ice get distinct effectiveness.
Our defenders use their REAL spread; the hidden-spread attacker uses a fixed de-timid offense (252/31/×1.1).
Lookup tables (`damage_tables.py`, on the `TypeEncoder` axis) are **non-persistent float32 buffers**. The
block is zeroed (incl. gradient) when no opp is active + per fainted defender; no `/0`. **Leak-safe**
(reads the PREDICTED belief + public obs only) — **forward-only, no new labels/loss** (the existing
`_move_belief_loss` already supervises the belief), so `Gen3Env` is untouched.
`damage_op` (bool) is the **state_dict-changing arch toggle** — gated in `check_compatible` (bool compare,
widens both projections), OFF = baseline byte-for-byte (NO `ARCH_SIGNATURE` bump). Hard-requires
`move_belief_mode` revealed|both (enforced at extractor build + the CLI). Threaded through
`current_model_version` / `arch_toggles_from_model` (the 4 opponent-load sites). This is config v19.

**Unified two-part move belief — prior fusion (`move_prior_fusion` / `--move-prior-fusion`, v20).** Unifies
the three overlapping opponent-move systems (the Smogon move-frequency **prior**, the learned move-belief
**prediction**, and the **damage** op) into ONE posterior. When on, `MoveBelief` treats its head output as
a learned **log-odds DELTA** fused with the prior: `posterior_logit = prior_logit(species) + head_delta`,
and **pins revealed moves** (opp move-id > 0, seen this battle) to a near-certain logit (`_REVEAL_LOGIT`).
So the stashed `last_move_belief_logits` (read by BOTH the damage op AND the `_move_belief_loss` BCE) is a
true **two-part belief** — *known moves certain, unknown moves prior⊕learned* — anchored at the Smogon
base rate at cold-start, with the head learning the in-battle correction (the BCE needs no change; gradient
implicitly targets `delta ≈ logit(truth) − logit(prior)`). The prior is a `[max_species, max_moves]`
log-odds buffer (`damage_tables.build_move_prior_logits`: `logit(clamp(Σ usage over move_ids→num, floor,
1−eps))`, HP num-237 sums typed usage), registered **non-persistent** on `MoveBelief` (no new params → the
state_dict is byte-identical on/off). So `move_prior_fusion` is a **FORWARD-BEHAVIOR toggle** like
`attend_unrevealed_opponents` (NOT weight-shape): gated in `check_compatible` (a resume flip feeds a
different belief), NO `ARCH_SIGNATURE` bump, OFF = the from-scratch head byte-for-byte. Requires
`move_belief_mode != off` (enforced at extractor build + CLI); threaded through `current_model_version` /
`arch_toggles_from_model`. Note the prior is keyed on the (revealed) species — hidden slots gather the
unknown-species floor (marginalizing the prior over the species belief is a later extension). This is config v20.

**Unified-architecture ablation (`mask_incoming_damage_obs` / `--mask-incoming-damage-obs`, v21).** Lets
the unified DamageOperator **replace the model's** view of the CPU `incoming_damage` obs block, A/B-ably,
**without deleting any code**. When on, `ObsUnpack` zeros the 51-dim incoming-damage / OHKO block out of
`non_matchup_rest` (a clone — never mutates the shared obs) so the policy/value/global-token stop seeing
it; the block STAYS in the obs vector at its fixed dim, and the **reward PBRS still reads the belief from
`live_view`** (unchanged — a PBRS potential must stay a fixed, model-independent function of state). This
is the "remove the functionality from the model when using the unified arch" knob: pair it with
`--damage-op --move-prior-fusion` and A/B vs the same run without the mask to test whether the learned
belief→damage op truly subsumes the usage-prior collapse. FORWARD-BEHAVIOR toggle like
`attend_unrevealed_opponents` (no weight-shape change — just zeros an obs slice; gated in
`check_compatible`, NO `ARCH_SIGNATURE` bump, OFF byte-identical); independent of `--damage-op` (a pure
A/B knob) and threaded through `current_model_version` / `arch_toggles_from_model`. This is config v21.

**Tri-state win-probability head (`win_prob_mode` / `--win-prob-mode`, v22).** A calibrated **P(win|state)**
readout the shaped critic can't give (V is expected *shaped* return — material Φ + PBRS + terminal,
PopArt-normalised — not win odds). `WinProbHead` (`features_extractor.py`) reads the whole-board
`value_pooled` *after* the CLS pools and emits ONE logit (sigmoid ⇒ P(win)); it is supervised by the
Monte-Carlo episode OUTCOME (win=1/loss=0) propagated to every step (`instrumented_ppo._win_prob_loss`,
folded at `win_prob_coef`). The tri-state controls BOTH module + gradient: **`none`** = no module (baseline
byte-for-byte; it is a SIDE readout — stashed at `last_win_prob_logits`, NEVER concatenated into pi/vf, so
projection dims are identical on/off and the future OUTCOME label can't leak); **`read_only`** = the head
trains its OWN params on a STOP-GRAD `value_pooled` (a pure, risk-free diagnostic — zero gradient to the
trunk, verified: `grad/win_prob_share` is 0); **`shaping`** = it reads a LIVE `value_pooled` so the win
objective also shapes the shared trunk (A/B vs read_only). `win_prob_mode` is the **structural +
resume-IMMUTABLE** toggle — gated in `check_compatible` with a STRING compare so BOTH `none`↔head (a
state_dict change) AND `read_only`↔`shaping` (same params, but flipping grad-flow mid-run is a silent
training change the user chose to forbid) FATAL on a resume mismatch. OFF reproduces baseline byte-for-byte
(NO `ARCH_SIGNATURE` bump). `win_prob_coef` is **training-only** (recorded for provenance, NOT
version-locked, inherited on a flagless resume). Threaded through `current_model_version` /
`arch_toggles_from_model` (the opp-load sites) so a win-prob-ON self-play run doesn't FATAL on its own
sentinels. The label is FUTURE (only known at episode end) so — unlike the per-step belief labels — it
cannot ride as a real obs key; the training side is in `src/agents/training/CLAUDE.md` → win-probability
head.

**Unified damage system — outgoing direction + learnset gate + the 3-roll representation (`damage_outgoing`
/ `move_candidate_floor`, v23).** Collapses the three opp-move/damage systems into one and adds the
owner-requested directions/representation (`designs/ai_v6/design_unified_damage_system.md`). Three parts:
(1) **`DamageOperator._rolls`** is now the single DRY physics core — the incoming kernel `_damage_rolls`
(opp active → our 6, incl. the bench rows = the **safe-switch** read, no separate block) AND the new
**`_outgoing_block`** (our active → opp active, PER MOVE in REQUEST-slot order = action logits 6+k, so the
policy head compares move A vs B — the equal-effectiveness tie-break; our moves one-hot/legality-masked via
`ctx.move_mask`, opp defender at a NEUTRAL 0-EV bulk estimate, OPP-side screens) both call it. Per-mon
incoming feature is now **12** `[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance`
(`_DMG_IDX_*`); outgoing is **17** = 4 moves × `[low,high,crit,pko]` + `p_outspeed`. `damage_outgoing` is a
STRUCTURAL toggle like `damage_op` (widens both projections; `check_compatible` bool; OFF byte-for-byte; NO
`ARCH_SIGNATURE` bump; requires `damage_op`). (2) **`move_candidate_floor`** (float, FORWARD-BEHAVIOR like
`move_prior_fusion`): 0.0 = legacy flat 0.02-floor prior; >0 drives `build_move_prior_logits(learnset_gate=
True, floor=…)` — a **LEGALITY-only** gate: a move a species can't learn (per `gen3_data.learnset`) → ~
`logit(eps)` (impossible), a legal move keeps its **true** Smogon usage (rare-but-liftable, NOT pruned — so
surprise-move anticipation survives), a legal-unobserved move gets the small `floor` base. (3)
**`--unified-damage {off,incoming,both}`** is the one CLI knob — it desugars into
`move_belief_mode`/`damage_op`/`move_prior_fusion`/`damage_outgoing` at parse time. Both v23 fields are
threaded through `current_model_version`/`_run_arch_toggles` (the 4 opp-load sites). **Accuracy is folded
into `pko` (`acc·P(KO|hit)`) AND exposed as a per-channel scalar** — the operator does every multiplication
so the ReLU head reasons additively. Leak-safe (public obs + the predicted belief only; pinned by
`damage_op_test.test_op_is_leak_free_of_privileged_keys`). The unified directions are GPU-operator outputs
(NOT CPU obs blocks) → obs dim unchanged (3457), obs-build perf gate untouched. This is config v23.

**Unified MOVE system — the move latent + per-status secondary effects (`move_latent` /
`move_belief_latent_coef`, v24, `gen3_unified_move_system_v1`).** Three pieces (design:
`designs/ai_v6/design_unified_move_system.md`). (1) **`MoveLatentEncoder`** (a child of `PokemonEncoder`,
built when `move_latent`): a context-free per-move latent `MLP(move_emb ⊕ type_emb ⊕ MOVE_ATTR[id]) →
MOVE_LATENT_DIM` where `MOVE_ATTR` (`damage_tables.build_move_attr`, a non-persistent buffer) is the
structured "what a move does" vector (BP / category / accuracy / priority / drain / per-status secondary
chances / utility flags). It's concatenated into the move-network input (widens it → STRUCTURAL, gated in
`check_compatible` like `damage_op`; OFF byte-identical) AND its `latent_table()` is the grading target.
(2) **The latent grading** (`instrumented_ppo._move_belief_latent_loss`, weight `move_belief_latent_coef`,
training-only): the predicted move distribution's expected latent `softmax(ml) @ latent_table` is regressed
by COSINE toward the true moveset's mean latent (stop-grad) + a VICReg floor — so near-moves grade as near
(Rock Slide ≈ Hidden Power Rock), the soft complement to the per-ID BCE. Leak-safe: `last_move_latent_table`
is a side stash (never in pi/vf, `is_grad_enabled`-gated so rollout skips it); the loss reads `known_moves`
only. (3) **The DamageOperator effect block** gains per-status SECONDARY probabilities — incoming
(`_DMG_INCOMING_SEC`=10, the opp active's damaging-move secondaries, `max_m(w·chance·acc)×Serene Grace(opp)`,
NO speed coupling — flinch's move-first dependence is left to attention) + per-OUR-move outgoing
(`_DMG_OUT_SEC`=40, `chance·acc × Serene Grace(us) × Shield Dust(opp)`). These are **intrinsic to `damage_op`**
(no separate flag) → incoming_dim 78→88, outgoing 17→57; a v23 `damage_op` checkpoint won't load into v24.
New buffers: `MOVE_SECONDARY[n,10]`, `MOVE_PRIORITY`, `MOVE_DRAIN/RECOIL`, `ABILITY_SECONDARY_MULT`
(attacker Serene Grace 2×), `ABILITY_SECONDARY_BLOCK` (defender Shield Dust 0×). `move_latent` +
`move_belief_latent_coef` are threaded through `current_model_version` / `_run_arch_toggles` /
`arch_toggles_from_model` (which v24 ALSO completed for the v23 `damage_outgoing` / `move_candidate_floor`
gap — `move_candidate_floor` is now stored on the root extractor). One umbrella knob: `--unified-moves
{off,incoming,both}`. This is config v24.

**Spread/speed belief + the disable-redundant master flag (`spread_belief` / `spread_belief_coef` /
`mask_active_move_scalars_obs` / `mask_move_effects_obs`, v25, `gen3_unified_spread_belief_v1`).** The THIRD
belief leg (moves ✓, species ✓, STATS). `SpreadBelief` (a phase like `MoveBelief`, built when
`spread_belief`) predicts the opp's hidden spread — the 5 derived stats {atk,def,spa,spd,spe} — per slot:
`believed = prior_mean + delta·prior_std` from `damage_tables.build_opp_spread_prior` (`[n_species,5,2]`
usage `(mean,std)` from the Smogon spreads, non-persistent buffer) ⊕ a **zero-init** learned head (cold-start
== prior), reinjected into revealed opp tokens (small-init residual), stashed at `last_spread_belief [B,6,5]`.
The `DamageOperator` `forward` + `_outgoing_block` take a `spread_belief` arg and consume the believed opp
atk/spa/def/spd/spe (gathered at `ctx.opp_active_local`, indices `_SB_ATK.._SB_SPE`) **in place of** the
hand-coded de-timid `252/×1.1` / neutral-0-EV constants — so the op's opponent stats are a learned belief,
not a fixed guess (None → the legacy constants, byte-identical). Predicts DERIVED stats (not EVs+nature) so
the op consumes the value directly (the head stays additive). `spread_belief` is STRUCTURAL (check_compatible);
`spread_belief_coef` is training-only — the **supervision loss is now WIRED** (`gen3_unified_spread_belief_v1`):
`instrumented_ppo._spread_belief_loss` regresses the believed derived stats (`last_spread_belief`) toward the
opponent's TRUE derived stats (a privileged training-only `belief_spread`/`belief_spread_mask` label from
agent2's own team, REVEALED slots only) via a scale-normalised smooth-L1, so the head LEARNS the opponent's
hidden EV spread instead of sitting at the usage-mean prior (which over-estimates the largest-EV stat → the
op then mis-prices damage/outspeed). Metrics ride `belief/spread_*` (mae, `largest_bias`→0, n_slots); 0.0 =
OFF (byte-identical, head gets only the indirect op-damage gradient). See `src/agents/training/CLAUDE.md` →
spread-belief supervision loss. The **`--unified-obs`** master flag flips three `ObsUnpack` forward-behavior masks
(`mask_incoming_damage_obs` + `mask_active_move_scalars_obs` [move_power+multiplier, requires
`damage_outgoing`] + `mask_move_effects_obs` [the 44-dim block]) that zero a now-GPU-subsumed obs region from
the model's view (clone-once, offsets from named `reactive_layout` entries, reward/PBRS untouched). All
threaded through `current_model_version`/`_run_arch_toggles`/`arch_toggles_from_model` (the 4 opp-load sites);
OFF byte-identical (no `ARCH_SIGNATURE` bump).

**Op physics parity (v26, `gen3_unified_op_physics_v1`).** INTRINSIC to `damage_op` (no new field): the op
now folds stat-stage **boosts** (offense/defence/speed, both directions — a +2 sweeper's Atk doubles),
**burn** (½ phys Atk), **weather** (rain ×1.5 Water/×0.5 Fire; sun the reverse), **paralysis** (×0.25 speed),
and **fixed-damage** moves (Seismic Toss/Night Shade = level HP, type-immunity-gated). Values-only (no new
`check_compatible` field); the version bump marks it; validated by the constructed Showdown probe
(`damage_op_probe_fuzz_test.py`, 19/19) + the random-game net.

**Op status-landing block (v27, `gen3_unified_status_landing_v1`).** The op's OUTGOING direction gains a
per-OUR-move **status-landing** block (`_DMG_STATUS`=8: P(a dedicated status move lands vs THIS opponent) +
a `known` bit per move, request-slot order == action 6+k) — the GPU home for the masked move-effect block's
`status_will_land`, so `--mask-move-effects-obs` no longer drops that signal. `DamageOperator._status_landing`
computes `inflicts·accuracy·(1−type_immune)·(1−ability_block)·(1−already_block)·(1−sleep_block)`, where:
per-MOVE **type immunity** (Thunder Wave→Ground, Toxic/Poison Gas/Poison Powder→Steel/Poison, Will-O-Wisp
→Fire, **+ Leech Seed→Grass** — the v26-deferred item; Stun Spore/Glare para + sleep powders have NONE);
**ability immunity** (revealed opp ability → exact `ABILITY_STATUS_BLOCK`, else the Smogon-prior marginal
`SPECIES_STATUS_BLOCK_PRIOR` — Snorlax Toxic ≈0.85·(1−0.86)); **already-statused** (a major status can't
double-apply; Leech Seed can); and **Sleep Clause** — a 2nd inflicted sleep fails if ANY opp mon is asleep
via a **non-Rest** source (the per-mon `sleep_is_deterministic` from `gen3_sleep_wake_belief_v1`, reused — a
Rest self-sleep does NOT consume our cap); and **Substitute** — a Sub on the opp active blocks EVERY status
move (incl. Leech Seed), read from the public Substitute volatile in `ctx.opp_ctx_raw` (`_SUBSTITUTE_CTX_IDX`,
derived from the obs layout). The gen3 rules are imported from `gen3_mechanics`
(`STATUS_MOVE_IMMUNITY`/`ABILITY_STATUS_IMMUNITY` — one source); the tables are built by
`damage_tables.build_status_landing` (non-persistent buffers, zero new params). **Shield Dust is N/A here**
(it only scales SECONDARY effects, never a primary status move); the uncovered residual is **Yawn** + a
**Leech-Seed-already-seeded** target. INTRINSIC to `damage_outgoing` (no new field) — it grows the outgoing
output dim, so a v26 `damage_outgoing` checkpoint won't load (the SB3 `load_state_dict` shape mismatch on the
projection Linear's `in_features` — the runtime-discovered projection dim is NOT a `ModelVersion` field, so
`check_compatible` passes); OFF (no `damage_outgoing`) byte-identical (no `ARCH_SIGNATURE` bump).
`--mask-move-effects-obs` now requires **both** `--move-latent` (structural identity) AND `--damage-outgoing`
(this block).

**Op Choice Band (v28, `gen3_unified_choice_band_v1`).** The op prices **Choice Band** (×1.5 physical Atk —
the dominant damage-relevant gen3 item). **OUTGOING:** our own CB (item KNOWN → `ctx.item_ids[our_active] ==
cb_num`) ×1.5 our physical Atk **deterministically** (values-only, applied at the Atk-STAT level so the
`core = k·A+2` floor isn't boosted, composing with boosts/burn). **INCOMING:** a **CB-CONDITIONAL physical
tail** (`_DMG_CB`=13 dims appended to the incoming block) — per our 6 mons `phys_high_cb` (max-roll with the
×1.5) + `pko_cb` (P(OHKO | CB)), then a shared `p_cb` scalar (P(opp active holds CB)). `p_cb` =
`SPECIES_CB_PRIOR[species]` (the Smogon item usage prior, `damage_tables.build_species_cb_prior`, non-persistent
buffer) collapsed to **1.0** (item revealed == CB) / **0.0** (any other revealed item) / the prior (unrevealed
`item_id==0`). The CB tail is **DECORRELATED** from the modal (no-CB) line so the head weights `pko_cb·p_cb`
itself — OHKO is a nonlinear threshold a mean-field ×(1+0.5·p_cb) would blur (same provide-the-fact rationale
as the crit-split). The CB-conditional rolls reuse `_damage_rolls` (it now returns `(high, low, crit, ko,
high_cb, ko_cb)`); fixed-damage moves are CB-invariant (the override replaces them). **NOT yet modelled:** the
move-LOCK (the predictability lever) + the `ChoiceBandTracker`'s move-lock DISPROOF (a documented follow-up;
the orphaned tracker would refine `p_cb`). INTRINSIC to `damage_op` (the incoming CB block grows the incoming
output dim → a v27 `damage_op` checkpoint won't load, SB3 `load_state_dict` in_features mismatch); OFF (no
`damage_op`) byte-identical (no `ARCH_SIGNATURE` bump).

**Distributional value head (v29, `value_dist_mode` / `--value-dist-mode`).** The `WinProbHead` pattern
applied to the VALUE target — an **interpretability** side readout (design:
`designs/ai_v6/design_distributional_value_critic.md`). `ValueDistHead` reads the whole-board
`value_pooled` *after* the pools and emits `value_dist_bins` logits over a fixed atom support
`linspace(vmin, vmax, bins)`; `softmax` is the critic's predicted **return DISTRIBUTION** (sharp =
confident, wide = uncertain, bimodal = coinflip — all invisible in the scalar V), stashed at
`last_value_dist_logits` and read ONLY by the (future) aux loss + the prober — **never** in pi/vf, so
projection dims are unchanged either way (a SIDE readout, leak-safe). Tri-state like `win_prob_mode`:
`none` = no module (baseline byte-for-byte); `read_only` = the head trains its OWN params on a STOP-GRAD
`value_pooled` (risk-free diagnostic, zero trunk gradient); `shaping` = its gradient also shapes the
trunk. The `atoms` buffer is **non-persistent** (deterministic from bins+range → out of the state_dict).
Versioning: `value_dist_mode` (str) + `value_dist_bins` (int, the head's output width) are
state_dict/forward toggles gated in `check_compatible`; the support `value_dist_vmin`/`value_dist_vmax`
is value-meaning → resume-only `check_value_dist` (like `value_tail_weight`); `value_dist_coef` (float)
is the **training-only** HL-Gauss loss weight (recorded for provenance + flagless-resume read-back, NOT
version-locked, like `win_prob_coef`). OFF reproduces baseline byte-for-byte (NO `ARCH_SIGNATURE` bump);
threaded through `current_model_version` / `arch_toggles_from_model` (mode + bins) / `_run_arch_toggles`
(the 4 opp-load sites). **Phase A is complete (interpretability-only side head):** the head + versioning,
the **HL-Gauss aux loss** (`instrumented_ppo._value_dist_loss` — a Gaussian-CDF-projected soft target,
edge-tail-absorbed, CE; folded at `value_dist_coef`; the target is PopArt-normalized when the scalar
critic is, so the support lives in normalized space — see `src/agents/training/CLAUDE.md`), **trace
capture** (`RLPlayer._value_dist` → a `value_dist` npz array), the **prober** histogram + spread/PIT
(`engine.build_value_dist` / `ValueDistView`, rendered in the Summary + the `analyze` CLI), and the
**launcher** `value_dist/*` aggregate metrics. **Phase B is now BUILT (v45, `gen3_dist_critic_v1`,
`value_from_dist` / `--value-from-dist`):** the distributional head BECOMES the critic — GAE /
bootstrap / deployment read `E[Z]` (`policy._critic_value` → `ValueDistHead.mean(logits)` → `_denorm`,
same PopArt peg), the HL-Gauss CE is the PRIMARY value loss (`vf_coef` weight, not `value_dist_coef`),
and the scalar `value_net` FREEZES as a fallback (MSE term dropped; PopArt still POPs it + keeps the
μ/σ peg alive for the CE's normalized targets). WARM-STARTABLE (no state_dict change, both heads
exist; the offline probe confirmed E[Z]≈V) → RESUME-IMMUTABLE (the `belief_grad_mode` class):
recorded on `ModelVersion`, resume-only `check_value_from_dist` (+ `--allow-value-from-dist-change`
migration hatch), EXCLUDED from `check_compatible`. NO `ARCH_SIGNATURE` bump; requires
`--value-dist-mode shaping`. A POLICY kwarg (like `use_popart`); tests in `dist_critic_test.py`.
Current `MODEL_CONFIG_VERSION` = **45**.

**Discrete top-K incoming move-space (v30, `damage_topk_k` / `--damage-topk`, `gen3_unified_topk_incoming_v1`).**
The `DamageOperator`'s incoming block collapses the opp active's whole moveset into the worst phys/spec
hit per defender (`_chan_max`) — losing WHICH move it is + the per-pivot consequences, so the policy
can't anticipate the discrete move or pick the immune/safe pivot. This adds a **discrete top-K block**
(behind `damage_topk_k`, an int; 0 = off; **default 5 when `--unified-moves`** auto-enables it — a gen3
mon runs 4 moves, so the 5th is the surprise/uncertain candidate; "reason about the 4th/5th move").
For the opp active's **K most-believed CANDIDATE moves** (`torch.topk` over `w_all` — real move-nums +
16 typed HP — indices DETACHED), per move it emits: the move **LATENT** identity (gathered from the
`MoveLatentEncoder`'s candidate latent table — real ⊕ **typed-HP** rows built by
`hp_latent_block`; DIFFERENTIABLE → sharpens the latent), the belief weight `w` (DIFFERENTIABLE →
sharpens the move belief), accuracy, is_phys (`_DMG_TOPK_MOVE` = 35, an opp-property shared across
defenders), then **per OUR mon** `[high, pko, status_lands]` (`_DMG_TOPK_DMG_PER` = 3) — the
discrete-move + per-pivot read. The `high`/`pko` GATHER from the SAME raw `_damage_rolls` `[B,6,C]`
tensors the worst-case block validates (so a damage-IMMUNE pivot reads exactly 0); `status_lands`
(`_incoming_status_lands`) is the immunity-folded incoming status threat — a DEDICATED status move's
landing (type/ability/already-statused immunity at OUR defender — **Thunder Wave → a Ground pivot = 0**,
Toxic→Steel/Poison, WoW→Fire, Leech Seed→Grass) OR a damaging move's major-status SECONDARY gated by
the damage landing. **Decorrelated** (damage/status are w-independent physics; the belief gradient
rides the `w` feature + the retained `_chan_max`; the latent gradient rides the gathered latent) — the
Jensen / "provide facts, let the head weight" principle. **Meaningful-K gate:** once all 4 opp-active
moves are revealed the moveset is closed → the 5th+ slot is zeroed (nothing left to reason about).
Added ALONGSIDE the worst-case `_chan_max` summary (the differentiable-op design §4.3 hybrid — the
clean switch-SAFETY anchor + the discrete-identity detail). `out_dim` grows by `_dmg_topk_dim(K) = K·53`
→ both projections; the candidate latent table is built in `forward_internal` (UNCONDITIONALLY when
topk on, NOT `is_grad_enabled`-gated, since the op output feeds both heads in rollout) and passed to
the op as `move_latent_all`. `damage_topk_k` is a **STRUCTURAL int** toggle (gated in `check_compatible`
with an unconditional int compare, like `opp_belief_cls_k`/`value_dist_bins`; OFF = 0 byte-for-byte, NO
`ARCH_SIGNATURE` bump). Hard-requires `damage_op` + `move_latent` (enforced at the extractor + CLI).
The op stashes `last_topk_idx`/`last_topk_w` (detached side reads, never fed forward) so the prober
decodes EXACT move names. Threaded through `current_model_version` / `arch_toggles_from_model` /
`_run_arch_toggles` (the 4 opp-load sites). `decode_damage_block(..., topk_k=K)` is the SoT mirror
(`incoming_topk` = the K moves + 6×K per-defender). Leak-safe (public obs + the predicted belief only;
pinned by `damage_op_test.test_topk_leak_free`). Design:
`designs/ai_v6/design_topk_incoming_moves.md`.

**Iterative damage refinement (v31, `damage_refine_rounds` / `--damage-refine-rounds`,
`gen3_iterative_damage_v1`).** The DamageOperator runs ONCE post-transformer (a one-shot read of the FINAL
belief). This recomputes a LEAN per-our-mon incoming-damage summary BETWEEN transformer layers — as the opp
token (hence the move belief read from it) is enriched by attention — and injects it back onto our-mon
tokens, so each layer attends over physics derived from the CURRENT belief (physics-in-the-loop, not
one-shot post-hoc), and the per-round read sharpens the move-belief head. `TeamTransformer.forward` gains a
`between_layers(tokens, i)` callback (called before each layer); the extractor builds the callback when
`damage_refine_rounds > 0`. Per round it: (1) re-reads the belief via **`MoveBelief.move_logits`** (the
posterior, factored out of `forward` — no reinjection), (2) computes **`DamageOperator.discrete_incoming(ctx,
logits)` → `[B, 6, _DMG_REFINE_FEATS=4]`** = `[phys_high, spec_high, phys_pko, spec_pko]` (the lean top-K
mirror of `_damage_rolls`: select the opp active's top-`_DMG_REFINE_K`=8 most-believed candidates, reuse the
shared `_rolls` formula — ~50× cheaper than the full `[B,6,~416]` sweep, so the per-round recompute is cheap;
v1 uses the LEGACY de-timid attacker offense, NO spread/boost/burn/weather/fixed-damage — the coarse
refinement signal), (3) injects via **`refine_proj`** (`Linear(_DMG_REFINE_FEATS, D_MODEL)`, **zero-init** →
the residual is EXACTLY 0 at init = true identity-at-init, gradient still flows; NO LayerNorm on the residual
branch). `refine_proj` is weight-tied across rounds (its SHAPE is N-independent). Decorrelated: the damage
physics is w-independent, the belief gradient rides the candidate's belief weight. The full post-transformer
op is unchanged + authoritative. `damage_refine_rounds` is a **STRUCTURAL int** toggle — gated in
`check_compatible` with an unconditional int compare (like `opp_belief_cls_k`): 0↔N adds/removes `refine_proj`
(state_dict change), N↔M is a forward-behavior change; OFF (0) byte-for-byte (NO `ARCH_SIGNATURE` bump).
Hard-requires `damage_op` (which pulls in `move_belief_mode revealed|both`); NOT `move_latent`; NOT
auto-enabled by `--unified-moves` (an explicit A/B lever). Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` (the 4 opp-load sites) + both `extractor_kwargs` sites.
Design: `designs/ai_v6/design_iterative_damage_refinement.md`.

**Outgoing per-move damage matrix (v34, `damage_matrices_outgoing` / `--damage-matrices outgoing`,
`gen3_per_move_matrices_v1`).** The legacy `_outgoing_block` prices our active's 4 moves vs the opp
**ACTIVE only**; this adds **`DamageOperator._outgoing_matrix`** — our 4 moves × the opp's **6 mons**
(active + REVEALED bench), per (move, opp mon) `[low, high, crit, pko, type_mult]` + a per-opp-mon
`revealed` bit (`_DMG_OMX` = 4·6·5 + 6 = 126) — so the policy prices a KO on a **switch-in** (the
equal-effectiveness tie-break extended to bench targets). **REVEALED-gated**: an unrevealed opp slot
(`ctx.opp_believed_mask`) or fainted mon is zeroed (Gen3 has no team preview; belief-driven outgoing-vs-
unrevealed is a TODO). Reuses the validated `_outgoing_block` physics (attacker CB/boost/burn; OPP-side
screens; per-defender bulk = SpreadBelief or neutral 0-EV; boosts ONLY on the opp active slot, bench reset;
fixed-damage override) broadcast over the 6 defenders — the **active column is byte-for-byte the single-
active block** (adversarially verified). Appended LAST (existing incoming/outgoing/topk offsets untouched);
STRUCTURAL bool toggle gated in `check_compatible` like `damage_op` (widens both projections via the op
out_dim); OFF byte-for-byte (no `ARCH_SIGNATURE` bump); requires `damage_op`. `decode_damage_block(...,
matrices_outgoing=True)` mirrors the layout (`outgoing_matrix`). Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` (the 4 opp-load sites) + both `extractor_kwargs` sites. The
INCOMING-matrix enrichment is the v35 sibling below.

**Incoming per-move damage matrix (v35, `damage_matrices_incoming` / `--damage-matrices incoming`,
`gen3_per_move_matrices_v1`).** The ENRICHED evolution of the v30 top-K block (`DamageOperator._incoming_matrix`,
extending `_topk_block`) — it **REUSES `damage_topk_k` as its K** (one knob — `--damage-topk K`, try 4/5/6,
default 5 — tunes both the lean top-K and the rich matrix) and **replaces the lean top-K block** at that K
(the op suppresses the lean block when `matrices_incoming`, so they never coexist; the matrix's width is
gated by the existing `damage_topk_k` int + the `damage_matrices_incoming` bool). Per opp-active top-K move:
a richer **header** `[latent(32), belief, accuracy,
is_phys, EXPLICIT effect bits(6: recovery/status/phaze/boost/hazard/protect), EXPLICIT secondary chances(10)]`
+ a richer per-(OUR mon, move) **cell** `[low, high, crit, pko, type_mult, status_lands]` (`_DMG_IMX_HEADER`=51,
`_DMG_IMX_CELL`=6). The effect/secondary bits are **gathered PER MOVE** (`MOVE_EFFECT_FLAGS`/`MOVE_SECONDARY`
at `topk_idx`, HP rows zero-extended) — un-collapsed, the GPU home for the mid-ladder "this move phazes / this
move flinches" nuance the worst-case `p_effect`/`p_sec` maxes collapsed (those are kept-but-superseded;
physical deletion is a deferred A/B). Reuses the validated `_damage_rolls` tensors (low/high/crit/ko gathered)
+ the candidate latent table (built in rollout when matrices_incoming, like topk); `type_mult` is the
effectiveness at OUR defender's types; decorrelated (belief rides `w`, latent rides the gather). STRUCTURAL
bool toggle gated in `check_compatible` like `damage_op`; OFF byte-for-byte (no `ARCH_SIGNATURE` bump);
requires `damage_op` + `move_latent`. `decode_damage_block(..., matrices_incoming_k=K)` mirrors the layout
(`incoming_matrix`). Threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles`
+ both `extractor_kwargs` sites. The two matrices compose under `--damage-matrices both`. Design:
`designs/ai_v6/design_per_move_damage_matrices.md`.

**Transposed outgoing matrix — switch-in offense (v39, `damage_matrices_outgoing_all` /
`--damage-matrices-outgoing-all`, `gen3_per_move_matrices_v1`).** The TRANSPOSE of v34's
`_outgoing_matrix`. v34 broadens the DEFENDER axis (our active's 4 moves × the opp's 6 mons); this broadens
the ATTACKER axis — **`DamageOperator._outgoing_attacker_matrix`** prices OUR **6 mons'** 4 moves → the opp
**ACTIVE** only. The problem it fixes: `_outgoing_block` / `_outgoing_matrix` only price the CURRENT active as
the attacker, so on a **FORCED SWITCH** (our active fainted → `_outgoing_block` zeroes) the policy picks
switch-ins **BLIND to offense** (a confirmed high-impact loss source); this prices every candidate switch-in's
offense vs the opp active. Per (attacker mon, move) cell `[low, high, crit, pko]`, then a per-attacker
`p_outspeed` block + an `alive` gate bit (`_DMG_OAX` = 6·16 + p_outspeed[6] + alive[6] = **108**; layout = all
cells, then the two trailing scalar blocks). **PARITY (the load-bearing requirement):** the OUR-ACTIVE mon's
row reproduces `_outgoing_block` **byte-for-byte** (its boosts/CB/burn + the request-ordered moves + the same
opp-active defender + the same shared `_rolls` kernel — pinned by
`damage_op_test.test_outgoing_attacker_matrix_active_row_matches_single_active`, atol 1e-5). Bench rows reuse
the **identical** physics with **NEUTRAL boosts** (gen3 resets boosts on switch — mirrors `_outgoing_matrix`'s
defender-boost handling: a `[B,6]` 1.0 multiplier with the active slot's boost scattered on) + the per-mon
**sorted-by-id** moves `all_move_ids[:, :TEAM_SIZE]` (bench mons have no current-decision request order; the
active slot is OVERWRITTEN with the request slice so its row ties out). Burn/CB compose per-mon (each mon's own
KNOWN condition/item); each attacker gated by its `alive` bit; the whole block zeroed with no opp active.
STRUCTURAL bool toggle gated in `check_compatible` like `damage_op` (widens both projections via the op
out_dim); OFF byte-for-byte (no `ARCH_SIGNATURE` bump); requires `damage_op`. Appended LAST (all existing
incoming/outgoing/topk/omx/imx offsets untouched). `decode_damage_block(..., matrices_outgoing_all=True)`
mirrors the layout (`outgoing_matrix_all` → per-attacker `{moves, p_outspeed, alive}`). Threaded through
`current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites.
Design: `designs/ai_v6/design_per_move_damage_matrices.md`.

**Bidirectional in-trunk threat (v36, `gen3_bidir_threat_trunk_v1`).** Makes the threat field bidirectional
AND in-trunk (the incoming refine only injected onto OUR tokens; outgoing was heads-only). Three toggles:
- **`--threat-refine-outgoing` (#1, STRUCTURAL).** A new lean **`DamageOperator.discrete_outgoing(ctx,
  species_probs)`** → `[B,6,_DMG_OUT_REFINE=4]` (`[phys_high,spec_high,phys_pko,spec_pko]`, our active's 4
  KNOWN moves → each opp mon), injected onto the OPP token slice `[TEAM_SIZE:2·TEAM_SIZE]` via a **zero-init
  `outgoing_proj`** in the SAME `refine_cb` between-layers loop (symmetric to `refine_proj`; identity-at-init).
  Requires `damage_op` + `damage_refine_rounds>0`.
- **`--threat-unrevealed-outgoing` (#2, FORWARD-behavior).** Prices `discrete_outgoing`'s UNREVEALED opp
  columns via the EXPECTED-LATENT read: keep the slot latent, marginalize `P(species)` (per-round from the
  factored **`BeliefHead.species_logits`**, mirroring `MoveBelief.move_logits`) through `SPECIES_EXP_MULT`
  (type chart × per-species expected ability immunity, folded from `gen3_ability_priors`) + `SPECIES_SPREAD_
  PRIOR` (E[def/spd] and E[maxhp] via E[base_hp] — the sentinel species 0 has zero base stats, so EVERYTHING
  comes from the belief), **P(KO) NULLED** (a full-HP switch-in is ~never OHKO'd). Decorrelated: the gradient
  rides `P(species)` (sharpens the species belief). Requires `threat_refine_outgoing` + a belief head
  (`--opp-belief-aux-coef>0`).
- **`--threat-prob-outspeed` (#3, FORWARD-behavior).** `DamageOperator._p_outspeed` divides the speed gap by
  the believed speed STD (`SPECIES_SPREAD_PRIOR`; sigmoid≈normal-CDF, ÷ std/1.702) instead of the fixed
  `_DMG_SPEED_SCALE` — uncertainty-aware. No new params.

New buffers (non-persistent, data-built): `SPECIES_TYPE`, `SPECIES_EXP_MULT`, `SPECIES_SPREAD_PRIOR`; needs a
new data fact, **species→types** (added to the extractor → `gen3_species.json` → `SpeciesData.types`). All
three OFF byte-identical (NO `ARCH_SIGNATURE` bump); gated in `check_compatible`; threaded through
`current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites.
Tests: `bidir_threat_test.py` (kernel + identity-at-init + grad-to-P(species)) + `bidir_threat_fuzz_test.py`
(real bridge battles — finiteness + pko-null-for-unrevealed + the expected-latent prices unrevealed
defenders). Design: `designs/ai_v6/design_bidirectional_threat_trunk.md`.

**Status-landing into the trunk (v37, `gen3_status_trunk_v1`, `--threat-status-refine`).** The LAST CPU-obs
deprecation gap. `status_will_land` (board-conditional: type × ability × already-statused × Sleep-Clause ×
Substitute) was heads-only (v27 `_status_landing`). It's a computed MECHANICS fact (the class of type
effectiveness), and learning it would force attention to correlate non-local info — so we COMPUTE it and
inject BOTH directions on the refine loop via two zero-init Linears:
- **`DamageOperator.discrete_incoming_status(ctx, move_logits)`** → `[B,6,_DMG_STATUS_REFINE=2]` = the opp
  active's top-K believed status moves → per OUR mon `[P(major), P(immobilize=para/frz/slp)]` (belief-weighted
  max; gradient sharpens the move belief). Injected onto OUR tokens via `status_in_proj`.
- **`DamageOperator.discrete_outgoing_status(ctx)`** → `[B,6,2]` = our active's status moves → per opp mon
  (REVEALED-gated), the in-trunk home for `status_will_land`. Injected onto OPP tokens via `status_out_proj`.
Reuses the v27 status-landing buffers (`MOVE_INFLICTS_STATUS`/`MOVE_STATUS_CAT`/`MOVE_STATUS_TYPE_IMMUNE`/
`ABILITY_STATUS_BLOCK`/`SPECIES_STATUS_BLOCK_PRIOR`); the major-vs-immobilize split (`_IMMOBILIZE_STATUS_CATS`
= par/frz/slp) keeps the trunk signal self-contained. STRUCTURAL bool (two Linears), OFF byte-identical (NO
`ARCH_SIGNATURE` bump), gated in `check_compatible`, requires `damage_op` + `damage_refine_rounds>0`, threaded
through `arch_toggles`/`current_model_version` + both `extractor_kwargs` sites. Completes the FULL
`--unified-obs` deprecation (deprecation-gap audit: every CPU-obs signal has a GPU home; honest residuals =
opp-recovery heads-only + Rest-cure coarsening). Tests: `bidir_threat_test.py` (+7 status: T-Wave→Ground=0
both ways, immobilize⊆major, revealed-gating, identity-at-init, grad) + `bidir_threat_fuzz_test.py` (status
invariants over real battles). Current `MODEL_CONFIG_VERSION` = **37**.

**Opponent HP-type belief + UNIFIED typed-HP candidates (v38, `gen3_opp_hp_typed_candidates_v1` /
`gen3_opp_hp_type_belief_v1`, `hp_type_belief_mode` / `--hp-type-belief {off,prior,learned}`).** Fixes the
DamageOperator rendering the opponent's Hidden Power as 0-damage/**"immune"** (a prober-surfaced GIGO) by
making HP **16 ORDINARY typed moves end-to-end** — eliminating the HP special-casing that bred the prober
ambiguity ("model bug or observability bug?"). Builds on main's `gen3_typed_hidden_power_ids_v1` (the typed
move-nums 355-370 with real BP 70 + type in the damage buffers; bare 237 = BP 0).
- **The op uses the REAL typed nums 355-370 as candidates** — the candidate axis is now `C = n_moves` (the
  synthetic appended-16 expansion, the old workaround for the 237 collision, is GONE — every `cat([MOVE_*,
  HP_*])` collapsed to the buffer, value-preserving since `MOVE_*[355-370]` == the appended values, verified).
  The bare typeless 237 (BP 0) is the **presence token, ALWAYS masked** from the damage candidates.
- **`DamageOperator._opp_candidate_weights(ctx, move_belief_logits, hp_type_belief=None)`** (the SINGLE source
  for all 3 candidate sites — `forward` + the lean `discrete_incoming`/`discrete_incoming_status` refine
  kernels) masks 237 + the raw 355-370 (the non-persistent `HP_CAND_MASK` buffer) and **scatters
  `P(HP present)·P(HP type)` onto 355-370** (`w.index_add(1, HP_TYPED_NUMS, …)`, autograd-safe). P(HP present)
  = `sigmoid(belief[237])` (reveal-pinned). The type source: `off` (mode 'off') = the obs `hp_probs`
  (effectiveness-narrowed, baseline); on (`prior`/`learned`) = the learned posterior ⊕ the Smogon
  `SPECIES_HP_PRIOR` floor (`build_hp_type_prior`), NARROWED by `hp_probs` (its hard zeros are CERTAIN; an
  off-meta-survivor fallback spreads uniform so it never re-immunes). Multiple un-ruled-out types stay live
  (a distribution, not argmax) → the top-K surfaces hp-ice + hp-grass DISTINCTLY at their real nums (365/363).
- **GIGO guard** (`build_damage_buffers`, throwing): `HP_TYPED_NUMS` is data-derived (the `hiddenpower<type>`
  nums in HP_TYPE_ORDER order) and the builder asserts `MOVE_TYPE_IDX[355+j]==HP_TYPE_IDX[j]`, `MOVE_BP[237]==0`,
  `MOVE_BP[typed]==70` — fail loud if the data drifts, never scatter the belief onto the wrong move.
- **`learned`** ALSO builds **`HPTypeBelief`** (a phase module like SpreadBelief): per opp slot a 16-way
  posterior `softmax(head_delta + log prior[species])` (zero-init head → cold-start == the Smogon prior),
  stashed at `last_hp_type_logits` (fed to the op — its damage gradient sharpens it — + the CE aux). It ALSO
  **reinjects** the presence-gated expected typed-HP embedding (`hp_soft_type(posterior)` × presence, small-init
  `reinject_proj`) into the revealed opp tokens, so attention + both heads reason over the believed HP type
  (not just the op's damage block). Label: training-only `hp_type_label`/`hp_type_mask` Dict keys from agent2's
  typed move-id (`build_hp_type_labels`), CE `instrumented_ppo._hp_type_belief_loss` at `--hp-type-belief-coef`
  (metrics `belief/hptype_*`). Leak-safe: the opp's OBS HP stays typeless 237 (no leak), the typed belief +
  label are model-internal/privileged-training-only.
- **Versioning:** the op forward-math changed (out_dim + projection widths UNCHANGED — C is internal — so NOT
  shape-caught) → **`ARCH_SIGNATURE` bumped to `gen3_opp_hp_typed_candidates_v1`** so a pre-unification
  `damage_op` checkpoint fails loud rather than silently computing the old HP candidates. `hp_type_belief_mode`
  is a STRING toggle gated in `check_compatible` (off↔prior forward; prior↔learned state_dict); the obs VECTOR
  dim is unchanged (the label is a separate Dict key). Requires `damage_op`; `hp_type_belief_coef` training-only.
  Threaded through `current_model_version`/`arch_toggles_from_model`/`_run_arch_toggles` + both extractor sites.
- **Prober:** the op's top-K candidates are real move-nums → `ProbeModel._topk_move_names` decodes them via the
  NORMAL num→id path with the type preserved (`hiddenpower(ice)`), bare 237 → `hiddenpower` — no HP-special
  index→type collapse (the old ambiguity is gone).

Tests: `hp_type_belief_test.py` (the immune-bug-and-fix, 237-always-masked + C=n_moves, narrowing + off-meta
fallback, cold-start==prior, the two-distinct-typed-HPs top-K at real nums 363/365, grad flow, modes, CE, the
GIGO/version gate, the v2 reinjection) + the extended `poke_env_gaps/belief_labels_fuzz_test.py` (HP-type label
== agent2's real type + no-leak, live). `MODEL_CONFIG_VERSION` was **38** at v38; the current value is **40**
(the v40 nature/EV note below), `ARCH_SIGNATURE` = **`gen3_opp_hp_typed_candidates_v1`**.

**Nature/EV generative spread belief + op-side marginalization (v40, `gen3_nature_ev_belief_v1`,
`spread_belief_nature` / `--spread-belief-nature` + `spread_belief_nature_marginalize` /
`--spread-belief-nature-marginalize`).** Fixes the `SpreadBelief` head's "over-estimates the largest EV"
order-statistic bias (`belief/spread_largest_bias` stuck ≈ −13–30): the ADDITIVE head predicts the DERIVED stat
directly — a point estimate that sits BETWEEN the nature ×1.1/×0.9 modes. **`--spread-belief-nature`** swaps it
for a GENERATIVE head: predict a NATURE categorical ⊕ its Smogon log-prior (`build_species_nature_prior`) + a
per-stat EV ⊕ its prior (`build_species_ev_prior`) — the move/HP-type prior-fusion pattern — assume IV 31, and
**COMPUTE** `believed = (2·base + 31 + E[EV]/4 + 5)·E[nature_mult]` (`build_species_base_stats` /
`build_nature_mult`). The nature coupling (exactly one stat ×1.1, one ×0.9 — shared probability mass) + the EV
budget are now STRUCTURAL, so the head can't inflate every stat → the bias is fixed at the source. The same
`believed [B,6,5]` interface feeds the op (projection widths UNCHANGED — it enriches the opp token); the head
ALSO stashes `last_spread_nature_logits [B,6,25]` + `last_spread_ev [B,6,5]` for the loss + the op. **Supervised**
by `instrumented_ppo._nature_ev_belief_loss` (nature CE + EV smooth_l1 over REVEALED slots, folded at the SAME
`spread_belief_coef`; metrics `belief/natureev_{nature_acc,nature_ce,ev_mae}`). The privileged label is the TRUE
(nature, EVs) **deterministically INVERTED** from agent2's known `mon.stats` (`damage_tables.invert_nature_evs`,
GIGO-guarded — gen3 hides the opp nature+EVs, so we invert the visible derived stats rather than need them in the
obs), emitted by `gen3_env._spread_labels` as the training-only `belief_nature`/`belief_ev`(+masks) Dict keys
(cached per battle). **`--spread-belief-nature-marginalize`** then makes the op MARGINALISE the nonlinear P(KO)
over the believed nature distribution (`DamageOperator._nature_marg_ko`): each incoming candidate uses ONE
offensive stat (atk physical / spa special), so a 3-point quadrature over {reduce ×0.9, neither, boost ×1.1} is
EXACT — restoring the ×1.1/×0.9 asymmetry the mean-field `ko` at E[mult] blurs (a near-OHKO the believed read
prices at 0 gets its true nonzero KO risk). Differentiable in the nature posterior → the op's KO gradient also
sharpens the nature head. `spread_belief_nature` is a STRUCTURAL toggle (different SpreadBelief params; requires
`spread_belief`); `spread_belief_nature_marginalize` is a FORWARD-BEHAVIOR toggle (no new params; requires
`spread_belief_nature` + `damage_op`) — both gated in `check_compatible`, OFF byte-for-byte (NO `ARCH_SIGNATURE`
bump), threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both
`extractor_kwargs` sites. Tests: `spread_belief_test.py` (buffers, inversion round-trip, OFF byte-identical params,
cold-start==generative-prior, the nature/EV loss + skip, marg reproduces-at-neutral / shifts-under-uncertainty /
fixed-damage-invariant / forward-pko-shift, marginalize-requires-nature gate). `MODEL_CONFIG_VERSION` = **40**.

**Belief trunk-gradient mode (v41, `gen3_belief_grad_mode_v1`, `belief_grad_mode` / `--belief-grad-mode {shaping,
detached}`).** A knob on how the four STATE-prediction belief heads (`MoveBelief`, `SpreadBelief`, `HPTypeBelief`,
and the `BeliefHead` species/moves/latent aux) couple to the shared trunk. **`shaping`** (default) = the heads READ
the live trunk, so their supervised loss + the op/policy gradient through them reshape it (current behavior).
**`detached`** = each head READS a stop-grad trunk (`opp_tokens.detach()` at the logit-read, gated by a per-head
`detach_read` attr the extractor stamps; the reinject WRITE keeps the LIVE `opp_tokens` identity term, so normal
policy training still shapes the trunk) — so NO belief-originated gradient reshapes the trunk, while the belief stays
COMPUTED, REINJECTED into the forward, and CONSUMED by the op (fully "in the system"). This kills the
belief↔policy gradient interference (let attention reason over the belief, but don't let predicting hidden state
drag the trunk at the policy's expense) — the "more accurate view that can't hurt" middle ground; the
representation-rank probe (the 128-dim trunk runs in ~3–5 effective dims) says capacity isn't the constraint, so
interference is the risk this isolates. **Crucially `detach()` is value-preserving** → the FORWARD
(eval / inference / a frozen pool / distill opponent) is BIT-IDENTICAL regardless of the mode; only the TRAINING
gradient differs. So it is a **RESUME-IMMUTABLE training hparam (the `vf_coef` class)**: recorded on `ModelVersion`,
enforced ONLY on the training-resume path by `check_belief_grad_mode` — an INTENTIONAL migration is
permitted with `--allow-belief-grad-mode-change` (detach() is value-preserving so the flip is
weight-safe; loud notice, next save records the new mode) — (+ `enforce_belief_grad_mode` on
`load_model_snapshot`), and **EXCLUDED from `check_compatible` / `_WEIGHT_FIELDS`** (gating a frozen opponent on it
would be a false rejection that breaks self-play). NO `ARCH_SIGNATURE` bump (forward identical); `shaping` is
byte-for-byte the v40 forward AND backward. Threaded through `current_model_version` / `arch_toggles_from_model` /
`_run_arch_toggles` + both `extractor_kwargs` sites; the CLI flag defaults `None` → `_resolve` so a flagless resume
inherits the saved mode. Tests: `belief_grad_mode_test.py` (detached forward == shaping bit-identical; a belief loss
reshapes the trunk under shaping but ZERO trunk-grad under detached while the head still trains; spread + aux heads
also trunk-isolated; the invalid-mode guard). The win-aligned heads (`win_prob_mode` / `value_dist_mode`) keep their
own `read_only`/`shaping`. `MODEL_CONFIG_VERSION` = **41**.

**Turn-history depth cut (v42, `N_HISTORY_TURNS` 10 → 7).** A retrain-class obs-DIM change (not a
forward-math/structural one): the observation carries 7 consecutive `TurnDelta` slots (159 dims each)
instead of 10, so the turn-history block is 1113 dims (was 1590) and the total observation is **2992**
(was 3469). The constant is the single source of truth at the top of `features_extractor.py`
(imported by `model_version.py` + the observation encoder). `n_history_turns` and `total_dim` are
already in `_WEIGHT_FIELDS`, so `check_compatible` auto-rejects any pre-v42 checkpoint via the obs-dim
weight-field check — **NO `ARCH_SIGNATURE` bump** (the weight-field check already catches it). The
history-token saliency decays hard (the model reads mostly the last 1–2 turns), so the cut is a cheap
retrain free-rider, not a behavioral regression.

**Public-value aux head (v43, `gen3_pubval_aux_v1`, `pubval_mode` / `--pubval-mode`).** The
`WinProbHead` pattern with an EXOGENOUS target: `PubValHead` (a named `WinProbHead` subclass — same
architecture, its own state_dict keys) reads `value_pooled` after the pools and stashes a [B,1]
`last_pubval_logits` regressed toward the **frozen HUMAN-replay-calibrated public value V_pub**
(`agents.training.pubval` + `data/gen3_pubval.json` — 170k rated gen3ou games, held-out AUC 0.734,
turn-1 AUC 0.500 leakage-clean). Dense per-step (the trunk sees WHEN the game swung — the
credit-assignment lever) and value-INDEPENDENT (human outcomes, not the self-play bootstrap — where
the win-prob head's MC label inherits the policy's blind spots). Tri-state `pubval_mode`
{none, read_only, shaping}: `none` = no module (baseline byte-for-byte); `read_only` = head-only on a
STOP-GRAD `value_pooled` (the "can the trunk carry V_pub?" learnability probe); `shaping` = the human
positional prior shapes the shared trunk. SIDE readout — never in pi/vf, never in GAE (V^human ≠ V^π);
the target rides a training-only `pubval_target` obs key computed env-side from PUBLIC state only
(leak-free). STRUCTURAL + resume-IMMUTABLE STRING gate in `check_compatible` (like `win_prob_mode`);
`pubval_coef` training-only (flagless-resume-inherited); OFF byte-for-byte (NO `ARCH_SIGNATURE` bump);
threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both
`extractor_kwargs` sites. Training half + the parity fuzz: `src/agents/training/CLAUDE.md` →
public-replay value aux. `MODEL_CONFIG_VERSION` = **43**.

**Team-archetype latent + head FiLM (v44, `gen3_zarch_film_v1`, `zarch_film` / `--zarch-film
{off,heads}` + `zarch_dim` / `--zarch-dim`).** The amortization-gap STORAGE fix
(`designs/learning/amortization_gap_and_conditioning.md`): one shared head averages conflicting
per-team strategies (probes: per-team distillation fixed the distilled teams, did NOT lift neighbors,
and regressed the rest — cancellation made visible); FiLM conditions the heads on a learned
team-archetype code so per-team gradients land in different modulated subspaces. Two modules:
- **`ZArchEncoder`** — z_arch [B, `zarch_dim`] from OUR team's **INVARIANT** facts only (species ⊕
  item ⊕ ability ⊕ mean move-emb ⊕ the 18-dim spread block; slots 0..5 of ctx): shared atom MLP
  (`ZARCH_ATOM_HIDDEN`=64) → **DeepSets mean** over the 6 mons → LayerNorm. Properties by
  construction: **team-static** (invariant inputs + deterministic — no VIB sampling in v1: a
  per-forward reparam sample would break team-static, add VIB noise to PPO's epoch-recomputed ratio,
  and break eval determinism; the LUT-first operating point needs no rate limiter — the bottleneck IS
  the dim), **permutation-invariant** (a team is a set; one swap = a 1/6 twist), and **trunk-decoupled**
  (every embedding-table read is `.detach()`ed — recon/VICReg/FiLM gradients touch ONLY the encoder's
  own params, verified by `zarch_test.test_recon_gradient_touches_only_zarch_params`). A `recon_head`
  emits species multi-hot logits (side readout, aux-only). Leak-trivial (our own public roster).
- **FiLM at the root heads** — `film_pi`/`film_vf` (`Linear(zarch_dim, 2·PROJECTION_DIM)`, **zero-init
  weight+bias**) modulate each head's POST-projection PRE-ReLU features: `h·(1+Δγ(z)) + Δβ(z)`.
  Post-projection so `pre_proj_norm` (LayerNorm) can't wash the per-feature scale out; identity-at-init
  ⇒ ON starts byte-identical (the `refine_proj` convention); separate per-head generators (value is
  archetype-conditional in its own way — the same board is winning-for-stall / losing-for-offense).
  Downstream of every other phase (incl. the DamageOperator concat) → composes with all toggles.

Stashes: `last_zarch` (live, read by forward()'s FiLM + the aux loss), `last_zarch_recon_logits` +
`last_zarch_species_ids` (grad-gated — training epochs only). The aux loss
(`instrumented_ppo._zarch_loss`, folded at `--zarch-recon-coef` [1.0] + `--zarch-vicreg-coef` [0.1]) =
species multi-hot recon BCE (the ANTI-COLLAPSE anchor — a constant z can't reconstruct different
teams; row 0 pad zeroed) + a VICReg per-dim variance floor `relu(1−std(z, batch))` (z is LayerNorm'd
per-SAMPLE, which does not prevent cross-batch collapse). Metrics `zarch/{recon_bce, recon_topk_acc,
std, vicreg}` + **`zarch/pr`** (participation ratio of the minibatch z cloud — the LIVE LUT-vs-style
dial: near `zarch_dim` = identity-spread/LUT-leaning, low = compressed style axes, →1 = collapse) +
`film/{pi,vf}_{gamma,beta}_norm` (aliveness) + the GENERIC-vs-CONDITIONING split
`film/{pi,vf}_dev` (mean |modulation|) vs `film/{pi,vf}_team_std` (per-dim modulation std ACROSS the
minibatch's teams — the true conditioning read: the z SIGNAL is recon-supervised so it can't collapse,
but nothing supervises the generators' USE of it, and RL alone can grow them on z's team-SHARED
component [generic capacity] while the per-team differential stays weak; `team_std`≈0 with `dev`
growing = that lazy mode — distillation pressure is the sharpening lever). Coefs are TRAINING-ONLY
(flagless-resume-inherited) and **auto-zeroed on a single-team pinned-`--trainee-team` run**
(constant z ⇒ degenerate variance floor; FiLM stays on as a learned per-team bias). Versioning:
`zarch_film` (STRING) + `zarch_dim` (unconditional INT — the generators' in_features) gated in
`check_compatible`; OFF byte-for-byte (NO `ARCH_SIGNATURE` bump); `MODEL_CONFIG_VERSION` = **44**;
threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both
`extractor_kwargs` sites. Tests: `zarch_test.py` (identity-at-init forward == baseline, OFF-no-modules,
team-static + permutation invariance, gradient isolation, the aux math, the v44 gate + migration).

**Per-team LUT (v46, `gen3_zarch_lut_v1`, `zarch_lut` / `--zarch-lut {off,add,only}`).** A FREE,
unconstrained conditioning code per pinned team, layered on the v44 z_arch. **What it tests:** the
multi-team exploiter ceiling — N=1 (0.84) / N=3 (0.835) / N=10 (0.825) all distil cleanly but **N=20
stalls (~0.66)**, and the FiLM diagnosis (`designs/learning/conditioning_architectures.md` §5b) is
SNR/ill-conditioning, not capacity: the DeepSets z is COMPOSITIONAL, so z-similar teams sit at
`z̄ + ε_i` with tiny ε, and `∂L/∂J ∝ δ ⊗ ε` means the generator's gradient is proportional to that
tiny residual. A **random-init** LUT makes the per-team codes large and ~orthogonal from step 0 —
exactly the intervention that story predicts should help. If N=20 still stalls with a free code, the
ceiling is NOT conditioning signal.

- **Modules** (`zarch_lut != off`): `zarch_lut_emb` = `Embedding(n_teams + 1, zarch_dim)` — **row 0 =
  unknown, ZERO-init**; rows 1..N `normal(0, 1)` — plus `zarch_lut_norm` (LayerNorm) and the
  PERSISTENT `zarch_lut_table [n_teams, 30]` buffer. Persistent because the team↔row mapping is
  learned-state-adjacent: a reload against a different table would re-key every code.
- **z fold** (in `forward_internal`, right after the recon read): `add` → `LN(z_deepsets + code)`
  (the practical form — composition still generalizes, and an UNMATCHED team hits the zero row so z
  is EXACTLY the DeepSets z); `only` → `LN(code)` (the sharpest ablation). The recon/VICReg aux keeps
  grading the COMPOSITIONAL encoder (pre-LUT) — reconstructing a roster from a free per-team code is
  trivially satisfiable, i.e. zero anti-collapse pressure.
- **Team identity from the OBSERVATION** (`_zarch_lut_index` + `agents.model.team_signature`): sorted
  species(6) ⊕ moves(24), so **no env / eval-worker / prober / frozen-opponent plumbing changes**.
  Both blocks sorted ⇒ invariant to team and move-slot order; both invariant WITHIN a battle (species
  never changes; our own moveset never changes). **Species alone is NOT enough** — measured on the
  def-20 cluster, 5 of 20 teams share a species roster, which would silently make the "per-team" code
  a per-PAIR code; species ⊕ moves is 20/20 unique. `build_roster_table` THROWS on a duplicate
  signature or a move-set mutator (Mimic/Transform/Sketch would break within-battle invariance).
- **The GIGO canary** is `zarch/lut_hit_frac` — a signature that fails to match falls through to row
  0 (unconditioned), silently turning the experiment into a no-op that looks like "the LUT didn't
  help". On a `--trainee-teams` run it MUST sit at ~1.0. Siblings: `zarch/lut_teams_seen`,
  `zarch/lut_code_dist` (mean pairwise cosine distance between learned rows — ~1.0 at random init;
  collapsing toward 0 = the codes merged back into one shared direction).
- **Versioning:** `zarch_lut` (STRING) + `zarch_lut_teams` (unconditional INT — the Embedding height,
  and a different count re-keys every code) gated in `check_compatible`; OFF byte-for-byte (NO
  `ARCH_SIGNATURE` bump); `MODEL_CONFIG_VERSION` = **46**. Requires `--zarch-film heads` +
  `--trainee-teams` (a fixed team set to key on; a full-pool run would miss every lookup). Threaded
  through `current_model_version` / `arch_toggles_from_model` + both `extractor_kwargs` sites (the
  opponent-load path passes a SHAPE-only placeholder table — the real rosters ride the persistent
  buffer in the state_dict). Tests: `zarch_lut_test.py` (signature permutation-invariance +
  same-roster separation + the duplicate/mutator/unknown-id guards; lookup + unknown→row-0; add-mode
  unmatched == the DeepSets z, asserted by scrambling the learned rows; distinct codes at init;
  only-mode ignores z; per-row gradient isolation; the extractor build guards, OFF byte-identity,
  persistent table, and the v46 gate + migration) + the bridge fuzz
  `poke_env_gaps/team_signature_fuzz_test.py` (the live signature is CONSTANT within a real battle
  AND equals the offline table entry — verified over 1498 decisions on 5 teams incl. 3 that share a
  species roster).

**Damage re-attend (v31, `damage_reattend` / `--damage-reattend`, `gen3_damage_reattend_v1`).** Lets
attention reason OVER the computed physics — today the `DamageOperator` block is concatenated POST-pool
into pi/vf, so NO attention ever sees it (and per-candidate switch reasoning is pooled away). When on,
`forward_internal` — AFTER the op computes `damage_block` — projects the op's per-OUR-mon INCOMING rows
(`damage_block[:, :TEAM_SIZE·_DMG_PER_MON]` → `[B,6,_DMG_PER_MON]`) onto the 6 our-team tokens via a
**small-init** `reattend_proj` (std=0.02) + `reattend_norm` LayerNorm residual, runs ONE more
`TransformerEncoderLayer` (`reattend_layer`, same d_model/heads/ffn as the trunk) over the 12 team tokens
(`ctx.all_fainted` key-mask → our↔opp re-attention), then the **CLS pools are derived ONCE on the
re-attended tokens** (`our_team_pooled`/`their_team_pooled`/`our_active_refined`/`value_pooled`) — so the
pi/vf pools are **damage-AWARE board summaries** instead of damage-blind ones. **Scope (be accurate):** this
is a BOARD-level enrichment of the shared representation — it is **NOT** first-class per-candidate switch
SCORING. The re-attended bench tokens are pooled back into one `our_pool`, and the stock action head reads a
single pooled vector, so the per-bench signal to the switch logits is still the concatenated per-slot damage
block; true per-candidate scoring would need a per-bench **pointer head** (a separate follow-up). The op
runs BEFORE the pools and the pools/side-readouts/hidden-opp/assembler all read the SAME (re-attended) state
(one consistent re-pool, no stale-`value_pooled` split). **Identity-at-init**: the `reattend_layer`'s output
paths (attention out-proj + FFN second linear) are zero-init'd, so at step 0 it ≈ identity and ON starts ≈
the `damage_op` baseline (clean A/B). Re-pooling preserves the pooled shapes ⇒ **projection widths
UNCHANGED**; the only state_dict change is the 3 modules, so it's a STRUCTURAL toggle like `opp_belief_slots`
(gated in `check_compatible` with a bool compare; OFF byte-for-byte; **NO `ARCH_SIGNATURE` bump**). Requires
`damage_op` (the incoming block is the source). PopArt strongly recommended (the extra shared-trunk layer
worsens value-grad contention — a soft warning fires without `--use-popart`; watch `grad/value_policy_logratio`).
Current `MODEL_CONFIG_VERSION` = **31**.

**Move-belief pre-fuse (v32, `move_belief_prefuse` / `--move-belief-prefuse`, `gen3_move_prefuse_v1`).**
Moves the `MoveBelief` reinjection from POST-transformer to PRE-transformer. By default the move belief is
predicted + reinjected into `their_team_out` AFTER the `TeamTransformer` (the believed moves are grafted
onto the already-refined opp tokens). When on, `forward_internal` instead reinjects into the opp ROLE
tokens BEFORE the transformer (`role_tokens[:, TEAM_SIZE:]`, after `belief_slots`), so the believed moves
**co-refine** with the species/team belief through the 2 attention layers — one mon's predicted moveset can
inform (and be informed by) the rest of the board. Both call sites share one `_apply_move_belief(opp_tokens,
ctx)` helper (mask per `move_belief_mode`, prior-fusion inputs from `ctx`), so the only difference is the
input tensor + timing; `last_move_belief_logits` is stashed identically (the damage op + BCE aux still read
it). This is the **SAME `MoveBelief` module/params** → state_dict identical, projection widths unchanged,
so it's a **FORWARD-BEHAVIOR toggle** like `move_prior_fusion` (gated in `check_compatible` with a bool
compare; OFF byte-for-byte; **NO `ARCH_SIGNATURE` bump**). Requires `move_belief_mode != off` (there must be
a head to reinject). Current `MODEL_CONFIG_VERSION` = **32**.

**Frozen pre-attention move belief (v47, `move_belief_single_compute` /
`--move-belief-single-compute`, `gen3_belief_single_compute_v1`).** Computes the move belief **exactly
once** per forward and freezes it. Prefuse (v32) moved the *reinjection* before the transformer, but the
`gen3_iterative_damage_v1` refine callback still **re-read** `MoveBelief.move_logits` off the current
(reinjected → attention-enriched) opp tokens on every round — so in the production config the belief was
computed **3×** (prefuse + `damage_refine_rounds`=2 re-reads), and the refine physics consumed a
different posterior than the one attention was handed. When on, `refine_cb` reuses the stashed
`last_move_belief_logits` instead:

> belief ONCE (pre-attention) → physics ONCE → N attention layers that **cannot** revise it.

Paired with `--damage-refine-rounds 1` the callback fires only before layer 0 (on pre-attention role
tokens), so both transformer layers reason over frozen physics — the `next_run_plan.md` item-3
"prefuse-style, ONE pre-layer-1 injection, no between-layer recompute" arm. The stash is **live, not
detached**: the op's damage gradient still reaches the same belief computation the reinjection used (one
posterior, one gradient path — do NOT `.detach()` it, that would silently sever the physics→belief
training signal the op exists to provide). Also strictly cheaper — one fewer belief head pass per
forward.

**Cold-start inertness is structural, and pinned by
`belief_single_compute_test.test_identity_at_init_forward_equals_per_round`:** under
`--move-prior-fusion` `move_head` is ZERO-init (the posterior IS the Smogon prior ⇒ token-independent,
so re-reading it off enriched tokens returns the same values), and `refine_proj` is ZERO-init (the
injection is multiplied by 0). Both must train away from zero before frozen-vs-per-round can differ at
all — so enabling the flag is risk-free at step 0. If that test ever fails, one of those zero-inits
changed and the guarantee is gone.

Same `MoveBelief` module/params → state_dict identical, projection widths unchanged, so it is a
**FORWARD-BEHAVIOR toggle** like `move_belief_prefuse` (gated in `check_compatible` with a bool compare;
OFF byte-for-byte; **NO `ARCH_SIGNATURE` bump**). **Requires `move_belief_prefuse`** — without it the
only belief is computed POST-transformer, so the refine callback has nothing to reuse and the flag would
be a silent no-op; enforced at both the CLI (`parser.error`) and the extractor (`ValueError`). Threaded
through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both
`extractor_kwargs` sites. Current `MODEL_CONFIG_VERSION` = **47**.

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

The live, maintained description of the extractor is **the "Phase-by-phase data flow" section
above** plus the root `CLAUDE.md` "Feature Extractor Architecture" summary. Keep those two in
sync when you change `features_extractor.py` — layers, dims, the token sequence, the CLS
pooling, the turn-history embedding, or active-context routing.

> `designs/ai_v3/README.md` holds an old Mermaid digraph + dimension table. It is a **frozen
> ai_v3 historical record** (1309-dim obs, the pre-unified-transformer attention paths) and is
> **NOT maintained** — do not update it for current-arch changes. It carries a banner saying so.
> If a fresh visual digraph of the ai_v4 arch is ever wanted, add a new one rather than editing
> the frozen ai_v3 one.
