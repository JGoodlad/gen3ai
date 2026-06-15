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

`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `TeamTransformer` → `[BeliefHead?]` →
`[MoveBelief?]` → `CLSPool` → `[DamageOperator?]` → `ProjectionAssembler`, then **two** root heads
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
- **Any change to the phase structure or forward math is a structural change → bump `ARCH_SIGNATURE`** in `model_version.py` (current: `gen3_sleep_wake_belief_v1` — the literal source of truth is `ARCH_SIGNATURE` in `model_version.py`; check it there, not this prose). Pure decompositions still change state_dict keys, so old checkpoints must fail loudly. Re-sourcing or re-meaning an obs block (e.g. own IV/EV/nature going from constant fallbacks to real values via the poke-env `backfill_teambuilder_spread` fix; the event-sourced TurnDelta fold + status/item transition history; routing the trapping signals — `trapped`/`maybe_trapped`/`attempted_switch_rejected` — into the obs; the action-aligned per-move effect block — `gen3_move_effects_v1`; the per-our-mon incoming-damage / OHKO belief block — `gen3_incoming_damage_v1`; **re-calibrating that belief's VALUES** — `gen3_incoming_damage_v2`, which added a gen3 crit term + raised the offensive-stat tail to de-timid P(KO), and widened the candidate set [revealed-HP typed expansion, Return/Frustration pricing, broader prior floor/cap] so the killing move isn't silently absent; same 33 dims, values only; or adding the `turns_since_progress` no-progress-clock scalar at `vec[14]` — `gen3_markovian_progress_v1`, obs dim 3390 → 3391; or **re-ordering** the per-move features (base power vec[0:4], type multiplier vec[4:8], the move-effect block) from `battle.available_moves` order to request-slot order so feature slot k aligns with action logit 6+k — `gen3_move_slot_align_v1`, same 3409 dims, VALUES only on the disabled-move / <4-move / no-opp-active cases, byte-identical otherwise; or adding the two **protect-success-odds** reactive scalars at `vec[15]`/`vec[16]` — `gen3_protect_odds_v1`, P(Protect succeeds NOW) per active mon from `LivePokemon.protect_counter`, obs dim 3409 → 3411; or adding the two static per-move status-cure bits — `cures_self_status` (Refresh) + `cures_team_status` (Heal Bell / Aromatherapy) — to the move-effect block so the head can connect a cure move to the per-mon status one-hots, `gen3_status_cure_moves_v1`, `MOVE_EFFECT_FEATURES` 9 → 11, obs dim 3411 → 3419; or adding the 3-dim per-mon SLEEP WAKE belief block — `sleep_is_deterministic` [Rest] + a COMPUTED `p_wake` (the verified gen3 sleep-RNG tables, opp time∈{2,3,4,5} / Rest time=3 / Early Bird halves, marginalising the opp Early-Bird prior) + `sleep_counter_reliable` — so the head reads the wake odds + Rest source poke-env can't expose instead of learning the sleep RNG, `gen3_sleep_wake_belief_v1`, `POKEMON_VECTOR_DIM` 106 → 109, obs dim 3419 → 3455) is likewise retrain-class even when individual dims are unchanged.
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
because it is Φ_progress's weight. All are recorded on
`ModelVersion` and enforced on resume by **`check_reward_config`** (FATAL on drift, since they silently
shift the reward/objective), excluded from `check_compatible`. They are reward-VALUE changes — **no
`ARCH_SIGNATURE` bump** (the network/obs are unchanged) — so a fresh run is needed to measure them but
old checkpoints don't fail an arch check. Current `MODEL_CONFIG_VERSION` = **22** (see the belief notes
below for v16–v22).

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
`out_dim = 6·_DMG_PER_MON + _DMG_EFFECT = 54`): per defender **8** features `[phys_chip, spec_chip,
phys_pko, spec_pko, phys_crit_delta, spec_crit_delta, p_outspeed, provenance]` — the SAME 8 feature
CHANNELS as `incoming_damage.py`'s PER_MON block (NOT modifier-for-modifier parity: the op applies
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
head. Current `MODEL_CONFIG_VERSION` = **22**.

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
