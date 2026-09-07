# designs/CHANGELOG.md — how the architecture got here

**This file is HISTORY. For what is true NOW, read [`ARCHITECTURE.md`](ARCHITECTURE.md).**

Every entry below describes the state of the world *at the time it was written*. Version numbers,
dims, flag defaults and measured percentages in this file are **not** claims about the current
model — many describe blocks that have since been deleted, flags that no longer exist, and
configurations the production run does not use. Do not quote a number from this file as current
without checking it against `ARCHITECTURE.md` (and, if it matters, against the code).

Everything here was moved **verbatim** out of the root `CLAUDE.md` on **2026-08-08** as part of the
current-state/history split. Nothing was reworded, reordered, or corrected — including the parts
that were already wrong when they were moved (those are catalogued in `ARCHITECTURE.md` → *Known
contradictions between the old prose and the code*). Chronological order is the original order.

The live source of truth for versioning **mechanics** (what to bump when) is
`src/agents/model/CLAUDE.md` → Model versioning. The live values are `MODEL_CONFIG_VERSION` and
`ARCH_SIGNATURE` in `src/agents/model/model_version.py`.

---

## 1. Model-version history, v16 → v59

Moved verbatim from root `CLAUDE.md` § Model Versioning (lines 901–1600 of the pre-split file).
It begins mid-paragraph, in the middle of the sentence that named the architecture-constant single
source of truth — that is how it read, and splitting it any more cleanly would mean rewording it.


The architecture-constant single source of truth is the module-level constants
(`ROLE_TOKEN_SIZE`, `PROJECTION_DIM`, `MOVE_NET_HIDDEN`, `ROLE_ENCODER_HIDDEN`,
`ACTIVE_CTX_HIDDEN`) at the top of `features_extractor.py`; `ARCH_SIGNATURE` /
`MODEL_CONFIG_VERSION` live in `model_version.py` (current `ARCH_SIGNATURE`:
**`gen3_edge_bias_trunk_v1`** — the v56 edge-bias trunk (physics as attention edges; see the v56 entry below); before it `gen3_op_block_trim_v1` (the v55 op block trim) and **`gen3_entity_move_seats_v1`** — the v54 move-entity seats (Stage 1 of the entity generation; the v54 entry below), and before that **`gen3_typed_hp_belief_v1`** — the v52 discrete typed-HP belief, stacking directly on
`gen3_pointer_native_v1` (the v51 pointer-native action head, the fresh-generation cross-era break —
the flat positional `action_net` is deleted and every action is scored from the token of the entity it
selects; see the v51 entry below). Under it the model **only ever reasons over DISCRETE typed Hidden Power**. The
presence×type composition `P(HP_t) = presence · P(type=t)` happens ONCE, in `HPTypeBelief.compose_typed_hp`,
right beside the move-belief head; from that point on the posterior carries HP at its 16 real typed move-nums
**355-370** and the bare typeless **237** is driven hard-off (a finite `-30` logit, not `-inf`, so the BCE sees
~0 loss and no NaN). 237 survives only as the belief's internal PRESENCE channel, read immediately before it is
masked. It supersedes `gen3_opp_hp_typed_candidates_v1`, which had made only the DamageOperator typed while the
belief, its labels, its prior, the token reinjection and the latent grading still spoke in 237.
**The invariant this buys is structural**: `Σ_t P(HP_t) == presence`, and presence is reveal-pinned, so once
the opponent has been SEEN using Hidden Power the belief can be unsure WHICH type it is but can never conclude
there is none — no penalty term, no coefficient. Two certain facts eliminate candidates first: **moveset
exhaustion** (4 moves revealed, none of them HP ⇒ presence 0, derived from `opp_move_ids` alone) and
**effectiveness narrowing** (the `HiddenPowerTracker`'s hard zeros in the obs `hp_probs` are certain physics, so
the type belief is restricted to the survivors and renormalised — with a uniform-over-survivors fallback so an
off-meta HP can never be renormalised back to "immune"). The **`--hp-type-belief` mode flag is DELETED**: its
`off` state was a correctness bug behind a flag (a typeless BP-0 candidate, and a REVEALED HP priced as
nonexistent because the obs `hp_probs` it sourced the type from is empty until HP actually fires), so the head is
unconditional whenever there is a move belief — and it no longer requires `--damage-op`, since the composition
lives in the belief. The op is now a plain consumer (no `hp_type_fix`, no `SPECIES_HP_PRIOR`, no
`hp_type_belief` argument), which also closes a real divergence: `forward` used to get the learned posterior
while `refine_candidates` did not, so the between-layers refine kernels priced HP off a different belief than the
head block. The **move-belief LABELS are now the TRUE TYPED num** (`gen3_env._move_num` no longer folds to 237) —
they used to supervise a dead channel while leaving the 16 typed ones as BCE negatives, i.e. actively training
"this opponent has no Hidden Power of any type". Leak-safety is unchanged: the labels are training-only Dict keys
(the same privileged fact `hp_type_label` already carried) and the OBSERVATION still shows the opponent's HP bare,
so the model must still guess the type. **The one deliberate hold-out is the TURN-HISTORY opp-move slot**, which
keeps num 237 — the history records what was OBSERVED, and the type genuinely was not. OUR-side HP carries its
distinct num + real type in the obs/history throughout (`gen3_typed_hidden_power_ids_v1`). A data-derived `HP_TYPED_NUMS` + a throwing GIGO
guard pin the 355-370 ↔ `HP_TYPE_ORDER` alignment. The prober decodes the op's typed-HP candidates via the
NORMAL move-name path (`hiddenpower(ice)`) — no HP-special collapse. Design:
`designs/ai_v6/design_typed_hidden_power_ids.md` + the model leaf's v38 note. It supersedes
`gen3_own_hp_typed_history_v1` (the hp_probs one-hot workaround is reverted) and stacks on
`gen3_op_move_align_v1` (the request-ordered active-req-moves block — `REACTIVE_DIM` 402 → 414, obs dim
3457 → 3469) and the prior `gen3_rest_loop_stall_v1` rest-loop clock re-meaning, back through
`gen3_wish_wired_v1` — which WIRES two reactive scalars (`vec[17]` our side, `vec[18]` opp side) with the
pending-Wish "floating heal" signal. gen3 Wish (gen4-inherited) heals the RECIPIENT's `maxhp/2` at the
END of the turn after cast, slot-keyed (survives faint / Roar-phaze / switch / self-KO), duration 2,
double-Wish fails. Because the heal is the recipient's own maxhp/2, the heal fraction is ALWAYS ≈0.5, so
each scalar is a flat `WISH_HEAL_FRACTION` (0.5) when a wish cast last turn resolves this turn, else 0 —
no max-HP read, GIGO-proof. poke-env tracks none of it → reconstructed from our event log
(`wish_belief.py`); fuzz-validated vs the real sim (every actual resolve was flagged pending the turn
before). It first reserved the dims (`gen3_wish_reserve_v1`, `REACTIVE_SCALAR_DIM` 17 → 19,
obs dim 3455 → 3457) — wiring them is a VALUES-only change (same dim). It stacks on three prior obs
changes: `gen3_protect_odds_v1` (2 reactive
protect-success scalars, obs 3409 → 3411); `gen3_status_cure_moves_v1` — two static per-move bits
**cures_self_status** (Refresh) + **cures_team_status** (Heal Bell / Aromatherapy), so the head
connects a status-cure move to the per-mon status one-hots (prober-verified gap: the head routed its
own status onto Recover/switch but never the cure move), `MOVE_EFFECT_FEATURES` 9 → 11 (3411 → 3419);
and `gen3_sleep_wake_belief_v1` — a 3-dim per-mon SLEEP WAKE belief block [`sleep_is_deterministic`
(Rest), a COMPUTED `p_wake` from the verified gen3 sleep-RNG tables (opp time∈{2,3,4,5}, Rest time=3,
Early Bird halves; opp Early-Bird prior marginalised; Rest source from the event log's `[from]` clause;
fuzz-calibrated vs the real sim RNG), `sleep_counter_reliable`], `POKEMON_VECTOR_DIM` 106 → 109
(3419 → 3455). All four are retrain-class; current
`MODEL_CONFIG_VERSION`: **38** (the v33–v38 additions are the bolded entries below) — v16 added the in-place
hidden-opponent belief-aux toggle `opp_belief_slots` + its coef `opp_belief_aux_coef`, v17 the
move-belief reinjection toggle `move_belief_mode` + `move_belief_coef`, v18 the latent-belief toggle
`opp_belief_latent` + `opp_belief_latent_coef`, v19 the differentiable damage-operator toggle
`damage_op`, v20 the unified-move-belief prior-fusion toggle `move_prior_fusion`, v21 the
unified-architecture ablation toggle `mask_incoming_damage_obs`, v22 the tri-state win-probability head
`win_prob_mode` (none/read_only/shaping) + its coef `win_prob_coef`, v23 the **unified damage system** —
the OUTGOING per-move direction `damage_outgoing` (our active → opp active, action-aligned — the
equal-effectiveness move tie-break) + the LEGALITY-only move-prior gate `move_candidate_floor` (>0 drives
moves a species can't learn to ~0 while legal moves keep their true usage — rare-but-liftable, never pruned,
so surprise-move anticipation survives), both reachable via the one `--unified-damage {off,incoming,both}` knob (which
desugars into `move_belief_mode`/`damage_op`/`move_prior_fusion`/`damage_outgoing`); the op's per-mon
feature is now the **3-roll + P(KO) + accuracy** representation `[low,high,crit,pko,accuracy]×{phys,spec}`
(`pko=acc·P(KO|hit)` — the operator does the multiplicative physics so the ReLU head stays additive); none
bump `ARCH_SIGNATURE` since each OFF is byte-identical and the directions are GPU-operator outputs (obs dim
unchanged at 3457). **v24 the unified MOVE system** (`gen3_unified_move_system_v1`) — the structural
`move_latent` toggle (a context-free `MoveLatentEncoder`: a mechanics-grounded per-move latent —
move/type embeddings ⊕ a structured `MOVE_ATTR` of BP/category/accuracy/priority/drain/per-status
secondary chances — concatenated into the move network, **and** the similarity-grading target so Rock
Slide ≈ Hidden Power Rock) + its training-only grading coef `move_belief_latent_coef` (cosine of the
predicted move distribution's expected latent → the true moveset's mean latent + VICReg). v24 ALSO
enriches the `DamageOperator`'s effect block with per-status SECONDARY probabilities — incoming (the opp
active's damaging-move para/flinch/freeze, accuracy-folded, ×Serene Grace) + per-OUR-move outgoing ("what
status can this move cause, with what probability", ×our Serene Grace, ×opp Shield Dust) — **intrinsic to
`--damage-op`** (no separate flag; the secondary data is newly extracted into `gen3_moves.json`). The one
umbrella knob is `--unified-moves {off,incoming,both}` (sets `--unified-damage` + `--move-latent` +
`--move-belief-latent-coef 0.05` + `--damage-topk 5` → the incoming matrix). **Since 2026-08-04 it
DEFAULTS to `both` on a FRESH run** — the unified system is the model; a flagless RESUME inherits the
checkpoint's saved component toggles verbatim (no desugar), and an explicit `off` is the DEPRECATED
ablation baseline (warns at startup; auto-zeroes the default `--hp-type-belief-coef`, which needs a
move belief). `move_latent` OFF stays byte-identical (NO `ARCH_SIGNATURE` bump); a v23
`--damage-op` checkpoint won't load into v24 (the op's output dim grew). **v25 the SPREAD belief +
disable-redundant master flag** (`gen3_unified_spread_belief_v1`) — `--spread-belief` (the THIRD belief
leg: predicts the opp's hidden SPREAD = 5 derived stats per slot from a usage prior ⊕ a learned head,
reinjected into the opp token, so the `DamageOperator` consumes BELIEVED opp stats instead of its
hand-coded de-timid/neutral constants) + its training-only `--spread-belief-coef` (speed supervision from
observed move order — flag wired, loss staged); and `--unified-obs`, ONE master switch that zeros the
now-GPU-subsumed CPU obs regions from the model's view (incoming-damage + active-move scalars + move-effect
block; granular `--mask-*-obs` underneath, reward/PBRS untouched). Pure-unified run = `--unified-moves both
--spread-belief --unified-obs`. OFF byte-identical. **v26 op-physics parity** (`gen3_unified_op_physics_v1`,
intrinsic to `--damage-op`, values-only) — the op now folds stat-stage boosts/burn/weather/paralysis +
fixed-damage moves (validated by the constructed Showdown probe `damage_op_probe_fuzz_test.py`, 19/19).
**v27 op status-landing** (`gen3_unified_status_landing_v1`, intrinsic to `--damage-outgoing`) — the op's
OUTGOING direction gains a per-OUR-move STATUS-LANDING block (8 dims: P(a dedicated status move lands vs THIS
opponent — Toxic/Will-O-Wisp/Thunder Wave/Spore/**Leech Seed**) + a `known` bit), the GPU home for the masked
move-effect `status_will_land`. Folds accuracy × per-MOVE type immunity (incl. the v26-deferred **Leech Seed
→Grass**) × ability immunity (revealed→exact, else the Smogon prior) × already-statused × **Sleep Clause** (a
2nd inflicted sleep fails; a Rest self-sleep does NOT consume our cap, reusing `sleep_is_deterministic`) ×
**Substitute** (a Sub blocks every status move incl. Leech Seed, read from the public volatile). gen3 rules
imported from `gen3_mechanics` (one source); Shield Dust is N/A (it only scales SECONDARY effects). A v26
`--damage-outgoing` checkpoint won't load (SB3 `load_state_dict` projection in_features mismatch — the dim is
runtime-discovered, not a `check_compatible` field). `--mask-move-effects-obs` now requires `--move-latent`
AND `--damage-outgoing`. **v28 op Choice Band** (`gen3_unified_choice_band_v1`, intrinsic to `--damage-op`) —
the op prices CB (×1.5 physical Atk): OUTGOING applies our own (known) CB ×1.5 deterministically; INCOMING
exposes a per-our-mon CB-CONDITIONAL physical tail (`phys_high_cb` + `P(OHKO|CB)`) + a shared `p_cb`
(P(opp holds CB) — a species usage prior collapsing to 0/1 on item reveal), **decorrelated** so the head
weights them (OHKO is a nonlinear threshold a mean-field blend would blur). Move-lock + the ChoiceBandTracker
disproof are a follow-up. **v29 the distributional VALUE head** (`gen3` interpretability side readout,
`value_dist_mode` none/read_only/shaping + `value_dist_bins`) — `ValueDistHead` reads `value_pooled` and
emits per-atom return-distribution logits (softmax = the critic's predicted return distribution; sharp =
confident, wide = uncertain, bimodal = coinflip), a SIDE readout stashed for the prober + a future aux
loss, **never in pi/vf** (projection dims unchanged → OFF byte-identical, no `ARCH_SIGNATURE` bump); mode +
bins gated in `check_compatible`, the support (vmin/vmax) resume-only. Phase-A foundation (head +
versioning); the distributional aux loss + capture/prober/launcher are follow-ons.
**v30 the DISCRETE top-K incoming move-space** (`gen3_unified_topk_incoming_v1`, `damage_topk_k` /
`--damage-topk`; the LEAN block described here is SUPERSEDED by v35's `_incoming_matrix` and DELETED at
v55 `gen3_op_block_trim_v1` — `damage_topk_k` now sizes only the matrix) — the `DamageOperator`'s incoming block collapses the opp active's whole moveset into the
worst phys/spec hit per defender (`_chan_max`), hiding WHICH move it is + the per-pivot consequences. This
adds a discrete block: for the opp active's **K most-believed CANDIDATE moves** (default K=5, auto-on under
`--unified-moves`; a mon runs 4 moves so the 5th is the surprise candidate) it surfaces — per move — its
move **LATENT** identity (gathered from the `MoveLatentEncoder`, incl. **typed-HP** rows so HP-Rock ≠
HP-Ice; differentiable → sharpens the latent) + belief weight (differentiable → sharpens the move belief)
+ accuracy + is_phys, then **per OUR mon** `[high, pko, status_lands]` — so the policy can anticipate the
discrete move AND pick the immune/safe pivot (damage-immune pivot = 0 from the chart; status-immune pivot,
e.g. **Thunder Wave → a Ground mon**, = 0 via `_incoming_status_lands`). Decorrelated physics (the belief
gradient rides the `w` feature, not the damage); the 5th slot is zeroed once all 4 opp moves are revealed;
added ALONGSIDE the worst-case `_chan_max` summary (the §4.3 hybrid). STRUCTURAL int (scales `out_dim` by
`K·53` → both projections; gated in `check_compatible` like `opp_belief_cls_k`; OFF=0 byte-identical, no
`ARCH_SIGNATURE` bump); requires `--damage-op` + `--move-latent`; threaded through `arch_toggles`; the
prober decodes exact move names from the stashed `last_topk_idx`.
**v31 the DAMAGE RE-ATTEND** (`gen3_damage_reattend_v1`, `damage_reattend` / `--damage-reattend`) — lets
attention reason OVER the computed physics (today the `DamageOperator` block is a POST-pool concat no
attention sees). When on, after the op computes the damage, its per-OUR-mon INCOMING rows are projected
(small-init, identity-at-init) onto the 6 our-team tokens, ONE more `TransformerEncoderLayer` re-attends the
12 team tokens (our↔opp), and the CLS pools are derived ONCE on the re-attended tokens — so the pi/vf pools
are **damage-AWARE board summaries** instead of damage-blind ones. It is a BOARD-level enrichment (the
"needs a per-bench pointer head" follow-up it originally deferred landed at v51 — the pointer head reads
the re-attended `our_team_out` per token, so a bench token now flows straight into its own switch
logit). STRUCTURAL like `opp_belief_slots` (adds 3 modules; re-pooling keeps
the pooled shapes ⇒ projection widths UNCHANGED; gated in `check_compatible`, OFF byte-identical, NO
`ARCH_SIGNATURE` bump); requires `--damage-op`; threaded through `arch_toggles`; PopArt strongly recommended
(soft-warns without it).
**v32 the MOVE-BELIEF PRE-FUSE** (`gen3_move_prefuse_v1`, `move_belief_prefuse` / `--move-belief-prefuse`) —
moves the `MoveBelief` reinjection from POST-transformer (the default — believed moves grafted onto the
already-refined opp tokens) to PRE-transformer (reinjected into the opp ROLE tokens before the body), so the
predicted moves **co-refine** with the species/team belief through the 2 attention layers. Same `MoveBelief`
module/params (one shared `_apply_move_belief` helper, only the input tensor + timing differ; the stashed
`last_move_belief_logits` is identical, so the damage op + BCE aux still read it) → state_dict identical,
projection widths unchanged. FORWARD-BEHAVIOR toggle like `move_prior_fusion` (gated in `check_compatible`,
OFF byte-identical, NO `ARCH_SIGNATURE` bump); requires `--move-belief-mode != off`; threaded through
`arch_toggles`.
**v33 ITERATIVE damage refinement** (`gen3_iterative_damage_v1`, `damage_refine_rounds` /
`--damage-refine-rounds N`) — the `DamageOperator` runs ONCE post-transformer (a one-shot read of the FINAL
belief). This recomputes a LEAN per-our-mon incoming-damage summary BETWEEN transformer layers — as the opp
token (hence the move belief) is enriched by attention — and injects it back onto our-mon tokens, so each
layer attends over physics from the FRESHEST belief (physics-in-the-loop), and the per-round read sharpens
the move-belief head. `TeamTransformer.forward` gains a `between_layers` callback (before each of the first
N layers); per round it re-reads the belief (`MoveBelief.move_logits`, the posterior — factored out of
`forward`), computes a LEAN `DamageOperator.discrete_incoming → [B,6,4]` `[phys_high, spec_high, phys_pko,
spec_pko]` (top-`_DMG_REFINE_K`=8 candidates, reusing the validated `_rolls` physics — ~50× cheaper than the
full ~416 sweep, so the per-round recompute is cheap), and injects via a **zero-init `refine_proj`** Linear
(true identity-at-init, gradient still flows; weight-tied across rounds → N-independent shape). STRUCTURAL int
gated in `check_compatible` (0↔N a state_dict change, N↔M a forward change; OFF=0 byte-identical, no
`ARCH_SIGNATURE` bump); requires `--damage-op` only (NOT `--move-latent`); NOT auto-set by `--unified-moves`
(an explicit A/B lever); threaded through `arch_toggles` + both extractor-kwargs sites.
**v34 the OUTGOING per-move DAMAGE MATRIX** (`gen3_per_move_matrices_v1`, `damage_matrices_outgoing` /
`--damage-matrices outgoing`) — the legacy outgoing block prices our active's 4 moves vs the opp ACTIVE
only; this adds `DamageOperator._outgoing_matrix`: our 4 moves × the opp's **6 mons** (active + REVEALED
bench), per (move, opp mon) `[low,high,crit,pko,type_mult]` + a per-opp-mon `revealed` bit — so the policy
prices a KO on a **switch-in** (the equal-effectiveness tie-break extended to bench targets). REVEALED-gated
(unrevealed opp slots zeroed — Gen3 has no team preview; belief-driven outgoing-vs-unrevealed is a TODO);
reuses the validated `_outgoing_block` physics broadcast over 6 defenders (the active column is byte-for-byte
the single-active block). STRUCTURAL bool toggle gated in `check_compatible` like `damage_op`; OFF
byte-identical (no `ARCH_SIGNATURE` bump); requires `--damage-op`; threaded through `arch_toggles` + both
extractor-kwargs sites.
**v35 the INCOMING per-move DAMAGE MATRIX** (`gen3_per_move_matrices_v1`, `damage_matrices_incoming` /
`--damage-matrices incoming`) — the ENRICHED evolution of the v30 top-K block (`_incoming_matrix`,
REUSES `--damage-topk K` as its K — one knob, try 4/5/6; since v55 deleted the lean top-K it superseded,
this is the ONLY block K sizes).
Per opp-active top-K move: a richer header `[latent, belief, acc,
is_phys, EXPLICIT effect bits(6: recovery/status/phaze/boost/hazard/protect), EXPLICIT secondary chances(10)]`
+ a richer per-(OUR mon, move) cell `[low,high,crit,pko,type_mult,status_lands]`. The effect/secondary bits
are **gathered PER MOVE** (un-collapsed — the mid-ladder "this move phazes / flinches" nuance the worst-case
`p_effect`/`p_sec` maxes collapsed; those are kept-but-superseded, deletion deferred to an A/B). Reuses the
validated `_damage_rolls` tensors + the candidate latent table; STRUCTURAL bool gated in `check_compatible`
like `damage_op`; OFF byte-identical; requires `--damage-op` + `--move-latent`. The two matrices compose
under `--damage-matrices {off,incoming,outgoing,both}`.
**v36 the BIDIRECTIONAL in-trunk THREAT field** (`gen3_bidir_threat_trunk_v1`) — makes the model's threat,
BOTH directions, dynamic (known⊕believed) and INFUSED INTO THE TRUNK so attention reasons over it. Three
toggles: **`--threat-refine-outgoing`** (#1) the SYMMETRIC mirror of the incoming refine — a new lean
`DamageOperator.discrete_outgoing` (our active's 4 known moves → each opp mon → `[phys_high,spec_high,
phys_pko,spec_pko]`) injected onto the OPP token slice via a **zero-init `outgoing_proj`** riding the SAME
between-layers `--damage-refine-rounds` loop (STRUCTURAL — a saved weight; requires `--damage-op` +
`--damage-refine-rounds>0`); **`--threat-unrevealed-outgoing`** (#2) the EXPECTED-LATENT defender — keep an
UNREVEALED opp mon LATENT and marginalize the move-belief's `P(species)` (read per-round from the factored
`BeliefHead.species_logits`) through `SPECIES_EXP_MULT[n_species,19]` (type chart × the per-species expected
ability immunity — Levitate/Water&Volt Absorb/Flash Fire/Thick Fat, folded from `gen3_ability_priors`) +
`SPECIES_SPREAD_PRIOR` (E[bulk]/E[maxhp]), with **P(KO) NULLED** (a full-HP switch-in is ~never OHKO'd —
owner decision, drops the Jensen-threshold complexity; forward toggle, no new params; requires
`--threat-refine-outgoing` + `--opp-belief-aux-coef>0`); **`--threat-prob-outspeed`** (#3) UNCERTAINTY-AWARE
`P(outspeed)` — divide the speed gap by the believed speed STD (`SPECIES_SPREAD_PRIOR`; sigmoid≈normal-CDF)
not a fixed scale (forward toggle, no new params). Needs a NEW data fact — **species→types** (added to the
extractor → `gen3_species.json` → `SpeciesData.types`; the obs still reads revealed types live). All three
OFF byte-identical (NO `ARCH_SIGNATURE` bump), version-gated, threaded through `arch_toggles` + both
extractor sites.
**v37 STATUS-LANDING into the trunk** (`gen3_status_trunk_v1`, `threat_status_refine` /
`--threat-status-refine`) — the LAST CPU-obs deprecation gap. The move-effect block's board-conditional
`status_will_land` was heads-only (v27 `_status_landing`); status immunity (type × ability ×
already-statused × Sleep-Clause × Substitute) is a computed MECHANICS fact (the class of type
effectiveness), and LEARNING it would force attention to correlate non-local info (the move's status intent
on one token, the defender's types+ability on another). So COMPUTE it and inject into the trunk, BOTH
directions: **INCOMING** `discrete_incoming_status` (opp active's top-K believed status moves → per OUR mon,
onto OUR tokens — "will I be statused") + **OUTGOING** `discrete_outgoing_status` (our active's status moves
→ per opp mon, revealed-gated, onto OPP tokens — the in-trunk home for the masked `status_will_land`), each
a per-defender `[P(major), P(immobilize=para/frz/slp)]` reusing the v27 status-landing physics + buffers via
two zero-init residuals on the refine loop. The major-vs-immobilize split makes the trunk signal
SELF-CONTAINED (no cross-move correlation). STRUCTURAL bool (adds two Linears); OFF byte-identical (NO
`ARCH_SIGNATURE` bump); requires `--damage-op` + `--damage-refine-rounds>0`; threaded through `arch_toggles`
+ both extractor sites. **Completes the FULL `--unified-obs` deprecation** (verified by a deprecation-gap
audit: every CPU-obs signal has a GPU home — damage→trunk/refine, status→trunk/v37, effects→move latent, PP
→per-mon slot, provenance/p_outspeed/crit→explicit op channels, per-move status_will_land+known→v27 heads;
honest residuals = opp-recovery heads-only + Rest-cure coarsening). The dedicated `pbrs_roar`
phaze-out-boosts PBRS is folded INTO `--all-shaping-pbrs` (no new flag/version, no `ARCH_SIGNATURE` bump).
At this point in the history `MODEL_CONFIG_VERSION` was **39** (the v38/v39 additions are the bolded
entries below); the CURRENT value is at the end of this section. Full design:
`designs/ai_v6/design_bidirectional_threat_trunk.md` (+ `gen3ai/tmp/{model_v36_full,stacking_levels}.png`)
(and `design_per_move_damage_matrices.md` for v34/v35, `design_iterative_damage_refinement.md` for v33,
`design_topk_incoming_moves.md` for v30, `design_distributional_value_critic.md` for v29,
`design_unified_move_system.md` for v24, `design_unified_damage_system.md` for v23).
**v38 UNIFIED typed-HP candidates + the opponent HIDDEN-POWER-TYPE belief** (SUPERSEDED by v52
`gen3_typed_hp_belief_v1` — the `--hp-type-belief` mode flag and the op-side scatter described here are GONE;
kept for the history of how the "immune" GIGO was first attacked) (`hp_type_belief_mode` /
`--hp-type-belief {off,prior,learned}`) — fixed the
DamageOperator rendering the opponent's Hidden Power as 0-damage/**"immune"** (a prober-surfaced GIGO) by
making HP **16 ordinary typed moves end-to-end**, eliminating the HP special-casing that bred the prober
ambiguity. Builds on main's `gen3_typed_hidden_power_ids_v1` (typed move-nums **355-370** with real BP 70 +
type; bare 237 = BP 0): the op now treats the opp's HP as those **real typed candidates** — the candidate axis
is `C = n_moves` (the synthetic appended-16 expansion, the old 237-collision workaround, is REMOVED), the bare
237 (BP 0) is the masked **presence token**, and the per-type HP belief is scattered onto 355-370. A shared
`DamageOperator._opp_candidate_weights` (the single source for all 3 candidate sites) masks 237 + the raw
355-370 (`HP_CAND_MASK`) and `index_add`s `P(HP present)·P(HP type)` onto `HP_TYPED_NUMS`. Type source —
**`off`**: the obs `hp_probs` (effectiveness-narrowed, the A/B baseline); **`prior`**: the Smogon
`SPECIES_HP_PRIOR` floor (`build_hp_type_prior`); **`learned`**: the `HPTypeBelief` head's posterior
`softmax(head_delta + log prior[species])` (zero-init → cold-start == prior), which the op consumes (its damage
gradient sharpens it) AND which **reinjects** the presence-gated expected typed-HP embedding into the opp token
(attention reasons over the believed type), supervised by a training-only CE
(`instrumented_ppo._hp_type_belief_loss`, `--hp-type-belief-coef`, metrics `belief/hptype_*`) against the
privileged true HP type from agent2's typed move-id (`Gen3Env._hp_type_labels`; the obs keeps the opp HP
typeless 237 → no leak). All on/off the belief NARROWS by the obs `hp_probs` (its effectiveness hard-zeros are
CERTAIN; an off-meta-survivor fallback spreads uniform so it never re-immunes). Multiple un-ruled-out types
stay live (a distribution, not argmax) → the top-K surfaces **hp-ice + hp-grass distinctly at their real nums
(365/363)** with real per-mon damage — the "force the model to guess which HP, simulate each" read. A
data-derived `HP_TYPED_NUMS` + a throwing GIGO guard pin the 355-370 ↔ `HP_TYPE_ORDER` alignment
(`MOVE_TYPE_IDX[355+j]==HP_TYPE_IDX[j]`, `MOVE_BP[237]==0`). The prober decodes the op's typed-HP candidates via
the NORMAL move-name path (`hiddenpower(ice)`) — no HP-special index→type collapse (the old ambiguity is gone).
The op's forward-math changed (out_dim + projection widths UNCHANGED — C is internal — so NOT shape-caught) →
the `ARCH_SIGNATURE` **bump** forces a clean reload of any pre-unification `damage_op` checkpoint;
`hp_type_belief_mode` is STRING-gated in `check_compatible`, the obs VECTOR dim is unchanged (the label is a
separate Dict key), `hp_type_belief_coef` is training-only. Requires `--damage-op`; threaded through
`current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites.
**v39 the TRANSPOSED outgoing DAMAGE MATRIX — switch-in offense** (`gen3_per_move_matrices_v1`;
`damage_matrices_outgoing_all` / `--damage-matrices-outgoing-all`) — the TRANSPOSE of v34's
`damage_matrices_outgoing`. v34 prices our ACTIVE's 4 moves × the opp's 6 mons (broadening the DEFENDER axis);
v39 broadens the ATTACKER axis: `DamageOperator._outgoing_attacker_matrix` prices OUR **6 MONS'** 4 moves → the
opp **ACTIVE** only. Fixes a confirmed high-impact error: today the op's outgoing block prices ONLY the current
active attacker, so on a **forced switch** (active fainted → the single-active block zeroes) the policy picks
switch-ins **BLIND to offense** — this surfaces what every candidate switch-in would DO to the opp active. Per
(attacker mon, move) cell `[low,high,crit,pko]` + a per-attacker `p_outspeed` + an `alive` bit (`_DMG_OAX` =
6·16 + 6 + 6 = **108**). **PARITY (the hard requirement):** the OUR-ACTIVE mon's row reproduces `_outgoing_block`
**byte-for-byte** (its boosts/CB/burn + request-ordered moves + the same opp-active defender + the same `_rolls`
kernel); bench rows reuse the SAME validated physics with **NEUTRAL boosts** (gen3 resets boosts on switch) +
the per-mon sorted-by-id moves (the active slot is overwritten with the request slice so it ties out). STRUCTURAL
bool toggle gated in `check_compatible` like `damage_op` (widens both projections via the op out_dim); OFF
byte-identical (NO `ARCH_SIGNATURE` bump); requires `--damage-op`. Appended LAST (all prior op offsets
untouched); `decode_damage_block(..., matrices_outgoing_all=True)` mirrors it (`outgoing_matrix_all`). Threaded
through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs`
sites. Design: `designs/ai_v6/design_per_move_damage_matrices.md`.
**v40 the NATURE/EV GENERATIVE spread belief + op nature-marginalization** (`gen3_nature_ev_belief_v1`;
`spread_belief_nature` / `--spread-belief-nature` + `spread_belief_nature_marginalize` /
`--spread-belief-nature-marginalize`) — fixes the `SpreadBelief` head's "over-estimates the largest EV"
order-statistic bias (`belief/spread_largest_bias`). The additive head predicts the DERIVED stat directly (a
point estimate sitting BETWEEN the nature ×1.1/×0.9 modes); **`--spread-belief-nature`** swaps it for a
GENERATIVE head — predict a NATURE categorical ⊕ Smogon log-prior + per-stat EVs ⊕ prior (the move/HP-type
prior-fusion pattern), IV 31, and **COMPUTE** `believed = (2·base + 31 + E[EV]/4 + 5)·E[nature_mult]`. The nature
coupling (one stat ×1.1, one ×0.9) + the EV budget become STRUCTURAL, so the head can't inflate every stat. Same
`believed [B,6,5]` op interface (projection widths UNCHANGED); supervised by nature CE + EV smooth_l1
(`_nature_ev_belief_loss`, folded at `spread_belief_coef`, metrics `belief/natureev_*`) against the TRUE
(nature, EVs) **deterministically INVERTED** from agent2's known `mon.stats` (`invert_nature_evs`, GIGO-guarded;
training-only `belief_nature`/`belief_ev` Dict keys — gen3 hides the opp nature/EVs so no leak).
**`--spread-belief-nature-marginalize`** then makes the op MARGINALISE the nonlinear P(KO) over the believed
nature distribution (3-point quadrature on each candidate's one offensive stat — EXACT — restoring the
asymmetry the mean-field `ko` at E[mult] blurs). `spread_belief_nature` STRUCTURAL (requires `--spread-belief`);
`marginalize` FORWARD-BEHAVIOR (requires it + `--damage-op`); both version-checked, OFF byte-identical (NO
`ARCH_SIGNATURE` bump). Smoke: `nature_acc` rises + `largest_bias` trends to 0.
**v41 the BELIEF TRUNK-GRADIENT MODE** (`gen3_belief_grad_mode_v1`; `belief_grad_mode` /
`--belief-grad-mode {shaping, detached}`) — a knob on whether the four STATE-prediction belief heads
(move / spread / hp-type / the species-moves-latent aux) reshape the shared trunk. `shaping` (default) =
they READ the live trunk so their gradient reshapes it (current behavior); `detached` = they READ a
STOP-GRAD trunk (`opp_tokens.detach()` at the logit-read; reinject WRITE keeps the live identity term) so
NO belief gradient reshapes the trunk — the belief stays computed / reinjected / consumed by the op (fully
"in the system"), it just can't drag the trunk toward predicting hidden state at the policy's expense
(kills the belief↔policy gradient interference). `detach()` is value-preserving → the FORWARD (eval /
frozen pool / distill opponent) is BIT-IDENTICAL; only the TRAINING gradient differs. So it is a
**RESUME-IMMUTABLE training hparam** (the `vf_coef` class): recorded on `ModelVersion`, enforced
resume-only via `check_belief_grad_mode` (intentional migration: `--allow-belief-grad-mode-change`),
EXCLUDED from `check_compatible` (a frozen opponent's forward is
unaffected, so gating it would break self-play). NO `ARCH_SIGNATURE` bump; `shaping` is byte-for-byte the
v40 forward+backward. The win-aligned heads (`--win-prob-mode` / `--value-dist-mode`) keep their own
`read_only`/`shaping`. Design rationale: a representation-rank probe found the 128-dim trunk runs in ~3–5
effective dims, so capacity isn't the constraint — the risk this isolates is gradient interference.
**v42 the TURN-HISTORY DEPTH cut** (`N_HISTORY_TURNS` 10 → 7) — a retrain-class obs-dim change: the
observation drops from 10 to 7 consecutive TurnDelta slots (159 dims each), so the turn-history block is
1113 dims (was 1590) and the total obs is **2992** (was 3469). `n_history_turns`/`total_dim` are already in
`_WEIGHT_FIELDS`, so `check_compatible` auto-rejects any pre-v42 checkpoint on the obs-dim weight-field check
(NO `ARCH_SIGNATURE` bump — the obs-dim weight-field check already catches it).
**v43 the PUBLIC-VALUE aux head** (`gen3_pubval_aux_v1`; `pubval_mode` / `--pubval-mode
{none,read_only,shaping}` + the training-only `--pubval-coef`, default 0.1) — `PubValHead` (the WinProbHead
pattern, a named subclass) reads `value_pooled` and is regressed toward the FROZEN human-replay-calibrated
public value **V_pub = P(win | PUBLIC board)** (`agents.training.pubval` + `data/gen3_pubval.json`: a
17-feature logistic over material/hazards/status/boosts/turn/weather aggregates, fit by `python -m
agents.training.pubval_calibration` on the 170k-game rated gen3ou replay corpus — held-out-by-game AUC 0.734,
turn-1 AUC 0.500 leakage-clean, calibrated). The value-INDEPENDENT exogenous signal (human outcomes, not the
self-play bootstrap) as a DENSE per-step shared-trunk target — the trunk sees WHEN the game swung (the
credit-assignment lever aimed at the measured defensive/positional value blindness). The target rides a
training-only `pubval_target` obs Dict key computed env-side per decision from the LiveView (PUBLIC state
only — leak-free; live↔corpus-parser parity is structural via ONE shared feature definition, guarded
end-to-end by `poke_env_gaps/pubval_parity_fuzz_test.py`). SIDE readout — never in pi/vf, NEVER in GAE
(V^human ≠ V^π). `read_only` = a stop-grad learnability probe ("can the trunk carry V_pub?"); `shaping` = the
human positional prior shapes the trunk (the experiment). STRUCTURAL + resume-immutable string gate (like
`win_prob_mode`); OFF byte-identical (NO `ARCH_SIGNATURE` bump). Metrics `pubval/*` (watch `mae`→0, not the
entropy-floored `bce`) + `grad/pubval_share`; the acceptance gate = the critic's defensive-AUC-by-style
transfer. Design: `designs/ai_v8/design_public_info_value.md`.
**v44 the TEAM-ARCHETYPE latent + head FiLM** (`gen3_zarch_film_v1`; `zarch_film` / `--zarch-film
{off,heads}` + `zarch_dim` / `--zarch-dim` [default 32 = `ZARCH_DIM`] + the training-only
`--zarch-recon-coef` [1.0] / `--zarch-vicreg-coef` [0.1]) — the amortization-gap **STORAGE** fix
(`designs/learning/amortization_gap_and_conditioning.md`: per-team distillation was shown to fix
greedy-local play on distilled teams but NOT generalize to neighbors AND to interfere with the rest —
the literal signature of conflicting per-team strategies cancelling in one shared head). `ZArchEncoder`
builds a **TEAM-STATIC, permutation-invariant DeepSets latent z_arch** over OUR team's **INVARIANT**
facts only — species ⊕ item ⊕ ability ⊕ moves (mean move-emb) ⊕ the 18-dim spread block, per-mon atom
MLP → mean over 6 → LayerNorm — DETERMINISTIC (no VIB sampling in v1: per-forward sampling would break
team-static, PPO's epoch ratio recompute, and eval determinism; LUT-first is the chosen operating
point) with **DETACHED embedding reads** (zero trunk gradient interference, the `belief_grad_mode`
philosophy). Two **zero-init FiLM generators** (one per root head) then modulate the post-projection
pre-ReLU head features `h·(1+Δγ(z)) + Δβ(z)` — identity-at-init, so ON starts byte-identical; per-team
gradients land in different rank-`zarch_dim` subspaces instead of cancelling. Anti-collapse = the
species multi-hot **reconstruction BCE** (a constant z can't reconstruct different teams; Species
Clause ⇒ lossless) + a **VICReg per-dim variance floor** (`zarch/std` is the collapse monitor;
`film/{pi,vf}_{gamma,beta}_norm` the deviation-from-identity read). Coefs auto-zeroed on a single-team
(pinned `--trainee-team`) run (z is constant there → degenerate variance floor; FiLM stays on as a
learned per-team bias). STRUCTURAL: `zarch_film` string + `zarch_dim` int gated in `check_compatible`
(the `value_dist_mode`/`bins` pattern); OFF byte-identical (NO `ARCH_SIGNATURE` bump); requires
nothing (independent of the belief/damage stack); threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites.
**v45 the DISTRIBUTIONAL VALUE CRITIC — Phase B** (`gen3_dist_critic_v1`; `value_from_dist` /
`--value-from-dist` + the migration hatch `--allow-value-from-dist-change`) — promotes the v29
`ValueDistHead` from a SIDE readout to the actual CRITIC. When on: GAE / bootstrap / deployment read
**E[Z]** (the distribution's mean, `policy._critic_value` → `head.mean(logits)` → `_denorm`, same
PopArt peg as the scalar), the **HL-Gauss CE becomes the PRIMARY value loss** (weighted by `vf_coef`,
not the aux `value_dist_coef`), and the scalar `value_net` FREEZES as a fallback + the E[Z]-vs-V
monitor (its MSE term dropped from the loss; PopArt still POPs it harmlessly + keeps the μ/σ peg
alive for the CE's normalized targets). The "Stop Regressing" recipe (Farebrother) — a categorical
critic resists the crystallization the scalar MSE breeds. **WARM-STARTABLE** on a `--value-dist-mode
shaping` lineage (the offline probe confirmed E[Z]≈V at pearson 0.988): no state_dict change (both
heads always exist) and the frozen forward's ACTION selection is unchanged, so it is RESUME-IMMUTABLE
(the `belief_grad_mode`/`vf_coef` class) — recorded on `ModelVersion`, enforced resume-only via
`check_value_from_dist`, EXCLUDED from `check_compatible` (gating a frozen opponent would false-reject
self-play). NO `ARCH_SIGNATURE` bump; requires `--value-dist-mode shaping` (the head must be a live
trunk-shaping critic). Threaded as a POLICY kwarg (`value_from_dist`, like `use_popart`) through both
`policy_kwargs` sites + the resume enforce; `value_share` (grad-balance) now points at the CE term.
**v46 the PER-TEAM LUT** (`gen3_zarch_lut_v1`; `zarch_lut` / `--zarch-lut {off,add,only}`) — a FREE,
unconstrained conditioning code per pinned `--trainee-teams` team, layered on the v44 z_arch. It tests
ONE thing: the multi-team exploiter ceiling (N=1 0.84 / N=3 0.835 / N=10 0.825 all distil cleanly, but
**N=20 stalls ~0.66**). The FiLM diagnosis is SNR/ill-conditioning, not capacity — the DeepSets z is
COMPOSITIONAL, so z-similar teams sit at `z̄ + ε_i` with tiny ε and the generator's gradient is
proportional to that residual; a **random-init** LUT makes the codes large and ~orthogonal from step 0,
which is exactly the intervention that story predicts. If N=20 still stalls with a free code, the
ceiling is NOT conditioning signal. `add` = `LN(z_deepsets + code)` (keeps composition — an UNMATCHED
team hits the ZERO-init row 0 ⇒ z is exactly the DeepSets z); `only` = `LN(code)` (the sharpest
ablation). The team is identified **from the OBSERVATION** (`agents.model.team_signature`: sorted
species(6) ⊕ moves(24)) so **no env / eval / prober / frozen-opponent plumbing changes**; species alone
is NOT enough (5 of the def-20 cluster's 20 teams share a roster — that would make the "per-team" code a
per-PAIR code), and `build_roster_table` THROWS on a duplicate signature or a move-set mutator
(Mimic/Transform/Sketch). The GIGO canary is **`zarch/lut_hit_frac`** (must be ~1.0 — a missed lookup
falls through to row 0 and silently makes the experiment a no-op) + `zarch/lut_code_dist`. STRUCTURAL
string + int (`zarch_lut_teams`, the Embedding height) gated in `check_compatible`; OFF byte-identical
(NO `ARCH_SIGNATURE` bump); requires `--zarch-film heads` + `--trainee-teams`.
**v47 the FROZEN pre-attention move belief** (`gen3_belief_single_compute_v1`;
`move_belief_single_compute` / `--move-belief-single-compute`) — compute the move belief **exactly
once** per forward and freeze it. Under `--move-belief-prefuse` the belief is predicted + reinjected
BEFORE the transformer, but the between-layers refine callback then **re-read** `move_logits` off the
(reinjected, then attention-enriched) opp tokens — so the belief was computed **3× in the production
config** (prefuse + one per `--damage-refine-rounds` round) and the physics consumed a *different*
posterior than the one attention was handed. ON, the refine kernels reuse the stashed pre-transformer
logits, giving the intended pipeline: **belief ONCE (pre-attention) → physics ONCE → N attention
layers that CANNOT revise it.** With `--damage-refine-rounds 1` the callback fires only before layer 0
(on pre-attention role tokens), so both transformer layers then reason over frozen physics — the
`next_run_plan` item-3 "prefuse-style, no between-layer recompute" arm, and the shape the owner
specified. The stash is **live, not detached**, so the op's damage gradient still reaches the same
belief computation the reinjection used (one posterior, one gradient path). Also strictly cheaper: one
fewer move-belief head pass per forward. **Cold-start inert by construction** (pinned by a test): under
`--move-prior-fusion` `MoveBelief.move_head` is ZERO-init (posterior == the Smogon prior ⇒
token-independent) AND `refine_proj` is ZERO-init (injection ×0), so frozen and per-round are
byte-identical at step 0 and can only diverge as those paths learn. FORWARD-BEHAVIOR toggle like
`move_belief_prefuse` (same `MoveBelief` params → state_dict identical, projection widths unchanged);
gated in `check_compatible` (bool); OFF byte-identical (NO `ARCH_SIGNATURE` bump); requires
`--move-belief-prefuse` (without it the only belief is POST-transformer, so there is nothing to reuse —
enforced at the CLI and the extractor). Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites. Tests:
`belief_single_compute_test.py`.
**v48 the CPU-DAMAGE DELETION** (`gen3_cpu_damage_deleted_v1`) — the delete step of the
`--unified-obs` deprecation playbook: the 51-dim incoming-damage block, the 44-dim move-effect block
and the 8 active-move scalars are removed from the OBSERVATION (reactive 414 → 311, obs 2992 → 2889),
along with the three `mask_*_obs` `ModelVersion` fields and the `--unified-obs` / `--mask-*-obs` CLI
flags. See the observation-vector section above for the rationale and the measured CPU refund.
`_migrate_config` **POPs** the three dead keys (`from_json_file` does `cls(**data)`, so a stale key
would raise `TypeError` rather than the clear arch error). Retrain-class, caught by the existing
obs-dim weight-field check — NO `ARCH_SIGNATURE` bump.
**v49 the CANDIDATE-AXIS CAP + the POINTER ACTION HEAD.**
**`damage_candidate_k` / `--damage-candidate-k K`** (`gen3_topk_candidates_v1`) — the op priced ALL
~400 move-nums per defender even though the opponent runs four moves; the belief already says which
candidates matter. This caps the INCOMING candidate axis at the K most-believed (per batch row,
selection DETACHED, gathered weights still differentiable so the belief gradient rides the survivors),
with **NO tail bound** — the truncated mass is DROPPED, so a rare-but-lethal candidate below rank K is
simply not priced (the on-policy probe measured top-16 owning 94.2% of channels with misses BIMODAL,
which is why the plan called the tail mandatory; this flag is the explicit trade). `_damage_rolls`'
per-candidate args became `[B,C]` (one call site). Measured: **+11.4% forward / +63.5% op at B=256**
(learner) but only **+0.3% at B=1** — the CPU/PFSP opponent is DISPATCH-bound (~14.3k aten calls at
~0.44 µs), not tensor-size bound, so this is a learner lever, not an eval-latency one.
FORWARD-BEHAVIOR (no new params), unconditional int gate; 0 byte-identical; requires `--damage-op`.
**`pointer_head` / `--pointer-head`** (`gen3_pointer_head_v1`) — score each action FROM THE TOKEN OF
THE ENTITY IT SELECTS: move logit *k* from the move at **REQUEST slot k**, switch logit *j* from
our-team token *j*. Fixes two measured defects structurally rather than by guard: **F2** (switch
logits are read from a permutation-INVARIANT pool, so a bench mon's token never reaches its own logit)
and the **ordering bug class** (`agents/action/ordering_integrity.py` exists solely because the
extractor reads moves SORTED-BY-ID while actions use REQUEST order — the head permutes by move-num
IDENTITY, making a misaligned logit unrepresentable). The per-move tokens already existed inside
`PokemonEncoder` (post move-self-attention, `[B,12,4,32]`) and were merely flattened away; the head
stashes them instead. It ADDS a **zero-init delta** to the flat head's logits (identity-at-init,
warm-starts from any checkpoint, clean A/B) — a guarantee that only actually holds because of the
**M1** fix below. The policy adds it in `_get_action_dist_from_latent`, the single point all three
logit sites funnel through. STRUCTURAL bool gate; OFF byte-identical. NO `ARCH_SIGNATURE` bump (both
OFF reproduce v48 exactly). **SUPERSEDED at v51** — the delta head and its `pointer_head` flag/field
are deleted; the pointer head became THE action head (see v51 below).
**v50 the PRE-ATTENTION UNIFIED DAMAGE OPERATOR** (`gen3_damage_op_prefuse_v1`; `damage_op_prefuse` /
`--damage-op-prefuse`) — ONE damage computation per forward instead of two. Today the op runs **twice**:
a LEAN `discrete_*` recompute inside the between-layers refine loop (×`--damage-refine-rounds`, 2 in the
production config) **plus** the FULL 835-dim block after the transformer. ON, the spread + HP-type
beliefs and the FULL op all run on the **PRE-transformer role tokens**, the per-OUR-mon incoming rows are
injected onto our tokens through a **zero-init `prefuse_proj`** (the `refine_proj` convention →
identity-at-init), and the **SAME full block is concatenated to both heads** — so the ledger-P1 head
dependency is preserved at full width, only its inputs move from refined to un-refined. **The
justification is CPU cost, not architecture:** at B=1 on CPU — the PFSP frozen-opponent regime, which
sits on the rollout critical path — the forward is 6.45 ms / 14,337 aten calls (~0.44 µs each, so
DISPATCH-bound), the op is ~75% of it (2.454 ms post-transformer + ~2.4 ms refine loop) and the attention
layers are 0.27 ms (4%); measured `--damage-refine-rounds` 2→1 = +14.0%, 2→0 = +28.2%. The
"attention now reasons over full-fidelity physics" story is **secondary and unsupported** — physics-into-
the-trunk measured NULL 3-for-3 (**K9/K10**) and the lean kernel was already a 91.8%-agreement proxy for
the full op (**K10a**). Two properties bound the risk, both test-pinned: the op **requires
`--move-belief-prefuse`**, so the move belief — its dominant input — is **bit-identical** in both shapes
and only the spread + HP-type posteriors are re-sourced; and at **cold start the block is bit-identical**
(every belief head is zero-init ⇒ token-independent posteriors), so the divergence is created by
TRAINING, not by the reordering. STRUCTURAL (adds `prefuse_proj`) → bool compare in `check_compatible`;
OFF byte-identical (NO `ARCH_SIGNATURE` bump); **mutually exclusive with `--damage-refine-rounds > 0`**
(the loop is what it replaces — and that also drops the v36/v37 outgoing/status trunk residuals, which
ride that loop and are NOT reproduced). Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites. **Measured:**
`tmp/pfsp_opponent_sweep.py` B=1 **6.452 → 4.617 ms (+28.2%, −4,126 aten calls)** — the same as
`--damage-refine-rounds 0` alone (4.620 ms), so the CPU win IS the loop deletion and the prefuse rides
along ~free; `tmp/damage_prefuse_kl.py` on 3000 real states puts the head-block re-sourcing at block
cosine **0.988** and masked KL **3.0% of the zero-block ceiling** (2.9% argmax flips) on a short-trained
snapshot — a floor, not a verdict.
**M1 — SB3 was destroying every zero-init in the extractor (FIXED 2026-08-01).** `ActorCriticPolicy._build()`
orthogonally re-inits EVERY `nn.Linear` in the feature extractor (`ortho_init` defaults True, nothing
overrode it), so **13** Linears documented as zero-init were random from step 0 in every real run —
`refine_proj`/`outgoing_proj`/`status_*_proj`/`film_pi`/`film_vf`, plus the belief heads whose
zero-init is what makes the **cold-start posterior equal the Smogon prior**. Guarded by
`Gen3FeaturesExtractor.restore_identity_init()`. **This puts a standing caveat on the K10 and D4
result families** — see `designs/research_state/ledger.md` → M1 and the model leaf.
**v51 the POINTER-NATIVE ACTION HEAD** (`gen3_pointer_native_v1`, NO flag — the fresh-generation
reset, `designs/ai_v9/design_pointer_action_head.md` §0) — the flat positional `action_net` is
DELETED (`Gen3DualHeadMaskablePolicy._build` swaps in a raising stub and rebuilds the optimizer) and
the `PointerNativeActionHead` is THE action head: move logit *k* ← the REQUEST-slot-k move token ⊕
its op cells `[low,high,crit,pko,p_land,known,sec×10]`, switch logit *j* ← our-team token *j* ⊕ its
incoming row + CB tail (+ OAX attacker row under `--damage-matrices-outgoing-all`), struggle ← the
context — with **`latent_pi`** as the shared context, so the op block / beliefs / FiLM condition
every score. Position-EQUIVARIANT (one shared scorer per entity; the `ordering_integrity.py`
sorted-vs-request bug class is unrepresentable at the logits). The op owns the cell layout
(`DamageOperator.pointer_cells`, offsets pinned against `decode_damage_block`); widths are 0 when a
source block is off (the Linear NARROWS, never zero-pads). Zero-init scorers built AFTER SB3's
ortho-init ⇒ cold-start policy is uniform-over-legal. The v49 `pointer_head` field is REMOVED
(`_migrate_config` POPs it); no gate exists because there is no off state — the cross-era break
rides the **`ARCH_SIGNATURE` bump**, so every pre-v51 checkpoint fails loud (owner decision
2026-08-03: no resume/warm-fork across the boundary; pools/opponents re-seed from the new lineage).
**v52 the DISCRETE typed HIDDEN POWER, end to end** (`ARCH_SIGNATURE` `gen3_typed_hp_belief_v1`) — the model
never reasons over a typeless Hidden Power again. See the `ARCH_SIGNATURE` paragraph above for the full
description: the presence×type composition moves into `HPTypeBelief.compose_typed_hp` next to the move head, so
the posterior EVERY consumer reads (damage op, top-K, move BCE, latent grading, token reinjection, prober)
carries HP at the 16 real typed nums 355-370 with the bare 237 hard-off; `Σ_t P(HP_t) = presence` makes "a
revealed HP must exist as some type" structural; moveset exhaustion and effectiveness narrowing eliminate
impossible types; the `--hp-type-belief` mode flag is deleted (its `off` state was a correctness bug) and the
head is unconditional under a move belief, no longer requiring `--damage-op`; the belief LABELS use the true
typed num (they previously trained the typed channels toward zero); and the learnset gate stops marking all 16
typed HPs unlearnable. `_migrate_config` **POPs** the dead `hp_type_belief_mode` key. Retrain-class — the
forward math changed while the projection widths did not, so the `ARCH_SIGNATURE` bump is what catches it.
Tests: `hp_type_belief_test.py` + the extended `poke_env_gaps/belief_labels_fuzz_test.py`.
**v53 the HP-BELIEF FACTORISATION ABLATION** (`gen3_hp_belief_ablation_v1`; `hp_belief_mode` /
`--hp-belief-mode {composed,flat}`) — measures what the v52 presence×type factorisation is actually WORTH.
BOTH arms reason over the DISCRETE typed HP nums 355-370 and drive the bare BP-0 num 237 hard-off via the
shared `mask_typeless_hp` helper — the typeless candidate is the "opp HP reads immune" bug, NOT the variable.
`composed` (DEFAULT) is byte-for-byte v52: the `HPTypeBelief` head, the structural `Σ_t P(HP_t) = presence`
(reveal-pinned) + moveset exhaustion + effectiveness narrowing. `flat` is the ABLATION: NO `HPTypeBelief`
head — the multi-label move head predicts the 16 typed channels INDEPENDENTLY, each off its own real
per-typed Smogon usage prior (the prior table already writes the typed cells' own rates beside the 237
presence sum), i.e. Hidden Power is treated exactly like any other move — no factorisation, no reveal
constraint, no narrowing. STRUCTURAL: `flat` drops a module (a state_dict change as well as a forward one),
STRING-gated in `check_compatible` (the `win_prob_mode` pattern), fresh-only; `_migrate_config` defaults
pre-v53 configs to `composed`. `--hp-belief-mode flat` AUTO-ZEROES `--hp-type-belief-coef` with a loud note
(the ablation builds no head, so there is no posterior for the CE to supervise; the zarch single-team
auto-zero precedent — the coef defaults to 0.05, so erroring would make the ablation flag fail out of the
box). Default byte-identical → NO `ARCH_SIGNATURE` bump. Tests: `hp_type_belief_test.py` (both arms
237-masked, the version gate, the invalid-mode raise, the migration default).
**v54 the MOVE ENTITY SEATS** (`gen3_entity_move_seats_v1`, Stage 1 of the entity generation —
`designs/ai_v9/design_generation_roadmap.md` §3) — MOVE tokens become first-class attention SEATS
in the unified trunk, appended AFTER the global token so every existing absolute slice
(team/history/global, the refine callback's tail cat) is position-stable. **E3** (unconditional):
our active's 4 request-ordered move tokens — the SAME identity-permuted tokens the pointer head
reads, permuted ONCE pre-transformer and projected 32 → d_model; **the pointer head now reads the
REFINED E3 seats** (post-attention, d_model-wide — its move tokens are board-aware). **E4**
(`entity_topk_seats` / `--entity-topk-seats K`, 0 = off): the opp active's top-K believed
threat-move seats — the op's `refine_candidates(k=K)` candidate definition (belief-weighted,
typed-HP-scattered, ONE source with the refine kernels), each seat `[move latent ⊕ belief w ⊕ acc
⊕ is_phys]`; index selection detached, `w` differentiable (the belief gradient rides the seats);
all K key-masked + zeroed when no opp active; requires `--damage-op-prefuse` + `--move-latent`. NO
edges yet (Stage 2). The token-type table grows 4 → 6 (`TOKEN_TYPE_OUR_MOVE` /
`TOKEN_TYPE_THEIR_THREAT`) and `TeamTransformer.forward` gains a generic `extra=(tokens, types,
pad)` seat path (returns the refined extra seats as a third output). Measured B=1 (threads=1): E4
K=5 = **+0.18 ms** on a ~3.1 ms prefuse-stack forward — on the spike's +0.19 ms prediction
(dispatch-bound, not FLOP-bound). E3's break is UNCONDITIONAL (state_dict: the token-type table,
`move_seat_proj`, the head's wider `move_proj`) → the `ARCH_SIGNATURE` bump carries it;
`entity_topk_seats` is a STRUCTURAL int gated in `check_compatible` (the `damage_topk_k` pattern).
Tests: `entity_seats_test.py` (seat-layout stability, masked-seat bit-identity no-leak, the
op-candidate single-source, the LayerNorm random-cotangent gradient probe).
**v55 the DamageOperator BLOCK TRIM** (`ARCH_SIGNATURE` `gen3_op_block_trim_v1`, NO flag) — the op sheds its
three least-used output families and one dead code path, acting on the ledger-**P1** per-block dependence
ablation (4000 real eval states, per-block zero → masked KL, ceiling 0.9385 = zeroing the whole op). OUT: the
opp-active per-STATUS **incoming SECONDARY** scalars (10 dims, **0.1%** of the ceiling — the most INERT
channel in the operator), the opp-active believed-**EFFECT** scalars (6 dims, **1.2%**), and the **OUTGOING
slp/psn/tox** per-move secondary columns (12 dims = 4 moves × 3). The first two are opp-active-level
collapses with **no defender axis** — they say "the opponent can flinch someone" without saying whom — and
v35's `_incoming_matrix` already carries the same facts PER MOVE (`_DMG_IMX_HDR_EFFECT`/`_DMG_IMX_HDR_SEC`)
and PER DEFENDER (`status_lands`, ledger **P4**: KL 0.0005 vs the collapse's 0.1446); deleting them also
removes the whole UNMASKED-belief read `w = sigmoid(...)` (and its `_EFF_*`/`_SEC_*` sparse buffers) from
the forward, leaving `w_all` as the op's single belief read. The third is **structural zeros, MEASURED not
asserted**: that block prices OUR moves, and gen3 has **no damaging move that inflicts sleep at all** while
the psn/tox carriers appear on **1 / 0 of the 773 `data/teams/` teams** (the INCOMING side keeps all 10 — it
faces the OPPONENT, who isn't restricted to our pool). ALSO deleted: **`_topk_block`**, the v30 LEAN top-K —
a strict SUBSET of the v35 incoming matrix that the matrix already suppressed, which the same cProfile
measured at **0 calls per forward** in the production build (dead code in the op's hottest file). So
`damage_topk_k` now means "the incoming matrix's K" and nothing else: `K>0` without
`damage_matrices_incoming` **raises** in both the extractor and the op (never a silent empty block), and the
CLI auto-enables the matrix when `--damage-topk` was set with no explicit `--damage-matrices` (the
`--unified-moves` path). Net **−28 dims** off both projection heads (op `out_dim` 835 → 807, `incoming_dim`
101 → 85). **This is a dims/complexity change, NOT a throughput one:** measured B=1 CPU forward 4.276 →
4.289 ms and 3534 → 3525 profiled calls/forward — i.e. no change, since the deleted work is ~9 aten calls
on a dispatch-bound forward and the lean block already never ran. Honest residual: with `--damage-matrices
incoming` OFF there is now no effect/secondary signal at all (the accepted trade at 1.3% of dependence).
The projection widths DO change, so a stale checkpoint would fail on a `load_state_dict` shape mismatch —
the `ARCH_SIGNATURE` bump turns that into a clear arch error instead; `MODEL_CONFIG_VERSION` 54 is a stamp
only (no field added/removed). Tests: `damage_op_test.py` + the constructed-physics oracle
`poke_env_gaps/damage_op_probe_fuzz_test.py` (22/22).
Orthogonal to v54's entity seats (those enter the TRUNK; this trims the op's HEAD-CONCAT output), so the
two compose — only the single shared `ARCH_SIGNATURE` string had to be sequenced.
**v56 the EDGE-BIAS TRUNK** (`gen3_edge_bias_trunk_v1`, Stage 2 of the entity generation —
`designs/ai_v9/design_generation_roadmap.md` §3 Stage 2) — computed physics becomes attention
EDGES. The encoder stack is swapped for `BiasedEncoderLayer` (the spike-proven clone: same math,
attention takes an additive per-pair per-head float bias via SDPA's additive mask; the key-pad
mask rides the same tensor as a -1e9 addend — stock-parity test-pinned) — an UNCONDITIONAL
state_dict change (fused `in_proj` keys) carried by the `ARCH_SIGNATURE` bump. The delivered
FAMILIES ride `edge_bias_families` / `--edge-bias-families {off,d,d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5}` (STRUCTURAL
str in `check_compatible`; `"d"` is the FROZEN d1,d3 alias — new families are explicit-only, so a
saved config never silently grows maps; growing the valid set is NOT a version bump, the string
gate catches any mismatch): **D1** = our active's 4 moves × the opp's 6 mons (the v34
outgoing-matrix kernel via `DamageOperator.pairwise_outgoing`) at the (E3 seat, opp-mon seat)
pairs + transpose; **D2** = every OUR mon's best offense vs the opp ACTIVE (the v39 switch-in
kernel move-collapsed via `pairwise_bench_outgoing`, cell `[best_high,best_pko,p_outspeed,alive]`)
at the (our-mon seat, opp-ACTIVE seat) pair via a one-hot column (batch-varying target); **D3** =
the opp's top-K believed moves × our 6 mons (the pre-collapse `_incoming_rolls` kernel via
`pairwise_incoming` — ONE physics body with the refine consumer — at the SAME detached candidate
selection the E4 seats stashed) at the (E4 seat, our-mon seat) pairs + transpose; **S1/S3** = the
v27/v37 status-landing kernels' NEW `per_pair` branches — our status moves × opp mons at the E3
pairs, the opp's believed status candidates × our mons at the E4 pairs (cells `[land, land·immob
(,w)]` — the un-collapsed "will THIS status move land on THIS mon"); **V** = the full mon↔mon
SPEED edge (`pairwise_speed`: P(our i outspeeds opp j) for every pair, cell `[p_outspeed,
both_alive, revealed_j]` — our real spread vs the believed/neutral opp spread, per-mon PUBLIC
para ×0.25 both sides, the uncertainty-aware sigmoid under `prob_outspeed`; v1 = NO stage boosts
either side, the coarse-signal convention. ⚠️ **v58 GIGO fix**: through gen-1 AND gen-2 this
kernel [+ C1's outspeed] read stat index 4 — SPECIAL DEFENSE — as "speed" [the main op's index-5
paths were always right]; fixed 2026-08-06 with named `_BS_*`/`_NAT_*` indices + the
Aerodactyl-vs-Snorlax discriminating regression test; both trained gens' V-edge audit numbers
measured the BUGGY feature) at the (our-mon, opp-mon) block; **D4** = the MISSING quadrant — every opp BENCH mon's believed
threat to every our mon (`pairwise_bench_incoming`: per opp mon j the top-K_bench=4 candidates
from ITS OWN slot of the composed posterior — the belief gradient reaches the BENCH move heads for
the first time — de-timid attacker, real-spread defenders; revealed+alive-gated; the opp ACTIVE
column ZEROED, that quadrant is D3's) at the same mon↔mon block; **T** = the gen3-critical TRAPPING edge (`pairwise_trap`: P(cannot
switch) BOTH directions — Shadow Tag / Arena Trap·grounded / Magnet Pull·Steel, our side exact,
opp side revealed-exact else the Smogon `SPECIES_TRAP_PRIOR`, Levitate folded into grounded;
fail-loud ability resolution) at the mon↔mon block — "my Dugtrio traps their weakened Blissey" is
a plan-defining edge; **X** = the ENTRY/EXIT edge (`pairwise_entry`: per mon, gen3 Spikes entry
chip on its OWN side ×grounded [Flying/Levitate immune, opp Levitate prior-folded] + Pursuit
exposure [belief-composed vs OUR mons, exact vs theirs] + Dark eff at the victim) at the
(mon, GLOBAL seat) pairs — "switching is not free", board-composable through the global token;
requires `--damage-op --damage-op-prefuse`; **G** = the per-mon END-OF-TURN HP LEDGER
(`pairwise_schedule`: signed maxhp fractions — Leftovers +1/16 [revealed-exact], sand/hail chip
−1/16 [Rock/Ground/Steel / Ice immune, live weather one-hot], burn/psn −1/8 + **Toxic at its
RAMPED next tick −(ticks+1)/16** from the PUBLIC obs toxic counter both sides
[owner-prioritized 2026-08-06 — the old flat −1/8 under-priced late-stall Toxic 3-4×], Leech
−1/8 on the seeded ACTIVE only — correctly, it clears on switch) at the same (mon, GLOBAL
seat) route as X; requires `--damage-op`; **C4** = the first CONSEQUENCE edge (`pairwise_protect`): at the E3 seat of a
Protect/Detect/Endure request slot, `[is_protect, p_success (the gen3_protect_odds_v1 obs
scalar), net_ours, net_theirs]` — the two ACTIVES' G-ledger sums, i.e. the turn a successful
Protect banks (their Toxic ramps, our Leftovers ticks); requires `--damage-op`; **C1** = the first
HYPOTHETICAL-WORLD damage consequence (`pairwise_boost`): per (E3 SETUP-move seat k, opp mon j)
the DELTA cells `[is_boost, d_best_high, d_best_pko, d_outspeed]` from RE-RUNNING the validated
outgoing-matrix kernel under slot k's post-boost stages (a `boost_delta` threaded into the
kernel's stage read — None byte-identical; `MOVE_SELF_BOOSTS` = the ~17 declarative pure-setup
moves via `MoveData.self_boosts` PLUS the runtime NON-GHOST CURSE branch [owner-prioritized:
+1 atk/+1 def/−1 spe from `gen3_mechanics.CURSE_NON_GHOST_BOOSTS`, resolved from the user's
live types — the −1 spe reads as a NEGATIVE d_outspeed; a Ghost user's Curse stays a zero row];
Belly Drum = the recorded TODO [niche: hp_cost channel + fails-below-half gate + the C1b
incoming-at-halved-HP re-run]; Defense Curl/evasion stay unpriced) + the speed-recipe outspeed
delta (stage folded BOTH worlds — here the stage IS the signal); the INCOMING halves are LIVE
too (**C1b**, `pairwise_boost_incoming`: the opp mons' believed candidates vs OUR ACTIVE at
current-vs-post-boost def/spd stages, 5 worlds on one world axis — Iron Defense/Amnesia/Bulk
Up/CM and Curse's +1 Def price their damage-taken deltas; cell = the 6-wide
`[is_boost, d_high, d_pko, d_outspeed, d_in_high, d_in_pko]`); **C3** = the RECOVERY-FLIP
consequence (`pairwise_recovery`): per (E3 recovery seat, opp mon) `[is_recovery, d_in_pko,
rest_sleep_turns]` — the believed-hit damage computed ONCE, the validated `_rolls` KO ramp
re-evaluated at the post-heal HP worlds (`MOVE_HEAL_FRACTION`: 0.5 plain + weather heals
[flat v1 approximation], 1.0 Rest, Wish EXCLUDED — delayed, the wish obs scalars own it) —
"does healing beat their KO", hitting −w exactly at the threshold flip — **and Rest's
DETERMINISTIC self-sleep cost is priced** (exactly 2 lost turns, 1 with Early Bird; our own
ability is KNOWN so the channel is exact, never a prior — the same verified
`expected_free_turns` tables as C2's opp-sleep channel); requires
`--damage-op`; **C2** = the STATUS-consequence edge (`pairwise_status_consequence`): per (E3
status seat, opp mon) `[is_status, land, d_their_outspeed, d_in_phys_high, d_sched,
d_in_all_slp, e_slp_free_turns]` — what LANDING would do behind S1's "will it land": T-Wave's
para flips P(outspeed) toward us (their spe ×0.25), WoW halves their worst believed PHYSICAL
hit, brn/psn add the flat −1/8 tick while **TOXIC lands at its TRUE first tick −1/16**
(split from plain psn by move num — they share immunity cat 5; the ramp thereafter = G's
live counter fact), and **SLEEP's tempo consequence is priced** — their whole believed threat
suspended (−worst hit, ANY category) for E[free turns] from the VERIFIED sleep hazard tables
(2.5 no-EB / exactly 1.0 revealed Early Bird, Smogon-prior-marginalised per mon —
`sleep_belief.expected_free_turns`, one source with the wake belief; Leech Seed deliberately
G/S1's fact); deltas RAW — decorrelated from `land`, the head composes consequence ×
probability like pko × accuracy; requires `--damage-op --damage-outgoing`;
**C5** = the BATON PASS receiver edge (`pairwise_baton`) — the FIRST family on the (E3 seat,
OUR-mon) route: per (BP seat, our mon j) `[is_bp, d_best_high, d_best_pko, d_outspeed]` — the
receiver's offense/outspeed vs the opp active INHERITING the active's current stages (the v39
switch-in kernel re-run under `inherit_stages=True`) minus its neutral baseline; no stages ⇒
all-zero (an unboosted BP is just a slow pivot); active column zeroed; requires `--damage-op`.
C1 also prices **Belly Drum** now (the curated +12-clamps-to-maximize row + the half-max-HP
`hp_cost` cell channel + the fails-below-half gate — c1 cell 7), and C3's weather heals fold
LIVE weather (2/3 sun / 1/4 other / 1/2 clear — sand-era Synthesis is honestly weak).
C1 is +2.1 ms B=1 EAGER (4 extra kernel runs) but the production PFSP path is COMPILED
where the dispatch fuses. Each family maps
its
cell through a ZERO-INIT `Linear(cell → 2·n_heads)` (one head-set per direction) ⇒ ON is
bitwise-identical to OFF at init (test-pinned, all six). All seat blocks are contiguous index ranges ⇒
delivery is slice assignment, compile-friendly (fullgraph-pinned). d1/s1/c1 require
`--damage-op --damage-outgoing`; d2/d4/v/t require `--damage-op`; d3/s3 require `--entity-topk-seats > 0`. **The op head-concat is
NOT deleted** — per the deprecation playbook (and the K9/K10 trunk-null history) the edge home
lands first; deletion waits on the per-family bias-ablation audit. Measured B=1 (threads=1):
+0.63 ms for both families on a ~3.5 ms prefuse+seats forward (still under the v50 4.62 ms
anchor; the concat deletion is the eventual refund). Tests: `edge_bias_test.py` (stock-layer
parity, ON==OFF bitwise at init, placement-only-at-documented-pairs, requirement gates,
fullgraph compile, map-gradient liveness, the v55 gate/migration).
**v57 the E5 TAIL-THREAT SEATS** (`gen3_entity_tail_seats_v1`; `entity_tail_seats` /
`--entity-tail-seats`) — the truncation INSURANCE: every candidate consumer top-Ks (the E4 seats,
D3/D4 edges) and DROPS the tail (the bimodal-miss finding: truncation loses candidates entirely,
never shaves them). 6 per-opp-mon seats carry `tail_proj([p_tail, worst_phys, worst_spec,
revealed])` — the beyond-top-K belief mass of THAT mon's composed posterior, worst_* = bound-ish
w·BP/150·acc split by category (defender-independent by design — a token, not an edge). NO new
token-type row (deliberate: growing the table 6 → 7 would change EVERY model's state_dict and
break loading in-generation checkpoints into newer code) — tail seats reuse
`TOKEN_TYPE_THEIR_THREAT` + a learned `tail_marker`. Appended LAST (the pointer stash's E3 block
is position-stable). STRUCTURAL bool in `check_compatible`; OFF byte-identical; requires
`--damage-op-prefuse` + `--entity-topk-seats > 0` (the tail is defined relative to the E4
truncation).
Current `MODEL_CONFIG_VERSION` = **57**, `ARCH_SIGNATURE` = **`gen3_edge_bias_trunk_v1`**.

---

### v67 — `gen3_deadline_clock_v1` (the deadline clock)

The observation CLOCK group goes **1 → 3 scalars**, so `GLOBAL_ENV_DIM` 18 → 20 and the obs
**2667 → 2669**. The group is `[log_elapsed, remaining_linear, log_remaining]`, all on the shared
`log(1 + MAX_TURNS)` denominator; `MAX_TURNS` (250) is now imported by `StallConfig.threshold` so
the obs normaliser and the turn the trainee actually forfeits on are one number.

**Why.** A forensic sweep of `ai_v9_09_gen8_beliefs_threat_inject_0811` at step 16,000,032 found
the critic reporting a **POSITIVE V(s) on the last decision before a −30 forfeit in 13 of 14**
timeout losses (mean **+9.33**, mean terminal TD surprise **−39.3**), and V *rising* over the
final 10 decisions in **10 of 14**. Over a 208-turn stall V moved −4.6 against a 60-point terminal
span, while the measured win probability fell 0.80 → 0.05 (20-rollout `replay_counterfactual` at
turns 41 / 121 / 208).

The single log-ELAPSED scalar is the mechanism: it spends **58.6%** of its range on turns 1–50 and
**4.0%** on turns 200–250, with the last 20 turns at **1.5%** — per-turn sensitivity at turn 249 is
**125× lower** than at turn 1. It is the right transform for "how far into the game am I" and the
inverse of what a deadline needs, since value near a cap is a function of turns REMAINING. The new
log-REMAINING channel gives those last 20 turns **55.1%** of its range (**37×** the old
resolution). That matters for credit assignment specifically: with `gae_lambda = 0.80` the terminal
reward reaches ~5 steps directly, so everything else arrives through the TD bootstrap chain — and
that chain must be anchored at the deadline link FIRST. A feature with no resolution there leaves
the chain with nothing to propagate.

Both remaining forms (linear + log) are supplied as raw facts rather than picking one — which the
model uses is its to learn. Remaining is clamped at 0 so an over-cap turn saturates instead of
going negative (linear) or NaN (log). Obs width and weight shapes move together (the move/global
context projections read `_gl['clock']['dim']`, replacing two hardcoded `1  # turn` literals), so
no migration is possible and the signature carries the break.

Obs-build gate: **3462.0 → 3463.0 calls/encode (+1, +0.03%)**, same session, `--reps 300`.

Current `MODEL_CONFIG_VERSION` = **67**, `ARCH_SIGNATURE` = **`gen3_deadline_clock_v1`**.

---

## 2. Superseded prose — root `CLAUDE.md` § Observation Vector

Moved verbatim from root `CLAUDE.md` (lines 740–802 of the pre-split file). The block table at the
top was correct against the code on 2026-08-08; the narrative under it was not (it describes a
414-dim reactive block with 19 scalars at `vec[14]`–`vec[18]`, which has not existed since v48).
The current layout is in `ARCHITECTURE.md` § Observation.

## Observation Vector

The full observation is a **2925-dim float32 vector** (`gen3_entity_recency_v1` — E9 step 1
added a 3-dim per-mon RECENCY block [turns_since_seen/acted/was_hit, turn-anchored,
log-saturated, both sides public, EpisodeTracker-sourced, fuzz-validated vs protocol truth];
per-mon slot 110 → 113, obs 2889 → 2925; before it a **2889-dim vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 113) | 678 | 0 |
| Opp team (6 × 113) | 678 | 678 |
| Active context ×2 (boosts + full volatiles, `VOLATILE_DIM`=44) | 116 | 1356 |
| Global env | 18 | 1472 |
| Reactive scalars (11) + matchups (288) + **active-req-moves** (12) | 311 | 1490 |
| Prev-turn action mask | 11 | 1801 |
| Turn history (`N_HISTORY_TURNS` × 159) | 1113 | 1812 |
| **Total** | **2925** | |

**`gen3_cpu_damage_deleted_v1` (v48) — the `--unified-obs` DELETE step.** Three CPU obs regions that
`--unified-obs` previously only **masked** from the model are now removed from the encoder entirely
(reactive 414 → 311, obs 2992 → 2889): the **51-dim incoming-damage / OHKO belief**, the **44-dim
action-aligned move-effect block**, and the **8 active-move scalars** (base-power ×4 + type-multiplier
×4). All three had live GPU homes (the `DamageOperator`'s incoming/outgoing blocks from the LEARNED
move belief, the `MoveLatentEncoder` move latent, the v27/v37 status-landing), so the masks existed
only to A/B the replacement — that A/B is settled and the producers are gone. The three
`--mask-*-obs` flags and `--unified-obs` are deleted with them. **This is a pure CPU refund on the
dominant rollout cost centre:** obs build was ~73% of per-decision controllable CPU, and the measured
benchmark moved **7,396 → 6,444 calls/encode (−12.9%)** with `state_encoder.encode` 0.456 → 0.363 ms
(−20%). `agents.observation.incoming_damage` (the math core) **stays** — the reward PBRS and the
prober both import it; only the obs WRITE is removed. Retrain-class: the obs-dim weight-field check
auto-rejects every pre-v48 checkpoint (no `ARCH_SIGNATURE` bump needed).

**The full per-block layout** — the 110-dim per-Pokémon slot (incl. a 3-dim
`gen3_sleep_wake_belief_v1` block: `sleep_is_deterministic` [Rest], computed `p_wake`, and
`sleep_counter_reliable` — zeros unless the mon is asleep), the 11-dim move slot, the 18-dim
spread block, global env, the 414-dim reactive block (**19 scalars** — the 14 prior + the
log-saturated **`turns_since_progress`** no-progress clock at `vec[14]`, `gen3_markovian_progress_v1`
(the no-progress reward keys on the SAME EpisodeTracker-owned counter) + the **2 protect-odds scalars**
at `vec[15]`/`vec[16]`, `gen3_protect_odds_v1` — P(a Protect/Detect/Endure succeeds NOW) for our /
the opp active mon, the gen3 floored-doubling stall odds (100/50/25/12.5, floor 1/8) from each mon's
`LivePokemon.protect_counter` (the only obs view of the stall counter; public both sides, no leak) +
the **2 `gen3_wish_wired_v1` `wish_floating` scalars** at `vec[17]`/`vec[18]` (our/opp side — the
pending-Wish "floating heal": `WISH_HEAL_FRACTION` ≈0.5 of the slot mon's max HP when a Wish cast last
turn resolves at the end of this turn, else 0; gen3 Wish heals the RECIPIENT's maxhp/2 so the fraction
is a constant ≈0.5 — GIGO-proof — slot-keyed so it survives faint/Roar-phaze/switch; reconstructed from
the event log since poke-env doesn't track it; fuzz-validated vs the real sim's resolve heals) — +
the 44-dim action-aligned
move-effect block, 4 slots × 11 feats (incl. the `gen3_status_cure_moves_v1` **cures_self_status** /
**cures_team_status** bits — Refresh self-cure, Heal Bell / Aromatherapy team-cure) + the **51-dim incoming-damage / OHKO belief block**
[`gen3_incoming_crit_split_v1`: per our mon, phys/spec expected-damage + the modal **no-crit** P(KO) +
the **crit-risk DELTA** per channel (crit-inclusive − no-crit ∈ [0, _CRIT_P] — a decorrelated "crit
tax" feature, so the model prices the modal line without over-weighting the coinflip) + P(outspeed) +
a **threat-provenance** scalar (1.0 = a revealed move, <1 = a usage-prior guess; 0.0 = no KO threat —
the "how much are we guessing" signal), then 3 opp recovery scalars] +
288 matchup + the **12-dim active-req-moves block** (`gen3_op_move_align_v1`: OUR active mon's 4 moves in
**REQUEST order** — `[move_num ×4, resolved_type_id ×4, legal_now ×4]` — the DamageOperator's OUTGOING
per-move blocks read THIS so their per-move output aligns with action logit 6+k, instead of the per-mon
block's sorted-by-id order; sits after the matchups, consumed only by the op via ObsUnpack, never the
raw-scalar path)), and the 159-dim
TurnDelta slot (incl. the embedded-ID manifest) — lives in **`src/agents/observation/CLAUDE.md`**.
Every offset is computed
from named constants; never hardcode indices.


---

## 3. Superseded prose — root `CLAUDE.md` § Feature Extractor Architecture

Moved verbatim from root `CLAUDE.md` (lines 805–874 of the pre-split file). The current phase chain,
with the production config's flags resolved, is in `ARCHITECTURE.md` § Feature extractor.

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py` is decomposed into named
phase `nn.Module`s chained by a thin orchestrator:

**`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `TeamTransformer` → `[BeliefHead?]` →
`[MoveBelief?]` → `CLSPool` → `[DamageOperator?]` → `ProjectionAssembler`**, then **two** root projection
heads (policy + value), each `pre_proj_norm` → `projection` → `ReLU`. Under `--damage-op-prefuse` (v50)
the `[MoveBelief?] → [SpreadBelief?] → [HPTypeBelief?] → [DamageOperator?]` group moves **before**
`TeamTransformer` and runs exactly once, injecting its per-our-mon incoming rows onto our role tokens
via a zero-init `prefuse_proj`; the same block still feeds both heads. The bracketed phases are
flag-gated (with all off the chain is the byte-for-byte baseline): `BeliefSlots`/`BeliefHead` under the
hidden-opponent **belief aux** (`--opp-belief-aux-coef>0`) — `BeliefSlots` fills the un-revealed
opponent team slots with distinct learned unknown-mon tokens (refined in-lineup by the transformer so
both heads attend over the imagined mons), and `BeliefHead` aux-supervises those refined tokens to
predict each hidden mon's species + moves (privileged labels, training-only, never in the forward).
Under `--opp-belief-latent-coef>0` `BeliefHead` also carries an asymmetric SimSiam **latent** predictor:
each believed slot's refined token is regressed (cosine) toward the stop-grad `pokemon_encoder` role-token
of the TRUE hidden mon — graded identity supervision the hard species CE can't give (target from a
training-only `belief_target_slots` obs key, stashed for the loss only, never in pi/vf — leak-safe).
`MoveBelief` (`--move-belief-mode`) predicts + reinjects each opp slot's moveset into its token (and under
`--move-prior-fusion` fuses the Smogon move-frequency **prior** into that prediction as a log-odds residual
+ pins revealed moves certain, so the belief is a unified posterior — *known certain, unknown prior⊕learned*;
`--move-belief-prefuse` moves this reinjection BEFORE the transformer so the believed moves co-refine through
attention instead of being grafted on after);
and `DamageOperator` (`--damage-op`) consumes that move belief's predicted moves to compute the believed-move
incoming damage to each of our mons (a differentiable gen3 calc), appended to **both** projection heads —
so the gradient sharpens the move belief toward real KO threats (`designs/ai_v6/design_differentiable_damage_op.md`);
its per-OUR-move OUTGOING block carries per-status SECONDARY probabilities (accuracy-folded, ×Serene Grace /
Shield Dust — `gen3_unified_move_system_v1`; over the 7 columns an OUR-side gen3 move can actually inflict,
`gen3_op_block_trim_v1`). Under `--damage-topk K` (which implies `--damage-matrices incoming`) it ALSO emits
the **DISCRETE incoming move-space** — the v35 `_incoming_matrix`: the opp active's K most-believed moves
INDIVIDUALLY, each with its move LATENT identity (gathered from `MoveLatentEncoder`, typed-HP-aware) +
belief + per-move effect/secondary bits + per-OUR-mon `[low, high, crit, pko, type_mult, status_lands]` — so
the policy reasons in the discrete move space (anticipate the move, pick the damage-/status-immune safe
pivot, e.g. Thunder-Wave→Ground=0) instead of only the collapsed worst-case. (The v30 LEAN top-K block this
supersedes, and the op's two opp-active-level effect/secondary COLLAPSES, are deleted at v55 — see
`gen3_op_block_trim_v1` below.) Inside `PokemonEncoder`, the
flag-gated `MoveLatentEncoder` (`--move-latent`) concatenates a context-free mechanics-grounded per-move
latent into the move network; its latent table is the Stage-3 similarity-grading target
(`--move-belief-latent-coef`, so Rock Slide ≈ Hidden Power Rock — `designs/ai_v6/design_unified_move_system.md`).
A separate optional `WinProbHead` (`--win-prob-mode none|read_only|shaping`) reads `value_pooled` and emits a
calibrated **P(win)** logit — a SIDE readout (stashed for the aux loss + the prober, **never** in pi/vf, so
projection dims are unchanged), supervised by the Monte-Carlo episode outcome (win=1/loss=0); `read_only`
stop-grads its input (a risk-free diagnostic), `shaping` lets the win objective shape the trunk. A sibling
`PubValHead` (`--pubval-mode`, v43) applies the same pattern with an EXOGENOUS target — the frozen
human-replay-calibrated public value V_pub (`data/gen3_pubval.json`), a dense per-step credit-assignment
signal from outside the self-play bootstrap (the v43 note below).
`forward` returns a `(pi_features, vf_features)` tuple — the transformer body is shared, but the
actor and critic read it through independent CLS pools and projection heads (the
**value-dedicated CLS readout**, H4 / Option C). It must be paired with
`Gen3DualHeadMaskablePolicy` (`src/agents/model/policy.py`), which unpacks the tuple and routes
each half to its own `mlp_extractor` branch; stock SB3 policies assume a single-tensor extractor
and won't work. **The action head is POINTER-NATIVE (v51, `gen3_pointer_native_v1`, no flag):**
the policy's `_build` deletes SB3's flat `action_net` (a raising stub takes its slot) and the
`PointerNativeActionHead` scores each action from the token of the entity it selects — move logit
k ← the REQUEST-slot-k move token ⊕ its op cells, switch logit j ← our-team token j ⊕ its
incoming/OAX cells, struggle ← the context — with `latent_pi` (the policy tower's output, so the
op block / beliefs / FiLM all condition it) as the shared context. Position-equivariant by
construction; zero-init scorers ⇒ uniform-over-legal cold start. **Since v52
(`gen3_entity_move_seats_v1`) MOVES are also attention SEATS in the trunk** — E3 (our active's 4
request-ordered move tokens, unconditional; the pointer head reads the REFINED seats) + E4 (the
opp active's top-`entity_topk_seats` believed threat moves) append after the global token via
`TeamTransformer`'s generic `extra` seat path (the v52 entry below). Both projection input dims
are auto-discovered via a dummy forward pass in `__init__`, so they stay correct as the
architecture changes.

**The phase-by-phase data flow (the 7-phase contract, dims, and the `ExtractorContext` /
`Embeddings` ownership rules) is documented in `src/agents/model/CLAUDE.md`.**



---

## 4. Superseded prose — `src/agents/model/CLAUDE.md` § Model versioning, per-version entries

Moved verbatim on **2026-08-08** (lines 297–1524 of the pre-split file). This was a SECOND
per-version narrative running in parallel with §1 above: it covered the same v6–v57 ground in
different words, and the two had drifted apart — the leaf stated `ARCH_SIGNATURE` twice with two
different values and `MODEL_CONFIG_VERSION` fourteen different ways in different paragraphs, none of
them the live 59. Keeping one changelog is the point of the split.

What stayed in the leaf is the versioning **mechanics** (what to bump when, the optimizer-reorder
guard, the resume-immutable-hparam playbook) — the part that is a rule rather than a record.

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
only. (3) **The DamageOperator effect block** gains per-status SECONDARY probabilities — ⚠️ the INCOMING
half described next was DELETED at v55 (`gen3_op_block_trim_v1`: 0.1% of measured dependence, no defender
axis — the per-move/per-defender form in `_incoming_matrix` replaces it), and the OUTGOING half now spans
only the 7 columns an OUR-side gen3 move can inflict (`_OUT_SEC_COLS`) — incoming
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
can't anticipate the discrete move or pick the immune/safe pivot. ⚠️ **The LEAN block described in this
section was DELETED at v55** (`gen3_op_block_trim_v1`) — v35's `_incoming_matrix` is its strict superset and
had already suppressed it in every production config (measured 0 calls/forward), so `damage_topk_k` now
sizes only that matrix. Read this section as the design rationale, not the live layout. This adds a
**discrete top-K block**
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
`gen3_per_move_matrices_v1`).** The ENRICHED evolution of the v30 top-K block
(`DamageOperator._incoming_matrix`) — it **REUSES `damage_topk_k` as its K** (one knob — `--damage-topk K`,
try 4/5/6, default 5). It originally SUPPRESSED the lean top-K at that K; since v55
(`gen3_op_block_trim_v1`) the lean block is DELETED outright, so this is the ONLY block K sizes and
`--damage-topk K` without `--damage-matrices incoming` RAISES rather than emitting nothing. Per opp-active top-K move:
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

**DISCRETE typed Hidden Power, end to end (v52, `gen3_typed_hp_belief_v1`).** The model never reasons over a
typeless Hidden Power. Supersedes v38 `gen3_opp_hp_typed_candidates_v1` / `gen3_opp_hp_type_belief_v1`, which
made only the DamageOperator typed while the belief, its labels, its prior, the token reinjection and the latent
grading all still spoke in the typeless num 237.

- **One composition, upstream of every consumer.** `HPTypeBelief.compose_typed_hp(move_logits, posterior,
  obs_hp_probs, opp_move_ids)` rewrites the raw move-belief posterior so HP exists ONLY at the 16 real typed
  move-nums 355-370, each carrying `logit(presence · P(type))`, with the bare 237 set to
  `_HP_PRESENCE_OFF_LOGIT` (−30: hard off but FINITE, so the multi-label BCE sees ~0 loss and no NaN). 237
  survives only as the belief's internal PRESENCE channel, read immediately before it is masked. It runs inside
  `_apply_move_belief`, between the move head's read and the reinjection — so `last_move_belief_logits`, the
  tensor the damage op / top-K / BCE / latent grading / prober all read, is already typed, and the soft-embed
  reinjection pulls on the REAL typed move-embedding rows instead of the typeless one (whose `MOVE_ATTR` /
  latent rows are deliberately all-zero).
- **The constraint is structural, not a penalty.** `Σ_t P(HP_t) == presence`, and `MoveBelief.move_logits` pins
  presence to `_REVEAL_LOGIT` the moment `hiddenpower` is revealed — so a seen Hidden Power can never be
  believed away. The belief can only be unsure ACROSS TYPES, which is the honest state.
- **Two certain-fact eliminations** ("discard the ones that don't make sense"), both inside the composition:
  **moveset exhaustion** (4 moves revealed and none is HP ⇒ presence 0; derived from `opp_move_ids` alone, so
  it needs no plumbing and agrees by construction with `HiddenPowerTracker.mark_no_hp`) and **effectiveness
  narrowing** (once HP has FIRED, the tracker's zeros in the obs `hp_probs` are CERTAIN physics → restrict the
  type belief to the survivors and renormalise, with a uniform-over-survivors fallback so an off-meta HP is
  never renormalised back to ~0, i.e. re-immuned).
- **The head is UNCONDITIONAL** whenever `move_belief is not None` — there is no `off`. The v38 tri-state's
  `off` was a correctness bug behind a flag: it left a typeless BP-0 candidate in the damage sweep AND sourced
  the type from the obs `hp_probs`, which is all-zero until the opponent actually fires HP, so a REVEALED
  Hidden Power was priced as nonexistent. It also no longer requires `damage_op`: the composition lives in the
  belief, so the typed posterior reaches the token reinjection / BCE / prober even with no operator.
  `--hp-type-belief-coef` (default 0.05) now controls only whether the privileged CE supervises the head on
  top of the damage + BCE gradients it already gets.
- **The op is a plain consumer.** `_opp_candidate_weights(ctx, move_belief_logits)` just masks the 237 presence
  channel; `hp_type_fix`, `SPECIES_HP_PRIOR` and the `hp_type_belief` forward argument are GONE, and
  `HP_CAND_MASK` no longer zeros the typed nums. This closes a real divergence: `forward` was passed the
  learned posterior while `refine_candidates` was not, so the between-layers refine kernels priced HP off the
  Smogon prior while the head block priced it off the learned belief.
- **Labels use the TRUE TYPED num** (`gen3_env._move_num` no longer folds to 237). A 237-keyed label supervised
  a dead channel while leaving the 16 typed ones as BCE NEGATIVES — the head was being trained toward "this
  opponent has no Hidden Power of any type", fighting the composition. Leak-safety is unchanged: these are
  training-only Dict keys carrying exactly the privileged fact `hp_type_label` already carried, and the
  OBSERVATION still shows the opponent's HP bare, so the type must still be guessed.
- **The PRIOR keeps its factorisation** and loses nothing: `damage_tables._belief_num` still sums typed usage
  onto 237 (= `P(runs SOME HP)`) and `build_hp_type_prior` holds the conditional `P(type | has HP)`; their
  product reconstructs each typed HP's own Smogon rate exactly (pinned by a test). The **learnset gate is
  fixed**: `gen3_learnset.json` lists only the bare `hiddenpower` (the type is an IV choice), so the gate used
  to drive all 16 typed nums to the `eps` IMPOSSIBLE floor for every species — harmless only because the
  composition overwrote those cells, which is exactly the GIGO shape not to leave lying around.
- **Deliberate hold-out: the TURN-HISTORY opp-move slot keeps 237.** The history records what was OBSERVED, and
  the type genuinely was not; there is no sound way to type it after the fact.
- **GIGO guard** (`build_damage_buffers`, throwing) is unchanged: `HP_TYPED_NUMS` is data-derived (via
  `damage_tables._hp_typed_nums()`) and the builder asserts `MOVE_TYPE_IDX[355+j]==HP_TYPE_IDX[j]`,
  `MOVE_BP[237]==0`, `MOVE_BP[typed]==70`.
- **Versioning:** the forward math changed with out_dim + projection widths UNCHANGED, so nothing shape-based
  catches it → **`ARCH_SIGNATURE` bumped to `gen3_typed_hp_belief_v1`**. `hp_type_belief_mode` is DELETED from
  `ModelVersion` / `check_compatible` / `extractor_arch` / `snapshot`, and `_migrate_config` **POPs** it
  (`from_json_file` does `cls(**data)`, so a stale key would raise a bare `TypeError` instead of the clear arch
  error). `hp_type_belief_coef` stays training-only. Retrain-class.
- **Prober:** unchanged and now unambiguous — the op's top-K candidates are real move-nums decoded via the
  NORMAL num→id path with the type preserved (`hiddenpower(ice)`); the bare 237 is never selected.

Tests: `hp_type_belief_test.py` (the Σ-typed-equals-presence constraint under adversarial posteriors, 237
hard-off + non-HP channels untouched, both eliminations, the off-meta-survivor regression, the immune bug in
the state that produced it, the op having no HP source + its two candidate sites agreeing, the two-distinct-
typed-HPs top-K, the typed-row soft-embed, cold-start==prior, grad to BOTH factors, the unconditional head, the
prior-factorisation and learnset-gate data pins, the v52 migration) + the extended
`poke_env_gaps/belief_labels_fuzz_test.py` (typed labels == agent2's real movesets + HP types, no-leak, live).

**The HP-belief factorisation ablation (v53, `gen3_hp_belief_ablation_v1`, `hp_belief_mode` /
`--hp-belief-mode {composed,flat}`).** Measures what the v52 presence×type factorisation is actually WORTH
against the null hypothesis "the multi-label move head can just learn 16 independent typed channels".

- **What is NOT the variable:** BOTH arms reason over the discrete typed nums 355-370 and drive the bare BP-0
  num 237 hard-off via the shared module-level helper `mask_typeless_hp` (both arms provably agree on it —
  pinned by a test). The typeless candidate is the original "opp HP reads immune" bug, so leaving it live in
  the `flat` arm would be re-introducing a correctness bug, not an ablation arm.
- **`composed` (DEFAULT)** is byte-for-byte the v52 forward: `HPTypeBelief` + `compose_typed_hp` (the
  structural `Σ_t P(HP_t) = presence`, reveal-pinned, + moveset exhaustion + effectiveness narrowing).
- **`flat` (the ABLATION)** builds NO `HPTypeBelief`: `_compose_hp_typed` short-circuits to
  `mask_typeless_hp(raw_move_logits)`, so the move head predicts the 16 typed channels INDEPENDENTLY — under
  `--move-prior-fusion` each off its own real per-typed Smogon usage rate, which `build_move_prior` already
  writes at 355-370 beside the 237 presence sum (no prior plumbing needed). No factorisation, no reveal
  constraint, no tracker narrowing: a seen Hidden Power CAN be believed away here — that is the point.
- **CLI ergonomics:** `--hp-belief-mode flat` AUTO-ZEROES `--hp-type-belief-coef` with a loud note (no head ⇒
  no posterior for the CE to supervise; the `--zarch-recon-coef` single-team auto-zero precedent — the coef
  defaults to 0.05, so erroring would make the ablation flag fail out of the box, while a silent no-op coef is
  the failure that actually matters).
- **Versioning:** STRUCTURAL — `flat` drops a module, so it is a state_dict change as well as a forward one;
  STRING-gated in `check_compatible` (the `win_prob_mode` pattern), fresh-only; `_migrate_config` defaults
  pre-v53 configs to `"composed"`; threaded through `extractor_arch` (v53 row) + `snapshot`. Default
  byte-identical → NO `ARCH_SIGNATURE` bump.

Tests: `hp_type_belief_test.py` (`flat` builds no head + masks 237 + leaves typed channels untouched, the
shared-mask agreement, the version gate mismatch, the invalid-mode raise, the v52→v53 migration default).
`MODEL_CONFIG_VERSION` = **53** at v53.
Tests: `hp_type_belief_test.py` (the immune-bug-and-fix, 237-always-masked + C=n_moves, narrowing + off-meta
fallback, cold-start==prior, the two-distinct-typed-HPs top-K at real nums 363/365, grad flow, modes, CE, the
GIGO/version gate, the v2 reinjection) + the extended `poke_env_gaps/belief_labels_fuzz_test.py` (HP-type label
== agent2's real type + no-leak, live). `MODEL_CONFIG_VERSION` was **38** at v38; the current value is **55**
(the v55 op BLOCK TRIM note below); `ARCH_SIGNATURE` is now **`gen3_op_block_trim_v1`** (v55, the op block
trim — before it `gen3_entity_move_seats_v1` [v54, the move-entity seats], `gen3_typed_hp_belief_v1`
[v52, the typed-HP belief note above] and `gen3_pointer_native_v1` [v51, the pointer-native head — see that
note]; `gen3_opp_hp_typed_candidates_v1` was current v38–v50).
== agent2's real type + no-leak, live). `MODEL_CONFIG_VERSION` was **38** at v38; the current value is **40**
(the v40 nature/EV note below); `ARCH_SIGNATURE` is now **`gen3_edge_bias_trunk_v1`** (v55, the edge-bias trunk — before it `gen3_entity_move_seats_v1`, v54, the
move-entity seats — before it `gen3_typed_hp_belief_v1` [v52, the typed-HP belief note above] and
`gen3_pointer_native_v1` [v51, the pointer-native head — see that note];
`gen3_opp_hp_typed_candidates_v1` was current v38–v50).

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

**Pre-attention unified damage operator (v50, `damage_op_prefuse` / `--damage-op-prefuse`,
`gen3_damage_op_prefuse_v1`).** ONE damage computation per forward instead of two. v47 made the
*belief* single-compute; the *physics* was still computed twice — a LEAN `discrete_*` recompute inside
the between-layers refine loop (× `damage_refine_rounds`) **plus** the FULL 835-dim block after the
transformer. When on, `_spread_hp_damage` (the shared helper holding `SpreadBelief` + `HPTypeBelief` +
the move-latent table + the full `DamageOperator`) runs on the **PRE-transformer opp role tokens**, the
per-OUR-mon incoming rows `[B,6,_DMG_PER_MON]` are added to our role tokens through the zero-init
`prefuse_proj`, and the same block is concatenated into both projection heads:

> beliefs ONCE (pre-attention) → physics ONCE → attention over that physics → heads read that same block

**The case is CPU cost.** At B=1 on CPU — the PFSP frozen-opponent forward, once per decision per env,
on the rollout critical path — the forward is 6.45 ms across 14,337 aten calls (~0.44 µs each ⇒
DISPATCH-bound, so the lever is the NUMBER of ops issued). The op is ~75% of that (2.454 ms
post-transformer + ~2.4 ms refine loop); the attention layers themselves are 0.266 ms (4%). Measured
`--damage-refine-rounds` 2→1 = **+14.0%**, 2→0 = **+28.2%**. **The architectural story is secondary and
this codebase's evidence is against it** — physics-into-the-trunk measured NULL 3-for-3 (ledger K9/K10:
`--damage-reattend`, the refine/threat channels, `--pubval-mode`), and K10a showed the lean kernel was
already a good proxy (91.8% argmax agreement on damage) so "fresher/fuller physics into attention" is a
difference of FORM, not content. Do not sell this as a strength lever.

**What it risks, and what bounds it.** Ledger P1 measured the op's HEAD CONCAT as the policy's largest
single dependency (zeroing the block = masked KL 0.9385 all / 0.4948 moves), so the block must keep
reaching the heads at full width — it does; only its INPUTS move from refined to un-refined tokens. Two
properties bound that shift, both pinned in `damage_op_prefuse_test.py`:
1. the toggle **requires `move_belief_prefuse`**, so the move belief — the op's dominant input — is
   computed pre-transformer in BOTH shapes and is **bit-identical**; the only re-sourced inputs are the
   **spread** and **HP-type** posteriors;
2. at **cold start the damage block is bit-identical** pre vs post, because every belief head is
   zero-init ⇒ its posterior is token-INDEPENDENT (move == Smogon prior, spread == usage prior, HP type
   == Smogon prior). The divergence is created by TRAINING, not by the reordering.
`tmp/damage_prefuse_kl.py` measures the trained-weights shift (block relative-L2 + masked KL) against
the zero-block ceiling re-measured in the same tree. **Measured (3000 real bridge decision states, the
500k-step `tmp/prefuse_probe_train.sh` snapshot, same weights, injection zeroed):** block cosine
**0.988** (relative L2 mean 0.054 / median 0.009), masked KL(post‖pre) **0.0005** vs a re-measured
zero-block ceiling of **0.0182** ⇒ the re-sourcing is **3.0% of the "delete the block" ceiling**, with
**2.9%** of argmax actions flipping. Read it as a FLOOR: the snapshot is short-trained (its absolute
ceiling 0.0182 is far below P1's 0.9385 on a fully-trained model — the ratio is the comparable number,
not the levels), belief heads further from zero diverge more, and a fresh run trains UNDER the
pre-attention shape rather than being swapped into it. **CPU win, same benchmark as the motivation**
(`tmp/pfsp_opponent_sweep.py`, B=1, 1 thread, min of 200, idle box): **6.452 → 4.617 ms = +28.2%**,
−4,126 aten calls. Note that `--damage-refine-rounds 0` alone measures 4.620 ms — i.e. **the CPU win IS
the refine-loop deletion**; what the prefuse adds on top is ~free (one zero-init Linear over 6 tokens)
and buys back a pre-attention physics path the bare deletion would lose.

STRUCTURAL — `prefuse_proj` is a saved parameter — so it is gated in `check_compatible` with a bool
compare (a mismatch is a state_dict mismatch on EVERY load, eval/pool/distill included); OFF is
byte-for-byte (**NO `ARCH_SIGNATURE` bump**). Requires `damage_op` + `move_belief_prefuse`, and is
**mutually exclusive with `damage_refine_rounds > 0`** (the loop is what it replaces; running both would
restore the double cost). Consequence to know: the v36/v37 OUTGOING + STATUS trunk residuals ride that
loop and are therefore NOT reproduced here — they are K10-null channels, and re-adding them would need
their own projections. `move_belief_single_compute` becomes redundant (with no refine loop the belief is
computed once by construction) but is harmless. Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites. Tests:
`damage_op_prefuse_test.py` (incl. identity-at-init on a REAL `MaskablePPO`-built policy — ledger M1).
**Pointer-NATIVE action head (v51, `gen3_pointer_native_v1`, NO flag — the fresh-generation reset,
`designs/ai_v9/design_pointer_action_head.md` §0).** The flat positional action head is DELETED and
the pointer head is THE action head, unconditionally. `Gen3DualHeadMaskablePolicy._build` calls
`super()._build` (SB3 creates the flat `action_net` Linear + ortho-inits + builds the optimizer),
then replaces `action_net` with a RAISING stub (`_NoFlatActionNet` — never `Identity`, which would
emit `latent_pi` as logits: a garbage policy, not a crash), builds `self.pointer_head`
(`PointerNativeActionHead`), and REBUILDS the optimizer (else the dead Linear's params ride every
checkpoint's optimizer state and the head's params silently never train — pinned by
`test_optimizer_covers_exactly_the_live_params`). `_get_action_dist_from_latent` — the single funnel
all three logit sites pass through — builds the distribution from the pointer logits directly.
Scoring (position-EQUIVARIANT: one shared scorer per entity token, so no logit row ever learns
"slot j" positionally and the `ordering_integrity.py` sorted-vs-request bug class is unrepresentable
at the logits):
- **move logit k** ← the move at REQUEST slot k (`PokemonEncoder.last_move_tokens`, stashed
  pre-flatten UNCONDITIONALLY, permuted sorted-by-id → request order by MOVE-NUM identity in
  `_request_order_move_tokens`; an unresolved slot is zeroed + `valid`-gated to logit 0) ⊕ its own
  op cells `[low,high,crit,pko, p_land, known, sec×10]` (16, when `damage_outgoing`);
- **switch logit j** ← our-team token j (post-transformer/post-reattend — board-aware) ⊕ its
  incoming row (12) + CB tail `[phys_high_cb_j, pko_cb_j, p_cb]` (when `damage_op`) ⊕ its OAX
  attacker row `[cells×16, p_outspeed_j, alive_j]` (when `damage_matrices_outgoing_all`) — the
  per-candidate defense AND offense read;
- **struggle** ← the context alone.
The context is **`latent_pi`** — the same policy-tower output the deleted flat head consumed, so the
op concat / beliefs / FiLM condition every pointer score. The op owns its cell layout:
`DamageOperator.pointer_cells(damage_block)` slices the SAME post-gain tensor the projection heads
consume (offsets mirrored against `decode_damage_block` by
`test_pointer_cells_match_decode_damage_block_with_every_block_enabled` — the OAX tail is sliced as
`out_dim − _DMG_OAX`, valid because OAX is appended LAST); widths ride
`pointer_{move,switch}_cell_dim` (0 when a source block is off — a missing block NARROWS the
head's Linear, never silently zero-pads). The extractor's side of the contract is the unconditional
per-forward stash `last_pointer_inputs = (tok_req, valid, our_team_out, move_cells, switch_cells)`.
The three scorers are ZERO-init and created AFTER SB3's ortho-init apply (so they survive without
the M1 guard) ⇒ all logits are exactly 0 at step 0 ⇒ the cold-start policy is
**uniform-over-legal** (the correct fresh-run init — there is no flat baseline to be byte-identical
to). Versioning: `MODEL_CONFIG_VERSION` 51; the v49 `pointer_head` FIELD is REMOVED
(`_migrate_config` POPs it — the v48 stale-key lesson); no `check_compatible` gate exists because
there is no off state — **the cross-era break rides the `ARCH_SIGNATURE` bump** (state_dict: no
`action_net.*` Linear, new `pointer_head.*` keys; forward changed for every model). No pre-v51
checkpoint loads (owner decision 2026-08-03: fresh generation — no resume/warm-fork across the
boundary; pools/opponents re-seed from the new lineage). `POINTER_HIDDEN` (64) lives in
`arch_constants.py`. Tests: `pointer_head_test.py` (permutation on a SCRAMBLED moveset, cell-offset
parity vs decode, uniform-at-init + funnel consistency + optimizer coverage + save→load logit
identity on a REAL MaskablePPO policy — the M1 rule).

**Move ENTITY seats (v54, `gen3_entity_move_seats_v1` — Stage 1 of the entity generation,
`designs/ai_v9/design_generation_roadmap.md` §3).** MOVE tokens become first-class attention SEATS
in the unified trunk. `EntityMoveSeats` builds two seat families, appended AFTER the global token
(so every absolute slice — team/history/global, the refine callback's tail-preserving cat — is
position-stable), entered via `TeamTransformer.forward`'s new generic `extra=(tokens, types, pad)`
path (third return = the refined extra seats; the token-type table grows 4 → 6:
`TOKEN_TYPE_OUR_MOVE`/`TOKEN_TYPE_THEIR_THREAT`):
- **E3 (unconditional)** — our active's 4 move tokens, permuted sorted-by-id → REQUEST order ONCE,
  pre-transformer (`_request_order_move_tokens`, the same call the v51 stash used post-transformer),
  projected 32 → d_model by `move_seat_proj`. An unresolved slot is a zero token AND key-masked.
  **The pointer head now reads the REFINED E3 seats** (`last_pointer_inputs[0]` is `[B,4,d_model]`;
  `fe.pointer_move_token_dim` = D_MODEL sizes the head's `move_proj`) — its move tokens are
  board-aware, the Stage-1 payoff.
- **E4 (`entity_topk_seats` / `--entity-topk-seats K`, 0 = off)** — the opp active's top-K believed
  threat moves: `DamageOperator.refine_candidates(k=K)` (the SAME belief-weighted typed-HP-scattered
  candidate definition the refine kernels use — one source), each seat
  `threat_seat_proj([latent(32) ⊕ w ⊕ acc ⊕ is_phys])`; idx detached, `w` differentiable (the
  belief gradient rides the seats); all K key-masked + zeroed when no opp active. Requires
  `damage_op_prefuse` + `move_latent` (the candidate weights + latent table must exist
  pre-transformer; `_spread_hp_damage` stashes `_entity_latent_table`).
Seat projections are ordinary trainable Linears (the `history_proj`/`global_proj` convention — NOT
zero-init; they carry new information, not a residual). Versioning: E3's break is UNCONDITIONAL →
the `ARCH_SIGNATURE` bump carries it; `entity_topk_seats` is a STRUCTURAL int in `check_compatible`
(the `damage_topk_k` pattern), `_migrate_config` v54 setdefaults it to 0. Measured B=1 (threads=1):
E4 K=5 = +0.18 ms on a ~3.1 ms prefuse-stack forward (the spike predicted +0.19 — dispatch-bound).
Tests: `entity_seats_test.py` — seat-layout stability, the masked-seat BIT-IDENTITY no-leak (all-True
pad ⇒ team tokens byte-identical to the no-extra forward), the op-candidate single-source, the E4
requirement gate, and the gradient probe (NOTE: probe LayerNorm-output paths with a RANDOM cotangent —
`.sum().backward()` lies in LN's backward null space and reads ~0 on a perfectly live path).

**The DamageOperator BLOCK TRIM (v55, `gen3_op_block_trim_v1`, NO flag).** The operator sheds its three
least-used output families and one dead code path, acting on the ledger-**P1** per-block dependence
ablation (4000 real eval states, exact producing snapshot, per-block zero → masked KL; ceiling = zeroing
the whole op = 0.9385). It is a DELETION, not a toggle — there is no off state and no new
`ModelVersion` field.

- **Incoming per-STATUS SECONDARY (10 dims) — 0.1% of the ceiling, the most INERT channel in the op.**
- **Incoming believed-EFFECT (6 dims) — 1.2%.**
  Both were opp-active-level belief-weighted maxes with **no defender axis**, so they answered "can the
  opponent flinch someone" without saying *whom*. v35's `_incoming_matrix` already carries the same facts
  **per move** (`_DMG_IMX_HDR_EFFECT` / `_DMG_IMX_HDR_SEC`) and **per defender** (`status_lands`) — ledger
  **P4** measured the un-collapsed form at KL 0.0005 against the collapse's 0.1446. Deleting them also
  removes the whole UNMASKED-belief read `w = sigmoid(...)` from `forward` (with the `_EFF_*`/`_SEC_*`
  sparse-gather buffers), leaving `_opp_candidate_weights`' `w_all` as the op's single belief read.
- **Outgoing slp/psn/tox secondary columns (12 dims = 4 moves × 3) — STRUCTURAL ZEROS.** This block prices
  OUR moves, and (measured, not asserted) gen3 has **no damaging move that inflicts sleep at all**, while
  the psn/tox carriers appear on **1 / 0 of the 773 `data/teams/` teams**. `_OUT_SEC_COLS` /
  `_OUT_SEC_KEEP` keep the surviving 7 in SECONDARY_COLS order so the gather stays a straight index.
  The INCOMING side keeps all 10 — it faces the OPPONENT, who is not restricted to our team pool.
- **`_topk_block` — the v30 LEAN top-K, measured at 0 CALLS PER FORWARD.** It is a strict subset of the
  v35 incoming matrix (header ⊂ header, cell `[high, pko, status_lands]` ⊂ `[low, high, crit, pko,
  type_mult, status_lands]`), and the matrix suppressed it at the same K — so in every production config
  it was dead code sitting in the op's hottest file. `damage_topk_k` now means **"the incoming matrix's
  K"** and nothing else: `K>0` without `damage_matrices_incoming` **raises** in BOTH the extractor and the
  op (never a silent empty block), and the CLI auto-enables the matrix when `--damage-topk` was set with
  no explicit `--damage-matrices` (the `--unified-moves` path, which auto-sets K=5).

Net **−28 dims** off both projection heads (op `out_dim` 835 → 807 on the full toggle set; `incoming_dim`
101 → 85). **This is a dims/complexity change, NOT a throughput one — say so.** Measured B=1 CPU forward
on the production stack: 4.276 → 4.289 ms and 3534 → 3525 profiled calls/forward, i.e. **no change** (the
deleted work is ~9 aten calls on a ~3.5k-call dispatch-bound forward, and the lean block already never
ran). The honest residual: with `--damage-matrices incoming` OFF there is now no effect/secondary signal
at all — that is the accepted trade at 1.3% of measured dependence.

Versioning: the projection widths DO change, so a stale checkpoint would fail on a `load_state_dict` shape
mismatch — the **`ARCH_SIGNATURE` bump to `gen3_op_block_trim_v1`** turns that into a clear arch error
instead. `MODEL_CONFIG_VERSION` 55 is a stamp only (no field added or removed, no migration work).
Tests: `damage_op_test.py` (the retargeted discrete-move-space family now reads `incoming_matrix`, the
three-way `damage_topk_k` dependency guard incl. the op's own raise, the DATA claim that no gen3 damaging
move inflicts sleep, and the decode's absent-key assertions) + the constructed-physics oracle
`poke_env_gaps/damage_op_probe_fuzz_test.py` (22/22).

**Orthogonal to v54's entity seats** — those add SEATS to the trunk, this trims the op's HEAD-CONCAT
output — so the two compose; only the single shared `ARCH_SIGNATURE` string had to be sequenced.

**Edge-bias trunk (v56, `gen3_edge_bias_trunk_v1` — Stage 2 of the entity generation).** The
encoder stack becomes `BiasedEncoderLayer` (fused-qkv clone of the stock layer; attention takes an
additive per-pair per-head float bias [B,H,n,n]; the key-padding mask rides the SAME tensor as a
`_KEY_PAD_NEG` = -1e9 addend — stock-masked-layer parity pinned by
`edge_bias_test.test_layer_matches_stock_transformer_layer`). `TeamTransformer.forward` builds the
bias ONCE per forward (shared by every layer) and takes an `edge_bias_fn` closure; `EdgeBias`
(`edge_bias_families` "off" | the FROZEN "d"=d1,d3 alias | an explicit comma list of
d1,d2,d3,s1,s3 — growing the valid set is NOT a version bump, the string gate catches mismatches)
writes the families at CONTIGUOUS seat-block slices (D2's opp-ACTIVE column is batch-varying →
delivered via a one-hot outer product):
**D1** `DamageOperator.pairwise_outgoing` (a reshape of the validated `_outgoing_matrix` — cell
`[low,high,crit,pko,type_mult,revealed]`) at (E3 seat k, opp-mon seat d) + transpose; **D2** `pairwise_bench_outgoing` (the v39
`_outgoing_attacker_matrix` move-collapsed — cell `[best_high,best_pko,p_outspeed,alive]`) at
(our-mon seat i, opp-ACTIVE seat); **D3** `DamageOperator.pairwise_incoming` (the pre-collapse
`_incoming_rolls`, factored out of `discrete_incoming` so refine + edges share ONE physics body —
cell `[high,pko,eff,is_phys,w]`) at (E4 seat c, our-mon seat i) + transpose, priced at the SAME
detached candidate selection the E4 seats stashed (`EntityMoveSeats.last_cand`); **S1/S3** the
v27/v37 status kernels' `per_pair=True` branches (same physics bodies, the category collapse not
taken) — S1 `[land, land·immob]` at the E3 pairs (requires damage_op+outgoing), S3
`[land, land·immob, w]` at the E4 pairs (requires entity seats); **V** `pairwise_speed` — the
full mon↔mon P(outspeed) block, cell `[p_outspeed, both_alive, revealed_j]` (real our spread vs
believed/neutral opp spread; public para ×0.25 both sides; NO stage boosts in v1 — the
coarse-signal convention; requires damage_op; ⚠️ **v58 GIGO fix 2026-08-06**: this kernel [+
C1's outspeed] shipped reading stat index 4 = SPECIAL DEFENSE as "speed" — both trained
generations' V edge priced bulk; the main op's index-5 p_outspeed paths were always correct.
Fixed via the named `_BS_*`/`_NAT_*` stat indices [use those, never bare integers — two stat
layouts coexist: base/iv/ev = hp,atk,def,spa,spd,spe vs nature = atk,def,spa,spd,spe] +
`consequence_edges_test.test_speed_reads_the_speed_stat_not_spd`, proven to FAIL on the buggy
kernel. Values-only → v58 is a version STAMP; pre-v58 checkpoints load but their v_map trained
on the buggy feature) at the static (our, opp) mon block; **D4** `pairwise_bench_incoming` — the missing
"what does the bench threaten" quadrant, per opp mon j the top-K_bench=4 candidates from its own
slot of the composed posterior, de-timid attacker + real-spread defenders + our screens,
revealed/alive-gated, ACTIVE column zeroed (D3's quadrant; requires damage_op) at the same
mon↔mon block; **T** `pairwise_trap` — P(cannot switch) both directions from the three gen3 trap
abilities (`build_trap_tables`: fail-loud id resolution + `SPECIES_TRAP_PRIOR` incl. the Levitate
column for the grounded check; our side exact, opp trapper revealed-exact else prior, unrevealed
opp victim → 0; requires damage_op) at the mon↔mon block; **X** `pairwise_entry` — entry/exit costs per mon (Spikes chip ×
grounded, Pursuit exposure both directions, Dark eff), delivered at the (mon, GLOBAL seat)
pairs (requires damage_op + damage_op_prefuse — the Pursuit belief must exist pre-trunk); **G**
`pairwise_schedule` — the per-mon end-of-turn HP ledger (Leftovers / weather chip / status tick /
active Leech, signed maxhp fractions; Toxic flat in v1 — the ramp is an E2 follow-up) at the same
(mon, GLOBAL) route (requires damage_op); **C4** `pairwise_protect` — the Protect-consequence
edge at the (Protect E3 seat, GLOBAL) pair: [is_protect, p_success, the two actives' G-ledger
nets] (requires damage_op); **C1** `pairwise_boost` — the first HYPOTHETICAL-WORLD damage
consequence: per (E3 setup-move seat k, opp mon j) the DELTA cells `[is_boost, d_best_high,
d_best_pko, d_outspeed]` from RE-RUNNING the validated `_outgoing_matrix` kernel under slot k's
post-boost stages (`boost_delta` threaded into the kernel's stage read — None byte-identical;
`MOVE_SELF_BOOSTS` from `MoveData.self_boosts`, the ~17 declarative pure-setup moves, PLUS the
runtime **non-Ghost Curse branch** — owner-prioritized, CurseLax/Curse-Registeel: `CURSE_BOOSTS`
[+1 atk/+1 def/−1 spe] from `gen3_mechanics.CURSE_NON_GHOST_BOOSTS`, gated by the user's live
types via `TYPE_IS_GHOST` since a type-conditional move can't live in the type-blind table [which
doubles as the rust engine's draw-free contract — guarded against growing a Curse row]; the −1
spe reads as a NEGATIVE d_outspeed; a Ghost user's Curse stays a zero row. Belly Drum = the
recorded TODO [niche]: needs an hp_cost cell channel + a fails-below-half gate + the C1b
incoming-at-halved-HP re-run; Defense Curl/evasion moves stay unpriced) + the `pairwise_speed`
recipe at the active row WITH stage folding for the spe
delta; **C1b (2026-08-06) completes the INCOMING halves** — `pairwise_boost_incoming` re-runs
the D4-recipe attackers (per opp mon j its own slot's top-k believed candidates, de-timid,
revealed+alive-gated, the ACTIVE column KEPT unlike D4 — it's who you boost in front of) vs
OUR ACTIVE as the lone defender (real spread + CURRENT def/spd stages) with the 5 worlds
(current + 4 slots) on a WORLD axis so the attacker side computes once, emitting
`[d_in_high, d_in_pko]` (≤0 — Iron Defense shrinks the worst physical believed hit and ignores
special, Amnesia the reverse, SD reads ~0 BY PHYSICS, Curse's +1 Def now prices); the cell is
the 7-wide concat `[is_boost, d_high, d_pko, d_outspeed, hp_cost, d_in_high, d_in_pko]`
(`_EDGE_C1_CELL` = 7 — `hp_cost` carries Belly Drum's half-max-HP price: the curated
model-side +12-clamps-to-maximize row [the selfBoosts JSON stays pure — the rust draw-free
contract] + the fails-below-half gate in `_setup_deltas`, so a failing BD is a zero row and
a working one is never a free +6), one `_setup_deltas` helper (table rows + the Curse branch) shared by
both kernels so they can never disagree on what a setup slot does; **C3** (2026-08-06)
`pairwise_recovery` — the heal-vs-KO FLIP at the same (E3 seat, opp-mon) route: the shared
`_believed_attackers` block (factored out of C1b — one attacker recipe for C1b/C3) vs OUR
ACTIVE, damage computed ONCE and only the `_rolls` KO ramp re-evaluated at the 5 post-heal-HP
worlds (`build_recovery_tables`/`MOVE_HEAL_FRACTION` from `MoveData.is_heal`: 0.5 plain +
flat-0.5 weather heals [v1 approx of 2/3-sun/1/4-other], 1.0 Rest, Wish
excluded — delayed, the wish obs scalars own it), cell `[is_recovery, d_in_pko,
rest_sleep_turns]` (`_EDGE_C3_CELL` = 3) — the third channel is **Rest's DETERMINISTIC
self-sleep cost** (owner-prioritized 2026-08-06: exactly 2 lost turns / 1 with Early Bird,
/4-normed; our OWN ability is KNOWN → `ABILITY_IS_EARLYBIRD` exact, never a prior; the same
verified `expected_free_turns` tables as C2's opp-sleep channel, `REST_MOVE_NUM`-keyed, zero
on non-Rest slots), family "c3", requires `damage_op`; **C2** (2026-08-06)
`pairwise_status_consequence` — what LANDING our status move DOES, behind S1's land: cell
`[is_status, land, d_their_outspeed, d_in_phys_high, d_sched, d_in_all_slp, e_slp_free_turns]`
(`_EDGE_C2_CELL` = 7) — para → Δ P(we outspeed) at their spe ×0.25 (the T-Wave fact), burn →
the worst believed PHYSICAL hit re-priced at Atk ×0.5 (the shared `_believed_attackers` +
`_active_defender` blocks), brn/psn → the flat −1/8 tick, **TOXIC → its TRUE first tick −1/16**
(distinguished from plain psn by `TOXIC_MOVE_NUM` — they share immunity cat 5; the ramp
thereafter = the G ledger's counter fact), and **SLEEP → the tempo consequence** (their whole
believed threat suspended: `d_in_all` = −worst hit ANY category, + `e_slp_free_turns` =
E[free turns]/4 from `sleep_belief.expected_free_turns` — DERIVED from the verified hazard
tables, per-mon Early-Bird-marginalised via `SPECIES_EARLYBIRD_PRIOR`/`ABILITY_IS_EARLYBIRD`
[revealed → exact 1.0]; Leech Seed deliberately excluded — G/S1's fact); deltas RAW
(decorrelated from `land` — the head composes, the pko×accuracy convention); `land` rides
the validated `discrete_outgoing_status(per_pair=True)` physics; family "c2", requires
`damage_op` + `damage_outgoing`; **C5** (2026-08-06) `pairwise_baton` — the Baton-Pass
RECEIVER edge, the first family on the (E3 seat, OUR-mon) route: per (BP seat, our mon j)
`[is_bp, d_best_high, d_best_pko, d_outspeed]` (`_EDGE_C5_CELL` = 4) — the v39 switch-in
kernel re-run under the new `inherit_stages=True` (every attacker row gets the active's
stages — the post-pass world is one flag away from the world D2 already prices) minus the
neutral baseline; zero deltas with no stages up; active column zeroed; v1 residuals: volatile
passing (Sub) unpriced, the receiver's incoming world a follow-up; family "c5", requires
`damage_op`; C3's weather heals fold LIVE weather via `MOVE_WEATHER_HEAL` (2/3 sun / 1/4
other / 1/2 clear); **and the G ledger's Toxic leg now carries the RAMP** —
−(ticks+1)/16 from the public obs toxic counter (`POKEMON_COUNTER_OFFSET+1`, both sides;
C4's banked-turn nets inherit it automatically); 4 extra kernel runs ⇒ +2.1 ms B=1 EAGER, but the production PFSP path is
COMPILED where the dispatch fuses — c1 is opt-in and not in the gen-2 config (requires
damage_op + damage_outgoing). Each family's map is a ZERO-INIT
`Linear(cell → 2·n_heads)` (one head-set per direction; auto-protected by `restore_identity_init`'s
observation capture) ⇒ families ON is BITWISE-identical to OFF at init. Under non-prefuse configs
D1 passes spread_belief=None (the pre-trunk read would be STALE — gated in forward_internal).
The op head-concat is NOT deleted (deprecation playbook: home first; the per-family bias-ablation
audit decides deletion). Versioning: the layer swap is UNCONDITIONAL (state_dict keys `in_proj.*`)
→ the `ARCH_SIGNATURE` bump carries it; `edge_bias_families` is a STRUCTURAL str in
`check_compatible` (the `win_prob_mode` pattern), `_migrate_config` v56 defaults "off". Measured
B=1 (threads=1): both families = +0.63 ms on a ~3.5 ms prefuse+seats forward. Tests:
`edge_bias_test.py` (stock parity, bitwise identity-at-init, placement-exactness, gates, fullgraph
compile, gradient liveness — random-cotangent probes; D1's zero grad on random obs is CORRECT, its
gates see no revealed opp) + `consequence_edges_test.py` (C1: the setup table rows incl. the
deliberate Belly Drum/Curse exclusions, boost_delta=None byte-identity, SD raises the physical
line / Agility moves ONLY the speed channel, non-setup slots exactly zero, the gate, 12-family
integration + bitwise identity-at-init).

**E5 tail-threat seats (v57, `gen3_entity_tail_seats_v1`, `entity_tail_seats`).** 6 per-opp-mon
seats summarizing the beyond-top-K tail of that mon's composed posterior — `[p_tail, worst_phys,
worst_spec, revealed]` → `tail_proj` + a learned `tail_marker` (NO new token-type row: the table
growing 6 → 7 would break loading in-generation checkpoints into newer code; tail seats reuse
`TOKEN_TYPE_THEIR_THREAT`). K = `entity_topk_seats` (one truncation definition with the E4 seats).
Appended LAST so the pointer stash's E3 slice is untouched. STRUCTURAL bool (adds
tail_proj/tail_marker + 6 seats); gated in `check_compatible`; OFF byte-identical; requires
`damage_op_prefuse` + `entity_topk_seats > 0`. Tests: `entity_seats_test.py` (gate, seat count,
stash stability, finiteness).

Current `MODEL_CONFIG_VERSION` = **57** (v51 pointer-native head + v52 typed-HP belief + v53
HP-belief ablation + v54 move-entity seats + v55 op block trim + v56 edge-bias trunk + v57 E5
tail seats — see those sections), `ARCH_SIGNATURE` = **`gen3_edge_bias_trunk_v1`**.



---

## 5. Superseded prose — `src/agents/observation/CLAUDE.md`, the two DELETED obs blocks

Moved verbatim on **2026-08-08** (lines 301–403 of the pre-sweep file). Both blocks — the 44-dim
action-aligned MOVE-EFFECT block and the 51-dim per-our-mon INCOMING-DAMAGE / OHKO belief block —
were removed from the observation by `gen3_cpu_damage_deleted_v1`; the leaf was still describing
them as present, 250 lines below its own banner saying they were gone. Their GPU homes are recorded
in `designs/ARCHITECTURE.md`.

`agents/observation/incoming_damage.py` (the math core described below) still EXISTS — the reward
PBRS and the prober import it. Only the observation WRITE was deleted, so the physics description
here remains accurate about the module even though the obs block it documents is gone.

**Move-effect block (44 dims, `gen3_move_effects_v1` + `gen3_status_cure_moves_v1`):** 4 move slots in **REQUEST order** (so
feature slot *k* lines up with action logit 6+*k* — enforced via `legal.move_slots` since
`gen3_move_slot_align_v1`; pinned by `move_alignment_fuzz_test.py`) × 11 features each — `is_boost`, `is_heal`,
`is_protect`, `is_phaze`, `is_hazard`, `inflicts_status`, `status_will_land`, `pp_fraction`,
`status_will_land_known`, **`cures_self_status`**, **`cures_team_status`**. The
only per-move signals that previously reached the policy head in action order were base power and
the type multiplier, so for status/utility moves (power 0, neutral multiplier) every option looked
identical at the head — the model could not tell a setup move from a heal from a wasted Toxic, nor
that a move CLEARS status. **`gen3_status_cure_moves_v1`** added the last two bits: `cures_self_status`
(Refresh — clears the user's own status) and `cures_team_status` (Heal Bell / Aromatherapy — clear
the whole party's). They are **static curated facts** (the cure lives in an onHit callback, invisible
declaratively → a curated override in the acquisition tool, like Belly Drum), read by the head against
the per-mon status one-hots it already sees — a **prober-verified gap**: with no cure bit, the head
conditioned its own status onto Recover/switch (intervention: removing a Toxic moved P(recover)/switch
~11pp each) but onto Refresh only ~1.5pp, so it under-used the cure (~1.4% when badly poisoned) and let
Toxic stack. `cures_team_status` is party-scoped on purpose so the model can value Heal Bell off the
BENCH statuses, not just the active's. The
static flags come from the `gen3_data.moves` facade (`MoveData.is_boost/is_heal/...`), derived in
the acquisition tool from the field **Showdown** keys each mechanic on (`flags.heal`,
`volatileStatus`, `forceSwitch`, `sideCondition`, primary `status`, declarative self-positive
boosts) PLUS a curated callback override for **Belly Drum** (its +6 Atk lives in an `onHit`
callback, invisible declaratively); **Memento** is correctly excluded (foe-target negative boosts
+ self-faint). Resolved **live** in the encoder: **Curse**'s setup (only a self-boost for a
non-Ghost user) and `status_will_land`. The latter is a **prior-weighted probability in [0,1]**
(`gen3_mechanics.status_land_probability`), built the same "priors first, then confirmation"
way the matchup cells handle abilities: it is 0 on a certain block (type immunity, already
statused, Substitute — ability-independent), else `1 − P(ability blocks this status)` over the
opponent's ability distribution (`_resolve_ability_distribution` — the Smogon prior for an
unrevealed mon, collapsing to an exact 0/1 once the ability is revealed via `-immune [from]
ability:`). So an unrevealed Snorlax reads ≈0.14 for Toxic (Immunity-dominated) instead of a naive
1. The trailing **`status_will_land_known`** bit disambiguates prior from confirmed — the SAME
routing the per-mon ability block uses for its `known` flag: 1 when the value rests on confirmed
info (a type-certain hard block, or the opponent's ability is revealed via `_ability_revealed`,
the exact predicate `AbilitiesEncoder` uses), 0 when it's still a Smogon-prior estimate a reveal
could move. Without it the model couldn't tell a confirmed 0.0/1.0 from a prior one (a real
discrepancy vs how abilities are routed; this closes it). Sits before the matchups → flows to BOTH
the policy and value projection heads via
`non_matchup_rest` (input widths auto-discovered). Garbage-in discipline: each static flag is
sourced from Showdown's actual representation, never guessed from the move name — see
`tools/pokemon_data_extractor/sync.py:build_moves`.

**Incoming-damage / OHKO belief block (51 dims, `gen3_incoming_crit_split_v1`, at reactive offset 63,
before the matchups → routed to both heads via `non_matchup_rest`):** the opponent active's threat to
*us* as a calibrated belief, not a calc. (This block is the fixed *usage-prior* collapse; the model-side
`DamageOperator` (`--damage-op`) computes the SAME kind of belief from the model's LEARNED move belief
instead, and `--mask-incoming-damage-obs` can zero this block out of the MODEL's view to A/B that
replacement — the block stays in the obs at its fixed dim, and the REWARD PBRS still reads it from
`live_view`. See `src/agents/model/CLAUDE.md` → the damage-operator / unified-belief notes.) Per our 6 team mons (slot-aligned): `[phys_expdmg_frac,
spec_expdmg_frac, phys_pko_nocrit, spec_pko_nocrit, phys_crit_delta, spec_crit_delta, p_outspeed,
threat_revealed]` (8 × 6 = 48), then 3 opp-active recovery scalars
`[recovery_rate, cures_status(P rest), recovery_known]`. **The per-mon field offsets are NAMED
constants (`IDX_PHYS_EXP … IDX_THREAT_REVEALED`, `IDX_RECOVERY_*`) in `incoming_damage.py` — the
single source of truth for this layout.** The producer assembles each slot FROM those names (with a
`_PER_MON_FIELDS == PER_MON` import-time assert) and every single-field consumer reads by name
(the reward PBRS `block[base + IDX_OUTSPEED]`, the fuzz test), so a future field insert can't
silently desync a read — the failure mode that once made the reward PBRS read `phys_crit_delta` as
`p_outspeed` (the crit-split pushed outspeed 4 → 6 but a hardcoded `block[base + 4]` stayed). Whole-
slot reads (the prober decode) full-tuple-unpack, which fails loudly on a width change instead. **`gen3_incoming_crit_split_v1` (PER_MON 5→8,
block 33→51, obs 3391→3409):** P(KO) is the modal `*_pko_nocrit` (the roll integration with NO crit —
the outcome you plan around); the crit risk is exposed as the **DELTA** `*_crit_delta`
(crit-inclusive − no-crit ∈ [0, `_CRIT_P`]) rather than the near-redundant absolute crit-inclusive line
(which equals nocrit + a ≤6% tail and is buried after standardization). The delta is the explicit crit
"tax" — a decorrelated feature a small net can read — so the policy/critic price the modal line without
over-weighting uncontrollable crit RNG (the prober's representation probe flagged the damage SPREAD as
under-encoded, and the plateau diagnosis showed RNG-driven critic craters; the prober reconstructs
crit-inclusive = nocrit + delta to preserve the loss-taxonomy meaning). `threat_revealed` is the
dominant KO threat's `p_in_set` provenance: **1.0 = a revealed move (we KNOW), <1.0 = a usage-prior
GUESS, 0.0 = no candidate can KO** (read jointly with the pko channels) — the "how much are we guessing"
signal (provide-the-fact, not bake-the-prior). P(KO)/expected-damage are the §6.1 belief —
**max over `revealed ∪ usage-prior` candidate moves** of `P(move in set) · P(KO|move)`, routed by gen3
**TYPE-category** (Bug/Rock/Ground/… physical, the rest special), using the gen3 damage formula with a
**fixed-damage branch** (Seismic Toss/Night Shade/Dragon Rage/Sonic Boom carry constant damage despite
the dex STATUS tag; respect type immunity — 0× vs Ghost), a **variable-power branch** (Return/Frustration
read BP 0 in the dex → priced at 102), the gen3 **Explosion/Self-Destruct Def-halve**
(gen≤4), Reflect/Light-Screen/Substitute/burn/weather modifiers, the opponent's offensive-stat tail
(the **0.95 max-EV+ percentile**, `priors.stat_distribution`), and a closed-form roll→P(KO) **blended
with a gen3 crit term** (`_CRIT_P`=1/16, ×2, screen-ignoring). `p_outspeed` is `P(our_spe > opp_spe)`
over the opp Speed *distribution* (the
hidden nature/EV) with observed boosts/paralysis. **v2 belief-VALUE recalibration** (same 33 dims, same
obs dim — values only, so retrain-class not weight-shape): the v1 belief was too timid on the near-OHKO
tail and silently zeroed missing coverage (run_20260606_204351: 17% of direct-hit deaths read
P(KO)<0.25). v2 (1) de-timids P(KO) — the **crit term** + the **raised offensive tail (0.85→0.95)** lift
the KO flag on near-OHKOs while expected-damage re-normalises to the MEAN (∝ `atk_mean`), so the chip
belief is unchanged; and (2) widens the candidate set so the killing move isn't silently absent — a
**revealed bare `hiddenpower`** (dex BP 0) expands into per-type candidates (~70 BP, typed from the **HP
tracker**'s observation-narrowed distribution / Smogon HP prior — the tracker is threaded into
`encode_block`), Return/Frustration are priced, and the prior **floor/cap widen (0.12→0.05, 4→6 per
channel)** so a low-usage super-effective coverage move survives into the pool (the per-defender max over
`p_in_set·P(KO)` is the real type-effectiveness gate, so extra low-usage candidates only ever surface a
genuine SE threat — they can't inflate a neutral one). **Two modules, deep split:** the pure, poke-env-free
math core (formula, roll→P(KO) + crit, P(outspeed), the `Candidate`/`Defender`/`AttackerThreat` beliefs,
`compute_team_block`) is `incoming_damage.py`; the board→belief extraction is `incoming_damage_encoder.py`
behind the single **`encode_block(live, hp_tracker)`** entry — it reads the current board **only through
the `LiveView` read-model** (no raw poke-env battle; `LivePokemon` carries the EV-computed `stats` +
integer `current_hp`/`max_hp` the belief needs), so the SAME `LiveView` built per decision feeds both the
obs path here AND the reward-shaping path (`reward_manager.py` PBRS) — one strict-API source, no
duplicate raw reads. Its only data reads are the per-species usage candidates + HP typing + offensive-stat
distributions, `lru_cache`d so only the per-defender damage math (and the rare revealed-HP expansion) is
per-decision. `reactive.py` passes `live` + the HP tracker; the reward PBRS passes `live` only (HP typing
falls back to the Smogon prior). Priors: `gen3_{move,spread,item,hidden_power}_priors.json` via
`gen3_data.priors`. Belief-not-calc → validated by calibration; the obs golden fixture pins the vector
byte-for-byte (the LiveView migration is value-neutral — golden parity holds against the v2 fixture).


---

## 6. Post-split entries (append-only — newest last)

### v60 — `gen3_entity_rehome_v1` (2026-08-08): Stage 3, the entity re-home

The flat observation's DERIVED blocks are deleted and every raw fact re-homed to its audited
entity home (`design_generation_roadmap.md` §3 Stage 3; executed after the gen-3 40M audit
confirmed the D/V edge families as a trained, load-bearing superset of the matchup signal —
d2 alone 16.3% flips at 40M). Obs **2925 → 2667**:

* **DELETED**: the two 144-dim matchup matrices (`our_matchups`/`their_matchups` — pair
  effectiveness is GPU-side: D1–D4/V edge cells carry `[low, high, crit, pko, type_mult,
  revealed]` from real gen3 physics + the learned belief, vs these matrices' bare `type_mult/4`
  off a fixed prior); `active_status` (byte-redundant with the active mon's condition one-hot);
  `forced_struggle` (derivable — all-zero `active_req_moves` legal bits on a move-request, and
  the action mask stays authoritative).
* **RE-HOMED to the per-mon slot** (`POKEMON_FULL_DIM` 113 → 116): `protect_odds` (now EVERY
  mon owns its stall state — a benched mon truthfully reads 1.0 since the counter resets on
  switch; was 2 active-only scalars), `trapped`/`maybe_trapped` (on OUR ACTIVE's slot — the
  trapped ENTITY; appended by state_encoder before the active flag, which stays LAST in the
  slot because `hp_and_active[:, :, -1]` is a load-bearing convention). The C4 Protect edge
  reads protect odds from the mon slot now (was `non_matchup_rest[GLOBAL_ENV_DIM+7]`).
* **KEPT as the lean 17-dim board block**: `fainted ×2`, `turns_since_progress`, `wish ×2`
  (E7/E6-class side/board facts) + the 12-dim request-order `active_req_moves`.
* **Model side**: `move_feature_blocks` loses the matchup + matchup-validity inputs (the
  per-cell validity block is deleted with them); `global_context` loses `forced_struggle`;
  `non_matchup_rest` narrows 29 → 23 (`pi_projection` 1137 → 1131, `vf_projection` 881 → 875);
  `ExtractorContext` drops `matchups_all`/`struggle_feature`.
* **CPU refund**: the reactive matchup loop was ~35% of the obs build
  (`effective_multiplier_by_types` was the #1 tottime line at 202 calls/encode) — deleted
  outright, along with reactive.py's `_expected_multiplier` family.
* Weight shapes AND obs meaning change together ⇒ `ARCH_SIGNATURE` `gen3_edge_bias_trunk_v1`
  → `gen3_entity_rehome_v1`, `MODEL_CONFIG_VERSION` 60 (stamp — no migration is possible).
  Fresh-lineage break: gen-4 is the first run of this world. Golden obs fixture regenerated
  (2667 dims, 991 decisions); obs-roundtrip fuzz bit-for-bit (627 decisions); trapping fuzz
  re-pointed to the per-mon bits (+ a bench-slots-stay-zero assertion); prober degrades
  gracefully (`om_off`/`tm_off` 0 = absent, ThreatView/saliency no-op).

### v60 addendum (2026-08-08, same day, still pre-any-v60-checkpoint): E2 active-context injection

The one §6 row the re-home commit left un-re-homed — "active context boosts + volatiles (116) →
E2 of the two actives" — lands: each side's 58-dim ctx block is scattered onto its ACTIVE mon's
role-encoder row (bench rows zero), so the entity owns its own boosts/volatiles. Additive (the
global-token and both projection routes remain); model-side only (obs unchanged, fixture/fuzz
untouched); role_input_dim +58. No new version stamp — no v60 checkpoint exists yet, so the v60
signature still carries the break. Pinned by `e2_ctx_injection_test.py` (active rows carry
exactly their side's ctx, bench rows read zero).

### v60 addendum 2 (2026-08-08, still pre-any-v60-checkpoint): the unrevealed-defender GIGO fix

`gen3_unrevealed_outgoing_prior_v1` (design_conditional_opponent_cells.md §4.1, item 4 —
owner-moved into gen-4): OUTGOING damage priced against UNREVEALED opponent slots read ZERO
(v34's revealed gate) — misleading exactly when switching matters most, the typeless-HP-immune
bug class. Now the shared `_outgoing_matrix` kernel (behind the D1 `pairwise_outgoing` cells,
the OMX flat block, and every C1 hypothetical re-run) prices hidden slots against the v36
EXPECTED-LATENT defender at a **Species-Clause-filtered gen3ou usage prior** (new
`SPECIES_USAGE_PRIOR` table + `gen3_data.priors.species_usage()`; a learned posterior can
override via the `species_probs` seam): E[mult] via `SPECIES_EXP_MULT`, E[def/spd] via the
spread-prior means, E[maxhp] via E[base HP], forced-alive full-HP switch-in. P(KO) stays
NULLED at hidden slots (a full-HP switch-in is ~never OHKO'd) and the `revealed` cell channel
stays 0 — magnitudes change, epistemics don't. Revealed columns byte-identical (everything
enters via `torch.where(believed, …)`). Unconditional in the v60 world (no flag — no v60
checkpoint exists). B=1 CPU forward +~0.3 ms. Gates: `unrevealed_outgoing_prior_test.py`
(prior marginal == direct recompute; Species-Clause zero-mass; revealed byte-identity;
forced-alive; the §4.1 non-zero guard), `damage_op_probe_fuzz_test` 22/22 (revealed physics
untouched), the fullgraph compile gate.

### v61 — `gen3_no_concat_v1` (2026-08-09): the op head-concat is dead; the multi-seed critic window

The owner's "no more concat" milestone (the gen-5 world), executed on the gen-4 stratified
evidence (53ef270): net policy dependence on the flat block +0.00%, all-edges-off ABOVE the
concat arm on flips (first time in the lineage), and the critic's magnitude content decodable
with the concat zeroed (act_threat vf r² 0.418) — the |dV| 4.75 was trained reliance on an open
window, not structural necessity.

* **Step 3 (the death):** `ProjectionAssembler` no longer appends the 660-dim block to either
  head — `pi_projection` 1131 → **471**, `vf_projection` 875 → **471**. The op lives on: pointer
  cells (policy, lossless per-action), prefuse token injection, the 15 edge families' cells, and
  `last_raw_block` for the probes/prober. The `edge_ablation_audit` concat arm's SEMANTICS
  changed with it: it now measures the seed-window route (vf) — its pi effect is structurally 0
  (pinned in its test).
* **Step 4 (the replacement window):** `MultiSeedValueReadout` — k=4 × 64 learned seed queries
  cross-attend over the op's per-our-mon incoming rows (`our_mon`), vf-only (+256). Ships WITH
  the `seeds/*` TB collapse contract (`seed_diagnostics.py`, logged every `train()` by
  instrumented_ppo) and the pre-registered VICReg trigger — readout MULTIPLICITY (P3 refuted
  width, never multiplicity).
* **Step 5 (the reduction site):** the arity-2→1 collapse is now the NAMED knob
  `DamageOperator._chan_max(..., how="hard_max")` — OA1 (`conditional(λ)`), the belief mean and
  PV-as-reduction become settings of this one call site, per design_op_tensors.md §3.2.
* **Deferred honestly:** OpTensors steps 1–2 (typed views; the E4/d3/s3 recompute dedup) —
  inverted order under the owner's gen-5 priority; the §9.1 evidence shows the removal was not
  waiting on them. They proceed as background work during gen-5.
* Gates: physics oracle 22/22 (revealed physics untouched); fullgraph compile green on the v61
  production config; full suite green; obs UNCHANGED (2667 — no fixture churn).
  `ARCH_SIGNATURE` `gen3_entity_rehome_v1` → `gen3_no_concat_v1`, `MODEL_CONFIG_VERSION` 61.
  Fresh lineage: gen-5 (`ai_v9_06_gen5_no_concat_0809`) is the first run of this world.

### v62 — `gen3_seed_vicreg_v1` (2026-08-10): the VICReg floor on the value-seed readout (built because the pre-registered trigger FIRED)

The v61 `value_seeds/*` collapse contract did its job on its first run: gen-5
(`ai_v9_06_gen5_no_concat_0809`) showed `out_cos` = 1.000 and `out_effective_rank` = 1.0 at every
measurement from 196k through 15M+ steps (`out_var` ≈ 5e-6; `query_cos` only 0.33 — distinct
queries, indistinguishable attention patterns), i.e. the k=4 `MultiSeedValueReadout` seeds pay for
four reads and deliver one. That is exactly the pre-registered VICReg trigger (eff-rank < k/2
sustained past ~2M), so this version ships the wiring it called for:

- **`agents/model/seed_vicreg.py`** — `seed_vicreg_loss(outputs [B,k,D])`: a variance hinge
  across the seed axis (`relu(γ − std_k)`, γ=1.0) + a **cross-seed covariance penalty on
  batch-centered outputs** (kills the "identical + constant offsets" cheat the variance term
  alone admits — the z_arch covariance-never-wired lesson, this time with a dedicated gate test
  `test_constant_offset_cheat_is_caught_by_covariance_term`). Terms logged as
  `value_seeds/vicreg_{var_term,cov_term}` + `value_seeds/vicreg_loss` beside the collapse
  contract.
- **`--value-seed-vicreg-coef`** (default 0.0 = OFF, byte-identical): folds `coef · loss` into the
  PPO loss per minibatch on the live `last_outputs` stash. Resume-IMMUTABLE (the vf_coef class):
  recorded on `ModelVersion` (config v62, migrate default 0.0), enforced on the training-resume
  path only (`check_value_seed_vicreg` via `enforce_value_seed_vicreg_coef`); frozen-opponent
  loads exempt. Enabled with no seed readout in the config → startup RuntimeError
  (`assert_seed_vicreg_wirable`), never a silent no-op.
- **TB prefix rename `seeds/*` → `value_seeds/*`** (owner request, same pass): "seed" alone is
  ambiguous with RNG seeds — these are the critic's value-readout seed queries. Applies to the
  v61 diagnostics family too (`value_seeds/{query_cos,out_cos,out_effective_rank,out_var}`).
  Gen-5's TB retains the old `seeds/*` tags (historical); gen-6 starts on the new prefix.
- No forward/weight-shape change at coef 0 → no `ARCH_SIGNATURE` bump (stays
  `gen3_no_concat_v1`). Intended ON at the gen-6 launch; accept if `out_effective_rank` rises
  toward k while `eval/elo` tracks gen-5's curve.

Landed in the same pass, OUTSIDE the version (`gen3_pair_reduce_v1` scaffolding — byte-identical,
no config field, no flag): `agents/model/pair_reduce.py`, the `design_pair_reduction.md` §8.1
steps 3–4 rungs. `DamageOperator(reduce_how=…)` (constructor-only; default `"hard_max"` builds
NOTHING — no params, no state_dict keys, no forward work) can build the Contract-W/L reducers
beside the legacy per-channel hard max: R1 `belief_mean` (α = w/Σw), R2W `learned` (zero-init
g ⇒ α(init) = normalize(w)), R2L `deepsets_{sum,max}` (φ carries second-order channels ⇒ E[o²] ⇒
variance ⇒ hedging expressible; ρ zero-init), R3 `multi` (the §5 PNA bundle). A non-default rung
only STASHES `last_reduced_extra` [B,6,extra_dim] — nothing consumes it yet; delivery (switch
cell / prefuse / seed rows) + the config/versioning it entails is the gen-6 boundary's work.
Gates in `pair_reduce_test.py`: forward-level G0 through the real op (flat block bit-for-bit
equal default-vs-multi, only `pair_reducer.*` keys added), G2 status-coherence (the per-channel
max provably describes a move-profile NO single opponent move has; one α fixes it), G4
candidate-invariance + defender-equivariance, α-init identity, second-moment recovery.

### v62 addendum (2026-08-10, same day, after gen-6's first 2M steps): the VICReg targets are SCALE-RELATIVE

The v62 entry above shipped ABSOLUTE targets — a per-dim std hinge at γ=1.0 and a raw
squared-covariance penalty. **Gen-6's first launch measured them as a failure and was killed at
2.06M steps**: `value_seeds/out_effective_rank` stayed pinned at exactly 1.000 across all 21
logged points while `vicreg_var_term` sat saturated at ~0.997 and `out_cos` at 0.9997. Diagnosis,
measured on the live gen-5 checkpoint (`tmp` probe over 256 stratified states):

| quantity | value |
|---|---|
| seed-output RMS | 0.207 |
| cross-seed std (what the term pushes) | 0.0015 |
| kv-row RMS | 0.246 |
| spread across the 6 mon rows | 0.141 |
| the shipped γ | **1.0** |

γ=1.0 is ~7× the entire signal's RMS, and since each seed output is a **convex combination of the
same six kv rows**, the achievable cross-seed std is bounded by the row spread (~0.14/dim). The
target was unreachable except by `kv_proj` inflating its own scale ~10× against every downstream
norm. A saturated ReLU hinge still has gradient (slope −1), so the term applied constant pressure
— the spread did move ~5× in 2M steps — but multiplicity, the quantity the whole feature exists
for, never budged. **An absolute target on a layer whose scale is learned is a bug, not a tuning
choice**, and the `query_cos` reading (0.128 — near-orthogonal queries) rules out the queries as
the cause: it is the attention distributions over six rows that are degenerate.

The fix (same version — no run has trained on the old form; no field, forward or weight-shape
change, so no `MODEL_CONFIG_VERSION`/`ARCH_SIGNATURE` bump):
- **Variance** → per-dim cross-seed std ÷ that dim's own RMS (batch+seed, **detached** so the
  gradient can only widen the seeds), hinged at `SEED_VICREG_GAMMA_REL` = **0.25** — "the seeds
  differ by ≥25% of the feature's own scale", inside the ~0.68 ceiling the row spread implies.
- **Covariance** → cross-seed **correlation** (not raw covariance) of the batch-centered outputs,
  so the penalty lives in [0,1] and is comparable across runs and scales.
- **Two new metrics**: `value_seeds/out_std_rel` (the achieved fraction, target 0.25) and
  `value_seeds/out_rms` (the magnitude **watchdog** — a relative target's one degenerate response
  is to shrink the feature rather than differentiate it, and this makes that visible; read the
  two together).
- Gates: `test_targets_are_scale_free` (100× rescale ⇒ identical loss — the direct regression for
  this bug), `test_realistic_collapsed_scale_still_pays_full_hinge` (the measured operating point
  above is now a closable gap rather than a saturated one), and the unchanged cov gate.

### v63 — `gen3_seed_quantile_v1` (2026-08-11): give each value seed a DIFFERENT JOB

**Why, from gen-6's measurement rather than from theory.** The v62 VICReg floor works numerically
(`out_std_rel` 0.002 → 0.53, correlation 1.00 → 0.19, `out_cos` → 0.87) yet `out_effective_rank`
sat at **1.05**. Decomposing the 4.8M checkpoint said exactly why:

| measurement | value | meaning |
|---|---|---|
| uncentered PR | 1.037 | ~1 distinct readout |
| **centered PR** | **0.846** | the deviations occupy **<1** direction |
| deviation / shared | 0.598 | the differences ARE large |

and the per-seed attention over our six mons was decisive — seeds 0/1/2 agreed to three decimals
(≈uniform) while seed 3 alone diverged. **One seed broke away and three stayed identical**, which
satisfies a variance hinge on average while leaving multiplicity untouched. The durable lesson:
**a repulsion penalty buys SPREAD, not MULTIPLICITY** — it has no vocabulary for "differ along
DIFFERENT axes"; only the covariance term does, and at equal weight it is too weak.

The positive fix: seed k predicts **quantile τ_k** (0.1/0.35/0.65/0.9) of the return, pinball
regressed, through **ONE SHARED `Linear(dim,1)`**. Four different τ ⇒ four different predictions
⇒ four different seed READS. Collapse stops being unpenalized and becomes loss-INCREASING —
**measured at 45.9%** higher loss on realistic normalized returns. The decomposition is also
semantic in a way decorrelation cannot be (the τ=0.1 seat must find what makes the downside).

- ⚠️ **The shared readout is load-bearing.** Per-seed projections would let the HEAD manufacture
  the spread from four identical inputs — success reported while the seeds stay collapsed, the
  z_arch silent-failure shape. Gate: `test_shared_readout_makes_collapse_strictly_worse`.
- ⚠️ **PURE pinball, NOT QR-DQN's Huber variant** — a deliberate reversal of the obvious default.
  Huber caps the gradient inside |u|<κ and pulls every estimate toward the median: MEASURED on
  N(0,2) at κ=1 it fitted ±2.18/±0.67 against true ±2.50/±0.78. QR-DQN wants that for bootstrapped
  TD targets; we regress an already-normalized MC return, and the shrink attacks the one thing this
  module creates — SEPARATION. Gate: `test_pinball_recovers_the_true_quantiles`.
- `--seed-quantile-coef` (0.0 = OFF, no module, byte-identical). The HEAD is STRUCTURAL +
  version-checked (`MODEL_CONFIG_VERSION` 63, migrate default False, bool compare in
  `check_compatible`, threaded through `arch_toggles_from_model` so a quantile-on run does not
  FATAL on its own sentinels); the COEF is training-only. No `ARCH_SIGNATURE` bump (off is baseline).
- New metrics: `value_seeds/quantile_{loss,spread,crossing_rate,pred_i}`. **The ordering invariant
  is the cheap read**: ascending τ ⇒ ascending predictions, so a collapsed readout shows spread → 0
  and crossing_rate → 1.
- **Honest scope:** ledger K1 killed the DISTRIBUTIONAL CRITIC as a win-rate lever (sub-Gaussian
  residuals). This makes a different claim — a k-dimensional target prevents a k-way readout from
  collapsing — which K1 does not refute. But un-collapsing is a MEANS: gen-5 matched gen-4 while
  fully collapsed, so multiplicity may still be worth ~0. Judge on `out_effective_rank` first,
  anchored ELO second.

### v64 — `gen3_value_threat_inject_v1` (2026-08-11): the critic reads a per-entity MAGNITUDE

**The seed line ends here, on a measurement.** v62 (repulsion) and v63 (per-seed quantiles) attacked
`MultiSeedValueReadout`'s collapse from opposite directions and landed in the same place. Gen-7's
quantile arm succeeded on its own terms — `quantile_crossing_rate` 0.456 → **0.000**,
`quantile_spread` 0.007 → **1.016** at 10.6M steps, so the four seeds really do predict four
ordered quantiles — yet `value_seeds/out_effective_rank` reached only **1.157** against a ceiling of
k=4, matching gen-6's centered PR **0.846**. The shared structural cause: a SHARED readout
constrains only each seed output's component along its own weight vector, leaving every orthogonal
direction unconstrained. **Seed multiplicity was not the axis the critic was missing**, and no
coefficient on either term changes that. Two independent nulls on the same object is a kill, not a
tuning problem.

- **The build.** `--value-threat-inject` adds, for each of OUR mons `j`, the op's α-weighted
  incoming row `Σ_k α_k · pair_in[k, j, :]` to that mon's post-transformer token via ONE shared
  zero-init `Linear(13, D_MODEL)`, on **the value pool's copy only**; `value_cls` then pools the
  augmented set. `agents/model/value_threat_inject.py`; the augmentation lives inside `CLSPool` so
  the augmented tensor is a LOCAL.
- **Why token content and not another readout seat.** An attention BIAS (the `d3` edge family) is
  softmax-normalised — it can rank defenders by threat and structurally cannot say "62% of max HP".
  An attention VALUE carries magnitude. This is the third delivery route named in
  `design_opponent_intent.md` §7a.2, and the first one tried on the critic since the concat died.
- **vf-only is STRUCTURAL, not a convention.** `our_cls`, `our_active_refined` and the pointer head
  read the untouched tensor, so `pi` is bit-identical for an ARBITRARY `W_inj`. Gated by
  randomising the projection to N(0, 5²) and asserting `torch.equal` on pi while vf must MOVE (the
  second half stops the gate passing vacuously on an inert route).
- **Equivariant in both axes.** α is shared across defenders by Contract W (no `J` index exists on
  it by signature) ⇒ invariant under permuting their moves; the row rides mon `j`'s own token ⇒
  equivariant under permuting ours; attention pooling is permutation-invariant. Gated by permuting
  our six mons and asserting `value_pooled` is unchanged — plus the counterpart gate that
  MIS-pairing rows to mons DOES change it, so invariance cannot pass by carrying no information.
- **It forces the op's reducer on.** R0 `hard_max` (production) builds no `PairReducer` and stashes
  nothing, so the flag switches `reduce_how` to the R1 `belief_mean` rung. Derived, never a second
  user knob: this arm tests DELIVERY, and a variable rung would confound that with the DISTRIBUTION
  question.
- **Two traps handled.** (1) `CLSPool` is constructed ~250 lines BEFORE `DamageOperator`, so the
  projection width comes from a new pure helper `pair_reduce_extra_dim(how, n_channels)` (which
  `PairReducer.__init__` now also calls, so they cannot drift) rather than from `self.damage_op` —
  reordering module construction to suit the feature would have shifted optimizer parameter
  POSITIONS and corrupted every resume. A post-construction assert ties the two widths together.
  (2) `W_inj` is in the `restore_identity_init()` capture set (ledger M1), gated on a REAL
  `MaskablePPO` build rather than a bare extractor — the only place SB3's ortho clobber is visible.
- **Honest scope.** v1 substitutes **α := normalize(w)**, a PRESENCE belief, where the design wants
  a supervised USAGE belief. Deliberate: it separates the DELIVERY claim (does a per-entity absolute
  in the value pool help?) from the DISTRIBUTION claim (does a learned α beat w?), so a null indicts
  the route and not the belief. G1-FINAL's null does not predict this one — it tested aggregation on
  the POLICY's cells and never changed critic delivery.
- Structural + version-checked (fresh runs only), `MODEL_CONFIG_VERSION` 63 → 64; OFF builds no
  module, leaves the op on `hard_max`, and is byte-identical. No `ARCH_SIGNATURE` bump.

### v67 — `gen3_opp_intent_v1` (2026-08-12): the model can finally say what it expects them to do

`design_opponent_intent.md` steps 2–5. Two supervised pointer heads over what the opponent chose:
**`α`** — a distribution over their K believed threat-move seats **plus SWITCH** — and **`β`** —
given a switch, which of their mons comes in.

- **The measured case, not a hunch** (`tmp/g2b_alpha_baseline.py`, gen-8 @26M, n=1676 attack
  decisions): the belief's top-K CONTAINS the move they clicked **85.8%** of the time but ranks it
  first only **51.8%** — 24.6% of the time the true move sits at rank 1, one slot away. That
  **34.0 pp** of in-the-seats-but-mis-ranked mass is exactly what a learned re-weighting can move,
  and it needs no new information. The hidden-team belief was greenlit on ~8–10 pp.
- **Both heads are POINTERS.** Seat k's logit comes from seat k's own refined E4 token through a
  SHARED scorer; bench slot j's from slot j's token. So permuting their moves permutes `α` and
  permuting their bench permutes `β`, exactly — gated both ways. A flat `Linear(ctx, K)` would have
  passed every shape test and learned "seat 0 is usually right" from the belief's own sort order,
  memorising the ordering `α` exists to correct.
- **Matching is by CANONICAL ID.** Seats are `w.topk(K)` and permute every turn, and they are built
  by the MODEL mid-forward, so the env cannot name them. It emits the move NUM; the loss locates it
  among the seats. A belief miss is MASKED, and `alpha_mask_rate` is logged as a first-class
  diagnostic — it is the belief's coverage failure, and conflating it with `α` being wrong would
  hide which component to fix.
- **DECISION vs CONSEQUENCE** (`opp_intent_labels.py`). A phaze (our Roar moved them) and a
  post-faint replacement both LOOK like switches in the `TurnDelta` and are not choices. Labelling
  either as a voluntary switch would teach `β` to predict our own move and would pad `α`'s
  denominator with rows where no decision happened. Both are masked; five cases, five gates.
- **The one-row alignment, and why it gets its own gate.** The env can only observe their turn-t
  action while building the obs for t+1, so the label sits one row AHEAD of the prediction.
  `align_labels_to_predictions` shifts it back **before `get()` shuffles** — the only point where
  the `[n_steps, n_envs]` adjacency still exists — and DROPS any pair whose successor starts a new
  episode. Without that drop, one battle's first decision is spliced onto the previous battle's
  last board: a silent cross-episode GIGO bug that no shape check and no loss curve would reveal.
- **Supervision only.** Both heads read a DETACHED input, so a null result says the head cannot
  predict the opponent — not that predicting the opponent perturbed the policy. Letting `α` shape
  the trunk is a later, separate experiment.
- **The interpretability deliverable ships with it** (`render_alpha` → the trace's `opp_intent`
  block): `α` as a ranked list of NAMED moves with probabilities. A turn where the model played
  around a Fire Blast and one where it never saw it coming are currently indistinguishable in every
  view we have. This is justified on that alone — gen-8 established that accurate beliefs do not
  automatically convert to Elo, so betting the head purely on strength would be the wrong bet.
- Structural + version-checked, `MODEL_CONFIG_VERSION` 66 → 67; OFF builds neither head and is
  byte-identical. Requires `--entity-topk-seats>0` (fail-loud: a pointer needs something to point
  at). No `ARCH_SIGNATURE` bump. `--opp-intent-coef` itself is training-only.
- Drive-by fix: `--help` crashed with `ValueError: unsupported format character 't'` — the
  `--team-pfsp` help had an unescaped `%` in "sub-50% team". argparse `%`-formats help strings, so
  the whole CLI's `--help` was dead for anyone who ran it without `2>/dev/null`.

### v72 — `gen3_t0_species_prior_v1`: the species belief the physics could never read

`--t0-species-prior`. The model already formed a conditional belief about which species sits in a
hidden opponent slot, and the `DamageOperator` already accepted an override for exactly that. They
were never connected, and the reason was the tier ordering rather than any missing machinery.

- **The defect.** `BeliefHead.species_prior_logits` (v69) is a naive-Bayes read over the revealed
  opponent team — the difference between "the average gen3ou mon" and "the fifth slot on a team that
  has already shown Tyranitar + Skarmory + Cloyster". But `BeliefHead` is **T2 DECIDE**
  (post-transformer, a training-only side readout) while the `DamageOperator` is **T1 REASON**
  (pre-transformer). A T1 consumer cannot read a T2 output, so the op fell back to
  `SPECIES_USAGE_PRIOR`, a static frequency table masked only by Species Clause. Every damage
  number against a hidden slot was computed against a population average while the model's own
  belief sat one tier too late to be used.
- **Two things already existed and had never met.** `unrevealed_species_probs`'s `species_probs`
  argument has been in the tree since `gen3_unrevealed_outgoing_prior_v1`, documented verbatim as
  "the future-learned-belief seam", with **zero** callers passing it. And `species_prior_logits`
  reads only `opp_species_ids` and `opp_believed_mask` — **no tokens at all** — so it was already
  T0-legal and nothing but its host module's tier kept it out of the physics.
- **The change is a re-homing, not new modelling.** The naive-Bayes math moves to
  `t0_species.species_team_prior_logits`, which `BeliefHead` now also calls, so there is ONE
  implementation rather than two that could drift. `T0SpeciesPrior` is parameter-free (two
  non-persistent buffers): the state_dict is identical, and no optimizer parameter position moves.
- **One belief, every site.** The seam existed only on `pairwise_outgoing`; the op's own `forward`
  and `pairwise_boost` both priced unrevealed defenders with no way to be told. Both now take the
  argument, and the extractor resolves the belief ONCE in `forward_internal` and hands the same
  tensor to all three. The gate asserts tensor IDENTITY, not equality — `pairwise_outgoing`'s
  docstring already promises the bias and the head concat can never disagree on a value, and two
  equal-but-separately-computed tensors is how that promise quietly stops holding.
- **Shape is 2-D on purpose.** `[B, n_species]`, not `[B, 6, n_species]`: the belief is a property
  of the opponent's TEAM, not of which hidden slot you ask about, so every slot's row would be
  identical anyway. It also lands exactly on the static branch's existing broadcast-the-RESULTS
  contract — the `[B,6,S]` expand is what mis-vectorized under Inductor CPU and took down gen-4's
  launch prewarm. A learned per-slot delta re-enters that shape and must not be added without a
  compile gate.
- **No learned delta at T0 (deliberate, v1).** A T0 read sees PRE-attention tokens, so a delta here
  is strictly weaker than the one `BeliefHead` gets at T2 — while the PRIOR half loses nothing at
  T0, because it never touched tokens. The informative half moves down a tier for free; the weak
  half is not built.
- Structural + version-checked, `MODEL_CONFIG_VERSION` 71 → 72; OFF routes `None` to every site and
  is byte-identical. No `ARCH_SIGNATURE` bump (OFF reproduces the previous forward exactly). The
  resume check is the only thing that can reject a mid-run flip — the state_dict is identical
  either way, so no shape check would ever fire.
- Independent of `species_prior_fusion`: that fuses the prior into the T2 aux readout, this feeds
  the T1 physics. Either is useful without the other, and they share the one implementation.

### `gen3_belief_label_only_v1` (2026-08-14): the belief the policy may read but not corrupt

A third `--belief-grad-mode`, **`label_only`** — and the point of the entry is that it cuts the
**opposite arrow** from `detached`, which the flag's existing name had made easy to misread.

There are four gradient routes between a state-prediction belief head and the rest of the network:

| | route | `shaping` | `detached` | `label_only` |
|---|---|---|---|---|
| A | label loss → belief head params | on | on | on |
| B | label loss → shared trunk (via the head's READ) | on | **CUT** | on |
| C | PPO loss → belief head params (via the WRITE) | on | on | **CUT** |
| D | PPO loss → shared trunk (normal training) | on | on | on |

`detached` (v41) cuts **B**: the heads read a stop-grad trunk, so the belief cannot drag the trunk
toward predicting hidden state at the policy's expense. It does **not** stop PPO from training the
heads — measured on a real extractor at the time of this change, a pure `pi+vf` loss under
`detached` deposited gradient mass 2113.7 on `move_belief.move_head`, 96.5 on
`spread_belief.stat_head` and 52.2 on `hp_type_belief_head.type_head`. `label_only` cuts **C**: the
belief is trained by its supervised labels alone.

**Why that is worth having.** The only fixed point of a supervised loss is the conditional
expectation, so a label-only head's output is a calibrated `P(hidden state | features)`. With PPO
gradient in it, the head is really an extra hidden layer of the policy that happens to carry a
label, free to drift off-calibration toward whatever value makes the preferred action look good.
Two consumers here depend on it being the former: the `DamageOperator` reads these posteriors **as
probabilities** (top-K weights, expected-latent marginalization), and the prober presents them to a
human as "what the model believes". It also closes the belief → op → policy → belief loop, which is
the standard way value-aware model learning produces self-fulfilling beliefs.

**And what it costs.** Pure MLE spends a fixed-capacity head uniformly over the label distribution;
the PPO gradient was the only signal saying *be precise about the 4× threat, not the filler move*.
`HPTypeBelief`'s own docstring records that the op's damage gradient is what sharpens the type head
— `label_only` turns that off deliberately. This is an experiment arm, not a strict improvement.

**The read stays live.** `label_only` cuts C but not B, so the label loss still teaches the trunk to
encode hidden state. Cutting both would leave a probe on a trunk with no incentive to carry the
information, still feeding the policy — a strictly worse estimator with a live consumer. That fourth
combination is deliberately not offered, which is also why this is a third value on the existing
flag rather than a second flag whose 2×2 would include it.

**Scope — the four heads with a forward path.** `MoveBelief`, `SpreadBelief`, `HPTypeBelief`, and
`AlphaIntentHead`. Alpha is the one that would have been missed: it is a pure readout *until*
`--intent-value-reduce`, which appends an alpha-weighted threat term to the critic half. It is
published unconditionally so that turning that flag on later cannot silently reopen the route. The
remaining supervised heads — `BeliefHead` (species/moves/latent), `WinProbHead`, `PubValHead`,
`BetaSwitchHead`, `SeedQuantileHead`, the ZArch recon head — are side readouts whose output never
re-enters the forward, so no policy gradient can reach them in any mode; that is now asserted rather
than assumed.

**The cut is at the PUBLISH BOUNDARY, not at the consumers.** `last_move_belief_logits` alone has
eleven forward readers (the op, `d3`/`d4`/`c1b`/`c2`/`c3`/`x`/`s3`, the E4/E5 seats, the reinject),
so a per-consumer rule is one forgotten site away from silently reopening the route. Instead each
head's `last_*` stash IS the stop-grad publication (`_publish_belief`), and the supervised losses
read the live tensor through a new `belief_supervision(name)` accessor. A consumer added tomorrow is
isolated by construction. The inverse hazard — a future loss reading the published stash and
training nothing, silently, since the loss value looks normal — is what the accessor's unknown-key
`KeyError` and the gate test below exist for.

**Detach the LOGITS, never the matmul output.** In `reinject_moves`,
`soft_emb = sigmoid(logits) @ move_embedding.weight`. Detaching `logits` leaves
`move_embedding.weight` its full gradient; detaching `soft_emb` would kill it — and since that table
also trains from `PokemonEncoder`, the loss would be silent rather than a visibly dead parameter.
Same trap in `HPTypeBelief.reinject` with `type_embedding` via `hp_soft_type`. The reinjection
adapters (`MoveBelief.reinject`, `SpreadBelief.reinject`, `HPTypeBelief.reinject_proj`) have no
supervised loss at all, so PPO is their only gradient source and they must keep training.

**Gate:** `belief_label_only_gate_test.py`, verified failing on revert. It builds a REAL
`MaskablePPO` (ledger M1 — SB3's ortho-init makes a bare-extractor assertion an assertion about a
construction path production never uses) and backprops from `evaluate_actions`, i.e. from the ACTION
LOGITS as well as the value — under `gen3_pointer_native_v1` there is no flat `action_net`, so a
test that summed `pi_features` would miss the entire pointer route and pass while the real PPO loss
still trained the heads. It asserts the supervised direction too. `SpreadBelief` needed a dedicated
direct test: with a random obs the species is unknown and its prior std is 0, so
`believed = mean + delta·std` carries no gradient in ANY mode, and `sum(LayerNorm(x))` is a constant
— two independent ways for the whole-policy assertion to look like it passed while testing nothing.

**Versioning.** No new field and no `ARCH_SIGNATURE` bump: `detach()` is value-preserving, so the
forward is bit-identical in all three modes and a frozen eval/pool/distill opponent plays
identically. It rides the existing resume-immutable `belief_grad_mode` check
(`--allow-belief-grad-mode-change` for an intentional migration on a converged checkpoint), and the
legal-value set now lives in one place, `features_extractor.BELIEF_GRAD_MODES`, which the CLI
`choices=` and the migration-notice table are pinned to agree with.

---

### v75 — the SimSiam LATENT belief is deleted

`opp_belief_latent`, `--opp-belief-latent-coef`, the `BeliefHead` latent predictor, the stop-grad
`last_belief_target_latent` stash, the `belief_target_slots` training-only obs key, and the
`Gen3Env` work that built that key every decision are all **removed**.

- **It was never fed forward.** The latent logits were stashed for the aux loss and nothing else —
  no path into `pi` or `vf`, no pointer logit, no head concat. That is what separates it from
  `--opp-belief-cls-k`, whose pooled hidden-opponent belief is appended to BOTH projections and so
  changes what the policy can compute at inference time. Deleting a side readout removes a training
  signal; it removes no capability.
- **It cost ~13% of the train step.** Per-flag marginal cost, measured on an idle box with the arms
  interleaved so drift cancels (each leg includes `--opp-belief-aux-coef 0.05`):

  | leg | train ms | marginal | rollout ms |
  |---|---|---|---|
  | baseline (belief stack off) | 2662.3 | — | 1673 |
  | `opp_belief_slots` | 2915.2 | (base) | 1693 |
  | `opp_belief_cls_k=6` | 3264.0 | **+348.7** | 1778 |
  | **`opp_belief_latent`** | **3256.3** | **+341.1** | 1697 |
  | `move_belief_mode=both` | 3033.5 | +118.3 | 1686 |
  | `spread_belief` | 2987.5 | +72.2 | 1695 |

  The train step is ~89% of production wall at 10 epochs (the root docs' "rollout is ~86% of wall"
  describes a different configuration and does not hold here — measured train share was 61% at 5
  epochs on this box). So this was real throughput, spent on a head the policy could not read.
- **Its own probe had already said so.** The belief latent/BYOL role-geometry probe found species
  geometry decodes strongly and the move-id table not at all, and concluded decodable ≠ helps.
- **What is NOT lost.** `BeliefSlots` still swaps learned unknown-mon tokens into the hidden
  opponent slots; the species CE and the moves BCE still supervise them (order-invariantly, via the
  same Hungarian assignment); `t0_species_prior` still hands the team-composition species belief to
  the T1 physics. The model's ability to predict the opponent's unrevealed mons is unchanged — what
  is gone is the second, graded way of expressing it.
- `_belief_aux_loss` now returns `(aux, metrics)` instead of a 3-tuple. `_LATENT_STD_TARGET` /
  `_LATENT_VICREG_WEIGHT` SURVIVE — they are shared with `_move_belief_latent_loss`, a different
  live feature (`--move-belief-latent-coef`), which is also why `move_latent` and every
  `movelatent_*` metric are untouched.
- `MODEL_CONFIG_VERSION` 74 → 75. The migration **REFUSES** a config recording
  `opp_belief_latent=True` rather than popping it: unlike the v71 forward-only flags, this one
  carried PARAMETERS, so such a checkpoint's state_dict holds keys the live extractor has no home
  for. `sanitize_dead_extractor_kwargs` applies the same rule to a saved zip's
  `features_extractor_kwargs`. No `ARCH_SIGNATURE` bump — a config that had it OFF is byte-identical.
### `gen3_op_tensors_views_v1` (2026-08-14, no version bump — byte-identical): the op's flat layout gets ONE slicer

`design_op_tensors.md` steps 1–2. `DamageOperator.tensors_from_block()` becomes the single walk
of the 660-dim flat block's layout, returning **`OpTensors`** — named zero-copy views
(`incoming_rows`, the CB tail, the outgoing/status groups, the opaque matrix renders) — and
raising if the walk doesn't tile `out_dim` (a region added without a view is unrepresentable).
Consumers re-pointed: `pointer_cells` assembles from the views; the prefuse injection and the
assembler's critic seed window read `damage_op.last_tensors.incoming_rows` (the assembler's
parameter is now the typed `seed_rows`, not the flat block). The flat block stays the
serialization for `decode_damage_block` / `last_raw_block` / the prober; dropping it from the
forward is step 3 (retrain-class — it shrinks `out_gain`). Also fixed en route: the delivery
graph still drew the **dead 660-dim op→both-heads concat edges** (stale since v61, pinned by its
own test) — replaced with the true routes (seed readout, intent reduce, hidden-opp pool), and the
two ablation-audit tools' assembler hooks re-anchored to the typed rows with a fired-assertion so
a future signature drift fails loud instead of silently measuring nothing.

Gate: unchanged state_dict keys + bitwise-identical pi/vf/raw-block/pointer-cells/α/β on 64 real
gen-9 eval states across three config arms (gen-9 prod, gen-9+intent_value_reduce, minimal), the
constructed-scenario physics oracle 22/22, the full model suite, and the real-compile test.

### v76 — `gen3_ctx_dedup_v1` (2026-08-14): the duplicated active-ctx head concat is deleted; the migration floor

Two changes, one landing:

- **The per-side ENCODED active contexts leave both heads.** `ProjectionAssembler` no longer
  builds `active_ctx_encoder` (`ACTIVE_CTX_HIDDEN` deleted from `arch_constants`, the
  `active_ctx_hidden` ModelVersion field retired); pi loses `our_ctx_enc + opp_ctx_enc` (64) and
  vf the same — both projection inputs narrow by 64. Rationale: duplicated delivery with a 1:1
  entity-native replacement already live — the E2 injection scatters each side's FULL raw 58-dim
  boosts+volatiles block onto its ACTIVE mon's role token, and the global token carries both raw
  blocks as a second route, so every scalar the encoder saw still reaches both heads through the
  trunk, entity-attached. **`non_matchup_rest` deliberately stays** (its only token route is the
  global token, which no pool reads directly — no 1:1 replacement exists) and so does the
  hidden-opp pool concat. state_dict keys + widths change → `ARCH_SIGNATURE` carries the break;
  fresh lineage.
- **`_migrate_config` grows a FLOOR** (`MIGRATION_FLOOR`, paired to the signature via
  `SIGNATURE_FIRST_VERSION` and pinned by `migration_floor_test.py`): configs older than the
  current signature's first stamped version are REFUSED with a diagnosis (naming the floor, the
  signature, and the metadata.json git_hash re-probe path) instead of walking migration branches
  whose output the arch gate would reject a moment later. The 63 pre-v67 branches and, with this
  bump, the v68–v75 branches (including v75's SimSiam-deletion pops, which landed concurrently —
  the zip-kwargs sanitizer keeps that judgment) are deleted — their setdefault/pop history is preserved as
  documentation inside `_migrate_config`. `model_version.py` shrinks ~2,700 → ~2,300 lines.
  When `ARCH_SIGNATURE` next changes, raise the floor in the same commit — the test fails
  otherwise.

`designs/production_config.json` is now the gen-9 run's config **carried forward to the v76
schema by hand** (the v70–v74 defaults applied, `active_ctx_hidden` dropped) — it stops being a
byte-identical run-config copy until the first v75 run launches, and exists so the compile gate,
the delivery graph and the viewer keep deriving from the real production feature set (which
surfaced a latent `NameError` in the graph generator's `belief_slots` branch the moment the real
config exercised it).

### v77 — `gen3_intent_move_cell_v1` (2026-08-14): G3 — α consumed on the POLICY side

The G3 gate of `design_conditional_execution.md`, built: the `c2` status-consequence family
re-delivered through the pointer MOVE cell as a per-action ABSOLUTE, α-conditioned —
`--intent-move-cell`, structural + version-checked, fresh-only, OFF (default) builds nothing.

- **The operator** (`agents/model/intent_move_cell.py` + `DamageOperator.
  pointer_intent_status_operands`): per request slot m, the k-dependent c2 channels become the
  exact α-expectation over the op's OWN top-K seat candidates — `e_burn_alpha = Σ_k α_k ·
  (high_k(atk×0.5) − high_k(atk))`, `e_slp_alpha = Σ_k α_k · (−high_k)` — on the
  **UNRENORMALIZED** move slice (`f(m, SWITCH) = 0` is exact: a switching active neither
  attacks nor stands to be statused, so α_SWITCH mass shrinks the terms toward zero rather
  than being renormalized into "they attacked"). The k-independent columns (Δ-outspeed, the
  residual tick, sleep's expected free turns) ride raw vs their active with the seat mass
  `alpha_stay` as a decorrelated channel; the SWITCH-branch value needs β and is deliberately
  deferred to the class-B mechanics. Contract W holds: ONE α, softmaxed once from the
  PUBLISHED logits, board-only, shared across channels.
- **Delivery**: the 7 raw channels pass a ZERO-INIT `Linear(7,7)` and concat onto the 13-dim
  move cell (pointer scorer in_features widen — the version gate names the cause). Identity at
  init on a REAL MaskablePPO build (M1-captured); reads `last_alpha_logits` = the v75
  PUBLICATION, so `label_only` keeps cutting the PPO→α route here exactly as at
  `intent_value_reduce` (gradient-gated both ways in `intent_move_cell_test.py`).
- **Alignment**: the operand axis is the op's `last_topk_idx` — the same axis
  `intent_axis_alignment_test` pins element-wise to α's seats — and a width mismatch raises
  (the `op move-order` bug class), never broadcasts. Missing α / missing operands at runtime
  raise; only the construction-time width probe contributes shaped zeros.
- **Gates run**: model suite 1156 green; ON-arm `torch.compile(fullgraph=True)` clean
  (max|Δ| 5.4e-07); seat-permutation invariance, SWITCH-mass shrinkage, OFF-builds-nothing,
  fail-loud arms all pinned; delivery graph gains the cell edge (OFF in production, so the
  committed snapshot is unchanged apart from provenance); bridge smoke with the flag ON.
- **What this does NOT decide**: the G3 VERDICT — whether the re-delivered family comes alive
  — needs a trained run's per-arm audit. The instrument ships OFF; the first flag-on run is
  the gate.

### v78 — `gen3_flag_surface_p1_v1` (2026-08-14): the flag registry, and the first deletions it enables

Phase 1 of the CLI flag-surface cleanup. `train_rl_agent.py` carried ~170 long-form flags, and
every model-relevant one had to be spelled out by hand in **five** places. Nothing enforced the
five agreeing, and every historical failure in the class was silent — a toggle that reaches the
extractor but not the recorded config version-checks a resume against an architecture it does not
build; one with an argparse entry but no `_resolve` line silently reverts to OFF on a flagless
resume.

**The enabler — `agents/model/flag_registry.py`.** One declarative row per extractor toggle
(name · default · TIER · CLASS · `since` · one-line meaning). From it, two of the five surfaces are
now **GENERATED** (`extractor_arch.ARCH_ARG_KEYS` / `FROZEN_ARCH_KWARGS`, and
`current_model_version`'s toggle dict via `arch_toggles_from_args` — `_run_arch_toggles` was a
second hand-kept list of the same toggles and is now derived), and the other three are
**VALIDATED** by `flag_registry_test.py`, which fails naming the missing site. It earned its keep
on the first run: **three rows whose flag name is not `--<field>`** — `--damage-topk` writes
`damage_topk_k`, and the `--damage-matrices` MODE flag desugars into both `damage_matrices_*`
bools. `designs/flag_registry.md` is generated from the same table (`--check` is the gate).

The TIER axis is the reusable part. A flag has three roles — SELECT (choose it at launch), RECORD
(write it into `model_config.json`), GATE (refuse a mismatched resume) — and only SELECT needs
argparse, so a **settled** toggle can lose its flag with no loss of explicitness: `config_only`
keeps the recorded field, the version gate and the extractor CONSTRUCTOR kwarg, and freezes the
launch value at the registry default. Pinned end-to-end by `config_only_pattern_test.py`.

**Deleted (8 fields, 2 modules, 1 module family), all OFF in production:**

- **The ZARCH family** — `zarch_film` / `zarch_dim` / `zarch_lut` / `zarch_lut_teams` /
  `zarch_recon_coef` / `zarch_vicreg_coef`, with `ZArchEncoder`, both FiLM generators, the per-team
  LUT Embedding, `team_signature.py` + its roster table + fuzz, `attach_zarch_lut`, the
  `--film-grad-accum-steps` group accumulator and the `film/*` + `zarch/*` TB families. **The line
  it existed to test is closed and the result was NULL twice over**: the LUT arm — a FREE per-team
  code, the sharpest possible removal of a conditioning-signal limit — moved the N=20 multi-team
  ceiling by **+0.024, CI [−0.016, +0.064]**, and the orthogonal 2×2 measured team COUNT (20→10,
  **+0.077 SIG**) dominating conditioning (**+0.027 n.s.**).
- **The SEED-PRESSURE pair** — `seed_quantile` (v63) and `value_seed_vicreg_coef` (v62), with
  `seed_quantile.py` and `seed_vicreg.py`. **Both cap at ~1-D differentiation of the k=4 value
  seeds, from opposite directions**: gen-6's VICReg satisfied every term at
  `out_effective_rank` 1.05 (three seeds identical, one breakaway); gen-7's quantile arm drove
  `crossing_rate` to 0.000 and `quantile_spread` to 1.016 — the seeds genuinely predict four
  ordered quantiles — at `out_effective_rank` **1.157 of 4**. A SHARED readout can only constrain
  each seed along its own weight vector; every orthogonal direction stays free, so no coefficient
  reaches it. `seed_diagnostics.py` (the MEASUREMENT) stays; only the pressures go.
- **`--use-showdown-bridge`**, the deprecated `--use-bridge=node` alias (see below).

**Migration.** POP for a config recording them OFF; **REFUSE** for `zarch_film != 'off'` or
`seed_quantile=True` — those named PARAMETERS, so a silent pop would load a `state_dict` with keys
nothing can place (the v75 `opp_belief_latent` precedent). Both the config JSON (`_migrate_config`)
and the zip's pickled kwargs (`sanitize_dead_extractor_kwargs`) handle them, because they are read
at different moments by different code. The judged loop's `bool(recorded) is not supported` test
had to be typed first — **`bool("off")` is True**, so a truthiness compare would have refused every
OFF production config, i.e. every checkpoint that exists; pinned by a named test.

**NO `ARCH_SIGNATURE` bump and the MIGRATION FLOOR stays 76.** Verified rather than asserted: the
extractor built from `designs/production_config.json` before and after has **200 identical
`state_dict` keys** (same digest), identical shapes, and `max|Δ| = 0.000e+00` on both `pi` and `vf`.

**Demoted to `config_only` (3, fields and gates deliberately UNCHANGED — a demotion removes the
SELECT role only):** `attend_unrevealed_opponents` frozen **ON** (a hard prerequisite of
`opp_belief_cls_k>0` / `opp_belief_slots` / `move_belief_mode != off`; no run since v16 turned it
off, and freezing it removes two auto-enable branches and one `parser.error`),
`value_active_readout` and `damage_matrices_outgoing_all` frozen OFF (never enabled in a gen-8/9/10
run).

**Bridge default → `rust`** (no field consequence; the flag is a runtime knob). Serverless is now
the normal way to run training AND eval: `node` stays an explicit value for the A/B arm and the
parity harness, `off` is the websocket/ladder path. The launcher's `child_uses_bridge` inverts with
it — an ABSENT `--use-bridge` is now a BRIDGE run, so no phantom `--showdown-port` is injected;
`default_port_test.py` now exists to catch a drift between the two defaults.

**One gate bug fixed on the way.** The rust soak (`bridge_session_fuzz_test.py --impl rust`) failed
on all 4 workers at exactly episode 5001 — the intended `recycle_every=5000` HEALTHY-child swap,
reported as a crash by a flat `len(pids) > 1` assertion. The bound is `1 + n_recycles`; as written
the gate could not run past 5000 episodes at all, which is precisely the multi-hour regime it
exists for. 20,000 rust episodes / ~1.69M steps clean before it tripped.

### v79 — `gen3_pair_history_v1` (2026-08-15): Tier H-A — the compiled history tier

`design_history_entity.md` §3 H-A, the first landing of the history redesign. Obs **2669 → 2921**.

- **H-A1 — last-action fields on the ACTIVE slots** (`POKEMON_FULL_DIM` 116 → 122): each side's
  most recent executed action as `[last_move_id, was_switch, hit, miss, fail, crit]` on its
  active mon's slot, bench rows zero (the protect/trapped per-entity-fact convention). The move
  id is an EMBEDDING id: `slice_pokemon_categoricals` extracts it for the move table and ZEROES
  its raw column inside `hp_and_active` (the manifest rule — a raw dex num never reaches a
  Linear), and the role input widens by the embedded width. CANT windows leave the previous
  action standing; MISS/FAIL/CRIT attach to their side's same-turn move; leads don't count
  (a placement, not an action).
- **H-A2 — the pair-history block + the `h` edge family**: per (their mon i, our mon j), the
  5-cell `[switch_ins_i_while_j_active, attacks_i_on_j, status_clicks_i_on_j,
  shared_field_turns, recency_of_last_pairing]`, log-saturated over the 10 cap, folded CPU-side
  by the EpisodeTracker's `PairHistoryTracker` from PUBLIC events (seq-idempotent; species↔slot
  joins are battle-stable), transported as a 180-dim obs block after reactive, and consumed by
  the new **`h`** edge family at the mon×mon block — zero-init, obs-fed (the one family whose
  cell IS compiled battle history), **not in the production families string** (opt-in arm; the
  obs widening is unconditional). A ratio delivery is CORRECT for tendencies, unlike magnitudes.
  The intended first consumers are α/β — their tendency inputs finally exist; watch
  `opp_intent/alpha_acc` across the first run that carries this.
- **The fuzz found a real bug before it shipped** (`pair_history_fuzz_test.py`, the recency-fuzz
  pattern with a fully independent event-log oracle): at a FORCED-SWITCH decision poke-env still
  reports the fainted mon as active, so the tracker's decision-time resync RESURRECTED an active
  the FAINT event had correctly cleared — pairing replacements against a dead mon (reproduced on
  a double-KO Explosion). Fixed with an alive-filter on the resync. The same pass also fixed a
  pre-existing cross-episode leak: `RecencyTracker` was missing from the env-path episode reset.
- Versioning: stamp-only branch (the v67 pattern) — total_dim + the widened role-encoder shapes
  are weight-field-caught; NO ARCH_SIGNATURE bump (the recency precedent); the family rides the
  recorded `edge_bias_families` string. Gate results recorded in the shipping commit.
  (Written as v78 pre-rebase; renumbered v79 on landing — `gen3_flag_surface_p1_v1` took 78
  concurrently, the v75/v76 renumbering precedent.)

### `gen3_smogon_cooccur_prior_v1` (2026-08-15, NO config bump — a data-source swap, retrain-class)

The species co-occurrence prior behind BOTH the v69 `species_prior_fusion` naive-Bayes read and
the v72 `T0SpeciesPrior` was sourced from the 719-team POOL (`data/teams/gen3_species_priors.json`)
— the one prior in the model that violated the owner rule stated 2026-08-15: **priors are always
Smogon-based, never pool-based**. `build_species_cooccur_prior` now derives from Smogon: the
marginal is `build_species_usage_prior`'s normalized usage share, and the lift comes from the
chaos `Teammates` joint (`gen3_teammate_priors.json` / `gen3_data.priors.teammates`, ~2.5M gen3ou
battles, 12-month merge), clamped to ±4 in place of the pool source's pseudo-count shrinkage;
forme keys fold into their base num. The swap VALIDATED the rule with a concrete artifact: the
pool's strongest positive pair (Cloyster→Aerodactyl, log-lift +1.32) measures **+0.23** on the
ladder — a sample-team archetype baked into two generations of belief priors. New anchors:
Jirachi|Flygon +1.12 (the JiraGon core), Forretress|Skarmory −2.51 (redundant spikers);
GIGO guards now pin the Tyranitar marginal AND positive Skarmory|Tyranitar lift. The math,
shapes, buffers and consumers are unchanged (the fusion parity test now checks the matmul
against a numpy walk over the same tensors instead of the pool estimator);
`agents.training.species_priors` remains as a pool-ANALYSIS tool only. No ModelVersion field
moved — like any `data/` prior change this is retrain-class, not resume-gated.

### v80 — `gen3_unified_value_readout_v1` (2026-08-15): the Stage-3 critic delivery contract, built

`design_unified_belief.md` §3 named the critic's delivery problem: four parallel magnitude routes
(value CLS pool, `MultiSeedValueReadout`, `--value-threat-inject`, the hidden-opp concat), two of
which exist only because the others could not reach vf — "the signature of a missing contract."
This ships the contract as an OPT-IN successor, ahead of the gen-11 audit that decides the
condemnations:

- **`UnifiedValueReadout`** (`--value-entity-pool`, STRUCTURAL v80 field): ONE attention pool
  over the critic's entity-row set — the 12 post-transformer team tokens plus (when the op
  exists) its 6 per-our-mon incoming rows — rows projected to `UVR_DIM` 64 with per-SOURCE type
  embeddings, pooled by `UVR_K` 4 learned queries, through a ZERO-INIT output projection
  (`UVR_OUT_DIM` 128) appended to vf ONLY after the assembler (the intent_value_reduce
  placement: pi untouched at ANY weight). Explicit softmax (the MultiSeedValueReadout pattern),
  so an all-masked row set degrades to a uniform average, never NaN, and the K×N attention is
  stashed (`last_att`). Works without the op (the row set shrinks to the team tokens).
- **Permutation-invariant by construction** — a row's identity rides its content + source tag,
  never its position; the pointer head stays the policy's per-action pool. Same object, two
  pools, one contract (§3's sentence, now code).
- **Fully declared**: a `flag_registry` row (the registry test drove out the two sites the hand
  pass missed — the `check_compatible` gate and the generated doc), argparse + `_resolve`,
  ModelVersion field + named mismatch error, v80 setdefault migration, and the zero-init out
  projection is picked up by the end-of-`__init__` identity-init sweep (SB3 ortho-clobber safe,
  asserted by test).
- **The audit gains an `entity_pool` arm** (`critic_route_audit.py`), so the run that enables
  this measures it with the same instrument that condemns its predecessors.
- OFF builds nothing — byte-identical baseline, no ARCH_SIGNATURE bump; production stays OFF
  (`config_version` 80 stamped). Gates: `value_entity_pool_test.py` (OFF-builds-nothing,
  zero-init survives policy build + contributes exactly 0, pi untouched at any weight while vf
  fires, masked rows get zero attention, all-masked is finite, fail-loud op-rows contract, the
  v80 migration stamp).

### v81 — `gen3_event_window_v1` (2026-08-16): Tier H-B — the event-token history window

`design_history_entity.md` §3 H-B, the second landing of the history redesign. Obs **2921 →
3529** (the 32×19 event-record block closes base at 2405); the CONSUMER is opt-in.

- **The fold** — `EventWindowTracker` (episode_tracker, the H-A machinery: seq-idempotent,
  running actives, alive-filtered decision resync, same per-decision window). One record per
  event in the H-B vocabulary (move / switch_in / faint / status applied/cured / boost /
  item_reveal / hazard / switch_rejected); MODIFIER events attach to their side's open
  same-turn MOVE record — DAMAGE lands on the target side and accumulates into the move's
  `hp_delta`, the effectiveness trio sets `eff`, MISS/FAIL/CRIT set flags (the H-A attach rule
  extended). `we_first` marks the first mover's records (speed-inversion evidence);
  `forced_window` tags events emitted while an active slot sat empty post-faint, and the
  arriving replacement is tagged BEFORE the flag clears (a fold-order bug the unit tests
  caught: appended-then-clear, so the forced switch-in reads as forced). v1 trims, recorded:
  no faint-cause multi-hot, no item/hazard content ids, SETBOOST/CLEARBOOST skipped.
- **The obs block** — rows most-recent-LAST, zero-padding at the FRONT, ids as embedding ids
  (species/move dex nums, the type/status vocabularies in `observation/constants.py`), recency
  log-saturated (the H-A curve), no Linear reads it raw. The side column caught a literal bug
  in test (`"our"` vs the actual `OURS == "ours"`) before the goldens froze it.
- **The consumer** — `--history-events` (STRUCTURAL v81 field): `EventSeats` — kind/status own
  small embeddings, actor/target species + the move id through the SHARED tables (one
  representation everywhere), 13 outcome/time scalars raw, one projection to d_model, a learned
  `event_marker`, and TOKEN_TYPE_HISTORY (the E5 precedent — no token-type table growth). Seats
  join the `extra` seam LAST, so every front-indexed seat slice (E3/E4/E5) is position-stable;
  PAD rows are key-masked. OFF builds nothing — byte-identical; production stays OFF
  (`config_version` 81 stamped). Declared T1 in the tier contract (trunk input, like
  entity_seats).
- **The goldens got STRONGER in the same pass**: `golden_obs_capture` had never run the env's
  `update_progress_clock` leg, so the progress-clock scalar, the recency triplet and BOTH H-A
  blocks were frozen as zeros in every golden since they shipped — the capture now runs the
  full 3-step protocol and threads all four tracker feeds, so the fixture finally pins them.
- Gates: `event_window_test.py` (fold attach/idempotence/forced-window/bounds, the 19-column
  obs contract, OFF-builds-nothing, PAD masking, the v81 migration stamp); the flag-registry
  validators (all five surfaces); the delivery graph auto-discovered the block
  (`obs.event_window` node). **The event-fold FUZZ ran and PASSES** —
  `poke_env_gaps/event_window_fuzz_test.py`, an independent from-scratch fold over the full
  event log: 30 battles / 2429 decisions / **73,008 checks, 0 failures** — and it TIGHTENED
  the fold before the goldens froze it: a `[from]`-claused DAMAGE (recoil / Sandstorm /
  status / item residuals) or a clause-free hit on a NON-target mon no longer attaches to the
  open move's `hp_delta` (unit-pinned on a Double-Edge recoil + sand + wrong-target trio).
  The remaining pre-enable gate is the obs benchmark on an idle box (the +608-dim fold cost),
  per `gen12_endofrun_runbook.md` §4. **[same day] BOTH pre-enable gates CLOSED**: benchmark
  re-baselined on the idle box (0.363 ms/decision full protocol; H-B marginal +0.040 ms — and
  the measurement-honesty finding that benchmark AND goldens had never threaded the
  tracker-fed blocks, both fixed), and the `--history-events` bridge smoke trains to completion.

### `gen3_event_ref_edges_v1` (2026-08-16, NO config bump — family vocabulary): Tier H-C built

The third history tier, completing the design's stack: the **`r` edge family** — per (event
seat e, mon token m), the STRUCTURAL `[is_actor, is_target]` reference cells
(`_event_reference_cells`, pure): species-num equality, SIDE-GATED so a mirror species across
teams can never false-link. Written at (the LAST-N event rows, the 12 mon cols) + transpose —
the two critical queries become single attention hops ("what did they click into THIS mon" =
mon j attending over its target-edges; "whom did they switch into" = the switch-in event's
actor-edge). Zero-init like every family; rides the recorded `edge_bias_families` string (no
new ModelVersion field — old code fail-louds on the unknown family); requires
`--history-events` (the seats are the rows), fail-loud in `__init__`. Events referencing
FAINTED mons: the mon KEY is masked, so the event→fainted hop is inert while mon→event
survives — the accepted v1 nuance, now stated at the cell constant. Gates:
`event_ref_edges_test.py` (the mirror side-gate exact, PAD links nothing, the requirement,
zero-init + a full ON forward). Also in this pass: the `critic_route_audit` gains the
`intent_reduce` arm (gen-9..11 all TRAIN the route; the runbook's Phase-3 list named it but no
arm existed), and building its fixture flushed a REAL latent v80 bug — `intent_value_reduce`'s
discovery-time early return skipped the entity-pool concat, so a both-flags build discovered a
vf width 128 short of runtime (now falls through; caught by the audit fixture's shape error the
first time both flags met).

### `gen3_belief_bank_v1` (2026-08-16, NO config bump — a code-shape fold, byte-identical)

`design_unified_belief.md` §4's consolidation, complete: **all six supervised belief losses**
(hidden-team Hungarian aux, move-belief BCE, move-latent grading, spread, nature/EV, hp-type)
now live in `agents/training/belief_bank.py` as declarative ROWS — arg spec
(stash/attr/obs/param) · coef key · metric prefix · historic loss key — and
`instrumented_ppo.train()`'s six inline verticals collapse into three `compute(site=…)` loops
at the EXACT positions of the blocks they replaced (`hidden_move` / `latent` / `revealed`).
The SITE tag is the byte-identity mechanism: float addition order is preserved because each
row folds where its block folded; the loss bodies moved verbatim (the old
`InstrumentedMaskablePPO._*_loss` statics remain as aliases, so every test and call site
resolves unchanged). A seventh supervised belief is now a registry row, not another
~35-line train() vertical. Gates: `belief_bank_test.py` (direct-call equality bit-for-bit,
site partition pinned, the attr/param arg kinds, the hidden-team `aux_loss`/unprefixed
conventions, all six aliases) + every pre-existing per-loss test file unchanged.

### v82 — `gen3_unified_value_readout_v2` (2026-08-16): the entity pool's COMPLETE row set

`--value-entity-pool-full` (STRUCTURAL v82 field, requires `--value-entity-pool`): the Stage-3
critic pool gains its last two row sources — the REFINED global token (source 3, never masked;
stashed by the transformer as `last_global_out`, a side output) and the hidden-opp belief
queries (source 4, `[B,K,D_MODEL]`, present when the HiddenOppBeliefPool exists). With them the
pool covers every content class the vf concat routes carry, so **every route the
critic_route_audit can condemn now has ONE successor**: `nmr`'s direct concat (global content),
the hidden-opp vf half (belief content), seed and threat (op-row content). Its OWN field on
purpose — `full` grows the source-embedding table 3→5 (a state_dict shape), and gen-12 trains a
v80-shape pool that must keep loading (`full=False` builds the byte-identical 3-row table,
pinned by test). Zero-init out projection unchanged ⇒ cold start still contributes exactly 0.

Same pass: the `event_seats` audit arm (the H-B "usage audit on the event seats" in the house
ablation form — key-mask all seats, read KL/|dV| on a trained run; NOT a zero-init route, so
nonzero at init is expected and asserted), and the **H-tier compile gates** —
production + `--history-events` + the full 17-family string including `h,r` compiles with
suppression OFF, matches eager <1e-5, and holds the ONE-GRAPH property (0 breaks) that the
6.5× B=1 win depends on — measured beside the live gen-12 run at load 42. Gen-13's
`--compile-opponents` path is proven before any launch relies on it.

### `gen3_endofrun_battery_v1` (2026-08-16): the end-of-run battery, mechanized

`python -m main.endofrun <run> [--ref <prev-run>]` — the per-generation verdict loop that was
hand-driven from a runbook every ~2 days becomes one command emitting ONE artifact
(`measurements/<run>_endofrun.{json,md}`). Steps fail SOFT with recorded reasons: the dense
anchored ladder read by the TAIL-4 matched-count convention (the ELO reading rules — an
under-sampled tail refuses to report rather than emitting the inflated mid-run number) → the §5
non-inferiority rule; `critic_route_audit` arms → the §2 deletion ratio (<20% of `all_off` |dV|
AND <2% flips) with the pre-registered substitutability CONFOUND note whenever the single arms
sum well under the joint; `edge_ablation_audit` per-family → the family-alive bar (≥½ the
median live family); `awareness_scan` twice → instrument directions vs the recorded gen-10
baselines (coverage judged on ALL outcomes — the loss-filter PIT bias is pre-registered). A
model-loading step on an arch-drifted run reports `needs_pinned_tree` with the exact worktree
commands instead of an error. The runbooks remain the registration of record; the runner cites
their rules and the rule functions are pure + pinned (`endofrun_test.py`).

### v83 — `gen3_item_belief_v1` (2026-08-16): the hidden ITEM becomes a belief, and the op's last static prior factor goes learnable

`--item-belief` builds **`ItemBelief`** (T0 RESOLVE): a per-opp-slot posterior over item nums —
the per-species Smogon item-usage prior (`build_item_prior`, row floor 1e-5, GIGO-anchored:
Blissey Leftovers 0.992) ⊕ a zero-init trunk delta, so the cold-start posterior equals the prior
EXACTLY (softmax of log-prior; protected from the SB3 ortho clobber by the identity-init sweep).
Its consumer is the op's Choice-Band-conditional tail: the UNREVEALED branch's `cb_prior` now
reads P(item==CB) from the **publication** (`last_item_logits`, `_publish_belief` — so under
`label_only` no PPO gradient reaches the head) instead of the static `SPECIES_CB_PRIOR` scalar;
the revealed 0/1 exactness gate is untouched, and `item_cb_prob=None` (flag off) is
byte-identical. The prior's CB column sits within 0.6% of the static table (measured max|Δ|
0.0059), so enabling is ~behavior-preserving at init and the delta must EARN its movement —
through the new CE and the op's damage gradient.

Why now: the evidence stream this head needs arrived at v79 — the tokens carry last-action
fields and pair tendencies, so "they clicked two DIFFERENT moves ⇒ not Choice-locked" is
representable, which no static per-species scalar can express.

Supervision is the **BeliefBank's seventh row** (`item_belief_loss`, site `revealed`, appended
last so the float-addition order of the existing six is untouched — the bank consolidation
paying off on its first arrival: the whole train()-side cost is one loss fn + one ROW).
`Gen3Env` emits `item_label`/`item_mask` (privileged true item num from agent2's own team,
matched by species; num 0 "nothing" is a CLASS, not PAD) gated on `--item-belief` +
`--item-belief-coef > 0` (default 0.05, auto-zeroed with a warning if the head is off).
STRUCTURAL, version-checked (`item_belief`, config v83, migration default False); the coef is
training-only, recorded for provenance. Gates: `item_belief_test.py` (11 — cold-start==prior,
CB-column tracking, op seam identities incl. the revealed-exactness pin, zero-init sweep
membership, labels, bank row, migration, check_compatible), plus the extended identity-init /
label-only / bank-partition / flag-registry pins.

### v84 — `gen3_intent_threshold_v1` (2026-08-16): five mechanics, one operator — and the critic finally gets a calibrated p_KO

`--intent-threshold` builds `design_conditional_execution.md` §3.0's shared primitive — the
single most important structural fact in that document: Focus Punch, Substitute, Endure,
Destiny Bond and Endeavor are the SAME computation, `p_thresh(τ,⋛) = Σ_k α_k·1[damage(k,me) ⋛ τ]`,
with a different threshold and direction. No new physics: `threshold_probs` is one gather +
three contractions over the op's existing `last_pair_cells` stash ([low, high, crit, ko, acc,
is_phys] per (defender, seat candidate), already oracle-gated by the damage probe fuzz),
α-weighted on the UNRENORMALIZED move slice (the missing SWITCH mass correctly reads "no damage
this turn" — the IntentValueReduce precedent). Focus Punch's immunity term — the doc's
likeliest-G0-mistake — falls out of the physics for free (immune ⇒ eff 0 ⇒ high 0 ⇒ no break).

Two zero-init consumers, one producer (probs computed once at the pointer stash where α first
exists): **`IntentThresholdMoveCell`** (T2) appends per-request-slot channels to the pointer
MOVE cell — `[is_fp·(1−p_fp_broken), is_sub·(1−p_sub_broken), is_endure·p_KO, is_dbond·p_KO,
is_endeavor·(1−p_KO), p_KO]` — the per-action-absolute channel measured to work; and
**`IntentThresholdValue`** (T3) appends `[p_KO, p_sub_broken, p_fp_broken]` to the CRITIC after
the entity pool (fall-through discovery — the ede5a88 lesson, now pinned with all three value
flags on at once). The critic half is the ledger-H1 payoff and stands whatever the G3 verdict
says about the mechanic cells: H1 measured the critic over-valuing a healthy self-KO trade
(dV ≈ +2.9 against a −2.7 reward) because "am I about to die" reached it only as
`_chan_max`'s hard max over believed moves; `p_KO` is the same tensor under the correct
functional. §3.0's second-moment point is pinned as a test: two candidate sets with the SAME
mean damage produce different sub-break readings — a threshold on the roll distribution is not
a function of its mean, which is why `max` could never represent any row of the table.

STRUCTURAL, version-checked (`intent_threshold`, config v84, migration default False); requires
`opp_intent` + `damage_op` at build and the top-K pair-cell stash at runtime (fail-loud, the
`op move-order` class). Gates: `intent_threshold_test.py` (15 — the contraction math, the
SWITCH-mass shrink, mean-vs-threshold, seat-permutation invariance, mechanic-gate routing,
zero-init on both heads + sweep membership, fail-louds, the all-value-flags discovery build,
migration + check_compatible) and the compile cell
(`test_intent_threshold_arch_compiles_to_one_graph`: production + the flag = ONE graph, 0
breaks, compiled == eager <1e-5). E2E smoke: round-trip + PPO healthy with the flag on.
Not yet done, recorded honestly: the full G0 constructed-scenario oracle for the per-mechanic
EV terms (the damage INPUTS are oracle-gated; the mechanic-level expectations are not), and the
class-B β-weighted branches (Explosion, Pursuit) stay at build-order step 7.

### v85 — `gen3_intent_conditional_v1` (2026-08-16): Counter becomes playable, flinch learns about switches, and Pursuit gets the RIGHT rule

`--intent-conditional` builds `design_conditional_execution.md`'s remaining steps 4+7 cells —
per-request-slot, α-contracted, over tensors the op already stashes (the pair cells' high/is_phys
columns, `last_topk_idx`, the outgoing per-move rolls, `p_outspeed`, the secondary flinch
column). One zero-init `Linear(8, INTENT_COND_MOVE_DIM)` on the pointer MOVE cell:

* **Counter / Mirror Coat** — the α-weighted CATEGORY sums (`Σ_k α_k·is_phys_k·dmg_k·high_k`
  and its special mirror) plus `p_category_match`. The doc's words: "the purest
  read-the-opponent moves in gen3 — literally unplayable without an intent model."
* **Flinch** — `p_outspeed · p_flinch · (1 − α_SWITCH)`: the raw chance already rode the
  secondary columns; the conditioning that makes it MEANINGFUL did not.
* **Explosion / Self-Destruct** — `p_executes = 1 − Σ_k α_k·is_protect_k` (the worst branch is
  α-visible) and the into-switch mass, decorrelated — the representation-side companions to
  ledger H1 (the doc argues conditioning Explosion's value on α is the representation fix for
  the same defect `--self-ko-hp-penalty` patches at the reward).
* **Pursuit — CORRECTED against the port.** The design's §3.6 formula weighted the doubled
  damage by a β-weighted switch-IN; `src/rust_sim/state.rs`'s pursuit interrupt (golden-gated
  vs Showdown) strikes the DEPARTING mon at ×2 BP never-miss before the switch resolves. So no
  β enters, and the cell carries `α_SWITCH` (the trigger) and `α_SWITCH·high` (the bonus)
  against the CURRENT active. The doc is the registration of record and was not edited; this
  entry records the discrepancy for the owner to reconcile.

**G2 is now MEASURED, not assumed** (`agents.model.mechanic_usage_baseline`, model-free over the
eval-trace `actions` blocks; artifact
`measurements/gen12_mechanic_usage_baseline.json`, 61,865 gen-12 decisions): Endure picked
**0.0%** of the 48 times it was legal, Substitute **0.9%** of 2,834, Counter **5.6%** of 591 at
9.2% mean prob, Explosion 5.6% of 5,719, Pursuit 3.4%, Protect 24.7%. These are the numbers the
v84/v85 retrains have to move — §3.7's "it is unsurprising if the current policy simply never
clicks them" is confirmed in the tail (Endure/Sub) and directionally right everywhere.

Same pass, the rest of the class-A set + the class-B β half (steps 5+6+7 complete):

* **Protect / Detect** (step 5) — `c4` carried the mechanical `p_success` multiplier and
  omitted the quantity it multiplies; the cell now carries the α-weighted avoided DAMAGE, the
  same obs decay-odds scalar, and the α mass on STATUS seats (typed from the data facade, so an
  immune damaging seat cannot masquerade as status), all decorrelated. Endure deliberately
  excluded from this gate — its value is v84's `p_KO` branch.
* **Magic Coat** (step 6) — its G0 oracle ran FIRST: five constructed scenarios on the
  reference sim (`measurements/gen3_magiccoat_reflectable_oracle.json`) resolved §3.12's
  UNVERIFIED set — foe-targeting status (Toxic/T-Wave/Leech Seed/WoW) BOUNCES, side-targeting
  Spikes does NOT (it lands on the user's own side). The cell's `is_reflectable` predicate
  encodes exactly that boundary and the test pins it.
* **Explosion's β half** (step 7 — the FIRST forward-side β consumer): the trade's target
  differs by branch, so the cell carries `α_stay·pko(boom, their active) +
  α_SWITCH·Σ_j β_j·pko(boom, arrival j)`. **β is now PUBLISHED like α** — the supervised
  intent CE keeps the LIVE view (`belief_supervision("beta_logits")`), the forward reads the
  stop-grad publication under `label_only`, so the class-B consumption cannot reopen the
  PPO→beta route. The arrival pko comes from the outgoing matrix (which prices an unrevealed
  arrival's P(KO) as NULLED — unrevealed β mass honestly contributes zero rather than a guess),
  which is why the flag requires `damage_matrices_outgoing`.

STRUCTURAL, version-checked (`intent_conditional`, config v85, migration default False);
requires opp_intent + damage_op + damage_outgoing + damage_matrices_outgoing (+ the top-K
stash at runtime, fail-loud).
Gates: `intent_conditional_test.py` (14 — the category math both directions, the
status-feeds-neither pin, the flinch/boom/pursuit α_SWITCH conditioning, permutation
invariance, width fail-loud, zero-init, the FULL intent stack v77+v84+v85 discovery build,
migration + check_compatible); the compile cell now builds production + BOTH intent riders
(one graph, 0 breaks, compiled == eager). E2E smoke: round-trip + PPO healthy with both flags.

### `gen3_op_candidate_dedup_v1` (2026-08-16): the E4/d3/s3 recompute-dedup — the open half of op_tensors step 2, closed byte-identically

The [B, n_moves] candidate-weight build (`_opp_candidate_weights`: the belief sigmoid + the
typed-HP scatter + the bare-237 presence mask) ran TWICE per production forward — once in the op
forward for the incoming matrix's top-K, once in the E4 seat builder's `refine_candidates`. The
op forward now stashes it (`last_w_all`, CLEARED at forward entry so a stale batch is
unrepresentable — None means the standalone fallback computes) and the seat builder reuses it
via `refine_candidates(w_all=…)`. Byte-identity PROVEN, not asserted: the production-config
pi/vf sha256 is unchanged pre/post (`3cab191a…`), and the regression test pins reuse ==
standalone bitwise plus the clear-at-entry. (Backward: mathematically identical; the shared
node sums its two consumers' gradients before the sigmoid backward where the duplicated path
distributed it — the last-ulp FP-ordering class this file already documents for
`_damage_rolls`.)

What the audit of the REMAINING "recompute" entries found, recorded so nobody re-opens them:

* **d3/s3 already share the candidate SELECTION** — both receive `entity_seats.last_cand`
  (the v55-era stash), so seat c, bias row c and the α seats name the same move by
  construction. Their per-candidate gathers are trivial.
* **`pairwise_incoming`'s lean rolls are NOT the incoming matrix's rolls** — the lean kernel
  prices a LEGACY de-timid attacker (no spread belief / boost / burn / weather / fixed) BY
  DESIGN ("the coarse signal; the full post-transformer op is authoritative"), so replacing it
  with the matrix's values would be a semantic upgrade, not a dedup — retrain-class, and it
  belongs to op_tensors step 3's era, not this pass.
* **`discrete_incoming` is DEAD in production** — the v70 refine-loop deletion orphaned its
  only live caller; tests alone reach it. So `_incoming_rolls` runs ONCE per forward and the
  design's "called from two sites" reading describes the pre-v70 world. Deleting the orphan is
  cleanup-journey material, deliberately not taken here.

### `gen3_forkserver_preload_v1` (2026-08-16): the forkserver compile preload — revived by fixing the fork hazard at its root

The 2026-08 `set_forkserver_preload` attempt wedged a real 48-env run (2 workers forked of 48,
parent blocked in `unix_stream_data_wait`, box at 0.2 load, no error anywhere) because `fork()`
copies every mutex but only the calling thread, and importing the extractor started poke-env's
GLOBAL asyncio loop thread — any `poke_env.x` import executed the eager package `__init__` →
`player` → `ps_client` → `concurrency`. The planned fix was a ~12-file model-layer refactor
(TYPE_CHECKING conversions + vendoring); the shipped fix is one level deeper and far smaller:
**`poke_env/__init__.py`, `poke_env/player/__init__.py` and `poke_env/battle/__init__.py` are
now LAZY (PEP 562)** — public surface unchanged (module `__getattr__` resolves names + submodules
on first access), the battle/data/enum subtrees the extractor needs are thread-free, and the loop
thread starts exactly when a player/client module is imported, which is what every training-side
consumer does anyway. The laziness also DISSOLVED an order-dependent circular import the eager
inits had been masking (`battle/__init__` → `battle.battle` → `player.battle_order` →
`battle.move` — fatal for any entry that began in `battle_order`; probed green in all orders).

On that foundation, `--compile-opponents-preload` arms `agents.model.compile_preload` via
`set_forkserver_preload`: the extractor is compiled ONCE in the forkserver (config as DATA via
`GEN3AI_PRELOAD_ARCH` — the `arch_kwargs_to_plain` seam built for exactly this) and every env
worker inherits the traced graph by fork (~0.12 s vs ~30 s per worker against a warm disk cache).
Three guards, all LOUD: `extractor_import_is_fork_safe()` (import ⇒ single-threaded — the old
"currently FAILS" pin now asserts the inverse and names the regression), `compile_threads = 1` +
`shutdown_compile_workers()` (the Inductor pool never exists), and the preload RAISING if any
non-main thread survives its compile — killing the forkserver bootstrap so env construction
fails with a traceback in the parent; the silent wedge is unrepresentable. Proven live: a real
4-worker `SubprocVecEnv` CPU run with the preload armed compiled once (41 s), forked all
workers, trained to completion — the exact scenario that hung in 2026-08. Runtime perf knob
(requires `--compile-opponents`, never versioned, re-pass each launch); when armed it replaces
the in-trainer cache prewarm. The honest sizing is unchanged (~50 s per 3 h restart): the reason
to ship it is that the architecture now permits it safely, not throughput.

### v86 — `gen3_op_lean_forward_v1` (2026-08-16): op_tensors step 3 lands, the lean physics go believed, and the dead kernels go entirely

**`gen3_op_dead_kernel_cleanup_v1` first** (no flag — pure deletion of test-only orphans found by
a caller audit): `discrete_incoming` + `discrete_outgoing` (both orphaned by the v70 refine-loop
deletion), `Embeddings.hp_latent_block` (the code's own comment had already declared it obsolete —
the typed HPs 355-370 carry their own latent rows), and the `_locate_active_slot` back-compat shim
(the free function is the live path). The tiered-pipeline pin upgrades from "the lean kernel is
never CALLED" to "it does not EXIST"; the expected-latent species-gradient property re-pins on the
LIVE consumer (`_outgoing_matrix`'s unrevealed pricing) instead of the deleted kernel. The audit's
non-dead verdicts are recorded too: OAX and the pair-reduce rungs are config-reachable/spared, and
the flagged property/dim accessors were false positives (attribute reads).

**`--op-drop-renders`** (design_op_tensors step 3): the flat forward block loses its three RENDER
regions — outgoing matrix, incoming matrix, OAX — which have had no forward consumer since the
head-concat's deletion (`gen3_no_concat_v1`). The matrices' SELECTION machinery still runs (the
top-K index α's seats align to, the pair cells, and the new typed `last_out_pko` stash the v85
boom cell now reads in BOTH modes — pre-gain, so honest probabilities rather than the
learned-gain-scaled render values; pinned equal to the old view at gain-init). The renders always
appended LAST, so every surviving offset is unchanged — and the step-3 "gone by construction"
claim is an executable fact: **pi/vf at init are BIT-IDENTICAL between the two modes** (pinned by
test), while `out_gain` shrinks (state_dict, hence its own version-checked flag). OFF is
byte-identical (the production sha probe reads unchanged: `3cab191a…`).

**`--op-believed-lean`**: the lean d3 physics (`_incoming_rolls`) price the attacker from the
BELIEVED spread instead of the legacy de-timid 252-EV/boosting-nature fiction — the B-spread
correctness fix (`design_opponent_intent.md` §4.5: "pair_in computed against a fictional
maximally-invested opponent... distorts the RELATIVE threat ordering") applied to the last
de-timid site the edges read. Requires `--spread-belief` (fail-loud, verified live). Forward-math
only — the version gate is the ONLY thing that rejects a mismatched resume, and the entry says so.

Both flags are gen-13+ riders; production stays OFF and byte-identical. Gates:
`op_lean_forward_test.py` (10 — the bit-identity pin, lean width + offset preservation, view
nulling + stash survival, the full v77+v84+v85 intent stack running lean, the pko source-switch
init-neutrality, the believed-vs-legacy pricing split on a real defender, migrations + both
check_compatible gates); the compile cell now builds production + the ENTIRE gen-13 candidate
stack (both intent riders + both v86 flags) as ONE graph, 0 breaks, compiled == eager.

### v87 — `gen3_value_direct_routes_v1` (2026-08-16): the critic finally sees the clock it loses on and the intent it weights by

Two direct CRITIC routes, both zero-init vf-tail appends (the `intent_value_reduce` placement,
fall-through discovery — the ede5a88 rule, now pinned with EVERY value part on at once):

* **`--value-clock`** — the v67 deadline clock (`gen3_deadline_clock_v1`) was built for exactly
  this reader: "a critic cannot price a deadline it has no resolution on", motivated by a
  positive V on the final decision in 13 of 14 timeout losses. Its only direct head route since
  the ctx-dedup era is the `non_matchup_rest` concat, which the route audit read as dead — the
  validated fix gets its own explicit route: the 3 raw scalars (log-elapsed, remaining-linear,
  log-remaining, sliced by the NEW named `CLOCK_OFFSET_IN_GLOBAL`, never a hand-counted index)
  through a zero-init projection, vf only.
* **`--value-intent`** — α/β reach the critic AS DISTRIBUTIONS for the first time: α entered vf
  only as a weighting inside `intent_value_reduce`'s physics cells, β not at all. The block was
  ORDERING (the T2 heads are scored after the pools), which the post-assembler tail dissolves.
  The route reads the PUBLICATIONS (stop-grad under `label_only` — no PPO→α/β route opens): α
  softmaxed over its K belief-sorted seats + SWITCH (the canonical axis every α consumer aligns
  to; a seat-count mismatch fails loud), β over the 6 team slots with the no-legal-candidate
  case gated to a clean zero, never a NaN softmax.

Both STRUCTURAL, version-checked (v87, migration defaults False; they widen the value
projection so a mismatch is shape-caught — the checks name the cause). pi is untouched at any
weight (vf-only concat, pinned). Gates: `value_routes_test.py` (10) and the compile cell now
builds production + the ENTIRE rider stack (v84+v85+v86+v87) as ONE graph, 0 breaks. E2E smoke
green with both flags.

### `gen3_op_stashes_v1` (2026-08-16): the typed refactor of the side-value surface — byte-identical

The OpTensors discipline applied to the STASH surface. The op carried ten independently-managed
`last_*` attributes with three different clearing conventions — `last_w_all` cleared at forward
entry (hand-built), the pair stashes cleared in an else-branch, and the top-K trio not cleared at
all (a `last_topk_idx` from the PREVIOUS batch silently survived any forward in which the matrix
path didn't run). They are now ONE `OpStashes` dataclass replaced as a unit at forward entry, so
a stale batch is unrepresentable for every stash at once. Reads stay on the documented `last_*`
surface (now read-only properties — the re-export convention), and a stray WRITE to a `last_*`
name fails loud instead of silently forking the state (pinned by test). The extractor half: the
anonymous 5-tuple `last_pointer_inputs` becomes the `PointerInputs` NamedTuple and the
threshold stash becomes `ThresholdProbs` — still tuples, so every positional unpack is
unchanged. Byte-identity proven (the production sha probe reads `3cab191a…` unchanged after
each phase), the compile cells hold (dynamo traces the properties + NamedTuples to the same one
graph), and the publication/`belief_supervision` surface is deliberately untouched — its typed
accessor and publish boundary are already the enforced contract.

## v88 — `gen3_dead_flag_purge_v1` (2026-08-16): three dead flags deleted outright, plus the whole pubval subsystem

The cleanup-journey deletion pattern (v75/v78) applied to the three surviving flags whose OFF
value was the only value any generation ever ran:

* **`value_active_readout`** (v10) — the active-token vf readout. Demoted to config_only at v78,
  frozen OFF, superseded by the multi-seed readout and `--value-threat-inject`. Field, gate and
  forward branch deleted.
* **`damage_matrices_outgoing_all`** (v39) — the OAX transposed outgoing flat block + its 18-dim
  pointer switch-cell extension (`_PTR_SWITCH_CELL_OAX`, the 33-wide switch cell). Deleted with
  its flag — **but the `_outgoing_attacker_matrix` KERNEL survives untouched as `d2`'s engine**
  (`pairwise_bench_outgoing` calls it every production forward): what died is the flat-block
  RENDER and its never-enabled head delivery, not the physics. The switch cell is 15 wide,
  unconditionally.
* **`pubval_mode` / `pubval_coef`** (v43) — the entire public-replay value aux subsystem:
  `PubValHead`, `agents.training.pubval`, `pubval_calibration`, `_pubval_loss`, the env target
  emission (`_pubval_target` + the `pubval_target`/`pubval_mask` obs keys), the argparse pair,
  the launcher labels, the parity fuzz, and the committed artifact `data/gen3_pubval.json`.
  Measured NULL as a lever; never ON in a production generation. The raw replay corpus and the
  design doc remain.

Migration: a v88 purge loop runs for EVERY config vintage — a recorded ON value names
parameters/widths the surviving code cannot rebuild and is REFUSED with the re-read-from-git_hash
diagnosis (the v75 rule); a recorded OFF/none value pops silently; `pubval_coef` (training-only,
INERT for a forward) pops unconditionally. `_DEAD_FEK_JUDGED`/`_DEAD_FEK_INERT` mirror the same
split in `snapshot.py`. Production byte-identity: the sha probe reads `3cab191a…` unchanged, and
the full test reconciliation (snapshot purge tests, the d2 kernel re-pinned through
`pairwise_bench_outgoing`, delivery graph + arch tables + flag doc regenerated) is green.


## The MODEL_CONFIG_VERSION narrative (moved verbatim from `model_version.py`, 2026-08-16)

The per-version comment block that headed `MODEL_CONFIG_VERSION` since v3. History —
do not quote as current; the live machinery keeps only a pointer here.

```
# v3: added `vf_coef` — the PPO value-loss coefficient, recorded so a training resume
#   with a different `--vf-coef` is a hard error (changing the value head's gradient
#   scale mid-run is a silent training change). It is NOT weight-shape-relevant, so it
#   is deliberately EXCLUDED from check_compatible()'s universal load-check (which gates
#   frozen eval / self-play-pool / distill opponents too, where vf_coef is irrelevant);
#   it is enforced only on the training-resume path via check_vf_coef(). Old configs
#   migrate to the SB3 default 0.5 (= the value every pre-flag run was trained with).
#
# v4: added the reward-config hparams — `bias_additivity` (--bias-additivity, the per-run
#   BIAS additive↔telescoping knob), `mat_alive_weight` (--mat-alive-weight, the material-PBRS
#   per-mon-alive weight), and `bias_redesign` (--bias-redesign, the staged no-progress-clock +
#   reframe enable). Like vf_coef, these are resume-immutable VALUE-meaning hparams (changing them
#   mid-run silently shifts the reward) but NOT weight-shape — enforced only on the training-resume
#   path via check_reward_config(), excluded from check_compatible(). Old configs migrate to the
#   defaults (the single-variable run: 1.0 / 1.25 / False).
#
# v5: added `switch_bias_weight` (--switch-bias-weight, the belief-risk-scaled stay-into-KO BIAS lever
#   for the under-switch pathology; design_reward_switching.md §7). Same resume-immutable VALUE-meaning
#   treatment as the v4 reward hparams (folded into check_reward_config, excluded from
#   check_compatible). Old configs migrate to 0.0 (OFF = the lever absent, behavior unchanged).
#
# v6: added `use_popart` (PopArt value-target normalization toggle). Unlike the v3-v5 VALUE-meaning
#   hparams, PopArt changes the value head's STRUCTURE (normalized output + mu/sigma buffers), so it
#   is enforced in check_compatible() (gates EVERY load), not the resume-only path. Old configs
#   default False (no PopArt).
#
# v7: added `draw_penalty` (--draw-penalty, the terminal reward for a DRAW / 250-turn timeout). Same
#   resume-immutable VALUE-meaning treatment as the v4-v5 reward hparams (folded into
#   check_reward_config, excluded from check_compatible). Old configs migrate to -30.0 (== a decisive
#   loss = the prior behavior, where a tie scored -VICTORY_VALUE).
#
# v8: added `attend_unrevealed_opponents` (--attend-unrevealed-opponents). A BEHAVIORAL toggle that
#   keeps the opponent's still-hidden party attendable in the transformer instead of key-masking it.
#   Like v6/use_popart it changes the forward pass (the mask, policy AND value) rather than a reward
#   meaning, so it is enforced in check_compatible(); but unlike PopArt it leaves the state_dict
#   identical (no weight-shape / ARCH_SIGNATURE change). Old configs default False (baseline masking).
#
# v9: added `opp_belief_cls_k` (--opp-belief-cls-k). A STRUCTURAL toggle: k distinct learned query
#   tokens (HiddenOppBeliefPool) summarise the unrevealed opp party and feed both heads. 0 = off; k>0
#   changes the state_dict (adds the module + widens both projection Linears by k*D_MODEL). Like
#   v6/use_popart it is enforced in check_compatible() — but as a plain int every distinct value (incl.
#   0↔N) is a weight-shape mismatch, so a single unconditional compare gates it. OFF (k=0) reproduces the
#   baseline arch byte-for-byte → NO ARCH_SIGNATURE bump. k>0 requires attend_unrevealed_opponents
#   (enforced at extractor build). Old configs default to 0.
#
# v10: added `value_active_readout` (--value-active-readout). A STRUCTURAL toggle: route the active
#   mon's refined token into the VALUE projection (the dual-head readout drops it; a probe found the
#   critic predicts an incoming self-KO at AUC 0.79 vs the policy's 0.90). ON widens the value
#   projection by D_MODEL; like v6/use_popart it is enforced in check_compatible(). OFF reproduces the
#   baseline value head byte-for-byte → NO ARCH_SIGNATURE bump. Old configs default False.
#
# v11: added `value_tail_weight` (--value-tail-weight). A resume-immutable VALUE-meaning hparam (like
#   vf_coef, NOT weight-shape): the tail-weighted value-loss β (CVaR-blend of the worst value misses).
#   0.0 = plain MSE. Enforced ONLY on the training-resume path via check_value_tail_weight, EXCLUDED
#   from check_compatible (a frozen opponent's forward never runs the value loss). No ARCH_SIGNATURE
#   bump (network/obs unchanged). Old configs migrate to 0.0.
#
# v12: added `self_ko_hp_penalty` (--self-ko-hp-penalty). A resume-immutable VALUE-meaning reward
#   hparam (like draw_penalty): a decision-time-HP-scaled penalty (−w·hp) for self-KOing a mon via
#   Explosion/Self-Destruct. The symmetric material PBRS prices a healthy 1-for-1 trade at ~0, so the
#   critic learns to value a full-HP self-KO POSITIVELY and the policy throws away healthy mons; this
#   restores a negative signal (scaled by HP, so legitimate low-HP sac-for-KO is spared). 0.0 = OFF.
#   Enforced via check_reward_config, EXCLUDED from check_compatible. No ARCH_SIGNATURE bump. Old
#   configs migrate to 0.0.
#
# v13: added `drop_redundant_bias` + `drop_switch_bias` (--drop-redundant-bias / --drop-switch-bias).
#   Two resume-immutable VALUE-meaning bools (like the v4-v7 reward hparams): the de-bias cleanup that
#   ZEROES audit-flagged distorting BIAS terms. `drop_redundant_bias` removes stall_tax + matchup_penalty
#   (redundant with the no-progress clock + --draw-penalty / pbrs_belief); `drop_switch_bias` removes the
#   hand-coded switch-strategy subsidy (switch_base / switch_bouncing_tax / escape_threat_switch / se_switch
#   / pivot_* / sleep_in / sleep_out). Folded into check_reward_config, EXCLUDED from check_compatible. No
#   ARCH_SIGNATURE bump (reward-value only). Old configs migrate to False (== the prior behavior).
#
# v14: added `all_shaping_pbrs` (--all-shaping-pbrs, "everything but stall": folds Φ_hazard/Φ_boost/
#   Φ_opp_boosts + Φ_status and ZEROES every BIAS term EXCEPT the anti-stall tilt `no_progress_tax`, so
#   all non-stall shaping is policy-invariant; the bad turn-ramp `stall_tax` is zeroed) and made
#   `no_progress_penalty` resume-immutable (it is now Φ_progress's weight). Resume-immutable VALUE-meaning
#   (check_reward_config), EXCLUDED from check_compatible, NO ARCH_SIGNATURE bump. Old configs migrate to
#   all_shaping_pbrs=False / no_progress_penalty=0.15 (== the prior behavior).
#
# v15: added `stall_pbrs` (--stall-pbrs, the "stall" companion switch: folds Φ_progress and zeroes
#   `no_progress_tax`+`stall_tax`, so the anti-stall signal is policy-invariant too). Run --all-shaping-pbrs
#   WITH --stall-pbrs for a fully-PBRS reward (whole BIAS class zero); without it, keep the no_progress
#   stall tilt as the single acknowledged BIAS. Same resume-immutable VALUE-meaning treatment
#   (check_reward_config), EXCLUDED from check_compatible, NO ARCH_SIGNATURE bump. Old configs migrate to
#   stall_pbrs=False (== the prior behavior).
#
# v16: added `opp_belief_slots` (the hidden-opponent BELIEF-AUX arch toggle) + `opp_belief_aux_coef`
#   (its training-only loss weight). opp_belief_slots is STRUCTURAL like opp_belief_cls_k / use_popart:
#   ON fills the un-revealed opp team slots with distinct learned unknown-mon tokens (refined in-lineup
#   by the transformer) and builds a BeliefHead emitting species/moves aux logits — a state_dict change,
#   so it is gated in check_compatible() with a dedicated bool compare. Requires attend_unrevealed_opponents
#   (enforced at extractor build). OFF reproduces the baseline arch byte-for-byte → NO ARCH_SIGNATURE bump.
#   opp_belief_aux_coef is a TRAINING-ONLY coefficient (like ent_coef): it scales the aux loss, affects no
#   forward pass, so it is recorded for provenance but NOT version-locked (NOT in check_compatible / any
#   check_*; a resume may change it freely). Old configs migrate to opp_belief_slots=False / coef=0.0.
#
# v17: added `move_belief_mode` (the move-prediction REINJECTION arch toggle: off|revealed|unrevealed|both) +
#   `move_belief_coef` (its training-only loss weight). move_belief_mode is STRUCTURAL like opp_belief_slots:
#   any value != "off" builds a MoveBelief module that predicts each opp mon's moveset, soft-embeds it
#   (sigmoid(logits) @ move_embedding) and ADDS the projection back onto the opp token BEFORE the CLS pools
#   — so the predicted moves flow through to both heads. The mode selects which slots get enriched
#   (revealed = seen mons, unrevealed = believed slots, both). It changes the state_dict (a new Linear head +
#   reinjection projection + LayerNorm), so it is gated in check_compatible() with a string compare. Requires
#   attend_unrevealed_opponents (enforced at extractor build). OFF reproduces the baseline arch byte-for-byte
#   → NO ARCH_SIGNATURE bump. move_belief_coef is a TRAINING-ONLY coefficient (like opp_belief_aux_coef):
#   it scales the move-belief supervised loss, affects no forward pass, so it is recorded for provenance but
#   NOT version-locked. Old configs migrate to move_belief_mode="off" / move_belief_coef=0.0.
# v18: added `opp_belief_latent` (the LATENT-belief arch toggle) + `opp_belief_latent_coef` (its training-only
#   loss weight). opp_belief_latent is STRUCTURAL like opp_belief_slots: ON adds an asymmetric SimSiam
#   predictor to BeliefHead that maps each believed slot's refined token into the pokemon_encoder role-token
#   space, where a cosine loss regresses it toward the STOP-GRAD encoder role-token of the TRUE hidden mon
#   (graded identity supervision the hard species CE can't give). It changes the state_dict (the predictor
#   MLP), so it is gated in check_compatible() with a bool compare. Requires opp_belief_slots (the believed
#   slots + BeliefHead it attaches to). OFF reproduces the baseline arch byte-for-byte → NO ARCH_SIGNATURE
#   bump. opp_belief_latent_coef is a TRAINING-ONLY coefficient (like opp_belief_aux_coef): it scales the
#   latent cosine+VICReg loss, affects no forward pass, recorded for provenance but NOT version-locked. Old
#   configs migrate to opp_belief_latent=False / opp_belief_latent_coef=0.0.
#
# v19: added `damage_op` (the differentiable GPU damage operator arch toggle). STRUCTURAL like
#   value_active_readout / opp_belief_slots: ON builds a `DamageOperator` that consumes the move
#   belief's PREDICTED moves for the opp active and emits a per-our-mon believed-move incoming-damage
#   block appended to BOTH projection heads, so it WIDENS both projection inputs (a state_dict change)
#   — gated in check_compatible() with a dedicated bool compare. The operator's lookup tables are
#   non-persistent buffers (fixed physics from data/), so the only state_dict deltas are the wider
#   projections. OFF reproduces the baseline arch byte-for-byte → NO ARCH_SIGNATURE bump. Hard-requires
#   move_belief_mode in {revealed, both} (the op reads the opp-active's predicted logits, only
#   supervised for a revealed mon) — enforced at extractor build + the CLI. It is forward-only (no new
#   labels / no loss term), so there is no training-only coefficient. Old configs migrate to False.
#
# v20: added `move_prior_fusion` (the unified two-part move belief). FORWARD-BEHAVIOR toggle like
#   `attend_unrevealed_opponents` (NOT weight-shape): the MoveBelief head's output becomes a learned
#   log-odds DELTA fused with the Smogon move-frequency prior — `posterior = prior_logit(species) +
#   head_delta`, revealed moves pinned certain — so the stashed move-belief logits (read by the damage
#   op + the BCE loss) carry a proper POSTERIOR (priors ⊕ prediction unified). The prior buffer is a
#   non-persistent lookup, no new params → state_dict byte-identical either way, but the forward differs
#   when ON, so (like attend_unrevealed_opponents / damage_op) it is gated in check_compatible — a resume
#   that flips it would feed a different belief. Requires move_belief_mode != off (enforced at extractor
#   build + CLI). OFF reproduces the from-scratch head byte-for-byte → NO ARCH_SIGNATURE bump. Old configs
#   migrate to False.
#
# v21: added `mask_incoming_damage_obs` (the unified-architecture ABLATION toggle). FORWARD-BEHAVIOR
#   toggle like attend_unrevealed_opponents (NOT weight-shape): ON zeros the 51-dim incoming-damage /
#   OHKO obs block out of the model's view (the block STAYS in the obs at a fixed dim; the reward PBRS
#   still reads the belief from live_view). Lets the unified DamageOperator's learned belief->damage
#   REPLACE the CPU usage-prior collapse for the MODEL, A/B-ably, without deleting any code. State_dict
#   byte-identical (just zeros an obs slice), but the forward differs, so it is gated in check_compatible.
#   OFF = baseline byte-for-byte (NO ARCH_SIGNATURE bump). Old configs migrate to False.
#
# v22: added `win_prob_mode` (the tri-state auxiliary WIN-PROBABILITY head: none|read_only|shaping) +
#   `win_prob_coef` (its training-only loss weight). win_prob_mode is the STRUCTURAL toggle: 'none' = no
#   module (baseline byte-for-byte); 'read_only'/'shaping' build a `WinProbHead` (a side readout off
#   value_pooled, NOT in pi/vf so projection dims are unchanged — the only state_dict delta is the head's
#   own params). It is gated in check_compatible with a STRING compare so that BOTH 'none'↔head (a
#   state_dict change) AND read_only↔shaping (same params, but the user-chosen resume-IMMUTABLE mode — a
#   mid-run grad-flow flip is a silent training change) are FATAL on a resume mismatch. OFF reproduces the
#   baseline arch byte-for-byte → NO ARCH_SIGNATURE bump. win_prob_coef is a TRAINING-ONLY coefficient
#   (like opp_belief_aux_coef): it scales the BCE aux loss, affects no forward pass, so it is recorded for
#   provenance but NOT version-locked (a resume may change it freely, and a flagless resume inherits it).
#   Old configs migrate to win_prob_mode="none" / win_prob_coef=1.0.
# v23: added `damage_outgoing` (the OUTGOING per-move damage direction of the unified DamageOperator) +
#   `move_candidate_floor` (the learnset + rarity-cap move-prior gate). damage_outgoing is STRUCTURAL like
#   damage_op (the per-move outgoing block widens BOTH projection heads), gated in check_compatible with a
#   bool compare; OFF = baseline byte-for-byte (NO ARCH_SIGNATURE bump), requires damage_op. move_candidate_floor
#   is a FORWARD-BEHAVIOR float like move_prior_fusion: 0.0 = OFF (legacy 0.02-floor prior, byte-identical),
#   >0 enables the learnset-legality + <floor rarity prune on the move prior (a different belief → gated in
#   check_compatible; the prior buffer is non-persistent so the state_dict is identical either way). Old
#   configs migrate to damage_outgoing=False / move_candidate_floor=0.0.
# v24: gen3_unified_move_system_v1. Added `move_latent` (the context-free MoveLatentEncoder arch toggle:
#   a mechanics-grounded per-move latent concatenated into the move network — STRUCTURAL like damage_op,
#   it WIDENS the move-network input → state_dict change, gated in check_compatible; OFF = baseline
#   byte-for-byte, NO ARCH_SIGNATURE bump) + `move_belief_latent_coef` (its training-only latent-grading
#   loss weight: cosine of the predicted move distribution's expected latent toward the true moveset's mean
#   latent so Rock Slide ≈ Hidden Power Rock — NOT version-locked, like move_belief_coef). ALSO in v24 the
#   DamageOperator's effect block is enriched with per-status SECONDARY probabilities (incoming + per-move
#   outgoing, Serene Grace / Shield Dust) — intrinsic to `damage_op` (no separate flag), so a v23
#   damage_op checkpoint won't load into v24 (the op's output dim grew); damage_op OFF stays byte-identical.
#   Old configs migrate to move_latent=False / move_belief_latent_coef=0.0.
# v25: gen3_unified_spread_belief_v1 + the disable-redundant-obs master flag. (1) `spread_belief` (the THIRD
#   belief leg — predicts the opp's hidden SPREAD = 5 derived stats per slot, reinjected into the opp token,
#   consumed by the DamageOperator to REPLACE its hand-coded de-timid/neutral opp-spread constants; STRUCTURAL
#   like opp_belief_slots — adds the SpreadBelief module, gated in check_compatible, OFF byte-identical, NO
#   ARCH_SIGNATURE bump) + `spread_belief_coef` (its training-only speed-supervision loss weight, NOT
#   version-locked). (2) the disable-redundant obs masks `mask_active_move_scalars_obs` +
#   `mask_move_effects_obs` (FORWARD-BEHAVIOR like mask_incoming_damage_obs — zero a now-GPU-subsumed obs
#   region from the model's view; the master --unified-obs flips all three). (3) the DamageOperator op effects
#   are further unified (MOVE_EFFECT_FLAGS folded into MOVE_ATTR; fixed-damage moves type-gated) — intrinsic to
#   damage_op. Old configs migrate every new field to False/0.0.
# v26: gen3_unified_op_physics_v1 — the DamageOperator reaches PARITY with the CPU incoming_damage block it
#   (optionally) masks, so --unified-obs no longer regresses the model's damage understanding. INTRINSIC to
#   damage_op (no new field): the op now applies stat-stage BOOSTS (offense/defence/speed, both directions —
#   a +2 sweeper's Atk doubles), BURN (½ physical Atk), WEATHER (rain ×1.5 Water/×0.5 Fire; sun the reverse),
#   PARALYSIS (×0.25 speed), and FIXED-DAMAGE moves (Seismic Toss/Night Shade = level HP, type-immunity-gated
#   — 0 vs Ghost). Values-only (no dims/state_dict change → no new check_compatible field); the version bump
#   marks it. Counter/Mirror Coat (return-damage) is deferred.
# v27: gen3_unified_status_landing_v1 — the op's OUTGOING direction gains a per-OUR-move STATUS-LANDING block
#   (8 dims: P(a dedicated status move lands vs THIS opponent) + a `known` bit per move) — the GPU home for
#   the masked move-effect block's `status_will_land`, so --mask-move-effects-obs no longer drops that signal.
#   It folds accuracy × per-MOVE type immunity (Thunder Wave→Ground, Toxic/Poison→Steel/Poison, Will-O-Wisp
#   →Fire, **+ Leech Seed→Grass**, the v26-deferred item) × ability immunity (revealed→exact, else the Smogon
#   ability-prior marginal) × already-statused (majors) × **Sleep Clause** (a 2nd inflicted sleep fails; a
#   Rest self-sleep does NOT consume the cap) × **Substitute** (a Sub blocks every status move incl. Leech
#   Seed). The gen3 rules are imported from gen3_mechanics (one source); Shield Dust is N/A here (it only
#   scales SECONDARY effects, never a primary status move). INTRINSIC to damage_outgoing (no new field) — it
#   grows the outgoing output dim, so a v26 damage_outgoing checkpoint won't load (the SB3 load_state_dict
#   shape mismatch on the projection Linear in_features — the runtime-discovered projection dim is NOT a
#   ModelVersion field, so check_compatible passes). OFF (no damage_outgoing) byte-identical; no
#   ARCH_SIGNATURE bump. Bare version marker.
# v28: gen3_unified_choice_band_v1 — the op prices CHOICE BAND (×1.5 physical Atk + move-lock; the dominant
#   damage-relevant gen3 item). OUTGOING: our own CB (item known) ×1.5 our physical Atk DETERMINISTICALLY
#   (values-only). INCOMING: a CB-CONDITIONAL physical tail per our 6 mons — `phys_high_cb` (max-roll with
#   the ×1.5) + `pko_cb` (P(OHKO | CB)) — plus a shared `p_cb` scalar (P(opp active holds CB) from
#   `SPECIES_CB_PRIOR`, the Smogon item usage prior, collapsed to 1.0/0.0 once the held/consumed item is
#   revealed). DECORRELATED from the modal (no-CB) line so the head weights them — OHKO is a nonlinear
#   threshold a mean-field ×(1+0.5·p_cb) would blur (same provide-the-fact rationale as the crit-split). The
#   ×1.5 is applied at the Atk-STAT level (so core=k·A+2's +2 floor isn't boosted) in BOTH directions. The
#   move-lock + the ChoiceBandTracker's move-lock DISPROOF are a documented follow-up. INTRINSIC to damage_op
#   (the incoming CB block grows the incoming output dim → a v27 damage_op checkpoint won't load, SB3
#   load_state_dict in_features mismatch); OFF (no damage_op) byte-identical; no ARCH_SIGNATURE bump. Marker.
# v29: added the distributional VALUE head (Phase A interpretability side readout) — `value_dist_mode`
#   (none|read_only|shaping STRUCTURAL toggle, like win_prob_mode) + `value_dist_bins` (the atom count =
#   the head's output Linear width, weight-shape like opp_belief_cls_k), both gated in check_compatible;
#   and the value-meaning support `value_dist_vmin`/`value_dist_vmax` (resume-only check_value_dist, like
#   value_tail_weight). A SIDE readout off value_pooled (NOT in pi/vf → projection dims unchanged), so
#   OFF (mode none) is baseline byte-for-byte — NO ARCH_SIGNATURE bump. Old configs migrate to
#   value_dist_mode="none" / bins=0 / vmin=vmax=0.0. Design: designs/ai_v6/design_distributional_value_critic.md.
# v30: gen3_unified_topk_incoming_v1 — the DamageOperator's DISCRETE top-K incoming move-space block.
#   `damage_topk_k` (int, 0 = off) = the number of the opp ACTIVE's most-believed CANDIDATE moves surfaced
#   INDIVIDUALLY (vs the worst-case `_chan_max` collapse that loses WHICH move it is). Per top-K move: its
#   move LATENT identity (gathered from the MoveLatentEncoder — DIFFERENTIABLE → sharpens the latent) +
#   belief weight (DIFFERENTIABLE → sharpens the move belief) + accuracy + is_phys, then per OUR mon
#   [high-roll, P(KO), status_lands] — the discrete-move + per-pivot (incl. damage-immunity 0 AND
#   status-immunity 0, e.g. Thunder-Wave→Ground) read that makes "anticipate the move / pick the safe
#   switch" decidable. Added ALONGSIDE the worst-case summary. K scales out_dim (hence both projection
#   in_features) → STRUCTURAL int gated in check_compatible (like opp_belief_cls_k); OFF (0) byte-for-byte
#   (NO ARCH_SIGNATURE bump). Requires damage_op + move_latent. A v29 damage_op checkpoint won't load into a
#   topk-ON op (projection in_features mismatch). Design: designs/ai_v6/design_topk_incoming_moves.md.
# v31: added `damage_reattend` (gen3_damage_reattend_v1) — re-attend the team tokens to the computed
#   DamageOperator physics, then re-derive the CLS pools, so the policy/value DECISION path (incl. the
#   switch logits) reads damage-contextualised summaries (today the damage block is a post-pool concat that
#   no attention sees). STRUCTURAL toggle like opp_belief_slots (adds a damage→token projection + LayerNorm
#   + one TransformerEncoder layer; re-pooling preserves the pooled shapes ⇒ projection WIDTHS unchanged),
#   gated in check_compatible (bool); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op.
# v32: added `move_belief_prefuse` (gen3_move_prefuse_v1) — move the MoveBelief reinjection from
#   POST-transformer to PRE-transformer, so the predicted opp moves co-refine with the species/team belief
#   through the 2 attention layers instead of being grafted on afterwards. FORWARD-BEHAVIOR toggle like
#   move_prior_fusion (same MoveBelief params → state_dict identical; only the call timing differs), gated
#   in check_compatible (bool); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires move_belief_mode != off.
# v33: gen3_iterative_damage_v1 — ITERATIVE damage refinement. `damage_refine_rounds` (int, 0 = off) is the
#   number of transformer layers (capped by TRANSFORMER_N_LAYERS in effect) before which the DamageOperator's
#   LEAN discrete incoming threat is recomputed from the CURRENT (being-enriched) opp tokens and injected back
#   onto our-mon tokens via a `refine_proj` Linear (zero-init → identity-at-init) — so each layer attends over
#   physics derived from the freshest move belief (physics-in-the-loop), and the per-round read sharpens the
#   move-belief head. STRUCTURAL: 0 builds no module (baseline forward byte-for-byte, NO ARCH_SIGNATURE bump);
#   N>0 builds refine_proj (its SHAPE is N-independent — weight-tied across rounds) and changes the forward, so
#   EVERY distinct value (0↔N a state_dict change; N↔M a forward-behavior change) is gated in check_compatible
#   with an unconditional int compare (like opp_belief_cls_k). Requires damage_op (→ the op physics + a
#   move_belief to re-read). Old configs migrate to 0. Design: designs/ai_v6/design_iterative_damage_refinement.md.
# v34: gen3_per_move_matrices_v1 — the OUTGOING per-move DAMAGE MATRIX. `damage_matrices_outgoing` (bool, off)
#   makes the DamageOperator ALSO emit our 4 moves × the opp's 6 mons (active + REVEALED bench) — per (move,
#   opp mon) [low,high,crit,pko,type_mult] + a per-opp-mon revealed bit — the bench extension of the single-
#   active outgoing block (price a KO on a switch-in). Unrevealed opp slots zeroed (belief-driven = TODO).
#   STRUCTURAL toggle like damage_op (widens both projection in_features); gated in check_compatible (bool);
#   OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op. Design: designs/ai_v6/design_per_move_damage_matrices.md.
# v35: gen3_per_move_matrices_v1 — the INCOMING per-move DAMAGE MATRIX. `damage_matrices_incoming` (bool, off)
#   makes the DamageOperator emit the ENRICHED top-K block: per opp-active move a header [latent, belief, acc,
#   is_phys, EXPLICIT effect bits(6), secondary chances(10)] + per (OUR mon, move) cell [low,high,crit,pko,
#   type_mult,status_lands] — the un-collapsed evolution of the v30 top-K + the deleted p_effect/p_sec maxes.
#   REUSES damage_topk_k as its K (one knob, try 4/5/6). Since gen3_op_block_trim_v1 deleted the lean top-K
#   block it superseded, this is the ONLY block K sizes; requires damage_op + move_latent.
#   STRUCTURAL toggle like damage_op (widens both projections via the op out_dim); gated in check_compatible
#   (bool); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Design: designs/ai_v6/design_per_move_damage_matrices.md.
# v36: gen3_bidir_threat_trunk_v1 — the BIDIRECTIONAL in-trunk threat field. `threat_refine_outgoing` (bool)
#   adds a zero-init `outgoing_proj` that injects a per-opp-mon OUTGOING-threat residual onto the OPP tokens
#   via the SAME between-layers refine loop (symmetric to the incoming refine; STRUCTURAL — a saved weight).
#   `threat_unrevealed_outgoing` (bool) prices that residual's UNREVEALED columns via the EXPECTED-LATENT
#   defender — marginalize the move-belief's P(species) through SPECIES_EXP_MULT (type chart × expected
#   ability immunity) + SPECIES_SPREAD_PRIOR (E[bulk]), with P(KO) NULLED (forward toggle, no new params).
#   `threat_prob_outspeed` (bool) makes P(outspeed) UNCERTAINTY-AWARE (÷ believed speed std, not a fixed
#   scale; forward toggle). All three OFF byte-for-byte (NO ARCH_SIGNATURE bump). threat_refine_outgoing
#   requires damage_op + damage_refine_rounds>0; threat_unrevealed_outgoing requires threat_refine_outgoing
#   (+ a belief head for P(species)). Design: designs/ai_v6/design_bidirectional_threat_trunk.md.
# v37: gen3_status_trunk_v1 — STATUS-LANDING into the trunk (the last CPU-obs deprecation gap).
#   `threat_status_refine` (bool) adds two zero-init Linears riding the refine loop: status_in_proj (incoming
#   "will I be statused" onto OUR tokens, from the opp active's believed status moves) + status_out_proj
#   (outgoing "can I status this opp mon" onto OPP tokens, revealed-gated, from our active's status moves),
#   each a per-defender [P(major), P(immobilize=para/frz/slp)] computed by reusing the v27 status-landing
#   physics (type × ability × already-statused × Sleep-Clause × Substitute). Status immunity is a computed
#   MECHANICS fact (the same class as type effectiveness) — handed over, not learned across non-local tokens.
#   STRUCTURAL (saved weights); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op +
#   damage_refine_rounds>0. Completes the FULL --unified-obs deprecation (the A/B is the arbiter). Design:
#   designs/ai_v6/design_bidirectional_threat_trunk.md.
# v39: gen3_per_move_matrices_v1 — the TRANSPOSED outgoing matrix. `damage_matrices_outgoing_all` (bool, off)
#   makes the DamageOperator ALSO emit our 6 MONS' 4 moves → the opp ACTIVE — per (attacker mon, move)
#   [low,high,crit,pko] + a per-attacker p_outspeed + an alive bit. The TRANSPOSE of v34's
#   damage_matrices_outgoing (our active's 4 moves × the opp's 6 mons): here the ATTACKER axis is our 6 mons,
#   the defender is the opp ACTIVE only. On a FORCED SWITCH our active is fainted → the single-active outgoing
#   block zeroes, so the policy picks switch-ins BLIND to offense; this prices every candidate switch-in. The
#   ACTIVE row reproduces _outgoing_block byte-for-byte (parity); bench rows reuse the SAME _rolls physics with
#   NEUTRAL boosts (gen3 resets on switch). STRUCTURAL toggle like damage_op (widens both projection
#   in_features via the op out_dim); gated in check_compatible (bool); OFF byte-for-byte (NO ARCH_SIGNATURE
#   bump). Requires damage_op. Design: designs/ai_v6/design_per_move_damage_matrices.md.
# v43: gen3_pubval_aux_v1 — the PUBLIC-information value aux head. `pubval_mode` (none|read_only|shaping,
#   the win_prob_mode pattern) builds a PubValHead off value_pooled, regressed toward the FROZEN
#   human-replay-calibrated public value V_pub (agents.training.pubval + data/gen3_pubval.json — 164k rated
#   gen3ou games, the value-INDEPENDENT exogenous signal; dense per-step, so the trunk sees WHEN the game
#   swung). SIDE readout (never in pi/vf, never in GAE); the target rides a training-only `pubval_target`
#   obs key computed env-side from PUBLIC state only. STRUCTURAL + resume-IMMUTABLE STRING gate like
#   win_prob_mode ('none'↔head = state_dict; read_only↔shaping = grad-flow); OFF byte-for-byte (NO
#   ARCH_SIGNATURE bump). `pubval_coef` training-only. Design: designs/ai_v8/design_public_info_value.md.
# v44: gen3_zarch_film_v1 — the team-archetype latent z_arch + head FiLM (the amortization-gap STORAGE
#   fix: per-team gradients modulate different rank-z subspaces instead of cancelling in the shared
#   heads — designs/learning/amortization_gap_and_conditioning.md). `zarch_film` (off|heads) builds a
#   ZArchEncoder (a TEAM-STATIC, permutation-invariant DeepSets code over OUR team's INVARIANT facts:
#   species ⊕ item ⊕ ability ⊕ moves ⊕ spread, detached embedding reads — zero trunk interference) +
#   two ZERO-INIT FiLM generators applied post-projection pre-ReLU per root head (identity-at-init ⇒
#   ON starts byte-identical). `zarch_dim` (int) is the latent width = the FiLM conditioning rank —
#   the generators' in_features, so every distinct value is a weight-shape mismatch (unconditional int
#   compare, the value_dist_bins pattern). STRING + INT gated in check_compatible; OFF (off/0) builds
#   no modules = baseline byte-for-byte (NO ARCH_SIGNATURE bump). `zarch_recon_coef` (species multi-hot
#   reconstruction BCE — the anti-collapse anchor) + `zarch_vicreg_coef` (per-dim variance floor) are
#   TRAINING-ONLY loss coefs (recorded for provenance + flagless-resume read-back, NOT version-locked).
# v46: gen3_zarch_lut_v1 — the per-team LUT on top of z_arch. `zarch_lut` (off|add|only) adds an
#   Embedding[n_teams+1, zarch_dim] (row 0 = unknown, ZERO-init; rows 1..N random-init so the per-team
#   codes are LARGE and ~orthogonal from step 0) + a LayerNorm, and the team is resolved from the
#   OBSERVATION by a sorted species(6) ⊕ moves(24) signature (agents.model.team_signature) — so NO
#   env/eval/prober/frozen-opponent plumbing changes. `zarch_lut_teams` (int) is the table height, a
#   weight-shape field (unconditional int compare, the zarch_dim pattern). It exists to test whether
#   the measured multi-team exploiter ceiling (N=1/3/10 distil cleanly, N=20 stalls) is a
#   conditioning-SIGNAL limit: the DeepSets z is COMPOSITIONAL, so z-similar teams sit at z̄ + a tiny
#   ε and the FiLM generator's gradient is proportional to that residual (ill-conditioned); a free
#   code removes exactly that limit. 'add' = LN(z_deepsets + code) keeps composition (an unmatched
#   team hits the zero row ⇒ exactly the DeepSets z); 'only' = LN(code), the sharpest ablation.
#   STRING + INT gated in check_compatible; OFF byte-for-byte (NO ARCH_SIGNATURE bump).
# v47: added `move_belief_single_compute` (gen3_belief_single_compute_v1) — compute the move belief
#   EXACTLY ONCE per forward (pre-attention) and FREEZE it. Under prefuse the belief is predicted +
#   reinjected before the transformer, but the between-layers refine callback then RE-READ move_logits
#   off the reinjected tokens: the belief was computed twice and the physics consumed a different
#   posterior than the one attention was handed. ON, the refine kernels reuse the stashed
#   pre-transformer logits ⇒ belief ONCE → physics ONCE → N attention layers that CANNOT revise it
#   (the frozen-belief arm of the iterative-refinement A/B; also one fewer head pass per forward).
#   FORWARD-BEHAVIOR toggle like move_belief_prefuse (same MoveBelief params → state_dict identical;
#   only which posterior the refine kernels read differs), gated in check_compatible (bool); OFF
#   byte-for-byte (NO ARCH_SIGNATURE bump). Requires move_belief_prefuse.
# v49: added `damage_candidate_k` (gen3_topk_candidates_v1) + `pointer_head` (gen3_pointer_head_v1).
#   `damage_candidate_k` (int, 0 = the full ~400-wide sweep) caps the DamageOperator's INCOMING
#   candidate axis at the K most-believed opponent moves, NO tail bound — the truncated mass is
#   dropped. FORWARD-BEHAVIOR (no new params; the per-candidate args just get narrower), gated with an
#   unconditional int compare like `damage_topk_k`; 0 is byte-identical.
#   `pointer_head` (bool) was the DELTA pointer head — a zero-init additive term on the flat head's
#   logits. REMOVED at v51: the pointer head became THE head (see below).
# v50: added `damage_op_prefuse` (gen3_damage_op_prefuse_v1) — ONE damage computation per forward,
#   PRE-attention. The op ran TWICE: a LEAN `discrete_*` recompute inside the between-layers refine
#   loop (×`damage_refine_rounds`) plus the FULL 835-dim block after the transformer. At B=1 on CPU —
#   the PFSP frozen-opponent regime that sits on the rollout critical path — the two together are ~75%
#   of a dispatch-bound 6.45 ms forward (the attention layers themselves are 0.27 ms). ON, the spread +
#   HP-type beliefs and the FULL op all run on the PRE-transformer role tokens, the per-OUR-mon incoming
#   rows are injected onto our tokens via a zero-init `prefuse_proj` (the refine_proj convention), and
#   the SAME full block is concatenated to both heads — so the P1 head-concat dependency is preserved at
#   full width, just sourced from a pre-attention belief. Mutually exclusive with damage_refine_rounds>0
#   (the loop is what it replaces); requires damage_op + move_belief_prefuse. The justification is CPU
#   cost; the "attention reasons over full-fidelity physics" story is secondary and, per K9/K10/K10a,
#   unlikely to pay on its own. STRUCTURAL (adds prefuse_proj) → bool compare in check_compatible; OFF
#   byte-identical (NO ARCH_SIGNATURE bump).
# v51: gen3_pointer_native_v1 — the FRESH-GENERATION pointer-native action head. The flat positional
#   `action_net` is DELETED (replaced in Gen3DualHeadMaskablePolicy._build by a raising stub) and the
#   `PointerNativeActionHead` is THE action head, unconditionally: move logit k from the REQUEST-slot-k
#   move token ⊕ its op cells [low,high,crit,pko,p_land,known,sec×10], switch logit j from our-team
#   token j ⊕ its incoming row + CB tail + OAX attacker row, struggle from the latent_pi context —
#   position-EQUIVARIANT (one shared scorer per entity; the sorted-vs-request ordering bug class is
#   unrepresentable at the logits). The v49 `pointer_head` FIELD is removed (POPped in
#   _migrate_config); no gate exists because there is no off state. Cross-era break carried by the
#   ARCH_SIGNATURE bump (see gen3_pointer_native_v1 below).
# v52: gen3_typed_hp_belief_v1 — the v38 tri-state `hp_type_belief_mode` FIELD is DELETED (POPped in
#   _migrate_config): the HP-type head is UNCONDITIONAL under a move belief, the presence×type
#   composition moved into `HPTypeBelief.compose_typed_hp` beside the move head, and every consumer
#   reads a posterior that carries HP at the 16 real typed nums 355-370 with the bare 237 hard-off.
#   Forward math changed with unchanged projection widths → the cross-era break rides the
#   ARCH_SIGNATURE bump (see gen3_typed_hp_belief_v1 below). `hp_type_belief_coef` stays training-only.
# v53: added `hp_belief_mode` (gen3_hp_belief_ablation_v1) — 'composed' (default, the v52 forward
#   byte-for-byte: HPTypeBelief + the presence×type factorisation + the two certain-fact eliminations)
#   vs 'flat' (the ABLATION: no HPTypeBelief head; the multi-label move head predicts the 16 typed
#   channels INDEPENDENTLY off their own per-typed Smogon priors — both arms still mask the bare 237
#   via the shared `mask_typeless_hp`). STRUCTURAL ('flat' drops a module) → STRING compare in
#   check_compatible (the win_prob_mode pattern); default byte-identical (NO ARCH_SIGNATURE bump).
# v54: gen3_entity_move_seats_v1 — Stage 1 of the entity generation (the roadmap's move-tokens-into-
#   the-body slice): MOVE tokens become first-class attention SEATS in the unified trunk, appended
#   after the global token. E3 (unconditional): our active's 4 request-ordered move tokens, projected
#   32 → d_model — the pointer head now reads the REFINED seats (post-attention, d_model-wide;
#   `move_seat_proj` + the token-type table growing 4 → 6 + the head's wider `move_proj` are the
#   unconditional state_dict changes → the `ARCH_SIGNATURE` bump below carries the break). E4
#   (`entity_topk_seats` int, 0 = off): the opp active's top-K believed threat-move seats — the op's
#   `refine_candidates(k=K)` candidate definition (belief-weighted, typed-HP-scattered) gathered as
#   `[latent ⊕ w ⊕ acc ⊕ is_phys]` per seat; adds `threat_seat_proj` (STRUCTURAL int, the
#   `damage_topk_k` gating pattern; requires damage_op_prefuse + move_latent). NO edges yet (Stage 2).
# v55: gen3_op_block_trim_v1 — NO new field; the DamageOperator's output SHRINKS by 28 dims and its
#   `damage_topk_k` knob changes meaning, so the version marks the break (the ARCH_SIGNATURE bump below
#   is what actually rejects an older checkpoint). Deleted, on the ledger-P1 per-block dependence
#   ablation: the opp-active-level believed-EFFECT scalars (6 dims, 1.2% of the zero-whole-op ceiling),
#   the opp-active per-STATUS incoming SECONDARY scalars (10 dims, 0.1% — the single most INERT channel
#   in the operator), and the OUTGOING per-move slp/psn/tox secondary columns (12 dims = 4 moves × 3,
#   structural zeros: gen3 has NO damaging move that inflicts sleep, and the psn/tox carriers appear on
#   1 / 0 of the 773 pool teams). Also deleted: `_topk_block`, the v30 LEAN top-K block — a strict
#   subset of the v35 `_incoming_matrix` that already suppressed it, which the same cProfile measured at
#   **0 calls per forward** in the production build. `damage_topk_k` now means "the incoming matrix's
#   K"; K>0 without `damage_matrices_incoming` is a hard error (never a silent empty block).
#   INDEPENDENT of v54's entity seats — the two touch disjoint machinery (seats enter the TRUNK; this
#   trims the op's HEAD-CONCAT output), so they compose and only the signature is shared.
# v56: gen3_edge_bias_trunk_v1 — Stage 2 of the entity generation (physics as attention EDGES):
#   the trunk's encoder stack becomes the BIASED clone (`BiasedEncoderLayer` — same math, but
#   attention takes an additive per-pair per-head float bias; the key-pad mask rides the same
#   tensor), an UNCONDITIONAL state_dict change (layer keys `in_proj.*` vs `self_attn.in_proj_*`)
#   → the ARCH_SIGNATURE bump below carries it. `edge_bias_families` (str, "off" default) gates the
#   FAMILIES delivered — each through a ZERO-INIT Linear(cell → 2·n_heads) map (identity at init):
#   D1 our-move→opp-mon (the v34 outgoing-matrix kernel), D2 our-mon→opp-ACTIVE (the v39 switch-in
#   kernel, move-collapsed, one-hot column), D3 threat-seat→our-mon (the pre-collapse incoming
#   kernel at the E4 candidate selection), S1 our-status-move→opp-mon + S3 threat-seat→our-mon
#   (the v27/v37 status-landing kernels' per_pair branches). "d" is the FROZEN d1,d3 alias (a saved
#   config never silently grows maps); new families are explicit comma-list only. Growing the VALID
#   family set is not a version bump — the string gate catches any mismatch. The op head-concat is
#   NOT deleted (deprecation playbook: home first, ablation audit before deletion).
# v57: added `entity_tail_seats` (gen3_entity_tail_seats_v1, E5) — 6 per-opp-mon TAIL-THREAT seats
#   summarizing the beyond-top-K belief mass every candidate consumer truncates ([p_tail, worst_phys,
#   worst_spec, revealed] → tail_proj + a learned tail_marker; NO new token-type row, deliberately —
#   growing the type table would break loading in-generation checkpoints into newer code). STRUCTURAL
#   bool (adds tail_proj + tail_marker + 6 seats per forward); OFF byte-identical; requires
#   damage_op_prefuse + entity_topk_seats>0 (the tail is defined relative to the E4 truncation).
# v58 is a STAMP (no field, no migration — the v26/v55 convention): the SpD-as-speed GIGO fix
# in pairwise_speed/pairwise_boost (the V/C1 kernels read stat index 4 = Special Defense as
# "speed"; both trained generations' V edge priced bulk). VALUES-only forward-math change: a
# pre-v58 checkpoint still LOADS, but its v_map/c1_map trained against the buggy feature — its
# V-edge inputs shift under fixed code (documented, accepted: gen-3 retrains under true physics).
# v60 is the gen3_entity_rehome_v1 STAMP (Stage-3 re-home; the ARCH_SIGNATURE carries the break —
# obs dim, POKEMON_FULL_DIM and the move/role net widths all move, so no migration is possible).
# v61 is the gen3_no_concat_v1 STAMP — the op head-concat deletion + the multi-seed critic
# readout (the gen-5 world; the signature carries the break).
# v62: added `value_seed_vicreg_coef` (gen3_seed_vicreg_v1) — the VICReg variance+covariance floor on
#   the MultiSeedValueReadout seed OUTPUTS (agents/model/seed_vicreg.py), built because the
#   pre-registered trigger in seed_diagnostics.py FIRED on gen-5 (seeds/out_effective_rank 1.0
#   sustained 13M+ steps — full seed collapse). Resume-immutable VALUE-meaning hparam (the
#   vf_coef class, NOT weight-shape): enforced only on the training-resume path via
#   check_value_seed_vicreg; excluded from check_compatible/_WEIGHT_FIELDS (a frozen opponent's
#   forward never touches it). 0.0 = OFF (loss byte-identical).
# v65: gen3_unconditional_move_legality_v1 — move-belief LEGALITY is now UNCONDITIONAL. A move a
#   species physically CANNOT LEARN always carries ~zero prior mass; there is no flag and no opt-out,
#   because it is a correctness property rather than a feature. `learnset_gate` is DELETED from
#   `damage_tables.build_move_prior_logits`, and `move_candidate_floor` (which used to double as the
#   on/off switch via `floor > 0.0`) is demoted to what its name says: the LEGAL-BUT-UNOBSERVED base
#   probability, default 0.02 (was 0.0). STAMP-ONLY migration — no new field. Nothing can be toggled,
#   so there is nothing to record; the version stamp exists to say "a pre-v65 checkpoint was trained
#   on a prior that gave phantom mass to unlearnable moves". Pre-v65 configs recorded
#   move_candidate_floor=0.0, which check_compatible now rejects against the 0.02 default — deliberate,
#   loud, and NOT migrated up (rewriting 0.0→0.02 would let an incompatible belief load silently).
#   NO ARCH_SIGNATURE bump: the prior buffer is non-persistent and unchanged in shape, and every
#   floor > 0 config produces a bit-identical buffer before and after (only floor == 0.0 changes).
# v67 is the gen3_deadline_clock_v1 STAMP — the obs CLOCK group goes 1 → 3 scalars (log-elapsed +
#   remaining-linear + log-remaining), so GLOBAL_ENV_DIM 18 → 20 and the obs 2667 → 2669, and the
#   move/global context widths that read `_gl['clock']['dim']` move with it. Obs width + weight
#   shapes change together, so no migration is possible — the ARCH_SIGNATURE carries the break.
#   Motivation (measured on 14/14 timeout losses at ai_v9_09 step 16M): the critic reported a
#   POSITIVE V on the last decision before a −30 forfeit in 13 of 14 games (mean +9.33, mean
#   terminal TD surprise −39.3) and was RISING into the forfeit in 10 of 14. The single
#   log-ELAPSED scalar gave the last 20 turns 1.5% of its range; log-REMAINING gives them 55.1%
#   (37×), which is the link TD must fit FIRST before it can bootstrap value back down a
#   200-turn episode.
# v70: gen3_refine_loop_removed_v1 — the between-layers refine loop is DELETED, and with it FIVE
#   fields: `damage_refine_rounds`, `threat_refine_outgoing`, `threat_unrevealed_outgoing`,
#   `threat_status_refine`, `move_belief_single_compute`. The loop was 0 rounds in production, and
#   0 rounds is exactly what made the three `threat_*` flags UNREACHABLE — each hard-requires
#   damage_refine_rounds>0, which is itself mutually exclusive with `damage_op_prefuse` (the
#   production placement), so setting any of them RAISED at extractor build.
#   `move_belief_single_compute` only chose which posterior the refine callback re-read; with no
#   callback it was INERT (production recorded it True and it did nothing).
#   The expected-latent OUTGOING math is NOT deleted — it was re-homed onto the live outgoing kernel
#   at a usage prior (gen3_unrevealed_outgoing_prior_v1) and runs unconditionally there.
#   Forward BIT-IDENTICAL on the production config; NO ARCH_SIGNATURE bump (no module built, no
#   weight shape moved, no forward value changed). Stale keys are POPped by the v70 migration.
# v71: gen3_tiered_pipeline_v1 — the TIER ORDER becomes the ONLY order. `move_belief_prefuse` and
#   `damage_op_prefuse` selected between a PRE- and a POST-transformer placement for the move belief
#   and for the spread/HP-type + DamageOperator group; production ran PRE for both. The POST call
#   sites are DELETED and the PRE placement is unconditional, so the two flags no longer select
#   anything and are removed. `damage_reattend` (v31) goes with them: it re-attended the physics onto
#   the team tokens AFTER the pools, which is a compensation for computing the physics post-attention
#   — the thing this step removes — and it was off in production.
#   `prefuse_proj` is now built whenever `damage_op` is on (it was gated on `damage_op_prefuse`,
#   which required `damage_op`), so the production state_dict is UNCHANGED.
#   Forward BIT-IDENTICAL on the production config; NO ARCH_SIGNATURE bump.
#   ⚠️ This BREAKS every non-prefuse config BY DESIGN, and the v71 migration REFUSES them with a
#   clear error rather than popping the key — `move_belief_prefuse` changed no weight shape, so a
#   silent pop would load a post-ordering checkpoint into a pre-ordering forward with nothing
#   downstream able to notice.
# v75: the SimSiam LATENT belief is DELETED — `opp_belief_latent` and `opp_belief_latent_coef` go,
#   along with the predictor MLP, the `belief_target_slots` training-only obs key that fed it, and
#   the env work that built that key every decision. It was a side readout: the latent never
#   entered pi or vf, so removing it changes no forward value on any config that had it off.
#   It cost ~13% of the train step (marginal +341 ms at the production batch, measured against a
#   +349 ms `opp_belief_cls_k=6` that DOES feed both projections), and its own role-geometry probe
#   concluded decodable != helps. Predicting the opponent's unrevealed mons is untouched: the
#   species CE, the moves BCE and the T0 species prior all remain.
#   ⚠️ The v75 migration REFUSED a config that recorded `opp_belief_latent=True` — unlike v71's
#   forward-only flags, this one carried PARAMETERS, so such a checkpoint's state_dict holds keys
#   the live extractor has no home for. (That branch is now below the v76 floor — the blanket
#   pre-generation refusal subsumes it; the zip-kwargs sanitizer keeps the per-field judgment.)
# v76 (gen3_ctx_dedup_v1): the per-side ENCODED active contexts are DELETED from both projection
#   heads — `ProjectionAssembler.active_ctx_encoder` no longer exists and both projection input
#   widths shrink by 2·32. The content was duplicated delivery with a 1:1 entity-native
#   replacement already live (the E2 injection scatters each side's FULL raw ctx block onto its
#   active token; the global token is a second route). The `active_ctx_hidden` ModelVersion
#   field goes with the module (no migration branch needed: the floor rises to 76 with the
#   signature, so no pre-v76 config is ever migrated). state_dict changes → the signature
#   carries the break; fresh lineage.
# v77 STRUCTURAL (gen3_intent_move_cell_v1, the G3 gate of design_conditional_execution.md):
#   `intent_move_cell` — the POLICY-side alpha consumer. The c2 status-consequence family is
#   re-delivered through the pointer MOVE cell as a per-action ABSOLUTE, alpha-conditioned: the
#   burn/sleep consequence channels become UNRENORMALIZED alpha-expectations over the op's own
#   top-K seat candidates (`f(m, SWITCH)=0` is exact — a switching active neither attacks nor
#   receives the status), the k-independent c2 columns ride raw vs the opp ACTIVE with the seat
#   mass `alpha_stay` as a decorrelated channel. ON widens the pointer move scorer's in_features
#   by INTENT_MOVE_CELL_DIM through a zero-init projection (identity at init, M1-guarded); OFF
#   builds no module and is byte-identical → NO ARCH_SIGNATURE bump. Requires opp_intent +
#   damage_op (+ damage_topk_k>0 at runtime, fail-loud). Old configs migrate to False.
# v78: gen3_flag_surface_p1_v1 — the TIER-1 flag-surface cleanup. EIGHT fields are DELETED with the
#   modules behind them; nothing is added.
#   (1) The ZARCH family — `zarch_film`, `zarch_dim`, `zarch_lut`, `zarch_lut_teams`,
#       `zarch_recon_coef`, `zarch_vicreg_coef` — goes with `ZArchEncoder`, the two FiLM generators,
#       the per-team LUT Embedding + its `team_signature` roster table, and `attach_zarch_lut`. The
#       line it existed to test is CLOSED and the result was NULL twice over: the LUT arm — a FREE
#       per-team code, i.e. the sharpest possible removal of the conditioning-signal limit — moved the
#       N=20 multi-team ceiling by +0.024 with CI [-0.016, +0.064], and the orthogonal 2x2 measured
#       COUNT (N 20->10, +0.077 SIG) dominating CONDITIONING (+0.027 n.s.). Every gen-8/9/10 run
#       recorded it OFF, so deleting it changes no production forward.
#   (2) The SEED-PRESSURE pair — `seed_quantile` (v63) + `value_seed_vicreg_coef` (v62) — goes with
#       `seed_quantile.py` and `seed_vicreg.py`. BOTH cap at ~1-D differentiation of the k=4 value
#       seeds and the two measurements meet in the middle: gen-6's VICReg satisfied every term with
#       out_effective_rank 1.05 (three seeds identical, one breakaway), and gen-7's quantile arm drove
#       crossing_rate to 0.000 with out_effective_rank 1.157 of 4. A SHARED readout can only constrain
#       each seed along its own weight vector, so no coefficient reaches the orthogonal directions —
#       multiplicity is not the missing axis. `seed_diagnostics.py` (the MEASUREMENT) stays.
#   MIGRATION: POP for a config that recorded them OFF; REFUSE one that recorded zarch_film != 'off'
#   or seed_quantile=True, on the v75 principle — those carried PARAMETERS the live extractor has no
#   home for, so a silent pop would load a state_dict with keys nothing can place.
#   `value_seed_vicreg_coef` and the two zarch coefs are training-only, so any value pops silently.
#   NO ARCH_SIGNATURE bump and the MIGRATION FLOOR stays 76: every deleted module was OFF in
#   production, so the production forward AND state_dict are bit-identical across this change
#   (verified: same state_dict keys, max|delta| 0.0 on pi/vf under designs/production_config.json).
#   Also in v78, with no field consequence: `--use-showdown-bridge` (the deprecated `--use-bridge=node`
#   alias) is deleted and `--use-bridge` now DEFAULTS to `rust`; and three settled toggles are DEMOTED
#   to the config_only tier (`attend_unrevealed_opponents` frozen ON, `value_active_readout` and
#   `damage_matrices_outgoing_all` frozen OFF) — their FIELDS and check_compatible gates are
#   deliberately UNCHANGED, because a demotion removes the SELECT role only. See
#   `agents.model.flag_registry` and designs/flag_registry.md.
# v79 (gen3_pair_history_v1, Tier H-A of design_history_entity.md): the COMPILED history tier.
#   Obs 2669 → 2921: per-mon LAST-ACTION fields (POKEMON_FULL_DIM 116 → 122 — the embedded
#   last-move id is manifest-routed, its raw column zeroed at the slice) + the 180-dim
#   pair-history block h[i,j] (6×6×5 tendency counters, EpisodeTracker-folded from PUBLIC
#   events, log-saturated). New edge family "h" (obs-fed, zero-init, mon×mon) joins the
#   edge_bias_families vocabulary — NOT in the production string, so the family is opt-in;
#   the obs widening is unconditional (retrain-class). No new ModelVersion field and NO
#   ARCH_SIGNATURE bump (the recency precedent): total_dim + the widened role-encoder shapes
#   are weight-field-caught, and the family rides the recorded edge_bias_families string.
# v80 (gen3_unified_value_readout_v1, Stage-3 T3-DELIVER of design_unified_belief.md §3): the
#   critic's UNIFIED ENTITY POOL — `value_entity_pool`, opt-in. K learned queries attention-pool
#   the critic's entity-row set (the 12 post-transformer team tokens + the op's 6 per-our-mon
#   incoming rows, per-source type embeddings, explicit NaN-safe softmax) through a ZERO-INIT
#   output projection appended to vf ONLY (the intent_value_reduce placement: pi untouched at
#   any weight). The designed SUCCESSOR contract of the bolt-on vf routes (seed readout /
#   threat-inject) the gen-11 critic_route_audit adjudicates — built so a condemned route has a
#   replacement the next generation can enable in the same config. OFF builds nothing
#   (byte-identical baseline; no ARCH_SIGNATURE bump); ON widens the value projection
#   (weight-field-caught).
# v81 (gen3_event_window_v1, Tier H-B of design_history_entity.md): the EVENT-TOKEN history
#   window. Obs 2921 → 3529: a 32×19 typed event-record block (last-N decision-relevant events —
#   move/switch_in/faint/status/boost/item/hazard/switch_rejected — with attributed hp_delta,
#   outcome/crit/effectiveness, we_first, log-saturated recency, forced-window phase tag),
#   folded by `EventWindowTracker` from PUBLIC events (seq-idempotent, the H-A machinery).
#   The obs widening is unconditional (retrain-class, weight-field-caught via total_dim); the
#   CONSUMER — `history_events`, the event SEATS joining the trunk with per-type projections +
#   the recency embedding — is opt-in (OFF builds nothing, byte-identical). v1 trims, recorded:
#   no faint-cause multi-hot, no item/hazard content ids, SETBOOST/CLEARBOOST skipped.
# v82 (gen3_unified_value_readout_v2): `value_entity_pool_full` — the entity pool's COMPLETE
#   row set (+the refined GLOBAL token, +the hidden-opp belief queries; sources 3 and 4). Its
#   own field because the source-embedding table grows 3→5 (a state_dict shape), keeping
#   v80-shape checkpoints (gen-12's) loadable under full=False. With this, every vf route the
#   critic_route_audit can condemn has ONE successor: the pool.
# v83 (gen3_item_belief_v1): `item_belief` — a learned posterior over each opp slot's HIDDEN
#   item (Smogon per-species item-usage prior ⊕ zero-init trunk delta; cold start == prior
#   exactly). Supervised as the BeliefBank's SEVENTH row (CE vs the privileged true item num
#   at revealed slots, --item-belief-coef). The op's Choice-Band-conditional tail consumes
#   P(CB) from the PUBLISHED posterior at the unrevealed branch, replacing the static
#   SPECIES_CB_PRIOR scalar there (revealed branch unchanged: exactness stays 0/1). Adds the
#   ItemBelief module (state_dict), so STRUCTURAL, version-checked, own flag.
# v84 (gen3_intent_threshold_v1): `intent_threshold` — the α-weighted THRESHOLD operator
#   `p_thresh(τ,⋛) = Σ_k α_k·1[damage(k,me) ⋛ τ]` (design_conditional_execution.md §3.0, build
#   step 3). One contraction over the op's existing per-candidate pair cells lands FIVE
#   mechanics at once through the pointer MOVE cell (Focus Punch executes / Sub survives /
#   Endure·p_KO / Destiny Bond·p_KO / Endeavor survives-to-act) and produces `p_KO` — the
#   calibrated "am I about to die this turn" — for the CRITIC (the ledger-H1 payoff: the
#   critic previously inferred it from _chan_max's hard max). Two zero-init projections
#   (state_dict): the move-cell block widens the pointer move cell by INTENT_THRESH_MOVE_DIM;
#   the vf block appends INTENT_THRESH_VF_DIM after the entity pool. STRUCTURAL, own flag.
# v85 (gen3_intent_conditional_v1): `intent_conditional` — the REMAINING α-conditioned mechanic
#   cells (design build steps 4+7): Counter / Mirror Coat as the α-weighted CATEGORY sums
#   ("the purest read-the-opponent moves in gen3 — literally unplayable without an intent
#   model"), flinch's missing (1−α_SWITCH) conditioning, Explosion's p_executes
#   (1 − Σα·is_protect) + into-switch mass (the H1 companion facts), and Pursuit's ×2
#   never-miss doubling trigger — CORRECTED against the rust port: the strike hits the
#   DEPARTING mon, not a β-weighted arrival, so no β enters. One zero-init projection widens
#   the pointer move cell by INTENT_COND_MOVE_DIM. STRUCTURAL, own flag.
# v86 (gen3_op_lean_forward_v1, design_op_tensors step 3): TWO flags. `op_drop_renders` — the
#   op's flat forward block loses its three RENDER regions (outgoing matrix / incoming matrix /
#   OAX), which have had no forward consumer since gen3_no_concat_v1; the matrices' SELECTION
#   machinery still runs and every consumer value survives as a typed stash, so out_dim (and
#   out_gain — a state_dict shape) shrink while every surviving offset is unchanged (renders
#   always appended last). `op_believed_lean` — the lean d3 physics (`_incoming_rolls`) price
#   the attacker from the BELIEVED spread instead of the legacy de-timid fiction (the B-spread
#   correctness fix at the last de-timid site the edges read); forward-math change, no shape.
# v87 (gen3_value_direct_routes_v1): two direct CRITIC routes, both zero-init vf-tail appends.
#   `value_clock` — the v67 deadline clock's 3 raw scalars get the explicit critic route the
#   fix was validated for (its surviving indirect route, the nmr concat, was audited dead).
#   `value_intent` — the published α/β posteriors AS DISTRIBUTIONS (α over its K belief-sorted
#   seats + SWITCH, β over the 6 slots): α previously reached vf only as a weighting inside
#   intent_value_reduce's physics cells and β not at all — the block was ORDERING (T2 heads vs
#   the assembler), which the post-assembler tail dissolves. Both widen the value projection
#   (state_dict), so mismatches are shape-caught; the checks name the cause.
# v88 (gen3_dead_flag_purge_v1): `value_active_readout` and `damage_matrices_outgoing_all` are
#   DELETED — both config_only frozen OFF since v78, never enabled in any gen-8+ run, and each
#   superseded twice over (the active read by the seed window then the entity pool; the OAX
#   render by d2, which keeps `_outgoing_attacker_matrix` as its physics engine). A config that
#   recorded either True is REFUSED (each widened a projection/out_dim the surviving code cannot
#   rebuild); False pops silently. `pubval_mode`/`pubval_coef` go the same way (measured NULL,
#   head never built in production): a recorded mode != 'none' is REFUSED (PubValHead carried
#   parameters), 'none' pops.
```

## The ARCH_SIGNATURE narrative (moved verbatim from `model_version.py`, 2026-08-16)

```
# v2 (gen3_unified_v2): turn-history TurnDelta slot expanded to 88 dims —
#   actor / target / switch_to species IDs (×6), boost deltas (×14), phase flag,
#   target_hp_delta, per-slot HP-level vectors, target-status onehots (×14, at
#   move-fire time, for Flash Fire-vs-frozen and sleep-talker reads). The history
#   embedding now reaches the species_embedding table for the first time, a new
#   wire that's not weight-compatible with v1 even if total_dim coincidentally
#   matched.
#
# v3 (gen3_abilities_v1): per-Pokémon ability block expanded 2 → 3 dims
#   ([ability1_id, ability2_id, known_flag]). For unrevealed opp slots the two
#   dex-possible Gen 3 abilities are written so the model has prior knowledge
#   (e.g. Snorlax = Immunity OR Thick Fat) instead of a flat zero. The role
#   encoder embeds BOTH ability IDs through the existing ability_embedding
#   table — a wire that didn't exist in v2. POKEMON_FULL_DIM 97 → 98, total
#   obs dim 2414 → 2426.
#
# v4 (gen3_abilities_v2): ability block grows to 4 dims with an inserted
#   `dominance` scalar — the Smogon-observed probability of ability1.
#   Layout becomes [ability1_id, ability2_id, dominance, known]. Priors are
#   now sourced from data/pokemon/gen3_ability_priors.json (top-2 by Smogon
#   usage), replacing the dex-slot-order approach from v3. POKEMON_FULL_DIM
#   98 → 99, total obs dim 2426 → 2438. The role encoder picks up the
#   dominance scalar as a passthrough float alongside the two ability
#   embeddings.
#
# v5 (gen3_move_outcome_v1): each turn-history TurnDelta slot gains move-outcome
#   reporting — our/opp move-outcome onehots (hit/miss/fail, ×6), our/opp crit
#   bits (×2), and the |cant| reason onehot widens 5 → 11 (recharge/taunt/
#   disable/imprison/truant/nopp added, with "move:"/"ability:" prefix
#   normalization). These are pass-through scalars routed through the existing
#   history embedding, inserted before the species-ID tail. TURN_DELTA_DIM
#   88 → 108 (+12 from the wider cant onehot, +8 from outcome/crit); total obs
#   dim shifts by N_HISTORY_TURNS × 20. Not weight-compatible with v4 — the
#   history projection input width changed.
#
# v6 (gen3_modular_v1): pure structural refactor — forward_internal decomposed
#   into phase nn.Modules (Embeddings / ObsUnpack / PokemonEncoder /
#   TeamTransformer / CLSPool / ProjectionAssembler). The math, dims, and outputs
#   are byte-identical to v5, but state_dict keys are now phase-prefixed
#   (e.g. move_network.* → pokemon_encoder.move_network.*, our_cls →
#   cls_pool.our_cls). Old checkpoints are intentionally incompatible so they
#   fail with a clean arch-family error instead of an SB3 strict-load KeyError.
#
# v7 (gen3_dual_value_v1): value-dedicated CLS readout (H4 / Option C). CLSPool
#   gains a third learned query (`value_cls`) that attends over all 12 team
#   tokens to produce a global value summary; ProjectionAssembler now emits a
#   (pi_combined, vf_combined) pair, and the root extractor has a second
#   projection head (`value_pre_norm` + `value_projection`). `forward` returns a
#   (pi_features, vf_features) tuple consumed by the new
#   `Gen3DualHeadMaskablePolicy`. The transformer body stays shared; only the
#   readout + projection + critic mlp branch are now independent. New weights and
#   a tuple-returning forward make this incompatible with v6 checkpoints.
#
# v8 (gen3_live_state_v1): the active-context + global-env blocks are re-sourced from
#   the event-sourced LiveView and substantially enriched (retrain-class). Active
#   context grows 23 → 55: the volatile block goes from a hand-picked 9 to the full
#   source-derived gen3 set (VOLATILE_DIM=41, crash-don't-drop, perish/stockpile
#   counters normalised) — recovering ~30 dropped volatiles (Disable/Encore/Taunt/
#   Destiny Bond/Curse/Yawn/Flash Fire/partial-trap/…). Global env grows 13 → 18:
#   weather is event-sourced with cause-aware permanence + turns-remaining (ability
#   weather = permanent, move weather = 5-turn countdown — read from the |-weather|
#   protocol, never guessed), the dead gen4+ weather slot is dropped, and per-side
#   Safeguard + Mist are added alongside Reflect/Light Screen. The weather feature the
#   extractor broadcasts into per-mon move context widens 6 → 7. Obs dim 2734 → 2823;
#   the global-token / active-ctx projection input widths all shift. Not weight-
#   compatible with v7.
#
# v9 (gen3_own_spread_v1): the own-team spread block (per-mon IVs/EVs/nature, 18 dims ×6
#   slots) now carries REAL data instead of constant fallbacks. gen3ou has no team preview,
#   so poke-env's apply_teambuilder_team (which matches the empty team-preview list) never
#   attached the spread, and own Pokemon.ivs/evs/nature stayed None — the spread block had
#   been emitting a constant vector (IVs all-31, EVs all-0, neutral nature) for every own mon,
#   i.e. zero signal. Fixed in the poke-env fork: Battle.parse_request now calls
#   backfill_teambuilder_spread() after building the team from the request, matching the
#   declared teambuilder team by species and filling in IVs/EVs/nature (spread only — it does
#   not re-run the full _update_from_teambuilder, so request-derived moves/PP/stats are
#   untouched). The obs spread block + LiveView read mon.ivs as before, now populated. Obs DIM
#   is unchanged (still 2823) — only the spread VALUES change — but the meaning of those dims
#   changes, so this is retrain-class: old checkpoints must not silently load.
#
# v10 (gen3_turn_delta_v2): TurnDelta is now folded from the event log (Step 4 of
#   the event-sourced battle migration). New per-decision-window fields: an 8-dim
#   faint-cause multi-hot per side (attack/hazard/weather/status/recoil/selfko/
#   leechseed/other), and our_attempted_move_id (the move we pressed, preserved even
#   when it never fired — freeze/sleep/flinch/cant/KO-before-act). attempted_switch_to
#   is NOT encoded (a pressed switch always executes, so it == switch_to); faint counts
#   live on the dataclass for reward but aren't encoded (redundant with the faint flags
#   + cause popcount). The cant one-hot switches to the authoritative gen3_effects vocab
#   (slp/frz/par/flinch/recharge/attract/disable/taunt/imprison/focuspunch/nopp/truant),
#   crash-don't-drop. Volatiles added to the active-context block: doomdesire/futuresight
#   (`-start` future-move volatiles) + the 11 gen3 ability-activation volatiles (Immunity/
#   Synchronize/Oblivious/Insomnia/Limber/OwnTempo/ShedSkin/StickyHold/SuctionCups/
#   VitalSpirit/MagmaArmor — poke-env's -activate path records them as effects; MagmaArmor
#   required adding Effect.MAGMA_ARMOR to the fork's enum); the event-log fuzz's per-decision
#   check + training smoke caught doomdesire/immunity. Ability activations now ALSO reveal
#   the opponent's ability persistently (abstract_battle -activate handler sets mon.ability
#   when None → per-mon ability block flips known=1), so the 11 ability-activation volatiles
#   COLLAPSE to one shared `ability_activated` slot (identity is in the ability block; the
#   volatile is just a hint to go look). VOLATILE_DIM 41 → 44. TurnDelta also folds STATUS
#   TRANSITIONS from the event log: our/opp status_applied + status_cured (4 × 7-dim
#   onehots) — the per-turn event (e.g. Lum Berry curing Toxic to enable a Dragon Dance),
#   distinct from the current-status snapshot; the cause-identity stays in the item/ability
#   block. Plus our/opp item-used BITS (2) marking an item was consumed/removed this window
#   (just a bit — the WHICH is in the per-mon item block, parity with ability_activated).
#   The embedded-ID positions are no longer hardcoded in the extractor: a single
#   TURN_DELTA_EMBEDDED_IDS manifest (in turn_delta_encoder) drives both the encoder
#   layout and features_extractor.embed_delta_slot (11 embedded IDs: 3 move + 2 type +
#   6 species). TURN_DELTA_DIM = 157, obs dim 2823 → 3299. Builds on v9 (own-team spread
#   backfill carries through). Not weight-compatible with v9.
#
# v11 (gen3_turn_delta_v3): turn-history window correctness fix. `prev_N_delta_vecs` was
#   folding each of the N history slots over `events_since(cursor)` — i.e. that turn's
#   cursor THROUGH NOW (no upper bound) — so every slot but the most-recent reported the
#   *latest* turn's event-derived fields (move/outcome/boosts/status/faint-cause), and the
#   per-step cost was O(N²). Now each slot folds exactly its own decision window
#   (`events_between(cursors[-1-i], cursors[-i])`; end=None for the most-recent). Obs dim is
#   unchanged (3299) — only the turn-history values change (older slots now carry their own
#   turn) — so this is retrain-class, not weight-shape-incompatible.
#
# v12 (gen3_trapping_signals_v1): route the three trapping signals into the model so it can
#   learn the hidden-information trap read (Arena Trap / Shadow Tag / Magnet Pull / Mean Look).
#   (1) + (2) two new reactive obs bits from the server-authoritative LegalActions snapshot —
#   trapped (confirmed cannot switch; redundant with the mask but explicit) and maybe_trapped
#   (the opponent MIGHT trap us; switches stay legal, so this is the only way the model can see
#   the risk before attempting a blind pivot and eating a rejection). They sit before the
#   matchups in the reactive block, so the extractor picks them up in non_matchup_rest;
#   REACTIVE_DIM 300 -> 302. (3) the rejected pivot becomes a first-class history event: a new
#   EventKind.CHOICE_REJECTED is recorded out-of-band (poke-env intercepts |error|[Unavailable
#   choice] before parse_message, so a duck-typed hook in _handle_battle_message calls
#   Gen3Battle.record_choice_rejected), TurnView folds it (attempted_rejected), TurnDelta gains
#   attempted_switch_rejected + the restored attempted_switch_to, and each TurnDelta slot gains
#   2 dims — an attempted_switch_rejected bit + the embedded attempted-switch species id
#   (manifest entry #12). TURN_DELTA_DIM 157 -> 159. Obs dim 3299 -> 3321 (+2 reactive +
#   N_HISTORY_TURNS x 2 history). Builds on v11. Not weight-compatible with v11.
# gen3_item_num_fix_v1: the per-Pokémon item id is now the true item-dex `num` (from data/, via
#   the gen3_data facade), not Showdown's `spritenum` as before. Obs dim unchanged (3321) and the
#   item embedding table size is unchanged (max_items=600 still covers the new max, 499), but the
#   item id -> item meaning is re-mapped for every item, so item embeddings learned under the old
#   ids are semantically invalid. Re-meaning an obs block is retrain-class. Builds on
#   gen3_trapping_signals_v1; not weight-compatible with it.
#
# gen3_move_effects_v1: action-aligned per-move EFFECT features in the reactive block. The only
#   per-move signals that previously reached the policy head in REQUEST (action) order were base
#   power and the type multiplier — so for status/utility moves (power 0, neutral multiplier) every
#   option looked identical at the head, and the model could not tell a setup move from a heal from
#   a wasted Toxic (it clicked immune Toxic into Poison-types for many turns). Now each of the 4
#   request-order move slots carries 9 flags — is_boost, is_heal, is_protect, is_phaze, is_hazard,
#   inflicts_status, status_will_land, pp_fraction, status_will_land_known. Static flags are derived
#   in the acquisition tool
#   from the field Showdown keys each mechanic on (flags.heal, volatileStatus, forceSwitch,
#   sideCondition, primary `status`, declarative self-positive boosts) PLUS a curated callback
#   override for Belly Drum (onHit-only boost); Curse's type-conditional setup is resolved live in
#   the encoder. status_will_land is a PRIOR-WEIGHTED probability in [0,1] (priors first, then
#   confirmation — same ability-distribution path as the matchup cells): 0 on a certain block
#   (type immunity / already statused / Substitute), else 1 − P(ability blocks the status) over the
#   opponent's Smogon ability prior, collapsing to 0/1 once the ability is revealed; the trailing
#   status_will_land_known bit flags confirmed-vs-prior with the SAME predicate the per-mon ability
#   block's `known` flag uses (revealed ability OR a type-certain hard block), so the policy can
#   tell a confirmed outcome from a prior estimate — parity with how abilities are routed. The block
#   sits before the matchups, so the extractor picks it up in non_matchup_rest → both policy and
#   value projection input widths grow (auto-discovered). REACTIVE_DIM 302 → 338; obs dim 3321 → 3357.
#   Builds on gen3_item_num_fix_v1; not weight-compatible with it.
# gen3_incoming_damage_v1: per-our-mon INCOMING-DAMAGE / OHKO BELIEF block (incoming_damage.py +
#   gen3_{move,spread,item}_priors): for the opp active vs each of our 6 mons, the phys/spec
#   expected-damage-fraction + mode-max P(KO) (gen3 damage formula + fixed-damage branch
#   [Seismic Toss/Night Shade/…] + Reflect/Screen/Sub/burn/weather modifiers + roll→P(KO), over the
#   usage-prior belief: revealed∪prior moves, offensive-tail stat) + P(outspeed) over the Speed
#   distribution, then 3 opp recovery scalars (Suicune-Rest discriminator). Sits after move-effects,
#   before the matchups → flows to both heads via non_matchup_rest (auto-discovered widths).
#   REACTIVE_DIM 338 → 371; obs dim 3357 → 3390. Builds on gen3_move_effects_v1; not weight-compatible.
# gen3_incoming_damage_v2: re-calibrates the incoming-damage / OHKO belief VALUES (same 33-dim block,
#   same obs dim 3390 — only the numbers change, so it's retrain-class, not weight-shape). Two
#   complementary belief-value fixes for the calibration tail found on run_20260606_204351 (17% of
#   direct-hit deaths read P(KO)<0.25): (1) P(KO) was too timid on near-OHKOs — the offensive-stat
#   tail percentile is raised 0.85→0.95 (the KO magnitude rides the tail; expected-damage
#   re-normalises to the mean, so the chip belief is unchanged) AND a gen3 critical-hit term
#   (_CRIT_P=1/16, ×2, screen-ignoring) is folded into P(KO), so a hit that only KOs on a strong set
#   or a crit reads a calibrated risk instead of ~0; (2) the candidate set is widened so the killing
#   move is no longer silently absent — a revealed bare Hidden Power (dex BP 0) expands into per-type
#   candidates (~70 BP, typed from the HP tracker's narrowed distribution / Smogon HP prior),
#   variable-power Return/Frustration (dex BP 0) are priced at 102 BP, and the prior floor/cap widen
#   (0.12→0.05, 4→6 per channel) so a low-usage super-effective coverage move survives into the pool
#   (the per-defender max over p_in_set·P(KO) is the real type-effectiveness gate). The HP tracker is
#   now threaded into the incoming-damage encoder. Not weight-compatible with v1 (the belief values a
#   reload would read are different → old critic readings of the block are invalid).
# gen3_markovian_progress_v1: adds the turns_since_progress reactive scalar (vec[14]) — the
#   log-saturated no-progress clock (design_markovian_reward_and_features.md §5.1), an
#   EpisodeTracker-owned cross-turn counter threaded into encode() like the HP tracker.
#   REACTIVE_SCALAR_DIM 14 → 15 → REACTIVE_DIM 371 → 372, obs dim 3390 → 3391. The scalar is
#   present in every run (the clock always tracks it for the obs); the no-progress PENALTY +
#   the obs-keyed reward reframes are gated on the reward's bias_redesign flag, so the
#   single-variable material-clutch-fix run and the bias-redesign run share one architecture.
#   The reward redesign also folds the material spine into a PBRS Φ_mat and renames the belief
#   PBRS field (pbrs_material → pbrs_belief); those are reward-VALUE changes (retrain-class) that
#   need no further arch bump. Not weight-compatible with gen3_incoming_damage_v2 (obs dim +1).
# gen3_incoming_crit_split_v1: SPLITS the incoming-damage belief's P(KO) into a modal no-crit line +
#   a per-channel crit-risk DELTA (crit-inclusive − no-crit ∈ [0, _CRIT_P]), and adds a per-mon
#   threat-PROVENANCE scalar (the dominant KO threat's p_in_set: 1.0 = a REVEALED move, <1.0 = a
#   usage-prior GUESS, 0.0 = no candidate can KO). Motivation: the model over-weighted uncontrollable
#   crit RNG (it should optimise EXPECTED value over the modal line, with crit as a priced tail) and had
#   no signal for how much of a threat is KNOWN vs guessed — both validated as gaps by the
#   representation-probe harness (the rep barely encodes damage spread). The crit risk is exposed as the
#   DELTA (not the near-redundant absolute crit-inclusive line, which is ≤6% above no-crit and gets
#   buried after standardization). INCOMING_PER_MON 5 → 8 → INCOMING_DMG_DIM 33 → 51 → REACTIVE_DIM
#   372 → 390, obs dim 3391 → 3409. Crit was ALREADY computed (folded into P(KO) since v2); this unblends
#   it as a delta + adds provenance, so the underlying numbers are unchanged — but the block layout/width
#   differ, so it is not weight-compatible with gen3_markovian_progress_v1.
# gen3_move_slot_align_v1: FIXES a per-move obs misalignment (GIGO). The active-move features in
#   reactive.py (base power vec[0:4], type multiplier vec[4:8], the 36-dim move-effect block) were
#   filled by iterating `battle.available_moves`, which poke-env builds with DISABLED moves dropped
#   (`available_moves_from_request`). The action mask / mapper index `legal.move_slots` (request-slot
#   order, disabled KEPT) → so under a disabled non-last slot (Disable / Taunt / Imprison / 0-PP) every
#   per-move feature shifted out of alignment with its action logit (feature slot k described a
#   DIFFERENT move than action 6+k), and the trailing slot kept the np.ones(4)/4 default — which decodes
#   to a phantom 4× super-effective KO threat on a legal action. Now the loop iterates request-slot
#   order via `_request_slot_moves` (disabled kept, typed-HP preserved) and the unwritten-slot default
#   is the neutral 0.25 (1×). Same dims (obs 3409 unchanged), VALUES only on the disabled-slot /
#   <4-move / no-opp-active cases — so it is retrain-class (not weight-shape), not byte-compatible with
#   gen3_incoming_crit_split_v1. The common all-moves-available decision is byte-identical.
# gen3_protect_odds_v1: adds TWO reactive scalars (vec[15] our active, vec[16] opp active) — P(a
#   Protect/Detect/Endure succeeds NOW) under the gen3 floored-doubling stall rule. Showdown's gen3
#   format inherits the stall condition through gen4 → gen5 (NOT the base data/conditions.ts *3): the
#   counter starts at 2 and DOUBLES each consecutive successful stall move (gen5), capped at 8 (gen4
#   counterMax → "the chance does not fall below 1/8") → 100/50/25/12.5 then a 12.5% floor. Sourced from
#   each active mon's `LivePokemon.protect_counter` (poke-env's consecutive-successful-stall counter,
#   reset on switch/faint/non-stall move/failed roll) via the LiveView read-model — never raw poke-env.
#   The model had no other view of the stall counter (poke-env doesn't enumerate the 'stall' volatile,
#   and turn-history saliency decays before a chain can be counted). Public for both sides (the opp's
#   counter derives entirely from their revealed move stream → no leak). REACTIVE_SCALAR_DIM 15 → 17 →
#   REACTIVE_DIM 390 → 392, obs dim 3409 → 3411. Verified: protect_success_prob_fuzz_test.py (encoded
#   scalar == the gen3-correct prob for the live protect_counter, + the empirical % match). Not
#   weight-compatible with gen3_move_slot_align_v1 (obs dim +2).
# gen3_status_cure_moves_v1: ADDS two static per-move EFFECT bits to the action-aligned move-effect
#   block — cures_self_status (Refresh clears the user's own status) and cures_team_status (Heal Bell /
#   Aromatherapy clear the whole party's). Motivation (prober-verified on ai_v6_01): the policy head
#   had NO per-move signal that a move CLEARS status — Refresh read as an inert move (base power 0, all
#   effect flags 0), so the head routed its own status onto Recover/switch (intervention: removing a
#   Toxic moved P(recover)/switch by ~11pp each but P(refresh) by ~1.5pp) and under-used the cure
#   (~1.4% when badly poisoned). The cure lives in an onHit callback (invisible declaratively), so the
#   bits are a curated override in the acquisition tool (like Belly Drum) and read against the per-mon
#   status one-hots the head already sees — provide the fact, let it learn. MOVE_EFFECT_FEATURES 9 → 11
#   → MOVE_EFFECTS_DIM 36 → 44 → REACTIVE_DIM 392 → 400, obs dim 3411 → 3419 (stacks on gen3_protect_odds_v1).
#   Not weight-compatible (move-effect block widened); the non-cure obs values are otherwise unchanged.
# gen3_sleep_wake_belief_v1: ADDS a 3-dim per-mon SLEEP WAKE belief block to each team slot (after the
#   HP block) — [sleep_is_deterministic (1.0 = Rest fixed-duration source), p_wake (COMPUTED P(wake on the
#   next move attempt) over the verified gen3 sleep tables: opp time=random(2,6)∈{2,3,4,5}, Rest time=3,
#   Early Bird halves; marginalised over the opp's Smogon Early-Bird prior, collapsing to exact 0/1 for our
#   own mon or a revealed opp), sleep_counter_reliable (0.0 once a Sleep Talk / Snore turn has corrupted
#   poke-env's +3-noisy counter)]. Motivation: poke-env exposes only Status.SLP + a noisy turn counter — NOT
#   the rolled duration, remaining time, or the source move — so a policy reading the raw counter must LEARN
#   the gen3 sleep RNG and cannot tell deterministic Rest from a random opp-sleep at the same counter. We
#   COMPUTE the wake odds (provide-the-fact) and read the Rest source from our event log's [from] clause
#   (poke-env discards it). Mechanics research + adversarial re-simulation: the four P(wake) tables were
#   re-derived bit-for-bit; Sleep Talk +3 counter-noise empirically confirmed → the reliability bit instead
#   of reconstructing Showdown's skippedTime switch refund. Fuzz-calibrated vs the real sim RNG. POKEMON_VECTOR_DIM
#   106 → 109 → POKEMON_FULL_DIM 107 → 110 (+3 per slot × 12), obs dim 3419 → 3455. Stacks on the same
#   unshipped change as the status-cure bits; not weight-compatible (per-mon slot widened).
# gen3_wish_reserve_v1: RESERVES two reactive scalars (vec[17] our side, vec[18] opp side) for a future
#   pending-Wish "floating heal" signal — NOT wired (the encoder leaves both 0.0). Reserved now so wiring
#   Wish later (a Wish queued for a side heals the mon switched in at the end of the next turn) is a
#   VALUES-only change with NO obs-dim / ARCH bump. REACTIVE_SCALAR_DIM 17 → 19 → REACTIVE_DIM 400 → 402,
#   obs dim 3455 → 3457. Pure placeholder: with the dims at 0 the obs is byte-identical to
#   gen3_sleep_wake_belief_v1 EXCEPT for the two reserved zeros + the shifted move-effect/incoming/matchup
#   offsets, so it is retrain-class (weight-shape) but carries no new information until Wish is wired.
# gen3_wish_wired_v1: WIRES the reserved wish_floating scalars (vec[17] our side, vec[18] opp side) with
#   the pending-Wish "floating heal" signal — a VALUES-only change (same obs dim 3457, no shape change), so
#   a gen3_wish_reserve_v1 checkpoint is retrain-class-incompatible only in the two dims' values. gen3 Wish
#   (INHERITS the gen4 condition, NOT base): heals the RECIPIENT's floor(maxhp/2) at the END of the turn
#   AFTER cast, SLOT-keyed (survives faint / Roar-phaze / switch / self-KO — the slot's occupant at resolve
#   is healed; gen3 sends replacements in mid-turn before residuals), duration 2, double-Wish on an occupied
#   slot FAILS, full-HP resolve is silent. poke-env tracks NONE of this → reconstructed from our event log
#   (observation/wish_belief.py): pending for a side iff it successfully cast Wish last turn (double-Wish-
#   aware). Because the heal is the RECIPIENT's maxhp/2, the heal fraction is ALWAYS ≈0.5 — so the encoded
#   value is a flat WISH_HEAL_FRACTION (0.5) when pending, 0.0 else: no max-HP read, no GIGO. Fuzz-calibrated
#   vs the real sim (the |-heal|[from] move: Wish resolve confirms the pending signal fired the turn before).
# gen3_rest_loop_stall_v1: RE-MEANS the turns_since_progress no-progress-clock scalar (vec[14],
#   gen3_markovian_progress_v1) — a REST-LOOP (our active Rested earlier this episode, woke, and re-Rested
#   without Sleep Talk) is now classified a NO_OP (stalled) instead of a free defensive heal, so it ADVANCES
#   the clock (obs) and CHARGES no_progress_tax (reward, when the clock charge is active — bias_redesign /
#   all_shaping_pbrs) like any other wheel-spin. A Sleep-Talk mon (legitimate act-while-asleep loop) and a
#   WINNING residual rest-stall (Toxic/Leech chipping the opp down → caught by _is_progress first) stay
#   exempt. VALUES-only on rest-loop turns (same obs dim 3457, no shape change) — but it re-means an obs
#   feature, so it is retrain-class: an old checkpoint won't load (loud arch-family error), which is correct
#   since it was trained with the prior clock semantics. (progress_clock.py: the heal-grace bypass.)
#   This rest-loop signature ALSO covers a SECOND no-progress-clock (vec[14]) refinement authored alongside
#   it and folded in WITHOUT its own signature bump (owner decision — a values-only clock change; the live
#   ARCH below has since moved on for unrelated reasons): a self-status-cure move (Refresh) used with NO
#   status to cure (`cures_self_status` + `our_status_cured is None`) is a NO_OP charged BEFORE the progress
#   check (a definitional-no-op short-circuit, like capped Spikes), so even a WINNING residual (our Leech
#   Seed / Toxic chipping the opp net-down) can't launder it into "progress" — the Refresh-spam-while-seeded
#   stall. (progress_clock.py: _is_wasted_self_cure short-circuit.)
# gen3_op_move_align_v1: FIXES a DamageOperator OUTGOING move-order bug. The op's per-move OUTGOING blocks
#   (_outgoing_block v23, _status_landing v27, _outgoing_matrix v34) emit one feature group per OUR move and
#   the POLICY head reads group k as action 6+k (request order) — but they READ ctx.all_move_ids[our_active],
#   the per-mon obs block, which is SORTED-BY-ID (the role token concatenates the 4 move encodings, so its
#   value is order-sensitive and the block can't be reordered). Sorted-by-id ≠ request order in ~96% of
#   decisions, so the outgoing tie-break / status-landing / switch-in-KO matrix were positionally misaligned
#   with the actions they inform (an under-`--unified-obs` correctness bug, since the CPU per-move blocks are
#   masked there). The FIX adds a request-ordered OUR-ACTIVE obs slice (reactive.py `active_req_moves`:
#   [move_num ×4, resolved_type_id ×4, legal_now ×4], from legal.move_slots, the same source the action mask +
#   move-effect block use) → ctx.our_active_req_move_{ids,type_ids,legal}; the 3 op methods read THAT (request
#   order) + gate with the current-decision legality (was the prev-turn, sorted-by-id move_mask). The v36/v37
#   refine OUTGOING methods (discrete_outgoing*) max-pool over our moves → order-invariant, left unchanged.
#   A STRUCTURAL/SHAPE change: REACTIVE_DIM 402 → 414, obs dim 3457 → 3469. Old checkpoints fail loudly (the
#   total_dim weight-shape check AND the arch-family signature), which is correct — they were trained with the
#   misaligned op. Guarded so it can't silently recur: move_alignment_fuzz_test asserts the obs slice IS in
#   legal.move_slots order, and damage_op_test asserts the op's outgoing slot k uses request-slot k.
# gen3_typed_hidden_power_ids_v1: gives each TYPED Hidden Power its OWN distinct move num so OUR side's
#   HP is represented by the move embedding itself, not a soft-type-blend workaround — a VALUES-only obs
#   change (same obs dim 3469 — it stacks on gen3_op_move_align_v1's reactive-block widening; NO
#   weight-shape change: the typed nums 355-370 are previously-unused rows
#   in the move embedding, max_moves=400). KNOWN→DISTINCT, UNKNOWN→TYPELESS+BELIEF:
#   - data/pokemon/gen3_moves.json: bare `hiddenpower` stays num 237; the 16 typed variants get distinct
#     nums 355-370 (deterministic, alphabetical — tools/pokemon_data_extractor/sync.py `_HP_TYPE_NUMS`).
#   - OUR side (type known): the obs move-id channel + the damage-op per-num tables (BP/type/attr/latent)
#     now carry the distinct num & real type, so our HP is a normal typed move (the feature extractor's
#     `is_hp_slot == 237` no longer matches it → it skips the hp_probs soft-type blend) and our OUTGOING
#     HP is priced correctly (was BP-0/type-0 before). The turn-history `our_move` also folds the distinct
#     num (via LegalActions.own_hp_typed_id). This SUPERSEDES the gen3_own_hp_typed_history_v1 hp_probs
#     one-hot workaround (reverted — own-HP hp_probs stays all-zero, correct since the blend is opp-only).
#   - OPPONENT side (type unrevealed — Gen3 never reveals it): the protocol gives bare `hiddenpower` → 237;
#     ALL opp-belief machinery stays on 237 — the HP tracker, the hp_probs soft-type blend, the damage-op
#     237→16-typed-candidate expansion, AND the move-belief PRIOR + LABELS (damage_tables._belief_num and
#     gen3_env._move_num fold every typed-HP usage/label back onto 237, so the opp-HP belief mass is NOT
#     scattered to 355-370). This known/unknown boundary is the load-bearing invariant (fuzzed by
#     move_id_decode_fuzz_test + hidden_power_typed_obs_fuzz_test). Design:
#     designs/ai_v6/design_typed_hidden_power_ids.md.
#   gen3_opp_hp_typed_candidates_v1: the DamageOperator now treats the OPPONENT's Hidden Power as 16
#     ORDINARY typed-move candidates at the distinct dex nums 355-370 (real BP/type from the typed-HP data
#     above) instead of a synthetic appended-16 block; the bare typeless 237 (BP 0) is the masked presence
#     token, and the per-type HP belief (mode off=obs / prior / learned) is scattered onto 355-370. A
#     FORWARD-MATH change to the op (the obs is unchanged + the op out_dim/projection widths are unchanged,
#     so it's not caught by shape checks) → bump ARCH_SIGNATURE so a pre-unification damage_op checkpoint
#     fails loud rather than silently computing the old HP candidates. The HP-type belief + the (v2) token
#     reinjection ride the existing `hp_type_belief_mode` (config v38).
#   gen3_pointer_native_v1 (the FRESH-GENERATION reset, designs/ai_v9/design_pointer_action_head.md §0):
#     the flat positional action head is DELETED — `Gen3DualHeadMaskablePolicy._build` replaces SB3's
#     `action_net` Linear with a raising stub and the `PointerNativeActionHead` scores every action from
#     the token of the entity it selects (move logit k ← the REQUEST-slot-k move token ⊕ its op cells;
#     switch logit j ← our-team token j ⊕ its incoming/OAX cells; struggle ← the latent_pi context) —
#     position-EQUIVARIANT by construction. The state_dict changes shape (no `action_net.*` Linear, new
#     `pointer_head.*` keys) AND the forward changes for every model, unconditionally (no flag), so the
#     signature carries the cross-era break: every pre-generation checkpoint fails the family check loud.
#     No old checkpoint is resumed/warm-forked across this boundary (owner decision, 2026-08-03); pools
#     and opponents are re-seeded from the new lineage.
#   gen3_typed_hp_belief_v1 (config v52 — stacks on gen3_pointer_native_v1): the model never reasons
#     over a typeless Hidden Power again. The presence×type composition `P(HP_t) = presence · P(type)`
#     moves into `HPTypeBelief.compose_typed_hp`, right beside the move-belief head, so the posterior
#     EVERY consumer reads (damage op, top-K, move BCE, latent grading, token reinjection, prober)
#     carries HP at the 16 real typed nums 355-370 with the bare BP-0 num 237 driven hard-off (a finite
#     -30 logit; 237 survives only as the internal PRESENCE channel). Supersedes the
#     gen3_opp_hp_typed_candidates_v1 op-side scatter above: the v38 tri-state `hp_type_belief_mode` is
#     DELETED (its 'off' state was a correctness bug — a revealed HP priced as nonexistent), the head is
#     unconditional under a move belief, the belief LABELS use the true typed num, and the op is a plain
#     consumer (no hp_type_fix / SPECIES_HP_PRIOR). Forward math changed with out_dim + projection
#     widths UNCHANGED → nothing shape-based catches it, so the signature carries the break.
#   gen3_entity_move_seats_v1 (config v54, Stage 1 of the entity generation — the roadmap's move-tokens
#     slice; stacks on gen3_typed_hp_belief_v1): move tokens become first-class attention SEATS in the
#     trunk, and the pointer head reads the REFINED E3 seats. UNCONDITIONAL state_dict changes for
#     every model (the token-type embedding table grows 4 → 6, `entity_seats.move_seat_proj` is new,
#     the pointer head's `move_proj` widens 32+cells → d_model+cells) plus an unconditional forward
#     change (4+ new seats in every attention pass) — no off state, so the signature carries the break
#     exactly like the v51 bump. The within-generation knob is `entity_topk_seats` (E4 threat seats),
#     gated in check_compatible. No pre-v54 checkpoint was ever trained (the generation's bumps all
#     landed same-day), so nothing is stranded.
#   gen3_op_block_trim_v1 (config v55 — stacks on gen3_entity_move_seats_v1): the DamageOperator sheds
#     its three least-used output families and one dead code path, on the ledger-P1 per-block dependence
#     ablation. OUT: the opp-active believed-EFFECT scalars (6 dims, 1.2%), the opp-active per-STATUS
#     incoming SECONDARY scalars (10 dims, 0.1% — INERT), the OUTGOING slp/psn/tox per-move secondary
#     columns (12 dims, structural zeros on the whole team pool), and the v30 LEAN `_topk_block` (0
#     calls/forward — a strict subset of the v35 incoming matrix, which suppressed it). Net −28 op dims
#     off BOTH projection heads, and the unmasked-belief `w` read leaves the forward entirely. The
#     projection widths DO change, so a stale checkpoint would fail on a state_dict shape mismatch —
#     the signature bump is what turns that into a clear arch error instead. Orthogonal to v54's entity
#     seats (trunk) — this trims the op's head-concat output — so the two compose; only the signature,
#     which is one shared string, had to be sequenced.
#   gen3_edge_bias_trunk_v1 (config v56, Stage 2 of the entity generation): the encoder stack is
#     the biased-attention clone — state_dict keys change for every model (no off state), so the
#     signature carries the break like v51/v54/v55. The within-generation knob is `edge_bias_families`
#     (which families are delivered; zero-init maps ⇒ ON starts identical to OFF).
#   gen3_entity_rehome_v1 (config v60, Stage 3 of the entity generation): the flat obs's DERIVED
#     blocks are deleted and every raw fact re-homed to its entity — the 288-dim CPU matchup
#     matrices and 6 of the 11 reactive scalars are GONE (obs 2925 → 2667), protect_odds /
#     trapped / maybe_trapped ride the per-mon slots (POKEMON_FULL_DIM 113 → 116), and the
#     PokemonEncoder's move/role nets narrow (matchup + validity + struggle inputs deleted).
#     Weight shapes AND obs meaning change together — a fresh-lineage break (gen-4).
#   gen3_no_concat_v1 (config v61, the gen-5 world): THE OP HEAD-CONCAT IS DEAD — the 660-dim
#     flat block no longer enters either projection (pi 1131→471); the critic's replacement
#     window is the multi-seed readout (MultiSeedValueReadout, k=4×64 over the op's per-our-mon
#     rows, vf-only, with the seeds/* TB collapse contract). Executed on the gen-4 stratified
#     evidence (53ef270): net policy dependence +0.00%, flips half of the acceptance clause met
#     by training, act_threat decodable concat-zeroed. state_dict changes (projection widths +
#     the new module) → the signature carries the break; fresh lineage (gen-5).
#   gen3_deadline_clock_v1 (config v67): the obs CLOCK group is 3 scalars, not 1 — log-ELAPSED
#     (opening structure) plus remaining-LINEAR and log-REMAINING (deadline structure). The old
#     single log-elapsed scalar put 58.6% of its range on turns 1–50 and 4.0% on turns 200–250,
#     i.e. it had almost no resolution at the forfeit cap the trainee actually loses on. Obs
#     2667 → 2669 (GLOBAL_ENV_DIM 18 → 20) and the move/global context projections widen with
#     `_gl['clock']['dim']` → state_dict changes; fresh lineage.
#   gen3_ctx_dedup_v1 (config v76): the assembler's per-side encoded active contexts are DELETED
#     from both heads (duplicated delivery — the E2 injection + the global token already carry
#     the full raw ctx blocks into the trunk). `active_ctx_encoder` state_dict keys removed,
#     both projection inputs narrow by 64 → the signature carries the break; fresh lineage.
```


## `gen3_model_file_split_v1` (2026-08-16): one responsibility per file — byte-identical, no version bump

Structural only; the production sha probe reads `3cab191a…` unchanged at every step. Four phases:

1. **`model_version.py` −1,029 lines**: the v3→v88 MODEL_CONFIG_VERSION narrative and the
   ARCH_SIGNATURE history moved verbatim into this file's two "narrative" sections above —
   61% of the module was history living as comments. The machinery keeps pointers.
2. **`snapshot.py` → `compile_opponents.py`**: checkpoint save/load and the CPU-opponent
   `torch.compile` path were two unrelated responsibilities glued together. The compile half
   (constants, `maybe_compile_extractor`, the eager fallback, `_COMPILE_VALIDATED`) now lives
   beside `compile_prewarm`/`compile_preload`/`compile_trainer`; production importers repointed;
   `snapshot.py` re-exports for history.
3. **`features_extractor.py` 4,247 → ~1,900 lines**: the phase modules move to sibling files —
   `extractor_ctx` (ExtractorContext/ObsUnpack/Embeddings), `encoders`, `team_transformer`
   (EdgeBias + the `_EDGE_*_CELL` definitions — `build_arch_viewer`'s source parser repointed),
   `pools`, `belief_heads`, `aux_value_heads`, `pointer_head`, `value_readouts`. The orchestrator
   (`Gen3FeaturesExtractor`, one responsibility, ~1,650 lines) and `ProjectionAssembler` stay;
   every moved name is re-imported explicitly, so the module remains the documented import hub,
   and the class stays DEFINED there because SB3 checkpoints pickle it by defining module.
4. **`damage_op.py` → `damage_op_layout.py`**: the `_DMG_*` offset/width constants, `OpTensors`
   and `decode_damage_block` (the block's SHAPE CONTRACT) split from the `DamageOperator` physics;
   full re-export as above.


## v89 — `gen3_value_pooled_routes_v1` (2026-08-17): the value routes finally reach the critic

**The bug (verified on gen-12's final_model.zip):** `--value-from-dist` makes
`_critic_value` read the dist head, and the dist head reads `value_pooled` — so everything
concatenated into the post-assembler vf tail was structurally disconnected from V and received
no value-loss gradient. Five routes were inert whenever the dist critic was on:
`intent_value_reduce` (v74), `value_entity_pool` (v80/82), `intent_threshold`'s vf half (v84),
`value_clock` and `value_intent` (v87). Proof, not inference: after gen-12's 25M steps,
`value_entity_pool.out_proj.weight` and `intent_value_reduce.proj.weight` were bit-exact ZERO
(their upstream layers frozen at init noise), while `cls_pool.value_threat_proj` — the one route
already injecting into `value_pooled` — trained to 0.117. Gen-11 and gen-12 both ran
`--value-from-dist`, so v74/v80 were dead for two full generations, and every endofrun route-audit
arm measuring a vf-tail route was measuring a dead limb.

**The fix:** every value route now INJECTS additively into `value_pooled` (the
`value_threat_inject` precedent) through a zero-init `D_MODEL`-wide projection —
`_value_pooled_routes` in the orchestrator is the single registry/seam. Since
`vf_parts[0] is value_pooled`, one wiring feeds BOTH critic parameterizations. Properties held:
zero-init (ON-at-init exact), vf-only at ANY weight (pi never reads value_pooled), M1 capture.
A structural bonus: additive injection is width-neutral, so the ede5a88 discovery-sizing bug
class (an early return hiding a vf part from the `value_pre_norm` dummy forward) is
unrepresentable, and the per-route width constants (`INTENT_VALUE_REDUCE_DIM`,
`INTENT_THRESH_VF_DIM`, `VALUE_CLOCK_DIM`, `VALUE_INTENT_DIM`, `UVR_OUT_DIM`) are deleted.

**The guard that was missing:** `value_route_gradient_test.py` backprops the critic (BOTH
parameterizations) through every route the registry yields and fails on any zero gradient —
one backward pass, generic over the seam, so the next route is covered by construction.

**Versioning:** MODEL_CONFIG_VERSION 89. `ARCH_SIGNATURE` unchanged — flag-OFF configs are
byte-identical. A <v89 checkpoint recording any of the five flags ON is REFUSED by the migration
(the v75 rule: its projection shapes no longer exist; re-read from its metadata git_hash); OFF
stamps forward. The production sha probe moves 3cab191a… → 694c1652… (production has
`intent_value_reduce` + `value_entity_pool` ON; their delivery is the change).


## `gen3_damage_op_mixins_v1` (2026-08-17): the operator's method families become mixins — byte-identical

`damage_op.py` was 2,759 lines, ~2,640 of them one class. The class has two coherent method
families with clean seams, now MIXINS the class inherits (`DamageOperator(DamageOperatorPairwise,
DamageOperatorBlocks, torch.nn.Module)`): `damage_op_pairwise.py` — the 17 `pairwise_*`
edge-family cell producers plus their shared believed-attacker helpers (~930 lines); and
`damage_op_blocks.py` — the outgoing/incoming/status flat-block builders including the OAX kernel
(d2's engine) and the discrete status probes (~880 lines). Mixins carry NO parameters, so the
state_dict, parameter order (the SB3 optimizer-by-position hazard) and forward are untouched —
verified by the production sha probe reading identically with and without the split, at the
gen-13 production config (`001e1140…`, the baseline since `26b2850` promoted the enable set).
`damage_op.py` keeps the ctor, stashes, core roll math, the pointer/slicer surface and the
forward (946 lines). Also deleted: `entity_spike_benchmark.py` — the closed Stage-2 feasibility
spike, which shadowed the production `BiasedEncoderLayer` class name; its measured results stay
cited in `team_transformer.py`/`pointer_head.py` with a git-history pointer.


## `gen3_extractor_stashes_v1` (2026-08-17): the extractor's side values become ONE typed container — byte-identical, plus a critic fail-loud

The OpStashes recipe (`gen3_op_stashes_v1`) applied to `Gen3FeaturesExtractor` itself, after v89
made the cost of the old pattern concrete: phases communicated through mutable `self.last_*`
instance stashes, cross-module consumers read them with `getattr(..., None)`, and nothing
type-level connected producer to consumer — which is how five value routes fed a concat the dist
critic never read, silently, for two generations.

**The container:** every per-forward side value the extractor exposes (`pointer_inputs`,
`alpha/beta_logits` + `alpha_seat_nums` + `thresh_probs`, the belief-bank publications
(`move_belief/spread/item/hp_type/belief_logits`, `opp_believed_mask`, `opp_active_local`,
`move_latent_table`), `damage_block`, `value_pooled`, `win_prob_logits`, `value_dist_logits`, the
internal T0→T1/T2 hand-offs `t0_species_probs` / `entity_latent_table`, and the LIVE
`belief_supervision` dict) lives in ONE `ExtractorStashes` dataclass that `forward_internal`
replaces at ENTRY — a stale cross-batch read is unrepresentable for any stash, uniformly (the old
code had at least three reset conventions, and `_entity_latent_table` was read back through a
`getattr` because nothing guaranteed it existed). Reads stay on read-only `last_*` properties (the
documented surface; every consumer keeps its spelling), writes go through `fe.stash.<field>`, and
a stray write to a legacy name raises. `_publish_belief`/`belief_supervision()` stop-grad
semantics are UNCHANGED — the dict just rides the container, so its per-forward clear is the same
entry replacement. `PokemonEncoder.last_move_tokens` deliberately STAYS on the encoder: each
producer module owns its own stash surface (the op keeps OpStashes); hoisting a submodule's stash
into the parent's container would be the cross-module write this change exists to remove.
`_last_hp_type_post` was found to have NO reader anywhere and is deleted.

**The hazard fix (policy.py):** `_critic_value` under `--value-from-dist` used to FALL BACK to
the scalar `value_net` when the dist head/logits were missing — but under value_from_dist that
net is FROZEN, so the fallback was a silently-wrong critic, the exact v89 shape. It now RAISES
(missing head, un-stashed logits, or a batch-size mismatch = stale stash), pinned by
`dist_critic_test.py`. The scalar path with the flag off is unchanged.

**Cross-module readers re-routed:** `policy.py` (`last_pointer_inputs`, `last_value_dist_logits`),
`instrumented_ppo.py` (`last_alpha_seat_nums`), `main/prober/model.py` (7 sites incl. the op's
`last_raw_block`/`last_topk_idx`), `agents/inference/player.py` (12 sites) — all now typed
property reads; no `getattr(..., None)` reach-across remains for extractor/op stashes.

**Riders:** (a) `observation_space` is annotated `spaces.Space` and documented DELIBERATELY
UNREAD (SB3's positional construction contract; `layout` is the dim source) — three probe-side
`type: ignore[arg-type]`s die with it; (b) absent `layout` now raises a named ValueError at ctor
entry (and the narrowing retires ~15 `type: ignore`s); (c) `delivery_graph.build_graph` raises
loudly on an op-less config instead of a deep AttributeError through a `cast`; (d) `ruff.toml`'s
TEMPORARY handoff section is CLOSED — all ~54 deferred findings fixed at the source; the two
measured damage_op re-exports (`POKEMON_COUNTER_OFFSET`, `_N_SECONDARY`) survive with inline
noqas naming their consumers.

**Versioning:** none — no state_dict, arch, or forward-math change. The production sha probe
reads `001e1140…` before and after (self-measured, same probe/config); mypy 0 errors; the compile
gate still traces one graph. Gate: `extractor_stashes_test.py` (stale-read unrepresentable, stray
writes loud on every property, container defaults, the layout raise).

## `gen3_static_widths_v1` (2026-08-17): the construction-time discovery forward is DELETED — widths are static arithmetic, and the old mechanism is the new mechanism's test

`Gen3FeaturesExtractor.__init__` used to MEASURE its projection-input widths by running a dummy
`forward_internal` under `self._intent_reduce_discovering = True` — zero-fill branches in the
pointer-cell blocks (`intent_move_cell`, `intent_threshold_move`, `intent_conditional`) and
skip-silently conditions in `_value_pooled_routes` let that pass complete while `alpha_head` was
still unbuilt. It was the last construction-time control flow interleaved into the runtime
forward, and the parent of a shipped bug class (ede5a88: an early `return` in a discovery branch
hid every vf part appended below it and built the critic 128 dims short, dying on the first real
forward only when two individually-tested flags met).

Since v89 (`gen3_value_pooled_routes_v1`) every value route injects ADDITIVELY into
`value_pooled`, so no width is emergent. The widths are now the pure module-level
`compute_projection_widths(layout, opp_belief_cls_k=…, damage_op=…)`: pi = 3·D_MODEL +
`non_matchup_rest` + k·D_MODEL; vf = D_MODEL + `non_matchup_rest` + k·D_MODEL +
(`VALUE_SEED_K·VALUE_SEED_DIM` iff the op). Only THREE inputs move a width — the layout's
scalar tail, the hidden-opp belief pool, and the op's seed window; a construction-time assert
ties the seed term to the built `MultiSeedValueReadout.out_dim`. The flag, the dummy-forward
invocation, all three zero-fill arms and all three skip-silently conditions are DELETED; the
runtime RAISE guards ("on but inputs missing" is loud, never a silent null) stay, now
unconditional. Module creation order is UNCHANGED (the SB3 positional-optimizer hazard), and the
dummy forward consumed no RNG (dropout 0 everywhere), so deletion is byte-neutral.

The old mechanism survives as the verifier: `projection_width_test.py` sweeps production
(`designs/production_config.json` via the `ARCH_ARG_KEYS` recipe), all-routes-on
(`value_route_gradient_test._ALL_ROUTES_ON`), minimal, and 9 targeted combos (belief pool at
k=3 and k=6, op on/off, `opp_belief_slots`, `value_entity_pool[_full]` with no op,
`value_threat_inject`, `history_events`, the full intent-cell stack), building each, running a
REAL forward, and asserting the measured concat widths equal the arithmetic and the built
`pre_proj_norm`/`projection`/`value_pre_norm`/`value_projection` shapes. A wrong width for any
combo fails in the suite, not at a production launch.

`restore_identity_init`'s M1 capture is untouched — `_identity_init_zeroed` scans weights
statically at the end of `__init__` and never depended on the forward having run.

**Versioning:** none — no state_dict, arch, or forward-math change. The production sha probe
reads `001e1140…` before and after (self-measured, same probe/config).

## `gen3_dead_kwarg_tripwire_v1` (2026-08-17): five deleted kwargs that never got a verdict — and the guard that makes the next one red

A kwarg deleted from `Gen3FeaturesExtractor.__init__` stays pickled in every archived checkpoint's
`policy_kwargs["features_extractor_kwargs"]`, and SB3 rebuilds the extractor from the ZIP rather
than from `model_config.json`. `snapshot.sanitize_dead_extractor_kwargs` handles that — INERT names
pop silently, JUDGED names REFUSE on a value the surviving forward cannot reproduce — but it is a
CURATED list, so it is exactly as complete as whoever last deleted a flag remembered to make it.
Nothing read the constructor's real signature, so a forgotten name produced no error at deletion
time, no failing test, and no warning; only a bare `TypeError` months later, on the three paths
(training resume, frozen pool opponents, eval workers) whose entire job is to sanitize-or-refuse
with judgment.

MEASURED over the 89 runs under `models/` carrying a checkpoint (via the prober's
`_dropped_extractor_kwargs`, pure set math over the live signature): 23 distinct rejected kwarg
names, **five in neither list**, present on **70 of 89 runs**. All five are JUDGED, because each is
the v71 shape exactly — a forward-behavioural toggle whose state_dict is byte-identical across its
values, or (one case) whose values disagree about whether a head's PARAMETERS exist:

* `mask_incoming_damage_obs` (61 runs), `mask_active_move_scalars_obs` / `mask_move_effects_obs`
  (58) — v48 `gen3_cpu_damage_deleted_v1`. The three `--unified-obs` ablation masks ZEROED an obs
  region out of the model's view; the source called them "an ablation toggle (no weight-shape
  change)". `False` = the region is read live, which the surviving forward still does → pops.
* `hp_type_belief_mode` (51) — v52 `gen3_typed_hp_belief_v1`. The tri-state `off|prior|learned` is
  gone; `HPTypeBelief` is UNCONDITIONAL under a move belief. `'learned'` is the only value whose
  state_dict carries that head (and the only one any archived run records) → pops; `'off'`/`'prior'`
  name a head-less forward the surviving code cannot build.
* `spread_belief_nature_marginalize` (55) — v66. The op marginalised P(KO) over the nature posterior
  instead of evaluating at its mode; no parameters either way, and `DamageOperator._nature_marg_ko`
  is deleted. `False` = mean-field, which the op still computes → pops.

`_migrate_config` needs NO matching entries, and that asymmetry is the reason the gap opened: all
five left the constructor below `MIGRATION_FLOOR`, so the CONFIG half collapsed into the blanket
PRE-GENERATION refusal, while the ZIP half — which carries no `config_version` for any floor to
apply to — kept needing a per-field judgment nobody wrote. Pinned by
`dead_kwargs_sanitize_test::test_pre_floor_fields_need_no_migrate_config_entry`.

Effect on the archive, self-measured before and after: **7 runs (`ai_v9_01`–`ai_v9_07`, all on
`spread_belief_nature_marginalize`) went from a bare TypeError to a clean load**; 69 still get the
judged `ModelVersionError` naming the git_hash to re-probe from, which is correct — they trained
under forwards this tree does not contain. 0 of 89 now TypeError. Note the gap between 70 carrying
an uncovered name and 7 surfacing it: the other 63 were masked only because a DIFFERENT curated
entry happened to refuse them first. That is luck, not coverage — delete one JUDGED entry and the
rest surface — and it is the argument for a tripwire rather than for patching the seven.

THE GUARD (`ctor_kwarg_snapshot_test.py`) is the durable half: a committed snapshot of the live
`__init__` kwarg set, so a REMOVED name fails red with the four-step instruction (judge it →
`_DEAD_FEK_*` → `_migrate_config` if at/above the floor → then update the snapshot), and a
resurrected name fails too (the sanitizer would silently strip a LIVE argument and re-default it on
every load). Same tripwire pattern as the delivery-graph gate: it converts "someone forgot" from
silent into red. The prober's second sanitizer STAYS and is not delegated to — it never refuses, by
design, because reading an archived model may be approximate as long as it says so.

**Versioning:** none — no state_dict, arch, or forward-math change. Production sha probe
`001e1140…` before and after (self-measured, same probe/config).

## `gen3_flag_requires_v1` (2026-08-17): flag dependencies become registry DATA, enforced in both directions

`flag_registry.py` was the single declaration of every extractor toggle across five hand-synced
surfaces — but a flag's DEPENDENCIES lived only as ~30 hand-written `raise ValueError` lines inside
`Gen3FeaturesExtractor.__init__`. Nothing outside that function knew them: `main.checkargs` could
not warn about an unsatisfiable recorded command, `designs/flag_registry.md` could not show the
graph, and "what is the minimum config that turns X on?" meant reading the constructor.

`ModelFlag.requires: tuple[str, ...]` is that data — the flags that must be ENABLED for this one to
be. `is_enabled` defines both ends (`False` / `0` / `'off'` / `'none'` are OFF) and is deliberately
NOT `bool()`: a mode string's off state is the truthy `'off'`, and reading it as enabled is a bug
this tree has already shipped once (the dead-kwarg sanitizer refused every OFF-mode checkpoint until
it stopped testing truthiness). `requirement_closure(name)` walks it transitively. **24 of 44**
toggles declare a dependency; the deepest closure is `intent_conditional`'s eight.

`flag_requires_test.py` enforces BOTH directions, because a declaration nothing checks is a comment
and a check nothing declares is invisible. FORWARD: for each declared pair, build the extractor with
the flag on plus its full closure and the one dependency off, and demand a `ValueError` naming that
dependency — plus a POSITIVE control that the closure-satisfied config actually BUILDS, which is the
half that catches an incomplete declaration. REVERSE: AST-scan `__init__` and collect every `raise`
guarded on two or more registry flags; each must be declared, or listed in `BESPOKE_COUPLINGS` with
a reason. All three mutations verified failing on revert (drop a `requires` entry; delete a ctor
raise; add a stale exemption).

The positive control EARNED ITS KEEP ON THE FIRST RUN, like the registry test before it: it found
that `value_dist_mode != 'none'` also needs `value_dist_vmax > value_dist_vmin`, enforced inside
`ValueDistHead` where neither `requires` nor a `__init__` scan can see it. It is a magnitude
RELATION between two numbers, not a switch, so it is recorded in the test's `_VALUE_RELATIONS` table
rather than forced into a shape `requires` cannot express.

The bespoke carve-out is one flag, not a category. Only `edge_bias_families` is exempt: its 17
family letters each carry their own requirement (most need `damage_op`, `d1/s1/c1/c2` also need
`damage_outgoing`, `d3/s3` need `entity_topk_seats > 0`, `r` needs `history_events`, `h` needs
nothing), so no flag-level statement about it is true. `damage_op` is NOT exempt — it declares
`move_belief_mode` while the constructor keeps the stronger `in {revealed, both}`, on the principle
that a weaker truth in the registry beats a blank. A stale exemption fails too.

Downstream: the generated `designs/flag_registry.md` grows a `requires` column plus the closure and
bespoke notes, and `python -m main.checkargs` reads the graph — a recorded command that enables a
flag while explicitly disabling one of its dependencies is now reported offline (exit 1, alongside
the unknown-flag half) instead of crashing the child ~40 s into a launch. It fires only on an
EXPLICIT negation: a resume inherits every unspecified flag from the checkpoint's config, so an
omitted dependency carries no information and reporting it would make the tool cry wolf.

**Versioning:** none — validation only, no state_dict, arch, or forward-math change; every valid
config builds exactly as before. Production sha probe `001e1140…` before and after (self-measured,
same probe/config).

---

### v90 — `gen3_frame_deletion_v1`: the TurnDelta lag frames and the prev-turn action mask are DELETED

**Obs 3529 → 2437** (−1092: the 1124 tail dims out, 32 new event-window dims in).
`ARCH_SIGNATURE` `gen3_ctx_dedup_v1` → **`gen3_frame_deletion_v1`**, `MIGRATION_FLOOR` 76 → 90.
Fresh weights: no pre-v90 checkpoint loads.

**The licence.** Gen-13.5 §4 measured the frames' critic dependence against the H-B event seats
that were built to replace them: `event_seats` dV **2.7714** vs `frames` **1.3015**, ratio 0.47
(`gen13_frames_arm_section4.json`, n=6000, falsified instrument — positive control + exact-zero
null arm). The seats carry roughly twice the dependence of the block they were meant to supersede,
so the frames were kept as a second, weaker copy of a job already being done.

**What went, precisely.** Two obs blocks at the END of the vector (so no offset moved):
the 11-dim prev-turn action mask and the 7 × 159 TurnDelta lag frames. With them:
`N_HISTORY_TURNS`; `TurnDeltaEncoder`'s obs role (the module survives — the prober decodes archived
runs with it); `EpisodeTracker.prev_N_delta_vecs` / `_encode_delta_slot` / the memoized
`_hist_vec_cache` / `prev_mask`; `Embeddings.embed_delta_slot`; `TeamTransformer.history_proj` and
`turn_history_pos_emb` and the 7 HISTORY seats (trunk tokens 20 → 13); `turn_delta_embed_dim`;
`TD_STRATEGIC_DIM`/`_OFFSET`; the `ModelVersion.n_history_turns` field; and the layout's
`turn_history_offset` / `turn_history_dim` / `n_history_turns` / `turn_delta_dim` / `prev_mask_dim`
keys. **`TurnDelta` itself STAYS** — it is the reward manager's per-decision input (~25 terms), the
reward tracker's fold, the battle recorder's source and the α/β intent label. Only its obs
encoding died.

**Two facts had no substitute. One was closed, one was not, and the difference is the point.**
A dV reading says whether the trained model LEANS on a block; it says nothing about whether each
fact in it has a home elsewhere. Auditing the coverage probes field-by-field against the 19 event
columns found two:

1. **`cant_reason` — CLOSED.** "This mon could not move, and why" (full paralysis / sleep / flinch
   / recharge) reached the model only through the frames. `EventKind.CANT` was already in the
   battle event log *with its reason*, and `TurnDelta` already folded it — but `EventWindowTracker`
   emitted nine event types and CANT was not one. Now it emits `EVENT_T_CANT` (vocabulary 10 → 11)
   with the reason in a NEW column 19 `cant_id` (`EVENT_TOKEN_DIM` 19 → 20), a 1-based index into
   `gen3_effects.CANT_REASONS` via the new `cant_reason_id()`, which shares `normalize_cant_reason`
   with the existing one-hot so the vocabulary tripwire stays single-sourced. `EventSeats` gains a
   `cant_emb` sized `CANT_DIM + 1` from that same vocabulary, so adding a gen3 cant reason widens
   both sides at once instead of silently clamping. It gets its OWN column rather than riding
   `status_id`: the two are mutually exclusive by `type_id`, so overloading would encode compactly
   and read wrongly, and a consumer that forgot the type check would take a cant reason for a
   status.
2. **`our_attempted_switch_spec` — NOT closed, knowingly.** Which bench mon a refused switch was
   aimed at. This one is structural rather than an omission: `Gen3Battle.record_choice_rejected`
   records that the attempted target "is not on the wire and is recovered at fold time from the
   action index", and the event window folds from EVENTS ALONE — closing it would change that
   tracker's contract, not add a missing row. What survives is the rejection FACT
   (`EVENT_T_SWITCH_REJECTED`) and trappedness itself (per-mon slots, `gen3_entity_rehome_v1`);
   what is lost is the identity of the refused target.

**One feature was RE-SOURCED rather than deleted.** The role encoder's per-move-slot validity read
`ctx.move_mask` — the PREVIOUS turn's legality in SORTED-BY-ID order — while the move slots it
gates are REQUEST-order aligned (`gen3_op_move_align_v1`). Stale *and* misindexed; the damage op
had already abandoned it for `our_active_req_move_legal` ("a stale + misordered gate") and left
this consumer behind. It now reads that same tensor: identical `[B, 4]` shape, current-decision
choosability, correctly aligned. `switch_validity` and `struggle_from_prev` are deleted outright
(role input narrows by 2) — current switch legality rides the Dict obs `action_mask`, and
forced-Struggle is "every `active_req_moves` legal bit is zero", the same derivation that already
retired the `forced_struggle` scalar.

**Tests were REPOINTED, not dropped, wherever the claim survived the block.** The typed-Hidden-Power
fuzz check (a known GIGO class — a typed HP collapsing to the bare 237 num is what made the
opponent's HP read as "immune") now reads the event window's move-num column instead of the frames.
The trapping-signals end-to-end check now asserts an `EVENT_T_SWITCH_REJECTED` row exists whenever
`delta.attempted_switch_rejected`, and carries an explicit note about the target identity it can no
longer check. `phase_modules_test` gained positive assertions that `total_dim == base_dim`, that the
five deleted layout keys are ABSENT, and that `ctx` no longer exposes `move_mask`/`switch_mask`/
`struggle_mask` — a deletion asserted is a deletion that cannot silently come back.

---

### v91 — `gen3_event_semantics_v1`: the last two coverage gaps close, and the residual-attribution GIGO that the audit walked into

**Obs 2437 → 2501** (+64 = 32 event rows × 2 new columns). `ARCH_SIGNATURE`
`gen3_frame_deletion_v1` → **`gen3_event_semantics_v1`**, config 90 → 91, `MIGRATION_FLOOR` → 91.

**1. THE BUG — the event window mis-attributed residual damage to the attacking move.**
Found while auditing the coverage gaps, not by any test. `EventWindowTracker` decided "this DAMAGE
is NOT the move's own hit" by testing `e.value.get("from")`. On a DAMAGE event the parser stores
the `[from]` clause under `value["reason"]` instead, so the raw key was **always absent** and the
guard **never fired**: every sandstorm / burn / poison / Leech Seed / recoil tick landing on the
move's target that turn was folded into the move's attributed magnitude. Measured **−0.3625 for a
−0.3000 hit** under sandstorm alone. Shipped in v81 and **trained through gen-13 and gen-14**.

The generator is worth more than the instance: the same `[from]` concept is stored under **two
different keys depending on event kind** — DAMAGE/HEAL/SETHP/STATUS write `value["reason"]`, while
ITEM/ENDITEM/WEATHER/effect kinds merge the cause dict so it lands in `value["from"]`. There were
two accessors and *neither was safe without knowing the kind*; both return None for half the
kinds, silently. `BattleEvent.from_clause` is the new kind-agnostic reader and the raw `from_cause`
now carries a warning. `TurnView` had it right all along (`turn_view.py` documents "stored as
`event.reason`"), which is why the reward path and `TurnDelta` were never affected — the damage was
confined to the H-B magnitude column the event seats read. Gate:
`event_window_test::test_residual_damage_is_not_folded_into_the_move_magnitude`, asserted on the
ARITHMETIC so a refactor that reaches for the wrong key again fails regardless of phrasing —
VERIFIED failing on revert (−0.425 with two residuals).

**2. `faint_cause_id` (column 20) — WHY a mon died.** The lag frames carried an 8-way multi-hot;
the `EVENT_T_FAINT` row had no cause column. The "a sequence makes it inferable" argument only
covers {attack, recoil, selfko} — **weather, status, hazard and Leech Seed deaths emit no preceding
event to infer from, because residual damage is not an event**, and that non-inferable half is
exactly the slow-attrition class the C6/§7 stall work keeps flagging. The tracker now records what
last damaged each side (cleared on switch-in, so a fresh mon inherits no chip history) and
classifies through `turn_view._classify_faint_cause` — the SAME function the TurnDelta fold uses,
so the two encodings cannot drift on what "weather" means. `faint_cause_id` lives in `turn_view`
beside the vocabulary and `_FAINT_CAUSE_TO_IDX` it indexes.

**3. `item_transition` (column 21) — an ENUM, not a flag.** `EVENT_T_ITEM_REVEAL` conflated
"revealed" with "gone". Gen3 has **three** ways an item stops being held and they mean different
things to a player: a CONSUMED berry is spent by its own trigger, a Knock Off REMOVAL is permanent
in ADV (unlike later gens), and a Trick/Thief/Covet SWAP means the OPPONENT now holds it — which is
information about their set, not only ours. A bare `consumed` flag would have left the conflation
half-alive, so the column is `revealed / consumed / removed / swapped`.

Both columns are embedding-routed, sized `+1` from the same vocabularies the encoder writes ids
from, so extending either widens producer and consumer together instead of clamping a new id onto
an existing row. Both verified to move BOTH heads, including the distinctions that motivated them
(hazard vs weather; removed vs swapped). The delivery graph diff is **meta-only** — no node or edge
moved, which is the right signature for columns added inside a block an existing seat already
reads.

**What this closes.** Of the four facts the frame-deletion coverage audit found with no
event-window home, three are now closed (`cant_reason` at v90, these two here) and one is ACCEPTED
on value grounds (`our_attempted_switch_spec` — and its "structurally unreachable" framing was
itself corrected: the fact IS available at emission, so payload enrichment would close it if its
value ever materialises). Ten of the 22 strict-xfail coverage probes flipped to passing, which is
what strict xfail is for — the remaining 12 are the single-row translator limit
(`delta_to_event_rows`), not obs gaps.

**Same pass — `gen3_damp_cant_v1` (register §3.7): the ability-sourced cant.** `ability: Damp`
blocks Explosion / Self-Destruct and emits a `|cant|`. It was absent from the cant vocabulary, and
`normalize_cant_reason` is crash-don't-drop, so the first blocked Explosion raised out of
`state_encoder.encode` and killed the episode — and, in training, the run. Damp is gen3-legal
(Quagsire, Golduck, Politoed, the Horsea and Paras lines) and Explosion is ubiquitous in gen3ou.

**A second defect rode the same row and a bare reason-add would have shipped it.** Showdown files
an ability-sourced cant against the ability HOLDER with the BLOCKED move as its argument —
`|cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax` — which at face value says
Quagsire could not use a move it never had, while the side that really lost its turn goes
unmentioned. The `[of]` mon is now resolved at EMISSION (`gen3_battle`, where the ident is still
resolvable) and preferred by the fold: the log gains the fact and the fold stays a pure function of
it, which is the same enrichment pattern the refused-switch-target gap would take if its value ever
materialised.

**And a third that the fix itself exposed.** `EventSeats.cant_emb` was sized `CANT_DIM + 1` = 13
rows; `damp`'s live id is 13, so it would have **clamped onto 12 = `truant`** — every blocked
Explosion silently read as loafing. Sized from `CANT_DIM_LIVE` now.

**Why `damp` is NOT in `CANT_REASONS`.** That tuple sizes `CANT_DIM`, which sizes `TURN_DELTA_DIM`
(159) — the lag-frame width **79 archived runs recorded**. The frames are deleted from the live
obs, so `TurnDeltaEncoder` is now purely the prober's decoder for that archive, and growing the
tuple would shift every offset after the cant block and make it mis-slice history *silently*, since
it would still return a plausible dict. So the archive vocabulary is FROZEN and
`CANT_REASONS_LIVE = CANT_REASONS + ("damp",)` carries the live path; the frozen one-hot refuses a
live-only reason loudly rather than mis-encoding it.
`event_window_test::test_the_archive_cant_vocabulary_is_FROZEN` makes the split enforced rather
than intended, and pins live ⊇ archive *in order* so nobody reorders the archive either.
## v92 — `gen3_td_consistency_aux_v1` (2026-08-17): the critic gets told that adjacent states are adjacent

**Ledger C5, rung 2's build half.** Pre-registration:
[`research_state/levers/td_consistency_aux.md`](research_state/levers/td_consistency_aux.md) —
unchanged by this entry, and it stays that way; its rung-2 gates are frozen.

The critic's only training signal is a per-state regression, `MSE(V(s_t), G_t)`. That pins each
state's LEVEL and says nothing whatever about the DIFFERENCE between two adjacent states, so
independent per-state noise ε arrives in `ΔV` at `2·Var(ε)` — precisely where the truth is nearly
constant. C4 measured it: self-KO ΔV RMSE 4.95 against a constant predictor at 1.33. And because
GAE reads ΔV, that dispersion is injected advantage noise on **every** transition, not only the
dramatic ones. Rung 1 (offline, frozen tokens, same population as the C4 gate) met its
pre-registered gate at λ=1.0 and λ=3.0 by adding the Bellman identity the critic already owes:

```
loss += λ · mean_pairs[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]
```

This ships the live-training half as `--td-aux-coef` (default 0.0 = OFF, loss byte-identical).
Both residual ends carry gradient — the residual-gradient (Baird) form the pre-registration
specifies, whose double-sampling bias it also names as the thing to watch.

**The engineering problem is not the loss, it is the pairs.** `RolloutBuffer.get()` yields a random
permutation, so a PPO minibatch contains **no adjacent pairs at all**. `agents/training/td_aux.py`
draws them from the buffer's surviving `[n_steps, n_envs]` structure as contiguous per-env runs
(512 rows in runs of 16), which is also the "K+1 forwards serve K pairs" economy the
pre-registration asks for — L states serve L−1 pairs, ~2× cheaper per pair than sampling pairs
independently, and rung 1 found segment batching beat a random-permutation control by 12% anyway.
Four correctness details, each of which fails silently if got wrong: the rows follow
`swap_and_flatten`'s ENV-MAJOR convention (`row = env·n_steps + t`) and `_td_aux_term` RAISES if the
buffer is not yet flattened rather than mis-pairing states with rewards; `rewards` and
`episode_starts` are not in `get()`'s flatten list and are read in their native shape; a pair whose
successor BEGINS an episode is **dropped, never zeroed** (zeroing would train `V(s_t) → r_t` at
every battle end — and it disposes of SB3's time-limit bootstrap for free, since that row's
successor always starts an episode); and the value comes from `policy.predict_values`, never a
hand-rolled path, because that method is what routes to the DISTRIBUTIONAL head's mean under
`--value-from-dist`, where the scalar `value_net` is frozen.

**Units.** `predict_values` returns real-unit values and buffer rewards are real-unit, so the raw
residual is real-unit — but under PopArt the value loss trains in normalized space, so the residual
is divided by σ. That is exactly the normalized-space residual
(`normalize(V) − normalize(r + γV′) = (V − r − γV′)/σ`, the μ cancels), so λ keeps rung-1's
calibrated meaning in both regimes.

**It folds per MINIBATCH, with its own sample and its own forward** — the search-teacher / OPD shape,
not the once-per-`train()` probe shape. Those probes are read-only; this one carries gradient, and a
once-per-`train()` fold would give it ONE contribution against the value loss's ~240, so λ would
have to be ~240× the pre-registered band to mean the same thing. Cost is bounded by the 512-state
constant rather than by `batch_size`: ≈10% of the train step at production shapes.

Metrics under `td_aux/`: `resid_rms` (the headline, should FALL — the live counterpart of the
offline dispersion instrument), `resid_mean` (SIGNED — rung 1 says this is dispersion suppression,
so a drifting bias means the residual-gradient term is moving the LEVEL instead), plus `loss`,
`n_pairs`, `scale` and `pair_drop_frac`. The trunk pull rides `grad/td_aux_share` /
`grad/td_aux_policy_cosine`; it reaches the trunk through the critic path only.

**Versioning:** `MODEL_CONFIG_VERSION` 91 → **92**, adding `td_aux_coef` — a `training_coef`-class
field, so it is recorded for provenance and for flagless-resume read-back (`_resolve`) and is
compared by NOTHING: not `check_compatible`, not any `check_*`. No `ARCH_SIGNATURE` bump; the
extractor is untouched, which is also why the flag is deliberately absent from
`agents/model/flag_registry.py` (that registry's scope is extractor toggles). A pre-v92 config
defaults it to 0.0. Not yet run: the pre-registered rung-2 fork A/B (λ=1.0 and λ=3.0) is a separate
decision.

---

### v93 — `gen3_pair_outcome_v1` (2026-08-17): the UNIFIED per-pair OUTCOME VECTOR (Phase A)

**Opt-in, `--pair-outcome-cell`, OFF in production.** Built unconditionally as zero-init machinery
under the owner's programme sequencing (`research_state/README.md`, "substrate before flywheel");
ENABLING waits on the exploiter gates. Phase A is the pointer-MOVE-cell half only — the switch cell
and every β-conditioned cell are Phase B and are deliberately not built.

**What it closes is a CURRENCY failure, one level below the reduction failure.** Traced to the sink
that actually decides, `design_pair_reduction.md` §2.1 states it exactly: the pointer switch cell is
fifteen numbers — ten damage, `p_outspeed`, `provenance`, and the Choice-Band tail — with **no
status coordinate in any currency at all**. In production `threat_status_refine` is `False`, so
incoming status reaches the policy ONLY through the `s3` edge family, i.e. as a softmax-normalised
**ratio**. "They'll click Will-O-Wisp, so bring the Natural Cure mon" is therefore unrepresentable
not because status was mis-reduced but because the two quantities never appear in the same vector in
the same units, and no reducer however expressive repairs that. It is also the most economical
reading of the **G1 n=299 null**: a 2800-dim SKYLINE over the un-collapsed pair grid could not beat
the collapsed summary (R0 0.403±0.034 · SKYLINE 0.413±0.037), which says the quantity the decision
turns on was never in the grid. So this entry does the thing §2.1 says binds FIRST — fix what the
message CARRIES, then reduce it.

**The vector.** `pair_in[their believed seat k, our mon j, :]`, width `_PAIR_OUTCOME_RAW` = **14**:
the op's six existing damage channels (`[low, high, crit, ko_ramp, acc, is_phys]`, its
`last_pair_cells`, unchanged) concatenated with eight new ones. Damage and status were computed in
two functions with two reductions, and **one α cannot weight two tensors** — the unification IS
component 1 of `design_opponent_intent.md` §5.1, not cleanup.

| # | coordinate | how it is computed |
|---|---|---|
| 6-11 | `p_par p_brn p_frz p_slp p_psn p_tox` | `_incoming_status_lands` VERBATIM (the oracle-gated per-pivot immunity physics — type at our defender's types, ability block, already-statused, damage-gated secondaries) SPLIT by the seat's status IDENTITY. The identity is `MOVE_STATUS_IDENT` (NEW: a one-hot built from the raw `status_inflicted`, so **Toxic and Poison Powder stay apart** — `MOVE_STATUS_CAT` folds both into category 5 because they share the Steel/Poison immunity) for a dedicated status move, and `MOVE_SECONDARY`'s L1-normalised major prefix for a damaging move's secondary. Exact in practice: every gen3 move with a major secondary has exactly one |
| 12 | `neutralization` | `Σ_s p_s · sev_s(j)` in units of *fraction of this mon's per-turn contribution destroyed*. `sev_brn = 0.5 · base_atk/(base_atk+base_spa)`; `sev_par = 0.25 + 0.75·Δp_outspeed` where Δ is the op's OWN outspeed logistic re-evaluated at ×0.25 speed; `sev_frz = sev_slp = 1.0`; `sev_psn = 1/8`, `sev_tox = 1/16`. **Every scalar is a gen3 RULE**, never a tuned prior — that is what lets it ship without a calibration artifact |
| 13 | `tempo_cost` | `P(any major status) × undo_turns(j)`. `undo_turns` = 1 for a cure move (NEW `MOVE_CURES_SELF_STATUS`: Refresh / Heal Bell / Aromatherapy — the facade keeps `cures_self` and `cures_team` apart by SCOPE; here only "is this mon clean afterwards" matters), the op's own `rest_sleep_noeb` (2.0, derived from the verified sleep hazard table) for Rest, else 0. The receiver is OUR mon, so its moveset is exact and there is no marginalisation question on that axis |

**Every new coordinate passes §9a's admission test** (*name two specific actions whose ordering it
flips*), and the answers are recorded in `pair_outcome.py`'s docstring rather than asserted:
the status columns flip **Swords Dance vs Earthquake** under a believed Spore (setup is worthless if
you may never act again; the ten damage numbers are IDENTICAL in both branches because Spore deals
none); `neutralization` flips **Substitute vs Calm Mind** on a physical attacker facing a likely
Will-O-Wisp (damage-only reads BOTH branches at 0.0); `tempo_cost` flips **Substitute vs Toxic** (a
sub spends a turn NOW to prevent a turn spent LATER — without the coordinate, "it lands and I cure
it" and "it never lands" are the same state, so the sub is never worth its turn).

**Two coordinates the design sketch listed are deliberately NOT shipped.** §5.1 lists
`p_status_land` and `p_immobilize`; both are **linear functions** of the six per-identity columns,
which pass through a `Linear` before anything else touches them, so §9a's derivability rule applies
in its plain form (*derivable from what IS delivered → do not add it*). The un-collapsed vector is
strictly richer — burn and sleep are not interchangeable to a physical attacker, and a single
`p_major` scalar asserts they are. `pair_outcome_test` pins the absence so a future "restoration"
has to delete a test first.

**The reduction is Contract W, enforced by SHAPE.** `reduce_pair_in(α, pair_in, gate, active)` is
`Σ_k α_k · pair_in[k, j, :]` and α carries **no channel axis and no defender axis**, so the flat
block's nine-independent-maxima incoherence (**D2** — up to nine different opponent moves describing
one defender) and a per-defender α (**D3** — Skarmory's row assuming Rock Slide while Blissey's
assumes Thunderbolt, illegitimate because they choose without seeing which mon you bring) are both
**shape errors** rather than properties a test hunts for. The load-bearing gate plants a
per-channel maximum and asserts the contract catches it; verified failing against the planted
violation.

**α, and the fallback that makes the flag independently enableable.** With `--opp-intent` on, α is
the softmax of the PUBLISHED α logits, **move slice only, UNRENORMALIZED** (the `IntentValueReduce`
/ v84 / v85 precedent — the missing SWITCH mass is the literally-correct statement that a switching
opponent applies no outcome to us this turn, so a high `α_SWITCH` shrinks every coordinate toward
zero together, a coherence only expressible because they share one α) and **stop-grad
unconditionally**: this is a POLICY-side consumer, and leaning on `belief_grad_mode=label_only` for
the cut would make the route's EXISTENCE a function of a TRAINING flag. With `--opp-intent` OFF it
falls back to the shipped **R1 `belief_mean`** rung, `α := w/Σw`, re-exported from `pair_reduce` so
the two cannot drift. ⚠️ **The fallback is not a cheap α**: `w` is a PRESENCE belief and α is a
USAGE belief — `α ≠ w` is the substantive modelling error the whole design names — and the fallback
sums to 1 where the α path sums to `1 − α_SWITCH`, because with no intent head there is no switch
belief to withhold mass for. What it buys is the separation §7a.2 asks for: the **DELIVERY** claim
(a per-action absolute in the currency the decision needs) testable apart from the **DISTRIBUTION**
claim. A seat closed by the meaningful-K gate is MASKED and its mass **not reassigned** — §4.2's "if
we can't name it, we don't train on it", applied in the forward.

**Delivery.** The reduced row for our ACTIVE defender rides every move cell through a zero-init
`Linear(14, 14)` (`PAIR_OUTCOME_MOVE_DIM`), per-action ABSOLUTE — the channel measured to work
(`d1` 12.17% / `d2` 19.25%) rather than the edge ratios the consequence families died on. The row
carries no per-slot dependence, so it rides identically on all four slots like `intent_threshold`'s
`p_ko` context channel; that is **not** a no-op, because the pointer scorer is an MLP over
`(move token ‖ cell)` — a constant cell modulates how each different token is scored, and the switch
logits (Phase B) do not receive it, so the move/switch balance moves too. A term that were merely
additive on the logit would cancel in the softmax; this one does not.

**Known limits, named rather than approximated:** status DURATION (`neutralization` is a per-turn
rate, so sleep and freeze read equal even though gen3 freeze does not self-thaw — an
expected-duration factor is not a rule and would have to be guessed); **physics mutation** (burning
Milotic multiplies its Def by 1.5 and moves every subsequent number in the matrix — a statement about
the successor state, out of scope for a one-ply reduction by §5.1); and a **held berry's auto-cure**
(Lum cures at zero tempo and near-zero neutralization; folding it in would PRE-BLEND two
probabilistic branches into one column, which §9's anti-patterns forbid).

**Versioning:** `MODEL_CONFIG_VERSION` 92 → **93**, adding the `structural` field
`pair_outcome_cell` with a `check_compatible` gate and a `<93 ⇒ False` migration. **No
`ARCH_SIGNATURE` bump** — the module is flag-gated and OFF builds nothing, so an existing
checkpoint's forward and `state_dict` are untouched (pinned: no module, no `state_dict` key, no
extra dim, identical pi/vf). ON is identity-at-init, asserted on a REAL `MaskablePPO` policy (ledger
M1 — the zero-init is captured by observation and re-zeroed after SB3's ortho pass). Also in this
pass, no version bump: `SECONDARY_MAJOR_N` was declared in **two** files and is now imported from
`damage_tables` by `damage_op_layout`.

**Gates:** 37 new tests in `pair_outcome_test.py` (coordinate-by-coordinate correctness through the
FULL op forward, the D2 planted-violation gate, the α fallback's exact identity with the shipped
rung, seat masking, seat-permutation invariance, OFF byte-identity, real-policy identity-at-init,
the version machinery, and the delivery-graph edges) plus a compile cell for the **fallback** branch
— which no other cell can reach, and an untested default branch is exactly the failure the
seedless-bridge lesson records. The ON+intent branch joins the existing one-graph intent cell.

---

### v94 — `gen3_pair_outcome_switch_v1` + `gen3_switch_branch_v1` (2026-08-17): the SWITCH cell, and the switch BRANCH (Phase B)

Phase B of the conditional-mechanics substrate, in one pass because the four items are one idea:
**Phase A fixed what the message CARRIES; Phase B fixes WHERE it lands and adds the per-action,
per-defender content Phase A deliberately did not build.** Two opt-in flags, both `structural`,
both OFF in production, both byte-identical when off and identity-at-init when on.

**1. `--pair-outcome-switch` — the SWITCH cell (`gen3_pair_outcome_switch_v1`).**
`design_pair_reduction.md` §2.1's own defect, at its own sink. Phase A delivered our ACTIVE
defender's α-reduced row to the pointer MOVE cells as *context*; §2.1 says in one line where the
decision actually happens:

> The decision *"they will click Will-O-Wisp, so bring the Natural Cure mon"* is made at the
> **switch logit**. The switch logit's per-action cell contains **no status information at all**.

`reduce_pair_in_all` runs Contract W at **every** defender — `Σ_k α_k · pair_in[k, j, :]`, `[B,6,14]`
— and `PairOutcomeSwitchCell` projects mon *j*'s own row into mon *j*'s own switch cell through a
zero-init `Linear(15, 15)`. It is the **first module ever to widen the pointer switch cell** (every
earlier α consumer rides the move cells or the value tail; `_PTR_SWITCH_CELL_IN` had been 15 since
the OAX tail's deletion). One α still serves all six rows, so **D3 — the per-defender α, "Skarmory's
row assuming they click Rock Slide while Blissey's assumes Thunderbolt" — remains a SHAPE error**;
the planted violation is gated. Equivariant in our team axis by construction. It requires
`damage_op` and **not** `pair_outcome_cell`: the two deliver ONE tensor to TWO sinks, and coupling
them would make a measured result unattributable to a sink.

One extra per-defender coordinate rides with the row — `spin_denied` =
`is_ghost(our mon j) · Σ_k α_k·is_rapidspin(k) · their_side_hazards`, the **defensive half of the
Pursuit mirror**. A gen-3 Rapid Spin fails outright against a Ghost, so a Ghost switch-in is hazard
insurance. Three independent events, so it is a CONJUNCTION and not one of §9's forbidden
pre-blends of probabilistic BRANCHES; the hazard stake is what turns a fact into a value, and with
no Spikes on their side the coordinate correctly reads 0.

**§9a admission** (name two actions whose ordering it flips), both SWITCH pairs, both the design's
own: **switch Swampert vs switch Celebi** into a Gengar believed to hold Will-O-Wisp + Thunderbolt —
Swampert reads **0.0 in every damage coordinate in both branches** (Ground/Water is Electric-immune,
burn deals no damage) and so wins forever, while `neutralization` says the burn destroys half of a
physical Swampert's per-turn contribution; and **switch Starmie vs switch Milotic** into a believed
Toxic, which tie on damage AND on `p_tox` and are separated only by `tempo_cost`. For `spin_denied`:
**switch Gengar vs switch Blissey** into a Starmie believed to hold Rapid Spin with three layers up
— the damage numbers prefer Blissey, the Spikes prefer Gengar.

**2-4. `--switch-branch-cell` — OA2 + the spinblock + Protect (`gen3_switch_branch_v1`).**
`design_conditional_opponent_cells.md` §2, plus two owner-specified mechanics that are the SAME
contraction and therefore belong in the same vector: `Σ over their options of (usage probability) ×
(a property of the option)`, over the branch in which they **switch**. Gen-3 is simultaneous-move,
so `P(they switch)` is ONE scalar for the turn (§2.1) — but the CONSEQUENCE is per-move, because
switches resolve first and our move lands on the ARRIVAL, which **β** names. Nine coordinates on the
move cell through a zero-init `Linear(9, 9)`:

* **OA2** — `e_high_switch` / `e_pko_switch` / `e_mult_switch` = `Σ_j β_j · omx[k, j, ·]` from the
  outgoing matrix, plus `wasted_ko = pko_stay(k)·α_SWITCH` (§2.3's named interaction, *"don't click
  the KO into the obvious switch"*) and the shared `a_switch` scalar. §2.3's rule is followed
  literally: the branches ship **DECORRELATED** — the stay branch already rides the op's own move
  cell — and never as the collapsed `(1−p)·stay + p·switch`. §9a: **click Earthquake vs click
  Spikes** against a Skarmory at 30% that will obviously pivot (Earthquake's `pko_stay` is high,
  its `e_high_switch` against the Gengar/Zapdos arrival is ~0, Spikes lands whatever arrives).
* **Rapid Spin** — `p_spin_blocked = is_ghost(their active)·a_stay + α_SWITCH·Σ_j β_j·P(slot j is
  Ghost)`, gated to the Rapid Spin request slot, with `spin_value_lost` = that × our-side hazards.
  **The Pursuit mirror, explicitly**: v85's Pursuit is `α_SWITCH` against a property of the
  DEPARTING mon, positive valence, no β (the sim strikes before the switch resolves); this is
  `α_SWITCH` through β against a property of the ARRIVING mon, negative valence (Rapid Spin resolves
  after). Same operator, opposite sign, and the difference is a fact about gen-3 resolution order,
  not a modelling choice. In gen 3 Rapid Spin is **Normal**, so a Ghost final defender means no
  damage AND no hazard removal — both halves of the click die together, which is why one probability
  suffices. `P(slot is Ghost)` is leak-free: revealed types where revealed, the hidden-team species
  posterior through a new `SPECIES_IS_GHOST` table where not. §9a: **click Rapid Spin vs click
  Hydro Pump** with a Gengar on their bench and `α_SWITCH` high.
* **Protect** — `protect_attack_mass = Σ_k α_k·is_damaging(k)` gated to Protect/Detect, and
  `protect_blocked_mass` = that × the obs floored-doubling `p_success`. The **`c4` successor**: that
  edge carries the mechanical consecutive-use decay and never asks *will they attack*. It is
  decorrelated from, not redundant with, v85's `e_dmg_avoided` (`Σ_k α_k·high_k`) — that is a
  MAGNITUDE where this is a MASS, and they come apart in both directions (a believed Spore has mass
  and no magnitude; a 4×-resisted Hidden Power the reverse). `is_damaging` is typed from the data
  facade, not from `high > 0`, so an immune damaging move cannot masquerade as a status move.
  `c4`'s edge family is untouched. §9a: **click Protect vs click Recover** at 60% against an
  opponent believed to be setting up.

**This flag REQUIRES `opp_intent` with NO fallback, and the asymmetry with the pair-outcome pair is
substantive rather than cautious.** The R1 `belief_mean` rung is a PRESENCE belief over their MOVES;
it has no switch class at all, so `α_SWITCH` would be identically 0 and every coordinate would
assert *"they never switch"*. A flag whose fallback silently states something false is worse than a
flag that says it needs the head. β has no prior-shaped substitute either.

**§4.1's HARD PREREQUISITE for OA2 is CLOSED, and this build depends on that.** OA2 was blocked
because v34's outgoing matrix was REVEALED-gated, so a β that correctly puts mass on unrevealed
slots would read ≈0 there — *"my move always lands on their active"*, misleading exactly when
switching matters most, the typeless-HP "immune" GIGO class. `gen3_unrevealed_outgoing_prior_v1`
prices an unrevealed slot against the EXPECTED-LATENT defender. **One residue is stated rather than
hidden:** `pko` is still NULLED at unrevealed slots by the op's owner rule (a full-HP switch-in is
~never OHKO'd), so `e_pko_switch` is systematically deflated in proportion to β's hidden mass —
which is why `e_high_switch` ships beside it and carries the magnitude there.

**Op-side producers.** Two new stashes, both behind the existing seam convention: `out_cells`
`[B,4,6,5]` (the outgoing grid un-collapsed; `out_pko` becomes a **view** of it, so the OA2
magnitudes and v85's boom pko can never describe different worlds) and `opp_p_ghost` `[B,6]` behind
`stash_opp_ghost`. Both α and β are read from the PUBLICATIONS and **stop-grad unconditionally** —
Phase A's rule, and the reason is that resting on `--belief-grad-mode label_only` makes a PPO route's
EXISTENCE a function of a TRAINING flag.

**Not modelled, named rather than approximated:** Rapid Spin also clears Leech Seed and partial-trap
from its user, so `spin_denied` under-prices a spin blocked on a seeded opponent; a Ghost KO'd on the
switch-in denies nothing; and `tempo_cost` still reads the mon's cure MOVESET, not the Natural Cure
ABILITY — a Phase A coordinate question, deliberately not smuggled in with a delivery change.

**Versioning:** `MODEL_CONFIG_VERSION` 93 → **94**, adding the `structural` fields
`pair_outcome_switch` and `switch_branch_cell`, each with its own `check_compatible` gate and a
`<94 ⇒ False` migration. **No `ARCH_SIGNATURE` bump** — both modules are flag-gated and OFF builds
neither, so an existing checkpoint's forward and `state_dict` are untouched.

**Gates:** 61 new tests across `pair_outcome_switch_test.py` (28) and `switch_branch_test.py` (33) —
exact arithmetic on every coordinate, the D3 planted-violation gate, the two reducers' agreement at
our active row (they are one contract with two gathers, and a drift would let the move cell and the
switch cell describe two different opponents on the same turn), BOTH invariances (α's seat axis and
β's their-bench axis, §5 gate 5), the `has_cand`-zero case (a uniform arrival belief is a claim, not
an absence), the op's ghost marginal exact-where-revealed / posterior-where-not, stop-grad on both
heads **verified failing on revert**, OFF byte-identity, real-`MaskablePPO` identity-at-init, the
version machinery and the delivery-graph edges. A new explain-only compile cell covers the pair's
one-graph property; it is separate from the v93 intent cell because **measured**, adding the two
flags there took that cell 25.5 s → 73.1 s, overrunning the 31 s default-tier budget.

---

### v95 — `gen3_conditional_threat_v1` + `gen3_pair_value_route_v1` + `gen3_status_economy_v1` (2026-08-17): OA1, the critic's route, and the status economy's missing paths (Phase C)

**The last phase of the conditional-mechanics substrate.** Phase A (v93) unified the CURRENCY of
what the opponent does to us; Phase B (v94) put that unified row on the SWITCH cell and added the
β-conditioned cells; Phase C adds the coordinates the row structurally cannot carry, gives the
CRITIC a route to any of it, and closes the coordinate gap Phase A named and Phase B deliberately
did not smuggle in.

Three items, two new opt-in flags, one in-place amendment. Every flag OFF ⇒ byte-identical.

#### 1. `--conditional-threat-cell` — OA1, the CONDITIONAL THREAT CELL (`design_conditional_opponent_cells.md` §1)

*"They'll Ice Beam my Salamence; switch to the mon that eats Ice Beam."* The **second** module ever
to widen the pointer SWITCH cell (Phase B was the first), carrying four α-contracted coordinates —
`CONDITIONAL_THREAT_SWITCH_DIM` = **4**, `conditional_threat.py`:

| coordinate | what it is | why the reduced row cannot carry it |
|---|---|---|
| `e_pko_acc` | `Σ_k α_k · ko_ramp(k,j)·acc(k)` | §0.2(2): *precompute every nonlinearity of two numbers IN THE OP.* The two factors ride the row DECORRELATED and a thin `tanh` scorer does not multiply two of its own inputs |
| `e_type_mult` | `Σ_k α_k · type_mult(k,j)` | the one cell channel NOT divided by the defender's own bulk — a structural immunity (`0.0`) apart from an incidental zero, and a read that survives the mon's HP moving |
| `margin_high` / `margin_crit` | `Σ_k α_k·high(k,j) − hp_frac(j)`, and the same on the crit roll | §0.2(3): *probabilities SATURATE; ship the MARGIN too.* `pko` is flat across "barely survives" and "survives comfortably", and equally flat across "dies by 60% of its bar" and "dies by 2%" |

**Three of §1.2's five clauses were SUPERSEDED and substituted rather than built** — the design
predates both shipped phases, and the substitution table is in the module docstring so the next
reader does not re-derive a plan from the old spec:

* **`λ` and the `w = softmax(λ·threat + log belief)` weighting are NOT built.** `pair_alpha` is the
  shipped distribution over the SAME seats; a second weighting would be a **second α**, i.e. exactly
  the D2/D3 family Contract W makes a shape error, sitting beside the shipped one and disagreeing
  with it. The cell owns two parameters — `proj.weight`, `proj.bias` — and a test pins that list.
* **`high` / `pko` / `status_lands` are already delivered** by `--pair-outcome-switch` on this exact
  cell; re-emitting them is duplicated delivery (it doubles a channel's effective weight and makes
  an ablation unattributable), and `status_lands = Σ_s p_s` is additionally barred by §9a's
  derivability rule.
* **§1.3's "also turn on `--damage-matrices-outgoing-all`" is VOID** — that flag was deleted outright
  at v88 (`gen3_dead_flag_purge_v1`), its OAX block with it, and re-adding one is not this phase's
  licence.

**§5's pre-registered gates, item by item:** §5.2 (ON == OFF bitwise at init on a REAL `MaskablePPO`
policy) and §5.5 (our-bench permutation equivariance) are gated here; §5.3/§5.4 are OA2's and were
gated at v94; §5.1's constructed-marginal convention is followed on every coordinate. §5.6's B=1 CPU
delta is expected ~0 by construction — the type multiplier is a `.detach()` of a tensor
`_incoming_matrix` already built.

**New op seam:** `stash_pair_type_mult` → `OpStashes.pair_type_mult` `[B,6,K]`, stashed inside
`_incoming_matrix` at exactly α's seat alignment. It is deliberately NOT a coordinate of `pair_in`
(whose width is a contract three consumers read), and deliberately not re-derived at the consumer —
the real move-num gather and the ability fold both live in the matrix, so re-deriving it is the
`op move-order` bug class with extra steps.

Requires `damage_op` + `damage_matrices_incoming`; **not** `opp_intent` — the R1 `belief_mean`
fallback is MEANINGFUL here (every coordinate is a *what lands on me if they attack* contraction, so
the unrenormalized slice's missing SWITCH mass correctly shrinks it toward zero), unlike the v94
`switch_branch_cell` case where a fallback would have asserted *"they never switch"*. **Not**
`pair_outcome_switch` either: two quantities, one sink, attributable separately.

#### 2. `--pair-value-route` — PV, the pair-VALUE CRITIC route (`design_opponent_intent.md` §7a(2))

Every other cell in this substrate delivers through `pointer_cells`, which is **policy-only**. PV
sends the α-reduced unified row for our mon *j* to the critic as **TOKEN CONTENT on mon j's own
token**, inside `CLSPool`, on the value pool's copy only — a second zero-init `Linear(14, 128)`
beside v64's `value_threat_proj`, stacking additively and independently with it.

**What it delivers that nothing else does:** v64 sends the `pair_reduce` rung's **13-wide DAMAGE**
summary. PV sends `pair_in` — Phase A's unified **14**-coordinate row, whose last eight are the six
status identities, `neutralization` and `tempo_cost`. The critic has no other per-entity route to
any of them: incoming status reaches vf only as the `s3` edge family's softmax-normalised **RATIO**
(`design_pair_reduction.md` §2.1). The two flags are therefore not two spellings of one arm.

**Token content, NOT the v89 `_value_pooled_routes` seam — on structure, not taste.** A seam route
yields one `[B, D_MODEL]` vector added AFTER pooling, so it would have to collapse the `J` axis
itself, and the only equivariant collapse is a **sum** — which cannot tell *one mon about to lose
90% of its bar* from *six mons losing 15% each*, and the first is a losing position while the second
is a normal turn. Token content does not collapse: the row rides the token that already carries the
mon's identity, HP and typing, and `value_cls`'s attention decides the weighting (§2b.2 — *you can
only preserve an axis you have output slots for*; here the tokens ARE the slots). The cost of that
choice is stated and paid: the seam's gradient guard does not cover it by construction, so
`value_route_gradient_test.py` gained a dedicated cell asserting BOTH token-content injections
receive critic gradient under BOTH parameterizations — the guard's real claim is *every zero-init
projection the critic depends on gets gradient*, not *every seam entry does*.

⚠️ **α here is the R1 `belief_mean` rung UNCONDITIONALLY, and that is ORDERING rather than
preference.** `value_cls` pools at T2 **before** the α/β heads are scored, so the publication does
not exist at that point in the forward; this is not a fallback that fires when a head is absent, and
the gate asserts the injected rows are byte-identical with `--opp-intent` ON (plus that the two
rungs genuinely differ on that seed, so the assertion is not vacuous). §7a(2) pre-registers exactly
this substitution as the way to separate the DELIVERY claim from the DISTRIBUTION claim.

⚠️ **THE C4 RE-ENTRY CONDITION, recorded verbatim in the flag registry, the CLI help, the module
docstring and `ARCHITECTURE.md`:** *any α/β-critic route may be BUILT opt-in but its ENABLING owes
the C4-style offline gate first.* Ledger row **C6** failed 2026-08-17 with route liveness PROVEN —
all five v89 routes trained off zero and `entity_pool` carried decisively (dV 6.28 = 110% of
all-off) — while the critic's stall-loss over-confidence **did not move** (gen-13 confident-band gap
+0.358, CI [0.23, 0.50]), and the delivery line was declared EXHAUSTED. Building this is cheap and
reversible; enabling it without that gate is the thing C6 forbids.

Requires `damage_op`. **Width-neutral** (additive injection), so nothing shape-based can see it
except the extra `state_dict` key — the version gate is the only thing that rejects a mismatched
resume, which is why it has one.

**Its own audit arm ships with it.** `critic_route_audit` gains `pair_value` (zeroing the rows on
their way into `CLSPool`, exactly as v64's `threat` arm does), included in `all_off` — so the
C4-style offline gate the re-entry condition demands can be run the moment a checkpoint carries the
route, in the same |dV| units as every other route. A nonzero KL/flip reading on that arm would
itself be a finding: it would mean the injection had leaked into `pi`.

#### 3. `gen3_status_economy_v1` — the two undo paths `tempo_cost` never had (AMENDMENT, no new flag)

Phase A read `undo_turns(j)` off mon j's **moveset only**, and Phase B's own test carried the residue
as a named limitation. Two paths were missing and both are real decisions:

* **Natural Cure is an ABILITY.** A Natural Cure mon sheds its status on switch-out — it HAS an
  answer — but Phase A priced it at **0.0**, the same number a mon with no answer at all reads. Two
  opposite facts arriving as one number is the currency failure this module exists to close, one
  level down.
* **A cleric on the BENCH is an answer for the whole team.** Heal Bell / Aromatherapy are party-wide
  in gen 3, so a statused mon that retreats behind the cleric is not stuck with it.

`undo_turns(j)` is now the **cheapest available path**, each number a gen3 rule:

| path | condition | turns | rule |
|---|---|---|---|
| self-cure move | mon j knows Refresh / Heal Bell / Aromatherapy | **1.0** | the cure consumes exactly the turn it is used on |
| **Natural Cure** | mon j's ability is Natural Cure | **1.0** | the status is shed on switch-out, and a switch consumes exactly one of our actions — the same single turn, needing no moveslot and no teammate |
| Rest | mon j knows Rest | **2.0** | the op's own `rest_sleep_noeb`, DERIVED from the verified sleep-hazard table |
| **cleric** | any OTHER ALIVE mon of ours carries a party-wide cure | **2.0** | switch to the cleric (1) + click it (1); the party-wide scope is what lets it reach mon j on the bench |
| none | — | **0.0** | nothing is spent because nothing is undone |

Three consequences, all deliberate:

1. **`min`, not the old `max(cure, rest)`.** The old form was a pick-the-nonzero idiom rather than a
   claim, and it priced a mon holding BOTH Refresh and Rest at 2.0 — the move it would not click.
2. **`0.0` cannot ALSO mean "free".** That is why Natural Cure is priced at its literal switch rather
   than at 0: reading it as free ("you were pivoting anyway") is a claim about our own POLICY, not
   about gen 3, and it would collide with the no-path sentinel besides.
3. ⚠️ **`neutralization` deliberately does NOT read the ability — the one place the literal
   instruction was declined, with the reason.** `neutralization` is a per-TURN rate; Natural Cure
   changes the status's **DURATION**, and duration is the quantity Phase A explicitly refuses to
   model without a rule to source a number from (the same refusal that leaves sleep and freeze
   equal). An ability-keyed discount would be exactly the tuned prior the `_NEUTRAL_*` block forbids.
   The PAIR identifies the case instead: `(neutralization full, tempo 1.0)` = *an answer exists at
   one turn*; `(neutralization full, tempo 0.0)` = *no answer exists*. Those were one vector before.

**The OTHER direction was checked and NOT built, as instructed.** Nothing prices OUR outgoing status
against THEIR mons in an undo currency — `_status_landing` / `discrete_outgoing_status` compute
P(lands) and nothing else — so there is no consumer for a Natural-Cure-by-ability-prior read on their
side, and building one would be delivery with no sink.

**§9a for the amendment.** *Natural Cure flips stay-and-absorb vs switch-out*: **click Protect vs
switch Starmie out** with a Toxic already on it and a setup sweeper across — every damage number,
every status probability and `neutralization` are identical in both branches, and `tempo_cost` read
0.0 in both, i.e. *"this mon has no answer, so there is nothing to gain by leaving."* *The cleric
discount flips absorb-the-Toxic vs hard-avoid*: **switch Swampert in vs switch Gengar in** against a
believed Toxic with Blissey alive on our bench holding Heal Bell — `p_tox` and `neutralization` rank
the immune Gengar first forever, while the cleric makes the Toxic on Swampert undoable at two turns
rather than never. Kill the Blissey and the board ranks the other way, because the path is gated on
the cleric being ALIVE.

New data tables: `MOVE_CURES_TEAM_STATUS` (party-wide cures only — kept APART from
`MOVE_CURES_SELF_STATUS`, which merges the two because only "is this mon clean afterwards" mattered
there) and `ABILITY_NATURAL_CURE`, resolved FAIL-LOUD at build so a data rename cannot silently make
every Natural Cure mon read as having no answer. New constants `_TEMPO_NATURAL_CURE_TURNS` = 1.0 and
`_TEMPO_CLERIC_TURNS` = 2.0, beside the existing block and under the same rule (a gen3 rule, never a
tuned prior).

#### Versioning

`MODEL_CONFIG_VERSION` 94 → **95**, adding the `structural` fields `conditional_threat_cell` and
`pair_value_route`, each with its own `check_compatible` gate and a `<95 ⇒ False` migration.
**No `ARCH_SIGNATURE` bump** — both modules are flag-gated and OFF builds neither.

⚠️ **But the v95 migration REFUSES a `<95` config that recorded `pair_outcome_cell` or
`pair_outcome_switch` ON**, because item 3 amends `tempo_cost`'s coordinate semantics under an
EXISTING flag: such a checkpoint's weights are unchanged but were trained against different numbers,
so it is re-read from its own `git_hash` rather than migrated (the v75 rule). No such checkpoint
exists — neither flag has ever been enabled in a run, which is exactly why an in-place amendment was
the right shape — so this is a latent guard, not a migration path.

#### Gates

**52 new tests.** `conditional_threat_test.py` (26): the coordinate table, the exact-arithmetic
§9a case for `e_pko_acc` (α = ½/½ over {Blizzard 70% acc OHKOing mon 0, Thunderbolt 100% OHKOing mon
1} — both mons read `Σα·ko_ramp` = 0.5 AND `Σα·acc` = 0.85, while P(dies) is **0.35** vs **0.50**),
the immunity case for `e_type_mult` (α = (0.25, 0.75) over (2.0, 0.5) = **0.875**; **0.0** for the
immune mon), both margins at both ends of the saturation (`+0.05` / `−0.32` and `+0.45` / `+0.08`),
the gate, the **planted D3 violation** (a per-defender α is a shape error), the two seat-axis
fail-louds, seat-permutation invariance and our-bench permutation EQUIvariance, the op seam
(Ground-vs-Electric **2.0**, Ground-vs-Flying exactly **0.0**), OFF byte-identity, ON contributing
exactly zero at init, the stacking order with Phase B, `requires=`, the fail-loud-not-zeros path, the
version machinery and the delivery-graph edges. `pair_value_route_test.py` (21): the module's
zero-init and shared-Linear equivariance, both fail-louds, the pool's refusal to skip silently, the
injected row being `reduce_pair_in_all` under R1 **exactly**, **the ordering claim** (rows
byte-identical with `--opp-intent` ON, and the two rungs proven to differ), pi bit-identity at an
ARBITRARY weight (including the pointer head's team tokens), critic gradient under both
parameterizations, stacking with `--value-threat-inject`, width-neutrality, real-`MaskablePPO`
identity-at-init, the version machinery and the graph edge. `pair_outcome_test.py` (+6) and
`pair_outcome_switch_test.py` (+1) cover the status economy on exact products of `p_major ×
undo_turns`, including *"bring the Natural Cure mon"* end to end.

A new explain-only compile cell (`test_phase_c_conditional_threat_and_pair_value_route_compile_to_
one_graph`) covers both flags plus Phase B's pair together — separate from the shared production
cell for the reason v94 measured (adding flags there took it 25.5 s → 73.1 s), and stacked because
the switch cell's `torch.cat` now runs twice on one path and `CLSPool` now chains two token-content
adds, neither of which was reachable before. mypy 0.

---

## v96 — `gen3_critic_route_wave_v1` (2026-08-18): the critic-route deletion wave — seven audited-dead routes, and the whole vf tail with them

**Seven items, one pass, every one of them registered no-appeal before the number existed.** The
gen-14 end-of-run battery (`measurements/gen14_route_audit_12391.json`, n=12,391 stratified states
off `ai_v9_16_gen14_framedel_v91_0817/final_model.zip`) closed the critic-route consolidation the
cleanup journey opened at Phase 3. What survives is `value_entity_pool` (dV **5.490** = 97% of
`all_off` 5.635), `threat` (**1.0686**, its registered deadline discharged) and the PV shelf.

| deleted | measured | verdict |
|---|---|---|
| the v61 `MultiSeedValueReadout` + `seed_diagnostics` + the `value_seeds/*` TB contract | **0.0000 bit-exact**, gen-13 AND gen-14 | dead twice |
| the `hidden_opp_belief` **VF half only** | dV **0.0000** (its PI half: KL 0.7396, flips **39.6%**) | dead half deleted, live half KEPT |
| the `non_matchup_rest` **VF concat** | dV **0.0000** (C1: the content substitutes through the global token) | dead half deleted, pi concat KEPT |
| `value_intent` | 0.156 vs a 0.39 bar | NULL |
| `intent_threshold`'s p_KO **vf route** (`IntentThresholdValue`) | 0.155 (gen-13), 0.136 (gen-14) | NULL — the POLICY move cell KEPT |
| `intent_value_reduce` | **0.3176** at 2× sample | below bar twice |
| `value_clock` | **0.2169** at 2× sample | below bar twice |

**The structural payoff is bigger than the sum of the parts: `vf_combined` IS `value_pooled`.** All
three unconditional deletions were members of the post-assembler vf tail, so removing them leaves
the value head reading exactly one tensor — the same one `--value-from-dist`'s dist-head critic
reads. That makes the **v89/M2 orphaned-branch bug class unrepresentable** rather than merely
fixed: there is no longer a second vf path for a critic parameterization to bypass. `vf` becomes a
CONSTANT `D_MODEL` in `compute_projection_widths` (no flag can move it), `ProjectionAssembler`
holds **zero parameters**, and every critic enrichment now goes through one of two declared seams —
`_value_pooled_routes` (additive, gradient-guarded) or the two `CLSPool` token-content injections.

**Measured on the live gen-15 production config** (`tmp/wave_byte_identity.py`, a real
`MaskablePPO` build so SB3's ortho pass and `restore_identity_init` both run): **2,611,948 →
2,071,162 parameters, −540,786 (−20.7%)**; 15 `state_dict` keys deleted (`assembler.seed_readout.*`
×3 and `intent_threshold_value.proj.*` ×2, over the three extractor aliases) and
`value_projection.in_features` **1177 → 128**. With the surviving parameters carried across
verbatim, the policy logits and the critic value are **BYTE-IDENTICAL** (sha `fdeefba75e34de5e` /
`8ffe3e2883c80555` before and after). That is the honest form of the OFF-equivalence claim here:
gen-15 records all three deleted FLAGS off, and the unconditional deletions were structurally dead
under `value_from_dist`.

**⚠️ `value_intent`'s RE-ENTRY CONDITION SURVIVES ITS DELETION** and is written into three places
(the registry tombstone in `features_extractor.__init__`, `_value_pooled_routes`' docstring, and
`pair_value_route.py`, which inherits it): *any future α/β-to-critic proposal passes the C4-style
offline gate FIRST* (ledger C6 — the delivery line is EXHAUSTED). It was deleted because the
measurement says the critic does not use it, not because the idea is unsound, and the seam makes
rebuilding it cheap.

#### The per-head deletions, and why they had to be per-head

`hidden_opp` and `nmr` were each deleted on ONE HALF of one arm. The ledger keeps the hidden-opp
incident because reading the module-level verdict — *"dV 0.0000, delete it"* — would have removed
the single largest policy input in the report. `phase_modules_test.py::
test_the_hidden_opp_belief_pi_half_is_a_live_policy_input` is the pin: the belief must still reach
pi and still MOVE it (two different beliefs must not produce identical policy features — a
dead-but-wired route satisfies a width assertion), and must not reach vf at any weight.
`intent_threshold_test.py::test_the_p_ko_policy_context_survives_the_vf_route_deletion` is the
same pin for the p_KO half: the flag stays ON in production, the move cell is measured in KL/flips
rather than |dV|, and perturbing p_KO must still move the pointer MOVE cells.

#### The instruments: an arm that outlives its subject RE-POINTS, it does not go quiet

`critic_route_audit` loses the `seed` and `intent_reduce` arms, the `vr_*` arms for the three
deleted seam routes, and the `hidden_opp` `both`/`pi`/`vf` split (now one arm — the split became
structural, and reporting `_both` and `_pi` as two identically-equal numbers is uninterpretable).
The generic `value_route` arm STAYS at one member: its whole value is covering the next route on
the day it is written.

**`edge_ablation_audit`'s `concat` arm is deleted with a note, and it is the cautionary case.** It
was built for the v61 op-concat deletion counterfactual and worked by zeroing the assembler's LAST
positional argument. The concat died at v61; from v76 that argument was `seed_rows`. So for three
generations it silently measured the multi-seed CRITIC READOUT under the name of a block that no
longer existed — and duly reported 0.0000 on every axis at n=12,391, identical to the dedicated
`seed` arm. Its twin `concat_cells` is a genuine live tripwire (KL **0.5682**, flips **0.3105** on
gen-14 — the largest policy dependence in that report) and STAYS, re-implemented to patch
`pointer_cells` alone. Same family as the allowlist entry that outlived its own fix.

#### Migration, and the ARCH_SIGNATURE decision

`ARCH_SIGNATURE` bumps to `gen3_critic_route_wave_v1` and `MIGRATION_FLOOR` follows to 96, per the
floor contract. **The bump is not optional and the reason is the interesting part**: `value_projection`
narrows and `assembler.seed_readout.*` leaves the `state_dict`, but NOTHING in `model_config.json`
records either — the seed readout was never a flag. Without the bump a gen-15 checkpoint would pass
`check_compatible` and then die on an opaque torch shape error inside `MaskablePPO.load`, which is
precisely the failure the signature exists to convert into a diagnosis. Consequence to state
plainly: **gen-16 is fresh weights**, and gen-15 is its eval reference and pool seed, never a warm
start.

Three FIELDS leave the config (`intent_value_reduce`, `value_clock`, `value_intent`). Both halves of
the v75 rule are written: `_migrate_config`'s v96 block (unreachable under the new floor, like
v90/v91's, and present for the record) and `snapshot._DEAD_FEK_JUDGED`, which IS reachable because a
checkpoint's pickled `features_extractor_kwargs` carry no `config_version` for a floor to catch.
Recorded ON ⇒ REFUSED with the re-read-from-`git_hash` diagnosis (each built a projection module, so
popping would hand SB3 an unplaceable `state_dict`); recorded OFF ⇒ popped silently. It matters
immediately: gen-15 records all three OFF, while the `ai_v9_17_tdaux_lam*` forks recorded
`intent_value_reduce` and `value_clock` ON and are therefore refused with a diagnosis rather than
TypeError-ing.

#### Files

DELETED: `value_routes.py` (+test), `intent_value_reduce.py` (+test), `seed_diagnostics.py`
(+test), `MultiSeedValueReadout` from `value_readouts.py`, `IntentThresholdValue` from
`intent_threshold.py`, the `VALUE_SEED_K` / `VALUE_SEED_DIM` / `_INTENT_CELL_FEATURES` /
`_INTENT_THRESH_RAW_VF` constants, the `value_seeds/*` emission in `instrumented_ppo`, and three
rows from `flag_registry` (49 → 46 toggles).

#### Gates

The pi-half and p_KO pins above; the `value_route_gradient_test` sweep re-pointed at the surviving
route set (still generic over `_value_pooled_routes` under both critic parameterizations, plus a
new cell asserting the four deleted attributes stay deleted); `critic_route_audit_test` gains
`test_the_deleted_vf_routes_stay_deleted`; `projection_width_test` now asserts `vf == D_MODEL`
rather than an arithmetic; `phase_modules_test` asserts `ProjectionAssembler` owns no parameters at
all. Five `test_migration_defaults_the_flag_off` cases become `test_a_pre_floor_config_is_REFUSED_
not_defaulted` (the v90 precedent — a test may not claim to cover a branch the floor makes
unreachable). `belief_label_only_gate_test`'s α cell moved to the PUBLICATION boundary and its
docstring records WHY both obvious re-pointings are vacuous: the SWITCH-cell α consumers stop-grad
α unconditionally, and this file's uniform-random obs masks all four MOVE logits. Artifact chain
regenerated top-down (delivery graph → viewer → `arch_tables` → `flag_registry.md`), all
`--check`-green; mypy 0.

⚠️ **One side effect worth flagging**: `arch_tables_test::test_production_config_matches_newest_run`
now SKIPS, because the signature bump opens the documented signature-bump window (no run exists at
the current architecture). It was RED before this change on four fields; two of them
(`intent_value_reduce`, `value_clock`) are moot now, and the other two (`all_shaping_pbrs`,
`draw_penalty`) were corrected to the live defaults in the same pass so the mirror is not left
silently stale behind a skip.

## v97 — `gen3_intent_label_bot_weight_v1` (2026-08-18): the intent head stops learning bots at full price

The opponent-intent heads are supervised by **what the opponent actually did** — and for the first
stretch of every fresh generation, that opponent is a bot. `heuristic_fraction` holds self-play at
**0% below `SELF_PLAY_START`**, so α/β's entire early curriculum is a set of scripted decision
trees. Measured on gen-11: supervised intent rows ran **100% bot at 2M and ~7% from 6M on**, and
bot rows score differently in a way that says they are a different problem — info gain **0.124 nats
vs pool 0.254**, with bot accuracy flat at ~0.50 for the whole run. The hazard is imprinting: the
head fits a decision tree while that is all it is shown, and carries the fit into pool play.

`--intent-label-bot-weight W` (default **1.0 = OFF**) is the per-sample answer. `W` multiplies the
α and β label losses on rows whose opponent class is `bot`; pool / stable / exploiter keep 1.0.

```
loss = Σ_i w_i · ce_i / n_sup          w_i = W on bot rows, 1.0 elsewhere
```

**The denominator is the decision.** Weighting before the mean at the unchanged `n_sup` keeps the
`--opp-intent-coef` semantics (`w ≡ 1` is the plain mean) AND makes the knob bite where it must:
normalising by `Σw` instead would render a 100%-bot minibatch identical to an unweighted one — i.e.
do nothing for the whole ramp this exists for. Masks compose rather than collide: `INTENT_IGNORE`
rows are dropped FIRST and the weight multiplies only survivors, so no weight can resurrect a
masked label and `W = 0` legally means "score bot rows for the metrics, train on none of them".

**No new plumbing.** The identity source is the EXISTING `opp_class` obs key (`gen3_opp_class_v1`,
v-none — it was never a config field), tagged per episode by `MaskableAgentWrapper`, pushed onto
the env at `reset()`, emitted beside the α/β labels, shifted with them by
`align_labels_to_predictions`, and already read in `train()` for the stratified metrics. The key
that splits the dashboards now also weights the loss. That promotion is why
`opp_class_plumbing_test.py` exists: the chain had **zero** test coverage while it was
metrics-only, and a break in it stops being a mislabelled chart the moment it decides supervision.

**Confined to α/β, deliberately.** The other supervised beliefs — species, move, item, spread,
nature/EV, HP-type — are **team truth**: what their team IS does not depend on who is piloting it,
so an opponent-class discount there would discard valid labels. Only INTENT is behaviour. The
`belief_bank` rows never see `opp_class`, pinned as a source fact.

New diagnostic **`opp_intent/label_bot_frac`** — the bot share of the α rows actually SUPERVISED
this minibatch. The per-class `alpha_n_supervised_*` counts carried the same information but are
gated on ≥2 rows and are counts, so nothing reported the ratio; it is emitted whether or not the
weight is set, because the decision to set it is made off this number. Every existing stratified
metric is unmoved — they measure the head, and a weighted loss must not move an accuracy.

**Class `training_coef`, and the default is a bit-identity, not an approximation.** At `W = 1.0`
the original unweighted `cross_entropy` call is taken unchanged (the weight helper returns `None`),
so the loss is equal to the last bit — asserted by exact equality over three opponent mixes rather
than `approx`. It scales a loss and touches no forward pass ⇒ **no `ARCH_SIGNATURE` bump**, absent
from `check_compatible`, no `check_*`; recorded on `ModelVersion` for provenance and so a flagless
resume inherits it via `_resolve`, exactly like `td_aux_coef`. Not in `flag_registry.py` — that
registry's scope is extractor architecture toggles, and this reaches the extractor not at all.
`MIGRATION_FLOOR` stays 96, so the v97 branch is the rare REACHABLE one: a v96 config sits at the
floor and genuinely lacks the field, which defaults to 1.0 = OFF.

**LOWERING it is a generation decision, not this change.** Pre-registered: fork A/B **W=1.0 vs
W=0.25** at the gen-16 launch beside the B-move supervision call, gated on
**`opp_intent/alpha_acc_pool`** (the `_pool` suffix — the bare key is a moving mix). W=0.25 wins
only on non-inferiority or better; a fall convicts the premise, meaning bot rows were carrying real
signal. `label_bot_frac` sizes the manipulation before the arm is spent.

End-to-end on a `--debug` CPU smoke (100% bots, so `label_bot_frac` reads exactly 1.0):
`opp_intent/alpha_loss` **1.99 → 1.84** at W=1.0 against **0.49 → 0.48** at W=0.25 — the designed
factor of four, in a real training loop rather than a unit test.

---

> ⚠️ **v98 and v99 have no entry here.** `gen3_cf_evidential_head_v1` (v98, the evidential Beta
> readout) and `gen3_cf_twin_heads_v1` (v99, the twin win-prob heads + the passive shadow critic)
> landed without appending to this file. Their design record is
> `designs/ai_v10/design_counterfactual_value_grounding.md` and
> `designs/research_state/cf_r1_runbook.md`; their version narrative is in `model_version.py`'s
> header comment, which is authoritative and complete. Noted rather than back-filled — this file is
> append-only history and writing an entry for a change one did not make would forge the record.

## v100 — `gen3_cf_coef_provenance_v1` (2026-08-22): the counterfactual coefficients stop evaporating on resume

**What moved.** Ten training-only knobs on the counterfactual value-grounding stack — `cf_records`,
`cf_records_keep`, `cf_winprob_coef`, `cf_head_only`, `cf_label_lag_steps`, `cf_label_likelihood`,
`cf_evidential_coef`, `cf_evidential_reg`, `cf_twin_coef`, `cf_shadow_coef` — leave the `--opd-coef`
genre (argparse-only) for the `td_aux_coef` one: RECORDED on `ModelVersion` for provenance and
`_resolve`-inherited on a flagless resume, **never gated**. No `ARCH_SIGNATURE` bump (each scales,
sources, filters or shapes a loss computed in the PPO step; none is read by the extractor forward,
none changes a weight shape), no `MIGRATION_FLOOR` change, and no `flag_registry.py` rows — that
registry declares EXTRACTOR toggles, and none of these builds a module. The v100 migration is v97's
shape, not v98/v99's: a `setdefault` to each flag's argparse default, which is not a guess but the
only possible past, since before this bump the fields did not exist and a flagless resume therefore
got exactly those values.

**The failure it closes is invisible by construction.** An R1 arm launched with
`--cf-winprob-coef 1.0` and resumed as `train_rl_agent.py --model <ckpt> --steps N` kept training,
kept logging, and simply stopped applying the term it existed to measure — no error, no FATAL, just
a metric that goes quiet and a result that reads as a null. It was strictly worse than a symmetric
loss would have been: the three STRUCTURAL cf flags (`cf_evidential` v98, `cf_twin_heads` /
`cf_shadow_critic` v99) were already recorded AND version-gated, so a flagless resume kept the HEAD
and dropped the COEFFICIENT that drives it — a head that exists, costs parameters, and does nothing.
The old mitigation ("the launcher forwards every non-launcher flag verbatim") only ever covered a
launcher-managed resume.

**The enabling defect, and the vacuity it exposed.** `_resolve(name, default)` fires on
`getattr(args, name) is None`. An argparse entry that defaults to anything else therefore makes its
own `_resolve` line **dead code** — while `flag_registry_test.test_cli_flags_have_a_resolve_line`
keeps passing, because the line is PRESENT. Five live `cli`-tier flags were in exactly that state:

| flag | default it carried | what a flagless resume did |
|---|---|---|
| `value_threat_inject` | `store_true, default=False` | **ON in the gen-17 production config** ⇒ FATAL at `check_compatible` |
| `opp_intent_coef` | `0.0` | `opp_intent` is DERIVED from it ⇒ same, on production |
| `cf_evidential` | `False` | silently reverted the v98 head |
| `cf_twin_heads` | `False` | silently reverted the v99 heads |
| `cf_shadow_critic` | `False` | silently reverted the v99 shadow critic |

All five now take `default=None` with `action=BoolFlag` where a bool (so `--no-<flag>` can still
turn one off explicitly on a resume), and `_resolve` supplies the OFF value for a fresh run.
**`test_cli_flags_argparse_default_is_none` is the new gate** and it closes the REACHABILITY half of
a contract whose PRESENCE half was already gated — the same vacuity class the twin-heads build hit
three times. It asserts against the **built parser** (`build_parser()._actions`), not the source
text: a default can be an expression, so only the constructed object knows it.

**Fresh-run behaviour is unchanged and pinned as such.** `cf_flags_test`'s default tests now assert
BOTH halves — `None` at `parse_args` (without which `_resolve` can never fire) and the OFF value
after `resolve_config` (what a fresh run actually gets) — because asserting only the second would
pass with the defaults back in argparse and the inheritance silently dead again. End-to-end: a
`--debug --steps 10000 --cf-winprob-coef 0.5 --cf-records` smoke records
`cf_winprob_coef: 0.5, cf_records: true` at `config_version: 100`, and a flagless
`--model <that checkpoint>` resolves both back.

**Rode along, same pass.** `arch_tables._COEF_MODULE` now DECLARES the loss-coefficient set instead
of only annotating it: selection was by the name suffix `*_coef` alone, which silently dropped every
loss weight not spelled that way — `intent_label_bot_weight` had been recorded since v97 and
appeared in **no** generated table (it renders `0.25 | ACTIVE` under the gen-17 config), and
`cf_evidential_reg` would have joined it.

## v101 — `gen3_capacity_telemetry_v1` (2026-08-23): saturation becomes a curve instead of a probe

**What landed.** `--capacity-telemetry` plus three cadence knobs (`--canary-reset-steps`,
`--capacity-cosine-every`, `--capacity-velocity-every`), and with them three `capacity/*` scalar
families that ride the PPO train step continuously: the PLASTICITY CANARY, the HALF-BATCH
TRUNK-GRADIENT COSINE, and the FIXED-PROBE FEATURE VELOCITY. All four flags are v100's genre exactly
— the `td_aux_coef` class: an argparse entry defaulting to `None`, a `_resolve` line, a recorded
`ModelVersion` field, **never gated**. No `ARCH_SIGNATURE` bump and no `flag_registry.py` rows: the
canary's head is owned by the **PPO object**, not the extractor, so nothing here reaches
`build_extractor_arch_kwargs`, adds a `state_dict` key, or takes a position in the policy
optimizer. The migration is v100's shape — a `setdefault` to each argparse default, which is not a
guess but the only possible past, since before this bump the telemetry did not exist.

**Why it exists.** Every previous answer to *"is the network out of capacity?"* in this tree has
been an expensive one-shot instrument — a rank sweep, an ablation, an offline battery — each of
which reports a NUMBER at a MOMENT. Saturation is not a moment; it is a trend, and a trend measured
twice is a line through two points. The design constraint was therefore cost, not sophistication:
anything that cannot run on every `train()` cannot produce the reading that matters.

**The canary is the centerpiece, and the RESET is the instrument.** A small detached head regresses
the trunk's `value_pooled` onto K=4 synthetic targets that are pure functions of the observation
(`target_k = tanh(obs @ P[:,k] / sqrt(obs_dim))`, `P[:,k]` drawn under `seed(k,e) = 20260823 + k +
1_000_000·e`, always CPU-seeded so the family is device- and process-independent). Every
`--canary-reset-steps` env steps ONE target is re-seeded, round-robin, and the head's weights are
deliberately NOT re-initialised: re-fitting a brand-new random function of the obs, from the same
representation, with the parameters the network has NOW, is what makes this a SUPPLY-side probe of
the representation rather than another readout of the policy. `capacity/canary_recovery` (post-reset
÷ pre-reset loss, read at a matched `capacity/canary_age`) is the one-number form.

**Scope, stated because the two are easy to conflate:** it measures the REPRESENTATION's richness,
not the policy's headroom. A rising `canary_loss` says the trunk carries less recoverable obs
structure than it did; it does not say the policy would be stronger if it carried more.

**The isolation is structural, not careful.** Nothing in this feature folds a term into `loss` or
writes `.grad` — the canary trains through its own Adam over its own parameters on an
unconditionally-detached input, the cosine reads gradients with `autograd.grad`, and the velocity
probe is `no_grad`. All three run at the END of the minibatch body, after the optimizer step, so the
placement is itself the proof; the `train()` fold sequence gains no step. ON and OFF produce
bit-identical policy updates, and both directions are gated on a real `MaskablePPO`.

**Measured overhead** (CPU, real 2,047,958-param policy over the live 2501-dim obs, 80 minibatches
per `train()` = the production ratio, arms interleaved, quiet box): **+2.38% / +2.52%** across two
independent runs, arms disjoint in both, against a <3% budget — and a conservative bound, since the
benchmark runs the baseline extractor chain whose cheaper forward inflates the probes' share. A
third run taken while a 4-worker pytest suite shared the box read +4.28% with 13% within-arm spread,
which is recorded in `capacity_overhead_benchmark.py` as the reason to read the spread before
believing the delta.

**The honest limitation.** `_capacity_state` is in `_excluded_save_params`, so the canary head, its
Adam state, the projection matrix and the frozen probe batch are re-created on every resume — and
the launcher restarts every 3 hours. The canary's loss jumps there and `canary_recovery` restarts
its curve. The trade was taken deliberately (pickling a diagnostic's optimizer into every checkpoint
is worse) and the usable reading is to compare recoveries WITHIN a restart window: at production
throughput a 3-hour window is ~16M env steps, so a 1M-step reset interval still fires ~16 times
inside one. The startup banner says so out loud.

## v102 — `policy_grad_coef` + `grad/distill_share` (2026-08-26): arm F's unblock and G1's dose meter

**What landed.** `--policy-grad-coef` — a weight on the PPO policy-gradient term ALONE (`loss =
policy_grad_coef · policy_loss + ent_coef · ent_loss + vf_term`; entropy, value and every aux keep
their own coefficients), default 1.0 with a short-circuit that uses the unscaled tensor itself, so
the default is byte-identical by construction (verified: SHA256 over every policy param after a
seeded 3-rollout run, identical vs the pre-change code). v100's `td_aux_coef` provenance genre
exactly: argparse `None`, `_resolve` to 1.0 with a `>= 0` guard, recorded on `ModelVersion`, v102
`setdefault` migration (1.0 is the only possible past — every pre-v102 run trained at an implicit
1.0), never gated. Plus **`grad/distill_share`**: the exploiter-distillation policy-KL term folded
into the grad-balance probe's `aux_terms` (same denominator as `grad/searchteacher_share`), once per
`train()`, read-only `autograd.grad`, verified update-neutral, absent (zero cost) when distillation
is off. No `ARCH_SIGNATURE` bump; no `state_dict` keys; no optimizer positions.

**Why it exists.** Arm F (the pure-distill-phase probe) stopped on its own pre-registered condition:
the policy-gradient term had no coefficient, and `vf_coef` is both resume-immutable and — under
`--value-from-dist` — the critic loss itself, so no existing flag could isolate the distillation
term. And the advantage-gated design's G1 arm is dose-matched on GRADIENT SHARE, not coefficient
(full-distribution KL and action-level CE at the same coefficient are wildly different doses), which
requires the share to be a logged quantity. One pass serves both. The flag was born `--pg-coef` and
renamed the same morning before any run recorded it — an opaque two-letter name on a recorded
provenance field is the `value_feat_cos` lesson waiting to recur.

## v103 — `gen3_distill_target_gate_v1` (2026-08-26): the fold recipe becomes a choice instead of a fate

**What landed.** The advantage-gated design's v1 scope, hours after its adjudication: a
**target-form selector** for the exploiter-distillation loss (`--distill-target {kl,action}` +
`--distill-topk K` — `kl` is the literal untouched call and stays the default; `action` at K=1 is
argmax cross-entropy, K ≥ n_actions reproduces full KL, and the §7.3 identities are pinned by
test); the **advantage gate** (`--distill-gate {none,advantage}` + `--distill-gate-tau` +
`--distill-beta` — fire only where teacher argmax ≠ the sampled action AND the student's own
normalized advantage says the action was a mistake); and the **rank tripwire**
(`--rank-tripwire {off,warn,abort}`, default warn, `agents/training/rank_tripwire.py`) — §4.1
verbatim: baseline = median of `rank/policy_pr` over train() calls [5,25), EMA half-life 10, WARN
at 0.90·base ×3 consecutive, TRIP at 0.80·base ×3 (latched; `abort` stops learn() cleanly with a
checkpoint), missing reading = counters frozen, never all-clear. §4.3 liveness metrics under
`distill/` (`gated_frac`, `n_gated`, `gate_agree_rate`, `mean_gate_adv`). All seven flags v100
provenance genre; config 102 → 103; no ARCH_SIGNATURE change, no state_dict or optimizer changes.
Byte-identity at defaults SHA256-verified on BOTH a plain arm and a distill-KL arm.

**Why it exists.** Five +3M arms plus tick-1 eliminated every teacher-side and knob-side
explanation for the fold's negative transfer and left the full-distribution KL's target form as
the last variable standing; the owner upheld the design's contest of flywheel D-F (amended:
full distribution remains the long-term aspiration, with the late-generation converged-trunk
retry as its pre-registered re-entry path). G1 (action, ungated, dose-matched on
`grad/distill_share`) is the discriminator; G2 (action + gate) is the product. The tripwire
exists so no fold ever runs blind to rank collapse again — it fires on all five known-bad arms
and on none of the controls.

## `gen3_distill_bias_at_coef0_v1` (2026-08-27): `--distill-team-bias` becomes effective at coef 0 — the control arm was never bias-matched

**What landed.** `main.train.config` now parses `--distill-teacher` into `args._distill_pairs`
whenever the flag is given, at ANY `--distill-coef`. The coefficient still gates everything that
costs something or changes a tensor — the teacher model LOADING (`model_build`), the loss fold
(`instrumented_ppo`), and `_distill_species`, which is what makes the env emit the training-only
`distill_mask` obs key — but the TEAM BOOKKEEPING no longer rides on it, so the trainee's team draw
is biased onto the teacher teams at coef 0 exactly as the flag says. The bias block moved into
`matchup_setup.apply_distill_team_bias` so it can be measured directly rather than only through a
whole run. Three guards ride along: `--distill-team-bias > 0` with no `--distill-teacher` is now a
`parser.error` (the flag's argparse default became `None`, resolving to 0.4 in `resolve_config`, so
a TYPED bias is distinguishable from an unset one and the guard can exist without refusing every
ordinary run); `--distill-teacher` alongside `--trainee-team/--trainee-teams` is refused for all
coefficients, since the bias REPLACES the trainee teambuilder and would silently discard a pin; and
the spec's bare-list check now looks at the FIRST comma segment only, which un-breaks the documented
single multi-team group `T1:a.txt,b.txt` (previously rejected unless another `;`-joined teacher
happened to follow it). No `ARCH_SIGNATURE` bump, no `MODEL_CONFIG_VERSION` bump, no `state_dict`
keys, no optimizer positions — a run with no `--distill-teacher`, and a run with teachers at coef
> 0, are both unchanged.

**Why it exists.** `ai_v9_58_R2CTRL_0827` — the rev-2 capstone's CONTROL arm — was launched with
five teachers, `--distill-coef 0` and `--distill-team-bias 0.4` precisely so the team distribution
would be held constant against the treatment arm while folding no loss. `_distill_pairs` was
populated only under `if args.distill_coef > 0`, so its effective bias was **0.0** while its argv,
its `metadata.json` and its startup banner all read 0.4. The arms therefore differed in the one
variable the design pinned, and nothing anywhere said so: this is the recorded-flag ≠ effective-
behavior class, which this project treats as drop-everything for the reason on display here — an
experiment does not fail loudly when its control is quietly a different experiment. Gate:
`src/main/distill_team_bias_test.py`, which MEASURES the draw (4000 draws, teacher-team fraction
0.4 ± 0.04; pre-fix 0 of 4000) rather than asserting the flag's value, and pins the coefficient-gated
half in both directions (no teacher network is loaded at coef 0; the loader IS reached above it).

---

## `gen3_exploiter_pool_ladder_v1` (2026-08-27): `--exploiter-ladder` — the exploiter's opponent becomes a curriculum, not a wall

**Motivation (C1).** `--exploiter <target>` trains a specialist against ONE frozen, full-strength
target from step 0. When that target is a near-twin of the trainee's own init — which is the
recommended recipe (`--exploiter X --model X`) — the trainee loses nearly every game. PPO's advantage
is a *within-batch* contrast, so a batch in which every episode ends in a loss carries almost no
information about WHICH decisions were the bad ones: the exploiter is starved of variance in outcome
rather than of capacity. `ExploiterTempRatchetCallback` (`gen3_exploiter_temp_anneal_v1`) already
attacks that along the STOCHASTICITY axis — one opponent, made to play noisily. This is the owner's
pool-ladder design for the other axis: keep every opponent's play honest and instead start against a
genuinely WEAKER one, promoting up a ladder that terminates at the real target.

**What landed.** `--exploiter-ladder` takes an ORDERED, weakest-first rung list — either an explicit
comma-separated list of checkpoint specs in the `--stable-opponents` grammar, or `auto:<run_dir>`,
which draws `--exploiter-ladder-rungs` (default 4) evenly-ELO-spaced snapshots from that run's
`snapshot_ladder/ladder.json`. The `--exploiter` target is always appended as the terminal rung. The
controller promotes one rung when the trainee's TRAINING win-rate vs the CURRENT rung clears
`--exploiter-ladder-gate` (default 0.55) over a completed window of `--exploiter-ladder-window`
games (default 500 — the `--exploiter-temp-ratchet-games` semantics and value). One-way: no
demotion, terminal rung sticky.

**The auto draw orders by ELO, not by step, and that is not a stylistic choice.** Training is not
monotone in strength: in `ai_v9_27_extremedial_probe_0823`'s 20-snapshot ladder, step 42.0M rates
1888.6 — the WEAKEST snapshot in the run — while 45.0M rates 2087.4. "The earliest N snapshots"
would therefore have built a curriculum whose third rung was weaker than its first.

**The gate reads ONE opponent, by construction rather than by convention.** The wrapper carries a
rung index alongside its own `(games, wins)` pair, zeroed in the same operation that swaps the rung
in, and the callback drops any worker row whose index is not the live one — so a promotion window
can never pool games played against two different rungs. Bot episodes under `--exploiter-keep-bots`
were already excluded from that counter, so the two flags compose without interacting: the ladder
changes WHO the non-bot opponent is, never how often it appears.

**The swap rides the existing opponent mechanism.** `env_method("set_exploiter_rung", index, zip,
config)` is the `set_self_play_target` / `set_exploiter_temperature` idiom, change-guarded so a
steady rung costs no IPC; the worker DEFERS it to the next `reset()` (an opponent's brain must not be
replaced mid-battle, and the episode in flight must be scored against the rung it was actually played
against) and then assigns into the persistent `RLPlayer` exactly as `_ensure_pool_model` does for a
self-play snapshot. The model loader is injected from `env_factory` as a closure, so it owns the arch
gate, the device and `--compile-opponents` — rungs compile like any other frozen opponent — and the
wrapper stays free of model-loading policy. Only WEIGHTS move: the target's pinned team (the
fold-back contract), its sampling temperature and the bot fraction are all untouched, so the
curriculum varies exactly one quantity and remains orthogonal to the temperature ratchet.

**Resume.** The live rung, per-rung counts and the promotion log are persisted atomically to
`<run>/exploiter_ladder_state.json` (on every promotion, plus every 20 rollouts) and restored at
training start — **by rung LABEL first**, so an edited ladder resumes at the same OPPONENT rather
than at whatever now occupies that index. This is the same lesson `exploiter_temp_state.json` records:
the launcher restarts the child every three hours, and without a restored artifact the curriculum
resets to rung 0 each time and never reaches the target, silently.

**Training-only, and OFF is byte-identical.** No weight-shape or forward change, so it is not
version-locked and carries no `flag_registry` entry — it lands in `cli_args` / `metadata.json` like
every train-loop knob. Rungs are resolved, arch-gated and load-validated in phase 2
(`matchup_setup`), so a bad rung is a `FATAL_CONFIG` at startup rather than a crash in every env
worker; with no flag, no rungs are stashed, no callback is registered, no `env_method` is ever
called, and the wrapper is built with `exploiter_rung_loader=None`, which makes every rung branch a
single `is None` test. `--exploiter-ladder` without `--exploiter` is a `parser.error` (the terminal
rung IS the target — a ladder without one has no destination). Gate:
`src/agents/training/exploiter_ladder_test.py` (52 tests), revert-verified on the four load-bearing
behaviors: dropping the live-rung filter, applying the swap immediately instead of at `reset()`,
skipping the resume restore, and admitting a demotion each fail a named test.

---

## `gen3_signal_rate_metrics_v1` (2026-08-28): the `signal/` group — how much action-attributable signal is actually arriving

**The gap.** Every meter this tree carries answers *how well is the update going* (`train/loss`,
`approx_kl`, `explained_variance`, `grad/*`, `rank/*`, `train/noise_scale`). None answered the prior
question — *is there anything in this rollout to learn from?* A run whose opponent has become a wall
and a run whose opponent has become a coin flip both produce a healthy-looking PPO step; the
difference only shows up months later as a flat ELO curve. `signal/` makes it a live curve.

**What landed.** Two always-on, flagless scalar families. Pure observability: no gradient path, no
flag, no extra battle, no env round trip, a handful of numpy means per rollout.

* **Advantage density**, recorded in `instrumented_ppo/ppo.py::train()` — `signal/adv_raw_std`,
  `signal/adv_raw_abs_mean`, `signal/adv_kurtosis` (excess, Fisher).
* **Outcome entropy**, recorded by the new `SignalMetricsCallback` — `signal/outcome_entropy`
  (rolling `p(1−p)` over 200 episodes, pooled) plus `_bots` / `_pool` / `_stable` / `_target` splits,
  the window's `outcome_win_rate` and `outcome_n[_<kind>]`, and — from
  `ExploiterLadderCallback._record`, which owns the only per-rung window —
  `signal/outcome_entropy_rung`.

**THE PAIR IS THE INSTRUMENT, and shipping only one of them would have been worse than shipping
neither.** Outcome entropy is MAXIMAL against a near-twin, which is exactly the regime where a single
action's effect on the outcome is smallest — the mirror paradox. So a high outcome entropy is not
"lots of signal", it is "lots of outcome VARIANCE", which is only signal to the extent the critic
localizes it onto actions; that localization is what the advantage density measures. High-entropy ×
low-density is the diagnosis this group exists to make, and neither axis alone can make it.

**Why the third moment.** Exploit signal is SPARSE — a few decisive turns inside a long stretch of
forced or irrelevant ones — so a healthy rollout is heavy-tailed. `adv_kurtosis` separates shape from
scale: the gate test builds a 0.5%-support rollout and an evenly-spread one with the SAME
`adv_raw_std` to 9 significant figures, and they read +195 and −2.0.

**Measured WHERE it still exists.** The advantage read sits before the epoch loop, off
`self.rollout_buffer.advantages`, because the minibatch loop's `normalize_advantage` forces std→1 per
minibatch and so destroys the exact quantity being measured. That also makes it transport-agnostic
(both the stock collector and `collect_rollouts_async` fill the same buffer) and free under
`--grad-accum-steps` (one read per rollout, not per optimizer step).

**⚠️ UNITS.** The advantages ride the run's PopArt-normalized returns, whose σ moves over training.
`adv_raw_std` / `adv_raw_abs_mean` compare WITHIN a run; only `adv_kurtosis`, scale-free by
construction, compares across runs. Stated in the code comment, the module docstring and the
training leaf, because a scalar whose units are silent will be cross-run compared eventually.

**⚠️ It is a TRIPWIRE, not the measurement.** The gold standard for attributable share remains the
offline counterfactual decomposition (`prober falsify-scan`'s luck / unattributed /
proven-`policy_reducible` bracket, and `cf_audit`), which re-rolls the real dice and sweeps
alternative actions. `signal/` reports only the critic's own opinion of its own rollout — it tells
you when to go and run one.

**`--async-rollout` is covered, not documented-around.** The stock collector publishes
`infos`/`dones` in the callback locals; the async collector publishes `wave_infos`/`wave_dones`. The
callback reads whichever is present. `WinProbLabelCallback` cannot do this — it needs the
`(step, env)` buffer row, which wave batching destroys, hence its inline capture inside the async
collector — but outcome entropy is a per-episode aggregate with no row alignment, so the wave form is
sufficient.

**Which opponent splits are REAL, and which are structurally impossible.** Only an integer crosses
the env-worker pipe, so the wrapper's four `OPP_CLASS_*` values are the whole available alphabet:
`bots` / `pool` / `stable` / `target` ship. WHICH heuristic bot does not (the class collapses random
and every heuristic into one) and WHICH pool snapshot does not (its frozen-at step is held by
`SnapshotPool` in the parent and never reaches the wrapper). `_rung` is the one finer split that
exists, and only because the ladder callback keeps its window in the parent process. The producer
side is one additive `info["opponent_class"]` key beside the existing `info["win_outcome"]`; SB3
reads only `episode` / `terminal_observation` / `TimeLimit.truncated` from an info dict, so nothing
downstream changes.

**NaN-safe, never fabricating.** An empty buffer publishes nothing; a constant rollout reports a real
std/abs-mean with `adv_kurtosis` NaN, and TensorBoard drops NaN — so a degenerate rollout leaves a
GAP rather than a 0.0 that would read identically to a genuine "advantage mass is evenly smeared".
Same rule on the outcome side: a window with no episodes yet publishes nothing rather than the 0.0 a
100%-loss wall would produce.

**Observability, proven rather than asserted.** `signal_metrics_test.py` (36 tests) runs one
`train()` with the estimator live and one with it monkeypatched to a no-op from the same captured
init, and requires the resulting parameters to be equal at `atol=0.0`; a companion test requires
`rollout_buffer.advantages` to be the same bytes and dtype after the read. The rest pins the
arithmetic (hand-computed constants + an independently written closed form), every degenerate input,
window eviction, kind routing, both rollout paths' locals, and the `OPP_CLASS_SUFFIX` ↔ wrapper-
constant correspondence — the integer is all that crosses the pipe, so a renumbering there would
otherwise silently relabel a curve.

---

## v104 — `gen3_winprob_pbrs_v1` (2026-08-29): the win-prob head gets a route to behavior — PBRS reward shaping

**`--win-prob-pbrs-coef`, default `0.0` = OFF and byte-identical (the module is not even imported).
Nothing has run it; no arm is registered.** ai_v12 route 1 —
`designs/ai_v12/design_winprob_behavior_coupling.md`.

**The correction this exists to answer.** `--win-prob-mode shaping` has been read for generations as
"the win-prob objective shapes the policy". It does not. It is **REPRESENTATION** shaping: the BCE
gradient reaches the shared trunk, subsidising outcome-predictive features there. There is no
gradient path anywhere from *predicting wins* to *choosing winning actions* — the logit is a SIDE
readout, never concatenated into pi/vf (leak-safety: its label is the privileged future outcome), so
the policy may ignore the subsidised features entirely and V compresses to its own target regardless.
The head is a **BAROMETER, not a coach**. Its labels are self-referential too — outcomes under the
current policy — so a habitual whiff that still wins 55% teaches it "55%", never "the whiff was the
mistake". "Shaping is live yet the bait loops persist" was never a dose mystery: the live mode was
never pointed at behavior.

**What this adds.** With φ(s) = σ(win-prob logit), DETACHED, every transition's reward becomes
`r + coef·(γ·φ(s′) − φ(s))`. The drop after a whiff is now literal reward the policy gradient must
answer for. It **suppresses without knowing the alternative** — softmax renormalization redistributes
the suppressed mass — which is the complement of what a distillation target does.

**The shield, and the caveat it does not cover.** Potential-based shaping leaves the optimal policy
set unchanged for any *fixed* φ (the shaping telescopes to `γ^T·φ(s_T) − φ(s_0)`, a constant per start
state), so a miscalibrated φ costs learning SPEED, not correctness. **Our φ is a LEARNED, DRIFTING
head.** Exact invariance therefore holds **per rollout** — PPO freezes the policy during collection
and φ is read once, after it, with the collection-time weights — and degrades to **approximate**
invariance across rollouts, bounded by φ's drift over one credit-assignment window. The lever wants a
MATURE base; a fresh run tests the shield's worst case. Mitigating: the G0 bias map found the head's
defect is **RESOLUTION, not offset**, and a blurry potential is a WEAK one, not a wrong one — a φ
constant over a set of states contributes nothing over that set and cannot mislead within it.

**Where it runs, and why there is only one window.** Env workers hold no model, so the reward cannot
be shaped where it is produced. `InstrumentedMaskablePPO.collect_rollouts` applies it after collection
and before `train()`: one batched `no_grad` φ forward over the buffer, the term added to
`rollout_buffer.rewards` in place, then `compute_returns_and_advantage` RE-RUN. Both collectors
compute GAE as their last act and PopArt reads `returns` at the top of `train()` — so this is the only
point at which the shaping can land in **RAW reward space** and still reach the advantages, which is
the order that keeps the value loss in the units of the stream being optimized.

**`--async-rollout` is COVERED, not documented around** — and the reason is the same one that forced
`WinProbLabelCallback`'s inline terminal capture. The per-step `last_win_prob_logits` stash is
readable from `_on_step` on the stock path, but the async collector forwards a *wave* of envs and its
callback locals cannot recover the env→row mapping. A batched re-forward gives BOTH paths the
identical quantity for ≈ one forward pass over the rollout (roughly `1/n_epochs` of one epoch), so the
transport-specific capture was not worth its two implementations.

**The two conventions are NOT the same case, and conflating them is the classic bug in this family.**
TERMINAL (`episode_starts[t+1] == 1` — the identical test SB3's own GAE uses for `next_non_terminal`,
so the two notions of "terminal" cannot drift apart): φ(s′) := 0, which is what makes the per-episode
discounted sum telescope to exactly `−coef·φ(s_0)`. BUFFER-BOUNDARY TRUNCATION (the episode is still
running when the rollout ends): φ(s′) is the **bootstrap** φ(s_T), not 0 — forcing 0 there charges the
policy a phantom penalty for the rollout merely ending. `TimeLimit.truncated` arrives as `done=True`
and takes the terminal branch, which here is arguably correct rather than approximate: the 250-turn
cap IS the forfeit deadline and the reward manager scores it as a real outcome. Pinned by test either
way, so a change to the timeout's semantics fails loudly.

**Fail-loud, never a silent no-op.** `> 0` with `--win-prob-mode none` is refused at config time (the
potential IS the head; under `none` no head is built). A negative coefficient is refused because it
inverts the potential — and the theorem still holds for `−φ`, so it would train, converge, and be
wrong. A missing head at runtime raises `WinProbPbrsError`. A shaping term the operator believes is on
while it does nothing is the invisible-regression class this project keeps eating.

**Metrics** `train/pbrs_{shaping_mean,shaping_absmean,phi_mean,reward_share}`. Under `train/`
deliberately: this is a property of the reward stream PPO is fitting, not of the head.
`pbrs_reward_share` — mean |shaping| over mean |UNSHAPED reward| — is the one that says whether a
coefficient is sane; its denominator is the unshaped stream on purpose, so the ratio does not flatter
itself as the coefficient rises.

**Provenance.** TRAINING-only, the `td_aux_coef` class exactly: recorded on `ModelVersion` for
provenance + flagless-resume read-back, never in `check_compatible`, no `ARCH_SIGNATURE` bump. It is
the first knob in that class to edit the REWARD STREAM rather than a loss term — worth saying, since
the class was defined by loss coefficients — but the provenance reasoning is unchanged: no forward
pass reads it, no weight shape depends on it. A pre-v104 config migrates to `0.0`, which is not a
guess: the flag did not exist.

**Tests** `agents/training/winprob_pbrs_test.py` (22). The telescoping identity on a hand case and
over 40 random episode layouts; truncation vs terminal; an off-by-one revert-catcher on the
`episode_starts[t+1]` test; grad-disabled + detached-to-numpy (fails if the `no_grad` is deleted);
coef-0 buffer identity plus the source contract that the import is local to the non-zero branch; the
raw-reward / GAE-recompute order; chunk-boundary coverage; the loud-refusal path; both config gates;
the v104 migration. Three revert-catchers were verified failing on a deliberate revert.

---

## `gen3_winprob_oneply_teacher_v1` (2026-08-29): `--search-teacher-mode` — the ExIt seam gains a second SUPPLY, not a second pipeline

**`--search-teacher-mode` defaults to `crater`, which is byte-identical to the behaviour before the
flag. Nothing has run `winprob_oneply`; no arm is registered.** ai_v12 routes 2+3 —
`designs/ai_v12/design_winprob_behavior_coupling.md`. No config-version bump: all three flags are
OPERATIONAL (the `--search-teacher` / `--teacher-search-budget` class), re-passed on resume and
recorded in `metadata.json`'s `cli_args`, not on `ModelVersion`.

**The question the new mode asks, and why it is a different one.** `crater` asks *"where did the
model lose the most value, and is there a strictly better LINE"* — value craters, falsify-gated to
reducible mistakes, then a depth-2 beam over the CRITIC. `winprob_oneply` asks *"at a decision the
model's own win-prob head calls CONTESTED, does a one-ply successor read prefer another action by a
margin that survives paired-rollout confirmation?"* It also takes **every outcome**, not only losses:
a whiff in a won game is still a whiff, and the head's self-referential labels — outcomes under the
current policy — are exactly why it never noticed.

**It is a SUPPLY, not a pipeline.** The output is the same `Correction` record, so the shard format,
`CorrectionBuffer`, `_searchteacher_loss` and `--search-teacher-coef` are untouched and cannot tell
the modes apart. Only selection and production are swapped, and the dispatch lives in ONE module
(`teacher/modes.py`) because there are three call sites — the per-cycle worker, the persistent worker
and the callback's own selection — and a mode string validated in three places will eventually mean
three things. A worker config with no `mode` key defaults to `crater`, so a config written by an
older parent still runs exactly as it did.

**The filters, in order.** CONTESTED (`n_legal ≥ 2` AND `|P(win) − 0.5| < --winprob-teacher-band`,
default 0.15) — **imported from `search_dividend.defensive.gate`**, not re-typed, so the teacher's
notion of "contested" and the searcher's cannot drift apart; model-free, off the trace's recorded
`win_probs`/`action_mask`. ONE-PLY (`ProbeSession.lookahead`, the **win-prob** read — a candidate
with no win-prob read is DROPPED, never scored from the critic, because a fall-back would silently
run a different teacher under the same flag). MARGIN (`--winprob-teacher-margin`, default 0.02,
against the PLAYED action rather than the runner-up — the target exists to move probability OFF what
the policy did). CONFIRMATION (`--teacher-confirm-rollouts`, the EXISTING flag: paired
`replay_counterfactual` rollouts, A\*'s Wilson LOWER bound against the played action's POINT rate —
asymmetric on purpose, because the failure it catches is a flattering estimate of the challenger).

⚠️ **CONFIRMATION IS A REQUIREMENT, NOT A REFINEMENT — the WINNER'S CURSE.** Defensive-search iter 2
un-throttled its allocator, produced **13× more evidence-certified overrules (1.8% → 5.82%)** and
landed the win rate on **0.5003 [0.4803, 0.5203] — the point estimate IS the null**. CRN pairing
removes dice noise *and* the shared offset, so what a separation procedure certifies is the leaf's
residual **differential** bias (RMS 0.122, larger than most true gaps) as much as signal.
**Statistical separation of a biased reader is not correctness** — and unlike route 1's PBRS, a
distillation target has **no invariance shield**: a wrong target simply trains the policy to be
wrong. `--teacher-confirm-rollouts 0` exists only as the design doc's E2 control arm.

**What keeps the mode alive rather than killing it:** probe K re-judged iter 2's 3,531 overrules
under opponent-MARGINALIZED ground truth and found **+0.0474 [+0.0216, +0.0730] per decision — REAL**.
The overrules were right; the per-decision → per-episode TRANSFER failed. A training target changes
the policy everywhere the network generalizes, not only at the ~2.2 decisions per game where a
searcher intervened.

**The default margin is 0.02, deliberately NOT the measured 0.122.** Shipping the leaf-bias RMS as
the floor collapses target volume by roughly an order of magnitude before any arm has asked whether
it should; E4 is where the volume/quality trade gets measured. If the head's differential bias is
ever fixed at source, this default and E4's premise both need re-measuring.

**Reused vs not, recorded so nobody "fixes" it.** `defensive.gate` + `DefensiveConfig`: imported.
`defensive.verdict`/`resolve_action`: not — they answer "which action do I PLAY". `racing.Racer` and
the budget/deadline machinery: not — they are the allocator, racing arms against a wall clock inside
a battle in flight, and the teacher works offline from a recorded reconstruction with no clock to
race; importing them would drag a live-battle dependency into a path that has none.
`playoff.PlayoffRunner`: not — it needs a live `SearchEngine` and a shared `Deadline`.

**Tests** `teacher/winprob_oneply_test.py` (40): every gate as a pure function including the
synthetic winner's-curse rejection and the asymmetry of the confirmation test; the mode seam (unknown
mode RAISES rather than falling back, both dispatch pairs, the two margins staying separate
parameters, both workers' `crater` fall-back, callback-time validation); the consumer contract (a
winprob `Correction` runs through the real `_searchteacher_loss`); crater-path argument identity; all
five config gates.

**Probe L folded in the same day (`d395556`+`bda8382`).** It fires the distillation branch — the
head ranks an alternative above the played action on **96.4%** of immune whiffs, **+0.213
[+0.177, +0.248]** over the tightest matched control, dice-invariant at two orders of magnitude
above its own measured floor, while the policy samples that alternative at a median **p = 0.002**.
It also supplies the structural argument this mode's existence rests on: **the head's ranking is
not a quantity the network computes** — it is the head composed with a SIMULATOR, one re-roll per
candidate action, and nothing in PPO performs that composition. No coefficient and no gradient
route can deliver it; only an explicit teacher that materializes the ranking and writes it back as
a policy target. Two consequences recorded in the design doc: the "shaping-dose ladder above 0.05"
lever is **refuted, not merely unselected**, and **v104's E1 coefficient ladder was re-sized by two
orders of magnitude** (it had been drafted against an assumed terminal reward of order 1; the live
scale is `VICTORY_VALUE = 30`, so `{0, 0.1, 0.3}` became `{0, 3, 9}`). `train/pbrs_reward_share`
is the metric that makes a homeopathic coefficient visible within one rollout rather than after a
generation.

## `gen3_scaffolding_gauge_v1` (2026-08-29): the shaped-vs-game value gap, live and offline — with two gauges because one number would have been a fabrication

Registered as an instrument by the value-function foundations ruling (ledger 596608e). Nothing here
changes a weight, an observation or a loss: an offline CLI plus one flagless TensorBoard scalar.

The critic estimates the **shaped** return; the win-prob head estimates the **game**. Neither is a
repair of the other — the two-head structure is the automatic consequence of choosing shaped
rewards. Their DIVERGENCE is the reward scaffolding still doing work, and its trajectory is the
registered signal for when shaping coefficients can begin annealing toward the pure game. That
divergence had never been computed.

**🚨 UNITS, which is the whole design constraint.** Recorded `V` is a PopArt-normalized shaped
return: its scale moves over training, its composition is whichever reward terms are on, its
horizon is `gamma`. **There is no general unit conversion to a win probability**, and under PBRS
with a good potential there is not even a monotone one to recover — the classic φ=V* result drives
`V_shaped` toward a CONSTANT (ledger db9bb5c). A single "the gap is X win-percent" number would
therefore have been a fabrication. So the gauge ships as TWO instruments, each labelled:

* **RANK gauge** — `(1 − Spearman ρ(V, P(win))) / 2`. Unit-free, PopArt-proof, always valid,
  invariant to any monotone reparameterization of either axis. CAN claim ordering agreement;
  CANNOT claim anything about magnitude, and goes AMBIGUOUS exactly at the constancy endpoint,
  where ρ degenerates into noise and a falling curve cannot be told from V running out of variance.
* **CALIBRATED-AFFINE gauge** — fit `q = clip(a·V+b, 0, 1)` against the REALIZED per-battle
  outcomes on the same slice, then report `rms|q − P(win)|` in probability units. CAN claim a
  distance in outcome units, because the fit puts both sides there by construction; CANNOT claim
  the two are natively commensurable — the map is a per-checkpoint FIT that does not transport.
  And part of every residual is the **affine family** being a worse outcome predictor rather than
  the heads disagreeing. That part is separated and shipped as `readout_penalty` (= Brier(readout)
  − Brier(head)), because a large `rms` with a large penalty is a readout finding, not a divergence
  finding — and a reader who cannot tell those apart will file the first as the second.

**The offline half:** `python -m main.scaffolding_gauge <run>` — model-FREE (it reads the recorded
`values` / `win_probs` columns out of `eval_traces`, so it works on a run whose checkpoints no
longer load), one row per checkpoint step, table + JSON + optional 3-panel PNG. The JSON carries a
`units` block with a `what` and a `cannot` for every metric, so a reader six months out cannot get
the number without the caveat.

**Every CI is a CLUSTER bootstrap over BATTLES.** Outcome labels are per-battle and broadcast to
every state, so an i.i.d. interval over states would be fabricated tightness of roughly
`sqrt(states-per-battle)` — measured here at ~7× on the fixture. The pooled-correlation Simpson
trap has cost this tree a finding before.

**A run with no win-prob head REFUSES** with that diagnosis instead of curving zeros. "The two
readouts agree perfectly" and "there is no second readout" must not render the same.

**The live half:** `train/scaffolding_gauge` (+ `_rho`, `_n`) — always on when the head exists,
flagless, gated on the head's EXISTENCE rather than on `win_prob_coef` (a `read_only` head at
coefficient 0 still says something worth curving). RANK FORM ONLY, deliberately: the live path has
no realized outcome labels for the states it is scoring, so the calibrated gauge is offline by
construction and publishing a live number labelled as the calibrated one would have been the worse
error. Read inside the minibatch loop right after the win-prob block — the one place both readouts
exist for the SAME states from the SAME forward — and **epoch 0 only**, because by epoch 3 the
policy that produced a pair is not the policy the pair would be attributed to. The logit is not
sigmoided (monotone ⇒ identical ρ, and float32 ranks never saturate).

**The CONSTANCY sanity row** (the db9bb5c prediction as a checkable one-liner): `v_std` / `v_iqr` /
`dispersion` plus the within-vs-between-battle split, per checkpoint, `--constancy` for that block
alone. Under PBRS with a good frozen potential `V_shaped` flattening **CONFIRMS** the theory rather
than indicting the critic. The split is what separates it from its look-alike: low `v_std` with
`within_frac ≈ 0` means V became a per-battle matchup lookup, which a raw std cannot distinguish
from a flattened potential.

Measured on a live run's traces (`ai_v9_69_R3F6CURR_0828`, 358 traces, 3 steps): rank gauge
0.152 / 0.149 / 0.214, affine rms 0.198 / 0.223 / 0.259 with a **negative** `readout_penalty`
throughout (the affine readout of V predicts outcomes BETTER than the head does on these slices)
and a bias of −0.17 to −0.25 (the head reads systematically more optimistic than V-implied outcome).
Recorded as an observation, not a finding: three points, a moving eval quota, and no control arm.

### Files + tests

New: `agents/training/scaffolding.py` (pure numpy — shared by the live scalar and the offline CLI
so they cannot drift), `main/scaffolding_gauge.py`. One edit to `instrumented_ppo/ppo.py` (collect
in the minibatch loop, publish beside the `signal/` group).

`scaffolding_test.py` (31) — the three known regimes come back exactly (monotone ⇒ 0, inverted ⇒ 1,
independent ⇒ ~0.5), affine-rescale invariance, the constant-axis NaN in every gauge, the
`readout_penalty` convicting the FAMILY on a constructed step function while a linear control
collapses it, the cluster bootstrap beating an i.i.d. one, and the live scalar's byte-identity /
no-key-without-a-head / NaN-safety / epoch-0-only read.
`main/scaffolding_gauge_test.py` (17) — a constructed three-regime trace tree end to end, the
all-NaN refusal, the seeded whole-battle cap, a corrupt npz counted rather than fatal.

**One methodology defect found and pinned while writing this.** `instrumented_ppo_test`'s
`_train_from_init` helper is only deterministic when its init snapshot is taken **before**
`learn()`: afterwards the optimizer's Adam state is populated, `deepcopy` + `load_state_dict` hands
back tensors it aliases rather than copies, and the next `train()` mutates the very snapshot it was
restored from. Two "identical" calls then drift by ~1e-3 and any byte-identity claim built on them
is vacuous. Measured before the byte-identity test here was believed, and recorded in that test's
comment.

## `gen3_exploitability_curve_v1` (2026-08-29): `python -m main.exploitability` — the meter a weak opponent cannot inflate, plotted deliberately for the first time


The learning note `2026-08-28_nash_exploitability_psro.md` named the machinery this program has
been reinventing and observed that **our exploiters ARE best-response computations and the
admission table IS an exploitability measurement** — the one meter a weak opponent cannot inflate,
and the quantity "the wheel turns twice" is a claim about. It had never been plotted deliberately.
This is the instrument, and it is pure bookkeeping over `fleet_admission`-schema artifacts that
already exist: no battles, no models, no traces.

Per generation: the best-response **net extraction** (mean + max over teachers, with the artifact's
own per-arm intervals carried through), target identity, the ceiling/headroom reframe the rev-3
admission adopted, the meter-vs-coverage team split (mechanical from the `COV_` prefix), the
budget/team normalization, consecutive-generation deltas, and a markdown row format the ledger
quotes directly.

**Two CIs, because they assume different things and neither is universally right.**
`ci_propagated` combines the per-arm standard errors assuming independence (optimistic — the arms
share reference cells); `ci_between_teachers` uses only the spread across teachers. On the real
rev-3 artifact the between-teacher interval is the WIDER one, which is itself the honest reading:
the six best responses disagree by more than their own sampling error.

**A delta whose interval straddles zero reads "NO DETECTABLE CHANGE", never a direction.** The sign
of a point estimate is not a finding, and the live rev-2 → rev-3 pair is exactly the case that
would have been mis-narrated (+0.0185, CI [−0.010, +0.047]).

**Four caveats ship inside the artifact**: it is a LOWER BOUND (a finite-budget fork is an
imperfect best responder — a falling curve is evidence, a flat one is not proof of Nash); the teams
are PINNED, a subgame restriction, so two generations compare only at matched team sets and partial
overlap is flagged with the mixture it introduces; the MAX is selected and upward-biased, its
interval a CI for that arm and never for the population maximum; and the two intervals' assumptions
are named.

**SCHEMA DRIFT REFUSES, LOUDLY** — exit 2, naming the offending key, seventeen distinct structural
checks before any arithmetic. And the `net = teacher − reference = ordered − seniority` identity is
**RECOMPUTED from the artifact's own per-team win records** rather than trusted; a mismatch refuses
rather than reading past it, with `--no-verify` printing both columns for the case where the
convention genuinely changed. This is the recorded-vs-effective/derived-key defect class, whose
costliest instance was averted only because a harness printed VERIFIED instead of assuming.

Validated against the LEDGER's own published tables: rev-2 reproduces the +0.1165 cluster (sd
0.0098) and rev-3 the six admitted arms (+0.0988 / +0.0863 / +0.1413 / +0.0700 / +0.1962 / +0.2175)
with the identity VERIFIED on all eleven, plus the ~0.69 teacher ceiling and the coverage-vs-meter
headroom split the reframe turns on (coverage target 0.41, headroom 0.26; meter target 0.61,
headroom 0.09).

### Files + tests

New: `main/exploitability.py`. `main/exploitability_test.py` (35) — the ledger's rev-3 table as the
fixture, and seventeen parametrized schema refusals each asserting that the message NAMES the
defect.

## v105 — `gen3_clean_world_config_v1` + `gen3_winprob_pbrs_source_v1` (2026-08-29): the clean-world reward becomes reachable by flag, and the potential can be FROZEN

**Five config keys, every default equal to the behaviour that shipped the day before.** A flagless
launch is byte-identical — the reward census, the terminal magnitudes and the φ read are all
unchanged — and `reward_class_composition` proves it rather than asserting it. No `ARCH_SIGNATURE`
bump: no weight shape moves and no forward pass reads any of this.

Spec: probe N, `designs/research_state/measurements/no_progress_tax_review_2026-08-29.md` §5-§6,
which enumerated four gaps (B1-B4) between the registered clean-world arm and what the flag surface
could express. This closes all four. **Nothing has run it; no arm is registered.**

### The gap, and why it was invisible

The clean arm's target composition is **1 TERMINAL ∈ {+1, −1}, draw −1, ZERO PBRS, ZERO BIAS**, with
a frozen mature win-prob head as the only dense signal. Three things stood in the way, and the first
is the interesting one:

**`--all-shaping-pbrs` does TWO jobs, and they are ANTI-CORRELATED.** It folds five potentials
(Φ_hazard / Φ_boost / Φ_opp_boosts / Φ_status / Φ_roar) *and* it is `_bias_term_active`'s master
gate. So `--no-all-shaping-pbrs` silences the potentials while **reviving 25 BIAS terms** — the
fully-additive objective the v9 era drifted into. Turning it ON gives 0 BIAS (well, 1: the
`no_progress_tax` tilt it deliberately keeps) but leaves five hand potentials live. "No hand PBRS
**and** no BIAS" sat in a hole between the two settings, reachable by no combination of the existing
flags. That is now an assertion in `clean_world_config_test.py`, not a remembered fact, so nobody
"simplifies" the new master flag away later.

**`pbrs_material` and `pbrs_belief` had no gate at all** — `_pbrs_term_active` returned `True`
unconditionally for both and neither fold had an early return.

**`VICTORY_VALUE = 30.0` was a module constant** in `reward_weights.py`, read at two sites, reachable
by no flag. A ±1 terminal was not expressible.

### B1 — `--hand-shaping` (master) + `--pbrs-material` / `--pbrs-belief` (individual)

`hand_shaping=False` makes all EIGHT `_fold_*_pbrs` early-return AND zeroes the whole BIAS class,
`no_progress_tax` included — which is exactly what distinguishes it from `--all-shaping-pbrs`, whose
whole design keeps that one tilt as the acknowledged anti-stall bias. The two individual flags are
**deliberately independent of `all_shaping_pbrs`** rather than folded into it, for the
anti-correlation reason above.

⚠️ **The honest framing, and it belongs in the experiment's write-up rather than buried here.** Every
PBRS term is **policy-invariant by construction** (`Φ(terminal)=0`, telescoping — the manager's own
docstrings and `928a00b`'s verification say so). Removing them therefore **cannot change the optimal
policy**; it changes learning dynamics and conceptual complexity. The clean-world claim's real content
is "the hand terms cost more in interference and tuning than they buy in credit assignment", **not**
"the hand terms bias the objective". The only class that biases the objective is BIAS, and that was
already flag-zeroable.

**A DRY fix rode along, and it is the kind that prevents a whole failure class.** The eight fold
conditions and `_pbrs_term_active` were two hand-maintained copies of one set of predicates — so
`reward_class_composition` could advertise a composition the folds did not implement, with nothing
comparing them. Every fold now calls `Gen3RewardManager._hand_pbrs_on(name)`, which delegates to the
same `_pbrs_term_active` the census reads. One declaration. The revert-catcher for it is behavioural
rather than structural: run a real four-turn `process_turn_reward` sequence that moves HP, spikes,
boosts and status, and assert that no term the census calls inactive ever became non-zero (the
default config emits all 7 + the tilt on that sequence; the clean config emits nothing at all).

### B2 / B3 — `--victory-value`, and the pre-cap tie

`RewardConfig.victory_value: float = 30.0`, substituted at both terminal sites. The second site is
B3: `finished and not won and not lost and turn < cap` — a **pre-cap tie** — shared the decisive-loss
branch as a hardcoded `-VICTORY_VALUE` literal. It now reads the field, so a ±1 arm cannot score a
rare tie at −30 beside a −1 loss. A 250-turn TIMEOUT still takes `draw_penalty`, unchanged.

Draw = loss needed no new flag: `--draw-penalty -1.0` alongside `--victory-value 1.0` expresses the
owner's ruling exactly.

🚨 **THE OUTCOME ORDERING IS THE HAZARD, not the magnitudes.** `draw_penalty = −35 < −30` exists
precisely so that stalling to the turn cap is strictly worse than losing cleanly. At `{+1, −1}` a
`draw_penalty` of `0.0` **inverts** it — the 250-turn stall becomes the best non-winning outcome and
a losing agent's optimal play is to run out the clock — and with `no_progress_tax`, `stall_tax` and
Φ_progress all removed, **nothing in the clean arm opposes that.** `resolve_config` now prints a loud
`[Reward] ⚠️ ORDERING` line whenever `draw_penalty > -victory_value`. A WARNING, not an error: an arm
may legitimately want it. `--victory-value <= 0` *is* refused outright (it flattens or inverts
win/loss, which trains correctly toward the wrong objective and no metric names it). **Stall rate and
mean game length are a PRIMARY safety endpoint on this arm.**

### B4 — `--win-prob-pbrs-source <ckpt>`: the potential can be FROZEN

v104's shield has one hole, stated in its own docstring: the invariance theorem assumes φ is a
**fixed** function of state, and ours is a head inside the network being trained, so exact invariance
holds per rollout and degrades to approximate across them. A frozen source removes the hole.

**One seam.** Only `winprob_pbrs.phi_model(model)` is new — it returns
`model._winprob_phi_source or model`, and `buffer_potentials` plus the bootstrap read it. Everything
downstream (terminal/truncation conventions, in-place reward mutation, the GAE re-run, the
`train/pbrs_*` metrics) is untouched. The loading is `--distill-teacher`'s path verbatim
(`_resolve_zip_and_config` → `load_foreign_opponent` → `set_training_mode(False)`), with the same
`os._exit(FATAL_CONFIG)` on a bad path rather than a crash-restart loop, and the same exclusion from
pickling — `_winprob_phi_source` joins `_excluded_save_params`, because a frozen foreign model must
never be embedded in our checkpoints.

⚠️ **A FULL frozen extractor forward is required; there is no head-only shortcut.**
`WinProbHead.forward(value_pooled)` consumes the whole-board value pool produced by *that* network's
own trunk with its own weights (`CLSPool.value_cls -> WinProbHead`). Running the frozen head over the
LIVE trunk's pooled features would compute a function of a representation the head never saw AND
would drift with the live trunk — destroying the exact property the frozen source exists to buy. The
frozen forward **replaces** the live-φ one rather than adding to it, so the compute is unchanged
(~1/`n_epochs` of one epoch); the new costs are one frozen extractor of memory and one load at
startup.

⚠️ **Two forwards on the post-rollout observation now, and the split is load-bearing.**
`last_values` is the GAE bootstrap and must stay the LIVE critic's — it is the collector's own
quantity, and the recomputed advantages have to be the shaped-stream counterpart of the ones
collection produced. φ(s_T) must come from the φ network. With no source the two coincide and it
stays ONE forward exactly as before; with one, frozen φ on the buffer rows and a live φ on the last
row would break the telescoping at every truncation boundary. Both halves are revert-verified.

A **prior-generation** φ is viable and is the point (that is where a mature potential lives):
`load_foreign_opponent` validates the obs FAMILY, and `_phi_obs` passes only the keys the source's
own space declares — the same filter the exploiter-distillation teachers use in `train()`.

**The coefficient carries the [−1,+1] mapping; φ stays `σ(logit)`.** `φ' = 2p − 1` is equivalent to
`coef ← 2·coef` plus a per-step constant `coef·b·(γ−1)` — at `b = −1` that is `+1e-4·coef` per step,
a small but **wrongly-signed stall incentive** in an arm that has deleted every anti-stall term (a
250-turn game accrues ≈ `0.025·coef`). Worse, it breaks `successor_potential`'s `φ(terminal) := 0`
convention, which is correct for a [0,1] potential and is the MIDDLE of a [−1,+1] one. So the ledger's
`2·P(win)−1` phrasing is implemented as `--win-prob-pbrs-coef 2c`, not as a second spelling of φ.

**`--compile-trainer`: the source is left EAGER, deliberately.** The compile patches the bound
`forward` of the LIVE policy's extractor for the per-minibatch train step; the frozen source runs once
per **rollout**, so a second Inductor graph buys a warm-up and nothing else. ⚠️ **UNEXERCISED and
recorded as such:** a real CUDA `torch.compile` with a frozen source attached has not been run —
`compile_trainer_extractor` refuses a non-cuda device, so the CPU test tier structurally cannot reach
it. What IS tested is the seam that makes it safe: the compile module never names
`_winprob_phi_source`, and replacing the live extractor's bound `forward` with a poisoned callable
leaves φ unchanged.

**Provenance.** The path is recorded on `ModelVersion` and **inherited on a flagless resume** — a
resume that silently reverted to live-φ would trade exact invariance for approximate with nothing
saying so. Startup prints the resolved zip plus its `arch_signature` and `config_version`: a
clean-world run is uninterpretable if the identity of its frozen potential is not pinned.

### The clean-world reward flag set, verbatim

```
--no-hand-shaping --victory-value 1.0 --draw-penalty -1.0 \
  --win-prob-mode read_only --win-prob-pbrs-coef <2c> --win-prob-pbrs-source models/<rev1>/checkpoints/<ckpt>.zip
```

`--win-prob-mode` governs the LIVE head only here — whether it trains as a diagnostic. `read_only` is
the right choice: risk-free, and it keeps a live φ trajectory to compare against the frozen one, which
is a free measurement of how far the potential has drifted from the run's own beliefs. Control arm =
today's defaults, same seeds/teams/steps.

**Raised but NOT solved, from probe N §7:** `train/pbrs_reward_share`'s denominator is the *unshaped*
stream, which on this arm is terminal-only and therefore near zero on most steps — the metric will
read enormous and uninformative exactly where coefficient sizing matters most. It needs a companion
for this arm (shaping absmean against the terminal magnitude, or per-episode discounted shaping sum
against ±1). Flagged as a build item.

### Tests

`agents/training/clean_world_config_test.py` (35, new) — default byte-identity across all four
fields and the parser; the clean-config smoke against the reward's own census; the
"no pre-existing combination reaches it" assertion; per-flag independence from `all_shaping_pbrs` in
both directions; the fold/census agreement over seven configs, checked by running real turns rather
than by re-reading the predicate; ±victory at both terminal sites including the pre-cap tie; the
ordering warning firing and *not* firing; resume-immutability with a round-trip through the error
message's own suggested flags; the v105 migration. `agents/training/winprob_pbrs_test.py` grows from
22 to 35 with the frozen-φ group, whose centrepiece is the identity test: point the source at the
run's **own current checkpoint**, through the real `load_foreign_opponent`, on a real
`Gen3FeaturesExtractor` — every φ must come back **bit-identical**, with an anti-vacuity twin that
drifts the live weights and requires the frozen φ not to move while the live φ does.

Revert-verified: re-inlining a fold's `all_shaping_pbrs` condition, restoring the pre-cap tie's
`-VICTORY_VALUE` literal, dropping the source from `phi_model`, and letting the bootstrap fall back
to the live head each fail a named test.
---

## v106 — `gen3_progress_clock_intent_v1` + `gen3_staller_protect_rng_v1` (2026-08-29): three OPT-IN fixes to the no-progress tax and the eval roster's one coin

**Grouped as one entry because all three are the same kind of change and land together: a defect
found by measurement, repaired behind a flag whose default is the shipped behaviour, so the fix is
a decision someone takes rather than one that happens to a live run.** Nothing here alters a
flagless run — the committed obs golden's 991 per-decision hashes are unchanged, and
`progress_clock_test.py` pins the clock's default `(n, last_penalty)` trace against a capture taken
from the pre-fix implementation.

### What was measured, and by whom

Two probes ran on 2026-08-29 and disagreed with the code in the same place from opposite
directions.

**Probe M — the CENSUS** (`designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md`):
over 2,677 battles / 147,204 decisions, **23.3% of every decision the model makes carries the
`no_progress_tax` charge** — 6.80 charges/battle × −0.15 = **−1.02 reward units per battle, 3.4% of
a win**. It then classified where those charges land.

**Probe N — the INTENT ARCHAEOLOGY** (`.../no_progress_tax_review_2026-08-29.md`): the design
document, the originating commit, and every line of `ProgressClock.update`, read against each
other. Its headline is that the owner's recollection — "I wrote it to punish only obviously
irrefutably poor choices" — is *half* right, and the wrong half is the interesting one.

**Together they found that 79% of all charges land on two paths, and neither is what the design
specified.**

### F1 — `--progress-decision-tense`: both gates now describe the decision being charged

`TurnDelta.phase_is_forced_switch` reads `curr_ctx.phase`. `curr_ctx` is built at embed time, from
the **upcoming** request — so the flag on the window of action `a_t` carries the phase of decision
`t+1`. `ProgressClock` has read it as "was this decision forced" since the term's originating
commit (`adc0fe4`), and the same call additionally hands the **upcoming** request's `legal` to the
trapped-vs-wall helplessness gate, which the design specified as "a switch being legal **this
decision**".

Both halves, measured:

* the clock **sits out on 19,503 full-agency decisions** — probe M's SITOUT class, 13.2% of the
  corpus and its **costliest arm at −5.1pp** — because our mon happened to be KO'd on them;
* the **zero-agency post-faint replacement**, which no action can rescue, is **charged 63.9% of the
  time (12,432 charges, 36.3% of all charges)** — and the design forbids exactly this in one
  sentence.

Probe M reached the alignment by measurement rather than assumption, and the intuitive answer was
wrong: testing both candidates against the clock's documented sit-out gave **0 / 10,442** violations
under the `t+1` reading against **8,710 / 10,424** under `t`. `progress_clock_test.py` runs that
same discriminator over a synthetic decision sequence and asserts the 0 moves to the other column
when the flag is on — the fix restated as the measurement that found it.

**The root cause is instructive and is why no test caught it for a year.**
`phase_is_forced_switch` was minted eleven days before the clock (`045e8b8`) **for the obs history
slot**, where "what phase did this window END in" is the natural and correct read. `adc0fe4` reused
the attribute for a question it does not answer. Same name, different tense — and **both readings
are true statements about the same delta**, so there is no assertion that could have failed.

So the fix does not re-point it. It adds a NEW field, **`TurnDelta.decision_was_forced_switch`**
(`prev_ctx.phase == "forced_switch"`), set at all four construction sites including the legacy
snapshot-diff builder, and threads **`legal_prev`** alongside `legal` into `ProgressClock.update`.
`phase_is_forced_switch` is left alone: the obs decoder, `opp_intent_labels.py` and
`reward_manager.py` all want the closing tense. `decision_was_forced_switch` has exactly one
consumer — the clock — and nothing encodes it into the observation (the TurnDelta lag frames were
deleted by `gen3_frame_deletion_v1`; the test asserts `TurnDeltaEncoder` emits a byte-identical
vector with the field flipped).

`legal_prev` is threaded unconditionally and CONSUMED only under the flag; where a caller cannot
capture it (`RewardTrackingMixin`, which builds contexts without a legality snapshot) it degrades to
`legal` — the pre-fix reading — rather than to "trapped", because a fix that silently zeroes a term
is worse than the off-by-one it replaces.

### F2b — `--progress-switch-freeze`: a voluntary pivot freezes instead of paying

**Charging a voluntary switch that lands nothing was DESIGNED, in writing, deliberately**
(`design_markovian_reward_and_features.md:714-718`), inheriting the job from the collapsed
`switch_bouncing_tax`. That is not drift. **The drift is what was around it.** In the design's
composition the same pivot also collected `switch_base +0.5`, `se_switch +0.2`,
`escape_threat_switch +0.25` and `pivot_* +0.10..0.15`; the net incentive on a tempo pivot was
POSITIVE. `928a00b` zeroed **every one of those credits and explicitly kept the tax** ("zero every
BIAS term except the no_progress_tax tilt"), and `43673ed` made that composition the default. **The
sign of the switch incentive flipped and nobody re-derived the term's meaning.** It is a COMPOSITION
drift, not a code drift — which is exactly why no test caught it either. The reward manager still
carries the comment that states the original intent as a quantity: *"the flat no-progress charge
(−0.15) does not out-weigh the per-switch reframes"*.

The empirical half, from probe M: **−0.101** expected charge per voluntary switch against **−0.010**
per move, a **10:1 differential**; and *within* the switch branch the tax's discrimination is
**INVERTED** — Δ mean `d_out` **+0.0103 [+0.0076, +0.0131]**, i.e. the charged switches are worth
*more* win probability than the exempt ones. The mechanism is structural: `_is_progress` has eight
clauses and **not one of them can be caused by a switch** (the ~27% of pivots that escape do so
because the opponent also committed, a contact ability fired, or a residual was already ticking).
The predicate asks "did OUR ACTION advance the board", and a pivot's whole value — position, tempo,
damage avoided — is a *state* fact it never looks at. So the tax prices an action KIND rather than
discriminating within it. **42.7% of all charges.**

Under the flag such a window FREEZES (no increment, no charge) rather than being classified NO_OP.
The freeze sits AFTER the classification, so the pivots that legitimately reset the clock still
reset it.

**Probe N offered two spellings and this is the one that ships.** The alternative (F2a — give
`_is_progress` a switch clause keyed on the incoming-KO belief falling, or on the switch-in's best
multiplier beating the outgoing mon's) would reintroduce precisely the hand-tuned switch heuristic
`928a00b` deleted on the argument that switching value is **learnable** from Φ_mat + `pbrs_belief` +
the terminal. Probe N recommends against it on the record of that commit, and so do we.

**The honest cost, stated rather than buried: a pure A↔B switch-loop becomes free.** The anti-stall
job is not lost — a pivot-loop still pays on every MOVE turn between the pivots, and `--draw-penalty`
plus the 250-turn forfeit remain the hard backstop — but **stall rate and mean game length are the
canary** on any arm that enables this.

### The config surface, and what "OFF by default" is worth

`progress_decision_tense` / `progress_switch_freeze` are `RewardConfig` fields: resume-immutable,
value-checked by `check_reward_config`, recorded in `model_config.json`, `MODEL_CONFIG_VERSION`
**105**, **no `ARCH_SIGNATURE` bump** (they change what the clock counts, not any weight shape). The
v105 migration defaults both to `False` for every archived config — not a guess, since the flags did
not exist, and `False` reproduces what every generation through gen-15 trained under.

They reach the clock through ONE call, **`ProgressClock.apply_reward_config(cfg)`**, used by both
`gen3_env.py` and `reward_tracker.py`. That is deliberate: a hand-threaded reward field was once
silently missed on the eval path and eval then measured a different reward than training, and the
`RewardConfig` docstring says so. One call means training and eval cannot drift on what the clock
does.

**Both are RETRAIN-CLASS when ON, and the blast radius is measured rather than asserted.** `n` is
the obs scalar *and* the charge basis — that identity is the Markovian design's whole premise, so a
fix that moved only the reward would break the thing the clock exists to provide. The confinement
test captures the deterministic golden 6-battle set under each arm and names every differing cell:
**`--progress-decision-tense` moves 49/991 decisions, `--progress-switch-freeze` 153/991, and in
both cases the ONLY column that ever differs is 1602** — `turns_since_progress` itself. No other
block moves, no dim moves, the decision count is unchanged (the trajectory does not branch).

**The control arm needs no code, and this is worth knowing before building one:** `--no-progress-penalty 0.0`
already zeroes every charge while leaving the obs counter ticking. (One cosmetic caveat probe N
raises: `_bias_term_active` does not read the magnitude, so `reward_class_composition` will still
announce `1 BIAS (no_progress_tax)` on such a run.)

### `gen3_staller_protect_rng_v1` — the eval roster's only coin becomes per-instance

`Gen3StallerPlayer` and `Gen3StallerV2Player` flip a 60% coin for Protect. **Every other bot in the
roster is deterministic**, and until now that coin came from the process-wide `random` module.

This surfaced as a **failed integrity check, not a code review**. The transfer-coefficient cell
(`.../transfer_coefficient_cell_2026-08-29.md` §4) ran the design's own falsifier: in a unit where
the treatment never fired, the two arms must be the *same battle*, so the paired difference must be
exactly 0. It came back **exactly 0.0000 over 2,693 pairs on the seven deterministic bots, with zero
divergences — and failed on exactly these two** (755 pairs, 4 divergences). That confinement is what
identified the cause: the searched arm awaits an executor while the control runs inline, so the
shared global coin lands differently with no treatment involved. Unbiased noise (3 divergences
favoured A, 1 favoured B) that inflates the discordant count in both directions and widens the
interval; the cell reported it and deliberately left the repair to whoever owns the shared eval
opponents.

The repair is opt-in: `protect_seed=<int>` at construction, or **`$GEN3AI_STALLER_SEED`** for every
staller in the process — the hook a paired-arm harness needs when it does not own the construction
site, since the bots are built deep inside `env_factory` / `eval_worker`. Either installs a private
`random.Random`; with neither, the coin is the `random` module itself and the default is
byte-identical. An unparseable `$GEN3AI_STALLER_SEED` **raises** rather than falling back, because a
seed that was meant to be set and silently was not would make an arm look reproducible while it is
not — the exact failure this fix exists to remove.

### Tests

`progress_clock_test.py` (24) — the default-path trace captured against the pre-fix implementation;
every fix assertion runs the same window through an OFF clock and an ON clock, so a revert collapses
the two and the test fails; probe M's alignment discriminator on a synthetic sequence.
`progress_clock_obs_confinement_integration_test.py` (6, `sim`) — the one-column confinement
measurement above. `turn_delta_event_fold_test.py` (+5) — the two tenses pinned apart across all
four `(prev_phase, curr_phase)` combinations, and the encoder's byte-identity with the new field
flipped. `opponents_test.py::TestStallerProtectRng` (7) — seeded arms agree under adversarial
interleaving, with its own revert arm showing unseeded arms diverge. `reward_defaults_test.py` (+3)
— both new defaults pinned OFF, and the census proven unchanged with either on.
---

## v107 — `gen3_q_winprob_head_v1` (2026-08-29): a Q head on the pointer's own action tokens

**The gap it closes, stated exactly.** Every value readout this network owns evaluates a STATE —
`value_net`, `WinProbHead`, `ValueDistHead`, `ShadowValueHead`, the evidential Beta head. So a
question a search teacher answers for free — *what is my win probability if I click Rock Slide?* —
costs us eleven simulator re-rolls plus eleven forwards, because the successors have to be
MANUFACTURED before anything can score them. That is the whole reason probe L's ranking "is not a
quantity the network computes": it is the head composed with a simulator, and PPO performs no such
composition (v104's entry, above, records the argument in its original form).

`QWinProbHead` is that composition, amortized. One shared zero-init readout scores each of the
eleven actions from the token of the entity that action selects — the SAME per-action tokens
`PointerNativeActionHead` scores, read off `stash.pointer_inputs` — with `value_pooled` as the
board context, and stashes `last_q_winprob_logits [B, 11]`. One forward, eleven `P(win | s, a)`.
Ledger 229e9f1 (the route-2 / R1-factory convergence) and 5edbd05 (E5 as a closed loop:
predict → ground → prioritize → teach → measure) are the design of record; this entry is step 1
(PREDICT) and step 2 (GROUND) built, with step 5's meter (MEASURE) shipped as a script.

**🚨 The starvation trap is the reason the label plumbing looks the way it does.** On-policy data
labels exactly ONE action per state, and probe L measured the policy sampling its own
better-ranked alternative at a median **p = 0.002**. A Q head trained on that stream is untrained
precisely on the never-tried moves — i.e. confidently wrong on the entire set a per-action readout
would ever be consulted about, because the shared scorer generalizes the taken-action signal onto
the unvisited columns with nothing to correct it. So the head's primary labels are
COUNTERFACTUAL: per-action re-rolls from the R1 factory, carried as an ADDITIVE-OPTIONAL `q_labels`
field on the existing v1 label row.

**What ships**

*Architecture (STRUCTURAL, `q_winprob_mode`).* `none` builds nothing — byte-for-byte the baseline.
`read_only` builds the head. There is deliberately NO `shaping` value: a per-action readout
carrying a counterfactual label is a strictly larger leak surface than a per-state one, so every
input is detached INSIDE the forward and trunk exposure becomes a later decision that owes its own
gate. Consequences: `pi`/`vf` are bit-identical whenever the head is built, and
`grad/q_winprob_share` reads exactly 0.0 by construction rather than by convention.

It differs from the four cf readouts (v98/v99) in one way that is not cosmetic: **the forward DOES
call it.** Eleven Q values are only useful if the forward that chose the action publishes them —
a rollout, an eval and the prober all need to read them without a second pass. So the contract is
not "never runs" but "runs and publishes only", and the tests pin that shape instead.

It is built LAST in `__init__` for the usual reason (SB3 restores optimizer state POSITIONALLY —
the ai_v6_13 "128 vs 5" crash — and appending leaves every earlier module's init RNG draw
untouched) **and for a second one that is specific to it**: it sizes its projections from the
POINTER CELL WIDTHS, so it must be constructed after every module that widens one (the op, the
intent cells, the pair-outcome cells, the switch branch, the conditional threat).

NO `ARCH_SIGNATURE` bump — `none` is byte-identical and `read_only`'s only output is a stash, so
`check_compatible`'s string compare is the sole gate, and it has to be: a flipped flag produces no
shape error anywhere. A resume that dropped it would load "successfully" and quietly stop training
the head; one that added it would supervise a freshly random head as the run's trained one.

*Labels (ADDITIVE-OPTIONAL at schema v1, not a schema bump).* `q_labels` is a LIST OF OBJECTS —
`{"action": int, "label": float, "n_rollouts": int}` — never parallel arrays. Three same-length
lists can be written in the wrong order by a producer and read as valid by the consumer; a
per-action object cannot, which is the order-mismatch rule applied to a wire format. `taken_action`
rides beside it, pairing with the existing `outcome_label` for the weak fallback. A malformed entry
is a counted FIELD skip: the row survives with its other three label streams intact, because a
producer bug in one stream must not cost the trainer the rest. Two new liveness scalars —
`cf/q_label_coverage` and, the one that matters, `cf/q_labels_per_row`, which is what separates a
live counterfactual factory from an on-policy trickle.

*Training (TWO coefficients, and the split is the point).* `--q-winprob-coef` folds a MASKED
binomial NLL over exactly the labelled `(state, action)` cells, normalized by `Σ(mask·n)` — mean
NLL per ROLLOUT, so the coefficient keeps its meaning across producers with different R AND across
minibatches with different label DENSITY. At full coverage it is EXACTLY
`cf_terms.cf_binomial_nll`, which is pinned rather than claimed. An unlabelled cell contributes
zero to numerator and denominator alike, never a zero target — a zero-filled absent label is
indistinguishable from a confident "this action loses".

`--q-winprob-onpolicy-coef` is the weak fallback (recorded outcome, at the taken action, n≡1),
**default 0.0 and separately weighted so the two can never be confused in a run's provenance**. It
exists because a starved-factory run should have something to show, not because it substitutes for
counterfactual labels; its bias is stated at the flag, in the fold's header, in `hparams`, and in
the launch banner.

Both fold on the SAME sample and the SAME extractor forward every cf term shares. Both re-apply
the head rather than reading `last_q_winprob_logits`, and the reason is specific: that forward runs
under `no_grad` whenever nothing downstream of it needs a graph — a condition computed from the
*scalar* term's settings, which know nothing about this one — so a term folded from the stash would
train exactly nothing while every metric looked healthy.

*The meter (E5 step 5).* `python -m main.q_amortization` compares the head's per-action row against
the prober's own one-ply `lookahead` sweep: Spearman, top-1 agreement, and the **amortization
residual**. Shrinking ⇒ the AlphaZero ratchet (search's value has migrated into the net, and search
must deepen to add anything); stubbornly large on a class of states ⇒ those are the states that
genuinely need live search, a triage signal rather than a defect. Two caveats are written into the
script itself: it is a PREDICTIVE meter and says nothing about behavior (iteration 2's lesson), and
its ground truth is itself a model read — `lookahead` scores each re-rolled successor with the same
checkpoint's critic, so a badly calibrated run shows a small residual against a wrong target.
`--self-check` runs the init-state sanity with no checkpoint, no traces and no simulator, and is
gated in the suite.

*Metrics.* `q_winprob/*` under its own prefix so a per-ACTION number can never be read as the
per-state win head's. `label_coverage` / `labels_per_row` are the documented FIRST read. The
`pred_spread` vs `label_spread` pair is the column that distinguishes a head that amortized the
SEARCH from one that amortized the VALUE — a head predicting each state's mean scores well on
`abs_err` and has learned nothing per-action. `train/q_winprob_loss` is ABSENT rather than 0.0 when
the fold starves, because a defaulted zero is a perfect score for a head that trained on nothing.

**Status: LATENT.** Nothing is enabled. `q_winprob_mode` defaults to `none`, both coefficients to
0.0, the production config does not carry the flag, and the R1 factory does not yet emit `q_labels`
— which is the next piece, and the one that decides whether any of this measures anything.

**Tests** `q_winprob_head_test.py` (18): the action-space column order proved from the inside (an
invalid move slot must zero exactly ITS column); zero-init ⇒ P = 0.5 exactly, asserted on a REAL
`MaskablePPO` build because SB3's ortho-init clobbers extractor zero-inits and a claim made on a
bare module is a claim about a path production does not take; OFF byte-identity and ON
bit-identity in pi/vf; the append-never-insert rule asserted on every PRIOR parameter tensor AND on
their order, not merely on the forward's output; the state_dict key census in both directions;
head-only proved by backprop (trunk takes nothing, the head takes something — either failure alone
is silent); switch-score equivariance under a team permutation, on a head whose scorer has been
un-zeroed so the property is not trivially true; the stash seam and the pointer-width dependency;
the v105 gate and migration. `q_winprob_terms_test.py` (21): the masked likelihood equalling the
scalar one exactly at full coverage; a masked cell taking no gradient at any logit; evidence
weighting as a 4× gradient ratio; the wire format's action index surviving to its own column
through non-adjacent out-of-order entries; every malformed-entry class as a FIELD skip;
duplicate-action keep-last; the fallback needing BOTH halves; an older producer's row supervising
nothing; the fold's read seam, its published zero-coverage, and a stale pointer stash raising
rather than supervising the wrong board. `q_amortization_test.py` (6): the shipped self-check
through both entry points, and `spearman` returning **None** on a constant row — an untrained head
emits one by construction, and reporting it as rho = 0.0 would merge "has learned nothing" with
"has learned something uncorrelated", which is the distinction the whole probe exists to make.

## The STALL-TAIL HARVEST pipeline — `main.harvest` / `winprob_finetune` / `main.harvest_meter` (2026-08-29)

**Offline tooling, no architecture change, no flag, no version bump.** Nothing in a training run
imports any of it; a trained checkpoint is read, never written. Recorded here because it is the
ai_v12 head-repair backbone and because three of its findings change what a reader should believe
about the trace corpus.

**What it does.** Three stages, each usable alone. `main.harvest` mines late-game decision states
from current-arch eval traces, re-scores each through the SUBJECT checkpoint's win-prob and
evidential heads, ranks them by a declared blend of head-vs-realized gap, confessed Beta
uncertainty and a doomed-tail bonus, and buys a `k`-of-`n` label per state by picking the recorded
battle up mid-game through the reconstruction layer and rolling to a terminal `n` times with fresh
dice. `agents.training.winprob_finetune` fits the `WinProbHead` on those labels with the trunk
absent (not merely frozen — a cached-`value_pooled` two-phase split), under a binomial likelihood
and inverse-frequency turn-slice weights. `main.harvest_meter` re-runs probe O's battery PAIRED,
pre vs post, on a battle-level holdout the producer committed to before buying a single label.

**Why a head fix and why now.** Probe O (`stall_tail_head_reading_2026-08-29.md`) measured the
win-prob head ending above 0.5 on **34.8%** of the tails of games it loses by construction, 4.3x
the ordinary-loss rate. Ledger `b63a96f` established the labels are correct and all three
mechanisms are DATA-shaped, so the defect is a CENSUS problem — missing discrimination mass at the
late time-slices — not a propagation one. This manufactures the mass; whether that suffices is the
reducibility question the meter exists to answer.

### Three measurements that changed the design, in the order they forced it

1. **A trace's recorded `win_probs` is not the subject's reading.** Measured up to **0.135** apart
   on the subject's OWN traces (the trace was written by an earlier checkpoint of the same run). So
   every candidate is re-scored by a fresh forward, and the rollouts run the subject as trainee via
   `ckpt_override` — otherwise `k/n` estimates the value of whichever policy happened to record the
   battle.

2. **Priority alone fills the sample with doomed states** — a 40-state draw came from 4 battles,
   all losing. A head fit only on those has a trivial perfect solution ("say 0 at every late turn")
   that is a bias, not a repair, and would destroy what probe O found the head does RIGHT
   (`LONG_WIN` reads **0.986** at 128 median turns). Hence a capped doomed share.

3. **Capping it was not enough**: the remainder, also ranked by priority, took **1** state of 300
   from a won battle — `|phi - realized|` is ~0 on a correctly-read win, so the control stratum was
   selected out of existence by the rule that makes the doomed stratum good. The coverage stratum
   is therefore balanced on the realized outcome. The meter's long-win control is only an honest
   falsification because of this, and `verdict_lines` names a detection gain bought with a control
   collapse a **FAILED RUN** in those words.

### 🚨 THE FAILED PILOT, and the stratum it bought

Pilot 1 ran the whole pipeline end to end — 200 states / 41 battles / **6,281 adjudicated
rollouts** (31.4 per state, **1.86%** timeouts, label noise floor 0.052) — fitted the head, and
metered it. The head came out **worse on every population**, and the untouched long-win control is
what identified why: held-out long losses `phi_T` **0.070 → 0.607** and long-WIN control `phi_T`
**0.943 → 0.567**, both collapsing toward ≈0.6. The head lost its dynamic range.

Not "late means lost" — that would move the two populations apart, not together. It is regression
to the SAMPLE mean, from a measured mismatch: the fit set spanned turns **60–152** (p50 90) with a
mean MC label of **0.621**, while the meter reads turns **96–239**, and **29.3% of eval turns were
above the harvest's maximum turn**. The head was fit on mid-game positions it wins 62% of the time
and never shown a losing tail.

Fixed by `--tail-frac` (default 0.5): a reserved share of the doomed budget takes a battle's last
`TAIL_K = 5` decisions, `TAIL_K` matching the meter's window on purpose. Same region, different
battles — the battles are held out by the producer, so this is generalization, not leakage.
`--tail-frac 0` ablates back to the failed behaviour, and `harvest_test` pins the property.

**The durable lesson generalizes past this pipeline: a label factory that never samples the region
its meter scores is extrapolating, and label QUALITY cannot fix a label-LOCATION problem.** Pilot
1's labels were excellent; they were labels for the wrong states. And it took the CONTROL to see
it — every doomed-tail metric alone reads "detection got worse", which is ambiguous with an
ordinary bad fit; only the control moving the opposite way names the collapse.

### The ANCHOR — `--anchor-coef`, default 0.3, and why a default of 0 would have been wrong

Pilot 2 added the tail stratum and **halved** the damage but did not remove it (control `phi_T`
−0.165 vs −0.376). The residual is the selection itself: a harvest is chosen where the head is
wrong, so its label mean is far from the population's, and an unconstrained fit of a 6-parameter
head lands on the sample mean. The damage scaling with the offset across two independent runs is
what identifies the mechanism.

`--anchor-coef` adds `c * mean((z − z0)^2)` against the SUBJECT's own logits — a trust region on
the head's FUNCTION, not on its weights. At **0.3** the control holds (`phi_T` −0.033), every
categorical metric is unchanged, and cap `phi_T` moves the right way (−0.109, n=3). At 1.0 and 3.0
the fit does not move at all (best epoch 0), so 0.3 is the largest dose that still lets the labels
speak. It defaults ON because shipping 0.0 would ship a measured-destructive setting; `0` opts out
and reproduces the pilots. The anchor is captured before the resume branch — taking it from a
partially-trained head changes the objective mid-run, and the bitwise resume test caught exactly
that.

**Honest state after the pilot: the pipeline is complete and non-destructive, and the reducibility
question is NOT answered.** 359 labelled states across two runs is a smoke of the machinery.

### 🚨 The cap-record finding — the KNOWN rust `forcelose` gap, now quantified

A 250-turn game ends by a forfeit logged as `["forcelose", side]`. Without it the record never
terminates and BOTH replay impls refuse it in the `replay` verb `materialize_from_record` depends
on, so **every** model-scored offline path is blocked — not just this pipeline's. Archive-wide:
**689 cap records, 543 (79%) carry it.** Within the current-arch corpus it is **8 of 48**, split
*exactly* at the documented 2026-08-24 rust `sim_bridge` fix (runs `_0819`.._0824`: 0 of 40;
runs `_0825`.._0828`: 8 of 8). The scarcity is therefore **transient** — every post-fix run adds
replayable caps — but it is why the doomed-tail population is caps PLUS long losses today, and why
the two are stratified separately and never pooled (the head reads them very differently:
`detect_le05` 0.652 on caps vs 0.94-0.95 on long losses).

**The missing forfeit is deliberately NOT synthesized.** Appending one would make all 40 replay and
the battle is a LOSS at exactly 250 turns, so a trainee forfeit is overwhelmingly likely — but
"overwhelmingly likely" is not a basis on which a label factory may manufacture an ending, and a
record may lack its terminal because the *opponent* forfeited. Skipped, counted, published.

### Replications worth recording

`STALL_LOSS` is still **empty**: the current-arch corpus holds **zero** non-capping stall losses,
so probe O's finding that "stall pattern" and "cap ending" are one population reproduces on a
disjoint slice. And supply is not the constraint anywhere else — 29,655 current-arch traces, **100%
carrying a reconstruction sibling**, and **54,487** labelable decisions at turn >= 60 across
**1,404** battles before the cap preflight, **44,558 / 1,364** after it (the difference is exactly
the 40 unterminated cap battles and their 9,929 decisions).

### Contracts and conventions

The label schema (`agents/training/harvest_schema.py`) is versioned and validated on every write
AND every read, kept deliberately separate from `cf_audit`'s v1 (which is a single-run bias-map
contract with a live consumer). `load_obs` is the one resolver both sides call and it verifies
`obs_sha1` — the check exists because `cf_audit` shipped a bug where `obs_npz` rows ignored
`decision_idx` and every default-mode label was rejected, with both halves of the contract tested
and neither running the other's real output. A timed-out rollout is its own bucket, excluded from
numerator and denominator alike. Artifacts land under `utils.paths.harvest_dir()` (gitignored,
`$GEN3AI_HARVEST_DIR` overrides), never inside `models/`.

## `gen3_search_depth2_chunk_gap_v1` (2026-08-29): the depth-≥2 search replay was fed a protocol with a HOLE in it — and the "chunk-transport double-encode" was never a transport bug

Task #38, filed off the first live depth-≥2 run (2026-08-23) as "chunk-transport double-encode —
active-mismatch warnings + a mojibake `KeyError` on non-ASCII nicknames". Both symptoms are ONE
defect, it is in the Python composition rather than the transport, and the encoding half is a red
herring that a fixed test now pins so the next reader does not re-derive it.

**The defect.** `SearchSession.expand_many` returns the arm's OWN ply — `search_driver.js` slices
`sess.chunks` at a per-expand baseline, because a deserialized battle cannot re-emit the historical
`|request|` lines a materializer needs, so composing the replayable protocol is deliberately the
caller's job. `ExpandedNode`'s docstring said the opposite ("the COMPLETE one-sided view, root →
this node"), and `main.search_dividend.search._expand_ply` believed it: it passed that bare suffix
as `Branch.chunks` alongside an `actions` list spanning the whole path. A depth-`d` successor was
therefore replayed as `prefix` (ending at the ROOT request) followed by ply `d`, with plies
`1..d-1` missing. **At depth 1 the two readings coincide**, which is exactly why nothing caught it
for as long as depth 1 was all that ran — and why every existing gate was green: the two callers
that DO deepen, `main.prober.better_line` (`our_suffixes=parent.our_suffixes + [...]`) and
`search_clone_parity_fuzz_test` check 3 (`suffix2 = ext.chunks + d2.chunks`), both accumulate
correctly, so the contract was right everywhere except in the one place the docstring was read.

**A hole is a different battle, not a coarser one.** poke-env keeps applying protocol lines to the
board it last saw, so the two reported symptoms follow mechanically:

* a switch inside the gap ⇒ every later reference logs `"Message thinks p1: X is active, but it's
  not"` (measured on a seeded 2-ply fixture: **6 warnings** for the bare-suffix composition, **0**
  for the cumulative one, and one decision fewer);
* an opponent REVEAL inside the gap ⇒ the `|switch|` carrying the species never arrives, so a later
  `|move|p2a: <nick>|…` reaches `get_pokemon` with no `details` and it constructs a Pokémon whose
  *species* is the NICKNAME ⇒ `KeyError: to_id_str(nickname)`.

**The mojibake is in the TEAM FILE, not the transport, and the KeyError does not need it.**
`'ptãra'` is `to_id_str("PtÃ©ra")`, and `data/teams/others/mcmegan/*.txt` literally hold the bytes
`c3 83 c2 a9` where `Ptéra` was meant — a double-encode committed at ACQUISITION time, carried
faithfully by every layer above (verified on both impls). On the same fixture `'airmure'` and
`'tyranocif'` raise identically; the only slot that SURVIVED the gap was the one whose nickname
equals its species (`Jirachi`). So the non-ASCII teams were merely the ones that crashed loudly.
The team bytes are deliberately **left alone** — team files are hashed into `pin_sha` and into
`gen3_team_archetypes.json`, so rewriting one is not a cosmetic edit — and the fact is pinned by a
test instead.

**The fix** is the accumulation the contract always wanted: `TreeNode` gains `chunks`, the
root→node protocol grown one ply at a time on the same line as `path`, and `_expand_ply` composes
`parent.chunks + suffix`. `ExpandedNode` / the `SearchSession` header / `search_driver.js`'s
`expandArm` comment now all state the suffix contract (the driver code is untouched — the JS diff
is comments only, so node↔rust driver parity is unchanged by construction).

**Gates.** `search_test.py::test_a_deepened_branch_carries_EVERY_ply_from_the_root_not_just_its_own`
(no sim; VERIFIED failing on revert) plus a depth-3 companion asserting the standing invariant —
one chunk group per action, in order. `depth2_replay_integration_test.py` (`sim`, ONE battle,
3.5 s) is the semantics end to end over BOTH impls: it picks a nicknamed benched opponent from the
request rather than hardcoding a slot, asserts ply 1 reveals it, asserts the driver's contract is
still per-ply (so a future cumulative driver turns the Python accumulation into a visible
double-count rather than a silent one), and asserts the cumulative composition replays with zero
warnings while the shipped one raises `KeyError: <nickname>`.

⚠️ **The first live depth-≥2 readings (2026-08-23) were taken under this defect** — deepening
engaged as designed (18-20 of ~23 searched decisions, realized depth 3, mean 1.83-2.11) but every
deepened arm was scored on a holed replay or dropped, so those numbers are void. At the DEFAULT
width caps none of this ever reached the shipped sweep: width absorbs the whole budget at every
swept budget, so the sweep ran at depth 1.

---

## TEAM PROMOTION — `python -m main.promote_teams` (2026-08-30)

The 40-team fleet needs 40 legal exploiter trainees; `--exploiter` refuses any trainee outside
`data/teams/sample/`, which holds **32**. This is the tool that widens it, and the widening is a
**seed-recorded UNIFORM RANDOM draw** from the validated pool, per the owner ruling of that day
(ledger 56bfd48): a hand-picked or headroom-ranked fleet makes its own result a *selection*
estimate rather than an unbiased estimate of pool-wide transferability. S1's headroom ranking
(`designs/ai_v12/team_slate_40.md`) is re-scoped to a covariate; its exclusion sets and validation
machinery carry forward unchanged.

    exclusions  →  random.Random(seed).shuffle(sorted(eligible))  →  validate_teams_locally
        →  copy into data/teams/sample/  →  PROMOTION_MANIFEST.{md,json}

**Exclusions are a committed artifact, re-verifiable against run metadata.** They were built from
frozen argv files in a session-scoped job directory that will not outlive the session, so
`designs/ai_v12/promotion_exclusions.json` is the durable copy: 12 taught (rev-2 F5a–e + rev-3
F6a–f, which contains the 9 meter teams and the 3 R3-coverage teams), 12 more pending under rev-4's
frozen R4S3a/b/c argvs (CONDITIONAL and reversible — if rev-4 is abandoned they return to the pool),
and the 2 held-out off-slice transfer instruments. **Union 26, eligible 693 of 719.**
`--verify-exclusions` re-derives every category that names a run from that run's own
`metadata.json` (`matchup_spec.read_recorded_trainee_teams`); the 11 launched runs verify
2-for-2 each, and the 3 unlaunched rev-4 runs report UNVERIFIABLE rather than mismatching.

### The defect the design is shaped around: promotion MOVES, it does not COPY

`TeamLoader`'s universe is the `teams.json` manifests, deduped by **resolved path** — not by team
text. So a team copied into `sample/` and listed in `sample/teams.json` while still listed in
`others/<author>/teams.json` is loaded **twice**, and drawn as an opponent twice as often as its
neighbours. That is precisely the `yak_attack`-was-66%-of-draws defect the 1601→719 dedupe fixed,
recreated on exactly the 40 teams the fleet is measuring. The tool therefore de-lists each promoted
team from its source manifest (leaving the `.txt` on disk, so the change is reversible from the
manifest alone) and then re-loads through `TeamLoader` and asserts the pool total is unchanged and
no sha appears twice. Rehearsed on a full copy of the real tree: 719 → 719 total, 32 → 72 sample,
687 → 647 other, zero duplicates.

### Three more failure modes, each closed by construction rather than by care

* **An invalid team is REPLACED, never dropped** — by the next candidate in the same seeded shuffle,
  recorded in the manifest with its errors and draw position. Dropping would silently shrink a fleet
  whose size is the experiment.
* **A broken validator would condemn the whole pool.** `validate_teams_locally` returns the same
  `{"valid": False}` shape for a dead node bridge as for an illegal team, so a missing
  `deps/pokemon-showdown/dist` reads as 693 bad teams. A known-good curated team rides in every
  batch as a positive control, and infrastructure error strings abort rather than "replace". Same
  genre as *a timeout is never a semantic outcome*.
* **The manifest is write-once.** A second draw under a different seed is REFUSED without `--force`
  — re-rolling the seed until the composition looks acceptable is the selection confound the random
  draw exists to avoid, and nothing else would have caught it after the fact.

Composition is **REPORTED, never corrected**: a random draw reproduces the pool's own archetype and
author-folder mix in expectation, and rebalancing it would put the confound back. Keys are
`sha1(team_text.strip())[:10]` throughout — the strip-normalized convention shared by
`team_archetypes.team_sha`, `MatchupSpec` pins and `TeamWinRateCallback`; a test fails if this
module ever hashes raw text, because the unstripped variant is a recorded derived-key defect
(`coverage_sample.py`, whose tell was every row carrying `"class": "?"`).

Modes: `--dry-run` (plans, writes nothing) · `--draw-only` (manifest for review, no promotion) ·
`--root <copy>` (rehearse the real promotion on a tree copy) · `--verify-exclusions` · `--force`.
A committed demo draw at seed 20260830 — 40 teams, 0 replacements, balance 11 / hyper_offense 10 /
offense 8 / semi_stall 7 / stall 4 — lives at `designs/ai_v12/promotion_dry_run_demo.{md,json}` and
is re-derived by the test suite on every run. Gates: `src/main/promote_teams_test.py` (22 tests,
~0.2 s, no node, no models).
## The PER-ACTION LABEL FACTORY — `cf_producer --q-labels` (`gen3_cf_q_labels_v1`, 2026-08-29)

v107 shipped `QWinProbHead` as a **trained consumer of a stream nobody wrote**. Its own entry said
so — "the producer does not yet emit `q_labels` — that is the next piece and the one that decides
whether any of this measures anything". This is that piece: the same tight-MC rollout the producer
already runs, once per **legal action**, on the **same dice**.

Nothing about it is on by default. `--no-q-labels` is byte-identical — the row's key set is a frozen
census in a test, and the DICE are too: the salt now routes through `cf_q_labels.q_arm_salt` but is
verbatim the string `_rollout` has always used, because a change there would make every existing
label file incomparable with every new one.

### The pairing, and why it is asserted rather than remembered

A sweep's product is a RANKING — *is Rock Slide better than Earthquake here?* — and at the
producer's R the per-arm standard error is ~0.18, so two siblings rolled out on independent dice
differ by noise before they differ by the action. Every arm therefore draws
`cf_q_labels.q_arm_seeds`, whose salt is a function of the BATTLE and the DECISION and `--seed` and
carries **no action term**. That is a property nobody should have to remember, so
`assert_paired_dice` adjudicates at the seam, on the seed lists the arms ACTUALLY received rather
than on seeds it re-derives — a check that recomputes its own input proves only that one function is
deterministic. The regression test expresses the bug: perturb the seeds per action and the sweep
refuses.

⚠️ **The pairing is over the SIM DICE and that is not the only randomness in a rollout.** Both sides
are a stochastic snapshot at temperature 1.0 and `torch.distributions.Categorical.sample` draws from
torch's global RNG, which no seed here reaches. The policy draws are an unpaired residual: it biases
nothing (both arms draw the same policy) and it cannot be closed by seeding, because the arms
diverge immediately and stop drawing even the same NUMBER of samples. Stated rather than hidden.

### The recorded action's arm is free, and its label is an identity

The row's per-state `label` IS the recorded action's counterfactual label — same salt, same R, same
substituted choice — so at `--q-rollouts == --rollouts` it is lifted verbatim and
`q_labels[recorded] == label` **exactly**, which is both a saved arm and the cheapest possible check
that the sweep is anchored to the measurement the row's own label came from. At a DIFFERENT R it is
re-rolled instead: an anchor with more evidence than the actions it anchors turns
`q[recorded] − q[other]` into a comparison between two sample sizes.

### The selection rule is declared, because a budget knob is a distribution decision

Cost multiplies by the arm count, so `--q-max-actions` is unavoidable; WHICH arms it drops gets a
version string (`cf_q_sweep_v1`, stamped on every row) for the same reason `SAMPLER_VERSION` does.
Recorded action first, then a **deterministic decision-keyed shuffle**. Both obvious orders are
wrong here and the wrongness is the point: **descending policy probability** rebuilds the on-policy
starvation this head exists to escape (probe L measured the policy sampling its own better-ranked
alternative at a median p = 0.002, so a probability-ordered cap spends the budget where the head is
already trained), and **action index** is a systematic preference for SWITCHES, since the space is
`[switch x6, move x4, struggle]` and a prefix of it is all switches — a capped sweep would teach the
head about switching and almost nothing about attacking, on exactly the move rounds where attacking
is the question. Pinned by a 400-decision statistical test with the index-ordered rule kept beside
it as the counterfactual it fails.

### Cost is METERED, not estimated — and the throttle counts it

A swept decision costs R rollouts **per legal action** instead of R total. `--max-labels-per-hour`
therefore counts every per-action arm the sweep actually ROLLS — the reused recorded arm costs
nothing, so it does not count — which keeps the cap a COST cap instead of silently letting a sidecar
multiply the box load by its arm count. `q_rows` / `q_entries_total` / `q_arms_rolled` /
`q_arms_reused` / `q_rollouts_total` / `q_wall_seconds` ride the state file and the heartbeat, and
`(q_arms_rolled + q_arms_reused) / q_rows` **is** the measured multiplier rather than an estimate of
it. Per-decision failures land in their own `q_skip_reasons`: a lost ARM is not a lost RECORD, and
folding the two would make `records_skipped` unreadable.

### PILOT — measured, both halves

**MEASURED 2026-08-29** over the newest 90 `cf_records` of `ai_v9_72_R3SELF_0828` against **that
run's own v107 checkpoint** — config version matched exactly, and of the 37 archived runs carrying
`cf_records` it is the only one current code can still load — copied read-only to a scratch dir.
CPU, `--impl rust`, `--torch-threads 1`, `nice -n 15`, compiled extractor (**35.67 → 3.83 ms,
9.3×**), beside a live trainer at load ~27-33. Flags: `--rollouts 4 --top-n 1 --q-labels --q-top-n 1
--q-rollouts 4 --q-max-actions 0`.

| producer | |
|---|---|
| records processed / skipped | **90 / 0** (anchor 1/1, `q_skip_reasons` empty) |
| rows written / rows swept | 90 / **90** |
| per-action entries | **693** |
| **arms per row — the ~n_legal MULTIPLIER** | **7.70** (min 3, max 9) |
| arms rolled / reused free | 603 / **90** — 13% of the sweep cost nothing |
| sweep rollouts / sweep wall | 2 412 / **1 375 s** of a 1 619 s cycle |
| **throughput** | **0.504 per-action labels/sec = 1.98 s/entry** (0.438 arms/sec) |

| through the REAL `CfLabelBuffer`, same step | |
|---|---|
| ingested / resident | **90 / 90** — **0 skipped rows, 0 field skips** |
| `cf/q_label_coverage` | **1.0000** |
| `cf/q_labels_per_row` | **7.70** — the number that separates a live factory from an on-policy trickle |
| labelled (s, a) cells | **693 of 990 = 70.0%** of the `[B, 11]` matrix |
| `taken_action` / `outcome_label` / `mc_return` coverage | 1.0000 / 1.0000 / 1.0000 |
| `q_labels[recorded] == label` | **90 / 90**, exactly |

And through the **real loss kernel** (`q_winprob_terms.q_masked_binomial_nll`) on that same batch:
at a zero-init head the NLL is **0.693147 — log 2 to six places**, which is the P = 0.5 prior it
must be; fitting the labels takes it to **0.4218**; and the gradient on every UNSWEPT cell is
**exactly 0.0** while the swept cells carry a live one. That last pair is the whole safety property
of the masked form — an unlabelled action must contribute nothing, never a zero target.

Read the throughput as a per-ARM cost, not a per-label one: a swept decision costs **~7.7×** a plain
one, i.e. ~15 s of sweep per row against ~2 s for the row's own label on this box. The heartbeat's
`labels 90 (+90 total, 693/h)` is the throttle doing its job — 693 is the ARM count in the window,
not the row count, which is precisely why `--max-labels-per-hour` had to start counting them.

### Contracts

`q_labels` is a **LIST OF OBJECTS**, each naming its own action index — never parallel arrays, for
the order-mismatch reason `cf_label_buffer`'s docstring gives, and demonstrated rather than asserted
by a test that shuffles the list and reads it back identically. It rides the SAME row as the
per-state label (the buffer dedups on the obs digest, so a second row for one state would collide
and one of them would vanish), and it is **additive-optional at schema v1**: `schema` is a REFUSAL
gate, so the sweep may never bump it. An arm whose rollouts all failed is OMITTED rather than
shipped at `n_rollouts: 0`, because the consumer builds its mask from PRESENCE and a zero-evidence
entry would mask ON a cell whose target is the `0.0` fallback — a confident loss for an action
nobody measured. `taken_action` travels with `q_labels` rather than taking a flag of its own: it is
free, but a flag for it alone would offer a run the on-policy-only regime the whole stream exists to
escape.

The arithmetic lives in a new pure module, `agents/training/cf_q_labels.py` — no model, no bridge,
no record — so the pairing rule, the selection rule and the wire shape are testable without a
simulator, and the 1 649-line producer does not grow toward the file-size bound.
`obs_materializer.scan_record` gained `capture_choices=True`, which fills each decision's full legal
action → sim-choice-string map inside the ONE replay it already runs; asking afterwards would cost a
`materialize_from_record` prefix replay per labelled decision. `_ReplayObsPlayer._choice_map` is now
the single definition shared with `map_actions_at`, so the two callers — one feeding the sim, one
feeding a label — cannot disagree about what "the choice string for action a" is.

---

## `gen3_last_snapshot_resolution_v1` — a bare run directory means the run's LAST SNAPSHOT (2026-09-06)

**Owner ruling, verbatim:** *"I would either prefer us do best against target or just do the last
snapshot. I feel like best against target will always have a nuance that we need to keep track of,
whereas the last one is probably what our metrics would measure anyway."* Taken as: a bare run
directory resolves to the run's **last snapshot**, not to `best_model/best_model.zip`.

### What it replaces, and what that cost

`agents.training.fixed_opponent_pool._resolve_zip_and_config`'s first rung was
`<run>/best_model/best_model.zip` — an export selected on **BOT win rate**, an opponent set with
nothing to do with what a teacher is being distilled FOR. Ledger 2026-09-06 (ARCH→TRANSFER probe H8,
*exploiter off-slice competence*) measured the consequence: for **2 of 8** unfunded R5F teachers
(`ai_v9_94_R5F02`, `ai_v9_98_R5F06`) the exported file was a **~0.93M-step exploiter rather than the
~2.93M final** (step 26,000,016 against 28,000,032; corroborated by mtimes — those two predate their
own `final_model.zip` by 1h32m / 1h04m where every other gap is 4-5 min). So "the teacher" a fold
distilled from was neither the last snapshot nor the best against its target; `teacher_distance`'s
UNF budget covariate (3.07M) was heterogeneous (≈2.43M mean) on the very axis it had found
rank-indistinguishable from D_off; and **nothing on disk recorded which file was used**.

### The rung order, for a BARE run dir (no `@step`)

| # | rung | file | `rule` |
|---|---|---|---|
| 1 | `latest_txt` | `<run>/latest.txt` (a run-RELATIVE path; both forms it holds resolve) | `last_snapshot` |
| 2 | `highest_checkpoint` | the highest-step `checkpoints/checkpoint_<N>_steps.zip`, **incl.** the SIGUSR1 `checkpoint_forced_<N>_<HHMMSS>.zip`; legacy run-root copies too | `last_snapshot` |
| 3 | `final_model` | `final_model.zip` / `final_model_interrupted.zip` (the higher of the two) | `last_snapshot` |
| 4 | `best_model_fallback` | `best_model/best_model.zip`, then the legacy `<run>/best_model.zip` — **LAST**, only for a run with nothing else, and it says so on stderr | `best_model_fallback` |

`<run>@<step>` (`explicit_step`) and any path ending `.zip` — **including
`best_model/best_model.zip`** — (`explicit_zip`) bypass the ladder entirely and are used verbatim.
Naming the file is how you pin it.

### 🚨 The disagreement rule: the higher `num_timesteps` wins, not the earlier rung

Rungs 1-3 are three names for "the end of this run" and they disagree in **both** directions. A
COMPLETED run writes `latest.txt → final_model.zip` *after* its last periodic checkpoint, so
`latest.txt` runs AHEAD — measured on all eight R5F runs (2026-09-06): `final_model.zip`
@**28,115,184** vs the highest checkpoint @**28,067,760**, **47,424 steps apart**, with rung 1 firing
for every one of them. An INTERRUPTED run can leave `latest.txt` naming a file a later
`final_model_interrupted.zip` has since passed. Taking the earlier rung is right in the first case
and wrong in the second, so the rule is **the file that trained furthest**, with the rung order
breaking a tie and deciding when nothing declares a step at all. `num_timesteps` comes from the SB3
zip's plain-JSON `data` member (`lineage.checkpoint_num_timesteps` — no torch, no model load),
falling back to the `checkpoint_<N>_steps.zip` filename. `best_model` is NOT on that tier: it is a
different SELECTION rule, so it never competes on steps and loses to every other rung even when it
trained further.

### One choke point, and a frozen wrapper over it

`resolve_model_ref(path, step=None) -> ResolvedModel(zip_path, config_path, run_base, run_dir, rung,
rule, num_timesteps)` is the single implementation. Every consumer reaches it: `--distill-teacher`
and `--win-prob-pbrs-source` (`main/train/model_build.py`), `--stable-opponents` and `--exploiter`
(via `resolve_stable_opponents`), `--exploiter-ladder`, `--warmstart-consensus`, and
`--distill-anchor-parent`. `run_spec_test.py` holds a census that fails — naming the file and its
flags — when one of them stops, and a second that fails when a consumer names a rung's filename for
itself. `_resolve_zip_and_config` survives as a **frozen 3-tuple wrapper**: the offline probe scripts
under `designs/research_state/measurements/arch_transfer_2026-09-05/` (`content_locality_v2`,
`exploiter_competence`) import it by name and unpack three values. **They measured the OLD rule's
files, by design, and stay as records of it.**

### Provenance — a fold now records which file it loaded

`metadata.json`'s `lineage` block gains four keys on every model reference (`fork_parent`, each
`teachers` entry, `exploiter_target`): `resolved_file`, `resolved_num_timesteps`, `resolution_rung`
and `resolution_rule`. `python -m main.lineage` prints them. The startup lines state the same thing
where an operator sees it: `🧪 [DISTILL]` emits one `teacher <k>: <spec> -> <zip> @<N> steps
[rung=… rule=…]` per teacher, `🐴 [STABLE]` / `🥊 [EXPLOITER]` the same per opponent, `🧊
[WinProbPBRS]` for its frozen φ.

🚨 **Every teacher loaded BEFORE this change went through the OLD rule** (`best_model` first, then
`final_model.zip`, then `<run>/best_model.zip`) and recorded nothing about it, so a pre-change run's
teacher identity is **not recoverable from its metadata**. `main.lineage` prints `resolved file not
recorded (pre gen3_last_snapshot_resolution_v1)` rather than re-resolving under today's rule — a
current answer presented as history is worse than no answer. A reference this change DID try and
fail to resolve records `unresolved`, so "not recorded" and "resolved to nothing" stay distinct.

### Not versioned

It changes which FILE a run loads, never a weight shape — no `ARCH_SIGNATURE` bump, no
`MODEL_CONFIG_VERSION` bump, absent from `ModelVersion.check_compatible` by design, and no
checkpoint on disk becomes incompatible with it.


## v108 — `gen3_dead_flag_purge_v2` (2026-09-06): one dead flag deleted, and the census that found only one

The v75/v78/v88 cleanup-journey pattern, run again as an end-state paydown before the win-prob-only
generation. This round is notable for what it did **not** delete: the census
(`designs/research_state/flag_census_2026-09-06.md`) classified the whole flag surface against four
conditions — never ON in any gen-9+ run, absent from the production config, no `state_dict` key
depending on it, and unnamed as a lever by any live design doc — and exactly **one** flag met all
four. Every other OFF flag is an armed lever, an umbrella desugar target, or a documented next step.
An empty-looking purge is the honest outcome when the previous rounds did their job.

* **`threat_prob_outspeed`** (v36, `gen3_bidir_threat_trunk_v1` #3) — the uncertainty-aware
  P(outspeed): divide the speed gap by the believed speed STD (`SPECIES_SPREAD_PRIOR`, sigmoid ≈
  normal CDF) instead of a fixed scale, so a high-variance opponent speed reads nearer 0.5 and a
  pinned one reads sharp. Deleted: the argparse entry, the `_resolve` line, the `flag_registry`
  entry, the `ModelVersion` field, the `check_compatible` gate, the `combination_checks` rule
  (`--threat-prob-outspeed requires --damage-op`), the constructor kwarg down through
  `snapshot` → `extractor_build` → `DamageOperator`, and the branch itself in
  `DamageOperator._p_outspeed`. Measured usage: **0 of 124 gen-9+ runs**; the 60 archive runs whose
  recorded command types it are all config ≤ 46, far below `MIGRATION_FLOOR` 96, so none of them
  can be resumed into the current architecture by any route. Zero mentions under
  `designs/research_state/`. **There is no replacement flag** — the surviving behaviour is the
  fixed-scale logistic that every gen-9+ run already used, so a stale command drops the flag and
  launches unchanged.

**Migration — REFUSED on True, and the reason is the interesting one.** Every other member of
`_DEAD_FEK_JUDGED` is there because its ON value named PARAMETERS, so popping it would hand SB3 an
unplaceable `state_dict`. This one is the inverse and the more dangerous shape: it built **no**
parameters, so a `True` checkpoint and a `False` checkpoint are **byte-identical in every key**. It
only chose a divisor. Popping `True` would therefore load cleanly, pass every shape gate, and run a
checkpoint under physics it was never trained on — permanently, with nothing able to notice. So
`True` raises with the v75 re-read-from-`git_hash` diagnosis and `False` pops silently, in both
`_migrate_config` (the config JSON, version-independent sanitizer) and
`snapshot.sanitize_dead_extractor_kwargs` (the pickled `features_extractor_kwargs`).
**"Loads cleanly" is the reason to refuse it, not a reason to allow it.**

**No `ARCH_SIGNATURE` bump, and that is the safety rule rather than a convenience.** No
`state_dict` key moves, so every gen-17 checkpoint stays loadable; a bump would refuse them all for
a deletion that cannot affect them, and the floor contract would drag `MIGRATION_FLOOR` up with it
in the same commit. `MODEL_CONFIG_VERSION` 107 → 108 (the stamp), `MIGRATION_FLOOR` unchanged at 96.
`deleted_toggles_v2_test.py` pins all of it — both sanitizers, the parser refusal, the dangling
combination check, and the signature/floor non-movement.

**Knowingly LEFT BEHIND, recorded rather than smuggled in.** `_p_outspeed` still accepts
`opp_spe_std`, and ~12 call sites across `damage_op` / `damage_op_blocks` / `damage_op_pairwise`
still compute it from `SPECIES_SPREAD_PRIOR[..., _SB_SPE, 1]` and pass it. The deleted *branch* was
code that never ran; those lookups **ran every forward and were discarded**, so removing them is
behaviour-preserving but is a live-hot-path edit in the tree's most physics-critical module rather
than a provably-inert one. It is banked as the named follow-up in the census, with the seam, instead
of riding along inside a purge. The unit test asserts the std cannot reach the divisor, so a
re-introduction has to change a test rather than silently change the physics.

## THE DIAGNOSTIC EXPORT — the reward's composition, the win-prob head's calibration, and the PopArt currency (2026-09-06)

`gen3_reward_term_export_v1` + `gen3_winprob_calibration_export_v1` + `gen3_popart_currency_readout_v1`.
**Logging only** — no loss term, no parameter touched, no forward changed. The byte-identity check is
in the report rather than the intent: two identically-seeded tiny PPO models through the REAL
`train()`, one with the new folds reachable and one with them neutralised, agreed on all eight
watched training scalars (`train/policy_gradient_loss`, `value_loss`, `entropy_loss`, `loss`,
`approx_kl`, `clip_fraction`, `explained_variance`, `grad_norm`) and on **12 of 12 parameter tensors
at max|Δ| 0.000e+00**.

The occasion is the next generation making the win-probability head the critic's only signal, over
runs measured in days. Four things the export could not say before it:

**1. WHAT THE REWARD IS MADE OF.** A run STATED its composition once at startup
(`reward_class_composition` → the `metadata.json` `reward_composition` block) and then never
mentioned its MAGNITUDES again. A term could be structurally present, listed in the banner, and
identically zero for a whole generation with nothing saying so. The new `reward/` group is one triple
per ACTIVE term per rollout — `<term>_mean`, `<term>_abs_mean`, `<term>_abs_share` — plus four class
rollups and four totals, in RAW REWARD units. **The share is `|·|`-weighted and it must be**: PBRS
terms telescope to zero by construction, so a SIGNED share reports every healthy potential as inert;
`abs_share` asks instead how much of the reward stream's MOVEMENT a term accounts for, and the signed
mean ships beside it as the place where the telescoping IS the reading. Measured on the smoke: the
per-term shares summed to **1.000001** and the class shares to **1.000000**.

**`reward/untracked_abs_mean` is a GIGO guard, not a rounding term.** It is
`mean|bd.total − Σ tracked terms|`, and the tracked set comes from the same `_pbrs_term_active` /
`_bias_term_active` predicates the folds are gated on — so a non-zero reading means the startup census
and the folds disagree about what this config emits, which is exactly the shape of the v9 drift
(`--all-shaping-pbrs` silently ceasing to be passed, unobservable for a year). It read **0.000000**
on the smoke. It is PUBLISHED rather than asserted because a reward manager must never take down a run.

The transport is an **`env_method` PULL, not an info-dict thread** — the reward is computed in the env
worker, and under `--async-rollout` the callback's step locals arrive wave-batched with no way to
recover which buffer row a step landed on. That is `TeamWinRateCallback`'s reasoning verbatim, and it
is why one seam covers both collectors. `RewardTermAccumulator.drain()` zeroes the window, so a
rollout boundary that pulls twice cannot double-count. ALWAYS ON without a flag: the accumulator folds
only the ACTIVE terms (9 of 35 under the production composition, 46 tags), so an opt-out would buy
nothing but the chance to forget it.

**2. WHETHER THE HEAD'S 0.7 MEANS 0.7.** `brier` is a PROPER scoring rule and decomposes as
`reliability − resolution + uncertainty`, so a head can trade calibration for sharpness and hold its
Brier flat — which is the one failure mode that matters when the head becomes the critic. `win_prob/`
gains `ece` (10-bin, count-weighted), `mce` (the worst READABLE bin — an average hides a head that is
badly wrong only on the confident tail), the reliability histogram itself as `rel_gap_b0..b9`, `rel_n`,
and every one of those again as `contested_*` restricted to material-EVEN decisions, because a
blowout's P(win) is trivially recoverable from material and the pooled ECE is flattered by exactly the
states nobody needs the head for. **An under-populated bin publishes NaN, never a gap of 0.0** — a
three-sample bin's "error" is sampling noise, and TensorBoard renders NaN as a hole. The accumulator
folds BIN COUNTS across epoch 0's minibatches rather than averaging per-minibatch ECEs, because an ECE
is nonlinear in the bin populations and the mean of the parts is not the statistic of the whole.

**3. WHAT THE HEAD SAYS AT THE OPENING BOARD, AGAINST WHAT THOSE GAMES DID.** `win_prob/start_*` is a
PAIRED calibration at the least-informed state, where a miscalibration cannot be excused by a lost
position. `win_target` is back-filled by `WinProbLabelCallback` from each episode's outcome to every
step of that episode, so at an episode-START row it IS the realized outcome of the game that starts
there — prediction and realization come from ONE set of episodes and `start_gap` is a paired
difference, not the difference of two independently-windowed averages. With the `opp_class` obs key
present it splits by opponent class, and `start_*_pool` is then literally *"the self-play win
probability at episode start vs the realized self-play win rate"*. Cost: one EAGER forward over the
episode-start rows (capped at 1024, a deterministic prefix, never sampled — a diagnostic that moves
because of its own RNG cannot be compared across arms), once per `train()`; eager `type(fe).forward`
for the capacity probe's reason, so no dynamo graph is added for a diagnostic.

Its unconditional companion is **`signal/outcome_win_rate_<kind>`**. `p(1−p)` is SYMMETRIC about 0.5,
so `outcome_entropy_pool = 0.16` means p = 0.2 **or** p = 0.8 and for two generations nothing in the
export said which. The realized per-class win rate is one more mean over a deque that was already
there.

**4. WHETHER THE CURRENCY CONVERSION IS CURRENT.** `popart/mu` and `popart/sigma` say what the
normalizer BELIEVES; `popart/norm_return_mean` and `popart/norm_return_std` apply that belief to this
rollout's own returns and read ≈0 and ≈1 when it is tracking. The smoke read **−1.618 and 0.525** on a
warming 10k-step run — i.e. the normalizer lagging the return scale by ~2×, which is precisely the
factor by which the value gradient is then mis-scaled against the shared trunk, and a fact no
combination of the three older `popart/*` tags stated.

**TWO RENAMES WERE CONSIDERED AND REFUSED, and the reasoning is the durable part.** Dashboards read
these names, so a rename has to be paid for by the old name being *misleading*, not merely non-obvious.

* **`train/value_target_std` is `train/return_std`.** The value targets ARE
  `rollout_buffer.returns`; the existing name is accurate and sits beside its own
  `return_mean`/`return_abs_max` family.
* **`train/advantage_mean` / `advantage_std` are `signal/adv_raw_mean` / `signal/adv_raw_std`.** The
  `signal/` group is where the RAW pre-normalization advantages are read — the one place they still
  exist, since `normalize_advantage` forces std→1 per minibatch — and two names for one number is
  worse than one non-obvious name. `adv_raw_mean` is the half that was genuinely missing and is added.
* **`win_prob/vs_critic_divergence` is `train/scaffolding_gauge`**, which has a documented section, an
  offline CLI (`python -m main.scaffolding_gauge`) and a companion `scaffolding_rho`.

**AND ONE NON-ADDITION, stated so nobody re-derives it.** A PopArt-normalized
`train/explained_variance` would be a **duplicate curve, not a second measurement**: EV is
`1 − Var(y − ŷ)/Var(y)`, PopArt applies the same affine map to `y` and `ŷ`, and a shared affine map
cancels exactly — `Var(a(y−ŷ))/Var(a·y)` is invariant and the `−μ` cancels inside both variances. SB3's
default is computed on the RAW shaped-return arrays (`policy._critic_value` de-normalizes, so both
`values` and `returns` are raw), and the normalized EV is numerically identical to it. The census
records the currency instead.

The census itself — every group, its cadence, its currency, where it is computed, and the four
currencies this trainer runs at once — is now a table in `src/agents/training/CLAUDE.md` →
*TensorBoard export census*, with the recount recipe beside the counts, because this corpus moves
faster than any prose about it.

---

## `gen3_winprob_critic_mode_v1` — THE WIN-PROB CRITIC, as a MODE (config v109, 2026-09-06)

**`--critic {shaped,winprob}`. The DEFAULT IS `shaped` and a flagless run is byte-identical** —
this commit makes the mode exist, correct and tested; the default flip and the `ARCH_SIGNATURE`
bump that forces fresh weights land in a later one, after an arm has run. Design of record:
[`ai_v12/design_winprob_only_critic.md`](ai_v12/design_winprob_only_critic.md), whose §6 gap list
this implements in its stated order (A2 census first, then B2/B3, then the mode).

### What the two modes build

| | `shaped` (default) | `winprob` |
|---|---|---|
| `policy._critic_value` | `_denorm(value_net(latent_vf))`, or `_denorm(E[Z])` under `--value-from-dist` | `sigmoid(win_head logit)` in **[0,1]**, no `_denorm` at all |
| the value LOSS | `vf_coef * MSE` (or the HL-Gauss CE under Phase B), in PopArt-normalized units | the win-prob head's **BCE against the terminal outcome**, at `vf_coef` |
| noise-scale group | `value` | **`value`** — never `aux` |
| `value_net` | trained | in NO loss graph (its term is dropped, exactly as under Phase B) |
| reward stream | 1 TERMINAL + 7 PBRS + 1 BIAS; ±30 with a −35 timeout | the TERMINAL **WIN INDICATOR** alone: `+victory_value` on a win, **0.0** on a loss, a tie AND a 250-turn timeout alike |
| PopArt | on | **refused**, at the launch AND at the policy constructor |
| `gamma` | 0.9999 (now `--gamma`'s default, read from `PBRS_GAMMA` rather than retyped) | **1.0** |
| `--win-prob-coef` | weights the auxiliary BCE | **refused** — one critic, one coefficient |

At `--victory-value 1.0` the undiscounted return from any state is exactly `1{win}`, so
**`V(s) = P(win | s)` with no approximation term** — the identity the whole mode rests on, and the
reason `--terminal-indicator` and `--victory-value 1.0` are requirements rather than suggestions.
A `+V/−V` terminal under a `[0,1]` critic would put the return and the critic in different scales
and give every terminal TD error a systematic, state-dependent offset (a loss reading `−V − V`
against a truth of `0 − V`).

### The four NEW things, and one deliberate non-thing

* **`--critic`** — STRUCTURAL, recorded, string-compared in `check_compatible`. The gate matters
  more than usual: BOTH routes return a `[B,1]` float tensor, so a flipped mode produces no shape
  error anywhere and would simply predict a different quantity for the rest of the run.
* **`--terminal-indicator`** and **`--arm-no-progress-tax`** — two resume-immutable `RewardConfig`
  fields, both defaulting to today's behaviour. The second is design gap **B4**: `--no-hand-shaping`
  zeroes the WHOLE BIAS class, and this re-arms `no_progress_tax` alone without reviving the other
  24 — the contingency for the anti-stall defence a `[0,1]` critic structurally gives up.
* **`--gamma`** — γ was a hardcoded `0.9999` at `model_build.py`'s `InstrumentedMaskablePPO(...)`
  call (design gap **B6**). It is now a flag whose shaped default is `reward_weights.PBRS_GAMMA`
  itself, so the PBRS invariance premise cannot be broken by a second copy of the number. **INERT
  ON A RESUME**, like `--lr`: SB3 restores the checkpoint's own γ, and the resume path now says so
  and re-points `reward_config.gamma` at the value actually in force. The
  `PBRS_GAMMA == model.gamma` assert is GATED on a potential actually being folded, on both build
  paths — with none folded there is nothing for the invariance claim to be about.
* **`win_prob/critic_*`** — the P(win)-currency reliability read, once per rollout, from
  `agents.training.scaffolding.reliability_table` (imported, so the live number and
  `main.scaffolding_gauge --reliability` are the SAME statistic). It sits BESIDE
  `gen3_value_diagnostics_v1`'s `win_prob/ece` / `mce` / `rel_gap_b*` rather than in a parallel
  prefix: those read the head's logits per MINIBATCH, this reads the buffer's recorded `values` —
  the P(win) GAE actually bootstrapped from. **The meter is `critic_resolution`, not
  `critic_reliability`**: the committed baseline measured this head at reliability ~0.002 against a
  resolution of 0.062 out of an available 0.182, i.e. already calibrated in the MEAN and starved of
  SEPARATION, so a promotion that improves ECE and leaves resolution flat has moved the meter that
  was never the disease.
* **The non-thing:** `value_dist_head` is NOT deleted. Under `--critic winprob` it is simply not
  built (`--value-dist-mode` is refused there). The A2 consumer census that must precede any
  deletion is now §6 of the design doc.

### THE CENSUS FINDING THAT SHAPED THE REFUSALS (design gap A2)

The census enumerated every consumer of the distributional head and asked which would break if the
head were merely absent. The load-bearing answer: **~15 sites gate on the `value_dist_mode` STRING
rather than on `value_dist_head is None`** — the PPO CE gate, the grad-balance value term,
`prober.model.value_dist_support`, `session.core._dist_support`, and the PIT/coverage scan. So a
mode that skipped the build while leaving the string set would produce **numbers with nothing
behind them**, silently: the run would train with no distributional loss while every flag and
`model_config.json` claimed it was on. That is why `--critic winprob` REFUSES a non-`none`
`value_dist_mode` on the RESOLVED value rather than on a typed one — an inherited `'shaping'` from
a fork parent is exactly the case that would slip through a `_typed` check.

### IMPLIED vs REQUIRED — an asymmetry of the flag surface, not a design preference

Three flags are IMPLIED by `--critic winprob` (`--win-prob-mode shaping`, `--gamma 1.0`,
`--no-use-popart`) because their argparse default is the `None` sentinel, so "unset" is
representable and an implication can never overwrite a typed value. Four are REQUIRED, each named
by its own refusal (`--no-hand-shaping`, `--terminal-indicator`, `--victory-value 1.0`,
`--draw-penalty 0`), because theirs are concrete (`True` / `False` / `30.0` / `−35.0`) — there,
"left alone" and "typed the default" are indistinguishable, so implying would silently overwrite a
choice AND make the refusal meant to catch a conflicting one unreachable. This tree's standing
preference for a composition-changing combination is the same (`--use-popart` requires an explicit
`--clip-range-vf none`): a self-documenting config beats a silent override.

**`resolve_critic_mode` runs BEFORE the `_resolve` inheritance sweep**, and the order is
load-bearing: a fork of a `shaped` parent would otherwise inherit that parent's `use_popart=True` /
`win_prob_mode='none'` from its recorded config, breaking the mode with a value nobody typed — on
the very command shape (a fork) it is most likely to be launched as. `main.checkargs` calls the
same function, so the offline answer and the launch answer cannot drift.

### The `--win-prob-pbrs-*` family — REFUSED, NOT DELETED (owner amendment, 2026-09-06)

The design's §3.7 recommended deleting the pair outright. The owner amended it twice on the day,
and both halves are recorded because the reasoning is what generalizes:

1. **The SELF-φ path is refused for a REASON.** With `V ≡ φ`, `coef·(γφ(s′) − φ(s))` IS the TD
   residual GAE already turns into the advantage — route 1 would add the advantage to the reward
   and then take the advantage of that. Its Ng shield is also at its structurally weakest exactly
   here (the theorem assumes a FIXED φ; ours is the head being trained).
2. **The FROZEN-φ path is refused as DEFERRED, and the message says so.** Exact invariance DOES
   hold for a fixed φ — the critic then learns `P(win) − φ_frozen`, recoverable at inference by
   adding φ back — so it is held for a later ablation, not judged wrong, and
   `agents/training/winprob_pbrs.py` is left intact. The ledger's registered SPARSE / SELF-φ /
   FROZEN-φ ladder stays one flag away.
3. **It is a BOOLEAN, not a scalar.** `--win-prob-pbrs-frozen <run|zip>` is declared now, in the
   shape the win-prob critic wants: on/off by presence, no coefficient. **The derivation:** under
   this critic the terminal is the win indicator and `V = P(win) ∈ [0,1]`, so φ = σ(logit) is
   ALREADY in the value currency and the currency-matched coefficient is exactly **1.0** — set
   internally and printed at startup, never a knob. (Under a ±1 terminal with `V = 2p − 1` the same
   argument gives 2 on φ = p; the code's own currency is the indicator one, hence 1.0.)
   `--win-prob-pbrs-coef` is refused under `winprob` with *"no coefficient: the potential is
   currency-matched; the dose ladder belonged to the shaped critic."* Nothing under `--critic
   shaped` changes: the old pair keeps its meaning until the default flip.

### The versioning, precisely

`MODEL_CONFIG_VERSION` **108 → 109** (`critic`, `terminal_indicator`, `no_progress_tax_armed`; a
pre-v109 config defaults all three to today's values — not a guess, since none of the three
existed). **`ARCH_SIGNATURE` UNCHANGED at `gen3_critic_route_wave_v1`**, deliberately: with
`shaped` the default, no module is added or removed, no `state_dict` key moves and the forward is
byte-identical. The signature bump belongs to the DEFAULT FLIP, where it is forced — a critic
trained to predict a shaped return cannot be warm-started into predicting a probability.
`MIGRATION_FLOOR` is untouched for the same reason: nothing here makes an existing checkpoint
unloadable.

### Riding along: `reward_composition.py`

`reward_manager.py` sat 11 lines under the file-size gate's 2,000-line hard bound, so the
stateless, config-duck-typed COMPOSITION ANNOUNCER (`_rc`, `_pbrs_term_active`,
`_bias_term_active`, `reward_class_composition`, `reward_config_digest`,
`format_reward_composition` — 117 lines) moved to its own module and is re-exported, exactly as
`reward_weights.py` was. A natural seam rather than an arbitrary cut: `reward_manager`'s subject is
the per-decision FOLD, and this module's is the static question *"which terms can this config emit
at all?"*. The gates stay the folds' own — `_hand_pbrs_on` still delegates to `_pbrs_term_active`
and `_apply_bias_drops` / `_active_bias` still read `_bias_term_active`, which is what keeps the
census and the folds from advertising different compositions.

**Two more gaps closed in the same pass, recorded because both are correctness rather than
features (the design's own "B2 and B3 before any arm" ordering).**

* **B9 — the explicit DRAW branch.** `battle.won` is a TRI-STATE (True / False / **None**, the
  last being a draw or the 250-turn timeout) and `MaskableAgentWrapper.step` reached `0.0` for the
  third case through a boolean test — by accident. It is now a named branch: **a draw is scored as
  a NOT-WIN by decision**, and it is SCORED, never dropped. `info["win_draw"]` publishes the fact,
  `SignalMetricsCallback` counts it on the terminal scan it already runs (both the sync and the
  wave-batched `--async-rollout` path), and **`signal/draw_rate` + `signal/n_terminals`** state the
  frequency per rollout. ⚠️ It is `signal/`, not the `train/draw_rate` §3.2 proposes: that callback
  has a PINNED prefix contract, and the draw rate's literal siblings (`signal/outcome_win_rate`,
  `signal/outcome_entropy`) are computed from the same `info` in the same loop.
  ⚠️ Note what this exposes rather than creates: under the SHAPED terminal the label and the
  objective disagree about draws — the label says not-win (`y = 0`, same as a loss) while the
  reward pays `--draw-penalty` (−35, i.e. WORSE than a −30 loss). Under `--terminal-indicator`
  they agree. That disagreement is a property of the shaped composition, not of this branch.
* **B2 — the `popart is None` audit.** All three sites the design names (`aux_terms.py`'s
  `/ popart.sigma`, `cf_terms`' `mc_return` normalize and its de-normalized readback) ALREADY
  branch, and the fourth (`_value_loss_from_se`) never had a branch to lose — it takes per-sample
  squared errors already in the caller's chosen space, so PopArt is `ppo.py`'s business. What was
  missing was a TEST: those paths were the rare case, and under `--critic winprob` (which refuses
  PopArt) they become the ONLY path, permanently. They are pinned now.

---

## `gen3_tb_relevance_v1` — the TensorBoard export, re-read against the win-prob era (2026-09-06)

**Owner ask: "review each tensorboard metric that we are emitting and see if they are still
relevant."** The answer was measured, not reasoned: every classification below comes from
`models/ai_v12_01_winprob_critic`'s own tfevents — **216 tags** over its first hours — cross-read
against two matched CPU debug smokes (a `--critic winprob` arm and a `shaped` one) and against the
153 static `logger.record(` sites.

`--critic winprob` changes no tag NAME. It removes the SOURCE behind several, and the recorders kept
publishing anyway — as flat constants, tautologies and byte-identical copies that a reader cannot
tell apart from a measurement. That is the whole finding.

| class | on the arm | |
|---|---:|---|
| LIVE | 166 | meaningful as published |
| **NOISE** | **31** | content-free *here*; 18 gated in this pass, 13 pending one clause in `calibration.contested_mask` |
| REDUNDANT | 19 | exact duplicates by identical formula or affine invariance — all KEPT, all named |
| CONDITIONAL | 48 observed elsewhere + ~14 whole families | correctly SILENT; PopArt and `value_dist/` are *refused* by the mode rather than merely off |
| **DEAD** | **0** | the v75 latent-belief and v88 `V_pub` purges left no orphaned recorder |

### The rule, applied four times

**A tag whose source is absent is not emitted — and the gate is on the SOURCE, never on the value.**
A shaped run is therefore byte-identical (verified: 172 tags, empty before/after diff), while the
winprob arm drops 18:

* `train/scaffolding_{gauge,rho,n}` — under this mode `V = sigmoid(win_prob_logit)`, so the gauge
  compares a quantity with **itself**: ρ ≡ 1.0, gauge ≡ 5.5e-13, flat for 35 rollouts.
  `live_gauge_metrics` now returns `{}` when the two rank vectors are identical, which is exactly
  "one is a monotone map of the other" and is the only condition under which the number is a
  tautology rather than a reading.
* `grad/win_prob_{share,norm_shared,policy_cosine}` — the critic loss IS the win-prob BCE, and
  `ppo.py` passes the SAME tensor object as `value_term` and as `aux_terms["win_prob"]`.
  `grad_balance_metrics` skips an aux term that `is value_term`.
* `win_prob/{brier,acc}_contested` · `contested_{frac,label_mean}` · `brier_material` ·
  `skill_vs_material` — `win_margin` is a by-product of the MATERIAL potential, so a composition
  with no material PBRS term (every `winprob` arm runs `--no-hand-shaping`) leaves it identically
  0.0. `contested_frac` then reads a flat 1.0, each `*_contested` is a byte copy of its pooled twin,
  `brier_material` is a flat 0.25 and `skill_vs_material` collapses to the affine transform
  `1 − 4·brier`. `_win_prob_loss` now treats a **spread-free** margin as absent.
* `reward/{bias_refund,class_refund}_*` — the refund is the BIAS class's accumulate-and-refund
  mechanism; with no bias term it is structurally 0.0. Dropping it from the tracked set
  **strengthens** `reward/untracked_abs_mean`: a refund that somehow became non-zero now reaches the
  GIGO guard instead of a curve nobody reads.

Plus `eval/{win_rate,mean_reward,mean_ep_len}_vs_pool` and `eval/sentinel_monotonicity`, which fell
back to a confident **0.0** and a perfect **1.0** when no sentinel had been measured — which is how
the live arm's first two cycles read, beside a `pool_snapshot_count` of 1. Only the EXPORT is gated;
`_check_promotion` and the ELO fit still consume the in-process defaults.

### The one that was not cosmetic

🚨 **The `grad/` duplicate sat in the shared DENOMINATOR too, so every `grad/*_share` on a winprob
run was deflated by the critic's own pull.** The arm's published norms are policy 0.4555, value
0.1380, win_prob 0.1380 (*the same number*), hp_type 0.0723, move_latent 0.0039 — and its published
`policy_share` of 0.5639 is `0.4555 / 0.8077`, a denominator carrying 0.1380 twice. The corrected
shares are **policy 0.680 · value 0.206 · aux 0.114**. The shares summed to 1.0 the whole time,
which is precisely why nothing caught it: the invariant everyone checks was preserved by the defect.

### What was NOT changed, and why

* **`eval/mean_reward_*` ≡ `eval/win_rate_*`, all 13, byte for byte** — at `--victory-value 1.0
  --draw-penalty 0 --terminal-indicator` the episode reward IS the win indicator. Kept: they are not
  redundant on a `shaped` run and the TUI reads both. Named here so nobody quotes one number twice
  and calls it two agreeing signals.
* `reward/class_terminal_*` ≡ `reward/win_loss_*`, `reward/total_*` ≡ the same, and both
  `*_abs_share` a constant 1.0 — a class rollup and a share partition over a SINGLE term. Kept:
  suppressing single-member rollups would change the shaped run's tag set, which was the one thing
  this pass promised not to do.
* `train/nonbot_fraction` ≡ `train/selfplay_fraction` with no `--stable-opponents`. Kept: it
  separates the moment a stable opponent is added, and a curve that appears mid-run is worse than a
  duplicate.
* `rank/tripwire_no_reading`, a flat 1.0 for the whole run, is the tripwire correctly reporting its
  own blindness — suppressing it would hide a watchdog that is watching nothing.

### Still open

**13 NOISE tags remain: `win_prob/contested_{ece,mce,rel_gap_b0…b9,rel_n}`.** They pass through
`calibration.contested_mask`, which returns `None` only when the margin KEY is missing, not when the
margin has no SPREAD; `_CalibrationAccumulator.metrics()` already returns `{}` for an unobserved
accumulator, so the family disappears the moment that one predicate learns the second case. It was
left alone because `calibration.py` was being edited concurrently, not because it is harder.

The deeper fix is a different change and belongs to whoever owns the reward manager: the material
margin is a cheap by-product, and computing it **unconditionally** would give the win-prob era back a
genuinely useful contested-vs-blowout split instead of removing it. Until then the honest read is
`python -m main.scaffolding_gauge --reliability --reliability-reweight`, which stratifies bot vs pool
sentinel — where the head's bias flips sign.

The classification, and the 28-tag "what to watch on a win-prob run" dashboard it produced, live in
`src/agents/training/CLAUDE.md` → *ERA RELEVANCE* and *What to watch on a WIN-PROB run*.
## `gen3_tb_inherit_v1` — A FORK INHERITS ITS PARENT'S TENSORBOARD CURVES + the CURATED logdir (2026-09-06)

Two changes to how this programme's training curves are READ. Neither touches training: no loss, no
forward, no weight shape, nothing on `ModelVersion`, nothing in `check_compatible`.

### 1. The fork prefix (`agents/training/tb_inherit.py`, `--tb-inherit` default ON)

**Bookkeeping over two facts the tree already had.** TensorBoard merges every
`events.out.tfevents.*` inside ONE run directory into one series per tag ordered by step — measured,
not assumed: `ai_v8_03_zarch_control_0718` carries **29** event files (one per launcher restart) in a
single `tb/` and renders as one curve. And a fork's global step CONTINUES the parent's
(`reset_num_timesteps=False`) — measured on that same run, whose `tb/` opens at step **148,401,356**,
precisely the `fork_step` its `lineage` block records. The parent's curve therefore occupies
`[0, fork_step]` and the fork's `[fork_step, …]`: two halves of one line, apart only because they
live in two directories.

At fork creation — the same place the `lineage` block is written — `<fork>/tb/` now gets a
**TRUNCATED copy of the parent's SCALAR events at steps ≤ `fork_step`**. Truncation is not optional:
the parent usually trained past the fork (`ai_v8_01` reached 170.6M having been forked at 148.4M) and
its tail would draw parent-only progress inside the fork's own step range. A fork-of-a-fork composes
for free, because the parent's `tb/` already carries its own prefix.

🚨 **THE DEFECT THIS SHIPPED WITH AND THE FIX, because the class generalizes.** A TensorBoard scalar
has two on-disk spellings — the classic `simple_value` field and a rank-0 tensor tagged with the
`scalars` plugin — and **`EventFileLoader` MIGRATES the first into the second as it reads**. Every
writer in this tree emits `simple_value`, so the first version copied what that loader returned and
therefore wrote the *migrated* form; `EventAccumulator` files those under `tensors`, not `scalars`.
The result: the events were in the file, the provenance file was correct, the event count and the
sha256 were correct, and **the scalars dashboard showed nothing** — the fork's prefix read back as
124 `tensors` tags and **0** `scalars` tags, beside its own 124 `scalars`. The reader is now
`LegacyEventFileLoader` (raw, no migration), and the tests write their synthetic parent in
`simple_value` form and read back through `EventAccumulator.Scalars` — the accessor the dashboard
itself uses. The original tests passed the whole time because they wrote the tensor form and read it
back through the same migrating loader: **a test that writes and reads one normalised form cannot
see a form bug.** Verified failing on revert.

Other properties: **idempotent** on `<fork>/tb/INHERITED_FROM.json` (a launcher restart that still
names the parent as `--model` would otherwise append a second copy of every series — a saw-tooth, not
an error); **scalars only** (measured 2026-09-06: every value in `models/*/tb/` is a scalar, so the
filter drops nothing today and exists so a later histogram cannot silently multiply the cost); the
fork decision is **read out of the lineage block**, never re-derived (pinned by an AST test); and it
**never raises** — a cosmetic convenience must not kill a launch, so a failure returns a reason and
the run starts its chart at `fork_step`. Cost: a few hundred KB against a 262 MB archive.

Verified end to end on a CPU smoke: a 6,144-step parent, a fork off its `final_model.zip`, prefix
2,801 events / 124 tags at steps 0..6,144, and `EventAccumulator` reading `rollout/ep_rew_mean` as
**one series of 47 points from 256 to 12,032** where the fork's own training produced 23.

**Existing forks are NOT backfilled.** `python -m main.tb_inherit --list` censuses them (**137**
missing a prefix); `--backfill … --apply` writes them. ⚠️ **105 of the 137 name a DERIVED parent** —
every `lineage` block on disk was written by `main.lineage --backfill`, i.e. by REGEXING `--model`
out of a recorded command — and the derivation can be wrong: **`ai_v8_01_zarch_film_0717` records
`role="fresh", fork_step=0` while its own `tb/` opens at step 148,401,356**, arithmetically impossible
for a fresh run (it was built by a hand-written `tmp/fork_zarch_v8.py` the regex cannot parse). Every
row prints the flag; the backfill is dry-run by default.

### 2. The curated logdir (`main/tb_curate.py`, `designs/tb_curated_runs.json`)

`tensorboard --logdir models/` served all **217** runs carrying a `tb/`, mostly two-hour ablation
arms — the long runs were lost in every chart's legend and the origin held all 217 in memory.
TensorBoard has no server-side "show only these" option, so **the logdir IS the selection**:
`python -m main.tb_curate --apply` maintains `<repo>/tb_curated/` as one symlink per curated run
(`<run> -> models/<run>/tb`, so each appears under its own NAME rather than `<run>/tb`), unioning the
committed list with **every LIVE run** (detected from `ps`, so the current arm always shows). It
creates and removes symlinks inside that directory and nothing else — never a byte under `models/`,
and it refuses to delete anything there that is not a symlink.

The `tensorboard.service` user unit was repointed and restarted: **8 runs served, origin RSS
688 MB → 257 MB.** Editing the list needs no restart (TensorBoard rescans its logdir).
## `gen3_frozen_phi_actor_only_v1` — the FROZEN-φ rung, as ACTOR-ONLY shaping (config v110, 2026-09-06)

`--win-prob-pbrs-frozen <run|zip>` becomes BUILDABLE under `--critic winprob`. It is the FROZEN-φ
rung of the SPARSE / SELF-φ / FROZEN-φ ladder registered in `designs/ai_v12/launch_runbook.md`, and
`design_winprob_only_critic.md` §3.7's owner amendment declared it — boolean by presence, coefficient
fixed at the currency-matched 1.0 — while HOLDING it for "a later frozen-φ ablation". This is that
ablation, and designing it turned out to need one sentence about the CRITIC rather than about the
shaping.

### What was actually blocking it, and it was NOT the invariance

The amendment held the rung on the grounds that exact Ng invariance *does* hold for a fixed φ, so the
refusal should read DEFERRED rather than wrong. That was correct about the theorem and did not go far
enough about the target. Route 1 (`winprob_pbrs.py`) adds the potential to `rollout_buffer.rewards`,
so the critic's regression target becomes the SHAPED return; with Φ(terminal) = 0 and γ = 1 that
telescopes to

```
G'(s) = 1{win} − φ(s)
```

which is **NEGATIVE wherever `φ(s) > 1{win}`** — on every state of every lost game the frozen head
was optimistic about. `V(s) = σ(z) ∈ [0,1]` cannot represent a negative number at all, so the critic
would be fitted to a target outside its own range and `V ≡ P(win)` would be false by a known,
state-dependent function. That identity is the search leaf's contract (`--score` collapses to
`win_prob` on this critic), the calibration gate's contract (`win_prob/critic_resolution`, §4.3 G1)
and the reason `--vf-coef` multiplies a BCE. The amendment's own escape — *"recoverable at inference
by adding φ back"* — is true of the ARITHMETIC and false of the TRAINING: a sigmoid cannot be fitted
to a target it cannot output, so there is nothing to add back to.

### The construction: shape the ACTOR, leave the CRITIC alone

§3.4 already separates the two (*"the critic's target and the advantage estimator are separate
choices, and this design changes only the first"*). This changes only the second.

* The **CRITIC** trains on the UNSHAPED terminal indicator, exactly as before — under `--critic
  winprob` its loss is the win-prob head's BCE against `win_target`, a Monte-Carlo outcome label in
  the obs dict that never reads `rewards`. `V` stays `P(win|s)` **bit-for-bit**.
* The **POLICY's advantages** are GAE over `r + γφ(s′) − φ(s)` with the UNSHAPED `V` as baseline.

Both are STRUCTURAL rather than promised. `agents/training/frozen_phi.py` writes **only**
`rollout_buffer.advantages`; `rewards` and `returns` are restored to what the collector produced —
by ASSIGNMENT from a snapshot, because `(a+b)−b` is not `a` in float32 — so `train/value_loss`,
`train/explained_variance` and `value_scale_metrics` all read the unshaped return. The two arms share
ONE recomputed `last_values`/`dones` pair, so their difference is the shaping and nothing else.

### It is still potential-based shaping, and the guarantee is the STRONGER one

Ng, Harada & Russell (1999): for a FIXED φ, `Σ γ^t (γφ(s_{t+1}) − φ(s_t)) = γ^T φ(s_T) − φ(s_0) =
−φ(s_0)` under `φ(terminal) := 0` — the sum depends only on the endpoints, so it adds the same
constant to every policy's return from a given start state. Our φ is a checkpoint's head, loaded once
and never trained, so the hypothesis holds **exactly** — the whole reason the frozen rung gets what
route 1's live-head form does not.

Restricting the term to the ADVANTAGE is a *restriction* of that transformation. At λ = 1 the
identity is exact and per-row:

```
A′_t − A_t = Σ_k γ^k (γφ(s_{t+k+1}) − φ(s_{t+k})) = γ^{T−t} φ(s_T) − φ(s_t) = −φ(s_t)
```

so the shaped advantage is the unshaped one minus a **function of the state alone** — a
state-dependent BASELINE, the textbook zero-bias modification of a policy gradient. At λ < 1 the sum
truncates geometrically and the argument applies to each partial sum. **The actor-only form inherits
the invariance AND avoids the one thing the reward-stream form would have cost.**

`φ(terminal) := 0` does double duty: it makes the sum telescope, and it stops the potential leaking
OUTCOME information (a frozen head at a terminal state could be read as predicting the result that
state just revealed; forcing 0 makes the last transition's shaping `−φ(s_{T−1})`, a function of the
state the agent ACTED in). The conventions are `winprob_pbrs.successor_potential`'s, **IMPORTED**, so
the two shaping paths cannot drift on the one convention the theorem rests on — including the
BUFFER-BOUNDARY truncation case, which is NOT forced to 0.

### The coefficient is 1.0, derived

Under this critic the terminal is the WIN INDICATOR at `--victory-value 1.0`, so the undiscounted
return is `1{win}` and `V = P(win|s) ∈ [0,1]`. φ = σ(logit) is already in that currency, so the
matched coefficient is exactly 1.0 — fixed internally, PRINTED at startup, never a knob. (The ±1
terminal alternative gives the same answer scaled by 2, and would additionally break `φ(terminal) :=
0`, which is the correct zero for a [0,1] potential and the MIDDLE of a [−1,+1] one.)

### What it buys, and what it costs

**BUYS: dense credit from a calibrated head.** The clean-world composition is 1 TERMINAL + 0 PBRS +
0 BIAS — ~1 bit per ~40 decisions — and the SPARSE arm is the famine test of whether that is
learnable. A frozen mature φ scores every transition without changing the optimum: the accelerant
§3.3 deleted, in the one form whose invariance is exact.

**COSTS: the frozen head's biases become part of every advantage.** §4.1's committed baseline
measured this class of head at reliability ~0.002 against a resolution of 0.062 out of an available
0.182 — calibrated in the MEAN, starved of SEPARATION. A blurry potential is a WEAK potential rather
than a wrong one, so the failure mode is "does less than hoped" rather than "teaches the wrong
thing" — the invariance covers the second. But **a FROZEN-φ arm that beats SPARSE has measured THIS
HEAD's separation, not dense credit in general.**

### The seams, and why they moved

Both live in `frozen_phi.py` — `shape_after_rollout` (the `collect_rollouts` seam, the ONE point both
rollout loops pass through, so `--async-rollout` is covered by construction) and `record_metrics` —
with `ppo.py` carrying one call each. That is the `distill_anchor.py` shape, taken because `ppo.py`
sat at **1,997 lines against the size ratchet's 2,000 hard bound**: the first draft's inline comment
blocks tripped the gate, and the gate's remedy is decomposition rather than a shorter comment. The
frozen network rides `_winprob_phi_source` — the attribute `--win-prob-pbrs-source` already uses,
the two being mutually exclusive by refusal — so `phi_model` and `_excluded_save_params` need no
second name and the foreign weights are never pickled into our checkpoint.

### Diagnostics

`pbrs/frozen_phi_{coef,mean,shaping_mean,shaping_absmean,episode_dose,episode_dose_n}` and
`signal/adv_shaped_minus_unshaped_{mean,absmean}`. **Read `frozen_phi_mean` first and it must be
FLAT** — φ is a fixed function of state, so a mean wandering like a live head's means the frozen
source is not the thing being read (the runbook §4.2 check: frozen `0.403 → 0.391 → 0.391` against
live-φ's `0.680 → 0.347 → 0.216`). At λ = 1 on complete episodes
`signal/adv_shaped_minus_unshaped_mean` is exactly `−coef ×` the φ mean, so the pair audits the
terminal convention on real episodes rather than only in the unit test.

### Config, refusals, provenance

Config **v110** — one training-only PATH field, `win_prob_pbrs_frozen`, in the v105
`win_prob_pbrs_source` mould. **NO `ARCH_SIGNATURE` bump**: no module is added or removed, no
`state_dict` key moves, no forward changes; the flag edits a numpy array between collection and
`train()`. It is `_resolve`-inherited, and that is load-bearing in a way it is not for a coefficient
— the flag is **boolean by PRESENCE**, so "not typed" and "off" are the same argv and a launcher
restart re-invoking the original command would otherwise convert a FROZEN-φ arm into the SPARSE arm
mid-run under the same run name and the same TB series.

The `win_prob_pbrs_frozen_is_held` refusal is replaced by two: `..._needs_the_winprob_critic` (under
`shaped` — a ROUTING answer naming `--win-prob-pbrs-coef` / `--win-prob-pbrs-source`, not a
deferral) and `..._needs_a_head` (the potential IS the head). The SELF-φ refusals are unchanged.
`metadata.json`'s `lineage` block gains a `winprob_phi_source` reference carrying `resolved_file` /
`resolution_rung` / `resolution_rule` — recorded on a FRESH run too, which is the case a
teachers-only block would have missed, since the FROZEN-φ arm is fresh by construction.

### Verification

`frozen_phi_test.py` (20): the telescoping identity in both forms (per-episode `−coef·φ(s_0)`; the
per-ROW `−coef·φ(s_t)` at λ = 1) on REAL `MaskableDictRolloutBuffer`s rather than a hand-rolled GAE;
`returns` bit-identical with and without the flag AND a bit-identical value loss computed from them;
`rewards` restored exactly; the advantage delta matching a GAE over the pure shaping stream at
λ = 0.9; the LOUD refusal when no frozen source is attached (the SELF-φ double counting); the
coefficient's linearity; the dose meter and its omission at a zero terminal scale; and byte-identity
on the REAL PPO update through `instrumented_ppo_test._train_from_init`, with the two-replay
anti-vacuity control.

**CPU smoke, 2026-09-06** (`--debug --steps 10000 --critic winprob --win-prob-pbrs-frozen <a 6k-step
source built in the smoke's own run dir>`): exit 0, `Training complete`, and the startup lines

```
🧊 [FrozenPhi] ACTOR-ONLY potential from …/phi_source/final_model.zip @6,144 steps
   [rung=latest_txt rule=last_snapshot] on cpu (arch_signature=gen3_critic_route_wave_v1,
   config_version=110)
   coefficient 1 — CURRENCY-MATCHED, not a knob: …
   ACTOR-ONLY: γφ(s′) − φ(s) is added to the advantages ONLY. …
```

with `frozen_phi_coef 1`, `frozen_phi_mean 0.00828`, `frozen_phi_episode_dose 0.00385` over 15
complete episodes, and `adv_shaped_minus_unshaped_mean −0.00361`. The launch command for
`ai_v12_03_winprob_frozenphi` is `design_winprob_only_critic.md` §5.4, validated by `checkargs`
(19 flags, 0 unrecognized) and by `python -m main.launcher --dry-run` (role FRESH, `✓ DRY RUN`).

### Riding along — the RECORDED reward composition vs the ANNOUNCED one

Observed on the live `ai_v12_01_winprob_critic` arm: `model_config.json` reads
`all_shaping_pbrs=True`, `pbrs_material=True`, `pbrs_belief=True` — their argparse defaults,
faithfully recorded — while the startup announcer prints `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully
policy-invariant)`. Both are correct and they disagree, because `--no-hand-shaping` makes all three
unreachable without changing what any of them RECORDS. A reader who opens the config concludes
shaping was on.

Two answers, and the split matters. `metadata.json`'s `reward_composition` block gains the
ANNOUNCER'S OWN STRING verbatim (`composition_line`), the per-class `class_shares` and
`inert_reward_flags` — the launch printed its composition and nothing kept it, since a launcher
rotates the child log away. And `model_config.json` gains `inert_reward_flags` **beside** the values
it describes, never in place of them: `check_reward_config` compares each RECORDED value against the
one `RewardConfig.from_args` builds from the RESUMING argv, and that argv still says
`all_shaping_pbrs=True` — so a config recording False would FATAL every restart of the very run the
annotation exists to describe, and that run restarts every three hours.

`inert_reward_flags` is DERIVED from the folds' own `_pbrs_term_active` / `_bias_term_active`
predicates through a small flag→terms table, so it cannot drift from the census; the only
hand-maintained fact is which switch reaches which term, and a typo there fails a test.
`draw_penalty` under `--terminal-indicator` is the one MAGNITUDE-shaped entry (the term is still
emitted; its number is simply not read) and is stated as its own rule. `bias_additivity` is
deliberately absent even with an empty BIAS class — it is inert by having nothing to act on rather
than by a gate, and listing it would make this "flags that happen not to matter" instead of "flags a
gate switched off". It is written by `save_model_snapshot` and POPped by `_migrate_config`'s
version-independent sanitizers, so `to_json()` stays exactly `asdict(self)` (several tests round-trip
it straight back through `ModelVersion(**…)`) and no field enters the weight-shape record. Gate:
`reward_composition_test.py` (17).
## `production_config_2026-09-06` — PRODUCTION MOVES TO THE WIN-PROB CRITIC

`designs/production_config.json` stops mirroring gen-17 (`models/ai_v9_21_gen17_pfspoff_0820`,
config v97) and becomes the **win-prob critic era** run `ai_v12_02_winprob_critic` (v109). This is a
mirror switch, not an architecture change: no `ARCH_SIGNATURE` bump, no `MODEL_CONFIG_VERSION` bump,
no code path added or removed. What changes is which real run the derived artifacts describe.

### The judgement

Which run counts as production is not derivable — `arch_tables_test.test_production_config_matches_
newest_run` only enforces that the mirror agrees with the NEWEST run in `models/` on every shared
field, and cannot tell an experiment arm from a generation. The call was made under the owner's pivot
to the win-prob era.

### The mirror was CONSTRUCTED, not copied — and that is the entry's real content

The era's first arm, `ai_v12_01_winprob_critic`, launched with a 38-flag argv carrying only the
critic block and the hyperparameters. Its recorded config sits **44 shared fields** from gen-17, of
which only 13 are the critic; the other 31 are the whole architecture surface reverting to its OFF
defaults — `edge_bias_families` `off`, `entity_topk_seats` 0, `entity_tail_seats` false,
`history_events` false, `opp_belief_slots` false, `opp_intent` false, all four pointer cells off,
the species/item/spread beliefs off, both value routes off, `op_drop_renders` false. Mirroring it
would have redefined the production architecture **by omission**, and every derived artifact would
have said so with a straight face: the delivery graph loses 60 of 120 nodes and 977 of 1103 edges,
`pi_projection` narrows 1177 → 409, the token sequence drops 29 → 17.

That arm was killed and relaunched with gen-17's full surface plus the critic block. The mirror is
therefore built the other way round: **gen-17's config migrated v97 → v109, with an explicit
13-key critic override list**, and then verified. Non-critic keys changed: **zero**.

| key | gen-17 | production | why |
|---|---|---|---|
| `critic` | `shaped` | `winprob` | the mode |
| `use_popart` | true | false | IMPLIED |
| `win_prob_coef` | 0.05 | 1.0 | the flag is REFUSED; 1.0 is its `_resolve` default |
| `value_dist_mode` / `_bins` / `_vmin` / `_vmax` | `shaping` / 51 / −12 / 12 | `none` / 0 / 0 / 0 | REFUSED |
| `value_from_dist` | true | false | REFUSED |
| `value_tail_weight` | 0.3 | 0.0 | REFUSED |
| `hand_shaping` | true | false | REQUIRED |
| `terminal_indicator` | false | true | REQUIRED |
| `victory_value` | 30.0 | 1.0 | REQUIRED |
| `draw_penalty` | −35.0 | 0.0 | REQUIRED |

`win_prob_mode` is `shaping` in both. `gamma` is not a `model_config.json` key at all — the mode
implies 1.0 and the value in force lives in `metadata.json`'s `cli_args`.

**`value_entity_pool` / `value_entity_pool_full` / `value_threat_inject` SURVIVE the swap** and stay
true: they inject additively into `value_pooled`, which is exactly what the win head reads.

### What moved downstream

The delivery graph loses exactly two nodes — `value_dist_head` and `loss.value_dist_hl_gauss` — and
the 13 edges into them (1103 → 1090). Every meta field except `config_version` is unchanged:
`n_tokens` 29, `extra_seats` 16, `op_out_dim` 138, `pi_projection_in` 1177, `vf_projection_in` 128,
all 17 edge families. §6's generated flag table moves the four `value_dist_*` rows to OFF,
`value_dist_coef` to `INERT — no value_dist_head`, `value_tail_weight` to OFF and `win_prob_coef` to
1.0, and gains the v109 schema rows (the `cf_*` family, `q_winprob_*`, `policy_grad_coef`,
`win_prob_pbrs_coef`) — all OFF or INERT.

### Two latent defects the switch exposed, both fixed here

* **`arch_tables._COEF_MODULE` refused two keys.** `policy_grad_coef` (v102) and
  `win_prob_pbrs_coef` (v108) had been recorded for generations but never appeared in a generated
  table, because `production_config.json` sat at v97 and the generator only sees keys the config
  carries. The generator REFUSED rather than guessing, which is the designed behaviour; both are
  declared now (`policy_grad_coef` as a core train-loop term like `vf_coef`, `win_prob_pbrs_coef`
  gated by `win_head`).
* **`delivery_graph.module_coverage`'s deleted-vs-gated-off discriminator was INERT.** It asked
  `hasattr(features_extractor, <graph token>)`, but the tokens are graph NODE IDS
  (`"value_dist_head"`, `"alpha_head"`, …), not class names, so it resolved for **no entry at all**
  and only stayed quiet while the production config built every declared module. It is now
  `buildable_child_names()` — an AST read of every `self.<name>` `extractor_build.__init__` assigns,
  which is literally the fact being asked. The failure it would have caused is the dangerous
  direction: reporting live, flag-gated-off modules as STALE declarations to be deleted.

### The argv finding

**gen-17's recorded `original_command` no longer launches on HEAD**, for a reason unrelated to the
critic: `--distill-team-bias 0.4` with no `--distill-teacher` is refused by
`main.train.combination_checks` (migrated 2026-09-06). It is not a `model_config.json` key, so it
cannot move the mirror — but a relaunch argv built by copying gen-17's command is rejected until it
is dropped. Separately, `--intent-label-bot-weight` has a `_resolve` default of **1.0** while gen-17
ran **0.25**, so a hand-rebuilt argv must pass it explicitly or silently diverge from this mirror.

### Verification

`ai_v12_02` had not written a `model_config.json` yet, so the mirror was verified against the
**relaunch argv** — gen-17's command minus the launcher-owned and refused flags plus the critic
block, parsed by the live `build_parser()` and resolved through the launch path's own
`resolve_critic_mode` → `desugar_umbrella_flags` → `resolve_config`. **All 105 argv-settable fields
match**; the other 23 are layout-derived and no argv can move them. Provenance, the full override
table and the re-verification instruction live in the new sibling
`designs/production_config.README.md` (JSON carries no comments, and `--sync-config` writes the file
verbatim, so the record cannot live inside it).

## `gen3_baselines_registry_v1` — THE NAMED BASELINES BECOME FIRST-CLASS OBJECTS

`designs/baselines.json` + `src/agents/training/baselines.py` + `python -m main.baselines`. No
architecture change, no `ARCH_SIGNATURE` bump, no `MODEL_CONFIG_VERSION` bump, no training code
touched: this is a **provenance** change, and what it changes is how a baseline is NAMED and
RESOLVED.

### The failure it closes

Owner, 2026-09-06: *"How can we prevent this issue in the future? I feel like this happens
frequently, like we lose what the stable baseline is?"*

The root cause is that **the baselines were not objects**. "Production" was a hand-copied
`designs/production_config.json` that nothing consumes at launch. The untaught meter's fixed
opponent was a string literal in `agents/training/untaught_meter.py`. The famine comparator and its
38-Elo floor were a sentence in one ledger entry. The curated TensorBoard set was decided by asking.
And the drift gate that nominally guarded the mirror,
`arch_tables_test.test_production_config_matches_newest_run`, compared it against whichever run
directory in `models/` had the newest **mtime** — a heuristic that cannot tell a production
generation from a two-hour ablation arm.

Two incidents the same day made the cost concrete. A win-prob arm was launched from a **design-doc
command block** carrying only the critic block and the hyperparameters, so 31 architecture fields
silently reverted to their OFF defaults (81 keys differed from the production surface; mirroring
that arm would have dropped 60 of 120 delivery-graph nodes and every derived artifact would have
agreed). Separately, the **fold parent** was nearly written into `production_config.json` as if it
were the production run.

### What a baseline now is

One JSON entry per NAME: the run, an **EXPLICIT** checkpoint (a `.zip`, a `.json`, or an `@step` —
never a bare run directory, so `gen3_last_snapshot_resolution_v1`'s last-snapshot rule cannot move
it), the commit, the config version, the arch signature, the file's sha256, `num_timesteps`, a
one-sentence purpose, `set_on`, and **`set_by`: the ledger entry that set it**. Seeded with
`production`, `v9_long_baseline`, `v9_fold_parent`, `v8_line`, `v8_parent`, `famine_comparator`,
`untaught_meter_opponent`, `untaught_meter_config`, and the `tb_curated` LIST.

`famine_comparator` carries `floor_elo: 38.0` and a `notes` field stating why it is 38 and NOT the
adjacent-node spread (172/186, dominated by the 2M→4M jump — steep early LEARNING, not noise) — the
bar and the run it is a bar against are one fact, and `validate()` fails if the number and its prose
drift apart. `v8_line` / `v8_parent` are `era_checkout_only`: they do not load under current code, so
a missing file there is a warning rather than an error.

### The `production` entry declares a CONSTRUCTION, not a copy

Since `production_config_2026-09-06` the mirror is gen-17's SURFACE **migrated v97 → v109 with a
13-key critic override block**. The entry therefore records the surface run, `config_mirror_version`
109, and all 13 overrides by value — and `compare_production` checks three separate claims: the
shared keys outside the override block are EQUAL (the half the mis-launched arm destroyed), a
key-set delta is legitimate **only** under the declared migration, and every override is present at
its declared value AND actually differs from the run (a STALE override is reported, because it
exempts a key from the only check that guards it).

**`test_production_config_matches_newest_run` is REPLACED by
`test_production_config_matches_the_registry`**, and the new test's docstring records why: which run
is production is a JUDGEMENT, so it is DECLARED rather than inferred. The signature-bump window is
still detected — it moved to its own test, keyed on the registry's recorded signature rather than on
the newest run's.

### Consumers read BY NAME, and say which run they meant

`main.untaught_meter`'s `--opponent` / `--config` defaults (the literals deleted; the engine exposes
`default_opponent()` / `default_config()` and `resolve_ref` expands a name, so `--baseline
v9_fold_parent` works too) · `main.critic_gate --parent` · `main.elo`'s positional run dir ·
`main.tb_curate`, which unions the registry's `tb_curated` list into the curated logdir as a third
source beside the committed list and the live runs. Every one prints
`baseline <name> = <run>@<step> (set <date>, <ledger title>)`.

**`main.critic_gate` also gains the FAMINE PRE-TEST** (`--famine-comparator`, default the registry
name; `--famine-floor-elo` to override). It reuses `ladder_section` — matched SNAPSHOT COUNT, never
matched step — and reports the trail against the floor, plus two things in print rather than in a
reader's head: that it computes the **LADDER half only** (the AND-gate's other half is
`win_rate_vs_bots`), and the pre-registered confound (the incumbent had PBRS *and* PopArt *and* the
shaped critic, so a trail inside the floor is **not** evidence of equivalence — only that starvation
has not been demonstrated). A comparator with no `floor_elo` and no explicit override REFUSES rather
than inventing a bar.

🚨 **THE DEFAULT YIELDS; AN EXPLICIT FLAG REFUSES** — the same asymmetry `--compile-trainer` settles
the same way, and it is the one clause the first cut of this got wrong. The default comparator is a
registry NAME resolving under `models/`, and `models/` is not committed: on a fresh clone, in CI, or
on any box that does not carry rev-1, a default that refused took the WHOLE read down over one
endpoint of five nobody had asked for. Measured before the fix: **24 of 37 `critic_gate_test` tests
failed under `GEN3AI_MODELS_DIR=/nonexistent`**, on a synthetic tmp tree that needs no archive at
all — the tool passed here only because this box happens to carry rev-1. So an unresolvable DEFAULT
is recorded as **NOT READ** (naming the run and the reason, never as `off`, which means somebody
CHOSE to skip it) and every other section still runs; an unresolvable comparator the caller NAMED is
still a refusal, because they asked for it. `--check` reports it as an `ok` line, not a problem —
the question `--check` answers is "would this read run?", and it would.

### Changing one is a PROCEDURE

`python -m main.baselines set <name> <run>/<file>.zip --reason "<ledger entry title>"` re-resolves
the target through the run-spec choke point, recomputes the sha256, re-reads the commit / version /
signature from the run itself, rewrites exactly one entry, and **prints the ledger line to append**.
It never edits the ledger — append-only, and WHY is the one field no tool can author. A bare run
directory is REFUSED with the reason. `check` validates everything and exits non-zero on any drift.

### Retention: a named baseline survives every tier

`archive_grooming_tiers.py` reads `baselines.protected_files()` twice — a registry-named run is tier
1 REFERENCED, and its named FILES are added to the keep-list inside `assert_safe_tiered()`, the ONE
choke point every tiered plan passes through. **MEASURED: all five named runs are already tier 1 by
the committed-file scan, so the run-level half is belt-and-braces and the FILE-level half is what is
load-bearing** — `untaught_meter_opponent` is `snapshots/snapshot_000024000000.zip`, a pool file no
checkpoint rule covers and that the snapshots rule keeps only while some fork or script happens to
name the run. A broken registry degrades to no protection rather than taking a dry run down.

### Gates

`src/main/baselines_test.py` (29, unmarked, 0.6 s): the committed registry validates end to end;
every checkpoint is explicit; the seeded name set; the floor/prose agreement; the error messages
naming the registry path and every available name; `describe()` working with **no archive at all**;
and every clause of `compare_production` as a pure function (surface drift, declared override, stale
override, missing override, wrong value, the migration key-delta both ways, `config_version` never
compared). Archive-backed: every file exists and its sha matches, every checkpoint resolves at an
EXPLICIT rung, `protected_files()` names every entry's file, and the committed mirror matches the
declared construction. Plus `archive_grooming_tiered_test.py` → *the BASELINE REGISTRY* (5),
`arch_tables_test.py`'s replaced drift gate, and the freshness gate over the new prose.

`src/main/critic_gate_test.py` → *(2b) the FAMINE pre-test* (7, unmarked): `off` and NOT READ render
differently; the trail is the NEGATED ladder delta (the sign is the whole point) and carries its
`half_computed` + `confound` strings; a trail past the floor is flagged; a comparator with no
recorded floor refuses naming all three ways out; the floor travels with its registry entry; and the
two halves of the default/explicit asymmetry — **an absent DEFAULT is NOT READ with every other
section still computed** (the portability property, verified failing on revert with the pre-fix
refusal) while an explicit one still refuses.


---

## `gen3_arch_surface_guard_v1` — "IS THIS THE ARCHITECTURE YOU MEANT?", asked at launch (config v111, 2026-09-06)

**The owner's question, verbatim, after the incident: "How can we prevent this issue in the future?
I feel like this happens frequently, like we lose what the stable baseline is?"**

### The incident this answers

`ai_v12_01_winprob_critic` was launched from a design document's 38-token command block — the critic
flags and the PPO knobs, and **no architecture surface**. Every architecture flag silently took its
OFF default. **31 keys of its `model_config.json` differ from `designs/production_config.json`**:
every edge family off (`edge_bias_families` `off` against production's 17-family string), zero
entity seats, `opp_belief_slots` / `opp_intent` / `history_events` / `item_belief` / `spread_belief`
/ `pair_outcome_cell` / `intent_threshold` all False, `move_belief_mode` `revealed`,
`intent_label_bot_weight` 1.0 against production's 0.25. It trained **24.4M steps over 25,131 s**
and was still holding the GPU when it was discovered a second time. Every number taken off it
measures a different model. Ledger: `2026-09-06 · INCIDENT`.

**Three gates ran before that launch and all three passed** — `python -m main.checkargs` exit 0,
`resolve_config` accepted, `python -m main.launcher --dry-run` clean. All three were RIGHT.

> **"it launches" and "it is the experiment" are INDEPENDENT checks, and only the RESOLVED-CONFIG
> DIFF tests the second.**

That sentence is the whole change. The gap was never a missing check; it was a missing QUESTION.
`arch_tables_test`'s drift gate would have gone red — but only for whoever next ran the suite, which
happened hours after the GPU started, and a launch-time answer is the only one that arrives before
the GPU-hours do.

### The guard

`main.train.arch_surface.report()` — **ONE function, four readers**: `python -m main.checkargs`,
`python -m main.launcher --dry-run`, the launcher's own `_prepare_session`, and the child's
`resolve_config`. Three copies of a guard is three things to keep in step, and this tree has paid
for that shape twice (the pre-`combination_checks` `parser.error` lines; the pre-registry
`ARCH_ARG_KEYS`).

**The key set is DERIVED, never hand-listed.** `flag_registry.arch_surface_flags()` = the
`structural` × `family=ARCH` rows — the toggles whose mismatch means a DIFFERENT NETWORK. A literal
list would have gone stale the first time a toggle landed and the guard would then silently
under-report; `arch_surface_test::test_no_hand_written_key_list_exists_in_the_module` is the
standing guard against re-introducing one. Everything excluded is excluded **by its own
declaration**, in the row that declares everything else about it:

| excluded | by | why |
|---|---|---|
| `family=CRITIC` (7 structural + 2 `resume_immutable`) | a NEW `Family` enum on `ModelFlag` | the readouts an experiment deliberately VARIES. `--critic winprob` IMPLIES `win_prob_mode` and REFUSES `--value-dist-mode` / `--value-from-dist`, so a guard demanding these match production would refuse **every critic arm** — the exact class of arm the incident was |
| `Klass.RUNTIME` / `Klass.TRAINING_COEF` | `Klass`, unchanged | not architecture |
| `Klass.RESUME_IMMUTABLE` | `Klass`, unchanged | the forward is identical (`belief_grad_mode`, the value-dist bounds) |

🚨 **THE COMPARED-KEY COUNT IS RECONCILED, NOT MERELY SMALLER.** A guard that compares fewer keys
than a reader's own count leaves them unable to tell an excluded row from a forgotten one — so the
difference is ARITHMETIC and every printed block carries it:
`39 arch + 7 critic + 3 non-structural = 49 registry rows` (`arch_surface.surface_partition()`,
measured 2026-09-06). A row that lands without a `family` decision breaks that identity rather than
quietly moving the surface. The hand-rolled check that validated the corrected relaunch compared all
49 and found 0 differing; this guard compares the 39 it declares and finds 0 on the same config.
Both are right, and the block says which question it asked.

### FRESH REFUSES; PINNED is ADVISORY; a FORK is INFO

* **FRESH** (no `--model`), un-pinned, drifting ⇒ **REFUSED** — exit 1 from `checkargs`,
  `FATAL_CONFIG` from `--dry-run` and from `_prepare_session`, naming every differing key with both
  values. `_prepare_session` asks at the LAST point before anything exists: immediately before
  `_create_run_worktree` on the pinned path and before the `makedirs` on `--no-pin`, so a refusal
  leaves no worktree, no run dir and no child.
* **PINNED to another commit ⇒ ADVISORY**, for `gen3_pinned_argv_parser_v1`'s exact reason: the
  mirror is THIS tree's, that commit has its own registry and its own `production_config.json`, and
  `--arch` does not exist before 2026-09-06 — so refusing it would be the same false POSITIVE that
  rule already fixed for the parser. **The diff is still computed and printed**, never dropped;
  that is the other half of the same lesson, and dropping it is how the guard would silently stop
  working the day a batch pins.
* **A FORK or RESTART is INFO only** — it INHERITS its parent's surface through
  `config.inherit_saved_flag`, so its silence is the parent's architecture, not a bare one.
* **`resolve_config` REPORTS and RECORDS but does not refuse**, and that is a placement decision:
  it runs in the CHILD, after the pin is resolved, the worktree created and the run dir made, so a
  refusal there is both late and a duplicate of `_prepare_session`'s.

🚨 **THIS IS NOT THE SAME FAILURE AS A REFUSED FLAG COMBINATION, AND THE TWO NEVER SHARE A MESSAGE,
A SUMMARY LINE OR A REFUSAL PATH.** Rebuilding the same arm from an older generation's recorded
`original_command` also fails — on nine flags `--critic winprob` SUBSUMES (`--use-popart`,
`--value-from-dist`, the four `--value-dist-*`, `--value-dist-coef`, `--win-prob-coef`,
`--value-tail-weight`; `checkargs` reported 5 refused combinations until they were stripped). That
failure is **LOUD and PRE-launch**: nothing starts, the operator fixes it in a minute. Arch drift is
**SILENT and POST-launch**: everything parses, the run starts, and seven GPU-hours later the config
diff is the only thing that would have told you. A guard that catches the first is no protection
against the second, so `checkargs` prints a separate closing verdict — *"✗ this command LAUNCHES —
and builds the wrong architecture"* — that fires only when no combination also failed.

### `--arch production` — so a document never carries the surface again

Applies every ARCH-surface key from `designs/production_config.json` as if typed, inside
`desugar_umbrella_flags` and **FIRST**, before every other desugar (`--unified-moves` and
`--damage-matrices` both only fill what is still unset, so running the umbrella first makes it a
DEFAULT that the sugar and every explicit flag alike still override — precedence in one direction,
top to bottom, no special cases). Refused on a resume (`combination_checks`'
`arch_umbrella_is_fresh_only`, read by the launch path and `checkargs` from one declaration): a fork
INHERITS its parent's surface, and writing production's values over that is a `check_compatible`
FATAL at best and a silently different network at worst.

**What it deliberately does NOT set is NAMED on every run**, because a validator whose silence reads
as coverage is the same failure one layer down: the CRITIC readouts (above), `--belief-grad-mode`,
and the **SUPERVISION DOSES** — `--move-belief-coef` · `--move-belief-latent-coef` ·
`--spread-belief-coef` · `--item-belief-coef` · `--hp-type-belief-coef` ·
`--intent-label-bot-weight`, all of which production trains at 0.05 (0.25 for the last) and a fresh
run defaults to 0.0 or 1.0. Those six are declared by a new `ModelFlag.coef_arg` on the toggle they
supervise, so the list is derived like everything else. **Measured completeness**: of the 31 mirror
keys the incident's own recorded config differs on, **26 are refused on the ARCH surface, 4 are
named as doses, and the 1 remainder (`opp_belief_aux_coef`) is the enable coefficient of
`opp_belief_slots`, which is itself refused** — not one can pass unmentioned
(`test_every_key_the_incident_lost_is_either_REFUSED_or_NAMED`).

### `arch_source` — config v111, provenance only

One string in `model_config.json`, the `win_prob_pbrs_source` class exactly: recorded, never gated,
absent from `_WEIGHT_FIELDS` and from every `check_*`, **no `ARCH_SIGNATURE` bump** (no module, no
`state_dict` key, no forward). `--arch production` stamps `production_config@<12 hex of the mirror's
git blob hash>` — a CONTENT hash computed without git, so it names what the mirror SAID rather than
when it was last touched; `--allow-nonproduction-arch` stamps the deliberate-drift form; a run that
used neither records `None`. The migration defaults it to `None` and **never infers it** — a run
whose surface happens to match today's mirror still did not SAY so, and a derived answer presented
as a record is worse than no answer. It exists because the incident's run recorded a bare
architecture with nothing on disk saying whether that was a decision or an accident.

### One reader for the mirror

`arch_surface` resolves `designs/production_config.json` through
`agents.training.baselines.production_config_path()` — the `production` entry's own declared
`config_mirror` (`gen3_baselines_registry_v1`) — rather than opening the path itself, and
`_validate_production_mirror` now reads it the same way. A second opinion about what "production"
is is exactly what the registry exists to remove.

### Gates

`src/main/train/arch_surface_test.py` (39, unmarked, 1.2 s), whose central test is the REAL recorded
argv read off `models/ai_v12_01_winprob_critic/metadata.json` (with the verbatim string as the
fallback, so it means the same thing on a box with no archive): it must be REFUSED and must name at
least twenty keys. Around that: the umbrella closes it and does not fight `--critic winprob`; an
explicit `--entity-topk-seats 0` beats the umbrella and is then refused (a default, not a lock); the
consent flag passes and is recorded; a fork is not gated but its diff is still printed; the umbrella
is refused on a resume; **the pinned-vs-fresh asymmetry on the surface an operator runs** (pinned
prints `ADVISORY` and never `✗ REFUSED`, fresh refuses); **the arch verdict never sharing a line
with a combination refusal**, with no subsumed critic flag appearing as arch drift; the surface
partition being exhaustive and printed; the guard reading a RECORDED `model_config.json` as well as
an argv (a derived row is read from `opp_intent_coef` OR the recorded `opp_intent` — reading only
the coefficient reported `opp_intent False` against the LIVE run, whose recorded value is `True`);
the corrected relaunch `ai_v12_02_winprob_critic` reading **0 of 39**; and `checkargs` and
`--dry-run` reaching the same verdict, which is the split this whole class of failure lives in.

`src/main/launch_runbook_test.py` gains **`test_the_runbook_ARCH_block_IS_the_production_surface`**
(parametrized over all five arms): the runbook's hand-pasted `$ARCH` block reports **0 of 39
ARCH-surface keys differing**, so "and it is the production surface" is a measurement rather than a
claim, and an edit to it fails with the document named.

### Docs

`designs/ai_v12/design_winprob_only_critic.md` §5.4 opens with **"A COMMAND BLOCK IN THIS DOCUMENT
IS NOT A LAUNCH COMMAND"** and the incident's numbers; its corrected command carries
`--arch production` plus the six doses. `designs/ai_v12/launch_runbook.md` gains §2.6 (why `$ARCH`
exists, the one flag that replaces it, the three classes it does not set, and the pinned-vs-fresh
asymmetry), a rewritten PRE-FLIGHT step 1, and the ARCH-SURFACE banner at the top of §4.1. Root
`CLAUDE.md` → *Will this command still launch?* and `src/main/launcher/CLAUDE.md` carry the same
truth for their own surface.
