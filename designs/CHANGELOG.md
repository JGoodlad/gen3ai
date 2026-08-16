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
