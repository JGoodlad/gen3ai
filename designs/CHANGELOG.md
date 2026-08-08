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

