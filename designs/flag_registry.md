# Flag Registry — the model-relevant toggles and where each one lives

**Generated.** The table below is emitted from `src/agents/model/flag_registry.py`; edit the
registry, not this file, then run `python -m agents.model.flag_registry`
(`--check` is the staleness gate, and `flag_registry_test.py` runs it).

## Why a registry

A toggle that changes the feature extractor has to be spelled out in **five** independent places:

| # | surface | what it buys |
|---|---|---|
| 1 | the `argparse` entry in `main.train_rl_agent` | a human can SET it |
| 2 | the `_resolve("name", default)` line beside it | a **flagless** resume INHERITS it |
| 3 | `extractor_arch.ARCH_ARG_KEYS` / `_DERIVED` / `FROZEN_ARCH_KWARGS` | it reaches the extractor |
| 4 | the `snapshot.current_model_version()` keyword | an eval/self-play WORKER rebuilds the same gate |
| 5 | the `ModelVersion` dataclass field | it is RECORDED and version-GATED |

Nothing enforced the five agreeing, and every historical failure in this class had the same shape:
a toggle that reaches the extractor but not the recorded config (so a resume version-checks against
an architecture it does not build), or one with an argparse entry but no `_resolve` line (so a
flagless resume silently reverts it to OFF).

Surfaces 3 and 4 are now **generated** from the registry, so they cannot drift. Surfaces 1, 2 and 5
are **validated** against it by `flag_registry_test.py`, which fails with a message naming the
missing site.

## The three roles, and why a flag can lose its CLI entry

A flag plays three independent roles — **SELECT** (choose it at launch), **RECORD** (write it into
`model_config.json`), **GATE** (refuse a mismatched resume). Only SELECT needs a CLI entry; RECORD
and GATE live in `ModelVersion` and are reached whether or not argparse ever heard of the toggle.
So a *settled* toggle can be demoted without losing any explicitness:

| tier | argparse | `_resolve` | recorded + gated | reachable for an experiment |
|---|---|---|---|---|
| `cli` | yes | yes | yes | via the flag |
| `config_only` | **no** | **no** | yes | via the extractor **constructor kwarg** |
| `constructor_only` | no | no | no | via the constructor only |

`config_only` is frozen at the registry's `default` for every CLI-launched run — that value is the
only one the CLI can now produce, so it must be the value production actually wants.
`constructor_only` is the deepest tier; `pair_reduce`'s `reduce_how` is the precedent.

## The four classes, and which gate each one picks

| class | a mismatch means | gate |
|---|---|---|
| `structural` | weights and/or the trained forward differ | `check_compatible` — runs on **every** load, frozen eval/pool/distill opponents included |
| `resume_immutable` | the forward is bit-identical; only TRAINING differs | a dedicated `check_*` on the **resume path only** — gating a frozen opponent on it would be a false rejection that breaks league play |
| `training_coef` | a loss weight moved | none; recorded for provenance, a resume may change it |
| `runtime` | a perf knob moved | none; never recorded, never inherited on resume |

Getting this wrong is not cosmetic in either direction: a `structural` toggle with no
`check_compatible` compare lets a resume silently flip the architecture, and a `resume_immutable`
toggle *inside* `check_compatible` makes a run FATAL on loading its own pool snapshots. Both
directions are asserted by `flag_registry_test.py`.

## Dependencies

A flag's PREREQUISITES are registry data too — the `requires` column below. They used to exist only
as hand-written `raise ValueError` lines inside `Gen3FeaturesExtractor.__init__`, invisible to
everything else, so no tool could answer "would this command launch?" or "what is the minimum config
that turns X on?". `flag_requires_test.py` holds the constructor and the registry to each other in
BOTH directions: every declared dependency must actually make the constructor raise (with a positive
control that the declared closure BUILDS, so an incomplete declaration fails too), and every
constructor raise coupling two registry flags must be declared here or listed as bespoke with a
reason. `python -m main.checkargs` reads the same graph to flag an unsatisfiable recorded command
offline.

## The registry

<!-- BEGIN GENERATED: registry-table -->
46 toggles — 44 `cli`, 2 `config_only`, 0 `constructor_only`.

| toggle | CLI | tier | class | family | default | since | requires | meaning |
|---|---|---|---|---|---|---|---|---|
| `attend_unrevealed_opponents` | — | `config_only` | `structural` | `arch` | `True` | v8 | — | keep the opponent's still-hidden party attendable instead of key-masking it |
| `opp_belief_cls_k` | `--opp-belief-cls-k` | `cli` | `structural` | `arch` | `0` | v9 | `attend_unrevealed_opponents` | k learned query tokens summarising the unrevealed opp party into both heads |
| `opp_belief_slots` | `--opp-belief-aux-coef` *(coef)* | `cli` | `structural` | `arch` | `False` | v16 | `attend_unrevealed_opponents` | learned unknown-mon tokens in the un-revealed opp slots + the BeliefHead |
| `move_belief_mode` | `--move-belief-mode` | `cli` | `structural` | `arch` | `'off'` | v17 | `attend_unrevealed_opponents` | predict + reinject each opp mon's moveset (off|revealed|unrevealed|both) |
| `damage_op` | `--damage-op` | `cli` | `structural` | `arch` | `False` | v19 | `move_belief_mode` | build the differentiable GPU DamageOperator |
| `move_prior_fusion` | `--move-prior-fusion` | `cli` | `structural` | `arch` | `False` | v20 | `move_belief_mode` | fuse the Smogon move-frequency prior into the move belief as a log-odds delta |
| `win_prob_mode` | `--win-prob-mode` | `cli` | `structural` | `critic` | `'none'` | v22 | — | auxiliary win-probability side head off value_pooled (none|read_only|shaping) |
| `damage_outgoing` | `--damage-outgoing` | `cli` | `structural` | `arch` | `False` | v23 | `damage_op` | the op's OUTGOING per-move direction (our active's moves -> the opp active) |
| `move_candidate_floor` | `--move-candidate-floor` | `cli` | `structural` | `arch` | `0.02` | v23 | — | the LEGAL-BUT-UNOBSERVED base probability of the move prior |
| `move_latent` | `--move-latent` | `cli` | `structural` | `arch` | `False` | v24 | — | the context-free MoveLatentEncoder concatenated into the move network |
| `spread_belief` | `--spread-belief` | `cli` | `structural` | `arch` | `False` | v25 | — | predict + reinject the opponent's hidden spread (5 derived stats per slot) |
| `damage_topk_k` | `--damage-topk` | `cli` | `structural` | `arch` | `0` | v30 | `damage_op`, `move_latent`, `damage_matrices_incoming` | K = how many of the opp active's believed moves the incoming matrix surfaces |
| `damage_matrices_outgoing` | `--damage-matrices` | `cli` | `structural` | `arch` | `False` | v34 | `damage_op` | our active's 4 moves x the opp's 6 mons, per-(move, mon) rolls |
| `damage_matrices_incoming` | `--damage-matrices` | `cli` | `structural` | `arch` | `False` | v35 | `damage_op`, `move_latent` | the enriched top-K incoming matrix (per opp move x per our mon) |
| `spread_belief_nature` | `--spread-belief-nature` | `cli` | `structural` | `arch` | `False` | v40 | `spread_belief` | swap SpreadBelief's additive head for the NATURE/EV generative head |
| `belief_grad_mode` | `--belief-grad-mode` | `cli` | `resume_immutable` | `arch` | `'shaping'` | v41 | — | which gradient arrow between the belief heads and the trunk is cut |
| `damage_candidate_k` | `--damage-candidate-k` | `cli` | `structural` | `arch` | `0` | v49 | `damage_op` | cap the op's incoming candidate sweep at the K most-believed opponent moves |
| `hp_belief_mode` | `--hp-belief-mode` | `cli` | `structural` | `arch` | `'composed'` | v53 | — | how the 16 typed Hidden-Power channels are produced (composed|flat) |
| `entity_topk_seats` | `--entity-topk-seats` | `cli` | `structural` | `arch` | `0` | v54 | `damage_op`, `move_latent` | E4 — the opp active's top-K believed threat-move attention seats |
| `edge_bias_families` | `--edge-bias-families` | `cli` | `structural` | `arch` | `'off'` | v56 | — | which physics families are delivered as additive per-pair attention biases |
| `entity_tail_seats` | `--entity-tail-seats` | `cli` | `structural` | `arch` | `False` | v57 | `damage_op`, `entity_topk_seats` | E5 — 6 per-opp-mon seats summarising the beyond-top-K belief mass |
| `consequence_topk` | `--consequence-topk` | `cli` | `structural` | `arch` | `6` | v59 | — | the consequence kernels' believed-candidate axis (C1b/C2/C3 k_cand + D4 k_bench) |
| `value_threat_inject` | `--value-threat-inject` | `cli` | `structural` | `arch` | `False` | v64 | `damage_op` | add the op's alpha-weighted incoming row to our tokens on the VALUE pool's copy |
| `opp_intent` | `--opp-intent-coef` *(coef)* | `cli` | `structural` | `arch` | `False` | v68 | `entity_topk_seats` | the alpha (their move) / beta (their switch-in) supervised pointer heads |
| `species_prior_fusion` | `--species-prior-fusion` | `cli` | `structural` | `arch` | `False` | v69 | `opp_belief_slots` | read BeliefHead's species head as a DELTA on the team-composition prior |
| `t0_species_prior` | `--t0-species-prior` | `cli` | `structural` | `arch` | `False` | v72 | — | feed the T1 physics the model's own species belief, not the static usage table |
| `opp_intent_grad_mode` | — | `config_only` | `structural` | `arch` | `'detached'` | v73 | — | whether alpha/beta's gradient reaches the shared trunk (detached|shaping) |
| `intent_move_cell` | `--intent-move-cell` | `cli` | `structural` | `arch` | `False` | v77 | `opp_intent`, `damage_op` | G3 — the c2 status-consequence family re-delivered, alpha-conditioned, through the pointer MOVE cell |
| `value_entity_pool_full` | `--value-entity-pool-full` | `cli` | `structural` | `arch` | `False` | v82 | `value_entity_pool` | the entity pool's COMPLETE row set: + the refined global token and the hidden-opp belief queries |
| `history_events` | `--history-events` | `cli` | `structural` | `arch` | `False` | v81 | — | Tier H-B: the obs event-window records join the trunk as event SEATS (shared species/move embeddings, recency as content, TOKEN_TYPE_HISTORY) |
| `value_entity_pool` | `--value-entity-pool` | `cli` | `structural` | `arch` | `False` | v80 | — | Stage-3 T3-DELIVER: ONE attention pool over the critic's entity rows (12 team tokens + op incoming rows), zero-init, vf-only |
| `item_belief` | `--item-belief` | `cli` | `structural` | `arch` | `False` | v83 | — | the hidden-ITEM belief head: per-opp-slot posterior over item nums, Smogon usage prior ⊕ zero-init trunk delta; the op's p_cb unrevealed branch consumes its publication (revealed stays exact 0/1) |
| `intent_threshold` | `--intent-threshold` | `cli` | `structural` | `arch` | `False` | v84 | `opp_intent`, `damage_op` | the α-weighted threshold operator p_thresh(τ,⋛): Focus Punch / Substitute / Endure / Destiny Bond / Endeavor through the pointer MOVE cell (+ p_KO as per-slot context) |
| `intent_conditional` | `--intent-conditional` | `cli` | `structural` | `arch` | `False` | v85 | `opp_intent`, `damage_op`, `damage_outgoing`, `damage_matrices_outgoing` | the remaining α-conditioned mechanic cells: Counter/Mirror Coat's category test, flinch's (1−α_SWITCH) term, Explosion's execute/into-switch facts + the β-weighted trade KO (the FIRST forward-side β consumer), Protect's α-weighted avoided quantities, Magic Coat's oracle-verified reflect set, Pursuit's ×2 doubling trigger (port-verified departing-target rule) |
| `pair_outcome_cell` | `--pair-outcome-cell` | `cli` | `structural` | `arch` | `False` | v93 | `damage_op` | the UNIFIED per-pair OUTCOME VECTOR + its α-weighted delivery: one pair_in[their move k, our mon j] carrying damage AND status-by-identity AND neutralization AND tempo_cost in the same currency, reduced by ONE α over the move axis (Contract W) and delivered to the pointer MOVE cell |
| `pair_outcome_switch` | `--pair-outcome-switch` | `cli` | `structural` | `arch` | `False` | v94 | `damage_op` | Phase B — the SAME α-reduced unified outcome row, per DEFENDER, delivered to the pointer SWITCH cell (+ spin_denied: our Ghost candidate denying their believed Rapid Spin, priced by the hazard stake it preserves) |
| `switch_branch_cell` | `--switch-branch-cell` | `cli` | `structural` | `arch` | `False` | v94 | `opp_intent`, `damage_op`, `damage_matrices_outgoing` | Phase B — OA2, the SWITCH-BRANCH move cell: E[our move | they switch] contracted over β (the arrival), kept DECORRELATED from the stay branch, plus the Rapid-Spin spinblock (the Pursuit mirror: α_SWITCH × β × P(arrival is Ghost)) and Protect's α-derived attack mass (the c4 successor — decay × will-they-attack) |
| `conditional_threat_cell` | `--conditional-threat-cell` | `cli` | `structural` | `arch` | `False` | v95 | `damage_op`, `damage_matrices_incoming` | Phase C — OA1, the CONDITIONAL THREAT CELL (the defensive pivot): the four α-contracted coordinates the reduced outcome row structurally cannot carry — e_pko_acc (accuracy x P(KO), the product §0.2(2) says the OP must form), e_type_mult (the one channel not divided by the defender's own bulk) and the two §0.2(3) MARGINS against our own HP (max roll and crit roll), on the pointer SWITCH cell |
| `op_drop_renders` | `--op-drop-renders` | `cli` | `structural` | `arch` | `False` | v86 | — | design_op_tensors step 3: the op's flat forward block loses the three RENDER regions (omx/imx/OAX — serialization-only since the concat's deletion); selection machinery + every consumer stash survive, out_gain shrinks |
| `op_believed_lean` | `--op-believed-lean` | `cli` | `structural` | `arch` | `False` | v86 | `spread_belief`, `damage_op` | the lean d3 physics price the attacker from the BELIEVED spread instead of the legacy de-timid fiction — the B-spread correctness fix at the last de-timid site the edges read |
| `ridealong_ensemble` | `--ridealong-ensemble` | `cli` | `structural` | `critic` | `0` | v126 | `win_prob_mode` | K DETACHED win-prob heads on value_pooled (bootstrap masks + randomized priors): their disagreement is V's EPISTEMIC uncertainty (0 = off) |
| `ridealong_rnd` | `--ridealong-rnd` | `cli` | `structural` | `critic` | `False` | v126 | — | a DETACHED RND novelty head (frozen random target + trained predictor over the running-normalised RAW observation): its error is how rarely a state was seen |
| `ridealong_adv` | `--ridealong-adv` | `cli` | `structural` | `critic` | `0` | v126 | — | K DETACHED per-action A heads over the pointer head's own tokens, centred under pi, regressed on the GAE advantage of the action taken (0 = off) |
| `ridealong_opp` | `--ridealong-opp` | `cli` | `structural` | `critic` | `0` | v126 | `opp_intent` | K DETACHED opponent-effect B heads over alpha's support (believed move seats by move id + SWITCH), centred under alpha, regressed on the same advantage (0 = off) |
| `ridealong_rnd_variants` | `--ridealong-rnd-variants` | `cli` | `structural` | `critic` | `'off'` | v127 | `ridealong_rnd` | the DETACHED RND VARIANT ENSEMBLE beside --ridealong-rnd (base): a canonical comma list of fast,decay,small,feat ('all' = every one; 'off' = none), each its own predictor + optimizer + statistics, compared PAIRED against base |
| `belief_tokens` | `--belief-tokens` | `cli` | `structural` | `arch` | `'blob'` | v136 | `t0_species_prior`, `move_belief_mode`, `opp_intent`, `opp_belief_slots` | X5's opponent-belief representation: 'blob' (today's constant hidden-slot tokens) or 'fixed_mass' (concrete species hypotheses with logistic fixed-size presence, a learned delta on the Smogon prior trained by a set BCE, and an OTHER token) |

**The ARCH SURFACE.** 39 of 46 toggles are `structural` AND `family` = `arch`. That set is what `--arch production` applies and what the ARCH-SURFACE guard compares against `designs/production_config.json` on every FRESH launch (`main.train.arch_surface`, read by `--dry-run`, `python -m main.checkargs` and the launcher alike). `family` = `critic` marks a readout an experiment deliberately VARIES (the win-prob critic implies one and refuses two others), so it is excluded from both; `training_coef` / `runtime` / `resume_immutable` are excluded by CLASS.

**Dependencies.** 29 of 46 toggles name a `requires`. The column lists only DIRECT dependencies; the transitive closure is `flag_registry.requirement_closure(name)` — e.g. enabling `intent_conditional` also pulls in `opp_intent`, `entity_topk_seats`, `damage_op`, `move_belief_mode`, `attend_unrevealed_opponents`, `move_latent`, `damage_outgoing`, `damage_matrices_outgoing`. "Enabled" follows `flag_registry.is_enabled`: `False` / `0` / `'off'` / `'none'` are OFF, everything else is ON.

Two constructor checks are STRONGER than the column can say, and stay hand-written in `Gen3FeaturesExtractor.__init__`: `damage_op` needs `move_belief_mode` in *{revealed, both}* specifically (the column can only say "enabled"), and `edge_bias_families` carries a requirement PER FAMILY LETTER — most families need `damage_op`, `d1/s1/c1/c2` also need `damage_outgoing`, `d3/s3` need `entity_topk_seats > 0`, `r` needs `history_events`, and `h` needs nothing — which no flag-level declaration can represent. `flag_requires_test.py` holds that list and fails if a new coupling appears in neither place.

**Notes**

- `attend_unrevealed_opponents` — DEMOTED (config_only) and frozen ON: it is a hard prerequisite of opp_belief_cls_k>0 / opp_belief_slots / move_belief_mode!=off, so no run since v16 has turned it off. The extractor kwarg still defaults False — the OFF baseline stays constructible, it is just no longer selectable from the CLI.
- `opp_belief_slots` — coef>0 is the enable signal; the COEF is a training hparam, the BOOL is the version-checked arch toggle.
- `move_candidate_floor` — must equal damage_tables._PRIOR_FLOOR; legality itself is unconditional (v65).
- `damage_matrices_outgoing` — set by the `--damage-matrices {off,outgoing,incoming,both}` MODE flag, which desugars into this bool and `damage_matrices_incoming` before `_resolve`.
- `damage_matrices_incoming` — the other half of the `--damage-matrices` mode desugar; it also REUSES `damage_topk_k` as its K.
- `belief_grad_mode` — detach() is value-preserving => the forward is bit-identical in every mode, so check_belief_grad_mode on the resume path only.
- `opp_intent` — coef>0 is the enable signal, like opp_belief_slots. `coef_arg` is the heads' BOT-label weight, which is a separate dose from the enable signal: production trains it at 0.25 and a fresh run defaults to 1.0, so `--arch production` names it rather than setting it.
- `opp_intent_grad_mode` — DEMOTED (config_only) 2026-08-23, sweep #2, frozen at its own default. MEASURED over 107 archived run configs: the 24 runs recording it are 'detached' UNANIMOUSLY, and the flag appears in ZERO of the 107 recorded launcher commands — nobody has ever typed it. The 'shaping' arm stays CONSTRUCTIBLE (the extractor kwarg is untouched), so re-opening the trunk-exposure question costs one constructor argument, not a revert. The three other unanimous-at-default flags found by the same census (consequence_topk=6, damage_candidate_k=0, hp_belief_mode=composed) were NOT demoted: all three ARE typed in the live run's command, so removing their argparse entries would make its launcher_command unlaunchable on restart — the cleanup journey's own live-run exclusion.
- `value_entity_pool_full` — requires value_entity_pool; a separate flag/shape so v80-table checkpoints keep loading. It is the SUCCESSOR the critic-route deletion wave actually landed on — the nmr vf concat, the hidden-opp vf half and the seed readout are all deleted, and this pool carries 97% of the critic's route dependence (gen-14, dV 5.490 of all_off 5.635).
- `history_events` — the obs BLOCK is unconditional (v81 widening); this flag builds only the consumer. Gen-13 candidate arm, gated on H-A's gen-12 verdict.
- `value_entity_pool` — the designed SUCCESSOR contract of the bolt-on vf routes, and the one the critic_route_audit picked: gen-14 dV 5.490 vs threat 1.069 and every other route below 0.32. The seed readout it succeeded is deleted; threat-inject KEEPS (its deadline discharged at 1.0686).
- `item_belief` — BeliefBank's seventh row (--item-belief-coef supervises the revealed slots). Cold start posterior == the Smogon prior exactly; its CB column is within ~0.6% of the static table (row-floor renorm), so enabling is ~behavior-preserving at init and the delta must EARN its movement.
- `intent_threshold` — design_conditional_execution.md §3.0 build-order step 3. Requires opp_intent + damage_op (+ the top-K pair-cell stash at runtime). The projection is zero-init ⇒ ON-at-init bit-identical. The flag used to build a SECOND consumer, the p_KO vf route (the ledger-H1 payoff); the critic-route deletion wave retired that half on dV 0.155/0.136 against a 0.39 bar. This flag is now POLICY-ONLY, and that is deliberate.
- `intent_conditional` — design_conditional_execution.md build steps 4+5+6+7. Requires opp_intent + damage_op + damage_outgoing + damage_matrices_outgoing (the arrival pko source). β is PUBLISHED like α (label_only cuts the PPO route at the same boundary). Zero-init ⇒ ON-at-init bit-identical; G3-gated like intent_threshold.
- `pair_outcome_cell` — design_opponent_intent.md §5.1/§5.3 + design_pair_reduction.md §2.1/§9a. Phase A — the MOVE-cell half; the switch cell and the β cells are Phase B. Requires damage_op (the physics has one source) but NOT opp_intent: with no intent head α falls back to the shipped R1 belief_mean rung (α := w/Σw), so the DELIVERY claim is testable apart from the DISTRIBUTION claim. Zero-init ⇒ ON-at-init bit-identical.
- `pair_outcome_switch` — design_pair_reduction.md §2.1's CANONICAL defect, at its own sink: the switch logit's cell holds ten damage numbers, one speed number, two belief-mass numbers and NO status coordinate in any currency, so 'they will click Will-O-Wisp, bring the Natural Cure mon' is unrepresentable. The FIRST module to widen the switch cell. Requires damage_op but NOT pair_outcome_cell — the two deliver one tensor to two sinks and coupling them would make a result unattributable. Zero-init ⇒ ON-at-init bit-identical.
- `switch_branch_cell` — design_conditional_opponent_cells.md §2 + the owner's Rapid Spin / Protect specs. Requires opp_intent with NO fallback, and that is substantive: the R1 belief_mean rung is a presence belief over their MOVES and carries no switch class, so α_SWITCH would be identically 0 and every coordinate would assert 'they never switch'. §4.1's hard prerequisite is CLOSED (gen3_unrevealed_outgoing_prior_v1 prices unrevealed arrivals against the expected-latent defender); the one residue is that pko stays NULLED there, so e_pko_switch is deflated in proportion to β's hidden mass while e_high_switch carries the magnitude. Zero-init ⇒ ON-at-init bit-identical.
- `conditional_threat_cell` — design_conditional_opponent_cells.md §1 + §0.2. THREE of §1.2's clauses are SUPERSEDED and the substitutions are recorded in conditional_threat.py: the λ-weighted `w` is NOT built (pair_alpha is the shipped distribution; a second one would be a second α), `high`/`pko`/`status_lands` are already delivered by pair_outcome_switch, and §1.3's --damage-matrices-outgoing-all is VOID (deleted at v88). Requires damage_op + damage_matrices_incoming (the only producer of the per-(defender, seat) type multiplier AND of the top-K seat axis), NOT opp_intent — the R1 belief_mean fallback is MEANINGFUL here because every coordinate is a 'what lands on me if they attack' contraction. Independent of pair_outcome_switch on purpose: two quantities, one sink, attributable separately. Zero-init ⇒ ON-at-init bit-identical.
- `op_drop_renders` — every surviving offset unchanged (renders appended last), so pi/vf at init are bit-identical to renders-on — pinned by test. The prober decodes a lean run's blocks with the run's own config flags.
- `op_believed_lean` — requires spread_belief + damage_op. Forward-math only (no state_dict change): the version gate is the ONLY thing rejecting a mismatched resume.
- `ridealong_ensemble` — The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface. It REQUIRES win_prob_mode: the members predict V's own win target.
- `ridealong_rnd` — The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface. The input is the observation, not the trunk features, so the novelty is not confounded by representation drift.
- `ridealong_adv` — The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface.
- `ridealong_opp` — The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface. It REQUIRES opp_intent: B's columns and centring are alpha's. The simple pre-X5 parameterisation, to be re-based onto X5's flat pointer.
- `ridealong_rnd_variants` — The X26 RND strategy comparison (owner, 2026-09-30: "ensemble RND, toss one a different learning rate or something, so we knock them out all at once"). `fast` = base's predictor at 10x the rate; `decay` = base's predictor pulled toward its init with a 10-update half-life; `small` = a 32-unit one-hidden-layer predictor; `feat` = base's shapes over the detached value_pooled. Declarations: `agents.model.ridealong_heads.RND_VARIANT_DECLS`. The observation variants share base's target and normalisation (paired). DETACHED exactly like the four heads (pinned bit-for-bit by `ridealong_update_test`, base itself included). STRUCTURAL: the variants' parameters are the state_dict delta. family=CRITIC: never on the production ARCH surface. It REQUIRES ridealong_rnd: base is the shared target and the reference.
- `belief_tokens` — X5 (designs/endstate/design_x5_belief_tokens.md §3.8). Production stays 'blob' until the X5 A/B rules; both arms build at ONE commit, so there is no ARCH_SIGNATURE bump until the losing arm is deleted. 'blob' builds nothing (byte-identical to the pre-X5 model). 'fixed_mass' builds `agents.model.hypothesis_set.HypothesisBuilder` from a private seed (no non-X5 init byte moves) and re-targets the hidden-team belief supervision to the set BCE. Build unit U2 stashes the hypothesis set; tokens entering the trunk and the op are U3, the flat pointer U4. It REQUIRES t0_species_prior (the scores are log P_T0 + delta), move_belief_mode (the active's move group), opp_intent (the pointer it re-bases) and opp_belief_slots (the presence BCE and BeliefHead's set BCE ride --opp-belief-aux-coef).
<!-- END GENERATED: registry-table -->

## Out of scope

The registry covers the **feature-extractor architecture toggles** — the things that pass through
`build_extractor_arch_kwargs`. Three neighbouring families are deliberately not here:

- **Training-only loss coefficients** (`move_belief_coef`, `opp_belief_aux_coef`,
  `spread_belief_coef`, `win_prob_coef`, `td_aux_coef`, …) — recorded on `ModelVersion` for provenance, never
  version-gated, and they never reach the extractor. Two of them *do* appear indirectly: a
  coefficient is the CLI surface for the `opp_belief_slots` and `opp_intent` toggles, which is why
  those rows carry `derived=True` and name the coef.
- **Reward-config and PPO hparams** (`vf_coef`, `draw_penalty`, `mat_alive_weight`,
  `all_shaping_pbrs`, …) — resume-immutable value-meaning fields with their own
  `check_reward_config` / `check_vf_coef`, on a different mechanism entirely.
- **Runtime perf knobs** (`--compile-opponents`, `--compile-trainer`, `--grad-checkpointing`,
  `--async-rollout`, …) — never versioned, never in `check_compatible`, and
  deliberately **not** inherited on resume: a resume gets each one's DEFAULT, not the value the
  original launch used. For a knob that defaults OFF that means re-passing the flag each launch;
  for the three compile knobs, which default ON since 2026-08-17, it means re-passing the
  `--no-` opt-out instead.
