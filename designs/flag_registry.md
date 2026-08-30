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
50 toggles — 48 `cli`, 2 `config_only`, 0 `constructor_only`.

| toggle | CLI | tier | class | default | since | requires | meaning |
|---|---|---|---|---|---|---|---|
| `attend_unrevealed_opponents` | — | `config_only` | `structural` | `True` | v8 | — | keep the opponent's still-hidden party attendable instead of key-masking it |
| `opp_belief_cls_k` | `--opp-belief-cls-k` | `cli` | `structural` | `0` | v9 | `attend_unrevealed_opponents` | k learned query tokens summarising the unrevealed opp party into both heads |
| `opp_belief_slots` | `--opp-belief-aux-coef` *(coef)* | `cli` | `structural` | `False` | v16 | `attend_unrevealed_opponents` | learned unknown-mon tokens in the un-revealed opp slots + the BeliefHead |
| `move_belief_mode` | `--move-belief-mode` | `cli` | `structural` | `'off'` | v17 | `attend_unrevealed_opponents` | predict + reinject each opp mon's moveset (off|revealed|unrevealed|both) |
| `damage_op` | `--damage-op` | `cli` | `structural` | `False` | v19 | `move_belief_mode` | build the differentiable GPU DamageOperator |
| `move_prior_fusion` | `--move-prior-fusion` | `cli` | `structural` | `False` | v20 | `move_belief_mode` | fuse the Smogon move-frequency prior into the move belief as a log-odds delta |
| `win_prob_mode` | `--win-prob-mode` | `cli` | `structural` | `'none'` | v22 | — | auxiliary win-probability side head off value_pooled (none|read_only|shaping) |
| `damage_outgoing` | `--damage-outgoing` | `cli` | `structural` | `False` | v23 | `damage_op` | the op's OUTGOING per-move direction (our active's moves -> the opp active) |
| `move_candidate_floor` | `--move-candidate-floor` | `cli` | `structural` | `0.02` | v23 | — | the LEGAL-BUT-UNOBSERVED base probability of the move prior |
| `move_latent` | `--move-latent` | `cli` | `structural` | `False` | v24 | — | the context-free MoveLatentEncoder concatenated into the move network |
| `spread_belief` | `--spread-belief` | `cli` | `structural` | `False` | v25 | — | predict + reinject the opponent's hidden spread (5 derived stats per slot) |
| `value_dist_mode` | `--value-dist-mode` | `cli` | `structural` | `'none'` | v29 | `value_dist_bins` | distributional VALUE side head off value_pooled (none|read_only|shaping) |
| `value_dist_bins` | `--value-dist-bins` | `cli` | `structural` | `0` | v29 | — | atom count = the value-dist head's output Linear width |
| `value_dist_vmin` | `--value-dist-vmin` | `cli` | `resume_immutable` | `0.0` | v29 | — | low end of the return range the value-dist atoms span |
| `value_dist_vmax` | `--value-dist-vmax` | `cli` | `resume_immutable` | `0.0` | v29 | — | high end of the return range the value-dist atoms span |
| `damage_topk_k` | `--damage-topk` | `cli` | `structural` | `0` | v30 | `damage_op`, `move_latent`, `damage_matrices_incoming` | K = how many of the opp active's believed moves the incoming matrix surfaces |
| `damage_matrices_outgoing` | `--damage-matrices` | `cli` | `structural` | `False` | v34 | `damage_op` | our active's 4 moves x the opp's 6 mons, per-(move, mon) rolls |
| `damage_matrices_incoming` | `--damage-matrices` | `cli` | `structural` | `False` | v35 | `damage_op`, `move_latent` | the enriched top-K incoming matrix (per opp move x per our mon) |
| `threat_prob_outspeed` | `--threat-prob-outspeed` | `cli` | `structural` | `False` | v36 | — | uncertainty-aware P(outspeed): divide the speed gap by the believed speed std |
| `spread_belief_nature` | `--spread-belief-nature` | `cli` | `structural` | `False` | v40 | `spread_belief` | swap SpreadBelief's additive head for the NATURE/EV generative head |
| `belief_grad_mode` | `--belief-grad-mode` | `cli` | `resume_immutable` | `'shaping'` | v41 | — | which gradient arrow between the belief heads and the trunk is cut |
| `damage_candidate_k` | `--damage-candidate-k` | `cli` | `structural` | `0` | v49 | `damage_op` | cap the op's incoming candidate sweep at the K most-believed opponent moves |
| `hp_belief_mode` | `--hp-belief-mode` | `cli` | `structural` | `'composed'` | v53 | — | how the 16 typed Hidden-Power channels are produced (composed|flat) |
| `entity_topk_seats` | `--entity-topk-seats` | `cli` | `structural` | `0` | v54 | `damage_op`, `move_latent` | E4 — the opp active's top-K believed threat-move attention seats |
| `edge_bias_families` | `--edge-bias-families` | `cli` | `structural` | `'off'` | v56 | — | which physics families are delivered as additive per-pair attention biases |
| `entity_tail_seats` | `--entity-tail-seats` | `cli` | `structural` | `False` | v57 | `damage_op`, `entity_topk_seats` | E5 — 6 per-opp-mon seats summarising the beyond-top-K belief mass |
| `consequence_topk` | `--consequence-topk` | `cli` | `structural` | `6` | v59 | — | the consequence kernels' believed-candidate axis (C1b/C2/C3 k_cand + D4 k_bench) |
| `value_threat_inject` | `--value-threat-inject` | `cli` | `structural` | `False` | v64 | `damage_op` | add the op's alpha-weighted incoming row to our tokens on the VALUE pool's copy |
| `opp_intent` | `--opp-intent-coef` *(coef)* | `cli` | `structural` | `False` | v68 | `entity_topk_seats` | the alpha (their move) / beta (their switch-in) supervised pointer heads |
| `species_prior_fusion` | `--species-prior-fusion` | `cli` | `structural` | `False` | v69 | `opp_belief_slots` | read BeliefHead's species head as a DELTA on the team-composition prior |
| `t0_species_prior` | `--t0-species-prior` | `cli` | `structural` | `False` | v72 | — | feed the T1 physics the model's own species belief, not the static usage table |
| `opp_intent_grad_mode` | — | `config_only` | `structural` | `'detached'` | v73 | — | whether alpha/beta's gradient reaches the shared trunk (detached|shaping) |
| `intent_move_cell` | `--intent-move-cell` | `cli` | `structural` | `False` | v77 | `opp_intent`, `damage_op` | G3 — the c2 status-consequence family re-delivered, alpha-conditioned, through the pointer MOVE cell |
| `value_entity_pool_full` | `--value-entity-pool-full` | `cli` | `structural` | `False` | v82 | `value_entity_pool` | the entity pool's COMPLETE row set: + the refined global token and the hidden-opp belief queries |
| `history_events` | `--history-events` | `cli` | `structural` | `False` | v81 | — | Tier H-B: the obs event-window records join the trunk as event SEATS (shared species/move embeddings, recency as content, TOKEN_TYPE_HISTORY) |
| `value_entity_pool` | `--value-entity-pool` | `cli` | `structural` | `False` | v80 | — | Stage-3 T3-DELIVER: ONE attention pool over the critic's entity rows (12 team tokens + op incoming rows), zero-init, vf-only |
| `item_belief` | `--item-belief` | `cli` | `structural` | `False` | v83 | — | the hidden-ITEM belief head: per-opp-slot posterior over item nums, Smogon usage prior ⊕ zero-init trunk delta; the op's p_cb unrevealed branch consumes its publication (revealed stays exact 0/1) |
| `intent_threshold` | `--intent-threshold` | `cli` | `structural` | `False` | v84 | `opp_intent`, `damage_op` | the α-weighted threshold operator p_thresh(τ,⋛): Focus Punch / Substitute / Endure / Destiny Bond / Endeavor through the pointer MOVE cell (+ p_KO as per-slot context) |
| `intent_conditional` | `--intent-conditional` | `cli` | `structural` | `False` | v85 | `opp_intent`, `damage_op`, `damage_outgoing`, `damage_matrices_outgoing` | the remaining α-conditioned mechanic cells: Counter/Mirror Coat's category test, flinch's (1−α_SWITCH) term, Explosion's execute/into-switch facts + the β-weighted trade KO (the FIRST forward-side β consumer), Protect's α-weighted avoided quantities, Magic Coat's oracle-verified reflect set, Pursuit's ×2 doubling trigger (port-verified departing-target rule) |
| `pair_outcome_cell` | `--pair-outcome-cell` | `cli` | `structural` | `False` | v93 | `damage_op` | the UNIFIED per-pair OUTCOME VECTOR + its α-weighted delivery: one pair_in[their move k, our mon j] carrying damage AND status-by-identity AND neutralization AND tempo_cost in the same currency, reduced by ONE α over the move axis (Contract W) and delivered to the pointer MOVE cell |
| `pair_outcome_switch` | `--pair-outcome-switch` | `cli` | `structural` | `False` | v94 | `damage_op` | Phase B — the SAME α-reduced unified outcome row, per DEFENDER, delivered to the pointer SWITCH cell (+ spin_denied: our Ghost candidate denying their believed Rapid Spin, priced by the hazard stake it preserves) |
| `switch_branch_cell` | `--switch-branch-cell` | `cli` | `structural` | `False` | v94 | `opp_intent`, `damage_op`, `damage_matrices_outgoing` | Phase B — OA2, the SWITCH-BRANCH move cell: E[our move | they switch] contracted over β (the arrival), kept DECORRELATED from the stay branch, plus the Rapid-Spin spinblock (the Pursuit mirror: α_SWITCH × β × P(arrival is Ghost)) and Protect's α-derived attack mass (the c4 successor — decay × will-they-attack) |
| `conditional_threat_cell` | `--conditional-threat-cell` | `cli` | `structural` | `False` | v95 | `damage_op`, `damage_matrices_incoming` | Phase C — OA1, the CONDITIONAL THREAT CELL (the defensive pivot): the four α-contracted coordinates the reduced outcome row structurally cannot carry — e_pko_acc (accuracy x P(KO), the product §0.2(2) says the OP must form), e_type_mult (the one channel not divided by the defender's own bulk) and the two §0.2(3) MARGINS against our own HP (max roll and crit roll), on the pointer SWITCH cell |
| `pair_value_route` | `--pair-value-route` | `cli` | `structural` | `False` | v95 | `damage_op` | Phase C — PV, the pair-VALUE CRITIC route: the α-reduced unified outcome row for our mon j injected as TOKEN CONTENT on mon j's own token inside CLSPool, on the VALUE pool's copy only — the first per-entity route by which the critic reads the status / neutralization / tempo currency at all (today it reaches vf only as the s3 edge family's softmax-normalised RATIO) |
| `op_drop_renders` | `--op-drop-renders` | `cli` | `structural` | `False` | v86 | — | design_op_tensors step 3: the op's flat forward block loses the three RENDER regions (omx/imx/OAX — serialization-only since the concat's deletion); selection machinery + every consumer stash survive, out_gain shrinks |
| `op_believed_lean` | `--op-believed-lean` | `cli` | `structural` | `False` | v86 | `spread_belief`, `damage_op` | the lean d3 physics price the attacker from the BELIEVED spread instead of the legacy de-timid fiction — the B-spread correctness fix at the last de-timid site the edges read |
| `cf_evidential` | `--cf-evidential` | `cli` | `structural` | `False` | v98 | — | the EVIDENTIAL Beta readout over P(win|state) off value_pooled — (α, β) via softplus+1, trained by the Beta-Binomial marginal likelihood of the counterfactual factory's rollout COUNTS |
| `cf_twin_heads` | `--cf-twin-heads` | `cli` | `structural` | `False` | v99 | `win_prob_mode` | the TWIN win-prob heads B and C off value_pooled — B trained on the cf-labelled states with SINGLE-OUTCOME labels, C on the same states with TIGHT-MC labels, both additionally carrying head A's on-policy BCE |
| `cf_shadow_critic` | `--cf-shadow-critic` | `cli` | `structural` | `False` | v99 | — | the passive SHADOW CRITIC off value_pooled — a value twin trained on tight-MC `mc_return` labels (the run's own shaped return, PopArt frame), which never computes an advantage and never enters GAE |
| `q_winprob_mode` | `--q-winprob-mode` | `cli` | `structural` | `'none'` | v107 | — | the PER-ACTION win-probability readout over the pointer head's own action tokens (none|read_only) — one forward, eleven P(win|s,a): the amortized one-ply search leaf |

**Dependencies.** 28 of 50 toggles name a `requires`. The column lists only DIRECT dependencies; the transitive closure is `flag_registry.requirement_closure(name)` — e.g. enabling `intent_conditional` also pulls in `opp_intent`, `entity_topk_seats`, `damage_op`, `move_belief_mode`, `attend_unrevealed_opponents`, `move_latent`, `damage_outgoing`, `damage_matrices_outgoing`. "Enabled" follows `flag_registry.is_enabled`: `False` / `0` / `'off'` / `'none'` are OFF, everything else is ON.

Two constructor checks are STRONGER than the column can say, and stay hand-written in `Gen3FeaturesExtractor.__init__`: `damage_op` needs `move_belief_mode` in *{revealed, both}* specifically (the column can only say "enabled"), and `edge_bias_families` carries a requirement PER FAMILY LETTER — most families need `damage_op`, `d1/s1/c1/c2` also need `damage_outgoing`, `d3/s3` need `entity_topk_seats > 0`, `r` needs `history_events`, and `h` needs nothing — which no flag-level declaration can represent. `flag_requires_test.py` holds that list and fails if a new coupling appears in neither place.

**Notes**

- `attend_unrevealed_opponents` — DEMOTED (config_only) and frozen ON: it is a hard prerequisite of opp_belief_cls_k>0 / opp_belief_slots / move_belief_mode!=off, so no run since v16 has turned it off. The extractor kwarg still defaults False — the OFF baseline stays constructible, it is just no longer selectable from the CLI.
- `opp_belief_slots` — coef>0 is the enable signal; the COEF is a training hparam, the BOOL is the version-checked arch toggle.
- `move_candidate_floor` — must equal damage_tables._PRIOR_FLOOR; legality itself is unconditional (v65).
- `value_dist_vmin` — value-MEANING, so check_value_dist on the resume path only.
- `value_dist_vmax` — value-MEANING, so check_value_dist on the resume path only.
- `damage_matrices_outgoing` — set by the `--damage-matrices {off,outgoing,incoming,both}` MODE flag, which desugars into this bool and `damage_matrices_incoming` before `_resolve`.
- `damage_matrices_incoming` — the other half of the `--damage-matrices` mode desugar; it also REUSES `damage_topk_k` as its K.
- `belief_grad_mode` — detach() is value-preserving => the forward is bit-identical in every mode, so check_belief_grad_mode on the resume path only.
- `opp_intent` — coef>0 is the enable signal, like opp_belief_slots.
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
- `pair_value_route` — design_opponent_intent.md §7a(2). ⚠️ C4 RE-ENTRY CONDITION: any α/β-critic route may be BUILT opt-in but its ENABLING owes the C4-style offline gate first (ledger C6 — five v89 routes trained off zero with entity_pool carrying decisively and the critic's stall over-confidence did NOT move; the delivery line is EXHAUSTED). Token content, NOT the v89 _value_pooled_routes seam: a post-pool additive route must collapse the J axis and the only equivariant collapse is a sum, which cannot tell one mon losing 90% from six losing 15%. α is the R1 belief_mean rung UNCONDITIONALLY — ORDERING, not preference: value_cls pools before the α/β heads are scored, and §7a(2) pre-registers that substitution as the DELIVERY-claim test. vf-only at ANY weight (the augmented tensor is a local); zero-init ⇒ ON-at-init bit-identical. Requires damage_op.
- `op_drop_renders` — every surviving offset unchanged (renders appended last), so pi/vf at init are bit-identical to renders-on — pinned by test. The prober decodes a lean run's blocks with the run's own config flags.
- `op_believed_lean` — requires spread_belief + damage_op. Forward-math only (no state_dict change): the version gate is the ONLY thing rejecting a mismatched resume.
- `cf_evidential` — IN the registry because it is a `Gen3FeaturesExtractor` constructor kwarg that builds a MODULE — the win_prob_mode / value_dist_mode precedent exactly, and the registry's declared scope is 'the things that pass through build_extractor_arch_kwargs'. Its two coefficients (--cf-evidential-coef / --cf-evidential-reg) are NOT: they are training-only loss weights in the --opd-coef / --cf-winprob-coef class, set on the MODEL rather than the extractor. Stronger than its two precedents in one way: there is no read_only/shaping split, because the head's input is detached UNCONDITIONALLY — it is a pure supervised readout that feeds nothing forward and is not even CALLED by the forward pass. So OFF is byte-identical AND ON-at-coef-0 is bit-identical in pi/vf (it is built LAST, so no earlier module's init RNG draw moves). No `requires`: it reads `value_pooled`, which is unconditional.
- `cf_twin_heads` — The owner-authorized amendment to the signed R1 pre-registration (ledger 2026-08-22 evening, 'Three owner sign-offs' item 3): the arm's primary comparison becomes a WITHIN-RUN paired head difference instead of a run-vs-run one, so B-A isolates coverage and C-B isolates pure variance reduction on an identical trunk over identical states. IN the registry for the `cf_evidential`/`win_prob_mode` reason: a Gen3FeaturesExtractor constructor kwarg that builds MODULES. Its coefficient (--cf-twin-coef) is NOT — a training-only loss weight in the --opd-coef class. Head-only ALWAYS in v1 (both twins read a DETACHED value_pooled in every term), so this measures the LABEL effect on a trunk frozen with respect to them; trunk exposure stays a cross-run question. Never called by the forward, built LAST: OFF byte-identical, ON-at-coef-0 bit-identical in pi/vf. No `requires` on cf_evidential (orthogonal readouts of the same unconditional value_pooled). It DOES require `win_prob_mode`, and that is declared rather than left to the CLI: head A is `win_head`, so `--win-prob-mode none` leaves the factorial with no control arm — and `checkargs` walks this column, so without it an operator validating a recorded launcher_command gets exit 0 on a command the child then refuses, which is the launch-crash-fix loop checkargs exists to end.
- `cf_shadow_critic` — The staged PROMOTION PATH for critic surgery, not the surgery: swapping the live critic for an MC-grounded one is a critic ROUTE change and owes the C4 offline gate, so this head accumulates the evidence (shadow-vs-live-V divergence on the same minibatch states) as a published number instead of an argument. The `pubval` structural precedent. Detached ALWAYS — there is no read_only/shaping split — and never called by the forward, so OFF is byte-identical and ON-at-coef-0 is bit-identical in pi/vf. Its coefficient (--cf-shadow-coef) is training-only, the --opd-coef class. No `requires`: it reads value_pooled, which is unconditional.
- `q_winprob_mode` — E5 step 1 (ledger 229e9f1 / 5edbd05). STRUCTURAL in the win_prob_mode mould — a Gen3FeaturesExtractor constructor kwarg that builds a MODULE, so its params are the state_dict delta and a bool-ish compare in check_compatible is the only thing that can reject a resume that flips it. TWO values, not three: there is deliberately NO `shaping`, because a per-action readout carrying a COUNTERFACTUAL label is a strictly larger leak surface than a per-state one, so trunk exposure is a later decision that owes its own gate. `read_only` detaches EVERY input (context, tokens and cells alike), so pi/vf are bit-identical at any coefficient; `none` does not build the module at all, so OFF is byte-for-byte the baseline and no earlier module's init RNG draw moves (built LAST). Unlike the four cf readouts the forward DOES call it — eleven Q values are only useful if the forward that chose the action can publish them. Its two coefficients (--q-winprob-coef, --q-winprob-onpolicy-coef) are NOT here: training-only loss weights in the --cf-winprob-coef class. No `requires` — it reads `value_pooled` and `stash.pointer_inputs`, both unconditional.
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
  `--async-rollout`, `--use-bridge`, …) — never versioned, never in `check_compatible`, and
  deliberately **not** inherited on resume: a resume gets each one's DEFAULT, not the value the
  original launch used. For a knob that defaults OFF that means re-passing the flag each launch;
  for the three compile knobs, which default ON since 2026-08-17, it means re-passing the
  `--no-` opt-out instead.
