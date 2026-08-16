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

## The registry

<!-- BEGIN GENERATED: registry-table -->
43 toggles — 40 `cli`, 3 `config_only`, 0 `constructor_only`.

| toggle | CLI | tier | class | default | since | meaning |
|---|---|---|---|---|---|---|
| `attend_unrevealed_opponents` | — | `config_only` | `structural` | `True` | v8 | keep the opponent's still-hidden party attendable instead of key-masking it |
| `opp_belief_cls_k` | `--opp-belief-cls-k` | `cli` | `structural` | `0` | v9 | k learned query tokens summarising the unrevealed opp party into both heads |
| `value_active_readout` | — | `config_only` | `structural` | `False` | v10 | route our active mon's refined token into the VALUE projection |
| `opp_belief_slots` | `--opp-belief-aux-coef` *(coef)* | `cli` | `structural` | `False` | v16 | learned unknown-mon tokens in the un-revealed opp slots + the BeliefHead |
| `move_belief_mode` | `--move-belief-mode` | `cli` | `structural` | `'off'` | v17 | predict + reinject each opp mon's moveset (off|revealed|unrevealed|both) |
| `damage_op` | `--damage-op` | `cli` | `structural` | `False` | v19 | build the differentiable GPU DamageOperator |
| `move_prior_fusion` | `--move-prior-fusion` | `cli` | `structural` | `False` | v20 | fuse the Smogon move-frequency prior into the move belief as a log-odds delta |
| `win_prob_mode` | `--win-prob-mode` | `cli` | `structural` | `'none'` | v22 | auxiliary win-probability side head off value_pooled (none|read_only|shaping) |
| `damage_outgoing` | `--damage-outgoing` | `cli` | `structural` | `False` | v23 | the op's OUTGOING per-move direction (our active's moves -> the opp active) |
| `move_candidate_floor` | `--move-candidate-floor` | `cli` | `structural` | `0.02` | v23 | the LEGAL-BUT-UNOBSERVED base probability of the move prior |
| `move_latent` | `--move-latent` | `cli` | `structural` | `False` | v24 | the context-free MoveLatentEncoder concatenated into the move network |
| `spread_belief` | `--spread-belief` | `cli` | `structural` | `False` | v25 | predict + reinject the opponent's hidden spread (5 derived stats per slot) |
| `value_dist_mode` | `--value-dist-mode` | `cli` | `structural` | `'none'` | v29 | distributional VALUE side head off value_pooled (none|read_only|shaping) |
| `value_dist_bins` | `--value-dist-bins` | `cli` | `structural` | `0` | v29 | atom count = the value-dist head's output Linear width |
| `value_dist_vmin` | `--value-dist-vmin` | `cli` | `resume_immutable` | `0.0` | v29 | low end of the return range the value-dist atoms span |
| `value_dist_vmax` | `--value-dist-vmax` | `cli` | `resume_immutable` | `0.0` | v29 | high end of the return range the value-dist atoms span |
| `damage_topk_k` | `--damage-topk` | `cli` | `structural` | `0` | v30 | K = how many of the opp active's believed moves the incoming matrix surfaces |
| `damage_matrices_outgoing` | `--damage-matrices` | `cli` | `structural` | `False` | v34 | our active's 4 moves x the opp's 6 mons, per-(move, mon) rolls |
| `damage_matrices_incoming` | `--damage-matrices` | `cli` | `structural` | `False` | v35 | the enriched top-K incoming matrix (per opp move x per our mon) |
| `threat_prob_outspeed` | `--threat-prob-outspeed` | `cli` | `structural` | `False` | v36 | uncertainty-aware P(outspeed): divide the speed gap by the believed speed std |
| `damage_matrices_outgoing_all` | — | `config_only` | `structural` | `False` | v39 | the TRANSPOSED outgoing matrix — our 6 mons' 4 moves -> the opp ACTIVE |
| `spread_belief_nature` | `--spread-belief-nature` | `cli` | `structural` | `False` | v40 | swap SpreadBelief's additive head for the NATURE/EV generative head |
| `belief_grad_mode` | `--belief-grad-mode` | `cli` | `resume_immutable` | `'shaping'` | v41 | which gradient arrow between the belief heads and the trunk is cut |
| `pubval_mode` | `--pubval-mode` | `cli` | `structural` | `'none'` | v43 | PUBLIC-information value aux head regressed toward the frozen V_pub logistic |
| `damage_candidate_k` | `--damage-candidate-k` | `cli` | `structural` | `0` | v49 | cap the op's incoming candidate sweep at the K most-believed opponent moves |
| `hp_belief_mode` | `--hp-belief-mode` | `cli` | `structural` | `'composed'` | v53 | how the 16 typed Hidden-Power channels are produced (composed|flat) |
| `entity_topk_seats` | `--entity-topk-seats` | `cli` | `structural` | `0` | v54 | E4 — the opp active's top-K believed threat-move attention seats |
| `edge_bias_families` | `--edge-bias-families` | `cli` | `structural` | `'off'` | v56 | which physics families are delivered as additive per-pair attention biases |
| `entity_tail_seats` | `--entity-tail-seats` | `cli` | `structural` | `False` | v57 | E5 — 6 per-opp-mon seats summarising the beyond-top-K belief mass |
| `consequence_topk` | `--consequence-topk` | `cli` | `structural` | `6` | v59 | the consequence kernels' believed-candidate axis (C1b/C2/C3 k_cand + D4 k_bench) |
| `value_threat_inject` | `--value-threat-inject` | `cli` | `structural` | `False` | v64 | add the op's alpha-weighted incoming row to our tokens on the VALUE pool's copy |
| `opp_intent` | `--opp-intent-coef` *(coef)* | `cli` | `structural` | `False` | v68 | the alpha (their move) / beta (their switch-in) supervised pointer heads |
| `species_prior_fusion` | `--species-prior-fusion` | `cli` | `structural` | `False` | v69 | read BeliefHead's species head as a DELTA on the team-composition prior |
| `t0_species_prior` | `--t0-species-prior` | `cli` | `structural` | `False` | v72 | feed the T1 physics the model's own species belief, not the static usage table |
| `opp_intent_grad_mode` | `--opp-intent-grad-mode` | `cli` | `structural` | `'detached'` | v73 | whether alpha/beta's gradient reaches the shared trunk (detached|shaping) |
| `intent_value_reduce` | `--intent-value-reduce` | `cli` | `structural` | `False` | v74 | append the alpha-weighted expected incoming threat to the critic's features |
| `intent_move_cell` | `--intent-move-cell` | `cli` | `structural` | `False` | v77 | G3 — the c2 status-consequence family re-delivered, alpha-conditioned, through the pointer MOVE cell |
| `value_entity_pool_full` | `--value-entity-pool-full` | `cli` | `structural` | `False` | v82 | the entity pool's COMPLETE row set: + the refined global token and the hidden-opp belief queries |
| `history_events` | `--history-events` | `cli` | `structural` | `False` | v81 | Tier H-B: the obs event-window records join the trunk as event SEATS (shared species/move embeddings, recency as content, TOKEN_TYPE_HISTORY) |
| `value_entity_pool` | `--value-entity-pool` | `cli` | `structural` | `False` | v80 | Stage-3 T3-DELIVER: ONE attention pool over the critic's entity rows (12 team tokens + op incoming rows), zero-init, vf-only |
| `item_belief` | `--item-belief` | `cli` | `structural` | `False` | v83 | the hidden-ITEM belief head: per-opp-slot posterior over item nums, Smogon usage prior ⊕ zero-init trunk delta; the op's p_cb unrevealed branch consumes its publication (revealed stays exact 0/1) |
| `intent_threshold` | `--intent-threshold` | `cli` | `structural` | `False` | v84 | the α-weighted threshold operator p_thresh(τ,⋛): Focus Punch / Substitute / Endure / Destiny Bond / Endeavor through the pointer MOVE cell, plus p_KO (the calibrated am-I-about-to-die) to the critic |
| `intent_conditional` | `--intent-conditional` | `cli` | `structural` | `False` | v85 | the remaining α-conditioned mechanic cells: Counter/Mirror Coat's category test, flinch's (1−α_SWITCH) term, Explosion's execute/into-switch facts + the β-weighted trade KO (the FIRST forward-side β consumer), Protect's α-weighted avoided quantities, Magic Coat's oracle-verified reflect set, Pursuit's ×2 doubling trigger (port-verified departing-target rule) |

**Notes**

- `attend_unrevealed_opponents` — DEMOTED (config_only) and frozen ON: it is a hard prerequisite of opp_belief_cls_k>0 / opp_belief_slots / move_belief_mode!=off, so no run since v16 has turned it off. The extractor kwarg still defaults False — the OFF baseline stays constructible, it is just no longer selectable from the CLI.
- `value_active_readout` — DEMOTED (config_only), frozen OFF: superseded by the MultiSeedValueReadout (v61) and --value-threat-inject (v64); never enabled in a gen-8/9/10 run.
- `opp_belief_slots` — coef>0 is the enable signal; the COEF is a training hparam, the BOOL is the version-checked arch toggle.
- `move_candidate_floor` — must equal damage_tables._PRIOR_FLOOR; legality itself is unconditional (v65).
- `value_dist_vmin` — value-MEANING, so check_value_dist on the resume path only.
- `value_dist_vmax` — value-MEANING, so check_value_dist on the resume path only.
- `damage_matrices_outgoing` — set by the `--damage-matrices {off,outgoing,incoming,both}` MODE flag, which desugars into this bool and `damage_matrices_incoming` before `_resolve`.
- `damage_matrices_incoming` — the other half of the `--damage-matrices` mode desugar; it also REUSES `damage_topk_k` as its K.
- `damage_matrices_outgoing_all` — DEMOTED (config_only), frozen OFF: never enabled in a gen-8/9/10 run; the switch-in offense read it prices is delivered by the d2 edge family.
- `belief_grad_mode` — detach() is value-preserving => the forward is bit-identical in every mode, so check_belief_grad_mode on the resume path only.
- `opp_intent` — coef>0 is the enable signal, like opp_belief_slots.
- `value_entity_pool_full` — requires value_entity_pool; a separate flag/shape so v80-table checkpoints (gen-12 trains one) keep loading. The one successor for every vf route the critic_route_audit may condemn (nmr concat, hidden-opp vf, seed, threat).
- `history_events` — the obs BLOCK is unconditional (v81 widening); this flag builds only the consumer. Gen-13 candidate arm, gated on H-A's gen-12 verdict.
- `value_entity_pool` — the designed SUCCESSOR contract of the bolt-on vf routes (seed readout / threat-inject) — those are adjudicated by the gen-11 critic_route_audit; this exists so a condemned route has a replacement the next generation can enable in the same config.
- `item_belief` — BeliefBank's seventh row (--item-belief-coef supervises the revealed slots). Cold start posterior == the Smogon prior exactly; its CB column is within ~0.6% of the static table (row-floor renorm), so enabling is ~behavior-preserving at init and the delta must EARN its movement.
- `intent_threshold` — design_conditional_execution.md §3.0 build-order step 3. Requires opp_intent + damage_op (+ the top-K pair-cell stash at runtime). Both projections zero-init ⇒ ON-at-init bit-identical; the p_KO critic half is the ledger-H1 payoff and stands whatever the G3 verdict says.
- `intent_conditional` — design_conditional_execution.md build steps 4+5+6+7. Requires opp_intent + damage_op + damage_outgoing + damage_matrices_outgoing (the arrival pko source). β is PUBLISHED like α (label_only cuts the PPO route at the same boundary). Zero-init ⇒ ON-at-init bit-identical; G3-gated like intent_threshold.
<!-- END GENERATED: registry-table -->

## Out of scope

The registry covers the **feature-extractor architecture toggles** — the things that pass through
`build_extractor_arch_kwargs`. Three neighbouring families are deliberately not here:

- **Training-only loss coefficients** (`move_belief_coef`, `opp_belief_aux_coef`,
  `spread_belief_coef`, `win_prob_coef`, …) — recorded on `ModelVersion` for provenance, never
  version-gated, and they never reach the extractor. Two of them *do* appear indirectly: a
  coefficient is the CLI surface for the `opp_belief_slots` and `opp_intent` toggles, which is why
  those rows carry `derived=True` and name the coef.
- **Reward-config and PPO hparams** (`vf_coef`, `draw_penalty`, `mat_alive_weight`,
  `all_shaping_pbrs`, …) — resume-immutable value-meaning fields with their own
  `check_reward_config` / `check_vf_coef`, on a different mechanism entirely.
- **Runtime perf knobs** (`--compile-opponents`, `--compile-trainer`, `--grad-checkpointing`,
  `--async-rollout`, `--use-bridge`, …) — never versioned, never in `check_compatible`, and
  deliberately **not** inherited on resume, so they must be re-passed each launch.
