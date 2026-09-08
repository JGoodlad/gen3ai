# The config-version ladder — v51 → v111, as of 2026-09-07

> **This directory is HISTORY — additive only.** A file here is written once, when narrative is
> lifted out of a `CLAUDE.md`, and is never updated afterwards; when the world changes, the
> `CLAUDE.md` changes and this file stays as the record of what was believed then.

Lifted verbatim from the **Code on main** cell of `designs/CLAUDE.md`'s state table on
**2026-09-07**, when that cell (21.7 KB in one table cell) was cut to a pointer. It is the
version-by-version narrative of `MODEL_CONFIG_VERSION` — what each bump changed, what it deleted,
and the evidence each deletion rode on.

🚨 **Read the live `MODEL_CONFIG_VERSION` / `ARCH_SIGNATURE` / `MIGRATION_FLOOR` from
`src/agents/model/model_version/constants.py` + `migrations.py`, never from this file** — every
number below was true on 2026-09-07 and nothing here is maintained. `designs/CHANGELOG.md` is the
append-only history of record; `designs/ARCHITECTURE.md` states what is true now.

---

`MODEL_CONFIG_VERSION` **111**, `ARCH_SIGNATURE` **`gen3_critic_route_wave_v1`**, `MIGRATION_FLOOR`
**96** (v110 = `gen3_frozen_phi_actor_only_v1`, v111 = `gen3_arch_surface_guard_v1`, both 2026-09-06
— this line was corrected from a stale **109** on 2026-09-07; ALWAYS read the live value from the
code, never from this prose) — all three read from `src/agents/model/model_version/constants.py` +
`migrations.py`, never from this prose. **The signature has NOT moved since v96**, so v97–v109 are
all additive/provenance bumps and every v96+ checkpoint still loads: **112 of the 217 runs under
`models/` carry the live signature and load today** (measured 2026-09-06; the script and the full
list are in
[`research_state/era_boundary_deprecation_2026-09-06.md`](../era_boundary_deprecation_2026-09-06.md)).
**v109 `gen3_winprob_critic_mode_v1` — `--critic {shaped,winprob}`, WHICH READOUT IS THE CRITIC**,
the first ai_v12 code to land (design of record
[`ai_v12/design_winprob_only_critic.md`](../../ai_v12/design_winprob_only_critic.md)). `shaped` is the
DEFAULT and is today's critic exactly, so a flagless run is byte-identical and there is NO
`ARCH_SIGNATURE` bump; `winprob` makes `V(s) = sigmoid(win_head logit)` in [0,1], the value loss
that head's BCE against the terminal WIN INDICATOR at `--vf-coef`, γ = 1, no PBRS, no PopArt, no
distributional head — `V(s) = P(win\|s)` with no approximation term, which is why
`--terminal-indicator` and `--victory-value 1.0` are REQUIREMENTS. **13 combination refusals** live
in `main.train.combination_checks` and `python -m main.checkargs` reports every one offline. **The
signature bump belongs to the DEFAULT FLIP, a separate commit** — a critic trained to predict a
shaped return cannot be warm-started into predicting a probability. The same bump carries the
resume-immutable `terminal_indicator` / `no_progress_tax_armed`, and `--gamma` stops being a
hardcoded `0.9999`. Prior: **v108 `gen3_dead_flag_purge_v2`** — the end-state flag census before the
ai_v12 runs
([`research_state/flag_census_2026-09-06.md`](../flag_census_2026-09-06.md)): the
registry and the argparse surface measured against the archive, and **exactly ONE flag deleted**
(`threat_prob_outspeed`, 0 of 123 gen-9+ runs ON at census time). ⚠️ **Recount before quoting a
surface count** — the census's own rule, and it has already moved: **49 registry entries and 261
argparse actions (260 excluding `-h`) on 2026-09-06**, against the 49 / 256 the census banked, and
217 runs / 126 gen-9+ against its 214 / 123. Its migration REFUSES a recorded `True` rather than
popping it — the flag built NO parameters, so a `True` and a `False` checkpoint are byte-identical
in every key and a silent pop would run a checkpoint under physics it never trained on with every
gate green; *loading cleanly is the reason to refuse, not a reason to allow*. **A purge that finds
little is what a healthy surface looks like.** Also v108-era: `gen3_last_snapshot_resolution_v1` —
**a bare run DIRECTORY now means the run's LAST SNAPSHOT**, not the bot-selected `best_model`. One
choke point (`fixed_opponent_pool.resolve_model_ref`) serves `--distill-teacher` /
`--stable-opponents` / `--exploiter` / `--warmstart-consensus` / `--distill-anchor-parent` /
`--win-prob-pbrs-source`, rungs `latest.txt` → highest-step `checkpoints/` zip → `final_model*.zip`
→ `best_model/best_model.zip` LAST, higher `num_timesteps` winning a disagreement; the resolved file
+ rung are recorded in `metadata.json`'s `lineage`. **Every teacher loaded before 2026-09-06 went
through the OLD best_model-first order** — 2 of 8 R5F teachers were ~0.93M-step checkpoints rather
than the ~2.93M finals — and `main.lineage` prints `resolved file not recorded` for those rather
than re-resolving them under today's rule. Prior: **v107 `gen3_q_winprob_head_v1`**
(`--q-winprob-mode read_only`: eleven `P(win\|s,a)` off the pointer stash, every input detached, no
`shaping` value by design), **v106 `gen3_progress_clock_intent_v1`** (`--progress-decision-tense` /
`--progress-switch-freeze`), **v105 `gen3_clean_world_config_v1`** (`--no-hand-shaping` — the master
off-switch for all eight hand PBRS potentials AND the whole BIAS class — plus `--pbrs-material` /
`--pbrs-belief` / `--victory-value` / `--win-prob-pbrs-source`), **v104 `gen3_winprob_pbrs_v1`**
(`--win-prob-pbrs-coef`, the first knob in the family that edits the REWARD STREAM rather than a
loss term), **v103 `gen3_distill_target_gate_v1`** (`--distill-target action` + top-K + the
advantage gate + the `rank/policy_pr` tripwire), **v102 `gen3_policy_grad_coef_v1`**
(`--policy-grad-coef 0` = arm F's pure-distill phase), **v101 `gen3_capacity_telemetry_v1`** (four
diagnostics that fold nothing into `loss`, recorded anyway because a diagnostic with unrecoverable
provenance is a number nobody can interpret later). **v100 `gen3_cf_coef_provenance_v1` — the
counterfactual COEFFICIENT family stops evaporating on resume:** the ten training-only cf knobs
(`cf_records`, `cf_records_keep`, `cf_winprob_coef`, `cf_head_only`, `cf_label_lag_steps`,
`cf_label_likelihood`, `cf_evidential_coef`, `cf_evidential_reg`, `cf_twin_coef`, `cf_shadow_coef`)
leave the `--opd-coef` genre for the `td_aux_coef` one — RECORDED for provenance and
`_resolve`-inherited on a flagless resume, never gated. The failure it closes is silent by
construction: an R1 arm resumed without re-typing `--cf-winprob-coef` kept training and simply
stopped applying the term it existed to measure, and because the three STRUCTURAL cf flags were
ALREADY recorded and gated, a resume could keep the head and drop the coefficient driving it. The
same pass found the enabling defect: `_resolve` fires only on `None`, and **five live cli-tier flags
had a `_resolve` line beside a non-None argparse default**, so the line was dead while the presence
test that checks for it passed — `cf_evidential` / `cf_twin_heads` / `cf_shadow_critic`, plus
`value_threat_inject` (ON in production) and `opp_intent_coef` (which `opp_intent` is DERIVED from),
the last two of which would have made a flagless resume of PRODUCTION FATAL at `check_compatible`.
`flag_registry_test.test_cli_flags_argparse_default_is_none` is the new gate for the REACHABILITY
half of that contract — the fourth vacuity class, caught by asserting on the built parser rather
than the source text. No `ARCH_SIGNATURE` bump, no floor change, no registry rows (the registry
declares extractor toggles; none of these builds a module). Prior: **v99 `gen3_cf_twin_heads_v1`**
(the twin win-prob heads + the passive shadow critic), **v98 `gen3_cf_evidential_head_v1`** (the
evidential Beta readout), **v97 `gen3_intent_label_bot_weight_v1`** (the bot-opponent α/β label
weight). **v96 `gen3_critic_route_wave_v1` — THE CRITIC-ROUTE DELETION WAVE:** seven audited-dead
critic routes deleted in one pass, and with them the ENTIRE post-assembler vf tail, so
**`vf_combined` IS `value_pooled`** — the same tensor `--value-from-dist`'s critic reads, which
makes the v89/M2 orphaned-branch class *unrepresentable* rather than merely fixed. Deleted on the
gen-14 battery (`measurements/gen14_route_audit_12391.json`, n=12,391): the v61
`MultiSeedValueReadout` + `seed_diagnostics` + the `value_seeds/*` TB contract (dV **0.0000
bit-exact**, twice) · the `hidden_opp_belief` **VF half only** (0.0000, while its **PI half flips
39.6% of argmaxes and KEEPS** — the per-head split the ledger keeps as a near-miss) · the
`non_matchup_rest` **VF concat** (0.0000; C1 measured the content substituting through the global
token; the pi concat KEEPS) · `value_intent` (0.156) · `intent_threshold`'s p_KO **vf route**
(0.155/0.136 — the POLICY move cell KEEPS) · `intent_value_reduce` (0.3176 at 2×) · `value_clock`
(0.2169 at 2×), all against a 0.39 bar. SURVIVING: `value_entity_pool` (dV **5.490 = 97% of
all_off**) and `value_threat_inject` (1.0686, deadline discharged). **Measured on the production
config: −540,786 params (−20.7%), `value_projection` 1177→128, policy logits and critic value
BYTE-IDENTICAL** with the surviving weights carried across. `value_intent`'s **re-entry condition
survives its deletion** (any α/β-critic proposal passes the C4 offline gate FIRST — ledger C6). The
signature bump is not optional: nothing in `model_config.json` records the seed readout, so it is
the only gate that can reject a pre-v96 checkpoint with a diagnosis ⇒ **gen-16 is fresh weights**,
gen-15 the eval reference. Also: `edge_ablation_audit`'s `concat` arm DELETED — it had been
measuring the seed readout under the name of a block dead since v61 (*an arm that outlives its
subject re-points, it does not go quiet*); `concat_cells` KEEPS as a live tripwire (KL 0.5682 /
flips 0.3105). Prior: **v95 `gen3_conditional_threat_v1` + `gen3_pair_value_route_v1` +
`gen3_status_economy_v1` (substrate Phase C):** OA1's 4 defensive-pivot coords on the switch cell;
PV as zero-init TOKEN CONTENT on the value pool's local copy (enabling owes the C4 offline gate —
the condition is verbatim in the registry/CLI/docs); undo_turns becomes min-over-paths with Natural
Cure (1.0) and the alive-gated bench-cleric path (2.0). **v94 `gen3_pair_outcome_switch_v1` +
`gen3_switch_branch_v1` (Phase B):** the α-reduced 14-coordinate row at every defender into the
SWITCH cell (first widening, 15→30) + spin_denied; OA2's β-weighted switch-branch expectations,
Rapid Spin spinblock (the documented Pursuit mirror), Protect's attack-mass conditioning;
switch_branch REQUIRES intent with NO fallback (a fallback would assert 'they never switch'). **v93
`gen3_pair_outcome_v1` (Phase A):** the 14-coordinate unified outcome vector (damage +
per-status-identity land probabilities + rule-derived neutralization + tempo_cost), ONE shared α
(Contract W — channel/defender axes are shape errors), R1 belief-mean fallback, reduced row to the
MOVE cells. All substrate flags OPT-IN and OFF in production; the enablement target is the mechanics
generation, gated by exploiter A/Bs (`designs/research_state/README.md` → Programme sequencing).
**v92 `--td-aux-coef`:** the TD-consistency auxiliary (C5 rung 2's instrument, training-coefficient
class). **v91 `gen3_event_semantics_v1`:** the event-window magnitude GIGO fix (dead `[from]` guard
— residuals summed into MOVE magnitude since v81), the `from_clause` key-class fix, faint_cause_id +
item_transition columns (obs 2437 → 2501), the Damp cant closure ([of]-keyed re-attribution;
archive-vs-live cant vocabulary split), MIGRATION_FLOOR → 91. Prior: **v90 `gen3_frame_deletion_v1`:
the TurnDelta lag frames are DELETED** (obs 3529 → 2437; trunk sequence 20 → 13 tokens — every edge
family indexes the GLOBAL seat as `2·TEAM_SIZE` and every extra seat as `_total_tokens + k`, so that
count is load-bearing). The H-B event window is now the LAST block ⇒ `total_dim == base_dim` and the
encoder's output IS the observation. `TurnDelta` itself SURVIVES (reward manager, reward tracker,
α/β labels) — only its obs encoding died. **The audit that licensed it could not see what it cost:**
dV measures DEPENDENCE, not per-fact COVERAGE, so a per-fact audit found `cant_reason` with no home
and CLOSED it (`EVENT_T_CANT` + column 19 `cant_id`, `EVENT_TOKEN_DIM` 19 → 20) and three more that
ship OPEN. Also: the role encoder's move-validity RE-SOURCED from the prev-turn sorted-by-id
`move_mask` to the current-decision request-order `our_active_req_move_legal` (stale AND misindexed
— the op had abandoned it and left this consumer behind). Raising the floor to 90 made every v77–v89
migration branch unreachable; their tests assert the refusal, and the dead branches are now DELETED
(follow-up discharged — with the floor at 96 the whole v77–v95 run is gone; `_migrate_config` keeps
only the version-INDEPENDENT sanitizers plus the genuinely post-floor v97 branch, and each deleted
branch's story is preserved verbatim in that file's PRE-FLOOR MIGRATION HISTORY comment). Prior: v89
`gen3_value_pooled_routes_v1`. **v89 `gen3_value_pooled_routes_v1`: the value routes finally reach
the critic** — `--value-from-dist`'s dist-head critic reads `value_pooled` ONLY, so the
post-assembler vf-tail concat was structurally disconnected: gen-12 proof,
`value_entity_pool.out_proj` and `intent_value_reduce.proj` bit-exact ZERO after 25M steps while
`value_threat_proj` (the one value_pooled route) trained to 0.117 — v74 and v80 were dead for TWO
generations, and every endofrun route-audit arm measuring a vf-tail route measured a dead limb. All
five routes (intent_value_reduce v74 / value_entity_pool v80+82 / intent_threshold-vf v84 /
value_clock + value_intent v87) now INJECT additively into `value_pooled` through zero-init D_MODEL
projections via ONE registry seam (`_value_pooled_routes`); `vf_parts[0] is value_pooled` so the
same wiring serves the scalar critic. Width-neutral ⇒ the ede5a88 discovery-sizing class is
unrepresentable; the per-route width constants deleted. NEW GUARD `value_route_gradient_test.py`:
one backward from each critic parameterization must reach every registered route's projection — the
test that would have caught this two generations ago. Migration: <v89 with any route ON is REFUSED
(shapes no longer exist); production sha 3cab191a→694c1652. **v88 `gen3_dead_flag_purge_v1`: three
dead flags DELETED OUTRIGHT, plus the whole pubval subsystem** (`value_active_readout` and
`damage_matrices_outgoing_all` — both v78 config_only demotions frozen OFF, never enabled in a
gen-8+ run — lose their fields, gates, and forwards; the OAX flat block goes with its flag while the
`_outgoing_attacker_matrix` KERNEL survives as `d2`'s engine; `pubval_mode`/`pubval_coef` +
`agents.training.pubval`, `pubval_calibration`, `PubValHead`, `_pubval_loss`, the parity fuzz and
`data/gen3_pubval.json` are deleted — measured NULL, never ON in production. The migration refuses a
recorded-ON value (the v75 rule: parameters the surviving code cannot rebuild ⇒ re-read from the
checkpoint's git_hash) and pops OFF silently; production sha unchanged (3cab191a). Same pass, no
version bump: `gen3_op_stashes_v1` — the op's 10 `last_*` attrs become ONE typed `OpStashes`
dataclass reset as a unit at forward entry (a stale cross-batch read is now unrepresentable), with
read-only `last_*` properties preserving every consumer; plus `PointerInputs`/`ThresholdProbs`
NamedTuples.) **v87 `gen3_value_direct_routes_v1`: two direct critic routes** (`--value-clock` — the
v67 deadline clock's raw scalars, the route the fix was validated for; `--value-intent` — the
published α/β posteriors as DISTRIBUTIONS, the ordering block dissolved by the post-assembler tail).
**v86 `gen3_op_lean_forward_v1`: op_tensors step 3 + believed lean physics** (`--op-drop-renders` —
the flat block's three render regions leave the forward, bit-identical at init since they had no
consumer; `--op-believed-lean` — the lean d3 physics price the believed spread, the B-spread fix at
the last de-timid site; plus `gen3_op_dead_kernel_cleanup_v1`: discrete_incoming/outgoing + two more
test-only orphans DELETED). **v85 `gen3_intent_conditional_v1`: the remaining α-conditioned mechanic
cells** (`--intent-conditional`, opt-in — Counter/Mirror Coat's category sums, flinch's missing
(1−α_SWITCH) term, Explosion's execute/into-switch facts, Pursuit's ×2 doubling trigger — CORRECTED
vs the design doc: the port-verified rule strikes the DEPARTING mon, no β; G2 usage baseline
MEASURED over 61,865 gen-12 decisions — Endure 0.0%, Sub 0.9%, Counter 5.6%). **v84
`gen3_intent_threshold_v1`: the α-weighted THRESHOLD operator** (`--intent-threshold`, opt-in —
`p_thresh(τ,⋛) = Σ_k α_k·1[damage(k,me) ⋛ τ]`: Focus Punch / Substitute / Endure / Destiny Bond /
Endeavor through the pointer MOVE cell in ONE contraction over the op's existing pair cells, plus
`p_KO` — the calibrated am-I-about-to-die — to the CRITIC, the ledger-H1 payoff; both projections
zero-init, one-graph-compile-gated; enabling waits on gen-12's intent_move_cell audit = the G3
verdict). **v83 `gen3_item_belief_v1`: the hidden ITEM as a belief** (`--item-belief`, opt-in — an
`ItemBelief` T0 head, Smogon per-species item-usage prior ⊕ zero-init trunk delta, cold-start
posterior == prior exactly; the op's Choice-Band tail consumes P(CB) from the publication at the
UNREVEALED branch instead of the static `SPECIES_CB_PRIOR` scalar — within 0.6% of it at init, so
enabling is ~behavior-preserving; supervised as the BeliefBank's SEVENTH row via
`--item-belief-coef`, labels `item_label`/`item_mask` from agent2's team). Same era:
`POLICY_ACTIVATION_FN` pinned (`gen3_policy_activation_pin_v1`) and per-edge-family liveness metrics
(`edge/<fam>_{weight,grad}_norm`). **v82 `gen3_unified_value_readout_v2`: the entity pool's COMPLETE
row set** (`--value-entity-pool-full`, opt-in — +the refined global token and the hidden-opp belief
queries; its own field so gen-12's v80-shape checkpoint keeps loading; with it every condemnable vf
route has ONE successor). Also `gen3_event_ref_edges_v1` (the `r` H-C family), `gen3_belief_bank_v1`
(all six supervised belief losses = declarative rows, three sites, byte-identical), the
`intent_reduce`/`event_seats`/`nmr` audit arms, and the H-tier compile gates. **v81
`gen3_event_window_v1`: Tier H-B BUILT** (obs 2921 → 3529 — the 32×19 typed event-record window
closes base; `EventWindowTracker` fold, seq-idempotent; the `--history-events` EVENT-SEAT consumer
is opt-in, TOKEN_TYPE_HISTORY, appended after E5 — position-stable; goldens strengthened to pin ALL
tracker-fed blocks, which they had silently asserted as zeros since H-A). **v80
`gen3_unified_value_readout_v1`: the Stage-3 T3-DELIVER critic contract BUILT**
(`--value-entity-pool`, opt-in zero-init vf-only UnifiedValueReadout — ONE attention pool over the
12 team tokens + the op's incoming rows; the designed successor of the seed/threat vf routes, with
its own `entity_pool` arm in the critic_route_audit; OFF in production until the gen-11 audit
adjudicates). Same pass, no version bump: `gen3_smogon_cooccur_prior_v1` — the v69/v72 species
co-occurrence prior is re-sourced from the POOL to SMOGON teammates (owner rule: ALL priors
Smogon-derived; only the MODEL gets bias against the pool, via training experience) — the pool's
strongest pair (Cloyster→Aerodactyl +1.32) measured +0.23 on 2.5M ladder battles. **v79
`gen3_pair_history_v1`: Tier H-A of `design_history_entity.md` BUILT** (obs 2669 → 2921: last-action
fields on the active slots + the 180-dim pair-history block; the new obs-fed zero-init `h` edge
family, opt-in — the pair-history fuzz caught a fainted-active-resurrection resync bug AND a
pre-existing recency cross-episode reset leak before shipping; stamp-only migration, no signature
bump). **v78 `gen3_flag_surface_p1_v1`** is the flag-surface cleanup, phase 1:
**`agents/model/flag_registry.py`** becomes the single declaration of every extractor toggle and of
the five hand-synced surfaces each one needs (argparse · `_resolve` · `ARCH_ARG_KEYS` ·
`current_model_version` · the `ModelVersion` field) — two are now GENERATED from it and three
VALIDATED by `flag_registry_test.py`, which found three real name drifts on its first run
(`--damage-topk`, and `--damage-matrices` desugaring into two fields). It introduces the **TIER**
axis (`cli` / `config_only` / `constructor_only`): a settled toggle can lose its CLI entry and keep
the recorded field, the version gate and the constructor kwarg — three demoted
(`attend_unrevealed_opponents` frozen ON, `value_active_readout` / `damage_matrices_outgoing_all`
frozen OFF). Eight fields DELETED with their modules, both closed research lines: the **zarch
family** (the LUT arm moved the N=20 ceiling +0.024, CI [-0.016,+0.064]; count dominates
conditioning) and the **seed-pressure pair** (`seed_quantile` + `value_seed_vicreg_coef` — both cap
at ~1-D of k=4 from opposite directions). Production forward + `state_dict` VERIFIED byte-identical
(200 keys, same digest, max|Δ| 0.0), so no signature bump and MIGRATION_FLOOR stays 76. Also:
**`--use-bridge` now defaults to `rust`** (serverless training AND eval by default;
`--use-showdown-bridge` deleted) and the launcher's port injection inverts with it.
`designs/flag_registry.md` is the generated table. Recent history: v77 intent move cell (G3), v76
`gen3_ctx_dedup_v1` deletes the duplicated active-ctx head concat (both projections narrow by 64;
the ctx rides the E2 injection + global token) and adds the **migration floor**
(`MIGRATION_FLOOR`/`SIGNATURE_FIRST_VERSION`: pre-generation configs are refused with a diagnosis
instead of walking dead branches; `model_version.py` −400 lines). Same pass, no version bump
(byte-identical): **`gen3_op_tensors_views_v1`** — `OpTensors` named views become the op flat
block's ONE slicer (`tensors_from_block`; consumers no longer hold offsets), and the delivery
graph's stale dead-concat edges were replaced with the true post-v61 routes (seed readout / intent
reduce / hidden-opp pool). Earlier: v75 SimSiam latent belief deleted (~13% of the train step) +
--belief-grad-mode label_only, v74 intent consumed (vf concat), v72 T0 species prior, v70/71 tiered
pipeline (prefuse unconditional, refine loop deleted), v68 opp_intent α/β, v67 deadline clock (the
generation wall), v65 unconditional move legality, v64 value-threat-inject, v63 seed quantile, v62
seed VICReg, v61 no-concat, v60 entity re-home, v51 pointer-native head.
`designs/production_config.json` tracks the LIVE code during a signature-bump window (v90); it
re-mirrors the newest run once gen-14 writes its `model_config.json` — the window is detected by
`arch_tables_test`, not papered over.
