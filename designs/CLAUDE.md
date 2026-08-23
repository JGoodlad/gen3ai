# designs/ — Version Map

This file tells Claude which `ai_vN` folder is relevant when reading or writing design
docs. Read it whenever you're about to touch anything in `designs/`.

**It is a version map, not an architecture reference.** For what the model actually is right now —
obs layout, phase chain, per-head inputs, the `DamageOperator` block, the edge families, and which
production flags are `INERT` — read [`ARCHITECTURE.md`](ARCHITECTURE.md). For how each version
changed things, read [`CHANGELOG.md`](CHANGELOG.md) (history; do not quote it as current).

---

## Critical: Training run ≠ Code version

**These two are almost always at different versions simultaneously.** A training run lasts
weeks; code changes happen daily. When the user says "update the doc" or "record what we
built", figure out which version applies to *what was just implemented*, not to *what is
currently training*.

To orient yourself:

- `git log --oneline -10 -- designs/ src/` — which `ai_vN` folder was most recently
  touched by commits? That's the version the code changes belong to.
- `designs/ai_vN/todo.md` — the in-progress version's todo has the most recent `✓ DONE`
  entries and open items; the running training run's todo is mostly done.
- When in doubt, ask: "is this an implementation doc for new code, or a record of what
  a running experiment does?"

**Current state as of 2026-08-22:**

| What | Version | Notes |
|------|---------|-------|
| **Active training run** | **ai_v9 gen-17 + the E-arms** | `ai_v9_21_gen17_pfspoff_0820` is **PRODUCTION** (launched 2026-08-20, config **v97** `gen3_critic_route_wave_v1`) and is what `designs/production_config.json` mirrors — **the substrate cells are ON in its base** (`pair_outcome_cell` / `pair_outcome_switch` / `switch_branch_cell` / `conditional_threat_cell`; `pair_value_route` stays OFF, it owes the C4 offline gate). `ai_v9_22_E1` … `ai_v9_25_E4` are gate EXPERIMENTS forked off that base, **not** production — each is byte-identical to gen-17 on every shared config field, which is why mirroring gen-17 satisfies the drift gate against all of them. Historical: | **gen-14** | `ai_v9_16_gen14_framedel_0817` (launched 2026-08-17, config **v90** `gen3_frame_deletion_v1`, FRESH WEIGHTS — the signature bump forbids a warm start; pinned to `fe910ee`). **One behavioural change: the 7×159 TurnDelta lag frames and the 11-dim prev-turn action mask are DELETED** (obs 3529 → 2437), licensed by gen-13.5 §4 (`event_seats` dV 2.7714 vs `frames` 1.3015). Flag delta off gen-13: `--value-intent` DROPPED (dV 0.1560); `--intent-value-reduce` (0.3826) and `--value-clock` (0.3370) KEPT ON deliberately — they sit in the registered TIE-BREAK zone and must stay live to be re-audited at ≥2× sample. Its battery is `designs/research_state/gen14_endofrun_runbook.md` (pre-registered BEFORE launch), and its open reconciliation is [`ai_v9/design_frame_deletion_coverage_gaps.md`](ai_v9/design_frame_deletion_coverage_gaps.md). The C5 TD-aux control fork `ai_v9_16_c5fork_control_0817` (27.1M, complete) is the banked baseline for the λ arms, which feed **gen-15**, not gen-14 — the frame deletion rides alone. Predecessor gen-13 `ai_v9_15_gen13_hb_events_stack_0817` (25.07M, config v89). Earlier: gen-12 `ai_v9_14_gen12_h_entitypool_shaping_0816` (launched 2026-08-16, config **v80** at save time; a pre-floor config — v90 raised `MIGRATION_FLOOR` to 90, so it no longer migrates and is read from its own git_hash). Four changes off the gen-11 base: the `h` pair-history edge family ON, `--value-entity-pool` ON (the Stage-3 critic pool trains live), `--intent-value-reduce` ON, and the win-prob shaping REVERTED to `--win-prob-coef 0.05` (the gen-11 verdict: ELO tied but the species belief regressed under the heavier shaping). Its end-of-run battery is `python -m main.endofrun models/ai_v9_14_gen12_h_entitypool_shaping_0816 --ref models/ai_v9_13_gen11_labelonly_winprob_0815`. Predecessor gen-11 `ai_v9_13_gen11_labelonly_winprob_0815` (launched 2026-08-15, config v77, pinned to its own commit — NOT `--sync-to-main`); its two changes off the gen-10 base: `--belief-grad-mode label_only` (the belief heads are trained by their SUPERVISED LABELS ALONE — no policy/value gradient reaches them; their trunk read stays live so the label loss still shapes the trunk) and `--win-prob-mode shaping --win-prob-coef 0.05` (the win-probability side readout now also shapes the trunk). Predecessors: gen-10 `ai_v9_12_gen10_t0prior_0814` (the T0-species-prior arm) and `ai_v9_11_gen10_intentfull_compiled_0814` (its A/B partner without the prior). ⚠️ Both were launched with a FULL explicit flag dump, so their `original_command` passes flags v78 deleted (`--zarch-*`, `--seed-quantile-coef`, `--value-seed-vicreg-coef`, `--film-grad-accum-steps`) — a `--sync-to-main` resume of either needs those stripped from the command first; a normal (git-hash-pinned) resume is unaffected. Earlier: gen-9 `ai_v9_10_gen9_intent_distcritic_0813` (the intent + distributional-critic arm), gen-8 `ai_v9_09_gen8_beliefs_threat_inject_0811` (beliefs + threat-inject; live sparse eval 2087±31 @26M, offline tail-4 ladder read it BELOW gen-4/5 — sparse-vs-dense unresolved), gen-7 (seed-quantile arm — quantiles learned, rank 1.157/4 ⇒ seed line closed), gen-6 (seed-VICReg arm — every term satisfied, rank ~1 ⇒ repulsion refuted), gen-5 `ai_v9_06_gen5_no_concat_0809` (25M, concat deletion at ELO parity with gen-4's 2096), gen-4 `run_20260808_212910` (v60 entity re-home; its stratified audits justified the concat deletion), gen-3 `run_20260807_135637_gen3` (40M, 15 edge families, ELO 2094@32M), gen-2, gen-1. The old ai_v8 lineage sits behind the ai_v9 signature wall. |
| **Code on main** | **ai_v9 (v100)** | `MODEL_CONFIG_VERSION` **100**, `ARCH_SIGNATURE` **`gen3_critic_route_wave_v1`**, `MIGRATION_FLOOR` **96**. **v100 `gen3_cf_coef_provenance_v1` — the counterfactual COEFFICIENT family stops evaporating on resume:** the ten training-only cf knobs (`cf_records`, `cf_records_keep`, `cf_winprob_coef`, `cf_head_only`, `cf_label_lag_steps`, `cf_label_likelihood`, `cf_evidential_coef`, `cf_evidential_reg`, `cf_twin_coef`, `cf_shadow_coef`) leave the `--opd-coef` genre for the `td_aux_coef` one — RECORDED for provenance and `_resolve`-inherited on a flagless resume, never gated. The failure it closes is silent by construction: an R1 arm resumed without re-typing `--cf-winprob-coef` kept training and simply stopped applying the term it existed to measure, and because the three STRUCTURAL cf flags were ALREADY recorded and gated, a resume could keep the head and drop the coefficient driving it. The same pass found the enabling defect: `_resolve` fires only on `None`, and **five live cli-tier flags had a `_resolve` line beside a non-None argparse default**, so the line was dead while the presence test that checks for it passed — `cf_evidential` / `cf_twin_heads` / `cf_shadow_critic`, plus `value_threat_inject` (ON in production) and `opp_intent_coef` (which `opp_intent` is DERIVED from), the last two of which would have made a flagless resume of PRODUCTION FATAL at `check_compatible`. `flag_registry_test.test_cli_flags_argparse_default_is_none` is the new gate for the REACHABILITY half of that contract — the fourth vacuity class, caught by asserting on the built parser rather than the source text. No `ARCH_SIGNATURE` bump, no floor change, no registry rows (the registry declares extractor toggles; none of these builds a module). Prior: **v99 `gen3_cf_twin_heads_v1`** (the twin win-prob heads + the passive shadow critic), **v98 `gen3_cf_evidential_head_v1`** (the evidential Beta readout), **v97 `gen3_intent_label_bot_weight_v1`** (the bot-opponent α/β label weight). **v96 `gen3_critic_route_wave_v1` — THE CRITIC-ROUTE DELETION WAVE:** seven audited-dead critic routes deleted in one pass, and with them the ENTIRE post-assembler vf tail, so **`vf_combined` IS `value_pooled`** — the same tensor `--value-from-dist`'s critic reads, which makes the v89/M2 orphaned-branch class *unrepresentable* rather than merely fixed. Deleted on the gen-14 battery (`measurements/gen14_route_audit_12391.json`, n=12,391): the v61 `MultiSeedValueReadout` + `seed_diagnostics` + the `value_seeds/*` TB contract (dV **0.0000 bit-exact**, twice) · the `hidden_opp_belief` **VF half only** (0.0000, while its **PI half flips 39.6% of argmaxes and KEEPS** — the per-head split the ledger keeps as a near-miss) · the `non_matchup_rest` **VF concat** (0.0000; C1 measured the content substituting through the global token; the pi concat KEEPS) · `value_intent` (0.156) · `intent_threshold`'s p_KO **vf route** (0.155/0.136 — the POLICY move cell KEEPS) · `intent_value_reduce` (0.3176 at 2×) · `value_clock` (0.2169 at 2×), all against a 0.39 bar. SURVIVING: `value_entity_pool` (dV **5.490 = 97% of all_off**) and `value_threat_inject` (1.0686, deadline discharged). **Measured on the production config: −540,786 params (−20.7%), `value_projection` 1177→128, policy logits and critic value BYTE-IDENTICAL** with the surviving weights carried across. `value_intent`'s **re-entry condition survives its deletion** (any α/β-critic proposal passes the C4 offline gate FIRST — ledger C6). The signature bump is not optional: nothing in `model_config.json` records the seed readout, so it is the only gate that can reject a pre-v96 checkpoint with a diagnosis ⇒ **gen-16 is fresh weights**, gen-15 the eval reference. Also: `edge_ablation_audit`'s `concat` arm DELETED — it had been measuring the seed readout under the name of a block dead since v61 (*an arm that outlives its subject re-points, it does not go quiet*); `concat_cells` KEEPS as a live tripwire (KL 0.5682 / flips 0.3105). Prior: **v95 `gen3_conditional_threat_v1` + `gen3_pair_value_route_v1` + `gen3_status_economy_v1` (substrate Phase C):** OA1's 4 defensive-pivot coords on the switch cell; PV as zero-init TOKEN CONTENT on the value pool's local copy (enabling owes the C4 offline gate — the condition is verbatim in the registry/CLI/docs); undo_turns becomes min-over-paths with Natural Cure (1.0) and the alive-gated bench-cleric path (2.0). **v94 `gen3_pair_outcome_switch_v1` + `gen3_switch_branch_v1` (Phase B):** the α-reduced 14-coordinate row at every defender into the SWITCH cell (first widening, 15→30) + spin_denied; OA2's β-weighted switch-branch expectations, Rapid Spin spinblock (the documented Pursuit mirror), Protect's attack-mass conditioning; switch_branch REQUIRES intent with NO fallback (a fallback would assert 'they never switch'). **v93 `gen3_pair_outcome_v1` (Phase A):** the 14-coordinate unified outcome vector (damage + per-status-identity land probabilities + rule-derived neutralization + tempo_cost), ONE shared α (Contract W — channel/defender axes are shape errors), R1 belief-mean fallback, reduced row to the MOVE cells. All substrate flags OPT-IN and OFF in production; the enablement target is the mechanics generation, gated by exploiter A/Bs (`designs/research_state/README.md` → Programme sequencing). **v92 `--td-aux-coef`:** the TD-consistency auxiliary (C5 rung 2's instrument, training-coefficient class). **v91 `gen3_event_semantics_v1`:** the event-window magnitude GIGO fix (dead `[from]` guard — residuals summed into MOVE magnitude since v81), the `from_clause` key-class fix, faint_cause_id + item_transition columns (obs 2437 → 2501), the Damp cant closure ([of]-keyed re-attribution; archive-vs-live cant vocabulary split), MIGRATION_FLOOR → 91. Prior: **v90 `gen3_frame_deletion_v1`: the TurnDelta lag frames are DELETED** (obs 3529 → 2437; trunk sequence 20 → 13 tokens — every edge family indexes the GLOBAL seat as `2·TEAM_SIZE` and every extra seat as `_total_tokens + k`, so that count is load-bearing). The H-B event window is now the LAST block ⇒ `total_dim == base_dim` and the encoder's output IS the observation. `TurnDelta` itself SURVIVES (reward manager, reward tracker, α/β labels) — only its obs encoding died. **The audit that licensed it could not see what it cost:** dV measures DEPENDENCE, not per-fact COVERAGE, so a per-fact audit found `cant_reason` with no home and CLOSED it (`EVENT_T_CANT` + column 19 `cant_id`, `EVENT_TOKEN_DIM` 19 → 20) and three more that ship OPEN. Also: the role encoder's move-validity RE-SOURCED from the prev-turn sorted-by-id `move_mask` to the current-decision request-order `our_active_req_move_legal` (stale AND misindexed — the op had abandoned it and left this consumer behind). Raising the floor to 90 made every v77–v89 migration branch unreachable; their tests assert the refusal, and the dead branches are now DELETED (follow-up discharged — with the floor at 96 the whole v77–v95 run is gone; `_migrate_config` keeps only the version-INDEPENDENT sanitizers plus the genuinely post-floor v97 branch, and each deleted branch's story is preserved verbatim in that file's PRE-FLOOR MIGRATION HISTORY comment). Prior: v89 `gen3_value_pooled_routes_v1`. **v89 `gen3_value_pooled_routes_v1`: the value routes finally reach the critic** — `--value-from-dist`'s dist-head critic reads `value_pooled` ONLY, so the post-assembler vf-tail concat was structurally disconnected: gen-12 proof, `value_entity_pool.out_proj` and `intent_value_reduce.proj` bit-exact ZERO after 25M steps while `value_threat_proj` (the one value_pooled route) trained to 0.117 — v74 and v80 were dead for TWO generations, and every endofrun route-audit arm measuring a vf-tail route measured a dead limb. All five routes (intent_value_reduce v74 / value_entity_pool v80+82 / intent_threshold-vf v84 / value_clock + value_intent v87) now INJECT additively into `value_pooled` through zero-init D_MODEL projections via ONE registry seam (`_value_pooled_routes`); `vf_parts[0] is value_pooled` so the same wiring serves the scalar critic. Width-neutral ⇒ the ede5a88 discovery-sizing class is unrepresentable; the per-route width constants deleted. NEW GUARD `value_route_gradient_test.py`: one backward from each critic parameterization must reach every registered route's projection — the test that would have caught this two generations ago. Migration: <v89 with any route ON is REFUSED (shapes no longer exist); production sha 3cab191a→694c1652. **v88 `gen3_dead_flag_purge_v1`: three dead flags DELETED OUTRIGHT, plus the whole pubval subsystem** (`value_active_readout` and `damage_matrices_outgoing_all` — both v78 config_only demotions frozen OFF, never enabled in a gen-8+ run — lose their fields, gates, and forwards; the OAX flat block goes with its flag while the `_outgoing_attacker_matrix` KERNEL survives as `d2`'s engine; `pubval_mode`/`pubval_coef` + `agents.training.pubval`, `pubval_calibration`, `PubValHead`, `_pubval_loss`, the parity fuzz and `data/gen3_pubval.json` are deleted — measured NULL, never ON in production. The migration refuses a recorded-ON value (the v75 rule: parameters the surviving code cannot rebuild ⇒ re-read from the checkpoint's git_hash) and pops OFF silently; production sha unchanged (3cab191a). Same pass, no version bump: `gen3_op_stashes_v1` — the op's 10 `last_*` attrs become ONE typed `OpStashes` dataclass reset as a unit at forward entry (a stale cross-batch read is now unrepresentable), with read-only `last_*` properties preserving every consumer; plus `PointerInputs`/`ThresholdProbs` NamedTuples.) **v87 `gen3_value_direct_routes_v1`: two direct critic routes** (`--value-clock` — the v67 deadline clock's raw scalars, the route the fix was validated for; `--value-intent` — the published α/β posteriors as DISTRIBUTIONS, the ordering block dissolved by the post-assembler tail). **v86 `gen3_op_lean_forward_v1`: op_tensors step 3 + believed lean physics** (`--op-drop-renders` — the flat block's three render regions leave the forward, bit-identical at init since they had no consumer; `--op-believed-lean` — the lean d3 physics price the believed spread, the B-spread fix at the last de-timid site; plus `gen3_op_dead_kernel_cleanup_v1`: discrete_incoming/outgoing + two more test-only orphans DELETED). **v85 `gen3_intent_conditional_v1`: the remaining α-conditioned mechanic cells** (`--intent-conditional`, opt-in — Counter/Mirror Coat's category sums, flinch's missing (1−α_SWITCH) term, Explosion's execute/into-switch facts, Pursuit's ×2 doubling trigger — CORRECTED vs the design doc: the port-verified rule strikes the DEPARTING mon, no β; G2 usage baseline MEASURED over 61,865 gen-12 decisions — Endure 0.0%, Sub 0.9%, Counter 5.6%). **v84 `gen3_intent_threshold_v1`: the α-weighted THRESHOLD operator** (`--intent-threshold`, opt-in — `p_thresh(τ,⋛) = Σ_k α_k·1[damage(k,me) ⋛ τ]`: Focus Punch / Substitute / Endure / Destiny Bond / Endeavor through the pointer MOVE cell in ONE contraction over the op's existing pair cells, plus `p_KO` — the calibrated am-I-about-to-die — to the CRITIC, the ledger-H1 payoff; both projections zero-init, one-graph-compile-gated; enabling waits on gen-12's intent_move_cell audit = the G3 verdict). **v83 `gen3_item_belief_v1`: the hidden ITEM as a belief** (`--item-belief`, opt-in — an `ItemBelief` T0 head, Smogon per-species item-usage prior ⊕ zero-init trunk delta, cold-start posterior == prior exactly; the op's Choice-Band tail consumes P(CB) from the publication at the UNREVEALED branch instead of the static `SPECIES_CB_PRIOR` scalar — within 0.6% of it at init, so enabling is ~behavior-preserving; supervised as the BeliefBank's SEVENTH row via `--item-belief-coef`, labels `item_label`/`item_mask` from agent2's team). Same era: `POLICY_ACTIVATION_FN` pinned (`gen3_policy_activation_pin_v1`) and per-edge-family liveness metrics (`edge/<fam>_{weight,grad}_norm`). **v82 `gen3_unified_value_readout_v2`: the entity pool's COMPLETE row set** (`--value-entity-pool-full`, opt-in — +the refined global token and the hidden-opp belief queries; its own field so gen-12's v80-shape checkpoint keeps loading; with it every condemnable vf route has ONE successor). Also `gen3_event_ref_edges_v1` (the `r` H-C family), `gen3_belief_bank_v1` (all six supervised belief losses = declarative rows, three sites, byte-identical), the `intent_reduce`/`event_seats`/`nmr` audit arms, and the H-tier compile gates. **v81 `gen3_event_window_v1`: Tier H-B BUILT** (obs 2921 → 3529 — the 32×19 typed event-record window closes base; `EventWindowTracker` fold, seq-idempotent; the `--history-events` EVENT-SEAT consumer is opt-in, TOKEN_TYPE_HISTORY, appended after E5 — position-stable; goldens strengthened to pin ALL tracker-fed blocks, which they had silently asserted as zeros since H-A). **v80 `gen3_unified_value_readout_v1`: the Stage-3 T3-DELIVER critic contract BUILT** (`--value-entity-pool`, opt-in zero-init vf-only UnifiedValueReadout — ONE attention pool over the 12 team tokens + the op's incoming rows; the designed successor of the seed/threat vf routes, with its own `entity_pool` arm in the critic_route_audit; OFF in production until the gen-11 audit adjudicates). Same pass, no version bump: `gen3_smogon_cooccur_prior_v1` — the v69/v72 species co-occurrence prior is re-sourced from the POOL to SMOGON teammates (owner rule: ALL priors Smogon-derived; only the MODEL gets bias against the pool, via training experience) — the pool's strongest pair (Cloyster→Aerodactyl +1.32) measured +0.23 on 2.5M ladder battles. **v79 `gen3_pair_history_v1`: Tier H-A of `design_history_entity.md` BUILT** (obs 2669 → 2921: last-action fields on the active slots + the 180-dim pair-history block; the new obs-fed zero-init `h` edge family, opt-in — the pair-history fuzz caught a fainted-active-resurrection resync bug AND a pre-existing recency cross-episode reset leak before shipping; stamp-only migration, no signature bump). **v78 `gen3_flag_surface_p1_v1`** is the flag-surface cleanup, phase 1: **`agents/model/flag_registry.py`** becomes the single declaration of every extractor toggle and of the five hand-synced surfaces each one needs (argparse · `_resolve` · `ARCH_ARG_KEYS` · `current_model_version` · the `ModelVersion` field) — two are now GENERATED from it and three VALIDATED by `flag_registry_test.py`, which found three real name drifts on its first run (`--damage-topk`, and `--damage-matrices` desugaring into two fields). It introduces the **TIER** axis (`cli` / `config_only` / `constructor_only`): a settled toggle can lose its CLI entry and keep the recorded field, the version gate and the constructor kwarg — three demoted (`attend_unrevealed_opponents` frozen ON, `value_active_readout` / `damage_matrices_outgoing_all` frozen OFF). Eight fields DELETED with their modules, both closed research lines: the **zarch family** (the LUT arm moved the N=20 ceiling +0.024, CI [-0.016,+0.064]; count dominates conditioning) and the **seed-pressure pair** (`seed_quantile` + `value_seed_vicreg_coef` — both cap at ~1-D of k=4 from opposite directions). Production forward + `state_dict` VERIFIED byte-identical (200 keys, same digest, max|Δ| 0.0), so no signature bump and MIGRATION_FLOOR stays 76. Also: **`--use-bridge` now defaults to `rust`** (serverless training AND eval by default; `--use-showdown-bridge` deleted) and the launcher's port injection inverts with it. `designs/flag_registry.md` is the generated table. Recent history: v77 intent move cell (G3), v76 `gen3_ctx_dedup_v1` deletes the duplicated active-ctx head concat (both projections narrow by 64; the ctx rides the E2 injection + global token) and adds the **migration floor** (`MIGRATION_FLOOR`/`SIGNATURE_FIRST_VERSION`: pre-generation configs are refused with a diagnosis instead of walking dead branches; `model_version.py` −400 lines). Same pass, no version bump (byte-identical): **`gen3_op_tensors_views_v1`** — `OpTensors` named views become the op flat block's ONE slicer (`tensors_from_block`; consumers no longer hold offsets), and the delivery graph's stale dead-concat edges were replaced with the true post-v61 routes (seed readout / intent reduce / hidden-opp pool). Earlier: v75 SimSiam latent belief deleted (~13% of the train step) + --belief-grad-mode label_only, v74 intent consumed (vf concat), v72 T0 species prior, v70/71 tiered pipeline (prefuse unconditional, refine loop deleted), v68 opp_intent α/β, v67 deadline clock (the generation wall), v65 unconditional move legality, v64 value-threat-inject, v63 seed quantile, v62 seed VICReg, v61 no-concat, v60 entity re-home, v51 pointer-native head. `designs/production_config.json` tracks the LIVE code during a signature-bump window (v90); it re-mirrors the newest run once gen-14 writes its `model_config.json` — the window is detected by `arch_tables_test`, not papered over. |
| **ai_v10** | **OPEN — nothing built** | The **exploiter-SCALING** chapter, opened 2026-08-16: why competence collapses between N=10 and N=20 teams when N=1..5 is trivial. One doc, [`ai_v10/design_exploiter_scaling.md`](ai_v10/design_exploiter_scaling.md) — the hypothesis (**no transferable team-scoped abstraction ⇒ sample cost linear in N**), the four competing accounts it must beat (H_rate / H_capacity / H_conflict / H_coverage), and a **pre-registered, unrun** test battery whose Tier 0 needs no extra GPU. Two NEW gen-12 measurements carry it: "it's just 6 1v1s" is **REFUTED** (bench→MOVE logit ratio **0.262**, n=4058 over 8 real exploiter teams) but the profile is FLAT, so the bench enters as a **SCALAR not as structure**; and team PACE class decodes from the raw obs at **0.456 on UNSEEN teams** while `pi_features` **0.211** and `value_pooled` **0.199** sit at chance (0.200) — **the abstraction is free in the input and the trunk discards it**. Where ai_v9 is the entity graph *inside* one battle, ai_v10 is *what transfers between teams*. The chapter also carries two owner-era operational/forward docs: [`design_flywheel_tick_tock.md`](ai_v10/design_flywheel_tick_tock.md) (the exploiter–generalist loop, decisions of record — needs refinement, not implementation-ready) and [`design_outcome_latent.md`](ai_v10/design_outcome_latent.md) (FORWARD, 2026-08-19: the per-action LEARNED outcome latent — route-3 delivery with learned content, the Spikes mechanism/horizon factorization, the G0→R3 ladder gated on behavioral deltas, and the richness-pressure menu ranked by this codebase's own body count — post-gen-16, unscheduled). A third forward doc, [`design_counterfactual_value_grounding.md`](ai_v10/design_counterfactual_value_grounding.md) (2026-08-22): the counterfactual label factory + the three reroll-based attacks on CRITIC BIAS (R1 tight-MC re-labels on visited states / R2 MC labels on counterfactual successors — the optimizer's-curse interruption and the bait cure claim / R3 k-step grounded targets), priced by the 2026-08-21 probe triad (162→28.4→~7.7 ms/label; opponent branches first; the prefix-sharing materializer is the one build item), gated G0–G4 on bias METERS not loss curves — the pre-registered new mechanism ledger C6 requires, attacking the critic's TARGETS while its closed delivery line stays closed. |
| **ai_v11** | **OPEN — nothing built** | The **human-ladder-replay** chapter, opened 2026-08-18: what we can learn from an EXTERNAL action distribution, and what survives the fact that spectator replays are **partial information**. One doc, [`ai_v11/design_human_replay_objectives.md`](ai_v11/design_human_replay_objectives.md) — the **OOD taxonomy** (our own team is fully known live and only partially recoverable from a replay; the request stream does not exist, so the mask is synthesised and systematically over-permissive) and a **pre-registered, unrun** four-rung objective ladder ordered by OOD-robustness: α/β on the human OPPONENT's actions (robust — that half of the obs is hidden live too) → outcome/value on human states → BC-regularization on the faithful subset (**gen-17 candidate**) → offline RL with team-completed acting sides. Its Phase-0 census is **RUN** (`tmp/replay_faithfulness_census.py`; 263,159 logs / 2.8 GB / 2026-05-18→2026-08-02): tier-A (6/6 bench, 4/4 moves) is **16.70%** of 30,146 ≥1500 decisions, own **item known on 3.93%** of own mons, own **spread is FABRICATED and flagged `spread_known=1`** (all-31 IVs / 0 EVs / neutral nature — a wrong value asserted as known, feeding `d1`/`d2`), the faithful stratum is **loss-enriched 1.29×**, α-label pairing is **92.04%**, and the human switch share reproduces model-free at **28.96%**. |
| **ai_v9** | **Stages 0–2 SHIPPED + Stage-3 half** | Roadmap: `design_generation_roadmap.md` (the operative staged plan, slice statuses current). **The op head-concat deletion is DONE (2026-08-09, `6aac795`) — it was the last of the stated goals and it landed on evidence, not on schedule:** gen-4's stratified end-of-run audits showed `FULL_CONCAT` net policy dependence **+0.00%**, all-edges-off flips (29.22%) **exceeding** the concat arm (22.70%) for the first time in the lineage, and `act_threat` still decodable from `vf` with the concat zeroed (r² 0.400 → 0.418), so the remaining `|dV|` 4.75 was trained reliance on an open window rather than structural necessity. Still open: C1b/C2/C3/C5 consequence edges, E9 history, and OpTensors steps 1–2 (typed views, recompute dedup) — deferred honestly to background work during gen-5, since the §9.1 evidence showed the removal was not waiting on them. **NEW forward design (not built): `design_conditional_opponent_cells.md`** — the magnitude rule for the entity world + the OA1 conditional threat cell (defensive pivot) and OA2 switch-branch move cell (punish the switch), plus PV pair-value attention (a critic route), the unrevealed-marginalisation prerequisite and pre-registered gates. **RESOLVED 2026-08-09 — read the two-route precondition below as history.** The 2026-08-08 amendment required OA1 (policy) **and** a critic route (PV *or* generalized token-content injection) to land before the concat could die, accepted only on **flips AND `|dV|`**. What actually happened: the **flips** half was met by training alone (all-edges-off 29.22% > concat 22.70% on gen-4), the policy side needed no OA1 at all (net concat dependence +0.00%), and the critic route that shipped was **neither** of the two candidates — it is `MultiSeedValueReadout`, readout **multiplicity** rather than width (P3 refuted width, never multiplicity). `|dV|` remained concat-led (4.62 vs 2.44) and was overridden on the conditional-coverage evidence above rather than waited out. **OA1/OA2 and PV therefore survive as forward designs on their own merits, no longer as preconditions for anything.** **OA1/OA2 are pointer CELLS, not edge families** — do not confuse them with the C1-C5 consequence edges. **NEW forward design (not built), 2026-08-12: `design_conditional_execution.md`** — the **OUTGOING** consumer of the `α`/`β` intent belief: a per-mechanic spec of every gen3 move whose value cannot be computed without knowing what the opponent will do. Same Contract-W contraction as the incoming side (`E[my move m] = Σ_k α_k · f(m,k)`, where `f` is a deterministic RULES lookup so `α` needs no change), but delivered to the pointer **MOVE cell** — a per-action absolute in a channel measured to WORK (`d1` 12.17% / `d2` 19.25%), unlike the defensive edges. **Its motivating evidence: the consequence families are ALREADY this idea, built with the wrong conditioning** — `c4` Protect carries `p_success` = the MECHANICAL consecutive-use decay odds and never asks "will they attack"; `x` Pursuit carries `pursuit_p` = P(they *carry* Pursuit), not P(they click it) × P(I switch). That is `α ≠ w` (presence vs usage) in a second, independent place, and it plausibly explains why every consequence family sits at the noise floor (c2 1.20% … c4 0.15%). **The structural find: FIVE mechanics are ONE operator** — `p_thresh(τ,⋛) = Σ_k α_k · 1[damage(k,me) ⋛ τ]` covers Focus Punch (τ=0,>), Substitute (25% maxhp,<), Endure (HP,≥), Destiny Bond (HP,≥ — same threshold, OPPOSITE valence) and Endeavor (HP,<); the `τ=HP` case is **`p_KO`**, the α-weighted P(I die this turn), which is nearly FREE (the op already computes per-move `pko` and `_chan_max` collapses it to a max — α turns that max into a calibrated probability) and is exactly the quantity ledger **H1**'s self-KO defect mis-values. Destiny Bond and Endeavor are the two mechanics whose value moves **OPPOSITE** to every damage feature the op computes (DB rises with P(they kill you); Endeavor rises as your HP falls). Taxonomy: class A this-turn / class B switch-contingent (`β`) / **class C long-horizon — Spikes is a switch RATE, not a one-ply conditional, and is DEFERRED because modelling it here would be confidently wrong**. Pool exposure over 773 team files: explosion 69.3%, spikes 45.7%, protect 38.2%, substitute 29.6%, focuspunch 26.1%, pursuit 21.0%, counter 8.9% — but §3 carries an explicit warning that **carriage frequency is a POOR prioritiser alone** (it under-weights decisive-but-rare mechanics; Destiny Bond is 0.8% and decides the games it appears in — the same blind spot that voided the `OUR_MOVE_OUTCOME` reading). Endure→Endeavor is a two-turn PLAN (out of scope) but **each leg is a one-ply conditional (in scope)** — that decomposition is what makes the pair tractable. Magic Coat's reflectable set is marked **UNVERIFIED** pending constructed-scenario oracle work. **G3 is the gate that prices the whole document: re-deliver ONE existing consequence family (`c2`, least-dead) through the move cell with α — if it stays at zero, the consequence line is dead and the other seven are not worth building.** Steps 1–7 need no training run. **NEW forward design (not built), 2026-08-11: `design_opponent_intent.md`** — the build for one sentence the model cannot express: *"they are likely to click **this**, so **this** is my answer."* Supplies the two things the pair-reduction operator was missing: a **distribution worth weighting by** (`α`, a SUPERVISED usage belief over their K believed move seats + `SWITCH`; `β` over their team slots, conditional on `SWITCH`) and an **outcome vector worth weighting** (one unified `pair_in` carrying damage AND status AND `neutralization` AND `tempo_cost` — today damage and status are computed in two functions with two reductions, and one `α` cannot weight two tensors). **The three-part framing is the doc's core claim:** a distribution + a rich outcome vector + the weighting done per-action before the logit — missing any one makes the other two useless, which is why **G1-FINAL's null was near-guaranteed** (it tested part 3 alone, on damage-only cells, with `w` substituted for `α`). Grounded in the gen-4 end-of-run edge audit: `d2` 19.25% / `d1` 12.17% (our offense) vs **`d3` 0.63%** (their believed threat, DOWN from 1.9% at gen-3 9.6M) — the entity system is overwhelmingly offensive because the anticipatory half is routed through edges, which carry a softmax-normalised RATIO and cannot deliver a per-action absolute. **Two owner reconciliations settled 2026-08-11:** (1) *both sides anticipate* — `α` may not depend on our REALIZED action but MAY depend on our POLICY, and since the policy is a function of the board, `α = f(board)` is already the right form; the forced change is that **`α`'s INPUT must include OUR outgoing physics** (`d1`/`d2` grids), and the fixed point is found by TRAINING (self-play), never solved at inference — reading our own policy logits would be level-3 but creates a forward-pass cycle, so it is deliberately not taken; (2) *belief-derived seats*, now governed by a **HARD OWNER CONSTRAINT (2026-08-11): the model must always pick among the belief's DISCRETE states and may never invent a move — interpretability is the reason.** So `α ∈ Δ^(K+1)` over named seats + `SWITCH`, no `UNKNOWN` slot and no learned property head (both proposed in earlier drafts and CUT — §4.6 keeps their causes of death). **ONE rule on both axes: "if we can't name it, we don't train on it"** — hard target when the belief holds it, **masked** otherwise, mask rate logged as a first-class diagnostic. (A property-similar soft-target scheme was drafted for the move axis and CUT: it yields a smeared object rather than the clean `P(seat | modeled)`, it injects the belief's non-random blind spots as a bias invisible in `α`'s accuracy, its similarity metric has no principled setting, and it was an unjustified asymmetry against `β`, which masks. Its motivating example also dissolved — under canonical-id matching a bare-`hiddenpower` seat MATCHES a used HP Ice.) Matching is by canonical id, never by index (seats permute per turn — the Hungarian precedent). **The division of labour that buys:** `α`/`β` own *which of the things we believe*, the belief head owns *whether we believe the right things* — two failure modes, two measurements, instead of one head absorbing the other's errors. **`β` answers "switch to WHOM"** — discrete over alive/non-active/revealed slots, masked in v1 when they bring an unrevealed mon (rate logged; **B1**, BUILT-but-never-run, is the named upgrade that turns that mask into a posterior soft-target). **`β` is also what makes the (bench × bench) offense grid actionable** — that grid alone is an unweighted outer product; with `β` it answers *"if I bring Skarmory and they pivot to Blissey, is Skarmory still doing anything?"*, so `d2`/`d5` are not independent cheap wins but the grid `β` needs. The RL loss is **stop-gradiented** out of both heads so a null is interpretable. **The constraint's real cost is a ceiling at belief quality, so §4.5 audits the WHOLE BELIEF STACK against the live config — seven legs, and EXACTLY ONE is supervised.** B-move (which moves they hold) **ON but UNSUPERVISED** (`move_belief_coef` `0.0`; `known_moves` already emitted + plumbed, BCE unconsumed — shaped only by the Smogon prior + RL gradient); B-hptype ✅ supervised 0.05, acc ≈0.91; B-spread ❌ OFF; B-team (B1) ❌ OFF (BUILT, never run); B-latent ❌ OFF; **B-item and B-ability are STATIC lookups that cannot improve with training** (`p_cb` = a species usage prior collapsing to 0/1 on reveal; abilities = Smogon per-species priors). **B-spread is a PHYSICS DEFECT, not a missing signal:** with it off the op prices every opponent's offense as `(2·base_atk + const) × 1.1` — 252 EV + boosting nature, uniformly, at **nine sites** (`damage_op.py:1727` et al) — so `pair_in` is computed against a fictional maximally-invested opponent, and the over-estimate scales with base stats so it distorts the RELATIVE threat ordering, not just the level. A better `α` over de-timid physics inherits the distortion ⇒ **B-spread is a correctness fix to component 1, not a third belief leg to stack.** Operationally they differ: `--move-belief-coef` is **training-only / resume-mutable**, `--spread-belief` is **STRUCTURAL / version-checked / FRESH-ONLY** (cannot join a running generation). **G2a runs FIRST and needs no head at all** (how often does the top-K hold what they clicked?); then G2b (does `α` beat `w`, `β` beat the alive-bench base rate?); G3b asserts the discrete constraint as a TEST. **G0–G7 need no training run.** Not this doc: physics mutation (Marvel Scale changing the whole matrix) is explicitly out of scope for a one-ply reduction. §9.1 records that the G1-FINAL SKYLINE is likely underpowered (2800 params on ~239 rows at L2 1e-3) and should not be read as "the grid is exhausted" until re-conditioned. **§7a (REVIEW, 2026-08-11) adds four notes:** (1) ⚠️ this is a **POLICY-side** design and the critic deficit is **separate** — the dense frozen-vs-frozen ladder reads gen-4 @24M **2080.6 (se 10.70)** vs gen-5 **2037.4 (se 10.10)**, and the sparse `eval/elo` at ±30 that reported "parity" could not resolve that. **§7a.1's amendment (2026-08-11) downgrades "cost ~44 Elo" to SUGGESTIVE, not established**, on three grounds read back from both runs' `snapshot_ladder/ladder.json`: the original "±11" was the **standard error, not the CI** (at 1.96·se the intervals are [2059.6, 2101.6] vs [2017.6, 2057.2] — disjoint by **2.4 Elo**, marginal); the **trajectories INTERLEAVE** (gen-5 is *ahead* at 14M and 16M, dead level at 20M); and **the entire 43-point gap is gen-4's final checkpoint jumping +30** while gen-5's stayed flat — a single-endpoint comparison across crossing trajectories cannot separate a real cost from a lucky last snapshot. Disambiguate by fitting the last 3–4 checkpoints, or laddering gen-4 @22M against gen-5 @24M. **The conclusion does NOT rest on this number**: the seed readout measuring ~1 effective direction under BOTH VICReg and quantile pressure is an independent, self-standing critic-side finding, so the scope gap is real on structural grounds and (2) remains the right response either way; (2) the **critic route that falls out of this design** — send the same `Σ_k α_k·pair_in[k,j,:]` row to the critic as **token content on our mon j's token**, pooled by `value_cls`: equivariant in BOTH axes (`α` invariant under permuting their moves since `g` is shared over `k`; the row rides mon `j`'s token; attention pooling is permutation-invariant), **no seeds** (which matters given seed collapse measured at ~1 effective direction under BOTH VICReg and quantile pressure), and testable BEFORE `α` exists by substituting `α := normalize(w)` (the shipped R1 rung) — separating the DELIVERY claim from the DISTRIBUTION claim; (3) an **alternative hypothesis for `d3` = 0.63%** the doc does not raise — a channel carrying DISTORTED content also reads low, and the de-timid defect corrupts exactly the relative threat ordering `d3` conveys, so **re-measure `d3` after the B-spread fix before concluding the channel was the problem**. **Refined 2026-08-11: the two are NOT mutually exclusive and the build order does NOT fork on them.** The channel argument is STRUCTURAL — an edge writes a ratio and cannot deliver a per-action absolute *however accurate the numbers being ratio-ed are*, and fixing de-timid does not move the `argmax_a` off the pointer logits. Both are almost certainly true at once, so the post-B-spread re-measurement buys a **magnitude, not a verdict**: it says how much of the 0.63% was content. Neither outcome removes the need for the per-action route; (4) the **coverage-risk fallback pre-registered**: `α ∈ Δ²` over {ATTACK, SWITCH} only — belief-free, zero mask rate, carries the largest single effect — ship it if G2a returns poor coverage. Plus schedule honesty: step 0b is fresh-only, so this is **gen-8 (foundation) + gen-9 (`α`)**, not one generation. **NEW forward design (not built), now scoped to RUN BESIDE GEN-5 and land at the gen-6 boundary: `design_pair_reduction.md`** — the deep spec of the one line `design_op_tensors.md` §3.2 sketches as `REDUCE(pair_in, over=MOVE_AXIS, how=…)`. Splits **contract** from **knob**: a weighted reducer must emit ONE distribution over the move axis per defender, shared across every channel, which kills the incoherence defect structurally (the flat/trunk block takes NINE independent maxima, so up to nine different opponent moves describe one defender). **§2.1 (added 2026-08-10, verified against the live tree) makes that understatement precise and WORSE on the path that decides:** the pointer **switch cell is 15 numbers — ten damage, `p_outspeed`, `provenance`, and the Choice-Band tail — with NO status coordinate in any currency at all.** In production `threat_status_refine` is `False`, so incoming status reaches the policy *only* as the `s3` edge family — an attention **bias**, i.e. a softmax-normalised RATIO. So "they'll click a status move, so bring the Natural Cure mon" is unrepresentable not because status was mis-reduced but because **the two quantities never meet in the same vector in the same units** — a CURRENCY failure one level below the reduction failure, which no reducer fixes. **This is the most likely reading of the G1 n=299 null** (no rung beats R0 beyond seed spread): the ladder was asked to improve an aggregation over a vector that never carried the quantity the decision turns on. §9a adds the admission test for new message coordinates (name two actions it flips) and the derivability rule + its gradient-starvation counter-rule; §2.1 names `neutralization` and `tempo` as the two missing ones, with `physics mutation` (Marvel Scale) explicitly out of scope for a one-ply reduction. Ladder R0 hard_max (byte-identity anchor, stays shipped) → R1 belief mean → R2 learned / Deep-Sets → R3 multi-aggregator default (GIN/PNA: no single aggregator suffices) → R4 = OA1. **Two claims carry it:** `α` must be computed from the board ALONE — not per defender, because they choose without seeing your switch — which kills the channel AND defender incoherence with one restriction; and **hedging is not a depth phenomenon** — taking the middle ground under uncertainty needs a *second moment* (`Σ α[o, o²]` ⇒ variance ⇒ a learned risk attitude), which `max` structurally cannot produce, so it is reachable one-ply. Also names what the doc is NOT: the "is this my last answer" scarcity feature is a different arity ([their MON × my mon], reduced over OUR axis) that the OpTensors typing rejects as a shape error (§11). Gates G0–G7; **steps 0–7 need no training run**, so the whole ladder is decidable *beside* gen-5 on frozen checkpoints — the one real cost of the concat having died the same day is that gen-5 trains against R0 `hard_max` and cannot change mid-run, so the earliest a better reducer ships inside a generation is **gen-6**. **ADOPTED 2026-08-10 — §8.1 is the plan of record**: step-0 carries a pre-registered downscope rule (unsuppressed `imx_CELLS` ≲7% ⇒ cheap rungs only), **seed VICReg is a named prerequisite of the critic route** (gen-5's `seeds/*` measured the k=4 readout COLLAPSED — `out_effective_rank` 1.0 sustained — so the trigger fired and `--seed-vicreg-coef` is being wired for gen-6), and G7 runs in the first post-gen-5 GPU window. **Step 0 is new and comes first: re-run the split audit on gen-5** — with no concat there is no competing route, so `imx_CELLS` finally means what it says (on gen-4 it read 6.53% shuffle *while* `FULL_CONCAT` carried the traffic). **Step 1 is already DONE on main** — `damage_op.py:534` `_chan_max(..., how="hard_max")` is the single named call site and any other `how` raises. **G7 — a single-team exploiter A/B with a behavioural-bifurcation readout — is the capability gate** (owner design 2026-08-09; fixed team = **Big Five + Starmie**: Tyranitar/Blissey/Gengar/Swampert/Starmie/Skarmory, chosen because every slot's value is branch-dependent; deliberately no capacity-matched arm, per P3 + the LUT nulls). §10.8 retracts an earlier VoI-ceiling argument of mine that was simply wrong. **NEW forward design (not built), 2026-08-14: `design_history_entity.md`** — E9 steps 2–3 made buildable: the three-tier history design (H-A compiled last-action fields + the pair-history edge family `h[i,j]` answering "what do they click into this mon / whom do they switch into"; H-B event tokens with RELATIVE recency embeddings replacing the 7×159 lag-indexed frames; H-C entity reference edges), the **last-turn-outcome admission rule** (a transition fact survives iff not derivable from current state AND it serves belief-INVERSION or TENDENCY estimation — crit flags live only as deflators on the damage evidence they explain; per-slot HP levels die as state duplication), and the field-by-field nothing-lost mapping from today's TurnDelta layout. Deletes 1124 obs dims when H-B lands; H-A ships independently first and feeds α/β their tendency inputs (measure `alpha_acc` across it). **NEW operational plan, 2026-08-14: `design_cleanup_journey.md`** — the flag/delivery cleanup phases as a decision record: the three flag roles (select/record/gate) and three explicitness tiers (CLI / config-only / constructor-only); Phase 1 issued (registry + consistency test, zarch + seed-pressure deletions, first demotions, the pubval decision package); the owner amendments — **training transport defaults to rust** (node kept as the parity arm; offline seams stay node; soak gate before the flip since the lock-in fix is a day old) and **diagnostics DEFAULT ON** (win_prob + the distributional readout in `read_only` on fresh runs — instruments, not levers; per-decision quantile/tail-mass trace recording feeding the prober's `knew_by_turn`/`lead_time`/`blind_loss` verdicts); Phase 2 launch-by-manifest; Phase 3 post-gen-9 audit deletions; and the **SPARED register** (pairwise kernels, pair_reduce rungs incl. R2/R3 awaiting a fair α retest, the diagnostic heads, node seams, hidden-opp pool) so no stale deletion list outlives its evidence. **Reading aid: the delivery digraph is browsable** — [`architecture_viewer.html`](architecture_viewer.html) via `file://`, or served live at `model.g5d.io` (`python -m agents.model.build_arch_viewer --serve`, which re-renders from the checkout on every request). Edge hue = what the channel physically carries, thickness = measured dependence at a selectable checkpoint, plus a path filter for "what does the critic read" — the fastest way to see the concat's critic-side residual above. It is **generated** from the committed graph snapshot + `research_state/measurements/`, so rebuild it with `python -m agents.model.build_arch_viewer` rather than editing the HTML; `--check` fails on drift. |

---

## Version summaries

### ai_v1
Initial end-to-end PPO pipeline. Basic observation encoding, action masking, first working
training loop. Mostly design/analysis docs — no stable training run yet.

### aI_v2 (note: mixed case in filesystem)
Feature extractor redesign. Shared move processor, role encoder, team attention heads.
First architecture that learned meaningful strategy beyond random.

### ai_v3
Stability and signal hardening. Goals: clean the pipeline, encode richer state, get to
a stable 60–70% win rate against fixed bots.

Key milestones in order: clean pipeline (`impl_step1`), observation features (`impl_step2`),
architecture improvements (`impl_step3`), reward shaping (`impl_step4`), hyperparameters
(`impl_step5`), active state signals (`impl_step6`), effectiveness + move order
(`impl_step7`), item consumption (`impl_step8`), reward overhaul (`impl_step9`), adaptive
training infrastructure (`impl_step10`). Also: launcher with restart loop, spectator mode.

**Training run:** The long-running v3 experiment (350M+ steps) is the most mature model.
It reached ~70–75% vs Heuristic, limited by the fixed-bot ceiling — the policy fights
entropy collapse (ent_coef rose 0.029→0.055) rather than improving further.

### ai_v4
Event-sourced battle layer, strict battle-API, observation richness, and obs-build
performance. *(Originally planned as the self-play/league chapter; that work was deferred to
ai_v5, and ai_v4 became the data-quality + encapsulation chapter that has to come first.)*

Key milestones in order (impl_step1–9): own-team IV/EV/nature spread (`impl_step1`), opponent
Hidden Power type inference (`impl_step2`), damaging-event attribution (`impl_step3`), unified
L=2 transformer feature extractor (`impl_step4`), move-outcome reporting (`impl_step5`), the
next-run bundle — accuracy + modular extractor + dual-head value + reward overhaul
(`impl_step6`), adaptive-LR KL band (`impl_step7`), strict battle-API + event-sourced TurnDelta
fold (`impl_step8`), and strict-API completion + trapping signals + the ~2× obs-build perf pass
(`impl_step9`). Net obs **3321-dim**, `ARCH_SIGNATURE = gen3_trapping_signals_v1`.

**Open tail:** pathology hunting (eval-replay analysis); plus the one unscheduled strict-API
sub-item, Phase 5b (true `LiveView` current-board event-fold — `todo_live_battle.md`). The
first v4-obs run is now live (the fresh fixed-bot run started 2026-05-31, see the state table
above) — the retired v3 run was on an older arch that can't load the v4 obs.

> **Folder name is canonical.** The v4→v5 relocation bumped the *folder* names; the in-folder
> content branding (titles, `designs/ai_vX/` cross-refs, inline "vN" mentions) has since been
> reconciled to match folder names across v5–v8. The state table above and these summaries
> follow the folder names. (Older git history predating that reconciliation may still show the
> pre-relocation labels.)

### ai_v5
Self-play / league play. The agent trains against frozen copies of itself (snapshot pool,
win-rate gating, sentinel monotonicity — Step 1, **code landed, not yet run**), then league
play with exploiters, PFSP, and a two-pool stable (Step 2, **forward design**). Prerequisites,
both designed here: **reward annealing** (`design_reward_annealing.md`, so the value head
learns win probability) and the **league tooling** (`design_league_tooling.md` — the
payoff-matrix runner + Nash/RPP/diversity metrics). Progress is measured by `win_rate_vs_bots`
+ Nash relative population performance (not plain ELO). Relocated here from the original ai_v4
plan.

### ai_v6
Two routes to an **anticipatory** agent — the original MCTS plan, now superseded as the
anticipation route by a search-free alternative:

- **Original (Step 5, superseded):** MCTS at inference + the world model that feeds it. Replay
  collection (**landed** — daemon running), behavioural cloning from human replays, the
  **team-completion model** (masked-slot prediction = the PIMC world-sampling step), the Node.js
  sim bridge, and MCTS itself (inference-time policy-improvement operator). Wang (2024) found MCTS
  gave 78.6% → 90.8% vs Heuristic. Now confined to the **L4 offline-teacher** bucket by the owner's
  no-search-on-the-model constraint (`designs/research_state/`).
- **Favored (Step 6, "Meaning B"):** **latent predictive representation** — a feedforward L3
  auxiliary objective that shapes the shared trunk so the single forward pass *anticipates* one
  ply, with **no runtime simulator or tree** (the sim is a supervision oracle only). Culminates in
  per-action **outcome-token injection** (a learned `g(trunk, action)` → one predicted-outcome
  latent token per legal action, attended by the policy). Incremental ladder with FREE offline
  kill-gates → `design_latent_predictive_representation.md` + `todo.md` Step 6.

Also: surgical checkpoint transfer and PPO embedding improvements.

### ai_v7
Specialisation and ladder play. Evaluate the v6 MCTS generalist across the 32 sample teams,
fine-tune a model per top team, and take them to the ranked Showdown ladder. Also integrates
**cheap** MCTS (shallow K=3 action sampling, depth 1) into the training loop.

### ai_v8
The conditioning/credit-assignment epoch on the v44 model family: the public-info value aux
(v43 `gen3_pubval_aux_v1`, `design_public_info_value.md`), the team-archetype latent + head
FiLM (v44 `gen3_zarch_film_v1` — the amortization-gap storage fix; **the whole zarch family was
DELETED at v78** after the LUT arm returned +0.024, CI [-0.016,+0.064] and the orthogonal 2x2 showed
team COUNT dominating conditioning), the discovery boosters
(team-blocked episodes, grad accumulation + NSR instrumentation, onesided team-PFSP), and the
**next-run pre-flight list** (`next_run_plan.md`: privileged critic, categorical value loss,
top-K=16+tail op candidates, refine=1, obs-skip, belief-grad decision). (The Rust simulator
work originally sketched for this slot shipped as `src/rust_sim/` under its own docs.)

### ai_v9 (the ACTIVE fresh generation)
The entity-graph generation. **The operative roadmap is `design_generation_roadmap.md`** —
it aligns the fresh-generation reset (2026-08-03: no old checkpoints, fresh pools,
position-equivariance first-class, adequacy judged generation-vs-generation by anchored
ELO), the staged sequence (Stage 0 pointer-native head → Stage 1 move tokens E3/E4/E5 →
Stage 2 physics as attention EDGE BIASES + op-concat deletion → Stage 3 declarative schema
+ obs re-home), and the E9 history decision (per-entity recency features → turn tokens →
entity-linked event tokens; recurrence RULED OUT — the obs must stay a pure function of the
event log for the forensic stack). **Stage 0 SHIPPED** (v51 `gen3_pointer_native_v1`,
`f25e708`): the flat `action_net` is deleted; `design_pointer_action_head.md` §0 is its
spec (the staged v49 delta-head sections below §0 are the superseded reasoning record). The
entity/edge INVENTORY (E1–E9, D/S/C/V/T/X, the nothing-lost audit) stays in
`design_entity_graph.md`. The ai_v8 `next_run_plan.md` staging predates the reset —
generation-crossing items there are superseded; re-triage the rest individually.

**E9 history is CLOSED OUT with `gen3_frame_deletion_v1` (2026-08-17):** the H-B event window
replaced the 7×159 TurnDelta lag frames, which are deleted (obs 3529 → 2437). Its open
reconciliation is [`design_frame_deletion_coverage_gaps.md`](ai_v9/design_frame_deletion_coverage_gaps.md)
— the three facts that ship WITHOUT an event-window home (the refused switch's target, the eight
faint causes, item-consumed), plus the methodological finding behind them: **a dV ablation says
whether the model LEANS on a block and cannot say whether each FACT in it has a home elsewhere**,
and a fact with no substitute reads LOW on dV exactly when the model never learned to use it. The
doc proposes the standing rule that an irreversible block deletion needs BOTH readings.

### ai_v10 (OPEN — the exploiter-SCALING chapter, + three forward docs)
Opened 2026-08-16. **Nothing built.** One document:
[`design_exploiter_scaling.md`](ai_v10/design_exploiter_scaling.md) — why exploiter competence
collapses between N=10 and N=20 teams when N=1..5 is trivial, and the **pre-registered battery**
that would settle it. Where ai_v9 is about the *entity graph inside one battle*, ai_v10 is about
*what transfers between teams*. The forward docs: the flywheel, the outcome latent, and
[`design_counterfactual_value_grounding.md`](ai_v10/design_counterfactual_value_grounding.md)
(2026-08-22 — the counterfactual label factory + the R1/R2/R3 critic-bias attacks, priced by the
2026-08-21 probe triad, gated on bias meters; the C6-compliant new mechanism).

Carries two NEW measurements on gen-12 (both held-out / causal, both on the historical exploiter
team sets): the "it's just 6 1v1s" reading is **REFUTED** — halving a benched mon's HP moves the
MOVE logits at ratio **0.262** of the same perturbation on the active (n=4058, 8 teams) — but the
profile is FLAT, so the bench enters as a **team-health SCALAR, not as structure**; and team PACE
class is decodable from the raw obs at **0.456 on unseen teams** while `pi_features` (0.211) and
`value_pooled` (0.199) sit at **chance (0.200)** — **the abstraction is free in the input and the
trunk discards it.** The probe splits BY TEAM, never by state: rosters are in the obs, so a
same-team split scores ~100% by memorising species sets and measures nothing.

Hypothesis: no transferable team-scoped abstraction ⇒ sample cost linear in N ⇒ past ~10 the
per-team budget falls below the threshold at which an exploit fires, and the best available policy
IS the generic one. Competing accounts kept live and separated by the battery: **H_rate** (D3's
leaky bucket, P(team)=1/N), **H_capacity** (zero-sum), **H_conflict** (pairwise interference),
**H_coverage** (only ~K distinct exploits exist). §7 pre-registers the three results that would
kill the hypothesis — written because the same session produced two clean mechanistic stories that
measurement killed outright.

### ai_v11 (PUNTED draft — OPEN — the human-ladder-replay chapter)
Opened 2026-08-18. **Nothing built.** One document,
[`design_human_replay_objectives.md`](ai_v11/design_human_replay_objectives.md), plus a
[`todo.md`](ai_v11/todo.md). Where ai_v9 is the entity graph inside one battle and ai_v10 is what
transfers between teams, **ai_v11 is what we can learn from an action distribution that is not our
own** — and what survives the owner's constraint that spectator replays are PARTIAL information.

**The spine is an OOD taxonomy, not an enthusiasm.** Our observation always carries our own full
team (six mons, four moves, item, exact IV/EV/nature spread) plus a server `|request|` stating the
legal set. A spectator log recovers our side only as the game exposed it — and the request stream
does not exist at all, so `log_reader._synth_legal` fabricates the mask (`trapped=False`, every
revealed move choosable) and it is systematically **over-permissive**. The asymmetry is what orders
the ladder: the OPPONENT half of a replay obs is in its native distribution (that side is hidden
live too), our half is not.

**Four rungs, ordered by OOD-robustness, each pre-registered with a gate and a kill:** (1) α/β
supervised on the human OPPONENT's actions — labels fully observable, opponent-side inputs
unshifted, and the `--intent-label-bot-weight` / `opp_class` seam generalises to a HUMAN class
(though offline rows need their own batch path, not the rollout buffer); (2) outcome/value
supervision on human states via the `WinProbLabelCallback` MC pattern, `read_only`, LEVEL claims
only; (3) BC-regularization restricted to the faithful, mask-audited subset — the **gen-17
candidate**; (4) offline RL with the acting side's unrevealed slots filled by the **team-completion
model** (BUILT, UNTRAINED), whose spread head must draw on `gen3_spread_priors.json` because no
replay ever states a spread.

**Phase-0 census RUN** (`tmp/replay_faithfulness_census.py`, 2026-08-18; corpus 263,159 logs /
2.8 GB / 2026-05-18→2026-08-02 — the frontier row's "~102k" is stale): at rating ≥1500, over 30,146
reconstructed decisions from 1,142 sides, fully-faithful (6/6 bench, 4/4 moves) decisions are
**16.70%**, own **item is known on 3.93%** of own mons and ability on 7.86%, only **8.64%** of own
mons reach 4/4 moves, **3.15% of sides fail to parse** (`UnknownVolatileError` — and
`human_agreement.py` swallows them silently), α-label pairing is **92.04%**, and the human switch
share reproduces the 2026-06-12 under-switching finding **model-free at 28.96%**. The sharpest
single finding: a replay-reconstructed own team is encoded with **`spread_known = 1.0` over an
all-31-IV / 0-EV / neutral-nature FABRICATION** — not a gap but a wrong value asserted as known,
feeding the `d1`/`d2` outgoing physics. And faithfulness is **not missing-at-random**: the faithful
stratum is **loss-enriched 1.29×**, so any outcome-labelled objective needs outcome-balanced
weights, pre-registered.

## Folder conventions

Each version folder has:
- `todo.md` — in-progress checklist; `✓ DONE` marks completed steps
- `impl_step*.md` — post-implementation records (what was built, constants set, files
  changed); these are the primary targets for `gen3ai-update-design-docs`
- `design_*.md` — forward-looking design docs written before implementation

When writing a new `impl_step*.md`, match the existing docs in that folder exactly —
heading levels, table style, and level of detail vary between versions.

## Cross-version docs (designs/ root)

- **`flag_registry.md`** — **GENERATED** (from `agents/model/flag_registry.py`; `python -m
  agents.model.flag_registry`, `--check` is the gate). The table of every model-relevant extractor
  toggle with its TIER (`cli` / `config_only` / `constructor_only`), CLASS (`structural` /
  `resume_immutable` / `training_coef` / `runtime`), default, `since` version and one-line meaning —
  plus the prose on why a settled flag can lose its CLI entry without losing explicitness, and which
  gate each class picks. Read it before adding or demoting a toggle; the rules live in
  `src/agents/model/CLAUDE.md` → The flag registry.
- **`design_pathologies.md`** — living model-pathology register: *what's wrong → what we changed →
  what we expect to be different next time*. **Review it before every retrain** and add a row after
  each eval noting whether a fix's predicted change actually landed. Spans the pathology-hunting
  effort (the ai_v4 tail) and the obs/reward fixes it motivates; currently records the
  `run_20260531_182804` findings, the `gen3_move_effects_v1` move-effect obs fix, and the open
  matchup-variance prior-vs-confirmed question.

## `learning/` — concept explainers (version-agnostic)

`designs/learning/` holds **durable teaching notes** — one markdown file per major concept,
each a two-level explainer (intuitive → technical, no code) grounded in *our* architecture
(flags, `ARCH_SIGNATURE`s, obs blocks, real file names). These are **always-current reference
docs**, not version-keyed impl records: if the architecture changes such that a note is wrong,
fix it in the same pass. The `/gen3ai-learning` skill creates and maintains them.

- **`marginalization_and_uncertainty.md`** — marginalize vs mean-field, Jensen's inequality,
  the threshold/tail problem (P(KO), P(outspeed)), and how a neural net actually represents and
  reasons over uncertainty (distribution-param heads, distributional RL / `ValueDistHead`,
  attention-as-marginalization, why MSE bakes in mean-field, factoring the marginalization into
  the differentiable `DamageOperator`). Also owns the **convex-combination primitive** —
  expectation *is* a convex combination, a convex *function* is defined by how it acts on one
  (= Jensen), and on a feature vector it combines coordinate-wise so it preserves units, range
  and scale (why `ValueDistHead`'s mean cannot leave `[v_min, v_max]`, and why an attention
  *value* carries a magnitude where an attention *bias* cannot).
- **`entity_tokens_biases_pointers.md`** — the ai_v9 concept vocabulary: what entity-based
  (entity-centric / relational) modeling is and where it came from (CNN weight-sharing →
  GNNs → Deep Sets → Transformers → AlphaStar/AlphaFold), why permutation equivariance beats a
  flat positional vector (weight sharing, hypothesis-space reduction, whole bug classes made
  *unrepresentable*) and what it costs, the **sorting rule** for where a fact lives
  (token / edge / distribution summary / attention), how expected damage is delivered (the
  `DamageOperator` as a *differentiable expert* whose gradient trains the move belief; the
  shipped v51 `pointer_cells` route vs the Stage-2 edge-bias route; **the output-slot ladder** —
  PMA / entity cross-attention / multi-query seeds / pair-token promotion as one dial, why we
  shipped only the key-side half of **Shaw et al. 2018** and what the value-side term buys, the
  OA1-as-conditional-expectation identity, where each option's cost lands, and the **seed-collapse**
  monitors), and how **history** is
  represented once time stops being positional (recency-on-entity → turn tokens → entity-linked
  event tokens; recurrence ruled out by the event-log-purity invariant). **Part 6** covers the
  ai_v9 **compositionality** result on the live v57 architecture: the sorting rule as a
  composition *contract* (partition by arity+certainty → locality of change; the G→C4 worked
  example; what each violation costs, with the measured P1/P4 and v34→v39 evidence), **routing
  vs payload** (one concrete E4-seat / D3-bias forward pass, what a softmax weight structurally
  cannot carry, the three delivery routes and the critic's dependence on the concat, the first
  edge-family ablation audit — outgoing dominant, incoming near-decorative — plus three
  falsifiable explanations), the **equivariance trade** (weight-sharing arithmetic, the bug
  classes made unrepresentable, and the four costs paid), the **hypothetical-world trick** that
  makes the remaining C family cheap (pure-function kernel; why the cell is a delta; where it
  ceilings vs real search), **the head funnel** (35 seats → 3 pooled vectors for pi and ONE for vf;
  the op concat as the only un-pooled route for both and the pointer head as a policy-only second
  one; why that predicts D2's |ΔV|; the P3 counterweight against "widen the value pool"), **what
  search would look like** (the CRN-anchored beam, C-deltas as a pruning layer before the expensive
  clone, equivariant candidate generation, why no-recurrence is what makes cloning legal, and the
  simultaneous-move correction that the object is an equilibrium not a best path), **entity
  structure vs FiLM/LoRA** (input-symmetry vs parameter-context factorisation; "share where a
  symmetry is real, condition where it is false"; edge-bias and FiLM as one hypernetwork shape at
  different clock speeds; where LoRA would attach and the two measured nulls standing against it),
  a **quiz + answer sketches** on designing the next family, and — **§6.9, the canonical
  statement** — **what stays POSITIONAL in the end state**: invariance vs equivariance vs true
  position-dependence, the full inventory (time and the two sides are real asymmetries and must
  survive; seat-index conventions and PV seeds are not positional), OA's per-axis symmetry table
  and its pre-registered permutation gate, and why the critic-route choice **7a vs 7b** is exactly
  a choice about one positional axis (expressiveness vs equivariance). ASCII diagrams throughout
  (seat layout, the eleven families as blocks in the from×to grid, the one concrete
  E4→D3→token→logit link, the head funnel, the search tree).
- **`shortcut_learning_and_feature_delivery.md`** — the input-side dual of
  `objective_richness_and_representation.md`: whether feeding a computed feature straight to the
  head makes the model "lazy," and when that is a plus. Gradient starvation (not "simplest
  function"), the ~1-bit-per-game RL amplifier, **amortization vs. bottleneck** (sufficiency for
  the decision is the whole variable), the **axis rule** ("never collapse an axis you must choose
  along" — the v30→v39 progression), the four tests that discriminate laziness from genuine use
  (ablation-KL / trunk linear probe / behavioural counterfactual / held-out), the measured P1
  ablation surprise (the model **ignores** collapsed summaries when un-collapsed ones sit beside
  them), the reframe "make the lazy path the correct path" (= what v51's pointer head does), and
  **Part 6 — the concat end-state**: why the edges grew *without* absorbing the op head-concat
  (paths compete only when they are substitutes), the structural argument that **softmax edge
  biases carry ratios, not absolutes** (so magnitude needs token content or per-action cells),
  the **pre-registered** delete-vs-re-home decision rule keyed to gen-3's audit, its four
  confounds (value head / first-mover / mid-curve / perturbation-mismatched arms), and why
  deleting the flat obs *dissolves* the starvation question rather than answering it.
- **`on_policy_self_distillation.md`** — on-policy distillation (OPD) as the dense-signal training
  regime, why it's ~7-10× more step-efficient than PPO (a full target distribution per state vs ~1
  bit/game), our `better-line` beam as the policy-improvement teacher, upgrading the `search-teacher`
  AWR-toward-`A*` to a full-distribution `KL(π' ‖ π)` (with the `V^{π*}`/GAE-bias dissolved by
  distilling the *policy* while the critic sees only confirmed returns), cheap Gumbel-top-k × opp-axis-
  collapse search under the expensive `DamageOperator` critic (≈8 evals/node vs 121), and the
  **team-subset exploiter** as where OPD compounds. Grounds ExIt/AlphaZero, Grill 2020, Gumbel MuZero,
  ReBeL/Student of Games onto our tooling.
- **`population_game_theory.md`** — the flywheel era's native language: strength as a MATRIX
  (payoff matrices, transitivity), Nash averaging (why uniform pool win-rates lie and duplicates
  inflate them), spinning tops (transitive spine + non-transitive width, and reading the exploiter
  "random walk" as motion along the width), PSRO (tock = oracle, PFSP pool = meta-solver, the tick's
  distillation as the projection step PSRO lacks), and exploitability lower bounds
  (exploiter_wr − baseline). Owns the "flat ELO: converged vs circling a cycle?" diagnostic.
- **`credit_assignment_and_value_errors.md`** — why critic surprise is the named enemy: one bit per
  game amplified by the critic into per-decision credit; GAE's λ as the bias-variance dial;
  BOOTSTRAP error propagation (value errors travel backward — TD-aux as consistency that suppresses
  noise but cannot create signal); PopArt/vf_coef as trunk arbitration; what distributional heads
  buy (awareness, PIT calibration) vs measurably did not (sub-Gaussian residuals, no tail to
  reweight); the FOUR critic-failure causes (input coverage / distribution / representation /
  horizon) with the instrument that separates each.
- **`imperfect_information_and_equilibria.md`** — what the belief stack approximates: information
  sets, CFR vocabulary, the PUBLIC BELIEF STATE (our species/move/spread/item posteriors as an
  empirical PBS), α as a trained equilibrium fixed point (why it must never read our own logits),
  determinization vs expectation, the exploitability-vs-exploitation corners (generalist ≈ safe
  center, exploiters = pure best response — safe only because distilled, never deployed), and the
  named scope cut: nothing models what OUR actions reveal.
- **`continual_learning_and_forgetting.md`** — the tick's quiet risk: forgetting as an
  optimization (not capacity) problem; the three fix families and which we already run (teacher
  data = REHEARSAL, the reason the fold has no 1/N wall; EWC unneeded while old data is free;
  architectural separation twice-nulled by the zarch/FiLM results + orthogonal-gradient finding);
  the measured anchors (76% retention, 93% headroom capture, the cosine audit as the conflict
  detector); the tick-tock as an EXPLICIT stability-plasticity schedule; the three-cause
  decay diagnostic (interference / capacity / drift), each with a different cure.
- **`quality_diversity_and_open_endedness.md`** — the archive view of the flywheel: MAP-Elites
  (slices = cells, teachers = elites, tock = mutation; coverage and QD-score as first-class
  metrics beside ELO), descriptor choice as THE load-bearing decision (composition axes today, the
  deferred fingerprint aux as a learned strategy descriptor), POET as the revolution-3+ question
  (let the system propose slices), and stepping stones (warm-fork as the weak form; exploiter→
  exploiter transfer as the strong form the task-arithmetic preview could detect).
- **`win_prob_decomposition.md`** — the five-axis taxonomy of "the critic was wrong", built
  empirically 2026-08-21/22: luck (single outcome vs tight-MC — 53% of the conviction class),
  calibration vs RESOLUTION (Murphy split — the blur is the disease), the population/ecology
  sign-flip, learnable vs the IRREDUCIBLE hidden-information floor (39% of the meter,
  concentrated — the Salamence coin; determinization as the instrument), and the epistemic layer
  (the evidential head confessing width). The assembled tree with numbers, the floor-subtraction
  rules, and the generalizing ladder (sample the outcome / stratify the population / bin the
  predictions / determinize the hidden state).
- **`activation_functions.md`** — is ReLU the right nonlinearity for our model? The three
  nonlinearity **tiers** we already live by (generic ReLU trunk = swappable · SB3 `[512,512]` tower =
  tanh by an SB3 *default* we never set · bounded pointer-head tanh = deliberate · the semantic
  sigmoid/softmax/clamp tier where the function *is* the quantity and swapping it is a physics bug),
  why dying-ReLU risk is low in our shallow LayerNorm-sandwiched trunk, and the one site where the
  choice is structural — the extractor **returns** `ReLU(·)`, so the whole policy/critic interface is
  non-negative and every signed quantity costs a channel pair. Records `gen3_policy_activation_pin_v1`:
  pinning `POLICY_ACTIVATION_FN = nn.Tanh` (behaviour-neutral — it *is* the current default) because an
  activation swap is retrain-class but weight-shape-NEUTRAL, so `check_compatible` cannot see it and an
  sb3-contrib upgrade would silently rewrite four policy layers. Any real swap must bump `ARCH_SIGNATURE`
  deliberately and re-pass the compile gates.
