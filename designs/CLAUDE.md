# designs/ — the folder map

Read this whenever you're about to touch anything in `designs/`. **It is a map, not an architecture
reference**: what the model IS now is [`ARCHITECTURE.md`](ARCHITECTURE.md); how it got here is
[`CHANGELOG.md`](CHANGELOG.md) (history; never quote it as current); what we BELIEVE about the
research is [`research_state/UNDERSTANDING.md`](research_state/UNDERSTANDING.md); where the system is
heading is [`endstate/README.md`](endstate/README.md). (The previous, longer leaf — with its
2026-09-07 version-state table — is frozen at
`research_state/claude_md_archive/designs_CLAUDE_2026-10-10.md`.)

**ALWAYS-CURRENT files** — updated in the same pass as the change that makes one stale, never
narrated, never appended to:

| file | states | its append-only counterpart |
|---|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | what the MODEL is now — §6's tables are GENERATED (`python -m agents.model.arch_tables`, pinned by `arch_tables_test.py`) and the PROSE around them is pinned by `src/mode_flag_doc_gate_test.py` against `production_config.json` | [`CHANGELOG.md`](CHANGELOG.md) |
| [`research_state/UNDERSTANDING.md`](research_state/UNDERSTANDING.md) | what we BELIEVE about the research now — every claim tagged SIGNIFICANT / WITHIN FLOOR / NOT DETECTED / EQUIVALENCE SUPPORTED / REFUTED / UNVERIFIED and pointing at its evidence. 🔎 **Find a ledger entry through the GENERATED [`research_state/ledger_index.md`](research_state/ledger_index.md)** (`python -m main.ledger_index --write`, gated by `src/ledger_index_gate_test.py`) — NEVER edit either by hand; `research_state/README.md` holds the heading convention and the rebase rule | [`research_state/ledger.md`](research_state/ledger.md) |
| every doc in [`endstate/`](endstate/) | the end-state designs, each ending with a **Decision record** (owner, 2026-09-27): a decision or build that differs updates the doc in the same commit, saying what changed and why. [`endstate/README.md`](endstate/README.md) is the index and the reading order | — |
| the eight leaf-detail trees below, and every `CLAUDE.md` under `designs/` | the detail lifted out of a code leaf, updated with the code exactly like the leaf | — |

When `UNDERSTANDING.md` and the ledger disagree, **the later ledger entry wins and the view is a
bug** — fix the view, never the ledger.

---

## 🚨 A run ≠ the code, and no `ai_vN` folder is the current chapter

**Read a run from the RUN** — its `model_config.json` + `metadata.json` (`python -m main.ops.run_ref`,
`main.lineage`) — and the code's version from `src/agents/model/model_version/constants.py`
(`MODEL_CONFIG_VERSION`, `ARCH_SIGNATURE`) + `migrations.py` (`MIGRATION_FLOOR`), never from prose.
They differ almost always; a pinned run trains on its pin's code.

**Now:** the Rustboro era (`rb_` runs, `src/utils/era.py`) on the Rust stack. Current design work
lives in [`endstate/`](endstate/); the ranked queues are `research_state/EXPERIMENT_BACKLOG.md`
(experiments), `ops/TASK_BACKLOG.md` (builds) and `ops/TECH_DEBT_BACKLOG.md` (debt). Production is
what `--arch production` resolves — `production_config.json` (+ its sibling `production_config.README.md`
for provenance) and the registry `baselines.json` — never a run named in prose. The `ai_vN` folders
below are the record of earlier chapters: read one for what was designed and why, then check the
code before believing it still holds.

---

## The leaf-detail trees (always-current, each OWNS its topic)

| tree | lifted out of | holds |
|---|---|---|
| [`model/`](model/) | `src/agents/model/CLAUDE.md` | the phase pipeline, readouts and value routes, file layout, op contracts, flag-registry rules, versioning, opponent intent, the architecture artifacts, typing. **`ARCHITECTURE.md` stays the doc of record for what the model IS** — a topic doc points at it and loses any disagreement with it |
| [`training/`](training/) | `src/agents/training/CLAUDE.md` | one doc per training topic (eval and rating, self-play and pool, the critic and value losses, the collector, the learner gates and lifecycle, …) |
| [`rust_sim/`](rust_sim/) | `src/rust_sim/CLAUDE.md` | the port's module map, the move census, the differential-gate ladder, the e2e capstone, regression pins, the four A/B fuzzers (`fuzzers.md`) and their closed findings, protocol emission, the websocket front end, the live reader; `port_build_log.md` is its closed coverage rounds |
| [`prober/`](prober/) | `src/main/prober/CLAUDE.md` (+ `web/`) | the analyze panels, result timeline, belief/threat views, the per-method session reference, counterfactual probes, arch drift, the sim impl, the tests, the result vocabulary, retention, and `/game` (the battle viewer) |
| [`launcher/`](launcher/) | `src/main/launcher/CLAUDE.md` | restarts and the resume contract, crashes and exit codes, the flag notes, pinning and worktrees, the launch guards, the child's interpreter, the TUI ([`launcher/README.md`](launcher/README.md) is the index) |
| [`observation/`](observation/) | `src/agents/observation/CLAUDE.md` | per-block field semantics, the volatile vocabulary, typing ([`observation/README.md`](observation/README.md) is the index; the Rust encoder itself is `rust_sim/encoder.md`) |
| [`tools/`](tools/) | `tools/CLAUDE.md` | the data extractor's per-builder notes, the Smogon priors (the denominator, the 12-month merge, the format-spec filter), the team downloaders and their encodings ([`tools/README.md`](tools/README.md) is the index) |
| [`ops/`](ops/) | the root `CLAUDE.md` | the SOPs and chapters below |

**`ops/` — procedures of record.** [`ops/TRAINING_RUN_SOP.md`](ops/TRAINING_RUN_SOP.md) (how a run is
launched, watched, killed, relaunched and read) and [`ops/ORCHESTRATOR_SOP.md`](ops/ORCHESTRATOR_SOP.md)
(the orchestrator's dispatch, landing, reporting and cadence, the scope of unasked action, agent
stalls and waiting, §7) are the PROCEDURES OF RECORD for the two long-lived sessions — an owner ruling
about how a session operates is written into the owning section there, not into a memory file.
[`ops/EXTERNAL_ANCHORS_SOP.md`](ops/EXTERNAL_ANCHORS_SOP.md) reads our strength against agents nobody
here trained (Metamon's policies, Foul Play) — the tiers, the greedy-vs-greedy rule and why, the
commands, the hazards; its box-specific paths are [`ops/anchors.json`](ops/anchors.json).
[`ops/testing.md`](ops/testing.md) and [`ops/training_runbook.md`](ops/training_runbook.md) are the
root's testing and training chapters. [`ops/TECH_DEBT_BACKLOG.md`](ops/TECH_DEBT_BACKLOG.md) is the one
tech-debt list (nothing on it is dispatched without the owner's word);
[`ops/deletion_pass_manifest.md`](ops/deletion_pass_manifest.md) records the post-M5 deletion pass,
each unit marked SHIPPED as it landed.

---

## Cross-version docs (designs/ root)

- **`flag_registry.md`** — **GENERATED** (from `agents/model/flag_registry.py`; `python -m
  agents.model.flag_registry`, `--check` is the gate): every model-relevant extractor toggle with its
  TIER (`cli` / `config_only` / `constructor_only`), CLASS, default, `since` version and meaning. Read
  it before adding or demoting a toggle; the rules live in `model/flag_registry_rules.md`.
- **`deleted_flags.md`** — every deleted flag and path with its citation; the `CLAUDE.md` freshness
  gate reads it for names a doc cites deliberately as history.
- **`production_config.json`** + **`production_config.README.md`** — the production config mirror and
  its provenance (the `recipe` block is hand-edited, together with `endstate/design_learner_recipe.md`
  §3.22; `src/recipe_doc_gate_test.py` fails when either moves alone). **`baselines.json`** — the named
  baselines, changed only by `python -m main.baselines set … --reason`.
- **`design_pathologies.md`** — the living model-pathology register: *what's wrong → what we changed →
  what we expect next time*. Review it before every retrain; add a row after each eval saying whether
  a fix's predicted change landed.

---

## Version folders — which is which (history)

**Folder conventions:** each has `todo.md` (`✓ DONE` marks completed steps), `impl_step*.md`
(post-implementation records — the targets of `/gen3ai-update-design-docs`; match the folder's
existing style exactly when writing a new one) and `design_*.md` (forward designs written before
implementation). These, and `todo.md`, are explicit-only — never auto-updated. The folder name is
canonical (the v4→v5 relocation renamed folders; older history may show pre-relocation labels).

| folder | the chapter |
|---|---|
| `ai_v1` | the initial end-to-end PPO pipeline |
| `aI_v2` *(mixed case on disk)* | the feature-extractor redesign — shared move processor, role encoder, team attention |
| `ai_v3` | stability and signal hardening, `impl_step1`–`10`; its README is a FROZEN historical digraph |
| `ai_v4` | data quality + encapsulation — own-team spreads, Hidden Power inference, damage attribution, the L=2 transformer, the adaptive-LR KL band |
| `ai_v5` | self-play / league — the snapshot pool with win-rate gating; exploiters / PFSP designs |
| `ai_v6` | anticipation — the MCTS route (superseded; search stays an offline teacher) and the latent-predictive auxiliary |
| `ai_v7` | specialisation and ladder play (forward designs) |
| `ai_v8` | the conditioning / credit-assignment epoch (the zarch family was DELETED at v78 — count dominates conditioning) |
| `ai_v9` | the entity-graph generation — `design_generation_roadmap.md` (pointer-native head, move tokens, physics as attention edge biases); `design_entity_graph.md` is the entity/edge inventory; `design_frame_deletion_coverage_gaps.md` carries the standing rule that **a dV ablation says whether the model LEANS on a block, never whether each FACT in it has a home elsewhere** |
| `ai_v10` | exploiter SCALING (forward docs, never built) — why exploiter competence collapses between N=10 and N=20 teams; `design_advantage_gated_distillation.md` records a fold lever that was DELETED (L3) |
| `ai_v11` | human ladder replays (forward doc, never built) — the OOD taxonomy of spectator replays; the Phase-0 census (16.70 % fully-faithful decisions; a reconstructed own team asserts `spread_known = 1.0` over a FABRICATION) |
| `ai_v12` | the win-probability-critic chapter (2026-08-29 → 09) — `design_winprob_only_critic.md` LANDED: the win-prob critic is now the ONLY critic (the shaped critic and every shaped-reward flag are deleted, `deleted_flags.md`); `launch_runbook.md`'s arms carry deleted flags and run only PINNED to ≤ `029cee83`; also the 40-team slate and `promotion_exclusions.json` (rebuilt by `python -m main.promote_teams --regenerate-exclusions`) |

---

## `learning/` — concept explainers (version-agnostic)

`designs/learning/` holds **durable teaching notes** — one file per major concept, a two-level
explainer (intuitive → technical, no code) grounded in *our* architecture. **Always-current**: if the
architecture changes such that a note is wrong, fix it in the same pass. The `/gen3ai-learning` skill
creates and maintains them. **Each note is its own table of contents — open it rather than trusting a
one-line summary.**

| note | what it owns |
|---|---|
| `entity_tokens_biases_pointers.md` | the ai_v9 vocabulary — equivariance, the **sorting rule** for where a fact lives (token / edge / summary / attention), the `DamageOperator` as a differentiable expert, the head funnel, §6.9 (what stays POSITIONAL in the end state) |
| `shortcut_learning_and_feature_delivery.md` | the input side — gradient starvation, amortization vs bottleneck, the **axis rule**, the four tests that separate laziness from genuine use |
| `objective_richness_and_representation.md` | its output-side dual: what a richer objective buys a representation |
| `marginalization_and_uncertainty.md` | marginalize vs mean-field, Jensen, the threshold/tail problem, the convex-combination primitive |
| `credit_assignment_and_value_errors.md` | GAE's λ, bootstrap error propagation, `vf_coef` as trunk arbitration, the FOUR critic-failure causes and the instrument for each |
| `win_prob_decomposition.md` | the five-axis taxonomy of "the critic was wrong" — luck, calibration vs RESOLUTION, the population sign-flip, the hidden-information floor, the epistemic layer |
| `imperfect_information_and_equilibria.md` | information sets, CFR vocabulary, the PUBLIC BELIEF STATE, α as a trained fixed point |
| `population_game_theory.md` | strength as a MATRIX — Nash averaging, spinning tops, PSRO, exploitability; "flat ELO: converged or circling a cycle?" |
| `on_policy_self_distillation.md` | OPD as the dense-signal regime, `better-line` as the improvement teacher |
| `negative_transfer_and_shared_functions.md` | why a fold on eight teams moves the other 711 — GIFT and LEAK as one displacement |
| `continual_learning_and_forgetting.md` | forgetting as an optimization problem, the fix families, the three-cause decay diagnostic |
| `quality_diversity_and_open_endedness.md` | MAP-Elites, descriptor choice, POET, stepping stones |
| `activation_functions.md` | the nonlinearity TIERS; why an activation swap is retrain-class yet weight-shape-NEUTRAL |
| `vacuous_tests_and_guards.md` | the failure indistinguishable from success — the taxonomy, the **arranged-vs-encountered** rule, the five design rules |

Also here: `amortization_gap_and_conditioning.md`, `conditioning_architectures.md`,
`distillation_flywheel_lessons.md`, `exercises_and_reading.md`,
`generalist_specialist_amortization_gap.md`, `latent_belief_metrics_and_collapse.md`,
`pbs_value_functions_and_search.md`, `popart_value_scale_and_currencies.md` (PopArt itself is deleted),
`regularization_and_noise_in_ppo.md`, `rl_concepts_studyguide.md`,
`self_discovered_archetype_latent.md`, `temperature_mixing_and_risk.md`.
