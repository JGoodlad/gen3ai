# designs/ — Version Map

This file tells Claude which `ai_vN` folder is relevant when reading or writing design
docs. Read it whenever you're about to touch anything in `designs/`.

**It is a version map, not an architecture reference.** For what the model actually is right now —
obs layout, phase chain, per-head inputs, the `DamageOperator` block, the edge families, and which
production flags are `INERT` — read [`ARCHITECTURE.md`](ARCHITECTURE.md). For how each version
changed things, read [`CHANGELOG.md`](CHANGELOG.md) (history; do not quote it as current). For what
we currently BELIEVE about the research — the mission, the era map, the flywheel's status, the live
win-prob-critic era, the retired hypotheses and the standing rules of evidence — read
[`research_state/UNDERSTANDING.md`](research_state/UNDERSTANDING.md).

**Three files here carry the ALWAYS-CURRENT obligation** and are updated in the same pass as the
change that makes one stale — never narrated, never appended to:

| file | states | its append-only counterpart |
|---|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | what the MODEL is now — §6's tables are GENERATED (`python -m agents.model.arch_tables`, pinned by `arch_tables_test.py`) and the PROSE around them is pinned by `src/mode_flag_doc_gate_test.py`, which compares every MODE-flag value the prose states against `production_config.json` | [`CHANGELOG.md`](CHANGELOG.md) |
| [`research_state/UNDERSTANDING.md`](research_state/UNDERSTANDING.md) | what we BELIEVE about the research now — every claim tagged SIGNIFICANT / WITHIN FLOOR / NOT DETECTED / EQUIVALENCE SUPPORTED / REFUTED / UNVERIFIED and pointing at the ledger entry or `measurements/` artifact behind it. 🔎 **Find a ledger entry through the GENERATED [`research_state/ledger_index.md`](research_state/ledger_index.md)** (date · line · title, `python -m main.ledger_index --write`, gated by `src/ledger_index_gate_test.py`) rather than a regex over the ledger — and NEVER edit either by hand; `research_state/README.md` holds the heading convention and the rebase rule | [`research_state/ledger.md`](research_state/ledger.md) |
| every `CLAUDE.md` under `designs/` | where to look | — |

When `UNDERSTANDING.md` and the ledger disagree, **the later ledger entry wins and the view is a
bug** — fix it, do not edit the ledger.

---

## Critical: Training run ≠ Code version

**These two are almost always at different versions at once.** A run lasts weeks; code changes
daily. When the user says "update the doc" or "record what we built", figure out which version
applies to *what was just implemented*, not to *what is currently training*. To orient yourself:
`git log --oneline -10 -- designs/ src/` (which `ai_vN` folder the code changes belong to) and
`designs/ai_vN/todo.md` (the in-progress version's has the most recent `✓ DONE` entries; the running
run's is mostly done).

**Current state as of 2026-09-07** — one status line per row. The narrative each cell used to carry
was lifted to `research_state/claude_md_archive/` on 2026-09-07 (that directory is HISTORY —
additive only, never updated afterwards). **Every number below is a snapshot; the run and the code
are the sources.**

| What | Version | Where the detail is |
|------|---------|-------|
| **Active training run** | 🟢 **`ai_v12_02_winprob_critic`** — the first VALID ai_v12 arm, relaunched 2026-09-06 20:27 | FRESH weights (the era has no warm start), config **v110** at launch, pinned to **`f971caf2`**, `--steps 75000000`, the production surface plus `--critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0 --draw-penalty 0` — the **SPARSE** rung of [`ai_v12/launch_runbook.md`](ai_v12/launch_runbook.md)'s ladder, and the only run in the archive recording `critic: "winprob"`. 🚨 **Stall rate and mean episode length are PRIMARY endpoints here**: a critic bounded in [0,1] cannot represent "a timeout is worse than a loss". 🚨 **Read a live run from the RUN** (`model_config.json` + `metadata.json`, `python -m main.ops.run_ref`), never from this table. Its dead predecessors, the G5 controls and the gen-14 → gen-1 history: [`designs_version_map_state_table_2026-09-07.md`](research_state/claude_md_archive/designs_version_map_state_table_2026-09-07.md) |
| **Code on main** | **ai_v12 chapter, ai_v9 architecture** | 🚨 **Read `MODEL_CONFIG_VERSION`, `ARCH_SIGNATURE` and `MIGRATION_FLOOR` from `src/agents/model/model_version/constants.py` + `migrations.py`, never from prose** — this cell carried a stale floor for a month. `ARCH_SIGNATURE` has not moved since v96, so the later bumps are additive/provenance and every v96+ checkpoint still loads. What each version from v51 on changed, deleted, and on what evidence: [`designs_version_map_config_version_ladder.md`](research_state/claude_md_archive/designs_version_map_config_version_ladder.md) (2026-09-07 snapshot; [`CHANGELOG.md`](CHANGELOG.md) is the maintained history, [`ARCHITECTURE.md`](ARCHITECTURE.md) the current truth). `production_config.json` mirrors an older config during a bump window — it is the drift gate's reference, not a description of the newest run |
| **ai_v12** | 🟢 **THE LIVE CHAPTER — code landed, arm 1 running** | [`ai_v12/`](ai_v12/) — the clean-world / win-probability-critic chapter. Summary below |
| **ai_v10** | **OPEN — nothing built** | [`ai_v10/`](ai_v10/) — exploiter SCALING, four forward docs. Summary below |
| **ai_v11** | **OPEN — nothing built** | [`ai_v11/`](ai_v11/) — human ladder replays. Summary below |
| **ai_v9** | **Stages 0–2 SHIPPED + Stage-3 half** | [`ai_v9/`](ai_v9/) — the entity-graph generation; `design_generation_roadmap.md` is the operative plan. The concat-deletion evidence and the six forward designs: [`designs_version_map_ai_v9_stages_and_forward_designs.md`](research_state/claude_md_archive/designs_version_map_ai_v9_stages_and_forward_designs.md). Summary below |

---

## Version summaries — which folder is which

The detail is in the folder; these say only what the chapter is *for*.

### ai_v1
The initial end-to-end PPO pipeline — obs encoding, action masking, the first training loop.

### aI_v2 (note: mixed case in filesystem)
Feature-extractor redesign — shared move processor, role encoder, team attention heads; the first
architecture that learned strategy beyond random.

### ai_v3
Stability and signal hardening, `impl_step1`–`impl_step10` (clean pipeline → obs features →
architecture → reward shaping → hyperparameters → active-state signals → effectiveness/move order →
item consumption → reward overhaul → adaptive training infra). Its 350M-step run reached ~70–75% vs
Heuristic and was ceilinged by the fixed bots.

### ai_v4
The data-quality + encapsulation chapter, `impl_step1`–`impl_step9`: own-team spreads, opponent
Hidden Power inference, damage attribution, the unified L=2 transformer extractor, the adaptive-LR
KL band, and the strict battle-API + event-sourced `TurnDelta` fold. *(Self-play was deferred out of
it into ai_v5.)* Open tail: pathology hunting, and Phase 5b (`todo_live_battle.md`).

> **Folder name is canonical.** The v4→v5 relocation bumped the *folder* names and the in-folder
> branding was reconciled to match across v5–v8; older git history may still show pre-relocation
> labels.

### ai_v5
Self-play / league play: snapshot pool with win-rate gating (Step 1, code landed), then exploiters /
PFSP / a two-pool stable (Step 2, forward design). Both prerequisites are designed here —
`design_reward_annealing.md` and `design_league_tooling.md`.

### ai_v6
Two routes to an **anticipatory** agent. The MCTS route (Step 5) is **superseded** — search is
confined to the L4 offline-teacher bucket by the owner's no-search-on-the-model constraint — though
its replay collection and the **team-completion model** landed. The favored route is Step 6,
`design_latent_predictive_representation.md`: a feedforward L3 auxiliary that makes one forward pass
anticipate one ply with no runtime tree, culminating in per-action outcome-token injection.

### ai_v7
Specialisation and ladder play: evaluate the generalist across the sample teams, fine-tune per top
team, take them to the ranked ladder; plus cheap shallow search in training.

### ai_v8
The conditioning / credit-assignment epoch on the v44 family: the public-info value aux, the
team-archetype latent + head FiLM (**the whole zarch family was DELETED at v78** — count dominates
conditioning), the discovery boosters, and `next_run_plan.md`. ⚠️ That plan predates the 2026-08-03
generation reset; generation-crossing items in it are superseded.

### ai_v9 (the entity-graph generation)
**The operative roadmap is `design_generation_roadmap.md`** — the fresh-generation reset (no old
checkpoints, position-equivariance first-class, adequacy judged generation-vs-generation by anchored
ELO) and the staged sequence: Stage 0 pointer-native head (SHIPPED, v51) → Stage 1 move tokens →
Stage 2 physics as attention edge biases + op-concat deletion → Stage 3 declarative schema + obs
re-home. The entity/edge INVENTORY stays in `design_entity_graph.md`. **E9 history is CLOSED OUT with
`gen3_frame_deletion_v1`** — the H-B event window replaced the TurnDelta lag frames; its open
reconciliation,
[`design_frame_deletion_coverage_gaps.md`](ai_v9/design_frame_deletion_coverage_gaps.md), carries
the standing rule that **a dV ablation says whether the model LEANS on a block, never whether each
FACT in it has a home elsewhere** — an irreversible deletion needs both readings. The
concat-deletion evidence, the OA1/OA2/PV and α/β forward designs and their gates:
[`designs_version_map_ai_v9_stages_and_forward_designs.md`](research_state/claude_md_archive/designs_version_map_ai_v9_stages_and_forward_designs.md).

### ai_v10 (OPEN — the exploiter-SCALING chapter)
Opened 2026-08-16. **Nothing built.** Where ai_v9 is the entity graph *inside* one battle, ai_v10 is
*what transfers between teams*: why exploiter competence collapses between N=10 and N=20 teams when
N=1..5 is trivial. [`design_exploiter_scaling.md`](ai_v10/design_exploiter_scaling.md) holds the
hypothesis (no transferable team-scoped abstraction ⇒ sample cost linear in N), the four accounts it
must beat (H_rate / H_capacity / H_conflict / H_coverage) and a pre-registered, unrun battery whose
Tier 0 needs no GPU. Two gen-12 measurements carry it: the bench enters as a team-health **SCALAR, not as structure**,
and team PACE class decodes from the raw obs on unseen teams while `pi_features` sits at chance —
**the abstraction is free in the input and the trunk discards it**. Four forward docs sit beside it: `design_flywheel_tick_tock.md`, `design_outcome_latent.md`,
`design_counterfactual_value_grounding.md` (the counterfactual label factory + the R1/R2/R3
critic-bias attacks) and `design_advantage_gated_distillation.md` (the DEEP-BRANCH fix for the fold
— it separates the distillation TARGET FORM from its JUDGE and contests flywheel **D-F**).

### ai_v11 (OPEN — the human-ladder-replay chapter)
Opened 2026-08-18. **Nothing built.** One doc,
[`design_human_replay_objectives.md`](ai_v11/design_human_replay_objectives.md): what an external
action distribution teaches, and what survives spectator replays being PARTIAL information. Its
spine is an **OOD taxonomy** — the opponent half of a replay obs is in its native distribution, our
half is not, and the request stream does not exist, so the legal mask is synthesised and
systematically over-permissive. Four rungs ordered by OOD-robustness: α/β on the
human OPPONENT's actions → outcome/value on human states → BC-regularization on the faithful subset
(the gen-17 candidate) → offline RL with team-completed acting sides. **Phase-0 census RUN**
(263,159 logs): fully-faithful decisions are **16.70%**, and a replay-reconstructed own team is
encoded with **`spread_known = 1.0` over a FABRICATION** — a wrong value asserted as known, feeding
the outgoing physics. Faithfulness is not missing-at-random, so any outcome-labelled objective needs
outcome-balanced weights.

### ai_v12 (🟢 THE LIVE CHAPTER — the clean-world / win-probability-critic era)
Opened 2026-08-29; code landed 2026-09-06, arm 1 running. **ai_v12 is what the win-probability head
becomes when it stops being a barometer and becomes the value function.**

- [`design_winprob_behavior_coupling.md`](ai_v12/design_winprob_behavior_coupling.md) — the plan of
  record: three routes turning the head into behavioural force, all BUILT and OFF. **Probe L fires
  the chapter:** the head ranks an alternative above the played action on **96.4% of immune whiffs**
  while the policy samples it at a median **p = 0.002**.
- [`design_winprob_only_critic.md`](ai_v12/design_winprob_only_critic.md) — **the design of record**,
  implemented as `gen3_winprob_critic_mode_v1`: `--critic {shaped,winprob}`, where `winprob` makes
  `V(s) = P(win|s)` with no approximation term (hence `--terminal-indicator` and `--victory-value
  1.0` are requirements). `shaped` is still the DEFAULT — **the default flip, its `ARCH_SIGNATURE`
  bump and §5.3's deletion list are a LATER commit, after an arm has run.**
- [`launch_runbook.md`](ai_v12/launch_runbook.md) — the three generation-scale arms **SPARSE /
  SELF-φ / FROZEN-φ**, identical but for where the potential comes from, ahead of them a paired 5M
  pre-test. `src/main/launch_runbook_test.py` parses its blocks OUT OF the document through the live
  parser, so a flag deleted anywhere fails a test naming this doc — but a runbook is not a
  registration; the arms are registered in the ledger.
- [`probe_risk_modulation_capstone.md`](ai_v12/probe_risk_modulation_capstone.md) — does a P(win)
  value function buy correct risk modulation? Three offline instruments with **frozen per-arm
  predictions**; a FLAT sparse slope falsifies "P(win) buys risk for free" and must be reported as
  loudly as a pass.

Also in the folder: the 40-team slate (`team_slate_40.{json,md}` + `team_slate_build.py`) and
`promotion_exclusions.json`, which `python -m main.promote_teams --regenerate-exclusions` rebuilds
from run metadata — it was first built from FROZEN ARGVS and went stale **with its union SIZE
unchanged**, so no count-shaped check saw it. **The era boundary the default flip implies is
CENSUSED, not decided** —
[`research_state/era_boundary_deprecation_2026-09-06.md`](research_state/era_boundary_deprecation_2026-09-06.md)
prices it (which flags become deletable, which runs stop loading, the ordered commit list with a
gate per commit); the flip waits on the live arm.

## `ops/` — operational procedure (SOP) documents, era-independent
[`ops/TRAINING_RUN_SOP.md`](ops/TRAINING_RUN_SOP.md) — how a run is launched (*it launches* vs *it is
the experiment*, two independent checks), watched (the four layers; the 55-minute fallback cron),
killed and relaunched, and read. [`ops/ORCHESTRATOR_SOP.md`](ops/ORCHESTRATOR_SOP.md) — the
orchestrator session's own: the three roles, dispatch, landing, banking, reporting and cadence, the
scope of unasked action, and the agent-stall + background-waiting mechanics (§7). **Both are the
PROCEDURES OF RECORD for the two long-lived sessions** (2026-09-07: the rules moved out of session
memory, which now holds only pointers), so an owner ruling about how a session operates is written
into the owning section there, not into a memory file.
[`ops/TECH_DEBT_BACKLOG.md`](ops/TECH_DEBT_BACKLOG.md) — the one tech-debt list; nothing on it is
dispatched without the owner's word.

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
  agents.model.flag_registry`, `--check` is the gate): every model-relevant extractor toggle with
  its TIER (`cli` / `config_only` / `constructor_only`), CLASS, default, `since` version and
  meaning, plus why a settled flag can lose its CLI entry without losing explicitness. Read it
  before adding or demoting a toggle; the rules live in `src/agents/model/CLAUDE.md`.
- **`design_pathologies.md`** — the living model-pathology register: *what's wrong → what we
  changed → what we expect next time*. **Review it before every retrain**, and add a row after each
  eval saying whether a fix's predicted change landed.

- **`research_state/UNDERSTANDING.md`** + **`research_state/ledger.md`** — the research pair, in the
  always-current table at the top of this file. The view is present tense; the ledger is how a
  belief changed, is append-only, and wins any disagreement.

## `learning/` — concept explainers (version-agnostic)

`designs/learning/` holds **durable teaching notes** — one file per major concept, each a two-level
explainer (intuitive → technical, no code) grounded in *our* architecture (flags, `ARCH_SIGNATURE`s,
obs blocks, real file names). They are **always-current reference docs**, not version-keyed impl
records: if the architecture changes such that a note is wrong, fix it in the same pass. The
`/gen3ai-learning` skill creates and maintains them. **Each note is its own table of contents — open
it rather than trusting a one-line summary.**

| note | what it owns |
|---|---|
| `entity_tokens_biases_pointers.md` | the ai_v9 vocabulary — equivariance, the **sorting rule** for where a fact lives (token / edge / summary / attention), the `DamageOperator` as a differentiable expert, the head funnel, and **§6.9, what stays POSITIONAL in the end state** |
| `shortcut_learning_and_feature_delivery.md` | the input side — gradient starvation, amortization vs bottleneck, the **axis rule**, the four tests that separate laziness from genuine use |
| `objective_richness_and_representation.md` | its output-side dual: what a richer objective buys a representation |
| `marginalization_and_uncertainty.md` | marginalize vs mean-field, Jensen, the threshold/tail problem (P(KO), P(outspeed)), the convex-combination primitive |
| `credit_assignment_and_value_errors.md` | GAE's λ, bootstrap error propagation, PopArt / `vf_coef` as trunk arbitration, the FOUR critic-failure causes and the instrument for each |
| `win_prob_decomposition.md` | the five-axis taxonomy of "the critic was wrong" — luck, calibration vs RESOLUTION, the population sign-flip, the IRREDUCIBLE hidden-information floor, the epistemic layer |
| `imperfect_information_and_equilibria.md` | information sets, CFR vocabulary, the PUBLIC BELIEF STATE, α as a trained fixed point, and the scope cut (nothing models what OUR actions reveal) |
| `population_game_theory.md` | strength as a MATRIX — payoff matrices, Nash averaging, spinning tops, PSRO, exploitability bounds; owns "flat ELO: converged or circling a cycle?" |
| `on_policy_self_distillation.md` | OPD as the dense-signal regime, why it is ~7–10× more step-efficient than PPO, `better-line` as the improvement teacher |
| `negative_transfer_and_shared_functions.md` | why a fold on eight teams moves the other 711 — GIFT and LEAK as one displacement, sign = teacher CONTENT, magnitude = DOSE |
| `continual_learning_and_forgetting.md` | forgetting as an optimization problem, the three fix families and which we run, the three-cause decay diagnostic |
| `quality_diversity_and_open_endedness.md` | the archive view of the flywheel — MAP-Elites, descriptor choice as THE decision, POET, stepping stones |
| `activation_functions.md` | the three nonlinearity TIERS, and why an activation swap is retrain-class yet weight-shape-NEUTRAL (`check_compatible` cannot see it) |
| `vacuous_tests_and_guards.md` | the failure whose symptom is indistinguishable from success — the seven-way taxonomy, the **arranged-vs-encountered** rule, the five design rules that make each class unrepresentable |

Also here: `amortization_gap_and_conditioning.md`, `conditioning_architectures.md`,
`distillation_flywheel_lessons.md`, `exercises_and_reading.md`,
`generalist_specialist_amortization_gap.md`, `latent_belief_metrics_and_collapse.md`,
`pbs_value_functions_and_search.md`, `popart_value_scale_and_currencies.md`,
`regularization_and_noise_in_ppo.md`, `rl_concepts_studyguide.md`,
`self_discovered_archetype_latent.md`, `temperature_mixing_and_risk.md`.
