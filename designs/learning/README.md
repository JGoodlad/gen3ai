# `designs/learning/` — concept explainers

Durable teaching notes, one file per major concept: a two-level explainer (intuitive, then technical, no
code) grounded in our architecture. **Always-current**: when the architecture changes so that a note is
wrong, fix the note in the same pass. The `/gen3ai-learning` command creates and maintains them, and adds
a row here for a new note. Each note is its own table of contents: open it rather than trusting the
one-line summary below.

**Status key.** *current*: its claims about our model hold. *banner*: the concept stands but the note carries a
dated `Correction` banner at the top (read it first; it names what was deleted or changed, at which version, and
points to `designs/ARCHITECTURE.md`). *SUPERSEDED*: the machinery the note analyses is deleted; read it as a
historical teaching record. The last full sweep was 2026-10-25 (every note checked against `ARCHITECTURE.md`,
`designs/deleted_flags.md` and the code). A banner's date is the date its claims were last checked.

| note | status | what it owns |
|---|---|---|
| `entity_tokens_biases_pointers.md` | banner | the ai_v9 vocabulary — equivariance, the **sorting rule** for where a fact lives (token / edge / summary / attention), the `DamageOperator` as a differentiable expert, the head funnel, §6.9 (what stays POSITIONAL in the end state); the observation size is read from `Gen3ObservationEncoder.get_layout()`, never quoted |
| `shortcut_learning_and_feature_delivery.md` | banner | the input side — gradient starvation, amortization vs bottleneck, the **axis rule**, the four tests that separate laziness from genuine use |
| `objective_richness_and_representation.md` | banner | its output-side dual: what a richer objective buys a representation |
| `marginalization_and_uncertainty.md` | banner | marginalize vs mean-field, Jensen, the threshold/tail problem, the convex-combination primitive |
| `credit_assignment_and_value_errors.md` | banner | GAE's λ, bootstrap error propagation, `vf_coef` as trunk arbitration, the FOUR critic-failure causes and the instrument for each (written about the deleted shaped critic) |
| `win_prob_decomposition.md` | banner | the five-axis taxonomy of "the critic was wrong" — luck, calibration vs RESOLUTION, the population sign-flip, the hidden-information floor, the epistemic layer |
| `imperfect_information_and_equilibria.md` | banner | information sets, CFR vocabulary, the PUBLIC BELIEF STATE, α as a trained fixed point |
| `population_game_theory.md` | banner | strength as a MATRIX — Nash averaging, spinning tops, PSRO, exploitability; "flat ELO: converged or circling a cycle?" (the flywheel it maps onto was never built) |
| `on_policy_self_distillation.md` | banner | OPD as the dense-signal regime, `better-line` as the improvement teacher (the `--opd-coef` / `--search-teacher` losses are deleted) |
| `negative_transfer_and_shared_functions.md` | banner | why a fold on eight teams moves the other 711 — GIFT and LEAK as one displacement |
| `continual_learning_and_forgetting.md` | banner | forgetting as an optimization problem, the fix families, the three-cause decay diagnostic |
| `quality_diversity_and_open_endedness.md` | banner | MAP-Elites, descriptor choice, POET, stepping stones |
| `activation_functions.md` | banner | the nonlinearity TIERS; why an activation swap is retrain-class yet weight-shape-NEUTRAL |
| `vacuous_tests_and_guards.md` | banner | the failure indistinguishable from success — the taxonomy, the **arranged-vs-encountered** rule, the five design rules |

Also here:

| note | status | what it owns |
|---|---|---|
| `amortization_gap_and_conditioning.md` | banner | the generalist/specialist gap argument and the conditioning design it ended in (the machinery is deleted; the argument stands) |
| `generalist_specialist_amortization_gap.md` | banner | the marginal trap: why self-play averages team-conditional play away (most named levers deleted) |
| `conditioning_architectures.md` | **SUPERSEDED** (v78) | FiLM / LoRA / MoE, where and on what to condition, the sample-economics and SNR argument, the ill-conditioning story |
| `self_discovered_archetype_latent.md` | **SUPERSEDED** (v78) | the z_arch latent: VIB, β, UMAP, the self-discovery spectrum, "tune a regularizer on held-out signal" |
| `regularization_and_noise_in_ppo.md` | **SUPERSEDED** as an example (v78, v131); the PPO-ratio argument stands | why dropout breaks PPO's importance ratio and a deterministic pathway does not |
| `popart_value_scale_and_currencies.md` | **SUPERSEDED** (PopArt deleted v131) | value-scale currencies, taught with PopArt as the case |
| `distillation_flywheel_lessons.md` | banner | the v8 reproduction campaign: what a fold measurably does, the instrument lessons |
| `latent_belief_metrics_and_collapse.md` | banner | collapse diagnostics (effective rank, std floors, VICReg) for latent heads |
| `pbs_value_functions_and_search.md` | banner | the study guide for imperfect-information search, PBS, CFR, ReBeL and the value function |
| `temperature_mixing_and_risk.md` | banner | temperature, mixing and risk in the policy distribution |
| `rl_concepts_studyguide.md` | banner | glossary: PPO, pathologies, curriculum, uncertainty, game theory, search, interpretability |
| `exercises_and_reading.md`, `exercises_2026-09-14.md` | banner / current (one-line fixes only) | study plans (toy builds and reading); not architecture claims |
