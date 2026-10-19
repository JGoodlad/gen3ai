# `designs/learning/` — concept explainers

Durable teaching notes, one file per major concept: a two-level explainer (intuitive, then technical, no
code) grounded in our architecture. **Always-current**: when the architecture changes so that a note is
wrong, fix the note in the same pass. The `/gen3ai-learning` command creates and maintains them, and adds
a row here for a new note. Each note is its own table of contents: open it rather than trusting the
one-line summary below.

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
`distillation_flywheel_lessons.md`, `exercises_and_reading.md`, `exercises_2026-09-14.md`,
`generalist_specialist_amortization_gap.md`, `latent_belief_metrics_and_collapse.md`,
`pbs_value_functions_and_search.md`, `popart_value_scale_and_currencies.md` (PopArt itself is deleted),
`regularization_and_noise_in_ppo.md`, `rl_concepts_studyguide.md`,
`self_discovered_archetype_latent.md`, `temperature_mixing_and_risk.md`.
