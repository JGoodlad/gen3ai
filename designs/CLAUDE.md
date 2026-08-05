# designs/ — Version Map

This file tells Claude which `ai_vN` folder is relevant when reading or writing design
docs. Read it whenever you're about to touch anything in `designs/`.

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

**Current state as of 2026-08-05:**

| What | Version | Notes |
|------|---------|-------|
| **Active training run** | **ai_v9 gen-2** | `run_20260805_060807` (worktree `gen2-run-0805` @ ffa851e): the FULL entity stack — all 11 edge families (d1,d2,d3,d4,s1,s3,v,t,x,g,c4) + E5 tail seats, 40M steps, fresh lineage. **Gen-1 COMPLETE** (`run_20260804_090512`, 40M, Bots 90.9% / Pool 76.0% final; 6 launch families) — its end-of-run edge audit (`edge_audit_40M.json`) showed the edges became load-bearing (all-off = 26.9% action flips). Judged gen-2-vs-gen-1 by anchored ELO. The old ai_v8 lineage sits behind the ai_v9 signature wall. |
| **Code on main** | **ai_v9 (v57)** | `MODEL_CONFIG_VERSION` **57**, `ARCH_SIGNATURE` **`gen3_edge_bias_trunk_v1`**. Stages 0–2 in: v51 pointer-native head, v52/53 typed-HP belief, v54 move-entity seats (E3/E4), v55 op block trim, v56 edge-bias trunk (now 12 families incl. the C1 post-boost consequence edge), v57 E5 tail seats. Stage-3 first half (the declarative obs schema view) shipped. |
| **ai_v9** | **Stages 0–2 SHIPPED + Stage-3 half** | Roadmap: `design_generation_roadmap.md` (the operative staged plan, slice statuses current). Still open: the op head-concat deletion (needs a concat-zeroing audit arm — the family audit alone doesn't measure redundancy), C1b/C2/C3/C5 consequence edges, Stage-3 generator half + the entity re-home (retrain-class, owner go/no-go), and E9 history. |

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
FiLM (v44 `gen3_zarch_film_v1` — the amortization-gap storage fix), the discovery boosters
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

---

## Folder conventions

Each version folder has:
- `todo.md` — in-progress checklist; `✓ DONE` marks completed steps
- `impl_step*.md` — post-implementation records (what was built, constants set, files
  changed); these are the primary targets for `gen3ai-update-design-docs`
- `design_*.md` — forward-looking design docs written before implementation

When writing a new `impl_step*.md`, match the existing docs in that folder exactly —
heading levels, table style, and level of detail vary between versions.

## Cross-version docs (designs/ root)

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
  the differentiable `DamageOperator`).
- **`entity_tokens_biases_pointers.md`** — the ai_v9 concept vocabulary: what entity-based
  (entity-centric / relational) modeling is and where it came from (CNN weight-sharing →
  GNNs → Deep Sets → Transformers → AlphaStar/AlphaFold), why permutation equivariance beats a
  flat positional vector (weight sharing, hypothesis-space reduction, whole bug classes made
  *unrepresentable*) and what it costs, the **sorting rule** for where a fact lives
  (token / edge / distribution summary / attention), how expected damage is delivered (the
  `DamageOperator` as a *differentiable expert* whose gradient trains the move belief; the
  shipped v51 `pointer_cells` route vs the Stage-2 edge-bias route), and how **history** is
  represented once time stops being positional (recency-on-entity → turn tokens → entity-linked
  event tokens; recurrence ruled out by the event-log-purity invariant).
- **`shortcut_learning_and_feature_delivery.md`** — the input-side dual of
  `objective_richness_and_representation.md`: whether feeding a computed feature straight to the
  head makes the model "lazy," and when that is a plus. Gradient starvation (not "simplest
  function"), the ~1-bit-per-game RL amplifier, **amortization vs. bottleneck** (sufficiency for
  the decision is the whole variable), the **axis rule** ("never collapse an axis you must choose
  along" — the v30→v39 progression), the four tests that discriminate laziness from genuine use
  (ablation-KL / trunk linear probe / behavioural counterfactual / held-out), the measured P1
  ablation surprise (the model **ignores** collapsed summaries when un-collapsed ones sit beside
  them), and the reframe "make the lazy path the correct path" (= what v51's pointer head does).
- **`on_policy_self_distillation.md`** — on-policy distillation (OPD) as the dense-signal training
  regime, why it's ~7-10× more step-efficient than PPO (a full target distribution per state vs ~1
  bit/game), our `better-line` beam as the policy-improvement teacher, upgrading the `search-teacher`
  AWR-toward-`A*` to a full-distribution `KL(π' ‖ π)` (with the `V^{π*}`/GAE-bias dissolved by
  distilling the *policy* while the critic sees only confirmed returns), cheap Gumbel-top-k × opp-axis-
  collapse search under the expensive `DamageOperator` critic (≈8 evals/node vs 121), and the
  **team-subset exploiter** as where OPD compounds. Grounds ExIt/AlphaZero, Grill 2020, Gumbel MuZero,
  ReBeL/Student of Games onto our tooling.
