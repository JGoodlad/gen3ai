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

**Current state as of 2026-08-10:**

| What | Version | Notes |
|------|---------|-------|
| **Active training run** | **ai_v9 gen-6** | `ai_v9_07_gen6_seed_vicreg_0810` (launched 2026-08-10): **the seed-VICReg arm** — gen-5's exact config + `--value-seed-vicreg-coef 0.1` (v62), the one trigger-fired lever from gen-5. Pre-registered acceptance: `value_seeds/out_effective_rank` rises toward k=4 within ~2M (gen-5 flatlined at 1.0 the whole run) AND offline anchored ELO @24M ≥ gen-5's 2065±30; seeds un-collapse WITHOUT an ELO move ⇒ multiplicity was insurance, not headroom — also a clean answer. Predecessor: gen-5 `ai_v9_06_gen5_no_concat_0809` (**COMPLETE**, 25M, final bots 90.8%, offline ELO **2065±30 @24M** / best 2071 @20M — **PARITY with gen-4's 2096**, CIs overlap: the concat deletion cost nothing and gained nothing, with the critic's k=4 seed readout COLLAPSED start-to-finish). **Its `model_config.json` is what "the production configuration" means** and `designs/production_config.json` is a verified byte-identical copy of it — [`ARCHITECTURE.md`](ARCHITECTURE.md) is derived from that file. Predecessors: gen-4 `run_20260808_212910` (COMPLETE, 25M, v60 entity re-home — the run whose stratified end-of-run audits justified the concat deletion), gen-3 `run_20260807_135637_gen3` (40M, 15 edge families; at 32.0M bots 0.911 / anchored ELO 2094), gen-2 `run_20260805_060807` (11 families), gen-1 `run_20260804_090512` (40M, bots 90.9% / pool 76.0%, 6 launch families; all-edges-off = 26.9% flips). The old ai_v8 lineage sits behind the ai_v9 signature wall. |
| **Code on main** | **ai_v9 (v62)** | `MODEL_CONFIG_VERSION` **62**, `ARCH_SIGNATURE` **`gen3_no_concat_v1`** — **v62 `gen3_seed_vicreg_v1`** adds `--value-seed-vicreg-coef` (the VICReg variance+COVARIANCE floor on the `MultiSeedValueReadout` outputs — resume-immutable, cov-term gate-tested, TB family renamed `seeds/*` → `value_seeds/*`) plus the INERT pair-reduction rungs (`pair_reduce.py`, constructor-only, production builds nothing; **G1 n=299 killed the delivery/G7 line for gen-6** — no rung beats R0 beyond seed spread — and step-0 `--site op` measured the reduced stats at 65.07% flips with the block's dims 85–660 write-only; ledger 2026-08-10). v61 (2026-08-09, `6aac795`): **the op head-concat is DELETED** (2026-08-09, `6aac795`): `ProjectionAssembler` no longer appends the 660-dim flat block to either head (pi_projection 1131 → 471, vf_projection 875 → 471), and the critic's replacement is `MultiSeedValueReadout` — k=4 × 64 learned seed queries cross-attending the op's per-our-mon rows, vf-only (+256), shipped **with** a `seeds/*` TensorBoard collapse contract (query/output cosine, uncentered effective rank, VICReg variance target, pre-registered trigger). The op itself lives on: pointer cells, prefuse injection, 15 edge families, `last_raw_block` for the probes. Predecessors: v60 Stage-3 entity re-home (obs 2925 → 2667), v51 pointer-native head, v52/53 typed-HP belief, v54 move-entity seats (E3/E4), v55 op block trim, v56 edge-bias trunk (15 families), v57 E5 tail seats, v58 the SpD-as-speed GIGO fix, v59 K=6 everywhere, E9 step 1 (`gen3_entity_recency_v1`). |
| **ai_v9** | **Stages 0–2 SHIPPED + Stage-3 half** | Roadmap: `design_generation_roadmap.md` (the operative staged plan, slice statuses current). **The op head-concat deletion is DONE (2026-08-09, `6aac795`) — it was the last of the stated goals and it landed on evidence, not on schedule:** gen-4's stratified end-of-run audits showed `FULL_CONCAT` net policy dependence **+0.00%**, all-edges-off flips (29.22%) **exceeding** the concat arm (22.70%) for the first time in the lineage, and `act_threat` still decodable from `vf` with the concat zeroed (r² 0.400 → 0.418), so the remaining `|dV|` 4.75 was trained reliance on an open window rather than structural necessity. Still open: C1b/C2/C3/C5 consequence edges, E9 history, and OpTensors steps 1–2 (typed views, recompute dedup) — deferred honestly to background work during gen-5, since the §9.1 evidence showed the removal was not waiting on them. **NEW forward design (not built): `design_conditional_opponent_cells.md`** — the magnitude rule for the entity world + the OA1 conditional threat cell (defensive pivot) and OA2 switch-branch move cell (punish the switch), plus PV pair-value attention (a critic route), the unrevealed-marginalisation prerequisite and pre-registered gates. **RESOLVED 2026-08-09 — read the two-route precondition below as history.** The 2026-08-08 amendment required OA1 (policy) **and** a critic route (PV *or* generalized token-content injection) to land before the concat could die, accepted only on **flips AND `|dV|`**. What actually happened: the **flips** half was met by training alone (all-edges-off 29.22% > concat 22.70% on gen-4), the policy side needed no OA1 at all (net concat dependence +0.00%), and the critic route that shipped was **neither** of the two candidates — it is `MultiSeedValueReadout`, readout **multiplicity** rather than width (P3 refuted width, never multiplicity). `|dV|` remained concat-led (4.62 vs 2.44) and was overridden on the conditional-coverage evidence above rather than waited out. **OA1/OA2 and PV therefore survive as forward designs on their own merits, no longer as preconditions for anything.** **OA1/OA2 are pointer CELLS, not edge families** — do not confuse them with the C1-C5 consequence edges. **NEW forward design (not built), 2026-08-11: `design_opponent_intent.md`** — the build for one sentence the model cannot express: *"they are likely to click **this**, so **this** is my answer."* Supplies the two things the pair-reduction operator was missing: a **distribution worth weighting by** (`α`, a SUPERVISED usage belief over their K believed move seats + `SWITCH`; `β` over their team slots, conditional on `SWITCH`) and an **outcome vector worth weighting** (one unified `pair_in` carrying damage AND status AND `neutralization` AND `tempo_cost` — today damage and status are computed in two functions with two reductions, and one `α` cannot weight two tensors). **The three-part framing is the doc's core claim:** a distribution + a rich outcome vector + the weighting done per-action before the logit — missing any one makes the other two useless, which is why **G1-FINAL's null was near-guaranteed** (it tested part 3 alone, on damage-only cells, with `w` substituted for `α`). Grounded in the gen-4 end-of-run edge audit: `d2` 19.25% / `d1` 12.17% (our offense) vs **`d3` 0.63%** (their believed threat, DOWN from 1.9% at gen-3 9.6M) — the entity system is overwhelmingly offensive because the anticipatory half is routed through edges, which carry a softmax-normalised RATIO and cannot deliver a per-action absolute. **Two owner reconciliations settled 2026-08-11:** (1) *both sides anticipate* — `α` may not depend on our REALIZED action but MAY depend on our POLICY, and since the policy is a function of the board, `α = f(board)` is already the right form; the forced change is that **`α`'s INPUT must include OUR outgoing physics** (`d1`/`d2` grids), and the fixed point is found by TRAINING (self-play), never solved at inference — reading our own policy logits would be level-3 but creates a forward-pass cycle, so it is deliberately not taken; (2) *belief-derived seats*, now governed by a **HARD OWNER CONSTRAINT (2026-08-11): the model must always pick among the belief's DISCRETE states and may never invent a move — interpretability is the reason.** So `α ∈ Δ^(K+1)` over named seats + `SWITCH`, no `UNKNOWN` slot and no learned property head (both proposed in earlier drafts and CUT — §4.6 keeps their causes of death). **ONE rule on both axes: "if we can't name it, we don't train on it"** — hard target when the belief holds it, **masked** otherwise, mask rate logged as a first-class diagnostic. (A property-similar soft-target scheme was drafted for the move axis and CUT: it yields a smeared object rather than the clean `P(seat | modeled)`, it injects the belief's non-random blind spots as a bias invisible in `α`'s accuracy, its similarity metric has no principled setting, and it was an unjustified asymmetry against `β`, which masks. Its motivating example also dissolved — under canonical-id matching a bare-`hiddenpower` seat MATCHES a used HP Ice.) Matching is by canonical id, never by index (seats permute per turn — the Hungarian precedent). **The division of labour that buys:** `α`/`β` own *which of the things we believe*, the belief head owns *whether we believe the right things* — two failure modes, two measurements, instead of one head absorbing the other's errors. **`β` answers "switch to WHOM"** — discrete over alive/non-active/revealed slots, masked in v1 when they bring an unrevealed mon (rate logged; **B1**, BUILT-but-never-run, is the named upgrade that turns that mask into a posterior soft-target). **`β` is also what makes the (bench × bench) offense grid actionable** — that grid alone is an unweighted outer product; with `β` it answers *"if I bring Skarmory and they pivot to Blissey, is Skarmory still doing anything?"*, so `d2`/`d5` are not independent cheap wins but the grid `β` needs. The RL loss is **stop-gradiented** out of both heads so a null is interpretable. **The constraint's real cost is a ceiling at belief quality, so §4.5 audits the WHOLE BELIEF STACK against the live config — seven legs, and EXACTLY ONE is supervised.** B-move (which moves they hold) **ON but UNSUPERVISED** (`move_belief_coef` `0.0`; `known_moves` already emitted + plumbed, BCE unconsumed — shaped only by the Smogon prior + RL gradient); B-hptype ✅ supervised 0.05, acc ≈0.91; B-spread ❌ OFF; B-team (B1) ❌ OFF (BUILT, never run); B-latent ❌ OFF; **B-item and B-ability are STATIC lookups that cannot improve with training** (`p_cb` = a species usage prior collapsing to 0/1 on reveal; abilities = Smogon per-species priors). **B-spread is a PHYSICS DEFECT, not a missing signal:** with it off the op prices every opponent's offense as `(2·base_atk + const) × 1.1` — 252 EV + boosting nature, uniformly, at **nine sites** (`damage_op.py:1727` et al) — so `pair_in` is computed against a fictional maximally-invested opponent, and the over-estimate scales with base stats so it distorts the RELATIVE threat ordering, not just the level. A better `α` over de-timid physics inherits the distortion ⇒ **B-spread is a correctness fix to component 1, not a third belief leg to stack.** Operationally they differ: `--move-belief-coef` is **training-only / resume-mutable**, `--spread-belief` is **STRUCTURAL / version-checked / FRESH-ONLY** (cannot join a running generation). **G2a runs FIRST and needs no head at all** (how often does the top-K hold what they clicked?); then G2b (does `α` beat `w`, `β` beat the alive-bench base rate?); G3b asserts the discrete constraint as a TEST. **G0–G7 need no training run.** Not this doc: physics mutation (Marvel Scale changing the whole matrix) is explicitly out of scope for a one-ply reduction. §9.1 records that the G1-FINAL SKYLINE is likely underpowered (2800 params on ~239 rows at L2 1e-3) and should not be read as "the grid is exhausted" until re-conditioned. **§7a (REVIEW, 2026-08-11) adds four notes:** (1) ⚠️ this is a **POLICY-side** design and the measured regression is **CRITIC-side** — the dense frozen-vs-frozen ladder reads **gen-4 2081±11 vs gen-5 2037±11 (DISJOINT)**, so the concat deletion probably cost **~44 Elo** and the sparse `eval/elo` at ±30 that reported "parity" simply could not resolve it; (2) the **critic route that falls out of this design** — send the same `Σ_k α_k·pair_in[k,j,:]` row to the critic as **token content on our mon j's token**, pooled by `value_cls`: equivariant in BOTH axes (`α` invariant under permuting their moves since `g` is shared over `k`; the row rides mon `j`'s token; attention pooling is permutation-invariant), **no seeds** (which matters given seed collapse measured at ~1 effective direction under BOTH VICReg and quantile pressure), and testable BEFORE `α` exists by substituting `α := normalize(w)` (the shipped R1 rung) — separating the DELIVERY claim from the DISTRIBUTION claim; (3) an **alternative hypothesis for `d3` = 0.63%** the doc does not raise — a channel carrying DISTORTED content also reads low, and the de-timid defect corrupts exactly the relative threat ordering `d3` conveys, so **re-measure `d3` after the B-spread fix before concluding the channel was the problem**; (4) the **coverage-risk fallback pre-registered**: `α ∈ Δ²` over {ATTACK, SWITCH} only — belief-free, zero mask rate, carries the largest single effect — ship it if G2a returns poor coverage. Plus schedule honesty: step 0b is fresh-only, so this is **gen-8 (foundation) + gen-9 (`α`)**, not one generation. **NEW forward design (not built), now scoped to RUN BESIDE GEN-5 and land at the gen-6 boundary: `design_pair_reduction.md`** — the deep spec of the one line `design_op_tensors.md` §3.2 sketches as `REDUCE(pair_in, over=MOVE_AXIS, how=…)`. Splits **contract** from **knob**: a weighted reducer must emit ONE distribution over the move axis per defender, shared across every channel, which kills the incoherence defect structurally (the flat/trunk block takes NINE independent maxima, so up to nine different opponent moves describe one defender). **§2.1 (added 2026-08-10, verified against the live tree) makes that understatement precise and WORSE on the path that decides:** the pointer **switch cell is 15 numbers — ten damage, `p_outspeed`, `provenance`, and the Choice-Band tail — with NO status coordinate in any currency at all.** In production `threat_status_refine` is `False`, so incoming status reaches the policy *only* as the `s3` edge family — an attention **bias**, i.e. a softmax-normalised RATIO. So "they'll click a status move, so bring the Natural Cure mon" is unrepresentable not because status was mis-reduced but because **the two quantities never meet in the same vector in the same units** — a CURRENCY failure one level below the reduction failure, which no reducer fixes. **This is the most likely reading of the G1 n=299 null** (no rung beats R0 beyond seed spread): the ladder was asked to improve an aggregation over a vector that never carried the quantity the decision turns on. §9a adds the admission test for new message coordinates (name two actions it flips) and the derivability rule + its gradient-starvation counter-rule; §2.1 names `neutralization` and `tempo` as the two missing ones, with `physics mutation` (Marvel Scale) explicitly out of scope for a one-ply reduction. Ladder R0 hard_max (byte-identity anchor, stays shipped) → R1 belief mean → R2 learned / Deep-Sets → R3 multi-aggregator default (GIN/PNA: no single aggregator suffices) → R4 = OA1. **Two claims carry it:** `α` must be computed from the board ALONE — not per defender, because they choose without seeing your switch — which kills the channel AND defender incoherence with one restriction; and **hedging is not a depth phenomenon** — taking the middle ground under uncertainty needs a *second moment* (`Σ α[o, o²]` ⇒ variance ⇒ a learned risk attitude), which `max` structurally cannot produce, so it is reachable one-ply. Also names what the doc is NOT: the "is this my last answer" scarcity feature is a different arity ([their MON × my mon], reduced over OUR axis) that the OpTensors typing rejects as a shape error (§11). Gates G0–G7; **steps 0–7 need no training run**, so the whole ladder is decidable *beside* gen-5 on frozen checkpoints — the one real cost of the concat having died the same day is that gen-5 trains against R0 `hard_max` and cannot change mid-run, so the earliest a better reducer ships inside a generation is **gen-6**. **ADOPTED 2026-08-10 — §8.1 is the plan of record**: step-0 carries a pre-registered downscope rule (unsuppressed `imx_CELLS` ≲7% ⇒ cheap rungs only), **seed VICReg is a named prerequisite of the critic route** (gen-5's `seeds/*` measured the k=4 readout COLLAPSED — `out_effective_rank` 1.0 sustained — so the trigger fired and `--seed-vicreg-coef` is being wired for gen-6), and G7 runs in the first post-gen-5 GPU window. **Step 0 is new and comes first: re-run the split audit on gen-5** — with no concat there is no competing route, so `imx_CELLS` finally means what it says (on gen-4 it read 6.53% shuffle *while* `FULL_CONCAT` carried the traffic). **Step 1 is already DONE on main** — `damage_op.py:534` `_chan_max(..., how="hard_max")` is the single named call site and any other `how` raises. **G7 — a single-team exploiter A/B with a behavioural-bifurcation readout — is the capability gate** (owner design 2026-08-09; fixed team = **Big Five + Starmie**: Tyranitar/Blissey/Gengar/Swampert/Starmie/Skarmory, chosen because every slot's value is branch-dependent; deliberately no capacity-matched arm, per P3 + the LUT nulls). §10.8 retracts an earlier VoI-ceiling argument of mine that was simply wrong. **Reading aid: the delivery digraph is browsable** — [`architecture_viewer.html`](architecture_viewer.html) via `file://`, or served live at `model.g5d.io` (`python -m agents.model.build_arch_viewer --serve`, which re-renders from the checkout on every request). Edge hue = what the channel physically carries, thickness = measured dependence at a selectable checkpoint, plus a path filter for "what does the critic read" — the fastest way to see the concat's critic-side residual above. It is **generated** from the committed graph snapshot + `research_state/measurements/`, so rebuild it with `python -m agents.model.build_arch_viewer` rather than editing the HTML; `--check` fails on drift. |

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
