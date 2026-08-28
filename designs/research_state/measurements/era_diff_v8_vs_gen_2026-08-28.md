# Era diff — the v8 exploiter-distillation arc vs the gen-era R2 fleet (2026-08-28)

**Question (Probe C).** Parent rigidity is REFUTED as the explanation for why v8's fold converted
(+69 anchored ELO; per-team piloting on the taught teams 0.438 → 0.710) while the gen era's
converts at ~13% (+1.6pp pooled from ~+11.65pp/slice teachers) with UNDIFFERENTIATED teachers
(`plasticity_forensics_v8_vs_gen_2026-08-28.md`). **What else differs?**

**Method.** Read-only archaeology. Every recipe fact below is quoted from a run's own
`metadata.json` — `cli_args` (the resolved argparse namespace the process actually ran with) is
preferred over `original_command` wherever both exist, because it records DEFAULTS as well as
typed flags. Architecture facts come from `model_config.json`, the param censuses in
`plasticity_forensics_v8_vs_gen_2026-08-28.json`, and live source. v8-era code was read via
`git show <era-hash>:<path>` without checking anything out. **No model was loaded and no training
was run.** Nothing under `models/` was written.

**Headline.** The two eras differ on far more than the record believed. The single largest
difference, and the one the ledger has an active *incorrect* entry about, is the **exploiter's
opponent regime**: v8's forks trained against a 50/50 bot mix with a win-rate-ratcheted difficulty
curriculum that all three ran to completion; **no gen-era fork — not the R2 fleet, not tick-1's
tocks — has ever had either.**

---

## 🔴 A CORRECTION TO THE RECORD, FIRST

`ledger.md` "CORRECTION 3 OF THE EVENING (amends 1022747's regime clause; owner-verified) — the
opponent regimes were IDENTICAL too (2026-08-25)" states:

> Recorded configs, both eras: v8 exploiters ran `team_pfsp off`, NO pool opponents, NO pfsp —
> opponents were the frozen self-target + **exploiter_bot_fraction 0.5** (half the episodes vs
> the 9 scripted bots). OUR tocks inherited the SAME 0.5 bot fraction […] **The opponent-variety
> leg of the narrowness hypothesis is dead.**

**That is false.** `exploiter_bot_fraction` is INERT unless `--exploiter-keep-bots` (an argparse
`store_true`) is passed — `src/agents/training/wrappers.py:380` reads
`if self._exploiter_keep_bots and self._rng.random() < self._exploiter_bot_fraction:`. The 0.5
appears in every run's `cli_args` because it is the flag's *default*, not because it was active.

| run | `exploiter_keep_bots` | `exploiter_bot_fraction` | `bot_weights` | `heuristic_floor` |
|---|---|---|---|---|
| ai_v8_06 / _09 / _13 | **True** | 0.5 (LIVE) | `aggressive_v2=3,heuristic2=3` | 0.2 |
| ai_v9_31 tock-1a · _32 tock-1b · _36 tock-1c | **False** | 0.5 (inert) | None | None |
| ai_v9_30 rev1_exploit | **False** | 0.5 (inert) | None | None |
| ai_v9_53–57 R2F5a–e | **False** | 0.5 (inert) | None | None |

Same for the difficulty curriculum: `exploiter_temp_start` is `5.0` with
`exploiter_temp_mode: "ratchet"` in all three v8 forks and `None` / `"fixed"` in every gen fork.
The v8 forks each left a run-dir artifact proving the ratchet ran —
`models/ai_v8_{06,09,13}_*/exploiter_temp_state.json` = `{"temp": 1.0, "n_ratchets": 16}` — and
**no gen run directory contains that file at all.**

The retraction chain (`3c12508` → `1022747` → CORRECTION 3) therefore closed the
opponent-variety branch on a misread of a default value, and the programme that followed (the
budget law, the breadth ladder, rev-3's two supply levers) was steered by that closure. **The
opponent-regime leg is reopened, and it is candidate C1 below.**

---

## Axis 1 — Exploiter recipe (the teachers)

| item | v8: ai_v8_06 / _09 / _13 | gen: ai_v9_53–57 (R2F5a–e) | provenance |
|---|---|---|---|
| init (`--model`) | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` @ **277,178,472** | `ai_v9_29_rev1_0823/final_model.zip` @ **25,067,760** | `cli_args.model`; ckpt filenames |
| `--exploiter` target | `models/ai_v8_04_distill_4teacher_0722` (the parent run dir = the init) | `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` (parent −1.07M) | `cli_args.exploiter` |
| teams pinned | **3 / 10 / 10 = 23 team files, 23 distinct** | **2 each = 10 slots, 9 distinct** (`eccfe630ec08de27` in BOTH F5a and F5e) | `cli_args.trainee_teams` |
| fork length | **7.77M / 18.16M / 13.20M** (→ 284,951,488 / 295,343,200 / 290,381,348) | **3.000M** each (25,067,760 → 28,067,760) | max checkpoint step |
| steps per pinned team | 2.59M / 1.82M / 1.32M | 1.50M | derived |
| **bots mixed in** | **YES — 50% of episodes**, roster weighted `aggressive_v2=3,heuristic2=3`, `heuristic_floor 0.2` | **NO** | `cli_args` (table above) |
| **difficulty curriculum** | **YES — target temp 5.0 → 1.0, `ratchet`, gated on trainee WR ≥ `0.55` per 500-game window. All three completed 16 ratchets** | **NO — `fixed`, target plays at `stable_opponent_temp 1.0` from step 0** | `cli_args`; `exploiter_temp_state.json` |
| `self_play` | False | False | *same* |
| `team_pfsp` / `team_block_episodes` | off / 1 | off / 1 | *same* |
| `ent_coef` | 0.05 | 0.02 | |
| `lr` / `min_lr` / `max_lr` | 7e-5 / 1e-5 / **6e-4** (adaptive band) | 3e-4 / 1e-5 / **None** | |
| `clip_range` | 0.10 | 0.15 | |
| `n_epochs` | 7 | 10 | |
| `batch_size` × `grad_accum_steps` = effective | 2048 × **16** = **32,768** | 2048 × **8** = **16,384** | |
| `vf_coef` | 0.2 | 0.5 | |
| `film_grad_accum_steps` | 1 (forks) | *(flag does not exist)* | |

⚠ **A pinning defect in the gen fleet** (not an explanation, but it should be fixed):
`data/teams/sample/eccfe630ec08de27.txt` is pinned by F5a (slot 1) *and* F5e (slot 2).
`Gen3Env._distill_mask` (`gen3_env.py:772-777`) iterates the teachers 1-indexed and `break`s on
the first match, so that team always resolves to **teacher 1** — **F5e's distillation KL never
fires on it**, and F5e effectively teaches on one team. In parallel, `matchup_setup.py:110-127`
builds `_team_strs` with all 10 entries including the duplicate, so `--distill-team-bias 0.4`
gives that team 2/10 = 8% of the bias and the other eight 4% each. Neither the spec parser nor
the env warns. The same team file also underlies tock-1a.

---

## Axis 2 — Distill recipe (the fold)

| item | **ai_v8_14_distill3** | R2KL (_61) | R2ACTION (_59) | R2TOPK (_60) |
|---|---|---|---|---|
| channel | full-distribution **forward** KL(π_teacher ‖ π_student) over the legal set, masked-mean on on-pin rows | identical | teacher top-1 renormalized over legal (= argmax CE) with AWR row weight `clamp(exp(\|Â\|/β), 20)` | top-3 renormalized + same AWR |
| `distill_coef` | **1.0** | 1.0 | 0.1810 | 0.1810 |
| `distill_target` / `topk` / `gate` | *(flags did not exist; the KL path)* | kl / 1 / none | **action** / 1 / none | **action** / 3 / none |
| KL direction | **FORWARD (teacher‖student, mass-covering)** — `distill_terms.py:89` | same | | |
| temperature | **NONE anywhere.** Both sides softmax at τ=1; there is no teacher-softening parameter in the code path in either era | | | |
| `distill_value_coef` | **0.0** | 0.0 | 0.0 | 0.0 |
| `distill_value_feat_coef` (FitNets cosine on `value_pooled`) | **0.0** | **0.5** | **0.5** | **0.5** |
| `distill_team_bias` | 0.4, split evenly over all teacher team files | 0.4 over 10 entries | | |
| teachers / teams | **3 teachers / 23 teams** | 5 / 9 distinct | | |
| **fold length** | **14.92M** (277,178,472 → 292,100,648) | **3.00M** | 3.00M | 3.00M |
| effective batch | 2048 × **16** = **32,768** | 2048 × **2** = **4,096** | 4,096 | 4,096 |
| PPO running beside the distill loss | YES | YES | YES | YES |
| states | on-policy rollout, masked to on-pin rows | same | same | same |
| on-pin distilled transitions (≈ bias × length) | **≈ 6.0M** | **≈ 1.2M** | ≈1.2M | ≈1.2M |

**The distillation CODE is byte-identical across the eras.** `git show
b13b30b:src/agents/training/instrumented_ppo.py` lines 449–477 and today's
`src/agents/training/instrumented_ppo/distill_terms.py:69-97` are the same function, character
for character, including the masked-mean, the `−1e9` illegal masking and the `None` empty-subset
guard. The multi-teacher fold loop (`_sel = (_tid_flat == _k)`, per-teacher KL, mean over active
teachers) is also unchanged. **Nothing about the distillation channel itself regressed.**

⚠ **No gen-era run has ever executed ai_v8_14's literal fold recipe.** R2KL is the arm that is
supposed to be "v8's channel", and it differs by one live term: `--distill-value-feat-coef 0.5`,
which ai_v8_14 ran at 0.0. That was a deliberate Phase-B ruling ("ai_v8_14 predates a6ae04f; its
0.0 is not the recipe, it is the gap the fix closed" — ledger, 2026-08-24), but it is still an
uncontrolled difference on the arm that carries the replication claim.

---

## Axis 3 — Ecology of the FOLD

| item | ai_v8_14 | R2ACTION / R2TOPK / R2KL | tick-1 (ai_v9_34, for reference) |
|---|---|---|---|
| `self_play` | True | True | True |
| `n_sentinels` | **10** | 5 | 5 |
| `pfsp_scale` | **2.5** | 0.0 | 0.0 |
| `pool_spread` | **True** | False | False |
| **`stable_opponents`** | **the 3 teachers** | **None** | the 2 tocks |
| `stable_opponent_selfplay_share` | **0.35** | 0.20 (inert — no stable opponents) | 0.35 |
| `stable_opponent_pfsp` | **True** | False | True |
| `team_pfsp` | **onesided** | off | off |
| **`team_block_episodes`** | **64** | **1** | **1** |
| bots | `aggressive_v2=3,heuristic2=3`, floor 0.2 | None | None |
| `eval_sentinel_greedy` | True | *(absent)* | *(absent)* |

`team_block_episodes` is the flag whose own comment (`matchup_setup.py:99-102`) names it "the
per-team gradient-density counter to the measured FiLM sample starvation". v8's fold held each
drawn trainee team for **64 consecutive episodes**; every gen fold redraws every episode.

---

## Axis 4 — Architecture

| item | v8 (config_version **45**, `arch_signature gen3_opp_hp_typed_candidates_v1`) | gen (**103**, `gen3_critic_route_wave_v1`) |
|---|---|---|
| obs `total_dim` | **2992** | **2501** |
| history block | 7 × 159 TurnDelta lag frames (`n_history_turns` present) | frames DELETED; 32 × 22 typed event window (`history_events`) |
| **action head** | flat `action_net = Linear(512, 11)` — **11 INDEPENDENT output rows** | `pointer_head` (`PointerNativeActionHead`): `move_score`, `switch_score`, `struggle_score`, each `Linear(POINTER_HIDDEN, 1)` applied to *every* candidate of its family — **3 output rows total, ZERO per-action free parameters**; `action_net` is a raising stub |
| G6 (action + value head) params | **6,156** | **56,196** |
| total params (SB3 triple-copy deduped) | 3,512,397 | 3,147,887 |
| G1 encoders | 201,363 (5.7%) | 221,631 (7.0%) |
| G2 trunk | 1,151,502 (32.8%) | 889,394 (28.3%) |
| G3 pools / value trunk | **861,326 (24.5%)** | **309,952 (9.9%)** |
| G4 aux/belief heads | **241,426 (6.9%)** | **620,090 (19.7%)** |
| G5 mlp_extractor | 1,050,624 (29.9%) | 1,050,624 (33.4%) |
| **per-team conditioning** | **`--zarch-film heads`, `--zarch-dim 32`, recon 1.0, vicreg 0.1, `--film-grad-accum-steps 6`** | **NONE — no team-conditioning mechanism exists** |
| aux heads only in v8 | `opp_belief_latent`(+coef), `pubval_mode`/`pubval_coef`, `value_active_readout`, `damage_refine_rounds 2`, `damage_reattend`, `threat_refine_outgoing`, `threat_unrevealed_outgoing`, `threat_status_refine`, `move_belief_prefuse`, `hp_type_belief_mode`, `spread_belief_nature_marginalize`, `active_ctx_hidden` | — |
| aux heads only in gen | `opp_intent`(+`intent_move_cell`/`intent_threshold`/`intent_conditional`), `cf_twin_heads`/`cf_shadow_critic`/`cf_evidential`, `pair_outcome_cell`/`pair_outcome_switch`, `switch_branch_cell`, `conditional_threat_cell`, `value_entity_pool`(+`_full`), `value_threat_inject`, `item_belief`, `history_events`, `entity_tail_seats`/`entity_topk_seats`, `edge_bias_families` (17 families), `species_prior_fusion`/`t0_species_prior`, `consequence_topk` | — |
| `belief_grad_mode` | shaping | shaping *(same)* |
| PopArt | on | on *(same)* |
| `vf_coef` / `value_dist_coef` / `win_prob_coef` | 0.2 / 0.2 / 0.2 | 0.5 / 1.0 / 0.05 |
| `move_belief_mode` / `opp_belief_cls_k` / `damage_topk_k` | revealed / 0 / 5 | both / 6 / 6 |
| `pi_features` participation ratio (forensics Phase B) | **50.24** | **20.59** |

**The pointer head is the structurally largest architectural change for a distillation objective**,
and the reason is not its parameter count. A flat `Linear(512, 11)` gives every action logit its
own 512-dim weight row, so gradient descent can raise "switch to slot 3" *in the states where a
teacher prefers it* by writing into row 3 alone — a LOCAL edit that leaves every representation
untouched. The pointer head has no such row. `switch_score` scores all six team tokens with the
same weights, `move_score` all four move tokens with the same weights
(`pointer_head.py:209-214`, applied per candidate in `forward`). The only way to change the
RELATIVE order of two switch candidates is to change their entity tokens (team_transformer output
→ shared trunk) or the shared context vector `latent_pi` (mlp_extractor → shared trunk).
**Distilling a decision into this architecture is structurally a representation edit, not a head
edit.** (See C3 for the evidence that both supports and complicates this.)

---

## Axis 5 — Parent maturity

| item | v8 parent (ai_v8_04) | gen parent (rev-1, ai_v9_29) |
|---|---|---|
| steps at fork | **277,178,472** | **25,067,760** |
| lineage | ai_v8_03_zarch_control (→267.6M) → **a 4-teacher distill fold** (`--distill-coef 1.0`, 4 ai_v7 specialist teachers, `--distill-team-bias 0.4`) → 277M | fresh generation, never folded |
| lr / band | 7e-5, adaptive 1e-5…6e-4 | 3e-4, min 1e-5, no max (band off) |
| `ent_coef` | 0.05 | 0.02 |
| Lyle `capacity_ratio` on `pi_features` | **1.154** (no capacity loss) | **0.948** (mild loss) |
| **baseline per-team piloting on the meter teams** | **0.438** (ledger D1) | **≈0.575** (derived: ZapDug readout, F5a 0.7075 at net +0.1325) |
| available headroom to the teacher | **≈0.28** (teacher ~0.72) | **≈0.12** (teachers ~0.69) |
| fold outcome | 0.438 → **0.710**, ELO 1986±26 → 2055±29 (disjoint) | +1.6pp pooled, n.s. |

The v8 parent is the product of a *previous* fold: the flywheel was already on its second turn
when the +69 was measured. rev-1 has never been folded.

---

## RANKED CANDIDATE EXPLANATIONS

Ranked by plausibility × testability. Each carries (a) mechanism, (b) existing evidence for and
against, (c) the cheapest discriminating measurement.

---

### C1 — The exploiter's OPPONENT CURRICULUM: 50% bots + a WR-ratcheted target
**Rank 1. The largest measured recipe difference, and the record's active claim about it is wrong.**

**(a) Mechanism.** An exploiter acquires slice-CONDITIONAL content only when its advantage signal
is large *and* attributable to what it did on that particular team. v8's fork opened against a
target at temperature 5.0 — near-uniform over the legal set, i.e. a nearly random opponent — and
was permitted to harden it one notch (×0.9) only after clearing a 55% training win rate over a
500-game window, sixteen times, floored at 1.0. That is a textbook auto-curriculum parked in the
max-advantage-signal zone for the whole fork: every stage produced large, sign-consistent
advantages that PPO could convert into a policy change specific to the board the pinned team
creates, and the opponent only became a full-strength mirror once the fork already had a
team-specific answer to defend. Half of the episodes additionally faced *scripted* bots — a
behaviourally alien opponent distribution that imposes a second, non-self-referential objective on
each pinned team and cannot be satisfied by whatever equilibrium a mirror match settles into. The
R2 fork faces one full-strength near-mirror (its own parent, 1.07M steps back) as the sole
opponent from step 0: win rate pinned near 0.5 by construction, advantages small and symmetric,
and the update dominated by entropy, value and ordinary drift — which is exactly what the
forensics measured (fork logit-shift ‖·‖ 231–257 against a no-fork control's **259**; on-slice
top-1 agreement 0.692–0.768 indistinguishable from off-slice 0.709–0.742).

**(b) Evidence.** FOR: `cli_args.exploiter_keep_bots` True in all three v8 forks and **False in
every gen fork including tick-1's tocks**; `exploiter_temp_mode` `ratchet`/`fixed`;
`exploiter_temp_state.json` present only under v8 with `n_ratchets: 16` (the curriculum verifiably
ran to completion in all three); `wrappers.py:380` proves the recorded 0.5 bot fraction is inert
without `keep_bots`. The forensics' undifferentiated-shift result is the predicted symptom of the
gen regime. AGAINST / limits: the R2 admission rows measure a real +11.65pp mean extraction, so the
gen fork *did* learn something — this candidate claims the difference is in the KIND of content
(global vs slice-conditional), not its existence, and that is consistent with a flat, high-precision
admission table (mean +0.1165, sd 0.0098) sitting beside flat differentiation.

**(c) Cheapest discriminator.** Re-run **ONE** R2 fork (3M, ≈2 GPU-hours) adding exactly
`--exploiter-keep-bots --exploiter-bot-fraction 0.5 --bot-weights aggressive_v2=3,heuristic2=3
--heuristic-floor 0.2 --exploiter-temp-start 5.0 --exploiter-temp-mode ratchet`, then re-run the
forensics' on-slice/off-slice top-1-agreement split (CPU, the scripts exist). **No fold is
needed to read it.** Prediction under C1: on-slice agreement drops BELOW off-slice — the v8
signature. A 2-arm version (bots-only vs ratchet-only) separates the two halves for 4 GPU-hours
and is worth it, because they are different mechanisms (opponent DIVERSITY vs advantage MAGNITUDE).

---

### C2 — FOLD DURATION and on-slice gradient DENSITY (14.9M vs 3.0M; block-64 vs block-1; 32,768 vs 4,096)
**Rank 2. Large, arithmetic, and one leg of it has never been tested.**

**(a) Mechanism.** The distillation term fires only on rows where the trainee pilots a teacher's
team. v8's fold ran **14.92M** steps at effective batch 32,768 with the trainee holding each drawn
team for **64 consecutive episodes**; the R2 folds ran **3.00M** at effective batch 4,096 with the
team redrawn every episode. Total on-pin distilled transitions differ ~5× (≈6.0M vs ≈1.2M) and
optimizer steps at a given batch differ 8×. Team-blocking is the second, independent leg: learning
a *conditional* mapping (do X on team A, Y on team B) requires consecutive same-condition gradient
to separate the condition from a global average. With block=1 across 48 envs the on-pin rows in
any minibatch are a thin, constantly-reshuffled sample of nine different conditions — the worst
case for that separation, and the same shape as this project's own measured FiLM sample-starvation
finding, which is precisely what `--team-block-episodes` was built to counter.

**(b) Evidence.** FOR: `cli_args.team_block_episodes` 64 vs 1, `grad_accum_steps` 16 vs 2, and the
checkpoint step ranges. The flag's own docstring names the mechanism. AGAINST / limits: rev-3
already tests +4.5M fold length and a 0.35 HI dose, so the *duration* leg is being addressed; the
`team_block_episodes` leg is not, and it is a one-flag change. Confounded with C1 if both are run
in the same arm.

**(c) Cheapest discriminator.** One 3M fold arm identical to R2ACTION except
`--team-block-episodes 64 --grad-accum-steps 16`. ≈2 GPU-hours, reads on the existing 9-team
piloting meter. This is the only untested factor on this axis.

---

### C3 — POINTER HEAD: copying a decision now requires moving the shared trunk
**Rank 3. The strongest structural story; the existing evidence cuts both ways, which is why it is third.**

**(a) Mechanism.** In v8, "prefer switch slot 3 in these states" could be written into one of
eleven independent 512-dim rows of `action_net` — a local edit costing the representation nothing.
The current head has no per-action parameters: `move_score` / `switch_score` / `struggle_score`
are one `Linear(H,1)` each, shared across all candidates of their family, so any change to them
shifts all four moves or all six switches together. The only route to a *relative* preference
change is through the entity tokens (team_transformer, shared trunk) or `latent_pi`
(mlp_extractor, shared trunk). A forward-KL copying objective therefore has nowhere cheap to land,
and must write into machinery every other team and every other head depends on. This predicts
BOTH observed pathologies from one cause: the fold's **all-or-nothing global rank collapse**
(`pi_features` 21.87 → 12.50 at *any* nonzero coefficient — a switch, not a dose response) is what
a copying objective forced into the shared trunk should produce; and the fork's failure to
differentiate on-slice is what happens when expressing a *conditional* preference requires
conditionally reshaping tokens rather than biasing an output row.

**(b) Evidence.** FOR: the head structure itself (`pointer_head.py:199-224`); G6 = 6,156 params /
11 rows vs 56,196 params / 3 rows; the all-or-nothing rank switch; per-parameter, the gen head
moved LESS than v8's (Δ² share 1.8% on 9× the parameters vs 0.7% on 1×). **AGAINST, and it is
real:** the forensics' function-level inter-fork cosine is FLAT across eras (+0.375 v8 vs +0.383
gen), so the head redesign does not change how aligned sibling forks are; and v8's fork delta was
**TRUNK-HEAVY** (G2 share 0.47 vs gen's 0.28, G2+G5 = 0.76 of Δ²) — v8 renovated the shared trunk
*more* and still folded back cleanly. That single number is the reason this is ranked below C1/C2:
"the flat head let v8 keep changes local" is falsified as a description of what v8 did. The
surviving, weaker form of C3 is about the *fold*, not the fork: v8's fold could absorb the
teachers' content into a cheap head while the gen fold cannot.

**(c) Cheapest discriminator — pure CPU, read-only, hours, no training.**
**Parameter-group grafting.** Take rev-1; graft ONLY R2ACTION's G6 (`pointer_head` + `value_net`)
onto it and measure teacher-agreement recovery on the recorded on-slice states; then graft only
G2+G5 (trunk); then both. Mirror it on v8 (graft ai_v8_14's `action_net` onto ai_v8_04). If v8's
fold content is head-attributable and the gen fold's is trunk-attributable, C3 is confirmed as a
description. Every input already exists and the forensics scripts do the forwards. The *causal*
test is dearer and should wait on the grafting read: bolt a zero-init residual `Linear(latent_pi,
11)` beside the pointer head (byte-identical at init) and re-run one fold arm — ~1 day of work
plus 2 GPU-hours.

---

### C4 — HEADROOM: v8's baseline was 0.438; the current baseline is ≈0.575
**Rank 4. Almost certainly a real contributing term; nearly free to estimate; hard to make decisive.**

**(a) Mechanism.** A fold's yield is bounded by what there is to take. v8's generalist was LOSING
on the taught teams (0.438) against teachers at ~0.72 — 0.28 of win rate sitting there, on teams
the parent was actively bad at. rev-1 sits at ≈0.575 with teachers at ≈0.69 — 0.12 available. A
"96% conversion" and a "13% conversion" computed against denominators that differ 2.3× are not
comparable quantities. Worse, the *kind* of content differs with the gap: what lifts a losing
policy to 0.71 is coarse (basic competence piloting an unfamiliar archetype, transferable as a
global shift); what lifts 0.575 to 0.69 is fine-grained conditional play — exactly the content a
global shift cannot carry, which closes the loop with the differentiation finding.

**(b) Evidence.** FOR: ledger D1 (0.438 → 0.710, teacher 0.72); the ZapDug readout (F5a-on-ZapDug
0.7075 at net +0.1325 ⇒ baseline 0.575). The ledger already names headroom as the surviving
alternative but has never measured it as a term. AGAINST: it explains a *ratio* difference, not
the differentiation finding — the R2 teachers being undifferentiated is not a headroom fact.

**(c) Cheapest discriminator.** Free first pass: regress conversion on baseline across every fold
in the archive (ai_v8_14, tick-1, fdA/B/C/E, G1/G2, R2ACTION/TOPK/KL) using each fold's own
recorded baseline piloting — n≈10 and confounded, but it prices the term. Cleaner: rev-3 is
already selecting coverage teams by *worst* rev-1 piloting; piggyback a per-team
conversion-vs-baseline readout on it at zero extra cost. That converts C4 from a caveat into a
measured slope.

---

### C5 — TEACHERS IN THE OPPONENT MIX + a richer fold ecology
**Rank 5. A real difference; the cheap version has already been tested and came back null; only the INTERACTION survives.**

**(a) Mechanism.** In v8's fold the three teachers were also *opponents* for 35% of the model-
opponent share with PFSP weighting, and a specialist stable opponent pilots its own pinned team
(`wrappers.py`). So on precisely the teams the KL was teaching, the student also had to *play
against* a policy that already executes that content: (i) it visits states in the teacher's own
region on-policy, which is where the copied advice is valid — the DAgger argument; (ii) the RL
gradient there points the same way as the KL rather than competing with it. The R2 folds have no
stable opponents at all; their only exposure to the taught teams is the 40% team bias against
generic self-play. v8's fold additionally ran `pfsp_scale 2.5`, `pool_spread`, 10 sentinels,
`team_pfsp onesided` and a bot floor — none of which any gen fold has.

**(b) Evidence.** AGAINST, and it is strong: arm **fdC** (coef 0.0, ecology ON at share 0.35)
measured **−1.2pp, n.s.**, and the programme recorded "the ECOLOGY EXONERATED"; **tick-1 DID carry**
`--stable-opponents` + 0.35 + `--stable-opponent-pfsp` and was graded INFERIOR on three meters.
FOR: fdC tested ecology *with no distill term*, and tick-1 tested it with 3M undifferentiated
teachers — neither is a test of "does teachers-as-opponents matter when the teachers are actually
differentiated", which is the only configuration the v8 arc ever ran. Ecology is plausibly a
multiplier on content, and a multiplier on zero is zero.

**(c) Cheapest discriminator.** It is an INTERACTION and cannot be cheaply tested before C1 lands.
Correct sequencing: make it the second arm of the C1 experiment — if a curriculum-trained fork
differentiates, fold it BOTH with and without `--stable-opponents <teachers>
--stable-opponent-selfplay-share 0.35 --stable-opponent-pfsp`. ≈2 extra GPU-hours on top of C1.

---

### C6 — the fold recipe was never literally replicated: `--distill-value-feat-coef 0.5` vs v8's 0.0
**Rank 6. Low prior, but the check costs minutes and it sits on the replication arm.**

**(a) Mechanism.** FitNets pulls the student's 128-dim `value_pooled` toward each teacher's by
cosine distance on the same on-pin rows the policy KL fires on. Five teachers 3M steps from a
common parent should have nearly-parallel value subspaces, making the term close to inert — but
"close to inert" is a shared-trunk pull on exactly the rows where the rank collapse was observed,
and nobody has read the number on this fleet. R2KL is the arm carrying the "this is v8's channel"
claim and it differs from ai_v8_14 by this one live term.

**(b) Evidence.** FOR: `cli_args.distill_value_feat_coef` 0.0 (ai_v8_14) vs 0.5 (every gen fold
including tick-1). AGAINST: ledger 4452 estimates FitNets alignment at 0.995 "at hour one" ⇒
≈0 gradient — though ledger 4340 records that this same metric was previously read INVERTED, so
the record's confidence here is low. The Phase-B ruling that adopted 0.5 is explicit and reasoned.

**(c) Cheapest discriminator.** Read `distill/*_value_feat_cos` straight out of the R2 arms'
TensorBoard — minutes, CPU, no new run. cos ≥ 0.99 throughout ⇒ C6 dies. Otherwise one 3M arm at
`--distill-value-feat-coef 0.0` (≈2 GPU-hours) settles it and simultaneously delivers the literal
v8 replication the programme has never run.

---

### C7 — Optimizer regime: ~4× the learning rate at ¼ the effective batch
**Rank 7. A plausible amplifier of the drift-anchor finding; heavily confounded with C1.**

**(a) Mechanism.** The gen fork runs `lr 3e-4` (v8: 7e-5), `clip_range 0.15` (0.10), `n_epochs 10`
(7), effective batch 16,384 forks / 4,096 folds (32,768). A larger, noisier step over a smaller
batch produces more *undirected* movement per unit of directed signal. That is exactly the
forensics' drift-anchor observation: R2CTRL — a verified no-fork continuation — drifts KL 0.3245
from rev-1, *inside* the forks' 0.269–0.349 range. If ordinary drift is the same size as the
fork's directed content, no downstream fold can separate them, and the teacher's distinctive part
is a small fraction of what the KL copies.

**(b) Evidence.** FOR: `cli_args`; the R2CTRL row; global rel-Frobenius 0.026 (gen) vs 0.021 (v8)
at a 3–4% *shorter* fork. AGAINST: rel-Frobenius differs by only 1.23× despite a 4.3× lr gap, so
the parameterization/PopArt/optimizer state absorb most of it; and the eras' parents differ in
maturity, which independently sets how far a given step moves the function.

**(c) Cheapest discriminator.** One 3M fork at `--lr 7e-5 --clip-range 0.10 --n-epochs 7
--grad-accum-steps 16`, then the fork-vs-CTRL logit-shift norm and the on/off-slice split (CPU
scripts exist; ≈2 GPU-hours). Run C1 first or factor them — a joint arm cannot attribute.

---

### C8 — no per-team CONDITIONING path exists in the current architecture
**Rank 8. A genuine era difference with strong existing evidence against it as a lever.**

**(a) Mechanism.** v8 carried `--zarch-film heads --zarch-dim 32` (recon 1.0, vicreg 0.1) plus
`--film-grad-accum-steps 6` — an explicit 32-dim team code FiLM-modulating the heads — and the
ledger's D1 row reports "far-z FiLM range +70%" for the fold itself, i.e. the fold *used* the
conditioning path. The current architecture has no team-conditioning mechanism of any kind. A
conditional teacher can only be copied if the student can represent the condition.

**(b) Evidence.** AGAINST, and it is close to decisive: this project's own LUT result — a free
per-team code did NOT close the N=20 gap (+0.024, CI [−0.016, +0.064]) — and the COUNT-dominates-
conditioning 2×2 (count +0.077 sig; conditioning +0.027 n.s.). Those were measured on a
*generalist* at N=20, not on a distillation target, so they are not a perfect refutation of the
distillation-specific claim; but the prior is strong and the tree has already climbed and
abandoned this ladder.

**(c) Cheapest discriminator.** None cheap, and the existing null makes it poor value. Recorded
here so the era diff is complete and so the FiLM-range figure in D1 is not mistaken for something
the current architecture could reproduce.

---

### C9 — Slice BREADTH: 23 teams total / 3–10 per teacher vs 2
**Rank 9 here only because Probe A is already dispatched on exactly this.**

**(a) Mechanism.** A 2-team pin can be satisfied by a global shift; a 10-team pin cannot, so
breadth *forces* team-conditional structure into the teacher. This is the mechanism the forensics
verdict nominated for v8's differentiation.

**(b) Evidence.** FOR: the differentiation contrast is exactly between 23-team-total teachers and
2-team teachers. AGAINST: the admission table is flat across the fleet (mean +0.1165, sd 0.0098,
five different team pairs), so extraction SIZE does not track breadth — though foldable CONTENT
might; and tock-1a's 4 teams showed no visible differentiation gain. The archive's breadth ladder
(2 / 2 / 3 / 4 / 9 / 23) has its top rung confounded with era, recipe and parent all at once —
which is the reason this diff exists.

**(c) Cheapest discriminator.** Probe A, already running: per-teacher on/off-slice differentiation
across the existing ladder on current-arch checkpoints, CPU-only. **Note for whoever reads it:
every rung of that ladder except the 23 is a gen-era fork with NO bots and NO temperature ratchet
(C1), so a FLAT result across 2→9 does not distinguish "breadth doesn't matter" from "the gen
regime suppresses differentiation at every breadth".**

---

## What this diff does NOT explain

- The distillation channel is byte-identical across eras, so nothing about the KL implementation
  regressed. The forward direction (mass-covering) and the absence of any temperature are
  properties of BOTH eras — they cannot be a difference, though they may still be a shared defect
  (that is Probe D's question, not this one).
- `belief_grad_mode shaping`, PopArt, `distill_team_bias 0.4`, on-policy state sourcing, PPO
  running beside the distill loss, `self_play` off in the forks, and `team_pfsp off` in the forks
  are the SAME in both eras.
- The eras' parents differ in maturity by 11× and in headroom by 2.3×, and no dimensionless
  statistic makes that go away. Every candidate above is stated as a contributing term, not a
  sole cause.

## Provenance index

| claim class | source |
|---|---|
| every recipe value | `models/<run>/metadata.json` → `cli_args` (resolved namespace) and `original_command` |
| step counts | `models/<run>/checkpoints/checkpoint_<N>_steps.zip` filenames |
| the v8 ratchet completed | `models/ai_v8_{06,09,13}_*/exploiter_temp_state.json` |
| bot-fraction gating | `src/agents/training/wrappers.py:380` |
| distill loss identity | `git show b13b30b:src/agents/training/instrumented_ppo.py` L449-477 vs `src/agents/training/instrumented_ppo/distill_terms.py:69-97` |
| team-bias split, `:*` resolution | `src/main/train/matchup_setup.py:106-137`, `src/agents/training/distill_spec.py` |
| duplicate-team defect | `src/agents/training/gen3_env.py:760-777` (`break` on first match) |
| head structure | `src/agents/model/pointer_head.py:164-224`, `src/agents/model/policy.py:58-120` |
| param censuses, drift/CKA/agreement figures | `designs/research_state/measurements/plasticity_forensics_v8_vs_gen_2026-08-28.json` |
| arch flag sets, obs dims | `models/<run>/model_config.json` |
| outcome figures (ELO, piloting, admission, fdA–E, tick-1) | `designs/research_state/ledger.md` (D1 row; the 2026-08-24…08-28 entries) |
