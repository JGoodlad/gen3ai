# As-built: Hidden-opponent belief aux head (`claude/belief-head`, 2026-06-13)

The in-place hidden-team belief, built end to end. Design intent + falsification live in
`design_offense_and_opponent_belief.md`; this is the as-built record of what shipped to the branch
(NOT yet to main/run). Live architecture detail is in `src/agents/model/CLAUDE.md` (BeliefSlots/
BeliefHead, v16) and `src/agents/training/CLAUDE.md` (labels + Hungarian loss + metrics).

## Why (one paragraph)
The opponent-ACTION head was falsified two ways (VoI ≈ 0.03 mon; the trunk already models the opp's
*visible-state* action at ~0.90 AUC). That falsification named a **separate, un-falsified gap**: the
opponent's HIDDEN team. Gen 3 OU has **no team preview**, so the ~3 unrevealed slots are genuinely
ABSENT from the obs — a representation probe cannot recover them. A pre-build learnability probe
(`models/saved_work/team_completion_learnability_pool.py`) showed conditioning on the revealed mons
beats the marginal usage prior by **+7pp recall / +8–10pp top-1** — real, addable signal. This is the
one "predict-it" lever that passes *unknown ∩ action-changing*.

## Architecture (the decisions, in order they were made)
1. **In-place belief slots, not a side pool.** `BeliefSlots` replaces the un-revealed opp team slots
   (the obs `species_known==0` slots) with **distinct learned unknown-mon tokens** BEFORE the
   transformer, so the imagined mons sit *in the opponent's lineup*, are refined by the same 12-token
   `TeamTransformer`, and are attended over by both CLS pools + the policy reasoning. Distinct per-slot
   init breaks the permutation-collapse the side-pool's queries were a workaround for. (Chosen over the
   `--opp-belief-cls-k` side-pool: "the network attends over it in latent space, the imagined mons are
   party members".) Off ⇒ not built; baseline byte-identical.
2. **Supervised by a species+moves aux head, BYOL-ready.** `BeliefHead` reads the refined opp tokens →
   a logits **dict** `{species, moves}`. Species CE + moves multi-label BCE. The dict return is the seam
   for a later BYOL/latent-matching target (see "Escalation" below) — chosen as v1 ("do 1 but allow a
   clean move to BYOL; collapse isn't a worry — the tokens get gradient from policy+value+aux").
3. **Order-invariant (Hungarian) matching.** A review caught that a fixed slot↔target assignment is
   reveal-dependent (the believed-slot window slides as mons are revealed), so a fixed slot token chases
   a shifting target. Fix ("do it right once"): per-sample min-CE-cost matching of the k believed-slot
   predictions to the k hidden mons (k! perms enumerated, vectorised per distinct k — no scipy, no
   per-sample loop). The anonymous slots now collectively cover the hidden SET.
4. **Privileged, training-only labels.** `Gen3Env._belief_labels` emits `belief_species[6]` /
   `belief_moves[6,4]` int Dict-obs keys from `battle2.team` (agent2's own full team). The believed
   mask is read **directly from the obs `species_known`** the model keys its injection on (single
   source — they can't diverge). Read ONLY by the aux loss; the forward reads only `obs["observation"]`
   → the omniscient labels never reach the acting path (the cardinal rule, re-verified by 2 reviews +
   the fuzz test's width/leak check).

## Robustness infrastructure (what makes it runnable)
- **Versioning.** `opp_belief_slots` is version-checked (`MODEL_CONFIG_VERSION` 16, no `ARCH_SIGNATURE`
  bump — off byte-identical). `opp_belief_aux_coef` is training-only (like `ent_coef`), **read back from
  the saved config on a flagless launcher resume** so the 3h restart preserves belief-ON.
- **Self-play / stable / distill interop.** `arch_toggles_from_model` → threaded into
  `current_model_version(**toggles)` at ALL 4 opponent-load sites (in-process pool + stable via
  `_run_arch_toggles`; `eval_worker` via cfg; `distill/worker` via config). Without it a belief-ON
  self-play run FATALs on its own (belief-ON) sentinels. **Verified both modes work**: the belief is
  internal to the forward (needs no labels for a forward), and `obs["observation"]` is unchanged, so a
  belief-OFF foreign stable opponent interoperates (the opponent gate is `arch_signature`-only).
- **Fail-loud, not silent garbage.** Out-of-vocab label id → raises (corrupt num pipeline). Non-leading-
  contiguous believed mask → raises (broken encoder packing). Legit-defensive guards (None battle, empty
  minibatch → None) kept.
- **Performance** (measured, `_belief_aux_loss` ~108ms/call CPU at production B): gather-before-softmax
  (~15ms saved, bit-identical), accuracy/P-R under `no_grad`, moves BCE skipped when weight==0. The
  Hungarian enum is <1ms — not the cost. Label path ~7µs/decision — negligible.
- **Observability** (12 `train/belief_*`): `species_acc` + `species_acc_above_chance` (vs 1/n_species),
  moves precision/recall, coverage, k_mean, and the shared-trunk grad-balance probe
  `grad/belief_share` + `belief_policy_cosine` — the principled "is the aux dominating / fighting the
  policy" signal. Tuning is empirical (documented loop in `training/CLAUDE.md`).

## Verification
2253 unit tests; `belief_labels_fuzz_test.py` (1792 real-battle decisions: labels == actual hidden team
+ no leak); a gradient-flow integration test (aux loss backprops through the stash to the belief params
AND the shared trunk); self-play/stable interop tests; [2] version-gate guard tests; 2 adversarial
review rounds (the first was invalid — its verify agents ran git in the wrong directory and rubber-
stamped; caught + re-run found real issues, all fixed).

## The open gate (honest)
It LEARNS (`species_acc` 0.08–0.16 vs ~0.003 chance immediately) but it is **UNMEASURED whether it
HELPS the policy** — belief is a means. Falsify-after-build = a fresh-run A/B (coef>0 vs 0) where
`belief_species_acc_above_chance` climbs AND the surprise-OHKO / hidden-mon **crater share falls** AND
win-rate is non-regressing. Risk: "learnable but inconsequential" (surprise-OHKO is ~as common in wins
as losses → may be informative-not-pivotal; pairs with the under-switching/commitment lever).

## Escalation: bring-your-own-latent (BYOL) target
v1 predicts DISCRETE labels (species CE). A latent target would instead regress the **continuous role
token** the model's own `PokemonEncoder` produces for the actual hidden mon (species+stats+typing+role
in one ~128-d vector), BYOL-style (predictor head + stop-grad on the target). It captures *similarity*
(predicting a similar wall is "less wrong" than a random mon — CE treats all wrong species equally) and
folds "role" in implicitly. The `BeliefHead` dict return + the privileged hidden-mon features are the
seam. See the build/measure plan in the same-named section of the project notes — gated on whether the
*discrete* v1 first proves it helps the policy at all.
