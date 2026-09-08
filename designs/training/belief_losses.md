# Training — the supervised belief losses

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Hidden-opponent belief aux loss (`--opp-belief-aux-coef`)

The training half of the in-place belief feature (model side in `src/agents/model/CLAUDE.md` →
`BeliefSlots`/`BeliefHead`, v16). Off by default. Two pieces live here:
- **Labels (`gen3_env.py`).** When `emit_belief_labels` (set from `--opp-belief-aux-coef>0`), `step()`
  and `reset()` merge two PRIVILEGED int64 Dict-obs keys into the trainee obs: `belief_species[6]`
  and `belief_moves[6,4]` — the opponent's still-hidden mons (species/move NUMs), sourced from
  `battle2.team` (agent2's own full team). The believed-slot mask is read **straight from the obs
  vector's per-slot `species_known`** (the SAME signal `BeliefSlots` keys its injection on) — single
  source of truth, so the label's believed slots can never diverge from where the model fills
  unknown-mon tokens. The pure builder is `agents.observation.belief_labels`. These keys are
  **training-only** (eval/self-play/inference never declare/need them) and read ONLY by the loss — the
  model forward reads only `obs["observation"]`, so the omniscient labels can't leak. **Fail-loud:**
  `_belief_labels` raises if the obs `species_known` is not leading-contiguous (a broken encoder
  packing invariant), rather than mis-slotting supervision.
- **Loss (`instrumented_ppo.py` `_belief_aux_loss`).** `train()` reads the per-minibatch stashed
  logits (`policy.features_extractor.last_belief_logits`, set by the `evaluate_actions` forward) + the
  label keys, and folds `opp_belief_aux_coef·(species_CE + moves_weight·moves_BCE)` into the loss.
  **Order-invariant (Hungarian / DETR):** the k believed-slot predictions are matched to the k hidden
  mons by per-sample min-CE-cost assignment (k! perms enumerated, vectorised per distinct k), so the
  anonymous slot tokens collectively cover the hidden SET rather than each chasing a reveal-shifting
  fixed target. Perf: species log-softmax on the GATHERED believed slots (not full `[B,6,S]`); moves
  BCE skipped when `moves_weight==0`; accuracy/P-R diagnostics under `no_grad`. **Fail-loud:** an
  out-of-vocab label id (impossible on real Gen-3 nums) RAISES — corrupt num pipeline, not a silent
  drop. Returns `None` on an empty (zero-believed) minibatch to avoid NaN-poisoning.
- **Metrics (`belief/*` — its OWN TB prefix, not `train/`, matching the `grad/`/`popart/`/`win_prob/`
  groups; rendered in the launcher TUI directly BELOW the `train/` block in the train column).** Headline
  `species_acc` + `species_acc_above_chance` (anchored to
  `1/n_species`); `moves_precision`/`moves_recall` (the opaque BCE alone can't tell if the ~4 true
  moves rank high); `coverage` (fraction of decisions with ≥1 believed slot) + `k_mean` (so acc is
  interpretable — k=1 vs k=5 differ); `species_ce`, `moves_bce`, `aux_loss`; plus `mask_rate` — the
  **uniform per-head coverage key** (`gen3_belief_mask_rate_v1`): fraction of the B×6 slot grid the
  head scored this minibatch. EVERY belief head emits it under its own prefix (`belief/mask_rate`
  hidden-team, `belief/spread_mask_rate`, `belief/natureev_mask_rate`, `belief/hptype_mask_rate`),
  comparable across heads and batch sizes where the older `n_slots` counts are not — the label-coverage
  baseline the belief-unification consolidation will judge per-head non-inferiority against. Note the
  conventions TILE: hidden-team masks HIDDEN slots, the spread/nature/hp-type heads mask REVEALED
  ones. **ALL SIX supervised belief losses live in `belief_bank.py`** (the design_unified_belief
  §4 code-shape fold, 2026-08-16): one declarative ROW per head (stash/attr/obs/param arg spec ·
  coef key · metric prefix · the `aux_loss` historic key for hidden-team) and `compute(site=…)`
  loops replace the six inline verticals at their THREE original train() positions
  (`hidden_move` = hidden-team Hungarian + move-belief BCE · `latent` = move-latent grading ·
  `revealed` = spread/nature-EV/hp-type) — the site tag is what preserves the float-addition
  sequence exactly (byte-identical), the old `InstrumentedMaskablePPO._*_loss` statics remain as
  aliases, and a seventh supervised belief is now a row, not a slice
  (`belief_bank_test.py::test_sites_partition_the_registry` pins the partition). **Balance:** the
  shared-trunk grad-balance probe (`grad_balance.py`) reports `grad/species_belief_share` (this CE's
  share of the common trunk-pull total) + `grad/species_belief_policy_cosine` — the principled "is the
  aux DOMINATING / fighting the policy" signal (and `grad/aux_share` for the COMBINED non-RL draw).
  **Tuning is empirical:** start `--opp-belief-aux-coef` small (0.1–0.3) so
  `species_belief_share` lands at a few %; confirm `species_acc_above_chance` climbs in warmup; if the
  policy degrades (`train/approx_kl` spikes, `entropy` collapses, `explained_variance` drops) while
  the share is high, the aux is fighting the actor → lower the coef. `--opp-belief-moves-weight`
  balances CE vs BCE (species dominates at 1.0). Both coefs are **training-only** (like `ent_coef`,
  NOT version-locked); the `opp_belief_slots` arch toggle they imply IS version-checked, and
  `--opp-belief-aux-coef` is **read back from the saved config on a flagless resume** (so a launcher
  restart preserves belief-ON instead of FATALing).
- **Tests.** Unit: `belief_aux_loss_test.py` (Hungarian order-invariance + min-cost-matching, empty
  guard, grad, fail-loud out-of-vocab, perf fast-path), `agents/observation/belief_labels_test.py`,
  `agents/model/belief_slots_test.py` (incl. end-to-end gradient flow through the stash to the belief
  params + shared trunk). **Fuzz** (real bridge battles, no server):
  `poke_env_gaps/belief_labels_fuzz_test.py` validates the emitted labels against the ACTUAL opponent
  team, the single-source mask invariant, the moves-⊆-moveset invariant, and the no-leak width check
  over thousands of live decisions:
  `python src/agents/training/poke_env_gaps/belief_labels_fuzz_test.py [n_battles]`.

## Move-belief reinjection loss (`--move-belief-mode` / `--move-belief-coef`)

The training half of the move-belief feature (model side: `src/agents/model/CLAUDE.md` → MoveBelief,
v17). The predicted moveset is REINJECTED into the opp token (it flows to both heads), AND supervised:
- **Labels (`gen3_env.py`).** When `move_belief_mode != "off"` (or species-belief on), the trainee obs
  carries `belief_moves[6,4]` (hidden slots, shared with the species aux) and — when mode ∈
  {revealed, both} — `known_moves[6,4]`: each REVEALED slot's FULL privileged moveset (so the head learns
  the as-yet-unrevealed moves). Both are training-only, sourced from `battle2.team`; builder
  `agents.observation.belief_labels.build_known_move_labels`. (`known_moves` keeps its name — it holds the
  privileged-*known* moveset of a revealed mon; the `revealed`/`unrevealed` mode names refer to the MON.)
- **Loss (`instrumented_ppo.py` `_move_belief_loss`).** Reads `last_move_belief_logits` + the move
  labels, folds `move_belief_coef · BCE` over two DISJOINT slot populations: **revealed** slots (direct
  multi-label BCE on `known_moves` — slot==species, no matching) and **unrevealed** slots (order-invariant
  Hungarian BCE on `belief_moves` — the believed slots are anonymous; cost is the assignment-relevant
  `-(pred·target)`, a cheap einsum). `mode` selects which population(s) are scored. Mode is read off the
  extractor (single source); coef is a model attr (training-only).
- **Metrics (`belief/move_*`).** `bce`, `precision`, `recall`, `revealed_slots`, `unrevealed_slots`,
  `loss`. The move-loss gradient ALSO reaches the trunk via the reinjection, so it is broken out on its
  own as `grad/move_belief_share` (+ `_norm_shared`/`_policy_cosine`) on the common trunk-pull total.
- **Versioning.** `move_belief_mode` (str) is the version-checked structural toggle (fresh-only;
  `unrevealed`/`both` additionally REQUIRE `--opp-belief-aux-coef>0` so the hidden slots carry
  learned tokens); `move_belief_coef` is training-only, **read back on a flagless resume**. It used
  to auto-force `--attend-unrevealed-opponents`; at v78 that toggle became **config_only frozen ON**,
  so the prerequisite holds by construction and the auto-force branch is deleted. The revealed-vs-unrevealed axis is the defensible-vs-omniscient A/B.
- **Tests.** Unit: `move_belief_loss_test.py` (direct-BCE, Hungarian order-invariance + min-cost match,
  mode gating, grad, fail-loud), `agents/model/move_belief_test.py` (module mask-gating + grad +
  per-mode wiring + off byte-identical), `belief_labels_test.py` (`build_known_move_labels`),
  `snapshot_test.py` (version gate + threading).

## Spread-belief supervision loss (`--spread-belief-coef`)

The training half of the THIRD belief leg (model side: `src/agents/model/CLAUDE.md` → SpreadBelief, v25).
The `SpreadBelief` head predicts the opponent's hidden SPREAD (the 5 derived stats {atk,def,spa,spd,spe}) and
the `DamageOperator` consumes it for damage + outspeed. WITHOUT this loss the head is **unsupervised** — it
gets only the weak/unaligned gradient leaking back through the op, so it sits at the usage-mean prior, which
**over-estimates the largest-EV stat** (the modal Smogon set maxes it) → the op mis-prices damage/outspeed
against the *modal* opponent, not the real one. Off by default (`--spread-belief-coef 0`). Two pieces:
- **Label (`gen3_env.py` → `belief_labels.build_known_spread_labels`).** When `emit_spread_labels`
  (= `--spread-belief` AND `--spread-belief-coef>0`), `_spread_labels` (INDEPENDENT of the species/move
  belief path, so `--spread-belief` works standalone) merges two TRAINING-ONLY Dict keys: `belief_spread`
  [6,5] (the TRUE derived stats of each REVEALED opp mon, matched BY SPECIES against agent2's own team's
  computed `mon.stats` — the privileged ground truth Gen 3 hides from the trainee even once the species is
  revealed) + `belief_spread_mask` [6] (1 = supervised). Believed/pad/incomplete-stat slots → mask 0. Read
  ONLY by the loss; the model forward reads only `obs["observation"]`. SPREAD_STAT_ORDER == the op's
  `_SB_ATK.._SB_SPE` consumption order (pinned by `spread_belief_loss_test` — the GIGO/order-mismatch guard).
- **Loss (`instrumented_ppo._spread_belief_loss`).** Reads the extractor's stashed `last_spread_belief`
  [6,5] (the believed stat VALUES the op consumes) + the label keys; folds `spread_belief_coef ·
  smooth_l1((believed − true)/_SPREAD_LOSS_SCALE)` over the masked (revealed) slots. The gradient flows
  believed → `stat_head` → opp tokens → trunk, so it is broken out as its OWN per-head share
  `grad/spread_belief_share` on the common-denominator grad-balance probe (it does NOT gate the
  probe-sample timing — it scores on near-always-present REVEALED slots). **Leak-safe:** the believed
  stats are a MODEL OUTPUT (the op's input), not a label; the true-spread label is training-only, read
  only here.
- **Metrics (`belief/spread_*`).** `mae` (believed-vs-true error in RAW stat points — should fall),
  `largest_bias` (signed error on each mon's LARGEST true stat — the "over-estimates the largest EV"
  diagnostic, → 0 as the head learns), `n_slots` (supervised slots/minibatch), `mask_rate` (the
  uniform coverage key — see the `belief/*` metrics bullet above), `loss`.
- **Nature/EV decomposition (`gen3_nature_ev_belief_v1`, v40, `--spread-belief-nature`).** The fix for the
  stuck `largest_bias`: the additive head predicts the DERIVED stat directly (a point estimate BETWEEN the
  nature ×1.1/×0.9 modes); the generative head predicts a NATURE categorical ⊕ Smogon prior + per-stat EVs ⊕
  prior and COMPUTES the derived stat, so the asymmetry + EV budget are structural. A SECOND loss term
  `_nature_ev_belief_loss` (nature CE + EV smooth_l1 over REVEALED slots, folded at the SAME
  `spread_belief_coef`, metrics `belief/natureev_{nature_acc,nature_ce,ev_mae,n_slots,mask_rate}`) supervises the
  decomposition DIRECTLY (the derived loss alone is many-to-one). Label: the TRUE (nature, EVs)
  **deterministically INVERTED** from agent2's `mon.stats` (`damage_tables.invert_nature_evs`, GIGO-guarded —
  gen3 hides them, so we invert the visible derived stats), emitted by `gen3_env._spread_labels` as
  training-only `belief_nature`/`belief_ev`(+masks), cached per battle. The op-side
  `--spread-belief-nature-marginalize` (an exact 3-point quadrature of P(KO) over the believed nature
  distribution) is **DELETED** (v66): measured on gen-8's own checkpoint across 1,075,200 alive
  (defender, candidate) cells it moved |ΔP(KO)| by 0.00000 at p50/p90/p95 and 0.00047 at p99, because a
  peaked nature posterior (top-1 mass 0.75) makes marginalising ≈ evaluating at the mode. Sound theory,
  absent magnitude — ledger K1's shape. Smoke: `nature_acc` rises toward the true nature,
  `largest_bias` trends to 0.
- **Versioning.** `spread_belief` (the head) is the version-checked structural toggle (v25, fresh-only);
  `spread_belief_coef` is **training-only** (inherited on a flagless resume, like `move_belief_coef`). The
  loss adds NO forward/weight change → no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump (a checkpoint trained
  at coef 0 can resume with coef>0 to start supervising — like enabling any aux).
- **Tests.** Unit: `spread_belief_loss_test.py` (masking, scale-normalised smooth_l1, grad ONLY to
  supervised slots, the `largest_bias` over-estimate detector, off→None, the stat-order GIGO pin),
  `belief_labels_test.py` (`build_known_spread_labels` species-match + mask + incomplete-stat skip). **Fuzz**
  (real bridge battles, no server): `poke_env_gaps/belief_labels_fuzz_test.py` validates `belief_spread` ==
  the actual revealed opp mons' true derived stats (`mon.stats`), believed/pad slots zero (no leak), and the
  OFF env declaring no spread keys, over thousands of live decisions. End-to-end smoke (`--debug
  --unified-moves both --spread-belief --spread-belief-coef 0.1 --n-steps 64`) confirms the roundtrip + the
  loss runs + `belief/spread_*` metrics.

## Opponent HP-type belief loss (`--hp-type-belief-coef`)

The training half of `gen3_typed_hp_belief_v1` (model side: `src/agents/model/CLAUDE.md` → DISCRETE typed
Hidden Power, v51). The opponent's Hidden Power is reasoned about ONLY as the 16 discrete typed moves; the
`HPTypeBelief` head supplies the type half of `P(HP_t) = presence · P(type=t)`, and this CE is its direct
supervision.
- **Label (training-only, privileged).** `Gen3Env._hp_type_labels` reads agent2's OWN team for each
  REVEALED opp mon's true Hidden Power type (the typed move-id suffix → `belief_labels.build_hp_type_labels` /
  `hp_type_idx_from_move_id`, in the `HIDDEN_POWER_TYPE_ORDER` index space) and emits the `hp_type_label` [6]
  / `hp_type_mask` [6] Dict keys (mask=1 only at a revealed slot whose species runs HP). Gen 3 NEVER reveals
  the opp HP type, so this can't ride the obs vector — it is leak-safe (a separate Dict key, read ONLY by the
  loss; the obs vector width is unchanged). Emitted when there is a move belief AND `--hp-type-belief-coef>0`.
- **Loss (`instrumented_ppo._hp_type_belief_loss`).** Reads the extractor's stashed `last_hp_type_logits`
  [6,16] (the prior⊕delta posterior) + the label keys; folds `hp_type_belief_coef · cross_entropy` over the
  masked (revealed-HP) slots. Gradient flows posterior → `hp_type_head` → opp tokens → trunk (joins the
  per-head grad-balance probe as `grad/hp_type_*`); `aux_probe_terms["hp_type"]`.
- **It is no longer the head's ONLY signal.** Since v51 the move-belief BCE labels use each Hidden Power's TRUE
  TYPED num, so the multi-label BCE lands directly on the composed typed channels — which trains the type
  posterior AND the presence channel jointly, through one gradient path. The damage operator's gradient rides
  the same channels. So `--hp-type-belief-coef 0` no longer means "unsupervised": it means "no dedicated CE on
  top". The default is **0.05**.
- **Metrics (`belief/hptype_*`).** `acc` (top-1 HP-type accuracy — should climb well above the 1/16≈0.06
  chance; a short bridge smoke reaches ~0.8 quickly since the head cold-starts at the Smogon prior), `loss`,
  `n_slots`, `mask_rate` (the uniform coverage key — see the `belief/*` metrics bullet above).
  `hp_type_belief_coef` is **training-only** (inherited on a flagless resume, like
  `spread_belief_coef`). The old version-checked `hp_type_belief_mode` is DELETED — the head is unconditional
  whenever there is a move belief, and it no longer requires `--damage-op`.
- **Tests.** Unit: `model/hp_type_belief_test.py` (the Σ-typed-equals-presence constraint, both certain-fact
  eliminations, the immune-bug regression, the op having no HP source of its own, the CE loss masking,
  `build_hp_type_labels`, the 16-axis GIGO pin, the v51 migration). **Fuzz** (real bridge battles): the
  extended `poke_env_gaps/belief_labels_fuzz_test.py` validates `hp_type_label` == each revealed HP-mon's true
  type, the TYPED move labels == the real opponent movesets, mask 0 on revealed-no-HP / believed / pad slots
  (no leak), and the OFF env declaring no HP-type keys. End-to-end smoke (`--debug --use-bridge=node
  --unified-moves both --spread-belief --hp-type-belief-coef 0.05`)
  confirms the roundtrip + `belief/hptype_*`.

## Opponent-class label weight (`--intent-label-bot-weight`, default 1.0 = OFF)

`gen3_intent_label_bot_weight_v1` — a per-sample weight on the opponent-intent (α/β) LABELS
produced against a heuristic **bot**; every other opponent class (pool / stable / exploiter) keeps
1.0. It exists because a bot's tendencies are not the meta's, and the curriculum guarantees the
head meets them first: `heuristic_fraction` is **0% self-play below `SELF_PLAY_START`**, so a fresh
generation trains 100% vs bots until the pool seeds. Measured on gen-11, supervised intent rows ran
**100% bot at 2M and ~7% from 6M on** — and bot rows score differently (info gain 0.124 nats vs
pool 0.254, accuracy flat ~0.50 all run). The risk this knob addresses is imprinting: α/β learning
a decision tree during the ramp and carrying it into pool play.

**The mechanism.** It reuses the EXISTING identity source — the `opp_class` obs key
(`gen3_opp_class_v1`), tagged once per episode by `MaskableAgentWrapper._select_episode_opponent`,
pushed onto the env at `reset()`, emitted beside the α/β labels by `Gen3Env._opp_intent_labels`,
shifted with them by `align_labels_to_predictions`, and already read in `train()` for the
stratified metrics. **No new obs key was added**; the key that splits the dashboards is now also
the key that weights the loss. `agents.model.opp_intent.intent_losses` takes a `bot_label_weight`
and folds it as

```
loss = Σ_i w_i · ce_i / n_sup        w_i = W on bot rows, 1.0 elsewhere
```

— weighted **before the mean, at the unchanged `n_sup` denominator**. Normalising by `Σw` instead
would make a 100%-bot minibatch identical to an unweighted one, i.e. do nothing in exactly the
regime the knob exists for; with `n_sup` a `w ≡ 1` batch reproduces the plain mean, so the
`--opp-intent-coef` semantics are untouched.

**Composition with the masks.** The masks run FIRST. A row masked by `INTENT_IGNORE` (unmodeled
seat, unrevealed β switch-in, non-switch decision) is dropped, and the weight multiplies only the
survivors — a masked bot row contributes nothing at any weight, and `W = 0` legally means "score
bot rows for the metrics, train on none of them".

**It is confined to α/β and that is a design claim, not an oversight.** The other supervised
beliefs — species, move, item, spread, nature/EV, HP-type — are **team truth**: what the
opponent's team IS does not depend on who is piloting it, so discounting a bot's rows there would
throw away valid labels. Only INTENT is behaviour. The `belief_bank` rows never see `opp_class`
(pinned by `opp_class_plumbing_test::test_only_the_intent_loss_takes_the_weight`).

**Diagnostic: `opp_intent/label_bot_frac`** — the bot share of the α rows actually SUPERVISED this
minibatch. The per-class `alpha_n_supervised_*` counts carry the same information but are gated on
≥2 rows and are counts, so nothing reported the ratio. It is emitted **whether or not the weight is
set**, because the decision to set it is made off this number. The existing stratified metrics are
untouched — they measure the head, and a weighted loss must not move an accuracy.

**Default 1.0 is a deliberate no-op.** At 1.0 the original unweighted `cross_entropy` call is taken
unchanged, so the loss is **bit-identical** (not merely close — pinned by exact equality over three
opponent mixes). Lowering it is a **generation/fork decision, not this change**: it moves the
supervision distribution, so it belongs at a launch boundary where it can be attributed.

**Pre-registered decision path.** Decide at the gen-16 launch, beside the B-move supervision call:
run the fork A/B **W=1.0 vs W=0.25**, gated on **`opp_intent/alpha_acc_pool`** (the `_pool` suffix,
never the bare key — the bare one is a moving mix). W=0.25 wins only if `alpha_acc_pool` is
non-inferior or better; a fall there means bot rows were carrying real signal and the knob goes
back to 1.0. `label_bot_frac` sizes the manipulation before the arm is run — if it is already ~0 at
the steps that matter, the arm is not worth a generation slot.

**Class: `training_coef`.** It scales a loss and touches no forward pass ⇒ no `ARCH_SIGNATURE`
bump, not in `check_compatible`, no `check_*` of its own; recorded on `ModelVersion`
(`MODEL_CONFIG_VERSION` v97) for provenance and so a **flagless resume inherits it** via `_resolve`,
exactly like `--td-aux-coef`. It is deliberately NOT in `agents/model/flag_registry.py` — that
registry's scope is extractor architecture toggles, and this reaches the extractor not at all
(same call as `--td-aux-coef`).

Tests: `agents/model/intent_label_bot_weight_test.py` (bit-identity at 1.0 on every mix, the
hand-computed weighted mean, the all-bot scale-down, non-bot classes never discounted, W=0 killing
the gradient, proportional gradient scaling, mask composition on both axes, β taking the same
per-row vector, `label_bot_frac`, the stratified metrics unmoved, the CLI/ModelVersion/migration
legs) and `agents/training/opp_class_plumbing_test.py` (the whole `opp_class` chain, which nothing
covered before it became load-bearing: the two hand-mirrored class tables agreeing, the wrapper tag
per opponent kind, the reset-time push onto the env, the env emission, the one-ahead shift, the
episode-boundary drop, buffer shuffle-alignment on a real `MaskableDictRolloutBuffer`, and the
train-loop call site).

