# Training — reward

Moved out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was split by topic.
The first section was REWRITTEN on **2026-09-26** when the shaped reward path was deleted; the
entropy sections below it are unchanged, including their dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.** *State-conditioned defensive-exploration
entropy* was added in the **2026-09-08** second pass.

---

## The reward — the TERMINAL alone (`reward_manager.py`); the no-progress clock (`progress_clock.py`)

**The shaped reward path is DELETED** (`gen3_shaped_reward_deletion_v1`, config v122, program_rust_core
§4 M3 row, owner-approved 2026-09-26). Gone: the eight PBRS potentials (`reward_potentials.py`), the
~25 BIAS terms (`reward_bias_terms.py`), the bias-additivity accumulate-and-refund, the no-progress
TAX (the clock's `last_penalty`), the suppressed-term fast path and its `GEN3AI_REWARD_VERIFY=1`
shadow twin (`reward_verify.py`), the eval-side reward clock in `RewardTracker`, and the 14 flags that
configured them (listed in `designs/deleted_flags.md`). Production had trained on the terminal alone
since the win-prob era, and the Rust core's reward (slice T) is the win indicator. The whole history —
every term, gate, measurement and hazard — is this file at the last pre-deletion commit:
`git show 029cee83:designs/training/reward.md`.

> **Where the reward lives now.**
>
> | Module | Holds |
> |---|---|
> | `reward_manager.py` | `Gen3RewardManager`: the terminal, the `reward/` export, the episode counters, the `win_margin` by-product; re-exports the three below |
> | `reward_config.py` | `RewardClass` (TERMINAL only), `RewardConfig`, `RewardBreakdown` (one field, `win_loss`) |
> | `reward_composition.py` | the census + its one-line announcer + `inert_reward_flags` + `reward_config_digest` |
> | `reward_weights.py` | `VICTORY_VALUE` (30.0, the default), `_TIMEOUT_TURN_CAP` (== the env's forfeit turn), `PBRS_GAMMA` (0.9999, the shaped-critic default discount) |
> | `material_margin.py` | the normalised material margin — the `win_margin` training-only OBS key, **not a reward term** |

**The terminal.** `RewardConfig` carries its three knobs, all resume-immutable (recorded in
`model_config.json`, value-checked by `ModelVersion.check_reward_config`):

| outcome | `--terminal-indicator` (PRODUCTION) | signed terminal (the default, `--critic shaped`) |
|---|---|---|
| win | `+victory_value` | `+victory_value` |
| decisive loss | `0.0` | `−victory_value` |
| pre-cap tie | `0.0` | `−victory_value` (shares the loss branch) |
| 250-turn TIMEOUT (a forfeit-loss detected by TURN COUNT, `turn >= _TIMEOUT_TURN_CAP`) | `0.0` | `--draw-penalty` (default −35) |

Production is `--critic winprob --terminal-indicator --victory-value 1.0 --draw-penalty 0`, so the
undiscounted return is exactly `1{win}` and V(s) == P(win|s). `--critic winprob` REQUIRES the other
three (`combination_checks`); under the indicator `--draw-penalty` is INERT (named in
`inert_reward_flags`) and any non-zero value is refused. Every non-terminal turn pays exactly 0.0.

**The parity proof (2026-09-26).** `reward_golden_test.py` (`gen3_reward_golden_v2`, routine tier,
`sim`) hashes every decision's reward, `win_loss` and `win_margin` over 30 bridge battles under six
terminal-only compositions, `production` read from the mirror. It was RECORDED at `029cee83` — the
last commit with the shaped code — and passes unchanged after the deletion: 2,772 decisions,
sha256 `2075c3f7…`. That is the measured statement that production's reward (and the `win_margin`
obs key) did not move.

**A SHAPED checkpoint cannot be resumed or forked on this code** — owner decision, 2026-09-26: it
REFUSES LOUDLY, never continuing silently on the terminal alone (a different objective under the
same run name). `agents.model.model_version.shaped_reward` recognises one from the RAW recorded
config (`hand_shaping` on — absent before v105 means on — or a reachable `--arm-no-progress-tax`),
`main.train.config.resolve_config` exits `FATAL_CONFIG` with the typed
`ShapedRewardCheckpointError`, and `main.checkargs` prints the same verdict ("shaped-reward parent :
YES … ✗ WOULD FAIL"). The fix it names: run it PINNED to a pre-deletion commit, ≤ `029cee83` (the
checkpoint's own recorded `git_hash` is the natural pin; the launcher pins a restart to it by
default, and `checkargs` reports such a pin as ADVISORY). ⚠️ Today this is belt-and-braces:
`MIGRATION_FLOOR` 121 already refuses every pre-v121 checkpoint, and every v121 run on record is a
win-prob arm. A FROZEN load (eval opponent, pool, teacher, prober) of any vintage is unaffected: the
deleted fields POP silently in `_migrate_config`, because a frozen forward never reads the reward.
Pinned by `src/agents/model/model_version/shaped_reward_test.py`, which fails if either refusal is
reverted. 🚨 **Arm S (the shaped comparator) is therefore not re-runnable on HEAD** — its
comparator runs pinned.

**The `reward/` export** (`reward_term_stats.py`, `gen3_reward_term_export_v1`) is unchanged in
shape: `reward/win_loss_*`, `reward/total_*`, and **`reward/untracked_abs_mean`, which must read
exactly 0.0** (the census and the fold agree). The PBRS / BIAS / refund rollups can no longer appear.

**The `win_margin` obs key** (`material_margin.py`) —
`clamp(2·ΔHP + 1.25·Δalive, ±6·3.25) / (6·3.25)` over the DECLARED team (unrevealed opp mons count
full-HP-alive), recomputed every turn by the manager and read by `gen3_env` for the win-prob head's
closeness-stratified metrics. It was Φ_mat's by-product; the potential went and the margin stayed,
with the deleted `--mat-alive-weight` frozen at the 1.25 every run trained with.
🚨 **A reward change may never move an OBSERVATION feature** — this one was once pinned at 0.0 for
the win-prob arm's whole life because it hid behind a reward gate.

### The no-progress clock — now an OBS-only counter

`ProgressClock` (`turns_since_progress`, reactive offset 2) is owned by `EpisodeTracker` and
updated at embed time; it classifies each decision window PROGRESS (reset `n`) / DENIED (freeze) /
NO_OP (increment, capped at `PROGRESS_CLOCK_CAP`). **Its reward half is deleted**: there is no
`last_penalty`, no `no_progress_penalty`, and nothing is charged — the classification vocabulary
("a charged no-op") survives in the code comments and test names and means "the clock advances".
The trapped-vs-wall `switch_legal` gate gated only the charge, so it no longer affects anything.

**The clock's two intent-restoring fixes — `--progress-decision-tense` / `--progress-switch-freeze`.**
Both default OFF (a flagless run's clock is byte-identical to every generation's —
`progress_clock_test.py`'s recorded `n` trace). They are resume-immutable `RewardConfig` fields
(recorded, value-checked, no `ARCH_SIGNATURE` bump) because they change the `turns_since_progress`
OBS scalar; they are threaded onto the clock by ONE call, `ProgressClock.apply_reward_config(cfg)`
from `gen3_env.py`. `--progress-decision-tense` points the forced-switch sit-out gate at the decision
that OPENED the window (`TurnDelta.decision_was_forced_switch`) instead of the one after it;
`--progress-switch-freeze` makes a voluntary switch that fails the (offense-only) progress predicate
FREEZE the clock instead of advancing it. Their motivating measurements (probes M and N,
`designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md` and
`no_progress_tax_review_2026-08-29.md`) were taken on the TAX; what they change today is the obs
scalar alone.

---

## State-conditioned defensive-exploration entropy (`--defensive-entropy-boost`)

`gen3_defensive_entropy_v1` — the answer to "the model under-uses Recover/Soft-Boiled/Wish/Refresh/Heal Bell
when safe" that does **NOT** touch the reward (so it can't create a stall incentive). Instead of biasing toward
healing (which would force you to hand-draw the good-defense-vs-stall line), it **explores** defensive moves more
and lets the *existing* anti-stall pressure be the guardrail (written when that was `--draw-penalty` + the
no-progress TAX; since the shaped-reward deletion, 2026-09-26, it is the 250-turn forfeit scoring as a non-win
and the obs deadline clock): the
model only KEEPS healing if the returns reward it, and a heal-war that drifts to a 250-turn draw is punished as
before. **The mechanism is ORTHOGONAL to the reward** — it explores the defensive option more but changes
nothing about its value, so if the critic learns healing is net-negative here (no-progress clock / racing
meta), the boost will NOT override that; it only surfaces the option. *Contingent* virtuous loop: IF the model
**discovers** defense is valuable (the returns must reward it), the self-play **opponents** become defensive
too, so the distribution self-enriches toward the patient meta self-play currently lacks.

- **The flag (`gen3_env._defensive_opportunity`).** Per decision, the env emits a training-only
  `defensive_opportunity` Dict-obs key = 1.0 when the trainee's ACTIVE mon has a *productive* defensive option:
  a legal `is_heal` move with HP below `_DEFENSIVE_HEAL_HP`=0.85, OR a legal self-cure (Refresh) while statused,
  OR a legal team-cure (Heal Bell/Aromatherapy) while any party member is statused; else 0.0 (forced switch →
  no moves → 0). Never raises (hot path). Read ONLY by the entropy term — never enters the pi/vf forward.
- **The boost (`instrumented_ppo`).** The per-decision entropy bonus is multiplied by `defensive_entropy_boost`
  on flagged decisions: `entropy_loss = -mean((1 + (B_eff−1)·flag)·entropy)`. `B=1.0` = OFF (byte-identical;
  also identical on any minibatch with no flagged decisions). `B_eff` anneals B→1 linearly over
  `--defensive-entropy-anneal-frac` of training (`_defensive_entropy_boost_eff`, 0 = constant) so exploration
  fades as the policy learns. The standard `train/entropy_loss` metric stays UNWEIGHTED; new `defent/*` metrics
  (`flagged_frac`, `boost_eff`, `entropy_flagged` vs `entropy_unflagged`) confirm the boost fired and raised
  entropy where intended.
- **Threading.** `--defensive-entropy-boost` (default 1.0) + `--defensive-entropy-anneal-frac` (default 0.0);
  the env emit is gated on `boost > 1.0`; the coefs are set on the model like `ent_coef` — **training-only, NOT
  version-locked, settable on resume** (no `model_config`/`ARCH` change). Try `--defensive-entropy-boost 3.0`.
  **Caveat (be honest):** the model already *samples* heals ~24% in safe spots, so exploration helps mainly at
  rare policy-collapse states (low HP + safe + revenge-killer coming) and can't manufacture a "heal→win" signal
  self-play lacks — it's complementary to, not a substitute for, a teacher/league. Watch the stall-rate canary.
  Tests: `defensive_entropy_test.py`.


## State-conditioned BAIT-exploration entropy (`--bait-entropy-boost`)

`gen3_bait_entropy_v1` — the same mechanism as the defensive boost above, on a different flag, and it
exists to answer ONE question. The bait verdict (`designs/research_state/ledger.md` → *E4 VERDICT*,
2026-08-23) closed the hunt with a stated mechanism: **exploration starvation at a saturated action** —
the whiff sits at p≈0.97, so the alternatives at p≈0.01-0.03 are never sampled and their advantage is
never realized. Everything upstream of the action was cleared: α/β know the switch, the critic already
ranks an alternative above the whiff in 21/23 loop decisions, and the E4 substrate arm moved the cells
and changed **11 decisions in 780**. What was never tested is the mechanism's own claim — that the
policy would fix this if it merely SAMPLED the alternatives. This flag is that test, and it is the
cheapest instrument that can separate the two remaining stories.

- **The flag (`gen3_env._bait_opportunity`).** Per decision, the env emits a training-only
  `bait_opportunity` Dict-obs key = 1.0 when the attack we would most likely click (`last_move` if it is
  still legal and damaging — the RE-CLICK — else the highest-base-power legal attack) deals **ZERO**
  damage to an **alive, revealed opponent BENCH** mon. Bench, not active, because in gen 3 the switch
  resolves first: the decision that whiffs is taken while the immune mon is still benched, which is also
  what makes the flag line up with the offline detector's whiff states. The zero-damage predicate is
  `baitbot.blocks` → `gen3_mechanics.effective_multiplier` → `data/` — ONE predicate shared with the
  scripted BaitBot opponent, so the flag fires on exactly the boards BaitBot exploits and no immunity
  table is hand-copied. Never raises (hot path); read ONLY by the entropy term, never in the pi/vf forward.
- **Three scope decisions, all deliberate.** (1) **REVEALED bench only** — using agent2's true team was
  available (the key is privileged) and refused: boosting entropy on a distinction the policy cannot make
  adds sampling noise with no learnable signal, and gen-15 settled that perception is not the gap.
  (2) **Ability immunities count once revealed** (Levitate/Water Absorb/Volt Absorb/Flash Fire), the same
  information the policy holds; type immunity always counts. (3) **The α half of the proposed predicate is
  NOT shipped** — α is published by the extractor inside the LEARNER's forward, and the flag is built in
  the env worker *before* any forward exists (the eval-time capture reads it off an in-process `RLPlayer`,
  a seam training does not have). There is nothing to emit as a second key; an α-gated variant would have
  to live at loss time and gate on the live policy's own moving α, which is a worse instrument for a probe.
- **The boost (`instrumented_ppo`).** Identical arithmetic to the defensive boost, on the same annealing
  schedule (`_annealed_entropy_boost`, shared so the two cannot drift): `entropy_loss = -mean(w·entropy)`
  with `w = (1 + (B_def−1)·flag_def)·(1 + (B_bait−1)·flag_bait)`. **Overlap semantics: multiplicative.**
  Each factor is exactly 1 off its own flag, so either boost alone is byte-identical to running it alone,
  and a decision flagged by both gets the product (they are near-disjoint in practice — "a heal is legal"
  vs "our attack is dead into their bench"). `B=1.0` = OFF, byte-identical *including on a fully populated
  flag column*. `train/entropy_loss` stays UNWEIGHTED; `baitent/{flagged_frac, boost_eff, entropy_flagged,
  entropy_unflagged}` say whether the boost fired and where.
- **Threading.** `--bait-entropy-boost` (default 1.0) + `--bait-entropy-anneal-frac` (default 0.0); the env
  emit is gated on `boost > 1.0`; the coefs are set on the model like `ent_coef` — **training-only, NOT
  version-locked, settable on resume** (no `model_config`/`ARCH` change, nothing in `flag_registry` — it
  reaches no extractor).

**The pre-registered readings** (write them down before the run, per the hunt doc's own rule):

| observation | reading |
|---|---|
| whiff / re-click rate falls under the boost and **STAYS** down past the anneal | **SAMPLING was the block.** The mechanism the verdict named is right, the correction is realizable on-policy, and the cheap lever generalizes to other saturated actions. |
| falls under the boost and **REVERTS** as `boost_eff → 1` | **CREDIT is convicted.** The alternatives were sampled, their advantage was estimated, and the policy still went back — so the correction has to arrive off-policy: R1/R2's counterfactual labels and the search-teacher/OPD inherit, exactly as the E4 entry's closing paragraph predicted. |
| never falls, at a healthy `baitent/flagged_frac` | neither — the boost did not move behaviour at all; read `entropy_flagged` vs `entropy_unflagged` first to confirm the boost actually reached the policy. |
| never falls, at a near-zero `baitent/flagged_frac` | a **DOSE** finding, not a mechanism finding: the states were not in the rollout. Raise exposure (a BaitBot-shaped opponent in the pool) before concluding anything. |

⚠️ `flagged_frac` is the exposure reading and must be quoted with any verdict — the E4 entry's own
ecology finding (BaitBot-shaped opponents propagate baiting through self-play, pivots 574→773) is why a
dose number is not optional here. **MEASURED at build time** (`--debug --steps 10000
--bait-entropy-boost 3.0`, 2026-08-23, default bot roster, early policy): `flagged_frac` **0.005-0.016**
— i.e. ~1% of decisions are bait boards at the default opponent mix, with `entropy_flagged` 1.74-1.88 vs
`entropy_unflagged` 1.65-1.69. That is the dose a probe arm inherits unless it deliberately raises
exposure, and it is small enough that a BaitBot-weighted pool is worth considering in the same launch.

Tests: `bait_entropy_test.py` (predicate units incl. the four ability immunities with negative controls,
the anneal, and the loss on the REAL `train()` path — OFF byte-identical with every row flagged, the boost
inert on unflagged rows, and the exact identity `(ent_coef=c, boost=B) ≡ (ent_coef=B·c, boost=1)`) +
`bait_opportunity_integration_test.py` (`sim`: the emission path through a real bridge battle, and the flag
CROSS-CHECKED against `main.prober.loops` — every detector `immune` whiff whose arrival was already
revealed must have been a flagged decision; **23 immune whiffs / 21 cross-checked / 0 disagreements** on a
PINNED matchup. The teams are pinned deliberately: drawn from the pool, only 2 of 14 sample-team pairs
produced any immune whiff at all and the cross-checked count ranged 0-48 run to run — a test whose sample
size is a random variable cannot carry a floor. The reverse direction is deliberately not asserted — an
opportunity predicate fires before the mistake, so it fires on states that never become whiffs).
