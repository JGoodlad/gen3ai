# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, and the bot-eval pipeline.
**How to launch training** (commands, flags) lives in the root `CLAUDE.md` → Training /
Launcher; this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are documented in
`src/agents/battle/CLAUDE.md`. The obs-build performance gate is in
`src/agents/observation/CLAUDE.md`.

## Reward redesign — registry + PBRS + the no-progress clock (`reward_manager.py`, `progress_clock.py`)

The reward (`Gen3RewardManager`) is organised as a **registry of class-tagged terms**
(design `designs/ai_v5/design_markovian_reward_and_features.md`). Every `RewardBreakdown` field is one
entry in `RewardBreakdown._REGISTRY` mapping name → `RewardClass`. The **BIAS class is folded
generically** off the registry (`_fold_bias_refund` sums `registry_fields(BIAS)`); TERMINAL and the
PBRS terms are **explicit named folds** (`_fold_material_pbrs` / `_fold_belief_pbrs` /
`_fold_status_pbrs` + the v13 `_fold_{progress,hazard,boost,opp_boosts}_pbrs`) because each PBRS term
carries its own `_prev_phi_*` telescoping state a generic
loop can't hold — `process_turn_reward` reads as a short phase sequence over these helpers:

- **TERMINAL** (`win_loss`, the ±30) — emitted as-is; never shaped/flag-affected. Out of scope.
- **PBRS** (always telescoping, objective-neutral; `Φ(terminal)=0`): `pbrs_material` (the material
  potential **Φ_mat**, design §2), `pbrs_belief` (the shipped incoming-KO belief PBRS — RENAMED from
  the mis-named `pbrs_material`), `pbrs_status` (the non-damaging-tempo status potential **Φ_status**,
  design §2.7 — `bias_redesign`- OR `all_shaping_pbrs`-gated, see below), and the **four v13/v14 end-state
  potentials** (see **End-state PBRS** below): `pbrs_progress` (**Φ_progress** =
  −`no_progress_penalty`·`progress_clock.value()`, the anti-stall clock as a telescoping potential —
  **`--stall-pbrs`-gated**; the other three are **`--all-shaping-pbrs`-gated**),
  `pbrs_hazard` (**Φ_hazard** = `HAZARD_WEIGHT`·(opp − our spike layers), design §2.6), `pbrs_boost`
  (**Φ_boost** = `BOOST_WEIGHT`·Σmax(0,our-active-boost)·hp_frac, the stored offense), and
  `pbrs_opp_boosts` (**Φ_opp_boosts** = −`OPP_BOOST_WEIGHT`·Σmax(0,opp-active-boost), the phaze value),
  and `pbrs_roar` (**Φ_roar** = −`ROAR_BOOST_WEIGHT`(0.25)·Σmax(0,opp-active-boost), the **DEDICATED**
  phaze-out-boosts PBRS — **folded INTO `--all-shaping-pbrs`** (no separate flag/version, owner request);
  same state-potential shape as `pbrs_opp_boosts` but its own weight, so a successful Roar pays out
  `+ROAR_BOOST_WEIGHT·(stages cleared)`. A PBRS can't be action-keyed without becoming a BIAS, so it IS the
  same potential — under `--all-shaping-pbrs` the two STACK; safe, both telescope to 0 → policy-invariant,
  the effect is just stronger proportional roar shaping).
  The field holds `γ·Φ(s′)−Φ(s)`; `PBRS_GAMMA` MUST ==
  the PPO gamma (asserted in `train_rl_agent.py` after the model is built — the manager is built first,
  in the env factory, so it can't assert in `__init__`).
- **BIAS** (everything else) — additive shaping whose additive↔telescoping mix is set by
  `--bias-additivity` λ∈[0,1] (`RewardConfig.bias_additivity`, default 1.0). Implemented as
  **accumulate-and-refund**: each BIAS term emits its current per-turn value; the manager accumulates
  `_bias_acc` and emits `bias_refund = −(1−λ)·Δacc` (the low-variance accumulator-potential spread). At
  **λ=1 the refund is identically 0** → byte-identical to the old additive biases (the no-op the
  registry-coverage / no-op-equivalence tests pin).

**Φ_mat** (`_compute_phi_mat`) = `MAT_HP_WEIGHT·(Σ our_hp − Σ opp_hp) + MAT_ALIVE_WEIGHT·(n_alive_ours
− n_alive_opp)`, over the **declared team size** (unrevealed opp mons = full-HP-alive → `Φ_mat(s_0)≈0`,
no opp-reveal jumps, no start-state variance). It REPLACES the old unconditional `hp_ours/hp_opp/
faint_ours/faint_opp` base spine — material no longer banks the lead, so every win returns +30 / loss
−30 (the clutch-vs-dominant fix). The old asymmetric `−0.75 FAINT_MATERIAL_PENALTY` is REMOVED (folded
into `MAT_ALIVE_WEIGHT=1.25`, a state potential, not a bias). The `+2.0` explosion literal is deleted
(survive-Explosion credit rides Φ_mat); `explosion_block` is kept.

**Φ_status** (`_compute_phi_status` / `_fold_status_pbrs`, `pbrs_status`) = `STATUS_TEMPO_WEIGHT·(opp_tempo
_statused − our_tempo_statused)` over **non-fainted par/slp/frz mons only** (`_TEMPO_STATUSES`). It
restores the *standing* value of a held non-damaging status that the event-form `status` reframe drops —
sleep/freeze/para "lose the opponent turns", value `Φ_mat` can't see (Toxic/burn/poison value is the chip
→ already in `Φ_mat`, so they're excluded to avoid a double-bridge). Nobody is statused at `s_0` →
`Φ_status(s_0)=0`, `Φ_status(terminal)=0` → it telescopes to **zero net** (policy-invariant dense signal,
not a net bias). **Gated on `bias_redesign`** (the default count-diff `status` BIAS already pays the
standing value → folding `Φ_status` there double-counts; OFF → `pbrs_status≡0`, `_prev_phi_status` stays
None, byte-identical default). It adds **no** resume-immutable field — it rides the existing
`bias_redesign` flag (design §2.7 / §7.4 hedge).

**The no-progress clock** (`ProgressClock`, `progress_clock.py`) is an episode-scoped
`turns_since_progress` counter **owned by `EpisodeTracker`** (NOT LiveView — it is cross-turn state;
precedent = `HiddenPowerTracker`). It is updated at `record()`/`embed_battle` time (so the obs is fresh
— poke-env runs `embed_battle` before `calc_reward`), and read by BOTH the obs encoder (`value()` →
the `vec[14]` scalar) and the reward (`last_penalty` → `no_progress_tax`), so **obs and reward key on
one value**. The ternary predicate per decision window: PROGRESS (our-attributed damage ≥3% / status
landed / hazard layer / forced opp commit / **an our-owned residual — Toxic/poison/burn or Leech
Seed/Curse/Nightmare — chipping the opp NET-down** → reset), DENIED (freeze), NO_OP (deliberate
wheel-spin → increment + charge, gated off on forced-switch windows and when no switch is legal).
DENIED splits two ways (`_denial_kind`): **exogenous** (miss / Protect-block / cant) is ALWAYS frozen;
a **productive heal** is frozen only for `HEAL_FREEZE_GRACE`=2 consecutive windows — a SUSTAINED heal
with no progress (the self-play mirror heal-war) then falls through to NO_OP and CHARGES, so the
250-turn stall finally registers. **Rest-loop (`gen3_rest_loop_stall_v1`):** a REST that already happened
this episode for the same species — i.e. our active woke and re-Rested — gets NO heal-grace at all
(`_update_rest_loop` sets `_is_rest_loop`, read in the heal branch), so a wake-then-re-Rest is a NO_OP
stalled turn the moment it repeats; a mon carrying **Sleep Talk** is exempt (looping Rest is a legitimate
act-while-asleep strategy, and our own moveset is fully known so the check is exact). **Setup-progress
(`gen3_setup_progress_v1`, unconditional correctness fix — clauses (vi)/(vii)/(viii) of `_is_progress`):** the
predicate had NO clause for an own stat-boost rising, a Substitute being made, or a Wish being cast, so a
PRODUCTIVE setup turn (a first Calm Mind / Dragon Dance / Swords Dance / Curse / Belly Drum, a fresh Sub, or a
Wish cast) was charged identically to an idle wheel-spin — the one stall-break route the reward actively
discouraged. Three clauses now count a **NON-redundant** setup as PROGRESS: our active's Σ positive boost
stages STRICTLY rose (a +6-capped repeat leaves the sum unchanged → still charged) OR a Substitute was NEWLY
created (a failed re-Sub while one is up → still charged) OR a **WISH was SUCCESSFULLY cast** (`gen3_wish_wired_v1`
— a pending ~50%-maxhp heal; a double-Wish FAILS → outcome 'fail' → still charged, keyed on the move id like
the Rest/Spikes clauses). Read from `live.ours.active.{boosts,volatiles}` + the delta's move id, with
`_prev_our_boost_sum`/`_prev_our_has_sub` trackers mirroring the spikes-layer pattern; the +6 cap +
Sub-can't-restack + its 25% HP cost bound how long it can keep resetting; in gen3 only our own move raises
our boosts and a switch-in is boostless (boosts reset on switch), so a pivot/opp action can't false-credit.
**Always-on** (not flag- or version-gated — a clock-predicate correctness fix like `gen3_rest_loop_stall_v1`,
but with NO `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump, so an in-flight run picks it up on resume). The
residual-PROGRESS
branch is what keeps a *winning* Toxic/Leech
defensive stall from being taxed (the discriminator is the opp net-losing HP; a heal-war where they
out-heal the tick still charges) — and because it runs FIRST, a winning *rest-stall* (Rest while Toxic
chips the opp down) is exempt too — validated end-to-end by `progress_clock_fuzz_test.py` (bridge, real
battles: a winning-residual window is never charged). The env (`gen3_env.py`) folds the delta once at
embed time, updates the clock, caches it for `calc_reward` (no double fold), and wires
`reward_manager.progress_clock = tracker.progress_clock`.

**Three futile-move short-circuits** (BEFORE the PROGRESS check, so an incidental opp switch — or, for
(3), a winning residual via clause (v) — can't launder them): **(1) capped Spikes** — Spikes used at the
3-layer cap can never add a layer, so it is charged as a NO_OP directly (a layer-ADDING Spikes still
resets via the hazard clause); **(2) filler RapidSpin** — RapidSpin with NO spikes on our side to clear is
a 20-BP filler pseudo-attack, so its trivial chip is barred from counting as progress and it falls through
to the NO_OP charge (a spin that genuinely clears our hazards, lands a KO, or is RNG-denied is handled
normally); **(3) wasted Refresh** (folded into `gen3_rest_loop_stall_v1`, `_is_wasted_self_cure`) — a self-status-cure
move (`cures_self_status`, i.e. Refresh) used with no status to cure (`our_status_cured is None`, not a cant)
does nothing, so it is charged as a NO_OP directly — crucially even when our Leech Seed / Toxic is chipping
the opp NET-down (which clause (v) would otherwise credit as progress), killing the observed
Refresh-spam-while-seeded stall (a Refresh that ACTUALLY cures a status sets `our_status_cured` → not wasted →
normal path). The first two target the self-play Spikes/RapidSpin wheel-spin loops the flat anti-spam taxes
missed; the third targets degenerate self-cure spam during a passive residual stall.

**Server-free reward parity (`reward_tracker.py`).** The offline reward path (`RewardTracker`, used by
`BattleRecorder` + the eval `RewardTrackingMixin`) has no `Gen3Env` to own the clock, so it OWNS a
per-battle `ProgressClock` itself and advances it before each `process_turn_reward` — mirroring the
env's embed-time timing. Without this, eval traces scored `no_progress_tax`=0 (clock absent) and the
prober **understated the training penalty on every stall/no-op turn**; now the recorded reward matches
training (the gate is still `all_shaping_pbrs`/`bias_redesign` in the run's `RewardConfig`, so a
default-config run stays byte-identical).

**Anti-stall terminal (`--draw-penalty`, default −30.0 = byte-unchanged).** The trainee FORFEITS a
stalled battle at the turn cap (`gen3_env` `ForfeitBattleOrder` at turn ≥ `StallConfig.threshold`), so
a 250-turn stall ends as a forfeit-**loss** (`lost=True`), NOT a tie. The terminal therefore detects a
timeout by **`live.turn >= _TIMEOUT_TURN_CAP`** (synced to `StallConfig.threshold`), not by won/lost:
`if won: +30; elif finished: draw_penalty if timed_out else −30`. Set `--draw-penalty -35` to make a
stall-to-cap strictly worse than a clean loss (cancels the γ=0.9999 discount pull of delaying an
inevitable −30). Resume-immutable, value-checked (`MODEL_CONFIG_VERSION 6→7`, `check_reward_config`).

**Staged rollout (`RewardConfig.bias_redesign`, `--bias-redesign`, default OFF).** OFF = the
**single-variable default run**: today's anti-spam taxes + roar/status/spikes, so the ONLY reward
change vs the live baseline is the material clutch-fix (clean attribution). ON = the no-progress clock
SUBSUMES the escalating anti-spam family (repetition/bouncing/dead-matchup/struggle suppressed) and the
clock charge is active. The `turns_since_progress` OBS scalar is present EITHER way (the clock always
tracks it), so both arms share one architecture and can A/B by resume. `--bias-additivity` /
`--mat-alive-weight` / `--bias-redesign` are resume-immutable, value-checked by
`ModelVersion.check_reward_config` (the same machinery as `--vf-coef`). Tests: `reward_redesign_test.py`
(registry coverage, Φ_mat telescoping + terminal-zeroing, **Φ_status non-damaging-only + gated-off-default
+ telescopes-to-zero**, bias no-op + parameterized blend, the bias_redesign reframes, the full
ProgressClock predicate), plus the updated `reward_manager_test.py`.

**Belief-risk-scaled switch BIAS lever (`--switch-bias-weight`, default 0.0 = OFF).** The shipped
`pbrs_belief` is policy-INVARIANT (a telescoping potential) so it can't move a *converged* under-switch
preference — verified on `run_20260607_102632`: switch-mass still inverts vs P(KO), stay-and-die ≈ 61%
== the V1 control. The fix (`design_reward_switching.md §7`, `impl_step6`) adds two **BIAS-class** terms
that *do* tilt the objective: `stay_risk_tax = max(−w·risk, −2.0)` for STAYING into a high imminent-KO
spot a safe pivot could escape, and `escape_risk_bonus = w·0.5·risk` for escaping it (asymmetric < the
tax → no farm). `risk = max(phys_pko,spec_pko)·(1−P(outspeed))` from the incoming belief. Hardened gates
(red-teamed): never tax a **trapped** stay (`_cur_can_switch` from the decision-time `ctx.mask`), an RNG
fizzle (`our_failed_to_move`), a KO'ing stay (`opp_fainted`), or a forced stay (a `_prev_safe_pivot`
bench mon with raw P(KO) ≤ `SAFE_PIVOT_PKO_MAX`=0.35 must exist; the escape bonus needs it too). Snapshots
are decision-time (set end of last turn / in `record_action`), read before `_fold_belief_pbrs` overwrites
them. **Reward-only — no obs/arch change** (ARCH unchanged; `MODEL_CONFIG_VERSION 4→5`), resume-immutable
(`check_reward_config`). Being BIAS-class it rides `--bias-additivity`, so a fixed weight at **λ=1 vs λ=0**
is the causal A/B for "is it the objective tilt that helps." Tests: `reward_redesign_test.py::TestSwitchBias`.

**HP-scaled self-KO penalty (`--self-ko-hp-penalty`, default 0.0 = OFF).** A grounded floor-leak fix
(2026-06-12 forensics on ai_v5_11): the policy confidently (median P≈0.5) explodes **healthy** mons —
~38% of all Explosion/Self-Destruct selections are at ≥80% HP (incl. turn-1 full-HP Metagross),
human-obvious blunders that cost ~0.95 mon. Mechanism (ruled out reward+exploration first): the
**reward is correct** (a healthy non-trade Explosion scores ≈−2.7; the finishing-blow mis-credit is
already guarded), but Φ_mat is **symmetric for a 1-for-1 trade** (our −hp/−alive cancels theirs → ~0),
so on the 77%-of-the-time trade the critic learns to value the post-self-KO board POSITIVELY
(measured `dV ≈ +2.9`), which **neutralizes the −2.7 reward in the PPO advantage** (`r+γV′ ≈ +1.5`, 74%
≥0) and the policy never un-learns it. (It is NOT `value_active_readout`/① — the no-① baseline explodes
just as much.) The fix is a **BIAS-class** term `−w·(our active HP fraction at decision time)` charged
when our mon self-KOs (`our_move_id ∈ SELF_KO_MOVES` + `we_fainted` + not `our_failed_to_move`), using
the `_our_active_hp_before` snapshot from `record_action`. Scaling by HP **spares the legitimate low-HP
"explode a dying mon for a KO"** (≈0 penalty). A static pre-check showed `w≈2.5` flips the healthy-trade
advantage negative; in a retrain the critic's over-valuation also drops as the TD target sharpens.
Reward-only — no obs/arch change (no `ARCH_SIGNATURE` bump; `MODEL_CONFIG_VERSION 11→12`),
resume-immutable (`check_reward_config`). **Validate by watching `win_rate_vs_bots` (82%→~95% target)
and the healthy-explosion rate fall.** Tests: `reward_redesign_test.py::TestSelfKoPenalty` (unit) +
`self_ko_penalty_fuzz_test.py` (bridge — real Explosion turns net exactly `−w·hp`, 0 elsewhere, OFF
byte-unchanged).

**De-bias cleanup (`--drop-redundant-bias` / `--drop-switch-bias`, default OFF).** A distortion audit
(ranking the BIAS terms by their ability to move the converged optimum away from win-maximization)
flagged three TIER-1 distorters; these two flags ZERO them in `_apply_bias_drops`, called **right
before** `_fold_bias_refund` so the dropped terms leave the bias accumulator too. Both default OFF =
byte-identical (the no-op tests pin it); each is resume-immutable + value-checked
(`MODEL_CONFIG_VERSION` v13, `check_reward_config`), no `ARCH_SIGNATURE` bump (reward-value only).
- **`--drop-redundant-bias`** drops `stall_tax` (a raw-turn-count ramp that also taxes a *winning*
  long game — the progress-aware `no_progress_tax` clock + the `--draw-penalty` terminal already cover
  stalling) and `matchup_penalty` (the same incoming-KO threat signal as the telescoping `pbrs_belief`
  PBRS term, but BIAS-class/additive → it distorts where `pbrs_belief` is policy-invariant).
- **`--drop-switch-bias`** drops the HAND-CODED switch-strategy subsidy (`switch_base`,
  `switch_bouncing_tax`, `escape_threat_switch`, `se_switch`, `pivot_protect/status/damage`,
  `sleep_out/in`) — switching value is LEARNABLE from `Φ_mat` + `pbrs_belief` + win/loss, so
  hand-rewarding it is a `provide-vs-learn` violation that biases the objective.

Two flags (not one) so the low-risk redundant removes can be attributed separately from the
behaviorally-uncertain switch family (which may have been doing real exploration-acceleration work).
The historical worst distorter — `finishing_blow` rewarding a self-KO Explosion — is already fixed
(guarded + the `+2.0` literal deleted), so it is not in scope. Tests:
`reward_redesign_test.py::TestBiasDrops` + `snapshot_test.py` (resume-immutability + v12→v13 migration).

**End-state PBRS — TWO switches (`--all-shaping-pbrs` + `--stall-pbrs`, both default OFF; v14/v15).** The
FINAL stage of the staged PBRS rollout: convert the last BIAS shaping to policy-invariant telescoping
potentials. Deliberately TWO switches so the stall tilt (which carries a documented regression risk) can
be A/B'd separately from everything else.
- **`--all-shaping-pbrs` ("everything but stall")** — (1) **folds** `Φ_hazard` =
  `HAZARD_WEIGHT`·(opp − our spike layers, design §2.6), `Φ_boost` = `BOOST_WEIGHT`·Σmax(0,our-active
  boost)·hp_frac, `Φ_opp_boosts` = −`OPP_BOOST_WEIGHT`·Σmax(0,opp-active boost), **and `Φ_status`**
  (its gate is now `bias_redesign OR all_shaping_pbrs`, so the tempo-status standing value is carried
  even without `--bias-redesign`); (2) **zeros EVERY BIAS term EXCEPT the anti-stall tilt
  `no_progress_tax`** — so `status`, `stall_tax`, `matchup_penalty`, the switch family, the anti-spam
  family, `spikes`/`futile_*`/`boost_utilized`/`roar`, and the redundant good-outcome bonuses
  (`finishing_blow`/`explosion_block`/`status_wasted`) all go. It also **activates the clock charge**
  (gate `bias_redesign OR all_shaping_pbrs`) so `no_progress_tax` is live as the kept tilt.
- **`--stall-pbrs` ("stall")** — **folds `Φ_progress`** = −`no_progress_penalty`·`progress_clock.value()`
  (the anti-stall clock as a telescoping potential) and **zeros `no_progress_tax` + `stall_tax`**, so the
  anti-stall signal is policy-invariant too.

Run **both** ⇒ the WHOLE BIAS class is zero → TERMINAL + PBRS only (fully policy-invariant). Run **only
`--all-shaping-pbrs`** ⇒ everything-else is PBRS but the progress-aware `no_progress_tax` survives as the
single acknowledged BIAS tilt (insurance against stall-regression — watch the stall-rate canary; the
terminal `--draw-penalty` remains the objective anchor either way). The zeroing lives in
`_apply_pbrs_suppression(bd)` (loops `registry_fields(BIAS)`, skipping `no_progress_tax` under
`all_shaping_pbrs`; zeroing the two stall terms under `stall_pbrs`), called **after** all PBRS folds +
the `_last_attack_had_effect` read and **before** `_apply_bias_drops` → `_fold_bias_refund`, so zeroed
terms leave the bias accumulator. Each new fold early-returns unless its switch is set, so with both OFF
the `_prev_phi_*` slots stay None and the four `pbrs_*` fields stay 0.0 → **byte-identical default**
(pinned by the no-op-equivalence + registry-coverage tests). Composes with the v13 drops (orthogonal,
run after). Resume-immutable + value-checked alongside the **now-recorded `no_progress_penalty`**
(Φ_progress's weight) — `MODEL_CONFIG_VERSION` v14/v15, `check_reward_config`, no `ARCH_SIGNATURE` bump.
Tests: `reward_redesign_test.py::{TestProgressPBRS, TestHazardPBRS, TestBoostPBRS, TestOppBoostsPBRS,
TestEndStateDrops, TestAllShapingPbrsNoOpDefault}` + `snapshot_test.py` (resume-immutability + v13→v14 +
v14→v15 migration).

## State-conditioned defensive-exploration entropy (`--defensive-entropy-boost`)

`gen3_defensive_entropy_v1` — the answer to "the model under-uses Recover/Soft-Boiled/Wish/Refresh/Heal Bell
when safe" that does **NOT** touch the reward (so it can't create a stall incentive). Instead of biasing toward
healing (which would force you to hand-draw the good-defense-vs-stall line), it **explores** defensive moves more
and lets the *existing* anti-stall reward (the `--draw-penalty` + the no-progress clock) be the guardrail: the
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

## MatchupSpec — the declared matchup (`matchup_spec.py`)

**The ONE explicit declaration of what a run's battles look like** (design:
`designs/ai_v8/design_matchup_config.md`, P0 built). One week produced four independent failures with a
shared root — *the matchup a run plays is assembled implicitly across seams that nothing forces to
agree*: the eval worker rebuilt its own default teams (specialists measured OOD), the env's single
`team=` fed BOTH sides (the training mirror), training/eval play modes drifted (stochastic
noise-farming), and the launcher's exit summary resolved "Last model" to a global-glob golden. The spec
makes the matchup EXPLICIT: built ONCE in `train_rl_agent` (`MatchupSpec.from_args(args)`), then
CONSUMED — never re-derived — by the consumers (the `plan.json` pattern).

- **`TeamSource`** — where one side's teams come from; its `build(all_teams, sample_teams)` is the ONLY
  constructor of that side's `Gen3Teambuilder` (the env factory no longer assembles builders inline).
  Kinds: `pool` (opponent default), `default_biased` (trainee default — full pool + 10% sample-team
  bias, `DEFAULT_TRAINEE_BIAS_PROB`), `pinned` (`--trainee-team`), `pin_biased` (the future
  `--trainee-team-prob` shape — supported, no CLI yet). Each is byte-parity with the legacy
  construction (pinned by `matchup_spec_test.py`). **The two sides are independent BY CONSTRUCTION**
  (`trainee_teams` / `opponent_teams` → `Gen3Env(team=, opponent_team=)`) — the mirror-bug class is
  structurally closed.
- **`PlayMode`** — how the frozen-NN opponents select actions (greedy | stochastic@temp, schedule
  fixed | anneal | ratchet). Descriptive in P0 — the executors (RLPlayer temp, the anneal/ratchet
  callbacks) already exist; the spec records the intent so echo/provenance say what a metric was
  measured under. `eval_opponent_play` defaults greedy; `eval_trainee_teams` defaults to
  `trainee_teams` (**the eval-OOD fix made structural**: eval pilots what training pilots).
- **Provenance** — `to_dict()` (pin fingerprints via sha1, not full text) + `spec_hash()` (a 10-hex
  **measurement-regime tag**: two runs/eras with different hashes are NOT metric-comparable) are
  stamped into `metadata.json` beside `cli_args` (`_matchup_spec` / `_matchup_spec_hash`).
- **Startup echo** — `summary_lines()` emits a `🧭 [MATCHUP <hash>]` block to the launcher Events
  panel: trainee teams, opponent teams + mix, exploiter target + play mode, eval regime — one glance
  at what the run actually plays.
- **The realized-matchup fuzz** (`poke_env_gaps/matchup_realized_fuzz_test.py`, bridge, no server) is
  the permanent mirror-catcher: it drives the REAL construction path (spec → builders →
  `Gen3Env(team=, opponent_team=)` → bridge) over real battles and asserts per episode that the
  trainee fields EXACTLY the declared pin, the opponent does NOT (the mirror signature), and opponent
  rosters VARY across episodes. P1+ (not built): controllers keyed on eval play modes, per-row regime
  tags, per-opponent team pools.

### Matchup provenance (what a run trained/evaled against — the diligence layer)

Four self-describing records, all metadata-only + additive (old readers unaffected), closing the
"a row/trace/checkpoint can't say what regime produced it" gap the OOD-eval era exposed:

- **`eval_results.jsonl` rows carry `matchup_hash` + `externals`** (`append_eval_result_row`):
  each append-only ladder row is stamped with the run's CURRENT declared-matchup hash (rows from
  different regimes/eras are distinguishable IN-FILE, not by dates), and the per-cycle vs-target
  record (`{ext label: {win_rate, counts}}` — e.g. the exploiter VERDICT) now survives in the
  jsonl instead of only the overwritten `latest_eval` + TensorBoard. Externals stay OUT of `bots`
  (the ELO fit's ladder is untouched).
- **`metadata.json:matchup_history`** (append-only, maintained by `save_model_snapshot` from the
  `cli_args` stamp): one `{hash, spec, recorded_at}` entry per ERA — a resume that changes the
  declared matchup appends a new era instead of silently overwriting the old one (cli_args keeps
  only the latest). Saves without cli_args (the periodic-checkpoint path) preserve it.
- **The resume MATCHUP-DRIFT guard** (`train_rl_agent`, warn-not-fatal): a `--model` resume whose
  declared matchup hash ≠ the run's recorded one emits a loud `⚠️ [MATCHUP DRIFT]` + the
  field-level diff (`matchup_spec.describe_drift`) — a mid-run curriculum change is legitimate,
  doing it SILENTLY is not. Launcher restarts forward flags verbatim → never fire it.
- **`eval_manifest.json` records the eval REGIME**: `matchup_hash`, `trainee_team_sha` (the pin
  the trainee piloted; None = pool), `opponent_pins` ({ext label: sha} for fold-back-pinned
  opponents) — a trace dir is self-describing about HOW its numbers were measured.
- **Checkpoint sidecars + `snapshot_history` entries carry `matchup_hash`** (via
  `record_checkpoint` → `_build_snapshot_entry`, like the `latest_eval` stamp) — each checkpoint
  is self-describing about what it was training against as of its save, robust to later eras.

Readers: `snapshot._read_matchup_hash(model_dir)` (current era) /
`snapshot.read_recorded_matchup(model_path)` (the drift guard's input). Tests:
`snapshot_test.py::test_matchup_*`/`test_eval_row_*`/`test_checkpoint_sidecar_*`,
`matchup_spec_test.py::test_describe_drift_*`, `eval_callback_test.py::test_eval_manifest_records_the_regime`.

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster.** Eval fires every `EVAL_FREQ_STEPS` (2M steps) and plays
`EVAL_GAMES` (100) games per opponent — one cadence, one game count, applied uniformly to
every bot *and* every self-play sentinel (no maturity tiers, no per-opponent caps). The
roster is the full set of eight archetype bots — both the v1 and v2 of each
(`heuristic`/`heuristic2`, `staller`/`staller_v2`, `aggressive`/`aggressive_v2`,
`setup_sweep`/`setup_sweep_v2`) — plus `random` as the eval-only "is-the-model-broken"
floor (excluded from `win_rate_vs_bots`). All nine are the single source of truth in
`_EVAL_OPPONENT_SPECS` / `eval_opponent_names()`, shared by the bot path, the self-play
path, and the worker. There is no roster flag — every bot always plays, because they play
differently and the playstyle diversity is the point. The flat numbers are safe precisely
because eval is non-blocking and **skips a cycle while the previous one is still running**
(below): a heavier roster self-throttles to a sparser cadence instead of needing tuned
ceilings.

`PerOpponentEvalCallback` (non-self-play path) does **not** eval in-process. On each
scheduled step it snapshots the live weights (`model.save`) and spawns `--eval-workers`
(default 3) `main.eval_worker` subprocesses that **work-steal at battle granularity** from a
shared pool, load the **frozen** snapshot, and play against the shared Showdown server (or the
in-process bridge) **without pausing training**. **The trainee's eval teambuilder follows the
run's `--trainee-team` pin** (`trainee_team_str` in the worker cfg → `eval_worker._build_trainee_tb`;
threaded by BOTH callbacks): a specialist run is measured piloting ITS OWN team. The worker used to
hardcode the default full-pool builder, so every specialist eval (win rates / ELO / `vs_ext`
verdicts) measured the model piloting random teams it never trained on — pure OOD; the
"ai_v7_05–08 plateau" was this instrumentation gap, not the training (see `eval_worker_test.py`,
the fix's pin). No pin → the default pool builder, byte-identical. **The companion TRAINING-side bug
(the mirror):** PokeEnv feeds its single `team=` kwarg to BOTH internal env agents, and the
per-episode opponent Players are decision-functions over `battle2` (agent2 does the networking), so
agent2's `_team` decides the opponent's REAL team — a `--trainee-team` pin therefore also pinned the
OPPONENTS, turning every specialist run's training into a single-team MIRROR vs bot pilots
(genuinely-won ~100% training WRs, fake curriculum; a probe on the exact path measured the same
checkpoint at 1.000 mirror vs 0.483 with real opponent teams). Fixed by the `Gen3Env(opponent_team=…)`
post-init seam (the `_battle_class` injection pattern), threaded unconditionally from the env factory
(`opponent_teambuilder`); `None` = the pre-fix both-sides behavior. Pinned by `gen3_env_test.py`. Each opponent's `EVAL_GAMES` are split into
**shard units** of `--eval-shard-games` (default 25 → 4 shards/opponent); a worker claims units
(atomic `O_EXCL` lock per `unit_id`), plays them, and publishes one `shard__<unit_id>.json` of
**raw** counts; the parent pools an opponent's shards back into one **exact** result. This is the
long-tail fix — when fewer opponents remain than workers, the straggler's remaining games spread
across idle workers instead of one worker grinding a whole opponent alone (workers are capped by
unit count, not opponent count). The whole mechanism lives in the **`eval_sharding/` package**
(below); when all workers finish the parent merges → TensorBoard + TUI + best-model (the winning
snapshot is promoted by copy, not re-saved). Forensic traces land under
`<run_dir>/eval_traces/step_<N>/<opponent>/` as a per-captured-battle triple (`write_battle_record`,
`battle_recorder.py`): `<outcome>_s<shard>_NNN_summary.json` (the human-readable per-decision dump —
each invocation also carries a **`belief`** block, the model's top-`BELIEF_TOPK` (3) most-likely species
per still-HIDDEN opp slot, present ONLY when the hidden-opponent belief is on and a slot is un-revealed;
`RLPlayer._decode_belief` → `inference/belief_decode`, see `src/agents/model/CLAUDE.md`) +
`<outcome>_s<shard>_NNN_states.npz` (raw obs/logits/values **+ the chosen `actions`** for the prober
and offline obs replay) +
**`<outcome>_s<shard>_NNN_replay.html`** — a self-contained, **browser-watchable** Showdown replay of
that battle (poke-env `save_replay` over the accumulated protocol stream). The first two are
prober-only; the HTML lets a human just open the game in a browser (no checkout, no prober) — the
only watchable replay for *non-stall* eval battles (stall games still get their own `stalls/*.html`).
The `s<shard>_` prefix namespaces the files so concurrent shards of one opponent never collide.
The filename stem is built by the single helper **`trace_filename_stem(outcome, trace_tag, idx)`**
(`<outcome>_<trace_tag><idx:03d>`) — the **one source of the naming contract** the prober's
`discovery._FNAME_RE` must invert. (When sharding added the `s<shard>_` infix, the prober's regex
didn't follow → every sharded trace parsed as outcome `"?"` and the **whole prober went blind**;
`eval_callback_test.test_trace_naming_contract` now pins that `discovery` parses exactly what
`trace_filename_stem` emits, so the producer↔consumer pair can't silently drift again.)
On `--use-showdown-bridge` runs each trace also gets a fourth sibling,
`…_NNN_reconstruction.json` — the battle's **full-information reconstruction record** (resolved
PRNG seed + both packed teams + the raw command log), captured at the bridge layer and joined to
the trace by battle tag (`utils/bridge/reconstruction.py`). It makes the battle fully replayable
and turn-re-rollable offline (`replay_battle` / `reroll_turn`), and
`agents.training.obs_materializer` can rebuild the trainee's one-sided obs from it bit-for-bit
(guarded by `obs_roundtrip_fuzz_test.py`). It is referee-view data in a **separate artifact** on
purpose — nothing in the obs/training path reads it (the one-sided/omniscient wall; see the bridge
README). Websocket eval simply doesn't produce it (degrades gracefully). All
three sit alongside a per-cycle
**`eval_manifest.json`** (`write_eval_manifest`) recording exactly which model produced them
— `num_timesteps`, `git_hash` + `arch_signature` (read from the run's `metadata.json` /
`model_config.json`), and a `snapshot` pointer. The eval snapshot is normally ephemeral
(`model.save` → workers load → deleted in `_cleanup`) and the eval `step` rarely lines up with
a persisted `<run>/checkpoints/checkpoint_<N>_steps.zip`, so the prober can't reload the *exact* weights unless
they're retained: `--keep-eval-snapshots N` copies the snapshot into
`eval_traces/step_<N>/snapshot.zip` (keeping the N most-recent) and points the manifest at it.
The prober consumes the manifest to load the exact model, falling back to the nearest
checkpoint. **The trainer grooms the traces it writes**: after each cycle
`_prune_eval_traces` keeps only the `--keep-eval-trace-steps` (default 20) most-recent eval
step dirs, and `_prune_eval_snapshots` keeps the `--keep-eval-snapshots` (default 10)
most-recent snapshots — so `eval_traces/` stays bounded without any external task
(`python -m main.prober.groom` is the manual fallback). **The same cycle also bounds the run's
two append-only debug dirs** via `_prune_run_artifacts` (`artifact_retention.py`, a dedicated
module — not bolted onto this busy callback): keep the `--keep-stalls` (default 50) most-recent
`stalls/stall_*.html` replays and the `--keep-crashes` (default 10) most-recent
`crashes/restart_err_*.txt` launcher diagnostics, newest-by-mtime, `0` = keep all. Same
producer-grooms-its-own-data contract; `python -m agents.training.artifact_retention <run_dir |
models_dir> [--apply]` is the manual fallback (dry-run by default; sweeps every run under a
`models/` tree). The eval summary itself is
written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, NOT
nested under a checkpoint) — robust to the async timing (an eval can finish after a
newer checkpoint, or before any checkpoint exists); `save_model_snapshot` carries it
forward so a later checkpoint never erases it. That top-level block is the canonical,
timing-robust record; **additionally, `record_checkpoint` stamps a point-in-time copy
of the then-current `latest_eval` into each checkpoint's entry** (both the per-checkpoint
sidecar `.json` and the run-level `snapshot_history` entry, under a `latest_eval` key) so
each checkpoint carries the most-recent eval+pool stats as of when it was saved. The
embedded block keeps its own `step`, so storing it under a possibly-newer checkpoint never
mislabels which weights were measured (`snapshot._read_latest_eval` reads it; the union
builder `_build_snapshot_entry` keeps sidecar + history in lockstep).

The frozen snapshot makes parallel eval correct (a worker can't read mutating in-memory
weights), and the fresh process returns all eval memory to the OS on exit (no fragmentation
in the trainer). Behaviors:
- A trigger that fires while the previous cycle still runs is **skipped** (logged) — on CPU
  an eval can outlast its interval; cadence just goes sparser.
- A worker crash is **logged-and-continued**, never fatal (its opponents are just missing
  for that cycle).
- **An operator can force an off-cadence eval** from the launcher's `f` button (confirm →
  SIGUSR2). The signal handler (`train_rl_agent._setup_signal_handlers`) only flags a
  process-global `request_forced_eval()`; whichever eval callback is active CONSUMES it on its
  next `_on_step` (the shared `eval_callback._ForcedEvalMixin._maybe_force_eval`, mixed into BOTH
  callbacks so the path can't drift) and launches a cycle immediately. A request that arrives
  while a cycle is already in flight is **rejected** and reported to the launcher Events panel —
  the same skip-while-running rule as the normal cadence. The forced launch consumes the current
  cadence bucket (`_last_eval_step = num_timesteps`) so the schedule check can't double-launch the
  same step; the next boundary still fires normally. Tests: `eval_callback_test.py` /
  `selfplay_callback_test.py` (`test_force_eval_*`).
- **Graceful shutdown waits for eval to finish**: a scheduled restart is self-initiated by
  `GracefulRestartCallback` at a rollout boundary and the launcher won't force-kill until the
  child overruns the deadline by `--restart-grace-minutes` (20 min), so the drain budget is a
  full `_ABORT_EVAL_DRAIN_SEC` (10 min) AFTER the checkpoint is saved — long enough for a CPU
  eval to complete. Even the pathological forced-SIGTERM case (already overran → ~90s SIGKILL)
  is safe: the checkpoint is saved first, only the in-flight eval can be lost.
- **On resume the last eval is re-published to the TUI** from the resumed checkpoint's
  `metadata.json` (`replay_last_eval_to_tui`), so the eval panel isn't blank until the next
  cycle. This covers the **self-play `pool` block too** — the aggregate (`win_rate_vs_pool`,
  `mean_reward_vs_pool`, monotonicity, snapshot count) and every per-sentinel row are
  re-published from the saved block, with the saved step tags, so Pool/sentinel rows survive
  a restart exactly like the bot rows (no waiting a full cadence for fresh numbers). Safe
  because the pool only changes at an eval-collect — the same moment the block is persisted —
  so the saved rows match the pool reconstructed from `snapshots/`. A pre-seed eval persists an
  empty `sentinels` list, which isn't re-published (nothing to show yet).

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `5` | Eval subprocesses per cycle; work-steal **shard units** from a shared pool. Capped at the unit count (≈ opponents × shards-per-opponent, so sharding lets the full pool help). Self-play doubles this (→ `10`) since sentinel matchups run the model for both players. |
| `--eval-shard-games` | `25` | Games per work-steal **shard unit** (battle-level work-stealing). Each opponent's `EVAL_GAMES` split into chunks any idle worker drains → the long tail collapses to one shard (≈4-shards-per-opponent default = ~4× shorter tail). Smaller = finer tail collapse but more player builds / (on websocket) more connection churn — the bridge is preferred for fine shards. `>= EVAL_GAMES` ⇒ one shard/opponent = the original opponent-level behaviour. Aggregation is exact (Σwon/Σfinished etc.); see the package below. |
| `--eval-device` | `cpu` | Device for eval-worker inference. `cpu` decouples eval from the training GPU. |
| `--eval-concurrency-per-worker` | `1` | Battles each worker overlaps **within** its claimed opponent (single-thread asyncio latency-hiding — NOT multi-core). `1` = today's sequential play. Threaded to the constructor's `eval_concurrency` → `cfg["concurrency"]` → `run_local_battles(concurrency=)` (bridge) / the player's `max_concurrent_battles` (websocket). See the concurrency note below. |
| `--keep-eval-snapshots` | `10` | Retain the N most-recent eval weight snapshots in `eval_traces/step_<N>/snapshot.zip` (~27MB each; default ≈270MB) for bit-exact prober replay. `0` writes the identity manifest only; the prober then loads the nearest persisted checkpoint. The trainer auto-prunes to this cap each cycle. |
| `--keep-eval-trace-steps` | `20` | The trainer keeps only the N most-recent eval **step dirs** under `eval_traces/` after each cycle (`0` = keep all), so forensic data stays bounded. `python -m main.prober.groom` is the manual fallback. |
| `--keep-stalls` | `50` | Each cycle keep only the N most-recent `stalls/stall_*.html` replays (`0` = keep all). A self-play run writes thousands (~80 KB each); this caps the dir. `artifact_retention.py`; CLI fallback `python -m agents.training.artifact_retention`. |
| `--keep-crashes` | `10` | Each cycle keep only the N most-recent `crashes/restart_err_*.txt` launcher diagnostics (`0` = keep all). Same module/CLI as `--keep-stalls`. |

**TD-residual tail metric (`eval/td_resid_tail_*`).** Each cycle also folds a **left-tail
statistic of the per-decision critic surprise** δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t) — the same
formula the prober uses (`main/prober/session.py::_td`, the single source of truth). `BattleRecorder`
accumulates δ live (one-step delayed backfill, closing each transition at the next `record()` when
the reward is finalized and V(s′) is known; the last decision has no δ). It costs **zero extra GPU**:
δ is computed only over the battles eval already captures forensically (where `need_aux=True` already
paid for V(s)), pooled per opponent (one `EvalRLPlayer` per matchup → `td_tail()`), and folded as a
**CVaR@5%** (mean of the worst 5%, `TD_TAIL_FRAC`; single min below `TD_TAIL_MIN_SAMPLES`=20). It
rides the exact win-rate plumbing — worker `shard__<unit_id>.json` (raw δ pooled across shards) → `merge_eval_results` →
`eval/td_resid_tail_vs_<opponent>` + `eval/td_resid_tail_mean` (TB + TUI), the `metadata.json`
`latest_eval` block (per-opponent + pool aggregate), and the append-only `eval_results.jsonl`. The
run's `model.gamma` is threaded into the worker (`base_cfg["gamma"]`) so the live δ matches the
prober's offline recompute (guarded by `td_residual_parity_fuzz_test.py`). More-negative = the critic
got blindsided more often — the **leading indicator for the critic-coverage obs work** (it moves in a
cycle or two, where saturated win-rate / gate-pinned `win_rate_vs_pool` / wide-CI ELO don't).

**Intra-worker concurrency (`--eval-concurrency-per-worker`, default `1` = sequential).** Each
worker overlaps up to N battles **within** its claimed opponent. This is **single-thread asyncio
latency-hiding, NOT multi-core** — everything (the obs build + PyTorch forward in `choose_move`, the
bridge/server I/O) runs on the one `POKE_LOOP` thread with BLAS pinned (`OMP/MKL=1`), so concurrency
only overlaps the time a worker is *blocked* on the bridge subprocess / websocket round-trip with
another battle's forward. The ceiling is **one core of compute**: a single-core bridge benchmark
(`/tmp/eval_concurrency_bench.py`, NN trainee vs bot and vs NN sentinel) measured ~**2.0× decisions/sec
at conc=3** on spare cores (plateau ~3; bot eval ≈2.0×, the heavier NN-vs-NN ≈1.8×) — i.e. about half
the per-decision wall-time at conc=1 was bridge I/O wait. **The old `_EVAL_SUBPROCESS_CONCURRENCY` = 1
default and its "measured slower" note were the *saturated* regime** (eval contending with training's
64 env workers for already-full cores — there the extra event-loop overhead nets negative); on **spare
cores (idle box / the cycle tail)** it's a clean ~2×. So the live gain runs between 1× and 2×
depending on how saturated the box is during the eval window; default stays `1` (opt-in). It does
**not** use idle cores at the tail — that needs *process-level* sharding (chunk one opponent across
workers); concurrency stacks multiplicatively on top of that (≈`2 × #shards`). Cross-opponent
parallelism is still the `--eval-workers` (5) subprocesses work-stealing the pool.

### Battle-level work-stealing (`eval_sharding/` package)

The *process-level* tail fix above is the `eval_sharding/` package — a small, deeply-encapsulated
unit with a narrow interface (4 focused files, no mega-file):

- **`units.py`** — `EvalItem` (one opponent the parent declares) + `ShardUnit` (a chunk of its
  games) + `plan_units(items, shard_games)`, a **pure** partition: split each item's games into
  ≤`shard_games` chunks (Σshards == n_games exactly), ordered LPT-ish (cost-descending items, shards
  round-robined) so every opponent starts early and the expensive ones lead.
- **`results.py`** — `ShardResult` (raw additive metrics: won/finished, reward+turn sums, the raw δ
  list — never a reduced ratio) + `aggregate`, which pools an opponent's shards back **exactly**:
  win_rate=Σwon/Σfinished, reward/ep_len count-weighted, and the TD tail by **pooling raw δ then one
  `td_tail`** (a CVaR can't be averaged). `td_tail` + its constants live here (the single source of
  truth; `eval_callback` re-exports them, so the dependency is one-way `eval_callback → eval_sharding`).
- **`pool.py`** — `ShardedEvalPool`, the deep coordinator. Parent: `write_plan(run_dir)` →
  `collect(result_dir)`. Worker: `from_plan(run_dir)` → `claim_next(claim_dir)` / `publish(...)`. It
  hides every filesystem mechanic; the worker never touches a lock file, the parent never touches a
  shard file. The plan (`plan.json`, items + shard_games) is the **single source of truth** both
  sides read — neither reconstructs the universe independently, so they can't drift.
- **`merge_eval_results`** is now a thin delegate to `ShardedEvalPool.collect` returning the same
  `merged` shape every downstream consumer already reads (record_per_opponent / build_bot_eval_block
  / record_elo / pool & externals blocks are **untouched**), plus additive `counts` (exact W/L) and
  `coverage` siblings.

**Exactness caveat (documented, by design):** win_rate / reward / ep_len are exact regardless of
`shard_games`. `td_resid_tail`'s *aggregation* is exact (pool the raw δ, compute the CVaR once), but
the *captured-battle sample* it's computed over shifts slightly with the shard count — the forensic
capture quota is per-unit (scaled `max(1, ⌈quota/shards⌉)`), so which battles contribute δ depends on
the split. It's a sampled diagnostic either way. Forensic trace files are namespaced by a per-unit
`trace_tag` (`{outcome}_s{shard}_{idx}`) so concurrent shards of one opponent don't collide in the
shared `eval_traces/step_<N>/<opponent>/` dir. Per-cycle `run_dir` is wiped at cleanup (and cleared
at launch), so no lock/shard/plan ever leaks across cycles. Sentinel/fixed opponent models are cached
per worker by path (immutable within a cycle → safe; the version check rides the first load) so a
fine split doesn't pay an N× 27MB deserialize. Worker rewrite: `eval_worker._play_unit` (one fresh
trainee + opponent per unit → independent measurement) + a per-worker model cache; tests:
`eval_sharding_test.py` (partition + aggregation-exactness property + claim-once + coverage),
`eval_sharding_fuzz_test.py` (real bridge battles through the real worker → exact pooled result).

### Rating-model seam (`rating.py`) — extensibility for Glicko-2 / TrueSkill

The live skill rating is anchored Bradley-Terry (`elo.py`), a *global batch* fit. `rating.py` is the
**ready drop-in point** for a different model without re-plumbing: `MatchRecord` (exact counts +
draws + `period_id` + optional opponent priors — the union BT, Glicko-2 and TrueSkill all need),
`RatingResult`, a `RatingModel` **batch** protocol, and `BradleyTerryRating` — a thin adapter over
`elo.fit_pairwise` whose ratings+SE are **byte-identical** to the live fit (pinned by `rating_test.py`).
`eval_rows_to_match_records` bridges the existing `EvalRow` history. The live `record_elo` path is
**deliberately unchanged** (zero risk): the seam exists and is tested, but routing through it buys
nothing until a new model is actually wanted — and Glicko-2 is *sequential* (period-by-period RD
carry-forward), so it needs the `SequentialRatingModel` sibling sketched in the module footer, not the
batch `fit`. Data fidelity is already in place: `eval_results.jsonl` now carries exact per-opponent
`counts` (additive, backward-compatible), so a future Glicko backfill has an exact ladder even under
partial shard coverage (where `win_rate × n_games` would be ambiguous).

## Self-play opponents (`--self-play`, gated behind pathology hunting)

When `--self-play` is set, `SelfPlayCallback` replaces `PerOpponentEvalCallback` and the
training opponents become frozen snapshots of the agent itself, drawn from a directory-backed
`SnapshotPool` (`snapshot_pool.py`; state reconstructed from `<run_dir>/snapshots/` on every
restart — no manifest). Design lives in `designs/ai_v5/`. Key behaviors:

- **Eval + promotion are NON-BLOCKING (frozen-snapshot subprocess), mirroring
  `PerOpponentEvalCallback`.** Self-play eval no longer runs in-process on the training thread.
  On a trigger step `SelfPlayCallback` freezes the live weights to disk (`model.save`) and
  spawns `--eval-workers`×2 (default 10) `main.eval_worker` subprocesses that **work-steal BOTH
  the bot roster AND up to `--n-sentinels` pool sentinels** (default 5; all split into shard units)
  from one shared pool (the
  worker's `_play_unit` SENTINEL branch plays the frozen trainee greedy vs each sentinel stochastic);
  training continues immediately. On a later
  `_on_step` poll the parent merges per-opponent + per-sentinel results → `win_rate_vs_bots` /
  `win_rate_vs_pool` / `sentinel_monotonicity`, records to TensorBoard + the TUI + metadata.json
  (with the `pool` block), persists `win_rate_vs_bots` (feeds `heuristic_fraction` next run),
  saves best by **copying** the frozen snapshot, and — if `win_rate_vs_pool > --promote-threshold`
  — **promotes the FROZEN snapshot into the pool by file-copy** (`SnapshotPool.add_from_path`):
  the live model has advanced since launch, so re-saving `self.model` would promote the wrong
  weights. Sentinels load via `load_model_snapshot` against the pool's shared `model_config.json`
  using `current_model_version(mappings)` — a stale-arch snapshot fails with `ModelVersionError`,
  never loads silently. The **only** training-thread work per cycle is the `model.save` freeze +
  one cheap `opponent_default_stats` IPC at collect; all battles / model loads / inference run in
  the worker processes, and the trainer holds no live eval connections (the worker rebuilds
  opponents/teambuilders/mappings itself). Skip-while-running, worker-crash-logged-and-continued,
  graceful-shutdown `drain()`, and resume-republish all behave exactly as the bot path above. The
  launch→poll→collect→drain mechanics are the **shared** `eval_callback.spawn_eval_workers` /
  `merge_eval_results` / `persist_eval_snapshot` / `prune_eval_*` / `replay_last_eval_to_tui`
  helpers, so the two non-blocking paths can't drift. `--debug --self-play --debug-eval` uses a
  fast eval cadence (every 4k steps, 3 games) so a short CPU smoke exercises seed → pool eval →
  promotion (a plain `--debug` smoke skips all eval by default — see `--debug-eval`).
- **Curriculum: thresholded ramp + LIVE per-episode fraction.** `heuristic_fraction`
  (`snapshot_pool.py`) is **0% self-play below `SELF_PLAY_START` (0.55)** — a weak model trains
  100% vs bots, no cycles wasted on a useless self-opponent — then smoothsteps `0.55→0.80` up to
  **90% self-play** (`HEURISTIC_FLOOR`=0.10 keeps a few % vs real bots for anti-forgetting). The
  three anchors are **configurable** — `--heuristic-floor` / `--self-play-start-wr` /
  `--self-play-full-wr` (defaults = the constants) thread through both the startup fraction and the
  live push, so a run can keep the coverage-punishing bots in the mix longer (raise `full` to ramp
  slower, raise `floor` for a bigger permanent bot slice). `--bot-weights name=w,…` additionally
  biases WHICH heuristic each episode draws (e.g. `aggressive_v2=3,heuristic2=3` → ~3× emphasis on
  the loss-analysis-flagged coverage bots; unlisted bots stay 1.0, omitted → uniform) — the weighted
  pick lives in `MaskableAgentWrapper._select_episode_opponent`, an O(1) in-memory `rng.choices`
  with zero per-step cost. All three default to the original behavior, so an unset run is unchanged.
  Crucially the heuristic-vs-pool split is **no longer fixed per process**: every training env
  picks its opponent **per episode** in `MaskableAgentWrapper.reset()` from a live
  `self_play_fraction`, and `SelfPlayCallback` pushes the fresh fraction (+ a `pool_generation`)
  to all envs via `training_env.env_method("set_self_play_target", …)` **after every eval**, so
  the ratio tracks measured strength mid-run with no restart. The opponent is a pure decision
  function over `env.battle2` (env.agent1/agent2 do the networking), so swapping it between
  episodes is free and safe — built `start_listening=False` (no idle connections), and the
  in-episode stale-decision path is untouched. The pool-vs-heuristic **coin flip is per-episode**
  (so the live fraction is honored exactly), but the pool **snapshot is (re)sampled+loaded only
  once per `pool_generation`**, NOT per episode: `load_model` deserializes a ~27MB MaskablePPO,
  and doing it every episode against an N-deep pool (LRU `lru_cache_size`=3) thrashed the workers
  — they blocked in `reset()` on the deserialize, dropping CPU to ~40% and FPS from ~1400 to ~500
  (regression fixed in `_select_episode_opponent`). A `pool_generation` bump (after a seed/promote)
  makes the worker re-scan + re-sample, so promotions become training opponents within a
  generation; diversity comes from 48 envs sampling independently + rotating each generation, not
  from per-episode churn. (`_n_pool_envs` / the `_maybe_engage_self_play` env-rebuild are gone.)
- **Opponent-mix reporting (`train/selfplay_fraction` / `train/stable_fraction` /
  `train/nonbot_fraction`).** The curriculum coin `sf` (`1 − heuristic_fraction(win_rate)`) pushed to
  the envs and persisted to `summary.json` is the **challenge-ENTRY** probability (= pool +
  un-mastered stable, *when* the challenge pick returns non-None) — NOT the pool share. So the
  reported metrics are derived separately by `SelfPlayCallback._opponent_mix_fractions(sf, pool_ready)`,
  a pure mirror of `MaskableAgentWrapper._select_episode_opponent` (it does **not** change selection).
  The four mutually-exclusive opponent types (bot / pool / un-mastered-stable / mastered-stable) sum
  to 1; the metrics report **`train/selfplay_fraction` = P(pool)** (REPOINTED — it used to log `sf`),
  **`train/stable_fraction` = P(any stable)** (un-mastered in the challenge **+** mastered in the
  weighted floor — a mastered stable "becomes a bot" so it's NOT in `sf`), and **`train/nonbot_fraction`
  = pool + stable** (= 1 − bot; bot is left implicit). `nonbot` is independent of the stable challenge
  share (it cancels); the per-bucket split needs three **reporting-only** inputs threaded into the
  callback from `train_rl_agent` (the capped `stable_challenge_share`, the `--bot-weights` vector, and
  `len(OPPONENT_CLASSES)` — the floor roster, which excludes eval-only `random`).
  With no stable opponents these reduce to `selfplay_fraction = nonbot = sf·P`, `stable = 0`.
  `_opponent_mix_fractions` is a hand-written **mirror** of the wrapper's selection, so the anti-drift
  guard is `wrappers_test.py::test_mix_fractions_match_actual_sampling`: it runs the REAL
  `_select_episode_opponent` thousands of times and asserts the empirical pool/stable shares match
  the analytic fractions (the per-case `selfplay_callback_test.py::test_opponent_mix_*` pin the math
  itself). A future selection change that isn't mirrored fails that cross-check.
- **Seeding is GATED on competence; the pool is a SLIDING WINDOW (nothing pinned) by default.** The
  pool is seeded only once win rate clears `SELF_PLAY_START` (at startup via `_maybe_seed_pool`, or the
  moment it crosses mid-run in `_collect_pending`), so the first self-play opponent is a
  *competent* model — never the random/weak step-0 seed of old. By default nothing is pinned: the
  oldest snapshot (incl. the seed) ages out as the window slides past `max_snapshots`, so the floor
  stays a recent self; anti-forgetting is the heuristic floor, not a pinned seed.
- **PFSP / league-lite (`--pfsp-scale`, `--pool-spread`; both OFF → byte-identical).** A pure recency
  window is a near-50% echo chamber (recent selves beat each other ~evenly), so it never up-weights the
  *kind* of self the trainee is actually losing to. Two opt-in knobs turn it into a prioritised
  curriculum:
    - **`--pfsp-scale S` (default 0.0)** — `SnapshotPool.sample()` blends a per-snapshot HARDNESS factor
      into the weight: `weight = recency × (1 + S·(1 − p))`, where `p` is the trainee's measured win-rate
      vs that snapshot. A self it loses to (`p→0`) is sampled up to `1+S`× more; one it dominates (`p→1`)
      keeps factor 1 — never starved, so coverage is preserved. An unmeasured snapshot uses the mean of the
      known rates (average difficulty); with **no** rates yet (cold start) every factor is 1 ⇒ pure recency.
      The per-snapshot win-rates are exactly the sentinel win-rates the eval already measures: each cycle
      `SelfPlayCallback._update_pfsp_ema` EMA-smooths them (`_PFSP_WR_EMA_BETA`=0.5, to damp ~100-game eval
      noise) and `_prune_and_push_pfsp` prunes the map to the live pool and pushes it to every env via
      `env_method("set_opponent_win_rates", {step: p})` (mirrors the `set_self_play_target` push;
      `MaskableAgentWrapper.set_opponent_win_rates` → `SnapshotPool.set_win_rates`). The map survives resume
      in `summary.json` (`pfsp_win_rates`). Headline signals: `eval/pfsp_hardest_win_rate` (the most
      up-weighted self) + `eval/pfsp_tracked_snapshots`. Try `1.0–2.0`.
    - **`--pool-spread` (default off)** — replaces the oldest-evicted window with **spread retention**
      (`SnapshotPool._evict_spread`): always keep the newest + the oldest (a weak early self = a forgetting
      tripwire PFSP can up-weight) and thin the most-redundant interior snapshot (smallest neighbour
      step-gap) to an even ladder. So PFSP weights over a genuinely diverse range of selves, not a
      recent-selves cluster. Pairs with `--pfsp-scale`; alone it just diversifies the window.

  Both are threaded into the `SnapshotPool` at **both** construction sites (the per-env-worker pool that
  samples, and the trainer-side pool used for honest sentinel-weight telemetry); off → no extra IPC and the
  legacy sampling/eviction byte-for-byte.

  **Honest caveats (it's a partial-coverage curriculum, not a full PFSP league):** (1) only the
  **`--n-sentinels` (default 5) evenly-spaced sentinels** the eval measures per cycle get a fresh win-rate —
  the other snapshots fall back to the cohort
  mean (treated as average difficulty), so on a 20-deep pool the default PFSP actively re-prioritises ≈¼ of the pool per
  cycle and an un-remeasured snapshot keeps its **last** EMA (a staleness bias toward selves you *used* to lose
  to — watch `eval/pfsp_hardest_win_rate` is tracking a moving target, not a fossil). (2) The `1 +` floor in the
  weight keeps coverage but makes the tilt mild: a self at `p=0.1` vs one at `p=0.5` differ only `(1+S·0.9)/(1+S·0.5)`
  (≈1.4× at `S=2`), and in a healthy gate-pinned pool the sentinel win-rates cluster near 50% so the realised
  prioritisation is modest — lean toward the high end of `S` (or beyond) if you want it to bite. PFSP touches
  **only which frozen opponent is sampled** — never the rollout, GAE, value target, promotion gate, or the
  `win_rate_vs_bots` curriculum ramp — so it cannot corrupt training; the worst case is "does little." A denser
  sentinel count under PFSP + a decay-toward-neutral for stale entries are the obvious follow-ups (deferred).
- **Full roster (v1 + v2 of every archetype).** Training (`OPPONENT_CLASSES`) and eval
  (`eval_opponent_names()` / `_EVAL_OPPONENT_SPECS`) both use all eight archetype bots —
  `{Heuristic, Heuristic2, Staller, StallerV2, Aggressive, AggressiveV2, SetupSweep,
  SetupSweepV2}` — because they play differently and the extra playstyle diversity is the
  point. There is no roster flag; the same nine names (eight bots + `random`) feed every
  path. `Random` is eval-only (a cheap "is the model broken" floor, excluded from
  `win_rate_vs_bots`); it is never a training opponent.
- **Resume state in `summary.json`.** `SelfPlayCallback` writes
  `<snapshot_dir>/summary.json` each eval (`win_rate_vs_bots`, `self_play_fraction`,
  `last_eval_step`, `seeded`, `pool_generation`) — `SnapshotPool.persist_summary`/`load_summary`.
  Read at `train_rl_agent` setup → the initial `self_play_fraction` (so a strong resumed model
  starts at the right ramp level, not the 0% cold-start) and the seed-gate decision. Distinct
  from the prober's `eval_traces/*/summary.json`; the legacy `win_rate_vs_bots.txt` is still read
  as a fallback.
- **Opponents sample, they don't argmax.** Training opponents are built with `stochastic=True`
  (now the `RLPlayer` default) so the learner trains against the policy's full action
  distribution — a richer, less-exploitable signal than the greedy move. Temperature is
  `--self-play-temp` (default `1.0` = the policy's own distribution; >1 flatter). **The measured
  trainee is always greedy** (`stochastic=False`) — that's what gives `win_rate_vs_bots`
  (curriculum) and `win_rate_vs_pool` (promotion) a stable, comparable control signal. The bots
  are deterministic rule-based players. The **pool sentinels default to stochastic@`--self-play-temp`**
  (mirroring how they act as training opponents) — so a sentinel matchup is greedy-trainee vs
  stochastic-sentinel, a deliberate asymmetry that inflates `win_rate_vs_pool` by a ~constant
  temperature handicap (≈15–20 pts; the [ELO caveat](#elo--skill-rating) below). **`--eval-sentinel-greedy`
  makes the sentinels greedy too** (`_play_unit` builds the sentinel opponent `stochastic=False`), so the
  matchup is best-vs-best and `win_rate_vs_pool` / the snapshot ELO reflect real skill (≈50% vs a
  recent self, ramping with sentinel age). It's eval-only — TRAINING opponents stay stochastic — and
  it auto-lowers `--promote-threshold` to `0.55` (else the handicap-free pool win rate never clears
  the 0.65 gate and the pool freezes). Default off so the live metric stays continuous until opted in.
- **Opponent snapshots are version-checked.** They load via `load_model_snapshot` (not a raw
  `MaskablePPO.load`), and `SnapshotPool` writes a shared `model_config.json` next to its
  snapshots, so an arch-mismatched snapshot fails with a clean `ModelVersionError` instead of
  loading mismatched weights.
- **The opponent RE-DECIDES on a stale decision; the trainee crashes** — split by who *owns* the
  decision. `SingleAgentWrapper` polls the opponent's `choose_move` on the *training* thread while
  POKE_LOOP mutates its battle, so by serialize time the captured snapshot (`ctx.legal`) can diverge
  from the live battle: POKE_LOOP parses an **in-flight turn-resolution during the model forward**,
  advancing `battle.turn` one ahead of `ctx.turn` (proven by the race trace — mutual Arena-Trap
  Dugtrios, the turn resolves mid-decision). `assert_decision_current` / `action_to_order` raise
  `StaleDecisionError`; handling then splits:
  - **Opponent** — its decision is *internal* to `step` (SB3 never sees it), so `RLPlayer.choose_move`
    catches the error and **re-decides on the now-current request**, bounded (`_OPP_REDECIDE_MAX`),
    with a valid default fallback only if the battle never settles. It must always return a valid
    order: SB3 has **no failed-step path** (a raise kills the `SubprocVecEnv` worker → parent hangs →
    worker-watchdog `os._exit`s → launcher restart). Each attempt's `embed_battle()` records its
    would-be decision into the rolling turn-history, so `choose_move` snapshots the tracker before
    the loop and `EpisodeTracker.restore()`s on a stale attempt — the superseded decision leaves
    **no phantom turn** in the opponent's turn-history obs (only the committed one survives; guarded
    by `redecide_rollback_fuzz_test.py` + `episode_tracker_test.py`). The re-decide guards only up to
    the order `choose_move` RETURNS; `SingleAgentWrapper.step` then re-serializes it via
    `self.env.order_to_action`, re-reading the battle **one more time** — a second, narrower window
    where it can finish/flip-to-wait under us (`ValueError ... not in valid orders ['/choose
    default']`). On that the wrapper falls back to the default order rather than crash (guarded by
    `single_agent_wrapper_test.py` + `order_to_action_race_fuzz_test.py`).
  - **Trainee** — its action is *SB3's*, computed outside `step` and not re-runnable mid-step, so a
    stale trainee decision **crashes** (`gen3_env`, no fallback): acting on it would corrupt its
    `(obs, action) → (reward, next_obs)` transition. Empirically it doesn't hit this — gated by the
    env's `race_get` request-wait (17 h vs-bots + self-play, zero trainee staleness).
  `_settle_opponent_battle` is a **pre-drain** that only trims how often the opponent re-decides — it
  can't drain *in-flight* messages, which is why re-decide (not settle) is the fix. The comprehensive
  `assert_decision_current` (every axis: moves+disabled, switches+species,
  force_switch/trapped/maybe_trapped/wait/struggle) is the detector; `train/selfplay_opp_redecide_rate`
  surfaces the resolved-race rate. **Full context — mechanism, the race trace, why it was hard, and the
  verification tiers — is in `race_fuzz_README.md`.** (`GEN3_FORCE_SELFPLAY` forces 100% self-play for
  the stress; `GEN3_RACE_TRACE=1` dumps the per-battle cross-thread interleaving into the
  `StaleDecisionError` **and** into the `race_get` silent-stall crash — see below. `StaleDecisionError`
  lives in `agents/action/mapper.py`.)
  - **Force-switch request-delivery deadlock (`_AsyncQueue.race_get`, `env.py`) — FIXED.** A
    *different* failure from the stale-decision race, and a latent bug **inherited verbatim from
    upstream poke-env 0.15.0**: `race_get` races a per-agent `queue.get()` against the
    `_waiting`/`_trying_again` coordination events, and can drop a request the server already
    delivered into the `battle_queue`. Two ways: **(1) stranding** — `asyncio.wait(FIRST_COMPLETED)`
    returns the instant any waiter completes, so an already-set **stale** event wins before the
    equally-ready `queue.get()` runs → `race_get` returns `None`, the agent is marked not-to-move,
    and its request sits unread; **(2) orphan theft** — `race_get` `cancel()`s the pending
    `queue.get()`, which a later `put` can resurrect to dequeue-and-discard the request.
    `_trying_again` goes stale because `env.step` cleared it only on the `None` path, and a
    re-request makes the battle non-`None`, skipping that clear. The trigger is the mutual
    Arena-Trap Dugtrio self-play mirror (trapped-switch `[Unavailable choice]` → stale
    `_trying_again`, then a faint → a `wait`+`forceSwitch` pair whose force-switch is stranded);
    rare (~1/8600 battles), so it only surfaced once self-play was on. **Fix:** `race_get` now
    `cancel()`s **and `await`s** the get to settle it (recovering its item, never orphaning it) and
    **prefers a queued battle over a stale event**, and `env.step` clears `_trying_again` the moment
    its agent receives a battle. Repro + regression guard: `forceswitch_deadlock_fuzz_e2e_test.py`
    (needs a `9XXX` server; `--widen` surfaces the timing race); unit coverage of both failure modes
    in `async_queue_disconnect_test.py`.
  - **Silent-stall watchdog (now a should-never-fire backstop).** Independently of the fix above,
    `race_get` bounds its wait by `_RACE_GET_TIMEOUT_S` (120 s, ~100× a normal step; override with
    `GEN3_RACE_GET_TIMEOUT_S`) and on a silent stall **raises `ShowdownException`** — a hard crash
    that propagates uncaught through the wrapper step chain to the SubprocVecEnv worker, so SB3
    discards the in-flight rollout (no fabricated transition reaches backprop) and the launcher
    restarts from the last checkpoint. It **crashes, never recovers in place** (recovering would feed
    PPO a stale `(obs, action) → (reward, next_obs)`). With `GEN3_RACE_TRACE=1` the wedged battle's
    interleaving is appended to the crash message via `race_trace.dump_recent()` (wedged battle
    ordered last so its newest events survive the launcher's last-100-line crash-file tail; the full
    trace is in `launcher_child.log`). `env.step` also emits `ENVSTEP` enter/race trace lines under
    `GEN3_RACE_TRACE` for debugging this handshake. Kept as defense-in-depth against any future
    request-delivery regression.
- **Self-play engages in the first process, not only after a restart.** The env is built before
  the model exists (the model needs the env's spaces), so on the first self-play process
  `_maybe_engage_self_play` seeds the pool from the loaded weights and rebuilds the env with
  pool opponents (then `set_env`). The worker watchdog is started *after* this, just before
  `learn()`. Later restarts find the pool already populated and skip the rebuild.
- **`--debug --self-play --debug-eval` exercises the real path** (seed → pool eval → promotion)
  on a fast eval cadence, so a CPU smoke against a `9XXX` server validates the wiring without
  disrupting the `:8001` training server (`--debug` skips all eval by default — `--debug-eval`
  opts in). `selfplay_opponent_fuzz_test.py` covers the opponent load + legal
  play (both modes) + version check in-process via the local bridge (no server).

## Stable (cross-run) opponents (`--stable-opponents`, `fixed_opponent_pool.py`)

Load a frozen model from **another, already-finished run** as a **fixed opponent** — measured
against in eval AND (under `--self-play`) played against in training. Design:
`designs/ai_v5/design_stable_opponents.md`.

**Training-mix participation (Stage 2) — "tossed in like a sentinel, becomes a bot when mastered":**
a stable opponent rides the *existing* pool-vs-heuristic split in `MaskableAgentWrapper`
(`wrappers.py`), no new source-model abstraction:
- **CHALLENGE bucket** (the self-play pool branch, competence-gated by `self_play_fraction`): the
  pool gets the BULK; un-mastered stable opponents share a **capped minority slice**
  (`STABLE_CHALLENGE_SHARE` = 0.20 in `wrappers.py`), so a single fixed opponent can never dominate
  training (multiple un-mastered ones SHARE the 20%, so the total stays bounded). It only enters the
  mix once the model clears `SELF_PLAY_START` (a weak model trains on bots first), and only under
  `--self-play` (without it, stable opponents are eval-only — a startup NOTE says so).
- **FLOOR bucket** (the heuristic-bot branch): once the trainee **masters** it
  (`win_rate_vs_ext_<run>` ≥ `--stable-opponent-mastered-wr`, default `0.80`, for
  `_MASTERY_CONFIRM_CYCLES`=2 consecutive cycles — a noise guard since the irreversible flip is
  one-way), it "becomes another bot" — moved to the always-on coverage floor (weighted like an
  unlisted bot). The eval callback tracks a **monotonic** mastered set + a per-label streak counter,
  recomputed each cycle (→ resume-safe), and pushes it via `env_method("set_stable_mastered", …)`,
  exactly like `set_self_play_target`. The recompute+push runs **early** in `_collect_pending` (with
  the training-mix telemetry below), so this cycle's challenge↔floor flips show up in both the pushed
  env state and the reported fractions. **Resume note:** the mastered set lives only in callback
  memory, so after a launcher restart a previously-mastered opponent reverts to the challenge bucket
  until the first post-restart eval re-confirms it (self-healing; bounded by the eval cadence).
- **Training-mix share is reported, not just eval win rate.** The stable opponents' actual slice of
  the training mix shows up in `train/stable_fraction` (challenge un-mastered + floor mastered), with
  `train/selfplay_fraction` (pool) and `train/nonbot_fraction` (their sum); see the Curriculum
  subsection's **Opponent-mix reporting** bullet above for the exact decomposition.
- **Dynamic within-slice selection (`--stable-opponent-pfsp`, default off).** A FLAT capped share
  splits the stable slice UNIFORMLY over the un-mastered opponents — so a generalist hardening against
  several exploiters at once spends equal budget on the axis it already handles and the one it's
  failing. Under `--stable-opponent-pfsp`, `MaskableAgentWrapper._pick_stable` weights the
  un-mastered-stable pick by **`1 − win_rate`** (floored 0.05) — the exploiter it's LOSING to worst
  gets most of the slice, and each fades as mastered (win_rate→1 ⇒ weight→0), then the mastery flip
  retires it to the floor. Win-rates are the same `win_rate_vs_ext_<label>` eval already computes,
  EMA-smoothed (`_PFSP_WR_EMA_BETA`) and pushed each cycle via `SelfPlayCallback._push_stable_mastered`
  → `env_method("set_stable_win_rates", …)` (mirrors the pool PFSP `set_opponent_win_rates`). **The
  TOTAL pool-vs-stable share is unchanged** (still `--stable-opponent-selfplay-share`), so the
  opponent-mix telemetry + the `test_mix_fractions_match_actual_sampling` anti-drift guard are
  unaffected — only WHICH un-mastered stable opponent is picked shifts. Training-only (not
  version-locked, forwarded on resume like `--pfsp-scale`); OFF = uniform, byte-identical. **Pairs
  with a raised `--stable-opponent-selfplay-share`.** Motivation: a flat 0.35 share (≈12% exposure
  each of 3 exploiters) left ai_v7_14's hardening flattening at ~0.30 vs the exploiters; the dynamic
  focus + a raised share is the fix. Tests: `wrappers_test.py::test_stable_pfsp_*`.
- The stable-opponent players are **built once per worker** (`load_foreign_opponent` in the env
  factory), so no per-episode reload; each plays **stochastic** at `--stable-opponent-temp` in
  TRAINING but **greedy (temp 0)** in EVAL (a clean yardstick).
- **Surfaced in the launcher Events panel** (via `emit`, like the `[SELFPLAY]` startup lines): a
  `🐴 [STABLE] N cross-run opponent(s): ext_<run> — eval greedy; training ≤<share> of self-play until
  mastered (win_rate ≥ <wr>)` line at startup (and a `🏇 [SELFPLAY] Mastered stable opponent(s) …`
  line on the challenge→floor flip), and each eval-summary event gains a `stable <pct>%` field. (Per-opponent `eval/win_rate_vs_ext_<run>` also rides the normal eval Metrics table.)

- **CLI:** simplest form is just the run dir — `--stable-opponents models/ai_v5_5_popart_N_0607`;
  the opponent is **labelled by the run-dir name** (`ext_ai_v5_5_popart_N_0607`, derived
  `best_model`/`snapshots`-aware so a direct `…/best_model/best_model.zip` path still yields the run
  name, not `best_model`). Optional per-entry suffixes: `@<step>` (a specific checkpoint; default
  `best_model`), `:<name>` (rename). **Per-opponent weights (`=<weight>`) are rejected** with a clear
  message (not supported). Knobs: `--stable-opponent-temp` (default 1.0 — the *training* play
  temperature; eval is always greedy) and `--stable-opponent-mastered-wr` (default 0.80 — the
  challenge→floor flip). Parsed + resolved at startup by `fixed_opponent_pool.resolve_stable_opponents`.
- **Compatibility = the OBSERVATION FAMILY only** (two axes: obs family vs model family — see the
  design §3). The gate is **same `arch_signature`** (`ModelVersion.check_opponent_compatible`,
  the obs-family proxy); a mismatch is a **startup FATAL** (`[StableOpponent] FATAL` →
  `TrainExitCode.FATAL_CONFIG`, surfaced to the TUI, no restart). Loaded inference-only via
  `snapshot.load_foreign_opponent` (`env=None`), which **skips `check_compatible`** — so
  `use_popart`/`vf_coef`/reward differences (irrelevant to an opponent's forward, which never reads
  the value head) don't block it. The example `models/ai_v5_5_popart_N_0607` shares HEAD's arch, so
  it loads despite being PopArt-on.
- **Label namespace `ext_<run>`** — underscore separator (NOT `ext:`) so the emitted metric tags are
  **uniform** with the rest (`eval/win_rate_vs_ext_<run>`, like `eval/win_rate_vs_sentinel_0`), no
  colons in TensorBoard. `is_external` (`startswith("ext_")`) keeps them out of the bot aggregates.
  Both eval callbacks (`PerOpponentEvalCallback` + `SelfPlayCallback`) add the `ext_` labels as
  `FIXED` `EvalItem`s (so they shard + ride the same plan); the worker's `_play_unit` FIXED branch
  (`eval_worker.py`) plays the **greedy trainee vs the greedy stable opponent** (a clean yardstick).
- **Metric set (deliberate, uniform across both callbacks):** per opponent —
  `eval/win_rate_vs_ext_<run>`, `eval/mean_reward_vs_ext_<run>`, `eval/mean_ep_len_vs_ext_<run>`;
  plus `eval/win_rate_vs_external` ONLY for a mini-league (2+ — with one it duplicates its row; it's
  an `_EVAL_SUMMARY` "vs External" row, not a fake per-opponent row); plus a `metadata.json:latest_eval`
  `externals` block. Kept **OUT of** `win_rate_vs_bots` (`bot_mean` excludes them), `win_rate_vs_pool`,
  the best-model aggregate, the `td_resid_tail_mean` headline, and **the ELO FIT itself** (no ladder
  distortion). **NOT emitted for ext:** `td_resid_tail` (a bot/sentinel critic-coverage diagnostic).
  The TUI renders each by its run name with an `(ext)` tag.
- **ELO shown in the eval table** (`record_external_elos`): the elo column for an `ext_` row PREFERS
  the opponent's **own recorded ELO** — read at startup from its `best_model.json` sidecar (or run
  `metadata.json`) `latest_eval.elo` into `FixedOpponentEntry.source_elo` (`_read_source_elo`). It's a
  well-fit, bot-anchored rating (cross-run-comparable since the bot anchors are stable) — e.g. 1902 for
  `ai_v5_5_popart_50m_0607`. **Fallback** (`external_elo`) when the opponent carries no recorded ELO:
  invert the BT win prob from the trainee's live rating + win rate (`R_opp = R_trainee −
  (400/ln10)·logit(wr)`, clamped ≈±676) — a rough single-edge estimate. Recorded as
  `eval/elo_vs_ext_<run>`; the opponent is NEVER a player in the fit itself (no ladder distortion).
- **`best_model/` is self-contained.** Saving the best model copies the run's `model_config.json` AND
  writes a `best_model.json` sidecar (`copy_run_config_to_best_model` + `write_best_model_sidecar`,
  both called from both eval callbacks' best-save). `best_model.json` reuses
  `snapshot.write_checkpoint_metadata` (the per-checkpoint sidecar code) so it carries the
  `latest_eval` block **incl. the run's ELO** —
  `best_model/{best_model.zip,model_config.json,best_model.json}` co-located (arch gate + carried ELO,
  no parent search). Backfilled for existing `models/*/best_model/` dirs.
- **Per-opponent pinned teams (the league FOLD-BACK contract).** A SPECIALIST stable opponent —
  one whose run pinned `--trainee-team` — pilots **ITS OWN team** here, not the shared pool
  (otherwise a trapper exploiter folds back piloting random teams and the pressure it was trained
  to apply evaporates — the realized-matchup lesson applied to the opponent side).
  `resolve_stable_opponents` reads the pin from the opponent run's `metadata.json:
  cli_args.trainee_team` (`_read_trainee_pin`) into `FixedOpponentEntry.team_str` — **fail-loud**:
  a recorded pin whose file is missing raises, and a pin that no longer matches the run's recorded
  MatchupSpec `pin_sha` raises (never a silent pool fallback). TRAINING: the env factory builds a
  per-entry pinned builder and `MaskableAgentWrapper._apply_opponent_team` switches
  `env.agent2._team` **per episode** to match the selected opponent (agent2 does the opponent-side
  networking, so its `_team` decides the opponent's real team — the mirror lesson); unpinned
  episodes restore the pool builder (the SAME instance, so team-draw RNG streams are unchanged);
  with no pinned opponent anywhere the wrapper never touches `agent2._team` (byte-identical). EVAL:
  `team_str` rides `to_cfg()` → the `EvalItem` → `eval_worker._fixed_opponent_tb`, so the FIXED
  branch measures the opponent piloting its pin (eval matches training, same rule as the trainee's
  own pin). The `[STABLE]`/`[EXPLOITER]` startup lines annotate `[pilots ITS OWN pin: <file>]`.
  Guard: `poke_env_gaps/opponent_pin_fuzz_test.py` (bridge, real battles — pinned episodes field
  EXACTLY the pin, bot episodes the pool).
- **Tests:** `fixed_opponent_pool_test.py` (parse + resolve + the arch FATAL gate + the pin
  resolve/fail-loud/sha cases + `register_exploiter_for_eval` dedup),
  `snapshot_test.py::*opponent*/*foreign*` (the loader + `check_opponent_compatible`), and the
  end-to-end `stable_opponent_fuzz_test.py` (bridge, no server — resolve + arch FATAL + foreign
  load + legal stochastic play) + `opponent_pin_fuzz_test.py` (the fold-back realized-team guard).

## Exploiter mode (`--exploiter`, `MaskableAgentWrapper._exploiter_player`)

A clean opponent-mix front-end for the league **exploiter** role: train a dedicated agent against
ONE fixed foreign model as the **sole opponent every episode** — to surface (and then patch, by
folding the exploiter back as a stable opponent / pool member) the non-robustness a *self-play* Nash
can't see. It needs **no `--self-play` / `--stable-opponents` / share fiddling** — point `--exploiter`
at the target and it's the only opponent.

- **Target resolution** reuses the stable-opponent path exactly: `--exploiter <run-dir|checkpoint
  spec>` → `resolve_stable_opponents` (a single `FixedOpponentEntry`, arch_signature-gated) +
  a weights-load validation in the main process (corrupt zip = startup `[Exploiter] FATAL` →
  `FATAL_CONFIG`, no restart loop). Emits a `🥊 [EXPLOITER]` line to the launcher Events panel.
- **Opponent mix**: the env factory builds ONE `RLPlayer` over the target per worker (stochastic at
  `--stable-opponent-temp`, a moving target), and `MaskableAgentWrapper._select_episode_opponent`
  **short-circuits** the whole challenge/floor/pool/stable selection when `exploiter_player` is set —
  the target is `self.opponent` every reset. `None` (default) = the normal selection, byte-identical.
- **Team-source guarantee — an exploiter may ONLY EVER pilot a vetted SAMPLE team.** The curated
  `data/teams/sample/` set is the tournament-proven, rock-solid roster; the ~687 `other` teams are
  bulk-downloaded and unvetted. `matchup_spec.validate_exploiter_trainee_is_sample(matchup,
  sample_teams)` (called at startup in `train_rl_agent`, FATAL → `FATAL_CONFIG`) enforces that a
  `mix_kind == 'exploiter'` run with a pinned `--trainee-team` pins a team whose strip-normalized
  fingerprint is in the sample set — else it refuses to launch with a clear message. Out of scope:
  non-exploiter runs (any pin allowed), and an exploiter with an UNPINNED trainee (a full-pool
  exploiter, not a single-team specialist). The shipped TSS pin IS a sample team, so it passes; a
  future multi-team exploiter pool must validate every member. Tests:
  `matchup_spec_test.py::test_exploiter_*sample*` + the e2e FATAL.
- **Mutually exclusive with `--self-play`** (arg-parse error — the exploiter needs no pool). Because
  it's not self-play, `_opp_version` (the arch gate for the foreign load) is set explicitly for this
  path before the factories are built. Training-only; not version-locked.
- **Temperature-annealing curriculum (`gen3_exploiter_temp_anneal_v1`, `--exploiter-temp-start`).** A
  from-scratch trainee vs a STRONG frozen target is crushed every game — the PPO advantage is ~0 (all
  losses look equally bad) and it never gets a foothold. This anneals the target's SAMPLING TEMPERATURE
  over training — a difficulty curriculum via opponent STOCHASTICITY (not by swapping opponents): start
  the target HOT (`--exploiter-temp-start`, e.g. 2.0 → flatter logits → noisier/weaker play, so the
  trainee wins some games and gets a learning signal) and linearly anneal it to `--exploiter-temp-end`
  (default 1.0 = the target's true play distribution) over `--exploiter-temp-anneal-frac` of `--steps`
  (default 0.2), held after. `ExploiterTempAnnealCallback` (`exploiter_temp_callback.py`) computes the
  temp from SB3's `_current_progress_remaining` each rollout (both the sync and async collectors call
  `on_rollout_start`) and pushes it to every env's exploiter `RLPlayer` via
  `env_method("set_exploiter_temperature", T)` — the `set_self_play_target` idiom;
  `MaskableAgentWrapper.set_exploiter_temperature` sets `RLPlayer._temperature` (read fresh each
  `choose_move`). Metric: `train/exploiter_temp` (TB + TUI). **Training-only** — no weight-shape/forward
  change, NOT version-locked, forwarded verbatim on resume (where `_current_progress_remaining` reflects
  the resumed step, so the anneal continues from the right point). Registered ONLY when
  `--exploiter-temp-start` is set → an off run makes no push (byte-identical, opponent plays at the fixed
  `--stable-opponent-temp`). Composes with `--exploiter-keep-bots` (the from-scratch specialist recipe:
  a bot floor + a temp-ramped strong target). Tests: `exploiter_temp_callback_test.py` (schedule +
  push/change-guard), `wrappers_test.py::test_set_exploiter_temperature_*`.
  - **Two modes (`--exploiter-temp-mode {fixed,ratchet}`, default `fixed`).** `fixed` = the linear
    time schedule above. **`ratchet` = DYNAMIC, win-rate-driven, one-way** (`ExploiterTempRatchetCallback`):
    the fixed schedule has to GUESS the right starting temperature (empirically ai_v7_06's fixed 2.0 start
    was too weak — a 1983-ELO target flattened by temp ~2 is still a wall for a from-scratch net, so half
    the games yielded ~no advantage signal). Instead, start the target near-trivial
    (`--exploiter-temp-start` HIGH, e.g. 5.0) and ratchet the temp DOWN (`*= --exploiter-temp-ratchet-factor`,
    default 0.9, floored at `--exploiter-temp-end`) only when the trainee's measured **training** WR vs the
    target clears `--exploiter-temp-ratchet-wr` (default 0.55, near the ~0.5 max-advantage-signal zone) over
    a window of `--exploiter-temp-ratchet-games` (default 500) target-games. It **never raises** the temp,
    so a plateauing trainee can't comfort-trap the controller into weakening the opponent (the failure mode
    of a symmetric setpoint controller) — mirroring the one-way stable-opponent mastery flip. The signal is
    the TRAINING WR at the current temp (NOT the greedy eval WR, which reads ~0 forever early): the wrapper
    counts per-episode outcomes vs the target (`_record_exploiter_outcome` / `exploiter_winrate_totals`,
    bot episodes excluded), and the callback diffs the cumulative totals via `env_method` each
    `on_rollout_end`. **Resume-safe:** the ratcheted temp is persisted to `<run>/exploiter_temp_state.json`
    and restored on a launcher restart (else a fresh child resets to the easy `temp_start` and undoes the
    ratcheting; the WR window restarts fresh). Metrics: `train/exploiter_temp` + `train/exploiter_target_wr`
    (hovers near the threshold) + `train/exploiter_temp_ratchets`. Requires `--exploiter-temp-start >
    --exploiter-temp-end`. Tests: `exploiter_temp_callback_test.py` (`_decide` one-way/floor + windowed
    control loop + resume round-trip), `wrappers_test.py::test_exploiter_winrate_totals_*`.
- **The target AUTO-registers for eval** (opponent-parity Proposal A,
  `fixed_opponent_pool.register_exploiter_for_eval`): `--exploiter` alone now produces the verdict
  metric `eval/win_rate_vs_ext_<target>` — the resolved target entry is appended to the eval-side
  fixed-opponent list, DEDUP-guarded (same resolved zip or colliding label → unchanged), so the
  historical `--exploiter X --stable-opponents X` recipe is byte-identical. Eval-only by
  construction (exploiter mode excludes `--self-play`, so the appended entry never joins the
  training mix). And per the fold-back contract above, a SPECIALIST target (its run pinned
  `--trainee-team`) is faced — and eval-measured — piloting **its own pinned team**
  (`exploiter_team` in the wrapper; the startup `[EXPLOITER]` line annotates the pin).
- **Usage:** `--exploiter <target> --model <target's checkpoint>` — init the exploiter from a strong
  checkpoint so it has a baseline to exploit from (the AlphaStar exploiter init). The verdict
  metric vs the target is automatic (above); an explicit `--stable-opponents <same target>` is
  harmless (dedup). The run dir defaults to a readable `models/exploiter_vs_<target>/` (not a
  date-stamp); override with `--run-name <name>`. Tests: `wrappers_test.py::test_exploiter_*`
  (sole-opponent + off-unchanged) + `test_pinned_*` (per-opponent teams),
  `fixed_opponent_pool_test.py::test_exploiter_*registration*`.

### Consensus warm-start (`--warmstart-consensus`, `warmstart.py`) — EXPLOITER-ONLY

`gen3_exploiter_consensus_warmstart_v1` — a low-bias INIT for a NEW exploiter: BEFORE training, build a
competent, archetype-NEUTRAL warm start by **disagreement-gated CONSENSUS distillation** of N mature
teacher exploiters into `--model` (the generalist init), then init the exploiter from it. The
`build_consensus_target` math (pure, in `warmstart.py`): `consensus` = mean of the N teachers' masked
action distributions; `d` = mean pairwise **Jensen-Shannon disagreement**; a quantile-normalized gate
`g∈[0,1]` sets a per-state temperature `T = 1 + (tmax−1)·g`; `target = softmax(log consensus / T)` over
legal actions — **SHARP where the teachers AGREE** (universal decisions the new exploiter just inherits)
and **FLAT where they DISAGREE** (archetype forks left high-entropy → the new exploiter specializes
FREELY, unbiased). BC also carries a KL anchor toward the student's OWN distribution (`anchor_coef`) so
the warm start RETAINS the generalist's competence. Distills in **function space** (teacher outputs) —
weight-averaging FAILED (from-scratch exploiters live in different loss basins → the average collapsed;
`tmp/average_exploiters.py`).

**Why EXPLOITER-ONLY (guarded, `parser.error` without `--exploiter`):** this SEEDS a new model with the
consensus + freedom to diverge. It must NOT touch generalist training, whose objective is the OPPOSITE —
absorb the DIVERGENT per-team specializations (that is `--distill-teacher`, one teacher per team-masked
state). Distilling the consensus into the generalist would sharpen agreement and blur divergence,
erasing the specialization it is trying to learn (and the generalist already ≈ the consensus → circular).
`--self-play` is excluded automatically (exploiter mode already forbids it).

**Integration.** `train_rl_agent` builds it ONCE into `<run>/warmstart/warmstart_consensus.zip` (via
`run_consensus_warmstart`, live over the local bridge) right before the model load, then re-points
`--model` at it. **Idempotent under launcher restarts:** skipped entirely once ANY training checkpoint
exists (the normal resume path continues the trained state); the warm start is arch-identical to `--model`
(its `model_config.json` is copied), so the resume-immutable checks stay valid. Standalone:
`python -m agents.training.warmstart --student <run> --teachers <run,...> --out <dir>`. OFF (flag unset) =
byte-identical. Knobs: `--warmstart-battles` (200), `--warmstart-bc-steps` (4000). Tests:
`warmstart_test.py` (the pure consensus/JS/gate/temperature math: identical→0 disagreement, sums-to-1 over
legal, mask respected, sharpens-agreement/flattens-disagreement, `tmax=1` recovers plain consensus).

## Team-side PFSP (`--team-pfsp`, `team_pfsp_callback.py`)

The TEAM-axis complement to the opponent-side `--pfsp-scale`: bias the TRAINEE's team sampling toward
the pool teams it is weakest on, so training spends gradient where the win-rate says there's headroom
instead of uniformly over ~700 pool teams (the documented "uniform team sampling = headroom" gap).
Three modes: **`off`** (default → byte-identical), **`measure`** (TRACK + persist the per-team
self-play win-rate WITHOUT biasing sampling — pure observability), **`var`** (measure + bias).

- **Variance weighting + cap + floor.** For pool team `i` the weight is `raw_i = --team-pfsp-floor +
  p_i·(1−p_i)` where `p_i` is the team's self-play win-rate EMA (seed 0.5 → an unmeasured team gets the
  MAX variance weight → explored), then capped `w_i = min(raw_i, --team-pfsp-cap·mean(raw))` (no team is
  sampled more than `cap`× the uniform share — the over-representation bound). `p·(1−p)` peaks at 50%
  (the most-to-learn matchups) and decays to the floor at both extremes, so it self-ignores both the
  teams we crush AND the truly-lost teams; the floor keeps nothing fully starved. `compute_team_pfsp_
  weights` is the pure, unit-tested math.
- **Self-play only, pool teams only.** The per-team win-rate is measured ONLY on self-play POOL battles
  (bots wash the signal out — we win ~0.99 vs bots): `MaskableAgentWrapper.step` records the outcome to
  the trainee's `Gen3Teambuilder` (`self.env.agent1._team`) only when `self.opponent is
  self._pool_player`. A bias/distill-pinned team (the `--distill-team-bias` branch) yields
  `_last_pool_idx=None` → its battle is never tracked (those teams get fixed exposure via the bias, not
  the win-rate weighting).
- **Centralized aggregation (NOT per-worker — ~700 teams makes a single worker's counts too sparse; NOT
  info-dict threading — that breaks under `--async-rollout`).** Each worker's teambuilder accumulates
  LOCAL windowed `(wins, games)` per pool team; `TeamPFSPCallback` every `update_every` (3) rollouts
  PULLs them from all workers via `env_method("drain_team_pfsp_counts")` (drain-zeroes each window), SUMs
  by pool index, EMA-smooths a global per-team win-rate, computes the capped weights, and PUSHes them
  back via `env_method("set_team_pfsp_weights", w)` → the teambuilder samples with
  `random.choices(weights=…)`.
- **Auditability + GIGO guard.** Each pool team carries a `team_sha` fingerprint
  (`sha1(team_str.strip())[:10]` — the SAME convention as `matchup_spec.pin_sha` / the archetype
  artifact, so a key JOINS every provenance record). The callback pulls them ONCE
  (`env_method("get_team_pfsp_keys")`) and verifies the per-INDEX team identity is IDENTICAL across
  every worker (**same pool SIZE ≠ same pool ORDER** — a diverged order would silently mis-attribute
  win-rates, which the cheap per-cycle size-only belt can't catch), then logs the weakest measured
  teams by `sha@win-rate` so the weighting is inspectable (which teams/archetypes the budget
  concentrates on), not an anonymous min/max scalar. Metrics
  `team_pfsp/{min_wr,max_wr,n_measured,weight_spread}`.
- **Persisted artifacts (both `measure` and `var`) → offline "which exploiter next".** Each update the
  callback writes to the run dir: `team_winrates.json` (the latest snapshot — per-team `{sha, win_rate,
  games, archetype}` sorted WEAKEST-FIRST, atomic-replaced; the weakest teams = candidate exploiter
  targets, and `archetype` is joined from `gen3_team_archetypes.json` via `team_sha` so it reads
  "weakest = stall-class") and an appended `team_winrates_history.jsonl` row `{step, wr:{sha:wr}}` (so
  the per-team win-rate is trackable OVER TIME offline — trends + noise, not just the latest). `measure`
  gives this signal on ANY self-play run without changing the team distribution.
- **Training-only, not version-locked.** Threaded into the TRAINEE teambuilder only (both the
  `matchup.trainee_teams.build` and the distill `Gen3Teambuilder` paths); the opponent builder is
  untouched. Registered ONLY when `--team-pfsp != off` (off → no callback, no `env_method`, exact-legacy
  `random.choice` → byte-identical); `var` pushes weights, `measure` never does. Forward it like
  `--pfsp-scale` on resume; no `model_config`/`ModelVersion` entry.
- **Tests.** `utils/teambuilder_test.py` (off==uniform RNG-identical, weighted sampling, record/drain,
  the cap+floor weight math), `team_pfsp_callback_test.py` (cross-worker aggregation, the pool-size GIGO
  guard, the `update_every` throttle, None-worker filtering).

## ELO / skill rating (`elo.py`, `bot_elo_calibration.py`, `main.elo`)

Once training is mostly self-play **pool play**, win-rate stops being legible: the promotion
gate only promotes when `win_rate_vs_pool > promote_threshold` and the pool is a *sliding window
of recent selves*, so `win_rate_vs_pool` is a treadmill pinned near 50-65% **by construction** —
it cannot trend up however much the model improves; `win_rate_vs_bots` saturates near 100%. The
ELO subsystem gives a single **absolute** number that genuinely rises with skill, anchored to the
fixed bots.

- **No new battles.** Every eval cycle already plays the trainee (greedy) vs all 9 bots and vs
  up to `--n-sentinels` (default 5) pool sentinels, `EVAL_GAMES` each — a full tournament-matrix
  row. `record_elo`
  (`eval_callback.py`, shared by BOTH callbacks) appends that row to an **append-only
  `<run>/eval_results.jsonl`** (`snapshot.append_eval_result_row`) — the canonical, restart-safe
  source of truth, distinct from the overwritten `metadata.json:latest_eval`.
- **The model = anchored Bradley-Terry** (`elo.fit_elo`): `P(i beats j)=σ((Rᵢ−Rⱼ)·ln10/400)`,
  fit in **batch** by penalized MLE (weak Gaussian prior keeps 100-0 records finite), SE from the
  inverse Hessian. Each bot is a player `bot:<name>`, each snapshot `snap:<step>` — a snapshot is
  the SAME player whether it appears as a cycle's trainee or later as a sentinel (unified by
  step), which links the whole ladder. Batch-BT (not online K-factor Elo) is drift-free and
  re-runnable; the fit is a few Newton steps over ~tens of players. **Not Glicko-2**: its
  volatility models skill drift, but snapshots are *frozen* — the drift is the *sequence* of
  snapshots (the ELO-vs-step curve); the per-player uncertainty (Glicko's valuable part) is the
  Hessian SE.
- **Anchor = a precomputed bot-vs-bot round-robin.** `python -m agents.training.bot_elo_calibration`
  plays all 36 bot pairs toward `--target-games` (default 5000) **in-process via the bridge — no
  server** (safe alongside a live run; it does use CPU — throttle with `--concurrency`), fits BT
  (`elo.fit_pairwise`, `random` pinned at `base`=1000), and writes the anchor. **Artifact split:**
  the immutable bot anchor (ratings, SEs, the 9×9 win-matrix, a non-transitivity `fit_quality`) is
  the only runtime input, so it lives in **`data/gen3_bot_elo_anchors.json`**; the raw game-count
  **store** (resume state) and the **heatmap** PNG are calibration provenance/viz, so they live with
  the ELO design work under **`designs/ai_v5/elo_calibration/`** (override with `--games-store` /
  `--heatmap`). The
  live/offline fits then **pin all 9 bots** to those high-confidence ratings and fit only
  snapshots — so a snapshot is well-grounded from its first cycle, and because the anchor is
  identical across runs, **snapshot ELOs are comparable run-to-run**. **Regenerate when bot logic
  changes** (the json records `git_hash` + date). Graceful fallback when the file is absent:
  `random` pinned at `base`, other bots float (rank/trend preserved, scale not cross-run-stable).
  Bots build once and are reused across pairs (`reset_battles` between) — building warms the data
  singletons (~4.5 s each), so per-pair rebuilds dominated cost; the full 5000-game job is a
  many-hour, run-overnight one-time cost.
- **Live (each eval cycle).** `record_elo` refits and records `eval/elo` + `eval/elo_ci` (95% CI
  half-width) to TensorBoard + the TUI dict, and stamps `elo`/`elo_ci` into `metadata.json:
  latest_eval` (so the resume-republish path shows ELO immediately after a restart — the saved
  headline is authoritative; and if a resumed checkpoint predates the `elo` field,
  `replay_last_eval_to_tui` **fits** the saved block's win rates via `elo.fit_from_block` to recover
  both the headline and each opponent's ELO, so the badge never blanks for a full cadence). The
  launcher
  surfaces a `🏅 ELO 1532 ±40` badge (`app.py::_elo_badge`) + an `elo` column in the eval panel:
  the model's rating on the `all` row, and each opponent's anchored ELO on its row
  (`_record_opponent_elos` records `eval/elo_vs_<bot>` + positional `eval/elo_vs_sentinel_<i>` to
  the TUI). The live number is the best estimate from data SO FAR (batch-BT is global → early
  points retro-adjust; the single-cycle per-sentinel ELO is rough — only the trainee is
  bot-anchored each cycle); the offline CLI re-fits canonically over the full per-snapshot history.
- **Offline (`python -m main.elo <run_dir>`).** Loads results (`--source auto|log|tb|meta` —
  `tb` **backfills an already-running run straight from TensorBoard, zero training change**), fits,
  and prints a ranked ladder + writes `elo_ratings.json` + an Elo-vs-step `elo_curve.png` (CI band
  + bot anchor lines). `--out` defaults to `<run>/elo/`; point elsewhere to analyze a LIVE run
  without writing into it.
- **Caveat (acceptable, noted in code):** by default the trainee is greedy but the sentinels are
  stochastic@temp, so a snapshot's rating blends greedy strength (when it's the cycle's trainee)
  with stochastic strength (when it's a later sentinel) — a roughly uniform shift that preserves the
  trend, but it does mean the same snapshot is scored in two regimes. **`--eval-sentinel-greedy`
  removes this** — sentinels play greedy too, so every snapshot is scored greedy in both roles and
  the ELO ladder is internally consistent (at the cost of a one-time scale shift vs prior cycles;
  the bot-anchored scale is preserved since trainee-vs-bot records are unchanged). Tests:
  `elo_test.py` (synthetic-ladder recovery, anchoring, perfect-score, loaders, `fit_pairwise`).

## Rollout collection: sync barrier vs `--async-rollout` (`async_vec_env.py`)

The default `SubprocVecEnv.step()` is a **per-step barrier** — the trainer waits for the slowest of
N env workers every step, so a slow battle turn / heavy opponent forward / oversubscription jitter
stalls the whole batch and the GPU policy-forward never overlaps CPU env-stepping. `--async-rollout`
swaps in **`AsyncSubprocVecEnv`** (per-env `send_step`/`poll_ready`/`recv_step` over the pipes +
**drain-safe `env_method`** — the eval callback's `set_self_play_target`/
`opponent_default_stats` fire mid-collection, so the override stashes in-flight step results before
any barrier RPC to avoid a pipe desync) and **`collect_rollouts_async`**, dispatched by
`InstrumentedMaskablePPO.collect_rollouts` when `model._async_rollout` is set.

The collector keeps every worker continuously in-flight, batch-forwards whichever envs are READY
(dynamic batch), and writes each env's transition into **its own buffer column**
(`MaskableDictRolloutBuffer`); collection ends when every column has `n_steps`. It is **exactly
on-policy** — PPO freezes the policy during collection, so this is a *scheduling* change (overlap
forward with stepping, drop the max-latency barrier), NOT an APPO-style algorithm change. Bookkeeping
(`num_timesteps`, GH-#633 timeout bootstrap, `_update_info_buffer`, `_last_*` carry-over, per-column
GAE) mirrors the stock loop exactly. The per-decision **mask rides in the Dict obs**
(`obs["action_mask"]`, = `last_ctx.mask`), so no per-env `env_method` and no wrapper change.

**Measured FPS (bridge, GPU forward, steady-state, heuristic opponents):** +20% at `--n-envs 16`;
**+14% at the production `--n-envs 64` (1489→1695)**; `--async-rollout --n-envs 32` matches `sync@64`
FPS with half the envs (≈half the env/bridge RAM). Off by default (stock `SubprocVecEnv`), ignored
under `--debug`. Caveat: benchmarked with heuristic opponents — re-bench under `--self-play` for the
production-regime number. Full design + benchmark table: `designs/ai_v5/design_async_rollout.md`.

## Gradient-balance + value-scale diagnostics (`grad_balance.py`)

The dual-head extractor shares ONE transformer trunk between the policy and value heads
(`src/agents/model/CLAUDE.md`); both losses' gradients compete there. When the value loss
dominates (large / unclipped, big-return scale) it **swamps the trunk** and the policy barely
updates — visible before only *indirectly* as suppressed `train/approx_kl` + `train/clip_fraction`
while `train/explained_variance` races ahead. `InstrumentedMaskablePPO.train()` now measures it
**directly** via the pure helpers in `grad_balance.py` (no SB3 / logging coupling → unit-tested in
`grad_balance_test.py`), recorded once per `train()` call through the standard logger → TensorBoard
**and** the launcher TUI (the new scalars ride the generic `MetricsExporterCallback` →
`ipc.send_metrics` path with zero extra wiring; ordering/labels live in `launcher/format.py`).

- **Gradient balance — every head's *pull* on the shared trunk, on ONE common denominator.** Sampled
  on the first minibatch (graph alive) by **read-only** `autograd.grad` probes (`retain_graph=True`, so
  the real `loss.backward()` is unaffected) against the shared-trunk params. "Shared" =
  `SHARED_TRUNK_PHASES = {embeddings, pokemon_encoder, team_transformer, assembler}` (the allow-list
  is the single source of truth), which **excludes** `cls_pool` (head-private `our_cls`/`their_cls`/
  `value_cls` queries) and both projection heads — only *truly contested* params count. With the
  belief / move / latent / move-latent / win-prob / value-dist auxiliaries there are now **many**
  competitors, not just value-vs-policy, so **every `grad/*_share` is on the SAME total**
  `T = ‖g_pi‖ + ‖g_vf‖ + Σ‖g_aux‖` — the shares are mutually comparable, **sum to ~1**, and any one
  term crowding out the rest is read off directly. (L1-of-norms — an upper-bound proxy, not a variance
  decomposition, since `‖a+b‖ ≠ ‖a‖+‖b‖` — but the same convention for every term.)
  - `grad/policy_share` + `grad/value_share` — the two RL heads' slices of the **whole** pie (ALWAYS
    present). Each is weighted by the live `ent_coef` / `vf_coef`, so `value_share` is a `vf_coef`
    tuning read — but it now *moves with the aux count* (it is value's slice of the full pie), so prefer
    the aux-independent `value_policy_logratio` below for the pure value/policy balance.
  - `grad/aux_share` (only when ≥1 aux is on) — Σ of all the aux shares, the **total non-RL draw** on
    the trunk: one curve for "are the scaffolds collectively crowding out policy/value".
  - `grad/value_policy_logratio` = `log10(‖g_value‖/‖g_policy‖)` — the **AUX-INDEPENDENT** value-vs-policy
    imbalance (a pure ratio of the two RL norms, unchanged by how many auxiliaries are on), *linear &
    non-saturating* (0 = balanced, >0 = value dominates, <0 = policy dominates, e.g. ≈+1.8 at a 66:1
    swamp). The legible gauge for **watching a PopArt / `vf_coef` fix land** — it moves linearly toward 0
    where `value_share` would crawl. `vf_coef` is **fixed per run** (recorded in `model_config.json`,
    FATAL to change on resume — it rescales this very gradient; tune on a fresh run; see
    `src/agents/model/CLAUDE.md` → resume-immutable training hparams).
  - `grad/policy_value_cosine` — scale-invariant (hence `vf_coef`-independent) structural-conflict
    signal: <0 ⟹ the two RL heads pull the trunk in opposing directions.
  - `grad/policy_norm_shared` / `grad/value_norm_shared` — the weighted norms, for absolute context.
  - **Per-aux breakout** (each present only when ITS head is active this minibatch — passed as the
    `aux_terms` dict): `grad/{species_belief, move_belief, latent, move_latent, win_prob, value_dist}_*`,
    each with `_share` (on the common `T`), `_norm_shared`, and `_policy_cosine` (<0 = that aux fights
    the policy). So the species CE, move BCE, SimSiam latent, move-latent grading, win-prob and value-dist
    pulls are **attributable individually** (the old combined `belief_share` lump is gone) — watch each
    sit small (~a few %); a spike with a degrading policy → lower THAT term's coef. `win_prob`/`value_dist`
    are ≈0 under `read_only` (stop-grad), real under `shaping`.
- **Value scale — PopArt prep.** From the full rollout buffer: `train/return_mean` / `train/return_std`
  / `train/return_abs_max` (exactly the `(μ, σ)` + tail an adaptive return normalizer / PopArt's ART
  half tracks) and `train/value_pred_std` (the value head's actual output spread). Watch these to SEE
  the non-stationary value-scale drift (reward annealing / policy improvement) that a static `vf_coef`
  cannot follow. Plus `train/grad_norm` (pre-clip total grad norm, mean over minibatches → grad-clip
  activity).

Cost: **2 partial backward passes on ONE minibatch per `train()` call**, plus **one more per ACTIVE
auxiliary** (species/move/latent/move-latent belief + win-prob + value-dist → up to ~8 total when every
head is on; each is the `aux_terms` dict's per-term `autograd.grad`) — all on the single sampled
minibatch, negligible vs the `n_epochs × n_minibatches` the loop already runs + trivial NumPy stats. The
probe is a **no-op**
(records nothing) when `shared_trunk_parameters` finds no matching modules (a non-Gen3 policy). **Why
it exists:** to prepare for **reducing `vf_coef`** and **adding return normalization (PopArt)** — both
target the value→trunk pressure, which can now be tuned to a number instead of inferred. (The
`+INSTRUMENTATION` markers in `instrumented_ppo.py` flag the added lines; the upstream-drift hash check
is unaffected since it hashes only `sb3_contrib.MaskablePPO.train`.)

## PopArt value-target normalization (`--use-popart`)

The fix for the swamping the diagnostics above reveal. `train()` reads `self.popart =
getattr(self.policy, "popart", None)` (built by the policy when `--use-popart`; see
`src/agents/model/CLAUDE.md` → PopArt for the math + version-checking). When present: once per
`train()` (before the epochs) `popart.update(self.rollout_buffer.returns, self.policy.value_net)`
advances the running `(mu, sigma)` **and** POP-rescales `value_net`; the value loss then becomes
`MSE(popart.normalize(returns), popart.normalize(values))` — the **normalized**-space loss, so the
value gradient into the shared trunk drops by ≈`sigma²` and stops swamping the policy. The policy's
value sites de-normalize, so `rollout_buffer.values` / GAE / advantages stay real-unit — the policy
path is untouched. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors otherwise —
self-documenting config; clipping is unnecessary with value normalization, and would clip in
un-normalized units). New diagnostics ride the same generic metrics path:
`popart/mu`, `popart/sigma` (watch them track `train/return_mean`/`return_std`),
`popart/value_weight_norm` (POP keeps it bounded). Under PopArt `train/value_loss` is the normalized
loss (≈O(1)) and `grad/value_policy_logratio` should fall from a large positive value toward ~0 (the
aux-independent value/policy balance — `grad/value_share` also drops but moves with the aux count, so the
log-ratio is the cleaner confirmation it worked).

## Tail-weighted value loss (`--value-tail-weight`)

A probe-driven critic-tail lever (off by default). A representation probe found the critic's TD-residual
tail is fat and barely anticipated (the V-tail crater the `eval/td_resid_tail` CVaR@5% already tracks),
so `InstrumentedMaskablePPO._value_loss_from_se` replaces the plain `F.mse_loss` at all **three** value
sites (PopArt-normalized / unclipped / clipped) with a **CVaR blend**:
`value_loss = (1−β)·MSE + β·mean(worst _VALUE_TAIL_FRAC=10% squared errors)`, computed in whichever
space the branch uses (NORMALIZED under PopArt, so the tail selection matches the loss scale). At **β=0
it is `se.mean()`, byte-identical to `F.mse_loss`** (the default no-op). β>0 makes the critic prioritise
the big over-claim misses it under-prices; it is **symmetric in error sign**, so V stays an unbiased
mean estimate and the GAE advantages the policy reads are unaffected — a weighting change, not a new
target. The hparam is set on the model after construction (like `_async_rollout`), **resume-immutable**
(recorded in `model_config.json`, FATAL to change on resume via `ModelVersion.check_value_tail_weight`,
`MODEL_CONFIG_VERSION` v11; excluded from `check_compatible` since a frozen opponent never runs the value
loss), and **not weight-shape** (no `ARCH_SIGNATURE` bump). Pairs with the v10 `--value-active-readout`
value-head fix (`src/agents/model/CLAUDE.md`); validate both by watching `eval/td_resid_tail` fall.
Tests: `instrumented_ppo_test.py` (β=0 == MSE, β>0 == the exact blend).

## Gradient accumulation (`--grad-accum-steps`)

A **GPU-memory lever** for keeping a large effective batch when the full minibatch OOMs. Stock
`MaskablePPO.train()` does one `forward → backward → optimizer.step()` **per minibatch**, so
`batch_size` couples the effective-batch size to the activation-memory peak — there is no
`accumulation_steps` knob upstream. `InstrumentedMaskablePPO.train()` adds one: with
`--grad-accum-steps K` it runs K `batch_size`-sized **micro-batches**, summing their gradients, and
calls `optimizer.step()` only **once per group of K**. Because gradients are additive and each
micro-loss is scaled by `1/K`, the accumulated gradient is the **exact** gradient of one
`(batch_size·K)` batch — but the backward graph only ever holds **one micro-batch's** activations.
So `--batch-size 4096 --grad-accum-steps 4` trains with the dynamics of `--batch-size 16384` at ~¼
the activation peak (the `DamageOperator`'s `[B,6,~416]` tensors + the grad-balance probe's retained
graph scale with the micro-batch, not the effective batch). `K=1` (default) is **byte-identical to
upstream** (one step per minibatch).

- **The step is gated on a full group** (`micro_in_group == accum`); a **trailing partial group**
  (#minibatches not divisible by accum) is flushed at epoch end with its accumulated grad rescaled
  `accum/micro_in_group` so the short group's step has the right magnitude. Grad-norm clipping
  (`max_grad_norm`) is applied **once per optimizer step** (per group) — i.e. to the full
  effective-batch gradient, exactly as the big batch would clip it.
- **Bit-exact when the rollout divides cleanly.** The accumulation math reproduces a literal
  `batch_size·K` batch to the float32 noise floor (~3e-8, empirically) **when `batch_size` divides the
  rollout (`n_steps·n_envs`) AND `K` divides the minibatch count** — then every group is `K` equal-size
  micro-batches. Production power-of-2 configs satisfy this (e.g. rollout 131072, `--batch-size 4096
  --grad-accum-steps 4` → 32 micro-batches, 8 groups, exact). For a NON-divisible rollout the single
  smaller remainder minibatch in the **final group of each epoch** is weighted as if full-size — a
  bounded mis-weighting of one remainder per epoch (≈8e-5 on params in a toy probe; negligible vs a
  100k-sample rollout, and no worse than stock SB3, which gives that remainder minibatch its own
  full-weight optimizer step).
- **KL early-stop** (`target_kl`, `None` by default so this path is dormant) discards the partial
  group (`zero_grad`, no step) on a trip — a true `(batch_size·K)` batch checks KL over the whole
  effective batch and would discard it as one unit.
- **The other (always-present) non-identity is per-micro-batch advantage normalization**
  (`normalize_advantage`, default on): stock SB3 already normalizes advantages *per-minibatch*, so
  here the normalization sample is the micro-batch (e.g. 4096) rather than the effective batch
  (16384). The difference is the normalization sample size — statistically negligible for batches of
  thousands (and it is this term, not the accumulation math, that the bit-exact check above isolates
  by running with `normalize_advantage=False`). (The grad-balance probe also samples on the first
  **micro**-batch instead of the first minibatch — a smaller, cheaper, still-representative sample; its
  `retain_graph` memory shrinks with the micro-batch.)
- **Not version-locked / not in `model_config.json`.** It is a pure train-loop knob (no forward
  change, no weight-shape effect, no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump) — like `batch_size`
  / `n_epochs`, **forwarded as a CLI flag on every launcher resume** (set on the model in both the
  fresh-build and resume paths of `train_rl_agent.py`; surfaced in `_model_hparams` for the sidecar).
  Change it freely on resume; only the *effective* batch (`batch_size·K`) matters for dynamics, so
  `--batch-size 16384` (K=1) and `--batch-size 4096 --grad-accum-steps 4` continue a run identically.
- **The upstream-drift hash check is unaffected** (it hashes only `sb3_contrib.MaskablePPO.train`).

Tests: `instrumented_ppo_test.py` — `test_grad_accum_matches_full_batch` runs the REAL `train()` on a
minimal `MaskablePPO` and asserts `K=accum` over `batch/K` micro-batches reproduces the parameter
update of `K=1` over the full batch to `rtol=1e-4` (parametrized over a divisible 16=4×4 **and** a
non-divisible 15=5×3 case that exercises the partial-group rescale), plus default-is-1 + source-marker
guards.

### Gradient noise scale (`train/noise_scale`) — "is the batch big enough?"

A **free byproduct of accumulation** (only emitted under `--grad-accum-steps >= 2`) that answers *how
big a batch is enough* with a number instead of intuition: the McCandlish et al. 2018 **simple
gradient noise scale** `B_simple = tr(Σ)/|G|²` — the critical batch size where gradient noise stops
dominating. Below it, a bigger batch buys ~linear per-step progress; above it, diminishing returns
(you're averaging out noise that was already small, and could shrink the batch for more update steps).

The estimator needs the squared gradient norm at **two batch sizes** — and accumulation produces
exactly that for free each `train()`: ‖g‖² of one micro-batch (`B=batch_size`, read from `.grad`
right after the first micro-batch's backward, un-scaled by `accum²`) and of the accumulated first
group (`B=batch_size·accum`, the pre-clip norm `clip_grad_norm_` already returns). From the model
`E‖Ĝ_B‖² = |G|² + tr(Σ)/B`, two `(B, ‖Ĝ_B‖²)` points pin both `|G|²` and `tr(Σ)` (`_noise_scale_estimate`,
pure/unit-tested). Both single-call estimates are noisy (either can go negative), so the **numerator and
denominator are EMA'd separately** (`_NOISE_SCALE_EMA_DECAY`=0.99 ≈ a few-hundred-call window) and only
then divided — and the scalar is emitted only once both EMAs are positive (so a warmup transient never
logs a garbage value). Cost: one extra global grad-norm read per `train()` (the group norm is reused
from clipping); no extra backward. EMA state is **process-local** (resets on a launcher restart →
re-converges in a few hundred calls; not saved).

Two scalars ride the standard logger → TensorBoard + launcher TUI (`format.py` labels `noise scale` /
`noise/batch`, in the train column by `train/grad_norm`):
- **`train/noise_scale`** = `B_simple` (compare directly to your effective batch `batch_size·accum`).
- **`train/noise_scale_ratio`** = `B_simple / (batch_size·accum)` — the actionable read: **≫1 ⇒
  noise-limited** (enlarge the effective batch), **≪1 ⇒ diminishing returns** (you have more than
  enough; could shrink for more/cheaper update steps), **~1 ⇒ the sweet spot**.

Tests: `instrumented_ppo_test.py` — `test_noise_scale_estimate_recovers_known_values` (the two-point
math recovers a planted `|G|²`/`tr(Σ)` exactly), `_smaller_batch_is_noisier_sign`, `_global_grad_sq`
matches a manual sum, and `_logged_only_when_accumulating` (real `train()`: skipped at accum=1, EMA
updated + scalar emitted at accum=2).

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
  interpretable — k=1 vs k=5 differ); `species_ce`, `moves_bce`, `aux_loss`. **Balance:** the
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
  auto-forces `--attend-unrevealed-opponents`; `unrevealed`/`both` additionally REQUIRE `--opp-belief-aux-coef>0`
  so the hidden slots carry learned tokens); `move_belief_coef` is training-only, **read back on a
  flagless resume**. The revealed-vs-unrevealed axis is the defensible-vs-omniscient A/B.
- **Tests.** Unit: `move_belief_loss_test.py` (direct-BCE, Hungarian order-invariance + min-cost match,
  mode gating, grad, fail-loud), `agents/model/move_belief_test.py` (module mask-gating + grad +
  per-mode wiring + off byte-identical), `belief_labels_test.py` (`build_known_move_labels`),
  `snapshot_test.py` (version gate + threading).

## Latent-belief loss (`--opp-belief-latent-coef`)

The training half of the latent-belief escalation (model side: `src/agents/model/CLAUDE.md` → LATENT
belief, v18). The species head predicts a hidden mon's IDENTITY discretely (CE); the latent head predicts
it in **role-token space** — graded supervision the CE can't give. REQUIRES `--opp-belief-aux-coef>0`
(it rides the species head's believed slots + Hungarian assignment).
- **Target (`gen3_env.py`).** When `--opp-belief-latent-coef>0` (threaded as `emit_belief_target`),
  `Gen3Env` emits a THIRD privileged training-only Dict-obs key `belief_target_slots` [6,107]: the FRESH
  per-mon obs encode (`pokemon_encoder.encode(mon, battle2, is_own=True)` + active=0) of each hidden mon
  at its believed slot, the SAME `assign_hidden_to_slots` assignment as `belief_species` (one mon per
  slot across both heads — no conflicting pulls), per-battle cached by species (a hidden mon is untouched
  → its fresh encode is stable while it is a target). Read ONLY by the loss; the model forward reads only
  `obs["observation"]`, so it can't leak.
- **Loss (`instrumented_ppo.py` `_belief_aux_loss`, the latent term).** The extractor stashed the
  prediction (`last_belief_logits["latent"]`) + the stop-grad target (`last_belief_target_latent`, the
  `pokemon_encoder` role-tokens of the true mons — a SimSiam stop-grad, the encoder is task-anchored so no
  EMA/collapse). On the **same species-CE Hungarian assignment**, the latent loss is the mean cosine
  distance over matched pairs + a **VICReg variance floor** on the predictions (collapse guard). Returned
  as the 3rd element of `_belief_aux_loss` and folded at `opp_belief_latent_coef`; its trunk gradient
  is broken out separately as `grad/latent_share` (passed in the `grad_balance_metrics(aux_terms=…)`
  dict as `"latent"`, on the common trunk-pull total) so the latent pull is attributable on its own.
- **Metrics (`belief/latent_*`).** `cosine` (similarity, higher = better identity match), `std`
  (the collapse monitor — **NO-GO if it →0 while `cosine`→1**), `vicreg`, `loss`, plus the
  **interpretability anchor** `cosine_baseline` (the cosine each prediction scores against a MISMATCHED
  true target — the non-zero null of the task-anchored, non-orthogonal role-token manifold) and
  `cosine_above_chance` = `cosine − cosine_baseline` (the *discriminative* signal, the latent analog of
  `species_acc_above_chance`). A small-but-positive `above_chance` with a healthy `std` is the
  "predicts the SET's mean role, not the per-mon identity" failure that `std` alone can miss.
  **Balance:** the latent term's trunk pull is broken out as **`grad/latent_share`** (+
  `grad/latent_norm_shared` / `grad/latent_policy_cosine`), on the common trunk-pull total alongside
  `grad/species_belief_share` / `grad/move_belief_share` (so the species CE, move BCE and latent are
  each separately attributable) — when tuning `--opp-belief-latent-coef` you can see whether the LATENT
  term specifically is the one swamping / fighting the policy. Watch it sit small (a few %); a spike with
  a degrading policy = lower the coef.
- **Versioning.** `opp_belief_latent` (bool) is the version-checked structural toggle (the predictor MLP;
  fresh-only; hard-requires `opp_belief_slots`); `opp_belief_latent_coef` is training-only, **read back on
  a flagless resume**. Threaded into `current_model_version` / `arch_toggles_from_model` so a latent-ON
  self-play run doesn't FATAL on its own sentinels (the 4 opp-load sites).
- **Tests.** Unit: `belief_aux_loss_test.py` (latent cosine + VICReg + grad + rides-species-matching
  order-invariance), `agents/model/belief_slots_test.py` (latent head shape, target-only-with-key,
  stop-grad, the **no-leak gate** `test_latent_target_is_no_leak`, off byte-identical projections).
  **Fuzz** (real bridge battles, no server): `poke_env_gaps/belief_target_fuzz_test.py` validates
  `belief_target_slots == an INDEPENDENT fresh encode` of the actual hidden mon the species label names,
  PAD slots zero, and the no-leak obs width, over thousands of live decisions:
  `python src/agents/training/poke_env_gaps/belief_target_fuzz_test.py [n_battles]`.

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
  diagnostic, → 0 as the head learns), `n_slots` (supervised slots/minibatch), `loss`.
- **Nature/EV decomposition (`gen3_nature_ev_belief_v1`, v40, `--spread-belief-nature`).** The fix for the
  stuck `largest_bias`: the additive head predicts the DERIVED stat directly (a point estimate BETWEEN the
  nature ×1.1/×0.9 modes); the generative head predicts a NATURE categorical ⊕ Smogon prior + per-stat EVs ⊕
  prior and COMPUTES the derived stat, so the asymmetry + EV budget are structural. A SECOND loss term
  `_nature_ev_belief_loss` (nature CE + EV smooth_l1 over REVEALED slots, folded at the SAME
  `spread_belief_coef`, metrics `belief/natureev_{nature_acc,nature_ce,ev_mae,n_slots}`) supervises the
  decomposition DIRECTLY (the derived loss alone is many-to-one). Label: the TRUE (nature, EVs)
  **deterministically INVERTED** from agent2's `mon.stats` (`damage_tables.invert_nature_evs`, GIGO-guarded —
  gen3 hides them, so we invert the visible derived stats), emitted by `gen3_env._spread_labels` as
  training-only `belief_nature`/`belief_ev`(+masks), cached per battle. `--spread-belief-nature-marginalize`
  (op-side, forward-behavior) makes the DamageOperator marginalise the nonlinear P(KO) over the believed
  nature distribution (an exact 3-point quadrature per candidate's offensive stat). Smoke: `nature_acc` rises
  toward the true nature, `largest_bias` trends to 0.
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

## Opponent HP-type belief loss (`--hp-type-belief learned` / `--hp-type-belief-coef`)

The training half of `gen3_opp_hp_type_belief_v1` (model side: `src/agents/model/CLAUDE.md` → opponent
HP-type belief, v38). The DamageOperator's typed-HP candidates were priced "immune" because the opp's HP
TYPE was unknown (the obs `hp_probs` is empty until HP fires); the `HPTypeBelief` head learns it, supervised
here.
- **Label (training-only, privileged).** `Gen3Env._hp_type_labels` reads agent2's OWN team for each
  REVEALED opp mon's true Hidden Power type (the typed move-id suffix → `belief_labels.build_hp_type_labels` /
  `hp_type_idx_from_move_id`, in the `HIDDEN_POWER_TYPE_ORDER` index space) and emits the `hp_type_label` [6]
  / `hp_type_mask` [6] Dict keys (mask=1 only at a revealed slot whose species runs HP). Gen 3 NEVER reveals
  the opp HP type, so this can't ride the obs vector — it is leak-safe (a separate Dict key, read ONLY by the
  loss; the obs vector width is unchanged). Emitted when `--hp-type-belief learned` AND `--hp-type-belief-coef>0`.
- **Loss (`instrumented_ppo._hp_type_belief_loss`).** Reads the extractor's stashed `last_hp_type_logits`
  [6,16] (the prior⊕delta posterior logits the op consumes) + the label keys; folds `hp_type_belief_coef ·
  cross_entropy` over the masked (revealed-HP) slots. Gradient flows posterior → `hp_type_head` → opp tokens →
  trunk (joins the per-head grad-balance probe as `grad/hp_type_*`); `aux_probe_terms["hp_type"]`.
- **Metrics (`belief/hptype_*`).** `acc` (top-1 HP-type accuracy — should climb above the 1/16≈0.06 chance),
  `loss`, `n_slots`. `hp_type_belief_coef` is **training-only** (inherited on a flagless resume, like
  `spread_belief_coef`); the head/op-fix structure is gated by the version-checked `hp_type_belief_mode`.
- **Tests.** Unit: `model/hp_type_belief_test.py` (the immune-bug-and-fix, bare-237 mask, effectiveness
  narrowing, cold-start==prior, op-consumes-posterior grad flow, the tri-state module build + projection
  parity, the CE loss masking, `build_hp_type_labels`, the 16-axis GIGO pin, the version gate). **Fuzz**
  (real bridge battles): the extended `poke_env_gaps/belief_labels_fuzz_test.py` validates `hp_type_label` ==
  each revealed HP-mon's true type, mask 0 on revealed-no-HP / believed / pad slots (no leak), and the OFF env
  declaring no HP-type keys. End-to-end smoke (`--debug --damage-op --move-belief-mode revealed
  --move-prior-fusion --hp-type-belief learned --hp-type-belief-coef 0.05 --n-steps 64`) confirms the
  roundtrip + `belief/hptype_*` (acc 0.1→0.5 over the smoke).

## Win-probability head (`--win-prob-mode` / `--win-prob-coef`)

The training half of the tri-state win-probability head (model side: `src/agents/model/CLAUDE.md` →
win-probability head, v22). A calibrated **P(win|state)** the shaped critic can't give — supervised by the
Monte-Carlo episode OUTCOME. Off by default (`--win-prob-mode none`). Three pieces live here:

- **The label is a FUTURE quantity** — the outcome is only known when the battle ends, so (unlike the
  per-step belief labels, which are privileged info known *each* step) it CANNOT ride as a real per-step
  obs key. The plumbing reuses the obs-dict-label STORAGE path with post-hoc population:
  - **`gen3_env.py`** declares two TRAINING-ONLY obs keys when `emit_win_target` (`--win-prob-mode != none`):
    `win_target` [1] + `win_mask` [1] (float32), and emits PLACEHOLDER zeros each step (`_merge_training_keys`).
    The rollout buffer therefore stores + shuffles them automatically (the belief-label path). Read ONLY by
    the loss; the model forward reads only `obs["observation"]`, so the OUTCOME can't leak.
  - **`MaskableAgentWrapper.step` (`wrappers.py`)** sets `info["win_outcome"]` (1.0 win / 0.0 loss-or-tie,
    from `battle1.won`) at the done step (before the VecEnv auto-resets).
  - **`WinProbLabelCallback` (`win_prob_callback.py`)** captures each terminal outcome during collection
    (SYNC: in `_on_step` at `rollout_buffer.pos`; ASYNC: the `collect_rollouts_async` collector records it
    inline at the env's just-written `(t, i)` buffer row — it owns the row, the wave-batched `on_step`
    can't recover it), into a shared `model._win_terminal_scratch` [n_steps, n_envs]. At `_on_rollout_end`
    (before `train()`) it propagates each episode's outcome BACKWARD to all its steps (γ_win = 1, undiscounted
    → P(win|s) = "probability this state leads to a win") and OVERWRITES the buffer's `win_target`/`win_mask`
    placeholders. The trailing IN-PROGRESS episode (no terminal yet in-buffer) gets `win_mask=0` and is
    excluded — never trained toward a fabricated label. Only added to the callback list when the head is on.
- **Loss (`instrumented_ppo.py` `_win_prob_loss`).** `train()` reads `last_win_prob_logits` (stashed by the
  `evaluate_actions` forward) + `rollout_data.observations["win_target"]`/`["win_mask"]`, folds
  `win_prob_coef · masked-BCE`. read_only vs shaping differ ONLY in whether the extractor stop-grads the
  head's input (the trunk gradient) — the loss term itself is identical. Folded whenever the extractor's
  `win_prob_mode != none` AND `win_prob_coef != 0`.
- **Metrics (`win_prob/*` — its OWN TB prefix, not `train/`, matching the `grad/`/`popart/`/`eval/`
  groups).** Calibration: `acc` (top-1 win/loss) + `brier` (lower = predicted P(win) tracks the win
  rate); `pred_mean` vs `label_mean` (base-rate-collapse watch); `coverage` (fraction with a known label);
  `loss`. **Information value (the aggregate Brier hides it — a blowout's P(win) is trivially recoverable
  from material):** `brier_contested`/`acc_contested` restrict to CLOSE games (`|win_margin| <
  _WIN_CONTESTED_TAU`=0.25, the normalized material margin from `_compute_phi_mat`, emitted as the
  `win_margin` obs key) — judge `brier_contested` vs a 50/50 game's ~0.25 no-skill floor;
  `contested_frac`/`contested_label_mean` (≈0.5 confirms even); and **`skill_vs_material`** = the Brier
  skill score vs a material-only baseline (`P_mat = clip(0.5+0.5·margin)`) — **>0 ⇒ the head adds info
  beyond counting mons** (the headline value number; `brier_material` is the baseline for context). The
  shared-trunk pull rides `grad/win_prob_share` (the `grad_balance_metrics(aux_terms=…)` `"win_prob"` entry) — **≈0 under
  read_only** (stop-grad, the live confirmation the diagnostic isn't perturbing the policy), real under
  shaping (watch it sit small; a spike with a degrading policy → lower `--win-prob-coef`).
- **Versioning.** `win_prob_mode` (str) is the structural + resume-IMMUTABLE toggle (any change FATALs;
  threaded into `current_model_version` / `arch_toggles_from_model` so a win-prob-ON self-play run doesn't
  FATAL on its own sentinels); `win_prob_coef` is training-only, **read back on a flagless resume**.
- **Forensic trace + prober.** `RLPlayer._win_prob` (`inference/player.py`) reads the stashed
  `last_win_prob_logits` at trace-capture time (sigmoid ⇒ P(win)) into the per-decision `state`, which
  `BattleRecorder.states_arrays` writes as a `win_probs` npz array (NaN = no head / not captured, parallel
  to `values`). The prober renders **P(win) + ΔP(win)** in the Summary + Outcome panels beside CRITIC's
  V/ΔV — "how a move moved the win odds" — model-free from that array (`engine.WinProbView`); `None`/absent
  on a non-win-prob run. See `src/main/prober/CLAUDE.md`.
- **Tests.** Unit: `agents/training/win_prob_test.py` (loss masking + None guards + the callback MC-fill
  backward-propagation + in-progress masking + sync-capture-at-pos + async-skip), `agents/model/
  win_prob_head_test.py` (module build, off byte-identical projection dims, the read_only-stop-grad /
  shaping-flows gradient gating, the v22 version gate). End-to-end `--debug --use-showdown-bridge
  --win-prob-mode read_only` smoke confirms the roundtrip + `train/win_prob_*` metrics + `win_prob_share`=0.

## Public-replay value aux — V_pub (`--pubval-mode` / `--pubval-coef`)

`gen3_pubval_aux_v1` (v43; design `designs/ai_v8/design_public_info_value.md`). The measured limiter is
the value function's positional blindness (defensive AUC ≈ 0.50 → advantage ≈ 0 on positional decisions);
V_pub is the value-INDEPENDENT exogenous signal that attacks it: **P(win | PUBLIC board), calibrated on
the human replay corpus**, wired as a dense per-step shared-trunk aux target — the trunk sees WHEN a game
swung (hazards/status/attrition priced by HUMAN outcomes), not only how it ended (the credit-assignment
lever). Where the win-prob head learns P(win) from SELF-PLAY outcomes (inheriting the bootstrap's blind
spots), V_pub's pricing comes from outside the loop. Never in pi/vf, never in GAE (V^human ≠ V^π). Off by
default; pieces:

- **The frozen artifact (`data/gen3_pubval.json`)** — a 17-feature logistic (`agents.training.pubval`:
  material diffs + spikes/status/boost/revealed diffs + absolutes + turn clock + weather one-hot — the
  POC-validated "crude aggregates"; richer identity features overfit, do not add them) trained by
  `python -m agents.training.pubval_calibration` on the rated gen3ou replay corpus
  (`replays/showdown/gen3ou/`). Current artifact: **170,769 games / 8.58M positions, held-out-by-game AUC
  0.7343, turn-1 AUC 0.500** (the leakage guard — the calibration CLI refuses to write if it drifts off
  ~0.5), calibrated ([0.8,1.0)→0.877); provenance in `meta`. The `bot_elo_calibration` artifact pattern.
- **The live target (`gen3_env._pubval_target`)** — when `emit_pubval_target` (= `--pubval-mode != none`),
  each decision folds the vetted **LiveView** through the SAME `PubSide`/`features()` the corpus parser
  used and emits `pubval_target` [1] + `pubval_mask` [1] as TRAINING-ONLY Dict keys (a REAL per-step value
  like `win_margin` — V_pub is a function of present public state, no callback back-fill). The artifact
  loads once at env construction (missing/stale → fail-loud with the regen command; also checked at
  arg-parse). Cost: one extra `live_view()` per decision, only when the flag is on.
- **The head + loss** — `PubValHead` (the WinProbHead architecture, a named subclass) reads
  `value_pooled`; tri-state `--pubval-mode {none,read_only,shaping}` (read_only = stop-grad learnability
  probe: CAN the trunk carry V_pub?; shaping = the human positional prior shapes the trunk — the
  experiment). `instrumented_ppo._pubval_loss` = masked soft-target BCE folded at `--pubval-coef`
  (default 0.1, training-only, flagless-resume-inherited).
- **Metrics (`pubval/*`)** — watch **`mae`** (|sigmoid − V_pub| → 0 as it fits; the raw `bce` floors at
  the soft target's own entropy, so its level is NOT the fit signal), `pred_mean` vs `target_mean`
  (base-rate-collapse watch), `coverage` (≈1). Trunk pull rides `grad/pubval_share` (+`_policy_cosine`):
  ≈0 under read_only, real under shaping. **The acceptance gate for the experiment is NOT these** — it is
  the critic's defensive-AUC-by-style transfer (the calibration-by-style probe) + WR/ELO vs a pubval-off
  control.
- **Versioning** — `pubval_mode` is STRUCTURAL + resume-immutable (STRING-gated in `check_compatible`
  like `win_prob_mode`; v43, migrate default "none"; threaded through `current_model_version` /
  `arch_toggles_from_model` / `_run_arch_toggles` + both extractor-kwargs sites); `pubval_coef`
  training-only. OFF byte-identical (no `ARCH_SIGNATURE` bump).
- **Tests** — `pubval_test.py` (the shared feature definition, the corpus parser incl.
  faint-clears-status/cureteam/boost-handlers, artifact round-trip + the committed-artifact sanity, the
  LiveView fold, the loss math/mask/guards), `model/pubval_head_test.py` (build, off-byte-identical,
  read_only-stop-grad vs shaping-flows, the v43 gate, migration), and the **parity fuzz**
  `poke_env_gaps/pubval_parity_fuzz_test.py` (bridge, no server): folds the trainee's OWN protocol
  stream through the corpus parser and asserts every `PubSide` field == the live fold at every decision
  + the emitted `pubval_target` == the artifact's prediction (anti-vacuous-run guard; the capture hook
  must install BEFORE `attach_bridge_transport` — the bridge captures the bound handler at attach time).

## Distributional value head (`--value-dist-mode` / `--value-dist-coef`)

The training half of the v29 interpretability side head (model side: `src/agents/model/CLAUDE.md` →
distributional value head). A categorical readout off `value_pooled` whose softmax is the critic's
predicted **return DISTRIBUTION** — the shape the scalar V collapses (sharp = confident, wide =
uncertain, bimodal = coinflip). **Phase A** (interpretability-only): it does NOT replace the scalar
critic, so the GAE/advantage/value-loss path is untouched — this loss is an ADD-ON, like the win-prob
aux. Design + the K1 honesty frame: `designs/ai_v6/design_distributional_value_critic.md`.

- **Loss (`instrumented_ppo._value_dist_loss`).** **HL-Gauss** (Farebrother et al. 2024): build a
  Gaussian-smoothed soft target by integrating `N(target, σ_g²)` (σ_g = 0.75·Δ) over each atom's bin,
  with the two EDGE bins absorbing the outer tails (graceful out-of-support handling), then cross-entropy
  against the head's `log_softmax`. `train()` reads the stashed `last_value_dist_logits` + the rollout
  return as the target, **PopArt-normalized when the scalar critic is** (so the target lands in the head's
  support space — set `--value-dist-vmin/vmax` to a normalized range like ±5 under `--use-popart`). Folded
  at `value_dist_coef`. Pure + static → unit-tested in `value_dist_loss_test.py`.
- **Metrics (`value_dist/*`).** Aggregate interpretability health under its own TB prefix (the
  `grad/`/`popart/`/`win_prob/` group convention): `ce`, `entropy` + `std` (fall as the critic sharpens),
  `pit_mean` (≈ 0.5 ⟺ **calibrated** — the PIT anchor), `mean_abs_err` (`|E[Z] − return|` in support
  units). Ride the generic logger → TensorBoard + launcher TUI (`value_dist/*` labels in `format.py`).
- **Versioning.** `value_dist_mode` (str) + `value_dist_bins` (int) are version-checked structural toggles
  (fresh-only); the support `vmin`/`vmax` is resume-immutable (`check_value_dist`); `value_dist_coef` is
  **training-only**, read back on a flagless resume (like `win_prob_coef`). Threaded into
  `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles`.
- **Forensic trace + prober.** `RLPlayer._value_dist` reads the stashed logits at capture (softmax ⇒ the
  per-atom distribution) → `BattleRecorder.states_arrays` writes a `value_dist [T, bins]` npz array (key
  OMITTED when the head is off → the prober's KeyError "unavailable" path; NaN rows = uncaptured). The
  prober renders the per-decision **histogram** + mean/std/P10–P90/entropy/bimodality
  (`engine.build_value_dist` → `ValueDistView`, model-free; in the Summary panel + the `analyze` CLI). See
  `src/main/prober/CLAUDE.md`.
- **Honesty gate.** Ledger **K1 already killed the distributional critic as a WIN-RATE lever** (sub-Gaussian
  residuals — no tail). This is justified on INTERPRETABILITY only; its strongest use is upgrading the
  prober calibration/`falsify-scan` luck-vs-mistake split (predicted spread vs realized return = a
  within-model PIT). "Learns ≠ helps" — validate calibration (PIT ≈ uniform), not win-rate.
- **Tests.** Unit: `value_dist_loss_test.py` (HL-Gauss math + diagnostics), `agents/model/
  value_dist_head_test.py` (module build, off byte-identical, grad gating, the v29 version gate),
  `main/prober/engine_test.py` (`build_value_dist`). End-to-end `--debug --debug-eval --use-showdown-bridge
  --value-dist-mode read_only` smoke captures a trace whose npz carries `value_dist`.

## Exploiter distillation (`--distill-teacher` / `--distill-coef` / `--distill-value-coef` / `--distill-value-feat-coef`)

`gen3_exploiter_distill_v1` — pour a frozen per-team SPECIALIST (an exploiter) into the generalist so it
learns to PILOT that team, closing the amortization gap the self-play average can't. `--distill-teacher`
takes `TEACHER:TEAM` colon pairs (comma-separated, N teachers — a checkpoint dir → `best_model.zip`, bound
to its Showdown team file); the env emits a training-only integer `distill_mask` obs key (0=none, k=teacher
k) on states where the trainee pilots team-k (biased there by `--distill-team-bias`, default 0.4; rest =
pool rehearsal → no forgetting). In `train()`, for each teacher a frozen forward gives π_teacher and
`distill_coef · KL(π_teacher ‖ π_student)` is folded, masked to that teacher's states; the per-teacher
mean-KLs are AVERAGED (a small-coverage teacher still contributes comparable gradient). Reuses the
`evaluate_actions` forward's stashed `_last_pi_distribution` (bit-identical, one fewer forward). Metrics
`distill/{kl, agree_rate, tK_kl, tK_coverage, n_teachers_active}`. OFF (coef 0 / no teacher) byte-identical;
training-only, NOT version-locked (inherited on a flagless resume). Validated (ai_v7_16→_19): offense
transfers (TSS-piloting 0.475→0.75) and HOLDS under the double-sided recipe (see the memory).

- **VALUE distillation (`gen3_exploiter_value_distill_v1`, `--distill-value-coef`, default 0 = OFF).** The
  policy KL is POLICY-ONLY — the student pilots the teacher's team with its own amortized (~4-dim) critic,
  so it mimics the MOVES but never gets the teacher's per-team VALUE (confirmed: value_cls effective rank
  FLAT across _14→_18→_19). This adds `distill_value_coef · MSE(V_teacher, V_student)` on the SAME
  teacher-team states, in the student's PopArt-normalized frame (`_value_distill_mse`, a static testable
  helper mirroring `_distill_loss`; teacher V from a frozen `predict_values`, real-unit → normalized).
  **Coherent despite V^π being policy-relative** because the policy KL simultaneously drives
  π_student→π_teacher there, so V_teacher becomes the right target — hence it **requires `--distill-coef > 0`**
  (arg-parse guard). Metrics `distill/{value_mse, tK_value_mse}`. **The A/B lever:** policy-only
  (`--distill-value-coef 0`) vs policy+value (>0), read out by the value_cls effective-rank probe
  (`rank_metrics.py`, `tmp/value_rank_compare.py`) — does distilling the value ENRICH it. OFF byte-identical
  (no teacher predict_values forward); training-only. **Distributional-value distill** (distil the teacher's
  `ValueDistHead` return distribution, enabling later archetype-token conditioning) is a future follow-on.
- **FITNETS value-FEATURE distillation (`gen3_exploiter_value_feat_distill_v1`, `--distill-value-feat-coef`,
  default 0 = OFF).** Matching only the teacher's SCALAR V CRYSTALLIZES the critic — the A/B on ai_v7_20
  confirmed `distill/value_mse` falls but the value_cls effective rank DROPS (4.15→3.55): a scalar target has
  only ~1 dim of information, so the critic collapses onto it. The FitNets (Romero 2015) "hint" fix distils
  the teacher's INTERMEDIATE representation instead: `distill_value_feat_coef · (1 − cos(value_pooled_student,
  value_pooled_teacher))` on the SAME teacher-team states, where `value_pooled` is the extractor's 128-dim
  value-CLS pool (`features_extractor.last_value_pooled`, stashed EVERY forward — the hint layer). So the
  trunk inherits the teacher's per-team value STRUCTURE, not just its output. **COSINE, not MSE** (the loss
  choice from the geometry analysis `tmp/fitnet_analysis.py`): the four teachers' value subspaces are low-rank
  (PR ~3–5 even for specialists), COMPLEMENTARY (TSS orthogonal 0.04–0.07 to the others, collective effRank
  ~12), and NON-competing (all pull-cosines positive) — so a scale-free directional pull transfers the correct
  structure without over-constraining a low-rank target the way a raw-magnitude MSE would; the student/teacher
  are common-ancestor forks (all forked from _14), so their `value_pooled` bases are approximately shared and a
  direct (regressor-free) cosine is meaningful (a lower bound on alignment). The student hint (from the
  `evaluate_actions` forward, WITH grad) + each teacher's hint (captured under `no_grad` right after the KL's
  `get_distribution` forward, detached — no extra teacher forward) go through the static `_value_feat_distill`
  (masked mean cosine distance, per-teacher averaged like the KL). **Requires `--distill-coef > 0`** (the
  policy KL makes the teacher's `value_pooled` the right target — V^π is policy-relative). Metrics
  `distill/{value_feat_cos, tK_value_feat_cos}`. **The A/B lever:** scalar (`--distill-value-coef`) vs FitNets
  (`--distill-value-feat-coef`), read out by the value_cls effective-rank probe — does the HINT enrich the
  critic where the scalar crystallized it. Composes with the scalar term (both can be on). OFF byte-identical
  (no teacher `value_pooled` read); training-only, NOT version-locked (inherited on a flagless resume).

Tests: `instrumented_ppo_test.py::test_distill_*` (policy KL: identical→0, masking, illegal-mask, None-guard,
grad-student-only, reuse-bit-identical, multi-teacher averaging) + `::test_value_distill_*` (equal→0, masking,
None-guard, PopArt-frame scaling, grad-student-only) + `::test_value_feat_distill_*` (aligned→0 + scale-free,
masking→cosine-distance, None-guards, grad-student-only).

## Search-as-teacher (`--search-teacher`, `teacher/` package)

Selective **Expert Iteration** — the offline-teacher plateau-breaker (design:
`designs/ai_v6/design_search_teacher.md`). Each cycle, **search + rollout-confirm the worst loss
craters** of recent eval traces and distil the VERIFIED-better action into the policy via an
**advantage-weighted CE aux loss (AWR)**. Off by default (`--search-teacher` absent / coef 0 ⇒
byte-identical). The "expert" is the prober's `better_line` beam + the rollout-confirm tiers
(`src/main/prober/`); this wires them into training. Package `src/agents/training/teacher/`:

- **`selection.py`** (`select_candidates`, Phase 0, model-free) — the two-stage funnel:
  `ProbeSession.scan` ranks the worst-ΔV loss craters → `falsifier.falsify_battle` gates to *reducible
  MISTAKEs* (not aleatoric LUCK — don't teach against dice) → expand to the crater **±window** (the
  cause is usually 1–2 turns BEFORE the value crater). Ranks by |δ|, caps at the budget.
- **`opponent_resolver.py`** (`resolve_opponent`) — the EXACT opponent: a `sentinel_<i>` trace → its
  `models/<run>/snapshots/snapshot_<step>.zip` (the positional index→step map is in
  `metadata.json:latest_eval.pool.sentinels[i].snapshot`, valid only for the latest cycle — which is
  what the teacher runs on); a bot → reproducible from its name; anything else → **`'unresolved'` →
  SKIPPED, never approximated** (distilling "A* beats a proxy" is a soundness failure, not a degrade).
- **`produce.py`** (`produce_correction`) — the 3-tier strictly-better gate: SEARCH (`session.better_line`
  with `interior_opponent='ckpt'`, the exact opp) → CONFIRM (rollout-to-end vs the same exact opp,
  Wilson CI) → GATE (keep only if the Wilson LOWER bound beats the played loss rate). Distils the
  **CONFIRMED** win-rate improvement (`confirmed − played`), never the critic's optimistic backed-up
  value (the Spore 95%-vs-62% lesson). Staleness re-verify: if the frozen trainee already argmaxes A*,
  skip (`already_known`).
- **`buffer.py`** (`Correction`, `CorrectionBuffer`) — a bounded recency RING of corrections, sampled
  (with its own forward) on each rollout minibatch inside `train()`. **STANDALONE, not the rollout buffer** — the searched states are
  off-policy (older eval traces), so they must never enter GAE / the clip objective. Lives on
  `model._correction_buffer`.
- **`callback.py`** (`SearchTeacherCallback`) + **`src/main/search_teacher_worker.py`** — the
  non-blocking driver mirrors the eval cadence: freeze the trainee, spawn frozen-snapshot worker
  subprocesses (own POKE_LOOP, spare cores — the live trunk mutates, so a thread is unsafe; isolation
  is why eval uses subprocesses too), each runs the search + confirm over a candidate slice (ONE warm
  `SearchSession` reused → the Node spawn is amortized), publishes a shard (obs `.npz` + scalars
  `.json`); the parent polls and fills the buffer. Skip-while-running, watchdog, crash-logged.
- **SUPPLY+POOL mode (`--teacher-persistent`)** — `teacher/generate.py` +
  `src/main/search_teacher_persistent_worker.py`. The per-cycle mode reads eval traces (a trickle every
  ~2M steps); the persistent mode is a LONG-LIVED worker pool that GENERATES its own fresh losses (the
  frozen trainee vs sampled current opponents — the recent pool snapshots + bots — recorded via the
  eval forensic path `begin_forensic_cycle` + `run_local_battles`) and searches them CONTINUOUSLY,
  dripping corrections into the buffer instead of a 2M-step burst. The parent RE-FREEZES the snapshot
  every `--teacher-refresh-steps` (default 500k, written to a polled `control.json`) so long-lived
  workers track the moving policy, and `_ingest`s correction shards incrementally each `_on_step`.
  Because the worker CHOSE the opponent, the exact-opponent is KNOWN directly (no sentinel-resolution
  fragility); `falsify_gate=False` here (supply is plentiful → the CONFIRM is the gate). Never touches
  the training hot path (a frozen-snapshot side activity, like eval). Validated end-to-end: one worker
  published 8 verified-better corrections from self-generated battles in ~150 s. Flags:
  `--teacher-persistent`, `--teacher-refresh-steps`, `--teacher-gen-battles`.
  - **Lifecycle hardening** (a long-lived, multi-process pool must self-heal — an adversarial review
    surfaced these): the parent `_reap_and_respawn`s a crashed worker on a step-backoff (so a dead
    worker can't silently drain the pool to zero — `teacher/workers_alive`/`worker_respawns_total`);
    snapshot pruning keeps the latest **three** (numeric `_version_key`, not lexical — `v10 > v9`) and
    the worker re-checks `os.path.exists` before every snapshot/opponent load + wraps both in try/except
    (a pruned/corrupt file SKIPS the iteration, never crashes); `_spawn_persistent` wipes stale shards +
    `gen_*` dirs from a prior crash/restart so a fresh pool never double-ingests; `_ingest` CONSUMES
    (deletes) a shard BEFORE buffering it (a delete failure DROPS it rather than re-globbing it into a
    duplicate); the worker's per-iteration `ProbeSession` is a context manager (drops its cached models)
    and the warm `SearchSession` recycles every `recycle_every` (Node V8-heap backstop; the launcher's
    3 h restart owns the rest). **The `_correction_buffer` is `_excluded_save_params` from the SB3 save**
    — it holds a `threading.Lock` that cloudpickle can't serialize (it would crash `model.save()` at the
    pre-train roundtrip smoke for EVERY `--search-teacher` run), and it's transient scaffolding like the
    rollout buffer (re-created empty on resume; keeps checkpoints small).

**The AWR aux loss** (`InstrumentedMaskablePPO._searchteacher_loss` + the `train()` fold): `coef ·
advantage-weighted CE(π(·|s), A*)` over a minibatch sampled from `_correction_buffer` with its OWN
policy forward (`get_distribution`); weight `w = clamp(exp(advantage/β), w_clip)`. The advantage is the
CONFIRMED win-rate improvement (NOT a critic advantage — the soundness point). The shared-trunk pull
rides `grad/searchteacher_share` / `_policy_cosine` (the live "is the teacher fighting the actor"
signal). `teacher/*` metrics: `agree_rate` (π ↔ A*, should RISE), `mean_adv`, `mean_w`, `loss`, `n`,
`buffer_size`, `corrections_per_cycle`, `yield`, `mean_confirmed_dwin`.

**On-policy self-distillation (OPD) — the KL upgrade of AWR (`--opd-coef`).** AWR distils only the
single verified-better action A*; OPD upgrades the distillation TARGET to the FULL improved distribution
**π'** via `opd_coef · KL(π' ‖ π_student)` (`InstrumentedMaskablePPO._opd_loss` + its own `train()` fold,
modelled EXACTLY on the AWR fold). π' is the softmax over LEGAL actions of the beam's per-action
**backed-up** values `(v(a) − max_legal_v) / opd_beta`, with a COMPLETED-Q floor (min legal value) for a
legal-but-unsearched slot and 0 on illegal slots — built worker-side in `produce.py` (`_build_pi_target`,
only when `build_pi_target`, so no cost off) and carried on the `Correction` as a NEW `pi_target [11]`
field (appended LAST, default None → an AWR-only run is backward-compatible). It travels the worker shard
`.npz` (like obs/mask, a NaN row = None) and `CorrectionBuffer.to_tensors` stacks it (all-present → a
tensor; **any-None → the key is None** so the KL None-guards — never a partial batch). The OPD fold
samples the **SAME** `_correction_buffer` (its own `get_distribution` forward), so a Correction carries
BOTH targets and a run can **A/B AWR vs KL** by which coef is set. `opd/*` metrics: `kl` (should FALL),
`agree_rate` (student ↔ π' mode, should RISE), `pi_target_entropy` (π' sharpness), `n`; the shared-trunk
pull rides `grad/opd_share` / `_policy_cosine`. **Training-only** (0 = byte-identical, NOT in
ModelVersion / `check_compatible` / any `check_*` → both A/B arms resume a pre-OPD checkpoint with zero
FATAL risk; coefs `_resolve`-inherited on a flagless resume). **Requires `--search-teacher`** (it fills
the buffer + its workers build π'; a `parser.error` guards `--opd-coef>0` without it).

**Why NOT value-only:** the search VALUE is the *improved-policy* value V^π*(s); regressing the PPO
critic (which must predict V^π for GAE) toward it biases advantages. So the signal is the **policy**
(AWR); the off-policy value term is wired but `--search-teacher-value-coef 0` by default (the
joint-ExIt A/B). **All training-only** (no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump; coefs
`_resolve`-inherited on a flagless resume, operational knobs forwarded by the launcher). **Honesty
gate:** the search *finding* a better line ≠ it *helping* — validate `eval/td_resid_tail` /
calibration / ELO on a `coef=0` A/B; and ~⅔ of grind losses are matchup-lost-from-turn-1 (UNCOACHABLE
per turn) — this attacks the thrown-late ⅓.

| Flag | default | role |
|---|---|---|
| `--search-teacher` | off | master enable (constructs the callback + buffer) |
| `--search-teacher-coef` | `0.0` | AWR policy CE weight (0 = byte-identical) |
| `--search-teacher-value-coef` | `0.0` | off-policy value term (OFF — soundness) |
| `--search-teacher-beta` | `1.0` | AWR temperature β |
| `--opd-coef` | `0.0` | OPD KL(π' ‖ π_student) weight (0 = byte-identical; requires `--search-teacher`) |
| `--opd-beta` | `1.0` | OPD softmax temperature β for π' |
| `--teacher-search-budget` | `200` | candidates searched per cycle |
| `--teacher-confirm-rollouts` | `8` | Monte-Carlo confirm games (the CI gate) |
| `--teacher-search-workers` | `3` | worker subprocesses per cycle |
| `--teacher-search-freq` | `0` | steps between cycles (0 = eval freq) |

**Tests** (`src/agents/training/teacher/*_test.py`): `buffer_test` (ring/sample/stack), `awr_loss_test`
(AWR math, masking, grad), `opponent_resolver_test` (bot/sentinel/unresolved, tmp metadata),
`produce_test` (the 3-tier gate with a fake session), `selection_test` (the funnel with a fake
ProbeSession + monkeypatched falsify), `callback_test` (shard→buffer collect + crash-graceful); plus
`instrumented_ppo_test.py::test_search_teacher_*` (the AWR fold in a real `train()` moves the policy
toward A*; off-by-default no-op). **OPD tests:** `instrumented_ppo_test.py::test_opd_*` (the `_opd_loss`
KL — 0 at the fixed point / >0 otherwise / None-guards / illegal-action masking — plus the real-`train()`
fold moving the policy toward π', off-byte-identical even with a populated buffer, and the AWR-only
π'-less buffer being skipped), `teacher/buffer_test` (`pi_target` roundtrip: all-present → tensor,
any-None → None), `teacher/produce_test::test_pi_target_*` (π' sums to 1 over legal / 0 illegal / peaks
A* / temperature flattens / completed-Q floor). End-to-end pipeline (selection → exact-opp search →
confirm → gate → Correction) validated against a real run.

## Process liveness guards (`watchdog.py`)

Two daemon-thread watchdogs keep a hung/abandoned run from lingering:

- **`start_subprocess_watchdog`** — for the `SubprocVecEnv` path. A crashed worker leaves the
  parent blocked on a pipe `recv` forever; this thread polls `processes` and `os._exit(1)`s the
  moment a worker dies with a nonzero exitcode. Started *after* env construction (and, in
  self-play, after `_maybe_engage_self_play` rebuilds the env), right before `learn()`. It is a
  **no-op on the `--debug` DummyVecEnv path** (no worker processes to watch).
- **`start_orphan_watchdog`** — for the `--debug` smoke path, which has no worker watchdog. A
  smoke run is a child of the launching shell/agent; if that parent dies the run is orphaned
  (PPID changes) and a hung smoke (e.g. a vanished `9XXX` server) would otherwise sit as a
  multi-GB zombie indefinitely. This thread captures the launching PPID up front and `os._exit`s
  when `os.getppid()` *changes* (by-change, not `== 1`, so PID-namespace subreapers count).
  Started early in `main()` inside the `if args.debug:` block — before team/env/server setup —
  so a startup hang is covered too. **Real launcher-managed runs keep a live parent and never
  arm it.** Regression test: `watchdog_test.py` (subprocess-driven orphan + no-false-fire).

## Showdown port threading (the `server_config` seam)

`train_rl_agent.py --showdown-port <port>` builds **one** `ServerConfiguration` in `main()`
via the single constructor `localhost_server_configuration(port)` (in
`poke_env.ps_client.server_configuration`) and threads it to **every** Showdown client —
the training-env players (carried into the `SubprocVecEnv` spawn workers via the env-factory
closures), eval, and self-play. Every player-creating callback takes a `server_config` param
(defaulting to port 8000 for standalone use) and builds its players from it — **never** from a
bare `LocalhostServerConfiguration` constant. `server_port_threading_test.py` is the
regression guard: it fails if any of these callbacks hardcodes the default port instead of
threading the configured one (the original bug had the now-retired replay recorder connecting
to :8000 while training ran on :8001; eval forensic traces inherit the same guard).
There is no environment variable; `train_rl_agent.py`'s own default is 8000, but the **launcher**
overrides it to 8001 before forwarding (see `src/main/launcher/CLAUDE.md`). The launcher
forwards `--showdown-port` verbatim (it strips only launcher-owned flags).
