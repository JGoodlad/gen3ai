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

**Anti-stall terminal (`--draw-penalty`, DEFAULT −35.0).** The trainee FORFEITS a
stalled battle at the turn cap (`gen3_env` `ForfeitBattleOrder` at turn ≥ `StallConfig.threshold`), so
a 250-turn stall ends as a forfeit-**loss** (`lost=True`), NOT a tie. The terminal therefore detects a
timeout by **`live.turn >= _TIMEOUT_TURN_CAP`** (synced to `StallConfig.threshold`), not by won/lost:
`if won: +30; elif finished: draw_penalty if timed_out else −30`. At the default −35.0 a stall-to-cap
is strictly worse than a clean loss, which cancels the γ=0.9999 discount pull of delaying an inevitable
−30. `--draw-penalty -30` restores the historical default (a tie scored identically to a decisive
loss); it was tuned under the additive-BIAS regime `--all-shaping-pbrs` replaces, which is why the two
defaults flipped together (owner decision 2026-08-18 — see **The reward COMPOSITION** below).
Resume-immutable, value-checked (`MODEL_CONFIG_VERSION 6→7`, `check_reward_config`).

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
≥0) and the policy never un-learns it. (It is NOT the old ① active-value readout — the no-① baseline explodes
just as much; that toggle is deleted, v88.) The fix is a **BIAS-class** term `−w·(our active HP fraction at decision time)` charged
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

**End-state PBRS — TWO switches (`--all-shaping-pbrs`, DEFAULT ON; `--stall-pbrs`, default OFF;
v14/v15).** The FINAL stage of the staged PBRS rollout: convert the last BIAS shaping to
policy-invariant telescoping potentials. Deliberately TWO switches so the stall tilt (which carries a
documented regression risk) can be A/B'd separately from everything else — which is also why only the
first of them defaults on.
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
the `_prev_phi_*` slots stay None and the four `pbrs_*` fields stay 0.0 — the byte-identical
`--no-all-shaping-pbrs` baseline (pinned by the no-op-equivalence + registry-coverage tests).
Composes with the v13 drops (orthogonal,
run after). `--no-all-shaping-pbrs` is the fallback and restores the additive objective in full.
Resume-immutable + value-checked alongside the **now-recorded `no_progress_penalty`**
(Φ_progress's weight) — `MODEL_CONFIG_VERSION` v14/v15, `check_reward_config`, no `ARCH_SIGNATURE` bump.
Tests: `reward_redesign_test.py::{TestProgressPBRS, TestHazardPBRS, TestBoostPBRS, TestOppBoostsPBRS,
TestEndStateDrops, TestAllShapingPbrsNoOpDefault}` + `snapshot_test.py` (resume-immutability + v13→v14 +
v14→v15 migration).

### The reward COMPOSITION — stated at launch, recorded in `metadata.json`

**A launch says what its reward is MADE OF.** `reward_class_composition(config)` (pure, in
`reward_manager.py`) returns the per-class ACTIVE-term census —
`{terminal, pbrs, bias, bias_terms, pbrs_terms}` — where ACTIVE means *"this config does not
structurally force the term to zero"* (it mirrors the `_fold_*_pbrs` early-returns,
`_apply_pbrs_suppression`, `_apply_bias_drops`, `_apply_progress_clock`, and the three weight-gated
terms). `format_reward_composition` renders the one line `train_rl_agent` emits at startup, to
stdout AND the launcher Events panel; the dict is written to `metadata.json` as
`reward_composition`, carried forward across saves like `cli_args`. It is duck-typed on field
names, so a recorded `ModelVersion` can be censused offline without reconstructing its config.

| config | composition |
|---|---|
| **default** | `1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)` |
| `--no-all-shaping-pbrs` | `1 TERMINAL + 2 PBRS + 26 BIAS` |
| `--stall-pbrs` (with the default) | `1 TERMINAL + 8 PBRS + 0 BIAS` — the zero-bias destination |

**Why it exists.** The v8→v9 drift (`designs/research_state/ledger.md`, 2026-08-18):
`--all-shaping-pbrs` simply stopped being passed at the fresh-generation boundary, so every
`ai_v9_*` run through gen-14 trained the 26-term additive objective while every validated `ai_v8_*`
run had trained the near-policy-invariant one. Nothing failed. Reward config is **training-only** —
no `ARCH_SIGNATURE` bump, absent from `check_compatible` — and no launch line stated the
composition, so the change was unobservable for a year. The census is the counter-measure and the
seed of the **launch-diff gate** the ledger registers: the field a new generation's resolved
command is diffed against its reference generation's.

⚠️ **The ledger's prose says "8 PBRS + 1 BIAS" (v8) and "3 PBRS + 28 BIAS" (v9); the census says
7/1 and 2/26.** The census is the measured one and the difference is definitional, not a
disagreement about the regimes: it counts terms a config can actually EMIT, where the hand-count
took the PBRS registry class size (8) — `pbrs_progress` gates on `--stall-pbrs`, which is off in
both regimes — and did not subtract the weight-gated BIAS terms (`stay_risk_tax` /
`escape_risk_bonus` at `switch_bias_weight` 0, `self_ko_penalty` at `self_ko_hp_penalty` 0). The
shape claim the ledger makes — ONE acknowledged bias term vs a couple of dozen additive ones —
holds exactly.

Pins: `src/main/reward_defaults_test.py` (both defaults, both opt-outs, both compositions, the
`RewardConfig` ↔ `ModelVersion` default agreement, and the actionable resume FATAL).

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
  bias, `DEFAULT_TRAINEE_BIAS_PROB`), `pinned` (`--trainee-team`), `pin_multi` (`--trainee-teams` — a
  SMALL FIXED SET sampled uniformly, the z-near multi-team exploiter / 1-vs-3-team A/B; `pin_str`
  mirrors `pin_strs[0]` so single-team consumers keep working, and unlike a single pin z_arch VARIES
  across the set), `pin_biased` (the future
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

## Faint attribution in the trace (`gen3_faint_attribution_v1`)

`BattleRecorder` writes one `<side>:<species>:fainted` event per faint. It detected the faint by
COUNT (`*_fainted_count` went up) and then labelled it with **`prev_ctx.*_active`** — the mon that
was active when the DECISION was made. That is the wrong mon whenever a switch resolved on the same
turn, and the trace then contradicted its own battle log two lines above:

```
we switch cloyster → jolteon
opp explosion → jolteon (now 0%)
we cloyster fainted            ← the protocol says JOLTEON fainted
```

**Measured on `ai_v9_17_tdaux_lam3_0818`: 25 of 466 turns named a mon that had not fainted.** Two
shapes produce it — WE switch and the switch-IN eats the hit; or the OPPONENT switches a mon in and
it dies the same turn (Claydol → Dugtrio, our Ice Beam KOs Dugtrio).

**The fix reads the newly-fainted species as a SET DIFFERENCE** over the two snapshots'
`*_fainted_species` — which `BattleContext` already carried, so no new state was needed. A set
difference rather than an HP transition because the second shape has no previous HP to fall from:
Dugtrio was never revealed before the turn it died on.

Two things followed from it, both of which the fuzz found rather than the design:

- **The HP-delta slot was wrong in the same way.** `our_ref` picked `prev_ctx.our_active` on a faint
  turn, so a switch-in that died had its damage read off the row of the mon that left (the recorded
  `hp_delta` read `+0%` while the switch-in went 273 → 0). It now uses the actually-fainted species.
- **ONE SIDE CAN LOSE TWO MONS IN A TURN.** An opponent mon is KO'd, its forced replacement switches
  in and dies to Spikes — both inside turn 34. The old `if delta.*_fainted:` shape could emit at
  most one event per side, so the second faint was silently unreported (1 of 36 faints in a
  4-battle fuzz). `_newly_fainted` returns a LIST and the caller emits one event per species.

**Blast radius: forensic only.** These event strings are read by the prober (the battle-log
timeline, `summary_flags`' `faint` flag) — the reward, the obs and the TurnDelta all compute faints
from their own state, so nothing in training consumed the wrong label. That is also why the fallback
is a slightly-wrong label rather than a raise: a forensic recorder must never take down a run.

**Gate: `poke_env_gaps/faint_attribution_fuzz_test.py`** (bridge, no server) — real battles with a
real `BattleRecorder`, validated against the **protocol log** (`|faint|pNa: Species`), which is the
sim's own statement and not another of our derived structures. It asserts species, side and
completeness per turn, and REPORTS its trigger coverage (`switch-in deaths`) so a clean run that
never exercised the bug says so instead of passing quietly. Measured: **123 faints / 50 switch-in
deaths / 0 mis-attributions**, and **44 mis-attributions when the fix is reverted**.

⚠️ **A protocol identifier carries the NICKNAME, not the species.** The team pool contains teams
whose nicknames are LOCALIZED species names (`Triopikeur` = Dugtrio, `Airmure` = Skarmory), which
reported 10 false failures until the harness resolved identifiers through poke-env's own
`battle.team` map. Any future protocol-vs-our-data comparison needs that map.

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster.** Eval fires every `EVAL_FREQ_STEPS` (2M steps) and plays
`EVAL_GAMES` (100) games per opponent — overridable per run with `--eval-games N` (threaded to both
callbacks via the `_schedule()` seam; n=100 → ±0.098 per-cell 95% CI, n=200 → ±0.069; the recorded
`n_games` tracks the actual cycle size) — one cadence, one game count, applied uniformly to
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
`RLPlayer._decode_belief` → `inference/belief_decode`, see `src/agents/model/CLAUDE.md` — and an
**`opp_intent`** block, the v67 `α`/`β` heads' read of what the OPPONENT was about to do: `α` a ranked
list of NAMED believed moves plus `SWITCH`, `β` the candidate switch-ins each named by the model's own
species posterior. Present only under `--opp-intent-coef>0`, so an intent-off run's trace is unchanged;
it is what the prober's `EXPECT` line and the web replay's per-turn *expect* line read) +
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
On a BRIDGE run each trace also gets a fourth sibling,
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
- **The hung-cycle watchdog is CONTENTION-SCALED** (`eval_cycle_timeout()` =
  `scale_timeout(_EVAL_CYCLE_TIMEOUT_SEC)`, 30 min baseline, shared by BOTH callbacks;
  `gen3_contention_robust_timeouts_v1`). Eval is the path most exposed to load — it runs
  concurrently with training *by design*, so it is contended 100% of the time, and the bullet
  above already concedes "an eval can outlast its interval". Firing early does **not** merely
  lose a cycle: `_abort_pending_cycle` kills the workers and collects **PARTIAL** results, which
  feed `win_rate_vs_bots` (the curriculum ramp), `win_rate_vs_pool` (the promotion gate) and the
  ELO fit — and the survivors are whichever shards got scheduled, not a random subsample. So a
  merely-slow cycle must never be mistaken for a hung one. The partial-coverage warning no longer
  asserts "worker crash mid-opponent" as the cause either (an overrun-kill produces an identical
  shortfall); it states the fact and appends `describe_contention()` so the reader can tell which
  happened. ⚠️ **Tests must read `eval_cycle_timeout()`, never the raw constant** — the two
  hung-cycle tests built a past timestamp from `_EVAL_CYCLE_TIMEOUT_SEC` and so passed on an idle
  box while failing on a loaded one; `GEN3AI_TIMEOUT_SCALE=6 pytest src/ -m "not integration and
  not e2e"` is the check that catches that class.
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
- **The cadence ANCHOR is restored on resume — CLAMPED to the current step**
  (`_ForcedEvalMixin._restore_last_eval_step`, shared by BOTH callbacks). `_last_eval_step` is
  in-memory and resets each process, so it is restored from metadata; otherwise the resumed step
  sits far past a boundary and a fresh `0` would eval on step 1. That is right for a launcher
  RESTART and **wrong for a FORK**: `resume_eval_metadata` is the SOURCE run's run-level
  `metadata.json`, whose `latest_eval.step` is where *that* run last evaluated, not the step of the
  older checkpoint being forked from. Measured 2026-08-21 on an exploiter fork of gen-17's
  9,084,672-step checkpoint out of a run that reached 25M: the anchor restored to **24,000,000**,
  and since the cadence test is `(now // freq) > (anchor // freq)`, the fork would have launched
  **ZERO eval cycles** until it itself reached 26M — no `win_rate_vs_*`, no `eval_results.jsonl`
  row, no ELO. A gate arm whose verdict IS an eval metric silently produces nothing to read.
  Clamping to the model's `num_timesteps` restores the intended meaning and the next boundary after
  the fork point fires normally; a restart is unaffected (its recorded step is at or behind the
  loaded one, so the clamp never bites) and the clamp announces itself with an `anchor is AHEAD`
  event that states the FACT rather than asserting a cause — a crash-restart that rewound past a
  completed eval reads identically to a fork.
  ⚠️ It reads **`self.model.num_timesteps`**, not `BaseCallback.num_timesteps` — the latter is a
  mirror SB3 only syncs inside `_on_step`, so at `_init_callback` time it is still `0` even on a 9M
  resume, and reading it would clamp every restart to 0 and re-eval on step 1 (observed live before
  the fix, as `this model is at 0`). Same family as `_warn_if_fork_pool_empty`: a fork inherits the
  base's weights but none of its run-directory state, and the silent failures live in that gap.
  Test: `eval_fork_cadence_test.py` (both callbacks, parametrized).

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `5` | Eval subprocesses per cycle; work-steal **shard units** from a shared pool. Capped at the unit count (≈ opponents × shards-per-opponent, so sharding lets the full pool help). Self-play doubles this (→ `10`) since sentinel matchups run the model for both players. |
| `--eval-games` | `None` (=`EVAL_GAMES`, 100) | Games per **opponent** per eval cycle. Raise for tighter sentinel/promotion CIs (200 → ±0.069) at proportionally more eval compute — work-stolen across the workers, off the training path. Shards/opponent = eval-games / `--eval-shard-games`. |
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

  **REVIVAL VERIFICATION (2026-08-18) — it SURVIVED; nothing needed repair.** PFSP was built
  ai_v8-era and never production-enabled, so gen-16 wanting it ON required checking whether code
  that no test-suite failure would have protected still worked across the fresh-generation reset,
  the frame deletion and two signature bumps. It did: **70/70 existing tests green unmodified**, and
  every call site is intact — both `SnapshotPool` constructions (env-worker + trainer-side), the
  `_update_pfsp_ema` fold in `_collect_pending`, the `_prune_and_push_pfsp` env push, the
  `summary.json` `pfsp_win_rates` resume-load, and `MaskableAgentWrapper.set_opponent_win_rates`.
  A `--debug --self-play --debug-eval --pfsp-scale 2.0 --pool-spread` CPU smoke ran to
  `Training complete` (exit 0). **What that smoke does NOT show, and why it can't:** pool seeding is
  gated on `win_rate_vs_bots >= SELF_PLAY_START` (0.55) and a fresh debug model sits at ~4%, so the
  pool stays empty and PFSP never weights anything — the smoke proves the flags launch and thread,
  not that they skew.
  **The gap the revival actually closed was in the TESTS, not the code.** Every pre-existing test
  exercised ONE link with the other side mocked (pool math / callback EMA / wrapper forwarding), so
  a green suite said nothing about the composition — the thing a revival has to prove. Two
  end-to-end tests now run measured win-rates through callback → `env_method` → wrapper →
  `SnapshotPool` → `sample()`: `test_measured_winrates_skew_real_sampling_end_to_end` asserts the
  empirical 40k-draw distribution matches the analytic weights (at `pfsp_scale=2.0`, win-rates
  0.1/0.5/0.9 and recency off ⇒ factors 2.8/2.0/1.2 ⇒ shares **0.467 / 0.333 / 0.200**, and the
  self we lose to is drawn **2.33×** as often as the one we dominate), and
  `test_pfsp_off_makes_no_push_and_no_skew_end_to_end` asserts the same composition at
  `pfsp_scale=0` makes **no IPC call at all** and leaves the draw uniform under the same win-rates.

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
  exploiter, not a single-team specialist). The shipped TSS pin IS a sample team, so it passes. **A
  multi-team exploiter (`--trainee-teams` → `pin_multi`) validates EVERY member likewise** (each of the
  N teams must be a sample). Tests: `matchup_spec_test.py::test_exploiter_*sample*` +
  `::test_pin_multi_*` + the e2e FATAL.
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
Four modes: **`off`** (default → byte-identical), **`measure`** (TRACK + persist the per-team
self-play win-rate WITHOUT biasing sampling — pure observability), **`var`** (measure + bias,
symmetric variance), **`onesided`** (measure + bias, losing side held at MAX).

- **Variance weighting + cap + floor.** For pool team `i` the weight is `raw_i = --team-pfsp-floor +
  w(p_i)` where `p_i` is the team's self-play win-rate EMA (seed 0.5 → an unmeasured team gets the
  MAX weight → explored), then capped `w_i = min(raw_i, --team-pfsp-cap·mean(raw))` (no team is
  sampled more than `cap`× the uniform share — the over-representation bound). **`var`**: `w(p) =
  p·(1−p)` — peaks at 50% and decays to the floor at BOTH extremes, so it self-ignores both the teams
  we crush AND the truly-lost teams. **`onesided`** (owner-requested, the z_arch/FiLM companion):
  `w(p) = 0.25 for p < 0.5, else p·(1−p)` (continuous at 0.5) — every sub-50% team stays MAXIMALLY
  sampled and only mastery retires a team, because under the conditioning hypothesis the weak-team
  tail is exactly the learnable headroom (the amortization gap): "truly lost" is the claim under
  test, not a sampling prior to bake in. The floor keeps nothing fully starved either way.
  `compute_team_pfsp_weights` is the pure, unit-tested math.
- **Team-blocked episodes (`--team-block-episodes`, default 1 = off, byte-identical).** Each env
  holds its drawn TRAINEE team for N consecutive episodes before redrawing
  (`Gen3Teambuilder.set_block_episodes`; the WHOLE draw is held — bias branch, PFSP weights,
  tracking index — so weights apply at redraw and outcomes attribute to the blocked team for the
  whole block; each SubprocVecEnv worker unpickles its own builder copy ⇒ blocks are per-env). The
  per-team gradient-DENSITY counter to the sample starvation the retired FiLM group measured
  (`film/noise_scale` ran ≈ 8–9× the batch before the v78 zarch deletion took that metric with it;
  the DENSITY argument stands on its own): per-episode redraw gives ~700 teams × ~4 episodes
  (~140 decisions) per rollout;
  at ~64 (≈ `n_steps`/ep_len — the phase-transition value) each env carries ONE team per rollout at
  ~2k decisions (~15× density) AND the block spans an update boundary, so the env replays the team
  right after its gradient landed (the mini-exploiter learn-and-retest loop — the piece of the
  exploiter regime per-episode redraw never provides). Acceptance: the fixed-matchup ablation
  probe's intact-vs-ablated gap widening. Trainee side
  only (opponent draws stay per-episode); training-only, NOT version-locked, resume-forwarded.
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

## Per-team win-rate tracking (`--team-wr-tracking`, DEFAULT ON, `team_winrate_callback.py`)

A first-class running record of how the trainee does **piloting each team**, keyed by `team_sha`.
The training loop always knew which team an episode piloted and how it ended; nothing kept the
record, so the three flywheel consumers that need it — the deficit thermostat, **headroom
capture's denominator**, and slice-curation evidence — each had to be a scratch script. This is
**instrumentation only: no prioritization consumer ships with it**, by design.

⚠️ **THE CONFOUND, and it is written into the artifact rather than only into this file.** A raw
per-team win rate conflates **PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP finding:
team-PFSP win rate was confounded by team strength). "Our win rate with team T is low" does not
mean "we pilot T badly". Anything that spends budget on this signal must first normalize against a
**team-strength baseline** — e.g. T's pool-average win rate under a reference pilot. The artifact
carries that sentence in its `notes` field so it travels with the numbers, plus the reminder to
read `by_class`: a pre-self-play curriculum phase is ~all `bot` episodes, where every team reads
~0.99.

- **The seam is an `env_method` PULL, not an info-dict thread — and that is the async decision.**
  Each worker's `Gen3Teambuilder` accumulates a windowed per-team, per-opponent-class count
  (`record_team_wr_outcome`), fed by `MaskableAgentWrapper._maybe_record_team_wr` at the terminal
  step beside the existing `win_outcome` capture; `TeamWinRateCallback._on_rollout_end` drains
  every worker (`drain_team_wr_counts`) at a rollout boundary. **This works identically under
  `SubprocVecEnv` and `--async-rollout`** because `AsyncSubprocVecEnv.env_method` is drain-safe (it
  stashes in-flight step results before the barrier RPC), whereas an info-dict route would have to
  know which buffer ROW a terminal landed on — knowledge only the async collector has, which is why
  the team-PFSP precedent avoided that route for the same reason.
  `test_aggregation_reads_env_method_and_never_the_info_dicts` pins it by feeding the callback a
  deliberately contradictory `self.locals["infos"]` and asserting the result ignores it.
- **The default uniform draw stays RNG-identical.** With `--team-pfsp off` (the default)
  `_draw_team` is `random.choice(self.packed_teams)`, which returns the team and not its index. The
  index is recovered by a **reverse dict lookup** (`_pool_index_by_packed`, built at construction),
  never by re-drawing it — so the byte-identity baseline is untouched
  (`test_default_uniform_draw_is_rng_identical_with_tracking`). Side effect worth knowing:
  `--team-block-episodes` caches `_last_pool_idx` for the block, which on the default path used to
  be `None`, so a blocked default run can now attribute its whole block to the team it held.
- **Stratified by opponent class** (`MaskableAgentWrapper.OPP_CLASS_*` / `OPP_CLASS_NAMES`), so a
  rate can always be split back out by who it was measured against. A bias/distill-pinned yield
  (`_last_pool_idx is None`) is never attributed to a pool team.
- **NO TensorBoard emission — owner rule** (design_flywheel_tick_tock.md §6b: per-team series
  would be noisy spam; "let's not spam it if the data won't be nice"). Pinned by
  `test_NOTHING_is_emitted_to_tensorboard` — a future "just one scalar" regression fails there.
- **The table rides `metadata.json`** as the top-level `team_win_rates` block (written via
  `snapshot.record_team_win_rates`, carried forward across checkpoints by `save_model_snapshot`
  exactly like `latest_eval` — one artifact per run holding per-team AND per-opponent records
  side by side):
  `{step, updated_at, n_teams_seen, n_games, opp_classes, notes, teams: {sha: {n, wins, wr,
  archetype, by_class}}}`. **RAW COUNTS, not a smoothed rate** — headroom capture needs a
  denominator, which is exactly what team-PFSP's EMA throws away. `archetype` is joined via
  `load_team_archetypes` on the same `team_sha`. **Restart-safe by load-and-continue**, and keyed
  by sha rather than pool index so a pool that was reordered or resized between runs still joins
  (`test_reload_is_keyed_by_sha_so_a_REORDERED_pool_still_joins`). A corrupt file starts fresh.
- **GIGO guard, throwing.** Counts arrive per pool INDEX and are keyed to a sha by the worker's own
  key list; if any worker's list disagrees the callback **raises**. Same pool SIZE is not the same
  pool ORDER, and a diverged order would attribute every per-team number to the wrong team.
- **Deliberately NOT coupled to `--team-pfsp`, and the overlap is real enough to state.**
  `--team-pfsp measure` also tracks a per-team win rate and also writes an archetype-joined
  `team_winrates.json`. Four differences make it unusable as this instrument: it is **off by
  default**; it measures **self-play POOL battles only** (bots wash out its weighting signal), so a
  pre-self-play generation records nothing; it keys per pool **INDEX** with the sha only for an
  audit line; and it stores an **EMA rate**, not counts. The two share the builder's "which team
  did I just yield" draw index (`_last_pool_idx`) and **nothing else** — separate counter tables,
  separate accessors, separate artifacts, deliberately differently-named files
  (`team_win_rates.json` vs `team_winrates.json`). If the owner later wants one tracker,
  consolidating team-PFSP's `measure` mode onto this table is the direction, not the reverse.
- **Flag class: training-runtime, like `--team-pfsp`.** Never reaches the extractor, scales no loss,
  changes no weight shape ⇒ **no `ARCH_SIGNATURE` bump, not in `model_config.json`/`ModelVersion`,
  not in `check_compatible`, and deliberately not in `agents/model/flag_registry.py`** (that
  registry's scope is extractor architecture toggles — the `--td-aux-coef` /
  `--intent-label-bot-weight` precedent, which are recorded on `ModelVersion` only because they
  scale a loss and want flagless-resume inheritance; this one does neither). Forwarded verbatim by
  the launcher like any non-launcher flag. `--no-team-wr-tracking` opts out (no callback, no
  `env_method`, the wrapper hook returns immediately).
- **Verified end to end** by a `--debug --steps 4000` CPU smoke: **96 teams / 103 games** recorded,
  archetypes joined (`semi_stall`, `balance`, `hyper_offense`), `by_class` correctly all-`bot` on a
  fresh run, the `notes` caveat present, and `teams/n_teams_seen` / `teams/n_games` on the TB
  event file. The four `wr_*` scalars need a team past the 10-game floor, which a 719-team uniform
  pool does not reach in 4000 steps — a `--trainee-team`-pinned smoke exercises them, and all six
  keys are pinned numerically by `test_sparse_tb_keys_are_summaries_not_one_series_per_team`.
- **Tests.** `team_winrate_callback_test.py` (29): the `team_sha` convention agreement with
  `team_archetypes.team_sha` incl. strip-normalization, the RNG-identity claim, the builder
  accumulator + drain-zeroing + bias-yield exclusion + PFSP-table independence, the wrapper hook and
  its off path, the callback's running math across workers AND windows, per-class restriction,
  `min_games`, the `update_every` throttle, None-worker filtering, the throwing order guard, the TB
  key set with hand-computed values, the artifact shape + confound note, the archetype join and its
  missing-artifact fallback, restart reload incl. the reordered-pool case and a corrupt file, and
  the `env_method`-not-infos seam claim. Plus `utils/teambuilder_test.py` (the off-path RNG identity
  now also asserts the index resolves while PFSP still ignores it).

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
- **The RAW matrix, at higher resolution.** `python -m agents.training.bot_matchup_matrix`
  accumulates the same round-robin (same bots, same team sampling, same bridge driver — it calls
  the calibration's own `_build_bot`/`_play_chunk`) as **raw per-pair `wins_a`/`wins_b`/`draws`/`n`**
  toward 10 000 games/pair in resumable chunks → `data/gen3_bot_matchups.json`. Draws stay
  separate and it **never writes the anchor** (regenerating that is an owner decision).
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

### Frozen-snapshot ELO ladder — the dense, pay-once resolution (`snapshot_ladder.py`)

The live ELO above is RESOLUTION-limited at the frontier: the fixed bots have SATURATED (we sit
~400 Elo above them, out on the flat tail of the logistic — a 10-Elo trainee move shifts its
bot-WR by ~0.5% against a 1.9%/200-game noise floor), so the bots pin the absolute LEVEL but the
fine ordering rides on the sparse, near-50% sentinel edges (±15 Elo CIs). Fix from the other side:
a promoted snapshot is FROZEN, so snapshot-A-vs-snapshot-B is a STATIONARY Bernoulli — measure it
ONCE (dense round-robin) and it is permanent. On each promotion, `SelfPlayCallback._spawn_snapshot_ladder_update`
fires a **DETACHED** `python -m agents.training.snapshot_ladder <run> --promote <step>` subprocess
(bridge, off the training path) that plays the new frozen node vs the current frozen pool
(`--snapshot-ladder-games`, default 100/pair; 0 disables) and appends to
`<run>/snapshot_ladder/games.jsonl` (**forever, race-safe line appends; a measured pair is NEVER
replayed**). `fit_ladder` combines that dense frozen-vs-frozen matrix with each snapshot's
historical bot edges (from `eval_results.jsonl` — the anchor connection) → an anchored BT fit
(`fit_pairwise`, bots pinned) written to `<run>/snapshot_ladder/ladder.json` (the sidecar metric);
`_record_ladder_elo` surfaces the latest promoted node's rating as `eval/ladder_elo` (+`_ci`) on
TB/TUI — the high-resolution counterpart to the saturated `eval/elo`. Snapshots load via
`load_foreign_opponent` (their own saved config → PopArt/toggles honored, `check_compatible`
skipped). `--backfill` pays the one-time back tax over the whole current pool (idempotent — skips
measured pairs); `--fit-only` refits without playing. `ladder.json.fit_quality.mean_abs_err`
QUANTIFIES non-transitivity (a scalar Elo is lossy if the pool is rock-paper-scissors — the dense
matrix at least measures it). Tests: `snapshot_ladder_test.py` (store accumulation/symmetry,
measure-once contract, fit-recovers-ordering, sidecar read).

### Hodge decomposition — the SPINE and the WIDTH (`hodge.py`)

A scalar rating is a **transitive** model by construction, so a BT fit cannot see a cycle: two
snapshots with identical ELO can have a lopsided head-to-head. `ladder.json`'s
`fit_quality.mean_abs_err` NOTICES the residue but reports it as one unitless number with **no
noise floor**, which cannot answer the only question that matters — *is the non-transitivity real,
or is it binomial noise on 100-game edges?* HodgeRank answers it by splitting the measured flows

```
Y_ij = logit(p_ij)   =   (r_i − r_j)   +   R_ij        w_ij = n_ij·p_ij·(1−p_ij)
                          ───────────       ────
                          TRANSITIVE        CYCLIC     (Fisher info of a logit = the weight)
```

where `r` is the weighted-least-squares (graph-Laplacian) solve — BT's quadratic cousin, reported
BESIDE the BT ratings so the estimators' disagreement is visible. The split is **exactly
w-orthogonal** (`Σw·Y² = Σw·(rᵢ−rⱼ)² + Σw·R²`, pinned by a test), so spine and width cannot be
traded against each other by refitting. Units: 1 logit = 400/ln10 ≈ **173.72 ELO**.

**The noise floor is the whole instrument.** Two nulls, both reported: an exact-mean analytic one
(`E[Σw·R²] = Σ(1 − w_e·Reff_e)` — per-edge effective resistance, i.e. Foster's `E−V+C` spread over
edges) and a **parametric bootstrap** that simulates games from the fitted transitive model and
re-runs the whole pipeline. `width_rms_excess = √(raw² − null²)` is the width that survives, with a
p-value for "width > noise".

**Width SCOPE — a pendant edge's residual is identically zero.** A player with one measured
opponent has that single edge as its whole normal equation, so counting it only inflates Σw and
deflates the RMS. Width statistics therefore default to the **triangle-supported subgraph**; the
spine is always fit over every edge. `n_triangles` + `n_width_edges` ride with every read.

- **Offline — THE instrument.** `python -m main.elo <run>` prints the block and writes it into
  `elo/elo_ratings.json` under `hodge` (flags: `--no-hodge`, `--hodge-bootstrap N`, `--hodge-seed`,
  `--hodge-with-bot-rr`). The graph is exactly `fit_ladder`'s: the dense frozen matrix
  (`snapshot_ladder/games.jsonl`) + every cycle's bot/sentinel edges. The static bot round-robin is
  **excluded by default** — its 36 edges carry ~2700 games each against a ladder edge's 100, so on
  the Fisher weighting they would carry ~99% of Σw and the "width" would become a property of the
  immutable shared anchor rather than of this run. `main.endofrun`'s §1 block carries the same read
  for the run and its `--ref`.
- **Live — two scalars beside `eval/elo`**, recorded by `record_elo` on the same cadence:
  `eval/hodge_width_elo` (excess width, ELO) and `eval/hodge_cyclic_fraction` (null-adjusted).
  Both also ride in the `eval_results.jsonl` row's `hodge` block for offline replotting, and a
  cycle whose graph had **no testable triangle** writes `recorded: false` + a reason there and
  records NOTHING to TB (never 0-as-a-stand-in, never NaN — a missing point and a suppressed one
  look identical in TensorBoard, and only one is a fact about the graph).

⚠️ **THE STAR-GRAPH SUBTLETY — read this before quoting a live width.** A cycle's own new games are
a **star** (trainee vs each opponent). A star is a tree; a tree has no cycles; so a width computed
on the cycle's games alone is *identically zero* — a fake instrument that would read "no
non-transitivity" forever. The triangles come from joining the trainee's edges to the **static
bot-vs-bot round-robin** in `data/gen3_bot_elo_anchors.json` (which does ship the raw 9×9
`win_matrix` + per-pair `pair_games`, so those are MEASURED edges; a future anchor carrying only
`ratings` falls back to edges reconstructed from them, which are transitive by construction and act
purely as a pinning prior — flagged in `caveats` when it happens). So the live metric means exactly
**"the trainee's matchup deviation from its own rating, over trainee×bot×bot triangles"** — nothing
about the pool's width. Sentinel edges are in the FIT (real spine information) but on no triangle,
so they are excluded from the width scope. And the live read is **weak by construction**: ~100-game
edges put the noise floor around 35-60 ELO, so only a gross cyclic profile clears it (measured on
gen-15's 12 cycles: p between 0.13 and 0.93, i.e. never significant on its own). **The offline
dense-ladder read is the real instrument; the ELO-reading rules below apply unchanged — never
narrate a mid-run width.**

**First reading (gen-15, `ai_v9_18_gen15_v8rewards_0818`, 21 players / 174 edges / 814 triangles,
300 bootstrap reps):** spine 939 ELO, width raw 58 → null 36 → **excess 46 ELO, p = 0.005**; cyclic
energy 6.3% raw / 3.8% null-adjusted; **3 significant 3-cycles**, all snapshot-vs-snapshot
(16M > 20M > 18M > 16M, curl +217 ELO z=4.3; 8M > 20M > 18M; 8M > 20M > 14M). gen-14 on the same
read (same 21/174/814 shape): spine 765, excess **26 ELO, p = 0.0033**, 2.2% null-adjusted, and **0
individually-significant cycles** — its width is real but diffuse. So both ladders are
overwhelmingly spine (96-98%) and both carry cycle content that is **not sampling noise** — the
first evidence here that the BT gate is a lossy projection by a *measured* amount, and that the
loss is bigger on gen-15. (p ≈ 0.003-0.005 is the bootstrap's floor `1/(B+1)`, not a coincidence:
no null replicate reached the observed width.) Tests: `hodge_test.py`.

#### 🚨 Reading an ELO: `ladder.json`, at matched SNAPSHOT COUNT, never mid-run

**Read `<run>/snapshot_ladder/ladder.json` — not `eval/elo`, not the per-cycle TB scalar.** On
gen-10's completed ladder the two agree at the end (24M: dense 2079 vs sparse 2102) but the dense
CI is **±10 vs ±29**. Precision is the reason to prefer it; it is *not* immune to the drift below.

**A snapshot's rating keeps moving until it stops gaining opponents.** Anchored BT is a GLOBAL
BATCH fit — every added player re-solves every rating — and the movement is a **systematic
downward bias on the newest node**, not noise. Measured over gen-10's 12 successive refits
(`snapshot_ladder/updater.log` records each one):

| snapshot | first fit | final fit (n=12) | drift |
|---|---|---|---|
| 2M | 1790 | 1705 | **−85** |
| 4M | 1945 | 1844 | **−101** |
| 12M | 2089 | 2021 | **−68** |
| 14M | 2088 | 2044 | **−44** |

Mechanism, and the SE is the tell (12M: 25.9 → 18.4 as it fell 2089 → 2021): a fresh snapshot's
only edges are ~90% wins over the bots. A saturated edge says *"≥380 Elo above"* with a likelihood
that is **flat upward**, so the MLE is inflated and under-constrained. The near-50% frozen-vs-frozen
edges are sharply informative, and as they accumulate they pull the chain down onto the anchor.
Dense measurement buys resolution; it does not buy an early answer.

**Therefore, two rules:**

1. **Never narrate a mid-run ELO or a mid-run delta.** The gen-10 12M delta read +108, +82, +73,
   +64 before settling at **+11** — four reported "results", all artifacts. Wait for the run to end.
2. **Cross-run comparison must be at matched snapshot count `n`, not matched step.** Both runs'
   node at n=k carries the same inflation, so it cancels; a live run's newest node against a
   finished run's *final* value does not. Worked example — gen-11 at n=7 vs gen-10's **n=7** fit
   (recoverable from `updater.log`) reads 14M: 2082 vs 2088 = **−6, tied**; against gen-10's n=12
   final it reads **+38**, which is the drift and nothing else.

`n_frozen_pairs_measured` / `n_pairs_possible` in `ladder.json` is the completeness check — a fit
at 21/21 pairs is internally complete but only 7 nodes deep, and depth is what the bias tracks.

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

## Compiled CPU opponents (`--compile-opponents`, DEFAULT ON) + BLAS thread pinning

> **Two independent compile flags, split by WHO and WHERE** (renamed 2026-08-14 from the
> single `--compile-extractor`, which said neither): **`--compile-opponents`** is the
> CPU/ROLLOUT half documented in this section — frozen opponents in the env workers.
> **`--compile-trainer`** is the GPU/LEARNER half, documented below. They are orthogonal;
> a run can take either, both, or neither.

**`--compile-opponents`** `torch.compile`s each frozen OPPONENT's feature extractor in the env workers
(pool / stable / exploiter loads, via `agents.model.snapshot.maybe_compile_extractor`). It is a
**runtime PERF knob** — never versioned, never in `check_compatible`.

🚨 **DEFAULT ON since 2026-08-17 (owner decision: the compile flags are FALLBACKS, not opt-ins).**
`--no-compile-opponents` is the way back to eager, and it takes `--compile-opponents-preload` with it
(the preload FOLLOWS this flag, so one flag turns the whole path off). `--compile-opponents-strict`
stays opt-in — default-ON is about the compile, not about the failure mode, and warn-and-fall-back
IS the fallback the default wants. **"Not inherited on resume" now cuts the other way**: a flagless
resume gets the compile ON, so it is the opt-out you must re-pass, not the flag. Pinned by
`src/main/compile_defaults_test.py` (defaults + opt-outs by value) and
`src/main/launcher/compile_flag_forwarding_test.py` (the launcher forwards all of them and owns no
default of its own).

**Why it works now when it didn't in June.** The 2026-06-30 attempt compiled only
`DamageOperator.forward` inside `policy.get_distribution` and measured **0.70× (slower)** — dynamo
overhead around a graph still running ~10k eager dispatches. Compiling the WHOLE extractor gives one
fused graph: `torch._dynamo.explain` reports **0 graph breaks / 1 graph**, and B=1 CPU
`get_distribution` goes **4.84 → 0.91 ms (5.3×)** on a real checkpoint, logits within 9.5e-7 of eager
with **0/16 argmax flips**.

**`suppress_errors` is GONE — the crash it hid was ONE op (2026-08-03).** The helper used to set
`torch._dynamo.config.suppress_errors = True` globally, because the expected-latent-defender read
(`BeliefHead.species_posterior`, then reached via `--threat-unrevealed-outgoing`) crashed the
Inductor CPU backend (`AssertionError: buf307`). That made the LITERAL production config
compile only PARTIALLY — dynamo falling back to eager per FRAME for the failing region, measured
6.48 → 1.78 ms = 3.6× — and made every OTHER backend failure in the process silent too.
`tmp/inductor_crash_repro.py` narrowed it to a single op: the softmax over species logits in the
expected-latent-defender read, which lowers to a `[B,6,n_species]` numerator + a `[B,6,1]` denominator
that the CPU scheduler asserts on while fusing. `BeliefHead.species_posterior` now spells the identical
math as `log_softmax(...).exp()`, which lowers cleanly. **The literal production arch now compiles
WHOLE with suppression OFF: 6.371 → 0.976 ms = 6.53×**, 1 graph / 0 breaks, max|Δ| vs eager 5.07e-07 —
nearly double the per-forward win, and backend failures are loud again. `tmp/softmax_variant_probe.py`
records that `.contiguous()`, `.clone()`, a 2-D reshape and a hand-rolled `exp / sum` **all still
FAIL**, so the spelling is load-bearing; it is pinned by `extractor_compiles_test.py` (default-ON,
a real compile — `GEN3AI_SKIP_COMPILE_TESTS=1` opts out; verified to FAIL if the old spelling returns).

**Measured end-to-end on the LITERAL production arch (`tmp/literal_arch_ab.sh`, 2026-08-03):**
marginal FPS **406.5 -> 541.8 = +33.3%** at `--n-envs 48`, 4 samples per arm, **ranges disjoint**
(off max 417 < on min 512), 48/48 workers compiled, 0 reverts. This arm is the one the earlier A/B
could not run: the `species_posterior` softmax used to crash Inductor, so that measurement had to
drop the expected-latent read plus the between-layers refine loop.

**READ THE TWO NUMBERS TOGETHER — the per-forward win has SATURATED.** Fixing the softmax doubled the
per-forward speedup (3.6x -> 6.53x), but end-to-end moved only 31.0% -> 33.3% (and those are
different arches, so even that 2.3pt is generous). Amdahl: the opponent forward is no longer the
rollout bottleneck. Whatever is left — obs build, protocol parse, bridge wait, the PPO update — now
dominates, so further compiler work on this path is spent effort. The next throughput lever has to
come from a different stage.

**Prior measurement, reduced arch** at the production `--n-envs 48` shape with
`--async-rollout --grad-checkpointing --self-play --self-play-use-cpu` against a seeded pool, marginal
FPS **498 → 653 = +31.0%**, 6 samples per arm, **ranges disjoint** (off max 512 < on min 614). It is
the first throughput lever here that the `SubprocVecEnv` barrier does NOT absorb — the win is *larger*
at 48 envs than at 8 (+26.6%). Adversarial checks: the compiled path is the real `pool:snapshot_*`
opponent, and `ep_len_mean` is unchanged (47.4 vs 45.9), so it is not an artifact of shorter battles.

Three properties make it cheap: the compile is keyed on the CODE OBJECT (a second extractor instance
in the same process compiles in **0.00 s**, so pool promotions are free), parameters are graph INPUTS
(a different checkpoint's `load_state_dict` does NOT recompile), and a shared
`TORCHINDUCTOR_CACHE_DIR` turns each worker's cold codegen into a cache hit.

Four guards, each protecting against a failure that actually happened while building it:
- **CUDA-context OOM.** Compiling even a CPU model in a CUDA-visible process initialises CUDA and takes
  ~252 MiB of card; ×48 workers is the June OOM. The helper sets `CUDA_VISIBLE_DEVICES=""` — but only
  when the caller passes **`hide_cuda=True`**. That used to be INFERRED from
  `torch.cuda.is_initialized()` as a proxy for "am I an env worker", which was correct only by accident
  of the call sites: the first main-process caller would have silently blinded the learner's GPU. It is
  now the caller's explicit declaration (all three training sites are env workers → `True`), and a
  caller that declares `hide_cuda=True` in a process that already holds a context is REFUSED rather
  than quietly compiled. Verified live: 48 compiled workers, exactly ONE context (the learner).
- **A compile that LOSES.** June measured 0.70× (dynamo overhead > fusion win on a fragmented graph),
  so the helper **times eager vs compiled at load and REVERTS** below a 1.05× floor. This used to be
  load-bearing because `suppress_errors` made a failed compile silent; with suppression gone a failure
  raises and is caught, and this is now a second line of defence against a merely-fragmented graph.
- **A LATE failure.** `torch.compile` guards on input properties, so an unseen shape can trigger a
  fresh trace at CALL time, long after load. `_eager_fallback_on_error` wraps the compiled callable so
  that degrades THIS opponent to eager (and says so) instead of killing a 3-hour run. This is the
  scoped replacement for global `suppress_errors`: same never-crash property, one model, and loud.
- **Resume safety.** It patches the BOUND `fe.forward`, never the module — `torch.compile(module)`
  would prefix every state_dict key with `_orig_mod.`. It also calls
  `Gen3FeaturesExtractor.disable_observation_debugger()` (a method, not a reach-in assignment to
  `fe._debugger`), because the debugger's numpy asserts inside `forward` make dynamo die creating a
  guard.

**Per-worker startup cost, measured (`tmp/compile_spawn_cost.py`, 16 workers, 16-core box).** Wall
clock until all workers are ready: **private cache per worker 163.4 s / cold shared cache 59.6 s /
warm shared cache 30.1 s.** So `TORCHINDUCTOR_CACHE_DIR` is not a nicety — without it the startup cost
nearly triples. The residual ~30 s is dynamo tracing + guard construction, which the on-disk cache
cannot remove, and it is paid once per launcher restart (every 3 h).

**Warm the Inductor cache in the parent — `agents.model.compile_prewarm` (BUILT).** Each env worker
compiles its own frozen opponent. `train_rl_agent` calls `prewarm_extractor_compile(...)` before the
vec env exists, so the workers hit a WARM shared on-disk cache instead of racing on a cold one:
**59.6 s -> 30.1 s** wall for 16 workers (`tmp/compile_spawn_cost.py`; a private cache per worker is
163.4 s, so `TORCHINDUCTOR_CACHE_DIR` is load-bearing). It builds the extractor from
`build_extractor_arch_kwargs(args)` — the same table the real model uses — so the cached codegen is
keyed to the graph the workers actually run; weights are graph INPUTS, not baked constants, so a
fresh random extractor warms the cache for every opponent checkpoint.

**THE FORKSERVER PRELOAD WORKS NOW (`--compile-opponents-preload`, `gen3_forkserver_preload_v1`,
2026-08-16) — and the fix was one level deeper than the plan.** SB3's `SubprocVecEnv` uses
`mp.get_context("forkserver")`, and a forkserver child inherits memory copy-on-write, so
`agents.model.compile_preload` (armed via `set_forkserver_preload`) compiles the extractor ONCE in
the forkserver and every worker inherits the traced graph (~0.12 s vs ~30 s per worker). The 2026-08
attempt at exactly this **wedged a real 48-env run** — 2 workers forked instead of 48, parent blocked
in `unix_stream_data_wait`, box at 0.2 load, no error anywhere — because `fork()` copies every mutex
but only the calling thread, and importing the extractor started poke-env's GLOBAL asyncio loop
thread: any `poke_env.x` import executed the eager package `__init__` → `player` → `ps_client` →
`concurrency`. The planned fix was a ~12-file model-layer refactor; the shipped fix is at the ROOT
instead — **`poke_env/__init__.py`, `poke_env/player/__init__.py` and `poke_env/battle/__init__.py`
are LAZY (PEP 562)**, so the enum/data/battle subtrees the extractor needs are thread-free, the
public surface is unchanged, and the loop thread starts exactly when a player/client module is
imported (what every training-side consumer does anyway). The laziness also dissolved an
order-dependent `battle ↔ player.battle_order` circular import the eager inits had been masking.

Three guards, all loud:
- `compile_prewarm.extractor_import_is_fork_safe()` is the executable invariant (import ⇒
  single-threaded), pinned by `compile_prewarm_test.py` — if the lazy init regresses, the suite
  fails before any run arms the preload.
- The preload pins `torch._inductor.config.compile_threads = 1` (the codegen pool never exists) and
  calls `shutdown_compile_workers()` anyway.
- After its compile the preload asserts `threading.active_count() == 1` and **RAISES otherwise**,
  killing the forkserver bootstrap so `SubprocVecEnv` construction fails with a traceback in the
  parent — the silent wedge is unrepresentable, not just unlikely.

Proven live 2026-08-16: a real 4-worker `SubprocVecEnv` CPU run with the preload armed compiled once
(41 s), forked all workers, trained to completion. When armed it REPLACES the in-trainer cache
prewarm (the forkserver compile populates the same on-disk cache, which the Popen'd eval workers
still hit). Honest sizing unchanged: all-workers-ready improves ~30 s → ~20 s at 16 workers (maybe
~75 s → ~25 s at 48), ~50 s per 3 h restart — the reason to have it is that the architecture now
permits it and the guard structure makes it safe, not throughput.

**DEFAULT since 2026-08-17: it FOLLOWS `--compile-opponents`** (tri-state — `None` = unset ⇒ follow),
so both ship on and `--no-compile-opponents` turns the pair off in one flag;
`--no-compile-opponents-preload` keeps the per-worker compile and reverts to the cache prewarm. The
"requires `--compile-opponents`" error now fires only on an EXPLICIT preload beside an explicitly-off
opponent compile — erroring on the pairing the DEFAULTS produce would have made
`--no-compile-opponents` itself a usage error, which is the regression
`compile_defaults_test.py::test_no_compile_opponents_alone_is_not_a_usage_error` exists to catch.

**What justifies defaulting the thing whose predecessor hung a run**: the predecessor's cause is
fixed at the ROOT (lazy `poke_env` inits ⇒ a thread-free extractor import, pinned by
`compile_prewarm_test.py`), and the failure MODE is inverted — a preload that cannot prove
single-threadedness RAISES during forkserver bootstrap, so `SubprocVecEnv` construction dies with a
traceback in the parent instead of wedging 2 of 48 workers in silence. A loud startup failure with a
one-flag opt-out is a defensible default; a silent 13-hour stall would not have been.
**The 48-worker FORK STORM is now measured** (2026-08-17, `tmp/preload_fork_probe.py`, CPU, beside a
live run): arm the preload, then `forkserver` `Pool(48)` — **48/48 workers forked, 41 distinct pids
took a task, 48/48 reported the compiled graph present in inherited memory**, 19.7 s wall after an
11 s preload compile. That is the exact mechanism that wedged at 2-of-48 before, so the count-specific
fear is addressed directly rather than by extrapolation from the 4-worker run.
**⚠️ What is STILL untested is the full 48-env TRAINING composition** — real `Gen3Env` workers,
bridge children, a mid-run pool promotion — not the fork itself. If it ever refuses, the message
names the surviving thread and `--no-compile-opponents-preload` is the immediate way past it.

**Its fail-loud path was also observed, by accident.** A malformed `GEN3AI_PRELOAD_ARCH` during that
probe made the preload's extractor construction raise: the child's traceback printed in full and the
parent died on `EOFError: unexpected EOF` out of `forkserver.read_signed`. Loud, immediate, no wedge
— but note the PARENT-side exception is not self-describing, so **the diagnosis is in the child's
stderr**, which under the launcher lands in `launcher_child.log`.

**Failure is LOUD (`--compile-opponents-strict`).** Falling back to eager is a ~6.5× regression on the
opponent forward that is otherwise invisible — the run just produces fewer steps/hour forever and
looks healthy. Every failure path (`DISABLED`, `REVERTED`, mid-run `FELL BACK`, mis-declared
`hide_cuda`) goes through `_compile_warn`: stderr **and** the launcher event stream, so it surfaces in
the TUI. `--compile-opponents-strict` promotes all of them to a `CompileExtractorError` for anyone who
would rather fail at startup than find it in the FPS graph a day later.

**Caught at CODE time — and for all FOUR compile targets, not just this one.**
`src/agents/model/extractor_compiles_test.py` owns the device x grad matrix, because Inductor's CPU
backend emits C++ and its CUDA backend emits **Triton** — different lowering paths with different
bugs, so a green CPU-forward test is not evidence about any other cell:

| | forward | forward + backward |
|---|---|---|
| **CPU** | ✅ the frozen self-play OPPONENT (this section) | ❌ **does not lower** — but only in ONE of Inductor's three C++ store kernels: `CppKernel`/`CppVecKernel` both emit `atomic_add`, while `CppTile2DKernel` (the transposed variant, chosen by index LAYOUT) carries `assert mode is None`. CONFIG-CONDITIONAL too: the scatter is a gather's backward, so `--belief-grad-mode label_only` (stop-grad belief publication) deletes it and the backward then compiles (bisected 2026-08-15) |
| **CUDA** | ✅ eval / inference on the card | ✅ the TRAINER's step — measured 150.85 → 86.21 ms fwd+bwd at batch 4096 (**1.75x**), i.e. ~+60% end-to-end FPS at the ~89% train share. NOT wired up; the test keeps the lever available |

The ❌ cell is a **limitation PIN** (`test_cpu_backward_still_does_not_compile`) and it FAILS IF THE
LIMITATION LIFTS — three things assume it holds, starting with `maybe_compile_extractor` routing
every grad-enabled call to eager. It matches the TRACEBACK, not the message: torch raises a bare
unannotated `AssertionError` whose `str()` is empty, so `str(exc)` has nothing to match on.

Each compile cell runs **by default** (~10 s each on a warm cache; `GEN3AI_SKIP_COMPILE_TESTS=1`
opts out), so "the model stopped compiling" fails the suite instead of silently costing throughput.

⚠️ **The CUDA cells SKIP under a normal `pytest` run** — the root `conftest.py` hard-sets
`CUDA_VISIBLE_DEVICES=""` for the whole suite so a stray `device="auto"` can never steal VRAM from
a live training run. **You cannot compile FOR cuda ON the cpu** (measured 2026-08-14, torch 2.5.1 /
triton 3.1.0: with the device hidden, an Inductor cuda compile dies `RuntimeError: No CUDA GPUs are
available` — the backend queries live device properties, so codegen is not a blind AOT
source→PTX step; and a `FakeTensorMode` trace only exercises **dynamo**, which is device-agnostic
anyway and never reaches the backend where the device-specific bugs live). So the CUDA cells need
the real card:

```bash
GEN3AI_TEST_ALLOW_GPU=1 pytest src/agents/model/extractor_compiles_test.py -q   # 8 passed
```

Even unhidden they refuse to run when the card is BUSY (a free-VRAM floor read via `nvidia-smi`, so
the *check* creates no CUDA context either) — a compile test must never be what OOMs a 20-hour run.
Every skip NAMES the cause and the knob rather than saying "no CUDA device", because a silent skip
on a box that is always training would turn the gate into a no-op that still reads green.

### The recurring promotion cost — measured, and it is SMALL (~2.7%)

Everything above sizes **startup**. There is a second bill during the run, when a self-play
promotion makes env workers compile the new opponent. Measured on gen-14, 2026-08-17
(`designs/research_state/measurements/gen14_pool_refresh_compile_cost.json`, n=2 events):

| event | excess over the 138.6 s baseline | compiles | path |
|---|---|---|---|
| iteration 22 | **+1095 s** | 48 | all *timed* — each process's FIRST compile |
| iteration 42 | **+77 s** | 27 of 48 | all *"reused this process's validated compile"* |

**Read the second row, not the first.** Iteration 22 is not a promotion in the steady-state sense —
it is where **self-play first activates**: the pool is seeded from empty, so all 48 workers at once
load a 41 MB checkpoint *and* pay their process's first compile (the `revalidate` branch, which also
times eager-vs-compiled). It happens once per run. The recurring cost is **+77 s per promotion ≈
2.7% of wall-clock ≈ 16 min over a 25M run**, and `--compile-opponents` is net **+40%**.

**The caches work.** The shared Inductor cache is HIT at a promotion (13 files written, vs 6600+ at
run startup), `SnapshotPool._model_cache` keeps one compile per worker per snapshot, and
`_COMPILE_VALIDATED` puts every compile after a process's first on the cheap path. Nothing here
needs fixing.

**The one-time event IS addressable, and the flag for it is now ON BY DEFAULT — `--compile-opponents-preload`.**
The +1095 s is 48 workers each paying their process's FIRST compile, and fork-inheritance is exactly
the thing that removes it: the preload compiles once in the forkserver and workers inherit the traced
graph copy-on-write (**0.12 s per worker vs ~30 s**). Note the on-disk Inductor cache and the fork
inheritance fix DIFFERENT halves — the disk cache removes codegen, the fork removes per-process
dynamo tracing and guard construction, which is the half that was left.

Why the cost landed at iteration 22 rather than at worker startup: **the pool is empty until the
first promotion**, so workers have nothing to compile when they fork, and their first compile is
deferred to the moment self-play activates.

Two limits, unchanged by the default flip — expect a SHRUNK event, not a gone one:
- It SHRINKS the event, it does not remove it — those 48 workers also each load a 41 MB checkpoint
  (`load_model_snapshot` → deserialize → build policy), which no compile flag touches.
- The 0.12 s figure is a standalone probe of STARTUP compiles, and the flag's live proof is a
  **4-worker** run. A snapshot extractor compiled 2M steps AFTER the fork should still hit the
  inherited dynamo state (same `forward` code object, same shapes) but that case is not directly
  measured. And the hang this flag's predecessor caused was specifically at **48 workers** — it is
  fail-loud now (it RAISES rather than wedging), so the risk is a loud crash at construction, not a
  silent 13 h stall, but **48 envs is still untested for the fixed version**, and defaulting it on
  is what schedules that test for the next fresh launch. **Do not retrofit it onto a LIVE run**
  (a launcher-pinned worktree keeps its own code, so a live run does not pick this up); if a fresh
  48-env launch refuses at construction, the message names the surviving thread and
  `--no-compile-opponents-preload` is the immediate way past it.

⚠️ **A `[SELFPLAY EVAL] … [Ns]` line beside a slow iteration is NOT its cause — eval is genuinely
non-blocking.** gen-13 ran an **1865 s** eval cycle inside a **395 s** iteration. Attributing
iteration cost to an overlapping eval (or vice versa) is a window coincidence; separate them by the
compile path (`timed` vs `reused`), which is what actually distinguishes the expensive event.

## Compiled GPU trainer (`--compile-trainer`, DEFAULT ON for cuda)

`torch.compile`s the LEARNER's feature extractor — the CUDA forward **and backward** the PPO step
runs. The other half of the pair above, and the larger of the two.

🚨 **DEFAULT since 2026-08-17, and it is the one default that could NOT be a flat `True`.** This
flag REFUSES a non-cuda device (the first row of the refusal table below), so `default=True` would
convert every working CPU invocation — the `--debug` smoke, a laptop, CI — into a `FATAL_CONFIG`
exit. The default is therefore **AUTO**, resolved by `train_rl_agent.resolve_compile_trainer_default`
(pure, injectable, unit-tested without a card):

| resolved device | `--debug` | default |
|---|---|---|
| `cuda` / `cuda:N` | no | **ON** |
| `auto` on a box with a card | no | **ON** |
| `auto` with no card, `cpu`, anything else explicit | no | OFF |
| any device, including an explicit `--device cuda` | **yes** | **OFF** |

`--debug` is excluded outright because a smoke exists to prove the pipeline in ~1 minute and a
multi-minute Inductor compile (plus a CUDA context taken from whatever run owns the card) defeats
that. **The REFUSAL is unchanged**: an explicit `--compile-trainer --device cpu` still exits
`FATAL_CONFIG` with the same message. `--no-compile-trainer` is the opt-out, and it is also how you
KEEP the ObservationDebugger — see the trade below, which every default cuda run now makes.

**⚠️ The device is only HALF the auto default, and the other half is easy to miss.**
`check_shape_stability` (below) refuses `--async-rollout` and a rollout that does not divide by
`--batch-size` — both correct for someone who ASKED for the compile, and both fatal for a DEFAULT,
because they would convert two classes of command that work today into a startup `FATAL_CONFIG`.
So `resolve_compile_trainer_auto` runs those same checks and, on a refusal, **leaves the default OFF
and says why** rather than refusing to launch:

```
⚡ --compile-trainer would be ON by default here, but this config cannot take it — leaving it
   OFF rather than refusing to launch. Reason: … (pass --compile-trainer explicitly to make
   this a hard error instead.)
```

The rule, and it generalises to any future default: **a default yields to the config the user typed
and announces it; an explicit flag refuses.** Pinned by `src/main/compile_defaults_test.py`
(`test_auto_yields_to_async_rollout_instead_of_refusing_to_launch`,
`test_auto_yields_to_a_rollout_that_does_not_divide_the_batch`, and
`test_an_explicit_flag_never_reaches_the_auto_path`, which holds the refusal in place).

**Measured** (2026-08-14, v76 `gen3_ctx_dedup_v1`, RTX 3080 Ti, the real
`MaskablePPO -> ActorCriticPolicy._build()` path, gen-9's own `cli_args`: batch 4096, PopArt on;
`policy.evaluate_actions` fwd+bwd, arms interleaved, 3 pairs, idle box):

| scope | eager | compiled | speedup |
|---|---|---|---|
| extractor only (**what ships**) | 155.1 ms | 88.5 ms | **1.753x** |
| whole `evaluate_actions` | 155.5 ms | 88.5 ms | 1.757x |

**~+62% end-to-end FPS** at the ~89% train share. **We compile the EXTRACTOR, and the second row is
why**: the two scopes measure the same to within 0.004x — the mlp_extractor, the pointer head and
the value head contribute nothing — so the whole-policy scope buys nothing for strictly more graph
(and more surface for SB3's distribution objects and the mask path to break on). Same win, smaller
blast radius. Also confirmed: rollout 2048x48 / batch 4096 = **exactly 24 minibatches, no
remainder**, so one graph and no per-epoch recompile.

**FAIL-LOUD BY DESIGN, and deliberately asymmetric with `--compile-opponents`.** The opponent path
warns and falls back to eager (`--compile-opponents-strict` opts into raising) because it prints a
`[CompileExtractor]` line either way. Here there is nothing to notice: a silent fallback trains
perfectly correctly and just produces ~38% fewer steps/hour forever. So every failure is fatal
(`CompileTrainerError` -> `TrainExitCode.FATAL_CONFIG`, so the launcher gives up instead of
restart-looping), and the flag has no `strict` variant because there is nothing to opt into. Four
refusals, each guarding an otherwise-invisible outcome:

| refusal | why |
|---|---|
| `--device cpu` | The CPU BACKWARD does not lower — `CppTile2DKernel.store` asserts on the `atomic_add` mode. Pinned by `extractor_compiles_test::test_cpu_backward_still_does_not_compile`, which builds at `belief_grad_mode="shaping"` ON PURPOSE: under `label_only` (production since gen-11) those gather-backwards do not exist and the compile succeeds, so an unpinned test would have gone green while testing nothing. Costs nothing in practice — the compiled backward we run is CUDA, where Triton emits `tl.atomic_add` |
| compile raised | bisect the op — the whole "torch cannot compile our model" story was ONE op (see `src/agents/model/CLAUDE.md`, the `species_posterior` precedent) |
| compiled is not faster (< 1.05x) | the graph fragmented or the backend fell back per-frame; the measured figure is ~1.75x, so parity is a defect |
| compiled disagrees with eager (> 1e-4) | a faster wrong model is not a win |

Every rejection **uninstalls** the compiled callable before raising, so the process never keeps
running something it just declared unacceptable.

**Two MORE refusals, decided at startup, and the reasoning behind them is counter-intuitive enough
to be worth stating.** Recompiles here are NORMAL: `share_features_extractor=True` means one
extractor serves both paths, so `fe.forward` is called at batch=`n_envs` during rollout and
batch=`batch_size` during train, alternating forever. **Measured** (2026-08-14): alternating two
shapes converges after ~6 calls to a fixed 17 graphs and then never recompiles again (steady state
8.8 ms at batch 48 / 74 ms at 512). So `torch._dynamo.config.error_on_recompile = True` would crash
a perfectly healthy run on its second call, and `automatic_dynamic_shapes` is what makes the
two-shape case work rather than being the hazard.

The actual hazard is dynamo's **`cache_size_limit` (8)**: exceed it for one code object and dynamo
falls back to **eager SILENTLY** — precisely the invisible ~1.75x regression this flag exists to
prevent. Two configs get there, both decidable before training starts, both now fatal
(`check_shape_stability`, pure and unit-tested):

| refused | why |
|---|---|
| `--async-rollout` | the async collector forwards whichever envs are READY, so the rollout batch VARIES every step — an unbounded shape set, guaranteed to exhaust the cache. The error prints both measured numbers (`--async-rollout` +14% at n_envs=64 vs `--compile-trainer` +62%) so the choice is informed, not blind |
| `n_steps*n_envs` not divisible by `batch_size` | the remainder minibatch is a THIRD shape, replayed every epoch, for no benefit. The error names a concrete divisor to use instead rather than leaving you to do arithmetic |

Production is safe by arithmetic — 2048x48 = 98304 = 24 x 4096 exactly, so exactly two shapes — but
that was luck of the config until these guards existed.

⚠️ **The validation runs at a SMALL batch on a ZERO observation, and both halves are load-bearing.**

*Why not the train batch.* Because **validating at `batch_size` needs MORE GPU memory than training
does** — validation runs the arm eager AND compiled in one process with Inductor's workspace on top,
where training only ever needs one of them. At batch 4096 that exceeds the card. This was learned by
shipping it: it took down a gen-10 launch that had been running at 935 fps, first as a mystifying
`CUDA error: invalid configuration argument` and then, once the obs was valid enough to get further,
as the plain `OutOfMemoryError` underneath. The small batch is not a shortcut around an unexplained
bug; it is the only shape the check can afford. The honesty problem it was meant to solve — a
batch-64 ratio reading as if it were the production figure — is fixed by NAMING the shape in the log
line instead.

*Why zeros and not `torch.rand`.* A random float vector is **not a valid observation** — the
ObservationDebugger rejects it outright — so it can drive the forward down branches no real battle
reaches. All-zero is the canonical "nothing known" state (every categorical id 0, every flag clear),
structurally legal, and it is what `snapshot._zero_obs` has always used on the opponent path. The
trainer path briefly diverged to `rand` for no reason and that is what disguised the OOM as a CUDA
config error.

*Where per-shape correctness IS checked.* `torch.compile` compiles lazily PER SHAPE, so the graphs
production trains with (batch `n_envs` for rollout, batch `batch_size` for train) are never the one
the startup check compiles. That gap is closed by
`compile_trainer_test::test_every_production_shape_agrees_with_eager`, which asserts compiled ==
eager at each shape on a free GPU where memory is not contended. Measured once against the live
gen-10 config on REAL observations off the rust bridge: **batch 48 -> 9.5e-07, batch 64 -> 7.2e-07,
batch 4096 -> 3.6e-06**, against a value scale of 2.111 — float32 rounding, not a wrong kernel.

**⚠️ It DROPS the ObservationDebugger, and that is a production-visible trade.** Dynamo cannot trace
the debugger's numpy asserts at all (it dies building a guard over a numpy bool), so this is
compile-or-debugger, not both. The debugger attaches at `log_level >= PERIODIC` — i.e. it is ON in
production — so this flag costs you the per-forward obs-integrity check for that run.

**With the default ON, that trade is now made by EVERY plain cuda run, with nobody having typed a
flag — which makes the announcement more load-bearing, not less.** It is said twice: once at
startup, when the auto default resolves to on (`⚡ --compile-trainer ON by default (device=cuda)`,
naming the debugger and `--no-compile-trainer`), and once from `compile_trainer_extractor` when the
debugger is actually dropped. Neither line is conditional on a launcher being attached. The opt-out
is the only way to keep the debugger.

**Mechanics.** Patches the BOUND `fe.forward`, never the module: `torch.compile(module)` returns an
`OptimizedModule` and prefixes every `state_dict` key with `_orig_mod.`, which would land in every
checkpoint of the run and make them unloadable by anything else. It runs immediately BEFORE
`_run_roundtrip_test`, which turns that existing save -> reload -> forward gate into a free check on
exactly that hazard. **Runtime perf knob**: never versioned, never in `check_compatible`, NOT
inherited on resume — but with the AUTO default that means a flagless cuda resume gets it ON, so it
is `--no-compile-trainer` you re-pass each launch, not the flag.

Tests: `agents/model/compile_trainer_test.py` (the verdicts are pure functions so every refusal is
testable without a GPU — a contract that needs a free card is a contract that gets checked rarely;
plus a CUDA test that the `state_dict` keys and a save/reload survive) and the compile itself in
`agents/model/extractor_compiles_test.py`.

### Every non-training model can use it

`maybe_compile_extractor` is safe to apply to ANY frozen model, because the wrapper routes
**grad-enabled calls to eager**. That matters: the compiled artifact is inference-only (under
`requires_grad` dynamo hands the graph to AOTAutograd, whose CPU backward codegen fails on this
model's scatter/`index_add` — the documented reason the June `--compile-damage-op` integration was
inference-only), and the prober backprops through this same extractor for gradient saliency.

| consumer | what is compiled | gate |
|---|---|---|
| training env workers | pool / stable / exploiter opponents | `--compile-opponents` (+ forkserver preload) |
| `eval_worker` | **the trainee** (plays every eval game) + sentinel + fixed opponents | `compile_extractor` cfg key, threaded from both eval callbacks |
| `search_teacher_persistent_worker` | trainee (per re-freeze) + opponent (per iteration) | `compile_extractor` cfg key |
| `snapshot_ladder` | both frozen ladder players | **default ON** — offline tool, nothing races it |
| prober (`session._load`) | the no-grad replay / rollout models | `--compile` (off by default) |
| `play.py` | nothing — it runs `RandomPlayer` vs `RandomPlayer`, no neural model | n/a |

Eval workers are fresh `Popen` processes, so they hit the shared on-disk Inductor cache the trainer
already warmed rather than inheriting anything; one worker plays hundreds of games, so the compile
repays many times over.

**Verified end to end**, not just wired: `python src/agents/training/eval_sharding_fuzz_test.py 4 2
--compile --neural-opponent` drives the REAL `eval_worker._run` over the bridge and logs
`eval-trainee: ON — 3.33 -> 0.67 ms (5.0x)` and `eval-opp:final_model.zip: ON`, with every exactness
assertion unchanged (units played + pooled exactly, full coverage, claim-exactly-once across two
workers) — the compile is value-preserving, which is why running the same fuzz both ways is the test.

⚠️ **`--debug-eval` does NOT exercise this.** Its final win-rate eval runs IN-PROCESS; it never
spawns an `eval_worker`, so it shows zero compile lines and proves nothing about this path.
⚠️ **A bots-only plan does not exercise the OPPONENT half.** Scripted bots have no extractor, so
`_get_opponent_model` never runs — `--neural-opponent` adds a FIXED (frozen neural) opponent, which
is the only kind that reaches it. That gap is why the opponent path went unverified at first, and
`src/main/eval_worker_compile_test.py` now pins the wiring in the fast unit suite.

**Validation is paid once per process** (`_COMPILE_VALIDATED`), and the reuse path STILL LOGS
(`ON (reused this process's validated compile)`) — it used to return silently, which made the eval
opponent's compile look like it had never run and cost a round of doubt. A success you cannot see in
the log is a success you will not trust. The eager-vs-compiled timing answers
"does this extractor's code object compile to something faster?", and `torch.compile` keys on exactly
that code object — so a second model in the same process cannot get a different answer. Consumers that
load models in a LOOP (the search-teacher worker rebuilds its opponent every iteration) would
otherwise re-pay ~15 eager forwards each time. Deliberately process-local: a fresh process
re-validates, since that is where a cold cache or a failing backend would actually show up.

The prober flag is OFF by default because a one-off `summary`/`list` never amortizes a ~10-20 s
compile; turn it on for the search-shaped commands (`better-line`, `falsify`, `falsify-scan`,
`replay-counterfactual`, `lookahead`), which do thousands of no-grad rollout forwards.

**BLAS thread pinning (not optional).** Each worker runs a full CPU opponent forward; at the library
default of one thread per core, N workers spawn N×cores competing threads. Measured on a 16-core box
with 8 neural-opponent envs: **6 fps at load average 110, vs 231 fps pinned** — a ~38× cliff.
`launcher/child.py` has always exported `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`, so production under
the launcher was never affected, but `python src/main/train_rl_agent.py …` (a documented entry point)
had no protection. `train_rl_agent` now sets them at import (`setdefault`, before torch is imported —
BLAS reads them at init), and each env worker additionally pins `torch.set_num_threads(1)` so an
explicit learner-side override can't silently un-pin the workers. Pinned by
`src/main/thread_pinning_test.py`; the compile guards by `src/agents/model/compile_extractor_test.py`
(incl. a regression that the global `suppress_errors` never comes back) and the uncompilable-op
regression by `src/agents/model/extractor_compiles_test.py`.

`tmp/production_cmd.py` reconstructs a runnable command from any run's `metadata.json` `cli_args`,
diffing against the live parser's defaults and REPORTING (never silently dropping) flags the tree no
longer has — that is how the production shape above was recovered.

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
    `aux_terms` dict): `grad/{species_belief, move_belief, move_latent, win_prob, value_dist}_*`,
    each with `_share` (on the common `T`), `_norm_shared`, and `_policy_cosine` (<0 = that aux fights
    the policy). So the species CE, move BCE, SimSiam latent, move-latent grading, win-prob and value-dist
    pulls are **attributable individually** (the old combined `belief_share` lump is gone) — watch each
    sit small (~a few %); a spike with a degrading policy → lower THAT term's coef. `win_prob`/`value_dist`
    are ≈0 under `read_only` (stop-grad), real under `shaping`.
- **Per-edge-family LIVENESS — `edge/<fam>_weight_norm` + `edge/<fam>_grad_norm`**
  (`edge_family_metrics`, sampled once per `train()` right after the backward so `.grad` is still
  populated; parameters only, so the forward — and therefore the CPU opponent path — pays nothing).
  Every family enters as a ZERO-INIT `Linear(cell → 2·n_heads)`, which means **a family that never
  learns anything is bit-identical in the logs to one that works**: both write zero into the
  attention bias and neither says a word. The v79 `h` (pair-history) family shipped into a
  production run with exactly that blindness, and the only recourse would have been a post-hoc
  ablation at run end.
  - `weight_norm` — how far the map moved off its zero init. **Has it learned anything?**
  - `grad_norm` — how hard the loss is pushing it right now. **Does anything want it to?**
  - Read as a PAIR: both ~0 = genuinely dead (the cell carries nothing the loss can use); weight ~0
    with grad > 0 = still climbing off init (the expected early reading); weight > 0 with grad ~0 =
    converged and contributing. Weight norm alone cannot separate the first two.
  - ⚠️ **Neither is an EFFECT SIZE.** Both scale with the cell's input magnitude, so a family with
    larger-magnitude inputs shows a bigger gradient regardless of usefulness — measured at init on
    the gen-12 config, `h` reads the largest `grad_norm` of all 16 families (0.0100 vs d3's 0.0032),
    which says it is alive and being pushed, **not** that it is the most useful. The per-family
    ABLATION audit remains the only thing that measures importance, and these must never be quoted
    in its place.
- **Per-CELL LIVENESS — `cell/<name>_weight_norm` + `cell/<name>_grad_norm`**
  (`cell_family_metrics`, same window, same backward, same parameters-only cost). The identical gap
  one layer over: `SwitchBranchMoveCell`, `PairOutcomeMoveCell`, `PairOutcomeSwitchCell` and
  `ConditionalThreatCell` each enter through a **ZERO-INIT `proj` Linear** — deliberately, so that
  ON-at-init is byte-identical to OFF and any measured effect is something the run LEARNED — which
  means an enabled cell that never learns contributes exactly zero to every action logit and looks
  exactly like one that works. gen-16 turns four of them on at once, in the run meant to decide
  whether the switch-branch channel kills the bait-loop pathology, where **"the behaviour did not
  change" and "the cell never came off zero" must not be the same observation**
  (`designs/research_state/bait_loop_hunt.md` §6 makes this the launch-window check).
  Read the pair exactly as the edge families' — and under the same ⚠️: a parameter magnitude is not
  an effect size. `CELL_FAMILIES` is DECLARED, not duck-typed, so a renamed cell breaks the test
  rather than going quietly unmonitored. Nothing is emitted for a cell that is off — it is absent
  from the extractor, and a zero would read as "enabled but dead", a different claim.
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
loss), and **not weight-shape** (no `ARCH_SIGNATURE` bump). The v10
`value_active_readout` value-head fix that used to pair with it is **deleted** (v88
`gen3_dead_flag_purge_v1` — it was never enabled in a gen-8+ run and the multi-seed readout /
`--value-threat-inject` superseded it; a checkpoint recording it ON is refused by the migration). Validate by watching
`eval/td_resid_tail` fall.
Tests: `instrumented_ppo_test.py` (β=0 == MSE, β>0 == the exact blend).

## TD-consistency auxiliary (`--td-aux-coef`, `td_aux.py`)

**What it fixes.** The critic's only signal is a PER-STATE regression, `MSE(V(s_t), G_t)`. That
constrains each state's LEVEL and says nothing about the DIFFERENCE between two adjacent states — so
independent per-state noise ε in V arrives in `ΔV` at `2·Var(ε)`, exactly where the truth is nearly
constant. Since ΔV is what GAE reads, that is injected advantage noise on **every** transition, not
just the dramatic ones. `--td-aux-coef λ` adds the Bellman identity the critic already owes, as an
explicit loss:

```
loss += λ · mean_pairs[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]
```

Both residual ends carry gradient (the residual-gradient / Baird form — see the *Cons* in the
pre-registration). `λ = 0.0` is the default and the whole block is skipped, so an OFF run is
byte-identical. **Pre-registered band: 1.0–3.0, 3.0 the favourite; `λ ≤ 0.1` measured significantly
WORSE than control offline, so the small-coef regime is to be avoided, not treated as "a bit of the
effect".** Full pre-registration (rung-1 evidence, the honest ceiling, the rung-2 gates):
`designs/research_state/levers/td_consistency_aux.md` (ledger C5). Do not edit that file — it is the
pre-registration.

**Where the pairs come from — this is the whole engineering problem.** `RolloutBuffer.get()` yields
a RANDOM PERMUTATION, so a PPO minibatch contains **no adjacent pairs at all**; the pairs have to be
drawn from the buffer's surviving `[n_steps, n_envs]` structure. `td_aux.sample_contiguous_pairs`
draws `TD_AUX_STATES` (512) rows as contiguous per-env runs of `TD_AUX_SEG_LEN` (16) and pairs their
adjacent rows. Four facts make it correct:

- **Row convention.** After the first `get()`, `observations` are `swap_and_flatten`ed to ENV-MAJOR
  (`row = env·n_steps + t`), so temporal adjacency survives; the sampler returns rows in exactly
  that convention and `_td_aux_term` **raises** if `generator_ready` is False rather than indexing
  an un-flattened array (which would silently mis-pair states with rewards at any `n_envs > 1`).
- **`rewards` / `episode_starts` are NOT in `get()`'s flatten list**, so they stay `[n_steps,
  n_envs]` and are read in their native shape (rewards are swapped to env-major at use).
- **Episode boundaries DROP the pair, never zero it.** `episode_starts[t+1] == 1` means the
  successor begins a new episode, so (t, t+1) is not a transition; zeroing would train
  `V(s_t) → r_t` at every battle end. This also disposes of SB3's time-limit bootstrap (which folds
  `γ·V(s_term)` into the stored reward at the done step): that row's successor always starts an
  episode, so the pair never forms.
- **Segments, not random pairs.** L contiguous states serve L−1 pairs off L forwards — the
  "K+1 forwards serve K pairs" economy the pre-registration calls for, ~2× cheaper per pair than
  independent pair sampling. Rung 1 also found whole-battle batching beat a random-permutation
  control by 12%, so the within-segment correlation is a feature.

**It runs per MINIBATCH, with its own sample and its own critic forward** — modelled on the
search-teacher / OPD folds, not on the once-per-`train()` diagnostic probes. Those are read-only;
this one carries gradient, and a once-per-`train()` fold would give it ONE contribution against the
value loss's `n_epochs × n_minibatches` (~240 in production), so λ would have to be ~240× rung-1's
band to mean the same thing. Cost is bounded by `TD_AUX_STATES`, not by `batch_size`: one extra
512-state critic forward per minibatch, ≈10% of the train step at production shapes.

**The value path is `policy.predict_values`, never a hand-rolled one.** That method is what routes
to the DISTRIBUTIONAL head's mean under `--value-from-dist` (where the scalar `value_net` is FROZEN)
and applies PopArt's de-normalization — reading `value_net` directly would train a critic the run
does not use.

**Units.** `predict_values` returns REAL-unit values and the buffer's rewards are real-unit, so the
raw residual is real-unit. But under PopArt the value loss trains in NORMALIZED space, so the
residual is divided by σ — which *is* the normalized-space residual, since
`normalize(V) − normalize(r + γV′) = (V − r − γV′)/σ` (the μ cancels). λ therefore keeps the meaning
rung 1 calibrated in both regimes; σ = 1.0 with PopArt off.

**Metrics (`td_aux/` prefix).** `resid_rms` is the headline — the quantity being minimised, the live
counterpart of the offline ΔV-dispersion instrument, and it should FALL. `resid_mean` (SIGNED) is
the no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias drifting
away from ~0 means the residual-gradient term is shifting the LEVEL rather than tightening it — read
it beside `train/explained_variance`. Also `loss`, `n_pairs`, `scale` (the σ the residual is
expressed in) and `pair_drop_frac` (share of candidate pairs lost to episode boundaries). The
shared-trunk pull rides `grad/td_aux_share` + `grad/td_aux_policy_cosine`; the term reaches the trunk
through the CRITIC path only, so `td_aux_share` against `value_share` is the read for "is the
consistency term crowding out the level regression it is meant to complement".

**Class: `training_coef`.** Scales a loss, touches no forward pass ⇒ NO `ARCH_SIGNATURE` bump, NOT in
`check_compatible` and no `check_*` of its own; recorded on `ModelVersion` (`MODEL_CONFIG_VERSION`
v90) purely for provenance and so a **flagless resume inherits it** via `_resolve`, exactly like
`--opp-belief-aux-coef`. It is deliberately NOT in `agents/model/flag_registry.py` — that registry's
scope is extractor architecture toggles, and this reaches the extractor not at all.

Tests: `td_aux_test.py` — the sampler (env-major row convention, (t, t+1) adjacency, boundary pairs
DROPPED not zeroed, the all-boundary degenerate → `None` not 0.0, the segment economy, fail-loud on a
flattened `episode_starts`), the residual math on a hand-built case, the PopArt scale identity, both
ends carrying gradient, and on a REAL `train()`: coef-0 byte-identity (asserted twice — identical
parameters AND the sampler monkeypatched to raise, so a future sampler change cannot perturb an off
run), coef>0 moving the update and logging every metric, gradient landing on `value_net`, and the
un-flattened-buffer refusal.

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
- **The NSR advisor (`_noise_scale_advice` / `_emit_noise_scale_warnings`)** — when the SMOOTHED
  ratio leaves the band, a `⚠️ [NOISE]` warning goes to the launcher **Events panel** (via
  `main.launcher.ipc.emit`; plain print standalone) naming the concrete fix: ratio > 2 → "raise
  `--grad-accum-steps` ~ratio× (free — no VRAM/FPS cost, same rollout)"; < 0.5 → "over-batched,
  lower it for more steps per sample". **Rate-limited to one warning per key per 30 min** and
  suppressed for the first ~20 EMA folds (warm-up false-alarm guard). Pure decision logic
  unit-tested (`instrumented_ppo_test.test_noise_scale_advice_bands_and_fixes`). A FiLM-group half
  (`film/noise_scale*`, `--film-grad-accum-steps` and `_GroupGradAccumulator`) measured the same
  thing for the conditioning params until v78 and was deleted with the zarch family.

Tests: `instrumented_ppo_test.py` — `test_noise_scale_estimate_recovers_known_values` (the two-point
math recovers a planted `|G|²`/`tr(Σ)` exactly), `_smaller_batch_is_noisier_sign`, `_global_grad_sq`
matches a manual sum, and `_logged_only_when_accumulating` (real `train()`: skipped at accum=1, EMA
updated + scalar emitted at accum=2).

## ⚠️ Reading a belief target: `belief_supervision(...)`, never `last_*`

Cross-cutting rule for **every** belief loss below (`gen3_belief_label_only_v1`). Under
`--belief-grad-mode label_only` the extractor's `last_move_belief_logits` / `last_spread_belief` /
`last_hp_type_logits` / `last_spread_nature_logits` / `last_spread_ev` / `last_alpha_logits` stashes
are **stop-grad publications** — that is how the mode stops the policy/value gradient reaching a
belief head through any of its forward consumers. A supervised loss must therefore read its target
through **`self.policy.features_extractor.belief_supervision("<key>")`**, which returns the LIVE
tensor (and the identical object under `shaping`/`detached`).

A loss that reads the `last_*` attribute instead trains **nothing** under `label_only`, and does so
**silently** — the loss value, its gradient norm and every `belief/*` metric look completely normal,
because the loss is still computed; only the graph behind it is gone. The accessor raises a
`KeyError` on an unknown key so a typo cannot degrade into that, and
`agents/model/belief_label_only_gate_test.py::test_every_belief_loss_still_trains_its_head` is the
guard that each key still deposits gradient on its own head. The full four-route table is in
`src/agents/model/CLAUDE.md` → `--belief-grad-mode`.

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

## Latent-belief loss — DELETED (v75)

`--opp-belief-latent-coef`, the `opp_belief_latent` arch toggle, the `BeliefHead` SimSiam predictor,
the `belief_target_slots` training-only obs key and the env work that built it are **gone**. Recorded
here because the reasoning generalises to every aux head on this trunk:

- **It was never fed forward.** The latent was a side readout — stashed for the loss, never
  concatenated into `pi` or `vf`. Contrast `--opp-belief-cls-k`, which appends its pooled belief to
  BOTH projections and therefore buys the policy something at inference time.
- **It cost ~13% of the train step.** Measured per-flag on an idle box with interleaved arms:
  marginal **+341 ms** of train time at the production batch, against a `cls_k=6` costing +349 ms
  that *does* feed forward, and a `spread_belief` costing +72 ms. The train step is ~89% of
  production wall at 10 epochs, so this was real throughput.
- **Its own probe had already concluded decodable ≠ helps** (the belief latent/BYOL role-geometry
  probe: species geometry decodes strongly, and nothing downstream was shown to use it).

**Predicting the opponent's unrevealed mons is untouched.** `BeliefSlots` still fills the hidden opp
slots with learned tokens, the species CE and moves BCE still supervise them, and the T0 species
prior still feeds the physics. What is gone is the *second, graded* way of saying the same thing.

Migration: `MODEL_CONFIG_VERSION` 75 REFUSES a config that recorded `opp_belief_latent=True` (the
predictor carried parameters, so such a state_dict has keys the live extractor cannot accept) and
pops it when false. `sanitize_dead_extractor_kwargs` applies the same rule to a saved zip's
`features_extractor_kwargs`.

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
  shaping-flows gradient gating, the v22 version gate). End-to-end `--debug --use-bridge=node
  --win-prob-mode read_only` smoke confirms the roundtrip + `train/win_prob_*` metrics + `win_prob_share`=0.

## `cf_audit` — the counterfactual audit instrument (`cf_audit.py`)

```bash
python -m agents.training.cf_audit models/<run> \
    [--rollouts 8] [--states 200] [--step N] [--checkpoint PATH] [--impl rust] [--out DIR]
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

**Offline and standalone — it trains nothing.** Given a run's bridge-eval traces (the ones with a
`*_reconstruction.json` sibling) and a loadable checkpoint, it manufactures the value labels the
on-policy stream structurally cannot produce: for a sampled decision it plays the **recorded**
action and rolls the rest of the battle out live **R** times with fresh post-divergence dice
against the RELOADED real opponent, and takes the win rate. Training sees one Monte-Carlo sample of
each state's value; this sees R of the same state.

It emits two things, independently useful:

| output | what |
|---|---|
| `<out>/bias_map.json` + `bias_map.md` | predicted win-prob vs tight-MC per stratum, with `sd_true_excess`, battle-clustered CIs, the sampler design and full accounting |
| `<out>/cf_labels/labels_<producer>_<step>.jsonl` | label rows in the **shared v1 schema** — the contract a training-side consumer reads |

**The meter is `sd_true_excess`, NOT the mean gap.** G0 (2026-08-22) measured the population-mean
predicted−MC gap at |0.05|–|0.07| *with a sign that flips with the population you weight to*, while
the true within-decile spread of P(win) is 0.11–0.36 — the per-state error is 2–6× the aggregate
offset, so the head's defect is **resolution**, not an optimism offset. The estimator subtracts the
R-rollout binomial floor from the observed within-cell variance:

```
Var(MC | cell) = Var(true p) + E[sampling var];   E[p̂(1−p̂)]/(R−1)  is EXACTLY unbiased for p(1−p)/R
sd_true_excess = sqrt(max(0, Var(MC) − E[p̂(1−p̂)]/(R−1)))
```

Subtracting the floor is what makes it a claim about the world rather than about R. **A lever that
merely re-centres the head moves the mean gap and leaves this untouched** — and would be scored a
success by the wrong meter, which is exactly why the meter is stated here.

**The EVIDENTIAL read — the pre-registered meter for `--cf-evidential`, and the reader it was
missing.** The Beta head reads the same `value_pooled` as the scalar one, so it **cannot remove** the
blur G0 measured; the only success available to it is *confessing* it — wide exactly where the states
behind a confidence bin disagree. So the meter is not the loss but
**`width_vs_blur_spearman`**: the rank correlation, ACROSS STRATA, between the head's mean epistemic
width and the measured `sd_true_excess`. When the audited checkpoint carries a `cf_evid_head`,
`cf_audit` forwards it over the labelled states and the resolution table gains `evid_width_mean` /
`evid_precision_mean` columns beside each decile's `sd_true_excess`.

- **Rank, not Pearson** — the claim is an ordering ("wider where blurrier") and the two quantities
  are not on a common scale (a Beta's std vs the within-cell sd of an R-rollout mean).
- **The CI is a bootstrap over BATTLES**, and each draw rebuilds the strata from scratch through the
  same `resolution_cells` the point estimate uses. A draw that loses a thin decile to the minimum-n
  floor is dropped and reported as `draws_usable` — a CI whose resamples ran different arithmetic
  from its point estimate is a CI of nothing.
- **A FLAT width scores `None`, never 0.** "Wide everywhere" and "width unrelated to blur" are the
  same null in outcome but different findings in diagnosis, and the flat-to-1-ulp case (a weighted
  average of a constant) otherwise falls through to a `corrcoef` that divides by ~1e-17 and reports
  a confident correlation of float noise.
- **A checkpoint without the head OMITS the columns** and prints a one-line note. Zeros would render
  "this run has no head" identically to "this head claims no uncertainty". The read is
  **best-effort** throughout: the audit's products are the labels and the bias map, so a model that
  will not load (architecture drift — 79 of 79 archived runs) costs the run its evidential columns
  and nothing else. `accounting.evidential_scored` says how many states were scored.
- Reads the head through `ProbeSession.probe_model()` → `ProbeModel.cf_evidential_batch()`. That
  method exists because the extractor forward **never calls the head**, so unlike `win_prob_at` there
  is no stash to read: it forwards the extractor and applies the head to `stash.value_pooled`
  itself — the same thing `_cf_evidential_term` does, which is what makes the offline number
  comparable with the live `cf/evid_*` scalars.

**Label trust before map trust.** The ANCHOR arm runs FIRST: recorded action + recorded dice must
reproduce the recorded battle outcome. Below `--anchor-tolerance` (default 0.9) the tool exits **3**
and writes NO labels — the bias map is still written, marked `label_trust_passed: false`, for
diagnosis only. A factory whose replay is not exact is GIGO, and a map computed from it measures the
bug. Pinned by `cf_audit_integration_test.py`.

**Selection-awareness.** Eval traces over-capture losses (an explicit win/loss quota), so a pooled
gap convicts the critic of the sampler's sins. Every aggregate is computed *within* an outcome
stratum and recombined at the frame's own population shares; every CI is a bootstrap over
**battles**, never states.

**Sampling** is `(confidence decile × battle outcome × turn tercile)` with a declared
`CONVICTION_BOOST` on the high-confidence-from-lost-battles region (the "0.827 class", the
population R1 supervises). The weights, the seed and `SAMPLER_VERSION` are written into every bias
map — a silent priority change is a distribution-shift confound for every downstream readout.

**The shared label schema (v1)** — one JSON object per line; treat it as a contract, version it
rather than editing it in place:

```json
{"schema": 1, "kind": "mc_winprob", "battle": "<record path>", "decision_idx": 12,
 "obs_sha1": "<sha1 of the obs float32 bytes>", "obs_npz": "<states.npz>::obs",
 "obs_inline": null, "label": 0.625, "n_rollouts": 8, "wilson_lo": 0.30, "wilson_hi": 0.86,
 "policy_step": 24000000, "opponent": "heuristic", "created_unix": 1.77e9}
```

`obs_npz` names the array and `decision_idx` selects its ROW; `--inline-obs` swaps that for a
base64 float32 payload when the traces won't travel with the labels. `obs_sha1` is always present
so a consumer can verify the row it loaded is the row that was labelled.

**Known coverage gaps, printed in the accounting and never silent:** turn-1 decisions (the offline
replay driver cannot open them — one per battle, 3.35% of move decisions) and forced-switch rounds
(the re-roll layer anchors at start-of-turn move rounds).

**Cost** is the rollouts, not the materializer: an R=8 label is ~0.9 s at load ~7 and ~2.8 s at load
~25 — *more* load-sensitive than `loadavg/cpus` predicts, so any throughput figure taken beside a
trainer is a lower bound. Prefix sharing (below) does not apply to a rollout-to-end label, which has
one arm; it is the lever for the one-ply counterfactual (`lookahead`) path.

**Tests.** `cf_audit_test.py` (pure: the stratifier, `sd_true_excess` validated at ZERO true effect
AND at a known nonzero one, the clustered bootstrap, Wilson, the schema writer) and
`cf_audit_integration_test.py` (`sim`: a real bridge battle it plays itself, run end to end at R=2
— including the anchor refusal).

## Prefix-sharing materialization (`obs_materializer.materialize_branches`)

K counterfactual arms of one decision share an identical prefix, and the materializer used to
replay it from turn 1 for **every** arm — the measured bottleneck of the counterfactual label path
(`arm_ms = 4.78 + 0.853·turn`, of which prefix replay is `2.53 + 0.855·turn`; the branched turn is
~0.5 ms and the obs encode ~1.8 ms). `materialize_branches` replays the prefix once, snapshots the
player's whole battle/tracker state at the branch decision, and restores it per arm.

- **Contract: exactly equivalent to per-arm `materialize_decisions`, bit-for-bit.** Measured on 6
  gen-17 eval battles / 59 decisions / 452 arms: **59/59 byte-identical**, **15.4 → 5.3 ms per arm
  (2.91×)**, rising with the branch turn (3.7–3.9× at turn 26–28) because the part it removes is the
  part that is linear in the turn. Gate: `obs_materializer_branch_integration_test.py`, which
  compares EVERY arm rather than a sample.
- The clone SHARES append-only immutable records (`BattleEvent`, `BattleContext`) instead of copying
  them — a **contract, not an inference**, and the reason the gate compares every arm: a broken
  contract shows up as arm 2+ reading history arm 1 mutated.
- `lookahead` uses it for its whole `(candidate × seed)` sweep.
## Counterfactual win-prob grounding (`--cf-records` / `--cf-winprob-coef`, `gen3_cf_label_plumbing_v1`)

The **trainer-side plumbing** for `designs/ai_v10/design_counterfactual_value_grounding.md` — its gate
**G3**, which is explicitly "tap + buffer + flags at coefficient zero, byte-identity gated". Rung **R1**
only: tight Monte-Carlo P(win) labels, delivered to the **win-prob head**. The label PRODUCER is a
separate, out-of-process program (`cf_producer.py`, § *The label PRODUCER DRIVER* below);
**nothing in this section produces a label**, and the two halves share only a file format.

**Why the win-prob head and why head-only first.** The G0 bias map (ledger 2026-08-22; 2,204 tight-MC
labels) found the head's defect is **RESOLUTION, not an optimism offset** — population-mean gaps are
|0.05|–|0.07| while the true within-decile spread of P(win) is 0.11–0.36, 80–95% of it real
state-to-state variance. Only tight-MC labels carry that within-bin separation; a single realized
outcome (what the on-policy BCE eats today) structurally cannot. The head is MC-native, so R1 needs no
route change and owes no C4 gate. `--cf-head-only` defaults **TRUE** because the safe stage comes first:
the term trains the head's own params and provably cannot perturb the trunk.

**The four pieces:**

- **The record TAP (`cf_records.py`, `--cf-records`, default OFF).** The bridge emits a `__RECON__`
  reconstruction record at the end of every episode; **training discards it** (`BridgeSession` keeps a
  single overwritten slot), which is precisely why a label producer cannot reach a training decision.
  `--cf-records` threads a `recon_sink` callable into `attach_bridge_transport`, and each env worker
  writes the record into `<run_dir>/cf_records/` as a **count-capped ring** (`--cf-records-keep`,
  default 512). Crash-safe (`.tmp` + `os.replace`), filenames sort chronologically
  (`<time_ns>_<pid>_<tag>_reconstruction.json`) so the prune needs no `stat`, and the cap is **GLOBAL** —
  every worker prunes the shared dir and a lost delete race is swallowed, which is what keeps the bound
  across `n_envs` AND across launcher restarts. **The cap only bounds the directory because the `.tmp`
  is bounded too**: `prune` matches on `RECON_SUFFIX`, so a `<...>_reconstruction.json.tmp` is invisible
  to it — a failed write therefore unlinks its own tmp, and `prune` additionally sweeps tmps OLDER than
  the oldest kept record (a crash between `open` and `os.replace` cannot unlink its own; a tmp being
  filled right now is newer than every record on disk, so the sweep can never race a writer). Without
  that, the full disk this module promises to survive leaked one file per episode per worker, forever.
  **The automatic prune is THROTTLED to one write in `prune_every` (16).** It is a full `readdir`
  running on the bridge reader's coroutine — the path every env step waits behind — and pruning per
  write paid that scan ~512 times to delete ~512 files. The price is a bounded transient overshoot
  (≤ `prune_every` unpruned writes per live writer) and it degrades gracefully **because the cap is
  global**: every writer sweeps the WHOLE directory, so one worker's next sweep collects every other
  worker's backlog, and a process that dies mid-backlog has its leftovers collected by the next
  one's first sweep. Bound: `keep + prune_every·n_writers` transiently, `keep` again the moment any
  writer sweeps. `prune()` itself is unthrottled (a caller that wants the cap now can have it).
  The artifact shape is byte-for-byte the one
  `reconstruction._write_artifact` writes, so `ReconstructionRecord.load()` reads a ring file directly.
  A write failure warns once and is swallowed — a full disk must not crash a run. `--cf-records`
  without `--use-bridge` is REFUSED (a websocket run emits no such frame; the flag would be a silent
  no-op).
- **The LABEL BUFFER (`cf_label_buffer.py`).** Watches `<run_dir>/cf_labels/labels_*.jsonl`, remembering a
  per-file byte OFFSET so an appending producer is read incrementally and a partial trailing line waits
  for the next poll instead of counting as malformed. **The offset is keyed on `(name, inode)`, and the
  map is pruned to the files still on disk** — a producer that DELETES and RECREATES `labels_x.jsonl`
  (an in-place rotation) gets a new inode, and keying on the name alone made the buffer seek past the
  new file's first `offset` bytes and drop those rows with no counter and no warning. "Never a silent
  accept" has a mirror: never a silent DROP. Schema v1 is in the module docstring; obs resolve
  `obs_inline` > `obs_npz` > skip. **Everything unexpected is a COUNTED skip, never a crash and never a
  silent accept**: unknown `schema`, unknown `kind`, malformed JSON, out-of-range label, unresolvable
  obs, an obs whose width ≠ this run's, and an `obs_sha1` that disagrees with its own bytes (the GIGO
  guard — it warns loudly once). `obs_npz` resolves `<path>::<key>` and **`decision_idx` selects the
  ROW** of a 2-D array (which is what `cf_audit` emits by default, one battle's whole obs matrix per
  row) through a small per-file LRU, so N rows of a battle open the archive once instead of N times.
  FIFO at `capacity`, and **staleness expiry** at `--cf-label-lag-steps`
  (default 150 000 ≈ one production PPO iteration): `age == bound` survives, `age == bound + 1` does not,
  enforced at ingest AND on every poll. `0` disables expiry.
- **The label-QUALITY trio (task #28), landed before the coefficient ever goes live.** At coefficient
  zero none of these costs anything; the moment the term is on, each is a silent change to what the
  critic is taught.
  - **DEDUP on the obs digest, keep-NEWEST.** A producer that re-labels a decision it already shipped
    (an overlapping cycle, a re-run over the same trace tree, a truncate-and-rewrite) would give that
    one state N× the weight of every other — a change to the sampler's declared distribution with no
    flag and no counter, which design decision-of-record 3 forbids. The resident row is REPLACED, not
    appended beside. Keep-newest because a fresher label is a strictly better estimate of the same
    state (measured under a policy closer to the consumer, and carrying more evidence if R grew), and
    the replacement re-enters at the FIFO tail rather than inheriting the old row's position.
    `cf/labels_replaced_total`. Measured before: a 5-row file rewritten in place left fill **6**.
  - **SYMMETRIC staleness** — the bound is on `abs(current_step − policy_step)`. A crash-restart
    resumes from the last checkpoint, so `num_timesteps` moves BACKWARDS while the label files still
    carry pre-crash steps; under a one-sided test those rows are **immortal** and quietly become the
    whole buffer. Live tell, measured: `cf/label_age_steps_p50` reading **−4,999,000**. Future rows
    expire like stale ones, are counted separately (`cf/labels_future_total`) and trip a one-time
    loud warning naming the cause — a negative age is a diagnosis, not noise.
  - **The ObservationDebugger is SUPPRESSED around the CF forward** (`--no-compile-trainer` runs, the
    only ones that still have it). The CF rows are recorded FOREIGN states — other episodes, other
    policy steps, read off disk — and the debugger's premise is "this is the board we are about to
    act on"; it was being handed 256 replayed rows per minibatch and reporting their integrity
    against the live env's expectations. `Gen3FeaturesExtractor.suppress_observation_debugger()` is a
    context manager that restores on the way out (including on an exception) — deliberately NOT
    `disable_observation_debugger()`, which is permanent and is the compile path's trade.
- **The LOSS (`instrumented_ppo._cf_winprob_term`).** Per minibatch — the `_td_aux_term` / search-teacher /
  OPD shape, and for the same reason: the labelled states are recorded PAST decisions, absent from this
  rollout, so they cannot ride `rollout_data`, and a once-per-`train()` fold would make the coefficient
  mean something different from every other aux. `_cf_sample_and_forward` samples up to `CF_SAMPLE_SIZE`
  (256) rows and runs ONE extractor forward (`{"observation": …}` is the only key the model reads); the
  term applies the win-prob head to `stash.value_pooled` — **detaching iff `cf_head_only`**. It
  The forward runs under **`no_grad` unless something downstream actually wants the graph** — the
  condition is computed exactly (`cf_head_only` OR a dead `cf_winprob_coef`), not assumed, because the
  one arm that needs it is `--no-cf-head-only` with a live coefficient, and silently dropping the
  graph there would turn the trunk-open A/B into two copies of head-only. Both heads still train
  their own params either way: `head(value_pooled)` is applied OUTSIDE the context, which is pinned
  on the parameter update rather than argued. It
  deliberately does NOT read `last_win_prob_logits`: that stash is produced under the extractor's own
  `win_prob_mode`, which governs the ON-POLICY win-prob BCE; this term's trunk exposure is a separate
  decision, and re-applying the head makes the two independent by construction. It CLOBBERS the
  minibatch's extractor stashes, so it is folded beside `_td_aux_term`, after every loss that reads one.
  The **evidential term (below) shares that ONE sample and that ONE forward** — two samples would pay
  twice for the block's whole cost and would make the two terms disagree about which states they scored.
- **The scalars.** `cf/*` is **producer liveness and is published whenever a buffer exists**, even if not
  one label ever arrived — `cf/buffer_fill`, `cf/label_age_steps_p50`, `cf/labels_ingested_total`,
  `cf/labels_expired_total`, `cf/labels_future_total`, `cf/labels_replaced_total`,
  `cf/labels_skipped_total`, plus `cf/rows_sampled` (rows the fold actually CONSUMED this `train()`,
  summed over minibatches — residency and throughput are different questions, and only the second
  goes to zero when a producer dies while its last labels are still resident). That is deliberate: an empty buffer that does not
  announce itself is this tree's oldest failure mode (the search-teacher's silent starvation), and a flat
  `labels_ingested_total` is unambiguous evidence the producer stopped, which reads completely differently
  from a rising `labels_expired_total` (a producer that is running but lagging). `train/cf_loss` +
  `train/cf_grad_share` are the TERM, only when it folded; `cf_grad_share` is lifted from the
  grad-balance probe's shared denominator (so it is comparable with `grad/policy_share`) and reads
  **exactly 0.0 under `--cf-head-only`** — that is its verification, not a defect.

**Flag class — plain argparse, deliberately.** All four are **training-only**, exactly the `--opd-coef`
class: not in `agents/model/flag_registry.py`, not on `ModelVersion`, not in `check_compatible`, no
version bump. Nothing here is weight-shape relevant. They are therefore **not read back on a flagless
resume** (there is no `ModelVersion` field to read them from); the launcher forwards every non-launcher
flag verbatim, so a launcher-managed resume keeps them, and a bare `train_rl_agent.py --model …` does
not. `--cf-winprob-coef > 0` REQUIRES `--win-prob-mode read_only|shaping` — `none` does not build a
`WinProbHead`, so a live coefficient would fold nothing for a whole run; the parser refuses it, and the
loss independently no-ops if the head is somehow absent.

### The LIKELIHOOD: `--cf-label-likelihood {binomial,bce}` (default **`binomial`**, `gen3_cf_binomial_likelihood_v1`)

The label schema carries `label` **and** `n_rollouts`, so the row's win COUNT is recoverable —
`w = round(label · n_rollouts)` — and the flat BCE was throwing that away. A 0.75 label from 4
rollouts and a 0.75 from 16 are the same number carrying **four times the evidence**; scoring them
identically is a modelling error, not a weighting preference.

```
w = round(label·n)            NLL_i = −[ w_i·log q_i + (n_i − w_i)·log(1 − q_i) ]
term = Σ NLL_i / Σ n_i        (mean NLL per ROLLOUT)
```

- **`binomial` is the DEFAULT**, and that is a deliberate break with the usual "new option defaults
  to old behaviour" rule: `--cf-winprob-coef` has never been live in a production run, so there is
  no trained behaviour to preserve and nothing to be compatible with. `bce` stays as the explicit
  A/B arm.
- **The normalization is `Σ NLL / Σ n`**, not `Σ NLL` and not `/mean(n)`. Two properties buy it: a
  producer that changes its R does not silently change the effective coefficient, and **at `n ≡ 1`
  it reduces EXACTLY to the mean BCE** the flat path computes (a one-rollout label is already 0 or
  1, so the round is the identity and `Σn = B`). That exact agreement is pinned bit-for-bit, which
  is what makes `binomial` a strict generalisation rather than a different objective.
- Computed through `softplus` (`−log σ(z) = softplus(−z)`), stable where `log(sigmoid(·))`
  underflows. A row whose producer omitted `n_rollouts` parses as 0 and is clamped to **one**
  observation — never a divide-by-zero, never a silently dropped row.
- Training-only, the `--opd-coef` class: no forward, no weight shape, no version bump, **not read
  back on a flagless resume**.
- `cf/n_rollouts_mean` rides beside `cf/loss` — under the binomial likelihood the loss is per
  rollout, so a producer that quietly changed R would otherwise move the loss with no visible cause.

### The EVIDENTIAL Beta head: `--cf-evidential` + `--cf-evidential-coef` / `--cf-evidential-reg` (`gen3_cf_evidential_head_v1`, v98)

**What it is for, and what it is NOT for.** G0 convicted the win-prob head of **RESOLUTION**: the
population-mean gaps are |0.05|–|0.07| while the true within-decile spread of P(win) is 0.11–0.36.
A point estimate cannot represent that spread at all. This head reads the same `value_pooled` and
therefore **cannot remove the blur** — it has no information the scalar head lacks. What it can do
is **CONFESS** it: emit a Beta whose width is large exactly where the states behind a confidence bin
disagree. A confessed width is actionable (the factory's priority sampler can label the states the
critic knows it cannot separate; the awareness stack can read it); a point estimate that is silently
wrong is not.

- **`CfEvidentialHead` (`agents/model/aux_value_heads.py`)** — the `WinProbHead` bottleneck widened
  from 1 logit to 2, mapped by `softplus(·) + 1` so **α, β ≥ 1**: the Beta stays UNIMODAL (α<1 puts
  mass at an endpoint, turning "uncertain" into "certain of both extremes") and the uniform
  `Beta(1,1)` is exactly reachable, so maximum ignorance is a representable state.
- **The loss is the Beta-Binomial MARGINAL likelihood** of the row's counts — `p` integrated out,
  not plugged in: `NLL = −[log B(α+w, β+n−w) − log B(α, β)]` (lgamma-based; `log C(n,w)` is dropped
  as a constant in α,β). That is the correct evidential objective for count data, and it does two
  things at once: pulls the mean toward `w/n` AND grows the precision `α+β` only as far as
  consistency across states supports. Normalized by `Σn` like the scalar term, so the two
  coefficients are in the same units (nats per rollout). Checked against
  `scipy.stats.betabinom.logpmf`, not against a re-derivation of itself.
- **`--cf-evidential-reg` (default 1e-3) is the standard evidential-overconfidence guard**:
  `KL(Beta(α,β) ‖ Beta(1,1))`, closed form via digamma/lgamma, exactly 0 at the reachable floor. It
  rides INSIDE the coefficient, so coefficient zero kills the regularizer too. Nothing in the
  likelihood bounds `α+β` on locally-consistent data, and an inflated precision makes the width —
  the entire product — meaningless.
- **ALWAYS DETACHED, with no mode to change that.** Unlike `win_prob_mode` / `value_dist_mode` there
  is no read_only/shaping split: the head feeds nothing forward, so letting it shape the trunk would
  be a training change with no consumer to justify it. `train/cf_evidential_grad_share` reads
  **exactly 0.0 by construction** — published so the contract is a live measurement, not a docstring.
- **It is not called by the extractor forward at all** (the training-side term applies it to the
  stashed `value_pooled`), and it is built **LAST** in `Gen3FeaturesExtractor.__init__`. So OFF is
  byte-identical AND **ON-at-coefficient-0 is BIT-identical in pi/vf** — a stronger claim than the
  two precedents make, and one that depends on the build order: a module inserted mid-constructor
  shifts the init RNG stream for everything after it.
- **Metrics `cf/evid_*`**: `nll`, `reg`, `alpha_mean`, `precision_mean` (α+β — the claimed
  evidence), `epistemic_std_mean` (**the headline**), `pred_mean`, `n`; plus
  `train/cf_evidential_loss` and `train/cf_evidential_grad_share`. Read `nll` and `precision_mean`
  together: a falling NLL with a runaway precision is the head buying its loss with certainty it has
  not earned. A per-decision `(α, β)` stash lands on `fe.last_cf_evidential` for a future trace
  capture; **the npz capture itself is NOT wired** (deliberately deferred). ⚠️ Note when picking that
  up: the stash is written **only by the train loop**, so wiring it through `RLPlayer` would capture
  nothing — the extractor forward never calls the head, so an honest per-decision capture has to
  CALL it at record time (as `ProbeModel.cf_evidential_batch` does) and add an npz key.
- 🔒 **THE PRE-REGISTERED READ, for the experiment that has not run yet:** the predicted Beta's
  width should **CORRELATE with the measured `sd_true_excess` per stratum** (the `cf_audit` bias
  map's meter). Wide everywhere and wide nowhere are the same null. A falling `nll` with a flat
  width-vs-`sd_true_excess` correlation is the standing learns≠helps kill, not a result.
  **That correlation now has a reader**: `cf_audit`'s `width_vs_blur_spearman` (§ *The EVIDENTIAL
  read* above) computes it with a battle-clustered bootstrap CI, so the meter is an instrument
  rather than an intention.

**Flag class — the split, and why.** `--cf-evidential` is **STRUCTURAL** and IS in
`agents/model/flag_registry.py` (v98, `cli`/`structural`): it is a `Gen3FeaturesExtractor`
constructor kwarg that builds a MODULE, which is exactly the registry's declared scope, and the
`win_prob_mode` / `value_dist_mode` precedent. It gets a `ModelVersion` field, a `check_compatible`
bool compare, a `MODEL_CONFIG_VERSION` bump to **98** with a migration defaulting pre-v98 configs
OFF, and a `snapshot.current_model_version` keyword (so a frozen eval/pool opponent's gate sees it).
**No `ARCH_SIGNATURE` bump** — optional side head, obs family unchanged, the value_dist precedent.
The gate matters more here than usual: because the head is never called by the forward, a mismatched
resume produces **no shape error anywhere**, so `check_compatible` is the only thing standing between
a flipped flag and a run that silently supervises a freshly-random head for good. The two
**coefficients** are training-only argparse (the `--opd-coef` class) and are deliberately NOT in the
registry — they are loss weights set on the model, never reaching the extractor.

`--cf-evidential-coef > 0` REQUIRES `--cf-evidential`, refused at the CLI. Unlike the win-prob case
the head cannot be added later to rescue a live coefficient: it is a state_dict change, so the
mistake would cost a whole run AND FATAL the resume that tried to fix it. The `cf_labels/` directory
is created when **either** consumer is live, so an evidential-only run is not silently starved.

**Gates.** `instrumented_ppo_test.py` pins the byte-identity that G3 is: a POPULATED buffer at
`cf_winprob_coef=0` yields the same parameter update as no buffer at all (the fold is gated on the
COEFFICIENT, not the buffer), and so does a live coef with no head. The two `cf_head_only` halves are
measured on the parameter update rather than asserted about a detach call — head-only moves the head and
leaves the trunk bit-identical; `--no-cf-head-only` moves the trunk. The same file pins the binomial
likelihood's exact properties as pure-function facts (`binomial == bce` bit-for-bit at `n≡1`; the
gradient ratio is exactly `n₂/n₁`; per-rollout normalization; `w` recovery; the `n=0` degradation) and
the evidential fold's three (ON-at-coef-0 byte-identical with the head in the optimizer; a live
coefficient reaching ONLY `cf_evid_head` — trunk AND win_head bit-identical; one shared sample and one
shared forward for both terms, counted). `agents/model/cf_evidential_head_test.py` holds the head's
maths (scipy cross-check, the hand-computed uniform-Beta anchor, `KL(Beta(1,1)‖Beta(1,1)) == 0`, the
regularizer actually moving α,β toward 1, the 1/√12 std anchor), the BIT-identity of ON's pi/vf, that
the forward never calls it, and the v98 gate + both migration legs.
`cf_label_buffer_test.py` covers FIFO, the exact expiry boundary (past AND future, both inclusive),
incremental polling, the partial-line case, every skip counter, dedup keep-newest + the
rewrite-converges case, the `obs_npz` row index and its per-file cache bound, the ring's
cap/atomicity/race-tolerance, the prune throttle's declared overshoot bound, **the launcher-restart
cap across sequential processes** (the one G3 sub-claim that used to stand on construction alone),
and that `batch_tensors` carries the rollout COUNT rather than just the ratio. The CF forward's two
guards are pinned in `instrumented_ppo_test.py` on the *stashed tensor* and the *parameter update*
rather than on a `with` statement: no graph under head-only, a graph in the trunk-open arm, both
heads still receiving their own gradients under `no_grad`, and the debugger suppressed-then-restored
(including on an exception). `main/cf_flags_test.py` covers
the defaults, both `--no-` spellings, the three new refusals and `checkargs`. End-to-end: a
`--debug --steps 10000` CPU smoke with fixture labels built from REAL episode obs.

### The TWIN HEADS + the SHADOW CRITIC (`--cf-twin-heads` / `--cf-shadow-critic`, `gen3_cf_twin_heads_v1`, v99)

**The owner-authorized amendment to the SIGNED R1 pre-registration** (ledger 2026-08-22 evening,
"Three owner sign-offs" item 3). It changes what the arm's primary comparison *is*, so read this
before reading the runbook's §2.

**The problem it solves.** R1 as signed compared two RUNS — an arm with `--cf-winprob-coef` and a
control without. Two runs differ in every random draw they ever make, and the primary meter carries
a MEASURED floor of ~39% of its own variance (`tmp/hidden_info_floor_report.md`). So a cross-run
difference has to clear noise the design cannot control, and a null would be uninterpretable.

**The design: three win-prob heads on ONE trunk, differing ONLY in their label stream.**

| head | module | trained by | isolates |
|---|---|---|---|
| **A** (control) | `win_head` — the EXISTING head, untouched | the on-policy single-outcome BCE, at `win_prob_coef` | — |
| **B** (coverage) | `cf_twin_head_b` | A's loss **+** the cf-labelled states with **SINGLE-OUTCOME** labels (n≡1) | **B−A = coverage/prioritization** |
| **C** (treatment) | `cf_twin_head_c` | A's loss **+** the same states with **TIGHT-MC** labels (n=R) | **C−B = pure variance reduction** |

That factorial is the mechanism split. `C−A` remains the original R1 claim; the amendment's value is
that it now decomposes. Because all three read the same `value_pooled` on the same rows in the same
minibatch, the trunk, the states, the seeds and the hidden-information floor are **identical by
construction**, not matched by design.

- **B and C are `WinProbHead` — the same class and capacity as A.** A difference of architectures
  would be a second explanation for every difference of scores, and nothing downstream would say so.
- **Head-only ALWAYS in v1.** Both twins read a DETACHED `value_pooled` in *every* term they take,
  including the on-policy mirror. So this measures the **LABEL effect on a trunk that is frozen with
  respect to them**; trunk exposure and policy transfer stay CROSS-RUN questions (runbook §0a,
  unamended). `train/cf_twin_grad_share` reads exactly 0.0 — published so the contract is a live
  measurement.
- **The mirror rides `win_prob_coef`, not `cf_twin_coef`.** All three heads must carry a
  bit-identical copy of the control objective, or B−A would confound "extra states" with "a
  different base objective".
- **B and C pull EQUALLY HARD.** `_cf_binomial_nll` normalizes by `Σn`, so a row's gradient is
  `(q − target)/B` whatever its n. B's n≡1 rows and C's n=R rows therefore differ only in the
  TARGET — which is what makes C−B a read of label PRECISION rather than of effective learning rate.
- ⚠️ **`cf/twin_b_coverage` is the FIRST thing to read.** A producer shipping no `outcome_label`
  trains B on nothing; B then equals A, the pre-registered C−B contrast silently becomes C−A, and
  every other counter reads healthy. That is the one way this arm produces a confident wrong answer.
  B's fold is skipped rather than trained on a zero-filled absent label, and the scalar says so.

**The SHADOW CRITIC** is the other half and a different job: a passive `ShadowValueHead` trained on
**`mc_return`** labels — the mean realized **shaped return** over the producer's rollouts, in the
units the live critic V actually predicts. It **never computes an advantage, never enters GAE,
feeds nothing forward, and reads `value_pooled.detach()` unconditionally** (the `pubval` structural
precedent). Swapping the live critic for an MC-grounded one is critic SURGERY and owes the C4
offline gate; this head is the **staged promotion path** that earns or refuses that gate without
risking a run.

- **The frame.** Under PopArt the head's raw output IS the normalized value and the target is
  `popart.normalize(mc_return)` — `_value_distill_mse`'s handling, for its reason (the coefficient
  stays scale-comparable with the value loss). Every reported metric is DE-normalized to real
  shaped-return units, which is the only frame a reader can interpret.
- 🔒 **THE METER is `cf/shadow_shadow_vs_live_v`** — the SIGNED real-unit mean of (shadow − live V)
  on the same states, with the live V taken off the *same* forward through `policy._critic_value`
  (never a hand-rolled `value_net` call, which under `--value-from-dist` reads a head the run does
  not use). A shadow sitting systematically BELOW the live critic is a live critic that is
  optimistic about the states the factory samples, **measured against ground truth rather than
  argued from a calibration curve**. `cf/shadow_live_v_vs_label` is its direct half; read them
  together, because the shadow is itself a fitted head and can be wrong too.

**The LABEL SCHEMA decision, and why it is not a version bump.** The three streams ride ONE row
(`outcome_label`, `mc_return` + `mc_return_n`, `reward_sha1` as additive-optional v1 fields) rather
than arriving as separate `kind`s. Two reasons, the first decisive: **`CfLabelBuffer` dedups on the
obs digest**, so a second row for the same state would collide and one would silently replace the
other. And one-row-per-state makes "heads B and C saw identical states" *structural* rather than
hoped-for. `schema` stays **1** because it is a REFUSAL gate — a consumer skips every row whose
version it does not know — so bumping it would make a new producer's output unreadable by an
existing trainer, which is the opposite of backward compatible. Old consumers ignore the new keys;
new consumers supervise nothing extra when they are absent.

**`mc_return` carries a REWARD DIGEST and is REFUSED on a mismatch.** A shaped return is a fact
about a board *under a reward composition*, so a return measured under a different `RewardConfig`
is a measurement of a **different value function**, not a noisier sample of ours — and there is no
shape error or range violation that would catch it. `reward_config_digest(config)` (a stable sha1
over every `RewardConfig` field) is stamped by the producer and handed to the buffer by the
trainer; a mismatch drops the **field** (never the row — its win-prob labels are still good), counts
`cf/labels_mc_return_rejected_total`, and warns once by name. The digest is only passed when
`--cf-shadow-coef > 0`: a run with no shadow head must not reject rows over a field it does not read.

**The producer side** (`cf_producer.py`): `outcome_label` is free (it already computes the recorded
outcome for the critic-surprise term). `mc_return` needs the server-free reward path —
`agents/training/cf_mc_return.py` wraps `RewardTracker`, keeps the per-turn rewards *in order*, and
folds them with `--gamma`. Two non-obvious facts live there: **`RewardTracker` accumulates an
UNDISCOUNTED total** (a return is `Σγᵏr` from a particular state, so the rewards must be captured
per turn), and **the divergence turn's own move is SCRIPTED**, so a tracker hooked only into the
live `choose_move` would begin at T+1 and its return would be missing `r_T` and carry an extra
factor of γ — against the very state the label is FOR. That is why `install_scripted_prefix` grew an
`on_scripted_decision` hook (default None, byte-identical): it REPORTS, and the producer's closure
decides. The reward config is read from the run's own `metadata.json` `cli_args` through the SAME
`RewardConfig.from_args` the trainer uses; when it cannot be read the default is used and the fact
is printed LOUDLY, because the digest will then simply not match and the trainer will say so.

**⚠️ TWO SEAM BUGS shipped in the first version of the `mc_return` path and were caught by
adversarial review, not by the tests.** Both produced plausible-looking labels; keep them in mind
before moving either seam.

1. **`action_to_order` is NOT a valid recording seam.** It looks ideal (the commit point; it raises
   `StaleDecisionError` on a superseded attempt) — but `counterfactual._invert_choice` calls it in a
   **LOOP over every legal index** to recover a recorded choice's action number, on every scripted
   decision of the prefix. Recording there fired 6-9 times per scripted turn with actions that were
   never played, each advancing the STATEFUL reward function. The seams are `_predict_best_action`
   (caches the committed `(idx, mask)`) + the player's own `choose_move` (the once-per-decision
   boundary; it must be wrapped BEFORE `install_scripted_prefix`, which captures it as its live
   delegate) + `_battle_finished_callback` (the terminal reward).
2. **The hook must `arm_at_next()` AND `note()`, in that order.** Arming alone left the first LIVE
   decision at T+1 as the armed one, so `r_T` was dropped and every label was `G(s_{T+1})` against an
   obs row for `s_T` — biased by whatever happened on the divergence turn (a KO there is the largest
   single shaping term), i.e. **correlated with the state and shaped exactly like a real signal**.

Both are pinned in `cf_mc_return_test.py`, the second with an explicit negative control showing the
buggy shape, because neither is visible in any scalar. Note what did NOT catch them: the
bridge-backed composition test asserted only that an `mc_return` was PRESENT. **A composition test
that checks presence rather than value is a presence test.**

**Two counters, not one, and the distinction is the same one twice.** `cf/labels_skipped_total` is
the ROW-level GIGO meter and must keep partitioning the input with `labels_ingested_total`; an
optional FIELD that is malformed or out of range ACCEPTS the row and counts into
`cf/labels_field_skipped_total`, and a reward-digest refusal counts into
`cf/labels_mc_return_rejected_total`. Folding any of these into the first would make "is the
producer feeding me garbage" climb at the ingestion rate on a buffer refusing nothing.

**The discount comes from the RewardConfig, not from a flag.** `reward_config_digest` hashes every
field including `gamma`, so folding the return at `cfg.gamma` puts the discount under the same GIGO
guard as the reward. `--gamma` survives only as an explicit override, and its help says what that
costs: a mistyped value ships returns folded against a different value function with the digest
still matching and every liveness counter reading healthy.

**⚠️ The ONE coupling head-only does NOT remove: the global gradient CLIP.**
`clip_grad_norm_` scales every gradient by `max_norm / total_norm` over ALL parameters, so any term
with a non-zero gradient anywhere perturbs the policy and value updates in the last bits. It is
tiny at a sane coefficient and it is shared by every aux this tree runs — but it is not zero, and an
arm claiming a bit-identical trunk must know which of the two mechanisms it is claiming.
`instrumented_ppo_test.py::test_the_only_coupling_between_a_headonly_term_and_the_trunk_is_the_GLOBAL_CLIP`
pins the pair: with the clip active the updates differ, with it raised out of the way they are
bit-identical. A genuine gradient leak would survive both.

**Flag class.** `--cf-twin-heads` and `--cf-shadow-critic` are **STRUCTURAL**, in
`agents/model/flag_registry.py` (v99, `cli`/`structural`), with `ModelVersion` fields, bool compares
in `check_compatible`, a `MODEL_CONFIG_VERSION` bump to **99** with a setdefault-False migration,
`snapshot.current_model_version` keywords, and **no `ARCH_SIGNATURE` bump** (optional side heads,
obs family unchanged) — the `cf_evidential` precedent exactly, and the gate matters for its reason:
the forward never calls these heads, so `check_compatible` is the ONLY thing that can catch a
flipped flag. `--cf-twin-coef` / `--cf-shadow-coef` are training-only argparse (the `--opd-coef`
class), deliberately not in the registry and **not read back on a flagless resume**.

Refusals, all at the CLI: `--cf-twin-coef > 0` requires `--cf-twin-heads`; `--cf-shadow-coef > 0`
requires `--cf-shadow-critic` (both are state_dict changes and cannot be added mid-run to rescue a
live coefficient); and **`--cf-twin-heads` requires `--win-prob-mode read_only|shaping`**, because
the twins mirror head A's loss and `none` builds no head A — the arm's control arm would silently
not exist.

**The AUDIT read** — `cf_audit` gained `attach_twin_heads` (one more batched forward, same
best-effort contract as `attach_evidential`) and two blocks:

- 🔒 **`twin_paired` is the amended PRIMARY.** Per row, `brier = (pred − mc)²` and
  `abs_err = |pred − mc|` for each head, **differenced across heads on the same row**, with a
  battle-clustered bootstrap CI on the difference. Two properties buy it over `sd_true_excess` here:
  the hidden-information floor **cancels exactly** (it is a property of the STATE, identical in
  every arm — the amended §2 argued it cancels at matched *step*; twins strengthen that to matched
  *state*), and no stratification means no selection correction is owed. **SIGN: these are ERROR
  scores, so a NEGATIVE difference means the first-named head is better.**
  ⚠️ A near-zero contrast with a near-zero `mean_abs_pred_diff` is a **coverage/dosage** reading,
  not the pre-registered null — the label streams did not separate the heads and there is nothing
  to decompose yet.
- **`twin_resolution`** is the G0 continuity link: each head's own `sd_true_excess` binned by its
  own prediction. Its cells are **UNWEIGHTED** and the block says so in its own `weighting` field —
  the population re-weighting is unavailable for B and C, because the eval frame carries only head
  A's predictions, so their decile membership over the whole frame is unknown. Absolute levels here
  are NOT comparable with the bias map's `population_weighted_sd_true_excess`.
- **`shadow`** carries `shadow_vs_live_v` (signed, battle-clustered) and `shadow_vs_live_v_abs`.

**Gates.** `agents/model/cf_twin_heads_test.py` holds the heads' contracts (the shadow's UNBOUNDED
range — a sigmoid creeping in would clamp every label while the MSE fell; the twins' identical
architecture; their INDEPENDENT init, so `cf/twin_b_vs_c_abs` at step 0 is not reading its own
initialization; the BIT-identity of ON's pi/vf for each flag and both together; that the forward
never calls any of them; the v99 gate on both flags and both migration legs; the registry rows; the
`current_model_version` threading). `instrumented_ppo_test.py` pins the routing and the isolation:
coefficient-zero byte-identity for each half, a live coefficient reaching ONLY its own heads (with
the clip raised — see above), **the ROUTING pin** (B's loss equals the binomial NLL of the OUTCOME
and demonstrably NOT of the tight-MC label, with the two set to opposite extremes), B's n≡1
weighting, B's skip-and-count when no row carries an outcome, the mirror's coefficient and its
detach, the shadow's PopArt frame and masking, and that all FOUR cf terms share ONE sample and ONE
forward. `cf_label_buffer_test.py` covers both schema directions (an old row still ingests; a new
row carries both streams), the out-of-range field skip that keeps the row, the reward-digest
refusal and its counter, the coverage scalars, and the masks in `batch_tensors`.
`cf_mc_return_test.py` pins the oldest-first discount and the deliberate one-decision arming delay.
`cf_audit_test.py` pins the sign convention, the honest null, the refusal to compare heads it does
not have, and the shadow block's signedness. `main/cf_flags_test.py` covers the defaults, both
negation forms, the four refusals and `checkargs`. **The composition** is
`cf_producer_integration_test.py` (`sim`): a real bridge battle → the ring → one producer cycle →
the REAL `CfLabelBuffer`, now additionally asserting every row carries a valid `outcome_label`,
that at least one carries an `mc_return` with its digest, that the buffer keeps them, and that a
buffer configured with a FOREIGN digest refuses the `mc_return` while keeping the row.

### The label PRODUCER DRIVER (`cf_producer.py`) — the piece that runs the loop

```bash
nohup nice -n 10 python -m agents.training.cf_producer \
    models/<run> [--rollouts 8] [--top-n 3] [--records-per-cycle 4] \
    [--max-labels-per-hour 2000] [--anchor-every 50] [--impl rust] \
    > models/<run>/cf_producer.log 2>&1 &
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

The tap rings records; the buffer consumes label rows; **this walks one to the other.** It is a
long-lived **standalone sidecar run beside a live trainer** — the `snapshot_ladder` /
`bot_matchup_matrix` pattern — and deliberately NOT auto-spawned by the trainer: producer and
consumer share only a file format (that is `cf_label_buffer`'s whole premise), and a producer the
trainer owned would make a label-path failure a *training* failure.

Each cycle: poll `<run>/cf_records/` for unprocessed records → refresh the freshest `checkpoints/`
snapshot (via `latest.txt`, else the highest-stepped zip; its step is stamped on every label) →
replay each record ONCE (which yields the realized outcome, every decision's obs, its mask, its
action index and its committed choice string, via `obs_materializer.scan_record`) → forward the
snapshot over the candidates → label the top `--top-n` by the declared priority → roll each out
`--rollouts` times → write one NEW file per batch to
`<run>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`.

⚠️ **THE ECOLOGY DECISION — read this before quoting any label this producer wrote.**
A training record carries **no opponent identity**. The tap's `__RECON__` frame holds the resolved
seed, both packed teams and the committed choices, and nothing that says *which policy* sat on the
other side — a self-play pool snapshot, one of the nine heuristic bots, or the trainee's own
weights. The label therefore cannot name the opponent it was measured against, and a value claim
that cannot name its population is not a value claim (the G0 rule: *never quote "the critic is
optimistic by X" without naming the population — the sign depends on it*). So v1 makes the
approximation **explicit rather than guessed**: every rollout is played by the **CURRENT snapshot
on BOTH sides, sampling stochastically at temperature 1.0** — the regime the training actor itself
plays in. That matches the ~90% self-play share of the training mixture, and it is wrong in a
KNOWN direction for the rest: on an episode whose opponent was a bot, a weaker opponent is replaced
by a stronger self-like one, so that label is biased LOW. Every row carries
`opponent: "self_current"` — never a bot name it cannot verify — so a reader can always tell a
producer label from a `cf_audit` label, whose opponent IS identified. Closing the approximation
means threading the opponent's identity through the training-side tap; it is not a change to
`cf_producer.py`. **Stochastic is the load-bearing half of the regime**, not a style: a greedy copy
of a net is strictly stronger than a temp-1.0 sample of it, and greedy rollouts biased the prober's
sentinel labels LOW by a measured +0.037 [+0.007, +0.066].

**Which side is the trainee.** A training record names none, so `_trainee_side` answers from the
transport's own invariant: `BridgeSession` seats `env.agent1` — the trainee — on **p1**, always. A
record that DOES name a trainee (an eval sibling handed to this tool) is honoured instead.

**The sampler is DECLARED and VERSIONED** (`cf_producer_priority_v1`), written into the state file
AND every label row, because a silent priority change is a distribution-shift confound for every
downstream readout (design decision-of-record 3):

| term | what | weight |
|---|---|---|
| `critic_surprise` | `\|P(win\|s) − realized outcome\|` — the **conviction region** G0 measured at +0.23, and the population R1 exists to supervise. A single realized outcome cannot say whether the head was wrong or the dice were (53% of that class was genuinely winning); tight-MC labels are the only instrument that separates them, so they are spent here first | **1.00** |
| `policy_entropy` | the masked action distribution's entropy ÷ `log(n_legal)` — the decisions the policy has not made up its mind about. **Normalized by the support size** so a 2-way coin flip outranks a 9-way near-certainty, which raw entropy inverts | **0.35** |

A tie (the turn cap) scores outcome **0.5**, not a loss — it is uninformative about conviction, not
evidence the head was wrong. A checkpoint with **no win-prob head** has no surprise term at all;
the producer says so once and ranks on entropy alone rather than reading a missing head as a
confident 0.0. Candidates are start-of-turn **move rounds** at turn ≥ 2 only (a forced-switch round
has no valid recorded answer to script, and the offline driver cannot open turn 1 — the same
declared gap `cf_audit` carries).

**Crash safety, and what it costs.** A record is claimed in `<run>/cf_producer_state.json` and the
state file is **fsync-replaced BEFORE its rollouts run**. So a crash mid-record loses that record's
labels and can NEVER double-label. That direction is deliberate: the buffer dedups on the obs
digest, so a duplicate is survivable — but it is also a silent re-weighting of the declared
sampler, and a record aged out of the ring unprocessed is simply a record that was not labelled.
Missing a label is free; mis-weighting the sampler is not. Pinned on the ORDER (the state file must
already be durable when `process_record` raises), not argued.

**The anchor rule, inherited from `cf_audit`.** At startup and every `--anchor-every` records, one
record is replayed FULLY SCRIPTED through the live bridge (`divergence_turn=None` — the correctness
oracle) and must reproduce the winner the offline replay driver reports. On failure the producer
**exits 3 and writes nothing further**: a factory whose replay is not exact is GIGO, and every
label after it would be a measurement of the bug. This anchor is *stronger* than `cf_audit`'s —
nothing is played by a policy, so a failure is unambiguously a defect rather than a die roll. An
anchor that CRASHED counts as a FAILURE, never a pass.

**Observability.** A separate process has no TensorBoard, so it prints one **heartbeat line per
cycle** and keeps `<run>/cf_producer_state.json` human-readable (indented; sampler + weights +
totals + the last heartbeat + skip reasons):

```
[cf_producer] cycle 2 | snapshot step 29,867,520 | records 1 pending / 3 done | labels 2
              (+6 total, 6/h) | anchor 1/1 | PRODUCING | load 23.6 | 9.2s
```

The trainer-side half of the contract is the `cf/*` scalars — `cf/labels_ingested_total` going flat
is what a dead producer looks like from over there (see the R1 runbook's launch-window table).

**Two guards on running beside a live trainer.** `--max-labels-per-hour` (default 2000, a sliding
one-hour window) keeps it a sidecar. `--stale-checkpoint-minutes` (default 90) **pauses production**
when no NEW checkpoint has appeared for that long — the trainer is probably gone, and a producer
grinding against a frozen snapshot either burns the box filling a buffer whose rows will expire, or
teaches the current policy an ancestor's values. It keeps WATCHING (a restarted trainer resumes it)
and announces itself exactly once in each direction. `--lag-warn-steps` (default 150 000, matching
the buffer's `DEFAULT_LAG_BOUND`) warns once when the snapshot in hand falls that far behind the
newest checkpoint.

**`obs_materializer.scan_record`** is the new read primitive under it. An eval trace ships its obs
and action indices in `states.npz`; a training record ships **neither** — only the seed, the teams
and the committed choice strings — so the only route to a training decision's observation is to
replay the one-sided protocol AND recover the action history by inverting those choices through the
real mapper. `scan_record` does both in ONE replay and returns `RecordDecision(index, turn, action,
choice, mask, obs)` rows. It shares `_InvertingReplayPlayer` with `infer_action_indices` (which
stays track-only), and both go through one `_encode_or_track` step so the two replay players cannot
drift on the one operation where drift would silently change an obs rather than fail.

**Tests.** `cf_producer_test.py` (pure: the priority arithmetic incl. the entropy normalization and
the tie rule, the state file's claim-before-work order and its bounded processed set, the throttle
and its sliding window, the stale-trainer pause + resume, the anchor's refusal / cadence /
crash-is-a-failure, the ecology field on every row, checkpoint resolution, and that every help
string renders). **The deliverable is `cf_producer_integration_test.py` (`sim`)**: a REAL bridge
battle → the REAL `CfRecordRing` in the TRAINING tap's shape (**`trainee_username` stripped**) →
ONE REAL producer cycle → the REAL `CfLabelBuffer`, asserting every row INGESTED with **zero skips**,
digests verifying, correct `policy_step`, and — the strongest assertion in the file — that the obs
the producer *materialized* is **bit-identical** to the obs the LIVE player encoded, which is the
only thing that proves the inverted action history did not desync the encoder's trackers. Both
halves of a two-process contract had unit tests when the last two contract bugs shipped; neither
test ever ran the other half's real output, which is why this file runs the composition.

## Public-replay value aux — V_pub — DELETED (v88 `gen3_dead_flag_purge_v1`)

The v43 pubval subsystem (`--pubval-mode`/`--pubval-coef`, `agents.training.pubval`,
`pubval_calibration`, `data/gen3_pubval.json`, `PubValHead`, `_pubval_loss`, the parity fuzz) is
**deleted** — it measured NULL as a lever and was never ON in a production generation. A checkpoint
recording `pubval_mode != "none"` is refused by the v88 migration (re-read it from the git_hash in
its metadata.json); `"none"` pops silently. The raw replay corpus (`replays/showdown/gen3ou/`,
local-only) and the design doc (`designs/ai_v8/design_public_info_value.md`) remain for history.

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
  `main/prober/engine_test.py` (`build_value_dist`). End-to-end `--debug --debug-eval --use-bridge=node
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
calibration / ELO on a `coef=0` A/B. (The old "~⅔ of grind losses are matchup-lost / UNCOACHABLE"
caveat is **RETRACTED** — model-judged recoverability is circular; treat those losses as headroom.)

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

**Sim engine (`impl`, no flag of its own).** Every child the teacher spawns — the generation
battles (`teacher/generate.py` → `run_local_battles`), the searches (`SearchSession`) and the
replay/re-roll driver (`ProbeSession`) — takes its engine from `SearchTeacherCallback(impl=…)`,
which `train_rl_agent` sources from the existing **`args.bridge_impl`** (so there is no new
user-facing flag; `"node"` when `--use-bridge` is off, which is the historical behavior). It rides
each worker's config JSON as an `"impl"` key. This closed a real silent gap:
`teacher/generate.py`'s `run_local_battles` call had **no** `impl=`, so on a `--use-bridge=rust`
run its battles would have been generated on node regardless.

**`--use-bridge=rust` + `--search-teacher` now RUNS** — the old hard `parser.error` is deleted
(`gen3_rust_search_driver_v1` / `gen3_rust_replay_driver_v1`: one `search_driver` binary serves both
offline verb families). Each LEG is gated on rust — `better_line` node≡rust candidate V (an
obs-level bit-identity claim), `search_clone_parity` (clone ≡ `reroll_many` at the obs), and the
counterfactual confirm leg — but the COMPOSITION is not: **no full multi-cycle teacher run has been
done end-to-end on rust.** Treat the first one as an experiment and fall back to `--use-bridge=node`
if a cycle misbehaves. That guard's OLD stated reason — the record's `input_log` being
replay-equivalent rather than byte-identical — was **wrong and is retracted**: no consumer reads the
committed-choice lines, so do not re-derive a plan from it. See `src/utils/bridge/README.md` →
*Offline driver transport* for the seam and the full gate table.

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
