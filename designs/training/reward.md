# Training — reward

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

## Reward redesign — registry + PBRS + the no-progress clock (`reward_manager.py`, `progress_clock.py`)

> **Where the reward lives.** `reward_manager.py` (the terms, the folds, `RewardConfig`,
> `RewardBreakdown` and the composition census) · `reward_weights.py` (every tunable MAGNITUDE —
> weights, bonuses, thresholds, clamps; re-exported by `reward_manager`, so the old import path
> still resolves) · `reward_verify.py` (the `GEN3AI_REWARD_VERIFY=1` shadow twin) ·
> `progress_clock.py` (the no-progress clock the reward READS). Changing a value in
> `reward_weights.py` is a RETRAIN-class change, not a knob.

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
`reactive_layout["turns_since_progress"]`, absolute obs column **1602**) and the reward
(`last_penalty` → `no_progress_tax`), so **obs and reward key on one value**. The ternary predicate per decision window: PROGRESS (our-attributed damage ≥3% / status
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

### The clock's two intent-restoring fixes — `--progress-decision-tense` / `--progress-switch-freeze`

**Both default OFF; a flagless run is byte-identical to what every generation through gen-15
trained** (proved by `gen3_data_obs_parity_integration_test`'s committed golden and by
`progress_clock_test.py`'s recorded default trace, captured against the pre-fix implementation).
They are `RewardConfig` fields — resume-immutable, value-checked, recorded in `model_config.json`,
`MODEL_CONFIG_VERSION` 105, **no `ARCH_SIGNATURE` bump** — and are threaded onto the clock by ONE
call, `ProgressClock.apply_reward_config(cfg)`, used by both `gen3_env.py` and `reward_tracker.py`
so training and eval cannot drift on what the clock does.

**Why they exist, in one line each.** Probe M censused what the tax actually charges
(`designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md`) and probe N traced the
term's intent against its implementation (`.../no_progress_tax_review_2026-08-29.md`). Between them
they found that **79% of all charges land on the two paths below**, and that neither is what the
design specified.

| flag | what it changes | measured motivation |
|---|---|---|
| `--progress-decision-tense` | both window GATES (the forced-switch sit-out **and** the trapped-vs-wall charge suppression) read the decision that OPENED the window instead of the one after it | the sit-out lands on **19,503 full-agency decisions** (13.2%, the costliest class at −5.1pp) while the **zero-agency post-faint replacement is charged 63.9%** of the time — **36.3% of all charges** |
| `--progress-switch-freeze` | a VOLUNTARY switch that fails `_is_progress` FREEZES the window (no increment, no charge) rather than being taxed | `_is_progress` is offense-only, so no switch can satisfy it by its own doing: **−0.101** expected charge per voluntary switch vs **−0.010** per move, and within the switch branch the discrimination is **INVERTED** (Δ mean `d_out` **+0.0103** [+0.0076, +0.0131] — the charged switches are worth *more* win probability). **42.7% of all charges** |

**The tense fix is ONE off-by-one in ONE call, and it has two halves that must move together.**
`phase_is_forced_switch` reads `curr_ctx.phase` — the phase of the request that CLOSES the window —
because it was minted eleven days earlier for the obs history slot, where that is the correct read.
`ProgressClock` reused it for "was the decision that opened this window forced", and no test could
catch it: **both readings are true statements about the same delta**. The same call also passed the
upcoming request's `legal` to the helplessness gate, so a mon genuinely trapped at `t` is charged
whenever its successor could switch. The fix adds a NEW field, `TurnDelta.decision_was_forced_switch`
(`prev_ctx.phase == "forced_switch"`), and threads `legal_prev` alongside `legal`;
**`phase_is_forced_switch` is deliberately untouched** — the obs decoder, `opp_intent_labels` and
`reward_manager` all want the closing tense, and re-pointing it would silently change what they mean.

**F2b, not F2a.** Probe N specified two spellings and this is the one that ships: the alternative
(give `_is_progress` a switch clause keyed on belief-delta or type-matchup) would reintroduce exactly
the hand-coded switch heuristic `928a00b` deleted on the argument that switching value is LEARNABLE
from Φ_mat + `pbrs_belief` + the terminal. The freeze is instead the **composition-corrected reading
of the original intent**: the design's "a pure tempo-pivot pays the toll once" was written for a
reward that also paid the same pivot `switch_base +0.5` / `se_switch +0.2` / `escape_threat +0.25`;
`928a00b` zeroed every one of those and explicitly kept the tax, so the sign of the net switch
incentive flipped with nobody re-deriving the term. The **honest cost** is that a pure A↔B
switch-loop becomes free — anti-stall survives via the move turns between pivots, `--draw-penalty`
and the 250-turn forfeit, and **stall rate / mean game length is the canary** on any arm that runs it.

**Both are RETRAIN-CLASS when ON, and the blast radius is measured rather than argued.** `n` is the
obs scalar as well as the charge basis (that identity is the Markovian design's whole premise — a
fix that moved only the reward would break it), so turning either on changes the observation stream.
`progress_clock_obs_confinement_integration_test.py` captures the golden 6-battle set under each arm
and reports every differing cell: **`--progress-decision-tense` 49/991 decisions, `--progress-switch-freeze`
153/991, and in both cases the ONLY column that ever differs is 1602** — `turns_since_progress`
itself. No other block moves, no dim moves, the trajectory does not branch.

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
is strictly worse than a clean loss, which cancels the γ=0.9999 discount pull (the `shaped` default;
`--critic winprob` runs γ=1.0 and REFUSES `--draw-penalty`) of delaying an inevitable
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
`ModelVersion.check_reward_config` (the same machinery as `--vf-coef`). Tests: the `reward_*_test.py`
per-term spec family over `reward_test_fakes.py`
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
is the causal A/B for "is it the objective tilt that helps." Tests: `reward_bias_terms_test.py::TestSwitchBias`.

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
and the healthy-explosion rate fall.** Tests: `reward_end_state_test.py::TestSelfKoPenalty` (unit) +
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
`reward_bias_terms_test.py::TestBiasDrops` + `snapshot_test.py` (resume-immutability + v12→v13 migration).

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
Tests: `reward_pbrs_progress_hazard_test.py::{TestProgressPBRS, TestHazardPBRS}`,
`reward_pbrs_boosts_roar_test.py::{TestBoostPBRS, TestOppBoostsPBRS}`,
`reward_end_state_test.py::{TestEndStateDrops, TestAllShapingPbrsNoOpDefault}` + `snapshot_test.py` (resume-immutability + v13→v14 +
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
| `--no-all-shaping-pbrs` | `1 TERMINAL + 2 PBRS + 25 BIAS` |
| `--stall-pbrs` (with the default) | `1 TERMINAL + 8 PBRS + 0 BIAS` — the zero-bias destination |
| **`--no-hand-shaping`** | `1 TERMINAL + 0 PBRS + 0 BIAS` — the CLEAN WORLD (see below) |

#### The CLEAN-WORLD switches (`gen3_clean_world_config_v1`, config v105)

**Four resume-immutable fields, every default equal to today's behaviour**, so a flagless launch is
byte-identical. Spec: probe N
(`designs/research_state/measurements/no_progress_tax_review_2026-08-29.md` §5).

| flag | default | what `false` / the other value does |
|---|---|---|
| `--hand-shaping` | ON | **the master.** All EIGHT `_fold_*_pbrs` early-return AND the whole BIAS class is zeroed, `no_progress_tax` included |
| `--pbrs-material` | ON | drops Φ_mat (the term had NO flag at all before) |
| `--pbrs-belief` | ON | drops the EMITTED Φ_belief term only — see the mutation note below |
| `--victory-value` | 30.0 | the ±terminal, promoted off the `reward_weights.VICTORY_VALUE` module constant |

🚨 **Why a master flag rather than "just turn `--all-shaping-pbrs` off": the two halves are
ANTI-CORRELATED across it.** `all_shaping_pbrs` does two jobs — it folds five potentials *and* it is
`_bias_term_active`'s master gate — so `--no-all-shaping-pbrs` silences the potentials while
**reviving 25 BIAS terms**. "No hand PBRS **and** no BIAS" sat in a hole between the two settings and
no combination of the pre-existing flags could reach it (asserted, in `clean_world_config_test.py`,
so nobody "simplifies" `hand_shaping` away later).

⚠️ **State this honestly in any write-up.** Every PBRS term is **policy-INVARIANT by construction**
(`Φ(terminal)=0`, telescoping), so removing them **cannot change the optimal policy** — it changes
learning dynamics and conceptual complexity. The clean-world claim's real content is "the hand terms
cost more in interference and tuning than they buy in credit assignment", never "the hand terms bias
the objective". The only class that biases the objective is BIAS, and that was flag-zeroable before.

🚨 **THE OUTCOME ORDERING is the clean arm's largest hazard.** `draw_penalty = -35 < -30` exists so
that stalling to the 250-turn cap is strictly worse than losing cleanly. At a ±1 terminal a
`draw_penalty` of `0.0` **inverts** that: the stall becomes the best non-winning outcome, and with
`no_progress_tax`, `stall_tax` and Φ_progress all removed, nothing else opposes it. The owner's
ruling is **draw = loss**. `resolve_config` prints a loud `[Reward] ⚠️ ORDERING` line whenever
`draw_penalty > -victory_value` — a warning, not an error, because an arm may want it — and
`--victory-value <= 0` is refused outright. **Register stall rate + mean game length as a PRIMARY
safety endpoint on this arm, not a secondary one.**

🚨 **THE ORDERING GUARD IS ONE OF THREE, and the other two are about SCALE** (`_terminal_scale_guards`
in `main/train/config.py`, added by the R1 adversarial review). `--victory-value` is the first flag
that can move the RETURN SCALE, and two older flags are quietly denominated in that same scale:

| `[Reward] ⚠️` line | fires when | why it is not just style |
|---|---|---|
| `ORDERING` | `draw_penalty > -victory_value` | a draw beats a loss ⇒ stalling is optimal for a losing agent |
| `TERMINAL SCALE` | `\|draw_penalty\| > 3 × victory_value` | `--victory-value 1.0` alone INHERITS the −35.0 default: a timeout 35× a clean loss, so the composition advertised as "1 TERMINAL" is really a stall-avoidance objective |
| `VALUE-DIST SUPPORT` | a value-dist head, PopArt OFF, and the achievable returns either fall outside `[vmin, vmax]` or span < 8 atoms | with PopArt off the HL-Gauss target is the RAW return, so the atom support and the terminal are in the SAME units — and under `--value-from-dist` that quantized `E[Z]` **is** the critic feeding GAE |

The third is the one that matters for the registered arm, because the clean/sparse arms run
**without PopArt** (ledger `2d38a4a`) while the gen-17-era shaped runs carried
`value_from_dist=True` over a support of `[−12, +12]` / 51 atoms. ⚠️ **The current production run
carries no distributional head at all** — `--critic winprob` refuses `--value-dist-mode`, so this
guard has nothing to check there; it is a `shaped`-critic guard. A ±1 terminal there lands inside ~4 atoms — a critic quantized
to ~0.5 on a ±1 scale — and nothing downstream tells that apart from a well-fitted one
(`value_dist/mean_abs_err` looks *better* as the support widens). All three are warnings, never
refusals: a launch that works today must not become a `FATAL_CONFIG`. Under PopArt the target is
`popart.normalize(returns)`, the support lives in units of standard deviations, and the guard is
skipped. That was true of every run launched before 2026-09-06; **`--critic winprob` refuses
PopArt**, so on the production run the guard is NOT skipped and the un-normalized branch is the only
branch — which is exactly why `ai_v12_intersection_test.py` §1-§2 pins it.

**The clean-world reward flag set, verbatim** (`CLEAN_WORLD_REWARD_FLAGS` in
`clean_world_config_test.py`; the dense signal is `--win-prob-pbrs-*`, below):

```
--no-hand-shaping --victory-value 1.0 --draw-penalty -1.0
```

Two implementation notes worth keeping:

- **`victory_value` covers the PRE-CAP TIE too.** `finished and not won and not lost and turn < cap`
  shared the decisive-loss branch as a hardcoded `-VICTORY_VALUE`; it now reads the field, so a ±1
  arm cannot score a rare tie at −30 beside a −1 loss. `MAT_HP_WEIGHT` / `MAT_ALIVE_WEIGHT` are
  calibrated against the 30 scale — moot under `--no-hand-shaping`, which is the composition the ±1
  terminal exists for.
- **`--no-pbrs-belief` gates the EMITTED FIELD ONLY.** `_fold_belief_pbrs` also snapshots the
  decision-time KO risk and safe-pivot flag, which the belief-scaled BIAS terms read; the manager's
  standing rule is that a gate skips a COMPUTE, never a cross-turn mutation.
- **The folds and the census are now ONE declaration.** Every `_fold_*_pbrs` calls
  `_hand_pbrs_on(name)` → `_pbrs_term_active`, the same predicate `reward_class_composition` reads.
  They were two hand-maintained copies of the same conditions, which is exactly how a census can
  advertise a composition the folds do not implement.

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

#### The census also drives a fast path — `_active_bias` (`gen3_reward_skip_suppressed_v1`)

`process_turn_reward` used to compute every BIAS helper and then hand the results to
`_apply_pbrs_suppression`, which under the `shaped` default composition zeroes ~20 of them (under
the production `--no-hand-shaping` composition the whole BIAS class and every PBRS fold are off, so
nothing is computed to suppress) — a movedex
walk (`dead_matchup_tax`), two effectiveness loops (`se_switch` / pivot) and a 12-mon status scan,
every decision, for numbers immediately overwritten with 0.0. The manager now derives
`self._active_bias` ONCE at `__init__` **from `_bias_term_active`** — the same function the census
above reads, never a second hand-copied name list (the v79 hand-copied-family-set lesson) — and
`_bias_active("<field>")` gates each pure value computation. Each gate names the field it ASSIGNS,
so a rename breaks the assignment beside it instead of silently un-gating a term.

**It is legal because activeness is a per-run CONSTANT**: every flag `_bias_term_active` reads
(`all_shaping_pbrs`, `stall_pbrs`, `bias_redesign`, `drop_*`, `switch_bias_weight`,
`self_ko_hp_penalty`) is resume-immutable and value-checked by `check_reward_config`, so a
constructor-time active set can never go stale mid-run. Where the mirror is imprecise it errs
ACTIVE (it does not model the progress clock's extra zeroing of repetition/struggle/dead_matchup
under `--bias-redesign`), which costs time, never correctness.

**The cut is COMPUTE-only, never a cross-turn MUTATION.** `_update_opp_se_threat`,
`_compute_spikes_bonus` (`_prev_opp_spikes`), `_compute_status_reward` (`_prev_*_statused`),
`_apply_switch_outcome` (`switch_count` / bounce depth / `_last_switched_from`) and the
`_last_opp_seen_by` update all stay **ungated**, so the manager's observable state is identical
turn for turn whether the skip fires or not. The one exception is `_compute_dead_matchup_tax`,
skipped whole despite mutating `_consecutive_dead_matchup_stays`, because that counter has ZERO
readers outside `reward_manager.py` — suppressed, it is write-only, not observable state.

**Measured** (2026-08-23, order-alternated same-process A/B, both arms on the same decision;
absolutes contaminated by a busy box, ratios are the claim): **~1.08× on `process_turn_reward`**
across four ~1500-decision runs, and a load-free **−20.3% Python calls per call**. Under
`--no-all-shaping-pbrs` (nothing suppressed) the ratio is 0.990× — a no-op, as required. Riding
along: `registry_fields` memoized, `total` summing a cached field-NAME tuple instead of
re-deriving `dataclasses.fields()` (which measured 7.4% of the stage — as much as the whole BIAS
family), and the Φ_opp_boosts/Φ_roar Σ (the same potential at two weights) computed once.

**Gates — this is THE OBJECTIVE, so bit-identity, not approximation.**
`reward_skip_parity_fuzz_test.py` plays real bridge battles and compares EVERY breakdown field
(with `!=`) between the production manager and a `_shadow=True` twin, across the three
compositions **on the same decision stream** — which is also what makes its trigger-coverage table
meaningful, since the `--no-all-shaping-pbrs` arm's firings are exactly what the production arm
skipped. It additionally asserts per turn that the skip is the suppression's exact COMPLEMENT, and
fails INCONCLUSIVE if the corpus never fired the required signals. `GEN3AI_REWARD_VERIFY=1`
(`reward_verify.py`) is the shadow mode: a lockstep full-computation twin asserted bit-identical
every turn — no CLI flag, because the skip is an internal swap and a default branch nobody runs is
the untested one. Derivation pins: `reward_skip_parity_test.py`.

**`reward_tracker.py` parity holds BY CONSTRUCTION** — it builds the same `Gen3RewardManager`
through the same factory with the run's `RewardConfig`, and this change adds no constructor input
the tracker path doesn't thread, so eval traces / falsify / `cf_mc_return` inherit it unchanged.

⚠️ **Do NOT "optimize" the Φ potentials by carrying/telescoping Φ** — recompute-from-the-memoized-
view IS the exactness guarantee, and the reasoning lives in `_pbrs_step`'s docstring where someone
would try it. The one expensive Φ input, `pbrs_belief`'s `encode_block` at **60% of the stage**,
got the safe answer instead — a content-keyed memo, next.

#### And a SECOND fast path when the census reads 0 PBRS + 0 BIAS — `_terminal_only` (`gen3_terminal_only_short_circuit_v1`)

**Under the win-prob arm's composition the reward IS `victory_value · 1{win}`, and almost
everything `process_turn_reward` does is dead.** `_active_bias` above already skips the ~25 gated
BIAS computes, so what was left running was the handful that were deliberately UNGATED *because
their cross-turn mutations feed BIAS terms* — and under a composition with no BIAS terms those
mutations have no reader at all. `_terminal_only` is that second gate, derived in `__init__` from
`reward_class_composition(self.config)` — **the SAME announcer** the startup line, the `reward/`
export and `_hand_pbrs_on` / `_bias_active` already read, never a second `hand_shaping` predicate
(the v79 lesson, one gate over). The shadow twin is excluded, so `GEN3AI_REWARD_VERIFY=1` verifies
BOTH fast paths at once.

⚠️ **The gate is the CENSUS, and `--arm-no-progress-tax` is why that is not pedantry.** It re-arms
`no_progress_tax` alone under `--no-hand-shaping` (design gap B4, the winprob arm's anti-stall
pressure) — so `hand_shaping == False` does NOT imply "no BIAS terms", and a flag-shaped gate would
skip the cross-turn state that one surviving term reads. Pinned as its own test.

**What is SKIPPED, with the reader each was proved dead**: `_fold_belief_pbrs` WHOLE — the
expensive one, since it gates only its emitted field while the `encode_block` above that gate
(historically **60% of this method**) runs unconditionally for two snapshots that feed
`_compute_stay_risk_tax` / `_apply_switch_outcome`; `_compute_spikes_bonus` and
`_compute_status_reward` (their `_prev_*` counters and the `_last_attack_had_effect` assembly reach
only `repetition_tax`); `_opp_active_boosts` / `_update_opp_se_threat` / `_opp_positive_boost_stages`;
the whole voluntary-switch block including the `_last_opp_seen_by` update; and the seven remaining
`_fold_*_pbrs` calls, which were already free early-returns.

**What still RUNS, and its consumer** — the ask was "keep the ones something else reads", so each is
named rather than assumed: `battle.live_view()` (memoized per state-epoch and shared with five
stages — the obs builds it either way, so this is a warm read); `_terminal(live)` (the reward
itself); `_last_breakdown` (`battle_recorder` writes `to_dict()` into every eval-trace decision, and
`reward_verify`'s twin diffs it); `_term_stats.observe` (the `reward/` export); and
`_apply_switch_outcome` **paired with** `_apply_pbrs_suppression` — the first advances
`switch_count` / `last_switch_turn` for the PERIODIC `🏁 Episode Finished` line, the second is what
zeroes the three BIAS fields it writes, so skipping either alone would change the reward. Both are
attribute and dict work; neither walks the board.

#### 🚨 WHAT THE SHORT CIRCUIT MUST NEVER SKIP: an OBSERVATION FEATURE (`gen3_obs_margin_unconditional_v1`)

**`_compute_phi_mat` runs UNCONDITIONALLY, every decision, in every composition** — the one
computation on this page that is not allowed to be skipped, and the census predicate does not gate
it. Its by-product `_last_material_margin` is `gen3_env`'s **`win_margin` OBSERVATION key** (read by
`value_terms._win_prob_loss`'s closeness-stratified metrics and its `skill_vs_material` Brier skill
score), and **an observation feature must not be a function of what the reward is made of.** The
skip rule one section up — *"this mutation's only readers are BIAS terms"* — is the right test for a
cross-turn carry and the WRONG one for anything the obs vector carries, because the obs has readers
the reward composition does not own.

It WAS a function of the reward, for the whole life of the win-prob arm: `_fold_material_pbrs`
early-returned under `--no-hand-shaping` (via `_pbrs_term_active`) *before* computing Φ_mat, so the
margin was pinned at **0.0**. Measured 2026-09-06 on a 6-alive-vs-2 board, shaped `+0.667` against
the winprob composition's `0.0`. Both consequences were silent and both look like measurements:
`|margin| < _WIN_CONTESTED_TAU` was always true, so `contested_frac ≡ 1.0` and every `*_contested`
tag was a byte-identical copy of its pooled sibling; and `P_mat ≡ 0.5`, so `skill_vs_material`
scored the head against a **coin flip** rather than against "just count the mons". (The six tags are
also on the NOISE list in the TensorBoard census above, gated by `_win_prob_loss`'s spread-free
guard — that guard is the *consumer* refusing to publish a degenerate split, and it stays; this fix
removes the degeneracy at the source, so the family becomes real again rather than merely absent.)

**The fix is the idiom `_fold_belief_pbrs` states one method down: the gate wraps the EMITTED FIELD,
never the compute.** Φ_mat is evaluated above the `_hand_pbrs_on("pbrs_material")` early return and
only the PBRS fold is gated, so the reward contribution is unchanged in every composition while the
obs feature is published always. **No margin-only fast path exists and none is wanted** — Φ_mat is
~4 sums over ≤12 mons, so the full potential IS the cheap path.

**Measured cost** (2026-09-06, paired in-process arms on one board, min-of-7 × 4000):
`_compute_phi_mat` is **0.00215 ms/call**, and adding it to a terminal-only decision is
**+0.0035 ms** — ~10–12% of that composition's `process_turn_reward`, ~2% of the shaped
composition's. The short circuit's ~6–7× win is intact (`trainer_turn_benchmark --pin-battles`,
468 decisions / 6 battles, load 33: **0.027 ms before → 0.030 ms after**, against the shaped
composition's ~0.15–0.17 ms; the same pair's untouched `build_delta` line moved 0.062 → 0.047 ms, so
read the paired in-process figure and not the benchmark's spread). ⚠️ **The SHAPED path calls it
exactly once either way** — the call was merely hoisted above a gate that was already passing — so
its cost and its reward stream are unchanged, which the parity fuzz's shaped arm re-proves.

Gates: `reward_terminal_only_skip_test.py::TheObservationMarginIsNotAFunctionOfTheReward` (the
bug's own 6-vs-2 reproduction, agreement across five compositions × five boards including the
`--arm-no-progress-tax` arm, the spread-across-boards property the contested split needs, the
shadow twin agreeing, and the anti-vacuity control — **4 of them verified failing on revert**),
`win_prob_test.py::test_a_REAL_margin_spread_selects_a_STRICT_SUBSET` (the consumer half, driven by
margins the real reward manager publishes) and its flat-margin twin, plus the parity fuzz's four
compositions.

**Measured** (2026-09-06, `trainer_turn_benchmark --decisions 400 --seed 0 --pin-battles`, 468
decisions / 6 battles per cell, load 41–45 on 16 cores so read the same-session RATIO):
`process_turn_reward` **0.150–0.164 ms → 0.021–0.025 ms**, i.e. from ~64% of the shaped
composition's cost to ~14% of it — the reward GROUP falls from 18–19% of our controllable CPU to
**8%**, and what remains is mostly `build_delta`. The **shaped arm is byte-identical and its
timings are unchanged** (0.169 ms both rounds).

**Gates.** `reward_skip_parity_fuzz_test.py` carries TERMINAL-only as a **fourth composition** — the
real-battle oracle, comparing every field of every turn against a `_shadow=True` twin that disables
both fast paths; it is the only thing that can catch a CROSS-TURN skip, since a single turn's
arithmetic looks identical whether or not a snapshot was carried. Measured over 12 battles: **PASS
on all 4 compositions** with full trigger coverage. `reward_terminal_only_skip_test.py` is the
routine-gate half (the census-derived flag over six compositions, the multi-turn stream against the
twin on both arms with an anti-vacuity control, and the `reward/` export reading `n_decisions`
intact with `untracked_abs_mean` exactly 0). Both revert arms verified failing: gating on
`hand_shaping` instead of the census fails on the re-armed tilt, and applying the skip
unconditionally fails the shaped stream on real reward differences.

⚠️ **BE HONEST ABOUT WHAT THE TERMINAL-ONLY FUZZ ARM PROVES, because it is WEAKER than the other
three and for a structural reason.** On the other compositions the twin computes a term the fast
path skipped and the comparison is a live field-by-field check. Here `_apply_pbrs_suppression`
zeroes the whole BIAS class on BOTH sides, so every PBRS/BIAS field is 0.0 either way and that half
of the comparison is satisfied by construction. What it genuinely checks is the TERMINAL fold,
`bd.total`, and that no skipped cross-turn snapshot leaks into a field the suppression does not
cover. **The claim underneath — "every skipped mutation's only readers are BIAS terms" — rests on
the CONSUMER CENSUS, not on the fuzz**: `_prev_active_ko_risk`, `_prev_safe_pivot`,
`_last_attack_had_effect`, `_prev_opp_spikes`, `_prev_our/opp_statused`, `_prev_opp_se_threat`,
`_last_opp_seen_by` and `_prev_opp_boosts` were each grepped tree-wide and read ONLY inside
`reward_manager.py`, only by BIAS-term helpers. **A future term that reads one of them from
somewhere else would not be caught here** — it would have to be caught by that grep being re-run,
which is why the readers are named at each skip site rather than left as "no consumer".

#### The belief-block memo — `IncomingBeliefMemo` (`gen3_belief_block_memo_v1`)

Φ_belief's `encode_block` measured **60.0% of `process_turn_reward`** (`compute_team_block` 42.5 /
`_attacker_threat` 10.8) — the largest single item anywhere in the per-decision CPU budget, and
`reward_manager.py` is the tree's **only** per-decision caller of that pipeline. It now answers
from a per-manager, **content-keyed** cache
(`agents/observation/incoming_damage_encoder.IncomingBeliefMemo`), threaded as
`encode_block(live, memo=…)` and cleared at `reset()` (episode scope).

**Two caches, both on content:**
- `attacker_state_key(live)` → the `AttackerThreat`. The key is `(species, types, move_ids, status,
  atk/spa/spe stages, our reflect, our light screen, weather)` — the opponent active keeps all of
  those for runs of turns.
- `(attacker_key, Defender)` → that mon's `PER_MON` row (`inc.compute_mon_row`, factored out of
  `compute_team_block` for exactly this). `Defender` is a frozen dataclass over primitives rebuilt
  fresh from the board every call, so it **is** its own key — no coverage question on that side.

Plus one algebraic identity in the inner loop: the crit branch computes
`gen3_damage_max(..., screen=False, …)`, which with **no screen up** has argument-for-argument the
same inputs as the modal call, so `dmax_crit == 2·dmax` exactly and the second formula evaluation
is skipped on the overwhelmingly common screenless board.

**Why content-keyed and not `_state_epoch`-scoped.** An epoch key is strictly WEAKER here: same
epoch ⇒ same content ⇒ the content key hits anyway, so the epoch adds no hit it does not already
have — while a per-turn scope would DELETE the cross-turn reuse that is the entire win. The reward
path calls `encode_block` exactly ONCE per decision, so a turn-scoped cache has a structural hit
rate of zero. Content-keying also makes the clone/deepcopy question vanish: unlike a cache keyed on
an object identity or a `battle_tag`, a cross-arm hit here is *correct* rather than a hazard.

**And it is NOT the telescoping Φ that `_pbrs_step` refuses.** Nothing accumulates and nothing is
carried; dropping the memo at any moment changes only speed (pinned:
`test_clearing_at_any_point_changes_nothing`). Content, not history.

**Measured** (2026-08-23, the same order-alternated same-process A/B; box at load 31-36, so
absolutes are contaminated and the RATIO is the claim): **~1.25× on `process_turn_reward`** over
five runs of 2000-5500 paired decisions (1.321 / 1.254 / 1.186 / 1.214 / 1.294), plus a load-free
**−24.0% Python calls per call** (484.2 → 368.1, a `sys.setprofile` count — a different instrument
from the cProfile figure above, not comparable to it). Cache hit rate 48-58% of lookups under
random play. Against the ~23-27% reward share that is ~5-6% of worker CPU.

**Gates.** The key's completeness is proven twice, and neither proof is a reading of the code by
eye: `incoming_damage_memo_test.py` **AST-walks `_attacker_threat`** for every attribute reached
from `live` (including through `getattr`) and fails if one appears that the key does not carry —
the "enumerate the doors, not the reads you noticed" discipline of the `live_view` epoch memo,
applied to a pure function — and the same file pins `AttackerThreat`'s field list, since the proof
is a claim about that constructor's arguments. `reward_skip_parity_fuzz_test.py` adds the
**differential** half on real boards: every decision's key is recorded against the freshly-derived
belief, and a key seen twice must carry the identical belief — 3,956 repeat-key tests over 1,003
distinct keys in a 4,959-decision run (verified failing: a deliberately under-keyed build reports
`two boards share attacker key … but differ on ['spa_tail','spa_mean']`).
That same fuzz covers the memo end-to-end **by construction**, because `_shadow=True` means
"compute everything the slow way": the twin runs with the skip disabled AND `_belief_memo = None`,
so `GEN3AI_REWARD_VERIFY=1` is a live per-field test of key completeness in any run. Memo hit
counts are printed and FLOORED — a clean run in which the cache never served anything would be
evidence about the uncached path only.

#### `reset()` clears every `_prev_phi_*`, DERIVED (`gen3_prev_phi_reset_v1`)

`Gen3RewardManager.reset()` cleared its PBRS carry-overs from a hand-written list, and that list
omitted `_prev_phi_roar` for the whole of its life: eight potentials declared in `__init__`, seven
cleared, nothing anywhere to notice. The set is now derived from the instance (`_prev_phi_fields`),
and the pin (`reward_manager_test.py::test_reset_clears_every_prev_phi_potential`) derives it the
same way — a hand list in the test would have passed the whole time too.

**The leak was benign by COINCIDENCE, not by contract**, which is why it is fixed rather than
documented. `_pbrs_step` zeroes Φ at a terminal fold, so a normal episode end leaves
`_prev_phi_roar == 0.0`; the next episode's first window then charges `γ·Φ_roar(s₁) − 0.0` where
the correct fresh start (`prev is None`) charges `0.0`. Equal only when `Φ_roar(s₁) == 0` — and s₁
is the board **after turn 1 resolves**, not the opening board, so an opponent that opens with
Dragon Dance / Calm Mind breaks it. Measured over random-play bridge battles: non-zero at the first
window in **3/185 battles (1.6%)**, a one-off ≈ −0.25 per positive boost stage — rare under random
play, and a trained boost-opener raises it. The second channel the
coincidence never covered at all: a `reset()` that lands MID-battle leaves the last **non-terminal**
Φ_roar, an arbitrary value. (Before `--all-shaping-pbrs` became the default in 2026-08 the fold
never ran at all, so `_prev_phi_roar` stayed `None` and there was nothing to leak.)
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
