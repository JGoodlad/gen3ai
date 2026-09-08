---
name: project_markovian_reward_design
description: "Markovian/PBRS reward redesign — IMPLEMENTED 2026-06-07 (gen3_markovian_progress_v1); EXTENDED 2026-06-12 to the all-shaping-PBRS end state: v12 de-bias drops + v13/v14 four new potentials behind TWO flags (--all-shaping-pbrs 'everything but stall' + --stall-pbrs 'stall'); both ON ⇒ zero BIAS"
metadata: 
  node_type: memory
  type: project
  originSessionId: 27395eb0-6bae-4f0e-90b0-6883176d557e
---

> **Archived 2026-09-08** — PBRS/hand shaping is DELETED under the ai_v12 win-prob critic (--no-hand-shaping). Preserved verbatim; nothing below is current.

**IMPLEMENTED 2026-06-07** (branch keen-clarke-9f8edf, not yet shipped). ARCH `gen3_markovian_progress_v1`,
obs 3390→3391, MODEL_CONFIG_VERSION 3→4. Verified: full unit suite 1919 pass; obs benchmark ~7173
calls/encode (below the 7.3k ref, no regression); golden regenerated + parity passes; 23 new
reward_redesign_test.py pass (telescoping/terminal-zero/bias-no-op/blend/ProgressClock); GPU smoke
round-trip PASSED + episodes complete (late GPU OOM = contention with the live run, not a bug). New file
progress_clock.py; touched reward_manager.py, episode_tracker.py, gen3_env.py, constants.py, reactive.py,
state_encoder.py, model_version.py, snapshot.py, train_rl_agent.py. DEFAULT = single-variable run
(--bias-redesign OFF → only the material clutch-fix changes vs baseline; the clock tracks the obs scalar
but charges nothing). DEFERRED (Markovian-purity polish, no default-run effect): roar/status _prev_*→
current-obs reframes, se_switch/switch_base hidden-gate drops.

`designs/ai_v5/design_markovian_reward_and_features.md` (written 2026-06-07): redesign every
`reward_manager.py` term to be either **PBRS** (objective-neutral) or **Markovian-w.r.t.-the-obs**
(a clean obs-keyed bias), + the feature-encoder enrichment each Markovian term needs. **Status:
design only, NOT implemented.** Adversarially red-teamed by a Workflow (5 analyses + 3 red-teams,
8 opus agents) before writing; §9 is the findings ledger.

**Locked decisions (the non-obvious ones the red-team forced):**
- **Φ_mat = 2·(Σour_hp − Σopp_hp) + 1.25·(n_alive_our − n_alive_opp)** computed over the **DECLARED
  team size** (unrevealed opp mons = full-HP-alive). The declared-team trick kills two bugs at once:
  the reveal-jump (opp HP is %-based + revealed-only) AND the Φ_mat(s_0) value-loss-target variance
  (declared → Φ_mat(s_0)≈0). MAT_ALIVE_WEIGHT=1.25 = FAINT_BASE+FAINT_MATERIAL_PENALTY (stated
  invariant, not midpoint-feel).
- **Removes the −0.75 FAINT_MATERIAL_PENALTY** (preservation is only instrumental; +30 + Φ_mat
  density teach it). Faint pain stays *immediate* via Φ_mat's alive-term drop (NOT via the +30
  terminal — gae_lambda=0.80 = ~5-step horizon argues against that).
- **PBRS telescoping/density is indexed over DECISION WINDOWS, not game-turns** (a faint splits a
  turn into ≥2 process_turn_reward calls). The terminal window MUST zero Φ_mat(s′) via is_terminal→0
  or a 6-0 win injects a +19.5 dominant-win bonus (highest-risk silent bug, first guard test).
- **Anti-spam collapse**: repetition/bouncing/dead-matchup/struggle/stall_tax → ONE Markovian
  `turns_since_progress` clock (new obs scalar, REACTIVE_SCALAR_DIM 14→15, obs 3390→3391). Progress
  = OUR-attributed damage ≥3% (NOT net opp_hp_delta — Sandstorm/Leech-Seed reset it for free), OR
  status-landed event, OR hazard layer added, OR forced opp commit; setup is NOT progress (closes
  the CalmMind↔Recover loop). Front-load shape: **FLAT c=0.15 recommended** over the log shape
  (gae_lambda=0.80 makes log's decaying marginal read as "continuing gets cheaper"); log is fallback.
- **Clock plumbing**: EpisodeTracker-owned (HiddenPowerTracker precedent — NOT LiveView, which is
  current-board-only), updated in `record()` (embed time) NOT process_turn_reward — because env.py
  runs embed_battle (:591) BEFORE calc_reward (:600), so updating in calc_reward = stale obs / off-by-one.
- **`pbrs_material` field is MIS-NAMED** (it's the shipped belief PBRS, not material). Rename →
  `pbrs_belief`; new `pbrs_material` = Φ_mat. Φ_total = Φ_mat + Φ_belief telescopes as a sum. The
  rename is a recorded-schema migration (eval_traces/prober/TD-residual), not pure.
- **NO reward annealing** ([[project_popart]] doc replaced): PBRS is de-biased from step 1 by
  construction, nothing to anneal. Pair the run with --use-popart (Φ_mat density feeds value target).
- **PBRS_GAMMA==model.gamma is genuinely unguarded** today (shipped doc falsely claims the assert
  exists); can't be a reward-manager constructor assert (model doesn't exist yet) — thread post-build.

**Refinements added in the same session (user-driven, in the doc §2.6/§2.7/§4.1.1):**
- **spikes + status → PBRS too** (not just the base spine). The +0.5/layer spikes credit and the ±0.3
  status credit are "credit-assignment bridges" that DOUBLE-COUNT the realized chip Φ_mat already
  rewards. Split each: standing-value credit → a PBRS potential (Φ_hazard, Φ_status; telescopes to 0,
  no double-count, no bias), wasted-action half → a Markovian futile bias (futile_spikes; status_wasted).
  Φ_total now = Φ_mat + Φ_belief + Φ_hazard + Φ_status (4 potentials, 4 _prev_phi, 4 pbrs_* fields).
  Dividing line made explicit: value eventually realized materially (HP/win) → PBRS; value that is ONLY
  the immediate nudge (anti-spam clock, futile family, switch tilts) → Markovian bias. Caveat: status is
  heterogeneous — Toxic/burn are material (clean), but par/slp/frz are non-damaging tempo (diffuse, only
  partly material) → PBRS valid but signal-weaker; guard = status-application-rate-must-not-collapse,
  else restore a small bias for non-damaging statuses only.
- **Misses are a THIRD clock outcome (FREEZE), not a no-op.** Progress predicate is now ternary:
  PROGRESS (reset) / DENIED (freeze, no charge) / NO_OP (increment+charge). A missed/Protect-blocked/
  full-para-prevented attempt at a damageable target = DENIED = freeze (attempted progress denied by
  RNG/opp, not stalling — don't punish uncontrollable variance, don't reset either). Deterministic no-ops
  (immune/capped/redundant/Spikes-at-3) still increment. Same exemption added to futile_attack (today it
  wrongly taxes a miss −0.05). Safe in gen3ou because Evasion Clause bans Double Team → a miss is always
  accuracy-based, no "spam a missing move to freeze forever" surface.
- **Defensive/support edge cases (§4.1.2) — the predicate was offense-only.** Governing rule: the clock
  stays OUT of anything a Φ potential already prices (no double-count). Productive Recover/Wish→Φ_mat,
  Aromatherapy/Rest-cure→Φ_status → FREEZE; wasted ones (full-HP heal, no-target cleric) → INCREMENT +
  futile. Seismic Toss/fixed-damage: connecting = progress; into Ghost = futile_immune — BUT today's
  futile_attack base_power==0 gate WRONGLY exempts it (fix: use incoming_damage.FIXED_DAMAGE set). Rest's
  asleep turns freeze (cant=slp); Sleep-Talk random call = reset if dmg else freeze. Failed Focus Punch =
  freeze via cant=focuspunch (misplay cost already priced by Φ_mat — we ate a hit for nothing).
- **REVERSED the stall_tax decision:** the clock is offense-centric and CANNOT catch DEFENSIVE stalls (a
  Recover-war under Sandstorm freezes through the clock). So RETAIN a gentle absolute-turn stall_tax
  (re-tuned to ~−10, its real integral today is −21.3 not the −10 its comment claims) alongside the clock
  + turn-250 forfeit. Earlier draft said "replace stall_tax"; the edge cases vindicate keeping it
  (complementary: clock = active wheel-spin, stall_tax = defensive/global length pressure).

**Reward REGISTRY + bias-additivity FLAG + material RESOLVED (session 2026-06-07, doc §1.1–1.3):**
- **Registry = single source of truth.** Every reward component is one entry {name, class, compute};
  manager folds by iterating the registry + per-class treatment (no per-term special-casing).
  RewardBreakdown/prober telemetry derived from it. Adding a reward = one entry.
- **Three classes:** TERMINAL (±30, untouchable), PBRS (always telescoping, always objective-neutral —
  Φ_mat + the shipped Φ_belief ONLY), BIAS (flag-controlled additive↔telescoping).
- **Material RESOLVED = always-on, NOT flag-gated.** Dropped --use-material-pbrs. The clutch-fix (every
  win +30, loss −30) is the deliberate default. Φ_mat lives in PBRS class unconditionally.
- **--bias-additivity (config bias_additivity), float[0,1], DEFAULT 1.0, affects ONLY BIAS class.**
  Accumulate-and-refund (NOT per-turn scaling, so λ=1 is a byte-exact no-op for event terms): each BIAS
  term emits its current per-turn value; refund −(1−λ)·acc → episode contribution λ·acc. 1.0=additive
  (=today), 0.0=fully telescoping(PBRS hint), λ=blend. Low-variance refund via the term's running-
  accumulator potential (not a terminal lump). Resume-immutable, fresh-only, ModelVersion-checked,
  per-run CONSTANT (not annealed) — IT IS the de-bias knob annealing wanted, but stationary.
- **spikes/status RECLASSIFIED from PBRS to BIAS** (telescoping form Φ_hazard/Φ_status reached at λ→0;
  default λ=1 = today's additive +0.5/layer, ±0.3). The bias-additivity flag is exactly the dial for
  the spikes/status double-count-vs-exploration judgment. So always-on Φ_total = Φ_mat + Φ_belief ONLY.
- **THE DEFAULT RUN = single-variable change** (only material clutch-fix; BIAS byte-identical to today)
  for clean attribution vs the live baseline. The bias redesign (clock replacing anti-spam, reframes,
  spikes/status de-bias) is STAGED — later A/B arms via (a) lowering bias_additivity or (b) enabling the
  redesigned BIAS entries. Proof = 2 offline replays: BIAS no-op (must be identical at λ=1) + MATERIAL
  clutch-fix (returns must collapse toward ±30 — SHOULD differ). New tests: registry-coverage (every
  field→1 entry/class), parameterized telescoping (episode-sum==λ·acc for λ∈{0,0.5,1}).

Coordinates with (does not redo) the shipped switch PBRS + re-gate (design_reward_switching.md,
commit 7483dd1). Builds on [[project_incoming_damage_outcome]] (the belief block Φ_belief reads) and
[[project_reward_shaping_verification]] (reward changes need retrain to measure → offline replay
falsifier over eval_traces as a pre-train proxy). Next step: implement + smoke + fresh A/B run.

**UPDATE 2026-06-08 — SHIPPED + RAN as the single-variable baseline (`bias_redesign` OFF), and the
`bias_redesign`-ON consequence discovered.** The markovian impl shipped and the single-variable run was
`models/ai_v5_5_popart_N_0607` (PopArt + self-play, `--no-bias-redesign --switch-bias-weight 0
--bias-additivity 1`). **KEY FINDING (the cost of staging it OFF):** with `--no-bias-redesign`,
`reward_manager._apply_progress_clock` early-returns → the `turns_since_progress` clock fed ONLY the obs
scalar and **charged ZERO reward all run** (the FLAT-0.15 NO_OP charge never reached the objective). So
the run produced **2247 stall files = 100% self-play MIRROR PP-exhaustion draws** (both bulky teams
Recover/Rest forever to the 250-turn cap). Two structural defects even when ON: the **DENIED-freeze**
(productive heal → frozen) lets a mutual heal-war run uncharged; the cap is a **forfeit-LOSS not a tie**
(gen3_env `ForfeitBattleOrder` at turn≥250 → `lost=True`, already −30, so it wasn't a draw-banking
incentive — discounting an inevitable −30 to turn 250 is the weak pull). **Anti-stall fix SHIPPED this
session** (d7aa983 + fuzz guard a11f234) — heal-war `HEAL_FREEZE_GRACE` streak cap + our-owned-residual
PROGRESS credit (so a WINNING Toxic/Leech stall is never charged) + a new `--draw-penalty` timeout
terminal (keyed on turn≥cap, `MODEL_CONFIG_VERSION 6→7`): full detail in [[project_anti_stall_fix]].
**Next staged run = `--bias-redesign --draw-penalty -35`** (turns the full redesign + anti-stall ON;
note `--bias-redesign` flips the WHOLE redesign: clock charge + Φ_status PBRS + status/switch reframes,
not just anti-stall). Run review: [[project_popart]] (PopArt validated) + [[project_incoming_damage_outcome]]
(threat-response).

**UPDATE 2026-06-12 — the ALL-SHAPING-PBRS end state, BUILT + tested (branch happy-morse-55fea2, NOT
shipped, NOT run).** Goal (user, repeated): make ALL shaping policy-invariant PBRS so it's "learnable not
biasing." Two stages landed:
- **v12 de-bias drops** (`MODEL_CONFIG_VERSION 11→12`): `--drop-redundant-bias` zeros `stall_tax` +
  `matchup_penalty` (redundant w/ the clock+`--draw-penalty` and `pbrs_belief`); `--drop-switch-bias`
  zeros the hand-coded switch subsidy (`switch_base`/`switch_bouncing_tax`/`escape_threat_switch`/
  `se_switch`/`pivot_*`/`sleep_*`). Both default OFF (byte-identical), resume-immutable, value-checked.
  Audit finding: distortion lives in a FEW terms, NOT the global λ knob; the historical worst distorter
  `finishing_blow`-on-self-KO was ALREADY fixed (guarded + +2.0 literal deleted).
- **v13/v14 PBRS completion** (`MODEL_CONFIG_VERSION 12→14`, no `ARCH_SIGNATURE` bump — reward-value only).
  FOUR new telescoping potentials mirroring `_compute_phi_mat`/`_pbrs_step`: **Φ_progress** =
  −`no_progress_penalty`·`progress_clock.value()` (reuses the obs clock), **Φ_hazard** =
  `HAZARD_WEIGHT`·(opp−our spikes, design §2.6), **Φ_boost** = `BOOST_WEIGHT`·Σmax(0,our-active boost)·hp,
  **Φ_opp_boosts** = −`OPP_BOOST_WEIGHT`·Σmax(0,opp-active boost). Behind **TWO flags** (user-specified
  "1 for everything but stall, 1 for stall"): **`--all-shaping-pbrs`** folds hazard/boost/opp_boosts +
  Φ_status (gate now `bias_redesign OR all_shaping_pbrs`) and **zeros EVERY BIAS term except the kept
  anti-stall tilt `no_progress_tax`**; **`--stall-pbrs`** folds Φ_progress and zeros
  `no_progress_tax`+`stall_tax`. **Both ON ⇒ entire BIAS class = 0 → TERMINAL+PBRS only.** Verified e2e
  (per-mode BIAS sum: default −0.01{stall_tax} / all-shaping-only −0.15{no_progress_tax} / both
  +0.00{}); 224 targeted tests pass; potentials telescope to net-0 + Φ(terminal)=0 + byte-identical default.
  The 4 potentials were built by a research+coverage+verify Workflow (14 agents, ultracode); I then SPLIT
  the workflow's single flag into the 2-flag design.
- **KEY CAVEAT (the anti-stall fork):** Φ_progress telescopes to net-0 (policy-invariant) → the actual
  "timeout = loss" objective is the `--draw-penalty` TERMINAL, NOT the shaping. `--all-shaping-pbrs` alone
  keeps `no_progress_tax` as the ONE acknowledged BIAS tilt (insurance vs the stall-regression risk from
  [[project_anti_stall_fix]]); flip `--stall-pbrs` for fully-PBRS but **watch the stall-rate canary**.
- **End-state launch:** `--all-shaping-pbrs` (everything-PBRS-except-stall-tilt) or
  `--all-shaping-pbrs --stall-pbrs` (fully PBRS). Fresh run only (reward change → resume FATALs by design).
  Measure on ELO/`td_resid_tail`/stall-rate, NOT saturated vs-bots. Frontier per
  [[project_model_frontier_roadmap]]: once reward is clean-PBRS, the levers are obs-coverage + distributional
  critic + league — NOT more reward terms (that lever is spent).
