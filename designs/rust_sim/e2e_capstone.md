# The E2E capstone — the generator, the modeled allow/blocklist, the gate, and what it found

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

The 220-battle real-team capstone (`harness/gen_e2e_fuzz.js` -> `tests/vectors/e2e_fuzz_golden.txt`
-> `tests/e2e_fuzz_test.rs`) is the port's closure gate: STRICT `filtered_diverged == 0` per
decision boundary to game-end. This document holds the generator's construction, the four modeled
move/ability/item sets that define the FILTERED gate, the per-batch admission record, the engine
bugs the capstone surfaced, and the coverage taxonomy's honest scope. The leaf keeps the
invariant, the coverage floors and the run command.

---

- **The generator** `harness/gen_e2e_fuzz.js` GLOBS `data/teams/*.txt` (the sample/ + others/
  pools) — 813 files as of 2026-09-07, 770 when the committed golden was generated — imports each
  with the real `Teams.import`, **validates it under gen3ou** (skips rejects/import-fails: 762
  valid today, 719 then), and packs it (the EXACT bytes `team::unpack` ingests). From a
  fixed **`MASTER_SEED`** it pairs distinct teams + a battle seed, and at EACH decision reads the sim
  request and picks a RANDOM legal choice from a SEPARATE seeded **choice-RNG** (mulberry32, recorded
  via `MASTER_SEED` so a failing battle re-runs deterministically) — RESTRICTED to mechanics the port
  models (a damaging fixed-BP move with a modeled secondary shape, **OR a modeled standalone status
  move, OR a modeled SELF-BOOST SETUP move**; else a SWITCH; else the battle is forced-unmodeled →
  dropped). It records the SAME per-decision golden as `gen_fullbattle_golden.js` + the boosts/confusion
  of `gen_secondary_golden.js` + a `statusMove` AND a `setupMove` flag per decision (so the run reports
  how many decisions exercise the status-move / setup-move layer), plus the two packed teams + the
  compact choice tokens, to `tests/vectors/e2e_fuzz_golden.txt`.
  **The battle FORMAT is `gen3customgame`** (no clauses → no SetStatus handler-sort shuffle), so the
  Rust gate uses `format_id: "gen3customgame"` (`sleep_clause` OFF) — a status move here draws accuracy
  + apply (+ sleep `random(2,6)`) but NOT the gen3ou SetStatus shuffle.
- **The modeled allow/blocklist** (the FILTERED gate): a move is modeled iff (damaging + fixed BP +
  a modeled secondary shape — none / single status / flinch / confusion / structured stat-boost, or
  Tri Attack) **OR a MODELED STANDALONE STATUS MOVE** (`MODELED_STATUS_MOVES`: Thunder Wave / Stun
  Spore / Glare / Poison Powder / Poison Gas / Toxic / Will-O-Wisp / Spore / Sleep Powder / Hypnosis /
  Sing / Lovely Kiss / Grass Whistle) **OR a MODELED SELF-BOOST SETUP move** (`MODELED_SETUP_MOVES` —
  DERIVED from the Rust data's `selfBoosts` so it's GIGO-proof in lockstep with the engine's
  `self_boost_spec`: Calm Mind / Dragon Dance / Swords Dance / Agility / Bulk Up / Amnesia / Barrier /
  Acid Armor / Iron Defense / Cosmic Power / Tail Glow / Meditate / Sharpen / Howl / Harden / Withdraw /
  Growth) **OR a MODELED SELF-HEAL RECOVERY move** (`MODELED_RECOVERY_MOVES`: Recover / Soft-Boiled /
  Slack Off / Milk Drink / Moonlight / Synthesis / Morning Sun / Rest / Splash — kept in lockstep with the
  port's `recovery_heal_amount` + `run_rest` arms) **OR a MODELED PROTECT move** (`MODELED_PROTECT_MOVES`:
  Protect / Detect — kept in lockstep with `run_protect`; Endure + the gen4+ Quick/Wide Guard / King's
  Shield stay excluded) — `isModeledMove` returns true for a category-Status move iff it is in ANY of the
  four sets; every other Status move — incl. Defense Curl/Minimize/Double Team [volatile/evasion], Belly
  Drum, Curse, **Wish / Heal Bell / Aromatherapy / Refresh / Leech Seed / Endure** — stays excluded), AND
  not `basePowerCallback` /
  `beforeTurnCallback` (Focus Punch) / `beforeMoveCallback` / `priorityChargeCallback` / `damageCallback`
  / `onModifyMove` (Thunder-in-rain) / ohko / multihit / recoil / drain / selfdestruct / forceSwitch /
  a top-level `self.boosts|volatileStatus` (Overheat/Outrage selfDrops) / Hidden Power (variable) / an
  explicit id blocklist; plain `priority` is fine (the port reads it). EVERY mon on BOTH teams must hold a MODELED ability (Intimidate / Sand Stream /
  Drizzle / Drought / Levitate / Flash Fire / Water&Volt Absorb / Thick Fat / Clear Body / White Smoke
  / Hyper Cutter / Keen Eye / Serene Grace / Shield Dust / Own Tempo / **Natural Cure** / **the STATUS_IMMUNE class** (Limber / Insomnia / Vital Spirit / Immunity / Water Veil / Magma Armor — `gen3_status_immune_v1`) / **the BATCH-1 classes** (Shell Armor / Battle Armor [CRIT_IMMUNE] / Chlorophyll / Swift Swim [WEATHER_SPEED] / Cloud Nine / Air Lock [WEATHER_NEGATE] / Speed Boost / Rain Dish [RESIDUAL] — `gen3_ability_batch1_v1`)) OR a provable
  no-op (Pressure / Sturdy / Rock Head / Early Bird / **Lightning Rod / Sticky Hold** [batch-1 class-(a)] / … in a damaging-move-only fuzz — **Plus / Minus moved OUT of the no-op set 2026-07-10**: the gen3 `onModifySpA` scans `getAllActive()` FOES INCLUDED, so a cross-field Plus↔Minus pair is SpA ×1.5; now MODELED, `gen3_plus_minus_v1`), and a MODELED item
  (none / Leftovers / Choice Band / type-boost ×1.1 / Sea Incense / Quick Claw). Real gen3OU teams are
  saturated with Natural Cure / Immunity / berries, so — with **`gen3_natural_cure_v1`** now admitting
  Natural Cure (naturalcure=254, the #1 team-carry gap) — **449 of 719** teams are fully filter-clean (was
  151 after `gen3_sun_freeze_immunity_v1` admitted the 8 DMG_MOD abilities; 22 before that; 18 before
  `gen3_trapping_v1` admitted arenatrap/magnetpull). **Admitting Natural Cure was the BIGGEST single
  admission lever yet (151 → 449)** — the gate pairs those with varied seeds/choices. The MODELED ABILITY
  set ALSO includes the DMG_MOD family (Torrent / Blaze / Overgrow / Swarm / Huge Power / Pure Power / Guts
  / Marvel Scale — `gen3_item_mechanics_v1` ability side, `gen3_sun_freeze_immunity_v1` admission) + the
  ACCURACY family (Compound Eyes / Sand Veil / Hustle — `gen3_accuracy_pipeline_v1`). **Natural Cure** (the
  sole gen-3-REACHABLE SWITCH_OUT-cure ability — Regenerator/Zero-to-Hero also carry `onSwitchOut` but no
  gen≤3 species can have them, so the `naturalcure`-only gate is complete for gen3) is MODELED in
  `turn.rs::execute_switch` — an alive outgoing
  `naturalcure` holder's major `status` is cleared to `None` (voluntary pivot OR phaze-drag; the tox stage +
  sleep counter reset), DRAW-FREE (`onSwitchOut`, `onCheckShow` undefined — the cure + its `[silent]`
  `-curestatus` reveal consume ZERO PRNG, so it is seed-neutral; probe-settled by
  `harness/probe_naturalcure_rng.js`), validated by `tests/naturalcure_test.rs` (280 game-end battles) +
  the NC1-NC3 pins.
- **The gate** `tests/e2e_fuzz_test.rs::e2e_fuzz_golden_matches_showdown` seeds a `BattleState` ONCE
  at the sim's pre-first-decision PRNG state and replays the recorded choices via `run_full_battle`
  WITHOUT re-seeding, asserting per DECISION BOUNDARY to game-end: active species/hp/maxhp/fainted/
  **status** + 5 **boost** stages + **confusion** + pokemon_left + request kind + first mover, AND the
  post-decision PRNG **seed** == the sim's, PLUS the final **winner**. **Result (ALL move layers now
  INCLUDED — `SPIKES`/`SUBSTITUTE`/`EXPLOSION`/`PHAZE` all `*_E2E_EXCLUDED = false` + the
  `gen3_taunt_disable_v1` `MODELED_RESTRICTION_MOVES` admission + the `gen3_trapping_v1`
  `arenatrap`/`magnetpull` ability admission + the `gen3_sun_freeze_immunity_v1` 8-ability DMG_MOD
  admission + the `gen3_natural_cure_v1` **Natural Cure** admission + the `gen3_status_immune_v1`
  6-ability STATUS_IMMUNE admission + the **`gen3_ability_batch1_v1`** batch-1 admission
  (CRIT_IMMUNE `shellarmor`/`battlearmor`, WEATHER_SPEED `chlorophyll`/`swiftswim`, WEATHER_NEGATE
  `cloudnine`/`airlock`, RESIDUAL `speedboost`/`raindish` + the class-(a) no-ops
  `plus`/`minus`/`lightningrod`/`stickyhold`) + the **`gen3_ability_batch2_v1`** batch-2 admission
  (CONTACT_PROC `static`/`poisonpoint`/`flamebody`/`effectspore` + CONTACT-recoil `roughskin` +
  BLOCK `soundproof`/`damp`/`suctioncups` + `synchronize`) + the **`gen3_berry_trace_shedskin_v1`**
  batch-3 admission (the 22 `berryEffect` berries → MODELED_ITEMS + `trace`/`shedskin` →
  MODELED_ABILITIES, `TRACE_COPYABLE` in lockstep) + the **`gen3_ability_batch4_v1`** batch-4
  admission (`truant`/`innerfocus`/`shadowtag`/`cutecharm`/`colorchange` → MODELED_ABILITIES +
  `kingsrock`/`focusband` → MODELED_ITEMS, `TRACE_COPYABLE` in lockstep) + the **MOVE-COVERAGE
  BATCH 1** admission (`gen3_move_coverage_batch1_v1`, 2026-07-12 — the DRAW-FREE post-hit effects
  RECOIL/DRAIN/SELF-DROP/ITEM-REMOVAL/RAPID-SPIN via `MODELED_{RECOIL,DRAIN,SELFDROP,ITEM_REMOVAL,
  RAPIDSPIN}_MOVES`, growing the filter-clean pool **719 → 722**, a CLEAN STRICT pass first-try, NO
  new engine bug; the pre-regen golden replayed BYTE-IDENTICAL [md5 `a23d77ac60d4af168b8a4428f0b465c9`
  UNCHANGED] then the deliberate regen shifted it to **`dac97afb25317cc9def204ccc9af0e8d`**),
  ZERO exclusions remaining): 220
  battles, ALL 220 bit-for-bit clean (`filtered_diverged == 0` over EVERY battle — STRICT, no escape
  hatch), **11049 decisions** to game-end, of which 4210 USE SPIKES, 353 USE PHAZE, 612 USE EXPLOSION,
  343 USE SUBSTITUTE, **114 USE TAUNT**, and **201 involve
  a TRAPPED mon** (an Arena-Trap/Magnet-Pull
  trapped active at a move boundary — the switch-legality fact + the mirror tie-shuffle draws replay
  bit-for-bit on real teams; **0 USE DISABLE** — no sample team carries Disable, the
  honest disclosure; the newly-admitted TRUANT teams are IN the corpus [Slaking's loaf/toggle rows
  replay bit-for-bit], while the 2 Inner-Focus teams are filter-clean but weren't DRAWN into the 220
  sample — Inner Focus is exercised by the dedicated batch-4 golden instead) (plus the
  status/setup/recovery/protect-move coverage + the
  batch-2 CONTACT_PROC / BLOCK / SYNCHRONIZE decisions + the batch-3 berry eats / trace copies /
  shed-skin residual rolls; the
  per-side spikes layers + the sub absorb + the phaze DRAG species + the explosion self-KO + the
  taunted + trapped flags all asserted bit-for-bit via the active species/HP + seed), 218 wins + 2
  ties, 1705
  forced-switch. The decision totals shifted again because admitting the batch-1
  / batch-2 / batch-3 / batch-4 classes (+ move-coverage batch 1) grew the filter-clean team pool
  **525 → 571 → 585 → 712 → 719 → 722 / 722** (shellarmor
  the big batch-1 lever; batch-2 +14, `synchronize` [the #1 taxonomy gap] + `effectspore` the levers;
  **batch-3 +127, the biggest admission since Natural Cure** — `lumberry`=64 + `salacberry`=46 +
  `trace`=9 the levers; **batch-4 +7 — `truant`=4 + `innerfocus`=2 were the LAST team-carry gaps: the
  ENTIRE 719-team real-gen3ou pool is now filter-clean, and the honest taxonomy's 300-battle
  UNFILTERED sweep is 300/300 clean with an EMPTY ability+item gap list for the first time**)
  hence the battle sampling — a CLEAN STRICT pass first-try for ALL FOUR batches, NO new engine bug (the
  STEP-1 weather-eachEvent fix + the DRAW-FREE / draw-bearing class models composed cleanly; the
  batch-2 CONTACT_PROC `randomChance` draws AFTER the move secondary, Damp/Soundproof/Suction Cups draw
  fewer, Synchronize's reflect is draw-free in customgame; the batch-3 eat is DRAW-FREE at the
  Leftovers residual slot, Starf's `sample` + the Figy confusion + Trace's n=1 `randomFoe` + Shed
  Skin's `randomChance(33,100)` the only new draws; the batch-4 Truant loaf/toggle + Shadow Tag +
  Color Change are draw-free-or-fewer while the KR/FB/CC procs ride probed positions — all
  seed-faithful). The golden
  is byte-reproducible at the committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220).
  **UPDATE (2026-07-14) — MOVE-COVERAGE BATCH 5 ADMITTED (`BATCH5_E2E_EXCLUDED = false`).**
  Counter / Mirror Coat / Endeavor + Return / Frustration / Flail / Reversal / Low Kick are IN
  the allow-list (`MODELED_BATCH5_{REACTIVE,VARBP}_MOVES`), Sleep Talk is CARRIER-conditionally
  pickable (`sleepTalkPoolModeled`), and the blocklist un-shadowing finally lets the modeled
  FIXED-DAMAGE five be PICKED. After the ONE e2e-surfaced fix (MC76 — a fixed-damage hit sets
  the Focus-Punch `lostFocus`; see the batch-5 section) the regenerated golden is **md5
  `614d47b9a5227dc7ad4e444b2e28313c`** (722/722 filter-clean teams, STRICT
  `filtered_diverged == 0` over **220 battles / 11825 decisions**, 219 wins + 1 tie; **240
  BATCH5-MOVE decisions + 271 FIXED-DAMAGE-MOVE decisions** — the fixed-damage family is
  finally EXERCISED on real teams, closing the old "0 fixed-damage decisions" disclosure.
  The review pass appended two per-DEC columns — `fixedMove[36]` / `batch5Move[37]` (DEC is
  now 38 fields) — so `e2e_fuzz_test.rs` gates **GATED coverage FLOORS**
  `fixed_damage_decisions >= 50` + `batch5_decisions >= 50` (teeth-verified: zeroing the
  flags fails the gate) instead of a generator statistic; the regen was verified
  CONTENT-IDENTICAL to the batch-5 golden (md5 `dd0664015014d7e473e400a29a47eed2`) modulo the
  appended columns + header comments (`ab_replay` accepts both 36- and 38-field DEC rows, so
  saved repro dirs stay replayable) —
  5876 STATUS-MOVE, 1020 SETUP, 916 RECOVERY, 654 PROTECT, 629 PHAZE-move [629 selected; 463
  drag decisions], 240 LEECH, 367 SUBSTITUTE, 140 TAUNT/DISABLE, 366 EXPLOSION; taxonomy
  300/300 clean, EMPTY gap lists). The handler-audit manifest grew 787 → **815 rows** (the
  counter/mirrorcoat conditions + the endeavor/sleeptalk hooks + the un-shadowed `damage`
  declaratives; `sleeptalk` is added to the audit surface EXPLICITLY since `isModeledMove`
  deliberately rejects it). The coverage scan (`scan_move_coverage.js`, classifier refreshed
  for batches 4/4b/4c/5): **662 / 722 teams fully engine-playable**, 0 MISMODELED moves left.
  **UPDATE (2026-07-14) — MOVE-COVERAGE BATCH 4c ADMITTED (`BATCH4C_E2E_EXCLUDED = false`).**
  Hyper Beam / Solar Beam / Doom Desire / Future Sight (+ the locked-turn `recharge` pseudo-move)
  are IN the allow-list (`MODELED_BATCH4C_MOVES`; `futuresight`/`doomdesire` OUT of
  `MOVE_ID_BLOCKLIST`; a belt-and-braces `flags.futuremove` reject keeps any future futuremove out;
  `pickMove` treats a locked `trapped:true` REQUEST as trapped so the capture never stalls on a
  doomed switch). A CLEAN STRICT pass FIRST-TRY: the regenerated golden is **md5
  `77c9205fef0cc0033a718fe549b4d5ca`** (722/722 filter-clean teams, STRICT `filtered_diverged == 0`
  over **220 battles / 11459 decisions**, 217 wins + 3 ties; taxonomy 300/300 clean, EMPTY gap
  lists). The pre-regen golden replayed BYTE-IDENTICAL first (md5 `64edcdcd5c6a63b1256fc23d3887d8c7`
  unchanged). HONEST COVERAGE NOTE: the sampled corpus EXERCISES **3 Solar Beam casts** (e2e_213 —
  a sun/solarbeam team) but **0 Hyper Beam / 0 Doom Desire decisions** — the 1 HB-carrier + 3
  DD-carrier battles never had the carrier mon active at a move boundary (the fixed-damage /
  leech-seed situation) — so HB/DD stay proven by the DEDICATED batch-4c golden (2927 HB / 1069
  future-resolve decisions) + the MC49-MC60 pins. The handler-audit manifest grew 767 → **787**
  rows (the mustrecharge/twoturnmove/futuremove conditions are enumerated explicitly in
  `ENGINE_CONDITIONS`, all implemented).
  **UPDATE (2026-07-12) — MOVE-COVERAGE BATCH 2 ADMITTED (`BATCH2_E2E_EXCLUDED = false`).** The
  cure / weather-set / stat-drop / screen classes are now IN the e2e allow-list too (Refresh / Heal
  Bell / Aromatherapy / Rain Dance / Sunny Day / Screech&c. / Light Screen / Reflect). The result
  block's counts above are the pre-batch-2 golden; with batch 2 admitted the committed golden is
  **md5 `738da13e9ab666ae50ead17bc6329a08`**, **722/722 filter-clean teams**, STRICT
  `filtered_diverged == 0` over **220 battles / 11176 decisions** (5295 STATUS-MOVE, 1077 SETUP,
  1046 RECOVERY, 583 PROTECT, 575 PHAZE, 232 LEECH, 435 SUBSTITUTE, 126 TAUNT/DISABLE, 371 EXPLOSION,
  3725 SPIKES-up, 197 TRAPPED). Admitting batch 2 surfaced ONE real-team-only bug — the
  `gen3_pressure_allyteam_v1` **Pressure × allyTeam PP-deduction** desync (e2e_182, root-caused NOT to
  residual ordering but to the port applying Pressure's −1 to Aromatherapy/Heal Bell — see the batch-2
  move-class section above for the full story + the `pressure_does_not_add_pp_for_an_allyteam_move`
  pin). It was a CLEAN STRICT pass after that one fix; the handler-audit manifest grew to 728 rows
  (the batch-2 move handlers, all implemented). The pre-batch-2 seed suites stay BYTE-IDENTICAL.
  **Admitting Natural Cure surfaced NO new engine bug** — the enlarged 449-clean-team corpus is a CLEAN
  STRICT pass first-try (the cure is a draw-free, well-localized status clear at the switch-out site, so it
  composed cleanly with every existing mechanic; unlike the trapping admission, which surfaced the
  Intimidate-vs-Substitute STATE gap).** A
  `status_present_rows >= 500` floor pins the status-move exercise, a `spikes_decisions >= 50` floor pins
  the Spikes exercise, a `substitute_decisions >= 50` floor pins the Substitute exercise, a
  `taunt_decisions >= 50` floor pins the Taunt exercise (NO disable floor — expected 0), and a
  `trapped_decisions >= 50` floor pins the trapping exercise. **The trapping admission surfaced +
  FIXED one more real-team-only engine bug — gen-3 INTIMIDATE vs SUBSTITUTE** (e2e_171/e2e_204: a
  Jynx that Substituted the turn before a Salamence switch-in kept Atk 0 in the sim while the port
  dropped −1; the gen3 mod's Intimidate SKIPS a subbed foe — `event::intimidate_on_start` now gates
  on the target's `substitute`, probe-verified SEED-NEUTRAL by
  `harness/probe_intimidate_substitute_rng.js`, pinned by the revert-verified
  `regression_test.rs::intimidate_into_a_substitute_is_a_noop`). **LEECH SEED is in the e2e allow-list** (`MODELED_LEECH_MOVES`,
  `LEECHSEED_E2E_EXCLUDED = false`) — its residual is DRAW-FREE so it can't shift the LCG the way the
  phaze `sample` does — and, since the batch-3 admission, the golden finally **EXERCISES it: 354
  LEECH-MOVE decisions** (the old honest "0 leech decisions" disclosure is CLOSED — the gen3ou leech
  users like Celebi/Venusaur pair Leech Seed with berries/Lum, which used to keep them off the
  filtered path). It also stays proven by its DEDICATED golden (`leechseed_test.rs`, 560 runs) + the 3
  regression pins. (`liquidooze` is now in **`MODELED_ABILITIES`** — `gen3_liquid_ooze_v1`: the
  drain/leech-seed heal REVERSAL is MODELED bit-for-bit [`apply_drain` / `apply_leech_seed` turn the
  would-be heal into damage on the healer, DRAW-FREE], no longer a `NOOP_ABILITIES`/fail-loud
  exclusion; proven by the DEDICATED `liquidooze_test.rs` golden + the LO1-LO3 pins.) The headline assertion tallies are CLEAN-ONLY: the per-decision loop
  breaks at the first divergence so post-desync rows are never counted (today nothing diverges, so it
  never trips). **The RECOVERY-move expansion surfaced + FIXED one more real-team-only engine bug — the
  RESIDUAL HANDLER GATHER ORDER** (a Gengar-vs-Gengar burn+Leftovers+sand turn where the residual KO'd
  the foe: the speed-sort's tie-group Fisher-Yates shuffle PERMUTES the tied handlers IN THEIR PRE-SORT
  ORDER, so the gather order is load-bearing for the same draw. The port gathered Leftovers BEFORE the
  status DoT per mon, but `findPokemonEventHandlers` gathers STATUS first then ITEM — so a 2-mon DoT tie
  shuffled to the WRONG side-order, running OUR burn DoT before the foe's game-ending DoT aborted the
  residual [we should have stayed un-chipped]. Now the port pushes the StatusDot handler BEFORE
  Leftovers per mon — the subOrder still sorts Leftovers ahead, but the shuffle reads the right order. A
  STATE-only desync the seed match masked). **The prior SETUP-move expansion also surfaced + FIXED — WATER/
  VOLT ABSORB heal is now ACCURACY-GATED** (a missed Hydro Pump into a Water Absorb Politoed wrongly
  healed it `maxhp/4`; the absorb is an `onTryHit` that fires only on a HIT — `run_move` gates
  `apply_absorb_heal` on `acc_hit`; the draw count is accuracy-only either way, so it was a STATE-only
  HP desync the seed match masked). **The prior status-move expansion surfaced + FIXED the TOXIC STAGE
  RESET on switch-in (`tox.onSwitchIn`:
  `effectState.stage = 0`):** a badly-poisoned mon that switched OUT and back was resuming at its prior
  high stage (over-chipping by `~maxhp/16` per missed reset). Caught ONLY because Toxic is now a pickable
  move and mons pivot mid-battle. (Sleep PERSISTS across a switch; only the tox stage resets.)
  **RE-MEANED by fix-queue #4 (`gen3_tox_stage_persists_v1`):** the reset lives in `run_switch`
  (the runSwitch-time `runEvent('SwitchIn')`), NOT `execute_switch`'s array swap — so a replacement
  whose queued runSwitch is CANCELLED by gen3 faint-cancels-all keeps its prior stage (pins TX1/TX2).
- **Engine bugs THIS capstone surfaced + FIXED** (real-team-only, the prior constructed goldens
  couldn't reach): (1) **Water/Volt Absorb HEAL** — the port treated them as bare damage-immunity;
  gen3 heals the defender `floor(maxhp/4)` on the absorbed type (draw-free `onTryHit`), applied at the
  immunity short-circuit (`apply_absorb_heal`) — and, per the SETUP-move expansion, now **accuracy-
  gated** (the heal is an `onTryHit` that fires only on a HIT, so a MISSED Water/Electric move no longer
  heals). (2) **Intimidate vs Clear Body / White Smoke /
  Hyper Cutter** — the lead/switch-in Intimidate ignored the foe's `onTryBoost` Atk-drop immunity (a
  no-op into Metagross etc.); now gated in `event::intimidate_on_start`. (3) **The end-of-turn
  RESIDUAL-vs-FAINT ordering + the cached `pokemon.speed` model** — the prior gap (17/220 weather
  battles, FIXED). (4) **The RESIDUAL HANDLER GATHER ORDER** (the RECOVERY-move expansion, see the
  result bullet above) — the status DoT must be gathered BEFORE Leftovers per mon (mirroring
  `findPokemonEventHandlers`'s status-before-item order) so the tie-group shuffle permutes correctly.
  (5) **The PROTECT residual duration-handlers + the no-delete-on-fail stall + the `willAct()` switch
  gate** (the PROTECT-move expansion) — three interlocking fixes the dedicated protect golden's
  no-weather scenarios couldn't reach: (a) the `protect`/`stall`/**`flinch`** volatiles each register a
  residual duration handler (`findPokemonEventHandlers(..., 'duration')`) that participates in the residual
  speed-sort tie-shuffle — a failed-protect-into-a-RockSlide-flinch ties stall+flinch (flinch was draw-free
  before protect; confusion has NO `duration` so no handler); (b) a FAILED stall roll does NOT delete the
  gen3 (resolved gen5-base) `stall` volatile — the counter PERSISTS (the gen8+-base delete model wrongly
  dropped the stall residual handler); (c) the `willAct()` gate fails a Protect vs a foe switch draw-free.
  All validated by the full suite (no regression).
  (6) **The FORCED-REPLACEMENT `updateSpeed` on commit** (the SPIKES expansion) — `commitChoices()` runs
  `this.updateSpeed()` at its TOP (battle.ts:3020) on EVERY choice commit, **including a mid-turn
  forced-replacement submit**. The port refreshed the cached `pokemon.speed` only at turn-start / residual /
  switch-in, NOT on a forced-replacement commit — so a foe paralyzed MID-TURN kept its STALE turn-start speed
  through the resumed-turn `eachEvent('Update')` tie-shuffles, **spuriously TYING with the fresh entrant**
  (a Jirachi-vs-Jirachi para mirror: the para'd foe must read its para speed 53, not its stale 212, during
  the post-replacement Updates — else 2 phantom tie-shuffle draws desync the seed). The port now calls
  `update_speed()` when a forced replacement commits (mirroring `commitChoices`), before the instaswitch sort
  + the resumed tail's Updates. A SEED-only desync the spikes-modified battle selection exposed (NOT a spikes
  bug — spikes layers were 0 on the switching side; the EntryHazard step was a verified no-op there). All
  validated by the full suite (no regression — fullbattle/secondary/protect forced-replacement parity holds).
- **The end-of-turn residual + cached-speed fix (was the last gap, 17/220; now bit-for-bit).** Two
  intertwined `turn.rs` corrections, both faithful ports of Showdown's control flow (verified against
  `battle.js`'s `fieldEvent('Residual')` + the eachEvent speed semantics, and a per-eachEvent SIM
  probe):
  - **Per-handler `faintMessages` in `run_residuals`.** `fieldEvent('Residual')` runs `faintMessages()`
    after EACH handler and `if (this.ended) return`s — so `fainted` is set BETWEEN handlers (a holder
    fainted by an earlier handler `continue`s its later ones) and a GAME-ENDING residual KO aborts the
    rest, while a NON-game-ending residual faint does NOT abort (the other active still ticks). The port
    applied all handlers in one loop then processed faints once — so a fast burned mon's status-DoT
    self-KO did not stop a slower foe's Leftovers (heal `~maxhp/16` off). (Showdown sorts residual
    handlers by `order→priority→SPEED→subOrder`, so SPEED outranks subOrder — a faster mon's DoT runs
    before a slower mon's Leftovers.) Now mirrors the `while`-loop exactly. `run_turn`'s post-residual
    faint gate was switched from `process_faints()`'s newly-fainted return (now always false — the
    residual already consumed it) to the STATE `any_active_fainted()`, so the trailing Update + Quick
    Claw are still correctly skipped on a residual-faint turn.
  - **The cached `pokemon.speed` model (`MonState::cached_speed`).** The `eachEvent` speed-tie shuffles
    AND the residual handler-sort read `pokemon.speed` — a CACHED value, NOT recomputed live on every
    read. Showdown (re)establishes it to the para/boost-aware `getActionSpeed()` at exactly three sites:
    `commitChoices` (turn start), the `residual` action start, and SWITCH-IN. Between them it goes STALE
    — so a mon paralyzed WHILE already active keeps its turn-start (full) speed through the move-phase
    shuffles and only drops at the residual, while a mon that SWITCHES IN paralyzed ties on its PARA
    speed immediately. The port read the live `effective_speed` everywhere, so its mid-turn shuffle tie
    decisions (hence the Fisher-Yates draw COUNT) desynced from the sim on para/switch turns. Now the
    port caches `cached_speed`, refreshes it at those three sites (`update_speed()` + `execute_switch`),
    and the shuffles/residual-sort read it — bit-for-bit. (Verified vs a per-`eachEvent` SIM probe: a
    Jirachi paralyzed while active shows spd 212 until the residual then 53; a Jirachi that switches in
    paralyzed shows spd 53 immediately.)
- **The coverage taxonomy** `tests/vectors/e2e_fuzz_taxonomy.txt` — a separate UNFILTERED sweep
  (random real-team pairs, NO ability/item pre-filter) that ranks coverage gaps by **STATIC TEAM
  COMPOSITION** — which unmodeled ability/item the PAIRED TEAMS CARRY (`classifyTeamsGaps`), NOT by the
  observed first-divergence cause, and **MOVE-LEVEL-BLIND** (the sweep only ever picks damaging-or-
  switch choices, so status moves / Spikes / Calm Mind are never chosen → never counted). It still runs
  each pair through the sim, but only to drop empty/errored battles — the ranked counts come from the
  static team scan, not the run. **Every random pair is gappy** (real teams always carry an unmodeled
  mechanic), so the ranked counts are the prioritized "which unmodeled mechanic do real teams carry
  most" remaining-work list — except that, post-batch-3, **most random pairs are now CLEAN**: with the
  **`gen3_berry_trace_shedskin_v1`** batch-3 admission (the 22 berries + `trace`/`shedskin`) the sweep
  reads **294 of 300 unfiltered battles filter-CLEAN**, the only remaining ABILITY gaps are
  `truant` (=4) > `innerfocus` (=2), and the ITEM gap list is **EMPTY** — `trace` (was the #1 gap),
  `lumberry` (=64), `salacberry` (=46), `petayaberry` and the whole berry family have ALL DROPPED OFF
  (modeled). Batch-2 already cleared `synchronize`/`effectspore`/`static`/`poisonpoint`/`soundproof`/
  `suctioncups`/`damp`/`roughskin`; earlier levers off: `immunity`/`shellarmor` (batch-1 +
  STATUS_IMMUNE), the DMG_MOD family (`torrent`/`blaze`/`guts`/…, `gen3_sun_freeze_immunity_v1`),
  `magnetpull`/`arenatrap` (`gen3_trapping_v1`), `naturalcure` (`gen3_natural_cure_v1`). The
  `ENGINE_GAP` line reads `none` (the engine is bit-for-bit; the prior weather
  residual-vs-faint gap is FIXED). This does NOT gate `cargo test` (it's the measured coverage map, not a
  pass/fail).


---

## The Turn rung's honest scope (superseded by the multi-turn layer)

  silently TIE on action speed (or vice versa) fails loudly at generation.
- **Honest scope / deferred at the SINGLE-turn layer** (the `turn_test.rs` golden
  uses no-residual moves so its post-turn HP is a clean function of only the modeled
  move draws): the multi-turn layer below adds the per-action `eachEvent` shuffles +
  residuals; still deferred everywhere — secondary effects (the per-move
  `random(100)`), status MOVES, switching, recoil/drain HP, status `onBeforeMove`
  draws (para/sleep/freeze), Leech Seed / Wish / non-Leftovers items, Thick Club /
  non-folded stat events, and protocol-string emission. The golden uses Earthquake /
  Surf / Tackle / Hydro Pump / Megahorn / Crabhammer / Swift (never_miss) and bulky
  defenders.

