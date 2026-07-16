# Move-coverage plan — supporting all gen3ou teams on the Rust port (pokesim)

**Scope:** which gen3 moves the `src/rust_sim` ENGINE can EXECUTE without a fail-loud, ranked by
team-unlock value, grouped into build-able classes — the batch roadmap to `--use-bridge=rust`
playing every `data/teams/` team. NODE/analysis-only; no engine code changed.

**Regenerate:**
```
# static map + team tallies + greedy set-cover (deterministic):
node src/rust_sim/harness/scan_move_coverage.js            # human report
SCAN_JSON=1 node src/rust_sim/harness/scan_move_coverage.js # machine JSON
# AUTHORITATIVE engine oracle (actually runs each move through the engine, catches fail-louds):
CARGO_TARGET_DIR=/tmp/pokesim_target_scope cargo build --release --bin scan_move_probe
printf 'wish\ndoubleedge\n...' | /tmp/pokesim_target_scope/release/scan_move_probe
```

## Headline numbers

- **Teams:** 773 `.txt` files → **722 valid** under gen3ou (51 validate-fails, 0 import-fails).
- **Distinct moves across the 722 teams: 108** (per `node harness/scan_move_coverage.js`, whose
  classifier mirrors `src/turn.rs`'s modeled sets — refreshed for batches 1+2).
  - **77 MODELED** — the engine runs them bit-for-bit (incl. **typed Hidden Power**, all 16 types;
    **BATCH 1** recoil/drain/self-drop/item-removal/rapid-spin + **BATCH 2** status-cure/weather-set/
    stat-drop/screens).
  - **MISMODELED (`--use-bridge=rust` would diverge silently): NONE LEFT.** Focus-Punch's beforeTurn
    queue + Pursuit's variable-BP are MODELED (**BATCH 4a**, `gen3_move_coverage_batch4_v1`), Beat Up's
    multi-strike stat-swap + Water Spout's variable-BP + Thunder's rain-accuracy are MODELED
    (**BATCH 4b**, `gen3_move_coverage_batch4b_v1`), and the last cluster — Hyper Beam's recharge +
    Solar Beam's two-turn charge + Doom Desire's (+ Future Sight's) future strike — is MODELED
    (**BATCH 4c**, `gen3_move_coverage_batch4c_v1`). Every remaining un-modeled move FAIL-LOUDs.
  - **23 UNMODELED** — the engine FAIL-LOUDs (`panic!` / `is not modeled`). Honest crash, no desync.
- **Teams already FULLY engine-playable (every move MODELED bit-for-bit): 722 / 722** (was 8
  pre-batch-1; 662 after batch 5; batch 6 → 718; **SNATCH** (`gen3_snatch_v1`) models the LAST
  status move → **722 / 722** — every `data/teams/` team fully engine-playable, 0 UNMODELED moves
  left. See the SNATCH section in `src/rust_sim/CLAUDE.md`.).
- **Teams with NO FAIL-LOUD (MISMODELED allowed to run-but-wrong): 176 / 722.**

The oracle used is the ENGINE's true coverage, verified by actually running each move through
`Battle::start_with_switchins → run_full_battle` (the `scan_move_probe` binary), **not** the e2e
picker `isModeledMove` (which false-rejects typed HP + the modeled fixed-damage family). All 108
classifications were empirically confirmed: **77 run without a fail-loud** (62 MODELED + 15
MISMODELED), **31 panic**. (The bare `hiddenpower` id panics in isolation, but real packed teams
carry the TYPED token `HiddenPowerGrass` → num 355-370, which the engine runs — so HP is MODELED.)

**Four moves the STATIC dex-flag heuristic first mis-called MODELED, caught by auditing every
"modeled" damaging move's dex handlers + the empirical probe:** Rapid Spin (`onAfterHit`
hazard-clear), Knock Off / Thief (`onAfterHit` item-removal), Doom Desire (`flags.futuremove`
delayed strike). All four RUN as plain damage (no fail-loud) but drop their side-effect → silent
desync. They are the reason the STATIC dex-flag pass alone is insufficient and the ENGINE oracle
is authoritative.

## ✅ BATCH 1 DONE (`gen3_move_coverage_batch1_v1`, 2026-07-12) — the DRAW-FREE mismodeled post-hit effects

The FIVE highest-frequency SILENT-DESYNC classes — a damaging move that RUNS but drops a
post-hit side-effect — are now MODELED bit-for-bit + e2e-admitted:

| Class | Members | Draw model (probe-settled) | Fix |
|---|---|---|---|
| **C_RECOIL** | Double-Edge / Take Down / Submission (`recoil:[num,den]`) | `max(floor(dmgDealt·num/den),1)` to the USER; Rock Head negates; fires behind a sub. DRAW-FREE. | `turn.rs::apply_recoil` |
| **C_DRAIN** | Absorb / Mega Drain / Giga Drain / Leech Life (`drain:[1,2]`) | USER heals the fraction of the damage dealt (floor non-sub / ceil behind a sub); heal-at-full fails. DRAW-FREE. Liquid Ooze reverses → fail-loud (excluded). Dream Eater's sleep-gate unmodeled → excluded. | `turn.rs::apply_drain` |
| **C_SELFDROP** | Overheat (self −2 SpA) / Superpower (self −1 Atk/−1 Def) | the drop applies (±6 clamp) AND gen3 `selfDrops` **DRAWS ONE `random(100)`** (the `secondaryRoll`, unconditional — `self.chance === undefined`). **NOT draw-free** — the reason the port's Overheat/Superpower were never seed-verified. | `turn.rs::apply_self_drops` + the extractor's `selfDrops` field |
| **C_ITEM_REMOVAL** | Knock Off (removes; gen3 no dmg boost) / Thief / Covet (steal iff attacker itemless) | `onAfterHit` — fires ONLY when the MON was damaged (NOT behind a sub). Sticky Hold blocks; Mail does NOT block these three. DRAW-FREE. | `turn.rs::apply_item_removal` |
| **C_RAPIDSPIN** | Rapid Spin | `onAfterHit` + `onAfterSubDamage` clear the USER's own Spikes + Leech Seed (+ partial-trap, N/A) — so it clears behind a sub too. DRAW-FREE. | `turn.rs::apply_rapid_spin` |

**Validation:** the class-sweep golden `harness/gen_movecoverage_batch1_golden.js` →
`tests/movecoverage_batch1_test.rs` (13 scenarios × 80 seeds = 1040 game-end battles, 10428
per-decision STATE(+HP+BOOSTS+SPIKES+LEECH+ITEM)+SEED assertions, byte-reproducible) + 8
revert-verified `tests/regression_test.rs` pins (MC1 recoil / MC1b Rock-Head / MC2 drain / MC3
selfDrops-random(100) / MC4 knock-off / MC4b Sticky-Hold / MC5 Thief / MC6 Rapid-Spin — the
DRAW-FREE ones share the seed "4448,587,55846,30246"; MC3 differs, proving the self-drop draw).
Probes kept: `harness/probe_batch1_movecoverage.js`, `probe_batch1_order.js`,
`probe_batch1_selfdrops_rng.js`, `probe_batch1_regression_rng.js`. **e2e-admitted** (the modeled
sets `MODELED_{RECOIL,DRAIN,SELFDROP,ITEM_REMOVAL,RAPIDSPIN}_MOVES` in `gen_e2e_fuzz.js`; the OLD
golden replays BYTE-IDENTICAL — md5 unchanged pre-regen — then regen unlocks 719 → 722 filter-clean
teams). See `src/rust_sim/CLAUDE.md` → the batch-1 move-class section.

## ✅ BATCH 2 DONE (`gen3_move_coverage_batch2_v1`, 2026-07-12) — the DRAW-friendly status-move classes

The FOUR DRAW-friendly category-Status move classes — status-cure / weather-set / stat-drop /
screens — are now MODELED bit-for-bit:

| Class | Members | Draw model (probe-settled) | Fix |
|---|---|---|---|
| **C_STATUS_CURE** | Refresh / Heal Bell / Aromatherapy | NEVER-MISS + DRAW-FREE. Refresh clears ANY status EXCEPT slp/frz/none (par/psn/**tox**/brn — Toxic IS cured). Heal Bell iterates the team (active+bench) SKIPPING a Soundproof ally; Aromatherapy `clearStatus` (no Soundproof gate). | `run_status_move` cure arms (`cures_self_status`/`cures_team_status`) |
| **C_WEATHER_SET** | Rain Dance (Rain) / Sunny Day (Sun) | a 5-turn TIMED weather (vs the permanent ability weather). DRAW-FREE at the move; the `eachEvent('WeatherChange')` tie-shuffle draws only on a speed tie. setWeather FAILS (draw-free) into the SAME weather, OVERWRITES a different one. 5-turn upkeep + expiry at the field residual. | `modeled_weather_set_move` + `apply_weather_chip` countdown |
| **C_STAT_DROP_MOVE** | Screech/Charm/Metal Sound/Feather Dance/Tickle/Fake Tears/Cotton Spore/Scary Face | accuracy roll (Screech/Metal Sound acc-85 CAN miss) + a DRAW-FREE `boost()` foe drop (Clear Body/Hyper Cutter/Soundproof gated). | `run_status_move` stat-drop arm (data-driven `statDropBoosts`) |
| **C_SCREEN** | Light Screen (½ special) / Reflect (½ physical) | NEVER-MISS + DRAW-FREE set (5-turn SIDE condition). **THE CRUX:** a damaging hit into a side with BOTH screens up draws ONE extra `random(0,2)` — the `ModifyDamagePhase1` handler-sort shuffle (the 2 screens' `onAnyModifyDamagePhase1` handlers tie). | `modeled_screen_move` + `SideState::{light_screen,reflect}` + the `run_move` ModifyDamagePhase1 shuffle |

**Validation:** the class-sweep golden `harness/gen_movecoverage_batch2_golden.js` →
`tests/movecoverage_batch2_test.rs` (17 scenarios × 80 seeds = 1360 game-end battles, 16178 per-decision
STATE(+HP+STATUS+BOOSTS+WEATHER+SCREENS)+SEED assertions, byte-reproducible) + **9 revert-verified
`tests/regression_test.rs` pins** (MC9-MC17 — MC17 is the double-screen ModifyDamagePhase1 shuffle
crux). Probes kept: `harness/probe_batch2_movecoverage.js`, `probe_batch2_regression_rng.js`. DATA: the
extractor emits `statDropBoosts` (obs-neutral, like `selfDrops`); Refresh/Heal Bell/Aromatherapy reuse
the existing `curesSelfStatus`/`curesTeamStatus` flags. **e2e — ADMITTED (`BATCH2_E2E_EXCLUDED =
false`, 2026-07-12), STRICT clean.** Admitting batch 2 to the e2e capstone surfaced ONE real-team-only
divergence — **e2e_182**, which was FIRST described as a "5-HP Blissey residual-heal-ordering gap" but
root-caused (via `harness/probe_e2e182_simtrace.js` + the sim probe) to a **`Pressure` × `allyTeam`
PP-deduction bug**, NOT a residual-order issue: the port applied the Pressure `−1` extra PP drop to
Blissey's Aromatherapy (an `allyTeam` move) under a Pressure Zapdos, because it keyed the extra on
`!targets_self` instead of the real rule (the Pressure foe fires its `onDeductPP` only when it is in the
move's `pressureTargets` — a FOE-directed target; `allyTeam` / `self` / `foeSide` never put the foe
there). The mis-drain exhausted Aromatherapy's 8 PP early, so the port REJECTED a legitimate late
Aromatherapy as out-of-PP → the script shifted and the battle desynced (a decision-count + state gap).
FIX: `turn.rs::pressure_targets_foe` (`gen3_pressure_allyteam_v1`), pinned by
`regression_test.rs::pressure_does_not_add_pp_for_an_allyteam_move` (revert-verified; ground truth
`harness/probe_pressure_allyteam_rng.js`). Batch 2 is now IN the e2e allow-list; the regenerated golden
is **md5 `738da13e9ab666ae50ead17bc6329a08`** (722/722 filter-clean teams, STRICT `filtered_diverged ==
0` over 220 battles / 11176 decisions). See `src/rust_sim/CLAUDE.md` → the batch-2 move-class section.

## ✅ BATCH 3 DONE (`gen3_move_coverage_batch3_v1`, 2026-07-12) — the STATEFUL DRAW-FREE move classes

**CURSE / WISH / BATON PASS — the three stateful move classes are MODELED bit-for-bit + e2e-ADMITTED.**
- **CURSE** (`curse`, type-conditional at `onModifyMove`) — NON-GHOST → a self-boost {atk:+1, def:+1,
  spe:-1} that DRAWS ONE `random(100)` via the gen3 `selfDrops` path (like Overheat, NOT draw-free — the
  probe-surfaced subtlety); GHOST → pays floor(maxhp/2) HP + lays the `curse` volatile on the FOE (the
  order-10 subOrder-8 residual chip floor(maxhp/4)/turn). Re-curse fails ([still]+-fail); curse-into-a-sub
  does nothing; a Ghost target is NOT immune. `turn.rs::run_status_move`'s curse arm + `apply_curse`;
  `MonState::curse`.
- **WISH** (`wish`, `slotCondition`, duration 2) — the slot-keyed order-7 delayed heal floor(maxhp/2) at
  N+1 (BEFORE the sand chip order 8 + all order-10 handlers — VERIFIED; two Wishes at equal speed
  tie-shuffle); double-Wish fails ([still]); heal-at-full is silent. `SideState::wish_pending`;
  `apply_wish`.
- **BATON PASS** (`batonpass`, `selfSwitch:'copyvolatile'`) — a self-switch passing the outgoing mon's
  boosts + the copyable (`noCopy==false`) volatiles (substitute / leech-seed / confusion / curse) to the
  entrant; no-bench fail; the entrant's `|switch|` carries `[from] Baton Pass`. The `copyVolatileFrom`
  snapshot lives in `execute_switch`; `SideState::baton_pass_pending`.

Validated by the DEDICATED golden `gen_movecoverage_batch3_golden.js` → `movecoverage_batch3_test.rs`
(16 scenarios × 80 seeds, 4980 decision rows, 1280 wins) + the **MC18-MC29** revert-verified
`regression_test.rs` pins (ground truth `harness/probe_batch3_regression_rng.js`; the MC23 Wish
residual-ORDER pin is a LIFE/DEATH order test — a low-HP mon under sand survives ONLY because the
order-7 Wish heals before the order-8 sand chip). **e2e ADMITTED** (`BATCH3_E2E_EXCLUDED = false`) — a
CLEAN STRICT pass first-try, NO new engine bug: the pre-regen golden replayed BYTE-IDENTICAL (md5
`738da13e…` unchanged, the batch-3 code a no-op on the old golden) then the deliberate regen shifted it
to **md5 `529ab3f0940f8f9cbab383fb26d2a696`** (722/722 filter-clean teams, STRICT `filtered_diverged ==
0` over 220 battles / 11163 decisions). See `src/rust_sim/CLAUDE.md` → the batch-3 move-class section.

## ✅ BATCH 4 + 4b DONE — the beforeTurnCallback + variable-BP/weather-accuracy damaging moves

**BATCH 4** (`gen3_move_coverage_batch4_v1`) modeled the two `beforeTurnCallback` damaging moves — **FOCUS
PUNCH** (the beforeTurn `|-singleturn|` + onTry cancel-if-hit) + **PURSUIT** (the switch-interrupt ×2
never-miss strike) — e2e-ADMITTED (golden md5 `fe1529609264be655f36032e0261868d`, 11481 decisions).

**BATCH 4b** (`gen3_move_coverage_batch4b_v1`, 2026-07-14) modeled the THREE remaining MISMODELED
single-turn damaging moves — **BEAT UP** (the multi-strike TYPELESS stat-swap: ally base-atk → SpA,
target base-def → SpD; ONE accuracy roll + per-strike crit+damage + the per-strike `eachEvent('Update')`;
the `beatup` `duration:1` residual handler; sets lostFocus), **THUNDER** (the id-gated weather-accuracy
mutation: rain never-miss / sun 50 / else 70), **WATER SPOUT** (`bp = max(floor(150·hp/maxhp),1)`,
draw-neutral). Validated by `movecoverage_batch4b_test.rs` (14 scenarios × 80 seeds = 1120 battles) + 7
revert-verified pins (MC39-MC45). **e2e ADMITTED** (`BATCH4B_E2E_EXCLUDED = false`) — golden md5
`64edcdcd5c6a63b1256fc23d3887d8c7` (STRICT `filtered_diverged == 0` over 220 battles / 11407 decisions),
after fixing THREE real-team-only bugs (the per-strike `eachEvent('Update')`, the beatup-volatile residual
duration tie, and Beat Up setting the target's Focus-Punch lostFocus). See `src/rust_sim/CLAUDE.md` → the
batch-4b section.

## ✅ BATCH 4c DONE (`gen3_move_coverage_batch4c_v1`, 2026-07-14) — the TURN-SPANNING move classes (the LAST MISMODELED cluster)

**HYPER BEAM (mustrecharge) / SOLAR BEAM (two-turn charge + sun skip) / DOOM DESIRE + FUTURE SIGHT
(the slot-keyed future strike) are MODELED bit-for-bit + e2e-ADMITTED — the MISMODELED set is now
EMPTY.**
- **HYPER BEAM** (`hyperbeam`, 150-BP Physical acc 90) — a SUCCESSFUL damaging hit (plain / sub-absorb /
  sub-BREAK / target-KO; NOT a miss / immune / Protect-block) applies `MonState::must_recharge`
  DRAW-FREE (`|-mustrecharge|`, printed before a KO's `|faint|`; the lock PERSISTS across the foe's
  force-switch). The LOCKED turn's request offers ONLY `{move:"Recharge",id:"recharge"}` + firm
  `trapped:true`; the turn is spent as `|cant|…|recharge` at the user's normal speed-order position —
  ZERO draws, NO PP (the gen3-resolved `mustrecharge.onBeforeMove` at priority **11** precedes EVERY
  status handler: a par'd/slp'd locked user rolls/decrements NOTHING), then the lock fully clears. The
  `duration: 2` volatile registers a NO_ORDER/subOrder-2 residual duration handler on the CAST turn's
  residual (the HB-mirror tie draw). Truant composes with NO special case (the recharge cant precedes
  the truant gate; the order-27 toggle consumes the loaf — HB/recharge/HB cadence, no truant cant on
  the landed path; a MISSED HB legitimately loafs next turn). Fail-loud siblings: blastburn /
  frenzyplant / hydrocannon.
- **SOLAR BEAM** (`solarbeam`, 120-BP Special Grass acc 100) — CHARGE turn: onBeforeMove draws fire
  first (a para roll IS drawn; a full-para cant = no charge, NO PP), PP deducted (−1; **−2 under a
  Pressure foe — Pressure applies at the CHARGE**), `[still]` + `|-prepare|`, ZERO move draws →
  `MonState::two_turn` (the `twoturnmove` volatile: duration 2 + the `solarbeam` sub-volatile =
  `charging`), which registers a NO_ORDER/subOrder-2 residual duration handler on BOTH residuals. FIRE
  turn: the locked single-move request; NO PP; accuracy 100 DRAWN → crit → damage
  (`|move|…|[from]lockedmove`). An ABORT on the fire turn (slp/par/frz/flinch cant) LOSES the charge
  (onMoveAborted; a fresh charge re-pays PP); a Protect-blocked fire consumes the charge (acc drawn,
  no crit/dmg). SUN (`effectiveWeather` — Cloud Nine-aware) SKIPS the charge (still + prepare + `-anim`
  then a normal 3-draw execution). Rain/sand/hail HALVE the BP (the gen3-resolved onBasePower
  chainModify(0.5) — gen3 DOES have the modern halving, probed rain 54 vs control 105; suppression-
  aware, read at damage time, draw-free). Fail-loud siblings: razorwind / skyattack / skullbash / fly /
  dig / dive / bounce.
- **DOOM DESIRE + FUTURE SIGHT** (`doomdesire` bp 120 Physical acc 85 / `futuresight` bp 80 Special
  acc 90 — probe-settled SAME mechanic) — the CAST (`onTry`, BEFORE the protect check — a cast-turn
  Protect does NOT block) draws exactly ONE `random(16)`: the cast-time TYPELESS damage SNAPSHOT
  (no STAB / no chart → never immune; cast-time stats/boosts; willCrit false) stored in
  `SideState::future_move` (the slot condition — duration 3, `FUTURE_RESIDUAL_ORDER = 11`, gathered
  every end-of-turn: Wish 7 → sand 8 → order-10s → **futuremove 11**; an equal-speed FS mirror
  tie-shuffles once per residual). A DOUBLE-CAST fails with a bare `|move|` line, ZERO draws, PP still
  deducted. The RESOLVE (the 1→0 tick, end of turn N+2): skip iff the slot occupant is fainted; else
  `|-end|…|move: <Name>`, remove the target's Protect, ONE accuracy roll, then the STORED number lands
  on WHOEVER occupies the slot (sub absorbs, no carry; Focus Band can roll) + the two
  `hitStepMoveHitLoop` `eachEvent('Update')`s with the in-loop `faintMessages` BETWEEN them (a resolve
  KO draws only ONE tie-Update; the Quick Claw defers past the forced replacement). Resolves even when
  the caster switched/fainted (slot semantics — the entrant takes the OLD stored damage). The bridge
  needs NO new request shape (the future-move class never locks the user).

Validated by the DEDICATED golden `gen_movecoverage_batch4c_golden.js` → `movecoverage_batch4c_test.rs`
(23 scenarios × 80 seeds = **1840 game-end battles, 16621 per-decision
STATE(+HP+STATUS+BOOSTS+SUB-HP+WISH+FUTURE-PENDING)+SEED assertions**, the DEC format extended with 2
per-side FUTURE-PENDING columns → 44 fields) + **12 `regression_test.rs` pin functions
MC49-MC60** (the cross-turn cruxes revert-verified) (ground truth `harness/probe_batch4c_regression_rng.js`; draw models settled by
`probe_batch4c_{hyperbeam,solarbeam,doomdesire}.js`) + the fs_mirror_tie golden scenario as the
resolve-KO-single-Update pin. The bridge serializes the LOCKED request (`serialize_active` /
`resolve_choice` / the firm trapped reject) per the probed shape. **e2e ADMITTED**
(`BATCH4C_E2E_EXCLUDED = false`; `futuresight`/`doomdesire` removed from `MOVE_ID_BLOCKLIST`; a
belt-and-braces `flags.futuremove` reject; the picker treats a locked `trapped:true` request as
trapped) — see `src/rust_sim/CLAUDE.md` → the batch-4c section for the regen result.

## ✅ BATCH 5 DONE (`gen3_move_coverage_batch5_v1`, 2026-07-14) — the REACTIVE fixed-damage family + the VARIABLE-BP family + SLEEP TALK

**NINE moves — the top of the greedy team-unlock list — MODELED bit-for-bit + e2e-ADMITTED:**

- **COUNTER / MIRROR COAT** — the order-5 `beforeTurnMove` volatile (`MonState::reactive`;
  the onStart RESETS `{slot:null, damage:0}` every selection turn — prev-turn damage never
  counts) + the priority-−101 onDamage RECORDER (`record_reactive_hit`: 2× each qualifying
  DIRECT foe **Move** hit — counter `Physical || bare hiddenpower`, mirrorcoat `Special &&
  !hiddenpower`; the gen3 TYPE-derived category; a sub-absorbed hit never records; MULTIHIT →
  2× the LAST strike, probed via Beat Up; Seismic-Toss-class fixed damage IS Physical →
  countered; Struggle IS countered; Beat Up's Special strikes arm MIRROR COAT). Execution
  (`run_fixed_damage_move`): un-armed → a **ZERO-DRAW** bare-`|move|` fail (no `-fail`, PP −1);
  armed → ONE accuracy draw (acc 100, NOT never-miss) then type immunity (Fighting→Ghost /
  Psychic→Dark → `-immune`), **NO crit / NO damage roll**, `landed` true (the in-tryMoveHit
  Update at a tie). `duration:1` → a NO_ORDER/subOrder-2 residual duration handler (the
  counter-mirror +4 draw delta: the order-5 pair sort tie + 2 trailing Updates + the residual
  duration tie — probed).
- **ENDEAVOR** — onTry fails at `hp >= target.hp` (**EQUALITY INCLUDED**, `|-fail|<user>`,
  ZERO draws, PP −1); else ONE accuracy draw, Normal→Ghost `-immune` after it, and the delta
  (`target.hp − user.hp` — never a KO) lands fixed-damage-style (a sub takes the number
  computed from the MON's hp; break, NO carry).
- **RETURN / FRUSTRATION / FLAIL / REVERSAL / LOW KICK** (`turn.rs::variable_bp`) — the
  engine-computed BP over a bp-0 data row, DRAW-NEUTRAL (probed seed-identical across
  happiness/HP/weight extremes): Return `floor(h·10/25) || 1` (h≤2 → the `||1` clamp → BP 1, a
  HIT not a fail); Frustration the 255-mirror; Flail/Reversal `ratio = max(floor(48·hp/maxhp),1)`
  → bands `<2:200, <5:150, <10:100, <17:80, <33:40, else 20` (gen3 is 48, NOT gen4's 64; they
  CAN crit — gen2's willCrit=false is NOT inherited); Low Kick the TARGET-`weighthg` ladder
  `≥2000:120, ≥1000:100, ≥500:80, ≥250:60, ≥100:40, else 20` (the NEW extractor field
  `gen3_species.json::weighthg` = round(weightkg·10); gen3 has NO ModifyWeight). The bp-0 row
  mis-derived category Status → re-derived Physical at the BP override; `blocked_by_taunt`
  carves the family out (probed: a taunted mon keeps Return/Flail/Counter selectable).
- **SLEEP TALK** — the slp onBeforeMove prints `|cant|slp` and **PROCEEDS** (`sleepUsable`;
  the counter still decrements; `MonState::sleep_skipped`++ per proceed, reset on a normal
  blocked cant, RESTORED `time += skippedTime` at the runSwitch SwitchIn — live-probed
  3→talk,talk→1,sk2→switch→3). The arm (`run_status_move`): onTry = asleep-only (an
  awake/wake-turn use fails SILENTLY); onTryHit = the choicelock gate (a PRIOR-turn lock →
  `[still]`+`-fail` BEFORE the sample; CB + Sleep Talk works exactly ONCE — the lock records
  Sleep Talk itself, and the lock THIS use sets does not count); onHit = the pool (slot order,
  `!nosleeptalk && !charge` — the NEW data-enumerated `noSleepTalk`/`isCharge` move flags; NO
  pp/disabled filter) → **ONE `sample` = `random(n)` even at n=1** → a 0-PP pick wastes the
  turn (`|cant|…|nopp|<id>`) → else the picked move runs via a bare `useMove` (the
  `sleep_talk_call` transient: no on_before_move / NO PP for the picked move / lastMove stays
  Sleep Talk; the FULL normal draw chain; the announce carries the byte-exact
  `|[from] Sleep Talk`). An asleep-called REST silently no-ops (`run_rest`'s asleep guard — no
  heal, no `random(2,6)`, no counter reset). Empty pool → `[still]`+`-fail`, zero draws.

**Validated:** the DEDICATED golden `gen_movecoverage_batch5_golden.js` →
`movecoverage_batch5_test.rs` (23 scenarios × 80 seeds = **1840 game-end battles, 18548
per-decision STATE+SEED assertions, 37096 HP assertions, 3090 asleep rows — a CLEAN first-try
pass**; the batch-4c 44-field DEC format reused, INJECT gains a per-slot `pp` set) + **16
revert-verified `regression_test.rs` pins (MC61-MC75 + the dex batch5_tests data pin)**, ground
truth `harness/probe_batch5_regression_rng.js`; draw/mechanic models settled by
`probe_batch5_{reactive,varbp,sleeptalk,reactive_edges}.js` (the edges probe settled Beat
Up→Mirror-Coat + Struggle→Counter). **e2e ADMITTED** (`BATCH5_E2E_EXCLUDED = false`;
`MODELED_BATCH5_{REACTIVE,VARBP}_MOVES`; the batch-5 nine removed from `MOVE_ID_BLOCKLIST` —
which ALSO un-shadowed the modeled fixed-damage five (seismictoss/nightshade/sonicboom/
dragonrage/superfang), whose blocklist rows had been overriding their documented
`MODELED_FIXED_DAMAGE_MOVES` early-admit; Sleep Talk's pickability is CARRIER-conditional via
`sleepTalkPoolModeled` — the CALLED move bypasses the picker, so the sampled pool must be
all-modeled; `snore`, the other gen-3 sleepUsable move, stays out/unmodeled). The
handler-audit surface explicitly adds `sleeptalk` (isModeledMove-false but engine-modeled);
the manifest grew 787 → **815 rows**. The coverage scan (`scan_move_coverage.js`, classifier
refreshed for batches 4/4b/4c/5 — the stale MISMODELED rows removed): **662 / 722 teams fully
engine-playable** (MISMODELED distinct moves: **0**; 12 fail-loud status moves remain, headed
by Perish Song 21 / Mean Look 12 / Endure 10).

## ✅ BATCH 6 DONE (`gen3_move_coverage_batch6_v1`, 2026-07-15) — the FINAL UNMODELED tail (13 moves)

**ENCORE / DESTINY BOND / ENDURE / PERISH SONG / MEAN LOOK / SPIDER WEB / BLOCK / BELLY DRUM /
CHARGE / MEMENTO / MIMIC / PAIN SPLIT / PSYCH UP — MODELED bit-for-bit + e2e-ADMITTED.**
- **ENCORE** — acc-100 draw + the `durationCallback` `random(3,7)` INSIDE addVolatile
  (already-encored fails accuracy-ONLY; no-lastMove / failencore / 0-PP-lastMove fails draw BOTH);
  `stored = willMove(target) ? rolled : rolled+1` (the Disable branch — MC79/MC80 are the
  same-seed perturbation pair); the `onOverrideAction` EXECUTION override (a queued different
  move runs AS the encored move, the ENCORED slot's PP deducts); `move_usable` restricts the
  request; the order-10/subOrder-14 residual tick + the 0-PP EARLY `-end`. Data: the extractor's
  `failEncore`/`failMimic` move flags (`gen3_moves.json`, obs-neutral).
- **DESTINY BOND** — a ZERO-draw cast (draw-free re-cast); the window closes at the user's NEXT
  move attempt (onBeforeMove −1 + onMoveAborted at every cant site); a FOE-Move KO while up
  faints the killer too via the `process_faints` worklist (|faint| victim → `-activate` →
  |faint| killer; both-last-mons → the gen-3 TIE); a residual / sub-absorbed / futuremove KO
  does NOT trigger (the record lives only at the Move damage sites).
- **ENDURE** — rides the protect stallingMove machinery with the SHARED `stall` counter
  (2→4→8, no-delete-on-fail, the willAct gate; gen3 priority 4); the `endure` volatile's
  priority-−10 onDamage clamp survives any MOVE damage (fixed damage + every multihit strike
  included) at 1 HP — residual damage still kills; every SUCCESS turn adds the endure+stall
  intra-mon NO_ORDER/subOrder-2 residual duration tie (ONE shuffle at ANY speed).
- **PERISH SONG** — draw-free in EVERY branch; all actives (incl. the caster) get perish
  4-at-apply (the boundary shows 3) ticked at the order-12 residual (LAST in the ladder); the
  1→0 tick prints perish0 + faints via the **DURATION-END `continue`** (NO per-handler
  faintMessages — the sim's fieldEvent duration-end branch — so a speed-tied mirror's mutual
  perish-out is a same-residual DOUBLE faint → the gen-3 TIE; the batch's one first-try pin
  failure, root-caused + revert-verified); Soundproof immune (the >=1-immune re-cast is a
  SILENT success; all-counted fails [still]); switch-out clears; Baton Pass PASSES it
  (noCopy false).
- **MEAN LOOK / SPIDER WEB / BLOCK** — draw-free linked FIRM-trap volatiles
  (`MonState::trapped_by` = the trapper's uid): `is_trapped`/`trap_is_firm` fold it (the
  Shadow-Tag request shape — `trapped:true` first request, `[Invalid choice]` reject, no
  re-request); a grounded GHOST IS trapped; the link ends the moment the TRAPPER leaves ANY
  way (execute_switch source-left clear + the process_faints corpse clear); a trapped mon's
  Baton Pass is LEGAL and the ENTRANT INHERITS the trap (noCopy false — the link re-points);
  a phaze drags through it; a SUBSTITUTE blocks; re-application fails.
- **BELLY DRUM** — the FLOAT `hp <= maxhp/2` gate integer-exact as `2*hp <= maxhp` (262/524
  fails, 263 succeeds); pays `floor(maxhp/2)` via directDamage (no Endure/Focus Band) then a
  SET to +6 (`-setboost`); atk>=6 / maxhp==1 fail. Draw-free.
- **CHARGE** — the `charge` volatile ×2s the next ELECTRIC move's BP (a BP-chain fold);
  CONSUMED by the user's next move attempt OF ANY KIND (`turn_loop`'s post-run_move
  onAfterMove/onMoveAborted consumption keyed on the OUTER move — a Baton Pass consumes it
  BEFORE the switch, so charge never actually survives a pass despite noCopy-false, probed
  MC98); NO gen3 SpD boost. Draw-free.
- **MEMENTO** — never-miss in the RESOLVED gen3 (the base acc-100 is overridden); the landed
  turn is ZERO draws TOTAL (self-faint via the deferred-faint protocol → gen3
  faint-cancels-all kills the foe's queued move; no Quick Claw); foe −2 Atk/−2 SpA through the
  shared boost machinery (Clear-Body gated; the user faints even when blocked/floored — a
  delta-0 `-unboost` emission nuance is skipped, state-identical); a Protect/Sub block → NO
  faint (ifHit).
- **MIMIC** — copies the target's lastMove over the Mimic slot (`pp = min(5, base)`, `maxpp =
  calculatePP(copied, 3)`) via `MonState::mimic_overlay`; the copied slot's PP decrements
  independently; `restore_mimic_overlay` reverts on switch-out/faint (Mimic's OWN remaining PP
  persists); sub / no-lastMove / failmimic / already-known fails, all draw-free.
- **PAIN SPLIT** — `avg = floor((u+t)/2)`, EACH side clamped at its OWN maxhp (the Gengar-vs-
  Blissey clamp case: Blissey takes the full loss, Gengar caps at 261); a sub blocks; works on
  a Ghost (Status ignoreImmunity). Draw-free.
- **PSYCH UP** — copies ALL 7 boost stages VERBATIM (zeros overwrite the user's own prior
  stages); NO protect flag (copies through a Protect); bypasssub. Draw-free.

**Validated:** the DEDICATED golden `gen_movecoverage_batch6_golden.js` →
`movecoverage_batch6_test.rs` (24 scenarios × 80 seeds = **1920 game-end battles, 22074
per-decision seed assertions, 44148 HP assertions, 1711 encore / 1200 perish / 6479
trapped rows, 1707 wins + 213 ties — a CLEAN FIRST-TRY pass**; the batch-4c/5 44-field DEC format
EXTENDED with SIX appended columns — p1/p2 ENCORE duration, PERISH counter, TRAPPED (the live
volatile, NOT the sim's endTurn-stale `pokemon.trapped` flag) → 50 fields) + **20
revert-verified `regression_test.rs` pins MC79-MC98** (ground truth
`harness/probe_batch6_regression_rng.js`; mechanics settled by
`probe_batch6_{locks,field_trap,utility,dexfacts}.js`; every crux revert FAILS its pin — the
one exception is the trapper-FAINT link clear, which is observationally REDUNDANT with the
replacement-switch clear + the fainted-foe `is_trapped` guard, kept as the faithful mirror and
documented at the site). The engine work also fixed the sim-faithful **duration-END
`continue`** in `run_residuals` (the perish mutual-faint tie) and extended
`DecisionRecord` with `encore`/`perish` columns. **e2e ADMITTED**
(`BATCH6_E2E_EXCLUDED = false`; `destinybond` OUT of `MOVE_ID_BLOCKLIST`; a per-DEC
`batch6Move` flag → DEC 39 fields + a GATED `batch6_decisions >= 50` floor): the pre-regen
golden replayed BYTE-IDENTICAL (md5 `614d47b9…` unchanged via `ab_replay`, ok:220), then the
deliberate regen shifted it to **md5 `02fe5d9a59955eaf0360e9d881f46a83`** — STRICT
`filtered_diverged == 0` over 220 battles / 11584 decisions, **58 batch-6 decisions**.
The admission surfaced + FIXED ONE real-team-only bug (e2e_7): a CONTACT **fixed-damage**
hit (Seismic Toss into an Effect Spore Breloom) must fire the defender's contact-proc
`onDamagingHit` — a latent batch-5-era gap, pinned MC99. The coverage scan
(classifier refreshed): **718 / 722 teams fully engine-playable** — the HONEST residual is
**SNATCH** (4 teams), the one gen-3 status-steal reactive, DELIBERATELY DEFERRED (unprobed —
it stays fail-loud; the task's twelve-move list did not include it and the quota gate closed
before a probe round).

## Top-5 build CLASSES by cumulative team-unlock (greedy set-cover)

Starting from **8** fully-playable teams, the greedy class order:

| # | Class | Members (this pool) | Teams in blocking-set | Unlocks | Cum. playable |
|---|---|---|---|---|---|
| 1 | **C_RAPIDSPIN** | Rapid Spin | 298 | 3 | **11** |
| 2 | **C_RECOIL** | Double-Edge (+ Struggle already modeled) | 266 | 12 | **23** |
| 3 | **C_CURSE** | Curse | 241 | 28 | **51** |
| 4 | **C_DELAYED_HEAL** | Wish | 213 | 27 | **78** |
| 5 | **C_FOCUSPUNCH** | Focus Punch | 196 | 37 | **115** |

Full greedy class-cover to 722/722 (35 classes) is in **§ Class-grouped batch plan** below. The
first 9 classes take the pool from 8 → **380**; the first 13 → **536**. (Rapid Spin is on the MOST
teams — 298 — but unlocks only 3 immediately because nearly every Rapid Spin team also carries a
second blocker; its value is that it's a LATE co-blocker on hundreds of teams.)

Note the huge "blocking-set" counts (266, 241, …) vs the smaller "unlocks": most teams carry
SEVERAL blocking moves, so a class unlocks a team only once its LAST blocker is also modeled — the
cumulative column is the real progress signal.

## Per-move engine-coverage map (all 46 non-MODELED moves)

`cov` = static class; `emp` = empirical engine verdict (ran = no fail-loud, panic = fail-loud).
Every row's `emp` matches its `cov` (MODELED/MISMODELED → ran, UNMODELED → panic).

| move | cov | emp | #teams | cat | mechanic (why it's not bit-for-bit) |
|---|---|---|---|---|---|
| rapidspin | **MODELED** | ran | 298 | Phys | ✅ batch 1 — hazard/leech clear (`onAfterHit`+`onAfterSubDamage`) |
| doubleedge | **MODELED** | ran | 266 | Phys | ✅ batch 1 — recoil `floor(dmg/3)` to the user |
| curse | **MODELED** | ran | 241 | Status | ✅ batch 3 — type-conditional (ghost HP-cost + curse residual; non-ghost self-boost + the selfDrops random(100)) |
| wish | **MODELED** | ran | 213 | Status | ✅ batch 3 — slot-keyed order-7 delayed heal (maxhp/2 at N+1; double-Wish fails; heal-at-full silent) |
| focuspunch | MISMODELED | ran | 196 | Phys | beforeTurn `|-singleturn|` + flinch-cancel gate not run |
| gigadrain | **MODELED** | ran | 182 | Spec | ✅ batch 1 — drain heal `floor(dmg/2)` |
| pursuit | MISMODELED | ran | 159 | Spec | variable-BP (×2 + hits on switch) — runs at flat bp 40 |
| batonpass | **MODELED** | ran | 158 | Status | ✅ batch 3 — copyVolatileFrom pass of boosts + sub/leech/confusion/curse to the entrant |
| beatup | MISMODELED | ran | 114 | Spec | multi-strike per healthy teammate — runs one flat hit |
| refresh | **MODELED** | ran | 89 | Status | ✅ batch 2 — self status cure (par/psn/tox/brn, draw-free) |
| counter | **MODELED** | ran | 65 | Phys | ✅ batch 5 — the reactive 2× return |
| return | **MODELED** | ran | 52 | Phys | ✅ batch 5 — happiness-scaled variable BP |
| raindance | **MODELED** | ran | 33 | Status | ✅ batch 2 — weather-SET (Rain, 5-turn timer + upkeep/expiry) |
| endeavor | **MODELED** | ran | 30 | Phys | ✅ batch 5 — the hp delta |
| sleeptalk | **MODELED** | ran | 29 | Status | ✅ batch 5 — the sample + called move |
| aromatherapy | **MODELED** | ran | 27 | Status | ✅ batch 2 — team status cure (clearStatus banner) |
| sunnyday | **MODELED** | ran | 23 | Status | ✅ batch 2 — weather-SET (Sun, 5-turn timer) |
| thunder | MISMODELED | ran | 22 | Spec | onModifyMove: never-miss in rain / 50% in sun — runs flat 70% |
| perishsong | **MODELED** | ran | 21 | Status | ✅ batch 6 — the order-12 field counter + the duration-end continue |
| screech | **MODELED** | ran | 18 | Status | ✅ batch 2 — foe −2 Def (acc-85 draw + draw-free boost) |
| knockoff | **MODELED** | ran | 17 | Phys | ✅ batch 1 — item removal (`onAfterHit`) |
| thief | **MODELED** | ran | 17 | Phys | ✅ batch 1 — item steal (`onAfterHit`) |
| meanlook | **MODELED** | ran | 12 | Status | ✅ batch 6 — the linked firm-trap volatile (+ spiderweb/block) |
| endure | **MODELED** | ran | 10 | Status | ✅ batch 6 — the shared-stall survive-at-1 |
| overheat | **MODELED** | ran | 6 | Spec | ✅ batch 1 — self −2 SpA + the selfDrops `random(100)` |
| destinybond | **MODELED** | ran | 6 | Status | ✅ batch 6 — the mutual-faint window |
| lightscreen | **MODELED** | ran | 5 | Status | ✅ batch 2 — side screen ½ special (5 turns + the both-screens ModifyDamagePhase1 shuffle) |
| superpower | **MODELED** | ran | 5 | Phys | ✅ batch 1 — self −1 Atk/−1 Def + the selfDrops `random(100)` |
| encore | **MODELED** | ran | 5 | Status | ✅ batch 6 — the lock + onOverrideAction |
| charm | **MODELED** | ran | 4 | Status | ✅ batch 2 — foe −2 Atk (acc-100 draw + draw-free boost) |
| snatch | UNMODELED | panic | 4 | Status | steal foe's next self-targeted move — **the ONE residual** (deferred, unprobed; fail-loud) |
| bellydrum | **MODELED** | ran | 3 | Status | ✅ batch 6 — the 2·hp<=maxhp gate + the +6 SET |
| metalsound | **MODELED** | ran | 3 | Status | ✅ batch 2 — foe −2 SpD (Soundproof-immune) |
| charge | **MODELED** | ran | 3 | Status | ✅ batch 6 — ×2 next Electric, consumed by any next move |
| reversal | **MODELED** | ran | 3 | Phys | ✅ batch 5 — the 48·hp/maxhp band ladder |
| hyperbeam | **MODELED** | ran | 2 | Phys | ✅ batch 4c — the mustrecharge lock (locked Recharge request, zero-draw cant turn) |
| memento | **MODELED** | ran | 1 | Status | ✅ batch 6 — the zero-draw self-faint + drops |
| mimic | **MODELED** | ran | 1 | Status | ✅ batch 6 — the slot overlay + revert |
| psychup | **MODELED** | ran | 1 | Status | ✅ batch 6 — the verbatim copy |
| flail | **MODELED** | ran | 1 | Phys | ✅ batch 5 — the band ladder |
| lowkick | **MODELED** | ran | 1 | Phys | ✅ batch 5 — the weighthg ladder |
| painsplit | **MODELED** | ran | 1 | Status | ✅ batch 6 — the clamped floor-average |
| frustration | **MODELED** | ran | 1 | Phys | ✅ batch 5 — the 255-mirror |
| solarbeam | **MODELED** | ran | 1 | Spec | ✅ batch 4c — the two-turn charge (locked fire request, sun skip, weather BP-halve) |
| waterspout | MISMODELED | ran | 1 | Spec | variable-BP (HP-scaled) — runs at flat bp 150 |
| doomdesire | **MODELED** | ran | 2 | Phys | ✅ batch 4c — the slot-keyed order-11 future strike (cast-time typeless snapshot; + futuresight) |

## Class-grouped batch plan (ordered by team-unlock; the roadmap)

Greedy CLASS set-cover (cum = teams fully bit-for-bit playable after modeling this + all above,
starting from 8). Per class: members, gen3 mechanic, whether it needs new RNG draws or is
draw-free, risk. Rows in greedy order.

| # | Class | cum | Members | Mechanic (gen3) | Draws? | Risk / notes |
|---|---|---|---|---|---|---|
| 1 | C_RAPIDSPIN | 11 | Rapid Spin | a 20-BP damaging move whose `onAfterHit` CLEARS the user's side hazards (Spikes) + the user's Leech Seed + partial-trap. Runs today as plain 20 damage, side-effect dropped. | draw-free | **Quick win, huge co-blocker (298 teams).** Reuse the existing `SideState::spikes` + `leech_seed` clears; apply after a landed hit. |
| 2 | C_RECOIL | 23 | Double-Edge (Struggle already modeled) | recoil = `floor(dmgDealt·recoil[num/den])`, applied to the USER draw-free after the hit; DE = ⅓. | draw-free | **Quick win.** Struggle already computes `floor(dmg/4)` recoil via `apply_damage`/`damage_of` — Double-Edge is the same shape with `[1,3]`. Rock Head negates (already a no-op ability). |
| 3 | C_CURSE | 51 | Curse | GHOST user: pay ½ HP, lay the `curse` volatile (¼ HP/turn residual); NON-Ghost: +1 Atk/+1 Def/−1 Spe (self-boost, draw-free). Type-conditional at `onModifyMove`. | draw-free | Medium. Two branches; the Ghost branch adds a new residual + a self-HP cost. Non-Ghost branch is a self_boost_spec variant (mostly reuses setup machinery). |
| 4 | C_DELAYED_HEAL | 78 | Wish | slot-keyed heal of the RECIPIENT's `maxhp/2` at end-of-NEXT turn (duration 2, survives faint/switch/phaze; double-Wish fails). | draw-free | **Python side already has `wish_belief.py` + a reserved obs slot.** New pending-heal residual state per side; a residual-order slot. Draw-free but stateful. |
| 5 | C_FOCUSPUNCH | 115 | Focus Punch | `beforeTurnCallback` queues a `|-singleturn|Focus Punch` at turn start; an `onTryMove` CANCELS the punch if the user took damage this turn. | +draw-order | **Risky.** Adds a new beforeTurn action-queue phase + a "took damage" flag read at execution → touches the turn loop's draw order (the whole reason `isModeledMove` rejects it). Probe the exact `beforeTurn` insertion + cant draw model. |
| 6 | C_DRAIN | 180 | Giga Drain (+ Absorb / Mega Drain / Leech Life if present) | heal the USER `floor(dmgDealt/2)` after the hit, draw-free. Liquid Ooze reverses it (already fail-loud-excluded). | draw-free | **Big lever, low risk.** Same post-hit HP-apply shape as Leech Seed's residual heal; reuse `apply_heal`. |
| 7 | C_PURSUIT | 226 | Pursuit | bp 40, DOUBLES (bp 80) + strikes BEFORE a switching foe (`onBeforeSwitchOut` / the pursuit interrupt). | +draw-order | **Risky.** The switch-interrupt (`pursuitfaint`) changes action ordering; the ×2-on-switch is a `basePowerCallback`. Two coupled mechanics. Probe the switch-catch draw model. |
| 8 | C_BATONPASS | 300 | Baton Pass | switch that TRANSFERS the user's boosts + pass-able volatiles (Sub, Leech Seed, confusion, etc.) to the incoming mon. | draw-free | Medium. New "pass state" carried across the switch-in; interacts with every volatile. Draw-free but broad surface. |
| 9 | C_BEATUP | 380 | Beat Up | one hit PER healthy (non-statused) team member, each a fixed base-Atk strike; `basePowerCallback` + multi-strike loop. | +draws | **Risky.** Multi-hit loop each with its own crit/roll draws → a whole new multi-strike draw model. A big unlocker (80 teams' LAST blocker). |
| 10 | C_STATUS_CURE | 428 | Refresh, Heal Bell, Aromatherapy | Refresh = self par/psn/brn cure; Heal Bell/Aromatherapy = whole-team status cure. Draw-free `onHit`. | draw-free | **Quick win.** Status = None clears (Refresh: self; Heal Bell: iterate the team). Soundproof interaction for Heal Bell (probe). |
| 11 | C_REACTIVE | 471 | Counter, Mirror Coat, Bide | Counter = 2× last PHYSICAL dmg taken this turn back at the attacker; Mirror Coat = 2× SPECIAL; Bide = store 2-turn then release. Needs a "damage taken this turn" ledger + a redirect-to-attacker. | +draw-order | **Risky.** Reactive damage needs a per-turn damage ledger + last-attacker tracking + priority handling; the OHKO/fixed-damage family the engine already fail-louds on. Probe each redirect. |
| 12 | C_VARIABLE_BP_STATUS | 506 | Return, Frustration, Flail, Reversal, Low Kick | `basePowerCallback` with dex bp 0 → the engine routes them as Status and fail-louds. Return/Frustration = happiness-scaled (constant per set); Flail/Reversal = HP-scaled; Low Kick = weight-scaled. | draw-free | Medium. Return/Frustration collapse to a per-set constant (compute from happiness at team-build → a fixed BP). Flail/Reversal/Low Kick need an HP/weight→BP table read at use time. Route out of the Status branch. |
| 13 | C_WEATHER_SET | 536 | Rain Dance, Sunny Day (+ Sandstorm/Hail) | set field weather for 5 turns via a status move. The weather RESIDUALS + speed/damage folds already exist (Sand Stream etc.). | draw-free | **Quick win.** The engine already models weather state, chip, speed ×2, and the eachEvent Weather tie-shuffle — this just adds the MOVE that SETS a 5-turn timer (vs the permanent ability weather). Probe the duration/upkeep tick + overwrite rules. |
| 14 | C_ENDEAVOR | 560 | Endeavor | set target HP = user's current HP (if target HP is higher); Normal → Ghost immune. Accuracy-only draw. | acc-only | Medium. Same fixed-damage routing as Seismic Toss (id-gated `fixed_damage_amount` add) but the amount reads BOTH mons' HP. Ghost immunity already handled by `move_is_immune`. |
| 15 | C_ITEM_REMOVAL | 580 | Knock Off, Thief (Covet) | a damaging move whose `onAfterHit` REMOVES (Knock Off) or STEALS (Thief/Covet) the target's item. Runs today as plain damage, item untouched. | draw-free | Medium. Needs `MonState::item` mutation (the batch-3 berry work already added the item-consumption field) + the item-change events. Draw-free. Interacts with Sticky Hold (no-op ability) + Mail (guard). |
| 16 | C_SLEEPTALK | 603 | Sleep Talk | while asleep, `sample` a random NON-Sleep-Talk move and call it (a nested move execution + its draws). | +draws | **Risky.** Recursive move execution + a `sample` draw + the called move's whole draw chain. Only reachable when asleep. |
| 17 | C_STAT_DROP_MOVE | 626 | Screech, Metal Sound (−2 SpD), Charm, Feather Dance (−2 Atk), Tickle, Fake Tears | a standalone foe stat-drop STATUS move: accuracy draw + `boost()` (draw-free, ±6 clamp, Clear Body/etc. immunity). | acc-only | **Quick win.** The structured secondary-boost machinery (`apply_secondary_boost`) + the standalone-status accuracy path already exist — this is a standalone status move whose whole effect is a foe stat drop. |
| 18 | C_ONMODIFYMOVE | 644 | Thunder | `onModifyMove` sets accuracy = ∞ in rain / 50 in sun (vs base 70). A DRAW-relevant accuracy mutation. | draw-relevant | Medium. Fold weather into the effAcc pipeline for Thunder (id-gated). A wrong effAcc flips hit/miss → seed desync, so probe the exact rain/sun accuracy. |
| 19 | C_PERISHSONG | 658 | Perish Song | lay a `perish` field counter on BOTH actives (3 → faint). Residual countdown. | draw-free | Medium. New per-mon perish counter + a residual tick + the both-sides faint. Draw-free. |
| 20 | C_SWITCH_TRAP_MOVE | 670 | Mean Look, Spider Web, Block | the trapping MOVES (`volatileStatus`-based partial trap of the foe's switch). The trapping ABILITIES (Arena Trap/Magnet Pull/Shadow Tag) are already modeled — reuse `is_trapped`. | draw-free | Medium. A new `trapped` volatile source at the switch-legality gate; mostly reuses the existing trapping infrastructure. |
| 21 | C_SELFDROP | 679 | Overheat, Superpower | `move.self.boosts` = a self stat drop AFTER the hit (Overheat −2 SpA, Superpower −1 Atk/−1 Def). Currently RUNS but skips the drop (STATE-only desync). | draw-free | **Quick win.** `boost()` is draw-free — apply the `move.self.boosts` to the user after the hit. State-only fix; low risk. (Verify no `selfDrops` random(100) — gen3 self.boosts are unconditional, draw-free.) |
| 22 | C_ENDURE | 689 | Endure | survive at 1 HP this turn (a Protect-family `stallingMove` with `volatileStatus:'endure'` + an `onDamage` clamp). | +stall-draw | Medium. Reuses the Protect stall-counter draw model (`run_protect`) + a new `onDamage` survive-at-1 clamp; currently fail-loud in `run_protect`. |
| 23 | C_DESTINYBOND | 693 | Destiny Bond | reactive `volatileStatus`: if the user faints before its next move, the attacker faints too. | draw-free | Medium. A reactive volatile + a faint-hook; low frequency. |
| 24 | C_ENCORE | 697 | Encore | lock the foe into its last move for N turns (a selection restriction). | +duration-draw | Medium. Reuses the taunt/disable selection-restriction + residual-duration machinery; a duration draw to probe. |
| 25 | C_SCREEN | 702 | Light Screen (Reflect if present) | side screen halving special (Light Screen) / physical (Reflect) damage for 5 turns. | draw-free | **Quick win.** The damage calc ALREADY folds `light_screen`/`reflect` (see `damage.rs`); this adds the MOVE that sets the 5-turn side condition. |
| 26 | C_SNATCH | 706 | Snatch | steal the foe's next self-targeted move this turn (a reactive redirect). | +draw-order | Risky-small. Reactive move-steal; rare. |
| 27 | C_BELLYDRUM | 709 | Belly Drum | pay ½ max HP, set Atk to +6 (an HP-cost boost, `onHit` not a declarative `boosts`). | draw-free | Quick-small. HP cost + a max-out to +6. |
| 28 | C_CHARGE_VOL | 712 | Charge | a `charge` volatile that ×2 the next Electric move + a +1 SpD. | draw-free | Quick-small. New volatile + a next-move damage fold. |
| 29 | C_TWOTURN | 715 | Hyper Beam (recharge), Solar Beam (charge) | 2-turn moves: Solar Beam charges turn 1 then hits turn 2 (skipped in sun); Hyper Beam hits then `mustrecharge` locks turn 2. Currently RUN as instant (draw + state desync). | +draw-order | **Risky.** A two-turn action model (charge/recharge volatile + a locked second turn) — touches the turn loop. Solar Beam's sun-skip + Hyper Beam's recharge each add/skip draws. |
| 30 | C_FUTUREMOVE | 717 | Doom Desire (Future Sight if present) | `flags.futuremove` + `onTry` queues a DELAYED strike that lands 2 turns later (bypasses the current turn). Currently RUNS as an instant hit → state + draw desync. | +draw-order | Medium-risky. A new delayed-move queue slot (like Wish but a damaging strike) + the strike-resolution draws 2 turns out. |
| 31 | C_BOOST_COPY | 718 | Psych Up | copy the foe's boost stages onto the user. | draw-free | Quick-small. |
| 32 | C_MOVE_COPY | 719 | Mimic (Sketch if present) | copy the foe's last move into a slot for the battle. | draw-free | Small. New per-slot move override. |
| 33 | C_PAINSPLIT | 720 | Pain Split | set both actives' HP to the average of the two. | draw-free | Quick-small. |
| 34 | C_SELF_FAINT | 721 | Memento | user faints, foe −2 Atk/−2 SpA. | draw-free | Small. Self-KO (reuse Explosion self-faint machinery) + a foe stat drop. |
| 35 | C_VARIABLE_BP_DMG | 722 | Water Spout (Eruption if present) | HP-scaled bp (`basePowerCallback`, dex bp 150). Runs at flat 150 today. | draw-free | Small. Add the `basePowerCallback` (bp·curHP/maxHP) for the id — it already reaches the damaging path. |

### Grouping quick-wins vs risky

- **Quick wins (draw-free, reuse existing machinery):** C_RAPIDSPIN (1, the biggest co-blocker —
  298 teams), C_RECOIL (2), C_DRAIN (6), C_STATUS_CURE (10), C_WEATHER_SET (13), C_STAT_DROP_MOVE
  (17), C_SELFDROP (21), C_SCREEN (25), + the small tail (C_BELLYDRUM/CHARGE_VOL/BOOST_COPY/
  PAINSPLIT/SELF_FAINT/VARIABLE_BP_DMG). Big early unlockers (rapidspin 298-carry, recoil
  266-carry, drain 182-carry) at low risk.
- **Risky (touch the turn loop / draw order):** C_FOCUSPUNCH (5), C_PURSUIT (7), C_BEATUP (9),
  C_REACTIVE (11), C_SLEEPTALK (16), C_TWOTURN (29), C_FUTUREMOVE (30). These add new action
  phases, multi-strike loops, reactive ledgers, delayed strikes, or nested move execution — the
  bit-for-bit hard part. Do them with dedicated probes (the project's `probe_*.js` pattern) first.
- **Stateful-but-draw-free:** C_CURSE (3), C_DELAYED_HEAL/Wish (4), C_BATONPASS (8),
  C_ITEM_REMOVAL (15), C_PERISHSONG (19). New persistent state (pending heal / passed volatiles /
  item change / perish counter) but no new draws — medium risk, mostly a residual-order +
  carry-across-switch question.

## Ambiguous "modeled vs not" — flag for a probe

These RUN today (no fail-loud) but SILENTLY MIS-MODEL a sub-mechanic — a `--use-bridge=rust`
battle would desync without crashing. **These are the priority audit targets** (they'll bite
before the fail-loud moves, which at least crash honestly). Confirm each with a dedicated probe:

- **rapidspin** — hazard/trap clear NOT applied → the user keeps Spikes/Leech Seed the sim
  cleared (STATE desync; **#298, the most-carried MISMODELED move**).
- **doubleedge** — recoil not applied (STATE desync every hit; #266).
- **gigadrain** — drain heal not applied (STATE desync; #182).
- **focuspunch** — beforeTurn queue + cancel-if-hit not run (DRAW-ORDER desync; #196).
- **pursuit / beatup / waterspout** — variable-BP runs at the flat placeholder BP (STATE desync;
  pursuit also mis-orders on a switch). #159 / #114 / #1.
- **knockoff / thief** — item removal/steal `onAfterHit` not applied (STATE desync — the target
  keeps its item, so every later item-fold diverges; #17 / #17).
- **thunder** — accuracy not weather-mutated (rain never-miss / sun 50%) → a hit/miss flip =
  SEED desync; #22.
- **overheat / superpower** — self stat-drop not applied (STATE desync; #6 / #5).
- **doomdesire** — future-move runs as an INSTANT hit instead of a queued 2-turns-out strike
  (DRAW + STATE desync; #2).
- **hyperbeam / solarbeam** — 2-turn lock/charge collapsed to instant (DRAW + STATE desync;
  #2 / #1).

**METHOD NOTE for the maintainer:** the STATIC dex-flag heuristic ALONE is insufficient — it
first mis-called rapidspin / knockoff / thief / doomdesire MODELED (they carry no charge/recoil/
drain/multihit flag, just an `onAfterHit`/`futuremove`). The `scan_move_coverage.js` classifier
now audits every "modeled" damaging move's dex handlers (`onTry`/`onHit`/`onAfterHit`/`onModifyMove`
/`self`/`futuremove`) and the `scan_move_probe` binary is the authoritative confirm. Re-run the
handler audit whenever the modeled set grows.

The 31 UNMODELED moves are SAFE in the sense that they fail-loud (crash, no silent desync) — the
engine's fail-loud law holds for them. The 11 MISMODELED above are the ones to fix or explicitly
block from `--use-bridge=rust` picking until modeled.
