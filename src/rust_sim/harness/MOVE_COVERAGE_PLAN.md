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
  - **MISMODELED (`--use-bridge=rust` would diverge silently):** Focus-Punch's beforeTurn queue +
    Pursuit's variable-BP are now MODELED (**BATCH 4a**, `gen3_move_coverage_batch4_v1`), and Beat Up's
    multi-strike stat-swap + Water Spout's variable-BP + Thunder's rain-accuracy are now MODELED
    (**BATCH 4b**, `gen3_move_coverage_batch4b_v1`). The ONLY remaining MISMODELED classes are Doom
    Desire's future strike + Hyper Beam's recharge — the *dangerous* ones (a silent desync, not a crash).
  - **23 UNMODELED** — the engine FAIL-LOUDs (`panic!` / `is not modeled`). Honest crash, no desync.
- **Teams already FULLY engine-playable (every move MODELED bit-for-bit): 71 / 722** (was 8 pre-batch;
  batches 1+2 combined).
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
duration tie, and Beat Up setting the target's Focus-Punch lostFocus). The ONLY MISMODELED classes left are
Doom Desire (future strike) + Hyper Beam (recharge). See `src/rust_sim/CLAUDE.md` → the batch-4b section.

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
| counter | UNMODELED | panic | 65 | Phys | reactive: returns 2× physical dmg taken |
| return | UNMODELED | panic | 52 | Phys | variable-BP, dex bp 0 → routed Status → fail-loud |
| raindance | **MODELED** | ran | 33 | Status | ✅ batch 2 — weather-SET (Rain, 5-turn timer + upkeep/expiry) |
| endeavor | UNMODELED | panic | 30 | Phys | sets target HP = user HP |
| sleeptalk | UNMODELED | panic | 29 | Status | picks + calls a random other move while asleep |
| aromatherapy | **MODELED** | ran | 27 | Status | ✅ batch 2 — team status cure (clearStatus banner) |
| sunnyday | **MODELED** | ran | 23 | Status | ✅ batch 2 — weather-SET (Sun, 5-turn timer) |
| thunder | MISMODELED | ran | 22 | Spec | onModifyMove: never-miss in rain / 50% in sun — runs flat 70% |
| perishsong | UNMODELED | panic | 21 | Status | field perish counter (both sides faint in 3) |
| screech | **MODELED** | ran | 18 | Status | ✅ batch 2 — foe −2 Def (acc-85 draw + draw-free boost) |
| knockoff | **MODELED** | ran | 17 | Phys | ✅ batch 1 — item removal (`onAfterHit`) |
| thief | **MODELED** | ran | 17 | Phys | ✅ batch 1 — item steal (`onAfterHit`) |
| meanlook | UNMODELED | panic | 12 | Status | switch-trap volatile |
| endure | UNMODELED | panic | 10 | Status | survive-at-1-HP (`onDamage`, not Protect) |
| overheat | **MODELED** | ran | 6 | Spec | ✅ batch 1 — self −2 SpA + the selfDrops `random(100)` |
| destinybond | UNMODELED | panic | 6 | Status | reactive: KOs the attacker if user faints |
| lightscreen | **MODELED** | ran | 5 | Status | ✅ batch 2 — side screen ½ special (5 turns + the both-screens ModifyDamagePhase1 shuffle) |
| superpower | **MODELED** | ran | 5 | Phys | ✅ batch 1 — self −1 Atk/−1 Def + the selfDrops `random(100)` |
| encore | UNMODELED | panic | 5 | Status | locks foe into last move |
| charm | **MODELED** | ran | 4 | Status | ✅ batch 2 — foe −2 Atk (acc-100 draw + draw-free boost) |
| snatch | UNMODELED | panic | 4 | Status | steal foe's next self-targeted move |
| bellydrum | UNMODELED | panic | 3 | Status | −½ HP, +6 Atk |
| metalsound | **MODELED** | ran | 3 | Status | ✅ batch 2 — foe −2 SpD (Soundproof-immune) |
| charge | UNMODELED | panic | 3 | Status | charge volatile (×2 next Electric) |
| reversal | UNMODELED | panic | 3 | Phys | variable-BP, dex bp 0 → routed Status → fail-loud |
| hyperbeam | MISMODELED | ran | 2 | Phys | recharge lock (`mustrecharge`) not applied |
| memento | UNMODELED | panic | 1 | Status | user faints, foe −2 Atk/−2 SpA |
| mimic | UNMODELED | panic | 1 | Status | copy foe's last move |
| psychup | UNMODELED | panic | 1 | Status | copy foe's boosts |
| flail | UNMODELED | panic | 1 | Phys | variable-BP (bp0 → fail-loud) |
| lowkick | UNMODELED | panic | 1 | Phys | variable-BP (bp0 → fail-loud) |
| painsplit | UNMODELED | panic | 1 | Status | average both HPs |
| frustration | UNMODELED | panic | 1 | Phys | variable-BP (bp0 → fail-loud) |
| solarbeam | MISMODELED | ran | 1 | Spec | 2-turn charge collapsed to 1 turn (skips the charge turn + draws) |
| waterspout | MISMODELED | ran | 1 | Spec | variable-BP (HP-scaled) — runs at flat bp 150 |
| doomdesire | MISMODELED | ran | 2 | Phys | future-move (delayed strike) runs as an instant hit |

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
