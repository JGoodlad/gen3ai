---
name: project_rust_sim_port
description: "The from-scratch Rust reimplementation of Pokémon Showdown (gen3ou, bit-for-bit) — src/rust_sim, capstone e2e fuzz DONE, NOT shipped"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6c77361-e444-44c6-bacd-a2bee362b610
---

> **Archived 2026-09-08** — 127 KB build log; src/rust_sim/CLAUDE.md + designs/rust_sim/port_build_log.md are the always-current docs of record. Preserved verbatim; nothing below is current.

A from-scratch **Rust port of the Pokémon Showdown battle sim**, scoped to **Gen 3 OU
singles**, whose hard invariant is **bit-for-bit identical output to upstream Showdown given
the same seed + teams + choices** (= matching the PRNG consumption ORDER+COUNT, not just the
mechanics). Built across the 2026-06-26/27 session. Lives in **`src/rust_sim/`** (crate
`pokesim`, **std-only, zero deps**, gen-generic behind a `gen` param). **NOT shipped** — all in
the worktree branch (the only main-checkout diffs are additive extractor fields:
`gen3_moves.json` critRatio/secondaryBoosts, `gen3_species.json` maxHP, `tools/`), awaiting
`/gen3ai-ship`. Full as-built detail in **`src/rust_sim/CLAUDE.md`** (the leaf doc, kept current).

**Built + validated, each by a DIFFERENTIAL golden** (a Node/Python harness captures the REAL
sim, a Rust test pins it): PRNG (ChaCha20 SodiumRNG + Gen5 LCG, ~2900 vectors), JSON reader,
dex (over `data/pokemon/*.json`, same source as `agents.gen3_data`), team unpack/pack, gen3
stats, state/`>start`, damage calc (31 exact scenarios), event dispatch (speed_sort + Fisher-
Yates tie-shuffle), and `turn.rs`: full turns → multi-turn+residuals → SWITCHING+post-faint →
**secondaries+onBeforeMove status** → **full battles to WIN/LOSS**.

**THE CAPSTONE (the user's "at the end, full e2e fuzz on real teams comparing output" ask) — DONE:**
`harness/gen_e2e_fuzz.js` + `tests/e2e_fuzz_test.rs` drive BOTH the omniscient in-process
BattleStream AND Rust `run_full_battle` over **REAL `data/teams/` teams** (770 → 719 valid; only
**18 are filter-clean** = all-modeled-ability+item) for **220 complete random battles to game-end**,
asserting per-decision state+status+boosts+confusion+**running PRNG seed**+winner **bit-for-bit**.
Gate is **STRICT** (`filtered_diverged==0`, `filtered_matched==220`, **no escape hatch**). After the
move layers below it carries **14378 seed + 26690 status + 26690 boost assertions, 220 wins**; the full
`cargo test` = **20 green binaries + 65 lib units**. Built via dynamic workflows (build→review, then
fix→verify), 3-lens adversarial review each round.

**Layers added AFTER the capstone** (each a workflow: build + differential golden + 3-lens review,
then e2e-allowlist-expand + regen + strict-gate re-confirm): **standalone STATUS MOVES** (Thunder
Wave/Stun Spore/Glare/Poison Powder/Poison Gas/Toxic/Will-O-Wisp + the 6 sleep moves — accuracy →
move-type immunity (TWave→Ground, Glare→Ghost) → try_set_status + the sleep `random(2,6)` duration +
Toxic stage-0 + Sleep Clause) and **self-targeting SETUP/BOOST MOVES** (17: Calm Mind/Dragon Dance/
Swords Dance/Agility/Bulk Up/Amnesia/Tail Glow/+Def&+Atk singles — draw-free `boost()`, data-driven
from a new obs-neutral `selfBoosts` extractor field); **RECOVERY MOVES** (Recover/Soft-Boiled/Slack
Off/Milk Drink = floor(maxhp/2); Moonlight/Synthesis/Morning Sun = gen3 PLAIN-integer weather heal
½/⅔/¼; Rest = full heal + status cure + a self-sleep that DRAWS-then-DISCARDS one `random(2,6)` and
overwrites to fixed Sleep(3)); and **PROTECT/DETECT** (the gen3 stall draw: FIRST protect draws
NOTHING, consecutive draws `randomChance(1,counter)` at floored 2/4/8, gen3 has NO delete-on-fail so
fails re-roll at the same denom; a blocked foe move draws its accuracy THEN is blocked, nothing
further); **SPIKES** (the first SIDE CONDITION — SideState.spikes 0..=3, never-miss draw-free move,
switch-in chip maxhp/8 ÷6 ÷4, Flying/Levitate immune, KO-on-entry chains a replacement); and **PHAZING**
(Roar/Whirlwind — random forced-switch: gen3 resolves to accuracy:100 NOT never-miss so it DRAWS, the
`sample` draws even at n==1, the dragged mon takes Spikes). The e2e capstone picks status/setup/recovery/
protect/SPIKES (latest regen: 14070 decisions, 9865 spikes-using, STRICT 220/220). **PHAZE is EXCLUDED
from the e2e** (`PHAZE_E2E_EXCLUDED`): bit-for-bit in its OWN golden + 3 regression pins, but the e2e
surfaced an UNRESOLVED stateful desync (multiple phazes + a secondary in one battle → a later phaze's
`sample` reads a SHIFTED PRNG position; SEED still matches = draw-POSITION not draw-COUNT bug). Documented
as a known bug in `EDGE_CASES.md`; the agent honestly excluded it rather than fake a pass — the e2e fuzz
catching a stateful bug the single-scenario goldens missed. Then **LEECH SEED** (7th move layer): the
move (acc 90, Grass-immune, already-seeded-fail) + a new LeechSeed residual at order 10/subOrder 5
(Leftovers s4 → LEECH s5 → status-DoT s6), drain maxhp/8 → seeder-active heal, draw-free; INCLUDED in
the e2e (draw-free, can't shift the LCG like phaze's sample) but exercises **0 e2e decisions** (no clean
team carries it — honestly disclosed) so it's validated by its dedicated golden + 3 regression pins.
Then **SUBSTITUTE** (8th move layer): sub HP=maxhp/4, absorbs damage (breaks at 0, no gen3 carry),
blocks status/stat-drop; KEY corrected model (agent overrode my WRONG hint via probe) — the secondary
`random(100)` IS STILL DRAWN against a sub (only the foe-effect suppressed; follow-on `random(2,6)`/
`random(3)` suppressed; a SELF-boost secondary still applies); confusion self-hit hits the MON; phaze
bypasses. Substitute's e2e exclusion surfaced an **8th engine bug** (NOT substitute's fault): a mid-turn
switch-in whose entrant TIES the opposing active under FRESH weather was missing the `setWeather →
eachEvent('WeatherChange')` speed-tie shuffle (e2e_84). **FIXED** in a follow-up workflow (`run_switch`
detects an actual weather CHANGE → fires exactly one tie-shuffle, gated so a no-op re-set draws nothing;
the draw-free `>start` path untouched). So **SUBSTITUTE is now IN the e2e** (284 sub-move decisions,
strict 220/220) + a revert-verified pin `switch_into_a_tie_under_sand_…`. **25 cargo suites / 117 tests
green.** Regression suite = 18 pins (8 bugs + phaze 3 + leech 3 + substitute 5 — well, the switch-tie one
makes 18). **1 KNOWN UNFIXED bug** (EDGE_CASES.md, honestly e2e-excluded): the phaze multi-sample
draw-POSITION desync (re-enabling phaze ALSO surfaced an unmodeled `destinybond` move) — DIFFERENT class
from the (now-fixed) switch-tie draw-COUNT bug.

**6 real engine bugs the e2e fuzz surfaced + FIXED** (only real teams reach them): (1) Water/Volt
Absorb HEAL `floor(maxhp/4)` (was bare immunity); (2) Intimidate respects foe `onTryBoost` immunity;
(3) **residual-vs-faint under weather** (the deep one) — Showdown runs `faintMessages()` after EACH
residual handler + `if(this.ended)return` (a fast burned mon's DoT self-KO ends before a slower foe's
Leftovers heal), AND eachEvent/residual tie-shuffles read a **CACHED `pokemon.speed`** (re-cached only
at turn-start/residual-start/switch-in) so a mon paralyzed WHILE active keeps full speed through the
move phase but one that SWITCHES IN paralyzed ties on para-speed at once; (4) Toxic stage reset on
switch-in (a pivot-out-and-back mon over-chipped); (5) Water/Volt Absorb now accuracy-gated (a MISSED
move no longer heals); (6) residual handler GATHER order (the selection-sort tie-shuffle permutes
handlers in pre-sort/gather order, so status-DoT must be GATHERED before Leftovers — Showdown gathers
STATUS→volatile→ITEM). Plus the gen3ou **`SetStatus` 2-clause tie-shuffle** draw (Sleep+Freeze Clause
Mods are runEvent handlers → a size-2 Fisher-Yates draw on every status apply; gen3customgame has 0 →
no shuffle; the omniscient e2e/prior goldens run customgame, the status golden runs gen3ou to exercise
it). All SIM-probe verified.

**Deferred tail** (e2e taxonomy ranks it by static team-composition carry, move-level-blind): Wish
(delayed heal), Endure, Heal Bell/Aromatherapy/Refresh (status cure), phaze (Roar/Whirlwind), entry
hazards (Spikes), Substitute, Baton Pass, Pursuit, Leech Seed, recoil/drain/Explosion/Hidden-Power(var
BP)/selfDrops moves, abilities Natural Cure/Torrent/Magnet Pull/Arena Trap/Guts/berries, and PROTOCOL
emission (the `|...|` byte stream — the level-2 gate poke-env would parse, the big remaining piece).
Real gen3ou is saturated with the deferred set (only 18/719 teams fully modelable).

**Edge-case backlog** in `src/rust_sim/EDGE_CASES.md` (user-noted 2026-06-28, NOT yet built): the big
theme is **TRAPPING + choice/request VALIDATION** which the e2e fuzz currently SIDESTEPS (it only
submits sim-legal choices, so a trapped-mon switch is never offered) — a standalone drop-in port must
compute `trapped` in LegalActions + handle the **server REJECTING an illegal switch + re-requesting**.
Items: Arena Trap (Dugtrio, grounded foes; mutual-trap when both are Dugtrio), Magnet Pull (Steel,
Magneton mirror), Mean Look/Spider Web/Block (trap MOVES; Ghost immune), Hyper Beam mustrecharge,
Substitute (HP cost, absorbs, blocks status/secondary, NOT phazing), Perish Song, **Taunt** (blocks
selecting status moves) + **Struggle** (forced when no usable move; gen3 typeless 50BP, recoil=¼ damage
dealt) — the MOVE-legality mirror of trapping (both need the port's own LegalActions/request layer).
Each → a named regression test when built. NOTE: user raised the weekly quota ceiling to 95% for the run.

**PROTOCOL EMISSION (level-2, the `|...|` byte stream poke-env parses) — Phase 1 + Phase 2 DONE**
(2026-06-30, both via workflow build→3-lens review, all SHIP). The engine is already bit-for-bit
RNG+state faithful; protocol is a **side output** of events that already happened. `protocol.rs`
carries a `ProtocolBuilder` — an **append-only, PRNG-free** line buffer on `BattleState` (the `log`
field), **default-DISABLED** so the seed path no-ops and draws nothing; `run_full_battle_logged`
enables it, emits framing, runs the SAME `run_full_battle`, returns `(BattleOutcome,
Vec<ProtocolLine>)`. Fiddly formatting is centralized: `MonRef` (`p<N>a: <Name>`), `HpStatus` (the 3
variants `x/y` / `x/y <status>` / `0 fnt`), `Cause` (`[from] item:` / `[from] move:` / `[from]
ability:` / bare). **Phase 1** = framing + turn/upkeep/move(+[miss]/[still])/switch/drag/-damage(+residual
[from] tags)/-heal/faint/-crit/-supereffective/-resisted/-immune/-miss/win/tie. **Phase 2** added the
status-move `|move|` ANNOUNCE + `-boost`/`-unboost`(by clamped-delta sign) + `-status`(+`[from] move:
Rest`)/`-curestatus`(+`[msg]`)/`cant` + `-weather`(set `[from] ability`+`[of]` / `[upkeep]` tick) +
`-ability`(Intimidate) + `-fail`(recover/rest-full, spikes-cap, stall-fail, phaze-no-bench, Substitute
`[weak]`) + `-sidestart`(Spikes) + Substitute `-start`/`-end`/`-activate|[damage]`(survived absorb
emitted INSTEAD of -damage; new SubAbsorb enum) + Protect `-singleturn`/`-activate` + Rest
`-heal|<hp> slp|[silent]`. The byte-diff gate `tests/protocol_test.rs` replays the capture golden
(`protocol_capture_golden.txt`) through `run_full_battle_logged`, filters both sides to the emitted
line types, asserts BYTE-EQUALITY: **51 battles / 5630 lines byte-equal** (Phase 1 was 30/1512), 9
scenarios. **OBSERVATION-ONLY PROVEN** — the whole seed suite stays green with BYTE-IDENTICAL counts
(e2e STRICT 14228 diverged 0, battle 2034, fullbattle 2053); only 2 engine changes, both
state/seed-invariant EMISSION reorders (the `|turn|N+1` marker to the next-turn loop top; the
weather-chip `-damage` order reads `each_event_shuffle`'s returned permutation). **136 cargo tests
green, 0 fail.** Still deferred (Phase 3+, each needs a new MECHANIC not just emission): **FIXED-DAMAGE
moves** (Seismic Toss / Night Shade — a `damageCallback` the engine currently no-ops; it BLOCKS the 2
remaining protocol scenarios status_para_and_boost_drop + secondary_status_flinch, both all-Seismic-Toss)
+ **Struggle / PP tracking** (3 recover_and_rest battles skipped) + the line types no captured battle
reaches (-sethp/-cureteam/-setboost/Haze family/-item/-prepare/-mustrecharge/-transform). Two LATENT
review findings (unreachable in any asserted scenario today, non-blocking): the non-Substitute `-damage`
emit is gated on `realized > 0` (a future 0-damage move would drop the line); `emit_switchin_ability_lines`
re-derives Intimidate immunity + hardcodes `atk|1` instead of reading the applied delta. Design:
`PROTOCOL_EMISSION_DESIGN.md`; grammar: `protocol_inventory.md`.

**FIXED-DAMAGE MOVES — DONE, bit-for-bit** (2026-06-30, workflow build→3-lens review, all SHIP).
The 9th move class: `damageCallback`/`damage:` moves that BYPASS getDamage (so NO crit + NO 16-way
damage roll — accuracy is the ONLY per-move draw, probe-confirmed vs `probe_fixeddamage_rng.js`).
Modeled bit-for-bit: **Seismic Toss / Night Shade** (`damage:'level'` = user's level), **Sonic Boom**
(20), **Dragon Rage** (40), **Super Fang** (`max(floor(target.hp/2),1)`). Draw gotchas the probe
settled: Seismic Toss/Night Shade/Dragon Rage are acc-100 but NOT never-miss → they STILL draw 1
accuracy roll (the phaze precedent); Sonic Boom/Super Fang are acc-90 (can miss); type immunity
(Fighting 0× Ghost → Seismic Toss immune; Ghost 0× Normal → Night Shade; Normal 0× Ghost → Sonic
Boom/Super Fang; Dragon Rage has NO gen3 immunity) is accuracy-drawn-THEN-`-immune`, SAME seed as a
landed hit; **Super Fang behind a Substitute halves the MON's hp not the sub's** (the fixed number then
routes through the existing `absorb_into_sub`). Routed in `run_move` BY ID (`is_fixed_damage_move`)
BEFORE the `category==Status` branch — the crux is these carry `basePower:0` so `derive_category`
mislabels them Status. The DEFERRED fixed-damage family (Psywave [variable RNG], the OHKO moves
Fissure/Horn Drill/Guillotine, Counter/Mirror Coat/Bide [reactive], Endeavor) is in
`is_fixed_damage_move` but has NO `fixed_damage_amount` → **PANICS fail-loud** in
`run_fixed_damage_move` (never a silent bp==0 no-op); pinned by a new `#[should_panic]` unit test
`deferred_fixed_damage_move_panics_fail_loud` (Counter). Validated by `tests/fixeddamage_test.rs` (9
scenarios × 80 seeds, 720 runs, 4144 seed + 8288 HP assertions, 2469 FD-hit decisions, 0 diverged) + 4
REVERT-VERIFIED regression pins (`seismic_toss_deals_user_level_damage`,
`seismic_toss_into_a_ghost_is_immune_accuracy_only_seed`, `night_shade_into_a_normal_is_immune`,
`fixed_damage_into_a_substitute`; ground truth from `probe_fixeddamage_regression_rng.js`).
**e2e:** `MODELED_FIXED_DAMAGE_MOVES` added to the allow-list, STRICT stays **220/220 diverged 0 /
14228** (byte-identical) — but **0 fixed-damage decisions** (no filter-clean team carries one, the
leech-seed situation) so it's proven by the dedicated golden + pins, honestly disclosed. **83 lib +
all 27 bins green** (regression 22→26). The 2 all-Seismic-Toss protocol scenarios (status_para_and_
boost_drop / secondary_status_flinch) stay deferred but their reason CHANGED: the Seismic-Toss lines
now replay BYTE-EXACT; the remaining blocker is a **forced-replacement request-boundary resume**
desync (a switching-layer nuance, same family as `forced_replacement_recaches_speed_seed`), NOT
fixed-damage — honestly re-characterized. Review nit noted (env-only, non-defect): `cargo test` can
FLAKE under heavy load if the shared `data/pokemon/*.json` is mid-rewrite by another process — run
`--test-threads=1` for a clean deterministic pass. NEXT candidates: the 2 open switching-layer bugs
(spikes-cascade re-enabling Explosion; phaze desync; the forced-replacement-resume one now ALSO blocks
the last 2 protocol scenarios) + the LegalActions/request-validation layer (trapping/Taunt/Struggle).

**EXPLOSION now e2e-INCLUDED bit-for-bit** (2026-06-30, workflow build→3-lens review, all SHIP;
regression 26→28). The "double-faint→double-replacement→SPIKES-cascade" bug that kept Explosion out of
the strict e2e turned out to be **TWO distinct draw-free STATE bugs** (the docs' cascade-chip hypothesis
was directionally right, mechanism wrong), both found by regenerating the e2e golden with Explosion
re-admitted (exactly 2 of 220 battles diverged, both seed_ok):
(1) **stale runSwitch survives a cascade** (`cancel_active_actions`) — in gen-3 singles `faintMessages`
runs `cancelAction(pokemon)` over `getAllActive`, which drops EVERY queued action of every active mon
INCLUDING the OTHER side's still-pending order-101 `runSwitch` (a runSwitch's `action.pokemon` is the
entrant). The port's comment wrongly said runSwitch isn't cancelled, so when the first entrant fainted on
its own Spikes, the foe's stale runSwitch re-chipped its already-settled entrant. Fix: cancel the
runSwitch when the side's active is NOT fainted (exact `cancelAction(getAllActive)` semantics).
(2) **confusion self-hit dropped Choice Band** (`apply_confusion_self_hit`) — gen-4 confusion (gen-3
inherits) uses the FULL `getDamage`, so CB ×1.5 folds into the typeless '???' self-hit; the port passed
empty atk_stat_mods. Fix: `resolve_atk_stat_mods(item, None, Physical)`.
Both DRAW-FREE (seed untouched); e2e now 220/220 diverged 0, 544 explosion decisions; the `==0` tripwire
flipped to a `>=50` coverage floor. 2 new revert-verified pins
(`double_replacement_cascade_does_not_rechip_the_other_sides_entrant`, `confusion_self_hit_applies_choice_band`).
Known scope gap (pre-existing, not gate-exercised, logged EDGE_CASES.md): the confusion self-hit still
ignores the confused mon's OWN-side Reflect (gen-3 Reflect reduces the typeless-physical self-hit) — a
STATE-only gap, no filter-clean battle hits it.

**PHAZE now e2e-INCLUDED bit-for-bit — the LAST e2e exclusion CLOSED** (2026-07-01, workflow
build→3-lens review all SHIP; regression 28→29). Prompted by the user's principle
([[feedback_determinism_means_fixable]]): "it HAD to be fixable — Showdown is deterministic." The
long-standing "multi-phaze `sample` draw-POSITION desync" was NOT the array-order bug the docs (and my
workflow hypothesis) guessed — the port's `possibleSwitches`/swap order matched the sim exactly. The REAL
root cause: **gen-3 Roar AND Whirlwind carry the `protect: 1` flag**, so a Protect/Detect on the TARGET
BLOCKS the phaze at `runEvent('TryHit')` (after the accuracy roll) → NO `forceSwitchFlag` → NO `dragIn` →
NO `sample` draw, leaving the target ACTIVE. The port's phaze arm (`run_status_move`) drew accuracy then
signalled the drag UNCONDITIONALLY — it never checked the Protect block that the leechseed/standalone-status
arms already do (`protect_blocks`). So into a Protecting foe the port dragged an EXTRA random mon; the
boundary SEED still matched (the extra `sample` was compensated by the divergent board downstream) while
the dragged mon (STATE) was wrong — which is exactly why a seed-only check missed it and it only surfaced
when a phaze hit a PROTECTING foe across a long history. FIX: a `protect_blocks(foe,foe_slot,false)` check
in the phaze arm (emit `-activate Protect`, no drag). Substitute does NOT block (Roar/Whirlwind carry
`bypasssub: 1`). Pinned by `phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target` (P4,
revert-verified: reverting drags Blissey/Skarmory AND breaks the strict e2e). The `destinybond` a
phaze-clean team can carry is blocklisted + fail-loud (`destinybond_status_move_panics_fail_loud`).
**RESULT: PHAZE_E2E_EXCLUDED=false; ALL 4 e2e layers (spikes/substitute/explosion/phaze) now INCLUDED —
ZERO exclusions, the strict e2e covers EVERY modeled move layer.** 220/220 diverged 0, 13367 decisions
(1035 phaze / 541 explosion / 252 sub / 8414 spikes), 219W+1T; golden byte-reproducible (both reviewers
regenerated the full 220-battle golden byte-identical). 84 lib + 29 regression + all 28 bins green. The
dedicated phaze golden (`phaze_test`, 560 runs, 10388 seed asserts) UNCHANGED. **0 KNOWN UNFIXED BUGS.**
The e2e decision total shifted 14228→13367 (219W+1T) because re-admitting phaze changed which moves are
pickable → different filter-clean battle sampling. Lesson (re-confirmed): a divergence is ALWAYS a fixable
port bug, never inherent — the docs called this "known unresolved" for days; a focused diagnostic workflow
found the concrete `protect:1` cause in one pass.

**PROTOCOL EMISSION now COMPLETE at the scenario level** (2026-07-01, workflow build→3-lens review all
SHIP; regression 29→30). Fixed the last blocker (the "forced-replacement resume desync" that kept 2
all-Seismic-Toss scenarios deferred) — which turned out NOT to be an engine control-flow bug but a
**capture-harness artifact**: `gen_protocol_capture.js`'s `fromPlan` reused a 3-move Tyranitar's move
indices after it was replaced mid-turn by a 2-move Snorlax, submitting an invalid `move 3`. The SIM's
`side.choose` REJECTS the illegal slot (draws 0, leaves the request OPEN, re-requests) and the capture
loop records a phantom zero-draw DEC row; the PORT's `run_full_battle` did NOT validate the slot → it RAN
a full turn (foe's move + residual + Quick Claw drew) → diverged. FIX = a `move_decision_is_legal`
reject-and-re-request gate (skip a `Move(K)` whose `K >= active.set.moves.len()`, draw/emit/record
nothing — mirroring the sim; a genuine robustness step toward the eventual LegalActions layer) + modeling
the already-statused-foe `|-fail|` line the port omitted (SAME status → `|-fail|target|par`; DIFFERENT →
`[still]`+`|-fail|user`; keyed on top-level `move.status`, draw-free). Both ZERO-DRAW → observation-only
(seed suite byte-identical: e2e 13367 diverged 0, battle 2034, fullbattle 2053, secondary 4328).
**protocol_test.rs = 63 battles / 7223 lines byte-equal (was 51/5630), DEFERRED_SCENARIOS EMPTY** — all
11 scenarios asserted; only 3 individual `recover_and_rest` battles stay per-battle-skipped for **Struggle
(no PP tracking)** — a genuinely different, future-mechanic reason. Pinned by
`forced_replacement_resume_runs_the_post_replacement_move_decision` (revert-verified). 147 tests green.
Remaining protocol gaps all need UNBUILT mechanics (Struggle/PP; the -sethp/-cureteam/-setboost/Haze/-item/
-prepare/-mustrecharge/-transform line types no captured battle reaches). The drop-in endgame (a
`BattleStream::write_line` returning `log.drain()` for the bridge's per-side privacy fold) is the last
protocol piece. NEXT frontier: the LegalActions/
request-validation layer (trapping/Taunt/Struggle+PP) — the `move_decision_is_legal` gate is its first
brick.

**PP TRACKING + STRUGGLE — modeled bit-for-bit** (2026-07-01, `gen3_pp_tracking_v1`, workflow build→3-lens
review all SHIP; regression 30→33, 152 tests). The first REAL brick of the LegalActions layer — and it
COMPLETES protocol emission (the 3 recover_and_rest Struggle battles now replay byte-exact → **protocol
66/8721, 0 deferred + 0 SKIPPED**). Probe-settled model (hints wrong twice again): **PP init** = `calculatePP`
with the ctor's hardcoded **3 default PP-ups** → `pp*8/5` integer (Surf 15→24, EQ 10→16, Struggle→1), NOT
raw moveset PP; **decrement** −1 DRAW-FREE, deducted AFTER `runEvent('BeforeMove')` PASSES (a full-para/
sleep/flinch/frozen/confusion-self-hit turn deducts 0), a MISS + an IMMUNE hit STILL decrement, no reset on
switch; **Pressure −2**. **Forced Struggle** when no usable move (all 0 PP, or a Choice-Band lock leaves only
its 0-PP slot); a 0-PP slot while another is usable is rejected draw-free (reuses `move_decision_is_legal`).
**Struggle** = typeless `'???'` (no STAB, hits a Ghost), 50 BP physical, **accuracy 100 → DRAWS** (acc+crit+
dmg+QC, NOT never-miss), **recoil = `max(floor(damage_dealt/4),1)`** (gen3 `calcRecoilDamage` Math.floor —
NOT round, NOT gen4 `trunc(maxhp/4)`). Engine: `MonState.move_pp`/`move_maxpp`/`choice_locked_move` +
`move_usable`/`must_struggle`/`deduct_pp`. DATA: `pp`/`noPPBoosts` added to `gen3_moves.json` via the
extractor (obs-neutral; worktree-local, does NOT touch the user's live training data). OBSERVATION-ONLY
(seed suite byte-identical: e2e 13367 diverged 0, battle 2034, fullbattle 2053, secondary 4328; e2e golden
MD5-reproducible). Validated by `pp_struggle_test.rs` (400 runs, 4424 seed + 8368 PP assertions, 1035
forced-Struggle/recoil, 378 Pressure−2) + 3 revert-verified pins. DEFERRED (fail-loud, the rest of
move-legality): **Taunt / Disable / Torment / Imprison** (extend `must_struggle`/`move_usable`). **Port
status: PRNG+state+protocol bit-for-bit; 0 known bugs; 0 e2e exclusions; PROTOCOL 100% COMPLETE (0 deferred/
skipped).** NEXT: the rest of LegalActions — trapping (Arena Trap/Magnet Pull/Mean Look), Taunt/Disable,
Hyper Beam recharge, Perish Song — on the PP + move-legality foundation now in place.

**TAUNT + DISABLE — modeled bit-for-bit + COMPLETED** (2026-07-01, `gen3_taunt_disable_v1`; a prior build
agent was killed mid-run, completion pass verified + finished it; regression 33→37, e2e regen). The
move-SELECTION-restriction layer on the PP/Choice-lock brick. **THE MOD-CHAIN LESSON (the cautionary tale,
now root-caused at the source level):** gen3 conditions `inherit: true` from the **gen4 mod**, which
REPLACES both base onStarts — base taunt's `duration++` is GONE (gen4 = bare `-start` → gen3 duration
FIXED 2, no draw, ALL branches, probe-confirmed), base disable's `willMove → duration--` is REPLACED by
gen4's `!willMove → duration++` (→ the SETTLED `stored = faster ? rolled : rolled+1`, random(2,6)). The
3-lens review that read only base data/moves.ts mis-predicted by a constant +1 — read the WHOLE mod chain,
let the probe decide. Taunt residual = order 10/subOrder 15 (gen4's, base's 15 SHADOWED — probed via a
fast-taunt `-end` before a slow brn `-damage`); disable residual = NO_ORDER/subOrder-2 Condition default
(gen3 deletes gen4's 10/13). onBeforeMove cants (queued moves): taunt priority 0 (AFTER para — +1 draw when
paralyzed), disable 7 (BEFORE para — NO roll), both draw-free + no PP (`probe_taunt_disable_onbeforemove_
rng.js`, the probe the killed agent's comments cited but never wrote). endTurn `runEvent('DisableMove')`
size-2 tie-shuffle for a taunt+disable / lock+disable mon. **Golden gate perturb-PROVEN** (rolled+1/+2 and
rolled±1 each FAIL at the exact free-up-boundary scenario, restore passes; 720 runs, 4723 seed asserts).
Pins TD1-TD4 all revert-verified (TD4 = the cant-vs-para draw ORDER — the 720-run golden does NOT cover a
paralyzed queued move, found by perturbation, closed with in-engine-para pin). e2e regen at committed knobs:
STRICT 220/220 diverged 0, 13561 decisions, **317 USE TAUNT** (real teams!) / **0 USE DISABLE** (no sample
team carries it — honest leech-style disclosure), taunt floor >=50 added; byte-reproducible (2 regens
identical). Protocol honesty: taunt/disable `-end` lines DEFERRED (no capture scenario). Docs (CLAUDE.md
section + module rows + regression map, EDGE_CASES ✅ BUILT, README) synced. DEFERRED fail-loud: Torment /
Imprison / Encore. NEXT: trapping (Arena Trap/Magnet Pull/Mean Look), Hyper Beam recharge, Perish Song.

**0-PP-lastMove Disable guard (TD5) — fixed same day** (2026-07-01, small workflow; reviews initially died
on the 5h session limit, resumed via resumeFromRunId with the build journal-cached). The sim draws acc +
`random(2,6)` then the gen4-inherited onStart 0-PP guard REJECTS the volatile (`[still]` retro-edit via
`attrLastMove` + `|-fail|`, volatiles EMPTY); the port recorded a phantom volatile. Fix = the 0-PP check
after the draws + a new `ProtocolBuilder::attr_last_move_still()` (the ONE documented append-only
exception, ports `Battle.attrLastMove('[still]')`). Pin TD5 revert-verified; the phantom ALSO desyncs the
seed (its duration handler ties the live stall handler → extra residual tie-shuffle) so both assert
families have teeth. **Review caught the build agent's dead-guard handoff** (`if false &&` left from its
last revert-verify — the delivered tree contradicted its own report; one-token fix applied in review; see
[[feedback_determinism_means_fixable]] second corollary). Suite now **168 tests / 38 pins green**; all
goldens byte-identical (fix unreachable by every current gate). EDGE_CASES gains the latent
`blocked_by_taunt` bp-0 `basePowerCallback` note (Return/Flail/etc. are Physical in the sim but bp-0→Status
in the port's derivation; all six unmodeled + fail-loud; fix when modeled).

**TRAPPING — Arena Trap + Magnet Pull + the switch-legality gate, BUILT** (2026-07-01/02,
`gen3_trapping_v1`; build capped out mid-review → reviews resumed next window via resumeFromRunId;
verdicts ship/ship/fix-then-ship, minors doc-only, closed inline). The user's marquee edge cases all
probe-settled + PINNED: **Dugtrio mirror** = mutual trap, ZERO extra draws (arenatrap is onFoe*);
**Magneton mirror** = mutual trap AND **4 PRNG draws/endTurn** (gen3 overrides magnetpull to onAny* →
both actives' handlers on TrapPokemon+MaybeTrapPokemon → speed-tied Fisher-Yates each; AT-vs-MP cross
+2); **rejected-switch** = draw-free, seed byte-identical, boundary stays open (reject-and-re-request,
mirrors move_decision_is_legal); **phaze BYPASSES** trapping (Roar drags a trapped mon); forced
replacements accepted; the trapper switches freely. PROBE SURPRISE (T5): a grounded GHOST **IS trapped**
in Showdown-gen3 (the gen3 dex resolves NO `trapped` type-immunity — the base-typechart source read
predicted Ghost escape, the probe refuted it; EDGE_CASES' old "Ghost immune" claim corrected). Engine:
`is_trapped` (grounded rule / Steel rule), the Switch arm of `move_decision_is_legal`,
`trap_event_shuffles` interleaved per-mon in the endTurn DisableMove→Trap→MaybeTrap order. **Bonus
e2e-surfaced engine fix (I1): gen3 Intimidate SKIPS a Substitute'd foe** — the e2e's first regen
legitimately diverged 2 battles (e2e_171/204, port Atk −1 vs sim 0), root-caused + fixed, never
excluded. Golden: 640 runs / 5771 seed + 8346 trapped asserts, byte-reproducible (all 3 reviewers
regenerated both goldens md5-identical). e2e: filter-clean **18→22 teams**, arenatrap/magnetpull GONE
from the gap list (new head torrent≈naturalcure 254), STRICT 220/220 diverged 0, 11673 decisions, 142
trapped (floor ≥50). **175 tests / 44 revert-verified pins green** (T1-T5 + I1). Deferred fail-loud:
Mean Look/Spider Web/Block (trap MOVES), Shadow Tag; note — **Baton Pass escapes trapping in the sim**
(probe-verified; when BP is built its selfSwitch must NOT consult is_trapped). Ship-time note: the 4
tracked data/tools files (gen3_moves pp/noPPBoosts/secondaryBoosts, gen3_species maxHP, extractor) must
land in the same /gen3ai-ship commit as src/rust_sim — the Rust dex reads them.

**A/B DIFFERENTIAL FUZZER — BUILT + FIRST HUNT DONE (the user's "dreamworld")** (2026-07-03, 2-lens
ship). `harness/ab_fuzz.js` (continuous driver) + `src/bin/ab_replay.rs` (non-panicking chunk replayer,
catch_unwind, kind taxonomy seed>state>status>…; replays a repro dir directly) + runbook in README.
3 modes: **randbats** (default — `Teams.generate('gen3randombattle',{seed})`, Showdown's OWN generator
per the user's pointer; set-adapter: unmodeled item→Leftovers / ability→modeled-alt else reject, adjust
rate DISCLOSED ~21%), **random** (modeled-universe sampler: 195 species/152 moves — every species
exercised), **pool** (the 22 filter-clean teams). Divergence → self-contained repro dir
(battle.txt + summary.json + replay_cmd). Fault-injection ×3 PROVEN (dropped draw→seed-kind / +1
dmg→state-kind / winner flip→winner-kind; reviewers redid independently). ~2.5-3.3k battles/hour.
Engine untouched (175 tests byte-identical; e2e golden regen byte-identical after the shared-exports
refactor of gen_e2e_fuzz.js). **FIRST-HUNT FIX QUEUE (repros in harness/ab_fuzz_out/smoke_*/divergences/,
logged EDGE_CASES.md):** (1) Facade — dist onBasePower ×2-when-statused unmodeled AND isModeledMove
doesn't reject onBasePower; (2) Pink Bow/Polkadot Bow + odd/rock/rose/wave incense — in MODELED_ITEMS but
NOT priced in resolve_atk_stat_mods (item-set/port mismatch); (3) accuracy/evasion STAGES not folded into
the to-hit roll (boosts[5] tracked but run_move rolls raw acc — Mud-Slap chains flip hit/miss);
(4) Substitute-turn seed cluster (varied-level randbats); (5) ice-freeze cluster (port freezes where sim
doesn't, seed matching); (6) switch-boundary seed cluster; (7) 1 panic = drag divergence → unmodeled Wish
slot (fail-loud worked). Smoke: pool 100/100 CLEAN; randbats 267ok/27div/1panic per 300; random
49ok/151div per 200 (dominated by 1-3). Reviewer minors: chunk-error battle-accounting drift; panic masks
the upstream kind (replay incrementally later); 1-in-200 verdict flake observed once (re-replay-on-save
hardening suggested). **OVERNIGHT RUN LAUNCHED**: pid 3527108, `--mode randbats --hours 12 --master-seed
20260703 --out harness/ab_fuzz_out/overnight_0703` (check ab_fuzz.log + divergences/ there). NEXT: triage
the queue — fix 1-3 first (cheap, kills most noise), then the substitute/freeze/switch clusters, then
rerun random mode.

**DATA-DRIVEN MECHANICS FRAMEWORK P1 (`gen3_item_mechanics_v1`) — DONE** (2026-07-03, user's strategic
shift: knock out CLASSES data-driven "like the Pokédex", not id-by-id; 2-lens ship after 1 minor closed).
`harness/dump_gen3_mechanics.js` extracts the RESOLVED `Dex.mod('gen3')` → the CLASS MAP
`tests/vectors/gen3_mechanics_inventory.md`: **132 items + 76 abilities, EVERY entry classified** with
params + modeled-status + DRAW tag; `--check` = drift gate vs the dist. Data: additive obs-neutral fields
in gen3_items.json (`typeBoost{type,mod,fold}`, `statMods`, `onlySpecies`, `choice`, `isBerry`) +
gen3_abilities.json (`dmgMod` params for the 10 DMG_MOD abilities — DATA-ONLY, wiring is next phase).
FIRST CLASS WIRED end-to-end: the stat/BP-modifier item class (24 TYPE_BOOST + 6 SPECIES_STAT + Choice
Band) — `resolve_atk/def_stat_mods`/`resolve_bp_mods` are now pure dex lookups, hardcoded match-arm GONE.
PROBE SURPRISES: incenses are **×1.2 (chainModify 4915/4096), NOT the ×1.1** the old comment assumed;
bows are a DIRECT float `bp*1.1` that SKIPS the chain (the runEvent integer guard — mirrored with a
scope note: safe via gen3 type-disjointness, revisit at DMG_MOD wiring); **gen3 Light Ball = SpA-ONLY ×2**
(mod-chain rewrite); DeepSeaScale = DEFENDER-side SpD ×2. Class-sweep golden `item_mods_test` (990
battles, 2664 seed asserts, every member + controls, byte-reproducible) + damage golden 31→48 scenarios +
IM1-IM6 revert-verified pins. e2e BYTE-UNCHANGED 220/220. **Fuzzer-measured parity gain: 31 of 87
item-carrying repros now replay clean.** 183 tests green; Python facade 75+ green. Review minor closed
inline: the draw-tag detector now also matches `this.sample`/`randomFoe` — **Starf Berry + Trace are
DRAW-tagged** (were mis-tagged free; landmine defused for the berry/Trace phases). ROADMAP (in CLAUDE.md
"## Data-driven mechanics"): 1 ACCURACY pipeline (acc/eva stages + Bright Powder/Lax Incense + Hustle +
Compound Eyes/Sand Veil — kills fuzzer cluster 3), 2 ability DMG_MOD (data ready — biggest e2e admission
lever: torrent 254/naturalcure 254/blaze 103/guts 50/marvelscale 35), 3 berries (one eatItem mechanism),
4 PROC_ITEM (King's Rock/Focus Band, draw-bearing), 5 CRIT/DRAIN/misc items, 6 Natural Cure/CONTACT_PROC/
Trace. Overnight fuzzer: 12,300 battles / 1,072 divergences / 17 panics captured so far (pid 3527108).

**AUTONOMOUS CRON (2026-07-03)**: trigger `trig_01G5kfGjVe28XG3QbXXmstvR` ("pokesim autonomous
mechanics-framework window"), cron `0 */5 * * *` (every 5h, fires into THIS session
session_018dnhJkahedTJ2UnaNoTbGC). Each firing: gate (--five-threshold 80 --weekly-threshold 70; skip if
STOP_WEEKLY or 5h>~50%), relaunch the ab_fuzz overnight hunt if dead, then launch ONE workflow = the next
data-driven-mechanics phase. PHASE ORDER (the roadmap, also in CLAUDE.md "## Data-driven mechanics"):
(A) ability DMG_MOD wiring [Torrent/Blaze/Overgrow/Swarm pinch ×1.5 BP + Huge/Pure Power Atk×2 + Guts
Atk×1.5-when-statused-ignore-burn + Marvel Scale Def×1.5-when-statused; dmgMod data ALREADY in
gen3_abilities.json; Hustle deferred to B] → (B) accuracy pipeline [acc/eva stages into the to-hit roll —
fuzzer's biggest cluster — + Bright Powder/Lax Incense/Hustle/Compound Eyes/Sand Veil] → (C) berries [one
eatItem mechanism; Starf onEat this.sample DRAWS] → (D) PROC_ITEM King's Rock/Focus Band [draw-bearing] →
(E) remaining item classes [CRIT/DRAIN/BOOST_RESTORE/CURE/SPEED/TAKE_ITEM_GUARD] → (F) STATUS_IMMUNE /
Natural Cure / CONTACT_PROC [Static/PoisonPoint/EffectSpore draw-bearing] / Trace [onStart randomFoe DRAWS]
→ then the A/B-fuzz fix queue (#43). To STOP the autonomy: mcp delete_trigger trig_01G5kfGjVe28XG3QbXXmstvR.
Weekly ceiling 70%, 5h ceiling 80%, one workflow/window.

**MECHANICS FRAMEWORK PHASE 2 — ability DMG_MOD class WIRED** (2026-07-04, autonomous cron window,
2-lens ship). Data-driven from P1's `dmgMod` fields (new `src/dex/abilities.rs`; `dex.ability()` now
returns `AbilityData`); engine consumes them in resolve_atk/def_stat_mods + resolve_bp_mods. Probe-settled
gen3 math: pinch **Torrent/Blaze/Overgrow/Swarm** = typed BP ×1.5 when **`3*hp<=maxhp`** (bit-exact
integer form of hp≤maxhp/3, boundary-probed); **Huge/Pure Power** = Atk ×2 physical-only (storedStats
un-doubled, ModifyAtk chain); **Guts** = Atk ×1.5 when statused AND SUPPRESSES the burn-halve (reuses the
port's `has_guts` burn-skip + the new ×1.5 fold; fires on confusion self-hit too); **Marvel Scale** =
DEFENDER-side Def ×1.5 when statused. Guts+CB stack accumulate-ONCE (×2.25). Hustle DEFERRED to the
accuracy phase (its ×0.8 acc side is draw-relevant); Thick Fat keeps its existing hook. Class-sweep golden
`ability_dmgmod_test` (330 battles, 1350 boosted rows, every member + controls, byte-reproducible) + damage
golden 48→63 scenarios + **AB1-AB5 revert-verified pins** (55 regression total). 190 tests green, Python
facade 88 green, drift-gate clean. **e2e ADMISSION HONESTLY DEFERRED**: admitting the 8 abilities grows the
filter-clean corpus **22→151 teams** (torrent/blaze/guts/marvelscale gaps VANISH) BUT one newly-admitted
battle (e2e_86) diverges 219/220 on a PRE-EXISTING **freeze/switch** gap (EDGE_CASES §5 = the fuzzer's
ice-freeze cluster, #43), NOT a DMG_MOD bug (proven in isolation) — so kept e2e at the committed 22-team
220/220 byte-unchanged, admission gated behind a commented MODELED_ABILITIES block. **⇒ The freeze/switch
fix (#43 ice cluster) is now HIGH VALUE: it unblocks the 22→151 DMG_MOD e2e admission AND is the fuzzer's
biggest cluster — consider doing it BEFORE or WITH the accuracy phase.** Roadmap now: (B) accuracy pipeline
[+ Hustle] / freeze-fix (both high-value) → (C) berries → (D) proc items → (E) remaining items → (F)
status-immune/Natural Cure/contact/Trace. Overnight fuzzer at chunk ~1027 (untouched).

**SUN-FREEZE IMMUNITY + WISP MOVE-ALIAS FIXED → DMG_MOD ADMITTED (22→151 e2e teams)** (2026-07-04,
2-lens ship — the biggest coverage jump). TWO bugs, both probe-settled + revert-pinned:
(1) **Sun-freeze immunity** (`gen3_sun_freeze_immunity_v1`, pin FZ1): gen3 base `sunnyday.onImmunity('frz')`
BLOCKS a freeze while the field is Sun (Drought/Sunny Day) — checked in setStatus BEFORE the SetStatus
clause shuffle, so DRAW-FREE (the freeze secondary's random(100) still fires, seed matches, but the freeze
must not land; an already-frozen mon PERSISTS under sun). The port froze anyway = the A/B fuzzer's
"ice-freeze cluster" (196 repros). Fix = a draw-free gate in `try_set_status` after the type-immunity check.
**176/196 ice-freeze repros now replay clean** (254 of ALL 2513 overnight repros ok). Proven by FZ1 with
**0 e2e decisions** (no filter-clean battle has a sun+ice turn — leech-seed situation).
(2) **Move-ID alias** (`gen3_move_alias_resolution_v1`, pin FZ2): the ACTUAL e2e-admission unblocker.
Showdown resolves move aliases at `dex.moves.get()` (aliases.ts: `wisp`→willowisp, `sd`→swordsdance,
`twave`→thunderwave, …) and RUNS the canonical move; the port's dex read only canonical ids, so e2e_86's
Gengar (packed as `wisp`) no-op'd → draw-count desync (35 vs 41 decisions). Fix = extractor emits
`gen3_move_aliases.json` (44 aliases; the 28 non-HP mirror Showdown byte-for-byte, the 16 typed-HP
deliberately resolve to the port's DISTINCT typed names per gen3_typed_hidden_power_ids_v1, NOT collapsed
237), dex `moves()` resolves through it. Obs-neutral (facade never loads it; extractor-parity pinned).
**RESULT: the 8 DMG_MOD abilities ADMITTED — e2e filter-clean corpus 22→151/719 (the biggest lever; DMG_MOD
gaps VANISH from the taxonomy, now naturalcure 254/immunity 97 on top), STRICT 220/220 diverged 0, 9963
decisions, byte-reproducible (sha 99572d89).** 193 tests / 57 pins green (55+FZ1+FZ2). Doc nit closed inline
(the e2e_86 attribution — it diverged on WISP not sun-freeze; the two fixes are independent). NEXT: (B)
accuracy pipeline (now the fuzzer's biggest remaining cluster) → (C) berries → (D) proc → (E) items → (F)
status-immune/NaturalCure/contact/Trace → remaining A/B clusters (facade onBasePower / substitute / switch /
wish-drag). Overnight fuzzer completed its 12h (30050 battles / 2546 div / 40 panic); relaunch check done.

**2026-07-03 OVERNIGHT DIVERGENCE HISTOGRAM (mechanism-level → evidence-based fix-priority reorder).** Mined
all 2586 overnight repros (randbats, 30050 battles) by `first_divergence.kind` + the move at the diverging
decision (active-slot-tracked). KIND: seed 1662 / state 685 / status 199 / panic 40. Dominant drivers by
MECHANISM (supersede the roadmap's berry order):
- **STATE (685) = FLASH FIRE ×1.5 fire-boost.** fireblast 274 + flamethrower 158 = 432 fire-move STATE
  divergences; **397/402 have a Flash Fire mon on a team** + direction consistent (sim deals MORE fire dmg →
  exp hp < got hp). The port models FF IMMUNITY but NOT the post-activation ×1.5 boost (the CLAUDE.md's
  documented deferred gap). Biggest single cluster, small well-defined damage fold, FF already in
  MODELED_ABILITIES → **highest-value NEXT phase after accuracy.**
- **SEED (1662) = substitute 274** then secondary damagers (icebeam/rockslide/sludgebomb/psychic/shadowball)
  = the substitute×secondary draw-interleaving cluster on varied teams (fix-queue #4; far bigger at scale
  than the 300-battle smoke's "7").
- **STATUS (199) = icebeam 81** = ice-freeze gating residue (FZ1 fixed SUN-freeze; this is a different
  freeze application-order / clause case, not sun).
- **facade 50 (STATE)** = the ×2-when-statused fix-queue #1.
ACCURACY co-occurrence ≈0 in randbats (randbats REJECTS sandveil/compoundeyes teams + adjusts
brightpowder→Leftovers, per the reject log) — so the RUNNING accuracy pipeline mainly clears `random`-mode
cluster-3 + the accMod item/ability class, NOT randbats. Correct/foundational, but low randbats yield.
**Post-accuracy priority REORDERED by measured impact: Flash Fire boost > substitute×secondary > ice-freeze
residue > facade > berries.** Measure each fix's parity gain by replaying the 2586 repros
(`target/release/ab_replay <repro-dir>` → count ok-flips).

**2026-07-04 ACCURACY PIPELINE DONE + INDEPENDENTLY VERIFIED (`gen3_accuracy_pipeline_v1`, Phase 3/B).**
Workflow whmqd4f45 (build + 2-lens adversarial review, tohit-math lens CONFIRMED bit-for-bit) landed; I
verified: dead-guard grep EMPTY, full Rust suite green (0 fail, exit 0), extractor-parity 9/9 (obs-neutral).
Probe-settled gen3 to-hit: `effAcc = move.acc × acc/eva STAGE TABLE [1,4/3,5/3,2,7/3,8/3,3] × accMod
handlers`, then ONE `random(100) < effAcc` (effAcc a RAW f64, NOT floored). THE INTEGER-GUARD (battle.ts:929):
a `chainModify` accMod member is SKIPPED once acc is a non-integer float — so a STAGE or a DIRECT multiply
having floated acc kills every chain member (95×0.9=85.5 BP-direct applies; 95×0.8 SandVeil-chain skipped when
acc float). accMod class (RESOLVED, not base .ts): brightpowder ×0.9 + laxincense ×0.95 (DIRECT multiplies,
stored as verbatim f64 — 196 last-bit diffs vs rationals), compoundeyes [13,10], sandveil [8,10]-in-sand (+ its
sandstorm-chip immunity, the only gen3 weather-chip onImmunity), hustle [3277,4096] gated on move.TYPE (not
category) + its dmgMod Atk ×1.5 as a separate pre-chain modify. New `src/dex/accmod.rs` (shared AccMod);
`turn.rs` effective_accuracy/roll_accuracy; extractor accMod fields (obs-neutral). 202 tests / AC1-AC5
revert-verified pins; e2e STILL 220/220 byte-UNCHANGED (BP/SandVeil/CompoundEyes/Hustle too rare in gen3 OU to
unlock new filter-clean teams; empty-path byte-identical). Fuzzer parity: ~70/132 smoke_random Mud-Slap
acc-stage repros now ok; randbats/overnight has 0 acc-stage repros. **NEXT phase is Flash Fire ×1.5 boost
(task #52, the histogram's #1 STATE driver ~432), NOT the roadmap's berries.** This firing: quota STOP_WEEKLY
(72% ≥ 70 autonomous cap) → launched NO new workflow; only closed out accuracy + relaunched the free fuzzer
(run rmr5wgnn2, randbats 12h, seed 1783141588).

**2026-07-07 FLASH FIRE ×1.5 BOOST DONE + VERIFIED (`gen3_flashfire_boost_v1`, task #52).** User raised the
weekly ceiling to 93% + asked to prioritize CLASS-level work ("all abilities or all moves"). Workflow
wf_a54a2c81-f95 (build + 2-lens adversarial review, both SHIP) landed the histogram's #1 STATE divergence
driver (fireblast+flamethrower dominated STATE; 397/402 fire-STATE repros had a FF mon). I verified: dead-guard
EMPTY, full suite 208 passed / 0 fail, e2e STRICT 220/220 byte-UNCHANGED. THE PROBE SURPRISE (overturned my
stat-mod hypothesis): the boost is **`onModifyDamagePhase1` chainModify(1.5)** — a DAMAGE-PHASE fold (same phase
as Reflect/Light Screen), NOT `onModifyAtk/SpA` (those are `undefined` in the resolved gen3 dist). It
ACCUMULATES with screens into ONE chainModify (sequential rounds diverge ~¼ of the time — proven). ACTIVATION:
`flashfire.onTryHit` arms AFTER the accuracy roll on a LANDED Fire hit (a MISS does NOT arm — gated on
`acc_hit` like Water/Volt Absorb), DRAW-FREE, cleared on switch-out+faint, a `frz` holder doesn't arm; WoW
never arms a real (Fire-type) holder. Engine-flag approach (a `flash_fire: bool` MonState volatile + the
damage.rs Phase1 fold) — NOT a dmgMod field (that framework is stat/BP folds), so gen3_abilities.json is
byte-unchanged (Python facade untouched). This COMPLETES the type-interaction ability class (Levitate +
Water/Volt Absorb immunities were done; FF boost was the last gap). 3 revert-verified pins (FF arms-on-hit/
not-miss, boosts-own-fire-move, clears-on-switch) + 2 calc pins. **Fuzzer parity: 185/200 (92.5%) FF-team
STATE repros flip to ok** (revert re-diverges → direct attribution). Closed the one substantive review nit
inline (an honest-scope comment: activation only at the damaging-move site → a non-Fire FF holder + WoW is
out-of-scope, impossible in gen3 OU). Weekly ~86%/93.

**2026-07-07 NATURAL CURE DONE + VERIFIED (task #56, engine-flag, obs-neutral).** Workflow wf_e2d2bcee-913
(build + 2-lens, both SHIP-clean). The #1 e2e team-carry gap. Verified: dead-guard EMPTY, 212 tests pass, e2e
byte-reproducible. PROBE resolved the deferred "CheckShow" question: `naturalcure.onSwitchOut` (**onCheckShow
UNDEFINED → no CheckShow gate**) → **DRAW-FREE / seed-neutral** (post-switch seed bit-identical NC vs non-NC,
all 6 statuses, voluntary AND phaze-drag). Cures on voluntary switch + phaze-drag-OUT (SwitchOut fires on
isDrag; only BeforeSwitchOut is !isDrag-gated), no-op on faint (`status=='fnt'` guard), resets tox stage +
sleep counter. Engine flag in `execute_switch` (justified single-member class; `gen3_abilities.json` UNCHANGED
→ obs-neutral, Python facade untouched). **E2E: filter-clean pool 151 → 449/719 (the BIGGEST admission lever
yet), STRICT 220/220 diverged==0, 12054 decisions, ZERO engine bugs surfaced (clean first-try). Taxonomy now:
immunity=97 > shellarmor > synchronize; items lumberry=64.** 3 revert-verified NC pins. Lens-2 e2e-HONESTY
check (the key adversarial gate): the generator drives ONLY the sim → structurally CANNOT mask a divergence
(every drop = an honest forced-unmodeled coverage limit, not a hidden exclusion). Doc nit fixed inline (NC =
sole gen3-REACHABLE switch-out-cure; Regenerator/Zero-to-Hero unreachable in gen≤3). **This session (user
raised weekly to 93% + "implement CLASSES"): Flash Fire ✅ + Natural Cure ✅ landed. NEXT: STATUS_IMMUNE
ability class (immunity=97, the #2 gap) — the e2e customgame path is likely DRAW-FREE (ability immunity is at
onTrySetStatus BEFORE the SetStatus event; the gen3ou 3rd-handler shuffle-size subtlety only bites the
dedicated gen3ou golden, not the customgame e2e). Weekly ~88%/93; ~2%/workflow so ~2 more fit.**

**2026-07-07 STATUS_IMMUNE ability class DONE + VERIFIED (`gen3_status_immune_v1`, task #57).** Workflow
wf_1bcf966c-765 (build + 2-lens, both SHIP). The #2 e2e gap (immunity=97). Verified: dead-guard EMPTY, 224
tests pass, e2e byte-reproducible (2 independent regens, md5 stable). MEMBERS (6, resolved dist num≤76):
Limber(par)/Insomnia(slp)/VitalSpirit(slp)/Immunity(psn+tox)/WaterVeil(brn) via **onSetStatus** + MagmaArmor(frz)
via **onImmunity**. OwnTempo/Oblivious block VOLATILES not major status → excluded; LeafGuard num102 → NOT gen3
(old hardcode was wrong, removed). **THE DRAW-MODEL SURPRISE (killed the feared "size-3 shuffle" — the crux):**
in gen3ou an onSetStatus ability adds a 3rd SetStatus handler, BUT speedSort (order→priority→SPEED→subOrder)
puts the ability handler [DEFINED speed = the mon's] in its OWN group, leaving the 2 clause handlers
[speed=undefined] a size-2 tie → shuffle(1,3) = EXACTLY 1 draw, byte-identical to the control's shuffle(0,2) →
draw count UNCHANGED (reproduced even at extreme speed). In customgame (the e2e format) the ability is the
event's only handler → size-1 → 0 draws → DRAW-FREE. Data-driven: `statusImmune{statuses,phase}` field
(dump_gen3_mechanics --check DERIVES it from the resolved onSetStatus/onImmunity handlers; obs-neutral, facade
AbilityData = id/num/name only). **E2E: filter-clean pool 449→525/719 (+76), STRICT 220/220 diverged==0, 11651
decisions, byte-reproducible; `immunity` DROPPED OFF the taxonomy (now shellarmor=39 leads > synchronize >
effectspore > trace).** 4 revert-verified SI pins. **REAL BUG the enlarged corpus surfaced + FIXED (NOT the
class): the EMPTY NATURE panic** — a Suicune packed with an OMITTED nature field (`||Item|Ability|moves||EVs`)
panicked `compute_stats` "unknown nature"; the sim treats empty nature as NEUTRAL/Serious (probe-verified) →
fixed to all-1.0 multipliers (a NON-empty unknown nature STILL hard-errors), 2 revert-verified stats.rs pins,
replayed by e2e_8/e2e_73. Honest residual: the STATUS_IMMUNE block doesn't emit the `-immune` protocol line
(deferred nicety; STATE+SEED verified).

**SESSION TOTAL (user raised weekly to 93% + "implement CLASSES, all abilities/moves"): 3 CLASSES landed +
verified — Flash Fire ✅ (task #52) + Natural Cure ✅ (#56) + STATUS_IMMUNE ✅ (#57). Filter-clean e2e pool
151 → 525/719 (a 3.5× growth), STRICT 220/220 diverged==0 throughout, dead-guard clean, 224 tests. Weekly 84 →
91% (stopped at 91 — a 4th would tip the 93 ceiling). NEXT classes (fresh weekly window Mon 21:00 UTC): the
taxonomy's remaining top gaps — shellarmor=39 CRIT_IMMUNE (Battle Armor/Shell Armor, crit-roll immunity) /
synchronize (status-REFLECT, draw-bearing — probe carefully) / berries (eatItem, Starf sample DRAWS) /
effectspore CONTACT_PROC (draw-bearing) / trace (onStart randomFoe DRAWS). Plus the A/B fix queue #43
(substitute×secondary ~274 SEED, ice-freeze residue).**

**2026-07-08 ABILITY BATCH-2 DONE + VERIFIED (draw-bearing reactive abilities, `gen3_ability_batch2_v1`, tasks
#76-86).** Workflow wf_97874bc9-cad (build + 2-lens). Verified: 243 tests, dead-guard EMPTY, e2e STRICT 220/220.
WIRED: **CONTACT_PROC** (static/poisonpoint/flamebody = randomChance(1,3) → status the ATTACKER; effectspore =
randomChance(1,10) then sample(3); roughskin = draw-free maxhp/16 recoil) — the proc draws INSIDE
runEvent('DamagingHit') AFTER the move's own secondary. **BLOCK** (suctioncups blocks the phaze drag [no sample],
soundproof blocks Sing/Grass Whistle/Roar [NOT a no-op], damp cancels the Explosion self-KO). **Synchronize** (was
the #1 gap — onAfterSetStatus reflect to the source, draw-free in customgame). DEFERRED honestly: Cute Charm
(attract volatile unmodeled), Color Change (type-override, 0 teams). **E2E: filter-clean 571 → 585/719; taxonomy
top gap now `trace`.** 7 build pins B2-1..B2-7.
**THE ADVERSARIAL REVIEW CAUGHT A REAL BUG THE BUILD AGENT MIS-REPORTED: CONTACT_PROC fired BEHIND A SUBSTITUTE.**
The build report claimed the proc fires "behind a Substitute"; LENS 1 probed the live sim + proved it does NOT
(onDamagingHit is on the MON, not the sub). The Rust gated on `is_contact && dealt > 0` (dealt>0 behind a
surviving sub) → a phantom randomChance + wrong status on the attacker. The e2e passed 220/220 only because no
e2e battle had a contact-proc-holder-behind-a-surviving-sub — a COVERAGE GAP, not a false pass. **I FIXED it
inline: gate on `!absorbed` (the SAME gate the fire-thaw uses — an onDamagingHit-on-the-mon can't fire when the
sub absorbed the hit), fixed 2 stale comments (call-site + golden harness), + added revert-verified pin B2-8
`contact_proc_does_not_fire_behind_a_surviving_substitute` (Chansey-Tackle into an Electabuzz-Static-sub; ground
truth seed 39376,31046,45923,49458; revert → seed 59445,… diverges). 243 tests, dead-guard EMPTY.** LESSON: the
adversarial review earns its keep — a build agent CAN mis-report a mechanic (assert a behavior the sim lacks) AND
the e2e can pass via a coverage gap; the independent probe is the arbiter (never trust a build report's mechanic
claim without a probe). **RESUMED 2026-07-08 (user: "Continue, use less than 60% of the 5h and less than 40% of the weekly"; model →
fable-5).** New caps 5h<60 / weekly<40 SUPERSEDE the old 98%. Old cron DELETED; NEW cron
trig_01Js7YdC4yW11c31Hk7wxVP3 (0 */5, gates 60/40, option-A batching + the behind-sub lesson + the
bake-quota-into-subagent-prompts lesson all in its prompt). **Batch 3 LAUNCHED (wf_23c4de13-e87, task #87):
BERRY item class (eatItem mechanism + CURE/HEAL/PINCH/PP rows — lumberry=64 the top item gap; cruxes: per-class
trigger site/threshold/timing, the residual order-10-sub-4 Leftovers-TIE shuffle, sub-absorbed vs pinch,
item→none, Starf sample) + Trace (#1 ability gap, n=1 randomFoe draw + unmodeled-copy e2e safety) + Shed Skin
(residual randomChance(33,100)). Batch 4 queue: Cute Charm (attract) / Color Change / King's Rock + Focus Band
PROC_ITEMs / remaining item classes / A/B fix queue #43.**

**2026-07-08 BATCH-3 BUILD = SAFE-PARTIAL (`gen3_berry_trace_shedskin_v1`) — FINISH IT BEFORE BATCH 4 (task
#94).** Build stopped honestly at 57% of the 60% 5h cap, e2e allow-lists UNTOUCHED (safe not-admitted state);
I verified 251/0 green + dead-guard EMPTY. WIRED+PINNED (BR1-BR5, revert-verified): the BERRY class (eatItem
draw-free+permanent; HEAL residual order-10-sub-4 = the EXACT Leftovers sort key, tie-identical to a Leftovers
mirror; threshold 2*hp<=maxhp; oran+10/sitrus+30/**Figy family floor(maxhp/8) — the RESOLVED dist, NOT the base
.ts /3 [mod-chain law strikes again]** + nature-gated confusion random(2,6); PINCH 4*hp<=maxhp +1 draw-free;
**Starf = ONE sample over non-capped stats, draws even at n=1, all-capped=no draw, +2** — modeled not deferred;
Lansat draw-free focusenergy; CURES at the first eachEvent('Update') BEFORE the holder's move; **LUM eats
IMMEDIATELY inside setStatus, AFTER a Synchronize reflect; LumRest cures BEFORE the heal → full-HP awake**;
sub-absorbed hit ⇒ NO trigger; KO ⇒ no eat; Leppa Update-site +10 on the first 0-PP slot) + TRACE (**n=1
randomFoe DRAWS** at a mid-battle switch-in; copies the CURRENT ability; **the copied onStart does NOT fire —
setAbility's gen>3 gate** — passive effects LIVE [traced Flash Fire absorbs+arms]; switch-out reverts; lead
trace = a >start-window draw applied draw-free; TRACE_COPYABLE fail-loud + the both-teams-modeled e2e safety)
+ SHED SKIN (residual order-10-**sub-3**, handler gathered UNCONDITIONALLY [tie-relevant even unstatused], ONE
randomChance(33,100) per statused residual, cure BEFORE the same-mon DoT). Golden: 32 scenarios × 40 = 1280
battles first-try. **REMAINING (task #94, in order): (1) the reviewer's OPEN FINDING — the <=-vs-< threshold
boundary is UNPINNED (mutating <=→< passes everything; need an even-maxhp probe at hp==maxhp/2 + /4 and a
boundary pin); (2) LENS 1 (draw-model review) NEVER RAN (orphaned quota-sleep — re-run it); (3) the e2e
admission+regen (was 585/719; backup the golden first; divergence→root-cause or restore).** LESSON: a
workflow's parallel review lens can silently NOT-run (quota-parked) while the workflow completes — always check
each lens actually executed. 5h hit 72% → PAUSE_5H; the 60/40 cron resumes after the ~04:30 reset.

**2026-07-08 BATCH 3 CLOSED — SHIP (wf_39a67029-922, tasks #93/#94 done; both lenses EXECUTED + passed).**
I verified 252/0 green + dead-guard EMPTY. (1) BOUNDARY VERDICT: the sim EATS at exact equality (hp==maxhp/2 +
maxhp/4; even-maxhp-400 Vaporeon oracle, both builder + Lens-1 independent probes; resolved source is `<=` with
float division) — **the engine's `<=` was already right, no latent bug**; pinned BR6
(`berry_thresholds_eat_at_exact_equality`), revert-verified by builder AND both reviewers, and Lens 1 proved
BR6 is the ONLY pin with teeth (the `<`-mutation passes the whole 1280-battle golden + the 220 e2e). (2) E2E
ADMISSION: 22 berries → MODELED_ITEMS + trace/shedskin → MODELED_ABILITIES (TRACE_COPYABLE lockstep 71==71
verified programmatically); **filter-clean pool 585 → 712/719 (99%!)**, STRICT 220/220 diverged==0 FIRST-TRY,
11054 decisions, byte-reproducible ×2 (md5 74be3fd2…), **Leech Seed finally exercised (354 decisions, was 0 —
the old honest disclosure CLOSED)**. Taxonomy: **item gap list EMPTY; ability gaps truant=4 + innerfocus=2
only; ENGINE_GAP none**. 252 tests. Non-blocking notes: the Synchronize→Lum protocol line ordering is
unverified territory (same class as the un-emitted Trace |-ability| line — level-2 protocol niceties).
**BATCH 4 queue (small tail!): Truant (every-other-turn cant) + Inner Focus + Shadow Tag + Cute Charm (attract
volatile) + Color Change + Forecast (forme) + King's Rock/Focus Band PROC_ITEMs; then the A/B fix queue #43
(substitute×secondary SEED cluster — now the fuzzer's dominant remaining class, ice-freeze residue, facade).
Fuzzer divergence rate history: 8.5% → 2.9% (the class work); expect another drop once the batch-3 engine
mechanics ride the randbats adapter.**

**2026-07-08 BATCH 4 ATTEMPT 1 = PRE-WIRING PARTIAL (wf_a00b082b-250, task #95) — RESUME VIA A FRESH
COMPLETION WORKFLOW, NOT resumeFromRunId.** All 3 agents (build + BOTH lenses) quota-paused at 68-72% of the
5h cap and the workflow returned with their PAUSE texts as results — the "background sleeps" are ORPHANED
(the batch-3 silent-park pattern ×3; resumeFromRunId would CACHE the useless pause outputs — don't).
Synthesis correctly said NO-GO/incomplete + honest. VERIFIED CLEAN BOUNDARY (me + Lens-2 statics): 252/0
green, dead-guard EMPTY, ZERO engine edits, ZERO allow-list admissions, no debris; **9 batch-4 probe files
WRITTEN** (probe_truant{,_edges}_rng / probe_innerfocus_{flinch,rng} / probe_shadowtag_rng /
probe_cutecharm_attract_rng / probe_colorchange_rng / probe_forecast_rng / probe_kingsrock_rng /
probe_focusband_rng / probe_batch4_lib.js) — the builder claims all 8 draw models settled but did NOT report
them (unverified). **NEXT WINDOW: a batch-4 COMPLETION workflow — build starts by RUNNING the written probes
+ REPORTING each member's draw model (Lens 1 must adversarially re-probe them), then wire + pin + admit per
the original batch-4 spec (Truant/Inner Focus [the Shield-Dust draw-count DIFFERENCE]/Shadow Tag/Cute
Charm+attract/Color Change+Forecast type-override/King's Rock [roll POSITION vs secondary vs contact-proc]/
Focus Band [per-hit-vs-per-lethal + deferred-faint]); e2e from 712/719; both lenses must DEMONSTRABLY
execute.** After batch 4: the A/B fix queue #43 (substitute×secondary).

**2026-07-08 BATCH 4 COMPLETE — THE MECHANICS TAIL IS DONE; e2e pool 712 → 719/719 = THE ENTIRE REAL TEAM POOL
(wf_7d29ce32-87e, tasks #95-103).** 7 of 8 wired + admitted: TRUANT (onBeforeMovePriority 9 cant, loaf turn
DRAW-FREE incl. no-para-roll; truantTurn arms on switch-in `turn!=0`, order-27 residual toggle — a
post-residual-DoT-KO replacement LOAFS its first turn, a mid-action replacement MOVES), INNER FOCUS (block at
the flinch APPLY — the secondary random(100) STILL draws, bit-identical to a Thick-Fat control; vs Shield
Dust's filter-the-draw), SHADOW TAG (universal trap incl. Flying/Levitate, ZERO endTurn draws like Arena Trap),
CUTE CHARM + ATTRACT (the randomChance(1,3)@DamagingHit draws UNCONDITIONALLY, gender gate draw-free inside
attract.onStart; attract onBeforeMove priority 2 = after confusion 3 / before par 1, '-activate' ALWAYS then
randomChance(1,2); source-leave AND holder-leave clear; Oblivious blocks draw-free; UNKNOWN-gender ratio
species = a construction-time sim `sample(['M','F'])` → the port fail-louds), COLOR CHANGE (types_override via
ONE `mon_types` choke point replacing 11 call sites — STAB/chart/status-immunity/sand/MagnetPull/LeechSeed/
weather/spikes/Combatant/is_trapped; does NOT fire behind a SUB [the batch-2 lesson held]; not on
Struggle/status/KO; switch-out reverts), KING'S ROCK (ONE extra trailing random(100) on 130 listed move ids;
ORDER = [own secondary]→[KR]→[contact proc]; Serene Grace threshold 20; Shield Dust suppresses; drawn-not-
applied behind a sub; Seismic Toss procs), FOCUS BAND (randomChance(1,10)@onDamage on EVERY damage event into
the holder — non-lethal, lethal, chips [draws, still faints — not a Move], Spikes, recoil, the CONFUSION
SELF-HIT [effectType Move → survivable]; NOT behind a sub; Explosion exploder draws nothing; survive=1 HP on
lethal MOVE damage only). FORECAST honestly DEFERRED (0 teams; forme-reporting surface un-designed;
Cloud-Nine composition unprobed; fail-loud excluded). **E2E: STRICT 220/220 diverged==0, 10636 decisions,
byte-reproducible ×2 (md5 a23d77ac…); taxonomy 300/300 unfiltered clean, EMPTY ability+item gap lists,
ENGINE_GAP none — a FIRST. 263 tests, 7 revert-verified B4 pins, dead-guard EMPTY (I verified 263/0 + EMPTY
independently).** Both lenses DEMONSTRABLY ran (fresh review probes on disk; Lens 1 re-derived KR order via a
new Meteor-Mash→Static triple-roll construction + the 130==130 move-list set-compare). Synthesis =
SAFE-PARTIAL pending ONLY a final taxonomy/md5 regen confirmation (the Lens-2 background check was orphaned;
I re-ran it locally): **BYTE-IDENTICAL ✓ (md5 a23d77ac… independent regen == committed == builder ×2) →
BATCH 4 = SHIP. The data-driven mechanics framework (phases/batches 1-4) is CLOSED: every gen3-reachable
ability + item in the real team pool is modeled or provably-no-op; 719/719 filter-clean.** **REMAINING QUEUE: the A/B fix queue #43
(substitute×secondary SEED cluster — the fuzzer's dominant residue), the ice-freeze residue, protocol level-2
niceties (Trace |-ability|, Synchronize→Lum ordering, -immune on status-immunity), Forecast, and the
long-horizon drop-in endgame (BattleStream::write_line).**

**2026-07-08 A/B FIX-QUEUE #1 SHIPPED (`gen3_shielddust_sub_v1`, wf_a9576364-6f5, tasks #104-105; both lenses
PASS w/ independent probes).** THE "substitute×secondary" CLUSTER WAS REALLY **SHIELD DUST BEHIND A
SUBSTITUTE**: SD's secondary filter is a TARGET-gathered ModifySecondaries handler; a sub-absorbed hit's
target is NULL → the filter never gathers → the secondary random(100) STILL DRAWS in the sim (holds for held
AND breaking subs, the Tri-Attack gate, and King's Rock's appended roll; a PASSING roll still applies nothing
— effect-suppression stands). The port filtered an SD defender UNCONDITIONALLY at 3 sites (apply_secondaries /
apply_triattack_secondary / apply_kings_rock_secondary) → one MISSING draw per secondary-into-an-SD-sub;
randbats is saturated w/ Venomoth (SD+sub+sludgebomb) → the dominance. FIX: `!absorbed_by_sub` on the SD
filter at the 3 sites. Pin `shield_dust_behind_a_substitute_still_draws_the_secondary` (revert-verified by
builder + Lens 1, which built its OWN different construction + closed the ground-truth circularity by
re-running the regression probe live). **PARITY: 362/777 corpus repros flip ok (95% of the SD non-facade
slice); 2 of 4 panics clear.** I verified: 264/0, dead-guard EMPTY, e2e md5 UNCHANGED (a23d77ac… — no
overreach; all 10 batch-2 sub baseline pins hold). TRIAGE HISTOGRAM (auto_0708_0304, 771): SD-sub 378 [FIXED]
> **facade 265 [now the #1 open — model onBasePower ×2-when-statused]** > other-secondary tail 61 [recurring
SACREDFIRE fingerprint — UNPROBED hypothesis: gen3 `flags.defrost` may bypass the frozen user's
randomChance(1,5) thaw roll] > switch-boundary 50 > 2 panics (wish-drag). **Weekly 38/40 → STOPPED at the
cap; next window (Mon 21:00 UTC reset or a raised cap): facade, then the sacredfire-defrost probe.**

**2026-07-09 (STOP_WEEKLY firing, free maintenance): THE 469-PANIC SPIKE TRIAGED + ADAPTER-FIXED.** The first
post-SD-fix fuzzer run showed panic=469 (was 4-40) — ALL are the batch-4 Cute Charm attract fail-loud on an
UNSPECIFIED gender (the sim `battle.sample(['M','F'])`s one at construction; the port can't re-derive it;
goldens pin genders but randbats sets don't) — reachable since the batch-4 admission let the adapter keep
`cutecharm`. NOT an engine bug — corpus noise. FIX (harness-side, the Hardy-nature precedent): the randbats
adapter now pins `set.gender='M'` when the dex species gender is '' (true ratio) — a serialization
normalization, both engines read the same packed team ('M' valid for every ratio species; fixed-M/F/N
untouched). ab_fuzz.js edited + syntax-checked; the LIVE run (JS already in-process) is undisturbed — the fix
rides the next relaunch. **THE REAL SIGNAL of that run: seed=124 + state=53 + status=1 = 178/15,248 ≈ 1.2%
divergence (was 2.3-2.9% pre-SD-fix) — the Shield-Dust fix confirmed live.** Weekly 46% → stopped.

**2026-07-09 A/B FIX-QUEUE #2 SHIPPED (`gen3_facade_v1` + `gen3_defrost_v1`, wf_a1035079-bd8; both lenses PASS
w/ fresh independent probes; I verified 266/0 + dead-guard EMPTY + e2e md5 UNCHANGED a23d77ac…).**
(1) **FACADE** (the #1 cluster): ×2 BP-chain member for psn/tox/par/brn (NOT slp; frz unreachable — no defrost
flag); the gen3 burn ATTACK-halve STILL applies (Facade doesn't ignore burn); Guts composes ×1.5 Atk +
halve-suppressed + ×2 BP ≈ ×3. **BONUS BUG the probe overturned: the port's "a DIRECT float multiply discards
the chain" shortcut was WRONG in the exact-integer case** — Pink Bow 70×1.1 == 77 EXACTLY in f64 → battle.js:709's
integer-guard passes → the accumulated chain RE-APPLIES (BP 154 not 118); fixed with the exact guard
`f == f.floor()` in damage.rs. Pins FA-a..e revert-verified ×2 (chain member + the integer-guard separately).
**Ok-flips: 333/344 + 143/145 facade repros.** The "admission question" was a STALE premise — isModeledMove
never rejected onBasePower (facade/sacredfire already admissible; no predicate change, no regen).
(2) **SACREDFIRE-DEFROST**: hypothesis REFINED by the probe — the frozen thaw roll IS drawn first; on a FAILED
roll a `flags.defrost` move (Sacred Fire/Flame Wheel, the only 2 gen3 carriers) PROCEEDS ANYWAY + thaws
draw-free (`|-curestatus|…[from] move:` BEFORE `|move|`); exactly +1 draw vs healthy. The port's
always-cant-on-failed-roll was the desync. Pin DF-a..c revert-verified. **Ok-flips: 49/58.**
(3) **RESIDUAL MAP (auto_0709_0805 re-replayed: 151 ok / 63 div / 2 panics)**: seed@move ~42 (mixed small —
icebeam ×7, no single fingerprint; ice-freeze-residue candidate within) > state ~12 (thunderbolt-vs-
Plusle/Minun ×5 — LIGHTNING ROD? it's in NOOP_ABILITIES as "redirect N/A singles" but the sim may still
no-op differently — worth a probe; willowisp ×2) > switch-boundary 5 > status 5 > 2 fail-loud panics
(sleeptalk + a bp-0-callback `return` move routing to the status guard — both unmodeled-move fail-louds
reached after upstream drag divergence). **Fuzzer trajectory: 8.5% → 2.9% → 1.3% → next run should land
~0.3-0.5% with facade+defrost in the replayer.** Weekly reset landed 07-09 (2%→6% used).

**2026-07-10 A/B FIX-QUEUE #3 SHIPPED — THE RESIDUAL TAIL IS CLEARED: 307/307 ok (wf_b1c2dc7d-5d9; both
lenses independently confirmed ALL SEVEN fixes; I verified 273/0 + dead-guard EMPTY + e2e md5 UNCHANGED).**
SEVEN engine bugs, each probe-settled + pinned + revert-verified: (1) **`gen3_plus_minus_v1`** — the batch-1
NO-OP classification was WRONG: gen3 onModifySpA scans getAllActive() INCL. FOES → Minus-vs-opposing-Plus =
SpA ×1.5 (special-only, draw-free; the batch-1 probe only tested partner-less). NOOP→MODELED reclass, union
unchanged → no regen. (2) **`gen3_ff_wisp_absorb_v1`** — WoW into a non-Fire un-statused un-subbed FF holder
(incl. a TRACED one) is ABSORBED (arms, no burn). (3) **`gen3_cloudnine_end_v1`** — Cloud Nine/Air Lock onEnd
fires eachEvent('WeatherChange') at BOTH End sites (switch-out pre-swap + faint pre-flag) → a tie-shuffle on
a cached-speed tie (localized via a sim draw-site stack trace). (4) **`gen3_ff_frozen_no_absorb_v1`** — a
FROZEN FF holder is NOT fire-immune (frz early-return in onTryHit) → full draws + thaw. (5+6)
**`gen3_fnt_clears_status_v1`** (the old switch-boundary cluster): checkFainted sets status='fnt' (para
ERASED) + clearVolatile ZEROES the corpse's boosts → replacement instaswitch sort reads PLAIN corpse speeds;
honestly RE-MEANED the old NC2 pin (it pinned port-internal state the sim never held). (7)
**`gen3_statusimmune_onupdate_cure_v1`** — the 6 STATUS_IMMUNE abilities carry onUpdate CURES (a slept mon
tracing Insomnia is cured at the first Update, draw-free). The 4 fail-loud panics' upstream divergences fell
inside these clusters. Ok-flip cascade 152→124→31→21→11→1→0. 273 tests / 7 new pins. NEXT: re-triage the
live auto_0709_2205 corpus with this binary (expect near-clean; survivors are genuinely NEW) + Lens-2's
parting note (one old-corpus repro w/ a seed-matching ~maxhp/16 HP delta). LESSON (recurring): a "provable
no-op" claim needs the CROSS-FIELD probe, not just the partner-less one — same family as the behind-sub
lesson.

**2026-07-10 FIX-QUEUE #4 SHIPPED — THE STEADY-STATE TAIL CLEARED 9/9 (wf_5dea3281-0d5, task #108; both
lenses executed + APPROVED; I verified 277/0 + dead-guard EMPTY + e2e md5 unchanged + closed all 3 findings
inline).** The first all-fixes 12h run: **9 div / 35,018 = 0.026%, 0 panics** (from 8.5% at fuzzer birth =
~330×). Disposition: 4 = mid-run-fix noise (battle-index split corroborated); 3 NEW bugs FIXED+PINNED:
(1) **`gen3_faint_queue_order_v1`** — faintMessages drains the faintQueue in ENQUEUE order, each corpse FULLY
processed (fainted=true) before the next corpse's ability-End → a mutual Explosion's self-KO'd user is
already inactive when the CN target's onEnd WeatherChange gathers (Lens-1 even ran the discriminating
CN-on-USER arm: 8 draws vs 7). Port walked side order when logging was off. (2)
**`gen3_fainted_no_ability_speed_v1`** — a corpse's ability handlers don't gather: fainted Swift-Swim Kingdra
getActionSpeed 368→184 (plain), tying a 184 corpse → the instaswitch shuffle. effective_speed gates on
!fainted. (3) **`gen3_tox_stage_persists_v1`** — the tox stage-0 reset fires via the RUNSWITCH-time
runEvent('SwitchIn'), NOT the raw swap → RESETS when the runSwitch RUNS but PERSISTS when gen3
faint-cancels-all CANCELS the queued runSwitch (the ab_1166 Mew KO). TX1+TX2 pin the PLACEMENT both ways.
The Lens-2 maxhp/16 lead WAS bug 3 (old corpus now 489/0). FINDINGS CLOSED INLINE: the stale contradictory
turn.rs comment (asserted the interim WRONG hypothesis — doc-poison) rewritten; probe_tox_stage_switch.js's
FALSE verdict fixed (it compared the non-discriminating 2nd residual; now keys on the re-entry stage).
**OPERATIONAL NOTES: (a) auto_0710_1305's pre-14:00 divergences (~192) are POISON (the builder's mid-edit
window raced the shared ab_replay binary the live fuzzer execs) — disregard them in the next triage (their
replays will be ok-noise); post-14:00 chunks are clean (0 div). (b) Future revert-verification: use an
isolated CARGO_TARGET_DIR, never the shared target/ while a fuzzer runs. (c) The dead-guard grep pattern now
includes MUTATION.** 277 tests / 4 new pins. **NOTHING OPEN: every known corpus replays clean. The port is at
true steady-state — the cron + fuzzer loop now just harvests genuinely-new exotics as they appear.**

**2026-07-11 HANDLER-COMPLETENESS AUDIT SHIPPED (wf_4d8dc981-ced, tasks #109-112; user-requested, 80% weekly
auth; both lenses SHIP; I verified 281/0 + dead-guard EMPTY + e2e md5 unchanged + applied the one finding
inline).** The dispatch-bus completeness guarantee as a STATIC gate: `harness/dump_gen3_handlers.js`
enumerates EVERY resolved handler key (FNV-1a body fingerprints → semantic-drift detection) across the full
reachable surface (74 abilities / 59 items / 27 conditions incl. clauses / 168 modeled moves);
`tests/vectors/gen3_handler_audit.json` = one row per (effect,hook) with disposition + grep-verified anchors;
curated in `harness/handler_audit_dispositions.js`; `--audit` gate wired into cargo test
(tests/handler_audit_test.rs), perturbation-proven 4 ways (new-handler/stale/fp-drift/dead-anchor). CENSUS:
664 rows = 597 implemented / 39 noop_justified / 28 unreachable_justified (post-reclass). **TWO REAL LATENT
BUGS the audit surfaced — invisible to the 719/719 e2e AND the 0.026% fuzzer: (1) `gen3_jump_kick_crash_v1` —
JK/HJK onMoveFail never modeled; the crash DRAWS crit + the 16-way roll (+2 draws), fires through Protect,
not vs Ghost, TARGET-maxhp/2 ceiling, Focus-Band-survivable; pins HA1/HA1b. (2) `gen3_freeze_clause_v1` —
Freeze Clause Mod never modeled (Sleep Clause was); a 2nd foe freeze must fail draw-free inside the SetStatus
event; pin HA2. Both zero-corpus-exposure = pure latent.** Lens-1's ONE finding (applied inline by me): the
flashfire onEnd rows were mislabeled unreachable — the sim fires singleEvent(End, ability) on EVERY
switch-out/faint; the port's volatile clear is state-equivalent → reclassified IMPL@execute_switch/
process_faints (597 count). THE FORK-MAINTENANCE PAYOFF: a future submodule bump that adds/changes ANY
handler = a red cargo test, not a silent divergence. REMAINING (long-horizon): protocol level-2 niceties
(Trace |-ability|, taunt/disable -end, FF [silent] -end, Synchronize→Lum ordering, -immune on
status-immunity), Forecast, the BattleStream::write_line drop-in endgame.

**2026-07-07 ABILITY BATCH-1 COMPLETE + VERIFIED (option A wider workflow, `gen3_ability_batch1_v1`, tasks
#63-75).** User: "use up to 98% weekly + DO A (batch classes into wider workflows)." First BATCHED workflow
enumerated all 31 remaining gen3 abilities + wired several draw-free classes at once. The build SELF-HALTED at its
DEFAULT 90% gate (didn't know the 98% auth) → a SAFE partial (wired, byte-identical seed suites, NOT admitted); a
completion workflow (98% auth BAKED IN + fix-first-then-admit ordering) finished it. Verified: 233 tests pass,
dead-guard EMPTY, e2e STRICT 220/220 diverged==0. LANDED classes: **CRIT_IMMUNE** (shellarmor/battlearmor — crit
`randomChance` DRAWN then overridden false via runEvent('CriticalHit'), DRAW-FREE), **WEATHER_SPEED**
(chlorophyll/swiftswim ×2 spe fold into the cached-speed model), **WEATHER_NEGATE** (cloudnine/airlock),
**RESIDUAL** (speedboost +1 spe activeTurns-gated / raindish maxhp/16 in rain, residualOrder 10 sub 3) + NO-OP
admissions (plus/minus/lightningrod/stickyhold). **PRE-EXISTING BUG FIXED (the crux WEATHER_SPEED exposed): the
sun/rain `eachEvent('Weather')` gating** — gen3 sun/rain fire eachEvent('Weather') UNCONDITIONALLY (bare
onFieldResidual body, no isWeather guard) but the port gated the end-of-turn weather tie-shuffle on Sand|Hail ONLY
→ a weather-turn speed TIE under sun/rain missed a draw. SUBTLETY (probe-caught, first fix got it wrong): Cloud
Nine/Air Lock SUPPRESS the sand/hail shuffle (isWeather reads effectiveWeather) but NOT sun/rain's → the fix
schedules off RAW field.weather for sun/rain, effective_weather() for sand/hail. Crux pin + 5 class pins (B1-B4b),
all revert-verified; the reviewer independently built the sharpest crux (an Omastar Swift-Swim weather-CREATED
tie) + the Cloud-Nine Δdraw matrix. **E2E: filter-clean 525 → 571/719 (+46, shellarmor the lever), 11630
decisions, CLEAN first-try (no new bug); taxonomy top gap now `synchronize`, ENGINE_GAP none.** HONEST: Forecast
DEFERRED (probe found it's a forme+TYPE change, NOT a no-op). STILL TODO (task #69): the BLOCK abilities (Suction
Cups phaze-block / Soundproof / Damp) — deferred, small. **LESSON (reusable): subagents SELF-GATE at the DEFAULT
90% weekly via their own check_usage calls — when the user raises the ceiling, BAKE the current authorization
("authorized to N%; if you self-check use --weekly-threshold N") into the build prompt or they halt early.**
Weekly hit 98% (the ceiling) → STOPPED. NEXT window (fresh budget): batch 2 = the draw-bearing procs
(static/poisonpoint/flamebody/cutecharm/effectspore CONTACT_PROC + synchronize + trace + shedskin + shadowtag +
roughskin/colorchange) + berries (lumberry) + the BLOCK abilities.

**2026-07-11 PROTOCOL REVIEW F1-F5 CLOSED (observation-only, workflow wf_775a2e34-8a3, tasks #116-121).**
All 5 Lens-1-diagnosed emission/boundary-layer findings fixed, each byte-verified by a NEW capture scenario;
ZERO engine/state/draw/seed impact (proven by e2e md5 `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED). **F1**
sub-blocked Leech Seed → `[still]`+`-fail` (turn.rs ~4048 (4b) arm); **F2** missed TryHit-ability immunity
(Flash Fire / Water&Volt Absorb) now acc_hit-gated → emits `[miss]`+`-miss` not `-immune` (Levitate/chart-0×
stay pre-accuracy `-immune`); **F3** landed Water/Volt Absorb → `-immune|[from] ability: <Name>`; **F4**
write_line choice-revision PROBED (sim is last-write-wins) but DOCUMENTED-not-implemented (`gen3_writeline_choice_revision_v1`
— an overwrite rule destabilized the writeline gate: a forced-replacement `>p2 switch N` dropped its chunk;
port is first-accepted-wins, unreachable in production since the bridge sends one choice/request); **F5** stale
event.rs:306 Trace-doc corrected. Protocol gate **114→132 battles / 16115→19348 lines** byte-equal, 22 scenarios,
0 deferred. BOTH adversarial lenses PASS (fresh independent sim captures). Lens-2 caught ONE honest bookkeeping
drift I then CLOSED: `writeline_capture_golden.txt` had the 3 new F-scenarios added to its harness but was never
regenerated (19 committed vs 22 from harness) — regenerated to **44 battles / 2377 writes / 7276 lines / 22
scenarios**, the 19-scenario prefix BYTE-IDENTICAL (md5 `59309785...` both), writeline_test green, CLAUDE.md
tallies updated. Handler-audit gate green (664 rows), user's manual `handler_audit_dispositions.js` edit
PRESERVED (md5 `488472b2`). A/B fuzzer unpoisoned throughout (workflow used isolated CARGO_TARGET_DIRs; shared
`ab_replay` untouched) — auto_0711_0307 at **2400/2400 ok / 0 diverged / 0 panic** (~7h, 215 species): the
fix-queue-#4 steady-state fixes hold bit-for-bit. NOT shipped (no /gen3ai-ship). **This closes the protocol
emission + write_line drop-in surface — the level-2 goal is done.**

**2026-07-11 NEXT MAJOR TASK = DROP-IN BRIDGE INTEGRATION (user-chosen). Phase 0 (capture oracle) DONE.**
Goal: a Rust binary speaking the EXACT stdin/stdout protocol of `src/utils/bridge/local_sim_bridge.js`
(START/CHOOSE/FORCELOSE/END → base64 per-side chunks + `__RECON__`/`__END__`), so `--use-showdown-bridge`
training+eval run with ZERO Node, and the search-teacher's clone-and-branch drops the JS `serializeBattle`
path. **The 2 real gaps** (rest is a thin wrapper — `write_line` already emits the omniscient stream
byte-for-byte): **G1** per-side `|request|{...}` JSON (poke-env's legal-action request — NOT emitted by the
crate today; pokesim already COMPUTES the legality [PP/Taunt/Disable/trapping/forced-replacement], so it's
serialization not new mechanics); **G2** per-side visibility split (`getPlayerStreams` equiv — gen3 has no
team-preview + an HP privacy fold: own HP exact `463/463`, foe HP as `%`). Plan = 5 phases, each differential
vs the real Node bridge: 0 capture oracle → 1 `|request|` emitter → 2 per-side split → 3 the bridge binary
(`src/bin/sim_bridge.rs` + `Battle::reseed` for CHOOSE resumeReseed + `__RECON__`) → 4 Python wiring
(`local_battle_runner.py`/`bridge_session.py`) + parity+FPS. **PHASE 0 built** (Node-only, no engine change):
`harness/gen_bridge_capture.js` → `tests/vectors/bridge_capture_golden.txt` (30 gen3ou battles, 2746 `|request|`
frames, byte-reproducible md5 `f90f89cf…`) + `bridge_request_schema_samples.json`. **REQUEST SCHEMA pinned from
the real sim**: move req `{active:[{moves:[{move,id,pp,maxpp,target,disabled}], maybeTrapped?, trapped?}],
side:{name,id,pokemon:[{ident,details,condition,active,stats,moves,baseAbility,item,pokeball}]}}`; forceSwitch
req `{forceSwitch:[true],noCancel:true,side}` (no `active`, fainted mon `condition:"0 fnt"`); a `{wait:true}`
req to the non-choosing side; `|request|` rides a `|sideupdate|` frame; typed-HP shows `move:"Hidden Power
Steel 70"` but `id:"hiddenpower"` (bare id — the opp-HP-typeless convention). **TRAPPING characterized (user
directive — Dugtrio/Magneton)**: `harness/probe_bridge_trapping.js` (9-case matrix, ALL PASS vs real sim) +
`gen_bridge_trapping_capture.js` → `bridge_trapping_golden.txt` (byte-repro md5 `91c44449…`, 4 `trapped:true`
frames) + `harness/BRIDGE_TRAPPING_NOTES.md` (fuzz plan). **THE STATE MACHINE the Phase-1 emitter MUST match**:
trap-ness recomputed PER REQUEST from `is_trapped` (NON-sticky); trapped+live-bench+no-reject-yet →
`maybeTrapped:true`; a REJECTED `>pN switch K` → emit `|error|[Unavailable choice] Can't switch: The active
Pokémon is trapped` + re-request with `trapped:true` (drop maybeTrapped, KEEP moves — only switch is disallowed,
moves array UNCHANGED); NO live bench → OMIT both flags; trapper leaves → neither next req. gen3 NEVER emits
`trapped:true` on the first request (why Phase-0's legal-only driver only saw maybeTrapped). Arena Trap =
grounded foes (Flying/Levitate escape; grounded GHOST IS trapped — T5); Magnet Pull = Steel foes (Skarmory
Steel/Flying trapped); Shadow Tag = same machine, unconditional. poke-env: `trapped`→empties available_switches,
`maybeTrapped`=display-only. NOT shipped (Phase 0 is untracked harness/goldens). NEXT = Phase 1 (`|request|`
emitter in the crate, validated vs these goldens) — will likely fuzz the trapping request path per the notes.

**2026-07-11 (cont.) PHASE 1 DONE + VERIFIED; cron retired; request/per-player A/B FUZZER building.** Phase 1
(the per-side `|request|` emitter G1+G2) built in **`src/bridge.rs`** (`run_full_battle_bridge(opts,cmds,dex)
→ BridgeStreams{p1,p2}` + the fold `derive_side`/`fold_hp_line` + `build_request`/`serialize_side`) + the
additive binary **`src/bin/bridge_replay.rs`** + gate **`tests/bridge_test.rs`**. INDEPENDENTLY VERIFIED:
`bridge_trapping_streams_byte_equal` PASSES (the full trapped state machine byte-identical to the real sim —
16 request frames / 4 `trapped:true` / 4 `|error|`), omniscient path UNTOUCHED (e2e md5 `a23d77ac…` unchanged,
protocol/writeline/e2e_fuzz green, 292 tests), dead-guard clean. **KEY derived facts** (for the CLAUDE.md
write-up): HP fold `pct=ceil(100*hp/maxhp)` clamped (`100 && hp<maxhp → 99`), `0 fnt` unchanged, status suffix
kept — BUT gen3customgame has `debug:true`→`reportExactHP` so BOTH sides see exact HP (only gen3ou folds);
per-mon request key order `ident,details,condition(secret HP),active,stats{atk,def,spa,spd,spe},moves[],
baseAbility,item,pokeball`; per-move `{move,id,pp,maxpp,target,disabled}`, typed-HP `id:"hiddenpower"` bare in
the move but TYPED id in `side.pokemon.moves`; `noCancel` only on forceSwitch when <2 non-wait requests; the
trapped re-request appends trailing `,"update":true`; owner-only lines (gen3 Pressure `-ability [silent]`,
Intimidate `-hint`) dropped for the other side via `|split`. **SCOPE finding:** the 30-battle
`bridge_capture_golden` is only PREFIX-equal (0/30 full) — the Phase-0 capture used unrestricted random choices
→ unmodeled moves (engine fail-louds) + unspecified genders (construction gender-ratio `sample` the engine
doesn't model). NOT a bridge defect → the A/B fuzzer MUST gen teams with MODELED MOVES + EXPLICIT GENDERS. One
observation-only crate fix: `switch_details` now appends `, M`/`, F` (was dropping gender; byte-neutral for the
genderless goldens). **User (2026-07-11) chose: build the request/per-player A/B fuzzer NOW + RETIRE the stale
autonomous cron** — DELETED `trig_01Js7YdC4yW11c31Hk7wxVP3` ("pokesim autonomous window", its batch queue was
complete; the port is shipped + interactive now). A/B fuzzer (`harness/bridge_ab_fuzz.js`) BUILDING in bg
(agentId a67121673019514e3): random teams (modeled+gendered) → Node getPlayerStreams oracle vs Rust
`bridge_replay`, byte-diff per-side chunks + `|request|` JSON, trapping rejected-switch probes, fault-injection
proof, isolated `/tmp/pokesim_target_bridge`, out `harness/bridge_ab_fuzz_out/` (NOT the live `ab_fuzz_out/`).
Live A/B fuzzer (omniscient) still clean at 15,873 battles / 0 diverged. Weekly 60% (live cap 75). Phase 1 +
fuzzer NOT shipped. NEXT after fuzzer: Phase 2 (per-side split validation over the wide corpus) → Phase 3
(the `sim_bridge` binary + `Battle::reseed`) → Phase 4 (Python wiring `local_battle_runner`/`bridge_session` +
parity+FPS).
**A/B FUZZER DONE + VERIFIED (2026-07-11).** `harness/bridge_ab_fuzz.js` (sibling of `ab_fuzz.js`; modes
trapping[default]/randbats/random/pool, `--format gen3customgame|gen3ou`) drives Node `getPlayerStreams` oracle
vs Rust `bridge_replay --ab` (new additive mode, first-div taxonomy preamble/perside/privacy/request/error/
chunk_count/panic), out `harness/bridge_ab_fuzz_out/` (gitignored, NOT the live `ab_fuzz_out/`), isolated
`/tmp/pokesim_target_bridge`. Smoke 200/200 ok (agent) + my independent 60/60 ok / 0 diverged (2672 req / 339
trapped:true / 243 error). Fault-injection proof: 3 perturbations each caught w/ right kind + restored
byte-identical. **2 REAL Phase-1 emitter bugs found+fixed (revert-verified, pinned in the now-3-scenario
`bridge_trapping_golden`)**: (1) `gen3_shadowtag_firm_trap_v1` — gen3 Shadow Tag sets `trapped=true` DIRECTLY
(first request `trapped:true`, NO maybeTrapped phase; a rejected switch draws `|error|[Invalid choice]` with NO
re-request — vs Arena Trap/Magnet Pull's `maybeTrapped`→`[Unavailable choice]`+re-request); fix = `turn.rs::
trap_is_firm` read in `bridge.rs`. (2) `gen3_struggle_activate_sideupdate_v1` — a forced-Struggle mon emits
`|-activate|<mon>|move: Struggle` as an owner-only per-side sideupdate line before the broadcast `|move|`.
VERIFIED: e2e md5 unchanged, bridge_test 3/3 + protocol/writeline green, dead-guard clean, 292 tests, live
omniscient fuzzer unaffected (21,347/0). **HONEST GAPS (disclosed, NOT fixed — pre-existing OMNISCIENT-stream,
orthogonal to the request layer, owned by ab_fuzz.js)**: non-L100 `details` level (`, L84` — port targets
L100); **mid-battle Intimidate `-unboost atk|0` at the −6 floor** (turn.rs:6667 hardcodes delta −1 → always
emits `atk|1`; sim emits clamped-applied 0; repro saved; fix needs threading the pre-drop boost through the
observation-only emit — a real byte bug, latent [no committed gate hits it], deferred); Toxic-`[from]`/status/
Water-Absorb clusters already in ab_fuzz.js's queue. NOT shipped — awaiting the user's `/gen3ai-ship`.

**2026-07-11 (cont.) INTIMIDATE FLOOR FIX + PHASE 2 (sim_bridge binary) DONE + VERIFIED.** (1) The A/B-fuzzer-found
Intimidate −6-floor emit bug FIXED: `turn.rs` hardcoded `|-unboost|<foe>|atk|1`; sim emits the CLAMPED-APPLIED
delta (`atk|0` at the −6 floor — probe-confirmed the line is emitted, NOT omitted; Showdown always reports a
boost/drop's clamped result, e.g. Swords Dance into +6 cap → `|-boost|…|atk|0`). Fix threads pre-drop foe Atk
stage (`intim_atk_pre` snapshot BEFORE `single_event_ability_start` applies+clamps) into `emit_ability_start_lines`
→ `applied = new − pre` ∈ {−1,0} via new `protocol.rs::unboost_atk_applied` (emits at 0). Pinned by new
`intimidate_atk_floor` protocol scenario (22→23 scenarios). Observation-only (e2e md5 unchanged, existing goldens
byte-identical, revert-verified fails at the atk|1-vs-atk|0 line). (2) **PHASE 2 = the `sim_bridge` DROP-IN BINARY**
(`src/bin/sim_bridge.rs`) speaking `local_sim_bridge.js`'s EXACT protocol (START/CHOOSE/FORCELOSE/END →
`pN <base64(chunk)>`/`__END__`/`__ERR__`, persistent mode, replay-from-genesis like write_line, hand-rolled
base64 no-deps). New `bridge.rs::run_full_battle_bridge_chunked[_ended]` → `BridgeChunks` preserving the
`getPlayerStreams` FLUSH boundaries (framing 3/side, turn=log+request/side, trapped-reject=error+re-request chunk,
Struggle=`-activate` chunk); `run_full_battle_bridge` delegates via `.flatten()` (pinned `chunked_flatten_equals_
flat_streams`). Validated by `harness/gen_sim_bridge_diff.js` (spawns BOTH node local_sim_bridge.js + Rust
sim_bridge, byte-diffs per-side `pN` chunks): my independent smoke 21/21 ok / 0 diverged (persistent, trapping, 15
trapped:true); agent's 35/40 + 111/120 persistent, 0 diverged (skips = speed-tie-lead construction-seed convention,
NOT a bridge defect — same pre-first-decision seed workaround the whole engine uses; `seed=None` training unaffected;
`advance_seed_for_construction` models the turn-0 Quick Claw, pinned `construction_seed_advance_matches_the_sim`).
**HONEST DEFERRALS** (documented in the binary, serve the search layer NOT core training): NO `__RECON__` (the port
has no byte-identical `input_log`; `local_battle_runner.py` degrades gracefully — verified) + `resumeReseed` ignored
(needs `Battle::reseed`, still `todo!()`). Observation-only: e2e md5 `a23d77ac…` unchanged, bridge/protocol/writeline/
e2e_fuzz green, full suite green, dead-guard clean. **NEXT PHASE = Python wiring**: swap `local_battle_runner.py:37/157`
(`_BRIDGE_JS` / `create_subprocess_exec("node", _BRIDGE_JS)`) + `bridge_session.py`'s spawn to exec the Rust
`sim_bridge` binary; run a `--use-showdown-bridge` training smoke + measure FPS. Weekly ~63-66% (climbing toward the
75 cap) → RECOMMEND `/gen3ai-ship` the bridge stack (Phase 1 + A/B fuzzer + Intimidate fix + Phase 2) + PAUSE before
the Python wiring this week. STILL uncommitted.

**2026-07-11 (cont.) BRIDGE STACK SHIPPED (5f05efc) + `--use-bridge=rust|node` PYTHON WIRING DONE+VERIFIED (GO,
uncommitted).** Bridge stack (Phase 1 + A/B fuzzer + Intimidate fix + Phase 2) landed on main as **5f05efc**.
Then the user RAISED the weekly cap to 83% + asked to wire the Rust-vs-Node bridge choice as `--use-bridge=rust/node`
(correctness > robustness > larger goal). BUILT + adversarially reviewed: `--use-bridge {off,node,rust}` (default
`off`=websocket; `--use-showdown-bridge` kept as a DEPRECATED alias for `=node`), resolved to `args.use_showdown_bridge`
(bool) + `args.bridge_impl` (node|rust) in `train_rl_agent.py`; threaded through `attach_bridge_transport(...,impl=)` /
`run_local_battles(...,impl=)` / eval_worker shard cfg / the callbacks / launcher `child_uses_bridge`. New
`utils/bridge/sim_bridge_bin.py` (`resolve_sim_bridge_bin` — `$POKESIM_SIM_BRIDGE_BIN` override → cargo build
`--release --bin sim_bridge`, FAIL-LOUD never silent node fallback; `bridge_spawn_argv(impl)` = the one impl→argv
seam). Deferral guards: `rust` emits a startup warning (no `__RECON__`/`resumeReseed`) + `parser.error`s if
`--search-teacher`/`--teacher-persistent` is on; forensic/reconstruction/counterfactual drivers stay node-only
(call `run_local_battles` without `impl=`). **2 REAL integration bugs found+fixed+pinned**: (1) poke-env serializes
choices by move-id/species-NAME (`move hiddenpowerice`/`switch Salamence`) NOT slot numbers → the Rust `parse_choice`
was slot-only + crashed; now resolves names→slots (pin `name_choices_resolve_to_the_same_slots_incl_typed_hidden_power`);
(2) NICKNAMES — a nicknamed mon (Zapdos="Electhor") emitted its `|switch|`/`|move|`-target ident by SPECIES not the
on-field nickname → poke-env `ValueError: team already has 6 pokemons`; fix = `turn.rs::display_name` returns the
nickname (+`switch_details` new `species_name`), pinned BOTH the omniscient path (`regression_test.rs:6316
nicknamed_mon_renders_nickname_in_every_ident_not_species`, `gen3_nickname_ident_v1`) AND the per-side bridge path
(`bridge.rs:1290 switch_and_move_ident_tokens_use_the_nickname_not_the_species`), both revert-verified, probe
`harness/probe_nickname_perside.js`. VERIFIED (independent): full `cargo test` all green (no concurrent-edit
contamination from 2 overlapping nickname agents), Python unit suite **3604 passed** (additive/flag-off byte-identical),
rust-vs-node parity **0 transport errors / 0 parse errors** (N=40), e2e md5 `a23d77ac…` unchanged, dead-guard clean.
**HONEST CAVEAT (the LARGER-GOAL blocker): `--use-bridge=rust` is NOT usable on real gen3ou teams yet** — the parity
run completed **0/40** default-pool battles because every real `data/teams/` gen3ou team carries a move the port
doesn't model (Wish/Aromatherapy/Baton Pass/…) → the engine FAIL-LOUDS (correct, no silent desync). Transport is
proven correct; the ENGINE'S MOVE COVERAGE blocks serverless rust training on the standard pool. NEXT for the larger
goal = **expand the port's modeled move set** (Wish/Aromatherapy/Baton Pass/… — the unmodeled-move tail) OR curate a
modeled-universe team pool. Wiring is correct+safe+additive → recommend `/gen3ai-ship` the wiring; move-coverage is the
next unit. Uncommitted (awaiting `/gen3ai-ship`).

**2026-07-12 MOVE-COVERAGE ARC STARTED (user: "support all gen3 teams → all aspects of gen3", weekly cap→83%).**
The `--use-bridge=rust` unlock needs the port to MODEL all moves our 722 gen3ou teams carry. SCOPED
(`harness/MOVE_COVERAGE_PLAN.md` + `scan_move_coverage.js` static + `src/bin/scan_move_probe.rs` empirical engine
oracle): abilities/items 100% done; the gap is 100% MOVES. Of 108 distinct moves across 722 teams: **62 MODELED /
15 MISMODELED (RUN but silent-desync — the CORRECTNESS priority) / 31 UNMODELED (fail-loud, safe)**; only 8/722
fully engine-playable. Greedy team-unlock: 35 classes → 722; first 9 → 380, first 13 → 536. **BATCH 1 DONE+VERIFIED
(uncommitted)** — the 5 DRAW-FREE-ish MISMODELED post-hit classes (fixed 7 of the 15 silent-desyncs): RECOIL
(`apply_recoil`, Double-Edge/Take Down/Submission, `recoil[num,den]` to user, Rock Head negates), DRAIN
(`apply_drain`, Giga Drain, floor/ceil-behind-sub of dmg, heal-at-full fails, Liquid Ooze excluded), SELF-DROP
(`apply_self_drops`, Overheat −2SpA/Superpower −1Atk−1Def — **PROBE OVERTURNED my "draw-free" assumption: gen3
`selfDrops` DRAWS one `random(100)` unconditionally**, needed a new extractor `selfDrops` field), ITEM-REMOVAL
(`apply_item_removal`, Knock Off removes [gen3 NO dmg boost] / Thief steals iff attacker itemless, Sticky Hold
blocks, `MonState::item_knocked_off` gate — a knocked-off mon can't take/gain an item, pin MC7), RAPID-SPIN
(`apply_rapid_spin`, clears user's own Spikes+Leech Seed via onAfterHit+onAfterSubDamage). VALIDATION: class-sweep
`gen_movecoverage_batch1_golden.js`→`movecoverage_batch1_test.rs` (1040 game-end battles, 10428 per-decision
STATE+SEED assertions), 10 revert-verified pins MC1-MC8, e2e OBSERVATION-NEUTRAL (pre-regen md5 unchanged) then
deliberate regen → new md5 **`dac97afb25317cc9def204ccc9af0e8d`** (was a23d77ac), 220 battles STRICT clean, other
seed suites BYTE-IDENTICAL, full cargo 50 binaries/308 tests green, dead-guard clean. 2 real-team bugs fixed (drain
CEIL float-epsilon → exact rational; itemKnockedOff gate) + 1 code-review bug (recoil/drain must use post-Focus-Band
`dealt`, pin MC8). NO version bump (engine-behavior, obs/data-neutral — extractor+obs parity pass). LOOSE END:
`scan_move_coverage.js` static classifier is STALE post-batch-1 (still lists the 7 as MISMODELED — engine models them
now; refresh its modeled-list in batch 2). **REMAINING: 8 MISMODELED (focuspunch/pursuit/beatup/thunder/hyperbeam/
solarbeam/doomdesire/waterspout — next correctness priority) + 31 UNMODELED.** Batch 2 (next weekly window): the
draw-free UNMODELED quick-wins (status-cure Refresh/HealBell/Aromatherapy, weather-set RainDance/SunnyDay, screens
LightScreen, stat-drop moves Screech/Charm) + the remaining MISMODELED. Weekly 75% (cap 83%) → SHIP batch 1 + PAUSE
until the Mon reset. Independently verified by me (cargo/e2e md5/dead-guard/pins).

**2026-07-12 BATCHES 1+2 SHIPPED (`8a1bd07`, cap→90%).** Batch 2 (draw-free UNMODELED status moves) done+verified
+ shipped WITH batch 1 in one commit. BATCH 2 classes: STATUS-CURE (`run_status_move` cure arms — Refresh cures
any major status EXCEPT slp/frz INCL. Toxic; Heal Bell iterates team + skips a Soundproof ally; Aromatherapy
`clearStatus` single `-cureteam`, no Soundproof gate), WEATHER-SET (Rain Dance/Sunny Day, 5-turn TIMED weather +
`apply_weather_chip` countdown; DRAW-FREE at distinct speed, the eachEvent shuffle draws ONLY on a speed tie;
setWeather-into-SAME fails draw-free, expiry emits `-weather none` + STILL fires the shuffle), STAT-DROP (data-driven
`statDropBoosts` new extractor field — Screech/Charm/Metal Sound/Feather Dance/Tickle/Fake Tears, acc roll +
draw-free boost), SCREENS (`SideState::{light_screen,reflect}` + the damage-calc fold — **PROBE CRUX: a hit into a
side with BOTH Reflect+Light Screen draws ONE extra `random(0,2)`** — the two screens' ModifyDamagePhase1 handlers
tie → size-2 shuffle; pin MC17). 9 pins MC9-MC17, class-sweep golden 1360 battles/16178 assertions. **HONEST
DEFERRAL — batch 2 kept OUT of the STRICT e2e (`BATCH2_E2E_EXCLUDED=true`, phaze precedent)**: admitting it fixed a
real Refresh-doesn't-cure-Toxic bug (6 divergences) but surfaced **`e2e_182`** — an Aromatherapy-cure-while-p1-switches
RESIDUAL-HEAL-ORDERING desync (port reaches full HP one residual tick early; state-only seed-matching) NOT
root-caused. Engine mechanics bit-for-bit by the dedicated golden; e2e golden stays `dac97afb…` (unchanged, strict
clean, 220/220). **`e2e_182` IS THE #1 CORRECTNESS FOLLOW-UP — root-cause it, then re-enable batch-2 e2e.** VERIFIED
(me): full cargo 51 binaries green, e2e md5 dac97afb unchanged, all pre-existing seed suites BYTE-IDENTICAL,
dead-guard clean, Python unit suite 3607 (gen3_moves.json data obs-neutral — extractor+obs parity green). SHIPPED
`8a1bd07` (batches 1+2, rebased over launcher-fork commit, no overlap). **COVERAGE NOW: 71/722 teams fully
engine-playable** (was 8); 77 MODELED / 8 MISMODELED / 23 UNMODELED. `scan_move_coverage.js` refreshed (was stale).
Remaining top gaps: curse(241 UNMODELED), wish(213 UNMODELED), focuspunch(196 MISMODELED), pursuit(159 MISMODELED),
batonpass(158), + the 8 MISMODELED silent-desyncs (focuspunch/pursuit/beatup/thunder/hyperbeam/solarbeam/doomdesire/
waterspout — the turn-loop risky ones). **NEXT: (a) root-cause e2e_182, (b) BATCH 3 = the stateful draw-free (Curse,
Wish [python wish_belief.py exists], Baton Pass) + the risky MISMODELED silent-desyncs with dedicated probes.**
Weekly ~80% (cap 90%).

**2026-07-12 (cont.) e2e_182 FIXED + BATCH 3 DONE (cap→92%; UNCOMMITTED, ready to ship w/ e2e_182).**
**e2e_182 root cause was NOT residual-ordering (red herring) — a PRE-EXISTING PRESSURE PP BUG**: the extra-PP
deduction keyed on `!targets_self`, but gen3 fires Pressure `onDeductPP` ONLY when the foe is in the move's
`pressureTargets` (a FOE-directed target). `allyTeam`/`self`/`allySide`/`foeSide` never put the foe there → deduct 1
not 2. So Aromatherapy (target `allyTeam`) under a Pressure foe wrongly drained 2 PP → exhausted early → the port
REJECTED it out-of-PP → `run_full_battle` pulled the WRONG next script token (Soft-Boiled) → desync. Fix
`turn.rs::pressure_targets_foe(target)` (`gen3_pressure_allyteam_v1`), pin PA1
`pressure_does_not_add_pp_for_an_allyteam_move` (revert→PP 8→6 fails). Batch 2 RE-ADMITTED to strict e2e
(`BATCH2_E2E_EXCLUDED=false`), md5 `dac97afb`→**`738da13e9ab666ae50ead17bc6329a08`**, 722/722 filter-clean, STRICT
clean. **BATCH 3 = Curse/Wish/Baton Pass** (top-3 remaining, 241/213/158): CURSE (non-Ghost self-boost
+1Atk/+1Def/−1Spe — **NOT draw-free, rides `selfDrops` random(100)** [re-probe-confirmed, curse=2 draws vs Harden=1];
Ghost lays `curse` on foe + pays floor(maxhp/2) + residual floor(maxhp/4)/turn order10 sub8), WISH (slot-keyed
`slotCondition` dur 2, order-**7** delayed heal floor(maxhp/2) at N+1 — fires BEFORE sand chip[8] + order-10;
survives switch/faint/phaze; matches `wish_belief.py`), BATON PASS (`selfSwitch:copyvolatile` — passes 7 boost stages
+ copyable volatiles sub/leech-seed/confusion/curse; major status does NOT pass). 12 pins MC18-MC29 (critical MC23
Wish-residual-order revert-verified: order 7→11 fails). e2e OBSERVATION-NEUTRAL then regen → md5
**`529ab3f0940f8f9cbab383fb26d2a696`**, **722/722 filter-clean, STRICT clean**. HONEST DEFERRAL (not an engine gap):
a bridge-`|request|`-JSON DISPLAY nuance (Curse non-Ghost `nonGhostTarget` renders `target:"self"` vs `"normal"`;
`return102` codec alias) — no legality/draw/state impact, handled by a documented normalizer in the scope-audit test
(gender/alias-tolerance precedent). VERIFIED (me): full cargo **52 binaries/332 tests green**, e2e md5 529ab3f0,
all other seed goldens BYTE-IDENTICAL, dead-guard clean, MC18-MC29 present. **COVERAGE 8 → 71 → 232/722 teams fully
engine-playable this session** (80 MODELED / 8 MISMODELED / 20 UNMODELED). Remaining: the 8 MISMODELED silent-desyncs
are the RISKY turn-loop ones (focuspunch 196 / pursuit 159 / beatup 114 / thunder / hyperbeam / solarbeam / doomdesire
/ waterspout) + UNMODELED counter(65)/return/etc. **NEXT BATCH 4 = the risky turn-loop silent-desyncs** (Focus Punch
beforeTurn, Pursuit switch-interrupt, Beat Up multi-strike, 2-turn Hyper/Solar Beam, Doom Desire future-move) — each
needs a DEDICATED probe (draw-order). Session ships: bridge stack `5f05efc`, `--use-bridge` `a86ee6d`, batches1+2
`8a1bd07`; e2e_182+batch3 UNCOMMITTED ready. Weekly ~85% (cap 92%).

**2026-07-13 BATCH 3 SHIPPED (`b2c045f`) + BATCH 4a DONE (Focus Punch + Pursuit; UNCOMMITTED, ready
to ship; cap→95%).** Batch 3 (Curse/Wish/Baton Pass + the Pressure-allyTeam e2e_182 fix) is now COMMITTED
as `b2c045f` (my prior note had it uncommitted). **BATCH 4a** = the 2 highest-value MISMODELED "risky
turn-loop draw-order" moves (`gen3_move_coverage_batch4_v1`), built via a probe→build→2-lens-review
workflow, both lenses **SHIP**: **FOCUS PUNCH** (196 teams — new `QAction::BeforeTurnMove` order-5 queue
action for the `|-singleturn|` + the onTry lost-focus cancel + flinch-block) + **PURSUIT** (159 — the
switch-interrupt strike at ×2/never-miss in `execute_switch`). Probes settled the cruxes: FP is
acc-100-DRAWS (not never-miss); a cancelled FP draws NOTHING; **chip absorbed by the user's OWN Substitute
does NOT break focus**; Pursuit's ×2 + never-miss both gate on `switchFlag`. Engine: +293 lines `turn.rs`
+ `state.rs` fields (`focus_punch`/`pursuit`/`pursuit_strike`). Validated: dedicated 1040-battle golden
(`movecoverage_batch4_test.rs`, 8866 seed + 17732 HP asserts) + **6 revert-verified pins MC30-MC35**; full
**53 binaries / 339 tests green**; all pre-existing seed goldens BYTE-IDENTICAL; dead-guard clean. **e2e
DEFERRED (`BATCH4_E2E_EXCLUDED=true`, PHAZE precedent):** admitting FP/Pursuit to the strict e2e surfaced 3
REAL divergences (of 220) in COMPLEX turns where a Pursuit-interrupt composes with Baton Pass + Roar +
double-faint → switch-array slot-assignment desync (e2e_11 rust 34 vs golden 55); committed e2e golden
stays `529ab3f0…` byte-identical. **THE BENCH-ORDER BUG IS `--use-bridge=rust`-REACHABLE (reviewer's
substantive point), not just an e2e artifact** — `execute_switch`'s interrupt is unconditional + Pursuit is
common; in the bridge the port IS the env so a rare multi-mechanic turn = silently-wrong obs. I added a
bridge-safety caveat to CLAUDE.md + fixed a false `move_has_before_turn_callback` doc comment (4 gen3 moves
carry beforeTurnCallback: counter/mirrorcoat[fail-loud]/pursuit/focuspunch). COVERAGE: **232 → ~250/722**
teams engine-playable. **NEXT (next weekly window, fresh budget): (1) the bench-order composition fix (#1
correctness follow-up — needs slot-array reconciliation across pursuitfaint-requeue/BatonPass/Roar/
double-faint; likely >1 fix; unblocks batch-4 e2e admission AND fixes the bridge-reachable bug), then
re-admit batch 4 to e2e; (2) Batch 4b = the remaining 6 MISMODELED (beatup/thunder/hyperbeam/solarbeam/
doomdesire/waterspout).** Deferred nits (non-blocking): 2 missing pins (pursuit-interrupt-into-hazards +
speed-tie-interrupt — reviewer verified bit-for-bit, seeds in the wf_a7d04ae2-5af journal). Quota: weekly
88% (cap RAISED to 95% this session); ran ONE heavy workflow this window, stopped clean rather than clip.

**2026-07-13 (cont.) BATCH 4a SHIPPED (`946145b`) + BENCH-ORDER BUG FIXED → BATCH 4 FULLY e2e-ADMITTED
(UNCOMMITTED, ready to ship; cap→98%).** Batch 4a (Focus Punch + Pursuit) shipped as `946145b`. Then a
diagnose→fix→2-lens-review workflow (both lenses **SHIP**) closed the #1 follow-up — the Pursuit-interrupt
"bench-order" desync — which was ROOT-CAUSED (sim as oracle) to a **Baton-Pass over-interception**: the port
fired the Pursuit interrupt for ANY non-drag switch-out incl. a Baton-Pass selfSwitch, but the sim SUPPRESSES
`BeforeSwitchOut` on a Baton Pass via `batonpass.self.onHit → skipBeforeSwitchOutEventFlag` (moves.ts:1109) —
so the port struck+KO'd the passer where the sim keeps it alive → the "fainted mon in a slot the sim has alive"
symptom. FIX = an `is_voluntary` gate on `execute_switch`'s interrupt (true only for `QAction::Switch`; false
for InstaSwitch/Baton-Pass/faint-replacement + drag). Admitting batch 4 to the e2e surfaced **2 MORE real bugs,
both root-caused + fixed**: (2) pursuit-interrupt first-mover attribution (~15 battles) → `pursuit_first_mover`
override; (3) **Choice-lock not released on item-removal** (e2e_126, Thief steals Choice Band) → clear
`choice_locked_move` in `apply_item_removal`. **RESULT: BATCH4_E2E_EXCLUDED=false, batch 4 FULLY ADMITTED,
STRICT 220/220 diverged 0, 11481 decisions (242 FP + 184 Pursuit genuinely exercised), new e2e md5
`fe1529609264be655f36032e0261868d`, taxonomy 300/300 clean — ZERO e2e exclusions again.** 4 revert-verified
pins MC36/36b/37/38 (incl. the 2 nit pins: pursuit-into-hazards + speed-tie-interrupt). 53 cargo groups /
regression_test 161 / handler_audit 759 rows green; all OTHER seed suites byte-identical (one review lens
INDEPENDENTLY regen'd the e2e golden byte-for-byte identical = strong confirmation). **The `--use-bridge=rust`
bridge-safety caveat is RESOLVED** (the composition is now bit-for-bit). CLAUDE.md updated. Uncommitted, ready
for `/gen3ai-ship`. Weekly 91% (cap 98%). NEXT: Batch 4b = the remaining MISMODELED silent-desyncs
(beatup 114 / thunder 22 / waterspout — single-turn variable-BP/accuracy; then the 2-turn class
hyperbeam/solarbeam/doomdesire).

**2026-07-14 BATCH 4b DONE — Beat Up + Thunder + Water Spout, fully e2e-admitted (UNCOMMITTED, ready to
ship with the bench-order fix; cap→98%).** Probe→build→2-lens-review→fix workflow. All three modeled
bit-for-bit + e2e-admitted (ZERO exclusions). **BEAT UP**: multi-strike TYPELESS stat-swap (`run_beat_up`) —
one strike per healthy non-statused party member (active user counts; statused/fainted skip); ally dex
baseStats.atk→attacker SpA, target baseStats.def→defender SpD at modifier=1 (flat BP 10, no boosts/items/
Choice-Band/burn); draw model = **ONE whole-move accuracy roll + per-strike (crit + damage + eachEvent
Update)**, stops at target faint (no Quick Claw on deciding faint); typeless ??? hits Ghost 1×, Light Screen
applies. **THUNDER**: id-gated onModifyMove weather-accuracy — **effective RAIN removes EXACTLY the accuracy
draw** (never_miss short-circuits the whole stage+accMod chain), SUN=50 / base=70 keep it (same random(100),
different threshold), Cloud Nine/Air Lock → base 70. **WATER SPOUT**: BP=150·hp/maxhp basePowerCallback,
fully draw-neutral. NO extractor/gen3_moves.json change (Beat Up reads dex base stats). New state:
`MonState::beat_up` (duration:1 volatile) + `ProtocolBuilder::hitcount`. Admitting surfaced **3 real bugs,
all fixed+pinned**: (1) multihit fires eachEvent('Update') PER STRIKE (e2e_52); (2) the beatup duration
volatile registers a residual duration handler (e2e_217 mirror → residual tie-shuffle); (3) a Beat Up direct
hit sets the target's Focus-Punch lostFocus (e2e_196). Review MAJOR (fixed in fix-phase): `beat_up` flag not
cleared on switch-out/phaze-out (`execute_switch` clearVolatile omitted it) → added `m.beat_up=false` + pins
MC46/47. **e2e STRICT 220/220 diverged 0, 11407 decisions, md5 `64edcdcd5c6a63b1256fc23d3887d8c7`.** 10
revert-verified pins MC39-MC48; cargo 54 suites green; handler-audit 767 rows; ALL other seed goldens
byte-identical. Both lenses net SHIP (Lens2 FIX_THEN_SHIP → the major fixed). Minor left (non-blocking): the
Water Spout `.max(1)` min-BP clamp + Beat Up empty-allies fizzle branches are unverified by golden/pin (edge
coverage, not a bug). **UNCOMMITTED worktree state** = bench-order fix (batch 4 e2e-admitted, md5 fe152960) +
batch 4b (md5 64edcdcd); 946145b (batch 4a) already on main. Session ran 3 workflows (batch-4a-build +
bench-order + batch-4b); weekly hit 96% (cap 98%) → STOPPED clean (2 pts headroom, not enough for another
workflow). NEXT: the 2-turn move class (hyperbeam recharge / solarbeam charge / doomdesire future-move — a
distinct turn-spanning-state mechanic), then batch 4c cleanup.

**2026-07-14 (fresh week, cap <50%) BATCH 4c DONE — Hyper Beam + Solar Beam + Doom Desire + FUTURE SIGHT,
fully e2e-admitted (UNCOMMITTED; both lenses SHIP; weekly only 14% after).** The turn-spanning class
(`gen3_move_coverage_batch4c_v1`) — ALL 8 original MISMODELED silent-desyncs now CLOSED. **HYPER BEAM**
(`MonState::must_recharge`): lock set DRAW-FREE on a successful damaging hit INCL. sub-absorb/sub-break/
target-KO (persists across the foe's force-switch), NOT on miss/immune/Protect-block; the locked turn =
zero-draw no-PP `|cant|recharge` at priority 11 — fires BEFORE every status handler (a par'd/slp'd locked
user rolls/decrements NOTHING — probed identical seed advance); switching ILLEGAL (request = single
`{"move":"Recharge","id":"recharge"}` + firm `trapped:true`); consumes Truant's loaf (removeVolatile);
**gen3 Quick Claw has NO item handler** (the endTurn roll is battle-level, possession-independent —
consistent w/ the port). **SOLAR BEAM** (`MonState::two_turn`): charge turn pays PP (−1/−2 Pressure) but
draws ZERO move draws (onTryMove `[still]`+`-prepare`, never reaches tryMoveHit); fire turn NO PP, normal
acc/crit/dmg; SUN skips (effectiveWeather — Cloud Nine unskips); **gen3 rain/hail/sand HALVES SB's BP**
(probe: rain 54 vs 105, state-only); fire-turn full-para loses the charge. **DOOM DESIRE + FUTURE SIGHT
(both modeled, probe-settled same-mechanic)**: slot-keyed future strike at N+2 (SideState pending — the
Wish precedent), gen3 TYPELESS damage, stats SNAPSHOTTED at cast, cast draws the screens Phase1 shuffle
(reviewer probe-verified), resolve = crit+dmg draws at the residual; double-cast fails; hits the slot's
NEW occupant. BUILD DISCOVERY (the only golden divergence, root-caused): a resolve-KO draws ONE
eachEvent Update not two (faintMessages between Update-A/B removes the corpse from getAllActive).
VALIDATED: 23-scenario × 80-seed golden (1840 battles, 16621 asserts) + 12 pins MC49-MC60; **e2e
re-admitted CLEAN FIRST-TRY: STRICT 220/220, md5 `77c9205fef0cc0033a718fe549b4d5ca`** (exercises 3 SB
casts; 0 HB/DD decisions in the sampled corpus — honest disclosure, proven by the dedicated golden);
55 cargo binaries green; handler-audit 787; bridge locked-request serialized probe-shaped (honest: not
yet byte-gated — a locked-request bridge capture scenario is a cheap future hardening). POST-REVIEW
INLINE FIXES (me): the must_recharge cant gate moved BEFORE the unmodeled-sibling panic (Lens-2 minor —
a locked mon never resolves a move; verified: regression 183 + batch4c + e2e green) + the PLAN.md pin
count. **REMAINING FRONTIER (scan, greedy unlock): counter(49 unlocks) → return(45) → endeavor(26) →
sleeptalk(26) → perishsong(15) → meanlook(12) → endure(8) → destinybond/encore/snatch/bellydrum/charge/
reversal/flail/frustration/lowkick/memento/mimic/painsplit/psychup → 722/722.** Scan classifier list is
STALE (still lists batch-4b/4c moves as MISMODELED — refresh next batch). UNCOMMITTED: batch 4c awaits
`/gen3ai-ship`. Hit PAUSE_5H (94% 5h; weekly 14%) → auto-resume sleep, next unit = Batch 5
(counter/mirrorcoat reactive + return/variable-BP family + sleeptalk).

**2026-07-15 BATCH 5 DONE — Counter/Mirror Coat/Endeavor + Return/Frustration/Flail/Reversal/Low Kick +
Sleep Talk (9 moves), fully e2e-admitted (UNCOMMITTED with 4c).** `gen3_move_coverage_batch5_v1`.
REACTIVE family: order-5 BeforeTurnMove volatile (`MonState::reactive`, onStart resets EVERY selection
turn — prev-turn damage never counts) + a priority −101 onDamage recorder (2× the LAST qualifying foe
Move hit; counter=Physical||bare-HP, mirrorcoat=Special&&!HP — **typed HP normalizes to bare
'hiddenpower' in-battle → Counter counters HP UNCONDITIONALLY even special-typed, Mirror Coat NEVER**;
sub-absorbs never arm; Seismic-Toss-class + Struggle ARE countered; Beat Up's last strike arms MC);
un-armed fail = ZERO-draw bare-|move| (no -fail!); armed = 1 acc draw → Fighting→Ghost / Psychic→Dark
-immune, NO crit/dmg rolls. ENDEAVOR: zero-draw |-fail| at hp>=target (equality fails), else acc-only,
sets target hp == user hp via sub/immunity machinery. VARIABLE-BP five: draw-neutral BP callbacks; NEW
obs-neutral `weighthg` species field + noSleepTalk/isCharge move flags (extractor run, Python suite 3611
green, dex facade parity unchanged). SLEEP TALK: the sample draw over the eligible pool + called-move
execution. Validated: batch5 golden (18548 seed asserts) + pins MC61-MC78; **e2e STRICT 220/220, md5
`614d47b9a5227dc7ad4e444b2e28313c`** (the fixed-damage family FINALLY exercised on real teams — 271
decisions + 240 batch-5, now GATED coverage floors); one real bug found in admission (Seismic Toss must
set Focus-Punch lostFocus — MC76). REVIEWS (resumed via resumeFromRunId after both lenses died on the 5h
session cap — the quota-parked-review hazard recurred; the resume path worked): both FIX_THEN_SHIP → fix
phase closed all: **MC77 Focus-Band 0-damage hit must NOT arm Counter/MC** (sim's Damage chain breaks on
falsy relayVar before the −101 recorder — a real latent bug), **Snore engine panic guard** (the build's
fail-loud claim was false; now true + should_panic pin), e2e floors gated, stale docs fixed, return102
nit refuted. **2nd reviewer `git checkout` incident** (wiped uncommitted turn.rs mid-review; restored
from pristine + re-proved; I then INDEPENDENTLY verified: 201 regression + all batch goldens + e2e green)
— PATTERN: forbid raw `git checkout -- <file>` in review prompts / tell reviewers to copy the tree.
Final: 388 tests / 201 pins green. Weekly 36% (cap 50). NEXT = BATCH 6 (the FINAL tail to 722/722, ~57
unlocks): perishsong(15)/meanlook-spiderweb-block(12)/endure(8)/destinybond(5)/encore(4)/snatch(4)/
bellydrum(3)/charge(2)/memento/mimic/painsplit/psychup.

**2026-07-15 (cont.) BATCH 6 BUILD DONE — 718/722 teams (99.4%); REVIEWS DEFERRED (budget split);
UNCOMMITTED with 4c+5.** `gen3_move_coverage_batch6_v1` — all 12 tail moves modeled bit-for-bit +
e2e-admitted: ENCORE (acc + durationCallback random(3,7), stored = willMove ? rolled : rolled+1 — the
Disable branch; NON-UNIFORM fail split: no-lastMove/Struggle/0-PP draw acc+duration, already-encored
draws acc ONLY; onOverrideAction deducts the ENCORED slot's PP; order-10/sub-14 residual tick + 0-PP
early -end), DESTINY BOND (zero-draw cast; window closes at the next move attempt incl. every cant site;
mutual faint = a process_faints WORKLIST, both-last-mons = TIE; residual/sub/futuremove KOs never
trigger), ENDURE (**SHARES Protect's stall ladder** 2→4→8 + willAct; priority-−10 clamp at every
MOVE-damage site incl. fixed damage + every multihit strike; does NOT guard residuals; success turn adds
the endure+stall intra-mon residual tie-shuffle), PERISH SONG (**fieldEvent DURATION-END `continue`** —
a perish onEnd faint skips per-handler faintMessages so a tied mirror's mutual perish-out is a TIE — the
batch's one first-pass pin failure, probe-settled), MEAN LOOK/SPIDER WEB/BLOCK (firm trap volatiles on
is_trapped + the request shape; trap-link clears when the trapper leaves; BP escapes; phaze drags),
BELLY DRUM/CHARGE (Baton-Passable but consumed by the BP move itself — probed)/MEMENTO/MIMIC/PAIN
SPLIT/PSYCH UP. Admission surfaced 1 LATENT batch-5-era bug (fixed+pinned MC99): a CONTACT fixed-damage
hit (Seismic Toss into Effect Spore) must fire the defender's contact-proc onDamagingHit. e2e STRICT
220/220, md5 `02fe5d9a59955eaf0360e9d881f46a83`, 58 batch-6 decisions (gated floor); 57 suites / 410
tests / 222 pins green (MC79-MC99, cp-aside revert method — NO git-checkout incidents this time; the
no-git-checkout rule is now baked in prompts). **SNATCH (4 teams) = the ONLY remaining gap**,
deliberately deferred fail-loud (never probed; pinned snatch_status_move_panics_fail_loud) — one probe
round + arm + golden scenario + pin closes 722/722. **REVIEWS NOT RUN** (the budget split: the workflow
script has an early-return before the Review phase — DELETE it + resume wf_01a1618d-571 to run the 2
lenses + fix against the cached build). Weekly ended at 52% (target <50 — a 2-point overshoot; the build
self-gated at 49.0 + deferred Snatch, the final verify pushed past). UNCOMMITTED: batches 4c + 5 + 6.
NEXT (new budget): (1) batch-6 reviews via resume, (2) Snatch mini-unit → 722/722, (3) /gen3ai-ship.

**2026-07-16 (new week, cap 5h<=70/weekly<50) — MOVE-COVERAGE ARC COMPLETE: 722/722.** (1) BATCH 6
ADVERSARIAL REVIEW run on the SHIPPED code (resumed wf_01a1618d-571 after deleting the budget-split
early-return + repointing the review prompts at `git diff fb1c6c9 HEAD` since batch 6 was already committed
in d61244e): **both lenses SHIP**, findings only minor/nit (the known Snatch gap; a documented dead-code
trap-link clear; batch-6 protocol emission not byte-gated — state+seed fully gated, consistent with all
prior batches). NO fixes owed → batch 6 needs no follow-up commit. (2) SNATCH modeled bit-for-bit
(`gen3_snatch_v1`, UNCOMMITTED) — the LAST unmodeled gen3ou move → **722/722 fully engine-playable** (the
--use-bridge=rust endgame). Probe overturned my hypotheses (SIM IS ORACLE): Wish/Spikes/Thunder Wave are
NOT snatchable; the eligible set is EXACTLY the 44 gen3 `flags.snatch` moves (new obs-neutral `isSnatchable`
extractor flag from `move.flags['snatch']`). Mechanic: priority +4 never-miss cast sets a duration-1
`snatch` volatile DRAW-FREE; the foe's snatchable self-move is intercepted at onAnyPrepareHit (after the
foe's |move| announce + PP deduct) → removeVolatile FIRST → `|-activate|…move: Snatch|[of] foe` → the
snatcher recursively useMoves the stolen move on ITSELF (Rest sleeps the snatcher +1 sleep draw, Belly Drum
costs the snatcher HP, Heal Bell cures the snatcher's team); Snatch adds ZERO draws of its own; a snatch
MIRROR draws 8 vs 7 (the duration-volatile residual tie — MC104, revert-verified). e2e CLEAN first-try, md5
`3155eb796cb4bf453c6053d769ba98e5`, 722/722 filter-clean; pins MC100-MC104; cargo 58 suites / 227
regression green; handler-audit 897. Both review lenses SHIP (2 coverage nits: 0 snatch e2e decisions
[4/722 teams — proven by dedicated golden, I fixed the stale CLAUDE.md claim inline]; ally-directed-steal
not in the golden's 7 scenarios). ARC TOTALS: **8 → 722/722 teams**, ZERO silent-desync moves, ~50 moves
modeled across batches 1-6 + snatch, ~9 latent engine bugs found+fixed via e2e admissions/reviews.
UNCOMMITTED (worktree, awaiting /gen3ai-ship): SNATCH only (batches 4c/5/6 already on main d61244e; batch 6
review-clean). Weekly 6% (cap 50), 5h 53% (cap 70). NEXT: /gen3ai-ship snatch; then --use-bridge=rust is
viable on the WHOLE gen3ou pool → the original larger goal (serverless rust training) is unblocked.

**2026-07-16 (cont.) OMNISCIENT-BYTE FUZZER BUILT (Phase A + picker widen) — 4 REAL BYTE BUGS FOUND+FIXED
(gen3_omniscient_byte_fuzz_v1, UNCOMMITTED).** User asked for "an even more robust fuzz test — ideally
validate every publicly observable aspect." 2-agent Explore mapped the gap: the OMNISCIENT |...| byte
stream is NEVER fuzzed CONTINUOUSLY — ab_fuzz checks reconstructed STATE+SEED+winner (not bytes); the byte
gates (protocol_test 132 battles / writeline_test 44) are FIXED captures topping out at BATCH-3-era
mechanics → every batch-4c/5/6/snatch emitted line is byte-emitted but never line-diffed. User chose "A +
widen picker". BUILT: ab_fuzz.js `--protocol` mode tees the REAL omniscient log (|t:|-normalized) → FMT/L
golden rows (state golden untouched → e2e byte-repro); ab_replay.rs byte branch replays via
run_full_battle_logged, filters BOTH sides through a DENYLIST (drop debug/error + |t:| norm — NOT
protocol_test's batch-3 ALLOWLIST, which would blind the fuzzer to the new line types — the probe's
load-bearing finding), first-divergence line-diffs to a new `kind=protocol`. Runs gen3customgame AND
gen3ou (the OU clause-shuffle draw path the e2e capstone never hits — reframe byte-correct, 0 tier/rule
div). Picker widened: Hidden Power now pickable (modeled); genuinely-unmodeled tail (OHKO/psywave/bide/
trick/fakeout/snore/eruption/…) kept excluded. FAULT-INJECTION PROVEN (perturb Leftovers cause → 6/6
kind=protocol at the exact line, cp-aside restore). **4 REAL BYTE BUGS FIXED, each latent (bytes never
fuzzed before), revert-pinned in tests/protocol_byte_fuzz_test.rs: BF1 typed-HP NAME LEAK (port emitted
"Hidden Power Ice", sim emits bare "Hidden Power" — gen3 hides HP type; a real --use-bridge=rust obs leak!)
→ run_move canonicalizes hiddenpower* → bare; BF2 Toxic residual `[from] tox`→`[from] psn`; BF3 self/side
move announce rendered FOE not USER (Light Screen/Sunny Day/Perish Song); BF4 Pursuit interrupt missing
`[from] Pursuit`.** All observation-only (seed goldens byte-identical). Reviews: Lens1 (fuzzer fidelity)
SHIP; Lens2 FIX_THEN_SHIP flagged a "protocol_test flaky |-crit|" MAJOR → REFUTED with determinism evidence
(no unsafe/static-mut/thread-local/HashMap-iter in the battle path; protocol_test 200/200 standalone + 6/6
consecutive parallel). Full cargo 420 passed / 59 binaries green; I independently re-verified. RESIDUAL
(the dominant follow-up = Phase A.2): ~8 general STATUS-MOVE EMISSION FORMS the constructed protocol golden
never covered — the `||[still]` did-nothing fail, status `-fail`/type-immunity `-immune`, Natural Cure
`-curestatus`, sleep `-status [from] move`, confusion `-start`, switch-in `-weather` framing ORDER, the
delta-0 `-boost` at the ±6 cap (the docs' admitted Memento/Intimidate-floor class), owner-only Pressure
`-ability`, `shiny` details, future-move/Beat-Up `-hint`. They hit ~every real-team battle → protocol-clean
rate only ~20-22%. Each = a clean probe→emit→capture-scenario follow-up. Weekly 11% (cap 50), 5h 30%
(cap 70). UNCOMMITTED (byte fuzzer + BF1-BF4). NEXT: Phase A.2 (fix the ~8 emission forms → clean rate →
high), then /gen3ai-ship the whole fuzzer stack.

**2026-07-16 (cont.) BYTE-FUZZER Phase A.2 + REGRESSION CORPUS — protocol-clean 25%→96.6%, ~20 emission
byte bugs fixed (UNCOMMITTED, ready to /gen3ai-ship).** A.2 fix-until-clean workflow enumerated 15 distinct
emission-form divergences (more than the ~8 expected — it self-surfaced Beat-Up per-hit -activate
OVER-emission, Volt/Water Absorb -heal-vs-immune, Protect-blocks-status → -activate-not-immune, Knock-Off
-hint over-emit, Leftovers residual ORDER, freeze-thaw -curestatus, shiny details flag, redundant
same-weather -weather double-emit) and FIXED all: the `||[still]` did-nothing framing + `-fail` (bare vs
`|heal` sub-tag), status type-immunity `-immune`, Natural Cure `-curestatus` on switch/phaze, sleep
`-status [from] move:`, confusion `-start`, the delta-0 `-boost` at ±6 cap (the admitted Memento/Intimidate
class), etc. **Clean rate 25.0% → 96.6%** (CG 97.75% / OU 95.0%). All observation-only → seed goldens
BYTE-IDENTICAL. Both review lenses FIX_THEN_SHIP (minor only): fixed the 2 real minors — (a) untracked
fuzzer-output dir ship hazard → broadened .gitignore to `/harness/ab_fuzz_out*` glob + removed the junk;
(b) BF-F15 stat-drop status move BLOCKED BY SUBSTITUTE mis-emitted (missing the FORM-1 [still]+bare-fail)
→ probed sim + fixed + pinned. **REGRESSION CORPUS (user asked "create folders to prevent regressions"):
new `tests/vectors/byte_fuzz_corpus/` (20 frozen full-battle fuzzer repros, one per fixed emission form +
gen3ou variants + README) + `tests/byte_fuzz_corpus_test.rs` (auto-discovers *.txt, replays each via
`env!("CARGO_BIN_EXE_ab_replay")` byte mode, asserts 0 kind=protocol, floor >=15) — the fuzzer's finds are
now a PERMANENT growable cargo gate; fault-injection-proven (perturb → fails naming the file).** Full cargo
green; 13 protocol_byte_fuzz pins. WHY THESE WENT UNCAUGHT (root cause, recorded [[feedback_validate_
observable_bytes]]): the arc validated MECHANICS via STATE+SEED reconstruction (blind to observation-only
byte forms); the byte gates were FIXED captures frozen at batch-3 → post-batch-3 emission was write-only.
KNOWN RESIDUAL (~3.4%, disclosed): turn-0 construction-shuffle speed-tie [of] attribution on same-species
mirror leads (documented deferral); a Leftovers residual-tie emit ORDER; the HP fixed-BP-70 state caveat
(~1.5% random-mode). UNCOMMITTED byte-fuzzer stack (fuzzer + BF1-BF4 + A.2 + BF-F15 + corpus + gitignore)
ready for one /gen3ai-ship. Weekly ~14% (cap 50), 5h ~53% (cap 70).
