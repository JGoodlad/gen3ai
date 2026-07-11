# Edge-case backlog — pokesim (gen3ou)

## ✅ FIXED — PROTOCOL review FINDINGS F1-F5 (emission/boundary-layer; `gen3_protocol_phase3_review_v1`, 2026-07-11)

Five Lens-1-diagnosed protocol gaps, all **EMISSION/BOUNDARY-layer** (no engine state/draw/seed impact → the
e2e golden md5 **`a23d77ac60d4af168b8a4428f0b465c9` stays UNCHANGED** + every seed suite byte-identical). Each
line form was nailed against the resolved `Dex.mod('gen3')` sim (the ONLY oracle) then byte-verified by a NEW
capture-golden scenario:

- **F1 — sub-blocked Leech Seed.** A Leech Seed into a SUBSTITUTE'd foe is blocked; the sim's `moveHit` returns
  `false` → gen3 retro-edits the announce to `|move|<user>|Leech Seed||[still]` + emits `|-fail|<user>` (IDENTICAL
  to the already-seeded fail form). The port's (4b) sub-block arm (`turn.rs`) emitted a bare `|move|`; now it applies
  `attr_last_move_still` + `-fail`. Scenario `leechseed_into_substitute` (Suicune-subs + Venusaur-Leech-Seed).
- **F2 — a MISSED `onTryHit`-class ability immunity emits `[miss]`+`-miss`, not `-immune`.** gen3 rolls accuracy
  BEFORE the TryHit event, so a MISSED Fire move into Flash Fire (or Water/Electric into Water/Volt Absorb) shows
  `|move|…|[miss]` + `|-miss|`. The immune-line emit is now gated on `acc_hit` for the TryHit-class abilities (FF /
  Water&Volt Absorb) ONLY — Levitate + type-chart 0× resolve at the PRE-accuracy `runImmunity` and ALWAYS report
  `-immune` even on a would-be miss (probe `probe_levitate_miss.js`: 40/40 immune). The absorb HEAL / FF arm were
  already `acc_hit`-gated, so this is EMISSION-only. Scenarios `flashfire_tryhit_miss` / `waterabsorb_tryhit_miss`.
- **F3 — a LANDED Water/Volt Absorb `-immune` carries `[from] ability: <Name>`.** The sim emits
  `|-immune|<t>|[from] ability: Water Absorb` (resp. Volt Absorb); the port emitted a plain `-immune`. Folded into
  the F2 scenarios' landed arm via `immune_from_ability`.
- **F4 — `write_line` choice-REVISION semantics = DOCUMENTED-not-fixed.** The sim's `side.choose` CLEARS + re-parses
  on every pre-commit `>pN` write (LAST-write-wins — probe `probe_f4_choice_revision.js`: `>p1 move 1` then
  `>p1 move 2` executes Earthquake, seed-identical to a single `>p1 move 2`; a 1→2→3 chain executes Rest). The port's
  replay-from-genesis accumulator is FIRST-accepted-wins. Documenting was the lower-risk correct option: the
  accumulator has no open-boundary marker, so a same-side overwrite can't be distinguished from a same side in an
  already-committed prior decision — an overwrite rule fired on the latter DESTABILIZES the writeline gate (a
  forced-replacement `>p2 switch N` dropped the replacement chunk — verified). The real bridge sends exactly ONE
  choice per request, so a revised `>pN` is unreachable in production. Documented in `battle.rs` write_line +
  CLAUDE.md.
- **F5 — stale doc.** `event.rs`'s mid-battle Trace copy comment said the `-ability` line "is left un-emitted — an
  honest level-2 gap"; it IS emitted (`turn.rs::emit_ability_start_lines::ability_traced`, byte-verified vs the
  `trace_switchin` capture). Comment corrected.

**Gate:** `protocol_test.rs` **114 → 132 battles / 16115 → 19348 lines byte-equal**, 19 → 22 scenarios (the 114
pre-review scenarios stay a byte-identical golden PREFIX). Probes: `probe_f1_f2_f3_lines.js` /
`probe_f2_ff_armed_miss.js` / `probe_levitate_miss.js` / `probe_f4_choice_revision.js`. Handler-audit gate green
(no disposition/anchor change — the absorb-heal + FF-arm anchors are unchanged). e2e md5 UNCHANGED; full suite green.

## ✅ BUILT + e2e-ADMITTED — BATCH-4: the FINAL mechanics tail (`gen3_ability_batch4_v1`, 2026-07-08, 712 → **719/719 — the ENTIRE pool filter-clean**) — Truant / Inner Focus / Shadow Tag / Cute Charm+attract / Color Change / King's Rock / Focus Band

The last seven members, every draw model PROBE-settled against the resolved `Dex.mod('gen3')`
(`probe_{truant,truant_edges,innerfocus,shadowtag,cutecharm_attract,colorchange,kingsrock,
kingsrock_order,focusband,focusband_confusion}_rng.js` + the shared `probe_batch4_lib.js`):
- **TRUANT** (`MonState::truant_turn`): onBeforeMove priority 9 (slp/frz 10 > truant 9 > flinch 8)
  cants iff the flag — DRAW-FREE (a loaf turn draws NOTHING: no para roll [Q2b], no PP; an asleep
  holder's sleep counter still decrements first). `onSwitchIn` arms `turn !== 0`; the order-27
  residual TOGGLES — so a MID-turn entrant (pivot / drag / action-faint replacement, Q3/Q3b/E2) is
  toggled back the same turn and MOVES its first full turn, while a POST-residual DoT-KO replacement
  (edge E1) keeps `true` and LOAFS. A speed-tied Truant MIRROR adds exactly ONE order-27 residual
  tie-shuffle draw (Q4: 9 shuffles vs the control's 8).
- **INNER FOCUS**: block at the volatile APPLY — the flinch secondary's `random(100)` STILL DRAWS
  (bit-identical draw stream to a Thick Fat control, cant=0 vs the control's flinch; CONTRAST Shield
  Dust, which FILTERS the secondary so the roll never draws). One gate in the flinch apply — covers a
  move's own flinch AND the King's Rock appended one.
- **SHADOW TAG**: traps the foe UNCONDITIONALLY (a Flying Skarmory / a Levitate Gengar IS trapped —
  no grounded/type gate; a MIRROR is MUTUALLY trapped — `onFoeTrapPokemon` has no fellow-holder
  exemption, only the display-only `onFoeMaybeTrapPokemon` skips ST holders) and adds ZERO draws
  (a Wobbuffet mirror's per-turn draw count == a no-trap control's — vs Magnet Pull's onAny* +2).
- **CUTE CHARM + the ATTRACT volatile** (`AbilityData::contact_attract`, `MonState::{gender,attract}`):
  the CC `randomChance(1,3)` draws UNCONDITIONALLY on a damaging CONTACT hit — the GENDER gate lives
  INSIDE `attract.onStart` (an F-into-F / genderless attacker still DRAWS the roll; the volatile fails
  draw-free). Attract: onBeforeMove priority 2 (confusion 3 > attract 2 > par 1), `-activate` emitted
  ALWAYS then `randomChance(1,2)` → cant on a pass; NO duration; cleared when the SOURCE leaves the
  field (`onUpdate`, `-end |Attract|[silent]`) or the HOLDER switches out; it sticks even on a subbed
  attacker; Oblivious blocks draw-free. **GENDER**: parsed from the packed set ('M'/'F'/'N'); an
  UNSPECIFIED gender on a ratio species makes the sim DRAW `battle.sample(['M','F'])` at Pokemon
  construction — an init draw the port does not model — so the attract gender-compare PANICS fail-loud
  on an unknown gender and every golden/probe pins genders explicitly (probe-verified: the post-start
  seed differs when gender is omitted).
- **COLOR CHANGE** (`MonState::types_override` + the ONE `mon_types` choke point — every live type
  read now honors the override: STAB / chart / status type-immunity [Toxic fails on a Poison-overridden
  Kecleon] / sand-chip immunity [a Rock-overridden Kecleon takes no chip] / Magnet Pull's Steel gate /
  Leech Seed's Grass gate): onDamagingHit sets `[move.type]` — DRAW-FREE; **NOT behind a Substitute**
  (the mon's DamagingHit never fires — the batch-2 behind-sub lesson probed EXPLICITLY: a TBolt into
  the sub leaves Kecleon Normal; the naive first read of the probe was wrong because turn 1's hit
  landed BEFORE the sub went up); not on the KO hit; never for typeless `???` (Struggle / the
  confusion self-hit); a repeat type is a no-op; switch-out REVERTS.
- **KING'S ROCK** (`ItemData::flinch_secondary` — the chance + the EXECUTION-DERIVED 130-id move list
  in `gen3_items.json`, `--check`-gated; the port canonicalizes typed-HP ids to the bare sim id): an
  ORDINARY appended TRAILING secondary — order **[move's own secondary] → [KR] → [foe's contact
  proc]** (probe O1/O2/O3); Serene Grace DOUBLES to 20 (the probed 10/15-land vs 31+-miss split);
  Shield Dust FILTERS (no draw); behind a sub it DRAWS but does not apply; a fixed-damage listed move
  (Seismic Toss) and Struggle both proc; a flinch onto an already-moved foe is inert (cleared turn-top).
- **FOCUS BAND** (`ItemData::survive_lethal` → `turn.rs::focus_band_damage`): the `randomChance(1,10)`
  draws FIRST — on EVERY Damage event into the holder (move hits, burn/psn/tox chips, sand/hail chips,
  the leech drain, Spikes, Struggle/Rough-Skin recoil, and the CONFUSION SELF-HIT [effectType 'Move' →
  it CAN be survived; probe `probe_focusband_confusion_rng.js`]) — while the survive-at-1-HP fires only
  when the roll passed AND the damage is lethal AND the effect is a MOVE (a lethal chip still faints;
  the Explosion self-KO is a faint, not a Damage event — no draw for the exploder). NOT drawn for a
  sub-absorbed hit (the mon's Damage event never runs).

**Validation**: the class-sweep golden `gen_ability_batch4_golden.js` → `tests/ability_batch4_test.rs`
(21 scenarios × 60 seeds = 1260 game-end battles, 4220 per-decision STATE+HP+STATUS+**TRAPPED**+SEED
rows + 1020 coverage rows, positive floors per member + THREE zero-cover controls [Inner Focus never
flinch-cants / F-into-F never attracts / no-item never flinches], byte-reproducible, md5-pinned) +
the **B4-1..B4-7 revert-verified pins** in `regression_test.rs` (each fails on its member's revert,
restores green — ground truth `probe_batch4_regression_rng.js`). The batch surfaced **NO new engine
bug** — the golden passed bit-for-bit first-try. Two HARNESS lessons (not engine bugs):
1. **first-mover on a voluntary-switch turn** — the sim's action queue runs a voluntary SWITCH before
   every move, so the golden's first-actor scan must count `|switch|` lines (earlier batch goldens
   never voluntarily switched at a move request, so the gap was latent).
2. **the sim ACCEPTS a 0-PP chosen move and `|cant|…|nopp|`s it at EXECUTION** (surfaced by a
   Pressure ×2-PP-deduction scenario that exhausted Body Slam's 24 PP): the port's strict
   request-legality gate deliberately REJECTS such a script — the e2e's request chooser never submits
   one, so this is a scripted-golden-only path; goldens must stay within request-legal choices.

**E2E ADMISSION**: `truant`/`innerfocus`/`shadowtag`/`cutecharm`/`colorchange` → MODELED_ABILITIES +
`kingsrock`/`focusband` → MODELED_ITEMS (`TRACE_COPYABLE` in lockstep). The deliberate regen at the
committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220) grew the filter-clean pool
**712 → 719/719 — the ENTIRE real-gen3ou team pool** (`truant`=4 + `innerfocus`=2 were the LAST
team-carry gaps), a CLEAN **STRICT** pass first-try (`filtered_diverged == 0` over 220 battles /
10636 decisions — 218 wins + 2 ties; the Truant teams are IN the corpus, the 2 Inner-Focus teams
are clean but weren't drawn into the 220 sample), **byte-reproducible ×2** (md5
`a23d77ac60d4af168b8a4428f0b465c9`). The honest taxonomy's 300-battle UNFILTERED sweep is now
**300/300 clean with an EMPTY ability+item gap list** — a first.

**STILL DEFERRED (the ONE remaining member): FORECAST** — a Castform forme+TYPE change under
rain/sun/hail (0 sample teams). `probe_forecast_rng.js` settled the weather→forme map (sun→Fire /
rain→Water / hail→Ice / sand→Normal), draw-freeness, revert-on-weather-end, bench-revert +
switch-in re-forme; the forme-change REPORTING surface (species id at decision boundaries) + the
Cloud-Nine effective-weather composition stay unprobed → deferred honestly, the e2e filter keeps
every Castform-Forecast team off the modeled path.

## ✅ BUILT — BATCH-3: the BERRY item classes + TRACE + SHED SKIN (`gen3_berry_trace_shedskin_v1`, 2026-07-07; e2e-ADMITTED 2026-07-08, 585 → 712/719)

ONE eatItem consumption mechanism (`MonState::item` — the CURRENT item; eaten → NONE permanently, no
switch-out revert) + 22 DATA-DRIVEN `berryEffect` rows in `gen3_items.json` (extractor curated table +
`dump_gen3_mechanics.js --check` derivation gate, obs-neutral, Python parity green), all PROBE-settled
(`probe_berry_rng.js` / `probe_berry_sub_tie_rng.js` / `probe_trace_shedskin_rng.js` — the sim is the only
oracle):
- **CURE (7)** — cheri/chesto/pecha(psn+tox)/rawst/aspear/persim(confusion)/lum: eats at the FIRST
  `eachEvent('Update')` after the condition (BEFORE the holder's own move — no full-para roll that turn;
  `turn.rs::run_update_items` at every Update site, draw-free). **LUM eats IMMEDIATELY inside setStatus**
  (onAfterSetStatus −1, AFTER a Synchronize reflect — probe line order pinned), incl. Rest's self-sleep
  (LumRest full-heal-awake, probe-verified).
- **HEAL (7)** — oran +10 / sitrus +30 / figy,wiki,mago,aguav,iapapa **floor(maxhp/8)** (the RESOLVED gen3
  amount, NOT the base .ts /3 — the mod-chain law) + a nature-gated confusion (`getNature().minus` ==
  flavor stat → addVolatile drawing random(2,6)): the RESIDUAL order 10 **subOrder 4 — the Leftovers
  slot**, gathered in the item position whenever HELD (probe (B): a berry-vs-Leftovers equal-speed mirror
  draws the IDENTICAL shuffle sequence), threshold `2*hp <= maxhp` EXACT.
- **PINCH (7)** — liechi/ganlon/salac/petaya/apicot +1 stage; **starf** ONE `sample` over the non-capped
  [atk,def,spa,spd,spe] (draws even for n=1) then +2; **lansat** the focusenergy volatile (crit stage +2,
  `MonState::focus_energy`, cleared on switch-out); threshold `4*hp <= maxhp` EXACT. Sub-absorbed hits
  don't trigger (hp untouched); a KO'd holder never eats; a second crossing after the eat is inert.
- **PP (1)** — leppa at the Update when a slot hits 0 PP: `min(pp+10, maxpp)` on the first 0-PP slot.
- **TRACE** — the gen3-RESOLVED onStart (base/gen4 seek/notrace machinery REPLACED): a MID-BATTLE
  switch-in draws the n=1 `randomFoe` sample and copies the foe's CURRENT ability with NO guard (No
  Ability/Wonder Guard/traced-through all copy); gen3 `setAbility` does NOT fire the copied onStart
  (`gen > 3` gate — no traced Intimidate/Drought activation); the copy is LIVE (`MonState::ability`, every
  engine read); switch-out REVERTS (re-entry re-traces). A LEAD trace's draw is a `>start`-window draw
  (pre-dates the seeded start → the port applies the copy draw-free there). FAIL-LOUD on an unmodeled copy
  (`event.rs::TRACE_COPYABLE` — MODELED∪NOOP mirror; the e2e both-teams filter makes it unreachable).
- **SHED SKIN** — residual order 10 **subOrder 3** (the Speed Boost/Rain Dish slot, gathered
  unconditionally — it ties the residual shuffle even unstatused): while STATUSED, ONE
  `randomChance(33,100)`; a pass cures BEFORE the same-mon DoT (subOrder 6) — a cure turn takes NO chip;
  confusion NOT cured; unstatused → NO draw.

**Validated:** `gen_berry_batch3_golden.js` → `tests/berry_batch3_test.rs` — 32 scenarios × 40 seeds =
1280 battles (mostly game-end; the 2 rest-loop scenarios capped), 6784 per-decision
STATE+HP+STATUS+**ITEM**+BOOSTS+SEED rows (13568 item + 12307 status/boost assertions), every wired member
covered (1005 covered rows; wrong-status / unstatused / lead-trace zero-cover controls enforced at 0),
byte-reproducible (stable md5). **BR1-BR6** revert-verified pins in `regression_test.rs` (ground truth
`probe_berry_batch3_regression_rng.js` + `probe_berry_threshold_boundary.js`): BR1 sitrus threshold + the
Leftovers-slot draw-identity twin, BR2
lum immediate cure-before-move (draw-free; the control's stuck para shifts the stream), BR3 starf's sample
draw + +2, BR4 shed skin's per-statused-residual roll vs a no-op control, BR5 trace's n=1 draw + live copy,
**BR6 the `<=`-vs-`<` threshold BOUNDARY** (the closed reviewer finding: the prior probes' odd-maxhp boards
made exact equality unreachable, so a `<=` → `<` mutation passed every golden + pin;
`probe_berry_threshold_boundary.js` constructs an EVEN-maxhp Vaporeon [400] landing EXACTLY on
hp == maxhp/2 / maxhp/4 — **the sim EATS at equality** [one-HP-above does not], so the engine's `<=` was
CORRECT; BR6 pins both boundaries via a Seismic-Toss grind and FAILS under the `<` mutation —
mutate-verified, then restored).
Full suite 252 green; pre-existing goldens byte-identical.

**e2e ADMITTED (2026-07-08):** the 22 `berryEffect` berries → `MODELED_ITEMS` + `trace`/`shedskin` →
`MODELED_ABILITIES` (the both-teams-all-modeled Trace safety; `TRACE_COPYABLE` in lockstep) + the ONE
deliberate regen (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220): filter-clean pool **585 → 712 / 719**
(the biggest admission since Natural Cure — lumberry=64 + salacberry=46 + trace=9 the levers), STRICT
`filtered_diverged == 0` over 220 battles / 11054 decisions FIRST-TRY (**NO new engine bug** — the eat is
draw-free at the Leftovers slot, the new draws [Starf sample / Figy confusion / Trace n=1 / Shed Skin
randomChance] all seed-faithful), **leech finally exercised** (354 LEECH-MOVE decisions, was 0 — the leech
users carried berries), byte-reproducible (regen ×2, identical md5). Taxonomy after: 294/300 unfiltered
battles clean; ability gaps `truant`=4 > `innerfocus`=2 only; **item gap list EMPTY**. Batch-4 deferred
list: Truant, Inner Focus, Cute Charm (attract volatile), Color Change (type-override), King's Rock /
Focus Band PROC_ITEMs,
CRIT_ITEMs (Scope Lens family), Shell Bell, White Herb, Mental Herb, Macho Brace, Mail, Shadow Tag,
Forecast; the gen-2 berry twins stay unmodeled (gen3ou-unobtainable).

Edge cases to model + **pin with a named regression test** when implemented (see the
`tests/regression_test.rs` practice + CLAUDE.md "Regression tests"). Not yet built.

> **User-noted 2026-06-28** (the first batch below). The common thread of most of these is
> **TRAPPING + choice/request VALIDATION**, which the e2e fuzz currently *sidesteps* — the fuzz
> only ever submits choices the **sim's request** already deems legal, so a trapped-mon switch is
> never even offered. A standalone drop-in port must compute legality itself (`LegalActions` must
> exclude switches when trapped) and must handle the **server rejecting an illegal choice +
> re-requesting** (the request/protocol layer, not yet built).

## ✅ BUILT — the ABILITY BATCH-2 DRAW-BEARING "reactive" classes + block tail (`gen3_ability_batch2_v1`, 2026-07-07)

The draw-bearing ability procs batch-1 deferred, each PROBE-settled (the sim is the only oracle) + validated
by the class-sweep golden `gen_ability_batch2_golden.js` → `tests/ability_batch2_test.rs` (960 game-end
battles, per-decision STATE+HP+STATUS+SEED, 3250 seed + 5540 status assertions, byte-for-byte) + the
**B2-1..B2-7** revert-verified pins in `regression_test.rs` (ground truth
`harness/probe_ability_batch2_regression_rng.js`).

- **CONTACT_PROC** (Static par / Poison Point psn / Flame Body brn / Effect Spore slp|par|psn) — a DATA-DRIVEN
  `onDamagingHit` (`AbilityData.contact_proc`): when the HOLDER is hit by a **CONTACT** move that dealt damage,
  it draws `randomChance(chance)` and (on a pass) statuses the ATTACKER. **THE DRAW-MODEL CRUX** (probe
  `probe_contact_proc_{rng,lands}.js`): the proc's `randomChance` draws INSIDE `runEvent('DamagingHit')` (gen<5,
  battle-actions.ts:982) which the sim fires **AFTER** the move's OWN `secondaries()` (line 957) — so the draw
  order is `[move secondary random(100)]` THEN `[contact-proc randomChance]`. Static/PP/FB = one
  `randomChance(1,3)`; **Effect Spore** = `randomChance(1,10)` then, on a pass, ONE `sample(["slp","par","psn"])`
  (a `random(3)`) — the nested draw (`probe_effectspore_sample.js`). It draws behind a Substitute + on a KO.
  Wired in `turn.rs::apply_contact_proc` (called from `run_move`'s landed-hit tail). Pins B2-1 (Static) + B2-2
  (Effect Spore's nested sample).
- **CONTACT recoil** (Rough Skin) — DRAW-FREE `baseMaxhp/16` recoil to the attacker on a contact hit. Pin B2-3.
- **BLOCK**: **Damp** cancels Explosion / Self-Destruct at `runEvent('TryMove')` (BEFORE the self-KO faint + the
  accuracy roll) — the user does NOT self-KO, the move draws NOTHING; **Soundproof** is immune to a SOUND move
  (Sing / Grass Whistle / Roar — accuracy drawn, then `-immune`, no status / no drag / no sample); **Suction
  Cups** blocks a phaze DRAG (the sim's `forceSwitch` runs `runEvent('DragOut')` in the MOVE BODY → `onDragOut`
  returns null → NO `forceSwitchFlag` → NO `dragIn` → NO `sample`; `-activate Suction Cups`, the holder STAYS).
  Probe `probe_block_abilities_rng.js`; pins B2-4 (Damp) + B2-5 (Soundproof) + B2-6 (Suction Cups).
- **SYNCHRONIZE** — reflect a foe-inflicted major status back to the SOURCE (slp/frz EXEMPT; tox→psn), wired at
  the single status choke point `turn.rs::try_set_status` (source-threaded). DRAW-FREE in gen3customgame (the
  e2e format); draws the reflected status's own clause shuffle in gen3ou (`probe_synchronize_rng.js`). Pin B2-7.

**DATA**: the CONTACT_PROC params + `contactRecoil`/`blocksSound`/`blocksExplosion`/`blocksPhazeDrag`/
`synchronize` are extracted into `gen3_abilities.json` (drift-gated by `dump_gen3_mechanics.js --check`); the
`contact` + `sound` move flags into `gen3_moves.json`. Obs-neutral (facade ignores them; extractor-parity
green). **e2e admission** grew the filter-clean pool **571 → 585 / 719** (synchronize [the #1 taxonomy gap] +
effectspore the levers); STRICT `filtered_diverged == 0` over 220 battles / 11790 decisions, byte-reproducible
at the committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220) — a CLEAN pass first-try, **NO new engine
bug** (the CONTACT_PROC draw-after-secondary + the BLOCK draw-count drops + the draw-free Synchronize reflect
composed cleanly). **DEFERRED to batch 3 (NOT admitted): Cute Charm** (its attract/infatuation VOLATILE + the
Attract move are unmodeled — the port has no attract volatile → can't represent it; the same `randomChance(1,3)`
but `addVolatile('attract')`, so it carries NO `contactProc` and stays off the filtered path), **Color Change**
(an ON_HIT type-OVERRIDE — 0 sample teams [Kecleon-only] + needs a `species_types` type-override thread, so
deferred to avoid the refactor + destabilization risk), **Synchronize's gen3ou clause-shuffle** interaction is
modeled but only exercised in a clause format (the e2e is customgame), **Trace** (an onStart randomFoe
ability-copy — a distinct switch-in draw), **Shed Skin** (a residual `randomChance(33,100)` status cure), and
**berries** (an item `eatItem` mechanism). ["static", "poison point", "flame body", "effect spore", "rough skin", "soundproof", "damp", "suction cups", "synchronize"]

## ✅ FIXED — sun/rain END-OF-TURN `eachEvent('Weather')` shuffle gated on Sand|Hail only (`gen3_ability_batch1_v1` STEP-1, 2026-07-07)

**THE BUG.** gen3 sun (`sunnyday`) + rain (`raindance`) fire `this.eachEvent('Weather')` at EVERY
end-of-turn **UNCONDITIONALLY** — the resolved `Dex.mod('gen3')` `onFieldResidual` body is a bare
`this.add('-weather',…,'[upkeep]'); this.eachEvent('Weather');` (NO `isWeather` guard). `eachEvent('Weather')`
speed-sorts the actives → on a speed **TIE** it draws ONE `random(0,2)` Fisher-Yates shuffle. The port
gated the end-of-turn weather tie-shuffle on `Sand | Hail` ONLY (the chip weathers) → a WEATHER-TURN
speed TIE under **sun/rain MISSED that shuffle draw** → a 1-draw desync on every later turn. Re-confirmed
vs the resolved dist by `harness/probe_weather_eachevent_sunrain.js` (rain/sun/sand all draw +1 on a tie;
distinct speed draws 0).

**THE SUBTLE SPLIT (probe-caught during the fix — the `isWeather` guard is NOT redundant).** sand/hail
wrap the call in `if (this.field.isWeather('<w>')) this.eachEvent('Weather')`, and `isWeather` reads
`effectiveWeather()` — so a **Cloud Nine / Air Lock** (WEATHER_NEGATE) mon SUPPRESSES the sand/hail
shuffle. Sun/rain have NO such guard → their shuffle fires **even under a negater**. VERIFIED full matrix
(Cloud-Nine Δdraw on a tie): rain +1, sun +1, sand 0, hail 0. My first fix (schedule off
`effective_weather()` for all) was WRONG for sun/rain-under-negater and the probe
`_tmp_negate.js`/`_tmp_matrix.js` caught it.

**THE FIX** (`turn.rs::run_residuals`): schedule the field weather-residual (which fires the shuffle in
`apply_weather_chip`) off the **RAW `field.weather`** for sun/rain (fires even under a negater), and off
`effective_weather()` for sand/hail (a negater suppresses those). `apply_weather_chip` ALWAYS fires the
shuffle then chips only the non-immune actives (Rain/Sun mons are all `weather_immune` → no chip), and
early-returns for Rain/Sun after the shuffle. SEED suites stayed BYTE-IDENTICAL (no pre-existing scenario
has a sun/rain weather-turn tie — the `battle_test` rain scenario is Kyogre-vs-Snorlax distinct-speed).
Pinned by the revert-verified `regression_test.rs::sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed`
(a Kyogre-Drizzle mirror, exact tie under rain; ground truth `harness/probe_weather_eachevent_tie_regression_rng.js`).

## ✅ BUILT — the ABILITY BATCH-1 draw-free / structural classes (`gen3_ability_batch1_v1`, 2026-07-07)

Four DRAW-FREE / STRUCTURAL ability classes wired + validated by the class-sweep golden
`gen_ability_batch1_golden.js` → `tests/ability_batch1_test.rs` (300 game-end battles, per-decision
STATE+HP+SPE-BOOST+SEED, byte-for-byte; 941 seed + 1582 spe-boost assertions) + the **B1-B4b** revert-verified
pins in `regression_test.rs`:
- **CRIT_IMMUNE** (Shell Armor / Battle Armor) — a hit into the holder NEVER crits (the crit roll is DRAWN
  then overridden false via `runEvent('CriticalHit')`; DRAW-FREE — `probe_critimmune_rng.js`). B1:
  `battle_armor_prevents_the_crit_but_draws_the_roll` (a seed where the control CRITS but Battle Armor
  doesn't, at the IDENTICAL post-turn seed).
- **WEATHER_SPEED** (Chlorophyll / Swift Swim) — ×2 effective speed in sun / rain, folded into the CACHED
  speed the tie-shuffles read (`effective_speed`). B2: `chlorophyll_speed_doubles_and_flips_the_first_mover_in_sun`
  (a slow Bellossom, ×2 = 272, OUTSPEEDS Groudon 216 in Drought-set sun — the first-mover flip; a no-op control does not).
- **WEATHER_NEGATE** (Cloud Nine / Air Lock) — suppresses the weather's EFFECTS (`effective_weather()`
  returns None: no chip, no speed ×2). B3: `cloud_nine_suppresses_the_sandstorm_chip` (a Psyduck takes NO
  sand chip; a Damp control takes maxhp/16).
- **RESIDUAL** (Speed Boost / Rain Dish) — at residualOrder 10 subOrder 3, DRAW-FREE: Speed Boost +1 spe/
  active-turn (activeTurns-gated), Rain Dish +maxhp/16 in rain. B4 `speed_boost_raises_the_spe_stage_by_one_each_active_turn`
  + B4b `rain_dish_heals_each_end_of_turn_in_rain`.

**e2e admission (STEP 3).** `shellarmor`/`battlearmor`/`chlorophyll`/`swiftswim`/`cloudnine`/`airlock`/
`speedboost`/`raindish` added to `MODELED_ABILITIES`; the class-(a) NO-OPS `plus`/`minus`/`lightningrod`/
`stickyhold` added to `NOOP_ABILITIES` (each PROVE-verified a true no-op vs an Insomnia control in the
modeled universe — `harness/probe_ability_batch1_noop_verify.js`). The filter-clean pool grew **525 → 571
/ 719** (shellarmor the big lever); STRICT `filtered_diverged == 0` over 220 battles / 11630 decisions,
byte-reproducible at the committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220) — a CLEAN strict
pass first-try, **NO new engine bug** (the STEP-1 weather-eachEvent fix + the DRAW-FREE class model composed
cleanly). **FORECAST is DEFERRED (NOT a no-op)** — its `onWeatherChange` changes Castform's forme + TYPE in
rain/sun/hail (the probe DIVERGES under those weathers: `Castform-Rainy`/Water etc.), so a filter-clean
Castform under weather would desync. Left OUT for batch 2 (a forme-change model). **DEFERRED to batch 2 (the
DRAW-BEARING procs):** static/poisonpoint/flamebody/cutecharm/effectspore/synchronize/shedskin/trace/
shadowtag/roughskin/colorchange (each a NEW `random` roll — a draw-order probe first). **→ BATCH 2 is now
DONE** (`gen3_ability_batch2_v1`, the section at the top of this file): static/poisonpoint/flamebody/
effectspore/roughskin/synchronize + the BLOCK tail (soundproof/damp/suctioncups) WIRED; cutecharm (attract
unmodeled), colorchange (type-override, 0 teams), shedskin/trace/shadowtag DEFERRED to batch 3.

## ✅ FIXED — STATUS_IMMUNE ability class (the #2 e2e team-carry gap; `gen3_status_immune_v1`, 2026-07-06)
The gen-3 abilities that grant immunity to a specific MAJOR status — the DATA-DRIVEN completion of the
status-immunity class. **`immunity` (=97 teams) was the #2 team-carry gap** after Natural Cure. The engine
already BLOCKED these in `try_set_status` (hardcoded), but they were EXCLUDED from the e2e
`MODELED_ABILITIES` (and `insomnia`/`vitalspirit` were wrongly parked in `NOOP_ABILITIES`), and a
clause-format target tripped a FAIL-LOUD panic. Now the class is DATA-DRIVEN + ADMITTED to the e2e.

**CLASS MEMBERSHIP + each member's immune status(es)** (enumerated from the RESOLVED `Dex.mod('gen3')` via
`harness/probe_statusimmune_enumerate.js` — num ≤ 76 = gen-3-legal):
| Ability | num | Immune status(es) | Handler | Blocks at |
|---|---|---|---|---|
| **Limber** | 7 | par | `onSetStatus` | the SetStatus event |
| **Insomnia** | 15 | slp | `onSetStatus` | the SetStatus event |
| **Immunity** | 17 | psn, tox | `onSetStatus` | the SetStatus event |
| **Magma Armor** | 40 | frz | `onImmunity` | `runStatusImmunity` (BEFORE the event) |
| **Water Veil** | 41 | brn | `onSetStatus` | the SetStatus event |
| **Vital Spirit** | 72 | slp | `onSetStatus` | the SetStatus event |

Plus **Own Tempo** (confusion — via `onTryAddVolatile`, ALREADY modeled in the confusion arm) and
**Oblivious** (attract — via `onImmunity('attract')`, N/A in the modeled move set). Both block a VOLATILE,
not a major status, so they are NOT STATUS_IMMUNE members. **Leaf Guard is num 102 → NOT gen-3** (the old
hardcode wrongly listed it; a gen-3 team can never carry it).

**THE PROBE-SETTLED DRAW MODEL (the crux — `harness/probe_statusimmune_{rng,setstatus_event,shuffle_size,
magmaarmor}.js` over the resolved dist, instrumenting the sim's PRNG + `findEventHandlers`):**
- **gen3customgame (the e2e format) — EVERY member is DRAW-FREE.** An `onSetStatus`-phase ability makes the
  SetStatus event's ONLY handler (size-1 → NO tie → NO shuffle); Magma Armor blocks at `runStatusImmunity`
  BEFORE the event. So a blocked status is draw-identical to a normal status apply — **admission is
  SEED-CLEAN** (the pre-existing suites stay byte-identical).
- **gen3ou — the SURPRISE that KILLED the old "size-3 shuffle" panic.** An `onSetStatus`-phase ability adds a
  3rd SetStatus handler, but `speedSort` sorts `order→priority→speed→subOrder`, and the ability handler
  carries a DEFINED `speed` (the mon's, e.g. 96) while the 2 clause handlers have `speed=undefined` → the
  ability sorts into its OWN group (index 0, no tie), leaving the **2 clauses a SIZE-2 tie** → `shuffle(list,
  1, 3)` draws EXACTLY ONE `random` — IDENTICAL to the control's size-2 `shuffle(list, 0, 2)`. So the draw
  COUNT is UNCHANGED (the old fail-loud panic claiming a size-3 shuffle was WRONG — refuted by
  `harness/probe_statusimmune_shuffle_size.js`, which shows `size:2, draws:1` in ALL cases). Magma Armor
  blocks BEFORE the event, so its clause shuffle NEVER fires (it's the sun-freeze gate's position).

**DATA-DRIVEN WIRING (the `gen3_item_mechanics_v1` / `gen3_accuracy_pipeline_v1` precedent):** the extractor
emits a `statusImmune: {statuses, phase}` field per member into `gen3_abilities.json` from the curated
`_GEN3_ABILITY_MECHANICS` table, drift-gated by `dump_gen3_mechanics.js --check` (which DERIVES the same from
the resolved `onSetStatus`/`onImmunity` handlers). Obs-neutral (the Python `agents.gen3_data` facade reads
only `num`/`name`; extractor-parity + obs-parity green). The Rust dex parses it into
`AbilityData.status_immune` (`StatusImmune {statuses, phase: SetStatus|Immunity}`); `try_set_status` reads
the data field: the `Immunity`-phase block gates BEFORE `set_status_event_shuffle` (Magma Armor), the
`SetStatus`-phase block AFTER it. The old hardcoded `status_ability_immune` + `ability_has_on_set_status`
match-arms are REMOVED; the FAIL-LOUD now fires only on a genuinely UNMODELED `onSetStatus` ability
(`ability_unmodeled_on_set_status`).

**VALIDATION:** the class-sweep golden `harness/gen_statusimmune_golden.js` → `tests/statusimmune_test.rs`
(12 scenarios × 40 seeds = 480 decisive battles to game-end; the block is observable on the ACTIVE-status
timeline — an immune Snorlax is status-moved every turn + STAYS `-`, vs a non-immune control that gets
STATUSED and diverges; 6 immune members + 4 status controls + a frz-STATE discriminator + 2 wrong-status
controls; a `-immune` block-marker floor for the 5 onSetStatus members + the STATE proof for Magma Armor's
SILENT secondary-freeze block; byte-reproducible stable md5). 4 revert-verified `regression_test.rs` pins
(`limber_blocks_paralysis_draw_free`, `insomnia_blocks_sleep_draw_free` [the draw-COUNT pin — a landed sleep
draws `random(2,6)`], `magma_armor_blocks_freeze` [+ the frozen no-ability control discriminator],
`immunity_blocks_tox_but_not_burn` [status-specificity]; ground truth
`harness/probe_statusimmune_regression_rng.js`). **e2e ADMISSION:** adding the 6 members to
`MODELED_ABILITIES` (+ moving `insomnia`/`vitalspirit` OUT of `NOOP_ABILITIES` — they genuinely block sleep)
grew the filter-clean pool **449 → 525 / 719** (+76 teams — the immunity=97 gap); the regenerated golden is
a STRICT `filtered_diverged == 0` pass at the committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220,
220/220 clean, 11651 decisions, 219 wins + 1 tie), byte-reproducible (identical md5 across two independent
regens). **`immunity` DROPS OFF the taxonomy's top-gaps list** (now `shellarmor`=39 > `synchronize` >
`effectspore` > `trace`; items `lumberry`=64 > `salacberry`).

**ONE ENGINE BUG the enlarged 525-clean corpus surfaced + FIXED (a REAL bug, not the STATUS_IMMUNE class) —
the EMPTY NATURE.** e2e_8/e2e_73 (STATUS_IMMUNE-team battles admitted by the regen) carry a **Suicune with an
OMITTED nature field** (`Suicune||Item|Ability|moves||EVs` — no nature token). The port's `compute_stats`
PANICKED `unknown nature ""` at start; the sim treats an empty nature as **NEUTRAL** (Serious — a nonexistent
nature contributes no plus/minus to `spreadModify`; VERIFIED vs the sim: an empty-nature Suicune's
`storedStats` == its Serious stats bit-for-bit). FIX (`stats.rs::compute_stats`): an EMPTY nature id computes
the neutral (all-1.0) multipliers instead of erroring (a NON-empty unknown nature is STILL a hard error — a
genuine typo would corrupt stats). Pinned by the revert-verified stats unit tests
`empty_nature_computes_the_neutral_stats` + `a_nonempty_unknown_nature_still_errors` (and replayed bit-for-bit
by the e2e golden's e2e_8/e2e_73). ["immunity", "status immune ability", "limber", "insomnia", "water veil", "magma armor", "empty nature"]

## ✅ FIXED — FLASH FIRE ×1.5 fire-boost (the deferred FF gap; the A/B fuzzer's #1 STATE cluster; `gen3_flashfire_boost_v1`, 2026-07-06)
The port modeled Flash Fire IMMUNITY (a Fire move deals 0 to an FF holder) but NOT the
post-activation **×1.5 boost** on the holder's OWN Fire moves — a **documented deferred gap** that
was the **evidence-based #1 STATE-divergence driver** in the A/B fuzzer (fireblast + flamethrower
dominate the STATE cluster; 397/402 fire-move STATE repros carry a Flash Fire mon — the sim deals
MORE fire damage than the port). Now MODELED bit-for-bit and it also COMPLETES the type-interaction
ability class (Levitate immunity, Water/Volt Absorb heal+immunity were done; FF's boost was the last
gap → the family is COMPLETE as a class).

**PROBE-SETTLED (the only oracle — `harness/probe_flashfire_rng.js` over the RESOLVED `Dex.mod('gen3')`
flashfire ability, NOT a base-source read):**
- **ACTIVATION** = `flashfire.onTryHit` — fires AFTER the accuracy roll (a **MISSED Fire move does NOT
  arm it**, probe A2), only for a Fire-type move on a non-self target. **DRAW-FREE** (probe A4). The
  holder takes 0 (the existing type-absorb immunity). The `onTryHit` SKIPS a `frz`-status holder (the
  `status === 'frz'` guard) and Will-O-Wisp into a Fire-type/statused/subbed target; **every gen-3 OU
  FF holder IS Fire-type**, so WoW never arms it (probe A3 — a synthetic non-Fire holder DOES arm on a
  landed WoW). Any non-`frz` status is irrelevant (probe A6). PERSISTS across turns; **CLEARED on
  switch-out + faint** (probe A5, `clearVolatile` → `flashfire.onEnd`).
- **THE BOOST** = the flashfire volatile's `onModifyDamagePhase1 chainModify(1.5)` — a **DAMAGE-PHASE
  fold (the SAME phase as Reflect/Light Screen), NOT an `onModifyAtk`/`onModifySpA` stat mod** (those
  handlers are `undefined` in the resolved gen-3 dist — probe B1). **`×1.5` = `chainModify(1.5)`**
  (probe B2). **Category-AGNOSTIC** — applies to BOTH physical AND special Fire moves (probe B4;
  though in gen-3 the type-based phys/spec split makes EVERY Fire move Special anyway). **NOT
  crit-bypassed** (no crit guard on the handler, unlike screens). **DRAW-FREE** (B3).

**IMPLEMENTATION (engine-flag, justified — FF is NOT a `dmgMod` stat/BP fold like the DMG_MOD family;
it is a volatile + a ModifyDamagePhase1 damage fold, structurally like screens):**
`MonState::flash_fire: bool` (set at the `acc_hit`-gated Fire-absorb site in `run_move` via
`apply_flash_fire_activation`; cleared in `execute_switch` + `process_faints`). `DamageContext::flash_fire`
(set by `build_damage_context` = armed attacker AND Fire move), folded in `damage.rs::modify_damage`'s
ModifyDamagePhase1 stage — **ACCUMULATED with any screen into ONE `chain_modify` modifier** (probe-confirmed:
sequential per-mod rounds DIVERGE for ~¼ of baseDamage values, so `FF ×1.5 ⊗ Light Screen ×0.5` must be
one accumulated modifier). Byte-identical when off (a single-mod `chain_modify([1,2])` == the old
`modify(bd,1,2)`, verified 0/20000 divergences). A **confusion self-hit** uses a typeless '???' move → not
Fire → no boost (verified).

**VALIDATION:** the class-sweep golden `harness/gen_flashfire_golden.js` → `tests/flashfire_test.rs`
(3 scenarios × 30 seeds = 90 decisive battles to game-end, 261 per-decision STATE+HP+SEED rows, 187
armed-boosted-hit rows: `ff_special_boost` + a wrong-type control + a not-activated control) + two
calc-level EXACT max-roll pins (`flash_fire_boost_exact_max_roll` = un-boosted 181 vs boosted 270 STAB
Flamethrower base; `flash_fire_light_screen_chain_combine_exact` = the accumulated `FF ⊗ Light Screen`
= 136). 3 revert-verified `regression_test.rs` pins (`flash_fire_boosts_the_holders_own_fire_move`,
`flash_fire_arms_on_a_landed_fire_hit_but_not_on_a_miss`, `flash_fire_clears_on_switch_out`; ground truth
`harness/probe_flashfire_regression_rng.js`). The e2e capstone stays **STRICT 220/220 `filtered_diverged==0`
byte-unchanged** (flashfire was already in `MODELED_ABILITIES` for immunity → the filter-clean pool is
unchanged at 151/719; the committed golden's battles don't exercise the boost path). **Fuzzer parity: of
200 replayed FlashFire-team STATE repros (completed overnight/auto dirs), 185 (92.5%) flip to `ok` with the
fix; a boost-revert re-diverges all 10 spot-checked.** The Light-Screen status MOVE is unmodeled in the port
(fail-loud) so the `FF ⊗ Light Screen` combine is a CALC-level pin, not a full-battle scenario.
["flash fire", "fire absorb boost"]

## ✅ FIXED — NATURAL CURE switch-out status cure (the #1 e2e team-carry gap; `gen3_natural_cure_v1`, 2026-07-06)
The port modeled no SWITCH_OUT ability. **Natural Cure** — the sole gen-3 switch-out-cure ability — was the
**#1 e2e team-carry blocker** (naturalcure=254 of 719 sample teams, on Blissey/Starmie/Celebi/Miltank/…),
so admitting it is the single biggest e2e-admission lever. The holder's MAJOR STATUS is CURED when it
SWITCHES OUT. Now MODELED bit-for-bit; it IS the SWITCH_OUT ability class (its only gen-3 member).

**PROBE-SETTLED (the only oracle — `harness/probe_naturalcure_rng.js` + `probe_naturalcure_dump.js` over
the RESOLVED `Dex.mod('gen3')` naturalcure ability, NOT a base-source read):**
- **THE TRIGGER** = `naturalcure.onSwitchOut` — **`onCheckShow` is `undefined`** in the resolved gen3 dist
  (resolving the long-DEFERRED "NaturalCure CheckShow" draw question the fullbattle section flagged as a
  possible switch-gate draw: there is **NO CheckShow gate**). The `onSwitchOut` body is
  `if (!pokemon.status || pokemon.status==='fnt') return; this.add('-curestatus', ..., '[silent]');
  pokemon.clearStatus();`
- **THE TIMING** = it fires in `switchIn`'s `runEvent('SwitchOut', oldActive)` on an ALIVE outgoing mon,
  BEFORE `clearVolatile()`. It fires on BOTH a VOLUNTARY pivot AND a phaze-**DRAG**-out (probe D4 — only
  `BeforeSwitchOut` is `!isDrag`-gated; `SwitchOut` fires regardless of `isDrag`, so a Roar/Whirlwind that
  drags the NC mon OUT cures it). It is a **NO-OP on a FAINT** (probe D5 — the `status==='fnt'`/empty guard;
  a fainted mon has nothing to cure).
- **THE DRAW MODEL (the crux)** = **DRAW-FREE** (probe D1). The cure + its `[silent]` `-curestatus` reveal
  consume **ZERO PRNG** — the sim's raw draw COUNT on a switch-out is IDENTICAL for (a) a statused NC mon,
  (b) a statused non-NC mon, and (c) an unstatused NC mon (all 6 statuses, voluntary AND drag; the
  post-switch SEED is byte-identical NC vs non-NC). So admitting Natural Cure is **SEED-NEUTRAL** for every
  pre-existing suite (verified: `secondary_golden.txt` md5 unchanged after regen).
- **WHICH STATUSES** = ALL of brn/par/psn/tox/slp/frz (probe D2). Clearing `status` to `None` drops the
  whole `Status` variant, so the **tox stage + sleep counter reset too**.

**IMPLEMENTATION (engine flag, justified — a single-member behavioral class with NO parameter, like the
flashfire/guts/magnetpull string checks; NOT a `dmgMod`/`accMod` data fold):** in `turn.rs::execute_switch`
(where the outgoing mon's volatiles are already cleared) — `if !m.fainted && to_id(&m.set.ability) ==
"naturalcure" { m.status = None; }`. The phaze-drag routes through the SAME `execute_switch` (via
`drag_in`), so the same gate cures a dragged-out NC mon. `gen3_abilities.json` is UNCHANGED (still
`{name,num}` for naturalcure) — obs-neutral, so the Python `agents.gen3_data` facade + the extractor
`--check` drift gate + `extractor_parity_test` are all untouched (the SWITCH_OUT class needs no data field).

**VALIDATION:** the class-sweep golden `harness/gen_naturalcure_golden.js` → `tests/naturalcure_test.rs`
(7 scenarios × 40 seeds = 280 decisive battles to game-end, 1936 per-decision STATE+STATUS+SEED rows, 3518
status assertions, 175 cure rows). The cure is made OBSERVABLE on the ACTIVE-status timeline: an NC mon is
statused IN-ENGINE by the foe, PIVOTS OUT (cured), PIVOTS BACK — and RETURNS UNSTATUSED; a non-NC control
runs the IDENTICAL plan/teams and RETURNS STILL statused, whereupon the tox ramp FAINTS it and the whole
battle DIVERGES (different winner + length + HP), so a modeled cure is the ONLY thing that makes the NC
scenario's timeline differ from the control's. 3 revert-verified `regression_test.rs` pins
(`natural_cure_cures_status_on_voluntary_switch_out` [+ the seed-neutral non-NC control],
`natural_cure_is_a_no_op_on_a_faint` [catches dropping the `!fainted` guard],
`natural_cure_phaze_drag_cures_the_dragged_out_mon`; ground truth
`harness/probe_naturalcure_regression_rng.js`). **e2e ADMISSION — the payoff:** adding `naturalcure` to
`gen_e2e_fuzz.js`'s `MODELED_ABILITIES` grew the filter-clean pool **151 → 449 / 719** (the BIGGEST single
admission lever; naturalcure DROPS OFF the taxonomy's top-gaps list, which now reads
`immunity=97 > lumberry=64 > …`). The regenerated e2e golden (committed knobs MASTER_SEED 0x1234abcd,
FILTERED_TARGET 220) is a **CLEAN STRICT pass — 220/220 bit-for-bit, `filtered_diverged == 0`, 12054
decisions** (203 involve a Natural-Cure carrier's switch boundary) — **NO new engine bug surfaced** (the
cure is a draw-free, well-localized status clear at the switch site, so it composed cleanly with every
existing mechanic). ["natural cure", "switch out cure", "naturalcure"]

## ✅ FIXED — forced-replacement request-boundary RESUME "phantom" = an INVALID move slot after a replacement (found un-deferring the last 2 protocol scenarios; ROOT-CAUSED + FIXED 2026-07-01)
The last two DEFERRED protocol scenarios (`status_para_and_boost_drop` + `secondary_status_flinch`, both
all-Seismic-Toss) are now **ASSERTED byte-exact** — the protocol byte-diff grew from 51 battles / 5630
lines to **63 battles / 7223 lines**, `DEFERRED_SCENARIOS` is EMPTY. The blocker was mischaracterized as
a "phantom zero-draw move-decision the port collapses"; the true cause is simpler and **not an engine
control-flow bug**:

**ROOT CAUSE — an INVALID scripted move slot after a replacement, which the sim REJECTS.** In
`status_para_and_boost_drop` battle 1/2/5, a mid-turn faint replaces a 3-move Tyranitar with a **2-move
Snorlax**; the omniscient capture harness (`gen_protocol_capture.js`), driving from a **stale per-turn
plan** that still uses Tyranitar's move indices, then submits `move 3` for Snorlax. The sim's
`side.choose` **REJECTS** it ("Your Snorlax doesn't have a move 3", `pokemon.ts`), drawing **NOTHING** and
leaving `requestState === 'move'` OPEN — so `commitChoices` never fires, no turn runs, and the real turn
executes on the NEXT (valid) submission. The `try{…}catch` capture loop records that rejected submission
as a **PHANTOM `move` DEC row whose `seedAfter` == the prior boundary's** (probe
`harness/probe_forced_replacement_queue.js`: the rejected `p1.choose("move 3") → false`, no
`commitChoices`, no `turnLoop`, `draws=0`). The port's `run_full_battle` did NOT validate the move slot —
it RAN a full turn for that decision (Snorlax's `move 3` no-op'd via `move_at → None`, but the FOE's
Seismic Toss + the residual + the end-of-turn Quick Claw **drew**), so from that decision onward the
per-decision seed AND the emitted line stream diverged (protocol_test diverged at `status_para_and_boost_drop/1`
filtered line 71: the golden ran another `|move|Body Slam`, the port went to `|`/`upkeep`).

**Zero-draw / observation-only:** the sim's real battle has 7 boundaries, not 8; the phantom consumes NO
PRNG. So the fix is a decision-boundary MAPPING correction, not a seed change — the e2e (13367 seed
assertions, `filtered_diverged 0`), battle (2034), fullbattle (2053), and secondary (4328) seed suites
stay **byte-identical**.

**FIX** (`turn.rs::run_full_battle` + `move_decision_is_legal`): at the top of each turn the driver now
VALIDATES a `move` decision — if any choosing side's `Move(K)` slot exceeds its CURRENT active mon's
movepool, the whole decision is SKIPPED (run no turn, emit nothing, draw nothing, record nothing) and the
next decision is re-pulled for the SAME boundary — mirroring the sim's reject-and-re-request. A valid
script never trips it, so the fullbattle / secondary / e2e goldens are unaffected.

**A SECOND, orthogonal PROTOCOL gap surfaced once the boundary aligned** (`status_para_and_boost_drop/4`):
a standalone major-status MOVE into an ALREADY-STATUSED foe emits a `|-fail|` line the port omitted.
Modeled bit-for-bit from `pokemon.ts::setStatus` (via `trySetStatus`'s `setStatus(this.status || status)`
re-pass, so `status.id === this.status`): **SAME status** (Thunder Wave→par into par) → `|-fail|<target>|par`
(fail on the TARGET, status token); **DIFFERENT status** (Thunder Wave→par into brn) → the move announce's
`[still]` empty-target form + `|-fail|<user>` (fail on the USER, no token). Both are **draw-free past the
accuracy roll** (the fail is emitted before `runEvent('SetStatus')`, so no clause shuffle even in gen3ou —
verified `harness/probe_status_move_fail_lines.js` + `probe_status_fail_accuracy.js`: the only per-move
draw is `randomChance(100,100)`). Wired in `run_status_move` (`foe_status_move_fail` classifier +
`StatusMoveFail`), keyed on the move having a top-level `move.status` field so a SECONDARY status (Body
Slam's par into a statused foe — `move.status` unset) correctly emits NOTHING.

**Pinned** by revert-verified `regression_test.rs::forced_replacement_resume_runs_the_post_replacement_move_decision`
(a constructed Aerodactyl-KO'd-by-Zapdos → 2-move-Snorlax replacement → an invalid `move 3` is SKIPPED;
ground-truth seeds from `harness/probe_forced_replacement_resume_regression_rng.js`; reverting the gate →
4 boundaries with wrong seeds → the pin fails). The status-`-fail` emission is guarded by the protocol
byte-diff itself (reverting it re-diverges at `|-fail|p2a: Blissey|par`). The 3 `recover_and_rest`
Struggle battles are now ALSO **replayed byte-exact** (see the PP-tracking + Struggle section below —
`gen3_pp_tracking_v1`) — `unreplayable_move` catches NOTHING and the protocol byte-diff grew again to
**66 battles / 8721 lines**, 0 skipped.

## ✅ FIXED — phaze multi-draw-turn `sample` desync = Protect BLOCKS Roar/Whirlwind (found by the e2e fuzz 2026-06-28; ROOT-CAUSED + FIXED 2026-07-01)
Phazing (Roar/Whirlwind) is now **INCLUDED in the e2e capstone** (`PHAZE_E2E_EXCLUDED = false`,
bit-for-bit, **1035 phaze-DRAG decisions across the 220-battle strict gate**, `filtered_diverged == 0`,
`phaze_decisions >= 50` coverage floor) after root-causing the stateful desync the single-scenario
goldens don't reach.

**The repro shape it presented** (correct, but the diagnosis below is the real cause): a real-team
battle where a phaze shares a turn with another drawing move; the FIRST phaze of the battle drags the
CORRECT mon, but a LATER phaze picks a DIFFERENT bench mon than the sim **while the post-turn SEED still
matches** (same total draw COUNT, but the `sample` reads the PRNG at a SHIFTED position, compensated
elsewhere). It looked like a draw-POSITION / list-ORDER bug — but the eligible-list ORDER + array-swap
were NEVER wrong (the port's `possibleSwitches`/swap matches the sim exactly).

**ROOT CAUSE — the port did NOT let Protect BLOCK a phaze.** Gen-3 **Roar AND Whirlwind carry the
`protect: 1` flag** (verified: `Dex.forGen(3).moves.get('roar').flags` = `{protect:1, …}`), so a
Protect / Detect on the TARGET BLOCKS the phaze at `runEvent('TryHit')` — AFTER the accuracy roll
(`data/mods/gen3/scripts.ts` line 439 accuracy → the base `tryMoveHit` TryHit), leaving the target
ACTIVE. A blocked phaze therefore sets **NO `forceSwitchFlag` → NO `dragIn` → NO `sample` draw** (the
runAction-tail drag never fires). The port's phaze arm (`turn.rs::run_status_move`) drew its accuracy
then signalled the drag UNCONDITIONALLY — it never checked the Protect block (the leechseed /
standalone-status arms already did, via `protect_blocks`, but the phaze arm was missed). So into a
protected foe the port dragged a random bench mon (an **EXTRA `sample`**) that the sim left in place.

**Why the seed still matched (the "compensated elsewhere"):** the extra `sample` shifts the PRNG one
draw at that turn, but the dragged mon differs vs the sim's un-dragged target; downstream the divergent
board consumes a compensating draw so the boundary seed re-converges — the per-decision seed assertion
matched while the dragged mon (STATE) was wrong. This is why the seed-only check didn't catch it, and why
it only surfaced when a phaze targeted a **Protecting** foe across a long switch history (the dedicated
phaze golden's scenarios never Protect INTO a phaze).

**The repro** (phaze_diff pd_1 dec5, phaze-focused differential): p1 Flygon uses **Protect** (its slot-0
move), foe Suicune Roars (priority −6, into the up Protect). The SIM keeps Flygon active (Roar blocked, no
drag); the PORT dragged **Aerodactyl** in — same boundary seed, wrong active mon.

**THE FIX** (`turn.rs`, the phaze arm): after the accuracy roll + before the `canSwitch`/drag signal, add
`if self.protect_blocks(foe, foe_slot, false) { return … }` (emit `-activate Protect`, no drag) — mirroring
the leechseed / standalone-status arms. DRAW-COUNT correct (accuracy drawn, NO sample). Substitute does
NOT block a phaze (Roar/Whirlwind carry `bypasssub: 1`), so there is intentionally no substitute check.

**Pinned by** `regression_test.rs::phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target` — a
constructed `gen3customgame` scenario (fast Skarmory Protects + a bench; slow Suicune Roars into it), fixed
init seed `13127,45333,18295,15391`, asserting the protector STAYS active + `phaze_drag == false` + the
post-turn SEED == real Showdown `3932,55062,24613,55040` (ground truth from `probe_phaze_regression_rng.js`
PHAZE-PROTECT). Verified a TRUE PIN — reverting the `protect_blocks` check trips both the species assertion
(the port drags Blissey) and the seed. See CLAUDE.md phaze section.

**Secondary issue admitting phaze surfaced (handled):** the phaze-clean corpus can carry a Gengar with
`destinybond` (a reactive `volatileStatus:'destinybond'` move, out of gen-3-modeled scope). It is now in
`gen_e2e_fuzz.js`'s `MOVE_ID_BLOCKLIST` (belt-and-braces beyond the Status-branch exclusion — `isModeledMove`
already returns false for it, so `pickMove` never picks it; the blocklist keeps a DB-carrying team off the
pickable path explicitly). The port FAIL-LOUDS on destinybond (the general unmodeled-status-move guard),
pinned by `turn.rs::destinybond_status_move_panics_fail_loud`. NOT modeled (reactive, out of scope).

## ✅ FIXED — double-faint → double-replacement → cascade `runSwitch` cancellation (found by the e2e fuzz 2026-06-30, surfaced by Explosion; FIXED same day)
Explosion / Self-Destruct is now **INCLUDED in the e2e capstone** (`EXPLOSION_E2E_EXCLUDED = false`,
bit-for-bit, **544 explosion-move / self-KO decisions across the 220-battle strict gate**,
`filtered_diverged == 0`) after fixing the two STATEFUL desyncs admitting Explosion surfaced — a
double-faint → double-replacement → cascade `runSwitch` mis-order (e2e_9) and a confusion self-hit ×
Choice-Band gap (e2e_194). NEITHER was the Explosion self-KO (proven exact by the 218 other clean
battles + the dedicated `explosion_test.rs` golden + the E1-E4 pins); Explosion was merely the common
way the fuzz produces the triggering DOUBLE FAINT.

**Root cause (e2e_9) — the port kept a STALE `runSwitch` through a cascade.** When a mutual double
faint forces BOTH sides to replace, both fresh entrants enqueue an order-101 `runSwitch` (the 2nd ties
the 1st → the splice draw). The two runSwitches then run in speed order. If the FIRST runSwitch to run
FAINTS its own entrant (its own side's Spikes KO on entry — the cascade), the sim's gen-3-singles
`faintMessages` (`battle.ts:2606-2616`: `for (const pokemon of this.getAllActive()) this.queue.cancelAction(pokemon)`,
"in gen 3, fainting skips all moves AND SWITCHES") REMOVES the OTHER side's still-pending `runSwitch` —
because `cancelAction(pokemon)` (`battle-queue.ts:329`) drops EVERY queued action whose `action.pokemon ===
pokemon`, and a `runSwitch`'s `action.pokemon` is the entrant (a getAllActive member). So the OTHER
entrant's Spikes chip is NEVER applied: it stays at FULL HP. The port's `cancel_active_actions`
(`turn.rs`) cancelled `Move` / `Switch` actions but NOT a pending `RunSwitch` → the stale foe runSwitch
survived the cascade + re-applied the foe's Spikes to its already-settled entrant (e2e_9 dec43: p2's
fresh entrant wrongly chipped, e.g. 403 → 353). SEED bit-for-bit (a queue splice is draw-free) — a pure
STATE (HP) mis-application. VERIFIED vs the omniscient sim (`harness/probe_cascade_hazard_order.js`,
`harness/probe_double_replacement_spikes_rng.js`): with the FAINTING side's runSwitch FIRST the foe
entrant is UNCHIPPED (403); with the SURVIVING side's runSwitch first the foe is chipped ONCE (its
runSwitch already ran → nothing to cancel). **The fix:** `cancel_active_actions` now also removes a
`RunSwitch { side }` when `sides[side].active` is NOT fainted (a getAllActive member) — the exact
`cancelAction(getAllActive)` semantics. DRAW-FREE (no PRNG) → the seed is untouched.

**Root cause (e2e_194) — the confusion self-hit dropped Choice Band.** gen-4 confusion (which gen-3
inherits, `data/mods/gen4/conditions.ts:74-83`) runs `this.actions.getDamage(pokemon, pokemon, 40)` — the
FULL `getDamage`, NOT the simplified `getConfusionDamage` (that's the base/gen7 path) — so the attacker's
`onModifyAtk` item **Choice Band ×1.5 (physical)** folds into the typeless self-hit. The port's
`apply_confusion_self_hit` passed NO atk stat mods → it used the stored Atk (not the CB-boosted Atk) → the
self-hit under-dealt (a Choice-Band Aerodactyl's self-hit used Atk 339, not the CB 508 → the mon kept too
much HP). SEED bit-for-bit (the self-hit draws the SAME `random(1,2)` + `random(16)` either way) — a pure
STATE (HP) mis-application. **The fix:** `apply_confusion_self_hit` now resolves
`resolve_atk_stat_mods(item, move_type=None, Physical)` — the SAME helper a real move uses — so Choice Band
applies (typeless '???' → NO type-boost item / Sea Incense). VERIFIED vs the sim
(`harness/probe_confusion_choiceband_regression_rng.js`: the self-hit rolls jump ~71 → 90-106).

**Pinned by** two revert-verified regression tests: `regression_test.rs::double_replacement_cascade_does_not_rechip_the_other_sides_entrant`
(a constructed mutual-Explosion double faint, 3 Spikes on p1 side + 1 Spike on p2 side, p1's FAST Jolteon
pre-damaged so its own 3-layer Spikes KO it on entry → the foe Snorlax stays FULL HP; STATE + per-decision
SEED, ground truth from `probe_double_replacement_cascade_regression_rng.js`) and
`confusion_self_hit_applies_choice_band` (a confused Choice-Band Aerodactyl self-hits with its CB-boosted
Atk; STATE + SEED, ground truth from `probe_confusion_choiceband_regression_rng.js`). Both verified TRUE
pins (reverting each fix trips the HP assertion: the cascade → foe re-chipped 459; the CB → self-hit
under-deals). The e2e `explosion_decisions >= 50` coverage floor is restored in `e2e_fuzz_test.rs`.

**KNOWN SCOPE GAP (pre-existing, NOT gate-exercised) — confusion self-hit ignores screens.** Because
gen-4 confusion runs the FULL `getDamage(pokemon, pokemon, 40)`, the confused mon's OWN-side **Reflect**
reduces the typeless-physical self-hit (`runEvent('ModifyDamagePhase1')`). `apply_confusion_self_hit` now
folds Choice Band (above) but still passes `reflect:false`/`light_screen:false` to `calc_damage`, so a
confused mon self-hitting **with its own Reflect up** would over-deal (a STATE-only HP desync; the draw
model — `random(1,2)` + `random(16)` — is unchanged, so the SEED stays bit-for-bit). This is pre-existing
(the pre-CB-fix code also passed `reflect:false`) and NOT reachable by the 220 filter-clean e2e battles
(none hits a screened confusion self-hit), so the strict gate stays bit-for-bit. FIX when needed: resolve
the confused mon's own-side Reflect/Light Screen into the self-hit `DamageContext` (a '???' physical
self-hit is subject to Reflect in gen-3), mirroring how a real physical move reads the defender-side
screens; then pin it with a Reflect-up confused-self-hit regression test.

## ✅ FIXED — switch-in-into-a-speed-TIE + fresh-weather `eachEvent('WeatherChange')`-shuffle desync (found by the e2e fuzz 2026-06-30, surfaced by Substitute; FIXED same day)
The MID-TURN switch-in `eachEvent('WeatherChange')` tie-shuffle is now MODELED — **Substitute is back
in the e2e capstone (`SUBSTITUTE_E2E_EXCLUDED = false`), bit-for-bit, 284 substitute-MOVE / 320
sub-up decisions across the 220-battle strict gate**, and the e2e_84 dec4 desync is closed.
- **Root cause:** `Field.setWeather` (field.ts:87) ends with `this.battle.eachEvent('WeatherChange',
  sourceEffect)` — a 2-active `speedSort(getAllActive())` that draws ONE `random(0,2)` Fisher-Yates
  tie-shuffle IFF the actives TIE on cached speed (and NOTHING on distinct speed; the gen-3 `>=7`
  Update-nest is NOT reached, so NO nested Update). When a Sand Stream / Drizzle / Drought entrant
  CHANGES the weather on a MID-TURN switch-in (the `runSwitch` ability `Start`), that shuffle fires
  INSIDE the runSwitch runAction, BEFORE its trailing `eachEvent('Update')`. The port set the weather
  draw-FREE and MISSED that shuffle → e2e_84 dec4 (`init_seed 52903,53571,56373,31187`): a 213-speed
  Tyranitar (Sand Stream) switches in while a 213-speed Suicune acts → the sim drew 8, the port 7.
  A pure draw-COUNT desync (STATE — HP / sub-HP / status / species — matched; only the seed diverged).
- **The fix** (`turn.rs`): `run_switch` now snapshots `(field.weather, field.weather_turns)` across the
  ability `Start` and returns whether the weather CHANGED; the `QAction::RunSwitch` handler in `turn_loop`
  fires one `each_event_shuffle()` (the `eachEvent('WeatherChange')`) when it did — for EVERY mid-turn
  switch-in (voluntary, forced-replacement, AND phaze-drag), since they all route through that one
  runAction. A same-ability-permanent-weather re-set is a `setWeather`-returns-false no-op (`run_switch`
  returns false → no shuffle), matching the sim. The `>start` switch-in path (`run_start_switchins` →
  `single_event_ability_start` DIRECTLY, NOT via `run_switch`) is UNAFFECTED — it stays draw-free, so the
  `switchin_test.rs` "dispatch draws ZERO" assertion still holds (the bounded `>start` step omits those
  draws; the e2e seeds at the post-`>start` state to absorb them).
- **Pinned by** `tests/regression_test.rs::switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed`
  — a constructed 213-vs-213 Suicune-lead-vs-(Suicune+Tyranitar-Sand-Stream) `gen3customgame` scenario,
  fixed init seed `52903,53571,56373,31187`, p1 switches the Sand-Stream Tyranitar into the 221-speed tie
  → asserts the post-turn SEED == real Showdown `54657,11218,22550,62890` (ground truth from
  `harness/probe_switch_tie_weather_regression_rng.js`, reseeded to the raw seed at the decision so it
  lines up with the draw-free `start_with_switchins`). A TRUE PIN — verified to FAIL (seed `39365,…`) if
  the `eachEvent('WeatherChange')` shuffle is reverted. The same class as the `forced_replacement_recaches_
  speed_seed` / `para_while_active_keeps_full_cached_speed_seed` cached-speed regressions. The
  `harness/probe_switch_sand.js` control proves the +1 shuffle (Sand-Stream switch-into-a-TIE draws 9 vs
  a no-weather switch-into-a-TIE 7 vs a Sand-Stream switch into a NON-tie 1).

## Trapping (abilities + moves) — was the #3/#4 e2e team-carry blocker; the ABILITIES are ✅ BUILT
- ✅ **Arena Trap (Dugtrio)** — **BUILT** (`gen3_trapping_v1`, `turn.rs::is_trapped` + the
  `move_decision_is_legal` Switch gate): traps GROUNDED foes (no voluntary switch); a Flying-type or
  Levitate foe is NOT trapped (gen-3 grounded == not-Flying && not-Levitate — the spikes rule; no
  Gravity/Iron Ball/Magnet Rise in gen3). PROBE-SETTLED SURPRISE: **a grounded GHOST (Sableye) IS
  trapped in Showdown-gen3** — the gen3 dex resolves NO `trapped` type-immunity (Ghost
  `damageTaken.trapped` = undefined; the cartridge gen6+ Ghost escape does NOT exist in this sim) —
  pinned by `grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3`. Arena Trap is `onFoe*` → 1
  handler per endTurn trap event → **ZERO draws** always. ["dugtrio trapped"]
- ✅ **Both mons are Dugtrio** — **BUILT + probe-verified**: MUTUAL Arena Trap — each Dugtrio traps the
  other (both `pokemon.trapped='hidden'`), neither can voluntarily switch; and the mirror consumes
  ZERO extra PRNG (probe: boundary seeds byte-identical to a Sand Veil control). Golden scenario
  `dugtrio_mirror_mutual_trap_no_draws` (trapping_test.rs). ["both are dugtrio"]
- ✅ **Magnet Pull (Magneton)** — **BUILT**: traps STEEL-type foes ONLY — groundedness IRRELEVANT
  (Skarmory, Steel/Flying, is trapped; probed). The MAGNETON MIRROR mutual-traps (Steel↔Steel) AND
  draws: gen3 overrides magnetpull to `onAnyTrapPokemon`/`onAnyMaybeTrapPokemon`
  (data/mods/gen3/abilities.ts) so BOTH actives' handlers register on EVERY endTurn
  TrapPokemon/MaybeTrapPokemon event → the speed-TIED mirror draws ONE Fisher-Yates tie-shuffle per
  event per mon = **4 draws per endTurn** (probe: 11/turn vs the Sturdy control's 7; a para that
  breaks the speed tie silences them); an Arena-Trap-vs-Magnet-Pull cross at equal speed draws 2
  (both events on the MP holder). Modeled by `turn.rs::trap_event_shuffles` INSIDE the endTurn
  per-mon loop (DisableMove → TrapPokemon → MaybeTrapPokemon per mon, before the gen3
  quickClawRoll). NO self-trap (`isAdjacent(self,self)` is false — a lone MP holder is only trapped
  if the FOE traps it). Pinned by `magnet_pull_traps_steel_only`. ["magnet pull on magneton"]
- ✅ **The CHOICE-REJECTION path** — **BUILT** (the switch mirror of the PP/forced-replacement
  reject-and-re-request gate): the sim's `chooseSwitch` at a `move` request rejects a trapped mon's
  voluntary switch DRAW-FREE ("Can't switch: The active Pokémon is trapped"; seed byte-identical,
  request stays open) — `move_decision_is_legal` now rejects a scripted `Switch` by a trapped mon
  (decision SKIPPED draw-free, next decision re-pulled). Forced replacements (`requestState ===
  'switch'`) are NEVER gated; a PHAZE (Roar) still DRAGS a trapped mon (pinned by
  `roar_drags_a_trapped_mon_out`); the trapping mon itself switches freely. REQUEST-DISPLAY nuance
  (informational, not modeled — no legality/draw impact): both abilities `tryTrap(true)` →
  `trapped='hidden'`, so the request JSON shows `maybeTrapped` until a rejected attempt patches it to
  `trapped: true`, and omits both flags when no bench is live (`canSwitchIn` false).
  ["attempting to switch but having it rejected"]
- **Mean Look / Spider Web / Block** (trapping MOVES) — DEFERRED, fail-loud: they are
  `volatileStatus` Status moves, so the port PANICS on them via the unmodeled-status-move guard (and
  the e2e allow-list excludes them). When built: the `trapped`-volatile flavor of `is_trapped` + a
  named pin. (The modern-gen Ghost trap-escape does not apply in Showdown-gen3 — see the Sableye
  surprise above.) ["mean look etc"]
- **Shadow Tag** — (rare in gen3 OU) traps ALL foes; DEFERRED — an unmodeled ability (not in the e2e
  `MODELED_ABILITIES`, so its teams stay off the filtered path; `is_trapped` returns false for it,
  which is only reachable in constructed scenarios that don't use it). Note gen3 Wobbuffet-mirror
  endless battles are a format-clause issue, not a port issue.

## ✅ FIXED — gen-3 INTIMIDATE vs SUBSTITUTE (found by the e2e fuzz 2026-07-01, surfaced by the `gen3_trapping_v1` corpus shift; FIXED same day)
- **The bug**: a MID-BATTLE Intimidate switch-in dropped the Atk of a foe BEHIND A SUBSTITUTE. The
  gen3 mod's Intimidate SKIPS a subbed foe (probe: sub up → NO `|-unboost|`, boosts unchanged; the
  block is SEED-NEUTRAL — identical draws/seeds with or without the sub, so it is STATE-only).
  `event::intimidate_on_start`'s doc even noted the quirk but called it "inert at switch-in (no subs
  exist turn 0)" — TRUE at battle start, STALE once mid-battle switching was built (the function runs
  on every entrant). Surfaced ONLY when the trapping e2e regen re-sampled the filter-clean pool
  (e2e_171/e2e_204: a Jynx Substituted the turn before a Salamence switch-in — sim Atk 0, port −1;
  a STATE-only divergence the seed match masked).
- **The fix**: `intimidate_on_start` gates on the target's `substitute` (before the
  Clear-Body-family `onTryBoost` gate). Probe: `harness/probe_intimidate_substitute_rng.js`;
  pin: `regression_test.rs::intimidate_into_a_substitute_is_a_noop` (revert-verified, both arms —
  the sub blocks AND the no-sub control still drops).

## Recharge / multi-turn lock
- **Hyper Beam (+ Giga Impact n/a gen3)** — after a HIT the user gains a `mustrecharge` volatile; its
  NEXT turn it is forced to "recharge" (a no-op `|cant|recharge` action, draws nothing for the move but
  the turn still runs). The forced no-action is a draw-COUNT subtlety (does the recharge turn skip the
  move's normal draws?). Gen3 Hyper Beam recharges even if the hit KO'd / missed? (verify: gen3 recharges
  on a LANDED hit only.) ["hyper beam must recharge"]
  - (Related, deferred: the lock-in moves — Thrash/Petal Dance/Outrage [2-3 turn lock + confusion],
    Rollout/Ice Ball, Uproar; and the charge moves — Solar Beam/Sky Attack/Razor Wind/Fly/Dig/Dive.)

## Substitute — BUILT ✅ (2026-06-30, `gen3_substitute_v1`)
Substitute is MODELED bit-for-bit (`MonState::substitute: Option<u16>`; `run_status_move`'s substitute
arm + `absorb_into_sub` + the secondary suppression + the status/leech sub-block). Validated by
`tests/substitute_test.rs` (9 scenarios × 80 seeds, 4320 decision rows) + 4 `tests/regression_test.rs`
pins. INCLUDED in the e2e capstone (`SUBSTITUTE_E2E_EXCLUDED = false`). The model:
- COST `floor(maxhp/4)` HP, creates a sub with that HP; never-miss; FAILS draw-free if already-subbed or
  `hp <= floor(maxhp/4)`. The sub ABSORBS incoming damage (sub HP drops; BREAKS at 0; the excess does NOT
  carry to the mon in gen-3).
- BLOCKS status moves + stat-drop secondaries (draw-free past accuracy). Does NOT block PHAZING
  (Roar/Whirlwind still drags the user). A confusion self-hit hits the **MON, not the sub** (the
  self-hit's `this.damage` bypasses the `onTryPrimaryHit` sub-intercept) — **CORRECTED** from the
  earlier backlog note (which wrongly said the self-hit hits the sub).
- **THE DRAW-COUNT SURPRISE (CORRECTED — settled by `harness/probe_substitute_secondary.js`):** the
  earlier backlog note (and the build task's stated assumption) claimed "the sub-block short-circuits
  BEFORE the secondary `random(100)`, drawing one fewer." **This is WRONG for gen-3.** The gen-3
  `secondaries()` iterates the now-`null` target list, so the per-move secondary `random(100)` IS STILL
  DRAWN against a sub (the SAME count as a bare hit) — only its EFFECT (status/stat-drop/flinch, AND any
  confusion `random(2,6)` / Tri-Attack `random(3)` follow-on) is SUPPRESSED. So a damaging move into a
  sub is DRAW-COUNT-NEUTRAL vs a bare hit (which is also why substitute, unlike phaze, can't introduce a
  multi-draw-turn position desync — it adds/removes NO draws). Pinned by
  `substitute_absorbs_a_hit_but_the_secondary_random_100_still_draws` +
  `tri_attack_into_a_sub_draws_random_100_but_not_the_sample_random_3`. ["substitute"]
- DEFERRED (fail-loud / out of gen-3 scope): Baton Pass passing a sub, Shed Tail (gen-9), the
  `bypasssub`/`infiltrates` move flags (none on a modeled gen-3 move), Liquid Ooze (already deferred).

## Explosion / Self-Destruct — the SELF-KO is BUILT ✅ (2026-06-30); e2e INCLUDED ✅ (bit-for-bit)
Explosion / Self-Destruct are MODELED bit-for-bit (the self-KO was already wired in `turn.rs`; this layer
VALIDATED the edges). A Normal PHYSICAL damaging move (BP 250 / 200 gen3) that HALVES the target's Def
(the `selfdestruct` flag, reused as `halves_def` in `damage.rs`) and faints the USER as part of the move.
Validated by `tests/explosion_test.rs` (7 scenarios × 80 seeds, 3688 decision rows, 7376 FAINTED
assertions, 880 self-KO rows, 294 sub-break boundaries) + 4 `tests/regression_test.rs` pins (E1-E4). The
model (verified vs `harness/probe_explosion_rng.js`):
- **THE SELF-KO IS UNCONDITIONAL + DRAW-FREE + PRECEDES THE HIT.** gen-3 `useMoveInner`
  (battle-actions.ts:501-503, `gen != 4 && selfdestruct == 'always'`) calls `this.battle.faint(pokemon)`
  — zeroing the user's HP + queuing its faint — BEFORE `trySpreadMoveHit`. So the USER FAINTS REGARDLESS
  of the hit outcome: a normal hit, into a **SUBSTITUTE** (the damage breaks the sub, no carry; the user
  still faints), into a **PROTECT** (the move is blocked, no foe damage; the user STILL faints — `-activate
  Protect` then `|faint|`), into a **GHOST** (Normal-immune, no damage; the user STILL faints), or a miss
  (gen-3 Explosion accuracy is 100, so no self-accuracy miss — but a hypothetical miss would still faint
  the user, the faint precedes the hit). The `run_move` self-KO sits AFTER `on_before_move` (a fully-para/
  asleep/flinched user never reaches `useMoveInner` → no self-KO) but BEFORE the accuracy/immunity/protect/
  miss checks — matching the source order exactly.
- **DRAW-COUNT:** Explosion draws the SAME as any damaging move — accuracy (`randomChance(100,100)`, the
  gen-3 acc-100 value, always passes but STILL draws) → crit → damage `random(16)`. It has NO secondary.
  The self-KO adds NO draw; but the resulting faint changes `pokemon_left` / can end the battle / force a
  replacement, so it CANCELS the foe's queued move (gen-3 singles) and draws NO trailing Quick Claw on a
  deciding faint. A blocked (Protect) / immune (Ghost) Explosion draws only its accuracy then the user
  faint (no crit/dmg, no Quick Claw). A MUTUAL Explosion (both last mons) is a true double-faint gen-3 TIE
  (`win(None)`). Pinned by E1 (Protect), E2 (Ghost), E3 (sub-break), E4 (mutual TIE).
- **SCENARIO gotcha (NOT an engine bug):** an Explosion-into-real-battle golden scenario initially desynced
  on a downstream **Body Slam** (a CONTACT move) KOing a foe with **Cute Charm** / **Static** — those
  abilities draw a contact `random` the port doesn't model. Fixed by giving the frail foes a NO-OP ability
  (Oblivious); Explosion itself is non-contact so it never triggers a contact ability.
- **e2e INCLUDED — bit-for-bit (`EXPLOSION_E2E_EXCLUDED = false`).** Admitting Explosion to the e2e
  capstone surfaced TWO STATEFUL desyncs in DIFFERENT layers (NOT the self-KO) — both now FIXED (see the ✅
  section above): a **double-faint → double-replacement → cascade `runSwitch` cancellation** (e2e_9 — the
  port kept a stale foe `runSwitch` through the cascade → re-chipped the foe entrant; fixed by cancelling a
  pending `RunSwitch` in `cancel_active_actions`, the `cancelAction(getAllActive)` semantics) and a
  **confusion self-hit × Choice Band** gap (e2e_194 — the self-hit dropped Choice Band; fixed by folding
  `resolve_atk_stat_mods` into `apply_confusion_self_hit`). With both fixed, all 220 filtered battles are
  bit-for-bit (`filtered_diverged == 0`) with **544 explosion-move / self-KO decisions**, so Explosion is
  INCLUDED with an `explosion_decisions >= 50` coverage floor. Pinned by
  `double_replacement_cascade_does_not_rechip_the_other_sides_entrant` + `confusion_self_hit_applies_choice_band`.
  ["explosion", "selfdestruct"]

## Perish Song
- **Perish Song** — sets a `perish3` counter on EVERY active mon (sound-based; Soundproof immune); the
  counter decrements each end-of-turn (`perish3→2→1→0`) and at 0 the mon FAINTS. A both-sides volatile +
  the end-of-turn faint. ["perish song"]

## ✅ FIXED (partially) — PP tracking + forced Struggle + Choice-Band lock (`gen3_pp_tracking_v1`, 2026-07-01)
The MOVE-choice side of request validation — the first brick of `LegalActions`. **PP tracking**, the
**forced-Struggle** fallback, and the **Choice-Band move-lock** are now MODELED bit-for-bit; the 3
`recover_and_rest` Struggle protocol battles REPLAY byte-exact (protocol byte-diff 63/7223 → **66/8721**,
0 skipped). Verified vs the omniscient sim (`harness/probe_pp_struggle_rng.js` + `..._regression_rng.js`):
- **PP init** — a moveslot's in-battle PP is `calculatePP(move, 3) = pp * 8 / 5` (the `Pokemon` ctor's
  hardcoded default **3 PP-ups** for every non-`noPPBoosts` move — NOT the moveset's raw PP), or the raw
  `pp` for a `noPPBoosts` move (Struggle = 1). Added `pp` + `noPPBoosts` to `gen3_moves.json` (via the
  extractor; obs-neutral, the facade ignores it, like `critRatio`) → `MoveData::max_pp()` →
  `MonState::move_pp` init.
- **PP decrement** — −1 per USE, **DRAW-FREE**, and ONLY when the mon actually MOVES (a full-para /
  sleep / flinch / frozen / confusion-self-hit turn deducts NOTHING — `deductPP` runs AFTER
  `runEvent('BeforeMove')` PASSES). A MISS / an IMMUNE hit STILL decrement. PP does NOT reset on
  switch-out (gen-3 — it PERSISTS).
- **Pressure −2** — a move TARGETING a Pressure holder deducts **2** PP (the `runEvent('DeductPP')`
  extra), DRAW-FREE. (CLAUDE.md called Pressure a "provable no-op in a damaging-move-only fuzz" — PP-wise
  it is NOT; this wires the −2.)
- **Choice-Band lock** — a Choice-item mon (gen-3: only **Choice Band**) LOCKS to the FIRST slot it uses
  (`choiceband.onModifyMove` → `choicelock`); every other slot is disabled. This is what forces Struggle
  when the LOCKED move hits 0 PP while other slots still have PP (the CB-Tyranitar exhausting Crunch →
  Struggle in the protocol battles). Cleared on switch-out / faint.
- **Forced Struggle** — when the mon has NO usable move (all slots 0 PP, OR the Choice lock leaves only a
  0-PP slot) `side.choose` substitutes `moveid:'struggle'` for the scripted `move K`. The port's queue-
  build sets a `struggle` flag from `must_struggle()`; a `move K` on a 0-PP slot while ANOTHER move is
  usable is REJECTED draw-free (the `move_decision_is_legal` PP gate, mirroring the sim's reject).
- **Struggle mechanics (the probe SETTLED the hints — do not trust them)** — type **typeless '???'** (no
  STAB, HITS everything INCLUDING Ghosts — a typeless move has no type-chart row → 1×), **50 BP
  PHYSICAL**, **accuracy 100 (NOT never-miss** → it DRAWS an accuracy roll), then crit + damage like any
  normal move. **RECOIL = `max(floor(damageDealt / 4), 1)`** — the gen-3 `data/mods/gen3/moves.ts`
  override `{recoil:[1,4], struggleRecoil:false}` + the gen-3 `scripts.ts::calcRecoilDamage` **`Math.floor`**
  (NOT the base-sim `Math.round`, NOT the gen4+ `struggleRecoil = maxhp/4`): floor(130/4)=32 not
  round=33 not maxhp/4. Applied DRAW-FREE. Struggle consumes NO PP + does not set the Choice lock. A
  Struggle turn draws the SAME as a normal damaging move (acc + crit + dmg + Quick Claw when not the
  deciding faint).
- **Truncation-marker fix** — the `|turn|N` marker is now emitted at the REQUEST (before choice
  validation), so a rejected fresh-turn `move` (out-of-range OR out-of-PP) still shows its marker
  (previously emitted only on the first VALID submission → a truncated capture whose final recorded
  decision is a rejected out-of-PP `move`, e.g. `spikes_and_phaze/2`'s 16th-Earthquake exhaustion, lost
  the trailing `|turn|21`). Observation-only — the seed suites (e2e 13367 / battle 2034 / fullbattle 2053
  / secondary 4328) stay BYTE-IDENTICAL; PP + the lock + the substitution are ALL draw-free.

Validated by `tests/pp_struggle_test.rs` (the per-decision STATE+HP+STATUS+**PP**+SEED+winner
differential: 400 runs, 4424 seed + 8368 PP assertions, 1035 forced-Struggle, 1035 recoil, 378
Pressure−2, 3292 immune-decrement) + the un-skipped protocol battles + a `max_pp` dex unit gate + 4
DETERMINISTIC revert-verified `regression_test.rs` pins (`pp_decrements_on_use_draw_free`,
`pressure_decrements_two_pp`, `no_usable_move_forces_struggle_and_struggle_recoil_is_gen3_quarter_damage_dealt`;
ground truth from `harness/probe_pp_struggle_regression_rng.js`).

**Taunt + Disable are now BUILT** (`gen3_taunt_disable_v1` — see the dedicated section below):
`move_usable`/`must_struggle` fold both restriction sources in (taunt blocks every Status slot, disable
the one recorded slot), composing with the PP + Choice-lock gates into forced Struggle. **STILL
DEFERRED** (the rest of move-legality): **Torment**, **Imprison**. Choice **Scarf/Specs** are gen4+
(N/A in gen3).

## Taunt + Disable — BUILT ✅ (2026-07-01, `gen3_taunt_disable_v1`)
The gen-3 move-SELECTION-restriction layer is MODELED bit-for-bit (`MonState::taunt: Option<u8>` +
`MonState::disable: Option<(usize, u8)>` + `MonState::last_move`; `run_status_move`'s taunt/disable
arms + the `move_usable`/`must_struggle` restriction + the `on_before_move` execution-time cants + the
residual duration ticks + the endTurn `runEvent('DisableMove')` handler-sort shuffle). Validated by
`tests/taunt_disable_test.rs` (9 scenarios × 80 seeds: 720 runs, 4723 seed assertions, 8595 taunt +
8595 disabled-slot assertions, free-up boundaries on BOTH disable branches) + 4 revert-verified
`tests/regression_test.rs` pins (TD1-TD4, ground truth `harness/probe_taunt_disable_regression_rng.js`).
INCLUDED in the e2e capstone's modeled set (`MODELED_RESTRICTION_MOVES`). The model:
- **TAUNT** (Dark, **accuracy 100** — NOT never-miss → DRAWS `randomChance(100,100)`): applies the
  `taunt` volatile, duration a **FIXED 2, NO duration draw** — and NO base-`onStart` `duration++`:
  gen3's condition `inherit: true`s from the **gen4 mod**, whose `onStart` is a plain `-start` (the
  base `data/moves.ts` onStart with the `activeTurns && !willMove` duration++ is SHADOWED). **A 3-lens
  review once read the BASE source and declared the port's constant-2 wrong — the probes
  (`probe_taunt_duration_branch.js`, independently re-run) REFUTED it: a taunter-SECOND-on-turn>=2
  stores the same 2.** While taunted every Status-category slot is un-selectable (the derived-Status
  set MINUS the fixed-damage family — Seismic Toss stays usable, probe-verified); a QUEUED status move
  is cant'd at execution (`onBeforeMove` priority 0 — AFTER the paralysis roll at 1), draw-free, no PP.
  Residual duration tick at **order 10 / subOrder 15** (gen4-inherited — NOT the base's order 15;
  probe: a FAST taunted mon's `-end` precedes a SLOW foe's brn `-damage` in the same residual).
- **DISABLE** (Normal, **accuracy 55** — CAN miss): disables the target's **lastMove** slot.
  `onTryHit` FAILS draw-free with no lastMove (never moved / just switched in / lastMove Struggle).
  On a landed hit: ONE `random(2,6)` (the gen3 durationCallback), then the gen4-inherited `onStart`
  does `duration++` iff the target ALREADY moved — **stored = disabler-faster ? rolled : rolled+1**
  (`turn.rs`, PROBE-SETTLED by `probe_disable_full_lifecycle.js` + siblings; here too a base-source
  reading — base onStart does `duration--` on `willMove` — mis-predicts by a constant +1 on both
  branches; the gen4 layer is the truth). The residual tick (order NO_ORDER / the Condition-default
  subOrder 2) frees the slot at 0; the golden pins BOTH branches' exact free-up boundaries. A QUEUED
  disabled move is cant'd at execution (`onBeforeMove` priority 7 — BEFORE confusion 3 + paralysis 1:
  a paralyzed+disabled mon draws NO para roll, the opposite of taunt), draw-free, no PP.
- **Both**: `protect: 1` (a Protect/Detect BLOCKS them, after their accuracy roll) + `bypasssub: 1`
  (a Substitute does NOT block); re-application FAILS draw-free (`addVolatile` false — an
  already-disabled target's re-Disable draws NO `random(2,6)`); both volatiles clear on switch-out +
  faint (and `last_move` resets, so a Disable into a fresh switch-in fails draw-free). A mon whose
  every slot is restricted (taunt × disable × Choice lock × 0 PP) is FORCED to Struggle. A
  taunt+disable (or Choice-lock+disable) mon draws ONE size-2 handler-sort shuffle at `endTurn`'s
  `runEvent('DisableMove')` (per active, array order — the only DisableMove-event draw).
- The Disable **onStart 0-PP guard** is modeled (`gen3_disable_zero_pp_v1`): a Disable landing on a
  target whose lastMove slot has **0 PP** left (last PP spent, lastMove not yet overwritten — e.g. a
  mon now forced to Struggle) consumes the accuracy + `random(2,6)` draws but the volatile is
  **REJECTED** (`-fail`, the announce retro-edited to `[still]`, NO `-start`, NO residual duration
  handler). Probe `harness/probe_disable_zero_pp_rng.js`; revert-verified pin TD5
  `disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile`. Pre-fix the port recorded a PHANTOM
  volatile whose residual duration handler could TIE a taunt/stall/flinch handler → an extra
  tie-shuffle draw → a latent seed desync (unreached by every pre-existing gate).
- e2e coverage (honest): several gen3ou sample teams carry TAUNT → real taunt decisions in the
  filtered gate; NO sample team carries DISABLE → 0 e2e disable decisions (the leech-seed situation) —
  disable is proven by its DEDICATED golden + the TD pins.
- **LATENT `blocked_by_taunt()` over-block — the bp-0 `basePowerCallback` family** (no code change;
  fix when they're modeled): the port's `derive_category` derives category from base power (bp 0 →
  Status), but **Return / Flail / Reversal / Low Kick / Magnitude / Present** are bp-0
  `basePowerCallback` moves that are **Physical** in the sim's gen-3 dex — gen3 Taunt does **NOT**
  block them (`move.category === 'Status'` reads the TRUE category), while `blocked_by_taunt()`
  (derived-Status MINUS the fixed-damage family) WOULD. All six are unmodeled + fail-loud today
  (`run_status_move` panics on an unmodeled status arm, and none is in the e2e `isModeledMove` set),
  so the over-block is unreachable — but whoever models them must carry the real Showdown category
  (or extend the `is_fixed_damage`-style exclusion to the basePowerCallback family) and pin
  taunted-Return staying selectable, or Taunt will wrongly restrict them.
- DEFERRED (fail-loud): **Torment**, **Imprison** (the remaining selection restrictions); Taunt/Disable
  vs Encore interplay (Encore itself unmodeled).

## Liquid Ooze (the Leech Seed drain reversal) — DEFERRED (fail-loud)
Leech Seed is now modeled (the `leechseed` MOVE + the residual drain/heal at order 10 subOrder 5, the
4-way residual-order interaction pinned). The ONE leech sub-case deferred: a **Liquid Ooze** target
(Tentacool/Tentacruel) REVERSES the drain — the SEEDER takes the damage instead of healing
(`onSourceTryHeal`, gen4-inherited: `canOoze = ['drain','leechseed']` → `this.damage(damage); return 0`).
Rare in gen-3 OU. The port FAIL-LOUDs in `apply_leech_seed` if a Liquid Ooze mon is ever seeded, and
**`liquidooze` was REMOVED from the e2e harness's `NOOP_ABILITIES`** (it is no longer a no-op now that
leech is modeled) so its teams are kept OFF the filtered gate. When built: model the reversal (the seeder
takes `floor(maxhp/8)`, no heal) + add a named deterministic regression pin + re-add `liquidooze` to the
no-op set. ["liquid ooze reverses leech"]

## A/B-fuzz smoke findings (2026-07-03, master seed 20260703) — KNOWN UNFIXED, repro'd
The first bounded A/B-fuzz smoke (`harness/ab_fuzz.js` — see CLAUDE.md "## A/B fuzzer") saved
standalone repros under `harness/ab_fuzz_out/smoke_{randbats,random}/divergences/` (pool: 100/100
clean). None of these touch the strict e2e gate (its corpus never exercises them). Replay any
repro with `target/release/ab_replay <repro-dir>`; a fix must flip it to `ok` + get a named
deterministic regression pin. Triaged clusters:
1. ✅ **FIXED (2026-07-09, `gen3_facade_v1` + `gen3_defrost_v1`) — Facade ×2-when-statused +
   the runEvent-tail INTEGER-GUARD; and the sacredfire FROZEN-defrost tail.** Facade carries
   the dist `onBasePower` (`isModeledMove` never rejected it) → admitted but priced flat BP 70.
   **PROBE-SETTLED MODEL** (`harness/probe_facade_gen3.js`): the handler is
   `if (pokemon.status && status !== 'slp') return chainModify(2)` — a BASE-POWER-phase CHAIN
   member (joins the one accumulated 4096 modifier), DRAW-FREE; psn/tox/par ×2 (70→140); brn ×2
   AND the gen3 burn damage-halve STILL applies (gen3 Facade does NOT ignore burn: max-roll 108
   == unstatused 108); burned GUTS composes Atk ×1.5 + halve-suppressed + BP ×2 (318). The probe
   ALSO overturned the port's "a Direct item DISCARDS the BP chain" shortcut: Pink Bow (Normal
   ×1.1 DIRECT float) + Facade (Normal chain) CO-FIRE and `70 * 1.1 == 77` EXACTLY in f64 → the
   sim's runEvent-tail guard (`relayVar === Math.abs(Math.floor(relayVar))`, battle.js:709)
   PASSES → the ×2 chain RE-APPLIES → BP **154**, not 77; `damage.rs` now implements the exact
   integer-guard. FIX: a `run_move` BP-chain member, id-gated per the fixed-damage precedent.
   **THE SACREDFIRE TAIL** (the §4 hypothesis, now PROBED — `harness/probe_sacredfire_defrost.js`):
   the resolved gen3 `frz.onBeforeMove` draws the 1/5 thaw roll FIRST (it DRAWS even for a
   defrost move — the hypothesis's "bypasses the roll" was WRONG), but on a FAILED roll a
   `flags.defrost` move (Sacred Fire / Flame Wheel — the only two gen3 carriers) PROCEEDS and
   thaws draw-free via `frz.onModifyMove` (`|-curestatus|…|[from] move:` BEFORE the `|move|`
   line): frozen defrost user = moved 25/25, thawed 25/25, EXACTLY +1 draw vs healthy. The
   port's old always-cant model was the draw-count desync. FIX: the freeze arm's defrost branch
   (`is_defrost_move`, id-gated — `gen3_moves.json` carries no flags object). **Pinned** by the
   revert-verified `regression_test.rs::facade_status_doubles_bp_and_composes` (FA-a..e, incl.
   the FA-d bow composition) + `frozen_defrost_move_bypasses_the_cant_and_thaws` (DF-a..c);
   ground truth `harness/probe_facade_defrost_regression_rng.js`; each of the 3 components
   (facade member / integer-guard / defrost branch) revert-fails its pin. **Parity:
   auto_0709_0805 (the gender-pinned corpus): 151/215 repros flip `ok` (facade-team 143/145,
   sacredfire/flamewheel-team 8/10); auto_0708_0304: now 707/845 ok (facade-team 333/344).
   NO admission change** — facade/sacredfire/flamewheel were ALREADY admitted by
   `isModeledMove`; the committed e2e golden replays 220/220 byte-identical (md5
   a23d77ac60d4af168b8a4428f0b465c9 unchanged). ["facade doubles when burned/poisoned/
   paralyzed", "sacred fire thaws its frozen user"]
2. ✅ **FIXED (2026-07-03, `gen3_item_mechanics_v1`) — Pink Bow / Polkadot Bow + odd/rock/rose/
   wave incense** — were in the harness `MODELED_ITEMS` as "×1.1 type-boosters" but ABSENT from
   the port's hardcoded `resolve_atk_stat_mods` match-arm (the motivating drift bug of the
   DATA-DRIVEN MECHANICS FRAMEWORK — see CLAUDE.md "## Data-driven mechanics"). The probe
   settled the REAL math: the bows are a DIRECT `basePower * 1.1` float replace; the incenses
   are `chainModify([4915,4096])` ≈ **×1.2, NOT ×1.1**. The whole item-modifier class is now a
   dex-data lookup (`ItemData.type_boost`/`stat_mods`/`choice`), the species items (Thick Club /
   gen3-SpA-only Light Ball / DeepSea* / Metal Powder / Soul Dew) landed with it, and the class
   is gated by `tests/item_mods_test.rs` (33 scenarios × 30 seeds) + the damage_test max-roll
   probes + 6 revert-verified `IM*` pins. Measured on the smoke_random repro corpus: 31 of the
   87 state-kind divergences whose teams carry a framework item now replay `ok` (the item
   cluster); the remainder still diverge via the OTHER open clusters (1/3-6 below) — a 12-mon
   random team nearly always carries several items, so item-on-team ≠ item-was-the-bug.
3. ✅ **FIXED (2026-07-03, `gen3_accuracy_pipeline_v1`) — the ACCURACY pipeline: acc/eva stages +
   accMod items/abilities folded into the to-hit roll.** Mud-Slap-class acc-drop secondaries are
   admitted (a `boosts` shape) + tracked in `boosts[5]`, but `run_move` used to roll the RAW move
   accuracy → hit/miss flips vs the sim (a SEED-kind divergence, since the accuracy `randomChance` is
   drawn then crit/damage draws follow ONLY on a hit). **ROOT CAUSE + SETTLED MATH (probe
   `harness/probe_accuracy_tohit.js` + `probe_accuracy_intguard.js` over the RESOLVED
   `Dex.mod('gen3')` — the ONLY oracle):** gen3 `tryMoveHit` computes `effAcc = move.accuracy × the
   acc/eva STAGE TABLE [3/3,4/3,5/3,6/3,7/3,8/3,9/3] × the accMod handlers`, then ONE `random(100) <
   effAcc` (an integer 0..99 vs the RAW f64 effAcc — NOT floored). The stages apply inline
   (attacker `boosts[5]` accuracy, defender `boosts[6]` evasion) BEFORE `runEvent('ModifyAccuracy')`,
   which folds the accMod members: Bright Powder ×0.9 / Lax Incense ×0.95 (a DIRECT float that mutates
   relayVar) + Compound Eyes ×1.3 / Sand Veil ×0.8-in-sand / Hustle ×3277/4096-physical (chainModify —
   accumulated into ONE 4096 modifier applied at the END ONLY when `acc` is a non-negative INTEGER, so
   a stage/direct-multiply float SKIPS every chain member — the runEvent integer-guard). The mod-chain
   law bit HARD here: the base `.ts` Bright Powder is `chainModify([3686,4096])` but the gen3 mod
   REWRITES it to `accuracy * 0.9`. **FIX** (`turn.rs::effective_accuracy`/`roll_accuracy` over the
   data-driven `dex/accmod.rs::AccMod`; the empty path is byte-identical to `randomChance(acc,100)`).
   Hustle ships FULLY (its Atk ×1.5 dmgMod — a separate pre-chain `modify` — pairs with the acc ×0.8);
   Sand Veil's `onImmunity('sandstorm')` sand-chip immunity folded into `weather_immune`. **Validated**
   by the class-sweep golden `gen_accuracy_golden.js` → `tests/accuracy_test.rs` (per-decision
   STATE+HP+SEED to game-end) + `effective_accuracy_matches_sim_probe` + the 4 revert-verified pins
   AC1-AC4. **Parity: ~70 / ~132 Mud-Slap acc-stage repros (smoke_random) now replay `ok`** (5/8
   spot-checked diverge again on a stage-fold revert; the remainder progress PAST the acc-drop to
   OTHER-cluster bugs — a random moveset carries several unmodeled moves). The randbats/overnight
   corpus carries no acc-lowering moves so it had 0 acc-stage repros; the e2e stays STRICT 220/220
   byte-unchanged (Bright Powder / Sand Veil / Compound Eyes / Hustle are too rare in the gen3 OU team
   pool to become filter-clean — accuracy is off the taxonomy's gap list).
4. ✅ **FIXED (2026-07-08, `gen3_shielddust_sub_v1`) — the substitute×secondary SEED cluster =
   SHIELD DUST wrongly filtering behind a SUBSTITUTE.** The triage of the NEWEST corpus
   (`harness/ab_fuzz_out/auto_0708_0304/divergences/`, 771 repros vs the CURRENT engine — kinds
   seed=649 / state=117 / panic=4 / status=1) ranked the mechanism histogram: **#1
   sub×secondary seed interleavings (378 distinct repros)** > #2 facade (265 — since ✅ FIXED, §1) >
   other-secondary tail (61) > misc/switch-boundary (50) > panic (4). **ROOT CAUSE (probe
   `harness/probe_sub_break_secondary_rng.js` over the resolved sim — the only oracle):** Shield
   Dust's secondary filter is a TARGET-gathered ModifySecondaries handler; when a SUBSTITUTE
   absorbs the hit the target list is `null`, the filter never gathers, and the secondary
   `random(100)` **STILL DRAWS** — for the move's OWN secondary (probe 2b vs the bare control 2a),
   the Tri Attack 20% gate (3b), AND the King's Rock appended secondary (4b); held AND breaking
   sub identically; the EFFECT stays sub-suppressed (a passing roll statuses nothing — seed
   2,7,1,8). The port filtered a Shield Dust defender UNCONDITIONALLY at all 3 sites
   (`apply_secondaries` / `apply_triattack_secondary` / `apply_kings_rock_secondary`) → one
   missing draw per secondary-into-a-Shield-Dust-sub → every later draw desynced. Randbats is
   saturated with ShieldDust+Substitute carriers (Venomoth's set is literally
   ShieldDust+sub+sludgebomb), which is why this dominated. **FIX** (`turn.rs`): the filter gates
   on `!absorbed_by_sub` at all 3 sites. **Parity: 362 / 777 corpus repros flip to `ok`
   (ShieldDust-team, non-facade repros: 347/365 = 95%); 2 of the 4 panics also clear** (they were
   downstream of the SD desync); a revert re-diverges all 3 spot-checked (ab_1086_0 / ab_1008_15 /
   ab_100_8) and fails the pin. **Pinned** by the revert-verified
   `regression_test.rs::shield_dust_behind_a_substitute_still_draws_the_secondary` (all 3 sites +
   the bare-filter control; ground truth `harness/probe_shielddust_sub_regression_rng.js`). The
   remaining post-fix corpus: facade 281 (§1 — since ✅ FIXED 2026-07-09, `gen3_facade_v1`), a
   ~114-repro non-facade tail (the **sacredfire** part — since ✅ PROBED + FIXED 2026-07-09,
   `gen3_defrost_v1`, see §1: the thaw roll still draws, the failed-roll defrost move proceeds
   + thaws; plus the old switch-boundary + ice-freeze-residue candidates, still open), and 2
   panics (the wish-drag class, §7).
5. ✅ **FIXED (2026-07-03, `gen3_sun_freeze_immunity_v1`) — Ice-move FREEZE gating: SUN blocks
   freeze.** The "ice-freeze cluster" (196 A/B repros, fingerprint `expected=None got=Some(Freeze)`,
   SEED STILL MATCHING) was the port FREEZING a mon the sim leaves un-frozen. **ROOT CAUSE (source
   read + probe-verified):** the base `sunnyday` weather registers `onImmunity(type, pokemon) { if
   effectiveWeather() !== 'sunnyday' return; if type === 'frz' return false; }` (conditions.ts) — so
   while the field weather is Sun (Drought / Sunny Day), a mon CANNOT be frozen. `setStatus` →
   `runStatusImmunity('frz')` → `runEvent('Immunity', 'frz')` returns FALSE, at the SAME
   `runStatusImmunity` position as the type immunity — checked BEFORE `runEvent('SetStatus')` (the
   gen3ou clause shuffle) and DRAW-FREE. So the freeze SECONDARY's `random(100)` still fires (the
   seed matches) but the freeze must simply not land. The minimal A/B repro was a Drought-Groudon
   Ice-Beamed on turn 1 (`ab_460_17`, dec 0). **FIX** (`turn.rs::try_set_status`): a draw-free gate
   `if effect == "frz" && self.field.weather == Some(Weather::Sun) { return; }` right after the
   type-immunity check, before the SetStatus shuffle (an already-FROZEN mon PERSISTS under sun — the
   gate is application-only, `frz` has no weather-cure). **Probe** (`harness/probe_sun_freeze_immunity.js`):
   under Drought the same seed that freezes in no-sun leaves the mon UN-frozen with an IDENTICAL draw
   count (customgame), ONE FEWER in gen3ou (the skipped clause shuffle). **Pinned** by the
   revert-verified `regression_test.rs::sun_blocks_freeze_secondary_draw_free` (ground truth
   `harness/probe_sun_freeze_regression_rng.js`). Parity: **176 / 196 ice-freeze repros now replay
   `ok`** (the remaining 20 progress PAST the freeze to other-cluster bugs 1/3/4/6). This fix CLEARED
   the fuzzer's ice-freeze cluster and is proven by pin FZ1 with **0 e2e decisions** (no filter-clean
   battle carries a sun+ice-move turn — the leech-seed situation). The ability DMG_MOD e2e ADMISSION
   (the 8 pinch/Huge-Pure/Guts/Marvel abilities now UNCOMMENTED in `gen_e2e_fuzz.js`'s
   `MODELED_ABILITIES`, filter-clean teams **22 → 151/719**, STRICT `filtered_diverged == 0` over 220
   battles / 9963 decisions) was actually unblocked by the **`wisp` MOVE-ALIAS fix (§8)** — the one
   cascade the enlarged corpus surfaced. (e2e_86 diverged on `wisp`; that battle runs SandStream, NOT
   Sun, so the sun-freeze gate did not gate the admission — the two fixes are independent.)
6. **Switch-boundary seed cluster** (5 randbats + 6 random repros) — a switching/entry
   draw-order case beyond the e2e team pool.
7. **1 panic (fail-loud WORKING)** — a Whirlwind-drag divergence led a later recorded slot onto
   the port-active's unmodeled Wish; the panic is the symptom, the drag divergence the bug.
8. ✅ **FIXED (2026-07-03, `gen3_move_alias_resolution_v1`) — MOVE-ID ALIAS (`wisp`).** Surfaced by
   the DMG_MOD e2e admission (§5): admitting Torrent grew the corpus to a Swampert battle (e2e_86)
   whose Gengar's packed team spells Will-O-Wisp as the alias **`wisp`**. Showdown resolves move
   aliases at `dex.moves.get()` (via `data/aliases.ts` — `wisp`→`willowisp`, `sd`→`swordsdance`,
   `twave`→`thunderwave`, …) and RUNS the canonical move (drawing its accuracy). The port's dex
   read only the CANONICAL `gen3_moves.json` keys, so `move_at → dex.moves("wisp")` returned `None`
   and `run_move` NO-OP'd the move — drawing NOTHING while the sim ran it (a draw-COUNT desync that
   cascaded the decision boundaries: e2e_86 rust 35 vs golden 41 decisions). **FIX:** the extractor
   emits `data/pokemon/gen3_move_aliases.json` (44 gen3 move aliases from `aliases.ts`, EXCLUDING the
   bare `hp`→`hiddenpower`; of these the **28 non-HP** aliases mirror Showdown byte-for-byte, while the
   **16 typed-HP** aliases [`hpice`→`hiddenpowerice`, …] deliberately resolve to the port's DISTINCT
   typed-HP names rather than Showdown's collapsed `hiddenpower`, per `gen3_typed_hidden_power_ids_v1`),
   and the Rust dex's `moves()` resolves through it. Obs-neutral (the Python `agents.gen3_data`
   facade never loads it; extractor-parity pinned by `test_move_aliases_builder_reproduces_committed`).
   **Pinned** by revert-verified `regression_test.rs::move_alias_wisp_resolves_and_runs_will_o_wisp`
   (ground truth `harness/probe_wisp_alias_regression_rng.js`) + the dex unit test
   `dex::alias_tests::move_aliases_resolve_to_the_canonical_move`.

## ✅ CLEARED — the A/B residual tail (2026-07-10): the auto_0709_0805 corpus replays 307/307 ok

The 2026-07-09 residual map's whole open queue is CLOSED. Re-triaging the now-complete
gender-pinned corpus (`harness/ab_fuzz_out/auto_0709_0805/divergences/`, 307 repros) with the
current binary gave 151 ok / 152 diverged / 4 panic; root-causing the survivors found **SEVEN
real engine bugs** (each probe-settled, fixed, revert-verified-pinned) — after which the corpus
replays **307/307 ok**, including the 4 fail-loud panics (their upstream drag divergences sat
inside the fixed clusters, so the recorded battles now stay on modeled paths to game-end):

1. **PLUS/MINUS cross-field SpA ×1.5** (`gen3_plus_minus_v1` — the thunderbolt-vs-Plusle/Minun
   STATE cluster, 18 repros). The gen3 RESOLVED `onModifySpA` scans `getAllActive()` — FOES
   INCLUDED (gen5+ narrowed it to allies) — so a Minus attacker facing a Plus active is ×1.5
   (paired ability ONLY; plus-vs-plus is nothing; SpA-only; draw-free; live while the partner
   is active). The old NOOP classification's "partner-less in singles → no-op" never tested the
   OPPOSING active. Probe `harness/probe_plus_minus_gen3.js` (maxRoll 90 vs 60 = ×1.5 both
   directions); pin `minus_boosts_spa_when_the_foe_active_has_plus`; ground truth
   `harness/probe_plusminus_ffwisp_regression_rng.js`. `plus`/`minus` stay ADMITTED (now
   modeled, not no-op — the JS allow-list union is unchanged, so the e2e golden is untouched).
2. **Will-O-Wisp into FLASH FIRE is ABSORBED** (`gen3_ff_wisp_absorb_v1` — the willowisp STATE
   cluster, incl. a TRACED FF on Porygon2). The resolved `flashfire.onTryHit` absorbs a landed
   WoW on a NON-Fire, status-free, un-subbed holder (the volatile ARMS, no burn); Fire-type /
   statused / subbed targets fall through to the normal gates (already modeled). The port
   burned it → a maxhp/8 DoT desync per residual. Probe `probe_flashfire_rng.js` A3 (was
   already measured — the status-arm wiring was the gap); pin
   `will_o_wisp_into_flash_fire_is_absorbed`.
3. **Cloud Nine / Air Lock `onEnd` → `eachEvent('WeatherChange')`** (`gen3_cloudnine_end_v1` —
   the dominant seed cluster, masquerading as the "icebeam tail": randbats Golduck-L81 mirrors).
   The resolved onEnd fires WeatherChange UNCONDITIONALLY (weather or not) at BOTH End sites:
   `switchIn`'s alive-outgoing ability End (pre-swap — voluntary pivot + drag; a fainted mon's
   replacement skips it) AND `faintMessages`' ability End (BEFORE `fainted = true`, so the
   dying holder still gathers) — one tie-shuffle iff the actives tie on cached speed. Pins
   `cloud_nine_switch_out_fires_the_weatherchange_shuffle` +
   `cloud_nine_faint_fires_the_weatherchange_shuffle`; ground truth
   `harness/probe_residual_tail_regression_rng.js`.
4. **A FROZEN Flash Fire holder is NOT fire-immune** (`gen3_ff_frozen_no_absorb_v1` — the
   flamethrower/fireblast/firepunch seed tail). `flashfire.onTryHit`'s
   `if (target.status === "frz") return` lets the Fire move proceed with FULL draws (and its
   fire-move thaw then cures the freeze); the port kept the frozen holder immune →
   accuracy-only + a phantom thaw roll on its own move (a 3-vs-9-draw desync, ab_1309_23). Pin
   `frozen_flash_fire_holder_is_not_fire_immune`.
5. **`checkFainted` sets `status = "fnt"`** (`gen3_fnt_clears_status_v1`, part 1 — the
   switch-boundary cluster). The corpse's gen3 `getActionSpeed()` no longer folds paralysis, so
   a fainted formerly-para'd Muk TIES its mirror in the replacement instaswitch sort (the
   ab_1182_15 missing draw). Port: `check_fainted` clears the status to `None`. (This re-meaned
   the NC2 pin `natural_cure_is_a_no_op_on_a_faint` — the old `Some(Burn)`-persists assertion
   pinned a port-internal representation the sim never held; NC1/NC3 keep the live-cure teeth.)
6. **`faintMessages → clearVolatile` ZEROES the corpse's boosts** (`gen3_fnt_clears_status_v1`,
   part 2). A +6-Agility Metagross corpse must sort the replacement at its PLAIN speed (tying a
   plain Swalot corpse → the shuffle draw), not a stale ×4 (the ab_806_16 missing draw —
   localized by temporary port instrumentation showing insta keys 580 vs 145). Port:
   `process_faints` zeroes `boosts`. Both parts pinned by
   `fainted_replacement_sort_clears_status_and_boosts` (revert-fails on EITHER component).
7. **The STATUS_IMMUNE `onUpdate` CURES** (`gen3_statusimmune_onupdate_cure_v1` — the status
   cluster, 9 repros, ALL Trace-Porygon2 boards). Each of the 6 resolved STATUS_IMMUNE members
   (Insomnia/Vital Spirit slp, Limber par, Immunity psn+tox, Water Veil brn, Magma Armor frz)
   carries an `onUpdate` that cures the holder's matching status — unreachable with the mon's
   own ability, but a slept Porygon2 that re-enters and TRACES Insomnia is cured at the first
   Update (draw-free). Port: `status_immune_on_update` at the Update sites (ability-before-item,
   ahead of the berry check). Pin `traced_status_immune_ability_cures_the_status_on_update`.

**Validation:** full suite green (273 passed / 0 failed, `--test-threads=1`); the committed e2e
golden UNTOUCHED (md5 `a23d77ac60d4af168b8a4428f0b465c9` — none of the seven conditions occur
in the 220-battle corpus, which is exactly why the strict gate never caught them); every pin
revert-verified (each fix reverted → its pin FAILS → restore byte-identical). The older
corpora are secondary signal only: auto_0708_1705 replays 332 ok / 157 diverged / 1130 panic —
but the panics are PRE-GENDER-PINNING noise (the batch-4 attract compare fail-louds on the
UNSPECIFIED genders those older teams carry), not engine bugs; only gender-pinned corpora
(auto_0709_0805 onward) are honest replay material.

**The remaining map (honest):** NOTHING remains open from auto_0709_0805. The live
auto_0709_2205 run (the first full run on the facade+defrost engine, before this session's
seven fixes) shows ~0.8% divergence at its own tally; its saved repros should be re-triaged
with THIS binary next session (expect most to be the seven clusters above; anything surviving
is genuinely new).

## ✅ CLEARED — A/B fix-queue #4 (2026-07-10): the auto_0709_2205 steady-state 9-repro corpus replays 9/9 ok

The first all-fixes 12h run's tally was **9 divergences / 35,018 battles (0.026%, 0 panics)** —
and every survivor was a genuinely new bug. Re-triage with the residual-tail binary: 4/9 already
ok (mid-run-fix noise: ab_13_22, ab_20_10, ab_21_24, ab_2_13 — the seven residual-tail clusters
landed while the run was writing). Root-causing the 5 true survivors found **THREE engine bugs**
(probe-settled via the NEW generic sim-side repro tracer `harness/probe_repro_simtrace.js` —
replay any repro dir through the REAL sim with per-draw call-site instrumentation — plus targeted
protocol/state probes); after the fixes the corpus replays **9/9 ok** (and auto_0709_0805 stays
307/307):

1. **`faintQueue` ENQUEUE order is DRAW-BEARING** (`gen3_faint_queue_order_v1` — ab_723_13 +
   ab_464_16, seed@the-explosion-turn). `faintMessages` drains the queue in enqueue order,
   fully processing each corpse (`fainted=true`, `isActive=false`) BEFORE the next corpse's
   ability-`End` — so on a mutual Explosion the USER (self-KO'd first in `useMove`) is already
   inactive when the Cloud Nine TARGET's `onEnd → eachEvent('WeatherChange')` fires: the dying
   holder gathers ALONE → NO tie-shuffle even on a cached-speed tie (Golduck-L81 184 ==
   Smeargle-L89 184). The port walked SIDE order when logging was off (`faint_emit_queue` was
   emission-only) → processed the side-0 CN corpse first → a phantom draw. Fix:
   `record_faint_order` is unconditional; `process_faints` always drains enqueue-order. Pin
   `double_faint_processes_corpses_in_enqueue_order`.
2. **A FAINTED mon's ability handlers no longer gather — Swift Swim corpse sorts PLAIN**
   (`gen3_fainted_no_ability_speed_v1` — ab_894_12, seed@the-double-replacement). Probe: rain
   up, Kingdra-L81 `getActionSpeed()` alive 368 → fainted 184, TYING the 184 Smeargle-L89
   corpse → the instaswitch shuffle draw the port missed (it applied `weather_speed` ×2 to the
   corpse). Fix: `effective_speed` gates the weather-speed ability mod on `!mon.fainted` (the
   para analogue was already `check_fainted`'s status-`fnt` clear). Pin
   `fainted_swift_swim_corpse_sorts_at_plain_speed`.
3. **The gen3 TOXIC-stage reset lives in the runSwitch's `SwitchIn` event — and is therefore
   SKIPPED when that runSwitch is CANCELLED** (`gen3_tox_stage_persists_v1` — ab_403_13
   state@46 + ab_1166_22 seed@38, and the fix-queue-#3 Lens-2 lead rmrcqwc2c_ab_793_13 in
   auto_0708_1705, state@38 hp 81-vs-97). The resolved `tox.onSwitchIn(){stage=0}` fires via
   the gen4-override `runSwitch`'s `runEvent('SwitchIn')` (mods/gen4/scripts.js:42) — the port
   reset at `execute_switch`'s array swap instead, which the gen3 faint-cancels-all rule never
   guards. THE LAW (three sim observations): a voluntary out-and-back RESETS
   (`probe_tox_stage_switch.js`: residuals 22 then 44, `statusState.stage` 1→2); a forced
   replacement whose runSwitch RUNS also resets (the 403 Umbreon: re-entered at stage 4, the
   resumed residual dealt 19 = stage 1); a replacement whose queued runSwitch is CANCELLED by
   its co-replacement's Spikes-faint KEEPS the stage (the 1166 Mew: re-entered tox at 13/263,
   heal +16 → 29, chip 32 = stage 2 → KO — where the port's reset left it alive at 13 and then
   rolled a phantom endTurn Quick Claw). Fix: the reset moved into `run_switch` (after
   EntryHazard, the source order). Pins `tox_stage_resets_when_the_runswitch_runs` +
   `tox_stage_persists_when_the_runswitch_is_cancelled` (the second pins the PLACEMENT — it
   fails under reset-at-execute_switch).

**The Lens-2 lead verdict:** REAL BUG, fixed — it was bug 3. Replaying the whole
auto_0708_1705 corpus on the fixed binary: **489 ok / 0 diverged / 1130 panic** (the panics
stay the known pre-gender-pinning attract fail-loud noise); re-adding the old
`execute_switch` reset reproduces the exact original fingerprint (state@38, hp 81 vs 97) on
rmrcqwc2c_ab_793_13, and the restored fix flips it ok — NOT gender noise.

**Validation:** ground truth `harness/probe_fixqueue4_regression_rng.js` (+
`probe_tox_stage_switch.js`, `probe_repro_simtrace.js`); full suite green (**277 passed /
0 failed**, `--test-threads=1`); the committed e2e golden UNTOUCHED (md5
`a23d77ac60d4af168b8a4428f0b465c9`); all four pins revert-verified (each fix reverted → its
pin FAILS → restored). **The remaining map: NOTHING open** across auto_0709_0805 (307/307),
auto_0709_2205 (9/9), and auto_0708_1705 (0 diverged); the next honest signal is the live
auto_0710_1305 run's fresh repros.

## ✅ FIXED ×2 — the HANDLER-COMPLETENESS AUDIT's first run (`gen3_handler_audit_v1`, 2026-07-10)

The audit (CLAUDE.md "## Data-driven mechanics" → "### Handler-completeness audit":
`harness/dump_gen3_handlers.js --audit` + the committed manifest
`tests/vectors/gen3_handler_audit.json`, wired into cargo via `tests/handler_audit_test.rs`)
enumerated every resolved handler key on the 664-row reachable surface (74 abilities / 59 items /
27 conditions / 168 modeled moves; census 595 implemented / 39 noop_justified /
30 unreachable_justified) and surfaced **two real latent bugs** — both of the exact "a handler at
a hook we never enumerated" class this file documents (Immunity onUpdate / Cloud Nine onEnd /
Plus-Minus / facade / tox onSwitchIn):

1. **Jump Kick / High Jump Kick crash damage MISSING** (`gen3_jump_kick_crash_v1`). Both moves
   pass `isModeledMove` (plain damaging, no secondary) but the resolved gen3 dist gives them an
   `onMoveFail`: a FAILED JK/HJK (an accuracy MISS, or a **Protect block**) crashes the USER for
   `clampIntRange(getDamage(source,target,move,true) / 2, 1, floor(TARGET.maxhp / 2))` — and that
   `getDamage` DRAWS the crit `randomChance` + the `random(16)` damage roll (a missed JK is
   exactly **+2 draws** vs a missed control move), so the port's silent no-op was BOTH an HP and
   a seed desync. Probe-settled (`harness/probe_jumpkick_crash_rng.js`): the clamp ceiling is the
   TARGET's maxhp/2 (crash 125 > the user's own 120 in probe A); the crash fires through Protect;
   a Fighting-immune (Ghost) target crashes nothing and draws nothing beyond accuracy (the
   `runImmunity("Fighting")` gate — the port's `-immune` short-circuit path matches); a
   sub-ABSORBED hit is a hit (no crash) while a miss vs a sub holder crashes; the crash can FAINT
   the user (and the deciding-faint Quick Claw skip then applies); the crash is a MOVE-effect
   Damage event (Focus Band rolls + can survive it). FIX: `turn.rs::apply_jump_kick_crash`
   (id-gated per the facade precedent) called at the genuine-miss + protect-block returns.
   PINS (sim ground truth `harness/probe_handler_audit_regression_rng.js`, revert-verified):
   `jump_kick_miss_crashes_the_user_with_crit_and_roll_draws` (HA1, exact crash HP + post-turn
   seed) + `jump_kick_crashes_through_a_protect_block` (HA1b). Exposure: ZERO — no e2e pool team
   and no A/B repro carries either move (the committed e2e golden is untouched, md5
   `a23d77ac60d4af168b8a4428f0b465c9`); a pure latent bug the static gate caught.

2. **Freeze Clause Mod MISSING** (`gen3_freeze_clause_v1`). The engine modeled Sleep Clause
   (the 2-clause SetStatus shuffle + the slp block) but never the freeze half: under gen3ou a
   SECOND foe-inflicted freeze on the same side must FAIL (`freezeclausemod.onSetStatus` returns
   false INSIDE the SetStatus event whose shuffle already drew → the block is DRAW-FREE — probe
   `harness/probe_freeze_clause_rng.js`: a blocked second freeze's turn draw count == a landed
   freeze's). A FAINTED frozen mon does NOT count — the sim sets `status = 'fnt'` on faint, so
   the rule's `pokemon.status === "frz"` scan only ever matches living mons (probe B). FIX:
   `turn.rs::try_set_status` frz gate + `side_has_frozen` (mirrors the sleep path; the same
   `sleep_clause` format flag — gen3's Standard ruleset carries both clauses). PIN
   (revert-verified): `freeze_clause_blocks_the_second_freeze_in_gen3ou` (HA2 — a scripted
   gen3ou freeze→switch→blocked-freeze with the sim's exact final seed). Exposure: unreachable
   in the e2e AND the A/B fuzzer (both run gen3customgame, clause-free) — a latent bug for
   every clause format (gen3ou etc.).

The audit also CONFIRMED (as `implemented`, with grep-verified anchors) several rows that were
past members of this bug class — the fire-hit thaw, Early Bird's double sleep decrement,
Pressure's +1 PP deduction, the Trace-route STATUS_IMMUNE `onUpdate` cures — and JUSTIFIED the
no-ops with the resolved source in hand (e.g. Rock Head is a true no-op ONLY because the
resolved handler exempts Struggle, the port's sole recoil source; the heal-berries'
`onTryEatItem` TryHeal guard is vacuous in the modeled universe). Any future dist change that
touches a fingerprinted body, adds a handler to a surface effect, or renames a modeling site
now FAILS `cargo test` until re-triaged.

## Notes
- Some of these remain EXCLUDED from the e2e fuzz `isModeledMove` / the modeled ability+item filter,
  so the strict `filtered_diverged == 0` gate is unaffected — but they keep real teams off the modeled
  path. Filter-clean teams: **525/719** after `gen3_status_immune_v1` admitted the STATUS_IMMUNE class
  (was 449 after `gen3_natural_cure_v1` admitted Natural Cure; 151 after `gen3_sun_freeze_immunity_v1`
  admitted the 8 DMG_MOD abilities; 22 before that; 18 before `gen3_trapping_v1` admitted
  arenatrap/magnetpull). Admitting the STATUS_IMMUNE class grew it **+76** (immunity=97, the #2 team-carry
  gap); Natural Cure was the biggest single lever (naturalcure=254). The remaining blockers are now
  **Shell Armor (=39) / Synchronize / Effect Spore / Trace + Lum/Salac berries** (see the taxonomy's
  top-gaps list — Natural Cure AND Immunity have both dropped off it).
- When each is built: add a dedicated, named, DETERMINISTIC regression test (constructed scenario +
  fixed seed) that FAILS if the fix is reverted — and for trapping, a test that the request/LegalActions
  correctly REJECTS the illegal switch.

## ✅ PROTOCOL PHASE 3 + write_line — three real gaps the new byte gates caught (FIXED 2026-07-10)

`gen3_protocol_phase3_v1` (8 new capture scenarios → protocol byte-diff 66/8721 → **114/16115**) and
`gen3_writeline_stream_v1` (the per-write `BattleStream::write_line` gate, 38 battles / 1722 writes)
surfaced three latent divergences no earlier gate could reach:

1. **Per-side choice acceptance (split-accept boundaries).** The sim's `side.choose` is PER-SIDE: at a
   `move` request one side's valid choice is ACCEPTED-and-HELD while the other side's invalid choice
   (e.g. a switch to a fainted slot) is rejected, and a later re-submission by the already-chosen side
   is DISCARDED — so a turn can commit with choices accepted at DIFFERENT submissions
   (`midswitch_ability_lines/2`: p2's `move 2` was held from the rejected decision; the turn ran it
   with p1's NEXT choice). The port's old whole-decision skip mis-mapped that split. FIX: a per-side
   pending-choice accumulator in `run_full_battle` (+ the same accumulation in the forced-replacement
   pull, so a double replacement can arrive as two one-sided writes). Zero-draw (boundary MAPPING
   only); no pre-Phase-3 golden script contains a split-accept, so every seed suite stayed
   byte-identical. Pin: `per_side_choice_acceptance_maps_split_accept_boundaries_to_the_sims_seeds`
   (replays the capture's DEC rows at SEED level — revert-verified).
2. **Switch-to-a-fainted-slot reject.** A scripted voluntary `switch N` naming a fainted /
   out-of-range / already-active slot must be SKIPPED draw-free (the sim's `chooseSwitch` "can't
   switch to a fainted Pokémon") — the port executed it (a `|switch|…|0 fnt` phantom entrant). FIX:
   the invalid-target gate in `choice_is_legal`. Pin:
   `rejected_switch_to_a_fainted_slot_is_skipped_draw_free` (revert-verified).
3. **Rest-at-full-HP `|-fail|<user>|heal`.** The sim's heal-fail carries the `heal` detail token; the
   port emitted a bare `|-fail|<user>`. Found by the per-write gate (recover_and_rest/1's write 68 —
   a woken Snorlax Rested at full HP deep in a truncated battle; the Phase-2 corpus never realized
   the branch). Emission-only; pinned by `protocol.rs::phase3_forms_and_the_miss_retro_edit` + the
   writeline golden.

Also settled by capture (not bugs, but previously-unverifiable forms now byte-pinned): the re-Taunt /
no-lastMove-Disable / already-disabled-Disable fails are `attrLastMove('[still]')` + `-fail` on the
USER (the port's old fail-on-target re-Taunt form was wrong and never gated); a missed STATUS move's
announce gains the `[miss]` attr; gen3 Intimidate vs a subbed foe emits ONLY the `|-hint|` line (no
`-ability`); a fainted armed-Flash-Fire holder emits NO `-end` (only an alive switch-out does); the
Leech Seed residual's `[of]` names the seeder side's CURRENT active. The `|turn|N+1` marker + batch
separator/`|t:|` attribution: the marker flushes at turn END (eager), the separator+`|t:|` with the
COMMITTING write — concatenated order unchanged (the protocol golden pins it), chunks now correct for
the streaming surface.
