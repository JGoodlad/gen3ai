# BATCH 8 + 9 research — draw models + gen3-resolved mechanics (probe-settled)

Pure investigation for the `src/rust_sim` port. **No engine file was touched.** Every claim below
is confirmed against the OMNISCIENT in-process `BattleStream` (`Dex.mod('gen3')` via
`Dex.forFormat('gen3customgame')`, no server) — the sim is the oracle, not the source read. The
`Dex.forFormat` dump was verified to correctly resolve the mod chain (it reproduced the gen4
`partiallytrapped`/`wonderguard`/`liquidooze`/`forecast` overrides byte-for-byte), so the
dexfacts layer is trustworthy as the hypothesis source; the live probes settle the draw counts.

Probe scripts (kept under `harness/`, reusable by the impl phase):
- `probe_batch89_dexfacts.js` — resolved gen3 handlers/metadata for every item below.
- `probe_batch89_haze_trick_yawn.js` + `probe_batch89_trick_edges.js` — Batch-8 moves.
- `probe_batch89_trap.js` — the partial-trap family (full lifecycle + duration distribution).
- `probe_batch89_transform.js` — Transform (draw-freeness, copy list, revert, fails).
- `probe_batch89_abilities_items.js` — Wonder Guard / Liquid Ooze / White Herb / Stick.
- `probe_batch89_forecast_cloudnine.js` — Forecast Cloud-Nine/Air-Lock composition + reporting.
- (existing) `probe_forecast_rng.js` — the pre-existing Forecast map/revert/draw-freeness probe.

**Baseline draw note.** Every gen3customgame turn draws ONE `randomChance(1,5)` (internally one
`random(5)`) at `endTurn` regardless of items — this is the port's already-modeled per-turn
end-of-turn roll. In every trace below, "the trap/move adds N draws" means N draws BEYOND this
baseline. Speed-tie `random(0,2)@speedSort` shuffles are the already-modeled eachEvent/action
tie-shuffles; they appear when a mechanic changes a mon's speed (Transform) but are NOT new draws.

---

# BATCH 8

## HAZE (`haze`, num 114) — full boost reset

**Resolved mechanic** (gen3 override, `data/mods/gen3/moves.ts`): a Status move, `type Ice`,
`accuracy: true`, `target: 'all'`, `priority 0`, `ignoreImmunity: true`. gen3 `onHitField`:
```
this.add('-clearallboost');
for (const pokemon of this.getAllActive()) pokemon.clearBoosts();
```
It is a **MOVE**, resolved in the action queue at the user's speed slot — **NOT a residual**
(confirmed: the cast turn resolved Splash then Haze in-turn, before endTurn). `getAllActive()`
clears **BOTH sides' actives, including the user itself** (probe: Snorlax's own +4 Atk was wiped).

**Draw model — ZERO draws.** `accuracy: true` ⇒ no accuracy roll; `clearBoosts()` is draw-free.
Cast turn had draws=1 (baseline endTurn only), identical to a Splash turn.
```
HAZE t3: draws=1 calls=[random(5)@endTurn ... (baseline only)]
 lines=["|move|p1a: Snorlax|Haze|p1a: Snorlax","|-clearallboost", ...]
 state: p1boosts all 0, p2boosts all 0
```

**State:** none new (reuses existing per-mon boost stages).

**Emission (byte-exact):** `|move|USER|Haze|USER` then a single `|-clearallboost|` line. There are
**NO per-mon `-clearboost` lines** — just the one field line; both sides' stages are zeroed silently.

**Gotchas:** clears the USER's own boosts too. Ignores type immunity for targeting (it's a field
effect). No interaction with Substitute (field-level). Trivial to model.

---

## TRICK (`trick`, num 271) — item swap. `switcheroo` is NOT gen3-legal.

**`switcheroo` (num 415) is a gen4 move** — no gen3 mon learns it; it stays out of the modeled
universe. Only `trick` matters.

**Resolved mechanic** (base, inherited unchanged into gen3): Status, `type Psychic`,
`accuracy: 100`, `target: 'normal'`, `ignoreImmunity: true`, flags `{protect, mirror, allyanim,
noassist, failcopycat}` — **NO `bypasssub`**. `onTryImmunity(target) => !target.hasAbility('stickyhold')`.
`onHit` swaps both items via `target.takeItem(source)` / `source.takeItem()`, failing (return false,
no swap) only when **both** would-be items are falsy or a `takeItem` returns false.

**Draw model — ONE accuracy draw (acc 100), then draw-free swap.** The swap adds nothing.
```
TRICK-swap t1: draws=2 = [random(100)+randomChance(100,100) accuracy] + baseline
 lines=["|move|p1a: Alakazam|Trick|p2a: Snorlax",
        "|-activate|p1a: Alakazam|move: Trick|[of] p2a: Snorlax",
        "|-item|p2a: Snorlax|Choice Band|[from] move: Trick",
        "|-item|p1a: Alakazam|Leftovers|[from] move: Trick"]
```

**Fail / block conditions (all probe-settled):**
- **Sticky Hold on target** → `onTryImmunity` → accuracy STILL drawn, then `|-immune|p2a: Muk`
  (plain `-immune`, **no `[from] ability` tag**). No swap.
- **Substitute on target** (no `bypasssub`) → accuracy drawn, then blocked: `|move|USER|Trick||[still]`
  + `|-fail|USER`, no swap. (`probe_batch89_trick_edges.js` — the earlier "empty output" was a probe
  bug: a Choice Band had locked the user into an earlier move, not a sim behavior.)
- **Both sides itemless** → accuracy drawn, then `|move|USER|Trick||[still]` + `|-fail|USER`.
- **Exactly one side has an item** → swaps anyway; the itemless loser shows
  `|-enditem|LOSER|Item|[silent]|[from] move: Trick` and the receiver `|-item|USER|Item|[from] move: Trick`.
- **Mail is SWAPPABLE in gen3** (task hypothesis was wrong): `|-item|p2a: Snorlax|Choice Band` +
  `|-item|p1a: Alakazam|beadmail` — Mail trades fine, both directions.
- **Berries are SWAPPABLE** (Lum Berry swapped fine). So in gen3, `takeItem` effectively never
  returns false except the Sticky Hold case (handled by `onTryImmunity`) — there are no Mega
  Stones / Z-crystals / Drives / Plates in gen3, and `gen<=4` `itemKnockedOff` is the only other
  `takeItem` gate (a Knock-Off'd item, edge).

**Emission order:** `|move|USER|Trick|TARGET`, `|-activate|USER|move: Trick|[of] TARGET`, then the
TARGET's new item line first, then the USER's new item line. Loser of a one-sided swap: `-enditem
[silent]`.

**State:** none new (reuses the per-mon `item` slot). Note: a Choice mon that Tricks away its Choice
Band is a live interaction with the existing choice-lock model (out of scope to re-derive here).

---

## YAWN (`yawn`, num 281) — delayed sleep

**Resolved mechanic** (base, unchanged into gen3): Status, `type Normal`, `accuracy: true`,
`target: 'normal'`, `volatileStatus: 'yawn'`, flags `{protect, reflectable, mirror, metronome}`.
`onTryHit(target)`: fails if `target.status || !target.runStatusImmunity('slp')`. The `yawn`
condition: `duration: 2`, `onResidualOrder: 10`, `onResidualSubOrder: 19`;
`onStart` → `|-start|TARGET|move: Yawn|[of] SOURCE`;
`onEnd(target)` → `|-end|TARGET|move: Yawn|[silent]` then `target.trySetStatus('slp', source)`.

**Draw model — the crux: the sleep `random(2,6)` fires at RESOLVE, not at cast.**
- **Cast (t1): ZERO draws.** `accuracy: true` ⇒ no accuracy roll; the volatile add is draw-free.
  ```
  YAWN t1: draws=1 (baseline only)
   lines=["|move|p1a: Snorlax|Yawn|p2a: Blissey","|-start|p2a: Blissey|move: Yawn|[of] p1a: Snorlax"]
   state: p2vol=["yawn"], p2status=""
  ```
- **Resolve (end of the turn AFTER cast, t2): ONE `random(2,6)` draw** — the sleep-duration roll,
  inside the `slp.onStart` reached via the yawn `onEnd`'s `trySetStatus`.
  ```
  YAWN t2: draws=2 = [random(2,6)@onStart=3 sleep-duration] + baseline
   lines=[... end of turn:] "|-end|p2a: Blissey|move: Yawn|[silent]","|-status|p2a: Blissey|slp"
   state: p2status="slp", slpTime=3
  ```
  `duration: 2` ⇒ cast mid-t1, decrements at end-t1 (2→1) and end-t2 (1→0 → onEnd fires). So sleep
  lands at the END of the turn after cast.

**Fail conditions (both draw-free):**
- **Target already has a major status** → `onTryHit` fail: `|move|USER|Yawn||[still]` + `|-fail|USER`,
  NO volatile, zero draws (probed vs a paralyzed target).
- **Target is sleep-immune** (Insomnia / Vital Spirit / already asleep / Safeguard via
  `runStatusImmunity('slp')`) → `|-immune|TARGET|[from] ability: Vital Spirit`, NO volatile,
  zero draws.

**State:** a `yawn` volatile with a duration-2 countdown + the source uid (for the `[of]` byte and
`trySetStatus` source). It registers a residual DURATION handler at `(onResidualOrder 10,
onResidualSubOrder 19)` — participates in the residual tie-shuffle at that (order, subOrder).

**Emission:** cast `|-start|TARGET|move: Yawn|[of] SOURCE`; resolve `|-end|TARGET|move: Yawn|[silent]`
then `|-status|TARGET|slp` (a normal, non-`[from] move:` status line — the source effect is the
`yawn` condition, whose `effectType` is not `Move`, so `slp.onStart` takes the plain `-status` branch).

**Gotchas:**
- The resolve `trySetStatus('slp')` re-runs the full sleep gate — if the target got statused between
  cast and resolve, the `-end [silent]` still fires but no sleep sets (draw-free in that case).
- **gen3ou Sleep Clause**: the resolve routes through the SAME `try_set_status`/SetStatus path as
  any sleep, so the port must fold Yawn's resolve into `try_set_status` (the Sleep-Clause block +
  the format's `set_status_event_shuffle` apply here). A yawn that would be the 2nd sleep FAILS at
  resolve under gen3ou (not probed here, but it's the existing sleep machinery — route it through
  `try_set_status`, don't special-case).

---

## PARTIAL-TRAP family — `wrap`/`bind`/`firespin`/`clamp`/`sandtomb`/`whirlpool`

The full gen3-legal set (all `volatileStatus: 'partiallytrapped'`, `ignoreImmunity: false`, i.e.
normal type effectiveness applies):

| move | num | type | cat | BP | acc |
|---|---|---|---|---|---|
| wrap | 35 | Normal | Physical (contact) | 15 | 85 |
| bind | 20 | Normal | Physical (contact) | 15 | 75 |
| firespin | 83 | Fire | Special | 15 | 70 |
| clamp | 128 | Water | Special (contact) | 35 | 75 |
| sandtomb | 328 | Ground | Physical | 15 | 70 |
| whirlpool | 250 | Water | Special | 15 | 70 |

**Resolved `partiallytrapped` condition — the mod-chain crux.** The base condition is SHADOWED by
the **gen4 override** (`data/mods/gen4/conditions.ts:110`), which gen3 inherits:
```
durationCallback(target, source) { if (source.hasItem('gripclaw')) return 6; return this.random(3, 7); }
onResidualOrder: 10, onResidualSubOrder: 9
```
Base `onStart` (inherited): `boundDivisor = source.hasItem('bindingband') ? 8 : 16` — **gen3 has no
Grip Claw / Binding Band, so always 16**.
Base `onResidual`: if the trapper is gone (`!trapper.isActive || trapper.hp <= 0 || !trapper.activeTurns`)
→ delete the volatile + silent `-end`; else `this.damage(pokemon.baseMaxhp / boundDivisor)` (= maxhp/16).
Base `onTrapPokemon`: `if (this.effectState.source?.isActive) pokemon.tryTrap()` → **firm `trapped = true`**.

**Draw model — the bit-for-bit crux (probe `probe_batch89_trap.js`):**
- **CAST**: the normal damaging-move draw chain (accuracy [per-move acc above] → crit → damage roll,
  plus any secondary — none for these) **+ ONE `random(3,7)` duration draw** in `durationCallback`,
  fired AFTER the damage lands (the volatile is added post-hit):
  ```
  WRAP t1 (bp15 phys contact, crit landed this seed): draws=5 =
    [random(100)+randomChance(85,100) accuracy]
    [random(16)+randomChance(1,16) crit]
    [random(16) damage]
    [random(3,7)@durationCallback=5 DURATION]         <-- the one trap-specific draw
    + baseline endTurn
   lines cast: "|move|USER|Wrap|TARGET","-crit","-damage",
               "|-activate|TARGET|move: Wrap|[of] USER"
  ```
- **RESIDUAL (each subsequent end-of-turn while trapped): DRAW-FREE** chip `floor(maxhp/16)` at
  `(onResidualOrder 10, onResidualSubOrder 9)`:
  ```
  WRAP t2..: draws=1 (baseline only); chip "|-damage|TARGET|HP|[from] move: Wrap|[partiallytrapped]"
  ```
- **DURATION SEMANTICS — chip turns = `random(3,7) − 1`** (the cast turn's own end also chips once).
  Over 120 seeds:
  ```
  random(3,7) distribution: {3:25, 4:26, 5:25, 6:27}   (uniform over {3,4,5,6})
  chip-turn count:          {2:25, 3:26, 4:25, 5:27}   (== duration−1 → 2..5 chip turns)
  ```
  So it is a **single `random(3,7)` next()-call** (NOT a `sample([...])` and NOT a `random(8)`),
  uniform over 4 values → 2–5 chip turns. This matches gen3 folklore ("2–5 turns") once you count
  chip turns = duration − 1.

- **RELEASE on trapper leaving** (probed): the turn the trapper switches out, the onResidual sees
  `!trapper.isActive` → deletes the volatile + `|-end|TARGET|Wrap|[partiallytrapped]|[silent]`, and
  **no chip that turn**. A trapper faint or `activeTurns == 0` (switched in this turn) releases the
  same way.
- **SWITCH-BLOCK** (probed): while trapped, a voluntary `switch` by the trapped mon is **REJECTED
  draw-free** (state stays on the trapped mon, seed unchanged) — the SAME `pokemon.trapped`
  truthiness gate as Arena Trap/Magnet Pull. Because `onTrapPokemon` calls `tryTrap()` (no `isHidden`
  arg) it is a **FIRM `trapped: true`** trap (the Shadow-Tag request shape: `trapped:true` on the
  first request, a rejected switch → `[Invalid choice]` with NO re-request), not the ability traps'
  `maybeTrapped`.

**State (new):** a per-mon `partial_trap: Option<{ source_uid, move_id, turns_left }>` (boundDivisor
is always 16 in gen3 — no need to store). Needs:
- `move_id` (Wrap/Bind/…) for the `-activate`/`-damage`/`-end` byte form `move: <Name>`.
- `source_uid` to check the trapper's `isActive`/`hp`/`activeTurns` each residual for the release.
- `turns_left` from the `random(3,7)` draw.
The switch-legality gate (`is_trapped` / the firm-trap request shape) must fold this volatile (source
active), and it Baton-Passes like a normal trap link would NOT (partiallytrapped is not passed).

**Emission (byte-exact):**
- cast: `|-activate|TARGET|move: Wrap|[of] USER` (after the `-damage`).
- chip: `|-damage|TARGET|HP|[from] move: Wrap|[partiallytrapped]`.
- natural expiry: `|-end|TARGET|Wrap|[partiallytrapped]` (note: `-end` uses the bare move name
  `Wrap`, NOT `move: Wrap`).
- trapper-left expiry: `|-end|TARGET|Wrap|[partiallytrapped]|[silent]`.

**Gotchas:**
- The `(10, 9)` residual subOrder is distinct from Leftovers (sub 4), Leech Seed, etc., so no tie
  normally; a **mutual wrap** (both sides trapped) puts two `(10,9)` handlers on tied speed → ONE
  extra residual tie-shuffle draw (the same class as the other residual duration handlers). Edge.
- The chip is a normal `this.damage()` — Focus Band can survive a chip-KO (onDamage), and a chip can
  faint (deferred-faint protocol). No Liquid-Ooze / drain interaction (it's not a heal).
- Type effectiveness applies at cast (e.g. Sand Tomb→Flying immune, Wrap/Bind→Ghost immune, Fire
  Spin→Fire resisted) — standard damaging-move immunity, accuracy drawn then `-immune`, no trap.
- Ghost is NOT immune to the trap itself once a non-Normal trapper connects (e.g. Fire Spin traps a
  Ghost) — the trap follows the damaging hit.

---

# BATCH 9

## TRANSFORM (`transform`, num 144) — copy the target

**Resolved mechanic** (base move, inherited): Status, `type Normal`, `accuracy: true`,
flags `{bypasssub, metronome, failencore}`, `onHit → pokemon.transformInto(target)` (return false →
`[still]` + `-fail`). The copy is `sim/pokemon.ts::transformInto`. **gen3-relevant branches (gen == 3):**

**Draw model — ZERO draws from the copy.** `transformInto` contains no `this.random` anywhere.
Confirmed vs a Splash control on the same seed — the copy itself adds nothing:
```
transform: draws = 3×random(0,2)@speedSort + baseline
splash:    draws = baseline only
```
The 3 `speedSort` draws are NOT transform draws — they are the already-modeled per-turn
eachEvent/action **speed-tie shuffles** that fire because Transform **copies the target's Speed
stat**, so in a 1v1 the two actives become speed-TIED and the existing tie machinery draws. The port
gets these for free once it copies the Speed stat correctly. **Transform is draw-free.**

**What is copied (gen3, authoritative from `transformInto` + confirmed by state reads):**
- `species` (setSpecies) — the reported species id becomes the target's forme (e.g. `snorlax`).
- `transformed = true`; `weighthg` (for Low Kick weight).
- `types` (target's `getTypes`) + `addedType`.
- **ALL stored stats EXCEPT HP** (the loop is over `storedStats`, which excludes hp). Ditto keeps its
  own HP/maxHP (probed: reverts to `237/237`).
- **moveSlots**: each copied with `pp = Math.min(5, move.pp)` and `maxpp = calculatePP(move, ppUps)`
  (gen<5 keeps the real maxpp), marked `virtual: true`. So every copied move has **5 PP**, maxpp = the
  real max (probed `swordsdance:5/48`, `bodyslam:5/15`, `rest:5/10`, `splash:5/64`). Hidden Power
  copies as `Hidden Power <hpType>` using the copied `hpType` (gen<5 copies the target's hpType).
- **boosts**: ALL stat stages copied from the target's CURRENT stages at transform time (probed: a
  Ditto that transforms BEFORE the target's Swords Dance copies 0 boosts — timing, not a miss).
- **ability**: gen>2 → `setAbility(target.ability)` (probed `thickfat`, `synchronize`).
- **NOT copied in gen3**: HP, item, status, and volatiles (the gen>=6 crit-volatile copy block does
  not apply).

**Fail conditions (gen3 → return false → `|move|USER|Transform||[still]` + `|-fail|USER`, draw-free):**
- target `fainted`.
- target `transformed` (`&& gen >= 2`) — probed via the Ditto mirror: the 2nd Transform fails.
- (illusion / Terapagos / Stellar — none gen3-relevant.)
- **NOT a fail vs Substitute in gen3** — the `substitute && gen >= 5` guard is gen5+, and Transform
  has `bypasssub`, so it copies THROUGH a sub (probed: `-transform` succeeded into a subbed Snorlax).
- The USER being already transformed does NOT block in gen3 (the `this.transformed && gen >= 5` guard
  is gen5+).

**Revert on switch-out** (probed): a transformed Ditto that switches out reverts to base Ditto
(species/stats/moves/types/ability/boosts all restored); re-entry shows `|switch|…|Ditto|237/237`.
While benched it reads as base `ditto`.

**State (new — a substantial overlay):** a per-mon `transform` overlay holding the copied
species/types/stats(−HP)/ability/boosts/moveSlots(5 PP each) + a `transformed` flag, applied over the
active mon and CLEARED on switch-out (revert). Using a copied move deducts that copied slot's PP
normally (probed: `calmmind:5/20` → `4/20` after use). This is the largest state job in the two batches.

**Emission:** `|-transform|USER|TARGET` (a plain Transform move has no `effect`, so no `[from]`
suffix). Copied-move use / revert-switch emit through the normal move/switch lines using the copied
species+moves.

**Gotchas:** the speed-copy → speed-tie interaction (above) is the only draw subtlety and it rides
existing machinery. Copied moves are `virtual` (they don't count toward the transformed mon's "real"
movepool for e.g. Encore/Disable edge cases). A transformed Ditto vs Encore: the `failencore` flag is
on Transform itself (it can't be Encored). Copied Hidden Power uses the copied hpType/hpPower.

---

## WONDER GUARD (`wonderguard`) — only super-effective damaging moves connect

**Resolved mechanic** (gen4 override inherited by gen3, `data/mods/gen4/abilities.ts:530`):
```
onTryHit(target, source, move) {
  if (move.id === 'firefang') { ...gen4 hint...; return; }   // firefang not gen3-legal
  if (target === source || move.category === 'Status' || move.type === '???') return;  // bypass
  if (target.runEffectiveness(move) <= 0 || !target.runImmunity(move)) {
    this.add('-immune', target, '[from] ability: Wonder Guard'); return null;
  }
}
```
`runEffectiveness(move)` is the log2-space effectiveness SUM (2× → +1, 4× → +2, 1× → 0, ½× → −1,
0× handled by runImmunity). So **a move connects only if `runEffectiveness > 0`** (strictly
super-effective) AND not type-immune. Neutral (0), resisted (<0), and immune moves are all BLOCKED.

**Draw model (probe `probe_batch89_abilities_items.js`):** the block happens at `onTryHit`, which
runs **AFTER the accuracy roll** and BEFORE crit/damage/secondary. So a blocked move draws ONLY its
accuracy roll, then `-immune`:
```
WG t1 (Tackle, Normal, neutral/immune vs Bug/Ghost): draws=2 = [accuracy] + baseline
 lines=["|move|USER|Tackle|Shedinja","|-immune|Shedinja|[from] ability: Wonder Guard"]
WG t2 (Ember, Fire SE vs Bug): HITS -> "-supereffective","-damage","faint"  (full draw chain)
```

**Bypasses Wonder Guard (probe-confirmed where noted):**
- **Status moves** (category Status) — pass; e.g. Leech Seed's `-start` applies normally (probed).
- **Self-targeting** (`target === source`).
- **Residual damage** — WG only has `onTryHit` (a move hook). Leech Seed RESIDUAL drain KILLED a
  1-HP Shedinja (probed): `|-damage|Shedinja|0 fnt|[from] Leech Seed` → faint. By the same logic,
  weather chip (sand/hail), burn/poison, and other residuals bypass WG.
- **`move.type === '???'` (Struggle)** — the `onTryHit` returns early → Struggle connects. **Source-
  confirmed only** (the live Struggle setup was too heavy to construct here); the branch is
  unambiguous and the analogous status/residual bypasses were probed. Low risk; flag if a byte-gate
  is wanted.

**Fixed-damage moves** are subject to the WG effectiveness check on their TYPE: Seismic Toss
(Fighting) vs Bug/Ghost → blocked (Fighting→Ghost immune); Night Shade (Ghost) vs Bug/Ghost → Ghost
is SE vs Ghost → **connects** for its fixed `level` damage. (Derived from the rule, not separately
probed — the rule is `runEffectiveness(move) > 0` on the move's type.)

**State:** none new (it's a read-only ability gate on the incoming-move path). Needs `runEffectiveness`
(the log2 effectiveness sum, which the port's damage layer already computes) exposed to the
move-immunity check.

**Emission:** `|-immune|TARGET|[from] ability: Wonder Guard`.

**Gotchas:** the effectiveness SUM must be the gen3 type chart's; the port already has this. A
0×-immune move also routes through WG's `-immune [from] ability: Wonder Guard` line (WG's onTryHit
fires before the plain type-immunity check), so the byte form differs from a normal `-immune`.

---

## FORECAST (`forecast`, Castform) — forme + type follows EFFECTIVE weather

**Resolved mechanic** (gen3 override = gen4 with different flags; base `onWeatherChange`):
`onSwitchInPriority: -2`, `onStart` → `singleEvent('WeatherChange')`, `onWeatherChange` maps
`pokemon.effectiveWeather()`: sun → Castform-Sunny (Fire), rain → Castform-Rainy (Water), hail →
Castform-Snowy (Ice), everything else (incl. **sand** and none) → Castform (Normal). Gated by
`baseSpecies === 'Castform' && !transformed`.

**Draw model — ZERO draws** (probe `probe_forecast_rng.js`: Forecast vs a Levitate control drew the
SAME per-turn counts). Every `-formechange` in every trace fired with draws = baseline only.

**The Cloud-Nine / Air-Lock composition (the previously-UNPROBED piece — `probe_batch89_forecast_cloudnine.js`):**
Forecast reads `effectiveWeather()`, which returns `''` while a WEATHER_NEGATE ability (Cloud Nine /
Air Lock) is active — so it composes cleanly with the port's existing `effective_weather()` model:
```
AL-opp t1: SunnyDay set (rawWeather=sunnyday) but Air Lock active -> effWeather='' ->
           Castform stays Normal (species castform, types Normal), NO formechange
AL-opp t2: Air Lock (Rayquaza) switches OUT -> effWeather=sunnyday ->
           "|-formechange|p1a: Castform|Castform-Sunny|[msg]|[from] ability: Forecast" (Fire)
CN-switchin: Castform switches in under Cloud-Nine-suppressed rain -> stays Normal (no forme)
```
When the suppressor leaves, the port's existing `gen3_cloudnine_end_v1` (Cloud Nine/Air Lock onEnd
fires a WeatherChange) drives Forecast's re-forme; a Forecast switch-in (onStart, priority −2) formes
off `effective_weather()` too.

**Revert/lifecycle (probed, existing probe):** weather END → revert to Normal
(`-formechange … Castform`); switch-OUT → reverts to base `castform` while benched; switch-IN under
standing weather → re-formes via onStart. Sand does NOT change the forme (stays Normal).

**State:** a per-mon current-forme derived from `effective_weather()` for a Castform-with-Forecast;
the reported species id at a boundary is the FORME (`castformrainy`, etc.), which the switch/details/
request bytes must carry.

**Emission (byte-exact):** `|-formechange|MON|Castform-Rainy|[msg]|[from] ability: Forecast` (and
`Castform-Sunny` / `Castform-Snowy` / `Castform` for revert). The `[msg]` tag is present.

**⚠️ OWNER DECISION (reporting surface):** modeling Forecast means the port must (a) report the forme
species id in switch/`-formechange`/request lines and (b) recompute the forme on every
weather-change / switch-in / suppressor-end. This is the "forme-change REPORTING surface" the CLAUDE.md
deferral flagged. The MECHANIC + draw model + Cloud-Nine composition are now fully settled; the open
question is purely whether the port wants to take on the forme-species reporting plumbing (0 gen3ou
sample teams carry Castform, so it only matters for random-battle / full-gen3 completeness). See
Recommendation.

---

## LIQUID OOZE (`liquidooze`) — drain/leech heal reversed into damage

**Resolved mechanic** (gen4 override inherited by gen3, `data/mods/gen4/abilities.ts:250`):
```
onSourceTryHeal(damage, target, source, effect) {
  const canOoze = ['drain', 'leechseed'];
  if (canOoze.includes(effect.id) && this.activeMove?.id !== 'dreameater') {
    this.damage(damage, null, null, null, true); return 0;
  }
}
```
So the healer takes `damage` instead of healing, for `effect.id ∈ {drain, leechseed}` — **but
Dream Eater is EXCLUDED** (its drain heal is NOT reversed in gen3, per the gen4 override).

**Draw model — ZERO draws** (draw-free damage redirection; probe: the drain move's normal
accuracy/crit/damage draws only, no extra):
```
LO-gigadrain t1: Venusaur Giga Drain hits Tentacruel(184/301) then
 "|-damage|p1a: Venusaur|243/301|[from] ability: Liquid Ooze|[of] p2a: Tentacruel"  (attacker takes ~heal amt)
LO-leechseed residual: "|-damage|Tentacruel|264/301|[from] Leech Seed|[of] Venusaur" THEN
 "|-damage|Venusaur|264/301|[from] ability: Liquid Ooze|[of] Tentacruel"  (seeder takes damage)
```

**Reverses:** drain moves (Giga Drain / Mega Drain / Absorb / Leech Life — `effect.id === 'drain'`)
and Leech Seed residual. **Does NOT reverse:** Dream Eater (excluded), Leftovers/Ingrain/Aqua-Ring/
Rest/Wish/Recover (not drain/leechseed — those are normal self-heals, untouched).

**State:** none new (a read-only ability on the heal path: `onSourceTryHeal` from the Liquid-Ooze
holder's perspective). The port's drain + leech-seed heal sites must consult the target/seeded mon's
ability.

**Emission:** `|-damage|HEALER|HP|[from] ability: Liquid Ooze|[of] OOZE_MON`. Order for Leech Seed:
the seeded mon's `-damage [from] Leech Seed` first, then the seeder's Liquid-Ooze `-damage`.

**Gotchas:** a drain into a Liquid Ooze mon can KO the ATTACKER (self-inflicted); Focus Band could
survive it (onDamage). The reversal amount equals the would-be heal (drain = ceil(dealt/2); Leech
Seed = maxhp/8), applied as damage.

---

## WHITE HERB (`whiteherb`) — restore lowered stats

**Resolved mechanic** (base item, inherited): triggers on `onAnyAfterMove` / `onAnySwitchIn` /
`onAnyAfterMega` / `onResidual(order 29)`. `onStart` scans `pokemon.boosts`; if any stage `< 0`, it
records the negatives → `useItem()`; `onUse` → `setBoost({negatives → 0})` + `-clearnegativeboost
[silent]`. Single-use.

**Draw model — ZERO draws.** Fires immediately after the stat-drop resolves (draw-free item trigger):
```
WH-growl: "|-unboost|Snorlax|atk|1" then IMMEDIATELY
          "|-enditem|Snorlax|White Herb","|-clearnegativeboost|Snorlax|[silent]"  (boosts back to 0)
WH-superpower: user's own -1 atk/-1 def restored right after the move: -enditem + -clearnegativeboost
WH-mixed: +2 atk then a -1 -> net +1 (still positive) -> White Herb does NOT fire (item retained)
```

**Rules (probed):**
- Triggers only if **at least one** stage is `< 0`.
- Restores **only the negative** stages to 0 (positives untouched — e.g. +2 then −1 leaves +1 and does
  NOT trigger, since the net is positive; a genuine −1 among +2s would restore only the −1).
- Fires **immediately after the causing move** (`onAnyAfterMove`) — including the USER's own self-drop
  moves (Superpower / Overheat / etc.) and a foe's stat-drop move (Growl / Intimidate on switch-in via
  `onAnySwitchIn`).
- Single consumption: `-enditem` fires once, item gone.

**State:** none new (reuses per-mon boost stages + the item slot). The trigger just needs to run the
negative-boost scan at the existing after-move / switch-in / residual sites.

**Emission:** `|-enditem|TARGET|White Herb` then `|-clearnegativeboost|TARGET|[silent]`.

**Gotchas:** the most common gen3 use is vs an Intimidate lead (onAnySwitchIn restores the −1 Atk) and
after a self-drop move (Overheat/Superpower). The `[silent]` on `-clearnegativeboost` means no
per-stat `-unboost`/`-boost` reversal lines — just the two lines above.

---

## STICK / LEEK (`stick`) — Farfetch'd crit boost

**Resolved mechanic** (base item, inherited): `onModifyCritRatio(critRatio, user)` →
`if (user.species.id === 'farfetchd') return critRatio + 2;`. `leek` is the gen8 rename (same num
259, itemUser Farfetch'd/Sirfetch'd) — **only `stick` is gen3-legal** (`leek` doesn't exist as an
obtainable gen3 item). It is a pure `critRatio + 2` fold, species-gated to Farfetch'd — the SAME
mechanism the port already models for Scope Lens / Lucky Punch (which are `critRatio + 1`).

**Draw model — no new draw.** The crit itself is the existing `randomChance(1, critMult)` at a higher
ratio. Crit-rate confirmed over 60 seeds:
```
Stick (Farfetch'd): crits=15/59 hits  (~25% ≈ critRatio stage 2 = 1/4)
(no item):          crits= 4/59 hits  (~7%  ≈ critRatio stage 0 = 1/16)
```

**State:** none new (folds into the existing `crit_ratio` accumulation). Just add `stick` to the
crit-item table with `+2` and a `only_species: Farfetch'd` gate (the port already carries `stick`
under CRIT_ITEM per the item-mechanics note — verify the gate is the species-conditional `+2`, not a
Scope-Lens-style unconditional `+1`).

**Emission:** none (crit boost is silent; the eventual `-crit` line is the normal one).

---

# Recommended implementation order (easiest / most-isolated first)

1. **STICK** — trivial: a species-gated `critRatio + 2` fold into the existing crit-item table.
   Zero draws, zero state, zero emission. (May already be present under CRIT_ITEM — just verify the
   +2 species gate.)
2. **HAZE** — draw-free field move, one `-clearallboost` line, clears all actives' boosts. No state.
3. **WHITE HERB** — draw-free item; run the negative-boost scan at the existing after-move/switch-in
   sites; two emission lines. No new state.
4. **LIQUID OOZE** — draw-free read-only ability on the drain + leech-seed heal sites (exclude Dream
   Eater); one `-damage [from] ability` line. No new state.
5. **WONDER GUARD** — draw-free read-only move-immunity gate (`runEffectiveness > 0` else `-immune`);
   bypassed by status/self/`???`/residuals. No new state. (Confirm the Struggle `???` bypass with a
   byte-gate if desired — currently source-confirmed only.)
6. **YAWN** — a duration-2 volatile whose `onEnd` routes into the EXISTING `try_set_status('slp')`
   (so the sleep `random(2,6)` + Sleep-Clause + SetStatus shuffle come for free). Small state
   (`yawn` volatile + source uid), draw-free cast.
7. **TRICK** — one accuracy draw + a draw-free swap; fail/block set is fully enumerated (Sticky Hold /
   Substitute / both-itemless; Mail+berries swap). No new state (reuses the item slot; watch the
   choice-lock interaction).
8. **PARTIAL-TRAP family** — the one truly new draw (`random(3,7)` at cast) + a draw-free maxhp/16
   residual chip + a firm-`trapped` switch-block folding into the existing trap gate. Modest state
   (`partial_trap` per-mon). Well-isolated but touches the residual + switch-legality layers.
9. **TRANSFORM** — the largest job: a full copy-overlay (species/types/stats−HP/ability/boosts/moves
   at 5 PP) with revert-on-switch, plus the copied-Speed → speed-tie interaction. Draw-free, but the
   most state.
10. **FORECAST** — deferred pending an OWNER DECISION on the forme-species REPORTING surface (below).
    The mechanic + draw model + Cloud-Nine composition are fully settled; only the reporting plumbing
    is open, and 0 gen3ou teams carry Castform (random-battle completeness only).

# Owner-decision / open flags

- **FORECAST reporting surface** — the only genuine design call. Modeling it requires the port to
  emit the forme species id in switch/`-formechange`/request lines and recompute the forme on every
  weather-change / switch-in / suppressor-end. Everything else about Forecast is nailed
  (`effectiveWeather()`-driven, draw-free, `-formechange|MON|Castform-<Forme>|[msg]|[from] ability:
  Forecast`, Cloud-Nine composes with the existing `effective_weather()`). Recommend deferring
  behind the other nine unless random-battle Castform coverage is wanted.
- **WONDER GUARD × Struggle** — the `move.type === '???'` bypass is source-confirmed but not
  live-probed (Struggle setup was heavy). Unambiguous branch; add a Struggle-into-Shedinja byte-gate
  if a regression pin is desired.
- **`switcheroo`** — gen4 move (num 415), NOT gen3-legal; leave out of the modeled set. If ever
  needed it is a Trick clone (Dark-typed) with a subtly different itemless-fail condition.
- **Dream Eater × Liquid Ooze** — the gen4 override EXCLUDES Dream Eater from the reversal (its drain
  heals normally vs Liquid Ooze). Make sure the port's Liquid-Ooze gate checks
  `activeMove != dreameater`, matching the mod chain, rather than reversing all `drain` heals.
