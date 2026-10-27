// PROBE: the FIX batch (BF) of the gen-3 full-parity program — seven pre-existing divergence classes in
// moves the engine already ran, each settled here against the REAL sim (the resolved `Dex.mod('gen3')`).
// Every scenario prints the omniscient lines (the secret half of each `|split|` pair, `|t:|` dropped) from
// `|turn|1` on, and the PRNG seed after each decision; the Rust pins in `tests/regression_test.rs`
// (`bf*_`) replay the same teams, seed and choices and assert these lines and seeds.
//
// SETTLED 2026-10-26 (run it to re-confirm; do not re-derive from source):
//
//   BF1 RAGE (data/moves.ts `rage.condition`):
//     a. the hit reaction is `onHit` on the TARGET's volatile, run by `runEvent('Hit')` inside
//        `runMoveEffects` (sim/battle-actions.ts:1304), i.e. AFTER `spreadDamage` printed `-damage` (and
//        after a drain heal, a partial-trap `-activate` and the move's own `onHit`), BEFORE selfDrops /
//        secondaries. So `|-damage|` precedes `|-boost|…|atk|1`.
//     b. `onBeforeMove` at PRIORITY 100 removes the volatile — before slp (10), frz, flinch, par (1) —
//        so a Rage user that is asleep / fully paralysed / flinched loses it at the ATTEMPT; a hit
//        afterwards boosts nothing.
//     c. `boost()` (sim/battle.ts:2017) returns on `!target.hp` (a KO'd holder gets nothing) and, at
//        +6, still prints the zero delta `|-boost|<mon>|atk|0` (a non-secondary, non-ability effect).
//     d. a 0-damage hit (False Swipe into a 1-HP holder) still runs `runEvent('Hit')`: the holder boosts.
//   BF2 FURY CUTTER (gen3 mod: `onHit: source.addVolatile('furycutter')`; base `basePowerCallback` adds the
//     volatile ONLY when absent — `move.hit` is undefined for a single-hit gen-3 move, data/mods/gen3/
//     scripts.ts:411 sets it only in the multihit loop): the multiplier DOUBLES in the move's `onHit`
//     (an `onRestart`), not when the base power is computed. So a use that never reaches `getDamage`
//     (a full paralysis, sleep, a Protect, a miss) changes nothing, and a use that computes damage but
//     never runs `onHit` (a SUBSTITUTE took it) leaves the multiplier where it was: Fury Cutter into a
//     sub twice does 10 BP both times. The volatile (duration 2, a residual duration handler) exists
//     only from the first damage computation on.
//   BF3 UPROAR wakes sleepers in the MOVE's `onTryHit` (data/moves.ts `uproar.onTryHit`), i.e.
//     `singleEvent('TryHit')` at the top of `spreadMoveHit` (sim/battle-actions.ts:1065), before the
//     substitute check, the crit roll and `-damage`: `|-curestatus|<foe>|slp|[msg]` comes BEFORE the
//     `|-damage|` line, whose HP field then carries no `slp`.
//   BF4 SMELLING SALTS' `onHit` calls `target.cureStatus()`, which returns at once on `!this.hp`
//     (sim/pokemon.ts:1676): a KO'd paralysed target is NOT cured (no `-curestatus`). It runs in
//     `runMoveEffects` (before secondaries), at the same step as Rage's reaction (move onHit first).
//   BF5 FALSE SWIPE on a 1-HP target: `onDamage` returns `target.hp - 1 = 0`; `spreadDamage` keeps a 0
//     (`if (targetDamage !== 0) clampIntRange(…, 1)`) and still prints `|-damage|<foe>|1/<max>`. Its
//     `onDamagePriority: -20` runs AFTER Endure's -10: a lethal swipe into an Endure user activates
//     Endure (`-activate|…|move: Endure`), and a 0 from Endure ends the relay before False Swipe runs.
//   BF6 REST by an INSOMNIA / VITAL SPIRIT user: `rest.onTry` (data/moves.ts:14976) fails it with
//     `|-fail|<u>|[from] ability: Insomnia|[of] <u>` BEFORE `onHit`'s `setStatus`, so no `SetStatus`
//     event runs and gen3ou's two clause handlers are never sorted: ZERO draws (== a Splash turn's seed).
//   BF8 (fuzz-found) the DEFROST thaw (`frz.onModifyMove`, resolved gen3 conditions) runs in `useMoveInner`, i.e.
//     after EVERY BeforeMove handler: a confused frozen Flame Wheel user prints the confusion `-activate`
//     BEFORE `-curestatus …|frz|[from] move: Flame Wheel`, and a confusion self-hit leaves it frozen.
//   BF9 (fuzz-found) `Pokemon.toString()` (sim/pokemon.ts) reads `isActive`, which `faintMessages` clears: a
//     Future Sight whose caster fainted in the same residual misses as `|-miss|p2: <caster>|<target>`.
//   BF10 (bridge-fuzz-found) LANSAT BERRY on a holder that already has the `focusenergy` volatile: `addVolatile`
//     fails silently (no `onRestart`): `-enditem …|[eat]` and NO `|-start|…|move: Focus Energy`.
//   BF11 (bridge-fuzz-found) a MIMIC'd Curse slot has no `target`, but `getMoveRequestData` (sim/pokemon.ts:998-1004)
//     sets a non-Ghost user's Curse to `nonGhostTarget` "self": the request shows `"target":"self"`.
//   BF7 the PURSUIT interrupt (gen4 `pursuit.condition.onBeforeSwitchOut`, inherited by gen 3) calls
//     `source.moveUsed(pursuit)` and `useMove` → `useMoveInner` sets `lastMoveUsed = pursuit` (data/mods/
//     gen3/scripts.ts:127). So after a self-overwriting MIMIC, a Pursuit strike makes `lastMove` Pursuit
//     (a real slot): a faster foe's DISABLE then disables Pursuit. CONVERSION 2 reads `lastMoveUsed`.
//
//   node src/rust_sim/harness/probe_bf_classes.js [label-filter]     (BF_JSON=1: one JSON row per scenario)
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const SET = (species, ability, moves, extra = '') =>
  `${species}||none|${ability}|${moves}|Hardy|85,85,85,85,85,85|M||||${extra}`;
const LV = (species, ability, moves, level) =>
  `${species}||none|${ability}|${moves}|Hardy|85,85,85,85,85,85|M|||${level}|`;

function omniscient(log) {
  // Keep the SECRET half of each `|split|pN` pair (exact HP), drop `|t:|`.
  const out = [];
  for (let i = 0; i < log.length; i++) {
    const l = log[i];
    if (l.startsWith('|split|')) { out.push(log[i + 1]); i += 2; continue; }
    if (l.startsWith('|t:|') || l.startsWith('|debug|')) continue;
    out.push(l);
  }
  return out;
}

function run(label, fmt, seed, p1, p2, choices, quiet = false) {
  const filt = process.argv[2];
  if (filt && !label.startsWith(filt)) return null;
  const b = new Battle({ formatid: fmt, seed });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: p2 });
  const seeds = [];
  for (const [c1, c2] of choices) {
    b.makeChoices(c1, c2);
    seeds.push(String(b.prng.getSeed()));
  }
  const lines = omniscient(b.log);
  const from = lines.indexOf('|turn|1');
  if (quiet) return lines.slice(from);
  if (process.env.BF_JSON) {
    console.log(JSON.stringify({ label, fmt, seed, p1, p2, choices, lines: lines.slice(from).filter(l => l && l !== '|'), seeds }));
    return null;
  }
  console.log(`\n== ${label}  fmt=${fmt} seed=${JSON.stringify(seed)}`);
  for (const l of lines.slice(from)) if (l) console.log('   ' + l);
  seeds.forEach((s, i) => console.log(`   seed after decision ${i}: ${s}`));
}

const M = (a, b) => [`move ${a}`, `move ${b}`];

// ---- BF1 RAGE ----
const TAUROS = SET('Tauros', 'intimidate', 'rage,splash,bellydrum');
const SNORLAX_TACKLE = SET('Snorlax', 'thickfat', 'tackle,splash,spore');
run('BF1a rage: -damage before -boost', 'gen3customgame', [1, 2, 3, 4], TAUROS, SNORLAX_TACKLE, [M(1, 1)]);
// Turn 1 Tauros (faster) Rages, the slower Snorlax Spores it; turn 2 Tauros is asleep (the cant
// still removes the volatile at priority 100) and Snorlax's Tackle boosts nothing.
run('BF1b rage: removed by a sleeping attempt', 'gen3customgame', [1, 2, 3, 4], TAUROS, SNORLAX_TACKLE,
  [M(1, 3), M(1, 1)]);
// +6: Belly Drum, then Rage, then the hit reports the zero delta.
run('BF1c rage: at +6 the hit prints atk|0', 'gen3customgame', [1, 2, 3, 4], TAUROS, SNORLAX_TACKLE,
  [M(3, 2), M(1, 1)]);
// A Level-5 Rage user KO'd by the hit: no boost.
run('BF1d rage: a KO-d holder gets no boost', 'gen3customgame', [1, 2, 3, 4],
  LV('Tauros', 'intimidate', 'rage,splash', 5) + ']' + SET('Tauros', 'intimidate', 'splash'),
  SET('Snorlax', 'thickfat', 'bodyslam,splash'), [M(1, 2), M(1, 1)]);

// The reaction runs for every damaging hit: a fixed-damage Seismic Toss, and once per multi-hit strike.
run('BF1e rage: a Seismic Toss boosts', 'gen3customgame', [1, 2, 3, 4], TAUROS,
  SET('Snorlax', 'thickfat', 'seismictoss,splash'), [M(1, 1)]);
run('BF1f rage: each Double Kick strike boosts', 'gen3customgame', [1, 2, 3, 4], TAUROS,
  SET('Snorlax', 'thickfat', 'doublekick,splash'), [M(1, 1)]);
// A Hyper Beam into a Rage holder: the reaction precedes the selfDrops `-mustrecharge`.
run('BF1g rage: before -mustrecharge', 'gen3customgame', [1, 2, 3, 4], TAUROS,
  SET('Snorlax', 'thickfat', 'hyperbeam,splash'), [M(1, 1)]);

// ---- BF2 FURY CUTTER ----
const SCIZOR = SET('Scizor', 'swarm', 'furycutter,splash');
// Turn 1 the faster Jolteon Protects (the cut never reaches getDamage), turn 2 it lands: BP 10.
run('BF2a furycutter: a Protect-blocked use leaves no counter', 'gen3customgame', [1, 2, 3, 4], SCIZOR,
  SET('Jolteon', 'voltabsorb', 'protect,splash'), [M(1, 1), M(1, 2), M(1, 2)]);
// Turn 1 the faster Jolteon puts up a Substitute; Scizor cuts it twice (both 10 BP: no onHit).
run('BF2b furycutter: into a substitute the counter does not double', 'gen3customgame', [1, 2, 3, 4],
  SCIZOR, SET('Jolteon', 'voltabsorb', 'substitute,splash'), [M(1, 1), M(1, 2), M(1, 2)]);
// Sleep: the faster Breloom Spores Scizor, whose cut is a `cant`.
run('BF2c furycutter: a sleeping attempt leaves no counter', 'gen3customgame', [1, 2, 3, 4], SCIZOR,
  SET('Breloom', 'effectspore', 'spore,splash'), [M(1, 1), M(1, 2), M(1, 2), M(1, 2), M(1, 2)]);

// ---- BF3 UPROAR ----
run('BF3 uproar: -curestatus before -damage', 'gen3customgame', [1, 2, 3, 4],
  SET('Exploud', 'soundproof', 'spore,uproar'), SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), M(2, 1)]);

// ---- BF4 SMELLING SALTS ----
run('BF4 smellingsalts: a KO-d par target is not cured', 'gen3customgame', [1, 2, 3, 4],
  SET('Hariyama', 'thickfat', 'thunderwave,smellingsalts'),
  LV('Magikarp', 'swiftswim', 'splash', 5) + ']' + SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), M(2, 1)]);

// ---- BF5 FALSE SWIPE ----
run('BF5a falseswipe: a 1-HP target still prints -damage', 'gen3customgame', [1, 2, 3, 4],
  SET('Scizor', 'swarm', 'falseswipe'), LV('Magikarp', 'swiftswim', 'splash', 5), [M(1, 1), M(1, 1)]);
run('BF5b falseswipe: a 0-damage hit still triggers Rage', 'gen3customgame', [1, 2, 3, 4],
  SET('Scizor', 'swarm', 'falseswipe'), LV('Magikarp', 'swiftswim', 'rage', 5), [M(1, 1), M(1, 1)]);

// Endure's `onDamage` (priority -10) runs BEFORE False Swipe's (-20): a lethal swipe into an Endure user
// activates Endure (hp - 1), and into a 1-HP Endure user Endure's 0 ends the Damage relay (False Swipe
// never runs) — both print `-activate|…|move: Endure` then the `-damage` line.
run('BF5c falseswipe: Endure activates first', 'gen3customgame', [1, 2, 3, 4],
  SET('Scizor', 'swarm', 'falseswipe'), LV('Magikarp', 'swiftswim', 'endure', 5), [M(1, 1)]);
run('BF5d falseswipe: a 1-HP Endure user', 'gen3customgame', [1, 2, 3, 4],
  SET('Scizor', 'swarm', 'falseswipe,splash'), LV('Magikarp', 'swiftswim', 'splash,endure', 5),
  [M(1, 1), M(1, 2)]);

// ---- BF6 REST under Insomnia, gen3ou (the two-clause SetStatus shuffle) ----
const HYPNO = (m) => SET('Hypno', 'insomnia', m);
run('BF6a rest: insomnia rest in gen3ou', 'gen3ou', [1, 2, 3, 4], HYPNO('rest,splash'),
  SET('Snorlax', 'thickfat', 'bodyslam,splash'), [M(2, 1), M(1, 2)]);
run('BF6b rest: control, a splash instead', 'gen3ou', [1, 2, 3, 4], HYPNO('rest,splash'),
  SET('Snorlax', 'thickfat', 'bodyslam,splash'), [M(2, 1), M(2, 2)]);

// ---- BF7 PURSUIT after a self-overwriting MIMIC ----
// Turn 1 the faster Jolteon Splashes, Tyranitar Mimics it (slot 1 := Splash, lastMove = mimic).
// Turn 2 Jolteon switches to Electrode; Tyranitar's Pursuit strikes it on the way out.
// Turn 3 the faster Electrode Disables Tyranitar's lastMove, then Tyranitar tries Pursuit.
const TTAR = SET('Tyranitar', 'sandstream', 'mimic,pursuit,bodyslam');
const BF7A_FOE = SET('Jolteon', 'voltabsorb', 'splash') + ']' + SET('Electrode', 'soundproof', 'disable,splash');
const BF7A_SCRIPT = [M(1, 1), ['move 2', 'switch 2'], M(2, 1)];
// Disable is accuracy 55: take the first seed [k,k,k,k] where it lands.
for (let k = 1; k <= 60; k++) {
  const ls = run('BF7a', 'gen3customgame', [k, k, k, k], TTAR, BF7A_FOE, BF7A_SCRIPT, true);
  if (ls === null) break;
  if (ls.some(l => l.startsWith('|-start|p1a: Tyranitar|Disable|'))) {
    run('BF7a pursuit: Disable after Mimic-then-Pursuit', 'gen3customgame', [k, k, k, k], TTAR, BF7A_FOE, BF7A_SCRIPT);
    break;
  }
}
// Conversion 2 reads `lastMoveUsed`: Body Slam (Normal) on turn 1, the Pursuit strike (Dark) on turn 2.
run('BF7b pursuit: Conversion 2 reads the strike', 'gen3customgame', [1, 2, 3, 4], TTAR,
  SET('Jolteon', 'voltabsorb', 'splash') + ']' + SET('Electrode', 'soundproof', 'conversion2,splash'),
  [M(3, 1), ['move 2', 'switch 2'], M(3, 1)]);

// ---- BF8 (fuzz-found) the DEFROST thaw comes after every BeforeMove handler ----
// A confused, frozen Ponyta uses Flame Wheel: the thaw roll fails, the defrost move proceeds, the confusion
// `-activate` prints, and only then (`frz.onModifyMove`, in useMoveInner) `-curestatus …|frz|[from] move:`.
// T1 Lapras Confuse Rays, T2 Lapras Ice Beams (a 10 % freeze), T3 Ponyta Flame Wheels: the first seed where
// T2 froze and T3 shows the confusion `-activate` without a natural thaw.
const PONYTA = SET('Ponyta', 'flashfire', 'flamewheel,splash');
const LAPRAS = SET('Lapras', 'waterabsorb', 'icebeam,confuseray,splash');
const BF8_SCRIPT = [M(2, 2), M(2, 1), M(1, 3)];
for (let k = 1; k <= 3000; k++) {
  const ls = run('BF8', 'gen3customgame', [k, k, k, k], PONYTA, LAPRAS, BF8_SCRIPT, true);
  if (ls === null) break;
  const t3 = ls.slice(ls.indexOf('|turn|3'));
  if (ls.some(l => l === '|-status|p1a: Ponyta|frz') && t3.some(l => l === '|-activate|p1a: Ponyta|confusion') &&
      t3.some(l => l.startsWith('|-curestatus|p1a: Ponyta|frz|[from] move:'))) {
    run('BF8a defrost: the confusion -activate precedes the thaw', 'gen3customgame', [k, k, k, k], PONYTA, LAPRAS, BF8_SCRIPT);
    break;
  }
}
// ... and a confusion SELF-HIT cancels the move before the thaw: the user stays frozen.
for (let k = 1; k <= 3000; k++) {
  const ls = run('BF8', 'gen3customgame', [k, k, k, k], PONYTA, LAPRAS, BF8_SCRIPT, true);
  if (ls === null) break;
  const t3 = ls.slice(ls.indexOf('|turn|3'));
  if (ls.some(l => l === '|-status|p1a: Ponyta|frz') && t3.some(l => l.includes('|[from] confusion')) &&
      !t3.some(l => l.startsWith('|-curestatus|p1a: Ponyta'))) {
    run('BF8b defrost: a confusion self-hit leaves the user frozen', 'gen3customgame', [k, k, k, k], PONYTA, LAPRAS, BF8_SCRIPT);
    break;
  }
}

// ---- BF9 (fuzz-found) a FAINTED future-move caster prints slot-less ----
// `Pokemon.toString()` reads `isActive`, which `faintMessages` clears: Natu casts Future Sight on turn 1, is
// False-Swiped to 1 HP on turn 3 and faints to the sandstorm in that turn's residual, BEFORE the strike lands;
// a MISSED strike then prints `|-miss|p2: Natu|p1a: Tyranitar`. The first seed whose strike misses.
const TTAR_FS = SET('Tyranitar', 'sandstream', 'falseswipe,splash');
const NATU = LV('Natu', 'synchronize', 'futuresight,splash', 5) + ']' + SET('Snorlax', 'thickfat', 'splash');
const BF9_SCRIPT = [M(2, 1), M(2, 2), M(1, 2)];
for (let k = 1; k <= 400; k++) {
  const ls = run('BF9', 'gen3customgame', [k, k, k, k], TTAR_FS, NATU, BF9_SCRIPT, true);
  if (ls === null) break;
  if (ls.some(l => l.startsWith('|-miss|p2: Natu|'))) {
    run('BF9 toref: a fainted caster misses slot-less', 'gen3customgame', [k, k, k, k], TTAR_FS, NATU, BF9_SCRIPT);
    break;
  }
}

// ---- BF10 (bridge-fuzz-found, `rmvpofecf_bab_2_6`) LANSAT BERRY on a holder that already has Focus Energy ----
// gen3 `lansatberry.onEat` is `pokemon.addVolatile('focusenergy')`; the condition has no `onRestart`, so on a
// holder that already carries it `addVolatile` returns false SILENTLY: `|-enditem|…|[eat]` and no `-start`.
// Tauros (312 HP) Focus Energies, Belly Drums (156) and Substitutes (78 = 1/4): the berry eats.
const TAUROS_L = (m) => `Tauros||lansatberry|intimidate|${m}|Hardy|85,85,85,85,85,85|M||||`;
run('BF10a lansat: already focused, no -start', 'gen3customgame', [1, 2, 3, 4],
  TAUROS_L('focusenergy,bellydrum,substitute'), SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), M(2, 1), M(3, 1)]);
run('BF10b lansat: control, not focused', 'gen3customgame', [1, 2, 3, 4],
  TAUROS_L('splash,bellydrum,substitute'), SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), M(2, 1), M(3, 1)]);

// ---- BF11 (bridge-fuzz-found, `rmvpofecf_bab_2_21`) a MIMIC'd CURSE in the request ----
// gen4's Mimic slot literal (inherited by gen 3) has no `target`, but `getMoveRequestData` (sim/pokemon.ts:998-1004)
// overrides a Curse slot for a NON-GHOST user with `nonGhostTarget` = "self": that slot DOES carry `"target":"self"`;
// a Ghost mimicker's Curse slot keeps the key dropped. Prints p1's move request after the Mimic.
function req(label, p1) {
  if (process.argv[2] && !label.startsWith(process.argv[2])) return;
  const b = new Battle({ formatid: 'gen3customgame', seed: [1, 2, 3, 4] });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: SET('Jolteon', 'voltabsorb', 'curse,splash') });
  b.makeChoices('move 1', 'move 1');
  const r = b.p1.active[0].getMoveRequestData();
  console.log(`\n== ${label}\n   ${JSON.stringify(r.moves)}`);
  if (process.env.BF_JSON) console.log(JSON.stringify({ label, req: JSON.stringify(r.moves) }));
}
req('BF11a mimic curse: a non-Ghost user shows target self', SET('Snorlax', 'thickfat', 'mimic,splash'));
req('BF11b mimic curse: a Ghost user drops the key', SET('Gengar', 'levitate', 'mimic,splash'));

// ---- BF12 (fuzz-found on the B6 + BF tree, `rmvprol00_ab_4_12`) OWN TEMPO's onUpdate ----
// `owntempo.onUpdate` (data/abilities.ts:3100; gen 7's override only drops onTryBoost) cures a CONFUSED holder:
// `-activate|<mon>|ability: Own Tempo`, then `removeVolatile('confusion')` -> confusion.onEnd `-end|<mon>|confusion`.
// Unreachable with the mon's own ability (onTryAddVolatile blocks it), but a Skill Swap hands Own Tempo to a
// confused mon (either side of the swap): the cure lands at the Update after the move, before the next mover.
const SMEARGLE_OT = SET('Smeargle', 'owntempo', 'confuseray,skillswap');
run('BF12a own tempo: the swap target is cured', 'gen3customgame', [1, 2, 3, 4], SMEARGLE_OT,
  SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), M(2, 1)]);
run('BF12b own tempo: the confused swapper is cured', 'gen3customgame', [5, 6, 7, 8],
  SET('Alakazam', 'synchronize', 'splash,skillswap'), SET('Slowbro', 'owntempo', 'confuseray,splash'),
  [M(1, 1), M(2, 2)]);
// ---- BF13 OBLIVIOUS's onUpdate (gen 5's override, inherited by gen 3: data/mods/gen5/abilities.ts:50) ----
// an infatuated holder: `removeVolatile('attract')` (attract.onEnd `-end|<mon>|Attract|[silent]`), then
// `-end|<mon>|move: Attract|[from] ability: Oblivious`; no `-activate` in gen 3.
const JYNX_F = 'Jynx||none|oblivious|attract,skillswap|Hardy|85,85,85,85,85,85|F||||';
run('BF13a oblivious: the infatuated swap target is cured', 'gen3customgame', [1, 2, 3, 4], JYNX_F,
  SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), M(2, 1)]);
// ---- BF14 (fuzz-found on the B6 + BF tree, `rmvprol00_ab_0_6`) a LOAFING Truant user's PURSUIT ----
// gen 3's `pursuit.beforeTurnCallback` (data/mods/gen3/moves.ts:480-482) returns before `addVolatile('pursuit')`
// when the user `hasAbility('truant') && truantTurn` (gen 3's Truant, data/mods/gen3/abilities.ts:197-212):
// no interrupt on a switching foe (BF14a), and no `pursuit` duration handler in the residual sort (BF14b: tied
// with a Rollout lock, it cost a shuffle draw).
const SLAKING_P = SET('Slaking', 'truant', 'pursuit,splash');
run('BF14a pursuit: a loafing Truant user does not interrupt a switch', 'gen3customgame', [1, 2, 3, 4], SLAKING_P,
  SET('Snorlax', 'thickfat', 'splash') + ']' + SET('Swampert', 'torrent', 'splash'),
  [M(2, 1), ['move 1', 'switch 2'], M(1, 1)]);
run('BF14b pursuit: no residual handler beside a Rollout lock', 'gen3customgame', [1, 2, 3, 4], SLAKING_P,
  SET('Snorlax', 'thickfat', 'rollout,splash'), [M(2, 1), M(1, 1), M(1, 1)]);
// ---- BF15 (fuzz-found on the B6 + BF tree, `rmvprol00_ab_2_5`) SYNCHRONIZE into a SAFEGUARD ----
// gen 3 / 4's Synchronize (data/mods/gen4/abilities.ts:469-478) calls `source.trySetStatus(id, target)` with no
// effect, so `setStatus` takes `this.battle.effect` = the Synchronize ability; Safeguard's `onSetStatus`
// (data/moves.ts:15596-15606) announces `-activate|<source>|move: Safeguard` for `effect.id === 'synchronize'`.
// A Static / Poison Point / Effect Spore / Flame Body status into a Safeguard stays silent (not a Move, not
// Synchronize). BF15a gen3customgame, BF15b gen3ou (the clause shuffle around the blocked reflect).
const ALAKAZAM_S = SET('Alakazam', 'synchronize', 'splash');
const BAYLEEF_S = SET('Bayleef', 'overgrow', 'safeguard,poisonpowder');
run('BF15a synchronize into a safeguard announces it', 'gen3customgame', [1, 2, 3, 4], ALAKAZAM_S, BAYLEEF_S,
  [M(1, 1), M(1, 2)]);
run('BF15b synchronize into a safeguard, gen3ou', 'gen3ou', [1, 2, 3, 4], ALAKAZAM_S, BAYLEEF_S,
  [M(1, 1), M(1, 2)]);
// ---- BF16 (found by the B8 agent) a MULTI-HIT move stops when its USER falls asleep or faints ----
// gen 3's multi-hit loop (data/mods/gen3/scripts.ts:409-410): `for (i = 0; i < hits && target.hp && pokemon.hp;
// i++) { if (pokemon.status === 'slp' && !isSleepUsable) break; ...}`. An Effect Spore sleep (BF16a) or a Rough
// Skin KO (BF16b, a 1-HP Shedinja) between strikes ends the move; `-hitcount` reports the strikes landed.
run('BF16a multihit: the user falls asleep to Effect Spore', 'gen3customgame', [4, 9, 14, 19],
  SET('Hariyama', 'thickfat', 'armthrust'), SET('Breloom', 'effectspore', 'splash'), [M(1, 1)]);
run('BF16b multihit: the user faints to Rough Skin', 'gen3customgame', [5, 6, 7, 8],
  SET('Shedinja', 'wonderguard', 'furyswipes') + ']' + SET('Snorlax', 'thickfat', 'splash'),
  SET('Sharpedo', 'roughskin', 'splash'), [M(1, 1), ['switch 2', '']]);
// ---- BF17 (the B7 x B8 merge) LOCK-ON / MIND READER before an OHKO move ----
// gen 3's OHKO accuracy (data/mods/gen3/scripts.ts:346-363): `if (!target.isSemiInvulnerable())` the level check
// (`-immune|[ohko]` for a lower-level user) and `accuracy = 30 + gap`; then `runEvent('Accuracy')`, where the user's
// `lockon` (data/moves.ts lockon.condition `onSourceAccuracy` -> true) skips the roll. A locked semi-invulnerable
// target skips the level check too (the Invulnerability event let the move through: `onSourceInvulnerability`).
const SMEARGLE_LO = (lv) => LV('Smeargle', 'owntempo', 'lockon,fissure', lv);
run('BF17a lockon ohko: no accuracy roll', 'gen3customgame', [1, 2, 3, 4], SMEARGLE_LO(100),
  LV('Snorlax', 'thickfat', 'splash', 100), [M(1, 1), M(2, 1)]);
run('BF17b lockon ohko: a lower-level user is still immune', 'gen3customgame', [1, 2, 3, 4], SMEARGLE_LO(50),
  LV('Snorlax', 'thickfat', 'splash', 100), [M(1, 1), M(2, 1)]);
run('BF17c lockon ohko: a locked digging target skips the level check', 'gen3customgame', [1, 2, 3, 4], SMEARGLE_LO(50),
  LV('Dugtrio', 'arenatrap', 'dig,splash', 100), [M(1, 2), M(2, 1)]);
// ---- BF12c / BF13b (the B10 x BF merge) the same Update cures through ROLE PLAY ----
// Role Play (data/moves.ts:15325-15346) sets the copied ability without a Start (gen 3); a copied Own Tempo on a
// confused user / Oblivious on an infatuated user is cured at the Update after the move, as through Skill Swap.
run('BF12c own tempo: copied by role play cures the confused user', 'gen3customgame', [5, 6, 7, 8],
  SET('Alakazam', 'synchronize', 'roleplay,splash'), SET('Slowbro', 'owntempo', 'confuseray,splash'),
  [M(2, 1), M(1, 2)]);
run('BF13b oblivious: copied by role play ends the attract on the user', 'gen3customgame', [5, 6, 7, 8],
  SET('Alakazam', 'synchronize', 'roleplay,splash'),
  'Jynx||none|oblivious|attract,splash|Hardy|85,85,85,85,85,85|F||||', [M(2, 1), M(1, 2)]);
// ---- BF18 (bridge-fuzz-found, `rmvpyo27k_bab_0_0`) a STRUGGLE does not inherit the chosen slot's MULTI-HIT ----
// With every PP spent, `side.choose` substitutes `moveid: 'struggle'` (sim/side.ts chooseMove) and the action runs
// the Struggle move (gen 3: 50 BP, accuracy 100, typeless, recoil 1/4; data/mods/gen3/moves.ts:580-586). The port
// read the multi-hit count from the stale chosen slot (Bonemerang here: 2 strikes, Fury Swipes: a [2,5] sample),
// so its Struggle struck twice. Absol's Pressure spends Bonemerang's 16 PP in 8 turns; turns 9-10 Struggle.
const SABLEYE_BONE = SET('Sableye', 'keeneye', 'bonemerang');
const ABSOL_REC = SET('Absol', 'pressure', 'recover,splash');
run('BF18a struggle: no multi-hit from the chosen slot', 'gen3customgame', [1, 2, 3, 4], SABLEYE_BONE, ABSOL_REC,
  [M(1, 1), M(1, 1), M(1, 1), M(1, 1), M(1, 1), M(1, 1), M(1, 1), M(1, 1), M(1, 1), M(1, 1)]);
// ---- BF19 (bridge-fuzz-found on the B10 tree, `rmvq1sg27_bab_1_5`) KING'S ROCK rolls on every BEAT UP strike ----
// gen 3's King's Rock (data/mods/gen3/items.ts kingsrock.onModifyMove) appends `{chance: 10, volatileStatus:
// 'flinch'}` for a listed move, Beat Up included; Beat Up runs gen 3's multi-hit loop (data/mods/gen3/scripts.ts:409),
// so each strike's `secondaries` rolls it after the strike's damage. The port rolled none for Beat Up.
const HOUNDOUR_KR = 'Houndour||kingsrock|flashfire|beatup|Hardy|85,85,85,85,85,85|M||||';
run('BF19a beat up: kings rock on every strike', 'gen3customgame', [5, 6, 7, 8],
  HOUNDOUR_KR + ']' + SET('Registeel', 'clearbody', 'splash') + ']' + SET('Tyranitar', 'sandstream', 'splash'),
  SET('Snorlax', 'thickfat', 'splash'), [M(1, 1)]);
// ---- BF20 (the B7 agent's unverified finding, confirmed) the SCREENS tie shuffle on BEAT UP and the CRASH ----
// gen 3's `modifyDamage` (data/mods/gen3/scripts.ts:61) runs `runEvent('ModifyDamagePhase1')`; with two or more screens
// up their `onAnyModifyDamagePhase1` handlers tie and `speedSort` shuffles them (one draw for two). Every `getDamage`
// reaches it: each Beat Up strike (BF20a) and the Jump Kick / High Jump Kick crash's `getDamage` (BF20b, into Protect).
run('BF20a screens: the phase-1 shuffle on every beat up strike', 'gen3customgame', [1, 2, 3, 4],
  SET('Houndour', 'flashfire', 'beatup,splash') + ']' + SET('Registeel', 'clearbody', 'splash'),
  SET('Snorlax', 'thickfat', 'reflect,lightscreen,splash'), [M(2, 1), M(2, 2), M(1, 3)]);
run('BF20b screens: the phase-1 shuffle on the high jump kick crash', 'gen3customgame', [1, 2, 3, 4],
  SET('Hitmonlee', 'limber', 'highjumpkick,splash'),
  SET('Snorlax', 'thickfat', 'reflect,lightscreen,protect'), [M(2, 1), M(2, 2), M(1, 3)]);
// ---- BF21 a CALLED move sets the user's lastMoveUsed (gen 3's useMoveInner, data/mods/gen3/scripts.ts:127) ----
// Every `useMove` sets `pokemon.lastMoveUsed`, the nested ones included: a Sleep-Talk-called Earthquake (BF21a) and a
// Snatch-stolen Swords Dance (BF21b) are what a faster foe's Conversion 2 then reads (Flying resists Ground; Ghost is
// immune to Normal). The port kept Sleep Talk / Snatch as the snatcher's or sleeper's last move used.
run('BF21a called move: sleep talk sets lastMoveUsed', 'gen3customgame', [1, 2, 3, 4],
  SET('Snorlax', 'thickfat', 'sleeptalk,earthquake'), 'Porygon2||none|trace|spore,conversion2,splash|Hardy|85,85,85,85,85,85|N||||',
  [M(1, 1), M(1, 2)]);
run('BF21b called move: a snatch steal sets lastMoveUsed', 'gen3customgame', [1, 2, 3, 4],
  SET('Snorlax', 'thickfat', 'snatch,splash'), 'Porygon2||none|trace|swordsdance,conversion2|Hardy|85,85,85,85,85,85|N||||',
  [M(1, 1), M(2, 2)]);
// ---- BF22 (review F1) a PRESENT heal still runs the target's Hit-event reactions ----
// `runMoveEffects` runs `runEvent('Hit', target, source, move)` after Present's heal branch (sim/battle-actions.ts);
// Present is Physical, so Rage boosts (BF22a) and a pending Focus Punch loses focus (BF22b, `cant … Focus Punch`).
run('BF22a present heal: rage boosts', 'gen3customgame', [17, 18, 19, 20], SET('Delibird', 'hustle', 'present,splash'),
  SET('Snorlax', 'thickfat', 'rage,splash'), [M(1, 1), M(1, 1), M(1, 1)]);
run('BF22b present heal: focus punch loses focus', 'gen3customgame', [5, 6, 7, 8], SET('Delibird', 'hustle', 'present,splash'),
  SET('Snorlax', 'thickfat', 'focuspunch,splash'), [M(1, 1), M(1, 1), M(1, 1)]);
// ---- BF23 (review F4) BEAT UP is typeless (gen 3 onModifyMove '???'), so Wonder Guard never blocks it ----
// A Porygon2 that Traced Shedinja's Wonder Guard takes every strike (the gen-4 wonderguard onTryHit returns for '???').
run('BF23a beat up: through a traced wonder guard', 'gen3customgame', [1, 2, 3, 4],
  SET('Shedinja', 'wonderguard', 'splash') + ']' + SET('Houndoom', 'flashfire', 'beatup,splash'),
  'Porygon2||none|trace|splash,tackle|Hardy|85,85,85,85,85,85|N||||', [['switch 2', 'move 1'], M(1, 1)]);
// ---- BF24 (review F5) a base power floored to 0 by the BasePower chain: getDamage returns 0 ----
// Endure leaves Camerupt at 1 HP; Eruption's BP is 1 after `clampIntRange`, and Thick Fat's chainModify(0.5) floors it
// to 0 (BF24a), as does Water Sport (BF24b): the crit roll draws, then no lines, no damage roll, a 0-delta `-damage`.
run('BF24a bp zero: eruption into thick fat', 'gen3customgame', [1, 2, 3, 4],
  SET('Camerupt', 'magmaarmor', 'endure,eruption'),
  SET('Swampert', 'torrent', 'earthquake,splash') + ']' + SET('Snorlax', 'thickfat', 'splash'), [M(1, 1), ['move 2', 'switch 2']]);
run('BF24b bp zero: eruption into water sport', 'gen3customgame', [1, 2, 3, 4],
  SET('Camerupt', 'magmaarmor', 'endure,eruption'), SET('Swampert', 'torrent', 'earthquake,watersport'), [M(1, 1), M(2, 2)]);
// ---- BF25 (the B11 agent's finding, fuzz rmvq9o5ho_ab_2_14) an infatuation ends at the Update after its SOURCE faints ----
// `attract.condition.onUpdate` (data/moves.ts) removes the volatile once its source is not active (`-end|…|Attract|[silent]`).
// A Future Sight strike in the residual runs gen 3's generic `hitStepMoveHitLoop` (sim/battle-actions.ts): an Update,
// `faintMessages`, then a second Update, so a source the strike KOs ends the infatuation right after `|faint|`, before
// `|upkeep|` and the replacement (BF25a). (A source KO'd by a move is replaced first; the switch path handles that.)
run('BF25a attract: a future sight ko of the source ends it at once', 'gen3customgame', [1, 2, 3, 4],
  'Clefairy||none|cutecharm|attract,splash|Hardy|85,85,85,85,85,85|M|||5|]' + SET('Snorlax', 'thickfat', 'splash'),
  'Duskull||none|levitate|futuresight,splash|Hardy|85,85,85,85,85,85|F||||', [M(1, 1), M(2, 2), M(2, 2), ['switch 2', '']]);
// ---- BF26 (the B11 agent's fuzz find rmvq6vp32_ab_1_4) the in-tryMoveHit Update precedes the attacker's SHELL BELL ----
// gen 3's `tryMoveHit` ends with `eachEvent('Update')` (data/mods/gen3/scripts.ts:470), BEFORE `useMove` runs
// `AfterMoveSecondarySelf` (Shell Bell): a Dynamic Punch that confuses a Persim Berry holder sees the berry eaten
// first, then the heal. The port ran that Update at the caller, after the heal.
run('BF26a shell bell: after the update that eats a persim berry', 'gen3customgame', [2, 3, 4, 5],
  'Machamp||shellbell|guts|dynamicpunch,splash|Hardy|85,85,85,85,85,85|M||||',
  'Snorlax||persimberry|thickfat|bodyslam,splash|Hardy|85,85,85,85,85,85|M||||', [M(2, 1), M(1, 2)]);
