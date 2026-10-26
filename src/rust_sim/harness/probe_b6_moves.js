// PROBE: gen3 full parity B6 (`gen3_b6_moves_v1`) — SWAGGER, FLATTER, FORESIGHT, ODOR SLEUTH,
// MAGIC COAT. Synchronous `Battle`, the omniscient split resolution, every PRNG call logged per
// decision (`randomChance(n,d)` logs its inner `random(d)` first: ONE draw, two log entries).
//
// RUN IT: `node src/rust_sim/harness/probe_b6_moves.js [label-filter]`
//
// ===================================== THE SETTLED SPEC =====================================
// The oracle is the RESOLVED `Dex.mod('gen3')`; file:line cites are `deps/pokemon-showdown`.
//
// SWAGGER (acc 90, `boosts: {atk: 2}`) / FLATTER (acc 100, `boosts: {spa: 1}`), both
// `volatileStatus: 'confusion'`, flags {protect, reflectable, mirror}, no bypasssub (data/moves.ts
// swagger / flatter; no mod overrides them). Order = gen-3 `tryMoveHit` (data/mods/gen3/
// scripts.ts tryMoveHit) then `spreadMoveHit` / `runMoveEffects` (sim/battle-actions.ts):
//   accuracy roll (1 draw) -> miss: `[miss]` + `-miss` (SW13)
//   TryHit: Protect `-activate|<t>|Protect` (SW9); Magic Coat bounces (MC3, MC24)
//   TryPrimaryHit: Substitute -> `|move|<u>|Swagger||[still]` + `-fail|<u>` (SW7)
//   boosts: `-boost|<t>|atk|<capped delta>`, the delta-0 line PRINTED at +6 (SW3), +1 from +5;
//           a raise, so Clear Body does not gate it (SW17)
//   confusion: Own Tempo `-immune|<t>|confusion|[from] ability: Own Tempo` (SW5), already
//           confused SILENT (SW2), Safeguard `-activate|<t>|move: Safeguard` (SW8), else
//           random(2,6) + `-start|<t>|confusion` (SW1); a confusion berry eats at the Update (SW12)
//   -fail: only a delta-0 boost (null) AND an already-confused target (false) combine to false:
//           `[still]` retro-edit + `-fail|<u>` AFTER the `-boost|..|0` line (SW4). Own Tempo /
//           Safeguard return null: a capped Swagger into either does NOT fail (SW6, SW18).
//
// FORESIGHT / ODOR SLEUTH (gen-3 acc 100, data/mods/gen3/moves.ts; flags {protect, mirror,
// bypasssub}, NOT reflectable, data/mods/gen4/moves.ts), both `volatileStatus: 'foresight'`:
//   accuracy roll vs the target's evasion (FS7: 50 into +3); an already-identified target's
//           positive evasion is zeroed for it too (FS13 d9: 100 into +6)
//   Protect blocks (FS5); a Substitute does NOT (FS4); Magic Coat does NOT bounce it (MC31)
//   already identified (either move): `[still]` + `-fail|<u>` after the accuracy roll (FS2, FS3)
//   success: `-start|<t>|Foresight` for BOTH moves (FS6), no draw beyond the accuracy roll
//   the volatile (data/moves.ts foresight condition): onModifyBoost zeroes POSITIVE evasion in
//           every accuracy check against the holder (FS7 d5: Cross Chop rolls 80 into +3);
//           onNegateImmunity: Normal / Fighting hit through Ghost, the Ghost factor NEUTRAL
//           (FS1: Cross Chop `-resisted` on Ghost/Poison Gengar, Body Slam neutral); Glare too
//           (FS8); Seismic Toss (FS10); cleared on switch-out (FS9); noCopy (Baton Pass).
//
// MAGIC COAT (priority +4, accuracy true, target self, flags {metronome}; condition duration 1,
// data/moves.ts magiccoat; the gen-4 mod's onTryHit removes the coat first and drops
// onAllyTryHitSide, data/mods/gen4/moves.ts magiccoat):
//   cast: `-singleturn|<u>|move: Magic Coat`, draw-free (MC1); an unused coat adds the residual
//           duration handler (MC1b mirror: one tie-shuffle more than the MC1c Splash control)
//   bounce: TryHit priority 2 (after Protect's 3; TryHit sorts left-to-right, no draw), AFTER the
//           original move's accuracy roll and BEFORE Soundproof (MC16b), the natural-immunity
//           report (MC15: Thunder Wave into a GROUND holder bounces) and the move's own onTryHit:
//           `|move|<holder>|<Move>|<user>|[from] Magic Coat` (MC2), then the move's full
//           `tryMoveHit` against the user with its OWN accuracy roll (the bounce's only draws),
//           the user's Substitute (MC11: `|move|<h>|Screech||[from] Magic Coat|[still]` +
//           `-fail|<h>`), immunity (MC5, MC6, MC13), ability (MC12, MC16, MC17), Synchronize
//           (MC20), Mental Herb (MC23); a missed original move is not bounced and keeps the coat
//           (MC14, MC21). No PP is charged to the holder; its `lastMoveUsed` becomes the bounced
//           move (Conversion 2 reads it), its `lastMove` stays Magic Coat (Encore locks it).
//   not reflectable in gen 3: Taunt (MC18), Foresight (MC31), Teeter Dance (MC32).
//   Snatch: no interplay (Magic Coat is not snatchable; MC19, MC19b).
// ============================================================================================
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

let draws = null;
for (const fn of ['random', 'randomChance', 'sample', 'shuffle']) {
  const o = PRNG.prototype[fn];
  PRNG.prototype[fn] = function (...a) {
    const r = o.apply(this, a);
    if (draws) draws.push(`${fn}(${a.map((x) => (Array.isArray(x) ? `[${x.length}]` : x)).join(',')})->${Array.isArray(r) ? '' : r}`);
    return r;
  };
}

const S = (sp, ab, moves, o = {}) =>
  `${sp}|${sp}|${o.item || 'Leftovers'}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;

const FILTER = process.argv[2] || '';
function run(label, p1, p2, script, o = {}) {
  if (FILTER && !label.includes(FILTER)) return;
  const battle = new Battle({ formatid: o.format || 'gen3customgame', seed: o.seed || [9, 9, 9, 9] });
  battle.setPlayer('p1', { name: 'P1', team: p1 });
  battle.setPlayer('p2', { name: 'P2', team: p2 });
  const out = [];
  for (const [k, [a, b]] of script.entries()) {
    draws = [];
    const from = battle.log.length;
    if (a) battle.choose('p1', a);
    if (b) battle.choose('p2', b);
    const lines = [];
    for (let i = from; i < battle.log.length; i++) {
      const l = battle.log[i];
      if (l.startsWith('|split|')) { lines.push(battle.log[i + 1]); i += 2; continue; }
      if (l === '' || l.startsWith('|t:|') || l.startsWith('|upkeep') || l.startsWith('|debug')) continue;
      lines.push(l);
    }
    out.push(`  -- decision ${k + 1} [${a} / ${b}]\n    ${lines.join('\n    ')}\n    DRAWS ${draws.length}: ${draws.join('  ')}`);
    draws = null;
  }
  console.log(`\n== ${label}\n${out.join('\n')}`);
}
const M = (a, b) => [`move ${a}`, `move ${b}`];

// ---------------- SWAGGER / FLATTER ----------------
const SW = S('Nidoking', 'PoisonPoint', ['swagger', 'flatter', 'splash', 'tackle']);
const LAX = (mv = ['splash', 'bellydrum', 'substitute', 'protect'], o = {}) => S('Snorlax', o.ab || 'ThickFat', mv, { evs: '85,85,85,85,85,85', ...o });
run('SW1 swagger plain', SW, LAX(), [M(1, 1)]);
run('SW2 swagger into confused', SW, LAX(), [M(1, 1), M(1, 1)]);
run('SW3 swagger at +6 (belly drum)', SW, LAX(), [M(3, 2), M(1, 1)]);
run('SW4 swagger at +6 and confused', SW, LAX(), [M(3, 2), M(1, 1), M(1, 1)]);
run('SW5 swagger into own tempo', SW, S('Slowbro', 'OwnTempo', ['splash', 'bellydrum']), [M(1, 1)]);
run('SW6 own tempo at +6', SW, S('Slowbro', 'OwnTempo', ['splash', 'bellydrum']), [M(3, 2), M(1, 1)]);
run('SW7 swagger into substitute', SW, LAX(), [M(3, 3), M(1, 1)]);
run('SW8 swagger into safeguard', SW, LAX(['splash', 'safeguard']), [M(3, 2), M(1, 1)]);
run('SW9 swagger into protect', SW, LAX(['splash', 'protect'], { evs: '85,85,85,85,85,252' }), [M(1, 2)]);
run('SW10 flatter plain', SW, LAX(), [M(2, 1)]);
run('SW11 flatter into confused', SW, LAX(), [M(2, 1), M(2, 1)]);
run('SW12 swagger into persim-like berry', SW, LAX(['splash'], { item: 'Bitter Berry' }), [M(1, 1)]);
run('SW13 swagger miss vs double team', SW, LAX(['doubleteam', 'splash'], { evs: '85,85,85,85,85,252' }), [M(3, 1), M(3, 1), M(3, 1), M(1, 2), M(1, 2), M(1, 2)]);
run('SW14 flatter at +6 spa', SW, LAX(['splash', 'tailglow']), [M(3, 2), M(3, 2), M(3, 2), M(2, 1)]);
run('SW15 swagger at +5 (caps at +1)', SW, LAX(['splash', 'swordsdance', 'growl']), [M(3, 3), M(3, 2), M(3, 2), M(3, 2), M(1, 1)]);
run('SW16 swagger into fainting? (KO later) / soundproof control', SW, S('Electrode', 'Soundproof', ['splash']), [M(1, 1)]);
run('SW18 capped swagger into safeguard (no fail)', SW, LAX(['splash', 'safeguard', 'bellydrum']), [M(3, 2), M(3, 3), M(1, 1)]);
run('SW17 swagger into hyper cutter/clear body (raises, not drops)', SW, S('Metagross', 'ClearBody', ['splash']), [M(1, 1)]);

// ---------------- FORESIGHT / ODOR SLEUTH ----------------
const FS = S('Machamp', 'Guts', ['foresight', 'odorsleuth', 'crosschop', 'bodyslam']);
const GEN = (mv = ['splash', 'substitute', 'protect', 'doubleteam'], o = {}) => S('Gengar', 'Levitate', mv, o);
run('FS1 foresight into ghost then fighting + normal', FS, GEN(), [M(1, 1), M(3, 1), M(4, 1)]);
run('FS1b control: no foresight', FS, GEN(), [M(3, 1), M(4, 1)]);
run('FS2 foresight twice', FS, GEN(), [M(1, 1), M(1, 1)]);
run('FS3 odor sleuth after foresight', FS, GEN(), [M(1, 1), M(2, 1)]);
run('FS4 foresight into substitute', FS, GEN(), [M(4, 2), M(1, 1), M(3, 1)]);
run('FS5 foresight into protect', FS, GEN(['splash', 'protect'], { evs: '85,85,85,85,85,252' }), [M(1, 2)]);
run('FS6 odor sleuth plain into normal', FS, LAX(), [M(2, 1)]);
run('FS7 foresight vs evasion: double team x3 then foresight then attack', FS, GEN(['doubleteam', 'splash']),
  [M(4, 1), M(4, 1), M(4, 1), M(1, 2), M(3, 2)]);
run('FS8 glare into foresighted ghost', S('Arbok', 'Intimidate', ['foresight', 'glare', 'splash']), GEN(['splash']), [M(2, 1), M(1, 1), M(2, 1)]);
run('FS9 foresighted ghost switches out and back', FS, GEN(['splash']) + ']' + S('Blissey', 'NaturalCure', ['splash'], { g: 'F' }),
  [M(1, 1), ['move 4', 'switch 2'], ['move 4', 'switch 2'], M(4, 1)]);
run('FS10 foresight user seismic toss on ghost', S('Machamp', 'Guts', ['foresight', 'seismictoss', 'counter', 'splash']), GEN(['splash', 'shadowball']), [M(1, 1), M(2, 1)]);
run('FS11 baton pass does not pass foresight', FS, GEN(['batonpass', 'splash']) + ']' + S('Misdreavus', 'Levitate', ['splash'], { g: 'F' }),
  [M(1, 1), M(4, 1), [null, 'switch 2'], M(4, 1)]);
run('FS12 foresight into sweet scent (negative evasion kept)', FS, GEN(['splash']), [M(1, 1)]);
run('FS13 foresight miss into double team', FS, GEN(['doubleteam', 'splash'], { evs: '85,85,85,85,85,252' }), [M(4, 1), M(4, 1), M(4, 1), M(4, 1), M(4, 1), M(4, 1), M(1, 1), M(1, 1), M(1, 1)]);

// ---------------- MAGIC COAT ----------------
const MC = (mv = ['magiccoat', 'splash'], o = {}) => S('Alakazam', o.ab || 'Synchronize', mv, o);
const USER = (mv, o = {}) => S('Gardevoir', o.ab || 'Trace', mv, { g: 'F', ...o });
run('MC1 magic coat alone', MC(), USER(['splash']), [M(1, 1)]);
run('MC1b both magic coat (residual tie)', MC(), MC(), [M(1, 1)]);
run('MC1c both splash control', MC(), MC(), [M(2, 2)]);
run('MC2 thunder wave bounced', MC(), USER(['thunderwave']), [M(1, 1)]);
run('MC3 swagger bounced', MC(), USER(['swagger']), [M(1, 1)]);
run('MC4 spore bounced', MC(), USER(['spore']), [M(1, 1)]);
run('MC5 toxic bounced onto poison', MC(), S('Muk', 'StickyHold', ['toxic']), [M(1, 1)]);
run('MC6 leech seed bounced onto grass', MC(), S('Venusaur', 'Overgrow', ['leechseed']), [M(1, 1)]);
run('MC6b leech seed bounced', MC(), USER(['leechseed']), [M(1, 1)]);
run('MC7 attract bounced', MC(), USER(['attract']), [M(1, 1)]);
run('MC7b attract same gender bounced', MC(), USER(['attract'], { g: 'M' }), [M(1, 1)]);
run('MC8 mean look bounced', MC(), USER(['meanlook', 'splash']), [M(1, 1), M(2, 2)]);
run('MC9 yawn bounced', MC(), USER(['yawn', 'splash']), [M(1, 1), M(2, 2)]);
run('MC10 growl bounced', MC(), USER(['growl']), [M(1, 1)]);
run('MC11 screech bounced into sub', MC(), USER(['substitute', 'screech']), [M(2, 1), M(1, 2)]);
run('MC12 confuse ray bounced onto own tempo', MC(), S('Slowbro', 'OwnTempo', ['confuseray']), [M(1, 1)]);
run('MC13 thunder wave bounced onto ground', MC(), S('Marowak', 'RockHead', ['thunderwave']), [M(1, 1)]);
run('MC14 hypnosis bounced (misses)', MC(), USER(['hypnosis']), [M(1, 1)], { seed: [1, 2, 3, 4] });
run('MC15 thunder wave into ground coat holder', S('Marowak', 'RockHead', ['magiccoat']), USER(['thunderwave']), [M(1, 1)]);
run('MC16 supersonic bounced onto soundproof', MC(), S('Electrode', 'Soundproof', ['supersonic']), [M(1, 1)]);
run('MC16b supersonic into soundproof coat holder', S('Electrode', 'Soundproof', ['magiccoat']), USER(['supersonic']), [M(1, 1)]);
run('MC17 growl bounced onto clear body', MC(), S('Metagross', 'ClearBody', ['growl']), [M(1, 1)]);
run('MC18 taunt not reflectable', MC(), USER(['taunt']), [M(1, 1)]);
run('MC19 snatch vs magic coat', MC(), USER(['snatch']), [M(1, 1)]);
run('MC19b snatch user swords dance vs coat', MC(['magiccoat', 'swordsdance']), USER(['snatch', 'swordsdance']), [M(2, 1)]);
run('MC20 toxic bounced onto synchronize user', MC(), S('Espeon', 'Synchronize', ['toxic']), [M(1, 1)]);
run('MC21 original move misses (sing)', MC(), USER(['sing']), [M(1, 1)], { seed: [1, 2, 3, 4] });
run('MC22 pressure coat holder: pp', S('Zapdos', 'Pressure', ['magiccoat']), USER(['thunderwave', 'splash']), [M(1, 1), M(1, 2)]);
run('MC23 bounced attract onto mental herb', MC(), USER(['attract'], { item: 'Mental Herb' }), [M(1, 1)]);
run('MC24 flatter bounced', MC(), USER(['flatter']), [M(1, 1)]);
run('MC25 sand attack bounced', MC(), USER(['sandattack']), [M(1, 1)]);
run('MC26 block bounced', MC(), USER(['block', 'splash']), [M(1, 1), M(2, 2)]);
run('MC27 will-o-wisp bounced', MC(), USER(['willowisp']), [M(1, 1)]);
run('MC28 bounce into already statused user', MC(), USER(['thunderwave', 'toxic']), [M(1, 1), M(1, 2)]);
run('MC29 magic coat then foe uses it a turn later (expired)', MC(), USER(['splash', 'thunderwave']), [M(1, 1), M(2, 2)]);
run('MC30 spore bounced sleep clause gen3ou', MC(['magiccoat', 'spore', 'splash']), USER(['spore', 'splash']) + ']' + USER(['splash']),
  [M(2, 2), ['move 2', 'switch 2'], M(1, 1)], { format: 'gen3ou' });
run('MC31 magic coat user faster/slower irrelevant; coat + foresight (not reflectable)', MC(), FS, [M(1, 1)]);
run('MC32 foe faints... coat holder uses coat, foe substitutes: sub then teeterdance', MC(), USER(['teeterdance']), [M(1, 1)]);
run('MC33 encore after bounce: encore the coat holder', MC(['magiccoat', 'splash']), USER(['thunderwave', 'encore']), [M(1, 1), M(2, 2)]);
run('MC34 yawn into a STATUSED coat holder (the coat precedes yawn onTryHit)', MC(['magiccoat', 'splash'], { ab: 'InnerFocus' }), USER(['willowisp', 'yawn']), [M(2, 1), M(1, 2)]);
run('MC35 sub-holding coat holder vs yawn/leech seed', MC(['substitute', 'magiccoat'], { ab: 'InnerFocus' }), USER(['yawn', 'leechseed']), [M(1, 1), M(2, 2)]);
