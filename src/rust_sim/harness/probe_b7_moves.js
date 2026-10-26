// probe_b7_moves.js — gen3 full parity B7 (`gen3_b7_moves_v1`): Stockpile, Spit Up, Swallow, Mud
// Sport, Water Sport, Mist, Lock-On, Mind Reader, against the real sim (gen3customgame). Re-runnable;
// each board prints its lines with every PRNG draw attributed to the line that was last emitted
// when it drew (`  ~ N draw(s)`), and a state dump (volatiles with layers / duration, side
// conditions, boosts, PP):
//   node src/rust_sim/harness/probe_b7_moves.js "$PWD/deps/pokemon-showdown" [filter]
//
// SETTLED (2026-10-26, against the RESOLVED `Dex.mod('gen3')` — the mod-chain law; citations are the
// file that supplies the resolved handler; the resolved gen-3 move is printed by the `resolved` board):
//   * STOCKPILE (`data/moves.ts` stockpile: accuracy true, self, flags snatch; `onTry` fails at 3
//     layers; the condition is `data/mods/gen3/moves.ts` stockpile — NO Def / SpD boosts, noCopy):
//     ZERO draws. `|-start|<u>|stockpile1..3`; a fourth is the BARE announce (gen-3 `tryMoveHit`'s
//     `singleEvent('Try')` false adds no line, `data/mods/gen3/scripts.ts`). Switch-out and faint
//     clear it SILENTLY (`clearVolatile` runs no onEnd); Baton Pass does not pass it; Snatch steals it.
//   * SPIT UP (`data/moves.ts` spitup, untouched by the mods; the gen-3 `init` re-categorises Normal
//     as PHYSICAL): `onTry` `!!source.volatiles['stockpile']` → with none, the BARE announce, ZERO
//     draws (before accuracy). Else a 100 × layers BP Normal move with the ordinary gen-3 damage —
//     accuracy, crit, the `randomizer` random factor (`data/mods/gen3/scripts.ts` modifyDamage: it
//     applies to every move, Spit Up included), Protect / Ghost / Substitute / Wonder Guard / Counter
//     as any other; Taunt does not block it. `onAfterMove` → `removeVolatile('stockpile')` →
//     `|-end|<u>|Stockpile`, AFTER the damage and the in-tryMoveHit Update, for every outcome (miss,
//     Protect, immune), and for a Sleep-Talk-called Spit Up too (`sim/battle-actions.ts` runMove takes
//     the activeMove for AfterMove).
//   * SWALLOW (`onTry` from `data/mods/gen4/moves.ts` — no Snatch exemption, so a snatcher needs its
//     OWN stockpile; `onHit` from `data/mods/gen7/moves.ts`): ZERO draws; no stockpile → bare. Else
//     `this.heal(this.modify(maxhp, [0.25, 0.5, 1][layers - 1]))` → `|-heal|<u>|<HP>` (482 maxhp:
//     +120 / +241 / full), at full HP `|-fail|<u>|heal`; then `|-end|<u>|Stockpile` either way.
//   * MUD SPORT / WATER SPORT (`data/mods/gen5/moves.ts` makes each a `volatileStatus` on the USER —
//     target `all`, so Pressure costs it a PP; `data/mods/gen4/moves.ts` sets `noCopy: false`,
//     `onAnyBasePowerPriority: 3` and `chainModify(0.5)`; no duration): ZERO draws on the cast,
//     `|-start|<u>|Mud Sport` / `|-start|<u>|move: Water Sport`; a recast `[still]` + `-fail|<u>`.
//     While the holder is active every Electric (Mud) / Fire (Water) move's BASE POWER is halved —
//     the holder's own, the foe's, Hidden Power / Weather Ball by their runtime type; two holders
//     quarter it. Baton Pass passes it; switch-out / faint end it silently. THE DRAW: every sport
//     volatile on an active mon WITH HP (`alliesAndSelf()` / `foes()` filter on `hp`, so an exploding
//     holder drops out) is a BasePower handler at priority 3, speed = its holder's, subOrder 2, so two
//     at one speed (both sports on one mon, or one each on equal-speed mons) TIE: `speedSort`
//     shuffles them, k-1 draws, on EVERY getDamage that reaches BasePower — any damaging move (per
//     strike), Beat Up per strike, the confusion self-hit, a High Jump Kick crash, a Future Sight
//     cast — after the crit roll, before ModifyDamagePhase1. No other gen-3 BasePower handler has
//     priority 3 (Helping Hand 10, Charge 9, the pinch abilities 2, Thick Fat 1, the rest 0).
//   * MIST (`data/moves.ts` mist: allySide, snatch; condition `duration: 5`, the gen-4 mod's
//     `onSideResidualOrder: 3`): ZERO draws; `|-sidestart|<side>|Mist`, a recast `[still]` +
//     `-fail|<u>`. `onTryBoost`: for a boost whose source is not the target, every NEGATIVE key of the
//     CAPPED boost is deleted — after the target's own Clear Body / Hyper Cutter / Keen Eye (the side
//     condition has speed 0, the mon its speed), so only the ability's line prints when it already
//     blocked; a stat at −6 (capped to 0) survives and prints `-unboost|…|0`. `|-activate|<t>|move:
//     Mist` unless the effect is a move SECONDARY: Growl / Screech / Charm / Tickle / Memento /
//     Intimidate show it; Psychic / Icy Wind / Crunch drop silently (the chance roll still drew).
//     Self drops (Overheat, Superpower, Curse) are untouched; Haze clears boosts and leaves Mist up; a
//     Substitute blocks a non-bypasssub status drop first. `|-sideend|<side>|Mist` at the fifth
//     residual; both sides misted tie at order 3 (one draw a residual), and a side's conditions are
//     gathered in `sideConditions` INSERTION order, which decides a tie's `-sideend` order.
//   * LOCK-ON / MIND READER (`data/moves.ts` with the gen-3 mod's `accuracy: 100`; the condition's
//     `noCopy: false` from `data/mods/gen4/moves.ts`; flags protect, mirror — no bypasssub): in
//     gen-3 `tryMoveHit` order: the Invulnerability miss (zero draws), the accuracy roll, then
//     TryHit — Protect (priority 3, `-activate|<t>|Protect`) before the move's `onTryHit` (the user
//     already holds `lockon` → `[still]` + `-fail|<u>`), then a Substitute (`[still]` + `-fail|<u>`).
//     Success: `source.addVolatile('lockon', target)` and `|-activate|<u>|move: Lock-On|[of] <t>`
//     (`move: Mind Reader`). The volatile names the target (the Pokemon object: a replacement is not
//     locked, the same mon switched back is): the holder's moves against it skip the accuracy roll
//     (`onSourceAccuracy` → true: no draw — damaging, status, a Sleep-Talk call, its Future Sight
//     strike; a held lock's own recast draws nothing either) and are not missed mid-Fly / Dig / Dive /
//     Bounce (`onSourceInvulnerability`, priority 1 → 0 stops the event: Ice Beam hits a Dig-ing
//     Golem, Perish Song reaches an airborne Swellow). `duration: 2`: a NO_ORDER / subOrder-2
//     residual duration handler at the holder's speed, ended silently — it covers the next turn and
//     lapses; a Lock-On mirror at equal speed draws one tie per residual. Baton Pass passes it (target
//     and duration); the holder's switch-out / faint clears it. Pressure applies. The OHKO moves stay
//     fail-loud (another batch owns them), so Lock-On's sure-hit on them is not exercised here.
'use strict';
const path = require('path');
const PS = process.argv[2] || path.resolve(__dirname, '../../../deps/pokemon-showdown');
const FILTER = process.argv[3] || '';
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' });
function run(label, p1, p2, script, opts = {}) {
  if (FILTER && !label.includes(FILTER)) return;
  const b = new Battle({ formatid: opts.format || 'gen3customgame', seed: opts.seed || [17, 29, 43, 59] });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: p2 });
  const rng = b.prng.rng;
  const next = rng.next.bind(rng);
  const at = [];
  rng.next = (...a) => { at.push(b.log.length); return next(...a); };
  const start = b.log.length;
  for (const [x, y] of script) { if (x) b.choose('p1', x); if (y) b.choose('p2', y); }
  console.log(`== ${label}`);
  for (let i = start; i < b.log.length; i++) {
    const l = b.log[i];
    if (l.startsWith('|split|')) continue;
    let n = at.filter((k) => k === i + 1).length;
    if (i > 0 && b.log[i - 2] && b.log[i - 2].startsWith('|split|')) continue;
    if (b.log[i - 1] && b.log[i - 1].startsWith('|split|')) n += at.filter((k) => k === i + 2).length;
    const shown = !(l === '' || l === '|' || l.startsWith('|t:|') || l.startsWith('|debug|') || l === '|upkeep');
    if (shown) console.log(l + (n ? `   ~ ${n} draw(s)` : ''));
    else if (n) console.log(`   (${l || 'blank'})   ~ ${n} draw(s)`);
  }
  if (opts.after) opts.after(b);
}
const pp = (b) => { for (const s of b.sides) for (const p of s.pokemon) console.log(`   [${p.name} ${p.hp}/${p.maxhp} ${p.status || '-'} vol=${Object.entries(p.volatiles).map(([k,v])=>k+(v.layers?':'+v.layers:'')+(v.duration?'/d'+v.duration:''))} sc=${Object.entries(s.sideConditions).map(([k,v])=>k+'/d'+v.duration)} boosts=${JSON.stringify(p.boosts)} pp=${p.moveSlots.map((m) => `${m.id}:${m.pp}`)}]`); };
const mv = (a, c) => [a && `move ${a}`, c && `move ${c}`];

if (require.main === module) {
  const R = require(path.join(PS, 'dist/sim')).Dex.mod('gen3');
  if (!FILTER || 'resolved'.includes(FILTER)) {
    console.log('== resolved');
    for (const id of ['stockpile', 'spitup', 'swallow', 'mudsport', 'watersport', 'mist', 'lockon', 'mindreader']) {
      const m = R.moves.get(id);
      const c = m.condition || {};
      console.log(`   ${id}: ${m.category} ${m.type} bp=${m.basePower} acc=${m.accuracy} target=${m.target} flags=${Object.keys(m.flags)} ` +
        `vs=${m.volatileStatus || '-'} sc=${m.sideCondition || '-'} cond{noCopy=${c.noCopy} duration=${c.duration} bpPrio=${c.onAnyBasePowerPriority} sideResidualOrder=${c.onSideResidualOrder}}`);
    }
  }
  const J = set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'splash']);
  // STOCKPILE / SPIT UP / SWALLOW
  run('stockpile x4 then spit up', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), BENCH, [mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(2, 1)], { after: pp });
  run('spit up / swallow at 0 layers', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), BENCH, [mv(2, 1), mv(3, 1)], { after: pp });
  run('swallow 1/2/3 layers', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), [mv(1, 1), mv(3, 1), mv(1, 1), mv(1, 1), mv(3, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(3, 1)], { after: pp });
  run('swallow at full hp', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), BENCH, [mv(1, 1), mv(3, 1)], { after: pp });
  run('spit up 3 layers (random factor: seed a)', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), set('Snorlax', '', 'ThickFat', ['splash'], { evs: '252,0,252,0,4,0' }), [mv(1, 1), mv(1, 1), mv(1, 1), mv(2, 1)], { seed: [3, 3, 3, 3], after: pp });
  run('spit up 3 layers (random factor: seed b)', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), set('Snorlax', '', 'ThickFat', ['splash'], { evs: '252,0,252,0,4,0' }), [mv(1, 1), mv(1, 1), mv(1, 1), mv(2, 1)], { seed: [4, 3, 3, 3], after: pp });
  run('spit up into Ghost', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), set('Gengar', '', 'Levitate', ['splash']), [mv(1, 1), mv(2, 1)], { after: pp });
  run('spit up into Protect', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), set('Gengar', '', 'Levitate', ['protect', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
  run('spit up miss vs +6 evasion', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), set('Blissey', '', 'NaturalCure', ['doubleteam', 'splash'], { g: 'F' }), [mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(2, 2), mv(2, 2)], { after: pp });
  run('spit up into substitute', set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), set('Blissey', '', 'NaturalCure', ['substitute', 'splash'], { g: 'F' }), [mv(1, 1), mv(2, 2)], { after: pp });
  run('stockpile switch out (silent) + baton pass', team(set('Snorlax', '', 'ThickFat', ['stockpile', 'batonpass']), BENCH), BENCH, [mv(1, 1), ['switch 2', 'move 1'], ['switch 2', 'move 1'], mv(1, 1), mv(2, 1), ['switch 2', null]], { after: pp });
  run('stockpile faint clears', team(set('Pichu', '', 'Static', ['stockpile', 'splash'], { lv: 5 }), set('Snorlax', '', 'ThickFat', ['spitup', 'splash'])), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 2), mv(1, 1), ['switch 2', null], mv(1, 2)], { after: pp });
  run('snatch stockpile / swallow', set('Snorlax', '', 'ThickFat', ['stockpile', 'swallow']), set('Sableye', '', 'KeenEye', ['snatch', 'splash', 'stockpile', 'swallow']), [mv(1, 1), mv(2, 3), mv(2, 1)], { after: pp });
  run('sleep talk -> spit up', set('Snorlax', '', 'ThickFat', ['stockpile', 'rest', 'sleeptalk', 'spitup']), set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), [mv(1, 1), mv(1, 1), mv(2, 1), mv(3, 1), mv(3, 1), mv(3, 1)], { after: pp });
  // MUD SPORT / WATER SPORT
  run('mud sport: control', set('Snorlax', '', 'ThickFat', ['splash', 'mudsport']), J, [mv(1, 2), mv(1, 1)], { after: pp });
  run('mud sport: halved', set('Snorlax', '', 'ThickFat', ['mudsport', 'splash']), J, [mv(1, 2), mv(2, 1)], { after: pp });
  run('mud sport: both mons quarter', set('Snorlax', '', 'ThickFat', ['mudsport', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'mudsport']), [mv(1, 2), mv(2, 1)], { after: pp });
  run('mud sport recast fails', set('Swampert', '', 'Torrent', ['mudsport', 'thunderpunch']), J, [mv(1, 2), mv(1, 2)], { after: pp });
  run('water sport vs flamethrower', set('Swampert', '', 'Torrent', ['watersport', 'splash']), set('Houndoom', '', 'EarlyBird', ['flamethrower', 'splash']), [mv(1, 1), mv(2, 1)], { after: pp });
  run('both sports same mon: tackle tie draw', set('Swampert', '', 'Torrent', ['watersport', 'mudsport', 'splash']), set('Houndoom', '', 'EarlyBird', ['tackle', 'splash']), [mv(1, 2), mv(2, 2), mv(3, 1)], { after: pp });
  run('one sport: tackle control', set('Swampert', '', 'Torrent', ['watersport', 'splash', 'splash']), set('Houndoom', '', 'EarlyBird', ['tackle', 'splash']), [mv(1, 2), mv(2, 2), mv(3, 1)], { after: pp });
  run('mud sport both mons equal speed', set('Swampert', '', 'Torrent', ['mudsport', 'tackle'], { evs: '0,0,0,0,0,0' }), set('Swampert', '', 'Torrent', ['mudsport', 'tackle'], { evs: '0,0,0,0,0,0' }), [mv(1, 1), mv(2, 2)], { after: pp });
  run('mud sport baton pass', team(set('Swampert', '', 'Torrent', ['mudsport', 'batonpass']), set('Zapdos', '', 'Pressure', ['thunderbolt'])), J, [mv(1, 2), mv(2, 2), ['switch 2', null], mv(1, 1)], { after: pp });
  run('sports vs pressure (target all)', set('Swampert', '', 'Torrent', ['mudsport', 'watersport']), set('Zapdos', '', 'Pressure', ['thunderbolt', 'splash']), [mv(1, 2), mv(2, 2)], { after: pp });
  run('both sports: confusion self-hit', set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'splash']), set('Gengar', '', 'Levitate', ['confuseray', 'splash']), [mv(1, 2), mv(2, 1), mv(3, 2), mv(3, 2), mv(3, 2)], { after: pp });
  run('confusion self-hit control', set('Snorlax', '', 'ThickFat', ['splash', 'splash', 'splash']), set('Gengar', '', 'Levitate', ['confuseray', 'splash']), [mv(1, 2), mv(2, 1), mv(3, 2), mv(3, 2), mv(3, 2)], { after: pp });
  run('both sports: holder explodes (0 HP drops out)', team(set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'explosion']), BENCH), BENCH, [mv(1, 1), mv(2, 1), mv(3, 1)], { after: pp });
  run('explode control', team(set('Snorlax', '', 'ThickFat', ['splash', 'splash', 'explosion']), BENCH), BENCH, [mv(1, 1), mv(2, 1), mv(3, 1)], { after: pp });
  run('both sports: beat up per strike', set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'splash']), team(set('Sneasel', '', 'InnerFocus', ['beatup', 'splash']), BENCH), [mv(1, 2), mv(2, 2), mv(3, 1)]);
  run('beat up control', set('Snorlax', '', 'ThickFat', ['splash', 'splash', 'splash']), team(set('Sneasel', '', 'InnerFocus', ['beatup', 'splash']), BENCH), [mv(1, 2), mv(2, 2), mv(3, 1)]);
  run('both sports: high jump kick crash', set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'protect']), set('Hitmonlee', '', 'Limber', ['highjumpkick', 'splash']), [mv(1, 2), mv(2, 2), mv(3, 1)]);
  run('hjk crash control', set('Snorlax', '', 'ThickFat', ['splash', 'splash', 'protect']), set('Hitmonlee', '', 'Limber', ['highjumpkick', 'splash']), [mv(1, 2), mv(2, 2), mv(3, 1)]);
  run('water sport: weather ball in sun', set('Snorlax', '', 'ThickFat', ['watersport', 'splash']), set('Castform', '', 'Forecast', ['sunnyday', 'weatherball']), [mv(1, 1), mv(2, 2)], { after: pp });
  run('sleep talk -> mist / mud sport', set('Snorlax', '', 'ThickFat', ['rest', 'sleeptalk', 'mudsport', 'mist']), set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), [mv(1, 1), mv(2, 1), mv(2, 1), mv(2, 1)], { seed: [5, 6, 7, 8], after: pp });
  // MIST
  run('mist vs growl / psychic secondary / screech, end', set('Lapras', '', 'ShellArmor', ['mist', 'splash']), set('Alakazam', '', 'Synchronize', ['growl', 'psychic', 'screech', 'splash']), [mv(1, 1), mv(2, 2), mv(2, 3), mv(2, 4), mv(2, 4), mv(2, 4), mv(2, 1)], { after: pp });
  run('mist vs icy wind 100% secondary', set('Lapras', '', 'ShellArmor', ['mist', 'splash']), set('Alakazam', '', 'Synchronize', ['icywind', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
  run('mist recast fails', set('Lapras', '', 'ShellArmor', ['mist', 'splash']), set('Alakazam', '', 'Synchronize', ['splash']), [mv(1, 1), mv(1, 1)], { after: pp });
  run('mist vs intimidate', team(set('Lapras', '', 'ShellArmor', ['mist', 'splash']), BENCH), team(BENCH, set('Gyarados', '', 'Intimidate', ['splash'])), [mv(1, 1), ['move 2', 'switch 2']], { after: pp });
  run('mist + self drops', set('Lapras', '', 'ShellArmor', ['mist', 'overheat', 'superpower', 'curse']), BENCH, [mv(1, 1), mv(2, 1), mv(3, 1), mv(4, 1)], { after: pp });
  run('mist + haze', set('Lapras', '', 'ShellArmor', ['mist', 'haze', 'swordsdance']), set('Blissey', '', 'NaturalCure', ['splash', 'haze'], { g: 'F' }), [mv(1, 1), mv(3, 1), mv(2, 1), mv(1, 1)], { after: pp });
  run('mist vs tickle / charm / memento', set('Lapras', '', 'ShellArmor', ['mist', 'splash']), team(set('Blissey', '', 'NaturalCure', ['tickle', 'charm', 'memento'], { g: 'F' }), BENCH), [mv(1, 1), mv(2, 1), mv(2, 2), mv(2, 3)], { after: pp });
  run('mist + clear body vs growl', set('Metagross', '', 'ClearBody', ['mist', 'splash']), set('Blissey', '', 'NaturalCure', ['growl'], { g: 'F' }), [mv(1, 1), mv(2, 1)], { after: pp });
  run('mist + hyper cutter vs intimidate', team(set('Kingler', '', 'HyperCutter', ['mist', 'splash']), BENCH), team(BENCH, set('Gyarados', '', 'Intimidate', ['splash'])), [mv(1, 1), ['move 2', 'switch 2']], { after: pp });
  run('mist + atk at -6: tickle / intimidate', team(set('Lapras', '', 'ShellArmor', ['mist', 'splash']), BENCH), team(set('Blissey', '', 'NaturalCure', ['charm', 'tickle'], { g: 'F' }), set('Gyarados', '', 'Intimidate', ['splash'])), [mv(2, 1), mv(2, 1), mv(2, 1), mv(1, 2), ['move 2', 'switch 2']], { after: pp });
  run('mist through a sub: tickle (bypasssub) / sand attack', set('Lapras', '', 'ShellArmor', ['mist', 'substitute']), set('Sableye', '', 'KeenEye', ['tickle', 'sandattack', 'splash']), [mv(1, 3), mv(2, 1), mv(2, 2)], { after: pp });
  run('both sides mist same turn: residual tie + end', set('Lapras', '', 'ShellArmor', ['mist', 'splash']), set('Articuno', '', 'Pressure', ['mist', 'splash']), [mv(1, 1), mv(2, 2), mv(2, 2), mv(2, 2), mv(2, 2), mv(2, 2)], { after: pp });
  run('mist + screens: insertion order decides the tie', set('Lapras', '', 'ShellArmor', ['reflect', 'mist', 'safeguard', 'splash']), set('Articuno', '', 'Pressure', ['lightscreen', 'mist', 'reflect', 'splash']), [mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4), mv(4, 4), mv(4, 4)], { after: pp });
  run('snatch mist', set('Lapras', '', 'ShellArmor', ['mist', 'splash']), set('Sableye', '', 'KeenEye', ['snatch', 'growl']), [mv(1, 1), mv(2, 2)], { after: pp });
  // LOCK-ON / MIND READER
  run('lock-on then zap cannon (no acc draw), lapse', set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'mindreader', 'splash']), BENCH, [mv(1, 1), mv(2, 1), mv(2, 1)], { after: pp });
  run('lock-on recast fails / mind reader', set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'mindreader', 'splash']), BENCH, [mv(1, 1), mv(1, 1), mv(3, 1), mv(4, 1), mv(3, 1)], { after: pp });
  run('lock-on into protect / substitute', set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'splash']), set('Blissey', '', 'NaturalCure', ['substitute', 'splash'], { g: 'F' }), [mv(3, 1), mv(1, 2), mv(2, 2)], { after: pp });
  run('a held lock + protect: protect first', set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'splash']), set('Blissey', '', 'NaturalCure', ['splash', 'protect'], { g: 'F' }), [mv(1, 1), mv(1, 2)], { after: pp });
  run('lock-on vs dig (slower target)', set('Porygon2', '', 'Trace', ['lockon', 'icebeam', 'zapcannon']), set('Golem', '', 'Sturdy', ['dig', 'splash']), [mv(1, 1), mv(2, 1)], { after: pp });
  run('dig: no lock control', set('Porygon2', '', 'Trace', ['splash', 'icebeam', 'zapcannon']), set('Golem', '', 'Sturdy', ['dig', 'splash']), [mv(1, 1), mv(2, 1)], { after: pp });
  run('lock-on into an airborne target misses', set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'icebeam'], { evs: '0,0,0,0,0,0' }), set('Swellow', '', 'Guts', ['fly', 'dig'], { g: 'F' }), [mv(1, 1), mv(1, 1)], { after: pp });
  run('lock-on, target switches, recast fails', set('Porygon2', '', 'Trace', ['lockon', 'zapcannon']), team(BENCH, set('Snorlax', '', 'ThickFat', ['splash'])), [mv(1, 1), ['move 1', 'switch 2']], { after: pp });
  run('lock-on BP tie', team(set('Regice', '', 'ClearBody', ['lockon', 'batonpass']), set('Regice', '', 'ClearBody', ['splash'])), set('Regice', '', 'ClearBody', ['lockon', 'splash']), [mv(1, 1), mv(2, 2), ['switch 2', null], mv(1, 2)], { after: pp });
  run('lock-on BP tie control', team(set('Regice', '', 'ClearBody', ['splash', 'batonpass']), set('Regice', '', 'ClearBody', ['splash'])), set('Regice', '', 'ClearBody', ['lockon', 'splash']), [mv(1, 1), mv(2, 2), ['switch 2', null], mv(1, 2)], { after: pp });
  run('lock-on mirror residual tie', set('Regice', '', 'ClearBody', ['lockon', 'splash']), set('Regice', '', 'ClearBody', ['lockon', 'splash']), [mv(1, 1), mv(2, 2), mv(2, 2)], { after: pp });
  run('lock-on then hypnosis (status accuracy)', set('Porygon2', '', 'Trace', ['lockon', 'hypnosis']), BENCH, [mv(1, 1), mv(2, 1)], { after: pp });
  run('lock-on then the future sight strike', set('Porygon2', '', 'Trace', ['futuresight', 'lockon', 'splash']), BENCH, [mv(1, 1), mv(3, 1), mv(2, 1)], { after: pp });
  run('future sight strike control', set('Porygon2', '', 'Trace', ['futuresight', 'lockon', 'splash']), BENCH, [mv(1, 1), mv(3, 1), mv(3, 1)]);
  run('lock-on then perish song vs airborne', set('Porygon2', '', 'Trace', ['lockon', 'perishsong', 'splash'], { evs: '0,0,0,0,0,252' }), set('Swellow', '', 'Guts', ['fly', 'splash'], { evs: '0,0,0,0,0,0', lv: 50 }), [mv(1, 1), mv(2, 1)], { after: pp });
  run('perish song vs airborne control', set('Porygon2', '', 'Trace', ['splash', 'perishsong', 'splash'], { evs: '0,0,0,0,0,252' }), set('Swellow', '', 'Guts', ['fly', 'splash'], { evs: '0,0,0,0,0,0', lv: 50 }), [mv(1, 1), mv(2, 1)], { after: pp });
  run('lock-on vs pressure', set('Porygon2', '', 'Trace', ['lockon', 'splash']), set('Zapdos', '', 'Pressure', ['splash']), [mv(1, 1)], { after: pp });
}
