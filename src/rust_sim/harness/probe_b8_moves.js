// probe_b8_moves.js — gen3 full parity B8 (`gen3_b8_moves_v1`): Bide, Magnitude, Psywave, Present,
// Triple Kick and the four OHKO moves (Fissure, Horn Drill, Guillotine, Sheer Cold), against the real
// sim (gen3customgame unless noted). Re-runnable; each board prints its lines with every PRNG draw
// attributed to the line that was last emitted when it drew (`  ~ N draw(s)`):
//   node src/rust_sim/harness/probe_b8_moves.js "$PWD/deps/pokemon-showdown" [filter]
//
// SETTLED (2026-10-26, against the RESOLVED `Dex.mod('gen3')` — the mod-chain law; the citations name
// the file that supplies each resolved handler; `scripts.ts` is `data/mods/gen3/scripts.ts`):
//   * BIDE (`data/moves.ts` bide; `data/mods/gen3/moves.ts:57` sets accuracy 100, priority 0 and
//     replaces the condition's `onBeforeMove`). NO duration draw: `duration: 3`, ticked by the
//     residual (a NO_ORDER duration handler, so it joins the end-of-turn tie group). CAST: `|move|<u>|
//     Bide|<u>` (the user is the target), ONE accuracy roll — accuracy 100, not `true`, so the user's
//     own accuracy / evasion stages and its own Bright Powder / Sand Veil / Hustle fold in and a +6
//     evasion user can miss itself (`|move|…|[miss]` + `|-miss|<u>|<u>`) — then `|-start|<u>|move:
//     Bide`; a King's Rock flinch secondary rolls against the USER (silent; a flinch joins the
//     residual tie group: one more draw). PP is paid once (Pressure does not apply: the target is
//     self). Locked (`onLockMove`): the request is `{"moves":[{"move":"Bide","id":"bide"}],
//     "trapped":true}`. WAIT turn (duration 2): `|-activate|<u>|move: Bide`, then
//     `beforeMoveCallback` ends runMove (no PP, no MoveAborted, no AfterMove). RELEASE (duration 1):
//     `|-end|<u>|move: Bide`; nothing stored → `|-fail|<u>`; else `tryMoveHit` with {accuracy 100,
//     damage 2×stored, Physical, contact+protect, Normal} and NO `|move|` line: one accuracy roll; a
//     Ghost is `-immune` (the type IS checked), Protect `-activate`s, Wonder Guard blocks, a sub
//     takes it, a contact ability procs (Rough Skin), no King's Rock, no Shell Bell; a mid-Fly / Dig
//     target misses with no draw and `[miss]` lands on the LAST `|move|` line (the foe's own Fly
//     charge that turn). Then `removeVolatile` → `|-end|<u>|move: Bide|[silent]`. Stored: every
//     Move-effect hit after the cast (Endure / Focus Band first, `onDamagePriority -101`), multi-hit
//     strikes, a Future Sight strike; not a hit its own sub took, not residual damage. The target is
//     the current foe active even when the damager switched out. ANY BeforeMove cant on a locked
//     turn (sleep, full paralysis, a confusion self-hit, Disable, Imprison) ends it with the
//     `[silent]` end after the cant line. A Roar drags it out silently. Sleep Talk cannot call it.
//   * MAGNITUDE (`data/moves.ts` magnitude): `onModifyMove` draws ONE `random(100)` BEFORE the `|move|`
//     announce (`scripts.ts:145`) — so a miss / Protect / immunity / Fly target still drew it — table
//     <5 → 4/10, <15 → 5/30, <35 → 6/50, <65 → 7/70, <85 → 8/90, <95 → 9/110, else 10/150; then
//     `onUseMoveMessage` (`scripts.ts:195`) prints `|-activate|<u>|move: Magnitude|<n>` right after
//     the announce, before any `-miss` / `-immune` / Protect line (a miss's `[miss]` still lands on
//     the `|move|` line). Then the ordinary damaging hit: accuracy, crit, damage roll; a Dig target is
//     hit and doubled, a Fly target missed with no draw; King's Rock listed.
//   * PSYWAVE (`data/moves.ts` damageCallback, accuracy 80 from `data/mods/gen5/moves.ts`): accuracy,
//     then ONE `random(50, 151)` inside `getDamage`: `r * level / 100`; to the mon `spreadDamage`
//     floors it and lifts it to 1; no crit, no damage roll, no effectiveness line; a Dark target
//     `-immune` after the accuracy roll; Wonder Guard blocks; King's Rock listed. Into a Substitute
//     the RAW float comes off the sub's HP (`data/mods/gen4/moves.ts` substitute), so a user below
//     level 100 leaves the sub at a fractional HP (board "psywave sub frac") — the port FAILS LOUD
//     there.
//   * PRESENT (`data/moves.ts` present): `onModifyMove` draws ONE `random(10)` before the announce:
//     <2 HEAL (`move.heal = [1, 4]`, base power left 0), <6 → 40, <9 → 80, else 120. A damage branch
//     is the ordinary hit (accuracy 90, crit, damage roll). The heal branch: accuracy roll, then
//     `getDamage` returns undefined at `if (!basePower)` BEFORE the crit roll; a Substitute fails it
//     (`|move|<u>|Present||[still]` + `|-fail|<u>`: gen 3's sub ignores `infiltrates`), a full-HP
//     target fails it (`|-fail|<t>|heal` + `[still]`), else `|-heal|<t>|<hp>` (floor(maxhp/4), no
//     `[from]`). A Ghost is `-immune` and Wonder Guard blocks, either branch.
//   * TRIPLE KICK (`data/moves.ts` triplekick; `multiaccuracy`, `multihit: 3`, `basePowerCallback`
//     `10 * move.hit`): the first strike is the move's accuracy roll; each later strike RE-ROLLS
//     accuracy first (`scripts.ts:414-439`: stages, ModifyAccuracy — Bright Powder — and Accuracy)
//     and a failed re-roll ends the move silently; per strike crit + damage roll at BP 10 / 20 / 30;
//     `-hitcount|<t>|N` counts the landed strikes; King's Rock and contact procs per strike.
//   * OHKO (Fissure / Horn Drill / Guillotine / Sheer Cold, `ohko: true`; `scripts.ts:346-361`,
//     `-ohko` 466; `getDamage` `if (move.ohko) return gen === 3 ? target.hp : …`,
//     `sim/battle-actions.ts:1610`): after the Invulnerability event (no OHKO move bypasses Fly / Dig
//     / Dive / Bounce), a user BELOW the target's level gets `|-immune|<t>|[ohko]` with NO draw, even
//     into a Ghost / Flying / Levitate target and before Protect; else ONE `randomChance(30 + level
//     gap, 100)` with every stage and ModifyAccuracy holder ignored (Bright Powder, +evasion). Then
//     TryHit: Protect, Wonder Guard, Sturdy (`|-immune|<t>|[from] ability: Sturdy`, base
//     `data/abilities.ts`), then the natural-immunity `-immune`. The hit deals the target's CURRENT HP:
//     a sub takes it (`-end` or `-activate …|[damage]` when the mon has less HP than its sub) and
//     prints `-ohko` right after; Endure / Focus Band save it at 1 (no `-ohko`); else `-damage …|0
//     fnt`, then `-ohko`, then `|faint|`. Lock-On / Mind Reader are NOT modeled here (another batch);
//     they bypass both the Invulnerability miss and this accuracy.
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
  const at = []; // log length at each draw
  rng.next = (...a) => { at.push(b.log.length); return next(...a); };
  const start = b.log.length;
  for (const [x, y] of script) { if (x) b.choose('p1', x); if (y) b.choose('p2', y); }
  console.log(`== ${label}`);
  for (let i = start; i < b.log.length; i++) {
    const l = b.log[i];
    if (l.startsWith('|split|')) continue;
    let n = at.filter((k) => k === i + 1).length;
    if (i > 0 && b.log[i - 2] && b.log[i - 2].startsWith('|split|')) continue; // the public half
    if (b.log[i - 1] && b.log[i - 1].startsWith('|split|')) n += at.filter((k) => k === i + 2).length;
    const shown = !(l === '' || l === '|' || l.startsWith('|t:|') || l.startsWith('|debug|') || l === '|upkeep');
    if (shown) console.log(l + (n ? `   ~ ${n} draw(s)` : ''));
    else if (n) console.log(`   (${l || 'blank'})   ~ ${n} draw(s)`);
  }
  if (opts.after) opts.after(b);
}
const pp = (b) => { for (const s of b.sides) for (const p of s.pokemon) console.log(`   [${p.name} ${p.hp}/${p.maxhp} ${p.status || '-'} vol=${Object.keys(p.volatiles)} pp=${p.moveSlots.map((m) => `${m.id}:${m.pp}`)}]`); };
const req = (b) => { for (const s of b.sides) console.log(`   [${s.id} request ${JSON.stringify(s.activeRequest && s.activeRequest.active)}]`); };
const subhp = (b) => { for (const s of b.sides) console.log(`   [${s.id} sub ${JSON.stringify(s.active[0].volatiles.substitute)}]`); };
const mv = (a, c) => [a && `move ${a}`, c && `move ${c}`];
const R = (moves, o = {}) => set('Raichu', o.item || '', 'Static', moves, o);
const seeds = (n) => Array.from({ length: n }, (_, k) => [k + 1, 2 * k + 3, 3 * k + 5, 4 * k + 7]);

// BIDE
run('bide basic (slow foe hits after the cast)', R(['bide', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 1), mv(1, 1), mv(1, 2)], { after: pp });
run('bide request while locked', R(['bide', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 1)], { after: req });
run('bide nothing stored', R(['bide', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 2), mv(1, 2), mv(1, 2)]);
run('bide release into a ghost', R(['bide', 'splash']), set('Gengar', '', 'Levitate', ['thunderbolt', 'splash']), [mv(1, 1), mv(1, 2), mv(1, 2)]);
run('bide damager switched out', R(['bide', 'splash']), team(set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), BENCH), [mv(1, 1), ['move 1', 'switch 2'], mv(1, 1)]);
run('bide multihit stored', set('Shuckle', '', 'Sturdy', ['bide', 'splash']), set('Cloyster', '', 'ShellArmor', ['spikecannon', 'splash']), [mv(1, 1), mv(1, 1), mv(1, 2)]);
run('bide future sight', set('Snorlax', '', 'ThickFat', ['bide', 'splash']), set('Alakazam', '', 'Synchronize', ['futuresight', 'splash']), [mv(2, 1), mv(2, 2), mv(1, 2), mv(1, 2), mv(1, 2)]);
run('bide own sub', R(['bide', 'substitute']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(2, 2), mv(1, 1), mv(1, 1), mv(1, 1)]);
run('bide protect at release', R(['bide', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'protect']), [mv(1, 1), mv(1, 1), mv(1, 2)]);
run('bide wonder guard at release', R(['bide', 'splash']), team(set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), set('Shedinja', '', 'WonderGuard', ['splash'])), [mv(1, 1), ['move 1', 'switch 2'], mv(1, 1)]);
run('bide rough skin at release', R(['bide', 'splash']), set('Sharpedo', '', 'RoughSkin', ['crunch', 'splash']), [mv(1, 1), mv(1, 1), mv(1, 2)]);
run('bide release vs a same-turn Fly', set('Shuckle', '', 'Sturdy', ['bide', 'splash']), set('Pidgeot', '', 'KeenEye', ['fly', 'quickattack']), [mv(1, 2), mv(1, 2), mv(1, 1)]);
run('bide sleep cancel', set('Snorlax', '', 'ThickFat', ['bide', 'splash']), set('Breloom', '', 'EffectSpore', ['spore', 'tackle']), [mv(1, 2), mv(1, 1), mv(2, 2)]);
run('bide confusion self-hit cancel', R(['bide', 'splash']), set('Snorlax', '', 'ThickFat', ['confuseray', 'bodyslam']), [mv(1, 1), mv(1, 2), mv(1, 2)], { seed: [5, 6, 7, 8] });
run('bide disabled', set('Snorlax', '', 'ThickFat', ['bide', 'splash']), set('Gengar', '', 'Levitate', ['disable', 'shadowball']), [mv(1, 2), mv(1, 1), mv(1, 2)]);
run('bide imprisoned', R(['bide', 'splash']), set('Smeargle', '', 'OwnTempo', ['tackle', 'imprison', 'bide']), [mv(1, 1), mv(1, 2), mv(1, 1)]);
run('bide roared out', team(R(['bide', 'splash']), set('Snorlax', '', 'ThickFat', ['splash'])), set('Snorlax', '', 'ThickFat', ['bodyslam', 'roar']), [mv(1, 1), mv(1, 2), mv(1, 1)]);
run('bide pressure (pp 1)', R(['bide', 'splash']), set('Zapdos', '', 'Pressure', ['drillpeck', 'splash']), [mv(1, 1), mv(1, 1), mv(1, 2)], { after: pp });
run('bide shell bell (no heal on release)', R(['bide', 'splash'], { item: 'Shell Bell' }), set('Snorlax', '', 'ThickFat', ['seismictoss', 'splash'], { lv: 30 }), [mv(1, 1), mv(1, 1), mv(1, 2)]);
run('bide +6 evasion misses itself', set('Snorlax', '', 'ThickFat', ['bide', 'doubleteam']), set('Jolteon', '', 'VoltAbsorb', ['tackle', 'splash']), [mv(2, 2), mv(2, 2), mv(2, 2), mv(2, 2), mv(2, 2), mv(2, 2), mv(1, 1), mv(1, 1)]);
for (const s of seeds(4)) run(`bide kings rock ${s}`, R(['bide', 'splash'], { item: 'Kings Rock' }), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 2)], { seed: s });
run('bide mirror', set('Snorlax', '', 'ThickFat', ['bide', 'tackle']), set('Snorlax', '', 'ThickFat', ['bide', 'tackle']), [mv(1, 2), mv(1, 2), mv(1, 2)]);
// MAGNITUDE
for (const s of seeds(3)) run(`magnitude ${s}`, set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1), mv(1, 1)], { seed: s });
run('magnitude vs flying', set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Skarmory', '', 'KeenEye', ['splash']), [mv(1, 1)]);
run('magnitude vs dig', set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Dugtrio', '', 'ArenaTrap', ['dig', 'splash']), [mv(1, 1)]);
run('magnitude vs fly', set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Pidgeot', '', 'KeenEye', ['fly', 'splash']), [mv(1, 1)]);
run('magnitude vs protect', set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Snorlax', '', 'ThickFat', ['protect']), [mv(1, 1)]);
run('magnitude vs sub', set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['substitute']), [mv(1, 1)]);
run('magnitude miss (bright powder, +6 evasion)', set('Golem', '', 'Sturdy', ['magnitude', 'splash']), set('Snorlax', 'Bright Powder', 'ThickFat', ['doubleteam']), [mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(1, 1)]);
// PSYWAVE
for (const lv of ['', 50, 37, 5, 1]) run(`psywave lv${lv || 100}`, set('Gengar', '', 'Levitate', ['psywave', 'splash'], { lv }), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1), mv(1, 1)]);
run('psywave vs dark', set('Gengar', '', 'Levitate', ['psywave', 'splash']), set('Umbreon', '', 'Synchronize', ['splash']), [mv(1, 1)]);
run('psywave vs wonder guard', set('Gengar', '', 'Levitate', ['psywave', 'splash']), set('Shedinja', '', 'WonderGuard', ['splash']), [mv(1, 1)]);
run('psywave vs sub (level 100)', set('Gengar', '', 'Levitate', ['psywave', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), [mv(2, 1), mv(1, 2), mv(1, 2)]);
run('psywave sub frac', set('Gengar', '', 'Levitate', ['psywave', 'splash'], { lv: 37 }), set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), [mv(2, 1), mv(1, 2)], { after: subhp });
// PRESENT
for (const s of seeds(5)) run(`present ${s}`, set('Delibird', '', 'Hustle', ['present', 'splash']), set('Snorlax', '', 'ThickFat', ['splash', 'bellydrum']), [mv(1, 2), mv(1, 1), mv(1, 1)], { seed: s });
run('present heal at full HP', set('Delibird', '', 'Hustle', ['present', 'splash']), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1)]);
run('present heal into a sub', set('Delibird', '', 'Hustle', ['present', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), [mv(2, 1), mv(1, 2)], { seed: [3, 7, 11, 15] });
run('present vs ghost', set('Delibird', '', 'Hustle', ['present', 'splash']), set('Gengar', '', 'Levitate', ['splash']), [mv(1, 1)]);
run('present vs wonder guard', set('Delibird', '', 'Hustle', ['present', 'splash']), set('Shedinja', '', 'WonderGuard', ['splash']), [mv(1, 1)]);
// TRIPLE KICK
for (const s of seeds(4)) run(`triple kick ${s}`, set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), set('Snorlax', '', 'ThickFat', ['doubleteam']), [mv(1, 1), mv(1, 1), mv(1, 1)], { seed: s });
run('triple kick vs ghost', set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), set('Gengar', '', 'Levitate', ['splash']), [mv(1, 1)]);
run('triple kick vs sub', set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), [mv(2, 1), mv(1, 2)]);
run('triple kick KO + static', set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), team(set('Pichu', '', 'Static', ['splash'], { lv: 5 }), BENCH), [mv(1, 1)]);
run('triple kick kings rock', set('Hitmontop', 'Kings Rock', 'Intimidate', ['triplekick', 'splash']), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1)]);
run('triple kick bright powder re-roll', set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), set('Snorlax', 'Bright Powder', 'ThickFat', ['splash']), [mv(1, 1), mv(1, 1), mv(1, 1)], { seed: [3, 3, 3, 3] });
// OHKO
run('fissure equal level', set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash']), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1)]);
run('fissure lower level', set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash'], { lv: 50 }), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1)]);
run('horn drill into a ghost', set('Rhydon', '', 'LightningRod', ['horndrill', 'splash']), set('Gengar', '', 'Levitate', ['splash']), [mv(1, 1)]);
run('horn drill into a ghost, lower level', set('Rhydon', '', 'LightningRod', ['horndrill', 'splash'], { lv: 50 }), set('Gengar', '', 'Levitate', ['splash']), [mv(1, 1)]);
run('fissure into flying', set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash']), set('Skarmory', '', 'KeenEye', ['splash']), [mv(1, 1)]);
run('sheer cold into sturdy', set('Lapras', '', 'ShellArmor', ['sheercold', 'splash']), set('Golem', '', 'Sturdy', ['splash'], { lv: 1 }), [mv(1, 1)]);
run('guillotine into a sub', set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash'], { lv: 1 }), [mv(2, 1), mv(1, 2), mv(1, 2)]);
run('guillotine into a sub that outlives its mon', set('Kingler', '', 'HyperCutter', ['guillotine', 'toxic', 'splash']), set('Snorlax', '', 'ThickFat', ['substitute', 'bellydrum', 'splash'], { lv: 1 }), [mv(2, 1), mv(3, 2), mv(1, 3)], { after: subhp });
run('guillotine endure', set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), set('Snorlax', '', 'ThickFat', ['endure', 'splash'], { lv: 1 }), [mv(1, 1)]);
run('guillotine protect', set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), set('Snorlax', '', 'ThickFat', ['protect', 'splash'], { lv: 1 }), [mv(1, 1)]);
run('guillotine wonder guard', set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), set('Shedinja', '', 'WonderGuard', ['splash']), [mv(1, 1)]);
run('fissure vs a faster dig', set('Whiscash', '', 'Oblivious', ['fissure', 'splash']), set('Dugtrio', '', 'ArenaTrap', ['dig', 'splash']), [mv(1, 1)]);
run('fissure ignores bright powder + evasion', set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash']), set('Snorlax', 'Bright Powder', 'ThickFat', ['doubleteam', 'splash'], { lv: 1 }), [mv(2, 1), mv(2, 1), mv(1, 1)]);
