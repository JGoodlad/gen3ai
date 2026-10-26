// probe_b5_moves.js — gen3 full parity B5 (`gen3_b5_moves_v1`): Spite, Grudge, Snore, Nightmare,
// Ingrain, and Rest under a live Uproar, against the real sim (gen3customgame unless noted).
// Re-runnable; each board prints its lines with every PRNG draw attributed to the line that was
// last emitted when it drew (`  ~ N draw(s)`):
//   node src/rust_sim/harness/probe_b5_moves.js "$PWD/deps/pokemon-showdown" [filter]
//
// SETTLED (2026-10-26, against the RESOLVED `Dex.mod('gen3')` — the mod-chain law; citations are
// the file that supplies the resolved handler):
//   * SPITE (`data/mods/gen3/moves.ts` spite, accuracy 100, flags protect/mirror/bypasssub):
//     DRAWS: the accuracy roll, then `onHit`'s `this.random(2, 6)` FIRST, then
//     `if (target.lastMove && target.deductPP(target.lastMove.id, roll))`. So a Spite that fails
//     (no lastMove, a Struggle, a lastMove at 0 PP, a self-overwritten Mimic id in no slot) still
//     drew 2. Success: `|-activate|<target>|move: Spite|<lastMove ID>|<roll>` — the raw id and the
//     ROLL even when fewer PP were left (`sim/pokemon.ts` deductPP clamps at 0). Fail: the
//     `[still]` retro-edit + `|-fail|<user>`. Protect blocks after the accuracy roll (no roll);
//     a Substitute does not. Pressure plays no part in the deduction (only on Spite's own PP:
//     the user pays 2 into a Pressure target). The lastMove is what `runMove` ran
//     (`moveUsed`): Sleep Talk, not the move it called; the Encored / Choice-locked move; the
//     Pursuit an interrupt ran; reset on switch-out (`clearVolatile`). Spite emptying an Encored
//     move ends the Encore at its residual; emptying a Choice-locked move forces Struggle. A
//     slot Spite empties AFTER the target chose it (Spite moved first) fails at the target's
//     `runMove` `deductPP`: `|cant|<target>|nopp|<Move name>`, draw-free, after BeforeMove (a
//     sleeper's `|cant|…|slp` first), with no MoveAborted / AfterMove and no lastMove update.
//   * GRUDGE (`data/moves.ts` grudge, untouched by the gen-3/4 mods; accuracy true, self): ZERO
//     draws, `|-singlemove|<user>|Grudge`. Removed SILENTLY at the holder's next `runMove`
//     (`onBeforeMovePriority: 100` — before recharge 11 / sleep & freeze 10 / truant / flinch /
//     paralysis), so a holder that is then asleep / paralysed has still lost it; a `useMove`
//     (a Sleep-Talk-called move, a Pursuit interrupt) runs no BeforeMove. `onFaint(target,
//     source, effect)`: `if (!source || source.fainted || !effect) return;` then, for a Move
//     effect without `futuremove`, every killer moveslot whose id is `source.lastMove.id` goes to
//     0 PP with `|-activate|<killer>|move: Grudge|<Move name>` — emitted inside
//     `faintMessages`' `runEvent('Faint')`, right after the holder's `|faint|`. `fainted` is set
//     only when the drain reaches a corpse, so a killer KO'd by its own recoil (queued after the
//     holder) still loses the PP; an Explosion user (queued first) does not; Future Sight never
//     triggers it. A Sleep-Talk-killer loses SLEEP TALK's PP.
//   * SNORE (`data/mods/gen4/moves.ts` snore, inherited): `sleepUsable`, `onTry(source) { return
//     source.status === 'slp' }` at gen-3 `tryMoveHit`'s `singleEvent('Try')`
//     (`data/mods/gen3/scripts.ts`): after the announce, before accuracy / Protect / immunity, and a
//     false there adds NO line. Awake (including the wake turn) = the bare announce, ZERO draws,
//     PP spent. Asleep: the gen-3 slp `onBeforeMove` (`data/mods/gen3/conditions.ts`) prints
//     `|cant|<mon>|slp`, `skippedTime++`, and lets it PROCEED as a 40-BP Normal (physical) sound
//     move: accuracy, crit, damage roll, 30 % flinch roll. Soundproof `-immune|…|[from] ability:
//     Soundproof` and a Ghost `-immune` after the accuracy roll; a Substitute takes the hit.
//     Sleep Talk can call it (`|move|<u>|Snore|<t>|[from] Sleep Talk`).
//   * NIGHTMARE (`data/mods/gen3/moves.ts` nightmare: accuracy true; the condition from
//     `data/mods/gen4/moves.ts`, `noCopy`): ZERO draws on the cast. TryHit order: Protect
//     (`-activate|<t>|Protect`) > the Substitute (no bypasssub) > `addVolatile` (already on) >
//     `onStart` (`status !== 'slp'` → false); every failure is the `[still]` retro-edit +
//     `|-fail|<user>`. Success `|-start|<t>|Nightmare`. Residual order 10 subOrder 7 (after
//     Leftovers 4 / Leech Seed 5 / status 6, before Curse 8): `this.damage(baseMaxhp / 4)` →
//     `|-damage|<t>|<hp> slp|[from] Nightmare`; a Focus Band holder draws its roll. EVERY sleep
//     cure — the natural wake, Uproar, Shed Skin, Chesto / Lum, an Insomnia gained by Skill Swap,
//     Heal Bell / Aromatherapy through Sleep Talk, Natural Cure on switch-out — runs
//     `cureStatus()` / `clearStatus()` (`sim/pokemon.ts:1678` / `:1757`), which adds
//     `|-end|<t>|Nightmare|[silent]` right after its cure line. A Shed Skin wake (subOrder 3) skips
//     the same residual's chip (`fieldEvent` drops a handler whose volatile is gone). (Uproar's
//     wake is printed at its TryHit, BEFORE the damage line — the port prints it after the damage,
//     a pre-existing `gen3_uproar_v1` order defect this batch reports and does not fix.)
//   * INGRAIN (base `data/moves.ts` ingrain; accuracy true, self, flags snatch): ZERO draws,
//     `|-start|<u>|move: Ingrain`; a second cast `[still]` + `|-fail|<u>`. Residual order 10
//     subOrder 1 (the first order-10 handler): `this.heal(baseMaxhp / 16)` →
//     `|-heal|<u>|<hp>|[from] Ingrain`, nothing at full HP. `onTrapPokemon` → a bare `tryTrap()`:
//     the request carries `"trapped":true` from the first request after the cast (a FIRM trap),
//     and the endTurn TrapPokemon event gathers it as a Condition handler (ties a Wrap /
//     Mean Look volatile on the same mon → one shuffle draw a turn). `onDragOut` →
//     `|-activate|<u>|move: Ingrain`, no `forceSwitchFlag`, so NO switch-in `sample` draw; Suction
//     Cups (`onDragOutPriority: 1`) runs first and wins. No `noCopy`: Baton Pass passes it (the
//     entrant heals off its own HP and is trapped). Snatch steals it.
//   * REST UNDER UPROAR: Rest's `onHit` → `setStatus('slp')` → `runEvent('SetStatus')` →
//     `uproar.onAnySetStatus` (`data/mods/gen3/moves.ts`): `|-fail|<mon>|slp|[from] Uproar`
//     (`|[msg]` for the uproarer itself), no `-status`, no heal, no `random(2,6)`. In gen3ou the
//     clause shuffle inside the same event still draws (uproar sorts alone).
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
const req = (b) => { for (const s of b.sides) console.log(`   [${s.id} request trapped=${!!(s.activeRequest && s.activeRequest.active && s.activeRequest.active[0].trapped)}]`); };
const mv = (a, c) => [a && `move ${a}`, c && `move ${c}`];

// SPITE
run('spite basic', set('Gengar', '', 'Levitate', ['spite', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(2, 1), mv(1, 1)], { after: pp });
run('spite no lastMove (draws the roll anyway)', set('Jolteon', '', 'VoltAbsorb', ['spite', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 1)]);
run('spite into Pressure / Substitute / Protect', set('Snorlax', '', 'ThickFat', ['spite', 'splash']), set('Jolteon', '', 'Pressure', ['substitute', 'protect', 'thunderbolt']), [mv(2, 3), mv(1, 1), mv(1, 2)], { after: pp });
run('spite drains to 0, prints the roll, then a Struggle lastMove', set('Gengar', '', 'Levitate', ['spite', 'splash']), set('Snorlax', 'Leftovers', 'ThickFat', ['zapcannon'], { evs: '252,0,252,0,4,0' }), [mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1)], { after: pp });
run('spite after Sleep Talk', set('Gengar', '', 'Levitate', ['spite', 'spore']), set('Snorlax', '', 'ThickFat', ['sleeptalk', 'bodyslam']), [mv(2, 1), mv(1, 1)], { after: pp });
run('spite empties the chosen move: cant nopp', set('Gengar', '', 'Levitate', ['spite', 'splash']), set('Snorlax', 'Leftovers', 'ThickFat', ['zapcannon', 'splash'], { evs: '252,0,252,0,4,0' }), [mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1)], { seed: [19975, 46914, 35677, 52368], after: pp });
run('spite on a Choice-locked move', set('Gengar', '', 'Levitate', ['spite', 'splash']), set('Snorlax', 'Choice Band', 'ThickFat', ['zapcannon', 'bodyslam'], { evs: '252,0,252,0,4,0' }), [mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1)], { after: pp });
// GRUDGE
run('grudge KO', team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), set('Houndoom', '', 'EarlyBird', ['flamethrower', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
run('grudge lost to a sleep cant', team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), set('Snorlax', '', 'ThickFat', ['spore', 'shadowball']), [mv(1, 1), mv(2, 2)]);
run('grudge vs a recoil-KOd killer', team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), team(set('Shedinja', '', 'WonderGuard', ['doubleedge', 'splash']), BENCH), [mv(1, 2), mv(2, 1)], { after: pp });
run('grudge vs a Sleep Talk killer', team(set('Shedinja', '', 'WonderGuard', ['slash', 'grudge']), BENCH), set('Snorlax', '', 'ThickFat', ['rest', 'sleeptalk', 'shadowball']), [mv(1, 1), mv(2, 2)], { after: pp });
run('grudge vs Explosion', team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), team(set('Electrode', '', 'Soundproof', ['explosion', 'splash']), BENCH), [mv(1, 2), mv(2, 1)]);
run('grudge vs Pursuit', team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), set('Tyranitar', '', 'SandStream', ['pursuit', 'splash']), [mv(1, 2), ['switch 2', 'move 1']], { after: pp });
// SNORE
run('snore awake', set('Snorlax', '', 'ThickFat', ['snore', 'rest']), set('Blissey', '', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' }), [mv(1, 2)]);
run('snore asleep, then the wake turn', set('Snorlax', '', 'ThickFat', ['snore', 'rest']), set('Blissey', '', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' }), [mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1)]);
run('snore via Sleep Talk', set('Snorlax', '', 'ThickFat', ['snore', 'rest', 'sleeptalk']), set('Blissey', '', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' }), [mv(2, 1), mv(3, 1), mv(3, 1)]);
run('snore into Soundproof', set('Snorlax', '', 'ThickFat', ['snore', 'rest']), set('Electrode', '', 'Soundproof', ['seismictoss', 'splash']), [mv(2, 1), mv(1, 1)]);
// NIGHTMARE
run('nightmare: awake fail, cast, chip vs Leftovers / Leech Seed, wake', set('Venusaur', '', 'Overgrow', ['nightmare', 'spore', 'leechseed']), set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), [mv(1, 1), mv(2, 1), mv(1, 1), mv(3, 1), mv(1, 1), mv(1, 1)], { after: pp });
run('nightmare into a Substitute', set('Gengar', '', 'Levitate', ['nightmare', 'splash']), set('Snorlax', '', 'ThickFat', ['substitute', 'rest']), [mv(2, 1), mv(2, 2), mv(1, 2)]);
run('nightmare + Shed Skin', set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), set('Dragonair', '', 'ShedSkin', ['splash']), [mv(1, 1), mv(2, 1), mv(3, 1), mv(3, 1), mv(3, 1)], { seed: [3, 2, 3, 4] });
run('nightmare + Natural Cure switch', set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), team(set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' }), set('Snorlax', '', 'ThickFat', ['splash'])), [mv(1, 1), mv(2, 1), ['move 3', 'switch 2']]);
run('nightmare + Uproar wake', set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'uproar']), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1), mv(2, 1), mv(3, 1)]);
run('nightmare + Focus Band KO roll', set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), team(set('Pichu', 'Focus Band', 'Static', ['splash'], { lv: 5 }), BENCH), [mv(1, 1), mv(2, 1), mv(3, 1), mv(3, 1)]);
// INGRAIN
run('ingrain heal, recast, trap', set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 1), mv(1, 1)], { after: req });
run('ingrain vs Roar / Whirlwind (no sample draw)', team(set('Cradily', '', 'Chlorophyll', ['ingrain', 'splash']), BENCH), set('Snorlax', '', 'ThickFat', ['roar', 'whirlwind']), [mv(1, 2), mv(2, 1)]);
run('ingrain + Suction Cups', team(set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), BENCH), set('Snorlax', '', 'ThickFat', ['roar', 'whirlwind']), [mv(1, 2), mv(2, 1)]);
run('ingrain Baton Pass', team(set('Cradily', '', 'Chlorophyll', ['ingrain', 'batonpass']), BENCH), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 1), mv(2, 1), ['switch 2', null], mv(1, 1)], { after: req });
run('ingrain + Wrap (TrapPokemon tie)', set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), set('Dragonite', '', 'InnerFocus', ['wrap', 'splash']), [mv(1, 1), mv(2, 2), mv(2, 2)]);
run('ingrain snatched', set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), set('Sableye', '', 'KeenEye', ['snatch', 'splash']), [mv(1, 1)]);
// REST UNDER UPROAR
run('rest under uproar (customgame)', set('Snorlax', '', 'ThickFat', ['rest', 'splash']), set('Exploud', '', 'Soundproof', ['uproar']), [mv(2, 1), mv(1, 1)]);
run('rest under uproar (gen3ou)', set('Snorlax', '', 'ThickFat', ['rest', 'bodyslam']), set('Exploud', '', 'Soundproof', ['uproar']), [mv(2, 1), mv(1, 1)], { format: 'gen3ou' });
