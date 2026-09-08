// PROBE: the gen-3 ATTRACT **MOVE** — draw model, emission forms, and every fail gate.
//
// THE SITUATION. The attract VOLATILE is already modelled bit-for-bit (`gen3_ability_batch4_v1`):
// `state.rs::MonState::attract`, the `onBeforeMove` priority-2 `-activate` + `randomChance(1,2)`
// immobilize, the source-left / holder-switch-out clears, and `status.rs::try_add_attract`'s
// gate ladder (already-attracted, fainted, the M<->F gender gate, Oblivious). But in gen 3 the
// port can only REACH it through a Cute Charm contact proc — the Attract MOVE is unmodelled, and
// it is 338 legal learners, the largest single-move gap left after ROUNDS 57-58.
//
// So the missing half is the MOVE ENTRY: the accuracy roll, the TryHit gates, and — the part no
// reading of `try_add_attract` would tell you — **which line each fail branch emits**. That
// function currently returns SILENTLY on every fail (correct for a Cute Charm proc, which has no
// move to report about) and hardcodes the success emission to the Cute Charm form
// `|-start|<t>|Attract|[from] ability: Cute Charm|[of] <src>`. A move cast needs a different
// success line and needs its failures to be VISIBLE.
//
// MEASURE, per branch: the DRAW COUNT, and the exact bytes.
//   1. plain hit, M into F            5. OBLIVIOUS target
//   2. same gender (F into F)         6. SUBSTITUTE
//   3. genderless target              7. SAFEGUARD (does the ward cover attraction?)
//   4. already attracted (2nd cast)   8. PROTECT
//
// RUN IT: `node src/rust_sim/harness/probe_attract_move.js`
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const d3 = Dex.mod('gen3');
const m = d3.moves.get('attract');
console.log('=== gen3-RESOLVED dex row ===');
console.log(
  `attract  target=${m.target} acc=${m.accuracy} cat=${m.category} vol=${m.volatileStatus} ` +
  `flags=${JSON.stringify(m.flags)} onTryHit=${!!m.onTryHit}`
);
const c = d3.conditions.get('attract');
console.log('attract condition keys:', Object.keys(c).join(','));
console.log('onStart:\n' + String(c.onStart).slice(0, 500));

let draws = [];
for (const fn of ['random', 'randomChance', 'sample']) {
  const o = PRNG.prototype[fn];
  PRNG.prototype[fn] = function (...a) {
    const r = o.apply(this, a);
    draws.push(`${fn}(${a})->${r}`);
    return r;
  };
}

async function run(label, p1, p2, script) {
  draws = [];
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const all = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) all.push(l); })();
  streams.omniscient.write('>start {"formatid":"gen3customgame","seed":[9,9,9,9]}');
  streams.omniscient.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  streams.omniscient.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise((r) => setTimeout(r, 180));
  let base = draws.length;
  for (const cmd of script) {
    base = draws.length;
    for (const part of cmd.split('\n')) streams.omniscient.write(part);
    await new Promise((r) => setTimeout(r, 180));
  }
  const i = all.indexOf('|turn|1');
  const shown = all.slice(i + 1).filter((l) =>
    /^\|(move|-start|-end|-fail|-immune|-activate|-miss|-cant|turn)\|/.test(l));
  console.log(`\n== ${label}\n  ${shown.join('\n  ')}`);
  console.log('  DRAWS (last decision):', draws.slice(base).join('  ') || '(none)');
}

// Gender is the WHOLE mechanic here, so every set pins it explicitly (the packed gender field).
const SET = (species, ability, moves, gender, item = 'Leftovers') =>
  `|${species}|${item}|${ability}|${moves}|Hardy|85,85,85,85,85,85|${gender}||||`;

(async () => {
  const caster = SET('Nidoking', 'Poison Point', 'attract,splash,splash,splash', 'M');
  const female = SET('Nidoqueen', 'Poison Point', 'splash,splash,splash,splash', 'F');
  const male = SET('Nidoking', 'Poison Point', 'splash,splash,splash,splash', 'M');
  const genderless = SET('Magneton', 'Sturdy', 'splash,splash,splash,splash', 'N');
  const oblivious = SET('Slowbro', 'Oblivious', 'splash,splash,splash,splash', 'F');
  const subber = SET('Nidoqueen', 'Poison Point', 'substitute,splash,splash,splash', 'F');
  const guarder = SET('Nidoqueen', 'Poison Point', 'safeguard,splash,splash,splash', 'F');
  const protector = SET('Nidoqueen', 'Poison Point', 'protect,splash,splash,splash', 'F');

  const T1 = ['>p1 move 1\n>p2 move 1'];
  // The blocker goes up on turn 1 (p1 splashes from slot 2), the cast lands on turn 2.
  const T2 = ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2'];

  await run('1. PLAIN HIT — M caster into an F target', caster, female, T1);
  // A second decision after the hit: the ATTRACTED mon tries to move, so the probe also captures
  // the volatile's own `-activate` + randomChance(1,2) immobilize on the MOVE path.
  await run('1b. the attracted target then tries to MOVE', caster, female,
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  await run('2. SAME GENDER — M into M', caster, male, T1);
  await run('3. GENDERLESS target', caster, genderless, T1);
  await run('4. ALREADY ATTRACTED — a second cast', caster, female, [...T1, ...T1]);
  await run('5. OBLIVIOUS target', caster, oblivious, T1);
  await run('6. SUBSTITUTE', caster, subber, T2);
  await run('7. SAFEGUARD — does the ward cover attraction?', caster, guarder, T2);
  await run('8. PROTECT (same turn — Protect lasts only its own turn)', caster, protector,
    ['>p1 move 1\n>p2 move 1']);
})();
