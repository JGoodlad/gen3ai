// PROBE: the gen-3 FOCUS ENERGY **MOVE** — draw model, emission form, and the re-cast branch.
//
// THE SITUATION, and it is the same shape as ROUND 57's confusion family and ROUND 59's Attract
// investigation: the VOLATILE is already modelled bit-for-bit — `state.rs::MonState::focus_energy`
// (+2 crit stages through `helpers.rs`'s crit-ratio read, cleared on switch-out) — but in gen 3 the
// port can only REACH it through a **Lansat Berry** eat. The Focus Energy MOVE is unmodelled, and
// it is **40 legal learners**.
//
// So the missing half is the MOVE ENTRY, and the things a reading of the volatile cannot tell you:
//   1. the DRAW MODEL — is it never-miss (no roll at all) or an accuracy-100 roll (one draw)?
//   2. the EMISSION FORM — `|-start|<user>|move: Focus Energy`? bare `Focus Energy`? something else?
//   3. the RE-CAST branch — does a second cast FAIL (and with which form), or silently re-apply?
//   4. does it interact with SNATCH (a self-targeting status move is snatchable in gen 3)?
//   5. the CRIT EFFECT is the point of the move — measure that it actually shifts the crit rate,
//      so the pin can assert a behavioural consequence and not just a flag.
//
// RUN IT: `node src/rust_sim/harness/probe_focus_energy.js`
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const d3 = Dex.mod('gen3');
const m = d3.moves.get('focusenergy');
console.log('=== gen3-RESOLVED dex row ===');
console.log(
  `focusenergy  target=${m.target} acc=${m.accuracy} cat=${m.category} vol=${m.volatileStatus} ` +
  `flags=${JSON.stringify(m.flags)} onTryHit=${!!m.onTryHit} onHit=${!!m.onHit}`
);
const c = d3.conditions.get('focusenergy');
console.log('condition keys:', Object.keys(c).join(','));
console.log('onStart:', String(c.onStart).slice(0, 260));
console.log('onModifyCritRatio:', String(c.onModifyCritRatio).slice(0, 200));

let draws = [];
for (const fn of ['random', 'randomChance', 'sample']) {
  const o = PRNG.prototype[fn];
  PRNG.prototype[fn] = function (...a) {
    const r = o.apply(this, a);
    draws.push(`${fn}(${a})->${r}`);
    return r;
  };
}

async function run(label, p1, p2, script, seed = [9, 9, 9, 9]) {
  draws = [];
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const all = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) all.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[${seed.join(',')}]}`);
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
    /^\|(move|-start|-end|-fail|-immune|-activate|-crit|-damage|turn)\|/.test(l));
  console.log(`\n== ${label}\n  ${shown.join('\n  ')}`);
  console.log('  DRAWS (last decision):', draws.slice(base).join('  ') || '(none)');
  return all;
}

const SET = (species, ability, moves, item = 'Leftovers') =>
  `|${species}|${item}|${ability}|${moves}|Hardy|85,85,85,85,85,85|M||||`;

(async () => {
  const user = SET('Machamp', 'Guts', 'focusenergy,tackle,splash,splash');
  const inert = SET('Snorlax', 'Immunity', 'splash,splash,splash,splash');
  const snatcher = SET('Snorlax', 'Immunity', 'snatch,splash,splash,splash');
  const T1 = ['>p1 move 1\n>p2 move 1'];

  await run('1. PLAIN CAST', user, inert, T1);
  await run('2. SECOND CAST (the volatile is already up)', user, inert, [...T1, ...T1]);
  await run('3. SNATCH — is a self-targeting Focus Energy stolen?', user, snatcher,
    ['>p1 move 1\n>p2 move 1']);

  // 5. THE CRIT EFFECT, measured as a RATE rather than asserted from the flag. 200 Tackles with
  //    and without the volatile; gen-3 crit stage 0 is 1/16 and +2 is 1/4, so the two rates must
  //    be far apart. A pin that only checked the flag would pass on an engine that set it and
  //    never read it.
  let critWith = 0, critWithout = 0;
  for (let k = 0; k < 200; k++) {
    const seed = [100 + k * 3, 200 + k * 7, 300 + k * 11, 400 + k * 13];
    const a = await quiet(user, inert, ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'], seed);
    const b = await quiet(user, inert, ['>p1 move 3\n>p2 move 1', '>p1 move 2\n>p2 move 1'], seed);
    if (a.some((l) => l.startsWith('|-crit|'))) critWith++;
    if (b.some((l) => l.startsWith('|-crit|'))) critWithout++;
  }
  console.log(`\n== 5. CRIT RATE over 200 seeds (Tackle on turn 2)`);
  console.log(`   with Focus Energy   : ${critWith}/200 = ${(critWith / 2).toFixed(1)}%  (gen3 +2 stages -> 1/4 = 25%)`);
  console.log(`   without (Splash ctl): ${critWithout}/200 = ${(critWithout / 2).toFixed(1)}%  (stage 0 -> 1/16 = 6.25%)`);
})();

async function quiet(p1, p2, script, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const all = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) all.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[${seed.join(',')}]}`);
  streams.omniscient.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  streams.omniscient.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise((r) => setTimeout(r, 40));
  for (const cmd of script) {
    for (const part of cmd.split('\n')) streams.omniscient.write(part);
    await new Promise((r) => setTimeout(r, 40));
  }
  return all;
}
