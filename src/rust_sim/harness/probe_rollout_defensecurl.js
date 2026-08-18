// PROBE: gen-3 ROLLOUT / ICE BALL + DEFENSE CURL. The BP ladder is 5 EXECUTIONS (not turns),
// doubling per hit, and DEFENSE CURL doubles it again — which is why the two must be modeled
// in the same pass. Damage is read as a per-turn HP delta on a Sturdy Snorlax so nothing
// faints mid-ladder; CRIT and MISS are tagged because both mimic a rung.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };

async function run(label, moves, script, seed) {
  draws = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  const p1 = `Marowak||none|rockhead|${moves}|Hardy|85,85,85,85,85,85|M||||`;
  const p2 = `Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||`;
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 150));
  console.log(`\n== ${label} seed=${JSON.stringify(seed)}`);
  for (const c of script) {
    const before = s.battle.sides[1].active[0].hp;
    const mark = ch.length, dm = draws.length;
    s.write(c); await new Promise(r => setTimeout(r, 150));
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n');
    const after = s.battle.sides[1].active[0].hp;
    const mv = (seg.match(/\|move\|p1a: [^|]+\|([^|]+)/) || [])[1] || '-';
    console.log(`   ${c.split('\n')[0].padEnd(14)} mv=${String(mv).padEnd(12)} dmg=${String(before - after).padStart(4)}` +
      `${/\|-crit\|/.test(seg) ? ' CRIT' : ''}${/\|-miss\|/.test(seg) ? ' MISS' : ''}` +
      `${/lockedmove/.test(seg) ? ' [locked]' : ''}  ${draws.slice(dm).join(' ')}`);
  }
}
(async () => {
  await run('ROLLOUT x6 (ladder 30/60/120/240/480, then restart)',
    'rollout,splash,defensecurl', Array(6).fill('>p1 move 1\n>p2 move 1'), [3,3,3,3]);
  await run('ROLLOUT after DEFENSE CURL (each rung x2)',
    'rollout,splash,defensecurl',
    ['>p1 move 3\n>p2 move 1', ...Array(4).fill('>p1 move 1\n>p2 move 1')], [3,3,3,3]);
  await run('ICE BALL x5',
    'iceball,splash,defensecurl', Array(5).fill('>p1 move 1\n>p2 move 1'), [3,3,3,3]);
})();
