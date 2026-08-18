// PROBE: gen-3 RAGE + SECRET POWER — the last two construction fail-louds.
//   RAGE  : `self.volatileStatus: 'rage'`. The condition's onHit boosts the USER's Atk +1
//           whenever a FOE lands a non-Status move while the volatile is up; onBeforeMove
//           (priority 100) REMOVES it before the user's next move — so it lasts exactly
//           until the user acts again.
//   SECRET POWER: an `onModifyMove` that swaps the secondary by TERRAIN. gen-3 has NO
//           terrain (`field.isTerrain('')` is true), so the handler RETURNS EARLY and the
//           base 30% paralysis secondary stands — i.e. it is an ordinary 70-BP Normal move
//           with a 30% par secondary. Measured, because "the handler exists" reads as
//           MISMODELED until you check what it does with no terrain.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const sp = Dex.mod('gen3').moves.get('secretpower');
console.log('== SECRET POWER dex: bp', sp.basePower, 'type', sp.type, 'cat', sp.category,
  'secondary', JSON.stringify(sp.secondary || sp.secondaries || null));

let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };

async function run(label, p1, p2, script, seed) {
  draws = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 150));
  console.log(`\n== ${label} seed=${JSON.stringify(seed)}`);
  for (const c of script) {
    const mark = ch.length, dm = draws.length;
    s.write(c); await new Promise(r => setTimeout(r, 150));
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n');
    const keep = seg.split('\n').filter(l => /^\|(-boost|-unboost|-status|-singlemove|-start|-end|move|-damage)\|/.test(l));
    console.log(`   ${c.split('\n')[0].padEnd(14)} atk=${s.battle.sides[0].active[0].boosts.atk}  ${draws.slice(dm).join(' ')}`);
    for (const l of keep) console.log('        ' + l);
  }
}
(async () => {
  const RAGER = 'Nidoking||none|poisonpoint|rage,splash|Hardy|85,85,85,85,85,85|M||||';
  const HITTER = 'Snorlax||none|sturdy|tackle,splash|Hardy|85,85,85,85,85,85|M||||';
  // The rage volatile is REMOVED by the user's own next move (onBeforeMove priority 100), so
  // the boost is only observable when the foe hits BETWEEN the Rage and the user's next
  // action — i.e. on the SAME turn, with the user moving first. Nidoking outspeeds Snorlax.
  await run('RAGE: cast + foe hits the SAME turn (Atk should rise)', RAGER, HITTER,
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'], [3,3,3,3]);
  await run('SECRET POWER with NO terrain (base 30% par secondary)',
    'Nidoking||none|poisonpoint|secretpower,splash|Hardy|85,85,85,85,85,85|M||||',
    'Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||',
    Array(4).fill('>p1 move 1\n>p2 move 1'), [3,3,3,3]);
})();
