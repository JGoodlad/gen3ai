// PROBE: gen-3 OUTRAGE / PETAL DANCE / THRASH — the shared `lockedmove` condition.
// Measures the OBSERVABLE: how many turns the user is locked, when the end-of-lock
// confusion lands, and where the duration draw sits. The condition carries TWO interacting
// counters (`duration: 2` refreshed by onRestart, and a `trueDuration = random(2,4)` ticked
// at the residual), so the turn count is MEASURED, not derived from the source.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };

async function run(label, moves, seed, n) {
  draws = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  const p1 = `Dragonite||none|innerfocus|${moves}|Hardy|85,85,85,85,85,85|M||||`;
  const p2 = `Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||`;
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 150));
  const rows = [];
  for (let i = 0; i < n; i++) {
    const mark = ch.length, dm = draws.length;
    s.write('>p1 move 1\n>p2 move 1');
    await new Promise(r => setTimeout(r, 150));
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n');
    const mv = (seg.match(/\|move\|p1a: [^|]+\|([^|]+)/) || [])[1] || '-';
    rows.push({ i: i + 1, mv,
      conf: /\|-start\|p1a: [^|]*\|confusion/.test(seg),
      lockedAttr: /\[from\] ?lockedmove/.test(seg),
      cant: /\|cant\|p1a/.test(seg),
      draws: draws.slice(dm).join(' ') });
  }
  console.log(`\n== ${label} seed=${JSON.stringify(seed)}`);
  for (const r of rows) console.log(`   t${r.i} move=${String(r.mv).padEnd(12)}${r.lockedAttr ? ' [lockedmove]' : ''}${r.conf ? '  CONFUSED' : ''}${r.cant ? ' CANT' : ''}  ${r.draws}`);
}
(async () => {
  for (const seed of [[3,3,3,3],[11,11,11,11],[7,7,7,7]])
    await run('OUTRAGE x6', 'outrage,splash', seed, 6);
  await run('THRASH x6', 'thrash,splash', [3,3,3,3], 6);
  await run('PETAL DANCE x6', 'petaldance,splash', [3,3,3,3], 6);
})();
