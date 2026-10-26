// PROBE: gen-3 MUSTRECHARGE (Hyper Beam) — a recharge turn the user never acts in (`gen3_recharge_expiry_v1`, found by the
// GIGO audit B-C1 re-run: a targeted game's Hyper Beam user recharged one turn too long).
// `mustrecharge` is `duration: 2` (data/conditions.ts). The recharge turn normally removes it in its own `onBeforeMove`, BEFORE
// that turn's residual, so only the cast turn's residual ticks it (2 -> 1). But a gen-3 faint SKIPS the rest of the turn's moves
// and switches (`Battle.faintMessages`: "in gen 3, fainting skips all moves and switches" -> `queue.cancelAction`), so when the
// foe faints BEFORE the recharging mon's action, the SECOND residual ticks the volatile 1 -> 0 and it EXPIRES: turn 3 is free.
//   S1  turn 2: the faster foe Explodes first (its faint cancels the Snorlax's recharge action); turn 3 Hyper Beam again.
//   S2  control: the foe does not faint, the Snorlax recharges on turn 2 and is free on turn 3.
//   node harness/probe_recharge_expiry.js [firstSeed lastSeed]
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const USER = 'Snorlax||none|sturdy|hyperbeam,splash|Hardy|85,85,85,85,85,85|M||||';
const FOE = 'Blissey||none|naturalcure|splash,explosion|Hardy|85,85,85,85,85,85|F||||]Chansey||none|naturalcure|splash|Hardy|85,85,85,85,85,85|F||||';

async function run(label, seed, steps) {
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${USER}"}\n>player p2 {"name":"P2","team":"${FOE}"}`);
  await sleep(120);
  const out = [];
  for (const st of steps) {
    const mark = ch.length;
    s.write(st);
    await sleep(120);
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n').split('\n')
      .filter(l => /^\|(move|cant|-mustrecharge|faint|turn|switch)\|/.test(l)).map(l => l.replace(/\|(p\da): /g, '|$1:')).join(' ');
    out.push(seg);
  }
  console.log(`${label} seed=${JSON.stringify(seed)} seedOut=${s.battle.prng.getSeed()}\n   ` + out.join('\n   '));
}
(async () => {
  const lo = Number(process.argv[2] || 1), hi = Number(process.argv[3] || 1);
  for (let k = lo; k <= hi; k++) {
    const seed = [k, k, k, k];
    await run('S1 foe explodes ', seed, ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2', '>p2 switch 2', '>p1 move 1\n>p2 move 1']);
    await run('S2 control      ', seed, ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1']);
  }
  process.exit(0);
})();
