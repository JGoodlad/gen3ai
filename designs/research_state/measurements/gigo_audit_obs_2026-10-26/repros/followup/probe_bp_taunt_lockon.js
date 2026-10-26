const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
async function run(label, p1, p2, steps) {
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await sleep(150);
  console.log('==', label);
  for (const st of steps) {
    const m = ch.length; s.write(st); await sleep(150);
    const seg = ch.slice(m).filter(x => !x.startsWith('sideupdate')).join('\n').split('\n').filter(l => /^\|(move|cant|-start|-end|switch|-activate|turn|-fail|-immune)\|/.test(l));
    console.log('  ', st.replace('\n', ' '), '->', seg.join(' ; '));
    const b = s.battle; const a = b.sides[0].active[0];
    console.log('     p1 active', a.name, 'volatiles', Object.keys(a.volatiles).join(','));
  }
  process.exit(0);
}
(async () => {
  const bp = 'Smeargle||none|owntempo|batonpass,lockon,splash,swordsdance|Hardy|85,85,85,85,85,85|M||||]Snorlax||none|immunity|splash|Hardy|85,85,85,85,85,85|M||||';
  const foe = 'Gengar||none|levitate|taunt,splash|Hardy|85,85,85,85,85,85|M||||';
  // a taunted mon cannot use Baton Pass; Lock-On then Baton Pass carries lockon
  await run('taunt blocks BP / lockon passes', bp, foe, ['>p1 move 3\n>p2 move 1', '>p1 move 1\n>p2 move 2', '>p1 move 2\n>p2 move 2', '>p1 move 1\n>p2 move 2', '>p1 switch 2']);
})();
