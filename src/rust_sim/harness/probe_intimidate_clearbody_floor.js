// Does gen-3 Clear Body still emit its -fail when the target's Atk is at the -6 FLOOR?
// The repro says NO: the sim emitted a delta-0 `-unboost|atk|0` instead. Reaching the floor
// needs a p1 that SURVIVES repeated Superpowers — the first attempt used a Zigzagoon, which
// was KO'd on turn 1 and desynced every later blind write (it only ever reached -1).
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
async function run(drops){
  const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  // Steelix: huge Def, resists Fighting -> survives many Superpowers.
  const p1='Steelix||leftovers|sturdy|splash|Impish|252,,252,,,|M||||]Salamence||none|intimidate|splash|Hardy|85,85,85,85,85,85|M||||';
  const p2='Regirock||leftovers|clearbody|superpower,splash|Impish|252,,252,,,|N||||';
  s.write(`>start {"formatid":"gen3customgame","seed":[3,3,3,3]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,150));
  for(let i=0;i<drops;i++){
    s.write('>p1 move 1\n>p2 move 1'); await new Promise(r=>setTimeout(r,120));
  }
  const atk=s.battle.sides[1].active[0].boosts.atk;
  const mark=ch.length;
  s.write('>p1 switch 2\n>p2 move 2');
  await new Promise(r=>setTimeout(r,200));
  const seg=ch.slice(mark).filter(x=>!x.startsWith('sideupdate')).join('\n')
    .split('\n').filter(l=>/^\|(-ability|-unboost|-fail)\|/.test(l));
  console.log(`\n== after ${drops} self-drops: Regirock atk stage = ${atk}`);
  for(const l of seg) console.log('   ', l);
  return atk;
}
(async()=>{ for (const n of [0,3,6,9]) await run(n); })();
