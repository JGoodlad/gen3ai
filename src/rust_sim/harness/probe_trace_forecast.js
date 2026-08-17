// PROBE (2026-08-17): what does the REAL sim do when TRACE copies Forecast onto a
// non-Castform, and then the weather changes? The port no-ops (forecast.rs gates on
// base_species_id == "castform"); this measures whether the sim agrees.
const path=require('path');
const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const p1='Porygon2||Leftovers|Trace|Recover,Thunderbolt,Icebeam,Toxic|Hardy|85,85,85,85,85,85|M||||';
const p2='Castform||Leftovers|Forecast|SunnyDay,Weatherball,Thunderbolt,Icebeam|Hardy|85,85,85,85,85,85|M||||';
(async()=>{
  const s=new BattleStream();
  const out=[];
  (async()=>{ for await (const c of s) out.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,300));
  s.write('>p1 move 1\n>p2 move 1');           // Castform uses Sunny Day
  await new Promise(r=>setTimeout(r,300));
  s.write('>p1 move 1\n>p2 move 1');
  await new Promise(r=>setTimeout(r,400));
  const lines=out.join('\n').split('\n').filter(l=>
    /-ability|formechange|-weather|^\|switch|^\|turn/.test(l));
  console.log(lines.join('\n'));
  console.log('\n--- VERDICT ---');
  const fc=lines.filter(l=>l.includes('formechange'));
  console.log('formechange lines:', fc.length ? fc : '(none)');
  console.log('any formechange on the TRACED Porygon2 (p1)?',
    fc.some(l=>l.includes('p1a'))?'YES':'NO');
})();
