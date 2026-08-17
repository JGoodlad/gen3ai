// PROBE: does SUBSTITUTE release a gen-3 partial trap (Wrap) on the user?
// Repro evidence (ab_14_8 turn 44) shows `-end ... [partiallytrapped]|[silent]` emitted
// between `-start Substitute` and the substitute's HP cost. Measure it, don't infer it.
const path=require('path');
const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
// p1 = the trapped target (has Substitute). p2 = the trapper (Wrap).
const p1='Electrode||Leftovers|Static|Substitute,Thunderbolt,Explosion,Protect|Hardy|85,85,85,85,85,85|M||||';
const p2='Shuckle||Leftovers|Sturdy|Wrap,Toxic,Rest,Protect|Hardy|85,85,85,85,85,85|M||||';
(async()=>{
  const s=new BattleStream(); const out=[];
  (async()=>{ for await (const c of s) out.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[7,7,7,7]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,300));
  // T1: p2 Wraps p1 (p1 protects nothing — use Thunderbolt so the trap lands)
  s.write('>p1 move 2\n>p2 move 1');
  await new Promise(r=>setTimeout(r,300));
  // T2: p1 uses SUBSTITUTE while trapped
  s.write('>p1 move 1\n>p2 move 2');
  await new Promise(r=>setTimeout(r,300));
  // T3: another turn — is the trap still ticking?
  s.write('>p1 move 2\n>p2 move 2');
  await new Promise(r=>setTimeout(r,400));
  const lines=out.join('\n').split('\n').filter(l=>
    /^\|(move|-start|-end|-damage|-heal|-activate|turn)\|/.test(l));
  console.log(lines.join('\n'));
  const ends=lines.filter(l=>l.includes('partiallytrapped')&&l.startsWith('|-end'));
  console.log('\n--- VERDICT ---');
  console.log('partial-trap -end lines:', ends.length?ends:'(none)');
  console.log('released on the Substitute turn?', ends.some(l=>l.includes('[silent]'))?'YES (silent = onResidual-style release)':'NO');
  const chipsAfter = lines.slice(lines.findIndex(l=>l.includes('-start')&&l.includes('Substitute')))
      .filter(l=>l.includes('partiallytrapped')&&l.startsWith('|-damage'));
  console.log('wrap CHIPS after Substitute started:', chipsAfter.length);
})();
