// probe_charge_selfko.js — REAL-Showdown ground truth for the CHARGE self-KO emission
// (`gen3_charge_selfko_no_end_v1`, random-mode byte-fuzz find ab_12_17 @ master-seed 200724).
// A Charge-holding mon that Self-Destructs (self-KOs) emits NO |-end|Charge line: the sim's
// Pokemon.removeVolatile returns false for a 0-HP mon, so charge.onAfterMove's removeVolatile
// is a no-op (the faint's later silent clearVolatile drops it). Run from src/rust_sim.
'use strict';
const path='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream,getPlayerStreams}=require(path+'/dist/sim/battle-stream.js');
const {Teams}=require(path+'/dist/sim/teams.js');
const mon=(sp,it,mv)=>({species:sp,item:it,ability:'No Ability',moves:mv,evs:{hp:252,atk:252},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},nature:'Serious',level:100,gender:'N'});
function tick(){return new Promise(r=>setTimeout(r,0));}
(async()=>{
  const stream=new BattleStream(); const s=getPlayerStreams(stream); const lines=[];
  (async()=>{for await(const ch of s.omniscient) for(const l of ch.split('\n')) if(l) lines.push(l);})();
  s.omniscient.write(`>start {"formatid":"gen3customgame","seed":[7,7,7,7]}`);
  s.omniscient.write(`>player p1 ${JSON.stringify({name:'P1',team:Teams.pack([mon('Electrode','',['charge','selfdestruct'])])})}`);
  s.omniscient.write(`>player p2 ${JSON.stringify({name:'P2',team:Teams.pack([mon('Snorlax','',['splash']), mon('Blissey','',['splash'])])})}`);
  for(let i=0;i<12;i++)await tick();
  s.omniscient.write('>p1 move 1'); s.omniscient.write('>p2 move 1'); // Charge / Splash
  for(let i=0;i<14;i++)await tick();
  s.omniscient.write('>p1 move 2'); s.omniscient.write('>p2 move 1'); // Self-Destruct / Splash
  for(let i=0;i<16;i++)await tick();
  console.log(lines.filter(l=>/Charge|Self-Destruct|faint|selfdestruct|-end|-activate|-damage/i.test(l)).join('\n'));
})().catch(e=>{console.error(e);process.exit(1);});
