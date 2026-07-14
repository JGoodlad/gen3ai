'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT='gen3customgame';
const IV31={hp:31,atk:31,def:31,spa:31,spd:31,spe:31}, EV0={hp:0,atk:0,def:0,spa:0,spd:0,spe:0};
function mon(s,m,o={}){return{species:s,item:o.item||'',ability:o.ability||'No Ability',moves:m,
  evs:{...EV0,...(o.evs||{})},ivs:o.ivs||IV31,nature:o.nature||'Serious',level:o.level||100,gender:o.gender||'N'};}
async function run(){
  const Battle=require(path.join(PS,'dist/sim/battle')).Battle;
  const origRunEvent=Battle.prototype.runEvent;
  Battle.prototype.runEvent=function(eventid, target, ...rest){
    if(eventid==='BeforeSwitchOut') console.log(`  runEvent BeforeSwitchOut on ${target&&target.name} skipFlag=${target&&target.skipBeforeSwitchOutEventFlag} switchFlag=${target&&target.switchFlag}`);
    return origRunEvent.call(this, eventid, target, ...rest);
  };
  const BQ=require(path.join(PS,'dist/sim/battle-queue')).BattleQueue;
  const origCancel=BQ.prototype.cancelMove;
  BQ.prototype.cancelMove=function(pokemon){ const r=origCancel.call(this,pokemon); console.log(`    cancelMove(${pokemon&&pokemon.name}) => ${r}`); return r; };
  const p1=[mon('Snorlax',['pursuit','bodyslam'],{evs:{spa:252}})];
  const p2=[mon('Jolteon',['batonpass','thunderbolt'],{evs:{spe:252,spa:252}}), mon('Vaporeon',['surf'])];
  const stream=new BattleStream();
  (async()=>{for await(const c of stream){}})();
  stream.write(`>start ${JSON.stringify({formatid:FORMAT,seed:[7,11,13,17]})}`);
  stream.write(`>player p1 ${JSON.stringify({name:'A',team:Teams.pack(p1)})}`);
  stream.write(`>player p2 ${JSON.stringify({name:'B',team:Teams.pack(p2)})}`);
  await new Promise(r=>setTimeout(r,0));
  console.log('--- dec0 ---');
  stream.write('>p1 move 1'); stream.write('>p2 move 1');
  await new Promise(r=>setTimeout(r,0));
  console.log('--- dec1 ---');
  stream.write('>p2 switch 2');
  await new Promise(r=>setTimeout(r,0));
}
run();
