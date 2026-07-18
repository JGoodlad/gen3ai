'use strict';
const path=require('path');
const PS=path.resolve(__dirname,'../../../deps/pokemon-showdown');
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream'));
// p1: Breloom (Spore first slot). p2: Zapdos (Rest first) + Suicune.
const p1='Breloom||Leftovers|EffectSpore|Spore,BrickBreak,HiddenPowerRock,FocusPunch|Jolly|,252,,,4,252|M|,,30,,30,30|||,Rock,,,,]Snorlax||Leftovers|Immunity|BodySlam,Earthquake,FocusPunch,SelfDestruct|Adamant|80,176,124,,108,20|M||||';
const p2='Zapdos||Leftovers|Pressure|Rest,Roar,Thunderbolt,Toxic|Calm|200,,,,252,56|N|,0,,,,|||]Suicune||Leftovers|Pressure|Rest,Surf,CalmMind,Roar|Bold|252,,228,,,28|N|,0,,,,|||';
(async()=>{
  const stream=new BattleStream();
  let out=[];
  (async()=>{ for await (const c of stream) out.push(c); })();
  stream.write(`>start {"formatid":"gen3ou","seed":[9,9,9,9]}`);
  stream.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  stream.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setImmediate(r));
  const battle=stream.battle;
  console.log('init seed:', String(battle.prng.getSeed()));
  // dec 0: p1 HP Rock (move 3), p2 Zapdos Rest (move 1)
  stream.write('>p1 move 3'); stream.write('>p2 move 1');
  await new Promise(r=>setImmediate(r));
  console.log('dec0 seedAfter:', String(battle.prng.getSeed()), 'p2 zapdos status:', battle.sides[1].pokemon[0].status);
  // dec 1: p2 switch Suicune (switch 2), p1 Spore (move 1)
  stream.write('>p2 switch 2'); stream.write('>p1 move 1');
  await new Promise(r=>setImmediate(r));
  console.log('dec1 seedAfter:', String(battle.prng.getSeed()));
  const sui=battle.sides[1].active[0];
  console.log('p1 active:', battle.sides[0].active[0].species.id, 'p2 active:', sui.species.id, 'status:', sui.status, 'hp:', sui.hp+'/'+sui.maxhp);
  console.log('p2 zapdos(bench) status:', battle.sides[1].pokemon.find(p=>p.species.id==='zapdos').status);
})().catch(e=>{console.error(e);process.exit(1);});
