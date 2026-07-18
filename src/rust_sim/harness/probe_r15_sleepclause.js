'use strict';
const path=require('path');
const PS=path.resolve(__dirname,'../../../deps/pokemon-showdown');
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream'));
// p1: Breloom (Spore) + a filler. p2: Zapdos (Rest, slower so it self-sleeps) + Suicune.
const p1='Breloom||Leftovers|EffectSpore|Spore,BrickBreak,HiddenPowerRock,FocusPunch|Jolly|,252,,,4,252|M|,,30,,30,30|||,Rock,,,,]Snorlax||Leftovers|Immunity|BodySlam,Earthquake,FocusPunch,SelfDestruct|Adamant|80,176,124,,108,20|M||||';
const p2='Zapdos||Leftovers|Pressure|Rest,Roar,Thunderbolt,Toxic|Calm|200,,,,252,56|N|,0,,,,|||]Suicune||Leftovers|Pressure|Rest,Surf,CalmMind,Roar|Bold|252,,228,,,28|N|,0,,,,|||';
(async()=>{
  const stream=new BattleStream();
  let log=[];
  (async()=>{ for await (const c of stream) log.push(c); })();
  stream.write(`>start {"formatid":"gen3ou","seed":[9,9,9,9]}`);
  stream.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  stream.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setImmediate(r));
  // Turn 1: p1 Breloom does nothing useful (BrickBreak), p2 Zapdos REST (self-sleep).
  stream.write('>p1 move 2'); // HiddenPowerRock (harmless-ish)
  stream.write('>p2 move 1'); // Rest -> Zapdos self-sleep
  await new Promise(r=>setImmediate(r));
  // Turn 2: p1 Breloom Spore, p2 switch to Suicune so Spore targets Suicune.
  stream.write('>p2 switch 2'); // bring Suicune in
  stream.write('>p1 move 1'); // Spore
  await new Promise(r=>setImmediate(r));
  const s=log.join('\n');
  const spore=s.split('\n').filter(l=>/Spore|Sleep Clause|slp|Suicune/.test(l));
  console.log(spore.join('\n'));
})().catch(e=>{console.error(e);process.exit(1);});
