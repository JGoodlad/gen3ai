// probe_e2e86_diverge.js — dump the OMNISCIENT sim's protocol log + per-decision seed
// for e2e_86 (the one battle that diverges after admitting the DMG_MOD abilities), to
// root-cause the dec22->dec23 draw divergence (freeze/switch-slot cascade behind the
// admitted Swampert[Torrent] battle). Replays the golden's EXACT recorded choices.
//
// Run:  node src/rust_sim/harness/probe_e2e86_diverge.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

const FORMAT = 'gen3customgame';
const INIT_SEED = [65434, 3821, 57496, 45668];
const P1 = 'Swampert||Leftovers|Torrent|Curse,Earthquake,HydroPump,IceBeam|Quiet|240,,,252,,16|||||]Metagross||Leftovers|ClearBody|Earthquake,ThunderPunch,HiddenPowerGrass,Explosion|Rash|168,,16,252,,72|N|,30,,30,,|||,Grass,,,,]Jirachi||Leftovers|SereneGrace|Thunderbolt,FirePunch,HiddenPowerGrass,DynamicPunch|Rash|112,,,220,,176|N|,30,,30,,|||,Grass,,,,]Gengar||Leftovers|Levitate|wisp,IcePunch,FirePunch,Explosion|Timid|4,,,252,,252|||||]Jolteon||Leftovers|VoltAbsorb|Substitute,Thunderbolt,HiddenPowerGrass,BatonPass|Timid|8,,,248,,252||,2,,30,,|||,Grass,,,,]Charizard||Leftovers|Blaze|DragonClaw,Flamethrower,HiddenPowerGrass,FocusPunch|Mild|,4,,252,,252||,30,,30,,|||,Grass,,,,';
const P2 = 'Zapdos||Leftovers|Pressure|ThunderWave,Toxic,Thunderbolt,HiddenPowerIce|Modest|216,,,136,,156|N|,,,,,30|||,Ice,,,,]Swampert||Leftovers|Torrent|Curse,Earthquake,HydroPump,IceBeam|Relaxed|240,,136,72,44,16|||||]Jirachi||Leftovers|SereneGrace|Thunderbolt,FirePunch,HiddenPowerGrass,DynamicPunch|Rash|112,,,220,,176|N|,30,,30,,|||,Grass,,,,]Tyranitar||Leftovers|SandStream|DragonDance,Earthquake,RockSlide,DoubleEdge|Adamant|184,104,,,,220|||||]Heracross||Leftovers|Guts|SwordsDance,Megahorn,BrickBreak,RockSlide|Jolly|4,252,,,,252|||||]Salamence||Leftovers|Intimidate|DragonClaw,Flamethrower,HiddenPowerGrass,BrickBreak|Rash|4,,,252,,252||,30,,30,,|||,Grass,,,,';

// The golden's recorded per-decision choices (col7 p1, col8 p2). '-' = that side has no
// choice this boundary (a forced-switch on the OTHER side, or a fainted side).
const CHOICES = [
  ['m2','m0'],['m3','m1'],['-','s5'],['m3','m1'],['-','s2'],['s4','m1'],['s3','s1'],['s3','m1'],
  ['s2','-'],['m0','s1'],['s5','m0'],['m0','m3'],['m1','m3'],['m0','m0'],['s1','-'],['m0','s1'],
  ['m0','m3'],['s5','m3'],['s4','m1'],['m3','m3'],['s5','-'],['m1','s1'],['s3','m3'],['m0','m3'],
  ['m1','m1'],['m2','s3'],['s3','s3'],['m1','m1'],['m3','s4'],['s4','s1'],['m3','s3'],['m3','m0'],
  ['m1','m0'],['s3','m0'],['m1','m0'],['m0','m1'],['m1','m2'],['m3','m2'],['s3','-'],['m0','m2'],['m0','m1'],
];
function tok(t) { // 'm0' -> 'move 1', 's3' -> 'switch 4'
  if (t === '-' || t === '') return null;
  const n = parseInt(t.slice(1), 10) + 1;
  return t[0] === 'm' ? `move ${n}` : `switch ${n}`;
}

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(INIT_SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: P1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: P2 })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let draws = 0;
  rng.next = (...a) => { draws++; return realNext(...a); };

  const LO = Number(process.env.LO || 20), HI = Number(process.env.HI || 24);
  for (let di = 0; di < CHOICES.length; di++) {
    const [c1, c2] = CHOICES[di];
    const seedBefore = battle.prng.getSeed();
    const logStart = log.length;
    draws = 0;
    const t1 = tok(c1), t2 = tok(c2);
    if (t1) { try { streams.omniscient.write(`>p1 ${t1}`); } catch (e) {} }
    if (t2) { try { streams.omniscient.write(`>p2 ${t2}`); } catch (e) {} }
    for (let i = 0; i < 8; i++) await tick();
    const seedAfter = battle.prng.getSeed();
    if (di >= LO && di <= HI) {
      console.log(`\n===== dec${di}  p1=${c1}(${t1}) p2=${c2}(${t2})  draws=${draws} =====`);
      console.log(`  seedBefore=${seedBefore}`);
      console.log(`  seedAfter =${seedAfter}`);
      const p1a = battle.sides[0].active[0], p2a = battle.sides[1].active[0];
      console.log(`  p1 active: ${p1a.species.name} hp=${p1a.hp}/${p1a.maxhp} status=${p1a.status || '-'}`);
      console.log(`  p2 active: ${p2a.species.name} hp=${p2a.hp}/${p2a.maxhp} status=${p2a.status || '-'}`);
      console.log('  --- protocol lines this decision ---');
      for (let k = logStart; k < log.length; k++) {
        const l = log[k];
        if (l.startsWith('|t:|') || l.startsWith('|request|') || l.startsWith('|split|') || l.startsWith('|debug|')) continue;
        console.log(`    ${l}`);
      }
    }
    if (battle.ended) { console.log(`\n(battle ended at dec${di})`); break; }
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e); process.exit(1); });
